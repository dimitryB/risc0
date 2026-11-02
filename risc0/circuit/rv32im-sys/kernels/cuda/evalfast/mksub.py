#!/usr/bin/env python3
# Copyright 2026 RISC Zero, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Split chain functions in two at their narrowest point.

usage: mksub.py in.cu out.cu --fns f1,f2,... [--lo 0.45] [--hi 0.55]

Each listed function f is cut at the body line between lo and hi of its length
with the smallest live set. The tail becomes a __noinline__ template function
f_t whose parameters are the live values (their types are deduced from the call,
so no type inference is needed) plus cycle, steps and poly_mix; f ends with
`return f_t(...)`. constexpr literals and the rk_s/rk_m/rk_c header are
re-materialised in the tail, and pure input loads (data, accum and mix are
read-only for the whole kernel) are re-issued there instead of being passed.

The tail runs the same statements in the same order on the same values, so the
result is bit-identical; the shorter functions are what ptxas register-allocates
better (fewer spills in the long main-chain bodies).
"""

import re
import sys

from efparse import functions

ID = re.compile(r'\b(_wx\d+|_cx\d+|x\d+|r\d+_x\d+|_h_\w+|arg\d+(?:_e\d)?|rkB[DAM]|args)\b')
DEF = re.compile(r'^(?:const\s+)?(?:auto|FpExt\*|Fp\*|FpExt|Fp|RkAcc|RkW|RkW1|constexpr Fp|constexpr FpExt)'
                 r'\s+(\w+)\s*(=|\()(.*)$')
LOAD = re.compile(r'^auto (\w+) = EF_LD\((\w+), \(uint32_t\)\(\d+u \* rk_s \+ \(\(rk_c - \d+u\) & rk_m\)\)\);$')
HEAD = ('  size_t mask = steps - 1; const uint32_t rk_s = (uint32_t)steps, rk_m = rk_s - 1u, '
        'rk_c = (uint32_t)cycle;')


def opt(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


def split(fn, f, lo, hi):
    body = f['body']
    params = [p.strip().split()[-1] for p in f['sig'].split(',')]
    info = []  # per body line: (defs, uses, text)
    hdr_end = 0
    for i, (_, l) in enumerate(body):
        t = l.strip()
        if 'rk_c = (uint32_t)cycle;' in t:
            hdr_end = i + 1
        d, u = set(), set()
        for part in [p.strip() for p in t.split(';') if p.strip()]:
            m = DEF.match(part)
            if m:
                d.add(m.group(1))
                u |= set(ID.findall(m.group(3)))
            else:
                u |= set(ID.findall(part))
        info.append((d, u - d, t))
    constexpr = {}
    remat = {}
    for i, (_, _, t) in enumerate(info):
        m = re.match(r'^constexpr (?:Fp|FpExt) (\w+)\(', t)
        if m:
            constexpr[m.group(1)] = (body[i][1], i)
        m = LOAD.match(t)
        if m:
            remat[m.group(1)] = (body[i][1], m.group(2))
    n = len(body)
    best = None
    for cut in range(max(hdr_end + 1, int(n * lo)), int(n * hi)):
        if body[cut][1].strip().startswith('//'):
            continue
        defined = set(params)
        for i in range(cut):
            defined |= info[i][0]
        used = set()
        for i in range(cut, n):
            used |= info[i][1]
        lv = (used & defined) - set(constexpr)
        rm = {v for v in lv if v in remat}
        live = sorted((lv - rm) | {remat[v][1] for v in rm}, key=lambda v: (len(v), v))
        if best is None or len(live) < best[0]:
            best = (len(live), cut, live, rm)
    assert best is not None, fn
    _, cut, live, rm = best
    live = [v for v in live if v not in ('cycle', 'steps', 'poly_mix')]
    tname = fn + '_t'
    tps = ', '.join('typename T%d' % i for i in range(len(live)))
    sig = 'size_t cycle, size_t steps, FpExt* poly_mix' + ''.join(
        ', T%d %s' % (i, v) for i, v in enumerate(live))
    used_c = set()
    for i in range(cut, n):
        used_c |= {c for c in info[i][1] & set(constexpr) if constexpr[c][1] < cut}
    head = [HEAD, '  (void)mask; (void)rk_s; (void)rk_m; (void)rk_c;']
    head += [constexpr[c][0] for c in sorted(used_c)] + [remat[v][0] for v in sorted(rm)]
    tail = [body[i][1] for i in range(cut, n)]
    func = 'template <%s>\n__device__ EF_NOINLINE FpExt %s(%s) {\n%s\n}\n' % (
        tps, tname, sig, '\n'.join(head + tail))
    call = '  return %s(cycle, steps, poly_mix%s);' % (tname, ''.join(', ' + v for v in live))
    newbody = [body[i][1] for i in range(cut)] + [call]
    print('%s: %d lines, cut at %d (%.2f), live %d (+%d reloaded)'
          % (fn, n, cut, cut / n, len(live), len(rm)))
    return func, newbody


def main():
    argv = sys.argv
    if len(argv) < 3 or '--fns' not in argv:
        sys.exit(__doc__)
    fns = opt(argv, '--fns', '').split(',')
    lo, hi = float(opt(argv, '--lo', 0.45)), float(opt(argv, '--hi', 0.55))
    with open(argv[1]) as fh:
        s = fh.read()
    F = functions(s)
    lines = s.split('\n')
    funcs, repl = [], {}
    for fn in fns:
        func, newbody = split(fn, F[fn], lo, hi)
        funcs.append(func)
        repl[(F[fn]['start'], F[fn]['end'])] = newbody
    # Body lines are start+1..end-1 (1-based); rewrite from the end so the indices stay valid.
    for (st, en), nb in sorted(repl.items(), reverse=True):
        lines[st:en - 1] = nb
    s = '\n'.join(lines)
    # The tail templates must be declared before their callers: insert them before the first chain
    # function definition.
    k = s.index('__device__ EF_NOINLINE FpExt rv32im_v2_19(')
    s = s[:k] + ''.join(funcs) + s[k:]
    with open(argv[2], 'w') as fh:
        fh.write(s)


if __name__ == '__main__':
    main()
