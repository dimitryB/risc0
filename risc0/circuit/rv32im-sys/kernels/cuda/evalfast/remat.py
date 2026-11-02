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

"""remat.py: rematerialize cheap slab values from the inputs, then re-colour the slab.

A per-row value stored to the slab (RK_ST1) whose definition is a small Fp expression tree over
input loads (EF_LD of data/accum/mix/out) and literal constants is recomputed at its reads
(RK_LD1) in later functions instead of being kept in local memory across the call chain. The
inputs are read-only during the kernel, so the recomputed value is identical (same Fp ops on
the same canonical inputs). A store whose reads are all rematerialized is dropped. The slab is
then re-coloured by liveness over the program-ordered trace (poly_fp -> rv32im_v2_19 -> ... -> 0),
which shrinks RK_NWORDS (the per-thread local footprint).
A line left empty by dropped stores is removed.
usage: remat.py in.cu out.cu [--maxops N] [--maxloads N] [--minspan N]
"""
import collections, heapq, re, sys
args = sys.argv
def opt(name, d):
    return int(args[args.index(name) + 1]) if name in args else d
MAXOPS, MAXLOADS, MINSPAN = opt('--maxops', 2), opt('--maxloads', 2), opt('--minspan', 1)
src = open(args[1]).read(); L = src.split('\n')
FDEF = re.compile(r'^__device__ (?:EF_NOINLINE )?FpExt (rv32im_v2_\d+|poly_fp)\((.*)\) \{$')
F = {}
i = 0
while i < len(L):
    m = FDEF.match(L[i])
    if m:
        j = i + 1
        while L[j] != '}': j += 1
        F[m.group(1)] = (m.group(2), i, j); i = j
    i += 1
order = ['poly_fp'] + ['rv32im_v2_%d' % k for k in range(19, -1, -1)]
fidx = {f: n for n, f in enumerate(order)}
def split_top(s):
    out, d, cur = [], 0, ''
    for ch in s:
        if ch == '(' : d += 1
        if ch == ')' : d -= 1
        if ch == ',' and d == 0: out.append(cur.strip()); cur = ''
        else: cur += ch
    out.append(cur.strip()); return out
# buffer identity of pointer params: poly_fp uses args[k]; follow the tail calls
buf = {'poly_fp': {'args[%d]' % k: 'B%d' % k for k in range(4)}}
for fn in order[:-1]:
    params, i, j = F[fn]
    for k in range(i + 1, j):
        m = re.match(r'\s*auto \w+ = (rv32im_v2_\d+)\(cycle, steps, poly_mix, (.*)\);$', L[k])
        if m:
            callee = m.group(1); a = [re.sub(r'/\*.*?\*/', '', x).strip() for x in split_top(m.group(2))]
            cp = [p.strip().split()[-1] for p in F[callee][0].split(',')][3:]
            assert len(cp) == len(a), (callee, len(cp), len(a))
            buf[callee] = {p: buf[fn].get(x) for p, x in zip(cp, a) if buf[fn].get(x)}
# per-function definitions
defs = {}
for fn in order:
    params, i, j = F[fn]; d = {}
    for k in range(i + 1, j):
        for st in L[k].split(';'):
            st = st.strip()
            m = re.match(r'auto (x\d+) = (.*)$', st)
            if m: d[m.group(1)] = m.group(2); continue
            m = re.match(r'constexpr Fp (x\d+)\((\d+)\)$', st)
            if m: d[m.group(1)] = 'CONST %s' % m.group(2)
    defs[fn] = d
LDRE = re.compile(r'EF_LD\(([\w\[\]]+), \(uint32_t\)\((\d+)u \* rk_s \+ \(\(rk_c - (\d+)u\) & rk_m\)\)\)$')
memo = {}
def tree(fn, v, depth=0):
    """-> (expr template with {B0..B3} buffer slots, ops, loads) or None"""
    key = (fn, v)
    if key in memo: return memo[key]
    r = None; e = defs[fn].get(v)
    if e is not None and depth < 8:
        if e.startswith('CONST '): r = ('Fp(%su)' % e[6:], 0, 0)
        elif e.startswith('EF_LD('):
            m = LDRE.match(e)
            if m and buf[fn].get(m.group(1)):
                r = ('EF_LD({%s}, (uint32_t)(%su * rk_s + ((rk_c - %su) & rk_m)))' % (buf[fn][m.group(1)], m.group(2), m.group(3)), 0, 1)
        else:
            m = re.match(r'(x\d+) ([-+*]) (x\d+)$', e)
            if m:
                a, b = tree(fn, m.group(1), depth + 1), tree(fn, m.group(3), depth + 1)
                if a and b: r = ('(%s %s %s)' % (a[0], m.group(2), b[0]), a[1] + b[1] + 1, a[2] + b[2])
    memo[key] = r; return r
# trace of slab events in program order
ev = []
for fn in order:
    params, i, j = F[fn]
    for k in range(i + 1, j):
        for m in re.finditer(r'RK_ST1\((\w+), (\d+)u, (\w+)\)', L[k]): ev.append((k, fn, 'S', (int(m.group(2)),), m.group(3)))
        for m in re.finditer(r'RK_ST4\((\w+), (\d+)u, (\d+)u, (\d+)u, (\d+)u, (\w+)\)', L[k]): ev.append((k, fn, 'S', tuple(int(m.group(q)) for q in range(2, 6)), m.group(6)))
        for m in re.finditer(r'RK_LD1\((\w+), (\d+)u\)', L[k]): ev.append((k, fn, 'L', (int(m.group(2)),), None))
        for m in re.finditer(r'RK_LD4\((\w+), (\d+)u, (\d+)u, (\d+)u, (\d+)u\)', L[k]): ev.append((k, fn, 'L', tuple(int(m.group(q)) for q in range(2, 6)), None))
# values: store event -> its loads
vals = []; cur = {}
for n, e in enumerate(ev):
    k, fn, kind, ws, var = e
    if kind == 'S':
        v = {'st': n, 'ws': ws, 'fn': fn, 'var': var, 'lds': []}
        for w in ws: cur[w] = v
        vals.append(v)
    else:
        vs = {id(cur[w]): cur[w] for w in ws}
        assert len(vs) == 1, 'RK_LD4 spanning values'
        list(vs.values())[0]['lds'].append(n)
# choose rematerializations
remat_ld = {}; drop_st = set(); nld = 0
for v in vals:
    if len(v['ws']) != 1: continue
    t = tree(v['fn'], v['var'])
    if not t: continue
    if t[1] > MAXOPS or t[2] > MAXLOADS: continue
    needb = set(re.findall(r'\{(B\d)\}', t[0]))
    chosen = [n for n in v['lds'] if fidx[ev[n][1]] - fidx[v['fn']] >= MINSPAN and needb <= set(buf[ev[n][1]].values())]
    for n in chosen: remat_ld[n] = t[0]
    nld += len(chosen)
    if len(chosen) == len(v['lds']): drop_st.add(v['st'])
# rewrite loads / stores (by line, in event order within a line)
byline = collections.defaultdict(list)
for n, e in enumerate(ev): byline[e[0]].append(n)
for k, ns in byline.items():
    l = L[k]; fn = ev[ns[0]][1]
    inv = {b: p for p, b in buf[fn].items()}
    for n in ns:
        kind, ws = ev[n][2], ev[n][3]
        if kind == 'L' and n in remat_ld:
            expr = remat_ld[n].format(**inv)
            if len(ws) == 1: pat_ = r'RK_LD1\(\w+, %du\)' % ws[0]
            else: pat_ = r'RK_LD4\(\w+, %s\)' % ', '.join('%du' % w for w in ws)
            l2 = re.sub(pat_, expr.replace('\\', '\\\\'), l, count=1)
            assert l2 != l; l = l2
        if kind == 'S' and n in drop_st:
            if len(ws) == 1: pat_ = r'RK_ST1\(\w+, %du, %s\);' % (ws[0], ev[n][4])
            else: pat_ = r'RK_ST4\(\w+, %s, %s\);' % (', '.join('%du' % w for w in ws), ev[n][4])
            l2 = re.sub(pat_, '', l, count=1)
            assert l2 != l, l; l = l2
    L[k] = l
# re-colour the remaining slab values
kept = [v for v in vals if v['st'] not in drop_st]
# a value lives from its store to its last non-remat load
newv = []
for v in kept:
    lds = [n for n in v['lds'] if n not in remat_ld]
    assert lds, 'kept store without loads'
    newv.append((ev[v['st']][0], fidx[v['fn']], v, max(lds)))
mapping = {}
# positions: (function index, line) gives program order since functions are visited in order
def pos(n): return (fidx[ev[n][1]], ev[n][0])
items = sorted(((pos(v['st']), pos(last), v) for _, _, v, last in newv), key=lambda x: x[0])
free = []; active = []; nw = 0
for st, en, v in items:
    while active and active[0][0] < st:
        _, _, ws = heapq.heappop(active); free.extend(ws)
    free.sort()
    need = len(v['ws']); got = []
    while free and len(got) < need: got.append(free.pop(0))
    while len(got) < need: got.append(nw); nw += 1
    v['new'] = tuple(got); heapq.heappush(active, (en, id(v), got))
# rewrite: each event maps its old words to the value's new words
for v in kept:
    mp = dict(zip(v['ws'], v['new']))
    for n in [v['st']] + [n for n in v['lds'] if n not in remat_ld]:
        mapping[n] = mp
for k, ns in byline.items():
    l = L[k]
    # rewrite the RK_ST/RK_LD occurrences of the line in event order
    evs = [n for n in ns if n in mapping]
    pat = re.compile(r'(RK_ST1\(\w+, |RK_LD1\(\w+, |RK_ST4\(\w+, |RK_LD4\(\w+, )((?:\d+u, )*\d+u)')
    it = iter(evs)
    def sub(m):
        n = next(it); mp = mapping[n]
        ws = [int(x[:-1]) for x in m.group(2).split(', ')]
        assert tuple(ws) == ev[n][3], (ws, ev[n])
        return m.group(1) + ', '.join('%du' % mp[w] for w in ws)
    L[k] = pat.sub(sub, l)
nwords = nw
# a line that held only dropped stores is removed (no whitespace-only lines)
out = '\n'.join(l for l in L if l == '' or l.strip())
out = re.sub(r'#define RK_NWORDS \d+u', '#define RK_NWORDS %du' % max(nwords, 1), out)
open(args[2], 'w').write(out)
print('remat: %d values, %d loads rematerialized, %d stores dropped, slab words %s' % (
    sum(1 for v in vals if any(n in remat_ld for n in v['lds'])), nld, len(drop_st), nwords), file=sys.stderr)
