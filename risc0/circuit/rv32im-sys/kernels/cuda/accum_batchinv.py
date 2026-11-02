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

# usage: accum_batchinv.py [--check]
#
# Rewrites exec_TopAccum in steps.cu (next to this script) in place: Montgomery
# batch inversion of the FpExt inv_0 calls inside each mux arm (the CUDA port of
# kernels/intel/accum_batchinv.py). Pure denominator slices (data/mix loads and
# arithmetic; no accum-buffer loads, no inv_0, no STORE) are hoisted to just
# before the first use of their chunk; each chunk of M inverses shares one Fp
# inversion (inv_chain) of the product of the norms c. Zero norms map to 0,
# exactly as inv(0) = 0. Bit-exact by construction (unique field inverse,
# canonical form).
#
# Re-run it after regenerating steps.cu. The rewrite is idempotent, so
# --check (run in CI) fails if steps.cu still has inverses to batch.
import os, re, sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'steps.cu')
CHECK = '--check' in sys.argv[1:]
M = 8
src = open(PATH, newline='').read()
fs = src.index('__device__ ComponentStruct exec_TopAccum(ExecContext& ctx')
fe = src.index('\n__device__ void step_TopAccum(', fs)
func = src[fs:fe].split('\n')

# arm boundaries: lines starting an arm condition, and the line where the condition ends ("{")
arm_starts = [i for i, l in enumerate(func) if re.match(r'^  (\} else )?if \(to_size_t\(', l)]
else_line = max(i for i, l in enumerate(func) if l == '  } else {')
arms = []
for k, s in enumerate(arm_starts):
    j = s
    while not func[j].rstrip().endswith('{'):
        j += 1
    end = arm_starts[k + 1] if k + 1 < len(arm_starts) else else_line
    arms.append((s, j, end))  # cond lines s..j, body j+1..end-1


def split_statements(lines):
    """Group body lines into statements (list of line lists). Comments attach to next stmt."""
    stmts, cur, depth, pending = [], [], 0, []
    for l in lines:
        st = l.strip()
        if not cur and st.startswith('//'):
            pending.append(l)
            continue
        cur.append(l)
        depth += l.count('(') - l.count(')') + l.count('{') - l.count('}')
        if depth == 0 and st.endswith(';'):
            stmts.append((pending, cur))
            cur, pending = [], []
    assert not cur, cur[:3]
    return stmts, pending


def_re = re.compile(r'^\s*(ExtVal|Val) (x\d+) =\s*(.*);\s*$', re.S)
var_re = re.compile(r'\bx\d+\b')


def emit_chunk(dens, tag, ind):
    L = []
    n = len(dens)
    for i, d in enumerate(dens):
        p = f'_bi{tag}_{i}'
        L.append(f'Fp {p}b0 = {d}.elems[0] * {d}.elems[0] + Fp(11) * ({d}.elems[1] * ({d}.elems[3] + {d}.elems[3]) - {d}.elems[2] * {d}.elems[2]);')
        L.append(f'Fp {p}b2 = {d}.elems[0] * ({d}.elems[2] + {d}.elems[2]) - {d}.elems[1] * {d}.elems[1] + Fp(11) * ({d}.elems[3] * {d}.elems[3]);')
        L.append(f'Fp {p}c = {p}b0 * {p}b0 + Fp(11) * {p}b2 * {p}b2;')
        L.append(f'bool {p}z = ({p}c == Fp(0));')
        L.append(f'Fp {p}cs = {p}z ? Fp(1) : {p}c;')
        L.append(f'Fp {p}pp = {p}cs;' if i == 0 else f'Fp {p}pp = _bi{tag}_{i-1}pp * {p}cs;')
    L.append(f'Fp _bi{tag}_r = inv_chain(_bi{tag}_{n-1}pp);')
    for i in range(n - 1, -1, -1):
        p = f'_bi{tag}_{i}'
        if i > 0:
            L.append(f'Fp {p}ic = {p}z ? Fp(0) : (_bi{tag}_r * _bi{tag}_{i-1}pp);')
            L.append(f'_bi{tag}_r = _bi{tag}_r * {p}cs;')
        else:
            L.append(f'Fp {p}ic = {p}z ? Fp(0) : _bi{tag}_r;')
    for i, d in enumerate(dens):
        p = f'_bi{tag}_{i}'
        L.append(f'{p}b0 *= {p}ic; {p}b2 *= {p}ic;')
        L.append(f'ExtVal _iv_{d} = ExtVal({d}.elems[0] * {p}b0 + Fp(11) * {d}.elems[2] * {p}b2, '
                 f'-{d}.elems[1] * {p}b0 + Fp(Fp::P - 11) * {d}.elems[3] * {p}b2, '
                 f'-{d}.elems[0] * {p}b2 + {d}.elems[2] * {p}b0, '
                 f'{d}.elems[1] * {p}b2 - {d}.elems[3] * {p}b0);')
    return [ind + x for x in L]


out = func[:arms[0][0]]
stats = []
ctr = 0
for a, (s, j, e) in enumerate(arms):
    out.extend(func[s:j + 1])
    stmts, trailing = split_statements(func[j + 1:e])
    text = ['\n'.join(c) for _, c in stmts]
    defs = {}
    for i, t in enumerate(text):
        m = def_re.match(t)
        if m:
            defs[m.group(2)] = (i, m.group(1), m.group(3))
    order, first_use = [], {}
    for i, t in enumerate(text):
        for v in re.findall(r'inv_0\((x\d+)\)', t):
            if v not in first_use:
                first_use[v] = i
                order.append(v)

    def slice_of(v, acc):
        if v not in defs or v in acc:
            return
        acc.add(v)
        for u in var_re.findall(defs[v][2]):
            slice_of(u, acc)

    ok = []
    for v in order:
        if v not in defs or defs[v][1] != 'ExtVal':
            continue
        sl = set()
        slice_of(v, sl)
        # only vars defined in this arm are in defs; anything else (x4, arg0...) is outer & fixed
        bad = any(('layout1' in defs[u][2]) or ('inv_0' in defs[u][2]) or ('STORE' in defs[u][2])
                  for u in sl)
        if bad:
            continue
        ok.append((v, sl))
    chunks = [ok[i:i + M] for i in range(0, len(ok), M)]
    ind = '    '
    insert_before = {}  # stmt index -> list of (moved stmt indices, chunk vars)
    moved = set()
    for ch in chunks:
        at = min(first_use[v] for v, _ in ch)
        need = set()
        for _, sl in ch:
            need |= sl
        idxs = sorted(defs[u][0] for u in need if defs[u][0] >= at and defs[u][0] not in moved)
        # all slice members defined before `at` stay in place; later ones move up
        moved |= set(idxs)
        insert_before.setdefault(at, []).append((idxs, [v for v, _ in ch]))
    replaced = {v for ch in chunks for v, _ in ch}
    body = []
    for i, (com, c) in enumerate(stmts):
        if i in insert_before:
            for idxs, vs in insert_before[i]:
                for k in idxs:
                    body.extend(stmts[k][0] + stmts[k][1])
                body.extend(emit_chunk(vs, ctr, ind))
                ctr += 1
        if i in moved:
            continue
        t = '\n'.join(com + c)
        t = re.sub(r'inv_0\((x\d+)\)', lambda m: f'_iv_{m.group(1)}' if m.group(1) in replaced else m.group(0), t)
        body.append(t)
    body.extend(trailing)
    out.extend(body)
    stats.append((a, len(order), len(ok)))
out.extend(func[else_line:])
res = src[:fs] + '\n'.join(out) + src[fe:]
tot = sum(x[1] for x in stats)
okc = sum(x[2] for x in stats)
if CHECK:
    if res != src:
        sys.exit(f'accum_batchinv: steps.cu has {okc} unbatched inverses; '
                 'run kernels/cuda/accum_batchinv.py and commit the result')
    print('accum_batchinv: steps.cu is up to date')
    sys.exit(0)
open(PATH, 'w', newline='').write(res)
print(f'M={M}: {okc}/{tot} inverses batched over {len(arms)} arms, {ctr} chunks')
for a, n, k in stats:
    print(f'  arm {a}: {k}/{n}')
