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

"""accfirst_cuda.py: CUDA port of the "accum block first" eval_check pass. Slices the LogUp/accum block (tail of
rv32im_v2_3 from the first accum load after the accumulator x1102, plus v2_2..v2_0 via the tail
call) into rk_accblk(); v2_3 returns x1102, so poly_fp() = main block and
tot = poly_fp() + rk_accblk() (the accumulator enters the tail only additively; the tail gets
x1102 := 0). The kernel runs the main block first, then the acc block.
Asserts: the tail and the pulled-in defs use no FpExt argument other than the zero base arg2,
pulled-in defs are pure (no slab stores / links), and every slab word the acc block reads is
written earlier inside the acc block (so the blocks can run in either order).
usage: accfirst_cuda.py in.cu out.cu
"""
import re, sys
src = open(sys.argv[1]).read()
m = re.search(r'^__device__ EF_NOINLINE FpExt rv32im_v2_3\(([^)]*)\) \{\n', src, re.M)
params = m.group(1); start = m.end(); end = src.index('\n}\n', start)
body = src[start:end].split('\n')
ACC = 'x1102'
sp = [i for i, l in enumerate(body) if re.match(r'\s*auto x231 = EF_LD\(arg8, ', l)]
assert len(sp) == 1, sp; sp = sp[0]
assert any(re.match(r'\s*FpExt %s = ' % ACC, l) for l in body[:sp])
dre = re.compile(r'\s*(?:auto|FpExt|Fp|RkAcc|RkW|RkW1|constexpr Fp|constexpr FpExt) (_?c?x\w+)\b')
defs = {}
for i, l in enumerate(body[:sp]):
    for st in l.split(';'):
        mm = dre.match(st)
        if mm: defs[mm.group(1)] = i
need = set(); work = []
def scan(l):
    for v in re.findall(r'\b(_?c?x\w+)\b', l):
        if v in defs and v != ACC and v not in need: need.add(v); work.append(v)
for l in body[sp:]: scan(l)
while work:
    v = work.pop(); l = body[defs[v]]
    assert 'RK_ST' not in l and 'rk_link' not in l and 'rk_cl' not in l and 'RK_LD' not in l, l
    scan(l)
pre = [body[i] for i in sorted(set(defs[v] for v in need))]
tail = body[sp:]
used_args = set(re.findall(r'\barg(\d+)\b', '\n'.join(pre + tail)))
pl = [p.strip() for p in params.split(',')]
ptr_params = [p for p in pl if '*' in p and p.split()[-1].startswith('arg')]
ptr_names = [p.split()[-1] for p in ptr_params]
for a in used_args:
    nm = 'arg' + a
    assert nm in ptr_names or nm == 'arg2', 'tail uses FpExt arg ' + nm
hdr = [l for l in body[:sp] if 'size_t mask' in l]
acc_fn = ('__device__ EF_NOINLINE FpExt rk_accblk(size_t cycle, size_t steps, FpExt* poly_mix, %s) {\n'
          '  const FpExt arg2 = FpExt(0); const FpExt %s = FpExt(0);\n' % (', '.join(ptr_params), ACC)
          + '\n'.join(hdr) + '\n' + '\n'.join(pre) + '\n' + '\n'.join(tail) + '\n}\n')
src = src[:start] + '\n'.join(body[:sp] + ['  return %s;' % ACC]) + src[end:]
k = src.index('__device__ FpExt poly_fp(size_t cycle, size_t steps, FpExt* poly_mix, Fp** args) {')
src = src[:k] + acc_fn + '\n' + src[k:]
# slab self-containment of the acc block: loads after stores within acc block (program order: tail, v2_2, v2_1, v2_0)
seq = '\n'.join(tail)
for fn in ['rv32im_v2_2', 'rv32im_v2_1', 'rv32im_v2_0']:
    mm = re.search(r'^__device__ EF_NOINLINE FpExt %s\([^)]*\) \{\n' % fn, src, re.M)
    seq += '\n' + src[mm.end():src.index('\n}\n', mm.end())]
written = set()
for line in seq.split('\n'):
    for st in line.split(';'):
        for mm in re.finditer(r'RK_LD[14]\(\w+, ((?:\d+u, )*\d+u)\)', st):
            for w in mm.group(1).split(', '): assert int(w[:-1]) in written, ('acc block reads word not written in it', st)
        for mm in re.finditer(r'RK_ST1\(\w+, (\d+)u,', st): written.add(int(mm.group(1)))
        for mm in re.finditer(r'RK_ST4\(\w+, (\d+)u, (\d+)u, (\d+)u, (\d+)u,', st): written.update(int(mm.group(q)) for q in range(1, 5))
# kernel: map v2_3's pointer params to kernel-level buffers by following the call chain
def split_top(s):
    out, d, cur = [], 0, ''
    for ch in s:
        if ch == '(': d += 1
        if ch == ')': d -= 1
        if ch == ',' and d == 0: out.append(cur.strip()); cur = ''
        else: cur += ch
    out.append(cur.strip()); return out
fns = {}
for mm in re.finditer(r'^__device__ (?:EF_NOINLINE )?FpExt (rv32im_v2_\d+|poly_fp)\(([^)]*)\) \{\n', src, re.M):
    fns[mm.group(1)] = (mm.group(2), mm.end(), src.index('\n}\n', mm.end()))
root = {'args[%d]' % q: 'args[%d]' % q for q in range(5)}
root['x33'] = 'args[4]'; root['x34'] = 'args[4]'
cur = 'poly_fp'; mp = root
for callee in ['rv32im_v2_%d' % q for q in range(19, 2, -1)]:
    p, a, b = fns[cur]
    mm = re.search(r'auto \w+ = %s\(cycle, steps, poly_mix, (.*)\);' % callee, src[a:b])
    args_ = [re.sub(r'/\*.*?\*/', '', x).strip() for x in split_top(mm.group(1))]
    cp = [q.strip().split()[-1] for q in fns[callee][0].split(',')][3:]
    mp = {pn: mp.get(x) for pn, x in zip(cp, args_) if mp.get(x)}
    cur = callee
call_args = []
for p in ptr_params:
    nm = p.split()[-1]; r = mp[nm]
    call_args.append('reinterpret_cast<%s>(%s)' % (p.rsplit(' ', 1)[0], 'args[%s]' % r[5:-1]))
call = 'rk::rk_accblk((size_t)cycle, (size_t)domain, pm, %s)' % ', '.join(call_args)
old = '  FpExt tot = rk::poly_fp((size_t)cycle, (size_t)domain, pm, args);\n'
assert old in src
new = '  FpExt tot = rk::poly_fp((size_t)cycle, (size_t)domain, pm, args);\n  FpExt rk_a = %s;\n  tot = tot + rk_a;\n' % call
src = src.replace(old, new)
open(sys.argv[2], 'w').write(src)
print('accfirst_cuda: %d tail lines, %d pulled-in defs, acc args %s' % (len(tail), len(pre), call_args), file=sys.stderr)
