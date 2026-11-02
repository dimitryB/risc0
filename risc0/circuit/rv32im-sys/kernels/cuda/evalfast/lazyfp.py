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

"""lazyfp.py: lazy (non-canonical, [0, 2P)) Fp add/sub/mul results in eval_check_ef.cu.

A lazy value L in [0, 2P) is only consumed where the result is still exact and canonical:
  * Montgomery mul (EF_FP_FAST subtraction-form REDC) with the other operand canonical (< P):
    a*b < 2P^2 -> hi < 0.9375P, r = hi - mulhi(m,P) in (-P, 0.9375P), min(r, r+P) canonical.
  * Fp * FpExt (4 such muls, FpExt elements canonical).
  * rk_link (folding) with a base whose hi < P (FpExt base, or an RkAcc from a folding rk_link):
    hi + 0.9375P + 1 < 2^32, the fold leaves hi < P.
  * rkw1_ff0 / rkw1_ff / rkw_ef0 / rkw_ef with neg=false and the other operand canonical.
Producers: a+b -> raw add (< 2P); a-b -> a + (P - b) in [1, 2P); a*b -> r + P in (0, 1.9375P).
usage: lazyfp.py in.cu out.cu
"""
import re, sys
src = open(sys.argv[1]).read()
lines = src.split('\n')
ID = re.compile(r'\b(x\d+|arg\d+(?:_e\d)?|_wx\d+)\b')
fstart = re.compile(r'^__device__ (?:EF_NOINLINE )?FpExt (rv32im_v2_\d+|poly_fp)\((.*)\) \{$')
HELP = '''
// ---- lazyfp.py: lazy [0, 2P) Fp results (see evalfast/lazyfp.py for the bounds) ----
__device__ __forceinline__ static ::risc0::Fp ef_ladd(::risc0::Fp a, ::risc0::Fp b) { return ::risc0::Fp::fromRaw(a.asRaw() + b.asRaw()); }
__device__ __forceinline__ static ::risc0::Fp ef_lsub(::risc0::Fp a, ::risc0::Fp b) { return ::risc0::Fp::fromRaw(a.asRaw() + (::risc0::Fp::P - b.asRaw())); }
__device__ __forceinline__ static ::risc0::Fp ef_lmul(::risc0::Fp a, ::risc0::Fp b) {
  const uint32_t x = a.asRaw(), y = b.asRaw(); const uint32_t lo = x * y; const uint32_t hi = uint32_t((uint64_t(x) * uint64_t(y)) >> 32);
  const uint32_t m = lo * ::risc0::Fp::M; return ::risc0::Fp::fromRaw(hi - uint32_t((uint64_t(m) * uint64_t(::risc0::Fp::P)) >> 32) + ::risc0::Fp::P);
}
'''
stats = dict(cand=0, add=0, sub=0, mul=0)
out = list(lines)
i = 0
while i < len(lines):
    m = fstart.match(lines[i])
    if not m:
        i += 1; continue
    j = i + 1
    while lines[j] != '}': j += 1
    body = range(i + 1, j)
    typ = {}
    for p in m.group(2).split(','):
        p = p.strip(); mm = re.match(r'(Fp|FpExt)(\*?) (\w+)$', p)
        if mm: typ[mm.group(3)] = ('P' if mm.group(2) else '') + mm.group(1)
    defs = {}; nfacc = set()
    for k in body:
        l = lines[k].strip()
        mm = re.match(r'FpExt (arg\d+)\(', l)
        if mm: typ[mm.group(1)] = 'FpExt'; continue
        mm = re.match(r'constexpr Fp (x\d+)\(', l)
        if mm: typ[mm.group(1)] = 'Fp'; continue
        mm = re.match(r'constexpr FpExt (x\d+)\(', l)
        if mm: typ[mm.group(1)] = 'FpExt'; continue
        mm = re.match(r'(Fp|FpExt|RkAcc) (x\d+) = (.*);$', l)
        if mm:
            typ[mm.group(2)] = mm.group(1)
            if mm.group(1) == 'RkAcc' and mm.group(3).startswith('rk_link_nf('): nfacc.add(mm.group(2))
            continue
        for st in l.split(';'):
            st = st.strip()
            mm = re.match(r'auto (x\d+) = (.*)$', st)
            if not mm: continue
            v, e = mm.group(1), mm.group(2)
            if e.startswith(('EF_LD(', 'RK_LD1(', 'rkw1_fin(')): typ[v] = 'Fp'
            elif e.startswith(('RK_LD4(', 'rkw_fin(', 'rk_xmul(', 'EF_PM(')): typ[v] = 'FpExt'
            else:
                mo = re.match(r'(\w+) ([-+*]) (\w+)$', e)
                if mo:
                    ta, tb = typ.get(mo.group(1), '?'), typ.get(mo.group(3), '?')
                    if mo.group(1).isdigit() or mo.group(3).isdigit(): typ[v] = '?'
                    elif ta == 'Fp' and tb == 'Fp':
                        typ[v] = 'Fp'; defs[v] = (k, mo.group(1), mo.group(2), mo.group(3))
                    elif 'FpExt' in (ta, tb): typ[v] = 'FpExt'
                    else: typ[v] = '?'
                else:
                    mo = re.match(r'-(\w+)$', e)
                    typ[v] = typ.get(mo.group(1), '?') if mo else '?'
    # uses
    uses = {v: [] for v in defs}
    for k in body:
        l = lines[k].strip()
        for st in l.split(';'):
            st = st.strip()
            if not st: continue
            ids = ID.findall(st)
            hit = [v for v in set(ids) if v in uses]
            if not hit: continue
            mm = re.match(r'auto (x\d+) = (.*)$', st)
            lhs = mm.group(1) if mm else None
            for v in hit:
                if v == lhs and ids.count(v) == 1: continue  # its definition
                uses[v].append((k, st))
    lazy = set()
    def ok_use(v, st):
        mm = re.match(r'auto x\d+ = (\w+) \* (\w+)$', st)
        if mm:
            a, b = mm.group(1), mm.group(2)
            if a == b: return False
            o = b if a == v else a
            if typ.get(o) == 'FpExt': return True
            if typ.get(o) == 'Fp' and o not in lazy: return True
            return False
        mm = re.search(r'\brk_link\((\w+), (\w+), EF_PM\(\d+\)\)', st)
        if mm and mm.group(2) == v and st.count(v) == 1:
            base = mm.group(1)
            return typ.get(base) == 'FpExt' or (typ.get(base) == 'RkAcc' and base not in nfacc)
        mm = re.match(r'(?:RkW1 _wx\d+ = rkw1_ff0\(|rkw1_ff\(_wx\d+, )(\w+), (\w+), false\)$', st)
        if mm:
            a, b = mm.group(1), mm.group(2)
            if a == b: return False
            o = b if a == v else a
            return typ.get(o) == 'Fp' and o not in lazy
        mm = re.match(r'(?:RkW _wx\d+ = rkw_ef0\(|rkw_ef\(_wx\d+, )(\w+), (\w+), false\)$', st)
        if mm and mm.group(1) == v and mm.group(2) != v: return True
        mm = re.match(r'FpExt x\d+ = \w+ \+ (\w+) \* (\w+) \* EF_PM\(\d+\)$', st)
        if mm and mm.group(1) == v and typ.get(mm.group(2)) == 'FpExt' and st.count(v) == 1: return True
        return False
    for v in sorted(defs, key=lambda v: defs[v][0]):
        stats['cand'] += 1
        k, a, op, b = defs[v]
        kind = {'+': 'add', '-': 'sub', '*': 'mul'}[op]
        if a in lazy and b in lazy: continue
        if kind in ('add', 'sub') and (a in lazy or b in lazy): continue
        if not uses[v]: continue
        if all(ok_use(v, st) for _, st in uses[v]):
            lazy.add(v); stats[kind] += 1
            fn = {'add': 'ef_ladd', 'sub': 'ef_lsub', 'mul': 'ef_lmul'}[kind]
            old = 'auto %s = %s %s %s;' % (v, a, op, b)
            assert old in out[k], (old, out[k])
            out[k] = out[k].replace(old, 'auto %s = %s(%s, %s);' % (v, fn, a, b))
    i = j + 1
res = '\n'.join(out)
anchor = '// ---- algebraic rewrite helpers (ec-algebra) ----'
assert anchor in res
res = res.replace(anchor, HELP + anchor, 1)
open(sys.argv[2], 'w').write(res)
print('lazyfp:', stats, file=sys.stderr)
