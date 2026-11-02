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

# Narrower chain-function calls in the ROCm eval_check: regen.py applies convert() last, after
# hoist_hip.py.
#
# The chain functions (rv32im_v2_N, rk_accblk) are EF_NOINLINE or too big to inline, and the
# AMDGPU calling convention passes every argument in VGPRs (v0-v31, the rest on the stack, i.e.
# in scratch), uniform or not; a callee that runs out of registers spills them again. Three
# kinds of argument are dropped here:
#  - cycle and steps are 32-bit values (the chains only use (uint32_t) casts of them): two
#    VGPRs each instead of four;
#  - the poly_mix pointer is unused on HIP (the mix powers come from ef_pm_raw);
#  - the data / accum / mix / out pointers and the FpExt* view of the slab: the pointers are
#    the same for every lane and call, so ef_prep stores them in a device table (ef_ptrs, like
#    ef_pm_raw: one eval_check at a time per device, on its own stream) that the callees read
#    with scalar loads; the FpExt* slab view is the Fp* one cast.
# Pointer identities come from the kernel's args[] array and the call sites (through local
# aliases); a parameter whose identity is not proven keeps its argument. Measured on the ef
# harness at po2 20 (medians of interleaved runs, bit-exact): RX 7900 XTX 174 -> 171 ms (most
# of it from the 32-bit cycle/steps), RX 9070 XT 183 -> 174 ms.
#
# usage: args_hip.py <in.cu> <out.cu>
import re
import sys

FNDEF = re.compile(r'^__device__ (?:EF_NOINLINE )?FpExt (\w+)\((.*)\) \{$', re.M)
FNSIG = re.compile(r'^(__device__ (?:EF_NOINLINE )?)FpExt (\w+)\((.*)\)( \{|;)$', re.M)
CALL = re.compile(r'(= )(rv32im_v2_\d+)\((.*)\);')
ALIAS = re.compile(r'(?:Fp|FpExt)\* (\w+) = ([^;]+);')
TABLE = {'data': 0, 'accum': 1, 'mix': 2, 'out': 3}


def _sub(s, old, new, count):
    n = s.count(old)
    if n != count:
        sys.exit('args_hip: expected %d of %r, found %d' % (count, old[:80], n))
    return s.replace(old, new)


def _strip(a):
    a = re.sub(r'/\*.*?\*/', '', a).strip()
    m = re.fullmatch(r'reinterpret_cast<\w+\*>\((.*)\)', a)
    return m.group(1).strip() if m else a


def _pname(p):
    return p.split()[-1].lstrip('*')


def _analyze(s):
    # (function, name) -> 'data' | 'accum' | 'mix' | 'out' | 'slab'
    kern = s[s.index('__global__ void ef_eval_split('):]
    arr = re.search(r'Fp\* args\[5\] = \{(.*?)\};', kern).group(1).split(', ')
    roots = {'args[%d]' % i: a for i, a in enumerate(arr)}
    defs, bodies = {}, {}
    for m in FNDEF.finditer(s):
        defs[m.group(1)] = [p.strip() for p in m.group(2).split(',')]
        bodies[m.group(1)] = s[m.end():s.index('\n}\n', m.end())]
    ident = {}
    acc = re.search(r'rk::rk_accblk\((.*?)\); dst = scratch;', kern).group(1).split(', ')
    for p, a in zip(defs['rk_accblk'], acc):
        if roots.get(_strip(a)):
            ident[('rk_accblk', _pname(p))] = roots[_strip(a)]
    calls = [(f, m.group(2), m.group(3).split(', ')) for f, b in bodies.items() for m in CALL.finditer(b)]

    def look(f, a):
        a = _strip(a)
        return ident.get((f, a)) or (roots.get(a) if f == 'poly_fp' else None)

    changed = True
    while changed:
        changed = False
        for f, b in bodies.items():
            for m in ALIAS.finditer(b):
                v = look(f, m.group(2))
                if v and ident.get((f, m.group(1))) != v:
                    ident[(f, m.group(1))] = v
                    changed = True
        for f, g, args in calls:
            for p, a in zip(defs[g], args):
                v = look(f, a)
                if '*' in p and v and ident.get((g, _pname(p))) != v:
                    ident[(g, _pname(p))] = v
                    changed = True
    return defs, ident


def convert(s, stats=None):
    # 32-bit cycle/steps, no poly_mix
    s = _sub(s, 'size_t cycle, size_t steps, FpExt* poly_mix, ', 'uint32_t cycle, uint32_t steps, ', 106)
    s = _sub(s, '(cycle, steps, poly_mix, ', '(cycle, steps, ', 20)
    s = _sub(s, 'rk::poly_fp((size_t)cycle, (size_t)domain, pm, args)', 'rk::poly_fp(cycle, domain, args)', 1)
    s = _sub(s, 'rk::rk_accblk((size_t)cycle, (size_t)domain, pm, ', 'rk::rk_accblk(cycle, domain, ', 1)
    # pointer table
    defs, ident = _analyze(s)
    drop, local = {}, {}
    for f, ps in defs.items():
        if f == 'poly_fp':
            continue
        names = [_pname(p) for p in ps]
        slab = [n for p, n in zip(ps, names) if p.startswith('Fp* ') and ident.get((f, n)) == 'slab']
        drop[f], local[f] = set(), []
        for i, (p, n) in enumerate(zip(ps, names)):
            a = ident.get((f, n))
            if not p.startswith(('Fp* ', 'FpExt* ')) or a is None:
                continue
            if a in TABLE and p.startswith('Fp* '):
                drop[f].add(i)
                local[f].append('  Fp* %s = EF_GPTR(%d);' % (n, TABLE[a]))
            elif a == 'slab' and p.startswith('FpExt* ') and slab:
                drop[f].add(i)
                local[f].append('  FpExt* %s = reinterpret_cast<FpExt*>(%s);' % (n, slab[0]))
    nsig = [0]

    def sig(m):
        f = m.group(2)
        if f not in drop:
            return m.group(0)
        nsig[0] += 1
        ps = [p.strip() for p in m.group(3).split(',')]
        tail = m.group(4) + ''.join('\n' + l for l in local[f]) if m.group(4) == ' {' else m.group(4)
        return '%sFpExt %s(%s)%s' % (m.group(1), f, ', '.join(p for i, p in enumerate(ps) if i not in drop[f]), tail)

    s = FNSIG.sub(sig, s)
    ncall = [0]

    def call(m):
        ncall[0] += 1
        args = m.group(3).split(', ')
        return '%s%s(%s);' % (m.group(1), m.group(2), ', '.join(a for i, a in enumerate(args) if i not in drop[m.group(2)]))

    s = CALL.sub(call, s)
    old = re.search(r'rk::rk_accblk\((.*?)\); dst = scratch;', s).group(1)
    s = _sub(s, old, ', '.join(a for i, a in enumerate(old.split(', ')) if i not in drop['rk_accblk']), 1)
    s = _sub(s, '#define EF_PM(i) ', '''// Kernel pointer table (data, accum, mix, out), written by ef_prep, read with scalar loads in the
// chain functions.
__device__ uint64_t ef_ptrs[4];
#if defined(__HIP_DEVICE_COMPILE__)
#define EF_GPTR(i) (reinterpret_cast<::risc0::Fp*>(((const __attribute__((address_space(4))) uint64_t*)(reinterpret_cast<uint64_t>(ef_ptrs)))[(i)]))
#else
#define EF_GPTR(i) (reinterpret_cast<::risc0::Fp*>(ef_ptrs[(i)]))
#endif
#define EF_PM(i) ''', 1)
    s = _sub(s, '''__global__ void ef_prep(const FpExt* pm_in, const Fp* out, const Fp* mix, FpExt* tab, uint32_t* qtab,
                        Fp rou, uint32_t po2) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
''', '''__global__ void ef_prep(const FpExt* pm_in, const Fp* out, const Fp* mix, FpExt* tab, uint32_t* qtab,
                        Fp rou, uint32_t po2, const Fp* data, const Fp* accum) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  ef_ptrs[0] = reinterpret_cast<uint64_t>(data);
  ef_ptrs[1] = reinterpret_cast<uint64_t>(accum);
  ef_ptrs[2] = reinterpret_cast<uint64_t>(mix);
  ef_ptrs[3] = reinterpret_cast<uint64_t>(out);
''', 1)
    s = _sub(s, 'ef_prep<<<1, 1, 0, stream>>>(b.pm, out, mix, b.tab, b.q, rou, po2);',
             'ef_prep<<<1, 1, 0, stream>>>(b.pm, out, mix, b.tab, b.q, rou, po2, data, accum);', 1)
    msg = ('args_hip: 32-bit cycle/steps, %d signatures and %d calls without %d pointer arguments'
           % (nsig[0], ncall[0], sum(len(v) for v in drop.values())))
    if stats is not None:
        stats.append(msg)
    else:
        print(msg)
    return s


if __name__ == '__main__':
    with open(sys.argv[1], newline='') as f:
        text = f.read()
    text = convert(text)
    with open(sys.argv[2], 'w', newline='') as f:
        f.write(text)
