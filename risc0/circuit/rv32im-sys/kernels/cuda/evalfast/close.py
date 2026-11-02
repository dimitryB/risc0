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

"""close.py: lazy closure accumulation for `FpExt Y = B + A1 * A2 * EF_PM(k);`.

Current code: u = A1*A2 (FpExt), u*PM_k as a full FpExt multiply (3 per-row Montgomery muls to
fold NBETA into the constant, 16 MACs, 4 reductions), then a canonical FpExt add.
New: the NBETA-folded constant comes from a second __constant__ table filled by ef_prep
(ef_pmn_raw[k] = (b1*NBETA, b2*NBETA, b3*NBETA, 0)); B enters as the hi word of a lazy 64-bit
accumulator (value B*2^32 = B*R, i.e. B after REDC); products are added in pairs (each pair < 2P^2,
hi grows < 0.9375P) with a fold after every pair so hi < P; rkw_fin reduces once.
--chain: when B is itself a closure result, continue its accumulator (skips its reduction; the
unused rkw_fin of the intermediate is dead code).
Also turns on the inline PTX mad.wide (EF_MADW_ASM) of the new helpers, CUDA only.
usage: close.py in.cu out.cu [--chain]
"""
import re, sys
src = open(sys.argv[1]).read(); chain = '--chain' in sys.argv
HELP = r'''
// ---- close.py: lazy closures B + u * PM_k with a precomputed NBETA-folded constant table ----
__device__ __forceinline__ static uint64_t ef_madw(uint32_t a, uint32_t b, uint64_t c) {
#ifdef EF_MADW_ASM
  uint64_t d; asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(d) : "r"(a), "r"(b), "l"(c)); return d;
#else
  return c + uint64_t(a) * uint64_t(b);
#endif
}
__device__ __forceinline__ static uint64_t ef_fold64(uint64_t t) {
  uint32_t h = uint32_t(t >> 32), f = h - Fp::P; h = h < f ? h : f;
  return (uint64_t(h) << 32) | uint32_t(t);
}
// pm row j of the NBETA-folded 4x4 matrix of constant k
#define EF_PMROW(b, n) const uint32_t X[4][4] = {{b.x, n.z, n.y, n.x}, {b.y, b.x, n.z, n.y}, {b.z, b.y, b.x, n.z}, {b.w, b.z, b.y, b.x}}
__device__ __forceinline__ static void rk_cl_acc(RkW& w, const FpExt& u, uint32_t k) {
  const uint32_t a0 = u.elems[0].asRaw(), a1 = u.elems[1].asRaw(), a2 = u.elems[2].asRaw(), a3 = u.elems[3].asRaw();
  const uint4 b = reinterpret_cast<const uint4*>(ef_pm_raw)[k], n = reinterpret_cast<const uint4*>(ef_pmn_raw)[k];
  EF_PMROW(b, n);
  for (int j = 0; j < 4; j++) {
    uint64_t t = (uint64_t(w.hi[j]) << 32) | w.lo[j];
    t = ef_fold64(ef_madw(a1, X[j][1], ef_madw(a0, X[j][0], t)));
    t = ef_fold64(ef_madw(a3, X[j][3], ef_madw(a2, X[j][2], t)));
    w.lo[j] = uint32_t(t); w.hi[j] = uint32_t(t >> 32);
  }
}
__device__ __forceinline__ static RkW rk_cl0(const FpExt& b, const FpExt& u, uint32_t k) {
  RkW w; for (int j = 0; j < 4; j++) { w.lo[j] = 0; w.hi[j] = b.elems[j].asRaw(); }
  rk_cl_acc(w, u, k); return w;
}
'''
pat = re.compile(r'^(\s*)FpExt (x\d+) = (\w+) \+ (\w+) \* (\w+) \* EF_PM\((\d+)\);$')
lines = src.split('\n'); n = nc = 0; have = set()
fn = None
for i, l in enumerate(lines):
    if re.match(r'^__device__ (EF_NOINLINE )?FpExt \w+\(.*\) \{$', l): have = set()
    m = pat.match(l)
    if not m: continue
    ind, y, b, a1, a2, k = m.groups()
    if chain and b in have:
        lines[i] = '%sRkW _c%s = _c%s; rk_cl_acc(_c%s, %s * %s, %su); FpExt %s = rkw_fin(_c%s);' % (ind, y, b, y, a1, a2, k, y, y); nc += 1
    else:
        lines[i] = '%sRkW _c%s = rk_cl0(%s, %s * %s, %su); FpExt %s = rkw_fin(_c%s);' % (ind, y, b, a1, a2, k, y, y)
    have.add(y); n += 1
src = '\n'.join(lines)
anchor = '// ---- algebraic rewrite helpers (ec-algebra) ----'
src = src.replace(anchor, HELP + anchor, 1)
old = '__constant__ uint32_t ef_pm_raw[EF_NPM * 4];\n'
assert old in src
src = src.replace(old, '__constant__ __align__(16) uint32_t ef_pm_raw[EF_NPM * 4];\n__constant__ __align__(16) uint32_t ef_pmn_raw[EF_NPM * 4];\n', 1)
# host: fill ef_pmn_raw from the final table
old = '    CUDA_OK(cudaMemcpyToSymbolAsync(ef_pm_raw, b.tab, EF_NPM * sizeof(FpExt), 0, cudaMemcpyDeviceToDevice, stream));\n'
assert old in src
src = src.replace(old, old + '    ef_prep_n<<<(EF_NPM + 127) / 128, 128, 0, stream>>>(b.tab, b.tabn);\n    CUDA_OK(cudaGetLastError());\n'
                  '    CUDA_OK(cudaMemcpyToSymbolAsync(ef_pmn_raw, b.tabn, EF_NPM * sizeof(FpExt), 0, cudaMemcpyDeviceToDevice, stream));\n')
old = 'struct EfBufs { FpExt* pm = nullptr; FpExt* tab = nullptr; uint32_t* q = nullptr; };'
assert old in src
src = src.replace(old, 'struct EfBufs { FpExt* pm = nullptr; FpExt* tab = nullptr; uint32_t* q = nullptr; FpExt* tabn = nullptr; };')
old = '    CUDA_OK(cudaMalloc(&bufs[dev].q, 4 * sizeof(uint32_t)));\n'
assert old in src
src = src.replace(old, old + '    CUDA_OK(cudaMalloc(&bufs[dev].tabn, EF_NPM * sizeof(FpExt)));\n')
old = '__launch_bounds__(EF_THREADS, EF_MINB)\n__global__ void ef_eval_check('
assert old in src
src = src.replace(old, '__global__ void ef_prep_n(const FpExt* tab, FpExt* tabn) {\n'
  '  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i >= EF_NPM) return;\n'
  '  tabn[i] = FpExt(tab[i].elems[1] * Fp(Fp::P - 11), tab[i].elems[2] * Fp(Fp::P - 11), tab[i].elems[3] * Fp(Fp::P - 11), Fp(0));\n}\n\n' + old)
old = '#define EF_WIDE 1\n'
assert src.count(old) == 1
src = src.replace(old, old + '#if !defined(__HIPCC__)\n#define EF_MADW_ASM 1  // inline PTX mad.wide; ROCm keeps the C++ path\n#endif\n')
open(sys.argv[2], 'w').write(src)
print('close: %d closures, %d chained' % (n, nc), file=sys.stderr)
