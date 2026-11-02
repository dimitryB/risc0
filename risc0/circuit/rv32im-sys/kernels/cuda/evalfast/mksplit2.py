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

"""mksplit2.py: phase-split eval_check (block-level warp specialization).

Replaces ef_eval_check by ef_eval_split: block 2*rb evaluates the main chain (poly_fp) of row block rb and
writes the unscaled partial to check; block 2*rb+1 evaluates the LogUp chain (rk_accblk) into a scratch
buffer; ef_combine computes check = (main + logup) * quot.  FpExt addition mod p is exact and both
partials are canonical, so the result is bit-identical to tot = poly_fp + rk_accblk; check *= quot.
usage: mksplit2.py in.cu out.cu   (in = the accfirst_cuda.py output)
"""
import sys
s=open(sys.argv[1]).read()
a=s.index('__launch_bounds__(EF_THREADS, EF_MINB)\n__global__ void ef_eval_check(')
b=s.index('#include "cuda.h"')
s=s[:a]+'''// Phase-split variant: even blocks evaluate the main chain (poly_fp), odd blocks the LogUp chain
// (rk_accblk) of the same 128 rows; ef_combine adds the two partials and scales by the quotient.
static_assert(EF_SPLIT_SHIFT == 1, "ef_eval_split has two phases: main (even blocks) and LogUp (odd)");
__launch_bounds__(EF_THREADS, EF_MINB)
__global__ void ef_eval_split(Fp* check, Fp* data, Fp* accum, Fp* mix, Fp* out, uint32_t domain, Fp* scratch) {
  FpExt* pm = nullptr;
  const uint32_t rb = blockIdx.x >> EF_SPLIT_SHIFT, ph = blockIdx.x & ((1u << EF_SPLIT_SHIFT) - 1u);
  uint32_t cycle = blockDim.x * rb + threadIdx.x;
  if (cycle >= domain) return;
  uint32_t lslab[RK_NWORDS];
  Fp* slab = reinterpret_cast<Fp*>(lslab);
  Fp* args[5] = {accum, data, out, mix, slab};
  FpExt t;
  Fp* dst;
  if (ph == 0) { t = rk::poly_fp((size_t)cycle, (size_t)domain, pm, args); dst = check; }
  else { t = rk::rk_accblk((size_t)cycle, (size_t)domain, pm, reinterpret_cast<Fp*>(args[4]), reinterpret_cast<FpExt*>(args[4]), reinterpret_cast<Fp*>(args[1]), reinterpret_cast<Fp*>(args[0]), reinterpret_cast<Fp*>(args[3])); dst = scratch; }
  for (uint32_t i = 0; i < 4; i++) dst[i * domain + cycle] = t.elems[i];
}
__global__ void ef_combine(Fp* check, const Fp* scratch, uint32_t domain) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= domain) return;
  Fp quot = Fp::fromRaw(ef_qtab[cycle & 3]);
  for (uint32_t i = 0; i < 4; i++) check[i * domain + cycle] = (check[i * domain + cycle] + scratch[i * domain + cycle]) * quot;
}
'''+s[b:]
# The scratch buffer comes from risc0_cuda_malloc, which drains the HAL's buffer pool and retries
# when the device is full (device_alloc.h, after "cuda.h").
s=s.replace('#include "cuda.h"\n', '#include "cuda.h"\n#include "device_alloc.h"\n', 1)
s=s.replace('#define EF_MINB 3\n','#define EF_MINB 3\n#endif\n#ifndef EF_SPLIT_SHIFT\n#define EF_SPLIT_SHIFT 1\n',1)
old='''    ef_eval_check<<<grid, block, 0, stream>>>(check, const_cast<Fp*>(data), const_cast<Fp*>(accum),
                                              const_cast<Fp*>(mix), const_cast<Fp*>(out), domain);'''
new='''    if (b.scratch_n < domain) {
      constexpr size_t EF_SCRATCH_PAD = 4096;
      if (b.scratch_base) CUDA_OK(cudaFree(b.scratch_base));  // synchronizes the device
      // Reset first: a failed allocation must not leave a dangling buffer that a later, smaller
      // domain would reuse.
      b.scratch_base = nullptr;
      b.scratch = nullptr;
      b.scratch_n = 0;
      CUDA_OK(risc0_cuda_malloc(&b.scratch_base, 4ull * domain * sizeof(Fp) + EF_SCRATCH_PAD));
      b.scratch = reinterpret_cast<Fp*>(b.scratch_base + EF_SCRATCH_PAD);
      b.scratch_n = domain;
    }
    Fp* s_scratch = b.scratch;
    ef_eval_split<<<grid << EF_SPLIT_SHIFT, block, 0, stream>>>(check, const_cast<Fp*>(data), const_cast<Fp*>(accum),
                                              const_cast<Fp*>(mix), const_cast<Fp*>(out), domain, s_scratch);
    ef_combine<<<grid, block, 0, stream>>>(check, s_scratch, domain);'''
assert old in s; s=s.replace(old,new,1)
old='struct EfBufs { FpExt* pm = nullptr; FpExt* tab = nullptr; uint32_t* q = nullptr; FpExt* tabn = nullptr; };'
new='''// scratch: per-device LogUp partials of the split kernel (4 Fp per row), offset
// by a 4 KB pad (EF_SCRATCH_PAD) to avoid the DRAM/L2 aliasing with `check`
// seen in the harness at po2=21.
struct EfBufs {
  FpExt* pm = nullptr; FpExt* tab = nullptr; uint32_t* q = nullptr; FpExt* tabn = nullptr;
  char* scratch_base = nullptr; Fp* scratch = nullptr; uint32_t scratch_n = 0;
};'''
assert old in s; s=s.replace(old,new,1)
old='CUDA_OK(cudaFuncSetCacheConfig((const void*)ef_eval_check, cudaFuncCachePreferL1));'
assert old in s; s=s.replace(old,'CUDA_OK(cudaFuncSetCacheConfig((const void*)ef_eval_split, cudaFuncCachePreferL1));')
old='ef_eval_check<<<1, 1, 0, stream>>>(nullptr, nullptr, nullptr, nullptr, nullptr, 0);'
assert old in s; s=s.replace(old,'ef_eval_split<<<1, 1, 0, stream>>>(nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr);')
assert 'ef_eval_check<<<' not in s and 'ef_eval_check,' not in s
open(sys.argv[2],'w').write(s)
