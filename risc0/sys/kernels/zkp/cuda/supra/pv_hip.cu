// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ROCm only: pv.cuh's Poseidon2 kernels in their own translation unit, which build.rs compiles
// with LLVM's ILP-first machine scheduler (-mllvm -amdgpu-sched-strategy=max-ilp). On RDNA3/4 the
// permutation is VALU-issue bound; the default occupancy-first schedule (91 VGPRs, 16 waves/SIMD)
// leaves 2-5% of the issue slots to dependency stalls, the ILP schedule (~150 VGPRs, 10 waves)
// recovers them, for small (latency-bound) launches as well. The scheduler choice is per TU, hence
// the separate file; the include sits in its own namespace so its kernels and constants do not
// clash with api.cu's instances (still used with RISC0_HIP_PV_TU=0).

#include <cstdint>

#include <hip/hip_runtime.h>

namespace pv_tu {
#include "pv.cuh"
} // namespace pv_tu

void pv_hip_rows(uint4* out, const uint32_t* m, uint32_t count, uint32_t col_size, hipStream_t s, bool v0) {
  uint32_t bs = count < 128 ? count : 128;
  if (v0)
    pv_tu::pv::rows0_k<<<(count + bs - 1) / bs, bs, 0, s>>>(out, m, count, col_size);
  else
    pv_tu::pv::rows_k<<<(count + bs - 1) / bs, bs, 0, s>>>(out, m, count, col_size);
}

void pv_hip_fold(uint4* out, const uint4* in, uint32_t n, hipStream_t s) {
  uint32_t bs = n < 128 ? n : 128;
  pv_tu::pv::fold_k<<<(n + bs - 1) / bs, bs, 0, s>>>(out, in, n);
}

void pv_hip_fold_tail(uint4* nodes, uint32_t n0, hipStream_t s) {
  pv_tu::pv::fold_tail_k<<<1, n0, 0, s>>>(nodes, n0);
}

// 4 lanes per permutation (pv.cuh fold_q4_k / fold_tail_q4_k): n <= 128 for the tail.
void pv_hip_fold_q4(uint4* out, const uint4* in, uint32_t n, hipStream_t s) {
  uint32_t t = 4 * n, bs = t < 256 ? t : 256;
  pv_tu::pv::fold_q4_k<<<(t + bs - 1) / bs, bs, 0, s>>>(out, in, n);
}

void pv_hip_fold_tail_q4(uint4* nodes, uint32_t n0, hipStream_t s) {
  pv_tu::pv::fold_tail_q4_k<<<1, 4 * n0, 0, s>>>(nodes, n0);
}
