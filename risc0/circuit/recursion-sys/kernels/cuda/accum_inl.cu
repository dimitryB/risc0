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

// parStepComputeAccum / parStepVerifyAccum with the generated step function
// force-inlined into the kernel (step_*_inl.cuh are step_compute_accum.cu /
// step_verify_accum.cu with the function made static __forceinline__, written by
// accum_inl_gen.py). CUDA: built in the unrestricted-register group (no -maxrregcount=96):
// cross-TU calls at 96 registers spilled ~14 KB per thread. HIP: in the amalgamated TU with
// ffi.cu, whose -amdgpu-early-inline-all=false leaves step_compute_accum / step_verify_accum
// called functions (their call frames spill to scratch); __forceinline__ still inlines these.
// On gfx11 held to 8 waves per SIMD (192 VGPRs, some spilled): unbounded they take 256 VGPRs
// (5 waves per SIMD), and the compute kernel ran ~12% slower on an RX 7900 XTX. On gfx12 (RX 9070
// XT) the unbounded kernels were the faster ones.
#include "cuda.h"
#include "step_compute_accum_inl.cuh"
#include "step_verify_accum_inl.cuh"

#if defined(__HIPCC__) && defined(__GFX11__)
#define ACCUM_INL_BOUNDS __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8)))
#else
#define ACCUM_INL_BOUNDS __launch_bounds__(256)
#endif

__global__ void ACCUM_INL_BOUNDS parStepComputeAccumInl(AccumContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < ctx->workCycles) {
    step_compute_accum_inl(ctx, ctx->totalCycles, cycle, ctx->buffers.ctrl, ctx->buffers.global,
                           ctx->buffers.data, ctx->buffers.mix, ctx->buffers.accum);
  }
}

__global__ void ACCUM_INL_BOUNDS parStepVerifyAccumInl(AccumContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < ctx->workCycles) {
    step_verify_accum_inl(ctx, ctx->totalCycles, cycle, ctx->buffers.ctrl, ctx->buffers.global,
                          ctx->buffers.data, ctx->buffers.mix, ctx->buffers.accum);
  }
}

#ifdef __HIPCC__
// The same kernels over the rows in perm's order (ffi.cu's recAccumPerm: the rows of each window
// grouped by step arm, so a wave takes fewer arms); the rows are independent.
__global__ void ACCUM_INL_BOUNDS parStepComputeAccumInlPerm(AccumContext* ctx, const uint32_t* perm) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < ctx->workCycles) {
    step_compute_accum_inl(ctx, ctx->totalCycles, perm[i], ctx->buffers.ctrl, ctx->buffers.global,
                           ctx->buffers.data, ctx->buffers.mix, ctx->buffers.accum);
  }
}

__global__ void ACCUM_INL_BOUNDS parStepVerifyAccumInlPerm(AccumContext* ctx, const uint32_t* perm) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < ctx->workCycles) {
    step_verify_accum_inl(ctx, ctx->totalCycles, perm[i], ctx->buffers.ctrl, ctx->buffers.global,
                          ctx->buffers.data, ctx->buffers.mix, ctx->buffers.accum);
  }
}

void risc0_recursion_launch_accum_inl_perm(int verify, unsigned grid, unsigned block, cudaStream_t stream,
                                           AccumContext* ctx, const uint32_t* perm) {
  if (verify) {
    parStepVerifyAccumInlPerm<<<grid, block, 0, stream>>>(ctx, perm);
  } else {
    parStepComputeAccumInlPerm<<<grid, block, 0, stream>>>(ctx, perm);
  }
}
#endif

void risc0_recursion_launch_accum_inl(int verify, unsigned grid, unsigned block, cudaStream_t stream,
                                      AccumContext* ctx) {
  if (verify) {
    parStepVerifyAccumInl<<<grid, block, 0, stream>>>(ctx);
  } else {
    parStepComputeAccumInl<<<grid, block, 0, stream>>>(ctx);
  }
}

// Warmup (off the GPU thread): loads both kernels and returns their largest per-thread local
// memory size, so the caller can pre-grow the context's stack reservation.
size_t risc0_recursion_accum_inl_warmup() {
  size_t need = 0;
  for (const void* f : {(const void*)parStepComputeAccumInl, (const void*)parStepVerifyAccumInl
#ifdef __HIPCC__
                        , (const void*)parStepComputeAccumInlPerm, (const void*)parStepVerifyAccumInlPerm
#endif
                       }) {
    cudaFuncAttributes attr;
    CUDA_OK(cudaFuncGetAttributes(&attr, f));
    need = attr.localSizeBytes > need ? attr.localSizeBytes : need;
  }
  return need;
}
