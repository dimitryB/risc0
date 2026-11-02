// Copyright 2024 RISC Zero, Inc.
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

#include "cuda.h"
#include <cstdlib>
#include "supra/fp.h"

extern __constant__ FpExt poly_mix[158];

namespace risc0::circuit::recursion::cuda {

// eval_check_split.cu: the same polynomial in several smaller kernels (RISC0_CUDA_REC_KERNELS=0
// restores the single kernel; the same switch controls the inlined accum kernels in
// ffi.cu).
void eval_check_split(cudaStream_t stream, Fp* check, const Fp* ctrl, const Fp* data, const Fp* accum,
                      const Fp* mix, const Fp* out, Fp rou, uint32_t po2, uint32_t domain,
                      const void* poly_mix_host);
static bool use_split() {
  static const bool on = [] { const char* e = getenv("RISC0_CUDA_REC_KERNELS"); return !(e && e[0] == '0'); }();
  return on;
}

__device__ FpExt poly_fp(uint32_t idx,
                         uint32_t size,
                         const Fp* ctrl,
                         const Fp* out,
                         const Fp* data,
                         const Fp* mix,
                         const Fp* accum);

__launch_bounds__(256, 1)
__global__ void eval_check(Fp* check,
                           const Fp* ctrl,
                           const Fp* data,
                           const Fp* accum,
                           const Fp* mix,
                           const Fp* out,
                           const Fp rou,
                           uint32_t po2,
                           uint32_t domain) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < domain) {
    FpExt tot = poly_fp(cycle, domain, ctrl, out, data, mix, accum);
#ifdef __HIPCC__
    Fp x = rou ^ (unsigned)cycle;
    Fp y = (Fp(3) * x) ^ (unsigned)(1 << po2);
#else
    Fp x = pow(rou, cycle);
    Fp y = pow(Fp(3) * x, 1 << po2);
#endif
    FpExt ret = tot * inv(y - Fp(1));
    check[domain * 0 + cycle] = ret[0];
    check[domain * 1 + cycle] = ret[1];
    check[domain * 2 + cycle] = ret[2];
    check[domain * 3 + cycle] = ret[3];
  }
}

} // namespace risc0::circuit::recursion::cuda

extern "C" const char* risc0_circuit_recursion_cuda_eval_check(Fp* check,
                                                               const Fp* ctrl,
                                                               const Fp* data,
                                                               const Fp* accum,
                                                               const Fp* mix,
                                                               const Fp* out,
                                                               const Fp& rou,
                                                               uint32_t po2,
                                                               uint32_t domain,
                                                               const FpExt* poly_mix_pows) {

  try {
    cudaStream_t stream = getPersistentStream();
    LaunchConfig cfg = getSimpleConfig(domain);
    if (risc0::circuit::recursion::cuda::use_split()) {
      // Uploads poly_mix to its own module's symbol (this one's is not read).
      risc0::circuit::recursion::cuda::eval_check_split(stream, check, ctrl, data, accum, mix, out, rou,
                                                        po2, domain, poly_mix_pows);
      return nullptr;
    }
#ifdef __HIPCC__
    {
      void* dev_ptr = nullptr;
      hipGetSymbolAddress(&dev_ptr, HIP_SYMBOL(poly_mix));
      hipMemcpyAsync(dev_ptr, poly_mix_pows, sizeof(poly_mix), hipMemcpyHostToDevice, stream);
    }
#else
    // Async memcpy on the same stream avoids a host-blocking sync that
    // serializes the persistent stream against the host. Stage into a
    // persistent host buffer so the source remains valid across the async
    // copy (rv32im uses the same pattern).
    {
      static char poly_mix_staging[sizeof(poly_mix)];
      memcpy(poly_mix_staging, poly_mix_pows, sizeof(poly_mix));
      cudaMemcpyToSymbolAsync(poly_mix, poly_mix_staging, sizeof(poly_mix),
                              0, cudaMemcpyHostToDevice, stream);
    }
#endif
    risc0::circuit::recursion::cuda::eval_check<<<cfg.grid, cfg.block, 0, stream>>>(
        check, ctrl, data, accum, mix, out, rou, po2, domain);
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Load the recursion eval_check kernel off the GPU thread (see ffi.cu warmup).
extern "C" const char* risc0_circuit_recursion_cuda_warmup_eval_check() {
#ifndef __HIPCC__
  cudaFuncAttributes attr;
  cudaError_t err = cudaFuncGetAttributes(&attr, risc0::circuit::recursion::cuda::eval_check);
  if (err != cudaSuccess) {
    return strdup(cudaGetErrorString(err));
  }
#endif
  return nullptr;
}
