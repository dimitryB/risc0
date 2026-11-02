// Copyright 2025 RISC Zero, Inc.
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

#pragma once

// Raw device allocations made outside the Rust HAL (grow-only kernel scratch: nttq's T2, the
// rv32im device preflight slots, the accum sort keys, CUB temp storage).
//
// The HAL caches freed device buffers in a per-thread pool (up to total VRAM - 8 GB) and drains
// it and retries when one of its own allocations fails. A raw cudaMalloc cannot see that pool, so
// when the pool and the current working set fill the device, it fails although most of the
// memory is only cached. risc0_cuda_malloc retries once after calling the hook the HAL registers
// (risc0_zkp_cuda_set_oom_hook): it drains the calling thread's pool, which is the proving
// thread's pool for every caller here, and trims the stream-ordered mempool.
//
// Include after "cuda.h" (or the HIP runtime).

#include <cstddef>

// C++17 inline variable: a single instance in the process, shared by every library that includes
// this header.
inline void (*risc0_cuda_oom_hook)() = nullptr;

inline cudaError_t risc0_cuda_malloc_raw(void** ptr, size_t bytes) {
#ifdef __HIPCC__
  constexpr cudaError_t kOutOfMemory = hipErrorOutOfMemory;
#else
  constexpr cudaError_t kOutOfMemory = cudaErrorMemoryAllocation;
#endif
  cudaError_t err = cudaMalloc(ptr, bytes);
  if (err == kOutOfMemory && risc0_cuda_oom_hook) {
    (void)cudaGetLastError();
    risc0_cuda_oom_hook();
    err = cudaMalloc(ptr, bytes);
  }
  return err;
}

template <typename T> inline cudaError_t risc0_cuda_malloc(T** ptr, size_t bytes) {
  return risc0_cuda_malloc_raw(reinterpret_cast<void**>(ptr), bytes);
}
