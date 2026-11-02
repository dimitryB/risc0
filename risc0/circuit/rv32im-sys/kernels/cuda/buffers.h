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

#include "fp.h"

namespace risc0::circuit::rv32im_v2::cuda {

struct Buffer {
  Fp* buf;
  size_t rows;
  size_t cols;
  bool checked;  // retained for FFI layout compatibility

#if defined(__HIPCC__) && !defined(RISC0_HIP_ROWS_PRIVATE)
  // HIP/AMDGPU: compiler barrier prevents reordering/coalescing of cross-row
  // reads/writes. The SHA2 precompile does back-references at distance 68;
  // without barriers hip-clang at -O2+ can reorder loads that produce a
  // corrupt witness trace, resulting in invalid segment proofs.
#if defined(RISC0_HIP_GLOBAL_BUFS) && defined(__HIP_DEVICE_COMPILE__)
  // Same accesses in the same order (barriers kept), through global rather than FLAT pointers
  // (a FLAT load also waits on the LDS/scalar counter): witgen_tu.cu on ROCm.
  __device__ void set(size_t row, size_t col, Fp val) {
    ((__attribute__((address_space(1))) uint32_t*)(uint32_t*)buf)[col * rows + row] = val.asRaw();
    asm volatile("" ::: "memory");
  }
  __device__ Fp get(size_t row, size_t col) {
    asm volatile("" ::: "memory");
    Fp v = Fp::fromRaw(
        ((const __attribute__((address_space(1))) uint32_t*)(const uint32_t*)buf)[col * rows + row]);
    asm volatile("" ::: "memory");
    return v;
  }
#else
  __device__ void set(size_t row, size_t col, Fp val) {
    buf[col * rows + row] = val;
    asm volatile("" ::: "memory");
  }
  __device__ Fp get(size_t row, size_t col) {
    asm volatile("" ::: "memory");
    Fp v = buf[col * rows + row];
    asm volatile("" ::: "memory");
    return v;
  }
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  // RISC0_HIP_ROWS_PRIVATE (no row is written by another thread of the launch): no barriers,
  // and global rather than FLAT accesses (the generic pointer made every load FLAT, which also
  // waits on the LDS counter).
  __device__ void set(size_t row, size_t col, Fp val) {
    ((__attribute__((address_space(1))) uint32_t*)(uint32_t*)buf)[col * rows + row] = val.asRaw();
  }
  __device__ Fp get(size_t row, size_t col) {
    return Fp::fromRaw(
        ((const __attribute__((address_space(1))) uint32_t*)(const uint32_t*)buf)[col * rows + row]);
  }
#else
  __device__ void set(size_t row, size_t col, Fp val) {
    buf[col * rows + row] = val;
  }
  __device__ Fp get(size_t row, size_t col) {
    return buf[col * rows + row];
  }
#endif
};

} // namespace risc0::circuit::rv32im_v2::cuda
