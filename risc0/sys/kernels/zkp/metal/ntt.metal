// Copyright 2022 Risc0, Inc.
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

#include "fp.h"
#include "fpext.h"

using namespace metal;

kernel void multi_bit_reverse(device Fp* io,
                              const device uint32_t& nBits,
                              uint gid [[thread_position_in_grid]]) {
  uint32_t rowSize = 1 << nBits;
  uint32_t idx = gid & (rowSize - 1);
  uint32_t s = gid >> nBits;
  uint32_t ridx = reverse_bits(idx) >> (32 - nBits);
  if (idx < ridx) {
    size_t idx1 = s * rowSize + idx;
    size_t idx2 = s * rowSize + ridx;
    Fp tmp = io[idx1];
    io[idx1] = io[idx2];
    io[idx2] = tmp;
  }
}

kernel void multi_poly_eval(device FpExt* out,
                            const device Fp* coeffs,
                            const device uint32_t* which,
                            const device FpExt* xs,
                            const device uint32_t& deg,
                            uint3 blockIdx [[threadgroup_position_in_grid]],
                            uint3 threadIdx [[thread_position_in_threadgroup]]) {
  device const Fp* cur_poly = coeffs + which[blockIdx.x] * deg;
  FpExt x = xs[blockIdx.x];
  FpExt stepx = pow(x, 256);
  FpExt powx = pow(x, threadIdx.x);
  FpExt tot;
  for (size_t i = threadIdx.x; i < deg; i += 256) {
    tot += powx * cur_poly[i];
    powx *= stepx;
  }
  threadgroup uint32_t totsBuf[256 * 4];
  threadgroup FpExt* tots = reinterpret_cast<threadgroup FpExt*>(totsBuf);
  tots[threadIdx.x] = tot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  unsigned cur = 256;
  while (cur) {
    cur /= 2;
    if (threadIdx.x < cur) {
      tots[threadIdx.x] = FpExt(tots[threadIdx.x]) + FpExt(tots[threadIdx.x + cur]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (threadIdx.x == 0) {
    out[blockIdx.x] = tots[0];
  }
}

kernel void batch_expand(device Fp* out,
                         const device Fp* in,
                         const device uint32_t& polyCount,
                         const device uint32_t& outSize,
                         const device uint32_t& inSize,
                         const device uint32_t& expandBits,
                         uint gid [[thread_position_in_grid]]) {
  for (uint32_t i = 0; i < polyCount; i++) {
    out[i * outSize + gid] = in[i * inSize + (gid >> expandBits)];
  }
}

kernel void gather_sample(device Fp* dst,
                          const device Fp* src,
                          const device uint32_t& idx,
                          const device uint32_t& stride,
                          uint gid [[thread_position_in_grid]]) {
  dst[gid] = src[gid * stride + idx];
}

kernel void multi_ntt_fwd_step(device Fp* io,
                               const device Fp* twiddles,
                               const device uint32_t& nBits,
                               const device uint32_t& sBits,
                               const device uint32_t& cSize,
                               uint3 gridDim [[threadgroups_per_grid]],
                               uint3 blockDim [[threads_per_threadgroup]],
                               uint3 blockIdx [[threadgroup_position_in_grid]],
                               uint3 threadIdx [[thread_position_in_threadgroup]]) {
  uint32_t gSize = 1 << (nBits - sBits);
  uint32_t sSize = 1 << (sBits - 1);
  uint32_t nSize = 1 << nBits;
  uint32_t sOff = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t sStep = blockDim.x * gridDim.x;
  uint32_t gOff = threadIdx.y + blockIdx.y * blockDim.y;
  uint32_t gStep = blockDim.y * gridDim.y;
  uint32_t cOff = threadIdx.z + blockIdx.z * blockDim.z;
  uint32_t cStep = blockDim.z * gridDim.z;

  const device Fp* stage = twiddles + sSize - 1;
  for (uint32_t s = sOff; s < sSize; s += sStep) {
    Fp curMul = stage[s];
    for (uint32_t g = gOff; g < gSize; g += gStep) {
      for (uint32_t c = cOff; c < cSize; c += cStep) {
        Fp a = io[c * nSize + g * 2 * sSize + s];
        Fp b = io[c * nSize + g * 2 * sSize + s + sSize];
        b *= curMul;
        io[c * nSize + g * 2 * sSize + s] = a + b;
        io[c * nSize + g * 2 * sSize + s + sSize] = a - b;
      }
    }
  }
}

kernel void multi_ntt_rev_step(device Fp* io,
                               const device Fp* twiddles,
                               const device uint32_t& nBits,
                               const device uint32_t& sBits,
                               const device uint32_t& cSize,
                               uint3 gridDim [[threadgroups_per_grid]],
                               uint3 blockDim [[threads_per_threadgroup]],
                               uint3 blockIdx [[threadgroup_position_in_grid]],
                               uint3 threadIdx [[thread_position_in_threadgroup]]) {
  uint32_t gSize = 1 << (nBits - sBits);
  uint32_t sSize = 1 << (sBits - 1);
  uint32_t nSize = 1 << nBits;
  uint32_t sOff = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t sStep = blockDim.x * gridDim.x;
  uint32_t gOff = threadIdx.y + blockIdx.y * blockDim.y;
  uint32_t gStep = blockDim.y * gridDim.y;
  uint32_t cOff = threadIdx.z + blockIdx.z * blockDim.z;
  uint32_t cStep = blockDim.z * gridDim.z;

  const device Fp* stage = twiddles + sSize - 1;
  for (uint32_t s = sOff; s < sSize; s += sStep) {
    Fp curMul = stage[s];
    for (uint32_t g = gOff; g < gSize; g += gStep) {
      for (uint32_t c = cOff; c < cSize; c += cStep) {
        Fp a = io[c * nSize + g * 2 * sSize + s];
        Fp b = io[c * nSize + g * 2 * sSize + s + sSize];
        io[c * nSize + g * 2 * sSize + s] = a + b;
        io[c * nSize + g * 2 * sSize + s + sSize] = (a - b) * curMul;
      }
    }
  }
}

// Keep the low twelve radix-2 stages (radix-4/8 groups within them) in a
// 16 KiB threadgroup tile. Every stage is separated by a group barrier; only
// the first load and final store touch device memory.
kernel void multi_ntt_fwd_tile(device Fp* io,
                               const device Fp* twiddles,
                               const device uint32_t& nBits,
                               const device uint32_t& firstBits,
                               uint2 group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint32_t words[4096];
  threadgroup Fp* tile = reinterpret_cast<threadgroup Fp*>(words);
  uint32_t base = group.y * (1u << nBits) + group.x * 4096;
  for (uint32_t i = lane; i < 4096; i += 256) tile[i] = io[base + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint32_t bits = firstBits; bits <= 12; bits++) {
    uint32_t span = 1u << (bits - 1);
    const device Fp* stage = twiddles + span - 1;
    for (uint32_t i = lane; i < 4096; i += 256) {
      if ((i & span) == 0) {
        Fp a = tile[i];
        Fp b = tile[i + span];
        Fp twiddle = stage[i & (span - 1)];
        b *= twiddle;
        tile[i] = a + b;
        tile[i + span] = a - b;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint32_t i = lane; i < 4096; i += 256) io[base + i] = tile[i];
}

kernel void multi_ntt_rev_tile(device Fp* io,
                               const device Fp* twiddles,
                               const device uint32_t& nBits,
                               uint2 group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint32_t words[4096];
  threadgroup Fp* tile = reinterpret_cast<threadgroup Fp*>(words);
  uint32_t base = group.y * (1u << nBits) + group.x * 4096;
  for (uint32_t i = lane; i < 4096; i += 256) tile[i] = io[base + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int bits = 12; bits >= 1; bits--) {
    uint32_t span = 1u << (bits - 1);
    const device Fp* stage = twiddles + span - 1;
    for (uint32_t i = lane; i < 4096; i += 256) {
      if ((i & span) == 0) {
        Fp a = tile[i];
        Fp b = tile[i + span];
        Fp twiddle = stage[i & (span - 1)];
        tile[i] = a + b;
        tile[i + span] = (a - b) * twiddle;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint32_t i = lane; i < 4096; i += 256) io[base + i] = tile[i];
}

// Final inverse pass for a commit. Read the scratch evaluations, finish the
// low stages in threadgroup memory, then normalize, zk-shift and write the
// coefficients in natural order. The bit reversal is in this final store.
kernel void multi_ntt_rev_tile_commit(device Fp* coeffs,
                                      const device Fp* scratch,
                                      const device Fp* twiddles,
                                      const device Fp* scales,
                                      const device uint32_t& nBits,
                                      uint2 group [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint32_t words[4096];
  threadgroup Fp* tile = reinterpret_cast<threadgroup Fp*>(words);
  uint32_t row_base = group.x * 4096;
  uint32_t column_base = group.y * (1u << nBits);
  for (uint32_t i = lane; i < 4096; i += 256) {
    tile[i] = scratch[column_base + row_base + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int bits = 12; bits >= 1; bits--) {
    uint32_t span = 1u << (bits - 1);
    const device Fp* stage = twiddles + span - 1;
    for (uint32_t i = lane; i < 4096; i += 256) {
      if ((i & span) == 0) {
        Fp a = tile[i];
        Fp b = tile[i + span];
        Fp twiddle = stage[i & (span - 1)];
        tile[i] = a + b;
        tile[i + span] = (a - b) * twiddle;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint32_t i = lane; i < 4096; i += 256) {
    uint32_t natural = reverse_bits(row_base + i) >> (32 - nBits);
    Fp value = tile[i];
    Fp scale = scales[natural];
    coeffs[column_base + natural] = value * scale;
  }
}

// The fused inverse writes natural coefficients; the LDE DIT needs their
// bit-reversed order, expanded by the blowup factor. Each thread reads one
// coefficient and writes its adjacent copies, across all columns in one grid.
kernel void batch_expand_bitrev(device Fp* out,
                                const device Fp* coeffs,
                                const device uint32_t& inBits,
                                const device uint32_t& expandBits,
                                uint2 group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]]) {
  uint32_t row = group.x * 256 + lane;
  uint32_t inSize = 1u << inBits;
  if (row >= inSize) return;
  uint32_t outSize = inSize << expandBits;
  uint32_t natural = reverse_bits(row) >> (32 - inBits);
  Fp value = coeffs[group.y * inSize + natural];
  uint32_t base = group.y * outSize + (row << expandBits);
  for (uint32_t j = 0; j < (1u << expandBits); j++) out[base + j] = value;
}
