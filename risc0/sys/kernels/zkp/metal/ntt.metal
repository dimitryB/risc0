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

// Two adjacent DIT stages form a radix-4 butterfly in registers. One thread
// owns each quadruple, so only one threadgroup barrier is needed per pair.
inline void ntt_fwd_radix4(threadgroup Fp* tile,
                           const device Fp* twiddles,
                           uint32_t bits,
                           uint32_t lane) {
  uint32_t quarter = 1u << (bits - 1);
  const device Fp* first = twiddles + quarter - 1;
  const device Fp* second = twiddles + 2 * quarter - 1;
  for (uint32_t j = lane; j < 1024; j += 256) {
    uint32_t s = j & (quarter - 1);
    uint32_t idx = (j / quarter) * 4 * quarter + s;
    Fp a0 = tile[idx], a1 = tile[idx + quarter];
    Fp a2 = tile[idx + 2 * quarter], a3 = tile[idx + 3 * quarter];
    Fp w0 = first[s], w1 = second[s], w2 = second[s + quarter];
    Fp t0 = a1 * w0, t1 = a3 * w0;
    Fp u0 = a0 + t0, v0 = a0 - t0;
    Fp u1 = a2 + t1, v1 = a2 - t1;
    Fp p0 = u1 * w1, p1 = v1 * w2;
    tile[idx] = u0 + p0;
    tile[idx + quarter] = v0 + p1;
    tile[idx + 2 * quarter] = u0 - p0;
    tile[idx + 3 * quarter] = v0 - p1;
  }
}

inline void ntt_rev_radix4(threadgroup Fp* tile,
                           const device Fp* twiddles,
                           uint32_t bits,
                           uint32_t lane) {
  uint32_t quarter = 1u << (bits - 2);
  const device Fp* first = twiddles + 2 * quarter - 1;
  const device Fp* second = twiddles + quarter - 1;
  for (uint32_t j = lane; j < 1024; j += 256) {
    uint32_t s = j & (quarter - 1);
    uint32_t idx = (j / quarter) * 4 * quarter + s;
    Fp a0 = tile[idx], a1 = tile[idx + quarter];
    Fp a2 = tile[idx + 2 * quarter], a3 = tile[idx + 3 * quarter];
    Fp w0 = first[s], w1 = first[s + quarter], w2 = second[s];
    Fp u0 = a0 + a2, u1 = a1 + a3;
    Fp v0 = (a0 - a2) * w0, v1 = (a1 - a3) * w1;
    tile[idx] = u0 + u1;
    tile[idx + quarter] = (u0 - u1) * w2;
    tile[idx + 2 * quarter] = v0 + v1;
    tile[idx + 3 * quarter] = (v0 - v1) * w2;
  }
}

// Keep the low twelve stages in a 16 KiB threadgroup tile. Only the first
// load and final store touch device memory.
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
  uint32_t bits = firstBits;
  for (; bits + 1 <= 12; bits += 2) {
    ntt_fwd_radix4(tile, twiddles, bits, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (bits == 12) {
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
  for (uint32_t bits = 12; bits >= 2; bits -= 2) {
    ntt_rev_radix4(tile, twiddles, bits, lane);
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
  for (uint32_t bits = 12; bits >= 2; bits -= 2) {
    ntt_rev_radix4(tile, twiddles, bits, lane);
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
