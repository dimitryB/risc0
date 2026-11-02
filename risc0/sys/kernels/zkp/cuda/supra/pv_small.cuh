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

// Latency-optimised Poseidon2 (BabyBear, width 24) for small Merkle levels and small row
// counts, where pv.cuh's thread-per-hash kernels are bound by one permutation's latency
// (~33k cycles) rather than by throughput. CUDA only (warp-size 32 shuffles).
//
//  * pvs::w — one warp per permutation, lane i holds cell i (lanes 24..31 hold 0). Partial rounds
//    are "lifted": the sum of cells 1..23 before partial round r is
//      T_r = a_r + sum_{q<r} c_{r-1-q} S_q,  a_r = sum_{i>=1} d_i^r x_i,  c_k = sum_{i>=1} d_i^k,
//    lane l accumulates T_l, so the serial chain per partial round is the sbox plus one product.
//    ~9.4k cycles per permutation.
//  * pvs::q — four lanes per permutation (lane l holds cells 4m + l), pv.cuh's partial-round
//    scheme. ~19k cycles, 4x the hashes per warp: best for mid-size levels.
//  Both reduce with min(r, r - p) (2 dependent ops vs ISETP + predicated IADD3) because they are
//  latency bound. Every value is canonical (or within pv.cuh's proven lazy bounds inside the
//  sbox), so the field values and the canonical outputs equal pv::permute's: bit-identical.
#pragma once

#include <cstdlib>
#include "pv.cuh"
#include "pv_small_consts.h"

namespace pvs {

using pv::M0;
using pv::mred;
using pv::P;

__device__ __forceinline__ uint32_t red1(uint32_t r) { return min(r, r - P); } // [0, 2^32) ok
__device__ __forceinline__ uint32_t addm(uint32_t a, uint32_t b) { return red1(a + b); }
__device__ __forceinline__ uint32_t mmul_lazy(uint32_t a, uint32_t b) { return mred((uint64_t)a * b); }
__device__ __forceinline__ uint32_t mmul(uint32_t a, uint32_t b) { return red1(mmul_lazy(a, b)); }
// x canonical -> x^7 canonical (bounds: pv.cuh)
__device__ __forceinline__ uint32_t sbox(uint32_t x) {
  uint32_t x2 = mmul_lazy(x, x);
  uint32_t x3 = mmul(x2, x);
  uint32_t x4 = mmul_lazy(x2, x2);
  return mmul(x3, x4);
}
__device__ __forceinline__ void m4(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  uint32_t t0 = addm(a, b), t1 = addm(c, d);
  uint32_t t2 = addm(addm(b, b), t1), t3 = addm(addm(d, d), t0);
  uint32_t t1_2 = addm(t1, t1), t0_2 = addm(t0, t0);
  uint32_t t4 = addm(addm(t1_2, t1_2), t3), t5 = addm(addm(t0_2, t0_2), t2);
  a = addm(t3, t5);
  b = t5;
  c = addm(t2, t4);
  d = t4;
}
// a * c (Montgomery), c constant with cm = c * M0 mod 2^32 (m does not wait for the product).
// a < 2p, c < p: t < 2p^2, canonical output.
__device__ __forceinline__ uint32_t cmul(uint32_t a, uint32_t c, uint32_t cm) {
  uint64_t t = (uint64_t)a * c;
  uint32_t m = a * cm;
  uint32_t r;
  asm("{\n\t.reg .u32 lo, hi, d;\n\tmov.b64 {lo,hi}, %1;\n\t"
      "mad.lo.cc.u32 d, %2, %3, lo;\n\tmadc.hi.u32 %0, %2, %3, hi;\n\t}"
      : "=r"(r)
      : "l"(t), "r"(m), "n"(P));
  return red1(r);
}
__device__ __forceinline__ uint32_t shfl(uint32_t v, int src) {
  return __shfl_sync(0xffffffffu, v, src);
}
__device__ __forceinline__ uint32_t shx(uint32_t v, int m) {
  return __shfl_xor_sync(0xffffffffu, v, m);
}

namespace w {

// External layer, lane-parallel: m4 inside each group of 4 lanes, then the column sums of the
// 6 groups gathered with parallel shuffles (lanes j, j+4, ..., j+20 only).
__device__ __forceinline__ uint32_t m_ext(uint32_t x, int lane) {
  const int base = lane & ~3, j = lane & 3;
  uint32_t a = shfl(x, base), b = shfl(x, base + 1), c = shfl(x, base + 2), d = shfl(x, base + 3);
  m4(a, b, c, d);
  uint32_t m = j == 0 ? a : (j == 1 ? b : (j == 2 ? c : d));
  uint32_t s0 = shfl(m, j), s1 = shfl(m, j + 4), s2 = shfl(m, j + 8), s3 = shfl(m, j + 12),
           s4 = shfl(m, j + 16), s5 = shfl(m, j + 20);
  return addm(m, addm(addm(addm(s0, s1), addm(s2, s3)), addm(s4, s5)));
}

// Per-lane constants, loaded once per kernel.
struct Lane {
  uint32_t rc[8], pw[21], cr[21], crm[21], dh, dhm, e0, e0m;
  __device__ __forceinline__ void load(int lane) {
#pragma unroll
    for (int r = 0; r < 8; r++)
      rc[r] = __ldg(&PVS_RCF[r][lane]);
#pragma unroll
    for (int r = 0; r < 21; r++) {
      pw[r] = __ldg(&PVS_PW[r][lane]);
      cr[r] = __ldg(&PVS_CR[r][lane]);
      crm[r] = __ldg(&PVS_CRM[r][lane]);
    }
    dh = __ldg(&PVS_DH[lane]);
    dhm = __ldg(&PVS_DHM[lane]);
    e0 = __ldg(&PVS_E0[0]);
    e0m = __ldg(&PVS_E0[1]);
  }
};

// x = this lane's cell (0 for lanes >= 24); returns the permuted cell (0 for lanes >= 24).
__device__ __forceinline__ uint32_t permute(uint32_t x, int lane, const Lane& K) {
  const bool act = lane < 24;
  x = m_ext(x, lane);
#pragma unroll
  for (int r = 0; r < 4; r++) {
    x = act ? sbox(addm(x, K.rc[r])) : 0u;
    x = m_ext(x, lane);
  }
  // a_r = sum_{i>=1} d_i^r x_i, reduce-scattered so that lane r holds a_r (pw = 0 on lanes 0, >= 24)
  uint32_t v[32];
#pragma unroll
  for (int r = 0; r < 32; r++)
    v[r] = r < 21 ? mmul(K.pw[r < 21 ? r : 0], x) : 0u;
#pragma unroll
  for (int m = 16, len = 16; m >= 1; m >>= 1, len >>= 1) {
    const bool hi = lane & m;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      if (j < len) {
        uint32_t send = hi ? v[j] : v[j + len];
        uint32_t keep = hi ? v[j + len] : v[j];
        v[j] = addm(keep, shx(send, m));
      }
    }
  }
  uint32_t acc = v[0];
  uint32_t Tr = shfl(acc, 0); // T_0 = sum_{i>=1} x_i
  uint32_t x0 = shfl(x, 0);
#pragma unroll
  for (int r = 0; r < 21; r++) {
    uint32_t y = sbox(addm(x0, PV_RCP[r]));
    uint32_t S = addm(y, Tr);                // sum of all cells after the sbox
    x = addm(cmul(x, K.dh, K.dhm), S);       // own cell: d_i x_i + S (lane 0 replaced below)
    x0 = addm(cmul(y, K.e0, K.e0m), Tr);     // (d_0 + 1) y + T_r
    acc = addm(acc, cmul(S, K.cr[r], K.crm[r]));
    Tr = shfl(acc, r + 1);                   // T_{r+1} (unused after r == 20)
  }
  x = lane == 0 ? x0 : (act ? x : 0u);
#pragma unroll
  for (int r = 4; r < 8; r++) {
    x = act ? sbox(addm(x, K.rc[r])) : 0u;
    x = m_ext(x, lane);
  }
  return act ? x : 0u;
}

// out[h] = H(in[2h], in[2h + 1]) for h < n; one warp per hash, 128-thread blocks.
__global__ __launch_bounds__(128) void fold_k(uint32_t* out, const uint32_t* in, uint32_t n) {
  const uint32_t h = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (h >= n)
    return; // whole warp
  Lane K;
  K.load(lane);
  uint32_t x = lane < 16 ? in[(size_t)h * 16 + lane] : 0u;
  x = permute(x, lane, K);
  if (lane < 8)
    out[(size_t)h * 8 + lane] = x;
}

// out[r] = hash of row r of the column-major dim_x x dim_y matrix m; one warp per row.
__global__ __launch_bounds__(128) void rows_k(uint32_t* out,
                                              const uint32_t* __restrict__ m,
                                              uint32_t dim_x,
                                              uint32_t dim_y) {
  const uint32_t row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= dim_x)
    return; // whole warp
  Lane K;
  K.load(lane);
  uint32_t x = 0;
  uint32_t i = 0;
  do {
    if (lane < 16)
      x = i + lane < dim_y ? __ldg(m + (size_t)(i + lane) * dim_x + row) : 0u;
    x = permute(x, lane, K);
    i += 16;
  } while (i < dim_y);
  if (lane < 8)
    out[(size_t)row * 8 + lane] = x;
}

} // namespace w

namespace q {

// value of cell 4m + k from lane k of this lane's group
__device__ __forceinline__ uint32_t grp(uint32_t v, int k) {
  return shfl(v, ((threadIdx.x & 31) & ~3) | k);
}
__device__ __forceinline__ void m_ext(uint32_t* y, int l) {
  uint32_t s;
#pragma unroll
  for (int m = 0; m < 6; m++) {
    uint32_t a = grp(y[m], 0), b = grp(y[m], 1), c = grp(y[m], 2), d = grp(y[m], 3);
    m4(a, b, c, d);
    uint32_t o = l == 0 ? a : (l == 1 ? b : (l == 2 ? c : d));
    y[m] = o;
    s = m ? addm(s, o) : o;
  }
#pragma unroll
  for (int m = 0; m < 6; m++)
    y[m] = addm(y[m], s);
}
__device__ __forceinline__ void full_round(uint32_t* y, int r, int l) {
#pragma unroll
  for (int m = 0; m < 6; m++)
    y[m] = sbox(addm(y[m], PV_RCF[r * 24 + 4 * m + l]));
  m_ext(y, l);
}
// pv.cuh's partial rounds (cells lazily in [0, 2p)), the 24-term sum split over the 4 lanes.
__device__ __forceinline__ void partial_rounds(uint32_t* y, int l) {
#pragma unroll 1
  for (int r = 0; r < 21; r++) {
    if (l == 0)
      y[0] = sbox(addm(red1(y[0]), PV_RCP[r]));
    uint64_t sum = 0;
#pragma unroll
    for (int m = 0; m < 6; m++)
      sum += y[m];
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    uint32_t st = mred(sum);
    uint32_t w = mred((uint64_t)st * 0x12f37bfbu);
#pragma unroll
    for (int m = 0; m < 6; m++)
      y[m] = mred((uint64_t)PV_DH[4 * m + l] * y[m] + w);
  }
#pragma unroll
  for (int m = 0; m < 6; m++)
    y[m] = red1(y[m]);
}
__device__ __forceinline__ void permute(uint32_t* y, int l) {
  m_ext(y, l);
#pragma unroll 1
  for (int r = 0; r < 4; r++)
    full_round(y, r, l);
  partial_rounds(y, l);
#pragma unroll 1
  for (int r = 4; r < 8; r++)
    full_round(y, r, l);
}
// out[i] = H(in[2i], in[2i + 1]); one 4-lane group per output. Launch with a multiple of 32
// threads per block (whole warps stay converged for the shuffles).
__global__ __launch_bounds__(128) void fold_k(uint32_t* out, const uint32_t* in, uint32_t n) {
  const uint32_t g = (blockIdx.x * blockDim.x + threadIdx.x) >> 2;
  const int l = threadIdx.x & 3;
  const bool act = g < n;
  const uint32_t gg = act ? g : 0;
  uint32_t y[6];
#pragma unroll
  for (int m = 0; m < 4; m++)
    y[m] = in[(size_t)gg * 16 + 4 * m + l];
  y[4] = 0;
  y[5] = 0;
  permute(y, l);
  if (act) {
    out[(size_t)gg * 8 + l] = y[0];
    out[(size_t)gg * 8 + 4 + l] = y[1];
  }
}
// out[r] = hash of row r (column-major dim_x x dim_y); one 4-lane group per row.
__global__ __launch_bounds__(128) void rows_k(uint32_t* out,
                                              const uint32_t* __restrict__ mat,
                                              uint32_t dim_x,
                                              uint32_t dim_y) {
  const uint32_t g = (blockIdx.x * blockDim.x + threadIdx.x) >> 2;
  const int l = threadIdx.x & 3;
  const bool act = g < dim_x;
  const uint32_t gg = act ? g : 0;
  uint32_t y[6] = {0, 0, 0, 0, 0, 0};
  uint32_t i = 0;
  do {
#pragma unroll
    for (int m = 0; m < 4; m++) {
      uint32_t c = i + 4 * m + l;
      y[m] = c < dim_y ? __ldg(mat + (size_t)c * dim_x + gg) : 0u;
    }
    permute(y, l);
    i += 16;
  } while (i < dim_y);
  if (act) {
    out[(size_t)gg * 8 + l] = y[0];
    out[(size_t)gg * 8 + 4 + l] = y[1];
  }
}

} // namespace q

// Measured crossovers (RTX 5090, merkle-small bench_lvl / bench_rows): warp kernels for
// n <= 2048, quad kernels for n <= 8192, pv.cuh's thread kernels above.
constexpr uint32_t W_MAX = 2048, Q_MAX = 8192;

// RISC0_CUDA_PV_SMALL=0 keeps pv.cuh's thread kernels for every size.
inline bool enabled() {
  static const bool on = [] { const char* e = std::getenv("RISC0_CUDA_PV_SMALL"); return !(e && e[0] == '0'); }();
  return on;
}

// Launch helpers (stream s); return false when n is too large for the latency kernels.
inline bool fold(cudaStream_t s, uint32_t* out, const uint32_t* in, uint32_t n) {
  if (!enabled()) return false;
  if (n <= W_MAX) {
    w::fold_k<<<(n * 32 + 127) / 128, 128, 0, s>>>(out, in, n);
    return true;
  }
  if (n <= Q_MAX) {
    q::fold_k<<<(n * 4 + 127) / 128, 128, 0, s>>>(out, in, n);
    return true;
  }
  return false;
}
inline bool rows(cudaStream_t s, uint32_t* out, const uint32_t* m, uint32_t dim_x, uint32_t dim_y) {
  if (!enabled()) return false;
  if (dim_x <= W_MAX) {
    w::rows_k<<<(dim_x * 32 + 127) / 128, 128, 0, s>>>(out, m, dim_x, dim_y);
    return true;
  }
  if (dim_x <= Q_MAX) {
    q::rows_k<<<(dim_x * 4 + 127) / 128, 128, 0, s>>>(out, m, dim_x, dim_y);
    return true;
  }
  return false;
}

} // namespace pvs
