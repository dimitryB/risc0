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

constexpr size_t kFriFold = 16;

/// Compute `ceil(log_2(in))`, i.e. find the smallest value `out` such that `2^out >= in`.
__device__ inline constexpr size_t log2Ceil(size_t in) {
  size_t r = 0;
  while ((1 << r) < in) {
    r++;
  }
  return r;
}

__global__ void batch_bit_reverse(Fp* io, const uint32_t nBits, const uint32_t count) {
  uint totIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (totIdx < count) {
    uint32_t rowSize = 1 << nBits;
    uint32_t idx = totIdx & (rowSize - 1);
    uint32_t s = totIdx >> nBits;
    uint32_t ridx = __brev(idx) >> (32 - nBits);
    if (idx < ridx) {
      size_t idx1 = s * rowSize + idx;
      size_t idx2 = s * rowSize + ridx;
      Fp tmp = io[idx1];
      io[idx1] = io[idx2];
      io[idx2] = tmp;
    }
  }
}

__global__ void batch_evaluate_any(
    FpExt* out, const Fp* coeffs, const uint32_t* which, const FpExt* xs, const uint32_t deg) {
  const Fp* cur_poly = coeffs + which[blockIdx.x] * deg;
  FpExt x = xs[blockIdx.x];
  FpExt stepx = pow(x, blockDim.x);
  // Horner's method: evaluate each thread's strided subsequence from high to low degree.
  // Each iteration: tot *= stepx (FpExt*FpExt = 16 Fp muls) + tot += coeff (1 Fp add).
  // Original was 20 Fp muls/iter (16 for powx*=stepx + 4 for powx*coeff). Saves 20%.
  uint32_t K = (deg + blockDim.x - 1) / blockDim.x;
  FpExt tot;
  for (int k = K - 1; k >= 0; k--) {
    tot *= stepx;
    size_t i = threadIdx.x + (size_t)k * blockDim.x;
    if (i < deg)
      tot += cur_poly[i];
  }
  tot *= pow(x, threadIdx.x);
  extern __shared__ uint32_t totsBuf[];
  FpExt* tots = reinterpret_cast<FpExt*>(totsBuf);
  tots[threadIdx.x] = tot;
  __syncthreads();
  unsigned cur = blockDim.x;
  while (cur) {
    cur /= 2;
    if (threadIdx.x < cur) {
      tots[threadIdx.x] = FpExt(tots[threadIdx.x]) + FpExt(tots[threadIdx.x + cur]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[blockIdx.x] = tots[0];
  }
}

__global__ void fri_fold(Fp* __restrict__ out,
                         const Fp* __restrict__ in_ptr,
                         const FpExt* __restrict__ mix,
                         const uint32_t count) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    // Cache mix value in register to avoid repeated global memory reads
    const FpExt mixVal = *mix;
    FpExt tot;
    FpExt curMix(1);
    #pragma unroll
    for (uint32_t i = 0; i < kFriFold; i++) {
      size_t rev_i = __brev(i) >> (32 - log2Ceil(kFriFold));
      size_t rev_idx = rev_i * count + idx;
      FpExt factor(in_ptr[0 * count * kFriFold + rev_idx],
                   in_ptr[1 * count * kFriFold + rev_idx],
                   in_ptr[2 * count * kFriFold + rev_idx],
                   in_ptr[3 * count * kFriFold + rev_idx]);
      tot += curMix * factor;
      curMix *= mixVal;
    }
    out[count * 0 + idx] = tot.elems[0];
    out[count * 1 + idx] = tot.elems[1];
    out[count * 2 + idx] = tot.elems[2];
    out[count * 3 + idx] = tot.elems[3];
  }
}

__global__ void gather_sample(
    Fp* dst, const Fp* src, const uint32_t idx, const uint32_t size, const uint32_t stride) {
  uint gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    dst[gid] = src[gid * stride + idx];
  }
}

__global__ void scatter(Fp* into,
                        const uint32_t* index,
                        const uint32_t* offsets,
                        const Fp* values,
                        const uint32_t count) {
  uint gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < count) {
    for (uint32_t idx = index[gid]; idx < index[gid + 1]; idx++) {
      into[offsets[idx]] = values[idx];
    }
  }
}

// GPU-side u32→bits decomposition. Each thread reads one packed (row, base_col, value)
// triple and writes 32 bit values into the data buffer.
__global__ void scatter_bits(Fp* into,
                             const uint32_t* data, // packed [row, base_col, value] triples
                             const uint32_t cycles,
                             const uint32_t count) {
  uint gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < count) {
    uint32_t row = data[gid * 3];
    uint32_t base = data[gid * 3 + 1];
    uint32_t val = data[gid * 3 + 2];
    for (uint32_t i = 0; i < 32; i++) {
      into[(base + i) * cycles + row] = Fp((val >> i) & 1);
    }
  }
}

__global__ void gather_digests(
    uint32_t* dst, const uint32_t* src, const uint32_t* indices, uint32_t count) {
  uint gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < count) {
    uint32_t src_idx = indices[gid];
    // Copy 8 uint32_t words (32 bytes = 1 Digest)
    for (int w = 0; w < 8; w++) {
      dst[gid * 8 + w] = src[src_idx * 8 + w];
    }
  }
}

__global__ void mix_poly_coeffs(FpExt* out,
                                const Fp* in,
                                const uint32_t* combos,
                                const FpExt* mixStart,
                                const FpExt* mix,
                                const uint32_t inputSize,
                                const uint32_t count) {
  // Precompute mix powers in shared memory: thread 0 computes all powers once,
  // then all threads read from shared. Eliminates per-thread FpExt*FpExt multiply
  // (16 Fp muls/iter) — each thread only does FpExt*Fp (4 muls/iter).
  extern __shared__ FpExt mixPows[];
  if (threadIdx.x == 0) {
    FpExt cur = *mixStart;
    FpExt m = *mix;
    for (uint32_t i = 0; i < inputSize; i++) {
      mixPows[i] = cur;
      cur *= m;
    }
  }
  __syncthreads();
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    for (size_t i = 0; i < inputSize; i++) {
      size_t id = combos[i];
      out[count * id + idx] += mixPows[i] * in[count * i + idx];
    }
  }
}

// ---- eval_u v2 (its reductions work in 32-lane groups; on ROCm, width-32 shuffles) ----
// Split-K over (chunk, eval point), per-point power tables, J coefficients per thread per
// step, one REDC per pair of products. Exact field arithmetic (canonical Montgomery form),
// so results are bit-identical to the Horner kernel batch_evaluate_any for any input.

#define EU_J 16
#define EU_THREADS 256

__device__ __forceinline__ uint32_t eu_addmod(uint32_t a, uint32_t b) {
  uint32_t r = a + b;
  return r >= Fp::P ? r - Fp::P : r;
}
// (a0*b0 + a1*b1) * 2^-32 mod p, inputs < p. Same REDC formulation as Fp::mul.
__device__ __forceinline__ uint32_t eu_redc_pair(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1) {
  uint64_t o64 = (uint64_t)a0 * b0 + (uint64_t)a1 * b1;  // < 2p^2 < p*2^32
  uint32_t low = -uint32_t(o64);
  uint32_t red = Fp::M * low;
  o64 += uint64_t(red) * uint64_t(Fp::P);  // < 2^64
  uint32_t ret = o64 >> 32;                 // < 2p
  return ret >= Fp::P ? ret - Fp::P : ret;
}

#ifdef __HIPCC__
// ROCm: lazy accumulation for eu2_main. Hi-word fold: subtract P once if hi >= P, which keeps a
// 64-bit sum of products congruent mod P*2^32 (so its REDC is unchanged) and bounds hi < P.
__device__ __forceinline__ uint64_t eu_fold(uint64_t v) {
  uint32_t lo = uint32_t(v), hi = uint32_t(v >> 32);
  hi = __builtin_elementwise_min(hi, hi - Fp::P);
  return (uint64_t(hi) << 32) | lo;
}
// REDC of a 64-bit sum whose hi word is < P (so o64 + red*P < 2P*2^32 < 2^64); canonical
// result. red = M * -lo from shift-adds as in Fp::mul_rdna (32-bit multiplies are quarter rate).
__device__ __forceinline__ uint32_t eu_redc64(uint64_t o64) {
  uint32_t lo = uint32_t(o64);
  uint32_t tt = lo + (lo << 27);
  asm("" : "+v"(tt));
  uint32_t red = (lo << 31) - tt;
  o64 += uint64_t(red) * uint64_t(Fp::P);
  uint32_t ret = o64 >> 32;
  return __builtin_elementwise_min(ret, ret - Fp::P);
}
#endif

// pw[e*EU_THREADS + t] = x_e^(J*t);  ps[e*S + s] = x_e^(s*chunk)
__global__ void eu2_pows(FpExt* pw, FpExt* ps, const FpExt* xs, uint32_t S, uint32_t chunk) {
  uint32_t e = blockIdx.x;
  uint32_t t = threadIdx.x;
  FpExt x = xs[e];
  pw[(size_t)e * EU_THREADS + t] = pow(x, (size_t)EU_J * t);
  for (uint32_t s = t; s < S; s += blockDim.x) {
    ps[(size_t)e * S + s] = pow(x, (size_t)s * chunk);
  }
}

__device__ __forceinline__ FpExt fpext_shfl_down(FpExt v, int d) {
  FpExt r;
#pragma unroll
  for (int i = 0; i < 4; i++) {
#ifdef __HIPCC__
    r.elems[i] = Fp::fromRaw(__shfl_down(v.elems[i].asRaw(), d, 32));
#else
    r.elems[i] = Fp::fromRaw(__shfl_down_sync(0xffffffff, v.elems[i].asRaw(), d));
#endif
  }
  return r;
}

// grid = (S, nEvals), block = EU_THREADS
__global__ void __launch_bounds__(EU_THREADS) eu2_main(FpExt* partial,
                                                       const Fp* __restrict__ coeffs,
                                                       const uint32_t* __restrict__ which,
                                                       const FpExt* __restrict__ xs,
                                                       const FpExt* __restrict__ pw,
                                                       const FpExt* __restrict__ ps,
                                                       uint32_t deg,
                                                       uint32_t chunk) {
  const uint32_t s = blockIdx.x;
  const uint32_t e = blockIdx.y;
  const uint32_t S = gridDim.x;
  const uint32_t t = threadIdx.x;
  const Fp* poly = coeffs + (size_t)which[e] * deg + (size_t)s * chunk;
  FpExt x = xs[e];
  FpExt xp[EU_J];
  xp[0] = FpExt(1);
#pragma unroll
  for (int j = 1; j < EU_J; j++)
    xp[j] = xp[j - 1] * x;
  // step = x^(J*THREADS)
  FpExt step = pw[(size_t)e * EU_THREADS + (EU_THREADS - 1)] * (xp[EU_J - 1] * x);
  const uint32_t iters = chunk / (EU_J * EU_THREADS);
  const uint4* p4 = reinterpret_cast<const uint4*>(poly);
#ifdef __HIPCC__
  // acc' = acc * step + inner, per output component as one exact 64-bit sum of 4 + J raw
  // Montgomery products (NBETA folded into step, as the lazy FpExt multiply in sys/cxx/fpext.h)
  // reduced once: 4 products < 4p^2 < 2^64; then hi is folded below P before each further pair
  // (P*2^32 + 2^32 + 2p^2 < 2^64). Exact field arithmetic with canonical results, so bit-identical
  // to the pairwise form below; about half the multiplies.
  const uint32_t b0 = step.elems[0].asRaw(), b1 = step.elems[1].asRaw();
  const uint32_t b2 = step.elems[2].asRaw(), b3 = step.elems[3].asRaw();
  const uint32_t b1n = (step.elems[1] * Fp(Fp::P - 11)).asRaw();
  const uint32_t b2n = (step.elems[2] * Fp(Fp::P - 11)).asRaw();
  const uint32_t b3n = (step.elems[3] * Fp(Fp::P - 11)).asRaw();
  const uint32_t sm[4][4] = {{b0, b3n, b2n, b1n}, {b1, b0, b3n, b2n}, {b2, b1, b0, b3n}, {b3, b2, b1, b0}};
  uint32_t a[4] = {0, 0, 0, 0};
#pragma unroll 2
  for (int k = (int)iters - 1; k >= 0; k--) {
    size_t base = ((size_t)k * EU_THREADS + t) * (EU_J / 4);
    uint32_t c[EU_J];
#pragma unroll
    for (int q = 0; q < EU_J / 4; q++) {
      uint4 v = __ldg(p4 + base + q);
      c[4 * q + 0] = v.x;
      c[4 * q + 1] = v.y;
      c[4 * q + 2] = v.z;
      c[4 * q + 3] = v.w;
    }
    uint32_t na[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint64_t o = (uint64_t)a[0] * sm[i][0];
      o += (uint64_t)a[1] * sm[i][1];
      o += (uint64_t)a[2] * sm[i][2];
      o += (uint64_t)a[3] * sm[i][3];
#pragma unroll
      for (int j = 0; j < EU_J; j += 2) {
        o = eu_fold(o);
        o += (uint64_t)c[j] * xp[j].elems[i].asRaw();
        o += (uint64_t)c[j + 1] * xp[j + 1].elems[i].asRaw();
      }
      na[i] = eu_redc64(eu_fold(o));
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
      a[i] = na[i];
  }
  FpExt acc(Fp::fromRaw(a[0]), Fp::fromRaw(a[1]), Fp::fromRaw(a[2]), Fp::fromRaw(a[3]));
#else
  FpExt acc;
#pragma unroll 2
  for (int k = (int)iters - 1; k >= 0; k--) {
    size_t base = ((size_t)k * EU_THREADS + t) * (EU_J / 4);
    uint32_t c[EU_J];
#pragma unroll
    for (int q = 0; q < EU_J / 4; q++) {
      uint4 v = __ldg(p4 + base + q);
      c[4 * q + 0] = v.x;
      c[4 * q + 1] = v.y;
      c[4 * q + 2] = v.z;
      c[4 * q + 3] = v.w;
    }
    // Pair-wise lazy reduction: one REDC per two products.
    FpExt inner;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t sacc = 0;
#pragma unroll
      for (int j = 0; j < EU_J; j += 2)
        sacc = eu_addmod(sacc, eu_redc_pair(c[j], xp[j].elems[i].asRaw(), c[j + 1], xp[j + 1].elems[i].asRaw()));
      inner.elems[i] = Fp::fromRaw(sacc);
    }
    acc = acc * step + inner;
  }
#endif
  acc = acc * pw[(size_t)e * EU_THREADS + t];
  // block reduce
#pragma unroll
  for (int d = 16; d > 0; d >>= 1)
    acc += fpext_shfl_down(acc, d);
#ifdef __HIPCC__
  // HIP doesn't allow __shared__ objects with constructors: raw storage, as FpExt.
  __shared__ uint32_t red_raw[EU_THREADS / 32 * 4] __attribute__((aligned(16)));
  FpExt* red = reinterpret_cast<FpExt*>(red_raw);
#else
  __shared__ FpExt red[EU_THREADS / 32];
#endif
  if ((t & 31) == 0)
    red[t >> 5] = acc;
  __syncthreads();
  if (t < 32) {
    FpExt v = (t < EU_THREADS / 32) ? red[t] : FpExt();
#pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      v += fpext_shfl_down(v, d);
    if (t == 0)
      partial[(size_t)e * S + s] = v * ps[(size_t)e * S + s];
  }
}

// grid = nEvals, block = 32
__global__ void eu2_reduce(FpExt* out, const FpExt* partial, uint32_t S) {
  uint32_t e = blockIdx.x;
  FpExt v;
  for (uint32_t s = threadIdx.x; s < S; s += 32)
    v += partial[(size_t)e * S + s];
#pragma unroll
  for (int d = 16; d > 0; d >>= 1)
    v += fpext_shfl_down(v, d);
  if (threadIdx.x == 0)
    out[e] = v;
}


// ---- mix_poly_coeffs v2 (CUDA and ROCm; no warp-size assumptions) ----
// one RMW of `out` per run of equal combo ids (instead of one per input),
// lazy pair REDC for the FpExt x Fp products, parallel mix-power table in shared memory.
// Exact field arithmetic with canonical outputs => bit-identical to the original kernel.

__device__ __forceinline__ uint32_t mx_addmod(uint32_t a, uint32_t b) {
  uint32_t r = a + b;
  return r >= Fp::P ? r - Fp::P : r;
}
// (a0*b0 + a1*b1) * 2^-32 mod p for inputs < p (same REDC as Fp::mul).
__device__ __forceinline__ uint32_t mx_redc_pair(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1) {
  uint64_t o64 = (uint64_t)a0 * b0 + (uint64_t)a1 * b1;
  uint32_t low = -uint32_t(o64);
  uint32_t red = Fp::M * low;
  o64 += uint64_t(red) * uint64_t(Fp::P);
  uint32_t ret = o64 >> 32;
  return ret >= Fp::P ? ret - Fp::P : ret;
}

__global__ void __launch_bounds__(256) mix_poly_coeffs_v2(FpExt* __restrict__ out,
                                                          const Fp* __restrict__ in,
                                                          const uint32_t* __restrict__ combos,
                                                          const FpExt* __restrict__ mixStart,
                                                          const FpExt* __restrict__ mix,
                                                          const uint32_t inputSize,
                                                          const uint32_t count) {
  extern __shared__ uint32_t mx_smem[];
  FpExt* mixPows = reinterpret_cast<FpExt*>(mx_smem);
  uint32_t* cids = mx_smem + 4 * inputSize;
  {
    FpExt ms = *mixStart;
    FpExt m = *mix;
    for (uint32_t i = threadIdx.x; i < inputSize; i += blockDim.x) {
      mixPows[i] = ms * pow(m, i);
      cids[i] = combos[i];
    }
  }
  __syncthreads();
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;
  uint32_t acc[4] = {0, 0, 0, 0};
  uint32_t i = 0;
  while (i < inputSize) {
    uint32_t id = cids[i];
    bool pair = (i + 1 < inputSize) && (cids[i + 1] == id);
    uint32_t c0 = __ldg(reinterpret_cast<const uint32_t*>(in) + (size_t)count * i + idx);
    uint32_t c1 = pair ? __ldg(reinterpret_cast<const uint32_t*>(in) + (size_t)count * (i + 1) + idx) : 0;
    const FpExt& m0 = mixPows[i];
    const FpExt& m1 = mixPows[pair ? i + 1 : i];
#pragma unroll
    for (int k = 0; k < 4; k++)
      acc[k] = mx_addmod(acc[k], mx_redc_pair(c0, m0.elems[k].asRaw(), c1, m1.elems[k].asRaw()));
    i += pair ? 2 : 1;
    if (i >= inputSize || cids[i] != id) {
      FpExt* o = out + (size_t)count * id + idx;
      FpExt v = *o;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        v.elems[k] = v.elems[k] + Fp::fromRaw(acc[k]);
        acc[k] = 0;
      }
      *o = v;
    }
  }
}

#ifdef __HIPCC__
// ---- mix_poly_coeffs v6 (ROCm) ----
// 2 consecutive elements per thread with nontemporal 8-byte loads issued 4 inputs ahead; the
// FpExt x Fp products of each combo id < MX_C summed exactly in 64-bit accumulators held in
// registers (hi folded below P before each product, so no overflow; one REDC per combo at the
// end) and one RMW of `out` per combo instead of one per run of equal ids; ids >= MX_C use a run
// accumulator and an RMW per run, as v2. Exact field arithmetic with canonical outputs =>
// bit-identical to v2. Needs an even count and an 8-byte aligned `in`.
#define MX_C 5
#define MX_V 2
#define MX_U 4
__device__ __forceinline__ uint64_t mx_fold(uint64_t v) {
  uint32_t lo = uint32_t(v), hi = uint32_t(v >> 32);
  hi = __builtin_elementwise_min(hi, hi - Fp::P);
  return (uint64_t(hi) << 32) | lo;
}
// REDC of a 64-bit sum whose hi word is < P; canonical result (red from shift-adds, as mul_rdna).
__device__ __forceinline__ uint32_t mx_redc64(uint64_t o64) {
  uint32_t lo = uint32_t(o64);
  uint32_t tt = lo + (lo << 27);
  asm("" : "+v"(tt));
  uint32_t red = (lo << 31) - tt;
  o64 += uint64_t(red) * uint64_t(Fp::P);
  uint32_t ret = o64 >> 32;
  return __builtin_elementwise_min(ret, ret - Fp::P);
}

__global__ void __launch_bounds__(256) mix_poly_coeffs_v6(FpExt* __restrict__ out,
                                                          const Fp* __restrict__ in,
                                                          const uint32_t* __restrict__ combos,
                                                          const FpExt* __restrict__ mixStart,
                                                          const FpExt* __restrict__ mix,
                                                          const uint32_t inputSize,
                                                          const uint32_t count) {
  extern __shared__ uint32_t mx_smem[];
  FpExt* mixPows = reinterpret_cast<FpExt*>(mx_smem);
  uint32_t* cids = mx_smem + 4 * inputSize;
  {
    FpExt ms = *mixStart;
    FpExt m = *mix;
    for (uint32_t i = threadIdx.x; i < inputSize; i += blockDim.x) {
      mixPows[i] = ms * pow(m, i);
      cids[i] = combos[i];
    }
  }
  __syncthreads();
  const uint32_t idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * MX_V;
  if (idx0 >= count)
    return;
  typedef uint32_t mx_vec_t __attribute__((ext_vector_type(MX_V)));
  const mx_vec_t* inv =
      reinterpret_cast<const mx_vec_t*>(reinterpret_cast<const uint32_t*>(in) + idx0);
  uint64_t o[MX_C + 1][MX_V][4];  // [MX_C]: the current run of an id >= MX_C
#pragma unroll
  for (int cc = 0; cc <= MX_C; cc++)
#pragma unroll
    for (int e = 0; e < MX_V; e++)
      o[cc][e][0] = o[cc][e][1] = o[cc][e][2] = o[cc][e][3] = 0;
  uint32_t used = 0;  // bit cc: combo cc accumulated (uniform)
  const size_t cstride = count / MX_V;
  for (uint32_t i0 = 0; i0 < inputSize; i0 += MX_U) {
    mx_vec_t c[MX_U];
#pragma unroll
    for (int u = 0; u < MX_U; u++)
      c[u] = (i0 + u < inputSize) ? __builtin_nontemporal_load(inv + cstride * (i0 + u)) : mx_vec_t(0);
#pragma unroll
    for (int u = 0; u < MX_U; u++) {
      const uint32_t i = i0 + u;
      if (i < inputSize) {
        const uint32_t id = cids[i];
        const uint32_t slot = id < MX_C ? id : MX_C;
        const uint4 m = reinterpret_cast<const uint4*>(mixPows)[i];
#pragma unroll
        for (int cc = 0; cc <= MX_C; cc++) {
          if (cc == (int)slot) {
#pragma unroll
            for (int e = 0; e < MX_V; e++) {
              o[cc][e][0] = mx_fold(o[cc][e][0]) + (uint64_t)c[u][e] * m.x;
              o[cc][e][1] = mx_fold(o[cc][e][1]) + (uint64_t)c[u][e] * m.y;
              o[cc][e][2] = mx_fold(o[cc][e][2]) + (uint64_t)c[u][e] * m.z;
              o[cc][e][3] = mx_fold(o[cc][e][3]) + (uint64_t)c[u][e] * m.w;
            }
          }
        }
        if (slot < MX_C) {
          used |= 1u << slot;
        } else if (i + 1 >= inputSize || cids[i + 1] != id) {
#pragma unroll
          for (int e = 0; e < MX_V; e++) {
            FpExt* op = out + (size_t)count * id + idx0 + e;
            FpExt v = *op;
#pragma unroll
            for (int k = 0; k < 4; k++) {
              v.elems[k] = v.elems[k] + Fp::fromRaw(mx_redc64(mx_fold(o[MX_C][e][k])));
              o[MX_C][e][k] = 0;
            }
            *op = v;
          }
        }
      }
    }
  }
#pragma unroll
  for (int cc = 0; cc < MX_C; cc++) {
    if (used & (1u << cc)) {
#pragma unroll
      for (int e = 0; e < MX_V; e++) {
        FpExt* op = out + (size_t)count * cc + idx0 + e;
        FpExt v = *op;
#pragma unroll
        for (int k = 0; k < 4; k++)
          v.elems[k] = v.elems[k] + Fp::fromRaw(mx_redc64(mx_fold(o[cc][e][k])));
        *op = v;
      }
    }
  }
}
#undef MX_C
#undef MX_V
#undef MX_U
#endif

// ---- batched gather_sample (all FRI queries of one tree in one launch) ----
__global__ void gather_samples(Fp* dst, const Fp* src, const uint32_t* idxs, const uint32_t n,
                               const uint32_t size, const uint32_t stride) {
  uint gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n * size) {
    uint32_t q = gid / size;
    uint32_t i = gid - q * size;
    dst[gid] = src[(size_t)i * stride + idxs[q]];
  }
}
