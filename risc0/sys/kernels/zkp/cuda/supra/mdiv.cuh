// Fused multi-root polynomial division for risc0 combos_divide (CUDA and ROCm).
// Bit-identical to calling sppark div_by_x_minus_z<rotate=true> once per root, in order, on
// each combo slice (including for inputs whose remainders are nonzero), but with
// 2 short launches per root "layer" (all combos of a layer batched) + 2, and no host syncs.
//
// One root z on a slice a[0..n): S_k = sum_{j>=k} a[j] z^(j-k) (suffix Horner); the output is
// a'[k] = S_{k+1} (k < n-1), a'[n-1] = S_0 (the remainder). Tiles of MD_T elements: per root,
// mdiv_scan (one block per slice) turns the tile aggregates into tile carries, and
// mdiv_apply produces the tile's outputs and, if another root follows, the next aggregates.
// a'[n-1] (the remainder, produced by tile 0) is written by the next launch.
#pragma once
#include <ff/baby_bear.hpp>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <mutex>

#ifndef MD_NTH
#define MD_NTH 256
#endif
#define MD_E 8
#define MD_T (MD_NTH * MD_E)
#define MD_MAXJ 8

// ROCm: no __grid_constant__ (the jobs are an ordinary by-value kernel argument), and the
// shuffles are sppark cuda2hip's 32-lane polyfills (the kernels need wave32: RDNA). The lanes
// they read across the wave boundary are discarded by the csel()s, as on CUDA.
#ifdef __HIPCC__
#define MD_GRID_CONST
#else
#define MD_GRID_CONST __grid_constant__
#endif

typedef bb31_4_t md4_t;

struct MdJob {
  md4_t* a;            // slice (n elements)
  md4_t* agg;          // per-tile aggregates (n / MD_T)
  md4_t* carry;        // per-tile carries (n / MD_T)
  md4_t* rem;          // remainder of this root (written by tile 0's apply)
  const md4_t* prev_rem; // remainder of the previous root of this slice, or null
  md4_t z, znext;
  int has_next;
};
struct MdJobs {
  MdJob j[MD_MAXJ];
  uint32_t n;
};

__device__ __forceinline__ md4_t md_shfl_down(const md4_t& v, int d) {
  md4_t r;
  const uint32_t* s = reinterpret_cast<const uint32_t*>(&v);
  uint32_t* o = reinterpret_cast<uint32_t*>(&r);
#pragma unroll
  for (int i = 0; i < 4; i++) o[i] = __shfl_down_sync(0xffffffffu, s[i], d);
  return r;
}
__device__ __forceinline__ md4_t md_shfl_up(const md4_t& v, int d) {
  md4_t r;
  const uint32_t* s = reinterpret_cast<const uint32_t*>(&v);
  uint32_t* o = reinterpret_cast<uint32_t*>(&r);
#pragma unroll
  for (int i = 0; i < 4; i++) o[i] = __shfl_up_sync(0xffffffffu, s[i], d);
  return r;
}
__device__ __forceinline__ md4_t md_zero() { md4_t r; r.zero(); return r; }

// Inclusive suffix scan over the block: returns V_t = sum_{u>=t} v_u * w^(u-t), w given.
// Also returns (via *next) V_{t+1} (zero for the last thread). NT threads, NT % 32 == 0,
// NT <= 1024. Shared scratch: sh[NT/32 + 33].
template <int NT>
__device__ __forceinline__ md4_t md_block_suffix_scan(md4_t v, md4_t w, md4_t* sh, md4_t* next) {
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  constexpr int NW = NT / 32;
  // warp-level Kogge-Stone; wp = w^d
  md4_t wp = w;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    md4_t o = md_shfl_down(v, d);
    md4_t c = v + o * wp;
    v = md4_t::csel(c, v, lane + d < 32);
    wp = wp * wp;
  }
  // wp = w^32. lane powers: pw[l] = w^(l+1)
  md4_t* pw = sh + NW + 1;
  {
    md4_t p = w;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      md4_t o = md_shfl_up(p, d);
      md4_t c = p * o;
      p = md4_t::csel(c, p, lane >= d);
    }
    if (warp == 0) pw[lane] = p;
  }
  if (lane == 0) sh[warp] = v;  // warp total
  __syncthreads();
  if (t == 0) {
    md4_t x = md_zero();
    sh[NW] = x;
    for (int k = NW - 1; k >= 0; k--) {
      x = sh[k] + wp * x;  // suffix over warps
      sh[k] = x;
    }
  }
  __syncthreads();
  md4_t xa = sh[warp + 1];  // carry from warps above (V at 32(warp+1))
  md4_t r = v + pw[31 - lane] * xa;
  md4_t nx = md_shfl_down(r, 1);
  if (lane == 31) nx = xa;
  *next = nx;
  __syncthreads();
  return r;
}


#ifdef MD_COAL
#define MD_PAD(e) ((e) + ((e) >> 3))
#define MD_BUF (MD_T + MD_T / 8)
// Coalesced tile load/store through shared memory (padded: conflict-free both ways).
__device__ __forceinline__ void md_load_tile(md4_t* x, const md4_t* src, md4_t* buf) {
#pragma unroll
  for (int j = 0; j < MD_E; j++) buf[MD_PAD(j * MD_NTH + threadIdx.x)] = src[j * MD_NTH + threadIdx.x];
  __syncthreads();
#pragma unroll
  for (int j = 0; j < MD_E; j++) x[j] = buf[MD_PAD(threadIdx.x * MD_E + j)];
  __syncthreads();
}
// skip_last: do not store the tile's last element.
__device__ __forceinline__ void md_store_tile(md4_t* dst, const md4_t* x, md4_t* buf, bool skip_last) {
#pragma unroll
  for (int j = 0; j < MD_E; j++) buf[MD_PAD(threadIdx.x * MD_E + j)] = x[j];
  __syncthreads();
#pragma unroll
  for (int j = 0; j < MD_E; j++) {
    int e = j * MD_NTH + threadIdx.x;
    if (!(skip_last && e == MD_T - 1)) dst[e] = buf[MD_PAD(e)];
  }
  __syncthreads();
}
#endif

// Root-0 tile aggregates: agg[i] = sum_{k in tile i} a[k] z^(k - i*T).
__global__ __launch_bounds__(MD_NTH) void mdiv_agg0(const MD_GRID_CONST MdJobs J) {
  __shared__ md4_t sh[MD_NTH / 32 + 33];
  const MdJob& jb = J.j[blockIdx.y];
  const size_t b = (size_t)blockIdx.x * MD_T + threadIdx.x * MD_E;
  md4_t z = jb.z;
#ifdef MD_COAL
  __shared__ md4_t buf[MD_BUF];
  md4_t x[MD_E];
  md_load_tile(x, jb.a + (size_t)blockIdx.x * MD_T, buf);
  md4_t p = x[MD_E - 1];
#pragma unroll
  for (int j = MD_E - 2; j >= 0; j--) p = x[j] + z * p;
#else
  md4_t p = jb.a[b + MD_E - 1];
#pragma unroll
  for (int j = MD_E - 2; j >= 0; j--) p = jb.a[b + j] + z * p;
#endif
  md4_t zE = z ^ (uint32_t)MD_E;
  md4_t nx;
  md4_t v = md_block_suffix_scan<MD_NTH>(p, zE, sh, &nx);
  if (threadIdx.x == 0) jb.agg[blockIdx.x] = v;
}

// Per slice (blockIdx.x), NTILES threads: carries C_i = sum_{u>i} agg_u z^(T(u-i-1)).
// Also installs the previous root's remainder at a[n-1] and in the top tile's aggregate.
template <int NTILES>
__global__ __launch_bounds__(NTILES) void mdiv_scan(const MD_GRID_CONST MdJobs J) {
  __shared__ md4_t sh[NTILES / 32 + 33];
  const MdJob& jb = J.j[blockIdx.x];
  const int t = threadIdx.x;
  md4_t z = jb.z;
  md4_t v = jb.agg[t];
  if (jb.prev_rem != nullptr && t == NTILES - 1) {
    md4_t r = *jb.prev_rem;
    jb.a[J.n - 1] = r;
    v = v + r * (z ^ (uint32_t)(MD_T - 1));
  }
  md4_t zT = z ^ (uint32_t)MD_T;
  md4_t nx;
  md_block_suffix_scan<NTILES>(v, zT, sh, &nx);
  jb.carry[t] = nx;
}

// Apply one root on each tile; optionally compute the next root's tile aggregates.
__global__ __launch_bounds__(MD_NTH) void mdiv_apply(const MD_GRID_CONST MdJobs J) {
  __shared__ md4_t sh[MD_NTH / 32 + 33];
  const MdJob& jb = J.j[blockIdx.y];
  const uint32_t tile = blockIdx.x, ntiles = gridDim.x;
  const size_t b = (size_t)tile * MD_T + threadIdx.x * MD_E;
  md4_t z = jb.z;
  md4_t x[MD_E];
#ifdef MD_COAL
  __shared__ md4_t buf[MD_BUF];
  md_load_tile(x, jb.a + (size_t)tile * MD_T, buf);
#else
#pragma unroll
  for (int j = 0; j < MD_E; j++) x[j] = jb.a[b + j];
#endif
  // thread-local Horner from the top
  md4_t p = x[MD_E - 1];
#pragma unroll
  for (int j = MD_E - 2; j >= 0; j--) p = x[j] + z * p;
  md4_t zE = z ^ (uint32_t)MD_E;
  md4_t C = jb.carry[tile];
  if (threadIdx.x == MD_NTH - 1) p = p + zE * C;
  md4_t nx;
  md_block_suffix_scan<MD_NTH>(p, zE, sh, &nx);
  // S at this thread's top boundary
  md4_t s = (threadIdx.x == MD_NTH - 1) ? C : nx;
  // outputs: out[b+j] = S_{b+j+1}
  md4_t o[MD_E];
#pragma unroll
  for (int j = MD_E - 1; j >= 0; j--) {
    o[j] = s;
    s = x[j] + z * s;
  }
  // s = S_b
  const bool top = (tile == ntiles - 1) && (threadIdx.x == MD_NTH - 1);
#ifdef MD_COAL
  md_store_tile(jb.a + (size_t)tile * MD_T, o, buf, tile == ntiles - 1);
#else
#pragma unroll
  for (int j = 0; j < MD_E; j++)
    if (!(top && j == MD_E - 1)) jb.a[b + j] = o[j];
#endif
  if (tile == 0 && threadIdx.x == 0) *jb.rem = s;
  if (jb.has_next) {
    md4_t zn = jb.znext;
    if (top) o[MD_E - 1].zero();
    md4_t q = o[MD_E - 1];
#pragma unroll
    for (int j = MD_E - 2; j >= 0; j--) q = o[j] + zn * q;
    md4_t znE = zn ^ (uint32_t)MD_E;
    md4_t v = md_block_suffix_scan<MD_NTH>(q, znE, sh, &nx);
    if (threadIdx.x == 0) jb.agg[tile] = v;
  }
}

// Install the last remainder of each slice at a[n-1].
__global__ void mdiv_final(const MD_GRID_CONST MdJobs J, uint32_t njobs) {
  uint32_t t = threadIdx.x;
  if (t < njobs) J.j[t].a[J.n - 1] = *J.j[t].prev_rem;
}

template <int NTILES>
static void mdiv_launch_scan(const MdJobs& J, int nj, cudaStream_t st) {
  mdiv_scan<NTILES><<<nj, NTILES, 0, st>>>(J);
}

// True if mdiv_run handles this shape (else use the per-root sppark path).
static inline bool mdiv_supported(size_t n, uint32_t num_combos, const uint32_t pows_per_combo[]) {
  if (num_combos == 0 || num_combos > MD_MAXJ || n % MD_T != 0) return false;
  size_t nt = n / MD_T;
  if (nt < 32 || nt > 1024 || (nt & (nt - 1)) != 0) return false;
  for (uint32_t c = 0; c < num_combos; c++)
    if (pows_per_combo[c] == 0) return false;
  return true;
}

// Divides each slice base + combo_indices[c] * n by its roots in order, exactly as
// num_roots sequential sppark div_by_x_minus_z<true> calls would; remainders (one per root,
// in order) are copied to host memory rems_h. Synchronizes `st` before returning.
static void mdiv_run(md4_t* base, size_t n, const uint32_t combo_indices[], const uint32_t pows_per_combo[],
                     const md4_t* all_pows, uint32_t num_combos, md4_t* rems_h, cudaStream_t st) {
  const uint32_t ntiles = n / MD_T;
  std::vector<size_t> roff(num_combos);
  size_t tot = 0;
  uint32_t maxm = 0;
  for (uint32_t c = 0; c < num_combos; c++) {
    roff[c] = tot;
    tot += pows_per_combo[c];
    maxm = std::max(maxm, pows_per_combo[c]);
  }
  md4_t* scratch = nullptr;
  const size_t scratch_bytes = (2 * (size_t)num_combos * ntiles + tot) * sizeof(md4_t);
#ifdef __HIPCC__
  // A grow-only scratch per device (cuda2hip's cudaMallocAsync/cudaFreeAsync are hipMalloc and
  // hipFree, and hipFree synchronizes the device), held under a lock until this function has
  // synchronized `st`.
  static std::mutex s_mtx;
  std::lock_guard<std::mutex> lock(s_mtx);
  {
    static md4_t* s_scratch[16] = {};
    static size_t s_scratch_bytes[16] = {};
    int dev = 0;
    CUDA_OK(cudaGetDevice(&dev));
    if (scratch_bytes > s_scratch_bytes[dev]) {
      if (s_scratch[dev]) CUDA_OK(cudaFree(s_scratch[dev]));
      s_scratch[dev] = nullptr;
      s_scratch_bytes[dev] = 0;
      CUDA_OK(cudaMalloc(&s_scratch[dev], scratch_bytes));
      s_scratch_bytes[dev] = scratch_bytes;
    }
    scratch = s_scratch[dev];
  }
#else
  CUDA_OK(cudaMallocAsync((void**)&scratch, scratch_bytes, st));
  // Freed (stream-ordered) on every path out, also when a launch check below throws.
  struct scratch_guard {
    md4_t* p;
    cudaStream_t st;
    ~scratch_guard() { (void)cudaFreeAsync(p, st); }
  } guard{scratch, st};
#endif
  md4_t* rems_d = scratch + 2 * (size_t)num_combos * ntiles;
  auto job = [&](uint32_t c, uint32_t r) {
    MdJob j;
    j.a = base + (size_t)combo_indices[c] * n;
    j.agg = scratch + (size_t)c * 2 * ntiles;
    j.carry = j.agg + ntiles;
    j.rem = rems_d + roff[c] + r;
    j.prev_rem = r > 0 ? rems_d + roff[c] + r - 1 : nullptr;
    j.z = all_pows[roff[c] + r];
    j.has_next = r + 1 < pows_per_combo[c];
    j.znext = j.has_next ? all_pows[roff[c] + r + 1] : j.z;
    return j;
  };
  MdJobs J;
  J.n = (uint32_t)n;
  int nj = 0;
  for (uint32_t c = 0; c < num_combos; c++) J.j[nj++] = job(c, 0);
  mdiv_agg0<<<dim3(ntiles, nj), MD_NTH, 0, st>>>(J);
  for (uint32_t r = 0; r < maxm; r++) {
    nj = 0;
    for (uint32_t c = 0; c < num_combos; c++)
      if (pows_per_combo[c] > r) J.j[nj++] = job(c, r);
    switch (ntiles) {
      case 32: mdiv_launch_scan<32>(J, nj, st); break;
      case 64: mdiv_launch_scan<64>(J, nj, st); break;
      case 128: mdiv_launch_scan<128>(J, nj, st); break;
      case 256: mdiv_launch_scan<256>(J, nj, st); break;
      case 512: mdiv_launch_scan<512>(J, nj, st); break;
      default: mdiv_launch_scan<1024>(J, nj, st); break;
    }
    mdiv_apply<<<dim3(ntiles, nj), MD_NTH, 0, st>>>(J);
  }
  nj = 0;
  for (uint32_t c = 0; c < num_combos; c++) {
    MdJob j = job(c, 0);
    j.prev_rem = rems_d + roff[c] + pows_per_combo[c] - 1;
    J.j[nj++] = j;
  }
  mdiv_final<<<1, 32, 0, st>>>(J, nj);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaMemcpyAsync(rems_h, rems_d, tot * sizeof(md4_t), cudaMemcpyDeviceToHost, st));
  CUDA_OK(cudaStreamSynchronize(st));
}
