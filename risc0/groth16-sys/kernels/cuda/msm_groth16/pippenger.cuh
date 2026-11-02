// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_PIPPENGER_CUH__
#define __SPPARK_MSM_PIPPENGER_CUH__

#include <cuda.h>
#include <cooperative_groups.h>
#include <cassert>

#include <util/vec2d_t.hpp>
#include <util/slice_t.hpp>

#include "sort.cuh"
#include "batch_addition.cuh"

#ifndef WARP_SZ
# define WARP_SZ 32
#endif
#ifdef __GNUC__
# define asm __asm__ __volatile__
#else
# define asm asm volatile
#endif

/*
 * Break down |scalars| to signed |wbits|-wide digits.
 */

#ifdef __CUDA_ARCH__
// Transposed scalar_t
template<class scalar_t>
class scalar_T {
    uint32_t val[sizeof(scalar_t)/sizeof(uint32_t)][WARP_SZ];

public:
    __device__ const uint32_t& operator[](size_t i) const  { return val[i][0]; }
    __device__ scalar_T& operator()(uint32_t laneid)
    {   return *reinterpret_cast<scalar_T*>(&val[0][laneid]);   }
    __device__ scalar_T& operator=(const scalar_t& rhs)
    {
        for (size_t i = 0; i < sizeof(scalar_t)/sizeof(uint32_t); i++)
            val[i][0] = rhs[i];
        return *this;
    }
};

template<class scalar_t>
__device__ __forceinline__
static uint32_t get_wval(const scalar_T<scalar_t>& scalar, uint32_t off,
                         uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1)
{
    uint32_t i = off / 32;
    uint64_t ret = scalar[i];

    if (i < top_i)
        ret |= (uint64_t)scalar[i+1] << 32;

    return ret >> (off%32);
}

__device__ __forceinline__
static uint32_t booth_encode(uint32_t wval, uint32_t wmask, uint32_t wbits)
{
    uint32_t sign = (wval >> wbits) & 1;
    wval = ((wval + 1) & wmask) >> 1;
    return sign ? 0-wval : wval;
}
#endif

template<class scalar_t>
__launch_bounds__(1024) __global__
void breakdown(vec2d_t<uint32_t> digits, const scalar_t scalars[], size_t len,
               uint32_t nwins, uint32_t wbits, bool mont = true)
{
    assert(len <= (1U<<31) && wbits < 32);

#ifdef __CUDA_ARCH__
    extern __shared__ scalar_T<scalar_t> xchange[];
    const uint32_t tid = threadIdx.x;
    const uint32_t tix = threadIdx.x + blockIdx.x*blockDim.x;

    const uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1;
    const uint32_t wmask = 0xffffffffU >> (31-wbits); // (1U << (wbits+1)) - 1;

    auto& scalar = xchange[tid/WARP_SZ](tid%WARP_SZ);

    #pragma unroll 1
    for (uint32_t i = tix; i < (uint32_t)len; i += gridDim.x*blockDim.x) {
        auto s = scalars[i];

#if 0
        s.from();
        if (!mont) s.to();
#else
        if (mont) s.from();
#endif

        // clear the most significant bit
        uint32_t msb = s.abs();
        msb <<= 31;

        scalar = s;

        #pragma unroll 1
        for (uint32_t bit0 = nwins*wbits - 1, win = nwins; --win;) {
            bit0 -= wbits;
            uint32_t wval = get_wval(scalar, bit0, top_i);
            wval = booth_encode(wval, wmask, wbits);
            if (wval) wval ^= msb;
            digits[win][i] = wval;
        }

        uint32_t wval = s[0] << 1;
        wval = booth_encode(wval, wmask, wbits);
        if (wval) wval ^= msb;
        digits[0][i] = wval;
    }
#endif
}

#ifndef LARGE_L1_CODE_CACHE
# if __CUDA_ARCH__-0 >= 800
#  define LARGE_L1_CODE_CACHE 1
#  define ACCUMULATE_NTHREADS 384
# else
#  define LARGE_L1_CODE_CACHE 0
#  define ACCUMULATE_NTHREADS (bucket_t::degree == 1 ? 384 : 256)
# endif
#endif

#ifndef MSM_NTHREADS
# define MSM_NTHREADS 256
#endif
#if MSM_NTHREADS < 32 || (MSM_NTHREADS & (MSM_NTHREADS-1)) != 0
# error "bad MSM_NTHREADS value"
#endif
#ifndef MSM_NSTREAMS
# define MSM_NSTREAMS 8
#elif MSM_NSTREAMS<2
# error "invalid MSM_NSTREAMS"
#endif

template<class bucket_t,
         class affine_h,
         class bucket_h = class bucket_t::mem_t,
         class affine_t = class bucket_t::affine_t>
__launch_bounds__(ACCUMULATE_NTHREADS) __global__
void accumulate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits,
                /*const*/ affine_h points_[], const vec2d_t<uint32_t> digits,
                const vec2d_t<uint32_t> histogram, uint32_t sid = 0)
{
    vec2d_t<bucket_h> buckets{buckets_, 1U<<--wbits};
    const affine_h* points = points_;

    static __device__ uint32_t streams[MSM_NSTREAMS];
    uint32_t& current = streams[sid % MSM_NSTREAMS];
    uint32_t laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));
    const uint32_t degree = bucket_t::degree;
    const uint32_t warp_sz = WARP_SZ / degree;
    const uint32_t lane_id = laneid / degree;

    uint32_t x, y;
#if 1
    __shared__ uint32_t xchg;

    if (threadIdx.x == 0)
        xchg = atomicAdd(&current, blockDim.x/degree);
    __syncthreads();
    x = xchg + threadIdx.x/degree;
#else
    x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0;
    x = __shfl_sync(0xffffffff, x, 0) + lane_id;
#endif

    while (x < (nwins << wbits)) {
        y = x >> wbits;
        x &= (1U << wbits) - 1;
        const uint32_t* h = &histogram[y][x];

        uint32_t idx, len = h[0];

        asm("{ .reg.pred %did;"
            "  shfl.sync.up.b32 %0|%did, %1, %2, 0, 0xffffffff;"
            "  @!%did mov.b32 %0, 0;"
            "}" : "=r"(idx) : "r"(len), "r"(degree));

        if (lane_id == 0 && x != 0)
            idx = h[-1];

        if ((len -= idx) && !(x == 0 && y == 0)) {
            const uint32_t* digs_ptr = &digits[y][idx];
            uint32_t digit = *digs_ptr++;

            affine_t p = points[digit & 0x7fffffff];
            bucket_t bucket = p;
            bucket.cneg(digit >> 31);

            while (--len) {
                digit = *digs_ptr++;
                p = points[digit & 0x7fffffff];
                if (sizeof(bucket) <= 128 || LARGE_L1_CODE_CACHE)
                    bucket.add(p, digit >> 31);
                else
                    bucket.uadd(p, digit >> 31);
            }

            buckets[y][x] = bucket;
        } else {
            buckets[y][x].inf();
        }

        x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0;
        x = __shfl_sync(0xffffffff, x, 0) + lane_id;
    }

    cooperative_groups::this_grid().sync();

    if (threadIdx.x + blockIdx.x == 0)
        current = 0;
}

template<class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(256) __global__
void integrate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits, uint32_t nbits)
{
    const uint32_t degree = bucket_t::degree;
    uint32_t Nthrbits = 31 - __clz(blockDim.x / degree);

    assert((blockDim.x & (blockDim.x-1)) == 0 && wbits-1 > Nthrbits);

    vec2d_t<bucket_h> buckets{buckets_, 1U<<(wbits-1)};
    extern __shared__ uint4 scratch_[];
    auto* scratch = reinterpret_cast<bucket_h*>(scratch_);
    const uint32_t tid = threadIdx.x / degree;
    const uint32_t bid = blockIdx.x;

    auto* row = &buckets[bid][0];
    uint32_t i = 1U << (wbits-1-Nthrbits);
    row += tid * i;

    uint32_t mask = 0;
    if ((bid+1)*wbits > nbits) {
        uint32_t lsbits = nbits - bid*wbits;
        mask = (1U << (wbits-lsbits)) - 1;
    }

    bucket_t res, acc = row[--i];

    if (i & mask) {
        if (sizeof(res) <= 128) res.inf();
        else                    scratch[tid].inf();
    } else {
        if (sizeof(res) <= 128) res = acc;
        else                    scratch[tid] = acc;
    }

    bucket_t p;

    #pragma unroll 1
    while (i--) {
        p = row[i];

        uint32_t pc = i & mask ? 2 : 0;
        #pragma unroll 1
        do {
            if (sizeof(bucket_t) <= 128) {
                p.add(acc);
                if (pc == 1) {
                    res = p;
                } else {
                    acc = p;
                    if (pc == 0) p = res;
                }
            } else {
                if (LARGE_L1_CODE_CACHE && degree == 1)
                    p.add(acc);
                else
                    p.uadd(acc);
                if (pc == 1) {
                    scratch[tid] = p;
                } else {
                    acc = p;
                    if (pc == 0) p = scratch[tid];
                }
            }
        } while (++pc < 2);
    }

    __syncthreads();

    buckets[bid][2*tid] = p;
    buckets[bid][2*tid+1] = acc;
}
#ifndef __HIPCC__
// Split integrate (group-element equal to integrate).
// ---- split integrate: identical per-thread loop, Nthrbits += log2(K), partials to scratch ----
template<class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(256) __global__
void integrate_split(bucket_h part[], const bucket_h buckets_[], uint32_t wbits, uint32_t nbits, uint32_t lgK)
{
    const uint32_t degree = bucket_t::degree;
    const uint32_t T = blockDim.x / degree;
    const uint32_t lgT = 31 - __clz(T);
    const uint32_t K = 1u << lgK;
    extern __shared__ uint4 scratch_[];
    auto* scratch = reinterpret_cast<bucket_h*>(scratch_);
    const uint32_t tid = threadIdx.x / degree;
    const uint32_t win = blockIdx.x >> lgK, sub = blockIdx.x & (K - 1);
    const uint32_t g = sub * T + tid;                       // global thread index within the window
    const bucket_h* row = buckets_ + ((size_t)win << (wbits - 1));
    uint32_t i = 1U << (wbits - 1 - lgT - lgK);             // L' buckets per thread
    row += (size_t)g * i;

    uint32_t mask = 0;
    if ((win + 1) * wbits > nbits) {
        uint32_t lsbits = nbits - win * wbits;
        mask = (1U << (wbits - lsbits)) - 1;
    }

    bucket_t res, acc = row[--i];
    if (i & mask) {
        if (sizeof(res) <= 128) res.inf();
        else                    scratch[tid].inf();
    } else {
        if (sizeof(res) <= 128) res = acc;
        else                    scratch[tid] = acc;
    }
    bucket_t p;
    #pragma unroll 1
    while (i--) {
        p = row[i];
        uint32_t pc = i & mask ? 2 : 0;
        #pragma unroll 1
        do {
            if (sizeof(bucket_t) <= 128) {
                p.add(acc);
                if (pc == 1) {
                    res = p;
                } else {
                    acc = p;
                    if (pc == 0) p = res;
                }
            } else {
                if (LARGE_L1_CODE_CACHE && degree == 1)
                    p.add(acc);
                else
                    p.uadd(acc);
                if (pc == 1) {
                    scratch[tid] = p;
                } else {
                    acc = p;
                    if (pc == 0) p = scratch[tid];
                }
            }
        } while (++pc < 2);
    }
    // same final register state as sppark: p = res (or scratch), acc = suffix total
    if (sizeof(bucket_t) > 128) { __syncthreads(); p = scratch[tid]; }
    bucket_h* out = part + (((size_t)win << (lgT + lgK)) + g) * 2;
    out[0] = p;
    out[1] = acc;
}

// combine: one (window, original thread t0) per thread. res = sum_j res_j + c * sum_j j*acc_j,
// acc = sum_j acc_j, c = L' / (mask+1) (number of res-sampled positions per sub-range).
template<class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(256) __global__
void integrate_combine(bucket_h buckets_[], const bucket_h part[], uint32_t nwins, uint32_t wbits, uint32_t nbits,
                       uint32_t lgT, uint32_t lgK)
{
    const uint32_t degree = bucket_t::degree;
    const uint32_t T = 1u << lgT, K = 1u << lgK;
    uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) / degree;
    if (idx >= nwins * T) return;
    const uint32_t win = idx >> lgT, t0 = idx & (T - 1);
    uint32_t mask = 0;
    if ((win + 1) * wbits > nbits) {
        uint32_t lsbits = nbits - win * wbits;
        mask = (1U << (wbits - lsbits)) - 1;
    }
    const uint32_t lgL1 = wbits - 1 - lgT - lgK;
    const uint32_t lgc = lgL1 - (31 - __clz(mask + 1));
    const bucket_h* pp = part + (((size_t)win << (lgT + lgK)) + (size_t)t0 * K) * 2;
    bucket_t res = pp[2 * (K - 1)], acc = pp[2 * (K - 1) + 1];
    bucket_t run = acc, W; W.inf();
    for (int j = (int)K - 2; j >= 0; j--) {
        bucket_t r = pp[2 * j], a = pp[2 * j + 1];
        W.add(run);          // adds sum_{j' > j} acc_j'
        res.add(r);
        run.add(a);
    }
    acc = run;
    for (uint32_t d = 0; d < lgc; d++) W.add(W);   // W *= c (xyzz add doubles on equal inputs)
    res.add(W);
    bucket_h* row = buckets_ + ((size_t)win << (wbits - 1));
    row[2 * t0] = res;
    row[2 * t0 + 1] = acc;
}

#endif
#undef asm

#ifndef SPPARK_DONT_INSTANTIATE_TEMPLATES
template __global__
void accumulate<bucket_t, affine_t::mem_t>(bucket_t::mem_t buckets_[],
                                           uint32_t nwins, uint32_t wbits,
                                           /*const*/ affine_t::mem_t points_[],
                                           const vec2d_t<uint32_t> digits,
                                           const vec2d_t<uint32_t> histogram,
                                           uint32_t sid);
template __global__
void batch_addition<bucket_t>(bucket_t::mem_t buckets[],
                              const affine_t::mem_t points[], size_t npoints,
                              const uint32_t digits[], const uint32_t& ndigits);
template __global__
void integrate<bucket_t>(bucket_t::mem_t buckets_[], uint32_t nwins,
                         uint32_t wbits, uint32_t nbits);
template __global__
void breakdown<scalar_t>(vec2d_t<uint32_t> digits, const scalar_t scalars[],
                         size_t len, uint32_t nwins, uint32_t wbits, bool mont);
#endif

#include <vector>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

template<class bucket_t, class point_t, class affine_t, class scalar_t,
         class affine_h = class affine_t::mem_t,
         class bucket_h = class bucket_t::mem_t>
class msm_t {
    const gpu_t& gpu;
    size_t npoints;
    uint32_t wbits, nwins;
    size_t batch_override = 0;
    uint32_t tail_sort_y = 0, tail_integ_k = 0;   // set_tail; 0 = sppark's kernels
    int sort_cap = -1;                             // co-resident sort blocks (lazy)
    bucket_h *d_buckets;
    affine_h *d_points;
    scalar_t *d_scalars;
    vec2d_t<uint32_t> d_hist;

    template<typename T> using vec_t = slice_t<T>;

    class result_t {
        bucket_t ret[MSM_NTHREADS/bucket_t::degree][2];
    public:
        result_t() {}
        inline operator decltype(ret)&()                    { return ret;    }
        inline const bucket_t* operator[](size_t i) const   { return ret[i]; }
    };

    constexpr static int lg2(size_t n)
    {   int ret=0; while (n>>=1) ret++; return ret;   }

public:
    // 0 = sppark's heuristic (4 batches for ~2^23 points); 1 = one pass when
    // points and scalars are already on the device (no upload to pipeline).
    void set_batch(size_t b) { batch_override = b; }
    // Groth16 tail (CUDA only, group-element identical results):
    // sort_y > 2 sorts that many windows per cooperative launch instead of pairs (falls back to
    // pairs when they don't fit co-resident); integ_k > 1 (a power of 2) splits each window's
    // integrate over K blocks plus a combine pass. 0 keeps sppark's kernels.
    void set_tail(uint32_t sort_y, uint32_t integ_k) { tail_sort_y = sort_y; tail_integ_k = integ_k; }
    msm_t(const affine_t points[], size_t np,
          size_t ffi_affine_sz = sizeof(affine_t), int device_id = -1)
        : gpu(select_gpu(device_id)), d_points(nullptr), d_scalars(nullptr)
    {
        npoints = (np+WARP_SZ-1) & ((size_t)0-WARP_SZ);

        wbits = 17;
        if (npoints > 192) {
            wbits = std::min(lg2(npoints + npoints/2) - 8, 18);
            if (wbits < 10)
                wbits = 10;
        } else if (npoints > 0) {
            wbits = 10;
        }
        nwins = (scalar_t::bit_length() - 1) / wbits + 1;

        uint32_t row_sz = 1U << (wbits-1);

        size_t d_buckets_sz = (nwins * row_sz)
                            + (gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);
        size_t d_blob_sz = (d_buckets_sz * sizeof(d_buckets[0]))
                         + (nwins * row_sz * sizeof(uint32_t))
                         + (points ? npoints * sizeof(d_points[0]) : 0);

        d_buckets = reinterpret_cast<decltype(d_buckets)>(gpu.Dmalloc(d_blob_sz));
        d_hist = vec2d_t<uint32_t>(&d_buckets[d_buckets_sz], row_sz);
        if (points) {
            d_points = reinterpret_cast<decltype(d_points)>(d_hist[nwins]);
            gpu.HtoD(d_points, points, np, ffi_affine_sz);
            npoints = np;
        } else {
            npoints = 0;
        }

    }
    inline msm_t(vec_t<affine_t> points, size_t ffi_affine_sz = sizeof(affine_t),
                 int device_id = -1)
        : msm_t(points, points.size(), ffi_affine_sz, device_id) {};
    inline msm_t(int device_id = -1)
        : msm_t(nullptr, 0, 0, device_id) {};
    ~msm_t()
    {
        gpu.sync();
        if (d_buckets) gpu.Dfree(d_buckets);
    }

private:
#ifndef __HIPCC__
    // Windows sorted per cooperative launch: tail_sort_y (set_tail) when at least 3 fit
    // co-resident, else 0 = the original pairs.
    uint32_t sort_windows(uint32_t grid_size)
    {
        if (tail_sort_y <= 2) return 0;
        if (sort_cap < 0) {
            int per_sm = 0;
            if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, sort_y, SORT_BLOCKDIM,
                    sizeof(uint32_t) << DIGIT_BITS) != cudaSuccess) {
                (void)cudaGetLastError();
                per_sm = 0;
            }
            sort_cap = per_sm * (int)gpu.sm_count();
        }
        uint32_t y = std::min<uint32_t>(tail_sort_y, (uint32_t)sort_cap / grid_size);
        return y > 2 ? y : 0;
    }
    uint32_t sort_rows()
    {
        uint32_t gs = gpu.sm_count() / 3;
        while (gs & (gs - 1)) gs -= (gs & (0 - gs));
        uint32_t y = sort_windows(gs);
        return y > 2 ? y : 2;
    }
#else
    uint32_t sort_rows() { return 2; }
#endif
    void digits(const scalar_t d_scalars[], size_t len,
                vec2d_t<uint32_t>& d_digits, vec2d_t<uint2>&d_temps, bool mont)
    {
        // Using larger grid size doesn't make 'sort' run faster, actually
        // quite contrary. Arguably because global memory bus gets
        // thrashed... Stepping far outside the sweet spot has significant
        // impact, 30-40% degradation was observed. It's assumed that all
        // GPUs are "balanced" in an approximately the same manner. The
        // coefficient was observed to deliver optimal performance on
        // Turing and Ampere...
        uint32_t grid_size = gpu.sm_count() / 3;
        while (grid_size & (grid_size - 1))
            grid_size -= (grid_size & (0 - grid_size));

        breakdown<<<2*grid_size, 1024, sizeof(scalar_t)*1024, gpu[2]>>>(
            d_digits, d_scalars, len, nwins, wbits, mont
        );
        CUDA_OK(cudaGetLastError());

        const size_t shared_sz = sizeof(uint32_t) << DIGIT_BITS;
#if 0
        uint32_t win;
        for (win = 0; win < nwins-1; win++) {
            gpu[2].launch_coop(sort, {grid_size, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, wbits-1, 0u);
        }
        uint32_t top = scalar_t::bit_length() - wbits * win;
        gpu[2].launch_coop(sort, {grid_size, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, top-1, 0u);
#else
        // On the other hand a pair of kernels launched in parallel run
        // ~50% slower but sort twice as much data...
        uint32_t top = scalar_t::bit_length() - wbits * (nwins-1);
#ifndef __HIPCC__
        if (uint32_t Y = sort_windows(grid_size)) {
            for (uint32_t w0 = 0; w0 < nwins; w0 += Y) {
                uint32_t y = std::min(Y, nwins - w0);
                gpu[2].launch_coop(sort_y, {{grid_size, y}, SORT_BLOCKDIM, shared_sz},
                                d_digits, len, w0, d_temps, d_hist,
                                wbits-1, nwins-1, top-1);
            }
            return;
        }
#endif
        uint32_t win;
        for (win = 0; win < nwins-1; win += 2) {
            gpu[2].launch_coop(sort, {{grid_size, 2}, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, wbits-1, win == nwins-2 ? top-1 : wbits-1);
        }
        if (win < nwins) {
            gpu[2].launch_coop(sort, {{grid_size, 1}, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, top-1, 0u);
        }
#endif
    }

public:
    // Deferred host collect. invoke_defer() runs every GPU step and the DtoH of the
    // per-window results, then returns; finish() does the host-side collect() later, e.g. on
    // another thread while the next MSM runs on the GPU. Same arithmetic, same result.
    // finish() may run concurrently with invoke()/invoke_defer() on the same msm_t: collect()
    // reads only nwins, wbits and gpu, which are fixed at construction.
    struct pending_t {
        std::vector<result_t> res;
        std::vector<bucket_t> ones;
        point_t partial;
    };
    void finish(point_t& out, pending_t& d)
    {
        point_t p;
        collect(p, d.res, d.ones);
        out = d.partial;
        out.add(p);
    }

    RustError invoke(point_t& out, const affine_t* points_, size_t npoints,
                                   const scalar_t* scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t),
                                   pending_t* defer = nullptr)
    {
        assert(this->npoints == 0 || npoints <= this->npoints);

        uint32_t lg_npoints = lg2(npoints + npoints/2);
        size_t batch = 1 << (std::max(lg_npoints, wbits) - wbits);
        batch >>= 6;
        batch = batch ? batch : 1;
        if (batch_override) batch = batch_override;
        uint32_t stride = (npoints + batch - 1) / batch;
        stride = (stride+WARP_SZ-1) & ((size_t)0-WARP_SZ);

        std::vector<result_t> res(nwins);
        std::vector<bucket_t> ones(gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);

        out.inf();
        point_t p;

        try {
            // |scalars| being nullptr means the scalars are pre-loaded to
            // |d_scalars|, otherwise allocate stride.
            size_t temp_sz = scalars ? sizeof(scalar_t) : 0;
            temp_sz = stride * std::max(sort_rows()*sizeof(uint2), temp_sz);

            // |points| being nullptr means the points are pre-loaded to
            // |d_points|, otherwise allocate double-stride.
            const char* points = reinterpret_cast<const char*>(points_);
            size_t d_point_sz = points ? (batch > 1 ? 2*stride : stride) : 0;
            d_point_sz *= sizeof(affine_h);

            size_t digits_sz = nwins * stride * sizeof(uint32_t);

            // Split integrate with K = 2^lgK sub-ranges per window (set_tail), when every
            // sub-range still covers at least two buckets per thread.
            const uint32_t lgT = lg2(MSM_NTHREADS/bucket_t::degree);
            uint32_t lgK = 0;
#ifndef __HIPCC__
            {
                uint32_t lk = tail_integ_k > 1 ? lg2(tail_integ_k) : 0;
                uint32_t top_ls = scalar_t::bit_length() - (nwins-1)*wbits;
                if (lk && (wbits-1) >= lgT+lk+1 && top_ls-1 >= lgT+lk)
                    lgK = lk;
            }
#endif
            size_t part_off = temp_sz + digits_sz + d_point_sz;
            part_off = (part_off + 255) & ~(size_t)255;
            size_t part_sz = lgK ? ((size_t)nwins << (lgT+lgK)) * 2 * sizeof(bucket_h) : 0;

            dev_ptr_t<uint8_t> d_temp{part_off + part_sz, gpu[2]};
            bucket_h* d_part = lgK ? (bucket_h*)&d_temp[part_off] : nullptr;

            vec2d_t<uint2> d_temps{&d_temp[0], stride};
            vec2d_t<uint32_t> d_digits{&d_temp[temp_sz], stride};

            scalar_t* d_scalars = scalars ? (scalar_t*)&d_temp[0]
                                          : this->d_scalars;
            affine_h* d_points = points ? (affine_h*)&d_temp[temp_sz + digits_sz]
                                        : this->d_points;

            size_t d_off = 0;   // device offset
            size_t h_off = 0;   // host offset
            size_t num = stride > npoints ? npoints : stride;
            event_t ev;

            if (scalars)
                gpu[2].HtoD(&d_scalars[d_off], &scalars[h_off], num);
            digits(&d_scalars[0], num, d_digits, d_temps, mont);
            gpu[2].record(ev);

            if (points)
                gpu[0].HtoD(&d_points[d_off], &points[h_off],
                            num,              ffi_affine_sz);

            for (uint32_t i = 0; i < batch; i++) {
                gpu[i&1].wait(ev);

                batch_addition<bucket_t><<<gpu.sm_count(), BATCH_ADD_BLOCK_SIZE,
                                           0, gpu[i&1]>>>(
                    &d_buckets[nwins << (wbits-1)], &d_points[d_off], num,
                    &d_digits[0][0], d_hist[0][0]
                );
                CUDA_OK(cudaGetLastError());

                gpu[i&1].launch_coop(accumulate<bucket_t, affine_h>,
                    {gpu.sm_count(), 0},
                    d_buckets, nwins, wbits, &d_points[d_off], d_digits, d_hist, i&1
                );
                gpu[i&1].record(ev);

#ifndef __HIPCC__
                if (lgK) {
                    integrate_split<bucket_t><<<nwins << lgK, MSM_NTHREADS,
                                      sizeof(bucket_t)*MSM_NTHREADS/bucket_t::degree,
                                      gpu[i&1]>>>(
                        d_part, d_buckets, wbits, scalar_t::bit_length(), lgK
                    );
                    CUDA_OK(cudaGetLastError());
                    uint32_t nthr = (nwins << lgT) * bucket_t::degree;
                    integrate_combine<bucket_t><<<(nthr + 127) / 128, 128, 0, gpu[i&1]>>>(
                        d_buckets, d_part, nwins, wbits, scalar_t::bit_length(), lgT, lgK
                    );
                    CUDA_OK(cudaGetLastError());
                } else
#endif
                {
                integrate<bucket_t><<<nwins, MSM_NTHREADS,
                                      sizeof(bucket_t)*MSM_NTHREADS/bucket_t::degree,
                                      gpu[i&1]>>>(
                    d_buckets, nwins, wbits, scalar_t::bit_length()
                );
                CUDA_OK(cudaGetLastError());
                }

                if (i < batch-1) {
                    h_off += stride;
                    num = h_off + stride <= npoints ? stride : npoints - h_off;

                    if (scalars)
                        gpu[2].HtoD(&d_scalars[0], &scalars[h_off], num);
                    gpu[2].wait(ev);
                    digits(&d_scalars[scalars ? 0 : h_off], num,
                           d_digits, d_temps, mont);
                    gpu[2].record(ev);

                    if (points) {
                        size_t j = (i + 1) & 1;
                        d_off = j ? stride : 0;
                        gpu[j].HtoD(&d_points[d_off], &points[h_off*ffi_affine_sz],
                                    num,              ffi_affine_sz);
                    } else {
                        d_off = h_off;
                    }
                }

                if (i > 0) {
                    collect(p, res, ones);
                    out.add(p);
                }

                gpu[i&1].DtoH(ones, d_buckets + (nwins << (wbits-1)));
                gpu[i&1].DtoH(res, d_buckets, sizeof(bucket_h)<<(wbits-1));
                gpu[i&1].sync();
            }
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        if (defer) {
            defer->res = std::move(res);
            defer->ones = std::move(ones);
            defer->partial = out;
            return RustError{cudaSuccess};
        }
        collect(p, res, ones);
        out.add(p);

        return RustError{cudaSuccess};
    }

    template<typename affine_ptr_t = const affine_h*,
             typename scalar_ptr_t = const scalar_t*>
    RustError invoke_defer(pending_t& pend, affine_ptr_t points, size_t npoints,
                           scalar_ptr_t scalars, bool mont = true)
    {
        const auto* p_ptr = &points[0];
        if (is_device_ptr<affine_ptr_t>::value) {
            d_points = (decltype(d_points))p_ptr;
            p_ptr = nullptr;
        }
        const auto* s_ptr = &scalars[0];
        if (is_device_ptr<scalar_ptr_t>::value) {
            d_scalars = const_cast<decltype(d_scalars)>(s_ptr);
            s_ptr = nullptr;
        }
        point_t dummy;
        return invoke(dummy, p_ptr, npoints, s_ptr, mont, sizeof(affine_t), &pend);
    }

#if 0
    RustError invoke(point_t& out, const affine_t* points, size_t npoints,
                                   gpu_ptr_t<scalar_t> scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        d_scalars = scalars;
        return invoke(out, points, npoints, nullptr, mont, ffi_affine_sz);
    }
#else
    template<typename affine_ptr_t = const affine_h*,
             typename scalar_ptr_t = const scalar_t*>
    RustError invoke(point_t& out, affine_ptr_t points, size_t npoints,
                                   scalar_ptr_t scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        const auto* p_ptr = &points[0];
        if (is_device_ptr<affine_ptr_t>::value) {
            d_points = (decltype(d_points))p_ptr;
            p_ptr = nullptr;
        }

        const auto* s_ptr = &scalars[0];
        if (is_device_ptr<scalar_ptr_t>::value) {
            d_scalars = const_cast<decltype(d_scalars)>(s_ptr);
            s_ptr = nullptr;
        }

        return invoke(out, p_ptr, npoints, s_ptr, mont, ffi_affine_sz);
    }
#endif

    RustError invoke(point_t& out, vec_t<scalar_t> scalars, bool mont = true)
    {   return invoke(out, nullptr, scalars.size(), scalars, mont);   }

    RustError invoke(point_t& out, vec_t<affine_t> points,
                                   const scalar_t* scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {   return invoke(out, points, points.size(), scalars, mont, ffi_affine_sz);   }

    RustError invoke(point_t& out, vec_t<affine_t> points,
                                   vec_t<scalar_t> scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {   return invoke(out, points, points.size(), scalars, mont, ffi_affine_sz);   }

    RustError invoke(point_t& out, const std::vector<affine_t>& points,
                                   const std::vector<scalar_t>& scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        return invoke(out, points.data(),
                           std::min(points.size(), scalars.size()),
                           scalars.data(), mont, ffi_affine_sz);
    }

private:
    point_t integrate_row(const result_t& row, uint32_t lsbits)
    {
        const int NTHRBITS = lg2(MSM_NTHREADS/bucket_t::degree);

        assert(wbits-1 > NTHRBITS);

        size_t i = MSM_NTHREADS/bucket_t::degree - 1;

        if (lsbits-1 <= NTHRBITS) {
            size_t mask = (1U << (NTHRBITS-(lsbits-1))) - 1;
            bucket_t res, acc = row[i][1];

            if (mask)   res.inf();
            else        res = acc;

            while (i--) {
                acc.add(row[i][1]);
                if ((i & mask) == 0)
                    res.add(acc);
            }

            return res;
        }

        point_t  res = row[i][0];
        bucket_t acc = row[i][1];

        while (i--) {
            point_t raise = acc;
            for (size_t j = 0; j < lsbits-1-NTHRBITS; j++)
                raise.dbl();
            res.add(raise);
            res.add(point_t{row[i][0]});
            if (i)
                acc.add(row[i][1]);
        }

        return res;
    }

    void collect(point_t& out, const std::vector<result_t>& res,
                               const std::vector<bucket_t>& ones)
    {
        struct tile_t {
            uint32_t x, y, dy;
            point_t p;
            tile_t() {}
        };
        std::vector<tile_t> grid(nwins);

        uint32_t y = nwins-1, total = 0;

        grid[0].x  = 0;
        grid[0].y  = y;
        grid[0].dy = scalar_t::bit_length() - y*wbits;
        total++;

        while (y--) {
            grid[total].x  = grid[0].x;
            grid[total].y  = y;
            grid[total].dy = wbits;
            total++;
        }

        std::vector<std::atomic<size_t>> row_sync(nwins); /* zeroed */
        counter_t<size_t> counter(0);
        channel_t<size_t> ch;

        auto n_workers = min((uint32_t)gpu.ncpus(), total);
        while (n_workers--) {
            gpu.spawn([&, this, total, counter]() {
                for (size_t work; (work = counter++) < total;) {
                    auto item = &grid[work];
                    auto y = item->y;
                    item->p = integrate_row(res[y], item->dy);
                    if (++row_sync[y] == 1)
                        ch.send(y);
                }
            });
        }

        point_t one = sum_up(ones);

        out.inf();
        size_t row = 0, ny = nwins;
        while (ny--) {
            auto y = ch.recv();
            row_sync[y] = -1U;
            while (grid[row].y == y) {
                while (row < total && grid[row].y == y)
                    out.add(grid[row++].p);
                if (y == 0)
                    break;
                for (size_t i = 0; i < wbits; i++)
                    out.dbl();
                if (row_sync[--y] != -1U)
                    break;
            }
        }
        out.add(one);
    }
};

template<class bucket_t, class point_t, class affine_t, class scalar_t> static
RustError mult_pippenger(point_t *out, const affine_t points[], size_t npoints,
                                       const scalar_t scalars[], bool mont = true,
                                       size_t ffi_affine_sz = sizeof(affine_t))
{
    try {
        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
        return msm.invoke(*out, slice_t<affine_t>{points, npoints},
                                scalars, mont, ffi_affine_sz);
    } catch (const cuda_error& e) {
        out->inf();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}
#endif
