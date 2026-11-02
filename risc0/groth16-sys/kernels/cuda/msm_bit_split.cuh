#pragma once
// Groth16 bit-split MSMs (CUDA only).
// MSM(P, w) = MSM(P_nb, w_nb + f_nb) - MSM(P_nb, f_nb) + sum_{i in S, w_i = 1} P_i
// S: witness indices that are 0/1 by construction (circom Band(x, 1) outputs). The first term
// keeps the fuzz (random scalars avoid bucket pile-ups in sppark's accumulate: the unfuzzed
// non-bit scalars repeat values like p-1 thousands of times, 3x slower); the second term is
// precomputed once. A device check counts S entries that are not 0/1 and the caller falls
// back to the full MSM path.
#ifndef __HIPCC__
#include <fstream>
#include <vector>

template <int W> struct split_blob_t { uint4 v[W]; };

template <int BYTES>
__global__ void split_gather_pts(void* out_, const void* in_, const uint32_t* idx, size_t n) {
  typedef split_blob_t<BYTES / 16> B;
  B* out = (B*)out_;
  const B* in = (const B*)in_;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    out[i] = in[idx[i]];
}

// out[i] = w[idx[i]] + f[idx[i]] (f may be null)
__global__ void split_gather_add(fr_t* out, const fr_t* w, const fr_t* f, const uint32_t* idx, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint32_t j = idx[i];
    fr_t a = w[j];
    if (f) a += f[j];
    out[i] = a;
  }
}

// Partial sums (one per logical thread) of the gathered S points whose scalar is 1; scalars
// are canonical (mont=false in the prover). Counts scalars not in {0,1} into *bad.
template <class bucket_t,
          class affine_h = typename bucket_t::affine_t::mem_t,
          class bucket_h = typename bucket_t::mem_t>
__global__ void split_bit_sum(bucket_h* out, int* bad, const affine_h* pts, const fr_t* w,
                           const uint32_t* idx, size_t n) {
#ifdef __CUDA_ARCH__
  typedef typename bucket_t::affine_t affine_t;
  const uint32_t deg = bucket_t::degree;
  size_t lt = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) / deg;
  size_t nlt = (gridDim.x * (size_t)blockDim.x) / deg;
  bucket_t acc;
  acc.inf();
  int nbad = 0;
  for (size_t i = lt; i < n; i += nlt) {
    const uint4* s = (const uint4*)&w[idx[i]];
    uint4 a = s[0], b = s[1];
    nbad += !((a.x <= 1) & ((a.y | a.z | a.w | b.x | b.y | b.z | b.w) == 0));
    if (a.x == 1) {
      affine_t p = pts[i];
      acc.add(p);
    }
  }
  out[lt] = acc;
  if (nbad && (threadIdx.x % deg) == 0)
    atomicAdd(bad, nbad);
#endif
}

template <class bucket_t, class bucket_h = typename bucket_t::mem_t>
__global__ void split_reduce_step(bucket_h* buf, size_t half, size_t cnt) {
#ifdef __CUDA_ARCH__
  size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) / bucket_t::degree;
  if (i < cnt) {
    bucket_t a;
    a = buf[i];
    bucket_t b;
    b = buf[i + half];
    a.add(b);
    buf[i] = a;
  }
#endif
}

// One MSM's split state: views into the in-place permuted points and -MSM(P_nb, f_nb).
template <class bucket_t, class point_t, class affine_t>
struct msm_split_t {
  affine_t* pn = nullptr;           // non-bit points: pts[0, nn)
  affine_t* pb = nullptr;           // bit points: pts[nn, n)
  gpu_ptr_t<typename bucket_t::mem_t> acc;
  size_t nthr = 0;
  point_t corr;                     // -MSM(P_nb, f_nb)

  // Allocates the bit-sum buffer. DEG: lanes per point on the device (1 for G1, 2 for G2); the
  // host's bucket_t::degree is 1 for both, so it is passed explicitly. Over-allocated 2x.
  template <int DEG>
  void alloc(const gpu_t& gpu) {
    nthr = (size_t)gpu.sm_count() * 2 * 256 / DEG;
    acc = gpu_ptr_t<typename bucket_t::mem_t>(
        (typename bucket_t::mem_t*)gpu.Dmalloc(2 * nthr * sizeof(bucket_t)));
  }

  // Permutes pts[0,n) in place to pts[perm[i]] (perm = [non-bit | bit] indices) through `tmp`
  // (room for n points), all on `gpu`'s stream. Allocates nothing.
  template <int PTB>
  void permute(const gpu_t& gpu, affine_t* pts, const uint32_t* d_perm, size_t n, size_t nn,
               void* tmp) {
    split_gather_pts<PTB><<<gpu.sm_count() * 4, 256, 0, gpu>>>(tmp, pts, d_perm, n);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpyAsync(pts, tmp, n * sizeof(affine_t), cudaMemcpyDeviceToDevice, gpu));
    pn = pts;
    pb = pts + nn;
  }

  // launches the bit sum + reduction on `gpu`; acc[0] holds the sum
  template <int DEG>
  void bit_sum(const gpu_t& gpu, int* d_bad, const fr_t* w, const uint32_t* d_bidx, size_t nb) {
    split_bit_sum<bucket_t><<<gpu.sm_count() * 2, 256, 0, gpu>>>(
        &acc[0], d_bad, (const typename affine_t::mem_t*)pb, w, d_bidx, nb);
    for (size_t m = nthr; m > 1;) {
      size_t h = (m + 1) / 2;
      split_reduce_step<bucket_t><<<(h * DEG + 255) / 256, 256, 0, gpu>>>(&acc[0], h, m - h);
      m = h;
    }
    CUDA_OK(cudaGetLastError());
  }
};
#endif
