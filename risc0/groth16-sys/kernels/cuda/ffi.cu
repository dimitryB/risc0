#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

#define FEATURE_BN254

#include "ff/alt_bn128-fp2.hpp"

#include "ec/affine_t.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbitwise-instead-of-logical"
#include "ec/jacobian_t.hpp"
#pragma clang diagnostic pop

#include "ec/xyzz_t.hpp"

typedef Affine_t<fp_t> affine_t;
typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;

typedef Affine_t<fp2_t> affine_fp2_t;
typedef jacobian_t<fp2_t> point_fp2_t;
typedef xyzz_t<fp2_t> bucket_fp2_t;

typedef fr_t scalar_t;

#ifndef __HIPCC__
#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#endif
#ifdef __HIPCC__
#include "msm/pippenger.cuh" // msm_groth16 is CUDA-only (inline PTX, <cuda.h>)
#else
#include "msm_groth16/pippenger.cuh"
#endif

#ifdef __HIPCC__
// G2 MSM kernel explicit instantiations (G1 handled by pippenger.cuh defaults).
// All four template arguments are specified explicitly so that the mangled names
// match the call sites in msm_t<bucket_fp2_t, point_fp2_t, affine_fp2_t, fr_t>.
template __global__
void accumulate<bucket_fp2_t, affine_fp2_t::mem_t,
                bucket_fp2_t::mem_t, affine_fp2_t>(
    bucket_fp2_t::mem_t buckets_[],
    uint32_t nwins, uint32_t wbits,
    /*const*/ affine_fp2_t::mem_t points_[],
    const vec2d_t<uint32_t> digits,
    const vec2d_t<uint32_t> histogram,
    uint32_t* counter);
template __global__
void batch_addition<bucket_fp2_t, affine_fp2_t::mem_t,
                    bucket_fp2_t::mem_t, affine_fp2_t>(
    bucket_fp2_t::mem_t buckets[],
    const affine_fp2_t::mem_t points[], size_t npoints,
    const uint32_t digits[], const uint32_t& ndigits);
template __global__
void integrate<bucket_fp2_t, bucket_fp2_t::mem_t>(
    bucket_fp2_t::mem_t buckets_[], uint32_t nwins,
    uint32_t wbits, uint32_t nbits);
template __global__
void reduce_rows<bucket_fp2_t, bucket_fp2_t::mem_t>(
    bucket_fp2_t::mem_t buckets_[], uint32_t nwins,
    uint32_t wbits, uint32_t nbits,
    uint32_t thr_per_sub);
#endif

namespace sppark::bn254 {
#include "ntt/ntt.cuh"
} // namespace sppark::bn254
using namespace sppark::bn254;

#include "util.cuh"

#include "pinned_upload.cuh"
#include "groth16_coeffs.cuh"
#include "groth16_srs.cuh"
#include "groth16_prover.cuh"

#ifdef __HIPCC__
// Explicit template instantiations for HIP device pass.
// The groth16_prover host code is hidden from the device pass (it uses
// blst-specific methods), so template __global__ functions called from it
// must be explicitly instantiated here.

// NTT kernels (inside the sppark::bn254 namespace where ntt.cuh was included)
namespace sppark::bn254 {
template __global__
void _GS_NTT<0, fr_t>(const unsigned int, const unsigned int,
    const unsigned int, const unsigned int,
    fr_t*, const fr_t (*)[WINDOW_SIZE],
    const fr_t*, const fr_t*, const fr_t*,
    const unsigned int, const bool, const fr_t,
    const unsigned int);
template __global__
void _GS_NTT<1, fr_t>(const unsigned int, const unsigned int,
    const unsigned int, const unsigned int,
    fr_t*, const fr_t (*)[WINDOW_SIZE],
    const fr_t*, const fr_t*, const fr_t*,
    const unsigned int, const bool, const fr_t,
    const unsigned int);
template __global__
void _GS_NTT<2, fr_t>(const unsigned int, const unsigned int,
    const unsigned int, const unsigned int,
    fr_t*, const fr_t (*)[WINDOW_SIZE],
    const fr_t*, const fr_t*, const fr_t*,
    const unsigned int, const bool, const fr_t,
    const unsigned int);
template __global__
void _CT_NTT<0, fr_t>(const unsigned int, const unsigned int,
    const unsigned int, const unsigned int,
    fr_t*, const fr_t (*)[WINDOW_SIZE],
    const fr_t*, const fr_t*, const fr_t*,
    const unsigned int, const bool, const fr_t,
    const unsigned int);
template __global__
void _CT_NTT<1, fr_t>(const unsigned int, const unsigned int,
    const unsigned int, const unsigned int,
    fr_t*, const fr_t (*)[WINDOW_SIZE],
    const fr_t*, const fr_t*, const fr_t*,
    const unsigned int, const bool, const fr_t,
    const unsigned int);
template __global__
void _CT_NTT<2, fr_t>(const unsigned int, const unsigned int,
    const unsigned int, const unsigned int,
    fr_t*, const fr_t (*)[WINDOW_SIZE],
    const fr_t*, const fr_t*, const fr_t*,
    const unsigned int, const bool, const fr_t,
    const unsigned int);
template __global__
void bit_rev_permutation<fr_t>(fr_t*, const fr_t*, uint32_t);
template __global__
void LDE_distribute_powers<fr_t>(fr_t*, uint32_t, uint32_t, bool,
    const fr_t (*)[WINDOW_SIZE], const unsigned int);
template __global__
void generate_partial_twiddles<fr_t>(fr_t (*)[WINDOW_SIZE], const fr_t);
template __global__
void generate_all_twiddles<fr_t>(fr_t*, const fr_t);
template __global__
void generate_radixX_twiddles_X<fr_t>(fr_t*, int, const fr_t);
} // namespace sppark::bn254

// Utility kernels
template __global__
void chacha_generate_random_scalars<8, fr_t>(fr_t*, const chacha_state, size_t);
#endif // __HIPCC__

#if !defined(__HIP_DEVICE_COMPILE__)
struct SetupParams {
  const char* pcoeffs_path;
  const char* fres_path;
  const char* srs_path;
};

struct ProveParams {
  const char* public_path;
  const char* proof_path;
  const fr_t* witness;
};

// Per-device cached SRS + prover to avoid reloading ~200MB of point data and
// reinitializing GPU memory on every Groth16 prove call.
// Preloaded on a background thread via risc0_groth16_cuda_preload().
// Raw pointers: intentionally leaked at exit to avoid CUDA driver teardown crash.
//
// Locking: `cached_mutex` guards the pointers themselves. `prove_mutex[dev]` is held for a whole
// prove call, while it uses the cached prover outside `cached_mutex`, and for a whole release,
// so release waits for a prove in flight instead of freeing the prover under it (and two proves
// on one device can't share the prover at once). Lock order: prove_mutex, then cached_mutex.
static constexpr int kMaxDevices = 16;
static SRS* cached_srs[kMaxDevices] = {};
static groth16_prover* cached_prover[kMaxDevices] = {};
static std::mutex cached_mutex;
static std::mutex prove_mutex[kMaxDevices];

// Builds the cached prover for `dev` on its cached SRS (caller holds cached_mutex). A prover whose
// construction throws after it started permuting the SRS points in place (the split MSMs, see
// groth16_prover::split_init) leaves them permuted: drop that SRS too, so the next prove reloads
// it instead of failing on (or building a prover over) permuted points.
static void build_cached_prover(int dev, const SetupParams* setup_params) {
  try {
    cached_prover[dev] = new groth16_prover(
        *cached_srs[dev], setup_params->pcoeffs_path, setup_params->fres_path);
  } catch (...) {
    if (cached_srs[dev]->split_permuted) {
      // Work queued on the SRS buffers (the permutation) must finish before they are freed.
      (void)cudaDeviceSynchronize();
      delete cached_srs[dev];
      cached_srs[dev] = nullptr;
    }
    throw;
  }
}

// The current device, checked to index the per-device caches.
static int current_device() {
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("groth16: cudaGetDevice failed: ") +
                             cudaGetErrorString(err));
  }
  if (dev < 0 || dev >= kMaxDevices) {
    throw std::runtime_error("groth16: device " + std::to_string(dev) +
                             " is beyond the supported " + std::to_string(kMaxDevices));
  }
  return dev;
}

// A pinned host buffer of at least `bytes` (one per process, grown as needed), for the
// witness: it uploads in one fast DMA. THP-backed (ThpPinned): pinning 180 MB takes ~13 ms instead
// of ~40-140 ms of cudaHostAlloc holding the driver lock. The Rust caller
// (risc0_groth16_sys::host_buffer) serializes its users; growing frees the old buffer.
static ThpPinned g16_witness_buf;
static std::mutex g16_witness_buf_mu;

extern "C" void* risc0_groth16_cuda_host_buffer(size_t bytes) {
  std::lock_guard<std::mutex> lock(g16_witness_buf_mu);
  if (bytes > g16_witness_buf.bytes) {
    try {
      g16_witness_buf.alloc(bytes);
    } catch (...) {
      g16_witness_buf.free_pinned();
      return nullptr;
    }
  }
  return g16_witness_buf.ptr;
}

// Create the pinned upload stager (130 MB, streams, events) now, at the start of a Groth16
// session while the STARK's first segment is still host-bound, instead of when the key loader
// starts (by then the STARK is GPU-bound, and the registration stalls its launches). Skipped
// (*created = 0) on a card with less than `min_total_vram` bytes, where the key isn't prestarted.
extern "C" const char* risc0_groth16_cuda_stager_init(size_t min_total_vram, int* created) {
  *created = 0;
  try {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b < min_total_vram)
      return nullptr;
    (void)key_stager::get();
    *created = 1;
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Free what risc0_groth16_cuda_stager_init and an early witness-buffer pin allocated, when
// the key prestart gave up (both are recreated on demand). The Rust caller holds the witness
// buffer's lock.
extern "C" void risc0_groth16_cuda_pinned_release() {
  key_stager* stager;
  {
    std::lock_guard<std::mutex> lock(cached_mutex);
    stager = key_stager::take();
  }
  // Outside cached_mutex: unpinning waits for the whole device (ThpPinned::free_pinned), and a
  // preload for another device should not wait for that.
  delete stager;
  std::lock_guard<std::mutex> lock(g16_witness_buf_mu);
  g16_witness_buf.free_pinned();
}

extern "C" const char* risc0_groth16_cuda_preload(SetupParams* setup_params) {
  try {
    int dev = current_device();
    std::lock_guard<std::mutex> lock(cached_mutex);
    if (!cached_srs[dev]) {
      cached_srs[dev] = new SRS(dev, setup_params->srs_path);
    }
    if (!cached_prover[dev]) {
      build_cached_prover(dev, setup_params);
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// The witness indices that are always 0/1 (the bit set of the split MSMs), used by the next
// prover construction.
extern "C" void risc0_groth16_cuda_set_bit_signals(const uint32_t* bits, size_t len) {
  std::lock_guard<std::mutex> lk(g16_bits_mu);
  g16_bits().assign(bits, bits + len);
}

// Load the proving key (SRS points + preprocessed coeffs) and build the prover in the
// background, e.g. while the STARK proves. Uploads go through the pinned stager on its own
// non-blocking streams (never pageable copies: those take the driver lock for their whole
// duration and stall every other thread's CUDA calls). Skipped (loaded = 0) on a card with less
// than `min_total_vram` bytes, where the key cannot sit beside a STARK proof.

extern "C" const char* risc0_groth16_cuda_prestart(SetupParams* setup_params,
                                                   size_t min_total_vram,
                                                   size_t* witness_len,
                                                   int* loaded) {
  *loaded = 0;
  *witness_len = 0;
  try {
    int dev = current_device();
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b < min_total_vram)
      return nullptr;
    g16_staged_upload = true;
    std::lock_guard<std::mutex> lock(cached_mutex);
    if (!cached_srs[dev]) {
      cached_srs[dev] = new SRS(dev, setup_params->srs_path);
    }
    if (!cached_prover[dev]) {
      build_cached_prover(dev, setup_params);
    }
    *witness_len = cached_srs[dev]->get_a_size();
    *loaded = 1;
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

extern "C" const char* risc0_groth16_cuda_prove(SetupParams* setup_params,
                                                ProveParams* prover_params) {

  try {
    int dev = current_device();
    std::lock_guard<std::mutex> prove_lock(prove_mutex[dev]);
    groth16_prover* prover = nullptr;
    // Release cached async memory pool allocations from prior GPU work (e.g.
    // STARK proving) so that the Groth16 prover has enough VRAM.
    // Only needed on first call (before SRS/prover are cached).
    {
      std::lock_guard<std::mutex> lock(cached_mutex);
      if (!cached_prover[dev]) {
        cudaMemPool_t pool;
        if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess)
          cudaMemPoolTrimTo(pool, 0);
      }

      if (!cached_srs[dev]) {
        cached_srs[dev] = new SRS(dev, setup_params->srs_path);
      }
      if (!cached_prover[dev]) {
        build_cached_prover(dev, setup_params);
      }
      prover = cached_prover[dev];
    }

    // Outside cached_mutex, so preload for another device isn't blocked for the proof; the prover
    // stays alive because release needs prove_mutex[dev], which this call holds.
    groth16_proof proof = prover->prove(prover_params->public_path, prover_params->witness);
    write_proof_file(prover_params->proof_path, proof);
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Free this device's cached prover and SRS, returning their GPU memory. The next prove call rebuilds
// them. Measured on 2026-10-07 that adds at most ~0.1 s: an RTX 5080 releasing after every proof wrapped
// in 2.08-2.12 s, an RTX 4090 keeping its cache in 2.01-2.05 s. (Either card's FIRST wrap in a process is
// ~0.9 s slower, mostly the first read of the 3.45 GB proving key; that is paid once either way.)
//
// Nothing else ever frees the cache, so without this it occupies the card for the life of the process. On a 16 GB card the next STARK proof has to fit beside it: measured on an
// RTX 5080, a 46M-cycle proof at po2 20 peaked at 15.8 of 16.3 GB with the cache resident, and failed
// outright in a different program order. Callers without that room call this after each proof.
extern "C" const char* risc0_groth16_cuda_release() {
  try {
    int dev = current_device();
    // Waits for a prove in flight on this device to finish with the prover.
    std::lock_guard<std::mutex> prove_lock(prove_mutex[dev]);
    // A prove that threw midway can leave kernels queued that use the prover's buffers, and
    // their frees (cudaFreeAsync on the legacy stream) aren't ordered after sppark's
    // non-blocking streams: let that work finish first.
    cudaDeviceSynchronize();
    {
      // Only around the pointers: the syncs wait for all of the device's work (a STARK on
      // another thread, say), and preload for other devices shouldn't wait for that.
      std::lock_guard<std::mutex> lock(cached_mutex);
      // The prover first: it was constructed from a reference to the SRS.
      delete cached_prover[dev];
      cached_prover[dev] = nullptr;
      delete cached_srs[dev];
      cached_srs[dev] = nullptr;
    }
    // gpu_ptr_t may free with cudaFreeAsync, so wait for those frees to reach the pool before
    // handing the pool's reservation back to the driver.
    cudaDeviceSynchronize();
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess)
      cudaMemPoolTrimTo(pool, 0);
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

#ifdef SRS_READ_COEFFS

extern "C" const char* risc0_groth16_cuda_setup(SetupParams* params) {
  try {
    SRS sw(0, params->srs_path);
    groth16_prover prover(sw);
    prover.write_precomputations_to_file(params->fres_path, params->pcoeffs_path);
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

#endif // SRS_READ_COEFFS
#endif // !__HIP_DEVICE_COMPILE__
