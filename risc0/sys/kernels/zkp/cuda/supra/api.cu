#include <cstdlib>
#include <ff/alt_bn128.hpp>
#include <ff/baby_bear.hpp>
#include <util/gpu_t.cuh>
#include <util/rusterror.h>

#include <polynomial/div_by_x_minus_z.cuh>
#include <polynomial/prefix_op.cuh>

#include "poseidon2.cuh"
// The lazy-reduction Poseidon2 kernels (pv.cuh), for CUDA and ROCm.
#include "pv.cuh"
#define PV_BLOCK 128
#include "mdiv.cuh"
#ifndef __HIPCC__
// Latency-bound small levels / row counts: warp- and quad-per-permutation kernels.
#include "pv_small.cuh"
#endif

#include <cstdlib>
#include <cstring>

// Whether to hash with pv.cuh's kernels rather than poseidon2.cuh's: always on CUDA; on ROCm
// unless RISC0_HIP_POSEIDON2=classic.
static bool use_pv_kernels() {
#ifdef __HIPCC__
  static const bool pv = [] {
    const char* v = std::getenv("RISC0_HIP_POSEIDON2");
    return !(v && std::strcmp(v, "classic") == 0);
  }();
  return pv;
#else
  return true;
#endif
}

#ifdef __HIPCC__
#include "../hip_hop.h"
// pv.cuh's kernels from their own TU (supra/pv_hip.cu, built with the max-ilp scheduler: 2-5% faster
// on RDNA3/4), and the top layers of a fold tree in one launch (fold_tail_k).
// RISC0_HIP_PV_TU=0: this file's instances (default scheduler) only; otherwise the max-ilp TU for
// every launch. (gfx12 used to run the default-scheduler instances for 2^19..2^22 permutations,
// for a ~0.27 ms idle seen after max-ilp kernels of >= ~1.5 ms on an RX 9070 XT; with the gfx12 hop
// queues it no longer follows these launches, and the max-ilp fold_k / rows_k are 2.5-4% faster.)
// RISC0_HIP_PV_TAIL=0: one launch per fold layer.
// Folds of at most 2^10 digests and the tail run 4 lanes per permutation (fold_q4_k,
// fold_tail_q4_k: latency-bound launches, ~20-50 us less per fold tree); RISC0_HIP_PV_Q4=0 off.
void pv_hip_rows(uint4* out, const uint32_t* m, uint32_t count, uint32_t col_size, hipStream_t s, bool v0);
void pv_hip_fold(uint4* out, const uint4* in, uint32_t n, hipStream_t s);
void pv_hip_fold_tail(uint4* nodes, uint32_t n0, hipStream_t s);
void pv_hip_fold_q4(uint4* out, const uint4* in, uint32_t n, hipStream_t s);
void pv_hip_fold_tail_q4(uint4* nodes, uint32_t n0, hipStream_t s);
static int env_mode(const char* name) { // 0, 1, or 2 when unset (or anything else)
  const char* v = std::getenv(name);
  return v && std::strcmp(v, "0") == 0 ? 0 : v && std::strcmp(v, "1") == 0 ? 1 : 2;
}
// Whether a launch of `perms` permutations uses the max-ilp TU.
static bool use_pv_tu(size_t) {
  static const bool on = env_mode("RISC0_HIP_PV_TU") != 0;
  return on;
}
static bool use_pv_tail() {
  static const bool on = env_mode("RISC0_HIP_PV_TAIL") != 0;
  return on;
}
static bool use_pv_q4() {
  static const bool on = env_mode("RISC0_HIP_PV_Q4") != 0;
  return on;
}
// RISC0_HIP_PV_ROWS0=1: rows0_k (rows_k with plain global loads), for A/B runs.
static bool use_pv_rows0() {
  static const bool on = env_mode("RISC0_HIP_PV_ROWS0") == 1;
  return on;
}
static constexpr uint32_t PV_Q4_MAX = 1024; // fold_q4_k up to this many digests
static constexpr uint32_t PV_TAIL = 128; // fold_tail_k's block size limit
// Launches of >= 2^19 permutations (~0.7-1 ms and up) run on a side hardware queue (hip_hop.h): the
// following fold layer, or the host's gpu.sync(), would otherwise wait ~0.2 ms more.
static void hip_pv_rows(uint4* out, const uint32_t* m, uint32_t count, uint32_t col_size, hipStream_t main) {
  const size_t perms = size_t(count) * ((col_size + 15) / 16);
  risc0_hip_hop::Scope s(main, risc0_hip_hop::want(risc0_hip_hop::kMerkleRows, perms, 19));
  if (use_pv_tu(perms)) {
    pv_hip_rows(out, m, count, col_size, s, use_pv_rows0());
  } else {
    uint32_t bs = count < PV_BLOCK ? count : PV_BLOCK;
    if (use_pv_rows0())
      pv::rows0_k<<<(count + bs - 1) / bs, bs, 0, s>>>(out, m, count, col_size);
    else
      pv::rows_k<<<(count + bs - 1) / bs, bs, 0, s>>>(out, m, count, col_size);
  }
}
static void hip_pv_fold(uint4* out, const uint4* in, uint32_t n, hipStream_t main) {
  risc0_hip_hop::Scope s(main, risc0_hip_hop::want(risc0_hip_hop::kMerkleFold, n, 19));
  if (n <= PV_Q4_MAX && use_pv_q4()) {
    pv_hip_fold_q4(out, in, n, s);
  } else if (use_pv_tu(n)) {
    pv_hip_fold(out, in, n, s);
  } else {
    uint32_t bs = n < PV_BLOCK ? n : PV_BLOCK;
    pv::fold_k<<<(n + bs - 1) / bs, bs, 0, s>>>(out, in, n);
  }
}
static void hip_pv_fold_tail(uint4* nodes, uint32_t n0, hipStream_t s) {
  if (use_pv_q4())
    pv_hip_fold_tail_q4(nodes, n0, s);
  else if (use_pv_tu(n0))
    pv_hip_fold_tail(nodes, n0, s);
  else
    pv::fold_tail_k<<<1, n0, 0, s>>>(nodes, n0);
}
#endif
#if defined(__HIPCC__) && !defined(__HIP_DEVICE_COMPILE__)
// HIP host pass: blst_256_t::operator[] is private, causing errors in
// poseidon254.cuh's __device__ helpers. Provide empty __global__ kernel
// bodies so hipcc can generate launch stubs; real implementations come
// from the device pass which includes poseidon254.cuh with mont_t types.
#include "poseidon254_constants.cuh"
template<typename bn254_t>
__global__ __launch_bounds__(512)
void _poseidon254_fold(bn254_t*, const bn254_t*, size_t) {}
template<typename bn254_t>
__global__ __launch_bounds__(512)
void _poseidon254_rows(bn254_t*, const fr_t*, size_t, uint32_t) {}
#else
#include "poseidon254.cuh"
#endif

// Workaround: cudaGetDeviceProperties returns multiProcessorCount=1
// in some VM/passthrough setups, while cudaDeviceGetAttribute returns
// the correct value. Cache the correct SM count for use in cooperative
// kernel launches.
static int get_real_sm_count() {
    int device = 0;
    cudaGetDevice(&device);
    static int sm_counts[16] = {};
    if (sm_counts[device] == 0) {
        cudaDeviceGetAttribute(&sm_counts[device], cudaDevAttrMultiProcessorCount, device);
    }
    return sm_counts[device];
}

#ifdef __HIPCC__
// Detect wavefront size: 32 = RDNA (gfx10/11/12), 64 = CDNA (gfx9xx).
// Used to select Poseidon2 block size matching launch_bounds.
static int get_warp_size() {
    int device = 0;
    hipGetDevice(&device);
    static int ws[16] = {};
    if (ws[device] == 0) {
        hipDeviceGetAttribute(&ws[device], hipDeviceAttributeWarpSize, device);
    }
    return ws[device];
}
#endif

extern "C" RustError::by_value sppark_poseidon2_init() {
  int device = 0;
  cudaGetDevice(&device);
  static bool initialized[16] = {};
  if (initialized[device])
    return RustError{cudaSuccess};

  const gpu_t& gpu = select_gpu(-1);
  try {
    // Allocate tiny device buffer and run 1-hash poseidon2_fold to trigger
    // CUDA module loading for poseidon2 kernels (~5-10ms first-use overhead).
    void* d_buf = nullptr;
    CUDA_OK(cudaMalloc(&d_buf, 2048));
    CUDA_OK(cudaMemset(d_buf, 0, 2048));
    _poseidon2_fold<<<1, 1, 0, gpu>>>(
        (poseidon_out_t*)d_buf,
        (const poseidon_in_t*)((char*)d_buf + 1024),
        1);
    CUDA_OK(cudaGetLastError());
    gpu.sync();
    CUDA_OK(cudaFree(d_buf));
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }
  initialized[device] = true;
  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_poseidon2_fold(poseidon_out_t* d_out, const poseidon_in_t* d_in, size_t num_hashes) {
  const gpu_t& gpu = select_gpu(-1);
  if (num_hashes == 0)
    return RustError{cudaSuccess};

#ifdef __HIPCC__
  // RDNA (gfx12xx, wave32): 256 threads matches launch_bounds(256,3)
  // CDNA (gfx9xx, wave64):  512 threads matches launch_bounds(512,2)
  const size_t P2_BLOCK = (get_warp_size() >= 64) ? 512 : 256;
#else
  const size_t P2_BLOCK = 256;
#endif
  size_t block_size = num_hashes < P2_BLOCK ? num_hashes : P2_BLOCK;
  // Ceil-div: previously used `num_hashes / P2_BLOCK` which silently dropped
  // up to P2_BLOCK-1 trailing hashes for non-pow2 input. _rows already used
  // ceil; aligning fold for correctness.
  size_t num_blocks = num_hashes < P2_BLOCK ? 1 : (num_hashes + P2_BLOCK - 1) / P2_BLOCK;

  try {
    (void)cudaGetLastError(); // consume any stale async errors
#ifndef __HIPCC__
    if (!pvs::fold(gpu, (uint32_t*)d_out, (const uint32_t*)d_in, (uint32_t)num_hashes)) {
      size_t bs = num_hashes < PV_BLOCK ? num_hashes : PV_BLOCK;
      pv::fold_k<<<(num_hashes + bs - 1) / bs, bs, 0, gpu>>>(
          (uint4*)d_out, (const uint4*)d_in, (uint32_t)num_hashes);
    }
#else
    if (use_pv_kernels()) {
      hip_pv_fold((uint4*)d_out, (const uint4*)d_in, (uint32_t)num_hashes, gpu);
    } else {
      _poseidon2_fold<<<num_blocks, block_size, 0, gpu>>>(d_out, d_in, num_hashes);
    }
#endif

    CUDA_OK(cudaGetLastError());

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_poseidon2_fold_tree(poseidon_out_t* nodes, uint32_t layers) {
  const gpu_t& gpu = select_gpu(-1);

#ifdef __HIPCC__
  // RDNA (gfx12xx, wave32): 256 threads matches launch_bounds(256,3)
  // CDNA (gfx9xx, wave64):  512 threads matches launch_bounds(512,2)
  const size_t P2_BLOCK = (get_warp_size() >= 64) ? 512 : 256;
#else
  const size_t P2_BLOCK = 256;
#endif

  try {
    (void)cudaGetLastError(); // consume any stale async errors
    for (int i = layers - 1; i >= 0; i--) {
      uint32_t layer_size = 1u << i;
#ifndef __HIPCC__
      if (!pvs::fold(gpu, (uint32_t*)(nodes + layer_size), (const uint32_t*)(nodes + 2 * layer_size),
                     layer_size)) {
        uint32_t bs = layer_size < PV_BLOCK ? layer_size : PV_BLOCK;
        pv::fold_k<<<(layer_size + bs - 1) / bs, bs, 0, gpu>>>(
            (uint4*)(nodes + layer_size), (const uint4*)(nodes + 2 * layer_size), layer_size);
      }
#else
      if (use_pv_kernels() && use_pv_tail() && layer_size <= PV_TAIL) {
        // This layer and all above it, in one launch.
        hip_pv_fold_tail((uint4*)nodes, layer_size, gpu);
        CUDA_OK(cudaGetLastError());
        break;
      }
      if (use_pv_kernels()) {
        hip_pv_fold((uint4*)(nodes + layer_size), (const uint4*)(nodes + 2 * layer_size),
                    layer_size, gpu);
      } else {
        size_t block_size = layer_size < P2_BLOCK ? layer_size : P2_BLOCK;
        size_t num_blocks = layer_size < P2_BLOCK ? 1 : layer_size / P2_BLOCK;
        _poseidon2_fold<<<num_blocks, block_size, 0, gpu>>>(
            nodes + layer_size,
            (const poseidon_in_t*)(nodes + 2 * layer_size),
            layer_size);
      }
#endif
      CUDA_OK(cudaGetLastError());
    }
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_poseidon2_rows(poseidon_out_t* d_out, const fr_t* d_in, uint32_t count, uint32_t col_size) {
  const gpu_t& gpu = select_gpu(-1);
  if (count == 0)
    return RustError{cudaSuccess};

#ifdef __HIPCC__
  // RDNA (gfx12xx, wave32): 256 threads matches launch_bounds(256,3)
  // CDNA (gfx9xx, wave64):  512 threads matches launch_bounds(512,2)
  const size_t P2_BLOCK = (get_warp_size() >= 64) ? 512 : 256;
#else
  const size_t P2_BLOCK = 256;
#endif
  size_t block_size = count < P2_BLOCK ? count : P2_BLOCK;
  size_t num_blocks = (count + block_size - 1) / block_size;

  try {
    (void)cudaGetLastError(); // consume any stale async errors
#ifndef __HIPCC__
    if (!pvs::rows(gpu, (uint32_t*)d_out, (const uint32_t*)d_in, count, col_size)) {
      uint32_t bs = count < PV_BLOCK ? count : PV_BLOCK;
      pv::rows_k<<<(count + bs - 1) / bs, bs, 0, gpu>>>(
          (uint4*)d_out, (const uint32_t*)d_in, count, col_size);
    }
#else
    if (use_pv_kernels()) {
      hip_pv_rows((uint4*)d_out, (const uint32_t*)d_in, count, col_size, gpu);
    } else {
      _poseidon2_rows<<<num_blocks, block_size, 0, gpu>>>(d_out, d_in, count, col_size);
    }
#endif

    CUDA_OK(cudaGetLastError());

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

static void compute_grid_block_size(size_t total_count, size_t& block_size, size_t& num_blocks) {
  size_t min_block_size = 4 * WARP_SZ;

  if (total_count < (block_size * num_blocks)) {
    size_t count_per_block = total_count / num_blocks;

    if (count_per_block > min_block_size) {
      block_size = ((count_per_block + min_block_size - 1) / min_block_size) * min_block_size;
      num_blocks = (total_count + block_size - 1) / block_size;
    } else {
      block_size = min_block_size;
      num_blocks = (total_count + min_block_size - 1) / min_block_size;
    }
  } else {
    size_t base_iter = (total_count + (num_blocks * block_size) - 1) / (num_blocks * block_size);
    size_t out_block_size = block_size;

    for (size_t cur_block_size = block_size - min_block_size; cur_block_size >= min_block_size;
         cur_block_size -= min_block_size) {
      size_t cur_iter =
          (total_count + (num_blocks * cur_block_size) - 1) / (num_blocks * cur_block_size);

      if (cur_iter != base_iter)
        break;
      out_block_size = cur_block_size;
    }

    block_size = out_block_size;
  }
}

extern "C" RustError::by_value
sppark_poseidon254_fold(alt_bn128::fr_t* d_out, const alt_bn128::fr_t* d_in, size_t num_hashes) {
  const gpu_t& gpu = select_gpu(-1);

  size_t block_size = 512;
  size_t num_blocks = gpu.sm_count();

  compute_grid_block_size(num_hashes, block_size, num_blocks);

  try {
    (void)cudaGetLastError(); // consume any stale async errors
    _poseidon254_fold<<<num_blocks, block_size, 0, gpu>>>(d_out, d_in, num_hashes);

    CUDA_OK(cudaGetLastError());

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_poseidon254_fold_tree(alt_bn128::fr_t* nodes, uint32_t layers) {
  const gpu_t& gpu = select_gpu(-1);

  size_t block_size = 512;
  size_t num_blocks = gpu.sm_count();

  try {
    (void)cudaGetLastError(); // consume any stale async errors
    for (int i = layers - 1; i >= 0; i--) {
      uint32_t layer_size = 1u << i;
      size_t bs = block_size;
      size_t nb = num_blocks;
      compute_grid_block_size(layer_size, bs, nb);

      _poseidon254_fold<<<nb, bs, 0, gpu>>>(
          nodes + layer_size,
          nodes + 2 * layer_size,
          layer_size);
      CUDA_OK(cudaGetLastError());
    }
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_poseidon254_rows(alt_bn128::fr_t* d_out, const fr_t* d_in, size_t count, uint32_t col_size) {
  const gpu_t& gpu = select_gpu(-1);

  size_t block_size = 512;
  size_t num_blocks = gpu.sm_count();

  compute_grid_block_size(count, block_size, num_blocks);

  try {
    (void)cudaGetLastError(); // consume any stale async errors
    _poseidon254_rows<<<num_blocks, block_size, 0, gpu>>>(d_out, d_in, count, col_size);

    CUDA_OK(cudaGetLastError());

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value sppark_prefix_product(fr4_t d_inout[/*count*/], uint32_t count) {
  const gpu_t& gpu = select_gpu(-1);

  try {
    prefix_op<Multiply<fr4_t>>(d_inout, count, gpu);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
supra_poly_divide(fr4_t d_inout[/*len*/], size_t len, fr4_t* remainder, const fr4_t& pow) {
  const gpu_t& gpu = select_gpu(-1);

  try {
    div_by_x_minus_z<true>(d_inout, len, pow, gpu, get_real_sm_count());
    gpu.DtoH(remainder, &d_inout[len - 1], 1);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
supra_poly_divide_batch(fr4_t d_inout[/*len*/], size_t len,
                        fr4_t* remainders, const fr4_t pows[],
                        uint32_t num_divides) {
  const gpu_t& gpu = select_gpu(-1);
  int sm_count = get_real_sm_count();

  try {
    for (uint32_t d = 0; d < num_divides; d++) {
      div_by_x_minus_z<true>(d_inout, len, pows[d], gpu, sm_count);
      gpu.DtoH(&remainders[d], &d_inout[len - 1], 1);
    }
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

// Batch polynomial division across multiple combo slices.
// Each combo slice (at combos_base + combo_indices[c] * stride) is divided
// by its respective roots in-place. Kernels are pipelined with a single
// gpu.sync() at the end for better GPU utilization; each division's remainder
// (the slice's last element) is copied out in stream order, into
// remainders[pows_offset + d], for the caller to check.
extern "C" RustError::by_value
supra_poly_divide_multi(fr4_t* combos_base, size_t stride,
                        const uint32_t combo_indices[],
                        const uint32_t pows_per_combo[],
                        const uint32_t all_pows_raw[],
                        uint32_t num_combos,
                        fr4_t* remainders) {
  const gpu_t& gpu = select_gpu(-1);
  int sm_count = get_real_sm_count();
  const fr4_t* all_pows = reinterpret_cast<const fr4_t*>(all_pows_raw);

  try {
#ifndef __HIPCC__
    // Fused multi-root division (bit-identical to the loop below; one sync). On unless
    // RISC0_CUDA_MDIV=0.
    static const bool s_mdiv = [] {
      const char* e = std::getenv("RISC0_CUDA_MDIV");
      return !(e && e[0] == '0');
    }();
    if (s_mdiv && mdiv_supported(stride, num_combos, pows_per_combo)) {
      mdiv_run(combos_base, stride, combo_indices, pows_per_combo, all_pows, num_combos, remainders, gpu);
      return RustError{cudaSuccess};
    }
#else
    // ROCm: the same fused division (mdiv.cuh), MD_MAXJ combos per mdiv_run, instead of one
    // cooperative launch and one pageable remainder copy per root (~2.2-2.8x faster at lg 20,
    // bit-identical; r0amd r8/L4 mdiv). On unless RISC0_HIP_MDIV=0.
    static const bool s_mdiv = [] {
      const char* e = std::getenv("RISC0_HIP_MDIV");
      return !(e && e[0] == '0');
    }();
    if (s_mdiv) {
      bool ok = num_combos > 0;
      for (uint32_t g = 0; ok && g < num_combos; g += MD_MAXJ)
        ok = mdiv_supported(stride, std::min<uint32_t>(MD_MAXJ, num_combos - g), pows_per_combo + g);
      if (ok) {
        size_t off = 0;
        for (uint32_t g = 0; g < num_combos; g += MD_MAXJ) {
          const uint32_t k = std::min<uint32_t>(MD_MAXJ, num_combos - g);
          size_t cnt = 0;
          for (uint32_t c = g; c < g + k; c++) cnt += pows_per_combo[c];
          mdiv_run(combos_base, stride, combo_indices + g, pows_per_combo + g, all_pows + off, k,
                   remainders + off, gpu);
          off += cnt;
        }
        return RustError{cudaSuccess};
      }
    }
#endif
#ifdef SPPARK_NO_COOPERATIVE_LAUNCH
    // Without cooperative launch each division runs as one block (see div_by_x_minus_z), so run
    // the combos' independent chains concurrently, one stream each, after the work queued on
    // gpu's stream. Each remainder is copied on the device as its division finishes, and all of
    // them to the host once at the end: a pageable device-to-host copy per division would block
    // the host between launches.
    (void)sm_count;
    constexpr uint32_t kStreams = 8;
    int dev = 0;
    CUDA_OK(cudaGetDevice(&dev));
    static cudaStream_t side[16][kStreams] = {};
    static cudaEvent_t done[16][kStreams] = {};
    static cudaEvent_t ready[16] = {};
    static fr4_t* d_rem[16] = {};
    static size_t d_rem_cap[16] = {};
    if (!ready[dev]) {
      CUDA_OK(cudaEventCreateWithFlags(&ready[dev], cudaEventDisableTiming));
      for (uint32_t k = 0; k < kStreams; k++) {
        CUDA_OK(cudaStreamCreateWithFlags(&side[dev][k], cudaStreamNonBlocking));
        CUDA_OK(cudaEventCreateWithFlags(&done[dev][k], cudaEventDisableTiming));
      }
    }
    size_t total_pows = 0;
    for (uint32_t c = 0; c < num_combos; c++)
      total_pows += pows_per_combo[c];
    if (total_pows > d_rem_cap[dev]) {
      if (d_rem[dev])
        CUDA_OK(cudaFree(d_rem[dev]));
      CUDA_OK(cudaMalloc(&d_rem[dev], total_pows * sizeof(fr4_t)));
      d_rem_cap[dev] = total_pows;
    }
    struct one_stream {
      cudaStream_t s;
      int sm_count() const { return 1; }
      operator cudaStream_t() const { return s; }
    };
    CUDA_OK(cudaEventRecord(ready[dev], gpu));
    uint32_t used = num_combos < kStreams ? num_combos : kStreams;
    for (uint32_t k = 0; k < used; k++)
      CUDA_OK(cudaStreamWaitEvent(side[dev][k], ready[dev], 0));
    size_t pows_offset = 0;
    for (uint32_t c = 0; c < num_combos; c++) {
      one_stream s{side[dev][c % kStreams]};
      fr4_t* slice = combos_base + combo_indices[c] * stride;
      for (uint32_t d = 0; d < pows_per_combo[c]; d++) {
        div_by_x_minus_z<true>(slice, stride, all_pows[pows_offset + d], s, 1);
        CUDA_OK(cudaMemcpyAsync(&d_rem[dev][pows_offset + d], &slice[stride - 1], sizeof(fr4_t),
                                cudaMemcpyDeviceToDevice, s.s));
      }
      pows_offset += pows_per_combo[c];
    }
    for (uint32_t k = 0; k < used; k++) {
      CUDA_OK(cudaEventRecord(done[dev][k], side[dev][k]));
      CUDA_OK(cudaStreamWaitEvent(gpu, done[dev][k], 0));
    }
    gpu.DtoH(remainders, d_rem[dev], total_pows);
    gpu.sync();
#else
    size_t pows_offset = 0;
    for (uint32_t c = 0; c < num_combos; c++) {
      fr4_t* slice = combos_base + combo_indices[c] * stride;
      for (uint32_t d = 0; d < pows_per_combo[c]; d++) {
        div_by_x_minus_z<true>(slice, stride, all_pows[pows_offset + d], gpu, sm_count);
        gpu.DtoH(&remainders[pows_offset + d], &slice[stride - 1], 1);
      }
      pows_offset += pows_per_combo[c];
    }
    gpu.sync();
#endif
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}
