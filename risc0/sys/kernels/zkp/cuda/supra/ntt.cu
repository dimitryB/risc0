#include "ff/baby_bear.hpp"
#include "ntt/ntt.cuh"

#include <cstdlib>
#include <cstring>

#ifdef __HIPCC__
#ifndef cudaErrorInvalidValue
#define cudaErrorInvalidValue hipErrorInvalidValue
#endif
// ROCm: commit_coeffs_lde, the expand+NTT and the bit reversal run the batched kernels of ntt_x.cuh /
// ntt_q.cuh (see hip_expand_ntt). RISC0_HIP_NTT=0 restores sppark's per-pass path. Read on every call
// so that one process can run and compare both paths.
static bool hip_fast_ntt() {
  const char* e = std::getenv("RISC0_HIP_NTT");
  return !(e && e[0] == '0');
}
#endif

// Expose protected bit_rev from NTT base class
struct NTT_helper : public NTT {
    using NTT::bit_rev;
};

// Batched LDE expand kernel: write all output elements in one coalesced pass.
// For each output position, either copy the corresponding input element
// (if position is a multiple of blowup) or write zero.
template<class fr_t>
__launch_bounds__(256)
__global__ void batch_lde_expand_kernel(
    fr_t* __restrict__ d_out,
    const fr_t* __restrict__ d_in,
    uint32_t domain_size,
    uint32_t lg_ext_size,
    uint32_t lg_blowup,
    uint32_t poly_count)
{
    uint32_t ext_size = 1u << lg_ext_size;
    uint32_t ext_mask = ext_size - 1;
    uint32_t blowup_mask = (1u << lg_blowup) - 1;
    uint64_t total = (uint64_t)ext_size * poly_count;

    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (uint64_t)gridDim.x * blockDim.x)
    {
        uint32_t pos_in_ext = (uint32_t)i & ext_mask;
        uint32_t col = (uint32_t)(i >> lg_ext_size);
        fr_t val;
        if ((pos_in_ext & blowup_mask) == 0) {
            uint32_t src_idx = pos_in_ext >> lg_blowup;
            val = d_in[(uint64_t)col * domain_size + src_idx];
        } else {
            val = fr_t(0);
        }
        d_out[i] = val;
    }
}

static inline int cached_sm_count() {
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    static int counts[16] = {};
    if (counts[device] == 0) {
        CUDA_OK(cudaDeviceGetAttribute(&counts[device], cudaDevAttrMultiProcessorCount, device));
    }
    return counts[device];
}

static inline void launch_batch_expand(const gpu_t& gpu,
                                       fr_t* d_out, fr_t* d_in,
                                       uint32_t domain_size,
                                       uint32_t lg_domain_size,
                                       uint32_t lg_blowup,
                                       uint32_t poly_count)
{
    uint32_t lg_ext = lg_domain_size + lg_blowup;
    uint32_t ext_domain_size = 1u << lg_ext;
    uint64_t total = (uint64_t)ext_domain_size * poly_count;

    uint32_t block_sz = 256;
    uint32_t max_blocks = (uint32_t)cached_sm_count() * 16;
    uint32_t need_blocks = (uint32_t)((total + block_sz - 1) / block_sz);
    uint32_t grid_sz = need_blocks < max_blocks ? need_blocks : max_blocks;

    batch_lde_expand_kernel<fr_t><<<grid_sz, block_sz, 0, (cudaStream_t)gpu>>>(
        d_out, d_in, domain_size, lg_ext, lg_blowup, poly_count);
}

extern "C" RustError::by_value sppark_init() {
  // Always call select_gpu(-1) to ensure the CUDA context is retained
  // on the calling thread for its current device.
  (void)select_gpu(-1);

  int device = 0;
  cudaGetDevice(&device);
  static bool initialized[16] = {};
  if (initialized[device])
    return RustError{cudaSuccess};

  // Use lg_domain_size=16 (64K elements) to exercise all NTT kernel variants
  // (CT_NTT<8,true/false>, GS_NTT<8,true/false>, batch_bit_reverse,
  // LDE_distribute_powers). The small size (256KB) runs in ~1ms but avoids
  // ~5-10ms of first-launch stalls during the actual proof.
  uint32_t lg_domain_size = 16;
  uint32_t domain_size = 1U << lg_domain_size;

  std::vector<fr_t> inout(domain_size, fr_t(0));
  inout[0] = fr_t(1);
  inout[1] = fr_t(1);

  const gpu_t& gpu = select_gpu(-1);

  try {
    NTT::Base(gpu,
              &inout[0],
              lg_domain_size,
              NTT::InputOutputOrder::NR,
              NTT::Direction::forward,
              NTT::Type::standard);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  initialized[device] = true;
  return RustError{cudaSuccess};
}

extern "C" RustError::by_value sppark_batch_expand(
    fr_t* d_out, fr_t* d_in, uint32_t lg_domain_size, uint32_t lg_blowup, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu(-1);

  try {
    launch_batch_expand(gpu, d_out, d_in, domain_size, lg_domain_size, lg_blowup, poly_count);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

#ifdef __HIPCC__
// On ROCm the per-column loops below run as batched launches instead (blockIdx.y picks the
// column): one launch per pass for all columns rather than one per column, which on Windows HIP
// cost more in launch overhead than the work in small columns (keccak commits ~1100 columns).
// Chunked so a launch stays within gridDim.y and sppark's 32-bit column offset
// (`(index_t)blockIdx.y * col_stride`).
template <class F>
static void for_column_batches(uint32_t poly_count, uint64_t col_stride, F launch) {
  uint64_t max_cols = 0xffffffffull / col_stride;
  if (max_cols > 65535)
    max_cols = 65535;
  for (uint32_t i = 0; i < poly_count; i += (uint32_t)max_cols) {
    uint32_t n = poly_count - i < max_cols ? poly_count - i : (uint32_t)max_cols;
    launch(i, n);
  }
}
#endif

extern "C" RustError::by_value
sppark_batch_NTT(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu(-1);

  try {
#ifdef __HIPCC__
    for_column_batches(poly_count, domain_size, [&](uint32_t i, uint32_t n) {
      NTT::Base_dev_ptr_batch(gpu, &d_inout[(uint64_t)i * domain_size], lg_domain_size,
                              NTT::InputOutputOrder::RN, NTT::Direction::forward,
                              NTT::Type::standard, n, domain_size);
    });
#else
    // Process each polynomial sequentially on the same stream
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT::Base_dev_ptr(gpu, &d_inout[(uint64_t)i * domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::RN,
                        NTT::Direction::forward,
                        NTT::Type::standard);
    }
#endif

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_batch_iNTT(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu(-1);

  try {
#ifdef __HIPCC__
    for_column_batches(poly_count, domain_size, [&](uint32_t i, uint32_t n) {
      NTT::Base_dev_ptr_batch(gpu, &d_inout[(uint64_t)i * domain_size], lg_domain_size,
                              NTT::InputOutputOrder::NR, NTT::Direction::inverse,
                              NTT::Type::standard, n, domain_size);
    });
#else
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT::Base_dev_ptr(gpu, &d_inout[(uint64_t)i * domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::NR,
                        NTT::Direction::inverse,
                        NTT::Type::standard);
    }
#endif

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_batch_zk_shift(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu(-1);

  try {
#ifdef __HIPCC__
    for_column_batches(poly_count, domain_size, [&](uint32_t i, uint32_t n) {
      NTT::LDE_powers_batch(gpu, &d_inout[(uint64_t)i * domain_size], lg_domain_size, n,
                            domain_size);
    });
#else
    // Apply ZK shift per polynomial
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT::LDE_powers(gpu, &d_inout[(uint64_t)i * domain_size], lg_domain_size);
    }
#endif

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_batch_iNTT_zk_shift(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu(-1);

  try {
#ifdef __HIPCC__
    for_column_batches(poly_count, domain_size, [&](uint32_t i, uint32_t n) {
      fr_t* cols = &d_inout[(uint64_t)i * domain_size];
      NTT::Base_dev_ptr_batch(gpu, cols, lg_domain_size, NTT::InputOutputOrder::NR,
                              NTT::Direction::inverse, NTT::Type::standard, n, domain_size);
      NTT::LDE_powers_batch(gpu, cols, lg_domain_size, n, domain_size);
    });
#else
    // iNTT + ZK shift per polynomial
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT::Base_dev_ptr(gpu, &d_inout[(uint64_t)i * domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::NR,
                        NTT::Direction::inverse,
                        NTT::Type::standard);
      NTT::LDE_powers(gpu, &d_inout[(uint64_t)i * domain_size], lg_domain_size);
    }
#endif

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

#ifndef __HIPCC__
static bool nttq_expand_ntt(const gpu_t& gpu, fr_t* d_out, fr_t* d_in, uint32_t lg,
                            uint32_t lg_blowup, uint32_t poly_count);
#else
static bool hip_expand_ntt(const gpu_t& gpu, fr_t* d_out, fr_t* d_in, uint32_t lg,
                           uint32_t lg_blowup, uint32_t poly_count);
static void hip_bit_reverse(const gpu_t& gpu, fr_t* d_inout, uint32_t lg, uint32_t poly_count);
#endif

extern "C" RustError::by_value
sppark_batch_expand_NTT(fr_t* d_out, fr_t* d_in,
                        uint32_t lg_domain_size, uint32_t lg_blowup, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;
  uint32_t ext_domain_size = domain_size << lg_blowup;
  uint32_t lg_ext = lg_domain_size + lg_blowup;

  const gpu_t& gpu = select_gpu(-1);

  try {
#ifndef __HIPCC__
    if (nttq_expand_ntt(gpu, d_out, d_in, lg_domain_size, lg_blowup, poly_count)) {
      gpu.sync();
      return RustError{cudaSuccess};
    }
#else
    if (hip_fast_ntt() && hip_expand_ntt(gpu, d_out, d_in, lg_domain_size, lg_blowup, poly_count)) {
      gpu.sync();
      return RustError{cudaSuccess};
    }
#endif
    // Batched expand: single kernel for all columns
    launch_batch_expand(gpu, d_out, d_in, domain_size, lg_domain_size, lg_blowup, poly_count);

#ifdef __HIPCC__
    for_column_batches(poly_count, ext_domain_size, [&](uint32_t i, uint32_t n) {
      NTT::Base_dev_ptr_batch(gpu, &d_out[(uint64_t)i * ext_domain_size], lg_ext,
                              NTT::InputOutputOrder::RN, NTT::Direction::forward,
                              NTT::Type::standard, n, ext_domain_size);
    });
#else
    // Forward NTT per polynomial on the extended domain
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT::Base_dev_ptr(gpu, &d_out[(uint64_t)i * ext_domain_size],
                        lg_ext,
                        NTT::InputOutputOrder::RN,
                        NTT::Direction::forward,
                        NTT::Type::standard);
    }
#endif

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

// Batched bit-reverse permutation on the sppark stream.
// Uses sppark's optimized bit_rev (shared memory + Z_COUNT blocking) per polynomial.
extern "C" RustError::by_value
sppark_batch_bit_reverse(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0 || poly_count == 0)
    return RustError{cudaSuccess};

  uint64_t domain_size = 1ULL << lg_domain_size;

  const gpu_t& gpu = select_gpu(-1);

  try {
#ifdef __HIPCC__
    if (hip_fast_ntt()) {
      hip_bit_reverse(gpu, d_inout, lg_domain_size, poly_count);
      gpu.sync();
      return RustError{cudaSuccess};
    }
#endif
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT_helper::bit_rev(&d_inout[i * domain_size], &d_inout[i * domain_size],
                          lg_domain_size, gpu);
    }

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

// Fused commit NTT and its batched helpers (ntt_x.cuh, ntt_q.cuh).
#include "ntt_x.cuh"

// Two-pass LDE of bit-reversed coefficients (d_in is restored to bit-reversed order).
static bool nttq_expand_ntt(const gpu_t& gpu, fr_t* d_out, fr_t* d_in, uint32_t lg,
                            uint32_t lg_blowup, uint32_t poly_count) {
  if (!nttq::supported(lg, lg_blowup) || !nttq::enabled())
    return false;
  bitrev_batched(gpu, d_in, lg, poly_count);
  nttq::forward(gpu, d_out, d_in, lg, lg_blowup, poly_count);
  bitrev_batched(gpu, d_in, lg, poly_count);
  return true;
}

#ifdef __HIPCC__
// RISC0_HIP_NTT_EXPAND=x: the CT path below for every size (no nttq LDE).
static bool hip_expand_q() {
  static const bool on = [] {
    const char* e = std::getenv("RISC0_HIP_NTT_EXPAND");
    return !(e && e[0] == 'x');
  }();
  return on;
}
// LDE of bit-reversed coefficients for the check group and FRI. lg 16..21 with blowup 4: nttq's
// two passes (pass_h). The input is bit-reversed out of place into the last quarter of each output
// column, which the forward column pass reads before the row pass of the same columns overwrites
// it; d_in is not touched. Check group lg 20: RX 7900 XTX 2.1 -> 1.5 ms, RX 9070 XT 3.0 -> 1.9 ms
// (opt-artifacts r0amd/r9/R4). Other sizes: the expand folded into the first CT pass (zero-padded
// input synthesized on load, trivial levels skipped), all columns in one launch per pass.
static bool hip_expand_ntt(const gpu_t& gpu, fr_t* d_out, fr_t* d_in, uint32_t lg,
                           uint32_t lg_blowup, uint32_t poly_count) {
  const uint32_t lg_ext = lg + lg_blowup;
  if (lg < 4 || lg_ext > 23 || lg_blowup == 0 || lg_blowup >= 6)
    return false;
#ifdef NTTQ_TUNE
  if (const char* e = std::getenv("NTT_EXPAND")) {
    if (e[0] == 'b') return false;
    if (e[0] == 'q') return nttq_expand_ntt(gpu, d_out, d_in, lg, lg_blowup, poly_count);
  }
#endif
  const size_t n = (size_t)1 << lg, N = n << lg_blowup;
  if (nttq::supported(lg, lg_blowup) && nttq::enabled() && hip_expand_q()) {
    bit_rev_t2_oop<32, 128><<<dim3(n / 32 / 32, poly_count), 128, 0, gpu>>>(
        d_out + (N - n), d_in, lg, n, N);
    CUDA_OK(cudaGetLastError());
    nttq::forward(gpu, d_out, d_out + (N - n), lg, lg_blowup, poly_count, N);
    return true;
  }
  ct_x(gpu, poly_count, d_in, n, lg_blowup, d_out, N, (int)lg_ext, true);
  return true;
}

static void hip_bit_reverse(const gpu_t& gpu, fr_t* d_inout, uint32_t lg, uint32_t poly_count) {
  bitrev_batched(gpu, d_inout, lg, poly_count);
}
#endif

#ifndef __HIPCC__
// Batched iNTT + zk-shift: the per-column iNTTs, then one zk-shift kernel over all columns.
extern "C" RustError::by_value
sppark_batch_iNTT_zk_shift_fused(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};
  uint32_t domain_size = 1U << lg_domain_size;
  const gpu_t& gpu = select_gpu(-1);
  try {
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT::Base_dev_ptr(gpu, &d_inout[(uint64_t)i * domain_size], lg_domain_size,
                        NTT::InputOutputOrder::NR, NTT::Direction::inverse, NTT::Type::standard);
    }
    nttx_batch_zk_shift(gpu, d_inout, lg_domain_size, poly_count);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }
  return RustError{cudaSuccess};
}

// Batched, tiled bit reversal of poly_count columns.
extern "C" RustError::by_value
sppark_batch_bit_reverse_tiled(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0 || poly_count == 0)
    return RustError{cudaSuccess};
  const gpu_t& gpu = select_gpu(-1);
  try {
    bitrev_batched(gpu, d_inout, lg_domain_size, poly_count);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }
  return RustError{cudaSuccess};
}

#endif // __HIPCC__

// Fused commit: d_coeffs = the natural-order zk-shifted coefficients of d_witness and
// d_evaluated = their 2^lg_blowup LDE, in L2-sized column chunks. d_witness is unchanged.
// Bit-exact with iNTT + zk-shift + expand + NTT + bit reversal for lg_domain_size 10..23.
extern "C" RustError::by_value
sppark_commit_coeffs_lde(fr_t* d_coeffs, fr_t* d_evaluated, const fr_t* d_witness,
                         uint32_t lg_domain_size, uint32_t lg_blowup, uint32_t poly_count) {
  if (lg_domain_size < 10 || lg_domain_size > 23)
    return RustError(cudaErrorInvalidValue, "commit_coeffs_lde: lg_domain_size out of range");
  const gpu_t& gpu = select_gpu(-1);
  try {
    nttx_commit(gpu, d_coeffs, d_evaluated, d_witness, lg_domain_size, lg_blowup, poly_count, 0);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }
  return RustError{cudaSuccess};
}
