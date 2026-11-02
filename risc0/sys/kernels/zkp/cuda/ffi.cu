// Copyright 2024 RISC Zero, Inc.
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

#include "cuda.h"
#include "fp.h"
#include "fpext.h"
#include "kernels.h"

#include "vendor/nvtx3/nvtx3.hpp"
#include "thp_pinned.h"
#include "device_alloc.h"

#include <cstdint>
#include <exception>

// eval_u v2 (kernels.cu)
#define EU_J 16
#define EU_THREADS 256
__global__ void eu2_pows(FpExt* pw, FpExt* ps, const FpExt* xs, uint32_t S, uint32_t chunk);
__global__ void eu2_main(FpExt* partial, const Fp* __restrict__ coeffs, const uint32_t* __restrict__ which,
                         const FpExt* __restrict__ xs, const FpExt* __restrict__ pw,
                         const FpExt* __restrict__ ps, uint32_t deg, uint32_t chunk);
__global__ void eu2_reduce(FpExt* out, const FpExt* partial, uint32_t S);
__global__ void mix_poly_coeffs_v2(FpExt* __restrict__ out, const Fp* __restrict__ in,
                                   const uint32_t* __restrict__ combos,
                                   const FpExt* __restrict__ mixStart,
                                   const FpExt* __restrict__ mix, const uint32_t inputSize,
                                   const uint32_t count);
#ifdef __HIPCC__
__global__ void mix_poly_coeffs_v6(FpExt* __restrict__ out, const Fp* __restrict__ in,
                                   const uint32_t* __restrict__ combos,
                                   const FpExt* __restrict__ mixStart,
                                   const FpExt* __restrict__ mix, const uint32_t inputSize,
                                   const uint32_t count);
#endif
__global__ void gather_samples(Fp* dst, const Fp* src, const uint32_t* idxs, const uint32_t n,
                               const uint32_t size, const uint32_t stride);
__global__ void combos_sub(FpExt* combos, const FpExt* deltas, const uint32_t* offsets,
                           const uint32_t count);

extern "C" {

void risc0_zkp_cuda_set_oom_hook(void (*hook)()) {
  risc0_cuda_oom_hook = hook;
}

// Trigger CUDA module loading for risc0-zkp kernels (eltwise, sha, etc.)
// by launching a dummy kernel. All kernels in this compilation unit share
// the same CUDA module, so loading one loads all.
const char* risc0_zkp_cuda_warmup() {
  try {
    cudaStream_t stream = getPersistentStream();
    // batch_bit_reverse with count=0 exits immediately (bounds check).
    batch_bit_reverse<<<1, 1, 0, stream>>>(nullptr, 0, 0);
    CUDA_OK(cudaStreamSynchronize(stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("warmup failed");
  }
  return nullptr;
}

// Sync the persistent risc0 stream. Must be called before sppark operations
// that read from buffers written by risc0 kernels (different CUDA stream).
const char* risc0_zkp_cuda_sync_stream() {
  try {
    CUDA_OK(cudaStreamSynchronize(getPersistentStream()));
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Async H2D copy: uses persistent stream to avoid device-wide sync from hipMemcpy.
const char* risc0_zkp_cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
  try {
    cudaStream_t stream = getPersistentStream();
    CUDA_OK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Async fill: uses persistent stream to avoid device-wide sync from hipMemsetD32.
const char* risc0_zkp_cuda_fill_u32(uint32_t* buf, uint32_t value, uint32_t count) {
  try {
    cudaStream_t stream = getPersistentStream();
    if (value == 0) {
      CUDA_OK(cudaMemsetAsync(buf, 0, (size_t)count * 4, stream));
    } else {
#ifdef __HIPCC__
      CUDA_OK(hipMemsetD32Async(buf, value, count, stream));
#else
      // cuMemsetD32Async returns CUresult (driver API) but CUDA_OK expects
      // cudaError_t (runtime API). Cast through the compatible error code.
      CUresult res = cuMemsetD32Async((CUdeviceptr)buf, value, count, stream);
      if (res != CUDA_SUCCESS) {
        throw std::runtime_error(fmt("cuMemsetD32Async failed: %d", (int)res));
      }
#endif
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

#ifdef __HIPCC__
// The prefill queue: a dedicated hardware queue limited to RISC0_HIP_PREFILL_CUS compute units
// (default 2, one WGP; 0 = all of them). A full-device fill beside the
// latency-bound eval_check prologue (ef_prep: one thread, then constant-table copies) slowed it
// from ~0.65 to ~2.7 ms on an RX 9070 XT by saturating DRAM; one WGP fills ~1 GB in ~10-20 ms
// beside ef_eval_split (~160 ms) instead. Callers hold risc0_hip_hop::lock().
static hipStream_t prefill_stream(int dev) {
  static hipStream_t streams[16] = {};
  static int state[16] = {};
  if (dev < 0 || dev >= 16)
    return nullptr;
  if (!state[dev]) {
    state[dev] = -1;
    const char* v = std::getenv("RISC0_HIP_PREFILL_CUS");
    int n = v && *v ? std::atoi(v) : 2;
    uint32_t cus[8];
    for (int i = 0; i < 8; i++) {
      int lo = 32 * i;
      cus[i] = n <= 0 || n >= lo + 32 ? ~0u : (n > lo ? (1u << (n - lo)) - 1 : 0u);
    }
    if (hipExtStreamCreateWithCUMask(&streams[dev], 8, cus) == hipSuccess) {
      state[dev] = 1;
    } else {
      (void)hipGetLastError();
      streams[dev] = nullptr;
    }
  }
  return streams[dev];
}

// Fill forked off the persistent stream onto the prefill queue: it runs after everything queued on
// the persistent stream so far, beside later work. *done is the event risc0_zkp_hip_fill_join
// waits for (nullptr: the fill was queued on the persistent stream).
const char* risc0_zkp_hip_fill_fork(uint32_t* buf, uint32_t value, size_t count, void** done) {
  try {
    hipStream_t main = getPersistentStream();
    *done = nullptr;
    std::unique_lock<std::recursive_mutex> lk(risc0_hip_hop::lock());
    int dev = -1;
    hipStream_t side = hipGetDevice(&dev) == hipSuccess ? prefill_stream(dev) : nullptr;
    static hipEvent_t forkEv[16] = {};
    hipEvent_t ev = nullptr;
    if (side && dev < 16 && (forkEv[dev] || hipEventCreateWithFlags(&forkEv[dev], hipEventDisableTiming) == hipSuccess) &&
        hipEventCreateWithFlags(&ev, hipEventDisableTiming) == hipSuccess &&
        hipEventRecord(forkEv[dev], main) == hipSuccess && hipStreamWaitEvent(side, forkEv[dev], 0) == hipSuccess) {
      CUDA_OK(value == 0 ? hipMemsetAsync(buf, 0, count * 4, side) : hipMemsetD32Async(buf, value, count, side));
      CUDA_OK(hipEventRecord(ev, side));
      *done = ev;
      return nullptr;
    }
    (void)hipGetLastError();
    if (ev)
      (void)hipEventDestroy(ev);
    CUDA_OK(value == 0 ? hipMemsetAsync(buf, 0, count * 4, main) : hipMemsetD32Async(buf, value, count, main));
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Order the persistent stream after a forked fill (host = 1: wait on the host instead) and release
// its event.
const char* risc0_zkp_hip_fill_join(void* done, uint32_t host) {
  try {
    CUDA_OK(risc0_hip_hop::join_fill(host ? nullptr : getPersistentStream(), (hipEvent_t)done));
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}
#endif

const char* risc0_zkp_cuda_eltwise_add_fp(Fp* out, const Fp* x, const Fp* y, uint32_t count) {
  return launchKernel(eltwise_add_fp, count, 0, out, x, y, count);
}

const char* risc0_zkp_cuda_eltwise_mul_factor_fp(Fp* io, Fp factor, uint32_t count) {
  return launchKernel(eltwise_mul_factor_fp, count, 0, io, factor, count);
}

const char* risc0_zkp_cuda_eltwise_copy_fp(Fp* out, const Fp* in, const uint32_t count) {
  return launchKernel(eltwise_copy_fp, count, 0, out, in, count);
}

const char* risc0_zkp_cuda_eltwise_copy_fp_region(Fp* into,
                                                  const Fp* from,
                                                  const uint32_t fromRows,
                                                  const uint32_t fromCols,
                                                  const uint32_t fromOffset,
                                                  const uint32_t fromStride,
                                                  const uint32_t intoOffset,
                                                  const uint32_t intoStride) {
  return launchKernel(eltwise_copy_fp_region,
                      fromRows,
                      0,
                      into,
                      from,
                      fromRows,
                      fromCols,
                      fromOffset,
                      fromStride,
                      intoOffset,
                      intoStride);
}

const char* risc0_zkp_cuda_eltwise_sum_fpext(Fp* out,
                                             const FpExt* in,
                                             const uint32_t to_add,
                                             const uint32_t count) {
  return launchKernel(eltwise_sum_fpext, count, 0, out, in, to_add, count);
}

const char* risc0_zkp_cuda_eltwise_zeroize_fp(Fp* elems, const uint32_t count) {
  return launchKernel(eltwise_zeroize_fp, count, 0, elems, count);
}

const char* risc0_zkp_cuda_eltwise_zeroize_fpext(FpExt* elems, const uint32_t count) {
  return launchKernel(eltwise_zeroize_fpext, count, 0, elems, count);
}

const char* risc0_zkp_cuda_fri_fold(Fp* out, const Fp* in, const FpExt* mix, const uint32_t count) {
  return launchKernel(fri_fold, count, 0, out, in, mix, count);
}

const char* risc0_zkp_cuda_mix_poly_coeffs(FpExt* out,
                                           const Fp* in,
                                           const uint32_t* combos,
                                           const FpExt* mixStart,
                                           const FpExt* mix,
                                           const uint32_t inputSize,
                                           const uint32_t count) {
  // v2 keeps the mix powers and combo ids in shared memory (20 bytes per input).
#ifdef __HIPCC__
  // ROCm: v6 (2 elements per thread, per-combo register accumulators; slower below 2^16 rows)
  // unless RISC0_HIP_MIX_V6=0.
  static const bool v6 = [] {
    const char* e = getenv("RISC0_HIP_MIX_V6");
    return !(e && e[0] == '0');
  }();
  if (v6 && inputSize <= 2048 && count >= (1u << 16) && (count % 2) == 0 &&
      (reinterpret_cast<uintptr_t>(in) & 7) == 0) {
    // >= 2^20 rows (~1.2 ms): on a side hardware queue (hip_hop.h).
    return launchKernelHop(risc0_hip_hop::want(risc0_hip_hop::kMix, count, 20), mix_poly_coeffs_v6, count / 2,
                           inputSize * (sizeof(FpExt) + sizeof(uint32_t)), out, in, combos, mixStart,
                           mix, inputSize, count);
  }
#endif
  if (inputSize <= 2048) {
    return launchKernel(mix_poly_coeffs_v2, count, inputSize * (sizeof(FpExt) + sizeof(uint32_t)),
                        out, in, combos, mixStart, mix, inputSize, count);
  }
  return launchKernel(mix_poly_coeffs, count, inputSize * sizeof(FpExt),
                      out, in, combos, mixStart, mix, inputSize, count);
}

const char* risc0_zkp_cuda_gather_samples(Fp* dst, const Fp* src, const uint32_t* idxs,
                                          const uint32_t n, const uint32_t size,
                                          const uint32_t stride) {
  return launchKernel(gather_samples, n * size, 0, dst, src, idxs, n, size, stride);
}

const char* risc0_zkp_cuda_combos_sub(FpExt* combos,
                                      const FpExt* deltas,
                                      const uint32_t* offsets,
                                      const uint32_t count) {
  return launchKernel(combos_sub, count, 0, combos, deltas, offsets, count);
}

const char* risc0_zkp_cuda_batch_bit_reverse(Fp* io, const uint32_t nBits, const uint32_t count) {
  return launchKernel(batch_bit_reverse, count, 0, io, nBits, count);
}

const char* risc0_zkp_cuda_batch_evaluate_any(FpExt* out,
                                              const Fp* coeffs,
                                              const uint32_t* which,
                                              const FpExt* xs,
                                              uint32_t shared_size,
                                              const uint32_t count,
                                              const uint32_t deg,
                                              const uint32_t nEvals) {
  // eval_u v2: split-K over (chunk, eval point) with per-point power tables. Falls back to
  // the original kernel for small or unaligned inputs.
  const uint32_t minChunk = EU_J * EU_THREADS;
  if (nEvals == 0 || deg < minChunk || (deg % minChunk) != 0 ||
      (reinterpret_cast<uintptr_t>(coeffs) & 15) != 0) {
    return launchKernel(batch_evaluate_any, count, shared_size, out, coeffs, which, xs, deg);
  }
  try {
    cudaStream_t stream = getPersistentStream();
#ifdef __HIPCC__
    // ROCm (lazy eu2_main): fewer, longer blocks measured ~5% faster on the rv32im data group.
    const uint32_t targetBlocks = 512;
#else
    const uint32_t targetBlocks = 1024;
#endif
    uint32_t maxS = deg / minChunk;
    uint32_t S = 1;
    while (S < maxS && (uint64_t)S * nEvals < targetBlocks)
      S <<= 1;
    uint32_t chunk = deg / S;
    size_t pwBytes = (size_t)nEvals * EU_THREADS * sizeof(FpExt);
    size_t psBytes = (size_t)nEvals * S * sizeof(FpExt);
    char* scratch = nullptr;
#ifdef __HIPCC__
    // A grow-only scratch buffer per device instead of the stream-ordered pool: the launches
    // that use it are ordered on the persistent stream, and growing frees with an implicit
    // device sync.
    {
      static char* s_scratch[16] = {};
      static size_t s_scratch_bytes[16] = {};
      int dev = getCachedDevice();
      if (pwBytes + 2 * psBytes > s_scratch_bytes[dev]) {
        if (s_scratch[dev])
          CUDA_OK(cudaFree(s_scratch[dev]));
        CUDA_OK(cudaMalloc(&s_scratch[dev], pwBytes + 2 * psBytes));
        s_scratch_bytes[dev] = pwBytes + 2 * psBytes;
      }
      scratch = s_scratch[dev];
    }
#else
    // Stream-ordered scratch: safe against concurrent callers sharing the stream.
    CUDA_OK(cudaMallocAsync(&scratch, pwBytes + 2 * psBytes, stream));
#endif
    FpExt* pw = reinterpret_cast<FpExt*>(scratch);
    FpExt* ps = reinterpret_cast<FpExt*>(scratch + pwBytes);
    FpExt* part = reinterpret_cast<FpExt*>(scratch + pwBytes + psBytes);
    eu2_pows<<<nEvals, EU_THREADS, 0, stream>>>(pw, ps, xs, S, chunk);
    eu2_main<<<dim3(S, nEvals), EU_THREADS, 0, stream>>>(part, coeffs, which, xs, pw, ps, deg, chunk);
    eu2_reduce<<<nEvals, 32, 0, stream>>>(out, part, S);
    CUDA_OK(cudaGetLastError());
#ifndef __HIPCC__
    CUDA_OK(cudaFreeAsync(scratch, stream));
#endif
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_zkp_cuda_gather_sample(
    Fp* dst, const Fp* src, const uint32_t idx, const uint32_t size, const uint32_t stride) {
  return launchKernel(gather_sample, size, 0, dst, src, idx, size, stride);
}

const char* risc0_zkp_cuda_scatter(Fp* into,
                                   const uint32_t* index,
                                   const uint32_t* offsets,
                                   const Fp* values,
                                   const uint32_t count) {
  return launchKernel(scatter, count, 0, into, index, offsets, values, count);
}

const char* risc0_zkp_cuda_scatter_bits(Fp* into,
                                        const uint32_t* data,
                                        const uint32_t cycles,
                                        const uint32_t count) {
  return launchKernel(scatter_bits, count, 0, into, data, cycles, count);
}

// Scatter from host memory using persistent device + pinned host buffers.
// Pinned host buffers avoid HIP's slow pageable memory staging path.
const char* risc0_zkp_cuda_scatter_from_host(Fp* into,
                                             const uint32_t* h_index,
                                             uint32_t index_count,
                                             const uint32_t* h_offsets,
                                             uint32_t offsets_count,
                                             const Fp* h_values,
                                             uint32_t values_count,
                                             uint32_t count) {
  try {
    cudaStream_t stream = getPersistentStream();
    int dev = getCachedDevice();

    // Per-device persistent device buffers (grow-only, reused across segments).
    static uint32_t* d_index[16] = {};
    static uint32_t* d_offsets[16] = {};
    static Fp* d_values[16] = {};
    static size_t cap_d_index[16] = {}, cap_d_offsets[16] = {}, cap_d_values[16] = {};

    // Per-device persistent pinned host staging buffers (grow-only with 25% headroom, huge
    // pages): segment sizes vary by a few percent, and each regrow is a synchronizing free.
    static ThpPinned h_pinned[16];

    size_t index_bytes = index_count * sizeof(uint32_t);
    size_t offsets_bytes = offsets_count * sizeof(uint32_t);
    size_t values_bytes = values_count * sizeof(Fp);
    size_t total_bytes = index_bytes + offsets_bytes + values_bytes;

    // Grow device buffers if needed.
    if (index_bytes > cap_d_index[dev]) {
      if (d_index[dev]) CUDA_OK(cudaFree(d_index[dev]));
      CUDA_OK(cudaMalloc(&d_index[dev], index_bytes));
      cap_d_index[dev] = index_bytes;
    }
    if (offsets_bytes > cap_d_offsets[dev]) {
      if (d_offsets[dev]) CUDA_OK(cudaFree(d_offsets[dev]));
      CUDA_OK(cudaMalloc(&d_offsets[dev], offsets_bytes));
      cap_d_offsets[dev] = offsets_bytes;
    }
    if (values_bytes > cap_d_values[dev]) {
      if (d_values[dev]) CUDA_OK(cudaFree(d_values[dev]));
      CUDA_OK(cudaMalloc(&d_values[dev], values_bytes));
      cap_d_values[dev] = values_bytes;
    }

    // Grow pinned host staging buffer if needed.
    if (total_bytes > h_pinned[dev].bytes) {
      h_pinned[dev].alloc(total_bytes + total_bytes / 4);
    }

    // Copy to pinned staging buffer (fast CPU memcpy).
    char* pin = (char*)h_pinned[dev].ptr;
    memcpy(pin, h_index, index_bytes);
    memcpy(pin + index_bytes, h_offsets, offsets_bytes);
    memcpy(pin + index_bytes + offsets_bytes, h_values, values_bytes);

    // Truly async H2D transfers from pinned memory.
    CUDA_OK(cudaMemcpyAsync(d_index[dev], pin, index_bytes,
                            cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_offsets[dev], pin + index_bytes, offsets_bytes,
                            cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_values[dev], pin + index_bytes + offsets_bytes, values_bytes,
                            cudaMemcpyHostToDevice, stream));

    // Launch scatter kernel on same stream (waits for DMA implicitly).
    LaunchConfig cfg = getCachedSimpleConfig(count);
    scatter<<<cfg.grid, cfg.block, 0, stream>>>(
        into, d_index[dev], d_offsets[dev], d_values[dev], count);

  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

// Scatter bits from host memory using persistent device + pinned host buffer.
const char* risc0_zkp_cuda_scatter_bits_from_host(Fp* into,
                                                  const uint32_t* h_data,
                                                  uint32_t triplet_count,
                                                  uint32_t cycles) {
  try {
    cudaStream_t stream = getPersistentStream();
    int dev = getCachedDevice();

    // Per-device persistent device buffer (grow-only, reused across segments).
    static uint32_t* d_bitdata[16] = {};
    static size_t cap_d_bitdata[16] = {};

    // Per-device persistent pinned host staging buffer (grow-only, huge pages).
    static ThpPinned h_pinned_bits[16];

    size_t data_bytes = (size_t)triplet_count * 3 * sizeof(uint32_t);

    if (data_bytes > cap_d_bitdata[dev]) {
      if (d_bitdata[dev]) CUDA_OK(cudaFree(d_bitdata[dev]));
      CUDA_OK(cudaMalloc(&d_bitdata[dev], data_bytes));
      cap_d_bitdata[dev] = data_bytes;
    }

    // Grow pinned host staging buffer if needed.
    if (data_bytes > h_pinned_bits[dev].bytes) {
      h_pinned_bits[dev].alloc(data_bytes + data_bytes / 4);
    }

    // Copy to pinned staging and do truly async H2D.
    memcpy(h_pinned_bits[dev].ptr, h_data, data_bytes);
    CUDA_OK(cudaMemcpyAsync(d_bitdata[dev], h_pinned_bits[dev].ptr, data_bytes,
                            cudaMemcpyHostToDevice, stream));

    // Launch scatter_bits kernel on same stream.
    LaunchConfig cfg = getCachedSimpleConfig(triplet_count);
    scatter_bits<<<cfg.grid, cfg.block, 0, stream>>>(
        into, d_bitdata[dev], cycles, triplet_count);

  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

const char*
risc0_zkp_cuda_sha_rows(ShaDigest* output, const Fp* matrix, uint32_t rowSize, uint32_t colSize) {
  return launchKernel(sha_rows, rowSize, 0, output, matrix, rowSize, colSize);
}

const char* risc0_zkp_cuda_sha_fold(ShaDigest* output, const ShaDigest* input, uint32_t count) {
  return launchKernel(sha_fold, count, 0, output, input, count);
}

const char* risc0_zkp_cuda_gather_digests(
    uint32_t* dst, const uint32_t* src, const uint32_t* indices, uint32_t count) {
  return launchKernel(gather_digests, count, 0, dst, src, indices, count);
}

const char* risc0_zkp_cuda_combos_prepare(FpExt* combos,
                                          const FpExt* coeffU,
                                          const uint32_t comboCount,
                                          const uint32_t cycles,
                                          const uint32_t regsCount,
                                          const uint32_t* regSizes,
                                          const uint32_t* regComboIds,
                                          const uint32_t checkSize,
                                          const FpExt* mix) {

  try {
    cudaStream_t stream = getPersistentStream();
    combos_prepare<<<1, 1, 0, stream>>>(
        combos, coeffU, regsCount, regSizes, regComboIds, cycles, mix, checkSize, comboCount);
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

} // extern "C"
