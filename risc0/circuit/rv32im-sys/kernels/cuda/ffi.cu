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
#include "steps.cuh"
#include "witgen.h"
#include "thp_pinned.h"
#include "device_alloc.h"

#ifndef __HIPCC__
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#endif

#include "vendor/nvtx3/nvtx3.hpp"

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#else
namespace nvtx3 { struct scoped_range { scoped_range(const char*) {} }; }
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <string.h>

#ifdef __HIPCC__
#include <array>
namespace cuda { namespace std { using ::std::array; } }
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
// cudaEventSynchronize is aliased by some sppark versions' cuda2hip.hpp and not others: use a
// name of this file's own.
static const auto rv32imEventSynchronize = hipEventSynchronize;
#else
#include <cuda/std/array>
static const auto rv32imEventSynchronize = cudaEventSynchronize;
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#endif

void risc0_rv32im_launch_stepAccumTU(unsigned, unsigned, cudaStream_t, void*, uint32_t, const uint32_t*);
#ifndef __HIPCC__
void risc0_rv32im_launch_par_stepExecTU(unsigned, unsigned, cudaStream_t, void*, uint32_t, uint32_t);
#else
void risc0_rv32im_launch_par_stepExecTU(unsigned, unsigned, cudaStream_t, void*, uint32_t, uint32_t,
                                        const uint32_t*);
void risc0_rv32im_launch_par_stepExecTU2(unsigned, unsigned, cudaStream_t, void*, uint32_t, uint32_t,
                                         const uint32_t*);
#endif
static bool g_verbose = (std::getenv("RISC0_VERBOSE") != nullptr);

namespace risc0::circuit::rv32im_v2::cuda {

#ifdef __HIPCC__
// Addition operator for hipcub::DeviceScan::InclusiveScan on Fp elements
struct AddOp {
  __device__ __forceinline__ Fp operator()(const Fp& a, const Fp& b) const { return a + b; }
};
#else
struct FpAddOp {
  __device__ __forceinline__ Fp operator()(const Fp& a, const Fp& b) const { return a + b; }
};

// Grow-only per-device CUB temp storage. thrust's temporary allocator does cudaMalloc/cudaFree
// per call (even with par_nosync) and cudaFree synchronizes: the host blocked for the whole of
// stepAccum and the GPU idled around each call (5 per segment).
static void* cubTemp(size_t bytes) {
  static void* s_ptr[16] = {};
  static size_t s_cap[16] = {};
  int dev = getCachedDevice();
  if (bytes > s_cap[dev]) {
    if (s_ptr[dev]) CUDA_OK(cudaFree(s_ptr[dev]));
    s_ptr[dev] = nullptr;
    s_cap[dev] = 0;
    CUDA_OK(risc0_cuda_malloc(&s_ptr[dev], bytes));
    s_cap[dev] = bytes;
  }
  return s_ptr[dev];
}
#endif

constexpr size_t kUserAccumSplit = kLayout_TopAccum.columns[0].col;

struct ExecBuffers {
  Buffer global;
  Buffer data;
  Buffer pre_data;
};

struct DeviceExecContext {
  Buffer* data;
  Buffer* pre_data;
  Buffer* global;
  PreflightTrace* preflight;
  LookupTables* tables;
};

struct AccumBuffers {
  Buffer data;
  Buffer accum;
  Buffer global;
  Buffer mix;
};

struct DeviceAccumContext {
  Buffer* data;
  Buffer* accum;
  Buffer* global;
  Buffer* mix;
  PreflightTrace* preflight;
  LookupTables* tables;
};

// Device copies of staged (pinned) preflight traces, one grow-only device buffer per pinned
// slot. A staged trace's copy is either prefetched on a side stream while the previous
// segment's finalize runs (risc0_circuit_rv32im_cuda_prefetch), or made on the main stream in
// setup_exec. `releasedEv` is recorded after the last kernel that reads the slot (accum), and a
// later prefetch into the same slot waits for it. `id` identifies the staged trace whose bytes
// the slot holds (pinned addresses can be reused after a regrow).
struct DevSlot {
  const uint8_t* pinned = nullptr;
  uint8_t* dev = nullptr;
  size_t cap = 0;
  uint64_t id = 0;
  bool ready = false;  // prefetched for `id`, not yet consumed
  bool released = false;
  bool in_use = false;  // the current segment's device preflight (setup_exec .. accum)
  cudaEvent_t readyEv = nullptr;
  cudaEvent_t releasedEv = nullptr;
};

// Guards the slots. Recursive: a slot's device allocation can run the HAL's out-of-memory hook,
// which calls risc0_circuit_rv32im_cuda_release_slots on the same thread.
static std::recursive_mutex g_slots_mu;
static DevSlot s_slots[16][8];

// Staging targets the process's CUDA device (getCachedDevice, like the rest of this library;
// the HAL proves on device 0). If a thread proves with another device current, the staged
// traces (pinned in another context) and their device slots are not used from then on: setup_exec
// stages on the proving thread as without staging.
static std::atomic<bool> g_staging_off{false};
static bool stagingDeviceOk() {
  if (g_staging_off.load(std::memory_order_relaxed)) return false;
  int cur = 0;
  if (cudaGetDevice(&cur) != cudaSuccess || cur != getCachedDevice()) {
    g_staging_off = true;
    return false;
  }
  return true;
}

static DevSlot* getDevSlot(const uint8_t* pinned) {
  int dev = getCachedDevice();
  DevSlot* freeSlot = nullptr;
  for (auto& sl : s_slots[dev]) {
    if (sl.pinned == pinned) return &sl;
    if (!sl.pinned && !freeSlot) freeSlot = &sl;
  }
  if (!freeSlot) {
    // More pinned slots than expected (regrows): recycle the least useful one (not ready).
    for (auto& sl : s_slots[dev]) {
      if (!sl.ready && !sl.in_use) { freeSlot = &sl; break; }
    }
    if (!freeSlot) throw std::runtime_error("rv32im prefetch: no free device slot");
  }
  freeSlot->pinned = pinned;
  freeSlot->ready = false;
  if (!freeSlot->readyEv) {
    CUDA_OK(cudaEventCreateWithFlags(&freeSlot->readyEv, cudaEventDisableTiming));
    CUDA_OK(cudaEventCreateWithFlags(&freeSlot->releasedEv, cudaEventDisableTiming));
  }
  return freeSlot;
}

static void growDevSlot(DevSlot* sl, size_t bytes) {
  if (bytes > sl->cap) {
    // cudaFree synchronizes the device, so no pending copy or kernel still uses the old buffer.
    if (sl->dev) CUDA_OK(cudaFree(sl->dev));
    sl->dev = nullptr;
    sl->cap = 0;
    CUDA_OK(risc0_cuda_malloc(&sl->dev, bytes));
    sl->cap = bytes;
  }
}

static cudaStream_t getCopyStream() {
  static cudaStream_t s_streams[16] = {};
  int dev = getCachedDevice();
#ifdef __HIPCC__
  // At the lowest priority, which ROCm serves from hardware queues of its own: on a queue shared
  // with other streams (GPU_MAX_HW_QUEUES, 4), the event recorded after a prefetch copy held
  // their kernels and copies until the copy was done (~9 ms at po2=20).
  if (!s_streams[dev]) {
    int least = 0, greatest = 0;
    CUDA_OK(hipDeviceGetStreamPriorityRange(&least, &greatest));
    CUDA_OK(hipStreamCreateWithPriority(&s_streams[dev], hipStreamNonBlocking, least));
  }
#else
  if (!s_streams[dev]) CUDA_OK(cudaStreamCreateWithFlags(&s_streams[dev], cudaStreamNonBlocking));
#endif
  return s_streams[dev];
}

// Static cache for device-side allocations that persist across segments.
// Eliminates ~42 cudaMallocAsync/cudaFreeAsync pairs per segment and avoids
// re-uploading ~18MB of preflight data in the accum phase (which doesn't use it).
struct DeviceCache {
  // Exec context (device pointers, reused across segments)
  DeviceExecContext* exec_ctx = nullptr;
  Buffer* d_exec_data = nullptr;
  Buffer* d_exec_pre_data = nullptr;
  Buffer* d_exec_global = nullptr;
  PreflightTrace* d_exec_preflight = nullptr;
  LookupTables* d_exec_tables = nullptr;

  // Accum context (device pointers, reused across segments)
  DeviceAccumContext* accum_ctx = nullptr;
  Buffer* d_accum_data = nullptr;
  Buffer* d_accum_accum = nullptr;
  Buffer* d_accum_global = nullptr;
  Buffer* d_accum_mix = nullptr;
  PreflightTrace* d_accum_preflight = nullptr;
  LookupTables* d_accum_tables = nullptr;

  // Shared lookup table memory (reused, zeroed between phases)
  LookupCell* d_tableU8 = nullptr;
  LookupCell* d_tableU16 = nullptr;

  // Variable-size preflight data (grow-only, shared between exec and accum)
  PreflightCycle* d_cycles = nullptr;
  size_t cycles_capacity = 0;
  MemoryTransaction* d_txns = nullptr;
  size_t txns_capacity = 0;
  uint8_t* d_bigintBytes = nullptr;
  size_t bigintBytes_capacity = 0;

  // The device preflight of the current segment: the own buffers above, or a DevSlot.
  PreflightCycle* a_cycles = nullptr;
  MemoryTransaction* a_txns = nullptr;
  uint8_t* a_bigintBytes = nullptr;
  DevSlot* a_slot = nullptr;

  // Persistent pinned host staging buffer for preflight data uploads (grow-only, huge pages).
  // Avoids HIP's slow pageable memory staging path for large H2D copies.
  ThpPinned pf_pin;


  void init(cudaStream_t stream) {
    if (exec_ctx) return;  // already initialized
    // Fixed-size allocations for exec context
    CUDA_OK(cudaMallocAsync(&exec_ctx, sizeof(DeviceExecContext), stream));
    CUDA_OK(cudaMallocAsync(&d_exec_data, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_exec_pre_data, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_exec_global, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_exec_preflight, sizeof(PreflightTrace), stream));
    CUDA_OK(cudaMallocAsync(&d_exec_tables, sizeof(LookupTables), stream));
    // Fixed-size allocations for accum context
    CUDA_OK(cudaMallocAsync(&accum_ctx, sizeof(DeviceAccumContext), stream));
    CUDA_OK(cudaMallocAsync(&d_accum_data, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_accum_accum, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_accum_global, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_accum_mix, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_accum_preflight, sizeof(PreflightTrace), stream));
    CUDA_OK(cudaMallocAsync(&d_accum_tables, sizeof(LookupTables), stream));
    // Shared lookup tables. These ~257 KB total (1 KB U8 + 256 KB U16) are
    // hammered by every cycle in par_stepExec/stepAccum across millions of
    // cycles per segment. Pin them into Ada's persisting-L2 cache window to
    // keep them resident across kernel launches.
    CUDA_OK(cudaMallocAsync(&d_tableU8, (1 << 8) * sizeof(uint32_t), stream));
    CUDA_OK(cudaMallocAsync(&d_tableU16, (1 << 16) * sizeof(uint32_t), stream));
#ifndef __HIPCC__
    // CUDA-only persisting-L2 cache tuning (Ampere+). HIP/AMD has no equivalent of
    // cudaLimitPersistingL2CacheSize / cudaStreamAttributeAccessPolicyWindow, so
    // skip this pure performance optimization on ROCm.
    //
    // A best-effort hint, so the carve-out is clamped to what the card allows and every error
    // is cleared. A fixed 24 MB carve-out fails with "invalid argument" on a card that allows
    // less (20 MB on an RTX 5060 Ti, none before Ampere), and the next
    // CUDA_OK(cudaGetLastError()) in witgen then reported it as its own launch failure: the
    // first proof of every fresh process failed there.
    {
      int dev = 0, persist_hw_max = 0, window_hw_max = 0;
      if (cudaGetDevice(&dev) == cudaSuccess) {
        (void)cudaDeviceGetAttribute(&persist_hw_max, cudaDevAttrMaxPersistingL2CacheSize, dev);
        (void)cudaDeviceGetAttribute(&window_hw_max, cudaDevAttrMaxAccessPolicyWindowSize, dev);
      }
      if (persist_hw_max > 0 && window_hw_max > 0) {
        static bool s_persisting_set = false;
        if (!s_persisting_set) {
          size_t l2_persist_max = 0;
          (void)cudaDeviceGetLimit(&l2_persist_max, cudaLimitPersistingL2CacheSize);
          // 24 MB carve-out where the card allows it (fits 257 KB easily)
          size_t want = std::min((size_t)24 << 20, (size_t)persist_hw_max);
          if (l2_persist_max < want) {
            (void)cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want);
          }
          s_persisting_set = true;
        }
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr = d_tableU16;
        attr.accessPolicyWindow.num_bytes =
            std::min(((size_t)((1 << 16) * sizeof(uint32_t))) + ((1 << 8) * sizeof(uint32_t)),
                     (size_t)window_hw_max);
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        (void)cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
      }
      (void)cudaGetLastError();
    }
#endif
  }

  // Grow-only reallocation for variable-size preflight buffers
  void ensure_preflight(cudaStream_t stream, size_t nCycles, size_t txnsLen, size_t bigintBytesLen) {
    if (nCycles > cycles_capacity) {
      if (d_cycles) CUDA_OK(cudaFreeAsync(d_cycles, stream));
      CUDA_OK(cudaMallocAsync(&d_cycles, nCycles * sizeof(PreflightCycle), stream));
      cycles_capacity = nCycles;
    }
    if (txnsLen > txns_capacity) {
      if (d_txns) CUDA_OK(cudaFreeAsync(d_txns, stream));
      CUDA_OK(cudaMallocAsync(&d_txns, txnsLen * sizeof(MemoryTransaction), stream));
      txns_capacity = txnsLen;
    }
    if (bigintBytesLen > bigintBytes_capacity) {
      if (d_bigintBytes) CUDA_OK(cudaFreeAsync(d_bigintBytes, stream));
      CUDA_OK(cudaMallocAsync(&d_bigintBytes, bigintBytesLen, stream));
      bigintBytes_capacity = bigintBytesLen;
    }
  }

  // Setup exec context: upload buffers + preflight data + zero tables
  // `staged`: optional pinned host buffer that already holds cycles|txns|bigintBytes in the
  // layout below (filled off the critical path by the preflight thread); the caller keeps it
  // alive until the stream has passed the copies.
  DeviceExecContext* setup_exec(ExecBuffers* buffers, PreflightTrace* preflight,
                                size_t lastCycle, cudaStream_t stream,
                                const char* staged = nullptr, uint64_t staged_id = 0) {
    init(stream);

    // Upload buffer descriptors
    CUDA_OK(cudaMemcpyAsync(d_exec_data, &buffers->data, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_exec_pre_data, &buffers->pre_data, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_exec_global, &buffers->global, sizeof(Buffer), cudaMemcpyHostToDevice, stream));

    // Upload preflight data via pinned staging buffer for truly async H2D.
    size_t cycles_bytes = lastCycle * sizeof(PreflightCycle);
    size_t txns_bytes = preflight->txnsLen * sizeof(MemoryTransaction);
    size_t bigint_bytes = preflight->bigintBytesLen;
    size_t total_pf_bytes = cycles_bytes + txns_bytes + bigint_bytes;

    char* pin = stagingDeviceOk() ? const_cast<char*>(staged) : nullptr;
    a_slot = nullptr;
    std::unique_lock<std::recursive_mutex> slots_lk(g_slots_mu, std::defer_lock);
    if (pin) {
      slots_lk.lock();
      DevSlot* sl = getDevSlot(reinterpret_cast<const uint8_t*>(pin));
      if (sl->ready && sl->id == staged_id && sl->cap >= total_pf_bytes) {
        // Prefetched on the copy stream.
        CUDA_OK(cudaStreamWaitEvent(stream, sl->readyEv, 0));
      } else {
        // A stale prefetch (another trace at this address) may still be writing the slot.
        if (sl->ready) CUDA_OK(cudaStreamWaitEvent(stream, sl->readyEv, 0));
        growDevSlot(sl, total_pf_bytes);
        CUDA_OK(cudaMemcpyAsync(sl->dev, pin, total_pf_bytes, cudaMemcpyHostToDevice, stream));
        sl->id = staged_id;
      }
      sl->ready = false;
      sl->in_use = true;
      a_slot = sl;
      a_cycles = reinterpret_cast<PreflightCycle*>(sl->dev);
      a_txns = reinterpret_cast<MemoryTransaction*>(sl->dev + cycles_bytes);
      a_bigintBytes = sl->dev + cycles_bytes + txns_bytes;
    } else {
    ensure_preflight(stream, lastCycle, preflight->txnsLen, preflight->bigintBytesLen);
    a_cycles = d_cycles;
    a_txns = d_txns;
    a_bigintBytes = d_bigintBytes;
    // Grow pinned staging buffer if needed.
    // Grow with 25% headroom: the trace grows over the first segments, and every regrow
    // costs ~20 ms of pinning on the critical path plus cudaFreeHost's implicit device
    // synchronization.
    if (total_pf_bytes > pf_pin.bytes) {
      pf_pin.alloc(total_pf_bytes + total_pf_bytes / 4);
    }

    // Copy to pinned staging (fast CPU memcpy).
    pin = (char*)pf_pin.ptr;
    memcpy(pin, preflight->cycles, cycles_bytes);
    memcpy(pin + cycles_bytes, preflight->txns, txns_bytes);
    if (bigint_bytes > 0) {
      memcpy(pin + cycles_bytes + txns_bytes, preflight->bigintBytes, bigint_bytes);
    }

    // Truly async H2D from pinned memory.
    CUDA_OK(cudaMemcpyAsync(d_cycles, pin, cycles_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_txns, pin + cycles_bytes, txns_bytes, cudaMemcpyHostToDevice, stream));
    if (bigint_bytes > 0) {
      CUDA_OK(cudaMemcpyAsync(d_bigintBytes, pin + cycles_bytes + txns_bytes, bigint_bytes, cudaMemcpyHostToDevice, stream));
    }
    }

    // Build and upload PreflightTrace descriptor
    PreflightTrace h_pf;
    h_pf.cycles = a_cycles;
    h_pf.txns = a_txns;
    h_pf.bigintBytes = a_bigintBytes;
    h_pf.txnsLen = preflight->txnsLen;
    h_pf.bigintBytesLen = preflight->bigintBytesLen;
    h_pf.tableSplitCycle = preflight->tableSplitCycle;
    CUDA_OK(cudaMemcpyAsync(d_exec_preflight, &h_pf, sizeof(PreflightTrace), cudaMemcpyHostToDevice, stream));

    // Zero and upload lookup tables
    CUDA_OK(cudaMemsetAsync(d_tableU8, 0, (1 << 8) * sizeof(uint32_t), stream));
    CUDA_OK(cudaMemsetAsync(d_tableU16, 0, (1 << 16) * sizeof(uint32_t), stream));
    LookupTables h_tables;
    h_tables.tableU8 = d_tableU8;
    h_tables.tableU16 = d_tableU16;
    CUDA_OK(cudaMemcpyAsync(d_exec_tables, &h_tables, sizeof(LookupTables), cudaMemcpyHostToDevice, stream));

    // Build and upload DeviceExecContext
    DeviceExecContext h_ctx;
    h_ctx.data = d_exec_data;
    h_ctx.pre_data = d_exec_pre_data;
    h_ctx.global = d_exec_global;
    h_ctx.preflight = d_exec_preflight;
    h_ctx.tables = d_exec_tables;
    CUDA_OK(cudaMemcpyAsync(exec_ctx, &h_ctx, sizeof(DeviceExecContext), cudaMemcpyHostToDevice, stream));

    return exec_ctx;
  }

  // Setup accum context: upload buffers only, reuse preflight from exec phase.
  // The accum circuit (step_TopAccum) only uses lookupDelta/lookupCurrent - it never
  // accesses preflight cycles/txns/bigintBytes, so we skip the ~18MB re-upload.
  DeviceAccumContext* setup_accum(AccumBuffers* buffers, PreflightTrace* preflight,
                                  size_t lastCycle, cudaStream_t stream) {
    // Upload buffer descriptors
    CUDA_OK(cudaMemcpyAsync(d_accum_data, &buffers->data, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_accum_accum, &buffers->accum, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_accum_global, &buffers->global, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_accum_mix, &buffers->mix, sizeof(Buffer), cudaMemcpyHostToDevice, stream));

    // Reuse preflight descriptor from exec phase (already on device)
    // Just update tableSplitCycle in case it changed (it shouldn't, but be safe)
    PreflightTrace h_pf;
    h_pf.cycles = a_cycles;
    h_pf.txns = a_txns;
    h_pf.bigintBytes = a_bigintBytes;
    h_pf.txnsLen = preflight->txnsLen;
    h_pf.bigintBytesLen = preflight->bigintBytesLen;
    h_pf.tableSplitCycle = preflight->tableSplitCycle;
    CUDA_OK(cudaMemcpyAsync(d_accum_preflight, &h_pf, sizeof(PreflightTrace), cudaMemcpyHostToDevice, stream));

    // Zero and upload lookup tables (must be fresh for accum phase)
    CUDA_OK(cudaMemsetAsync(d_tableU8, 0, (1 << 8) * sizeof(uint32_t), stream));
    CUDA_OK(cudaMemsetAsync(d_tableU16, 0, (1 << 16) * sizeof(uint32_t), stream));
    LookupTables h_tables;
    h_tables.tableU8 = d_tableU8;
    h_tables.tableU16 = d_tableU16;
    CUDA_OK(cudaMemcpyAsync(d_accum_tables, &h_tables, sizeof(LookupTables), cudaMemcpyHostToDevice, stream));

    // Build and upload DeviceAccumContext
    DeviceAccumContext h_ctx;
    h_ctx.data = d_accum_data;
    h_ctx.accum = d_accum_accum;
    h_ctx.global = d_accum_global;
    h_ctx.mix = d_accum_mix;
    h_ctx.preflight = d_accum_preflight;
    h_ctx.tables = d_accum_tables;
    CUDA_OK(cudaMemcpyAsync(accum_ctx, &h_ctx, sizeof(DeviceAccumContext), cudaMemcpyHostToDevice, stream));

    return accum_ctx;
  }
};

static DeviceCache g_cache[16];

static inline DeviceCache& get_cache() {
  int dev = getCachedDevice();
  return g_cache[dev];
}

__device__ ::cuda::std::array<uint32_t, 2>
divide_rv32im(uint32_t numer, uint32_t denom, uint32_t signType) {
  uint32_t onesComp = (signType == 2);
  bool negNumer = signType && int32_t(numer) < 0;
  bool negDenom = signType == 1 && int32_t(denom) < 0;
  if (negNumer) {
    numer = -numer - onesComp;
  }
  if (negDenom) {
    denom = -denom - onesComp;
  }
  uint32_t quot;
  uint32_t rem;
  if (denom == 0) {
    quot = 0xffffffff;
    rem = numer;
  } else {
    quot = numer / denom;
    rem = numer % denom;
  }
  uint32_t quotNegOut = (negNumer ^ negDenom) - ((denom == 0) * negNumer);
  uint32_t remNegOut = negNumer;
  if (quotNegOut) {
    quot = -quot - onesComp;
  }
  if (remNegOut) {
    rem = -rem - onesComp;
  }
  return {quot, rem};
}

__device__ ::cuda::std::array<Val, 5> extern_getMemoryTxn(ExecContext& ctx, Val addrElem) {
  uint32_t addr = addrElem.asUInt32();
  // Plain post-increment: each thread handles exactly one cycle, so no
  // contention. atomicAdd is unnecessary and expensive on RDNA4.
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx++;
  const MemoryTransaction& txn = ctx.preflight.txns[txnIdx];

  // Assertions removed: in parallel witgen, union layout overwrites can cause
  // transient address mismatches during constraint evaluation. The eval_check
  // stage independently verifies all constraints on the final witness.
  return {
      txn.prevCycle,
      txn.prevWord & 0xffff,
      txn.prevWord >> 16,
      txn.word & 0xffff,
      txn.word >> 16,
  };
}

__device__ void extern_lookupDelta(ExecContext& ctx, Val table, Val index, Val count) {
  // printf("lookupDelta(table: %u, index: %u, count: %u, P: %u)\n",
  //        table.asUInt32(),
  //        index.asUInt32(),
  //        count.asUInt32(),
  //        Fp::P);
  ctx.tables.lookupDelta(table, index, count);
}

__device__ Val extern_lookupCurrent(ExecContext& ctx, Val table, Val index) {
  Val ret = ctx.tables.lookupCurrent(table, index);
  // printf("lookupCurrent(table: %u, index: %u): %u\n",
  //        table.asUInt32(),
  //        index.asUInt32(),
  //        ret.asUInt32());
  return ret;
}

__device__ void
extern_memoryDelta(ExecContext& ctx, Val addr, Val cycle, Val dataLow, Val dataHigh, Val count) {
  // printf("memoryDelta\n");
  // ctx.tables.memoryDelta(
  //     addr.asUInt32(), cycle.asUInt32(), dataLow.asUInt32() | (dataHigh.asUInt32() << 16),
  //     count);
}

__device__ uint32_t extern_getDiffCount(ExecContext& ctx, Val cycle) {
  // printf("getDiffCount\n");
  uint32_t cycleU32 = cycle.asUInt32();
  return ctx.preflight.cycles[cycleU32 / 2].diffCount[cycleU32 % 2];
}

__device__ Val extern_isFirstCycle_0(ExecContext& ctx) {
  // printf("isFirstCycle\n");
  return ctx.cycle == 0;
}

__device__ ::cuda::std::array<Val, 4> extern_divide(
    ExecContext& ctx, Val numerLow, Val numerHigh, Val denomLow, Val denomHigh, Val signType) {
  // printf("divide\n");
  uint32_t numer = numerLow.asUInt32() | (numerHigh.asUInt32() << 16);
  uint32_t denom = denomLow.asUInt32() | (denomHigh.asUInt32() << 16);
  auto [quot, rem] = divide_rv32im(numer, denom, signType.asUInt32());
  ::cuda::std::array<Val, 4> ret;
  ret[0] = quot & 0xffff;
  ret[1] = quot >> 16;
  ret[2] = rem & 0xffff;
  ret[3] = rem >> 16;
  return ret;
}

__device__ void extern_print(ExecContext& ctx, Val v) {
  // printf("LOG: %u\n", v.asUInt32());
}

__device__ ::cuda::std::array<Val, 2> extern_getMajorMinor(ExecContext& ctx) {
  uint8_t major = ctx.preflight.cycles[ctx.cycle].major;
  uint8_t minor = ctx.preflight.cycles[ctx.cycle].minor;
  return {major, minor};
}

__device__ Val extern_hostReadPrepare(ExecContext& ctx, Val fp, Val len) {
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx;
  uint32_t word = ctx.preflight.txns[txnIdx].word;
  // printf("[%lu]: hostReadPrepare(txnIdx: %zu, word: 0x%08x)\n", ctx.cycle, txnIdx, word);
  return word;
}

__device__ Val
extern_hostWrite(ExecContext& ctx, Val fdVal, Val addrLow, Val addrHigh, Val lenVal) {
  // printf("hostWrite\n");
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx;
  return ctx.preflight.txns[txnIdx].word;
}

__device__ ::cuda::std::array<Val, 2> extern_nextPagingIdx(ExecContext& ctx) {
  uint32_t pagingIdx = ctx.preflight.cycles[ctx.cycle].pagingIdx;
  uint32_t machineMode = ctx.preflight.cycles[ctx.cycle].machineMode;
  // printf("nextPagingIdx: (0x%05x, %u)\n", pagingIdx, machineMode);
  return {pagingIdx, machineMode};
}

__device__ ::cuda::std::array<Val, 16> extern_bigIntExtern(ExecContext& ctx) {
  ::cuda::std::array<Val, 16> ret;
  size_t bigintIdx = ctx.preflight.cycles[ctx.cycle].bigintIdx;
  for (size_t i = 0; i < 16; i++) {
    ret[i] = ctx.preflight.bigintBytes[bigintIdx + i];
  }
  return ret;
}

__device__ void nextStep(DeviceExecContext* ctx, uint32_t cycle) {
  ExecContext execCtx(*ctx->preflight, *ctx->tables, cycle);
  Buffer dataBuf = *ctx->data;
  Buffer preDataBuf = *ctx->pre_data;
  // When pre_data points to same memory as data, null it out to eliminate
  // the per-load preDataBuf branch (one check per cycle vs thousands per cycle).
  if (preDataBuf.buf == dataBuf.buf) {
    preDataBuf.buf = nullptr;
  }
  MutableBufObj data(dataBuf, preDataBuf);
  Buffer globalBuf = *ctx->global;
  GlobalBufObj global(globalBuf);
  step_Top(execCtx, &data, &global);
}

// The five per-cycle columns the host injector used to scatter for every row (cycle, next pc
// low/high, next state, next machine mode), written from the uploaded preflight cycles. Runs
// after the back scatter and before par_stepExec, so the same values win as before (the host
// injector set these after the back entries of the same row).
__global__ void injectCycleCols(Fp* data, const PreflightCycle* cycles, uint32_t n, uint32_t rows,
                                uint32_t cCycle, uint32_t cPcLow, uint32_t cPcHigh,
                                uint32_t cState, uint32_t cMode) {
  uint32_t row = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= n) {
    return;
  }
  const PreflightCycle& c = cycles[row];
  data[size_t(cCycle) * rows + row] = Fp(row);
  data[size_t(cPcLow) * rows + row] = Fp(c.pc & 0xffff);
  data[size_t(cPcHigh) * rows + row] = Fp(c.pc >> 16);
  data[size_t(cState) * rows + row] = Fp(c.state);
  data[size_t(cMode) * rows + row] = Fp(uint32_t(c.machineMode));
}

#ifdef __HIPCC__
// perm (optional): the launch's cycles in the order to run them (absolute cycle numbers), see
// witgenPerm.
__global__ void par_stepExec(DeviceExecContext* ctx, uint32_t start, uint32_t count,
                             const uint32_t* perm = nullptr) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= count) {
    return;
  }
  nextStep(ctx, perm ? perm[cycle] : start + cycle);
}
#else
__global__ void par_stepExec(DeviceExecContext* ctx, uint32_t start, uint32_t count) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= count) {
    return;
  }
  nextStep(ctx, start + cycle);
}
#endif

__global__ void rev_stepExec(DeviceExecContext* ctx, uint32_t split, uint32_t lastCycle) {
  for (uint32_t cycle = split; cycle-- > 0;) {
    nextStep(ctx, cycle);
  }
  for (uint32_t cycle = lastCycle; cycle-- > split;) {
    nextStep(ctx, cycle);
  }
}

__global__ void fwd_stepExec(DeviceExecContext* ctx, uint32_t count) {
  for (uint32_t cycle = 0; cycle < count; cycle++) {
    nextStep(ctx, cycle);
  }
}

#ifdef __HIPCC__
// CUDA runs stepAccumA (accum_tu.cu) instead: steps.cu defines step_TopAccum for ROCm only.
__global__ void stepAccum(DeviceAccumContext* ctx, uint32_t count, const uint32_t* perm = nullptr) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= count) {
    return;
  }
  // Optional cycle order (cycles grouped by major), so a warp mostly runs one mux arm.
  if (perm) {
    cycle = perm[cycle];
  }

  ExecContext execCtx(*ctx->preflight, *ctx->tables, cycle);
  Buffer dataBuf = *ctx->data;
  MutableBufObj data(dataBuf);
  Buffer accumBuf = *ctx->accum;
  MutableBufObj accum(accumBuf, /*zeroBack=*/kUserAccumSplit);
  Buffer mixBuf = *ctx->mix;
  GlobalBufObj mix(mixBuf);
  Buffer globalBuf = *ctx->global;
  GlobalBufObj global(globalBuf);
  step_TopAccum(execCtx, &accum, &data, &global, &mix);
}
#endif

// keys[i] = (i >> winBits) << 8 | cycles[i].major, vals[i] = i: the input of stepAccum's cycle
// sort. Grouping by major only within 2^winBits-row windows keeps the rows each window's warps
// touch (~14 MB of data/accum columns at 2^14) L2-resident across the per-major passes; a
// global sort by major re-fetched each data sector from DRAM once per major (1.8 GB/segment).
__global__ void majorKeys(const PreflightCycle* cycles, uint32_t* keys, uint32_t* vals, uint32_t n,
                          uint32_t winBits) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    keys[i] = (winBits < 32 ? ((i >> winBits) << 8) : 0u) | cycles[i].major;
    vals[i] = i;
  }
}

#ifdef __HIPCC__
// Cycles [0, n) of the device preflight (cycles) grouped by major within 2^win-row windows
// (stable hipcub radix sort over majorKeys' keys): returns the permutation (device, valid
// until the next call with the same slot on this device). Slot 0: witgen, 1: accum.
static uint32_t* hipMajorPerm(int slot, const PreflightCycle* cycles, uint32_t n, uint32_t win,
                              cudaStream_t stream) {
  static uint32_t* s_keys[16][2][2] = {};
  static uint32_t* s_perm[16][2][2] = {};
  static size_t s_cap[16][2] = {};
  static void* s_sort_temp[16][2] = {};
  static size_t s_sort_temp_bytes[16][2] = {};
  int dev = getCachedDevice();
  if (n > s_cap[dev][slot]) {
    for (int b = 0; b < 2; b++) {
      if (s_keys[dev][slot][b]) CUDA_OK(hipFree(s_keys[dev][slot][b]));
      if (s_perm[dev][slot][b]) CUDA_OK(hipFree(s_perm[dev][slot][b]));
      CUDA_OK(hipMalloc(&s_keys[dev][slot][b], n * sizeof(uint32_t)));
      CUDA_OK(hipMalloc(&s_perm[dev][slot][b], n * sizeof(uint32_t)));
    }
    s_cap[dev][slot] = n;
  }
  auto cfg = getSimpleConfig(n);
  majorKeys<<<cfg.grid, cfg.block, 0, stream>>>(cycles, s_keys[dev][slot][0], s_perm[dev][slot][0], n, win);
  uint32_t maxWin = (win < 32 && n) ? ((n - 1) >> win) : 0;
  int endBit = 8;
  while (maxWin) { endBit++; maxWin >>= 1; }
  hipcub::DoubleBuffer<uint32_t> keys(s_keys[dev][slot][0], s_keys[dev][slot][1]);
  hipcub::DoubleBuffer<uint32_t> vals(s_perm[dev][slot][0], s_perm[dev][slot][1]);
  size_t temp_bytes = 0;
  CUDA_OK(hipcub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys, vals, int(n), 0, endBit, stream));
  if (temp_bytes > s_sort_temp_bytes[dev][slot]) {
    if (s_sort_temp[dev][slot]) CUDA_OK(hipFree(s_sort_temp[dev][slot]));
    CUDA_OK(hipMalloc(&s_sort_temp[dev][slot], temp_bytes));
    s_sort_temp_bytes[dev][slot] = temp_bytes;
  }
  CUDA_OK(hipcub::DeviceRadixSort::SortPairs(
      s_sort_temp[dev][slot], temp_bytes, keys, vals, int(n), 0, endBit, stream));
  return vals.Current();
}

// Whether the current device is RDNA4 (gfx12xx).
static bool sortGfx12() {
  std::lock_guard<std::recursive_mutex> g(risc0_hip_hop::lock());
  return risc0_hip_hop::is_gfx12(getCachedDevice());
}

// par_stepExec's first launch (cycles before the table split) runs its cycles grouped by major
// within 2^RISC0_HIP_WITGEN_SORT-row windows (0 = cycle order), as stepAccum does: a wave then
// mostly runs one arm of step_Top's mux. Same witness: each cycle still runs once and writes only
// its own row; the rows it reads back hold the injected values, which their own cycles write
// again (the same values) in any order, as with the unsorted launch, whose waves run in no fixed
// order either; the lookup-table counts are commutative atomic adds. Default 14 on gfx11 and 12
// on gfx12 (fib 1M succinct, steps.cu's par_stepExec phase 1 per segment: RX 7900 XTX 7.21 ms at
// 12, 6.83-6.85 at 14, 7.03-7.09 at 13 and 15; RX 9070 XT 7.22 at 12, 7.27 at 11, 7.57 at 13, 8.15
// at 14). With witgen_tu2's par_stepExecW2 (hipWitgenTu's default) one less looked better, not yet
// the default: RX 7900 XTX 4.48 ms at 13 vs 4.64 at 14, RX 9070 XT 4.65 at 11 vs 4.80-4.82 at 12.
static uint32_t witgenSortWin() {
  static const int env = [] {
    const char* e = getenv("RISC0_HIP_WITGEN_SORT");
    return e ? atoi(e) : -1;
  }();
  return env >= 0 ? uint32_t(env) : sortGfx12() ? 12u : 14u;
}

// stepAccum's sort window (log2 rows; cycles are grouped by major within 2^win rows, see majorKeys):
// RISC0_HIP_ACCUM_SORT (4..24), else 11. A smaller window keeps each wave's cycles close together
// (row locality of the trace accesses) at the cost of more major divergence (a window holds about one
// partly filled wave per major present). Fib 1M succinct, stepAccumA per segment, windows
// 9/10/11/12/13: RX 7900 XTX 5.45/4.49/3.98/3.97/3.95 ms, RX 9070 XT 6.36/5.07/4.64/4.66/5.02 ms (8:
// 7.13 and 8.79 ms). prove_busy (few majors), 9/10/11/12: RX 7900 XTX 3.10/3.18/3.18/3.21 ms, RX 9070
// XT 3.46 (bimodal per segment, 3.0 or 3.9)/4.00/4.03/4.17 ms.
static uint32_t accumSortWin() {
  static const uint32_t win = [] {
    const char* e = getenv("RISC0_HIP_ACCUM_SORT");
    int w = e ? atoi(e) : 11;
    return uint32_t(w < 4 ? 4 : w > 24 ? 24 : w);
  }();
  return win;
}

// Both major sorts in one launch, in LDS (RISC0_HIP_LSORT, on unless 0): at witgen, wmsSort
// writes par_stepExec's permutation (2^winW-row windows of [0, nW)) and stepAccum's (2^winA-row
// windows of [0, nA)) from one read of the cycles' majors, instead of majorKeys + a hipcub radix
// sort for each (~13 and ~23 launches, ~0.2 and ~0.33 ms per segment at po2 20). The same
// permutations: within each window the rows ordered by (major, row), as the stable radix sort of
// majorKeys' (window, major) keys gives. One 256-thread block per superwindow of 2^S rows,
// S = max(winW, winA) <= kWmsMaxBits: the block stages the superwindow's majors in LDS (bytes,
// padded by one word per 32 words so that chunk reads hit different banks), then ranks each
// window's rows: each thread (large windows: the whole block, one window after another) or lane
// (small windows: one wave per window) owns a contiguous chunk, and per major present a prefix
// sum over the chunks' counts gives each chunk's first slot. The stepAccum permutation is kept for
// the accum of the same trace (wmsAccum); injectCycleCols' columns are written from the same reads.
constexpr uint32_t kWmsMaxBits = 15;
// Windows of at least 2^kWmsBlockBits rows are ranked by the whole block, smaller ones by one wave.
constexpr uint32_t kWmsBlockBits = 12;
// par_stepExec's (major, minor) keys (RISC0_HIP_WITGEN_MINOR) need a second LDS byte array: up to
// 2^14-row superwindows.
constexpr uint32_t kWmsMinorMaxBits = 14;

__device__ __forceinline__ uint32_t wmsPad(uint32_t word) {
  return word + (word >> 5);
}

// LDS words for one byte array of a superwindow of 2^S rows (one pad word per 32 words).
__host__ __device__ inline uint32_t wmsLdsWords(uint32_t S) {
  const uint32_t words = ((1u << S) + 3) / 4;
  return words + words / 32 + 1;
}

// injectCycleCols' five columns, written by wmsSort from the same reads of the cycles (data null:
// not fused).
struct WmsInject {
  Fp* data = nullptr;
  uint32_t rows = 0, cCycle = 0, cPcLow = 0, cPcHigh = 0, cState = 0, cMode = 0;
};

// The keys (bytes) in rows [c0, c1) of the superwindow: bit k & 31 of p[k >> 5] for each key k.
__device__ __forceinline__ void wmsScan(const uint32_t* sKey, uint32_t c0, uint32_t c1,
                                        uint32_t (&p)[8]) {
#pragma unroll
  for (int j = 0; j < 8; j++) p[j] = 0;
  for (uint32_t w = c0 >> 2; w * 4 < c1; w++) {
    const uint32_t word = sKey[wmsPad(w)];
#pragma unroll
    for (uint32_t k = 0; k < 4; k++) {
      const uint32_t i = w * 4 + k, m = (word >> (8 * k)) & 0xff;
      const uint32_t bit = (i >= c0 && i < c1) ? 1u << (m & 31) : 0u;
#pragma unroll
      for (uint32_t j = 0; j < 8; j++) p[j] |= (m >> 5) == j ? bit : 0u;
    }
  }
}

// How many rows of [c0, c1) have key b.
__device__ __forceinline__ uint32_t wmsCount(const uint32_t* sKey, uint32_t c0, uint32_t c1,
                                             uint32_t b) {
  uint32_t c = 0;
  for (uint32_t w = c0 >> 2; w * 4 < c1; w++) {
    const uint32_t word = sKey[wmsPad(w)];
#pragma unroll
    for (uint32_t k = 0; k < 4; k++) {
      const uint32_t i = w * 4 + k;
      c += (i >= c0 && i < c1 && ((word >> (8 * k)) & 0xff) == b) ? 1u : 0u;
    }
  }
  return c;
}

// Writes base + i for each row i of [c0, c1) with key b, in order, from perm[p].
__device__ __forceinline__ void wmsWrite(const uint32_t* sKey, uint32_t c0, uint32_t c1, uint32_t b,
                                         uint32_t base, uint32_t* perm, uint32_t p) {
  for (uint32_t w = c0 >> 2; w * 4 < c1; w++) {
    const uint32_t word = sKey[wmsPad(w)];
#pragma unroll
    for (uint32_t k = 0; k < 4; k++) {
      const uint32_t i = w * 4 + k;
      if (i >= c0 && i < c1 && ((word >> (8 * k)) & 0xff) == b) perm[p++] = base + i;
    }
  }
}

__device__ __forceinline__ uint32_t wmsWaveScan(uint32_t c, uint32_t lane) {
  uint32_t incl = c;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const uint32_t o = __shfl_up(incl, d, 32);
    if (lane >= (uint32_t)d) incl += o;
  }
  return incl;
}

// One wave ranks rows [lo, lo + L) of the superwindow (base: its first row) into perm by (key,
// row): lane l owns a contiguous chunk, and per key present a wave prefix sum over the chunks'
// counts gives each lane's first slot. Wave-uniform control flow.
__device__ void wmsRankWave(const uint32_t* sKey, uint32_t lo, uint32_t L, uint32_t base,
                            uint32_t* perm) {
  const uint32_t lane = threadIdx.x & 31;
  // Chunks of E rows (a multiple of 4, so each starts at an LDS word).
  const uint32_t E = (((L + 31) >> 5) + 3) & ~3u;
  const uint32_t c0 = min(lo + lane * E, lo + L), c1 = min(c0 + E, lo + L);
  uint32_t p[8];
  wmsScan(sKey, c0, c1, p);
#pragma unroll
  for (int j = 0; j < 8; j++)
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) p[j] |= __shfl_xor(p[j], d, 32);
  uint32_t pos0 = base + lo;  // first output slot of the current key
#pragma unroll
  for (int j = 0; j < 8; j++) {
    for (uint32_t bits = p[j]; bits; bits &= bits - 1) {  // wave-uniform
      const uint32_t b = j * 32 + __builtin_ctz(bits);
      const uint32_t c = wmsCount(sKey, c0, c1, b);
      const uint32_t incl = wmsWaveScan(c, lane);
      const uint32_t tot = __shfl(incl, 31, 32);
      if (c) wmsWrite(sKey, c0, c1, b, base, perm, pos0 + incl - c);
      pos0 += tot;
    }
  }
}

// The same with the whole block (thread t owns a chunk; block reductions and prefix sums through
// sRed, which holds at least 8 + blockDim.x / 32 words). Block-uniform control flow.
__device__ void wmsRankBlock(const uint32_t* sKey, uint32_t lo, uint32_t L, uint32_t base,
                             uint32_t* perm, uint32_t* sRed) {
  const uint32_t t = threadIdx.x, lane = t & 31, wave = t >> 5, nWaves = blockDim.x >> 5;
  const uint32_t E = (((L + blockDim.x - 1) / blockDim.x) + 3) & ~3u;
  const uint32_t c0 = min(lo + t * E, lo + L), c1 = min(c0 + E, lo + L);
  uint32_t p[8];
  wmsScan(sKey, c0, c1, p);
#pragma unroll
  for (int j = 0; j < 8; j++)
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) p[j] |= __shfl_xor(p[j], d, 32);
  if (t < 8) sRed[t] = 0;
  __syncthreads();
  if (lane < 8) {
#pragma unroll
    for (int j = 0; j < 8; j++)
      if (lane == (uint32_t)j) atomicOr(&sRed[j], p[j]);
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; j++) p[j] = sRed[j];
  uint32_t pos0 = base + lo;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    for (uint32_t bits = p[j]; bits; bits &= bits - 1) {  // block-uniform
      const uint32_t b = j * 32 + __builtin_ctz(bits);
      const uint32_t c = wmsCount(sKey, c0, c1, b);
      const uint32_t incl = wmsWaveScan(c, lane);
      __syncthreads();  // the previous key's reads of sRed are done
      if (lane == 31) sRed[8 + wave] = incl;
      __syncthreads();
      uint32_t off = 0, tot = 0;
      for (uint32_t w = 0; w < nWaves; w++) {
        const uint32_t v = sRed[8 + w];
        off += w < wave ? v : 0u;
        tot += v;
      }
      if (c) wmsWrite(sKey, c0, c1, b, base, perm, pos0 + off + incl - c);
      pos0 += tot;
    }
  }
  __syncthreads();  // sRed[0..7] is reset by the next call
}

// One 256-thread block per superwindow of 2^S rows. wminor: par_stepExec's keys are
// (min(major, 31), minor & 7) instead of the major (second byte array; S <= kWmsMinorMaxBits).
__global__ void __launch_bounds__(256) wmsSort(const PreflightCycle* __restrict__ cycles,
                                               uint32_t S,
                                               uint32_t* __restrict__ permW, uint32_t nW, uint32_t winW,
                                               uint32_t* __restrict__ permA, uint32_t nA, uint32_t winA,
                                               uint32_t wminor, WmsInject inj) {
  extern __shared__ uint32_t sMaj[];  // wmsLdsWords(S) words per byte array
  __shared__ uint32_t sRed[8 + 32];
  uint32_t* sMin = sMaj + wmsLdsWords(S);
  const uint32_t base = blockIdx.x << S;
  const uint32_t nMax = nW > nA ? nW : nA;
  const uint32_t len = min(1u << S, nMax - base);
  uint8_t* sb = reinterpret_cast<uint8_t*>(sMaj);
  uint8_t* sm = reinterpret_cast<uint8_t*>(sMin);
  for (uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
    const PreflightCycle& c = cycles[base + i];
    const uint32_t at = wmsPad(i >> 2) * 4 + (i & 3);
    sb[at] = c.major;
    if (wminor) sm[at] = uint8_t((min(uint32_t(c.major), 31u) << 3) | (c.minor & 7u));
    if (inj.data) {
      // As injectCycleCols (rows [0, nA): nA = lastCycle).
      const uint32_t row = base + i;
      const size_t rows = inj.rows;
      inj.data[size_t(inj.cCycle) * rows + row] = Fp(row);
      inj.data[size_t(inj.cPcLow) * rows + row] = Fp(c.pc & 0xffff);
      inj.data[size_t(inj.cPcHigh) * rows + row] = Fp(c.pc >> 16);
      inj.data[size_t(inj.cState) * rows + row] = Fp(c.state);
      inj.data[size_t(inj.cMode) * rows + row] = Fp(uint32_t(c.machineMode));
    }
  }
  __syncthreads();
  const uint32_t* sW = wminor ? sMin : sMaj;
  // Large windows: the whole block, one window after another.
  for (int k = 0; k < 2; k++) {
    uint32_t* perm = k ? permA : permW;
    const uint32_t win = k ? winA : winW, n = k ? nA : nW;
    if (!perm || win < kWmsBlockBits) continue;
    for (uint32_t lo = 0; lo < (1u << S) && base + lo < n; lo += 1u << win)
      wmsRankBlock(k ? sMaj : sW, lo, min(1u << win, n - base - lo), base, perm, sRed);
  }
  // Small windows: one wave each.
  const uint32_t wave = threadIdx.x >> 5, nWaves = blockDim.x >> 5;
  const uint32_t jobsW = (permW && winW < kWmsBlockBits) ? (1u << (S - winW)) : 0;
  const uint32_t jobsA = (permA && winA < kWmsBlockBits) ? (1u << (S - winA)) : 0;
  for (uint32_t j = wave; j < jobsW + jobsA; j += nWaves) {
    const bool isW = j < jobsW;
    const uint32_t win = isW ? winW : winA, n = isW ? nW : nA;
    const uint32_t lo = (isW ? j : j - jobsW) << win;  // window start in the superwindow
    if (base + lo >= n) continue;
    wmsRankWave(isW ? sW : sMaj, lo, min(1u << win, n - base - lo), base, isW ? permW : permA);
  }
}

// Launches wmsSort: permW (null: none) gets [0, nW) in 2^winW-row windows, permA [0, nA) in
// 2^winA-row windows; winW, winA <= kWmsMaxBits (winW is ignored when permW is null). wminor:
// permW by (major, minor), when the superwindow is at most 2^kWmsMinorMaxBits rows (else by major).
static inline void wmsLaunch(const PreflightCycle* cycles, uint32_t* permW, uint32_t nW, uint32_t winW,
                             uint32_t* permA, uint32_t nA, uint32_t winA, bool wminor,
                             const WmsInject& inj, hipStream_t stream) {
  const uint32_t S = permW ? (winW > winA ? winW : winA) : winA;
  const uint32_t nMax = permW && nW > nA ? nW : nA;
  const uint32_t blocks = (nMax + (1u << S) - 1) >> S;
  const uint32_t two = (wminor && permW && S <= kWmsMinorMaxBits) ? 1u : 0u;
  wmsSort<<<blocks, 256, wmsLdsWords(S) * 4 * (1 + two), stream>>>(cycles, S, permW, nW, winW, permA,
                                                                   nA, winA, two, inj);
}

// RISC0_HIP_LSORT_CHECK=1: also build each permutation with the hipcub path and compare (a host
// sync per segment; for validation).
__global__ void wmsCompare(const uint32_t* a, const uint32_t* b, uint32_t n, uint32_t* bad) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n && a[i] != b[i]) atomicAdd(bad, 1u);
}

static bool lsortOn() {
  static const bool on = [] {
    const char* e = getenv("RISC0_HIP_LSORT");
    return !(e && e[0] == '0');
  }();
  return on;
}

// RISC0_HIP_WITGEN_MINOR=1: par_stepExec's windows ordered by (major, minor) instead of the major
// (step_Top's arms also branch on the minor), when the superwindow is at most 2^kWmsMinorMaxBits rows.
static bool witgenMinor() {
  static const bool on = [] {
    const char* e = getenv("RISC0_HIP_WITGEN_MINOR");
    return e && e[0] == '1';
  }();
  return on;
}

static bool lsortCheck() {
  static const bool on = [] {
    const char* e = getenv("RISC0_HIP_LSORT_CHECK");
    return e && e[0] == '1';
  }();
  return on;
}

// The accum permutation wmsSort made at witgen, for the accum of the same trace (the device
// preflight `cycles`, `n` cycles, window `win`); consumed once.
struct WmsState {
  uint32_t* perm[2] = {nullptr, nullptr};  // 0: witgen, 1: accum
  size_t cap[2] = {0, 0};
  uint32_t* bad = nullptr;
  const PreflightCycle* accumCycles = nullptr;
  uint32_t accumN = 0, accumWin = 0;
  bool accumValid = false;
};
static WmsState s_wms[16];

static uint32_t wmsCheckCount(const uint32_t* a, const uint32_t* b, uint32_t n, cudaStream_t stream) {
  WmsState& st = s_wms[getCachedDevice()];
  if (!st.bad) CUDA_OK(hipMalloc(&st.bad, sizeof(uint32_t)));
  CUDA_OK(hipMemsetAsync(st.bad, 0, sizeof(uint32_t), stream));
  auto cfg = getSimpleConfig(n);
  wmsCompare<<<cfg.grid, cfg.block, 0, stream>>>(a, b, n, st.bad);
  uint32_t h = 0;
  CUDA_OK(hipMemcpyAsync(&h, st.bad, sizeof(uint32_t), hipMemcpyDeviceToHost, stream));
  CUDA_OK(hipStreamSynchronize(stream));
  return h;
}

// Witgen: both permutations in one launch when RISC0_HIP_LSORT is on and both windows are at most
// 2^kWmsMaxBits rows; returns false otherwise (the caller sorts as before). *wperm: par_stepExec's
// permutation (nullptr when winW is 0 or nW <= 1).
static bool wmsEligible(uint32_t winW, uint32_t nA) {
  return lsortOn() && winW <= kWmsMaxBits && accumSortWin() <= kWmsMaxBits && nA > 0;
}

// inj: injectCycleCols' columns to write too (data null: none; the caller injects them itself).
static bool wmsWitgen(const PreflightCycle* cycles, uint32_t nW, uint32_t winW, uint32_t nA,
                      cudaStream_t stream, const uint32_t** wperm, const WmsInject& inj) {
  const uint32_t winA = accumSortWin();
  if (!wmsEligible(winW, nA)) return false;
  WmsState& st = s_wms[getCachedDevice()];
  st.accumValid = false;
  const bool doW = winW && nW > 1;
  const size_t need[2] = {doW ? nW : 0u, nA};
  for (int k = 0; k < 2; k++) {
    if (need[k] > st.cap[k]) {
      if (st.perm[k]) CUDA_OK(hipFree(st.perm[k]));
      st.perm[k] = nullptr;
      st.cap[k] = 0;
      CUDA_OK(hipMalloc(&st.perm[k], need[k] * sizeof(uint32_t)));
      st.cap[k] = need[k];
    }
  }
  wmsLaunch(cycles, doW ? st.perm[0] : nullptr, nW, winW, st.perm[1], nA, winA, witgenMinor(), inj,
            stream);
  CUDA_OK(cudaGetLastError());
  *wperm = doW ? st.perm[0] : nullptr;
  if (lsortCheck() && doW && !witgenMinor()) {
    const uint32_t* ref = hipMajorPerm(0, cycles, nW, winW, stream);
    uint32_t bad = wmsCheckCount(st.perm[0], ref, nW, stream);
    fprintf(stderr, "[lsort-check] witgen n=%u win=%u mismatches=%u\n", nW, winW, bad);
  }
  st.accumCycles = cycles;
  st.accumN = nA;
  st.accumWin = winA;
  st.accumValid = true;
  return true;
}

// Accum: wmsSort's permutation for this trace, if the witgen made one (consumed), else nullptr.
static uint32_t* wmsAccum(const PreflightCycle* cycles, uint32_t n) {
  WmsState& st = s_wms[getCachedDevice()];
  if (!st.accumValid || st.accumCycles != cycles || st.accumN != n || st.accumWin != accumSortWin())
    return nullptr;
  st.accumValid = false;
  return st.perm[1];
}
#endif

// finalizeAccum fused with eltwise_zeroize_fp(accum) (INVALID -> 0), one pass over all rows.
// Same result as finalizeAccum followed by the zeroize: machine cells are finalized (an unset
// one counting as 0), every other cell is only zeroized. prev[] (last 4 columns, row back1) is
// read raw while row back1's thread may zeroize it concurrently; those cells are inclusive-scan
// outputs and never INVALID, so the zeroize cannot change them.
__global__ void finalizeZeroizeAccum(DeviceAccumContext* ctx, uint32_t lastCycle) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (!ctx) return;
  Buffer& accum = *ctx->accum;
  const size_t rows = accum.rows, cols = accum.cols;
  if (cycle >= rows) return;
  Fp* buf = accum.buf;
  auto z = [&](size_t i) { if (buf[i].asRaw() == Fp::INVALID) buf[i] = Fp(); };
  if (cycle < lastCycle) {
    size_t machineColumns = (cols - kUserAccumSplit) / 4;
    size_t back1 = (cycle + lastCycle - 1) % lastCycle;
    Fp prev[4];
    for (int k = 0; k < 4; k++) prev[k] = buf[(cols - 4 + k) * rows + back1];
    for (size_t j = 0; j < machineColumns - 1; j++) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        size_t i = (kUserAccumSplit + j * 4 + k) * rows + cycle;
        // An unset (INVALID) cell counts as 0: INVALID + prev[k] is unreduced when prev[k] is
        // 0, and the NTT mishandles it (see phase 3 of risc0_circuit_rv32im_cpu_accum).
        Fp cur = buf[i].asRaw() == Fp::INVALID ? Fp() : buf[i];
        buf[i] = cur + prev[k];
      }
    }
    for (size_t c = 0; c < kUserAccumSplit; c++) z(c * rows + cycle);
    for (size_t c = kUserAccumSplit + (machineColumns - 1) * 4; c < cols; c++) z(c * rows + cycle);
  } else {
    for (size_t c = 0; c < cols; c++) z(c * rows + cycle);
  }
}

static bool envOn(const char* name) {
  const char* e = getenv(name);
  return !(e && e[0] == '0');
}

// ROCm: stepAccum from accum_tu.cu (as on CUDA, plus non-virtual buffer objects and accesses
// without compiler barriers; see witgen.h) unless RISC0_HIP_ACCUM_TU=0, which runs steps.cu's
// stepAccum.
#ifdef __HIPCC__
static bool accumTu() {
  static const bool on = envOn("RISC0_HIP_ACCUM_TU");
  return on;
}
#endif

#ifdef __HIPCC__
// ROCm: par_stepExec's first and second launches run witgen_tu.cu's copy of step_Top's call graph
// (build.rs: static functions in their own TU, global rather than FLAT buffer accesses with the
// same per-access compiler barriers, buffers.h). Default (RISC0_HIP_WITGEN_TU=2, witgen_tu2's
// par_stepExecW2): force-inlined with launch bounds (256, 1): 256 VGPRs, 16 KB of LDS (no
// __shared__ in the source: promoted private arrays) and 3.3 KB instead of 11.6 KB of scratch per
// lane. Fib 1M succinct, phase 1 per segment: RX 7900 XTX 7.42 -> 4.62 ms, RX 9070 XT 7.24 -> 4.82
// ms (phase 2 173 -> 78 and 137 -> 61 us). RISC0_HIP_WITGEN_TU=1 (witgen_tu's par_stepExecW):
// inlined at the compiler's discretion with (256, 8) (the same 192-VGPR cap as par_stepExec): 6.02
// / 6.07 ms (117 / 103 us). 0: steps.cu's par_stepExec. Measured and not used: the noinline call
// graph (as on CUDA) 9.31 / 8.20 ms, discretionary inlining with (256, 1) 7.37 / 6.44 ms.
static int hipWitgenTu() {
  static const int v = [] {
    const char* e = getenv("RISC0_HIP_WITGEN_TU");
    int x = e ? atoi(e) : 2;
    return (x >= 0 && x <= 2) ? x : 2;
  }();
  return v;
}

static void hipLaunchWitgenTu(int v, unsigned grid, unsigned block, cudaStream_t stream, void* ctx,
                              uint32_t start, uint32_t count, const uint32_t* perm) {
  if (v == 2)
    risc0_rv32im_launch_par_stepExecTU2(grid, block, stream, ctx, start, count, perm);
  else
    risc0_rv32im_launch_par_stepExecTU(grid, block, stream, ctx, start, count, perm);
  CUDA_OK(cudaGetLastError());
}
#endif

// witgen_tu.cu (par_stepExec's call graph in a standalone noinline TU): on unless
// RISC0_CUDA_WITGEN_TU=0.
static bool witgenTu() {
#ifdef __HIPCC__
  return false;
#else
  static const bool on = envOn("RISC0_CUDA_WITGEN_TU");
  return on;
#endif
}

__global__ void finalizeAccum(DeviceAccumContext* ctx, uint32_t lastCycle) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= lastCycle) {
    return;
  }

  Buffer& accum = *ctx->accum;

  size_t machineColumns = (accum.cols - kUserAccumSplit) / 4;
  size_t back1 = (cycle + lastCycle - 1) % lastCycle;
  Fp prev[4];
  for (size_t k = 0; k < 4; k++) {
    prev[k] = accum.get(back1, accum.cols - 4 + k);
  }
  for (size_t j = 0; j < machineColumns - 1; j++) {
    for (size_t k = 0; k < 4; k++) {
      size_t col = kUserAccumSplit + j * 4 + k;
      // An unset (INVALID) cell counts as 0: INVALID + prev[k] is unreduced when prev[k] is 0,
      // and the NTT mishandles it (see phase 3 of risc0_circuit_rv32im_cpu_accum).
      Fp cur = accum.get(cycle, col);
      if (cur.asRaw() == Fp::INVALID) {
        cur = Fp();
      }
      accum.set(cycle, col, cur + prev[k]);
    }
  }
}

} // namespace risc0::circuit::rv32im_v2::cuda

constexpr size_t kStepModeParallel = 0;
constexpr size_t kStepModeSeqForward = 1;
constexpr size_t kStepModeSeqReverse = 2;

extern "C" {

using namespace risc0::circuit::rv32im_v2::cuda;

const char* risc0_circuit_rv32im_cuda_witgen_v2(uint32_t mode,
                                                ExecBuffers* buffers,
                                                PreflightTrace* preflight,
                                                uint32_t lastCycle,
                                                const uint8_t* staged,
                                                const uint32_t* cycle_cols,
                                                uint64_t staged_id);

const char* risc0_circuit_rv32im_cuda_witgen(uint32_t mode,
                                             ExecBuffers* buffers,
                                             PreflightTrace* preflight,
                                             uint32_t lastCycle) {
  return risc0_circuit_rv32im_cuda_witgen_v2(mode, buffers, preflight, lastCycle, nullptr, nullptr, 0);
}

// Copy a staged trace (`bytes` at pinned `staged`, identified by `id`) to its device slot on a
// side stream, overlapping the GPU work in flight; the witgen that consumes it waits for the
// copy. Called on the proving thread (it holds the GPU lock).
//
// `out_ev` (nullable): if a copy was enqueued, receives a new event recorded after it. The
// owner of the pinned buffer must cudaEventSynchronize it (risc0_circuit_rv32im_cuda_event_wait)
// before the buffer is reused or freed: the DMA reads the pinned memory asynchronously.
const char* risc0_circuit_rv32im_cuda_prefetch(const uint8_t* staged, size_t bytes, uint64_t id,
                                               void** out_ev) {
  cudaStream_t cs = nullptr;
  try {
    if (out_ev) *out_ev = nullptr;
    if (!stagingDeviceOk()) return nullptr;
    std::lock_guard<std::recursive_mutex> lk(g_slots_mu);
    DevSlot* sl = getDevSlot(staged);
    if (sl->ready && sl->id == id) return nullptr;
    if (sl->in_use || sl == get_cache().a_slot) return nullptr;  // the current segment's
    growDevSlot(sl, bytes);
    cs = getCopyStream();
    // Created before the copy is queued: once it is, the caller must get an event to wait for.
    cudaEvent_t ev = nullptr;
    if (out_ev) CUDA_OK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    if (sl->released) CUDA_OK(cudaStreamWaitEvent(cs, sl->releasedEv, 0));
    sl->ready = false;
    CUDA_OK(cudaMemcpyAsync(sl->dev, staged, bytes, cudaMemcpyHostToDevice, cs));
    if (out_ev) {
      *out_ev = (void*)ev;
      CUDA_OK(cudaEventRecord(ev, cs));
    }
    CUDA_OK(cudaEventRecord(sl->readyEv, cs));
    sl->id = id;
    sl->ready = true;
  } catch (const std::exception& err) {
    // A copy from `staged` may be queued, and the caller may recycle it once this returns.
    if (cs) (void)cudaStreamSynchronize(cs);
    if (out_ev && *out_ev) {
      (void)cudaEventDestroy((cudaEvent_t)*out_ev);
      *out_ev = nullptr;
    }
    return strdup(err.what());
  } catch (...) {
    if (cs) (void)cudaStreamSynchronize(cs);
    if (out_ev && *out_ev) {
      (void)cudaEventDestroy((cudaEvent_t)*out_ev);
      *out_ev = nullptr;
    }
    return strdup("Generic exception");
  }
  return nullptr;
}

// Free the device buffers of the staged-trace slots that the current segment is not using
// (they are reallocated on demand). Called through the HAL's release hooks: by
// clear_buffer_pool, and when a raw device allocation fails. cudaFree waits for the device, so
// no copy into a freed slot is still in flight.
const char* risc0_circuit_rv32im_cuda_release_slots() {
  try {
    std::lock_guard<std::recursive_mutex> lk(g_slots_mu);
    for (auto& sl : s_slots[getCachedDevice()]) {
      if (sl.in_use || !sl.dev) continue;
      CUDA_OK(cudaFree(sl.dev));
      sl.dev = nullptr;
      sl.cap = 0;
      sl.ready = false;
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

// 1 once staging was turned off because a thread proved with another device current (see
// stagingDeviceOk); the Rust side then stops staging traces.
int risc0_circuit_rv32im_cuda_staging_disabled() {
  return g_staging_off.load() ? 1 : 0;
}

// Wait for and destroy an event returned by risc0_circuit_rv32im_cuda_prefetch. Thread-safe
// (called from whichever thread drops the staged trace).
const char* risc0_circuit_rv32im_cuda_event_wait(void* ev) {
  try {
    if (ev) {
      CUDA_OK(rv32imEventSynchronize((cudaEvent_t)ev));
      CUDA_OK(cudaEventDestroy((cudaEvent_t)ev));
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

// Pinned host memory for the preflight staging slots (filled by the preflight thread), backed
// by huge pages (ThpPinned): cudaHostAlloc of the ~250 MB slots cost 50-85 ms cold while
// holding the driver lock.
static std::mutex g_host_slots_mu;
static std::unordered_map<void*, ThpPinned>& hostSlots() {
  static std::unordered_map<void*, ThpPinned> m;
  return m;
}

const char* risc0_circuit_rv32im_cuda_host_alloc(size_t bytes, void** out) {
  try {
    ThpPinned p;
    p.alloc(bytes);
    *out = p.ptr;
    std::lock_guard<std::mutex> lk(g_host_slots_mu);
    hostSlots()[p.ptr] = p;
  } catch (const std::exception& err) {
    *out = nullptr;
    return strdup(err.what());
  } catch (...) {
    *out = nullptr;
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_circuit_rv32im_cuda_host_free(void* ptr) {
  try {
    ThpPinned p;
    {
      std::lock_guard<std::mutex> lk(g_host_slots_mu);
      auto it = hostSlots().find(ptr);
      if (it == hostSlots().end()) throw std::runtime_error("host_free: unknown pointer");
      p = it->second;
      hostSlots().erase(it);
    }
    {
      // Its device slot (if any) loses its owner: the next staged trace without one reuses the
      // device buffer, and risc0_circuit_rv32im_cuda_release_slots frees it.
      std::lock_guard<std::recursive_mutex> lk(g_slots_mu);
      for (auto& sl : s_slots[getCachedDevice()]) {
        if (sl.pinned == ptr && !sl.in_use) {
          sl.pinned = nullptr;
          sl.ready = false;
        }
      }
    }
    p.free_pinned();  // see ThpPinned::free_pinned
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

// Error path of witgen: copies from the caller's pinned staging buffer (`staged`) or from pf_pin
// may already be queued. The caller may recycle or free `staged` as soon as the error returns, so
// wait for the queued work first (errors ignored: the original error is reported).
static void witgenErrorSync() noexcept {
  try {
    (void)cudaStreamSynchronize(getPersistentStream());
  } catch (...) {
  }
  (void)cudaGetLastError();
}

// `staged` (nullable): pinned cycles|txns|bigintBytes, see setup_exec. `cycle_cols` (nullable):
// the five data columns of injectCycleCols, which the host injector then left out.
const char* risc0_circuit_rv32im_cuda_witgen_v2(uint32_t mode,
                                                ExecBuffers* buffers,
                                                PreflightTrace* preflight,
                                                uint32_t lastCycle,
                                                const uint8_t* staged,
                                                const uint32_t* cycle_cols,
                                                uint64_t staged_id) {
  try {
    cudaStream_t stream = getPersistentStream();
    auto t0 = std::chrono::steady_clock::now();
    DeviceExecContext* d_ctx = get_cache().setup_exec(
        buffers, preflight, lastCycle, stream, reinterpret_cast<const char*>(staged), staged_id);
    auto t1 = std::chrono::steady_clock::now();
    size_t split = preflight->tableSplitCycle;
#ifdef __HIPCC__
    // RISC0_HIP_LSORT: wmsSort (parallel mode) writes these columns from its read of the cycles.
    WmsInject inj;
    const bool fuseInject =
        cycle_cols && mode == kStepModeParallel && wmsEligible(witgenSortWin(), lastCycle);
    if (fuseInject) {
      inj.data = buffers->data.buf;
      inj.rows = buffers->data.rows;
      inj.cCycle = cycle_cols[0];
      inj.cPcLow = cycle_cols[1];
      inj.cPcHigh = cycle_cols[2];
      inj.cState = cycle_cols[3];
      inj.cMode = cycle_cols[4];
    }
    if (cycle_cols && !fuseInject) {
#else
    if (cycle_cols) {
#endif
      auto cfg = getSimpleConfig(lastCycle);
      injectCycleCols<<<cfg.grid, cfg.block, 0, stream>>>(
          buffers->data.buf, get_cache().a_cycles, lastCycle, buffers->data.rows,
          cycle_cols[0], cycle_cols[1], cycle_cols[2], cycle_cols[3], cycle_cols[4]);
      CUDA_OK(cudaGetLastError());
    }

#ifdef __HIPCC__
    s_wms[getCachedDevice()].accumValid = false;  // set again below by wmsWitgen
#endif
    switch (mode) {
    case kStepModeParallel: {
      size_t phase2Count = lastCycle - split;
      {
        nvtx3::scoped_range range("par_stepExec");
        auto cfg1 = getSimpleConfig(split);
        auto cfg2 = getSimpleConfig(phase2Count);
#ifndef __HIPCC__
        // witgen_tu.cu: step_Top's call graph as static (noinline) functions in a standalone TU,
        // power-of-two rows only. Kept noinline on purpose: the fully inlined build reorders the
        // back-row loads of rows written by neighbouring threads (latent race) and is not bit-exact.
        auto pow2 = [](size_t r) { return r && (r & (r - 1)) == 0; };
        if (witgenTu() && pow2(buffers->data.rows) &&
            pow2(buffers->pre_data.rows ? buffers->pre_data.rows : buffers->data.rows)) {
          risc0_rv32im_launch_par_stepExecTU(cfg1.grid.x, cfg1.block.x, stream, d_ctx, 0, split);
          risc0_rv32im_launch_par_stepExecTU(cfg2.grid.x, cfg2.block.x, stream, d_ctx, split, phase2Count);
        } else
#endif
        {
#ifdef __HIPCC__
          const uint32_t win = witgenSortWin();
          const uint32_t* wperm = nullptr;
          if (!wmsWitgen(get_cache().a_cycles, split, win, lastCycle, stream, &wperm, inj)) {
            if (inj.data) {  // not expected: fuseInject implies wmsWitgen runs
              auto cfg = getSimpleConfig(lastCycle);
              injectCycleCols<<<cfg.grid, cfg.block, 0, stream>>>(
                  inj.data, get_cache().a_cycles, lastCycle, inj.rows, inj.cCycle, inj.cPcLow,
                  inj.cPcHigh, inj.cState, inj.cMode);
            }
            wperm = (win && split > 1) ? hipMajorPerm(0, get_cache().a_cycles, split, win, stream)
                                       : nullptr;
          }
          // witgen_tu.cu's par_stepExecW2 unless RISC0_HIP_WITGEN_TU=0/1 (see hipWitgenTu).
          const int wtu = hipWitgenTu();
          auto isPow2 = [](size_t r) { return r && (r & (r - 1)) == 0; };
          const bool pow2rows = isPow2(buffers->data.rows) &&
              isPow2(buffers->pre_data.rows ? buffers->pre_data.rows : buffers->data.rows);
          // Each phase on a side hardware queue (hip_hop.h) when large: they run ~2-5 ms each.
          {
            risc0_hip_hop::Scope s(stream, risc0_hip_hop::want(risc0_hip_hop::kRvWitgen, split, 18));
            if (wtu && pow2rows)
              hipLaunchWitgenTu(wtu, cfg1.grid.x, cfg1.block.x, s, d_ctx, 0, split, wperm);
            else
              par_stepExec<<<cfg1.grid, cfg1.block, 0, s>>>(d_ctx, 0, split, wperm);
          }
          {
            risc0_hip_hop::Scope s(stream, risc0_hip_hop::want(risc0_hip_hop::kRvWitgen, phase2Count, 18));
            if (wtu && pow2rows)
              hipLaunchWitgenTu(wtu, cfg2.grid.x, cfg2.block.x, s, d_ctx, split, phase2Count, nullptr);
            else
              par_stepExec<<<cfg2.grid, cfg2.block, 0, s>>>(d_ctx, split, phase2Count);
          }
#else
          par_stepExec<<<cfg1.grid, cfg1.block, 0, stream>>>(d_ctx, 0, split);
          par_stepExec<<<cfg2.grid, cfg2.block, 0, stream>>>(d_ctx, split, phase2Count);
#endif
        }
      }
    } break;
    case kStepModeSeqForward:
    case kStepModeSeqReverse: {
      // Sequential modes don't need pre_data (no cross-thread race).
      // Null it out so MutableBufObj falls through to the mutable data buffer.
      Buffer nullBuf = {nullptr, 0, 0, false};
      CUDA_OK(cudaMemcpyAsync(get_cache().d_exec_pre_data, &nullBuf, sizeof(Buffer),
                               cudaMemcpyHostToDevice, stream));
      if (mode == kStepModeSeqForward) {
        fwd_stepExec<<<1, 1, 0, stream>>>(d_ctx, lastCycle);
      } else {
        rev_stepExec<<<1, 1, 0, stream>>>(d_ctx, split, lastCycle);
      }
      CUDA_OK(cudaStreamSynchronize(stream));
    } break;
    }
    auto t2 = std::chrono::steady_clock::now();
    if (g_verbose) fprintf(stderr, "      [ffi_witgen] ctx_setup: %.1fms, kernels(async): %.1fms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::milli>(t2 - t1).count());
  } catch (const std::exception& err) {
    witgenErrorSync();
    return strdup(err.what());
  } catch (...) {
    witgenErrorSync();
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_circuit_rv32im_cuda_accum(AccumBuffers* buffers,
                                            PreflightTrace* preflight,
                                            uint32_t lastCycle) {
  try {
    cudaStream_t stream = getPersistentStream();

    auto t0 = std::chrono::steady_clock::now();
    DeviceAccumContext* d_ctx = get_cache().setup_accum(buffers, preflight, lastCycle, stream);
    auto t1 = std::chrono::steady_clock::now();
    auto cfg = getSimpleConfig(lastCycle);

    {
      nvtx3::scoped_range range("stepAccum");
      // Run stepAccum's cycles grouped by major (stable sort), so warps diverge less. Same
      // output: each cycle still runs once and writes only its own row, and its only
      // back-reads (zeroBack) don't depend on other cycles. The buffers are grow-only and
      // per device; callers hold the process-wide GPU lock (risc0_zkp::hal::cuda::exclusive).
      uint32_t* perm = nullptr;
#ifndef __HIPCC__
      static uint32_t* s_keys[16] = {};
      static uint32_t* s_perm[16] = {};
      static uint32_t* s_keys2[16] = {};
      // Sort window (rows, log2): cycles are grouped by major within 2^14-row windows.
      const uint32_t s_win = 14;
      static uint32_t* s_perm2[16] = {};
      static size_t s_cap[16] = {};
      {
        int dev = getCachedDevice();
        if (lastCycle > s_cap[dev]) {
          if (s_keys[dev]) CUDA_OK(cudaFree(s_keys[dev]));
          if (s_perm[dev]) CUDA_OK(cudaFree(s_perm[dev]));
          if (s_keys2[dev]) CUDA_OK(cudaFree(s_keys2[dev]));
          if (s_perm2[dev]) CUDA_OK(cudaFree(s_perm2[dev]));
          s_keys[dev] = s_perm[dev] = s_keys2[dev] = s_perm2[dev] = nullptr;
          s_cap[dev] = 0;
          CUDA_OK(risc0_cuda_malloc(&s_keys[dev], lastCycle * sizeof(uint32_t)));
          CUDA_OK(risc0_cuda_malloc(&s_perm[dev], lastCycle * sizeof(uint32_t)));
          CUDA_OK(risc0_cuda_malloc(&s_keys2[dev], lastCycle * sizeof(uint32_t)));
          CUDA_OK(risc0_cuda_malloc(&s_perm2[dev], lastCycle * sizeof(uint32_t)));
          s_cap[dev] = lastCycle;
        }
        const uint32_t win = s_win ? s_win : 32;
        majorKeys<<<cfg.grid, cfg.block, 0, stream>>>(get_cache().a_cycles, s_keys[dev], s_perm[dev], lastCycle, win);
        // Stable LSD radix sort on the (window, major) keys, only over the key bits in use:
        // the same permutation as thrust::stable_sort_by_key on the full keys.
        uint32_t maxWin = (win < 32 && lastCycle) ? ((lastCycle - 1) >> win) : 0;
        int endBit = 8;
        while (maxWin) { endBit++; maxWin >>= 1; }
        cub::DoubleBuffer<uint32_t> dk(s_keys[dev], s_keys2[dev]);
        cub::DoubleBuffer<uint32_t> dv(s_perm[dev], s_perm2[dev]);
        size_t tb = 0;
        CUDA_OK(cub::DeviceRadixSort::SortPairs(nullptr, tb, dk, dv, (int)lastCycle, 0, endBit, stream));
        CUDA_OK(cub::DeviceRadixSort::SortPairs(cubTemp(tb), tb, dk, dv, (int)lastCycle, 0, endBit, stream));
        perm = dv.Current();
      }
#else
      // The same order with hipcub's radix sort (stable) over the same (window, major) keys.
      // Measured on an RX 9070 XT, unsorted stepAccum was ~4x slower per cycle than on an
      // Arc Pro B70.
      static uint32_t* s_keys[16][2] = {};
      static uint32_t* s_perm[16][2] = {};
      static size_t s_cap[16] = {};
      static void* s_sort_temp[16] = {};
      static size_t s_sort_temp_bytes[16] = {};
      // The permutation wmsSort already made at witgen (RISC0_HIP_LSORT), if any.
      uint32_t* fused = wmsAccum(get_cache().a_cycles, lastCycle);
      if (fused && !lsortCheck()) {
        perm = fused;
      } else {
        int dev = getCachedDevice();
        if (lastCycle > s_cap[dev]) {
          for (int b = 0; b < 2; b++) {
            if (s_keys[dev][b]) CUDA_OK(hipFree(s_keys[dev][b]));
            if (s_perm[dev][b]) CUDA_OK(hipFree(s_perm[dev][b]));
            CUDA_OK(hipMalloc(&s_keys[dev][b], lastCycle * sizeof(uint32_t)));
            CUDA_OK(hipMalloc(&s_perm[dev][b], lastCycle * sizeof(uint32_t)));
          }
          s_cap[dev] = lastCycle;
        }
        // Sort window (rows, log2): cycles are grouped by major within 2^win rows (accumSortWin).
        const uint32_t win = accumSortWin();
        majorKeys<<<cfg.grid, cfg.block, 0, stream>>>(
            get_cache().a_cycles, s_keys[dev][0], s_perm[dev][0], lastCycle, win);
        uint32_t maxWin = lastCycle ? ((lastCycle - 1) >> win) : 0;
        int endBit = 8;
        while (maxWin) { endBit++; maxWin >>= 1; }
        hipcub::DoubleBuffer<uint32_t> keys(s_keys[dev][0], s_keys[dev][1]);
        hipcub::DoubleBuffer<uint32_t> vals(s_perm[dev][0], s_perm[dev][1]);
        size_t temp_bytes = 0;
        CUDA_OK(hipcub::DeviceRadixSort::SortPairs(
            nullptr, temp_bytes, keys, vals, int(lastCycle), 0, endBit, stream));
        if (temp_bytes > s_sort_temp_bytes[dev]) {
          if (s_sort_temp[dev]) CUDA_OK(hipFree(s_sort_temp[dev]));
          CUDA_OK(hipMalloc(&s_sort_temp[dev], temp_bytes));
          s_sort_temp_bytes[dev] = temp_bytes;
        }
        CUDA_OK(hipcub::DeviceRadixSort::SortPairs(
            s_sort_temp[dev], temp_bytes, keys, vals, int(lastCycle), 0, endBit, stream));
        perm = vals.Current();
        if (fused) {
          uint32_t bad = wmsCheckCount(fused, perm, lastCycle, stream);
          fprintf(stderr, "[lsort-check] accum n=%u win=%u mismatches=%u\n", lastCycle, win, bad);
          perm = fused;
        }
      }
#endif
      // accum_tu.cu: the stepAccum call graph force-inlined in a whole-program TU (no ABI
      // calls or stack-resident buffer objects), the only accum step CUDA builds carry (see
      // accum_tu_gen.py). It needs power-of-two rows (RISC0_ROWS_POW2): rows = 2^po2.
      auto pow2 = [](size_t r) { return r && (r & (r - 1)) == 0; };
#ifdef __HIPCC__
      {
      // On a side hardware queue (hip_hop.h) when large (~4 ms at 2^20 cycles).
      risc0_hip_hop::Scope s(stream, risc0_hip_hop::want(risc0_hip_hop::kRvAccum, lastCycle, 18));
      if (!accumTu() || !pow2(buffers->data.rows) || !pow2(buffers->accum.rows)) {
        stepAccum<<<cfg.grid, cfg.block, 0, s>>>(d_ctx, lastCycle, perm);
      } else
        risc0_rv32im_launch_stepAccumTU(cfg.grid.x, cfg.block.x, s, d_ctx, lastCycle, perm);
      }
#else
      if (!pow2(buffers->data.rows) || !pow2(buffers->accum.rows)) {
        throw std::runtime_error("rv32im accum: rows must be a power of two");
      }
      risc0_rv32im_launch_stepAccumTU(cfg.grid.x, cfg.block.x, stream, d_ctx, lastCycle, perm);
#endif

      size_t rows = buffers->accum.rows;
#ifdef __HIPCC__
      // Cached temp storage for hipcub scans (avoids hipMallocAsync per segment)
      {
        size_t col0 = buffers->accum.cols - 4;
        Fp* itFirst = buffers->accum.buf + col0 * rows;
        size_t n = lastCycle;
        size_t temp_bytes = 0;
        hipcub::DeviceScan::InclusiveScan(nullptr, temp_bytes, itFirst, itFirst, AddOp(), n, stream);

        int scan_dev = getCachedDevice();
        static void* s_scan_temp[16] = {};
        static size_t s_scan_temp_bytes[16] = {};
        if (temp_bytes > s_scan_temp_bytes[scan_dev]) {
          if (s_scan_temp[scan_dev]) CUDA_OK(hipFree(s_scan_temp[scan_dev]));
          CUDA_OK(hipMalloc(&s_scan_temp[scan_dev], temp_bytes));
          s_scan_temp_bytes[scan_dev] = temp_bytes;
        }
        for (size_t j = 0; j < 4; j++) {
          Fp* itBegin = buffers->accum.buf + (col0 + j) * rows;
          hipcub::DeviceScan::InclusiveScan(s_scan_temp[scan_dev], temp_bytes, itBegin, itBegin, AddOp(), n, stream);
        }
      }
#else
      {
        size_t tb = 0;
        Fp* col0 = buffers->accum.buf + (buffers->accum.cols - 4) * rows;
        CUDA_OK(cub::DeviceScan::InclusiveScan(nullptr, tb, col0, col0, FpAddOp(), (int)lastCycle, stream));
        void* temp = cubTemp(tb);
        for (size_t j = 0; j < 4; j++) {
          size_t col = buffers->accum.cols - 4 + j;
          Fp* itBegin = buffers->accum.buf + col * rows;
          // Fp addition is associative and exact, so any scan order gives the same sums.
          CUDA_OK(cub::DeviceScan::InclusiveScan(temp, tb, itBegin, itBegin, FpAddOp(), (int)lastCycle, stream));
        }
      }
#endif

      // finalizeAccum fused with the INVALID -> 0 zeroize of the accum buffer.
      auto cfgz = getSimpleConfig(rows);
#ifdef __HIPCC__
      {
        // ~1.1 ms at 2^20 rows: on a side hardware queue (hip_hop.h); the event below is
        // recorded on `stream` after the scope, so it still follows the kernel.
        risc0_hip_hop::Scope s(stream, risc0_hip_hop::want(risc0_hip_hop::kRvAccum, rows, 20));
        finalizeZeroizeAccum<<<cfgz.grid, cfgz.block, 0, s>>>(d_ctx, lastCycle);
      }
#else
      finalizeZeroizeAccum<<<cfgz.grid, cfgz.block, 0, stream>>>(d_ctx, lastCycle);
#endif
      // Last reader of this segment's device preflight: a prefetch into its slot waits for this.
      if (DevSlot* sl = get_cache().a_slot) {
        std::lock_guard<std::recursive_mutex> lk(g_slots_mu);
        CUDA_OK(cudaEventRecord(sl->releasedEv, stream));
        sl->released = true;
        sl->in_use = false;
      }
    }
    auto t2 = std::chrono::steady_clock::now();
    if (g_verbose) fprintf(stderr, "      [ffi_accum] ctx_setup: %.1fms, kernels: %.1fms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::milli>(t2 - t1).count());

  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

// 1 when risc0_circuit_rv32im_cuda_accum also zeroizes accum (finalizeZeroizeAccum), so the
// caller skips its separate eltwise_zeroize of the accum buffer.
int risc0_circuit_rv32im_cuda_accum_zeroizes() {
  return 1;
}

const char* risc0_circuit_rv32im_cuda_warmup() {
  try {
    cudaStream_t stream = getPersistentStream();
    // Launch major kernels with count=0 to trigger full binary loading.
    // With count=0: "if (cycle >= count) return;" exits immediately, no memory access.
    // This avoids ~5-10ms of first-launch stalls during the actual proof.
    par_stepExec<<<1, 1, 0, stream>>>(nullptr, 0, 0);
#ifdef __HIPCC__
    stepAccum<<<1, 1, 0, stream>>>(nullptr, 0);
    risc0_rv32im_launch_stepAccumTU(1, 1, stream, nullptr, 0, nullptr);
    finalizeZeroizeAccum<<<1, 1, 0, stream>>>(nullptr, 0);
    if (int v = hipWitgenTu()) hipLaunchWitgenTu(v, 1, 1, stream, nullptr, 0, 0, nullptr);
#endif
    injectCycleCols<<<1, 1, 0, stream>>>(nullptr, nullptr, 0, 0, 0, 0, 0, 0, 0);
    majorKeys<<<1, 1, 0, stream>>>(nullptr, nullptr, nullptr, 0, 0);
#ifndef __HIPCC__
    risc0_rv32im_launch_stepAccumTU(1, 1, stream, nullptr, 0, nullptr);
    risc0_rv32im_launch_par_stepExecTU(1, 1, stream, nullptr, 0, 0);
    finalizeZeroizeAccum<<<1, 1, 0, stream>>>(nullptr, 0);
#endif
    CUDA_OK(cudaStreamSynchronize(stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

} // extern "C"
