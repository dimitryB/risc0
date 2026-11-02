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

#pragma once

// ROCm only: run a long kernel on a side hardware queue ("hop").
//
// On RDNA3/RDNA4 (RX 7900 XTX, RX 9070 XT; ROCm 7.2) a long kernel (>~1 ms, grid larger than one
// round of waves) delays whatever comes next on its own hardware queue: the next dispatch starts,
// and the queue's completion signal reaches the host, ~170-250 us after the kernel's last wave
// ends. The same dependency through an event on ANOTHER hardware queue resolves in ~11-14 us
// (host wakeup ~40 us). Neither stream priority, the barrier bit, timing events, a tiny kernel in
// between nor a host sync avoids it on the same queue. On RDNA3 the delay also disappears once the
// device has more than 8 compute queues (GPU_MAX_HW_QUEUES=8, set by risc0_zkp::hal::hip), so hops
// are on by default only on RDNA4 (gfx12), where the queue count does not change it.
//
// A hop scope moves the kernels launched in it to a per-device side stream, bracketed by events:
// main records e0, side waits for e0, the kernels run on side, side records e1, main waits for
// e1. Ordering is exactly that of launching on `main`: the scope's kernels start after all work
// queued on `main` before it, and all work queued on `main` after the scope (kernels, copies,
// event records, stream-ordered frees, host synchronization of `main`, the legacy stream's
// implicit sync) runs after them. The side stream is created with a full CU mask, which gives it
// a dedicated hardware queue (ROCclr shares the ordinary streams' queues, GPU_MAX_HW_QUEUES per
// priority, so a plain side stream may land on main's queue). The scope holds a process-wide lock
// from the e0 record to the e1 wait, so concurrent hops (any thread or device) serialize on the
// host for a few microseconds, and one side stream per device carries every hop in host order
// (every dependency points to earlier work: no cycles).
//
// Use only around kernel launches expected to run >~1 ms (the hop itself costs ~5-15 us), never
// around host synchronization or copies from pageable memory. RISC0_HIP_HOP: unset = all call
// sites on gfx12 devices and none elsewhere, 0 = off, else a mask of call sites (below) on every
// device.

#if defined(__HIPCC__)

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace risc0_hip_hop {

// Call sites (bits of RISC0_HIP_HOP).
enum : unsigned {
  kMerkleRows = 1, // zkp: Poseidon2 rows_k (sppark stream)
  kMerkleFold = 2, // zkp: Poseidon2 fold_k layers (sppark stream)
  kMix = 4,        // zkp: mix_poly_coeffs_v6
  kRvWitgen = 8,   // rv32im: par_stepExec
  kRvAccum = 16,   // rv32im: stepAccum, finalizeZeroizeAccum
  kRecCheck = 32,  // recursion: ec_part<P>
  kRecStep = 64,   // recursion: parStepExec, accum compute / verify
};

constexpr unsigned kAuto = ~0u; // RISC0_HIP_HOP unset: every call site, gfx12 devices only

inline unsigned mask() {
  static const unsigned m = [] {
    const char* v = std::getenv("RISC0_HIP_HOP");
    if (!v || !*v)
      return kAuto;
    unsigned m = unsigned(std::strtoul(v, nullptr, 0));
    return m == kAuto ? kAuto >> 1 : m; // an explicit all-ones mask means every device
  }();
  return m;
}
// Whether call site `bit` hops a kernel of `items` work items (>= 2^min_log2 of them). With the
// default mask the device's architecture decides in Scope.
inline bool want(unsigned bit, size_t items, unsigned min_log2) {
  return (mask() & bit) && items >= (size_t(1) << min_log2);
}

inline std::recursive_mutex& lock() {
  static std::recursive_mutex m;
  return m;
}

// Callers hold lock(). Whether device `dev` is RDNA4 (gfx12xx).
inline bool is_gfx12(int dev) {
  static signed char known[16] = {}; // 0 unknown, 1 gfx12, -1 other
  if (dev < 0 || dev >= 16)
    return false;
  if (!known[dev]) {
    hipDeviceProp_t prop;
    bool yes = hipGetDeviceProperties(&prop, dev) == hipSuccess && std::strncmp(prop.gcnArchName, "gfx12", 5) == 0;
    (void)hipGetLastError();
    known[dev] = yes ? 1 : -1;
  }
  return known[dev] > 0;
}

// A stream with a dedicated hardware queue (ROCclr does not pool streams created with a CU mask).
inline hipStream_t dedicated_stream() {
  uint32_t cus[8];
  for (uint32_t& m : cus)
    m = ~0u; // all CUs (bits past the device's CU count are ignored)
  hipStream_t s = nullptr;
  if (hipExtStreamCreateWithCUMask(&s, 8, cus) != hipSuccess) {
    (void)hipGetLastError();
    return nullptr;
  }
  return s;
}

struct Side {
  hipStream_t stream = nullptr;
  hipEvent_t e0 = nullptr;
  hipEvent_t e1 = nullptr;
  int state = 0; // 0 not tried, 1 ready, -1 creation failed
};

// Callers hold lock(); `dev` is the current device.
inline Side* side(int dev) {
  static Side sides[16];
  if (dev < 0 || dev >= 16)
    return nullptr;
  Side& s = sides[dev];
  if (s.state == 0) {
    s.state = -1;
    s.stream = dedicated_stream();
    if (s.stream && hipEventCreateWithFlags(&s.e0, hipEventDisableTiming) == hipSuccess &&
        hipEventCreateWithFlags(&s.e1, hipEventDisableTiming) == hipSuccess) {
      s.state = 1;
    } else {
      (void)hipGetLastError();
    }
  }
  return s.state == 1 ? &s : nullptr;
}

// Launch on the scope (it converts to the stream to use): the side stream when hopping, else main.
class Scope {
  hipStream_t main_;
  hipStream_t stream_;
  Side* side_ = nullptr;
  std::unique_lock<std::recursive_mutex> lk_;

public:
  explicit Scope(hipStream_t main, bool want = true) : main_(main), stream_(main) {
    if (!want || !main)
      return;
    // The side stream of the current device; only when that is main's device too.
    int dev = -1;
    hipDevice_t sdev = -1;
    if (hipGetDevice(&dev) != hipSuccess || hipStreamGetDevice(main, &sdev) != hipSuccess || sdev != dev) {
      (void)hipGetLastError();
      return;
    }
    lk_ = std::unique_lock<std::recursive_mutex>(lock());
    Side* s = (mask() != kAuto || is_gfx12(dev)) ? side(dev) : nullptr;
    if (s && s->stream != main && hipEventRecord(s->e0, main) == hipSuccess &&
        hipStreamWaitEvent(s->stream, s->e0, 0) == hipSuccess) {
      side_ = s;
      stream_ = s->stream;
    } else {
      // Off or failed before anything went to the side stream: launch on main.
      lk_.unlock();
    }
  }
  ~Scope() {
    if (!side_)
      return;
    // A launch error stays pending for the caller (HIP keeps the last error across these calls).
    if (hipEventRecord(side_->e1, side_->stream) != hipSuccess ||
        hipStreamWaitEvent(main_, side_->e1, 0) != hipSuccess) {
      // Order everything later on main after the side's work the slow way.
      (void)hipStreamSynchronize(side_->stream);
    }
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

  operator hipStream_t() const { return stream_; }
  bool hopped() const { return side_ != nullptr; }
};

// Make `stream` wait for event `done` recorded on another stream (nullptr stream: the host waits),
// then release the event (risc0_zkp_hip_fill_join: the prefill queue's fills).
inline hipError_t join_fill(hipStream_t stream, hipEvent_t done) {
  if (!done)
    return hipSuccess;
  hipError_t err = stream ? hipStreamWaitEvent(stream, done, 0) : hipEventSynchronize(done);
  if (err != hipSuccess)
    (void)hipEventSynchronize(done);
  (void)hipEventDestroy(done);
  return err;
}

} // namespace risc0_hip_hop

#endif // __HIPCC__
