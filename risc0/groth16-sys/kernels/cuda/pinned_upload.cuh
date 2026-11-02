#pragma once
// Host -> device upload of mmap'd proving-key sections through pinned staging (CUDA).
// A pageable cudaMemcpyAsync holds a driver-wide lock for its whole duration (measured: other
// threads' cuMemAlloc / cuMemcpyHtoD / cudaStreamCreate block on pthread_rwlock_wrlock for
// 90-490 ms each), so a background load stalls any concurrent CUDA work (the STARK prover).
// Here worker threads copy chunks into pinned buffers and each chunk is uploaded with a short
// async copy from pinned memory on the stager's own non-blocking streams; chunk buffers are
// recycled through events.
//
// One process-wide stager (THP-pinned buffers allocated once, 128 MB), used by the background key
// load (`risc0_groth16_cuda_prestart`) and, after it, by later loads. Callers serialize on
// cached_mutex.
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

// ThpPinned (risc0-sys kernels/zkp/cuda/thp_pinned.h, passed in by build.rs: that directory
// can't go on the include path, its cuda.h would shadow the toolkit's).
#include RISC0_THP_PINNED_H

// cudaEventSynchronize is aliased by some sppark versions' cuda2hip.hpp and not others: use a
// name of this file's own.
#ifdef __HIPCC__
static const auto g16EventSynchronize = hipEventSynchronize;
#else
static const auto g16EventSynchronize = cudaEventSynchronize;
#endif

// The Groth16 cold-path optimizations (background key load, compact coefficients, bit-split
// MSMs, tuned MSM batching): on unless RISC0_CUDA_G16_COLD=0 (also read by the Rust side).
static bool g16_cold_enabled() {
  static const bool on = [] {
    const char* e = getenv("RISC0_CUDA_G16_COLD");
    return !(e && e[0] == '0');
  }();
  return on;
}

// Diagnostics on stderr with RISC0_VERBOSE set (as in rv32im-sys's CUDA library).
static bool g16_verbose() {
  static const bool on = getenv("RISC0_VERBOSE") != nullptr;
  return on;
}

// Set once a background key load has run: from then on key uploads use the stager.
static std::atomic<bool> g16_staged_upload{false};

static bool g16_staged() { return g16_staged_upload.load(); }

class key_stager {
  static constexpr size_t CH = 16 << 20;
  static constexpr int NTHR = 4, NB = 2;
  ThpPinned pool;  // NTHR * NB chunk buffers (one THP-backed registration)
  char* bufs[NTHR * NB] = {};
  cudaEvent_t ev[NTHR * NB] = {};
  cudaStream_t st[NTHR] = {};
  int device = -1;  // the streams' and events' device

  static std::mutex& inst_mu() { static std::mutex m; return m; }
  static key_stager*& inst() { static key_stager* s = nullptr; return s; }

public:
  key_stager() {
    try {
      CUDA_OK(cudaGetDevice(&device));
      pool.alloc(NTHR * NB * CH);
      for (int i = 0; i < NTHR * NB; i++) {
        bufs[i] = (char*)pool.ptr + i * CH;
        CUDA_OK(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming));
      }
      for (int t = 0; t < NTHR; t++)
        CUDA_OK(cudaStreamCreateWithFlags(&st[t], cudaStreamNonBlocking));
    } catch (...) {
      release();
      throw;
    }
  }
  ~key_stager() { release(); }
  // The process-wide stager, created on first use (risc0_groth16_cuda_stager_init creates it at
  // the start of a Groth16 session, while the STARK is still host-bound). Lives until take();
  // never destroyed at exit (like the cached SRS), avoiding teardown ordering issues. Users hold
  // cached_mutex while they use the returned reference.
  // Its streams belong to the device current at creation: on another device a new one replaces
  // it (the pinned memory is portable, but streams and events are not).
  static key_stager& get() {
    std::lock_guard<std::mutex> lk(inst_mu());
    int dev = 0;
    CUDA_OK(cudaGetDevice(&dev));
    if (inst() && inst()->device != dev) {
      delete inst();
      inst() = nullptr;
    }
    if (!inst()) inst() = new key_stager();
    return *inst();
  }
  // Detach the stager (caller holds cached_mutex, so no upload is in flight); the next get()
  // creates a new one. The caller deletes the result outside cached_mutex: freeing the pinned
  // memory waits for the whole device (ThpPinned::free_pinned).
  static key_stager* take() {
    std::lock_guard<std::mutex> lk(inst_mu());
    key_stager* s = inst();
    inst() = nullptr;
    return s;
  }
  // Copies `bytes` from host `src` to device `dst`; returns when the data is on the device
  // (every chunk's copy has completed, so `src` may be unmapped afterwards).
  void upload(void* dst, const void* src, size_t bytes) {
    size_t nch = (bytes + CH - 1) / CH;
    std::atomic<size_t> next{0};
    std::atomic<int> err{0};
    std::vector<std::thread> th;
    for (int t = 0; t < NTHR; t++)
      th.emplace_back([&, t] {
        int k = 0;
        for (size_t c; (c = next.fetch_add(1)) < nch; k++) {
          int b = t * NB + (k % NB);
          // The buffer's previous copy (if any) must be done before it is overwritten.
          if (g16EventSynchronize(ev[b]) != cudaSuccess) err = 1;
          size_t o = c * CH, n = std::min(CH, bytes - o);
          memcpy(bufs[b], (const char*)src + o, n);
          if (cudaMemcpyAsync((char*)dst + o, bufs[b], n, cudaMemcpyHostToDevice, st[t]) != cudaSuccess)
            err = 1;
          if (cudaEventRecord(ev[b], st[t]) != cudaSuccess) err = 1;
        }
        if (cudaStreamSynchronize(st[t]) != cudaSuccess) err = 1;
      });
    for (auto& x : th)
      x.join();
    if (err)
      throw sppark_error{EIO, "staged upload failed: "};
  }

private:
  void release() {
    for (int t = 0; t < NTHR; t++)
      if (st[t]) {
        cudaStreamSynchronize(st[t]);
        cudaStreamDestroy(st[t]);
        st[t] = nullptr;
      }
    for (int i = 0; i < NTHR * NB; i++)
      if (ev[i]) {
        cudaEventDestroy(ev[i]);
        ev[i] = nullptr;
      }
    pool.free_pinned();
  }
};
