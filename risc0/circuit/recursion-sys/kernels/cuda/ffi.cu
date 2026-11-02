// Copyright 2025 RISC Zero, Inc.
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

#include "context.h"
#include "cuda.h"
#include "fp.h"
#include "fpext.h"
#include "kernels.h"
#include "thp_pinned.h"

#ifndef __HIPCC__
#include "vendor/nvtx3/nvtx3.hpp"
#else
namespace nvtx3 { struct scoped_range { scoped_range(const char*) {} }; }
#endif

#include <cstring>
#ifdef __HIPCC__
#include <array>
namespace cuda { namespace std { using ::std::array; } }
#else
#include <cuda/std/array>
#endif
#ifndef __HIPCC__
#include <cuda_runtime.h>
#endif
#include <exception>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#ifdef _WIN32
#include <mutex>
#include <vector>
#endif
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <vector>
#include <cstdlib>
#include <map>
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
#endif

// Async recursion witgen (off by default on Windows: see recAsync; kill switch
// RISC0_CUDA_REC_WITGEN=0 restores the synchronous path): the witgen uploads and the accum's initial ones go on the context's stream
// (cudaMemcpyAsync/cudaMemsetAsync, a fill kernel) instead of the legacy stream, thrust's temporary storage comes from a per-thread cache and runs on that
// stream (no cudaMalloc/cudaFree per proof: cudaFree synchronizes the device and both take a
// driver ioctl that costs ~1 ms when the host is busy), the cached buffers grow with 25%
// headroom, and the warmup pre-grows the stack reservation and loads the split eval_check module.
// On HIP as well, where in addition (see RecStream and HostAccumContext::calcPrefixProducts) the
// context's stream is kept per thread and the accum's prefix products run on it, so that the
// witgen and the accum each wait on the host only at their start and end.
#define REC_ASYNC_AVAILABLE 1
static bool recAsync() {
#ifdef REC_ASYNC_AVAILABLE
  static const bool on = [] {
    const char* e = std::getenv("RISC0_CUDA_REC_WITGEN");
#ifdef _WIN32
    // Off by default on Windows: the contexts are managed memory, which the host reads between
    // kernels (e.g. verifyWom's numCycles), and under WDDM (no concurrentManagedAccess) a host
    // access while a kernel is in flight faults (STATUS_IN_PAGE_ERROR). RISC0_CUDA_REC_WITGEN=1
    // turns it on.
    return e && e[0] == '1';
#else
    return !(e && e[0] == '0');
#endif
  }();
  return on;
#else
  return false;
#endif
}

static uint32_t recGrow(uint32_t need) {
  return recAsync() ? need + need / 4 : need;
}

#ifdef REC_ASYNC_AVAILABLE
// thrust temporary allocations from a per-thread free list. Blocks are only reused by later
// thrust calls on the same thread, which run on the same (or a synchronized) stream.
struct RecThrustAlloc {
  typedef char value_type;
  std::multimap<std::ptrdiff_t, char*> freeBlocks;
  std::map<char*, std::ptrdiff_t> used;

  char* allocate(std::ptrdiff_t n) {
    char* p = nullptr;
    auto it = freeBlocks.lower_bound(n);
    if (it != freeBlocks.end()) {
      p = it->second;
      n = it->first;
      freeBlocks.erase(it);
    } else {
      n += n / 4;
      CUDA_OK(cudaMalloc(&p, n));
    }
    used[p] = n;
    return p;
  }

  void deallocate(char* p, size_t) {
    auto it = used.find(p);
    if (it == used.end()) return;
    freeBlocks.insert({it->second, p});
    used.erase(it);
  }

  ~RecThrustAlloc() {
#ifndef _WIN32
    for (auto& b : freeBlocks) cudaFree(b.second);
#endif
  }
};
static thread_local RecThrustAlloc g_recThrustAlloc;
extern "C" const char* rec_eval_check_split_warmup(size_t* localBytes);  // eval_check_split.cu
size_t risc0_recursion_accum_inl_warmup();                                  // accum_inl.cu
#endif

#ifdef __HIPCC__
// The witgen and accum contexts' stream. A context used to create its own (CudaStream), but on
// ROCm hipStreamCreate + hipStreamDestroy take ~0.3 ms (RX 7900 XTX / RX 9070 XT, ROCm 7.2), twice
// per recursion proof, between the GPU work of the proof. With the async path each thread keeps
// one stream per device instead (the same kind: hipStreamCreate's, blocking). Each witgen or accum
// call synchronizes it before returning, so the next call on the thread finds it idle, as it
// would a new stream. The kill switch keeps a stream per context.
class RecStream {
  cudaStream_t stream = nullptr;
  bool owned = false;

  struct PerThread {
    cudaStream_t streams[16] = {};
    ~PerThread() {
#ifndef _WIN32
      // As the thread's buffer caches below (on Windows a thread_local destructor runs under the
      // loader lock, where the runtime must not be called: the stream is left to process exit).
      for (cudaStream_t s : streams)
        if (s)
          cudaStreamDestroy(s);
#endif
    }
  };

public:
  RecStream() {
    if (!recAsync()) {
      cudaStreamCreate(&stream);
      owned = true;
      return;
    }
    static thread_local PerThread perThread;
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    if (device < 0 || device >= 16)
      throw std::runtime_error("recursion stream: device index out of range");
    if (!perThread.streams[device])
      CUDA_OK(cudaStreamCreate(&perThread.streams[device]));
    stream = perThread.streams[device];
  }
  ~RecStream() {
    if (owned) {
      cudaStreamDestroy(stream);
    } else if (std::uncaught_exceptions()) {
      // An error left the call early: let nothing it queued outlive it (its copies read the
      // caller's host buffers), as a new stream's destruction would have.
      cudaStreamSynchronize(stream);
    }
  }
  RecStream(const RecStream&) = delete;
  RecStream& operator=(const RecStream&) = delete;

  inline operator cudaStream_t() const { return stream; }
};
#else
using RecStream = CudaStream;
#endif

constexpr size_t kStepModeParallel = 0;
constexpr size_t kStepModeSeqForward = 1;
constexpr size_t kStepModeSeqReverse = 2;

namespace sppark {
void calcPrefixProducts(void* d_inout, uint32_t count);
} // namespace sppark

__device__ void nextStepExec(ExecContext* ctx, uint32_t cycle, uint32_t count) {
  if (cycle == 0 || ctx->isParSafeExec(cycle)) {
    Fp* args[3]{ctx->buffers.ctrl, ctx->buffers.global, ctx->buffers.data};
    // printf("step_exec(%u)\n", cycle);
    step_exec(ctx, ctx->totalCycles, cycle++, args[0], args[1], args[2], nullptr, nullptr);
    while (cycle < count && !ctx->isParSafeExec(cycle)) {
      // printf("  step_exec(%u)\n", cycle);
      step_exec(ctx, ctx->totalCycles, cycle++, args[0], args[1], args[2], nullptr, nullptr);
    }
  }
}

__global__ void parStepExec(ExecContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  // isParSafe == 2 marks the head of a chain whose rows the host already computed.
  if (cycle < count && ctx->trace->cycles[cycle].isParSafe != 2) {
    nextStepExec(ctx, cycle, count);
  }
}

#ifdef __HIPCC__
// Parallel exec in chain order (ROCm; see RISC0_HIP_REC_SORT below). In parStepExec a wave covers
// 32 consecutive cycles: the thread of each chain head runs its whole chain (the following cycles
// that are not parallel-safe: Poseidon2 and checked-bytes rows, up to 15 of them; longer ones run
// on the host) while the other lanes, mostly one-row micro/macro ops or chain rows, idle, and the
// rows of one wave take different step_exec arms. parStepExecPerm runs the same rows from a
// permutation that groups the heads by chain length (longest first), by the arms of the head and
// of its first chain row and by the head's micro opcodes or macro sub-op (recRowClass), so each
// wave runs one kind of chain or op in lockstep; the rows that are not heads come last and return
// at once. The rows are independent (that is what parallel-safe means; the unsorted kernel's waves
// also run in any order), so the witness is the same. Per segment (fib 1M succinct, lift + join)
// 7.6 -> 3.8 ms on an RX 7900 XTX, 9.5 -> 6.4 ms on an RX 9070 XT, the sort included.
constexpr uint32_t kRecSortBits = 26;
constexpr uint32_t kRecNotHead = (1u << kRecSortBits) - 1;

// RISC0_HIP_REC_SORT: the launches that run their rows in sorted order, a bit mask (default 7): 1
// parStepExec (chain order), 2 the accum kernels and 4 parStepVerifyWom (arm order); 0 turns them
// all off.
static uint32_t recSortMask() {
  static const uint32_t mask = [] {
    const char* e = std::getenv("RISC0_HIP_REC_SORT");
    return e ? uint32_t(std::strtoul(e, nullptr, 10)) : 7u;
  }();
  return mask;
}

static bool recSort() {
  return recSortMask() & 1;
}

static bool recAccumSort() {
  return recSortMask() & 2;
}

static bool recVerifyWomSort() {
  return recSortMask() & 4;
}

__device__ inline uint32_t recArm(const Fp* ctrl, size_t steps, uint32_t cycle) {
  uint32_t arm = 0;
  for (uint32_t k = 1; k <= 7; k++)
    arm = ctrl[k * steps + cycle] != 0 ? k : arm;
  return arm;
}

// The step arm (bits 12..14) and the branch within it (bits 0..11) that a row's step takes: for a
// micro-op row its three opcodes (ctrl columns 8, 12, 16), for a macro-op row which of its nine
// sub-op flags (columns 9..17) is set.
__device__ inline uint32_t recRowClass(const Fp* ctrl, size_t steps, uint32_t cycle) {
  uint32_t arm = recArm(ctrl, steps, cycle), sub = 0;
  if (arm == 1) {
    for (uint32_t k = 0; k < 3; k++)
      sub = (sub << 4) | min(ctrl[(8 + 4 * k) * steps + cycle].asUInt32(), 15u);
  } else if (arm == 2) {
    for (uint32_t k = 0; k < 9; k++)
      sub = ctrl[(9 + k) * steps + cycle] != 0 ? k + 1 : sub;
  }
  return (arm << 12) | sub;
}

// win == 0: one global order (key bits [0, 26)). win > 0: the same grouping within each
// 2^win-cycle window, chain length clamped to 15 (bits [0, 22) plus the window index above them), so
// a wave's rows stay closer in the trace. RDNA4 uses kRecExecWinGfx12: parStepExecPerm 3.01 -> 1.89
// ms per lift/join on an RX 9070 XT (fib 1M succinct; windows of 2^8 / 2^12 cycles: 2.66 / 2.05). On
// an RX 7900 XTX the windows were slower (1.74 -> 2.34 ms at 2^10), so 0 there.
// RISC0_HIP_REC_EXECWIN overrides it.
constexpr uint32_t kRecExecWinGfx12 = 10;
__global__ void recExecKeys(ExecContext* ctx, uint32_t* keys, uint32_t* vals, uint32_t win) {
  uint32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t n = ctx->trace->numCycles;
  if (c >= n) return;
  const PreflightCycle* cy = ctx->trace->cycles;
  uint32_t ps = cy[c].isParSafe;
  uint32_t key = win ? (1u << 22) - 1 : kRecNotHead;
  if ((c == 0 || ps != 0) && ps != 2) {
    uint32_t len = 0;
    while (c + 1 + len < n && len < 255 && cy[c + 1 + len].isParSafe == 0)
      len++;
    const Fp* ctrl = ctx->buffers.ctrl;
    size_t steps = ctx->totalCycles;
    uint32_t arm1 = len ? recArm(ctrl, steps, c + 1) : 0;
    if (win)
      key = ((15 - min(len, 14u)) << 18) | (arm1 << 15) | recRowClass(ctrl, steps, c);
    else
      key = ((255 - len) << 18) | (arm1 << 15) | recRowClass(ctrl, steps, c);
  }
  if (win)
    key |= (c >> win) << 22;
  keys[c] = key;
  vals[c] = c;
}

__global__ void parStepExecPerm(ExecContext* ctx, const uint32_t* perm) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  if (i < count) {
    uint32_t cycle = perm[i];
    if (ctx->trace->cycles[cycle].isParSafe != 2)
      nextStepExec(ctx, cycle, count);
  }
}

// Device buffers of the sorted launches: keys and values (double-buffered, cap each) and hipcub's
// temporary storage. Per thread, like g_recThrustAlloc: reused only by later sorts on the same
// thread, which run on the same stream (RecStream) after the kernels that read the previous
// permutation.
struct RecSortBufs {
  uint32_t* buf = nullptr;
  size_t cap = 0;
  void* temp = nullptr;
  size_t tempBytes = 0;
#ifndef _WIN32
  // Freed at thread exit, like g_execCache (see ~CachedExecBuffers).
  ~RecSortBufs() {
    cudaFree(buf);
    cudaFree(temp);
  }
#endif
};
static thread_local RecSortBufs g_recSort;

// Room for n keys and values; returns the first key array (the values start at 2 * cap).
static uint32_t* recSortKeys(uint32_t n) {
  RecSortBufs& b = g_recSort;
  if (n > b.cap) {
    if (b.buf) CUDA_OK(cudaFree(b.buf));
    b.cap = n + n / 4;
    CUDA_OK(cudaMalloc(&b.buf, b.cap * 4 * sizeof(uint32_t)));
  }
  return b.buf;
}

// Sorts the n (key, value) pairs written at recSortKeys(n) by key bits [0, endBit); returns the
// values (device; valid until the next sort on this thread).
static const uint32_t* recSortPairs(cudaStream_t stream, uint32_t n, int endBit) {
  RecSortBufs& b = g_recSort;
  hipcub::DoubleBuffer<uint32_t> keys(b.buf, b.buf + b.cap), vals(b.buf + 2 * b.cap, b.buf + 3 * b.cap);
  size_t need = 0;
  CUDA_OK(hipcub::DeviceRadixSort::SortPairs(nullptr, need, keys, vals, int(n), 0, endBit, stream));
  if (need > b.tempBytes) {
    if (b.temp) CUDA_OK(cudaFree(b.temp));
    b.tempBytes = need + need / 4;
    CUDA_OK(cudaMalloc(&b.temp, b.tempBytes));
  }
  CUDA_OK(hipcub::DeviceRadixSort::SortPairs(b.temp, need, keys, vals, int(n), 0, endBit, stream));
  return vals.Current();
}

// Permutations kept across proofs (RISC0_HIP_REC_PERMKEEP=0: made in every proof). Both sorts'
// keys are functions of the program: recAccumKeys reads only ctrl, and recExecKeys reads ctrl and
// the cycles' isParSafe flags, which the preflight derives from the code alone (and the host
// chains, which mark their heads 2, are the program's long chains). A program's ctrl buffer is
// cached on its proving thread (PROVER_CACHE, prove/mod.rs), so its device address names the
// program in the proofs of that thread. Any permutation of [0, n) gives the same witness (exec:
// each cycle's thread runs its chain if it is a head, accum and verifyWom: independent rows), so
// the key only guards the grouping, never the result. Saves a recExecKeys and a recAccumKeys with
// their radix sorts per lift/join after a program's first proof: ~0.4 ms (RX 7900 XTX) and ~0.55
// ms (RX 9070 XT) of serial GPU time.
struct RecKeptPerm {
  uint32_t* buf = nullptr;
  const Fp* ctrl = nullptr;
  uint32_t steps = 0, n = 0, tag = 0;
  uint64_t used = 0;
};
constexpr int kRecKeptPerms = 8;
struct RecKeptPerms {
  RecKeptPerm e[kRecKeptPerms];
  uint64_t clock = 0;
#ifndef _WIN32
  // Freed at thread exit, like g_recSort.
  ~RecKeptPerms() {
    for (auto& x : e)
      cudaFree(x.buf);
  }
#endif
};
static thread_local RecKeptPerms g_recKept;

static bool recPermKeep() {
  static const bool on = [] {
    const char* e = std::getenv("RISC0_HIP_REC_PERMKEEP");
    return !(e && e[0] == '0');
  }();
  return on;
}

// The kept permutation for the key, or nullptr.
static const uint32_t* recKeptFind(const Fp* ctrl, uint32_t steps, uint32_t n, uint32_t tag) {
  RecKeptPerms& k = g_recKept;
  for (auto& x : k.e) {
    if (x.buf && x.ctrl == ctrl && x.steps == steps && x.n == n && x.tag == tag) {
      x.used = ++k.clock;
      return x.buf;
    }
  }
  return nullptr;
}

// Keeps a copy of the n-entry device permutation `perm` (stream-ordered after its sort) under the
// key, in place of the least recently used one; returns the copy.
static const uint32_t* recKeptStore(cudaStream_t stream, const Fp* ctrl, uint32_t steps, uint32_t n,
                                    uint32_t tag, const uint32_t* perm) {
  RecKeptPerms& k = g_recKept;
  RecKeptPerm* v = &k.e[0];
  for (auto& x : k.e)
    v = x.used < v->used ? &x : v;
  // cudaFree waits for the device: nothing still reads the evicted permutation.
  if (v->buf) CUDA_OK(cudaFree(v->buf));
  *v = RecKeptPerm{};
  CUDA_OK(cudaMalloc(&v->buf, size_t(n) * sizeof(uint32_t)));
  CUDA_OK(cudaMemcpyAsync(v->buf, perm, size_t(n) * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
  v->ctrl = ctrl;
  v->steps = steps;
  v->n = n;
  v->tag = tag;
  v->used = ++k.clock;
  return v->buf;
}

// The cycles [0, n) in chain order. `ctrl` and `steps`: the host copies of ctx's (the kept
// permutation's key).
static const uint32_t* recSortedCycles(cudaStream_t stream, ExecContext* ctx, const Fp* ctrl,
                                       uint32_t steps, uint32_t n, const LaunchConfig& cfg) {
  static const int winEnv = [] {
    const char* e = std::getenv("RISC0_HIP_REC_EXECWIN");
    return e ? int(std::strtoul(e, nullptr, 10)) : -1;
  }();
  int dev = 0;
  uint32_t win = winEnv >= 0 ? uint32_t(winEnv)
                 : (cudaGetDevice(&dev) == cudaSuccess && risc0_hip_hop::is_gfx12(dev)) ? kRecExecWinGfx12
                                                                                        : 0u;
  // The window index must fit above bit 22.
  uint32_t endBit = kRecSortBits;
  if (win) {
    while (win < 31 && (uint64_t(n - 1) >> win) >= (1u << 10))
      win++;
    endBit = 22;
    while (endBit < 32 && (uint64_t(n - 1) >> win) >> (endBit - 22) != 0)
      endBit++;
  }
  uint32_t tag = 1 + win; // 0: recAccumPerm's
  if (recPermKeep()) {
    if (const uint32_t* kept = recKeptFind(ctrl, steps, n, tag))
      return kept;
  }
  uint32_t* k = recSortKeys(n);
  recExecKeys<<<cfg.grid, cfg.block, 0, stream>>>(ctx, k, k + 2 * g_recSort.cap, win);
  const uint32_t* perm = recSortPairs(stream, n, int(endBit));
  return recPermKeep() ? recKeptStore(stream, ctrl, steps, n, tag, perm) : perm;
}

// Rows of the one-step-per-row kernels (parStepComputeAccumInl / parStepVerifyAccumInl,
// parStepVerifyWom; independent rows) in a permutation that groups the rows of each
// 2^kRecAccumWin-row window by recRowClass, so a wave takes fewer branches while its rows stay
// close in the trace. Per segment (fib 1M succinct): accum compute + verify 3.4 -> 1.5 ms (RX 7900
// XTX) and 3.6 -> 1.5 ms (RX 9070 XT), parStepVerifyWom 1.0 -> 0.7 and 1.1 -> 0.7 ms, plus
// ~0.1 ms of recAccumKeys and the sorts.
constexpr uint32_t kRecAccumWin = 10;

__global__ void recAccumKeys(const Fp* ctrl, uint32_t steps, uint32_t n, uint32_t* keys, uint32_t* vals) {
  uint32_t c = blockDim.x * blockIdx.x + threadIdx.x;
  if (c >= n) return;
  keys[c] = ((c >> kRecAccumWin) << 15) | recRowClass(ctrl, steps, c);
  vals[c] = c;
}

static const uint32_t* recAccumPermSort(cudaStream_t stream, const Fp* ctrl, uint32_t steps, uint32_t n) {
  uint32_t* k = recSortKeys(n);
  recAccumKeys<<<(n + 255) / 256, 256, 0, stream>>>(ctrl, steps, n, k, k + 2 * g_recSort.cap);
  int endBit = 15;
  while (endBit < 32 && ((uint64_t(n - 1) >> kRecAccumWin) >> (endBit - 15)) != 0)
    endBit++;
  return recSortPairs(stream, n, endBit);
}

// One proof's three arm-order launches (parStepVerifyWom, then the accum compute and verify
// passes) use the same permutation (the same ctrl rows): it is made once, at the witgen's
// verifyWom, and kept in its own buffer for the accum calls of that proof (keyed by ctrl, steps
// and n; a new witgen on this thread makes it again). Any permutation of [0, n) gives the same
// witness (the rows are independent), so the key only guards the grouping. Saves two
// recAccumKeys + radix sorts (~10 kernels each) per lift/join. RISC0_HIP_REC_PERMCACHE=0 turns
// it off.
struct RecPermCache {
  uint32_t* buf = nullptr;
  size_t cap = 0;
  const Fp* ctrl = nullptr;
  uint32_t steps = 0, n = 0;
  bool valid = false;
#ifndef _WIN32
  // Freed at thread exit, like g_recSort.
  ~RecPermCache() { cudaFree(buf); }
#endif
};
static thread_local RecPermCache g_recPerm;

static bool recPermCacheOn() {
  static const bool on = [] {
    const char* e = std::getenv("RISC0_HIP_REC_PERMCACHE");
    return !(e && e[0] == '0');
  }();
  return on;
}

// `fresh`: make it again (the witgen's call); else reuse a matching one.
static const uint32_t* recAccumPerm(cudaStream_t stream, const Fp* ctrl, uint32_t steps, uint32_t n,
                                    bool fresh = false) {
  if (recPermKeep()) {
    if (const uint32_t* kept = recKeptFind(ctrl, steps, n, 0))
      return kept;
    return recKeptStore(stream, ctrl, steps, n, 0, recAccumPermSort(stream, ctrl, steps, n));
  }
  if (!recPermCacheOn())
    return recAccumPermSort(stream, ctrl, steps, n);
  RecPermCache& c = g_recPerm;
  if (!fresh && c.valid && c.ctrl == ctrl && c.steps == steps && c.n == n)
    return c.buf;
  c.valid = false;
  const uint32_t* perm = recAccumPermSort(stream, ctrl, steps, n);
  if (n > c.cap) {
    // cudaFree waits for the device: nothing still reads the old buffer.
    if (c.buf) CUDA_OK(cudaFree(c.buf));
    c.buf = nullptr;
    c.cap = 0;
    CUDA_OK(cudaMalloc(&c.buf, size_t(n) * sizeof(uint32_t)));
    c.cap = n;
  }
  CUDA_OK(cudaMemcpyAsync(c.buf, perm, size_t(n) * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
  c.ctrl = ctrl;
  c.steps = steps;
  c.n = n;
  c.valid = true;
  return c.buf;
}

// verifyWom's sort of the WOM argument rows (ROCm; RISC0_HIP_REC_WOMSORT=0 turns it off). There
// thrust::sort of the 20-byte rows is a merge sort: 2.6 / 3.3 ms per segment (fib 1M succinct) on
// an RX 7900 XTX / RX 9070 XT, ~0.55 / ~0.85 ms with this. Its output is the unique ascending
// order of the rows' (addr, value) words as compared by operator< (all five as asUInt32).
// recWomSort makes the same array from a radix sort by the first word only: a stable sort of
// (addr.asUInt32(), index) pairs and a gather. The rows of an address then stay in their original
// order, which is already their sorted order when they are all equal: in a valid trace an address
// holds the one value it was written with, and the unused rows all are the kInvalidPattern row.
// The gather flags each row that is out of order (it compares each row with its predecessor), and
// recWomFix sorts the rows of those addresses by the whole tuple, so the result is thrust's for
// any rows (checked bit for bit against thrust::sort on random rows, with and without addresses
// that hold several values).
static bool recWomSortOn() {
  static const bool on = [] {
    const char* e = std::getenv("RISC0_HIP_REC_WOMSORT");
    return !(e && e[0] == '0');
  }();
  return on;
}

__global__ void recWomKeys(const WomArgumentRow* rows, uint32_t n, WomArgumentRow* copy, uint32_t* keys,
                           uint32_t* vals) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  WomArgumentRow r = rows[i];
  copy[i] = r;
  keys[i] = r.addr.asUInt32();
  vals[i] = i;
}

__global__ void recWomGather(const WomArgumentRow* copy, const uint32_t* perm, uint32_t n, WomArgumentRow* rows,
                             uint32_t* flags, uint32_t* unsorted) {
  uint32_t j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j >= n) return;
  WomArgumentRow r = copy[perm[j]];
  rows[j] = r;
  bool out = j > 0 && r < copy[perm[j - 1]];
  flags[j] = out;
  if (out)
    atomicAdd(unsorted, 1u);
}

__device__ inline void recWomSwap(WomArgumentRow* a, uint32_t i, uint32_t j) {
  WomArgumentRow t = a[i];
  a[i] = a[j];
  a[j] = t;
}

__device__ void recWomSift(WomArgumentRow* a, uint32_t root, uint32_t end) {
  for (;;) {
    uint32_t child = 2 * root + 1;
    if (child >= end) return;
    if (child + 1 < end && a[child] < a[child + 1]) child++;
    if (!(a[root] < a[child])) return;
    recWomSwap(a, root, child);
    root = child;
  }
}

// Only does work when the gather flagged rows (never for a valid trace): the thread of a flagged
// row finds the first row of its address (whose flag is 0: it is greater than its predecessor),
// and the one that claims it heap-sorts that address's rows. The rows it permutes all have the
// same addr.asUInt32(), which is all the other threads read of them.
__global__ void recWomFix(WomArgumentRow* rows, uint32_t n, uint32_t* flags, const uint32_t* unsorted) {
  uint32_t j = blockDim.x * blockIdx.x + threadIdx.x;
  if (*unsorted == 0 || j >= n || flags[j] != 1) return;
  uint32_t key = rows[j].addr.asUInt32();
  uint32_t s = j;
  while (s > 0 && rows[s - 1].addr.asUInt32() == key)
    s--;
  if (atomicCAS(&flags[s], 0u, 2u) != 0) return;
  uint32_t e = j + 1;
  while (e < n && rows[e].addr.asUInt32() == key)
    e++;
  WomArgumentRow* a = rows + s;
  uint32_t m = e - s;
  for (uint32_t i = m / 2; i-- > 0;)
    recWomSift(a, i, m);
  for (uint32_t end = m - 1; end > 0; end--) {
    recWomSwap(a, 0, end);
    recWomSift(a, 0, end);
  }
}

// recWomSort's buffers: a copy of the rows, and cap flags followed by the unsorted count (per
// thread, like g_recSort).
struct RecWomSortBufs {
  WomArgumentRow* copy = nullptr;
  uint32_t* flags = nullptr;
  size_t cap = 0;
#ifndef _WIN32
  ~RecWomSortBufs() {
    cudaFree(copy);
    cudaFree(flags);
  }
#endif
};
static thread_local RecWomSortBufs g_recWomSort;

// rows[0, n) sorted as thrust::sort sorts them, on `stream`.
static void recWomSort(cudaStream_t stream, WomArgumentRow* rows, uint32_t n) {
  RecWomSortBufs& b = g_recWomSort;
  if (n > b.cap) {
    if (b.copy) CUDA_OK(cudaFree(b.copy));
    if (b.flags) CUDA_OK(cudaFree(b.flags));
    b.cap = n + n / 4;
    CUDA_OK(cudaMalloc(&b.copy, b.cap * sizeof(WomArgumentRow)));
    CUDA_OK(cudaMalloc(&b.flags, (b.cap + 1) * sizeof(uint32_t)));
  }
  WomArgumentRow* copy = b.copy;
  uint32_t* flags = b.flags;
  uint32_t* unsorted = flags + b.cap;
  uint32_t* k = recSortKeys(n);
  uint32_t grid = (n + 255) / 256;
  recWomKeys<<<grid, 256, 0, stream>>>(rows, n, copy, k, k + 2 * g_recSort.cap);
  const uint32_t* perm = recSortPairs(stream, n, 31);  // asUInt32() < P < 2^31
  CUDA_OK(cudaMemsetAsync(unsorted, 0, sizeof(uint32_t), stream));
  recWomGather<<<grid, 256, 0, stream>>>(copy, perm, n, rows, flags, unsorted);
  recWomFix<<<grid, 256, 0, stream>>>(rows, n, flags, unsorted);
}
#endif

// Rows of sequential chains executed on the host (see risc0_circuit_recursion_cpu_chains).
struct HostChains {
  const uint32_t* heads;    // chain head cycles
  uint32_t numHeads;
  const uint32_t* cycles;   // one entry per chain row
  const uint32_t* rows;     // numRows * dataCols raw Fp, row-major
  const uint32_t* womIdx;   // numRows
  const uint32_t* womRows;  // numRows * kMaxWomRowsPerCycle * 5
  uint32_t numRows;
  uint32_t dataCols;
};

__global__ void markHostChains(ExecContext* ctx, const uint32_t* heads, uint32_t n) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) ctx->trace->cycles[heads[i]].isParSafe = 2;
}

__global__ void scatterHostChains(ExecContext* ctx, const uint32_t* cycles, const uint32_t* rows,
                                  const uint32_t* womIdx, const WomArgumentRow* womRows,
                                  uint32_t n, uint32_t dataCols) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n * dataCols) return;
  uint32_t i = t / dataCols, k = t % dataCols, c = cycles[i];
  reinterpret_cast<uint32_t*>(ctx->buffers.data)[(size_t)k * ctx->totalCycles + c] = rows[t];
  if (k == 0) ctx->womIndex[c] = womIdx[i];
  if (k < kMaxWomRowsPerCycle)
    ctx->womRows[(size_t)c * kMaxWomRowsPerCycle + k] = womRows[(size_t)i * kMaxWomRowsPerCycle + k];
}

__global__ void fwdStepExec(ExecContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  if (cycle == 0) {
    Fp* args[3]{ctx->buffers.ctrl, ctx->buffers.global, ctx->buffers.data};
    for (uint32_t cycle = 0; cycle < count; cycle++) {
      // printf("step_exec(%u)\n", cycle);
      step_exec(ctx, ctx->totalCycles, cycle, args[0], args[1], args[2], nullptr, nullptr);
    }
  }
}

__global__ void revStepExec(ExecContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  if (cycle == count - 1) {
    Fp* args[3]{ctx->buffers.ctrl, ctx->buffers.global, ctx->buffers.data};
    for (uint32_t i = 0; i < count; i++) {
      uint32_t cycle = count - i - 1;
      nextStepExec(ctx, cycle, count);
    }
  }
}

__global__ void injectWomBacks(ExecContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  if (cycle < 1 || cycle >= count) {
    return;
  }

  Fp* data = ctx->buffers.data;
  uint32_t totalCycles = ctx->totalCycles;
  uint32_t idx = ctx->womIndex[cycle];
  if (idx) {
    const WomArgumentRow& prev = ctx->womRows[idx - 1];
    data[0 * totalCycles + cycle - 1] = prev.addr;
    data[1 * totalCycles + cycle - 1] = prev.value.elems[0];
    data[2 * totalCycles + cycle - 1] = prev.value.elems[1];
    data[3 * totalCycles + cycle - 1] = prev.value.elems[2];
    data[4 * totalCycles + cycle - 1] = prev.value.elems[3];
  } else {
    data[0 * totalCycles + cycle - 1] = 0;
    data[1 * totalCycles + cycle - 1] = 0;
    data[2 * totalCycles + cycle - 1] = 0;
    data[3 * totalCycles + cycle - 1] = 0;
    data[4 * totalCycles + cycle - 1] = 0;
  }
}

__global__ void parStepVerifyWom(ExecContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  if (cycle < count) {
    Fp* args[3]{ctx->buffers.ctrl, ctx->buffers.global, ctx->buffers.data};
    step_verify_mem(ctx, ctx->totalCycles, cycle, args[0], args[1], args[2], nullptr, nullptr);
  }
}

#ifdef __HIPCC__
// parStepVerifyWom over the cycles in perm's order (recAccumPerm: each window's cycles grouped by
// step arm); the cycles are independent.
__global__ void parStepVerifyWomPerm(ExecContext* ctx, const uint32_t* perm) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  if (i < count) {
    uint32_t cycle = perm[i];
    Fp* args[3]{ctx->buffers.ctrl, ctx->buffers.global, ctx->buffers.data};
    step_verify_mem(ctx, ctx->totalCycles, cycle, args[0], args[1], args[2], nullptr, nullptr);
  }
}
#endif

__global__ void fwdStepVerifyWom(ExecContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  if (cycle == 0) {
    Fp* args[3]{ctx->buffers.ctrl, ctx->buffers.global, ctx->buffers.data};
    for (uint32_t cycle = 0; cycle < count; cycle++) {
      // printf("step_verify_mem: %u\n", cycle);
      step_verify_mem(ctx, ctx->totalCycles, cycle, args[0], args[1], args[2], nullptr, nullptr);
    }
  }
}

void risc0_recursion_launch_accum_inl(int verify, unsigned grid, unsigned block, cudaStream_t stream,
                                      AccumContext* ctx);
#ifdef __HIPCC__
void risc0_recursion_launch_accum_inl_perm(int verify, unsigned grid, unsigned block, cudaStream_t stream,
                                           AccumContext* ctx, const uint32_t* perm);  // accum_inl.cu
#endif
// accum_inl.cu: the accum steps force-inlined into the kernels, and eval_check split into
// standalone kernels (ffi_supra.cu): on unless RISC0_CUDA_REC_KERNELS=0. The switch selects
// kernels, not sources: with it off, parStepComputeAccum / parStepVerifyAccum still run the
// step_*_accum.cu sources with the branch-free asserts (bit-identical results either way).
static bool accumInl() {
  static const bool on = [] {
    const char* e = getenv("RISC0_CUDA_REC_KERNELS");
    return !(e && e[0] == '0');
  }();
  return on;
}

__global__ void parStepComputeAccum(AccumContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < ctx->workCycles) {
    step_compute_accum(ctx,
                       ctx->totalCycles,
                       cycle,
                       ctx->buffers.ctrl,
                       ctx->buffers.global,
                       ctx->buffers.data,
                       ctx->buffers.mix,
                       ctx->buffers.accum);
  }
}

__global__ void parStepVerifyAccum(AccumContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < ctx->workCycles) {
    step_verify_accum(ctx,
                      ctx->totalCycles,
                      cycle,
                      ctx->buffers.ctrl,
                      ctx->buffers.global,
                      ctx->buffers.data,
                      ctx->buffers.mix,
                      ctx->buffers.accum);
  }
}

#ifdef _WIN32
// Thread exit for the per-thread GPU buffer caches below, on Windows.
//
// MSVC constructs namespace-scope thread_locals on every thread in the process, and
// destroys them in the thread-detach callback, under the loader lock, where CUDA must
// not be called: a thread that never touched the GPU deadlocked there, initializing a
// CUDA context for its `cudaFree(nullptr)`. (GCC builds thread_locals on first use and
// destroys them outside any loader lock, so elsewhere they are freed as before.) Here a
// cache is never freed at thread exit: its buffers, if it has any, are parked for the
// next thread on the same device, which is host-side bookkeeping alone.
template <typename State> class ParkedBuffers {
  std::mutex lock;
  std::vector<State> parked;

public:
  static ParkedBuffers& get() {
    static ParkedBuffers pool;
    return pool;
  }

  void park(const State& state) {
    std::lock_guard<std::mutex> guard(lock);
    parked.push_back(state);
  }

  // Moves a parked state for `device` into `out`, if there is one.
  void take(int device, State& out) {
    std::lock_guard<std::mutex> guard(lock);
    for (size_t i = 0; i < parked.size(); i++) {
      if (parked[i].device == device) {
        out = parked[i];
        parked.erase(parked.begin() + i);
        return;
      }
    }
  }
};
#endif

struct ExecBuffersState {
  ExecContext* ctx = nullptr;
  PreflightTrace* trace = nullptr;
  WomArgumentRow* womRows = nullptr;
  uint32_t* womIndex = nullptr;
  FpExt* wom = nullptr;
  PreflightCycle* cycles = nullptr;
  FpExt* iops = nullptr;
  uint32_t maxCycles = 0;
  uint32_t maxWoms = 0;
  uint32_t maxIops = 0;
  int device = -1;
};

// Cached GPU buffers for HostExecContext to avoid repeated alloc/free overhead.
// Pre-allocated to the max size seen and reused across calls.
struct CachedExecBuffers : ExecBuffersState {
  void ensure(uint32_t numCycles, uint32_t numWoms, uint32_t numIops) {
#ifdef _WIN32
    if (!ctx) {
      CUDA_OK(cudaGetDevice(&device));
      ParkedBuffers<ExecBuffersState>::get().take(device, *this);
    }
#endif
    if (!ctx) {
      CUDA_OK(cudaMallocManaged(&ctx, sizeof(ExecContext)));
      CUDA_OK(cudaMallocManaged(&trace, sizeof(PreflightTrace)));
    }
    if (numCycles > maxCycles) {
      numCycles = recGrow(numCycles);
      cudaFree(womRows);
      cudaFree(womIndex);
      cudaFree(cycles);
      CUDA_OK(cudaMalloc(&womRows, numCycles * kMaxWomRowsPerCycle * sizeof(WomArgumentRow)));
      CUDA_OK(cudaMalloc(&womIndex, numCycles * sizeof(uint32_t)));
      CUDA_OK(cudaMalloc(&cycles, numCycles * sizeof(PreflightCycle)));
      maxCycles = numCycles;
    }
    if (numWoms > maxWoms) {
      numWoms = recGrow(numWoms);
      cudaFree(wom);
      CUDA_OK(cudaMalloc(&wom, numWoms * sizeof(FpExt)));
      maxWoms = numWoms;
    }
    if (numIops > maxIops) {
      numIops = recGrow(numIops);
      cudaFree(iops);
      CUDA_OK(cudaMalloc(&iops, numIops * sizeof(FpExt)));
      maxIops = numIops;
    }
  }

  ~CachedExecBuffers() {
#ifdef _WIN32
    if (ctx) {
      ParkedBuffers<ExecBuffersState>::get().park(*this);
    }
#else
    cudaFree(iops);
    cudaFree(cycles);
    cudaFree(wom);
    cudaFree(womIndex);
    cudaFree(womRows);
    cudaFree(trace);
    cudaFree(ctx);
#endif
  }
};

static thread_local CachedExecBuffers g_execCache;

#ifdef __HIPCC__
// Witgen inputs uploaded ahead of their proof (HIP; RISC0_HIP_REC_STAGE=0 turns it off). A recursion
// proof's witgen used to start by copying its trace and host chains (~50 MB: ~1.4 ms of PCIe on an
// RX 7900 XTX or RX 9070 XT) to the device while the GPU computed nothing. `prepare`, on its own
// thread while the GPU proves the previous proof, now hands its page-locked copies to
// risc0_circuit_recursion_hip_stage: they are copied into one of two slots' device buffers on a
// non-blocking stream, and the witgen of that preflight claims the slot by the host addresses and
// sizes of its inputs, waits for the copies' event on its stream and reads the slot's buffers. A
// live preflight's page-locked buffer is its own (the pool hands a buffer to one owner at a time),
// and a slot is cleared when its witgen is done with it or, if never claimed, when that buffer is
// released (risc0_circuit_recursion_hip_unstage, which first waits for the copies), so a key can
// never match another preflight's data. With both slots taken a preflight is not staged and its
// witgen copies as before.
static bool recStage() {
  static const bool on = [] {
    const char* e = std::getenv("RISC0_HIP_REC_STAGE");
    return recAsync() && !(e && e[0] == '0');
  }();
  return on;
}

struct StagedInputs {
  // Key: the preflight's page-locked arrays and their sizes.
  const void* wom = nullptr;
  uint32_t numWoms = 0, numCycles = 0, numIops = 0;
  const void* heads = nullptr;
  uint32_t numHeads = 0, numRows = 0, dataCols = 0;
  int device = -1;
  bool valid = false; // staged and not claimed
  bool busy = false;  // claimed by a running witgen
  FpExt* dWom = nullptr;
  size_t womCap = 0;
  PreflightCycle* dCycles = nullptr;
  size_t cyclesCap = 0;
  FpExt* dIops = nullptr;
  size_t iopsCap = 0;
  uint32_t* dHc = nullptr; // host chains, laid out as in applyHostChains
  size_t hcCap = 0;        // words
  cudaStream_t stream = nullptr;
  cudaEvent_t done = nullptr;
};

constexpr int kStageSlots = 2;
static std::mutex g_stageMu;
static StagedInputs g_stage[kStageSlots];

// The copy stream of the current device (callers hold g_stageMu), shared by the slots: one with
// its own hardware queue (hip_hop.h's dedicated_stream). ROCclr maps plain streams onto the
// GPU_MAX_HW_QUEUES shared queues, and with one (the RDNA4 default) the copies waited behind the
// running proof's kernels and overlapped nothing.
static cudaStream_t stageStream(int device) {
  static cudaStream_t streams[16] = {};
  if (device < 0 || device >= 16)
    throw std::runtime_error("recursion staging: device index out of range");
  if (!streams[device]) {
    cudaStream_t s = risc0_hip_hop::dedicated_stream();
    if (!s) CUDA_OK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    streams[device] = s;
  }
  return streams[device];
}

template <typename T> static void stageGrow(T*& p, size_t& cap, size_t n) {
  n = n ? n : 1;
  if (n <= cap) return;
  n += n / 4;
  // The slot is idle (neither staged nor claimed): nothing reads the old buffer.
  if (p) cudaFree(p);
  p = nullptr;
  cap = 0;
  CUDA_OK(cudaMalloc(&p, n * sizeof(T)));
  cap = n;
}

// The slot staged for `trace` on this thread's device, now claimed by the caller; or nullptr.
static StagedInputs* stageClaim(const PreflightTrace* trace) {
  if (!recStage()) return nullptr;
  int device = 0;
  CUDA_OK(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lk(g_stageMu);
  for (auto& s : g_stage) {
    if (s.valid && !s.busy && s.device == device && s.wom == trace->wom &&
        s.numWoms == trace->numWoms && s.numCycles == trace->numCycles &&
        s.numIops == trace->numIops) {
      s.valid = false;
      s.busy = true;
      return &s;
    }
  }
  return nullptr;
}

// Gives a claimed slot back when the witgen call ends. Declared before the HostExecContext, so it
// runs after the context's destruction: by then the witgen's stream is synchronized (the last
// step syncs it; on an exception RecStream's destructor does), and nothing reads the buffers.
struct StageClaim {
  StagedInputs* s;
  explicit StageClaim(StagedInputs* s) : s(s) {}
  ~StageClaim() {
    if (!s) return;
    std::lock_guard<std::mutex> lk(g_stageMu);
    s->busy = false;
    s->wom = nullptr;
    s->heads = nullptr;
  }
  StageClaim(const StageClaim&) = delete;
  StageClaim& operator=(const StageClaim&) = delete;
};
#endif


struct HostExecContext {
  ExecContext* ctx;
  RecStream stream;
  LaunchConfig cfg;
  // ctx->trace->numCycles, kept on the host: ctx is managed memory (see verifyWom).
  uint32_t numCycles;
#ifdef __HIPCC__
  // For recAccumPerm (host copies, like numCycles).
  const Fp* ctrlBuf = nullptr;
  uint32_t steps = 0;
  // This witgen's inputs, if they were uploaded ahead (see StagedInputs).
  StagedInputs* staged = nullptr;
#endif

  HostExecContext(ExecBuffers* buffers, PreflightTrace* trace, size_t totalCycles
#ifdef __HIPCC__
                  , StagedInputs* stagedInputs = nullptr
#endif
                  )
      : cfg(getSimpleConfig(trace->numCycles)), numCycles(trace->numCycles) {
    g_execCache.ensure(trace->numCycles, trace->numWoms, trace->numIops);

    ctx = g_execCache.ctx;
    ctx->buffers.ctrl = buffers->ctrl;
    ctx->buffers.data = buffers->data;
    ctx->buffers.global = buffers->global;
    ctx->totalCycles = totalCycles;
#ifdef __HIPCC__
    ctrlBuf = buffers->ctrl;
    steps = totalCycles;
#endif

    ctx->trace = g_execCache.trace;
    ctx->trace->numWoms = trace->numWoms;
    ctx->trace->numCycles = trace->numCycles;
    ctx->trace->numIops = trace->numIops;

    ctx->trace->wom = g_execCache.wom;
    ctx->trace->cycles = g_execCache.cycles;
    ctx->trace->iops = g_execCache.iops;
    ctx->womRows = g_execCache.womRows;
    ctx->womIndex = g_execCache.womIndex;
    if (recAsync()) {
      // Stream-ordered before the witgen kernels. The host arrays (preflight's page-locked
      // staging, or pageable when that is off, which the driver stages before returning) stay
      // alive until this witgen call returns, after its final stream synchronize. The device
      // pointers come from g_execCache, not from ctx / ctx->trace: those are managed memory,
      // which the host must not touch while work is queued on a device without concurrent
      // managed access.
#ifdef __HIPCC__
      if (stagedInputs) {
        // Uploaded ahead (see StagedInputs): the kernels read the slot's copies, after them.
        CUDA_OK(cudaStreamWaitEvent(stream, stagedInputs->done, 0));
        staged = stagedInputs;
        ctx->trace->wom = staged->dWom;
        ctx->trace->cycles = staged->dCycles;
        ctx->trace->iops = staged->dIops;
      } else {
#endif
      CUDA_OK(cudaMemcpyAsync(g_execCache.wom, trace->wom, trace->numWoms * sizeof(FpExt),
                              cudaMemcpyHostToDevice, stream));
      CUDA_OK(cudaMemcpyAsync(g_execCache.cycles, trace->cycles,
                              trace->numCycles * sizeof(PreflightCycle), cudaMemcpyHostToDevice,
                              stream));
      CUDA_OK(cudaMemcpyAsync(g_execCache.iops, trace->iops, trace->numIops * sizeof(FpExt),
                              cudaMemcpyHostToDevice, stream));
#ifdef __HIPCC__
      }
#endif
      CUDA_OK(cudaMemsetAsync(g_execCache.womRows, kInvalidPattern,
                              trace->numCycles * kMaxWomRowsPerCycle * sizeof(WomArgumentRow),
                              stream));
      CUDA_OK(cudaMemsetAsync(g_execCache.womIndex, 0, trace->numCycles * sizeof(uint32_t), stream));
      return;
    }
    CUDA_OK(cudaMemcpy(
        ctx->trace->wom, trace->wom, trace->numWoms * sizeof(FpExt), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(ctx->trace->cycles,
                       trace->cycles,
                       trace->numCycles * sizeof(PreflightCycle),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(
        ctx->trace->iops, trace->iops, trace->numIops * sizeof(FpExt), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(ctx->womRows,
                       kInvalidPattern,
                       trace->numCycles * kMaxWomRowsPerCycle * sizeof(WomArgumentRow)));
    CUDA_OK(cudaMemset(ctx->womIndex, 0, trace->numCycles * sizeof(uint32_t)));
  }

  ~HostExecContext() {
    // Buffers are owned by g_execCache, not freed here
  }

  // Upload the host-executed chains, write their rows, and mark their heads to be skipped.
  void applyHostChains(const HostChains* hc) {
    nvtx3::scoped_range range("hostChains");
    static_assert(sizeof(WomArgumentRow) == 20, "WomArgumentRow layout");
    if (hc->dataCols != 128) throw std::runtime_error("host chains: unexpected data width");
#ifdef __HIPCC__
    if (staged && staged->heads == hc->heads && staged->numHeads == hc->numHeads &&
        staged->numRows == hc->numRows && staged->dataCols == hc->dataCols) {
      // Uploaded ahead with the trace (same layout as below; the stream waits for the copies).
      uint32_t* sHeads = staged->dHc;
      uint32_t* sCycles = sHeads + hc->numHeads;
      uint32_t* sRows = sCycles + hc->numRows;
      uint32_t* sWomIdx = sRows + (size_t)hc->numRows * hc->dataCols;
      uint32_t* sWomRows = sWomIdx + hc->numRows;
      markHostChains<<<(hc->numHeads + 255) / 256, 256, 0, stream>>>(ctx, sHeads, hc->numHeads);
      uint32_t n = hc->numRows * hc->dataCols;
      scatterHostChains<<<(n + 255) / 256, 256, 0, stream>>>(
          ctx, sCycles, sRows, sWomIdx, reinterpret_cast<const WomArgumentRow*>(sWomRows), hc->numRows, hc->dataCols);
      return;
    }
#endif
    static thread_local uint32_t* dBuf = nullptr;
    static thread_local size_t dCap = 0;
    size_t nWords = hc->numHeads + (size_t)hc->numRows * (2 + hc->dataCols + kMaxWomRowsPerCycle * 5);
    if (nWords > dCap) {
      if (recAsync()) nWords += nWords / 4;
      if (dBuf) cudaFree(dBuf);
      CUDA_OK(cudaMalloc(&dBuf, nWords * 4));
      dCap = nWords;
    }
    uint32_t* dHeads = dBuf;
    uint32_t* dCycles = dHeads + hc->numHeads;
    uint32_t* dRows = dCycles + hc->numRows;
    uint32_t* dWomIdx = dRows + (size_t)hc->numRows * hc->dataCols;
    uint32_t* dWomRows = dWomIdx + hc->numRows;
    CUDA_OK(cudaMemcpyAsync(dHeads, hc->heads, hc->numHeads * 4, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(dCycles, hc->cycles, hc->numRows * 4, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(dRows, hc->rows, (size_t)hc->numRows * hc->dataCols * 4, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(dWomIdx, hc->womIdx, hc->numRows * 4, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(dWomRows, hc->womRows, (size_t)hc->numRows * kMaxWomRowsPerCycle * 20, cudaMemcpyHostToDevice, stream));
    markHostChains<<<(hc->numHeads + 255) / 256, 256, 0, stream>>>(ctx, dHeads, hc->numHeads);
    uint32_t n = hc->numRows * hc->dataCols;
    scatterHostChains<<<(n + 255) / 256, 256, 0, stream>>>(
        ctx, dCycles, dRows, dWomIdx, reinterpret_cast<const WomArgumentRow*>(dWomRows), hc->numRows, hc->dataCols);
    // Ordered before doStepExec on `stream`; the host buffers are pageable (staged at enqueue)
    // and the caller keeps them alive until this witgen call returns (it syncs).
  }

  void doStepExec(uint32_t mode) {
    nvtx3::scoped_range range("stepExec");
    switch (mode) {
    case kStepModeParallel: {
#ifdef __HIPCC__
      {
        // The sort's kernels go on `stream` before the hop scope opens, so the hopped
        // kernel (which waits only for work recorded before the scope) sees the permutation.
        const uint32_t* perm =
            (recSort() && numCycles > 1) ? recSortedCycles(stream, ctx, ctrlBuf, steps, numCycles, cfg) : nullptr;
        // ~4-5 ms per lift/join (~2^18 cycles): on a side hardware queue (hip_hop.h).
        risc0_hip_hop::Scope s(stream, risc0_hip_hop::want(risc0_hip_hop::kRecStep, size_t(cfg.grid.x) * cfg.block.x, 16));
        if (perm)
          parStepExecPerm<<<cfg.grid, cfg.block, 0, s>>>(ctx, perm);
        else
          parStepExec<<<cfg.grid, cfg.block, 0, s>>>(ctx);
      }
#else
      parStepExec<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
#endif
    } break;
    case kStepModeSeqForward: {
      fwdStepExec<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
    } break;
    case kStepModeSeqReverse: {
      revStepExec<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
    } break;
    }
#ifdef __HIPCC__
    // verifyWom's work follows on the same stream (thrust without a host sync, see there).
    if (!recAsync())
#endif
    CUDA_OK(cudaStreamSynchronize(stream));
  }

  void verifyWom(uint32_t mode) {
    nvtx3::scoped_range range("verifyWom");
    // The values of ctx->trace->numCycles, ctx->womRows and ctx->womIndex, read from their host
    // copies: ctx is managed memory, and on HIP the async path no longer waits for the exec kernel
    // (doStepExec) before this point.
    WomArgumentRow* womRows = g_execCache.womRows;
    uint32_t* womIndex = g_execCache.womIndex;

#ifdef REC_ASYNC_AVAILABLE
    if (recAsync()) {
      // Same thrust algorithms (the sort compares every field, so its output is unique), on the
      // context stream with cached temporary storage; par_nosync: no host wait in between.
#ifdef __HIPCC__
      auto policy = thrust::hip::par_nosync(g_recThrustAlloc).on(stream);
#else
      auto policy = thrust::cuda::par_nosync(g_recThrustAlloc).on(stream);
#endif
      {
        nvtx3::scoped_range range("sortWom");
#ifdef __HIPCC__
        if (recWomSortOn())
          recWomSort(stream, womRows, numCycles * kMaxWomRowsPerCycle);
        else
#endif
        thrust::sort(policy, womRows, womRows + numCycles * kMaxWomRowsPerCycle);
      }
      {
        nvtx3::scoped_range range("scan");
        thrust::exclusive_scan(policy, womIndex, womIndex + numCycles, womIndex);
      }
    } else
#endif
    {
      {
        nvtx3::scoped_range range("sortWom");
        thrust::sort(thrust::device, womRows, womRows + numCycles * kMaxWomRowsPerCycle);
      }
      {
        nvtx3::scoped_range range("scan");
        thrust::exclusive_scan(thrust::device, womIndex, womIndex + numCycles, womIndex);
      }
    }

    {
      nvtx3::scoped_range range("injectWomBacks");
      injectWomBacks<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
      // No intermediate sync — the next kernel runs on the same stream and
      // is naturally ordered after this one.
    }

    {
      nvtx3::scoped_range range("stepVerifyWom");
#ifdef __HIPCC__
      if (recVerifyWomSort() && numCycles > 1) {
        const uint32_t* perm = recAccumPerm(stream, ctrlBuf, steps, numCycles, true);
        parStepVerifyWomPerm<<<cfg.grid, cfg.block, 0, stream>>>(ctx, perm);
      } else
#endif
      parStepVerifyWom<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
      CUDA_OK(cudaStreamSynchronize(stream));
    }
  }
};

struct AccumBuffersState {
  AccumContext* ctx = nullptr;
  FpExt* accum = nullptr;
  uint32_t maxWorkCycles = 0;
  int device = -1;
};

#if defined(SPPARK_NO_COOPERATIVE_LAUNCH) || defined(__HIPCC__)
// In-place inclusive prefix products of an FpExt array without a grid-wide sync (no cooperative
// launch on this platform, or on HIP, none wanted: see calcPrefixProducts): each block scans a
// chunk and writes its total, one thread scans the totals, then each chunk past the first is
// multiplied by the product of the chunks before it.
// FpExt multiplication is associative, so the products are those of the sequential scan.
constexpr uint32_t kScanThreads = 256;
constexpr uint32_t kScanPerThread = 8;
constexpr uint32_t kScanChunk = kScanThreads * kScanPerThread;

__global__ void __launch_bounds__(kScanThreads)
    prefixProductsChunks(FpExt* io, uint32_t n, FpExt* totals) {
  // Raw storage: HIP has no __shared__ objects with constructors.
  __shared__ uint32_t raw[kScanThreads * 4] __attribute__((aligned(16)));
  FpExt* part = reinterpret_cast<FpExt*>(raw);
  uint32_t t = threadIdx.x;
  size_t base = (size_t)blockIdx.x * kScanChunk + (size_t)t * kScanPerThread;
  FpExt run(1);
  for (uint32_t k = 0; k < kScanPerThread; k++) {
    size_t i = base + k;
    if (i < n) {
      run = run * io[i];
      io[i] = run;
    }
  }
  part[t] = run;
  __syncthreads();
  // Inclusive scan of the threads' products (Hillis-Steele).
  for (uint32_t d = 1; d < kScanThreads; d <<= 1) {
    FpExt v = part[t];
    if (t >= d)
      v = part[t - d] * v;
    __syncthreads();
    part[t] = v;
    __syncthreads();
  }
  if (t > 0) {
    FpExt before = part[t - 1];
    for (uint32_t k = 0; k < kScanPerThread; k++) {
      size_t i = base + k;
      if (i < n)
        io[i] = before * io[i];
    }
  }
  if (t == kScanThreads - 1)
    totals[blockIdx.x] = part[t];
}

// One thread: totals[b] becomes the product of chunks 0..b (inclusive).
__global__ void prefixProductsTotals(FpExt* totals, uint32_t nblocks) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  for (uint32_t b = 1; b < nblocks; b++)
    totals[b] = totals[b - 1] * totals[b];
}

__global__ void prefixProductsFixup(FpExt* io, uint32_t n, const FpExt* totals) {
  size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x + kScanChunk;
  if (i < n)
    io[i] = totals[i / kScanChunk - 1] * io[i];
}
#endif

// Cached GPU buffers for HostAccumContext
struct CachedAccumBuffers : AccumBuffersState {
  void ensure(uint32_t workCycles) {
#ifdef _WIN32
    if (!ctx) {
      CUDA_OK(cudaGetDevice(&device));
      ParkedBuffers<AccumBuffersState>::get().take(device, *this);
    }
#endif
    if (!ctx) {
      CUDA_OK(cudaMallocManaged(&ctx, sizeof(AccumContext)));
    }
    if (workCycles > maxWorkCycles) {
      workCycles = recGrow(workCycles);
      cudaFree(accum);
      CUDA_OK(cudaMalloc(&accum, workCycles * sizeof(FpExt)));
      maxWorkCycles = workCycles;
    }
  }

  ~CachedAccumBuffers() {
#ifdef _WIN32
    if (ctx) {
      ParkedBuffers<AccumBuffersState>::get().park(*this);
    }
#else
    cudaFree(accum);
    cudaFree(ctx);
#endif
  }
};

static thread_local CachedAccumBuffers g_accumCache;

__global__ void fillOneExt(FpExt* p, uint32_t n) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    p[i] = FpExt(1);
  }
}

struct HostAccumContext {
  AccumContext* ctx;
  RecStream stream;
  LaunchConfig cfg;
  // ctx->workCycles, kept on the host: ctx is managed memory, which the host must not touch while
  // the steps may run (see syncSteps).
  uint32_t workCycles;
#ifdef __HIPCC__
  // For recAccumPerm (host copies, like workCycles).
  const Fp* ctrlBuf = nullptr;
  uint32_t steps = 0;
#endif

  HostAccumContext(AccumBuffers* buffers, size_t workCycles, size_t totalCycles)
      : cfg(getSimpleConfig(workCycles)), workCycles(workCycles) {
    g_accumCache.ensure(workCycles);

    ctx = g_accumCache.ctx;
    ctx->buffers.ctrl = buffers->ctrl;
    ctx->buffers.global = buffers->global;
    ctx->buffers.data = buffers->data;
    ctx->buffers.mix = buffers->mix;
#ifdef __HIPCC__
    ctrlBuf = buffers->ctrl;
    steps = totalCycles;
#endif
    ctx->buffers.accum = buffers->accum;
    ctx->totalCycles = totalCycles;
    ctx->workCycles = workCycles;

    ctx->accum = g_accumCache.accum;
#ifdef SPPARK_NO_COOPERATIVE_LAUNCH
    // Its prefix products run on the device too (see calcPrefixProducts).
    const bool device_fill = true;
#else
    const bool device_fill = recAsync();
#endif
    if (device_fill) {
      // Fill with FpExt(1) on the device instead of uploading a host vector of ones.
      fillOneExt<<<(workCycles + 255) / 256, 256, 0, stream>>>(g_accumCache.accum, workCycles);
      CUDA_OK(cudaGetLastError());
      return;
    }
    std::vector<FpExt> accumInit(workCycles, FpExt(1));
    CUDA_OK(cudaMemcpy(
        ctx->accum, accumInit.data(), workCycles * sizeof(FpExt), cudaMemcpyHostToDevice));
  }

  ~HostAccumContext() {
    // Buffers owned by g_accumCache, not freed here
  }

  // Whether the host waits for the stream after each step (and for the device before
  // verifyAccum, for sppark's stream). Not with the async path on HIP: there every step, the
  // prefix products included (see calcPrefixProducts), runs on the context's stream, which orders
  // them, after the cudaDeviceSynchronize at the accum's start (risc0_circuit_recursion_cuda_accum)
  // ordered it after the work of every other stream; verifyAccum still waits at the end.
  static bool syncSteps() {
#ifdef __HIPCC__
    return !recAsync();
#else
    return true;
#endif
  }

  void computeAccum() {
    nvtx3::scoped_range range("computeAccum");
#ifdef __HIPCC__
    {
      // The witgen's verifyWom permutation when it matches (RecPermCache), else made here. Its
      // kernels go on `stream` before the hop scope opens (the hopped kernel waits only for work
      // recorded before the scope).
      const uint32_t* perm = (accumInl() && recAccumSort() && workCycles > 1)
                                 ? recAccumPerm(stream, ctrlBuf, steps, workCycles)
                                 : nullptr;
      // ~1 ms per lift/join: on a side hardware queue (hip_hop.h).
      risc0_hip_hop::Scope s(stream, risc0_hip_hop::want(risc0_hip_hop::kRecStep, size_t(cfg.grid.x) * cfg.block.x, 16));
      if (perm)
        risc0_recursion_launch_accum_inl_perm(0, cfg.grid.x, cfg.block.x, s, ctx, perm);
      else if (accumInl())
        risc0_recursion_launch_accum_inl(0, cfg.grid.x, cfg.block.x, s, ctx);
      else
        parStepComputeAccum<<<cfg.grid, cfg.block, 0, s>>>(ctx);
    }
#else
    if (accumInl()) {
      risc0_recursion_launch_accum_inl(0, cfg.grid.x, cfg.block.x, stream, ctx);
    } else
    parStepComputeAccum<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
#endif
    if (syncSteps()) {
      CUDA_OK(cudaStreamSynchronize(stream));
    } else {
      CUDA_OK(cudaGetLastError());
    }
  }

  void calcPrefixProducts() {
    nvtx3::scoped_range range("calcPrefixProducts");
#if defined(SPPARK_NO_COOPERATIVE_LAUNCH) || defined(__HIPCC__)
#ifdef SPPARK_NO_COOPERATIVE_LAUNCH
    // sppark's prefix_op needs a cooperative launch, which this platform has not (see sppark's
    // util/cuda2hip.hpp): a three-kernel scan instead (see prefixProductsChunks).
    const bool ownScan = true;
#else
    // On ROCm sppark's prefix_op is a cooperative launch on sppark's own stream, between host
    // waits for the context's stream and for sppark's (~0.7 ms of idle GPU per proof on an
    // RX 9070 XT). With the async path, the same products by the three-kernel scan, on the
    // context's stream.
    const bool ownScan = recAsync();
#endif
    if (ownScan) {
      // The host copies of ctx->workCycles and ctx->accum (managed memory, see syncSteps).
      uint32_t n = workCycles;
      FpExt* io = g_accumCache.accum;
      uint32_t nblocks = (n + kScanChunk - 1) / kScanChunk;
      static thread_local FpExt* totals = nullptr;
      static thread_local uint32_t totalsCap = 0;
      if (nblocks > totalsCap) {
        if (totals)
          CUDA_OK(cudaFree(totals));
        CUDA_OK(cudaMalloc(&totals, nblocks * sizeof(FpExt)));
        totalsCap = nblocks;
      }
      prefixProductsChunks<<<nblocks, kScanThreads, 0, stream>>>(io, n, totals);
      if (nblocks > 1) {
        prefixProductsTotals<<<1, 1, 0, stream>>>(totals, nblocks);
        uint32_t rest = n - kScanChunk;
        prefixProductsFixup<<<(rest + 255) / 256, 256, 0, stream>>>(io, n, totals);
      }
      CUDA_OK(cudaGetLastError());
      if (syncSteps())
        CUDA_OK(cudaStreamSynchronize(stream));
      return;
    }
#endif
    sppark::calcPrefixProducts(ctx->accum, ctx->workCycles);
    CUDA_OK(cudaStreamSynchronize(stream));
  }

  void verifyAccum() {
    nvtx3::scoped_range range("verifyAccum");
    if (syncSteps())
      CUDA_OK(cudaDeviceSynchronize());
#ifdef __HIPCC__
    {
      // The witgen's verifyWom permutation when it matches (RecPermCache), else made here. Its
      // kernels go on `stream` before the hop scope opens (the hopped kernel waits only for work
      // recorded before the scope).
      const uint32_t* perm = (accumInl() && recAccumSort() && workCycles > 1)
                                 ? recAccumPerm(stream, ctrlBuf, steps, workCycles)
                                 : nullptr;
      // ~1 ms per lift/join, then the host waits: on a side hardware queue (hip_hop.h).
      risc0_hip_hop::Scope s(stream, risc0_hip_hop::want(risc0_hip_hop::kRecStep, size_t(cfg.grid.x) * cfg.block.x, 16));
      if (perm)
        risc0_recursion_launch_accum_inl_perm(1, cfg.grid.x, cfg.block.x, s, ctx, perm);
      else if (accumInl())
        risc0_recursion_launch_accum_inl(1, cfg.grid.x, cfg.block.x, s, ctx);
      else
        parStepVerifyAccum<<<cfg.grid, cfg.block, 0, s>>>(ctx);
    }
#else
    if (accumInl()) {
      risc0_recursion_launch_accum_inl(1, cfg.grid.x, cfg.block.x, stream, ctx);
    } else
    parStepVerifyAccum<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
#endif
    CUDA_OK(cudaStreamSynchronize(stream));
  }
};

extern "C" {

const char* risc0_circuit_recursion_cuda_witgen_chains(uint32_t mode,
                                                       ExecBuffers* buffers,
                                                       PreflightTrace* trace,
                                                       uint32_t totalCycles,
                                                       const HostChains* hostChains);

const char* risc0_circuit_recursion_cuda_witgen(uint32_t mode,
                                                ExecBuffers* buffers,
                                                PreflightTrace* trace,
                                                uint32_t totalCycles) {
  return risc0_circuit_recursion_cuda_witgen_chains(mode, buffers, trace, totalCycles, nullptr);
}

const char* risc0_circuit_recursion_cuda_witgen_chains(uint32_t mode,
                                                       ExecBuffers* buffers,
                                                       PreflightTrace* trace,
                                                       uint32_t totalCycles,
                                                       const HostChains* hostChains) {
  try {
    // Sync all GPU work before accessing buffers written by external streams
    // (e.g., Rust HAL's persistent stream). hipMemcpy doesn't reliably
    // synchronize across all streams on ROCm.
    CUDA_OK(cudaDeviceSynchronize());
#ifdef __HIPCC__
    StageClaim claim(stageClaim(trace));
    HostExecContext ctx(buffers, trace, totalCycles, claim.s);
#else
    HostExecContext ctx(buffers, trace, totalCycles);
#endif
    if (hostChains && hostChains->numHeads && mode == kStepModeParallel) {
      ctx.applyHostChains(hostChains);
    }
    ctx.doStepExec(mode);
    ctx.verifyWom(mode);
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Load the recursion witgen/accum kernels (CUDA lazy loading) off the GPU thread, without
// launching anything: cudaFuncGetAttributes forces the module load.
const char* risc0_circuit_recursion_cuda_warmup() {
  try {
    cudaFuncAttributes attr;
    CUDA_OK(cudaFuncGetAttributes(&attr, parStepExec));
    CUDA_OK(cudaFuncGetAttributes(&attr, injectWomBacks));
    CUDA_OK(cudaFuncGetAttributes(&attr, parStepVerifyWom));
    CUDA_OK(cudaFuncGetAttributes(&attr, markHostChains));
    CUDA_OK(cudaFuncGetAttributes(&attr, scatterHostChains));
    CUDA_OK(cudaFuncGetAttributes(&attr, parStepComputeAccum));
    CUDA_OK(cudaFuncGetAttributes(&attr, parStepVerifyAccum));
#ifdef __HIPCC__
    CUDA_OK(cudaFuncGetAttributes(&attr, recExecKeys));
    CUDA_OK(cudaFuncGetAttributes(&attr, parStepExecPerm));
    CUDA_OK(cudaFuncGetAttributes(&attr, recAccumKeys));
    CUDA_OK(cudaFuncGetAttributes(&attr, recWomKeys));
    CUDA_OK(cudaFuncGetAttributes(&attr, recWomGather));
    CUDA_OK(cudaFuncGetAttributes(&attr, recWomFix));
    CUDA_OK(cudaFuncGetAttributes(&attr, parStepVerifyWomPerm));
#endif
#ifdef REC_ASYNC_AVAILABLE
    if (recAsync()) {
      // The first parStepExec launch paid ~4 ms growing the context's local-memory
      // reservation on the GPU thread (its stack frame exceeds every earlier kernel's); grow it
      // here, at process start, instead. For the current device, the HAL's (device 0); on any
      // other the driver still grows it at the first launch, as before.
      size_t need = 0;
      for (const void* f : {(const void*)parStepExec, (const void*)injectWomBacks,
                            (const void*)parStepVerifyWom, (const void*)parStepComputeAccum,
                            (const void*)parStepVerifyAccum}) {
        CUDA_OK(cudaFuncGetAttributes(&attr, f));
        need = attr.localSizeBytes > need ? attr.localSizeBytes : need;
      }
      size_t more = risc0_recursion_accum_inl_warmup();
      need = more > need ? more : need;
      if (const char* e = rec_eval_check_split_warmup(&more)) return e;
      need = more > need ? more : need;
#ifndef __HIPCC__
      size_t cur = 0;
      CUDA_OK(cudaDeviceGetLimit(&cur, cudaLimitStackSize));
      if (need > cur) CUDA_OK(cudaDeviceSetLimit(cudaLimitStackSize, need));
#else
      // Nothing to pre-grow: ROCm sizes a dispatch's scratch from its kernel's private segment
      // (hipLimitStackSize only bounds the dynamic stack), so the loads above are the warmup.
      (void)need;
#endif
    }
#endif
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Page-locked host memory for the recursion witgen inputs (prepared off the GPU thread), backed
// by huge pages (ThpPinned): cudaHostAlloc pins 4 KB pages while holding the driver lock, which
// stalls the proving thread's CUDA calls.
static std::mutex g_host_bufs_mu;
static std::unordered_map<void*, ThpPinned>& hostBufs() {
  static std::unordered_map<void*, ThpPinned> m;
  return m;
}

const char* risc0_circuit_recursion_cuda_host_alloc(size_t bytes, void** out) {
  *out = nullptr;
  try {
    ThpPinned p;
    p.alloc(bytes);
    std::lock_guard<std::mutex> lk(g_host_bufs_mu);
    hostBufs()[p.ptr] = p;
    *out = p.ptr;
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_circuit_recursion_cuda_host_free(void* ptr) {
  try {
    ThpPinned p;
    {
      std::lock_guard<std::mutex> lk(g_host_bufs_mu);
      auto it = hostBufs().find(ptr);
      if (it == hostBufs().end()) throw std::runtime_error("host_free: unknown pointer");
      p = it->second;
      hostBufs().erase(it);
    }
    p.free_pinned(); // synchronizes the device first, like cudaFreeHost
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

#ifdef __HIPCC__
// Upload a prepared preflight's page-locked inputs (and host chains, if any) ahead of its witgen;
// see StagedInputs. Called on the preparing thread; does nothing if both slots are taken. The
// host arrays must stay alive until risc0_circuit_recursion_hip_unstage(trace->wom) returns.
const char* risc0_circuit_recursion_hip_stage(const PreflightTrace* trace, const HostChains* hc) {
  StagedInputs* s = nullptr;
  try {
    if (!recStage()) return nullptr;
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lk(g_stageMu);
    for (auto& x : g_stage) {
      if (!x.valid && !x.busy && (x.device < 0 || x.device == device)) {
        s = &x;
        break;
      }
    }
    if (!s) return nullptr;
    s->device = device;
    if (!s->stream) s->stream = stageStream(device);
    if (!s->done) CUDA_OK(cudaEventCreateWithFlags(&s->done, cudaEventDisableTiming));
    stageGrow(s->dWom, s->womCap, trace->numWoms);
    stageGrow(s->dCycles, s->cyclesCap, trace->numCycles);
    stageGrow(s->dIops, s->iopsCap, trace->numIops);
    CUDA_OK(cudaMemcpyAsync(s->dWom, trace->wom, trace->numWoms * sizeof(FpExt),
                            cudaMemcpyHostToDevice, s->stream));
    CUDA_OK(cudaMemcpyAsync(s->dCycles, trace->cycles, trace->numCycles * sizeof(PreflightCycle),
                            cudaMemcpyHostToDevice, s->stream));
    CUDA_OK(cudaMemcpyAsync(s->dIops, trace->iops, trace->numIops * sizeof(FpExt),
                            cudaMemcpyHostToDevice, s->stream));
    s->heads = nullptr;
    s->numHeads = s->numRows = s->dataCols = 0;
    if (hc && hc->numHeads && hc->dataCols == 128) {
      size_t nWords = hc->numHeads + (size_t)hc->numRows * (2 + hc->dataCols + kMaxWomRowsPerCycle * 5);
      stageGrow(s->dHc, s->hcCap, nWords);
      uint32_t* dHeads = s->dHc;
      uint32_t* dCycles = dHeads + hc->numHeads;
      uint32_t* dRows = dCycles + hc->numRows;
      uint32_t* dWomIdx = dRows + (size_t)hc->numRows * hc->dataCols;
      uint32_t* dWomRows = dWomIdx + hc->numRows;
      CUDA_OK(cudaMemcpyAsync(dHeads, hc->heads, hc->numHeads * 4, cudaMemcpyHostToDevice, s->stream));
      CUDA_OK(cudaMemcpyAsync(dCycles, hc->cycles, hc->numRows * 4, cudaMemcpyHostToDevice, s->stream));
      CUDA_OK(cudaMemcpyAsync(dRows, hc->rows, (size_t)hc->numRows * hc->dataCols * 4, cudaMemcpyHostToDevice, s->stream));
      CUDA_OK(cudaMemcpyAsync(dWomIdx, hc->womIdx, hc->numRows * 4, cudaMemcpyHostToDevice, s->stream));
      CUDA_OK(cudaMemcpyAsync(dWomRows, hc->womRows, (size_t)hc->numRows * kMaxWomRowsPerCycle * 20, cudaMemcpyHostToDevice, s->stream));
      s->heads = hc->heads;
      s->numHeads = hc->numHeads;
      s->numRows = hc->numRows;
      s->dataCols = hc->dataCols;
    }
    CUDA_OK(cudaEventRecord(s->done, s->stream));
    s->wom = trace->wom;
    s->numWoms = trace->numWoms;
    s->numCycles = trace->numCycles;
    s->numIops = trace->numIops;
    s->valid = true;
  } catch (const std::exception& err) {
    // Not staged: let no copy queued above outlive the call (they read the caller's buffers).
    if (s && s->stream) cudaStreamSynchronize(s->stream);
    return strdup(err.what());
  }
  return nullptr;
}

// Witness buffer passes for the recursion witgen (prove/witgen.rs, HIP; RISC0_HIP_REC_INIT=0 keeps
// the generic HAL passes). A buffer is column-major with 2^logRows rows per column, and the
// witgen writes only rows below `live` (its cycle count). The generic path fills the whole buffer
// with INVALID, uploads the ZK noise (one random element repeated: `vec![x; n]`) and copies it
// into the last rows of every column with eltwise_copy_fp_region (one thread per column walking
// its rows: ~125 us for the data group on an RX 7900 XTX, plus the host-side upload), then
// zeroizes the whole buffer. Equivalent result here: rows >= live start as zero instead of
// INVALID (nothing writes them before the zeroize would have), the noise is a fill kernel, and
// the zeroize only reads rows < live. All on the risc0 persistent stream, like the HAL passes.
__global__ void recInitRows(uint32_t* buf, uint32_t logRows, uint32_t live, uint64_t total) {
  uint64_t i = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (i >= total) return;
  uint32_t row = uint32_t(i & ((uint64_t(1) << logRows) - 1));
  constexpr uint32_t kInv = 0xffffffffu; // Fp::INVALID
  if (row + 4 <= live) {
    *reinterpret_cast<uint4*>(buf + i) = make_uint4(kInv, kInv, kInv, kInv);
  } else if (row >= live) {
    *reinterpret_cast<uint4*>(buf + i) = make_uint4(0, 0, 0, 0);
  } else {
    for (uint32_t k = 0; k < 4; k++)
      buf[i + k] = row + k < live ? kInv : 0;
  }
}

__global__ void recFillTail(uint32_t* buf, uint32_t value, uint32_t logRows, uint32_t zk, uint32_t cols) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= zk * cols) return;
  uint32_t col = t / zk, r = t % zk;
  buf[(size_t(col) << logRows) + (size_t(1) << logRows) - zk + r] = value;
}

__global__ void recZeroizeRows(uint32_t* buf, uint32_t logRows, uint32_t live) {
  uint32_t row = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (row >= live) return;
  uint32_t* p = buf + (size_t(blockIdx.y) << logRows) + row;
  constexpr uint32_t kInv = 0xffffffffu;
  if (row + 4 <= live) {
    uint4 v = *reinterpret_cast<const uint4*>(p);
    if (v.x == kInv || v.y == kInv || v.z == kInv || v.w == kInv) {
      v.x = v.x == kInv ? 0 : v.x;
      v.y = v.y == kInv ? 0 : v.y;
      v.z = v.z == kInv ? 0 : v.z;
      v.w = v.w == kInv ? 0 : v.w;
      *reinterpret_cast<uint4*>(p) = v;
    }
  } else {
    for (uint32_t k = 0; row + k < live; k++)
      if (p[k] == kInv) p[k] = 0;
  }
}

const char* risc0_circuit_recursion_hip_init_rows(uint32_t* buf, uint32_t logRows, uint32_t live,
                                                  uint32_t cols) {
  try {
    uint64_t total = uint64_t(cols) << logRows;
    if (logRows < 2) throw std::runtime_error("recursion init_rows: rows < 4");
    uint64_t threads = total / 4;
    recInitRows<<<uint32_t((threads + 255) / 256), 256, 0, getPersistentStream()>>>(buf, logRows, live, total);
    CUDA_OK(cudaGetLastError());
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

const char* risc0_circuit_recursion_hip_fill_tail(uint32_t* buf, uint32_t value, uint32_t logRows,
                                                  uint32_t zk, uint32_t cols) {
  try {
    if (zk > (1u << logRows)) throw std::runtime_error("recursion fill_tail: zk > rows");
    uint32_t n = zk * cols;
    if (n) {
      recFillTail<<<(n + 255) / 256, 256, 0, getPersistentStream()>>>(buf, value, logRows, zk, cols);
      CUDA_OK(cudaGetLastError());
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

const char* risc0_circuit_recursion_hip_zeroize_rows(uint32_t* buf, uint32_t logRows, uint32_t live,
                                                     uint32_t cols) {
  try {
    if (live && cols) {
      dim3 grid((live / 4 + 255) / 256 + 1, cols);
      recZeroizeRows<<<grid, 256, 0, getPersistentStream()>>>(buf, logRows, live);
      CUDA_OK(cudaGetLastError());
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Forget the inputs staged from the page-locked buffer at `wom` if no witgen claimed them, after
// their copies are done (before the buffer is reused or freed).
const char* risc0_circuit_recursion_hip_unstage(const void* wom) {
  try {
    std::lock_guard<std::mutex> lk(g_stageMu);
    for (auto& s : g_stage) {
      if (s.valid && s.wom == wom) {
        s.valid = false;
        s.wom = nullptr;
        s.heads = nullptr;
        CUDA_OK(cudaEventSynchronize(s.done));
      }
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}
#endif

const char* risc0_circuit_recursion_cuda_accum(AccumBuffers* buffers,
                                               uint32_t workCycles,
                                               uint32_t totalCycles) {
  try {
    // Sync all GPU work before accessing buffers written by external streams
    CUDA_OK(cudaDeviceSynchronize());
    HostAccumContext ctx(buffers, workCycles, totalCycles);
    ctx.computeAccum();
    ctx.calcPrefixProducts();
    ctx.verifyAccum();
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

} // extern "C"
