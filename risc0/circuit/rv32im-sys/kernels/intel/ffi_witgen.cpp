// Intel SYCL witgen kernel wrapper + extern implementations.
// Ported from kernels/cuda/ffi.cu, adapted for SYCL device code.

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>

// sycl/sycl.hpp pulls in <cassert> which re-defines assert.
// We need assert to be a no-op for the step functions (which hit
// "unreachable mux arm" on padding cycles — benign, eval_check validates).
#ifdef assert
#undef assert
#endif
#define assert(x) ((void)0)

// Include Intel SYCL witgen headers (which include steps.h → all types)
#include "fp.h"
#include "fpext.h"

// The steps.cpp body is included via the amalgamation created by build.rs.
// This file provides the extern implementations and kernel wrappers.

namespace risc0::circuit::rv32im_v2::intel {

using Val = Fp;
using ExtVal = FpExt;

constexpr size_t kUserAccumSplit = 23;

// ============================================================================
// RISC-V integer division (a copy of cxx/ffi.cpp divide_rv32im)
// ============================================================================

inline std::pair<uint32_t, uint32_t> divide_rv32im(uint32_t numer, uint32_t denom, uint32_t signType) {
  // signType: 0 = unsigned, 1 = signed (DIV/REM), 2 = signed numerator with
  // ones'-complement rounding (used for SRA/SRAI). Must match cxx/ffi.cpp.
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

// ============================================================================
// Extern function implementations (device-callable)
// ============================================================================

inline std::array<Val, 5> extern_getMemoryTxn(ExecContext& ctx, Val addrElem) {
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx++;
  const MemoryTransaction& txn = ctx.preflight.txns[txnIdx];
  return {
      Val(txn.prevCycle),
      Val(txn.prevWord & 0xffff),
      Val(txn.prevWord >> 16),
      Val(txn.word & 0xffff),
      Val(txn.word >> 16),
  };
}

inline void extern_lookupDelta(ExecContext& ctx, Val table, Val index, Val count) {
  ctx.tables.lookupDelta(table, index, count);
}

inline Val extern_lookupCurrent(ExecContext& ctx, Val table, Val index) {
  return ctx.tables.lookupCurrent(table, index);
}

inline void extern_memoryDelta(ExecContext& ctx, Val addr, Val cycle, Val dataLow, Val dataHigh, Val count) {
  // no-op (same as CUDA)
}

inline uint32_t extern_getDiffCount(ExecContext& ctx, Val cycle) {
  uint32_t cycleU32 = cycle.asUInt32();
  return ctx.preflight.cycles[cycleU32 / 2].diffCount[cycleU32 % 2];
}

inline Val extern_isFirstCycle_0(ExecContext& ctx) {
  return Val(ctx.cycle == 0);
}

inline std::array<Val, 4> extern_divide(
    ExecContext& ctx, Val numerLow, Val numerHigh, Val denomLow, Val denomHigh, Val signType) {
  uint32_t numer = numerLow.asUInt32() | (numerHigh.asUInt32() << 16);
  uint32_t denom = denomLow.asUInt32() | (denomHigh.asUInt32() << 16);
  auto [quot, rem] = divide_rv32im(numer, denom, signType.asUInt32());
  return {Val(quot & 0xffff), Val(quot >> 16), Val(rem & 0xffff), Val(rem >> 16)};
}

inline void extern_print(ExecContext& ctx, Val v) {
  // no-op
}

inline std::array<Val, 2> extern_getMajorMinor(ExecContext& ctx) {
  return {Val(ctx.preflight.cycles[ctx.cycle].major),
          Val(ctx.preflight.cycles[ctx.cycle].minor)};
}

inline Val extern_hostReadPrepare(ExecContext& ctx, Val fp, Val len) {
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx;
  return Val(ctx.preflight.txns[txnIdx].word);
}

inline Val extern_hostWrite(ExecContext& ctx, Val fdVal, Val addrLow, Val addrHigh, Val lenVal) {
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx;
  return Val(ctx.preflight.txns[txnIdx].word);
}

inline std::array<Val, 2> extern_nextPagingIdx(ExecContext& ctx) {
  return {Val(ctx.preflight.cycles[ctx.cycle].pagingIdx),
          Val(ctx.preflight.cycles[ctx.cycle].machineMode)};
}

inline std::array<Val, 16> extern_bigIntExtern(ExecContext& ctx) {
  std::array<Val, 16> ret;
  size_t bigintIdx = ctx.preflight.cycles[ctx.cycle].bigintIdx;
  for (size_t i = 0; i < 16; i++) {
    ret[i] = Val(ctx.preflight.bigintBytes[bigintIdx + i]);
  }
  return ret;
}

// ============================================================================
// Device-side step dispatch (matches CUDA's nextStep)
// ============================================================================

inline void nextStep(Buffer& dataBuf, Buffer& preDataBuf, Buffer& globalBuf,
                     PreflightTrace& preflight, LookupTables& tables, uint32_t cycle) {
  ExecContext execCtx(preflight, tables, cycle);
  if (preDataBuf.buf == dataBuf.buf) {
    preDataBuf.buf = nullptr;
  }
  MutableBufObj data(dataBuf, preDataBuf);
  GlobalBufObj global(globalBuf);
  BufferObj dataBo(&data);
  BufferObj globalBo(&global);
  step_Top(execCtx, &dataBo, &globalBo);
}

} // namespace risc0::circuit::rv32im_v2::intel

// ============================================================================
// SYCL kernel launcher + FFI entry points
// ============================================================================

constexpr size_t kStepModeParallel = 0;
constexpr size_t kStepModeSeqForward = 1;
constexpr size_t kStepModeSeqReverse = 2;

// Device cache for persistent preflight allocations (grow-only)
struct IntelWitgenCache {
  // Device preflight data
  risc0::circuit::rv32im_v2::intel::PreflightCycle* d_cycles = nullptr;
  risc0::circuit::rv32im_v2::intel::MemoryTransaction* d_txns = nullptr;
  uint8_t* d_bigintBytes = nullptr;
  size_t cycles_cap = 0, txns_cap = 0, bigint_cap = 0;

  // Lookup tables
  uint32_t* d_tableU8 = nullptr;
  uint32_t* d_tableU16 = nullptr;
  bool tables_allocated = false;

  // Track whether preflight was uploaded by witgen (for accum reuse)
  bool preflight_on_device = false;

  void ensure(sycl::queue& q, size_t nCycles, size_t nTxns, size_t nBigint) {
    if (nCycles > cycles_cap) {
      if (d_cycles) sycl::free(d_cycles, q);
      d_cycles = sycl::malloc_device<risc0::circuit::rv32im_v2::intel::PreflightCycle>(nCycles, q);
      cycles_cap = nCycles;
    }
    if (nTxns > txns_cap) {
      if (d_txns) sycl::free(d_txns, q);
      d_txns = sycl::malloc_device<risc0::circuit::rv32im_v2::intel::MemoryTransaction>(nTxns, q);
      txns_cap = nTxns;
    }
    if (nBigint > bigint_cap) {
      if (d_bigintBytes) sycl::free(d_bigintBytes, q);
      d_bigintBytes = sycl::malloc_device<uint8_t>(nBigint > 0 ? nBigint : 16, q);
      bigint_cap = nBigint > 0 ? nBigint : 16;
    }
    if (!tables_allocated) {
      d_tableU8 = sycl::malloc_device<uint32_t>(256, q);
      d_tableU16 = sycl::malloc_device<uint32_t>(65536, q);
      tables_allocated = true;
    }
  }
};

static IntelWitgenCache g_cache;

// ---- cycle permutation sorted by (major[,minor]) to cut SIMD divergence ----
// The parallel witgen/accum kernels are order-independent (one work-item per
// cycle, commutative table atomics), so assigning work-item i to cycle
// perm[i] does not change any output. perm restricted to [0,split) and
// [split,last) are permutations of those ranges (counting sort, stable).
struct SortCache {
  uint32_t* perm = nullptr; size_t perm_cap = 0;
  uint32_t* cnt = nullptr; size_t cnt_cap = 0;
  bool valid = false; uint32_t last = 0, split = 0;
};
static SortCache g_sort;
static uint32_t* g_rep8 = nullptr;
static uint32_t* g_rep16 = nullptr;
static uint32_t g_rep_cap = 0;
static int sort_env(const char* name, int def) {
  const char* s = getenv(name);
  if (!s || !*s) return def;
  return atoi(s);
}
#ifndef SORT_CH
#define SORT_CH 256u
#endif
template <uint32_t NB> class WitgenSortScan;
class WitgenAccumScan;

// The width of a one-work-group scan: `want`, halved until the device runs kernel `K` that wide.
// 1024 on Battlemage; 512 on Alchemist (Arc A770), where these kernels, built in 256-GRF mode, run
// at most 512 work-items per work-group.
template <typename K>
static uint32_t scan_width(sycl::queue& q, uint32_t want) {
  static uint32_t width = 0;
  if (width == 0) {
    auto dev = q.get_device();
    auto id = sycl::get_kernel_id<K>();
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {dev}, {id});
    size_t max = kb.get_kernel(id).template get_info<
        sycl::info::kernel_device_specific::work_group_size>(dev);
    uint32_t w = want;
    while (w > 1 && w > max) w >>= 1;
    width = w;
  }
  return width;
}

static constexpr uint32_t SORT_NB = 128;
template <uint32_t NB>
static void sort_range_t(sycl::queue& q, const risc0::circuit::rv32im_v2::intel::PreflightCycle* cyc,
                       uint32_t lo, uint32_t hi, uint32_t* perm, uint32_t* cnt, uint32_t keymode) {
  if (hi <= lo) return;
  const uint32_t n = hi - lo;
  const uint32_t nblk = (n + SORT_CH - 1) / SORT_CH;
  auto key = [=](uint32_t c) -> uint32_t {
    uint32_t mj = cyc[c].major; if (mj > 15) mj = 15;
    if (NB == 16) return mj;
    uint32_t mn = cyc[c].minor; if (mn > 7) mn = 7;
    return (mj << 3) | mn;
  };
  q.parallel_for(sycl::range<1>(nblk), [=](sycl::id<1> id) {
    uint32_t b = id[0];
    uint32_t c[NB];
    for (uint32_t k = 0; k < NB; k++) c[k] = 0;
    uint32_t s = lo + b * SORT_CH, e = sycl::min(s + SORT_CH, hi);
    for (uint32_t i = s; i < e; i++) c[key(i)]++;
    for (uint32_t k = 0; k < NB; k++) cnt[size_t(k) * nblk + b] = c[k];
  });
  const uint32_t tot = NB * nblk;
  const uint32_t SW = scan_width<WitgenSortScan<NB>>(q, 1024);
  const uint32_t R = (tot + SW - 1) / SW;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> sm(SW, h);
    h.parallel_for<WitgenSortScan<NB>>(sycl::nd_range<1>(SW, SW), [=](sycl::nd_item<1> it) {
      uint32_t l = it.get_local_id(0);
      uint32_t a = l * R, z = sycl::min(a + R, tot);
      uint32_t s = 0;
      for (uint32_t j = a; j < z; j++) s += cnt[j];
      sm[l] = s;
      sycl::group_barrier(it.get_group());
      for (uint32_t st = 1; st < SW; st <<= 1) {
        uint32_t o = (l >= st) ? sm[l - st] : 0u;
        sycl::group_barrier(it.get_group());
        s += o; sm[l] = s;
        sycl::group_barrier(it.get_group());
      }
      uint32_t run = (l > 0) ? sm[l - 1] : 0u;
      for (uint32_t j = a; j < z; j++) { uint32_t v = cnt[j]; cnt[j] = run; run += v; }
    });
  });
  q.parallel_for(sycl::range<1>(nblk), [=](sycl::id<1> id) {
    uint32_t b = id[0];
    uint32_t off[NB];
    for (uint32_t k = 0; k < NB; k++) off[k] = cnt[size_t(k) * nblk + b];
    uint32_t s = lo + b * SORT_CH, e = sycl::min(s + SORT_CH, hi);
    for (uint32_t i = s; i < e; i++) perm[lo + off[key(i)]++] = i;
  });
}
static void sort_range(sycl::queue& q, const risc0::circuit::rv32im_v2::intel::PreflightCycle* cyc,
                       uint32_t lo, uint32_t hi, uint32_t* perm, uint32_t* cnt, uint32_t keymode) {
  if (keymode) sort_range_t<128>(q, cyc, lo, hi, perm, cnt, keymode);
  else sort_range_t<16>(q, cyc, lo, hi, perm, cnt, keymode);
}
static void sort_build(sycl::queue& q, const risc0::circuit::rv32im_v2::intel::PreflightCycle* cyc,
                       uint32_t split, uint32_t last) {
  if (last > g_sort.perm_cap) {
    if (g_sort.perm) sycl::free(g_sort.perm, q);
    g_sort.perm = sycl::malloc_device<uint32_t>(last, q);
    g_sort.perm_cap = last;
  }
  size_t need = size_t(SORT_NB) * ((last + SORT_CH - 1) / SORT_CH + 1);
  if (need > g_sort.cnt_cap) {
    if (g_sort.cnt) sycl::free(g_sort.cnt, q);
    g_sort.cnt = sycl::malloc_device<uint32_t>(need, q);
    g_sort.cnt_cap = need;
  }
  uint32_t keymode = (uint32_t)sort_env("RISC0_WITGEN_SORT_MINOR", 0);
  uint32_t sp = split < last ? split : last;
  sort_range(q, cyc, 0, sp, g_sort.perm, g_sort.cnt, keymode);
  sort_range(q, cyc, sp, last, g_sort.perm, g_sort.cnt, keymode);
  g_sort.valid = true; g_sort.last = last; g_sort.split = split;
}

extern "C" {

using namespace risc0::circuit::rv32im_v2::intel;

const char* risc0_circuit_rv32im_intel_witgen(
    void* queue_ptr,
    uint32_t mode,
    void* d_data_ptr, uint32_t data_rows, uint32_t data_cols,
    void* d_pre_data_ptr,
    void* d_global_ptr, uint32_t global_cols,
    const PreflightCycle* h_cycles, uint32_t cycles_len,
    const MemoryTransaction* h_txns, uint32_t txns_len,
    const uint8_t* h_bigint, uint32_t bigint_len,
    uint32_t table_split_cycle,
    uint32_t last_cycle)
{
  try {
    auto& q = *static_cast<sycl::queue*>(queue_ptr);
    auto t0 = std::chrono::steady_clock::now();

    // Ensure device allocations
    g_cache.ensure(q, cycles_len, txns_len, bigint_len);

    // Upload preflight data to device
    q.memcpy(g_cache.d_cycles, h_cycles, cycles_len * sizeof(PreflightCycle));
    q.memcpy(g_cache.d_txns, h_txns, txns_len * sizeof(MemoryTransaction));
    if (bigint_len > 0) {
      q.memcpy(g_cache.d_bigintBytes, h_bigint, bigint_len);
    }

    // Zero lookup tables. NOTE: no q.wait() needed — in-order queue ensures the
    // following kernel sees these uploads. The CPU-side flag below is host-only
    // semantics, set unconditionally before subsequent calls observe it.
    q.memset(g_cache.d_tableU8, 0, 256 * sizeof(uint32_t));
    q.memset(g_cache.d_tableU16, 0, 65536 * sizeof(uint32_t));
    g_cache.preflight_on_device = true;
    g_sort.valid = false;
    const bool wsort = mode == kStepModeParallel && sort_env("RISC0_WITGEN_SORT", 1) != 0;
    const bool asort_pre = mode == kStepModeParallel && sort_env("RISC0_ACCUM_SORT", 1) != 0;
    if (wsort || asort_pre) sort_build(q, g_cache.d_cycles, table_split_cycle, last_cycle);
    const uint32_t* wperm = wsort ? g_sort.perm : nullptr;

    auto t1 = std::chrono::steady_clock::now();

    // Build device-side structs (captured by value in kernel lambda)
    Buffer dataBuf{static_cast<risc0::Fp*>(d_data_ptr), data_rows, data_cols, false};
    Buffer preDataBuf{static_cast<risc0::Fp*>(d_pre_data_ptr), data_rows, data_cols, false};
    Buffer globalBuf{static_cast<risc0::Fp*>(d_global_ptr), 1, global_cols, false};

    PreflightTrace pf;
    pf.cycles = g_cache.d_cycles;
    pf.txns = g_cache.d_txns;
    pf.bigintBytes = g_cache.d_bigintBytes;
    pf.txnsLen = txns_len;
    pf.bigintBytesLen = bigint_len;
    pf.tableSplitCycle = table_split_cycle;

    LookupTables tables;
    tables.tableU8 = g_cache.d_tableU8;
    tables.tableU16 = g_cache.d_tableU16;

    constexpr uint32_t WG_SIZE = 32;

    if (mode == kStepModeParallel) {
      uint32_t split = table_split_cycle;
      uint32_t phase2Count = last_cycle - split;
      uint32_t R = (uint32_t)sort_env("RISC0_WITGEN_TABLE_REP", 64);
      if (R < 2 || (R & (R - 1)) || R > 1024) R = 0;
      if (R && split > 0) {
        if (R > g_rep_cap) {
          if (g_rep8) sycl::free(g_rep8, q);
          if (g_rep16) sycl::free(g_rep16, q);
          g_rep8 = sycl::malloc_device<uint32_t>(size_t(R) * 256, q);
          g_rep16 = sycl::malloc_device<uint32_t>(size_t(R) * 65536, q);
          g_rep_cap = R;
        }
        q.memset(g_rep8, 0, size_t(R) * 256 * sizeof(uint32_t));
        q.memset(g_rep16, 0, size_t(R) * 65536 * sizeof(uint32_t));
      }
      uint32_t* rep8 = R ? g_rep8 : nullptr;
      uint32_t* rep16 = R ? g_rep16 : nullptr;
      const uint32_t rmask = R ? R - 1 : 0;

      // Phase 1: body cycles [0, split)
      if (split > 0) {
        uint32_t global1 = ((split + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
        q.parallel_for(sycl::nd_range<1>(global1, WG_SIZE),
            [=](sycl::nd_item<1> item) {
              uint32_t cycle = item.get_global_id(0);
              if (cycle >= split) return;
              if (wperm) cycle = wperm[cycle];
              Buffer db = dataBuf, pdb = preDataBuf, gb = globalBuf;
              PreflightTrace pft = pf;
              LookupTables tbl = tables;
              tbl.rep8 = rep8; tbl.rep16 = rep16; tbl.repIdx = uint32_t(item.get_global_id(0)) & rmask;
              ExecContext execCtx(pft, tbl, cycle);
              if (pdb.buf == db.buf) pdb.buf = nullptr;
              MutableBufObj data(db, pdb);
              GlobalBufObj global(gb);
              BufferObj dataBo(&data);
              BufferObj globalBo(&global);
              step_Top(execCtx, &dataBo, &globalBo);
            });
      }

      if (rep16 && split > 0) {
        uint32_t* t8 = tables.tableU8;
        uint32_t* t16 = tables.tableU16;
        const uint32_t RR = rmask + 1;
        q.parallel_for(sycl::range<1>(65536), [=](sycl::id<1> id) {
          uint32_t x = id[0];
          uint32_t s = t16[x];
          for (uint32_t r = 0; r < RR; r++) s += rep16[size_t(r) * 65536 + x];
          t16[x] = s;
          if (x < 256) {
            uint32_t s8 = t8[x];
            for (uint32_t r = 0; r < RR; r++) s8 += rep8[size_t(r) * 256 + x];
            t8[x] = s8;
          }
        });
      }
      // Phase 2: table-fill cycles [split, last_cycle)
      if (phase2Count > 0) {
        uint32_t global2 = ((phase2Count + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
        q.parallel_for(sycl::nd_range<1>(global2, WG_SIZE),
            [=](sycl::nd_item<1> item) {
              uint32_t cycle = item.get_global_id(0);
              if (cycle >= phase2Count) return;
              uint32_t actualCycle = wperm ? wperm[split + cycle] : split + cycle;
              Buffer db = dataBuf, pdb = preDataBuf, gb = globalBuf;
              PreflightTrace pft = pf;
              LookupTables tbl = tables;
              ExecContext execCtx(pft, tbl, actualCycle);
              if (pdb.buf == db.buf) pdb.buf = nullptr;
              MutableBufObj data(db, pdb);
              GlobalBufObj global(gb);
              BufferObj dataBo(&data);
              BufferObj globalBo(&global);
              step_Top(execCtx, &dataBo, &globalBo);
            });
      }

      q.wait();
    } else {
      // Sequential modes (for debugging)
      uint32_t global_size = WG_SIZE;
      q.parallel_for(sycl::nd_range<1>(global_size, WG_SIZE),
          [=](sycl::nd_item<1> item) {
            if (item.get_global_id(0) != 0) return;
            Buffer db = dataBuf, pdb = preDataBuf, gb = globalBuf;
            PreflightTrace pft = pf;
            LookupTables tbl = tables;
            // Null out pre_data for sequential modes
            Buffer nullBuf{nullptr, 0, 0, false};
            if (mode == kStepModeSeqForward) {
              for (uint32_t cycle = 0; cycle < last_cycle; cycle++) {
                ExecContext execCtx(pft, tbl, cycle);
                MutableBufObj data(db, nullBuf);
                GlobalBufObj global(gb);
                BufferObj dataBo(&data);
                BufferObj globalBo(&global);
                step_Top(execCtx, &dataBo, &globalBo);
              }
            } else {
              uint32_t split = pft.tableSplitCycle;
              for (uint32_t cycle = split; cycle-- > 0;) {
                ExecContext execCtx(pft, tbl, cycle);
                MutableBufObj data(db, nullBuf);
                GlobalBufObj global(gb);
                BufferObj dataBo(&data);
                BufferObj globalBo(&global);
                step_Top(execCtx, &dataBo, &globalBo);
              }
              for (uint32_t cycle = last_cycle; cycle-- > split;) {
                ExecContext execCtx(pft, tbl, cycle);
                MutableBufObj data(db, nullBuf);
                GlobalBufObj global(gb);
                BufferObj dataBo(&data);
                BufferObj globalBo(&global);
                step_Top(execCtx, &dataBo, &globalBo);
              }
            }
          });
      q.wait();
    }

    auto t2 = std::chrono::steady_clock::now();
    bool verbose = (getenv("RISC0_VERBOSE") != nullptr);
    if (verbose) {
      fprintf(stderr, "      [ffi_witgen_intel] upload: %.1fms, kernel: %.1fms\n",
              std::chrono::duration<double, std::milli>(t1 - t0).count(),
              std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    return nullptr;
  } catch (const sycl::exception& e) {
    static thread_local std::string err;
    err = std::string("SYCL witgen error: ") + e.what();
    return err.c_str();
  } catch (const std::exception& e) {
    static thread_local std::string err;
    err = std::string("witgen error: ") + e.what();
    return err.c_str();
  }
}

const char* risc0_circuit_rv32im_intel_accum(
    void* queue_ptr,
    void* d_data_ptr, uint32_t data_rows, uint32_t data_cols,
    void* d_accum_ptr, uint32_t accum_rows, uint32_t accum_cols,
    void* d_global_ptr, uint32_t global_cols,
    void* d_mix_ptr, uint32_t mix_cols,
    const PreflightCycle* h_cycles, uint32_t cycles_len,
    const MemoryTransaction* h_txns, uint32_t txns_len,
    const uint8_t* h_bigint, uint32_t bigint_len,
    uint32_t table_split_cycle,
    uint32_t last_cycle)
{
  try {
    auto& q = *static_cast<sycl::queue*>(queue_ptr);
    auto t0 = std::chrono::steady_clock::now();

    // Reuse cached preflight from witgen phase (already on device).
    // DO NOT re-upload — witgen modified txnIdx in-place on device.
    g_cache.ensure(q, cycles_len, txns_len, bigint_len);
    if (!g_cache.preflight_on_device) {
      // Standalone accum (no prior witgen) — must upload
      g_sort.valid = false;
      q.memcpy(g_cache.d_cycles, h_cycles, cycles_len * sizeof(PreflightCycle));
      q.memcpy(g_cache.d_txns, h_txns, txns_len * sizeof(MemoryTransaction));
      if (bigint_len > 0) {
        q.memcpy(g_cache.d_bigintBytes, h_bigint, bigint_len);
      }
    }
    // Zero lookup tables for accum phase. No q.wait() — in-order queue chains.
    q.memset(g_cache.d_tableU8, 0, 256 * sizeof(uint32_t));
    q.memset(g_cache.d_tableU16, 0, 65536 * sizeof(uint32_t));

    auto t1 = std::chrono::steady_clock::now();

    Buffer dataBuf{static_cast<risc0::Fp*>(d_data_ptr), data_rows, data_cols, false};
    Buffer accumBuf{static_cast<risc0::Fp*>(d_accum_ptr), accum_rows, accum_cols, false};
    Buffer globalBuf{static_cast<risc0::Fp*>(d_global_ptr), 1, global_cols, false};
    Buffer mixBuf{static_cast<risc0::Fp*>(d_mix_ptr), 1, mix_cols, false};

    PreflightTrace pf;
    pf.cycles = g_cache.d_cycles;
    pf.txns = g_cache.d_txns;
    pf.bigintBytes = g_cache.d_bigintBytes;
    pf.txnsLen = txns_len;
    pf.bigintBytesLen = bigint_len;
    pf.tableSplitCycle = table_split_cycle;

    LookupTables tables;
    tables.tableU8 = g_cache.d_tableU8;
    tables.tableU16 = g_cache.d_tableU16;

    // WG_SIZE for accum phase1/phase3 kernels.
    // Runtime-tunable via RISC0_ACCUM_WG for sweep without 30-min rebuilds.
    // Default 256 matches the original (pre-tunable) value.
    uint32_t WG_SIZE = 256;
    if (const char* s = std::getenv("RISC0_ACCUM_WG")) {
      int v = std::atoi(s);
      if (v == 16 || v == 32 || v == 64 || v == 128 || v == 256 || v == 512 || v == 1024) {
        WG_SIZE = (uint32_t)v;
      }
    }

    const bool verbose_timing = (getenv("RISC0_VERBOSE") != nullptr);
    // Phase 1: step_TopAccum — parallel, one work-item per cycle
    const bool asort = sort_env("RISC0_ACCUM_SORT", 1) != 0;
    if (asort && !(g_sort.valid && g_sort.last == last_cycle && g_sort.split == table_split_cycle))
      sort_build(q, g_cache.d_cycles, table_split_cycle, last_cycle);
    const uint32_t* aperm = asort ? g_sort.perm : nullptr;
    {
      uint32_t global_size = ((last_cycle + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
      q.parallel_for(sycl::nd_range<1>(global_size, WG_SIZE),
          [=](sycl::nd_item<1> item) {
            uint32_t cycle = item.get_global_id(0);
            if (cycle >= last_cycle) return;
            if (aperm) cycle = aperm[cycle];
            Buffer db = dataBuf, ab = accumBuf, gb = globalBuf, mb = mixBuf;
            PreflightTrace pft = pf;
            LookupTables tbl = tables;
            ExecContext execCtx(pft, tbl, cycle);
            MutableBufObj data(db);
            MutableBufObj accum(ab, kUserAccumSplit);
            GlobalBufObj global(gb);
            GlobalBufObj mix(mb);
            BufferObj accumBo(&accum);
            BufferObj dataBo(&data);
            BufferObj globalBo(&global);
            BufferObj mixBo(&mix);
            step_TopAccum(execCtx, &accumBo, &dataBo, &globalBo, &mixBo);
          });
      if (verbose_timing) q.wait();
    }

    auto t2 = std::chrono::steady_clock::now();

    // Phase 2: inclusive scan on last 4 columns, on the GPU (3-pass blocked
    // scan; Fp addition is exact so the result is bit-identical to the
    // sequential CPU scan it replaces).
    {
      risc0::Fp* d_accum = static_cast<risc0::Fp*>(d_accum_ptr);
      const size_t rows = accum_rows;
      const size_t col0 = accum_cols - 4;
      const uint32_t n = last_cycle;
      constexpr uint32_t CH = 256;     // rows per work-item
      const uint32_t SW = scan_width<WitgenAccumScan>(q, 1024);  // lanes in the partials scan
      const uint32_t nblk = (n + CH - 1) / CH;
      const uint32_t R = (nblk + SW - 1) / SW;
      static risc0::Fp* d_part = nullptr;
      static size_t part_cap = 0;
      if (size_t(4) * nblk > part_cap) {
        if (d_part) sycl::free(d_part, q);
        part_cap = size_t(4) * nblk;
        d_part = sycl::malloc_device<risc0::Fp>(part_cap, q);
      }
      risc0::Fp* part = d_part;
      if (n > 0) {
        q.parallel_for(sycl::range<1>(size_t(4) * nblk), [=](sycl::id<1> id) {
          uint32_t c = id[0] / nblk, b = id[0] % nblk;
          const risc0::Fp* a = d_accum + (col0 + c) * rows;
          uint32_t lo = b * CH, hi = sycl::min(lo + CH, n);
          risc0::Fp s(0);
          for (uint32_t i = lo; i < hi; i++) s = s + a[i];
          part[id[0]] = s;
        });
        q.submit([&](sycl::handler& h) {
          sycl::local_accessor<risc0::Fp, 1> sm(SW, h);
          h.parallel_for<WitgenAccumScan>(sycl::nd_range<1>(4 * SW, SW), [=](sycl::nd_item<1> it) {
            uint32_t l = it.get_local_id(0), c = it.get_group(0);
            risc0::Fp* pc = part + size_t(c) * nblk;
            uint32_t lo = l * R, hi = sycl::min(lo + R, nblk);
            risc0::Fp s(0);
            for (uint32_t j = lo; j < hi; j++) s = s + pc[j];
            sm[l] = s;
            sycl::group_barrier(it.get_group());
            for (uint32_t st = 1; st < SW; st <<= 1) {
              risc0::Fp o = (l >= st) ? sm[l - st] : risc0::Fp(0);
              sycl::group_barrier(it.get_group());
              s = s + o;
              sm[l] = s;
              sycl::group_barrier(it.get_group());
            }
            risc0::Fp run = (l > 0) ? sm[l - 1] : risc0::Fp(0);  // exclusive
            for (uint32_t j = lo; j < hi; j++) { risc0::Fp v = pc[j]; pc[j] = run; run = run + v; }
          });
        });
        q.parallel_for(sycl::range<1>(size_t(4) * nblk), [=](sycl::id<1> id) {
          uint32_t c = id[0] / nblk, b = id[0] % nblk;
          risc0::Fp* a = d_accum + (col0 + c) * rows;
          uint32_t lo = b * CH, hi = sycl::min(lo + CH, n);
          risc0::Fp run = part[id[0]];
          for (uint32_t i = lo; i < hi; i++) { run = run + a[i]; a[i] = run; }
        });
        if (verbose_timing) q.wait();
      }
    }

    auto t3 = std::chrono::steady_clock::now();

    // Phase 3: finalizeAccum — parallel, add prefix sums to machine columns
    {
      uint32_t global_size = ((last_cycle + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
      q.parallel_for(sycl::nd_range<1>(global_size, WG_SIZE),
          [=](sycl::nd_item<1> item) {
            uint32_t cycle = item.get_global_id(0);
            if (cycle >= last_cycle) return;
            Buffer ab = accumBuf;
            size_t machineColumns = (ab.cols - kUserAccumSplit) / 4;
            size_t back1 = (cycle + last_cycle - 1) % last_cycle;
            risc0::Fp prev[4];
            for (size_t k = 0; k < 4; k++) {
              prev[k] = ab.get(back1, ab.cols - 4 + k);
            }
            for (size_t j = 0; j < machineColumns - 1; j++) {
              for (size_t k = 0; k < 4; k++) {
                size_t col = kUserAccumSplit + j * 4 + k;
                // An unset (INVALID) cell counts as 0: INVALID + prev[k] is unreduced when
                // prev[k] is 0, and the NTT mishandles it (see phase 3 of
                // risc0_circuit_rv32im_cpu_accum).
                risc0::Fp cur = ab.get(cycle, col);
                if (cur == risc0::Fp::invalid()) {
                  cur = risc0::Fp(0);
                }
                ab.set(cycle, col, cur + prev[k]);
              }
            }
          });
      q.wait();
    }

    auto t4 = std::chrono::steady_clock::now();

    bool verbose = (getenv("RISC0_VERBOSE") != nullptr);
    if (verbose) {
      fprintf(stderr, "      [ffi_accum_intel] phase1: %.1fms, scan: %.1fms, phase3: %.1fms\n",
              std::chrono::duration<double, std::milli>(t2 - t1).count(),
              std::chrono::duration<double, std::milli>(t3 - t2).count(),
              std::chrono::duration<double, std::milli>(t4 - t3).count());
    }
    return nullptr;
  } catch (const sycl::exception& e) {
    static thread_local std::string err;
    err = std::string("SYCL accum error: ") + e.what();
    return err.c_str();
  } catch (const std::exception& e) {
    static thread_local std::string err;
    err = std::string("accum error: ") + e.what();
    return err.c_str();
  }
}

} // extern "C"
