// Intel SYCL eval_check kernel for rv32im circuit (monolithic).
// This file is appended to the monolithic amalgamation.
// Multi-pass kernels are in separate .so files (pass1, pass2).

#include <sycl/sycl.hpp>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

using namespace risc0;

static const char* make_error(const char* msg) { return strdup(msg); }

// This source is built twice when the targets include GPUs without Xe2's DPAS (see
// intel_non_xmx_targets in build.rs): as risc0_rv32im_intel, with the XMX linking chains, and
// as risc0_rv32im_intel_noxmx (RISC0_RV32IM_EVAL_NOXMX), without them, exporting the same
// functions with an _noxmx suffix. The first library's exports pick one per device (end of file).
#ifdef RISC0_RV32IM_EVAL_NOXMX
#define RV_EVAL_FN(name) extern "C" const char* name##_noxmx
#else
#define RV_EVAL_FN(name) static const char* name##_local
#endif

#ifdef RK_NWORDS
// eval_slab.py moved poly_fp's per-row arrays into a slot-major global slab:
// RK_NWORDS u32 words per row, in blocks of RK_STRIDE rows. The launch is tiled
// so the slab covers one tile; tiles run in order on the single in-order queue
// and each row only reads words it wrote, so reuse across tiles is safe.
// g_slab_mutex guards the slab globals: eval_check holds it while it picks
// the slab and enqueues the tiles, and slab_free (also on release) waits for
// the queue before freeing, so kernels already enqueued by any thread finish
// first.
static std::mutex g_slab_mutex;
static uint32_t* g_slab = nullptr;
static size_t g_slab_words = 0;
static sycl::queue* g_slab_queue = nullptr;
// Largest tile that could be allocated; after a failure, later calls start
// from it instead of draining the queue to retry the full size every time.
static uint32_t g_tile_cap = 1u << 26;

static void slab_free() {
    if (g_slab) {
        g_slab_queue->wait();  // sycl::free does not wait for queued kernels
        sycl::free(g_slab, *g_slab_queue);
    }
    g_slab = nullptr;
    g_slab_words = 0;
    g_slab_queue = nullptr;
}

static size_t slab_words_for(uint32_t rows) {
    return (size_t(rows) + RK_STRIDE - 1) / RK_STRIDE * RK_NWORDS * RK_STRIDE;
}

// Rows per launch: min(tile, domain). The slab is always sized for a full
// tile (2^20 rows = 1.4 GB by default, 2^18 rows = 350 MB on cards under
// 20 GiB; RISC0_EVAL_SLAB_TILE_LOG2 overrides), independent of the domain, so
// segments of different po2 never regrow it (a regrow drains the queue). If
// that allocation fails the tile halves, down to 2^14 rows, and g_tile_cap
// keeps the smaller size until release.
//
// On a 12 GB Arc B580 under Windows, where over-budget allocations spill into
// shared system memory instead of failing, the 1.4 GB slab (held from the first
// segment to the end of the session) pushed the next session's segments into
// shared memory: po2 20 segments 27.5 s -> 39.3 s per 32M cycles, or 30.1 s with
// a Groth16 wrap between sessions. With 2^18 rows: 27.3-27.6 s either way, and
// the kernel itself is no slower (403 vs 406 ms per po2 20 segment).
static uint32_t slab_tile(sycl::queue* q, uint32_t domain) {
    constexpr uint64_t kSmallCard = uint64_t(20) << 30;
    uint32_t tile =
        q->get_device().get_info<sycl::info::device::global_mem_size>() < kSmallCard
            ? 1u << 18
            : 1u << 20;
    if (const char* s = std::getenv("RISC0_EVAL_SLAB_TILE_LOG2")) {
        int v = std::atoi(s);
        if (v >= 9 && v <= 26) tile = 1u << v;
    }
    if (tile > g_tile_cap) tile = g_tile_cap;
    for (;;) {
        const uint32_t launch = tile < domain ? tile : domain;
        if (g_slab && g_slab_queue == q && g_slab_words >= slab_words_for(launch)) return launch;
        slab_free();
        const size_t need = slab_words_for(tile);
        g_slab = sycl::malloc_device<uint32_t>(need, *q);
        if (g_slab) {
            g_slab_words = need;
            g_slab_queue = q;
            return launch;
        }
        if (tile <= (1u << 14)) throw std::runtime_error("eval_check: slab allocation failed");
        tile >>= 1;
        g_tile_cap = tile;
    }
}
#endif

RV_EVAL_FN(risc0_circuit_rv32im_intel_eval_check)(
    void* queue_ptr, void* d_check,
    const void* d_data, const void* d_accum, const void* d_out, const void* d_mix,
    const void* d_poly_mix, uint32_t rou_raw, uint32_t po2, uint32_t domain)
{
    auto* q = static_cast<sycl::queue*>(queue_ptr);
    try {
        auto* check = static_cast<Fp*>(d_check);
        auto* data = static_cast<Fp*>(const_cast<void*>(d_data));
        auto* accum = static_cast<Fp*>(const_cast<void*>(d_accum));
        auto* out = static_cast<Fp*>(const_cast<void*>(d_out));
        auto* mix = static_cast<Fp*>(const_cast<void*>(d_mix));
        auto* poly_mix = static_cast<FpExt*>(const_cast<void*>(d_poly_mix));
#ifdef RISC0_INTEL_EVAL_FAST
        // The vanishing quotient 1/((3*rou^cycle)^(2^po2) - 1) takes only 4
        // values over the coset (rou^(2^po2) is a primitive 4th root), so the
        // host appends them after the 458 poly_mix powers (see intel.rs).
        const Fp* qtab = reinterpret_cast<const Fp*>(poly_mix + 458);
#endif
        Fp rou_val;
        std::memcpy(&rou_val, &rou_raw, sizeof(uint32_t));
        // WG_SIZE=512 measured 11% E2E faster than WG=256 on B70 (BMG-G31)
        // at PO2=20 segments; +44% on tiny workloads. eval_check kernel
        // resource budget caps WG at 512 (compiled with `simd_size=16`,
        // `eu_thread_count=4`, 256-GRF; 1024 exceeds resource limit).
        // Runtime-tunable via RISC0_EVAL_CHECK_WG for further sweeps.
        uint32_t WG_SIZE = 512;
        if (const char* s = std::getenv("RISC0_EVAL_CHECK_WG")) {
            int v = std::atoi(s);
            if (v == 16 || v == 32 || v == 64 || v == 128 || v == 256 || v == 512) {
                WG_SIZE = (uint32_t)v;
            }
        }
        const uint32_t wgsz = WG_SIZE;
        constexpr uint32_t POLY_MIX_COUNT = 458;
#ifdef RK_NWORDS
        std::lock_guard<std::mutex> slab_lock(g_slab_mutex);
        const uint32_t tile = slab_tile(q, domain);
        uint32_t* slab = g_slab;
        for (uint32_t tile_base = 0; tile_base < domain; tile_base += tile) {
        const uint32_t rows = (domain - tile_base) < tile ? (domain - tile_base) : tile;
        uint32_t global_size = ((rows + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
#else
        uint32_t global_size = ((domain + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
#endif
        q->submit([&](sycl::handler& h) {
            sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(POLY_MIX_COUNT), h);
            h.parallel_for(sycl::nd_range<1>(global_size, WG_SIZE),
                [=](sycl::nd_item<1> item) {
#ifdef RK_NWORDS
                    uint32_t lrow = item.get_global_id(0);
                    uint32_t cycle = tile_base + lrow;
#else
                    uint32_t cycle = item.get_global_id(0);
#endif
                    uint32_t lid = item.get_local_id(0);
                    for (uint32_t i = lid; i < POLY_MIX_COUNT; i += wgsz) slm[i] = poly_mix[i];
                    sycl::group_barrier(item.get_group());
#ifdef RK_NWORDS
                    if (lrow >= rows) return;
                    Fp* args[5] = {accum, data, out, mix,
                                   reinterpret_cast<Fp*>(slab + (size_t)(lrow / RK_STRIDE) * RK_NWORDS * RK_STRIDE +
                                                         (lrow % RK_STRIDE))};
#else
                    if (cycle >= domain) return;
                    Fp* args[4] = {accum, data, out, mix};
#endif
                    FpExt* pm = slm.get_multi_ptr<sycl::access::decorated::no>().get();
                    FpExt tot = circuit::rv32im_v2::poly_fp(
                        (size_t)cycle, (size_t)domain, pm, args);
#ifdef RISC0_INTEL_EVAL_FAST
                    Fp quot = qtab[cycle & 3];
#else
                    Fp x = Fp(3) * pow(rou_val, cycle);
                    Fp y = pow(x, uint32_t(1) << po2);
                    Fp quot = inv(y - Fp(1));
#endif
                    for (uint32_t i = 0; i < 4; i++)
                        check[i * domain + cycle] = tot.elems[i] * quot;
                });
        });
#ifdef RK_NWORDS
        }
#endif
        return nullptr;
    } catch (const sycl::exception& e) { return make_error(e.what()); }
    catch (const std::exception& e) { return make_error(e.what()); }
    catch (...) { return make_error("Unknown error in intel eval_check"); }
}

RV_EVAL_FN(risc0_circuit_rv32im_intel_eval_check_sync)(void* queue_ptr) {
    auto* q = static_cast<sycl::queue*>(queue_ptr);
    try { q->wait(); return nullptr; }
    catch (const sycl::exception& e) { return make_error(e.what()); }
    catch (const std::exception& e) { return make_error(e.what()); }
    catch (...) { return make_error("Unknown error in eval_check_sync"); }
}

// Frees eval_check's scratch (the RK_NWORDS slab) between proving sessions so
// it does not stay resident; it is reallocated on the next eval_check.
RV_EVAL_FN(risc0_circuit_rv32im_intel_eval_check_release)() {
    try {
#ifdef RK_NWORDS
        std::lock_guard<std::mutex> slab_lock(g_slab_mutex);
        slab_free();
        g_tile_cap = 1u << 26;  // next session may try the full tile again
#endif
        return nullptr;
    } catch (const sycl::exception& e) { return make_error(e.what()); }
    catch (const std::exception& e) { return make_error(e.what()); }
    catch (...) { return make_error("Unknown error in eval_check_release"); }
}

#ifndef RISC0_RV32IM_EVAL_NOXMX
// The exports: this library's kernel on GPUs with Xe2's DPAS (Battlemage, Lunar Lake), and
// risc0_rv32im_intel_noxmx's, loaded from this library's directory, on the others. The XMX
// linking chains are written for Xe2's SIMD16 DPAS; on Alchemist (Arc A770), whose DPAS runs at
// SIMD8, they gave wrong values in every row (built ahead of time or by the driver alike).
// RISC0_INTEL_EVAL_NOXMX=1 picks the second library on any GPU.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <string>

namespace {
using EvalFn = const char* (*)(void*, void*, const void*, const void*, const void*, const void*,
                               const void*, uint32_t, uint32_t, uint32_t);
using ReleaseFn = const char* (*)();

struct NoXmx {
    bool use = false;
    EvalFn eval = nullptr;
    ReleaseFn release = nullptr;
    std::string error;
};

bool has_xe2_dpas(const sycl::device& d) {
    namespace sx = sycl::ext::oneapi::experimental;
    const auto a = d.get_info<sx::info::device::architecture>();
    return a == sx::architecture::intel_gpu_bmg_g21 ||
           a == sx::architecture::intel_gpu_bmg_g31 || a == sx::architecture::intel_gpu_lnl_m;
}

// Loads the second library from this library's directory.
void load_noxmx(NoXmx& n) {
#ifdef _WIN32
    HMODULE self = nullptr;
    char path[MAX_PATH];
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(&has_xe2_dpas), &self) ||
        !GetModuleFileNameA(self, path, MAX_PATH)) {
        n.error = "eval_check: cannot locate risc0_rv32im_intel";
        return;
    }
    std::string file(path);
    file = file.substr(0, file.find_last_of("\\/") + 1) + "risc0_rv32im_intel_noxmx.dll";
    HMODULE lib = LoadLibraryA(file.c_str());
    if (lib) {
        n.eval = reinterpret_cast<EvalFn>(
            GetProcAddress(lib, "risc0_circuit_rv32im_intel_eval_check_noxmx"));
        n.release = reinterpret_cast<ReleaseFn>(
            GetProcAddress(lib, "risc0_circuit_rv32im_intel_eval_check_release_noxmx"));
    }
#else
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&has_xe2_dpas), &info) || !info.dli_fname) {
        n.error = "eval_check: cannot locate librisc0_rv32im_intel";
        return;
    }
    std::string file(info.dli_fname);
    file = file.substr(0, file.find_last_of('/') + 1) + "librisc0_rv32im_intel_noxmx.so";
    void* lib = dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (lib) {
        n.eval = reinterpret_cast<EvalFn>(dlsym(lib, "risc0_circuit_rv32im_intel_eval_check_noxmx"));
        n.release = reinterpret_cast<ReleaseFn>(
            dlsym(lib, "risc0_circuit_rv32im_intel_eval_check_release_noxmx"));
    }
#endif
    if (!n.eval || !n.release) {
        n.error = "eval_check: this GPU has no Xe2 DPAS and needs " + file +
                  ", built when RISC0_INTEL_SYCL_TARGETS includes a target other than Battlemage "
                  "or Lunar Lake (the default includes spir64)";
        n.eval = nullptr;
        n.release = nullptr;
    }
}

std::once_flag g_noxmx_once;
NoXmx g_noxmx;

// Decided once, for the first queue's device (a process proves on one device).
const NoXmx& noxmx_for(sycl::queue* q) {
    std::call_once(g_noxmx_once, [q] {
        const char* force = std::getenv("RISC0_INTEL_EVAL_NOXMX");
        g_noxmx.use = (force && std::strcmp(force, "0") != 0) || !has_xe2_dpas(q->get_device());
        if (g_noxmx.use) load_noxmx(g_noxmx);
    });
    return g_noxmx;
}
}  // namespace

extern "C" const char* risc0_circuit_rv32im_intel_eval_check(
    void* queue_ptr, void* d_check, const void* d_data, const void* d_accum, const void* d_out,
    const void* d_mix, const void* d_poly_mix, uint32_t rou_raw, uint32_t po2, uint32_t domain) {
    try {
        const NoXmx& n = noxmx_for(static_cast<sycl::queue*>(queue_ptr));
        if (n.use) {
            if (!n.eval) return make_error(n.error.c_str());
            return n.eval(queue_ptr, d_check, d_data, d_accum, d_out, d_mix, d_poly_mix, rou_raw,
                          po2, domain);
        }
    } catch (const std::exception& e) { return make_error(e.what()); }
    return risc0_circuit_rv32im_intel_eval_check_local(queue_ptr, d_check, d_data, d_accum, d_out,
                                                       d_mix, d_poly_mix, rou_raw, po2, domain);
}

extern "C" const char* risc0_circuit_rv32im_intel_eval_check_sync(void* queue_ptr) {
    return risc0_circuit_rv32im_intel_eval_check_sync_local(queue_ptr);
}

extern "C" const char* risc0_circuit_rv32im_intel_eval_check_release() {
    if (g_noxmx.use && g_noxmx.release) {
        if (const char* err = g_noxmx.release()) return err;
    }
    return risc0_circuit_rv32im_intel_eval_check_release_local();
}
#endif
