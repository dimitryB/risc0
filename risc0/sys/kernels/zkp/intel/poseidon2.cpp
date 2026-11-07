// Poseidon2 hash for BabyBear field — Intel ESIMD implementation.
// Matches risc0's CUDA sppark Poseidon2 (poseidon2.cuh + poseidon2_constants.cuh).
// SIMD16: each lane processes a different hash in parallel.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "bb31_field.hpp"

namespace esimd = sycl::ext::intel::esimd;
namespace syclex = sycl::ext::intel::experimental;
namespace syclex_oneapi = sycl::ext::oneapi::experimental;

// ============================================================================
// Poseidon2 Parameters
// ============================================================================
static constexpr uint32_t CELLS = 24;       // state width
static constexpr uint32_t CELLS_RATE = 16;  // absorption rate
static constexpr uint32_t CELLS_OUT = 8;    // digest size (output elements)
static constexpr uint32_t ROUNDS_HALF_FULL = 4;
static constexpr uint32_t ROUNDS_PARTIAL = 21;

// ============================================================================
// Round Constants (213 values, non-Montgomery form)
// Converted to Montgomery at init time.
// Layout: [4*24 full-round constants, 21 partial-round constants, 4*24 full-round constants]
// ============================================================================
static constexpr uint32_t NUM_ROUND_CONSTANTS = ROUNDS_HALF_FULL * CELLS * 2 + ROUNDS_PARTIAL;

// From poseidon2_constants.cuh — extracted directly from CUDA source
static const uint32_t ROUND_CONSTANTS_RAW[NUM_ROUND_CONSTANTS] = {
    0x0FA20C37,0x0795BB97,0x12C60B9C,0x0EABD88E,0x096485CA,0x07093527,0x1B1D4E50,0x30A01ACE,
    0x3BD86F5A,0x69AF7C28,0x3F94775F,0x731560E8,0x465A0ECD,0x574EF807,0x62FD4870,0x52CCFE44,
    0x14772B14,0x4DEDF371,0x260ACD7C,0x1F51DC58,0x75125532,0x686A4D7B,0x54BAC179,0x31947706,
    0x29799D3B,0x6E01AE90,0x203A7A64,0x4F7E25BE,0x72503F77,0x45BD3B69,0x769BD6B4,0x5A867F08,
    0x4FDBA082,0x251C4318,0x28F06201,0x6788C43A,0x4C6D6A99,0x357784A8,0x2ABAF051,0x770F7DE6,
    0x1794B784,0x4796C57A,0x724B7A10,0x449989A7,0x64935CF1,0x59E14AAC,0x0E620BB8,0x3AF5A33B,
    0x4465CC0E,0x019DF68F,0x4AF8D068,0x08784F82,0x0CEFDEAE,0x6337A467,0x32FA7A16,0x486F62D6,
    0x386A7480,0x20F17C4A,0x54E50DA8,0x2012CF03,0x5FE52950,0x09AFB6CD,0x2523044E,0x5C54D0EF,
    0x71C01F3C,0x60B2C4FB,0x4050B379,0x5E6A70A5,0x418543F5,0x71DEBE56,0x1AAD2994,0x3368A483,
    0x07A86F3A,0x5EA43FF1,0x2443780E,0x4CE444F7,0x146F9882,0x3132B089,0x197EA856,0x667030C3,
    0x2317D5DC,0x0C2C48A7,0x56B2DF66,0x67BD81E9,0x4FCDFB19,0x4BAAEF32,0x0328D30A,0x6235760D,
    0x12432912,0x0A49E258,0x030E1B70,0x48CAEB03,0x49E4D9E9,0x1051B5C6,0x6A36DBBE,0x4CFF27A5,
    0x1DA78EC2,0x730B0924,0x3EB56CF3,0x5BD93073,0x37204C97,0x51642D89,0x66E943E8,0x1A3E72DE,
    0x70BEB1E9,0x30FF3B3F,0x4240D1C4,0x12647B8D,0x65D86965,0x49EF4D7C,0x47785697,0x46B3969F,
    0x5C7B7A0E,0x7078FC60,0x4F22D482,0x482A9AEE,0x6BEB839D,
    0x032959AD,0x2B18AF6A,0x55D3DC8C,0x43BD26C8,0x0C41595F,0x7048D2E2,0x00DB8983,0x2AF563D7,
    0x6E84758F,0x611D64E1,0x1F9977E2,0x64163A0A,0x5C5FC27B,0x02E22561,0x3A2D75DB,0x1BA7B71A,
    0x34343F64,0x7406B35D,0x19DF8299,0x6FF4480A,0x514A81C8,0x57AB52CE,0x6AD69F52,0x3E0C0E0D,
    0x48126114,0x2A9D62CC,0x17441F23,0x485762BB,0x2F218674,0x06FDC64A,0x0861B7F2,0x3B36EEE6,
    0x70A11040,0x04B31737,0x3722A872,0x2A351C63,0x623560DC,0x62584AB2,0x382C7C04,0x3BF9EDC7,
    0x0E38FE51,0x376F3B10,0x5381E178,0x3AFC61C7,0x5C1BCB4D,0x6643CE1F,0x2D0AF1C1,0x08F583CC,
    0x5D6FF60F,0x6324C1E5,0x74412FB7,0x70C0192E,0x0B72F141,0x4067A111,0x57388C4F,0x351009EC,
    0x0974C159,0x539A58B3,0x038C0CFF,0x476C0392,0x3F7BC15F,0x4491DD2C,0x4D1FEF55,0x04936AE3,
    0x58214DD4,0x683C6AAD,0x1B42F16B,0x6DC79135,0x2D4E71EC,0x3E2946EA,0x59DCE8DB,0x6CEE892A,
    0x47F07350,0x7106CE93,0x3BD4A7A9,0x2BFE636A,0x430011E9,0x001CD66A,0x307FAF5B,0x0D9EF3FE,
    0x6D40043A,0x2E8F470C,0x1B6865E8,0x0C0E6C01,0x4D41981F,0x423B9D3D,0x410408CC,0x263F0884,
    0x5311BBD0,0x4DAE58D8,0x30401CEA,0x09AFA575,0x4B3D5B42,0x63AC0B37,0x5FE5BB14,0x5244E9D4,
};

// M_INT diagonal constants (24 values, non-Montgomery form)
static const uint32_t M_INT_DIAG_RAW[CELLS] = {
    0x409133F0, 0x1667A8A1, 0x06A6C7B6, 0x6F53160E,
    0x273B11D1, 0x03176C5D, 0x72F9BBF9, 0x73CEBA91,
    0x5CDEF81D, 0x01393285, 0x46DAEE06, 0x065D7BA6,
    0x52D72D6F, 0x05DD05E0, 0x3BAB4B63, 0x6ADA3842,
    0x2FC5FBEC, 0x770D61B0, 0x5715AAE9, 0x03EF0E90,
    0x75B6C770, 0x242ADF5F, 0x00D0CA4C, 0x36C0E388,
};

// Host-side Montgomery encoding (same as bb31_field.hpp's convention)
static uint32_t host_mont_encode(uint32_t val) {
    return (uint32_t)(((uint64_t)val << 32) % bb31::MOD);
}

// Host-side constants in Montgomery form (computed once)
static uint32_t RC_MONT[NUM_ROUND_CONSTANTS];
static uint32_t DIAG_MONT[CELLS];
static bool g_poseidon2_host_init = false;

// Cached device buffers (allocated once per GPU)
static uint32_t* g_d_rc = nullptr;
static uint32_t* g_d_diag = nullptr;

static void ensure_poseidon2_host_constants() {
    if (g_poseidon2_host_init) return;
    for (uint32_t i = 0; i < NUM_ROUND_CONSTANTS; i++)
        RC_MONT[i] = host_mont_encode(ROUND_CONSTANTS_RAW[i]);
    for (uint32_t i = 0; i < CELLS; i++)
        DIAG_MONT[i] = host_mont_encode(M_INT_DIAG_RAW[i]);
    g_poseidon2_host_init = true;
}

static void ensure_poseidon2_device_constants(sycl::queue& q) {
    ensure_poseidon2_host_constants();
    if (!g_d_rc) {
        g_d_rc = sycl::malloc_device<uint32_t>(NUM_ROUND_CONSTANTS, q);
        g_d_diag = sycl::malloc_device<uint32_t>(CELLS, q);
        q.memcpy(g_d_rc, RC_MONT, NUM_ROUND_CONSTANTS * 4);
        q.memcpy(g_d_diag, DIAG_MONT, CELLS * 4);
        q.wait();
    }
}

// ============================================================================
// gate0 OpenCL-source kernel-bundle path.
//
// Compiles the gate0 poseidon2.cl (embedded as a C++ raw string) via SYCL's
// ext_oneapi_source kernel_bundle API, caches the executable bundle + kernel
// handles per (context, device), and provides drop-in replacements for the
// ESIMD `esimd_poseidon2_fold` / `_rows` entrypoints.
//
// gate0's kernels expect:
//   - round constants in PADDED 696-entry layout (24 cells per row, including
//     zero-padded partial rounds) — we allocate `g_d_rc_padded` for this.
//   - diag in Mont form — `g_d_diag` already is, reuse it.
//   - an r_sq scalar = BabyBear R^2 mod P, for in-kernel normal→Mont conversion.
//   - inputs/outputs in NORMAL form (matching ESIMD's API contract).
// ============================================================================
#include "poseidon2_gate0_cl.inc"   // defines POSEIDON2_GATE0_CL_SRC
#include "poseidon2_commit_cl.inc"  // defines POSEIDON2_COMMIT_CL_SRC
#include "poseidon2_lift_a.inc"     // defines POSEIDON2_LIFT_A (v9 DPAS tiles)

// ============================================================================
// The OpenCL kernels compiled ahead of time (risc0-sys build.rs,
// poseidon2_ahead_of_time): one native binary per Intel target the build
// compiles for and per source, loaded through Level Zero. Any other GPU, or a
// build without targets, compiles the embedded source as before.
// ============================================================================
#if __has_include("poseidon2_aot.inc")
#include "poseidon2_aot.inc"
#else
struct Poseidon2Aot { const char* device; const char* source; const unsigned char* data; size_t size; };
static const Poseidon2Aot* const POSEIDON2_AOT = nullptr;
static const size_t POSEIDON2_AOT_COUNT = 0;
#endif
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// The Level Zero calls, looked up at run time: the SYCL runtime's Level Zero
// adapter has the loader in the process already, and linking it here would
// make this library need it wherever it is built.
struct Poseidon2ZeApi {
    decltype(&zeModuleCreate) module_create = nullptr;
    decltype(&zeKernelCreate) kernel_create = nullptr;
    decltype(&zeModuleBuildLogGetString) log_get = nullptr;
    decltype(&zeModuleBuildLogDestroy) log_destroy = nullptr;
};

static const Poseidon2ZeApi* poseidon2_ze_api() {
    static Poseidon2ZeApi api;
    static bool ok = [] {
#if defined(_WIN32)
        HMODULE h = GetModuleHandleA("ze_loader.dll");
        if (!h) h = LoadLibraryA("ze_loader.dll");
        auto sym = [&](const char* n) -> void* { return h ? reinterpret_cast<void*>(GetProcAddress(h, n)) : nullptr; };
#else
        void* h = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen("libze_loader.so.1", RTLD_NOW);
        auto sym = [&](const char* n) -> void* { return h ? dlsym(h, n) : nullptr; };
#endif
        api.module_create = reinterpret_cast<decltype(api.module_create)>(sym("zeModuleCreate"));
        api.kernel_create = reinterpret_cast<decltype(api.kernel_create)>(sym("zeKernelCreate"));
        api.log_get = reinterpret_cast<decltype(api.log_get)>(sym("zeModuleBuildLogGetString"));
        api.log_destroy = reinterpret_cast<decltype(api.log_destroy)>(sym("zeModuleBuildLogDestroy"));
        return api.module_create && api.kernel_create;
    }();
    return ok ? &api : nullptr;
}

// The ocloc name of the device's architecture, for the ones a build can target.
static const char* poseidon2_aot_device(const sycl::device& dev) {
    namespace sxa = sycl::ext::oneapi::experimental;
    switch (dev.get_info<sxa::info::device::architecture>()) {
    case sxa::architecture::intel_gpu_bmg_g21: return "bmg_g21";
    case sxa::architecture::intel_gpu_bmg_g31: return "bmg_g31";
    case sxa::architecture::intel_gpu_acm_g10: return "acm_g10";
    case sxa::architecture::intel_gpu_acm_g11: return "acm_g11";
    case sxa::architecture::intel_gpu_acm_g12: return "acm_g12";
    default: return nullptr;
    }
}

struct Poseidon2Native {
    sycl::kernel_bundle<sycl::bundle_state::executable> bundle;
    ze_module_handle_t module;
};

// The device's ahead-of-time binary of `source` ("gate0", "commit") as an
// executable bundle, or nothing: no binary for this device, not Level Zero,
// or the driver refused it (then the source is compiled instead).
static std::optional<Poseidon2Native> poseidon2_native(sycl::queue& q, const char* source) {
    if (POSEIDON2_AOT_COUNT == 0 || q.get_backend() != sycl::backend::ext_oneapi_level_zero) return std::nullopt;
    const char* device = poseidon2_aot_device(q.get_device());
    if (!device) return std::nullopt;
    const Poseidon2Aot* aot = nullptr;
    for (size_t i = 0; i < POSEIDON2_AOT_COUNT; ++i) {
        if (!std::strcmp(POSEIDON2_AOT[i].device, device) && !std::strcmp(POSEIDON2_AOT[i].source, source)) {
            aot = &POSEIDON2_AOT[i];
        }
    }
    const Poseidon2ZeApi* ze = poseidon2_ze_api();
    if (!aot || !ze) return std::nullopt;
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    ze_module_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    desc.format = ZE_MODULE_FORMAT_NATIVE;
    desc.inputSize = aot->size;
    desc.pInputModule = aot->data;
    ze_module_handle_t module = nullptr;
    ze_module_build_log_handle_t log = nullptr;
    ze_result_t r = ze->module_create(ze_ctx, ze_dev, &desc, &module, &log);
    if (log && ze->log_get && ze->log_destroy) {
        if (r != ZE_RESULT_SUCCESS) {
            size_t n = 0;
            ze->log_get(log, &n, nullptr);
            std::string text(n, '\0');
            ze->log_get(log, &n, text.data());
            std::fprintf(stderr, "[poseidon2] %s binary for %s refused (0x%x), compiling the source: %s\n",
                         source, device, static_cast<unsigned>(r), text.c_str());
        }
        ze->log_destroy(log);
    }
    if (r != ZE_RESULT_SUCCESS) return std::nullopt;
    auto bundle = sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero, sycl::bundle_state::executable>(
        {module, sycl::ext::oneapi::level_zero::ownership::transfer}, q.get_context());
    return Poseidon2Native{bundle, module};
}

static sycl::kernel poseidon2_native_kernel(const Poseidon2Native& native, sycl::queue& q, const std::string& name) {
    ze_kernel_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    desc.pKernelName = name.c_str();
    ze_kernel_handle_t kernel = nullptr;
    if (poseidon2_ze_api()->kernel_create(native.module, &desc, &kernel) != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("zeKernelCreate failed for " + name);
    }
    return sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
        {native.bundle, kernel, sycl::ext::oneapi::level_zero::ownership::transfer}, q.get_context());
}

static constexpr uint32_t POSEIDON2_R2_SQ = 1172168163u;  // R^2 mod P
static constexpr uint32_t POSEIDON2_RC_PADDED = 696;       // gate0 padded layout

static uint32_t* g_d_rc_padded = nullptr;
static std::optional<sycl::kernel> g_k_hash_fold;
static std::optional<sycl::kernel> g_k_hash_rows_24;
static std::optional<sycl::kernel> g_k_hash_rows_generic;  // any col_size, Mont-IO REPLACE sponge
static std::optional<sycl::kernel> g_k_hash_rows_v6;       // commit-path: register-resident absorb + prefetch (SIMD16, WG 256)
static std::optional<sycl::kernel> g_k_hash_fold_v3;       // commit-path: Shoup diag + literal RC
static bool g_p2_v6 = false, g_p2_fold_v3 = false;  // set only once the kernels exist
static size_t g_rows_lws = 1024;
// Last argument of the commit-path rows/fold kernels: the Mont diagonal for
// v3..v7, the partial-round lift table (POSEIDON2_LIFT_A) for v9.
static uint16_t* g_d_lift_a = nullptr;
static uint32_t* g_rows_diag = nullptr;
static uint32_t* g_fold_diag = nullptr;

static void ensure_poseidon2_opencl_kernels(sycl::queue& q) {
    // The commit-path kernels (v6 rows / v3 fold) are optional: if their
    // bundle is switched off or fails to build, the gate0 kernels are used.
    if (g_k_hash_fold && g_k_hash_rows_24 && g_k_hash_rows_generic && g_d_rc_padded) return;

    ensure_poseidon2_host_constants();

    // Build padded round-constants table (696 entries).
    static uint32_t RC_PADDED[POSEIDON2_RC_PADDED];
    static bool padded_built = false;
    if (!padded_built) {
        std::memset(RC_PADDED, 0, sizeof(RC_PADDED));
        uint32_t src_off = 0, dst_off = 0;
        // Full rounds 0..3: 4 rows × 24 cells.
        for (uint32_t r = 0; r < ROUNDS_HALF_FULL; ++r) {
            for (uint32_t i = 0; i < CELLS; ++i)
                RC_PADDED[dst_off + i] = RC_MONT[src_off + i];
            src_off += CELLS; dst_off += CELLS;
        }
        // Partial rounds 4..(4+21-1): 21 rows, only cell 0 used.
        for (uint32_t r = 0; r < ROUNDS_PARTIAL; ++r) {
            RC_PADDED[dst_off] = RC_MONT[src_off];
            src_off += 1; dst_off += CELLS;
        }
        // Full rounds (4+21)..(4+21+4-1).
        for (uint32_t r = 0; r < ROUNDS_HALF_FULL; ++r) {
            for (uint32_t i = 0; i < CELLS; ++i)
                RC_PADDED[dst_off + i] = RC_MONT[src_off + i];
            src_off += CELLS; dst_off += CELLS;
        }
        padded_built = true;
    }

    if (!g_d_rc_padded) {
        g_d_rc_padded = sycl::malloc_device<uint32_t>(POSEIDON2_RC_PADDED, q);
        q.memcpy(g_d_rc_padded, RC_PADDED, POSEIDON2_RC_PADDED * 4);
        // diag already populated in g_d_diag by ensure_poseidon2_device_constants;
        // call it to be safe.
        ensure_poseidon2_device_constants(q);
        q.wait();
    }

    if (!g_k_hash_fold) {
        namespace sx = sycl::ext::oneapi::experimental;
        // The ahead-of-time binaries first (built in the mode the source build ends up in).
        static std::optional<Poseidon2Native> native_gate0 = poseidon2_native(q, "gate0");
        if (native_gate0) {
            try {
                g_k_hash_fold         = poseidon2_native_kernel(*native_gate0, q, "poseidon2_hash_fold_mont_io");
                g_k_hash_rows_24      = poseidon2_native_kernel(*native_gate0, q, "poseidon2_hash_rows_24_mont_io");
                g_k_hash_rows_generic = poseidon2_native_kernel(*native_gate0, q, "poseidon2_hash_rows_mont_io");
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[poseidon2] gate0 binary unusable, compiling the source: %s\n", e.what());
                g_k_hash_fold.reset();
                g_k_hash_rows_24.reset();
                g_k_hash_rows_generic.reset();
                native_gate0.reset();
            }
        }
        if (!native_gate0) {
            auto src_kb = sx::create_kernel_bundle_from_source(
                q.get_context(), sx::source_language::opencl,
                std::string(POSEIDON2_GATE0_CL_SRC));
            // Use _mont_io variants: skip to_mont(input)/from_mont(output) since
            // the ESIMD prover pipeline keeps buffers in Mont form throughout.
            // In 256-GRF mode first. Alchemist (Arc A770) cannot run these kernels' 256-wide SIMD32
            // work-groups in 256-GRF mode (UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE from
            // urKernelCreate), so there they are built in the default mode.
            auto build_gate0 = [&](bool large_grf) {
                std::string log;
                std::vector<std::string> opts = {"-cl-std=CL3.0", "-cl-mad-enable"};
                if (large_grf) opts.push_back("-cl-intel-256-GRF-per-thread");
                auto exe_kb = sx::build(src_kb, sx::properties{
                    sx::build_options(opts), sx::save_log(&log)});
                if (!log.empty()) {
                    std::fprintf(stderr, "[poseidon2-opencl] build log:\n%s\n", log.c_str());
                }
                g_k_hash_fold         = exe_kb.ext_oneapi_get_kernel("poseidon2_hash_fold_mont_io");
                g_k_hash_rows_24      = exe_kb.ext_oneapi_get_kernel("poseidon2_hash_rows_24_mont_io");
                g_k_hash_rows_generic = exe_kb.ext_oneapi_get_kernel("poseidon2_hash_rows_mont_io");
            };
            try {
                build_gate0(true);
            } catch (const sycl::exception&) {
                g_k_hash_fold.reset();
                g_k_hash_rows_24.reset();
                g_k_hash_rows_generic.reset();
                build_gate0(false);
            }
        }
        // Commit-path kernels: separate small bundle built in 128-GRF mode
        // (SIMD16, 8 threads/EU; state fits without spill). Measured 185 -> 134 ms
        // for 211 cols x 2^22 rows on B70, bit-identical.
        const bool want_v6 = !std::getenv("RISC0_P2_ROWS_V6_OFF");
        const bool want_fold_v3 = !std::getenv("RISC0_P2_FOLD_V3_OFF");
        if (want_v6 || want_fold_v3) {
            try {
                static std::optional<Poseidon2Native> native_commit = poseidon2_native(q, "commit");
                std::optional<sycl::kernel_bundle<sycl::bundle_state::executable>> exe2;
                if (!native_commit) {
                    auto src2 = sx::create_kernel_bundle_from_source(
                        q.get_context(), sx::source_language::opencl,
                        std::string(POSEIDON2_COMMIT_CL_SRC));
                    std::string log2;
                    std::vector<std::string> opts2 = {"-cl-std=CL3.0", "-cl-mad-enable"};
                    exe2 = sx::build(src2, sx::properties{sx::build_options(opts2), sx::save_log(&log2)});
                    if (!log2.empty()) std::fprintf(stderr, "[poseidon2-commit] build log:\n%s\n", log2.c_str());
                }
                auto get2 = [&](const std::string& name) -> sycl::kernel {
                    return native_commit ? poseidon2_native_kernel(*native_commit, q, name)
                                         : exe2->ext_oneapi_get_kernel(name);
                };
                // SIMD16 + madw(zero addend) + 1-block register prefetch. Robust to the
                // GRF mode: 152 ms (IGC_TotalGRFNum=256 forced, as in the bench scripts) /
                // 135 ms (128 GRF) vs 185 ms production, 211 cols x 2^22. The NEO persistent
                // cache does not key on IGC_TotalGRFNum, so do not select kernels by env.
                // v7: v3 structure (literal RC, Shoup diag) with subtraction-form REDC,
                // no madw, 1024-wide WGs; IGC picks SIMD16 in the default 128-GRF mode.
                // 211 cols x 2^22: 121 ms vs 135 (v6) / 151 (v6 under IGC_TotalGRFNum=256).
                // Needs IGC_TotalGRFNum unset (it forces 256 GRF on this bundle: v7 -> 181 ms).
                // v9: v7 with the 21 partial rounds evaluated as fixed linear
                // forms on XMX (DPAS, exact; tools/poseidon2_lift_gen.py) and a
                // signed-Montgomery S-box: rows -34%, fold -37% vs v7, bit-identical.
                // RISC0_P2_ROWS / RISC0_P2_FOLD select another kernel by suffix (e.g. v7, v6, v3).
                // v9 is for Xe2 (Battlemage): on Alchemist (Arc A770), whose DPAS runs at
                // SIMD8, v9's rows gave wrong hashes (hash_rows_poseidon2 failed; v7 passed).
                namespace sxa = sycl::ext::oneapi::experimental;
                const auto arch = q.get_device().get_info<sxa::info::device::architecture>();
                const bool alchemist = arch == sxa::architecture::intel_gpu_acm_g10 ||
                                       arch == sxa::architecture::intel_gpu_acm_g11 ||
                                       arch == sxa::architecture::intel_gpu_acm_g12;
                const char* rsel = std::getenv("RISC0_P2_ROWS");
                const char* fsel = std::getenv("RISC0_P2_FOLD");
                std::string rname = rsel ? rsel : (alchemist ? "v7" : "v9");
                std::string fname = fsel ? fsel : (alchemist ? "v7" : "v9");
                if (rname == "v9" || fname == "v9") {
                    if (!g_d_lift_a) {
                        g_d_lift_a = sycl::malloc_device<uint16_t>(sizeof(POSEIDON2_LIFT_A) / 2, q);
                        if (!g_d_lift_a) throw std::runtime_error("lift table allocation failed");
                        q.memcpy(g_d_lift_a, POSEIDON2_LIFT_A, sizeof(POSEIDON2_LIFT_A)).wait();
                    }
                }
                g_rows_lws = (rname == "v6" || rname == "v7pf" || rname == "v7s16") ? 256 : 1024;
                g_k_hash_rows_v6 = get2("poseidon2_hash_rows_mont_io_" + rname);
                g_k_hash_fold_v3 = get2("poseidon2_hash_fold_mont_io_" + fname);
                g_rows_diag = (rname == "v9") ? reinterpret_cast<uint32_t*>(g_d_lift_a) : g_d_diag;
                g_fold_diag = (fname == "v9") ? reinterpret_cast<uint32_t*>(g_d_lift_a) : g_d_diag;
                // v7/v9 rows require 1024-wide work-groups; if the JIT could not
                // provide that (e.g. IGC_TotalGRFNum=256 forces 256-GRF mode),
                // use v6 (256-wide).
                size_t rows_max = g_k_hash_rows_v6->get_info<
                    sycl::info::kernel_device_specific::work_group_size>(q.get_device());
                if (rows_max < g_rows_lws) {
                    std::fprintf(stderr, "[poseidon2-commit] %s rows kernel max WG %zu < %zu; using v6\n",
                                 rname.c_str(), rows_max, g_rows_lws);
                    g_k_hash_rows_v6 = get2("poseidon2_hash_rows_mont_io_v6");
                    g_rows_lws = 256;
                    g_rows_diag = g_d_diag;
                }
                // fold v3 is launched with 1024-wide work-groups.
                size_t fold_max = g_k_hash_fold_v3->get_info<
                    sycl::info::kernel_device_specific::work_group_size>(q.get_device());
                g_p2_v6 = want_v6;
                g_p2_fold_v3 = want_fold_v3 && fold_max >= 1024;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[poseidon2-commit] disabled, using gate0 kernels: %s\n", e.what());
                g_p2_v6 = g_p2_fold_v3 = false;
            }
        }
    }
}

// ============================================================================
// Poseidon2 Permutation Core (SIMD16: each lane = independent hash)
// ============================================================================

// 4x4 circulant multiply: circ(2,3,1,1) * diag structure
// Input/output: 4 Vec16 values (each lane independent)
// CSE optimized: hoist doubled values to avoid redundant computation
ESIMD_INLINE void multiply_by_4x4_circulant(bb31::Vec16& x0, bb31::Vec16& x1,
                                              bb31::Vec16& x2, bb31::Vec16& x3) {
    auto t0 = bb31::field_add(x0, x1);
    auto t1 = bb31::field_add(x2, x3);
    auto t2 = bb31::field_add(bb31::field_add(x1, x1), t1); // 2*x1 + t1
    auto t3 = bb31::field_add(bb31::field_add(x3, x3), t0); // 2*x3 + t0
    auto double_t1 = bb31::field_add(t1, t1);  // CSE: compute once
    auto double_t0 = bb31::field_add(t0, t0);  // CSE: compute once
    auto t4 = bb31::field_add(bb31::field_add(double_t1, double_t1), t3); // 4*t1 + t3
    auto t5 = bb31::field_add(bb31::field_add(double_t0, double_t0), t2); // 4*t0 + t2
    x0 = bb31::field_add(t3, t5); // t6
    x1 = t5;
    x2 = bb31::field_add(t2, t4); // t7
    x3 = t4;
}

// External MDS matrix multiply (block-circulant from 4x4 blocks)
ESIMD_INLINE void multiply_by_m_ext(bb31::Vec16 cells[CELLS]) {
    // Apply 4x4 circulant to each of 6 groups independently
    // Then accumulate column sums and add back
    bb31::Vec16 sums[4] = {bb31::Vec16(0u), bb31::Vec16(0u),
                            bb31::Vec16(0u), bb31::Vec16(0u)};
    for (uint32_t g = 0; g < 6; g++) {
        multiply_by_4x4_circulant(cells[g*4], cells[g*4+1], cells[g*4+2], cells[g*4+3]);
        sums[0] = bb31::field_add(sums[0], cells[g*4]);
        sums[1] = bb31::field_add(sums[1], cells[g*4+1]);
        sums[2] = bb31::field_add(sums[2], cells[g*4+2]);
        sums[3] = bb31::field_add(sums[3], cells[g*4+3]);
    }
    // Unrolled 6 groups × 4 cells (was: for i in 24, cells[i]+=sums[i%4]).
    // Lets IGC scalar-replace sums[] and avoid reloading from the modulo-indexed array.
    cells[0]  = bb31::field_add(cells[0],  sums[0]);
    cells[1]  = bb31::field_add(cells[1],  sums[1]);
    cells[2]  = bb31::field_add(cells[2],  sums[2]);
    cells[3]  = bb31::field_add(cells[3],  sums[3]);
    cells[4]  = bb31::field_add(cells[4],  sums[0]);
    cells[5]  = bb31::field_add(cells[5],  sums[1]);
    cells[6]  = bb31::field_add(cells[6],  sums[2]);
    cells[7]  = bb31::field_add(cells[7],  sums[3]);
    cells[8]  = bb31::field_add(cells[8],  sums[0]);
    cells[9]  = bb31::field_add(cells[9],  sums[1]);
    cells[10] = bb31::field_add(cells[10], sums[2]);
    cells[11] = bb31::field_add(cells[11], sums[3]);
    cells[12] = bb31::field_add(cells[12], sums[0]);
    cells[13] = bb31::field_add(cells[13], sums[1]);
    cells[14] = bb31::field_add(cells[14], sums[2]);
    cells[15] = bb31::field_add(cells[15], sums[3]);
    cells[16] = bb31::field_add(cells[16], sums[0]);
    cells[17] = bb31::field_add(cells[17], sums[1]);
    cells[18] = bb31::field_add(cells[18], sums[2]);
    cells[19] = bb31::field_add(cells[19], sums[3]);
    cells[20] = bb31::field_add(cells[20], sums[0]);
    cells[21] = bb31::field_add(cells[21], sums[1]);
    cells[22] = bb31::field_add(cells[22], sums[2]);
    cells[23] = bb31::field_add(cells[23], sums[3]);
}

// Internal MDS matrix multiply: cell[i] = sum(all) + diag[i] * cell[i]
ESIMD_INLINE void multiply_by_m_int(bb31::Vec16 cells[CELLS],
                                     const bb31::Vec16 diag[CELLS]) {
    bb31::Vec16 sum = cells[0];
    for (uint32_t i = 1; i < CELLS; i++)
        sum = bb31::field_add(sum, cells[i]);
    for (uint32_t i = 0; i < CELLS; i++)
        cells[i] = bb31::field_add(sum, bb31::mont_mul(diag[i], cells[i]));
}

// S-box: x^7 = x * x^2 * x^4
ESIMD_INLINE bb31::Vec16 sbox(bb31::Vec16 x) {
    auto x2 = bb31::mont_mul(x, x);
    auto x4 = bb31::mont_mul(x2, x2);
    auto x6 = bb31::mont_mul(x4, x2);
    return bb31::mont_mul(x6, x);
}

// Full Poseidon2 permutation with cached constants.
// Both round constants (213 scalars = 27 GRF) and M_INT diagonal (24 scalars = 3 GRF)
// are cached in scalar private arrays, eliminating ~10K+ device memory reads per permutation.
// (Previous comment claimed 522 GRF — that was wrong: 213 broadcast scalars, not Vec16s.)
ESIMD_INLINE void poseidon2_mix(bb31::Vec16 cells[CELLS],
                                 const uint32_t* prc,   // round constants (device ptr)
                                 const uint32_t* pdiag) { // M_INT diagonal (device ptr)
    // Cache ALL round constants in scalar registers (213 uint32_t = 27 GRF).
    // Constants are broadcast to all 16 SIMD lanes via Vec16(scalar) — no vector storage needed.
    uint32_t rc_s[NUM_ROUND_CONSTANTS];
    for (uint32_t i = 0; i < NUM_ROUND_CONSTANTS; i++)
        rc_s[i] = *(prc + i);
    uint32_t rc_off = 0;

    // Cache M_INT diagonal in scalar registers (24 uint32_t = 3 GRF).
    uint32_t diag_s[CELLS];
    for (uint32_t i = 0; i < CELLS; i++)
        diag_s[i] = *(pdiag + i);

    // Initial external matrix
    multiply_by_m_ext(cells);

    // First half of full rounds — gate0 layer-pattern sbox.
    //
    // Was: per-cell sbox (`cells[i] = sbox(cells[i])` in a #pragma-unroll loop).
    // Now: layer-by-layer x→x²→x⁴→x⁶→x⁷, exposing 24-way independent-mul ILP
    // at every multiply layer instead of a 4-mul dependency chain per cell.
    // Source: inteldebug/gate0/poseidon2.cl::add_rc_sbox_layer_mont.
    // Standalone OpenCL bench measured 2.71× over the per-cell pattern; the
    // layer pattern uses 4×CELLS = 96 Vec16 scratch (~96 GRFs at 256-GRF)
    // which is well within the per-thread budget on Xe2-HPG.
    for (uint32_t r = 0; r < ROUNDS_HALF_FULL; r++) {
        bb31::Vec16 x[CELLS], x2v[CELLS], x4v[CELLS], x6v[CELLS];
        #pragma unroll
        for (uint32_t i = 0; i < CELLS; i++)
            x[i] = bb31::field_add(cells[i], bb31::Vec16(rc_s[rc_off + i]));
        rc_off += CELLS;
        #pragma unroll
        for (uint32_t i = 0; i < CELLS; i++) x2v[i] = bb31::mont_mul(x[i], x[i]);
        #pragma unroll
        for (uint32_t i = 0; i < CELLS; i++) x4v[i] = bb31::mont_mul(x2v[i], x2v[i]);
        #pragma unroll
        for (uint32_t i = 0; i < CELLS; i++) x6v[i] = bb31::mont_mul(x4v[i], x2v[i]);
        #pragma unroll
        for (uint32_t i = 0; i < CELLS; i++) cells[i] = bb31::mont_mul(x6v[i], x[i]);
        multiply_by_m_ext(cells);
    }

    // Partial rounds — tree reduction for M_INT sum (depth 5 vs depth 23)
    // Note: diag product precomputation was tested but REGRESSED performance
    // (pushed register pressure past small GRF threshold, halving occupancy).
    for (uint32_t r = 0; r < ROUNDS_PARTIAL; r++) {
        cells[0] = bb31::field_add(cells[0], bb31::Vec16(rc_s[rc_off]));
        rc_off += 1;
        cells[0] = sbox(cells[0]);
        // M_INT: sum + diag[i] * cells[i]
        // Tree reduction: 24 elements → 12 pairs → 6 → 3 → 2 → 1
        bb31::Vec16 s0  = bb31::field_add(cells[0],  cells[1]);
        bb31::Vec16 s1  = bb31::field_add(cells[2],  cells[3]);
        bb31::Vec16 s2  = bb31::field_add(cells[4],  cells[5]);
        bb31::Vec16 s3  = bb31::field_add(cells[6],  cells[7]);
        bb31::Vec16 s4  = bb31::field_add(cells[8],  cells[9]);
        bb31::Vec16 s5  = bb31::field_add(cells[10], cells[11]);
        bb31::Vec16 s6  = bb31::field_add(cells[12], cells[13]);
        bb31::Vec16 s7  = bb31::field_add(cells[14], cells[15]);
        bb31::Vec16 s8  = bb31::field_add(cells[16], cells[17]);
        bb31::Vec16 s9  = bb31::field_add(cells[18], cells[19]);
        bb31::Vec16 s10 = bb31::field_add(cells[20], cells[21]);
        bb31::Vec16 s11 = bb31::field_add(cells[22], cells[23]);
        bb31::Vec16 q0 = bb31::field_add(s0, s1);
        bb31::Vec16 q1 = bb31::field_add(s2, s3);
        bb31::Vec16 q2 = bb31::field_add(s4, s5);
        bb31::Vec16 q3 = bb31::field_add(s6, s7);
        bb31::Vec16 q4 = bb31::field_add(s8, s9);
        bb31::Vec16 q5 = bb31::field_add(s10, s11);
        bb31::Vec16 h0 = bb31::field_add(q0, q1);
        bb31::Vec16 h1 = bb31::field_add(q2, q3);
        bb31::Vec16 h2 = bb31::field_add(q4, q5);
        bb31::Vec16 sum = bb31::field_add(bb31::field_add(h0, h1), h2);

        #pragma unroll
        for (uint32_t i = 0; i < CELLS; i++)
            cells[i] = bb31::field_add(sum, bb31::mont_mul(bb31::Vec16(diag_s[i]), cells[i]));
    }

    // Second half of full rounds — gate0 layer-pattern sbox (see comment above).
    for (uint32_t r = 0; r < ROUNDS_HALF_FULL; r++) {
        bb31::Vec16 x[CELLS], x2v[CELLS], x4v[CELLS], x6v[CELLS];
        #pragma unroll
        for (uint32_t i = 0; i < CELLS; i++)
            x[i] = bb31::field_add(cells[i], bb31::Vec16(rc_s[rc_off + i]));
        rc_off += CELLS;
        #pragma unroll
        for (uint32_t i = 0; i < CELLS; i++) x2v[i] = bb31::mont_mul(x[i], x[i]);
        #pragma unroll
        for (uint32_t i = 0; i < CELLS; i++) x4v[i] = bb31::mont_mul(x2v[i], x2v[i]);
        #pragma unroll
        for (uint32_t i = 0; i < CELLS; i++) x6v[i] = bb31::mont_mul(x4v[i], x2v[i]);
        #pragma unroll
        for (uint32_t i = 0; i < CELLS; i++) cells[i] = bb31::mont_mul(x6v[i], x[i]);
        multiply_by_m_ext(cells);
    }
}

// ============================================================================
// Queue creation
// ============================================================================
static sycl::queue create_poseidon2_queue() {
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (auto& d : gpu_devices) {
        auto name = d.get_info<sycl::info::device::name>();
        if (name.find("Intel") != std::string::npos ||
            name.find("0xe2") != std::string::npos)
            return sycl::queue(d, sycl::property_list{sycl::property::queue::in_order{}});
    }
    throw std::runtime_error("Intel GPU not found");
}

extern "C" {

// ============================================================================
// hash_rows: hash each column of a row-major matrix
// Input: matrix[row * count + col], rows = col_size, cols = count
// Output: out[col * CELLS_OUT + c] for c=0..7
// SIMD16: 16 columns processed per thread
// ============================================================================
static void poseidon2_rows_impl(sycl::queue& q,
                                uint32_t* d_out, const uint32_t* d_in,
                                uint32_t count, uint32_t col_size, bool allow_opencl);

void esimd_poseidon2_rows(sycl::queue& q,
                           uint32_t* d_out, const uint32_t* d_in,
                           uint32_t count, uint32_t col_size) {
    poseidon2_rows_impl(q, d_out, d_in, count, col_size, true);
}

// ESIMD-only variant for esimd_warmup. Launching the generic OpenCL rows
// kernel during warmup (ahead of the twiddle/NTT warmup and the first rv32im
// eval_check) leaves the monolithic eval_check kernel -- ~515 KB private
// memory per thread -- computing garbage in ~30 random SIMD16 threads per
// launch at po2=16, i.e. invalid segment proofs. The kernel itself is exact
// (and the warmup fold's OpenCL launch is harmless); this looks like a driver
// private-surface issue, found empirically. Keep warmup on the ESIMD path it
// used before the generic OpenCL rows kernel existed.
void esimd_poseidon2_rows_esimd(sycl::queue& q,
                                uint32_t* d_out, const uint32_t* d_in,
                                uint32_t count, uint32_t col_size) {
    poseidon2_rows_impl(q, d_out, d_in, count, col_size, false);
}

static void poseidon2_rows_impl(sycl::queue& q,
                                uint32_t* d_out, const uint32_t* d_in,
                                uint32_t count, uint32_t col_size, bool allow_opencl) {
    // [gate0 OpenCL fast path]
    // - col_size == 24: bespoke `poseidon2_hash_rows_24_mont_io` (FRI commit width).
    // - col_size != 24: generic `poseidon2_hash_rows_mont_io` (loop variant of
    //   the same REPLACE-style sponge ESIMD uses). Lights up the rv32im
    //   data/code/accum/check merkle commits and recursion's data merkle
    //   (col_size=128) — they were previously falling through to the slower
    //   ESIMD path.
    // Default: ON. Set RISC0_POSEIDON2_OPENCL_OFF=1 for emergency rollback.
    // Set RISC0_POSEIDON2_ROWS_GENERIC_OFF=1 to keep only the col_size==24
    // fast path lit (for bisection of the generic kernel).
    bool opencl_ok = allow_opencl && !std::getenv("RISC0_POSEIDON2_OPENCL_OFF");
    bool generic_ok = opencl_ok && !std::getenv("RISC0_POSEIDON2_ROWS_GENERIC_OFF");
    if (opencl_ok && (col_size == 24 || generic_ok)) {
        ensure_poseidon2_opencl_kernels(q);
        const uint32_t r_sq = POSEIDON2_R2_SQ;
        bool v6 = (col_size != 24) && g_p2_v6 && g_k_hash_rows_v6.has_value();
        sycl::kernel k = (col_size == 24) ? *g_k_hash_rows_24 : (v6 ? *g_k_hash_rows_v6 : *g_k_hash_rows_generic);
        uint32_t* d_dst = d_out;
        const uint32_t* d_src = d_in;
        uint32_t* d_rc_pad = g_d_rc_padded;
        uint32_t* d_diag = v6 ? g_rows_diag : g_d_diag;
        uint32_t cnt = count;
        uint32_t cs = col_size;
        const size_t lws = v6 ? g_rows_lws : 1024;
        const size_t gws = ((size_t(count) + lws - 1) / lws) * lws;
        q.submit([&](sycl::handler& cgh) {
            cgh.set_arg(0, d_dst);
            cgh.set_arg(1, d_src);
            cgh.set_arg(2, cnt);
            cgh.set_arg(3, cs);       // generic kernel actually reads col_size; _24 ignores
            cgh.set_arg(4, r_sq);     // unused by both _mont_io kernels (kept for ABI)
            cgh.set_arg(5, d_rc_pad);
            cgh.set_arg(6, d_diag);
            cgh.parallel_for(sycl::nd_range<1>(gws, lws), k);
        });
        return;
    }

    ensure_poseidon2_device_constants(q);

    auto* pin = d_in;
    auto* pout = d_out;
    auto* prc = g_d_rc;
    auto* pdiag = g_d_diag;
    uint32_t num_threads = (count + 15) / 16;

    // R2-1: Request 256-GRF mode for this kernel. ISA disasm of round-1 build
    // showed grf_count=128 with significant r[a0.X] indirection on cells[24];
    // 256-GRF should let IGC keep cells in flat GRF without indirection.
    q.parallel_for(sycl::range<1>(num_threads),
        syclex_oneapi::properties{syclex::grf_size<256>},
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base_col = idx[0] * 16;
            if (base_col >= count) return;

            // Initialize state to zero
            bb31::Vec16 cells[CELLS];
            for (uint32_t i = 0; i < CELLS; i++)
                cells[i] = bb31::Vec16(0u);

            // Sponge: absorb col_size elements per column, CELLS_RATE at a time
            uint32_t absorbed = 0;
            while (absorbed < col_size) {
                uint32_t chunk = col_size - absorbed;
                if (chunk > CELLS_RATE) chunk = CELLS_RATE;

                // Load chunk elements for 16 columns
                for (uint32_t j = 0; j < chunk; j++) {
                    uint32_t row = absorbed + j;
                    // Matrix is row-major: element at (row, col) = pin[row * count + col]
                    if (base_col + 16 <= count) {
                        cells[j] = esimd::block_load<uint32_t, 16>(pin + row * count + base_col);
                    } else {
                        // Tail: load available columns, zero-pad
                        cells[j] = bb31::Vec16(0u);
                        for (uint32_t lane = 0; lane < 16 && base_col + lane < count; lane++) {
                            cells[j][lane] = *(pin + row * count + base_col + lane);
                        }
                    }
                }
                // Zero-pad remaining rate cells
                for (uint32_t j = chunk; j < CELLS_RATE; j++)
                    cells[j] = bb31::Vec16(0u);

                absorbed += chunk;
                poseidon2_mix(cells, prc, pdiag);
            }

            // If nothing was absorbed (col_size == 0), do one permutation
            if (col_size == 0) {
                poseidon2_mix(cells, prc, pdiag);
            }

            // Output: cells[0..7] as digest in AoS layout
            // out[(base_col+lane) * CELLS_OUT + c] = cells[c][lane]
            for (uint32_t c = 0; c < CELLS_OUT; c++) {
                if (base_col + 16 <= count) {
                    esimd::simd<uint32_t, 16> lane(0u, 1u);
                    auto byte_off = ((base_col + lane) * CELLS_OUT + c) * (uint32_t)sizeof(uint32_t);
                    esimd::scatter<uint32_t, 16>(pout, byte_off, cells[c]);
                } else {
                    for (uint32_t lane = 0; lane < 16 && base_col + lane < count; lane++)
                        *(pout + (base_col + lane) * CELLS_OUT + c) = cells[c][lane];
                }
            }
        });
}

// ============================================================================
// hash_fold: hash pairs of digests for Merkle tree folding
// Input: in[i * CELLS_RATE + j] for i=0..num_hashes-1, j=0..15 (2 digests = 16 elements)
// Output: out[i * CELLS_OUT + c] for c=0..7
// SIMD16: 16 pairs processed per thread
// ============================================================================
void esimd_poseidon2_fold(sycl::queue& q,
                           uint32_t* d_out, const uint32_t* d_in,
                           uint32_t num_hashes) {
    // [gate0 OpenCL fast path] Dispatch the gate0 poseidon2_hash_fold OpenCL
    // kernel via SYCL's source-bundle API. Standalone-bench measured 2.71×
    // over the ESIMD path on B70; in production prove the Merkle-commit
    // savings show up as ~46 ms/seg.
    // Default: ON. Set RISC0_POSEIDON2_OPENCL_OFF=1 to fall back to ESIMD.
    if (!std::getenv("RISC0_POSEIDON2_OPENCL_OFF")) {
        ensure_poseidon2_opencl_kernels(q);
        const uint32_t r_sq = POSEIDON2_R2_SQ;
        // v3 only for real tree layers: esimd_warmup's 1-hash fold keeps the
        // original kernel so the warmup launch sequence is unchanged.
        const bool v3 = g_p2_fold_v3 && g_k_hash_fold_v3.has_value() && num_hashes >= 1024;
        sycl::kernel k = v3 ? *g_k_hash_fold_v3 : *g_k_hash_fold;
        uint32_t* d_dst = d_out;
        const uint32_t* d_src = d_in;
        uint32_t* d_rc_pad = g_d_rc_padded;
        uint32_t* d_diag = v3 ? g_fold_diag : g_d_diag;
        uint32_t nh = num_hashes;
        // LWS=1024 was the optimum measured for hash_fold on B70 in the
        // gate0 standalone bench (+5% over LWS=256).
        const size_t lws = 1024;
        const size_t gws = ((size_t(num_hashes) + lws - 1) / lws) * lws;
        q.submit([&](sycl::handler& cgh) {
            cgh.set_arg(0, (void*)d_dst);
            cgh.set_arg(1, (void*)d_src);
            cgh.set_arg(2, (void*)d_rc_pad);
            cgh.set_arg(3, (void*)d_diag);
            cgh.set_arg(4, nh);
            cgh.set_arg(5, r_sq);
            cgh.parallel_for(sycl::nd_range<1>(gws, lws), k);
        });
        return;
    }

    ensure_poseidon2_device_constants(q);

    auto* pin = d_in;
    auto* pout = d_out;
    auto* prc = g_d_rc;
    auto* pdiag = g_d_diag;
    uint32_t num_threads = (num_hashes + 15) / 16;
    // DIAG: print pre-call output
    if (std::getenv("RISC0_POSEIDON2_TRACE")) {
        static int esimd_trace_count = 0;
        if (esimd_trace_count < 3) {
            // capture ESIMD output after kernel runs
            esimd_trace_count++;
        }
    }

    q.parallel_for(sycl::range<1>(num_threads),
        syclex_oneapi::properties{syclex::grf_size<256>},
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            if (base >= num_hashes) return;

            // Initialize state
            bb31::Vec16 cells[CELLS];
            for (uint32_t i = 0; i < CELLS; i++)
                cells[i] = bb31::Vec16(0u);

            // Load 16 input elements per hash (= 2 digests)
            // Input layout: in[hash_idx * 16 + j]
            for (uint32_t j = 0; j < CELLS_RATE; j++) {
                if (base + 16 <= num_hashes) {
                    esimd::simd<uint32_t, 16> lane(0u, 1u);
                    auto byte_off = ((base + lane) * CELLS_RATE + j) * (uint32_t)sizeof(uint32_t);
                    cells[j] = esimd::gather<uint32_t, 16>(pin, byte_off);
                } else {
                    cells[j] = bb31::Vec16(0u);
                    for (uint32_t lane = 0; lane < 16 && base + lane < num_hashes; lane++)
                        cells[j][lane] = *(pin + (base + lane) * CELLS_RATE + j);
                }
            }
            // Capacity cells stay zero

            // Single permutation (16 elements fits in one rate block)
            poseidon2_mix(cells, prc, pdiag);

            // Output digest: out[hash_idx * CELLS_OUT + c]
            for (uint32_t c = 0; c < CELLS_OUT; c++) {
                if (base + 16 <= num_hashes) {
                    esimd::simd<uint32_t, 16> lane(0u, 1u);
                    auto byte_off = ((base + lane) * CELLS_OUT + c) * (uint32_t)sizeof(uint32_t);
                    esimd::scatter<uint32_t, 16>(pout, byte_off, cells[c]);
                } else {
                    for (uint32_t lane = 0; lane < 16 && base + lane < num_hashes; lane++)
                        *(pout + (base + lane) * CELLS_OUT + c) = cells[c][lane];
                }
            }
        });
    if (std::getenv("RISC0_POSEIDON2_TRACE")) {
        static int t = 0;
        if (t < 3) {
            q.wait();
            uint32_t host_out[8];
            q.memcpy(host_out, d_out, 8 * 4).wait();
            std::fprintf(stderr, "[fold-esimd  #%d] num_hashes=%u dst[0..8]=", t, num_hashes);
            for (int i = 0; i < 8; ++i) std::fprintf(stderr, "%08x ", host_out[i]);
            std::fprintf(stderr, "\n");
            t++;
        }
    }
}

// ============================================================================
// Validation: test poseidon2_mix against known test vector
// ============================================================================
int32_t validate_poseidon2() {
    try {
    auto q = create_poseidon2_queue();
    ensure_poseidon2_host_constants();

    // Known test vector from Rust: input [0,1,2,...,23] (normal form)
    // Expected output (normal form):
    uint32_t expected_normal[CELLS] = {
        0x2ED3E23D, 0x12921FB0, 0x0E659E79, 0x61D81DC9,
        0x32BAE33B, 0x62486AE3, 0x1E681B60, 0x24B91325,
        0x2A2EF5B9, 0x50E8593E, 0x5BC818EC, 0x10691997,
        0x35A14520, 0x2BA6A3C5, 0x279D47EC, 0x55014E81,
        0x5953A67F, 0x2F403111, 0x6B8828FF, 0x1801301F,
        0x2749207A, 0x3DC9CF21, 0x3C985BA2, 0x57A99864,
    };

    // Convert to Montgomery form
    uint32_t input_mont[CELLS];
    uint32_t expected_mont[CELLS];
    for (uint32_t i = 0; i < CELLS; i++) {
        input_mont[i] = host_mont_encode(i);
        expected_mont[i] = host_mont_encode(expected_normal[i]);
    }

    // Run on GPU: use 1 hash with 24 input elements (via hash_rows with col_size=24, count=1)
    // But actually we need to test the permutation directly.
    // Instead, let's test via hash_fold with a specially crafted input.
    // Simpler: test the permutation via CPU emulation using our Montgomery arithmetic.

    // CPU test of poseidon2_mix
    // We need to run the permutation on the CPU with our constants.
    // Use the host_mont_mul from ntt_kernel.cpp pattern.
    auto host_mul = [](uint32_t a, uint32_t b) -> uint32_t {
        uint64_t prod = (uint64_t)a * b;
        uint32_t lo = (uint32_t)prod, hi = (uint32_t)(prod >> 32);
        uint32_t red = lo * 0x77ffffffu;
        uint64_t rprod = (uint64_t)red * bb31::MOD;
        uint32_t rlo = (uint32_t)rprod, rhi = (uint32_t)(rprod >> 32);
        uint32_t carry = ((uint64_t)lo + rlo) >= (1ULL << 32) ? 1 : 0;
        uint32_t res = hi + rhi + carry;
        return res >= bb31::MOD ? res - bb31::MOD : res;
    };
    auto host_add = [](uint32_t a, uint32_t b) -> uint32_t {
        uint32_t r = a + b; return r >= bb31::MOD ? r - bb31::MOD : r;
    };
    auto host_sbox = [&](uint32_t x) -> uint32_t {
        uint32_t x2 = host_mul(x, x);
        uint32_t x4 = host_mul(x2, x2);
        uint32_t x6 = host_mul(x4, x2);
        return host_mul(x6, x);
    };

    // CPU permutation using run_perm helper
    auto cpu_4x4 = [&](uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
        uint32_t t0 = host_add(a, b);
        uint32_t t1 = host_add(c, d);
        uint32_t t2 = host_add(host_add(b, b), t1);
        uint32_t t3 = host_add(host_add(d, d), t0);
        uint32_t ft1 = host_add(host_add(t1, t1), host_add(t1, t1));
        uint32_t ft0 = host_add(host_add(t0, t0), host_add(t0, t0));
        uint32_t t4 = host_add(ft1, t3);
        uint32_t t5 = host_add(ft0, t2);
        a = host_add(t3, t5);
        b = t5;
        c = host_add(t2, t4);
        d = t4;
    };
    auto run_perm = [&](uint32_t* c) {
        uint32_t off = 0;
        // Copy to local
        uint32_t lc[CELLS]; for (uint32_t i = 0; i < CELLS; i++) lc[i] = c[i];
        // m_ext on lc
        {
            uint32_t s[4] = {0,0,0,0};
            for (uint32_t g = 0; g < 6; g++) {
                cpu_4x4(lc[g*4], lc[g*4+1], lc[g*4+2], lc[g*4+3]);
                for (uint32_t j = 0; j < 4; j++) s[j] = host_add(s[j], lc[g*4+j]);
            }
            for (uint32_t i = 0; i < CELLS; i++) lc[i] = host_add(lc[i], s[i%4]);
        }
        for (uint32_t r = 0; r < ROUNDS_HALF_FULL; r++) {
            for (uint32_t i = 0; i < CELLS; i++) lc[i] = host_add(lc[i], RC_MONT[off+i]);
            off += CELLS;
            for (uint32_t i = 0; i < CELLS; i++) lc[i] = host_sbox(lc[i]);
            uint32_t s[4] = {0,0,0,0};
            for (uint32_t g = 0; g < 6; g++) {
                cpu_4x4(lc[g*4], lc[g*4+1], lc[g*4+2], lc[g*4+3]);
                for (uint32_t j = 0; j < 4; j++) s[j] = host_add(s[j], lc[g*4+j]);
            }
            for (uint32_t i = 0; i < CELLS; i++) lc[i] = host_add(lc[i], s[i%4]);
        }
        for (uint32_t r = 0; r < ROUNDS_PARTIAL; r++) {
            lc[0] = host_add(lc[0], RC_MONT[off]); off++;
            lc[0] = host_sbox(lc[0]);
            uint32_t sum = 0;
            for (uint32_t i = 0; i < CELLS; i++) sum = host_add(sum, lc[i]);
            for (uint32_t i = 0; i < CELLS; i++) lc[i] = host_add(sum, host_mul(DIAG_MONT[i], lc[i]));
        }
        for (uint32_t r = 0; r < ROUNDS_HALF_FULL; r++) {
            for (uint32_t i = 0; i < CELLS; i++) lc[i] = host_add(lc[i], RC_MONT[off+i]);
            off += CELLS;
            for (uint32_t i = 0; i < CELLS; i++) lc[i] = host_sbox(lc[i]);
            uint32_t s[4] = {0,0,0,0};
            for (uint32_t g = 0; g < 6; g++) {
                cpu_4x4(lc[g*4], lc[g*4+1], lc[g*4+2], lc[g*4+3]);
                for (uint32_t j = 0; j < 4; j++) s[j] = host_add(s[j], lc[g*4+j]);
            }
            for (uint32_t i = 0; i < CELLS; i++) lc[i] = host_add(lc[i], s[i%4]);
        }
        for (uint32_t i = 0; i < CELLS; i++) c[i] = lc[i];
    };
    // Test 1: Permutation test vector [0,1,...,23]
    uint32_t tv_cells[CELLS];
    for (uint32_t i = 0; i < CELLS; i++) tv_cells[i] = input_mont[i];
    run_perm(tv_cells);

    int32_t errors = 0;
    for (uint32_t i = 0; i < CELLS; i++) {
        if (tv_cells[i] != expected_mont[i]) {
            if (errors < 5)
                fprintf(stderr, "  poseidon2 TV mismatch at %u: got=%08x exp=%08x\n",
                        i, tv_cells[i], expected_mont[i]);
            errors++;
        }
    }
    fprintf(stderr, "  poseidon2 permutation (CPU): %s\n", errors == 0 ? "PASS" : "FAIL");
    fflush(stderr);

    // Test 2: hash_fold with zero digests
    auto* d_in = sycl::malloc_device<uint32_t>(CELLS_RATE, q);
    auto* d_out = sycl::malloc_device<uint32_t>(CELLS_OUT, q);
    auto* h_in = sycl::malloc_host<uint32_t>(CELLS_RATE, q);
    auto* h_out = sycl::malloc_host<uint32_t>(CELLS_OUT, q);
    for (uint32_t i = 0; i < CELLS_RATE; i++) h_in[i] = 0;
    q.memcpy(d_in, h_in, CELLS_RATE * 4); q.wait();
    esimd_poseidon2_fold(q, d_out, d_in, 1);
    q.memcpy(h_out, d_out, CELLS_OUT * 4); q.wait();

    uint32_t ref_cells[CELLS];
    for (uint32_t i = 0; i < CELLS; i++) ref_cells[i] = 0;
    run_perm(ref_cells);

    int fold_errs = 0;
    for (uint32_t c = 0; c < CELLS_OUT; c++) {
        if (h_out[c] != ref_cells[c]) {
            fprintf(stderr, "  poseidon2 fold mismatch at %u: gpu=%08x cpu=%08x\n",
                    c, h_out[c], ref_cells[c]);
            fold_errs++;
        }
    }
    fprintf(stderr, "  poseidon2 hash_fold (GPU): %s\n", fold_errs == 0 ? "PASS" : "FAIL");
    fflush(stderr);

    sycl::free(d_in, q); sycl::free(d_out, q);
    sycl::free(h_in, q); sycl::free(h_out, q);

    return errors + fold_errs;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what()); fflush(stderr);
        return -99;
    }
}

// ============================================================================
// Benchmark: hash_fold throughput (Poseidon2 permutations per second)
// ============================================================================
struct P2BenchResult {
    double kernel_ns;
    double hashes;
    int32_t correct;
};

P2BenchResult bench_poseidon2_fold(uint32_t num_hashes) {
    try {
    auto q = create_poseidon2_queue();
    ensure_poseidon2_host_constants();

    auto* d_in = sycl::malloc_device<uint32_t>(num_hashes * CELLS_RATE, q);
    auto* d_out = sycl::malloc_device<uint32_t>(num_hashes * CELLS_OUT, q);
    // Zero-init
    q.memset(d_in, 0, num_hashes * CELLS_RATE * 4);
    q.memset(d_out, 0, num_hashes * CELLS_OUT * 4);
    q.wait();

    // Warm up
    for (int w = 0; w < 50; w++)
        esimd_poseidon2_fold(q, d_out, d_in, num_hashes);
    q.wait();

    // Timed runs: take best of 15
    constexpr int SAMPLES = 15;
    double best_ns = 1e18;
    for (int s = 0; s < SAMPLES; s++) {
        auto start = std::chrono::high_resolution_clock::now();
        esimd_poseidon2_fold(q, d_out, d_in, num_hashes);
        q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        double ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        if (ns < best_ns) best_ns = ns;
    }

    P2BenchResult result;
    result.kernel_ns = best_ns;
    result.hashes = double(num_hashes);
    result.correct = 1;

    sycl::free(d_in, q); sycl::free(d_out, q);
    return result;
    } catch (...) {
        return P2BenchResult{0.0, 0.0, -1};
    }
}

P2BenchResult bench_poseidon2_rows(uint32_t count, uint32_t col_size) {
    try {
    auto q = create_poseidon2_queue();
    ensure_poseidon2_host_constants();

    auto* d_in = sycl::malloc_device<uint32_t>(count * col_size, q);
    auto* d_out = sycl::malloc_device<uint32_t>(count * CELLS_OUT, q);
    q.memset(d_in, 0, count * col_size * 4);
    q.memset(d_out, 0, count * CELLS_OUT * 4);
    q.wait();

    // Warm up
    for (int w = 0; w < 20; w++)
        esimd_poseidon2_rows(q, d_out, d_in, count, col_size);
    q.wait();

    constexpr int SAMPLES = 15;
    double best_ns = 1e18;
    for (int s = 0; s < SAMPLES; s++) {
        auto start = std::chrono::high_resolution_clock::now();
        esimd_poseidon2_rows(q, d_out, d_in, count, col_size);
        q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        double ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        if (ns < best_ns) best_ns = ns;
    }

    P2BenchResult result;
    result.kernel_ns = best_ns;
    result.hashes = double(count);
    result.correct = 1;

    sycl::free(d_in, q); sycl::free(d_out, q);
    return result;
    } catch (...) {
        return P2BenchResult{0.0, 0.0, -1};
    }
}

} // extern "C"
