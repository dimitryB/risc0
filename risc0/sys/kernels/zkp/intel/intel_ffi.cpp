// Intel GPU FFI wrapper layer for risc0 HAL integration.
// Provides C-compatible function signatures matching risc0/risc0/sys/src/intel.rs.
// Each function accepts an opaque queue handle, calls ESIMD kernels, returns const char*.

#include <sycl/sycl.hpp>
#include <cstring>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <cstdlib>
#include <algorithm>
#include <deque>

// On Windows, Level Zero's USM pool off, set while this library loads (before any thread runs,
// and before SYCL initializes Level Zero, which reads it then). The HAL pools device buffers
// itself (risc0-zkp's Intel HAL), and the Groth16 prover allocates in a context of its own.
// Buffers they freed stayed in the adapter's pool, still counted in this process's video memory
// usage, where neither could reuse them, so later allocations spilled into shared system memory:
// on a 12 GB Arc B580 at po2=20, with the pool, the third session's segments took 35.6 s instead
// of 28-30 s. UR_L0_DISABLE_USM_ALLOCATOR=0 (or its older name,
// SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR) keeps it.
// Also on Windows, immediate command lists and no copy engine for Level Zero's legacy adapter,
// which the loader picks for an Alchemist card (Arc A770) when no visible device supports the V2
// adapter. Without both, the recursion receipts came out wrong on an A770 (the next join's
// preflight failed an equality check) unless a host wait followed each eval_check. Either
// variable set keeps its value. On V2, the A770 lost the device (UR_RESULT_ERROR_DEVICE_LOST)
// a few po2=20 segments into a session, so V2 is not forced here; with a Battlemage card also
// visible the loader picks V2 for both. Battlemage runs on V2, which ignores these.
#ifdef _WIN32
static const bool kLevelZeroEnv = [] {
    if (!std::getenv("UR_L0_DISABLE_USM_ALLOCATOR") &&
        !std::getenv("SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR"))
        _putenv_s("UR_L0_DISABLE_USM_ALLOCATOR", "1");
    if (!std::getenv("UR_L0_USE_IMMEDIATE_COMMANDLISTS") &&
        !std::getenv("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"))
        _putenv_s("UR_L0_USE_IMMEDIATE_COMMANDLISTS", "1");
    if (!std::getenv("UR_L0_USE_COPY_ENGINE") &&
        !std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE"))
        _putenv_s("UR_L0_USE_COPY_ENGINE", "0");
    return true;
}();
#endif

// SYCL's persistent kernel cache on, unless SYCL_CACHE_PERSISTENT says otherwise: GPUs without
// ahead-of-time code (e.g. Alchemist, Arc A770) get their kernels compiled from SPIR-V by the
// driver on first use, which for the eval_check kernels takes minutes; cached, only once.
static const bool kSyclCacheEnv = [] {
#ifdef _WIN32
    if (!std::getenv("SYCL_CACHE_PERSISTENT")) _putenv_s("SYCL_CACHE_PERSISTENT", "1");
#else
    setenv("SYCL_CACHE_PERSISTENT", "1", 0);
#endif
    return true;
}();

// Host-to-device staging. A pageable q.memcpy is NOT reliably captured at
// enqueue: NEO stages only the first copy from a given host address, and a
// later copy from the same address that is enqueued before the host has seen
// the earlier one complete reads the buffer when the GPU runs it. Callers
// routinely free or reuse their buffer (or stack slot) right after
// esimd_memcpy_htod returns, so every upload is copied into a pinned chunk
// that is not reused until its copy's event has completed. Same CPU memcpy
// the driver's staging already does; no measured cost.
// RISC0_INTEL_HTOD_STAGING=0 restores the direct (unsafe) copy for A/B tests.
namespace htod_staging {
struct Chunk { char* p; size_t cap; };
struct Inflight { Chunk c; sycl::event e; };
static constexpr size_t SMALL = size_t(64) << 10, BIG = size_t(4) << 20, CAP = size_t(512) << 20;
static std::mutex mu;
static std::vector<Chunk> free_small, free_big;
static std::deque<Inflight> inflight;
static size_t pinned_total = 0;

// Return completed chunks to the free lists; with block_one, wait for the oldest.
static void reclaim(bool block_one) {
    while (!inflight.empty()) {
        auto& f = inflight.front();
        if (f.e.get_info<sycl::info::event::command_execution_status>() !=
            sycl::info::event_command_status::complete) {
            if (!block_one) return;
            f.e.wait();
            block_one = false;
        }
        (f.c.cap == SMALL ? free_small : free_big).push_back(f.c);
        inflight.pop_front();
    }
}

static Chunk get(sycl::queue& q, size_t len) {
    const bool small = len <= SMALL;
    auto& fl = small ? free_small : free_big;
    reclaim(false);
    for (;;) {
        if (!fl.empty()) { Chunk c = fl.back(); fl.pop_back(); return c; }
        const size_t cap = small ? SMALL : BIG;
        if (pinned_total + cap <= CAP || inflight.empty()) {
            char* p = sycl::malloc_host<char>(cap, q);
            if (!p) return Chunk{nullptr, 0};  // caller falls back to a blocking copy
            pinned_total += cap;
            return Chunk{p, cap};
        }
        reclaim(true);
    }
}

static bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("RISC0_INTEL_HTOD_STAGING");
        return !(e && std::strcmp(e, "0") == 0);
    }();
    return on;
}

static void copy(sycl::queue& q, char* dst, const char* src, size_t bytes) {
    std::lock_guard<std::mutex> g(mu);
    for (size_t off = 0; off < bytes;) {
        const size_t len = std::min(bytes - off, BIG);
        Chunk c = get(q, len);
        if (!c.p) {
            q.memcpy(dst + off, src + off, len).wait();  // no pinned memory: copy synchronously
        } else {
            std::memcpy(c.p, src + off, len);
            inflight.push_back({c, q.memcpy(dst + off, c.p, len)});
        }
        off += len;
    }
}
}  // namespace htod_staging

// Forward declarations for ESIMD kernel functions (defined in other .cpp files)
extern "C" {
    // NTT (ntt_kernel.cpp)
    void gpu_forward_ntt_fast(sycl::queue& q, uint32_t* d_data, uint32_t lg_n);
    void gpu_inverse_ntt_fast(sycl::queue& q, uint32_t* d_data, uint32_t lg_n);
    void gpu_forward_ntt_no_wait(sycl::queue& q, uint32_t* d_data, uint32_t lg_n);
    void gpu_inverse_ntt_no_wait(sycl::queue& q, uint32_t* d_data, uint32_t lg_n);
    void gpu_forward_ntt_expand4_no_wait(sycl::queue& q, uint32_t* d_out, const uint32_t* d_coeffs, uint32_t lg_n);
    void gpu_inverse_ntt_zk_shift_no_wait(sycl::queue& q, uint32_t* d_data,
                                          uint32_t lg_n, const uint32_t* d_zk_powers);
    void esimd_batch_bit_reverse(sycl::queue& q, uint32_t* io, uint32_t nBits, uint32_t count);

    // Poseidon2 (poseidon2.cpp)
    void esimd_poseidon2_rows(sycl::queue& q, uint32_t* d_out, const uint32_t* d_in,
                               uint32_t count, uint32_t col_size);
    void esimd_poseidon2_rows_esimd(sycl::queue& q, uint32_t* d_out, const uint32_t* d_in,
                               uint32_t count, uint32_t col_size);
    void esimd_poseidon2_fold(sycl::queue& q, uint32_t* d_out, const uint32_t* d_in,
                               uint32_t num_hashes);

    // Poseidon-254 (poseidon254.cpp)
    void esimd_poseidon254_fold(sycl::queue& q, uint32_t* d_out, const uint32_t* d_in,
                                 uint32_t count);
    void esimd_poseidon254_rows(sycl::queue& q, uint32_t* d_out, const uint32_t* d_in,
                                 uint32_t row_size, uint32_t col_size);

    // Eltwise (eltwise_ops.cpp)
    void esimd_eltwise_add_fp(sycl::queue& q, uint32_t* out, const uint32_t* x,
                               const uint32_t* y, uint32_t count);
    void esimd_eltwise_copy_fp(sycl::queue& q, uint32_t* out, const uint32_t* in, uint32_t count);
    void esimd_eltwise_copy_fp_region(sycl::queue& q, uint32_t* into, const uint32_t* from,
                                       uint32_t fromRows, uint32_t fromCols,
                                       uint32_t fromOffset, uint32_t fromStride,
                                       uint32_t intoOffset, uint32_t intoStride);
    void esimd_eltwise_sum_fpext(sycl::queue& q, uint32_t* out, const uint32_t* in,
                                  uint32_t to_add, uint32_t count);
    void esimd_eltwise_zeroize_fp(sycl::queue& q, uint32_t* elems, uint32_t count);
    void esimd_gather_sample(sycl::queue& q, uint32_t* dst, const uint32_t* src,
                              uint32_t idx, uint32_t size, uint32_t stride);
    void esimd_scatter(sycl::queue& q, uint32_t* into, const uint32_t* index,
                        const uint32_t* offsets, const uint32_t* values, uint32_t count);

    // FRI/Polynomial (fri_poly_ops.cpp)
    void esimd_fri_fold(sycl::queue& q, uint32_t* d_out, const uint32_t* d_in,
                         const uint32_t* d_mix, uint32_t count);
    void esimd_mix_poly_coeffs(sycl::queue& q, uint32_t* d_out, const uint32_t* d_in,
                                const uint32_t* d_combos, const uint32_t* d_mix_start,
                                const uint32_t* d_mix, uint32_t inputSize, uint32_t count);
    void esimd_batch_evaluate_any(sycl::queue& q, uint32_t* d_out, const uint32_t* d_coeffs,
                                   const uint32_t* d_which, const uint32_t* d_xs,
                                   uint32_t count, uint32_t deg);
    // R2-A04: CUDA-style 1-WG-per-eval Horner variant (5.5-6.7% E2E)
    void esimd_batch_evaluate_any_cuda_style(sycl::queue& q, uint32_t* d_out, const uint32_t* d_coeffs,
                                              const uint32_t* d_which, const uint32_t* d_xs,
                                              uint32_t count, uint32_t deg);
    void esimd_combos_prepare(sycl::queue& q, uint32_t* d_combos, const uint32_t* d_coeffU,
                               uint32_t comboCount, uint32_t cycles, uint32_t regsCount,
                               const uint32_t* d_regSizes, const uint32_t* d_regComboIds,
                               uint32_t checkSize, const uint32_t* d_mix);
}

// Helper: duplicate error message string (caller frees via C free())
static const char* make_error(const char* msg) {
    return strdup(msg);
}

// Helper macro for exception-safe wrappers
#define FFI_WRAP(body) \
    try { body; return nullptr; } \
    catch (const sycl::exception& e) { return make_error(e.what()); } \
    catch (const std::exception& e) { return make_error(e.what()); } \
    catch (...) { return make_error("Unknown C++ exception"); }

// ============================================================================
// Queue lifecycle
// ============================================================================

extern "C" {

// Shared device and context for all queues.
static sycl::device* g_shared_device = nullptr;
static sycl::context* g_shared_context = nullptr;

void* esimd_create_queue() {
    try {
        if (!g_shared_device) {
            auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
            for (auto& d : gpu_devices) {
                auto name = d.get_info<sycl::info::device::name>();
                if (name.find("Intel") != std::string::npos ||
                    name.find("0xe2") != std::string::npos) {
                    g_shared_device = new sycl::device(d);
                    g_shared_context = new sycl::context(*g_shared_device);
                    break;
                }
            }
            if (!g_shared_device) return nullptr;
        }
        auto* q = new sycl::queue(*g_shared_context, *g_shared_device, sycl::property_list{
            sycl::property::queue::in_order{}});
        return static_cast<void*>(q);
    } catch (...) {
        return nullptr;
    }
}

// Total memory of the GPU the queues run on, in bytes; 0 before the first queue exists.
size_t esimd_device_total_memory() {
    return g_shared_device ? g_shared_device->get_info<sycl::info::device::global_mem_size>() : 0;
}

void esimd_destroy_queue(void* queue) {
    if (queue) {
        auto* q = static_cast<sycl::queue*>(queue);
        q->wait();
        delete q;
    }
}

void esimd_sync(void* queue) {
    auto* q = static_cast<sycl::queue*>(queue);
    q->wait();
}

void* esimd_malloc_device(void* queue, size_t bytes) {
    auto* q = static_cast<sycl::queue*>(queue);
    return sycl::malloc_device<uint8_t>(bytes, *q);
}

void esimd_free_device(void* queue, void* ptr) {
    auto* q = static_cast<sycl::queue*>(queue);
    sycl::free(ptr, *q);
}

void esimd_memcpy_htod(void* queue, void* dst, const void* src, size_t bytes) {
    auto* q = static_cast<sycl::queue*>(queue);
    if (htod_staging::enabled()) {
        // The source is copied before this returns, so the caller may free
        // or reuse it immediately. See htod_staging above.
        htod_staging::copy(*q, static_cast<char*>(dst), static_cast<const char*>(src), bytes);
        return;
    }
    q->memcpy(dst, src, bytes);
    // No wait: in-order queue guarantees subsequent kernels see the data.
    // Unsafe if the caller reuses `src` before the copy runs (see htod_staging).
}

void esimd_memcpy_dtoh(void* queue, void* dst, const void* src, size_t bytes) {
    auto* q = static_cast<sycl::queue*>(queue);
    q->memcpy(dst, src, bytes);
    q->wait();  // Must wait: host reads dst immediately after return.
}

void esimd_memset(void* queue, void* ptr, int value, size_t bytes) {
    auto* q = static_cast<sycl::queue*>(queue);
    q->memset(ptr, value, bytes);
    // No wait: in-order queue serializes subsequent operations.
}

void esimd_memcpy_dtod(void* queue, void* dst, const void* src, size_t bytes) {
    auto* q = static_cast<sycl::queue*>(queue);
    q->memcpy(dst, src, bytes);
    // No wait: in-order queue serializes subsequent operations.
}

// GPU query: extract one column of data + Merkle auth path, entirely on device.
// Avoids downloading the entire data matrix to CPU.
const char* esimd_query_ffi(void* queue, void* d_out, const void* d_data,
                            const void* d_tree, uint32_t querySize,
                            uint32_t rows, uint32_t cols, uint32_t idx) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        auto* out = static_cast<uint32_t*>(d_out);
        auto* data = static_cast<const uint32_t*>(d_data);
        auto* tree = static_cast<const uint32_t*>(d_tree);

        q->parallel_for(sycl::range<1>(querySize), [=](sycl::id<1> gid) {
            uint32_t i = gid[0];
            if (i < cols) {
                // Data column value: data[idx + col * rows] (column-major)
                out[i] = data[idx + i * rows];
            } else {
                uint32_t i2 = i - cols;
                uint32_t up = i2 / 8;
                uint32_t elem = i2 % 8;
                uint32_t cidx = (idx + rows) >> up;
                uint32_t other = (cidx % 2) ? cidx - 1 : cidx + 1;
                // Tree is Digest[2*rows], each Digest = 8 uint32_t
                out[i] = tree[other * 8 + elem];
            }
        });
        q->wait();
    )
}

// GPU expand: single-pass fused zero+scatter for stride=4 (INV_RATE=4).
// Writes [coeff, 0, 0, 0] blocks in one pass — avoids separate memset.
const char* esimd_batch_expand_ffi(void* queue, void* d_out, const void* d_in,
                                    uint32_t in_rows, uint32_t out_rows,
                                    uint32_t cols, uint32_t exp_po2) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        auto* out = static_cast<uint32_t*>(d_out);
        auto* in_ptr = static_cast<const uint32_t*>(d_in);
        uint32_t stride = 1u << exp_po2;

        if (stride == 4) {
            // Single-pass: each thread writes [coeff, 0, 0, 0] for one input element.
            // No separate memset needed — output is fully written in one pass.
            size_t total = size_t(in_rows) * cols;
            q->parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
                size_t idx = gid[0];
                size_t r = idx % in_rows;
                size_t c = idx / in_rows;
                size_t out_base = r * 4 + c * out_rows;
                out[out_base]     = in_ptr[r + c * in_rows];
                out[out_base + 1] = 0;
                out[out_base + 2] = 0;
                out[out_base + 3] = 0;
            });
        } else {
            // Generic path for other strides
            q->memset(out, 0, (size_t)out_rows * cols * sizeof(uint32_t));
            size_t total = size_t(in_rows) * cols;
            q->parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
                size_t idx = gid[0];
                size_t r = idx % in_rows;
                size_t c = idx / in_rows;
                out[r * stride + c * out_rows] = in_ptr[r + c * in_rows];
            });
        }
        // R2-A05: removed q->wait(). In-order queue serializes the
        // subsequent NTT/poseidon2 kernels with the writes here; host
        // reads of `out` go through esimd_memcpy_dtoh which waits.
    )
}

// Batch forward NTT: process multiple columns with one sync at the end.
// Avoids 700 separate q.wait() calls in batchExpandAndEvaluate.
const char* esimd_batch_forward_ntt(void* queue, void* d_data,
                                     uint32_t lg_n, uint32_t poly_count, uint32_t stride) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        auto* data = static_cast<uint32_t*>(d_data);
        uint32_t n = 1u << lg_n;
        for (uint32_t c = 0; c < poly_count; c++) {
            // Submit NTT without intermediate q.wait() — queue is in-order
            gpu_forward_ntt_no_wait(*q, data + size_t(c) * stride, lg_n);
        }
        // R2-A05: removed q->wait(). All `gpu_forward_ntt_no_wait` submits
        // are serialized by the in-order queue with whatever the caller
        // submits next (or with esimd_memcpy_dtoh for host reads).
    )
}

// Fused LDE: x4 zero-expand + forward NTT in one pass per polynomial, reading
// the 2^(lg_out-2) coefficients directly (no materialised expanded buffer).
// Same result as esimd_batch_expand_ffi(stride 4) + esimd_batch_forward_ntt.
const char* esimd_batch_expand_fwd_fused(void* queue, void* d_out, const void* d_in,
                                         uint32_t lg_out, uint32_t lg_blowup, uint32_t cols) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        if (lg_blowup != 2 || lg_out < 14 || lg_out >= 26) throw std::runtime_error("expand_fwd_fused: unsupported");
        auto* out = static_cast<uint32_t*>(d_out);
        auto* in = static_cast<const uint32_t*>(d_in);
        size_t N = size_t(1) << lg_out; size_t n = N >> 2;
        for (uint32_t c = 0; c < cols; c++)
            gpu_forward_ntt_expand4_no_wait(*q, out + c * N, in + c * n, lg_out);
    )
}

// Batch inverse NTT: same optimization for batchInterpolate.
const char* esimd_batch_inverse_ntt(void* queue, void* d_data,
                                     uint32_t lg_n, uint32_t poly_count, uint32_t stride) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        auto* data = static_cast<uint32_t*>(d_data);
        for (uint32_t c = 0; c < poly_count; c++) {
            gpu_inverse_ntt_no_wait(*q, data + size_t(c) * stride, lg_n);
        }
        // R2-A05: removed q->wait(). Symmetric with esimd_batch_forward_ntt
        // — the in-order queue serializes; host reads self-sync via
        // esimd_memcpy_dtoh.
    )
}

} // close extern "C" for C++ ESIMD code

// ============================================================================
// NTT expand and zk_shift kernels (C++ linkage needed for ESIMD lambdas)
// ============================================================================

#include <sycl/ext/intel/esimd.hpp>
#include "bb31_field.hpp"
#include "combos_gpu.hpp"
#include "ntt_twiddles.hpp"
namespace esimd = sycl::ext::intel::esimd;

// Host-side Montgomery multiply (for precomputing zk_shift table)
static uint32_t host_mont_mul_ffi(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * b;
    uint32_t lo = (uint32_t)prod, hi = (uint32_t)(prod >> 32);
    uint32_t red = lo * bb31::M0;
    uint64_t rprod = (uint64_t)red * bb31::MOD;
    uint32_t rlo = (uint32_t)rprod, rhi = (uint32_t)(rprod >> 32);
    uint32_t carry = ((uint64_t)lo + rlo) >= (1ULL << 32) ? 1 : 0;
    uint32_t res = hi + rhi + carry;
    return res >= bb31::MOD ? res - bb31::MOD : res;
}

static uint32_t host_mont_pow_ffi(uint32_t base, uint32_t exp) {
    uint32_t result = bb31::ONE;
    while (exp > 0) {
        if (exp & 1) result = host_mont_mul_ffi(result, base);
        base = host_mont_mul_ffi(base, base);
        exp >>= 1;
    }
    return result;
}

// Scalar bit-reverse
static uint32_t bit_rev_ffi(uint32_t val, uint32_t nbits) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < nbits; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

// batch_expand (LDE duplication): out[i] = in[i >> lg_blowup]
// Each coefficient is duplicated blowup times. This matches the CPU convention.
// The first lg_blowup NTT butterfly levels are then no-ops on this pattern,
// so a full NTT produces the same result as the CPU's partial NTT.
static void esimd_expand_single(sycl::queue& q, uint32_t* d_out, const uint32_t* d_in,
                                 uint32_t in_size, uint32_t lg_blowup) {
    uint32_t blowup = 1u << lg_blowup;
    uint32_t out_size = in_size << lg_blowup;
    auto* pout = d_out;
    auto* pin = d_in;

    // Each thread processes 16 output elements
    uint32_t num_threads = (out_size + 15) / 16;
    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            if (base + 16 <= out_size) {
                // out[i] = in[i >> lg_blowup]
                esimd::simd<uint32_t, 16> indices(base, 1u);
                indices >>= lg_blowup;
                auto byte_offsets = indices * (uint32_t)sizeof(uint32_t);
                auto vals = esimd::gather<uint32_t, 16>(pin, byte_offsets);
                esimd::block_store(pout + base, vals);
            } else {
                for (uint32_t i = base; i < out_size; i++)
                    *(pout + i) = *(pin + (i >> lg_blowup));
            }
        });
    q.wait();
}

// zk_shift: data[idx] *= 3^bit_rev(pos, bits) for each element
// Single polynomial. Called in a loop for poly_count.
// Precomputes power table on host, uploads, then pointwise multiply.
static void esimd_zk_shift_single(sycl::queue& q, uint32_t* d_data, uint32_t lg_n) {
    uint32_t n = 1u << lg_n;

    // Precompute 3^bit_rev(i, lg_n) for all i on HOST
    // group_gen = 3 in BabyBear (RISC Zero convention)
    uint32_t gen_mont = (uint32_t)(((uint64_t)3 << 32) % bb31::MOD); // Montgomery(3)

    auto* h_powers = sycl::malloc_host<uint32_t>(n, q);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t rev = bit_rev_ffi(i, lg_n);
        h_powers[i] = host_mont_pow_ffi(gen_mont, rev);
    }

    auto* d_powers = sycl::malloc_device<uint32_t>(n, q);
    q.memcpy(d_powers, h_powers, n * sizeof(uint32_t));
    q.wait();

    // Pointwise multiply: data[i] *= powers[i]
    auto* pd = d_data;
    auto* pp = d_powers;
    uint32_t num_threads = (n + 15) / 16;
    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            if (base + 16 <= n) {
                auto vals = esimd::block_load<uint32_t, 16>(pd + base);
                auto pows = esimd::block_load<uint32_t, 16>(pp + base);
                esimd::block_store(pd + base, bb31::mont_mul(vals, pows));
            } else {
                for (uint32_t i = base; i < n; i++) {
                    uint64_t prod = (uint64_t)*(pd + i) * *(pp + i);
                    uint32_t lo = (uint32_t)prod, hi = (uint32_t)(prod >> 32);
                    uint32_t red = lo * bb31::M0;
                    uint64_t rprod = (uint64_t)red * bb31::MOD;
                    uint32_t rlo = (uint32_t)rprod, rhi = (uint32_t)(rprod >> 32);
                    uint32_t carry = ((uint64_t)lo + rlo) >= (1ULL << 32) ? 1 : 0;
                    uint32_t res = hi + rhi + carry;
                    *(pd + i) = res >= bb31::MOD ? res - bb31::MOD : res;
                }
            }
        });
    q.wait();

    sycl::free(d_powers, q);
    sycl::free(h_powers, q);
}

extern "C" { // reopen extern "C" for FFI functions

// ============================================================================
// NTT
// ============================================================================

const char* esimd_forward_ntt(void* queue, void* d_data, uint32_t lg_n) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(gpu_forward_ntt_fast(*q, static_cast<uint32_t*>(d_data), lg_n))
}

const char* esimd_inverse_ntt(void* queue, void* d_data, uint32_t lg_n) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(gpu_inverse_ntt_fast(*q, static_cast<uint32_t*>(d_data), lg_n))
}

const char* esimd_batch_bit_reverse_ffi(void* queue, void* d_io, uint32_t n_bits, uint32_t count) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_batch_bit_reverse(*q, static_cast<uint32_t*>(d_io), n_bits, count); q->wait())
}

// Batch expand: for each of poly_count polynomials, expand from in_size to out_size with zero-fill
const char* esimd_batch_expand(void* queue, void* d_out, const void* d_in,
                                uint32_t lg_domain_size, uint32_t lg_blowup, uint32_t poly_count) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        uint32_t in_size = 1u << lg_domain_size;
        uint32_t out_size = in_size << lg_blowup;
        auto* out = static_cast<uint32_t*>(d_out);
        auto* in_ptr = static_cast<const uint32_t*>(d_in);
        for (uint32_t c = 0; c < poly_count; c++) {
            esimd_expand_single(*q, out + c * out_size,
                                 const_cast<uint32_t*>(in_ptr + c * in_size),
                                 in_size, lg_blowup);
        }
    )
}

// Combined batch_expand + forward_NTT. Output is in NR (bit-reversed) order,
// matching the HAL convention (CPU uses DIF which also produces NR output).
const char* esimd_batch_expand_and_evaluate_ntt(void* queue,
                                                  void* d_out, const void* d_in,
                                                  uint32_t lg_domain_size, uint32_t lg_blowup,
                                                  uint32_t poly_count) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        uint32_t in_size = 1u << lg_domain_size;
        uint32_t out_size = in_size << lg_blowup;
        uint32_t lg_out = lg_domain_size + lg_blowup;
        auto* out = static_cast<uint32_t*>(d_out);
        auto* in_ptr = static_cast<const uint32_t*>(d_in);

        // Step 1: Zero-pad all polynomials (coefficients at 0..in_size-1, zeros after)
        for (uint32_t c = 0; c < poly_count; c++) {
            esimd_expand_single(*q, out + c * out_size,
                                 const_cast<uint32_t*>(in_ptr + c * in_size),
                                 in_size, lg_blowup);
        }

        // Step 2: Forward NTT (CT DIT, NR ordering) per polynomial
        // Output is in bit-reversed (NR) order, matching CPU's DIF convention.
        for (uint32_t c = 0; c < poly_count; c++) {
            gpu_forward_ntt_fast(*q, out + c * out_size, lg_out);
        }
        q->wait();
    )
}

// Cached zk_shift power tables (one per lg_domain_size)
static uint32_t* g_zk_shift_powers[28] = {};
static uint32_t g_zk_shift_sizes[28] = {};

static std::mutex g_zk_shift_mutex;

static uint32_t* ensure_zk_shift_powers(sycl::queue& q, uint32_t lg_n) {
    std::lock_guard<std::mutex> lock(g_zk_shift_mutex);
    if (g_zk_shift_powers[lg_n] && g_zk_shift_sizes[lg_n] == (1u << lg_n))
        return g_zk_shift_powers[lg_n];

    // table[i] = Mont(3^bitrev(i)): the powers 3^k by repeated Montgomery
    // multiplication (O(n), ~5 ms at lg 20 vs ~55 ms for n independent
    // square-and-multiply pows), then the bit-reversal permutation.
    uint32_t n = 1u << lg_n;
    uint32_t gen_mont = (uint32_t)(((uint64_t)3 << 32) % bb31::MOD);
    std::vector<uint32_t> pw(n);
    pw[0] = (uint32_t)(((uint64_t)1 << 32) % bb31::MOD);
    for (uint32_t k = 1; k < n; k++)
        pw[k] = host_mont_mul_ffi(pw[k - 1], gen_mont);
    auto* h_powers = sycl::malloc_host<uint32_t>(n, q);
    for (uint32_t i = 0; i < n; i++)
        h_powers[i] = pw[bit_rev_ffi(i, lg_n)];

    if (g_zk_shift_powers[lg_n]) sycl::free(g_zk_shift_powers[lg_n], q);
    g_zk_shift_powers[lg_n] = sycl::malloc_device<uint32_t>(n, q);
    g_zk_shift_sizes[lg_n] = n;
    q.memcpy(g_zk_shift_powers[lg_n], h_powers, n * sizeof(uint32_t));
    q.wait();
    sycl::free(h_powers, q);
    return g_zk_shift_powers[lg_n];
}

// Build (and cache) the zk_shift power table for lg_n ahead of first use.
const char* esimd_prepare_zk_shift(void* queue, uint32_t lg_n) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        if (lg_n >= 28) throw std::runtime_error("esimd_prepare_zk_shift: lg_n out of range");
        ensure_zk_shift_powers(*q, lg_n);
    )
}

// Batch zk_shift: for each of poly_count polynomials, multiply by 3^bit_rev(pos)
// Power table is CACHED across calls (only depends on lg_domain_size).
const char* esimd_batch_zk_shift(void* queue, void* d_data,
                                   uint32_t lg_domain_size, uint32_t poly_count) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        uint32_t n = 1u << lg_domain_size;
        auto* data = static_cast<uint32_t*>(d_data);

        // Get cached power table (computed once per lg_domain_size)
        auto* d_powers = ensure_zk_shift_powers(*q, lg_domain_size);

        // Apply to each polynomial using the shared power table
        auto* pp = d_powers;
        uint32_t num_threads = (n + 15) / 16;
        for (uint32_t c = 0; c < poly_count; c++) {
            auto* pd = data + c * n;
            q->parallel_for(sycl::range<1>(num_threads),
                [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                    uint32_t base = idx[0] * 16;
                    if (base + 16 <= n) {
                        auto vals = esimd::block_load<uint32_t, 16>(pd + base);
                        auto pows = esimd::block_load<uint32_t, 16>(pp + base);
                        esimd::block_store(pd + base, bb31::mont_mul(vals, pows));
                    }
                });
        }
        // R2-A05: removed q->wait(). The captured kernel-side pointers
        // (`pp`, `pd`) reference device memory that stays valid until the
        // next esimd_free_device; d_powers is cached. The in-order queue
        // serializes subsequent ops.
    )
}

// Batch inverse NTT with zk_shift fused into the final store.
// Replaces esimd_batch_inverse_ntt + esimd_batch_zk_shift back-to-back —
// eliminates one full-bandwidth pointwise-multiply pass per polynomial.
// The zk_shift power table is the same (cached) one as esimd_batch_zk_shift.
const char* esimd_batch_inverse_ntt_zk_shift(void* queue, void* d_data,
                                              uint32_t lg_domain_size,
                                              uint32_t poly_count, uint32_t stride) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        auto* data = static_cast<uint32_t*>(d_data);
        auto* d_powers = ensure_zk_shift_powers(*q, lg_domain_size);
        for (uint32_t c = 0; c < poly_count; c++) {
            gpu_inverse_ntt_zk_shift_no_wait(*q, data + size_t(c) * stride,
                                              lg_domain_size, d_powers);
        }
        // R2-A05: removed q->wait(). Symmetric with esimd_batch_inverse_ntt;
        // d_powers cached, in-order queue handles inter-op ordering, host
        // reads self-sync via esimd_memcpy_dtoh.
    )
}

void gpu_inverse_ntt_zk_shift_from_no_wait(sycl::queue& q, uint32_t* d_dst, const uint32_t* d_src,
                                           uint32_t lg_n, const uint32_t* d_zk_powers);

// Out-of-place esimd_batch_inverse_ntt_zk_shift: dst = INTT_zk(src), src unchanged.
const char* esimd_batch_inverse_ntt_zk_shift_from(void* queue, void* d_dst, const void* d_src,
                                                   uint32_t lg_domain_size,
                                                   uint32_t poly_count, uint32_t stride) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        if (lg_domain_size >= 26) throw std::runtime_error("inverse_ntt_zk_shift_from: lg >= 26 unsupported");
        auto* dst = static_cast<uint32_t*>(d_dst);
        auto* src = static_cast<const uint32_t*>(d_src);
        auto* d_powers = ensure_zk_shift_powers(*q, lg_domain_size);
        for (uint32_t c = 0; c < poly_count; c++) {
            gpu_inverse_ntt_zk_shift_from_no_wait(*q, dst + size_t(c) * stride,
                                                  src + size_t(c) * stride,
                                                  lg_domain_size, d_powers);
        }
    )
}

// ============================================================================
// Poseidon2 Hash
// ============================================================================

const char* esimd_poseidon2_hash_rows(void* queue, void* d_out, const void* d_in,
                                        uint32_t count, uint32_t col_size) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_poseidon2_rows(*q, static_cast<uint32_t*>(d_out),
                                   static_cast<const uint32_t*>(d_in), count, col_size))
}

const char* esimd_poseidon2_hash_fold(void* queue, void* d_out, const void* d_in,
                                        uint32_t num_hashes) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_poseidon2_fold(*q, static_cast<uint32_t*>(d_out),
                                   static_cast<const uint32_t*>(d_in), num_hashes))
}

// ============================================================================
// Poseidon-254 Hash (BN254 field)
// ============================================================================

const char* esimd_poseidon254_hash_rows(void* queue, void* d_out, const void* d_in,
                                          uint32_t row_size, uint32_t col_size) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_poseidon254_rows(*q, static_cast<uint32_t*>(d_out),
                                     static_cast<const uint32_t*>(d_in), row_size, col_size))
}

const char* esimd_poseidon254_hash_fold(void* queue, void* d_out, const void* d_in,
                                          uint32_t num_hashes) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_poseidon254_fold(*q, static_cast<uint32_t*>(d_out),
                                     static_cast<const uint32_t*>(d_in), num_hashes))
}

// ============================================================================
// Element-wise operations
// ============================================================================

const char* esimd_eltwise_add_fp_ffi(void* queue, void* out, const void* x,
                                       const void* y, uint32_t count) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_eltwise_add_fp(*q, static_cast<uint32_t*>(out),
                                   static_cast<const uint32_t*>(x),
                                   static_cast<const uint32_t*>(y), count))
}

const char* esimd_eltwise_copy_fp_ffi(void* queue, void* out, const void* inp, uint32_t count) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_eltwise_copy_fp(*q, static_cast<uint32_t*>(out),
                                    static_cast<const uint32_t*>(inp), count))
}

const char* esimd_eltwise_copy_fp_region_ffi(void* queue, void* into, const void* from,
                                               uint32_t from_rows, uint32_t from_cols,
                                               uint32_t from_offset, uint32_t from_stride,
                                               uint32_t into_offset, uint32_t into_stride) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_eltwise_copy_fp_region(*q, static_cast<uint32_t*>(into),
                                           static_cast<const uint32_t*>(from),
                                           from_rows, from_cols, from_offset, from_stride,
                                           into_offset, into_stride))
}

const char* esimd_eltwise_sum_fpext_ffi(void* queue, void* out, const void* inp,
                                          uint32_t to_add, uint32_t count) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_eltwise_sum_fpext(*q, static_cast<uint32_t*>(out),
                                      static_cast<const uint32_t*>(inp), to_add, count))
}

const char* esimd_eltwise_zeroize_fp_ffi(void* queue, void* elems, uint32_t count) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_eltwise_zeroize_fp(*q, static_cast<uint32_t*>(elems), count))
}

const char* esimd_gather_sample_fp_ffi(void* queue, void* dst, const void* src,
                                         uint32_t idx, uint32_t size, uint32_t stride) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_gather_sample(*q, static_cast<uint32_t*>(dst),
                                  static_cast<const uint32_t*>(src), idx, size, stride))
}

const char* esimd_scatter_fp_ffi(void* queue, void* into, const void* index,
                                   const void* offsets, const void* values, uint32_t count) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_scatter(*q, static_cast<uint32_t*>(into),
                            static_cast<const uint32_t*>(index),
                            static_cast<const uint32_t*>(offsets),
                            static_cast<const uint32_t*>(values), count))
}

// Gather digests from `src` (Digest = 8 u32s) at scattered `indices` into `dst`.
// Replaces N individual D2H get_at calls with one gather kernel + one bulk D2H.
const char* esimd_gather_digests_ffi(void* queue, void* dst, const void* src,
                                       const void* indices, uint32_t count) {
    auto* q = static_cast<sycl::queue*>(queue);
    auto* d = static_cast<uint32_t*>(dst);
    const auto* s = static_cast<const uint32_t*>(src);
    const auto* idx = static_cast<const uint32_t*>(indices);
    FFI_WRAP(
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
                uint32_t src_off = idx[i] * 8u;
                uint32_t dst_off = (uint32_t)i * 8u;
                for (int w = 0; w < 8; ++w) {
                    d[dst_off + w] = s[src_off + w];
                }
            });
        })
    )
}

// ============================================================================
// FRI and polynomial operations
// ============================================================================

const char* esimd_fri_fold_ffi(void* queue, void* d_out, const void* d_in,
                                 const void* d_mix, uint32_t count) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_fri_fold(*q, static_cast<uint32_t*>(d_out),
                              static_cast<const uint32_t*>(d_in),
                              static_cast<const uint32_t*>(d_mix), count))
}

const char* esimd_mix_poly_coeffs_ffi(void* queue, void* d_out, const void* d_in,
                                        const void* d_combos, const void* d_mix_start,
                                        const void* d_mix, uint32_t input_size, uint32_t count) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_mix_poly_coeffs(*q, static_cast<uint32_t*>(d_out),
                                    static_cast<const uint32_t*>(d_in),
                                    static_cast<const uint32_t*>(d_combos),
                                    static_cast<const uint32_t*>(d_mix_start),
                                    static_cast<const uint32_t*>(d_mix),
                                    input_size, count))
}

const char* esimd_batch_evaluate_any_ffi(void* queue, void* d_out, const void* d_coeffs,
                                           const void* d_which, const void* d_xs,
                                           uint32_t count, uint32_t deg) {
    auto* q = static_cast<sycl::queue*>(queue);
    // R2-A04 opt-in: RISC0_BATCH_EVAL_CUDA_STYLE=1 selects the CUDA-style
    // 1-WG-per-eval Horner kernel. Default OFF on B70 (BMG-G31) because the
    // CUDA-style kernel measured parity with ESIMD in wall time AND introduced
    // intermittent SIGABRT (3/5 runs vs 0/5 with ESIMD). Kept as opt-in for
    // future arches where the trade-off may flip.
    const char* mode = std::getenv("RISC0_BATCH_EVAL_CUDA_STYLE");
    bool use_cuda_style = mode && (mode[0] == '1' || (mode[0] == 'o' && mode[1] == 'n'));
    if (use_cuda_style) {
        FFI_WRAP(esimd_batch_evaluate_any_cuda_style(*q, static_cast<uint32_t*>(d_out),
                                                      static_cast<const uint32_t*>(d_coeffs),
                                                      static_cast<const uint32_t*>(d_which),
                                                      static_cast<const uint32_t*>(d_xs), count, deg))
    } else {
        FFI_WRAP(esimd_batch_evaluate_any(*q, static_cast<uint32_t*>(d_out),
                                           static_cast<const uint32_t*>(d_coeffs),
                                           static_cast<const uint32_t*>(d_which),
                                           static_cast<const uint32_t*>(d_xs), count, deg))
    }
}

const char* esimd_combos_prepare_ffi(void* queue, void* d_combos, const void* d_coeff_u,
                                       uint32_t combo_count, uint32_t cycles,
                                       uint32_t regs_count, const void* d_reg_sizes,
                                       const void* d_reg_combo_ids,
                                       uint32_t check_size, const void* d_mix) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(esimd_combos_prepare(*q, static_cast<uint32_t*>(d_combos),
                                   static_cast<const uint32_t*>(d_coeff_u),
                                   combo_count, cycles, regs_count,
                                   static_cast<const uint32_t*>(d_reg_sizes),
                                   static_cast<const uint32_t*>(d_reg_combo_ids),
                                   check_size, static_cast<const uint32_t*>(d_mix)))
}

// ============================================================================
// GPU combos divide + finalize (avoid expensive CPU fallback + PCIe transfers)
// ============================================================================

// BabyBear scalar arithmetic for SYCL kernels (non-ESIMD)
namespace bb31_sycl {
    static constexpr uint32_t P = 0x78000001;
    static constexpr uint32_t M = 0x88000001;
    static constexpr uint32_t NBETA = 0x40000018;

    static inline uint32_t fp_add(uint32_t a, uint32_t b) {
        uint32_t r = a + b;
        return (r >= P ? r - P : r);
    }
    static inline uint32_t fp_sub(uint32_t a, uint32_t b) {
        uint32_t r = a - b;
        return (r > P ? r + P : r);
    }
    static inline uint32_t fp_mul(uint32_t a, uint32_t b) {
        uint64_t o64 = uint64_t(a) * uint64_t(b);
        uint32_t low = -uint32_t(o64);
        uint32_t red = M * low;
        o64 += uint64_t(red) * uint64_t(P);
        uint32_t ret = uint32_t(o64 >> 32);
        return (ret >= P ? ret - P : ret);
    }

    // FpExt = 4 × Fp stored as consecutive uint32_t[4]
    struct FpExt4 {
        uint32_t e[4];
    };

    static inline FpExt4 fpext_add(FpExt4 a, FpExt4 b) {
        return {fp_add(a.e[0], b.e[0]), fp_add(a.e[1], b.e[1]),
                fp_add(a.e[2], b.e[2]), fp_add(a.e[3], b.e[3])};
    }

    // FpExt * FpExt (schoolbook with NBETA)
    static inline FpExt4 fpext_mul(FpExt4 a, FpExt4 b) {
        uint32_t nb = NBETA;
        return {
            fp_add(fp_mul(a.e[0], b.e[0]),
                   fp_mul(nb, fp_add(fp_add(fp_mul(a.e[1], b.e[3]),
                                             fp_mul(a.e[2], b.e[2])),
                                     fp_mul(a.e[3], b.e[1])))),
            fp_add(fp_add(fp_mul(a.e[0], b.e[1]),
                          fp_mul(a.e[1], b.e[0])),
                   fp_mul(nb, fp_add(fp_mul(a.e[2], b.e[3]),
                                     fp_mul(a.e[3], b.e[2])))),
            fp_add(fp_add(fp_add(fp_mul(a.e[0], b.e[2]),
                                 fp_mul(a.e[1], b.e[1])),
                          fp_mul(a.e[2], b.e[0])),
                   fp_mul(nb, fp_mul(a.e[3], b.e[3]))),
            fp_add(fp_add(fp_add(fp_mul(a.e[0], b.e[3]),
                                 fp_mul(a.e[1], b.e[2])),
                          fp_mul(a.e[2], b.e[1])),
                   fp_mul(a.e[3], b.e[0]))
        };
    }
}

// GPU polynomial division: divide combos[:, comboId] by (x - z) for each DivideInfo entry.
// IMPORTANT: Multiple DivideInfo entries may share the same comboId (different z values).
// These MUST be processed sequentially since each division modifies the column in-place
// and the next division operates on the result. We launch one GPU kernel per division,
// serialized via the in-order queue.
//
// Memory layout: HalMatrix<FpExt> is column-major with FpExt = 4 consecutive Fp.
//   combos(i, j).elems[k] = raw[j * rows * 4 + i * 4 + k]
const char* esimd_combos_divide_ffi(void* queue, void* d_combos,
                                     uint32_t rows, uint32_t combos_cols,
                                     const void* d_info, uint32_t info_count,
                                     void* d_remainders) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        auto* combos = static_cast<uint32_t*>(d_combos);
        auto* info = static_cast<const uint32_t*>(d_info);
        auto* remainders = static_cast<uint32_t*>(d_remainders);
        uint32_t col_stride = rows * 4;

        // Each DivideInfo entry divides one combo column by (x - z). The CPU
        // caller (prover.rs) batches multiple back values per combo, so multiple
        // entries may share the same comboId and must be serialized for that combo.
        //
        // Launch one work-item per division. Within a single work-item we process
        // ALL entries for that comboId sequentially (since they share the column),
        // not just one entry. This requires the CPU caller to group entries by combo.
        //
        // Caller's chunks list iterates combo_count first, so entries are already
        // in groups: [(combo0, pows...), (combo1, pows...), ...]
        // We can process each combo in parallel — each gets a work item that
        // sequentially does all its divisions.

        // The prover.rs builds chunks as: for combo_id 0..combo_count, push
        // (i, pows). Then info entries are flattened: for each (i, pows), push
        // info[k] = (i, z) for each z in pows.
        // This means info entries are sorted by comboId first, then by pow index.
        // So we can find groups by detecting comboId changes.

        // Build group offsets on host (info is device-side, but small)
        std::vector<uint32_t> h_info(info_count * 5);
        q->memcpy(h_info.data(), info, info_count * 5 * sizeof(uint32_t)).wait();

        // Find unique comboIds and their entry ranges
        struct ComboGroup { uint32_t comboId; uint32_t start; uint32_t count; };
        std::vector<ComboGroup> groups;
        for (uint32_t k = 0; k < info_count;) {
            uint32_t cid = h_info[k * 5];
            uint32_t end = k + 1;
            while (end < info_count && h_info[end * 5] == cid) end++;
            groups.push_back({cid, k, end - k});
            k = end;
        }

        // Upload group descriptors to device
        uint32_t n_groups = groups.size();
        uint32_t* d_groups = sycl::malloc_device<uint32_t>(n_groups * 3, *q);
        std::vector<uint32_t> h_groups_flat(n_groups * 3);
        for (uint32_t i = 0; i < n_groups; i++) {
            h_groups_flat[i * 3 + 0] = groups[i].comboId;
            h_groups_flat[i * 3 + 1] = groups[i].start;
            h_groups_flat[i * 3 + 2] = groups[i].count;
        }
        q->memcpy(d_groups, h_groups_flat.data(), n_groups * 3 * sizeof(uint32_t)).wait();

        // Launch one work-item per group; each processes its comboId's divisions sequentially.
        q->parallel_for(sycl::range<1>(n_groups), [=](sycl::id<1> gid) {
            uint32_t g = gid[0];
            uint32_t comboId = d_groups[g * 3 + 0];
            uint32_t start = d_groups[g * 3 + 1];
            uint32_t count = d_groups[g * 3 + 2];
            uint32_t* col = combos + comboId * col_stride;

            for (uint32_t e = 0; e < count; e++) {
                uint32_t k = start + e;
                uint32_t off = k * 5;
                bb31_sycl::FpExt4 z = {info[off + 1], info[off + 2],
                                         info[off + 3], info[off + 4]};

                bb31_sycl::FpExt4 cur = {0, 0, 0, 0};
                for (uint32_t ii = rows; ii-- > 0;) {
                    uint32_t base = ii * 4;
                    bb31_sycl::FpExt4 coeff = {col[base], col[base+1],
                                                 col[base+2], col[base+3]};
                    bb31_sycl::FpExt4 next = bb31_sycl::fpext_add(
                        bb31_sycl::fpext_mul(z, cur), coeff);
                    col[base]   = cur.e[0];
                    col[base+1] = cur.e[1];
                    col[base+2] = cur.e[2];
                    col[base+3] = cur.e[3];
                    cur = next;
                }
                remainders[k * 4]     = cur.e[0];
                remainders[k * 4 + 1] = cur.e[1];
                remainders[k * 4 + 2] = cur.e[2];
                remainders[k * 4 + 3] = cur.e[3];
            }
        }).wait();

        sycl::free(d_groups, *q);
    )
}

// GPU combos finalize: sum FpExt columns per row, extract to Fp matrix.
//   combos: FpExt[rows × combos_cols], layout combos(i,j).e[k] = raw[j*rows*4 + i*4 + k]
//   out: Fp[rows × 4] column-major, out(i,j) = raw[j*rows + i]
// combos_divide_v2: divide each combo column by (x - z) for every (col, z)
// entry, in entry order per column, on the device (blocked synthetic
// division, batched per round). h_info: host array of info_count x
// {col, z0, z1, z2, z3} (z raw Montgomery words). h_rem: host array of
// info_count x 4 words receiving the remainders. Synchronous. Returns an error
// if any remainder is nonzero (same contract as the CPU path's assert_eq!).
const char* esimd_combos_divide_v2_ffi(void* queue, void* d_combos, uint32_t rows, uint32_t ncols,
                                       const void* h_info, uint32_t info_count, void* h_rem) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        const uint32_t* in = static_cast<const uint32_t*>(h_info);
        std::vector<combos_gpu::DivEntry> info(info_count);
        for (uint32_t k = 0; k < info_count; k++) {
            info[k].col = in[5 * k];
            for (int c = 0; c < 4; c++) info[k].z.e[c] = in[5 * k + 1 + c];
        }
        std::vector<combos_gpu::E4> rem;
        std::string err = combos_gpu::combos_divide_v2(*q, static_cast<uint32_t*>(d_combos), rows, ncols,
                                                       info, rem, nullptr);
        if (h_rem && !rem.empty()) std::memcpy(h_rem, rem.data(), rem.size() * sizeof(combos_gpu::E4));
        if (!err.empty()) throw std::runtime_error(err);
    )
}

const char* esimd_combos_finalize_ffi(void* queue, void* d_out, const void* d_combos,
                                       uint32_t rows, uint32_t combos_cols) {
    auto* q = static_cast<sycl::queue*>(queue);
    FFI_WRAP(
        auto* out = static_cast<uint32_t*>(d_out);
        auto* combos = static_cast<const uint32_t*>(d_combos);
        uint32_t col_stride = rows * 4;

        q->parallel_for(sycl::range<1>(rows), [=](sycl::id<1> gid) {
            uint32_t i = gid[0];
            bb31_sycl::FpExt4 tot = {0, 0, 0, 0};
            for (uint32_t j = 0; j < combos_cols; j++) {
                const uint32_t* col = combos + j * col_stride;
                uint32_t base = i * 4;
                bb31_sycl::FpExt4 val = {col[base], col[base+1],
                                           col[base+2], col[base+3]};
                tot = bb31_sycl::fpext_add(tot, val);
            }
            // Output is Fp[rows × 4] column-major: out(i, k) = out[k * rows + i]
            out[i]            = tot.e[0];
            out[rows + i]     = tot.e[1];
            out[2*rows + i]   = tot.e[2];
            out[3*rows + i]   = tot.e[3];
        });
        q->wait();
    )
}

// Pre-warm GPU caches and SYCL runtime to eliminate first-segment overhead.
void esimd_warmup(void* queue, uint32_t max_lg_n) {
    auto* q = static_cast<sycl::queue*>(queue);
    // Allocate a small buffer for warmup operations
    uint32_t n = 1u << 14;  // 16K elements — triggers twiddle computation for lg=14
    auto* d = sycl::malloc_device<uint32_t>(n, *q);
    q->memset(d, 0, n * 4);
    q->wait();

    // Trigger twiddle precomputation by running actual NTTs at target sizes
    // Forward NTT at max_lg_n triggers twiddle computation for that size and all smaller
    if (max_lg_n >= 4) {
        // Run a small forward NTT to warm up twiddle cache
        gpu_forward_ntt_fast(*q, d, 14);
        // Run a small inverse NTT
        gpu_inverse_ntt_fast(*q, d, 14);
    }

    // Trigger poseidon2 constant initialization (ESIMD path only; see
    // esimd_poseidon2_rows_esimd).
    esimd_poseidon2_rows_esimd(*q, d, d, 1, 16);

    // Warm up poseidon2 fold (Merkle tree folding)
    // fold needs in[16*num_hashes], out[8*num_hashes]; with num_hashes=1 fits in d
    esimd_poseidon2_fold(*q, d, d, 1);

    // Warm up eltwise kernels
    esimd_eltwise_zeroize_fp(*q, d, 16);
    esimd_eltwise_copy_fp(*q, d, d, 16);

    // Warm up batch bit-reverse (nBits=4 => 16-element permutation, count=1)
    esimd_batch_bit_reverse(*q, d, 4, 1);

    // Warm up batch expand (stride=4 fast path used by esimd_batch_expand_ffi)
    // in_rows=16, out_rows=64, cols=1, exp_po2=2 (stride=4)
    {
        auto* d2 = sycl::malloc_device<uint32_t>(64, *q);
        q->memset(d2, 0, 64 * 4);
        q->wait();
        esimd_batch_expand_ffi(queue, d2, d, 16, 64, 1, 2);
        q->wait();  // sycl::free does not wait for kernels using the buffer
        sycl::free(d2, *q);
    }

    // Warm up batch zk_shift (caches power table + JITs pointwise multiply kernel)
    // lg_domain_size=4 => 16 elements, poly_count=1
    esimd_batch_zk_shift(queue, d, 4, 1);

    // Warm up FRI fold: out[4*count], in[8*count], mix[4]; count=1
    {
        auto* d_mix4 = sycl::malloc_device<uint32_t>(4, *q);
        q->memset(d_mix4, 0, 4 * 4);
        q->wait();
        esimd_fri_fold(*q, d, d, d_mix4, 1);
        q->wait();  // sycl::free does not wait for kernels using the buffer
        sycl::free(d_mix4, *q);
    }

    // Warm up mix_poly_coeffs: count=1, inputSize=1
    {
        auto* d_scratch = sycl::malloc_device<uint32_t>(16, *q);
        q->memset(d_scratch, 0, 16 * 4);
        q->wait();
        esimd_mix_poly_coeffs(*q, d, d_scratch, d_scratch, d_scratch, d_scratch, 1, 1);
        q->wait();  // sycl::free does not wait for kernels using the buffer
        sycl::free(d_scratch, *q);
    }

    // Warm up batch_evaluate_any: count=1, deg=1
    {
        auto* d_eval = sycl::malloc_device<uint32_t>(16, *q);
        q->memset(d_eval, 0, 16 * 4);
        q->wait();
        esimd_batch_evaluate_any(*q, d, d_eval, d_eval, d_eval, 1, 1);
        q->wait();  // sycl::free does not wait for kernels using the buffer
        sycl::free(d_eval, *q);
    }

    // Now pre-compute twiddles for the actual target sizes by running NTTs
    // The twiddle cache expands lazily, so running at lg=max_lg_n covers all
    if (max_lg_n > 14) {
        auto* d_big = sycl::malloc_device<uint32_t>(1u << max_lg_n, *q);
        q->memset(d_big, 0, (1u << max_lg_n) * 4);
        q->wait();
        gpu_forward_ntt_fast(*q, d_big, max_lg_n);
        gpu_inverse_ntt_fast(*q, d_big, max_lg_n);
        q->wait();  // sycl::free does not wait for kernels using the buffer
        sycl::free(d_big, *q);
    }

    q->wait();  // sycl::free does not wait for kernels using the buffer

    sycl::free(d, *q);
}

} // extern "C"
