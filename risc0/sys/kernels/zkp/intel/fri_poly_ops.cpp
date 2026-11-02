// FRI and polynomial HAL kernels for Intel GPU STARK prover.
// Implements: fri_fold, mix_poly_coeffs, batch_evaluate_any, combos_prepare, fri_prove.
// Uses the "4 × Vec16" FpExt layout (each lane = independent value).

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include "bb31_field.hpp"
#include "bb31_fpext4.hpp"
#include "combos_gpu.hpp"
#include <array>
#include <map>
#include <vector>
#include <stdexcept>
#include <mutex>

namespace esimd = sycl::ext::intel::esimd;
using namespace fpext4;

static constexpr uint32_t kFriFold = 16;
static constexpr uint32_t kFriFoldBits = 4; // log2(kFriFold)

// Bit-reverse a value within nbits bits (scalar)
static inline uint32_t bit_rev_scalar(uint32_t val, uint32_t nbits) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < nbits; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

static sycl::queue create_fri_queue() {
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

// Forward declarations for scalar FpExt helpers (defined later, used by fri_fold precomputation)
static inline void scalar_ext_mul(uint32_t* r, const uint32_t* a, const uint32_t* b);

// ============================================================================
// fri_fold: FRI folding — accumulate kFriFold=16 FpExt values with mix powers
// Input: planar FpExt layout: in[comp * count * kFriFold + rev_i * count + idx]
// Output: planar FpExt layout: out[comp * count + idx]
// Each thread processes one output index.
// ============================================================================
void esimd_fri_fold(sycl::queue& q,
                     uint32_t* d_out, const uint32_t* d_in,
                     const uint32_t* d_mix, // 4 u32 values = 1 FpExt (Montgomery)
                     uint32_t count) {
    // Precompute mix^0..mix^15 on HOST to break the serial dependency chain
    // in the kernel's inner loop. This eliminates 16 fpext_mul(curMix, mix) calls.
    uint32_t h_mix[4];
    auto* h_tmp = sycl::malloc_host<uint32_t>(4, q);
    q.memcpy(h_tmp, d_mix, 16); q.wait();
    for (int c = 0; c < 4; c++) h_mix[c] = h_tmp[c];
    sycl::free(h_tmp, q);

    uint32_t h_powers[kFriFold * 4]; // 16 FpExt values = 256 bytes
    // mix^0 = [ONE, 0, 0, 0]
    h_powers[0] = bb31::ONE; h_powers[1] = 0; h_powers[2] = 0; h_powers[3] = 0;
    // mix^1..mix^15 via scalar multiply
    for (uint32_t i = 1; i < kFriFold; i++) {
        uint32_t prev[4] = {h_powers[(i-1)*4], h_powers[(i-1)*4+1], h_powers[(i-1)*4+2], h_powers[(i-1)*4+3]};
        scalar_ext_mul(h_powers + i*4, prev, h_mix);
    }

    // The kernel walks the input rows in order, row r holding the value that mix^bitrev(r)
    // multiplies, so hand it the powers in that order: h_powers_by_row[r] = mix^bitrev(r).
    // (Indexing the rows by bitrev(fi) in the kernel instead, as before, makes the Windows
    // driver's IGC (32.0.101.8976) fail on the unrolled loop with "Internal Compiler Error:
    // Divide by zero" or "Access violation". The sum is the same either way.)
    uint32_t h_powers_by_row[kFriFold * 4];
    for (uint32_t r = 0; r < kFriFold; r++) {
        uint32_t fi = bit_rev_scalar(r, kFriFoldBits);
        for (int c = 0; c < 4; c++) h_powers_by_row[r * 4 + c] = h_powers[fi * 4 + c];
    }

    auto* d_powers = sycl::malloc_device<uint32_t>(kFriFold * 4, q);
    q.memcpy(d_powers, h_powers_by_row, kFriFold * 4 * 4); q.wait();

    auto* pout = d_out;
    auto* pin = d_in;
    auto* ppow = d_powers;

    // SIMD16: each thread processes 16 consecutive output indices.
    // Mix powers precomputed — no serial dependency in inner loop.
    uint32_t num_threads = (count + 15) / 16;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;

            FpExt4 tot;     // zero-initialized

            // fi is the input row here; ppow[fi] is the power that row's value takes.
            for (uint32_t fi = 0; fi < kFriFold; fi++) {
                uint32_t rev_base = fi * count + base;

                // Load the precomputed power for row fi (broadcast to all 16 lanes)
                FpExt4 curMix(bb31::Vec16(*(ppow + fi*4 + 0)),
                              bb31::Vec16(*(ppow + fi*4 + 1)),
                              bb31::Vec16(*(ppow + fi*4 + 2)),
                              bb31::Vec16(*(ppow + fi*4 + 3)));

                FpExt4 factor;
                if (base + 16 <= count) {
                    factor.c[0] = esimd::block_load<uint32_t, 16>(pin + 0*count*kFriFold + rev_base);
                    factor.c[1] = esimd::block_load<uint32_t, 16>(pin + 1*count*kFriFold + rev_base);
                    factor.c[2] = esimd::block_load<uint32_t, 16>(pin + 2*count*kFriFold + rev_base);
                    factor.c[3] = esimd::block_load<uint32_t, 16>(pin + 3*count*kFriFold + rev_base);
                } else {
                    factor = FpExt4();
                    for (uint32_t lane = 0; lane < 16 && base + lane < count; lane++) {
                        uint32_t ri = fi * count + base + lane;
                        factor.c[0][lane] = *(pin + 0*count*kFriFold + ri);
                        factor.c[1][lane] = *(pin + 1*count*kFriFold + ri);
                        factor.c[2][lane] = *(pin + 2*count*kFriFold + ri);
                        factor.c[3][lane] = *(pin + 3*count*kFriFold + ri);
                    }
                }

                // No serial curMix *= mix needed — mix^fi is precomputed!
                tot = fpext_add(tot, fpext_mul(curMix, factor));
            }

            // Store results: block_store for 16 consecutive planar values
            if (base + 16 <= count) {
                esimd::block_store(pout + 0*count + base, tot.c[0]);
                esimd::block_store(pout + 1*count + base, tot.c[1]);
                esimd::block_store(pout + 2*count + base, tot.c[2]);
                esimd::block_store(pout + 3*count + base, tot.c[3]);
            } else {
                for (uint32_t lane = 0; lane < 16 && base + lane < count; lane++) {
                    *(pout + 0*count + base + lane) = tot.c[0][lane];
                    *(pout + 1*count + base + lane) = tot.c[1][lane];
                    *(pout + 2*count + base + lane) = tot.c[2][lane];
                    *(pout + 3*count + base + lane) = tot.c[3][lane];
                }
            }
        });
    q.wait();
    sycl::free(d_powers, q);
}

// ============================================================================
// mix_poly_coeffs: Mix polynomial coefficients into combo groups
// For each coefficient idx, iterate over inputSize polynomials,
// accumulating cur * in[count*i + idx] into out[count*combos[i] + idx]
// ============================================================================
void esimd_mix_poly_coeffs(sycl::queue& q,
                            uint32_t* d_out, const uint32_t* d_in, const uint32_t* d_combos,
                            const uint32_t* d_mix_start, const uint32_t* d_mix,
                            uint32_t inputSize, uint32_t count) {
    // combos-v2: power table + register accumulation (one RMW per combo run).
    if (inputSize == 0 || count == 0) return;
    uint32_t* pw = sycl::malloc_device<uint32_t>(size_t(inputSize) * 4, q);
    combos_gpu::mix_pw_table(q, pw, d_mix_start, d_mix, inputSize);
    if (count % 4 == 0)
        combos_gpu::mix_poly_coeffs_v3(q, d_out, d_in, d_combos, pw, inputSize, count, nullptr, 0);
    else
        combos_gpu::mix_poly_coeffs_v2(q, d_out, d_in, d_combos, pw, inputSize, count, nullptr, 0);
    q.wait();
    sycl::free(pw, q);
}

// ============================================================================
// combos_prepare: Single-threaded combo polynomial subtraction
// Subtracts U polynomial contributions from combo polynomial coefficients.
// ============================================================================
// combos_prepare: runs on HOST (single-threaded, same as CUDA <<<1,1>>>)
// Uses USM shared memory so host can directly access device data
void esimd_combos_prepare(sycl::queue& q,
                           uint32_t* d_combos, const uint32_t* d_coeffU,
                           uint32_t comboCount, uint32_t cycles, uint32_t regsCount,
                           const uint32_t* d_regSizes, const uint32_t* d_regComboIds,
                           uint32_t checkSize, const uint32_t* d_mix) {
    // combos-v2: device-side U subtraction (no 2x(comboCount+1)*cycles*16 B host round trip).
    // Async on the in-order queue: d_coeffU/d_regSizes/d_regComboIds/d_mix are pool
    // buffers the caller drops on return, so keep the wait (cost: one queue drain).
    combos_gpu::combos_prepare_v2(q, d_combos, d_coeffU, comboCount, cycles, regsCount,
                                  d_regSizes, d_regComboIds, checkSize, d_mix, nullptr);
    q.wait();
}

// ============================================================================
// Scalar FpExt helpers for single-thread kernels (batch_evaluate_any, combos_prepare)
// ============================================================================
static inline uint32_t scalar_mont_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * b;
    uint32_t lo = (uint32_t)prod, hi = (uint32_t)(prod >> 32);
    uint32_t red = lo * 0x77ffffffu;
    uint64_t rprod = (uint64_t)red * bb31::MOD;
    uint32_t rlo = (uint32_t)rprod, rhi = (uint32_t)(rprod >> 32);
    uint32_t carry = ((uint64_t)lo + rlo) >= (1ULL << 32) ? 1 : 0;
    uint32_t res = hi + rhi + carry;
    return res >= bb31::MOD ? res - bb31::MOD : res;
}
static inline uint32_t scalar_field_add(uint32_t a, uint32_t b) {
    uint32_t r = a + b; return r >= bb31::MOD ? r - bb31::MOD : r;
}
static inline void scalar_ext_mul(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    uint32_t nb = bb31::NBETA;
    r[0] = scalar_field_add(scalar_mont_mul(a[0],b[0]), scalar_mont_mul(nb, scalar_field_add(scalar_field_add(scalar_mont_mul(a[1],b[3]),scalar_mont_mul(a[2],b[2])),scalar_mont_mul(a[3],b[1]))));
    r[1] = scalar_field_add(scalar_field_add(scalar_mont_mul(a[0],b[1]),scalar_mont_mul(a[1],b[0])), scalar_mont_mul(nb, scalar_field_add(scalar_mont_mul(a[2],b[3]),scalar_mont_mul(a[3],b[2]))));
    r[2] = scalar_field_add(scalar_field_add(scalar_field_add(scalar_mont_mul(a[0],b[2]),scalar_mont_mul(a[1],b[1])),scalar_mont_mul(a[2],b[0])), scalar_mont_mul(nb, scalar_mont_mul(a[3],b[3])));
    r[3] = scalar_field_add(scalar_field_add(scalar_field_add(scalar_mont_mul(a[0],b[3]),scalar_mont_mul(a[1],b[2])),scalar_mont_mul(a[2],b[1])), scalar_mont_mul(a[3],b[0]));
}
static inline void scalar_ext_pow(uint32_t* result, const uint32_t* base_in, uint32_t n) {
    uint32_t b[4], tmp[4];
    for (int c=0;c<4;c++) { result[c]=(c==0?bb31::ONE:0); b[c]=base_in[c]; }
    while (n > 0) {
        if (n & 1) { scalar_ext_mul(tmp, result, b); for (int c=0;c<4;c++) result[c]=tmp[c]; }
        scalar_ext_mul(tmp, b, b); for (int c=0;c<4;c++) b[c]=tmp[c];
        n >>= 1;
    }
}

// ============================================================================
// batch_evaluate_any: Evaluate polynomials at arbitrary FpExt points
// SIMD16: each workgroup evaluates 16 polynomials simultaneously (one per lane).
// 64 threads per WG, each thread strides through coefficients.
// SLM tree reduction across 64 threads, with FpExt4 (Vec16) values.
// ============================================================================
void esimd_batch_evaluate_any_orig(sycl::queue& q,
                               uint32_t* d_out,       // FpExt output (4 u32 per eval, AoS)
                               const uint32_t* d_coeffs, // Fp coefficients (flat)
                               const uint32_t* d_which,  // polynomial index per eval
                               const uint32_t* d_xs,     // FpExt eval points (4 u32 per point, AoS)
                               uint32_t count,            // number of evaluations
                               uint32_t deg) {            // degree (coefficients per poly)
    if (count == 0) return;

    // WG_SIZE is runtime-tunable via RISC0_BATCH_EVAL_WG (default 64).
    // Empirical: WG=64 is the production sweet spot on B70. WG=32 is ~3%
    // slower; WG≥128 *hangs* the kernel (SLM/barrier resource issue not
    // diagnosed). SLM is sized for the safe max we'll allow at runtime.
    constexpr uint32_t WG_SIZE_MAX = 64;
    constexpr uint32_t SLM_SIZE = WG_SIZE_MAX * 4 * 16 * sizeof(uint32_t);
    uint32_t WG_SIZE = 64;
    if (const char* s = std::getenv("RISC0_BATCH_EVAL_WG")) {
        int v = std::atoi(s);
        if (v == 16 || v == 32 || v == 64) {
            WG_SIZE = (uint32_t)v;
        }
    }

    // Each workgroup handles 16 evaluations. Total WGs = ceil(count/16).
    uint32_t num_groups = (count + 15) / 16;

    auto* pout = d_out;
    auto* pcoeffs = d_coeffs;
    auto* pwhich = d_which;
    auto* pxs = d_xs;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
                esimd::slm_init<SLM_SIZE>();
                uint32_t lid = item.get_local_id(0);
                uint32_t grp = item.get_group(0);
                uint32_t eval_base = grp * 16; // first of 16 evaluations for this WG

                // Load which[] and xs[] for 16 evaluations (one per lane)
                // Gather: each lane reads from a different evaluation's data
                esimd::simd<uint32_t, 16> lane(0u, 1u); // {0,1,...,15}
                esimd::simd<uint32_t, 16> eval_idx = eval_base + lane;

                // Build per-lane polynomial base offset: which[eval_idx] * deg
                // Guard: lanes beyond count get poly_idx=0 (harmless, result discarded)
                esimd::simd<uint32_t, 16> poly_off;
                {
                    auto which_byte = eval_idx * (uint32_t)sizeof(uint32_t);
                    auto valid = eval_idx < count;
                    auto which_vals = esimd::gather<uint32_t, 16>(pwhich, which_byte, valid,
                                                                   esimd::simd<uint32_t, 16>(0u));
                    poly_off = which_vals * deg;
                }

                // Load 16 FpExt evaluation points (AoS: 4 contiguous u32 per eval)
                FpExt4 x_val;
                {
                    auto xs_byte = eval_idx * 4 * (uint32_t)sizeof(uint32_t);
                    auto valid = eval_idx < count;
                    x_val.c[0] = esimd::gather<uint32_t, 16>(pxs, xs_byte, valid, esimd::simd<uint32_t, 16>(0u));
                    x_val.c[1] = esimd::gather<uint32_t, 16>(pxs, xs_byte + 4, valid, esimd::simd<uint32_t, 16>(0u));
                    x_val.c[2] = esimd::gather<uint32_t, 16>(pxs, xs_byte + 8, valid, esimd::simd<uint32_t, 16>(0u));
                    x_val.c[3] = esimd::gather<uint32_t, 16>(pxs, xs_byte + 12, valid, esimd::simd<uint32_t, 16>(0u));
                }

                // Compute stepx = x^WG_SIZE and powx = x^lid (per-lane, independent)
                FpExt4 stepx = fpext_pow(x_val, WG_SIZE);
                FpExt4 powx = fpext_pow(x_val, lid);

                // Strided polynomial evaluation: each thread processes coeff[lid, lid+64, lid+128, ...]
                FpExt4 tot; // zero-initialized
                for (uint32_t i = lid; i < deg; i += WG_SIZE) {
                    // Gather coefficient from 16 different polynomials
                    // coeff[lane] = pcoeffs[poly_off[lane] + i]
                    auto coeff_byte = (poly_off + i) * (uint32_t)sizeof(uint32_t);
                    auto coeff = esimd::gather<uint32_t, 16>(pcoeffs, coeff_byte);

                    // FpExt * Fp: powx * coeff (component-wise)
                    FpExt4 term = fpext_mul_fp(powx, coeff);
                    tot = fpext_add(tot, term);

                    // Advance power: powx *= stepx
                    powx = fpext_mul(powx, stepx);
                }

                // Store FpExt4 to SLM: each thread writes 4 Vec16 values
                // Layout: SLM[lid * 4 * 16 * 4 + c * 16 * 4 + lane * 4]
                // Simplified: thread lid, component c → SLM offset (lid * 64 + c * 16) * 4 bytes
                uint32_t slm_base = lid * 64; // in u32 units: lid * 4 components * 16 lanes
                for (int c = 0; c < 4; c++)
                    esimd::slm_block_store<uint32_t, 16>((slm_base + c * 16) * 4, tot.c[c]);
                esimd::barrier();

                // Tree reduction in SLM (64 threads → 1)
                for (uint32_t stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        uint32_t my_base = lid * 64;
                        uint32_t partner_base = (lid + stride) * 64;
                        for (int c = 0; c < 4; c++) {
                            auto a = esimd::slm_block_load<uint32_t, 16>((my_base + c * 16) * 4);
                            auto b = esimd::slm_block_load<uint32_t, 16>((partner_base + c * 16) * 4);
                            esimd::slm_block_store<uint32_t, 16>((my_base + c * 16) * 4,
                                                                  bb31::field_add(a, b));
                        }
                    }
                    esimd::barrier();
                }

                // Thread 0 writes 16 evaluation results (AoS: 4 contiguous u32 per eval)
                if (lid == 0) {
                    FpExt4 result;
                    for (int c = 0; c < 4; c++)
                        result.c[c] = esimd::slm_block_load<uint32_t, 16>(c * 16 * 4);

                    // Scatter to AoS output: out[eval_idx * 4 + c]
                    auto valid = eval_idx < count;
                    for (int c = 0; c < 4; c++) {
                        auto out_byte = (eval_idx * 4 + c) * (uint32_t)sizeof(uint32_t);
                        esimd::scatter<uint32_t, 16>(pout, out_byte, result.c[c], valid);
                    }
                }
            });
    });
    q.wait();
}

// ============================================================================
// R2-A04: CUDA-style 1-WG-per-eval batch_evaluate_any (Horner reduction)
//
// Mirrors risc0/sys/kernels/zkp/cuda/kernels.cu:batch_evaluate_any.
// One work-group per polynomial evaluation, WG_SIZE threads per Horner
// reduction. Coalesced contiguous coefficient reads + 20% fewer mont_muls
// than the ESIMD SIMD-16 gather kernel; 15-30× more WGs (saturates B70).
// Estimated 2-3× kernel speedup → 5.5-6.7% E2E.
//
// Math identity: out[e] = Σ_{i=0..deg-1} coeffs[which[e]*deg + i] · x[e]^i,
// computed as a top-down Horner reduction (associativity preserved over bb31).
//
// Bit-exact with CUDA reference (R3-A02 verified).
//
// Tunable: RISC0_BATCH_EVAL_WG_CUDA (default 256; one of 64/128/256/512/1024).
// ============================================================================
void esimd_batch_evaluate_any_cuda_style(sycl::queue& q,
                                          uint32_t* d_out,
                                          const uint32_t* d_coeffs,
                                          const uint32_t* d_which,
                                          const uint32_t* d_xs,
                                          uint32_t count,
                                          uint32_t deg) {
    if (count == 0) return;

    uint32_t WG_SIZE = 256;
    if (const char* s = std::getenv("RISC0_BATCH_EVAL_WG_CUDA")) {
        int v = std::atoi(s);
        if (v == 64 || v == 128 || v == 256 || v == 512 || v == 1024) {
            WG_SIZE = static_cast<uint32_t>(v);
        }
    }

    const uint32_t num_groups = count;
    auto* pout = d_out;
    auto* pcoeffs = d_coeffs;
    auto* pwhich = d_which;
    auto* pxs = d_xs;

    q.submit([&](sycl::handler& cgh) {
        // SLM: one FpExt (4 u32) per work-item.
        sycl::local_accessor<uint32_t, 1> tots(sycl::range<1>(WG_SIZE * 4), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                const uint32_t lid = item.get_local_id(0);
                const uint32_t grp = item.get_group(0);
                const uint32_t wg = item.get_local_range(0);

                // Per-WG setup: read which[grp] and xs[grp] (one polynomial)
                const uint32_t poly_off = pwhich[grp] * deg;
                const uint32_t* cur_poly = pcoeffs + poly_off;

                // Load FpExt x (AoS: 4 u32 starting at xs[grp*4])
                uint32_t x[4] = { pxs[grp*4+0], pxs[grp*4+1], pxs[grp*4+2], pxs[grp*4+3] };

                // stepx = x^WG_SIZE  (scalar; identical across threads of this WG)
                uint32_t stepx[4];
                scalar_ext_pow(stepx, x, wg);

                // Horner reduction: K = ceil(deg / WG_SIZE)
                const uint32_t K = (deg + wg - 1) / wg;
                uint32_t tot[4] = { 0, 0, 0, 0 };

                for (int k = (int)K - 1; k >= 0; --k) {
                    // tot *= stepx
                    uint32_t newtot[4];
                    scalar_ext_mul(newtot, tot, stepx);
                    tot[0] = newtot[0]; tot[1] = newtot[1];
                    tot[2] = newtot[2]; tot[3] = newtot[3];

                    // tot += coeff[lid + k*wg]  (Fp added into component 0)
                    size_t i = (size_t)lid + (size_t)k * wg;
                    if (i < deg) {
                        tot[0] = scalar_field_add(tot[0], cur_poly[i]);
                    }
                }

                // tot *= x^lid
                {
                    uint32_t pwx[4];
                    scalar_ext_pow(pwx, x, lid);
                    uint32_t t2[4];
                    scalar_ext_mul(t2, tot, pwx);
                    tot[0] = t2[0]; tot[1] = t2[1];
                    tot[2] = t2[2]; tot[3] = t2[3];
                }

                // Store partial sum to SLM (4 u32 per thread)
                tots[lid*4 + 0] = tot[0];
                tots[lid*4 + 1] = tot[1];
                tots[lid*4 + 2] = tot[2];
                tots[lid*4 + 3] = tot[3];
                item.barrier(sycl::access::fence_space::local_space);

                // Tree reduction in SLM (component-wise field add)
                for (uint32_t stride = wg >> 1; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        tots[lid*4 + 0] = scalar_field_add(tots[lid*4 + 0],
                                                          tots[(lid + stride)*4 + 0]);
                        tots[lid*4 + 1] = scalar_field_add(tots[lid*4 + 1],
                                                          tots[(lid + stride)*4 + 1]);
                        tots[lid*4 + 2] = scalar_field_add(tots[lid*4 + 2],
                                                          tots[(lid + stride)*4 + 2]);
                        tots[lid*4 + 3] = scalar_field_add(tots[lid*4 + 3],
                                                          tots[(lid + stride)*4 + 3]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Thread 0 writes the FpExt result (AoS: 4 u32)
                if (lid == 0) {
                    pout[grp*4 + 0] = tots[0];
                    pout[grp*4 + 1] = tots[1];
                    pout[grp*4 + 2] = tots[2];
                    pout[grp*4 + 3] = tots[3];
                }
            });
    });
    q.wait();
}

// ============================================================================
// Validation
// ============================================================================
int32_t validate_fri_poly_ops() {
    try {
    auto q = create_fri_queue();

    // Test fri_fold with simple data
    constexpr uint32_t COUNT = 4;
    // Input: 4 * kFriFold * COUNT Fp values (planar FpExt)
    constexpr uint32_t IN_SIZE = 4 * kFriFold * COUNT;
    auto* d_in = sycl::malloc_device<uint32_t>(IN_SIZE, q);
    auto* d_out = sycl::malloc_device<uint32_t>(4 * COUNT, q);
    auto* d_mix = sycl::malloc_device<uint32_t>(4, q);
    auto* h_in = sycl::malloc_host<uint32_t>(IN_SIZE, q);
    auto* h_out = sycl::malloc_host<uint32_t>(4 * COUNT, q);

    // Initialize input: FpExt = [ONE, 0, 0, 0] in planar layout
    // Planar: comp 0 = ONE for all, comp 1-3 = 0
    for (uint32_t i = 0; i < kFriFold * COUNT; i++) h_in[0 * kFriFold * COUNT + i] = bb31::ONE;
    for (uint32_t i = 0; i < kFriFold * COUNT; i++) h_in[1 * kFriFold * COUNT + i] = 0;
    for (uint32_t i = 0; i < kFriFold * COUNT; i++) h_in[2 * kFriFold * COUNT + i] = 0;
    for (uint32_t i = 0; i < kFriFold * COUNT; i++) h_in[3 * kFriFold * COUNT + i] = 0;
    // Mix = [ONE, 0, 0, 0] (= scalar 1 in extension field)
    uint32_t h_mix[4] = {bb31::ONE, 0, 0, 0};
    q.memcpy(d_in, h_in, IN_SIZE * 4);
    q.memcpy(d_mix, h_mix, 4 * 4); q.wait();

    esimd_fri_fold(q, d_out, d_in, d_mix, COUNT);
    q.memcpy(h_out, d_out, 4 * COUNT * 4); q.wait();

    // With mix=[1,0,0,0] and all inputs=[1,0,0,0], each output should be
    // sum of 16 * [1,0,0,0] = [16,0,0,0] in extension field
    // 16 in Montgomery = host_mont_encode(16)
    uint32_t sixteen_mont = (uint32_t)(((uint64_t)16 << 32) % bb31::MOD);
    bool fold_ok = true;
    for (uint32_t i = 0; i < COUNT; i++) {
        if (h_out[0 * COUNT + i] != sixteen_mont || h_out[1 * COUNT + i] != 0 ||
            h_out[2 * COUNT + i] != 0 || h_out[3 * COUNT + i] != 0) {
            fprintf(stderr, "  fri_fold mismatch at %u: [%08x,%08x,%08x,%08x] exp [%08x,0,0,0]\n",
                    i, h_out[i], h_out[COUNT+i], h_out[2*COUNT+i], h_out[3*COUNT+i], sixteen_mont);
            fold_ok = false; break;
        }
    }
    fprintf(stderr, "  fri_fold: %s\n", fold_ok ? "PASS" : "FAIL");

    sycl::free(d_in, q); sycl::free(d_out, q); sycl::free(d_mix, q);
    sycl::free(h_in, q); sycl::free(h_out, q);

    return fold_ok ? 0 : -1;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what()); fflush(stderr);
        return -99;
    }
}

} // extern "C"

// ============================================================================
// eval_u power-table path (mb5/eval-u-deep). Drop-in for esimd_batch_evaluate_any.
//  - D2H which/xs (tiny), group taps by polynomial, polys by their distinct-point set
//    (split into chunks of <= 6 points), NP polys x NB points per job.
//  - X[pt][c][i] = x_pt^i (planar FpExt) generated once per call (one table per distinct point).
//  - job kernel: chunk-major split-K (S chunks), acc += coeff * X (4 mont per coeff-tap).
//  - reduce: out[t] = sum over chunks (field addition is exact => bit-identical to the
//    production kernel, whose result is the same field element).
// All host tables are uploaded with an immediate wait (no pageable-source lifetime hazard).
// Fallback to the original kernel for deg < 16384 (or not a power of two) or RISC0_EVAL_U_PT=0.
// The static scratch below is guarded by g_mutex.
// ============================================================================
namespace evalu_pt {
static uint32_t* g_X = nullptr; static size_t g_X_words = 0;
static uint32_t* g_part = nullptr; static size_t g_part_words = 0;
static uint32_t* g_tab = nullptr; static size_t g_tab_words = 0;
static uint32_t* g_pts = nullptr; static size_t g_pts_words = 0;
static std::mutex g_mutex;
static uint32_t* grow(sycl::queue& q, uint32_t*& p, size_t& cap, size_t words) {
    if (words > cap) {
        if (p) { q.wait(); sycl::free(p, q); }  // sycl::free does not wait
        p = nullptr;
        cap = 0;
        size_t want = words + words / 4 + 64;
        p = sycl::malloc_device<uint32_t>(want, q);
        if (!p) throw std::runtime_error("batch_evaluate_any: scratch allocation failed");
        cap = want;
    }
    return p;
}

static sycl::event gen_xtable(sycl::queue& q, uint32_t* X, const uint32_t* pts, uint32_t npts, uint32_t n, uint32_t L) {
    uint32_t nch = n / L, N = npts * nch;
    uint32_t nthr = N == 0 ? 16 : (N + 15) / 16 * 16;
    return q.parallel_for(sycl::nd_range<1>(nthr, 16), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
        uint32_t g = it.get_global_id(0);
        if (g >= N) return;
        uint32_t pt = g / nch, ch = g % nch;
        uint32_t i0 = ch * L;
        FpExt4 x{Vec16(pts[pt * 4 + 0]), Vec16(pts[pt * 4 + 1]), Vec16(pts[pt * 4 + 2]), Vec16(pts[pt * 4 + 3])};
        FpExt4 cur = fpext_pow(x, i0);
        esimd::simd<uint32_t, 16> lane(0u, 1u);
        FpExt4 pw = x;
        for (int bit = 0; bit < 4; bit++) {
            FpExt4 t = fpext_mul(cur, pw);
            auto m = ((lane >> bit) & 1u) != 0u;
            for (int c = 0; c < 4; c++) cur.c[c].merge(t.c[c], m);
            pw = fpext_mul(pw, pw);
        }
        for (uint32_t i = i0; i < i0 + L; i += 16) {
            for (int c = 0; c < 4; c++) esimd::block_store<uint32_t, 16>(X + ((size_t)(pt * 4 + c) * n + i), cur.c[c]);
            cur = fpext_mul(cur, pw);
        }
    });
}

ESIMD_INLINE uint32_t lane_reduce(Vec16 v) {
    esimd::simd<uint32_t, 8> v8 = v.select<8, 1>(0) + v.select<8, 1>(8); v8.merge(v8 - bb31::MOD, v8 >= bb31::MOD);
    esimd::simd<uint32_t, 4> v4 = v8.select<4, 1>(0) + v8.select<4, 1>(4); v4.merge(v4 - bb31::MOD, v4 >= bb31::MOD);
    esimd::simd<uint32_t, 2> v2 = v4.select<2, 1>(0) + v4.select<2, 1>(2); v2.merge(v2 - bb31::MOD, v2 >= bb31::MOD);
    uint32_t s = v2[0] + v2[1];
    return s >= bb31::MOD ? s - bb31::MOD : s;
}

// tables (u32, device): poff[njobs*NP], jpt[njobs*NB], jout[njobs*NP*NB] (0xffffffff = unused slot)
template <int NP, int NB>
static sycl::event eval_job(sycl::queue& q, const uint32_t* base, const uint32_t* poff, const uint32_t* jpt,
                            const uint32_t* jout, uint32_t njobs, const uint32_t* X, uint32_t* partial,
                            uint32_t n, uint32_t S) {
    uint32_t L = n / S, N = njobs * S;
    uint32_t nthr = N == 0 ? 16 : (N + 15) / 16 * 16;
    return q.parallel_for(sycl::nd_range<1>(nthr, 16), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
        uint32_t g = it.get_global_id(0);
        if (g >= N) return;
        uint32_t ch = g / njobs, job = g % njobs;
        uint32_t i0 = ch * L;
        FpExt4 acc[NP][NB];
        const uint32_t* pp[NP];
#pragma unroll
        for (int p = 0; p < NP; p++) pp[p] = base + (size_t)poff[job * NP + p] * n;
        const uint32_t* xp[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) xp[b] = X + (size_t)jpt[job * NB + b] * 4 * n;
        for (uint32_t i = i0; i < i0 + L; i += 16) {
            FpExt4 xv[NB];
#pragma unroll
            for (int b = 0; b < NB; b++)
#pragma unroll
                for (int c = 0; c < 4; c++) xv[b].c[c] = esimd::block_load<uint32_t, 16>(xp[b] + (size_t)c * n + i);
#pragma unroll
            for (int p = 0; p < NP; p++) {
                Vec16 cv = esimd::block_load<uint32_t, 16>(pp[p] + i);
#pragma unroll
                for (int b = 0; b < NB; b++)
#pragma unroll
                    for (int c = 0; c < 4; c++)
                        acc[p][b].c[c] = bb31::field_add(acc[p][b].c[c], bb31::mont_mul(cv, xv[b].c[c]));
            }
        }
#pragma unroll
        for (int p = 0; p < NP; p++)
#pragma unroll
            for (int b = 0; b < NB; b++) {
                uint32_t o = jout[(job * NP + p) * NB + b];
                if (o != 0xffffffffu) {
                    esimd::simd<uint32_t, 4> r;
#pragma unroll
                    for (int c = 0; c < 4; c++) r[c] = lane_reduce(acc[p][b].c[c]);
                    esimd::block_store<uint32_t, 4>(partial + ((size_t)o * S + ch) * 4, r);
                }
            }
    });
}

// out[t*4+c] = sum_k partial[(t*S + k)*4 + c], S % 4 == 0.
static sycl::event reduce_partials(sycl::queue& q, uint32_t* out, const uint32_t* partial, uint32_t ntaps, uint32_t S) {
    uint32_t nthr = ntaps == 0 ? 16 : (ntaps + 15) / 16 * 16;
    return q.parallel_for(sycl::nd_range<1>(nthr, 16), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
        uint32_t t = it.get_global_id(0);
        if (t >= ntaps) return;
        const uint32_t* p = partial + (size_t)t * S * 4;
        Vec16 s(0u);
        for (uint32_t j = 0; j < S * 4; j += 16) s = bb31::field_add(s, esimd::block_load<uint32_t, 16>(p + j));
        esimd::simd<uint32_t, 8> a8 = s.select<8, 1>(0) + s.select<8, 1>(8);
        a8.merge(a8 - bb31::MOD, a8 >= bb31::MOD);
        esimd::simd<uint32_t, 4> a4 = a8.select<4, 1>(0) + a8.select<4, 1>(4);
        a4.merge(a4 - bb31::MOD, a4 >= bb31::MOD);
        esimd::block_store<uint32_t, 4>(out + (size_t)t * 4, a4);
    });
}

static int np_for(int nb) { return nb == 1 ? 16 : nb == 2 ? 8 : nb <= 4 ? 4 : 2; }

static void launch_job(sycl::queue& q, int NB, const uint32_t* base, const uint32_t* poff, const uint32_t* jpt,
                       const uint32_t* jout, uint32_t njobs, const uint32_t* X, uint32_t* partial, uint32_t n, uint32_t S) {
    switch (NB) {
    case 1: eval_job<16, 1>(q, base, poff, jpt, jout, njobs, X, partial, n, S); break;
    case 2: eval_job<8, 2>(q, base, poff, jpt, jout, njobs, X, partial, n, S); break;
    case 3: eval_job<4, 3>(q, base, poff, jpt, jout, njobs, X, partial, n, S); break;
    case 4: eval_job<4, 4>(q, base, poff, jpt, jout, njobs, X, partial, n, S); break;
    case 5: eval_job<2, 5>(q, base, poff, jpt, jout, njobs, X, partial, n, S); break;
    case 6: eval_job<2, 6>(q, base, poff, jpt, jout, njobs, X, partial, n, S); break;
    default: throw std::runtime_error("evalu_pt: bad NB");
    }
}

static bool enabled() {
    static int v = -1;
    if (v < 0) { const char* e = std::getenv("RISC0_EVAL_U_PT"); v = (e && e[0] == '0') ? 0 : 1; }
    return v == 1;
}
} // namespace evalu_pt

extern "C" {
void esimd_batch_evaluate_any(sycl::queue& q, uint32_t* d_out, const uint32_t* d_coeffs, const uint32_t* d_which,
                              const uint32_t* d_xs, uint32_t count, uint32_t deg) {
    using namespace evalu_pt;
    if (count == 0) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!enabled() || deg < 16384 || (deg & (deg - 1)) != 0) {
        esimd_batch_evaluate_any_orig(q, d_out, d_coeffs, d_which, d_xs, count, deg);
        if (enabled() && count == 1 && deg == 1) {
            // warmup call from intel_ffi: JIT the power-table kernels with empty grids
            uint32_t* tmp = grow(q, g_tab, g_tab_words, 64);
            for (int nb = 1; nb <= 6; nb++) launch_job(q, nb, tmp, tmp, tmp, tmp, 0, tmp, tmp, 1024, 64);
            reduce_partials(q, tmp, tmp, 0, 4);
            gen_xtable(q, tmp, tmp, 0, 1024, 1024);
            q.wait();
        }
        return;
    }
    // 1. host copies of which / xs
    std::vector<uint32_t> which(count), xs((size_t)count * 4);
    q.memcpy(which.data(), d_which, count * 4);
    q.memcpy(xs.data(), d_xs, (size_t)count * 16);
    q.wait();
    // 2. distinct points, taps per poly (tap order preserved)
    std::vector<std::array<uint32_t, 4>> pts;
    std::map<std::array<uint32_t, 4>, uint32_t> pidx;
    std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> by_poly;  // poly -> (tap, point)
    for (uint32_t t = 0; t < count; t++) {
        std::array<uint32_t, 4> k{xs[t * 4], xs[t * 4 + 1], xs[t * 4 + 2], xs[t * 4 + 3]};
        auto it = pidx.find(k);
        uint32_t pi;
        if (it == pidx.end()) { pi = pts.size(); pidx[k] = pi; pts.push_back(k); } else pi = it->second;
        by_poly[which[t]].push_back({t, pi});
    }
    // 3. classes keyed by point list (chunks of <= 6)
    std::map<std::vector<uint32_t>, std::vector<std::pair<uint32_t, std::vector<uint32_t>>>> classes;
    for (auto& [poly, tl] : by_poly) {
        for (size_t s0 = 0; s0 < tl.size(); s0 += 6) {
            size_t e = std::min(tl.size(), s0 + 6);
            std::vector<uint32_t> key, taps;
            for (size_t j = s0; j < e; j++) { key.push_back(tl[j].second); taps.push_back(tl[j].first); }
            classes[key].push_back({poly, taps});
        }
    }
    struct JobDesc { int NB; uint32_t njobs; size_t poff, jpt, jout; };
    std::vector<JobDesc> jds;
    std::vector<uint32_t> tab;
    for (auto& [key, polys] : classes) {
        int NB = key.size(), NP = np_for(NB);
        uint32_t njobs = (polys.size() + NP - 1) / NP;
        JobDesc jd{NB, njobs, 0, 0, 0};
        jd.poff = tab.size();
        for (uint32_t j = 0; j < njobs * NP; j++) tab.push_back(j < polys.size() ? polys[j].first : polys[0].first);
        jd.jpt = tab.size();
        for (uint32_t j = 0; j < njobs; j++) for (int b = 0; b < NB; b++) tab.push_back(key[b]);
        jd.jout = tab.size();
        for (uint32_t j = 0; j < njobs * NP; j++)
            for (int b = 0; b < NB; b++) tab.push_back(j < polys.size() ? polys[j].second[b] : 0xffffffffu);
        jds.push_back(jd);
    }
    const uint32_t npts = pts.size();
    const uint32_t S = std::min<uint32_t>(1024, deg / 16);
    const uint32_t XL = std::min<uint32_t>(4096, deg);
    uint32_t* X = grow(q, g_X, g_X_words, (size_t)npts * 4 * deg);
    uint32_t* part = grow(q, g_part, g_part_words, (size_t)count * S * 4);
    uint32_t* dtab = grow(q, g_tab, g_tab_words, tab.size());
    uint32_t* dpts = grow(q, g_pts, g_pts_words, (size_t)npts * 4);
    q.memcpy(dtab, tab.data(), tab.size() * 4);
    q.memcpy(dpts, pts.data(), (size_t)npts * 16);
    q.wait();  // host vectors die at return; never leave an async copy from them in flight
    gen_xtable(q, X, dpts, npts, deg, XL);
    for (auto& jd : jds)
        launch_job(q, jd.NB, d_coeffs, dtab + jd.poff, dtab + jd.jpt, dtab + jd.jout, jd.njobs, X, part, deg, S);
    reduce_partials(q, d_out, part, count, S);
    q.wait();
}
} // extern "C"
