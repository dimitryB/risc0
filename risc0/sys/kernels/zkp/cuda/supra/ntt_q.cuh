// Four-step fused commit NTT ("nttq") for lg_domain_size 16..21, lg_blowup 2 (CUDA only).
//   per column: W (evaluations) --colpass--> T1 --rowpass (* n^-1 * g^k)--> C (natural coefficients)
//               C (rows < Lc/4 of the [Lc][Lr] LDE matrix) --coset colpass--> T2 --rowpass--> E
// Each pass computes batched line DFTs of length L = 2^LOGL (256..2048) as 16 x 16 x P3 stages in
// registers with two shared-memory exchanges, TB lines per block (512-thread blocks where possible).
// The forward column pass splits the zero-padded column DFT of length Lc into its 4 cosets
// k = 4a + b (blockIdx.z = b): DFT_{Lc/4}(x_m * w_Lc^(m b)), so it reads only the nonzero rows.
// Four-step twiddles and zk-shift factors are chained per thread (one table gather per thread).
// Scratch T1/T2 lines are discarded from L2 after their last read (no write-back); the final LDE
// is written with streaming stores. All arithmetic is sppark's canonical Montgomery bb31, so the
// output equals the radix-2 path bit for bit.
// The forward passes run CHF columns at a time, alternating between NS side streams with their own
// T2 scratch (L2-resident); the side streams are joined back into the gpu stream before returning.
#pragma once
#include <mutex>
#include "../device_alloc.h"

namespace nttq {

__host__ __device__ constexpr int brev_c(int i, int bits) {
    int r = 0;
    for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    return r;
}
template<int LOGP, int LG>
__device__ __forceinline__ void dit_level(fr_t* w, const fr_t* rootsP)
{
    if constexpr (LG <= LOGP) {
        constexpr int P = 1 << LOGP, len = 1 << LG, half = len >> 1;
        #pragma unroll
        for (int s = 0; s < P; s += len) {
            #pragma unroll
            for (int k = 0; k < half; k++) {
                fr_t u = w[s + k], t = w[s + k + half];
                if (k != 0) t = t * rootsP[k << (LOGP - LG)];
                w[s + k] = u + t;
                w[s + k + half] = u - t;
            }
        }
        dit_level<LOGP, LG + 1>(w, rootsP);
    }
}
// in-register DFT_P, natural in / natural out (radix-2 DIT); rootsP[e] = w_P^e, e < P/2
template<int LOGP>
__device__ __forceinline__ void reg_dft(fr_t* v, const fr_t* rootsP)
{
    constexpr int P = 1 << LOGP;
    fr_t w[P];
    #pragma unroll
    for (int i = 0; i < P; i++) w[i] = v[brev_c(i, LOGP)];
    dit_level<LOGP, 1>(w, rootsP);
    #pragma unroll
    for (int i = 0; i < P; i++) v[i] = w[i];
}

struct Tw {             // w_N^e = A[e & 4095] * B[e >> 12]
    const fr_t* A;
    const fr_t* B;
    unsigned lgN;
};

// Per-(line size, direction) tables:
//   r16[e] = w_16^e (e < 8), r16[16 + e] = w_16^e (e < 10); rP3[e] = w_P3^e (e < P3/2)
//   t2[m*16 + k] = w_256^(k*m) (m, k < 16); t3[m*256 + k] = w_L^(k*m) (m < P3, k < 256)
struct LineTabs { const fr_t* r16; const fr_t* rP3; const fr_t* t2; const fr_t* t3;
#ifdef __HIPCC__
                  const fr_t* cw = nullptr;   // forward coset column pass only (pass_h, MODE 2)
#endif
};

template<int LOGL, int TB> struct Smem {
    static constexpr int L = 1 << LOGL;
    static constexpr int LS = L + L / 32 + (32 / TB > 0 ? 32 / TB : 1);  // padded line stride
    static constexpr int TW = 256 + 256 * (1 << (LOGL - 8));            // t2 + t3 staged after the lines
    __device__ static __forceinline__ int a(int i) { return i + (i >> 5); }
};

__device__ __forceinline__ fr_t twv(const Tw& tw, unsigned e) {
    e &= (1u << tw.lgN) - 1;
    return tw.A[e & 4095] * tw.B[e >> 12];
}

enum : unsigned { F_DISCARD_IN = 1, F_CS_OUT = 2 };
#ifndef __HIPCC__
__device__ __forceinline__ void st_cs(fr_t* p, fr_t x) { asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(p), "r"(*(uint32_t*)&x) : "memory"); }
// the 128-byte line's contents become undefined; it is dropped from L2 without a write-back
__device__ __forceinline__ void discard_l2(const void* p) { asm volatile("discard.global.L2 [%0], 128;" :: "l"(p) : "memory"); }
#else
__device__ __forceinline__ void st_cs(fr_t* p, fr_t x) { *p = x; }
__device__ __forceinline__ void discard_l2(const void*) {}
#endif

template<int LOGL, int TB, int MODE, int MINB>
__global__ void __launch_bounds__(TB * (1 << LOGL) / 16, MINB)
pass_k(const fr_t* __restrict__ in, fr_t* __restrict__ out, size_t ics, size_t ocs,
       unsigned lgX, unsigned rows_in, LineTabs T, Tw tw,
       const fr_t* __restrict__ lo, const fr_t* __restrict__ hi, fr_t scale, unsigned flags)
{
    constexpr int L = 1 << LOGL, LOGP3 = LOGL - 8, P3 = 1 << LOGP3;
    static_assert(LOGP3 >= 0 && LOGP3 <= 4, "LOGL 8..12");
    using SM = Smem<LOGL, TB>;
    constexpr int NT = TB * L / 16;
    constexpr int LP1 = L / 16;
    extern __shared__ fr_t smem[];
    in += blockIdx.y * ics;
    out += blockIdx.y * ocs;
    const size_t base0 = (size_t)blockIdx.x * TB;
    const int tid = threadIdx.x;
    const unsigned bz = MODE == 2 ? blockIdx.z : 0;

    fr_t r16[8];
    #pragma unroll
    for (int e = 0; e < 8; e++) r16[e] = T.r16[e];
    fr_t* st2 = smem + TB * SM::LS;
    fr_t* st3 = st2 + 256;
    for (int i = tid; i < 256; i += NT) st2[i] = T.t2[i];
    for (int i = tid; i < 256 * P3; i += NT) st3[i] = T.t3[i];
    const fr_t* T2p = st2; const fr_t* T3p = st3;

    {
        int c, j;
        if (MODE != 1) { c = tid % TB; j = tid / TB; }
        else           { c = tid / LP1; j = tid % LP1; }
        fr_t v[16];
        if (MODE == 0) {
            const size_t S = (size_t)1 << lgX;
            const fr_t* src = in + base0 + c + (size_t)j * S;
            if (rows_in == L / 4) {
                #pragma unroll
                for (int m = 0; m < 4; m++) v[m] = src[(size_t)m * LP1 * S];
                fr_t x1 = v[1], x2 = v[2], x3 = v[3], x0 = v[0];
                fr_t r4[2] = {r16[0], r16[4]};
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    fr_t y[4];
                    y[0] = x0;
                    y[1] = b == 0 ? x1 : x1 * T.r16[16 + 1 * b];
                    y[2] = b == 0 ? x2 : x2 * T.r16[16 + 2 * b];
                    y[3] = b == 0 ? x3 : x3 * T.r16[16 + 3 * b];
                    reg_dft<2>(y, r4);
                    #pragma unroll
                    for (int a = 0; a < 4; a++) v[4 * a + b] = y[a];
                }
            } else {
                #pragma unroll
                for (int m = 0; m < 16; m++) v[m] = src[(size_t)m * LP1 * S];
                reg_dft<4>(v, r16);
            }
        } else if (MODE == 2) {
            const size_t S = (size_t)1 << lgX;
            const fr_t* src = in + base0 + c + (size_t)j * S;
            #pragma unroll
            for (int m = 0; m < 16; m++) v[m] = src[(size_t)m * LP1 * S];
            if (bz != 0) {
                // twist x_row *= w_Lc^(row*b) = w_N^(row*b*S), row = j + m*LP1
                fr_t t = twv(tw, (unsigned)(j * bz) << lgX);
                fr_t st = twv(tw, (unsigned)(LP1 * bz) << lgX);
                #pragma unroll
                for (int m = 0; m < 16; m++) {
                    v[m] = v[m] * t;
                    if (m + 1 < 16) t = t * st;
                }
            }
            reg_dft<4>(v, r16);
        } else {
            const fr_t* src = in + (base0 + c) * L;
            #pragma unroll
            for (int m = 0; m < 16; m++) v[m] = src[j + m * LP1];
            reg_dft<4>(v, r16);
        }
        fr_t* sl = smem + c * SM::LS;
        #pragma unroll
        for (int m = 0; m < 16; m++) sl[SM::a(j * 16 + m)] = v[m];
    }
    __syncthreads();
    if (MODE == 1 && (flags & F_DISCARD_IN)) {
        // the block's input rows [base0, base0 + TB) * L are contiguous and dead now
        constexpr int NL = TB * L / 32;               // 128-byte lines
        const char* b = (const char*)(in + base0 * L);
        for (int i = tid; i < NL; i += NT) discard_l2(b + (size_t)i * 128);
    }
    {
        int c = tid / LP1, j = tid % LP1, k = j & 15;
        fr_t* sl = smem + c * SM::LS;
        fr_t v[16];
        #pragma unroll
        for (int m = 0; m < 16; m++) v[m] = sl[SM::a(j + m * LP1)];
        #pragma unroll
        for (int m = 1; m < 16; m++) v[m] = v[m] * T2p[m * 16 + k];
        reg_dft<4>(v, r16);
        __syncthreads();
        int b = (j >> 4) * 256 + k;
        #pragma unroll
        for (int m = 0; m < 16; m++) sl[SM::a(b + m * 16)] = v[m];
    }
    __syncthreads();
    {
        fr_t rP[P3 / 2 > 0 ? P3 / 2 : 1];
        #pragma unroll
        for (int e = 0; e < P3 / 2; e++) rP[e] = T.rP3[e];
        constexpr int ITEMS = TB * 256, IPT = ITEMS / NT;
        constexpr unsigned KM = MODE == 2 ? 4 : 1;
        fr_t step, tI, sI, gI, gsI, gs;
        {
            const int c0 = tid % TB, j0 = tid / TB;
            constexpr unsigned JS = NT / TB;                 // j advance per item
            if (MODE != 1) {
                unsigned col = (unsigned)(base0 + c0);
                step = twv(tw, col * (256u * KM));
                tI = twv(tw, col * ((unsigned)j0 * KM + bz));
                sI = twv(tw, col * (JS * KM));
            } else if (lo) {
                size_t k0 = base0 + c0 + ((size_t)j0 << lgX);
                gI = scale * (lo[k0 & 1023] * hi[k0 >> 10]);
                size_t ki = (size_t)JS << lgX, ks = (size_t)256 << lgX;
                gsI = lo[ki & 1023] * hi[ki >> 10];
                gs = lo[ks & 1023] * hi[ks >> 10];
            }
        }
        #pragma unroll
        for (int it = 0; it < IPT; it++) {
            int item = tid + it * NT;
            int c = item % TB, j = item / TB;
            const fr_t* sl = smem + c * SM::LS;
            fr_t v[P3];
            #pragma unroll
            for (int m = 0; m < P3; m++) v[m] = sl[SM::a(j + m * 256)];
            #pragma unroll
            for (int m = 1; m < P3; m++) v[m] = v[m] * T3p[m * 256 + j];
            reg_dft<LOGP3>(v, rP);
            fr_t t = tI, g = gI;
            #pragma unroll
            for (int m = 0; m < P3; m++) {
                unsigned k = (j + m * 256) * KM + bz;
                fr_t x = v[m];
                if (MODE != 1) {
                    x = x * t;
                    if (m + 1 < P3) t = t * step;
                    out[((size_t)k << lgX) + base0 + c] = x;
                } else {
                    size_t kk = base0 + c + ((size_t)k << lgX);
                    if (lo) { x = x * g; if (m + 1 < P3) g = g * gs; }
                    if (flags & F_CS_OUT) st_cs(out + kk, x); else out[kk] = x;
                }
            }
            if (it + 1 < IPT) {
                if (MODE != 1) tI = tI * sI;
                else if (lo) gI = gI * gsI;
            }
        }
    }
}

#ifdef __HIPCC__
// ---- RDNA variant of pass_k ----------------------------------------------------------------
// On RX 7900 XTX (gfx1100) and RX 9070 XT (gfx1201) the passes are VALU-bound rather than
// DRAM-bound, and bb31 Montgomery products are most of the issue time (opt-artifacts
// r0amd/ntt2). pass_h is pass_k with the same thread/LDS layout and:
//  * the Montgomery product in plain C (fr_t's v_mad_u64_u32 asm statement is opaque to the
//    scheduler) as a signed REDC: 2 quarter-rate multiplies instead of 3 (see mul);
//  * add/sub as min(a + b, a + b - p) / min(a - b, a - b + p), without a carry SGPR;
//  * lazy reduction: butterfly outputs that are next only multiplied stay in [-p, p) (see dit);
//  * 32-bit byte offsets from a uniform column base (saddr global loads/stores);
//  * MODE 2: the coset twist w_Lc^(row*b), row = n1 + LP1*n2, is applied as w_64^(b*n2) (uniform
//    per block) before the first radix-16 and as w_Lc^(b*n1), folded into a per-coset copy of
//    the second stage's twiddle table, instead of a per-element chained product;
//  * gfx12, forward passes: paired stores (PS, below).
// Every rewrite is an identity in the field and every stored value is canonical, so the output is
// bit-identical to pass_k and to sppark's radix-2 path.
namespace hq {
constexpr uint32_t P = 0x78000001u;
__device__ __forceinline__ uint32_t add(uint32_t a, uint32_t b) { uint32_t s = a + b; return min(s, s - P); }
__device__ __forceinline__ uint32_t sub(uint32_t a, uint32_t b) { uint32_t d = a - b; return min(d, d + P); }
// An opaque copy: keeps LLVM from folding the shift-adds of mul() back into a v_mul_lo_u32.
__device__ __forceinline__ uint32_t opq(uint32_t x) { asm("" : "+v"(x)); return x; }
// a * b * 2^-32 mod p, canonical, for a in [-p, p) (int32 bits: canonical or a lazy output) and b
// canonical: signed REDC, 2 quarter-rate ops instead of 3. t = a * b (v_mad_i64_i32), |t| < p^2;
// m = (int32)(t * p^-1 mod 2^32), p^-1 = 2^31 + 2^27 + 1, from two v_lshl_add_u32; t - m * p
// (one v_mad_i64_i32) is 0 mod 2^32 and |t - m * p| < p^2 + 2^31 p < 2^63, so its high word r is
// the exact quotient, == a * b * 2^-32 (mod p), |r| <= (p^2 + 2^31 p) / 2^32 < 0.97p; min(r, r + p)
// as u32 is canonical.
__device__ __forceinline__ uint32_t mul(uint32_t a, uint32_t b)
{
    int64_t t = (int64_t)(int32_t)a * (int32_t)b;
    uint32_t lo = (uint32_t)t;
    int32_t m = (int32_t)(opq(lo + (lo << 27)) + (lo << 31));
    uint32_t r = (uint32_t)((uint64_t)(t + (int64_t)m * -(int64_t)P) >> 32);
    return min(r, r + P);
}
// element offset i within one column (at most 2^23 elements), scaled to bytes in 32 bits
__device__ __forceinline__ uint32_t ld(const uint32_t* base, uint32_t i)
{   return *(const uint32_t*)((const char*)base + (i << 2));   }
__device__ __forceinline__ void st(uint32_t* base, uint32_t i, uint32_t x)
{   *(uint32_t*)((char*)base + (i << 2)) = x;   }
// Lazy reduction: an output that is next used only as the left operand of mul (a 't' with k != 0
// at the next level, or with LZ any output of the last level) is left in [-p, p) as int32 bits
// (u + t - p, u - t); mul(a, b) is canonical for such a and canonical b, red() makes it canonical.
// Everything else stays canonical.
template<int LOGP, int LG, bool LZ>
__device__ __forceinline__ void dit(uint32_t* w, const uint32_t* rootsP)
{
    if constexpr (LG <= LOGP) {
        constexpr int PP = 1 << LOGP, len = 1 << LG, half = len >> 1;
        #pragma unroll
        for (int s = 0; s < PP; s += len) {
            const bool nt = LG < LOGP ? ((s >> LG) & 1) != 0 : LZ;   // outputs feed mul next
            #pragma unroll
            for (int k = 0; k < half; k++) {
                uint32_t u = w[s + k], t = w[s + k + half];
                if (k != 0) t = mul(t, rootsP[k << (LOGP - LG)]);
                w[s + k] = nt && (k != 0 || LG == LOGP) ? u + t - P : add(u, t);
                w[s + k + half] = nt ? u - t : sub(u, t);
            }
        }
        dit<LOGP, LG + 1, LZ>(w, rootsP);
    }
}
template<int LOGP, bool LZ = false>
__device__ __forceinline__ void dft(uint32_t* v, const uint32_t* rootsP)
{
    constexpr int PP = 1 << LOGP;
    uint32_t w[PP];
    #pragma unroll
    for (int i = 0; i < PP; i++) w[i] = v[brev_c(i, LOGP)];
    dit<LOGP, 1, LZ>(w, rootsP);
    #pragma unroll
    for (int i = 0; i < PP; i++) v[i] = w[i];
}
__device__ __forceinline__ uint32_t red(uint32_t a) { return min(a, a + P); }   // [-p, p) -> canonical
__device__ __forceinline__ uint32_t tw_at(const Tw& tw, unsigned e) {
    e &= (1u << tw.lgN) - 1;
    return mul(*(const uint32_t*)&tw.A[e & 4095], *(const uint32_t*)&tw.B[e >> 12]);
}
// coset tables (cw): [b*16 + n2] = w_64^(b*n2), then per coset b a [P3][CS] table at 64 + b*T2N:
// [n1a*CS + m*16 + a1] = w_256^(a1*m) * w_Lc^(b*(n1a + P3*m)); CS pads the n1a rows apart in LDS banks.
constexpr int CS = 272;
template<int LOGL, int MODE> constexpr int t2n() { return MODE == 2 ? (1 << (LOGL - 8)) * CS : 256; }
template<int LOGL, int TB, int MODE> constexpr size_t smem_bytes()
{   return ((size_t)TB * Smem<LOGL, TB>::LS + t2n<LOGL, MODE>() + 256 * (1 << (LOGL - 8))) * 4;   }
static bool is_gfx12(int dev)
{
    static int v[16];   // 0 unknown, 1 no, 2 yes
    if (v[dev] == 0) {
        cudaDeviceProp p;
        CUDA_OK(cudaGetDeviceProperties(&p, dev));
        v[dev] = strncmp(p.gcnArchName, "gfx12", 5) == 0 ? 2 : 1;
    }
    return v[dev] == 2;
}
// Paired stores (pass_h PS) for the forward passes on gfx12. On gfx1100 and for the inverse passes
// they gain nothing or lose (opt-artifacts r0amd/ntt2 tlg_prmask).
static bool paired(int dev, int mode, bool zk)
{
#ifdef NTTQ_TUNE
    // NTTQ_PAIR=<hex mask>: bit 0 inverse col, bit 1 inverse row, bit 2 forward row, bit 3 forward col
    if (const char* e = getenv("NTTQ_PAIR")) {
        const int pass = mode == 0 ? 0 : mode == 2 ? 3 : zk ? 1 : 2;
        return (strtoul(e, nullptr, 16) >> pass) & 1;
    }
#endif
    return mode != 0 && !zk && is_gfx12(dev);
}
} // namespace hq

// PS (paired stores): a block runs the three phases for 2 groups of TB lines in turn and holds the
// first group's results in registers, so that every store instruction writes 2 x TB consecutive
// elements of one output row (64 bytes for TB 8) instead of TB. Output rows lie 2^lgX elements
// apart; 32-byte pieces of them are slow on gfx12 (256-byte L2 lines).
template<int LOGL, int TB, int MODE, int MINB, bool PS>
__global__ void __launch_bounds__(TB * (1 << LOGL) / 16, MINB)
pass_h(const fr_t* __restrict__ in_, fr_t* __restrict__ out_, size_t ics, size_t ocs,
       unsigned lgX, unsigned rows_in, LineTabs T, Tw tw,
       const fr_t* __restrict__ lo_, const fr_t* __restrict__ hi_, fr_t scale_, unsigned flags)
{
    using namespace hq;
    constexpr int L = 1 << LOGL, LOGP3 = LOGL - 8, P3 = 1 << LOGP3;
    static_assert(LOGP3 >= 0 && LOGP3 <= 4, "LOGL 8..12");
    static_assert(!PS || 2 * TB <= 32, "PS");
    constexpr int PR = PS ? 2 : 1;
    using SM = Smem<LOGL, TB>;
    constexpr int NT = TB * L / 16;
    constexpr int LP1 = L / 16;
    constexpr int T2N = t2n<LOGL, MODE>();
    constexpr int ITEMS = TB * 256, IPT = ITEMS / NT;
    extern __shared__ uint32_t smem_h[];
    const uint32_t* in = (const uint32_t*)in_ + blockIdx.y * ics;
    uint32_t* out = (uint32_t*)out_ + blockIdx.y * ocs;
    const uint32_t* lo = (const uint32_t*)lo_;
    const uint32_t* hi = (const uint32_t*)hi_;
    const int tid = threadIdx.x;
    const unsigned bz = MODE == 2 ? blockIdx.z : 0;
    const uint32_t* cw = (const uint32_t*)T.cw;

    uint32_t r16[8];
    #pragma unroll
    for (int e = 0; e < 8; e++) r16[e] = *(const uint32_t*)&T.r16[e];
    uint32_t rP[P3 / 2 > 0 ? P3 / 2 : 1];
    #pragma unroll
    for (int e = 0; e < P3 / 2; e++) rP[e] = *(const uint32_t*)&T.rP3[e];
    uint32_t* st2 = smem_h + TB * SM::LS;
    uint32_t* st3 = st2 + T2N;
    {
        const uint32_t* g2 = MODE == 2 ? cw + 64 + bz * T2N : (const uint32_t*)T.t2;
        for (int i = tid; i < T2N; i += NT) st2[i] = g2[i];
        for (int i = tid; i < 256 * P3; i += NT) st3[i] = ((const uint32_t*)T.t3)[i];
    }
    uint32_t xa[PS ? IPT * P3 : 1];

    #pragma unroll
    for (int h = 0; h < PR; h++) {
    const uint32_t base0 = (blockIdx.x * PR + h) * TB;
    if (h) __syncthreads();
    {
        int c, j;
        if (MODE != 1) { c = tid % TB; j = tid / TB; }
        else           { c = tid / LP1; j = tid % LP1; }
        uint32_t v[16];
        if (MODE != 1) {
            const uint32_t o = base0 + c + ((uint32_t)j << lgX), ms = (uint32_t)LP1 << lgX;
            #pragma unroll
            for (int m = 0; m < 16; m++) v[m] = ld(in, o + m * ms);
        } else {
            const uint32_t o = ((base0 + c) << LOGL) + j;
            #pragma unroll
            for (int m = 0; m < 16; m++) v[m] = ld(in, o + m * LP1);
        }
        if (MODE == 2 && bz != 0) {
            const uint32_t* t1 = cw + bz * 16;              // w_64^(b*n2), uniform
            #pragma unroll
            for (int m = 1; m < 16; m++) v[m] = mul(v[m], t1[m]);
        }
        dft<4, true>(v, r16);
        uint32_t* sl = smem_h + c * SM::LS;
        #pragma unroll
        for (int m = 0; m < 16; m++) sl[SM::a(j * 16 + m)] = v[m];
    }
    __syncthreads();
    {
        int c = tid / LP1, j = tid % LP1, k = j & 15;
        uint32_t* sl = smem_h + c * SM::LS;
        uint32_t v[16];
        #pragma unroll
        for (int m = 0; m < 16; m++) v[m] = sl[SM::a(j + m * LP1)];
        if (MODE == 2) {
            const uint32_t* t2 = st2 + (j >> 4) * CS + k;  // w_256^(k*m) * w_Lc^(b*(j/16 + P3*m))
            v[0] = bz != 0 ? mul(v[0], t2[0]) : red(v[0]);
            #pragma unroll
            for (int m = 1; m < 16; m++) v[m] = mul(v[m], t2[m * 16]);
        } else {
            v[0] = red(v[0]);
            #pragma unroll
            for (int m = 1; m < 16; m++) v[m] = mul(v[m], st2[m * 16 + k]);
        }
        dft<4, true>(v, r16);
        __syncthreads();
        int b = (j >> 4) * 256 + k;
        #pragma unroll
        for (int m = 0; m < 16; m++) sl[SM::a(b + m * 16)] = v[m];
    }
    __syncthreads();
    {
        constexpr unsigned KM = MODE == 2 ? 4 : 1;
        uint32_t step, tI, sI, gI, gsI, gs;
        {
            const int c0 = tid % TB, j0 = (tid / TB) ^ (PS && h == 1 ? 1 : 0);   // see j below
            constexpr unsigned JS = NT / TB;                 // j advance per item
            if (MODE != 1) {
                unsigned col = base0 + c0;
                step = tw_at(tw, col * (256u * KM));
                tI = tw_at(tw, col * ((unsigned)j0 * KM + bz));
                sI = tw_at(tw, col * (JS * KM));
            } else if (lo) {
                size_t k0 = base0 + c0 + ((size_t)j0 << lgX);
                gI = mul(*(const uint32_t*)&scale_, mul(lo[k0 & 1023], hi[k0 >> 10]));
                size_t ki = (size_t)JS << lgX, ks = (size_t)256 << lgX;
                gsI = mul(lo[ki & 1023], hi[ki >> 10]);
                gs = mul(lo[ks & 1023], hi[ks >> 10]);
            }
        }
        #pragma unroll
        for (int it = 0; it < IPT; it++) {
            int item = tid + it * NT;
            int c = item % TB, j = item / TB;
            if (PS && h == 1) j ^= 1;                    // the partner lane's row (see the stores)
            const uint32_t* sl = smem_h + c * SM::LS;
            uint32_t v[P3];
            #pragma unroll
            for (int m = 0; m < P3; m++) v[m] = sl[SM::a(j + m * 256)];
            v[0] = red(v[0]);
            #pragma unroll
            for (int m = 1; m < P3; m++) v[m] = mul(v[m], st3[m * 256 + j]);
            // outputs: four-step twiddle or zk-shift product next, else stored (canonical)
            if (MODE != 1) dft<LOGP3, true>(v, rP);
            else if (lo) dft<LOGP3, true>(v, rP);
            else dft<LOGP3, false>(v, rP);
            uint32_t t = tI, g = gI;
            #pragma unroll
            for (int m = 0; m < P3; m++) {
                uint32_t k = (j + m * 256) * KM + bz;
                uint32_t x = v[m];
                if (MODE != 1) {
                    x = mul(x, t);
                    if (m + 1 < P3) t = mul(t, step);
                } else if (lo) {
                    x = mul(x, g);
                    if (m + 1 < P3) g = mul(g, gs);
                }
                if (PS) {
                    if (h == 0) { xa[it * P3 + m] = x; continue; }
                    // this lane holds group 0 row j ^ 1 (a) and group 1 row j (x), the partner lane
                    // (j ^ 1) the other two: store 1 writes row j & ~1 of both groups, store 2 row
                    // j | 1, each 2 x TB consecutive elements
                    const uint32_t sel = j & 1, bb = base0 - TB;
                    const uint32_t k1 = ((j & ~1) + m * 256) * KM + bz, k2 = k1 + KM;
                    const uint32_t a = xa[it * P3 + m];
                    st(out, bb + c + (sel ^ 1) * TB + (k1 << lgX), sel ? a : x);
                    st(out, bb + c + sel * TB + (k2 << lgX), sel ? x : a);
                    continue;
                }
                st(out, base0 + c + (k << lgX), x);
            }
            if (it + 1 < IPT) {
                if (MODE != 1) tI = mul(tI, sI);
                else if (lo) gI = mul(gI, gsI);
            }
        }
    }
    }
}

// cw for the forward coset column pass of line length 2^LOGL in an N = 2^lgN transform with
// columns of length Lc = 2^lgLc (= 4 * 2^LOGL): see hq::CS.
__global__ void k_coset_tabs(fr_t* cw, fr_t wN, unsigned lgN, unsigned lgLc, unsigned LOGL)
{
    const unsigned N = 1u << lgN, P3 = 1u << (LOGL - 8), T2N = P3 * hq::CS;
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < 64) {
        unsigned b = t >> 4, n2 = t & 15;
        cw[t] = wN ^ (unsigned)(((uint64_t)b * n2 * (N / 64)) & (N - 1));
    }
    if (t < 4 * T2N) {
        unsigned b = t / T2N, r = t % T2N, n1a = r / hq::CS, q = r % hq::CS;
        unsigned m = q >> 4, a1 = q & 15;
        uint64_t e = (uint64_t)a1 * m * (N / 256) + (uint64_t)b * (n1a + P3 * m) * (N >> lgLc);
        cw[64 + t] = q < 256 ? wN ^ (unsigned)(e & (N - 1)) : fr_t(0u);
    }
}
#endif // __HIPCC__

__global__ void k_pow_table(fr_t* out, fr_t base, unsigned count, unsigned stride)
{
    unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < count) out[e] = base ^ (e * stride);
}
__global__ void k_line_tabs(fr_t* r16, fr_t* rP3, fr_t* t2, fr_t* t3, fr_t wN, int lgN, int LOGL)
{
    fr_t wL = wN ^ (1u << (lgN - LOGL));
    int L = 1 << LOGL, P3 = 1 << (LOGL - 8);
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < 8) r16[t] = wL ^ (unsigned)(t * (L / 16));
    if (t >= 16 && t < 26) r16[t] = wL ^ (unsigned)((t - 16) * (L / 16));
    if (t < P3 / 2) rP3[t] = wL ^ (unsigned)(t * (L / P3));
    if (t < 256) { int m = t / 16, k = t % 16; t2[t] = wL ^ (unsigned)((k * m) * (L / 256)); }
    if (t < 256 * P3) { int m = t / 256, k = t % 256; t3[t] = wL ^ (unsigned)(k * m); }
}


static LineTabs make_line(const gpu_t& gpu, fr_t w, unsigned lgN, unsigned LOGL)
{
    fr_t *r16, *rP3, *t2, *t3;
    int P3 = 1 << (LOGL - 8);
    CUDA_OK(risc0_cuda_malloc(&r16, 32 * sizeof(fr_t)));
    CUDA_OK(risc0_cuda_malloc(&rP3, 8 * sizeof(fr_t)));
    CUDA_OK(risc0_cuda_malloc(&t2, 256 * sizeof(fr_t)));
    CUDA_OK(risc0_cuda_malloc(&t3, 256 * P3 * sizeof(fr_t)));
    k_line_tabs<<<(256 * P3 + 255) / 256, 256, 0, gpu>>>(r16, rP3, t2, t3, w, (int)lgN, (int)LOGL);
    CUDA_OK(cudaGetLastError());
    return LineTabs{r16, rP3, t2, t3};
}

template<int LOGL, int TB, int MODE, int MINB>
static void launch(cudaStream_t st, unsigned cols, const fr_t* in, fr_t* out, size_t ics, size_t ocs,
                   unsigned lgX, unsigned rows_in, const LineTabs& LT, Tw tw,
                   const fr_t* lo, const fr_t* hi, fr_t scale, unsigned flags)
{
    constexpr int L = 1 << LOGL;
#ifdef __HIPCC__
    // pass_h (see above); NTTQ_TUNE builds: NTTQ_OLD=1 runs pass_k instead
#ifdef NTTQ_TUNE
    static const bool old = [] { const char* e = getenv("NTTQ_OLD"); return e && e[0] == '1'; }();
    if (!old)
#endif
    if (!(MODE == 0 && rows_in == L / 4))   // pass_h has no zero-padded column input: pass_k below
    {
        constexpr size_t shh = hq::smem_bytes<LOGL, TB, MODE>();
        static std::once_flag onceh[16][2];
        int dev;
        CUDA_OK(cudaGetDevice(&dev));
        auto run = [&](auto k, unsigned groups) {
            std::call_once(onceh[dev][groups - 1], [&] {
                CUDA_OK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shh));
            });
            dim3 grid((unsigned)(((size_t)1 << lgX) / (TB * groups)), cols, MODE == 2 ? 4 : 1);
            k<<<grid, TB * L / 16, shh, st>>>(in, out, ics, ocs, lgX, rows_in, LT, tw, lo, hi, scale, flags);
            CUDA_OK(cudaGetLastError());
        };
        if constexpr (2 * TB <= 32) {
            if (((size_t)1 << lgX) % (2 * TB) == 0 && hq::paired(dev, MODE, lo != nullptr)) {
                run(pass_h<LOGL, TB, MODE, MINB, true>, 2);
                return;
            }
        }
        run(pass_h<LOGL, TB, MODE, MINB, false>, 1);
        return;
    }
#endif
    const size_t sh = ((size_t)TB * Smem<LOGL, TB>::LS + Smem<LOGL, TB>::TW) * sizeof(fr_t);
    // per device: the attribute is device state
    static std::once_flag once[16];
    int dev;
    CUDA_OK(cudaGetDevice(&dev));
    std::call_once(once[dev], [&] {
        CUDA_OK(cudaFuncSetAttribute(pass_k<LOGL, TB, MODE, MINB>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    });
    dim3 grid((unsigned)(((size_t)1 << lgX) / TB), cols, MODE == 2 ? 4 : 1);
    pass_k<LOGL, TB, MODE, MINB><<<grid, TB * L / 16, sh, st>>>(in, out, ics, ocs, lgX, rows_in, LT, tw,
                                                                lo, hi, scale, flags);
    CUDA_OK(cudaGetLastError());
}

template<int MODE>
static void pass(cudaStream_t st, unsigned LOGL, unsigned TB, unsigned cols, const fr_t* in, fr_t* out,
                 size_t ics, size_t ocs, unsigned lgX, unsigned rows_in, const LineTabs& LT, Tw tw,
                 const fr_t* lo = nullptr, const fr_t* hi = nullptr, fr_t scale = fr_t(1), unsigned flags = 0)
{
#define NTTQ_INST(LL, TT, MB) if (LOGL == LL && TB == TT) { launch<LL, TT, MODE, MB>(st, cols, in, out, ics, ocs, lgX, rows_in, LT, tw, lo, hi, scale, flags); return; }
#ifndef __HIPCC__
    NTTQ_INST(8, 16, 4) NTTQ_INST(8, 32, 2) NTTQ_INST(9, 16, 2) NTTQ_INST(9, 32, 1)
    NTTQ_INST(9, 8, 3) NTTQ_INST(10, 8, 2) NTTQ_INST(11, 8, 1)
#else
    // Shared memory is capped at 64 KB per block: no (9, 32) or (11, 8); (11, 4) instead.
    NTTQ_INST(8, 16, 4) NTTQ_INST(8, 32, 2) NTTQ_INST(9, 16, 2)
    NTTQ_INST(9, 8, 3) NTTQ_INST(10, 8, 2) NTTQ_INST(11, 4, 1)
#ifdef NTTQ_TUNE
    NTTQ_INST(10, 4, 2) NTTQ_INST(8, 8, 4) NTTQ_INST(9, 4, 4)
#endif
#endif
#undef NTTQ_INST
    throw cuda_error(cudaErrorInvalidValue, "nttq: no instance");
}

struct Tables {
    LineTabs col, row;
    fr_t* A = nullptr; fr_t* B = nullptr;
    unsigned lgN, lgLc, lgLr;
};
// lgN = lgLc + lgLr; the column line has length 2^colL (colL = lgLc, or lgLc - 2 for the coset pass)
static const Tables& tables(const gpu_t& gpu, unsigned lgN, unsigned lgLr, unsigned colL, bool inverse)
{
    static Tables tabs[16][2][24][13];
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    Tables& T = tabs[gpu.id()][inverse][lgN][lgLr];
    if (T.A == nullptr) {
        const fr_t* roots = inverse ? inverse_roots_of_unity : forward_roots_of_unity;
        T.lgN = lgN; T.lgLr = lgLr; T.lgLc = lgN - lgLr;
        T.col = make_line(gpu, roots[lgN], lgN, colL);
        T.row = make_line(gpu, roots[lgN], lgN, lgLr);
#ifdef __HIPCC__
        if (!inverse) {   // coset column pass tables (pass_h, MODE 2)
            fr_t* cw;
            const unsigned n = 64 + 4 * (1u << (colL - 8)) * hq::CS;
            CUDA_OK(risc0_cuda_malloc(&cw, n * sizeof(fr_t)));
            k_coset_tabs<<<(n + 255) / 256, 256, 0, gpu>>>(cw, roots[lgN], lgN, T.lgLc, colL);
            CUDA_OK(cudaGetLastError());
            T.col.cw = cw;
        }
#endif
        size_t nB = lgN > 12 ? (size_t)1 << (lgN - 12) : 1;
        fr_t *A, *B;
        CUDA_OK(risc0_cuda_malloc(&A, 4096 * sizeof(fr_t)));
        CUDA_OK(risc0_cuda_malloc(&B, nB * sizeof(fr_t)));
        k_pow_table<<<16, 256, 0, gpu>>>(A, roots[lgN], 4096, 1);
        k_pow_table<<<(unsigned)((nB + 255) / 256), 256, 0, gpu>>>(B, roots[lgN], (unsigned)nB, 4096);
        CUDA_OK(cudaGetLastError());
        gpu.sync();
        T.B = B;
        T.A = A;
    }
    return T;
}

// Per-size schedule (measured on RTX 5090): inverse row line 2^IR (column 2^(lg-IR)),
// forward row line 2^FR (coset column 2^(lg-FR)), TB per pass, side streams, columns per forward
// launch, columns per inverse chunk.
struct Cfg { unsigned IR, FR, ITC, ITR, FTC, FTR, NS, CHF, CHI; };
#ifdef __HIPCC__
// ROCm schedule, measured on RX 7900 XTX (gfx1100) and RX 9070 XT (gfx1201) with 211 columns
// (opt-artifacts r0amd/ntt sweep2, ntt2 sched*): instances that fit 64 KB of shared memory, no side
// streams (the forward chunk's T2 stays in the Infinity Cache). Per device (the current one): gfx11
// runs forward chunks of 48 MB of T2 (96 MB Infinity Cache), gfx12 of at most 32 MB (64 MB).
static Cfg cfg(unsigned lg) {
#ifdef NTTQ_TUNE
    if (const char* e = getenv("NTTQ_CFG")) {
        Cfg c;
        if (sscanf(e, "%u,%u,%u,%u,%u,%u,%u,%u,%u", &c.IR, &c.FR, &c.ITC, &c.ITR, &c.FTC, &c.FTR, &c.NS, &c.CHF, &c.CHI) == 9)
            return c;
    }
#endif
    int dev;
    CUDA_OK(cudaGetDevice(&dev));
    const bool g12 = hq::is_gfx12(dev);
    switch (lg) {
    case 16: return Cfg{8, 8, 32, 32, 32, 32, 0, 32, 32};
    case 17: return g12 ? Cfg{8, 9, 16, 32, 32, 16, 0, 16, 16} : Cfg{8, 9, 16, 32, 32, 16, 0, 24, 24};
    case 18: return g12 ? Cfg{9, 10, 16, 16, 16, 8, 0, 8, 8} : Cfg{9, 10, 16, 16, 16, 8, 0, 12, 12};
    case 19: return g12 ? Cfg{10, 10, 8, 8, 16, 8, 0, 4, 4} : Cfg{10, 10, 8, 8, 16, 8, 0, 6, 6};
    case 20: return g12 ? Cfg{10, 10, 8, 8, 8, 8, 0, 2, 4} : Cfg{10, 10, 8, 8, 8, 8, 0, 3, 3};
    default: return Cfg{11, 10, 8, 4, 4, 8, 0, 1, 1};   // 21
    }
}
#else
static Cfg cfg(unsigned lg) {
    switch (lg) {
    case 16: return Cfg{8, 8, 16, 16, 16, 16, 0, 32, 32};
    case 17: return Cfg{8, 9, 32, 32, 16, 16, 0, 32, 32};
    case 18: return Cfg{9, 10, 16, 16, 16, 8, 2, 2, 8};
    case 19: return Cfg{10, 10, 8, 8, 16, 8, 2, 1, 2};
    case 20: return Cfg{10, 10, 8, 8, 8, 8, 2, 1, 2};
    default: return Cfg{10, 10, 8, 8, 8, 8, 2, 1, 1};   // 21
    }
}
#endif

static bool supported(unsigned lg, unsigned lgB) { return lgB == 2 && lg >= 16 && lg <= 21; }

// The four-step commit NTT and LDE (this file): on unless RISC0_CUDA_NTTQ=0.
static bool enabled() {
    static const bool on = [] { const char* e = getenv("RISC0_CUDA_NTTQ"); return !(e && e[0] == '0'); }();
    return on;
}

// Scratch T2 of the forward passes: one allocation sized for the device's largest schedule, made on
// first use and never regrown. A regrow at a po2 increase came after the HAL's buffer pool and the
// new working set had filled the device, and failed (composite 20,21 in one process).
// NS == 0 uses T2[0] only (N * CHF elements); NS > 0 alternates T2[0] and T2[1].
static size_t t2_elems(unsigned lg)
{
    const Cfg G = cfg(lg);
    return ((size_t)4 << lg) * G.CHF * (G.NS ? 2 : 1);
}
static size_t t2_capacity()
{
    size_t m = 0;
    for (unsigned lg = 16; lg <= 21; lg++) m = std::max(m, t2_elems(lg));
    return m;
}

struct Side {
    cudaStream_t s[2] = {};
    cudaEvent_t ev[2] = {};
    cudaEvent_t inv = {};
    fr_t* T2 = nullptr;
    fr_t* t2(unsigned lg, unsigned x) const { return T2 + x * (t2_elems(lg) / 2); }
};
// one Side per device shared by commit and forward (both hold the mutex and join before returning)
static std::mutex side_mtx;
static Side& side(const gpu_t& gpu)
{
    static Side sides[16];
    Side& S = sides[gpu.id()];
    if (S.s[0] == nullptr) {
        CUDA_OK(risc0_cuda_malloc(&S.T2, t2_capacity() * sizeof(fr_t)));
        for (int i = 0; i < 2; i++) {
            CUDA_OK(cudaStreamCreateWithFlags(&S.s[i], cudaStreamNonBlocking));
            CUDA_OK(cudaEventCreateWithFlags(&S.ev[i], cudaEventDisableTiming));
        }
        CUDA_OK(cudaEventCreateWithFlags(&S.inv, cudaEventDisableTiming));
    }
    return S;
}

// forward LDE of `cols` natural-order coefficient columns Cn into E (issued on stream s)
static void fwd_cols(cudaStream_t s, const Tables& F, const Cfg& G, fr_t* T2, unsigned lg, unsigned kf,
                     const fr_t* Cn, fr_t* E, size_t ics = 0)
{
    const unsigned lgN = lg + 2;
    const size_t n = (size_t)1 << lg, N = n << 2;
    pass<2>(s, lg - F.lgLr, G.FTC, kf, Cn, T2, ics ? ics : n, N, F.lgLr, 0, F.col, Tw{F.A, F.B, lgN});
    pass<1>(s, F.lgLr, G.FTR, kf, T2, E, N, N, F.lgLc, 0, F.row, Tw{nullptr, nullptr, 0},
            nullptr, nullptr, fr_t(1), F_DISCARD_IN | F_CS_OUT);
}

// C (natural shifted coefficients) and E (LDE) of `cols` columns of W; W unchanged.
// lo/hi: zk-shift factor tables (nttx_tables), so lo[k & 1023] * hi[k >> 10] = g^k.
static void commit(const gpu_t& gpu, fr_t* C, fr_t* E, const fr_t* W, unsigned lg, unsigned lgB,
                   unsigned cols, const fr_t* lo, const fr_t* hi)
{
    const Cfg G = cfg(lg);
    const unsigned lgN = lg + lgB;
    const size_t n = (size_t)1 << lg, N = n << lgB;
    const Tables& I = tables(gpu, lg, G.IR, lg - G.IR, true);
    const Tables& F = tables(gpu, lgN, G.FR, lg - G.FR, false);
    std::lock_guard<std::mutex> lock(side_mtx);
    Side& S = side(gpu);
    cudaStream_t st = gpu;
    const fr_t ninv = domain_size_inverse[lg];
    unsigned rr = 0;
    for (unsigned c0 = 0; c0 < cols; c0 += G.CHI) {
        const unsigned k = std::min(G.CHI, cols - c0);
        // T1 lives in E of these columns: their forward passes are only issued below.
        fr_t* T1 = E + c0 * N;
        pass<0>(st, I.lgLc, G.ITC, k, W + c0 * n, T1, n, n, I.lgLr, 1u << I.lgLc, I.col, Tw{I.A, I.B, lg});
        pass<1>(st, I.lgLr, G.ITR, k, T1, C + c0 * n, n, n, I.lgLc, 0, I.row, Tw{nullptr, nullptr, 0},
                lo, hi, ninv, F_DISCARD_IN);
        if (G.NS == 0) {
            for (unsigned c = c0; c < c0 + k; c += G.CHF)
                fwd_cols(st, F, G, S.t2(lg, 0), lg, std::min(G.CHF, c0 + k - c), C + c * n, E + c * N);
            continue;
        }
        CUDA_OK(cudaEventRecord(S.inv, st));
        for (int i = 0; i < 2; i++) CUDA_OK(cudaStreamWaitEvent(S.s[i], S.inv, 0));
        for (unsigned c = c0; c < c0 + k; c += G.CHF) {
            const unsigned x = rr++ & 1;
            fwd_cols(S.s[x], F, G, S.t2(lg, x), lg, std::min(G.CHF, c0 + k - c), C + c * n, E + c * N);
        }
    }
    if (G.NS) for (int i = 0; i < 2; i++) {
        CUDA_OK(cudaEventRecord(S.ev[i], S.s[i]));
        CUDA_OK(cudaStreamWaitEvent(st, S.ev[i], 0));
    }
}

// E = LDE of natural-order coefficients Cn (cols columns), same scheme as commit's forward half.
// ics: column stride of Cn (default n). With ics == N, Cn may lie inside E (column c's input in
// column c's output): a chunk's input is read by its column pass before its row pass writes E.
static void forward(const gpu_t& gpu, fr_t* E, const fr_t* Cn, unsigned lg, unsigned lgB, unsigned cols,
                    size_t ics = 0)
{
    const Cfg G = cfg(lg);
    const unsigned lgN = lg + lgB;
    const size_t n = (size_t)1 << lg, N = n << lgB;
    if (ics == 0) ics = n;
    const Tables& F = tables(gpu, lgN, G.FR, lg - G.FR, false);
    std::lock_guard<std::mutex> lock(side_mtx);
    Side& S = side(gpu);
    cudaStream_t st = gpu;
    if (G.NS == 0) {
        for (unsigned c = 0; c < cols; c += G.CHF)
            fwd_cols(st, F, G, S.t2(lg, 0), lg, std::min(G.CHF, cols - c), Cn + c * ics, E + c * N, ics);
        return;
    }
    if (ics != n)
        throw cuda_error(cudaErrorInvalidValue, "nttq::forward: input stride with side streams");
    CUDA_OK(cudaEventRecord(S.inv, st));
    for (int i = 0; i < 2; i++) CUDA_OK(cudaStreamWaitEvent(S.s[i], S.inv, 0));
    unsigned rr = 0;
    for (unsigned c = 0; c < cols; c += G.CHF) {
        const unsigned x = rr++ & 1;
        fwd_cols(S.s[x], F, G, S.t2(lg, x), lg, std::min(G.CHF, cols - c), Cn + c * n, E + c * N);
    }
    for (int i = 0; i < 2; i++) {
        CUDA_OK(cudaEventRecord(S.ev[i], S.s[i]));
        CUDA_OK(cudaStreamWaitEvent(st, S.ev[i], 0));
    }
}

} // namespace nttq
