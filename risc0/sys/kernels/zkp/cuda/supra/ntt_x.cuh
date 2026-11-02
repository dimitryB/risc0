// Vendored + extended copies of sppark 0.1.14 narrow CT/GS kernels.
// Extensions (all arithmetic is identical to sppark -> bit-exact):
//   * batching: blockIdx.y selects a column (in/out column strides)
//   * out-of-place: separate input pointer for the pass (used for first pass)
//   * lg_expand (CT stage-0 only): synthesise zero-padded LDE input on load:
//       X[i] = (i & (2^e-1)) ? 0 : in[i >> e]          (in is bit-reversed coeffs)
//   * post-multiply (GS last pass only): r *= lo[k & 1023] * hi[k >> 10],
//       k = bit_rev(idx, lg) -- fuses the zk-shift (LDE_distribute_powers).
#pragma once

struct XArgs {
    const fr_t* in;       // pass input (may equal out)
    fr_t* out;            // pass output
    size_t in_stride;     // column stride (elements)
    size_t out_stride;
    unsigned lg_expand;   // CT stage 0 only
    const fr_t* mul_lo;   // GS last pass only (nullptr = none)
    const fr_t* mul_hi;
    fr_t* nat = nullptr;  // GS last pass only: also write natural-order copy
    size_t nat_stride = 0;
    bool skip_expand = false;  // CT stage 0: skip the first lg_expand (trivial) levels
};

template<int z_count, bool coalesced = false, class fr_t>
__launch_bounds__(768, 1) __global__
void _CT_X(const unsigned int radix, const unsigned int lg_domain_size,
           const unsigned int stage, const unsigned int iterations,
           XArgs a, const fr_t (*d_partial_twiddles)[WINDOW_SIZE],
           const fr_t (*d_plus_one_twiddles)[1024],
           const fr_t* d_radix6_twiddles, const fr_t* d_radixX_twiddles,
           bool is_intt, const fr_t d_domain_size_inverse)
{
    __builtin_assume(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
    __builtin_assume(radix <= 10);
    __builtin_assume(iterations <= radix);
    __builtin_assume(stage <= lg_domain_size - iterations);
    extern __shared__ fr_t shared_exchange[];

    const fr_t* d_in = a.in + blockIdx.y * a.in_stride;
    fr_t* d_out = a.out + blockIdx.y * a.out_stride;

    index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    const index_t diff_mask = (1 << (iterations - 1)) - 1;
    const index_t inp_mask = ((index_t)1 << stage) - 1;
    const index_t out_mask = ((index_t)1 << (stage + iterations - 1)) - 1;

    const index_t tiz = (tid & ~diff_mask) * z_count + (tid & diff_mask);
    const index_t thread_ntt_pos = (tiz >> (iterations - 1)) & inp_mask;

    index_t idx0 = (tiz & ~out_mask) | ((tiz << stage) & out_mask);
    idx0 = idx0 * 2 + thread_ntt_pos;
    index_t idx1 = idx0 + ((index_t)1 << stage);

    fr_t r[2][z_count];

    if (coalesced) {
        coalesced_load<z_count>(r[0], d_in, idx0, stage + 1);
        coalesced_load<z_count>(r[1], d_in, idx1, stage + 1);
        transpose<z_count>(r[0]);
        __syncwarp();
        transpose<z_count>(r[1]);
    } else {
        unsigned int z_shift = inp_mask==0 ? iterations : 0;
        if (a.lg_expand && a.skip_expand) {
            // first lg_expand DIT levels on [c,0,..,0] produce [c,c,..,c]: replicate, skip them
            #pragma unroll
            for (int z = 0; z < z_count; z++)
                r[0][z] = r[1][z] = d_in[(idx0 + (z << z_shift)) >> a.lg_expand];
        } else if (a.lg_expand) {
            const index_t emask = ((index_t)1 << a.lg_expand) - 1;
            #pragma unroll
            for (int z = 0; z < z_count; z++) {
                index_t i0 = idx0 + (z << z_shift), i1 = idx1 + (z << z_shift);
                r[0][z].zero(); r[1][z].zero();
                if ((i0 & emask) == 0) r[0][z] = d_in[i0 >> a.lg_expand];
                if ((i1 & emask) == 0) r[1][z] = d_in[i1 >> a.lg_expand];
            }
        } else {
            #pragma unroll
            for (int z = 0; z < z_count; z++) {
                r[0][z] = d_in[idx0 + (z << z_shift)];
                r[1][z] = d_in[idx1 + (z << z_shift)];
            }
        }
    }

    if (stage != 0) {
        unsigned int thread_ntt_idx = (tiz & diff_mask) * 2;
        unsigned int nbits = MAX_LG_DOMAIN_SIZE - stage;
        index_t idx0 = bit_rev(thread_ntt_idx, nbits);
        index_t root_idx0 = idx0 * thread_ntt_pos;
        index_t root_idx1 = root_idx0 + (thread_ntt_pos << (nbits - 1));

        fr_t first_root, second_root;
#ifdef NOTW
        first_root = d_partial_twiddles[0][root_idx0 & 63]; second_root = d_partial_twiddles[0][root_idx1 & 63];
#else
        get_intermediate_roots(first_root, second_root,
                               root_idx0, root_idx1, d_partial_twiddles);
#endif
        r[0][0] = r[0][0] * first_root;
        r[1][0] = r[1][0] * second_root;

        if (z_count > 1) {
            unsigned int off = nbits >= 10 ? (nbits - 10) : 0;
            unsigned int scale = nbits >= 10 ? 0 : (10 - nbits);

            thread_ntt_idx <<= scale;
            fr_t first_root_z = d_plus_one_twiddles[off][thread_ntt_idx];
            fr_t second_root_z = d_plus_one_twiddles[off][thread_ntt_idx + (1<<scale)];

            #pragma unroll
            for (int z = 1; z < z_count; z++) {
                first_root *= first_root_z;
                second_root *= second_root_z;
                r[0][z] = r[0][z] * first_root;
                r[1][z] = r[1][z] * second_root;
            }
        }
    }

    const unsigned s0 = (a.lg_expand && a.skip_expand) ? a.lg_expand : 1u;
    if (s0 == 1) {
    #pragma unroll
    for (int z = 0; z < z_count; z++) {
        fr_t t = r[1][z];
        r[1][z] = r[0][z] - t;
        r[0][z] = r[0][z] + t;
    }
    }

    #pragma unroll 1
    for (unsigned int s = s0; s < min(iterations, 6u); s++) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t root = d_radix6_twiddles[rank << (6 - (s + 1))];

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = fr_t::csel(r[1][z], r[0][z], pos);
            t.shfl_bfly(laneMask);
            r[0][z] = fr_t::csel(r[0][z], t, pos);
            r[1][z] = fr_t::csel(t, r[1][z], pos);
            t = root * r[1][z];
            r[1][z] = r[0][z] - t;
            r[0][z] = r[0][z] + t;
        }
    }

    #pragma unroll 1
    for (unsigned int s = 6; s < iterations; s++) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t root = d_radixX_twiddles[rank << (radix - (s + 1))];

        fr_t (*xchg)[z_count] = reinterpret_cast<decltype(xchg)>(shared_exchange);

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = fr_t::csel(r[1][z], r[0][z], pos);
            xchg[threadIdx.x][z] = t;
        }
        __syncthreads();
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = xchg[threadIdx.x ^ laneMask][z];
            r[0][z] = fr_t::csel(r[0][z], t, pos);
            r[1][z] = fr_t::csel(t, r[1][z], pos);
            t = root * r[1][z];
            r[1][z] = r[0][z] - t;
            r[0][z] = t + r[0][z];
        }
        __syncthreads();
    }

    if (is_intt && (stage + iterations) == lg_domain_size) {
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = r[0][z] * d_domain_size_inverse;
            r[1][z] = r[1][z] * d_domain_size_inverse;
        }
    }

    index_t mask = (index_t)((1 << iterations) - 1) << stage;
    index_t rotw = idx0 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx0 = (idx0 & ~mask) | (rotw & mask);
    rotw = idx1 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx1 = (idx1 & ~mask) | (rotw & mask);

#ifndef NTTX_NO_BARRIER_FIX
    // V3: same barrier as the c1/host sppark fix (WAR on shared_exchange before transpose)
    if (coalesced && iterations > 6) __syncthreads();
#endif
    if (coalesced) {
        transpose<z_count>(r[0]);
        __syncwarp();
        transpose<z_count>(r[1]);
        coalesced_store<z_count>(d_out, idx0, r[0], stage);
        coalesced_store<z_count>(d_out, idx1, r[1], stage);
    } else {
        unsigned int z_shift = inp_mask==0 ? iterations : 0;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            d_out[idx0 + (z << z_shift)] = r[0][z];
            d_out[idx1 + (z << z_shift)] = r[1][z];
        }
    }
}

template<int z_count, bool coalesced = false, class fr_t>
__launch_bounds__(768, 1) __global__
void _GS_X(const unsigned int radix, const unsigned int lg_domain_size,
           const unsigned int stage, const unsigned int iterations,
           XArgs a, const fr_t (*d_partial_twiddles)[WINDOW_SIZE],
           const fr_t (*d_plus_one_twiddles)[1024],
           const fr_t* d_radix6_twiddles, const fr_t* d_radixX_twiddles,
           bool is_intt, const fr_t d_domain_size_inverse)
{
    __builtin_assume(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
    __builtin_assume(radix <= 10);
    __builtin_assume(iterations <= radix);
    __builtin_assume(stage <= lg_domain_size && stage >= iterations);
    extern __shared__ fr_t shared_exchange[];

    const fr_t* d_in = a.in + blockIdx.y * a.in_stride;
    fr_t* d_out = a.out + blockIdx.y * a.out_stride;

    index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    const index_t diff_mask = (1 << (iterations - 1)) - 1;
    const index_t inp_mask = ((index_t)1 << (stage - 1)) - 1;
    const index_t out_mask = ((index_t)1 << (stage - iterations)) - 1;

    const index_t tiz = (tid & ~diff_mask) * z_count + (tid & diff_mask);

    index_t idx0 = (tiz & ~inp_mask) * 2;
    idx0 += (tiz << (stage - iterations)) & inp_mask;
    idx0 += (tiz >> (iterations - 1)) & out_mask;
    index_t idx1 = idx0 + ((index_t)1 << (stage - 1));

    fr_t r[2][z_count];

    if (coalesced) {
        coalesced_load<z_count>(r[0], d_in, idx0, stage - iterations);
        coalesced_load<z_count>(r[1], d_in, idx1, stage - iterations);
        transpose<z_count>(r[0]);
        __syncwarp();
        transpose<z_count>(r[1]);
    } else {
        unsigned int z_shift = out_mask==0 ? iterations : 0;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = d_in[idx0 + (z << z_shift)];
            r[1][z] = d_in[idx1 + (z << z_shift)];
        }
    }

    #pragma unroll 1
    for (unsigned int s = iterations; --s >= 6;) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t root = d_radixX_twiddles[rank << (radix - (s + 1))];

        fr_t (*xchg)[z_count] = reinterpret_cast<decltype(xchg)>(shared_exchange);

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = r[0][z] - r[1][z];
            r[0][z] = r[0][z] + r[1][z];
            r[1][z] = t * root;
            t = fr_t::csel(r[1][z], r[0][z], pos);
            xchg[threadIdx.x][z] = t;
        }
        __syncthreads();
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = xchg[threadIdx.x ^ laneMask][z];
            r[0][z] = fr_t::csel(r[0][z], t, pos);
            r[1][z] = fr_t::csel(t, r[1][z], pos);
        }
        __syncthreads();
    }

    #pragma unroll 1
    for (unsigned int s = min(iterations, 6u); --s >= 1;) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t root = d_radix6_twiddles[rank << (6 - (s + 1))];

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = r[0][z] - r[1][z];
            r[0][z] = r[0][z] + r[1][z];
            r[1][z] = t * root;
            t = fr_t::csel(r[1][z], r[0][z], pos);
            t.shfl_bfly(laneMask);
            r[0][z] = fr_t::csel(r[0][z], t, pos);
            r[1][z] = fr_t::csel(t, r[1][z], pos);
        }
    }

    #pragma unroll
    for (int z = 0; z < z_count; z++) {
        fr_t t = r[0][z] - r[1][z];
        r[0][z] = r[0][z] + r[1][z];
        r[1][z] = t;
    }

    if (stage - iterations != 0) {
        index_t thread_ntt_pos = (tiz & inp_mask) >> (iterations - 1);
        unsigned int thread_ntt_idx = (tiz & diff_mask) * 2;
        unsigned int nbits = MAX_LG_DOMAIN_SIZE - (stage - iterations);
        index_t idx0 = bit_rev(thread_ntt_idx, nbits);
        index_t root_idx0 = idx0 * thread_ntt_pos;
        index_t root_idx1 = root_idx0 + (thread_ntt_pos << (nbits - 1));

        fr_t first_root, second_root;
#ifdef NOTW
        first_root = d_partial_twiddles[0][root_idx0 & 63]; second_root = d_partial_twiddles[0][root_idx1 & 63];
#else
        get_intermediate_roots(first_root, second_root,
                               root_idx0, root_idx1, d_partial_twiddles);
#endif
        r[0][0] = r[0][0] * first_root;
        r[1][0] = r[1][0] * second_root;

        if (z_count > 1) {
            unsigned int off = nbits >= 10 ? (nbits - 10) : 0;
            unsigned int scale = nbits >= 10 ? 0 : (10 - nbits);

            thread_ntt_idx <<= scale;
            fr_t first_root_z = d_plus_one_twiddles[off][thread_ntt_idx];
            fr_t second_root_z = d_plus_one_twiddles[off][thread_ntt_idx + (1<<scale)];

            #pragma unroll
            for (int z = 1; z < z_count; z++) {
                first_root *= first_root_z;
                second_root *= second_root_z;
                r[0][z] = r[0][z] * first_root;
                r[1][z] = r[1][z] * second_root;
            }
        }
    }

    if (is_intt && stage == iterations) {
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = r[0][z] * d_domain_size_inverse;
            r[1][z] = r[1][z] * d_domain_size_inverse;
        }
    }

    index_t mask = (index_t)((1 << iterations) - 1) << (stage - iterations);
    index_t rotw = idx0 & mask;
    rotw = (rotw << 1) | (rotw >> (iterations - 1));
    idx0 = (idx0 & ~mask) | (rotw & mask);
    rotw = idx1 & mask;
    rotw = (rotw << 1) | (rotw >> (iterations - 1));
    idx1 = (idx1 & ~mask) | (rotw & mask);

#ifndef NTTX_NO_BARRIER_FIX
    // V3: same barrier as the c1/host sppark fix (WAR on shared_exchange before transpose)
    if (coalesced && iterations > 6) __syncthreads();
#endif
    if (coalesced) {
        transpose<z_count>(r[0]);
        __syncwarp();
        transpose<z_count>(r[1]);
        coalesced_store<z_count>(d_out, idx0, r[0], stage - iterations + 1);
        coalesced_store<z_count>(d_out, idx1, r[1], stage - iterations + 1);
    } else {
        unsigned int z_shift = out_mask==0 ? iterations : 0;
        const bool post = a.mul_lo != nullptr && stage == iterations;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            index_t i0 = idx0 + (z << z_shift), i1 = idx1 + (z << z_shift);
            fr_t v0 = r[0][z], v1 = r[1][z];
            if (post) {
                index_t k0 = bit_rev(i0, lg_domain_size), k1 = bit_rev(i1, lg_domain_size);
                v0 = v0 * (a.mul_lo[k0 & 1023] * a.mul_hi[k0 >> 10]);
                v1 = v1 * (a.mul_lo[k1 & 1023] * a.mul_hi[k1 >> 10]);
            }
            d_out[i0] = v0;
            d_out[i1] = v1;
            if (a.nat != nullptr && stage == iterations) {
                fr_t* nat = a.nat + blockIdx.y * a.nat_stride;
                nat[bit_rev(i0, lg_domain_size)] = v0;
                nat[bit_rev(i1, lg_domain_size)] = v1;
            }
        }
    }
}

// Launchers mirroring sppark's step selection.
struct X_launcher {
    const NTTParameters& p;
    const stream_t& stream;
    unsigned cols;
    bool intt;

    template<bool CT>
    void step(int lg, int stage, int iterations, XArgs a)
    {
        const int radix = iterations < 6 ? 6 : iterations;
        index_t num_threads = (index_t)1 << (lg - 1);
        index_t block_size = 1 << (radix - 1);
        block_size = (num_threads <= block_size) ? num_threads : block_size;
        index_t num_blocks = (num_threads + block_size - 1) / block_size;
        const int Z_COUNT = 256/8/sizeof(fr_t);
        size_t shared_sz = sizeof(fr_t) << (radix - 1);
#define XARGS radix, lg, stage, iterations, a, p.partial_twiddles, p.plus_one_twiddles, \
              p.twiddles[0], p.twiddles[radix-6], intt, domain_size_inverse[lg]
        if (CT) {
            if (num_blocks < Z_COUNT)
                _CT_X<1><<<dim3(num_blocks, cols), block_size, shared_sz, stream>>>(XARGS);
            else if (stage == 0 || lg < 12)
                _CT_X<Z_COUNT><<<dim3(num_blocks/Z_COUNT, cols), block_size, Z_COUNT*shared_sz, stream>>>(XARGS);
            else
                _CT_X<Z_COUNT, true><<<dim3(num_blocks/Z_COUNT, cols), block_size, Z_COUNT*shared_sz, stream>>>(XARGS);
        } else {
            if (num_blocks < Z_COUNT)
                _GS_X<1><<<dim3(num_blocks, cols), block_size, shared_sz, stream>>>(XARGS);
            else if (stage == iterations || lg < 12)
                _GS_X<Z_COUNT><<<dim3(num_blocks/Z_COUNT, cols), block_size, Z_COUNT*shared_sz, stream>>>(XARGS);
            else
                _GS_X<Z_COUNT, true><<<dim3(num_blocks/Z_COUNT, cols), block_size, Z_COUNT*shared_sz, stream>>>(XARGS);
        }
#undef XARGS
        CUDA_OK(cudaGetLastError());
    }
};

// Same step splits as sppark CT_NTT / GS_NTT (lg in 11..30).
static void ct_steps(int lg, int st[3], int& n) {
    if (lg <= 10) { st[0] = lg; n = 1; }
    else if (lg <= 18) { int s = lg/2; st[0] = s + lg%2; st[1] = s; n = 2; }
    else { int s = lg/3, rem = lg%3; st[0] = s; st[1] = s + (lg==29); st[2] = s + (lg==29 ? 1 : rem); n = 3; }
}
#ifdef __HIPCC__
// RDNA4 (gfx12): the last CT pass, whose rows are 2^stage elements apart, runs at a third of the
// other passes' bandwidth when it is long (lg 22: 7,7,8 = 12 + 12 + 40 ms for 211 columns on an
// RX 9070 XT; 8,8,6 = 12 + 12 + 17 ms). Keep it short there; RDNA3 keeps sppark's split.
static bool hip_is_gfx12(int dev) {
    static int v[16] = {};
    if (v[dev] == 0) v[dev] = std::strncmp(gpu_props(dev).gcnArchName, "gfx12", 5) == 0 ? 1 : 2;
    return v[dev] == 1;
}
#endif
static void gs_steps(int lg, int st[3], int& n) {
    if (lg <= 10) { st[0] = lg; n = 1; }
    else if (lg <= 18) { int s = lg/2; st[0] = s; st[1] = s + lg%2; n = 2; }
    else { int s = lg/3, rem = lg%3; st[0] = s + (lg==29 ? 1 : rem); st[1] = s + (lg==29); st[2] = s; n = 3; }
}

// Forward CT (RN), first pass out-of-place from `in` (optionally zero-expand synth).
static void ct_x(const stream_t& s, unsigned cols, const fr_t* in, size_t in_stride, unsigned lg_expand,
                 fr_t* out, size_t out_stride, int lg, bool skip = false)
{
    const auto& p = NTTParameters::all(false)[s];
    X_launcher L{p, s, cols, false};
    int st[3], n; ct_steps(lg, st, n);
#ifdef __HIPCC__
    if (n == 3 && hip_is_gfx12(s.id())) {
        st[0] = st[1] = (lg + 2) / 3;
        st[2] = lg - 2 * st[0];
    }
#endif
    int stage = 0;
    for (int i = 0; i < n; i++) {
        XArgs a{i == 0 ? in : out, out, i == 0 ? in_stride : out_stride, out_stride,
                i == 0 ? lg_expand : 0u, nullptr, nullptr};
        a.skip_expand = skip && i == 0 && lg_expand > 0 && lg_expand < 6 && (unsigned)st[0] > lg_expand;
        L.step<true>(lg, stage, st[i], a);
        stage += st[i];
    }
}

// Inverse GS (NR), first pass out-of-place from `in`, optional post-multiply on last pass.
static void gs_x(const stream_t& s, unsigned cols, const fr_t* in, size_t in_stride,
                 fr_t* out, size_t out_stride, int lg, const fr_t* lo, const fr_t* hi,
                 fr_t* nat = nullptr, size_t nat_stride = 0)
{
    const auto& p = NTTParameters::all(true)[s];
    X_launcher L{p, s, cols, true};
    int st[3], n; gs_steps(lg, st, n);
    int stage = lg;
    for (int i = 0; i < n; i++) {
        XArgs a{i == 0 ? in : out, out, i == 0 ? in_stride : out_stride, out_stride, 0u,
                i == n-1 ? lo : nullptr, i == n-1 ? hi : nullptr,
                i == n-1 ? nat : nullptr, nat_stride};
        L.step<false>(lg, stage, st[i], a);
        stage -= st[i];
    }
}

// Batched in-place/out-of-place bit reversal (sppark bit_rev_permutation_z + column offset).
template<unsigned int Z_COUNT, class fr_t>
__launch_bounds__(192, 2) __global__
void bit_rev_z_batched(fr_t* out0, const fr_t* in0, uint32_t lg_domain_size, size_t stride)
{
    const uint32_t LG_Z_COUNT = lg2(Z_COUNT);
    fr_t* out = out0 + blockIdx.y * stride;
    const fr_t* in = in0 + blockIdx.y * stride;

    extern __shared__ int xchg_bit_rev[];
    fr_t (*xchg)[Z_COUNT][Z_COUNT] = reinterpret_cast<decltype(xchg)>(xchg_bit_rev);

    uint32_t gid = threadIdx.x / Z_COUNT;
    uint32_t idx = threadIdx.x % Z_COUNT;
    uint32_t rev = bit_rev(idx, LG_Z_COUNT);

    index_t step = (index_t)1 << (lg_domain_size - LG_Z_COUNT);
    index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    #pragma unroll 1
    do {
        (Z_COUNT > warpSize) ? __syncthreads() : __syncwarp();
        index_t group_idx = tid >> LG_Z_COUNT;
        index_t group_rev = bit_rev(group_idx, lg_domain_size - 2*LG_Z_COUNT);
        if (group_idx > group_rev)
            continue;
        index_t base_idx = group_idx * Z_COUNT + idx;
        index_t base_rev = group_rev * Z_COUNT + idx;
        fr_t regs[Z_COUNT];
        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++) {
            xchg[gid][i][rev] = (regs[i] = in[i * step + base_idx]);
            if (group_idx != group_rev)
                regs[i] = in[i * step + base_rev];
        }
        (Z_COUNT > warpSize) ? __syncthreads() : __syncwarp();
        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            out[i * step + base_rev] = xchg[gid][rev][i];
        if (group_idx == group_rev)
            continue;
        (Z_COUNT > warpSize) ? __syncthreads() : __syncwarp();
        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            xchg[gid][i][rev] = regs[i];
        (Z_COUNT > warpSize) ? __syncthreads() : __syncwarp();
        #pragma unroll
        for (uint32_t i = 0; i < Z_COUNT; i++)
            out[i * step + base_idx] = xchg[gid][rev][i];
    } while (Z_COUNT <= WARP_SZ && (tid += blockDim.x*gridDim.x) < step);
}

#ifdef __HIPCC__
// ROCm: bit_rev_z_batched<64> keeps 64 values per thread in registers and spills on RDNA3/4
// (256 VGPRs, ~300 B/lane scratch); a kernel with scratch also costs ~35-115 us at each
// dispatch boundary there. This one swaps the Z x Z tiles g and rev(g) of a column through two
// padded LDS tiles (no register array, no scratch, conflict-free transposed reads): element
// (i, c) of tile g, at i*step + g*Z + c, moves to rev(c)*step + rev(g)*Z + rev(i), i.e. the
// same bit reversal. Every block reads both of its tiles before writing either, and distinct
// blocks own disjoint tile pairs, so in place is safe. Harness: r0amd r8/L4/brev (bit-identical,
// 2-3x faster at lg 18-20).
template<unsigned int Z, unsigned int NT>
__launch_bounds__(NT) __global__
void bit_rev_t2_batched(fr_t* io0, uint32_t lg, size_t stride)
{
    static_assert(sizeof(fr_t) == sizeof(uint32_t), "bit_rev_t2_batched moves 32-bit elements");
    constexpr uint32_t LZ = lg2(Z);
    __shared__ uint32_t A[Z][Z + 1], B[Z][Z + 1];
    uint32_t* io = reinterpret_cast<uint32_t*>(io0 + blockIdx.y * stride);
    const index_t g = blockIdx.x;
    const index_t gr = bit_rev(g, lg - 2 * LZ);
    if (g > gr)
        return;
    const index_t step = (index_t)1 << (lg - LZ);
    const bool two = g != gr;
    #pragma unroll 4
    for (uint32_t e = threadIdx.x; e < Z * Z; e += NT) {
        const uint32_t i = e / Z, c = e % Z;
        A[i][c] = io[i * step + g * Z + c];
        if (two)
            B[i][c] = io[i * step + gr * Z + c];
    }
    __syncthreads();
    #pragma unroll 4
    for (uint32_t e = threadIdx.x; e < Z * Z; e += NT) {
        const uint32_t r = e / Z, c = e % Z;
        const uint32_t rc = bit_rev(c, LZ), rr = bit_rev(r, LZ);
        io[r * step + gr * Z + c] = A[rc][rr];
        if (two)
            io[r * step + g * Z + c] = B[rc][rr];
    }
}
#endif

#ifdef __HIPCC__
// Out-of-place bit_rev_t2_batched: column c of in (stride is) bit-reversed into column c of out
// (stride os); in and out must not overlap.
template<unsigned int Z, unsigned int NT>
__launch_bounds__(NT) __global__
void bit_rev_t2_oop(fr_t* out0, const fr_t* __restrict__ in0, uint32_t lg, size_t is, size_t os)
{
    constexpr uint32_t LZ = lg2(Z);
    __shared__ uint32_t A[Z][Z + 1], B[Z][Z + 1];
    const uint32_t* in = reinterpret_cast<const uint32_t*>(in0 + blockIdx.y * is);
    uint32_t* out = reinterpret_cast<uint32_t*>(out0 + blockIdx.y * os);
    const index_t g = blockIdx.x;
    const index_t gr = bit_rev(g, lg - 2 * LZ);
    if (g > gr)
        return;
    const index_t step = (index_t)1 << (lg - LZ);
    const bool two = g != gr;
    #pragma unroll 4
    for (uint32_t e = threadIdx.x; e < Z * Z; e += NT) {
        const uint32_t i = e / Z, c = e % Z;
        A[i][c] = in[i * step + g * Z + c];
        if (two)
            B[i][c] = in[i * step + gr * Z + c];
    }
    __syncthreads();
    #pragma unroll 4
    for (uint32_t e = threadIdx.x; e < Z * Z; e += NT) {
        const uint32_t r = e / Z, c = e % Z;
        const uint32_t rc = bit_rev(c, LZ), rr = bit_rev(r, LZ);
        out[r * step + gr * Z + c] = A[rc][rr];
        if (two)
            out[r * step + g * Z + c] = B[rc][rr];
    }
}
#endif

static void bitrev_batched(const stream_t& s, fr_t* io, unsigned lg, unsigned cols)
{
    const uint32_t Z = 64;   // 256 / sizeof(fr_t)
    size_t n = (size_t)1 << lg;
    if (n < Z * Z) {
        for (unsigned c = 0; c < cols; c++)
            bit_rev_permutation<<<n <= 1024 ? 1 : n / 64, n <= 1024 ? n : 64, 0, s>>>(io + c * n, io + c * n, lg);
    } else {
#ifdef __HIPCC__
        bit_rev_t2_batched<32, 128><<<dim3(n / 32 / 32, cols), 128, 0, s>>>(io, lg, n);
#else
        bit_rev_z_batched<Z><<<dim3(n / Z / Z, cols), Z, Z * Z * sizeof(fr_t), s>>>(io, io, lg, n);
#endif
    }
    CUDA_OK(cudaGetLastError());
}

// ======================= V3 integration API =======================
// zk-shift tables: lo[t] * hi[u] = g^(t + 1024u), built from sppark's own LDE_powers on an
// all-ones vector (pw[i] = g^bitrev(i)), so the factors are the same field elements.
__global__ void nttx_fill_one(fr_t* p, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = fr_t(1);
}
__global__ void nttx_gather_pow(fr_t* lo, fr_t* hi, const fr_t* pw, unsigned lg) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned n = 1u << lg;
    if (t < 1024) lo[t] = (t < n) ? pw[bit_rev((index_t)t, lg)] : pw[0];
    const unsigned nhi = lg > 10 ? (1u << (lg - 10)) : 1u;
    if (t < nhi) hi[t] = lg > 10 ? pw[bit_rev((index_t)t << 10, lg)] : pw[0];
}
// batched zk shift on bit-reversed data (replacement for per-column LDE_distribute_powers)
__global__ void nttx_zk_shift_batched(fr_t* io, unsigned lg, size_t total, const fr_t* lo, const fr_t* hi) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= total) return;
    index_t k = bit_rev((index_t)(i & (((size_t)1 << lg) - 1)), lg);
    io[i] = io[i] * (lo[k & 1023] * hi[k >> 10]);
}

struct NttxTables { fr_t* lo = nullptr; fr_t* hi = nullptr; };

static const NttxTables& nttx_tables(const gpu_t& gpu, unsigned lg) {
    static NttxTables tabs[16][32];
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    int dev = gpu.id();
    NttxTables& T = tabs[dev][lg];
    if (T.lo == nullptr) {
        size_t n = (size_t)1 << lg;
        size_t nhi = lg > 10 ? ((size_t)1 << (lg - 10)) : 1;
        fr_t* pw;
        CUDA_OK(cudaMalloc(&pw, n * sizeof(fr_t)));
        CUDA_OK(cudaMalloc(&T.lo, 1024 * sizeof(fr_t)));
        CUDA_OK(cudaMalloc(&T.hi, nhi * sizeof(fr_t)));
        nttx_fill_one<<<(unsigned)((n + 255) / 256), 256, 0, gpu>>>(pw, n);
        NTT::LDE_powers(gpu, pw, lg);
        size_t m = nhi > 1024 ? nhi : 1024;
        nttx_gather_pow<<<(unsigned)((m + 255) / 256), 256, 0, gpu>>>(T.lo, T.hi, pw, lg);
        gpu.sync();
        CUDA_OK(cudaFree(pw));
    }
    return T;
}

static void nttx_batch_zk_shift(const gpu_t& gpu, fr_t* io, unsigned lg, unsigned cols) {
    const NttxTables& T = nttx_tables(gpu, lg);
    size_t total = ((size_t)cols) << lg;
    nttx_zk_shift_batched<<<(unsigned)((total + 255) / 256), 256, 0, gpu>>>(io, lg, total, T.lo, T.hi);
    CUDA_OK(cudaGetLastError());
}

// L2-sized chunk: ~96 MB / (bytes touched per column = 4n coeffs + 4N evals).
static unsigned nttx_auto_chunk(unsigned lg, unsigned lgB) {
    size_t per_col = ((size_t)4 << lg) + ((size_t)4 << (lg + lgB));
    size_t c = ((size_t)96 << 20) / per_col;
    unsigned k = 1;
    while ((size_t)k * 2 <= c && k < 64) k *= 2;
    return k;
}

// Fused commit: C = bitrev(zk_shift(iNTT(W))) (natural-order shifted coeffs, == today's
// coeffs after PolyGroup::new), E = NTT_RN(expand(zk_shift(iNTT(W)))). W is not modified.
#include "ntt_q.cuh"

static void nttx_commit(const gpu_t& gpu, fr_t* C, fr_t* E, const fr_t* W, unsigned lg, unsigned lgB,
                        unsigned cols, unsigned chunk) {
    const NttxTables& T = nttx_tables(gpu, lg);
    if (chunk == 0 && nttq::supported(lg, lgB) && nttq::enabled()) {
        nttq::commit(gpu, C, E, W, lg, lgB, cols, T.lo, T.hi);
        return;
    }
    if (chunk == 0) chunk = nttx_auto_chunk(lg, lgB);
    size_t n = (size_t)1 << lg, N = n << lgB;
    for (unsigned c0 = 0; c0 < cols; c0 += chunk) {
        unsigned k = std::min(chunk, cols - c0);
        gs_x(gpu, k, W + c0 * n, n, C + c0 * n, n, (int)lg, T.lo, T.hi);
        ct_x(gpu, k, C + c0 * n, n, lgB, E + c0 * N, N, (int)(lg + lgB));
        bitrev_batched(gpu, C + c0 * n, lg, k);
    }
}
