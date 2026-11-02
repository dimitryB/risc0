#!/usr/bin/env python3
# Copyright 2026 RISC Zero, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The poly_mix chains of the ROCm eval_check on the RDNA4 (gfx12) matrix units: regen.py applies
# convert() to to_hip.py's output. RDNA's 32-bit multiplies are quarter rate, and a chain link
# (rk_link_k / rk_link_nf_k with an Fp term) is four 32x32->64 multiply-adds; an int8 WMMA on an
# RX 9070 XT does 3.8x the field products of v_mad_u64_u32.
#
# A chain is base + sum_l c_l * pm[k_l] (FpExt pm, Fp c, Montgomery raw values) followed by
# rk_fin's REDC: canonical(base + R^-1 * sum_l c_l * pm_l). On gfx12 each lane (one point) stores
# its c_l in an LDS tile [slot][point] instead; per 4 links and 16-point tile one
# V_WMMA_I32_16X16X16_IU8 computes D[(j, b)][p] = sum_{l, a} Q[k_l][(j, b)][a] * byte_a(c_l[p]),
# Q[k][(j, b)][a] = byte_b(pm_raw[k][j] * 256^a mod P), so T_j = sum_b D[(j, b)] << 8b is
# congruent to sum_l c_l * pm_l[j] (exact in i32 for <= 64 links: x = d0 + 256 d1 < 2^32), and
# base + REDC(T_j) is the same canonical value as the scalar chain (bit-exact). The LDS
# addressing does the point-per-lane -> operand-layout transposition both ways; the A operand
# rows (Q) come from ef_wq_raw, filled per call by ef_prep_q. Lanes hold D^T, i.e. one (point,
# component) per lane and register group, so the combine and REDC run there and only the 4
# reduced words per point go back through LDS.
#
# Per chain with >= wmin Fp links and a free staging region (the current circuit's chains never
# overlap: one region suffices, an overlapping chain would stay scalar): each Fp link becomes an
# ef_wst store into a ring of `slots` LDS slots; each block of 4 staged links one ef_wblk (4 WMMAs,
# one per 16-point tile), issued `lag` blocks later (inc) or per segment (burst) so the tile reads
# overlap later work, its A operand loaded at the block's first link (qpre); every `group` links
# (<= 64, exact i32 sums) and at the chain end an ef_wfin flush. FpExt links stay scalar in an
# RkAcc (efs_N) that the flush adds. Every rewritten line keeps the scalar statement as
# EF_W((gfx12 form), scalar form), so other targets (gfx11, and CUDA, which never runs this pass)
# compile exactly the old code.
#
# The defaults are regen.py's (an RX 9070 XT at po2 20, ef/ech harness, medians of 5 interleaved
# runs: 256.9 -> 235.3 ms, bit-exact; burst form 238.1, without the lag 236.0, wmin 4 235.5,
# wmin 8 237.9, 16 slots 236.3).
#
# usage: wmma_hip.py <in.cu> <out.cu> [--wmin N] [--slots N] [--group N] [--regions N] [--inc 0|1]
#                    [--lag N] [--qpre 0|1]
import re
import sys

LINK = re.compile(r'^  RkAcc (x\d+) = (rk_link(?:_nf)?_k)\((\w+), (\w+), (\d+)u\);$')
FIN = re.compile(r'^  FpExt (x\d+) = rk_fin\((rk_link(?:_nf)?_k)\((\w+), (\w+), (\d+)u\)\);$')

HELPERS = r'''// ---- wmma_hip.py: poly_mix chains on the gfx12 matrix units (V_WMMA_I32_16X16X16_IU8) ----
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__)) && !defined(EF_NO_WMMA)
#define EF_WM 1
#endif
#define EF_WUNP(...) __VA_ARGS__
#ifdef EF_WM
#define EF_W(w, ...) EF_WUNP w
#else
#define EF_W(w, ...) __VA_ARGS__
#endif
#ifdef EF_WM
typedef int ef_v4i __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(3))) uint32_t ef_lu32;
typedef const __attribute__((address_space(1))) uint32_t ef_wgu32;
#define EF_WSLOTS %(slots_total)du  // staging slots per wave (%(regions)d regions of %(slots)d)
#define EF_WSTR 80u    // dwords per slot: 64 points + 16 pad, so the tile reads are bank-conflict free
__shared__ uint32_t ef_wlds[(EF_THREADS / 64) * EF_WSLOTS * EF_WSTR];
// lane, its wave's staging tile, rdo = (lane / 16) * EF_WSTR + lane %% 16 (operand row group = link)
struct EfW { ef_lu32* st; ef_wgu32* q; uint32_t lane, rdo; };
__device__ __forceinline__ static EfW ef_wsetup() {
  EfW w;
  w.lane = threadIdx.x & 63u;
  w.rdo = (w.lane >> 4) * EF_WSTR + (w.lane & 15u);
  w.st = (ef_lu32*)(ef_wlds + (threadIdx.x >> 6) * (EF_WSLOTS * EF_WSTR));
  // The A operand table through an opaque SGPR base (an empty asm): each ef_wq load is then base +
  // lane offset + an immediate offset, instead of a 64-bit PC-relative SGPR address per constant
  // that the scheduler hoisted and spilled to VGPR lanes (~600 s_getpc and ~970 v_readlane per wave).
  uint64_t q = reinterpret_cast<uint64_t>(ef_wq_raw);
  asm("" : "+s"(q));
  w.q = (ef_wgu32*)q;
  return w;
}
#define EF_WSETUP const EfW efw = ef_wsetup();
__device__ __forceinline__ static void ef_wst(const EfW& w, uint32_t slot, Fp c) { w.st[slot * EF_WSTR + w.lane] = c.asRaw(); }
// A operand: Q rows of the links with constants k0..k0+3, or ka..kd with n < 4 real links (zero rows)
__device__ __forceinline__ static int ef_wq(const EfW& w, uint32_t k0) { return (int)(w.q + k0 * 16u)[w.lane]; }
__device__ __forceinline__ static int ef_wq4(const EfW& w, uint32_t ka, uint32_t kb, uint32_t kc, uint32_t kd, uint32_t n) {
  const uint32_t g = w.lane >> 4, k = g == 0 ? ka : g == 1 ? kb : g == 2 ? kc : kd;
  const int v = (int)w.q[k * 16u + (w.lane & 15u)];
  return g < n ? v : 0;
}
// 4 links (slots s0..s0+3) x 4 point tiles: unsigned bytes, i32 accumulation
__device__ __forceinline__ static void ef_wblk(const EfW& w, ef_v4i* d, uint32_t s0, int a) {
#pragma unroll
  for (int t = 0; t < 4; t++)
    d[t] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12(false, a, false, (int)w.st[s0 * EF_WSTR + w.rdo + 16u * t], d[t], false);
}
// Flush: lane (G = lane / 16, r) holds d[t] = (b = 0..3) of component G, point 16t + r: T = sum_b d_b << 8b
// (< 2^48), REDC (T_hi < P), back to point-per-lane through slots s0..s0+3, + base (canonical).
__device__ __forceinline__ static FpExt ef_wfin(const EfW& w, const ef_v4i* d, uint32_t s0, const FpExt& base) {
#pragma unroll
  for (int t = 0; t < 4; t++) {
    const uint32_t d0 = d[t][0], d1 = d[t][1], d2 = d[t][2], d3 = d[t][3];
    const uint32_t x = d0 + (d1 << 8), y = d2 + (d3 << 8);
    const uint32_t lo = x + (y << 16), hi = (y >> 16) + (lo < x ? 1u : 0u);
    const uint32_t m = Fp::mulM(lo), tt = hi - rk_mulhi(m, Fp::P);
    w.st[s0 * EF_WSTR + w.rdo + 16u * t] = rk_umin(tt, tt + Fp::P);
  }
  uint32_t r[4];
#pragma unroll
  for (int j = 0; j < 4; j++) { const uint32_t v = base.elems[j].asRaw() + w.st[(s0 + j) * EF_WSTR + w.lane]; r[j] = rk_umin(v, v - Fp::P); }
  return FpExt(Fp::fromRaw(r[0]), Fp::fromRaw(r[1]), Fp::fromRaw(r[2]), Fp::fromRaw(r[3]));
}
#else
#define EF_WSETUP
// smem_hip.py's LogUp alias words (gfx12 variant) when the WMMA staging is compiled out (EF_NO_WMMA)
__shared__ uint32_t ef_wlds[15u * EF_THREADS];
#endif
'''

PREP_Q = r'''// wmma_hip.py: the WMMA A operands (gfx12 chains): Q[k][n] for n = (j = pd[n / 4], b = n % 4), byte a =
// byte b of (pm_raw[k][j] * 256^a mod P); pd = {0, 2, 1, 3} is the wave64 D row-group order.
__global__ void ef_prep_q(const FpExt* tab) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= EF_NPM * 16u) return;
  const uint32_t k = i / 16u, n = i % 16u, j = (0x3120u >> (4u * (n / 4u))) & 15u, b = n % 4u;
  const uint64_t x = tab[k].elems[j].asRaw();
  uint32_t dw = 0;
  for (uint32_t a = 0; a < 4; a++) dw |= uint32_t((((x << (8u * a)) % Fp::P) >> (8u * b)) & 255u) << (8u * a);
  ef_wq_raw[i] = dw;
}
'''


def convert(s, wmin=6, slots=24, group=None, regions=1, inc=1, lag=1, qpre=1, stats=None):
    if group is None:
        group = 64 // slots * slots  # links per exact i32 accumulation: <= 64, whole segments
    assert slots % 4 == 0 and group % slots == 0 and group <= 64 and slots >= 4
    assert not lag or (inc and lag < slots // 4)
    lines = s.split('\n')
    # ---- parse chains (function-local, straight-line bodies) ----
    chains, open_ch, ftypes = [], {}, {}
    func = None
    for i, l in enumerate(lines):
        if l.startswith(('__device__', '__global__', 'template')) and l.rstrip().endswith('{'):
            # the generated phase functions (straight-line bodies); helpers and kernels are skipped
            func = i if re.match(r'__device__ (EF_NOINLINE )?\w+ (rv32im_v2_\d+|poly_fp|rk_accblk|rk_hoist)\(', l) else None
            ftypes = {}
            for m in re.finditer(r'(FpExt\*?|Fp\*?) (arg\w+)', l):
                ftypes[m.group(2)] = m.group(1)
        if func is None:
            continue
        for m in re.finditer(r'\bFpExt (\w+)\b', l):
            ftypes[m.group(1)] = 'FpExt'
        m = re.match(r'\s*(FpExt|Fp|auto|RkAcc|constexpr Fp) (x\d+)\s*(?:=|\()(.*)', l)
        if m:
            t, v, rhs = m.groups()
            if t == 'auto':
                if re.match(r'\s*(rk_fin|rkw_fin|RK_LD4)', rhs) or 'FpExt' in rhs:
                    t = 'FpExt'
                elif re.match(r'\s*(RK_LD1|EF_LD|rkw1_fin)', rhs):
                    t = 'Fp'
                else:
                    ids = re.findall(r'\b(x\d+|arg\w+)\b', rhs)
                    t = 'FpExt' if any(ftypes.get(x) == 'FpExt' for x in ids) else 'Fp'
            ftypes[v] = t
        m = LINK.match(l) or FIN.match(l)
        if not m:
            if re.search(r'rk_link(_nf)?_k\(', l):
                sys.exit('wmma_hip: unexpected link line %d: %s' % (i + 1, l[:120]))
            continue
        out, kind, base, c, k = m.groups()
        ch = open_ch.pop(base, None)
        if ch is None:
            ch = {'func': func, 'base': base, 'links': []}
            chains.append(ch)
        ct = ftypes.get(c, 'Fp')
        if ct not in ('Fp', 'FpExt'):
            sys.exit('wmma_hip: link %d: unknown term type' % (i + 1))
        ch['links'].append((i, ct, c, int(k), kind))
        if LINK.match(l):
            open_ch[out] = ch
        else:
            ch['fin'] = (i, out)
    if open_ch:
        sys.exit('wmma_hip: %d chains without rk_fin' % len(open_ch))
    # every intermediate RkAcc is only the next link's base
    for ch in chains:
        f0 = ch['func']
        f1 = next(j for j in range(f0, len(lines)) if lines[j] == '}')
        body = '\n'.join(lines[f0:f1])
        for (i, _, _, _, _) in ch['links'][:-1]:
            v = LINK.match(lines[i]).group(1)
            if len(re.findall(r'\b%s\b' % v, body)) != 2:
                sys.exit('wmma_hip: %s (line %d) has other uses' % (v, i + 1))
    # ---- choose chains, staging regions ----
    free = list(range(regions))
    ends = []  # (fin line, region)
    chosen = []
    nfp_all = 0
    for ch in chains:
        first = ch['links'][0][0]
        while ends and ends[0][0] < first:
            free.append(ends.pop(0)[1])
            free.sort()
        nfp = sum(1 for l in ch['links'] if l[1] == 'Fp')
        nfp_all += nfp
        if nfp < wmin or len(ch['links']) < 2:
            continue
        if not free:
            ch['noreg'] = True
            continue
        ch['region'] = free.pop(0)
        ends.append((ch['fin'][0], ch['region']))
        ends.sort()
        chosen.append(ch)
    def qop(blk):
        ks = [x[3] for x in blk]
        if len(ks) == 4 and all(ks[q] == ks[0] + q for q in range(4)):
            return 'ef_wq(efw, %du)' % ks[0]
        k4 = ks + [ks[0]] * (4 - len(ks))
        return 'ef_wq4(efw, %du, %du, %du, %du, %du)' % (tuple(k4) + (len(ks),))

    # ---- rewrite ----
    gfx = {}  # line -> gfx12 statements
    funcs = set()
    for n, ch in enumerate(chains):
        if 'region' not in ch:
            continue
        funcs.add(ch['func'])
        R0 = ch['region'] * slots
        fp = [l for l in ch['links'] if l[1] == 'Fp']
        has_ext = len(fp) != len(ch['links'])
        ngroups = (len(fp) + group - 1) // group
        d = 'efd_%d' % n
        pre = ['ef_v4i %s[4] = {};' % d]
        if has_ext:
            pre.append('RkAcc efs_%d = rk_acc(%s);' % (n, ch['base']))
        if ngroups > 1:
            pre.append('FpExt efb_%d = %s;' % (n, 'FpExt()' if has_ext else ch['base']))
        gfx.setdefault(ch['links'][0][0], []).extend(pre)
        # blocks of 4 staged links (per segment of `slots`, per exact group of `group` links) and the
        # line after whose staging each block's WMMAs go: the segment end (burst), or the end of the
        # block `lag` places later in the group (incremental), so the tile reads overlap later work
        pre_q, emit, flush = {}, {}, {}
        for g in range(ngroups):
            gl = fp[g * group:(g + 1) * group]
            blocks = []
            for s0 in range(0, len(gl), slots):
                seg = gl[s0:s0 + slots]
                for b0 in range(0, len(seg), 4):
                    blocks.append((g * group + s0 + b0, seg[b0:b0 + 4], b0, seg[-1][0]))
            for bi, (f, blk, slot0, seg_last) in enumerate(blocks):
                if inc:
                    at = blocks[min(bi + lag, len(blocks) - 1)][1][-1][0]
                else:
                    at = seg_last
                if qpre:
                    pre_q[blk[0][0]] = 'const int efq_%d_%d = %s;' % (n, f, qop(blk))
                    a = 'efq_%d_%d' % (n, f)
                else:
                    a = qop(blk)
                emit.setdefault(at, []).append('ef_wblk(efw, %s, %du, %s);' % (d, R0 + slot0, a))
            if g < ngroups - 1:
                flush[gl[-1][0]] = 'efb_%d = ef_wfin(efw, %s, %du, efb_%d);' % (n, d, R0, n)
            if g > 0:
                pre_q[gl[0][0]] = '%s[0] = %s[1] = %s[2] = %s[3] = ef_v4i{}; ' % ((d,) * 4) + pre_q.get(gl[0][0], '')
        fpos = {l[0]: idx for idx, l in enumerate(fp)}
        for (i, ct, c, k, kind) in ch['links']:
            st = gfx.setdefault(i, [])
            if ct == 'FpExt':
                st.append('efs_%d = rk_link_k(efs_%d, %s, %du);' % (n, n, c, k))
                continue
            if i in pre_q:
                st.append(pre_q[i].strip())
            st.append('ef_wst(efw, %du, %s);' % (R0 + (fpos[i] % group) % slots, c))
            st.extend(emit.get(i, []))
            if i in flush:
                st.append(flush[i])
        fi, out = ch['fin']
        if ngroups > 1:
            base = ('rk_fin(efs_%d) + efb_%d' % (n, n)) if has_ext else 'efb_%d' % n
        else:
            base = ('rk_fin(efs_%d)' % n) if has_ext else ch['base']
        gfx.setdefault(fi, []).append('FpExt %s = ef_wfin(efw, %s, %du, %s);' % (out, d, R0, base))
    for i, st in gfx.items():
        lines[i] = '  EF_W((%s), %s)' % (' '.join(st), lines[i].strip())
    for f in sorted(funcs, reverse=True):
        lines.insert(f + 1, '  EF_WSETUP')
    s = '\n'.join(lines)

    def sub(old, new, count=1):
        nonlocal s
        c = s.count(old)
        if c != count:
            sys.exit('wmma_hip: expected %d of %r, found %d' % (count, old[:80], c))
        s = s.replace(old, new)

    sub('// ---- end helpers ----\n',
        HELPERS % {'slots_total': slots * regions, 'regions': regions, 'slots': slots} + '// ---- end helpers ----\n')
    sub('__constant__ __align__(16) uint32_t ef_pmn_raw[EF_NPM * 4];\n',
        '__constant__ __align__(16) uint32_t ef_pmn_raw[EF_NPM * 4];\n'
        '// The gfx12 WMMA A operands of every poly_mix constant (wmma_hip.py, ef_prep_q).\n'
        '__device__ uint32_t ef_wq_raw[EF_NPM * 16];\n')
    sub('// Phase-split variant: even blocks', PREP_Q + '\n// Phase-split variant: even blocks')
    sub('''  uint32_t cycle = blockDim.x * rb + threadIdx.x;
  if (cycle >= domain) return;
''', '''  uint32_t cycle = blockDim.x * rb + threadIdx.x;
#ifdef EF_WM
  // The WMMA chains need whole waves: a partial wave (domains below 64 rows only) evaluates
  // wrapped rows and skips their stores.
  const bool ef_live = cycle < domain;
  if (domain == 0 || (!ef_live && (domain & 63u) == 0)) return;
  if (!ef_live) cycle &= domain - 1u;
#else
  const bool ef_live = true;
  if (cycle >= domain) return;
#endif
''')
    sub('  for (uint32_t i = 0; i < 4; i++) dst[i * domain + cycle] = t.elems[i];\n}\n',
        '  if (ef_live)\n    for (uint32_t i = 0; i < 4; i++) dst[i * domain + cycle] = t.elems[i];\n}\n')
    sub('''    EF_COPY_SYM(ef_pmn_raw, b.tabn, EF_NPM * sizeof(FpExt));
''', '''    EF_COPY_SYM(ef_pmn_raw, b.tabn, EF_NPM * sizeof(FpExt));
    ef_prep_q<<<(EF_NPM * 16 + 127) / 128, 128, 0, stream>>>(b.tab);
    CUDA_OK(cudaGetLastError());
''')
    nl = sum(1 for ch in chosen for l in ch['links'] if l[1] == 'Fp')
    msg = ('wmma_hip: %d of %d chains, %d of %d Fp links on the matrix units (wmin %d, %d x %d slots, group %d), '
           '%d chains without a free region' % (len(chosen), len(chains), nl, nfp_all, wmin, regions, slots, group,
                                                sum(1 for ch in chains if ch.get('noreg'))))
    if stats is not None:
        stats.append(msg)
    return s


if __name__ == '__main__':
    args = sys.argv[1:]
    opts = {}
    for name in ('wmin', 'slots', 'group', 'regions', 'inc', 'lag', 'qpre'):
        if '--' + name in args:
            i = args.index('--' + name)
            opts[name] = int(args[i + 1])
            del args[i:i + 2]
    with open(args[0], newline='') as f:
        text = f.read()
    st = []
    out = convert(text, stats=st, **opts)
    print(st[0])
    with open(args[1], 'w', newline='') as f:
        f.write(out)
