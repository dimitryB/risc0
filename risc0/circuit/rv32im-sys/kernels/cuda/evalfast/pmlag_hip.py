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

# Lagged poly_mix constant loads in the ROCm eval_check on gfx11 and gfx12 (regen.py: both variants,
# after smem_hip.py).
#
# Each scalar chain link rk_link(_nf)_k(base, c, k) reads its FpExt constant ef_pm_raw[k] with a
# uniform (SGPR) load. On gfx1100 the AMDGPU scheduler hoists those loads far ahead in s_load_b512
# batches; with ~100 SGPRs free the batches are spilled to VGPR lanes and read back one v_readlane
# at a time (~14k v_readlane per main-phase wave). Here the link W positions further on in the same
# chain loads its constant through ef_pml(): the table pointer passes through an empty asm (SGPR in
# and out, no instruction) that also takes the accumulator of this link as an operand, so the load
# cannot be issued before that link's accumulator exists: at most W links of constants are in
# flight. A link with an FpExt term reads both of its constant rows (ef_pm_raw, ef_pmn_raw) the same
# way (rk_link_kxl), and so does each chained closure rk_cl_acc(w, u, k) (rk_cl_accl), with the
# accumulator it was copied from (the previous closure) as the dependency. Other targets read the
# constants directly (EF_DEP is 0 there).
#
# The default variant (gfx11) rewrites the scalar form of every EF_W((gfx12 form), scalar form) line
# and the plain links; the gfx12 variant (plain=True) only the plain links (the WMMA form of an
# EF_W line is what gfx12 compiles). Measured on the ef harness at po2 20 (interleaved medians,
# bit-exact): RX 7900 XTX 136.3 -> 123.5 ms (W 4, links only); RX 9070 XT, with wmma_hip.py's opaque
# A-operand base, 134.5 -> 131.5 ms (plain links, FpExt-term links and closures).
#
# define(s) adds only the helpers (both variants carry them, so archsel_hip.py sees no conditional
# directive in a differing run); convert(s) adds the helpers and rewrites the links and closures.
import re

HELP = r'''// pmlag_hip.py: on gfx11 and gfx12 each scalar chain link's constant loaded at most W links ahead,
// each FpExt-term link's and chained closure's constant rows not before the accumulator they depend on.
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX11__) || defined(__GFX12__))
#define EF_DEP(e) (e)
__device__ __forceinline__ static ::risc0::FpExt ef_pml(uint32_t k, uint32_t dep) {
  uint64_t p = reinterpret_cast<uint64_t>(ef_pm_raw);
  asm("" : "+s"(p) : "v"(dep));
  const uint4 v = ((const __attribute__((address_space(4))) uint4*)p)[k];
  return ::risc0::FpExt(::risc0::Fp::fromRaw(v.x), ::risc0::Fp::fromRaw(v.y), ::risc0::Fp::fromRaw(v.z), ::risc0::Fp::fromRaw(v.w));
}
__device__ __forceinline__ static uint4 ef_pmu(const uint32_t* tab, uint32_t k, uint32_t dep) {
  uint64_t p = reinterpret_cast<uint64_t>(tab);
  asm("" : "+s"(p) : "v"(dep));
  return ((const __attribute__((address_space(4))) uint4*)p)[k];
}
template <typename A> __device__ __forceinline__ static RkAcc rk_link_kl(const A& base, Fp c, uint32_t k, uint32_t dep) { return rk_link(base, c, ef_pml(k, dep)); }
template <typename A> __device__ __forceinline__ static RkAcc rk_link_nf_kl(const A& base, Fp c, uint32_t k, uint32_t dep) { return rk_link_nf(base, c, ef_pml(k, dep)); }
#else
#define EF_DEP(e) 0u
__device__ __forceinline__ static uint4 ef_pmu(const uint32_t* tab, uint32_t k, uint32_t) { return reinterpret_cast<const uint4*>(tab)[k]; }
template <typename A> __device__ __forceinline__ static RkAcc rk_link_kl(const A& base, Fp c, uint32_t k, uint32_t) { return rk_link_k(base, c, k); }
template <typename A> __device__ __forceinline__ static RkAcc rk_link_nf_kl(const A& base, Fp c, uint32_t k, uint32_t) { return rk_link_nf_k(base, c, k); }
#endif
// rk_link_k(base, FpExt c, k) and rk_cl_acc(w, u, k) with the constant rows through ef_pmu
template <typename A> __device__ __forceinline__ static RkAcc rk_link_kxl(const A& base, FpExt c, uint32_t k, uint32_t dep) {
  RkAcc s = rk_acc(base);
  const uint32_t a0 = c.elems[0].asRaw(), a1 = c.elems[1].asRaw(), a2 = c.elems[2].asRaw(), a3 = c.elems[3].asRaw();
  const uint4 b = ef_pmu(ef_pm_raw, k, dep), n = ef_pmu(ef_pmn_raw, k, dep);
  EF_PMROW(b, n);
  for (int j = 0; j < 4; j++) {
    uint64_t t = ef_fold64((uint64_t(s.hi[j]) << 32) | s.lo[j]);
    t = ef_fold64(ef_madw(a1, X[j][1], ef_madw(a0, X[j][0], t)));
    t = ef_fold64(ef_madw(a3, X[j][3], ef_madw(a2, X[j][2], t)));
    s.lo[j] = uint32_t(t); s.hi[j] = uint32_t(t >> 32);
  }
  return s;
}
__device__ __forceinline__ static void rk_cl_accl(RkW& w, const FpExt& u, uint32_t k, uint32_t dep) {
  const uint32_t a0 = u.elems[0].asRaw(), a1 = u.elems[1].asRaw(), a2 = u.elems[2].asRaw(), a3 = u.elems[3].asRaw();
  const uint4 b = ef_pmu(ef_pm_raw, k, dep), n = ef_pmu(ef_pmn_raw, k, dep);
  EF_PMROW(b, n);
  for (int j = 0; j < 4; j++) {
    uint64_t t = (uint64_t(w.hi[j]) << 32) | w.lo[j];
    t = ef_fold64(ef_madw(a1, X[j][1], ef_madw(a0, X[j][0], t)));
    t = ef_fold64(ef_madw(a3, X[j][3], ef_madw(a2, X[j][2], t)));
    w.lo[j] = uint32_t(t); w.hi[j] = uint32_t(t >> 32);
  }
}
template <typename A> __device__ __forceinline__ static RkAcc rk_link_kl(const A& base, FpExt c, uint32_t k, uint32_t dep) { return rk_link_kxl(base, c, k, dep); }
template <typename A> __device__ __forceinline__ static RkAcc rk_link_nf_kl(const A& base, FpExt c, uint32_t k, uint32_t dep) { return rk_link_kxl(base, c, k, dep); }
'''
ANCHOR = '// ---- wmma_hip.py: poly_mix chains on the gfx12 matrix units'
FN = re.compile(r'^__device__ (?:EF_NOINLINE |EF_CHAIN )?FpExt (rv32im_v2_\d+|poly_fp|rk_accblk)\(.*\) \{$')
LINK = re.compile(r'(RkAcc|FpExt) (x\d+) = (rk_fin\()?(rk_link(?:_nf)?_k)\((\w+), (\w+), (\d+u)\)(\)?);')
# a chained closure: RkW _cxN = _cxM; ... rk_cl_acc(_cxN, u, k);
CLDEF = re.compile(r'RkW (_cx\d+) = (_cx\d+);')
CL = re.compile(r'rk_cl_acc\((_cx\d+), (.*?), (\d+u)\);')


def define(s):
    assert s.count(ANCHOR) == 1
    return s.replace(ANCHOR, HELP + ANCHOR)


def convert(s, w=4, plain=False, stats=None):
    lines = define(s).split('\n')
    n = nf = ncl = 0
    i = 0
    while i < len(lines):
        if not FN.match(lines[i]):
            i += 1
            continue
        end = lines.index('}', i + 1)
        nf += 1
        res = []
        prev = {}
        for k in range(i + 1, end):
            line = lines[k]
            for m in CLDEF.finditer(line):
                prev[m.group(1)] = m.group(2)

            def cl(m):
                nonlocal ncl
                src = prev.get(m.group(1))
                if src is None:
                    return m.group(0)
                ncl += 1
                return 'rk_cl_accl(%s, %s, %s, EF_DEP(%s.lo[0]));' % (m.group(1), m.group(2), m.group(3), src)
            if 'rk_cl_acc(' in line:
                line = CL.sub(cl, line)
                lines[k] = line
            if line.startswith('  EF_W(('):
                if plain:
                    continue  # gfx12 compiles the WMMA form: only the plain links are lagged there
                # only the scalar form of an EF_W((gfx12 form), scalar form) line
                depth, cut = 0, None
                for q in range(line.index('EF_W(') + 5, len(line)):
                    if line[q] == '(':
                        depth += 1
                    elif line[q] == ')':
                        depth -= 1
                        if depth == 0:
                            cut = q + 1
                            break
                head, tail = line[:cut], line[cut:]
            else:
                head, tail = '', line
            m = LINK.search(tail)
            if not m:
                continue
            ty, name, fin, fn, base, c, kk, close = m.groups()
            if len(res) >= w:
                new = '%s %s = %s%sl(%s, %s, %s, EF_DEP(%s))%s;' % (ty, name, fin or '', fn, base, c, kk,
                                                                   res[-w], close)
                tail = tail[:m.start()] + new + tail[m.end():]
                n += 1
            res.append('%s.lo[0]' % name if ty == 'RkAcc' else '%s.elems[0].asRaw()' % name)
            lines[k] = head + tail
        i = end + 1
    msg = ('pmlag_hip: %d functions, %d %slinks and %d closures with lagged constants (W %d)'
           % (nf, n, 'plain ' if plain else '', ncl, w))
    if stats is not None:
        stats.append(msg)
    else:
        print(msg)
    return '\n'.join(lines)
