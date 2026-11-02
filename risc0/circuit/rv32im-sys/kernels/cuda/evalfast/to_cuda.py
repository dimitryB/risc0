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

# Convert the Intel evalfast pipeline output (pure C++ device code + SYCL
# wrapper) into the CUDA eval_check translation unit, eval_check_ef.cu, which
# exports risc0_circuit_rv32im_cuda_eval_check{,_dep} and the warmup entry.
# Run through regen.py, not directly.
#
# usage: to_cuda.py <fast.cpp> <out.cu>
import re
import sys

src, out = sys.argv[1], sys.argv[2]
# Shapes of the current circuit; a change here means the pipeline output
# changed, so re-verify the kernel (bit-exactness harness) before updating.
EXPECT_LOADS = 3474
POLY_MIX_COUNT = 458  # NUM_POLY_MIX_POWERS (risc0-circuit-rv32im info.rs)
full = open(src).read()
lines = full.split('\n')
end = next(i for i, l in enumerate(lines) if l.startswith('// Intel SYCL eval_check kernel'))
body = lines[:end]
res = []
for l in body:
    if l.startswith('#include "fp.h"'):
        res.append('#include "ef_fp.h"')
        continue
    if l.startswith('#include "fpext.h"'):
        res.append('#include "ef_fpext.h"')
        continue
    if l.startswith('// Auto-generated amalgamation for SYCL'):
        res.append('// evalfast (Intel pipeline output) converted to CUDA by to_cuda.py')
        continue
    if l.startswith('#define RK_LD1(') :
        res.append('#ifdef EF_SLAB_ASM')
        res.append('__device__ __forceinline__ uint32_t ef_ldl(const void* p, uint32_t w) { uint32_t v; uint32_t a = (uint32_t)__cvta_generic_to_local(p) + 4u * w; asm volatile("ld.local.u32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }')
        res.append('__device__ __forceinline__ void ef_stl(void* p, uint32_t w, uint32_t v) { uint32_t a = (uint32_t)__cvta_generic_to_local(p) + 4u * w; asm volatile("st.local.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }')
        res.append('#define RK_LD1(p, w) (static_cast<const ::risc0::Fp>(::risc0::Fp::fromRaw(ef_ldl((p), (w)))))')
        res.append('#define RK_ST1(p, w, v) ef_stl((p), (w), ::risc0::Fp(v).asRaw())')
        res.append('#else')
        res.append(l)
        continue
    if l.startswith('#define RK_ST1('):
        res.append(l)
        res.append('#endif')
        continue
    if l.startswith('#define RK_STRIDE'):
        res.append('#ifndef RK_STRIDE\n#define RK_STRIDE 1u\n#endif')
        continue
    l = re.sub(r'^__attribute__\(\(always_inline\)\) static inline', '__device__ __forceinline__ static', l)
    l = re.sub(r'^template <typename A> __attribute__\(\(always_inline\)\) static inline',
               'template <typename A> __device__ __forceinline__ static', l)
    l = re.sub(r'^__attribute__\(\(noinline\)\) ', '__device__ EF_NOINLINE ', l)
    l = re.sub(r'^(FpExt (rv32im_v2_\d+|poly_fp)\()', r'__device__ \1', l)
    l = re.sub(r'^constexpr uint16_t (rk_q[ab])\[\]', r'__device__ const uint16_t \1[]', l)
    # poly_mix accesses with literal indices -> EF_PM(i)
    l = re.sub(r'\bpoly_mix\[(\d+)\]', r'EF_PM(\1)', l)
    res.append(l)
text = '\n'.join(res)
hs = text.index('void rk_hoist(')
he = text.index('\n}\n', hs)
text = text[:hs] + re.sub(r'EF_PM\((\d+)\)', r'poly_mix[\1]', text[hs:he]) + text[he:]
# 2) global (read-only) loads: X[(uint32_t)(...)] -> EF_LD(X, (uint32_t)(...))
def conv_loads(t):
    out = []; i = 0
    pat = re.compile(r'(/\*[a-z]+=\*/)?\b(args\[[0-9]\]|arg[0-9]+)\[\(uint32_t\)\(')
    n = 0
    while True:
        m = pat.search(t, i)
        if not m:
            out.append(t[i:]); break
        out.append(t[i:m.start()])
        j = m.end(); depth = 1
        while depth:
            c = t[j]
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            j += 1
        assert t[j] == ']', t[m.start():j+5]
        out.append('EF_LD(%s, (uint32_t)(%s' % (m.group(2), t[m.end():j]))
        out.append(')')
        i = j + 1; n += 1
    assert n == EXPECT_LOADS, 'to_cuda: wrapped %d global loads, expected %d' % (n, EXPECT_LOADS)
    print('to_cuda: %d global loads wrapped' % n)
    return ''.join(out)
text = conv_loads(text)

# ---- EF_WIDE: CUDA 64-bit multiply-accumulate forms of the lazy helpers ----
# Each is bit-identical to the 32-bit lo/hi form (the hi word wraps mod 2^32 in
# both), but lets ptxas emit IMAD.WIDE.U32 with a 64-bit addend instead of
# IMAD + IMAD.HI + compare/select carry chains.
WIDE = {
 'static void rkw_mac(RkW& w, int j, uint32_t x, uint32_t y) {':
 """  uint64_t t = ((uint64_t(w.hi[j]) << 32) | w.lo[j]) + uint64_t(x) * uint64_t(y);
  w.lo[j] = uint32_t(t); w.hi[j] = uint32_t(t >> 32);""",
 'static void rkw1_ff(RkW1& w, Fp a, Fp b, bool neg) {':
 """  const uint32_t x = neg ? Fp::P - a.asRaw() : a.asRaw(), y = b.asRaw();
  uint64_t t = ((uint64_t(w.hi) << 32) | w.lo) + uint64_t(x) * uint64_t(y);
  w.hi = rkw1_fold(uint32_t(t >> 32)); w.lo = uint32_t(t);""",
 'static RkAcc rk_link(const A& base, Fp c, const FpExt& pm) {':
 """  RkAcc s = rk_acc(base); uint32_t cr = c.asRaw();
  for (int j = 0; j < 4; j++) {
    uint64_t t = ((uint64_t(s.hi[j]) << 32) | s.lo[j]) + uint64_t(cr) * uint64_t(pm.elems[j].asRaw());
    uint32_t h = uint32_t(t >> 32);
    s.lo[j] = uint32_t(t); s.hi[j] = rk_umin(h, h - Fp::P);
  }
  return s;""",
 'static RkAcc rk_link_nf(const A& base, Fp c, const FpExt& pm) {':
 """  RkAcc s = rk_acc(base); uint32_t cr = c.asRaw();
  for (int j = 0; j < 4; j++) {
    uint64_t t = ((uint64_t(s.hi[j]) << 32) | s.lo[j]) + uint64_t(cr) * uint64_t(pm.elems[j].asRaw());
    s.lo[j] = uint32_t(t); s.hi[j] = uint32_t(t >> 32);
  }
  return s;""",
}
for head, nb in WIDE.items():
    i = text.index(head)
    bs = text.index('\n', i) + 1
    be = text.index('\n}\n', bs)
    old = text[bs:be]
    text = text[:bs] + '#ifdef EF_WIDE\n' + nb + '\n#else\n' + old + '\n#endif' + text[be:]
left = [x for x in re.findall(r'\bpoly_mix\[[^\]]*\]', text) if not re.match(r'poly_mix\[\d+\]', x)]
# rk_hoist(poly_mix, ...) reads poly_mix[...] literally too (converted above).
assert not left, 'non-literal poly_mix index: %s' % left[:5]
assert ('constexpr uint32_t POLY_MIX_COUNT = %d;' % POLY_MIX_COUNT) in full, 'POLY_MIX_COUNT changed'
RK_QN = int(re.search(r'constexpr uint32_t RK_QN = (\d+);', text).group(1))
RK_HN = int(re.search(r'constexpr uint32_t RK_HN = (\d+);', text).group(1))
NPM = POLY_MIX_COUNT + RK_QN + RK_HN
max_pm = max(int(x) for x in re.findall(r'\bEF_PM\((\d+)\)', text))
assert max_pm < NPM, 'poly_mix index %d out of the %d-entry table' % (max_pm, NPM)

prelude = r'''
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#ifndef EF_NOINLINE
#define EF_NOINLINE __noinline__
#endif
#define EF_NPC @POLY_MIX_COUNT@u
#define EF_NPM (EF_NPC + @RK_QN@u + @RK_HN@u)
#define EF_LD(p, i) (::risc0::Fp::fromRaw(__ldg(reinterpret_cast<const uint32_t*>(p) + (i))))
// The full poly_mix table: EF_NPC powers, RK_QN products, RK_HN hoisted terms.
__constant__ uint32_t ef_pm_raw[EF_NPM * 4];
#define EF_PM(i) (::risc0::FpExt(::risc0::Fp::fromRaw(ef_pm_raw[4 * (i) + 0]), ::risc0::Fp::fromRaw(ef_pm_raw[4 * (i) + 1]), ::risc0::Fp::fromRaw(ef_pm_raw[4 * (i) + 2]), ::risc0::Fp::fromRaw(ef_pm_raw[4 * (i) + 3])))
'''

kernel = r'''
using namespace risc0;
namespace rk = risc0::circuit::rv32im_v2;

static_assert(EF_NPM == EF_NPC + rk::RK_QN + rk::RK_HN, "poly_mix table size");

__constant__ uint32_t ef_qtab[4];

// One thread: poly_mix table (powers, products, hoisted out/mix terms) and the
// 4-entry vanishing quotient table 1/((3*rou^c)^(2^po2) - 1), c = cycle & 3.
__global__ void ef_prep(const FpExt* pm_in, const Fp* out, const Fp* mix, FpExt* tab, uint32_t* qtab,
                        Fp rou, uint32_t po2) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (uint32_t i = 0; i < EF_NPC; i++) tab[i] = pm_in[i];
  for (uint32_t i = 0; i < rk::RK_QN; i++) tab[EF_NPC + i] = pm_in[rk::rk_qa[i]] * pm_in[rk::rk_qb[i]];
  rk::rk_hoist(tab, out, mix, tab + EF_NPC + rk::RK_QN);
  for (uint32_t c = 0; c < 4; c++) {
    Fp x = Fp(3) * pow(rou, c);
    Fp y = pow(x, size_t(1) << po2);
    qtab[c] = inv(y - Fp(1)).asRaw();
  }
}

__launch_bounds__(EF_THREADS, EF_MINB)
__global__ void ef_eval_check(Fp* check, Fp* data, Fp* accum, Fp* mix, Fp* out, uint32_t domain) {
  FpExt* pm = nullptr;  // poly_mix is read through EF_PM (constant memory)
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= domain) return;
  uint32_t lslab[RK_NWORDS];
  Fp* slab = reinterpret_cast<Fp*>(lslab);
  Fp* args[5] = {accum, data, out, mix, slab};
  FpExt tot = rk::poly_fp((size_t)cycle, (size_t)domain, pm, args);
  Fp quot = Fp::fromRaw(ef_qtab[cycle & 3]);
  for (uint32_t i = 0; i < 4; i++) check[i * domain + cycle] = tot.elems[i] * quot;
}

#include "cuda.h"
static cudaStream_t getEvalCheckStream() {
  int dev = getCachedDevice();
  static cudaStream_t streams[16] = {};
  if (!streams[dev]) CUDA_OK(cudaStreamCreateWithFlags(&streams[dev], cudaStreamNonBlocking));
  return streams[dev];
}
static cudaEvent_t getEvalCheckEvent() {
  int dev = getCachedDevice();
  static cudaEvent_t events[16] = {};
  if (!events[dev]) CUDA_OK(cudaEventCreateWithFlags(&events[dev], cudaEventDisableTiming));
  return events[dev];
}
struct EfBufs { FpExt* pm = nullptr; FpExt* tab = nullptr; uint32_t* q = nullptr; };
static EfBufs& efBufs() {
  int dev = getCachedDevice();
  static EfBufs bufs[16];
  if (!bufs[dev].pm) {
    CUDA_OK(cudaMalloc(&bufs[dev].pm, EF_NPC * sizeof(FpExt)));
    CUDA_OK(cudaMalloc(&bufs[dev].tab, EF_NPM * sizeof(FpExt)));
    CUDA_OK(cudaMalloc(&bufs[dev].q, 4 * sizeof(uint32_t)));
    CUDA_OK(cudaFuncSetCacheConfig((const void*)ef_eval_check, cudaFuncCachePreferL1));
  }
  return bufs[dev];
}

extern "C" {
const char* risc0_circuit_rv32im_cuda_eval_check(Fp* check, const Fp* ctrl, const Fp* data,
                                                 const Fp* accum, const Fp* mix, const Fp* out,
                                                 const Fp& rou, uint32_t po2, uint32_t domain,
                                                 const FpExt* poly_mix_pows) {
  try {
    cudaStream_t stream = getEvalCheckStream();
    {
      cudaEvent_t event = getEvalCheckEvent();
      CUDA_OK(cudaEventRecord(event, getPersistentStream()));
      CUDA_OK(cudaStreamWaitEvent(stream, event, 0));
    }
    EfBufs& b = efBufs();
    // Pageable source: cudaMemcpyAsync stages it before returning; the static
    // copy keeps it valid regardless (as the original kernel's staging did).
    static FpExt staging[EF_NPC];
    memcpy(staging, poly_mix_pows, sizeof(staging));
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpyAsync(b.pm, staging, sizeof(staging), cudaMemcpyHostToDevice, stream));
    ef_prep<<<1, 1, 0, stream>>>(b.pm, out, mix, b.tab, b.q, rou, po2);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpyToSymbolAsync(ef_qtab, b.q, 4 * sizeof(uint32_t), 0, cudaMemcpyDeviceToDevice, stream));
    CUDA_OK(cudaMemcpyToSymbolAsync(ef_pm_raw, b.tab, EF_NPM * sizeof(FpExt), 0, cudaMemcpyDeviceToDevice, stream));
    const uint32_t block = EF_THREADS;
    const uint32_t grid = (domain + block - 1) / block;
    ef_eval_check<<<grid, block, 0, stream>>>(check, const_cast<Fp*>(data), const_cast<Fp*>(accum),
                                              const_cast<Fp*>(mix), const_cast<Fp*>(out), domain);
    CUDA_OK(cudaGetLastError());
    static int s_force_sync = -1, s_skip_sync = -1;
    if (s_force_sync < 0) {
      s_force_sync = std::getenv("RISC0_EVAL_CHECK_FORCE_SYNC") ? 1 : 0;
      s_skip_sync = std::getenv("RISC0_EVAL_CHECK_SKIP_SYNC") ? 1 : 0;
    }
    bool needs_sync;
    if (s_force_sync) needs_sync = true;
    else if (s_skip_sync) needs_sync = false;
    else {
      static int s_cc_major = -1;
      if (s_cc_major < 0) cudaDeviceGetAttribute(&s_cc_major, cudaDevAttrComputeCapabilityMajor, getCachedDevice());
      needs_sync = s_cc_major >= 12;
    }
    if (needs_sync) CUDA_OK(cudaStreamSynchronize(stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_circuit_rv32im_cuda_eval_check_dep() {
  try {
    cudaEvent_t event = getEvalCheckEvent();
    CUDA_OK(cudaEventRecord(event, getEvalCheckStream()));
    CUDA_OK(cudaStreamWaitEvent(getPersistentStream(), event, 0));
    CUDA_OK(cudaStreamSynchronize(getEvalCheckStream()));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_circuit_rv32im_cuda_warmup_eval_check() {
  try {
    cudaStream_t stream = getEvalCheckStream();
    ef_eval_check<<<1, 1, 0, stream>>>(nullptr, nullptr, nullptr, nullptr, nullptr, 0);
    CUDA_OK(cudaStreamSynchronize(stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}
} // extern "C"
'''

# Internal linkage for the device functions: as external symbols, their host stubs
# (poly_fp, rv32im_v2_N, ...) would collide with the CPU circuit's (kernels/cxx)
# and could replace them at link time.
NS_OPEN = 'namespace risc0::circuit::rv32im_v2 {'
NS_CLOSE = '} // namespace risc0::circuit::rv32im_v2'
assert text.count(NS_OPEN) == 1 and text.count(NS_CLOSE) == 1
text = (text.replace(NS_OPEN, NS_OPEN + '\nnamespace {')
        .replace(NS_CLOSE, '} // namespace\n' + NS_CLOSE))

# insert prelude after the includes (before the namespace)
idx = text.index(NS_OPEN)
prelude = (prelude.replace('@POLY_MIX_COUNT@', str(POLY_MIX_COUNT))
           .replace('@RK_QN@', str(RK_QN)).replace('@RK_HN@', str(RK_HN)))
text = text[:idx] + prelude + text[idx:] + '\n' + kernel
text = '''// GENERATED by kernels/cuda/evalfast/regen.py -- do not edit; re-run it instead.
//
// rv32im eval_check for CUDA (NVIDIA only; ROCm builds eval_check_combined.cu):
// the Intel evalfast pipeline output (kernels/intel/evalfast) converted by
// to_cuda.py, plus CUDA-specific 64-bit multiply-accumulate forms (EF_WIDE)
// and launch bounds tuned for sm_120. Bit-exact with eval_check_combined.cu
// for canonical inputs (every Fp < P), which is what the prover passes: NTT
// outputs, witgen data and transcript values. The lazy reductions rely on it;
// non-canonical inputs give different (wrong) results, where the original
// kernel happens to tolerate them.
#define EF_EXT_LAZY 1
#define RISC0_INTEL_EVAL_LOHI 1
#define EF_SLAB_ASM 1
#define EF_FP_FAST 1
#define EF_WIDE 1
#ifndef EF_THREADS
#define EF_THREADS 128
#endif
#ifndef EF_MINB
#define EF_MINB 3
#endif
''' + text
open(out, 'w').write(text)
print('to_cuda: wrote %s (%d lines)' % (out, text.count('\n')))
