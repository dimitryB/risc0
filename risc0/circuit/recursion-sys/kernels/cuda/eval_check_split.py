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

# usage: eval_check_split.py [--check]
#
# Writes eval_check_split.cu (next to this script) from the generated
# eval_check.cu: poly_fp is split into kNumParts kernels (ec_part<P>) along its
# gated-sum tree. Each part re-applies the gate/mix factors of the tree nodes it
# touches and adds the previous parts' sum left in `check`; the last part scales
# by the quotient. Only the regrouping of field sums changes, so the result is
# bit-identical to eval_check.cu. Subtrees costing more than TARGET_OPS are
# split into their terms; leaves are packed into parts of at most GROUP_OPS ops
# (the measured optimum on sm_120: 6 parts).
#
# The split source then goes through ec_lazyredc.py (lazy Montgomery reduction
# of the poly_mix chains, plus the module warmup entry point), whose pinned
# counts must match.
#
# build.rs compiles the result standalone for CUDA and ROCm (not relocatable,
# which is ~25% slower on CUDA), so it keeps its own __constant__ poly_mix,
# filled by eval_check_split() itself. Re-run this after regenerating
# eval_check.cu; --check (run in CI) regenerates it from eval_check.cu through
# both passes and fails if eval_check_split.cu is out of date.
import os, re, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ec_lazyredc
import ec_lazyfp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'eval_check_split.cu')
CHECK = '--check' in sys.argv[1:]
BASE_OPTION = '--metal-base-output'
BASE_OUTPUT = None
if BASE_OPTION in sys.argv[1:]:
    pos = sys.argv.index(BASE_OPTION)
    if pos + 1 >= len(sys.argv) or sys.argv[pos + 1].startswith('--'):
        sys.exit('eval_check_split: --metal-base-output requires a path')
    BASE_OUTPUT = sys.argv[pos + 1]
TARGET_OPS, GROUP_OPS = 2200, 3000
EXPECT = 'total ops 12075 leaves 65 groups 6 max leaf 2049'
target, gtarget = TARGET_OPS, GROUP_OPS
src = open(os.path.join(HERE, 'eval_check.cu'), newline='').read().split('\n')
defs = {}; order = []
rx = re.compile(r'^\s*(Fp|FpExt|auto) (x\d+) = (.*);\s*$')
rc = re.compile(r'^\s*Fp (x\d+)\((\d+)\);\s*$')
ret = None
for l in src:
    m = rx.match(l)
    if m: defs[m.group(2)] = (m.group(1), m.group(3)); order.append(m.group(2)); continue
    m = rc.match(l)
    if m: defs[m.group(1)] = ('Fp', None, m.group(2)); order.append(m.group(1)); continue
    m = re.match(r'^\s*return (x\d+);', l)
    if m: ret = m.group(1)
tok = re.compile(r'\bx\d+\b')
def deps(v):
    d = defs[v]; return tok.findall(d[1]) if d[1] is not None else []
def slice_of(vs):
    seen = set(); st = list(vs)
    while st:
        v = st.pop()
        if v in seen: continue
        seen.add(v); st.extend(deps(v))
    return seen
def cost(vs): return sum(1 for v in slice_of(vs) if defs[v][1] is not None)
ra = re.compile(r'^(x\d+) \+ (x\d+) \* (x\d+) \* (poly_mix\[\d+\])$')
rb = re.compile(r'^(x\d+) \+ (poly_mix\[\d+\]) \* (x\d+)$')
# node: dict(f=[factors], kids=[nodes]) or leaf dict(f=[factors], b=var)
def terms_of(v):
    d = defs[v]
    if d[1] == 'FpExt(0)': return []
    m = ra.match(d[1])
    if m and defs[m.group(1)][0] == 'FpExt' and defs[m.group(3)][0] == 'FpExt':
        return terms_of(m.group(1)) + [([m.group(2), m.group(4)], m.group(3))]
    m = rb.match(d[1])
    if m and defs[m.group(1)][0] == 'FpExt':
        return terms_of(m.group(1)) + [([m.group(2)], m.group(3))]
    return [([], v)]
def leafvars(n):
    return [n['b']] + [x for x in n['f'] if x.startswith('x')]
def build(f, b, depth):
    n = {'f': f, 'b': b}
    if defs[b][0] == 'FpExt':
        sub = terms_of(b)
        if len(sub) > 1 and cost([b]) > target:
            return {'f': f, 'kids': [build(sf, sb, depth + 1) for sf, sb in sub]}
    return n
root = {'f': [], 'kids': [build(sf, sb, 0) for sf, sb in terms_of(ret)]}
# leaves with paths
leaves = []
def walk(n, path):
    if 'kids' in n:
        for i, k in enumerate(n['kids']): walk(k, path + [(id(n), i)])
    else:
        leaves.append((path, n))
walk(root, [])
lc = [cost(leafvars(n)) for p, n in leaves]
groups = []; cur = []; curv = []
for (p, n), c in zip(leaves, lc):
    if cur and cost(curv + leafvars(n)) > gtarget:
        groups.append(cur); cur = []; curv = []
    cur.append((p, n)); curv += leafvars(n)
if cur: groups.append(cur)
STATS = 'total ops %d leaves %d groups %d max leaf %d' % (cost([ret]), len(leaves), len(groups), max(lc))
def emit_node(n, members, lines, ctr):
    """value expression of node restricted to member leaves (by id)"""
    if 'kids' not in n:
        if id(n) not in members: return None
        b = n['b']; base = b if defs[b][0] == 'FpExt' else None
        if base is None:   # Fp base: pm * x
            return ' * '.join(n['f'] + [b])
        return ' * '.join([b] + n['f']) if n['f'] else b
    parts = [emit_node(k, members, lines, ctr) for k in n['kids']]
    parts = [p for p in parts if p is not None]
    if not parts: return None
    ctr[0] += 1; t = 's%d' % ctr[0]
    lines.append('  FpExt %s = %s;' % (t, parts[0] if parts[0].startswith('s') or ' ' not in parts[0] and defs.get(parts[0], ('',))[0] == 'FpExt' else parts[0]))
    for p in parts[1:]:
        lines.append('  %s = %s + %s;' % (t, t, p))
    if n['f']:
        return ' * '.join([t] + n['f'])
    return t
if STATS != EXPECT:
    sys.exit('eval_check_split: the split changed shape (%s, expected %s): re-verify the kernel '
             '(bit-exactness and timing) before updating EXPECT' % (STATS, EXPECT))
out = []
for gi, g in enumerate(groups):
    members = {id(n) for p, n in g}
    vs = set()
    for p, n in g: vs |= set(leafvars(n))
    # factors of ancestor nodes are needed too
    def anc(n):
        r = set()
        if 'kids' in n:
            for k in n['kids']:
                a = anc(k)
                if a is not None:
                    r |= a
                    r |= {x for x in n['f'] if x.startswith('x')}
            return r if r or any(('kids' not in k and id(k) in members) for k in n['kids']) else None
        return set() if id(n) in members else None
    a = anc(root) or set()
    vs |= a
    sl = slice_of(vs)
    out.append('__device__ __forceinline__ FpExt poly_fp_part%d(uint32_t idx, uint32_t size, const Fp* ctrl, const Fp* out, const Fp* data, const Fp* mix, const Fp* accum) {' % gi)
    out.append('  uint32_t mask = size - 1;')
    for v in order:
        if v not in sl: continue
        d = defs[v]
        out.append('  Fp %s(%s);' % (v, d[2]) if d[1] is None else '  %s %s = %s;' % (d[0], v, d[1]))
    lines = []; ctr = [0]
    e = emit_node(root, members, lines, ctr)
    out += lines
    out.append('  return %s;\n}' % e)
out.append('constexpr int kNumParts = %d;' % len(groups))
out.append('__device__ __forceinline__ FpExt poly_fp_part(int p, uint32_t idx, uint32_t size, const Fp* ctrl, const Fp* out, const Fp* data, const Fp* mix, const Fp* accum) {')
out.append('  switch (p) {')
for gi in range(len(groups)):
    out.append('  case %d: return poly_fp_part%d(idx, size, ctrl, out, data, mix, accum);' % (gi, gi))
out.append('  }\n  return FpExt(0);\n}')
new = ('''// GENERATED by kernels/cuda/eval_check_split.py from eval_check.cu -- do not edit; re-run it instead.
// Recursion eval_check split into kernels by the gated-sum tree of poly_fp (sum regrouping only,
// so the result is bit-identical).
#include "cuda.h"
#include "supra/fp.h"
#include <cstdint>

constexpr size_t INV_RATE = 4;
// Compiled standalone (not relocatable), so this module needs its own copy of the mix powers:
// an `extern __constant__` would not alias ffi_supra.cu's poly_mix without -rdc.
__constant__ FpExt poly_mix[158];

#ifdef __HIPCC__
// RDNA: 128-thread blocks (per po2 18 call: 21.1 vs 23.2 ms on gfx1100, 29.5 vs 32.1 ms on gfx1201),
// and at most 216 VGPRs: 7 waves per SIMD instead of 5 or 6, for a few spills, with build.rs's
// iterative-maxocc scheduler (per call, all six parts: 20.9 -> 16.5 ms on gfx1100, 28.8 -> 18.5 ms
// on gfx1201; 192 and 232 VGPRs were slower on both). Per part (medians of 6-8 interleaved harness
// runs at po2 18, us): gfx1201 parts 1 and 4 with 192 VGPRs (8 waves per SIMD): 3214 -> 3140 and
// 2676 -> 2625, the full call 16.51 -> 16.37 ms; the other parts and every gfx1100 part were no
// faster with 192, 240 or no bound.
constexpr uint32_t kBlock = 128;
#define EC_PART_VGPR(n) __launch_bounds__(kBlock, 1) __attribute__((amdgpu_num_vgpr(n)))
#if defined(__gfx1201__)
#define EC_PART_BOUNDS_1 EC_PART_VGPR(192)
#define EC_PART_BOUNDS_4 EC_PART_VGPR(192)
#else
#define EC_PART_BOUNDS_1 EC_PART_VGPR(216)
#define EC_PART_BOUNDS_4 EC_PART_VGPR(216)
#endif
#define EC_PART_BOUNDS_0 EC_PART_VGPR(216)
#define EC_PART_BOUNDS_2 EC_PART_VGPR(216)
#define EC_PART_BOUNDS_3 EC_PART_VGPR(216)
#define EC_PART_BOUNDS_5 EC_PART_VGPR(216)
#else
constexpr uint32_t kBlock = 256;
#define EC_PART_BOUNDS __launch_bounds__(kBlock, 1)
#endif

namespace risc0::circuit::recursion::cuda {
''' + '\n' + '\n'.join(out) + '\n' + '''
template <int P>
__device__ __forceinline__ void ec_part_body(Fp* check, const Fp* ctrl, const Fp* data, const Fp* accum,
                                             const Fp* mix, const Fp* out, const Fp rou, uint32_t po2,
                                             uint32_t domain) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= domain) return;
  FpExt tot = poly_fp_part(P, cycle, domain, ctrl, out, data, mix, accum);
  if (P > 0) {
    // The previous parts' sum, left in `check` by the kernel before on the same stream.
    tot = FpExt(check[cycle], check[domain + cycle], check[2 * domain + cycle], check[3 * domain + cycle]) + tot;
  }
  if (P == kNumParts - 1) {
    // x ^ e is what supra/fp.h's pow() computes (which is ambiguous with HIP's math overloads).
    Fp x = rou ^ cycle;
    Fp y = (Fp(3) * x) ^ (1u << po2);
    tot = tot * inv(y - Fp(1));
  }
  check[domain * 0 + cycle] = tot[0];
  check[domain * 1 + cycle] = tot[1];
  check[domain * 2 + cycle] = tot[2];
  check[domain * 3 + cycle] = tot[3];
}

#ifdef __HIPCC__
// One kernel per part, each with its own register bound (explicit specializations: an attribute's
// argument cannot depend on P).
template <int P>
__global__ void ec_part(Fp* check, const Fp* ctrl, const Fp* data, const Fp* accum, const Fp* mix,
                        const Fp* out, const Fp rou, uint32_t po2, uint32_t domain);
#define EC_PART_KERNEL(P) template <> EC_PART_BOUNDS_##P __global__ void ec_part<P>(Fp* check, const Fp* ctrl, const Fp* data, const Fp* accum, const Fp* mix, const Fp* out, const Fp rou, uint32_t po2, uint32_t domain) { ec_part_body<P>(check, ctrl, data, accum, mix, out, rou, po2, domain); }
EC_PART_KERNEL(0)
EC_PART_KERNEL(1)
EC_PART_KERNEL(2)
EC_PART_KERNEL(3)
EC_PART_KERNEL(4)
EC_PART_KERNEL(5)
static_assert(kNumParts == 6, "one EC_PART_KERNEL per part");
#else
template <int P>
EC_PART_BOUNDS __global__ void ec_part(Fp* check, const Fp* ctrl, const Fp* data, const Fp* accum,
                                       const Fp* mix, const Fp* out, const Fp rou, uint32_t po2,
                                       uint32_t domain) {
  ec_part_body<P>(check, ctrl, data, accum, mix, out, rou, po2, domain);
}
#endif

template <int P>
static void launch_parts(cudaStream_t stream, uint32_t grid, uint32_t block, Fp* check, const Fp* ctrl,
                         const Fp* data, const Fp* accum, const Fp* mix, const Fp* out, Fp rou, uint32_t po2,
                         uint32_t domain) {
  if constexpr (P < kNumParts) {
#ifdef __HIPCC__
    {
      // Each part (~1.4-5.6 ms at domain 2^20) on a side hardware queue (hip_hop.h): the next
      // part, queued on `stream` after this scope, still follows it.
      risc0_hip_hop::Scope s(stream, risc0_hip_hop::want(risc0_hip_hop::kRecCheck, domain, 18));
      ec_part<P><<<grid, block, 0, s>>>(check, ctrl, data, accum, mix, out, rou, po2, domain);
    }
#else
    ec_part<P><<<grid, block, 0, stream>>>(check, ctrl, data, accum, mix, out, rou, po2, domain);
#endif
    launch_parts<P + 1>(stream, grid, block, check, ctrl, data, accum, mix, out, rou, po2, domain);
  }
}

void eval_check_split(cudaStream_t stream, Fp* check, const Fp* ctrl, const Fp* data, const Fp* accum,
                      const Fp* mix, const Fp* out, Fp rou, uint32_t po2, uint32_t domain,
                      const void* poly_mix_host) {
  // poly_mix_host is pageable: the copy is staged before this call returns.
  CUDA_OK(cudaMemcpyToSymbolAsync(poly_mix, poly_mix_host, sizeof(poly_mix), 0, cudaMemcpyHostToDevice, stream));
  uint32_t block = kBlock, grid = (domain + block - 1) / block;
  launch_parts<0>(stream, grid, block, check, ctrl, data, accum, mix, out, rou, po2, domain);
}

} // namespace risc0::circuit::recursion::cuda
''')

if BASE_OUTPUT:
    open(BASE_OUTPUT, 'w', newline='').write(new)
new, counts = ec_lazyredc.rewrite(new)
if counts != ec_lazyredc.EXPECT:
    sys.exit('eval_check_split: ec_lazyredc counts %s != EXPECT %s (input changed: re-validate with '
             'the lazy-REDC harness, then update ec_lazyredc.EXPECT)' % (counts, ec_lazyredc.EXPECT))
STATS += '; lazy REDC: chain terms %d, folds %d, from %d, materialized %d' % counts

# HIP (RDNA) only: a cheaper Fp product for some parts. 32-bit multiplies (v_mad_u64_u32,
# v_mad_i64_i32, v_mul_lo_u32) issue at a quarter of the add rate; sppark's bb31_t product costs
# three of them (a*b, m = lo * M0, + m*p), this one two, with a signed REDC: m = (int32)(t * p^-1
# mod 2^32), p^-1 = 2^31 + 2^27 + 1, from shift-adds (an empty asm keeps LLVM from folding them
# back into a v_mul_lo_u32), and t - m*p from one v_mad_i64_i32, whose low word is 0, so the high
# word is the exact quotient r. For a, b in [0, p), |t| < p^2 and |r| < (p^2 + 2^31 p) / 2^32 < p,
# and min_u32(r, r + p) maps r to [0, p): both forms return the canonical a * b * 2^-32 mod p, so
# the result is bit-identical. A compile-time constant operand is replaced by its even
# representative in (-p, p) (c or c - p): t is then even, lo * 2^31 == 0 mod 2^32, and m is one
# shift-add. Only where it measured faster (each part's register allocation changes with it; per
# call, po2 18, with the bounds and scheduler above): part 3 on gfx11 and gfx12, parts 4 and 5 on
# gfx11 only (parts 3/4/5: 1739/2727/4018 -> 1483/2545/3627 us on an RX 7900 XTX; part 3: 1800 ->
# 1468 us on an RX 9070 XT; the other parts, and parts 4/5 on gfx12, were slower with it). Each
# such part is wrapped in a namespace whose operator* is found before sppark's hidden friend (an
# exact match against a derived-to-base conversion); the CUDA build sees the same tokens as before.
EC_MUL_PARTS = {3: 'ec_mul_all', 4: 'ec_mul_gfx11', 5: 'ec_mul_gfx11'}
EC_MUL_NS = {'ec_mul_all': 'defined(__HIP_DEVICE_COMPILE__)', 'ec_mul_gfx11': 'defined(__GFX11__)'}
EC_MUL_BODY = '''namespace NAME {
__device__ __forceinline__ uint32_t opq(uint32_t x) {
  asm("" : "+v"(x));
  return x;
}
__device__ __forceinline__ Fp operator*(Fp a, Fp b) {
#if COND
  constexpr uint32_t P = 0x78000001u;
  uint32_t av = *a, bv = *b;
  int64_t t;
  int32_t m;
  if (__builtin_constant_p(bv) || __builtin_constant_p(av)) {
    bool cb = __builtin_constant_p(bv);
    uint32_t c = cb ? bv : av, x = cb ? av : bv;
    int32_t ce = (c & 1) ? int32_t(c - P) : int32_t(c);
    t = int64_t(int32_t(x)) * ce;
    uint32_t lo = uint32_t(t);
    m = int32_t(lo + (lo << 27));
  } else {
    t = int64_t(uint64_t(av) * bv);
    uint32_t lo = uint32_t(t);
    m = int32_t(opq(lo + (lo << 27)) + (lo << 31));
  }
  uint32_t r = uint32_t(uint64_t(t + int64_t(m) * -int64_t(P)) >> 32);
  return Fp(__builtin_elementwise_min(r, r + P));
#else
  return Fp(static_cast<const bb31_base&>(a) * static_cast<const bb31_base&>(b));
#endif
}
} // namespace NAME
'''
k = 'namespace risc0::circuit::recursion::cuda {\n'
assert new.count(k) == 1
new = new.replace(k, k + '#ifdef __HIPCC__\n' + ''.join(
    EC_MUL_BODY.replace('NAME', n).replace('COND', c) for n, c in EC_MUL_NS.items()) + '#endif\n', 1)
for part, name in EC_MUL_PARTS.items():
    head = '__device__ __forceinline__ FpExt poly_fp_part%d(' % part
    assert new.count(head) == 1, head
    i = new.index(head)
    j = new.index('\n}\n', i) + 3
    new = (new[:i] + '#ifdef __HIPCC__\nnamespace %s {\n#endif\n' % name + new[i:j] +
           '#ifdef __HIPCC__\n} // namespace %s\nusing %s::poly_fp_part%d;\n#endif\n' % (name, name, part) +
           new[j:])

# HIP (RDNA) only: per part and architecture, the measured faster of ec_lazyfp.py's forms (uniform
# trace addressing 'col'; lazy Fp sums of products 'lz1'/'lz2'; raw buffer trace loads 'buf', alone
# or after another form as 'X+buf'), each under
# `#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX1x__)` with the unchanged body as the #else
# branch (so the CUDA build sees the same tokens). Per call, po2 18, median of 3 interleaved
# harness runs, bit-identical check output: gfx1100 part 0 3275 -> 3162 us (col), part 1 3013 ->
# 2739 (lz2), part 2 1727 -> 1410 (lz1); gfx1201 part 4 3040 -> 2909 (lz1); all six parts, 4
# interleaved runs: RX 7900 XTX 15.69 -> 14.86 ms, RX 9070 XT 17.40 -> 16.94 ms. Then 'buf' (medians
# of 10 interleaved runs, us): gfx1100 part 0 3188 -> 2774, part 1 2756 -> 2268, part 2 1366 -> 976,
# part 3 1498 -> 1443, part 4 2546 -> 2342 (part 5 3654 -> 3729, kept); gfx1201 part 0 3548 -> 3387,
# part 2 1606 -> 1554, part 4 2796 -> 2701 (parts 1, 3, 5 slower, with more spills); the full call
# (6 interleaved runs): RX 7900 XTX 14.80 -> 14.08 ms, RX 9070 XT 16.95 -> 16.61 ms. Every other
# part/arch pair was slower with each form (the forms cut VALU work 10-25% but change the part's
# register allocation, 216 VGPRs with the maxocc scheduler); re-measure before changing this.
EC_VARIANTS = {0: {'gfx11': 'col+buf', 'gfx12': 'buf'}, 1: {'gfx11': 'lz2+buf'}, 2: {'gfx11': 'lz1+buf', 'gfx12': 'buf'}, 3: {'gfx11': 'buf'}, 4: {'gfx11': 'buf', 'gfx12': 'lz1+buf'}}
EC_VARIANTS_EXPECT = {'col': 268, 'buf': 1815, 'lz2_prod': 1093, 'lz2_fin': 562, 'lz1_prod': 1167, 'lz1_fin': 727}
new, vcounts = ec_lazyfp.select(new, EC_VARIANTS)
if vcounts != EC_VARIANTS_EXPECT:
    sys.exit('eval_check_split: ec_lazyfp counts %s != %s (input changed: re-validate the variants '
             'with the harness, then update EC_VARIANTS_EXPECT)' % (vcounts, EC_VARIANTS_EXPECT))
STATS += '; HIP variants: %s' % ', '.join('%s %d' % kv for kv in sorted(vcounts.items()))

try:
    cur = open(OUT, newline='').read()
except FileNotFoundError:
    cur = ''
if CHECK:
    if cur != new:
        sys.exit('eval_check_split: eval_check_split.cu is out of date; '
                 'run kernels/cuda/eval_check_split.py and commit the result')
    print('eval_check_split: eval_check_split.cu is up to date (%s)' % STATS)
else:
    open(OUT, 'w', newline='').write(new)
    print('eval_check_split: wrote eval_check_split.cu (%s)' % STATS)
