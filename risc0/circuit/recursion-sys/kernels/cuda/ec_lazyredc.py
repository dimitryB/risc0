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

"""Post-pass for the split recursion eval_check kernel (eval_check_split.cu).

eval_check_split.py imports this module and applies rewrite() to the split
source before writing (or, with --check, comparing) eval_check_split.cu.
Standalone use, for experiments: ec_lazyredc.py IN.cu OUT.cu

1. Lazy Montgomery reduction of the mix chains

       FpExt xN = xM + poly_mix[k] * xC;      (xC: Fp)

   Each of the four lanes accumulates the raw 64-bit products
   poly_mix[k].u[i] * xC.val in a uint64_t (RecLz). The high word is folded
   below P whenever one more product could overflow 64 bits (the bound is
   tracked exactly here, per variable), and one REDC + final subtraction
   materializes the FpExt (`xN_m`) where a non-chain statement uses it. REDC
   is linear mod P and its output is canonical, so the result is bit-identical
   to the per-term Montgomery multiply-add.

   Canonical-input assumption: every operand is a canonical Montgomery residue
   in [0, P): the poly_mix limbs (reduced host Elems), the chain Fp operands
   (outputs of bb31_t arithmetic, which always reduces) and the chain base
   FpExt. The bound (P - 1)^2 per product depends on it; a non-canonical raw
   value in [P, 2^32) could overflow the uint64_t accumulator. The prover's
   buffers only ever hold canonical values.

2. A module warmup entry point, rec_eval_check_split_warmup(size_t*), that
   loads the ec_part kernels and the poly_mix symbol off the GPU thread (the
   first lift otherwise pays ~3.7 ms of lazy module loading) and returns their
   largest per-thread local memory size (ffi.cu pre-grows the stack limit).

The pass refuses already-rewritten input (MARK); EXPECT pins its counts.
"""

import re
import sys

P = 0x78000001
U64 = (1 << 64) - 1
PROD = (P - 1) ** 2  # one raw product of canonical residues
FOLDED = P * (1 << 32) - 1  # bound after a fold (high word < P)
MARK = '// ---- ec_lazyredc.py: lazy Montgomery accumulation of FpExt += poly_mix[k] * Fp ----'
NS = 'namespace risc0::circuit::recursion::cuda {'
NS_END = '} // namespace risc0::circuit::recursion::cuda'
# chain terms, folds, chains started from a materialized FpExt, materializations
EXPECT = (1060, 400, 18, 119)

CHAIN = re.compile(r'^(\s*)FpExt (x\d+) = (x\d+) \+ poly_mix\[(\d+)\] \* (x\d+);$')
TOK = re.compile(r'\bx\d+\b')
LHS = re.compile(r'^\s*(?:FpExt |Fp )?(x\d+) =')

HELPERS = MARK + r'''
// Canonical-input assumption: every operand is a residue in [0, P) (see the generator).
struct RecLz {
  uint64_t a[4];
};
__device__ __forceinline__ uint32_t rec_raw(const Fp& x) {
  return *reinterpret_cast<const uint32_t*>(&x);
}
__device__ __forceinline__ RecLz rec_lz_from(const FpExt& e) {
  const uint32_t* u = reinterpret_cast<const uint32_t*>(&e);
  RecLz r;
  for (int i = 0; i < 4; i++)
    r.a[i] = uint64_t(u[i]) << 32;
  return r;
}
__device__ __forceinline__ RecLz rec_lz_zero() {
  RecLz r;
  for (int i = 0; i < 4; i++)
    r.a[i] = 0;
  return r;
}
__device__ __forceinline__ RecLz rec_lz_add(RecLz r, int k, const Fp& c) {
  const uint32_t* m = reinterpret_cast<const uint32_t*>(&poly_mix[k]);
  uint32_t cv = rec_raw(c);
  for (int i = 0; i < 4; i++)
    r.a[i] += uint64_t(m[i]) * cv;
  return r;
}
// High word (< 2^32 < 3P) reduced below P: subtracting P * 2^32 keeps the value mod P.
__device__ __forceinline__ RecLz rec_lz_fold(RecLz r) {
  for (int i = 0; i < 4; i++) {
    uint32_t lo = uint32_t(r.a[i]), hi = uint32_t(r.a[i] >> 32);
    hi = min(hi, hi - 0x78000001u);
    hi = min(hi, hi - 0x78000001u);
    r.a[i] = (uint64_t(hi) << 32) | lo;
  }
  return r;
}
// Montgomery REDC of a folded accumulator (< P * 2^32), canonical output.
__device__ __forceinline__ FpExt rec_lz_fin(RecLz r) {
  r = rec_lz_fold(r);
  FpExt e;
  uint32_t* u = reinterpret_cast<uint32_t*>(&e);
  for (int i = 0; i < 4; i++) {
    uint32_t red = uint32_t(r.a[i]) * 0x77ffffffu;
    uint64_t t = r.a[i] + uint64_t(red) * 0x78000001u;
    uint32_t v = uint32_t(t >> 32);
    u[i] = v >= 0x78000001u ? v - 0x78000001u : v;
  }
  return e;
}
'''

WARMUP_NS = '''
template <int P> static void warm_parts(size_t* localBytes) {
  if constexpr (P < kNumParts) {
    cudaFuncAttributes attr;
    CUDA_OK(cudaFuncGetAttributes(&attr, ec_part<P>));
    if (attr.localSizeBytes > *localBytes)
      *localBytes = attr.localSizeBytes;
    warm_parts<P + 1>(localBytes);
  }
}

'''

WARMUP_C = '''
// Loads this module's kernels and its poly_mix symbol off the GPU thread (the first lift
// otherwise paid ~3.7 ms of lazy loading: 1.4 ms in the symbol copy, ~0.8 ms per first launch)
// and returns their largest per-thread local memory size.
extern "C" const char* rec_eval_check_split_warmup(size_t* localBytes) {
  try {
    *localBytes = 0;
    risc0::circuit::recursion::cuda::warm_parts<0>(localBytes);
    void* p = nullptr;
    CUDA_OK(cudaGetSymbolAddress(&p, poly_mix));
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}
'''


def rewrite(text):
    if MARK in text:
        sys.exit('ec_lazyredc: input already rewritten')
    src = text.split('\n')
    zero = set(re.findall(r'FpExt (x\d+) = FpExt\(0\);', text))
    lazy = {}  # var -> bound of its accumulator
    mat = set()  # materialized vars (xN_m defined)
    out = []
    nchain = nfold = nfrom = nfin = 0
    helpers_done = False
    for l in src:
        if not helpers_done and l == NS:
            out += [l, '', HELPERS.rstrip('\n')]
            helpers_done = True
            continue
        m = CHAIN.match(l)
        if m:
            ind, v, b, k, c = m.groups()
            if b in lazy:
                bound, base = lazy[b], b
            elif b in zero:
                bound, base = 0, 'rec_lz_zero()'
            else:
                bound = (P - 1) << 32
                base = 'rec_lz_from(%s)' % (b + '_m' if b in mat else b)
                nfrom += 1
            if bound + PROD > U64:
                base = 'rec_lz_fold(%s)' % base
                bound = FOLDED
                nfold += 1
            bound += PROD
            assert bound <= U64
            lazy[v] = bound
            nchain += 1
            out.append('%sRecLz %s = rec_lz_add(%s, %s, %s);' % (ind, v, base, k, c))
            continue
        # Any other statement: materialize the lazy vars it reads.
        lhs = LHS.match(l)
        used = [t for t in TOK.findall(l) if t in lazy and not (lhs and t == lhs.group(1))]
        for t in dict.fromkeys(used):
            if t not in mat:
                ind = re.match(r'^(\s*)', l).group(1)
                out.append('%sFpExt %s_m = rec_lz_fin(%s);' % (ind, t, t))
                mat.add(t)
                nfin += 1
        if used:
            l = TOK.sub(lambda mm: mm.group(0) + '_m' if mm.group(0) in lazy else mm.group(0), l)
        out.append(l)
    if not helpers_done:
        sys.exit('ec_lazyredc: namespace line not found')
    text = '\n'.join(out)
    # warmup entry point
    for inc in ('#include <stdexcept>', '#include <cstring>'):
        if inc not in text:
            text = text.replace('#include <cstdint>\n', '#include <cstdint>\n' + inc + '\n', 1)
    i = text.rindex(NS_END)
    text = text[:i] + WARMUP_NS.lstrip('\n') + NS_END + '\n' + WARMUP_C + text[i + len(NS_END) + 1:]
    return text, (nchain, nfold, nfrom, nfin)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    text, counts = rewrite(open(sys.argv[1], newline='').read())
    print('ec_lazyredc: chain terms %d, folds %d, from %d, materialized %d' % counts,
          file=sys.stderr)
    if counts != EXPECT:
        sys.exit('ec_lazyredc: counts %s != EXPECT %s (generator input changed: re-validate with '
                 'the lazy-REDC harness, then update EXPECT)' % (counts, EXPECT))
    open(sys.argv[2], 'w', newline='').write(text)


if __name__ == '__main__':
    main()
