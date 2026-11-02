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

"""linkext.py: rk_link(base, FpExt c, EF_PM(k)) without the per-row FpExt multiply.

The FpExt overload computed p = c * pm (3 per-row Montgomery muls folding NBETA into the
constant pm, 16 MACs, 4 reductions) and then added p to the accumulator hi words. Now the
products go straight into the 64-bit accumulator using the precomputed NBETA-folded table
(ef_pmn_raw, see close.py): fold hi below P (an rk_link_nf base has hi < 1.47P), then two
products (< 2P^2, hi + 0.9375P < 2^32) and a fold, twice. Every call site becomes
rk_link_k(base, c, k), overloaded on the type of c (Fp: unchanged code path).
Must run after close.py (which adds ef_pmn_raw and its host-side fill).
usage: linkext.py in.cu out.cu
"""
import re, sys
src = open(sys.argv[1]).read()
assert 'ef_pmn_raw' in src, 'run close.py first'
HELP = r'''
// ---- linkext.py: rk_link with an FpExt term, products accumulated lazily ----
template <typename A> __device__ __forceinline__ static RkAcc rk_link_k(const A& base, Fp c, uint32_t k) { return rk_link(base, c, EF_PM(k)); }
template <typename A> __device__ __forceinline__ static RkAcc rk_link_nf_k(const A& base, Fp c, uint32_t k) { return rk_link_nf(base, c, EF_PM(k)); }
template <typename A> __device__ __forceinline__ static RkAcc rk_link_k(const A& base, FpExt c, uint32_t k) {
  RkAcc s = rk_acc(base);
  const uint32_t a0 = c.elems[0].asRaw(), a1 = c.elems[1].asRaw(), a2 = c.elems[2].asRaw(), a3 = c.elems[3].asRaw();
  const uint4 b = reinterpret_cast<const uint4*>(ef_pm_raw)[k], n = reinterpret_cast<const uint4*>(ef_pmn_raw)[k];
  EF_PMROW(b, n);
  for (int j = 0; j < 4; j++) {
    uint64_t t = ef_fold64((uint64_t(s.hi[j]) << 32) | s.lo[j]);
    t = ef_fold64(ef_madw(a1, X[j][1], ef_madw(a0, X[j][0], t)));
    t = ef_fold64(ef_madw(a3, X[j][3], ef_madw(a2, X[j][2], t)));
    s.lo[j] = uint32_t(t); s.hi[j] = uint32_t(t >> 32);
  }
  return s;
}
template <typename A> __device__ __forceinline__ static RkAcc rk_link_nf_k(const A& base, FpExt c, uint32_t k) { return rk_link_k(base, c, k); }
'''
n = 0
def rep(m):
    global n
    n += 1
    return '%s_k(%s, %s, %su)' % (m.group(1), m.group(2), m.group(3), m.group(4))
LINK = re.compile(r'\b(rk_link(?:_nf)?)\((\w+), ([^()]*?), EF_PM\((\d+)\)\)')
src2 = '\n'.join(LINK.sub(rep, l) for l in src.split('\n'))
anchor = '// ---- algebraic rewrite helpers (ec-algebra) ----'
assert anchor in src2
# the template definitions themselves keep EF_PM(k) (no literal index there)
src2 = src2.replace(anchor, HELP + anchor, 1)
open(sys.argv[2], 'w').write(src2)
print('linkext: %d call sites' % n, file=sys.stderr)
