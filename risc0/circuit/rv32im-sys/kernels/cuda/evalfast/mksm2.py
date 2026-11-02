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

# mksm2.py in.cu out.cu NMAIN NLU : per-phase hot slab words in shared memory (for the phase-split kernel).
# Main-chain functions map their NMAIN hottest words, LogUp-chain functions their NLU hottest words, onto the
# same static smem array (a block runs only one phase). smem words = max(NMAIN, NLU).
import re,sys,collections
from efparse import functions
src,out,NM,NL=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
s=open(src).read(); F=functions(s); lines=s.split('\n')
LU={'rk_accblk','rv32im_v2_2','rv32im_v2_1','rv32im_v2_0'}
cnt={'m':collections.Counter(),'l':collections.Counter()}
for fn,f in F.items():
    if not (fn.startswith('rv32im_v2_') or fn in ('poly_fp','rk_accblk')): continue
    g='l' if fn in LU else 'm'
    for ln,l in f['body']:
        for m in re.finditer(r'RK_(LD|ST)4\(\w+, (\d+)u, (\d+)u, (\d+)u, (\d+)u',l):
            for w in m.groups()[1:]: cnt[g][int(w)]+=1
        for m in re.finditer(r'RK_(LD|ST)1\(\w+, (\d+)u',l): cnt[g][int(m.group(2))]+=1
    for i in range(f['start'],f['end']):
        lines[i-1]=re.sub(r'RK_(LD|ST)(4|1)\(', lambda m: f'RK_{m.group(1)}{m.group(2)}{g.upper()}(', lines[i-1])
s='\n'.join(lines)
hm=[w for w,_ in cnt['m'].most_common(NM)]; hl=[w for w,_ in cnt['l'].most_common(NL)]
print('main words',len(cnt['m']),'acc',sum(cnt['m'].values()),'hot',sum(cnt['m'][w] for w in hm),
      '| lu words',len(cnt['l']),'acc',sum(cnt['l'].values()),'hot',sum(cnt['l'][w] for w in hl))
N=max(NM,NL,1)
tab=''
for g,h in (('M',hm),('L',hl)):
    tab+=f'__device__ __forceinline__ constexpr int ef_hot{g}(uint32_t w) {{ switch (w) {{ ' + ' '.join(f'case {w}u: return {k};' for k,w in enumerate(h)) + ' default: return -1; } }\n'
    tab+=f'''#define RK_LD1{g}(p, w) (static_cast<const ::risc0::Fp>(::risc0::Fp::fromRaw(ef_hot{g}(w) >= 0 ? EF_SMW(ef_hot{g}(w) < 0 ? 0 : ef_hot{g}(w)) : ef_ldl((p), (w)))))
#define RK_ST1{g}(p, w, v) do {{ if (ef_hot{g}(w) >= 0) EF_SMW(ef_hot{g}(w) < 0 ? 0 : ef_hot{g}(w)) = ::risc0::Fp(v).asRaw(); else ef_stl((p), (w), ::risc0::Fp(v).asRaw()); }} while (0)
#define RK_LD4{g}(p, w0, w1, w2, w3) (static_cast<const ::risc0::FpExt>(::risc0::FpExt(RK_LD1{g}(p, w0), RK_LD1{g}(p, w1), RK_LD1{g}(p, w2), RK_LD1{g}(p, w3))))
#define RK_ST4{g}(p, w0, w1, w2, w3, v) do {{ ::risc0::FpExt _rk_v = (v); RK_ST1{g}(p, w0, _rk_v.elems[0]); RK_ST1{g}(p, w1, _rk_v.elems[1]); RK_ST1{g}(p, w2, _rk_v.elems[2]); RK_ST1{g}(p, w3, _rk_v.elems[3]); }} while (0)
'''
hdr=f'#define EF_NHOT {N}\n__shared__ uint32_t ef_sm[EF_NHOT][EF_THREADS];\n#define EF_SMW(k) ef_sm[k][threadIdx.x]\n'+tab
k=s.index('#define RK_ST4(p, w0, w1, w2, w3, v)'); k=s.index('\n',k)+1
s=s[:k]+hdr+s[k:]
# The hot words take EF_NHOT * EF_THREADS * 4 bytes of static shared memory per block: ask for the
# largest shared-memory carveout so EF_MINB blocks fit per SM (the default cache preference
# would leave it to the driver).
old='CUDA_OK(cudaFuncSetCacheConfig((const void*)ef_eval_split, cudaFuncCachePreferL1));'
assert s.count(old)==1
s=s.replace(old,'CUDA_OK(cudaFuncSetAttribute((const void*)ef_eval_split, cudaFuncAttributePreferredSharedMemoryCarveout, 100));')
open(out,'w').write(s)
