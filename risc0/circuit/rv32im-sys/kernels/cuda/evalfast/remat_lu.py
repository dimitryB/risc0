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

"""remat_lu.py: rematerialize LogUp-chain slab FpExt values at their reads.

The LogUp chain (rk_accblk -> rv32im_v2_2 -> rv32im_v2_1 -> rv32im_v2_0) passes FpExt values through the
per-thread slab (RK_ST4 / RK_LD4).  Most are pure functions of read-only inputs (data/accum/mix loads,
poly_mix constants, literals).  For each RK_LD4 whose producing store has a pure backward slice within the
cost limits, the slice is cloned (vars renamed) in place of the load; stores whose reads are all
rematerialized are dropped.  Same Fp/FpExt operations on the same canonical inputs -> identical values.
usage: remat_lu.py in.cu out.cu [--maxg N] [--maxops N]
"""
import re,sys,collections
from efparse import ID, functions, stmts
a=sys.argv
def opt(n,d): return a[a.index(n)+1] if n in a else d
MAXG=int(opt('--maxg',99)); MAXOPS=int(opt('--maxops',999)); ONLY=('rv32im_v2_0','rv32im_v2_1','rv32im_v2_2')
s=open(a[1]).read(); F=functions(s); L=s.split('\n')
order=['rk_accblk','rv32im_v2_2','rv32im_v2_1','rv32im_v2_0']
# pointer-arg -> buffer per function (D=data, A=accum, M=mix)
bufmap={'rk_accblk':{'arg7':'D','arg8':'A','arg9':'M'},'rv32im_v2_2':{'arg3':'D','arg4':'A','arg5':'M'},
        'rv32im_v2_1':{'arg4':'D','arg5':'A'},'rv32im_v2_0':{'arg4':'D'}}
ST=[]  # store instances in execution order: dict(fn, idx, word, val, slice, ok, reads)
info={}
def parse(fn):
    st=stmts(F[fn]['body']); defs={}; upds=collections.defaultdict(list)
    for i,x in enumerate(st):
        if x['kind']=='def':
            for d in x['defs']: defs[d]=i
        if x['kind']=='upd': upds[x['defs'][0]].append(i)
    return st,defs,upds
P={fn:parse(fn) for fn in order}
last={}
loads=[]  # (fn, stmt idx, var, word, store instance)
for fn in order:
    st,defs,upds=P[fn]
    for i,x in enumerate(st):
        t=x['text']
        if 'RK_LD4' in t:
            m=re.search(r'RK_LD4\(\w+, (\d+)u',t); w=int(m.group(1))
            inst=last.get(w)
            loads.append((fn,i,x['defs'][0],w,inst))
            if inst is not None: inst['reads'].append(len(loads)-1)
        if x['kind']=='st':
            m=re.match(r'RK_ST4\(\w+, (\d+)u, \d+u, \d+u, \d+u, (.*)\)$',t); w=int(m.group(1)); val=m.group(2).strip()
            # backward slice (statement indices)
            sl=set(); ok=True; g=0; ops=0
            stack=[v for v in ID.findall(val)]
            seen=set()
            while stack:
                v=stack.pop()
                if v in seen: continue
                seen.add(v)
                if v.startswith('arg'):
                    if v not in bufmap[fn]: ok=False
                    continue
                if v not in defs: ok=False; continue
                idxs=[defs[v]]+upds.get(v,[])
                for k in idxs:
                    tt=st[k]['text']
                    if 'RK_LD4' in tt or 'rk_link' in tt or 'rk_fin' in tt or '_cx' in tt or 'rv32im_v2' in tt: ok=False
                    if k not in sl:
                        sl.add(k)
                        if 'EF_LD' in tt: g+=1
                        elif not tt.startswith('constexpr') and 'EF_PM' not in tt: ops+=1
                    stack+= [u for u in st[k]['uses'] if u!=v]
            inst=dict(fn=fn,idx=i,word=w,val=val,slice=sorted(sl),ok=ok and g<=MAXG and ops<=MAXOPS,g=g,ops=ops,reads=[])
            ST.append(inst); last[w]=inst
# decide remat per load
remat={}  # (fn,idx) -> replacement lines at the load site
cnt=collections.Counter()
for li,(fn,i,var,w,inst) in enumerate(loads):
    if inst is None or fn not in ONLY or not inst['ok']: cnt['keep']+=1; continue
    pfn=inst['fn']; pst=P[pfn][0]
    def clone(prefix):
        def ren(t):
            t=re.sub(r'\b(_wx\d+|_cx\d+|x\d+)\b', lambda m: prefix+m.group(1), t)
            t=re.sub(r'\b(arg\d+)\b', lambda m: 'rkB'+bufmap[pfn][m.group(1)] if m.group(1) in bufmap[pfn] else m.group(1), t)
            return t
        return [ren(pst[k]['text'])+';' for k in inst['slice']], ren(inst['val'])
    lines,val=clone(f'r{li}_')
    remat[(fn,i)]=lines+[f'auto {var} = static_cast<const ::risc0::FpExt>(::risc0::FpExt({val}));']; cnt['remat']+=1
    inst.setdefault('done',set()).add(li)
dropst=set()
for inst in ST:
    if inst['reads'] and all(r in inst.get('done',()) for r in inst['reads']): dropst.add((inst['fn'],inst['idx']))
print('loads',len(loads),dict(cnt),'stores',len(ST),'dropped',len(dropst),
      'remat cost g/ops:',sum(ST_['g'] for ST_ in ST if (ST_['fn'],ST_['idx']) in dropst),sum(ST_['ops'] for ST_ in ST if (ST_['fn'],ST_['idx']) in dropst))
# rewrite: map statement indices back to lines. Re-emit each function body statement-wise.
for fn in order:
    st=P[fn][0]; f=F[fn]
    body=[]
    for i,x in enumerate(st):
        if (fn,i) in remat: body+= ['  '+t for t in remat[(fn,i)]]
        elif (fn,i) in dropst: body.append('  // remat_lu: store dropped')
        else: body.append('  '+x['text']+';')
    ins=[]
    if fn=='rk_accblk': ins=['  Fp* rkBD = arg7; Fp* rkBA = arg8; Fp* rkBM = arg9;']
    L[f['start']:f['end']-1]=['\x00']*(f['end']-1-f['start'])
    L[f['start']]='\n'.join(ins+body)
s='\n'.join(x for x in L if x!='\x00')
# signatures/calls: add rkBD/rkBA/rkBM params to v2_2/v2_1/v2_0
for fn in ['rv32im_v2_2','rv32im_v2_1','rv32im_v2_0']:
    s=re.sub(r'(__device__ (?:EF_NOINLINE )?FpExt '+fn+r'\(size_t cycle[^)]*)\)', r'\1, Fp* rkBD, Fp* rkBA, Fp* rkBM)', s)
    s=re.sub(r'(= '+fn+r'\(cycle, steps, poly_mix,[^;]*)\);', r'\1, rkBD, rkBA, rkBM);', s)
open(a[2],'w').write(s)
