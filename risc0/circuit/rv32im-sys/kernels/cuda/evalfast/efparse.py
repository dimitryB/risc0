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

# Parser for the generated eval_check functions: statements with defs/uses.
import re
ID=re.compile(r'\b(_wx\d+|_cx\d+|x\d+|arg\d+(?:_e\d)?)\b')
def functions(s):
    out={}; lines=s.split('\n'); i=0
    while i<len(lines):
        m=re.match(r'^__device__ (?:EF_NOINLINE )?(?:FpExt|void) (\w+)\((.*)\) \{$',lines[i])
        if m:
            name=m.group(1); j=i+1; body=[]
            while lines[j]!='}': body.append((j+1,lines[j])); j+=1
            out[name]=dict(start=i+1,end=j+1,sig=m.group(2),body=body); i=j
        i+=1
    return out
def stmts(body):
    """yield dicts: line, text, defs(list), uses(set), kind"""
    res=[]
    for ln,l in body:
        t=l.strip()
        if not t or t.startswith('//'): continue
        for part in [p.strip() for p in t.split(';')]:
            if not part or part.startswith('//'): continue
            d=[]; kind='other'
            m=re.match(r'^(?:const\s+)?(?:auto|FpExt|Fp|RkAcc|RkW|RkW1|constexpr Fp|constexpr FpExt|size_t|const uint32_t)\s+(\w+)\s*(=|\()(.*)$',part)
            rhs=part
            if m:
                d=[m.group(1)]; rhs=m.group(3); kind='def'
            elif part.startswith('RK_ST4') or part.startswith('RK_ST1'): kind='st'
            elif re.match(r'^rk(w|w1)?_\w+\((_wx|_cx)\d+',part) or re.match(r'^rk_cl_acc\(',part):
                v=re.match(r'^\w+\((\w+)',part).group(1); d=[v]; kind='upd'
            elif part.startswith('return'): kind='ret'
            uses=set(ID.findall(rhs))-set(d if kind=='def' else [])
            if kind=='upd': uses|=set(d)
            res.append(dict(line=ln,text=part,defs=d,uses=uses,kind=kind))
    return res
