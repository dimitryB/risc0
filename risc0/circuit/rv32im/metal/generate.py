#!/usr/bin/env python3
"""Translate the pinned RV32IM polynomial's address spaces, preserving its arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from split import split_chain


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calls(body):
    # Generated calls can span lines. Balance parentheses rather than matching
    # an argument list with a regex; preserve all original arguments verbatim.
    matches=list(re.finditer(r'\brv32im_v2_\d+\(',body))
    for m in reversed(matches):
        depth=1;end=m.end()
        while depth:
            if body[end]=='(':depth+=1
            elif body[end]==')':depth-=1
            end+=1
        body=body[:end-1]+', poly_mix'+body[end-1:]
    return body


def normalize(body):
    # Compare generated CUDA against the authoritative CPU oracle. These are
    # spelling changes plus commutation of field products, not simplification
    # of the circuit: both Fp and FpExt multiply in the same commutative field.
    body=re.sub(r'//[^\n]*|/\*.*?\*/','',body,flags=re.S)
    body=re.sub(r'\bcycle\b','idx',body)
    body=re.sub(r'\bsteps\b','size',body)
    body=body.replace('kInvRate','INV_RATE').replace('size_t mask','uint32_t mask')
    body=re.sub(r'\b(?:constexpr\s+)?(?:FpExt|Fp|auto)\s+(x\d+)',r'var \1',body)
    body=re.sub(r'\s+','',body).replace('idx,size,poly_mix,','idx,size,')
    for i,name in [(0,'accum'),(1,'data'),(2,'out'),(3,'mix')]:body=body.replace(f'args[{i}]',name)
    body=re.sub(r'(x\d+)\*(poly_mix\[\d+\])',r'\2*\1',body)
    return body.replace('{','(').replace('}',')')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split-chains',default=','.join(f'rv32im_v2_{i}' for i in range(20)),
                        help='comma-separated generated chain functions to split for WP1')
    parser.add_argument('--fast-ext',action='store_true',
                        help='use the in-tree evalfast lazy extension-field reduction in MSL')
    parser.add_argument('--fast-fp',action='store_true',
                        help='use the in-tree EF_FP_FAST Montgomery forms in MSL')
    args=parser.parse_args();engine=args.engine.resolve()
    split_names=set(filter(None,args.split_chains.split(',')))
    assert split_names <= {f'rv32im_v2_{i}' for i in range(20)}
    base=engine/'risc0/circuit/rv32im-sys/kernels/cuda'
    chunks=sorted(base.glob('eval_check_[0-9].cu'))
    assert len(chunks)==4
    functions={};sources=[];cuda_bodies={}
    for path in chunks:
        sources.append(path)
        text=path.read_text()
        found=list(re.finditer(r'__device__ FpExt (rv32im_v2_\d+|poly_fp)\((.*?)\) \{(.*?)\n\}',text,re.S))
        assert len(found)==text.count('__device__ FpExt '), 'generated function structure changed'
        for m in found:
            name,signature,body=m.groups()
            cuda_bodies[name]=normalize(body)
            structural=re.sub(r'FpExt x\d+\{[0-9, ]+\};','',body)
            assert name not in functions and '{' not in structural and '}' not in structural
            signature=re.sub(r'const Fp\*', 'const device Fp*',signature)
            signature=re.sub(r'(?<!device )\b(Fp(?:Ext)?)\*',r'thread \1*',signature)
            signature+=', const device FpExt* poly_mix'
            prefix='__attribute__((noinline)) ' if name!='poly_fp' else ''
            declaration=f'{prefix}FpExt {name}({signature})'
            functions[name]=(declaration,calls(body))
    assert set(functions)=={'poly_fp'}|{f'rv32im_v2_{i}' for i in range(20)}
    cpu_bodies={}
    for path in sorted((base.parent/'cxx').glob('rust_poly_fp_*.cpp')):
        sources.append(path)
        for m in re.finditer(r'FpExt (rv32im_v2_\d+|poly_fp)\([^;{]+?\) \{(.*?)\n\}',path.read_text(),re.S):
            assert m[1] not in cpu_bodies
            cpu_bodies[m[1]]=normalize(m[2])
    assert cpu_bodies==cuda_bodies, 'CPU/CUDA polynomial differs beyond the reviewed spelling/field-product normalization'
    source='// Derived from the pinned generated circuit; do not hand-edit arithmetic.\n'
    for path in [engine/'risc0/build_kernel/kernels/metal'/name for name in ['fp.h','fpext.h']]:
        sources.append(path)
        source+=f'\n#line 1 "{path.name}"\n'
        header=path.read_text()
        if path.name=='fp.h' and args.fast_fp:
            bodies={
                'add': '''uint32_t r = a + b;
    uint32_t s = r - P;
    return r < s ? r : s;''',
                'sub': '''uint32_t r = a - b;
    uint32_t s = r + P;
    return r < s ? r : s;''',
                'mul': '''uint32_t lo = a * b;
    uint32_t hi = uint32_t((uint64_t(a) * uint64_t(b)) >> 32);
    uint32_t m = lo * M;
    uint32_t r = hi - uint32_t((uint64_t(m) * P) >> 32);
    uint32_t s = r + P;
    return r < s ? r : s;''',
            }
            for operation,body in bodies.items():
                pattern=rf'(  static constexpr uint32_t {operation}\(uint32_t a, uint32_t b\) \{{).*?(\n  \}})'
                header,count=re.subn(pattern,lambda m:m[1]+'\n    '+body+m[2],header,count=1,flags=re.S)
                assert count==1, f'Metal Fp {operation} shape changed'
        if path.name=='fpext.h' and args.fast_ext:
            fast=Path(__file__).resolve().parent/'fast-ext.metal';sources.append(fast)
            pattern=r'  constexpr FpExt operator\*\(FpExt rhs\) const \{.*?\n  \}'
            header,count=re.subn(pattern,lambda _:fast.read_text().rstrip(),header,count=1,flags=re.S)
            assert count==1, 'Metal FpExt multiplication shape changed'
        for line in header.splitlines():
            if line.strip() in ['#pragma once','#include "fp.h"']:continue
            assert not line.strip().startswith('#include "'),line
            source+=line+'\n'
    compat=Path(__file__).resolve().parent/'field-compat.metal'
    source+=compat.read_text();sources.append(compat)
    source+='\nusing namespace metal;\nconstant uint INV_RATE = 4;\n'
    source+='\n'.join(d+';' for d,_ in functions.values())+'\n'
    split_records=[]
    for name,(declaration,body) in functions.items():
        if name in split_names:
            helper,body,record=split_chain(name,declaration,body)
            source+=f'\n#line 1 "{name}_tail.msl"\n{helper}'
            split_records.append(record)
        source+=f'\n#line 1 "{name}.msl"\n{declaration} {{{body}\n}}\n'
    wrapper=Path(__file__).resolve().parent/'eval-check.metal'
    source+=wrapper.read_text();sources.append(wrapper)
    if split_names:
        sources.append(Path(__file__).resolve().parent/'split.py')
    info=engine/'risc0/circuit/rv32im/src/zirgen/info.rs';sources.append(info)
    powers=re.search(r'pub const POLY_MIX_POWERS: &\[usize\] = &\[(.*?)\];',info.read_text(),re.S)
    exponents=[int(x) for x in re.findall(r'\d+',powers[1])]
    assert len(exponents)==458
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(source)
    report={'engine_root':str(engine),
        'generator_sha256':sha(Path(__file__)),'source_sha256':sha(args.output),
        'source_files':{str(p):sha(p) for p in sources},'function_count':len(functions),
        'poly_mix_exponents':exponents,
        'cpu_cuda_normalized_body_sha256':{name:hashlib.sha256(body.encode()).hexdigest() for name,body in cuda_bodies.items()},
        'translation':'thread/device pointer address spaces, explicit poly_mix argument, noinline function boundaries; original polynomial statements retained',
        'split_chains':split_records,'fast_ext':args.fast_ext,'fast_fp':args.fast_fp}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
