#!/usr/bin/env python3
"""Translate the pinned evalfast pipeline, including hot slab and chain tails, to MSL."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regenerate(engine):
    """Run the pinned evalfast passes through chain-tail splitting."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(engine / 'risc0/circuit/rv32im-sys/kernels/cuda/evalfast'))
    import regen

    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('RISC0_', 'SINK_LEAD', 'PYTHON'))}
    environment.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1',
                       RISC0_INTEL_EVAL_XMX='0')
    with tempfile.TemporaryDirectory(prefix='rv32im-evalfast-') as directory:
        root = Path(directory)
        cse = root / 'cse'
        cse.mkdir()
        regen.run(environment, 'kernels/intel/poly_fp_cse.py', 'kernels/cxx',
                  str(cse), '--top', '16', '--min-share', '10')
        amalg = root / 'amalg.cpp'
        amalg.write_text(regen.amalgamate(str(cse)))
        fast = root / 'fast.cpp'
        regen.run(environment, 'kernels/intel/evalfast/pipeline.py',
                  str(amalg), str(fast), str(root / 'work'))
        source = root / 'base.cu'
        regen.run(environment, str(Path(regen.HERE) / 'to_cuda.py'),
                  str(fast), str(source))
        for index, (tool, extra, expected) in enumerate(regen.PASSES):
            next_source = root / f'p{index}.cu'
            log = regen.run(environment, str(Path(regen.HERE) / tool),
                            str(source), str(next_source), *extra)
            assert all(line in log for line in expected), (tool, log[-1000:])
            source = next_source
            if tool == regen.HIP_LAST:
                # 32 hot words x 256 threads x 4 bytes exhausts Metal's 32 KiB
                # threadgroup budget. The other slab words stay private.
                for next_tool, next_args in (
                    ('mksm2.py', ['32', '32']),
                    ('mksub.py', regen.PASSES[-1][1]),
                ):
                    next_source = root / f'{next_tool}.cu'
                    log = regen.run(environment, str(Path(regen.HERE) / next_tool),
                                    str(source), str(next_source), *next_args)
                    if next_tool == 'mksub.py':
                        assert all(line in log for line in regen.MKSUB), log[-2000:]
                    source = next_source
                break
        return source.read_text()


def headers(engine):
    out = ["// Generated evalfast phase-split candidate; not wired into proving.\n"]
    for name in ("fp.h", "fpext.h"):
        path = engine / "risc0/build_kernel/kernels/metal" / name
        for line in path.read_text().splitlines():
            if line.strip() in ("#pragma once", '#include "fp.h"'):
                continue
            assert not line.strip().startswith('#include "'), line
            out.append(line + "\n")
    out.append((Path(__file__).with_name("field-compat.metal")).read_text())
    out.append("\nusing namespace metal;\n")
    return "".join(out)


def pointer_templates(text):
    pat = re.compile(r'(?m)^((?:__attribute__\(\(noinline\)\) )?FpExt (rv32im_v2_\d+)\()([^\n]+?)(\)(?:;| \{))')
    seen = set()

    def convert(match):
        prefix, name, args, suffix = match.groups()
        pointers = [param for param in re.findall(r'\b(?:Fp|FpExt)\* (\w+)\b', args)
                    if param != 'poly_mix']
        assert pointers and len(pointers) == len(set(pointers)), name
        for index, param in enumerate(pointers):
            args, count = re.subn(r'\b(?:Fp|FpExt)\* ' + param + r'\b',
                                  f'P{index} {param}', args, count=1)
            assert count == 1
        template = 'template <' + ', '.join(f'typename P{i}' for i in range(len(pointers))) + '>\n'
        seen.add(name)
        return template + prefix + args + suffix

    text, count = pat.subn(convert, text)
    assert seen == {f'rv32im_v2_{i}' for i in range(20)}, (count, seen)
    return text


def shared_calls(code):
    """Carry Metal's explicit threadgroup pointer through the generated calls."""
    names = re.compile(r'\b(?:rv32im_v2_\d+(?:_t)?|poly_fp|rk_accblk)\(')
    replacements = []
    declarations = 0
    calls = 0
    for match in names.finditer(code):
        depth = 1
        end = match.end()
        while depth:
            assert end < len(code), match.group()
            if code[end] == '(':
                depth += 1
            elif code[end] == ')':
                depth -= 1
            end += 1
        line = code.rfind('\n', 0, match.start()) + 1
        prefix = code[line:match.start()]
        declaration = bool(re.search(r'\bFpExt\s+$', prefix))
        if declaration:
            assert code[end:end + 2] in (' {', ';\n') or code[end] == ';'
            addition = ', threadgroup uint* ef_sm, uint ef_lane'
            declarations += 1
        else:
            addition = ', ef_sm, ef_lane'
            calls += 1
        replacements.append((end - 1, addition))
    for pos, addition in reversed(replacements):
        code = code[:pos] + addition + code[pos:]
    assert declarations >= 40 and calls >= 20, (declarations, calls)
    return code


def shared_macros(code):
    assert '#define EF_NHOT 32' in code
    code = code.replace('__shared__ uint32_t ef_sm[EF_NHOT][EF_THREADS];', '')
    code = code.replace('#define EF_SMW(k) ef_sm[k][threadIdx.x]',
                        '#define EF_SMW(k) ef_sm[(k) * EF_THREADS + ef_lane]')
    for phase in ('M', 'L'):
        code, load_count = re.subn(
            rf'(?m)^#define RK_LD1{phase}\(p, w\).+$',
            f'#define RK_LD1{phase}(p, w) (ef_hot{phase}(w) >= 0 ? '
            f'Fp::fromRaw(EF_SMW(ef_hot{phase}(w))) : '
            'reinterpret_cast<const thread Fp*>(p)[(size_t)(w)])', code)
        code, store_count = re.subn(
            rf'(?m)^#define RK_ST1{phase}\(p, w, v\).+$',
            f'#define RK_ST1{phase}(p, w, v) do {{ Fp _rk_scalar = Fp(v); '
            f'if (ef_hot{phase}(w) >= 0) EF_SMW(ef_hot{phase}(w)) = _rk_scalar.asRaw(); '
            'else reinterpret_cast<thread Fp*>(p)[(size_t)(w)] = _rk_scalar; } while (0)', code)
        assert load_count == store_count == 1
    assert 'threadIdx' not in code and '__shared__' not in code
    return code


def transform(src):
    start = src.index('namespace risc0::circuit::rv32im_v2 {')
    end = src.index('} // namespace risc0::circuit::rv32im_v2', start) + len('} // namespace risc0::circuit::rv32im_v2')
    code = src[start:end]
    code = code.replace('::risc0::FpExt', 'FpExt').replace('::risc0::Fp', 'Fp')
    code = code.replace('__device__ __forceinline__', 'inline')
    code = code.replace('__device__ EF_NOINLINE', '__attribute__((noinline))')
    code = code.replace('__device__ ', '')
    code = code.replace('EF_NOINLINE', '__attribute__((noinline))')
    code = re.sub(r'(?m)^constexpr (size_t|uint32_t) (\w+) = ([^;]+);',
                  r'constant \1 \2 = \3;', code)
    code = re.sub(r'\bconst (RkW|RkW1|RkAcc|FpExt|A)&', r'const thread \1&', code)
    code = re.sub(r'(?<!\w)(RkW|RkW1|RkAcc|FpExt|A)&', r'thread \1&', code)
    code = code.replace('const thread thread ', 'const thread ')
    code = code.replace('const thread FpExt&', 'FpExt').replace('const thread A&', 'A')
    code = code.replace('rk_qa[]', 'rk_qa[]').replace('rk_qb[]', 'rk_qb[]')
    code = re.sub(r'(?m)^const uint16_t (rk_q[ab]\[\])', r'constant ushort \1', code)
    # The CUDA slab-asm branch is disabled. The portable branch is a thread Fp array.
    code = code.replace('#define RK_LD1(p, w) (static_cast<const Fp>(Fp::fromRaw(reinterpret_cast<const uint32_t*>(p)[(size_t)(w) * RK_STRIDE])))',
                        '#define RK_LD1(p, w) (reinterpret_cast<const thread Fp*>(p)[(size_t)(w) * RK_STRIDE])')
    code = code.replace('#define RK_ST1(p, w, v) (reinterpret_cast<uint32_t*>(p)[(size_t)(w) * RK_STRIDE] = Fp(v).asRaw())',
                        '#define RK_ST1(p, w, v) (reinterpret_cast<thread Fp*>(p)[(size_t)(w) * RK_STRIDE] = Fp(v))')
    # The inactive CUDA PTX and slab-asm branches are removed by the MSL preprocessor.
    code = code.replace('const uint4 b = reinterpret_cast<const uint4*>(ef_pm_raw)[k], n = reinterpret_cast<const uint4*>(ef_pmn_raw)[k];',
                        'const FpExt pm = poly_mix[k];\n'
                        '  const uint4 b = uint4(pm.elems[0].asRaw(), pm.elems[1].asRaw(), pm.elems[2].asRaw(), pm.elems[3].asRaw());\n'
                        '  const uint4 n = uint4((pm.elems[1] * Fp(Fp::P - 11)).asRaw(), (pm.elems[2] * Fp(Fp::P - 11)).asRaw(), (pm.elems[3] * Fp(Fp::P - 11)).asRaw(), 0u);')
    assert 'ef_pm_raw' not in code and 'ef_pmn_raw' not in code
    # Helper calls gain the current function's poly_mix parameter through macros.
    helpers = ('rk_cl_acc', 'rk_cl0', 'rk_link_k', 'rk_link_nf_k')
    # Rename definitions only: generated calls below use the wrapper macros.
    for name in helpers:
        pattern = re.compile(r'(?m)^(.*?\b)' + name + r'(\([^\n]*?)(\) \{)')
        code, count = pattern.subn(lambda m: m[1] + name + '_impl' + m[2] + ', const device FpExt* poly_mix' + m[3], code)
        assert count >= 1, (name, count)
    code = code.replace('rk_cl_acc(w, u, k);', 'rk_cl_acc_impl(w, u, k, poly_mix);')
    code = code.replace('return rk_link_k(base, c, k);', 'return rk_link_k_impl(base, c, k, poly_mix);')
    macro_point = code.index('// ---- end helpers ----')
    macro_point = code.index('\n', macro_point) + 1
    macros = ''.join(f'#define {name}(a,b,c) {name}_impl(a,b,c,poly_mix)\n' for name in helpers)
    code = code[:macro_point] + macros + code[macro_point:]
    code = pointer_templates(code)
    # All generated chain calls carry the dynamic evalfast table.
    code = code.replace('FpExt* poly_mix', 'const device FpExt* poly_mix')
    code = code.replace('const device const device FpExt*', 'const device FpExt*')
    code = code.replace('const FpExt* poly_mix', 'const device FpExt* poly_mix')
    code = code.replace('const Fp* out, const Fp* mix, FpExt* tab',
                        'const device Fp* out, const device Fp* mix, device FpExt* tab')
    code = code.replace('const device FpExt* poly_mix, Fp** args',
                        'const device FpExt* poly_mix, const device Fp* accum, const device Fp* data, const device Fp* out, const device Fp* mix, thread Fp* slab')
    for i, name in enumerate(('accum', 'data', 'out', 'mix', 'slab')):
        code = code.replace(f'args[{i}]', name)
    code = code.replace('Fp* x33 = reinterpret_cast<Fp*>(slab);', 'thread Fp* x33 = slab;')
    code = code.replace('FpExt* x34 = reinterpret_cast<FpExt*>(slab);',
                        'thread FpExt* x34 = reinterpret_cast<thread FpExt*>(slab);')
    code = code.replace('Fp* rkBD = arg7; Fp* rkBA = arg8; Fp* rkBM = arg9;',
                        'const device Fp* rkBD = arg7; const device Fp* rkBA = arg8; const device Fp* rkBM = arg9;')
    code = code.replace('Fp* arg0, FpExt* arg6, Fp* arg7, Fp* arg8, Fp* arg9)',
                        'thread Fp* arg0, thread FpExt* arg6, const device Fp* arg7, const device Fp* arg8, const device Fp* arg9)')
    assert 'Fp** args' not in code and 'args[' not in code
    code = shared_macros(code)
    return shared_calls(code)


KERNELS = r'''
using namespace risc0::circuit::rv32im_v2;
kernel void ef_prep(const device FpExt* pm_in [[buffer(0)]],
                    const device Fp* out [[buffer(1)]], const device Fp* mix [[buffer(2)]],
                    device FpExt* tab [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  for (uint i = 0; i < EF_NPC; i++) tab[i] = pm_in[i];
  for (uint i = 0; i < RK_QN; i++) tab[EF_NPC + i] = pm_in[rk_qa[i]] * pm_in[rk_qb[i]];
  rk_hoist(tab, out, mix, tab + EF_NPC + RK_QN);
}
kernel void ef_eval_split(device Fp* check [[buffer(0)]],
                          const device Fp* data [[buffer(1)]],
                          const device Fp* accum [[buffer(2)]],
                          const device Fp* mix [[buffer(3)]],
                          const device Fp* out [[buffer(4)]],
                          const device FpExt* poly_mix [[buffer(5)]],
                          device Fp* scratch [[buffer(6)]],
                          constant uint& domain [[buffer(7)]],
                          uint tid [[thread_position_in_grid]],
                          uint ef_lane [[thread_index_in_threadgroup]]) {
  if (tid >= 2u * domain) return;
  const uint phase = tid >= domain;
  const uint cycle = tid - phase * domain;
  threadgroup uint ef_sm[EF_NHOT * EF_THREADS];
  Fp slab[RK_NWORDS];
  FpExt value;
  if (phase == 0u) value = poly_fp(cycle, domain, poly_mix, accum, data, out, mix, slab, ef_sm, ef_lane);
  else value = rk_accblk(cycle, domain, poly_mix, slab,
                         reinterpret_cast<thread FpExt*>(slab), data, accum, mix, ef_sm, ef_lane);
  device Fp* dst = phase == 0u ? check : scratch;
  for (uint i = 0; i < 4; i++) dst[i * domain + cycle] = value.elems[i];
}
kernel void ef_combine(device Fp* check [[buffer(0)]],
                       const device Fp* scratch [[buffer(1)]],
                       const device Fp* normalization [[buffer(2)]],
                       constant uint& domain [[buffer(3)]],
                       uint cycle [[thread_position_in_grid]]) {
  if (cycle >= domain) return;
  const Fp quotient = normalization[cycle & 3u];
  for (uint i = 0; i < 4; i++)
    check[i * domain + cycle] = (check[i * domain + cycle] + scratch[i * domain + cycle]) * quotient;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--input', type=Path,
                        help='precomputed p6 stage for diagnosis; otherwise regenerate in-tree')
    parser.add_argument('--reference-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    src = args.input.read_text() if args.input else regenerate(args.engine.resolve())
    assert 'rk_accblk' in src and 'ef_eval_split' in src and 'EF_NHOT 32' in src
    text = headers(args.engine.resolve())
    text += '#define EF_NPC 458u\n#define EF_NPM 627u\n#define EF_WIDE 1\n#define EF_THREADS 256u\n'
    text += '#define EF_LD(p, i) ((p)[(i)])\n'
    text += '#define EF_PM(i) (poly_mix[(i)])\n'
    text += transform(src)
    text += KERNELS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    report = {'source_sha256': sha(args.output),
              'input_sha256': hashlib.sha256(src.encode()).hexdigest(),
              'generator_sha256': sha(Path(__file__)), 'phase_split': True,
              'proof_eligible': False, 'cpu_oracle_checked': False,
              'entry_point': 'ef_prep',
              'entry_points': ['ef_prep', 'ef_eval_split', 'ef_combine'],
              'poly_mix_exponents': json.loads(args.reference_metadata.read_text())['poly_mix_exponents'],
              'evalfast_stage': 'through mksm2.py (32 words per phase) and mksub.py',
              'threadgroup_bytes': 32 * 256 * 4}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
