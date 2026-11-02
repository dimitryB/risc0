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

"""Regenerate kernels/cuda/eval_check_ef.cu (CUDA) and eval_check_ef_hip.cu (ROCm)
from the generated circuit.

usage: regen.py [--check]

The CUDA eval_check kernel is the Intel evalfast pipeline output converted by
to_cuda.py. The chain, the same one build.rs runs for the Intel mono kernel
(with RISC0_INTEL_EVAL_XMX=0, as XMX is Intel-only):

  1. kernels/intel/poly_fp_cse.py kernels/cxx --top 16 --min-share 10
  2. the amalgamation build.rs writes as intel_eval_check_amalg.cpp
  3. kernels/intel/evalfast/pipeline.py
  4. kernels/cuda/evalfast/to_cuda.py

followed by the CUDA post-passes in this directory (PASSES below):

  5. remat.py: slab values that are small Fp trees over input loads are
     recomputed at their reads; the slab is re-coloured
  6. lazyfp.py: lazy [0, 2P) Fp add/sub/mul where every consumer allows it
  7. close.py: closures B + u * PM_k accumulated lazily in 64 bits with an
     NBETA-folded constant table (+ inline PTX mad.wide, CUDA only)
  8. linkext.py: rk_link with an FpExt term accumulated the same way
  9. accfirst_cuda.py: the LogUp/accum block sliced out of the main chain
 10. mksplit2.py: main / LogUp block-type phase split (ef_eval_split, even
     blocks main and odd blocks LogUp, plus ef_combine)
 11. remat_lu.py: LogUp slab FpExt values recomputed at their reads
 12. mksm2.py: per-phase 64 hottest slab words in shared memory
 13. mksub.py: every chain function split in two at its narrowest point
     between 45% and 55% of its body (the tail is a __noinline__ template)

Each pass prints what it did; EXPECT pins those lines, so a change in the
circuit or a pass that changes the shape of the kernel fails here first: re-run
the bit-exactness battery on the new kernel before updating the pins.

The ROCm kernel, eval_check_ef_hip.cu, is the chain's output after
HIP_LAST (remat_lu.py) in to_hip.py's HIP forms: mksm2.py's shared-memory
words cap the occupancy on RDNA, and mksub.py's splits only add calls there.
wmma_hip.py then moves the poly_mix chains onto the matrix units for gfx12
(RDNA4) devices only; other targets keep the scalar statements. hoist_hip.py
issues the trace loads one window of statements ahead of their uses, and
args_hip.py narrows the chain-function calls (32-bit cycle/steps, the kernel
pointers read from a device table instead of passed in VGPRs); inline_hip.py
then inlines the chain functions into the kernel (no callee-saved register
saves at every call) and smem_hip.py moves each phase's hottest slab words to
LDS on gfx11.
wmma_hip.py through smem_hip.py run twice: the default (24 WMMA staging slots
per wave, 32 LDS words per phase) and, for gfx12, 12 slots and 16 LDS words
(smem_hip.py's LDS words on gfx12 too); pmlag_hip.py then lags the scalar
links' constant loads (and the FpExt-term links' and chained closures' constant
rows) in both: in the default (gfx11) every scalar link, in the gfx12 variant the
plain links outside the WMMA chains. archsel_hip.py merges the outputs,
the runs of lines that differ under #if defined(__GFX12__), and checks that
each side selects back its input.

The results are committed, so the GPU builds need no Python. Re-run this after
changing the generated circuit (kernels/cxx/rust_poly_fp_*.cpp) or any of the
passes; --check (run in CI) fails if a committed file is out of date.
"""

import difflib
import os
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archsel_hip  # noqa: E402
import args_hip  # noqa: E402
import hoist_hip  # noqa: E402
import inline_hip  # noqa: E402
import opq_hip  # noqa: E402
import pmlag_hip  # noqa: E402
import smem_hip  # noqa: E402
import to_hip  # noqa: E402
import wmma_hip  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SYS = os.path.normpath(os.path.join(HERE, '..', '..', '..'))  # rv32im-sys
OUT = os.path.join(SYS, 'kernels', 'cuda', 'eval_check_ef.cu')
OUT_HIP = os.path.join(SYS, 'kernels', 'cuda', 'eval_check_ef_hip.cu')
HIP_LAST = 'remat_lu.py'
# wmma_hip.py (after to_hip.py): the poly_mix chains on the gfx12 matrix units.
WMMA_EXPECT = ('wmma_hip: 136 of 282 chains, 4672 of 4948 Fp links on the matrix units (wmin 6, 1 x 24 '
               'slots, group 48), 0 chains without a free region')
# hoist_hip.py (after wmma_hip.py): the trace loads issued one window of statements ahead of their uses.
HOIST_EXPECT = 'hoist_hip: 22 functions, 3844 trace loads issued one window ahead (window 8)'
# args_hip.py (last): 32-bit cycle/steps; the kernel pointers from ef_ptrs instead of call arguments.
ARGS_EXPECT = 'args_hip: 32-bit cycle/steps, 101 signatures and 20 calls without 95 pointer arguments'
# inline_hip.py (last): the chain functions inlined into ef_eval_split, one opaque row index each.
INLINE_EXPECT = 'inline_hip: 21 chain functions inlined, 22 opaque row indices'
# smem_hip.py (last): on gfx11 each phase's 32 most accessed slab words in LDS.
SMEM_EXPECT = ('smem_hip: 22 functions; gfx11 LDS words: main 32 of 85 (539 of 751 accesses), LogUp 32 of 104 '
               '(96 of 268)')
# pmlag_hip.py (last): the scalar links' constants loaded <= 4 links ahead, the FpExt-term links' and
# chained closures' constant rows not before the accumulator they depend on; in the gfx12 variant the
# plain links (outside the WMMA chains) only.
PMLAG_EXPECT = ('pmlag_hip: 22 functions, 5034 links and 157 closures with lagged constants (W 4)')
PMLAG_GFX12_EXPECT = ('pmlag_hip: 22 functions, 364 plain links and 157 closures with lagged constants (W 4)')
# opq_hip.py (both variants; the barriers are empty outside gfx11)
OPQ_EXPECT = 'opq_hip: EF_OPQ64 at 5 multiply-accumulate sites (gfx11)'
# The gfx12 variant (archsel_hip.py): 12 WMMA staging slots per wave (7.5 KB of LDS per block) and
# smem_hip.py's LDS words on gfx12 too, 16 per phase (8 KB).
WMMA_GFX12_SLOTS = 12
SMEM_GFX12_WORDS = 16
# ... and 31 LogUp words: the 15 past ef_sm's 16 rows in the WMMA staging array (unused by LogUp blocks).
SMEM_GFX12_LU_WORDS = 31
WMMA_GFX12_EXPECT = ('wmma_hip: 136 of 282 chains, 4672 of 4948 Fp links on the matrix units (wmin 6, 1 x 12 slots, group 60), 0 chains without a free region')
HOIST_GFX12_EXPECT = HOIST_EXPECT
SMEM_GFX12_EXPECT = ('smem_hip: 22 functions; gfx11 LDS words: main 16 of 85 (334 of 751 accesses), LogUp 31 of 104 (93 of 268)')
ARCHSEL_EXPECT = ('archsel_hip: 3496 runs of 9531 lines selected by __GFX12__')
NS = 'namespace risc0::circuit::rv32im_v2 {'

# mksub.py cut points: function, body lines, cut line, live values passed to the tail (+ input
# loads re-issued there instead).
MKSUB = [
    'rv32im_v2_0: 1329 lines, cut at 640 (0.48), live 28 (+90 reloaded)',
    'rv32im_v2_1: 1238 lines, cut at 592 (0.48), live 32 (+69 reloaded)',
    'rv32im_v2_2: 1131 lines, cut at 528 (0.47), live 35 (+57 reloaded)',
    'rv32im_v2_3: 999 lines, cut at 452 (0.45), live 8 (+23 reloaded)',
    'rv32im_v2_4: 1322 lines, cut at 594 (0.45), live 76 (+2 reloaded)',
    'rv32im_v2_5: 1283 lines, cut at 581 (0.45), live 15 (+0 reloaded)',
    'rv32im_v2_6: 1132 lines, cut at 509 (0.45), live 85 (+40 reloaded)',
    'rv32im_v2_7: 1335 lines, cut at 687 (0.51), live 29 (+10 reloaded)',
    'rv32im_v2_8: 1023 lines, cut at 481 (0.47), live 35 (+1 reloaded)',
    'rv32im_v2_9: 1077 lines, cut at 486 (0.45), live 35 (+7 reloaded)',
    'rv32im_v2_10: 1378 lines, cut at 620 (0.45), live 12 (+36 reloaded)',
    'rv32im_v2_11: 1088 lines, cut at 489 (0.45), live 102 (+43 reloaded)',
    'rv32im_v2_12: 1222 lines, cut at 642 (0.53), live 65 (+25 reloaded)',
    'rv32im_v2_13: 1250 lines, cut at 591 (0.47), live 13 (+51 reloaded)',
    'rv32im_v2_14: 1240 lines, cut at 672 (0.54), live 43 (+83 reloaded)',
    'rv32im_v2_15: 1280 lines, cut at 610 (0.48), live 67 (+57 reloaded)',
    'rv32im_v2_16: 998 lines, cut at 541 (0.54), live 15 (+24 reloaded)',
    'rv32im_v2_17: 1186 lines, cut at 533 (0.45), live 46 (+18 reloaded)',
    'rv32im_v2_18: 1038 lines, cut at 504 (0.49), live 19 (+13 reloaded)',
    'rv32im_v2_19: 1015 lines, cut at 465 (0.46), live 112 (+25 reloaded)',
]

# (pass, extra arguments, lines its output must contain)
PASSES = [
    ('remat.py', ['--maxops', '4', '--maxloads', '4'],
     ['remat: 675 values, 1421 loads rematerialized, 675 stores dropped, slab words 324']),
    ('lazyfp.py', [], ["lazyfp: {'cand': 6032, 'add': 94, 'sub': 503, 'mul': 265}"]),
    ('close.py', ['--chain'], ['close: 231 closures, 157 chained']),
    ('linkext.py', [], ['linkext: 5118 call sites']),
    ('accfirst_cuda.py', [], ['accfirst_cuda: 209 tail lines, 41 pulled-in defs']),
    ('mksplit2.py', [], []),
    ('remat_lu.py', ['--maxg', '4', '--maxops', '8'],
     ["loads 135 {'keep': 41, 'remat': 94} stores 115 dropped 89"]),
    ('mksm2.py', ['64', '64'], ['main words 85 acc 751 hot 683 | lu words 104 acc 268 hot 188']),
    ('mksub.py', ['--fns', ','.join('rv32im_v2_%d' % i for i in range(20)), '--lo', '0.45', '--hi', '0.55'],
     MKSUB),
]

POST_HEADER = '''//
// CUDA post-passes (kernels/cuda/evalfast, chained by regen.py): slab values
// rematerialised from the inputs (remat.py), lazy [0, 2P) Fp results
// (lazyfp.py), lazy 64-bit closure and FpExt link accumulation with inline PTX
// mad.wide (close.py, linkext.py), the LogUp block sliced out of the main chain
// (accfirst_cuda.py), a main / LogUp block-type phase split (ef_eval_split +
// ef_combine, mksplit2.py), LogUp slab values rematerialised at their reads
// (remat_lu.py), the per-phase 64 hottest slab words in shared memory
// (mksm2.py) and every chain function split in two at its narrowest point
// (mksub.py). The rematerialisation relies on data, accum and mix being
// read-only for the whole kernel.
'''
HIP_POST_HEADER = '''//
// ROCm build (kernels/cuda/evalfast/regen.py): the CUDA post-passes through
// remat_lu.py (remat, lazyfp, close and linkext with the C++ 64-bit
// multiply-adds, accfirst_cuda, the mksplit2 phase split, remat_lu), without
// mksm2.py's shared-memory slab words, which cap the occupancy on RDNA, or
// mksub.py's splits, in to_hip.py's HIP forms, with ef_fpext.h's lazy FpExt
// multiply. On gfx12 (RDNA4) the poly_mix chains run on the matrix units
// (wmma_hip.py: int8 WMMAs over LDS-staged constraint values, bit-exact; other
// targets compile the scalar chains), then the trace loads issued one window of
// statements ahead (hoist_hip.py), narrower chain-function calls (args_hip.py),
// every chain function inlined into the kernel (inline_hip.py), with the trace
// loads through per-32-column buffer resources (to_hip.py) and, on gfx11, the
// hottest slab words in LDS (smem_hip.py). gfx12 compiles its own variant of
// the lines that differ (archsel_hip.py): 12 WMMA staging slots per wave and
// 16 LDS slab words per phase (~134 instead of ~137 ms on an RX 9070 XT), 15
// more LogUp words in the WMMA staging area, which LogUp blocks do not use
// (smem_hip.py), the WMMA A operands through an opaque SGPR base (wmma_hip.py)
// and the poly_mix constants of the plain chain links, the FpExt-term links and
// the closures loaded no earlier than pmlag_hip.py allows (~130.8 ms); other
// targets, and the host pass, read the default: 24 slots, 32 words on gfx11,
// where each scalar chain link's poly_mix constant is loaded at most 4 links
// ahead (pmlag_hip.py: ~123.5 instead of ~136 ms on an RX 7900 XTX, no
// v_readlane-restored SGPR spills; FpExt-term links and closures too, ~2 ms
// less), the lazy 64-bit accumulators are not reassociated (opq_hip.py, ~5 ms
// less) and the kernel runs 3 blocks per SIMD (build.rs: EF_MINB_GFX11, ~115.7
// ms before pmlag_hip.py's closures and opq_hip.py). At po2 20 with 64-wide waves and EF_MINB
// 4 (build.rs): ~139 ms on an RX 7900 XTX (gfx1100) and ~137 ms on an RX 9070 XT
// (gfx1201), against ~156 and 141 ms inlined with global loads and the scratch
// slab, 160 and 169 ms with the chain functions called (callee-saved register
// saves around every call), 217 and 236 ms
// before the uniform addressing, hoist_hip, args_hip and the shift-add / v_min
// reductions, and 326 and 454 ms for the first ROCm kernel (the generator's output
// before its CUDA post-passes). The rematerialisation relies on data, accum and
// mix being read-only for the whole kernel.
'''
HEADER_END = '// kernel happens to tolerate them.\n'


def run(env, *args):
    r = subprocess.run([sys.executable, *args], env=env, cwd=SYS, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit('regen: %s failed (%d):\n%s%s' % (os.path.basename(args[0]), r.returncode,
                                                   r.stdout[-2000:], r.stderr[-2000:]))
    return r.stdout + r.stderr


def rust_lines(s):
    # str::lines(): split on '\n', strip one trailing '\r', no final empty line.
    lines = s.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    return [l[:-1] if l.endswith('\r') else l for l in lines]


def amalgamate(poly_fp_dir):
    # Must match build.rs (build_intel_kernels, "intel_eval_check_amalg.cpp")
    # without RISC0_LSC_HINTS.
    out = ['// Auto-generated amalgamation for SYCL device code\n',
           '#include "fp.h"\n', '#include "fpext.h"\n', '#include <cstdint>\n',
           NS + '\n', 'constexpr size_t kInvRate = 4;\n']
    for i in range(4):
        with open(os.path.join(poly_fp_dir, 'rust_poly_fp_%d.cpp' % i), newline='') as f:
            src = f.read()
        start = src.index(NS) + len(NS)
        for line in rust_lines(src[start:src.rindex('}')]):
            if line.startswith('FpExt rv32im_v2_') and '{' in line:
                out.append('__attribute__((noinline)) ')
            out.append(line + '\n')
    out.append('} // namespace risc0::circuit::rv32im_v2\n')
    with open(os.path.join(SYS, 'kernels', 'intel', 'eval_check.cpp'), newline='') as f:
        out.append(f.read())
    return ''.join(out)


def main():
    check = '--check' in sys.argv[1:]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('RISC0_', 'SINK_LEAD', 'PYTHON'))}
    env.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', RISC0_INTEL_EVAL_XMX='0')
    with tempfile.TemporaryDirectory() as tmp:
        cse = os.path.join(tmp, 'cse')
        os.makedirs(cse)
        run(env, 'kernels/intel/poly_fp_cse.py', 'kernels/cxx', cse,
            '--top', '16', '--min-share', '10')
        amalg = os.path.join(tmp, 'amalg.cpp')
        with open(amalg, 'w', newline='') as f:
            f.write(amalgamate(cse))
        fast = os.path.join(tmp, 'fast.cpp')
        run(env, 'kernels/intel/evalfast/pipeline.py', amalg, fast, os.path.join(tmp, 'work'))
        gen = os.path.join(tmp, 'base.cu')
        run(env, os.path.join(HERE, 'to_cuda.py'), fast, gen)
        for n, (tool, extra, expect) in enumerate(PASSES):
            nxt = os.path.join(tmp, 'p%d.cu' % n)
            log = run(env, os.path.join(HERE, tool), gen, nxt, *extra)
            for line in expect:
                if line not in log:
                    sys.exit('regen: EXPECT failed: %s did not print\n  %s\nit printed:\n%s'
                             % (tool, line, log))
            gen = nxt
            if tool == HIP_LAST:
                with open(gen, newline='') as f:
                    hip = f.read()
        with open(gen, newline='') as f:
            new = f.read()
    outputs = []

    def chain(hip, slots, wmma_expect, nhot, nlu, hoist_expect, smem_expect, plain, lag_expect):
        wstats = []
        hip = wmma_hip.convert(hip, slots=slots, stats=wstats)
        if wstats != [wmma_expect]:
            sys.exit('regen: EXPECT failed: wmma_hip.py printed\n  %s\nnot\n  %s' % (wstats, wmma_expect))
        hstats = []
        hip = hoist_hip.convert(hip, stats=hstats)
        if hstats != [hoist_expect]:
            sys.exit('regen: EXPECT failed: hoist_hip.py printed\n  %s\nnot\n  %s' % (hstats, hoist_expect))
        astats = []
        hip = args_hip.convert(hip, stats=astats)
        if astats != [ARGS_EXPECT]:
            sys.exit('regen: EXPECT failed: args_hip.py printed\n  %s\nnot\n  %s' % (astats, ARGS_EXPECT))
        istats = []
        hip = inline_hip.convert(hip, stats=istats)
        if istats != [INLINE_EXPECT]:
            sys.exit('regen: EXPECT failed: inline_hip.py printed\n  %s\nnot\n  %s' % (istats, INLINE_EXPECT))
        sstats = []
        hip = smem_hip.convert(hip, nmain=nhot, nlu=nlu, gfx12=True, lu_alias=nlu > nhot, stats=sstats)
        if sstats != [smem_expect]:
            sys.exit('regen: EXPECT failed: smem_hip.py printed\n  %s\nnot\n  %s' % (sstats, smem_expect))
        pstats = []
        hip = pmlag_hip.convert(hip, plain=plain, stats=pstats)
        if pstats != [lag_expect]:
            sys.exit('regen: EXPECT failed: pmlag_hip.py printed\n  %s\nnot\n  %s' % (pstats, lag_expect))
        ostats = []
        hip = opq_hip.convert(hip, stats=ostats)
        if ostats != [OPQ_EXPECT]:
            sys.exit('regen: EXPECT failed: opq_hip.py printed\n  %s\nnot\n  %s' % (ostats, OPQ_EXPECT))
        return hip
    hip = to_hip.convert(hip)
    # gfx12 (RDNA4) devices compile the variant, every other target the default (archsel_hip.py).
    default = chain(hip, 24, WMMA_EXPECT, 32, 32, HOIST_EXPECT, SMEM_EXPECT, False, PMLAG_EXPECT)
    variant = chain(hip, WMMA_GFX12_SLOTS, WMMA_GFX12_EXPECT, SMEM_GFX12_WORDS, SMEM_GFX12_LU_WORDS,
                    HOIST_GFX12_EXPECT, SMEM_GFX12_EXPECT, True, PMLAG_GFX12_EXPECT)
    mstats = []
    hip = archsel_hip.merge(default, variant, '__GFX12__', stats=mstats)
    if mstats != [ARCHSEL_EXPECT]:
        sys.exit('regen: EXPECT failed: archsel_hip.py printed\n  %s\nnot\n  %s' % (mstats, ARCHSEL_EXPECT))
    for path, text, post in ((OUT, new, POST_HEADER), (OUT_HIP, hip, HIP_POST_HEADER)):
        assert text.count(HEADER_END) == 1
        text = text.replace(HEADER_END, HEADER_END + post)
        bad = [i + 1 for i, l in enumerate(text.split('\n')) if l != l.rstrip()]
        if bad:
            sys.exit('regen: trailing whitespace on %d generated lines of %s (first: %d)'
                     % (len(bad), os.path.basename(path), bad[0]))
        outputs.append((path, text))
    stale = []
    for path, text in outputs:
        name = 'kernels/cuda/' + os.path.basename(path)
        try:
            with open(path, newline='') as f:
                old = f.read()
        except FileNotFoundError:
            old = ''
        if check:
            if old != text:
                diff = difflib.unified_diff(old.splitlines(True), text.splitlines(True),
                                            'committed/' + name, 'regenerated/' + name)
                sys.stdout.writelines(list(diff)[:200])
                stale.append(name)
            else:
                print('regen: %s is up to date' % name)
        else:
            with open(path, 'w', newline='') as f:
                f.write(text)
            print('regen: wrote %s' % path)
    if stale:
        sys.exit('regen: %s out of date; run kernels/cuda/evalfast/regen.py and commit the result'
                 % ' and '.join(stale))


if __name__ == '__main__':
    main()
