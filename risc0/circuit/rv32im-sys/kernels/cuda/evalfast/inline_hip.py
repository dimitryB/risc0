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

# Every chain function of the ROCm eval_check inlined into ef_eval_split: regen.py applies
# convert() last, after args_hip.py.
#
# The chain functions (rv32im_v2_N, rk_accblk) were EF_NOINLINE: each phase is one call chain
# (poly_fp -> rv32im_v2_19 -> ... -> rv32im_v2_3, rk_accblk -> rv32im_v2_2 -> ... -> rv32im_v2_0),
# mostly tail calls. Every callee uses all 192 VGPRs, so under the AMDGPU calling convention it
# saved and restored the 80 callee-saved VGPRs (v40-v47, v56-v63, ..., v184-v191) and 41
# callee-saved SGPRs (through VGPR lanes) around its body, whatever its caller kept there; the
# five calls whose callee takes more stack arguments than the caller received could not be tail
# calls and nested their frames. On gfx1100 that was 2612 scratch loads and 2333 scratch stores
# per main-phase wave next to 3784 trace loads, and the scratch traffic competed with the trace
# columns' re-reads (which later chain functions repeat) for the L2 and the Infinity Cache: trace
# loads that do not allocate there (nontemporal / DLC) cost +10% on gfx1100 and +17% on gfx1201.
# HIP has no calling convention without callee-saved registers, so the chain is inlined instead
# (ef_eval_split's scratch: 8556 -> ~2.5 KB per lane on gfx1100). Each function's row index goes
# through an empty asm statement (EF_OPQ) so its trace loads are not merged with the same
# column's loads in other functions: CSE across the inlined bodies would hold those values in
# VGPRs for the rest of the chain. Measured on the ef harness at po2 20 (medians of interleaved
# runs, bit-exact): RX 7900 XTX 160.4 -> 150.6 ms, RX 9070 XT 168.7 -> 141.3 ms (inlined
# without EF_OPQ: 143.7 ms; inlining only some of the functions was slower on gfx1100). The
# compile takes several minutes per architecture longer.
#
# usage: inline_hip.py <in.cu> <out.cu>
import re
import sys

DEF = re.compile(r'^__device__ EF_NOINLINE FpExt (rv32im_v2_\d+|rk_accblk)\(', re.M)
ROW = ('const uint32_t rk_s = EF_UNI32((uint32_t)steps), rk_m = rk_s - 1u, '
       'rk_c = (uint32_t)cycle;')
OPQ_DEF = '''// inline_hip.py: the chain functions are inlined into ef_eval_split (EF_CHAIN), each with its own
// opaque row index (EF_OPQ: an empty asm statement, unique per function) so that loads of the same
// trace word in different functions are not merged.
#define EF_CHAIN __attribute__((always_inline))
#if defined(__HIP_DEVICE_COMPILE__)
#define EF_OPQ(x, n) asm("; ef_row " #n : "+v"(x))
#else
#define EF_OPQ(x, n)
#endif
'''


def _sub(s, old, new, count):
    n = s.count(old)
    if n != count:
        sys.exit('inline_hip: expected %d of %r, found %d' % (count, old[:80], n))
    return s.replace(old, new)


def convert(s, stats=None):
    s, nfn = DEF.subn(lambda m: '__device__ EF_CHAIN FpExt %s(' % m.group(1), s)
    if nfn != 21:
        sys.exit('inline_hip: expected 21 chain functions, found %d' % nfn)
    n = [0]

    def row(m):
        n[0] += 1
        return ('uint32_t rk_c_o = (uint32_t)cycle; EF_OPQ(rk_c_o, %d); const uint32_t rk_s = '
                'EF_UNI32((uint32_t)steps), rk_m = rk_s - 1u, rk_c = rk_c_o;' % n[0])

    s = re.sub(re.escape(ROW), row, s)
    if n[0] != 22:
        sys.exit('inline_hip: expected 22 row indices, found %d' % n[0])
    s = _sub(s, 'namespace risc0::circuit::rv32im_v2 {\n', OPQ_DEF + 'namespace risc0::circuit::rv32im_v2 {\n', 1)
    msg = 'inline_hip: %d chain functions inlined, %d opaque row indices' % (nfn, n[0])
    if stats is not None:
        stats.append(msg)
    else:
        print(msg)
    return s


if __name__ == '__main__':
    with open(sys.argv[1], newline='') as f:
        text = f.read()
    text = convert(text)
    with open(sys.argv[2], 'w', newline='') as f:
        f.write(text)
