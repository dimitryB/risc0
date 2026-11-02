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

# Software prefetch of the trace loads in the ROCm eval_check: regen.py applies convert() last,
# after to_hip.py and wmma_hip.py.
#
# Every chain function (rv32im_v2_N, poly_fp, rk_accblk) is one straight-line block that reads
# ~200 trace words through EF_LDC, mostly right where they are used. At 192 VGPRs (EF_MINB 4)
# the AMDGPU scheduler keeps that order, so a load's latency (an L2 hit at best) is waited for a
# few instructions after its issue, and four waves per SIMD do not cover it. Here the body is
# cut into windows of `window` statement lines, and every load first used in window j becomes
# a raw uint32 value read at the start of window j - 1 (one value per (pointer, column, row
# offset) and function, the form CSE gave already); the uses read that value. The scheduler
# still owns the final order, but starts from loads issued 8-16 statements ahead of their
# uses. Measured on the ef harness at po2 20 (medians of interleaved runs, bit-exact): RX 7900
# XTX 186.3 -> 174.3 ms, RX 9070 XT 193.3 -> 182.4 ms (window 8; 4..32 and 2-window lags within
# 1 ms on gfx1100, window 8 best on gfx1201; batching each window's loads behind a non-volatile
# asm operand list was slower: 176-183 ms).
#
# usage: hoist_hip.py <in.cu> <out.cu> [--window N]
import re
import sys

FN = re.compile(r'^__device__ (EF_NOINLINE )?FpExt (rv32im_v2_\d+|poly_fp|rk_accblk)\(.*\) \{$')
LD = re.compile(r'EF_LDC\((\w+|args\[\d\]), (\d+)u, (\d+)u\)')


def _stmt_start(body, h):
    # a whole statement line that follows a complete one: a place to insert declarations
    l = body[h]
    return (l.startswith('  ') and not l.startswith('   ') and l.rstrip().endswith((';', ')'))
            and body[h - 1].rstrip().endswith((';', ')', '{')))


def convert(s, window=8, stats=None):
    lines = s.split('\n')
    out = []
    nfn = nld = 0
    i = 0
    while i < len(lines):
        if not FN.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        end = lines.index('}', i + 1)
        body = lines[i + 1:end]
        st = next(k for k, b in enumerate(body) if 'const uint32_t rk_s =' in b) + 1
        first = {}
        for k in range(st, len(body)):
            for m in LD.finditer(body[k]):
                first.setdefault(m.groups(), k)
        name = {key: 'efl_%s_%s_%s' % (re.sub(r'\W', '', key[0]), key[1], key[2]) for key in first}
        decl = {}
        for key, k in first.items():
            h = st + max(0, (k - st) // window - 1) * window
            while h > st and not _stmt_start(body, h):
                h -= 1
            decl.setdefault(h, []).append('  const uint32_t %s = EF_LDCR(%s, %su, %su);' % ((name[key],) + key))
        out.append(lines[i])
        for k, b in enumerate(body):
            out.extend(decl.get(k, []))
            out.append(LD.sub(lambda m: '::risc0::Fp::fromRaw(%s)' % name[m.groups()], b) if k >= st else b)
        out.append('}')
        nfn += 1
        nld += len(first)
        i = end + 1
    msg = 'hoist_hip: %d functions, %d trace loads issued one window ahead (window %d)' % (nfn, nld, window)
    if stats is not None:
        stats.append(msg)
    else:
        print(msg)
    return '\n'.join(out)


if __name__ == '__main__':
    args = sys.argv[1:]
    opts = {'--window': 8}
    while len(args) > 2:
        opts[args[-2]] = int(args[-1])
        args = args[:-2]
    with open(args[0], newline='') as f:
        text = f.read()
    text = convert(text, opts['--window'])
    with open(args[1], 'w', newline='') as f:
        f.write(text)
