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

# The hottest slab words of the ROCm eval_check in LDS on gfx11 (RDNA3): regen.py applies convert()
# after inline_hip.py. mksm2.py's scheme (CUDA) on the HIP kernel: the main-chain functions map their
# `nmain` most accessed slab words, the LogUp-chain functions their `nlu` most accessed words, onto one
# __shared__ array ef_sm[N][EF_THREADS] (a block runs one phase; [word][thread] keeps the 32 lanes of
# an LDS access on distinct banks); the other words stay in the scratch slab. With 32 words a
# 128-thread block takes 16 KB, so 8 blocks (4 waves per SIMD, the VGPR-bound occupancy) still fit a
# 128 KB WGP. gfx12 keeps the scratch slab at the default 24 WMMA staging slots (wmma_hip.py: 15 KB of
# the block's 16 KB share); with gfx12=True (regen.py's gfx12 variant: 12 slots, 16 words per phase) the
# LDS words are used there too. Other targets were not measured. Measured on the ef harness at po2 20
# (medians of interleaved runs, bit-exact), RX 7900 XTX: 142.7 -> 138.5 ms on the inlined kernel
# with buffer trace loads (160.5 -> 156.3 ms with the chain functions called; 16 words 158.1).
# With lu_alias=True (regen.py's gfx12 variant) ef_sm keeps `nmain` rows and the LogUp words past them
# live in wmma_hip.py's staging array ef_wlds, which LogUp blocks never use (15 words per thread at 12
# slots): 31 LogUp words for the LDS of 16 (RX 9070 XT, ef harness: -1.0 ms).
#
# usage: smem_hip.py <in.cu> <out.cu> [--nmain N] [--nlu N]
import collections
import re
import sys

FN = re.compile(r'^__device__ (?:EF_NOINLINE |EF_CHAIN )?FpExt (rv32im_v2_\d+|poly_fp|rk_accblk)\(.*\) \{$')
LU = {'rk_accblk', 'rv32im_v2_2', 'rv32im_v2_1', 'rv32im_v2_0'}
ANCHOR = '#define RK_ST4(p, w0, w1, w2, w3, v)'


def convert(s, nmain=32, nlu=32, gfx12=False, lu_alias=False, stats=None):
    lines = s.split('\n')
    cnt = {'m': collections.Counter(), 'l': collections.Counter()}
    nfn = 0
    i = 0
    while i < len(lines):
        m = FN.match(lines[i])
        if not m:
            i += 1
            continue
        end = lines.index('}', i + 1)
        g = 'l' if m.group(1) in LU else 'm'
        nfn += 1
        for k in range(i + 1, end):
            l = lines[k]
            for mm in re.finditer(r'RK_(?:LD|ST)4\(\w+, (\d+)u, (\d+)u, (\d+)u, (\d+)u', l):
                for w in mm.groups():
                    cnt[g][int(w)] += 1
            for mm in re.finditer(r'RK_(?:LD|ST)1\(\w+, (\d+)u', l):
                cnt[g][int(mm.group(1))] += 1
            lines[k] = re.sub(r'RK_(LD|ST)(4|1)\(', lambda q: 'RK_%s%s%s(' % (q.group(1), q.group(2), g.upper()), l)
        i = end + 1
    s = '\n'.join(lines)
    hot = {'M': [w for w, _ in cnt['m'].most_common(nmain)], 'L': [w for w, _ in cnt['l'].most_common(nlu)]}
    n = nmain if lu_alias else max(nmain, nlu, 1)
    out = ['// smem_hip.py: on gfx11%s the hottest slab words of each phase in LDS (ef_sm), the rest in the'
           % (' and gfx12' if gfx12 else ''),
           '// scratch slab; other targets use the slab for every word.',
           '#if defined(__HIP_DEVICE_COMPILE__) && %s' % ('(defined(__GFX11__) || defined(__GFX12__))' if gfx12
                                                         else 'defined(__GFX11__)'),
           '#define EF_NHOT %du' % n,
           '__shared__ uint32_t ef_sm[EF_NHOT][EF_THREADS];',
           '#define EF_SMW(k) ef_sm[k][threadIdx.x]']
    if lu_alias:
        out += ['// LogUp words past the EF_NHOT rows: in ef_wlds (wmma_hip.py), unused by LogUp blocks',
                'static_assert(%du <= EF_NHOT + sizeof(ef_wlds) / sizeof(ef_wlds[0]) / EF_THREADS, "LogUp alias words");'
                % len(hot['L']),
                '#define EF_SMWL(k) ((k) < EF_NHOT ? ef_sm[(k) < EF_NHOT ? (k) : 0][threadIdx.x] : '
                'ef_wlds[((k) < EF_NHOT ? 0 : (k) - EF_NHOT) * EF_THREADS + threadIdx.x])']
    for g in ('M', 'L'):
        out.append('__device__ __forceinline__ constexpr int ef_hot%s(uint32_t w) { switch (w) { %s default: return -1; } }'
                   % (g, ' '.join('case %du: return %d;' % (w, k) for k, w in enumerate(hot[g]))))
        smw = 'EF_SMWL' if lu_alias and g == 'L' else 'EF_SMW'
        out.append('#define RK_LD1%s(p, w) (static_cast<const ::risc0::Fp>(::risc0::Fp::fromRaw(ef_hot%s(w) >= 0 ? '
                   '%s(ef_hot%s(w) < 0 ? 0 : ef_hot%s(w)) : ef_ldl((p), (w)))))' % (g, g, smw, g, g))
        out.append('#define RK_ST1%s(p, w, v) do { if (ef_hot%s(w) >= 0) %s(ef_hot%s(w) < 0 ? 0 : ef_hot%s(w)) = '
                   '::risc0::Fp(v).asRaw(); else ef_stl((p), (w), ::risc0::Fp(v).asRaw()); } while (0)' % (g, g, smw, g, g))
    out.append('#else')
    for g in ('M', 'L'):
        out.append('#define RK_LD1%s RK_LD1' % g)
        out.append('#define RK_ST1%s RK_ST1' % g)
    out.append('#endif')
    for g in ('M', 'L'):
        out.append('#define RK_LD4%s(p, w0, w1, w2, w3) (static_cast<const ::risc0::FpExt>(::risc0::FpExt(RK_LD1%s(p, w0), '
                   'RK_LD1%s(p, w1), RK_LD1%s(p, w2), RK_LD1%s(p, w3))))' % (g, g, g, g, g))
        out.append('#define RK_ST4%s(p, w0, w1, w2, w3, v) do { ::risc0::FpExt _rk_v = (v); RK_ST1%s(p, w0, _rk_v.elems[0]); '
                   'RK_ST1%s(p, w1, _rk_v.elems[1]); RK_ST1%s(p, w2, _rk_v.elems[2]); RK_ST1%s(p, w3, _rk_v.elems[3]); } '
                   'while (0)' % (g, g, g, g, g))
    if s.count(ANCHOR) != 1:
        sys.exit('smem_hip: expected 1 RK_ST4 definition, found %d' % s.count(ANCHOR))
    k = s.index('\n', s.index(ANCHOR)) + 1
    s = s[:k] + '\n'.join(out) + '\n' + s[k:]
    msg = ('smem_hip: %d functions; gfx11 LDS words: main %d of %d (%d of %d accesses), LogUp %d of %d (%d of %d)'
           % (nfn, len(hot['M']), len(cnt['m']), sum(cnt['m'][w] for w in hot['M']), sum(cnt['m'].values()),
              len(hot['L']), len(cnt['l']), sum(cnt['l'][w] for w in hot['L']), sum(cnt['l'].values())))
    if stats is not None:
        stats.append(msg)
    else:
        print(msg)
    return s


if __name__ == '__main__':
    args = sys.argv[1:]
    opts = {'--nmain': 32, '--nlu': 32}
    while len(args) > 2:
        opts[args[-2]] = int(args[-1])
        args = args[:-2]
    with open(args[0], newline='') as f:
        text = f.read()
    text = convert(text, opts['--nmain'], opts['--nlu'])
    with open(args[1], 'w', newline='') as f:
        f.write(text)
