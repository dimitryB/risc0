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

# Opaque lazy 64-bit accumulators in the ROCm eval_check on gfx11 (regen.py: both variants, after
# pmlag_hip.py; the macro is empty on other targets, so both variants carry the same text).
#
# A lazy 64-bit multiply-accumulate t = ((hi << 32) | lo) + x * y lets LLVM reassociate it into
# (x * y + lo) + (hi << 32): v_mad_u64_u32 with a zero-extended lo, then a 64-bit add of the high
# word (v_add_co_u32 of 0 + v_add_co_ci_u32, plus v_mov_b32 zeroing the mad's high addend):
# ~7.3k v_add_co pairs per main-phase wave on gfx1100. EF_OPQ64 passes each result through an empty
# asm (VGPR pair in and out, no instruction), which keeps one v_mad_u64_u32 per product. Measured
# on the ef harness at po2 20 (interleaved medians, bit-exact): see regen.py.

MARK = '// ---- lincomb.py: lazily reduced FpExt linear combinations ----\n'
DEFINE = ('// opq_hip.py: on gfx11 a lazy 64-bit accumulator LLVM may not reassociate (an empty asm, no instruction).\n'
          '#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__)\n'
          '#define EF_OPQ64(t) asm("" : "+v"(t))\n'
          '#else\n'
          '#define EF_OPQ64(t)\n'
          '#endif\n')
SITES = [
    ('  uint64_t t = ((uint64_t(w.hi[j]) << 32) | w.lo[j]) + uint64_t(x) * uint64_t(y);\n',
     '  uint64_t t = ((uint64_t(w.hi[j]) << 32) | w.lo[j]) + uint64_t(x) * uint64_t(y);\n  EF_OPQ64(t);\n', 1),
    ('  uint64_t t = ((uint64_t(w.hi) << 32) | w.lo) + uint64_t(x) * uint64_t(y);\n',
     '  uint64_t t = ((uint64_t(w.hi) << 32) | w.lo) + uint64_t(x) * uint64_t(y);\n  EF_OPQ64(t);\n', 1),
    ('    uint64_t t = ((uint64_t(s.hi[j]) << 32) | s.lo[j]) + uint64_t(cr) * uint64_t(pm.elems[j].asRaw());\n',
     '    uint64_t t = ((uint64_t(s.hi[j]) << 32) | s.lo[j]) + uint64_t(cr) * uint64_t(pm.elems[j].asRaw());\n    EF_OPQ64(t);\n', 2),
    ('  return c + uint64_t(a) * uint64_t(b);\n#endif\n}\n',
     '  uint64_t d = c + uint64_t(a) * uint64_t(b);\n  EF_OPQ64(d);\n  return d;\n#endif\n}\n', 1),
]


def convert(s, stats=None):
    assert s.count(MARK) == 1
    s = s.replace(MARK, DEFINE + MARK)
    n = 0
    for old, new, cnt in SITES:
        assert s.count(old) == cnt, (old[:70], s.count(old))
        s = s.replace(old, new)
        n += cnt
    msg = 'opq_hip: EF_OPQ64 at %d multiply-accumulate sites (gfx11)' % n
    if stats is not None:
        stats.append(msg)
    else:
        print(msg)
    return s
