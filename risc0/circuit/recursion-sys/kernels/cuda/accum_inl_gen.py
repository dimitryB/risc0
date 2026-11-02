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

# usage: accum_inl_gen.py [--check]
#
# Post-processes the generated recursion accum steps, step_compute_accum.cu and
# step_verify_accum.cu (next to this script), in place, and derives the
# force-inlined copies that accum_inl.cu includes (CUDA only):
#
# 1. branch-free asserts: every `assert(E);` becomes `c1_bad |= !(E);` (and
#    `assert(x != Fp::invalid());` becomes `c1_bad |= (x == Fp::invalid());`),
#    with one trap at the end of the function if any check failed. The checks
#    and the values they read are unchanged; only where a failure stops the
#    kernel moves (one branch per cycle instead of one per assert).
#    HIP has no __trap(): extern.cuh, which both the .cu files and their
#    _inl.cuh copies include, maps it to __builtin_trap() under __HIPCC__.
# 2. step_{compute,verify}_accum_inl.cuh: the same function renamed *_inl and
#    made `static __device__ __forceinline__`.
#
# Re-run it after regenerating the accum steps; the rewrite is idempotent, so
# --check (run in CI) fails if any of the four files is out of date.
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = '--check' in sys.argv[1:]
DECL = '  uint32_t c1_bad = 0;'
TRAP = ['  if (c1_bad) __trap();']
INVALID = re.compile(r'^(\s*)assert\((\w+) != Fp::invalid\(\)\);$')
ASSERT = re.compile(r'^(\s*)assert\((.*)\);$')


def asserts(src, fn):
    lines = src.split('\n')
    start = lines.index('__device__ void %s(' % fn)
    end = len(lines) - 1
    while lines[end] != '}':
        end -= 1
    head, body, tail = lines[:start], lines[start:end], lines[end:]
    # an already-transformed body: drop the trap so it is re-emitted below
    body = [l for l in body if l not in TRAP]
    if DECL not in body:
        k = body.index('  Fp extern_outs[32];')
        body.insert(k + 1, DECL)
    n = 0
    for i, l in enumerate(body):
        m = INVALID.match(l)
        if m:
            body[i] = '%sc1_bad |= (%s == Fp::invalid());' % (m.group(1), m.group(2))
            n += 1
            continue
        m = ASSERT.match(l)
        if m:
            body[i] = '%sc1_bad |= !(%s);' % (m.group(1), m.group(2))
            n += 1
    assert not any('assert(' in l for l in body), fn
    return '\n'.join(head + body + TRAP + tail), n


stale = []
for name in ('compute', 'verify'):
    fn = 'step_%s_accum' % name
    path = os.path.join(HERE, fn + '.cu')
    inl_path = os.path.join(HERE, fn + '_inl.cuh')
    src = open(path, newline='').read()
    new, n = asserts(src, fn)
    old = '__device__ void %s(' % fn
    assert new.count('\n' + old) == 1
    inl = new.replace('\n' + old, '\nstatic __device__ __forceinline__ void %s_inl(' % fn)
    try:
        cur_inl = open(inl_path, newline='').read()
    except FileNotFoundError:
        cur_inl = ''
    for p, cur, want in ((path, src, new), (inl_path, cur_inl, inl)):
        if cur != want:
            stale.append(os.path.basename(p))
            if not CHECK:
                open(p, 'w', newline='').write(want)
    if n:
        print('accum_inl_gen: %s: %d asserts made branch-free' % (fn, n))
if CHECK:
    if stale:
        sys.exit('accum_inl_gen: %s out of date; run kernels/cuda/accum_inl_gen.py and commit the result'
                 % ', '.join(stale))
    print('accum_inl_gen: step_{compute,verify}_accum{.cu,_inl.cuh} are up to date')
else:
    print('accum_inl_gen: %s' % ('updated ' + ', '.join(stale) if stale else 'nothing to do'))
