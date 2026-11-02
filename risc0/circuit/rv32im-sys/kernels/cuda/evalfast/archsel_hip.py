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

# Per-architecture variants of the ROCm eval_check in one file: regen.py runs the passes after
# wmma_hip.py twice, with different parameters, and merge() joins the two outputs line by line.
# Every run of lines where they differ becomes
#
#   #if defined(<macro>) // archsel_hip
#   <the variant's lines>
#   #else // archsel_hip
#   <the default's lines>
#   #endif // archsel_hip
#
# so a device compile for the macro's targets reads exactly the variant and every other compile
# (other targets and the host pass) exactly the default: merge() checks that selecting either side
# of its directives gives back the two inputs byte for byte, and that no differing run contains a
# conditional directive of its own.
import difflib
import re

TAG = ' // archsel_hip'
# Conditional directives may not sit in a differing run (they would nest across its #if/#else);
# #define lines may.
COND = re.compile(r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b')


def select(s, variant):
    """The lines a compile with (variant=True) or without the macro reads from merge()'s output."""
    out, keep, depth = [], True, 0
    for line in s.split('\n'):
        if line.startswith('#if defined(') and line.endswith(TAG):
            assert depth == 0
            depth, keep = 1, variant
        elif line == '#else' + TAG:
            assert depth == 1
            keep = not variant
        elif line == '#endif' + TAG:
            assert depth == 1
            depth, keep = 0, True
        elif keep:
            out.append(line)
    assert depth == 0
    return '\n'.join(out)


def merge(default, variant, macro, stats=None):
    a, b = default.split('\n'), variant.split('\n')
    out, runs, lines = [], 0, 0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            out.extend(a[i1:i2])
            continue
        assert not any(COND.match(l) for l in a[i1:i2] + b[j1:j2]), (i1, j1)
        out.append('#if defined(%s)%s' % (macro, TAG))
        out.extend(b[j1:j2])
        out.append('#else' + TAG)
        out.extend(a[i1:i2])
        out.append('#endif' + TAG)
        runs += 1
        lines += (i2 - i1) + (j2 - j1)
    s = '\n'.join(out)
    assert select(s, False) == default and select(s, True) == variant
    msg = 'archsel_hip: %d runs of %d lines selected by %s' % (runs, lines, macro)
    print(msg)
    if stats is not None:
        stats.append(msg)
    return s
