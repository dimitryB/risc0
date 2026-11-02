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

"""HIP-only per-part variants of the split recursion eval_check (eval_check_split.cu).

eval_check_split.py imports this module and applies select() last. For each part named in its
table, the part's function gets an RDNA-architecture-specific body under
`#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__)` (or __GFX12__); the `#else` branch is
the unchanged body, so the CUDA build and the HIP host pass see the same tokens as before. The
variants:

- 'col': trace loads from the uniform column base p + c * size (SGPRs) at a 32-bit row byte
  offset (global_load saddr + voffset), as rv32im's to_hip.py does.
- 'lz1' / 'lz2': lazy Fp sums of products. An Fp sum/difference whose terms include products is
  kept as a raw 64-bit accumulator x standing for x * 2^-32 mod P (a raw product a * b of
  canonical residues is their Montgomery product, c << 32 is c itself); the high word is folded
  below P only when one more term could overflow 64 bits (the bound is tracked exactly here), and
  one REDC + v_min makes a canonical Fp only where one is needed (a product operand, an FpExt or
  lazy-chain statement). Products used once are evaluated at their use. 'lz1' keeps one
  accumulator step per source statement; 'lz2' flattens each single-use sum tree into one list of
  terms (products first). REDC is linear mod P and its output canonical, so for canonical inputs
  (the assumption ec_lazyredc.py documents) the result is bit-identical.

Which part gets which variant is measured (eval_check_split.py's EC_VARIANTS): the variants cut
VALU work 10-25% but change each part's register allocation, which decided the outcome per part.
"""

import re

P = 0x78000001
U64 = (1 << 64) - 1
PROD = (P - 1) ** 2
PC = P << 32            # an accumulator below PC needs no fold before REDC
CB = (P - 1) << 32      # canonical value as an accumulator term
NB = P << 32            # P - c (c canonical) as a term
MNB = P * (P - 1)       # (P - a) * b

HEAD = re.compile(r'__device__ __forceinline__ FpExt poly_fp_part(\d+)\(')
CONST = re.compile(r'^(\s*)Fp (x\d+)\((\d+)\);\s*$')
LOAD = re.compile(r'^(\s*)Fp (x\d+) = (ctrl|data|accum|out|mix)\[.*\];\s*$')
ARITH = re.compile(r'^(\s*)Fp (x\d+) = (x\d+) ([-+*]) (x\d+);\s*$')
TOK = re.compile(r'\bx\d+\b')
LHS = re.compile(r'^\s*(?:Fp|FpExt|RecLz|auto) (x\d+)\b')
COL = re.compile(r'\b(ctrl|data|accum)\[(\d+) \* size \+ \(\(idx - INV_RATE \* (\d+)\) & mask\)\]')
UCOL = re.compile(r'REC_UCOL\((ctrl|data|accum), (\d+), (\d+)\)')

HELPERS = r'''#ifdef __HIPCC__
// ---- ec_lazyfp.py: lazy Fp sums of products and uniform trace addressing (HIP, some parts) ----
// An accumulator x (uint64_t) stands for x * 2^-32 mod P. Canonical inputs assumed (as above).
__device__ __forceinline__ uint64_t rlf_mul(Fp a, Fp b) { return uint64_t(*a) * uint64_t(*b); }
__device__ __forceinline__ uint64_t rlf_c(Fp a) { return uint64_t(*a) << 32; }
__device__ __forceinline__ uint64_t rlf_neg(Fp a) { return uint64_t(0x78000001u - *a) << 32; }
__device__ __forceinline__ uint64_t rlf_mneg(Fp a, Fp b) { return uint64_t(0x78000001u - *a) * uint64_t(*b); }
// High word below 2P (fold1) or 3P (fold2) reduced below P.
__device__ __forceinline__ uint64_t rlf_fold1(uint64_t x) {
  uint32_t hi = uint32_t(x >> 32);
  hi = min(hi, hi - 0x78000001u);
  return (uint64_t(hi) << 32) | uint32_t(x);
}
__device__ __forceinline__ uint64_t rlf_fold2(uint64_t x) {
  uint32_t hi = uint32_t(x >> 32);
  hi = min(hi, hi - 0x78000001u);
  hi = min(hi, hi - 0x78000001u);
  return (uint64_t(hi) << 32) | uint32_t(x);
}
// REDC of x < P * 2^32: canonical x * 2^-32 mod P.
__device__ __forceinline__ Fp rlf_fin(uint64_t x) {
  uint32_t m = uint32_t(x) * 0x77ffffffu;
  uint64_t u = x + uint64_t(m) * 0x78000001u;
  uint32_t r = uint32_t(u >> 32);
  return Fp(min(r, r - 0x78000001u));
}
#endif
#if defined(__HIP_DEVICE_COMPILE__)
// Trace column c of p at row (idx - INV_RATE * k) mod size, from the uniform column base p + c * size
// (SGPRs) at a 32-bit row byte offset (global_load saddr + voffset).
#define REC_UCOL(p, c, k) (Fp(*(const __attribute__((address_space(1))) uint32_t*)(reinterpret_cast<uint64_t>( \
    reinterpret_cast<const char*>((p) + (c) * size) + (((idx - uint32_t(INV_RATE * (k))) & mask) << 2)))))
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX11__) || defined(__GFX12__))
// Trace column c of p at row (idx - INV_RATE * k) mod size as a raw buffer load (RDNA3/RDNA4, 'buf'
// parts): one resource per REC_BG = 32 columns, based at the group's first column; the column within
// the group as the SGPR offset (c % 32) * size * 4 and the row as the VGPR offset, which every column
// read at that row shares (instead of a 64-bit VGPR address per column). Offsets stay below 4 GB for
// size <= 2^25 (po2 <= 23). gfx11 loads carry SLC (as rv32im's eval_check, R3).
#define REC_BG 32u
#if defined(__GFX11__)
#define REC_BAUX 2
#else
#define REC_BAUX 0
#endif
__device__ __forceinline__ __amdgpu_buffer_rsrc_t rec_rsrc(const Fp* p, uint32_t g, uint32_t size) {
  return __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char*>(reinterpret_cast<const char*>(p) + uint64_t(g) * (REC_BG * 4u) * size), (short)0,
      (int)0xffffffff, 0x31004000);
}
#define REC_BCOL(p, c, k) (Fp((uint32_t)__builtin_amdgcn_raw_buffer_load_b32(rec_rsrc((p), (c) / REC_BG, size), \
    ((idx - uint32_t(INV_RATE * (k))) & mask) << 2, ((c) % REC_BG) * size * 4u, REC_BAUX)))
#endif
#endif
'''


class Part:
    """the lazy form of one part's body (straight-line statements)"""

    def __init__(self, lines, flat):
        self.flat = flat
        self.out = []
        self.defs = {}      # var -> (op, a, b): Fp arithmetic statements
        # var -> 'c' (canonical Fp var), 'l' (uint64 var_l), 'lc' (both), 'p' (pending), 'gone' (inlined)
        self.state = {}
        self.bound = {}     # var -> bound of var_l
        self.uses = {}
        self.wmemo = {}
        self.ctr = dict(prod=0, fin=0, fold1=0, fold2=0, canon_op=0, fused_neg=0, ksub=0)
        for t in TOK.findall('\n'.join(lines)):
            self.uses[t] = self.uses.get(t, 0) + 1
        for l in lines:
            m = ARITH.match(l)
            if m:
                self.uses[m.group(2)] -= 1  # its definition
        self.lines = lines

    def canon_avail(self, v):
        return self.state.get(v) in ('c', 'lc')

    def worthy(self, v):
        """v is (or would be) an accumulator that saves at least one REDC"""
        st = self.state.get(v)
        if st == 'l':
            return True
        if st != 'p':
            return False
        if v not in self.wmemo:
            op, a, b = self.defs[v]
            self.wmemo[v] = op == '*' or self.worthy(a) or self.worthy(b)
        return self.wmemo[v]

    def fold(self, e, b):
        if b < PC:
            return e, b
        if b < 2 * PC:
            self.ctr['fold1'] += 1
            return 'rlf_fold1(%s)' % e, PC - 1
        self.ctr['fold2'] += 1
        return 'rlf_fold2(%s)' % e, PC - 1

    def add(self, ea, ba, eb, bb):
        if ba + bb > U64:
            if ba >= bb:
                ea, ba = self.fold(ea, ba)
            else:
                eb, bb = self.fold(eb, bb)
        if ba + bb > U64:
            ea, ba = self.fold(ea, ba)
            eb, bb = self.fold(eb, bb)
        assert ba + bb <= U64
        return '%s + %s' % (ea, eb), ba + bb

    def canon_op(self, v):
        op, a, b = self.defs[v]
        ca, cb = self.need_canon(a), self.need_canon(b)
        self.out.append('  Fp %s = %s %s %s;' % (v, ca, op, cb))
        self.state[v] = 'c'
        self.ctr['canon_op'] += 1

    def neg_lazy(self, e, bd):
        """K - x for an accumulator x (K a multiple of P * 2^32 above its bound)"""
        if bd >= 2 * PC:
            e, bd = self.fold(e, bd)
        k = PC if bd < PC else 2 * PC
        self.ctr['ksub'] += 1
        return '(0x%xull - (%s))' % (k, e), k

    def terms(self, v, sign, out):
        """flatten the single-use pending +/- tree under v into signed terms"""
        op, a, b = self.defs[v]
        for x, sg in ((a, sign), (b, sign if op == '+' else -sign)):
            st = self.state.get(x)
            if st == 'p' and self.uses[x] == 1 and self.defs[x][0] in '+-' and self.worthy(x):
                self.state[x] = 'gone'
                self.terms(x, sg, out)
            elif st == 'p' and self.uses[x] == 1 and self.defs[x][0] == '*':
                self.state[x] = 'gone'
                out.append((sg, 'prod', self.defs[x][1], self.defs[x][2]))
            else:
                out.append((sg, 'var', x, None))

    def term_expr(self, t):
        sg, kind, a, b = t
        if kind == 'prod':
            ca, cb = self.need_canon(a), self.need_canon(b)
            self.ctr['prod'] += 1
            if sg > 0:
                return 'rlf_mul(%s, %s)' % (ca, cb), PROD
            self.ctr['fused_neg'] += 1
            return 'rlf_mneg(%s, %s)' % (ca, cb), MNB
        if not self.worthy(a):
            ca = self.need_canon(a)
            return ('rlf_c(%s)' % ca, CB) if sg > 0 else ('rlf_neg(%s)' % ca, NB)
        e, bd = self.need_lazy(a)
        return (e, bd) if sg > 0 else self.neg_lazy(e, bd)

    def expr(self, v):
        """accumulator expression (expr, bound) of a pending, worthy v (or of a single lazy use)"""
        op, a, b = self.defs[v]
        if op == '*':
            ca, cb = self.need_canon(a), self.need_canon(b)
            self.ctr['prod'] += 1
            return 'rlf_mul(%s, %s)' % (ca, cb), PROD
        if not self.flat:
            ea, ba = self.need_lazy(a)
            if op == '+':
                eb, bb = self.need_lazy(b)
            elif not self.worthy(b):
                eb, bb = 'rlf_neg(%s)' % self.need_canon(b), NB
            elif self.state.get(b) == 'p' and self.defs[b][0] == '*' and self.uses[b] == 1:
                _, c, d = self.defs[b]
                cc, cd = self.need_canon(c), self.need_canon(d)
                self.state[b] = 'gone'
                self.ctr['fused_neg'] += 1
                self.ctr['prod'] += 1
                eb, bb = 'rlf_mneg(%s, %s)' % (cc, cd), MNB
            else:
                eb, bb = self.neg_lazy(*self.need_lazy(b))
            return self.add(ea, ba, eb, bb)
        ts = []
        self.terms(v, 1, ts)
        ts.sort(key=lambda t: 0 if t[1] == 'prod' else 1)
        acc = None
        for t in ts:
            e, bd = self.term_expr(t)
            if acc is None:
                acc = (e, bd)
            else:
                e2, b2 = self.add(acc[0], acc[1], e, bd)
                acc = ('(%s)' % e2, b2)
        return acc

    def emit_named(self, v):
        """pending v as a named variable: canonical (not worthy) or accumulator"""
        if not self.worthy(v):
            self.canon_op(v)
            return
        e, b = self.expr(v)
        self.out.append('  uint64_t %s_l = %s;' % (v, e))
        self.state[v] = 'l'
        self.bound[v] = b

    def need_lazy(self, v):
        st = self.state.get(v)
        if st in ('c', 'lc'):
            return 'rlf_c(%s)' % v, CB
        if st == 'l':
            return '%s_l' % v, self.bound[v]
        if st == 'p':
            if self.uses[v] == 1:
                op, a, b = self.defs[v]
                if self.worthy(v):
                    e, bd = self.expr(v)
                elif op == '+':
                    ca, cb = self.need_canon(a), self.need_canon(b)
                    e, bd = 'rlf_c(%s) + rlf_c(%s)' % (ca, cb), 2 * CB
                else:
                    ca, cb = self.need_canon(a), self.need_canon(b)
                    e, bd = 'rlf_c(%s) + rlf_neg(%s)' % (ca, cb), CB + NB
                self.state[v] = 'gone'
                return ('(%s)' % e if ' + ' in e else e), bd
            self.emit_named(v)
            return self.need_lazy(v)
        raise KeyError('need_lazy: %s in state %s' % (v, st))

    def need_canon(self, v):
        st = self.state.get(v)
        if st in ('c', 'lc'):
            return v
        if st == 'p':
            if not self.worthy(v):
                self.canon_op(v)
                return v
            if self.uses[v] == 1 or self.defs[v][0] == '*':
                e, bd = self.fold(*self.expr(v))
                self.out.append('  Fp %s = rlf_fin(%s);' % (v, e))
                self.ctr['fin'] += 1
                self.state[v] = 'c'
                return v
            self.emit_named(v)
            return self.need_canon(v)
        if st == 'l':
            e, bd = self.fold('%s_l' % v, self.bound[v])
            self.out.append('  Fp %s = rlf_fin(%s);' % (v, e))
            self.ctr['fin'] += 1
            self.state[v] = 'lc'
            return v
        raise KeyError('need_canon: %s in state %s' % (v, st))

    def run(self):
        for l in self.lines:
            m = CONST.match(l) or LOAD.match(l)
            if m:
                self.out.append(l)
                self.state[m.group(2)] = 'c'
                continue
            m = ARITH.match(l)
            if m:
                _, v, a, op, b = m.groups()
                self.defs[v] = (op, a, b)
                self.state[v] = 'p'
                continue
            lhs = LHS.match(l)
            for t in dict.fromkeys(TOK.findall(l)):
                if lhs and t == lhs.group(1):
                    continue
                if t in self.state:
                    self.need_canon(t)
            self.out.append(l)
        return self.out


def variant(fn, spec, counts):
    lines = fn.split('\n')
    hdr, body = lines[0], lines[1:-2]
    if spec.endswith('+buf') or spec == 'buf':
        # 'buf' (alone or after another form): the trace loads as raw buffer loads (REC_BCOL).
        base = spec[:-4] if spec.endswith('+buf') else None
        if base:
            body = variant(fn, base, counts).split('\n')[1:-2]
        n = sum(len(COL.findall(l)) + len(UCOL.findall(l)) for l in body)
        body = [UCOL.sub(lambda m: 'REC_BCOL(%s, %s, %s)' % m.groups(), COL.sub(lambda m: 'REC_BCOL(%s, %s, %s)' % m.groups(), l)) for l in body]
        counts['buf'] = counts.get('buf', 0) + n
        return '\n'.join([hdr] + body + ['}', ''])
    if spec == 'col':
        nb = [COL.sub(lambda m: 'REC_UCOL(%s, %s, %s)' % m.groups(), l) for l in body]
        counts['col'] = counts.get('col', 0) + sum(len(COL.findall(l)) for l in body)
    else:
        assert spec in ('lz1', 'lz2'), spec
        pt = Part(body, spec == 'lz2')
        nb = pt.run()
        counts[spec + '_prod'] = counts.get(spec + '_prod', 0) + pt.ctr['prod']
        counts[spec + '_fin'] = counts.get(spec + '_fin', 0) + pt.ctr['fin']
    return '\n'.join([hdr] + nb + ['}', ''])


def select(text, table):
    """table = {part: {'gfx11': spec, 'gfx12': spec}}: returns (text, counts)"""
    ns = 'namespace risc0::circuit::recursion::cuda {\n'
    assert text.count(ns) == 1
    i = text.index(ns) + len(ns)
    text = text[:i] + HELPERS + text[i:]
    out = []
    pos = 0
    counts = {}
    for m in HEAD.finditer(text):
        sel = table.get(int(m.group(1)), {})
        if not any(sel.values()):
            continue
        s = m.start()
        e = text.index('\n}\n', s) + 3
        fn = text[s:e]
        blk = ''
        for arch, macro in (('gfx11', '__GFX11__'), ('gfx12', '__GFX12__')):
            if sel.get(arch):
                blk += '#%s defined(__HIP_DEVICE_COMPILE__) && defined(%s)\n' % ('elif' if blk else 'if', macro)
                blk += variant(fn, sel[arch], counts)
        out.append(text[pos:s] + blk + '#else\n' + fn + '#endif\n')
        pos = e
    out.append(text[pos:])
    return ''.join(out), counts
