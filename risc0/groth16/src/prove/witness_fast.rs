// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A faster evaluator for circom-witnesscalc graphs over BN254 Fr (`cuda` shrink_wrap).
//!
//! circom-witnesscalc does every multiplication as a 512-bit product and a long division,
//! and every division by a constant with a fresh modular inverse. Here the graph is compiled
//! once into a straight-line program:
//! - Values stay in canonical form, as in the reference evaluator, so every operation this
//!   module doesn't implement falls back to the reference code with the same result.
//! - Multiplications use Montgomery multiplication with precomputed constants:
//!   `a * c = mont(a, c*R)`, `a / c = mont(a, c^-1*R)`, `a * b = mont(mont(a, b), R^2)`.
//! - Values live in a small set of recycled slots (~30K for the rzup graph) instead of one
//!   per node (~18M).
//!
//! `compile` checks what the evaluation relies on (operands precede their node, constants
//! are canonical) and returns `None` otherwise; the caller then uses the reference evaluator.

use circom_witnesscalc::{
    field::{FieldOperations, U254},
    graph::{Node, Nodes, NodesStorage, Operation, VecNodes},
};

pub(crate) type L = [u64; 4];
const P: L = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];
const INV: u64 = 0xc2e1f593efffffff; // -P^-1 mod 2^64

#[inline(always)]
fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let t = (a as u128) + (b as u128) * (c as u128) + (carry as u128);
    (t as u64, (t >> 64) as u64)
}

#[inline(always)]
fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = (a as u128) + (b as u128) + (carry as u128);
    (t as u64, (t >> 64) as u64)
}

#[inline(always)]
fn geq_p(r: &L) -> bool {
    for i in (0..4).rev() {
        if r[i] != P[i] {
            return r[i] > P[i];
        }
    }
    true
}

#[inline(always)]
fn sub_p(r: &mut L) {
    let mut borrow = 0u64;
    for i in 0..4 {
        let (d, b1) = r[i].overflowing_sub(P[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        r[i] = d;
        borrow = (b1 | b2) as u64;
    }
}

/// CIOS Montgomery multiplication, no-carry variant (P < 2^254). Inputs and output < P.
#[inline(always)]
pub(crate) fn mont_mul(a: &L, b: &L) -> L {
    let mut t = [0u64; 4];
    for i in 0..4 {
        let (t0, mut c) = mac(t[0], a[0], b[i], 0);
        let m = t0.wrapping_mul(INV);
        let (_, mut c2) = mac(t0, m, P[0], 0);
        for j in 1..4 {
            let (x, c_) = mac(t[j], a[j], b[i], c);
            c = c_;
            let (y, c2_) = mac(x, m, P[j], c2);
            c2 = c2_;
            t[j - 1] = y;
        }
        t[3] = c + c2;
    }
    if geq_p(&t) {
        sub_p(&mut t);
    }
    t
}

#[inline(always)]
pub(crate) fn add_mod(a: &L, b: &L) -> L {
    let mut r = [0u64; 4];
    let mut c = 0;
    for i in 0..4 {
        let (x, c_) = adc(a[i], b[i], c);
        r[i] = x;
        c = c_;
    }
    if geq_p(&r) {
        sub_p(&mut r);
    }
    r
}

#[inline(always)]
pub(crate) fn sub_mod(a: &L, b: &L) -> L {
    let mut r = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (d, b1) = a[i].overflowing_sub(b[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        r[i] = d;
        borrow = (b1 | b2) as u64;
    }
    if borrow != 0 {
        let mut c = 0;
        for i in 0..4 {
            let (x, c_) = adc(r[i], P[i], c);
            r[i] = x;
            c = c_;
        }
    }
    r
}

/// One node of the program. Operands are slot ids; `k` indexes `Program::kmul`/`kraw`.
#[derive(Clone, Copy)]
pub(crate) enum Ins {
    MulVV(u32, u32),
    /// `a * c` or `a / c` with a constant c: `mont(a, kmul[k])`.
    MulVK(u32, u32),
    Add(u32, u32),
    /// `a + kraw[k]`
    AddVK(u32, u32),
    Sub(u32, u32),
    /// `a & kraw[k]`
    And(u32, u32),
    /// Any other node: the reference operation on the slots in `Program::slow_ops`.
    Slow,
}

/// Node-indexed instructions (operands are node ids), for `witness_par` (CUDA only).
#[cfg(feature = "cuda")]
pub(crate) struct NodeIns {
    pub(crate) ins: Vec<Ins>,
    pub(crate) kmul: Vec<L>,
    pub(crate) kraw: Vec<L>,
    pub(crate) r2: L,
}

#[cfg(feature = "cuda")]
pub(crate) fn compile_node_ins(
    nodes: &Nodes<U254, VecNodes>,
    signals: &[usize],
) -> Option<NodeIns> {
    let p = from_l(&P);
    let n = nodes.nodes.len();
    if n >= u32::MAX as usize || signals.len() >= u32::MAX as usize {
        return None;
    }
    // Operands precede their node (the evaluation reads slots without checks), and constants
    // are canonical (add_mod/sub_mod assume operands < P).
    for i in 0..n {
        if operands(&nodes.nodes.get(i).unwrap())
            .iter()
            .any(|&d| d != usize::MAX && d >= i)
        {
            return None;
        }
    }
    if nodes.constants.iter().any(|c| *c >= p) || signals.iter().any(|&s| s >= n) {
        return None;
    }

    // R = 2^256 mod p
    let r_mod = {
        let two128: U254 = U254::from(1u64) << 128usize;
        let r128: U254 = two128 % p;
        r128.mul_mod(r128, p)
    };
    let r2 = r_mod.mul_mod(r_mod, p);
    let cval = |i: usize| -> Option<U254> {
        match nodes.nodes.get(i).unwrap() {
            Node::Constant(c) => Some(nodes.constants[c]),
            _ => None,
        }
    };

    // Instructions with node-id operands.
    let mut kmul: Vec<L> = Vec::new();
    let mut kraw: Vec<L> = Vec::new();
    let mut kmul_idx = std::collections::HashMap::<(L, bool), u32>::new();
    let mut kraw_idx = std::collections::HashMap::<L, u32>::new();
    let mut mul_k = |c: U254, inverse: bool| -> u32 {
        *kmul_idx.entry((to_l(&c), inverse)).or_insert_with(|| {
            let m = if inverse { c.inv_mod(p).unwrap() } else { c };
            kmul.push(to_l(&m.mul_mod(r_mod, p)));
            (kmul.len() - 1) as u32
        })
    };
    let mut raw_k = |c: U254| -> u32 {
        *kraw_idx.entry(to_l(&c)).or_insert_with(|| {
            kraw.push(to_l(&c));
            (kraw.len() - 1) as u32
        })
    };
    let mut ins = Vec::with_capacity(n);
    for i in 0..n {
        let x = match nodes.nodes.get(i).unwrap() {
            Node::Op(op, a, b) => match (op, cval(a), cval(b)) {
                (Operation::Mul, None, None) => Ins::MulVV(a as u32, b as u32),
                (Operation::Mul, Some(c), None) => Ins::MulVK(b as u32, mul_k(c, false)),
                (Operation::Mul, None, Some(c)) => Ins::MulVK(a as u32, mul_k(c, false)),
                (Operation::Div, None, Some(c)) if c != U254::ZERO => {
                    Ins::MulVK(a as u32, mul_k(c, true))
                }
                (Operation::Add, None, None) => Ins::Add(a as u32, b as u32),
                (Operation::Add, Some(c), None) => Ins::AddVK(b as u32, raw_k(c)),
                (Operation::Add, None, Some(c)) => Ins::AddVK(a as u32, raw_k(c)),
                (Operation::Sub, None, None) => Ins::Sub(a as u32, b as u32),
                (Operation::Band, None, Some(c)) => Ins::And(a as u32, raw_k(c)),
                _ => Ins::Slow,
            },
            _ => Ins::Slow,
        };
        ins.push(x);
    }

    Some(NodeIns {
        ins,
        kmul,
        kraw,
        r2: to_l(&r2),
    })
}

pub(crate) struct Program {
    ins: Vec<Ins>,
    /// Destination slot per node.
    dst: Vec<u32>,
    /// Per node: the first witness index it is the value of (or u32::MAX).
    out_idx: Vec<u32>,
    /// Witness indices that repeat another's value: (index, index it copies).
    out_dups: Vec<(u32, u32)>,
    /// Operand slots of each `Slow` node, in node order.
    slow_ops: Vec<(u32, u32, u32)>,
    /// Montgomery-ready multiplicands (c*R or c^-1*R).
    kmul: Vec<L>,
    /// Canonical constants (for Add and And).
    kraw: Vec<L>,
    r2: L,
    nslots: usize,
}

pub(crate) fn to_l(x: &U254) -> L {
    *x.as_limbs()
}

pub(crate) fn from_l(x: &L) -> U254 {
    U254::from_limbs(*x)
}

pub(crate) fn operands(node: &Node) -> [usize; 3] {
    match *node {
        Node::Op(_, a, b) => [a, b, usize::MAX],
        Node::UnoOp(_, a) => [a, usize::MAX, usize::MAX],
        Node::TresOp(_, a, b, c) => [a, b, c],
        _ => [usize::MAX; 3],
    }
}

/// Compile `nodes` into a program computing the witness `signals`; `None` if the graph breaks
/// an assumption of the evaluator (see the module docs).
pub(crate) fn compile(nodes: &Nodes<U254, VecNodes>, signals: &[usize]) -> Option<Program> {
    let p = from_l(&P);
    let n = nodes.nodes.len();
    if n >= u32::MAX as usize || signals.len() >= u32::MAX as usize {
        return None;
    }
    // Operands precede their node (the evaluation reads slots without checks), and constants
    // are canonical (add_mod/sub_mod assume operands < P).
    for i in 0..n {
        if operands(&nodes.nodes.get(i).unwrap())
            .iter()
            .any(|&d| d != usize::MAX && d >= i)
        {
            return None;
        }
    }
    if nodes.constants.iter().any(|c| *c >= p) || signals.iter().any(|&s| s >= n) {
        return None;
    }

    // R = 2^256 mod p
    let r_mod = {
        let two128: U254 = U254::from(1u64) << 128usize;
        let r128: U254 = two128 % p;
        r128.mul_mod(r128, p)
    };
    let r2 = r_mod.mul_mod(r_mod, p);
    let cval = |i: usize| -> Option<U254> {
        match nodes.nodes.get(i).unwrap() {
            Node::Constant(c) => Some(nodes.constants[c]),
            _ => None,
        }
    };

    // Instructions with node-id operands.
    let mut kmul: Vec<L> = Vec::new();
    let mut kraw: Vec<L> = Vec::new();
    let mut kmul_idx = std::collections::HashMap::<(L, bool), u32>::new();
    let mut kraw_idx = std::collections::HashMap::<L, u32>::new();
    let mut mul_k = |c: U254, inverse: bool| -> u32 {
        *kmul_idx.entry((to_l(&c), inverse)).or_insert_with(|| {
            let m = if inverse { c.inv_mod(p).unwrap() } else { c };
            kmul.push(to_l(&m.mul_mod(r_mod, p)));
            (kmul.len() - 1) as u32
        })
    };
    let mut raw_k = |c: U254| -> u32 {
        *kraw_idx.entry(to_l(&c)).or_insert_with(|| {
            kraw.push(to_l(&c));
            (kraw.len() - 1) as u32
        })
    };
    let mut ins = Vec::with_capacity(n);
    for i in 0..n {
        let x = match nodes.nodes.get(i).unwrap() {
            Node::Op(op, a, b) => match (op, cval(a), cval(b)) {
                (Operation::Mul, None, None) => Ins::MulVV(a as u32, b as u32),
                (Operation::Mul, Some(c), None) => Ins::MulVK(b as u32, mul_k(c, false)),
                (Operation::Mul, None, Some(c)) => Ins::MulVK(a as u32, mul_k(c, false)),
                (Operation::Div, None, Some(c)) if c != U254::ZERO => {
                    Ins::MulVK(a as u32, mul_k(c, true))
                }
                (Operation::Add, None, None) => Ins::Add(a as u32, b as u32),
                (Operation::Add, Some(c), None) => Ins::AddVK(b as u32, raw_k(c)),
                (Operation::Add, None, Some(c)) => Ins::AddVK(a as u32, raw_k(c)),
                (Operation::Sub, None, None) => Ins::Sub(a as u32, b as u32),
                (Operation::Band, None, Some(c)) => Ins::And(a as u32, raw_k(c)),
                _ => Ins::Slow,
            },
            _ => Ins::Slow,
        };
        ins.push(x);
    }

    // Witness outputs: each node's first witness index, and the repeats.
    let mut out_idx = vec![u32::MAX; n];
    let mut out_dups = Vec::new();
    for (w, &s) in signals.iter().enumerate() {
        if out_idx[s] == u32::MAX {
            out_idx[s] = w as u32;
        } else {
            out_dups.push((w as u32, out_idx[s]));
        }
    }

    // Slot allocation: a node's slot is freed after its last use.
    let mut last = vec![0u32; n];
    for i in 0..n {
        for d in operands(&nodes.nodes.get(i).unwrap()) {
            if d != usize::MAX {
                last[d] = i as u32;
            }
        }
    }
    let mut free: Vec<u32> = Vec::new();
    let mut nslots = 0u32;
    let mut slot = vec![u32::MAX; n];
    let mut dst = vec![0u32; n];
    let mut slow_ops = Vec::new();
    for i in 0..n {
        let ds = operands(&nodes.nodes.get(i).unwrap());
        let s = |x: u32| slot[x as usize];
        ins[i] = match ins[i] {
            Ins::MulVV(a, b) => Ins::MulVV(s(a), s(b)),
            Ins::MulVK(a, k) => Ins::MulVK(s(a), k),
            Ins::Add(a, b) => Ins::Add(s(a), s(b)),
            Ins::AddVK(a, k) => Ins::AddVK(s(a), k),
            Ins::Sub(a, b) => Ins::Sub(s(a), s(b)),
            Ins::And(a, k) => Ins::And(s(a), k),
            Ins::Slow => {
                let m = |d: usize| if d == usize::MAX { u32::MAX } else { slot[d] };
                slow_ops.push((m(ds[0]), m(ds[1]), m(ds[2])));
                Ins::Slow
            }
        };
        // Operands whose last use is this node are read before its slot is written.
        for d in ds {
            if d != usize::MAX && last[d] == i as u32 && slot[d] != u32::MAX {
                free.push(slot[d]);
                slot[d] = u32::MAX;
            }
        }
        let sl = free.pop().unwrap_or_else(|| {
            nslots += 1;
            nslots - 1
        });
        dst[i] = sl;
        if last[i] > i as u32 {
            slot[i] = sl;
        } else {
            free.push(sl);
        }
    }

    Some(Program {
        ins,
        dst,
        out_idx,
        out_dups,
        slow_ops,
        kmul,
        kraw,
        r2: to_l(&r2),
        nslots: nslots as usize,
    })
}

/// Evaluate the witness into `out` (one canonical little-endian value per signal).
pub(crate) fn eval(p: &Program, nodes: &Nodes<U254, VecNodes>, inputs: &[U254], out: &mut [L]) {
    assert_eq!(p.ins.len(), nodes.nodes.len());
    assert!(
        out.len()
            > p.out_dups
                .iter()
                .map(|&(w, _)| w as usize)
                .max()
                .unwrap_or(0)
    );
    assert!(p
        .out_idx
        .iter()
        .all(|&w| w == u32::MAX || (w as usize) < out.len()));
    let ff = &nodes.ff;
    let mut v: Vec<L> = vec![[0; 4]; p.nslots];
    let (r2, km, kr) = (&p.r2, &p.kmul, &p.kraw);
    let mut slow = p.slow_ops.iter();
    for i in 0..p.ins.len() {
        // Safety: compile() checked that operands precede their node, so every slot index is
        // one it allocated (< nslots), and every k is an index it pushed.
        let x = unsafe {
            let g = |s: u32| v.get_unchecked(s as usize);
            match *p.ins.get_unchecked(i) {
                Ins::MulVV(a, b) => mont_mul(&mont_mul(g(a), g(b)), r2),
                Ins::MulVK(a, k) => mont_mul(g(a), km.get_unchecked(k as usize)),
                Ins::Add(a, b) => add_mod(g(a), g(b)),
                Ins::AddVK(a, k) => add_mod(g(a), kr.get_unchecked(k as usize)),
                Ins::Sub(a, b) => sub_mod(g(a), g(b)),
                Ins::And(a, k) => {
                    let (x, c) = (g(a), kr.get_unchecked(k as usize));
                    [x[0] & c[0], x[1] & c[1], x[2] & c[2], x[3] & c[3]]
                }
                Ins::Slow => {
                    let &(a, b, c) = slow.next().unwrap();
                    let g = |s: u32| from_l(&v[s as usize]);
                    let val: U254 = match nodes.nodes.get(i).unwrap() {
                        Node::Unknown => panic!("Unknown node"),
                        Node::Constant(c) => nodes.constants[c],
                        Node::Input(j) => inputs[j],
                        Node::Op(op, _, _) => ff.op_duo(op, g(a), g(b)),
                        Node::UnoOp(op, _) => ff.op_uno(op, g(a)),
                        Node::TresOp(..) => {
                            if g(a) == U254::ZERO {
                                g(c)
                            } else {
                                g(b)
                            }
                        }
                    };
                    to_l(&val)
                }
            }
        };
        v[p.dst[i] as usize] = x;
        let w = p.out_idx[i];
        if w != u32::MAX {
            out[w as usize] = x;
        }
    }
    for &(w, src) in &p.out_dups {
        out[w as usize] = out[src as usize];
    }
}
