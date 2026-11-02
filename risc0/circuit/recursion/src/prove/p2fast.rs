// Faster Poseidon2 permutation for the recursion preflight (bit-identical to
// `risc0_zkp::core::hash::poseidon2::poseidon2_mix`).
//
// The linear layers (M_EXT, the M_INT row sum) are done in u64 integer arithmetic with one
// reduction per output instead of a modular add per step; multiplications by 2 and 4 become
// shifts instead of Montgomery multiplies. Every value is canonical ([0, P)) in Montgomery form
// at each step that is reduced, so the output equals the reference exactly. Inputs that are not
// canonical take the reference path (its add/mul behave differently on them).
// On x86_64 the same code is compiled a second time with AVX2 enabled (runtime-detected), which
// lets the compiler vectorize the 24-wide S-box and M_INT loops.

use risc0_zkp::{
    core::hash::poseidon2::{
        poseidon2_mix, CELLS, M_INT_DIAG_HZN, ROUNDS_HALF_FULL, ROUNDS_PARTIAL, ROUND_CONSTANTS,
    },
    field::baby_bear::BabyBearElem as Fp,
};
use std::sync::LazyLock;

const P: u32 = 15 * (1 << 27) + 1;
const M: u32 = 0x88000001;

struct Consts {
    rc: Vec<u32>,
    diag: [u32; CELLS],
}

static CONSTS: LazyLock<Consts> = LazyLock::new(|| {
    let rc = ROUND_CONSTANTS
        .iter()
        .map(|e| e.as_u32_montgomery())
        .collect();
    let mut diag = [0u32; CELLS];
    for i in 0..CELLS {
        diag[i] = M_INT_DIAG_HZN[i].as_u32_montgomery();
    }
    Consts { rc, diag }
});

#[inline(always)]
fn mul(lhs: u32, rhs: u32) -> u32 {
    let mut o64: u64 = (lhs as u64).wrapping_mul(rhs as u64);
    let low: u32 = 0u32.wrapping_sub(o64 as u32);
    let red = M.wrapping_mul(low);
    o64 += (red as u64).wrapping_mul(P as u64);
    let ret = (o64 >> 32) as u32;
    if ret >= P {
        ret - P
    } else {
        ret
    }
}

#[inline(always)]
fn add(l: u32, r: u32) -> u32 {
    let x = l.wrapping_add(r);
    if x >= P {
        x - P
    } else {
        x
    }
}

#[inline(always)]
fn red(x: u64) -> u32 {
    (x % P as u64) as u32
}

#[inline(always)]
fn sbox(x: u32) -> u32 {
    let x2 = mul(x, x);
    let x4 = mul(x2, x2);
    let x6 = mul(x4, x2);
    mul(x6, x)
}

#[inline(always)]
fn mext(c: &mut [u32; CELLS]) {
    let mut o = [0u64; CELLS];
    let mut ts = [0u64; 4];
    for i in 0..CELLS / 4 {
        let x0 = c[i * 4] as u64;
        let x1 = c[i * 4 + 1] as u64;
        let x2 = c[i * 4 + 2] as u64;
        let x3 = c[i * 4 + 3] as u64;
        let t0 = x0 + x1;
        let t1 = x2 + x3;
        let t2 = 2 * x1 + t1;
        let t3 = 2 * x3 + t0;
        let t4 = 4 * t1 + t3;
        let t5 = 4 * t0 + t2;
        let t6 = t3 + t5;
        let t7 = t2 + t4;
        o[i * 4] = t6;
        o[i * 4 + 1] = t5;
        o[i * 4 + 2] = t7;
        o[i * 4 + 3] = t4;
        ts[0] += t6;
        ts[1] += t5;
        ts[2] += t7;
        ts[3] += t4;
    }
    for i in 0..CELLS {
        c[i] = red(o[i] + ts[i % 4]);
    }
}

#[inline(always)]
fn mix_canonical(c: &mut [u32; CELLS], k: &Consts) {
    let mut round = 0;
    mext(c);
    for _ in 0..ROUNDS_HALF_FULL {
        for (i, x) in c.iter_mut().enumerate() {
            *x = sbox(add(*x, k.rc[round * CELLS + i]));
        }
        mext(c);
        round += 1;
    }
    for _ in 0..ROUNDS_PARTIAL {
        c[0] = sbox(add(c[0], k.rc[round * CELLS]));
        let s: u64 = c.iter().map(|&x| x as u64).sum();
        let s = red(s);
        for (x, &d) in c.iter_mut().zip(k.diag.iter()) {
            *x = add(s, mul(d, *x));
        }
        round += 1;
    }
    for _ in 0..ROUNDS_HALF_FULL {
        for (i, x) in c.iter_mut().enumerate() {
            *x = sbox(add(*x, k.rc[round * CELLS + i]));
        }
        mext(c);
        round += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2")]
unsafe fn mix_canonical_avx2(c: &mut [u32; CELLS], k: &Consts) {
    mix_canonical(c, k)
}

static MODE: LazyLock<u8> = LazyLock::new(|| {
    // 0: reference, 1: scalar fast, 2: AVX2 fast. RISC0_CUDA_REC_PREP=0 forces the reference.
    if !super::host_prep_enabled() {
        return 0;
    }
    #[cfg(target_arch = "x86_64")]
    let m = if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("bmi2") {
        2
    } else {
        1
    };
    #[cfg(not(target_arch = "x86_64"))]
    let m = 1;
    // Self-check against the reference on fixed vectors; fall back to it on any mismatch.
    let mut s: u64 = 0x243F6A8885A308D3;
    for t in 0..64u32 {
        let mut a = [Fp::new(0); CELLS];
        for (i, x) in a.iter_mut().enumerate() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let v = match t % 3 {
                0 => (s % P as u64) as u32,
                1 => P - 1 - (s % 3) as u32,
                _ => (i as u32 * t) % P,
            };
            *x = Fp::new_raw(v);
        }
        let mut b = a;
        poseidon2_mix(&mut a);
        mix_dispatch(&mut b, m);
        if a != b {
            return 0;
        }
    }
    m
});

#[inline(always)]
fn mix_dispatch(cells: &mut [Fp; CELLS], mode: u8) {
    let mut c = [0u32; CELLS];
    for i in 0..CELLS {
        c[i] = cells[i].as_u32_montgomery();
    }
    if c.iter().any(|&x| x >= P) || mode == 0 {
        poseidon2_mix(cells);
        return;
    }
    let k = &*CONSTS;
    match mode {
        #[cfg(target_arch = "x86_64")]
        2 => unsafe { mix_canonical_avx2(&mut c, k) },
        _ => mix_canonical(&mut c, k),
    }
    for i in 0..CELLS {
        cells[i] = Fp::new_raw(c[i]);
    }
}

/// Drop-in for `poseidon2_mix`.
pub(crate) fn poseidon2_mix_fast(cells: &mut [Fp; CELLS]) {
    mix_dispatch(cells, *MODE)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_reference() {
        let mut s: u64 = 1;
        for t in 0..200_000u32 {
            let mut a = [Fp::new(0); CELLS];
            for x in a.iter_mut() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                *x = Fp::new_raw(match t % 4 {
                    0 => (s % P as u64) as u32,
                    1 => [0, 1, P - 1, P - 2][(s % 4) as usize],
                    2 => (s as u32) | 0x8000_0000, // non-canonical: reference path
                    _ => P - 1 - (s % 16) as u32,
                });
            }
            let mut b = a;
            poseidon2_mix(&mut a);
            poseidon2_mix_fast(&mut b);
            assert_eq!(a, b);
        }
    }
}
