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

//! An implementation of Poseidon targeting the Snark friendly field with a
//! security of 127 bits.

mod consts;

use alloc::sync::Arc;

use ff::{Field, PrimeField};
use risc0_core::field::{
    baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
    Elem, ExtElem,
};

use self::consts::{Fr, FrRepr, CELLS, ROUNDS_HALF_FULL, ROUNDS_PARTIAL};

use super::{HashFn, HashSuite, Rng, RngFactory};
use crate::core::digest::Digest;

fn add_round_constants(cells: &mut [Fr; CELLS], round: usize) {
    for (i, cell) in cells.iter_mut().enumerate() {
        *cell += consts::round_constants()[round * CELLS + i];
    }
}

fn sbox(x: Fr) -> Fr {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x4
}

fn do_full_sboxes(cells: &mut [Fr; CELLS]) {
    for cell in cells.iter_mut() {
        *cell = sbox(*cell);
    }
}

fn do_partial_sboxes(cells: &mut [Fr; CELLS]) {
    cells[0] = sbox(cells[0]);
}

fn multiply_by_mds(cells: &mut [Fr; CELLS]) {
    let old_cells = *cells;
    for (i, cell) in cells.iter_mut().enumerate() {
        let mut tot = Fr::ZERO;
        for (j, old_cell) in old_cells.iter().enumerate() {
            tot += consts::mds()[i * CELLS + j] * *old_cell;
        }
        *cell = tot;
    }
}

fn full_round(cells: &mut [Fr; CELLS], round: usize) {
    add_round_constants(cells, round);
    do_full_sboxes(cells);
    multiply_by_mds(cells);
}

fn partial_round(cells: &mut [Fr; CELLS], round: usize) {
    add_round_constants(cells, round);
    do_partial_sboxes(cells);
    multiply_by_mds(cells);
}

fn poseidon_mix(cells: &mut [Fr; CELLS]) {
    let mut round = 0;
    for _i in 0..ROUNDS_HALF_FULL {
        full_round(cells, round);
        round += 1;
    }
    for _i in 0..ROUNDS_PARTIAL {
        partial_round(cells, round);
        round += 1;
    }
    for _i in 0..ROUNDS_HALF_FULL {
        full_round(cells, round);
        round += 1;
    }
}

/// A hash implementation for Poseidon in a Snark friendly field
struct Poseidon254HashFn;

/// Returns the field representation of a given digest.
pub fn digest_to_fr(digest: &Digest) -> Fr {
    let mut repr: FrRepr = FrRepr::default();
    repr.as_mut().clone_from_slice(digest.as_bytes());
    Fr::from_repr_vartime(repr).unwrap()
}

/// Returns the canonical little-endian byte representation of a digest as a BN254 Fr element.
/// This is equivalent to `digest_to_fr(digest).to_repr()` but avoids exposing the `Fr` type.
pub fn digest_to_fr_bytes(digest: &Digest) -> [u8; 32] {
    let fr = digest_to_fr(digest);
    let repr = fr.to_repr();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(repr.as_ref());
    bytes
}

fn fr_to_digest(fr: &Fr) -> Digest {
    let repr = fr.to_repr();
    Digest::try_from(repr.as_ref()).unwrap()
}

fn unpadded_hash<'a, I>(iter: I) -> Digest
where
    I: Iterator<Item = &'a BabyBearElem>,
{
    let mut cells = [Fr::ZERO; CELLS];
    let mut mul = Fr::ONE;
    let mut idx = 1;
    let mut count = 0;
    let p = Fr::from(15 * (1 << 27) + 1);
    for val in iter {
        cells[idx] += mul * Fr::from(val.as_u32() as u64);
        mul *= p;
        count += 1;
        if count == 8 {
            mul = Fr::ONE;
            count = 0;
            idx += 1;
        }
        if idx == 3 {
            poseidon_mix(&mut cells);
            cells[1] = Fr::ZERO;
            cells[2] = Fr::ZERO;
            idx = 1;
        }
    }
    if idx != 1 || count != 0 {
        poseidon_mix(&mut cells);
    }
    fr_to_digest(&cells[0])
}

impl HashFn<BabyBear> for Poseidon254HashFn {
    fn hash_pair(&self, a: &Digest, b: &Digest) -> Box<Digest> {
        let mut cells = [Fr::ZERO; CELLS];
        cells[1] = digest_to_fr(a);
        cells[2] = digest_to_fr(b);
        poseidon_mix(&mut cells);
        Box::new(fr_to_digest(&cells[0]))
    }

    fn hash_elem_slice(&self, slice: &[BabyBearElem]) -> Box<Digest> {
        Box::new(unpadded_hash(slice.iter()))
    }

    fn hash_ext_elem_slice(&self, slice: &[BabyBearExtElem]) -> Box<Digest> {
        Box::new(unpadded_hash(
            slice.iter().flat_map(|ee| ee.subelems().iter()),
        ))
    }

    fn is_digest_valid(&self, digest: &Digest) -> bool {
        let mut repr: FrRepr = FrRepr::default();
        repr.as_mut().clone_from_slice(digest.as_bytes());
        Fr::from_repr_vartime(repr).is_some()
    }
}

/// An rng implementation for Poseidon in a Snark friendly field
struct Poseidon254Rng {
    // The cells of the sponge
    cells: [Fr; CELLS],
}

impl Poseidon254Rng {
    fn new() -> Self {
        Self {
            cells: [Fr::ZERO; CELLS],
        }
    }
}

impl Rng<BabyBear> for Poseidon254Rng {
    fn mix(&mut self, val: &Digest) {
        self.cells[1] += digest_to_fr(val);
        poseidon_mix(&mut self.cells);
    }

    fn random_bits(&mut self, bits: usize) -> u32 {
        let mut source = self.cells[2];
        poseidon_mix(&mut self.cells);
        let mut out: u32 = 0;
        let div2 = Fr::from(2).invert().unwrap();
        for i in 0..bits {
            let low_bit = source.is_odd().unwrap_u8();
            out |= (low_bit as u32) << i;
            source -= Fr::from(low_bit as u64);
            source *= div2;
        }
        out
    }

    fn random_elem(&mut self) -> BabyBearElem {
        let mut source = self.cells[2];
        poseidon_mix(&mut self.cells);
        let mut out = BabyBearElem::ZERO;
        let div2 = Fr::from(2).invert().unwrap();
        let mul2 = BabyBearElem::from(2u32);
        let mut mul = BabyBearElem::ONE;
        for _ in 0..160 {
            let low_bit = source.is_odd().unwrap_u8();
            source -= Fr::from(low_bit as u64);
            source *= div2;
            out += mul * BabyBearElem::from(low_bit as u32);
            mul *= mul2;
        }
        out
    }

    fn random_ext_elem(&mut self) -> BabyBearExtElem {
        risc0_core::field::ExtElem::from_subelems((0..4).map(|_| self.random_elem()))
    }
}

struct PoseidonRngFactory;

impl RngFactory<BabyBear> for PoseidonRngFactory {
    fn new_rng(&self) -> Box<dyn Rng<BabyBear> + Send> {
        Box::new(Poseidon254Rng::new())
    }
}

/// A hash suite of the SNARK-friendly version of Poseidon
pub struct Poseidon254HashSuite;

impl Poseidon254HashSuite {
    /// Construct a new Poseidon254HashSuite
    pub fn new_suite() -> HashSuite<BabyBear> {
        HashSuite {
            name: "poseidon254".into(),
            hashfn: Arc::new(Poseidon254HashFn {}),
            rng: Arc::new(PoseidonRngFactory {}),
        }
    }
}

#[cfg(test)]
mod tests {
    use test_log::test;

    use super::*;

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_field_arithmetic_parity() {
        use crate::hal::{
            metal::{KernelArg, MetalHalPoseidon254},
            Buffer, Hal,
        };

        fn from_limbs(limbs: [u32; 8]) -> Fr {
            let mut repr = FrRepr::default();
            let mut bytes = [0u8; 32];
            for (index, limb) in limbs.iter().enumerate() {
                bytes[index * 4..index * 4 + 4].copy_from_slice(&limb.to_le_bytes());
            }
            repr.as_mut().copy_from_slice(&bytes);
            Fr::from_repr_vartime(repr).unwrap()
        }

        fn to_limbs(value: Fr) -> [u32; 8] {
            let repr = value.to_repr();
            let bytes = repr.as_ref();
            std::array::from_fn(|index| {
                u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
            })
        }

        let modulus_minus_one = [
            0xf0000000, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029,
            0x30644e72,
        ];
        let montgomery_r = [
            0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695, 0x7879462e, 0x666ea36f, 0x9a07df2f,
            0x0e0a77c1,
        ];
        let montgomery_r2 = [
            0xae216da7, 0x1bb8e645, 0xe35c59e3, 0x53fe3ab1, 0x53bb8085, 0x8c49833d, 0x7f4e44a5,
            0x0216d0b1,
        ];
        let mut values = vec![
            Fr::ZERO,
            Fr::ONE,
            -Fr::ONE,
            from_limbs(montgomery_r),
            from_limbs(montgomery_r2),
            from_limbs(modulus_minus_one),
        ];
        let mut state = 0x9e3779b97f4a7c15u64;
        for _ in 0..128 {
            let mut limbs = [0u32; 8];
            for limb in &mut limbs {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *limb = state as u32;
            }
            limbs[7] &= 0x1fffffff;
            values.push(from_limbs(limbs));
        }
        let pairs: Vec<_> = values
            .iter()
            .copied()
            .zip(values.iter().rev().copied())
            .collect();
        let mut input = Vec::with_capacity(pairs.len() * 16);
        for (a, b) in &pairs {
            input.extend_from_slice(&to_limbs(*a));
            input.extend_from_slice(&to_limbs(*b));
        }
        let gpu = MetalHalPoseidon254::new();
        let in_buffer = gpu.copy_from_u32("field_probe_input", &input);
        let out_buffer = gpu.alloc_u32("field_probe_output", input.len());
        gpu.dispatch_by_name(
            "poseidon254_field_probe",
            &[
                in_buffer.as_arg(),
                out_buffer.as_arg(),
                KernelArg::Integer(pairs.len() as u32),
            ],
            pairs.len() as u64,
        );
        let actual = out_buffer.to_vec();
        for (index, (a, b)) in pairs.iter().enumerate() {
            assert_eq!(
                &actual[index * 16..index * 16 + 8],
                &to_limbs(*a + *b),
                "add {index}"
            );
            assert_eq!(
                &actual[index * 16 + 8..index * 16 + 16],
                &to_limbs(*a * *b),
                "mul {index}"
            );
        }
    }

    #[test]
    fn p254_test_vectors() {
        let mut input = Vec::<BabyBearElem>::new();
        let mut output = Vec::<BabyBearElem>::new();
        for i in 1..6 {
            input.push(BabyBearElem::from(i as u32));
        }
        let mut iop = Poseidon254Rng::new();
        let hasher = Poseidon254HashFn {};
        let digest1 = *hasher.hash_elem_slice(&input);
        let digest2 = *hasher.hash_pair(&digest1, &digest1);
        let digest3 = *hasher.hash_pair(&digest1, &digest2);
        iop.mix(&digest3);
        output.push(BabyBearElem::from(iop.random_bits(7)));
        output.push(BabyBearElem::from(iop.random_elem()));
        for _ in 0..23 {
            input.push(iop.random_elem());
        }
        iop.mix(&hasher.hash_elem_slice(&input));
        output.push(BabyBearElem::from(iop.random_elem()));
        tracing::info!("Output = {:?}", &output);
        assert!(output[0].as_u32() == 5);
        assert!(output[1].as_u32() == 328085114);
        assert!(output[2].as_u32() == 726238606);
    }
}
