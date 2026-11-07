// Copyright 2025 RISC Zero, Inc.
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

use anyhow::{ensure, Result};
use risc0_zkp::field::Elem as _;

use crate::{
    execute::bigint::BIGINT_WIDTH_BYTES,
    zirgen::circuit::{BigIntAccumStateLayout, ExtVal, LAYOUT_TOP_ACCUM},
};

use super::bigint::{BigIntState, PolyOp};

const BIGINT_ACCUM_STATE_LAYOUT: &BigIntAccumStateLayout = LAYOUT_TOP_ACCUM.user._0.state;
const BIGINT_ACCUM_STATE_COUNT: usize = 3 * 4;

const MAX_POWERS: usize = BIGINT_WIDTH_BYTES + 1;

#[derive(Clone, Debug)]
pub(crate) struct BigIntAccumState {
    pub poly: ExtVal,
    pub term: ExtVal,
    pub total: ExtVal,
}

impl BigIntAccumState {
    fn new() -> Self {
        Self {
            poly: ExtVal::ZERO,
            term: ExtVal::ONE,
            total: ExtVal::ZERO,
        }
    }

    pub(crate) const fn offsets() -> [usize; BIGINT_ACCUM_STATE_COUNT] {
        [
            BIGINT_ACCUM_STATE_LAYOUT.poly._super.offset,
            BIGINT_ACCUM_STATE_LAYOUT.poly._super.offset + 1,
            BIGINT_ACCUM_STATE_LAYOUT.poly._super.offset + 2,
            BIGINT_ACCUM_STATE_LAYOUT.poly._super.offset + 3,
            BIGINT_ACCUM_STATE_LAYOUT.term._super.offset,
            BIGINT_ACCUM_STATE_LAYOUT.term._super.offset + 1,
            BIGINT_ACCUM_STATE_LAYOUT.term._super.offset + 2,
            BIGINT_ACCUM_STATE_LAYOUT.term._super.offset + 3,
            BIGINT_ACCUM_STATE_LAYOUT.total._super.offset,
            BIGINT_ACCUM_STATE_LAYOUT.total._super.offset + 1,
            BIGINT_ACCUM_STATE_LAYOUT.total._super.offset + 2,
            BIGINT_ACCUM_STATE_LAYOUT.total._super.offset + 3,
        ]
    }

    pub(crate) fn as_array(&self) -> [u32; BIGINT_ACCUM_STATE_COUNT] {
        let poly = self.poly.elems();
        let term = self.term.elems();
        let total = self.total.elems();

        [
            poly[0].into(),
            poly[1].into(),
            poly[2].into(),
            poly[3].into(),
            term[0].into(),
            term[1].into(),
            term[2].into(),
            term[3].into(),
            total[0].into(),
            total[1].into(),
            total[2].into(),
            total[3].into(),
        ]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BigIntAccum {
    pub state: BigIntAccumState,
    powers: [ExtVal; MAX_POWERS],
    neg_poly: ExtVal,
}

impl BigIntAccum {
    pub(crate) fn new(mix: ExtVal) -> Self {
        let mut powers = [ExtVal::default(); MAX_POWERS];
        let mut cur = ExtVal::ONE;
        for power in powers.iter_mut() {
            *power = cur;
            cur *= mix;
        }

        let neg_poly = powers
            .iter()
            .take(BIGINT_WIDTH_BYTES)
            .fold(ExtVal::ZERO, |acc, power| {
                acc + *power * ExtVal::from_u32(128)
            });

        Self {
            state: BigIntAccumState::new(),
            powers,
            neg_poly,
        }
    }

    pub(crate) fn step(&mut self, state: &BigIntState) -> Result<()> {
        let delta_poly = state
            .bytes
            .iter()
            .zip(self.powers.iter().take(BIGINT_WIDTH_BYTES))
            .fold(ExtVal::ZERO, |acc, (coeff, power)| {
                acc + *power * ExtVal::from_u32(*coeff as u32)
            });
        let new_poly = self.state.poly + delta_poly;

        match state.poly_op {
            PolyOp::Reset => self.reset(),
            PolyOp::Shift => {
                self.state.poly = new_poly * self.powers[BIGINT_WIDTH_BYTES];
            }
            PolyOp::SetTerm => {
                self.state.poly = ExtVal::ZERO;
                self.state.term = new_poly;
            }
            PolyOp::AddTotal => {
                let coeff = ExtVal::from_u32(state.coeff) - ExtVal::from_u32(4);
                self.state.total += coeff * self.state.term * new_poly;
                self.state.poly = ExtVal::ZERO;
                self.state.term = ExtVal::ONE;
            }
            PolyOp::Carry1 => {
                self.state.poly += (delta_poly - self.neg_poly) * ExtVal::from_u32(64 * 256);
            }
            PolyOp::Carry2 => self.state.poly += delta_poly * ExtVal::from_u32(256),
            PolyOp::EqZero => {
                let carry = self.powers[1] - ExtVal::from_u32(256);
                let goal = self.state.total + new_poly * carry;
                ensure!(goal == ExtVal::ZERO, "Invalid eqz in bigint accum");
                self.reset();
            }
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.state = BigIntAccumState::new();
    }
}
