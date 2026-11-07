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

//! The bigint2 verify program: run by the executor and by the prover's preflight alike, so that a
//! guest whose witness the circuit would reject fails at execution — with the same memory writes,
//! in the same order, and the same cycle count as the proof will have.

use anyhow::{anyhow, bail, ensure, Result};
use derive_more::Debug;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive as _;
use risc0_binfmt::WordAddr;
use smallvec::{smallvec, SmallVec};

use super::{BigIntBytes, BigIntExec, BigIntWitness, BIGINT_WIDTH_BYTES, BIGINT_WIDTH_WORDS};
use crate::{
    execute::{
        platform::MEMORY_END_ADDR,
        r0vm::{LoadOp, Risc0Context},
        CycleState, WORD_SIZE,
    },
    MAX_INSN_CYCLES,
};

#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "prove"), allow(dead_code))] // the preflight records every field
pub(crate) struct BigIntState {
    pub is_ecall: bool,
    pub mode: u32,
    pub pc: WordAddr,
    pub poly_op: PolyOp,
    pub coeff: u32,
    pub bytes: BigIntBytes,
    pub next_state: CycleState,
}

#[derive(Debug)]
#[debug("{poly_op:?}({mem_op:?}, c:{coeff}, r:{reg}, o:{offset})")]
pub(crate) struct Instruction {
    pub poly_op: PolyOp,
    pub mem_op: MemoryOp,
    pub coeff: i32,
    pub reg: u32,
    pub offset: u32,
}

#[derive(Clone, Copy, Debug, FromPrimitive, PartialEq)]
pub(crate) enum PolyOp {
    Reset,
    Shift,
    SetTerm,
    AddTotal,
    Carry1,
    Carry2,
    EqZero,
}

#[derive(Clone, Copy, Debug, FromPrimitive, PartialEq)]
pub(crate) enum MemoryOp {
    Read,
    Write,
    Check,
}

impl Instruction {
    // instruction encoding:
    // 3  2   2  2    1               0
    // 1  8   4  1    6               0
    // mmmmppppcccaaaaaoooooooooooooooo
    pub fn decode(insn: u32) -> Result<Self> {
        Ok(Self {
            mem_op: MemoryOp::from_u32((insn >> 28) & 0x0f)
                .ok_or_else(|| anyhow!("Invalid mem_op in bigint program"))?,
            poly_op: PolyOp::from_u32((insn >> 24) & 0x0f)
                .ok_or_else(|| anyhow!("Invalid poly_op in bigint program"))?,
            coeff: ((insn >> 21) & 0x07) as i32 - 4,
            reg: (insn >> 16) & 0x1f,
            offset: insn & 0xffff,
        })
    }
}

/// Most cycles a verify program's run may count. The executor splits segments leaving each
/// instruction fewer than [`MAX_INSN_CYCLES`]; this ecall's are these and the machine ecall's own
/// (its paging, the executor counts against the same bound).
pub(crate) const MAX_RUN_CYCLES: usize = MAX_INSN_CYCLES - 2;

struct BigInt {
    state: BigIntState,
    program: BytePolyProgram,
}

/// Run the verify program of the bigint ecall `exec` decoded: from `exec.verify_program_ptr` to its
/// closing `Reset`, checking the witness as the circuit does and storing what its `Write`
/// instructions store, in their order. `on_cycle` sees each cycle (the preflight records them).
/// Returns the cycles it ran: the ecall's own and one per instruction.
///
/// A program whose ecall would take more cycles than one instruction may is refused (see
/// [`MAX_RUN_CYCLES`]); so is one that grows its polynomials or multiplies past what any bigint2
/// program does ([`MAX_POLY_COEFFS`], [`MAX_POLY_MUL_WORK`]), to bound the time it can take.
pub(crate) fn run<C: Risc0Context>(
    ctx: &mut C,
    exec: &BigIntExec,
    mut on_cycle: impl FnMut(&mut C, CycleState, &BigIntState),
) -> Result<usize> {
    let mut bigint = BigInt {
        state: BigIntState {
            is_ecall: true,
            mode: exec.mode,
            pc: exec.verify_program_ptr,
            poly_op: PolyOp::Reset,
            coeff: 0,
            bytes: Default::default(),
            next_state: CycleState::BigIntStep,
        },
        program: BytePolyProgram::new(),
    };
    on_cycle(ctx, CycleState::BigIntEcall, &bigint.state);
    let mut cycles = 1;
    while bigint.state.next_state == CycleState::BigIntStep {
        cycles += 1;
        ensure!(
            cycles <= MAX_RUN_CYCLES,
            "bigint verify program runs over the {MAX_INSN_CYCLES} cycles one instruction may take"
        );
        bigint.step(ctx, &exec.witness)?;
        on_cycle(ctx, CycleState::BigIntStep, &bigint.state);
    }
    Ok(cycles)
}

impl BigInt {
    fn step(&mut self, ctx: &mut impl Risc0Context, witness: &BigIntWitness) -> Result<()> {
        self.state.pc.inc();
        let insn = Instruction::decode(ctx.load_u32(LoadOp::Record, self.state.pc)?)?;

        let base =
            ctx.load_aligned_addr_from_machine_register(LoadOp::Record, insn.reg as usize)?;
        // The circuit adds the offset to the base's low half allowing one carry out of it, on every
        // instruction (a Check's address too, used or not).
        ensure!(
            (base.baddr().0 & 0xffff) + insn.offset * (BIGINT_WIDTH_BYTES as u32) < 0x2_0000,
            "bigint offset {} out of reach of its base {base:?}",
            insn.offset
        );
        let addr = base + insn.offset * BIGINT_WIDTH_WORDS as u32;

        tracing::trace!("step({:?}, {insn:?}, {addr:?})", self.state.pc);
        if matches!(insn.mem_op, MemoryOp::Read | MemoryOp::Write) {
            check_addr(addr, self.state.mode)?;
        }
        if insn.mem_op == MemoryOp::Check && insn.poly_op != PolyOp::Reset {
            if !self.program.in_carry {
                self.program.in_carry = true;
                self.program.total_carry = self.program.total.clone();
                let mut carry = 0;

                // Do carry propagation
                for i in 0..self.program.total_carry.len() {
                    let coeff = self.program.total_carry.get_mut(i);

                    *coeff = coeff.checked_add(carry).ok_or_else(out_of_range)?;
                    ensure!(*coeff % 256 == 0, "bad carry");
                    *coeff /= 256;
                    carry = *coeff;
                }
                tracing::trace!("carry propagate complete");
            }

            let base_point = 128 * 256 * 64;
            for (i, ret) in self.state.bytes.iter_mut().enumerate() {
                let offset = insn.offset as usize;
                let idx = offset * BIGINT_WIDTH_BYTES + i;
                // Past the polynomial's end the preflight's `get` panics: an error, here.
                ensure!(
                    idx < self.program.total_carry.len(),
                    "bigint check past the end of the polynomial"
                );
                let coeff = *self.program.total_carry.get(idx) as u32;
                let value = coeff.wrapping_add(base_point);
                match insn.poly_op {
                    PolyOp::Carry1 => *ret = ((value >> 14) & 0xff) as u8,
                    PolyOp::Carry2 => *ret = ((value >> 8) & 0x3f) as u8,
                    PolyOp::Shift | PolyOp::EqZero => *ret = (value & 0xff) as u8,
                    _ => {
                        bail!("Invalid poly_op in bigint program")
                    }
                }
            }
        } else if insn.mem_op == MemoryOp::Read {
            for i in 0..BIGINT_WIDTH_WORDS {
                let word = ctx.load_u32(LoadOp::Record, addr + i)?;
                for (j, byte) in word.to_le_bytes().iter().enumerate() {
                    self.state.bytes[i * WORD_SIZE + j] = *byte;
                }
            }
        } else if !addr.is_null() {
            self.state.bytes = *witness
                .get(&addr)
                .ok_or_else(|| anyhow!("Missing bigint witness: {addr:?}"))?;
            if insn.mem_op == MemoryOp::Write {
                let words: &[u32] = bytemuck::cast_slice(&self.state.bytes);
                for (i, word) in words.iter().enumerate() {
                    ctx.store_u32(addr + i, *word)?;
                }
            }
        }

        self.program.step(&insn, &self.state.bytes)?;

        self.state.is_ecall = false;
        self.state.poly_op = insn.poly_op;
        self.state.coeff = (insn.coeff + 4) as u32;
        self.state.next_state = if !self.state.is_ecall && insn.poly_op == PolyOp::Reset {
            CycleState::Decode
        } else {
            CycleState::BigIntStep
        };

        Ok(())
    }
}

/// Most coefficients one of a program's polynomials may have. A bigint2 program's grow with its
/// operands' bytes: the longest in RISC Zero's own (a 4096-bit modular multiply) reach 1,556. The
/// circuit sets no bound; one this long is built to take time, at execution and in the preflight.
pub(crate) const MAX_POLY_COEFFS: usize = 1 << 13;

/// Most coefficient products a program's multiplications may take, all told: RISC Zero's own take
/// at most ~803,000 (the same multiply). Without a bound, 25,000 cycles of shifts and one multiply
/// could take seconds.
pub(crate) const MAX_POLY_MUL_WORK: u64 = 1 << 25;

/// The circuit's rule for the address a `Read` or `Write` names (`BigIntAddr`): its byte address's
/// upper half is neither zero nor 0xBFFF, and in user mode below it. The pager reads and writes far
/// more (the zero page, kernel memory, a null `Write` it skipped): a program naming such an address
/// ran here and no proof of it could.
fn check_addr(addr: WordAddr, mode: u32) -> Result<()> {
    let high = addr.0 >> 14;
    ensure!(
        addr < MEMORY_END_ADDR && high != 0 && high != 0xbfff && (mode == 1 || high < 0xbfff),
        "Invalid bigint address: {addr:?}"
    );
    Ok(())
}

#[derive(Debug)]
pub(crate) struct BytePolyProgram {
    pub(crate) in_carry: bool,
    pub(crate) poly: BytePolynomial,
    pub(crate) term: BytePolynomial,
    pub(crate) total: BytePolynomial,
    pub(crate) total_carry: BytePolynomial,
    mul_work: u64,
}

impl BytePolyProgram {
    pub(crate) fn new() -> Self {
        Self {
            in_carry: false,
            poly: BytePolynomial::zero(),
            term: BytePolynomial::one(),
            total: BytePolynomial::zero(),
            total_carry: BytePolynomial::default(),
            mul_work: 0,
        }
    }

    pub(crate) fn step(&mut self, insn: &Instruction, witness: &[u8]) -> Result<()> {
        let delta_poly = BytePolynomial {
            coeffs: witness.iter().map(|&byte| byte as i32).collect(),
        };

        let new_poly = self.poly.add(&delta_poly)?;
        match insn.poly_op {
            PolyOp::Reset => {
                self.reset();
            }
            PolyOp::Shift => {
                self.poly = new_poly.shift()?;
            }
            PolyOp::SetTerm => {
                self.poly = BytePolynomial::zero();
                self.term = new_poly.clone();
            }
            PolyOp::AddTotal => {
                let term = new_poly
                    .mul(&self.term, &mut self.mul_work)?
                    .mul_const(insn.coeff)?;
                self.total = self.total.add(&term)?;
                self.term = BytePolynomial::one();
                self.poly = BytePolynomial::zero();
            }
            PolyOp::Carry1 => {
                let neg_poly = BytePolynomial {
                    coeffs: smallvec![-128; BIGINT_WIDTH_BYTES],
                };
                let carry = delta_poly.add(&neg_poly)?.mul_const(64 * 256)?;
                self.poly = self.poly.add(&carry)?;
            }
            PolyOp::Carry2 => {
                self.poly = self.poly.add(&delta_poly.mul_const(256)?)?;
            }
            PolyOp::EqZero => {
                let bp = BytePolynomial {
                    coeffs: smallvec![-256, 1, 0, 0],
                };
                self.total = self.total.add(&bp.mul(&new_poly, &mut self.mul_work)?)?;
                self.total.eqz()?;
                self.reset();
                self.in_carry = false;
            }
        }

        tracing::trace!(
            "delta_poly[0]: {}, new_poly[0]: {}, poly[0]: {}, term[0]: {}, total[0]: {}",
            delta_poly.get(0),
            new_poly.get(0),
            self.poly.get(0),
            self.term.get(0),
            self.total.get(0),
        );

        Ok(())
    }

    fn reset(&mut self) {
        self.poly = BytePolynomial::zero();
        self.term = BytePolynomial::one();
        self.total = BytePolynomial::zero();
    }
}

/// A polynomial over the integers, one coefficient per byte position, as a verify program builds
/// it; its lengths stay multiples of four, as upstream's four-lane chunks kept them.
///
/// The arithmetic is exact. Upstream's wrapped at 32 bits, and a coefficient that wrapped to zero
/// passed its `eqz` here while the prover's accumulator, which checks the same identity in the
/// circuit's field (where 2^32 is not zero), failed: a guest could execute cleanly and fail every
/// proof. A result out of the `i32` range is refused instead; every program that never wrapped
/// computes exactly what it did.
#[derive(Clone, Debug, Default)]
pub(crate) struct BytePolynomial {
    coeffs: SmallVec<[i32; 64]>,
}

fn out_of_range() -> anyhow::Error {
    anyhow!("bigint verify program's arithmetic leaves the 32-bit range")
}

fn narrow(value: i64) -> Result<i32> {
    i32::try_from(value).map_err(|_| out_of_range())
}

impl BytePolynomial {
    pub(crate) fn one() -> Self {
        Self {
            coeffs: smallvec![1, 0, 0, 0],
        }
    }

    pub(crate) fn zero() -> Self {
        Self {
            coeffs: smallvec![0; 4],
        }
    }

    fn checked(coeffs: SmallVec<[i32; 64]>) -> Result<Self> {
        ensure!(
            coeffs.len() <= MAX_POLY_COEFFS,
            "bigint verify program grows a polynomial past {MAX_POLY_COEFFS} coefficients"
        );
        Ok(Self { coeffs })
    }

    pub(crate) fn shift(&self) -> Result<Self> {
        let mut coeffs = SmallVec::with_capacity(BIGINT_WIDTH_BYTES + self.coeffs.len());
        coeffs.resize(BIGINT_WIDTH_BYTES, 0);
        coeffs.extend_from_slice(&self.coeffs);
        Self::checked(coeffs)
    }

    pub(crate) fn add(&self, rhs: &Self) -> Result<Self> {
        let (long, short) = if self.coeffs.len() >= rhs.coeffs.len() {
            (&self.coeffs, &rhs.coeffs)
        } else {
            (&rhs.coeffs, &self.coeffs)
        };
        let coeffs = long
            .iter()
            .enumerate()
            .map(|(i, &a)| narrow(a as i64 + short.get(i).copied().unwrap_or(0) as i64))
            .collect::<Result<_>>()?;
        Ok(Self { coeffs })
    }

    pub(crate) fn mul_const(&self, rhs: i32) -> Result<Self> {
        let coeffs = self
            .coeffs
            .iter()
            .map(|&a| narrow(a as i64 * rhs as i64))
            .collect::<Result<_>>()?;
        Ok(Self { coeffs })
    }

    /// The product, of length the sum of the two; `work` counts the coefficient products it takes,
    /// against [`MAX_POLY_MUL_WORK`].
    pub(crate) fn mul(&self, rhs: &Self, work: &mut u64) -> Result<Self> {
        let (lhs, rhs) = (&self.coeffs, &rhs.coeffs);
        *work += (lhs.len() * rhs.len()) as u64;
        ensure!(
            *work <= MAX_POLY_MUL_WORK,
            "bigint verify program's multiplications take over {MAX_POLY_MUL_WORK} products"
        );
        // Each sum is of at most min(len) products, each bounded by the two largest magnitudes:
        // within i64, it is exact.
        let max_abs = |c: &[i32]| {
            c.iter()
                .map(|a| a.unsigned_abs() as u128)
                .max()
                .unwrap_or(0)
        };
        let bound = max_abs(lhs) * max_abs(rhs) * lhs.len().min(rhs.len()) as u128;
        if bound > i64::MAX as u128 {
            return Err(out_of_range());
        }
        let mut sums = vec![0i64; lhs.len() + rhs.len()];
        for (i, &a) in lhs.iter().enumerate() {
            for (sum, &b) in sums[i..].iter_mut().zip(rhs.iter()) {
                *sum += a as i64 * b as i64;
            }
        }
        Self::checked(sums.into_iter().map(narrow).collect::<Result<_>>()?)
    }

    pub(crate) fn eqz(&self) -> Result<()> {
        ensure!(
            self.coeffs.iter().all(|&coeff| coeff == 0),
            "Invalid eqz in bigint program"
        );
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.coeffs.len()
    }

    #[inline(always)]
    pub(crate) fn get(&self, idx: usize) -> &i32 {
        &self.coeffs[idx]
    }

    #[inline(always)]
    pub(crate) fn get_mut(&mut self, idx: usize) -> &mut i32 {
        &mut self.coeffs[idx]
    }
}

#[cfg(test)]
mod tests {
    use risc0_binfmt::ByteAddr;

    use super::*;
    use crate::execute::{platform::*, r0vm::TestRisc0Context};

    /// Register holding the arena the test programs address.
    const ARENA: usize = REG_A0;
    const BASE: ByteAddr = ByteAddr(ZERO_PAGE_END_ADDR.0 + 0x1000);
    const PROGRAM: ByteAddr = ByteAddr(ZERO_PAGE_END_ADDR.0 + 0x8000);

    fn insn(mem_op: MemoryOp, poly_op: PolyOp, offset: u32) -> u32 {
        insn_coeff_reg(mem_op, poly_op, 0, ARENA, offset)
    }

    fn insn_coeff_reg(
        mem_op: MemoryOp,
        poly_op: PolyOp,
        coeff: i32,
        reg: usize,
        offset: u32,
    ) -> u32 {
        ((mem_op as u32) << 28)
            | ((poly_op as u32) << 24)
            | (((coeff + 4) as u32) << 21)
            | ((reg as u32) << 16)
            | offset
    }

    /// A context with the arena register set and `program` laid out after the verify pointer (the
    /// first instruction is read at the pointer plus one), and the ecall's decoded `exec`.
    fn setup(program: &[u32], witness: BigIntWitness) -> (TestRisc0Context, BigIntExec) {
        let mut ctx = TestRisc0Context::default();
        ctx.store_register(MACHINE_REGS_ADDR.waddr(), ARENA, BASE.0)
            .unwrap();
        let ptr = PROGRAM.waddr();
        for (i, word) in program.iter().enumerate() {
            ctx.store_u32(ptr + 1 + i, *word).unwrap();
        }
        let exec = BigIntExec {
            mode: 0,
            verify_program_ptr: ptr,
            witness,
        };
        (ctx, exec)
    }

    fn witness(entries: &[(u32, BigIntBytes)]) -> BigIntWitness {
        entries
            .iter()
            .map(|(offset, bytes)| (BASE.waddr() + offset * BIGINT_WIDTH_WORDS as u32, *bytes))
            .collect()
    }

    fn run_counting(ctx: &mut TestRisc0Context, exec: &BigIntExec) -> (Result<usize>, usize) {
        let mut seen = 0;
        let r = run(ctx, exec, |_, _, _| seen += 1);
        (r, seen)
    }

    #[test]
    fn writes_what_its_write_instructions_name_and_counts_its_cycles() {
        let bytes: BigIntBytes = std::array::from_fn(|i| i as u8 + 1);
        // Offset 1 is in the witness but no instruction writes it.
        let (mut ctx, exec) = setup(
            &[insn(MemoryOp::Write, PolyOp::Reset, 0)],
            witness(&[(0, bytes), (1, [9; BIGINT_WIDTH_BYTES])]),
        );
        let (r, seen) = run_counting(&mut ctx, &exec);
        assert_eq!(r.unwrap(), 2, "the ecall's cycle and one per instruction");
        assert_eq!(seen, 2, "the preflight is told of each of them");
        let written = ctx
            .load_region(LoadOp::Peek, BASE, BIGINT_WIDTH_BYTES)
            .unwrap();
        assert_eq!(written, bytes);
        ctx.assert_uninit(BASE + BIGINT_WIDTH_BYTES);
    }

    #[test]
    fn a_read_sees_memory_before_a_later_write() {
        // Read offset 0 (memory as it is), then write the witness there: in program order.
        let old: BigIntBytes = [7; BIGINT_WIDTH_BYTES];
        let new: BigIntBytes = [0; BIGINT_WIDTH_BYTES];
        let (mut ctx, exec) = setup(
            &[
                insn(MemoryOp::Read, PolyOp::SetTerm, 0),
                insn(MemoryOp::Write, PolyOp::Reset, 0),
            ],
            witness(&[(0, new)]),
        );
        ctx.store_region(BASE, &old).unwrap();
        let mut seen = vec![];
        let cycles = run(&mut ctx, &exec, |_, _, state| seen.push(state.bytes)).unwrap();
        assert_eq!(cycles, 3);
        assert_eq!(seen[1], old, "the Read saw memory as it was");
        assert_eq!(seen[2], new, "the Write stored the witness");
        assert_eq!(
            ctx.load_region(LoadOp::Peek, BASE, BIGINT_WIDTH_BYTES)
                .unwrap(),
            new
        );
    }

    #[test]
    fn a_write_with_no_witness_is_refused() {
        let (mut ctx, exec) = setup(
            &[insn(MemoryOp::Write, PolyOp::Reset, 1)],
            witness(&[(0, [1; BIGINT_WIDTH_BYTES])]),
        );
        let err = run(&mut ctx, &exec, |_, _, _| {}).unwrap_err();
        assert!(err.to_string().contains("Missing bigint witness"), "{err}");
    }

    #[test]
    fn a_witness_that_fails_its_check_is_refused() {
        // EqZero over a nonzero witness: the circuit's "Invalid eqz".
        let (mut ctx, exec) = setup(
            &[insn(MemoryOp::Write, PolyOp::EqZero, 0)],
            witness(&[(0, [1; BIGINT_WIDTH_BYTES])]),
        );
        let err = run(&mut ctx, &exec, |_, _, _| {}).unwrap_err();
        assert!(err.to_string().contains("Invalid eqz"), "{err}");

        // A zero one passes, and the program goes on to its Reset.
        let (mut ctx, exec) = setup(
            &[
                insn(MemoryOp::Write, PolyOp::EqZero, 0),
                insn(MemoryOp::Write, PolyOp::Reset, 0),
            ],
            witness(&[(0, [0; BIGINT_WIDTH_BYTES])]),
        );
        assert_eq!(run(&mut ctx, &exec, |_, _, _| {}).unwrap(), 3);
    }

    #[test]
    fn an_undecodable_instruction_is_refused() {
        let (mut ctx, exec) = setup(&[0x0f00_0000], BigIntWitness::new());
        let err = run(&mut ctx, &exec, |_, _, _| {}).unwrap_err();
        assert!(err.to_string().contains("Invalid poly_op"), "{err}");
    }

    #[test]
    fn a_program_longer_than_an_instruction_may_take_is_refused() {
        // Never a Reset: it would run on through memory. Stopped at the per-instruction budget.
        let program = vec![insn(MemoryOp::Read, PolyOp::SetTerm, 0); MAX_RUN_CYCLES];
        let (mut ctx, exec) = setup(&program, BigIntWitness::new());
        ctx.store_region(BASE, &[0; BIGINT_WIDTH_BYTES]).unwrap();
        let (r, seen) = run_counting(&mut ctx, &exec);
        assert!(r.unwrap_err().to_string().contains("runs over"));
        assert_eq!(
            seen, MAX_RUN_CYCLES,
            "it runs the budget, and not one cycle more"
        );

        // One shorter, ending in its Reset, fits exactly: with the machine ecall's cycle, one under
        // the per-instruction bound.
        let mut program = vec![insn(MemoryOp::Read, PolyOp::SetTerm, 0); MAX_RUN_CYCLES - 2];
        program.push(insn(MemoryOp::Read, PolyOp::Reset, 0));
        let (mut ctx, exec) = setup(&program, BigIntWitness::new());
        ctx.store_region(BASE, &[0; BIGINT_WIDTH_BYTES]).unwrap();
        let cycles = run(&mut ctx, &exec, |_, _, _| {}).unwrap();
        assert_eq!(cycles, MAX_RUN_CYCLES);
        assert_eq!(1 + cycles, MAX_INSN_CYCLES - 1);
    }

    /// Reads of zeroed memory, `carries` Carry1s (each adds -2^21 to every coefficient), then a
    /// Read EqZero and the Reset.
    fn carry1_program(carries: usize) -> Vec<u32> {
        let mut program = vec![insn(MemoryOp::Read, PolyOp::Carry1, 0); carries];
        program.push(insn(MemoryOp::Read, PolyOp::EqZero, 0));
        program.push(insn(MemoryOp::Read, PolyOp::Reset, 0));
        program
    }

    #[test]
    fn a_coefficient_that_would_wrap_to_zero_is_refused() {
        // 2,048 Carry1s make every coefficient -2^32: zero, wrapped in 32 bits, which upstream's
        // check passed, and which the prover's accumulator (the circuit's field) does not.
        let (mut ctx, exec) = setup(&carry1_program(2048), BigIntWitness::new());
        ctx.store_region(BASE, &[0; BIGINT_WIDTH_BYTES]).unwrap();
        let err = run(&mut ctx, &exec, |_, _, _| {}).unwrap_err();
        assert!(err.to_string().contains("32-bit range"), "{err}");

        // One leaves -2^21: in range, and not zero.
        let (mut ctx, exec) = setup(&carry1_program(1), BigIntWitness::new());
        ctx.store_region(BASE, &[0; BIGINT_WIDTH_BYTES]).unwrap();
        let err = run(&mut ctx, &exec, |_, _, _| {}).unwrap_err();
        assert!(err.to_string().contains("Invalid eqz"), "{err}");
    }

    #[test]
    fn a_product_that_would_wrap_to_zero_is_refused() {
        // poly = term = 2^16 (Carry1 of [132, 128, ...]), then AddTotal adds 2^32 to the total.
        let mut bytes = [128; BIGINT_WIDTH_BYTES];
        bytes[0] = 132;
        let (mut ctx, exec) = setup(
            &[
                insn(MemoryOp::Read, PolyOp::Carry1, 0),
                insn(MemoryOp::Read, PolyOp::SetTerm, 1),
                insn(MemoryOp::Read, PolyOp::Carry1, 0),
                insn_coeff_reg(MemoryOp::Read, PolyOp::AddTotal, 1, ARENA, 1),
                insn(MemoryOp::Read, PolyOp::EqZero, 1),
                insn(MemoryOp::Read, PolyOp::Reset, 1),
            ],
            BigIntWitness::new(),
        );
        ctx.store_region(BASE, &bytes).unwrap();
        ctx.store_region(BASE + BIGINT_WIDTH_BYTES, &[0; BIGINT_WIDTH_BYTES])
            .unwrap();
        let err = run(&mut ctx, &exec, |_, _, _| {}).unwrap_err();
        assert!(err.to_string().contains("32-bit range"), "{err}");
    }

    #[test]
    fn reads_and_writes_only_where_the_circuit_allows() {
        const REG: usize = REG_A1;
        let at = |addr: u32, mode: u32, mem_op: MemoryOp| {
            let (mut ctx, mut exec) = setup(
                &[insn_coeff_reg(mem_op, PolyOp::Reset, 0, REG, 0)],
                [(ByteAddr(addr).waddr(), [0; BIGINT_WIDTH_BYTES])].into(),
            );
            exec.mode = mode;
            ctx.store_register(MACHINE_REGS_ADDR.waddr(), REG, addr)
                .unwrap();
            ctx.store_region(ByteAddr(addr), &[0; BIGINT_WIDTH_BYTES])
                .unwrap();
            run(&mut ctx, &exec, |_, _, _| {}).map(|_| ())
        };
        for mem_op in [MemoryOp::Read, MemoryOp::Write] {
            for (addr, mode) in [
                (0, 0),
                (0, 1),
                (ZERO_PAGE_END_ADDR.0 - 16, 0),
                (0xbfff_0000, 0),
                (0xbfff_0000, 1),
                (0xc000_0000, 0),
            ] {
                let err = at(addr, mode, mem_op).unwrap_err();
                assert!(
                    err.to_string().contains("Invalid bigint address"),
                    "{mem_op:?} {addr:#x} mode {mode}: {err}"
                );
            }
            for (addr, mode) in [
                (ZERO_PAGE_END_ADDR.0, 0),
                (0xbffe_fff0, 0),
                (0xc000_0000, 1),
                (0xffff_fff0, 1),
            ] {
                at(addr, mode, mem_op).unwrap();
            }
        }

        // A Check names no address (the closing Reset's is x0's, null).
        let (mut ctx, exec) = setup(
            &[insn_coeff_reg(MemoryOp::Check, PolyOp::Reset, 0, 0, 0)],
            BigIntWitness::new(),
        );
        ctx.store_register(MACHINE_REGS_ADDR.waddr(), 0, 0).unwrap();
        run(&mut ctx, &exec, |_, _, _| {}).unwrap();
    }

    #[test]
    fn an_offset_past_one_carry_from_its_base_is_refused() {
        // BASE's low half is 0x1000: offsets up to 0x1EFF keep the sum under 0x20000.
        let at = |offset: u32| {
            let (mut ctx, exec) = setup(
                &[insn(MemoryOp::Read, PolyOp::Reset, offset)],
                BigIntWitness::new(),
            );
            let addr = BASE + offset as usize * BIGINT_WIDTH_BYTES;
            ctx.store_region(addr, &[0; BIGINT_WIDTH_BYTES]).unwrap();
            run(&mut ctx, &exec, |_, _, _| {}).map(|_| ())
        };
        at(0x1eff).unwrap();
        let err = at(0x1f00).unwrap_err();
        assert!(
            err.to_string().contains("out of reach of its base"),
            "{err}"
        );
    }

    #[test]
    fn a_polynomial_grown_past_the_bound_is_refused() {
        // Each Shift adds BIGINT_WIDTH_BYTES coefficients to the 16 the first Read gives.
        let shifts = MAX_POLY_COEFFS / BIGINT_WIDTH_BYTES;
        let mut program = vec![insn(MemoryOp::Read, PolyOp::Shift, 0); shifts];
        program.push(insn(MemoryOp::Read, PolyOp::Reset, 0));
        let (mut ctx, exec) = setup(&program, BigIntWitness::new());
        ctx.store_region(BASE, &[0; BIGINT_WIDTH_BYTES]).unwrap();
        let (r, seen) = run_counting(&mut ctx, &exec);
        assert!(r
            .unwrap_err()
            .to_string()
            .contains("grows a polynomial past"));
        assert_eq!(seen, shifts, "refused at the last shift");

        program.remove(0);
        let (mut ctx, exec) = setup(&program, BigIntWitness::new());
        ctx.store_region(BASE, &[0; BIGINT_WIDTH_BYTES]).unwrap();
        run(&mut ctx, &exec, |_, _, _| {}).unwrap();
    }

    #[test]
    fn multiplications_past_the_bound_are_refused() {
        // AddTotal multiplies poly by term, each grown to `len` coefficients by Shifts.
        let len = MAX_POLY_COEFFS / 2;
        let shifts = len / BIGINT_WIDTH_BYTES - 1;
        let products = (MAX_POLY_MUL_WORK / (len * len) as u64) as usize;
        let mut program = vec![];
        for _ in 0..=products {
            program.extend(vec![insn(MemoryOp::Read, PolyOp::Shift, 0); shifts]);
            program.push(insn(MemoryOp::Read, PolyOp::SetTerm, 0));
            program.extend(vec![insn(MemoryOp::Read, PolyOp::Shift, 0); shifts]);
            program.push(insn(MemoryOp::Read, PolyOp::AddTotal, 0));
        }
        program.push(insn(MemoryOp::Read, PolyOp::Reset, 0));
        let (mut ctx, exec) = setup(&program, BigIntWitness::new());
        ctx.store_region(BASE, &[0; BIGINT_WIDTH_BYTES]).unwrap();
        let (r, seen) = run_counting(&mut ctx, &exec);
        assert!(r
            .unwrap_err()
            .to_string()
            .contains("multiplications take over"));
        // The ecall's cycle and every instruction's but the last multiply's and the Reset's.
        assert_eq!(seen, program.len() - 1, "refused at the last multiply");
    }
}
