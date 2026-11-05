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

pub(crate) mod bigint;
pub(crate) mod byte_poly;
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) mod cuda_stage;
pub(crate) mod paged_map;
pub(crate) mod poseidon2;
pub(crate) mod preflight;
pub(crate) mod sha2;
#[cfg(test)]
mod tests;

use std::{iter::zip, sync::LazyLock};

use anyhow::{Context, Result};

static VERBOSE: LazyLock<bool> = LazyLock::new(|| std::env::var("RISC0_VERBOSE").is_ok());

/// On CUDA and ROCm the witgen kernel writes the five per-cycle columns (CYCLE_COLS) from the
/// uploaded cycles, so the host injector leaves them out (5 of its ~5.1 entries per row at
/// po2=20). Part of the witgen staging feature (`RISC0_CUDA_WITGEN_STAGE=0` keeps them in the
/// injector).
static DEFER_CYCLE_COLS: LazyLock<bool> = LazyLock::new(|| {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    return cuda_stage::enabled();
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    false
});

/// Data columns set for every row from its preflight cycle: cycle, next_pc_low, next_pc_high,
/// next_state, next_machine_mode (in this order; the CUDA kernel takes them as such).
pub(crate) const CYCLE_COLS: [u32; 5] = [
    LAYOUT_TOP.cycle._super.offset as u32,
    LAYOUT_TOP.next_pc_low._super.offset as u32,
    LAYOUT_TOP.next_pc_high._super.offset as u32,
    LAYOUT_TOP.next_state_0._super.offset as u32,
    LAYOUT_TOP.next_machine_mode._super.offset as u32,
];
use preflight::PreflightTrace;
use rayon::prelude::*;
use risc0_binfmt::{PovwNonce, WordAddr};
use risc0_core::scope;
use risc0_zkp::{
    core::digest::DIGEST_WORDS,
    field::{Elem as _, ExtElem as _},
    hal::{Buffer as _, Hal},
};

use self::{
    bigint::BigIntState,
    byte_poly::{BigIntAccum, BigIntAccumState},
    preflight::Back,
};
use super::hal::{CircuitAccumulator, CircuitWitnessGenerator, MetaBuffer, StepMode};
use crate::{
    execute::{
        platform::MERKLE_TREE_END_ADDR, poseidon2::Poseidon2State, segment::Segment,
        sha2::Sha2State,
    },
    zirgen::circuit::{
        CircuitField, ExtVal, Val, LAYOUT_GLOBAL, LAYOUT_TOP, REGCOUNT_ACCUM, REGCOUNT_CODE,
        REGCOUNT_DATA, REGCOUNT_GLOBAL, REGCOUNT_MIX,
    },
};

#[derive(Clone, Default)]
pub struct PreflightResults {
    global: Vec<Val>,
    injector: Injector,
    cycles: usize,
    trace: PreflightTrace,
    po2: u32,
    /// Row indices where Back::BigInt entries occur (pre-extracted for accum).
    bigint_rows: Vec<usize>,
}

impl PreflightResults {
    pub fn new(segment: &Segment, rand_z: ExtVal) -> Result<Self> {
        scope!("preflight_result_new");

        let tp0 = std::time::Instant::now();
        let trace = segment.preflight(rand_z)?;
        let tp1 = std::time::Instant::now();

        tracing::trace!("{segment:#?}");
        tracing::trace!("{trace:#?}");

        let cycles = trace.cycles.len();
        assert!(cycles <= 1 << segment.po2, "cycles <= 1 << segment.po2");
        let cycles = 1 << segment.po2;

        let global = build_global_vec(segment, &trace);
        let mut trace = trace;
        trace.cycle_cols_deferred = *DEFER_CYCLE_COLS;
        let (injector, bigint_rows) = build_injector(&trace, cycles);
        // Stage the trace for the GPU upload here, on the preflight thread, off the critical path.
        let tp2 = std::time::Instant::now();
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        {
            trace.staged = cuda_stage::stage(&trace);
        }
        tracing::debug!(
            "preflight: trace={:.1}ms global+injector={:.1}ms stage={:.1}ms (cycles={} txns={})",
            (tp1 - tp0).as_secs_f64() * 1000.0,
            (tp2 - tp1).as_secs_f64() * 1000.0,
            tp2.elapsed().as_secs_f64() * 1000.0,
            trace.cycles.len(),
            trace.txns.len(),
        );

        Ok(Self {
            global,
            injector,
            cycles,
            trace,
            po2: segment.po2,
            bigint_rows,
        })
    }

    pub fn po2(&self) -> u32 {
        self.po2
    }

    /// The Fiat-Shamir header as far as the segment's claim determines it, from the CPU-side
    /// global vector: `Some` where the claim sets a global, `None` where it leaves it unset
    /// (INVALID), then po2. [`WitnessGenerator::header`] checks the device globals against it.
    ///
    /// Unset globals are not checked: witgen may legitimately fill them. E.g. it writes the
    /// output digest at the end of every segment from guest memory, where a session resumed
    /// after a pause still holds the digest from before it, while the claim has an output only
    /// for the terminating segment; the verifier ignores the output unless the segment
    /// terminates (`Rv32imV2Claim::decode`).
    pub fn claim_header(&self) -> Vec<Option<Val>> {
        use risc0_core::field::Elem as _;
        let mut header: Vec<Option<Val>> = self
            .global
            .iter()
            .map(|elem| elem.is_valid().then_some(*elem))
            .collect();
        header.push(Some(Val::new_raw(self.po2)));
        header
    }
}

pub(crate) struct WitnessGenerator<H: Hal> {
    cycles: usize,
    pub global: MetaBuffer<H>,
    pub code: MetaBuffer<H>,
    /// Present until `accum()` consumes it to produce the witgen.accum buffer.
    /// At po2>=22 it's dropped immediately after step_accum finishes (saves
    /// 3.4 GB of VRAM during commit_group(ACCUM) + eval_check on a 24 GB card).
    pub data: Option<MetaBuffer<H>>,
    /// Accum buffer is allocated lazily inside `accum()` (not in `new()`) so that
    /// its 1.6+ GB footprint at po2=22 doesn't stack with DATA's coeffs + evaluated
    /// allocations during `commit_group(DATA)`. Always Some after `accum()` returns.
    pub accum: Option<MetaBuffer<H>>,
    pub trace: PreflightTrace,
    bigint_rows: Vec<usize>,
}

/// Once a segment's witgen buffers are freed: let the HAL refill them in the background (beside
/// eval_check) for a next segment of the same size, as `hal_generate_witness` (data) and `accum`
/// allocate them. A no-op on HALs without `prefill_elem` (all but HIP), and above po2 21, where
/// eval_check needs that memory back.
pub(crate) fn prefill_next_segment<H, C>(hal: &H, circuit_hal: &C, cycles: usize)
where
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
    C: CircuitWitnessGenerator<H>,
{
    if cycles > 1 << 21 {
        return;
    }
    let zero_data = cfg!(feature = "cuda") || circuit_hal.takes_zeroed_data();
    let data_init = if zero_data { Val::ZERO } else { Val::INVALID };
    hal.prefill_elem(cycles * REGCOUNT_DATA, data_init);
    hal.prefill_elem(cycles * REGCOUNT_ACCUM, Val::INVALID);
}

impl<H> WitnessGenerator<H>
where
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
{
    pub fn cycles(&self) -> usize {
        self.cycles
    }

    pub fn new<C: CircuitWitnessGenerator<H>>(
        hal: &H,
        circuit_hal: &C,
        preflight_results: PreflightResults,
        mode: StepMode,
    ) -> Result<Self> {
        scope!("witness_generator_new");

        let tn0 = std::time::Instant::now();
        let (global, code, data) = Self::hal_generate_witness(
            hal,
            circuit_hal,
            mode,
            &preflight_results.trace,
            preflight_results.global,
            preflight_results.cycles,
            preflight_results.injector,
        )?;
        let tn1 = std::time::Instant::now();

        let result = Self {
            cycles: preflight_results.cycles,
            global,
            code,
            data: Some(data),
            accum: None,
            trace: preflight_results.trace,
            bigint_rows: preflight_results.bigint_rows,
        };
        if *VERBOSE {
            eprintln!(
                "      [witgen_new] hal_gen={:.1}ms struct_build={:.1}ms",
                (tn1 - tn0).as_secs_f64() * 1000.0,
                tn1.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(result)
    }

    /// The Fiat-Shamir header: the globals as witgen left them on the device, then `po2`.
    ///
    /// Witgen writes the state out, output digest and terminate state into the globals,
    /// and eval_check checks the constraints against these device globals, so the header
    /// the verifier checks them against must be read from here too. `expected` is the
    /// header as far as the segment's claim determines it
    /// ([`PreflightResults::claim_header`]). If a global it determines differs, witgen
    /// computed something other than the executor's claim, and the seal won't prove that
    /// claim: that is logged, and asserted in debug builds.
    pub fn header(&self, po2: u32, expected: &[Option<Val>]) -> Vec<Val> {
        let mut header = Vec::with_capacity(self.global.cols + 1);
        // Already zeroized (INVALID -> ZERO) on the device; valid_or_zero is a no-op then.
        self.global
            .buf
            .view(|view| header.extend(view.iter().map(|elem| elem.valid_or_zero())));
        header.push(Val::new_raw(po2));
        let mismatched: Vec<(usize, u32, u32)> = zip(&header, expected)
            .enumerate()
            .filter_map(|(i, (a, b))| match b {
                Some(b) if a != b => Some((i, a.as_u32(), b.as_u32())),
                _ => None,
            })
            .collect();
        if !mismatched.is_empty() || header.len() != expected.len() {
            tracing::warn!(
                "rv32im witgen globals differ from the segment's claim at po2 {po2}: \
                 (index, device, claim) {mismatched:?}; lengths {} vs {}",
                header.len(),
                expected.len()
            );
            debug_assert!(
                false,
                "rv32im witgen globals differ from the segment's claim"
            );
        }
        header
    }

    #[allow(clippy::type_complexity)]
    fn hal_generate_witness<C: CircuitWitnessGenerator<H>>(
        hal: &H,
        circuit_hal: &C,
        mode: StepMode,
        trace: &PreflightTrace,
        global: Vec<Val>,
        cycles: usize,
        injector: Injector,
    ) -> Result<(MetaBuffer<H>, MetaBuffer<H>, MetaBuffer<H>), anyhow::Error> {
        scope!("hal_generate_witness");

        let tw0 = std::time::Instant::now();
        let global = MetaBuffer {
            buf: hal.copy_from_elem("global", &global),
            rows: 1,
            cols: REGCOUNT_GLOBAL,
            checked: true,
        };
        let code = MetaBuffer::new("code", hal, cycles, REGCOUNT_CODE, false);
        let tw1 = std::time::Instant::now();
        // On CUDA (and with circuit HALs that take it, e.g. HIP), zero-fill the data buffer
        // instead of INVALID-filling it, so the zeroize pass over it after witgen can be skipped:
        // the cells witgen doesn't write end up 0 either way (replay-verified: the witness after
        // zeroize is unchanged). Not for the CPU HAL: its checked sets expect INVALID.
        let zero_data = cfg!(feature = "cuda") || circuit_hal.takes_zeroed_data();
        let data = scope!("alloc(data)", {
            if zero_data {
                MetaBuffer {
                    buf: hal.alloc_elem_init("data", cycles * REGCOUNT_DATA, Val::ZERO),
                    rows: cycles,
                    cols: REGCOUNT_DATA,
                    checked: true,
                }
            } else {
                MetaBuffer::new("data", hal, cycles, REGCOUNT_DATA, true)
            }
        });
        let tw2 = std::time::Instant::now();
        // On CDNA (MI300X), allocate a separate read-only pre_data buffer to avoid
        // race conditions in parallel witgen. On consumer GPUs (RDNA4), skip it to
        // save ~844MB VRAM and halve scatter time — eval_check verifies correctness.
        let use_pre_data = std::env::var_os("RISC0_USE_PRE_DATA").is_some();
        let pre_data_buf = if use_pre_data {
            // Zero-init instead of INVALID: witgen back-references into unscattered
            // pre_data positions must resolve to valid Fp; INVALID (0xFFFFFFFF) is
            // non-canonical and can trigger UB in downstream arithmetic.
            let buf = hal.alloc_elem_init("pre_data", cycles * REGCOUNT_DATA, Val::ZERO);
            Some(MetaBuffer {
                buf,
                rows: cycles,
                cols: REGCOUNT_DATA,
                checked: false,
            })
        } else {
            None
        };
        let tw3 = std::time::Instant::now();
        let ts0 = std::time::Instant::now();
        hal.scatter(
            &data.buf,
            &injector.index,
            &injector.offsets,
            &injector.values,
        );
        let ts1 = std::time::Instant::now();
        if let Some(ref pd) = pre_data_buf {
            hal.scatter(
                &pd.buf,
                &injector.index,
                &injector.offsets,
                &injector.values,
            );
        }
        let ts2 = std::time::Instant::now();
        hal.scatter_bits(&data.buf, &injector.bit_data, cycles as u32);
        let ts3 = std::time::Instant::now();
        if let Some(ref pd) = pre_data_buf {
            hal.scatter_bits(&pd.buf, &injector.bit_data, cycles as u32);
        }
        // The injector left out the per-cycle columns: write them from the host where the
        // circuit HAL doesn't (and always into pre_data), after the back entries as before.
        if trace.cycle_cols_deferred {
            let host_data = !circuit_hal.writes_cycle_cols();
            if host_data || pre_data_buf.is_some() {
                let cols = build_cycle_col_injector(trace, cycles);
                if host_data {
                    hal.scatter(&data.buf, &cols.index, &cols.offsets, &cols.values);
                }
                if let Some(ref pd) = pre_data_buf {
                    hal.scatter(&pd.buf, &cols.index, &cols.offsets, &cols.values);
                }
            }
        }
        let tw4 = std::time::Instant::now();
        if *VERBOSE {
            eprintln!("      [scatter_detail] s1={:.1}ms s2={:.1}ms sb1={:.1}ms sb2={:.1}ms total={:.1}ms",
            (ts1-ts0).as_secs_f64()*1000.0,
            (ts2-ts1).as_secs_f64()*1000.0,
            (ts3-ts2).as_secs_f64()*1000.0,
            (tw4-ts3).as_secs_f64()*1000.0,
            (tw4-tw3).as_secs_f64()*1000.0,
        );
        }
        // Drop 126MB of injector Vecs in background while GPU runs generate_witness.
        // After scatter + scatter_bits, all host data has been copied to GPU (sync memcpy).
        let inj_idx = injector.index.len();
        let inj_off = injector.offsets.len();
        let inj_val = injector.values.len();
        let inj_bits = injector.bit_data.len() / 3;
        let _drop_handle = std::thread::spawn(move || drop(injector));
        // When pre_data is disabled, pass &data as pre_data — GPU kernel reads
        // back() values from the same buffer (minor race risk, eval_check validates).
        let pre_data_ref = pre_data_buf.as_ref().unwrap_or(&data);
        circuit_hal
            .generate_witness(mode, trace, &global, &data, pre_data_ref)
            .context("witness generation failure")?;
        let tw5 = std::time::Instant::now();
        scope!("zeroize", {
            hal.eltwise_zeroize_elem(&global.buf);
            hal.eltwise_zeroize_elem(&code.buf);
            if !zero_data {
                hal.eltwise_zeroize_elem(&data.buf);
            }
        });
        let tw6 = std::time::Instant::now();
        if *VERBOSE {
            eprintln!("      [hal_witgen] global+code={:.1}ms data+pre_data={:.1}ms accum={:.1}ms scatter={:.1}ms ffi={:.1}ms zeroize={:.1}ms (inj: idx={} off={} val={} bits={})",
            (tw1-tw0).as_secs_f64()*1000.0,
            (tw2-tw1).as_secs_f64()*1000.0,
            (tw3-tw2).as_secs_f64()*1000.0,
            (tw4-tw3).as_secs_f64()*1000.0,
            (tw5-tw4).as_secs_f64()*1000.0,
            (tw6-tw5).as_secs_f64()*1000.0,
            inj_idx, inj_off, inj_val, inj_bits,
        );
        }
        Ok((global, code, data))
    }

    pub fn accum<C: CircuitAccumulator<H>>(
        &mut self,
        hal: &H,
        circuit_hal: &C,
        mix: &[Val],
    ) -> Result<MetaBuffer<H>> {
        self.accum_with_hook(hal, circuit_hal, mix, false, || ())
            .map(|(mix, ())| mix)
    }

    pub fn accum_with_overlap<C: CircuitAccumulator<H>, R>(
        &mut self,
        hal: &H,
        circuit_hal: &C,
        mix: &[Val],
        overlap: impl FnOnce() -> R,
    ) -> Result<(MetaBuffer<H>, R)> {
        self.accum_with_hook(hal, circuit_hal, mix, true, overlap)
    }

    fn accum_with_hook<C: CircuitAccumulator<H>, R>(
        &mut self,
        hal: &H,
        circuit_hal: &C,
        mix: &[Val],
        concurrent: bool,
        overlap: impl FnOnce() -> R,
    ) -> Result<(MetaBuffer<H>, R)> {
        let ta0 = std::time::Instant::now();

        // Lazily allocate the accum MetaBuffer (~1.6 GB at po2=22). Deferring this
        // until now keeps peak VRAM during `commit_group(DATA)` lower — DATA's coeffs
        // + evaluated already push close to the 24 GB limit on a 4090 at po2=22.
        let accum = MetaBuffer::new("accum", hal, self.cycles, REGCOUNT_ACCUM, true);
        self.accum = Some(accum);
        let accum_buf = self.accum.as_ref().unwrap();

        // use final mix to compute BigIntAccumPowers
        let last_mix = ExtVal::from_subelems(mix[mix.len() - 4..].iter().cloned());

        // inject BigIntAccumState backs
        let mut injector = Injector::new(self.cycles);
        let mut bigint_accum = BigIntAccum::new(last_mix);

        for &row in &self.bigint_rows {
            if let Back::BigInt(idx) = self.trace.backs[row] {
                let state = &self.trace.bigint_states[idx as usize];
                bigint_accum.step(state)?;
                for (col, value) in zip(BigIntAccumState::offsets(), bigint_accum.state.as_array())
                {
                    injector.set(row, col, value);
                }
                injector.push();
            }
        }
        if *VERBOSE {
            eprintln!(
                "    [accum] bigint_inject: {:.1}ms",
                ta0.elapsed().as_secs_f64() * 1000.0
            );
        }

        hal.scatter(
            &accum_buf.buf,
            &injector.index,
            &injector.offsets,
            &injector.values,
        );
        if *VERBOSE {
            eprintln!(
                "    [accum] scatter: {:.1}ms",
                ta0.elapsed().as_secs_f64() * 1000.0
            );
        }

        let mix = MetaBuffer {
            buf: hal.copy_from_elem("mix", mix),
            rows: 1,
            cols: REGCOUNT_MIX,
            checked: true,
        };
        if *VERBOSE {
            eprintln!(
                "    [accum] mix_upload: {:.1}ms",
                ta0.elapsed().as_secs_f64() * 1000.0
            );
        }

        let data_ref = self.data.as_ref().expect("witgen.data missing");
        let overlap_result = circuit_hal.step_accum_with_overlap(
            &self.trace,
            data_ref,
            accum_buf,
            &self.global,
            &mix,
            concurrent,
            overlap,
        )?;
        if *VERBOSE {
            eprintln!(
                "    [accum] step_accum: {:.1}ms",
                ta0.elapsed().as_secs_f64() * 1000.0
            );
        }

        if !circuit_hal.step_accum_zeroizes() {
            scope!("zeroize(accum)", {
                hal.eltwise_zeroize_elem(&accum_buf.buf);
            });
        }
        if *VERBOSE {
            eprintln!(
                "    [accum] zeroize: {:.1}ms",
                ta0.elapsed().as_secs_f64() * 1000.0
            );
        }

        // At po2>=22, drop witgen.data right after step_accum reads it.
        // Frees 3.4 GB that's otherwise live through commit_group(ACCUM)
        // and eval_check. No other path reads it on the current HAL.
        if self.cycles >= (1 << 22) {
            self.data = None;
        }

        Ok((mix, overlap_result))
    }
}

/// Serial preflight post-processing on the preflight thread (RISC0_CUDA_SERIAL_PREFLIGHT=0
/// restores the rayon versions): the injector build when the cycle columns are deferred, and the pinned
/// staging copy (`cuda_stage`). The preflight thread runs beside the proving thread's own rayon
/// work, and on a loaded host the rayon versions waited for busy workers: at +32 runnable threads
/// the parallel injector took 54-106 ms vs 2.6 ms serial (only ~0.1 entries per row remain once
/// the cycle columns are deferred), and the parallel stage copy 29.6 vs 15.5 ms. On a quiet host
/// the parallel versions win by under 2 ms, off the critical path.
pub(crate) static SERIAL_PREFLIGHT: LazyLock<bool> =
    LazyLock::new(|| std::env::var("RISC0_CUDA_SERIAL_PREFLIGHT").as_deref() != Ok("0"));

fn build_injector(trace: &PreflightTrace, cycles: usize) -> (Injector, Vec<usize>) {
    scope!("build_injector");
    if trace.cycle_cols_deferred && *SERIAL_PREFLIGHT {
        return build_injector_serial(trace, cycles);
    }
    build_injector_parallel(trace, cycles)
}

fn build_injector_parallel(trace: &PreflightTrace, cycles: usize) -> (Injector, Vec<usize>) {
    const CYCLE_COL: usize = LAYOUT_TOP.cycle._super.offset;
    const NEXT_PC_LOW: usize = LAYOUT_TOP.next_pc_low._super.offset;
    const NEXT_PC_HIGH: usize = LAYOUT_TOP.next_pc_high._super.offset;
    const NEXT_STATE: usize = LAYOUT_TOP.next_state_0._super.offset;
    const NEXT_MACHINE_MODE: usize = LAYOUT_TOP.next_machine_mode._super.offset;
    const ECALL_S0: usize = LAYOUT_TOP.inst_result.arm8.s0._super.offset;
    const ECALL_S1: usize = LAYOUT_TOP.inst_result.arm8.s1._super.offset;
    const ECALL_S2: usize = LAYOUT_TOP.inst_result.arm8.s2._super.offset;
    const SET_CYCLE_COUNT: u32 = 5;

    let n = trace.backs.len();
    let cycle_cols = !trace.cycle_cols_deferred;
    let set_cycle_count = if cycle_cols { SET_CYCLE_COUNT } else { 0 };

    // Steps 1+2: Compute exclusive prefix sums in a single pass.
    // scatter entries (without SHA2 u32 bit decomposition — those go to bit_data)
    let mut index = Vec::with_capacity(n + 1);
    advise_huge(&index);
    index.push(0u32);
    let mut total = 0u32;
    // bit entries: each SHA2 u32 becomes one packed (row, base_col, value) triple
    let mut bit_index = Vec::with_capacity(n + 1);
    advise_huge(&bit_index);
    bit_index.push(0u32);
    let mut bit_total = 0u32;
    // Pre-extract BigInt row indices for accum phase (avoids 1M-entry scan later)
    let mut bigint_rows = Vec::new();
    for (row, back) in trace.backs.iter().enumerate() {
        total += set_cycle_count
            + match back {
                Back::None => 0,
                Back::Ecall(..) => 3,
                Back::Poseidon2(_) => Poseidon2State::offsets().len() as u32,
                Back::Sha2(_) => Sha2State::fp_offsets().len() as u32, // u32s moved to bit_data
                Back::BigInt(_) => BigIntState::offsets().len() as u32,
            };
        index.push(total);
        bit_total += match back {
            Back::Sha2(_) => Sha2State::u32_offsets().len() as u32, // 3 u32s per SHA2 cycle
            _ => 0,
        };
        bit_index.push(bit_total);
        if matches!(back, Back::BigInt(_)) {
            bigint_rows.push(row);
        }
    }
    let total = total as usize;
    let bit_total = bit_total as usize;

    // Step 3: Pre-allocate output arrays (skip zero-fill since every element is overwritten)
    let mut offsets = Vec::<u32>::with_capacity(total);
    let mut values = Vec::<Val>::with_capacity(total);
    advise_huge(&offsets);
    advise_huge(&values);
    // SAFETY: All elements [0..total) will be written by the parallel fill below.
    unsafe {
        offsets.set_len(total);
        values.set_len(total);
    }
    // Packed (row, base_col, value) triples for GPU-side bit decomposition
    let mut bit_data = Vec::<u32>::with_capacity(bit_total * 3);
    advise_huge(&bit_data);
    unsafe {
        bit_data.set_len(bit_total * 3);
    }

    // Step 4: Fill in parallel (each row writes to non-overlapping region)
    let offsets_ptr = offsets.as_mut_ptr();
    let values_ptr = values.as_mut_ptr();
    let bit_data_ptr = bit_data.as_mut_ptr();

    // SAFETY: Each thread writes to [index[row]..index[row+1]), which are
    // non-overlapping ranges determined by the prefix sum above.
    let offsets_send = offsets_ptr as usize;
    let values_send = values_ptr as usize;
    let bit_data_send = bit_data_ptr as usize;

    trace.backs.par_iter().enumerate().for_each(|(row, back)| {
        let cycle = &trace.cycles[row];
        let mut pos = index[row] as usize;
        let mut bit_pos = bit_index[row] as usize;

        let offsets_p = offsets_send as *mut u32;
        let values_p = values_send as *mut Val;
        let bit_data_p = bit_data_send as *mut u32;

        let mut set = |col: usize, value: u32| {
            let idx = col * cycles + row;
            unsafe {
                *offsets_p.add(pos) = idx as u32;
                *values_p.add(pos) = Val::new(value);
            }
            pos += 1;
        };

        let mut set_bits = |col: usize, value: u32| {
            unsafe {
                *bit_data_p.add(bit_pos * 3) = row as u32;
                *bit_data_p.add(bit_pos * 3 + 1) = col as u32;
                *bit_data_p.add(bit_pos * 3 + 2) = value;
            }
            bit_pos += 1;
        };

        match back {
            Back::None => {}
            Back::Ecall(s0, s1, s2) => {
                set(ECALL_S0, *s0);
                set(ECALL_S1, *s1);
                set(ECALL_S2, *s2);
            }
            Back::Poseidon2(idx) => {
                let p2_state = &trace.p2_states[*idx as usize];
                for (col, value) in zip(Poseidon2State::offsets(), p2_state.as_array()) {
                    set(col, value);
                }
            }
            Back::Sha2(idx) => {
                let sha2_state = &trace.sha2_states[*idx as usize];
                for (col, value) in zip(Sha2State::fp_offsets(), sha2_state.fp_array()) {
                    set(col, value);
                }
                for (col, value) in zip(Sha2State::u32_offsets(), sha2_state.u32_array()) {
                    set_bits(col, value);
                }
            }
            Back::BigInt(idx) => {
                let state = &trace.bigint_states[*idx as usize];
                for (col, value) in zip(BigIntState::offsets(), state.as_array()) {
                    set(col, value);
                }
            }
        }

        if cycle_cols {
            set(CYCLE_COL, row as u32);
            set(NEXT_PC_LOW, cycle.pc & 0xffff);
            set(NEXT_PC_HIGH, cycle.pc >> 16);
            set(NEXT_STATE, cycle.state);
            set(NEXT_MACHINE_MODE, cycle.machine_mode as u32);
        }
    });

    // Rows without entries (most of them once the cycle columns are left out) add nothing to
    // the scatter but one thread each: drop their (repeated) bounds. Same writes in the same
    // per-row order; the scatter kernel never needs the row number.
    if !cycle_cols {
        index.dedup();
    }

    (
        Injector {
            rows: cycles,
            offsets,
            values,
            index,
            bit_data,
        },
        bigint_rows,
    )
}

/// [`build_injector`] without the cycle columns, in one serial pass: the same entries in the same
/// (row) order, and the same deduplicated row bounds.
fn build_injector_serial(trace: &PreflightTrace, cycles: usize) -> (Injector, Vec<usize>) {
    const ECALL_S0: usize = LAYOUT_TOP.inst_result.arm8.s0._super.offset;
    const ECALL_S1: usize = LAYOUT_TOP.inst_result.arm8.s1._super.offset;
    const ECALL_S2: usize = LAYOUT_TOP.inst_result.arm8.s2._super.offset;
    debug_assert!(trace.cycle_cols_deferred);

    let mut offsets = Vec::<u32>::new();
    let mut values = Vec::<Val>::new();
    let mut bit_data = Vec::<u32>::new();
    let mut bigint_rows = Vec::new();
    // Row bounds, deduplicated as `index.dedup()` does in the parallel version: a bound is
    // pushed only when it differs from the previous one (rows with entries end a run).
    let mut index = vec![0u32];
    for (row, back) in trace.backs.iter().enumerate() {
        let mut set = |col: usize, value: u32| {
            offsets.push((col * cycles + row) as u32);
            values.push(Val::new(value));
        };
        match back {
            Back::None => continue,
            Back::Ecall(s0, s1, s2) => {
                set(ECALL_S0, *s0);
                set(ECALL_S1, *s1);
                set(ECALL_S2, *s2);
            }
            Back::Poseidon2(idx) => {
                let p2_state = &trace.p2_states[*idx as usize];
                for (col, value) in zip(Poseidon2State::offsets(), p2_state.as_array()) {
                    set(col, value);
                }
            }
            Back::Sha2(idx) => {
                let sha2_state = &trace.sha2_states[*idx as usize];
                for (col, value) in zip(Sha2State::fp_offsets(), sha2_state.fp_array()) {
                    set(col, value);
                }
                for (col, value) in zip(Sha2State::u32_offsets(), sha2_state.u32_array()) {
                    bit_data.extend_from_slice(&[row as u32, col as u32, value]);
                }
            }
            Back::BigInt(idx) => {
                bigint_rows.push(row);
                let state = &trace.bigint_states[*idx as usize];
                for (col, value) in zip(BigIntState::offsets(), state.as_array()) {
                    set(col, value);
                }
            }
        }
        let end = offsets.len() as u32;
        if *index.last().unwrap() != end {
            index.push(end);
        }
    }

    (
        Injector {
            rows: cycles,
            offsets,
            values,
            index,
            bit_data,
        },
        bigint_rows,
    )
}

/// The per-cycle columns (CYCLE_COLS) as an injector, for HALs that don't write them.
fn build_cycle_col_injector(trace: &PreflightTrace, cycles: usize) -> Injector {
    let n = trace.cycles.len();
    let index: Vec<u32> = (0..=n).map(|row| (row * CYCLE_COLS.len()) as u32).collect();
    let mut offsets = vec![0u32; n * CYCLE_COLS.len()];
    let mut values = vec![Val::ZERO; n * CYCLE_COLS.len()];
    offsets
        .par_chunks_mut(CYCLE_COLS.len())
        .zip(values.par_chunks_mut(CYCLE_COLS.len()))
        .enumerate()
        .for_each(|(row, (offs, vals))| {
            let cycle = &trace.cycles[row];
            let v = [
                row as u32,
                cycle.pc & 0xffff,
                cycle.pc >> 16,
                cycle.state,
                cycle.machine_mode as u32,
            ];
            for i in 0..CYCLE_COLS.len() {
                offs[i] = (CYCLE_COLS[i] as usize * cycles + row) as u32;
                vals[i] = Val::new(v[i]);
            }
        });
    Injector {
        rows: cycles,
        offsets,
        values,
        index,
        bit_data: vec![],
    }
}

fn build_global_vec(segment: &Segment, trace: &PreflightTrace) -> Vec<Val> {
    scope!("build_global_vec");

    let mut global = vec![Val::INVALID; REGCOUNT_GLOBAL];

    // state in
    for (i, word) in segment.claim.pre_state.as_words().iter().enumerate() {
        let low = word & 0xffff;
        let high = word >> 16;
        global[LAYOUT_GLOBAL.state_in.values[i].low._super.offset] = low.into();
        global[LAYOUT_GLOBAL.state_in.values[i].high._super.offset] = high.into();
    }

    // input digest
    for (i, word) in segment.claim.input.as_words().iter().enumerate() {
        let low = word & 0xffff;
        let high = word >> 16;
        global[LAYOUT_GLOBAL.input.values[i].low._super.offset] = low.into();
        global[LAYOUT_GLOBAL.input.values[i].high._super.offset] = high.into();
    }

    // rand_z
    for (i, &elem) in trace.rand_z.elems().iter().enumerate() {
        global[LAYOUT_GLOBAL.rng._super.offset + i] = elem;
    }

    // is_terminate
    let is_terminate = if segment.claim.terminate_state.is_some() {
        1u32
    } else {
        0u32
    };
    global[LAYOUT_GLOBAL.is_terminate._super.offset] = is_terminate.into();

    // shutdown_cycle
    global[LAYOUT_GLOBAL.shutdown_cycle._super.offset] = segment.segment_threshold.into();

    // state out (written by GPU witgen in exec_ControlStoreRoot)
    for (i, word) in segment.claim.post_state.as_words().iter().enumerate() {
        let low = word & 0xffff;
        let high = word >> 16;
        global[LAYOUT_GLOBAL.state_out.values[i].low._super.offset] = low.into();
        global[LAYOUT_GLOBAL.state_out.values[i].high._super.offset] = high.into();
    }

    // output digest (written by GPU witgen in exec_ControlSuspend)
    if let Some(output) = &segment.claim.output {
        for (i, word) in output.as_words().iter().enumerate() {
            let low = word & 0xffff;
            let high = word >> 16;
            global[LAYOUT_GLOBAL.output.values[i].low._super.offset] = low.into();
            global[LAYOUT_GLOBAL.output.values[i].high._super.offset] = high.into();
        }
    }

    // terminate state (written by GPU witgen in exec_ECallTerminate/exec_ControlSuspend)
    if let Some(ts) = &segment.claim.terminate_state {
        let a0: u32 = ts.a0.into();
        let a1: u32 = ts.a1.into();
        global[LAYOUT_GLOBAL.term_a0low._super.offset] = (a0 & 0xffff).into();
        global[LAYOUT_GLOBAL.term_a0high._super.offset] = (a0 >> 16).into();
        global[LAYOUT_GLOBAL.term_a1low._super.offset] = (a1 & 0xffff).into();
        global[LAYOUT_GLOBAL.term_a1high._super.offset] = (a1 >> 16).into();
    }

    // povw nonce
    // Split the U256 nonce into LE shorts and assign to the globals.
    let nonce = segment.povw_nonce.unwrap_or(PovwNonce::ZERO);
    for (i, short) in nonce.to_u16s().into_iter().enumerate() {
        match i % 2 {
            0 => {
                global[LAYOUT_GLOBAL.povw_nonce.values[i / 2].low._super.offset] =
                    Val::from_u64(short as u64);
            }
            1 => {
                global[LAYOUT_GLOBAL.povw_nonce.values[i / 2].high._super.offset] =
                    Val::from_u64(short as u64);
            }
            _ => unreachable!(),
        }
    }

    global
}

#[derive(Clone, Debug, Default)]
struct Injector {
    rows: usize,
    offsets: Vec<u32>,
    values: Vec<Val>,
    index: Vec<u32>,
    /// Packed (row, base_col, value) triples for GPU-side u32→bits decomposition.
    bit_data: Vec<u32>,
}

impl Injector {
    fn new(rows: usize) -> Self {
        let mut index = Vec::with_capacity(rows + 1);
        index.push(0);
        Self {
            rows,
            offsets: vec![],
            values: vec![],
            index,
            bit_data: vec![],
        }
    }

    fn push(&mut self) {
        self.index.push(self.offsets.len() as u32);
    }

    fn set(&mut self, row: usize, col: usize, value: u32) {
        let idx = col * self.rows + row;
        self.offsets.push(idx as u32);
        self.values.push(value.into());
    }
}

fn node_addr_to_idx(addr: WordAddr) -> u32 {
    (MERKLE_TREE_END_ADDR - addr).0 / DIGEST_WORDS as u32
}

fn node_idx_to_addr(idx: u32) -> WordAddr {
    MERKLE_TREE_END_ADDR - idx * DIGEST_WORDS as u32
}

/// Advise the kernel to back a freshly allocated (not yet touched) vector with transparent huge
/// pages. The preflight trace and injector vectors are ~250 MB at po2=20, written once per segment;
/// with 4 KB pages their page faults were ~half of preflight time (preflight body 80 -> 42 ms at
/// po2=20), and the first segment's preflight is on the critical path. Huge pages also make the
/// background munmap of the trace short, which otherwise holds the process mmap lock for 10-30 ms
/// and stalls concurrent driver calls that map memory. Only whole 2 MB pages inside the
/// allocation are advised; contents are unaffected. RISC0_CUDA_THP=0 disables it. Linux only.
pub(crate) fn advise_huge<T>(v: &Vec<T>) {
    #[cfg(target_os = "linux")]
    {
        static ON: LazyLock<bool> =
            LazyLock::new(|| std::env::var("RISC0_CUDA_THP").as_deref() != Ok("0"));
        const HUGE: usize = 2 << 20;
        const MADV_HUGEPAGE: std::os::raw::c_int = 14;
        extern "C" {
            fn madvise(
                addr: *mut std::os::raw::c_void,
                len: usize,
                advice: std::os::raw::c_int,
            ) -> std::os::raw::c_int;
        }
        let bytes = v.capacity() * std::mem::size_of::<T>();
        if !*ON || bytes < 2 * HUGE {
            return;
        }
        let start = v.as_ptr() as usize;
        let lo = (start + HUGE - 1) & !(HUGE - 1);
        let hi = (start + bytes) & !(HUGE - 1);
        if hi > lo {
            // SAFETY: [lo, hi) lies inside the vector's allocation; MADV_HUGEPAGE only changes
            // how the kernel backs the pages, never their contents.
            unsafe {
                madvise(lo as *mut std::os::raw::c_void, hi - lo, MADV_HUGEPAGE);
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = v;
}
