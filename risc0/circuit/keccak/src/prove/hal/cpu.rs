// Copyright 2024 RISC Zero, Inc.
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

use std::rc::Rc;

use anyhow::Result;
use rayon::{iter::IntoParallelIterator, prelude::*};
use risc0_circuit_keccak_sys::{
    risc0_circuit_keccak_cpu_poly_fp, risc0_circuit_keccak_cpu_witgen, RawBuffer, RawExecBuffers,
    RawPreflightTrace, ScatterInfo,
};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    core::{hash::poseidon2::Poseidon2HashSuite, log2_ceil},
    field::{map_pow, Elem as _, ExtElem as _, RootsOfUnity as _},
    hal::{
        cpu::{CpuBuffer, CpuHal},
        AccumPreflight, CircuitHal,
    },
    INV_RATE,
};

use crate::{
    prove::{KeccakProver, KeccakProverImpl, GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{CircuitField, ExtVal, Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_DATA},
        info::POLY_MIX_POWERS,
    },
};

use super::{
    CircuitWitnessGenerator, ForwardPreflightOrder, MetaBuffer, PreflightCycleOrder,
    PreflightTrace, StepMode,
};

#[derive(Default)]
pub struct CpuCircuitHal;

impl CircuitWitnessGenerator<CpuHal<CircuitField>> for CpuCircuitHal {
    type PreferredPreflightOrder = ForwardPreflightOrder;
    const HOST_WITNESS: bool = true;

    fn scatter_preflight(
        &self,
        into: &MetaBuffer<CpuHal<CircuitField>>,
        infos: &[ScatterInfo],
        data: &[u32],
    ) -> Result<()> {
        scope!("scatter");
        scatter_preflight_host(&mut into.buf.as_slice_mut(), into.rows, infos, data);
        Ok(())
    }

    fn generate_witness<O: PreflightCycleOrder>(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace<O>,
        global: &MetaBuffer<CpuHal<CircuitField>>,
        data: &MetaBuffer<CpuHal<CircuitField>>,
    ) -> Result<()> {
        scope!("witgen");
        generate_witness_host(
            mode,
            preflight,
            HostMetaBuffer::of(global, &mut global.buf.as_slice_mut()),
            HostMetaBuffer::of(data, &mut data.buf.as_slice_mut()),
        )
    }
}

/// A host buffer of the witness, with its shape (see [MetaBuffer]).
pub(crate) struct HostMetaBuffer<'a> {
    pub buf: &'a mut [Val],
    pub rows: usize,
    pub cols: usize,
    pub checked_reads: bool,
}

impl<'a> HostMetaBuffer<'a> {
    pub fn of<H: risc0_zkp::hal::Hal>(meta: &MetaBuffer<H>, buf: &'a mut [Val]) -> Self {
        assert_eq!(buf.len(), meta.rows * meta.cols);
        Self {
            buf,
            rows: meta.rows,
            cols: meta.cols,
            checked_reads: meta.checked_reads,
        }
    }
}

/// Scatter the preflight's `data` into `into`, a witness buffer of `rows` rows, on the CPU.
///
/// In parallel, as CUDA does it (a thread per info): the infos set disjoint cells, and their writes
/// land a column apart, so one thread spent 0.8 s at po2 17 waiting on memory.
pub(crate) fn scatter_preflight_host(
    into: &mut [Val],
    rows: usize,
    infos: &[ScatterInfo],
    data: &[u32],
) {
    struct Cells(*mut Val, usize);
    // SAFETY: each cell is written by one info only, and `into` outlives the parallel loop.
    unsafe impl Sync for Cells {}
    let cells = Cells(into.as_mut_ptr(), into.len());
    infos.par_chunks(1024).for_each(|chunk| {
        let cells = &cells;
        for info in chunk {
            let inner_count = 32 / info.bits;
            let mask: u32 = (1 << info.bits) - 1;
            for i in 0..info.count as u32 {
                let from_idx = info.offset + (i / inner_count);
                let word = data[from_idx as usize];
                let j = i % inner_count;
                let val = (word >> (j * info.bits)) & mask;
                let col = info.col as u32 + i;
                let into_idx = col as usize * rows + info.row as usize;
                assert!(into_idx < cells.1);
                // SAFETY: in bounds (checked above), and no other info writes this cell.
                unsafe { *cells.0.add(into_idx) = val.into() };
            }
        }
    });
}

/// Generate the witness into host buffers, on the CPU.
pub(crate) fn generate_witness_host<O: PreflightCycleOrder>(
    mode: StepMode,
    preflight: &PreflightTrace<O>,
    global: HostMetaBuffer,
    data: HostMetaBuffer,
) -> Result<()> {
    let cycles = preflight.cycle;
    assert_eq!(cycles, data.rows);
    tracing::debug!("witgen: {cycles}");

    let buffers = RawExecBuffers {
        global: RawBuffer {
            buf: global.buf.as_ptr(),
            rows: global.rows,
            cols: global.cols,
            checked_reads: global.checked_reads,
        },
        data: RawBuffer {
            buf: data.buf.as_ptr(),
            rows: data.rows,
            cols: data.cols,
            checked_reads: data.checked_reads,
        },
    };
    let run_order: Vec<u32> = preflight
        .cycles
        .values()
        .flatten()
        .map(|c| c.cycle)
        .collect();
    let preimage_idx: Vec<u32> = preflight
        .cycles
        .values()
        .flatten()
        .map(|c| c.preimage_idx)
        .collect();
    let preflight = RawPreflightTrace {
        preimages: preflight.preimages.as_ptr(),
        preimages_count: preflight.preimages.len() as u32,
        preimage_idxs: preimage_idx.as_ptr(),
        run_order: run_order.as_ptr(),
    };
    ffi_wrap(|| unsafe {
        risc0_circuit_keccak_cpu_witgen(mode as u32, &buffers, &preflight, cycles as u32)
    })
}

impl CircuitHal<CpuHal<CircuitField>> for CpuCircuitHal {
    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &CpuBuffer<Val>,
        _io: &CpuBuffer<Val>,
        _data: &CpuBuffer<Val>,
        _mix: &CpuBuffer<Val>,
        _accum: &CpuBuffer<Val>,
        _steps: usize,
    ) {
    }

    fn eval_check(
        &self,
        check: &CpuBuffer<Val>,
        groups: &[&CpuBuffer<Val>],
        globals: &[&CpuBuffer<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) {
        scope!("eval_check");

        let check = check.as_slice();
        let accum = groups[REGISTER_GROUP_ACCUM].as_slice();
        let data = groups[REGISTER_GROUP_DATA].as_slice();
        let mix = globals[GLOBAL_MIX].as_slice();
        let out = globals[GLOBAL_OUT].as_slice();

        tracing::debug!(
            "check: {}, data: {}, accum: {}, mix: {} out: {}",
            check.len(),
            data.len(),
            accum.len(),
            mix.len(),
            out.len()
        );

        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);

        // SAFETY: Convert a borrow of a cell into a raw const slice so that we can pass
        // it over the thread boundary. This should be safe because the scope of the
        // usage is within this function and each thread access will not overlap with
        // each other.

        let data = unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) };
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr(), out.len()) };
        let poly_mix_pows = poly_mix_pows.as_slice();

        (0..domain).into_par_iter().for_each(|cycle| {
            let args = &[data.as_ptr(), out.as_ptr()];
            let mut tot = ExtVal::ZERO;
            unsafe {
                risc0_circuit_keccak_cpu_poly_fp(
                    cycle,
                    domain,
                    poly_mix_pows.as_ptr(),
                    args.as_ptr(),
                    &mut tot,
                )
            };
            let x = Val::ROU_FWD[po2 + EXP_PO2].pow(cycle);
            // TODO: what is this magic number 3?
            let y = (Val::new(3) * x).pow(1 << po2);
            let ret = tot * (y - Val::new(1)).inv();

            // SAFETY: This conversion is to make the check slice mutable, which should be
            // safe because each thread access will not overlap with each other.
            let check =
                unsafe { std::slice::from_raw_parts_mut(check.as_ptr() as *mut Val, check.len()) };
            for i in 0..ExtVal::EXT_SIZE {
                check[i * domain + cycle] = ret.elems()[i];
            }
        });
    }
}

#[allow(dead_code)]
pub fn keccak_prover() -> Result<Box<dyn KeccakProver>> {
    let hash_suite = Poseidon2HashSuite::new_suite();
    let hal = Rc::new(CpuHal::new(hash_suite));
    let circuit_hal = Rc::new(CpuCircuitHal);
    Ok(Box::new(KeccakProverImpl { hal, circuit_hal }))
}
