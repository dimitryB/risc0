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

use std::sync::Arc;

use anyhow::{bail, Result};
use risc0_circuit_recursion_sys::{
    risc0_circuit_recursion_cuda_accum, risc0_circuit_recursion_cuda_eval_check,
    risc0_circuit_recursion_cuda_witgen, risc0_circuit_recursion_cuda_witgen_chains,
    risc0_circuit_recursion_hip_fill_tail, risc0_circuit_recursion_hip_init_rows,
    risc0_circuit_recursion_hip_zeroize_rows, RawAccumBuffers, RawExecBuffers, RawHostChains,
    RawPreflightTrace, StepMode,
};
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    core::log2_ceil,
    field::{
        baby_bear::{BabyBearElem, BabyBearExtElem},
        map_pow, RootsOfUnity,
    },
    hal::{
        hip::{
            BufferImpl as HipBuffer, HipHal, HipHalPoseidon2, HipHalPoseidon254, HipHalSha256,
            HipHash, HipHashPoseidon2, HipHashPoseidon254, HipHashSha256,
        },
        AccumPreflight, Buffer, CircuitHal,
    },
    INV_RATE,
};

use crate::{
    prove::{RecursionProver, RecursionProverImpl},
    GLOBAL_MIX, GLOBAL_OUT, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CTRL, REGISTER_GROUP_DATA,
};

use super::{CircuitAccumulator, CircuitWitnessGenerator};

type HipCircuitHalSha256 = HipCircuitHal<HipHashSha256>;
type HipCircuitHalPoseidon2 = HipCircuitHal<HipHashPoseidon2>;
type HipCircuitHalPoseidon254 = HipCircuitHal<HipHashPoseidon254>;

struct HipCircuitHal<CH: HipHash> {
    _hal: Arc<HipHal<CH>>, // retain a reference to ensure the context remains valid
}

impl<CH: HipHash> HipCircuitHal<CH> {
    pub fn new(_hal: Arc<HipHal<CH>>) -> Self {
        risc0_zkp::hal::hip::register_release_hook(super::super::pinned::release);
        Self { _hal }
    }
}

impl<CH: HipHash> CircuitWitnessGenerator<HipHal<CH>> for HipCircuitHal<CH> {
    fn generate_witness(
        &self,
        mode: StepMode,
        total_cycles: u32,
        preflight: &RawPreflightTrace,
        ctrl: &HipBuffer<BabyBearElem>,
        data: &HipBuffer<BabyBearElem>,
        global: &HipBuffer<BabyBearElem>,
    ) -> Result<()> {
        let ctrl = ctrl.as_device_ptr();
        let data = data.as_device_ptr();
        let global = global.as_device_ptr();
        let buffers = RawExecBuffers {
            ctrl: ctrl.as_ptr() as *const BabyBearElem,
            data: data.as_ptr() as *const BabyBearElem,
            global: global.as_ptr() as *const BabyBearElem,
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cuda_witgen(mode, &buffers, preflight, total_cycles)
        })
    }

    fn generate_witness_with_chains(
        &self,
        mode: StepMode,
        total_cycles: u32,
        preflight: &RawPreflightTrace,
        host_chains: Option<&crate::prove::witgen::HostChains>,
        ctrl: &HipBuffer<BabyBearElem>,
        data: &HipBuffer<BabyBearElem>,
        global: &HipBuffer<BabyBearElem>,
    ) -> Result<()> {
        let Some(hc) = host_chains else {
            return self.generate_witness(mode, total_cycles, preflight, ctrl, data, global);
        };
        let ctrl = ctrl.as_device_ptr();
        let data = data.as_device_ptr();
        let global = global.as_device_ptr();
        let buffers = RawExecBuffers {
            ctrl: ctrl.as_ptr() as *const BabyBearElem,
            data: data.as_ptr() as *const BabyBearElem,
            global: global.as_ptr() as *const BabyBearElem,
        };
        // The page-locked copies, if the preflight staged them (see `pinned`).
        let p = hc.pinned.unwrap_or([
            hc.heads.as_ptr() as usize,
            hc.cycles.as_ptr() as usize,
            hc.rows.as_ptr() as usize,
            hc.wom_idx.as_ptr() as usize,
            hc.wom_rows.as_ptr() as usize,
        ]);
        let raw = RawHostChains {
            heads: p[0] as *const u32,
            num_heads: hc.heads.len() as u32,
            cycles: p[1] as *const u32,
            rows: p[2] as *const u32,
            wom_idx: p[3] as *const u32,
            wom_rows: p[4] as *const u32,
            num_rows: hc.cycles.len() as u32,
            data_cols: (hc.rows.len() / hc.cycles.len().max(1)) as u32,
        };
        // `hc` outlives the call, which synchronizes before returning.
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cuda_witgen_chains(
                mode,
                &buffers,
                preflight,
                total_cycles,
                &raw,
            )
        })
    }

    // RISC0_HIP_REC_INIT=0: the generic HAL passes (fill, noise upload + region copy, zeroize).
    fn has_row_passes(&self) -> bool {
        static ON: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var("RISC0_HIP_REC_INIT").as_deref() != Ok("0"));
        *ON
    }

    fn init_rows(&self, buf: &HipBuffer<BabyBearElem>, rows: usize, live: usize) -> Result<()> {
        let (log_rows, cols) = row_shape(buf, rows)?;
        let p = buf.as_device_ptr().as_ptr() as *mut u32;
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_hip_init_rows(p, log_rows, live.min(rows) as u32, cols)
        })
    }

    fn fill_tail(
        &self,
        buf: &HipBuffer<BabyBearElem>,
        rows: usize,
        zk: usize,
        value: BabyBearElem,
    ) -> Result<()> {
        let (log_rows, cols) = row_shape(buf, rows)?;
        if zk > rows {
            bail!("fill_tail: {zk} rows of {rows}");
        }
        let p = buf.as_device_ptr().as_ptr() as *mut u32;
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_hip_fill_tail(
                p,
                value.as_u32_montgomery(),
                log_rows,
                zk as u32,
                cols,
            )
        })
    }

    fn zeroize_rows(&self, buf: &HipBuffer<BabyBearElem>, rows: usize, live: usize) -> Result<()> {
        let (log_rows, cols) = row_shape(buf, rows)?;
        let p = buf.as_device_ptr().as_ptr() as *mut u32;
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_hip_zeroize_rows(p, log_rows, live.min(rows) as u32, cols)
        })
    }
}

/// (log2 rows, columns) of a column-major buffer of `rows` (a power of two, >= 4) rows per column.
fn row_shape(buf: &HipBuffer<BabyBearElem>, rows: usize) -> Result<(u32, u32)> {
    if !rows.is_power_of_two() || rows < 4 || buf.size() % rows != 0 {
        bail!("row passes: {} elements in rows of {rows}", buf.size());
    }
    Ok((rows.trailing_zeros(), (buf.size() / rows) as u32))
}

impl<CH: HipHash> CircuitAccumulator<HipHal<CH>> for HipCircuitHal<CH> {
    fn accumulate(
        &self,
        work_cycles: u32,
        total_cycles: u32,
        ctrl: &HipBuffer<BabyBearElem>,
        global: &HipBuffer<BabyBearElem>,
        data: &HipBuffer<BabyBearElem>,
        mix: &HipBuffer<BabyBearElem>,
        accum: &HipBuffer<BabyBearElem>,
    ) -> Result<()> {
        let ctrl = ctrl.as_device_ptr();
        let global = global.as_device_ptr();
        let data = data.as_device_ptr();
        let mix = mix.as_device_ptr();
        let accum = accum.as_device_ptr();
        let buffers = RawAccumBuffers {
            ctrl: ctrl.as_ptr() as *const BabyBearElem,
            global: global.as_ptr() as *const BabyBearElem,
            data: data.as_ptr() as *const BabyBearElem,
            mix: mix.as_ptr() as *const BabyBearElem,
            accum: accum.as_ptr() as *const BabyBearElem,
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cuda_accum(&buffers, work_cycles, total_cycles)
        })
    }
}

impl<CH: HipHash> CircuitHal<HipHal<CH>> for HipCircuitHal<CH> {
    fn eval_check(
        &self,
        check: &HipBuffer<BabyBearElem>,
        groups: &[&HipBuffer<BabyBearElem>],
        globals: &[&HipBuffer<BabyBearElem>],
        poly_mix: BabyBearExtElem,
        po2: usize,
        steps: usize,
    ) {
        let ctrl = groups[REGISTER_GROUP_CTRL];
        let data = groups[REGISTER_GROUP_DATA];
        let accum = groups[REGISTER_GROUP_ACCUM];
        let mix = globals[GLOBAL_MIX];
        let out = globals[GLOBAL_OUT];
        tracing::debug!(
            "check: {}, code: {}, data: {}, accum: {}, mix: {} out: {}",
            check.size(),
            ctrl.size(),
            data.size(),
            accum.size(),
            mix.size(),
            out.size()
        );
        tracing::debug!(
            "total: {}",
            (check.size() + ctrl.size() + data.size() + accum.size() + mix.size() + out.size()) * 4
        );

        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let rou = BabyBearElem::ROU_FWD[po2 + EXP_PO2];
        let poly_mix_pows = map_pow(poly_mix, crate::info::POLY_MIX_POWERS);

        let check = check.as_device_ptr();
        let ctrl = ctrl.as_device_ptr();
        let data = data.as_device_ptr();
        let accum = accum.as_device_ptr();
        let mix = mix.as_device_ptr();
        let out = out.as_device_ptr();

        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cuda_eval_check(
                check.as_ptr() as *const BabyBearElem,
                ctrl.as_ptr() as *const BabyBearElem,
                data.as_ptr() as *const BabyBearElem,
                accum.as_ptr() as *const BabyBearElem,
                mix.as_ptr() as *const BabyBearElem,
                out.as_ptr() as *const BabyBearElem,
                &rou as *const BabyBearElem,
                po2 as u32,
                domain as u32,
                poly_mix_pows.as_ptr(),
            )
        })
        .unwrap();
    }

    #[allow(unused)]
    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        ctrl: &HipBuffer<BabyBearElem>,
        io: &HipBuffer<BabyBearElem>,
        data: &HipBuffer<BabyBearElem>,
        mix: &HipBuffer<BabyBearElem>,
        accum: &HipBuffer<BabyBearElem>,
        steps: usize,
    ) {
        unimplemented!()
    }
}

pub(crate) fn recursion_prover(hashfn: &str) -> Result<Box<dyn RecursionProver>> {
    match hashfn {
        "poseidon2" => {
            let hal = Arc::new(HipHalPoseidon2::new());
            let circuit_hal = Arc::new(HipCircuitHalPoseidon2::new(hal.clone()));
            Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
        }
        "poseidon_254" => {
            let hal = Arc::new(HipHalPoseidon254::new());
            let circuit_hal = Arc::new(HipCircuitHalPoseidon254::new(hal.clone()));
            Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
        }
        "sha-256" => {
            let hal = Arc::new(HipHalSha256::new());
            let circuit_hal = Arc::new(HipCircuitHalSha256::new(hal.clone()));
            Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
        }
        _ => bail!("Unsupported hashfn: {hashfn}"),
    }
}

#[cfg(all(test, feature = "test"))]
mod tests {
    use std::sync::Arc;

    use risc0_core::field::baby_bear::BabyBear;
    use risc0_zkp::{
        core::hash::sha::Sha256HashSuite,
        hal::{cpu::CpuHal, hip::HipHalSha256},
    };
    use test_log::test;

    use super::HipCircuitHal;
    use crate::prove::hal::cpu::CpuCircuitHal;

    #[test]
    fn eval_check() {
        const PO2: usize = 4;
        let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
        let gpu_hal = Arc::new(HipHalSha256::new());
        let gpu_eval = HipCircuitHal::new(gpu_hal.clone());
        crate::testutil::eval_check(&cpu_hal, CpuCircuitHal, gpu_hal.as_ref(), gpu_eval, PO2);
    }
}
