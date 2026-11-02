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

use std::{cell::RefCell, rc::Rc};

use anyhow::Result;
use rayon::prelude::*;
use risc0_circuit_keccak_sys::{
    risc0_circuit_keccak_cuda_eval_check, risc0_circuit_keccak_cuda_reset, ScatterInfo,
};
use risc0_core::{
    field::{
        baby_bear::{BabyBearElem, BabyBearExtElem},
        map_pow, Elem, RootsOfUnity,
    },
    scope,
};
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    core::log2_ceil,
    field::ExtElem as _,
    hal::{
        hip::{BufferImpl as HipBuffer, HipHal, HipHalPoseidon2, HipHash, HipHashPoseidon2},
        AccumPreflight, Buffer, CircuitHal,
    },
    INV_RATE,
};

use crate::{
    prove::{KeccakProver, KeccakProverImpl, GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE, REGISTER_GROUP_DATA},
        info::{NUM_POLY_MIX_POWERS, POLY_MIX_POWERS},
    },
};

use super::{
    cpu::{generate_witness_host, scatter_preflight_host, HostMetaBuffer},
    CircuitWitnessGenerator, ForwardPreflightOrder, MetaBuffer, PreflightCycleOrder,
    PreflightTrace, StepMode,
};

pub struct HipCircuitHal<CH: HipHash> {
    _hal: Rc<HipHal<CH>>, // retain a reference to ensure the context remains valid
    // The host copy of a data buffer `scatter_preflight` has filled, and that buffer's address:
    // `generate_witness` continues from it rather than downloading the buffer.
    scattered: RefCell<Option<(usize, Vec<Val>)>>,
}

impl<CH: HipHash> HipCircuitHal<CH> {
    pub fn new(hal: Rc<HipHal<CH>>) -> Self {
        Self {
            _hal: hal,
            scattered: RefCell::new(None),
        }
    }
}

fn device_addr<T>(buf: &HipBuffer<T>) -> usize {
    buf.as_device_ptr().as_ptr() as usize
}

impl<CH: HipHash> Drop for HipCircuitHal<CH> {
    fn drop(&mut self) {
        hip_reset();
    }
}

// The witness is made on the host, as on Intel GPUs: under HIP the GPU witness generation
// (par_stepExec) fails a device assertion from po2 16 (an RX 9070 XT, HIP SDK 7.2 on Windows),
// which there takes the device down with it.
impl<CH: HipHash> CircuitWitnessGenerator<HipHal<CH>> for HipCircuitHal<CH> {
    type PreferredPreflightOrder = ForwardPreflightOrder;
    const HOST_WITNESS: bool = true;

    fn scatter_preflight(
        &self,
        into: &MetaBuffer<HipHal<CH>>,
        infos: &[ScatterInfo],
        from: &[u32],
    ) -> Result<()> {
        scope!("scatter");
        // `into` was just allocated all INVALID (see `MetaBuffer::new`): make that on the host
        // rather than download it.
        let mut host: Vec<Val> = (0..into.buf.size())
            .into_par_iter()
            .map(|_| Val::INVALID)
            .collect();
        scatter_preflight_host(&mut host, into.rows, infos, from);
        *self.scattered.borrow_mut() = Some((device_addr(&into.buf), host));
        Ok(())
    }

    fn generate_witness<O: PreflightCycleOrder>(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace<O>,
        global: &MetaBuffer<HipHal<CH>>,
        data: &MetaBuffer<HipHal<CH>>,
    ) -> Result<()> {
        scope!("witgen");
        let mut host_data = match self.scattered.take() {
            Some((addr, host)) if addr == device_addr(&data.buf) => host,
            _ => data.buf.to_vec(),
        };
        let mut host_global = global.buf.to_vec();
        generate_witness_host(
            mode,
            preflight,
            HostMetaBuffer::of(global, &mut host_global),
            HostMetaBuffer::of(data, &mut host_data),
        )?;
        global.buf.copy_from_host(&host_global);
        data.buf.copy_from_host(&host_data);
        Ok(())
    }
}

impl<CH: HipHash> CircuitHal<HipHal<CH>> for HipCircuitHal<CH> {
    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &HipBuffer<BabyBearElem>,
        _io: &HipBuffer<BabyBearElem>,
        _data: &HipBuffer<BabyBearElem>,
        _mix: &HipBuffer<BabyBearElem>,
        _accum: &HipBuffer<BabyBearElem>,
        _steps: usize,
    ) {
    }

    fn eval_check(
        &self,
        check: &HipBuffer<BabyBearElem>,
        groups: &[&HipBuffer<BabyBearElem>],
        globals: &[&HipBuffer<BabyBearElem>],
        poly_mix: BabyBearExtElem,
        po2: usize,
        steps: usize,
    ) {
        scope!("eval_check");

        let accum = groups[REGISTER_GROUP_ACCUM];
        let ctrl = groups[REGISTER_GROUP_CODE];
        let data = groups[REGISTER_GROUP_DATA];
        let mix = globals[GLOBAL_MIX];
        let out = globals[GLOBAL_OUT];
        tracing::debug!(
            "check: {}, ctrl: {}, data: {}, accum: {}, mix: {} out: {}",
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

        tracing::debug!("steps: {steps}, domain: {domain}, po2: {po2}, rou: {rou:?}");
        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);
        let poly_mix_pows: &[u32; BabyBearExtElem::EXT_SIZE * NUM_POLY_MIX_POWERS] =
            BabyBearExtElem::as_u32_slice(poly_mix_pows.as_slice())
                .try_into()
                .unwrap();

        ffi_wrap(|| unsafe {
            risc0_circuit_keccak_cuda_eval_check(
                check.as_device_ptr(),
                ctrl.as_device_ptr(),
                data.as_device_ptr(),
                accum.as_device_ptr(),
                mix.as_device_ptr(),
                out.as_device_ptr(),
                &rou as *const BabyBearElem,
                po2 as u32,
                domain as u32,
                poly_mix_pows.as_ptr(),
            )
        })
        .unwrap();
    }
}

pub type HipCircuitHalPoseidon2 = HipCircuitHal<HipHashPoseidon2>;

pub fn keccak_prover() -> Result<Box<dyn KeccakProver>> {
    let hal = Rc::new(HipHalPoseidon2::new());
    let circuit_hal = Rc::new(HipCircuitHalPoseidon2::new(hal.clone()));
    Ok(Box::new(KeccakProverImpl { hal, circuit_hal }))
}

fn hip_reset() {
    ffi_wrap(|| unsafe { risc0_circuit_keccak_cuda_reset() }).unwrap();
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use risc0_core::field::baby_bear::BabyBear;
    use risc0_zkp::{
        core::hash::poseidon2::Poseidon2HashSuite,
        hal::{cpu::CpuHal, hip::HipHalPoseidon2},
    };
    use test_log::test;

    use super::HipCircuitHalPoseidon2;
    use crate::prove::hal::cpu::CpuCircuitHal;

    #[test]
    fn eval_check() {
        const PO2: usize = 4;
        let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Poseidon2HashSuite::new_suite());
        let gpu_hal = Rc::new(HipHalPoseidon2::new());
        let gpu_eval = HipCircuitHalPoseidon2::new(gpu_hal.clone());
        crate::prove::testutil::eval_check(
            &cpu_hal,
            CpuCircuitHal,
            gpu_hal.as_ref(),
            gpu_eval,
            PO2,
        );
    }
}
