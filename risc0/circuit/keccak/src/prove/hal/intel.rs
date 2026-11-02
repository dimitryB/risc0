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

//! The keccak circuit on Intel GPUs, on the zkp crate's Intel HAL: eval_check in SYCL kernels
//! built from the CUDA sources (see keccak-sys/build.rs), the witness generated on the CPU.
//!
//! The witness generation is the CPU's because the GPU compiler cannot build the generated step
//! code as a kernel: given an hour, it ran out of memory (over 40 GB). On the CPU it takes about
//! 1.3 s at po2 17, against 14 s for the whole proof there.

use std::{cell::RefCell, rc::Rc};

use anyhow::Result;
use rayon::prelude::*;
use risc0_circuit_keccak_sys::{risc0_circuit_keccak_intel_eval_check, ScatterInfo};
use risc0_core::{
    field::{
        baby_bear::{BabyBearElem, BabyBearExtElem},
        map_pow, Elem, RootsOfUnity,
    },
    scope,
};
use risc0_sys::{ffi_wrap, intel::get_queue};
use risc0_zkp::{
    core::log2_ceil,
    field::ExtElem as _,
    hal::{
        intel::{BufferImpl as IntelBuffer, IntelHal, IntelHalPoseidon2, IntelHash},
        AccumPreflight, Buffer as _, CircuitHal,
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

pub struct IntelCircuitHal<IH: IntelHash> {
    _hal: Rc<IntelHal<IH>>, // retain a reference to ensure the context remains valid
    // The host copy of a data buffer `scatter_preflight` has filled, and that buffer's address:
    // `generate_witness` continues from it rather than downloading the buffer.
    scattered: RefCell<Option<(usize, Vec<Val>)>>,
}

impl<IH: IntelHash> IntelCircuitHal<IH> {
    pub fn new(hal: Rc<IntelHal<IH>>) -> Self {
        Self {
            _hal: hal,
            scattered: RefCell::new(None),
        }
    }
}

fn device_ptr<T>(buf: &IntelBuffer<T>) -> *mut u32 {
    buf.as_device_ptr().0 as *mut u32
}

impl<IH: IntelHash> CircuitWitnessGenerator<IntelHal<IH>> for IntelCircuitHal<IH> {
    type PreferredPreflightOrder = ForwardPreflightOrder;
    const HOST_WITNESS: bool = true;

    fn scatter_preflight(
        &self,
        into: &MetaBuffer<IntelHal<IH>>,
        infos: &[ScatterInfo],
        from: &[u32],
    ) -> Result<()> {
        scope!("scatter");
        // `into` was just allocated all INVALID (see `MetaBuffer::new`): make that on the host, in
        // parallel, rather than download it (0.13 s for 574 MB at po2 17).
        let mut host: Vec<Val> = (0..into.buf.size())
            .into_par_iter()
            .map(|_| Val::INVALID)
            .collect();
        scatter_preflight_host(&mut host, into.rows, infos, from);
        *self.scattered.borrow_mut() = Some((device_ptr(&into.buf) as usize, host));
        Ok(())
    }

    fn generate_witness<O: PreflightCycleOrder>(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace<O>,
        global: &MetaBuffer<IntelHal<IH>>,
        data: &MetaBuffer<IntelHal<IH>>,
    ) -> Result<()> {
        scope!("witgen");
        let mut host_data = match self.scattered.take() {
            Some((ptr, host)) if ptr == device_ptr(&data.buf) as usize => host,
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

impl<IH: IntelHash> CircuitHal<IntelHal<IH>> for IntelCircuitHal<IH> {
    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &IntelBuffer<BabyBearElem>,
        _io: &IntelBuffer<BabyBearElem>,
        _data: &IntelBuffer<BabyBearElem>,
        _mix: &IntelBuffer<BabyBearElem>,
        _accum: &IntelBuffer<BabyBearElem>,
        _steps: usize,
    ) {
    }

    fn eval_check(
        &self,
        check: &IntelBuffer<BabyBearElem>,
        groups: &[&IntelBuffer<BabyBearElem>],
        globals: &[&IntelBuffer<BabyBearElem>],
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

        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let rou = BabyBearElem::ROU_FWD[po2 + EXP_PO2];
        let rou: u32 = BabyBearElem::as_u32_slice(&[rou])[0];

        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);
        let poly_mix_pows: &[u32; BabyBearExtElem::EXT_SIZE * NUM_POLY_MIX_POWERS] =
            BabyBearExtElem::as_u32_slice(poly_mix_pows.as_slice())
                .try_into()
                .unwrap();

        ffi_wrap(|| unsafe {
            risc0_circuit_keccak_intel_eval_check(
                get_queue(),
                device_ptr(check),
                device_ptr(ctrl),
                device_ptr(data),
                device_ptr(accum),
                device_ptr(mix),
                device_ptr(out),
                rou,
                po2 as u32,
                domain as u32,
                poly_mix_pows.as_ptr(),
            )
        })
        .unwrap();
    }
}

pub type IntelCircuitHalPoseidon2 = IntelCircuitHal<risc0_zkp::hal::intel::IntelHashPoseidon2>;

pub fn keccak_prover() -> Result<Box<dyn KeccakProver>> {
    let hal = Rc::new(IntelHalPoseidon2::new());
    let circuit_hal = Rc::new(IntelCircuitHalPoseidon2::new(hal.clone()));
    Ok(Box::new(KeccakProverImpl { hal, circuit_hal }))
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use risc0_core::field::baby_bear::BabyBear;
    use risc0_zkp::{
        core::hash::sha::Sha256HashSuite,
        hal::{cpu::CpuHal, intel::IntelHalSha256},
    };
    use test_log::test;

    use crate::prove::hal::cpu::CpuCircuitHal;

    #[test]
    fn eval_check() {
        const PO2: usize = 4;
        let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
        let cpu_eval = CpuCircuitHal;
        let gpu_hal = Rc::new(IntelHalSha256::new());
        let gpu_eval = super::IntelCircuitHal::new(gpu_hal.clone());
        crate::prove::testutil::eval_check(&cpu_hal, cpu_eval, gpu_hal.as_ref(), gpu_eval, PO2);
    }
}
