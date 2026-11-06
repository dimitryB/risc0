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

//! Poseidon2 recursion over shared Metal buffers. Witness generation and
//! accumulation use the pinned CPU routines directly on those shared buffers.
//! Each GPU-only NTT or Merkle stage completes at its batch boundary. CPU
//! witness and accumulation access requires that no batch is pending.
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use super::{CircuitAccumulator, CircuitWitnessGenerator};
use crate::{
    prove::{RecursionProver, RecursionProverImpl},
    taps::TAPSET,
    CircuitImpl, GLOBAL_MIX, GLOBAL_OUT, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CTRL,
    REGISTER_GROUP_DATA,
};
use anyhow::{bail, ensure, Result};
use metal::{CompileOptions, ComputePipelineDescriptor};
use risc0_circuit_recursion_sys::{
    risc0_circuit_recursion_cpu_accum, risc0_circuit_recursion_cpu_eval_check,
    risc0_circuit_recursion_cpu_witgen, RawAccumBuffers, RawExecBuffers, RawPreflightTrace,
    StepMode,
};
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    adapter::CircuitInfo,
    core::log2_ceil,
    field::{
        baby_bear::{BabyBearElem, BabyBearExtElem},
        map_pow, RootsOfUnity,
    },
    hal::{
        metal::{BufferImpl, KernelArg, MetalHalPoseidon2},
        AccumPreflight, Buffer, CircuitHal, Hal,
    },
    INV_RATE,
};

const SOURCE: &str = include_str!(env!("RECURSION_METAL_SOURCE"));
const SPLIT_SOURCE: &str = include_str!(env!("RECURSION_METAL_SPLIT_SOURCE"));
static COMPLETED_CHECKS: AtomicU64 = AtomicU64::new(0);
static EVALUATION_POINTS: AtomicU64 = AtomicU64::new(0);
static CPU_DIFFERENTIAL_CHECKS: AtomicU64 = AtomicU64::new(0);

fn split_enabled() -> bool {
    std::env::var("RISC0_RECURSION_METAL_SPLIT").as_deref() != Ok("0")
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct MetalRecursionStats {
    pub completed_constraint_checks: u64,
    pub evaluation_points: u64,
    pub cpu_differential_checks: u64,
    pub eval_check_split: bool,
}

pub fn metal_recursion_stats() -> MetalRecursionStats {
    MetalRecursionStats {
        completed_constraint_checks: COMPLETED_CHECKS.load(Ordering::Relaxed),
        evaluation_points: EVALUATION_POINTS.load(Ordering::Relaxed),
        cpu_differential_checks: CPU_DIFFERENTIAL_CHECKS.load(Ordering::Relaxed),
        eval_check_split: split_enabled(),
    }
}

pub fn metal_recursion_kernel_source() -> &'static str {
    if split_enabled() {
        SPLIT_SOURCE
    } else {
        SOURCE
    }
}

pub(crate) struct MetalCircuitHal {
    hal: Arc<MetalHalPoseidon2>,
    kernels: Vec<ComputePipelineDescriptor>,
    split: bool,
}

impl MetalCircuitHal {
    fn new(hal: Arc<MetalHalPoseidon2>) -> Result<Self> {
        let split = split_enabled();
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = hal
            .device
            .new_library_with_source(if split { SPLIT_SOURCE } else { SOURCE }, &options)
            .map_err(|error| anyhow::anyhow!("recursion Metal compilation failed: {error}"))?;
        let names: Vec<String> = if split {
            (0..6)
                .map(|part| format!("eval_check_part{part}"))
                .collect()
        } else {
            vec!["eval_check".to_owned()]
        };
        let kernels = names
            .iter()
            .map(|name| {
                let function = library.get_function(name, None).map_err(|error| {
                    anyhow::anyhow!("missing recursion Metal evaluator {name}: {error}")
                })?;
                let kernel = ComputePipelineDescriptor::new();
                kernel.set_compute_function(Some(&function));
                Ok(kernel)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            hal,
            kernels,
            split,
        })
    }

    fn validate_witness(
        total: u32,
        ctrl: &BufferImpl<BabyBearElem>,
        data: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
    ) -> Result<()> {
        ensure!(total.is_power_of_two(), "invalid recursion cycle count");
        ensure!(
            ctrl.size() == total as usize * TAPSET.group_size(REGISTER_GROUP_CTRL),
            "invalid ctrl buffer"
        );
        ensure!(
            data.size() == total as usize * TAPSET.group_size(REGISTER_GROUP_DATA),
            "invalid data buffer"
        );
        ensure!(
            global.size() == CircuitImpl::OUTPUT_SIZE,
            "invalid global buffer"
        );
        Ok(())
    }
}

impl CircuitWitnessGenerator<MetalHalPoseidon2> for MetalCircuitHal {
    fn generate_witness(
        &self,
        mode: StepMode,
        total: u32,
        preflight: &RawPreflightTrace,
        ctrl: &BufferImpl<BabyBearElem>,
        data: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
    ) -> Result<()> {
        self.hal.assert_no_pending_batch();
        Self::validate_witness(total, ctrl, data, global)?;
        let buffers = RawExecBuffers {
            ctrl: ctrl.as_ptr().cast(),
            data: data.as_ptr().cast(),
            global: global.as_ptr().cast(),
        };
        // StorageModeShared + synchronous command completion, with allocations
        // retained by these buffer references throughout the FFI call.
        ffi_wrap(|| unsafe { risc0_circuit_recursion_cpu_witgen(mode, &buffers, preflight, total) })
    }

    fn upload_host_witness(
        &self,
        data: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
        host_data: Vec<BabyBearElem>,
        host_global: Vec<BabyBearElem>,
    ) -> Result<()> {
        self.hal.assert_no_pending_batch();
        ensure!(
            data.size() == host_data.len() && global.size() == host_global.len(),
            "invalid host witness lengths"
        );
        data.view_mut(|view| view.copy_from_slice(&host_data));
        global.view_mut(|view| view.copy_from_slice(&host_global));
        Ok(())
    }
}

impl CircuitAccumulator<MetalHalPoseidon2> for MetalCircuitHal {
    fn accumulate(
        &self,
        work: u32,
        total: u32,
        ctrl: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
        data: &BufferImpl<BabyBearElem>,
        mix: &BufferImpl<BabyBearElem>,
        accum: &BufferImpl<BabyBearElem>,
    ) -> Result<()> {
        self.hal.assert_no_pending_batch();
        Self::validate_witness(total, ctrl, data, global)?;
        ensure!(
            work <= total && mix.size() == CircuitImpl::MIX_SIZE,
            "invalid accumulation inputs"
        );
        ensure!(
            accum.size() == total as usize * TAPSET.group_size(REGISTER_GROUP_ACCUM),
            "invalid accum buffer"
        );
        let buffers = RawAccumBuffers {
            ctrl: ctrl.as_ptr().cast(),
            global: global.as_ptr().cast(),
            data: data.as_ptr().cast(),
            mix: mix.as_ptr().cast(),
            accum: accum.as_ptr().cast(),
        };
        ffi_wrap(|| unsafe { risc0_circuit_recursion_cpu_accum(&buffers, work, total) })
    }
}

impl CircuitHal<MetalHalPoseidon2> for MetalCircuitHal {
    fn eval_check(
        &self,
        check: &BufferImpl<BabyBearElem>,
        groups: &[&BufferImpl<BabyBearElem>],
        globals: &[&BufferImpl<BabyBearElem>],
        poly_mix: BabyBearExtElem,
        po2: usize,
        steps: usize,
    ) {
        let domain = steps
            .checked_mul(INV_RATE)
            .expect("recursion domain overflow");
        assert_eq!(steps, 1usize.checked_shl(po2 as u32).unwrap());
        assert!(domain <= u32::MAX as usize);
        assert_eq!(check.size(), domain * 4);
        for group in [
            REGISTER_GROUP_CTRL,
            REGISTER_GROUP_DATA,
            REGISTER_GROUP_ACCUM,
        ] {
            assert_eq!(groups[group].size(), domain * TAPSET.group_size(group));
        }
        assert_eq!(globals[GLOBAL_MIX].size(), CircuitImpl::MIX_SIZE);
        assert_eq!(globals[GLOBAL_OUT].size(), CircuitImpl::OUTPUT_SIZE);
        let powers_host = map_pow(poly_mix, crate::info::POLY_MIX_POWERS);
        let powers = self
            .hal
            .copy_from_extelem("recursion_poly_mix", &powers_host);
        let rou = BabyBearElem::ROU_FWD[po2 + log2_ceil(INV_RATE)];
        let args = [
            check.as_arg(),
            groups[REGISTER_GROUP_CTRL].as_arg(),
            groups[REGISTER_GROUP_DATA].as_arg(),
            groups[REGISTER_GROUP_ACCUM].as_arg(),
            globals[GLOBAL_MIX].as_arg(),
            globals[GLOBAL_OUT].as_arg(),
            powers.as_arg(),
            KernelArg::Integer(rou.as_u32_montgomery()),
            KernelArg::Integer(po2 as u32),
            KernelArg::Integer(domain as u32),
        ];
        if self.split {
            self.hal.with_batch("recursion_eval_check_split", || {
                for kernel in &self.kernels {
                    self.hal.dispatch(kernel, &args, domain as u64, None);
                }
            });
        } else {
            self.hal
                .dispatch(&self.kernels[0], &args, domain as u64, None);
        }
        COMPLETED_CHECKS.fetch_add(1, Ordering::Relaxed);
        EVALUATION_POINTS.fetch_add(domain as u64, Ordering::Relaxed);
        if std::env::var("RISC0_RECURSION_METAL_CHECK_CPU").as_deref() == Ok("1") {
            self.hal.assert_no_pending_batch();
            let mut cpu = vec![BabyBearElem::new(0); check.size()];
            let buffers = RawAccumBuffers {
                ctrl: groups[REGISTER_GROUP_CTRL].as_ptr().cast(),
                global: globals[GLOBAL_OUT].as_ptr().cast(),
                data: groups[REGISTER_GROUP_DATA].as_ptr().cast(),
                mix: globals[GLOBAL_MIX].as_ptr().cast(),
                accum: groups[REGISTER_GROUP_ACCUM].as_ptr().cast(),
            };
            ffi_wrap(|| unsafe {
                risc0_circuit_recursion_cpu_eval_check(
                    &buffers,
                    powers_host.as_ptr(),
                    cpu.as_mut_ptr(),
                    rou,
                    po2 as u32,
                    domain as u32,
                )
            })
            .expect("recursion CPU constraint oracle failed");
            let gpu = unsafe {
                std::slice::from_raw_parts(check.as_ptr().cast::<BabyBearElem>(), check.size())
            };
            for (index, (actual, expected)) in gpu.iter().zip(cpu.iter()).enumerate() {
                assert_eq!(
                    actual.as_u32_montgomery(),
                    expected.as_u32_montgomery(),
                    "recursion Metal differs from CPU: po2={po2}, index={index}"
                );
            }
            CPU_DIFFERENTIAL_CHECKS.fetch_add(1, Ordering::Relaxed);
            tracing::info!(
                po2,
                domain,
                "recursion Metal valid-witness CPU differential passed"
            );
        }
        tracing::info!(
            po2,
            domain,
            split = self.split,
            completed_dispatches = self.hal.completed_dispatches(),
            completed_command_buffers = self.hal.completed_command_buffers(),
            "Poseidon2 recursion Metal constraint check completed"
        );
    }

    fn accumulate(
        &self,
        _: &AccumPreflight,
        _: &BufferImpl<BabyBearElem>,
        _: &BufferImpl<BabyBearElem>,
        _: &BufferImpl<BabyBearElem>,
        _: &BufferImpl<BabyBearElem>,
        _: &BufferImpl<BabyBearElem>,
        _: usize,
    ) {
        unreachable!("recursion uses CircuitAccumulator");
    }
}

pub(crate) fn recursion_prover(hashfn: &str) -> Result<Box<dyn RecursionProver>> {
    match std::env::var("RISC0_RECURSION_HAL").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("cpu") => super::cpu::recursion_prover(hashfn),
        Ok("metal") if hashfn == "poseidon2" => metal::objc::rc::autoreleasepool(|| {
            let hal = Arc::new(MetalHalPoseidon2::new());
            let circuit = Arc::new(MetalCircuitHal::new(hal.clone())?);
            tracing::info!(
                hashfn,
                witness = "cpu-shared",
                accumulation = "cpu-shared",
                "recursion Metal enabled"
            );
            Ok(Box::new(RecursionProverImpl::new(hal, circuit)) as Box<dyn RecursionProver>)
        }),
        Ok("metal") => {
            tracing::info!(
                hashfn,
                "recursion CPU selected; Metal supports Poseidon2 only"
            );
            super::cpu::recursion_prover(hashfn)
        }
        _ => bail!("RISC0_RECURSION_HAL must be cpu or metal"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use risc0_zkp::{
        core::hash::poseidon2::Poseidon2HashSuite, field::baby_bear::BabyBear, hal::cpu::CpuHal,
    };

    #[test]
    fn constraint_evaluator_matches_cpu_with_offsets_and_reuse() {
        metal::objc::rc::autoreleasepool(|| {
            let cpu = CpuHal::<BabyBear>::new(Poseidon2HashSuite::new_suite());
            let gpu = Arc::new(MetalHalPoseidon2::new());
            let circuit = MetalCircuitHal::new(gpu.clone()).unwrap();
            for po2 in [4, 7, 10, 4] {
                let p = crate::testutil::EvalCheckParams::new(po2);
                let expected =
                    crate::testutil::eval_check_impl(&p, &cpu, &super::super::cpu::CpuCircuitHal);
                // Nonzero offsets on all inputs and output, with sentinels
                // around the output to detect an out-of-range GPU write.
                let input = |name, values: &[BabyBearElem]| {
                    let mut padded = vec![BabyBearElem::new(17); 3];
                    padded.extend_from_slice(values);
                    gpu.copy_from_elem(name, &padded).slice(3, values.len())
                };
                let ctrl = input("ctrl", &p.code);
                let data = input("data", &p.data);
                let accum = input("accum", &p.accum);
                let mix = input("mix", &p.mix);
                let out = input("out", &p.out);
                let sentinel = BabyBearElem::new(42);
                let storage = gpu.copy_from_elem("check", &vec![sentinel; p.domain * 4 + 6]);
                let check = storage.slice(3, p.domain * 4);
                circuit.eval_check(
                    &check,
                    &[&accum, &ctrl, &data],
                    &[&mix, &out],
                    p.poly_mix,
                    p.po2,
                    p.steps,
                );
                assert_eq!(check.to_vec(), expected);
                let full = storage.to_vec();
                assert!(full[..3]
                    .iter()
                    .chain(&full[full.len() - 3..])
                    .all(|x| *x == sentinel));
            }
            assert!(gpu.completed_dispatches() >= 4);
        });
    }
}
