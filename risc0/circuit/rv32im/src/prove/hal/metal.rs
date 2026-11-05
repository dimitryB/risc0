//! Opt-in RV32IM prover using synchronous shared Metal buffers. CPU witness and
//! accumulation retain the pinned routines and pre_data aliasing. A single
//! proving thread owns the HAL for its lifetime, also sequencing recursion.
use super::{
    CircuitAccumulator, CircuitWitnessGenerator, MetaBuffer, SegmentProver, SegmentProverImpl,
    StepMode,
};
use crate::{
    prove::{witgen::preflight::PreflightTrace, GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{ExtVal, Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE, REGISTER_GROUP_DATA},
        info::POLY_MIX_POWERS,
        taps::TAPSET,
        CircuitImpl,
    },
};
use anyhow::{ensure, Result};
use metal::{CompileOptions, ComputePipelineDescriptor, MTLSize};
use rayon::prelude::*;
use risc0_circuit_rv32im_sys::{
    risc0_circuit_rv32im_cpu_accum, risc0_circuit_rv32im_cpu_poly_fp,
    risc0_circuit_rv32im_cpu_witgen, RawAccumBuffers, RawBuffer, RawExecBuffers, RawPreflightTrace,
};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    adapter::CircuitInfo,
    field::{map_pow, Elem, ExtElem as _, RootsOfUnity},
    hal::{
        metal::{BufferImpl, KernelArg, MetalHalPoseidon2},
        AccumPreflight, Buffer, CircuitHal, Hal,
    },
    INV_RATE,
};
use std::{
    cell::RefCell,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Weak,
    },
};

const SOURCE: &str = include_str!(env!("RV32IM_METAL_SOURCE"));
const EVALFAST_SOURCE: &str = include_str!(env!("RV32IM_EVALFAST_METAL_SOURCE"));
fn evalfast_enabled() -> bool {
    std::env::var("RISC0_RV32IM_TAIL_SPLIT").as_deref() != Ok("1")
}
static CPU_CHECKS: AtomicU64 = AtomicU64::new(0);
static CHECKS: AtomicU64 = AtomicU64::new(0);
static POINTS: AtomicU64 = AtomicU64::new(0);
static WITNESS_CALLS: AtomicU64 = AtomicU64::new(0);
static ACCUM_CALLS: AtomicU64 = AtomicU64::new(0);
thread_local! { static HALS: RefCell<Vec<Weak<MetalHalPoseidon2>>> = const { RefCell::new(Vec::new()) }; }

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct MetalRv32imStats {
    pub completed_constraint_checks: u64,
    pub evaluation_points: u64,
    pub completed_dispatches: u64,
    pub cpu_witness_calls: u64,
    pub cpu_differential_checks: u64,
    pub cpu_accumulation_calls: u64,
}
pub fn metal_rv32im_stats() -> MetalRv32imStats {
    MetalRv32imStats {
        completed_constraint_checks: CHECKS.load(Ordering::Relaxed),
        cpu_differential_checks: CPU_CHECKS.load(Ordering::Relaxed),
        evaluation_points: POINTS.load(Ordering::Relaxed),
        cpu_witness_calls: WITNESS_CALLS.load(Ordering::Relaxed),
        cpu_accumulation_calls: ACCUM_CALLS.load(Ordering::Relaxed),
        completed_dispatches: HALS.with(|hals| {
            hals.borrow()
                .iter()
                .filter_map(Weak::upgrade)
                .map(|h| h.completed_dispatches())
                .sum()
        }),
    }
}
pub fn metal_rv32im_kernel_source() -> &'static str {
    if evalfast_enabled() {
        EVALFAST_SOURCE
    } else {
        SOURCE
    }
}

struct MetalCircuitHal {
    hal: Arc<MetalHalPoseidon2>,
    kernels: Vec<ComputePipelineDescriptor>,
    evalfast: bool,
}
impl MetalCircuitHal {
    fn new(hal: Arc<MetalHalPoseidon2>) -> Result<Self> {
        let started = std::time::Instant::now();
        let evalfast = evalfast_enabled();
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = hal
            .device
            .new_library_with_source(if evalfast { EVALFAST_SOURCE } else { SOURCE }, &options)
            .map_err(|e| anyhow::anyhow!("RV32IM Metal compilation failed: {e}"))?;
        let names: &[&str] = if evalfast {
            &["ef_prep", "ef_eval_split", "ef_combine"]
        } else {
            &["eval_check"]
        };
        let mut kernels = Vec::with_capacity(names.len());
        for name in names {
            let function = library
                .get_function(name, None)
                .map_err(|e| anyhow::anyhow!(e))?;
            let kernel = ComputePipelineDescriptor::new();
            kernel.set_compute_function(Some(&function));
            hal.device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| anyhow::anyhow!(e))?;
            kernels.push(kernel);
        }
        HALS.with(|hals| hals.borrow_mut().push(Arc::downgrade(&hal)));
        tracing::info!(
            seconds = started.elapsed().as_secs_f64(),
            evalfast,
            "RV32IM Metal evaluator compiled"
        );
        Ok(Self {
            hal,
            kernels,
            evalfast,
        })
    }
}

// StorageModeShared and synchronous command completion precede CPU writes.
// Pass mapped raw pointers directly: do not create shared Rust slices around
// memory the C++ witness/accumulation routines are allowed to mutate.
fn raw(buffer: &MetaBuffer<MetalHalPoseidon2>) -> RawBuffer {
    assert_eq!(
        buffer.buf.size(),
        buffer.rows.checked_mul(buffer.cols).unwrap()
    );
    RawBuffer {
        buf: buffer.buf.as_ptr().cast::<Val>(),
        rows: buffer.rows,
        cols: buffer.cols,
        checked: buffer.checked,
    }
}
fn raw_trace(trace: &PreflightTrace) -> RawPreflightTrace {
    RawPreflightTrace {
        cycles: trace.cycles.as_ptr(),
        txns: trace.txns.as_ptr(),
        bigint_bytes: trace.bigint_bytes.as_ptr(),
        txns_len: trace.txns.len().try_into().unwrap(),
        bigint_bytes_len: trace.bigint_bytes.len().try_into().unwrap(),
        table_split_cycle: trace.table_split_cycle,
    }
}
impl CircuitWitnessGenerator<MetalHalPoseidon2> for MetalCircuitHal {
    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<MetalHalPoseidon2>,
        data: &MetaBuffer<MetalHalPoseidon2>,
        _pre_data: &MetaBuffer<MetalHalPoseidon2>,
    ) -> Result<()> {
        scope!("cpu_shared_witgen");
        let mut pre_data = raw(data);
        pre_data.checked = false;
        let buffers = RawExecBuffers {
            global: raw(global),
            data: raw(data),
            pre_data,
        };
        let trace = raw_trace(preflight);
        ffi_wrap(|| unsafe {
            risc0_circuit_rv32im_cpu_witgen(
                mode as u32,
                &buffers,
                &trace,
                preflight.cycles.len().try_into().unwrap(),
            )
        })?;
        WITNESS_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}
impl CircuitAccumulator<MetalHalPoseidon2> for MetalCircuitHal {
    fn step_accum(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<MetalHalPoseidon2>,
        accum: &MetaBuffer<MetalHalPoseidon2>,
        global: &MetaBuffer<MetalHalPoseidon2>,
        mix: &MetaBuffer<MetalHalPoseidon2>,
    ) -> Result<()> {
        scope!("cpu_shared_accumulate");
        let buffers = RawAccumBuffers {
            data: raw(data),
            accum: raw(accum),
            global: raw(global),
            mix: raw(mix),
        };
        let trace = raw_trace(preflight);
        ffi_wrap(|| unsafe {
            risc0_circuit_rv32im_cpu_accum(
                &buffers,
                &trace,
                preflight.cycles.len().try_into().unwrap(),
            )
        })?;
        ACCUM_CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl CircuitHal<MetalHalPoseidon2> for MetalCircuitHal {
    fn eval_check(
        &self,
        check: &BufferImpl<Val>,
        groups: &[&BufferImpl<Val>],
        globals: &[&BufferImpl<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) {
        assert!((14..=20).contains(&po2), "RV32IM Metal supports po2 14..20");
        assert_eq!(steps, 1usize << po2);
        assert_eq!(INV_RATE, 4);
        let domain = steps.checked_mul(INV_RATE).unwrap();
        assert_eq!(check.size(), domain * 4);
        for group in [
            REGISTER_GROUP_CODE,
            REGISTER_GROUP_DATA,
            REGISTER_GROUP_ACCUM,
        ] {
            assert_eq!(groups[group].size(), domain * TAPSET.group_size(group));
        }
        assert_eq!(globals[GLOBAL_OUT].size(), CircuitImpl::OUTPUT_SIZE);
        assert_eq!(globals[GLOBAL_MIX].size(), CircuitImpl::MIX_SIZE);
        let powers = self
            .hal
            .copy_from_extelem("rv32im_poly_mix", &map_pow(poly_mix, POLY_MIX_POWERS));
        let rou = Val::ROU_FWD[po2 + 2];
        let normalization: Vec<_> = (0..4)
            .map(|cycle| {
                let x = rou.pow(cycle);
                ((Val::new(3) * x).pow(steps) - Val::ONE).inv()
            })
            .collect();
        let factors = self
            .hal
            .copy_from_elem("rv32im_normalization", &normalization);
        // Synchronous completion precedes CPU access, input release, and the
        // next segment. The separate phases preserve the CPU oracle output.
        const THREADS: usize = 256;
        if self.evalfast {
            let table = self.hal.alloc_extelem("rv32im_evalfast_table", 627);
            let scratch = self.hal.alloc_elem("rv32im_evalfast_scratch", domain * 4);
            self.hal.dispatch(
                &self.kernels[0],
                &[
                    powers.as_arg(),
                    globals[GLOBAL_OUT].as_arg(),
                    globals[GLOBAL_MIX].as_arg(),
                    table.as_arg(),
                ],
                1,
                Some((MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1))),
            );
            self.hal.dispatch(
                &self.kernels[1],
                &[
                    check.as_arg(),
                    groups[REGISTER_GROUP_DATA].as_arg(),
                    groups[REGISTER_GROUP_ACCUM].as_arg(),
                    globals[GLOBAL_MIX].as_arg(),
                    globals[GLOBAL_OUT].as_arg(),
                    table.as_arg(),
                    scratch.as_arg(),
                    KernelArg::Integer(domain as u32),
                ],
                (2 * domain) as u64,
                Some((
                    MTLSize::new((2 * domain / THREADS) as u64, 1, 1),
                    MTLSize::new(THREADS as u64, 1, 1),
                )),
            );
            self.hal.dispatch(
                &self.kernels[2],
                &[
                    check.as_arg(),
                    scratch.as_arg(),
                    factors.as_arg(),
                    KernelArg::Integer(domain as u32),
                ],
                domain as u64,
                Some((
                    MTLSize::new((domain / THREADS) as u64, 1, 1),
                    MTLSize::new(THREADS as u64, 1, 1),
                )),
            );
        } else {
            self.hal.dispatch(
                &self.kernels[0],
                &[
                    check.as_arg(),
                    groups[REGISTER_GROUP_CODE].as_arg(),
                    groups[REGISTER_GROUP_DATA].as_arg(),
                    groups[REGISTER_GROUP_ACCUM].as_arg(),
                    globals[GLOBAL_MIX].as_arg(),
                    globals[GLOBAL_OUT].as_arg(),
                    powers.as_arg(),
                    factors.as_arg(),
                    KernelArg::Integer(po2 as u32),
                    KernelArg::Integer(domain as u32),
                ],
                domain as u64,
                Some((
                    MTLSize::new((domain / THREADS) as u64, 1, 1),
                    MTLSize::new(THREADS as u64, 1, 1),
                )),
            );
        }
        if std::env::var("RISC0_RV32IM_METAL_CHECK_CPU").as_deref() == Ok("1") {
            // All GPU writes completed above. The shared buffers remain alive
            // and immutable throughout these CPU oracle reads; no extra trace
            // copies or second HAL are needed.
            let view = |buffer: &BufferImpl<Val>| unsafe {
                std::slice::from_raw_parts(buffer.as_ptr().cast::<Val>(), buffer.size())
            };
            let data = view(groups[REGISTER_GROUP_DATA]);
            let accum = view(groups[REGISTER_GROUP_ACCUM]);
            let mix = view(globals[GLOBAL_MIX]);
            let out = view(globals[GLOBAL_OUT]);
            let actual = view(check);
            let pows = map_pow(poly_mix, POLY_MIX_POWERS);
            (0..domain).into_par_iter().for_each(|cycle| {
                let args = [accum.as_ptr(), data.as_ptr(), out.as_ptr(), mix.as_ptr()];
                let mut total = ExtVal::ZERO;
                ffi_wrap(|| unsafe { risc0_circuit_rv32im_cpu_poly_fp(cycle, domain,
                    pows.as_ptr(), args.as_ptr(), &mut total) }).expect("CPU constraint oracle failed");
                let x = rou.pow(cycle);
                let y = (Val::new(3) * x).pow(steps);
                let expected = total * (y - Val::ONE).inv();
                for component in 0..4 {
                    assert_eq!(actual[component * domain + cycle], expected.elems()[component],
                        "RV32IM Metal differs from CPU: po2={po2}, cycle={cycle}, component={component}");
                }
            });
            CPU_CHECKS.fetch_add(1, Ordering::Relaxed);
            tracing::info!(
                po2,
                domain,
                "RV32IM Metal valid-witness CPU differential passed"
            );
        }
        CHECKS.fetch_add(1, Ordering::Relaxed);
        POINTS.fetch_add(domain as u64, Ordering::Relaxed);
        tracing::info!(po2, domain, "RV32IM Metal constraint check completed");
    }
    fn accumulate(
        &self,
        _: &AccumPreflight,
        _: &BufferImpl<Val>,
        _: &BufferImpl<Val>,
        _: &BufferImpl<Val>,
        _: &BufferImpl<Val>,
        _: &BufferImpl<Val>,
        _: usize,
    ) {
        unreachable!("RV32IM uses CircuitAccumulator");
    }
}

pub(crate) fn segment_prover() -> Result<Box<dyn SegmentProver>> {
    metal::objc::rc::autoreleasepool(|| {
        let hal = Arc::new(MetalHalPoseidon2::new());
        ensure!(
            hal.device.has_unified_memory(),
            "RV32IM Metal requires unified memory"
        );
        let circuit = Arc::new(MetalCircuitHal::new(hal.clone())?);
        let factory = move || (hal.clone(), circuit.clone());
        Ok(Box::new(SegmentProverImpl::new(factory)) as Box<dyn SegmentProver>)
    })
}
