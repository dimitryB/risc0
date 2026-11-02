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

// Used by the CPU and Intel paths; the CUDA/ROCm path below takes an Arc.
#[cfg_attr(any(feature = "cuda", feature = "rocm"), allow(unused_imports))]
use std::rc::Rc;

use risc0_binfmt::{MemoryImage, Program};
use risc0_zkp::field::Elem;
use test_log::test;

use crate::{
    execute::{
        testutil::{self, NullSyscall, DEFAULT_SESSION_LIMIT},
        DEFAULT_SEGMENT_LIMIT_PO2,
    },
    prove::{hal::StepMode, witgen::WitnessGenerator, PreflightResults},
    zirgen::circuit::{ExtVal, REGCOUNT_DATA},
    MAX_INSN_CYCLES,
};

fn run_preflight(program: Program) {
    let image = MemoryImage::new_kernel(program);
    let result = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        MAX_INSN_CYCLES,
        DEFAULT_SESSION_LIMIT,
        &NullSyscall,
        None,
    )
    .unwrap();
    let segments = result.segments;
    let segment = segments.first().unwrap();

    let mut rng = rand::rng();
    let rand_z = ExtVal::random(&mut rng);

    segment.preflight(rand_z).unwrap();
}

#[test]
fn basic() {
    run_preflight(testutil::kernel::basic());
}

#[test]
fn simple_loop() {
    run_preflight(testutil::kernel::simple_loop(500000));
}

fn fwd_rev_ab_test(program: Program) {
    let image = MemoryImage::new_kernel(program);

    let session = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        MAX_INSN_CYCLES,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();

    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            use risc0_zkp::hal::cuda::CudaHalPoseidon2;
            use crate::prove::hal::cuda::CudaCircuitHalPoseidon2;
            let hal = std::sync::Arc::new(CudaHalPoseidon2::new());
            let circuit_hal = CudaCircuitHalPoseidon2::new(hal.clone());
        } else if #[cfg(feature = "rocm")] {
            use risc0_zkp::hal::hip::HipHalPoseidon2;
            use crate::prove::hal::hip::HipCircuitHalPoseidon2;
            let hal = std::sync::Arc::new(HipHalPoseidon2::new());
            let circuit_hal = HipCircuitHalPoseidon2::new(hal.clone());
        // } else if #[cfg(any(all(target_os = "macos", target_arch = "aarch64"), target_os = "ios"))] {
        //     use risc0_zkp::hal::metal::MetalHalSha256;
        //     use crate::prove::hal::metal::MetalCircuitHal;
        //     let hal = Rc::new(MetalHalSha256::new());
        //     let circuit_hal = MetalCircuitHal::new(hal.clone());
        } else {
            let suite = risc0_zkp::core::hash::poseidon2::Poseidon2HashSuite::new_suite();
            let hal = Rc::new(risc0_zkp::hal::cpu::CpuHal::new(suite));
            let circuit_hal = crate::prove::hal::cpu::CpuCircuitHal;
        }
    }

    let mut rng = rand::rng();
    let rand_z = ExtVal::random(&mut rng);

    let segments = session.segments;
    for segment in segments {
        tracing::debug!("fwd");

        let preflight_results = PreflightResults::new(&segment, rand_z).unwrap();

        let fwd_witgen = WitnessGenerator::new(
            hal.as_ref(),
            &circuit_hal,
            preflight_results.clone(),
            StepMode::SeqForward,
        )
        .unwrap();
        tracing::debug!("rev");
        let rev_witgen = WitnessGenerator::new(
            hal.as_ref(),
            &circuit_hal,
            preflight_results.clone(),
            StepMode::SeqReverse,
        )
        .unwrap();
        let cycles = 1 << segment.po2;
        let fwd_vec = fwd_witgen.data.as_ref().unwrap().to_vec();
        let rev_vec = rev_witgen.data.as_ref().unwrap().to_vec();
        for row in 0..cycles {
            let fwd_row = &fwd_vec[row * REGCOUNT_DATA..row * REGCOUNT_DATA + REGCOUNT_DATA];
            let rev_row = &rev_vec[row * REGCOUNT_DATA..row * REGCOUNT_DATA + REGCOUNT_DATA];
            assert_eq!(fwd_row, rev_row, "cycle: {row}");
        }
    }
}

#[test]
fn fwd_rev_ab_basic() {
    fwd_rev_ab_test(testutil::kernel::basic());
}

#[test]
fn fwd_rev_ab_split() {
    fwd_rev_ab_test(testutil::kernel::simple_loop(2000));
}

/// Intel parallel witgen must match canonical CPU sequential witgen bit-for-bit,
/// and be deterministic across repeated runs. Knobs:
///   RISC0_WITGEN_AB_ITERS (simple_loop count, default 300_000)
///   RISC0_WITGEN_AB_PO2   (segment limit po2, default 20)
///   RISC0_WITGEN_AB_REPS  (Intel repetitions per segment, default 5)
#[cfg(feature = "intel")]
#[test]
#[ignore = "requires an Intel GPU; run explicitly"]
fn intel_parallel_matches_cpu_seq() {
    let iters = std::env::var("RISC0_WITGEN_AB_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300_000);
    intel_ab(testutil::kernel::simple_loop(iters));
}

/// Under the RISC Zero 3.0 guest ABI, sys_poseidon2 passes word addresses while
/// the circuit reads byte addresses, so any call with a non-null pointer cannot
/// be proven on any backend (fixed upstream only in v5, risc0#3431). Execution
/// matches upstream 3.0; preflight must reject the segment with a clear error
/// instead of producing an inconsistent witness.
#[test]
fn poseidon2_ecall_rejected_by_preflight() {
    let image = MemoryImage::new_kernel(testutil::kernel::poseidon2(2));
    let session = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        MAX_INSN_CYCLES,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();
    let mut rng = rand::rng();
    let rand_z = ExtVal::random(&mut rng);
    let err = PreflightResults::new(&session.segments[0], rand_z)
        .err()
        .expect("preflight must reject a non-null sys_poseidon2 call");
    assert!(
        err.to_string().contains("sys_poseidon2"),
        "unexpected error: {err}"
    );
}

#[cfg(feature = "intel")]
fn intel_ab(program: Program) {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use risc0_zkp::hal::intel::IntelHalPoseidon2;

    use crate::prove::hal::intel::IntelCircuitHalPoseidon2;

    fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
    let po2: usize = env_or("RISC0_WITGEN_AB_PO2", DEFAULT_SEGMENT_LIMIT_PO2);
    let reps: usize = env_or("RISC0_WITGEN_AB_REPS", 5);

    let image = MemoryImage::new_kernel(program);
    let session = testutil::execute(
        image,
        po2,
        MAX_INSN_CYCLES,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();

    let intel_hal = Arc::new(IntelHalPoseidon2::new());
    let intel_circuit = IntelCircuitHalPoseidon2::new(intel_hal.clone());
    let suite = risc0_zkp::core::hash::poseidon2::Poseidon2HashSuite::new_suite();
    let cpu_hal = Rc::new(risc0_zkp::hal::cpu::CpuHal::new(suite));
    let cpu_circuit = crate::prove::hal::cpu::CpuCircuitHal;

    let mut rng = rand::rng();
    let rand_z = ExtVal::random(&mut rng);
    let mut total_bad = 0usize;

    eprintln!(
        "[witgen-ab] {} segment(s), po2 limit={po2}",
        session.segments.len()
    );
    for segment in session.segments {
        let pf = PreflightResults::new(&segment, rand_z).unwrap();
        let cycles = 1usize << segment.po2;
        let cpu = WitnessGenerator::new(
            cpu_hal.as_ref(),
            &cpu_circuit,
            pf.clone(),
            StepMode::SeqForward,
        )
        .unwrap()
        .data
        .as_ref()
        .unwrap()
        .to_vec();

        let mut first_intel: Option<Vec<_>> = None;
        for rep in 0..reps {
            let intel = WitnessGenerator::new(
                intel_hal.as_ref(),
                &intel_circuit,
                pf.clone(),
                StepMode::Parallel,
            )
            .unwrap()
            .data
            .as_ref()
            .unwrap()
            .to_vec();

            let mut by_col: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
            let mut bad = 0usize;
            for row in 0..cycles {
                for col in 0..REGCOUNT_DATA {
                    let i = row * REGCOUNT_DATA + col;
                    if intel[i] != cpu[i] {
                        bad += 1;
                        by_col.entry(col).or_insert((0, row)).0 += 1;
                    }
                }
            }
            let run_to_run = first_intel
                .as_ref()
                .map(|f| f.iter().zip(&intel).filter(|(a, b)| a != b).count());
            eprintln!(
                "[witgen-ab] seg={} po2={} rep={rep}: {bad} cells differ from CPU across {} cols; \
                 vs Intel rep0: {:?}",
                segment.index,
                segment.po2,
                by_col.len(),
                run_to_run
            );
            for (col, (n, first_row)) in by_col.iter().take(16) {
                eprintln!("    col {col:4}: {n:8} rows differ (first row {first_row})");
            }
            total_bad += bad;
            if first_intel.is_none() {
                first_intel = Some(intel);
            }
        }
    }
    assert_eq!(
        total_bad, 0,
        "Intel parallel witgen diverged from CPU sequential witgen"
    );
}

/// HIP witness generation must match the CPU's bit for bit (the fwd/rev tests compare the GPU
/// with itself). The accumulator is left to the prove tests: the CPU's fails on Windows.
#[cfg(feature = "rocm")]
#[test]
fn hip_matches_cpu() {
    use std::sync::Arc;

    use risc0_zkp::hal::hip::HipHalPoseidon2;

    use crate::{prove::hal::hip::HipCircuitHalPoseidon2, zirgen::circuit::Val};

    fn first_diff(label: &str, cpu: &[Val], gpu: &[Val], width: usize) -> usize {
        let bad: Vec<usize> = (0..cpu.len()).filter(|&i| cpu[i] != gpu[i]).collect();
        if let Some(&i) = bad.first() {
            eprintln!(
                "[hip-ab] {label}: {} of {} differ; first row {} col {}: cpu={:?} gpu={:?}",
                bad.len(),
                cpu.len(),
                i / width,
                i % width,
                cpu[i],
                gpu[i]
            );
        }
        bad.len()
    }

    let image = MemoryImage::new_kernel(testutil::kernel::simple_loop(2000));
    let session = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        MAX_INSN_CYCLES,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();

    let gpu_hal = Arc::new(HipHalPoseidon2::new());
    let gpu_circuit = HipCircuitHalPoseidon2::new(gpu_hal.clone());
    let suite = risc0_zkp::core::hash::poseidon2::Poseidon2HashSuite::new_suite();
    let cpu_hal = Rc::new(risc0_zkp::hal::cpu::CpuHal::new(suite));
    let cpu_circuit = crate::prove::hal::cpu::CpuCircuitHal;

    let mut rng = rand::rng();
    let rand_z = ExtVal::random(&mut rng);
    let mut bad = 0;
    for segment in session.segments {
        let pf = PreflightResults::new(&segment, rand_z).unwrap();
        let cpu = WitnessGenerator::new(
            cpu_hal.as_ref(),
            &cpu_circuit,
            pf.clone(),
            StepMode::SeqForward,
        )
        .unwrap();
        let gpu =
            WitnessGenerator::new(gpu_hal.as_ref(), &gpu_circuit, pf, StepMode::Parallel).unwrap();
        bad += first_diff(
            "data",
            &cpu.data.as_ref().unwrap().to_vec(),
            &gpu.data.as_ref().unwrap().to_vec(),
            REGCOUNT_DATA,
        );
    }
    assert_eq!(bad, 0);
}
