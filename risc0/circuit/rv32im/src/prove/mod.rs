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

mod hal;
#[cfg(test)]
mod tests;
mod witgen;

use anyhow::Result;
use cfg_if::cfg_if;
use risc0_core::scope;

use crate::execute::segment::Segment;

pub use risc0_zkp::hal::cpu::MetalFoldStats;
pub use witgen::PreflightResults;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub use self::hal::metal::{metal_rv32im_stats, metal_rv32im_kernel_source, MetalRv32imStats};

/// The `rand_z` of a segment preflight ([PreflightResults::new]): random, or one with the
/// `witgen_debug` feature (reproducible witnesses).
pub fn preflight_rand_z() -> crate::zirgen::circuit::ExtVal {
    cfg_if! {
        if #[cfg(feature = "witgen_debug")] {
            <crate::zirgen::circuit::ExtVal as risc0_core::field::Elem>::ONE
        } else {
            <crate::zirgen::circuit::ExtVal as risc0_core::field::Elem>::random(&mut rand::rng())
        }
    }
}

const GLOBAL_MIX: usize = 0;
const GLOBAL_OUT: usize = 1;

pub type Seal = Vec<u32>;

pub trait SegmentProver {
    /// Cumulative GPU execution evidence, available only for explicit hybrid provers.
    fn metal_stats(&self) -> Option<MetalFoldStats> {
        None
    }

    fn prove(&self, segment: &Segment) -> Result<Seal> {
        scope!("prove");
        let results = self.preflight(segment)?;
        self.prove_core(results)
    }

    fn preflight(&self, segment: &Segment) -> Result<PreflightResults>;

    fn prove_core(&self, preflight_results: PreflightResults) -> Result<Seal>;

    /// Pipeline-aware proving: performs the main phase for this segment
    /// (overlapping GPU eval_check from the previous segment), completes the
    /// previous segment's deferred finalize, then launches eval_check for
    /// the current segment.
    ///
    /// Returns the seal for the PREVIOUS segment (None on first call).
    fn prove_begin(&self, preflight_results: PreflightResults) -> Result<Option<Seal>> {
        // Default: sequential, no pipelining.
        Ok(Some(self.prove_core(preflight_results)?))
    }

    /// Complete the last segment's deferred finalize and return its seal.
    /// Only valid after at least one prove_begin call.
    fn prove_end(&self) -> Result<Seal> {
        anyhow::bail!("prove_end: no pending work (default impl)")
    }

    /// Discards any seal left pending by an earlier pipeline, e.g. a session
    /// that failed between prove_begin and prove_end. Call before a new one.
    fn reset_pipeline(&self) {}
}

/// Frees the Intel eval_check scratch slab (~1.4 GB at po2=20) between
/// proving sessions; it is reallocated by the next eval_check. Best effort:
/// a failure is logged, not returned, since the proof is already complete.
#[cfg(feature = "intel")]
pub fn release_intel_scratch() {
    if let Err(e) = risc0_sys::ffi_wrap(|| unsafe {
        risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_eval_check_release()
    }) {
        tracing::warn!("releasing Intel eval_check scratch failed: {e}");
    }
}

pub fn segment_prover() -> Result<Box<dyn SegmentProver>> {
    match std::env::var("RISC0_SEGMENT_HAL").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("cpu") => {},
        Ok("metal") => {
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            return self::hal::metal::segment_prover();
            #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
            anyhow::bail!("RV32IM Metal is not compiled into this engine");
        },
        _ => anyhow::bail!("RISC0_SEGMENT_HAL must be cpu or metal"),
    }
    cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::hal::cuda::segment_prover()
        } else if #[cfg(feature = "rocm")] {
            self::hal::hip::segment_prover()
        // } else if #[cfg(any(all(target_os = "macos", target_arch = "aarch64"), target_os = "ios"))] {
        // self::hal::metal::segment_prover(hashfn)
        } else if #[cfg(feature = "intel")] {
            self::hal::intel::segment_prover()
        } else {
            self::hal::cpu::segment_prover()
        }
    }
}

/// Explicit CPU segment prover with only Poseidon2 Merkle folding on Metal.
#[cfg(all(feature = "metal-hybrid", target_os = "macos", target_arch = "aarch64"))]
pub fn metal_segment_prover(staging_bytes: usize) -> Result<Box<dyn SegmentProver>> {
    self::hal::cpu::metal_segment_prover(staging_bytes)
}

/// Trigger HIP module loading so the first kernel launch doesn't stall.
/// This is a no-op on non-ROCm builds. Mirrors cuda_warmup() but adds
/// hipInit/hipSetDevice for HIP runtime initialization.
#[cfg(feature = "rocm")]
pub fn rocm_warmup() {
    use risc0_sys::ffi_wrap;
    // Initialize HIP runtime and select the thread's device.
    risc0_zkp::hal::hip::auto_configure_hw_queues();
    unsafe {
        risc0_sys::hip::hipInit(0);
        risc0_sys::hip::hipSetDevice(risc0_zkp::hal::hip::get_device_for_thread());
    }
    // Warmup rv32im circuit kernels (par_stepExec, stepAccum, finalizeAccum, eval_check).
    // The C function names are the same regardless of CUDA or HIP compilation.
    let _ = ffi_wrap(|| unsafe { risc0_circuit_rv32im_sys::risc0_circuit_rv32im_cuda_warmup() });
    let _ = ffi_wrap(|| unsafe {
        risc0_circuit_rv32im_sys::risc0_circuit_rv32im_cuda_warmup_eval_check()
    });
    // Warmup sppark NTT kernels (normally loaded during HipHal::new -> sppark_init)
    let err = unsafe { risc0_sys::cuda::sppark_init() };
    if err.code != 0 {
        tracing::warn!("sppark_init warmup failed: {err}");
    }
    // Warmup poseidon2 kernels
    let err = unsafe { risc0_sys::cuda::sppark_poseidon2_init() };
    if err.code != 0 {
        tracing::warn!("sppark_poseidon2_init warmup failed: {err}");
    }
    // Warmup risc0-zkp kernels (eltwise, sha, bit_reverse, etc.)
    extern "C" {
        fn risc0_zkp_cuda_warmup() -> *const std::os::raw::c_char;
    }
    let _ = ffi_wrap(|| unsafe { risc0_zkp_cuda_warmup() });
}

/// Trigger CUDA module loading so the first kernel launch doesn't stall.
/// This is a no-op on non-CUDA builds.
#[cfg(feature = "cuda")]
pub fn cuda_warmup() {
    use risc0_sys::ffi_wrap;
    // Warmup rv32im circuit kernels (par_stepExec, stepAccum, finalizeAccum, eval_check)
    let _ = ffi_wrap(|| unsafe { risc0_circuit_rv32im_sys::risc0_circuit_rv32im_cuda_warmup() });
    let _ = ffi_wrap(|| unsafe {
        risc0_circuit_rv32im_sys::risc0_circuit_rv32im_cuda_warmup_eval_check()
    });
    // Warmup sppark NTT kernels (normally loaded during CudaHal::new → sppark_init)
    let err = unsafe { risc0_sys::cuda::sppark_init() };
    if err.code != 0 {
        tracing::warn!("sppark_init warmup failed: {err}");
    }
    // Warmup poseidon2 kernels
    let err = unsafe { risc0_sys::cuda::sppark_poseidon2_init() };
    if err.code != 0 {
        tracing::warn!("sppark_poseidon2_init warmup failed: {err}");
    }
    // Warmup risc0-zkp kernels (eltwise, sha, bit_reverse, etc.)
    extern "C" {
        fn risc0_zkp_cuda_warmup() -> *const std::os::raw::c_char;
    }
    let _ = ffi_wrap(|| unsafe { risc0_zkp_cuda_warmup() });
}
