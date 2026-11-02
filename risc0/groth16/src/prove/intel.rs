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

//! Groth16 on Intel GPUs (`intel` feature): the SYCL prover in `risc0-groth16-sys`
//! (`kernels/intel`), with the witness computed in-process (see [super::fast]).
//!
//! It uses the rzup `risc0-groth16` component's proving key, preprocessed coefficients and
//! fuzzed MSM results, as the CUDA prover does, and produces the same kind of proof.
//! `RISC0_GROTH16_INTEL_PROVER=docker` selects the Docker prover instead.

use std::path::Path;

use anyhow::{anyhow, Context as _, Result};
use risc0_groth16_sys::{ProverParams, SetupParams};
use tempfile::tempdir;

use super::component::component_dir;
use crate::{ProofJson, Seal};

/// Whether to prove on the GPU (the default) rather than with the Docker prover.
pub(crate) fn enabled() -> bool {
    std::env::var("RISC0_GROTH16_INTEL_PROVER").as_deref() != Ok("docker")
}

fn preload(dir: &Path) -> Result<()> {
    let setup = SetupParams::new(dir).context("failed to create groth16 setup params")?;
    risc0_groth16_sys::intel_preload(&setup).context("failed to load the groth16 proving key")
}

/// Start loading the proving key onto the GPU in the background.
pub(crate) fn prestart() -> Result<()> {
    let dir = component_dir()?;
    std::thread::spawn(move || {
        if let Err(e) = preload(&dir) {
            tracing::warn!("groth16 proving key preload failed: {e:#}");
        }
    });
    Ok(())
}

/// Load the proving key onto the GPU now.
pub(crate) fn preload_srs() -> Result<()> {
    preload(&component_dir()?)
}

/// Free the proving key, coefficients and scratch cached on the GPU.
pub(crate) fn release_srs() -> Result<()> {
    risc0_groth16_sys::intel_release().context("failed to release the groth16 prover")
}

pub(crate) fn shrink_wrap(identity_p254_seal_bytes: &[u8]) -> Result<Seal> {
    let dir = component_dir()?;
    let setup = SetupParams::new(&dir).context("failed to create groth16 setup params")?;

    let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
    let t0 = std::time::Instant::now();
    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1000.0;

    // The proving key's upload (first use only) overlaps the witness computation.
    let loader = {
        let dir = dir.clone();
        std::thread::spawn(move || preload(&dir))
    };
    let witness = super::fast::witness(identity_p254_seal_bytes)
        .context("failed to calculate the groth16 witness")?;
    let witness_ms = ms(t0);
    loader
        .join()
        .map_err(|_| anyhow!("the groth16 proving key loader panicked"))??;
    let loaded_ms = ms(t0);

    let work_dir = tempdir().context("failed to create a temporary directory")?;
    let params = ProverParams::new(work_dir.path(), witness.as_ptr() as *const u8)
        .context("failed to create groth16 prover parameters")?;
    risc0_groth16_sys::intel_prove(&params, &setup).context("groth16 prove failed")?;
    if verbose {
        eprintln!(
            "[groth16-intel] witness={witness_ms:.1}ms key loaded={loaded_ms:.1}ms              proved={:.1}ms",
            ms(t0)
        );
    }

    let contents = std::fs::read_to_string(params.proof_path.as_path())
        .context("failed to read the groth16 proof")?;
    let proof_json: ProofJson =
        serde_json::from_str(&contents).context("failed to decode the groth16 proof JSON")?;
    proof_json.try_into()
}
