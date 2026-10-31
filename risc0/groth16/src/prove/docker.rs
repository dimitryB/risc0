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

//! This module contains functions for running a Groth16 prover using Docker.
//!
//! Docker is used here as a way to provide [gnark] and the required prover key in a single package.
//! Proving with Groth16 is currently only supported using Docker.
//!
//! [gnark]: https://github.com/Consensys/gnark

use std::{
    path::Path,
    process::{Command, Stdio},
};

use anyhow::{bail, Result};
use tempfile::tempdir;

use super::seal_to_json::to_json;
use crate::{ProofJson, Seal};

/// Groth16 a given seal of an `identity_p254` receipt into a Groth16 `Seal`.
/// Requires running Docker on an x86 architecture.
pub(crate) fn shrink_wrap(identity_p254_seal_bytes: &[u8]) -> Result<Seal> {
    if !is_docker_installed() {
        bail!("Please install docker first.")
    }

    // Intel builds: in-process witness + pre-started prover (same image).
    #[cfg(feature = "intel")]
    if fast_enabled() {
        match super::fast::preload() {
            Ok(()) => return super::fast::shrink_wrap(identity_p254_seal_bytes),
            Err(e) => {
                tracing::warn!("fast shrink_wrap unavailable, using the plain Docker path: {e:#}")
            }
        }
    }

    let tmp_dir = tempdir()?;
    let work_dir = std::env::var("RISC0_WORK_DIR");
    let work_dir = work_dir.as_ref().map(Path::new).unwrap_or(tmp_dir.path());

    tracing::debug!("seal-to-json");
    std::fs::write(work_dir.join("seal.r0"), identity_p254_seal_bytes)?;
    let seal_path = work_dir.join("input.json");
    let proof_path = work_dir.join("proof.json");
    let seal_json = to_json(identity_p254_seal_bytes)?;
    std::fs::write(seal_path, seal_json)?;

    tracing::debug!("risc0-groth16-prover");
    let mut command = Command::new("docker");
    command.arg("run");
    // A native Mac supervisor owns this exact container and removes it if the
    // worker is killed. Other platforms retain their existing invocation.
    #[cfg(target_os = "macos")]
    if let Ok(name) = std::env::var("RISC0_GROTH16_CONTAINER_NAME") {
        command.args([
            "--name",
            &name,
            "--platform",
            "linux/amd64",
            "--pull",
            "never",
            "--network",
            "none",
            "--memory",
            "24g",
            "--cpus",
            "6",
            "--entrypoint",
            "timeout",
        ]);
    }
    command
        .arg("--rm")
        .arg("-v")
        .arg(format!("{}:/mnt", work_dir.to_string_lossy()))
        .arg(IMAGE);
    #[cfg(target_os = "macos")]
    if std::env::var_os("RISC0_GROTH16_CONTAINER_NAME").is_some() {
        command.args(["1800", "/app/prover.sh"]);
    }
    let output = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;
    if !output.status.success() {
        bail!(
            "docker returned failure exit code: {:?}",
            output.status.code()
        );
    }

    tracing::debug!("Parsing proof");
    let contents = std::fs::read_to_string(proof_path)?;
    let proof_json: ProofJson = serde_json::from_str(&contents)?;
    proof_json.try_into()
}

pub(crate) const IMAGE: &str = "risczero/risc0-groth16-prover:v2025-04-03.1";

/// RISC0_GROTH16_FAST=0 disables the intel fast path.
#[cfg(feature = "intel")]
pub(crate) fn fast_enabled() -> bool {
    std::env::var("RISC0_GROTH16_FAST").as_deref() != Ok("0")
}

fn is_docker_installed() -> bool {
    Command::new("docker")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}
