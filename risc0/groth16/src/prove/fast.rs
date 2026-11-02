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

//! Faster Docker-based shrink_wrap (`intel` feature).
//!
//! The pinned prover image runs the circom C++ witness generator (~2.8 s,
//! single-threaded) and then gnark. This module instead
//!  * computes the witness in-process from the rzup `risc0-groth16` witness
//!    graph (same circuit; the witness is byte-identical to the C++
//!    generator's) with a Montgomery-form evaluator (~0.3 s), and
//!  * starts the image's gnark `prover` early (it spends ~1 s loading the
//!    constraint system and proving key) and feeds it the witness through a
//!    named pipe in the mounted work directory.
//! The prover, proving key and circuit are the pinned image's, so proofs are
//! unchanged. If the rzup component is missing, the caller falls back to the
//! plain Docker path.
//!
//! Cleanup: the parent opens the pipe read-write as soon as the prover is
//! started, so a prover that has opened the pipe sees EOF and exits when this
//! process closes it (or dies). Dropping an unused handle also `docker kill`s
//! the (named) container, and the prover runs under `timeout` so that a
//! container cannot outlive a killed parent by more than PROVER_TIMEOUT_S.

use super::{docker::IMAGE, witness};
use crate::{ProofJson, Seal};
use anyhow::{bail, Context, Result};
use std::{
    fs::File,
    io::BufWriter,
    process::{Child, Command, Stdio},
    sync::Mutex,
};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Pre-started prover
// ---------------------------------------------------------------------------

/// Upper bound on a prover container's lifetime (a proof takes ~5 s).
const PROVER_TIMEOUT_S: u32 = 600;

struct Prover {
    child: Option<Child>,
    name: String,
    dir: TempDir,
    fifo: Option<File>,
}

impl Drop for Prover {
    fn drop(&mut self) {
        // Closing our end of the pipe gives the prover EOF; it then exits
        // and docker removes the container (--rm).
        self.fifo.take();
        if let Some(mut child) = self.child.take() {
            let _ = Command::new("docker")
                .args(["kill", &self.name])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            let _ = child.wait();
        }
    }
}

static PRESTARTED: Mutex<Option<Prover>> = Mutex::new(None);

fn start_prover() -> Result<Prover> {
    let dir = tempfile::tempdir()?;
    let fifo_path = dir.path().join("witness.wtns");
    let status = Command::new("mkfifo").arg(&fifo_path).status()?;
    if !status.success() {
        bail!("mkfifo failed");
    }
    static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let name = format!(
        "risc0-groth16-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let child = Command::new("docker")
        .arg("run")
        .arg("--rm")
        .arg("--name")
        .arg(&name)
        .arg("-v")
        .arg(format!("{}:/mnt", dir.path().to_string_lossy()))
        .arg("-w")
        .arg("/app")
        .arg("--entrypoint")
        .arg("timeout")
        .arg(IMAGE)
        .arg(PROVER_TIMEOUT_S.to_string())
        .args([
            "prover",
            "stark_verify.cs",
            "stark_verify_final.pk.dmp",
            "/mnt/witness.wtns",
            "/mnt/proof.json",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::from(File::create(dir.path().join("prover.log"))?))
        .spawn()
        .context("failed to start the groth16 prover container")?;
    // Read-write open of a FIFO does not block on Linux and keeps a writer
    // alive until we are done (see the module docs).
    let fifo = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&fifo_path)?;
    Ok(Prover {
        child: Some(child),
        name,
        dir,
        fifo: Some(fifo),
    })
}

/// Start the prover container ahead of shrink_wrap so that it loads the
/// proving key in the background. No-op if one is already waiting.
pub(crate) fn prestart() -> Result<()> {
    let mut slot = PRESTARTED.lock().unwrap();
    if slot.is_none() {
        *slot = Some(start_prover()?);
    }
    Ok(())
}

pub(crate) fn shrink_wrap(seal_bytes: &[u8]) -> Result<Seal> {
    let t0 = std::time::Instant::now();
    witness::preload()?;
    let mut prover = match PRESTARTED.lock().unwrap().take() {
        Some(p) => p,
        None => start_prover()?,
    };
    {
        let fifo = prover.fifo.take().unwrap();
        witness::write(seal_bytes, BufWriter::with_capacity(1 << 20, fifo))?;
    }
    let t_wit = t0.elapsed();
    let output = prover
        .child
        .take()
        .unwrap()
        .wait_with_output()
        .context("groth16 prover container")?;
    if !output.status.success() {
        let log = std::fs::read_to_string(prover.dir.path().join("prover.log")).unwrap_or_default();
        bail!(
            "groth16 prover failed ({:?}): {}",
            output.status.code(),
            log.lines().last().unwrap_or("")
        );
    }
    let contents = std::fs::read_to_string(prover.dir.path().join("proof.json"))
        .context("groth16 prover produced no proof")?;
    tracing::debug!(
        "shrink_wrap (fast): witness {:.1} ms, total {:.1} ms",
        t_wit.as_secs_f64() * 1000.0,
        t0.elapsed().as_secs_f64() * 1000.0
    );
    let proof_json: ProofJson = serde_json::from_str(&contents)?;
    proof_json.try_into()
}

pub(crate) use super::witness::{graph_path, preload};
