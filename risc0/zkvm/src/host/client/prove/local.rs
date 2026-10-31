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

use anyhow::Result;

use super::{Executor, Prover, ProverOpts};
use crate::{
    get_prover_server,
    host::server::{exec::executor::ExecutorImpl, session::NullSegmentRef},
    ExecutorEnv, ProveInfo, Receipt, SegmentInfo, SessionInfo, VerifierContext,
};

/// A [Prover] implementation that selects a [ProverServer][crate::ProverServer] by calling
/// [get_prover_server].
pub struct LocalProver {
    name: String,
    #[cfg(feature = "metal-hybrid")]
    metal_staging_bytes: Option<usize>,
    #[cfg(feature = "metal-hybrid")]
    metal_request: std::sync::Mutex<()>,
    #[cfg(feature = "metal-hybrid")]
    metal_reports: std::sync::Arc<std::sync::Mutex<Vec<MetalSegmentReport>>>,
}

/// Actual, completed Poseidon2 Merkle work for one locally proved segment.
/// This diagnostic is separate from all receipt and worker wire formats.
#[cfg(feature = "metal-hybrid")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct MetalSegmentReport {
    /// Segment index within this proof request.
    pub segment_index: u32,
    /// Segment cycle bound as a power of two.
    pub po2: u32,
    /// Successfully completed Metal command buffers, excluding empty work.
    pub completed_dispatches: u64,
    /// Merkle fold calls delegated to Metal.
    pub fold_calls: u64,
    /// Digests consumed by those folds.
    pub input_digests: u64,
    /// Digests produced by those folds.
    pub output_digests: u64,
    /// Maximum temporary shared Metal buffer allocation.
    pub peak_staging_bytes: usize,
    /// Configured cap for temporary shared Metal buffers.
    pub staging_limit_bytes: usize,
}

impl LocalProver {
    /// Construct a [LocalProver].
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            #[cfg(feature = "metal-hybrid")]
            metal_staging_bytes: None,
            #[cfg(feature = "metal-hybrid")]
            metal_request: Default::default(),
            #[cfg(feature = "metal-hybrid")]
            metal_reports: Default::default(),
        }
    }
    /// Require Metal for segment Poseidon2 Merkle folds. Other stages remain CPU.
    /// The staging budget must be between 4 KiB and 16 MiB. Device, compilation,
    /// allocation or dispatch failures fail the request without CPU fallback.
    /// This selection is local only and is never serialized to a remote prover.
    #[cfg(feature = "metal-hybrid")]
    pub fn new_metal(name: &str, staging_bytes: usize) -> Self {
        Self {
            metal_staging_bytes: Some(staging_bytes),
            ..Self::new(name)
        }
    }

    /// Completed segment work from the latest prove request (including partial
    /// progress if a later stage failed). Compression does not change it.
    #[cfg(feature = "metal-hybrid")]
    pub fn metal_stage_reports(&self) -> Vec<MetalSegmentReport> {
        self.metal_reports.lock().unwrap().clone()
    }
}

impl Prover for LocalProver {
    fn prove_with_ctx(
        &self,
        env: ExecutorEnv<'_>,
        ctx: &VerifierContext,
        elf: &[u8],
        opts: &ProverOpts,
    ) -> Result<ProveInfo> {
        #[cfg(feature = "metal-hybrid")]
        if let Some(staging_bytes) = self.metal_staging_bytes {
            use crate::host::server::prove::prover_impl::ProverImpl;
            use crate::ProverServer;
            // Keep diagnostics request-local even when callers share this prover.
            // Acquire before clearing counters or waiting for the Metal device.
            let _request = self.metal_request.lock().unwrap();
            self.metal_reports.lock().unwrap().clear();
            anyhow::ensure!(!opts.dev_mode(), "Metal acceptance requires real proving");
            let prover =
                ProverImpl::new_metal(opts.clone(), staging_bytes, self.metal_reports.clone())?;
            return prover.prove_with_ctx(env, ctx, elf);
        }
        get_prover_server(opts)?.prove_with_ctx(env, ctx, elf)
    }

    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn compress(&self, opts: &ProverOpts, receipt: &Receipt) -> Result<Receipt> {
        get_prover_server(opts)?.compress(opts, receipt)
    }
}

impl Executor for LocalProver {
    fn execute(&self, env: ExecutorEnv<'_>, elf: &[u8]) -> Result<SessionInfo> {
        let mut segments = Vec::new();
        let session = ExecutorImpl::from_elf(env, elf)
            .unwrap()
            .run_with_callback(|segment| {
                segments.push(SegmentInfo {
                    po2: segment.po2() as u32,
                    cycles: segment.user_cycles(),
                });
                Ok(Box::new(NullSegmentRef))
            })?;

        let receipt_claim = session.claim()?;
        Ok(SessionInfo {
            segments,
            journal: session.journal.unwrap_or_default(),
            exit_code: session.exit_code,
            receipt_claim: Some(receipt_claim),
        })
    }
}
