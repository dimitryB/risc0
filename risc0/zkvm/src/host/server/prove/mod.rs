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

//! Run the zkVM guest and prove its results.

pub(crate) mod dev_mode;
pub(crate) mod keccak;
pub(crate) mod prover_impl;
#[cfg(test)]
mod tests;
pub(crate) mod union_peak;

use std::rc::Rc;

use anyhow::{anyhow, bail, ensure, Context, Result};
use risc0_core::field::baby_bear::{BabyBear, Elem, ExtElem};
use risc0_groth16::prove::shrink_wrap_with;
use risc0_zkp::hal::{CircuitHal, Hal};

use self::{dev_mode::DevModeProver, prover_impl::ProverImpl};
use crate::{
    claim::{receipt::UnionClaim, Unknown},
    host::prove_info::ProveInfo,
    receipt::{
        CompositeReceipt, Groth16Receipt, Groth16ReceiptVerifierParameters, InnerAssumptionReceipt,
        InnerReceipt, SegmentReceipt, SuccinctReceipt,
    },
    sha::Digestible,
    ExecutorEnv, PreflightResults, ProverOpts, Receipt, ReceiptClaim, ReceiptKind, Segment,
    Session, VerifierContext, WorkClaim,
};

/// Verify a Groth16 receipt this prover just produced, before handing it out.
/// A faulty shrink-wrap (prover, witness or GPU bug) then fails here, instead of
/// surfacing later as an invalid proof, e.g. on chain. Costs a few milliseconds.
pub(crate) fn check_groth16_receipt(receipt: &Groth16Receipt<ReceiptClaim>) -> Result<()> {
    receipt
        .verify_integrity_with_context(&VerifierContext::default())
        .map_err(|e| {
            // A bad seal may come from a damaged derived-data cache of the Groth16 prover:
            // discard those, so the next wrap rebuilds them instead of failing the same way.
            risc0_groth16::prove::invalidate_caches();
            anyhow!("Groth16 receipt failed verification: {e}")
        })
}

mod private {
    use super::{dev_mode::DevModeProver, prover_impl::ProverImpl};

    pub trait Sealed {}
    impl Sealed for ProverImpl {}
    impl Sealed for DevModeProver {}
}

/// A ProverServer can execute a given ELF binary and produce a [ProveInfo] which contains a
/// [Receipt][crate::Receipt] that can be used to verify correct computation.
pub trait ProverServer: private::Sealed {
    /// Prove the specified ELF binary.
    fn prove(&self, env: ExecutorEnv<'_>, elf: &[u8]) -> Result<ProveInfo> {
        self.prove_with_ctx(env, &VerifierContext::default(), elf)
    }

    /// Prove the specified ELF binary using the specified [VerifierContext].
    fn prove_with_ctx(
        &self,
        env: ExecutorEnv<'_>,
        ctx: &VerifierContext,
        elf: &[u8],
    ) -> Result<ProveInfo>;

    /// Prove the specified [Session].
    fn prove_session(&self, ctx: &VerifierContext, session: &Session) -> Result<ProveInfo>;

    /// Prove the specified [Segment].
    fn prove_segment(&self, ctx: &VerifierContext, segment: &Segment) -> Result<SegmentReceipt> {
        tracing::debug!("prove_segment");
        let results = self.segment_preflight(segment)?;
        let receipt = self.prove_segment_core(ctx, results)?;
        if std::env::var("RISC0_SKIP_VERIFY").is_err() {
            receipt
                .verify_integrity_with_context(ctx)
                .context("verify segment")?;
        }
        Ok(receipt)
    }

    /// Run preflight on the specified [Segment].
    fn segment_preflight(&self, segment: &Segment) -> Result<PreflightResults>;

    /// Prove the specified [Segment] which has had preflight run on it.
    fn prove_segment_core(
        &self,
        ctx: &VerifierContext,
        preflight_results: PreflightResults,
    ) -> Result<SegmentReceipt>;

    /// Prove the specified keccak request
    fn prove_keccak(&self, request: &crate::ProveKeccakRequest)
        -> Result<SuccinctReceipt<Unknown>>;

    /// Lift a [SegmentReceipt] into a [SuccinctReceipt]
    fn lift(&self, receipt: &SegmentReceipt) -> Result<SuccinctReceipt<ReceiptClaim>>;

    /// Join two [SuccinctReceipt] into a [SuccinctReceipt]
    fn join(
        &self,
        a: &SuccinctReceipt<ReceiptClaim>,
        b: &SuccinctReceipt<ReceiptClaim>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>>;

    /// Unite two [SuccinctReceipt] into a [SuccinctReceipt]
    fn union(
        &self,
        a: &SuccinctReceipt<Unknown>,
        b: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<UnionClaim>>;

    /// Resolve an assumption from a conditional [SuccinctReceipt] by providing a [SuccinctReceipt]
    /// proving the validity of the assumption.
    fn resolve(
        &self,
        conditional: &SuccinctReceipt<ReceiptClaim>,
        assumption: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>>;

    /// Lift a [SegmentReceipt] into a [SuccinctReceipt] with a proof of verifiable work (PoVW) claim.
    fn lift_povw(
        &self,
        receipt: &SegmentReceipt,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>>;

    /// Join two [SuccinctReceipt] with proof of verifiable work (PoVW) claims into a
    /// [SuccinctReceipt] a PoVW claim.
    fn join_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        b: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>>;

    /// Join two [SuccinctReceipt] with proof of verifiable work (PoVW) claims into a
    /// [SuccinctReceipt], and unwrap the result (see [ProverServer::unwrap_povw]).
    fn join_unwrap_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        b: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>>;

    /// Resolve an assumption from a conditional [SuccinctReceipt] with a proof of verifiable work
    /// (PoVW) claim by providing a [SuccinctReceipt] proving the validity of the assumption.
    fn resolve_povw(
        &self,
        conditional: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        assumption: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>>;

    /// Resolve an assumption from a conditional [SuccinctReceipt] with a proof of verifiable work
    /// (PoVW) claim by providing a [SuccinctReceipt] proving the validity of the assumption, and
    /// unwrap the result (see [ProverServer::unwrap_povw]).
    fn resolve_unwrap_povw(
        &self,
        conditional: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        assumption: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>>;

    /// Remove the proof of verifiable work (PoVW) information from a [SuccinctReceipt] to produce
    /// a [SuccinctReceipt] over the underlying claim.
    fn unwrap_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>>;

    /// Convert a [SuccinctReceipt] with a Poseidon hash function that uses a 254-bit field
    fn identity_p254(
        &self,
        a: &SuccinctReceipt<ReceiptClaim>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>>;

    /// Compress a [CompositeReceipt] into a single [SuccinctReceipt].
    ///
    /// A [CompositeReceipt] may contain an arbitrary number of receipts assembled into
    /// segments and assumptions. Together, these receipts collectively prove a top-level
    /// [ReceiptClaim](crate::ReceiptClaim). This function compresses all of the constituent receipts of a
    /// [CompositeReceipt] into a single [SuccinctReceipt] that proves the same top-level claim. It
    /// accomplishes this by iterative application of the recursion programs including lift, join,
    /// and resolve.
    fn composite_to_succinct(
        &self,
        receipt: &CompositeReceipt,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        <Self as Compress<_>>::composite_to_succinct(self, receipt)
    }

    /// Convert a composite receipt to a succinct work claim receipt.
    fn composite_to_succinct_povw(
        &self,
        receipt: &CompositeReceipt,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        <Self as Compress<_>>::composite_to_succinct(self, receipt)
    }

    /// Compress a [SuccinctReceipt] into a [Groth16Receipt].
    fn succinct_to_groth16(
        &self,
        receipt: &SuccinctReceipt<ReceiptClaim>,
    ) -> Result<Groth16Receipt<ReceiptClaim>> {
        let t0 = std::time::Instant::now();
        // Let the Groth16 prover load its proving key while identity_p254 runs.
        if let Err(e) = risc0_groth16::prove::prestart_prover() {
            tracing::warn!("groth16 prover prestart failed (non-fatal): {e}");
        }
        let ident_receipt = self.identity_p254(receipt).unwrap();
        let ident_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let seal_bytes = ident_receipt.get_seal_bytes();
        let seal_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // Hand the GPU to the Groth16 prover, then take it back. See `release_hal_buffers`. The
        // first release runs while the wrap computes its witness on another thread.
        #[cfg(any(feature = "rocm", feature = "intel"))]
        prover_impl::release_cached_provers();
        let t2 = std::time::Instant::now();
        let seal = shrink_wrap_with(&seal_bytes, release_hal_buffers)?.to_vec();
        release_hal_buffers();
        let wrap_ms = t2.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "[succinct_to_groth16] identity_p254={ident_ms:.1}ms get_seal={seal_ms:.1}ms \
             shrink_wrap={wrap_ms:.1}ms total={:.1}ms",
            t0.elapsed().as_secs_f64() * 1000.0,
        );

        let groth16_receipt = Groth16Receipt {
            seal,
            claim: receipt.claim.clone(),
            verifier_parameters: Groth16ReceiptVerifierParameters::default().digest(),
        };
        check_groth16_receipt(&groth16_receipt)?;
        Ok(groth16_receipt)
    }

    /// Compress a receipt into one with a smaller representation.
    ///
    /// The requested target representation is determined by the [ReceiptKind] specified on the
    /// provided [ProverOpts]. If the receipt is already at least as compressed as the requested
    /// kind, this is a no-op.
    fn compress(&self, opts: &ProverOpts, receipt: &Receipt) -> Result<Receipt> {
        match &receipt.inner {
            InnerReceipt::Composite(inner) => match opts.receipt_kind {
                ReceiptKind::Composite => Ok(receipt.clone()),
                ReceiptKind::Succinct => {
                    let succinct_receipt = self.composite_to_succinct(inner)?;
                    Ok(Receipt::new(
                        InnerReceipt::Succinct(succinct_receipt),
                        receipt.journal.bytes.clone(),
                    ))
                }
                ReceiptKind::Groth16 => {
                    let succinct_receipt = self.composite_to_succinct(inner)?;
                    let groth16_receipt = self.succinct_to_groth16(&succinct_receipt)?;
                    Ok(Receipt::new(
                        InnerReceipt::Groth16(groth16_receipt),
                        receipt.journal.bytes.clone(),
                    ))
                }
            },
            InnerReceipt::Succinct(inner) => match opts.receipt_kind {
                ReceiptKind::Composite | ReceiptKind::Succinct => Ok(receipt.clone()),
                ReceiptKind::Groth16 => {
                    let groth16_receipt = self.succinct_to_groth16(inner)?;
                    Ok(Receipt::new(
                        InnerReceipt::Groth16(groth16_receipt),
                        receipt.journal.bytes.clone(),
                    ))
                }
            },
            InnerReceipt::Groth16(_) => match opts.receipt_kind {
                ReceiptKind::Composite | ReceiptKind::Succinct | ReceiptKind::Groth16 => {
                    Ok(receipt.clone())
                }
            },
            InnerReceipt::Fake(_) => {
                ensure!(
                    opts.dev_mode(),
                    "dev mode must be enabled to compress fake receipts"
                );
                Ok(receipt.clone())
            }
        }
    }
}

trait Lift<Claim> {
    fn lift(&self, segment_receipt: &SegmentReceipt) -> anyhow::Result<SuccinctReceipt<Claim>>;
}

impl<P: ProverServer + ?Sized> Lift<ReceiptClaim> for P {
    fn lift(
        &self,
        segment_receipt: &SegmentReceipt,
    ) -> anyhow::Result<SuccinctReceipt<ReceiptClaim>> {
        <Self as ProverServer>::lift(self, segment_receipt)
    }
}

impl<P: ProverServer + ?Sized> Lift<WorkClaim<ReceiptClaim>> for P {
    fn lift(
        &self,
        segment_receipt: &SegmentReceipt,
    ) -> anyhow::Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        self.lift_povw(segment_receipt)
    }
}

trait Join<Claim> {
    fn join(
        &self,
        a: &SuccinctReceipt<Claim>,
        b: &SuccinctReceipt<Claim>,
    ) -> anyhow::Result<SuccinctReceipt<Claim>>;
}

impl<P: ProverServer + ?Sized> Join<ReceiptClaim> for P {
    fn join(
        &self,
        a: &SuccinctReceipt<ReceiptClaim>,
        b: &SuccinctReceipt<ReceiptClaim>,
    ) -> anyhow::Result<SuccinctReceipt<ReceiptClaim>> {
        <Self as ProverServer>::join(self, a, b)
    }
}

impl<P: ProverServer + ?Sized> Join<WorkClaim<ReceiptClaim>> for P {
    fn join(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        b: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> anyhow::Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        self.join_povw(a, b)
    }
}

trait Resolve<Claim> {
    fn resolve(
        &self,
        cond: &SuccinctReceipt<Claim>,
        assum: &SuccinctReceipt<Unknown>,
    ) -> anyhow::Result<SuccinctReceipt<Claim>>;
}

impl<P: ProverServer + ?Sized> Resolve<ReceiptClaim> for P {
    fn resolve(
        &self,
        cond: &SuccinctReceipt<ReceiptClaim>,
        assum: &SuccinctReceipt<Unknown>,
    ) -> anyhow::Result<SuccinctReceipt<ReceiptClaim>> {
        <Self as ProverServer>::resolve(self, cond, assum)
    }
}

impl<P: ProverServer + ?Sized> Resolve<WorkClaim<ReceiptClaim>> for P {
    fn resolve(
        &self,
        cond: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        assum: &SuccinctReceipt<Unknown>,
    ) -> anyhow::Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        self.resolve_povw(cond, assum)
    }
}

trait Compress<Claim> {
    fn composite_to_succinct(
        &self,
        composite_receipt: &CompositeReceipt,
    ) -> anyhow::Result<SuccinctReceipt<Claim>>;
}

impl<P, Claim> Compress<Claim> for P
where
    P: Lift<Claim>
        + Join<Claim>
        + Resolve<Claim>
        + Lift<ReceiptClaim>
        + Join<ReceiptClaim>
        + Resolve<ReceiptClaim>
        + ?Sized,
{
    fn composite_to_succinct(
        &self,
        composite_receipt: &CompositeReceipt,
    ) -> anyhow::Result<SuccinctReceipt<Claim>> {
        let t_pipeline = std::time::Instant::now();
        // Compress all receipts in the top-level session into one succinct receipt for the session.
        let continuation_receipt = composite_receipt
            .segments
            .iter()
            .try_fold(
                None,
                |left: Option<SuccinctReceipt<Claim>>, right: &SegmentReceipt| -> Result<_> {
                    let lifted = self.lift(right)?;
                    let result = match left {
                        Some(left) => self.join(&left, &lifted)?,
                        None => lifted,
                    };
                    Ok(Some(result))
                },
            )?
            .ok_or_else(|| {
                anyhow!("malformed composite receipt has no continuation segment receipts")
            })?;

        eprintln!(
            "[composite_to_succinct] lift/join done: {:.1}s",
            t_pipeline.elapsed().as_secs_f64()
        );

        // Compress assumptions and resolve them to get the final succinct receipt.
        let result = composite_receipt.assumption_receipts.iter().try_fold(
            continuation_receipt,
            |conditional: SuccinctReceipt<Claim>, assumption: &InnerAssumptionReceipt| match assumption {
                InnerAssumptionReceipt::Succinct(assumption) => self.resolve(&conditional, assumption),
                InnerAssumptionReceipt::Composite(assumption) => {
                    self.resolve(&conditional, &SuccinctReceipt::<ReceiptClaim>::into_unknown(self.composite_to_succinct(assumption)?))
                }
                InnerAssumptionReceipt::Fake(_) => bail!(
                    "compressing composite receipts with fake receipt assumptions is not supported"
                ),
                InnerAssumptionReceipt::Groth16(_) => bail!(
                    "compressing composite receipts with Groth16 receipt assumptions is not supported"
                )
            },
        )?;

        eprintln!(
            "[composite_to_succinct] total: {:.1}s",
            t_pipeline.elapsed().as_secs_f64()
        );
        Ok(result)
    }
}

/// A pair of [Hal] and [CircuitHal].
#[derive(Clone)]
pub struct HalPair<H, C>
where
    H: Hal<Field = BabyBear, Elem = Elem, ExtElem = ExtElem>,
    C: CircuitHal<H>,
{
    /// A [Hal] implementation.
    pub hal: Rc<H>,

    /// An [CircuitHal] implementation.
    pub circuit_hal: Rc<C>,
}

impl Session {
    /// For each segment, call [ProverServer::prove_session] and collect the
    /// receipts.
    pub fn prove(&self) -> Result<ProveInfo> {
        let prover = get_prover_server(&ProverOpts::default())?;
        prover.prove_session(&VerifierContext::default(), self)
    }
}

/// Select a [ProverServer] based on the specified [ProverOpts].
pub fn get_prover_server(opts: &ProverOpts) -> Result<Rc<dyn ProverServer>> {
    if opts.dev_mode() {
        eprintln!("WARNING: proving in dev mode. This will not generate valid, secure proofs.");
        return Ok(Rc::new(DevModeProver::new()));
    }

    Ok(Rc::new(ProverImpl::new(opts.clone())))
}

/// Exclusive use of the CUDA GPU against other threads, for one proving call: hold the result for
/// the call. The provers cache their HALs per thread, so the HALs can't hold it (see
/// `risc0_zkp::hal::cuda::singleton`); the calls that prove do. Reentrant on one thread.
#[cfg(feature = "cuda")]
pub(crate) fn exclusive_gpu() -> impl Sized {
    risc0_zkp::hal::cuda::exclusive()
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn exclusive_gpu() -> impl Sized {}

/// Return the STARK HAL's cached device memory to the driver, around the Groth16 wrap.
///
/// The wrap allocates through sppark rather than through the HAL, so it cannot reclaim what the HAL
/// is caching and fails instead; see `risc0_zkp::hal::cuda::clear_buffer_pool`. Called BEFORE the wrap
/// so it fits beside what the STARK left behind, and AFTER it so the next proof's STARK is not
/// competing with the wrap's freed scratch.
///
/// On ROCm the HAL has the same pool, released by its own `clear_buffer_pool`. On Windows a HIP
/// allocation beyond the card's memory does not fail but spills into shared system memory, so
/// without this a session's keccak proofs ran beside the segments' cached buffers on an RX 9070 XT
/// (16 GB), with up to 12.5 GB spilled and each keccak eval_check at ~7x its time.
///
/// On Intel the HAL's pool (capped at 16 GiB) is released the same way: neither the Groth16 prover
/// nor the keccak eval_check allocates through it. Without this, a session's keccak proofs and its
/// wrap, and the next session's segments, ran with the segments' idle buffers still cached and
/// spilled into shared system memory on an Arc Pro B70 (32 GB), at 2-5x the time.
pub(crate) fn release_hal_buffers() {
    #[cfg(feature = "cuda")]
    risc0_zkp::hal::cuda::clear_buffer_pool();
    #[cfg(feature = "rocm")]
    risc0_zkp::hal::hip::clear_buffer_pool();
    #[cfg(feature = "intel")]
    risc0_zkp::hal::intel::release_idle_buffers();
}
