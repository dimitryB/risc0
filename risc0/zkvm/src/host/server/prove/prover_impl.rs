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

use std::{cell::RefCell, collections::HashMap, sync::mpsc, sync::LazyLock};

use anyhow::{anyhow, bail, ensure, Context, Result};

static VERBOSE: LazyLock<bool> = LazyLock::new(|| std::env::var("RISC0_VERBOSE").is_ok());

use super::{
    keccak::{prove_keccak, prove_keccak_with},
    ProverServer,
};
use risc0_circuit_rv32im::prove::SegmentProver;

thread_local! {
    // ManuallyDrop prevents TLS cleanup from dropping CUDA buffers after the CUDA
    // context is already destroyed (which causes a panic/abort on process exit).
    // GPU memory is reclaimed by CUDA context teardown regardless.
    static CACHED_SEGMENT_PROVER: std::mem::ManuallyDrop<RefCell<Option<Box<dyn SegmentProver>>>> =
        std::mem::ManuallyDrop::new(RefCell::new(None));
}

/// Completes, and discards, whatever segment the cached prover still has pending when dropped.
struct DrainSegmentProverOnExit<'a>(&'a ProverImpl);

impl Drop for DrainSegmentProverOnExit<'_> {
    fn drop(&mut self) {
        // An error here means nothing was pending, or the GPU work failed; the slot is idle either
        // way. Not while panicking: the prover's RefCell may be the thing that is borrowed.
        if !std::thread::panicking() {
            let _ = self.0.with_segment_prover(|sp| sp.prove_end());
        }
    }
}

/// Drop this thread's cached segment prover and the device buffers it keeps between segments
/// (see `release_cached_provers`). Only between sessions: the prover must be idle.
#[cfg(any(feature = "rocm", feature = "intel"))]
fn release_cached_segment_prover() {
    CACHED_SEGMENT_PROVER.with(|cell| {
        if let Ok(mut borrow) = cell.try_borrow_mut() {
            *borrow = None;
        }
    });
}

/// Before the Groth16 wrap on ROCm and Intel, drop the provers cached between proofs (this
/// thread's segment prover and recursion provers), then the HAL's pool. On an RX 9070 XT (16 GB,
/// Windows) the wrap's ~7 GB proving key beside those long-lived buffers made the driver move
/// ~2 GB of them into shared system memory, where they stayed: every later session ran ~2.5x
/// slower. On an Arc B580 (12 GB, Windows) at po2=20 the next session's segments took 106 s
/// instead of 28 (54-65 s with this alone; the rest was the HAL's pool, see `pool_max_bytes` in
/// risc0-zkp's Intel HAL). Rebuilt by the next session, in device memory.
#[cfg(any(feature = "rocm", feature = "intel"))]
pub(crate) fn release_cached_provers() {
    release_cached_segment_prover();
    risc0_circuit_recursion::prove::release_cached_provers();
}

fn with_segment_prover<R>(f: impl FnOnce(&dyn SegmentProver) -> Result<R>) -> Result<R> {
    let _gpu = super::exclusive_gpu();
    CACHED_SEGMENT_PROVER.with(|cell| {
        {
            let mut borrow = cell.borrow_mut();
            if borrow.is_none() {
                *borrow = Some(risc0_circuit_rv32im::prove::segment_prover()?);
            }
        }
        let borrow = cell.borrow();
        f(borrow.as_ref().unwrap().as_ref())
    })
}
use crate::{
    claim::merge::Merge,
    host::{
        client::prove::opts::ReceiptKind,
        prove_info::ProveInfo,
        recursion::{identity_p254, join, join_with_opts, lift, lift_with_opts, resolve},
        server::{exec::executor::ExecutorImpl, prove::union_peak::UnionPeak},
    },
    mmr::MerkleMountainAccumulator,
    receipt::{
        Groth16Receipt, Groth16ReceiptVerifierParameters, InnerReceipt, SegmentReceipt,
        SuccinctReceipt,
    },
    recursion::prove::{
        join_povw, join_unwrap_povw, lift_povw, resolve_povw, resolve_unwrap_povw, union,
        unwrap_povw,
    },
    sha::Digestible,
    Assumption, AssumptionReceipt, CompositeReceipt, ExecutorEnv, InnerAssumptionReceipt,
    MaybePruned, Output, PreflightResults, ProverOpts, Receipt, ReceiptClaim, Segment, Session,
    SuccinctReceiptVerifierParameters, UnionClaim, Unknown, VerifierContext, WorkClaim,
};

/// CUDA context creation and module loading (`cuda_warmup`), started by [ProverImpl::new] so it
/// overlaps guest execution and taken by the first session that proves. Warmup only loads
/// modules early, so a session that finds it taken (or done) simply skips it.
#[cfg(feature = "cuda")]
enum CudaWarmup {
    NotStarted,
    Running(std::thread::JoinHandle<()>),
    Taken,
}

#[cfg(feature = "cuda")]
static CUDA_WARMUP: std::sync::Mutex<CudaWarmup> = std::sync::Mutex::new(CudaWarmup::NotStarted);

#[cfg(feature = "cuda")]
fn spawn_cuda_warmup() -> std::thread::JoinHandle<()> {
    std::thread::spawn(risc0_circuit_rv32im::prove::cuda_warmup)
}

/// The recursion kernels' module loading, started by a non-composite session.
#[cfg(feature = "cuda")]
static CUDA_REC_WARMUP: std::sync::Mutex<CudaWarmup> =
    std::sync::Mutex::new(CudaWarmup::NotStarted);

/// Wait for the recursion warmup, if one is running (it makes no GPU launches and takes no
/// GPU lock, so this cannot deadlock against a session).
#[cfg(feature = "cuda")]
fn join_cuda_rec_warmup() {
    let warmup = std::mem::replace(&mut *CUDA_REC_WARMUP.lock().unwrap(), CudaWarmup::Taken);
    if let CudaWarmup::Running(handle) = warmup {
        let _ = handle.join();
    }
}

/// An implementation of a Prover that runs locally.
pub struct ProverImpl {
    opts: ProverOpts,
}

impl ProverImpl {
    /// Construct a [ProverImpl].
    pub fn new(opts: ProverOpts) -> Self {
        // Intel: GPU init (queue, warmup, zk_shift table) overlaps guest
        // execution and segment-0 preflight instead of preceding segment 0.
        #[cfg(feature = "intel")]
        if std::env::var("RISC0_INTEL_EARLY_INIT").as_deref() != Ok("0") {
            risc0_zkp::hal::intel::start_background_init(vec![opts.max_segment_po2 as u32]);
        }
        #[cfg(feature = "cuda")]
        {
            let mut warmup = CUDA_WARMUP.lock().unwrap();
            if matches!(*warmup, CudaWarmup::NotStarted) {
                *warmup = CudaWarmup::Running(spawn_cuda_warmup());
            }
        }
        // ROCm: pick the per-arch HW queue count on this thread, before
        // the warmup threads make the first HIP call.
        #[cfg(feature = "rocm")]
        risc0_zkp::hal::hip::auto_configure_hw_queues();
        // Parse the Groth16 witness graph (and build its parallel schedule) while the guest
        // executes instead of beside the STARK.
        #[cfg(feature = "cuda")]
        if opts.receipt_kind == ReceiptKind::Groth16 {
            risc0_groth16::prove::preload_graph_background();
        }
        Self { opts }
    }

    fn with_segment_prover<R>(&self, f: impl FnOnce(&dyn SegmentProver) -> Result<R>) -> Result<R> {
        with_segment_prover(f)
    }

    /// Parallel STARK proving: both GPUs prove segments simultaneously, then recursion runs
    /// sequentially. This halves STARK time at the cost of sequential recursion.
    #[cfg(feature = "rocm")]
    fn prove_session_parallel_stark(
        &self,
        ctx: &VerifierContext,
        session: &Session,
    ) -> Result<ProveInfo> {
        let t_total = std::time::Instant::now();
        let max_po2 = self.opts.max_segment_po2;
        let num_segments = session.segments.len();

        // Preload circom graph in background.
        let _graph_preload = if self.opts.receipt_kind == ReceiptKind::Groth16 {
            Some(std::thread::spawn(|| {
                if let Err(e) = risc0_groth16::prove::preload_graph() {
                    eprintln!("[groth16] graph preload failed (non-fatal): {e}");
                }
            }))
        } else {
            None
        };

        // Warmup both GPUs sequentially (sppark init has shared static state).
        let warmup0 = std::thread::spawn(|| {
            risc0_zkp::hal::hip::switch_to_device(0);
            risc0_circuit_rv32im::prove::rocm_warmup();
        });
        let _ = warmup0.join();
        let warmup1 = std::thread::spawn(|| {
            risc0_zkp::hal::hip::switch_to_device(1);
            risc0_circuit_rv32im::prove::rocm_warmup();
        });
        let _ = warmup1.join();

        // Split segments between GPUs.
        // GPU1 (7900 XTX, gfx1100) is ~15% faster per segment than GPU0 (9070 XT, gfx1201)
        // due to higher memory BW and more CUs. Give GPU0 ~46% and GPU1 ~54%.
        let split = (num_segments * 46 + 50) / 100;
        eprintln!(
            "[parallel-stark] {} segments: GPU0={}, GPU1={}",
            num_segments,
            split,
            num_segments - split
        );

        let seg_verifier_params_digest = ctx
            .segment_verifier_parameters
            .as_ref()
            .ok_or_else(|| anyhow!("segment receipt verifier parameters missing from context"))?
            .digest();
        let hashfn = self.opts.hashfn.clone();

        // Resolve all segments upfront (needed for Send to threads).
        let all_segments: Vec<Segment> = session
            .segments
            .iter()
            .map(|s| s.resolve())
            .collect::<Result<_>>()?;

        // Thread function: prove a subset of segments on a specific GPU device.
        let prove_subset = |device: i32,
                            segments: Vec<Segment>,
                            vp_digest: crate::sha::Digest,
                            hashfn: String,
                            max_po2: usize|
         -> Result<Vec<SegmentReceipt>> {
            risc0_zkp::hal::hip::switch_to_device(device);

            // Each thread gets its own SegmentProver (thread-local).
            let sp = risc0_circuit_rv32im::prove::segment_prover()?;
            let mut receipts = Vec::with_capacity(segments.len());
            let mut prev_po2_stash: u32 = 0;

            for (local_idx, segment) in segments.iter().enumerate() {
                let t0 = std::time::Instant::now();
                ensure!(
                    segment.po2() <= max_po2,
                    "segment po2 exceeds max: {} > {}",
                    segment.po2(),
                    max_po2
                );
                let rand_z = risc0_circuit_rv32im::prove::preflight_rand_z();
                let preflight =
                    risc0_circuit_rv32im::prove::PreflightResults::new(&segment.inner, rand_z)?;
                let po2 = preflight.po2();

                // Use prove_begin/prove_end pipeline for eval_check overlap.
                let prev_seal = sp.prove_begin(preflight)?;

                // Process previous segment's seal (if any).
                if let Some(seal) = prev_seal {
                    let prev_seg = &segments[local_idx - 1];
                    let prev_po2 = prev_po2_stash;
                    let mut claim = ReceiptClaim::decode_from_seal_v2(&seal, Some(prev_po2))?;
                    claim.output = prev_seg.output.clone().into();
                    receipts.push(SegmentReceipt {
                        seal,
                        index: prev_seg.index,
                        hashfn: hashfn.clone(),
                        claim,
                        verifier_parameters: vp_digest,
                    });
                }
                prev_po2_stash = po2;

                if *VERBOSE {
                    eprintln!(
                        "[parallel-stark] GPU{device} seg {}: {:.1}ms",
                        segment.index,
                        t0.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }

            // Complete last segment's deferred finalize.
            let last_seal = sp.prove_end()?;
            let last_seg = segments.last().unwrap();
            let mut claim = ReceiptClaim::decode_from_seal_v2(&last_seal, Some(prev_po2_stash))?;
            claim.output = last_seg.output.clone().into();
            receipts.push(SegmentReceipt {
                seal: last_seal,
                index: last_seg.index,
                hashfn: hashfn.clone(),
                claim,
                verifier_parameters: vp_digest,
            });

            Ok(receipts)
        };

        // Split segments and spawn two STARK threads.
        let segs0 = all_segments[..split].to_vec();
        let segs1 = all_segments[split..].to_vec();
        let vp0 = seg_verifier_params_digest;
        let vp1 = seg_verifier_params_digest;
        let hf0 = hashfn.clone();
        let hf1 = hashfn.clone();

        let t_stark = std::time::Instant::now();
        let handle0 = std::thread::Builder::new()
            .name("stark-gpu0".into())
            .spawn(move || prove_subset(0, segs0, vp0, hf0, max_po2))?;
        let handle1 = std::thread::Builder::new()
            .name("stark-gpu1".into())
            .spawn(move || prove_subset(1, segs1, vp1, hf1, max_po2))?;

        let receipts0 = handle0
            .join()
            .map_err(|_| anyhow!("STARK GPU0 thread panicked"))??;
        let receipts1 = handle1
            .join()
            .map_err(|_| anyhow!("STARK GPU1 thread panicked"))??;

        let stark_ms = t_stark.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[parallel-stark] STARK done: {:.1}ms ({} + {} receipts)",
            stark_ms,
            receipts0.len(),
            receipts1.len()
        );

        // Merge receipts in segment index order.
        let mut segments: Vec<SegmentReceipt> = Vec::with_capacity(num_segments);
        segments.extend(receipts0);
        segments.extend(receipts1);
        segments.sort_by_key(|r| r.index);

        // Merge output + assumptions into the last segment.
        let (assumptions, _session_assumption_receipts): (Vec<_>, Vec<_>) =
            session.assumptions.iter().cloned().unzip();
        segments
            .last_mut()
            .ok_or_else(|| anyhow!("session is empty"))?
            .claim
            .output
            .merge_with(
                &session
                    .journal
                    .as_ref()
                    .map(|journal| Output {
                        journal: MaybePruned::Pruned(journal.digest()),
                        assumptions: assumptions.into(),
                    })
                    .into(),
            )?;

        let verifier_parameters = ctx
            .composite_verifier_parameters()
            .ok_or_else(|| anyhow!("composite receipt verifier parameters missing from context"))?
            .digest();

        let composite_receipt = CompositeReceipt {
            segments,
            assumption_receipts: Vec::new(),
            verifier_parameters,
        };

        // This path does not verify segment seals as it proves them, so verify the whole receipt
        // here, as prove_session does on the sequential path.
        if std::env::var("RISC0_SKIP_VERIFY").is_err() {
            composite_receipt.verify_integrity_with_context(ctx)?;
        }
        check_claims(
            &session.claim()?,
            "composite",
            MaybePruned::Value(composite_receipt.claim()?),
        )?;

        if self.opts.receipt_kind == ReceiptKind::Composite {
            let receipt = Receipt::new(
                InnerReceipt::Composite(composite_receipt),
                session.journal.clone().unwrap_or_default().bytes,
            );
            return Ok(ProveInfo {
                receipt,
                work_receipt: None,
                stats: session.stats(),
            });
        }

        // Recursion: multi-GPU lift+join pipeline.
        // Lift(N+1) on device 0 overlaps with join(acc, lift(N)) on device 1.
        // Saves ~175ms per step (42 steps = ~7.3s).
        let t_rec = std::time::Instant::now();
        let lift_device = 0;
        let join_device = 1;

        let segments_ref = &composite_receipt.segments;
        let num_segs = segments_ref.len();
        let recursion_opts = if self.opts.max_segment_po2 > crate::receipt::DEFAULT_MAX_PO2 {
            ProverOpts::from_max_po2(self.opts.max_segment_po2)
                .with_receipt_kind(ReceiptKind::Succinct)
        } else {
            ProverOpts::succinct()
        };
        let _skip_verify = std::env::var("RISC0_SKIP_VERIFY").is_ok();

        // Multi-GPU recursion pipeline:
        // - Persistent lift thread on device 0 (preserves thread-local PROVER_CACHE
        //   across all lifts → cached ctrl PolyGroup reuse).
        // - Main thread does joins on device 1.
        // - Pipeline: lift(N+1) overlaps with join(N).
        //
        // IMPORTANT: The main thread must NEVER call lift_with_opts, because the
        // RecursionProverImpl caches its HipHal (which holds a per-device
        // ReentrantMutex lock) in thread-local storage. If the main thread cached
        // a device 0 prover, the persistent lift thread would deadlock trying to
        // acquire device 0's lock.

        // Channel-based persistent lift thread: receives segments, returns lifted receipts.
        let (lift_tx, lift_rx) = std::sync::mpsc::sync_channel::<(SegmentReceipt, ProverOpts)>(1);
        let (result_tx, result_rx) =
            std::sync::mpsc::sync_channel::<Result<SuccinctReceipt<ReceiptClaim>>>(1);

        let lift_thread =
            std::thread::Builder::new()
                .name("lift-gpu0".into())
                .spawn(move || {
                    risc0_zkp::hal::hip::switch_to_device(lift_device);
                    // Process lift requests until channel closes.
                    while let Ok((seg, opts)) = lift_rx.recv() {
                        let result = lift_with_opts(&seg, opts);
                        if result_tx.send(result).is_err() {
                            break; // main thread dropped receiver
                        }
                    }
                })?;

        // Send lift(0).
        lift_tx.send((segments_ref[0].clone(), recursion_opts.clone()))?;
        let mut accumulator = result_rx
            .recv()
            .map_err(|_| anyhow!("lift thread died"))??;
        if *VERBOSE {
            eprintln!("[parallel-recursion] step 0/{num_segs}: lift on device {lift_device}");
        }

        // Send lift(1) ahead (will run while main thread is idle or doing join).
        if num_segs > 1 {
            lift_tx.send((segments_ref[1].clone(), recursion_opts.clone()))?;
        }

        // Pipelined loop: join(N) on device 1 overlaps with lift(N+1) on device 0.
        for step_idx in 1..num_segs {
            let t_step = std::time::Instant::now();

            // Wait for lift(step_idx) to complete.
            let lifted = result_rx
                .recv()
                .map_err(|_| anyhow!("lift thread died"))??;
            let lift_ms = t_step.elapsed().as_secs_f64() * 1000.0;

            // Send lift(step_idx+1) BEFORE running join.
            // This way lift(N+1) runs on device 0 while join(N) runs on device 1.
            if step_idx + 1 < num_segs {
                lift_tx.send((segments_ref[step_idx + 1].clone(), recursion_opts.clone()))?;
            }

            // Run join on device 1 (overlaps with lift(N+1) on device 0).
            let t_join = std::time::Instant::now();
            risc0_zkp::hal::hip::switch_to_device(join_device);
            let joined = join_with_opts(&accumulator, &lifted, recursion_opts.clone())?;
            let join_ms = t_join.elapsed().as_secs_f64() * 1000.0;

            if *VERBOSE {
                eprintln!(
                    "[parallel-recursion] step {step_idx}/{num_segs}: lift_wait={lift_ms:.1}ms join={join_ms:.1}ms total={:.1}ms",
                    t_step.elapsed().as_secs_f64() * 1000.0
                );
            }
            accumulator = joined;
        }

        // Shut down lift thread.
        drop(lift_tx);
        let _ = lift_thread.join();

        let succinct_receipt = accumulator;
        let rec_ms = t_rec.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[parallel-stark] recursion done: {:.1}ms (multi-GPU pipeline)",
            rec_ms
        );

        if self.opts.receipt_kind == ReceiptKind::Succinct {
            let receipt = Receipt::new(
                InnerReceipt::Succinct(succinct_receipt),
                session.journal.clone().unwrap_or_default().bytes,
            );
            return Ok(ProveInfo {
                receipt,
                work_receipt: None,
                stats: session.stats(),
            });
        }

        // Groth16 wrapping.
        // Use device 1 (7900 XTX) for Groth16 — full-rate INT32 makes MSM ~2.5x faster.
        risc0_zkp::hal::hip::switch_to_device(1);
        let groth16_receipt = self.succinct_to_groth16(&succinct_receipt)?;
        let receipt = Receipt::new(
            InnerReceipt::Groth16(groth16_receipt),
            session.journal.clone().unwrap_or_default().bytes,
        );

        eprintln!(
            "[parallel-stark] total: {:.1}ms",
            t_total.elapsed().as_secs_f64() * 1000.0
        );

        Ok(ProveInfo {
            receipt,
            work_receipt: None,
            stats: session.stats(),
        })
    }

    fn prove_session_impl(&self, ctx: &VerifierContext, session: &Session) -> Result<ProveInfo> {
        // The whole session, segments to Groth16: its calls nest on this thread.
        let _gpu = super::exclusive_gpu();
        tracing::debug!(
            "prove_session: exit_code = {:?}, journal = {:?}, segments: {}",
            session.exit_code,
            session.journal.as_ref().map(hex::encode),
            session.segments.len()
        );

        ensure!(
            self.opts.hashfn == "poseidon2",
            "provided `ProverOpts` has unsupported `hashfn` value of \"{}\"; \
            supported `hashfn` values are: \"poseidon2\".",
            &self.opts.hashfn
        );

        // The segment prover is cached per thread; drop any seal a failed
        // earlier session left in its pipeline.
        self.with_segment_prover(|sp| {
            sp.reset_pipeline();
            Ok(())
        })?;

        // Check for parallel STARK mode: both GPUs prove segments simultaneously.
        #[cfg(feature = "rocm")]
        if std::env::var("RISC0_PARALLEL_STARK").is_ok()
            && risc0_zkp::hal::hip::device_count() >= 2
            && self.opts.receipt_kind != ReceiptKind::Composite
            && !session.povw_job_id.is_some()
            && session.assumptions.is_empty()
            && session.pending_keccaks.is_empty()
            // Each GPU must get at least one segment: an empty half has nothing to finalize.
            && session.segments.len() >= 2
        {
            return self.prove_session_parallel_stark(ctx, session);
        }

        // CUDA module loading, joined before the first segment: the thread ProverImpl::new
        // started, or a new one if none was (a one-time cost per process).
        #[cfg(feature = "cuda")]
        let mut warmup_handle = {
            let mut warmup = CUDA_WARMUP.lock().unwrap();
            match std::mem::replace(&mut *warmup, CudaWarmup::Taken) {
                CudaWarmup::Running(handle) => Some(handle),
                CudaWarmup::NotStarted => Some(spawn_cuda_warmup()),
                CudaWarmup::Taken => None,
            }
        };
        // Recursion kernels load on their own thread, once per process, off the composite's
        // path; the first lift joins it.
        #[cfg(feature = "cuda")]
        if self.opts.receipt_kind != ReceiptKind::Composite {
            let mut warmup = CUDA_REC_WARMUP.lock().unwrap();
            if matches!(*warmup, CudaWarmup::NotStarted) {
                *warmup = CudaWarmup::Running(std::thread::spawn(
                    risc0_circuit_recursion::prove::cuda_warmup,
                ));
            }
        }
        #[cfg(feature = "rocm")]
        let mut warmup_handle = Some(std::thread::spawn(|| {
            risc0_circuit_rv32im::prove::rocm_warmup();
        }));
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        let mut warmup_handle: Option<std::thread::JoinHandle<()>> = None;

        // Load the Groth16 proving key while the STARK proves (the loader starts once segment
        // 0's prove_begin returns, see `prestart_key_go`).
        #[cfg(feature = "cuda")]
        if self.opts.receipt_kind == ReceiptKind::Groth16 {
            risc0_groth16::prove::prestart_key();
        }
        // Preload circom graph on background thread so the ~728ms file read + parse
        // overlaps with segment proving and recursion.  Only needed for Groth16.
        #[cfg(any(feature = "cuda", feature = "rocm", feature = "intel"))]
        let _graph_preload = if self.opts.receipt_kind == ReceiptKind::Groth16 {
            Some(std::thread::spawn(|| {
                if let Err(e) = risc0_groth16::prove::preload_graph() {
                    eprintln!("[groth16] graph preload failed (non-fatal): {e}");
                }
            }))
        } else {
            None
        };

        let skip_verify = std::env::var("RISC0_SKIP_VERIFY").is_ok();
        if skip_verify {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    "RISC0_SKIP_VERIFY is set: the prover will not verify the segments, composite \
                     or succinct receipts it produces; only their claims are checked"
                )
            });
        }
        let max_po2 = self.opts.max_segment_po2;

        // Pipeline: overlap preflight(N+1) CPU work with prove_core(N) GPU work.
        // Preflight is ~275ms of pure CPU emulation; prove_core is ~443ms of GPU work.
        // By overlapping them, we hide preflight behind GPU time (saves ~37% per segment).
        let mut segments = Vec::new();
        let mut pending_preflight: Option<
            std::thread::JoinHandle<Result<(Segment, PreflightResults)>>,
        > = None;

        // Background receipt pipeline: decode(N) + verify(N) overlaps with prove(N+1).
        // After prove_core returns the raw seal, we immediately start prove_core(N+1)
        // while a background thread decodes the seal, builds the receipt, and verifies.
        // This saves ~5ms of decode time from the critical path per segment.
        let verify_params = ctx.segment_verifier_parameters.clone();
        // Precompute the verifier_parameters digest once (avoids per-segment SHA256).
        let seg_verifier_params_digest = ctx
            .segment_verifier_parameters
            .as_ref()
            .ok_or_else(|| anyhow!("segment receipt verifier parameters missing from context"))?
            .digest();
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        type LiftPrepOpt = Option<rec_pipe::Prep>;
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        type LiftPrepOpt = Option<()>;
        let mut pending_receipt: Option<
            std::thread::JoinHandle<Result<(SegmentReceipt, LiftPrepOpt)>>,
        > = None;
        // Lift/join during segment proving (see rec_pipe). On ROCm only when RISC0_CUDA_REC_PIPE=1
        // asks for it, and not with a recursion device (the multi-GPU recursion worker below lifts
        // and joins instead). There a segment proof finalizes inline (prove_begin returns with
        // the GPU done), so lifting between segments overlaps no GPU work: it saved ~0.1 s of
        // 20 s on fib 1M on an RX 7900 XTX and nothing on general-compute, and on a 16 GB RX 9070
        // XT with the Groth16 prover resident (a succinct proof after a Groth16 wrap) it made
        // every segment ~40 ms slower (presumably the buffer pool, whose segment and recursion
        // buffers no longer fit side by side; see release_pool_unless_fits).
        #[cfg(feature = "rocm")]
        let rec_pipe_ok = risc0_zkp::hal::hip::recursion_device().is_none()
            && std::env::var("RISC0_CUDA_REC_PIPE").as_deref() == Ok("1");
        #[cfg(all(feature = "cuda", not(feature = "rocm")))]
        let rec_pipe_ok = true;
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        let mut rec_pipe: Option<rec_pipe::RecPipe> = {
            if rec_pipe::enabled()
                && rec_pipe_ok
                && self.opts.receipt_kind != ReceiptKind::Composite
                && session.povw_job_id.is_none()
                && session.assumptions.is_empty()
                && session.pending_keccaks.is_empty()
            {
                let recursion_opts = if self.opts.max_segment_po2 > crate::receipt::DEFAULT_MAX_PO2
                {
                    ProverOpts::from_max_po2(self.opts.max_segment_po2)
                        .with_receipt_kind(ReceiptKind::Succinct)
                } else {
                    ProverOpts::succinct()
                };
                Some(rec_pipe::RecPipe::new(recursion_opts))
            } else {
                None
            }
        };
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        let lift_opts: Option<ProverOpts> = rec_pipe.as_ref().map(|p| p.opts());
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        let early_seal = rec_pipe.is_some();
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        let early_seal = false;
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        let lift_opts: Option<ProverOpts> = None;
        // Metadata from the previous iteration, paired with the seal returned by prove_begin.
        let mut prev_seal_meta: Option<(u32, u32, Option<Output>, String)> = None;

        // Multi-GPU: overlap recursion on secondary device with STARK proving.
        // Send SegmentReceipts to a background thread that runs lift+join on device 1.
        #[cfg(feature = "rocm")]
        let multi_gpu_active = risc0_zkp::hal::hip::recursion_device().is_some()
            && self.opts.receipt_kind != ReceiptKind::Composite
            && !session.povw_job_id.is_some()
            && session.assumptions.is_empty();
        #[cfg(not(feature = "rocm"))]
        let multi_gpu_active = false;

        let (mut recursion_tx, recursion_worker) = if multi_gpu_active {
            let (tx, rx) = mpsc::channel::<SegmentReceipt>();
            let max_po2 = self.opts.max_segment_po2;
            let want_groth16 = self.opts.receipt_kind == ReceiptKind::Groth16;
            let handle = std::thread::Builder::new()
                .name("recursion-worker".into())
                .spawn(move || -> Result<(SuccinctReceipt<ReceiptClaim>, Option<SuccinctReceipt<ReceiptClaim>>)> {
                    #[cfg(feature = "rocm")]
                    {
                        let device = risc0_zkp::hal::hip::recursion_device().unwrap();
                        risc0_zkp::hal::hip::switch_to_device(device);
                    }

                    let recursion_opts =
                        if max_po2 > crate::receipt::DEFAULT_MAX_PO2 {
                            ProverOpts::from_max_po2(max_po2)
                                .with_receipt_kind(ReceiptKind::Succinct)
                        } else {
                            ProverOpts::succinct()
                        };

                    let mut accumulator: Option<SuccinctReceipt<ReceiptClaim>> = None;

                    for (idx, segment_receipt) in rx.iter().enumerate() {
                        let t0 = std::time::Instant::now();
                        let lifted =
                            lift_with_opts(&segment_receipt, recursion_opts.clone())?;
                        let lift_ms = t0.elapsed().as_secs_f64() * 1000.0;

                        accumulator = match accumulator.take() {
                            Some(left) => {
                                let t1 = std::time::Instant::now();
                                let joined = join_with_opts(
                                    &left,
                                    &lifted,
                                    recursion_opts.clone(),
                                )?;
                                let join_ms = t1.elapsed().as_secs_f64() * 1000.0;
                                if *VERBOSE {
                                    eprintln!(
                                        "[multi-gpu recursion] step {idx}: \
                                         lift={lift_ms:.1}ms join={join_ms:.1}ms"
                                    );
                                }
                                Some(joined)
                            }
                            None => {
                                if *VERBOSE {
                                    eprintln!(
                                        "[multi-gpu recursion] step {idx}: \
                                         lift={lift_ms:.1}ms (first)"
                                    );
                                }
                                Some(lifted)
                            }
                        };
                    }

                    let succinct = accumulator.ok_or_else(|| {
                        anyhow!("no segments received by recursion worker")
                    })?;

                    // Run identity_p254 here to overlap with remaining STARK segments.
                    let ident = if want_groth16 {
                        let t0 = std::time::Instant::now();
                        let r = identity_p254(&succinct)?;
                        if *VERBOSE {
                            eprintln!(
                                "[multi-gpu recursion] identity_p254={:.1}ms",
                                t0.elapsed().as_secs_f64() * 1000.0
                            );
                        }
                        Some(r)
                    } else {
                        None
                    };

                    Ok((succinct, ident))
                })?;
            (Some(tx), Some(handle))
        } else {
            (None, None)
        };
        #[cfg(feature = "rocm")]
        if multi_gpu_active {
            eprintln!(
                "[multi-gpu] recursion worker started on device {}",
                risc0_zkp::hal::hip::recursion_device().unwrap_or(0)
            );
        }

        // A session that leaves the pipeline early (an invalid segment, an error) leaves its last
        // segment's finalize pending in the CACHED prover, and the next session's first
        // prove_begin would hand that stale seal back as its own previous segment's. Complete it
        // on the way out, so the next session starts with the prover idle.
        let pipeline_guard = DrainSegmentProverOnExit(self);

        for (seg_idx, segment_ref) in session.segments.iter().enumerate() {
            let t_seg = std::time::Instant::now();

            // Get segment + preflight results: from pipelined thread or compute here.
            let (segment, results) = if let Some(handle) = pending_preflight.take() {
                handle
                    .join()
                    .map_err(|_| anyhow!("preflight thread panicked"))??
            } else {
                // First segment: compute synchronously.
                let segment = segment_ref.resolve()?;
                #[cfg(any(feature = "cuda", feature = "rocm"))]
                let results = if rec_pipe.is_some() {
                    // Commit the recursion ctrl groups on the idle GPU while the CPU runs the
                    // first segment's preflight (for the recursion pipe, which lifts during the
                    // session).
                    let po2 = segment.po2();
                    let g16 = self.opts.receipt_kind == ReceiptKind::Groth16;
                    let ropts = if self.opts.max_segment_po2 > crate::receipt::DEFAULT_MAX_PO2 {
                        ProverOpts::from_max_po2(self.opts.max_segment_po2)
                            .with_receipt_kind(ReceiptKind::Succinct)
                    } else {
                        ProverOpts::succinct()
                    };
                    let seg_ref = &segment;
                    let max_po2 = self.opts.max_segment_po2;
                    std::thread::scope(|sc| -> Result<PreflightResults> {
                        // Not `segment_preflight`: on another thread it would create a segment
                        // prover, whose HAL init waits for the GPU lock this session holds.
                        let h = sc.spawn(move || -> Result<PreflightResults> {
                            ensure!(
                                seg_ref.po2() <= max_po2,
                                "segment po2 exceeds max on ProverOpts: {} > {}",
                                seg_ref.po2(),
                                max_po2
                            );
                            let rand_z = risc0_circuit_rv32im::prove::preflight_rand_z();
                            let inner = risc0_circuit_rv32im::prove::PreflightResults::new(
                                &seg_ref.inner,
                                rand_z,
                            )?;
                            Ok(PreflightResults {
                                inner,
                                terminate_state: seg_ref.inner.claim.terminate_state,
                                output: seg_ref.output.clone(),
                                segment_index: seg_ref.index,
                            })
                        });
                        let t = std::time::Instant::now();
                        if let Some(handle) = warmup_handle.take() {
                            let _ = handle.join();
                        }
                        #[cfg(feature = "cuda")]
                        join_cuda_rec_warmup();
                        let tw = t.elapsed().as_secs_f64() * 1e3;
                        let r = crate::host::recursion::prove::Prover::prewarm(&[po2], &ropts, g16);
                        tracing::debug!(
                            "recursion prewarm: warmups={tw:.1}ms prewarm done at {:.1}ms ok={}",
                            t.elapsed().as_secs_f64() * 1e3,
                            r.is_ok()
                        );
                        let res = h.join().map_err(|_| anyhow!("preflight thread panicked"))?;
                        tracing::debug!(
                            "recursion prewarm: segment 0 preflight joined at {:.1}ms",
                            t.elapsed().as_secs_f64() * 1e3
                        );
                        res
                    })?
                } else {
                    self.segment_preflight(&segment)?
                };
                #[cfg(not(any(feature = "cuda", feature = "rocm")))]
                let results = self.segment_preflight(&segment)?;
                (segment, results)
            };
            let t_preflight = t_seg.elapsed();

            for hook in &session.hooks {
                hook.on_pre_prove_segment(&segment);
            }

            // Join warmup thread before first segment's prove_core.
            if seg_idx == 0 {
                if let Some(handle) = warmup_handle.take() {
                    let _ = handle.join();
                }
            }

            // Start preflight for NEXT segment in background while GPU proves current.
            if seg_idx + 1 < session.segments.len() {
                let next_segment = session.segments[seg_idx + 1].resolve()?;
                pending_preflight = Some(std::thread::spawn(move || {
                    ensure!(
                        next_segment.po2() <= max_po2,
                        "segment po2 exceeds max: {} > {}",
                        next_segment.po2(),
                        max_po2
                    );
                    let rand_z = risc0_circuit_rv32im::prove::preflight_rand_z();
                    let inner = risc0_circuit_rv32im::prove::PreflightResults::new(
                        &next_segment.inner,
                        rand_z,
                    )?;
                    let pr = PreflightResults {
                        inner,
                        terminate_state: next_segment.inner.claim.terminate_state,
                        output: next_segment.output.clone(),
                        segment_index: next_segment.index,
                    };
                    Ok((next_segment, pr))
                }));
            }

            // Extract metadata before prove_begin consumes the inner preflight.
            let po2 = results.inner.po2();
            let segment_index = results.segment_index;
            let output = results.output;
            let hashfn = self.opts.hashfn.clone();

            release_pool_on_po2_growth(po2);

            // GPU prove_begin: main phase (overlapping with prev eval_check) +
            // complete prev deferred finalize + launch eval_check for this segment.
            // Returns the PREVIOUS segment's seal (None on first call).
            let t_psc = std::time::Instant::now();
            #[cfg(feature = "metal-rv32im")]
            let prev_seal = risc0_zkp::hal::metal::with_profile_segment(segment_index, || {
                self.with_segment_prover(|sp| sp.prove_begin(results.inner))
            })?;
            #[cfg(not(feature = "metal-rv32im"))]
            let prev_seal = self.with_segment_prover(|sp| sp.prove_begin(results.inner))?;
            let prove_begin_ms = t_psc.elapsed().as_secs_f64() * 1000.0;
            // Segment 0's GPU work (incl. its eval_check) is queued: release the Groth16 key
            // loader, so its pinned uploads and cold allocations do not compete with
            // segment 0's.
            #[cfg(feature = "cuda")]
            if self.opts.receipt_kind == ReceiptKind::Groth16 {
                risc0_groth16::prove::prestart_key_go();
            }
            // Early seal (with the recursion pipe): prove_begin finalizes this segment inline and
            // only holds its seal back; take it now (prove_end just returns it), so this
            // segment's receipt and lift preparation start one iteration earlier.
            let mut cur_meta = Some((po2, segment_index, output, hashfn));
            let prev_seal = if early_seal {
                // prove_begin returns no seal in this mode: it finalized the previous segment
                // when that one began.
                ensure!(
                    prev_seal.is_none(),
                    "segment pipeline: unexpected seal from prove_begin with the early seal"
                );
                prev_seal_meta = cur_meta.take();
                Some(self.with_segment_prover(|sp| sp.prove_end())?)
            } else {
                prev_seal
            };

            // If we got a previous segment's seal, process it.
            if let Some(seal) = prev_seal {
                // Collect the background receipt from the segment before that.
                if let Some(handle) = pending_receipt.take() {
                    let (receipt, _prep) = handle
                        .join()
                        .map_err(|_| anyhow!("receipt thread panicked"))??;
                    // Multi-GPU: send receipt to recursion worker for parallel lift+join
                    if let Some(ref tx) = recursion_tx {
                        let _ = tx.send(receipt.clone());
                    }
                    #[cfg(any(feature = "cuda", feature = "rocm"))]
                    if let (Some(p), Some(prep)) = (rec_pipe.as_mut(), _prep) {
                        p.add(segments.len(), prep, Some(receipt.claim.clone()));
                    }
                    segments.push(receipt);
                    #[cfg(any(feature = "cuda", feature = "rocm"))]
                    if let Some(p) = rec_pipe.as_mut() {
                        p.poll()?;
                    }
                }

                // Start background receipt for the PREVIOUS segment.
                let (prev_po2, prev_seg_idx, prev_output, prev_hashfn) = prev_seal_meta
                    .take()
                    .ok_or_else(|| anyhow!("segment pipeline: seal without metadata"))?;
                let params = verify_params.clone();
                let vp_digest = seg_verifier_params_digest;
                let skip_verify_copy = skip_verify;
                let lift_opts_copy = lift_opts.clone();
                pending_receipt = Some(std::thread::spawn(move || {
                    let mut claim = ReceiptClaim::decode_from_seal_v2(&seal, Some(prev_po2))?;
                    claim.output = prev_output.into();
                    let receipt = SegmentReceipt {
                        seal,
                        index: prev_seg_idx,
                        hashfn: prev_hashfn,
                        claim,
                        verifier_parameters: vp_digest,
                    };
                    #[cfg(any(feature = "cuda", feature = "rocm"))]
                    let prep =
                        lift_opts_copy.map(|o| rec_pipe::spawn_lift_prep(receipt.clone(), o));
                    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
                    let prep = {
                        let _ = lift_opts_copy;
                        None
                    };
                    if !skip_verify_copy {
                        receipt
                            .verify_integrity_with_context(&VerifierContext {
                                segment_verifier_parameters: params,
                                ..VerifierContext::empty()
                            })
                            .with_context(|| format!("verify segment {prev_seg_idx}"))?;
                    }
                    Ok((receipt, prep))
                }));
            }

            // Store current segment's metadata for next iteration.
            if let Some(m) = cur_meta {
                prev_seal_meta = Some(m);
            }

            let t_prove = t_seg.elapsed() - t_preflight;
            let t_total = t_seg.elapsed();
            if *VERBOSE {
                eprintln!(
                "[prove_session] seg {seg_idx}: preflight={:.1}ms prove_begin={:.1}ms prove={:.1}ms total={:.1}ms",
                t_preflight.as_secs_f64() * 1000.0,
                prove_begin_ms,
                t_prove.as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
            );
            }

            for hook in &session.hooks {
                hook.on_post_prove_segment(&segment);
            }
        }

        // The loop is done. The last segment's deferred finalize was completed by the early seals,
        // or is completed just below, which empties the prover whether or not it succeeds: there is
        // nothing left for the guard to drain.
        std::mem::forget(pipeline_guard);

        if early_seal {
            // Every seal was taken in the loop: collect the last receipt (verified on its
            // thread); its lift's claim is the merged one, set when the pipe drains.
            if let Some(handle) = pending_receipt.take() {
                let (receipt, _prep) = handle
                    .join()
                    .map_err(|_| anyhow!("receipt thread panicked"))??;
                #[cfg(any(feature = "cuda", feature = "rocm"))]
                if let (Some(p), Some(prep)) = (rec_pipe.as_mut(), _prep) {
                    p.add(segments.len(), prep, None);
                    p.poll()?;
                }
                segments.push(receipt);
            }
        } else {
            // Complete the last segment's deferred finalize.
            let last_seal = self.with_segment_prover(|sp| sp.prove_end())?;
            let mut last_seal = Some(last_seal);

            // Parallel tail verify (without the recursion pipe, whose early seal verifies every
            // receipt on its own thread; RISC0_TAIL_VERIFY=0 disables): build the last segment's
            // receipt and verify it on its own thread while the second-to-last one (started after
            // the previous prove_begin) finishes verifying, so the two CPU verifications overlap
            // instead of running back to back.
            let mut par_last: Option<std::thread::JoinHandle<Result<SegmentReceipt>>> = None;
            if !skip_verify && std::env::var("RISC0_TAIL_VERIFY").as_deref() != Ok("0") {
                let (last_po2, last_seg_idx, last_output, last_hashfn) = prev_seal_meta
                    .take()
                    .ok_or_else(|| anyhow!("segment pipeline: seal without metadata"))?;
                let seal = last_seal.take().unwrap();
                let mut claim = ReceiptClaim::decode_from_seal_v2(&seal, Some(last_po2))?;
                claim.output = last_output.into();
                let receipt = SegmentReceipt {
                    seal,
                    index: last_seg_idx,
                    hashfn: last_hashfn,
                    claim,
                    verifier_parameters: seg_verifier_params_digest,
                };
                let params = verify_params.clone();
                par_last = Some(std::thread::spawn(move || -> Result<SegmentReceipt> {
                    receipt
                        .verify_integrity_with_context(&VerifierContext {
                            segment_verifier_parameters: params,
                            ..VerifierContext::empty()
                        })
                        .with_context(|| format!("verify segment (last) {}", receipt.index))?;
                    Ok(receipt)
                }));
            }

            // Collect the second-to-last receipt (if any).
            if let Some(handle) = pending_receipt.take() {
                let (receipt, _prep) = handle
                    .join()
                    .map_err(|_| anyhow!("receipt thread panicked"))??;
                // Multi-GPU: send to recursion worker
                if let Some(ref tx) = recursion_tx {
                    let _ = tx.send(receipt.clone());
                }
                segments.push(receipt);
            }

            // Process the last segment's seal.
            if let Some(h) = par_last.take() {
                segments.push(h.join().map_err(|_| anyhow!("receipt thread panicked"))??);
            } else {
                let last_seal = last_seal.take().unwrap();
                let (last_po2, last_seg_idx, last_output, last_hashfn) = prev_seal_meta
                    .take()
                    .ok_or_else(|| anyhow!("segment pipeline: seal without metadata"))?;
                let mut claim = ReceiptClaim::decode_from_seal_v2(&last_seal, Some(last_po2))?;
                claim.output = last_output.into();
                let receipt = SegmentReceipt {
                    seal: last_seal,
                    index: last_seg_idx,
                    hashfn: last_hashfn,
                    claim,
                    verifier_parameters: seg_verifier_params_digest,
                };
                if !skip_verify {
                    receipt
                        .verify_integrity_with_context(&VerifierContext {
                            segment_verifier_parameters: verify_params.clone(),
                            ..VerifierContext::empty()
                        })
                        .with_context(|| format!("verify segment (last) {}", receipt.index))?;
                }
                segments.push(receipt);
            }
        } // !early_seal

        let (assumptions, session_assumption_receipts): (Vec<_>, Vec<_>) =
            session.assumptions.iter().cloned().unzip();

        // Merge the output, including journal digest and assumptions, into the last segment.
        segments
            .last_mut()
            .ok_or_else(|| anyhow!("session is empty"))?
            .claim
            .output
            .merge_with(
                &session
                    .journal
                    .as_ref()
                    .map(|journal| Output {
                        journal: MaybePruned::Pruned(journal.digest()),
                        assumptions: assumptions.into(),
                    })
                    .into(),
            )
            .context("failed to merge output into final segment claim")?;

        // Multi-GPU: send last receipt (with merged output) to recursion worker.
        if let Some(tx) = recursion_tx.take() {
            let _ = tx.send(segments.last().unwrap().clone());
            // Channel closed on drop → worker finishes after processing this receipt.
        }

        let verifier_parameters = ctx
            .composite_verifier_parameters()
            .ok_or_else(|| anyhow!("composite receipt verifier parameters missing from context"))?
            .digest();

        let mut zkr_receipts = HashMap::new();
        let mut keccak_receipts: MerkleMountainAccumulator<UnionPeak> =
            MerkleMountainAccumulator::new();

        let keccak_ctx = VerifierContext::default()
            .with_succinct_verifier_parameters(SuccinctReceiptVerifierParameters::for_keccak());

        // The keccak prover's witness generation and sppark's NTTs allocate outside the STARK HAL,
        // so they cannot reclaim what the segments left in its buffer pool: release it first, as
        // for the Groth16 wrap. Without this, the first keccak proof after the segments ran out of
        // memory on an RTX 5080 (16 GB) and on an RTX 4090 limited to 12 GB.
        if !session.pending_keccaks.is_empty() {
            super::release_hal_buffers();
        }

        let t_keccaks = std::time::Instant::now();
        std::thread::scope(|scope| -> Result<()> {
            // On Intel and AMD GPUs a keccak proof's witness is made on the CPU, about half of
            // the proof's time: make each on a thread, a proof ahead of the GPU, so the two
            // overlap. (On CUDA the GPU makes it.) Returning early drops `witnesses`, which ends
            // the thread.
            let witnesses = (cfg!(any(feature = "intel", feature = "rocm"))
                && session.pending_keccaks.len() > 1)
                .then(|| {
                    let (tx, rx) = mpsc::sync_channel(1);
                    let requests = &session.pending_keccaks;
                    scope.spawn(move || {
                        for request in requests {
                            let witness = risc0_circuit_keccak::prove::host_witness(
                                &request.input,
                                request.po2,
                            );
                            if tx.send(witness).is_err() {
                                break;
                            }
                        }
                    });
                    rx
                });
            for (i, proof_request) in session.pending_keccaks.iter().enumerate() {
                let t_keccak = std::time::Instant::now();
                let witness = match &witnesses {
                    Some(rx) => Some(
                        rx.recv()
                            .map_err(|_| anyhow!("the keccak witness thread ended early"))?
                            .context("keccak witness")?,
                    ),
                    None => None,
                };
                let wait_ms = t_keccak.elapsed().as_secs_f64() * 1000.0;
                let receipt = prove_keccak_with(proof_request, witness).context("prove keccak")?;
                let prove_ms = t_keccak.elapsed().as_secs_f64() * 1000.0;
                receipt.verify_integrity_with_context(&keccak_ctx)?;
                tracing::debug!("adding keccak assumption: {}", receipt.claim.digest());
                let t_insert = std::time::Instant::now();
                keccak_receipts.insert(receipt)?;
                if *VERBOSE {
                    eprintln!(
                        "[keccak] {i}: po2={} witness wait={wait_ms:.1}ms \
                         prove+lift={prove_ms:.1}ms insert={:.1}ms total={:.1}ms",
                        proof_request.po2,
                        t_insert.elapsed().as_secs_f64() * 1000.0,
                        t_keccak.elapsed().as_secs_f64() * 1000.0,
                    );
                }
            }
            Ok(())
        })?;

        // NOTE: Calling keccak_receipts.root() proves the union tree.
        let t_root = std::time::Instant::now();
        let keccak_root = keccak_receipts.root();
        if *VERBOSE && !session.pending_keccaks.is_empty() {
            eprintln!(
                "[keccak] {} proofs: root={:.1}ms total={:.1}ms",
                session.pending_keccaks.len(),
                t_root.elapsed().as_secs_f64() * 1000.0,
                t_keccaks.elapsed().as_secs_f64() * 1000.0,
            );
        }
        // On ROCm, start the segments' lifts and joins from fresh device memory: the keccak
        // proofs peak near a 16 GB card's capacity, and on Windows buffers the HAL pooled
        // through that peak could be in shared system memory. With keccak at po2 18 on an RX
        // 9070 XT the lifts and joins after it ran ~1.6x slower (recursion data commits 28 ->
        // 93 ms). The keccak lift programs' cached ctrl groups aren't needed after this either.
        #[cfg(feature = "rocm")]
        if !session.pending_keccaks.is_empty() {
            release_cached_provers();
            super::release_hal_buffers();
        }
        if let Ok(root_receipt) = keccak_root {
            let assumption = Assumption {
                claim: root_receipt.claim.digest(),
                control_root: root_receipt.control_root()?,
            };

            tracing::debug!("keccak root assumption: {:?}", assumption);
            zkr_receipts.insert(assumption, root_receipt.clone());
        }

        // TODO: add test case for when a single session refers to the same assumption multiple times
        let inner_assumption_receipts: Vec<_> = session_assumption_receipts
            .into_iter()
            .map(|assumption_receipt| match assumption_receipt {
                AssumptionReceipt::Proven(receipt) => Ok(receipt),
                AssumptionReceipt::Unresolved(assumption) => {
                    let receipt = zkr_receipts.get(&assumption).ok_or_else(|| {
                        anyhow!("no receipt available for unresolved assumption: {assumption:#?}")
                    })?;
                    Ok(InnerAssumptionReceipt::Succinct(receipt.clone()))
                }
            })
            .collect::<Result<_>>()?;

        let composite_receipt = CompositeReceipt {
            segments,
            assumption_receipts: inner_assumption_receipts,
            verifier_parameters,
        };

        let session_claim = session.claim()?;

        // Verify the receipt to catch if something is broken in the proving process. Each
        // segment's seal was verified as it was proved, so it is not verified again here; the
        // rest of the composite is: the segments' chaining, exit codes and outputs, and the
        // assumption receipts. RISC0_SKIP_VERIFY skips that, but never the claim check, which
        // costs nothing.
        if !skip_verify {
            composite_receipt.verify_integrity_inner(ctx, false)?;
        }
        check_claims(
            &session_claim,
            "composite",
            MaybePruned::Value(composite_receipt.claim()?),
        )?;

        if self.opts.receipt_kind == ReceiptKind::Composite {
            let receipt = Receipt::new(
                InnerReceipt::Composite(composite_receipt),
                session.journal.clone().unwrap_or_default().bytes,
            );
            return Ok(ProveInfo {
                receipt,
                work_receipt: None,
                stats: session.stats(),
            });
        }

        let (succinct_receipt, work_receipt, pre_ident) = if let Some(handle) = recursion_worker {
            // Multi-GPU: collect the SuccinctReceipt from the background recursion worker.
            // Assumptions already checked to be empty at multi_gpu_active decision.
            let t_wait = std::time::Instant::now();
            let (succinct, ident) = handle
                .join()
                .map_err(|_| anyhow!("recursion worker thread panicked"))??;
            let wait_ms = t_wait.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[multi-gpu] recursion worker finished (wait={wait_ms:.1}ms)");
            (succinct, None, ident)
        } else {
            match session.povw_job_id.is_some() {
                true => {
                    let work_receipt = self.composite_to_succinct_povw(&composite_receipt)?;
                    let unwrapped = self.unwrap_povw(&work_receipt)?;
                    (unwrapped, Some(work_receipt), None)
                }
                #[cfg(any(feature = "cuda", feature = "rocm"))]
                false if rec_pipe.is_some() => {
                    let n = composite_receipt.segments.len();
                    let last = composite_receipt.segments.last().unwrap().claim.clone();
                    let t = std::time::Instant::now();
                    let p = rec_pipe.take().unwrap();
                    let r = p.drain(last, n)?;
                    tracing::debug!(
                        "composite_to_succinct: recursion pipe tail {:.1}ms",
                        t.elapsed().as_secs_f64() * 1e3
                    );
                    (r, None, None)
                }
                false => (self.composite_to_succinct(&composite_receipt)?, None, None),
            }
        };

        // composite_to_succinct calls lift/join without the per-receipt
        // verify_integrity of the default Compress impl; each join verifies its
        // inputs in-circuit, so verifying the final receipt covers them. Above the
        // default po2 the recursion allows the wider set of control IDs, and so
        // must the verifier.
        let verify_succinct = |succinct_receipt: &SuccinctReceipt<ReceiptClaim>| -> Result<()> {
            let max_po2 = self.opts.max_segment_po2;
            let wide_ctx;
            let succinct_ctx = if max_po2 > crate::receipt::DEFAULT_MAX_PO2 {
                wide_ctx = VerifierContext::default()
                    .with_dev_mode(ctx.dev_mode())
                    .with_suites(ctx.suites.clone())
                    .with_succinct_verifier_parameters(
                        SuccinctReceiptVerifierParameters::from_max_po2(max_po2),
                    );
                &wide_ctx
            } else {
                ctx
            };
            succinct_receipt
                .verify_integrity_with_context(succinct_ctx)
                .context("verify succinct")
        };
        // For Groth16, verify the succinct receipt on a scoped thread alongside
        // succinct_to_groth16 (CPU work beside identity_p254 and the wrap); joined, and its error
        // returned, before the Groth16 receipt is. RISC0_CUDA_G16_TAIL=0 (the Groth16 tail
        // switch, see risc0-groth16) verifies first.
        let par_verify = !skip_verify
            && self.opts.receipt_kind == ReceiptKind::Groth16
            && pre_ident.is_none()
            && std::env::var("RISC0_CUDA_G16_TAIL").as_deref() != Ok("0");
        if !skip_verify && !par_verify {
            verify_succinct(&succinct_receipt)?;
        }

        if self.opts.receipt_kind == ReceiptKind::Succinct {
            let receipt = Receipt::new(
                InnerReceipt::Succinct(succinct_receipt),
                session.journal.clone().unwrap_or_default().bytes,
            );
            return Ok(ProveInfo {
                receipt,
                work_receipt: work_receipt.map(Into::into),
                stats: session.stats(),
            });
        }

        let groth16_receipt = if let Some(ident_receipt) = pre_ident {
            // identity_p254 already computed by recursion worker (overlapped with STARK).
            use risc0_groth16::prove::shrink_wrap_with;
            if let Err(e) = risc0_groth16::prove::prestart_prover() {
                eprintln!("[groth16] prover prestart failed (non-fatal): {e}");
            }
            let t0 = std::time::Instant::now();
            let seal_bytes = ident_receipt.get_seal_bytes();
            // See `release_hal_buffers`: the wrap cannot reclaim what the STARK HAL is caching.
            // Released while the wrap computes its witness on another thread.
            #[cfg(any(feature = "rocm", feature = "intel"))]
            release_cached_provers();
            let seal = shrink_wrap_with(&seal_bytes, super::release_hal_buffers)?.to_vec();
            super::release_hal_buffers();
            let wrap_ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[succinct_to_groth16] identity_p254=pre-computed shrink_wrap={wrap_ms:.1}ms \
                 total={wrap_ms:.1}ms"
            );
            let groth16_receipt = Groth16Receipt {
                seal,
                claim: succinct_receipt.claim.clone(),
                verifier_parameters: Groth16ReceiptVerifierParameters::default().digest(),
            };
            super::check_groth16_receipt(&groth16_receipt)?;
            groth16_receipt
        } else if par_verify {
            let succinct_receipt = &succinct_receipt;
            let verify_succinct = &verify_succinct;
            std::thread::scope(|sc| -> Result<_> {
                let verify = sc.spawn(move || verify_succinct(succinct_receipt));
                let groth16 = self.succinct_to_groth16(succinct_receipt);
                verify
                    .join()
                    .map_err(|_| anyhow!("succinct verify thread panicked"))??;
                groth16
            })?
        } else {
            // succinct_to_groth16 verifies the receipt before returning it.
            self.succinct_to_groth16(&succinct_receipt)?
        };

        if self.opts.receipt_kind == ReceiptKind::Groth16 {
            let receipt = Receipt::new(
                InnerReceipt::Groth16(groth16_receipt),
                session.journal.clone().unwrap_or_default().bytes,
            );
            return Ok(ProveInfo {
                receipt,
                work_receipt: work_receipt.map(Into::into),
                stats: session.stats(),
            });
        }

        // As long as the checks above are exhaustive, this code is unreachable. If this statement
        // is reached, this is an implementation error.
        unreachable!(
            "proving not implemented for receipt kind {:?}",
            self.opts.receipt_kind
        );
    }
}

impl ProverServer for ProverImpl {
    fn prove(&self, env: ExecutorEnv<'_>, elf: &[u8]) -> Result<ProveInfo> {
        let ctx = VerifierContext::default().with_dev_mode(self.opts.dev_mode());
        self.prove_with_ctx(env, &ctx, elf)
    }

    fn prove_with_ctx(
        &self,
        env: ExecutorEnv<'_>,
        ctx: &VerifierContext,
        elf: &[u8],
    ) -> Result<ProveInfo> {
        #[allow(unused_mut)]
        let mut env = env;
        // Intel and CUDA: succinct/groth16 receipts are faster with po2=21 segments (half the
        // lifts and joins; composite proving costs more per cycle at po2=21, but less than the
        // recursion it saves). Only when the caller did not choose a segment size;
        // RISC0_SUCCINCT_SEGMENT_PO2 (or RISC0_INTEL_SUCCINCT_SEGMENT_PO2) overrides, e.g. 20
        // keeps the executor default. CUDA needs ~28 GB of device memory for po2=21. Intel keeps
        // the executor default: on the 32 GB Arc Pro B70, po2=21 ran out of device memory in long
        // sessions and after a Groth16 wrap in the same process (an Ethereum block, a STARK
        // proof after the Groth16 prover cached its key), where po2=20 did not.
        #[cfg(any(feature = "intel", feature = "cuda"))]
        if self.opts.receipt_kind != ReceiptKind::Composite && env.segment_limit_po2.is_none() {
            let po2 = std::env::var("RISC0_SUCCINCT_SEGMENT_PO2")
                .or_else(|_| std::env::var("RISC0_INTEL_SUCCINCT_SEGMENT_PO2"))
                .ok()
                .and_then(|v| v.parse::<u32>().ok());
            #[cfg(feature = "cuda")]
            let po2 = po2.or_else(|| {
                let total = risc0_zkp::hal::cuda::device_total_memory().unwrap_or(0);
                (total >= 30 << 30).then_some(21)
            });
            if let Some(po2) = po2 {
                env.segment_limit_po2 = Some(po2.min(self.opts.max_segment_po2 as u32));
            }
        }
        let session = ExecutorImpl::from_elf(env, elf)?.run()?;
        self.prove_session(ctx, &session)
    }

    fn prove_session(&self, ctx: &VerifierContext, session: &Session) -> Result<ProveInfo> {
        let info = self.prove_session_impl(ctx, session)?;
        // Return this session's pooled GPU buffers instead of holding them idle
        // until the next session (cross-session reuse saves no measurable
        // time, and stale sizes, e.g. from another po2, would crowd a later
        // session). Only on success: freeing after DEVICE_LOST never returns.
        #[cfg(feature = "intel")]
        {
            risc0_zkp::hal::intel::release_idle_buffers();
            risc0_circuit_rv32im::prove::release_intel_scratch();
        }
        Ok(info)
    }

    fn segment_preflight(&self, segment: &Segment) -> Result<PreflightResults> {
        tracing::debug!("segment_preflight");

        ensure!(
            segment.po2() <= self.opts.max_segment_po2,
            "segment po2 exceeds max on ProverOpts: {} > {}",
            segment.po2(),
            self.opts.max_segment_po2
        );
        let inner = self.with_segment_prover(|sp| sp.preflight(&segment.inner))?;

        Ok(PreflightResults {
            inner,
            terminate_state: segment.inner.claim.terminate_state,
            output: segment.output.clone(),
            segment_index: segment.index,
        })
    }

    fn prove_segment_core(
        &self,
        ctx: &VerifierContext,
        preflight_results: PreflightResults,
    ) -> Result<SegmentReceipt> {
        tracing::debug!("prove_segment_core");

        ensure!(
            self.opts.hashfn == "poseidon2",
            "provided `ProverOpts` has unsupported `hashfn` value of \"{}\"; \
            supported `hashfn` values are: \"poseidon2\".",
            &self.opts.hashfn
        );

        let po2 = preflight_results.inner.po2();
        release_pool_on_po2_growth(po2);
        let t_psc = std::time::Instant::now();
        #[cfg(feature = "metal-rv32im")]
        let seal =
            risc0_zkp::hal::metal::with_profile_segment(preflight_results.segment_index, || {
                self.with_segment_prover(|sp| sp.prove_core(preflight_results.inner))
            })?;
        #[cfg(not(feature = "metal-rv32im"))]
        let seal = self.with_segment_prover(|sp| sp.prove_core(preflight_results.inner))?;
        if *VERBOSE {
            eprintln!(
                "[prove_segment_core] prove_core: {:.1}ms",
                t_psc.elapsed().as_secs_f64() * 1000.0
            );
        }
        let t_dec = std::time::Instant::now();
        let mut claim = ReceiptClaim::decode_from_seal_v2(&seal, Some(po2))?;
        if *VERBOSE {
            eprintln!(
                "[prove_segment_core] decode: {:.1}ms (seal len: {})",
                t_dec.elapsed().as_secs_f64() * 1000.0,
                seal.len()
            );
        }
        claim.output = preflight_results.output.into();

        let verifier_parameters = ctx
            .segment_verifier_parameters
            .as_ref()
            .ok_or_else(|| anyhow!("segment receipt verifier parameters missing from context"))?
            .digest();
        let receipt = SegmentReceipt {
            seal,
            index: preflight_results.segment_index,
            hashfn: self.opts.hashfn.clone(),
            claim,
            verifier_parameters,
        };

        Ok(receipt)
    }

    fn lift(&self, receipt: &SegmentReceipt) -> Result<SuccinctReceipt<ReceiptClaim>> {
        let t0 = std::time::Instant::now();
        let receipt = if self.opts.max_segment_po2 > crate::receipt::DEFAULT_MAX_PO2 {
            let recursion_opts = ProverOpts::from_max_po2(self.opts.max_segment_po2)
                .with_receipt_kind(ReceiptKind::Succinct);
            lift_with_opts(receipt, recursion_opts)?
        } else {
            lift(receipt)?
        };
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if std::env::var("RISC0_SKIP_VERIFY").is_err() {
            let t1 = std::time::Instant::now();
            receipt.verify_integrity().context("verify lift")?;
            if *VERBOSE {
                eprintln!(
                    "[lift] prove={prove_ms:.1}ms verify={:.1}ms",
                    t1.elapsed().as_secs_f64() * 1000.0
                );
            }
        } else {
            if *VERBOSE {
                eprintln!("[lift] prove={prove_ms:.1}ms verify=skipped");
            }
        }
        Ok(receipt)
    }

    fn lift_povw(
        &self,
        receipt: &SegmentReceipt,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        lift_povw(receipt)
    }

    fn join(
        &self,
        a: &SuccinctReceipt<ReceiptClaim>,
        b: &SuccinctReceipt<ReceiptClaim>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        let t0 = std::time::Instant::now();
        let receipt = if self.opts.max_segment_po2 > crate::receipt::DEFAULT_MAX_PO2 {
            let recursion_opts = ProverOpts::from_max_po2(self.opts.max_segment_po2)
                .with_receipt_kind(ReceiptKind::Succinct);
            join_with_opts(a, b, recursion_opts)?
        } else {
            join(a, b)?
        };
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if std::env::var("RISC0_SKIP_VERIFY").is_err() {
            let t1 = std::time::Instant::now();
            receipt.verify_integrity().context("verify join")?;
            if *VERBOSE {
                eprintln!(
                    "[join] prove={prove_ms:.1}ms verify={:.1}ms",
                    t1.elapsed().as_secs_f64() * 1000.0
                );
            }
        } else {
            if *VERBOSE {
                eprintln!("[join] prove={prove_ms:.1}ms verify=skipped");
            }
        }
        Ok(receipt)
    }

    fn join_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        b: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        join_povw(a, b)
    }

    fn join_unwrap_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        b: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        join_unwrap_povw(a, b)
    }

    fn resolve(
        &self,
        conditional: &SuccinctReceipt<ReceiptClaim>,
        assumption: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        let t0 = std::time::Instant::now();
        let receipt = resolve(conditional, assumption)?;
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if std::env::var("RISC0_SKIP_VERIFY").is_err() {
            let t1 = std::time::Instant::now();
            receipt.verify_integrity().context("verify resolve")?;
            if *VERBOSE {
                eprintln!(
                    "[resolve] prove={prove_ms:.1}ms verify={:.1}ms",
                    t1.elapsed().as_secs_f64() * 1000.0
                );
            }
        } else {
            if *VERBOSE {
                eprintln!("[resolve] prove={prove_ms:.1}ms verify=skipped");
            }
        }
        Ok(receipt)
    }

    fn resolve_povw(
        &self,
        conditional: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        assumption: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        resolve_povw(conditional, assumption)
    }

    fn resolve_unwrap_povw(
        &self,
        conditional: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        assumption: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        resolve_unwrap_povw(conditional, assumption)
    }

    fn identity_p254(
        &self,
        a: &SuccinctReceipt<ReceiptClaim>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        // TODO: figure out how to verify this
        identity_p254(a)
    }

    fn prove_keccak(
        &self,
        request: &crate::ProveKeccakRequest,
    ) -> Result<SuccinctReceipt<Unknown>> {
        // TODO: figure out how to verify this
        prove_keccak(request)
    }

    fn union(
        &self,
        a: &SuccinctReceipt<Unknown>,
        b: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<UnionClaim>> {
        let t0 = std::time::Instant::now();
        let receipt = union(a, b)?;
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if std::env::var("RISC0_SKIP_VERIFY").is_err() {
            let t1 = std::time::Instant::now();
            receipt.verify_integrity().context("verify union")?;
            if *VERBOSE {
                eprintln!(
                    "[union] prove={prove_ms:.1}ms verify={:.1}ms",
                    t1.elapsed().as_secs_f64() * 1000.0
                );
            }
        } else {
            if *VERBOSE {
                eprintln!("[union] prove={prove_ms:.1}ms verify=skipped");
            }
        }
        Ok(receipt)
    }

    fn unwrap_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        unwrap_povw(a)
    }

    /// Pipelined composite_to_succinct: left-fold lift+join with background CPU
    /// preflight overlapping GPU work.
    ///
    /// On CUDA, the tree-based approach with rayon batch preflighting is faster.
    /// On HIP/ROCm, the linear pipelined approach is ~2.5x faster because HIP's
    /// per-proof overhead makes CPU-GPU overlap more valuable than batch parallelism.
    fn composite_to_succinct(
        &self,
        composite_receipt: &CompositeReceipt,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        use crate::host::recursion::prove::{
            join_with_opts, lift_with_opts, make_succinct_receipt, Prover as RecursionProverJob,
        };

        let t_pipeline = std::time::Instant::now();
        let segments = &composite_receipt.segments;
        let num_segments = segments.len();
        eprintln!(
            "[composite_to_succinct] starting: {} segments, {} assumptions (pipelined)",
            num_segments,
            composite_receipt.assumption_receipts.len()
        );

        let mut accumulator: Option<SuccinctReceipt<ReceiptClaim>> = None;

        let recursion_opts = if self.opts.max_segment_po2 > crate::receipt::DEFAULT_MAX_PO2 {
            ProverOpts::from_max_po2(self.opts.max_segment_po2)
                .with_receipt_kind(ReceiptKind::Succinct)
        } else {
            ProverOpts::succinct()
        };

        #[cfg(feature = "cuda")]
        join_cuda_rec_warmup();

        // Cross-segment pipelining: bg thread does CPU preflight of next lift
        // while main thread runs the GPU join of the previous one. Validated
        // ~3% E2E win at composite=42 segs.
        //
        // Note: GPU overlap (running the full lift on a second queue) was
        // attempted via RISC0_LIFT_JOIN_OVERLAP env gate; produced invalid
        // proofs on B70 (single-CCS hardware, shared recursion HAL state).
        // See memory: project_lift_join_overlap_invalid.md
        let mut pending_prepared_lift: Option<std::thread::JoinHandle<Result<RecursionProverJob>>> =
            None;

        // Run lift(k+1) on the GPU while a background thread prepares
        // join(acc, lift(k)) (preflight + host witgen, see
        // RISC0_RECURSION_PREWITGEN); then run the prepared join while the
        // background thread prepares lift(k+2). The fold order (and therefore
        // the receipt) is unchanged; only the GPU/CPU overlap differs.
        // Default on; RISC0_RECURSION_REORDER=0 uses the sequential loop below.
        let reorder = !matches!(std::env::var("RISC0_RECURSION_REORDER").as_deref(), Ok("0"));
        if reorder && num_segments >= 2 {
            let spawn_lift_prep =
                |idx: usize| -> Result<std::thread::JoinHandle<Result<RecursionProverJob>>> {
                    let p = RecursionProverJob::new_lift(&segments[idx], recursion_opts.clone())?;
                    Ok(std::thread::spawn(move || -> Result<RecursionProverJob> {
                        let mut p = p;
                        p.prepare()?;
                        Ok(p)
                    }))
                };
            let finish_lift = |h: std::thread::JoinHandle<Result<RecursionProverJob>>,
                               idx: usize|
             -> Result<SuccinctReceipt<ReceiptClaim>> {
                let mut prover = h
                    .join()
                    .map_err(|_| anyhow!("lift prepare thread panicked"))??;
                let receipt = prover.run()?;
                let claim_decoded = ReceiptClaim::decode(&mut receipt.out_stream())?;
                let claim = claim_decoded.merge(&segments[idx].claim)?;
                make_succinct_receipt(prover, receipt, claim)
            };
            let t0 = std::time::Instant::now();
            // Prepare lift0, lift1 (and lift2) concurrently up front, instead of lift0's
            // preflight on the GPU thread and lift1's only after lift0 (~40 ms of idle GPU).
            let h0 = spawn_lift_prep(0)?;
            let h1 = spawn_lift_prep(1)?;
            let mut pending: Option<std::thread::JoinHandle<Result<RecursionProverJob>>> =
                if num_segments > 2 {
                    Some(spawn_lift_prep(2)?)
                } else {
                    None
                };
            // ROCm: the programs' first use, off the GPU's critical path. A background thread
            // loads the join program and the other segment po2s' lift programs and builds their
            // host ctrl (else ~15 ms of program load on this thread before the first join and
            // before the last lift, and ~15 ms of host ctrl in each one's first witgen), and this
            // thread uploads and commits lift0's program's ctrl while the first preparations run
            // (the GPU idles until lift0's is done). RISC0_REC_WARM=0: off.
            #[cfg(feature = "rocm")]
            if std::env::var("RISC0_REC_WARM").as_deref() != Ok("0") {
                let mut po2s: Vec<usize> = Vec::new();
                for seg in segments.iter() {
                    if let Ok(p) = RecursionProverJob::lift_po2(seg) {
                        if !po2s.contains(&p) {
                            po2s.push(p);
                        }
                    }
                }
                let rest: Vec<usize> = po2s.iter().skip(1).copied().collect();
                let opts = recursion_opts.clone();
                let _ = std::thread::Builder::new()
                    .name("rec-preload".into())
                    .spawn(move || {
                        if let Err(e) = RecursionProverJob::preload(&rest, &opts) {
                            tracing::debug!("recursion preload: {e}");
                        }
                    });
                if let Some(&p) = po2s.first() {
                    if let Err(e) = RecursionProverJob::prewarm_lift(p, &recursion_opts) {
                        tracing::debug!("recursion prewarm: {e}");
                    }
                }
            }
            let mut acc = finish_lift(h0, 0)?;
            let mut cur = finish_lift(h1, 1)?;
            if *VERBOSE {
                eprintln!(
                    "[composite_to_succinct] reorder: lift0+lift1={:.1}ms",
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            for k in 1..num_segments {
                let t_step = std::time::Instant::now();
                // background: prepare join(acc, cur)
                let jp = RecursionProverJob::new_join(&acc, &cur, recursion_opts.clone())?;
                let join_claim = acc.claim.join(&cur.claim)?.value()?;
                let jh = std::thread::spawn(move || -> Result<RecursionProverJob> {
                    let mut p = jp;
                    p.prepare()?;
                    Ok(p)
                });
                // GPU: next lift (prepared during the previous join)
                let next = match pending.take() {
                    Some(h) => Some(finish_lift(h, k + 1)?),
                    None => None,
                };
                let lift_ms = t_step.elapsed().as_secs_f64() * 1000.0;
                let mut jprover = jh
                    .join()
                    .map_err(|_| anyhow!("join prepare thread panicked"))??;
                if k + 2 < num_segments {
                    pending = Some(spawn_lift_prep(k + 2)?);
                }
                let t_join = std::time::Instant::now();
                let receipt = jprover.run()?;
                let claim_decoded = ReceiptClaim::decode(&mut receipt.out_stream())?;
                let claim = claim_decoded.merge(&join_claim)?;
                acc = make_succinct_receipt(jprover, receipt, claim)?;
                let join_ms = t_join.elapsed().as_secs_f64() * 1000.0;
                if *VERBOSE {
                    eprintln!("[composite_to_succinct] reorder step {k}/{num_segments}: next_lift={lift_ms:.1}ms join={join_ms:.1}ms total={:.1}ms",
                        t_step.elapsed().as_secs_f64() * 1000.0);
                }
                match next {
                    Some(n) => cur = n,
                    None => break,
                }
            }
            accumulator = Some(acc);
        }

        for step_idx in 0..(if reorder && num_segments >= 2 {
            0
        } else {
            num_segments
        }) {
            let t_step = std::time::Instant::now();

            // Get the lifted receipt.
            let lifted: SuccinctReceipt<ReceiptClaim> = match pending_prepared_lift.take() {
                Some(handle) => {
                    let mut prover = handle
                        .join()
                        .map_err(|_| anyhow!("lift prepare thread panicked"))??;
                    let receipt = prover.run()?;
                    let claim_decoded = ReceiptClaim::decode(&mut receipt.out_stream())?;
                    let claim = claim_decoded.merge(&segments[step_idx].claim)?;
                    make_succinct_receipt(prover, receipt, claim)?
                }
                None => {
                    // First segment: compute full lift (no pipelining available yet).
                    lift_with_opts(&segments[step_idx], recursion_opts.clone())?
                }
            };
            let lift_ms = t_step.elapsed().as_secs_f64() * 1000.0;

            // Start CPU preflight for next lift in background.
            if step_idx + 1 < num_segments {
                let opts_next = recursion_opts.clone();
                let next_prover = RecursionProverJob::new_lift(&segments[step_idx + 1], opts_next)?;
                pending_prepared_lift =
                    Some(std::thread::spawn(move || -> Result<RecursionProverJob> {
                        let mut p = next_prover;
                        p.prepare()?;
                        Ok(p)
                    }));
            }

            // Run join on main thread (GPU) while background thread prepares next lift.
            let result = match accumulator.take() {
                Some(left) => {
                    let t_join = std::time::Instant::now();
                    let joined = join_with_opts(&left, &lifted, recursion_opts.clone())?;
                    let join_ms = t_join.elapsed().as_secs_f64() * 1000.0;
                    if *VERBOSE {
                        eprintln!(
                            "[composite_to_succinct] step {step_idx}/{num_segments}: \
                             lift={lift_ms:.1}ms join={join_ms:.1}ms total={:.1}ms",
                            t_step.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    joined
                }
                None => {
                    if *VERBOSE {
                        eprintln!(
                            "[composite_to_succinct] step {step_idx}/{num_segments}: \
                             lift={lift_ms:.1}ms (first)"
                        );
                    }
                    lifted
                }
            };
            accumulator = Some(result);
        }

        eprintln!(
            "[composite_to_succinct] lift/join done: {:.3}s",
            t_pipeline.elapsed().as_secs_f64()
        );

        let continuation_receipt = accumulator.ok_or_else(|| {
            anyhow!("malformed composite receipt has no continuation segment receipts")
        })?;

        // Compress assumptions and resolve them (same as generic implementation).
        let result = composite_receipt.assumption_receipts.iter().try_fold(
            continuation_receipt,
            |conditional, assumption| match assumption {
                InnerAssumptionReceipt::Succinct(assumption) => {
                    self.resolve(&conditional, assumption)
                }
                InnerAssumptionReceipt::Composite(assumption) => {
                    self.resolve(
                        &conditional,
                        &SuccinctReceipt::<ReceiptClaim>::into_unknown(
                            <Self as super::Compress<_>>::composite_to_succinct(self, assumption)?,
                        ),
                    )
                }
                InnerAssumptionReceipt::Fake(_) => bail!(
                    "compressing composite receipts with fake receipt assumptions is not supported"
                ),
                InnerAssumptionReceipt::Groth16(_) => bail!(
                    "compressing composite receipts with Groth16 receipt assumptions is not supported"
                ),
            },
        )?;

        eprintln!(
            "[composite_to_succinct] total: {:.1}s",
            t_pipeline.elapsed().as_secs_f64()
        );
        Ok(result)
    }
}

fn check_claims(
    session_claim: &ReceiptClaim,
    other_name: &str,
    other_claim: MaybePruned<ReceiptClaim>,
) -> Result<()> {
    let session_claim_digest = session_claim.digest();
    let other_claim_digest = other_claim.digest();
    if session_claim_digest != other_claim_digest {
        tracing::debug!("session claim and {other_name} do not match");
        tracing::debug!("session claim: {session_claim:#?}");
        tracing::debug!("{other_name} claim: {other_claim:#?}");
        bail!(
            "session claim: {} != {other_name} claim: {}",
            hex::encode(session_claim_digest),
            hex::encode(other_claim_digest)
        );
    }
    Ok(())
}

/// The HAL's buffer pool keeps up to (total VRAM - 8 GB) of freed buffers, sized for the po2 it
/// last proved. When a larger po2 follows in the same process, those cached buffers do not fit
/// the new sizes and, with the new working set, exhausted the device for allocations made
/// outside the pool (nttq's T2, eval_check's scratch): composite 20,21 in one process failed
/// with "out of memory". Release the pool once per po2 increase (the calling thread's pool:
/// segment proving runs on this thread).
fn release_pool_on_po2_growth(po2: u32) {
    #[cfg(feature = "cuda")]
    {
        use std::sync::atomic::{AtomicU32, Ordering};
        static MAX_PO2: AtomicU32 = AtomicU32::new(0);
        let prev = MAX_PO2.fetch_max(po2, Ordering::SeqCst);
        if prev != 0 && po2 > prev {
            super::release_hal_buffers();
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = po2;
}

/// Lift (and left-fold join) segment receipts on the main thread between segment proofs, with
/// every preparation (preflight + host chains) on background threads. Each join's preparation
/// starts as soon as its left input exists (streaming the right input in when its lift
/// finishes), and ready joins run before lifts.
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod rec_pipe {
    use super::*;
    use crate::host::recursion::prove::{make_succinct_receipt, Prover as Job};
    use risc0_circuit_recursion::prove::InputStream;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::thread::JoinHandle;
    use std::time::Instant;

    /// A recursion proof's preparation thread (preflight + host chains; CPU only). Joined when
    /// dropped, so that on error paths no preparation outlives the proving call that started it.
    pub(super) struct Prep(Option<JoinHandle<Result<(Job, f64)>>>);

    impl Prep {
        fn spawn(f: impl FnOnce() -> Result<(Job, f64)> + Send + 'static) -> Self {
            Self(Some(std::thread::spawn(f)))
        }

        fn is_finished(&self) -> bool {
            self.0.as_ref().is_none_or(|h| h.is_finished())
        }

        fn join(mut self) -> Result<(Job, f64)> {
            let h = self
                .0
                .take()
                .ok_or_else(|| anyhow!("recursion preparation already joined"))?;
            h.join()
                .map_err(|_| anyhow!("recursion preparation panicked"))?
        }
    }

    impl Drop for Prep {
        fn drop(&mut self) {
            if let Some(h) = self.0.take() {
                let _ = h.join();
            }
        }
    }

    /// Closes a streaming join's input when dropped. The join's preparation thread blocks in
    /// `InputStream::pull` until the right input is fed and the stream closed; if the pipe fails
    /// before (or while) feeding it, closing makes the preparation fail and exit instead of
    /// waiting forever. Closing an already closed stream is a no-op.
    struct CloseOnDrop(Arc<InputStream>);

    impl Drop for CloseOnDrop {
        fn drop(&mut self) {
            self.0.close();
        }
    }

    /// The interleaved recursion pipeline (this module, the early seal, the streaming joins and
    /// the recursion ctrl prewarm during segment 0): on unless `RISC0_CUDA_REC_PIPE=0`.
    pub(super) fn enabled() -> bool {
        static ON: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var("RISC0_CUDA_REC_PIPE").as_deref() != Ok("0"));
        *ON
    }

    /// When the only ready lift is the right input of the join after the pending (fed) one, wait
    /// up to this long for that join's preparation, so the next join's left half can prepare
    /// while the lift runs.
    const JOIN_FIRST_MS: f64 = 15.0;

    static T0: std::sync::LazyLock<Instant> = std::sync::LazyLock::new(Instant::now);

    fn now_ms() -> f64 {
        T0.elapsed().as_secs_f64() * 1e3
    }

    pub(super) fn spawn_lift_prep(seal_receipt: SegmentReceipt, opts: ProverOpts) -> Prep {
        let _ = *T0;
        Prep::spawn(move || {
            let t = Instant::now();
            let mut p = Job::new_lift(&seal_receipt, opts)?;
            p.prepare()?;
            Ok((p, t.elapsed().as_secs_f64() * 1e3))
        })
    }

    struct PendingJoin {
        /// Streaming: the stream and the left receipt, until `b` is fed. Declared before `prep`
        /// so that on drop the stream is closed before the preparation thread is joined.
        feed: Option<(CloseOnDrop, SuccinctReceipt<ReceiptClaim>)>,
        prep: Prep,
        /// Joined claim, once `b` is known.
        claim: Option<ReceiptClaim>,
        a_claim: MaybePruned<ReceiptClaim>,
        idx: usize,
    }

    pub(super) struct RecPipe {
        opts: ProverOpts,
        lift_preps: BTreeMap<usize, (Prep, Option<ReceiptClaim>)>,
        lifted: BTreeMap<usize, SuccinctReceipt<ReceiptClaim>>,
        acc: Option<SuccinctReceipt<ReceiptClaim>>,
        next: usize,
        join: Option<PendingJoin>,
        last_claim: Option<ReceiptClaim>,
        warmed: bool,
        gpu_ms: f64,
        hold_since: Option<Instant>,
    }

    impl RecPipe {
        pub(super) fn new(opts: ProverOpts) -> Self {
            let _ = *T0;
            Self {
                opts,
                lift_preps: BTreeMap::new(),
                lifted: BTreeMap::new(),
                acc: None,
                next: 0,
                join: None,
                last_claim: None,
                warmed: false,
                gpu_ms: 0.0,
                hold_since: None,
            }
        }

        pub(super) fn opts(&self) -> ProverOpts {
            self.opts.clone()
        }

        pub(super) fn add(&mut self, idx: usize, prep: Prep, claim: Option<ReceiptClaim>) {
            tracing::trace!("rec_pipe: lift prep {idx} spawned at {:.1}", now_ms());
            self.lift_preps.insert(idx, (prep, claim));
        }

        fn warm(&mut self) {
            if !self.warmed {
                #[cfg(feature = "cuda")]
                join_cuda_rec_warmup();
                self.warmed = true;
            }
        }

        /// Feed `b` to a streaming join whose right lift is done.
        fn maybe_feed(&mut self) -> Result<()> {
            let Some(j) = self.join.as_mut() else {
                return Ok(());
            };
            if j.feed.is_none() {
                return Ok(());
            }
            let Some(b) = self.lifted.remove(&j.idx) else {
                return Ok(());
            };
            let (stream, a) = j.feed.take().unwrap();
            Job::feed_join_b(&stream.0, &a, &b)?;
            j.claim = Some(j.a_claim.join(&b.claim)?.value()?);
            tracing::trace!("rec_pipe: join ->{} fed b at {:.1}", j.idx, now_ms());
            Ok(())
        }

        fn maybe_start_join(&mut self) -> Result<()> {
            if self.join.is_some() {
                return self.maybe_feed();
            }
            if self.acc.is_none() && self.next == 0 {
                if let Some(l0) = self.lifted.remove(&0) {
                    self.acc = Some(l0);
                    self.next = 1;
                }
            }
            let Some(_) = self.acc.as_ref() else {
                return Ok(());
            };
            let idx = self.next;
            if self.lifted.contains_key(&idx) {
                let a = self.acc.take().unwrap();
                let b = self.lifted.remove(&idx).unwrap();
                let claim = a.claim.join(&b.claim)?.value()?;
                let opts = self.opts.clone();
                let a_claim = a.claim.clone();
                let h = Prep::spawn(move || {
                    let t = Instant::now();
                    let mut p = Job::new_join(&a, &b, opts)?;
                    p.prepare()?;
                    Ok((p, t.elapsed().as_secs_f64() * 1e3))
                });
                self.join = Some(PendingJoin {
                    prep: h,
                    claim: Some(claim),
                    feed: None,
                    a_claim,
                    idx,
                });
                self.next += 1;
                tracing::trace!(
                    "rec_pipe: join ->{idx} prep start (full) at {:.1}",
                    now_ms()
                );
            } else if self.lift_preps.contains_key(&idx) {
                let a = self.acc.take().unwrap();
                let a_claim = a.claim.clone();
                let (job, stream) = Job::new_join_streaming(&a, self.opts.clone())?;
                let stream = CloseOnDrop(stream);
                let h = Prep::spawn(move || {
                    let t = Instant::now();
                    let mut p = job;
                    p.prepare()?;
                    Ok((p, t.elapsed().as_secs_f64() * 1e3))
                });
                self.join = Some(PendingJoin {
                    prep: h,
                    claim: None,
                    feed: Some((stream, a)),
                    a_claim,
                    idx,
                });
                self.next += 1;
                tracing::trace!(
                    "rec_pipe: join ->{idx} prep start (stream) at {:.1}",
                    now_ms()
                );
            }
            Ok(())
        }

        /// Run one ready unit of GPU work; with `block`, wait for one if none is ready.
        /// Returns false when nothing was run (and, with `block`, nothing remains).
        fn step(&mut self, block: bool) -> Result<bool> {
            self.maybe_start_join()?;
            let join_ready = self
                .join
                .as_ref()
                .is_some_and(|j| j.feed.is_none() && j.prep.is_finished());
            if join_ready {
                let j = self.join.take().unwrap();
                let claim = j.claim.unwrap();
                let (mut job, prep_ms) = j.prep.join()?;
                self.warm();
                let t = Instant::now();
                let at = now_ms();
                let receipt = job.run()?;
                let decoded = ReceiptClaim::decode(&mut receipt.out_stream())?;
                let merged = decoded.merge(&claim)?;
                self.acc = Some(make_succinct_receipt(job, receipt, merged)?);
                let run_ms = t.elapsed().as_secs_f64() * 1e3;
                self.gpu_ms += run_ms;
                tracing::debug!(
                    "rec_pipe: join ->{} at {at:.1} prep={prep_ms:.1} run={run_ms:.1}",
                    j.idx
                );
                return Ok(true);
            }
            let ready = self
                .lift_preps
                .iter()
                .find(|(_, (h, c))| h.is_finished() && (c.is_some() || self.last_claim.is_some()))
                .map(|(k, _)| *k);
            // The only ready lift is the right input of the join after the pending (fed) one:
            // wait up to JOIN_FIRST_MS for that join's preparation (see JOIN_FIRST_MS).
            if let (Some(k), Some(j)) = (ready, self.join.as_ref()) {
                if block && j.feed.is_none() && k == self.next {
                    let since = *self.hold_since.get_or_insert_with(Instant::now);
                    if since.elapsed().as_secs_f64() * 1e3 < JOIN_FIRST_MS {
                        std::thread::sleep(std::time::Duration::from_micros(100));
                        return Ok(true);
                    }
                }
            }
            self.hold_since = None;
            if let Some(k) = ready {
                let (h, claim) = self.lift_preps.remove(&k).unwrap();
                let claim = match claim {
                    Some(c) => c,
                    None => self.last_claim.clone().unwrap(),
                };
                let (mut job, prep_ms) = h.join()?;
                self.warm();
                let t = Instant::now();
                let at = now_ms();
                let receipt = job.run()?;
                let decoded = ReceiptClaim::decode(&mut receipt.out_stream())?;
                let merged = decoded.merge(&claim)?;
                self.lifted
                    .insert(k, make_succinct_receipt(job, receipt, merged)?);
                let run_ms = t.elapsed().as_secs_f64() * 1e3;
                self.gpu_ms += run_ms;
                tracing::debug!("rec_pipe: lift {k} at {at:.1} prep={prep_ms:.1} run={run_ms:.1}");
                self.maybe_feed()?;
                return Ok(true);
            }
            if !block {
                return Ok(false);
            }
            if self.join.is_none() && self.lift_preps.is_empty() && self.lifted.is_empty() {
                return Ok(false);
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
            Ok(true)
        }

        /// Run all work whose preparation is done, without waiting.
        pub(super) fn poll(&mut self) -> Result<()> {
            let t = Instant::now();
            let mut n = 0;
            while self.step(false)? {
                n += 1;
            }
            if n > 0 {
                tracing::debug!(
                    "rec_pipe: poll ran {n} proofs in {:.1}ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
            }
            Ok(())
        }

        pub(super) fn drain(
            mut self,
            last_claim: ReceiptClaim,
            num_segments: usize,
        ) -> Result<SuccinctReceipt<ReceiptClaim>> {
            let t = Instant::now();
            self.last_claim = Some(last_claim);
            while self.step(true)? {}
            ensure!(
                self.next == num_segments || (num_segments == 1 && self.acc.is_some()),
                "recursion pipe: folded {} of {num_segments} segments",
                self.next
            );
            tracing::debug!(
                "rec_pipe: drain {:.1}ms (total recursion run {:.1}ms) end at {:.1}",
                t.elapsed().as_secs_f64() * 1e3,
                self.gpu_ms,
                now_ms()
            );
            // ROCm: like composite_to_succinct's "lift/join done" line, for the benchmarks.
            #[cfg(feature = "rocm")]
            eprintln!(
                "[rec_pipe] {num_segments} segments: tail {:.3}s, lift/join runs {:.3}s",
                t.elapsed().as_secs_f64(),
                self.gpu_ms / 1e3
            );
            self.acc
                .take()
                .ok_or_else(|| anyhow!("recursion pipe: no receipt"))
        }
    }
}
