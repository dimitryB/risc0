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

pub(crate) mod cpu;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "rocm")]
pub(crate) mod hip;
#[cfg(feature = "intel")]
pub(crate) mod intel;

use std::{
    cell::RefCell,
    sync::{Arc, LazyLock},
};

use anyhow::Result;
use risc0_core::scope;

static VERBOSE: LazyLock<bool> = LazyLock::new(|| std::env::var("RISC0_VERBOSE").is_ok());
use risc0_zkp::{
    adapter::{CircuitInfo as _, PROOF_SYSTEM_INFO},
    core::digest::Digest,
    field::Elem as _,
    hal::{Buffer, CircuitHal, Hal},
    prove::{poly_group::PolyGroup, Prover},
};

use super::{
    witgen::{preflight::PreflightTrace, PreflightResults, WitnessGenerator},
    Seal, SegmentProver,
};
use crate::{
    execute::segment::Segment,
    zirgen::{
        circuit::{
            CircuitField, ExtVal, Val, REGCOUNT_MIX, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE,
            REGISTER_GROUP_DATA,
        },
        taps::TAPSET,
        CircuitImpl,
    },
    RV32IM_SEAL_VERSION,
};

pub(crate) struct MetaBuffer<H: Hal> {
    pub buf: H::Buffer<H::Elem>,
    pub rows: usize,
    pub cols: usize,
    pub checked: bool,
}

impl<H> MetaBuffer<H>
where
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
{
    pub fn new(name: &'static str, hal: &H, rows: usize, cols: usize, checked: bool) -> Self {
        let buf = hal.alloc_elem_init(name, rows * cols, Val::INVALID);
        Self {
            buf,
            rows,
            cols,
            checked,
        }
    }

    #[cfg(test)]
    pub fn to_vec(&self) -> Vec<Val> {
        self.buf.to_vec()
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum StepMode {
    Parallel,
    SeqForward,
    SeqReverse,
}

pub(crate) trait CircuitWitnessGenerator<H: Hal> {
    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<H>,
        data: &MetaBuffer<H>,
        pre_data: &MetaBuffer<H>,
    ) -> Result<()>;
}

pub(crate) trait CircuitAccumulator<H: Hal> {
    fn step_accum(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<H>,
        accum: &MetaBuffer<H>,
        global: &MetaBuffer<H>,
        mix: &MetaBuffer<H>,
    ) -> Result<()>;
}

pub(crate) struct SegmentProverImpl<H, C, F>
where
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H>,
    F: Fn() -> (Arc<H>, Arc<C>),
{
    hal_factory: F,
    cached_hal: RefCell<Option<(Arc<H>, Arc<C>)>>,
    /// Cached code PolyGroup (always zeros, same for every segment at same po2).
    cached_code_group: RefCell<Option<(PolyGroup<H>, usize)>>,
    /// Cached Fiat-Shamir setup digests. PROOF_SYSTEM_INFO and CIRCUIT_INFO are
    /// build-time constants; hashing them every segment costs ~1 ms on CPU.
    cached_fs_seed_digests: RefCell<Option<(Box<Digest>, Box<Digest>)>>,
    /// Seal of the segment finished by the previous prove_begin call, returned
    /// by the next prove_begin (or prove_end) to keep the pipelined API.
    pending_seal: RefCell<Option<Seal>>,
}

impl<H, C, F> SegmentProverImpl<H, C, F>
where
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H>,
    F: Fn() -> (Arc<H>, Arc<C>),
{
    pub fn new(hal_factory: F) -> Self {
        Self {
            hal_factory,
            cached_hal: RefCell::new(None),
            cached_code_group: RefCell::new(None),
            cached_fs_seed_digests: RefCell::new(None),
            pending_seal: RefCell::new(None),
        }
    }

    fn get_hal(&self) -> (Arc<H>, Arc<C>) {
        let mut cached = self.cached_hal.borrow_mut();
        if let Some(ref hal) = *cached {
            return (hal.0.clone(), hal.1.clone());
        }
        let hal = (self.hal_factory)();
        *cached = Some((hal.0.clone(), hal.1.clone()));
        (hal.0, hal.1)
    }
}

impl<H, C, F> SegmentProver for SegmentProverImpl<H, C, F>
where
    // No Send/Sync bounds: prove_begin finalizes on the calling thread (see below), and the CUDA
    // and HIP HALs are not Send.
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal> + 'static,
    C: CircuitHal<H> + CircuitWitnessGenerator<H> + CircuitAccumulator<H> + 'static,
    F: Fn() -> (Arc<H>, Arc<C>),
{
    fn metal_stats(&self) -> Option<super::MetalFoldStats> {
        self.get_hal().0.metal_stats()
    }

    fn preflight(&self, segment: &Segment) -> Result<PreflightResults> {
        scope!("preflight");

        cfg_if::cfg_if! {
            if #[cfg(feature = "witgen_debug")] {
                let rand_z = ExtVal::ONE;
            } else {
                let mut rng = rand::rng();
                let rand_z = ExtVal::random(&mut rng);
            }
        }
        PreflightResults::new(segment, rand_z)
    }

    fn prove_core(&self, preflight_results: PreflightResults) -> Result<Seal> {
        scope!("prove_core");
        let t0 = std::time::Instant::now();

        let mode = if std::env::var_os("RISC0_WITGEN_SEQ").is_some() {
            StepMode::SeqForward
        } else {
            StepMode::Parallel
        };

        let (hal, circuit_hal) = self.get_hal();
        if *VERBOSE {
            eprintln!(
                "[prove_core] hal_factory: {:.1}ms",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        let t1 = std::time::Instant::now();
        let po2 = preflight_results.po2();
        let expected_header = preflight_results.claim_header();
        let mut witgen =
            WitnessGenerator::new(hal.as_ref(), circuit_hal.as_ref(), preflight_results, mode)?;
        let header = witgen.header(po2, &expected_header);
        if *VERBOSE {
            eprintln!(
                "[prove_core] witgen: {:.1}ms",
                t1.elapsed().as_secs_f64() * 1000.0
            );
        }

        let code = &witgen.code.buf;
        let data = &witgen.data.as_ref().expect("witgen.data present").buf;

        let t2 = std::time::Instant::now();
        let seal = scope!("prove_inner", {
            tracing::debug!("prove_inner");

            let mut prover = Prover::new(hal.as_ref(), TAPSET);
            let hashfn = &hal.get_hash_suite().hashfn;

            // Add a version tag to the start of the seal. It's not intended for
            // this value to be consumed by downstream lift predicates. Instead
            // it's meant to be a header which can be used by future versions in
            // case we'd like to adjust the encoding of the seal.
            //
            // We write via the IOP to avoid having to prepend a value to an
            // already allocated vector.
            prover.iop().write_u32_slice(&[RV32IM_SEAL_VERSION]);

            let mt0 = std::time::Instant::now();
            let (mix, global_clone) = scope!("main", {
                // At the start of the protocol, seed the Fiat-Shamir transcript with context information
                // about the proof system and circuit.
                prover
                    .iop()
                    .commit(&hashfn.hash_elem_slice(&PROOF_SYSTEM_INFO.encode()));
                prover
                    .iop()
                    .commit(&hashfn.hash_elem_slice(&CircuitImpl::CIRCUIT_INFO.encode()));

                let header_digest = hashfn.hash_elem_slice(&header);
                prover.iop().commit(&header_digest);
                prover.iop().write_field_elem_slice(header.as_slice());
                prover.set_po2(po2 as usize);
                if *VERBOSE {
                    eprintln!(
                        "  [main] setup: {:.1}ms",
                        mt0.elapsed().as_secs_f64() * 1000.0
                    );
                }

                // Code buffer is always zeros (INVALID → zeroized, never written to).
                // Cache its PolyGroup to skip iNTT/expand/merkle on subsequent segments
                // with the same po2. Different po2 = different domain size = must recompute.
                {
                    let mut cache = self.cached_code_group.borrow_mut();
                    if let Some((ref cached, cached_po2)) = *cache {
                        if cached_po2 == po2 as usize {
                            prover.commit_cached_group(REGISTER_GROUP_CODE, cached.clone());
                            if *VERBOSE {
                                eprintln!(
                                    "  [main] commit(code) [cached]: {:.1}ms",
                                    mt0.elapsed().as_secs_f64() * 1000.0
                                );
                            }
                        } else {
                            prover.commit_group(REGISTER_GROUP_CODE, code);
                            *cache = prover
                                .get_group(REGISTER_GROUP_CODE)
                                .cloned()
                                .map(|g| (g, po2 as usize));
                            if *VERBOSE {
                                eprintln!("  [main] commit(code) [recomputed, po2 changed {cached_po2}->{po2}]: {:.1}ms", mt0.elapsed().as_secs_f64() * 1000.0);
                            }
                        }
                    } else {
                        prover.commit_group(REGISTER_GROUP_CODE, code);
                        *cache = prover
                            .get_group(REGISTER_GROUP_CODE)
                            .cloned()
                            .map(|g| (g, po2 as usize));
                        if *VERBOSE {
                            eprintln!(
                                "  [main] commit(code) [computed+cached]: {:.1}ms",
                                mt0.elapsed().as_secs_f64() * 1000.0
                            );
                        }
                    }
                }
                prover.commit_group(REGISTER_GROUP_DATA, data);
                if *VERBOSE {
                    eprintln!(
                        "  [main] commit(data): {:.1}ms",
                        mt0.elapsed().as_secs_f64() * 1000.0
                    );
                }

                // Make the mixing values
                let mix: [Val; REGCOUNT_MIX] = std::array::from_fn(|_| prover.iop().random_elem());

                let mix = witgen.accum(hal.as_ref(), circuit_hal.as_ref(), &mix)?;
                if *VERBOSE {
                    eprintln!(
                        "  [main] accum: {:.1}ms",
                        mt0.elapsed().as_secs_f64() * 1000.0
                    );
                }

                prover.commit_group(REGISTER_GROUP_ACCUM, &witgen.accum.as_ref().unwrap().buf);
                if *VERBOSE {
                    eprintln!(
                        "  [main] commit(accum): {:.1}ms",
                        mt0.elapsed().as_secs_f64() * 1000.0
                    );
                }

                // Clone tiny global buffer (90 elements = 360 bytes) so witgen can
                // be dropped BEFORE eval_check launches — needed to fit po2=22
                // in 24 GB (frees ~5 GB of witgen buffers).
                let global_clone = hal.alloc_elem("global_clone", witgen.global.buf.size());
                hal.eltwise_copy_elem(&global_clone, &witgen.global.buf);

                (mix, global_clone)
            });

            // Drop witgen before eval_check launches to reclaim witgen.data
            // (3.4 GB at po2=22) + witgen.accum (1.65 GB). Previously dropped
            // inside a post_eval_check closure to overlap with GPU compute,
            // but that keeps witgen live across eval_check's peak VRAM use.
            // Free the host-side trace while eval_check runs (see prove_begin).
            let trace = std::mem::take(&mut witgen.trace);
            drop(witgen);

            let t3 = std::time::Instant::now();
            let result = prover.finalize_with_hook(
                &[&mix.buf, &global_clone],
                circuit_hal.as_ref(),
                move || drop(trace),
            );

            if *VERBOSE {
                eprintln!(
                    "[prove_core] prove_inner: {:.1}ms (main: {:.1}ms, finalize: {:.1}ms)",
                    t2.elapsed().as_secs_f64() * 1000.0,
                    (t3 - t2).as_secs_f64() * 1000.0,
                    t3.elapsed().as_secs_f64() * 1000.0
                );
                eprintln!(
                    "[prove_core] TOTAL: {:.1}ms",
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            result
        });

        Ok(seal)
    }

    fn prove_begin(&self, preflight_results: PreflightResults) -> Result<Option<Seal>> {
        scope!("prove_begin");
        let t0 = std::time::Instant::now();

        let mode = if std::env::var_os("RISC0_WITGEN_SEQ").is_some() {
            StepMode::SeqForward
        } else {
            StepMode::Parallel
        };

        let (hal, circuit_hal) = self.get_hal();

        let po2 = preflight_results.po2();
        let expected_header = preflight_results.claim_header();
        let mut witgen =
            WitnessGenerator::new(hal.as_ref(), circuit_hal.as_ref(), preflight_results, mode)?;
        let header = witgen.header(po2, &expected_header);
        let t_witgen = t0.elapsed();

        let code = &witgen.code.buf;
        let data = &witgen.data.as_ref().expect("witgen.data present").buf;

        let mut prover = Prover::new(hal.as_ref(), TAPSET);
        let hashfn = &hal.get_hash_suite().hashfn;

        prover.iop().write_u32_slice(&[RV32IM_SEAL_VERSION]);

        // Seed Fiat-Shamir transcript with cached PROOF_SYSTEM_INFO and
        // CIRCUIT_INFO digests (both are build-time constants; hashing every
        // segment costs ~1 ms on CPU).
        {
            let mut cache = self.cached_fs_seed_digests.borrow_mut();
            if cache.is_none() {
                let p = hashfn.hash_elem_slice(&PROOF_SYSTEM_INFO.encode());
                let c = hashfn.hash_elem_slice(&CircuitImpl::CIRCUIT_INFO.encode());
                *cache = Some((p, c));
            }
            let (p, c) = cache.as_ref().unwrap();
            prover.iop().commit(p.as_ref());
            prover.iop().commit(c.as_ref());
        }

        let header_digest = hashfn.hash_elem_slice(&header);
        prover.iop().commit(&header_digest);
        prover.iop().write_field_elem_slice(header.as_slice());
        prover.set_po2(po2 as usize);

        // Code group caching
        {
            let mut cache = self.cached_code_group.borrow_mut();
            if let Some((ref cached, cached_po2)) = *cache {
                if cached_po2 == po2 as usize {
                    prover.commit_cached_group(REGISTER_GROUP_CODE, cached.clone());
                } else {
                    prover.commit_group(REGISTER_GROUP_CODE, code);
                    *cache = prover
                        .get_group(REGISTER_GROUP_CODE)
                        .cloned()
                        .map(|g| (g, po2 as usize));
                }
            } else {
                prover.commit_group(REGISTER_GROUP_CODE, code);
                *cache = prover
                    .get_group(REGISTER_GROUP_CODE)
                    .cloned()
                    .map(|g| (g, po2 as usize));
            }
        }

        let td = std::time::Instant::now();
        prover.commit_group(REGISTER_GROUP_DATA, data);
        let t_data_commit = td.elapsed();

        let ta = std::time::Instant::now();
        let mix: [Val; REGCOUNT_MIX] = std::array::from_fn(|_| prover.iop().random_elem());
        let mix = witgen.accum(hal.as_ref(), circuit_hal.as_ref(), &mix)?;
        let t_accum = ta.elapsed();

        let tac = std::time::Instant::now();
        prover.commit_group(
            REGISTER_GROUP_ACCUM,
            &witgen.accum.as_ref().unwrap().buf,
        );
        let t_accum_commit = tac.elapsed();

        let global_clone = hal.alloc_elem("global_clone", witgen.global.buf.size());
        hal.eltwise_copy_elem(&global_clone, &witgen.global.buf);

        let prev_seal = self.pending_seal.borrow_mut().take();

        // Drop witgen BEFORE eval_check launches to free 5 GB of VRAM
        // (witgen.data 3.4 GB + witgen.accum 1.65 GB at po2=22). The eval_check
        // kernel only needs globals (copied to global_clone) and the committed
        // PolyGroups' `evaluated` buffers, not any witgen buffer.
        // The host-side preflight trace holds no VRAM; free it in the hook,
        // which runs while eval_check is on the GPU, not on the critical path
        // (~13 ms/segment at po2=20 on Intel).
        let trace = std::mem::take(&mut witgen.trace);
        drop(witgen);

        let t_main = t0.elapsed();

        // Start finalize: launch eval_check on stream2 (async on GPU).
        let mut deferred = prover.start_finalize_with_hook(
            &[&mix.buf, &global_clone],
            circuit_hal.as_ref(),
            move || drop(trace),
        );
        // Keep globals alive until eval_check completes.
        deferred.keep_alive_buf(mix.buf);
        deferred.keep_alive_buf(global_clone);

        // Finalize on this thread. A background finalize thread overlapping the
        // next segment's GPU work (allocating and freeing device memory
        // concurrently with it) intermittently corrupted device data on Intel
        // (GPU page faults / invalid proofs), and holding two segments in
        // flight does not fit po2=21 in VRAM. With one in-order queue the GPU
        // work serialized anyway, so the overlap bought ~nothing.
        let seal = deferred.complete(hal.as_ref(), circuit_hal.as_ref());
        *self.pending_seal.borrow_mut() = Some(seal);

        if *VERBOSE {
            eprintln!(
            "[prove_begin] witgen={:.1}ms data_commit={:.1}ms accum={:.1}ms accum_commit={:.1}ms main={:.1}ms total={:.1}ms",
            t_witgen.as_secs_f64() * 1000.0,
            t_data_commit.as_secs_f64() * 1000.0,
            t_accum.as_secs_f64() * 1000.0,
            t_accum_commit.as_secs_f64() * 1000.0,
            t_main.as_secs_f64() * 1000.0,
            t0.elapsed().as_secs_f64() * 1000.0
        );
        }

        Ok(prev_seal)
    }

    fn reset_pipeline(&self) {
        self.pending_seal.borrow_mut().take();
    }

    fn prove_end(&self) -> Result<Seal> {
        scope!("prove_end");
        self.pending_seal
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow::anyhow!("prove_end: no pending seal"))
    }
}
