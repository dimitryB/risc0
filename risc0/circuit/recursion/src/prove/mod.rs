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

//! Prover implementation for the recursion VM.
//!
//! This module contains the recursion [Prover].

mod hal;
mod preflight;
mod program;
mod witgen;
pub mod zkr;

use std::{
    cell::RefCell,
    collections::HashMap,
    collections::VecDeque,
    fmt::Debug,
    sync::{Arc, LazyLock},
};

use anyhow::Result;
use cfg_if::cfg_if;
use risc0_core::scope;

static VERBOSE: LazyLock<bool> = LazyLock::new(|| std::env::var("RISC0_VERBOSE").is_ok());
use risc0_zkp::{
    adapter::{CircuitInfo, PROOF_SYSTEM_INFO},
    core::digest::Digest,
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        Elem as _,
    },
    hal::{Buffer, CircuitHal, Hal},
    prove::poly_group::PolyGroup,
};
use serde::{Deserialize, Serialize};

use self::{
    hal::{CircuitAccumulator, CircuitWitnessGenerator},
    witgen::WitnessGenerator,
};
use crate::{
    taps::TAPSET, CircuitImpl, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CTRL, REGISTER_GROUP_DATA,
};

pub use self::preflight::Preflight;
pub use self::program::Program;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub use self::hal::metal::{metal_recursion_stats, metal_recursion_kernel_source, MetalRecursionStats};

// TODO: Automatically generate this constant from the circuit somehow without
// messing up bootstrap dependencies.
/// Size of the code group in the taps of the recursion circuit.
const RECURSION_CODE_SIZE: usize = 23;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RecursionReceipt {
    pub seal: Vec<u32>,
    pub output: Vec<u32>,
}

impl RecursionReceipt {
    /// Total number of bytes used by the seal of this receipt.
    pub fn seal_size(&self) -> usize {
        core::mem::size_of_val(self.seal.as_slice())
    }

    /// Allocates a [VecDeque] and copies the output stream into it for decoding.
    pub fn out_stream(&self) -> VecDeque<u32> {
        let mut vec: VecDeque<u32> = VecDeque::new();
        vec.extend(self.output.iter());
        vec
    }
}

pub trait RecursionProver {
    fn prove(
        &self,
        program: &Program,
        input: VecDeque<u32>,
        preflight: Option<Preflight>,
    ) -> Result<RecursionReceipt>;
}

pub fn recursion_prover(hashfn: &str) -> Result<Box<dyn RecursionProver>> {
    cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::hal::cuda::recursion_prover(hashfn)
        } else if #[cfg(feature = "rocm")] {
            self::hal::hip::recursion_prover(hashfn)
        } else if #[cfg(feature = "intel")] {
            self::hal::intel::recursion_prover(hashfn)
        } else if #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))] {
            self::hal::metal::recursion_prover(hashfn)
        } else {
            self::hal::cpu::recursion_prover(hashfn)
        }
    }
}

/// Prover for the recursion circuit.
pub struct Prover {
    program: Arc<Program>,
    hashfn: String,
    input: VecDeque<u32>,
    /// Pre-computed preflight result (set by `prepare()`).
    prepared_preflight: Option<Preflight>,
}

/// Kinds of digests recognized by the recursion program language.
// NOTE: Default is additionally a recognized type in the recursion program language. It's not
// yet supported here because some of the code in this module assumes Poseidon2 is Default.
#[non_exhaustive]
pub enum DigestKind {
    Poseidon2,
    Sha256,
}

impl Prover {
    /// Creates a new prover with the given recursion program.
    pub fn new(program: Arc<Program>, hashfn: &str) -> Self {
        Self {
            program,
            hashfn: hashfn.to_string(),
            input: VecDeque::new(),
            prepared_preflight: None,
        }
    }

    /// Pre-compute the CPU preflight. Can be called from any thread.
    /// If called before `run()`, `run()` uses the pre-computed result
    /// instead of computing the preflight on the GPU thread.
    pub fn prepare(&mut self) -> Result<()> {
        let input = std::mem::take(&mut self.input);
        let mut preflight = Preflight::new(input);
        for (cycle, row) in self.program.code_by_row().enumerate() {
            preflight.step(cycle, row)?;
        }
        #[cfg(feature = "intel")]
        // Default on; RISC0_RECURSION_PREWITGEN=0 keeps witgen on the GPU thread.
        if !matches!(
            std::env::var("RISC0_RECURSION_PREWITGEN").as_deref(),
            Ok("0")
        ) {
            let t0 = std::time::Instant::now();
            let w = self::witgen::host_witgen(&self.program, &preflight)?;
            *preflight.host_witness.lock().unwrap() = Some(w);
            if *VERBOSE {
                eprintln!(
                    "[recursion_prewitgen] host witgen on prepare thread: {:.1}ms",
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        self.prepared_preflight = Some(preflight);
        Ok(())
    }

    /// Add a set of u32s to the input for the recursion program.
    pub fn add_input(&mut self, input: &[u32]) {
        self.input.extend(input);
    }

    /// Add a digest to the input for the recursion program.
    pub fn add_input_digest(&mut self, digest: &Digest, kind: DigestKind) {
        match kind {
            // Poseidon2 digests consist of  BabyBear field elems and do not need to be split.
            DigestKind::Poseidon2 => self.add_input(digest.as_words()),
            // SHA-256 digests need to be split into 16-bit half words to avoid overflowing.
            DigestKind::Sha256 => self.add_input(bytemuck::cast_slice(
                &digest
                    .as_words()
                    .iter()
                    .copied()
                    .flat_map(|x| [x & 0xffff, x >> 16])
                    .map(BabyBearElem::new)
                    .collect::<Vec<_>>(),
            )),
        }
    }

    /// Run the prover, producing a receipt of execution for the recursion circuit over the loaded
    /// program and input.
    pub fn run(&mut self) -> Result<RecursionReceipt> {
        // Exclusive use of the GPU against other threads for this proof; the cached prover's HAL
        // doesn't hold it (see `risc0_zkp::hal::cuda::singleton`).
        #[cfg(feature = "cuda")]
        let _gpu = risc0_zkp::hal::cuda::exclusive();
        // Cache provers by hashfn to avoid recreating HAL + circuit_hal each call.
        // Using HashMap instead of single-entry cache prevents eviction when hashfn
        // alternates (e.g., "poseidon2" for lift/join → "poseidon_254" for identity_p254),
        // preserving each prover's cached ctrl PolyGroup across calls.
        thread_local! {
            static PROVER_CACHE: RefCell<HashMap<String, Box<dyn RecursionProver>>> =
                RefCell::new(HashMap::new());
        }
        let hashfn = self.hashfn.clone();
        // Take input instead of cloning to avoid copying ~4-8MB of seal data per call.
        let input = std::mem::take(&mut self.input);
        let preflight = self.prepared_preflight.take();
        PROVER_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if !cache.contains_key(&hashfn) {
                let p = recursion_prover(&hashfn)?;
                cache.insert(hashfn.clone(), p);
            }
            // Pass &Program to avoid cloning ~24MB code vector per call.
            // Pass pre-computed preflight if available (from prepare()).
            cache
                .get(&hashfn)
                .unwrap()
                .prove(&self.program, input, preflight)
        })
    }
}

pub(crate) struct RecursionProverImpl<H, C>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H>,
{
    hal: Arc<H>,
    circuit_hal: Arc<C>,
    // Cache ctrl PolyGroup by program to skip iNTT/expand/merkle on
    // repeated proofs with the same program (e.g. 43 lifts all use the same ZKR).
    cached_ctrl_group: RefCell<HashMap<CtrlKey, PolyGroup<H>>>,
    // Cache ctrl GPU buffer by program to skip re-uploading ~24MB per proof.
    cached_ctrl_buffer: RefCell<HashMap<CtrlKey, H::Buffer<H::Elem>>>,
    // The code of the program each key's entries above were built from. The key only
    // samples the code, so a hit is checked against this exactly (`claim_ctrl_caches`).
    cached_ctrl_code: RefCell<HashMap<CtrlKey, Arc<Vec<BabyBearElem>>>>,
}

/// Programs whose ctrl the device caches keep at once; more clears them (each holds a
/// ~24 MB device buffer and its committed group).
const CTRL_DEVICE_CACHE_CAP: usize = 8;

/// Whether two programs' code is identical, compared in parallel (~24 MB for a lift).
fn same_code(a: &[BabyBearElem], b: &[BabyBearElem]) -> bool {
    use rayon::prelude::*;
    const CHUNK: usize = 1 << 16;
    a.len() == b.len()
        && a.par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .all(|(x, y)| x == y)
}

/// (code_rows, po2, sampled code fingerprint). All recursion programs share
/// RECURSION_PO2, so code_rows alone would be the only distinguishing field;
/// two programs of equal length would silently reuse the wrong ctrl group and
/// yield proofs that fail verification. The fingerprint samples the head,
/// tail and a stride of the code so distinct programs get distinct keys
/// without hashing the full ~24 MB on every proof.
type CtrlKey = (usize, usize, u64);

fn ctrl_cache_key(program: &Program) -> CtrlKey {
    use std::hash::{Hash, Hasher};

    const EDGE: usize = 1024;
    const SAMPLES: usize = 256;
    let code = &program.code;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    code.len().hash(&mut h);
    let mut feed = |e: &BabyBearElem| e.as_u32_montgomery().hash(&mut h);
    code.iter().take(EDGE).for_each(&mut feed);
    code.iter().rev().take(EDGE).for_each(&mut feed);
    let stride = (code.len() / SAMPLES).max(1);
    code.iter().step_by(stride).for_each(&mut feed);
    (program.code_rows(), program.po2, h.finish())
}

impl<H, C> RecursionProver for RecursionProverImpl<H, C>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem> + 'static,
    C: CircuitHal<H> + CircuitWitnessGenerator<H> + CircuitAccumulator<H>,
{
    fn prove(
        &self,
        program: &Program,
        input: VecDeque<u32>,
        preflight: Option<Preflight>,
    ) -> Result<RecursionReceipt> {
        scope!("prove");
        let t_total = std::time::Instant::now();

        let t0 = std::time::Instant::now();
        let (preflight, preflight_was_cached) = if let Some(pf) = preflight {
            (pf, true)
        } else {
            (self.preflight(program, input)?, false)
        };
        let preflight_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = std::time::Instant::now();
        let ctrl_key = ctrl_cache_key(program);
        self.claim_ctrl_caches(ctrl_key, program);
        let cached_ctrl = self.cached_ctrl_buffer.borrow().get(&ctrl_key).cloned();
        let witgen = WitnessGenerator::new(
            self.hal.as_ref(),
            self.circuit_hal.as_ref(),
            program,
            &preflight,
            cached_ctrl,
        )?;
        // Cache the ctrl GPU buffer for reuse by subsequent same-program proofs.
        if !self.cached_ctrl_buffer.borrow().contains_key(&ctrl_key) {
            self.cached_ctrl_buffer
                .borrow_mut()
                .insert(ctrl_key, witgen.ctrl.clone());
        }
        let witgen_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let global = &witgen.global;

        let seal = scope!("prove", {
            let mut prover = risc0_zkp::prove::Prover::new(self.hal.as_ref(), TAPSET);
            let hashfn = &self.hal.get_hash_suite().hashfn;

            let mix = scope!("main", {
                // At the start of the protocol, seed the Fiat-Shamir transcript with context information
                // about the proof system and circuit.
                prover
                    .iop()
                    .commit(&hashfn.hash_elem_slice(&PROOF_SYSTEM_INFO.encode()));
                prover
                    .iop()
                    .commit(&hashfn.hash_elem_slice(&CircuitImpl::CIRCUIT_INFO.encode()));

                // Concat globals and po2 into a vector.
                let global_len = global.size();
                let mut header = vec![BabyBearElem::ZERO; global_len + 1];
                global.view_mut(|view| {
                    for (i, elem) in view.iter_mut().enumerate() {
                        *elem = elem.valid_or_zero();
                        header[i] = *elem;
                    }
                    header[global_len] = BabyBearElem::new_raw(program.po2 as u32);
                });

                let header_digest = hashfn.hash_elem_slice(&header);
                prover.iop().commit(&header_digest);
                prover.iop().write_field_elem_slice(header.as_slice());
                prover.set_po2(program.po2);

                // Cache ctrl PolyGroup to skip iNTT/expand/merkle on repeated proofs
                // with the same program (`ctrl_key`, whose entries `claim_ctrl_caches`
                // checked belong to exactly this program).
                let t0 = std::time::Instant::now();
                let ctrl_cached;
                {
                    let mut cache = self.cached_ctrl_group.borrow_mut();
                    if let Some(cached) = cache.get(&ctrl_key) {
                        prover.commit_cached_group(REGISTER_GROUP_CTRL, cached.clone());
                        ctrl_cached = true;
                    } else {
                        prover.commit_group(REGISTER_GROUP_CTRL, &witgen.ctrl);
                        if let Some(group) = prover.get_group(REGISTER_GROUP_CTRL).cloned() {
                            cache.insert(ctrl_key, group);
                        }
                        ctrl_cached = false;
                    }
                }
                let ctrl_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = std::time::Instant::now();
                prover.commit_group(REGISTER_GROUP_DATA, &witgen.data);
                let data_commit_ms = t0.elapsed().as_secs_f64() * 1000.0;

                // Make the mixing values
                let mix: [BabyBearElem; CircuitImpl::MIX_SIZE] =
                    std::array::from_fn(|_| prover.iop().random_elem());

                let t0 = std::time::Instant::now();
                let mix = witgen.accum(&self.hal, self.circuit_hal.as_ref(), &mix)?;
                let accum_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = std::time::Instant::now();
                prover.commit_group(REGISTER_GROUP_ACCUM, &witgen.accum);
                let accum_commit_ms = t0.elapsed().as_secs_f64() * 1000.0;

                if *VERBOSE {
                    eprintln!("[recursion_prove] preflight={preflight_ms:.1}ms(cached={preflight_was_cached}) witgen={witgen_ms:.1}ms ctrl={ctrl_ms:.1}ms(cached={ctrl_cached}) data_commit={data_commit_ms:.1}ms accum={accum_ms:.1}ms accum_commit={accum_commit_ms:.1}ms");
                }

                mix
            });

            let t0 = std::time::Instant::now();
            let seal = prover.finalize(&[&mix, global], self.circuit_hal.as_ref());
            let fri_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if *VERBOSE {
                eprintln!(
                    "[recursion_prove] fri={fri_ms:.1}ms total={:.1}ms",
                    t_total.elapsed().as_secs_f64() * 1000.0
                );
            }
            seal
        });

        Ok(RecursionReceipt {
            seal,
            output: preflight.output,
        })
    }
}

impl<H, C> RecursionProverImpl<H, C>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H>,
{
    pub fn new(hal: Arc<H>, circuit_hal: Arc<C>) -> Self {
        Self {
            hal,
            circuit_hal,
            cached_ctrl_group: RefCell::new(HashMap::new()),
            cached_ctrl_buffer: RefCell::new(HashMap::new()),
            cached_ctrl_code: RefCell::new(HashMap::new()),
        }
    }

    /// Make the device ctrl caches' entries under `key` belong to exactly `program`. The key
    /// samples the code, so another program of the same size can share it: if the entries
    /// were built from different code (or there are none), they are dropped, to be rebuilt
    /// from `program`'s own code, instead of committing another program's ctrl.
    fn claim_ctrl_caches(&self, key: CtrlKey, program: &Program) {
        let mut codes = self.cached_ctrl_code.borrow_mut();
        if codes
            .get(&key)
            .is_some_and(|code| same_code(code, &program.code))
        {
            return;
        }
        self.cached_ctrl_buffer.borrow_mut().remove(&key);
        self.cached_ctrl_group.borrow_mut().remove(&key);
        if codes.len() >= CTRL_DEVICE_CACHE_CAP {
            codes.clear();
            self.cached_ctrl_buffer.borrow_mut().clear();
            self.cached_ctrl_group.borrow_mut().clear();
        }
        codes.insert(key, Arc::new(program.code.clone()));
    }

    fn preflight(&self, program: &Program, input: VecDeque<u32>) -> Result<Preflight> {
        scope!("preflight");

        let mut preflight = Preflight::new(input);
        for (cycle, row) in program.code_by_row().enumerate() {
            preflight.step(cycle, row)?
        }

        Ok(preflight)
    }
}
