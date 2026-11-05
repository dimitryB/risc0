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

use std::sync::LazyLock;

use risc0_core::{
    field::{Elem, ExtElem, RootsOfUnity},
    scope, scope_with,
};

use crate::{
    core::{digest::Digest, log2_ceil, poly::poly_interpolate},
    hal::{Buffer, CircuitHal, Hal},
    prove::{fri::fri_prove, merkle::MerkleTreeProver, poly_group::PolyGroup, write_iop::WriteIOP},
    taps::TapSet,
    INV_RATE,
};

static VERBOSE: LazyLock<bool> = LazyLock::new(|| std::env::var("RISC0_VERBOSE").is_ok());

/// Object to generate a zero-knowledge proof of the execution of some circuit.
pub struct Prover<'a, H: Hal> {
    hal: &'a H,
    taps: &'a TapSet<'a>,
    iop: WriteIOP<H::Field>,
    groups: Vec<Option<PolyGroup<H>>>,
    cycles: usize,
    po2: usize,
}

/// Holds the state needed to complete a deferred finalize (after eval_check
/// has been launched asynchronously on a separate GPU stream).
pub struct DeferredFinalize<H: Hal> {
    taps: &'static TapSet<'static>,
    iop: WriteIOP<H::Field>,
    groups: Vec<Option<PolyGroup<H>>>,
    cycles: usize,
    po2: usize,
    check_poly: H::Buffer<H::Elem>,
    ft0: std::time::Instant,
    /// GPU buffers that must stay alive until eval_check completes (e.g., globals
    /// passed to eval_check that are read asynchronously on a separate stream).
    keep_alive: Vec<H::Buffer<H::Elem>>,
}

fn make_coeffs<H: Hal>(hal: &H, witness: &H::Buffer<H::Elem>, count: usize) -> H::Buffer<H::Elem> {
    scope!("make_coeffs");
    let t0 = std::time::Instant::now();
    let coeffs = hal.alloc_elem("coeffs", witness.size());
    // Do interpolate + zk_shift (f(x) -> f(3x), multiplying coefficients c_i by 3^i)
    #[cfg(not(feature = "circuit_debug"))]
    hal.batch_interpolate_ntt_zk_shift_from(&coeffs, witness, count);
    #[cfg(feature = "circuit_debug")]
    {
        hal.eltwise_copy_elem(&coeffs, witness);
        hal.batch_interpolate_ntt(&coeffs, count);
    }
    if std::env::var_os("RISC0_VERBOSE").is_some() {
        eprintln!(
            "[make_coeffs] name={} count={count} size={} intt_zk={:.1}ms",
            witness.name(),
            witness.size() / count,
            t0.elapsed().as_secs_f64() * 1000.0,
        );
    }
    coeffs
}

impl<'a, H: Hal> Prover<'a, H> {
    /// Creates a new prover.
    pub fn new(hal: &'a H, taps: &'a TapSet) -> Self {
        Self {
            hal,
            taps,
            iop: WriteIOP::new(hal.get_hash_suite().rng.as_ref()),
            groups: std::iter::repeat_with(|| None)
                .take(taps.num_groups())
                .collect(),
            cycles: 0,
            po2: usize::MAX,
        }
    }

    /// Accesses the prover's IOP to commit or read random data.
    pub fn iop(&mut self) -> &mut WriteIOP<H::Field> {
        &mut self.iop
    }

    /// Sets the number of cycles to 2^po2.  This must be called
    /// once after new() before any commit_group() calls.
    pub fn set_po2(&mut self, po2: usize) {
        assert_eq!(self.po2, usize::MAX);
        assert_eq!(self.cycles, 0);
        self.po2 = po2;
        self.cycles = 1 << po2;
    }

    /// Commits a given buffer to the IOP; the values must not subsequently
    /// change.
    pub fn commit_group(&mut self, tap_group_index: usize, witness: &H::Buffer<H::Elem>) {
        scope_with!("commit_group({})", witness.name());
        let group_size = self.taps.group_size(tap_group_index);
        assert_eq!(witness.size() % group_size, 0);
        assert_eq!(witness.size() / group_size, self.cycles);
        assert!(
            self.groups[tap_group_index].is_none(),
            "Attempted to commit group {} more than once",
            self.taps.group_name(tap_group_index)
        );
        let fused = !cfg!(feature = "circuit_debug")
            && self.hal.has_commit_coeffs_lde(log2_ceil(self.cycles));
        let group = if fused {
            let coeffs = self.hal.alloc_elem("coeffs", witness.size());
            let evaluated = self
                .hal
                .alloc_elem("evaluated", group_size * self.cycles * INV_RATE);
            self.hal
                .commit_coeffs_lde(&coeffs, &evaluated, witness, group_size);
            PolyGroup::new_with_evaluated(
                self.hal,
                coeffs,
                evaluated,
                group_size,
                self.cycles,
                witness.name(),
            )
        } else {
            let coeffs = make_coeffs(self.hal, witness, group_size);
            PolyGroup::new(self.hal, coeffs, group_size, self.cycles, witness.name())
        };
        let group_ref = self.groups[tap_group_index].insert(group);

        group_ref.merkle.commit(&mut self.iop);

        // At po2>=22, spill this group's `evaluated` to host after commit.
        // It's not needed again until eval_check; freeing the device buffer
        // (3.4-13 GB depending on group) makes room for the next group's
        // allocations. `DeferredFinalize::complete` restores them before
        // circuit_hal.eval_check runs.
        if self.po2 >= 22 {
            let evaluated_bytes = group_ref
                .evaluated
                .as_ref()
                .map(|e| e.size() * std::mem::size_of::<H::Elem>())
                .unwrap_or(0);
            if evaluated_bytes >= (1 << 30) {
                // >= 1 GB
                group_ref.release_evaluated_to_host(self.hal);
            }
        }

        tracing::debug!(
            "{} group root: {}",
            self.taps.group_name(tap_group_index),
            group_ref.merkle.root()
        );
    }

    /// Returns a reference to the PolyGroup for the given tap group, if committed.
    pub fn get_group(&self, tap_group_index: usize) -> Option<&PolyGroup<H>> {
        self.groups[tap_group_index].as_ref()
    }

    /// Injects a pre-built PolyGroup (e.g., cached from a previous segment) and
    /// commits its merkle root to the IOP. Skips iNTT/expand/hash recomputation.
    pub fn commit_cached_group(&mut self, tap_group_index: usize, group: PolyGroup<H>) {
        assert!(
            self.groups[tap_group_index].is_none(),
            "Attempted to commit group {} more than once",
            self.taps.group_name(tap_group_index)
        );

        let group_ref = self.groups[tap_group_index].insert(group);
        group_ref.merkle.commit(&mut self.iop);
    }

    /// Generates the proof and returns the seal.
    pub fn finalize<C>(self, globals: &[&H::Buffer<H::Elem>], circuit_hal: &C) -> Vec<u32>
    where
        C: CircuitHal<H>,
        H: 'static,
    {
        self.finalize_with_hook(globals, circuit_hal, || {})
    }

    /// Generates the proof and returns the seal, calling `post_eval_check` after
    /// launching eval_check but before needing its results. This allows CPU work
    /// (e.g., freeing buffers) to overlap with GPU eval_check computation.
    pub fn finalize_with_hook<C, F>(
        self,
        globals: &[&H::Buffer<H::Elem>],
        circuit_hal: &C,
        post_eval_check: F,
    ) -> Vec<u32>
    where
        C: CircuitHal<H>,
        F: FnOnce(),
        H: 'static,
    {
        let hal = self.hal;
        let deferred = self.start_finalize_with_hook(globals, circuit_hal, post_eval_check);
        deferred.complete(hal, circuit_hal)
    }

    /// Launch eval_check asynchronously and return a DeferredFinalize to complete. Work the
    /// caller queues before `complete` can overlap eval_check, which runs on a separate stream;
    /// `complete` waits for it (on the host) before it frees what eval_check reads.
    ///
    /// NOTE: `taps` must be 'static for DeferredFinalize to be stored across calls.
    pub fn start_finalize_with_hook<C, F>(
        mut self,
        globals: &[&H::Buffer<H::Elem>],
        circuit_hal: &C,
        post_eval_check: F,
    ) -> DeferredFinalize<H>
    where
        C: CircuitHal<H>,
        F: FnOnce(),
        H: 'static,
    {
        let ft0 = std::time::Instant::now();

        // Set the poly mix value, which is used for constraint compression in the
        // DEEP-ALI protocol.
        let poly_mix = self.iop.random_ext_elem();
        let domain = self.cycles * INV_RATE;
        let ext_size = H::ExtElem::EXT_SIZE;

        // Restore any host-spilled `evaluated` buffers before eval_check.
        // At po2>=22, commit_group moved these off-device to free VRAM for
        // subsequent groups' allocations; we bring them back for the kernel
        // to read.
        for pg in self.groups.iter_mut() {
            if let Some(pg) = pg.as_mut() {
                if pg.evaluated.is_none() && pg.evaluated_host.is_some() {
                    pg.restore_evaluated_from_host(self.hal);
                }
            }
        }

        // Now generate the check polynomial.
        let _ft_alloc = std::time::Instant::now();
        let check_poly = self.hal.alloc_elem("check_poly", ext_size * domain);
        let _t_alloc = _ft_alloc.elapsed();

        let groups: Vec<&_> = self
            .groups
            .iter()
            .map(|pg| pg.as_ref().unwrap().evaluated.as_ref().unwrap())
            .collect();
        let _ft1 = std::time::Instant::now();
        circuit_hal.eval_check(
            &check_poly,
            groups.as_slice(),
            globals,
            poly_mix,
            self.po2,
            self.cycles,
        );
        let _t_eval_check = _ft1.elapsed();
        // eval_check is async on GPU — run CPU work while GPU computes
        let _ft_drop = std::time::Instant::now();
        post_eval_check();
        let _t_drop = _ft_drop.elapsed();
        if std::env::var_os("RISC0_VERBOSE").is_some() {
            eprintln!(
                "[start_finalize] alloc={:.1}ms eval_check_launch={:.1}ms post_eval_check={:.1}ms total={:.1}ms",
                _t_alloc.as_secs_f64() * 1000.0,
                _t_eval_check.as_secs_f64() * 1000.0,
                _t_drop.as_secs_f64() * 1000.0,
                ft0.elapsed().as_secs_f64() * 1000.0,
            );
        }

        // SAFETY: taps is always &'static TAPSET in practice.
        let taps: &'static TapSet<'static> =
            unsafe { std::mem::transmute::<&'_ TapSet<'_>, &'static TapSet<'static>>(self.taps) };

        DeferredFinalize {
            taps,
            iop: self.iop,
            groups: self.groups,
            cycles: self.cycles,
            po2: self.po2,
            check_poly,
            ft0,
            keep_alive: Vec::new(),
        }
    }
}

impl<H: Hal> DeferredFinalize<H> {
    /// Add a GPU buffer that must stay alive until eval_check completes.
    /// Used to prevent freeing buffers that eval_check reads asynchronously.
    pub fn keep_alive_buf(&mut self, buf: H::Buffer<H::Elem>) {
        self.keep_alive.push(buf);
    }

    /// Complete the deferred finalize: wait for eval_check, do iNTT, and
    /// produce the final proof seal.
    pub fn complete<C: CircuitHal<H>>(mut self, hal: &H, circuit_hal: &C) -> Vec<u32> {
        scope!("finalize_complete");
        let _ft0 = self.ft0;
        let ext_size = H::ExtElem::EXT_SIZE;
        let ft = std::time::Instant::now();

        // Insert GPU dependency: make persistent stream wait for eval_check stream.
        circuit_hal.eval_check_dep();
        let t_eval_check_dep = ft.elapsed();

        // On tight-VRAM GPUs, release evaluated buffers (both PolyGroup.evaluated
        // AND merkle.matrix) to free memory for check PolyGroup + FRI allocations.
        // Evaluated will be reconstructed from coefficients before FRI batch_prove.
        // po2>=22 is always low_mem regardless of VRAM (peak > 32 GB even on 24 GB cards);
        // po2==21 is low_mem only on <20 GB cards.
        let low_mem = self.po2 >= 22
            || (hal.gpu_total_memory() < 20 * 1024 * 1024 * 1024 && self.po2 >= 21);
        if low_mem {
            for pg in self.groups.iter_mut() {
                if let Some(pg) = pg.as_mut() {
                    pg.release_evaluated();
                }
            }
        }

        // Restore coeffs to device for all groups. Any group that had its
        // coeffs spilled to host (via release_coeffs) at commit time to save
        // VRAM during merkle/eval_check is reloaded here before the eval_u
        // and mix_poly_coeffs phases below.
        for pg in self.groups.iter_mut() {
            if let Some(pg) = pg.as_mut() {
                pg.restore_coeffs(hal);
            }
        }

        // Convert to coefficients.
        hal.batch_interpolate_ntt(&self.check_poly, ext_size);
        let t_check_intt = ft.elapsed();

        // The next step is to convert the degree 4*n check polynomial into 4 degree n
        // polynomials so that f(x) = g0(x^4) + g1(x^4) x + g2(x^4) x^2 + g3(x^4)
        // x^3.  To do this, we normally would grab all the coefficients of f(x) =
        // sum_i c_i x^i where i % 4 == 0 and put them into a new polynomial g0(x) =
        // sum_i d0_i*x^i, where d0_i = c_(i*4).
        //
        // Amazingly, since the coefficients are bit reversed, the coefficients of g0
        // are all already next to each other and in bit-reversed for g0, as are
        // the coefficients of g1, etc. So really, we can just reinterpret 4 polys of
        // invRate*size to 16 polys of size, without actually doing anything.

        // Make the PolyGroup + add it to the IOP;
        let check_group = PolyGroup::new(hal, self.check_poly, H::CHECK_SIZE, self.cycles, "check");
        check_group.merkle.commit(&mut self.iop);
        tracing::debug!("checkGroup: {}", check_group.merkle.root());
        let t_check_commit = ft.elapsed();

        // Now pick a value for Z, which is used as the DEEP-ALI query point.
        cfg_if::cfg_if! {
            if #[cfg(feature = "circuit_debug")] {
                let z = if let Some(bad_z) = bad_z {
                    self.iop.write_field_elem_slice(bad_z.subelems());
                    bad_z
                } else {
                    self.iop.random_ext_elem()
                };
            } else {
                let z = self.iop.random_ext_elem();
            }
        }
        tracing::debug!("Z = {z:?}");

        // Get rev rou for size
        let back_one = H::ExtElem::from_subfield(&H::Elem::ROU_REV[self.po2]);
        let mut all_xs = Vec::new();

        // Now, we evaluate each group at the appropriate points (relative to Z).
        // From here on out, we always process groups in accum, code, data order,
        // since this is the order used by the codegen system (alphabetical).
        // Sometimes it's a requirement for matching generated code, but even when
        // it's not we keep the order for consistency.

        let _eu_verbose = std::env::var_os("RISC0_VERBOSE").is_some();
        let _eu_t0 = std::time::Instant::now();
        let mut eval_u: Vec<H::ExtElem> = Vec::new();
        scope!("eval_u", {
            // Calculate total taps to allocate a single combined output buffer.
            let total_taps: usize = self
                .groups
                .iter()
                .enumerate()
                .map(|(id, _)| self.taps.group_taps(id).count())
                .sum();
            let combined_out = hal.alloc_extelem("eval_u_combined", total_taps);

            // Pre-compute all which/xs values and upload as single combined buffers.
            let mut all_which_flat = Vec::with_capacity(total_taps);
            let mut all_xs_flat = Vec::with_capacity(total_taps);
            for (id, _) in self.groups.iter().enumerate() {
                for tap in self.taps.group_taps(id) {
                    all_which_flat.push(tap.offset() as u32);
                    let x = back_one.pow(tap.back()) * z;
                    all_xs_flat.push(x);
                    all_xs.push(x);
                }
            }
            let which_buf = hal.copy_from_u32("which", &all_which_flat);
            let xs_buf = hal.copy_from_extelem("xs", &all_xs_flat);

            hal.with_gpu_stage("eval_u", || {
                let mut offset = 0;
                for (id, pg) in self.groups.iter().enumerate() {
                    let pg = pg.as_ref().unwrap();
                    let group_size = self.taps.group_taps(id).count();
                    hal.batch_evaluate_any(
                        pg.coeffs.as_ref().expect("coeffs must be restored"),
                        pg.count,
                        &which_buf.slice(offset, group_size),
                        &xs_buf.slice(offset, group_size),
                        &combined_out.slice(offset, group_size),
                    );
                    offset += group_size;
                }
            });

            // Single D2H transfer instead of one per group.
            combined_out.view(|view| {
                eval_u.extend_from_slice(view);
            });
        });
        let _eu_t_be = _eu_t0.elapsed();

        // The check group's evaluation at z^EXT_SIZE does not depend on the interpolation below:
        // queue it first (async on GPU HALs), so it runs while the host interpolates (ROCm builds;
        // RISC0_EVAL_U_EARLY=0 keeps the old order).
        let z_pow = z.pow(ext_size);
        let check_eval = |hal: &H| {
            let which = Vec::from_iter(0u32..H::CHECK_SIZE as u32);
            let xs = vec![z_pow; H::CHECK_SIZE];
            let out = hal.alloc_extelem("out", H::CHECK_SIZE);
            let which = hal.copy_from_u32("which", which.as_slice());
            let xs = hal.copy_from_extelem("xs", xs.as_slice());
            hal.batch_evaluate_any(
                check_group.coeffs.as_ref().expect("coeffs must be restored"),
                H::CHECK_SIZE,
                &which,
                &xs,
                &out,
            );
            out
        };
        static EVAL_U_EARLY: LazyLock<bool> = LazyLock::new(|| {
            cfg!(feature = "rocm")
                && !matches!(std::env::var("RISC0_EVAL_U_EARLY").as_deref(), Ok(v) if v.starts_with('0'))
        });
        let early_check = EVAL_U_EARLY.then(|| check_eval(hal));
        // The FRI mixing buffer's zero fill does not depend on the mix either: queue it now too.
        let combo_count = self.taps.combos_size();
        let early_combos = EVAL_U_EARLY.then(|| {
            hal.alloc_extelem_zeroed("combos", self.cycles * (combo_count + 1))
        });

        // Now, convert the values to coefficients via interpolation
        let mut coeff_u = vec![H::ExtElem::ZERO; eval_u.len()];
        let _eu_t_interp_start = std::time::Instant::now();
        scope!("poly_interpolate", {
            let mut pos = 0;
            for reg in self.taps.regs() {
                poly_interpolate(
                    &mut coeff_u[pos..],
                    &all_xs[pos..],
                    &eval_u[pos..],
                    reg.size(),
                );
                pos += reg.size();
            }
        });
        let _eu_t_interp = _eu_t_interp_start.elapsed();

        // Add in the coeffs of the check polynomials.
        let _eu_t_misc_start = std::time::Instant::now();
        let mut _eu_t_be2 = std::time::Duration::ZERO;
        let mut _eu_t_hash = std::time::Duration::ZERO;
        let mut _eu_t_iop_w = std::time::Duration::ZERO;
        let mut _eu_t_iop_c = std::time::Duration::ZERO;
        scope!("misc", {
            let _bt = std::time::Instant::now();
            let out = match early_check {
                Some(out) => out,
                None => check_eval(hal),
            };
            out.view(|view| {
                coeff_u.extend(view);
            });
            _eu_t_be2 = _bt.elapsed();

            tracing::debug!("Size of U = {}", coeff_u.len());
            let _wt = std::time::Instant::now();
            self.iop.write_field_elem_slice(&coeff_u);
            _eu_t_iop_w = _wt.elapsed();
            let _ht = std::time::Instant::now();
            let hash_u = hal
                .get_hash_suite()
                .hashfn
                .hash_ext_elem_slice(coeff_u.as_slice());
            _eu_t_hash = _ht.elapsed();
            let _ct = std::time::Instant::now();
            self.iop.commit(&hash_u);
            _eu_t_iop_c = _ct.elapsed();
            // Set the mix value, which is used for FRI batching.
        });
        let _eu_t_misc = _eu_t_misc_start.elapsed();
        if _eu_verbose {
            eprintln!(
                "      [eval_u_decomp] eval_u_block={:.1}ms interp={:.1}ms misc(total)={:.1}ms (be2={:.1}ms iop_w={:.1}ms hash_u={:.1}ms iop_c={:.1}ms)",
                _eu_t_be.as_secs_f64() * 1000.0,
                _eu_t_interp.as_secs_f64() * 1000.0,
                _eu_t_misc.as_secs_f64() * 1000.0,
                _eu_t_be2.as_secs_f64() * 1000.0,
                _eu_t_iop_w.as_secs_f64() * 1000.0,
                _eu_t_hash.as_secs_f64() * 1000.0,
                _eu_t_iop_c.as_secs_f64() * 1000.0,
            );
        }

        let t_eval_u = ft.elapsed();

        let mix = self.iop.random_ext_elem();
        tracing::debug!("Mix = {mix:?}");

        // Do the coefficient mixing
        // Begin by making a zeroed output buffer
        let combos = scope!(
            "alloc(combos)",
            match early_combos {
                Some(combos) => combos,
                None => hal.alloc_extelem_zeroed("combos", self.cycles * (combo_count + 1)),
            }
        );
        scope!("mix_poly_coeffs", {
            let mut cur_mix = H::ExtElem::ONE;

            // Pre-compute all which arrays (combo_ids) for groups + check, upload once.
            let total_group_regs: usize = self
                .groups
                .iter()
                .enumerate()
                .map(|(id, _)| self.taps.group_size(id))
                .sum();
            let mut all_mix_which = Vec::with_capacity(total_group_regs + H::CHECK_SIZE);
            for (id, _) in self.groups.iter().enumerate() {
                for reg in self.taps.group_regs(id) {
                    all_mix_which.push(reg.combo_id() as u32);
                }
            }
            let check_which_offset = all_mix_which.len();
            for _ in 0..H::CHECK_SIZE {
                all_mix_which.push(combo_count as u32);
            }
            let mix_which_buf = hal.copy_from_u32("which", &all_mix_which);

            hal.with_gpu_stage("mix_poly", || {
                let mut which_offset = 0;
                for (id, pg) in self.groups.iter().enumerate() {
                    let pg = pg.as_ref().unwrap();
                    let group_size = self.taps.group_size(id);
                    hal.mix_poly_coeffs(
                        &combos,
                        &cur_mix,
                        &mix,
                        pg.coeffs.as_ref().expect("coeffs must be restored"),
                        &mix_which_buf.slice(which_offset, group_size),
                        group_size,
                        self.cycles,
                    );
                    cur_mix *= mix.pow(group_size);
                    which_offset += group_size;
                }

                hal.mix_poly_coeffs(
                    &combos,
                    &cur_mix,
                    &mix,
                    check_group.coeffs.as_ref().expect("coeffs must be restored"),
                    &mix_which_buf.slice(check_which_offset, H::CHECK_SIZE),
                    H::CHECK_SIZE,
                    self.cycles,
                );
            });
        });

        scope!("load_combos", {
            let reg_sizes: Vec<_> = self.taps.regs().map(|x| x.size() as u32).collect();
            let reg_combo_ids: Vec<_> = self.taps.regs().map(|x| x.combo_id() as u32).collect();

            scope!("prepare", {
                hal.combos_prepare(
                    &combos,
                    &coeff_u,
                    combo_count,
                    self.cycles,
                    &reg_sizes,
                    &reg_combo_ids,
                    &mix,
                );
            });

            scope!("divide", {
                let mut chunks = vec![];

                // Divide each element by (x - Z * back1^back) for each back
                for i in 0..combo_count {
                    let mut pows = vec![];
                    for &back in self.taps.get_combo(i).slice() {
                        pows.push(z * back_one.pow(back.into()));
                    }
                    chunks.push((i, pows));
                }

                // Divide check polys by z^EXT_SIZE
                chunks.push((combo_count, vec![z_pow]));

                hal.combos_divide(&combos, chunks, self.cycles);
            });
        });

        let t_combos = ft.elapsed();

        // Sum the combos up into one final polynomial + make it into 4 Fp polys.
        // Additionally, it needs to be bit reversed to make everyone happy
        let final_poly_coeffs = scope!("sum", {
            let final_poly_coeffs = hal.alloc_elem("final_poly_coeffs", self.cycles * ext_size);
            hal.eltwise_sum_extelem(&final_poly_coeffs, &combos);
            final_poly_coeffs
        });

        // Finally do the FRI protocol to prove the degree of the polynomial
        scope!(
            "bit_rev",
            hal.batch_bit_reverse(&final_poly_coeffs, ext_size)
        );
        tracing::debug!("FRI-proof, size = {}", final_poly_coeffs.size() / ext_size);
        let t_pre_fri = ft.elapsed();

        if low_mem {
            // Low-memory path: reconstruct evaluated on-demand for each group
            // during FRI batch_prove, then release immediately after.
            let groups_cell = core::cell::RefCell::new(std::mem::take(&mut self.groups));
            fri_prove(
                hal,
                &mut self.iop,
                &final_poly_coeffs,
                |indices: &[usize]| -> Vec<Vec<(Vec<H::Elem>, Vec<Digest>)>> {
                    let mut groups = groups_cell.borrow_mut();
                    let mut trees: Vec<Vec<(Vec<H::Elem>, Vec<Digest>)>> = Vec::new();
                    for pg in groups.iter_mut() {
                        let pg = pg.as_mut().unwrap();
                        pg.restore_evaluated(hal);
                        trees.push(pg.merkle.batch_prove(hal, indices));
                        pg.release_evaluated();
                    }
                    trees.push(check_group.merkle.batch_prove(hal, indices));
                    let n = indices.len();
                    let num_trees = trees.len();
                    let mut result: Vec<Vec<(Vec<H::Elem>, Vec<Digest>)>> = Vec::with_capacity(n);
                    for q in 0..n {
                        let mut query_data = Vec::with_capacity(num_trees);
                        for tree in trees.iter_mut() {
                            query_data.push(std::mem::take(&mut tree[q]));
                        }
                        result.push(query_data);
                    }
                    result
                },
            );
        } else {
            // Normal path: evaluated buffers still in memory
            fri_prove(
                hal,
                &mut self.iop,
                &final_poly_coeffs,
                |indices: &[usize]| -> Vec<Vec<(Vec<H::Elem>, Vec<Digest>)>> {
                    let mut jobs: Vec<(&MerkleTreeProver<H>, &[usize])> = self
                        .groups
                        .iter()
                        .map(|pg| (&pg.as_ref().unwrap().merkle, indices))
                        .collect();
                    jobs.push((&check_group.merkle, indices));
                    let mut trees: Vec<Vec<(Vec<H::Elem>, Vec<Digest>)>> =
                        MerkleTreeProver::batch_prove_many(hal, &jobs);
                    let n = indices.len();
                    let num_trees = trees.len();
                    let mut result: Vec<Vec<(Vec<H::Elem>, Vec<Digest>)>> = Vec::with_capacity(n);
                    for q in 0..n {
                        let mut query_data = Vec::with_capacity(num_trees);
                        for tree in trees.iter_mut() {
                            query_data.push(std::mem::take(&mut tree[q]));
                        }
                        result.push(query_data);
                    }
                    result
                },
            );
        }

        let proven_soundness_error =
            super::soundness::proven::<H>(self.taps, final_poly_coeffs.size());
        tracing::debug!("proven_soundness_error: {proven_soundness_error:?}");

        let conjectured_security =
            super::soundness::toy_model_security::<H>(self.taps, final_poly_coeffs.size());
        tracing::debug!("conjectured_security: {conjectured_security:?}");

        let t_total = ft.elapsed();
        if *VERBOSE {
            eprintln!(
                "[finalize] eval_check_dep={:.1}ms check_intt={:.1}ms check_commit={:.1}ms eval_u={:.1}ms combos={:.1}ms fri={:.1}ms total={:.1}ms",
                t_eval_check_dep.as_secs_f64() * 1000.0,
                (t_check_intt - t_eval_check_dep).as_secs_f64() * 1000.0,
                (t_check_commit - t_check_intt).as_secs_f64() * 1000.0,
                (t_eval_u - t_check_commit).as_secs_f64() * 1000.0,
                (t_combos - t_eval_u).as_secs_f64() * 1000.0,
                (t_total - t_pre_fri).as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
            );
        }

        // Return final proof
        let proof = self.iop.proof;
        tracing::debug!("Proof size = {}", proof.len());
        proof
    }
}
