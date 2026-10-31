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

use std::sync::Arc;

use anyhow::Result;
use parking_lot::Mutex;
// CPU accum FFI used as fallback while GPU accum is being debugged
use risc0_core::scope;
use risc0_zkp::{
    core::log2_ceil,
    field::{map_pow, Elem as _, RootsOfUnity as _},
    hal::{
        intel::{
            BufferImpl as IntelBuffer, IntelHal, IntelHalPoseidon2, IntelHash, IntelHashPoseidon2,
        },
        AccumPreflight, CircuitHal, Hal,
    },
    INV_RATE,
};

use super::{
    CircuitAccumulator, CircuitWitnessGenerator, MetaBuffer, SegmentProver, SegmentProverImpl,
    StepMode,
};
use crate::{
    prove::{witgen::preflight::PreflightTrace, GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{ExtVal, Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_DATA},
        info::POLY_MIX_POWERS,
    },
};

/// Returns true when `phase` is listed in the comma-separated
/// `RISC0_PHASE_DISABLE` env var. Used to disable specific optimization
/// phases for failure-bisection without recompiling. Phase names match
/// the convention in EVAL_CHECK_PLAN_V5 (e.g. `multipass`, `block_read`,
/// `tree_decomp`, `subgroup_partition`).
fn phase_disabled(phase: &str) -> bool {
    // Trim and reject empty queries symmetrically with how CSV entries
    // are normalized. Without this, `phase_disabled(" multipass ")`
    // would silently fail to match `RISC0_PHASE_DISABLE=multipass`, and
    // `phase_disabled("")` would match an empty CSV token from
    // `RISC0_PHASE_DISABLE=",,"`.
    let phase = phase.trim();
    if phase.is_empty() {
        return false;
    }
    std::env::var("RISC0_PHASE_DISABLE")
        .ok()
        .map(|v| {
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .any(|s| s.eq_ignore_ascii_case(phase))
        })
        .unwrap_or(false)
}

// The Intel witgen/accum FFI returns a pointer into a thread_local std::string
// it owns. `ffi_wrap` would free() it, and the thread_local destructor frees it
// again at thread exit (glibc "double free detected in tcache"), so copy only.
fn borrowed_err_wrap(inner: impl FnOnce() -> *const std::os::raw::c_char) -> Result<()> {
    let c_ptr = inner();
    if c_ptr.is_null() {
        return Ok(());
    }
    let msg = unsafe { std::ffi::CStr::from_ptr(c_ptr) }
        .to_string_lossy()
        .into_owned();
    Err(anyhow::anyhow!(msg))
}

pub struct IntelCircuitHal<IH: IntelHash> {
    _hal: Arc<IntelHal<IH>>,
    // Keep eval_check inputs alive until eval_check_dep().
    // Arc+Mutex (instead of Rc+RefCell) so IntelCircuitHal is Send — required
    // for the finalize-overlap path that moves DeferredFinalize into the
    // receipt background thread.
    eval_check_poly_mix: Mutex<Option<IntelBuffer<u32>>>,
    // Host source of the poly_mix upload: esimd_memcpy_htod does not wait, so
    // the words must outlive the queued copy. Kept until the next eval_check
    // replaces them (several host syncs later).
    eval_check_poly_mix_host: Mutex<Option<Vec<u32>>>,
    eval_check_inter_fp: Mutex<Option<IntelBuffer<u32>>>,
    eval_check_inter_ext: Mutex<Option<IntelBuffer<u32>>>,
}

impl<IH: IntelHash> IntelCircuitHal<IH> {
    pub fn new(_hal: Arc<IntelHal<IH>>) -> Self {
        Self {
            _hal,
            eval_check_poly_mix: Mutex::new(None),
            eval_check_poly_mix_host: Mutex::new(None),
            eval_check_inter_fp: Mutex::new(None),
            eval_check_inter_ext: Mutex::new(None),
        }
    }
}

impl<IH: IntelHash> CircuitWitnessGenerator<IntelHal<IH>> for IntelCircuitHal<IH> {
    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<IntelHal<IH>>,
        data: &MetaBuffer<IntelHal<IH>>,
        pre_data: &MetaBuffer<IntelHal<IH>>,
    ) -> Result<()> {
        scope!("intel_witgen");
        let cycles = preflight.cycles.len();
        tracing::debug!("witgen: {cycles} cycles (GPU)");

        let queue = risc0_sys::intel::get_queue();
        borrowed_err_wrap(|| unsafe {
            risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_witgen(
                queue,
                mode as u32,
                data.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                data.rows as u32,
                data.cols as u32,
                pre_data.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                global.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                global.cols as u32,
                preflight.cycles.as_ptr(),
                preflight.cycles.len() as u32,
                preflight.txns.as_ptr(),
                preflight.txns.len() as u32,
                preflight.bigint_bytes.as_ptr(),
                preflight.bigint_bytes.len() as u32,
                preflight.table_split_cycle,
                cycles as u32,
            )
        })?;

        Ok(())
    }
}

impl<IH: IntelHash> CircuitAccumulator<IntelHal<IH>> for IntelCircuitHal<IH> {
    fn step_accum(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<IntelHal<IH>>,
        accum: &MetaBuffer<IntelHal<IH>>,
        global: &MetaBuffer<IntelHal<IH>>,
        mix: &MetaBuffer<IntelHal<IH>>,
    ) -> Result<()> {
        scope!("intel_accumulate");
        let cycles = preflight.cycles.len();

        // GPU accum compiled at -Os (testing if different opts avoid icpx -O1 miscompilation)
        tracing::debug!("accumulate: {cycles} cycles (GPU, -Os)");
        let queue = risc0_sys::intel::get_queue();
        borrowed_err_wrap(|| unsafe {
            risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_accum(
                queue,
                data.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                data.rows as u32,
                data.cols as u32,
                accum.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                accum.rows as u32,
                accum.cols as u32,
                global.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                global.cols as u32,
                mix.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                mix.cols as u32,
                preflight.cycles.as_ptr(),
                preflight.cycles.len() as u32,
                preflight.txns.as_ptr(),
                preflight.txns.len() as u32,
                preflight.bigint_bytes.as_ptr(),
                preflight.bigint_bytes.len() as u32,
                preflight.table_split_cycle,
                cycles as u32,
            )
        })?;

        Ok(())
    }
}

impl<IH: IntelHash> CircuitHal<IntelHal<IH>> for IntelCircuitHal<IH> {
    fn eval_check(
        &self,
        check: &IntelBuffer<Val>,
        groups: &[&IntelBuffer<Val>],
        globals: &[&IntelBuffer<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) {
        scope!("eval_check");

        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);
        // eval_check.cpp hardcodes this count (POLY_MIX_COUNT) and, in the
        // RISC0_INTEL_EVAL_FAST build, reads the quotient table right after it.
        assert_eq!(poly_mix_pows.len(), 458);

        let rou = Val::ROU_FWD[po2 + EXP_PO2];

        // Vanishing quotient table: the kernel's per-row
        // 1 / ((3 * rou^cycle)^(2^po2) - 1) equals qtab[cycle % 4] with
        // qtab[k] = 1 / (3^(2^po2) * w4^k - 1), w4 = rou^(2^po2) a primitive
        // 4th root of unity. The fast kernel reads it; the other ignores it.
        let y0 = Val::new(3).pow(1 << po2);
        let w4 = rou.pow(1 << po2);
        let qtab: Vec<Val> = (0..INV_RATE)
            .map(|k| (y0 * w4.pow(k) - Val::ONE).inv())
            .collect();

        // Upload poly_mix_pows (+ qtab) to GPU. Stored in the struct to keep
        // it alive until eval_check_dep(). Both are in device (Montgomery) form.
        let mut poly_mix_words: Vec<u32> = unsafe {
            std::slice::from_raw_parts(
                poly_mix_pows.as_ptr() as *const u32,
                poly_mix_pows.len() * 4,
            )
        }
        .to_vec();
        poly_mix_words.extend(qtab.iter().map(|q| q.as_u32_montgomery()));
        let poly_mix_buf: IntelBuffer<u32> = IntelBuffer::copy_from("poly_mix", &poly_mix_words);
        *self.eval_check_poly_mix_host.lock() = Some(poly_mix_words);

        let rou_raw: u32 = unsafe { std::mem::transmute(rou) };

        let queue = risc0_sys::intel::get_queue();

        // Phase 0b: respect RISC0_PHASE_DISABLE for failure-bisection. When
        // `multipass` is listed there, fall back to the monolithic kernel
        // even if RISC0_MULTIPASS is set — useful for narrowing whether a
        // miscompare is multipass-specific.
        let use_multipass =
            std::env::var_os("RISC0_MULTIPASS").is_some() && !phase_disabled("multipass");

        if use_multipass {
            // 2-way multi-pass: allocate intermediate buffer, run pass1 then pass2
            // Intermediate: 197 Fp (u32) + 9 FpExt (7 by-val + 2 x34 cross-boundary)
            let n_inter_fp = 197 * domain;
            let n_inter_ext = 9 * domain * 4;
            let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
            let t0 = if verbose {
                Some(std::time::Instant::now())
            } else {
                None
            };

            let inter_fp = self._hal.alloc_u32("inter_fp", n_inter_fp);
            let inter_ext = self._hal.alloc_u32("inter_ext", n_inter_ext);

            // Pass 1: upper chain → writes intermediate
            risc0_sys::intel::esimd_check(unsafe {
                risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_eval_check_pass1(
                    queue,
                    inter_fp.as_device_ptr().0 as *mut std::ffi::c_void,
                    inter_ext.as_device_ptr().0 as *mut std::ffi::c_void,
                    groups[REGISTER_GROUP_DATA].as_device_ptr().0 as *const std::ffi::c_void,
                    groups[REGISTER_GROUP_ACCUM].as_device_ptr().0 as *const std::ffi::c_void,
                    globals[GLOBAL_OUT].as_device_ptr().0 as *const std::ffi::c_void,
                    globals[GLOBAL_MIX].as_device_ptr().0 as *const std::ffi::c_void,
                    poly_mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
                    domain as u32,
                )
            });

            if verbose {
                // Sync to measure pass1 time
                risc0_sys::intel::esimd_check(unsafe {
                    risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_eval_check_sync(queue)
                });
                let t1 = std::time::Instant::now();
                eprintln!(
                    "      [eval_check_pass1] {:.1}ms",
                    t1.duration_since(t0.unwrap()).as_secs_f64() * 1000.0
                );
            }

            let t1 = if verbose {
                Some(std::time::Instant::now())
            } else {
                None
            };

            // Pass 2: reads intermediate → writes check buffer
            risc0_sys::intel::esimd_check(unsafe {
                risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_eval_check_pass2(
                    queue,
                    check.as_device_ptr().0 as *mut std::ffi::c_void,
                    inter_fp.as_device_ptr().0 as *const std::ffi::c_void,
                    inter_ext.as_device_ptr().0 as *const std::ffi::c_void,
                    groups[REGISTER_GROUP_DATA].as_device_ptr().0 as *const std::ffi::c_void,
                    groups[REGISTER_GROUP_ACCUM].as_device_ptr().0 as *const std::ffi::c_void,
                    globals[GLOBAL_OUT].as_device_ptr().0 as *const std::ffi::c_void,
                    globals[GLOBAL_MIX].as_device_ptr().0 as *const std::ffi::c_void,
                    poly_mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
                    rou_raw,
                    po2 as u32,
                    domain as u32,
                )
            });

            if verbose {
                risc0_sys::intel::esimd_check(unsafe {
                    risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_eval_check_sync(queue)
                });
                let t2 = std::time::Instant::now();
                eprintln!(
                    "      [eval_check_pass2] {:.1}ms",
                    t2.duration_since(t1.unwrap()).as_secs_f64() * 1000.0
                );
            }

            // Keep intermediate buffers alive until eval_check_dep()
            *self.eval_check_inter_fp.lock() = Some(inter_fp);
            *self.eval_check_inter_ext.lock() = Some(inter_ext);
        } else {
            // Monolithic: single kernel call
            risc0_sys::intel::esimd_check(unsafe {
                risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_eval_check(
                    queue,
                    check.as_device_ptr().0 as *mut std::ffi::c_void,
                    groups[REGISTER_GROUP_DATA].as_device_ptr().0 as *const std::ffi::c_void,
                    groups[REGISTER_GROUP_ACCUM].as_device_ptr().0 as *const std::ffi::c_void,
                    globals[GLOBAL_OUT].as_device_ptr().0 as *const std::ffi::c_void,
                    globals[GLOBAL_MIX].as_device_ptr().0 as *const std::ffi::c_void,
                    poly_mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
                    rou_raw,
                    po2 as u32,
                    domain as u32,
                )
            });
        }

        // Keep poly_mix_buf alive until eval_check_dep() or next eval_check()
        *self.eval_check_poly_mix.lock() = Some(poly_mix_buf);
        shadow_eval_check(check, groups, globals, poly_mix, po2, steps);
    }

    fn eval_check_dep(&self) {
        // eval_check ran on the single in-order queue, so any later reuse of
        // these buffers is ordered after it; they can go back to the pool now.
        *self.eval_check_poly_mix.lock() = None;
        *self.eval_check_inter_fp.lock() = None;
        *self.eval_check_inter_ext.lock() = None;
    }

    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &IntelBuffer<Val>,
        _global: &IntelBuffer<Val>,
        _data: &IntelBuffer<Val>,
        _mix: &IntelBuffer<Val>,
        _accum: &IntelBuffer<Val>,
        _steps: usize,
    ) {
        unimplemented!()
    }
}

pub type IntelCircuitHalPoseidon2 = IntelCircuitHal<IntelHashPoseidon2>;

pub fn segment_prover() -> Result<Box<dyn SegmentProver>> {
    let hal_factory = || {
        let hal = Arc::new(IntelHalPoseidon2::new());
        let circuit_hal = Arc::new(IntelCircuitHalPoseidon2::new(hal.clone()));
        (hal, circuit_hal)
    };
    Ok(Box::new(SegmentProverImpl::new(hal_factory)))
}

// ===========================================================================
// Phase 0b — bit-exact test harness for Intel SYCL eval_check.
//
// Mirror of `cuda.rs::tests::eval_check` (cuda.rs:329-366) plus directed
// boundary cases covering BabyBear edge values (0, p-1) and one-hot column
// inputs. Mono and multipass paths are both exercised and required to
// produce byte-identical outputs to the CPU reference.
//
// Optional regression mode: with `RISC0_PHASE_0B_GENERATE_GOLDEN=1`, tests
// write golden `.bin` files for the PO2=4 mono/multipass outputs; without
// the env var, tests load the goldens (when present) and assert byte-for-
// byte equality. This pins behavior so future kernel rewrites must
// explicitly regenerate goldens.
//
// `RISC0_PHASE_DISABLE` env var is wired into the multipass selection
// above so failure-bisection can skip individual optimization phases as
// they land in Phase 1+.
// ===========================================================================
#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::path::PathBuf;
    use std::sync::{Arc, OnceLock};

    use parking_lot::{Mutex, MutexGuard};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use risc0_core::field::{baby_bear::BabyBear, Elem, ExtElem};
    use risc0_zkp::{
        adapter::CircuitInfo as _,
        core::hash::sha::Sha256HashSuite,
        hal::{cpu::CpuHal, intel::IntelHalSha256, Buffer, CircuitHal, Hal},
    };
    use test_log::test;

    use super::{IntelCircuitHal, INV_RATE};
    use crate::{
        prove::hal::cpu::CpuCircuitHal,
        zirgen::{
            circuit::{
                ExtVal, Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE, REGISTER_GROUP_DATA,
            },
            taps::TAPSET,
            CircuitImpl,
        },
    };

    /// Process-global lock acquired by every test that mutates env vars.
    /// `cargo test` runs in parallel by default and `libc::setenv` is not
    /// thread-safe relative to concurrent `getenv` reads, so without this
    /// guard six of the seven Phase 0b tests race on RISC0_MULTIPASS and
    /// RISC0_PHASE_DISABLE.
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    fn env_lock() -> &'static Mutex<()> {
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    /// RAII guard that captures a set of env vars on construction and
    /// restores their original values on Drop (even on panic). Holds the
    /// global ENV_LOCK for its lifetime so env-mutating tests serialize.
    struct EnvGuard {
        saved: Vec<(&'static str, Option<OsString>)>,
        _guard: MutexGuard<'static, ()>,
    }

    impl EnvGuard {
        fn new(keys: &[&'static str]) -> Self {
            let _guard = env_lock().lock();
            let saved = keys.iter().map(|k| (*k, std::env::var_os(k))).collect();
            Self { saved, _guard }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    /// Compare two `&[Val]` slices and panic with a focused diff on
    /// mismatch: first divergent index, two cells of context on each
    /// side, and a hint about regenerating goldens. Avoids the
    /// 256-element vector dump that `assert_eq!` produces by default.
    fn assert_check_eq(expected: &[Val], actual: &[Val], label: &str) {
        if expected.len() != actual.len() {
            panic!(
                "{label}: length mismatch expected={} actual={}",
                expected.len(),
                actual.len()
            );
        }
        let total_diffs = expected.iter().zip(actual).filter(|(a, b)| a != b).count();
        if total_diffs > 0 {
            let i = expected
                .iter()
                .zip(actual)
                .position(|(a, b)| a != b)
                .expect("count > 0 implies position exists");
            let lo = i.saturating_sub(2);
            let hi = (i + 3).min(expected.len());
            // Index column is decimal-padded to 8 chars: production PO2=20
            // produces up to ExtVal::EXT_SIZE * INV_RATE * 2^20 = 2^24 ≈ 16M
            // indices, which fits in 8 decimal digits exactly. Format width
            // is padding-only, so larger PO2 still prints correctly (just
            // unaligned).
            let mut ctx = String::new();
            for j in lo..hi {
                let mark = if j == i { ">>" } else { "  " };
                ctx.push_str(&format!(
                    "\n  {mark} [{j:8}] expected=0x{:08x} actual=0x{:08x}",
                    expected[j].as_u32(),
                    actual[j].as_u32()
                ));
            }
            panic!(
                "{label}: {total_diffs} of {} elements differ; first at index {i}{ctx}\n\
                 hint: rerun with --nocapture; set RISC0_PHASE_0B_GENERATE_GOLDEN=1 to \
                 regenerate goldens if the kernel change is intentional",
                expected.len()
            );
        }
    }

    /// Deterministic RNG seeds — distinct per test so a failure is
    /// isolatable to the test that chose the seed.
    const SEED_RANDOM: u64 = 0xB12E_BEEF_DEAD_C0DE;
    const SEED_ONE_HOT: u64 = 0x01EC_AFEB_ABEF_00D0;
    const SEED_ZERO_FILL: u64 = 0xBA5E_BA11_FACE_0FF1;
    const SEED_PMINUS1: u64 = 0xDEAD_BEEF_BADF_00D1;

    /// Bundle of inputs passed to a single `eval_check` invocation.
    /// Mirrors `cuda.rs::tests::EvalCheckParams` field-for-field so test
    /// code parallels easily.
    struct EvalCheckParams {
        po2: usize,
        steps: usize,
        domain: usize,
        code: Vec<Val>,
        data: Vec<Val>,
        accum: Vec<Val>,
        mix: Vec<Val>,
        out: Vec<Val>,
        poly_mix: ExtVal,
    }

    impl EvalCheckParams {
        /// Random inputs, deterministic via `StdRng::seed_from_u64`.
        /// `StdRng` aliases ChaCha12Rng in rand 0.9 (tracked by the rand
        /// crate's reproducibility guarantees within a minor version).
        ///
        /// RNG byte-consumption order — code, data, accum, mix, out,
        /// poly_mix — is part of the test contract: reordering field
        /// initializers below would silently invalidate any goldens
        /// derived from the same seed.
        fn random(po2: usize, seed: u64) -> Self {
            let mut rng = StdRng::seed_from_u64(seed);
            let steps = 1 << po2;
            let domain = steps * INV_RATE;
            let code_size = TAPSET.group_size(REGISTER_GROUP_CODE);
            let data_size = TAPSET.group_size(REGISTER_GROUP_DATA);
            let accum_size = TAPSET.group_size(REGISTER_GROUP_ACCUM);
            Self {
                po2,
                steps,
                domain,
                code: random_fps(&mut rng, code_size * domain),
                data: random_fps(&mut rng, data_size * domain),
                accum: random_fps(&mut rng, accum_size * domain),
                mix: random_fps(&mut rng, CircuitImpl::MIX_SIZE),
                out: random_fps(&mut rng, CircuitImpl::OUTPUT_SIZE),
                poly_mix: ExtVal::random(&mut rng),
            }
        }

        /// All trace cells filled with the same Val. Use Val::ZERO and
        /// Val::new(p-1) (== Val::ZERO - Val::ONE in BabyBear) to exercise
        /// boundary arithmetic.
        fn boundary_fill(po2: usize, fill: Val, seed: u64) -> Self {
            let steps = 1 << po2;
            let domain = steps * INV_RATE;
            let code_size = TAPSET.group_size(REGISTER_GROUP_CODE);
            let data_size = TAPSET.group_size(REGISTER_GROUP_DATA);
            let accum_size = TAPSET.group_size(REGISTER_GROUP_ACCUM);
            // poly_mix and out/mix get a non-zero deterministic value so the
            // accumulation chain isn't trivially zero (which would mask
            // ordering bugs in the kernel).
            let mut rng = StdRng::seed_from_u64(seed);
            Self {
                po2,
                steps,
                domain,
                code: vec![fill; code_size * domain],
                data: vec![fill; data_size * domain],
                accum: vec![fill; accum_size * domain],
                mix: random_fps(&mut rng, CircuitImpl::MIX_SIZE),
                out: random_fps(&mut rng, CircuitImpl::OUTPUT_SIZE),
                poly_mix: ExtVal::random(&mut rng),
            }
        }

        /// One-hot in the data buffer at (column, row). All other cells are
        /// zero across all groups. Tests that a single non-zero cell
        /// propagates correctly through the constraint evaluation.
        fn one_hot_data(po2: usize, column: usize, row: usize) -> Self {
            let steps = 1 << po2;
            let domain = steps * INV_RATE;
            let code_size = TAPSET.group_size(REGISTER_GROUP_CODE);
            let data_size = TAPSET.group_size(REGISTER_GROUP_DATA);
            let accum_size = TAPSET.group_size(REGISTER_GROUP_ACCUM);
            assert!(column < data_size, "column {} out of range", column);
            assert!(row < domain, "row {} out of range", row);
            let mut data = vec![Val::ZERO; data_size * domain];
            // Layout: data[col * domain + row]. Match the kernel's column-
            // major indexing.
            data[column * domain + row] = Val::ONE;
            let mut rng = StdRng::seed_from_u64(SEED_ONE_HOT);
            Self {
                po2,
                steps,
                domain,
                code: vec![Val::ZERO; code_size * domain],
                data,
                accum: vec![Val::ZERO; accum_size * domain],
                mix: random_fps(&mut rng, CircuitImpl::MIX_SIZE),
                out: random_fps(&mut rng, CircuitImpl::OUTPUT_SIZE),
                poly_mix: ExtVal::random(&mut rng),
            }
        }
    }

    fn random_fps<E: Elem>(rng: &mut impl Rng, size: usize) -> Vec<E> {
        (0..size).map(|_| E::random(rng)).collect()
    }

    /// Run eval_check against an arbitrary HAL pair, return the check buffer.
    ///
    /// `eval_check_dep()` is called between submission and `view()` to
    /// mirror the production prover's sequencing. The CPU HAL's
    /// `eval_check_dep` is a default-impl no-op.
    fn eval_check_impl<H, C>(params: &EvalCheckParams, hal: &H, circuit_hal: &C) -> Vec<H::Elem>
    where
        H: Hal<Elem = Val, ExtElem = ExtVal>,
        C: CircuitHal<H>,
    {
        let check = hal.alloc_elem("check", ExtVal::EXT_SIZE * params.domain);
        let code = hal.copy_from_elem("code", &params.code);
        let data = hal.copy_from_elem("data", &params.data);
        let accum = hal.copy_from_elem("accum", &params.accum);
        let mix = hal.copy_from_elem("mix", &params.mix);
        let out = hal.copy_from_elem("out", &params.out);
        circuit_hal.eval_check(
            &check,
            &[&accum, &code, &data],
            &[&mix, &out],
            params.poly_mix,
            params.po2,
            params.steps,
        );
        circuit_hal.eval_check_dep();
        let mut ret = vec![H::Elem::ZERO; check.size()];
        check.view(|view| ret.clone_from_slice(view));
        ret
    }

    /// Run the Intel HAL's eval_check inside a scoped multipass setting.
    /// The EnvGuard:
    /// - serializes against other env-mutating tests (panic-safe restore);
    /// - captures both RISC0_MULTIPASS and RISC0_PHASE_DISABLE so that a
    ///   CI environment exporting RISC0_PHASE_DISABLE=multipass cannot
    ///   silently downgrade multipass tests to the mono path.
    ///
    /// Removes only the `multipass` token from RISC0_PHASE_DISABLE so the
    /// explicit `multipass` argument always wins, while preserving any
    /// other phase entries (e.g. `block_read`, `tree_decomp`,
    /// `subgroup_partition`) that CI may legitimately need disabled for
    /// the same run. Without selective clearing, future Phase 1.5+ tests
    /// would silently re-enable phases CI tried to disable.
    fn intel_eval_check(params: &EvalCheckParams, multipass: bool) -> Vec<Val> {
        let _env = EnvGuard::new(&["RISC0_MULTIPASS", "RISC0_PHASE_DISABLE"]);
        if let Ok(prev) = std::env::var("RISC0_PHASE_DISABLE") {
            let kept: Vec<&str> = prev
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("multipass"))
                .collect();
            if kept.is_empty() {
                std::env::remove_var("RISC0_PHASE_DISABLE");
            } else {
                std::env::set_var("RISC0_PHASE_DISABLE", kept.join(","));
            }
        }
        if multipass {
            std::env::set_var("RISC0_MULTIPASS", "1");
        } else {
            std::env::remove_var("RISC0_MULTIPASS");
        }
        let hal = Arc::new(IntelHalSha256::new());
        let circuit_hal = IntelCircuitHal::new(hal.clone());
        eval_check_impl(params, hal.as_ref(), &circuit_hal)
    }

    fn cpu_eval_check(params: &EvalCheckParams) -> Vec<Val> {
        let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
        let cpu_eval = CpuCircuitHal;
        eval_check_impl(params, &cpu_hal, &cpu_eval)
    }

    /// Path to the golden file for a given mode. Anchors on
    /// `CARGO_MANIFEST_DIR` at compile time (via `env!`) so the path is
    /// stable under cargo test, cargo nextest, or direct invocation of
    /// the test binary regardless of cwd.
    fn golden_path(mode: &str, po2: usize) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join("intel_eval_check")
            .join(format!("{mode}_po2_{po2}.bin"))
    }

    /// Convert `&[Val]` to portable little-endian bytes of the NORMAL-form
    /// (decoded) u32 representation. Independent of host endianness AND
    /// of the Montgomery encoding constants — a future toolchain change
    /// to `R = 2^32 mod P` does not silently invalidate goldens.
    fn vals_to_normal_le_bytes(vals: &[Val]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            out.extend_from_slice(&v.as_u32().to_le_bytes());
        }
        out
    }

    /// Generate or compare against a golden .bin file for byte-exact
    /// regression testing. Three modes:
    /// - `RISC0_PHASE_0B_GENERATE_GOLDEN=1`: write the golden;
    /// - `RISC0_REQUIRE_GOLDEN=1`: missing golden is a panic (CI uses this);
    /// - default: missing golden logs a notice and skips the assertion
    ///   (lets a fresh checkout pass before goldens land).
    fn assert_or_save_golden(mode: &str, po2: usize, observed: &[Val]) {
        let path = golden_path(mode, po2);
        let observed_bytes = vals_to_normal_le_bytes(observed);
        if std::env::var_os("RISC0_PHASE_0B_GENERATE_GOLDEN").is_some() {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("failed to mkdir golden parent");
            }
            std::fs::write(&path, &observed_bytes).expect("failed to write golden");
            eprintln!(
                "wrote golden: {} ({} bytes)",
                path.display(),
                observed_bytes.len()
            );
            return;
        }
        match std::fs::read(&path) {
            Ok(expected) => {
                assert_eq!(
                    observed_bytes,
                    expected,
                    "golden mismatch for {} (path={})",
                    mode,
                    path.display()
                );
            }
            Err(_) => {
                if std::env::var_os("RISC0_REQUIRE_GOLDEN").is_some() {
                    panic!(
                        "golden missing at {} and RISC0_REQUIRE_GOLDEN=1 is set — \
                         run with RISC0_PHASE_0B_GENERATE_GOLDEN=1 on Battlemage \
                         hardware to populate",
                        path.display()
                    );
                }
                eprintln!(
                    "note: golden missing at {} — run with RISC0_PHASE_0B_GENERATE_GOLDEN=1 \
                     to populate (skipping byte-exact assertion this run)",
                    path.display()
                );
            }
        }
    }

    /// 1. Random differential — Intel mono path matches CPU reference.
    #[test]
    fn eval_check_random_mono() {
        const PO2: usize = 4;
        let params = EvalCheckParams::random(PO2, SEED_RANDOM);
        let cpu = cpu_eval_check(&params);
        let intel = intel_eval_check(&params, /* multipass = */ false);
        assert_check_eq(&cpu, &intel, "Intel mono vs CPU");
        assert_or_save_golden("mono_random", PO2, &intel);
    }

    /// Random differential at po2=21, where the data group's evaluated buffer
    /// exceeds 4 GiB and any 32-bit byte addressing would wrap.
    #[test]
    #[ignore = "po2=21: ~10 GB host and GPU memory, several minutes"]
    fn eval_check_random_mono_po2_21() {
        const PO2: usize = 21;
        let params = EvalCheckParams::random(PO2, SEED_RANDOM);
        let cpu = cpu_eval_check(&params);
        let intel = intel_eval_check(&params, /* multipass = */ false);
        assert_eq!(cpu.len(), intel.len());
        let bad: Vec<usize> = (0..cpu.len()).filter(|&i| cpu[i] != intel[i]).collect();
        eprintln!(
            "[eval_check po2=21] {}/{} differ; first={:?} last={:?}",
            bad.len(),
            cpu.len(),
            bad.first(),
            bad.last()
        );
        assert!(
            bad.is_empty(),
            "Intel eval_check diverged from CPU at po2=21"
        );
    }

    /// Intel eval_check matches CPU at the small production po2s, on both
    /// the mono and multipass paths.
    #[test]
    #[ignore = "slow: CPU reference at production po2"]
    fn eval_check_random_small_po2s() {
        let mut failures = vec![];
        for po2 in [15usize, 16, 17] {
            let params = EvalCheckParams::random(po2, SEED_RANDOM);
            let cpu = cpu_eval_check(&params);
            for multipass in [false, true] {
                let intel = intel_eval_check(&params, multipass);
                let bad = (0..cpu.len()).filter(|&i| cpu[i] != intel[i]).count();
                eprintln!(
                    "[eval_check po2={po2} multipass={multipass}] {bad}/{} differ",
                    cpu.len()
                );
                if bad > 0 {
                    failures.push((po2, multipass));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "Intel eval_check diverged: {failures:?}"
        );
    }

    /// 2. Random differential — Intel multipass path matches CPU reference.
    #[test]
    fn eval_check_random_multipass() {
        const PO2: usize = 4;
        let params = EvalCheckParams::random(PO2, SEED_RANDOM);
        let cpu = cpu_eval_check(&params);
        let intel = intel_eval_check(&params, /* multipass = */ true);
        assert_check_eq(&cpu, &intel, "Intel multipass vs CPU");
        assert_or_save_golden("multipass_random", PO2, &intel);
    }

    /// 3. Mono == multipass == CPU (tripod check). Comparing only mono to
    /// multipass would miss a shared kernel bug where both produce the
    /// same wrong answer; anchoring against CPU closes that gap.
    #[test]
    fn eval_check_mono_vs_multipass() {
        const PO2: usize = 4;
        let params = EvalCheckParams::random(PO2, SEED_RANDOM);
        let cpu = cpu_eval_check(&params);
        let mono = intel_eval_check(&params, false);
        let multipass = intel_eval_check(&params, true);
        assert_check_eq(&cpu, &mono, "tripod: mono vs CPU");
        assert_check_eq(&cpu, &multipass, "tripod: multipass vs CPU");
        assert_check_eq(&mono, &multipass, "tripod: mono vs multipass");
    }

    /// 4. Boundary fill with Val::ZERO. Exercises the kernel's handling of
    /// the additive identity throughout the trace.
    #[test]
    fn eval_check_zero_fill() {
        const PO2: usize = 4;
        let params = EvalCheckParams::boundary_fill(PO2, Val::ZERO, SEED_ZERO_FILL);
        let cpu = cpu_eval_check(&params);
        let intel = intel_eval_check(&params, false);
        assert_check_eq(&cpu, &intel, "zero_fill: mono vs CPU");
        let multipass = intel_eval_check(&params, true);
        assert_check_eq(&cpu, &multipass, "zero_fill: multipass vs CPU");
    }

    /// 5. Boundary fill with p-1 (== -1 in BabyBear). Exercises Mont mul
    /// overflow paths — many constraint computations multiply (p-1) * x
    /// which exercises the `t = ax + bx` and lazy-reduction code paths
    /// that were the focus of round-1 review of FRI fold kernels.
    #[test]
    fn eval_check_p_minus_1_fill() {
        const PO2: usize = 4;
        // Val::new(p-1) where p = 0x78000001. In BabyBear: -Val::ONE.
        let p_minus_1 = Val::ZERO - Val::ONE;
        let params = EvalCheckParams::boundary_fill(PO2, p_minus_1, SEED_PMINUS1);
        let cpu = cpu_eval_check(&params);
        let intel = intel_eval_check(&params, false);
        assert_check_eq(&cpu, &intel, "p_minus_1_fill: mono vs CPU");
        let multipass = intel_eval_check(&params, true);
        assert_check_eq(&cpu, &multipass, "p_minus_1_fill: multipass vs CPU");
    }

    /// 6. One-hot data column. Exposes any column-stride or per-row
    /// indexing bug that would silently zero the propagation of a
    /// single live cell.
    #[test]
    fn eval_check_one_hot_data() {
        const PO2: usize = 4;
        let data_size = TAPSET.group_size(REGISTER_GROUP_DATA);
        // Pick a non-trivial interior column + interior row so we don't
        // accidentally land on a boundary the kernel special-cases.
        let column = data_size / 3;
        let row = (1usize << PO2) * INV_RATE / 2 + 1;
        let params = EvalCheckParams::one_hot_data(PO2, column, row);
        let cpu = cpu_eval_check(&params);
        let intel = intel_eval_check(&params, false);
        assert_check_eq(&cpu, &intel, "one_hot_data: mono vs CPU");
        let multipass = intel_eval_check(&params, true);
        assert_check_eq(&cpu, &multipass, "one_hot_data: multipass vs CPU");
    }

    /// 8. Phase 0c profiling workload: runs `eval_check` `RISC0_PHASE0C_ITERS`
    /// times (default 100) at PO2 `RISC0_PHASE0C_PO2` (default 4) with
    /// 10-iter warmup, prints min/median/p99 latency to stdout. Gated
    /// `#[ignore]` so it doesn't run under `cargo test`; invoke explicitly
    /// via `cargo test --release --features intel --ignored phase0c_bench`
    /// (typically wrapped in `unitrace`). Honors
    /// `RISC0_PHASE0C_MULTIPASS=1` (the bench-only knob; do NOT set
    /// `RISC0_MULTIPASS` directly — the bench overrides it).
    #[test]
    #[ignore = "Phase 0c profiling workload — long-running, GPU-only"]
    fn phase0c_bench() {
        use std::time::Instant;
        let _env = EnvGuard::new(&["RISC0_MULTIPASS", "RISC0_PHASE_DISABLE"]);
        let po2: usize = std::env::var("RISC0_PHASE0C_PO2")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let iters: usize = std::env::var("RISC0_PHASE0C_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100)
            .max(1); // Guard against accidental ITERS=0; index OOB would
                     // panic on times[len/2] otherwise.
        let warmup: usize = std::env::var("RISC0_PHASE0C_WARMUP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        let multipass = std::env::var_os("RISC0_PHASE0C_MULTIPASS").is_some();

        eprintln!("phase0c_bench: po2={po2} iters={iters} warmup={warmup} multipass={multipass}");

        // Build params once; the kernel reads input buffers, doesn't mutate
        // them, so the same params can be reused across iterations.
        let params = EvalCheckParams::random(po2, SEED_RANDOM);

        // Configure multipass selector. EnvGuard restores on Drop.
        if multipass {
            std::env::set_var("RISC0_MULTIPASS", "1");
        } else {
            std::env::remove_var("RISC0_MULTIPASS");
        }
        // Always strip RISC0_PHASE_DISABLE for the bench: bisection during
        // profiling is done by re-launching the bench, not by leaking env.
        std::env::remove_var("RISC0_PHASE_DISABLE");

        let hal = Arc::new(IntelHalSha256::new());
        let circuit_hal = IntelCircuitHal::new(hal.clone());

        // Warmup: discard timing, just amortize SYCL queue init + first-time
        // ESIMD JIT into device cache.
        for _ in 0..warmup {
            let _ = eval_check_impl(&params, hal.as_ref(), &circuit_hal);
        }

        // Measure.
        let mut times: Vec<u128> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = eval_check_impl(&params, hal.as_ref(), &circuit_hal);
            times.push(t.elapsed().as_nanos());
        }
        times.sort_unstable();

        let min = times[0];
        let p50 = times[times.len() / 2];
        let p99 = times[(times.len() * 99) / 100];
        let max = *times.last().unwrap();
        let mean: u128 = times.iter().sum::<u128>() / times.len() as u128;
        // Coefficient of variation = stddev / mean. Used for V5's CoV<10%
        // reproducibility gate.
        let var: f64 = times
            .iter()
            .map(|&x| {
                let d = x as f64 - mean as f64;
                d * d
            })
            .sum::<f64>()
            / times.len() as f64;
        let cov = var.sqrt() / mean as f64;

        // Machine-readable single line for run_profile.sh / classify.py.
        println!(
            "PHASE0C_BENCH po2={po2} iters={iters} multipass={multipass} \
             min_ns={min} p50_ns={p50} p99_ns={p99} max_ns={max} \
             mean_ns={mean} cov={cov:.4}"
        );
    }

    /// 7. RISC0_PHASE_DISABLE=multipass should suppress multipass even
    /// when RISC0_MULTIPASS is set. Verifies the bisection mechanism.
    ///
    /// EnvGuard ensures (a) we serialize against other env-mutating tests
    /// and (b) restoration runs even on panic, so no state leaks to
    /// downstream tests.
    #[test]
    fn phase_disable_overrides_multipass() {
        let _env = EnvGuard::new(&["RISC0_MULTIPASS", "RISC0_PHASE_DISABLE"]);

        // Plain match.
        std::env::set_var("RISC0_PHASE_DISABLE", "multipass");
        std::env::set_var("RISC0_MULTIPASS", "1");
        assert!(
            super::phase_disabled("multipass"),
            "phase_disabled should report multipass disabled when env lists it"
        );
        assert!(
            !super::phase_disabled("not_listed"),
            "unrelated phases must remain enabled"
        );
        assert!(
            super::phase_disabled("MultiPass"),
            "phase matching must be ASCII case-insensitive"
        );
        assert!(
            !super::phase_disabled(""),
            "empty query never matches even with CSV empties"
        );

        // Whitespace + leading/trailing/double-comma robustness.
        std::env::set_var("RISC0_PHASE_DISABLE", " multipass ,, block_read ,");
        assert!(
            super::phase_disabled("multipass"),
            "leading/trailing whitespace must be trimmed"
        );
        assert!(
            super::phase_disabled("block_read"),
            "second entry after empty token must still match"
        );
        assert!(
            !super::phase_disabled(""),
            "empty query must not match the empty CSV tokens"
        );
    }
}

/// Debug aid: with `RISC0_INTEL_EVAL_CHECK_SHADOW=1`, wait for the GPU
/// eval_check, rerun it on the CPU from the same inputs and report diverging
/// cells. This is how the po2=16 warmup-order bug (whole SIMD16 threads of the
/// monolithic kernel computing garbage) was found; random-input A/B tests
/// cannot see failures that depend on runtime state. Very slow; off by default.
fn shadow_eval_check(
    check: &IntelBuffer<Val>,
    groups: &[&IntelBuffer<Val>],
    globals: &[&IntelBuffer<Val>],
    poly_mix: ExtVal,
    po2: usize,
    steps: usize,
) {
    use risc0_zkp::{
        core::hash::poseidon2::Poseidon2HashSuite,
        hal::{cpu::CpuHal, Buffer as _},
    };
    if std::env::var_os("RISC0_INTEL_EVAL_CHECK_SHADOW").is_none() {
        return;
    }
    let cpu_hal: CpuHal<risc0_core::field::baby_bear::BabyBear> =
        CpuHal::new(Poseidon2HashSuite::new_suite());
    let g: Vec<_> = groups
        .iter()
        .map(|b| cpu_hal.copy_from_elem("g", &b.to_vec()))
        .collect();
    let gl: Vec<_> = globals
        .iter()
        .map(|b| cpu_hal.copy_from_elem("gl", &b.to_vec()))
        .collect();
    let c = cpu_hal.alloc_elem("check", check.size());
    let gr: Vec<&_> = g.iter().collect();
    let glr: Vec<&_> = gl.iter().collect();
    <crate::prove::hal::cpu::CpuCircuitHal as CircuitHal<
        CpuHal<risc0_core::field::baby_bear::BabyBear>,
    >>::eval_check(
        &crate::prove::hal::cpu::CpuCircuitHal,
        &c,
        &gr,
        &glr,
        poly_mix,
        po2,
        steps,
    );
    let (gv, cv) = (check.to_vec(), c.to_vec());
    let bad = gv.iter().zip(&cv).filter(|(a, b)| a != b).count();
    let first = gv.iter().zip(&cv).position(|(a, b)| a != b);
    eprintln!(
        "[eval_check shadow] po2={po2} groups={:?} bad={bad}/{} first={first:?}",
        groups.iter().map(|b| b.size()).collect::<Vec<_>>(),
        gv.len()
    );
    if bad > 0 {
        let domain = steps * INV_RATE;
        let idx: Vec<usize> = (0..gv.len()).filter(|&i| gv[i] != cv[i]).collect();
        let mut by_comp = [0usize; 4];
        let mut by_mod16 = [0usize; 16];
        let mut rows = std::collections::BTreeSet::new();
        for &i in &idx {
            by_comp[i / domain] += 1;
            by_mod16[(i % domain) % 16] += 1;
            rows.insert(i % domain);
        }
        let r: Vec<_> = rows.iter().take(24).collect();
        eprintln!(
            "[eval_check shadow] by_comp={by_comp:?} rows={} first_rows={r:?}",
            rows.len()
        );
        eprintln!("[eval_check shadow] by_row_mod16={by_mod16:?}");
        let mut cyc = std::collections::BTreeMap::new();
        for &row in &rows {
            *cyc.entry(row % INV_RATE).or_insert(0usize) += 1;
        }
        eprintln!(
            "[eval_check shadow] by_row_mod_INV_RATE={cyc:?} max_row={:?}",
            rows.iter().last()
        );
    }
}
