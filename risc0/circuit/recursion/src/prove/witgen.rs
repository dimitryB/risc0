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

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, LazyLock, Mutex,
};

use anyhow::{Context as _, Result};
use risc0_circuit_recursion_sys::{RawPreflightTrace, StepMode};
use risc0_core::scope;
use risc0_zkp::{
    adapter::{CircuitInfo as _, TapsProvider as _},
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        Elem as _,
    },
    hal::{Buffer as _, Hal},
};

use crate::{CircuitImpl, CIRCUIT};

use super::{preflight::Preflight, CircuitAccumulator, CircuitWitnessGenerator, Program};

// Transposed (column-major, zero-padded to 2^po2 rows) ctrl vectors, shared by
// the GPU-thread witgen path and the background prewitgen path.
//
// Keyed by `ctrl_cache_key` (code_rows, po2, sampled code fingerprint) - the
// same identity the device ctrl caches in prove/mod.rs use - and every hit is
// validated EXACTLY against the program code (all rows, all columns) before it
// is used, so two programs with an equal key can never share a ctrl. (The
// previous key, (code_rows, po2), silently ignored program identity.) The
// full compare is ~24M strided reads split across columns with rayon; it runs
// at most once per proof and, with prewitgen, off the GPU thread.
// Bounded to CTRL_CACHE_CAP programs (lift, join, resolve, identity...).
const CTRL_CACHE_CAP: usize = 4;
type HostCtrlCache = Vec<(super::CtrlKey, Arc<Vec<BabyBearElem>>)>;
static CTRL_CACHE: LazyLock<Mutex<HostCtrlCache>> = LazyLock::new(|| Mutex::new(Vec::new()));

fn ctrl_matches(ctrl: &[BabyBearElem], zkr: &Program, total_cycles: usize) -> bool {
    use rayon::prelude::*;
    let ctrl_size = zkr.code_size;
    let rows = zkr.code_rows();
    if ctrl.len() != total_cycles * ctrl_size {
        return false;
    }
    ctrl.par_chunks(total_cycles).enumerate().all(|(j, col)| {
        (0..rows).all(|i| col[i] == zkr.code[i * ctrl_size + j])
            && col[rows..].iter().all(|x| *x == BabyBearElem::ZERO)
    })
}

/// Return the transposed ctrl for `zkr`, from the cache if an exactly matching
/// entry exists, otherwise built (and cached).
pub(crate) fn host_ctrl_for(zkr: &Program) -> Arc<Vec<BabyBearElem>> {
    let total_cycles = 1usize << zkr.po2;
    let key = super::ctrl_cache_key(zkr);
    let hit = CTRL_CACHE
        .lock()
        .unwrap()
        .iter()
        .find(|e| e.0 == key)
        .map(|e| e.1.clone());
    if let Some(c) = hit {
        if ctrl_matches(&c, zkr, total_cycles) {
            return c;
        }
    }
    let ctrl_size = CIRCUIT.ctrl_size();
    assert_eq!(ctrl_size, zkr.code_size);
    // One column per task: contiguous writes (the row-major loop wrote with a 1 MB stride).
    let mut v = vec![BabyBearElem::ZERO; total_cycles * ctrl_size];
    {
        use rayon::prelude::*;
        let rows = zkr.code_rows();
        v.par_chunks_mut(total_cycles)
            .enumerate()
            .for_each(|(j, col)| {
                for (i, x) in col[..rows].iter_mut().enumerate() {
                    *x = zkr.code[i * ctrl_size + j];
                }
            });
    }
    let v = Arc::new(v);
    let mut cache = CTRL_CACHE.lock().unwrap();
    cache.retain(|e| e.0 != key);
    if cache.len() >= CTRL_CACHE_CAP {
        cache.remove(0);
    }
    cache.push((key, v.clone()));
    v
}

/// RISC0_RECURSION_PREWITGEN_CHECK: "1"/"always" every proof, "once" = the
/// first prewitgen proof of each program in this process, default never (set it
/// in CI; a bad witness still fails the next join and the final verify).
/// The check regenerates the witness through the HAL path into the device
/// buffers and compares it with the host (prewitgen) witness before upload,
/// i.e. it compares two independently computed witnesses (unlike the host
/// shadow check, which only compares a buffer with its own upload source).
fn prewitgen_check_due(key: super::CtrlKey) -> bool {
    static CHECKED: LazyLock<Mutex<Vec<super::CtrlKey>>> = LazyLock::new(|| Mutex::new(Vec::new()));
    match std::env::var("RISC0_RECURSION_PREWITGEN_CHECK").as_deref() {
        Ok("0") => false,
        Ok("1") | Ok("always") => true,
        Ok("once") => {
            let mut c = CHECKED.lock().unwrap();
            if c.contains(&key) {
                false
            } else {
                c.push(key);
                true
            }
        }
        _ => std::env::var_os("RISC0_RECURSION_SHADOW_CHECK").is_some(),
    }
}

fn first_mismatch(a: &[BabyBearElem], b: &[BabyBearElem]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x.as_u32_montgomery() != y.as_u32_montgomery())
}

/// Rows of the long sequential (non-par-safe) witgen chains, computed on the host.
///
/// The device witgen runs each chain in one GPU thread; the longest lift chain is a
/// 1439-cycle Poseidon2 sponge that costs ~23 ms of the ~26 ms stepExec. A host core runs
/// it ~20x faster per step, in parallel with the others, off the GPU thread (`prepare`).
pub(crate) struct HostChains {
    pub heads: Vec<u32>,
    pub cycles: Vec<u32>,
    pub rows: Vec<u32>,
    pub wom_idx: Vec<u32>,
    pub wom_rows: Vec<u32>,
    /// Addresses of page-locked copies of the five arrays above (same contents).
    pub pinned: Option<[usize; 5]>,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn pinned_ptr<T>(p: Option<&super::pinned::PinnedTrace>, i: usize) -> Option<*const T> {
    p.map(|p| p.ptr::<T>(i))
}

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
fn pinned_ptr<T>(_p: Option<&()>, _i: usize) -> Option<*const T> {
    None
}

/// Chains at least this long run on the host.
const HOST_CHAIN_MIN_LEN: usize = 16;

/// When set, every witness generated with host chains is regenerated without them and compared
/// (a test hook; see [crate::prove::set_host_chains_check]).
pub(crate) static HOST_CHAINS_CHECK: AtomicBool = AtomicBool::new(false);
/// How many witnesses the check compared.
pub(crate) static HOST_CHAINS_CHECKED: AtomicUsize = AtomicUsize::new(0);

// RISC0_HIP_REC_ZCHECK=1: check each row-pass witness against the generic passes (validation).
static ROW_PASS_CHECK: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("RISC0_HIP_REC_ZCHECK").as_deref() == Ok("1"));
static ROW_PASS_CHECKED: AtomicUsize = AtomicUsize::new(0);

/// Execute the chains of at least `HOST_CHAIN_MIN_LEN` cycles on the host. `None` if there are
/// none, or if that fails (e.g. a chain writes a global): the device witgen then runs every
/// chain itself, as without host chains.
pub(crate) fn host_chains(zkr: &Program, preflight: &Preflight) -> Option<HostChains> {
    match try_host_chains(zkr, preflight) {
        Ok(chains) => chains,
        Err(err) => {
            tracing::debug!("host chains: falling back to device witgen: {err}");
            None
        }
    }
}

fn try_host_chains(zkr: &Program, preflight: &Preflight) -> Result<Option<HostChains>> {
    let t = &preflight.trace;
    Ok(try_host_chains_range(zkr, &t.wom, &t.cycles, &t.iops, 0, true)?.0)
}

/// The chains of `cycles[lo..]` (`lo` a chain start). With `complete == false` the trace
/// is a prefix of a running preflight: the last, possibly unfinished, chain is left out, and the
/// returned boundary is where a later pass over the full trace must start.
pub(crate) fn try_host_chains_range(
    zkr: &Program,
    wom: &[BabyBearExtElem],
    cycles: &[risc0_circuit_recursion_sys::RawPreflightCycle],
    iops: &[BabyBearExtElem],
    lo: usize,
    complete: bool,
) -> Result<(Option<HostChains>, usize)> {
    use risc0_circuit_recursion_sys::risc0_circuit_recursion_cpu_chains;
    let min_len = HOST_CHAIN_MIN_LEN;
    let n = cycles.len();
    let mut chains: Vec<u32> = Vec::new();
    let mut heads: Vec<u32> = Vec::new();
    let mut start = lo;
    let mut num_rows = 0usize;
    for c in lo + 1..=n {
        // c == n: the end of the trace.
        let closed = cycles.get(c).map_or(complete, |cy| cy.is_par_safe != 0);
        if closed {
            let len = c - start;
            if len >= min_len {
                chains.push(start as u32);
                chains.push(len as u32);
                heads.push(start as u32);
                num_rows += len;
            }
            start = c;
        }
    }
    let boundary = if complete { n } else { start };
    if heads.is_empty() {
        return Ok((None, boundary));
    }
    let total_cycles = 1usize << zkr.po2;
    let data_cols = CIRCUIT.data_size();
    assert_eq!(zkr.code.len(), zkr.code_rows() * zkr.code_size);
    let raw_trace = RawPreflightTrace {
        wom: wom.as_ptr(),
        cycles: cycles.as_ptr(),
        iops: iops.as_ptr(),
        num_woms: wom.len() as u32,
        num_cycles: n as u32,
        num_iops: iops.len() as u32,
    };
    let mut out = HostChains {
        heads,
        cycles: vec![0; num_rows],
        rows: vec![0; num_rows * data_cols],
        wom_idx: vec![0; num_rows],
        wom_rows: vec![0; num_rows * 9 * 5],
        pinned: None,
    };
    risc0_sys::ffi_wrap(|| unsafe {
        risc0_circuit_recursion_cpu_chains(
            zkr.code.as_ptr(),
            zkr.code_size as u32,
            &raw_trace,
            total_cycles as u32,
            data_cols as u32,
            chains.as_ptr(),
            out.heads.len() as u32,
            out.cycles.as_mut_ptr(),
            out.rows.as_mut_ptr(),
            out.wom_idx.as_mut_ptr(),
            out.wom_rows.as_mut_ptr(),
        )
    })?;
    Ok((Some(out), boundary))
}

/// Concatenate the chains of two disjoint passes.
#[cfg_attr(not(any(feature = "cuda", feature = "rocm")), allow(dead_code))]
pub(crate) fn merge_host_chains(
    a: Option<HostChains>,
    b: Option<HostChains>,
) -> Option<HostChains> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(mut a), Some(b)) => {
            a.heads.extend(b.heads);
            a.cycles.extend(b.cycles);
            a.rows.extend(b.rows);
            a.wom_idx.extend(b.wom_idx);
            a.wom_rows.extend(b.wom_rows);
            Some(a)
        }
    }
}

pub(crate) struct WitnessGenerator<H: Hal> {
    work_cycles: u32,
    total_cycles: u32,
    pub global: H::Buffer<H::Elem>,
    pub ctrl: H::Buffer<H::Elem>,
    pub data: H::Buffer<H::Elem>,
    pub accum: H::Buffer<H::Elem>,
}

impl<H> WitnessGenerator<H>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
{
    pub fn new<C: CircuitWitnessGenerator<H>>(
        hal: &H,
        circuit_hal: &C,
        zkr: &Program,
        preflight: &Preflight,
        cached_ctrl_buffer: Option<H::Buffer<H::Elem>>,
    ) -> Result<Self> {
        scope!("witgen");

        let total_cycles = 1 << zkr.po2;

        let global = vec![BabyBearElem::INVALID; CircuitImpl::OUTPUT_SIZE];
        let global = hal.copy_from_elem("global", &global);

        // Use pre-existing GPU ctrl buffer if provided (avoids ~24MB re-upload per proof).
        let ctrl = if let Some(buf) = cached_ctrl_buffer {
            buf
        } else {
            let ctrl_data = host_ctrl_for(zkr);
            hal.copy_from_elem("ctrl", &ctrl_data)
        };

        let work_cycles = zkr.code_rows() as u32;
        // The HAL's row passes (HIP): rows >= work_cycles start as zero, the noise is a fill and
        // the zeroize reads only rows < work_cycles (see `CircuitWitnessGenerator::init_rows`).
        // Not with a host-generated witness, whose upload writes the whole buffer.
        let row_passes =
            circuit_hal.has_row_passes() && preflight.host_witness.lock().unwrap().is_none();
        let init_data = |name: &'static str| -> Result<H::Buffer<H::Elem>> {
            let size = total_cycles * CIRCUIT.data_size();
            if row_passes {
                let buf = hal.alloc_elem(name, size);
                circuit_hal.init_rows(&buf, total_cycles, work_cycles as usize)?;
                Ok(buf)
            } else {
                Ok(hal.alloc_elem_init(name, size, BabyBearElem::INVALID))
            }
        };
        let data = init_data("data")?;
        let accum = hal.alloc_elem_init(
            "accum",
            total_cycles * CIRCUIT.accum_size(),
            BabyBearElem::INVALID,
        );

        #[cfg(any(feature = "cuda", feature = "rocm"))]
        let pinned_guard = preflight.pinned.lock().unwrap();
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        let pinned = pinned_guard.as_ref();
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        let pinned: Option<&()> = None;
        let raw_trace = RawPreflightTrace {
            wom: pinned_ptr(pinned, 0).unwrap_or(preflight.trace.wom.as_ptr()),
            cycles: pinned_ptr(pinned, 1).unwrap_or(preflight.trace.cycles.as_ptr()),
            iops: pinned_ptr(pinned, 2).unwrap_or(preflight.trace.iops.as_ptr()),
            num_woms: preflight.trace.wom.len() as u32,
            num_cycles: work_cycles,
            num_iops: preflight.trace.iops.len() as u32,
        };

        let host_witness = preflight.host_witness.lock().unwrap().take();
        if let Some((host_data, host_global)) = host_witness {
            if prewitgen_check_due(super::ctrl_cache_key(zkr)) {
                // Independent reference: the normal HAL witgen (device ctrl,
                // device-initialised INVALID data/global) into the real
                // buffers; upload_host_witness below overwrites them with the
                // host witness, which must be identical.
                let t0 = std::time::Instant::now();
                circuit_hal
                    .generate_witness(
                        StepMode::Parallel,
                        total_cycles as u32,
                        &raw_trace,
                        &ctrl,
                        &data,
                        &global,
                    )
                    .context("witness generation failure (prewitgen check)")?;
                let ref_data = data.to_vec();
                let ref_global = global.to_vec();
                if let Some(i) = first_mismatch(&ref_data, &host_data) {
                    anyhow::bail!(
                        "prewitgen witness mismatch in data at elem {i} (col {}, row {})",
                        i / total_cycles,
                        i % total_cycles
                    );
                }
                if let Some(i) = first_mismatch(&ref_global, &host_global) {
                    anyhow::bail!("prewitgen witness mismatch in global at elem {i}");
                }
                eprintln!(
                    "[recursion_prewitgen_check] host witness == HAL witness ({} data + {} global elems, {:.1}ms)",
                    ref_data.len(),
                    ref_global.len(),
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            circuit_hal
                .upload_host_witness(&data, &global, host_data, host_global)
                .context("witness upload failure")?;
        } else {
            // Long chains from `prepare`, or computed here (a few ms of host time that saves
            // ~20 ms of single-thread GPU time).
            let mut chains = preflight.host_chains.lock().unwrap().take();
            if chains.is_none() && cfg!(any(feature = "cuda", feature = "rocm")) {
                chains = host_chains(zkr, preflight);
            } else if let Some(hc) = chains.as_mut() {
                // Upload the chain rows from the page-locked copy, if there is one.
                if let Some(h) = pinned_ptr::<u32>(pinned, 3) {
                    hc.pinned = Some([
                        h as usize,
                        pinned_ptr::<u32>(pinned, 4).unwrap() as usize,
                        pinned_ptr::<u32>(pinned, 5).unwrap() as usize,
                        pinned_ptr::<u32>(pinned, 6).unwrap() as usize,
                        pinned_ptr::<u32>(pinned, 7).unwrap() as usize,
                    ]);
                }
            }
            circuit_hal
                .generate_witness_with_chains(
                    StepMode::Parallel,
                    total_cycles as u32,
                    &raw_trace,
                    chains.as_ref(),
                    &ctrl,
                    &data,
                    &global,
                )
                .context("witness generation failure")?;
            if chains.is_some() && HOST_CHAINS_CHECK.load(Ordering::Relaxed) {
                let ref_data = init_data("data_ref")?;
                let ref_global = hal.copy_from_elem(
                    "global_ref",
                    &vec![BabyBearElem::INVALID; CircuitImpl::OUTPUT_SIZE],
                );
                circuit_hal
                    .generate_witness(
                        StepMode::Parallel,
                        total_cycles as u32,
                        &raw_trace,
                        &ctrl,
                        &ref_data,
                        &ref_global,
                    )
                    .context("witness generation failure (host chains check)")?;
                let (a, b) = (data.to_vec(), ref_data.to_vec());
                if let Some(i) = first_mismatch(&a, &b) {
                    anyhow::bail!(
                        "host chains witness mismatch at col {} row {}",
                        i / total_cycles,
                        i % total_cycles
                    );
                }
                if let Some(i) = first_mismatch(&global.to_vec(), &ref_global.to_vec()) {
                    anyhow::bail!("host chains global mismatch at {i}");
                }
                HOST_CHAINS_CHECKED.fetch_add(1, Ordering::Relaxed);
            }
            if row_passes && *ROW_PASS_CHECK {
                // Validation (RISC0_HIP_REC_ZCHECK=1): the witness from the row passes equals the
                // generic one (all INVALID at the start, zeroized after) up to the noise.
                let t0 = std::time::Instant::now();
                let ref_data = hal.alloc_elem_init(
                    "data_ref",
                    total_cycles * CIRCUIT.data_size(),
                    BabyBearElem::INVALID,
                );
                let ref_global = hal.copy_from_elem(
                    "global_ref",
                    &vec![BabyBearElem::INVALID; CircuitImpl::OUTPUT_SIZE],
                );
                circuit_hal
                    .generate_witness(
                        StepMode::Parallel,
                        total_cycles as u32,
                        &raw_trace,
                        &ctrl,
                        &ref_data,
                        &ref_global,
                    )
                    .context("witness generation failure (row pass check)")?;
                let (a, b) = (data.to_vec(), ref_data.to_vec());
                // Raw words: INVALID is 0xffffffff, zero is 0 (Montgomery form).
                let inv = BabyBearElem::INVALID.as_u32_montgomery();
                let zero = |x: u32| if x == inv { 0 } else { x };
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    let (x, y) = (x.as_u32_montgomery(), y.as_u32_montgomery());
                    let row = i % total_cycles;
                    let got = if row < work_cycles as usize {
                        zero(x)
                    } else {
                        x
                    };
                    if got != zero(y) {
                        anyhow::bail!(
                            "row pass witness mismatch at col {} row {row} ({x:#x} vs {y:#x})",
                            i / total_cycles
                        );
                    }
                }
                if let Some(i) = first_mismatch(&global.to_vec(), &ref_global.to_vec()) {
                    anyhow::bail!("row pass global mismatch at {i}");
                }
                let n = ROW_PASS_CHECKED.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!(
                    "[rec_zcheck] row passes == generic witness #{n} ({} rows, {work_cycles} live, {:.1} ms)",
                    total_cycles,
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
        }

        // Add random noise to end of the data columns
        scope!("noise", {
            use risc0_zkp::ZK_CYCLES;
            let mut rng = rand::rng();
            if row_passes {
                // The same noise: `vec![x; n]` repeats one random element.
                circuit_hal.fill_tail(
                    &data,
                    total_cycles,
                    ZK_CYCLES,
                    BabyBearElem::random(&mut rng),
                )?;
            } else {
                let noise = vec![BabyBearElem::random(&mut rng); ZK_CYCLES * CIRCUIT.data_size()];
                hal.eltwise_copy_elem_slice(
                    &data,
                    &noise,
                    CIRCUIT.data_size(),      // from_rows
                    ZK_CYCLES,                // from_cols
                    0,                        // from_offset
                    ZK_CYCLES,                // from_stride
                    total_cycles - ZK_CYCLES, // into_offset
                    total_cycles,             // into_stride
                );
            }
        });

        // Zero out 'invalid' entries in data and output.
        scope!("zeroize", {
            if row_passes {
                circuit_hal.zeroize_rows(&data, total_cycles, work_cycles as usize)?;
            } else {
                hal.eltwise_zeroize_elem(&data);
            }
            hal.eltwise_zeroize_elem(&global);
        });

        Ok(Self {
            work_cycles,
            total_cycles: total_cycles as u32,
            global,
            ctrl,
            data,
            accum,
        })
    }

    pub fn accum<C: CircuitAccumulator<H> + CircuitWitnessGenerator<H>>(
        &self,
        hal: &H,
        circuit_hal: &C,
        mix: &[BabyBearElem],
    ) -> Result<H::Buffer<H::Elem>> {
        scope!("accum");

        let mix = hal.copy_from_elem("mix", mix);

        // Add random noise to end of accum
        scope!("noise", {
            use risc0_zkp::ZK_CYCLES;
            let mut rng = rand::rng();
            let total_cycles = self.total_cycles as usize;
            if circuit_hal.has_row_passes() {
                // The same noise: `vec![x; n]` repeats one random element.
                circuit_hal.fill_tail(
                    &self.accum,
                    total_cycles,
                    ZK_CYCLES,
                    BabyBearElem::random(&mut rng),
                )?;
            } else {
                let noise = vec![BabyBearElem::random(&mut rng); ZK_CYCLES * CIRCUIT.accum_size()];
                hal.eltwise_copy_elem_slice(
                    &self.accum,
                    &noise,
                    CIRCUIT.accum_size(),     // from_rows
                    ZK_CYCLES,                // from_cols
                    0,                        // from_offset
                    ZK_CYCLES,                // from_stride
                    total_cycles - ZK_CYCLES, // into_offset
                    total_cycles,             // into_stride
                );
            }
        });

        circuit_hal.accumulate(
            self.work_cycles,
            self.total_cycles,
            &self.ctrl,
            &self.global,
            &self.data,
            &mix,
            &self.accum,
        )?;

        scope!("zeroize", {
            hal.eltwise_zeroize_elem(&self.accum);
            hal.eltwise_zeroize_elem(&self.global);
        });

        Ok(mix)
    }
}

/// Run the CPU witness generator entirely on the host (no device buffers).
/// Bit-identical to the HAL path: data/global start as INVALID, ctrl is the
/// transposed program code, exactly as WitnessGenerator::new sets them up.
#[cfg(feature = "intel")]
pub(crate) fn host_witgen(
    zkr: &Program,
    preflight: &Preflight,
) -> Result<(Vec<BabyBearElem>, Vec<BabyBearElem>)> {
    use risc0_circuit_recursion_sys::{risc0_circuit_recursion_cpu_witgen, RawExecBuffers};
    let total_cycles = 1usize << zkr.po2;
    // Exact program identity (key + full validation), shared with the HAL path.
    let ctrl = host_ctrl_for(zkr);
    // ~128 MB: fill in parallel (a serial fill is ~30 ms on the prepare thread).
    let n = total_cycles * CIRCUIT.data_size();
    let mut data: Vec<BabyBearElem> = Vec::with_capacity(n);
    {
        use rayon::prelude::*;
        data.spare_capacity_mut()
            .par_chunks_mut(1 << 16)
            .for_each(|c| {
                c.iter_mut().for_each(|x| {
                    x.write(BabyBearElem::INVALID);
                })
            });
        // SAFETY: every element of the spare capacity was initialised above.
        unsafe { data.set_len(n) };
    }
    let mut global = vec![BabyBearElem::INVALID; CircuitImpl::OUTPUT_SIZE];
    // The C++ witgen consumes the trace (extern.cpp: `trace->cycles[c].iopIdx++`),
    // so run it on a private copy of `cycles` (~2 MB). This keeps
    // `preflight.trace` pristine, which (a) lets the prewitgen check re-run the
    // HAL witgen on the same preflight as an independent reference and (b)
    // avoids mutating through a pointer derived from `&Preflight`.
    let mut cycles = preflight.trace.cycles.clone();
    let raw_trace = RawPreflightTrace {
        wom: preflight.trace.wom.as_ptr(),
        cycles: cycles.as_mut_ptr(),
        iops: preflight.trace.iops.as_ptr(),
        num_woms: preflight.trace.wom.len() as u32,
        num_cycles: zkr.code_rows() as u32,
        num_iops: preflight.trace.iops.len() as u32,
    };
    let buffers = RawExecBuffers {
        ctrl: ctrl.as_ptr(),
        data: data.as_mut_ptr(),
        global: global.as_mut_ptr(),
    };
    risc0_sys::ffi_wrap(|| unsafe {
        risc0_circuit_recursion_cpu_witgen(
            StepMode::Parallel,
            &buffers,
            &raw_trace,
            total_cycles as u32,
        )
    })?;
    Ok((data, global))
}
