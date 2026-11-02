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

use std::{collections::HashMap, path::Path, sync::OnceLock};

use anyhow::{Context as _, Result};
use circom_witnesscalc::deserialize_inputs2;
use circom_witnesscalc::field::U254;
use circom_witnesscalc::graph::{
    evaluate, Node, Nodes, NodesInterface, NodesStorage, Operation, VecNodes,
};
use circom_witnesscalc::storage::proto_deserializer::deserialize_witnesscalc_graph_from_bytes;
use circom_witnesscalc::InputSignalsInfo;
use risc0_groth16_sys::{ProverParams, SetupParams, WitnessParams};
use rzup::{Component, Rzup, Version};
use tempfile::tempdir;

use crate::{ProofJson, Seal};

use super::seal_to_json::to_json;

/// Cached rzup root directory path.
static ROOT_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

fn get_root_dir() -> Result<&'static Path> {
    let dir = ROOT_DIR.get_or_init(|| {
        // `RISC0_GROTH16_DIR` names the component's directory outright, bypassing rzup:
        // rzup does not run on Windows (`Platform::detect` knows only Linux and macOS),
        // so there the files are installed by hand and located through this.
        if let Some(dir) = std::env::var_os("RISC0_GROTH16_DIR") {
            return dir.into();
        }
        let version = Version::new(0, 1, 0);
        Rzup::new()
            .and_then(|r| r.get_version_dir(&Component::Risc0Groth16, &version))
            .ok()
            .or_else(|| rzup_layout_dir(&version))
            .expect(
                "Missing required `risc0-groth16` rzup component. \
                To install it, ensure that your `rzup` version is >= 0.5.0, \
                and then run `rzup install risc0-groth16`. \
                Where rzup does not run (Windows), copy the component's files to \
                ~/.risc0/extensions/v0.1.0-risc0-groth16, or set RISC0_GROTH16_DIR \
                to a directory holding them.",
            )
    });
    Ok(dir.as_path())
}

/// Where rzup installs the component, when it is there: `$RISC0_HOME` (else
/// `~/.risc0`) `/extensions/v<version>-risc0-groth16`. For hosts where `Rzup::new`
/// fails outright, such as Windows, which its `Platform::detect` does not know, so
/// that a copy installed by hand in the same place is found the same way.
fn rzup_layout_dir(version: &Version) -> Option<std::path::PathBuf> {
    let risc0_home = std::env::var_os("RISC0_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(|home| std::path::PathBuf::from(home).join(".risc0"))
        })?;
    let dir = risc0_home
        .join("extensions")
        .join(format!("v{version}-risc0-groth16"));
    dir.is_dir().then_some(dir)
}

/// Cached deserialized circom graph to avoid re-parsing ~241MB on each call (~554ms).
struct CachedGraph {
    nodes: Box<dyn NodesInterface>,
    signals: Vec<usize>,
    input_mapping: InputSignalsInfo,
    /// The graph compiled for `witness_fast` (None if it doesn't qualify).
    fast: Option<super::witness_fast::Program>,
    /// The parallel evaluator (None if the cold-path switch is off or it doesn't qualify).
    par: Option<super::witness_par::ParProgram>,
}

// Safety: The underlying Nodes<U254, VecNodes> contains Vec<Node>, Field<U254>, HashMap —
// all Send+Sync. The graph is written once (OnceLock init) and read immutably after.
unsafe impl Send for CachedGraph {}
unsafe impl Sync for CachedGraph {}

static CACHED_GRAPH: OnceLock<CachedGraph> = OnceLock::new();

/// Witness indices whose signal is `x & 1` (a Band node with the constant 1), i.e.
/// always 0 or 1: the bit set S of the split MSMs. Derived from the graph itself (no index
/// file); the prover still checks every S scalar at prove time and falls back to the full MSMs
/// if one is not 0/1, so S only affects speed, never the proof.
static BIT_SIGNALS: OnceLock<Vec<u32>> = OnceLock::new();

fn bit_signals(nodes: &Nodes<U254, VecNodes>, signals: &[usize]) -> Vec<u32> {
    let one = U254::from(1u64);
    // Wire 0 is the constant 1 and wires 1..=n_public the public signals: never in S (the CUDA
    // library drops such indices too, and any out of range or repeated).
    let num_public = crate::verifying_key()
        .0
        .gamma_abc_g1
        .len()
        .saturating_sub(1);
    let mut out = Vec::new();
    for (wi, &si) in signals.iter().enumerate() {
        if wi <= num_public || wi > u32::MAX as usize {
            continue;
        }
        if let Some(Node::Op(Operation::Band, _a, b)) = nodes.nodes.get(si) {
            if let Some(Node::Constant(ci)) = nodes.nodes.get(b) {
                if nodes.constants[ci] == one {
                    out.push(wi as u32);
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn get_cached_graph(graph_path: &Path) -> &'static CachedGraph {
    CACHED_GRAPH.get_or_init(|| {
        let graph_bytes = std::fs::read(graph_path).expect("failed to read circom graph file");
        let (nodes, signals, input_mapping) =
            deserialize_witnesscalc_graph_from_bytes(&graph_bytes)
                .expect("failed to deserialize circom graph");
        // The bit set and the two evaluator compiles only read the graph: run them side by side
        // (witness_par::compile alone is ~0.7 s, and the graph is on the Groth16 critical path
        // when it is not ready by the end of the STARK).
        let vn = nodes.as_any().downcast_ref::<Nodes<U254, VecNodes>>();
        let (bits, fast, par) = std::thread::scope(|s| {
            let par = s.spawn(|| {
                if !cold_path_enabled() {
                    return None;
                }
                let t = std::time::Instant::now();
                let par = vn.and_then(|nodes| super::witness_par::compile(nodes, &signals));
                tracing::debug!(
                    "groth16: witness_par compile {:.1}ms ok={}",
                    t.elapsed().as_secs_f64() * 1e3,
                    par.is_some()
                );
                par
            });
            let bits = s.spawn(|| {
                vn.map(|nodes| bit_signals(nodes, &signals))
                    .unwrap_or_default()
            });
            let fast = vn.and_then(|nodes| super::witness_fast::compile(nodes, &signals));
            (
                bits.join().expect("groth16 bit-set derivation panicked"),
                fast,
                par.join().expect("groth16 witness_par compile panicked"),
            )
        });
        let _ = BIT_SIGNALS.set(bits);
        if fast.is_none() {
            tracing::warn!("groth16: the fast witness evaluator does not support this graph");
        }
        CachedGraph {
            nodes,
            signals,
            input_mapping,
            fast,
            par,
        }
    })
}

/// Preload and cache the deserialized circom graph for witness calculation.
/// Call early (e.g. on a background thread) to overlap the ~554ms graph
/// deserialization with other work (e.g. STARK proving).
pub(crate) fn preload_graph() -> Result<()> {
    let root_dir = get_root_dir()?;
    let witness_params = WitnessParams::new(root_dir);
    let _ = get_cached_graph(&witness_params.graph_path);
    Ok(())
}

/// Preload SRS and prover into C++ static cache.
/// Must be called when GPU is idle (after STARK proving, before shrink_wrap).
pub(crate) fn preload_srs() -> Result<()> {
    let root_dir = get_root_dir()?;
    let setup_params =
        SetupParams::new(root_dir).context("failed to create groth16 setup params for preload")?;
    risc0_groth16_sys::preload(&setup_params).context("failed to preload groth16 SRS")?;
    Ok(())
}

/// The Groth16 cold-path optimizations: background proving-key load during the STARK
/// (`prestart_key`), early witness-graph parse, compact coefficients, bit-split MSMs, tuned MSM
/// batching and the parallel witness evaluator. On by default; `RISC0_CUDA_G16_COLD=0` disables
/// all of them (the same switch is read by the CUDA library).
pub(crate) fn cold_path_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("RISC0_CUDA_G16_COLD").as_deref() != Ok("0"));
    *ON
}

/// The Groth16 tail: the key loader's pinned buffers allocated at session start
/// (`pin_early`), the witness computed beside `shrink_wrap_with`'s `pre_gpu`, and, in the CUDA
/// library, the deferred-collect / 4-window sort / split-integrate MSMs. The zkvm prover also
/// verifies the succinct receipt beside the Groth16 wrap. On by default; `RISC0_CUDA_G16_TAIL=0`
/// disables all of them (the same switch is read by the CUDA library and the zkvm prover).
pub(crate) fn tail_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("RISC0_CUDA_G16_TAIL").as_deref() != Ok("0"));
    *ON
}

/// Minimum device memory for `prestart_key`: the key sits beside a STARK proof (peak
/// 27.0-27.5 GB at po2=21 with the key loaded).
const PRESTART_MIN_VRAM: usize = 30 << 30;

/// Start loading the proving key on a background thread (pinned staging, own streams), then
/// pre-allocate the pinned witness buffer, so that neither is on the shrink_wrap path. Meant to
/// be called when a Groth16 proof is about to be produced. The loader waits for
/// `prestart_key_go` (segment 0's GPU work queued), at most 2 s. Skipped on cards with less than
/// `PRESTART_MIN_VRAM` or with the cold-path switch off.
pub(crate) fn prestart_key() {
    static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !cold_path_enabled() || STARTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    // The bit set S (split MSMs) comes from the parsed graph: make sure a parse is running.
    std::thread::spawn(|| {
        let _ = preload_graph();
    });
    std::thread::spawn(|| {
        // The pinned allocations (the key loader's upload stager and the witness buffer) come
        // first, at time zero while the first segment is still host-bound: each pin holds the
        // CUDA driver lock and, landing in a GPU-bound segment, stalls the STARK's launches.
        let pinned_early = pin_early();
        // Let the first STARK segment's cold allocations and uploads go first.
        {
            let (m, cv) = &PRESTART_GO;
            let g = m.lock().unwrap();
            let _ = cv
                .wait_timeout_while(g, std::time::Duration::from_secs(2), |go| !*go)
                .unwrap();
        }
        let t = std::time::Instant::now();
        // S must be set before the prover is built (its constructor permutes the SRS).
        while BIT_SIGNALS.get().is_none() && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        if let Some(bits) = BIT_SIGNALS.get() {
            risc0_groth16_sys::set_bit_signals(bits);
        }
        let t_bits = t.elapsed();
        let r = (|| -> Result<Option<usize>> {
            let root_dir = get_root_dir()?;
            let setup = SetupParams::new(root_dir)?;
            let len = risc0_groth16_sys::prestart(&setup, PRESTART_MIN_VRAM)?;
            if let Some(len) = len {
                witness_len_cache_write(len);
                // Pin the witness buffer now unless pin_early already pinned this size.
                if pinned_early != Some(len) {
                    drop(risc0_groth16_sys::host_buffer(len * 32)?);
                }
            }
            Ok(len)
        })();
        // The key isn't prestarted (small card, missing files, error): don't keep what
        // pin_early allocated for it; the wrap allocates on demand as without the prestart.
        if pinned_early.is_some() && !matches!(r, Ok(Some(_))) {
            risc0_groth16_sys::pinned_release();
        }
        tracing::debug!(
            "groth16: prestart_key done in {:.1}ms (bit set wait {:.1}ms): {:?}",
            t.elapsed().as_secs_f64() * 1e3,
            t_bits.as_secs_f64() * 1e3,
            r.as_ref().map_err(|e| format!("{e:#}"))
        );
        if let Err(e) = r {
            tracing::warn!("groth16 key prestart failed (non-fatal): {e:#}");
        }
    });
}

/// Create the key loader's pinned upload stager and pin the witness buffer at the length cached
/// by the previous run (none on a first-ever run: then the loader pins it once the key is loaded,
/// as before). Returns the witness length pinned, `Some(0)` if only the stager was created, and
/// `None` if nothing was allocated (tail switch off, a card below `PRESTART_MIN_VRAM`, or an
/// error).
fn pin_early() -> Option<usize> {
    if !tail_enabled() {
        return None;
    }
    let t = std::time::Instant::now();
    match risc0_groth16_sys::stager_init(PRESTART_MIN_VRAM) {
        Ok(true) => {}
        Ok(false) => return None,
        Err(e) => {
            tracing::warn!("groth16 stager init failed (non-fatal): {e:#}");
            return None;
        }
    }
    let len = witness_len_cache_read()
        .filter(|&len| risc0_groth16_sys::host_buffer(len * 32).is_ok())
        .unwrap_or(0);
    tracing::debug!(
        "groth16: pin_early {:.1}ms, witness buffer pinned for {len} signals",
        t.elapsed().as_secs_f64() * 1e3
    );
    Some(len)
}

/// The Groth16 cache directory, the CUDA library's (kernels/cuda/disk_cache.cuh):
/// RISC0_G16_CACHE_DIR, else $XDG_CACHE_HOME/risc0-groth16, else ~/.cache/risc0-groth16.
fn cache_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RISC0_G16_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CACHE_HOME")
                .map(|x| std::path::PathBuf::from(x).join("risc0-groth16"))
        })
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|h| std::path::PathBuf::from(h).join(".cache/risc0-groth16"))
        })
}

/// `witness_len` in the Groth16 cache directory.
fn witness_len_cache_path() -> Option<std::path::PathBuf> {
    Some(cache_dir()?.join("witness_len"))
}

/// Delete the CUDA library's derived-data caches (compact coefficients, split-MSM corrections)
/// and drop the prover built from them, so the next wrap rebuilds both from the proving key.
/// See `prove::invalidate_caches`.
pub(crate) fn invalidate_caches() {
    if let Err(e) = release_srs() {
        tracing::warn!("groth16: release after a failed seal check: {e:#}");
    }
    let Some(dir) = cache_dir() else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if (name.starts_with("coeffs_compact_") || name.starts_with("split_corr_"))
            && name.ends_with(".bin")
        {
            match std::fs::remove_file(entry.path()) {
                Ok(()) => tracing::warn!("groth16: removed cache file {name}"),
                Err(e) => tracing::warn!("groth16: could not remove cache file {name}: {e}"),
            }
        }
    }
}

fn witness_len_cache_read() -> Option<usize> {
    let s = std::fs::read_to_string(witness_len_cache_path()?).ok()?;
    let v: usize = s.trim().parse().ok()?;
    // At most 2^28 signals (8 GB of witness): anything else is not ours.
    (v > 0 && v < (1 << 28)).then_some(v)
}

/// Write `len` through a temporary file and a rename, so concurrent processes never read a
/// partial value. A stale value only costs a regrow.
fn witness_len_cache_write(len: usize) {
    if witness_len_cache_read() == Some(len) {
        return;
    }
    let Some(path) = witness_len_cache_path() else {
        return;
    };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, format!("{len}\n")).is_ok() && std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

static PRESTART_GO: (std::sync::Mutex<bool>, std::sync::Condvar) =
    (std::sync::Mutex::new(false), std::sync::Condvar::new());

/// Release the `prestart_key` loader (idempotent).
pub(crate) fn prestart_key_go() {
    let (m, cv) = &PRESTART_GO;
    let mut g = m.lock().unwrap();
    if !*g {
        *g = true;
        cv.notify_all();
    }
}

/// Drop the prover and SRS that `preload_srs` and `shrink_wrap` cached on the GPU. See
/// `prove::release_srs`.
pub(crate) fn release_srs() -> Result<()> {
    risc0_groth16_sys::release().context("failed to release the cached groth16 SRS")
}

pub(crate) fn shrink_wrap(seal_bytes: &[u8]) -> Result<Seal> {
    shrink_wrap_with(seal_bytes, || {})
}

/// [shrink_wrap], running `pre_gpu` on the calling thread while the witness is computed on a
/// scoped helper thread (with the tail switch off, `pre_gpu` runs first). `pre_gpu` has returned
/// before the GPU part of the wrap starts.
pub(crate) fn shrink_wrap_with(seal_bytes: &[u8], pre_gpu: impl FnOnce()) -> Result<Seal> {
    if !tail_enabled() {
        pre_gpu();
        return shrink_wrap_inner(seal_bytes, || {});
    }
    shrink_wrap_inner(seal_bytes, pre_gpu)
}

fn shrink_wrap_inner(seal_bytes: &[u8], pre_gpu: impl FnOnce()) -> Result<Seal> {
    tracing::info!("shrink_wrap: {} seal bytes", seal_bytes.len());

    let root_dir = get_root_dir()?;
    let tmp_dir = tempdir().context("failed to create temporary directory")?;
    let work_dir = std::env::var("RISC0_WORK_DIR");
    let work_dir = work_dir.as_ref().map(Path::new).unwrap_or(tmp_dir.path());
    let setup_params =
        SetupParams::new(root_dir).context("failed to create groth16 work directories")?;

    let inputs = to_json(seal_bytes)?;

    // Start SRS preload on background thread while witness calc runs on CPU.
    // SRS loading (disk I/O + GPU upload) overlaps with witness computation.
    let srs_preload = {
        let setup_clone = SetupParams::new(root_dir)
            .context("failed to create groth16 setup params for preload")?;
        std::thread::spawn(move || {
            let _ = risc0_groth16_sys::preload(&setup_clone);
        })
    };

    let witness_params = WitnessParams::new(root_dir);
    // The graph (parsed in the background since the session started) is waited for on the
    // helper thread too, so `pre_gpu` also overlaps that wait.
    let witness = std::thread::scope(|sc| {
        let h = sc.spawn(|| calc_witness(get_cached_graph(&witness_params.graph_path), &inputs));
        pre_gpu();
        h.join()
            .unwrap_or_else(|_| Err(anyhow::anyhow!("groth16 witness thread panicked")))
    })
    .context("failed to calculate groth16 witness")?;

    // Wait for SRS preload to complete before taking the GPU lock.
    let _ = srs_preload.join();

    {
        let _lock = risc0_zkp::hal::cuda::singleton().lock();

        let prover_params = ProverParams::new(work_dir, witness.as_ptr())
            .context("failed to create groth16 prover parameters")?;
        risc0_groth16_sys::prove(&prover_params, &setup_params)
            .context("failed to run groth16 prove operation")?;

        let contents = std::fs::read_to_string(prover_params.proof_path.as_path())
            .context("failed to read groth16 prove output file")?;
        let proof_json: ProofJson = serde_json::from_str(&contents)
            .context("failed to decode groth16 prove output file JSON")?;
        proof_json.try_into()
    }
}

enum CalcWitness {
    Vec(Vec<wtns_file::FieldElement<32>>),
    /// Canonical little-endian values in the pinned buffer, held until the prove has read it.
    Pinned(risc0_groth16_sys::HostBuffer),
}

impl CalcWitness {
    fn as_ptr(&self) -> *const u8 {
        match self {
            CalcWitness::Vec(w) => w.as_ptr() as *const u8,
            CalcWitness::Pinned(buf) => buf.as_ptr(),
        }
    }
}

fn calc_witness(graph: &CachedGraph, inputs: &str) -> Result<CalcWitness> {
    let nodes = graph
        .nodes
        .as_any()
        .downcast_ref::<Nodes<U254, VecNodes>>()
        .ok_or_else(|| anyhow::anyhow!("unexpected graph node type"))?;

    // Parse inputs into field elements.
    let input_list: HashMap<String, Vec<U254>> = deserialize_inputs2(inputs.as_bytes(), &nodes.ff)
        .map_err(|e| anyhow::anyhow!("failed to deserialize inputs: {e}"))?;

    // Map parsed inputs to the input index array.
    let mut max_idx: usize = 0;
    for (offset, len) in graph.input_mapping.values() {
        let idx = offset + len;
        if idx > max_idx {
            max_idx = idx;
        }
    }
    let mut input_vec = vec![U254::ZERO; max_idx + 1];
    input_vec[0] = U254::from(1u64);
    for (key, value) in &input_list {
        let (offset, len) = graph.input_mapping[key];
        if len != value.len() {
            anyhow::bail!("Invalid input length for {}", key);
        }
        for (i, v) in value.iter().enumerate() {
            input_vec[offset + i] = *v;
        }
    }

    if let Some(program) = graph.par.as_ref() {
        let mut buf = risc0_groth16_sys::host_buffer(graph.signals.len() * 32)
            .context("failed to allocate the pinned witness buffer")?;
        super::witness_par::eval(program, nodes, &input_vec, buf.as_limbs_mut());
        return Ok(CalcWitness::Pinned(buf));
    }
    if let Some(program) = graph.fast.as_ref() {
        let mut buf = risc0_groth16_sys::host_buffer(graph.signals.len() * 32)
            .context("failed to allocate the pinned witness buffer")?;
        super::witness_fast::eval(program, nodes, &input_vec, buf.as_limbs_mut());
        return Ok(CalcWitness::Pinned(buf));
    }

    // Evaluate the witness graph directly (skip WTNS encode/decode roundtrip).
    // Temporarily redirect stdout to suppress circom_witnesscalc's println! calls
    // which add ~100-500ms of terminal I/O overhead.
    let result: Vec<U254> = {
        #[cfg(unix)]
        let _guard = suppress_stdout();
        evaluate(
            &nodes.ff,
            &nodes.nodes,
            &input_vec,
            &graph.signals,
            &nodes.constants,
        )
    };

    // Convert U254 values to FieldElement<32> (both are 32-byte little-endian).
    let witness: Vec<wtns_file::FieldElement<32>> = result
        .iter()
        .map(|a| {
            TryInto::<[u8; 32]>::try_into(a.as_le_slice())
                .unwrap()
                .into()
        })
        .collect();

    Ok(CalcWitness::Vec(witness))
}

/// RAII guard that redirects stdout to /dev/null and restores on drop.
/// Suppresses println! calls from third-party libraries during critical sections.
#[cfg(unix)]
struct StdoutGuard {
    saved_fd: std::os::unix::io::RawFd,
}

#[cfg(unix)]
impl Drop for StdoutGuard {
    fn drop(&mut self) {
        unsafe {
            extern "C" {
                fn dup2(oldfd: i32, newfd: i32) -> i32;
                fn close(fd: i32) -> i32;
            }
            dup2(self.saved_fd, 1);
            close(self.saved_fd);
        }
    }
}

#[cfg(unix)]
fn suppress_stdout() -> StdoutGuard {
    unsafe {
        extern "C" {
            fn dup(fd: i32) -> i32;
            fn dup2(oldfd: i32, newfd: i32) -> i32;
        }
        let saved = dup(1);
        if let Ok(devnull) = std::fs::File::open("/dev/null") {
            use std::os::unix::io::AsRawFd;
            dup2(devnull.as_raw_fd(), 1);
        }
        StdoutGuard { saved_fd: saved }
    }
}
