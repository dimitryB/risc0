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

use std::{path::Path, sync::OnceLock};

use anyhow::{anyhow, Context as _, Result};
use circom_witnesscalc::deserialize_inputs2;
use circom_witnesscalc::field::FieldOperations;
use circom_witnesscalc::field::U254;
use circom_witnesscalc::graph::{Node, Nodes, NodesInterface, NodesStorage, VecNodes};
use circom_witnesscalc::storage::proto_deserializer::deserialize_witnesscalc_graph_from_bytes;
use circom_witnesscalc::InputSignalsInfo;
use num_traits::{One, Zero};
use rayon::prelude::*;
use risc0_core::field::baby_bear::BabyBearElem;
use risc0_groth16_sys::{ProverParams, SetupParams, WitnessParams};
use risc0_zkp::core::{
    digest::{Digest, DIGEST_WORDS},
    hash::poseidon_254::digest_to_fr_bytes,
};
use tempfile::tempdir;

use crate::{ProofJson, Seal};

use super::seal_format::{IopType, K_SEAL_ELEMS, K_SEAL_TYPES, K_SEAL_WORDS};

/// Preload and cache the circom graph for witness calculation, plus
/// preload the SRS and prover on the GPU.
/// Call this early (e.g. on a background thread) to overlap the ~1.2s graph
/// read+parse and ~1.2s SRS load with segment proving.
pub(crate) fn preload_graph() -> Result<()> {
    let root_dir = super::component::component_dir()?;
    let witness_params = WitnessParams::new(&root_dir);
    let pg = PARSED_GRAPH.get_or_init(|| load_graph(&witness_params.graph_path));

    // Also compile the graph for the fast evaluator, or failing that precompute BFS level data
    // for parallel evaluation (~570ms). This runs on the background thread during segment
    // proving.
    let nodes = pg
        .nodes
        .as_any()
        .downcast_ref::<Nodes<U254, VecNodes>>()
        .expect("expected U254 nodes in circom graph");
    if fast_program(pg, nodes).is_none() {
        pg.level_data
            .get_or_init(|| compute_level_data(&nodes.nodes));
    }

    Ok(())
}

fn load_graph(graph_path: &Path) -> ParsedGraph {
    let t0 = std::time::Instant::now();
    let graph_bytes = std::fs::read(graph_path).expect("failed to read circom graph file");
    let t1 = std::time::Instant::now();
    let (nodes, signals, input_mapping) = deserialize_witnesscalc_graph_from_bytes(&graph_bytes)
        .expect("failed to parse circom graph");
    let t2 = std::time::Instant::now();
    eprintln!(
        "[groth16_witness] graph cached: read={:.1}ms parse={:.1}ms",
        (t1 - t0).as_secs_f64() * 1000.0,
        (t2 - t1).as_secs_f64() * 1000.0,
    );
    ParsedGraph {
        nodes,
        signals,
        input_mapping,
        level_data: OnceLock::new(),
        fast: OnceLock::new(),
    }
}

/// The graph compiled for `witness_fast` (as on CUDA), or None if it doesn't qualify.
fn fast_program<'a>(
    pg: &'a ParsedGraph,
    nodes: &Nodes<U254, VecNodes>,
) -> Option<&'a super::witness_fast::Program> {
    pg.fast
        .get_or_init(|| {
            let fast = super::witness_fast::compile(nodes, &pg.signals);
            if fast.is_none() {
                tracing::warn!("groth16: the fast witness evaluator does not support this graph");
            }
            fast
        })
        .as_ref()
}

/// Convert seal bytes directly to a Vec of U254 values, bypassing JSON serialization.
/// This produces the same values as `to_json` + `deserialize_inputs2` but ~10x faster
/// by avoiding string formatting, JSON generation, and JSON parsing.
fn seal_to_u254_values(seal_bytes: &[u8]) -> Result<Vec<U254>> {
    let mut iop = vec![0u32; K_SEAL_WORDS];
    iop.copy_from_slice(bytemuck::cast_slice(seal_bytes));

    let mut values = Vec::with_capacity(K_SEAL_ELEMS);
    let mut pos = 0;
    for seal_type in K_SEAL_TYPES.iter().take(K_SEAL_ELEMS) {
        match seal_type {
            IopType::Fp => {
                let value = BabyBearElem::new_raw(iop[pos]).as_u32();
                pos += 1;
                values.push(U254::from(value as u64));
            }
            _ => {
                let digest = Digest::try_from(&iop[pos..pos + DIGEST_WORDS])?;
                pos += DIGEST_WORDS;
                // Convert digest → Fr field element → canonical LE bytes → U254
                let bytes = digest_to_fr_bytes(&digest);
                values.push(U254::from_le_bytes(bytes));
            }
        }
    }
    Ok(values)
}

/// Free the Groth16 prover and SRS that `shrink_wrap` caches on the GPU (the same cache as on CUDA,
/// see `risc0_groth16_sys::release`). Without this they hold ~7 GB of an RX 9070 XT (16 GB) for the
/// life of the process, and on Windows the next proof's STARK spills into shared system memory.
pub(crate) fn release_srs() -> Result<()> {
    risc0_groth16_sys::release().context("failed to release the cached groth16 SRS")
}

pub(crate) fn shrink_wrap(seal_bytes: &[u8]) -> Result<Seal> {
    let t_total = std::time::Instant::now();
    tracing::info!("shrink_wrap: {} seal bytes", seal_bytes.len());

    let t0 = std::time::Instant::now();
    let root_dir = super::component::component_dir()?;
    let setup_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let tmp_dir = tempdir().context("failed to create temporary directory")?;
    let work_dir = std::env::var("RISC0_WORK_DIR");
    let work_dir = work_dir.as_ref().map(Path::new).unwrap_or(tmp_dir.path());
    let setup_params =
        SetupParams::new(&root_dir).context("failed to create groth16 work directories")?;

    let t0 = std::time::Instant::now();
    let iop_values = seal_to_u254_values(seal_bytes)?;
    let seal_convert_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // RISC0_GROTH16_DEVICE overrides groth16 device. Otherwise device 0.
    let groth16_device: i32 = std::env::var("RISC0_GROTH16_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let t0 = std::time::Instant::now();
    if groth16_device == 0 {
        let _lock = risc0_zkp::hal::hip::singleton().lock();
        risc0_zkp::hal::hip::clear_buffer_pool();
    } else {
        risc0_zkp::hal::hip::switch_to_device(groth16_device);
    }
    let clear_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Start SRS preloading on background thread (GPU work, ~1.2s)
    let setup_params_clone =
        SetupParams::new(&root_dir).context("failed to create groth16 setup params for preload")?;
    let preload_handle = std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        risc0_zkp::hal::hip::switch_to_device(groth16_device);
        let result = risc0_groth16_sys::preload(&setup_params_clone);
        (result, t0.elapsed().as_secs_f64() * 1000.0)
    });

    // Compute circom witness on CPU (~2s) — overlaps with SRS preload
    let t0 = std::time::Instant::now();
    let witness_params = WitnessParams::new(&root_dir);
    let witness = calc_witness_binary(&witness_params.graph_path, iop_values)
        .context("failed to calculate groth16 witness")?;
    let witness_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Wait for SRS preload to finish
    let t0 = std::time::Instant::now();
    let (preloaded, preload_ms) = preload_handle
        .join()
        .map_err(|_| anyhow::anyhow!("SRS preload thread panicked"))?;
    preloaded.context("SRS preload failed")?;
    let preload_wait_ms = t0.elapsed().as_secs_f64() * 1000.0;

    {
        let _lock = risc0_zkp::hal::hip::singleton_for_device(groth16_device).lock();

        let prover_params = ProverParams::new(work_dir, witness.as_ptr())
            .context("failed to create groth16 prover parameters")?;
        let t0 = std::time::Instant::now();
        risc0_groth16_sys::prove(&prover_params, &setup_params)
            .context("failed to run groth16 prove operation")?;
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = std::time::Instant::now();
        let contents = std::fs::read_to_string(prover_params.proof_path.as_path())
            .context("failed to read groth16 prove output file")?;
        let read_ms = t0.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "[groth16_shrink_wrap] rzup={setup_ms:.1}ms seal_convert={seal_convert_ms:.1}ms \
             clear_pool={clear_ms:.1}ms witness={witness_ms:.1}ms preload={preload_ms:.1}ms \
             preload_wait={preload_wait_ms:.1}ms gpu_prove={prove_ms:.1}ms read_proof={read_ms:.1}ms \
             device={groth16_device} total={:.1}ms",
            t_total.elapsed().as_secs_f64() * 1000.0,
        );

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

/// Precomputed BFS-level data for parallel DAG evaluation.
/// Cached because the circuit graph is always the same.
struct LevelData {
    /// Snapshot of all nodes for fast indexed access.
    all_nodes: Vec<Node>,
    /// For each BFS level, the range of indices into `level_indices`.
    level_offsets: Vec<usize>,
    /// Node indices grouped by BFS level (flat array, sliced by level_offsets).
    level_indices: Vec<usize>,
    /// Total number of BFS levels.
    num_levels: usize,
}

/// Cached parsed circom graph. Parsing the 241MB graph file takes ~732ms;
/// caching the parsed result eliminates this cost on repeated calls.
/// Also skips the WTNS encode/decode round-trip on every evaluation.
struct ParsedGraph {
    nodes: Box<dyn NodesInterface>,
    signals: Vec<usize>,
    input_mapping: InputSignalsInfo,
    level_data: OnceLock<LevelData>,
    fast: OnceLock<Option<super::witness_fast::Program>>,
}

// SAFETY: The concrete type behind `dyn NodesInterface` is `Nodes<U254, VecNodes>`,
// which is Send+Sync (all fields: String, Field<U254>, Vec<Node>, Vec<U254>,
// HashMap<U254, usize>). The trait just lacks Send+Sync supertraits.
unsafe impl Send for ParsedGraph {}
unsafe impl Sync for ParsedGraph {}

static PARSED_GRAPH: OnceLock<ParsedGraph> = OnceLock::new();

/// Wrapper to allow sending a raw pointer across rayon threads.
/// SAFETY: The caller must ensure no data races (non-overlapping writes per thread).
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// Read value at offset. SAFETY: caller must ensure offset is valid and data is initialized.
    #[inline]
    unsafe fn read_at(&self, offset: usize) -> T {
        self.0.add(offset).read()
    }
    /// Write value at offset. SAFETY: caller must ensure offset is valid and non-overlapping.
    #[inline]
    unsafe fn write_at(&self, offset: usize, val: T) {
        self.0.add(offset).write(val);
    }
}

/// Compute BFS-level data for a circom DAG. This is deterministic for a given graph
/// and is cached alongside the parsed graph via OnceLock.
fn compute_level_data(ns: &VecNodes) -> LevelData {
    let t0 = std::time::Instant::now();
    let n = ns.len();

    // Snapshot all nodes for fast indexed access.
    let all_nodes: Vec<Node> = (0..n).map(|i| ns.get(i).unwrap()).collect();
    let t_snapshot = std::time::Instant::now();

    // Compute BFS level for each node.
    let mut levels = vec![0u32; n];
    let mut max_level: u32 = 0;
    for i in 0..n {
        let lv = match all_nodes[i] {
            Node::Input(_) | Node::Constant(_) | Node::Unknown => 0,
            Node::UnoOp(_, a) => levels[a] + 1,
            Node::Op(_, a, b) => levels[a].max(levels[b]) + 1,
            Node::TresOp(_, a, b, c) => levels[a].max(levels[b]).max(levels[c]) + 1,
        };
        levels[i] = lv;
        if lv > max_level {
            max_level = lv;
        }
    }
    let t_levels = std::time::Instant::now();

    // Group node indices by level.
    let num_levels = (max_level + 1) as usize;
    let mut level_sizes = vec![0usize; num_levels];
    for &lv in &levels {
        level_sizes[lv as usize] += 1;
    }
    let mut level_offsets = vec![0usize; num_levels + 1];
    for i in 0..num_levels {
        level_offsets[i + 1] = level_offsets[i] + level_sizes[i];
    }
    let mut level_indices = vec![0usize; n];
    let mut cursor = level_offsets[..num_levels].to_vec();
    for i in 0..n {
        let lv = levels[i] as usize;
        level_indices[cursor[lv]] = i;
        cursor[lv] += 1;
    }
    let t_group = std::time::Instant::now();

    let max_width = level_sizes.iter().copied().max().unwrap_or(0);
    let avg_width = n as f64 / num_levels as f64;

    eprintln!(
        "[groth16_witness] DAG cached: nodes={n} levels={num_levels} max_width={max_width} \
         avg_width={avg_width:.0} snapshot={:.1}ms levels={:.1}ms group={:.1}ms total={:.1}ms",
        (t_snapshot - t0).as_secs_f64() * 1000.0,
        (t_levels - t_snapshot).as_secs_f64() * 1000.0,
        (t_group - t_levels).as_secs_f64() * 1000.0,
        (t_group - t0).as_secs_f64() * 1000.0,
    );

    LevelData {
        all_nodes,
        level_offsets,
        level_indices,
        num_levels,
    }
}

/// Parallel BFS-level evaluation of the circom witness DAG using rayon.
///
/// The graph is topologically sorted (all dependency indices < current index).
/// Level data is precomputed and cached; only the actual evaluation runs per call.
fn evaluate_parallel(
    pg: &ParsedGraph,
    ff: &circom_witnesscalc::field::Field<U254>,
    ns: &VecNodes,
    inputs: &[U254],
    outputs: &[usize],
    constants: &[U254],
) -> Vec<U254> {
    let ld = pg.level_data.get_or_init(|| compute_level_data(ns));
    let n = ld.all_nodes.len();

    // Allocate values array without zero-initialization.
    // SAFETY: Every element values[i] is written exactly once (at its BFS level) before any
    // read. Level 0 writes inputs/constants, levels 1+ write computed values. The topological
    // sort invariant guarantees all dependencies are written before they're read.
    let mut values = Vec::<U254>::with_capacity(n);
    unsafe { values.set_len(n) };

    // Level 0: Input/Constant nodes.
    let lv0_start = ld.level_offsets[0];
    let lv0_end = ld.level_offsets[1];
    for idx in lv0_start..lv0_end {
        let i = ld.level_indices[idx];
        values[i] = match ld.all_nodes[i] {
            Node::Input(inp) => inputs[inp],
            Node::Constant(c) => constants[c],
            _ => U254::zero(),
        };
    }

    // Levels 1+: parallel evaluation using rayon.
    // SAFETY: Within each level, writes are non-overlapping. All reads are from earlier levels.
    let values_ptr = SendPtr(values.as_mut_ptr());
    const PAR_THRESHOLD: usize = 256;

    for lv in 1..ld.num_levels {
        let start = ld.level_offsets[lv];
        let end = ld.level_offsets[lv + 1];
        let slice = &ld.level_indices[start..end];

        if slice.len() >= PAR_THRESHOLD {
            let vp = values_ptr;
            let nodes = &ld.all_nodes;
            slice.par_iter().for_each(move |&i| {
                let v = eval_node_sp(&nodes[i], vp, ff);
                unsafe { vp.write_at(i, v) };
            });
        } else {
            for &i in slice {
                let v = eval_node_sp(&ld.all_nodes[i], values_ptr, ff);
                unsafe { values_ptr.write_at(i, v) };
            }
        }
    }

    outputs.iter().map(|&i| values[i]).collect()
}

/// Evaluate a single node given the values array via SendPtr.
#[inline]
fn eval_node_sp(
    node: &Node,
    values: SendPtr<U254>,
    ff: &circom_witnesscalc::field::Field<U254>,
) -> U254 {
    // SAFETY: All dependency indices a,b,c are guaranteed < current node index
    // by the topological sort invariant, and have already been written.
    match *node {
        Node::Input(_) | Node::Constant(_) | Node::Unknown => unreachable!(),
        Node::UnoOp(op, a) => {
            let va = unsafe { values.read_at(a) };
            ff.op_uno(op, va)
        }
        Node::Op(op, a, b) => {
            let va = unsafe { values.read_at(a) };
            let vb = unsafe { values.read_at(b) };
            ff.op_duo(op, va, vb)
        }
        Node::TresOp(_, a, b, c) => {
            let va = unsafe { values.read_at(a) };
            // TresOp is always TernCond: if a == 0 then c else b
            if va.is_zero() {
                unsafe { values.read_at(c) }
            } else {
                unsafe { values.read_at(b) }
            }
        }
    }
}

#[allow(dead_code)]
fn calc_witness(graph_path: &Path, inputs: &str) -> Result<CalcWitness> {
    let pg = PARSED_GRAPH.get_or_init(|| load_graph(graph_path));

    let nodes = pg
        .nodes
        .as_any()
        .downcast_ref::<Nodes<U254, VecNodes>>()
        .ok_or_else(|| anyhow!("expected U254 nodes in circom graph"))?;

    let t0 = std::time::Instant::now();

    // Parse inputs JSON → HashMap<String, Vec<U254>>
    let input_map = deserialize_inputs2(inputs.as_bytes(), &nodes.ff)
        .map_err(|err| anyhow!("input parse failure: {err}"))?;

    // Map inputs to flat array by signal offset (mirrors private create_inputs)
    let mut max_idx: usize = 0;
    for (offset, len) in pg.input_mapping.values() {
        let idx = offset + len;
        if idx > max_idx {
            max_idx = idx;
        }
    }
    let mut flat_inputs = vec![U254::zero(); max_idx + 1];
    flat_inputs[0] = U254::one();
    for (key, value) in &input_map {
        let &(offset, len) = pg
            .input_mapping
            .get(key.as_str())
            .ok_or_else(|| anyhow!("unknown input signal: {key}"))?;
        anyhow::ensure!(len == value.len(), "invalid input length for {key}");
        for (i, v) in value.iter().enumerate() {
            flat_inputs[offset + i] = *v;
        }
    }

    let t1 = std::time::Instant::now();

    // Evaluate the circuit graph using parallel BFS-level evaluation
    let result = evaluate_parallel(
        pg,
        &nodes.ff,
        &nodes.nodes,
        &flat_inputs,
        &pg.signals,
        &nodes.constants,
    );

    let t2 = std::time::Instant::now();

    // Zero-copy transmute Vec<U254> → Vec<FieldElement<32>>.
    // SAFETY: U254 = ruint::Uint<254, 4> is #[repr(transparent)] over [u64; 4] (32 bytes).
    // FieldElement<32> wraps [u8; 32] (32 bytes). On little-endian, the byte representations
    // are identical (confirmed by U254::as_le_slice). The Vec's alignment (8) exceeds
    // FieldElement's requirement (1).
    const _: () =
        assert!(std::mem::size_of::<U254>() == std::mem::size_of::<wtns_file::FieldElement<32>>());
    let witness: Vec<wtns_file::FieldElement<32>> = unsafe {
        let mut result = std::mem::ManuallyDrop::new(result);
        Vec::from_raw_parts(
            result.as_mut_ptr() as *mut wtns_file::FieldElement<32>,
            result.len(),
            result.capacity(),
        )
    };

    let t3 = std::time::Instant::now();
    eprintln!(
        "[groth16_witness] inputs={:.1}ms evaluate={:.1}ms convert={:.1}ms total={:.1}ms",
        (t1 - t0).as_secs_f64() * 1000.0,
        (t2 - t1).as_secs_f64() * 1000.0,
        (t3 - t2).as_secs_f64() * 1000.0,
        (t3 - t0).as_secs_f64() * 1000.0,
    );

    Ok(CalcWitness::Vec(witness))
}

/// Binary-input variant of calc_witness that accepts pre-computed U254 values
/// for the "iop" signal, bypassing JSON serialization/parsing entirely.
fn calc_witness_binary(graph_path: &Path, iop_values: Vec<U254>) -> Result<CalcWitness> {
    let pg = PARSED_GRAPH.get_or_init(|| load_graph(graph_path));

    let nodes = pg
        .nodes
        .as_any()
        .downcast_ref::<Nodes<U254, VecNodes>>()
        .ok_or_else(|| anyhow!("expected U254 nodes in circom graph"))?;

    let t0 = std::time::Instant::now();

    // Look up the "iop" signal's offset and length in the flat input array
    let &(iop_offset, iop_len) = pg
        .input_mapping
        .get("iop")
        .ok_or_else(|| anyhow!("missing 'iop' input signal in circom graph"))?;
    anyhow::ensure!(
        iop_len == iop_values.len(),
        "iop signal length mismatch: expected {iop_len}, got {}",
        iop_values.len()
    );

    // Build flat_inputs directly (same logic as calc_witness but without JSON parsing)
    let mut max_idx: usize = 0;
    for (offset, len) in pg.input_mapping.values() {
        let idx = offset + len;
        if idx > max_idx {
            max_idx = idx;
        }
    }
    let mut flat_inputs = vec![U254::zero(); max_idx + 1];
    flat_inputs[0] = U254::one();
    // Copy iop values directly into the flat array at the signal's offset
    flat_inputs[iop_offset..iop_offset + iop_len].copy_from_slice(&iop_values);

    let t1 = std::time::Instant::now();

    // The fast evaluator (as on CUDA), straight into the pinned buffer the prover uploads from.
    if let Some(program) = fast_program(pg, nodes) {
        let mut buf = risc0_groth16_sys::host_buffer(pg.signals.len() * 32)
            .context("failed to allocate the pinned witness buffer")?;
        super::witness_fast::eval(program, nodes, &flat_inputs, buf.as_limbs_mut());
        eprintln!(
            "[groth16_witness] inputs={:.1}ms evaluate(fast)={:.1}ms total={:.1}ms",
            (t1 - t0).as_secs_f64() * 1000.0,
            t1.elapsed().as_secs_f64() * 1000.0,
            t0.elapsed().as_secs_f64() * 1000.0,
        );
        return Ok(CalcWitness::Pinned(buf));
    }

    // Evaluate the circuit graph using parallel BFS-level evaluation
    let result = evaluate_parallel(
        pg,
        &nodes.ff,
        &nodes.nodes,
        &flat_inputs,
        &pg.signals,
        &nodes.constants,
    );

    let t2 = std::time::Instant::now();

    // Zero-copy transmute Vec<U254> → Vec<FieldElement<32>>.
    const _: () =
        assert!(std::mem::size_of::<U254>() == std::mem::size_of::<wtns_file::FieldElement<32>>());
    let witness: Vec<wtns_file::FieldElement<32>> = unsafe {
        let mut result = std::mem::ManuallyDrop::new(result);
        Vec::from_raw_parts(
            result.as_mut_ptr() as *mut wtns_file::FieldElement<32>,
            result.len(),
            result.capacity(),
        )
    };

    let t3 = std::time::Instant::now();
    eprintln!(
        "[groth16_witness] inputs={:.1}ms evaluate={:.1}ms convert={:.1}ms total={:.1}ms",
        (t1 - t0).as_secs_f64() * 1000.0,
        (t2 - t1).as_secs_f64() * 1000.0,
        (t3 - t2).as_secs_f64() * 1000.0,
        (t3 - t0).as_secs_f64() * 1000.0,
    );

    Ok(CalcWitness::Vec(witness))
}
