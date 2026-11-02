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

//! Faster Docker-based shrink_wrap (`intel` feature).
//!
//! The pinned prover image runs the circom C++ witness generator (~2.8 s,
//! single-threaded) and then gnark. This module instead
//!  * computes the witness in-process from the rzup `risc0-groth16` witness
//!    graph (same circuit; the witness is byte-identical to the C++
//!    generator's) with a Montgomery-form evaluator (~0.3 s), and
//!  * starts the image's gnark `prover` early (it spends ~1 s loading the
//!    constraint system and proving key) and feeds it the witness through a
//!    named pipe in the mounted work directory.
//! The prover, proving key and circuit are the pinned image's, so proofs are
//! unchanged. If the rzup component is missing, the caller falls back to the
//! plain Docker path.
//!
//! Cleanup: the parent opens the pipe read-write as soon as the prover is
//! started, so a prover that has opened the pipe sees EOF and exits when this
//! process closes it (or dies). Dropping an unused handle also `docker kill`s
//! the (named) container, and the prover runs under `timeout` so that a
//! container cannot outlive a killed parent by more than PROVER_TIMEOUT_S.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Mutex, OnceLock},
};

use anyhow::{anyhow, bail, Context, Result};
use ark_bn254::Fr;
use ark_ff::{BigInt, Field as _, PrimeField, Zero as _};
use circom_witnesscalc::{
    field::{FieldOperations, U254},
    graph::{Node, Nodes, NodesStorage, Operation, VecNodes},
    storage::proto_deserializer::deserialize_witnesscalc_graph_from_bytes,
};
use rayon::prelude::*;
use risc0_core::field::baby_bear::BabyBearElem;
use risc0_zkp::core::{
    digest::{Digest, DIGEST_WORDS},
    hash::poseidon_254::digest_to_fr_bytes,
};
use tempfile::TempDir;

use super::seal_format::{IopType, K_SEAL_ELEMS, K_SEAL_TYPES, K_SEAL_WORDS};
use crate::{ProofJson, Seal};

use super::docker::IMAGE;

/// SHA-256 of rzup risc0-groth16 v0.1.0 `stark_verify_graph.bin`; its witness
/// was checked byte-identical to the pinned image's generator.
const GRAPH_SHA256: &str = "51020d6fd9ded962";

// ---------------------------------------------------------------------------
// Witness graph -> straight-line Montgomery program
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Op {
    Leaf,
    Mul(usize, usize),
    MulC(usize, Fr),
    Add(usize, usize),
    Sub(usize, usize),
    Gen,
}

struct Program {
    nodes: Vec<Node>,
    prog: Vec<Op>,
    consts: Vec<Fr>,
    signals: Vec<usize>,
    input_offset: usize,
    input_len: usize,
    ff: circom_witnesscalc::field::Field<U254>,
    prime: U254,
}

fn to_fr(x: &U254) -> Fr {
    Fr::from_bigint(BigInt::<4>(*x.as_limbs())).expect("graph value >= field modulus")
}

fn to_u(x: &Fr) -> U254 {
    U254::from_limbs(x.into_bigint().0)
}

static PROGRAM: OnceLock<std::result::Result<Program, String>> = OnceLock::new();

pub(crate) fn graph_path() -> Result<PathBuf> {
    Ok(super::component::component_dir()?.join("stark_verify_graph.bin"))
}

fn build_program() -> Result<Program> {
    let t0 = std::time::Instant::now();
    let bytes = std::fs::read(graph_path()?).context("failed to read the witness graph")?;
    let digest = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&bytes));
    if !digest.starts_with(GRAPH_SHA256) {
        bail!("unexpected witness graph (sha256 {digest}); expected rzup risc0-groth16 0.1.0");
    }
    let (nodes_box, signals, input_mapping) = deserialize_witnesscalc_graph_from_bytes(&bytes)
        .map_err(|e| anyhow!("failed to parse the witness graph: {e:?}"))?;
    let nodes = nodes_box
        .as_any()
        .downcast_ref::<Nodes<U254, VecNodes>>()
        .ok_or_else(|| anyhow!("unexpected witness graph node type"))?;
    // The seal is the only input signal ("iop"), K_SEAL_ELEMS long.
    if input_mapping.len() != 1 {
        bail!(
            "witness graph has {} input signals, expected 1",
            input_mapping.len()
        );
    }
    let &(input_offset, input_len) = input_mapping.values().next().unwrap();
    if input_len != K_SEAL_ELEMS {
        bail!("witness graph input has {input_len} elements, expected {K_SEAL_ELEMS}");
    }
    let n = nodes.nodes.len();
    let all: Vec<Node> = (0..n).map(|i| nodes.nodes.get(i).unwrap()).collect();
    let consts: Vec<Fr> = nodes.constants.iter().map(to_fr).collect();
    // Division by a constant becomes multiplication by its inverse.
    let prog = all
        .iter()
        .map(|nd| match *nd {
            Node::Input(_) | Node::Constant(_) | Node::Unknown => Op::Leaf,
            Node::Op(Operation::Mul, a, b) => Op::Mul(a, b),
            Node::Op(Operation::Add, a, b) => Op::Add(a, b),
            Node::Op(Operation::Sub, a, b) => Op::Sub(a, b),
            Node::Op(Operation::Div, a, b) => match all[b] {
                Node::Constant(c) => Op::MulC(a, consts[c].inverse().unwrap_or(Fr::zero())),
                _ => Op::Gen,
            },
            _ => Op::Gen,
        })
        .collect();
    let prime = nodes.ff.prime;
    let ff = nodes.ff.clone();
    tracing::debug!(
        "groth16 witness graph prepared in {:.1} ms ({n} nodes)",
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Ok(Program {
        nodes: all,
        prog,
        consts,
        signals,
        input_offset,
        input_len,
        ff,
        prime,
    })
}

fn program() -> Result<&'static Program> {
    PROGRAM
        .get_or_init(|| build_program().map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
}

/// Read, check and precompile the witness graph (~1 s); call early on a
/// background thread.
pub(crate) fn preload() -> Result<()> {
    program().map(|_| ())
}

/// Seal words -> circuit input values (same as seal_to_json + the C++
/// generator's JSON parsing).
fn seal_inputs(seal_bytes: &[u8]) -> Result<Vec<U254>> {
    if seal_bytes.len() != K_SEAL_WORDS * 4 {
        bail!(
            "identity_p254 seal has {} bytes, expected {}",
            seal_bytes.len(),
            K_SEAL_WORDS * 4
        );
    }
    let iop: &[u32] = bytemuck::cast_slice(seal_bytes);
    let mut values = Vec::with_capacity(K_SEAL_ELEMS);
    let mut pos = 0;
    for seal_type in K_SEAL_TYPES.iter().take(K_SEAL_ELEMS) {
        match seal_type {
            IopType::Fp => {
                values.push(U254::from(BabyBearElem::new_raw(iop[pos]).as_u32() as u64));
                pos += 1;
            }
            _ => {
                let digest = Digest::try_from(&iop[pos..pos + DIGEST_WORDS])?;
                pos += DIGEST_WORDS;
                values.push(U254::from_le_bytes(digest_to_fr_bytes(&digest)));
            }
        }
    }
    Ok(values)
}

/// The circuit's witness for an `identity_p254` seal, as canonical little-endian field
/// elements (the GPU prover's input).
pub(crate) fn witness(seal_bytes: &[u8]) -> Result<Vec<wtns_file::FieldElement<32>>> {
    let prog = program()?;
    witness_elements(prog, &seal_inputs(seal_bytes)?)
}

/// Evaluate the witness and write it in .wtns format to `out`.
fn write_witness(prog: &Program, inputs: &[U254], out: impl Write) -> Result<()> {
    let w = witness_elements(prog, inputs)?;
    let prime = wtns_file::FieldElement::from(prog.prime.to_le_bytes::<32>());
    wtns_file::WtnsFile::<32>::from_vec(w, prime)
        .write(out)
        .context("failed to write the witness")?;
    Ok(())
}

/// Evaluate the witness graph: the signals' values, canonical little-endian.
fn witness_elements(prog: &Program, inputs: &[U254]) -> Result<Vec<wtns_file::FieldElement<32>>> {
    let n = prog.nodes.len();
    let mut v: Vec<Fr> = vec![Fr::zero(); n];
    for i in 0..n {
        v[i] = match prog.prog[i] {
            // Input signal 0 is the constant 1; the seal occupies
            // [input_offset, input_offset + input_len); any other input is 0.
            Op::Leaf => match prog.nodes[i] {
                Node::Input(0) => Fr::from(1u64),
                Node::Input(x)
                    if x >= prog.input_offset && x < prog.input_offset + prog.input_len =>
                {
                    to_fr(&inputs[x - prog.input_offset])
                }
                Node::Input(x) if x < prog.input_offset => Fr::zero(),
                Node::Input(x) => bail!("witness graph input index {x} out of range"),
                Node::Constant(c) => prog.consts[c],
                _ => Fr::zero(),
            },
            Op::Mul(a, b) => v[a] * v[b],
            Op::MulC(a, c) => v[a] * c,
            Op::Add(a, b) => v[a] + v[b],
            Op::Sub(a, b) => v[a] - v[b],
            Op::Gen => match prog.nodes[i] {
                Node::Op(op, a, b) => to_fr(&(&prog.ff).op_duo(op, to_u(&v[a]), to_u(&v[b]))),
                Node::UnoOp(op, a) => to_fr(&(&prog.ff).op_uno(op, to_u(&v[a]))),
                Node::TresOp(_, a, b, c) => {
                    if v[a].is_zero() {
                        v[c]
                    } else {
                        v[b]
                    }
                }
                _ => unreachable!(),
            },
        };
    }
    // In parallel: 5.6M conversions out of Montgomery form.
    Ok(prog
        .signals
        .par_iter()
        .map(|&i| wtns_file::FieldElement::from(to_u(&v[i]).to_le_bytes::<32>()))
        .collect())
}

// ---------------------------------------------------------------------------
// Pre-started prover
// ---------------------------------------------------------------------------

/// Upper bound on a prover container's lifetime (a proof takes ~5 s).
const PROVER_TIMEOUT_S: u32 = 600;

struct Prover {
    child: Option<Child>,
    name: String,
    dir: TempDir,
    fifo: Option<File>,
}

impl Drop for Prover {
    fn drop(&mut self) {
        // Closing our end of the pipe gives the prover EOF; it then exits
        // and docker removes the container (--rm).
        self.fifo.take();
        if let Some(mut child) = self.child.take() {
            let _ = Command::new("docker")
                .args(["kill", &self.name])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            let _ = child.wait();
        }
    }
}

static PRESTARTED: Mutex<Option<Prover>> = Mutex::new(None);

fn start_prover() -> Result<Prover> {
    let dir = tempfile::tempdir()?;
    let fifo_path = dir.path().join("witness.wtns");
    let status = Command::new("mkfifo").arg(&fifo_path).status()?;
    if !status.success() {
        bail!("mkfifo failed");
    }
    static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let name = format!(
        "risc0-groth16-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let child = Command::new("docker")
        .arg("run")
        .arg("--rm")
        .arg("--name")
        .arg(&name)
        .arg("-v")
        .arg(format!("{}:/mnt", dir.path().to_string_lossy()))
        .arg("-w")
        .arg("/app")
        .arg("--entrypoint")
        .arg("timeout")
        .arg(IMAGE)
        .arg(PROVER_TIMEOUT_S.to_string())
        .args([
            "prover",
            "stark_verify.cs",
            "stark_verify_final.pk.dmp",
            "/mnt/witness.wtns",
            "/mnt/proof.json",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::from(File::create(dir.path().join("prover.log"))?))
        .spawn()
        .context("failed to start the groth16 prover container")?;
    // Read-write open of a FIFO does not block on Linux and keeps a writer
    // alive until we are done (see the module docs).
    let fifo = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&fifo_path)?;
    Ok(Prover {
        child: Some(child),
        name,
        dir,
        fifo: Some(fifo),
    })
}

/// Start the prover container ahead of shrink_wrap so that it loads the
/// proving key in the background. No-op if one is already waiting.
pub(crate) fn prestart() -> Result<()> {
    let mut slot = PRESTARTED.lock().unwrap();
    if slot.is_none() {
        *slot = Some(start_prover()?);
    }
    Ok(())
}

pub(crate) fn shrink_wrap(seal_bytes: &[u8]) -> Result<Seal> {
    let t0 = std::time::Instant::now();
    let prog = program()?;
    let mut prover = match PRESTARTED.lock().unwrap().take() {
        Some(p) => p,
        None => start_prover()?,
    };
    let inputs = seal_inputs(seal_bytes)?;
    {
        let fifo = prover.fifo.take().unwrap();
        write_witness(prog, &inputs, BufWriter::with_capacity(1 << 20, fifo))?;
    }
    let t_wit = t0.elapsed();
    let output = prover
        .child
        .take()
        .unwrap()
        .wait_with_output()
        .context("groth16 prover container")?;
    if !output.status.success() {
        let log = std::fs::read_to_string(prover.dir.path().join("prover.log")).unwrap_or_default();
        bail!(
            "groth16 prover failed ({:?}): {}",
            output.status.code(),
            log.lines().last().unwrap_or("")
        );
    }
    let contents = std::fs::read_to_string(prover.dir.path().join("proof.json"))
        .context("groth16 prover produced no proof")?;
    tracing::debug!(
        "shrink_wrap (fast): witness {:.1} ms, total {:.1} ms",
        t_wit.as_secs_f64() * 1000.0,
        t0.elapsed().as_secs_f64() * 1000.0
    );
    let proof_json: ProofJson = serde_json::from_str(&contents)?;
    proof_json.try_into()
}
