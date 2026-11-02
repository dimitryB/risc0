//! Native macOS wrapping with a bounded, owned Go prover child.
use crate::{ProofJson, Seal};
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Read},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::OnceLock,
    time::{Duration, Instant},
};

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("native-groth16 requires macOS ARM64");
#[cfg(any(feature = "cuda", feature = "rocm", feature = "intel"))]
compile_error!("native-groth16 cannot be combined with CUDA, ROCm, or Intel");

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    files: BTreeMap<String, String>,
}
struct Config {
    root: PathBuf,
    timeout: Duration,
    threads: usize,
}
static CONFIG: OnceLock<std::result::Result<Config, String>> = OnceLock::new();

fn number(name: &str, default: usize, max: usize) -> Result<usize> {
    let value = std::env::var(name)
        .map(|s| s.parse())
        .unwrap_or(Ok(default))
        .with_context(|| format!("invalid {name}"))?;
    ensure!((1..=max).contains(&value), "{name} must be in 1..={max}");
    Ok(value)
}

fn load() -> Result<Config> {
    let root = PathBuf::from(
        std::env::var_os("RISC0_GROTH16_NATIVE_DIR")
            .context("set RISC0_GROTH16_NATIVE_DIR to the sealed native helper bundle")?,
    )
    .canonicalize()?;
    ensure!(
        std::fs::metadata(root.join("manifest.json"))?.len() <= 16384,
        "native manifest exceeds size limit"
    );
    let bytes = std::fs::read(root.join("manifest.json"))?;
    ensure!(bytes.len() <= 16384, "native manifest exceeds size limit");
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    ensure!(
        manifest.schema_version == 1,
        "unsupported native manifest version"
    );
    let names = [
        "prover",
        "stark_verify.cs",
        "stark_verify_final.pk.dmp",
        "stark_verify_graph.bin",
    ];
    ensure!(
        manifest.files.len() == names.len(),
        "incomplete native artifact manifest"
    );
    for name in names {
        let expected = manifest
            .files
            .get(name)
            .with_context(|| format!("manifest missing {name}"))?;
        ensure!(
            expected.len() == 64 && expected.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid artifact digest"
        );
        let path = root.join(name).canonicalize()?;
        ensure!(
            path.parent() == Some(root.as_path()),
            "native artifact escapes bundle"
        );
        if name == "stark_verify_graph.bin" {
            ensure!(
                *expected == super::witness::GRAPH_SHA256,
                "unexpected native witness graph"
            );
        }
        let mut file = File::open(&path)?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        ensure!(
            hex::encode(hash.finalize()) == *expected,
            "native artifact hash mismatch: {name}"
        );
    }
    let mut header = [0u8; 8];
    File::open(root.join("prover"))?.read_exact(&mut header)?;
    ensure!(
        header == [0xcf, 0xfa, 0xed, 0xfe, 0x0c, 0, 0, 1],
        "native prover must be a Darwin ARM64 executable"
    );
    Ok(Config {
        root,
        timeout: Duration::from_secs(number("RISC0_GROTH16_NATIVE_TIMEOUT", 1800, 1800)? as u64),
        threads: number("RISC0_GROTH16_NATIVE_THREADS", 6, 18)?,
    })
}

fn config() -> Result<&'static Config> {
    CONFIG
        .get_or_init(|| load().map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Validate the helper bundle before worker admission.
pub fn preflight() -> Result<()> {
    config()?;
    Ok(())
}

/// Prepare the witness graph outside the handshake deadline when desired.
pub fn preload() -> Result<()> {
    preflight()?;
    super::witness::preload()
}

/// Write a deterministic witness for comparison with the reference generator.
pub fn write_witness(seal: &[u8], output: &Path) -> Result<()> {
    preflight()?;
    let file = File::options().write(true).create_new(true).open(output)?;
    super::witness::write(seal, BufWriter::new(file))
}

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Produce a complete Groth16 seal without Docker or remote proving.
pub fn shrink_wrap(seal: &[u8]) -> Result<Seal> {
    preflight()?;
    let config = config()?;
    let start = Instant::now();
    let work = match std::env::var_os("RISC0_WORK_DIR") {
        Some(root) => tempfile::tempdir_in(root)?,
        None => tempfile::tempdir()?,
    };
    let witness = work.path().join("witness.wtns");
    write_witness(seal, &witness)?;
    ensure!(
        start.elapsed() < config.timeout,
        "native witness generation exceeded deadline"
    );
    let output = work.path().join("proof.json");
    let remaining = (config.timeout - start.elapsed()).as_secs().max(1);
    let child = Command::new(config.root.join("prover"))
        .arg(format!("--native-parent={}", std::process::id()))
        .arg(format!("--native-timeout={remaining}s"))
        .arg(config.root.join("stark_verify.cs"))
        .arg(config.root.join("stark_verify_final.pk.dmp"))
        .arg(&witness)
        .arg(&output)
        .env_clear()
        .env("GOMAXPROCS", config.threads.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("start native Groth16 prover")?;
    let mut child = OwnedChild(child);
    let status = loop {
        if let Some(status) = child.0.try_wait()? {
            break status;
        }
        if start.elapsed() >= config.timeout {
            bail!("native Groth16 prover exceeded deadline");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    ensure!(status.success(), "native Groth16 prover failed: {status}");
    ensure!(
        std::fs::metadata(&output)?.len() <= 16384,
        "native proof JSON exceeds size limit"
    );
    let proof: ProofJson = serde_json::from_slice(&std::fs::read(output)?)?;
    tracing::info!(
        witness_and_wrap_ms = start.elapsed().as_secs_f64() * 1000.0,
        "native Groth16 wrap complete"
    );
    proof.try_into()
}

/// Effective global Rayon pool size, frozen by native worker startup.
pub fn rayon_threads() -> usize { rayon::current_num_threads() }
