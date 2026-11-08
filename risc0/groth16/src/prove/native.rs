//! Native macOS wrapping with a bounded, owned Go prover child.
use crate::{ProofJson, Seal};
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Mutex, OnceLock},
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
    identities: BTreeMap<String, FileIdentity>,
    timeout: Duration,
    threads: usize,
}
static CONFIG: OnceLock<std::result::Result<Config, String>> = OnceLock::new();

#[derive(Clone, PartialEq, Eq)]
struct FileIdentity {
    size: u64,
    modified_sec: i64,
    modified_nsec: i64,
    inode: u64,
    device: u64,
}

struct VerifiedBundle {
    root: PathBuf,
    identities: BTreeMap<String, FileIdentity>,
}

type Verification = std::result::Result<VerifiedBundle, String>;
type VerificationHandle = Mutex<Option<std::thread::JoinHandle<Verification>>>;
static VERIFICATION: OnceLock<std::result::Result<VerificationHandle, String>> = OnceLock::new();

fn number(name: &str, default: usize, max: usize) -> Result<usize> {
    let value = std::env::var(name)
        .map(|s| s.parse())
        .unwrap_or(Ok(default))
        .with_context(|| format!("invalid {name}"))?;
    ensure!((1..=max).contains(&value), "{name} must be in 1..={max}");
    Ok(value)
}

fn identity(path: &Path) -> Result<FileIdentity> {
    let metadata = path.metadata()?;
    Ok(FileIdentity {
        size: metadata.len(),
        modified_sec: metadata.mtime(),
        modified_nsec: metadata.mtime_nsec(),
        inode: metadata.ino(),
        device: metadata.dev(),
    })
}

fn bundle_root() -> Result<PathBuf> {
    let root = PathBuf::from(
        std::env::var_os("RISC0_GROTH16_NATIVE_DIR")
            .context("set RISC0_GROTH16_NATIVE_DIR to the sealed native helper bundle")?,
    )
    .canonicalize()?;
    // Keep the cheap architecture check on the startup path. The large-file
    // manifest hashes run on the verification thread instead.
    let mut header = [0u8; 8];
    File::open(root.join("prover"))?.read_exact(&mut header)?;
    ensure!(
        header == [0xcf, 0xfa, 0xed, 0xfe, 0x0c, 0, 0, 1],
        "native prover must be a Darwin ARM64 executable"
    );
    Ok(root)
}

fn verify_bundle(root: PathBuf) -> Result<VerifiedBundle> {
    let mut identities = BTreeMap::new();
    let manifest_path = root.join("manifest.json").canonicalize()?;
    ensure!(manifest_path.parent() == Some(root.as_path()),
        "native manifest escapes bundle");
    let manifest_before = identity(&manifest_path)?;
    ensure!(
        manifest_before.size <= 16384,
        "native manifest exceeds size limit"
    );
    let bytes = std::fs::read(&manifest_path)?;
    ensure!(bytes.len() <= 16384, "native manifest exceeds size limit");
    ensure!(identity(&manifest_path)? == manifest_before,
        "native manifest changed during verification");
    identities.insert("manifest.json".into(), manifest_before);
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
        let before = identity(&path)?;
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
        ensure!(identity(&path)? == before,
            "native artifact changed during verification: {name}");
        identities.insert(name.into(), before);
    }
    Ok(VerifiedBundle { root, identities })
}

/// Begin full manifest verification without delaying the worker handshake.
/// `config()` joins this exact result before any helper can be started.
pub fn start_verification() -> Result<()> {
    if CONFIG.get().is_some() {
        return Ok(());
    }
    VERIFICATION
        .get_or_init(|| {
            let root = bundle_root().map_err(|e| format!("{e:#}"))?;
            let handle = std::thread::Builder::new()
                .name("native-bundle-verify".into())
                .spawn(move || verify_bundle(root).map_err(|e| format!("{e:#}")))
                .map_err(|e| format!("{e:#}"))?;
            Ok(Mutex::new(Some(handle)))
        })
        .as_ref()
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn load_config() -> Result<Config> {
    let verified = match VERIFICATION.get() {
        Some(Ok(state)) => match state.lock().unwrap().take() {
            Some(handle) => handle
                .join()
                .map_err(|_| anyhow::anyhow!("native bundle verification thread panicked"))?
                .map_err(anyhow::Error::msg)?,
            None => verify_bundle(bundle_root()?)?,
        },
        Some(Err(error)) => anyhow::bail!("{error}"),
        None => verify_bundle(bundle_root()?)?,
    };
    Ok(Config {
        root: verified.root,
        identities: verified.identities,
        timeout: Duration::from_secs(number("RISC0_GROTH16_NATIVE_TIMEOUT", 1800, 1800)? as u64),
        threads: number("RISC0_GROTH16_NATIVE_THREADS", 6, 18)?,
    })
}

fn identities_match(config: &Config) -> bool {
    config.identities.iter().all(|(name, expected)| {
        let path = config.root.join(name);
        path.canonicalize().ok().as_ref().is_some_and(|p| p.parent() == Some(config.root.as_path()))
            && identity(&path).ok().as_ref() == Some(expected)
    })
}

fn config() -> Result<&'static Config> {
    let config = CONFIG
        .get_or_init(|| load_config().map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if !identities_match(config) {
        // An in-process verdict is never trusted after size, mtime, inode or
        // device changes. Rehash instead of spawning a helper on stale assets.
        verify_bundle(config.root.clone())?;
    }
    Ok(config)
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

/// Owns the helper across identity_p254. Dropping it on any early exit reaps the child.
pub struct Prestarted {
    child: OwnedChild,
    work: tempfile::TempDir,
    started: Instant,
    output_log: Option<std::thread::JoinHandle<()>>,
}

/// Start key loading while identity_p254 runs. The helper waits on stdin for the witness.
pub fn prestart() -> Result<Prestarted> {
    preflight()?;
    let config = config()?;
    let work = match std::env::var_os("RISC0_WORK_DIR") {
        Some(root) => tempfile::tempdir_in(root)?,
        None => tempfile::tempdir()?,
    };
    let output = work.path().join("proof.json");
    let started = Instant::now();
    let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
    let mut child = Command::new(config.root.join("prover"))
        .arg(format!("--native-parent={}", std::process::id()))
        .arg(format!("--native-timeout={}s", config.timeout.as_secs()))
        .arg("--native-no-gc-setup")
        .arg(config.root.join("stark_verify.cs"))
        .arg(config.root.join("stark_verify_final.pk.dmp"))
        .arg("-")
        .arg(&output)
        .env_clear()
        .env("GOMAXPROCS", config.threads.to_string())
        .stdin(Stdio::piped())
        .stdout(if verbose {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stderr(Stdio::null())
        .spawn()
        .context("start native Groth16 prover")?;
    let output_log = child.stdout.take().map(|stdout| {
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => eprintln!("[native_helper] {line}"),
                    Err(_) => break,
                }
            }
        })
    });
    Ok(Prestarted {
        child: OwnedChild(child),
        work,
        started,
        output_log,
    })
}

/// Produce a complete Groth16 seal without Docker or remote proving.
pub fn shrink_wrap(seal: &[u8]) -> Result<Seal> {
    shrink_wrap_started(seal, None)
}

/// Stream the witness to an already loading helper, or start one for direct callers.
pub fn shrink_wrap_started(seal: &[u8], prestarted: Option<Prestarted>) -> Result<Seal> {
    let mut prover = match prestarted {
        Some(prover) => prover,
        None => prestart()?,
    };
    let config = config()?;
    ensure!(
        prover.started.elapsed() < config.timeout,
        "native Groth16 prover exceeded deadline"
    );
    let witness_start = Instant::now();
    let stdin = prover
        .child
        .0
        .stdin
        .take()
        .context("native prover stdin unavailable")?;
    let mut writer = BufWriter::new(stdin);
    super::witness::write(seal, &mut writer).context("stream native witness")?;
    writer.flush().context("flush native witness")?;
    drop(writer);
    let witness_ms = witness_start.elapsed().as_secs_f64() * 1000.0;
    let status = loop {
        if let Some(status) = prover.child.0.try_wait()? {
            break status;
        }
        if prover.started.elapsed() >= config.timeout {
            bail!("native Groth16 prover exceeded deadline");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    ensure!(status.success(), "native Groth16 prover failed: {status}");
    if let Some(output_log) = prover.output_log.take() {
        let _ = output_log.join();
    }
    let output = prover.work.path().join("proof.json");
    ensure!(
        std::fs::metadata(&output)?.len() <= 16384,
        "native proof JSON exceeds size limit"
    );
    let proof: ProofJson = serde_json::from_slice(&std::fs::read(output)?)?;
    tracing::info!(
        witness_ms,
        prestart_and_wrap_ms = prover.started.elapsed().as_secs_f64() * 1000.0,
        "native Groth16 wrap complete"
    );
    proof.try_into()
}

/// Effective global Rayon pool size, frozen by native worker startup.
pub fn rayon_threads() -> usize {
    rayon::current_num_threads()
}
