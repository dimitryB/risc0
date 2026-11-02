use std::time::Instant;

use risc0_zkvm::{get_prover_server, ExecutorEnv, ExecutorImpl, ProverOpts, VerifierContext};
use risc0_zkvm_methods::{bench::BenchmarkSpec, BENCH_ELF};

fn main() {
    let iters: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_900_000);
    let po2: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let spec = BenchmarkSpec::SimpleLoop { iters };

    // RISC0_EARLY_PROVER=1: construct the prover (which starts the CUDA warmup) before executing
    // the guest, the order LocalProver::prove and prove_e2e use, so CUDA init overlaps execution.
    let opts = ProverOpts::composite().with_hashfn("poseidon2".to_string());
    let early_prover =
        std::env::var_os("RISC0_EARLY_PROVER").map(|_| get_prover_server(&opts).unwrap());

    eprintln!("Executing guest (SimpleLoop {iters} iters, segment_limit_po2={po2})...");
    let env = ExecutorEnv::builder()
        .write(&spec)
        .unwrap()
        .segment_limit_po2(po2)
        .build()
        .unwrap();
    let session = ExecutorImpl::from_elf(env, BENCH_ELF)
        .unwrap()
        .run()
        .unwrap();
    eprintln!(
        "  segments: {}, total_cycles: {}, user_cycles: {}",
        session.segments.len(),
        session.total_cycles,
        session.user_cycles,
    );

    let prover = early_prover.unwrap_or_else(|| get_prover_server(&opts).unwrap());
    let ctx = VerifierContext::default();

    eprintln!("Proving {} segments...", session.segments.len());
    let t0 = Instant::now();
    let prove_info = prover.prove_session(&ctx, &session).unwrap();
    let elapsed = t0.elapsed();

    println!();
    println!(
        "prove_session: {:.2}s ({} segments, {:.1}ms/segment)",
        elapsed.as_secs_f64(),
        session.segments.len(),
        elapsed.as_secs_f64() * 1000.0 / session.segments.len() as f64,
    );
    println!("  stats: {:?}", prove_info.stats);
}
