The RISC Zero Proving System

These are the core algorithms that prove and verify the execution of zk-STARK circuits.

# Crate Feature Flags

The following [crate feature flags](https://doc.rust-lang.org/cargo/reference/features.html) are available.

| Feature | Target(s)         | Implies    | Description                                                                                |
| ------- | ----------------- | ---------- | ------------------------------------------------------------------------------------------ |
| cuda    |                   | prove, std | Turns on CUDA GPU acceleration for the prover. Requires CUDA toolkit to be installed.      |
| metal   | macos             | prove, std | Enables Metal primitives. Circuit-prover selection is explicit; see [macOS proving](../../README.md#macos-proving). |
| prove   | all except rv32im | std        | Enables the prover, incompatible within the zkvm guest.                                    |
| std     | all               |            | Support for the Rust stdlib.                                                               |
