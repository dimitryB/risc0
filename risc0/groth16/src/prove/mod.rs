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

//! # Groth16 Prover

#[cfg(any(feature = "rocm", feature = "intel"))]
mod component;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "native-groth16")))]
mod docker;
#[cfg(all(feature = "intel", not(any(feature = "cuda", feature = "rocm"))))]
mod fast;
#[cfg(feature = "rocm")]
mod hip;
#[cfg(all(feature = "intel", not(any(feature = "cuda", feature = "rocm"))))]
mod intel;
#[cfg(feature = "native-groth16")]
pub mod native;
mod seal_format;
mod seal_to_json;
#[cfg(feature = "native-groth16")]
mod witness;
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod witness_fast;
#[cfg(feature = "cuda")]
mod witness_par;

use anyhow::Result;

use crate::Seal;

pub use self::seal_to_json::to_json;

/// Preload the circom graph for Groth16 witness calculation.
/// Call on a background thread to overlap the ~728ms graph read+parse
/// with other work (e.g. composite_to_succinct recursion proving).
pub fn preload_graph() -> Result<()> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::cuda::preload_graph()
        } else if #[cfg(feature = "rocm")] {
            self::hip::preload_graph()
        } else if #[cfg(feature = "native-groth16")] {
            self::native::preload()
        } else if #[cfg(feature = "intel")] {
            self::fast::preload()
        } else {
            Ok(()) // docker mode doesn't use local graph
        }
    }
}

/// Start [preload_graph] on a background thread (CUDA with the cold-path optimizations on, see
/// `RISC0_CUDA_G16_COLD`; no-op otherwise), e.g. while the guest executes.
pub fn preload_graph_background() {
    #[cfg(feature = "cuda")]
    if self::cuda::cold_path_enabled() {
        std::thread::spawn(|| {
            let _ = self::cuda::preload_graph();
        });
    }
}

/// Keeps a native helper owned across identity_p254 and reaps it on early exit.
#[derive(Default)]
pub struct PrestartedProver {
    #[cfg(feature = "native-groth16")]
    native: Option<native::Prestarted>,
}

/// Start the Groth16 prover ahead of [shrink_wrap] so that it loads its
/// proving key while the caller finishes `identity_p254`.
pub fn prestart_prover() -> Result<PrestartedProver> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "native-groth16")] {
            Ok(PrestartedProver { native: Some(self::native::prestart()?) })
        } else if #[cfg(all(feature = "intel", not(any(feature = "cuda", feature = "rocm"))))] {
            if self::intel::enabled() {
                // Uploads the proving key to the GPU in the background.
                self::intel::prestart()?;
                return Ok(PrestartedProver::default());
            }
            // Only check that the graph exists here: building the witness
            // program (~1 s) stays on the preload thread, off this path.
            if self::docker::fast_enabled()
                && self::fast::graph_path().map(|p| p.exists()).unwrap_or(false)
            {
                self::fast::prestart()?;
            }
            Ok(PrestartedProver::default())
        } else {
            Ok(PrestartedProver::default())
        }
    }
}

/// Start loading the Groth16 proving key in the background (CUDA; no-op elsewhere, on cards
/// with less than 30 GB, or with `RISC0_CUDA_G16_COLD=0`). The loader waits for
/// [prestart_key_go] (or 2 s).
pub fn prestart_key() {
    #[cfg(feature = "cuda")]
    self::cuda::prestart_key();
}

/// Release a pending [prestart_key] loader (no-op if none is waiting).
pub fn prestart_key_go() {
    #[cfg(feature = "cuda")]
    self::cuda::prestart_key_go();
}

/// Preload the SRS and Groth16 prover into C++ static cache.
/// Call after STARK proving is complete and GPU is idle.
pub fn preload_srs() -> Result<()> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::cuda::preload_srs()
        } else if #[cfg(feature = "rocm")] {
            Ok(()) // hip.rs handles SRS loading internally
        } else if #[cfg(feature = "intel")] {
            if self::intel::enabled() {
                self::intel::preload_srs()
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }
}

/// Free the Groth16 prover and SRS that `shrink_wrap` caches on the GPU, returning that memory to
/// the device. The next `shrink_wrap` rebuilds them; measured, that adds at most ~0.1 s per proof
/// (2.08-2.12 s wraps on an RTX 5080 releasing every time, against 2.01-2.05 s on an RTX 4090 keeping
/// the cache).
///
/// The cache otherwise lives as long as the process, so every later STARK proof on that device has to
/// fit beside it. On a card without the room — measured on a 16 GB RTX 5080, a 46M-cycle STARK at po2
/// 20 peaked at 15.8 of 16.3 GB with the cache resident — call this after each proof. On a card with
/// room, keeping the cache is faster. Deciding which is the caller's business, since only the caller
/// knows what it will prove next.
pub fn release_srs() -> Result<()> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::cuda::release_srs()
        } else if #[cfg(feature = "rocm")] {
            self::hip::release_srs()
        } else if #[cfg(feature = "intel")] {
            if self::intel::enabled() {
                self::intel::release_srs()
            } else {
                Ok(())
            }
        } else {
            // Docker mode holds nothing.
            Ok(())
        }
    }
}

/// Discard the Groth16 prover's derived-data caches: the files the CUDA library keeps in its cache
/// directory (compact coefficients and split-MSM corrections, normally validated by format, key
/// and checksum) and the prover cached in memory. For a caller whose Groth16 seal failed
/// verification: the next [shrink_wrap] rebuilds them from the proving key. No-op without CUDA.
pub fn invalidate_caches() {
    #[cfg(feature = "cuda")]
    self::cuda::invalidate_caches();
}

/// Like [shrink_wrap], but runs `pre_gpu` on the calling thread while the witness is computed on
/// a helper thread (CUDA; elsewhere `pre_gpu` runs first). `pre_gpu` has returned before the GPU
/// part of the wrap starts. Used to overlap releasing the STARK HAL's buffer pool (thread-local,
/// ~35 ms of frees) with the CPU-only witness generation.
pub fn shrink_wrap_with(identity_p254_seal_bytes: &[u8], pre_gpu: impl FnOnce()) -> Result<Seal> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::cuda::shrink_wrap_with(identity_p254_seal_bytes, pre_gpu)
        } else {
            pre_gpu();
            shrink_wrap(identity_p254_seal_bytes)
        }
    }
}

/// Consume a prestarted prover after identity_p254. Dropping the handle on an
/// error or cancellation kills the native helper before the temporary work is removed.
pub fn shrink_wrap_with_startup(
    identity_p254_seal_bytes: &[u8],
    startup: PrestartedProver,
    pre_gpu: impl FnOnce(),
) -> Result<Seal> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "native-groth16")] {
            pre_gpu();
            self::native::shrink_wrap_started(identity_p254_seal_bytes, startup.native)
        } else {
            drop(startup);
            shrink_wrap_with(identity_p254_seal_bytes, pre_gpu)
        }
    }
}

/// Produce a Groth16 proof from an `identity_p254` seal.
pub fn shrink_wrap(identity_p254_seal_bytes: &[u8]) -> Result<Seal> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::cuda::shrink_wrap(identity_p254_seal_bytes)
        } else if #[cfg(feature = "rocm")] {
            self::hip::shrink_wrap(identity_p254_seal_bytes)
        } else if #[cfg(feature = "native-groth16")] {
            self::native::shrink_wrap(identity_p254_seal_bytes)
        } else if #[cfg(feature = "intel")] {
            if self::intel::enabled() {
                self::intel::shrink_wrap(identity_p254_seal_bytes)
            } else {
                self::docker::shrink_wrap(identity_p254_seal_bytes)
            }
        } else {
            self::docker::shrink_wrap(identity_p254_seal_bytes)
        }
    }
}
