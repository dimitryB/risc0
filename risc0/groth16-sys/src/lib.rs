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

use std::{
    ffi::{c_char, CStr, CString, NulError},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub use sppark::Error as SpparkError;

pub struct SetupParams {
    pub pcoeffs_path: RawPath,
    pub fres_path: RawPath,
    pub srs_path: RawPath,
}

impl SetupParams {
    pub fn new(root_dir: &Path) -> anyhow::Result<Self> {
        Ok(SetupParams {
            pcoeffs_path: root_dir.join("preprocessed_coeffs.bin").try_into()?,
            fres_path: root_dir.join("fuzzed_msm_results.bin").try_into()?,
            srs_path: root_dir.join("stark_verify_final.zkey").try_into()?,
        })
    }
}

pub struct WitnessParams {
    pub graph_path: PathBuf,
}

impl WitnessParams {
    pub fn new(root_dir: &Path) -> Self {
        WitnessParams {
            graph_path: root_dir.join("stark_verify_graph.bin"),
        }
    }
}

pub struct ProverParams {
    pub public_path: RawPath,
    pub proof_path: RawPath,
    pub witness: *const u8,
}

impl ProverParams {
    pub fn new(root_dir: &Path, witness: *const u8) -> anyhow::Result<Self> {
        Ok(Self {
            public_path: root_dir.join("public.json").try_into()?,
            proof_path: root_dir.join("proof.json").try_into()?,
            witness,
        })
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn preload(setup_params: &SetupParams) -> anyhow::Result<()> {
    let setup_params = RawSetupParams {
        pcoeffs_path: setup_params.pcoeffs_path.c_str.as_ptr(),
        fres_path: setup_params.fres_path.c_str.as_ptr(),
        srs_path: setup_params.srs_path.c_str.as_ptr(),
    };
    ffi_wrap(|| unsafe { risc0_groth16_cuda_preload(&setup_params) })
}

/// Load the proving key and build the prover on the current device without taking any
/// lock other than the cache's, through pinned staging (safe to run beside a STARK proof).
/// Returns the witness length when it loaded (or found) the key, `None` when skipped because the
/// card has less than `min_total_vram` bytes.
#[cfg(feature = "cuda")]
pub fn prestart(
    setup_params: &SetupParams,
    min_total_vram: usize,
) -> anyhow::Result<Option<usize>> {
    let raw = RawSetupParams {
        pcoeffs_path: setup_params.pcoeffs_path.c_str.as_ptr(),
        fres_path: setup_params.fres_path.c_str.as_ptr(),
        srs_path: setup_params.srs_path.c_str.as_ptr(),
    };
    let mut len: usize = 0;
    let mut loaded: i32 = 0;
    ffi_wrap(|| unsafe {
        risc0_groth16_cuda_prestart(&raw, min_total_vram, &mut len, &mut loaded)
    })?;
    Ok((loaded != 0).then_some(len))
}

/// The witness indices that are always 0/1 (the bit set of the split MSMs), used by the
/// next prover construction. Takes effect only if set before the prover is built.
#[cfg(feature = "cuda")]
pub fn set_bit_signals(bits: &[u32]) {
    unsafe { risc0_groth16_cuda_set_bit_signals(bits.as_ptr(), bits.len()) }
}

/// Free the Groth16 prover and SRS cached on the current device, returning their GPU memory. The next
/// `prove` rebuilds them. See `risc0_groth16_cuda_release`.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn release() -> anyhow::Result<()> {
    ffi_wrap(|| unsafe { risc0_groth16_cuda_release() })
}

/// The process-wide pinned host buffer for the witness (see `risc0_groth16_cuda_host_buffer`),
/// held exclusively: other callers wait until this is dropped, so keep it until the `prove`
/// that reads it has returned. Unlike a `MutexGuard` it may be dropped on another thread than
/// the one that took it (the witness is computed on a helper thread, see `shrink_wrap_with`).
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub struct HostBuffer {
    ptr: *mut u8,
    len: usize,
}

// Safety: the buffer is a process-lifetime pinned allocation (it is only freed or regrown by a
// holder of the HOST_BUFFER_BUSY flag), and only the HostBuffer holding the flag touches it.
#[cfg(any(feature = "cuda", feature = "rocm"))]
unsafe impl Send for HostBuffer {}

/// Whether a [HostBuffer] is live; the condvar wakes waiters when it is dropped.
#[cfg(any(feature = "cuda", feature = "rocm"))]
static HOST_BUFFER_BUSY: (std::sync::Mutex<bool>, std::sync::Condvar) =
    (std::sync::Mutex::new(false), std::sync::Condvar::new());

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn host_buffer_acquire() {
    let (m, cv) = &HOST_BUFFER_BUSY;
    let mut busy = m.lock().unwrap_or_else(|e| e.into_inner());
    while *busy {
        busy = cv.wait(busy).unwrap_or_else(|e| e.into_inner());
    }
    *busy = true;
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn host_buffer_release() {
    let (m, cv) = &HOST_BUFFER_BUSY;
    *m.lock().unwrap_or_else(|e| e.into_inner()) = false;
    cv.notify_one();
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl Drop for HostBuffer {
    fn drop(&mut self) {
        host_buffer_release();
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl HostBuffer {
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// The buffer as `len / 32` 32-byte field elements (four little-endian u64 limbs).
    pub fn as_limbs_mut(&mut self) -> &mut [[u64; 4]] {
        // Safety: ptr is a page-aligned allocation of at least len bytes that only the holder
        // of the busy flag uses.
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut [u64; 4], self.len / 32) }
    }
}

/// Lock the pinned witness buffer, growing it to at least `bytes`.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn host_buffer(bytes: usize) -> anyhow::Result<HostBuffer> {
    host_buffer_acquire();
    let ptr = unsafe { risc0_groth16_cuda_host_buffer(bytes) };
    if ptr.is_null() {
        host_buffer_release();
        return Err(anyhow!("pinned host allocation of {bytes} bytes failed"));
    }
    Ok(HostBuffer { ptr, len: bytes })
}

/// Create the pinned upload stager of the key loader now (see
/// `risc0_groth16_cuda_stager_init`). Returns false, allocating nothing, on a card with less than
/// `min_total_vram` bytes.
#[cfg(feature = "cuda")]
pub fn stager_init(min_total_vram: usize) -> anyhow::Result<bool> {
    let mut created: i32 = 0;
    ffi_wrap(|| unsafe { risc0_groth16_cuda_stager_init(min_total_vram, &mut created) })?;
    Ok(created != 0)
}

/// Free the stager and the pinned witness buffer (both are recreated on demand), e.g. when
/// they were allocated early for a key prestart that then gave up. Waits for a live [HostBuffer].
#[cfg(feature = "cuda")]
pub fn pinned_release() {
    host_buffer_acquire();
    unsafe { risc0_groth16_cuda_pinned_release() };
    host_buffer_release();
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn prove(prover_params: &ProverParams, setup_params: &SetupParams) -> anyhow::Result<()> {
    let setup_params = RawSetupParams {
        pcoeffs_path: setup_params.pcoeffs_path.c_str.as_ptr(),
        fres_path: setup_params.fres_path.c_str.as_ptr(),
        srs_path: setup_params.srs_path.c_str.as_ptr(),
    };
    let prover_params = RawProverParams {
        public_path: prover_params.public_path.c_str.as_ptr(),
        proof_path: prover_params.proof_path.c_str.as_ptr(),
        witness: prover_params.witness,
    };

    ffi_wrap(|| unsafe { risc0_groth16_cuda_prove(&setup_params, &prover_params) })
}

/// Load the proving key and coefficients onto the Intel GPU, where they stay cached until
/// [intel_release]. [intel_prove] does this itself if needed.
#[cfg(feature = "intel")]
pub fn intel_preload(setup_params: &SetupParams) -> anyhow::Result<()> {
    let setup_params = setup_params.raw();
    ffi_wrap(|| unsafe { risc0_groth16_intel_preload(&setup_params) })
}

/// Prove with the Intel GPU prover; writes the proof to `prover_params.proof_path`.
#[cfg(feature = "intel")]
pub fn intel_prove(prover_params: &ProverParams, setup_params: &SetupParams) -> anyhow::Result<()> {
    let setup_params = setup_params.raw();
    let prover_params = RawProverParams {
        public_path: prover_params.public_path.c_str.as_ptr(),
        proof_path: prover_params.proof_path.c_str.as_ptr(),
        witness: prover_params.witness,
    };
    ffi_wrap(|| unsafe { risc0_groth16_intel_prove(&setup_params, &prover_params) })
}

/// Free the Intel GPU prover's cached proving key, coefficients and scratch (several GB).
#[cfg(feature = "intel")]
pub fn intel_release() -> anyhow::Result<()> {
    ffi_wrap(|| unsafe { risc0_groth16_intel_release() })
}

#[cfg(feature = "intel")]
impl SetupParams {
    fn raw(&self) -> RawSetupParams {
        RawSetupParams {
            pcoeffs_path: self.pcoeffs_path.c_str.as_ptr(),
            fres_path: self.fres_path.c_str.as_ptr(),
            srs_path: self.srs_path.c_str.as_ptr(),
        }
    }
}

#[cfg(all(any(feature = "cuda", feature = "rocm"), feature = "setup"))]
pub fn setup(params: &SetupParams) -> anyhow::Result<()> {
    let raw_params = RawSetupParams {
        pcoeffs_path: params.pcoeffs_path.c_str.as_ptr(),
        fres_path: params.fres_path.c_str.as_ptr(),
        srs_path: params.srs_path.c_str.as_ptr(),
    };
    ffi_wrap(|| unsafe { risc0_groth16_cuda_setup(&raw_params) })
}

#[cfg_attr(
    not(any(feature = "cuda", feature = "rocm", feature = "intel")),
    allow(dead_code)
)]
#[repr(C)]
struct RawProverParams {
    pub public_path: *const c_char,
    pub proof_path: *const c_char,
    pub witness: *const u8,
}

#[cfg_attr(
    not(any(feature = "cuda", feature = "rocm", feature = "intel")),
    allow(dead_code)
)]
#[repr(C)]
struct RawSetupParams {
    pub pcoeffs_path: *const c_char,
    pub fres_path: *const c_char,
    pub srs_path: *const c_char,
}

extern "C" {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn risc0_groth16_cuda_prove(
        setup: *const RawSetupParams,
        params: *const RawProverParams,
    ) -> *const c_char;

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn risc0_groth16_cuda_preload(setup: *const RawSetupParams) -> *const c_char;

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn risc0_groth16_cuda_release() -> *const c_char;

    #[cfg(feature = "cuda")]
    fn risc0_groth16_cuda_set_bit_signals(bits: *const u32, len: usize);

    #[cfg(feature = "cuda")]
    fn risc0_groth16_cuda_prestart(
        setup: *const RawSetupParams,
        min_total_vram: usize,
        witness_len: *mut usize,
        loaded: *mut i32,
    ) -> *const c_char;

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn risc0_groth16_cuda_host_buffer(bytes: usize) -> *mut u8;

    #[cfg(feature = "cuda")]
    fn risc0_groth16_cuda_stager_init(min_total_vram: usize, created: *mut i32) -> *const c_char;

    #[cfg(feature = "cuda")]
    fn risc0_groth16_cuda_pinned_release();

    #[cfg(feature = "intel")]
    fn risc0_groth16_intel_preload(setup: *const RawSetupParams) -> *const c_char;

    #[cfg(feature = "intel")]
    fn risc0_groth16_intel_prove(
        setup: *const RawSetupParams,
        params: *const RawProverParams,
    ) -> *const c_char;

    #[cfg(feature = "intel")]
    fn risc0_groth16_intel_release() -> *const c_char;

    #[cfg(all(any(feature = "cuda", feature = "rocm"), feature = "setup"))]
    fn risc0_groth16_cuda_setup(params: *const RawSetupParams) -> *const c_char;
}

#[cfg_attr(
    not(any(feature = "cuda", feature = "rocm", feature = "intel")),
    allow(dead_code)
)]
fn ffi_wrap<F>(mut inner: F) -> Result<()>
where
    F: FnMut() -> *const c_char,
{
    extern "C" {
        fn free(str: *const c_char);
    }

    let c_ptr = inner();
    if c_ptr.is_null() {
        Ok(())
    } else {
        let what = unsafe {
            let msg = CStr::from_ptr(c_ptr)
                .to_str()
                .unwrap_or("Invalid error msg pointer")
                .to_string();
            free(c_ptr);
            msg
        };
        Err(anyhow!(what))
    }
}

#[cfg_attr(
    not(any(feature = "cuda", feature = "rocm", feature = "intel")),
    allow(dead_code)
)]
pub struct RawPath {
    path: PathBuf,
    c_str: CString,
}

impl RawPath {
    pub fn as_path(&self) -> &Path {
        &self.path
    }
}

impl TryFrom<&Path> for RawPath {
    type Error = NulError;

    fn try_from(value: &Path) -> Result<Self, Self::Error> {
        Ok(RawPath {
            path: value.to_path_buf(),
            c_str: CString::new(value.as_os_str().as_encoded_bytes())?,
        })
    }
}

impl TryFrom<PathBuf> for RawPath {
    type Error = NulError;

    fn try_from(value: PathBuf) -> Result<Self, Self::Error> {
        RawPath::try_from(value.as_path())
    }
}
