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

//! Pinned staging of the preflight trace for the CUDA and HIP witgen upload.
//!
//! The witgen FFI used to copy the trace (cycles, txns, bigint bytes: ~124 MB at po2=20) into
//! a pinned buffer on the proving thread, with the GPU idle (~9.5 ms/segment). The preflight
//! thread now does that copy right after the preflight, into a slot from a small pool of pinned
//! buffers, and the trace owns the slot ([`PreflightTrace::staged`]) until it is dropped (after
//! the eval_check launch, long after the stream has passed the H2D copies, which precede the
//! globals readback in `WitnessGenerator::header`).
//!
//! The pools here are process-wide: the CUDA HAL proves on one device per process (device 0;
//! hosts with several GPUs run a process per GPU with `CUDA_VISIBLE_DEVICES`, as r0vm does). If a
//! thread proves with another device current anyway, the CUDA library stops using the staged
//! traces and [`enabled`] turns off (see `stagingDeviceOk` in ffi.cu); the pinned memory is
//! registered portable, so a trace staged before that is still valid in any context.
//!
//! On ROCm (the same library, built with hipcc) the device is looked up per call, so staging
//! stays on with several devices: a trace prefetched by another device's proving thread
//! (`RISC0_PARALLEL_STARK`) is then copied again on its own device's stream, as unstaged.
//! Measured at po2=20 (RX 7900 XTX / RX 9070 XT), the proving thread's copy into its pinned
//! buffer left the GPU idle ~5 ms/segment and the upload ~8.7 ms more.
//!
//! [`release`] (a HAL release hook) frees what the pools hold.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, LazyLock, Mutex, Weak,
};

use rayon::prelude::*;
use risc0_circuit_rv32im_sys::{
    risc0_circuit_rv32im_cuda_event_wait, risc0_circuit_rv32im_cuda_host_alloc,
    risc0_circuit_rv32im_cuda_host_free, risc0_circuit_rv32im_cuda_prefetch,
    risc0_circuit_rv32im_cuda_release_slots, risc0_circuit_rv32im_cuda_staging_disabled,
    RawMemoryTransaction, RawPreflightCycle,
};
use risc0_sys::ffi_wrap;

use super::{
    bigint::BigIntState,
    preflight::{Back, PreflightTrace},
};
use crate::execute::{poseidon2::Poseidon2State, sha2::Sha2State};
use crate::zirgen::circuit::ExtVal;

/// The rv32im witgen staging feature: pinned staging of the preflight trace on the preflight
/// thread, its prefetch to the device during the previous segment's finalize, the per-cycle
/// columns written on the device, and the recycling of trace vectors. On unless
/// `RISC0_CUDA_WITGEN_STAGE=0`.
pub(crate) fn enabled() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var("RISC0_CUDA_WITGEN_STAGE").as_deref() != Ok("0"));
    *ON && unsafe { risc0_circuit_rv32im_cuda_staging_disabled() } == 0
}

/// Free pinned slots: (address, capacity).
static POOL: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Pinned slots too small for the current traces, freed by [`release`] rather than when they are
/// replaced: unpinning waits for the whole device while holding a driver lock, which would
/// stall the proving thread's launches mid-segment (see `ThpPinned::free_pinned`).
static RETIRED: Mutex<Vec<usize>> = Mutex::new(Vec::new());
const MAX_RETIRED: usize = 4;

/// HAL release hook (`risc0_zkp::hal::{cuda,hip}::register_release_hook`): the device buffers
/// of the staged-trace slots not in use, and, unless `device_only`, the pinned slots, the
/// recycled trace vectors and word-transaction maps (all rebuilt on demand). Never blocks on the
/// pools' locks.
pub(crate) fn release(device_only: bool) {
    if let Err(err) = ffi_wrap(|| unsafe { risc0_circuit_rv32im_cuda_release_slots() }) {
        tracing::debug!("cuda_stage: release_slots failed: {err}");
    }
    if device_only {
        return;
    }
    let mut free: Vec<usize> = Vec::new();
    if let Ok(mut pool) = POOL.try_lock() {
        free.extend(pool.drain(..).map(|(ptr, _)| ptr));
    }
    if let Ok(mut retired) = RETIRED.try_lock() {
        free.append(&mut retired);
    }
    for ptr in free {
        let _ = ffi_wrap(|| unsafe { risc0_circuit_rv32im_cuda_host_free(ptr as *mut _) });
    }
    if let Ok(mut pool) = TRACE_POOL.try_lock() {
        pool.clear();
    }
    super::paged_map::WordTxnMap::trim_pool();
}

pub(crate) struct StagedTrace {
    ptr: *mut u8,
    cap: usize,
    /// Byte lengths of cycles, txns and bigint bytes, back to back from `ptr`.
    lens: [usize; 3],
    /// Unique per staging (the device copy is matched by it, not by address).
    pub(crate) id: u64,
    /// Event recorded after this slot's prefetch copy (0: none). The DMA reads the pinned
    /// slot asynchronously, so Drop waits for it before the slot goes back to the pool.
    copy_ev: Mutex<usize>,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Staged traces not yet prefetched (pushed by the preflight thread).
static READY: Mutex<Vec<Weak<StagedTrace>>> = Mutex::new(Vec::new());

/// Start the device copy of the staged traces that are ready, on a side stream overlapping the
/// GPU work in flight. Call on the proving thread (GPU lock held), between segments' GPU work:
/// the CUDA HAL calls it once eval_check is done, while the finalize kernels run (the HIP HAL
/// also just before eval_check).
pub(crate) fn prefetch_ready() {
    if !enabled() {
        return;
    }
    let ready: Vec<Weak<StagedTrace>> = std::mem::take(&mut *READY.lock().unwrap());
    for weak in ready {
        // Dropped (already proved): nothing to do.
        if let Some(staged) = weak.upgrade() {
            let total: usize = staged.lens.iter().sum();
            let mut ev: *mut std::ffi::c_void = std::ptr::null_mut();
            if let Err(err) = ffi_wrap(|| unsafe {
                risc0_circuit_rv32im_cuda_prefetch(staged.ptr, total, staged.id, &mut ev)
            }) {
                tracing::debug!("cuda_stage: prefetch failed: {err}");
            }
            if !ev.is_null() {
                let old = std::mem::replace(&mut *staged.copy_ev.lock().unwrap(), ev as usize);
                if old != 0 {
                    let _ =
                        ffi_wrap(|| unsafe { risc0_circuit_rv32im_cuda_event_wait(old as *mut _) });
                }
            }
            // `staged` (the Arc) may be the last reference: its Drop waits for the copy.
        }
    }
}

// SAFETY: the slot is plain pinned host memory owned by this value.
unsafe impl Send for StagedTrace {}
unsafe impl Sync for StagedTrace {}

impl StagedTrace {
    /// The staged bytes, if they are those of `trace` (same lengths).
    pub(crate) fn ptr_for(&self, trace: &PreflightTrace) -> Option<*const u8> {
        (self.lens == byte_lens(trace)).then_some(self.ptr as *const u8)
    }
}

impl Drop for StagedTrace {
    fn drop(&mut self) {
        // The prefetch DMA may still be reading the slot (e.g. a trace dropped unproven on an
        // error path): wait for it before another trace can be staged into the slot.
        let ev = *self.copy_ev.get_mut().unwrap_or_else(|e| e.into_inner());
        if ev != 0 {
            if let Err(err) =
                ffi_wrap(|| unsafe { risc0_circuit_rv32im_cuda_event_wait(ev as *mut _) })
            {
                // Don't recycle a slot that may still be read by the device.
                tracing::warn!("cuda_stage: prefetch event wait failed: {err}; leaking slot");
                return;
            }
        }
        POOL.lock().unwrap().push((self.ptr as usize, self.cap));
    }
}

fn byte_lens(trace: &PreflightTrace) -> [usize; 3] {
    [
        std::mem::size_of_val(trace.cycles.as_slice()),
        std::mem::size_of_val(trace.txns.as_slice()),
        trace.bigint_bytes.len(),
    ]
}

fn take_slot(len: usize) -> Option<(*mut u8, usize)> {
    let mut stale = Vec::new();
    let found = {
        let mut pool = POOL.lock().unwrap();
        match pool.iter().position(|&(_, cap)| cap >= len) {
            Some(i) => Some(pool.swap_remove(i)),
            None => {
                // Too small for this trace: free one (the trace grows over the first segments).
                if let Some(slot) = pool.pop() {
                    stale.push(slot);
                }
                None
            }
        }
    };
    let mut overflow = Vec::new();
    {
        let mut retired = RETIRED.lock().unwrap();
        retired.extend(stale.into_iter().map(|(ptr, _)| ptr));
        // Bounded: the traces stop growing after the first segments, so this is rare.
        while retired.len() > MAX_RETIRED {
            overflow.push(retired.remove(0));
        }
    }
    for ptr in overflow {
        let _ = ffi_wrap(|| unsafe { risc0_circuit_rv32im_cuda_host_free(ptr as *mut _) });
    }
    if let Some((ptr, cap)) = found {
        return Some((ptr as *mut u8, cap));
    }
    let cap = len + len / 4;
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    match ffi_wrap(|| unsafe { risc0_circuit_rv32im_cuda_host_alloc(cap, &mut ptr) }) {
        Ok(()) if !ptr.is_null() => Some((ptr as *mut u8, cap)),
        Ok(()) => None,
        Err(err) => {
            tracing::debug!("cuda_stage: pinned alloc of {cap} bytes failed: {err}");
            None
        }
    }
}

fn par_copy(dst: *mut u8, src: &[u8]) {
    const CHUNK: usize = 4 << 20;
    // One serial copy unless RISC0_CUDA_SERIAL_PREFLIGHT=0 (see `SERIAL_PREFLIGHT`).
    if *super::SERIAL_PREFLIGHT {
        // SAFETY: `dst` has room for `src` (see `stage`).
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        return;
    }
    let dst = dst as usize;
    src.par_chunks(CHUNK)
        .enumerate()
        .for_each(|(i, chunk)| unsafe {
            std::ptr::copy_nonoverlapping(
                chunk.as_ptr(),
                (dst as *mut u8).add(i * CHUNK),
                chunk.len(),
            );
        });
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data repr(C) structs / bytes, read only.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Copy `trace`'s device inputs into a pinned slot (None: staging off or no pinned memory, the
/// witgen FFI then stages on the proving thread as before).
pub(crate) fn stage(trace: &PreflightTrace) -> Option<Arc<StagedTrace>> {
    if !enabled() {
        return None;
    }
    let lens = byte_lens(trace);
    let total: usize = lens.iter().sum();
    let (ptr, cap) = take_slot(total.max(1))?;
    let cycles: &[RawPreflightCycle] = &trace.cycles;
    let txns: &[RawMemoryTransaction] = &trace.txns;
    par_copy(ptr, as_bytes(cycles));
    par_copy(unsafe { ptr.add(lens[0]) }, as_bytes(txns));
    par_copy(unsafe { ptr.add(lens[0] + lens[1]) }, &trace.bigint_bytes);
    let staged = Arc::new(StagedTrace {
        ptr,
        cap,
        lens,
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        copy_ev: Mutex::new(0),
    });
    READY.lock().unwrap().push(Arc::downgrade(&staged));
    Some(staged)
}

// Reuse the trace vectors of finished segments: a fresh ~300 MB of vectors per po2=20 segment
// costs the preflight thread page faults and zeroing, and the txns vector its doubling
// reallocations.

type TraceVecs = (
    Vec<RawPreflightCycle>,
    Vec<RawMemoryTransaction>,
    Vec<u8>,
    Vec<Back>,
    (Vec<Poseidon2State>, Vec<Sha2State>, Vec<BigIntState>),
);
static TRACE_POOL: Mutex<Vec<TraceVecs>> = Mutex::new(Vec::new());

pub(crate) fn recycled_trace(total_cycles: usize, rand_z: ExtVal) -> PreflightTrace {
    let vecs = if enabled() {
        TRACE_POOL.lock().unwrap().pop()
    } else {
        None
    };
    let (mut cycles, mut txns, bigint_bytes, mut backs, side) = vecs.unwrap_or_default();
    // A fresh (not recycled) trace: reserve it up front and back it with huge pages before
    // the first touch (see `advise_huge`).
    let fresh = cycles.capacity() == 0;
    cycles.reserve(total_cycles);
    backs.reserve(total_cycles);
    if fresh {
        txns.reserve(total_cycles * 5);
        super::advise_huge(&cycles);
        super::advise_huge(&txns);
        super::advise_huge(&backs);
    }
    // (No struct update syntax: PreflightTrace implements Drop.)
    let mut trace = PreflightTrace::default();
    trace.cycles = cycles;
    trace.txns = txns;
    trace.bigint_bytes = bigint_bytes;
    trace.backs = backs;
    (trace.p2_states, trace.sha2_states, trace.bigint_states) = side;
    trace.rand_z = rand_z;
    trace
}

pub(crate) fn recycle_trace(trace: &mut PreflightTrace) {
    if !enabled() || trace.cycles.capacity() == 0 {
        return;
    }
    let mut vecs = (
        std::mem::take(&mut trace.cycles),
        std::mem::take(&mut trace.txns),
        std::mem::take(&mut trace.bigint_bytes),
        std::mem::take(&mut trace.backs),
        (
            std::mem::take(&mut trace.p2_states),
            std::mem::take(&mut trace.sha2_states),
            std::mem::take(&mut trace.bigint_states),
        ),
    );
    vecs.0.clear();
    vecs.1.clear();
    vecs.2.clear();
    vecs.3.clear();
    vecs.4 .0.clear();
    vecs.4 .1.clear();
    vecs.4 .2.clear();
    let mut pool = TRACE_POOL.lock().unwrap();
    if pool.len() < 2 {
        pool.push(vecs);
    }
}
