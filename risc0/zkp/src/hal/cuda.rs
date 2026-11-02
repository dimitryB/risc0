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
    cell::RefCell, collections::HashMap, fmt::Debug, marker::PhantomData, mem::ManuallyDrop,
    rc::Rc, sync::OnceLock,
};

use anyhow::{bail, Context as _, Result};
use cust::{
    device::DeviceAttribute,
    memory::{DeviceCopy, DevicePointer, GpuBuffer},
    prelude::*,
};
use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use risc0_core::{
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        Elem, ExtElem, RootsOfUnity,
    },
    scope,
};
use risc0_sys::{cuda::*, ffi_wrap};

use super::{tracker, Buffer, Hal};
use crate::{
    core::{
        digest::Digest,
        hash::{
            poseidon2::Poseidon2HashSuite, poseidon_254::Poseidon254HashSuite,
            sha::Sha256HashSuite, HashSuite,
        },
        log2_ceil,
    },
    FRI_FOLD,
};

/// Thread-local pool of CUDA device buffers, keyed by byte size.
/// Reuses freed buffers to avoid cuMemAlloc/cuMemFree overhead (~20ms/proof).
struct BufferPool {
    cache: HashMap<usize, Vec<DeviceBuffer<u8>>>,
    total_cached: usize,
}

const BUFFER_POOL_MAX_BYTES: usize = 16 << 30; // 16 GB (floor of the cap)
/// Headroom the pool cap leaves below the device's total memory.
const BUFFER_POOL_HEADROOM: usize = 8 << 30; // 8 GB

/// Cap on the bytes of large buffers the pool caches: max(16 GB, device total - 8 GB), i.e.
/// (of device 0, the HAL's only device: see `CudaHal::new_from_hash`; hosts with several GPUs run a
/// process per GPU with `CUDA_VISIBLE_DEVICES`, as r0vm does),
/// 24 GB on a 32 GB card, where the fixed 16 GB cap freed (and later re-allocated with a cold
/// cuMemAlloc) part of a po2=21 segment's working set every segment. RISC0_CUDA_POOL_MAX_GB
/// overrides it (16 = the previous behavior). Evaluated on the first push, when a context is
/// current.
static POOL_MAX: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    if let Some(g) = std::env::var("RISC0_CUDA_POOL_MAX_GB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return g << 30;
    }
    match cust::memory::mem_get_info() {
        Ok((_free, total)) => BUFFER_POOL_MAX_BYTES.max(total.saturating_sub(BUFFER_POOL_HEADROOM)),
        Err(_) => BUFFER_POOL_MAX_BYTES,
    }
});
const BUFFER_POOL_SMALL_THRESHOLD: usize = 64 << 10; // 64 KB - always pool small buffers

impl BufferPool {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            total_cached: 0,
        }
    }

    fn pop(&mut self, size: usize) -> Option<DeviceBuffer<u8>> {
        if let Some(bufs) = self.cache.get_mut(&size) {
            if let Some(buf) = bufs.pop() {
                self.total_cached -= size;
                return Some(buf);
            }
        }
        None
    }

    fn push(&mut self, size: usize, buf: DeviceBuffer<u8>) {
        // Always pool small buffers (cuMemFree overhead > storage cost).
        // Only enforce cap for large buffers.
        if size >= BUFFER_POOL_SMALL_THRESHOLD && self.total_cached + size > *POOL_MAX {
            drop(buf);
            return;
        }
        self.total_cached += size;
        self.cache.entry(size).or_default().push(buf);
    }

    /// Release every cached buffer. Used as a last-resort when a large
    /// allocation fails so the driver can reclaim fragmented device memory.
    fn drain(&mut self) {
        self.cache.clear();
        self.total_cached = 0;
    }
}

thread_local! {
    static BUFFER_POOL: RefCell<BufferPool> = RefCell::new(BufferPool::new());
}

// Raw CUDA driver bindings for memory pool trimming, needed on large-alloc
// retry paths: cuMemAlloc will fail against a fragmented-but-technically-free
// device address space if the stream-ordered mempool is hoarding reservations.
// Cust doesn't surface cuMemPoolTrimTo, so bind it directly.
#[repr(C)]
struct CuMemPoolOpaque {
    _private: [u8; 0],
}
type CuMemPool = *mut CuMemPoolOpaque;
extern "C" {
    fn cuDeviceGetDefaultMemPool(pool: *mut CuMemPool, dev: i32) -> i32;
    fn cuMemPoolTrimTo(pool: CuMemPool, min_bytes_to_keep: usize) -> i32;
    fn cuCtxSynchronize() -> i32;
}

fn trim_cuda_mempool() {
    unsafe {
        // Flush in-flight stream-ordered frees back to the pool, then trim to zero.
        let _ = cuCtxSynchronize();
        let mut pool: CuMemPool = std::ptr::null_mut();
        if cuDeviceGetDefaultMemPool(&mut pool, 0) == 0 && !pool.is_null() {
            let _ = cuMemPoolTrimTo(pool, 0);
        }
    }
}

/// Called by raw device allocations in the kernel libraries (`risc0_cuda_malloc` in
/// `device_alloc.h`: nttq scratch, the rv32im preflight slots, sort keys) when cudaMalloc fails:
/// releases this thread's cached buffers and trims the mempool, so the allocation can be retried,
/// as `RawBuffer::new` does for the HAL's own allocations. Those callers run on the proving
/// thread, which owns the pool that holds the memory.
extern "C" fn drain_pool_on_oom() {
    let _ = std::panic::catch_unwind(|| {
        run_release_hooks(true);
        let released = BUFFER_POOL
            .try_with(|pool| {
                pool.try_borrow_mut()
                    .map(|mut pool| {
                        let cached = pool.total_cached;
                        pool.drain();
                        cached
                    })
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        trim_cuda_mempool();
        tracing::debug!(
            "raw device allocation failed: released {:.2} GiB of cached buffers, retrying",
            released as f64 / (1u64 << 30) as f64
        );
    });
}

/// Caches kept outside this HAL's pool by the circuit crates: device scratch of their kernel
/// libraries, and host staging pools (pinned buffers, recycled trace vectors).
static RELEASE_HOOKS: std::sync::Mutex<Vec<fn(bool)>> = std::sync::Mutex::new(Vec::new());

/// Register `hook` (once; repeated registrations of the same function are ignored) to release a
/// cache that `clear_buffer_pool` cannot see. It is called with `device_only = false` by
/// [clear_buffer_pool] (free everything that can be rebuilt: device scratch not in use, host
/// pools), and with `device_only = true` when a raw device allocation fails on the proving thread
/// (free device memory not in use right now). It must not block on locks its own code may hold
/// around a device allocation (use `try_lock`).
pub fn register_release_hook(hook: fn(bool)) {
    let mut hooks = RELEASE_HOOKS.lock().unwrap_or_else(|e| e.into_inner());
    if !hooks.iter().any(|h| std::ptr::fn_addr_eq(*h, hook)) {
        hooks.push(hook);
    }
}

fn run_release_hooks(device_only: bool) {
    let hooks = RELEASE_HOOKS
        .try_lock()
        .map(|h| h.clone())
        .unwrap_or_default();
    for hook in hooks {
        hook(device_only);
    }
}

fn register_oom_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe { risc0_sys::cuda::risc0_zkp_cuda_set_oom_hook(drain_pool_on_oom) });
}

/// Release every device buffer this thread's `BUFFER_POOL` is caching, and trim the CUDA
/// stream-ordered mempool back to the driver. The CUDA counterpart of `hal::hip::clear_buffer_pool`.
///
/// The pool keeps freed buffers for reuse and is capped at `BUFFER_POOL_MAX_BYTES` (16 GB), which on
/// a 16 GB card bounds nothing: after a STARK proof it can be holding most of the device. Allocations
/// made through this HAL recover from that on failure (the drain-and-retry in `RawBuffer::new`), but
/// code that allocates on its own has no such fallback. The Groth16 prover is that code — it uploads
/// its SRS and allocates MSM scratch through sppark's `cudaMallocAsync` — so on an RTX 5080 (16.3 GB)
/// a 46M-cycle proof finished its whole STARK, left ~10.5 GB cached here, and then failed in the wrap.
/// Call this when the HAL's work is done and someone else's is about to begin.
///
/// Thread-local, like the pool: it releases only what the CALLING thread cached.
pub fn clear_buffer_pool() {
    run_release_hooks(false);
    let released = BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let cached = pool.total_cached;
        pool.drain();
        cached
    });
    trim_cuda_mempool();
    tracing::info!(
        "clear_buffer_pool: released {:.2} GiB of cached device buffers and trimmed the CUDA mempool",
        released as f64 / (1u64 << 30) as f64
    );
}

// The GPU becomes unstable as the number of concurrent provers grow.
//
// Held for a whole proving call by the code that proves (`exclusive`; the rv32im session, each
// recursion and keccak proof, the Groth16 wrap), and by `CudaHal::new` while it initializes. Not
// for a `CudaHal`'s lifetime: the provers cache their HALs per thread, so a HAL holding it would
// keep it as long as its thread lives, and every other thread that proves would wait forever.
pub fn singleton() -> &'static ReentrantMutex<()> {
    static ONCE: OnceLock<ReentrantMutex<()>> = OnceLock::new();
    ONCE.get_or_init(|| ReentrantMutex::new(()))
}

/// Exclusive use of the GPU for one proving call, against other threads: hold the returned guard
/// for the call. Reentrant, so a call nested in another on the same thread goes straight through.
pub fn exclusive() -> ReentrantMutexGuard<'static, ()> {
    singleton().lock()
}

/// Total memory of CUDA device 0 in bytes, or None if there is no usable device.
pub fn device_total_memory() -> Option<usize> {
    cust::init(CudaFlags::empty()).ok()?;
    cust::device::Device::get_device(0)
        .and_then(|d| d.total_memory())
        .ok()
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DeviceElem(pub BabyBearElem);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DeviceExtElem(pub BabyBearExtElem);

unsafe impl DeviceCopy for DeviceExtElem {}

pub trait CudaHash {
    /// Create a hash implementation
    fn new() -> Self
    where
        Self: Sized;

    /// Run the hash_fold function
    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize);

    /// Run the hash_fold_tree function (fold all layers in a single FFI call)
    fn hash_fold_tree(&self, io: &BufferImpl<Digest>, layers: usize) {
        // Default: fall back to per-level hash_fold
        for i in (0..layers).rev() {
            let layer_size = 1 << i;
            self.hash_fold(io, layer_size);
        }
    }

    /// Run the hash_rows function
    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>);

    /// Return the HashSuite
    fn get_hash_suite(&self) -> &HashSuite<BabyBear>;
}

pub struct CudaHashSha256 {
    suite: HashSuite<BabyBear>,
}

impl CudaHash for CudaHashSha256 {
    fn new() -> Self {
        CudaHashSha256 {
            suite: Sha256HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        let input = io.as_device_ptr_with_offset(2 * output_size);
        let output = io.as_device_ptr_with_offset(output_size);

        extern "C" {
            fn risc0_zkp_cuda_sha_fold(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe { risc0_zkp_cuda_sha_fold(output, input, output_size as u32) }).unwrap();
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);

        extern "C" {
            fn risc0_zkp_cuda_sha_rows(
                output: DevicePointer<u8>,
                matrix: DevicePointer<u8>,
                row_size: u32,
                col_size: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_sha_rows(
                output.as_device_ptr(),
                matrix.as_device_ptr(),
                row_size as u32,
                col_size as u32,
            )
        })
        .unwrap();
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

/// `DeviceBuffer::copy_from` from pageable memory (cuMemcpyHtoD) returns once the data is in the
/// driver's staging buffer; the DMA to the device may still be in flight on the legacy stream.
/// Kernels on the persistent stream (a blocking stream) are ordered after it, but sppark's
/// kernels run on non-blocking streams and could read the buffer before the copy lands
/// (hash_fold_tree read stale leaves about 1 run in 10 at 2^14 leaves). Wait for the legacy
/// stream, which holds only that copy. `set_32` (cuMemsetD32) is asynchronous on the legacy
/// stream as well and gets the same wait.
fn wait_pageable_upload() {
    let res = unsafe { cust::sys::cuStreamSynchronize(std::ptr::null_mut()) };
    assert_eq!(
        res,
        cust::sys::cudaError_enum::CUDA_SUCCESS,
        "cuStreamSynchronize failed"
    );
}

/// Wait, on the host, for everything queued on the risc0 persistent stream.
///
/// sppark's kernels (the hashes, NTTs, zk_shift) run on sppark's own non-blocking stream, which is
/// not ordered after the persistent stream. Anything they read that a risc0 kernel wrote has to be
/// complete first, or they read it half-written.
fn sync_persistent_stream() {
    extern "C" {
        fn risc0_zkp_cuda_sync_stream() -> *const std::os::raw::c_char;
    }
    ffi_wrap(|| unsafe { risc0_zkp_cuda_sync_stream() }).unwrap();
}

pub struct CudaHashPoseidon2 {
    suite: HashSuite<BabyBear>,
}

impl CudaHash for CudaHashPoseidon2 {
    fn new() -> Self {
        CudaHashPoseidon2 {
            suite: Poseidon2HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        let err = unsafe {
            let input = io.as_device_ptr_with_offset(2 * output_size);
            let output = io.as_device_ptr_with_offset(output_size);
            sync_persistent_stream();
            sppark_poseidon2_fold(output, input, output_size)
        };
        if err.code != 0 {
            panic!("Failure during hash_fold: {err}");
        }
    }

    fn hash_fold_tree(&self, io: &BufferImpl<Digest>, layers: usize) {
        sync_persistent_stream();
        let err = unsafe { sppark_poseidon2_fold_tree(io.as_device_ptr(), layers as u32) };
        if err.code != 0 {
            panic!("Failure during hash_fold_tree: {err}");
        }
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);

        sync_persistent_stream();
        let err = unsafe {
            sppark_poseidon2_rows(
                output.as_device_ptr(),
                matrix.as_device_ptr(),
                row_size.try_into().unwrap(),
                col_size.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during hash_rows: {err}");
        }
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

pub struct CudaHashPoseidon254 {
    suite: HashSuite<BabyBear>,
}

impl CudaHash for CudaHashPoseidon254 {
    fn new() -> Self {
        CudaHashPoseidon254 {
            suite: Poseidon254HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        let err = unsafe {
            let input = io.as_device_ptr_with_offset(2 * output_size);
            let output = io.as_device_ptr_with_offset(output_size);
            sync_persistent_stream();
            sppark_poseidon254_fold(output, input, output_size)
        };
        if err.code != 0 {
            panic!("Failure during hash_fold: {err}");
        }
    }

    fn hash_fold_tree(&self, io: &BufferImpl<Digest>, layers: usize) {
        sync_persistent_stream();
        let err = unsafe { sppark_poseidon254_fold_tree(io.as_device_ptr(), layers as u32) };
        if err.code != 0 {
            panic!("Failure during hash_fold_tree: {err}");
        }
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);

        sync_persistent_stream();
        let err = unsafe {
            sppark_poseidon254_rows(
                output.as_device_ptr(),
                matrix.as_device_ptr(),
                row_size,
                col_size.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during hash_rows 254: {err}");
        }
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

pub struct CudaHal<Hash: CudaHash + ?Sized> {
    pub max_threads: u32,
    hash: Option<Box<Hash>>,
    // Use primary context (None) to avoid per-instance CUDA module loading overhead.
    // The primary context persists for the process lifetime, so modules load once.
    _context: Option<Context>,
}

pub type CudaHalSha256 = CudaHal<CudaHashSha256>;
pub type CudaHalPoseidon2 = CudaHal<CudaHashPoseidon2>;
pub type CudaHalPoseidon254 = CudaHal<CudaHashPoseidon254>;

// ====================== VMM (cuMemCreate / cuMemMap) allocator ======================
// Composes a large contiguous virtual allocation from smaller physical chunks,
// defeating device-memory fragmentation at po2=22 on 24 GB cards.
//
// Bindings come from cust_raw (re-exported via risc0_sys::cuda::*), but to
// keep this file self-contained we link the driver functions directly.
#[allow(dead_code)]
mod vmm {
    use std::os::raw::{c_int, c_uchar, c_ulonglong, c_ushort, c_void};

    pub type CUdeviceptr = u64;
    pub type CUmemGenericAllocationHandle = u64;

    #[repr(i32)]
    #[derive(Copy, Clone)]
    pub enum CUmemLocationType {
        Device = 1,
    }
    #[repr(i32)]
    #[derive(Copy, Clone)]
    pub enum CUmemAllocationType {
        Pinned = 1,
    }
    #[repr(i32)]
    #[derive(Copy, Clone)]
    pub enum CUmemAllocationHandleType {
        None = 0,
    }
    #[repr(i32)]
    #[derive(Copy, Clone)]
    pub enum CUmemAccessFlags {
        ReadWrite = 3,
    }
    #[repr(i32)]
    #[derive(Copy, Clone)]
    pub enum CUmemAllocationGranularity {
        Minimum = 0,
        Recommended = 1,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct CUmemLocation {
        pub type_: c_int,
        pub id: c_int,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Default)]
    pub struct CUmemAllocationPropFlags {
        pub compression_type: c_uchar,
        pub gpu_direct_rdma_capable: c_uchar,
        pub usage: c_ushort,
        pub reserved: [c_uchar; 4],
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct CUmemAllocationProp {
        pub type_: c_int,
        pub requested_handle_types: c_int,
        pub location: CUmemLocation,
        pub win32_handle_meta_data: *mut c_void,
        pub alloc_flags: CUmemAllocationPropFlags,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct CUmemAccessDesc {
        pub location: CUmemLocation,
        pub flags: c_int,
    }

    extern "C" {
        pub fn cuMemAddressReserve(
            ptr: *mut CUdeviceptr,
            size: usize,
            alignment: usize,
            addr: CUdeviceptr,
            flags: c_ulonglong,
        ) -> c_int;
        pub fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> c_int;
        pub fn cuMemCreate(
            handle: *mut CUmemGenericAllocationHandle,
            size: usize,
            prop: *const CUmemAllocationProp,
            flags: c_ulonglong,
        ) -> c_int;
        pub fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> c_int;
        pub fn cuMemMap(
            ptr: CUdeviceptr,
            size: usize,
            offset: usize,
            handle: CUmemGenericAllocationHandle,
            flags: c_ulonglong,
        ) -> c_int;
        pub fn cuMemUnmap(ptr: CUdeviceptr, size: usize) -> c_int;
        pub fn cuMemSetAccess(
            ptr: CUdeviceptr,
            size: usize,
            desc: *const CUmemAccessDesc,
            count: usize,
        ) -> c_int;
        pub fn cuMemGetAllocationGranularity(
            granularity: *mut usize,
            prop: *const CUmemAllocationProp,
            option: c_int,
        ) -> c_int;
    }

    fn make_prop(device: c_int) -> CUmemAllocationProp {
        CUmemAllocationProp {
            type_: CUmemAllocationType::Pinned as c_int,
            requested_handle_types: CUmemAllocationHandleType::None as c_int,
            location: CUmemLocation {
                type_: CUmemLocationType::Device as c_int,
                id: device,
            },
            win32_handle_meta_data: std::ptr::null_mut(),
            alloc_flags: CUmemAllocationPropFlags::default(),
        }
    }

    /// Round `v` up to the nearest multiple of `a` (power of two or not).
    fn round_up(v: usize, a: usize) -> usize {
        ((v + a - 1) / a) * a
    }

    /// VMM-backed allocation. Owns both the virtual reservation and the
    /// physical chunks it's composed from. `ptr` is a normal CUdeviceptr
    /// that works with every CUDA API (memcpy, kernel launch, etc.).
    pub struct VmmAllocation {
        pub ptr: CUdeviceptr,
        pub size: usize,
        reserve_size: usize,
        chunks: Vec<(CUmemGenericAllocationHandle, usize)>, // (handle, mapped_bytes)
    }

    impl VmmAllocation {
        /// Allocate `size` bytes on `device`, carved into `~chunk_size` chunks.
        /// Returns None on any driver failure (caller falls back to cust).
        pub fn new(size: usize, device: c_int, chunk_size_hint: usize) -> Option<Self> {
            if size == 0 {
                return None;
            }
            let prop = make_prop(device);

            // Query granularity once; chunk size and reserve size must both be multiples.
            let mut gran: usize = 0;
            let rc = unsafe {
                cuMemGetAllocationGranularity(
                    &mut gran,
                    &prop,
                    CUmemAllocationGranularity::Recommended as c_int,
                )
            };
            if rc != 0 || gran == 0 {
                tracing::warn!("vmm: cuMemGetAllocationGranularity failed rc={rc}");
                return None;
            }

            let chunk_size = round_up(chunk_size_hint.max(gran), gran);
            let reserve_size = round_up(size, gran);

            // Reserve the virtual address range.
            let mut base: CUdeviceptr = 0;
            let rc = unsafe { cuMemAddressReserve(&mut base, reserve_size, gran, 0, 0) };
            if rc != 0 {
                tracing::warn!("vmm: cuMemAddressReserve({reserve_size}) failed rc={rc}");
                return None;
            }

            // Allocate + map physical chunks back-to-back.
            let mut alloc = VmmAllocation {
                ptr: base,
                size,
                reserve_size,
                chunks: Vec::with_capacity(reserve_size / chunk_size + 1),
            };

            let access = CUmemAccessDesc {
                location: prop.location,
                flags: CUmemAccessFlags::ReadWrite as c_int,
            };

            let mut offset = 0usize;
            while offset < reserve_size {
                let this_chunk = chunk_size.min(reserve_size - offset);
                let mut handle: CUmemGenericAllocationHandle = 0;
                let rc = unsafe { cuMemCreate(&mut handle, this_chunk, &prop, 0) };
                if rc != 0 {
                    tracing::warn!(
                        "vmm: cuMemCreate(chunk {} / {} bytes at offset {offset}) failed rc={rc}",
                        this_chunk,
                        reserve_size
                    );
                    drop(alloc);
                    return None;
                }
                let chunk_ptr = base + offset as u64;
                let rc = unsafe { cuMemMap(chunk_ptr, this_chunk, 0, handle, 0) };
                if rc != 0 {
                    tracing::warn!("vmm: cuMemMap failed rc={rc}");
                    unsafe { cuMemRelease(handle) };
                    drop(alloc);
                    return None;
                }
                let rc = unsafe { cuMemSetAccess(chunk_ptr, this_chunk, &access, 1) };
                if rc != 0 {
                    tracing::warn!("vmm: cuMemSetAccess failed rc={rc}");
                    unsafe { cuMemUnmap(chunk_ptr, this_chunk) };
                    unsafe { cuMemRelease(handle) };
                    drop(alloc);
                    return None;
                }
                alloc.chunks.push((handle, this_chunk));
                offset += this_chunk;
            }

            tracing::info!(
                "vmm: reserved {} bytes ({} chunks of {})",
                reserve_size,
                alloc.chunks.len(),
                chunk_size
            );
            Some(alloc)
        }
    }

    impl Drop for VmmAllocation {
        fn drop(&mut self) {
            // Unmapping is immediate, not stream-ordered: kernels still queued that use this
            // memory -- on any stream, e.g. eval_check's when a panic unwinds mid-finalize --
            // would fault or read garbage. VMM buffers are the >= 8 GB ones, so the wait is
            // rare (unless RISC0_VMM_ALLOC forces VMM for every allocation: then every free
            // waits for the device). Best effort: a Drop must not panic, e.g. at teardown.
            let _ = unsafe { super::cuCtxSynchronize() };
            let mut off = 0usize;
            for (h, sz) in self.chunks.drain(..) {
                unsafe {
                    cuMemUnmap(self.ptr + off as u64, sz);
                    cuMemRelease(h);
                }
                off += sz;
            }
            if self.reserve_size > 0 {
                unsafe {
                    cuMemAddressFree(self.ptr, self.reserve_size);
                }
            }
        }
    }

    /// 8 GB: VMM path kicks in automatically at or above this size.
    pub const VMM_AUTO_THRESHOLD: usize = 8usize << 30;
    /// 256 MB physical chunks.
    pub const VMM_CHUNK_SIZE: usize = 256usize << 20;

    pub fn should_use_vmm(size: usize) -> bool {
        if std::env::var("RISC0_VMM_ALLOC").ok().as_deref() == Some("1") {
            return true;
        }
        if std::env::var("RISC0_VMM_ALLOC").ok().as_deref() == Some("0") {
            return false;
        }
        size >= VMM_AUTO_THRESHOLD
    }
}
// ====================================================================================

struct RawBuffer {
    name: &'static str,
    buf: ManuallyDrop<DeviceBuffer<u8>>,
    // Some() when the buffer is backed by the VMM allocator. On drop,
    // we release the VMM mapping directly and SKIP the BufferPool cache
    // (pooling VMM buffers by exact size yields little benefit at po2=22).
    vmm: Option<vmm::VmmAllocation>,
    /// When true, bypass BUFFER_POOL caching on drop and return the memory
    /// directly to the CUDA driver via cuMemFree. Used for host-spilled
    /// buffers whose size doesn't match any future allocation, so caching
    /// them just wastes VRAM headroom.
    bypass_pool: bool,
}

impl RawBuffer {
    pub fn new(name: &'static str, size: usize) -> Self {
        tracing::trace!("alloc: {size} bytes, {name}");
        tracker().lock().unwrap().alloc(size);

        // VMM path: compose a large contiguous virtual allocation from ~256 MB
        // physical chunks. Used for big buffers (>= 8 GB) or when forced via
        // RISC0_VMM_ALLOC=1. Avoids fragmentation failures at po2=22.
        if vmm::should_use_vmm(size) {
            // Before attempting, drain our pool + the driver mempool so that
            // every free byte is up for grabs as physical chunks.
            BUFFER_POOL.with(|pool| pool.borrow_mut().drain());
            trim_cuda_mempool();
            if let Some(allocation) = vmm::VmmAllocation::new(size, 0, vmm::VMM_CHUNK_SIZE) {
                // Wrap the VMM virtual pointer in a DeviceBuffer so the rest
                // of this file keeps working unchanged. We never drop the
                // DeviceBuffer itself (that would call cuMemFree on a VMM
                // pointer, which is illegal); the VmmAllocation owns cleanup.
                let dptr = DevicePointer::<u8>::from_raw(allocation.ptr as cust::sys::CUdeviceptr);
                let buf = unsafe { DeviceBuffer::<u8>::from_raw_parts(dptr, size) };
                return Self {
                    name,
                    buf: ManuallyDrop::new(buf),
                    vmm: Some(allocation),
                    bypass_pool: false,
                };
            }
            tracing::warn!(
                "vmm: alloc of {size} bytes failed for {name}; falling back to cuMemAlloc"
            );
        }

        let buf = BUFFER_POOL
            .with(|pool| pool.borrow_mut().pop(size))
            .unwrap_or_else(|| {
                // First attempt: direct alloc.
                if let Ok(b) = unsafe { DeviceBuffer::uninitialized(size) } {
                    return b;
                }
                // Retry after draining our pool AND the CUDA stream-ordered mempool.
                // Large allocations (>10 GB) often fail due to mempool reservations
                // holding freed-but-retained memory; trimming reclaims it.
                tracing::warn!(
                    "allocation of {size} bytes for {name} failed; draining BUFFER_POOL + trimming CUDA mempool and retrying"
                );
                BUFFER_POOL.with(|pool| pool.borrow_mut().drain());
                trim_cuda_mempool();
                unsafe { DeviceBuffer::uninitialized(size) }
                    .context(format!("allocation failed on {name}: {size} bytes"))
                    .unwrap()
            });
        Self {
            name,
            buf: ManuallyDrop::new(buf),
            vmm: None,
            bypass_pool: false,
        }
    }

    pub fn set_u32(&mut self, value: u32) {
        self.buf.set_32(value).unwrap();
        wait_pageable_upload();
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        let size = self.buf.len();
        tracing::trace!("free: {size} bytes, {}", self.name);
        tracker().lock().unwrap().free(size);

        if self.vmm.is_some() {
            // VMM-backed: DO NOT run DeviceBuffer::drop (it would call
            // cuMemFree on a cuMemMap'd virtual address, which is illegal).
            // Forget the buffer wrapper; the VmmAllocation's Drop releases
            // the actual mappings + handles + virtual reservation.
            let _forget = unsafe { ManuallyDrop::take(&mut self.buf) };
            std::mem::forget(_forget);
            // VmmAllocation is dropped automatically when `self` is destroyed.
            return;
        }

        // Cache the buffer for reuse instead of calling cuMemFree.
        // Safety: self.buf is not accessed after take() since we're in Drop,
        // and ManuallyDrop's own drop is a no-op.
        let buf = unsafe { ManuallyDrop::take(&mut self.buf) };
        if self.bypass_pool {
            // Return memory directly to the driver (DeviceBuffer drop ->
            // cuMemFree). Used for host-spilled buffers where caching the
            // orphaned-size allocation wastes VRAM headroom.
            drop(buf);
        } else {
            // During thread shutdown the pool may already be gone; then `buf`, owned by the
            // closure that never ran, is freed directly instead (as on HIP).
            let _ = BUFFER_POOL.try_with(|pool| pool.borrow_mut().push(size, buf));
        }
    }
}

#[derive(Clone)]
pub struct BufferImpl<T> {
    buffer: Rc<RefCell<RawBuffer>>,
    size: usize,
    offset: usize,
    marker: PhantomData<T>,
}

#[inline]
fn unchecked_cast<A, B>(a: &[A]) -> &[B] {
    let new_len = std::mem::size_of_val(a) / std::mem::size_of::<B>();
    unsafe { std::slice::from_raw_parts(a.as_ptr() as *const B, new_len) }
}

#[inline]
fn unchecked_cast_mut<A, B>(a: &mut [A]) -> &mut [B] {
    let new_len = std::mem::size_of_val(a) / std::mem::size_of::<B>();
    unsafe { std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut B, new_len) }
}

impl<T> BufferImpl<T> {
    fn new(name: &'static str, size: usize) -> Self {
        let bytes_len = std::mem::size_of::<T>() * size;
        assert!(bytes_len > 0);
        BufferImpl {
            buffer: Rc::new(RefCell::new(RawBuffer::new(name, bytes_len))),
            size,
            offset: 0,
            marker: PhantomData,
        }
    }

    pub fn copy_from(name: &'static str, slice: &[T]) -> Self {
        // scope!("copy_from");
        let bytes_len = std::mem::size_of_val(slice);
        assert!(bytes_len > 0);
        let mut buffer = RawBuffer::new(name, bytes_len);
        let bytes = unchecked_cast(slice);
        buffer.buf.copy_from(bytes).unwrap();
        wait_pageable_upload();

        BufferImpl {
            buffer: Rc::new(RefCell::new(buffer)),
            size: slice.len(),
            offset: 0,
            marker: PhantomData,
        }
    }

    pub fn as_device_ptr(&self) -> DevicePointer<u8> {
        let ptr = self.buffer.borrow_mut().buf.as_device_ptr();
        let offset = self.offset * std::mem::size_of::<T>();
        unsafe { ptr.offset(offset.try_into().unwrap()) }
    }

    /// Mark this buffer so that when all Rc clones are dropped, the underlying
    /// device memory is returned to the driver immediately instead of being
    /// cached in BUFFER_POOL. Use for host-spilled allocations whose size is
    /// unlikely to be requested again soon.
    pub fn set_bypass_pool(&self) {
        self.buffer.borrow_mut().bypass_pool = true;
    }

    pub fn as_device_ptr_with_offset(&self, offset: usize) -> DevicePointer<u8> {
        let ptr = self.buffer.borrow_mut().buf.as_device_ptr();
        let offset = (self.offset + offset) * std::mem::size_of::<T>();
        unsafe { ptr.offset(offset.try_into().unwrap()) }
    }
}

impl<T: Clone> Buffer<T> for BufferImpl<T> {
    fn name(&self) -> &'static str {
        self.buffer.borrow().name
    }

    fn size(&self) -> usize {
        self.size
    }

    fn slice(&self, offset: usize, size: usize) -> BufferImpl<T> {
        assert!(offset + size <= self.size());
        BufferImpl {
            buffer: self.buffer.clone(),
            size,
            offset: self.offset + offset,
            marker: PhantomData,
        }
    }

    fn get_at(&self, idx: usize) -> T {
        let item_size = std::mem::size_of::<T>();
        let buf = self.buffer.borrow_mut();
        let offset = (self.offset + idx) * item_size;
        let ptr = unsafe { buf.buf.as_device_ptr().offset(offset as isize) };
        let device_slice = unsafe { DeviceSlice::from_raw_parts(ptr, item_size) };
        let host_buf = device_slice.as_host_vec().unwrap();
        let slice: &[T] = unchecked_cast(&host_buf);
        slice[0].clone()
    }

    fn view<F: FnOnce(&[T])>(&self, f: F) {
        scope!("view");
        let item_size = std::mem::size_of::<T>();
        let buf = self.buffer.borrow_mut();
        let offset = self.offset * item_size;
        let len = self.size * item_size;
        let ptr = unsafe { buf.buf.as_device_ptr().offset(offset as isize) };
        let device_slice = unsafe { DeviceSlice::from_raw_parts(ptr, len) };
        let host_buf = device_slice.as_host_vec().unwrap();
        let slice = unchecked_cast(&host_buf);
        f(slice);
    }

    fn view_mut<F: FnOnce(&mut [T])>(&self, f: F) {
        scope!("view_mut");
        let mut buf = self.buffer.borrow_mut();
        let mut host_buf = buf.buf.as_host_vec().unwrap();
        let slice = unchecked_cast_mut(&mut host_buf);
        f(&mut slice[self.offset..]);
        buf.buf.copy_from(&host_buf).unwrap();
        wait_pageable_upload();
    }

    fn to_vec(&self) -> Vec<T> {
        let buf = self.buffer.borrow_mut();
        let host_buf = buf.buf.as_host_vec().unwrap();
        let slice = unchecked_cast(&host_buf);
        slice.to_vec()
    }

    fn set_bypass_pool(&self) {
        self.buffer.borrow_mut().bypass_pool = true;
    }
}

impl<CH: CudaHash> Default for CudaHal<CH> {
    fn default() -> Self {
        Self::new()
    }
}

impl<CH: CudaHash + ?Sized> CudaHal<CH> {
    pub fn new() -> Self
    where
        CH: Sized,
    {
        Self::new_from_hash(Box::new(CH::new()))
    }

    fn new_from_hash(hash: Box<CH>) -> Self {
        let _lock = singleton().lock();

        let err = unsafe { sppark_init() };
        if err.code != 0 {
            panic!("Failure during sppark_init: {err}");
        }

        cust::init(CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap();
        // Use the primary context (from sppark_init/cust::init) instead of creating
        // a new context. This avoids per-instance CUDA module loading (~100ms for
        // large rv32im kernels) since modules persist in the primary context.

        // Warmup: create the persistent CUDA stream and load the risc0-zkp kernel
        // module. This retains the primary CUDA context so that subsequent
        // DeviceBuffer allocations don't fail with "invalid device context".
        extern "C" {
            fn risc0_zkp_cuda_warmup() -> *const std::os::raw::c_char;
        }
        let _ = ffi_wrap(|| unsafe { risc0_zkp_cuda_warmup() });
        register_oom_hook();

        let mut hal = Self {
            max_threads: max_threads as u32,
            _context: None,
            hash: None,
        };
        hal.hash = Some(hash);
        hal
    }

    /// Synchronize the risc0 persistent CUDA stream.
    /// Must be called before sppark operations that read from buffers
    /// last written by risc0 kernels (which use a different CUDA stream).
    fn sync_stream() {
        sync_persistent_stream();
    }

    #[allow(dead_code)]
    fn poly_divide(
        &self,
        polynomial: &BufferImpl<BabyBearExtElem>,
        pow: BabyBearExtElem,
    ) -> BabyBearExtElem {
        let mut remainder = BabyBearExtElem::ZERO;
        let poly_size = polynomial.size();
        let pow = pow.to_u32_words();

        Self::sync_stream();
        let err = unsafe {
            supra_poly_divide(
                polynomial.as_device_ptr(),
                poly_size,
                &mut remainder as *mut _ as *mut u32,
                pow.as_ptr(),
            )
        };

        if err.code != 0 {
            panic!("Failure during supra_poly_divide: {err}");
        }

        remainder
    }

    #[allow(dead_code)]
    fn poly_divide_batch(
        &self,
        polynomial: &BufferImpl<BabyBearExtElem>,
        pows: &[BabyBearExtElem],
    ) -> Vec<BabyBearExtElem> {
        let num_divides = pows.len();
        let poly_size = polynomial.size();
        let mut remainders = vec![BabyBearExtElem::ZERO; num_divides];
        let pows_words: Vec<u32> = pows.iter().flat_map(|p| p.to_u32_words()).collect();
        Self::sync_stream();

        let err = unsafe {
            supra_poly_divide_batch(
                polynomial.as_device_ptr(),
                poly_size,
                remainders.as_mut_ptr() as *mut u32,
                pows_words.as_ptr(),
                num_divides as u32,
            )
        };

        if err.code != 0 {
            panic!("Failure during supra_poly_divide_batch: {err}");
        }

        remainders
    }
}

impl CudaHal<dyn CudaHash> {
    pub fn new_from_hash_suite(hash_suite: HashSuite<BabyBear>) -> Result<Self> {
        let hash_suite_box = match &hash_suite.name[..] {
            "poseidon2" => Box::new(CudaHashPoseidon2::new()) as Box<dyn CudaHash>,
            "poseidon254" => Box::new(CudaHashPoseidon254::new()) as Box<dyn CudaHash>,
            "sha-256" => Box::new(CudaHashSha256::new()) as Box<dyn CudaHash>,
            other => bail!("unsupported hash_fn {other}"),
        };
        Ok(Self::new_from_hash(hash_suite_box))
    }
}

impl<CH: CudaHash + ?Sized> Hal for CudaHal<CH> {
    type Field = BabyBear;
    type Elem = BabyBearElem;
    type ExtElem = BabyBearExtElem;
    type Buffer<T: Clone + Debug + PartialEq> = BufferImpl<T>;

    fn alloc_elem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::Elem> {
        BufferImpl::new(name, size)
    }

    fn alloc_elem_init(
        &self,
        name: &'static str,
        size: usize,
        value: Self::Elem,
    ) -> Self::Buffer<Self::Elem> {
        let buffer = self.alloc_elem(name, size);
        buffer
            .buffer
            .borrow_mut()
            .set_u32(value.as_u32_montgomery());
        buffer
    }

    fn copy_from_elem(&self, name: &'static str, slice: &[Self::Elem]) -> Self::Buffer<Self::Elem> {
        BufferImpl::copy_from(name, slice)
    }

    fn alloc_extelem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        BufferImpl::new(name, size)
    }

    fn alloc_extelem_zeroed(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        let buffer = self.alloc_extelem(name, size);
        buffer.buffer.borrow_mut().set_u32(0);
        buffer
    }

    fn copy_from_extelem(
        &self,
        name: &'static str,
        slice: &[Self::ExtElem],
    ) -> Self::Buffer<Self::ExtElem> {
        BufferImpl::copy_from(name, slice)
    }

    fn alloc_digest(&self, name: &'static str, size: usize) -> Self::Buffer<Digest> {
        BufferImpl::new(name, size)
    }

    fn copy_from_digest(&self, name: &'static str, slice: &[Digest]) -> Self::Buffer<Digest> {
        BufferImpl::copy_from(name, slice)
    }

    fn alloc_u32(&self, name: &'static str, size: usize) -> Self::Buffer<u32> {
        BufferImpl::new(name, size)
    }

    fn copy_from_u32(&self, name: &'static str, slice: &[u32]) -> Self::Buffer<u32> {
        BufferImpl::copy_from(name, slice)
    }

    fn batch_expand_into_evaluate_ntt(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        poly_count: usize,
        _expand_bits: usize,
    ) {
        let out_size = output.size() / poly_count;
        let in_size = input.size() / poly_count;
        let expand_bits = log2_ceil(out_size / in_size);
        assert_eq!(output.size(), out_size * poly_count);
        assert_eq!(input.size(), in_size * poly_count);
        assert_eq!(out_size, in_size * (1 << expand_bits));
        let in_bits = log2_ceil(in_size);

        let row_size = output.size() / poly_count;
        assert_eq!(row_size * poly_count, output.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        assert!(n_bits >= expand_bits);
        assert!(n_bits < Self::Elem::MAX_ROU_PO2);
        Self::sync_stream();

        let err = unsafe {
            sppark_batch_expand_NTT(
                output.as_device_ptr(),
                input.as_device_ptr(),
                in_bits.try_into().unwrap(),
                expand_bits.try_into().unwrap(),
                poly_count.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during batch_expand_NTT: {err}");
        }
    }

    fn batch_interpolate_ntt(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        assert!(n_bits < Self::Elem::MAX_ROU_PO2);
        Self::sync_stream();

        let err = unsafe {
            sppark_batch_iNTT(
                io.as_device_ptr(),
                n_bits.try_into().unwrap(),
                count.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during batch_interpolate_ntt: {err}");
        }
    }

    fn has_commit_coeffs_lde(&self, po2: usize) -> bool {
        // The range sppark_commit_coeffs_lde is verified bit-exact for.
        (10..=23).contains(&po2)
    }

    fn commit_coeffs_lde(
        &self,
        coeffs: &Self::Buffer<Self::Elem>,
        evaluated: &Self::Buffer<Self::Elem>,
        witness: &Self::Buffer<Self::Elem>,
        count: usize,
    ) {
        let row_size = witness.size() / count;
        assert_eq!(row_size * count, witness.size());
        assert_eq!(coeffs.size(), witness.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        let lg_blowup = log2_ceil(crate::INV_RATE);
        assert_eq!(evaluated.size(), witness.size() << lg_blowup);
        assert!(n_bits + lg_blowup < Self::Elem::MAX_ROU_PO2);
        Self::sync_stream();
        let err = unsafe {
            sppark_commit_coeffs_lde(
                coeffs.as_device_ptr(),
                evaluated.as_device_ptr(),
                witness.as_device_ptr(),
                n_bits as u32,
                lg_blowup as u32,
                count as u32,
            )
        };
        if err.code != 0 {
            panic!("Failure during commit_coeffs_lde: {err}");
        }
    }

    fn batch_bit_reverse(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << bits);
        Self::sync_stream();
        let err = unsafe {
            sppark_batch_bit_reverse_tiled(io.as_device_ptr(), bits as u32, count as u32)
        };
        if err.code != 0 {
            panic!("Failure during batch_bit_reverse_tiled: {err}");
        }
    }

    fn batch_evaluate_any(
        &self,
        coeffs: &Self::Buffer<Self::Elem>,
        poly_count: usize,
        which: &Self::Buffer<u32>,
        xs: &Self::Buffer<Self::ExtElem>,
        out: &Self::Buffer<Self::ExtElem>,
    ) {
        let po2 = log2_ceil(coeffs.size() / poly_count);
        let count = 1 << po2;
        assert_eq!(poly_count * count, coeffs.size());
        let eval_count = which.size();
        assert_eq!(xs.size(), eval_count);
        assert_eq!(out.size(), eval_count);

        let threads_per_block = self.max_threads / 4;
        const BYTES_PER_WORD: u32 = 4;
        const WORDS_PER_FPEXT: u32 = 4;
        let shared_size = threads_per_block * BYTES_PER_WORD * WORDS_PER_FPEXT;
        let kernel_count = out.size() * threads_per_block as usize;

        extern "C" {
            fn risc0_zkp_cuda_batch_evaluate_any(
                output: DevicePointer<u8>,
                coeffs: DevicePointer<u8>,
                which: DevicePointer<u8>,
                xs: DevicePointer<u8>,
                shared_size: u32,
                kernel_count: u32,
                count: u32,
                eval_count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_batch_evaluate_any(
                out.as_device_ptr(),
                coeffs.as_device_ptr(),
                which.as_device_ptr(),
                xs.as_device_ptr(),
                shared_size,
                kernel_count as u32,
                count as u32,
                eval_count as u32,
            )
        })
        .unwrap();
    }

    fn gather_sample(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        idx: usize,
        size: usize,
        stride: usize,
    ) {
        extern "C" {
            fn risc0_zkp_cuda_gather_sample(
                dst: DevicePointer<u8>,
                src: DevicePointer<u8>,
                idx: u32,
                size: u32,
                stride: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_gather_sample(
                dst.as_device_ptr(),
                src.as_device_ptr(),
                idx as u32,
                size as u32,
                stride as u32,
            )
        })
        .unwrap();
    }

    fn gather_samples(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        indices: &[usize],
        size: usize,
        stride: usize,
    ) {
        if indices.len() < 2 {
            for (q, &idx) in indices.iter().enumerate() {
                self.gather_sample(&dst.slice(q * size, size), src, idx, size, stride);
            }
            return;
        }
        let idxs: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        // copy_from is a synchronous copy on the legacy stream, which waits for the persistent
        // stream: so idx_buf can go back to the pool right after the async launch below (the
        // next copy into it waits for this kernel), as with the other temporaries here.
        let idx_buf = self.copy_from_u32("gather_idx", &idxs);
        extern "C" {
            fn risc0_zkp_cuda_gather_samples(
                dst: DevicePointer<u8>,
                src: DevicePointer<u8>,
                idxs: DevicePointer<u8>,
                n: u32,
                size: u32,
                stride: u32,
            ) -> *const std::os::raw::c_char;
        }
        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_gather_samples(
                dst.as_device_ptr(),
                src.as_device_ptr(),
                idx_buf.as_device_ptr(),
                idxs.len() as u32,
                size as u32,
                stride as u32,
            )
        })
        .unwrap();
    }

    fn has_unified_memory(&self) -> bool {
        false
    }

    fn trim_device_memory(&self) {
        // Trim only the CUDA stream-ordered mempool (where sppark's multi-GB
        // NTT scratch lives after cudaFreeAsync). Do NOT drain our own
        // BUFFER_POOL — small-buffer reuse there is still valuable.
        trim_cuda_mempool();
    }

    fn batch_get_digest_at(&self, buf: &Self::Buffer<Digest>, indices: &[usize]) -> Vec<Digest> {
        if indices.is_empty() {
            return Vec::new();
        }
        let count = indices.len();
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let indices_buf = self.copy_from_u32("gather_indices", &indices_u32);
        let output_buf: BufferImpl<Digest> = BufferImpl::new("gather_output", count);

        extern "C" {
            fn risc0_zkp_cuda_gather_digests(
                dst: DevicePointer<u8>,
                src: DevicePointer<u8>,
                indices: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_gather_digests(
                output_buf.as_device_ptr(),
                buf.as_device_ptr(),
                indices_buf.as_device_ptr(),
                count as u32,
            )
        })
        .unwrap();

        let mut result = Vec::new();
        output_buf.view(|view| {
            result = view.to_vec();
        });
        result
    }

    fn zk_shift(&self, io: &Self::Buffer<Self::Elem>, poly_count: usize) {
        let bits = log2_ceil(io.size() / poly_count);
        assert_eq!(io.size(), poly_count * (1 << bits));

        Self::sync_stream();
        let err = unsafe {
            sppark_batch_zk_shift(
                io.as_device_ptr(),
                bits.try_into().unwrap(),
                poly_count.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during zk_shift: {err}");
        }
    }

    fn batch_interpolate_ntt_zk_shift(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        assert!(n_bits < Self::Elem::MAX_ROU_PO2);
        Self::sync_stream();

        let err = unsafe {
            sppark_batch_iNTT_zk_shift_fused(
                io.as_device_ptr(),
                n_bits.try_into().unwrap(),
                count.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during batch_iNTT_zk_shift: {err}");
        }
    }

    fn mix_poly_coeffs(
        &self,
        output: &Self::Buffer<Self::ExtElem>,
        mix_start: &Self::ExtElem,
        mix: &Self::ExtElem,
        input: &Self::Buffer<Self::Elem>,
        combos: &Self::Buffer<u32>,
        input_size: usize,
        count: usize,
    ) {
        let mix_start = self.copy_from_extelem("mix_start", &[*mix_start]);
        let mix = self.copy_from_extelem("mix", &[*mix]);

        extern "C" {
            fn risc0_zkp_cuda_mix_poly_coeffs(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                combos: DevicePointer<u8>,
                mix_start: DevicePointer<u8>,
                mix: DevicePointer<u8>,
                input_size: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_mix_poly_coeffs(
                output.as_device_ptr(),
                input.as_device_ptr(),
                combos.as_device_ptr(),
                mix_start.as_device_ptr(),
                mix.as_device_ptr(),
                input_size as u32,
                count as u32,
            )
        })
        .unwrap();
    }

    fn eltwise_add_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input1: &Self::Buffer<Self::Elem>,
        input2: &Self::Buffer<Self::Elem>,
    ) {
        assert_eq!(output.size(), input1.size());
        assert_eq!(output.size(), input2.size());
        let count = output.size();

        extern "C" {
            fn risc0_zkp_cuda_eltwise_add_fp(
                out: DevicePointer<u8>,
                x: DevicePointer<u8>,
                y: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_eltwise_add_fp(
                output.as_device_ptr(),
                input1.as_device_ptr(),
                input2.as_device_ptr(),
                count as u32,
            )
        })
        .unwrap();
    }

    fn eltwise_sum_extelem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::ExtElem>,
    ) {
        let count = output.size() / Self::ExtElem::EXT_SIZE;
        let to_add = input.size() / count;
        assert_eq!(output.size(), count * Self::ExtElem::EXT_SIZE);
        assert_eq!(input.size(), count * to_add);

        extern "C" {
            fn risc0_zkp_cuda_eltwise_sum_fpext(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                to_add: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_eltwise_sum_fpext(
                output.as_device_ptr(),
                input.as_device_ptr(),
                to_add as u32,
                count as u32,
            )
        })
        .unwrap();
    }

    fn eltwise_copy_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
    ) {
        let count = output.size();
        assert_eq!(count, input.size());

        extern "C" {
            fn risc0_zkp_cuda_eltwise_copy_fp(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_eltwise_copy_fp(
                output.as_device_ptr(),
                input.as_device_ptr(),
                count as u32,
            )
        })
        .unwrap();
    }

    fn eltwise_zeroize_elem(&self, elems: &Self::Buffer<Self::Elem>) {
        extern "C" {
            fn risc0_zkp_cuda_eltwise_zeroize_fp(
                elems: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_eltwise_zeroize_fp(elems.as_device_ptr(), elems.size() as u32)
        })
        .unwrap();
    }

    fn scatter(
        &self,
        into: &Self::Buffer<Self::Elem>,
        index: &[u32],
        offsets: &[u32],
        values: &[Self::Elem],
    ) {
        if index.is_empty() {
            return;
        }

        let count = index.len() - 1;
        if count == 0 {
            return;
        }

        extern "C" {
            fn risc0_zkp_cuda_scatter_from_host(
                into: DevicePointer<u8>,
                h_index: *const u32,
                index_count: u32,
                h_offsets: *const u32,
                offsets_count: u32,
                h_values: *const u8,
                values_count: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_scatter_from_host(
                into.as_device_ptr(),
                index.as_ptr(),
                index.len() as u32,
                offsets.as_ptr(),
                offsets.len() as u32,
                values.as_ptr() as *const u8,
                values.len() as u32,
                count as u32,
            )
        })
        .unwrap();
    }

    fn scatter_bits(&self, into: &Self::Buffer<Self::Elem>, bit_data: &[u32], cycles: u32) {
        let count = bit_data.len() / 3;
        if count == 0 {
            return;
        }

        extern "C" {
            fn risc0_zkp_cuda_scatter_bits_from_host(
                into: DevicePointer<u8>,
                h_data: *const u32,
                triplet_count: u32,
                cycles: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_scatter_bits_from_host(
                into.as_device_ptr(),
                bit_data.as_ptr(),
                count as u32,
                cycles,
            )
        })
        .unwrap();
    }

    fn eltwise_copy_elem_slice(
        &self,
        into: &Self::Buffer<Self::Elem>,
        from: &[Self::Elem],
        from_rows: usize,
        from_cols: usize,
        from_offset: usize,
        from_stride: usize,
        into_offset: usize,
        into_stride: usize,
    ) {
        let from = self.copy_from_elem("from", from);

        extern "C" {
            fn risc0_zkp_cuda_eltwise_copy_fp_region(
                into: DevicePointer<u8>,
                from: DevicePointer<u8>,
                from_rows: u32,
                from_cols: u32,
                from_offset: u32,
                from_stride: u32,
                into_offset: u32,
                into_stride: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_eltwise_copy_fp_region(
                into.as_device_ptr(),
                from.as_device_ptr(),
                from_rows as u32,
                from_cols as u32,
                from_offset as u32,
                from_stride as u32,
                into_offset as u32,
                into_stride as u32,
            )
        })
        .unwrap();
    }

    fn fri_fold(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        mix: &Self::ExtElem,
    ) {
        let count = output.size() / Self::ExtElem::EXT_SIZE;
        assert_eq!(output.size(), count * Self::ExtElem::EXT_SIZE);
        assert_eq!(input.size(), output.size() * FRI_FOLD);
        let mix = self.copy_from_extelem("mix", &[*mix]);

        extern "C" {
            fn risc0_zkp_cuda_fri_fold(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                mix: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_fri_fold(
                output.as_device_ptr(),
                input.as_device_ptr(),
                mix.as_device_ptr(),
                count as u32,
            )
        })
        .unwrap();
    }

    fn hash_fold(&self, io: &Self::Buffer<Digest>, input_size: usize, output_size: usize) {
        assert_eq!(input_size, 2 * output_size);
        self.hash.as_ref().unwrap().hash_fold(io, output_size);
    }

    fn hash_fold_tree(&self, io: &Self::Buffer<Digest>, layers: usize) {
        self.hash.as_ref().unwrap().hash_fold_tree(io, layers);
    }

    fn hash_rows(&self, output: &Self::Buffer<Digest>, matrix: &Self::Buffer<Self::Elem>) {
        self.hash.as_ref().unwrap().hash_rows(output, matrix);
    }

    fn get_hash_suite(&self) -> &HashSuite<Self::Field> {
        self.hash.as_ref().unwrap().get_hash_suite()
    }

    fn prefix_products(&self, io: &Self::Buffer<Self::ExtElem>) {
        io.view_mut(|io| {
            for i in 1..io.len() {
                io[i] *= io[i - 1];
            }
        });
    }

    fn combos_prepare(
        &self,
        combos: &Self::Buffer<Self::ExtElem>,
        coeff_u: &[Self::ExtElem],
        combo_count: usize,
        cycles: usize,
        reg_sizes: &[u32],
        reg_combo_ids: &[u32],
        mix: &Self::ExtElem,
    ) {
        // Compute the per-element deltas on the host (coeff_u is host data) and apply them
        // with one small parallel kernel, instead of a single-thread device loop. The
        // temporaries are freed after the async launch; see gather_samples.
        let mut deltas: std::collections::BTreeMap<u32, Self::ExtElem> =
            std::collections::BTreeMap::new();
        let mut cur = Self::ExtElem::ONE;
        let mut pos = 0usize;
        for (size, combo_id) in reg_sizes.iter().zip(reg_combo_ids.iter()) {
            for j in 0..*size as usize {
                let off = (cycles * *combo_id as usize + j) as u32;
                let e = deltas.entry(off).or_insert(Self::ExtElem::ZERO);
                *e += cur * coeff_u[pos + j];
            }
            cur *= *mix;
            pos += *size as usize;
        }
        for _ in 0..Self::CHECK_SIZE {
            let off = (cycles * combo_count) as u32;
            let e = deltas.entry(off).or_insert(Self::ExtElem::ZERO);
            *e += cur * coeff_u[pos];
            pos += 1;
            cur *= *mix;
        }
        let offs: Vec<u32> = deltas.keys().copied().collect();
        let vals: Vec<Self::ExtElem> = deltas.values().copied().collect();
        let offs_buf = self.copy_from_u32("combos_offs", &offs);
        let vals_buf = self.copy_from_extelem("combos_deltas", &vals);
        extern "C" {
            fn risc0_zkp_cuda_combos_sub(
                combos: DevicePointer<u8>,
                deltas: DevicePointer<u8>,
                offsets: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }
        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_combos_sub(
                combos.as_device_ptr(),
                vals_buf.as_device_ptr(),
                offs_buf.as_device_ptr(),
                offs.len() as u32,
            )
        })
        .unwrap();
    }

    fn combos_divide(
        &self,
        combos: &Self::Buffer<Self::ExtElem>,
        chunks: Vec<(usize, Vec<Self::ExtElem>)>,
        cycles: usize,
    ) {
        scope!("combos_divide");
        let combo_indices: Vec<u32> = chunks.iter().map(|(i, _)| *i as u32).collect();
        let pows_per_combo: Vec<u32> = chunks.iter().map(|(_, pows)| pows.len() as u32).collect();
        let all_pows_flat: Vec<u32> = chunks
            .iter()
            .flat_map(|(_, pows)| pows.iter().flat_map(|p| p.to_u32_words()))
            .collect();

        let mut remainders =
            vec![BabyBearExtElem::ZERO; chunks.iter().map(|(_, pows)| pows.len()).sum()];

        Self::sync_stream();
        let err = unsafe {
            supra_poly_divide_multi(
                combos.as_device_ptr(),
                cycles,
                combo_indices.as_ptr(),
                pows_per_combo.as_ptr(),
                all_pows_flat.as_ptr(),
                chunks.len() as u32,
                remainders.as_mut_ptr() as *mut u32,
            )
        };
        if err.code != 0 {
            panic!("Failure during supra_poly_divide_multi: {err}");
        }
        super::report_nonzero_remainders(&chunks, &remainders);
    }
}

#[cfg(test)]
mod tests {
    use test_log::test;

    use super::{CudaHalPoseidon2, CudaHalSha256};
    use crate::hal::testutil;

    #[test]
    #[should_panic]
    fn check_req() {
        testutil::check_req(CudaHalSha256::new());
    }

    #[test]
    fn eltwise_add_elem() {
        testutil::eltwise_add_elem(CudaHalSha256::new());
    }

    #[test]
    fn eltwise_copy_elem() {
        testutil::eltwise_copy_elem(CudaHalSha256::new());
    }

    #[test]
    fn eltwise_sum_extelem() {
        testutil::eltwise_sum_extelem(CudaHalSha256::new());
    }

    #[test]
    fn hash_rows_sha256() {
        testutil::hash_rows(CudaHalSha256::new());
    }

    #[test]
    fn hash_fold_sha256() {
        testutil::hash_fold(CudaHalSha256::new());
    }

    #[test]
    fn hash_rows_poseidon2() {
        testutil::hash_rows(CudaHalPoseidon2::new());
    }

    #[test]
    fn hash_fold_poseidon2() {
        testutil::hash_fold(CudaHalPoseidon2::new());
    }

    #[test]
    fn hash_fold_tree_poseidon2() {
        testutil::hash_fold_tree(CudaHalPoseidon2::new());
    }

    #[test]
    fn fri_fold() {
        testutil::fri_fold(CudaHalSha256::new());
    }

    #[test]
    fn batch_expand_into_evaluate_ntt() {
        testutil::batch_expand_into_evaluate_ntt(CudaHalSha256::new());
    }

    #[test]
    fn batch_expand_into_evaluate_ntt_large() {
        testutil::batch_expand_into_evaluate_ntt_large(CudaHalSha256::new());
    }

    #[test]
    fn batch_interpolate_ntt() {
        testutil::batch_interpolate_ntt(CudaHalSha256::new());
    }

    #[test]
    fn batch_bit_reverse() {
        testutil::batch_bit_reverse(CudaHalSha256::new());
    }

    #[test]
    fn batch_evaluate_any() {
        testutil::batch_evaluate_any(CudaHalSha256::new());
    }

    #[test]
    fn gather_sample() {
        testutil::gather_sample(CudaHalSha256::new());
    }

    #[test]
    fn zk_shift() {
        testutil::zk_shift(CudaHalSha256::new());
    }

    #[test]
    fn mix_poly_coeffs() {
        testutil::mix_poly_coeffs(CudaHalSha256::new());
    }

    #[test]
    fn commit_coeffs_lde() {
        testutil::commit_coeffs_lde(CudaHalSha256::new());
    }
}
