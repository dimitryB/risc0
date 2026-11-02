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

//! Hardware Abstraction Layer (HAL) for Intel GPU acceleration via SYCL/ESIMD.

use std::{
    fmt::Debug,
    marker::PhantomData,
    mem::ManuallyDrop,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};

use parking_lot::{Mutex, ReentrantMutex};
use risc0_core::{
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        ExtElem, RootsOfUnity,
    },
    scope,
};
/// See [risc0_sys::intel::start_background_init].
pub use risc0_sys::intel::start_background_init;
use risc0_sys::intel::{self, esimd_check, get_queue, DevicePointer, IntelDeviceBuffer};

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

// ============================================================================
// Singleton lock (one IntelHal per process)
// ============================================================================

pub fn singleton() -> &'static ReentrantMutex<()> {
    static ONCE: OnceLock<ReentrantMutex<()>> = OnceLock::new();
    ONCE.get_or_init(|| ReentrantMutex::new(()))
}

// ============================================================================
// IntelHash trait and implementations
// ============================================================================

pub trait IntelHash {
    /// Create a hash implementation
    fn new() -> Self
    where
        Self: Sized;

    /// Run the hash_fold function
    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize);

    /// Run the hash_rows function
    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>);

    /// Return the HashSuite
    fn get_hash_suite(&self) -> &HashSuite<BabyBear>;
}

/// Narrows a size/count/index for an FFI kernel argument. Kernels take u32;
/// a silent wrap would make them process the wrong range instead of failing.
#[inline]
fn ffi_u32(x: usize) -> u32 {
    u32::try_from(x).unwrap_or_else(|_| panic!("Intel kernel argument {x} exceeds u32"))
}

// ---- Poseidon2 (ESIMD GPU) ----

pub struct IntelHashPoseidon2 {
    suite: HashSuite<BabyBear>,
}

impl IntelHash for IntelHashPoseidon2 {
    fn new() -> Self {
        IntelHashPoseidon2 {
            suite: Poseidon2HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        let input_ptr = io.as_device_ptr_with_offset(2 * output_size);
        let output_ptr = io.as_device_ptr_with_offset(output_size);
        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_poseidon2_hash_fold(
                queue,
                output_ptr.0 as *mut std::ffi::c_void,
                input_ptr.0 as *const std::ffi::c_void,
                ffi_u32(output_size),
            )
        });
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);
        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_poseidon2_hash_rows(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                matrix.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(row_size),
                ffi_u32(col_size),
            )
        });
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

// ---- SHA-256 (CPU fallback) ----

pub struct IntelHashSha256 {
    suite: HashSuite<BabyBear>,
}

impl IntelHash for IntelHashSha256 {
    fn new() -> Self {
        IntelHashSha256 {
            suite: Sha256HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        // CPU fallback with rayon parallelism
        io.view_mut(|buf| {
            let hashfn: &dyn crate::core::hash::HashFn<BabyBear> = self.suite.hashfn.as_ref();
            let (_, rest) = buf.split_at_mut(output_size);
            let (out_slice, in_slice) = rest.split_at_mut(output_size);
            use rayon::prelude::*;
            out_slice.par_iter_mut().enumerate().for_each(|(idx, out)| {
                *out = *hashfn.hash_pair(&in_slice[idx * 2], &in_slice[idx * 2 + 1]);
            });
        });
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / row_size;
        assert_eq!(matrix.size(), col_size * row_size);

        // CPU fallback with rayon parallelism
        matrix.view(|matrix_data| {
            output.view_mut(|out_data| {
                let hashfn: &dyn crate::core::hash::HashFn<BabyBear> = self.suite.hashfn.as_ref();
                use rayon::prelude::*;
                out_data.par_iter_mut().enumerate().for_each(|(row, out)| {
                    let mut row_elems = Vec::with_capacity(col_size);
                    for col in 0..col_size {
                        row_elems.push(matrix_data[col * row_size + row]);
                    }
                    *out = *hashfn.hash_elem_slice(&row_elems);
                });
            });
        });
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

// ---- Poseidon-254 (GPU ESIMD) ----

pub struct IntelHashPoseidon254 {
    suite: HashSuite<BabyBear>,
}

impl IntelHash for IntelHashPoseidon254 {
    fn new() -> Self {
        IntelHashPoseidon254 {
            suite: Poseidon254HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        let input_ptr = io.as_device_ptr_with_offset(2 * output_size);
        let output_ptr = io.as_device_ptr_with_offset(output_size);
        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_poseidon254_hash_fold(
                queue,
                output_ptr.0 as *mut std::ffi::c_void,
                input_ptr.0 as *const std::ffi::c_void,
                ffi_u32(output_size),
            )
        });
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);
        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_poseidon254_hash_rows(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                matrix.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(row_size),
                ffi_u32(col_size),
            )
        });
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

// ============================================================================
// IntelHal struct
// ============================================================================

pub struct IntelHal<Hash: IntelHash + ?Sized> {
    hash: Option<Box<Hash>>,
}

// SAFETY: IntelHal holds no thread-pinned state. Underlying buffers use
// Arc<Mutex<RawBuffer>>, the SYCL queue handle is a process-wide singleton
// safe to access from any thread, and the singleton ReentrantMutex is
// acquired per-method on entry to GPU-touching methods (not held across
// thread boundaries).
unsafe impl<Hash: IntelHash + ?Sized> Send for IntelHal<Hash> where Hash: Send {}
unsafe impl<Hash: IntelHash + ?Sized> Sync for IntelHal<Hash> where Hash: Sync {}

pub type IntelHalPoseidon2 = IntelHal<IntelHashPoseidon2>;
pub type IntelHalSha256 = IntelHal<IntelHashSha256>;
pub type IntelHalPoseidon254 = IntelHal<IntelHashPoseidon254>;

// ============================================================================
// Buffer Pool — caches device allocations for reuse (matches CUDA HAL pattern)
//
// One process-wide pool rather than a thread_local one: buffers may be dropped
// on threads other than the allocating one, and a thread_local pool frees its
// whole cache when its thread exits. Device frees concurrent with GPU work on
// other threads coincided with intermittent stale device data on Intel (GPU
// page faults / DEVICE_LOST), and a free after DEVICE_LOST never returns, so
// nothing here is freed at thread or process exit. Reuse across threads is
// ordered by the single in-order queue: a buffer is only popped after its
// previous owner's last submission.
// ============================================================================

use std::{collections::HashMap, sync::LazyLock};

struct IntelBufferPool {
    cache: HashMap<usize, Vec<IntelDeviceBuffer>>,
    total_cached: usize,
}

/// The most the pool keeps in idle buffers: a quarter of the GPU's memory, at most 16 GiB. A
/// fixed 16 GiB is more than an Arc B580's whole memory (12 GB), and allocations made inside the
/// C++ kernels can't take it back from the pool the way [RawBuffer::new] can.
const POOL_MAX_BYTES: usize = 16 << 30;
const POOL_SMALL_THRESHOLD: usize = 64 << 10; // Always pool buffers under 64 KiB

impl IntelBufferPool {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            total_cached: 0,
        }
    }

    fn pop(&mut self, size: usize) -> Option<IntelDeviceBuffer> {
        if let Some(bufs) = self.cache.get_mut(&size) {
            if let Some(buf) = bufs.pop() {
                self.total_cached -= size;
                return Some(buf);
            }
        }
        None
    }

    /// Returns the buffer back if the pool is full, so the caller frees it
    /// outside the pool lock.
    fn push(
        &mut self,
        size: usize,
        buf: IntelDeviceBuffer,
        max_bytes: usize,
    ) -> Option<IntelDeviceBuffer> {
        if size < POOL_SMALL_THRESHOLD || self.total_cached + size <= max_bytes {
            self.total_cached += size;
            self.cache.entry(size).or_default().push(buf);
            None
        } else {
            Some(buf)
        }
    }

    fn take_all(&mut self) -> HashMap<usize, Vec<IntelDeviceBuffer>> {
        self.total_cached = 0;
        std::mem::take(&mut self.cache)
    }
}

/// Frees all pooled (idle) buffers. Call between proving sessions, when no GPU
/// work is in flight on another thread: otherwise buffers sized for an earlier
/// workload (e.g. a different po2) stay allocated, up to the pool cap, for the
/// life of the process and can crowd out a later, larger session.
pub fn release_idle_buffers() {
    // esimd_free_device does not wait for queued work; drain the queue first.
    unsafe { intel::esimd_sync(get_queue()) };
    let idle = BUFFER_POOL.lock().take_all();
    drop(idle);
}

static BUFFER_POOL: LazyLock<Mutex<IntelBufferPool>> =
    LazyLock::new(|| Mutex::new(IntelBufferPool::new()));

/// Cards with less memory than this keep only small buffers in the pool on Windows (see
/// [pool_max_bytes]).
const POOL_SMALL_CARD_BYTES: usize = 20 << 30;

/// The pool's cap for this GPU (see [POOL_MAX_BYTES]). The device's memory is known once the
/// queue exists, which every buffer's allocation has ensured.
///
/// On Windows a card with less than [POOL_SMALL_CARD_BYTES] pools only small buffers. There an
/// allocation beyond the card's memory does not fail but spills into shared system memory, so
/// [RawBuffer::new]'s release of the pool on a failed allocation never happens, and idle buffers
/// kept from earlier proofs pushed later ones off the card. On a 12 GB Arc B580 at po2=20, 3 GiB
/// of pooled buffers left the next sessions' segments at 42-46 s after the first's 28 s (with up
/// to 1.2 GB in shared memory); pooling only small buffers kept them at 28-30 s, with the first
/// session no slower. Sessions at po2=19 took ~3% longer without the pool (64.4 s, from 62.1).
fn pool_max_bytes() -> usize {
    static MAX: AtomicUsize = AtomicUsize::new(0);
    let max = MAX.load(Ordering::Relaxed);
    if max != 0 {
        return max;
    }
    let total = intel::device_total_memory();
    if total == 0 {
        return POOL_MAX_BYTES;
    }
    let max = if cfg!(windows) && total < POOL_SMALL_CARD_BYTES {
        // Nonzero, as 0 means not yet known; [IntelBufferPool::push] keeps small buffers anyway.
        1
    } else {
        (total / 4).min(POOL_MAX_BYTES)
    };
    MAX.store(max, Ordering::Relaxed);
    max
}

// ============================================================================
// RawBuffer - RAII device allocation with tracking + pool
// ============================================================================

struct RawBuffer {
    name: &'static str,
    buf: ManuallyDrop<IntelDeviceBuffer>,
}

impl RawBuffer {
    pub fn new(name: &'static str, size: usize) -> Self {
        tracing::trace!("alloc: {size} bytes, {name}");
        tracker().lock().unwrap().alloc(size);
        let pooled = BUFFER_POOL.lock().pop(size);
        let buf = pooled.unwrap_or_else(|| {
            IntelDeviceBuffer::uninitialized(size).unwrap_or_else(|_| {
                // Cached buffers of other sizes may be holding the VRAM.
                // esimd_free_device does not wait for queued work that may
                // still use them; drain the queue first.
                let all = BUFFER_POOL.lock().take_all();
                if !all.is_empty() {
                    unsafe { intel::esimd_sync(get_queue()) };
                }
                drop(all);
                IntelDeviceBuffer::uninitialized(size).unwrap_or_else(|e| {
                    panic!("Intel GPU allocation failed on {name}: {size} bytes: {e}")
                })
            })
        });
        Self {
            name,
            buf: ManuallyDrop::new(buf),
        }
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        let size = self.buf.len();
        tracing::trace!("free: {size} bytes, {}", self.name);
        tracker().lock().unwrap().free(size);
        let buf = unsafe { ManuallyDrop::take(&mut self.buf) };
        let max_bytes = pool_max_bytes();
        let overflow = BUFFER_POOL.lock().push(size, buf, max_bytes);
        if overflow.is_some() {
            // Over the pool cap (reachable in long sessions): the buffer is freed, and
            // esimd_free_device does not wait for queued kernels still using it.
            unsafe { intel::esimd_sync(get_queue()) };
        }
        drop(overflow);
    }
}

// ============================================================================
// BufferImpl<T> - typed view over RawBuffer
// ============================================================================

#[derive(Clone)]
pub struct BufferImpl<T> {
    // Arc<Mutex<...>> instead of Rc<RefCell<...>> so buffers can be sent
    // across thread boundaries (used for backgrounded finalize/receipt work).
    // RawBuffer carries device pointers + size; the mutex serialises any
    // mutating access (D2H/H2D copies, memset). Method-call frequency is
    // low enough that parking_lot::Mutex overhead vs RefCell is negligible.
    buffer: Arc<Mutex<RawBuffer>>,
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
            buffer: Arc::new(Mutex::new(RawBuffer::new(name, bytes_len))),
            size,
            offset: 0,
            marker: PhantomData,
        }
    }

    pub fn copy_from(name: &'static str, slice: &[T]) -> Self {
        let bytes_len = std::mem::size_of_val(slice);
        assert!(bytes_len > 0);
        let mut buffer = RawBuffer::new(name, bytes_len);
        let bytes: &[u8] = unchecked_cast(slice);
        buffer.buf.copy_from(bytes).unwrap();
        BufferImpl {
            buffer: Arc::new(Mutex::new(buffer)),
            size: slice.len(),
            offset: 0,
            marker: PhantomData,
        }
    }

    pub fn as_device_ptr(&self) -> DevicePointer<u8> {
        let ptr = self.buffer.lock().buf.as_device_ptr();
        let offset = self.offset * std::mem::size_of::<T>();
        unsafe { ptr.offset(offset.try_into().unwrap()) }
    }

    pub fn as_device_ptr_with_offset(&self, offset: usize) -> DevicePointer<u8> {
        let ptr = self.buffer.lock().buf.as_device_ptr();
        let offset = (self.offset + offset) * std::mem::size_of::<T>();
        unsafe { ptr.offset(offset.try_into().unwrap()) }
    }

    /// Upload host data directly to the GPU buffer without downloading first.
    /// More efficient than view_mut() when the entire buffer is being overwritten,
    /// since view_mut() does a needless D2H transfer before the H2D upload.
    pub fn copy_from_host(&self, data: &[T]) {
        assert_eq!(data.len(), self.size, "copy_from_host: size mismatch");
        let item_size = std::mem::size_of::<T>();
        let byte_offset = self.offset * item_size;
        let byte_len = self.size * item_size;
        let bytes: &[u8] = unchecked_cast(data);
        let buf = self.buffer.lock();
        let queue = get_queue();
        let ptr = buf.buf.as_device_ptr();
        unsafe {
            intel::esimd_memcpy_htod(
                queue,
                (ptr.0 as *mut u8).add(byte_offset) as *mut std::ffi::c_void,
                bytes.as_ptr() as *const std::ffi::c_void,
                byte_len,
            );
        }
    }
}

impl<T: Clone> Buffer<T> for BufferImpl<T> {
    fn name(&self) -> &'static str {
        self.buffer.lock().name
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
        let buf = self.buffer.lock();
        let offset = (self.offset + idx) * item_size;
        let host_buf = buf.buf.as_host_vec_range(offset, item_size).unwrap();
        let slice: &[T] = unchecked_cast(&host_buf[..]);
        slice[0].clone()
    }

    fn view<F: FnOnce(&[T])>(&self, f: F) {
        scope!("view");
        let item_size = std::mem::size_of::<T>();
        let buf = self.buffer.lock();
        let offset = self.offset * item_size;
        let len = self.size * item_size;
        let host_buf = buf.buf.as_host_vec_range(offset, len).unwrap();
        let slice: &[T] = unchecked_cast(&host_buf[..]);
        f(slice);
    }

    fn view_mut<F: FnOnce(&mut [T])>(&self, f: F) {
        scope!("view_mut");
        let item_size = std::mem::size_of::<T>();
        let byte_offset = self.offset * item_size;
        let byte_len = self.size * item_size;
        let mut buf = self.buffer.lock();
        let total_bytes = buf.buf.len();

        if byte_offset == 0 && byte_len == total_bytes {
            // Fast path: BufferImpl covers the entire RawBuffer.
            // Download only our region, modify, upload only our region.
            let mut host_buf = buf.buf.as_host_vec().unwrap();
            let slice = unchecked_cast_mut(&mut host_buf);
            f(slice);
            buf.buf.copy_from(&host_buf).unwrap();
        } else {
            // Slice path: download just the slice region, modify, upload just the region.
            let mut host_buf = buf.buf.as_host_vec_range(byte_offset, byte_len).unwrap();
            let slice: &mut [T] = unchecked_cast_mut(&mut host_buf);
            f(slice);
            // Upload just the modified region back to the correct offset in device memory.
            let queue = get_queue();
            let ptr = buf.buf.as_device_ptr();
            unsafe {
                intel::esimd_memcpy_htod(
                    queue,
                    (ptr.0 as *mut u8).add(byte_offset) as *mut std::ffi::c_void,
                    host_buf.as_ptr() as *const std::ffi::c_void,
                    byte_len,
                );
            }
        }
    }

    fn to_vec(&self) -> Vec<T> {
        let item_size = std::mem::size_of::<T>();
        let buf = self.buffer.lock();
        let offset = self.offset * item_size;
        let len = self.size * item_size;
        let host_buf = buf.buf.as_host_vec_range(offset, len).unwrap();
        let slice: &[T] = unchecked_cast(&host_buf[..]);
        slice.to_vec()
    }
}

// ============================================================================
// IntelHal construction
// ============================================================================

impl<IH: IntelHash> Default for IntelHal<IH> {
    fn default() -> Self {
        Self::new()
    }
}

impl<IH: IntelHash + ?Sized> IntelHal<IH> {
    pub fn new() -> Self
    where
        IH: Sized,
    {
        Self::new_from_hash(Box::new(IH::new()))
    }

    fn new_from_hash(hash: Box<IH>) -> Self {
        // Acquire singleton lock during init only — released at end of fn so
        // IntelHal itself can be Send. GPU-touching methods re-acquire the
        // lock per call (singleton is a ReentrantMutex; same-thread reentry
        // is free).
        let _init_lock = singleton().lock();

        // Ensure the SYCL queue is initialized (creates on first call)
        let queue = get_queue();

        // Pre-warm twiddle tables + SYCL runtime (only once per process; may
        // already have run on the background init thread).
        intel::warmup_once(queue);

        let mut hal = Self { hash: None };
        hal.hash = Some(hash);
        // _init_lock dropped here — hal becomes Send-friendly.
        hal
    }
}

// ============================================================================
// Hal trait implementation for IntelHal
// ============================================================================

impl<IH: IntelHash + ?Sized> Hal for IntelHal<IH> {
    type Field = BabyBear;
    type Elem = BabyBearElem;
    type ExtElem = BabyBearExtElem;
    type Buffer<T: Clone + Debug + PartialEq> = BufferImpl<T>;

    fn has_unified_memory(&self) -> bool {
        false
    }

    fn get_hash_suite(&self) -> &HashSuite<Self::Field> {
        self.hash.as_ref().unwrap().get_hash_suite()
    }

    // ---- Allocation ----

    fn alloc_elem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::Elem> {
        BufferImpl::new(name, size)
    }

    fn alloc_extelem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        BufferImpl::new(name, size)
    }

    fn alloc_digest(&self, name: &'static str, size: usize) -> Self::Buffer<Digest> {
        BufferImpl::new(name, size)
    }

    fn alloc_u32(&self, name: &'static str, size: usize) -> Self::Buffer<u32> {
        BufferImpl::new(name, size)
    }

    /// Override the default alloc_elem_init to avoid the expensive
    /// download-fill-upload round-trip. Uses GPU-side memset/set_32 instead.
    ///
    /// The default impl does: alloc -> download (uninitialized!) -> fill on CPU -> upload.
    /// For large buffers (e.g. data=844MB at po2=20), this wastes seconds on PCIe transfers.
    /// Instead, we use GPU-side memset for common patterns:
    /// - INVALID (0xFFFFFFFF): memset(0xFF) -- all bytes are 0xFF
    /// - ZERO (0x00000000): memset(0)
    /// - Other values: set_32 (uploads a host buffer)
    fn alloc_elem_init(
        &self,
        name: &'static str,
        size: usize,
        value: Self::Elem,
    ) -> Self::Buffer<Self::Elem> {
        let buf = BufferImpl::new(name, size);
        let bytes_len = size * std::mem::size_of::<Self::Elem>();
        // Access the raw Montgomery representation directly.
        // SAFETY: Elem is #[repr(transparent)] wrapping u32 and derives NoUninit,
        // so transmuting to u32 is guaranteed safe.
        let raw_bits: u32 = unsafe { std::mem::transmute(value) };
        if raw_bits == 0xFFFFFFFF {
            // INVALID sentinel: all bytes are 0xFF
            buf.buffer.lock().buf.memset(0xFF_u8 as i32, bytes_len);
        } else if raw_bits == 0 {
            // ZERO: all bytes are 0x00
            buf.buffer.lock().buf.memset(0, bytes_len);
        } else {
            // General case: use set_32
            buf.buffer.lock().buf.set_32(raw_bits);
        }
        buf
    }

    /// Override the default alloc_extelem_zeroed to use GPU-side memset(0)
    /// instead of download-fill-upload round-trip.
    fn alloc_extelem_zeroed(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        let buf = BufferImpl::new(name, size);
        let bytes_len = size * std::mem::size_of::<Self::ExtElem>();
        buf.buffer.lock().buf.memset(0, bytes_len);
        buf
    }

    // ---- Copy-from host ----

    fn copy_from_elem(&self, name: &'static str, slice: &[Self::Elem]) -> Self::Buffer<Self::Elem> {
        BufferImpl::copy_from(name, slice)
    }

    fn copy_from_extelem(
        &self,
        name: &'static str,
        slice: &[Self::ExtElem],
    ) -> Self::Buffer<Self::ExtElem> {
        BufferImpl::copy_from(name, slice)
    }

    fn copy_from_digest(&self, name: &'static str, slice: &[Digest]) -> Self::Buffer<Digest> {
        BufferImpl::copy_from(name, slice)
    }

    fn copy_from_u32(&self, name: &'static str, slice: &[u32]) -> Self::Buffer<u32> {
        BufferImpl::copy_from(name, slice)
    }

    // ---- NTT operations ----

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

        let lg_out = log2_ceil(out_size);
        assert_eq!(out_size, 1 << lg_out);
        assert!(lg_out < Self::Elem::MAX_ROU_PO2);

        let queue = get_queue();

        // Fused LDE for the x4 blowup: one pass per polynomial reads the
        // coefficients and writes the evaluations, without materialising the
        // zero-expanded buffer (same output). RISC0_FUSED_EXPAND_OFF=1 keeps
        // the two-step path below.
        // The fused pass is the v3 SLM kernel, so RISC0_NTT_V3_OFF disables it too.
        static FUSED_EXPAND: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var_os("RISC0_FUSED_EXPAND_OFF").is_none()
                && std::env::var_os("RISC0_NTT_V3_OFF").is_none()
        });
        if expand_bits == 2 && (14..26).contains(&lg_out) && *FUSED_EXPAND {
            esimd_check(unsafe {
                intel::esimd_batch_expand_fwd_fused(
                    queue,
                    output.as_device_ptr().0 as *mut std::ffi::c_void,
                    input.as_device_ptr().0 as *const std::ffi::c_void,
                    ffi_u32(lg_out),
                    ffi_u32(expand_bits),
                    ffi_u32(poly_count),
                )
            });
            return;
        }

        // Step 1: GPU expand — fused zero+scatter in one pass.
        // Uses stride-scatter pattern matching the CPU's DIF expand_bits skip.
        esimd_check(unsafe {
            intel::esimd_batch_expand_ffi(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                input.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(in_size),
                ffi_u32(out_size),
                ffi_u32(poly_count),
                ffi_u32(expand_bits),
            )
        });

        // Step 2: Batch forward NTT — all polynomials submitted without intermediate sync.
        esimd_check(unsafe {
            intel::esimd_batch_forward_ntt(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                ffi_u32(lg_out),
                ffi_u32(poly_count),
                ffi_u32(out_size),
            )
        });
    }

    fn batch_interpolate_ntt(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        assert!(n_bits < Self::Elem::MAX_ROU_PO2);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_batch_inverse_ntt(
                queue,
                io.as_device_ptr().0 as *mut std::ffi::c_void,
                ffi_u32(n_bits),
                ffi_u32(count),
                ffi_u32(row_size),
            )
        });
    }

    fn batch_bit_reverse(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << bits);

        let queue = get_queue();
        // count parameter to FFI is total elements, not poly count
        esimd_check(unsafe {
            intel::esimd_batch_bit_reverse_ffi(
                queue,
                io.as_device_ptr().0 as *mut std::ffi::c_void,
                ffi_u32(bits),
                ffi_u32(io.size()),
            )
        });
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

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_batch_evaluate_any_ffi(
                queue,
                out.as_device_ptr().0 as *mut std::ffi::c_void,
                coeffs.as_device_ptr().0 as *const std::ffi::c_void,
                which.as_device_ptr().0 as *const std::ffi::c_void,
                xs.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(eval_count),
                ffi_u32(count),
            )
        });
    }

    fn zk_shift(&self, io: &Self::Buffer<Self::Elem>, poly_count: usize) {
        let bits = log2_ceil(io.size() / poly_count);
        assert_eq!(io.size(), poly_count * (1 << bits));

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_batch_zk_shift(
                queue,
                io.as_device_ptr().0 as *mut std::ffi::c_void,
                ffi_u32(bits),
                ffi_u32(poly_count),
            )
        });
    }

    // T1-C: Fused inverse-NTT + zk_shift. The standalone zk_shift kernel reads
    // and writes the full buffer; fusing the multiply into the INTT final store
    // eliminates one full-bandwidth pass per polynomial. Uses the same cached
    // power table as zk_shift, so first-touch cost is unchanged.
    fn batch_interpolate_ntt_zk_shift(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        assert!(n_bits < Self::Elem::MAX_ROU_PO2);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_batch_inverse_ntt_zk_shift(
                queue,
                io.as_device_ptr().0 as *mut std::ffi::c_void,
                ffi_u32(n_bits),
                ffi_u32(count),
                ffi_u32(row_size),
            )
        });
    }

    // Out-of-place: the first inverse-NTT pass reads the witness directly
    // (saves the make_coeffs copy, ~3 ms per po2=20 segment).
    fn batch_interpolate_ntt_zk_shift_from(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        count: usize,
    ) {
        assert_eq!(dst.size(), src.size());
        let row_size = src.size() / count;
        assert_eq!(row_size * count, src.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        assert!(n_bits < Self::Elem::MAX_ROU_PO2);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_batch_inverse_ntt_zk_shift_from(
                queue,
                dst.as_device_ptr().0 as *mut std::ffi::c_void,
                src.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(n_bits),
                ffi_u32(count),
                ffi_u32(row_size),
            )
        });
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
        let mix_start_buf = self.copy_from_extelem("mix_start", &[*mix_start]);
        let mix_buf = self.copy_from_extelem("mix", &[*mix]);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_mix_poly_coeffs_ffi(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                input.as_device_ptr().0 as *const std::ffi::c_void,
                combos.as_device_ptr().0 as *const std::ffi::c_void,
                mix_start_buf.as_device_ptr().0 as *const std::ffi::c_void,
                mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(input_size),
                ffi_u32(count),
            )
        });
    }

    // ---- Element-wise operations ----

    fn eltwise_add_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input1: &Self::Buffer<Self::Elem>,
        input2: &Self::Buffer<Self::Elem>,
    ) {
        assert_eq!(output.size(), input1.size());
        assert_eq!(output.size(), input2.size());
        let count = output.size();

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_eltwise_add_fp_ffi(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                input1.as_device_ptr().0 as *const std::ffi::c_void,
                input2.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(count),
            )
        });
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

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_eltwise_sum_fpext_ffi(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                input.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(to_add),
                ffi_u32(count),
            )
        });
    }

    fn eltwise_copy_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
    ) {
        let count = output.size();
        assert_eq!(count, input.size());

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_eltwise_copy_fp_ffi(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                input.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(count),
            )
        });
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
        if from_rows > 0 && from_cols > 0 {
            let last = (from_rows - 1, from_cols - 1);
            assert!(from_offset + last.0 * from_stride + last.1 < from.len());
            assert!(into_offset + last.0 * into_stride + last.1 < into.size());
        }
        let from_buf = self.copy_from_elem("from", from);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_eltwise_copy_fp_region_ffi(
                queue,
                into.as_device_ptr().0 as *mut std::ffi::c_void,
                from_buf.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(from_rows),
                ffi_u32(from_cols),
                ffi_u32(from_offset),
                ffi_u32(from_stride),
                ffi_u32(into_offset),
                ffi_u32(into_stride),
            )
        });
    }

    fn eltwise_zeroize_elem(&self, elems: &Self::Buffer<Self::Elem>) {
        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_eltwise_zeroize_fp_ffi(
                queue,
                elems.as_device_ptr().0 as *mut std::ffi::c_void,
                ffi_u32(elems.size()),
            )
        });
    }

    // ---- FRI ----

    fn fri_fold(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        mix: &Self::ExtElem,
    ) {
        let count = output.size() / Self::ExtElem::EXT_SIZE;
        assert_eq!(output.size(), count * Self::ExtElem::EXT_SIZE);
        assert_eq!(input.size(), output.size() * FRI_FOLD);
        let mix_buf = self.copy_from_extelem("mix", &[*mix]);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_fri_fold_ffi(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                input.as_device_ptr().0 as *const std::ffi::c_void,
                mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(count),
            )
        });
    }

    // ---- Hashing ----

    fn hash_rows(&self, output: &Self::Buffer<Digest>, matrix: &Self::Buffer<Self::Elem>) {
        self.hash.as_ref().unwrap().hash_rows(output, matrix);
    }

    fn hash_fold(&self, io: &Self::Buffer<Digest>, input_size: usize, output_size: usize) {
        assert_eq!(input_size, 2 * output_size);
        self.hash.as_ref().unwrap().hash_fold(io, output_size);
    }

    /// R2-A07: Override `hash_fold_tree` to stop GPU folding at `cutoff`
    /// (default 1024 = gate0 OpenCL LWS) and finish remaining layers on CPU
    /// via rayon + hashfn.hash_pair.
    ///
    /// Rationale: gate0 fold kernel hard-codes LWS=1024 (poseidon2.cpp:528).
    /// Layers with num_hashes < 1024 launch a full 1024-WI WG → wasted
    /// dispatch latency. ~7 merkle trees per seg × ~10 sub-LWS tail layers
    /// ≈ 70 wasted submits/seg. CPU-finishing 1023 hash_pair ops on rayon is
    /// faster than the dispatch overhead. CPU `hash_pair` is bit-exact with
    /// the kernel's Mont-form I/O (project_poseidon2_mont_convention.md).
    /// Tunable: RISC0_HASH_FOLD_CPU_CUTOFF (0 disables; default 1024).
    fn hash_fold_tree(&self, io: &Self::Buffer<Digest>, layers: usize) {
        // Default cutoff=512 based on cutoff-sweep (2 runs/setting on
        // prove_and_verify 42-seg): 512 → 146.61s, 1024 → 146.91s, 2048 → 147.24s.
        // 512 means GPU folds down to output_size=512, then CPU finishes one more
        // layer than at cutoff=1024. Within measurement noise but reproducible.
        let cutoff: usize = std::env::var("RISC0_HASH_FOLD_CPU_CUTOFF")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);

        if cutoff == 0 || cutoff >= (1 << layers) {
            for i in (0..layers).rev() {
                let layer_size = 1 << i;
                self.hash_fold(io, layer_size * 2, layer_size);
            }
            return;
        }

        let cutoff_log2 = (usize::BITS - 1 - cutoff.leading_zeros()) as usize;
        let cutoff_po2: usize = 1 << cutoff_log2;

        // GPU phase: fold while output_size >= cutoff_po2.
        for i in (cutoff_log2..layers).rev() {
            let layer_size = 1 << i;
            self.hash_fold(io, layer_size * 2, layer_size);
        }

        // CPU phase: finish remaining cutoff_log2 layers using rayon + hash_pair.
        // view_mut on a slice (intel.rs:454+) D2H's the region, runs the closure,
        // and H2D's the modified bytes back to the device.
        let region = io.slice(1, 2 * cutoff_po2 - 1);
        let hash_suite = self.hash.as_ref().unwrap().get_hash_suite();
        let hashfn = hash_suite.hashfn.clone();
        region.view_mut(|buf| {
            use rayon::prelude::*;
            for i in (0..cutoff_log2).rev() {
                let out_off = (1usize << i) - 1;
                let in_off = (1usize << (i + 1)) - 1;
                let layer_size = 1usize << i;
                let (outs_full, ins_full) = buf.split_at_mut(in_off);
                let outs = &mut outs_full[out_off..out_off + layer_size];
                let ins = &ins_full[..2 * layer_size];
                outs.par_iter_mut().enumerate().for_each(|(idx, out)| {
                    *out = *hashfn.hash_pair(&ins[2 * idx], &ins[2 * idx + 1]);
                });
            }
        });
    }

    // ---- Gather / Scatter ----

    fn batch_get_digest_at(&self, buf: &Self::Buffer<Digest>, indices: &[usize]) -> Vec<Digest> {
        if indices.is_empty() {
            return Vec::new();
        }
        let count = indices.len();
        let indices_u32: Vec<u32> = indices.iter().map(|&i| ffi_u32(i)).collect();
        let indices_buf = self.copy_from_u32("gather_indices", &indices_u32);
        let output_buf: BufferImpl<Digest> = BufferImpl::new("gather_output", count);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_gather_digests_ffi(
                queue,
                output_buf.as_device_ptr().0 as *mut std::ffi::c_void,
                buf.as_device_ptr().0 as *const std::ffi::c_void,
                indices_buf.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(count),
            )
        });

        let mut result = Vec::new();
        output_buf.view(|view| {
            result = view.to_vec();
        });
        result
    }

    fn gather_sample(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        idx: usize,
        size: usize,
        stride: usize,
    ) {
        // The kernel dereferences raw device pointers; mirror the CPU HAL's
        // slice bounds checks so a bad call panics instead of faulting the GPU.
        assert!(dst.size() >= size, "gather_sample: dst too small");
        if size > 0 {
            assert!(
                (size - 1) * stride + idx < src.size(),
                "gather_sample: src out of range"
            );
        }
        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_gather_sample_fp_ffi(
                queue,
                dst.as_device_ptr().0 as *mut std::ffi::c_void,
                src.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(idx),
                ffi_u32(size),
                ffi_u32(stride),
            )
        });
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
        // The kernel writes through raw device pointers (see gather_sample).
        let n = index[count] as usize;
        assert!(
            n <= offsets.len() && n <= values.len(),
            "scatter: index exceeds inputs"
        );
        debug_assert!(
            index.windows(2).all(|w| w[0] <= w[1]),
            "scatter: index not sorted"
        );
        debug_assert!(
            offsets[..n].iter().all(|&o| (o as usize) < into.size()),
            "scatter: offset out of range"
        );

        // Upload index, offsets, values to device then call ESIMD scatter
        let index_buf = self.copy_from_u32("scatter_index", index);
        let offsets_buf = self.copy_from_u32("scatter_offsets", offsets);
        let values_buf = self.copy_from_elem("scatter_values", values);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_scatter_fp_ffi(
                queue,
                into.as_device_ptr().0 as *mut std::ffi::c_void,
                index_buf.as_device_ptr().0 as *const std::ffi::c_void,
                offsets_buf.as_device_ptr().0 as *const std::ffi::c_void,
                values_buf.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(count),
            )
        });
    }

    // ---- prefix_products: CPU fallback ----

    fn prefix_products(&self, io: &Self::Buffer<Self::ExtElem>) {
        io.view_mut(|io| {
            for i in 1..io.len() {
                io[i] *= io[i - 1];
            }
        });
    }

    // combos_divide on the device: blocked synthetic division (B=64-element
    // blocks, one Horner pass, a W<=1024-lane SLM suffix scan for the carries,
    // one rewrite pass that also produces the next division's block values),
    // batched per round over the columns. ~4 ms at po2=20 vs ~110 ms for the
    // view_mut round trip + CPU divide. Results are bit-identical (exact field
    // arithmetic). The remainder == 0 check is preserved: the FFI reads the
    // remainders back and fails, and we re-check them here.
    fn combos_divide(
        &self,
        combos: &Self::Buffer<Self::ExtElem>,
        chunks: Vec<(usize, Vec<Self::ExtElem>)>,
        cycles: usize,
    ) {
        scope!("combos_divide");
        if cycles < (1 << 13) || std::env::var_os("RISC0_INTEL_CPU_COMBOS_DIVIDE").is_some() {
            // Small domains (GPU path has ~1 ms fixed cost; CPU wins below 2^13 rows)
            // and opt-out: CPU path, identical to the trait default.
            use rayon::prelude::*;
            combos.view_mut(|combos| {
                combos.par_chunks_exact_mut(cycles).zip(chunks).for_each(
                    |(combo_slice, (i, pows))| {
                        for pow in pows {
                            let remainder = crate::core::poly::poly_divide(combo_slice, pow);
                            assert_eq!(
                                remainder,
                                <BabyBearExtElem as risc0_core::field::Elem>::ZERO,
                                "i: {i}"
                            );
                        }
                    },
                );
            });
            return;
        }
        assert!(
            cycles.is_power_of_two(),
            "combos_divide: cycles {cycles} not a power of two"
        );
        let ncols = combos.size() / cycles;
        assert_eq!(ncols * cycles, combos.size());
        let mut info: Vec<u32> = Vec::new();
        for (col, pows) in &chunks {
            assert!(*col < ncols, "combos_divide: column {col} >= {ncols}");
            for pow in pows {
                info.push(*col as u32);
                info.extend_from_slice(unchecked_cast::<Self::ExtElem, u32>(std::slice::from_ref(
                    pow,
                )));
            }
        }
        let count = info.len() / 5;
        if count == 0 {
            return;
        }
        let mut rems = vec![0u32; count * 4];
        esimd_check(unsafe {
            intel::esimd_combos_divide_v2_ffi(
                get_queue(),
                combos.as_device_ptr().0 as *mut std::ffi::c_void,
                ffi_u32(cycles),
                ffi_u32(ncols),
                info.as_ptr() as *const std::ffi::c_void,
                ffi_u32(count),
                rems.as_mut_ptr() as *mut std::ffi::c_void,
            )
        });
        for (k, r) in rems.chunks_exact(4).enumerate() {
            assert!(
                r.iter().all(|&w| w == 0),
                "combos_divide: nonzero remainder (entry {k})"
            );
        }
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
        scope!("combos_prepare");
        // Upload small parameter buffers to GPU
        let coeff_u_buf = self.copy_from_extelem("coeff_u", coeff_u);
        let reg_sizes_buf = self.copy_from_u32("reg_sizes", reg_sizes);
        let reg_combo_ids_buf = self.copy_from_u32("reg_combo_ids", reg_combo_ids);
        let mix_buf = self.copy_from_extelem("mix", &[*mix]);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_combos_prepare_ffi(
                queue,
                combos.as_device_ptr().0 as *mut std::ffi::c_void,
                coeff_u_buf.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(combo_count),
                ffi_u32(cycles),
                ffi_u32(reg_sizes.len()),
                reg_sizes_buf.as_device_ptr().0 as *const std::ffi::c_void,
                reg_combo_ids_buf.as_device_ptr().0 as *const std::ffi::c_void,
                ffi_u32(Self::CHECK_SIZE),
                mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
            )
        });
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use test_log::test;

    use super::{IntelHalPoseidon2, IntelHalSha256};
    use crate::hal::testutil;

    #[test]
    #[should_panic]
    fn check_req() {
        testutil::check_req(IntelHalSha256::new());
    }

    #[test]
    fn eltwise_add_elem() {
        testutil::eltwise_add_elem(IntelHalSha256::new());
    }

    #[test]
    fn eltwise_copy_elem() {
        testutil::eltwise_copy_elem(IntelHalSha256::new());
    }

    #[test]
    fn eltwise_sum_extelem() {
        testutil::eltwise_sum_extelem(IntelHalSha256::new());
    }

    #[test]
    fn hash_rows_sha256() {
        testutil::hash_rows(IntelHalSha256::new());
    }

    #[test]
    fn hash_fold_sha256() {
        testutil::hash_fold(IntelHalSha256::new());
    }

    #[test]
    fn hash_rows_poseidon2() {
        testutil::hash_rows(IntelHalPoseidon2::new());
    }

    #[test]
    fn hash_fold_poseidon2() {
        testutil::hash_fold(IntelHalPoseidon2::new());
    }

    #[test]
    fn fri_fold() {
        testutil::fri_fold(IntelHalSha256::new());
    }

    #[test]
    fn batch_expand_into_evaluate_ntt() {
        testutil::batch_expand_into_evaluate_ntt(IntelHalSha256::new());
    }

    #[test]
    fn batch_interpolate_ntt() {
        testutil::batch_interpolate_ntt(IntelHalSha256::new());
    }

    #[test]
    fn batch_bit_reverse() {
        testutil::batch_bit_reverse(IntelHalSha256::new());
    }

    #[test]
    fn batch_evaluate_any() {
        testutil::batch_evaluate_any(IntelHalSha256::new());
    }

    #[test]
    fn gather_sample() {
        testutil::gather_sample(IntelHalSha256::new());
    }

    #[test]
    fn zk_shift() {
        testutil::zk_shift(IntelHalSha256::new());
    }

    #[test]
    fn mix_poly_coeffs() {
        testutil::mix_poly_coeffs(IntelHalSha256::new());
    }

    // At po2=21 the data group's evaluated matrix (211 polys x 2^23) is ~7 GB,
    // past 4 GiB; any 32-bit byte addressing wraps there. Run the ops that touch
    // it at that size and compare against the CPU HAL.
    #[test]
    #[ignore = "requires an Intel GPU; run explicitly"]
    fn batch_evaluate_any_power_tables_match_cpu() {
        // deg >= 2^14 takes the power-table path (eval_u). Taps share a few
        // distinct points across polynomials, like the prover's back-taps.
        use crate::{
            core::hash::poseidon2::Poseidon2HashSuite,
            field::{
                baby_bear::{BabyBearElem as Elem, BabyBearExtElem as ExtElem},
                Elem as _,
            },
            hal::{cpu::CpuHal, Buffer as _, Hal as _},
        };
        use rand::{rngs::StdRng, Rng as _, SeedableRng as _};

        let gpu = IntelHalPoseidon2::new();
        let cpu = CpuHal::new(Poseidon2HashSuite::new_suite());
        let mut rng = StdRng::seed_from_u64(11);
        let mut bad_total = 0;
        for (lg, polys, npoints) in [
            (14usize, 40usize, 3usize),
            (16, 120, 9),
            (20, 64, 9),
            (21, 24, 5),
        ] {
            let deg = 1usize << lg;
            let coeffs: Vec<Elem> = (0..polys * deg).map(|_| Elem::random(&mut rng)).collect();
            let points: Vec<ExtElem> = (0..npoints).map(|_| ExtElem::random(&mut rng)).collect();
            let mut which = vec![];
            let mut xs = vec![];
            for p in 0..polys {
                // 1..=7 taps per poly, so classes of 1..6 points and a split class.
                let taps = 1 + rng.random_range(0..7usize);
                for t in 0..taps {
                    which.push(p as u32);
                    xs.push(points[(p + 2 * t) % npoints]);
                }
            }
            let g_out = gpu.alloc_extelem("out", which.len());
            gpu.batch_evaluate_any(
                &gpu.copy_from_elem("coeffs", &coeffs),
                polys,
                &gpu.copy_from_u32("which", &which),
                &gpu.copy_from_extelem("xs", &xs),
                &g_out,
            );
            let c_out = cpu.alloc_extelem("out", which.len());
            cpu.batch_evaluate_any(
                &cpu.copy_from_elem("coeffs", &coeffs),
                polys,
                &cpu.copy_from_u32("which", &which),
                &cpu.copy_from_extelem("xs", &xs),
                &c_out,
            );
            let bad = g_out
                .to_vec()
                .iter()
                .zip(&c_out.to_vec())
                .filter(|(a, b)| a != b)
                .count();
            eprintln!(
                "[eval_any] lg={lg} polys={polys} taps={}: {bad} differ",
                which.len()
            );
            bad_total += bad;
        }
        assert_eq!(
            bad_total, 0,
            "power-table batch_evaluate_any diverged from CPU"
        );
    }

    #[test]
    #[ignore = "requires an Intel GPU; run explicitly"]
    fn poseidon254_hash_ops_match_cpu() {
        use crate::{
            core::hash::poseidon_254::Poseidon254HashSuite,
            field::{baby_bear::BabyBearElem as Elem, Elem as _},
            hal::{cpu::CpuHal, Buffer as _, Hal as _},
        };
        use rand::{rngs::StdRng, SeedableRng as _};

        let gpu = super::IntelHalPoseidon254::new();
        let cpu = CpuHal::new(Poseidon254HashSuite::new_suite());
        let mut rng = StdRng::seed_from_u64(9);
        let mut failures = vec![];
        for &cols in &[1usize, 7, 8, 9, 16, 17, 24, 33, 64] {
            for &rows in &[1usize, 2, 8, 15, 16, 17, 1024, 1 << 16] {
                let data: Vec<Elem> = (0..rows * cols).map(|_| Elem::random(&mut rng)).collect();
                let g_out = gpu.alloc_digest("out", rows);
                gpu.hash_rows(&g_out, &gpu.copy_from_elem("in", &data));
                let c_out = cpu.alloc_digest("out", rows);
                cpu.hash_rows(&c_out, &cpu.copy_from_elem("in", &data));
                let bad = g_out
                    .to_vec()
                    .iter()
                    .zip(&c_out.to_vec())
                    .filter(|(a, b)| a != b)
                    .count();
                if bad > 0 {
                    eprintln!("[p254 rows] cols={cols} rows={rows}: {bad} differ");
                    failures.push(format!("rows {cols}x{rows}"));
                }
            }
        }
        // Whole-tree fold, including the GPU/CPU cutoff and sub-16 layers.
        for layers in [1usize, 3, 4, 5, 9, 10, 11, 16] {
            let leaves = 1usize << layers;
            let digests: Vec<crate::core::digest::Digest> = (0..2 * leaves)
                .map(|_| {
                    // Digests are canonical BN254 Fr encodings: keep the top
                    // limb below r's (0x30644e72).
                    let mut w: Vec<u32> = (0..8).map(|_| Elem::random(&mut rng).as_u32()).collect();
                    w[7] %= 0x3000_0000;
                    crate::core::digest::Digest::try_from(w.as_slice()).unwrap()
                })
                .collect();
            let g = gpu.copy_from_digest("tree", &digests);
            let c = cpu.copy_from_digest("tree", &digests);
            gpu.hash_fold_tree(&g, layers);
            for i in (0..layers).rev() {
                cpu.hash_fold(&c, 2 << i, 1 << i);
            }
            let (gv, cv) = (g.to_vec(), c.to_vec());
            if gv[1] != cv[1] {
                eprintln!("[p254 fold] layers={layers}: root differs");
                failures.push(format!("fold {layers}"));
            }
        }
        assert!(failures.is_empty(), "Poseidon-254 mismatches: {failures:?}");
    }

    #[test]
    #[ignore = "requires an Intel GPU; run explicitly"]
    fn poseidon2_hash_rows_shapes_match_cpu() {
        use crate::{
            core::hash::poseidon2::Poseidon2HashSuite,
            field::{baby_bear::BabyBearElem as Elem, Elem as _},
            hal::{cpu::CpuHal, Buffer as _, Hal as _},
        };
        use rand::{rngs::StdRng, SeedableRng as _};

        let gpu = IntelHalPoseidon2::new();
        let cpu = CpuHal::new(Poseidon2HashSuite::new_suite());
        let mut rng = StdRng::seed_from_u64(7);
        let mut failures = vec![];
        // Column counts of the rv32im/recursion commit groups plus sponge-block
        // edge cases; row counts span the small-po2 domains.
        for &cols in &[1usize, 15, 16, 17, 23, 24, 25, 32, 33, 48, 103, 128, 211] {
            for &rows in &[16usize, 1024, 1 << 14, 1 << 16, 1 << 18] {
                let data: Vec<Elem> = (0..rows * cols).map(|_| Elem::random(&mut rng)).collect();
                let g_in = gpu.copy_from_elem("in", &data);
                let g_out = gpu.alloc_digest("out", rows);
                gpu.hash_rows(&g_out, &g_in);
                let c_in = cpu.copy_from_elem("in", &data);
                let c_out = cpu.alloc_digest("out", rows);
                cpu.hash_rows(&c_out, &c_in);
                let g = g_out.to_vec();
                let c = c_out.to_vec();
                let bad = g.iter().zip(&c).filter(|(a, b)| a != b).count();
                if bad > 0 {
                    let first = g.iter().zip(&c).position(|(a, b)| a != b);
                    eprintln!("[hash_rows] cols={cols} rows={rows}: {bad} differ, first={first:?}");
                    failures.push((cols, rows));
                }
            }
        }
        // Whole-tree fold: small layers, the CPU cutoff, and large layers
        // that use the separate fold kernel (>= 1024 hashes per layer).
        for layers in [1usize, 4, 9, 10, 11, 14, 18, 20] {
            let n = 2usize << layers;
            let digests: Vec<crate::core::digest::Digest> = (0..n)
                .map(|_| {
                    let w: Vec<u32> = (0..8)
                        .map(|_| Elem::random(&mut rng).as_u32_montgomery())
                        .collect();
                    crate::core::digest::Digest::try_from(w.as_slice()).unwrap()
                })
                .collect();
            let g = gpu.copy_from_digest("tree", &digests);
            let c = cpu.copy_from_digest("tree", &digests);
            gpu.hash_fold_tree(&g, layers);
            for i in (0..layers).rev() {
                cpu.hash_fold(&c, 2 << i, 1 << i);
            }
            let (gv, cv) = (g.to_vec(), c.to_vec());
            let bad = (1..(1usize << layers)).filter(|&i| gv[i] != cv[i]).count();
            if bad > 0 {
                eprintln!("[hash_fold] layers={layers}: {bad} interior nodes differ");
                failures.push((0, layers));
            }
        }
        assert!(
            failures.is_empty(),
            "hash_rows/hash_fold mismatches: {failures:?}"
        );
    }

    #[test]
    #[ignore = "needs ~10 GB of GPU memory and several minutes"]
    fn large_buffer_ops_match_cpu() {
        use crate::{
            core::hash::poseidon2::Poseidon2HashSuite,
            field::{baby_bear::BabyBearElem as Elem, Elem as _},
            hal::{cpu::CpuHal, Buffer as _, Hal as _},
        };

        fn report<T: PartialEq>(name: &str, gpu: &[T], cpu: &[T], elem_bytes: usize) -> usize {
            assert_eq!(gpu.len(), cpu.len(), "{name}: length");
            let gib4 = (4usize << 30) / elem_bytes;
            let mut bad = 0;
            let mut beyond = 0;
            let mut first = None;
            for (i, (a, b)) in gpu.iter().zip(cpu).enumerate() {
                if a != b {
                    bad += 1;
                    beyond += (i >= gib4) as usize;
                    first.get_or_insert(i);
                }
            }
            eprintln!(
                "[large] {name}: {bad}/{} differ ({beyond} at/after the 4 GiB mark, index {gib4}); first={first:?}",
                gpu.len()
            );
            bad
        }

        let gpu = IntelHalPoseidon2::new();
        let cpu = CpuHal::new(Poseidon2HashSuite::new_suite());
        let count = 211;
        let expand_bits = 2;
        let steps = 1 << 21;
        let domain = steps << expand_bits;

        let mut rng = rand::rng();
        let input: Vec<Elem> = (0..count * steps).map(|_| Elem::random(&mut rng)).collect();
        let gpu_in = gpu.copy_from_elem("in", &input);
        let cpu_in = cpu.copy_from_elem("in", &input);
        drop(input);

        let gpu_eval = gpu.alloc_elem("eval", count * domain);
        let cpu_eval = cpu.alloc_elem("eval", count * domain);
        gpu.batch_expand_into_evaluate_ntt(&gpu_eval, &gpu_in, count, expand_bits);
        cpu.batch_expand_into_evaluate_ntt(&cpu_eval, &cpu_in, count, expand_bits);
        let mut bad = report(
            "batch_expand_into_evaluate_ntt",
            &gpu_eval.to_vec(),
            &cpu_eval.to_vec(),
            4,
        );

        let gpu_digests = gpu.alloc_digest("rows", domain);
        let cpu_digests = cpu.alloc_digest("rows", domain);
        gpu.hash_rows(&gpu_digests, &gpu_eval);
        cpu.hash_rows(&cpu_digests, &cpu_eval);
        bad += report(
            "hash_rows",
            &gpu_digests.to_vec(),
            &cpu_digests.to_vec(),
            32,
        );

        for idx in [0, domain / 2, domain - 1] {
            let gpu_col = gpu.alloc_elem("sample", count);
            let cpu_col = cpu.alloc_elem("sample", count);
            gpu.gather_sample(&gpu_col, &gpu_eval, idx, count, domain);
            cpu.gather_sample(&cpu_col, &cpu_eval, idx, count, domain);
            bad += report(
                &format!("gather_sample idx={idx}"),
                &gpu_col.to_vec(),
                &cpu_col.to_vec(),
                4,
            );
        }

        assert_eq!(bad, 0, "Intel HAL diverged from CPU on a >4 GiB buffer");
    }

    // NTT-family ops at the sizes the prover uses for po2=20/21 (coeffs at
    // lg = po2, the check polynomial at lg = po2 + 2), compared to the CPU HAL.
    #[test]
    #[ignore = "large: several GB of GPU memory"]
    fn ntt_ops_at_prover_sizes_match_cpu() {
        use crate::{
            core::hash::poseidon2::Poseidon2HashSuite,
            field::{baby_bear::BabyBearElem as Elem, Elem as _},
            hal::{cpu::CpuHal, Buffer as _, Hal as _},
        };

        let gpu = IntelHalPoseidon2::new();
        let cpu = CpuHal::new(Poseidon2HashSuite::new_suite());
        let mut rng = rand::rng();
        let mut bad_total = 0;
        let mut cases = vec![(20, 4), (21, 4), (21, 211), (22, 4), (23, 4)];
        // Small-po2 domains (lg = po2 + 2) at the commit-group widths.
        for lg in 14..=19 {
            cases.extend([(lg, 16), (lg, 103), (lg, 211)]);
        }
        for (lg, count) in cases {
            let n = count << lg;
            let input: Vec<Elem> = (0..n).map(|_| Elem::random(&mut rng)).collect();
            for op in [
                "interpolate",
                "interpolate_zk_shift",
                "bit_reverse",
                "zk_shift",
            ] {
                let g = gpu.copy_from_elem("io", &input);
                let c = cpu.copy_from_elem("io", &input);
                match op {
                    "interpolate" => {
                        gpu.batch_interpolate_ntt(&g, count);
                        cpu.batch_interpolate_ntt(&c, count);
                    }
                    "interpolate_zk_shift" => {
                        gpu.batch_interpolate_ntt_zk_shift(&g, count);
                        cpu.batch_interpolate_ntt_zk_shift(&c, count);
                    }
                    "bit_reverse" => {
                        gpu.batch_bit_reverse(&g, count);
                        cpu.batch_bit_reverse(&c, count);
                    }
                    _ => {
                        gpu.zk_shift(&g, count);
                        cpu.zk_shift(&c, count);
                    }
                }
                let (gv, cv) = (g.to_vec(), c.to_vec());
                let bad: Vec<usize> = (0..n).filter(|&i| gv[i] != cv[i]).collect();
                eprintln!(
                    "[ntt] lg={lg} count={count} {op}: {}/{n} differ; first={:?}",
                    bad.len(),
                    bad.first()
                );
                bad_total += bad.len();
            }
        }
        // Commit path: coefficients at 2^(lg-bits) expanded into the 2^lg
        // domain. Blowup x4 (bits=2) takes the fused expand+NTT pass for
        // lg 14..25 (production: lg 22 at po2=20, 23 at po2=21); other blowups
        // take expand + the regular forward NTT.
        let mut lde_cases = vec![];
        for lg in 14..=21 {
            for count in [16usize, 103, 211] {
                lde_cases.push((lg, count, 2usize));
            }
        }
        lde_cases.extend([
            (22, 16, 2),
            (22, 211, 2),
            (23, 16, 2),
            (24, 4, 2),
            (16, 16, 1),
            (20, 16, 1),
            (19, 16, 3),
            (22, 8, 3),
        ]);
        for (lg, count, bits) in lde_cases {
            let n_in = count << (lg - bits);
            let input: Vec<Elem> = (0..n_in).map(|_| Elem::random(&mut rng)).collect();
            let (gi, ci) = (
                gpu.copy_from_elem("in", &input),
                cpu.copy_from_elem("in", &input),
            );
            let go = gpu.alloc_elem("out", count << lg);
            let co = cpu.alloc_elem("out", count << lg);
            gpu.batch_expand_into_evaluate_ntt(&go, &gi, count, bits);
            cpu.batch_expand_into_evaluate_ntt(&co, &ci, count, bits);
            let (gv, cv) = (go.to_vec(), co.to_vec());
            let bad = gv.iter().zip(&cv).filter(|(a, b)| a != b).count();
            eprintln!("[ntt] lg={lg} count={count} bits={bits} expand_evaluate: {bad} differ");
            bad_total += bad;
        }
        assert_eq!(bad_total, 0, "Intel NTT-family ops diverged from CPU");
    }

    // Callers free or reuse host buffers as soon as an upload returns (scatter
    // injectors, view_mut write-back, set_32). The upload must not read host
    // memory after returning, even when the in-order queue is backed up.
    #[test]
    fn upload_survives_immediate_host_reuse() {
        use crate::{
            field::{baby_bear::BabyBearElem as Elem, Elem as _},
            hal::{Buffer as _, Hal as _},
        };

        let hal = IntelHalSha256::new();
        const BUSY_PO2: usize = 22;
        const BUSY_POLYS: usize = 16;
        let busy = hal.alloc_elem_init("busy", BUSY_POLYS << BUSY_PO2, Elem::ONE);

        const N: usize = 1 << 22;
        let expected: Vec<Elem> = (0..N as u32).map(Elem::new).collect();
        for round in 0..4 {
            // ~160 ms of queued GPU work so the upload cannot execute before
            // the host buffer is overwritten.
            for _ in 0..32 {
                hal.batch_interpolate_ntt(&busy, BUSY_POLYS);
            }
            let mut host = expected.clone();
            let dev = hal.copy_from_elem("upload", &host);
            host.fill(Elem::new(0xdead));
            drop(host);
            let back = dev.to_vec();
            let bad = back.iter().zip(&expected).filter(|(a, b)| a != b).count();
            assert_eq!(
                bad, 0,
                "round {round}: {bad}/{N} uploaded elements read after host reuse"
            );
        }
    }

    // NEO captures only the FIRST pageable copy from a given host address; a
    // second copy from the same address, enqueued before the host has seen the
    // first one complete, reads the buffer when the GPU runs it. A single upload
    // per buffer (as above) never hits that, so exercise repeated addresses: a
    // Vec refilled in place, a stack value uploaded twice from one frame, and
    // alloc_elem_init (set_32) twice at the same size. All behind queued work.
    #[test]
    fn upload_survives_same_address_reuse() {
        use crate::{
            field::{baby_bear::BabyBearElem as Elem, Elem as _},
            hal::{Buffer as _, Hal as _},
        };

        fn upload_one(hal: &IntelHalSha256, x: Elem) -> super::BufferImpl<Elem> {
            let v = [x];
            hal.copy_from_elem("one", &v)
        }

        fn check(round: usize, name: &str, got: Vec<Elem>, want: impl Fn(usize) -> Elem) {
            let wrong = got
                .iter()
                .enumerate()
                .filter(|(i, v)| **v != want(*i))
                .count();
            assert_eq!(wrong, 0, "round {round}: {name}: {wrong} elements wrong");
        }

        let hal = IntelHalSha256::new();
        const BUSY_PO2: usize = 22;
        const BUSY_POLYS: usize = 16;
        let busy = hal.alloc_elem_init("busy", BUSY_POLYS << BUSY_PO2, Elem::ONE);
        const N: usize = 1 << 16;
        for round in 0..3 {
            for _ in 0..32 {
                hal.batch_interpolate_ntt(&busy, BUSY_POLYS);
            }
            let a_vals: Vec<Elem> = (0..N as u32).map(Elem::new).collect();
            let b_vals: Vec<Elem> = (0..N as u32).map(|i| Elem::new(i + 7)).collect();
            let mut host = a_vals.clone();
            let a = hal.copy_from_elem("a", &host);
            host.copy_from_slice(&b_vals);
            let b = hal.copy_from_elem("b", &host);
            host.fill(Elem::new(0xdead));
            let s1 = upload_one(&hal, Elem::new(11));
            let s2 = upload_one(&hal, Elem::new(22));
            let f1 = hal.alloc_elem_init("f1", N, Elem::new(3));
            drop(f1);
            let f2 = hal.alloc_elem_init("f2", N, Elem::new(5));
            let f3 = hal.alloc_elem_init("f3", N, Elem::new(9));
            check(round, "vec 1st", a.to_vec(), |i| a_vals[i]);
            check(round, "vec 2nd", b.to_vec(), |i| b_vals[i]);
            check(round, "stack 1st", s1.to_vec(), |_| Elem::new(11));
            check(round, "stack 2nd", s2.to_vec(), |_| Elem::new(22));
            check(round, "set_32 1st", f2.to_vec(), |_| Elem::new(5));
            check(round, "set_32 2nd", f3.to_vec(), |_| Elem::new(9));
        }
    }

    // Out-of-place interpolate + zk_shift at every lg the prover can use,
    // including the copy-fallback sizes, compared to the CPU HAL; the source
    // must be left unchanged. (The fallback once used a device-to-device
    // q.memcpy that was not reliably ordered before the NTT kernels.)
    #[test]
    #[ignore = "large"]
    fn interpolate_zk_shift_from_matches_cpu() {
        use crate::{
            core::hash::poseidon2::Poseidon2HashSuite,
            field::{baby_bear::BabyBearElem as Elem, Elem as _},
            hal::{cpu::CpuHal, Buffer as _, Hal as _},
        };
        let gpu = IntelHalPoseidon2::new();
        let cpu = CpuHal::new(Poseidon2HashSuite::new_suite());
        let mut rng = rand::rng();
        let mut cases: Vec<(usize, usize)> = (14..=23usize)
            .map(|lg| (lg, if lg >= 22 { 3 } else { 5 }))
            .collect();
        cases.extend([(18, 103), (19, 103), (21, 33)]);
        let mut bad_total = 0;
        for (lg, count) in cases {
            let n = count << lg;
            let input: Vec<Elem> = (0..n).map(|_| Elem::random(&mut rng)).collect();
            let gs = gpu.copy_from_elem("src", &input);
            let cs = cpu.copy_from_elem("src", &input);
            let gd = gpu.alloc_elem_init("dst", n, Elem::new(7));
            let cd = cpu.alloc_elem("dst", n);
            gpu.batch_interpolate_ntt_zk_shift_from(&gd, &gs, count);
            cpu.batch_interpolate_ntt_zk_shift_from(&cd, &cs, count);
            let (gv, cv, sv) = (gd.to_vec(), cd.to_vec(), gs.to_vec());
            let bad = (0..n).filter(|&i| gv[i] != cv[i]).count();
            let src_bad = (0..n).filter(|&i| sv[i] != input[i]).count();
            if bad + src_bad > 0 {
                eprintln!("lg={lg} count={count}: {bad}/{n} differ, src modified {src_bad}");
            }
            bad_total += bad + src_bad;
        }
        assert_eq!(bad_total, 0);
    }
}
