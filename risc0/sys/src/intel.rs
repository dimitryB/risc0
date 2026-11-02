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

//! Raw SYCL/ESIMD FFI bindings for Intel GPU support.
//! Provides device memory management and kernel dispatch via the ESIMD kernel library.

use std::ffi::{c_void, CStr};
use std::os::raw::c_int;
use std::sync::Once;

// ============================================================================
// FFI declarations for the ESIMD kernel library (intel_ffi.cpp)
// ============================================================================

extern "C" {
    // Queue lifecycle
    pub fn esimd_create_queue() -> *mut c_void;
    pub fn esimd_device_total_memory() -> usize;
    pub fn esimd_destroy_queue(queue: *mut c_void);
    pub fn esimd_sync(queue: *mut c_void);

    // Device memory management
    pub fn esimd_malloc_device(queue: *mut c_void, bytes: usize) -> *mut c_void;
    pub fn esimd_free_device(queue: *mut c_void, ptr: *mut c_void);
    pub fn esimd_memcpy_htod(
        queue: *mut c_void,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    );
    pub fn esimd_memcpy_dtoh(
        queue: *mut c_void,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    );
    pub fn esimd_memset(queue: *mut c_void, ptr: *mut c_void, value: c_int, bytes: usize);
    pub fn esimd_memcpy_dtod(
        queue: *mut c_void,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    );

    // Warmup
    pub fn esimd_warmup(queue: *mut c_void, max_lg_n: u32);
    pub fn esimd_prepare_zk_shift(queue: *mut c_void, lg_n: u32) -> *const std::os::raw::c_char;

    // NTT
    pub fn esimd_forward_ntt(
        queue: *mut c_void,
        d_data: *mut c_void,
        lg_n: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_inverse_ntt(
        queue: *mut c_void,
        d_data: *mut c_void,
        lg_n: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_batch_forward_ntt(
        queue: *mut c_void,
        d_data: *mut c_void,
        lg_n: u32,
        poly_count: u32,
        stride: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_batch_inverse_ntt(
        queue: *mut c_void,
        d_data: *mut c_void,
        lg_n: u32,
        poly_count: u32,
        stride: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_batch_bit_reverse_ffi(
        queue: *mut c_void,
        d_io: *mut c_void,
        n_bits: u32,
        count: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_batch_expand(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_in: *const c_void,
        lg_domain_size: u32,
        lg_blowup: u32,
        poly_count: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_batch_expand_and_evaluate_ntt(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_in: *const c_void,
        lg_domain_size: u32,
        lg_blowup: u32,
        poly_count: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_batch_zk_shift(
        queue: *mut c_void,
        d_data: *mut c_void,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_batch_inverse_ntt_zk_shift(
        queue: *mut c_void,
        d_data: *mut c_void,
        lg_domain_size: u32,
        poly_count: u32,
        stride: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_batch_inverse_ntt_zk_shift_from(
        queue: *mut c_void,
        d_dst: *mut c_void,
        d_src: *const c_void,
        lg_domain_size: u32,
        poly_count: u32,
        stride: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_batch_expand_ffi(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_in: *const c_void,
        in_rows: u32,
        out_rows: u32,
        cols: u32,
        exp_po2: u32,
    ) -> *const std::os::raw::c_char;
    /// Fused x4 LDE (expand + forward NTT); see intel_ffi.cpp.
    pub fn esimd_batch_expand_fwd_fused(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_in: *const c_void,
        lg_out: u32,
        lg_blowup: u32,
        cols: u32,
    ) -> *const std::os::raw::c_char;

    // Poseidon2
    pub fn esimd_poseidon2_hash_rows(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_in: *const c_void,
        count: u32,
        col_size: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_poseidon2_hash_fold(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_in: *const c_void,
        num_hashes: u32,
    ) -> *const std::os::raw::c_char;

    // Poseidon-254 (BN254 field)
    pub fn esimd_poseidon254_hash_rows(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_in: *const c_void,
        row_size: u32,
        col_size: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_poseidon254_hash_fold(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_in: *const c_void,
        num_hashes: u32,
    ) -> *const std::os::raw::c_char;

    // Element-wise operations
    pub fn esimd_eltwise_add_fp_ffi(
        queue: *mut c_void,
        out: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        count: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_eltwise_copy_fp_ffi(
        queue: *mut c_void,
        out: *mut c_void,
        inp: *const c_void,
        count: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_eltwise_copy_fp_region_ffi(
        queue: *mut c_void,
        into: *mut c_void,
        from: *const c_void,
        from_rows: u32,
        from_cols: u32,
        from_offset: u32,
        from_stride: u32,
        into_offset: u32,
        into_stride: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_eltwise_sum_fpext_ffi(
        queue: *mut c_void,
        out: *mut c_void,
        inp: *const c_void,
        to_add: u32,
        count: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_eltwise_zeroize_fp_ffi(
        queue: *mut c_void,
        elems: *mut c_void,
        count: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_gather_sample_fp_ffi(
        queue: *mut c_void,
        dst: *mut c_void,
        src: *const c_void,
        idx: u32,
        size: u32,
        stride: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_scatter_fp_ffi(
        queue: *mut c_void,
        into: *mut c_void,
        index: *const c_void,
        offsets: *const c_void,
        values: *const c_void,
        count: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_gather_digests_ffi(
        queue: *mut c_void,
        dst: *mut c_void,
        src: *const c_void,
        indices: *const c_void,
        count: u32,
    ) -> *const std::os::raw::c_char;

    // FRI and polynomial operations
    pub fn esimd_fri_fold_ffi(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_in: *const c_void,
        d_mix: *const c_void,
        count: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_mix_poly_coeffs_ffi(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_in: *const c_void,
        d_combos: *const c_void,
        d_mix_start: *const c_void,
        d_mix: *const c_void,
        input_size: u32,
        count: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_batch_evaluate_any_ffi(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_coeffs: *const c_void,
        d_which: *const c_void,
        d_xs: *const c_void,
        count: u32,
        deg: u32,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_combos_prepare_ffi(
        queue: *mut c_void,
        d_combos: *mut c_void,
        d_coeff_u: *const c_void,
        combo_count: u32,
        cycles: u32,
        regs_count: u32,
        d_reg_sizes: *const c_void,
        d_reg_combo_ids: *const c_void,
        check_size: u32,
        d_mix: *const c_void,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_combos_divide_ffi(
        queue: *mut c_void,
        d_combos: *mut c_void,
        rows: u32,
        combos_cols: u32,
        d_info: *const c_void,
        info_count: u32,
        d_remainders: *mut c_void,
    ) -> *const std::os::raw::c_char;
    /// Device-side combos division (blocked synthetic division). `h_info`: host array of
    /// `info_count` x {col, z[4] raw Montgomery}; `h_rem`: host array of `info_count` x 4 u32.
    /// Synchronous; returns an error string if any remainder is nonzero.
    pub fn esimd_combos_divide_v2_ffi(
        queue: *mut c_void,
        d_combos: *mut c_void,
        rows: u32,
        ncols: u32,
        h_info: *const c_void,
        info_count: u32,
        h_rem: *mut c_void,
    ) -> *const std::os::raw::c_char;
    pub fn esimd_combos_finalize_ffi(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_combos: *const c_void,
        rows: u32,
        combos_cols: u32,
    ) -> *const std::os::raw::c_char;

    // GPU query (extract column + Merkle auth path on device)
    pub fn esimd_query_ffi(
        queue: *mut c_void,
        d_out: *mut c_void,
        d_data: *const c_void,
        d_tree: *const c_void,
        query_size: u32,
        rows: u32,
        cols: u32,
        idx: u32,
    ) -> *const std::os::raw::c_char;
}

// ============================================================================
// Singleton SYCL queue
// ============================================================================

static INIT: Once = Once::new();
static mut QUEUE: *mut c_void = std::ptr::null_mut();

// All GPU work goes through this one in-order queue. With a second
// concurrently executing queue (formerly used to overlap eval_check/finalize
// with the next segment's witgen), device data was intermittently corrupted on
// Arc Pro B70 (xe + NEO 26.09), surfacing as GPU page faults
// (UR_RESULT_ERROR_DEVICE_LOST; faulting writes consistent with the injector
// scatter reading stale indices) or as invalid proofs. The single queue, plus
// the process-wide buffer pool in risc0-zkp's Intel HAL (which depends on it
// for cross-thread reuse ordering), removed the failure at no measured
// throughput cost.

fn create_queue_once() -> *mut c_void {
    unsafe {
        INIT.call_once(|| {
            QUEUE = esimd_create_queue();
            if QUEUE.is_null() {
                panic!("Intel GPU: esimd_create_queue() returned null -- no Intel GPU found");
            }
        });
        QUEUE
    }
}

static BG_PENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static BG_INIT: std::sync::Mutex<Option<std::thread::JoinHandle<()>>> = std::sync::Mutex::new(None);
static BG_JOIN: std::sync::Mutex<()> = std::sync::Mutex::new(());
static WARMUP: Once = Once::new();

/// Total memory of the Intel GPU the singleton queue runs on, in bytes; 0 before [get_queue]
/// has created the queue.
pub fn device_total_memory() -> usize {
    unsafe { esimd_device_total_memory() }
}

/// Get the singleton SYCL queue, creating it on first call.
/// Panics if no Intel GPU is found. If [start_background_init] is running,
/// waits for it first, so no other GPU work overlaps the warmup.
pub fn get_queue() -> *mut c_void {
    if BG_PENDING.load(std::sync::atomic::Ordering::Acquire) {
        // Every caller other than the background thread waits here until the
        // init has been joined. Without this lock, a second thread arriving
        // while the first was inside join() found no handle and went ahead,
        // overlapping the warmup. (The background thread itself never calls
        // get_queue; the thread-id check below guards against a deadlock.)
        let _join = BG_JOIN.lock().unwrap();
        if BG_PENDING.load(std::sync::atomic::Ordering::Acquire) {
            let handle = BG_INIT.lock().unwrap().take();
            if let Some(h) = handle {
                if h.thread().id() == std::thread::current().id() {
                    *BG_INIT.lock().unwrap() = Some(h);
                } else {
                    let _ = h.join();
                    BG_PENDING.store(false, std::sync::atomic::Ordering::Release);
                }
            }
        }
    }
    create_queue_once()
}

/// Run the one-time kernel warmup (twiddles, JIT of the hot kernels) on `queue`.
pub fn warmup_once(queue: *mut c_void) {
    WARMUP.call_once(|| unsafe { esimd_warmup(queue, 22) });
}

/// Create the queue, run the warmup and build the zk_shift tables for
/// `lgs` on a background thread (e.g. while the guest executes and segment
/// 0 is preflighted). The first [get_queue] call from another thread joins
/// it. Same work and order as the lazy path, just earlier.
pub fn start_background_init(lgs: Vec<u32>) {
    let mut slot = BG_INIT.lock().unwrap();
    if slot.is_some() || INIT.is_completed() {
        return;
    }
    BG_PENDING.store(true, std::sync::atomic::Ordering::Release);
    *slot = Some(std::thread::spawn(move || {
        let queue = create_queue_once();
        warmup_once(queue);
        for lg in lgs {
            esimd_check(unsafe { esimd_prepare_zk_shift(queue, lg) });
        }
    }));
}

/// Synchronize the SYCL queue (wait for all submitted work to complete).
/// WARNING: This blocks the CPU.
pub fn sync() {
    unsafe { esimd_sync(get_queue()) };
}

// ============================================================================
// FFI error checking
// ============================================================================

/// Check an ESIMD FFI return code (null = success, non-null = error string).
/// The error string is allocated with strdup() on the C++ side; we free it here.
#[inline]
pub fn esimd_check(err: *const std::os::raw::c_char) {
    if !err.is_null() {
        let msg = unsafe {
            let s = CStr::from_ptr(err)
                .to_str()
                .unwrap_or("invalid error string")
                .to_string();
            extern "C" {
                fn free(ptr: *const std::os::raw::c_char);
            }
            free(err);
            s
        };
        panic!("Intel GPU error: {msg}");
    }
}

// ============================================================================
// Device pointer wrapper (matches cuda.rs DevicePointer for FFI compatibility)
// ============================================================================

/// Opaque device pointer for Intel GPU memory.
/// Compatible with the DevicePointer<T> used in cuda.rs / hip.rs FFI.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct DevicePointer<T>(pub *mut T);

impl<T> DevicePointer<T> {
    /// Get the raw pointer value.
    pub fn as_ptr(&self) -> *const T {
        self.0 as *const T
    }

    /// Offset the pointer by `count` bytes (matching cust's byte-offset semantics).
    pub unsafe fn offset(&self, count: isize) -> DevicePointer<T> {
        DevicePointer((self.0 as *mut u8).offset(count) as *mut T)
    }
}

// ============================================================================
// RAII device buffer
// ============================================================================

/// RAII wrapper for Intel GPU device memory (SYCL USM device allocation).
pub struct IntelDeviceBuffer {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for IntelDeviceBuffer {}

impl IntelDeviceBuffer {
    /// Allocate uninitialized device memory.
    pub fn uninitialized(size: usize) -> anyhow::Result<Self> {
        let queue = get_queue();
        let ptr = unsafe { esimd_malloc_device(queue, size) };
        if ptr.is_null() {
            anyhow::bail!("esimd_malloc_device failed for {size} bytes");
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            len: size,
        })
    }

    /// Copy host data to device.
    pub fn copy_from(&mut self, data: &[u8]) -> anyhow::Result<()> {
        assert!(
            data.len() <= self.len,
            "copy_from: data ({}) exceeds buffer ({})",
            data.len(),
            self.len
        );
        let queue = get_queue();
        unsafe {
            esimd_memcpy_htod(
                queue,
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len(),
            );
        }
        Ok(())
    }

    /// Copy device data to a new host Vec.
    pub fn as_host_vec(&self) -> anyhow::Result<Vec<u8>> {
        let mut vec = vec![0u8; self.len];
        let queue = get_queue();
        unsafe {
            esimd_memcpy_dtoh(
                queue,
                vec.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                self.len,
            );
        }
        Ok(vec)
    }

    /// Copy a range of device data to a new host Vec.
    /// `offset` and `len` are in bytes.
    pub fn as_host_vec_range(&self, offset: usize, len: usize) -> anyhow::Result<Vec<u8>> {
        assert!(
            offset + len <= self.len,
            "range [{offset}..{}] exceeds buffer size {}",
            offset + len,
            self.len
        );
        let mut vec = vec![0u8; len];
        let queue = get_queue();
        unsafe {
            esimd_memcpy_dtoh(
                queue,
                vec.as_mut_ptr() as *mut c_void,
                self.ptr.add(offset) as *const c_void,
                len,
            );
        }
        Ok(vec)
    }

    /// Set all bytes to `value`.
    pub fn memset(&mut self, value: i32, size: usize) {
        let queue = get_queue();
        unsafe { esimd_memset(queue, self.ptr as *mut c_void, value, size) };
    }

    /// Set all 32-bit words to `value`. Equivalent to HipDeviceBuffer's set_32().
    pub fn set_32(&mut self, value: u32) {
        if value == 0 {
            let queue = get_queue();
            unsafe { esimd_memset(queue, self.ptr as *mut c_void, 0, self.len) };
        } else {
            // For non-zero: upload a host buffer filled with the value.
            // SYCL doesn't have a memsetD32 equivalent, so we do it manually.
            let count = self.len / 4;
            let host_buf: Vec<u32> = vec![value; count];
            let queue = get_queue();
            unsafe {
                esimd_memcpy_htod(
                    queue,
                    self.ptr as *mut c_void,
                    host_buf.as_ptr() as *const c_void,
                    self.len,
                );
            }
        }
    }

    /// Get the device pointer.
    pub fn as_device_ptr(&self) -> DevicePointer<u8> {
        DevicePointer(self.ptr)
    }

    /// Get the buffer length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the buffer has zero length.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for IntelDeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let queue = get_queue();
            unsafe { esimd_free_device(queue, self.ptr as *mut c_void) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
