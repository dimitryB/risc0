// Page-locked staging of the recursion witgen inputs, filled on the prepare thread.
//
// The GPU-thread uploads of the preflight trace (wom, cycles, iops) and the host-chain rows are
// pageable cudaMemcpys: ~3.5 ms per join after the last segment and ~13 ms per join between
// segments, where the driver's bounce-buffer copies compete with the next segment's host work.
// From page-locked memory they run at full PCIe rate. Buffers come from a small process-wide pool
// and go back to it when the preflight that owns them is dropped (after its witgen call, which
// synchronizes, so no copy from them can still be in flight). Buffers beyond the pool's capacity
// are only unpinned by `release` (a HAL release hook, e.g. clear_buffer_pool), not where they are
// dropped: unpinning waits for the whole device while holding a driver lock, which would stall
// the proving thread mid-segment (see ThpPinned::free_pinned).

use std::sync::Mutex;

use risc0_circuit_recursion_sys::{
    risc0_circuit_recursion_cuda_host_alloc, risc0_circuit_recursion_cuda_host_free,
    RawPreflightCycle,
};
use risc0_zkp::field::baby_bear::BabyBearExtElem;

pub(crate) struct Pinned {
    ptr: usize,
    cap: usize,
}

static POOL: Mutex<Vec<Pinned>> = Mutex::new(Vec::new());
const POOL_CAP: usize = 6;
/// Buffers dropped while the pool was full, unpinned by [release].
static RETIRED: Mutex<Vec<usize>> = Mutex::new(Vec::new());
const MAX_RETIRED: usize = 4;

fn host_free(ptr: usize) {
    let p = ptr as *mut std::ffi::c_void;
    let _ = risc0_sys::ffi_wrap(|| unsafe { risc0_circuit_recursion_cuda_host_free(p) });
}

/// HAL release hook (`risc0_zkp::hal::cuda::register_release_hook`, or `hal::hip`'s): unless
/// `device_only`, unpin the pooled and retired buffers (re-pinned on demand). Never blocks on the
/// pool's lock.
pub(crate) fn release(device_only: bool) {
    if device_only {
        return;
    }
    let mut free: Vec<usize> = Vec::new();
    if let Ok(mut pool) = POOL.try_lock() {
        for mut b in pool.drain(..) {
            free.push(std::mem::take(&mut b.ptr));
        }
    }
    if let Ok(mut retired) = RETIRED.try_lock() {
        free.append(&mut retired);
    }
    free.into_iter().for_each(host_free);
}

impl Pinned {
    fn take(bytes: usize) -> Option<Pinned> {
        {
            let mut pool = POOL.lock().unwrap();
            if let Some(i) = pool
                .iter()
                .enumerate()
                .filter(|(_, b)| b.cap >= bytes)
                .min_by_key(|(_, b)| b.cap)
                .map(|(i, _)| i)
            {
                return Some(pool.swap_remove(i));
            }
        }
        // Headroom so the next job of a similar size reuses it.
        let cap = (bytes + bytes / 4 + (1 << 20)) & !((1 << 20) - 1);
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        risc0_sys::ffi_wrap(|| unsafe { risc0_circuit_recursion_cuda_host_alloc(cap, &mut p) })
            .ok()?;
        Some(Pinned {
            ptr: p as usize,
            cap,
        })
    }

    fn base(&self) -> *mut u8 {
        self.ptr as *mut u8
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        let mut evicted = self.ptr;
        {
            let mut pool = POOL.lock().unwrap();
            if pool.len() < POOL_CAP {
                pool.push(Pinned {
                    ptr: self.ptr,
                    cap: self.cap,
                });
                evicted = 0;
            } else if let Some(small) = pool
                .iter_mut()
                .filter(|b| b.cap < self.cap)
                .min_by_key(|b| b.cap)
            {
                // Keep the larger buffer: later jobs of this size then find one.
                evicted = std::mem::replace(&mut small.ptr, self.ptr);
                small.cap = self.cap;
            }
        }
        self.ptr = 0;
        if evicted != 0 {
            let mut overflow = 0;
            {
                let mut retired = RETIRED.lock().unwrap();
                retired.push(evicted);
                // Bounded (rare: only while job sizes keep growing).
                if retired.len() > MAX_RETIRED {
                    overflow = retired.remove(0);
                }
            }
            if overflow != 0 {
                host_free(overflow);
            }
        }
    }
}

/// Byte offsets of each array in the buffer.
pub(crate) struct PinnedTrace {
    buf: Pinned,
    off: [usize; 8],
    /// ROCm: handed to the witgen upload-ahead (`stage`); forgotten there before `buf` returns
    /// to the pool.
    #[cfg(feature = "rocm")]
    staged: bool,
}

#[cfg(feature = "rocm")]
impl Drop for PinnedTrace {
    fn drop(&mut self) {
        if self.staged {
            let wom = self.ptr::<u8>(0) as *const std::ffi::c_void;
            let _ = risc0_sys::ffi_wrap(|| unsafe {
                risc0_circuit_recursion_sys::risc0_circuit_recursion_hip_unstage(wom)
            });
        }
    }
}

fn copy_par(dst: *mut u8, src: &[u8]) {
    use rayon::prelude::*;
    const CH: usize = 1 << 20;
    let dst = dst as usize;
    src.par_chunks(CH).enumerate().for_each(|(i, c)| unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), (dst + i * CH) as *mut u8, c.len());
    });
}

impl PinnedTrace {
    /// wom, cycles, iops, then (if any) chain heads, cycles, rows, wom_idx, wom_rows.
    pub(crate) fn new(
        wom: &[BabyBearExtElem],
        cycles: &[RawPreflightCycle],
        iops: &[BabyBearExtElem],
        chains: Option<&super::witgen::HostChains>,
    ) -> Option<Self> {
        let parts: [&[u8]; 8] = {
            let e: &[u32] = &[];
            let (h, c, r, wi, wr) = match chains {
                Some(hc) => (
                    &hc.heads[..],
                    &hc.cycles[..],
                    &hc.rows[..],
                    &hc.wom_idx[..],
                    &hc.wom_rows[..],
                ),
                None => (e, e, e, e, e),
            };
            unsafe {
                [
                    std::slice::from_raw_parts(
                        wom.as_ptr() as *const u8,
                        std::mem::size_of_val(wom),
                    ),
                    std::slice::from_raw_parts(
                        cycles.as_ptr() as *const u8,
                        std::mem::size_of_val(cycles),
                    ),
                    std::slice::from_raw_parts(
                        iops.as_ptr() as *const u8,
                        std::mem::size_of_val(iops),
                    ),
                    bytemuck::cast_slice(h),
                    bytemuck::cast_slice(c),
                    bytemuck::cast_slice(r),
                    bytemuck::cast_slice(wi),
                    bytemuck::cast_slice(wr),
                ]
            }
        };
        let mut off = [0usize; 8];
        let mut total = 0usize;
        for (i, p) in parts.iter().enumerate() {
            off[i] = total;
            total += (p.len() + 63) & !63;
        }
        let buf = Pinned::take(total.max(64))?;
        for (i, p) in parts.iter().enumerate() {
            if !p.is_empty() {
                copy_par(unsafe { buf.base().add(off[i]) }, p);
            }
        }
        Some(PinnedTrace {
            buf,
            off,
            #[cfg(feature = "rocm")]
            staged: false,
        })
    }

    /// ROCm: start uploading these inputs to the device for the witgen of this preflight (the
    /// sizes are the witgen's: the same arrays, and as many rows as it reads), while the GPU
    /// proves the previous proof. Best effort: unstaged, the witgen uploads them itself.
    #[cfg(feature = "rocm")]
    pub(crate) fn stage(
        &mut self,
        num_woms: u32,
        num_cycles: u32,
        num_iops: u32,
        chains: Option<&super::witgen::HostChains>,
    ) {
        use risc0_circuit_recursion_sys::{RawHostChains, RawPreflightTrace};
        let trace = RawPreflightTrace {
            wom: self.ptr(0),
            cycles: self.ptr(1),
            iops: self.ptr(2),
            num_woms,
            num_cycles,
            num_iops,
        };
        let raw = chains.filter(|hc| !hc.heads.is_empty()).map(|hc| RawHostChains {
            heads: self.ptr(3),
            num_heads: hc.heads.len() as u32,
            cycles: self.ptr(4),
            rows: self.ptr(5),
            wom_idx: self.ptr(6),
            wom_rows: self.ptr(7),
            num_rows: hc.cycles.len() as u32,
            data_cols: (hc.rows.len() / hc.cycles.len().max(1)) as u32,
        });
        let hc = raw.as_ref().map_or(std::ptr::null(), |r| r as *const RawHostChains);
        // Set first (unstage is a no-op for inputs that were not staged: a failed call waits
        // for its own copies).
        self.staged = true;
        let _ = risc0_sys::ffi_wrap(|| unsafe {
            risc0_circuit_recursion_sys::risc0_circuit_recursion_hip_stage(&trace, hc)
        });
    }

    pub(crate) fn ptr<T>(&self, i: usize) -> *const T {
        unsafe { self.buf.base().add(self.off[i]) as *const T }
    }
}
