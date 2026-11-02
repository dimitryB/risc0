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

#[cfg(feature = "cuda")]
use cust::memory::DevicePointer;

/// ABI-compatible substitute for cust::memory::DevicePointer when using ROCm.
/// HIP device pointers are native `*mut T` (vs CUDA's CUdeviceptr which is u64),
/// but both are 8 bytes on 64-bit and pass identically in FFI.
#[cfg(feature = "rocm")]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct DevicePointer<T>(pub *mut T);

#[cfg(feature = "rocm")]
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

pub use sppark::Error as SpparkError;

extern "C" {
    /// Registers the function that raw device allocations in the kernel libraries call when
    /// cudaMalloc runs out of memory, before retrying once (see `device_alloc.h`).
    pub fn risc0_zkp_cuda_set_oom_hook(hook: extern "C" fn());

    pub fn sppark_init() -> sppark::Error;
    pub fn sppark_poseidon2_init() -> sppark::Error;

    pub fn sppark_batch_expand(
        d_out: DevicePointer<u8>,
        d_in: DevicePointer<u8>,
        lg_domain_size: u32,
        lg_blowup: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_batch_NTT(
        d_inout: DevicePointer<u8>,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_batch_iNTT(
        d_inout: DevicePointer<u8>,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_batch_zk_shift(
        d_inout: DevicePointer<u8>,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_batch_iNTT_zk_shift(
        d_inout: DevicePointer<u8>,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_batch_iNTT_zk_shift_fused(
        d_inout: DevicePointer<u8>,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_batch_bit_reverse_tiled(
        d_inout: DevicePointer<u8>,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_commit_coeffs_lde(
        d_coeffs: DevicePointer<u8>,
        d_evaluated: DevicePointer<u8>,
        d_witness: DevicePointer<u8>,
        lg_domain_size: u32,
        lg_blowup: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_batch_bit_reverse(
        d_inout: DevicePointer<u8>,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_batch_expand_NTT(
        d_out: DevicePointer<u8>,
        d_in: DevicePointer<u8>,
        lg_domain_size: u32,
        lg_blowup: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_poseidon2_fold(
        d_out: DevicePointer<u8>,
        d_in: DevicePointer<u8>,
        num_hashes: usize,
    ) -> sppark::Error;

    pub fn sppark_poseidon2_fold_tree(nodes: DevicePointer<u8>, layers: u32) -> sppark::Error;

    pub fn sppark_poseidon2_rows(
        d_out: DevicePointer<u8>,
        d_in: DevicePointer<u8>,
        count: u32,
        col_size: u32,
    ) -> sppark::Error;

    pub fn sppark_poseidon254_fold(
        d_out: DevicePointer<u8>,
        d_in: DevicePointer<u8>,
        num_hashes: usize,
    ) -> sppark::Error;

    pub fn sppark_poseidon254_fold_tree(nodes: DevicePointer<u8>, layers: u32) -> sppark::Error;

    pub fn sppark_poseidon254_rows(
        d_out: DevicePointer<u8>,
        d_in: DevicePointer<u8>,
        count: usize,
        col_size: u32,
    ) -> sppark::Error;

    pub fn supra_poly_divide(
        polynomial: DevicePointer<u8>,
        poly_size: usize,
        remainder: *mut u32,
        pow: *const u32,
    ) -> sppark::Error;

    pub fn supra_poly_divide_batch(
        polynomial: DevicePointer<u8>,
        poly_size: usize,
        remainders: *mut u32,
        pows: *const u32,
        num_divides: u32,
    ) -> sppark::Error;

    pub fn supra_poly_divide_multi(
        combos_base: DevicePointer<u8>,
        stride: usize,
        combo_indices: *const u32,
        pows_per_combo: *const u32,
        all_pows: *const u32,
        num_combos: u32,
        remainders: *mut u32,
    ) -> sppark::Error;
}
