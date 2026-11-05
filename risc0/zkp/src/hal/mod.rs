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

//! Hardware Abstraction Layer (HAL) for accelerating the ZKP system.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dual;
#[cfg(feature = "rocm")]
pub mod hip;
#[cfg(feature = "intel")]
pub mod intel;
#[cfg(all(
    feature = "metal",
    any(all(target_os = "macos", target_arch = "aarch64"), target_os = "ios")
))]
pub mod metal;

use std::{
    fmt::Debug,
    sync::{Mutex, OnceLock},
};

use risc0_core::{
    field::{Elem, ExtElem, Field, RootsOfUnity},
    scope,
};

use crate::{
    core::{digest::Digest, hash::HashSuite, poly::poly_divide},
    INV_RATE,
};

pub trait Buffer<T>: Clone {
    fn name(&self) -> &'static str;

    fn size(&self) -> usize;

    fn slice(&self, offset: usize, size: usize) -> Self;

    fn get_at(&self, idx: usize) -> T;

    fn view<F: FnOnce(&[T])>(&self, f: F);

    fn view_mut<F: FnOnce(&mut [T])>(&self, f: F);

    fn to_vec(&self) -> Vec<T>;

    /// Opt-in hint: when all clones of this Buffer are dropped, release the
    /// underlying device memory directly to the driver instead of caching it
    /// in a reuse pool. HALs without a reuse pool can leave this as the
    /// default no-op.
    fn set_bypass_pool(&self) {}
}

/// One Merkle tree's query gathers for [Hal::gather_multi].
pub struct GatherJob<'a, H: Hal> {
    pub matrix: &'a H::Buffer<H::Elem>,
    pub col_size: usize,
    pub row_size: usize,
    pub rows: &'a [usize],
    pub nodes: &'a H::Buffer<Digest>,
    pub node_indices: &'a [usize],
}

pub trait Hal {
    /// Delimit dispatches that do not require an intervening CPU read.
    /// Backends without command batching execute the closure directly.
    fn with_gpu_stage<R>(&self, _label: &'static str, f: impl FnOnce() -> R) -> R
    where
        Self: Sized,
    {
        f()
    }
    type Field: Field<Elem = Self::Elem, ExtElem = Self::ExtElem>;
    type Elem: Elem + RootsOfUnity;
    type ExtElem: ExtElem<SubElem = Self::Elem>;
    type Buffer<T: Clone + Debug + PartialEq>: Buffer<T>;

    const CHECK_SIZE: usize = INV_RATE * Self::ExtElem::EXT_SIZE;

    fn has_unified_memory(&self) -> bool;

    /// Ask the HAL to return any cached / stream-ordered mempool memory back
    /// to the driver. Called at GPU-memory-pressure points (e.g. after a host
    /// spill at po2>=22) so the next large allocation can succeed.
    /// Default no-op for HALs without a reuse cache.
    fn trim_device_memory(&self) {}

    /// Returns the amount of free GPU memory in bytes.
    /// Returns `usize::MAX` for CPU HALs or when the information is unavailable.
    fn gpu_free_memory(&self) -> usize {
        usize::MAX
    }

    /// Returns the total GPU memory in bytes.
    /// Returns `usize::MAX` for CPU HALs or when the information is unavailable.
    fn gpu_total_memory(&self) -> usize {
        usize::MAX
    }

    fn get_hash_suite(&self) -> &HashSuite<Self::Field>;

    fn alloc_digest(&self, name: &'static str, size: usize) -> Self::Buffer<Digest>;
    fn alloc_elem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::Elem>;
    fn alloc_extelem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem>;
    fn alloc_u32(&self, name: &'static str, size: usize) -> Self::Buffer<u32>;

    fn alloc_elem_init(
        &self,
        name: &'static str,
        size: usize,
        value: Self::Elem,
    ) -> Self::Buffer<Self::Elem> {
        let buffer = self.alloc_elem(name, size);
        buffer.view_mut(|slice| {
            slice.fill(value);
        });
        buffer
    }

    /// Hint that `alloc_elem_init` will soon be called with this `size` and `value` (e.g. by the
    /// next segment): a HAL may fill a buffer it holds in the background now and hand it out
    /// then. The default does nothing.
    fn prefill_elem(&self, _size: usize, _value: Self::Elem) {}

    fn alloc_extelem_zeroed(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        let buffer = self.alloc_extelem(name, size);
        buffer.view_mut(|slice| {
            slice.fill(Self::ExtElem::ZERO);
        });
        buffer
    }

    fn copy_from_digest(&self, name: &'static str, slice: &[Digest]) -> Self::Buffer<Digest>;
    fn copy_from_elem(&self, name: &'static str, slice: &[Self::Elem]) -> Self::Buffer<Self::Elem>;
    fn copy_from_extelem(
        &self,
        name: &'static str,
        slice: &[Self::ExtElem],
    ) -> Self::Buffer<Self::ExtElem>;
    fn copy_from_u32(&self, name: &'static str, slice: &[u32]) -> Self::Buffer<u32>;

    fn batch_expand_into_evaluate_ntt(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        count: usize,
        expand_bits: usize,
    );

    fn batch_interpolate_ntt(&self, io: &Self::Buffer<Self::Elem>, count: usize);

    fn batch_bit_reverse(&self, io: &Self::Buffer<Self::Elem>, count: usize);

    fn batch_evaluate_any(
        &self,
        coeffs: &Self::Buffer<Self::Elem>,
        poly_count: usize,
        which: &Self::Buffer<u32>,
        xs: &Self::Buffer<Self::ExtElem>,
        out: &Self::Buffer<Self::ExtElem>,
    );

    fn zk_shift(&self, io: &Self::Buffer<Self::Elem>, count: usize);

    fn batch_interpolate_ntt_zk_shift(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        self.batch_interpolate_ntt(io, count);
        self.zk_shift(io, count);
    }

    /// True if `commit_coeffs_lde` is implemented for columns of `2^po2` rows.
    fn has_commit_coeffs_lde(&self, _po2: usize) -> bool {
        false
    }

    /// Fused commit: `coeffs` = the natural-order zk-shifted coefficients of `witness`
    /// (what `make_coeffs` + `PolyGroup::new`'s bit-reverse produce) and `evaluated` = the
    /// INV_RATE LDE (what `batch_expand_into_evaluate_ntt` produces). `witness` is unchanged.
    fn commit_coeffs_lde(
        &self,
        _coeffs: &Self::Buffer<Self::Elem>,
        _evaluated: &Self::Buffer<Self::Elem>,
        _witness: &Self::Buffer<Self::Elem>,
        _count: usize,
    ) {
        unimplemented!()
    }

    /// `dst = interpolate_zk_shift(src)`, leaving `src` unchanged.
    fn batch_interpolate_ntt_zk_shift_from(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        count: usize,
    ) {
        self.eltwise_copy_elem(dst, src);
        self.batch_interpolate_ntt_zk_shift(dst, count);
    }

    #[allow(clippy::too_many_arguments)]
    fn mix_poly_coeffs(
        &self,
        out: &Self::Buffer<Self::ExtElem>,
        mix_start: &Self::ExtElem,
        mix: &Self::ExtElem,
        input: &Self::Buffer<Self::Elem>,
        combos: &Self::Buffer<u32>,
        input_size: usize,
        count: usize,
    );

    fn eltwise_add_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input1: &Self::Buffer<Self::Elem>,
        input2: &Self::Buffer<Self::Elem>,
    );

    fn eltwise_sum_extelem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::ExtElem>,
    );

    fn eltwise_copy_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
    );

    #[allow(clippy::too_many_arguments)]
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
    );

    fn eltwise_zeroize_elem(&self, elems: &Self::Buffer<Self::Elem>);

    fn fri_fold(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        mix: &Self::ExtElem,
    );

    fn hash_rows(&self, output: &Self::Buffer<Digest>, matrix: &Self::Buffer<Self::Elem>);

    fn hash_fold(&self, io: &Self::Buffer<Digest>, input_size: usize, output_size: usize);

    fn hash_fold_tree(&self, io: &Self::Buffer<Digest>, layers: usize) {
        for i in (0..layers).rev() {
            let layer_size = 1 << i;
            self.hash_fold(io, layer_size * 2, layer_size);
        }
    }

    /// Hash rows and all Merkle levels before the caller reads the root.
    fn hash_tree(
        &self,
        nodes: &Self::Buffer<Digest>,
        matrix: &Self::Buffer<Self::Elem>,
        rows: usize,
        layers: usize,
    ) {
        self.hash_rows(&nodes.slice(rows, rows), matrix);
        self.hash_fold_tree(nodes, layers);
    }

    fn gather_sample(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        idx: usize,
        size: usize,
        stride: usize,
    );

    /// Gather `size` strided samples for each index in `indices` into consecutive slices of
    /// `dst` (`dst[q * size + i] = src[idxs[q] + i * stride]`). Default: one gather_sample per
    /// index. CUDA overrides this with a single launch.
    fn gather_samples(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        indices: &[usize],
        size: usize,
        stride: usize,
    ) {
        for (q, &idx) in indices.iter().enumerate() {
            self.gather_sample(&dst.slice(q * size, size), src, idx, size, stride);
        }
    }

    /// Batch-read digests at scattered indices. Default impl uses individual get_at() calls.
    /// CUDA overrides this with a gather kernel + single D2H transfer.
    fn batch_get_digest_at(&self, buf: &Self::Buffer<Digest>, indices: &[usize]) -> Vec<Digest> {
        indices.iter().map(|&idx| buf.get_at(idx)).collect()
    }

    /// The Merkle query gathers of several trees in one host round trip, where the HAL supports
    /// it (`None` otherwise): for each job, the `rows.len() * col_size` elements
    /// `matrix[row + i * row_size]` (query-major, as `gather_samples`) and the digests at
    /// `node_indices` (as `batch_get_digest_at`).
    #[allow(clippy::type_complexity)]
    fn gather_multi(
        &self,
        _jobs: &[GatherJob<'_, Self>],
    ) -> Option<Vec<(Vec<Self::Elem>, Vec<Digest>)>>
    where
        Self: Sized,
    {
        None
    }

    fn scatter(
        &self,
        into: &Self::Buffer<Self::Elem>,
        index: &[u32],
        offsets: &[u32],
        values: &[Self::Elem],
    );

    /// Scatter u32 values as 32-bit decompositions into the data buffer.
    /// `bit_data` contains packed (row, base_col, value) triples. Each triple
    /// writes 32 field elements: data[(base_col+i)*cycles + row] = (value>>i)&1.
    fn scatter_bits(&self, into: &Self::Buffer<Self::Elem>, bit_data: &[u32], cycles: u32) {
        let n = bit_data.len() / 3;
        if n == 0 {
            return;
        }
        let mut index = Vec::with_capacity(n + 1);
        let mut offsets = Vec::with_capacity(n * 32);
        let mut vals: Vec<Self::Elem> = Vec::with_capacity(n * 32);
        for i in 0..n {
            index.push((i * 32) as u32);
            let row = bit_data[i * 3];
            let base = bit_data[i * 3 + 1];
            let value = bit_data[i * 3 + 2];
            for bit in 0..32u32 {
                offsets.push((base + bit) * cycles + row);
                vals.push(Self::Elem::from_u64(((value >> bit) & 1) as u64));
            }
        }
        index.push((n * 32) as u32);
        self.scatter(into, &index, &offsets, &vals);
    }

    fn prefix_products(&self, io: &Self::Buffer<Self::ExtElem>);

    #[allow(clippy::too_many_arguments)]
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
        combos.view_mut(|combos| {
            scope!("combos_prepare", {
                let mut cur_pos = 0;
                let mut cur = Self::ExtElem::ONE;
                // Subtract the U coeffs from the combos
                for (reg_size, reg_combo_id) in reg_sizes.iter().zip(reg_combo_ids) {
                    let reg_size = *reg_size as usize;
                    let reg_combo_id = *reg_combo_id as usize;
                    for i in 0..reg_size {
                        combos[cycles * reg_combo_id + i] -= cur * coeff_u[cur_pos + i];
                    }
                    cur *= *mix;
                    cur_pos += reg_size;
                }
                // Subtract the final 'check' coefficients
                for _ in 0..Self::CHECK_SIZE {
                    combos[cycles * combo_count] -= cur * coeff_u[cur_pos];
                    cur_pos += 1;
                    cur *= *mix;
                }
            });
        });
    }

    fn combos_divide(
        &self,
        combos: &Self::Buffer<Self::ExtElem>,
        chunks: Vec<(usize, Vec<Self::ExtElem>)>,
        cycles: usize,
    ) {
        use rayon::prelude::*;

        scope!("combos_divide");

        combos.view_mut(|combos| {
            combos
                .par_chunks_exact_mut(cycles)
                .zip(chunks)
                .for_each(|(combo_slice, (i, pows))| {
                    for pow in pows {
                        let remainder = poly_divide(combo_slice, pow);
                        assert_eq!(remainder, Self::ExtElem::ZERO, "i: {i}");
                    }
                });
        });
    }
}

#[derive(Clone, Default)]
pub struct AccumPreflight {
    pub is_par_safe: Vec<u8>,
}

pub trait CircuitHal<H: Hal> {
    #[allow(clippy::too_many_arguments)]
    fn accumulate(
        &self,
        preflight: &AccumPreflight,
        ctrl: &H::Buffer<H::Elem>,
        io: &H::Buffer<H::Elem>,
        data: &H::Buffer<H::Elem>,
        mix: &H::Buffer<H::Elem>,
        accum: &H::Buffer<H::Elem>,
        steps: usize,
    );

    /// Compute check polynomial.
    fn eval_check(
        &self,
        check: &H::Buffer<H::Elem>,
        // Register groups, e.g. accum, code, data.  These should have one row for each cycle.
        groups: &[&H::Buffer<H::Elem>],
        // Globals.  These should have one row total.
        globals: &[&H::Buffer<H::Elem>],
        poly_mix: H::ExtElem,
        po2: usize,
        steps: usize,
    );

    /// Insert a GPU-side dependency so that subsequent operations on the main
    /// stream wait for eval_check (running on a separate stream) to complete.
    /// Default is no-op (for CPU HAL or when eval_check uses the same stream).
    fn eval_check_dep(&self) {}
}

pub fn tracker() -> &'static Mutex<MemoryTracker> {
    static ONCE: OnceLock<Mutex<MemoryTracker>> = OnceLock::new();
    ONCE.get_or_init(|| Mutex::new(MemoryTracker::default()))
}

#[derive(Debug, Default)]
pub struct MemoryTracker {
    pub total: isize,
    pub peak: isize,
}

impl MemoryTracker {
    pub fn reset(&mut self) {
        self.total = 0;
        self.peak = 0;
    }

    pub fn alloc(&mut self, size: usize) {
        self.total += size as isize;
        self.peak = self.peak.max(self.total);
    }

    pub fn free(&mut self, size: usize) {
        self.total -= size as isize;
    }
}

/// Reports every combo whose division by `(X - z)` left a nonzero remainder, for the GPU HALs'
/// batched `combos_divide`. `remainders` holds one entry per root, in `chunks` order.
///
/// The CPU HAL, and upstream's GPU HALs, assert that each remainder is zero: a nonzero one means the
/// combo polynomial does not vanish at the query point, so something upstream of this stage (the
/// witness, the coefficients, the evaluations, the check polynomial) is inconsistent and the seal
/// will not verify. Logged rather than asserted, so a bad proof still fails where callers already
/// handle it (verifying the segment) instead of panicking the prover; but it is named here, at the
/// stage it first shows.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn report_nonzero_remainders<E: ExtElem>(chunks: &[(usize, Vec<E>)], remainders: &[E]) {
    let mut remainders = remainders.iter();
    for (combo, pows) in chunks {
        for pow in 0..pows.len() {
            if remainders.next().is_some_and(|r| *r != E::ZERO) {
                tracing::error!(
                    "combos_divide: combo {combo} has a nonzero remainder at root {pow}; this \
                     segment's proof will not verify"
                );
            }
        }
    }
}

#[cfg(test)]
#[allow(unused)]
mod testutil {
    use std::rc::Rc;

    use rand::RngCore;
    use risc0_core::field::{
        baby_bear::{BabyBearElem, BabyBearExtElem},
        Elem, ExtElem,
    };

    use super::{dual::DualHal, Hal};
    use crate::{
        core::digest::Digest,
        hal::{cpu::CpuHal, Buffer},
        FRI_FOLD, INV_RATE,
    };

    const COUNTS: [usize; 7] = [1, 9, 12, 1001, 1024, 1025, 1024 * 1024];
    const DATA_SIZE: usize = 224;

    fn generate_elem<H: Hal, R: RngCore>(hal: &H, rng: &mut R, size: usize) -> H::Buffer<H::Elem> {
        let values: Vec<H::Elem> = (0..size).map(|_| H::Elem::random(rng)).collect();
        hal.copy_from_elem("values", &values)
    }

    fn generate_extelem<H: Hal, R: RngCore>(
        hal: &H,
        rng: &mut R,
        size: usize,
    ) -> H::Buffer<H::ExtElem> {
        let values: Vec<H::ExtElem> = (0..size).map(|_| H::ExtElem::random(rng)).collect();
        hal.copy_from_extelem("values", &values)
    }

    pub(crate) fn batch_bit_reverse<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let steps = 1 << 12;
        let count = DATA_SIZE;
        let domain = steps * INV_RATE;
        let io_size = count * domain;

        let io = generate_elem(&hal, &mut rng, io_size);
        hal.batch_bit_reverse(&io, count);
    }

    pub(crate) fn batch_evaluate_any<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let eval_size = 865;
        let poly_count = 223;
        let steps = 1 << 16;
        let coeffs_size = steps * poly_count;

        let z = H::ExtElem::random(&mut rng);
        let z_pow = z.pow(H::ExtElem::EXT_SIZE);

        let coeffs = generate_elem(&hal, &mut rng, coeffs_size);
        let which = hal.copy_from_u32("which", &vec![0; eval_size]);
        let xs = hal.copy_from_extelem("xs", &vec![z_pow; eval_size]);
        let out = hal.alloc_extelem("out", eval_size);

        hal.batch_evaluate_any(&coeffs, poly_count, &which, &xs, &out);
    }

    pub(crate) fn batch_expand_into_evaluate_ntt<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let count = DATA_SIZE;
        let expand_bits = 2;
        let steps = 1 << 16;
        let domain = steps * INV_RATE;
        let input_size = count * steps;
        let output_size = count * domain;

        let input = generate_elem(&hal, &mut rng, input_size);
        let output = hal.alloc_elem("output", output_size);
        hal.batch_expand_into_evaluate_ntt(&output, &input, count, expand_bits);
    }

    pub(crate) fn batch_expand_into_evaluate_ntt_large<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));
        for (po2, count) in [(16, 5), (17, 3), (18, 3), (19, 2), (20, 3), (21, 2)] {
            let steps = 1 << po2;
            let input = generate_elem(&hal, &mut rng, count * steps);
            let output = hal.alloc_elem("output", count * steps * INV_RATE);
            hal.batch_expand_into_evaluate_ntt(&output, &input, count, 2);
            // a second call checks that the input was left unchanged
            hal.batch_expand_into_evaluate_ntt(&output, &input, count, 2);
        }
    }

    pub(crate) fn batch_interpolate_ntt<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let count = DATA_SIZE;
        let steps = 1 << 16;
        let domain = steps * INV_RATE;
        let io_size = count * domain;

        let io = generate_elem(&hal, &mut rng, io_size);
        hal.batch_interpolate_ntt(&io, count);
    }

    pub(crate) fn gather_sample<H: Hal>(hal: H) {
        let mut rng = rand::rng();
        let rows = 1000;
        let cols = 900;
        let idx = 400;
        let src_size = rows * cols;
        let src = hal.alloc_elem("src", src_size);
        let dst = hal.alloc_elem("dst", rows);
        src.view_mut(|buf| {
            for x in 0..cols {
                for y in 0..rows {
                    let value = H::Elem::random(&mut rng);
                    buf[y * cols + x] = value;
                }
            }
        });
        hal.gather_sample(&dst, &src, idx, rows, cols);
        src.view(|src| {
            dst.view(|dst| {
                for y in 0..rows {
                    assert_eq!(src[y * cols + idx], dst[y]);
                }
            });
        });
    }

    pub(crate) fn check_req<H: Hal>(hal: H) {
        let a = hal.alloc_elem("a", 10);
        let b = hal.alloc_elem("b", 20);
        hal.eltwise_add_elem(&a, &b, &b);
    }

    pub(crate) fn eltwise_add_elem<H: Hal>(hal_gpu: H) {
        for (x, count) in COUNTS.iter().enumerate() {
            let a = hal_gpu.alloc_elem("a", *count);
            let b = hal_gpu.alloc_elem("b", *count);
            let o = hal_gpu.alloc_elem("o", *count);
            let mut golden = Vec::with_capacity(*count);

            let mut rng = rand::rng();
            a.view_mut(|a| {
                b.view_mut(|b| {
                    assert_eq!(a.len(), b.len());
                    for i in 0..a.len() {
                        a[i] = H::Elem::random(&mut rng);
                        b[i] = H::Elem::random(&mut rng);
                    }
                    for i in 0..a.len() {
                        golden.push(a[i] + b[i]);
                    }
                });
            });

            hal_gpu.eltwise_add_elem(&o, &a, &b);

            o.view(|o| {
                for i in 0..o.len() {
                    assert_eq!(o[i], golden[i], "x: {x}, count: {count}, i: {i}");
                }
            });
        }
    }

    pub(crate) fn eltwise_copy_elem<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        for count in COUNTS {
            let input = generate_elem(&hal_gpu, &mut rng, count);
            let output = hal_gpu.alloc_elem("output", count);
            hal_gpu.eltwise_copy_elem(&output, &input);
            output.view(|output| {
                input.view(|input| assert_eq!(output, input));
            });
        }
    }

    pub(crate) fn eltwise_sum_extelem<H: Hal>(hal_gpu: H) {
        const COUNT: usize = 1024 * 1024;

        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let input = generate_extelem(&hal, &mut rng, COUNT);
        let output = hal.alloc_elem("output", COUNT);
        hal.eltwise_sum_extelem(&output, &input);
    }

    pub(crate) fn fri_fold<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));
        for count in COUNTS {
            let output_size = count * H::ExtElem::EXT_SIZE;
            let input_size = output_size * FRI_FOLD;

            let output = hal.alloc_elem("output", output_size);
            let mix = H::ExtElem::random(&mut rng);
            let input = generate_elem(&hal, &mut rng, input_size);
            hal.fri_fold(&output, &input, &mix);
        }
    }

    pub(crate) fn mix_poly_coeffs<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let combo_count = 100;
        let steps = 1 << 12;
        let domain = steps * INV_RATE;
        let input_size = H::CHECK_SIZE * domain;
        let output_size = steps * (combo_count + 1);
        let combos = vec![0; H::CHECK_SIZE];
        let mix_start = H::ExtElem::random(&mut rng);
        let mix = H::ExtElem::random(&mut rng);

        // Zeroed, as the prover allocates it: mix_poly_coeffs adds into it, and a GPU HAL's
        // buffer pool can hand back a previous test's buffer.
        let output = hal.alloc_extelem_zeroed("output", output_size);
        let combos = hal.copy_from_u32("combos", &combos);
        let input = generate_elem(&hal, &mut rng, input_size);

        hal.mix_poly_coeffs(
            &output,
            &mix_start,
            &mix,
            &input,
            &combos,
            H::CHECK_SIZE,
            steps,
        );
    }

    pub(crate) fn hash_fold<H: Hal>(hal_gpu: H) {
        const INPUTS: usize = 1024;
        const OUTPUTS: usize = INPUTS / 2;
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));
        let io = hal.alloc_digest("io", INPUTS * 2);
        io.view_mut(|g| {
            for i in 0..INPUTS {
                g[i + INPUTS] = Digest::from([
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                ]);
            }
        });
        hal.hash_fold(&io, INPUTS, OUTPUTS);
    }

    /// `hash_fold_tree` (which GPU HALs implement as one call) against the CPU HAL.
    pub(crate) fn hash_fold_tree<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        // 13 and 14 reach the quad-per-permutation small-level kernels (pv_small.cuh, 2048 < n <= 8192).
        for layers in [1, 2, 7, 12, 13, 14] {
            let leaves = 1 << layers;
            let digests: Vec<Digest> = (0..leaves * 2)
                .map(|_| Digest::from(std::array::from_fn(|_| rng.next_u32() / 3)))
                .collect();
            let gpu_io = hal_gpu.copy_from_digest("io", &digests);
            let cpu_io = hal_cpu.copy_from_digest("io", &digests);
            hal_gpu.hash_fold_tree(&gpu_io, layers);
            hal_cpu.hash_fold_tree(&cpu_io, layers);
            assert!(
                gpu_io.to_vec()[1..] == cpu_io.to_vec()[1..],
                "layers={layers}"
            );
        }
    }

    pub(crate) fn hash_rows<H: Hal<Elem = BabyBearElem>>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));
        // Includes every column count the provers hash (1, 12, 16, 23, 64, 103, 128, 211) and
        // rows past one CUDA block.
        // 2049..8193 cover the warp / quad / thread-per-hash boundaries of pv_small.cuh.
        let rows = [1, 2, 3, 4, 10, 129, 1000, 2048, 2049, 8192, 8193];
        let cols = [1, 12, 16, 23, 32, 64, 103, 128, 211];
        for row_count in rows {
            for col_count in cols {
                let matrix_size = row_count * col_count;
                let matrix = generate_elem(&hal, &mut rng, matrix_size);
                let output = hal.alloc_digest("output", row_count);
                hal.hash_rows(&output, &matrix);
            }
        }
    }

    pub(crate) fn slice<H: Hal<Elem = BabyBearElem>>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let rows = 4096;
        let cols = 256;
        let matrix_size = rows * cols;

        let nodes = hal.alloc_digest("nodes", rows * 2);
        let matrix = generate_elem(&hal, &mut rng, matrix_size);
        hal.hash_rows(&nodes.slice(rows, rows), &matrix);
    }

    /// `commit_coeffs_lde` against the unfused path (`make_coeffs` + `PolyGroup::new`) on the
    /// CPU HAL.
    pub(crate) fn commit_coeffs_lde<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let lg_blowup = crate::core::log2_ceil(INV_RATE);
        for (po2, count) in [
            (10, 1),
            (10, 7),
            (11, 64),
            (13, 33),
            (16, 211),
            (18, 5),
            (21, 2),
            (20, 9),
            (21, 3),
            (22, 1),
            (17, 7),
            (19, 3),
            (18, 33),
        ] {
            assert!(hal_gpu.has_commit_coeffs_lde(po2));
            let size = (1 << po2) * count;
            let values: Vec<H::Elem> = (0..size).map(|_| H::Elem::random(&mut rng)).collect();
            let witness = hal_gpu.copy_from_elem("witness", &values);
            let coeffs = hal_gpu.alloc_elem("coeffs", size);
            let evaluated = hal_gpu.alloc_elem("evaluated", size * INV_RATE);
            hal_gpu.commit_coeffs_lde(&coeffs, &evaluated, &witness, count);

            let cpu_witness = hal_cpu.copy_from_elem("witness", &values);
            let cpu_coeffs = hal_cpu.alloc_elem("coeffs", size);
            hal_cpu.batch_interpolate_ntt_zk_shift_from(&cpu_coeffs, &cpu_witness, count);
            let cpu_evaluated = hal_cpu.alloc_elem("evaluated", size * INV_RATE);
            hal_cpu.batch_expand_into_evaluate_ntt(&cpu_evaluated, &cpu_coeffs, count, lg_blowup);
            hal_cpu.batch_bit_reverse(&cpu_coeffs, count);

            assert!(
                witness.to_vec() == values,
                "po2={po2} count={count}: witness changed"
            );
            assert!(
                coeffs.to_vec() == cpu_coeffs.to_vec(),
                "po2={po2} count={count}: coeffs differ"
            );
            assert!(
                evaluated.to_vec() == cpu_evaluated.to_vec(),
                "po2={po2} count={count}: evaluated differs"
            );
        }
    }

    pub(crate) fn zk_shift<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));
        let counts = [(1000, (1 << 8)), (900, (1 << 12))];
        for (poly_count, steps) in counts {
            let count = poly_count * steps;
            let io = generate_elem(&hal, &mut rng, count);
            hal.zk_shift(&io, poly_count);
        }
    }
}
