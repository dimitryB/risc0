// Copyright 2026 RISC Zero, Inc.
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

use alloc::{sync::Arc, vec::Vec};

use parking_lot::Mutex;

use risc0_core::scope;

use crate::{
    core::digest::Digest,
    hal::{Buffer, Hal},
    merkle::MerkleTreeParams,
    prove::write_iop::WriteIOP,
};

pub struct MerkleTreeProver<H: Hal> {
    params: MerkleTreeParams,

    // The retained matrix of values (Option to allow releasing for VRAM savings)
    pub(crate) matrix: Option<H::Buffer<H::Elem>>,

    // A heap style array where node N has children 2*N and 2*N+1.  The size of
    // this buffer is (1 << (layers + 1)) and begins at offset 1 (zero is unused
    // to make indexing nicer).
    nodes: H::Buffer<Digest>,

    // The root value
    root: Digest,

    // Pre-allocated buffer for gather_sample in prove(), avoids per-query alloc/free
    sample_buf: Option<H::Buffer<H::Elem>>,

    // Cached top nodes for commit() — shared across clones via Arc.
    // Populated on first commit(), reused on subsequent commits to avoid GPU
    // D2H. Arc+spin::Mutex (instead of Rc+RefCell) so MerkleTreeProver is
    // Send — required for the finalize-overlap path that moves the prover
    // state into a background thread.
    cached_top_nodes: Arc<Mutex<Option<Vec<Digest>>>>,
}

impl<H: Hal> Clone for MerkleTreeProver<H> {
    fn clone(&self) -> Self {
        Self {
            params: self.params,
            matrix: self.matrix.clone(),
            nodes: self.nodes.clone(),
            root: self.root,
            sample_buf: self.sample_buf.clone(),
            cached_top_nodes: self.cached_top_nodes.clone(),
        }
    }
}

impl<H: Hal> MerkleTreeProver<H> {
    /// Release the matrix buffer to free GPU memory.
    /// After this, batch_prove will panic unless set_matrix() is called first.
    pub fn release_matrix(&mut self) {
        self.matrix = None;
    }

    /// Set the matrix buffer (e.g., after reconstructing evaluated from coefficients).
    pub fn set_matrix(&mut self, matrix: H::Buffer<H::Elem>) {
        self.matrix = Some(matrix);
    }

    /// Check if the matrix is present.
    pub fn matrix_is_some(&self) -> bool {
        self.matrix.is_some()
    }
}

impl<H: Hal> MerkleTreeProver<H> {
    /// Generate a merkle tree from a matrix of values.
    ///
    /// The proofs will prove a single 'column' of values in the tree at a
    /// certain row. Layout is presumed to be packed row-major.
    /// The number of queries represents the expected # of queries and
    /// determines the size of the 'top' layer. It is important that the
    /// verifier is constructed with identical size parameters, including # of
    /// queries, or verification may fail.
    ///
    /// matrix: `rows * cols`
    /// rows: `domain = steps * INV_RATE`, `steps` is always a power of 2.
    /// cols: `count = circuit_cols`
    pub fn new(
        hal: &H,
        matrix: &H::Buffer<H::Elem>,
        rows: usize,
        cols: usize,
        queries: usize,
    ) -> Self {
        assert_eq!(matrix.size(), rows * cols);
        let params = MerkleTreeParams::new(rows, cols, queries);
        // Allocate nodes
        let nodes = hal.alloc_digest("nodes", rows * 2);
        // hash each column
        hal.hash_rows(&nodes.slice(rows, rows), matrix);
        // For each layer, hash up the layer below
        scope!("hash_fold", {
            hal.hash_fold_tree(&nodes, params.layers);
        });
        // ROCm: the root and the top layer (written by `commit`) in one download: nodes[1..2 *
        // top_size) holds both (root at 1, top layer at top_size..2 * top_size).
        let (root, top_nodes) = if cfg!(feature = "rocm") {
            let top_size = params.top_size;
            let mut root = Digest::ZERO;
            let mut top_nodes = Vec::new();
            nodes.slice(1, 2 * top_size - 1).view(|view| {
                root = view[0];
                top_nodes = view[top_size - 1..].to_vec();
            });
            (root, Some(top_nodes))
        } else {
            (nodes.get_at(1), None)
        };
        let sample_buf = if !hal.has_unified_memory() && params.col_size > 0 {
            Some(hal.alloc_elem("sample_reuse", params.col_size))
        } else {
            None
        };
        MerkleTreeProver {
            params,
            matrix: Some(matrix.clone()),
            nodes,
            root,
            sample_buf,
            cached_top_nodes: Arc::new(Mutex::new(top_nodes)),
        }
    }

    /// Write the 'top' of the merkle tree and commit to the root.
    pub fn commit(&self, iop: &mut WriteIOP<H::Field>) {
        scope!("commit");
        let top_size = self.params.top_size;
        let cached = self.cached_top_nodes.lock();
        if let Some(ref top_nodes) = *cached {
            // Use cached CPU data — no GPU D2H needed.
            iop.write_pod_slice(top_nodes);
        } else {
            drop(cached);
            let slice = self.nodes.slice(top_size, top_size);
            slice.view(|view| {
                *self.cached_top_nodes.lock() = Some(view.to_vec());
                iop.write_pod_slice(view);
            });
        }
        iop.commit(self.root());
    }

    /// Get the root digest of the tree.
    pub fn root(&self) -> &Digest {
        &self.root
    }

    /// Generate a proof at a given index, and return the values at that column.
    ///
    /// The format of the proof is always:
    /// 1) The column itself
    /// 2) The 'other' digests up to the top.
    ///
    /// It is presumed the verifier is given the index of the row from other
    /// parts of the protocol, and verification will of course fail if the
    /// wrong row is specified.
    pub fn prove(&self, hal: &H, iop: &mut WriteIOP<H::Field>, idx: usize) -> Vec<H::Elem> {
        assert!(idx < self.params.row_size);
        let matrix = self
            .matrix
            .as_ref()
            .expect("matrix released; call set_matrix() first");
        let mut out = Vec::with_capacity(self.params.col_size);
        if hal.has_unified_memory() {
            matrix.view(|view| {
                for i in 0..self.params.col_size {
                    out.push(view[idx + i * self.params.row_size]);
                }
            });
        } else if let Some(ref sample) = self.sample_buf {
            hal.gather_sample(
                sample,
                matrix,
                idx,
                self.params.col_size,
                self.params.row_size,
            );
            sample.view(|view| {
                out.extend_from_slice(view);
            });
        }
        iop.write_field_elem_slice::<H::Elem>(out.as_slice());
        // Collect all sibling indices for the proof path
        let mut node_indices = Vec::new();
        let mut cur_idx = idx + self.params.row_size;
        while cur_idx >= 2 * self.params.top_size {
            let low_bit = cur_idx % 2;
            cur_idx /= 2;
            node_indices.push(2 * cur_idx + (1 - low_bit));
        }
        // Batch-read all sibling digests (single D2H on CUDA)
        let siblings = hal.batch_get_digest_at(&self.nodes, &node_indices);
        for digest in &siblings {
            iop.write_pod_slice(&[*digest]);
        }
        out
    }

    /// Sibling node indices of the Merkle paths of `indices`, and how many each path has.
    fn sibling_indices(&self, indices: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut all = Vec::new();
        let mut counts = Vec::with_capacity(indices.len());
        for &idx in indices {
            let mut count = 0;
            let mut cur_idx = idx + self.params.row_size;
            while cur_idx >= 2 * self.params.top_size {
                let low_bit = cur_idx % 2;
                cur_idx /= 2;
                all.push(2 * cur_idx + (1 - low_bit));
                count += 1;
            }
            counts.push(count);
        }
        (all, counts)
    }

    /// [Self::batch_prove] for several trees (each with its own indices), with the GPU gathers
    /// of all of them in one host round trip where the HAL supports it (`Hal::gather_multi`).
    #[allow(clippy::type_complexity)]
    pub fn batch_prove_many(
        hal: &H,
        jobs: &[(&Self, &[usize])],
    ) -> Vec<Vec<(Vec<H::Elem>, Vec<Digest>)>> {
        let fallback = || -> Vec<Vec<(Vec<H::Elem>, Vec<Digest>)>> {
            jobs.iter()
                .map(|(tree, indices)| tree.batch_prove(hal, indices))
                .collect()
        };
        if hal.has_unified_memory()
            || jobs.iter().any(|(t, _)| t.matrix.is_none() || t.params.col_size == 0)
        {
            return fallback();
        }
        for (tree, indices) in jobs {
            for &idx in indices.iter() {
                assert!(idx < tree.params.row_size);
            }
        }
        let paths: Vec<(Vec<usize>, Vec<usize>)> =
            jobs.iter().map(|(t, indices)| t.sibling_indices(indices)).collect();
        let gjobs: Vec<crate::hal::GatherJob<'_, H>> = jobs
            .iter()
            .zip(&paths)
            .map(|((t, indices), (nodes, _))| crate::hal::GatherJob {
                matrix: t.matrix.as_ref().unwrap(),
                col_size: t.params.col_size,
                row_size: t.params.row_size,
                rows: indices,
                nodes: &t.nodes,
                node_indices: nodes,
            })
            .collect();
        let Some(gathered) = hal.gather_multi(&gjobs) else {
            return fallback();
        };
        gathered
            .into_iter()
            .zip(jobs.iter().zip(&paths))
            .map(|((cols, digests), ((t, indices), (_, counts)))| {
                let col_size = t.params.col_size;
                let mut off = 0;
                (0..indices.len())
                    .map(|q| {
                        let c = cols[q * col_size..(q + 1) * col_size].to_vec();
                        let d = digests[off..off + counts[q]].to_vec();
                        off += counts[q];
                        (c, d)
                    })
                    .collect()
            })
            .collect()
    }

    /// Batch-prove multiple indices at once, returning data without writing to IOP.
    /// GPU work is batched: N gather_sample kernels + 1 D2H, 1 batch digest read + 1 D2H.
    /// The caller is responsible for writing to IOP in the correct order.
    pub fn batch_prove(&self, hal: &H, indices: &[usize]) -> Vec<(Vec<H::Elem>, Vec<Digest>)> {
        let n = indices.len();
        if n == 0 {
            return Vec::new();
        }
        let matrix = self
            .matrix
            .as_ref()
            .expect("matrix released; call set_matrix() first");
        let col_size = self.params.col_size;

        // Collect column data for all queries
        let mut all_col_data: Vec<Vec<H::Elem>> = Vec::with_capacity(n);
        if hal.has_unified_memory() {
            matrix.view(|view| {
                for &idx in indices {
                    assert!(idx < self.params.row_size);
                    let mut out = Vec::with_capacity(col_size);
                    for i in 0..col_size {
                        out.push(view[idx + i * self.params.row_size]);
                    }
                    all_col_data.push(out);
                }
            });
        } else if col_size > 0 {
            // GPU path: batch all gather_sample kernels (async), single D2H via view()
            let batch_buf = hal.alloc_elem("batch_samples", n * col_size);
            for &idx in indices.iter() {
                assert!(idx < self.params.row_size);
            }
            hal.gather_samples(&batch_buf, matrix, indices, col_size, self.params.row_size);
            batch_buf.view(|view| {
                for i in 0..n {
                    all_col_data.push(view[i * col_size..(i + 1) * col_size].to_vec());
                }
            });
        } else {
            all_col_data.resize_with(n, Vec::new);
        }

        // Compute all sibling node indices across all queries
        let mut all_node_indices = Vec::new();
        let mut per_query_digest_count = Vec::new();
        for &idx in indices {
            let mut count = 0;
            let mut cur_idx = idx + self.params.row_size;
            while cur_idx >= 2 * self.params.top_size {
                let low_bit = cur_idx % 2;
                cur_idx /= 2;
                all_node_indices.push(2 * cur_idx + (1 - low_bit));
                count += 1;
            }
            per_query_digest_count.push(count);
        }

        // Single batch digest read (1 H2D + 1 kernel + 1 D2H on CUDA)
        let all_digests = hal.batch_get_digest_at(&self.nodes, &all_node_indices);

        // Assemble per-query results
        let mut results = Vec::with_capacity(n);
        let mut digest_offset = 0;
        for (i, &count) in per_query_digest_count.iter().enumerate() {
            let digests = all_digests[digest_offset..digest_offset + count].to_vec();
            digest_offset += count;
            results.push((core::mem::take(&mut all_col_data[i]), digests));
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use risc0_core::field::{
        baby_bear::{BabyBear, BabyBearElem},
        Elem,
    };

    use super::*;
    use crate::{
        core::{
            hash::{poseidon2::Poseidon2HashSuite, sha::Sha256HashSuite, HashSuite},
            log2_ceil,
        },
        hal::cpu::CpuHal,
        verify::{MerkleTreeVerifier, ReadIOP, VerificationError},
    };

    fn init_prover<H: Hal>(
        hal: &H,
        rows: usize,
        cols: usize,
        queries: usize,
    ) -> MerkleTreeProver<H> {
        // Initialize a prover with leaves 0..size
        let size: u32 = (rows * cols) as u32;
        let mut data: Vec<H::Elem> = Vec::new();
        for val in 0..size {
            data.push(H::Elem::from_u64((u32::MAX / 2) as u64 - val as u64));
        }
        let matrix = hal.copy_from_elem("matrix", data.as_slice());

        MerkleTreeProver::new(hal, &matrix, rows, cols, queries)
    }

    fn bad_row_access(suite: HashSuite<BabyBear>, rows: usize, cols: usize, queries: usize) {
        let hal = CpuHal::new(suite);
        let prover = init_prover(&hal, rows, cols, queries);
        let mut iop = WriteIOP::new(hal.get_hash_suite().rng.as_ref());
        prover.prove(&hal, &mut iop, rows);
    }

    fn bad_row_access_all(rows: usize, cols: usize, queries: usize) {
        bad_row_access(Sha256HashSuite::new_suite(), rows, cols, queries);
        bad_row_access(Poseidon2HashSuite::new_suite(), rows, cols, queries);
    }

    fn possibly_bad_verify(
        suite: HashSuite<BabyBear>,
        rows: usize,
        cols: usize,
        queries: usize,
        bad_query: usize,
        manipulate_proof: bool,
    ) {
        let hal = CpuHal::new(suite);
        let hashfn = hal.get_hash_suite().hashfn.as_ref();
        let rng = hal.get_hash_suite().rng.as_ref();
        let prover = init_prover(&hal, rows, cols, queries);

        let mut iop = WriteIOP::new(rng);
        prover.commit(&mut iop);
        for _query in 0..queries {
            let r_idx = iop.rng.random_bits(log2_ceil(rows)) as usize;
            let col = prover.prove(&hal, &mut iop, r_idx);
            for (c_idx, col) in col.iter().enumerate() {
                assert_eq!(
                    *col,
                    BabyBearElem::from_u64((u32::MAX / 2) as u64 - ((r_idx + c_idx * rows) as u64))
                );
            }
        }
        if manipulate_proof {
            let mut rng = rand::rng();
            let manip_idx = rng.random_range(0..iop.proof.len());
            iop.proof[manip_idx] ^= 1;
        }
        let mut r_iop = ReadIOP::new(&iop.proof, rng);
        let verifier = MerkleTreeVerifier::new(&mut r_iop, hashfn, rows, cols, queries).unwrap();
        assert_eq!(verifier.root(), prover.root());
        let mut err = false;
        for query in 0..queries {
            let r_idx = r_iop.random_bits(log2_ceil(rows)) as usize;
            if query == bad_query {
                assert_ne!(
                    rows, 1,
                    "Cannot test for bad query if there is only one row"
                );
                let r_idx = (r_idx + 1) % rows;
                let verification = verifier.verify(&mut r_iop, hashfn, r_idx);
                match verification {
                    Ok(_) => {
                        panic!("Merkle tree wrongly passed verify when tested on the wrong row")
                    }
                    Err(VerificationError::InvalidProof) => {}
                    Err(_) => panic!("Merkle tree failed validation for an unexpected reason"),
                }
                err = true;
                break;
            }
            let col = verifier.verify(&mut r_iop, hashfn, r_idx).unwrap();
            for (c_idx, cell) in col.iter().enumerate().take(cols) {
                assert_eq!(
                    *cell,
                    BabyBearElem::from((u32::MAX / 2) - ((r_idx + c_idx * rows) as u32))
                );
            }
        }
        if !err {
            r_iop.verify_complete().unwrap();
        }
    }

    fn possibly_bad_verify_all(
        rows: usize,
        cols: usize,
        queries: usize,
        bad_query: usize,
        manipulate_proof: bool,
    ) {
        possibly_bad_verify(
            Sha256HashSuite::new_suite(),
            rows,
            cols,
            queries,
            bad_query,
            manipulate_proof,
        );
        possibly_bad_verify(
            Poseidon2HashSuite::new_suite(),
            rows,
            cols,
            queries,
            bad_query,
            manipulate_proof,
        );
    }

    fn randomize_sizes() -> (usize, usize, usize) {
        // Chooses random values of `rows`, `cols`, and `queries` such that:
        // `rows` is a power of 2
        // `cols` & `queries` have a wide distribution but tend to take small values
        let mut rng = rand::rng();
        let rows = 1 << (rng.random_range(0..10));
        let cols_po2 = rng.random_range(0..10);
        let cols = (rng.random_range(0..(1 << cols_po2))) + 1;
        let queries_po2 = rng.random_range(0..10);
        let queries = (rng.random_range(0..(1 << queries_po2))) + 1;
        (rows, cols, queries)
    }

    #[test]
    #[should_panic(expected = "assertion failed: idx < self.params.row_size")]
    fn merkle_cpu_1_1_1_bad_row_access() {
        bad_row_access_all(1, 1, 1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: idx < self.params.row_size")]
    fn merkle_cpu_4_4_2_bad_row_access() {
        bad_row_access_all(4, 4, 2);
    }

    #[test]
    #[should_panic(expected = "assertion failed: idx < self.params.row_size")]
    fn merkle_cpu_randomized_bad_row_access() {
        let (rows, cols, queries) = randomize_sizes();
        bad_row_access_all(rows, cols, queries);
    }

    #[test]
    fn merkle_cpu_1_1_1_verify() {
        // Test a complete verification with no bad queries (by setting bad_query out of
        // range)
        possibly_bad_verify_all(1, 1, 1, 4, false);
    }

    #[test]
    fn merkle_cpu_4_4_2_verify() {
        // Test a complete verification with no bad queries (by setting bad_query out of
        // range)
        possibly_bad_verify_all(4, 4, 2, 4, false);
    }

    #[test]
    fn merkle_cpu_randomized_verify() {
        for _rep in 0..100 {
            let (rows, cols, queries) = randomize_sizes();
            // Test a complete verification with no bad queries (by setting bad_query out of
            // range)
            possibly_bad_verify_all(rows, cols, queries, queries + 1, false);
        }
    }

    #[test]
    fn merkle_cpu_2_1_1_bad_query() {
        // n.b. since we test bad queries by incrementing the row, we can't test for a
        // bad query with rows == 1
        possibly_bad_verify_all(2, 1, 1, 0, false);
    }

    #[test]
    fn merkle_cpu_4_4_2_bad_query() {
        let mut rng = rand::rng();
        let queries = 2;
        // Test a complete verification with a bad query
        let bad_query = rng.random_range(0..queries);
        possibly_bad_verify_all(4, 4, queries, bad_query, false);
    }

    #[test]
    fn merkle_cpu_randomized_bad_query() {
        let mut rng = rand::rng();
        let (rows, cols, queries) = randomize_sizes();
        // At least two rows are required to test querying an incorrect row
        let rows = if rows == 1 { 2 } else { rows };
        // Test a complete verification with a bad query
        let bad_query = rng.random_range(0..queries);
        possibly_bad_verify_all(rows, cols, queries, bad_query, false);
    }

    #[test]
    #[should_panic]
    fn merkle_cpu_1_1_1_verify_manipulated() {
        for _rep in 0..50 {
            // Test a verification with a manipulated proof but no bad queries (by setting
            // bad_query out of range) Do this multiple times as the
            // manipulation location is random
            possibly_bad_verify_all(1, 1, 1, 2, true);
        }
    }

    #[test]
    #[should_panic]
    fn merkle_cpu_4_4_2_verify_manipulated() {
        for _rep in 0..50 {
            // Test a verification with a manipulated proof but no bad queries (by setting
            // bad_query out of range) Do this multiple times as the
            // manipulation location is random
            possibly_bad_verify_all(4, 4, 2, 4, true);
        }
    }

    #[test]
    #[should_panic]
    fn merkle_cpu_randomized_verify_manipulated() {
        for _rep in 0..50 {
            let (rows, cols, queries) = randomize_sizes();
            // Test a verification with a manipulated proof but no bad queries (by setting
            // bad_query out of range) Do this multiple times as the
            // manipulation location is random
            possibly_bad_verify_all(rows, cols, queries, queries + 1, true);
        }
    }
}
