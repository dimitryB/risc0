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

//! Bounded Poseidon2 Merkle folding with CPU-owned proving buffers.

use std::{cell::Cell, mem::size_of};

use anyhow::{ensure, Result};

use super::{
    cpu::{CpuBuffer, CpuHashFold, MetalFoldStats},
    metal::MetalHalPoseidon2,
    Buffer, Hal,
};
use crate::core::digest::Digest;

/// Runs only Merkle pair hashes on Metal. Witness generation, NTTs, constraint
/// evaluation, leaf hashing, FRI arithmetic, and recursion remain on the CPU.
pub struct MetalPoseidon2Fold {
    hal: MetalHalPoseidon2,
    stats: Cell<MetalFoldStats>,
}

impl MetalPoseidon2Fold {
    pub fn new(staging_bytes: usize) -> Result<Self> {
        ensure!(
            (4096..=16 * 1024 * 1024).contains(&staging_bytes),
            "Metal fold staging budget must be between 4 KiB and 16 MiB"
        );
        Ok(Self {
            hal: metal::objc::rc::autoreleasepool(MetalHalPoseidon2::new),
            stats: Cell::new(MetalFoldStats {
                staging_limit_bytes: staging_bytes,
                ..MetalFoldStats::default()
            }),
        })
    }
}

impl CpuHashFold for MetalPoseidon2Fold {
    fn hash_fold(&self, io: &CpuBuffer<Digest>, input_size: usize, output_size: usize) -> bool {
        assert_eq!(
            input_size,
            output_size.checked_mul(2).expect("fold size overflow")
        );
        assert!(io.size() >= input_size.checked_mul(2).expect("fold size overflow"));
        let mut stats = self.stats.get();
        let outputs_per_chunk = stats.staging_limit_bytes / (4 * size_of::<Digest>());
        let before = self.hal.completed_dispatches();
        for offset in (0..output_size).step_by(outputs_per_chunk) {
            let count = outputs_per_chunk.min(output_size - offset);
            metal::objc::rc::autoreleasepool(|| {
                // The retained kernel uses heap indices: outputs [n,2n), inputs [2n,4n).
                // Shared memory permits direct copies; no intermediate host Vec is needed.
                let staging = self.hal.alloc_digest("metal_fold_staging", count * 4);
                let input = staging.slice(count * 2, count * 2);
                let source_start = input_size + offset * 2;
                input.view_mut(|dst| {
                    io.view(|src| {
                        dst.copy_from_slice(&src[source_start..source_start + count * 2])
                    });
                });
                self.hal.hash_fold(&staging, count * 2, count);
                staging.slice(count, count).view(|src| {
                    io.view_mut(|dst| {
                        dst[output_size + offset..output_size + offset + count]
                            .copy_from_slice(src);
                    });
                });
                // Drain completed command buffers as well as Rust buffer owners before
                // allocating the next chunk, so autoreleased resources cannot accumulate.
            });
            stats.peak_staging_bytes = stats
                .peak_staging_bytes
                .max(count * 4 * size_of::<Digest>());
        }
        stats.completed_dispatches += self.hal.completed_dispatches() - before;
        stats.fold_calls += 1;
        stats.output_digests += output_size as u64;
        stats.input_digests += input_size as u64;
        self.stats.set(stats);
        // Opting in never falls back to CPU hashing, even for an empty fold.
        true
    }

    fn metal_stats(&self) -> Option<MetalFoldStats> {
        Some(self.stats.get())
    }
}

#[cfg(test)]
mod tests {
    use risc0_core::field::baby_bear::BabyBear;

    use super::*;
    use crate::{core::hash::poseidon2::Poseidon2HashSuite, hal::cpu::CpuHal};

    #[test]
    fn bounded_fold_matches_cpu_with_offsets_and_chunk_tails() {
        let cpu = CpuHal::<BabyBear>::new(Poseidon2HashSuite::new_suite());
        let hybrid = CpuHal::<BabyBear, _>::with_hash_fold(
            Poseidon2HashSuite::new_suite(),
            MetalPoseidon2Fold::new(4096).unwrap(),
        );
        for outputs in [1, 2, 31, 32, 33, 1024] {
            let values: Vec<Digest> = (0..outputs * 4 + 22)
                .map(|i| Digest::from([i as u32 + 1; 8]))
                .collect();
            let expected = cpu.copy_from_digest("expected", &values);
            let actual = hybrid.copy_from_digest("actual", &values);
            cpu.hash_fold(&expected.slice(13, outputs * 4), outputs * 2, outputs);
            hybrid.hash_fold(&actual.slice(13, outputs * 4), outputs * 2, outputs);
            expected.view(|expected| actual.view(|actual| assert_eq!(actual, expected)));
        }
        let stats = hybrid.metal_stats().unwrap();
        assert_eq!(stats.fold_calls, 6);
        assert_eq!(stats.output_digests, 1123);
        assert_eq!(stats.input_digests, 2246);
        assert_eq!(stats.completed_dispatches, 38);
        assert_eq!(stats.peak_staging_bytes, 4096);
        assert_eq!(stats.staging_limit_bytes, 4096);
    }

    #[test]
    fn default_cpu_hal_remains_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CpuHal<BabyBear>>();
        assert!(MetalPoseidon2Fold::new(4095).is_err());
        assert!(MetalPoseidon2Fold::new(16 * 1024 * 1024 + 1).is_err());
    }
}
