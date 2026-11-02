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

use anyhow::Result;
use risc0_circuit_recursion_sys::{RawPreflightTrace, StepMode};
use risc0_zkp::hal::Hal;

pub(crate) mod cpu;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "rocm")]
pub(crate) mod hip;
#[cfg(feature = "intel")]
pub(crate) mod intel;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod metal;

pub(crate) trait CircuitWitnessGenerator<H: Hal> {
    fn generate_witness(
        &self,
        mode: StepMode,
        total_cycles: u32,
        preflight: &RawPreflightTrace,
        ctrl: &H::Buffer<H::Elem>,
        data: &H::Buffer<H::Elem>,
        global: &H::Buffer<H::Elem>,
    ) -> Result<()>;

    /// Like `generate_witness`, with the rows of some sequential chains already computed on
    /// the host. HALs without support ignore them and run every chain.
    #[allow(clippy::too_many_arguments)]
    fn generate_witness_with_chains(
        &self,
        mode: StepMode,
        total_cycles: u32,
        preflight: &RawPreflightTrace,
        _host_chains: Option<&super::witgen::HostChains>,
        ctrl: &H::Buffer<H::Elem>,
        data: &H::Buffer<H::Elem>,
        global: &H::Buffer<H::Elem>,
    ) -> Result<()> {
        self.generate_witness(mode, total_cycles, preflight, ctrl, data, global)
    }

    /// Upload a witness that was generated on the host ahead of time.
    fn upload_host_witness(
        &self,
        _data: &H::Buffer<H::Elem>,
        _global: &H::Buffer<H::Elem>,
        _host_data: Vec<H::Elem>,
        _host_global: Vec<H::Elem>,
    ) -> Result<()> {
        anyhow::bail!("upload_host_witness not supported by this HAL")
    }

    /// Whether this HAL has the witness buffer passes below (`init_rows`, `fill_tail`,
    /// `zeroize_rows`); without them the witgen uses the generic HAL passes.
    fn has_row_passes(&self) -> bool {
        false
    }

    /// A fresh column-major buffer of `rows` rows per column: INVALID in rows [0, live), zero
    /// after (the generic path fills it all with INVALID and zeroizes it after the witgen, which
    /// writes no row >= live).
    fn init_rows(&self, _buf: &H::Buffer<H::Elem>, _rows: usize, _live: usize) -> Result<()> {
        anyhow::bail!("init_rows not supported by this HAL")
    }

    /// `value` in the last `zk` rows of every column.
    fn fill_tail(
        &self,
        _buf: &H::Buffer<H::Elem>,
        _rows: usize,
        _zk: usize,
        _value: H::Elem,
    ) -> Result<()> {
        anyhow::bail!("fill_tail not supported by this HAL")
    }

    /// INVALID -> zero in rows [0, live) of every column.
    fn zeroize_rows(&self, _buf: &H::Buffer<H::Elem>, _rows: usize, _live: usize) -> Result<()> {
        anyhow::bail!("zeroize_rows not supported by this HAL")
    }
}

pub(crate) trait CircuitAccumulator<H: Hal> {
    #[allow(clippy::too_many_arguments)]
    fn accumulate(
        &self,
        work_cycles: u32,
        total_cycles: u32,
        ctrl: &H::Buffer<H::Elem>,
        global: &H::Buffer<H::Elem>,
        data: &H::Buffer<H::Elem>,
        mix: &H::Buffer<H::Elem>,
        accum: &H::Buffer<H::Elem>,
    ) -> Result<()>;
}
