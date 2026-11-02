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

use std::collections::BTreeSet;

use anyhow::{anyhow, bail, ensure, Result};
use derive_more::Debug;
use num_traits::FromPrimitive as _;
use risc0_binfmt::{ByteAddr, WordAddr};
use risc0_circuit_rv32im_sys::{RawMemoryTransaction, RawPreflightCycle};
use risc0_core::scope;
use risc0_zkp::core::digest::DIGEST_WORDS;

use crate::{
    execute::{
        bigint::BigIntBytes,
        node_idx,
        pager::{page_idx, PageState, PagedMemory},
        platform::*,
        poseidon2::{Poseidon2, Poseidon2State},
        r0vm::{EcallKind, LoadOp, Risc0Context, Risc0Machine},
        rv32im::{disasm, Emulator, InsnKind},
        segment::Segment,
        sha2::Sha2State,
    },
    zirgen::circuit::ExtVal,
};

use super::{
    bigint::BigIntState,
    node_addr_to_idx, node_idx_to_addr,
    paged_map::{PagedMap, WordTxnMap},
    poseidon2::Checksum,
};

/// Per-row side data. The large states live in side vectors of the trace (`p2_states`,
/// `sha2_states`, `bigint_states`) and the row keeps their index: this keeps a row at 16 bytes
/// (it was 160, the size of `Poseidon2State`), and the 1M-row vector is written by the body
/// and read by the injector and the zcheck pass every segment.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) enum Back {
    #[default]
    None,
    Ecall(u32, u32, u32),
    /// Index into `PreflightTrace::p2_states`.
    Poseidon2(u32),
    /// Index into `PreflightTrace::sha2_states`.
    Sha2(u32),
    /// Index into `PreflightTrace::bigint_states`.
    BigInt(u32),
}

const _: () = assert!(std::mem::size_of::<Back>() == 16);

#[derive(Clone, Debug, Default)]
pub(crate) struct PreflightTrace {
    #[debug("{}", cycles.len())]
    pub cycles: Vec<RawPreflightCycle>,
    #[debug("{}", txns.len())]
    pub txns: Vec<RawMemoryTransaction>,
    #[debug("{}", bigint_bytes.len())]
    pub bigint_bytes: Vec<u8>,
    #[debug("{}", backs.len())]
    pub backs: Vec<Back>,
    #[debug("{}", p2_states.len())]
    pub p2_states: Vec<Poseidon2State>,
    #[debug("{}", sha2_states.len())]
    pub sha2_states: Vec<Sha2State>,
    #[debug("{}", bigint_states.len())]
    pub bigint_states: Vec<BigIntState>,
    pub table_split_cycle: u32,
    pub rand_z: ExtVal,
    /// Pinned copy of cycles|txns|bigint_bytes for the CUDA/HIP witgen upload (see cuda_stage).
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    #[debug(skip)]
    pub staged: Option<std::sync::Arc<super::cuda_stage::StagedTrace>>,
    /// The injector left out the five per-cycle columns (CYCLE_COLS): witgen writes them.
    pub cycle_cols_deferred: bool,
}

// Return the big vectors for reuse by a later segment's preflight (see cuda_stage).
#[cfg(any(feature = "cuda", feature = "rocm"))]
impl Drop for PreflightTrace {
    fn drop(&mut self) {
        super::cuda_stage::recycle_trace(self);
    }
}

pub(crate) struct Preflight<'a> {
    pub trace: PreflightTrace,
    segment: &'a Segment,
    pager: PagedMemory,
    pc: ByteAddr,
    machine_mode: u32,
    cur_write: usize,
    cur_read: usize,
    user_cycle: u32,
    txn_idx: u32,
    bigint_idx: u32,
    user_cycles: u32,
    /// orig_words + prev_cycle (see `WordTxnMap`).
    word_txns: WordTxnMap,
    page_memory: PagedMap,
    /// First guest sys_poseidon2 ecall this segment cannot prove (pc, a0, a1, a2).
    unprovable_poseidon2: Option<(u32, u32, u32, u32)>,
}

impl Segment {
    pub(crate) fn preflight(&self, rand_z: ExtVal) -> Result<PreflightTrace> {
        scope!("preflight");
        tracing::debug!("preflight: {self:#?}");

        let mut preflight = Preflight::new(self, rand_z);
        preflight.read_povw_nonce()?;
        preflight.read_pages()?;
        preflight.body()?;
        if let Some((pc, a0, a1, a2)) = preflight.unprovable_poseidon2 {
            bail!(
                "sys_poseidon2 with non-null pointers cannot be proven by the RISC Zero 3.0 \
                 rv32im circuit (pc={pc:#010x}, a0={a0:#010x}, a1={a1:#010x}, a2={a2:#010x}): \
                 the circuit reads the pointers as byte addresses, while the 3.0 guest ABI \
                 passes word addresses. Fixed upstream only in v5 (risc0#3431); this tree keeps \
                 the 3.0 ABI for compatibility. Execution is unaffected."
            );
        }
        preflight.write_pages()?;
        preflight.generate_tables()?;
        preflight.wrap_memory_txns()?;
        preflight.update_p2_zcheck()?;
        std::mem::replace(&mut preflight.word_txns, WordTxnMap::placeholder()).recycle();

        tracing::trace!("paging_cycles: {}", preflight.pager.cycles);

        Ok(preflight.trace)
    }
}

fn get_digest_addr(idx: u32) -> WordAddr {
    MERKLE_TREE_START_ADDR + DIGEST_WORDS as u32 * (2 * MEMORY_PAGES as u32 - idx)
}

impl<'a> Preflight<'a> {
    fn new(segment: &'a Segment, rand_z: ExtVal) -> Self {
        tracing::debug!("po2: {}", segment.po2);
        let total_cycles = 1 << segment.po2;

        let mut page_memory = PagedMap::default();
        for (&node_idx, digest) in segment.partial_image.digests() {
            let node_addr = node_idx_to_addr(node_idx);
            for i in 0..DIGEST_WORDS {
                page_memory.insert(&(node_addr + i), digest.as_words()[i]);
            }
        }
        // Reserve the trace up front and back it with huge pages (see `advise_huge`). SimpleLoop
        // makes ~4.2 memory transactions per cycle; a segment that makes more grows `txns`.
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        let trace = super::cuda_stage::recycled_trace(total_cycles, rand_z);
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        let trace = {
            let trace = PreflightTrace {
                cycles: Vec::with_capacity(total_cycles),
                txns: Vec::with_capacity(total_cycles * 5),
                backs: Vec::with_capacity(total_cycles),
                rand_z,
                ..Default::default()
            };
            super::advise_huge(&trace.cycles);
            super::advise_huge(&trace.txns);
            super::advise_huge(&trace.backs);
            trace
        };
        Self {
            trace,
            segment,
            pager: PagedMemory::new(
                segment.partial_image.clone(),
                false, /* tracing_enabled */
            ),
            pc: ByteAddr(0),
            machine_mode: 0,
            cur_write: 0,
            cur_read: 0,
            txn_idx: 0,
            bigint_idx: 0,
            user_cycle: 0,
            user_cycles: 0,
            word_txns: WordTxnMap::take(),
            page_memory,
            unprovable_poseidon2: None,
        }
    }

    // Do page in
    pub fn read_pages(&mut self) -> Result<()> {
        self.read_root()?;
        let activity = self.pager.loaded_pages();
        Poseidon2::read_start(self)?;
        for node_idx in activity.nodes {
            Poseidon2::read_node(self, node_idx)?;
        }
        self.machine_mode = 1;
        for page_idx in activity.pages {
            Poseidon2::read_page(self, page_idx)?;
        }
        self.machine_mode = 2;
        Poseidon2::read_done(self)?;
        self.user_cycles = 0;
        Ok(())
    }

    // Run main execution
    pub fn body(&mut self) -> Result<()> {
        let mut emu = Emulator::new();
        Risc0Machine::resume(self)?;
        while self.user_cycles < self.segment.suspend_cycle {
            Risc0Machine::step(&mut emu, self)?;
        }
        tracing::debug!(
            "suspend(cycle: {}:{}, pc: {:?}, mode: {})",
            self.trace.cycles.len(),
            self.user_cycles,
            self.pc,
            self.machine_mode
        );
        Risc0Machine::suspend(self)
    }

    // Do page out
    pub fn write_pages(&mut self) -> Result<()> {
        let activity = self.pager.dirty_pages();
        self.pager.commit();
        Poseidon2::write_start(self)?;
        for &page_idx in activity.pages.iter().rev() {
            Poseidon2::write_page(self, page_idx)?;
        }
        self.machine_mode = 4;
        for &node_idx in activity.nodes.iter().rev() {
            Poseidon2::write_node(self, node_idx)?;
        }
        self.machine_mode = 5;
        Poseidon2::write_done(self)?;
        self.machine_mode = 0;
        self.write_root()
    }

    // Do table reification
    pub fn generate_tables(&mut self) -> Result<()> {
        self.trace.table_split_cycle = self.trace.cycles.len() as u32;
        self.fini()
    }

    // Now, go back and update memory transactions to wrap around
    fn wrap_memory_txns(&mut self) -> Result<()> {
        for txn in self.trace.txns.iter_mut() {
            // tracing::trace!("{txn:?}");
            let addr = WordAddr(txn.addr);
            // The final cycle of this address and its original value (not modified here).
            let wt = *self.word_txns.get(&addr).unwrap();
            let last_cycle = wt.prev;
            if last_cycle == u32::MAX {
                // (The old prev_cycle map's `get(..).unwrap()` panicked here too.)
                bail!("memory transaction at {addr:?} without a final cycle");
            }
            if txn.prev_cycle == u32::MAX {
                // If first cycle for a particular address, set 'prev_cycle' to final cycle
                txn.prev_cycle = last_cycle;
            } else {
                // Otherwise, compute cycle diff and another diff
                ensure!(txn.cycle != txn.prev_cycle);
                let diff = txn.cycle - 1 - txn.prev_cycle;
                self.trace.cycles[(diff / 2) as usize].diff_count[(diff % 2) as usize] += 1;
            }

            // If last cycle, set final value to original value
            if txn.cycle == last_cycle {
                txn.word = if wt.has_orig { wt.orig } else { 0 };
            }
        }
        Ok(())
    }

    fn update_p2_zcheck(&mut self) -> Result<()> {
        let mut checksum = Checksum::new(&self.trace.rand_z);
        for (row, back) in self.trace.backs.iter().enumerate() {
            if let Back::Poseidon2(p2_idx) = *back {
                let p2_state = &mut self.trace.p2_states[p2_idx as usize];
                let cycle = &self.trace.cycles[row];
                let next_cycle = &self.trace.cycles[row + 1];
                let state = CycleState::from_u32((cycle.major as u32 - 7) * 8 + cycle.minor as u32)
                    .unwrap();
                if state == CycleState::PoseidonLoadIn {
                    checksum.start();
                    for (i, txn_idx) in (cycle.txn_idx..next_cycle.txn_idx).enumerate() {
                        let txn = &self.trace.txns[txn_idx as usize];
                        checksum.add(p2_state.load_tx_type, i, txn);
                    }
                }
                match state {
                    CycleState::PoseidonLoadIn
                    | CycleState::PoseidonExtRound
                    | CycleState::PoseidonIntRound => {
                        p2_state.zcheck = checksum.zcheck;
                    }
                    _ => {
                        checksum.clear();
                    }
                }
            }
        }

        Ok(())
    }

    fn fini(&mut self) -> Result<()> {
        let start_cycles = self.trace.cycles.len();

        for i in (16..256).step_by(16) {
            self.add_cycle_special(
                CycleState::ControlTable,
                CycleState::ControlTable,
                i,
                0,
                Back::None,
            );
        }
        self.machine_mode = 1;
        for i in (0..64 * 1024).step_by(16) {
            self.add_cycle_special(
                CycleState::ControlTable,
                CycleState::ControlTable,
                i,
                0,
                Back::None,
            );
        }
        self.machine_mode = 0;
        self.add_cycle_special(
            CycleState::ControlTable,
            CycleState::ControlDone,
            0,
            0,
            Back::None,
        );

        if self.segment.claim.terminate_state.is_none() {
            let segment_threshold = self.segment.segment_threshold as usize;
            if self.trace.cycles.len() < segment_threshold {
                bail!("Stopping segment too early");
            }
            let diff = self.trace.cycles.len() - segment_threshold;
            self.trace.cycles[diff / 2].diff_count[diff % 2] += 1;
        }

        self.machine_mode = 1;
        self.add_cycle_special(
            CycleState::ControlDone,
            CycleState::ControlDone,
            0,
            0,
            Back::None,
        );
        assert_eq!(self.trace.cycles.len() - start_cycles, RESERVED_CYCLES);

        let last_cycle = 1 << self.segment.po2;
        for _ in self.trace.cycles.len()..last_cycle {
            self.add_cycle_special(
                CycleState::ControlDone,
                CycleState::ControlDone,
                0,
                0,
                Back::None,
            );
        }
        Ok(())
    }

    fn read_povw_nonce(&mut self) -> Result<()> {
        let addr = POVW_NONCE_START_ADDR;
        for i in 0..DIGEST_WORDS {
            self.load_u32(LoadOp::Record, addr + i)?;
        }
        self.add_cycle_special(
            CycleState::LoadRootAndNonce,
            CycleState::LoadRootAndNonce,
            0,
            0,
            Back::None,
        );
        Ok(())
    }

    fn read_root(&mut self) -> Result<()> {
        let addr = get_digest_addr(1);
        for i in 0..DIGEST_WORDS {
            self.load_u32(LoadOp::Record, addr + i)?;
        }
        self.add_cycle_special(
            CycleState::LoadRootAndNonce,
            CycleState::PoseidonEntry,
            0,
            0,
            Back::None,
        );
        Ok(())
    }

    fn write_root(&mut self) -> Result<()> {
        let addr = get_digest_addr(1);
        for i in 0..DIGEST_WORDS {
            self.load_u32(LoadOp::Record, addr + i)?;
        }
        self.add_cycle_special(
            CycleState::StoreRoot,
            CycleState::ControlTable,
            0,
            0,
            Back::None,
        );
        Ok(())
    }

    fn add_cycle(
        &mut self,
        state: CycleState,
        pc: u32,
        major: u8,
        minor: u8,
        paging_idx: u32,
        back: Back,
    ) {
        let cycle = RawPreflightCycle {
            state: state as u32,
            pc,
            major,
            minor,
            machine_mode: self.machine_mode as u8,
            padding: 0,
            user_cycle: self.user_cycle,
            txn_idx: self.txn_idx,
            paging_idx,
            bigint_idx: self.bigint_idx,
            diff_count: [0, 0],
        };
        // tracing::trace!("[{}]: {cycle:?}", self.trace.cycles.len());
        self.trace.cycles.push(cycle);
        self.trace.backs.push(back);
        self.txn_idx = self.trace.txns.len() as u32;
        self.bigint_idx = self.trace.bigint_bytes.len() as u32;
    }

    fn add_witness(&mut self, bytes: &BigIntBytes) {
        self.trace.bigint_bytes.extend_from_slice(bytes);
    }

    fn add_cycle_insn(&mut self, state: CycleState, pc: u32, insn: InsnKind) {
        match insn {
            InsnKind::Eany => {
                // Technically we need to switch on the machine mode *entering* the EANY
                if self.trace.cycles.last().unwrap().machine_mode != 0 {
                    self.add_cycle(
                        state,
                        pc,
                        major::ECALL0,
                        ecall_minor::MACHINE_ECALL,
                        0,
                        Back::None,
                    );
                } else {
                    self.add_cycle(
                        state,
                        pc,
                        major::CONTROL0,
                        control_minor::USER_ECALL,
                        0,
                        Back::None,
                    );
                }
            }
            InsnKind::Mret => {
                self.add_cycle(
                    state,
                    pc,
                    major::CONTROL0,
                    control_minor::MRET,
                    0,
                    Back::None,
                );
            }
            _ => {
                self.add_cycle(state, pc, insn.major(), insn.minor(), 0, Back::None);
            }
        }
    }

    fn add_cycle_special(
        &mut self,
        cur_state: CycleState,
        next_state: CycleState,
        pc: u32,
        paging_idx: u32,
        back: Back,
    ) {
        let raw_cur_state = cur_state as u32;
        let major = (7 + raw_cur_state / 8) as u8;
        let minor = (raw_cur_state % 8) as u8;
        // tracing::trace!("add_cycle_special(cur_state: {cur_state:?}, next_state: {next_state:?}, major: {major}, minor: {minor})");
        self.add_cycle(next_state, pc, major, minor, paging_idx, back);
    }

    pub(crate) fn on_bigint_cycle(&mut self, cur_state: CycleState, bigint: &BigIntState) {
        self.add_witness(&bigint.bytes);
        let back = Back::BigInt(push_idx(&mut self.trace.bigint_states, bigint));
        self.add_cycle_special(
            cur_state,
            bigint.next_state,
            self.pc.0,
            0, //paging_idx
            back,
        );
        self.user_cycles += 1;
    }
}

impl Risc0Context for Preflight<'_> {
    fn get_pc(&self) -> ByteAddr {
        self.pc
    }

    fn set_pc(&mut self, addr: ByteAddr) {
        self.pc = addr;
    }

    fn set_user_pc(&mut self, _addr: ByteAddr) {
        // no-op
    }

    fn get_machine_mode(&self) -> u32 {
        self.machine_mode
    }

    fn set_machine_mode(&mut self, mode: u32) {
        tracing::trace!("[{}] set_machine_mode({mode})", self.trace.cycles.len());
        self.machine_mode = mode;
    }

    fn resume(&mut self) -> Result<()> {
        self.add_cycle_special(
            CycleState::Resume,
            CycleState::Resume,
            self.pc.0,
            0,
            Back::None,
        );
        let input_words = self.segment.claim.input.as_words();
        for (i, word) in input_words.iter().enumerate() {
            self.store_u32(GLOBAL_INPUT_ADDR.waddr() + i, *word)?;
        }
        self.add_cycle_special(
            CycleState::Resume,
            CycleState::Decode,
            self.pc.0,
            0,
            Back::None,
        );
        Ok(())
    }

    fn suspend(&mut self) -> Result<()> {
        self.pc = ByteAddr(0);
        self.add_cycle_special(CycleState::Suspend, CycleState::Suspend, 0, 0, Back::None);
        for i in 0..DIGEST_WORDS {
            self.load_u32(LoadOp::Record, GLOBAL_OUTPUT_ADDR.waddr() + i)?;
        }
        self.machine_mode = 3;
        self.add_cycle_special(
            CycleState::Suspend,
            CycleState::PoseidonEntry,
            0,
            0,
            Back::None,
        );
        Ok(())
    }

    fn on_insn_start(
        &mut self,
        kind: InsnKind,
        decoded: &crate::execute::rv32im::DecodedInstruction,
    ) -> Result<()> {
        tracing::trace!(
            "[{}]: {:?}> {}",
            self.trace.cycles.len(),
            self.pc,
            disasm(kind, decoded)
        );
        Ok(())
    }

    fn on_insn_end(&mut self, kind: InsnKind) -> Result<()> {
        self.add_cycle_insn(CycleState::Decode, self.pc.0, kind);
        self.user_cycle += 1;
        self.user_cycles += 1;
        Ok(())
    }

    fn trap_rewind(&mut self) {
        self.trace.txns.truncate(self.txn_idx as usize);
    }

    // Pass memory ops to pager + record
    #[inline(always)]
    fn load_u32(&mut self, op: LoadOp, addr: WordAddr) -> Result<u32> {
        if op == LoadOp::Peek {
            return self.pager.peek(addr);
        }

        // tracing::trace!("load_u32: {addr:?}");
        let cycle = (2 * self.trace.cycles.len()) as u32;
        // MERKLE_TREE_START_ADDR is the first address in a special region of memory that is
        // outside of the user-addressible range. This region also contains the PoVW nonce.
        let word = if addr >= MERKLE_TREE_START_ADDR {
            if addr < MERKLE_TREE_END_ADDR {
                self.page_memory
                    .get(&addr)
                    .ok_or_else(|| anyhow!("Invalid load from page memory"))?
            } else if addr >= POVW_NONCE_START_ADDR && addr < POVW_NONCE_END_ADDR {
                self.segment.povw_nonce.unwrap_or_default().to_u32s()
                    [(addr - POVW_NONCE_START_ADDR).0 as usize]
            } else {
                bail!("invalid memory access in special region: addr = {addr:x?}")
            }
        } else {
            self.pager.load(addr)?
        };
        if op == LoadOp::Record {
            let wt = self.word_txns.get_mut(&addr);
            if !wt.has_orig {
                wt.orig = word;
                wt.has_orig = true;
            }
            let prev_cycle = std::mem::replace(&mut wt.prev, cycle);
            let txn = RawMemoryTransaction {
                addr: addr.0,
                cycle,
                word,
                prev_cycle,
                prev_word: word,
            };
            // tracing::trace!("txn: {txn:?}");
            self.trace.txns.push(txn);
        }
        Ok(word)
    }

    #[inline(always)]
    fn store_u32(&mut self, addr: WordAddr, word: u32) -> Result<()> {
        let cycle = (2 * self.trace.cycles.len() + 1) as u32;
        let prev_word = if addr >= MEMORY_END_ADDR {
            self.page_memory
                .insert(&addr, word)
                .ok_or_else(|| anyhow!("Invalid store to page memory"))?
        } else {
            let prev_word = self.pager.load(addr)?;
            self.pager.store(addr, word)?;
            prev_word
        };
        let prev_cycle = std::mem::replace(&mut self.word_txns.get_mut(&addr).prev, cycle);
        let txn = RawMemoryTransaction {
            addr: addr.0,
            cycle,
            word,
            prev_cycle,
            prev_word,
        };
        // tracing::trace!("txn: {txn:?}");
        self.trace.txns.push(txn);
        Ok(())
    }

    fn on_ecall_cycle(
        &mut self,
        cur_state: CycleState,
        next_state: CycleState,
        s0: u32,
        s1: u32,
        s2: u32,
        _kind: EcallKind,
    ) -> Result<()> {
        self.add_cycle_special(cur_state, next_state, self.pc.0, 0, Back::Ecall(s0, s1, s2));
        self.user_cycles += 1;
        Ok(())
    }

    fn on_terminate(&mut self, _a0: u32, _a1: u32) -> Result<()> {
        // no-op
        Ok(())
    }

    fn host_read(&mut self, _fd: u32, buf: &mut [u8]) -> Result<u32> {
        if self.cur_read >= self.segment.read_record.len() {
            bail!("Invalid segment: unexpected read record");
        }
        let record = &self.segment.read_record[self.cur_read];
        let rlen = record.len();
        if rlen > buf.len() {
            bail!("Invalid segment: truncated read record");
        }
        buf[..rlen].copy_from_slice(record);
        self.cur_read += 1;
        Ok(rlen as u32)
    }

    fn host_write(&mut self, _fd: u32, _buf: &[u8]) -> Result<u32> {
        if self.cur_write >= self.segment.write_record.len() {
            bail!("Invalid segment: unexpected write record");
        }
        self.cur_write += 1;
        Ok(self.segment.write_record[self.cur_write])
    }

    fn on_sha2_cycle(&mut self, cur_state: CycleState, sha2: &Sha2State) {
        let back = Back::Sha2(push_idx(&mut self.trace.sha2_states, sha2));
        self.add_cycle_special(
            cur_state,
            sha2.next_state,
            self.pc.0,
            node_addr_to_idx(sha2.state_out_addr),
            back,
        );
        self.user_cycles += 1;
    }

    fn on_poseidon2_cycle(&mut self, cur_state: CycleState, p2: &Poseidon2State) {
        // First cycle of a guest ecall (mode 1), before any load: the state
        // still holds the raw register values. Only null pointers are read the
        // same way by the executor (word addresses) and the circuit (byte
        // addresses); see the error in `preflight`.
        if p2.mode == 1
            && cur_state == CycleState::PoseidonEntry
            && self.unprovable_poseidon2.is_none()
            && (p2.state_addr != 0 || p2.buf_out_addr != 0 || (p2.buf_in_addr != 0 && p2.count > 0))
        {
            self.unprovable_poseidon2 =
                Some((self.pc.0, p2.state_addr, p2.buf_in_addr, p2.buf_out_addr));
        }
        let back = Back::Poseidon2(push_idx(&mut self.trace.p2_states, p2));
        self.add_cycle_special(
            cur_state,
            p2.next_state,
            self.pc.0,
            node_addr_to_idx(WordAddr(p2.buf_out_addr)),
            back,
        );
        self.user_cycles += 1;
    }

    fn ecall_bigint(&mut self) -> Result<()> {
        super::bigint::ecall_preflight(self)
    }
}

/// Append a copy of `state` to a side vector and return its index (see [`Back`]).
fn push_idx<T: Clone>(v: &mut Vec<T>, state: &T) -> u32 {
    v.push(state.clone());
    (v.len() - 1) as u32
}

impl InsnKind {
    pub fn major(&self) -> u8 {
        (*self as u32 / 8) as u8
    }

    pub fn minor(&self) -> u8 {
        (*self as u32 % 8) as u8
    }
}

#[derive(Clone, Default, Debug)]
pub(crate) struct PagingActivity {
    pub pages: BTreeSet<u32>,
    pub nodes: BTreeSet<u32>,
}

impl PagingActivity {
    fn new(pages: BTreeSet<u32>) -> Self {
        let mut nodes = BTreeSet::new();
        for &page_idx in pages.iter() {
            let mut node_idx = node_idx(page_idx);
            while node_idx != 1 {
                let parent_idx = node_idx / 2;
                // tracing::trace!("add node: {node_idx:#010x}, parent_idx: {parent_idx:#010x}");
                if !nodes.insert(parent_idx) {
                    break;
                }
                node_idx = parent_idx;
            }
        }
        Self { pages, nodes }
    }
}

impl PagedMemory {
    pub(crate) fn loaded_pages(&self) -> PagingActivity {
        tracing::trace!("loaded_pages: {:#010x?}", self.image.get_page_indexes());
        PagingActivity::new(self.image.get_page_indexes())
    }

    pub(crate) fn dirty_pages(&self) -> PagingActivity {
        let pages = self
            .page_states
            .iter()
            .filter(|(node_idx, state)| {
                *node_idx >= MEMORY_PAGES as u32 && *state == PageState::Dirty
            })
            .map(|(node_idx, _)| page_idx(node_idx))
            .collect();
        PagingActivity::new(pages)
    }
}
