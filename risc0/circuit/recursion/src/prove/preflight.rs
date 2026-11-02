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

use std::collections::{HashMap, VecDeque};

use anyhow::{bail, ensure, Result};
use risc0_circuit_recursion_sys::RawPreflightCycle;
use risc0_core::field::baby_bear::{BabyBearElem as Fp, BabyBearExtElem as FpExt};
use risc0_zkp::{
    core::{
        digest::{DIGEST_SHORTS, DIGEST_WORDS, WORD_SIZE},
        hash::{poseidon2::CELLS, sha::SHA256_INIT},
    },
    field::{Elem as _, ExtElem},
    layout,
};
use sha2::digest::generic_array::GenericArray;

use crate::layout::{RecursionMicroInstLayout, CODE_LAYOUT};

const CHECKED_COEFFS_PER_POLY: usize = 16;
const BABY_BEAR_TO_MONTGOMERY: u32 = 0xFFFFFFE;
const BABY_BEAR_FROM_MONTGOMERY: u32 = 0x38400000;
const OPERANDS: usize = 3;
const OPS_PER_CYCLE: usize = 3;

// Values for micro inst "opcode"
mod micro_op {
    pub const CONST: u32 = 0;
    pub const ADD: u32 = 1;
    pub const SUB: u32 = 2;
    pub const MUL: u32 = 3;
    pub const INV: u32 = 4;
    pub const EQ: u32 = 5;
    pub const READ_IOP_HEADER: u32 = 6;
    pub const READ_IOP_BODY: u32 = 7;
    pub const MIX_RNG: u32 = 8;
    pub const SELECT: u32 = 9;
    pub const EXTRACT: u32 = 10;
}

#[derive(Default)]
pub(crate) struct PreflightTrace {
    // Contents of the write-only memory
    pub wom: Vec<FpExt>,
    pub cycles: Vec<RawPreflightCycle>,
    pub iops: Vec<FpExt>,
}

/// Input words that arrive while the preflight runs (streaming join preflight). The
/// preflight blocks when it needs more words than it holds until `push` or `close`.
#[derive(Default)]
pub struct InputStream {
    q: std::sync::Mutex<(VecDeque<u32>, bool)>,
    cv: std::sync::Condvar,
}

impl InputStream {
    pub fn push(&self, words: &[u32]) {
        let mut g = self.q.lock().unwrap();
        g.0.extend(words.iter().copied());
        self.cv.notify_all();
    }
    pub fn close(&self) {
        let mut g = self.q.lock().unwrap();
        g.1 = true;
        self.cv.notify_all();
    }
    /// Move at least `n` words (or all, once closed) into `into`; returns the wait in ns.
    fn pull(&self, into: &mut VecDeque<u32>, n: usize) -> u64 {
        let t = std::time::Instant::now();
        let mut g = self.q.lock().unwrap();
        while into.len() + g.0.len() < n && !g.1 {
            g = self.cv.wait(g).unwrap();
        }
        into.extend(g.0.drain(..));
        t.elapsed().as_nanos() as u64
    }
}

pub struct Preflight {
    pub(crate) trace: PreflightTrace,
    input: VecDeque<u32>,
    stream: Option<std::sync::Arc<InputStream>>,
    /// Nanoseconds spent blocked on `stream`.
    pub stream_wait_ns: u64,
    /// Words consumed from the input so far.
    pub consumed: usize,

    poseidon2_state: [Fp; CELLS],
    sha_state: [u32; DIGEST_WORDS],
    sha_load: [u32; 16],
    sha_init_pos: usize,
    sha_load_pos: usize,
    sha_fini_pos: usize,

    // Cycles, and what checked byte inputs were read during them (stored in u32s in little endian order)
    byte_reads: HashMap<usize, Vec<u32>>,

    // All the data written by the program
    pub(crate) output: Vec<u32>,

    // Cached powers of evaluation point for checked bytes.  Index is the WOM address of the evaluation
    // point.
    eval_pts: HashMap<u32, [FpExt; CHECKED_COEFFS_PER_POLY]>,

    cur_iop_body: VecDeque<Vec<Fp>>,
    iop_idx: u32,

    /// Host-computed witness (data, global) from `Prover::prepare` (Intel
    /// RISC0_RECURSION_PREWITGEN): lets the CPU witgen run on the pipelining
    /// thread instead of the GPU thread.
    pub(crate) host_witness: std::sync::Mutex<Option<(Vec<Fp>, Vec<Fp>)>>,

    /// Rows of the long sequential witgen chains, executed on the host (see
    /// `witgen::host_chains`), so the GPU witgen does not run them one thread at a time.
    pub(crate) host_chains: std::sync::Mutex<Option<super::witgen::HostChains>>,

    /// Page-locked copy of the trace and host chains (see `host_prep_enabled`).
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    pub(crate) pinned: std::sync::Mutex<Option<super::pinned::PinnedTrace>>,
}

impl Preflight {
    pub fn new(input: VecDeque<u32>) -> Self {
        Preflight {
            trace: PreflightTrace::default(),
            input,
            stream: None,
            stream_wait_ns: 0,
            consumed: 0,

            poseidon2_state: [Fp::ZERO; CELLS],
            sha_state: [0; 8],
            sha_load: [0; 16],
            sha_init_pos: 0,
            sha_load_pos: 0,
            sha_fini_pos: 0,

            byte_reads: HashMap::new(),
            output: Vec::new(),
            eval_pts: HashMap::new(),

            cur_iop_body: VecDeque::new(),
            iop_idx: 0,
            host_witness: std::sync::Mutex::new(None),
            host_chains: std::sync::Mutex::new(None),
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            pinned: std::sync::Mutex::new(None),
        }
    }

    /// Streaming variant: `input` is the prefix available now, the rest arrives on `stream`.
    pub fn new_streaming(input: VecDeque<u32>, stream: std::sync::Arc<InputStream>) -> Self {
        let mut p = Self::new(input);
        p.stream = Some(stream);
        p
    }

    pub fn is_streaming(&self) -> bool {
        self.stream.is_some()
    }

    /// Make `n` input words available (waiting for the stream if there is one). Fails if the
    /// input ends first (e.g. the stream was closed early on its producer's error path).
    fn need(&mut self, n: usize) -> Result<()> {
        if self.input.len() < n {
            if let Some(s) = &self.stream {
                self.stream_wait_ns += s.pull(&mut self.input, n);
            }
        }
        ensure!(
            self.input.len() >= n,
            "recursion preflight: input ended after {} words, {n} more needed",
            self.consumed
        );
        self.consumed += n;
        Ok(())
    }

    fn read_input_word(&mut self) -> Result<u32> {
        self.need(1)?;
        Ok(self.input.pop_front().unwrap())
    }

    fn read_iop_header(&mut self, count: Fp, k_flip: Fp) -> Result<()> {
        let count = count.as_u32() as usize;
        let k_flip = k_flip.as_u32();
        let k = k_flip as usize / 2;
        let flip = (k_flip & 1) == 1;
        tracing::trace!("read_iop_header(count: {count}, k: {k}, flip: {flip})");

        assert!(self.cur_iop_body.is_empty());
        self.need(if k == 2 { count } else { k * count })?;
        if k == 2 {
            self.cur_iop_body.extend(
                self.input
                    .drain(..count)
                    .map(|elem| vec![Fp::new(elem & 0xffff), Fp::new(elem >> 16)]),
            );
        } else {
            let arr: Vec<u32> = self.input.drain(..k * count).collect();
            for i in 0..count {
                let poly: Vec<Fp> = (0..k)
                    .map(|j| {
                        Fp::new_raw(if flip {
                            arr[i * k + j]
                        } else {
                            arr[j * count + i]
                        })
                    })
                    .collect();
                self.cur_iop_body.push_back(poly);
            }
        }
        Ok(())
    }

    fn read_iop_body(&mut self, do_mont: Fp) -> FpExt {
        let mut front = self.cur_iop_body.pop_front().unwrap();
        front.resize(FpExt::EXT_SIZE, Fp::ZERO);
        if do_mont != Fp::ZERO {
            for elem in front.iter_mut().take(FpExt::EXT_SIZE) {
                *elem *= Fp::from(BABY_BEAR_TO_MONTGOMERY);
            }
        }
        FpExt::from_subelems(front)
    }

    fn wom_read(&self, addr: Fp) -> FpExt {
        self.trace.wom[addr.as_u32() as usize]
    }

    fn wom_write(&mut self, addr: Fp, val: FpExt) -> Result<()> {
        let addr = addr.as_u32() as usize;

        // assert_eq!(addr, self.wom.len());
        if self.trace.wom.len() <= addr {
            self.trace.wom.resize(addr + 1, FpExt::ZERO);
        }
        let mem_contents = &mut self.trace.wom[addr];
        if *mem_contents != FpExt::ZERO && *mem_contents != val {
            bail!("WOM {addr} overwritten with {val:?} from {mem_contents:?}");
        }

        self.trace.wom[addr] = val;
        Ok(())
    }

    fn get(&self, code: &[Fp], reg: &layout::Reg) -> Fp {
        code[reg.offset]
    }

    pub fn step(&mut self, cycle: usize, code: &[Fp]) -> Result<()> {
        tracing::trace!("top[{cycle}]: {code:?}");
        let is_par_safe = if self.get(code, CODE_LAYOUT.code.select.macro_ops) == Fp::ONE {
            self.macro_op(cycle, code)?
        } else if self.get(code, CODE_LAYOUT.code.select.micro_ops) == Fp::ONE {
            self.micro_ops(cycle, code)?
        } else if self.get(code, CODE_LAYOUT.code.select.checked_bytes) == Fp::ONE {
            self.checked_bytes(cycle, code)?
        } else if self.get(code, CODE_LAYOUT.code.select.poseidon2_load) == Fp::ONE {
            self.poseidon2_load(cycle, code)?
        } else if self.get(code, CODE_LAYOUT.code.select.poseidon2_full) == Fp::ONE {
            self.poseidon2_full(cycle)?
        } else if self.get(code, CODE_LAYOUT.code.select.poseidon2_partial) == Fp::ONE {
            self.poseidon2_partial(cycle)?
        } else if self.get(code, CODE_LAYOUT.code.select.poseidon2_store) == Fp::ONE {
            self.poseidon2_store(cycle, code)?
        } else {
            bail!("Illegal recursion op");
        };

        self.trace.cycles.push(RawPreflightCycle {
            iop_idx: self.iop_idx,
            is_par_safe: is_par_safe as u32,
        });

        self.iop_idx = self.trace.iops.len() as u32;

        Ok(())
    }

    fn checked_bytes(&mut self, cycle: usize, code: &[Fp]) -> Result<bool> {
        let inst = CODE_LAYOUT.code.inst.checked_bytes;
        let prep_full = self.get(code, inst.prep_full).as_u32();
        let keep_coeffs = self.get(code, inst.keep_coeffs).as_u32();
        let keep_upper_state = self.get(code, inst.keep_upper_state).as_u32();
        if keep_coeffs == 1 {
            for c in &mut self.poseidon2_state[..DIGEST_SHORTS] {
                *c *= Fp::new(256);
            }
        } else {
            self.poseidon2_state[..DIGEST_SHORTS].fill(Fp::ZERO);
        }
        if keep_upper_state != 1 {
            self.poseidon2_state[DIGEST_SHORTS..].fill(Fp::ZERO);
        }

        let eval_pt_addr = self.get(code, inst.eval_point);
        tracing::trace!("Checked bytes: eval_pt={eval_pt_addr:?} prep_full = {prep_full} keep_coeffs = {keep_coeffs} keep_upper_state = {keep_upper_state}");

        let write_addr = self.get(code, CODE_LAYOUT.code.write_addr);
        let mut evaluated = FpExt::ZERO;
        let eval_pt = self.eval_pts.get(&eval_pt_addr.as_u32());
        let eval_pt_pows = match eval_pt {
            Some(eval_pt_pows) => *eval_pt_pows,
            None => {
                let eval_pt = self.wom_read(eval_pt_addr);
                let mut pows = [FpExt::ONE; CHECKED_COEFFS_PER_POLY];
                for i in 1..CHECKED_COEFFS_PER_POLY {
                    pows[i] = pows[i - 1] * eval_pt;
                }
                tracing::trace!("Caching eval pt {eval_pt:?} powers: {pows:?}");
                self.eval_pts.insert(eval_pt_addr.as_u32(), pows);
                pows
            }
        };
        let mut cycle_input: Vec<u32> = Vec::new();
        for i in 0..WORD_SIZE {
            let word = self.read_input_word()?;
            let bytes = word.to_le_bytes();
            for (j, b) in bytes.iter().enumerate() {
                let elem_idx = i * WORD_SIZE + j;
                let val = Fp::new(*b as u32);
                tracing::trace!("  loading[{elem_idx}]: {val:?}");
                self.poseidon2_state[elem_idx] += val;
                evaluated += val * eval_pt_pows[elem_idx];
            }
            cycle_input.push(word);
        }
        let old_elem = self.byte_reads.insert(cycle, cycle_input);
        assert!(
            old_elem.is_none(),
            "Duplicate cycle reads for checked bytes cycle"
        );
        self.wom_write(write_addr, evaluated)?;
        Ok(false)
    }

    fn poseidon2_load(&mut self, _cycle: usize, code: &[Fp]) -> Result<bool> {
        let inst = CODE_LAYOUT.code.inst.poseidon2_load;
        let do_mont = self.get(code, inst.do_mont).as_u32();
        let prep_full = self.get(code, inst.prep_full).as_u32();
        let keep_state = self.get(code, inst.keep_state).as_u32();
        let keep_upper_state = self.get(code, inst.keep_upper_state).as_u32();
        let g1 = self.get(code, inst.group.g1).as_u32();
        let g2 = self.get(code, inst.group.g2).as_u32();
        let group = (g1 + g2 * 2) as usize;
        tracing::trace!(
            "Poseidon2 Load: group: {group}, prep_full: {prep_full}, keep_state: {keep_state}, keep_upper_state: {keep_upper_state}, do_mont: {do_mont}",
        );
        if keep_state != 1 {
            if keep_upper_state != 1 {
                self.poseidon2_state = [Fp::ZERO; CELLS];
            } else {
                for i in 0..DIGEST_SHORTS {
                    self.poseidon2_state[i] = Fp::ZERO;
                }
            }
        }
        for i in 0..DIGEST_WORDS {
            let addr = self.get(code, inst.inputs[i]);
            let mut load = self.wom_read(addr).elems()[0];
            if do_mont != 0 {
                load *= Fp::from(BABY_BEAR_FROM_MONTGOMERY);
            }
            tracing::trace!("  loading[{i}]: {load:?}");
            self.poseidon2_state[group * DIGEST_WORDS + i] += load;
        }
        Ok(false)
    }

    fn poseidon2_full(&mut self, _cycle: usize) -> Result<bool> {
        tracing::trace!("Poseidon2 full");
        Ok(false)
    }

    fn poseidon2_partial(&mut self, _cycle: usize) -> Result<bool> {
        tracing::trace!("Poseidon2 partial");
        super::p2fast::poseidon2_mix_fast(&mut self.poseidon2_state);
        Ok(false)
    }

    fn poseidon2_store(&mut self, _cycle: usize, code: &[Fp]) -> Result<bool> {
        let inst = CODE_LAYOUT.code.inst.poseidon2_store;
        let do_mont = self.get(code, inst.do_mont).as_u32();
        let g1 = self.get(code, inst.group.g1).as_u32();
        let g2 = self.get(code, inst.group.g2).as_u32();
        let group = (g1 + g2 * 2) as usize;
        tracing::trace!("Poseidon2 store, group = {}, doMont = {}", group, do_mont);
        let write_addr = self.get(code, CODE_LAYOUT.code.write_addr).as_u32() as usize;
        for i in 0..DIGEST_WORDS {
            let addr = Fp::new((write_addr + i) as u32);
            let mut store = self.poseidon2_state[group * DIGEST_WORDS + i];
            if do_mont != 0 {
                store *= Fp::from(BABY_BEAR_TO_MONTGOMERY);
            }
            tracing::trace!("  storing[{i}]: {store:?}");
            self.wom_write(addr, FpExt::from(store))?;
        }
        Ok(false)
    }

    fn macro_op(&mut self, cycle: usize, code: &[Fp]) -> Result<bool> {
        let macro_ops = CODE_LAYOUT.code.inst.macro_ops;
        let opcode = macro_ops.opcode;
        let args: [Fp; OPERANDS] = core::array::from_fn(|i| self.get(code, macro_ops.operand[i]));
        let args_u32: [u32; OPERANDS] = core::array::from_fn(|i| args[i].into());
        let write_addr = self.get(code, CODE_LAYOUT.code.write_addr);

        tracing::trace!("write_addr = {write_addr:?}");

        if self.get(code, opcode.bit_and_elem) == Fp::ONE {
            self.bit_and_elem(args, write_addr)
        } else if self.get(code, opcode.bit_op_shorts) == Fp::ONE {
            self.bit_op_shorts(args, args_u32, write_addr)
        } else if self.get(code, opcode.sha_init) == Fp::ONE {
            self.sha_init(cycle)
        } else if self.get(code, opcode.sha_load) == Fp::ONE {
            self.sha_load(cycle, args, args_u32)
        } else if self.get(code, opcode.sha_mix) == Fp::ONE {
            self.sha_mix(cycle)
        } else if self.get(code, opcode.sha_fini) == Fp::ONE {
            self.sha_fini(cycle, args)
        } else {
            // nop
            // wom_init
            // wom_fini
            // set_global
            Ok(true)
        }
    }

    fn bit_and_elem(&mut self, args: [Fp; OPERANDS], write_addr: Fp) -> Result<bool> {
        let in_a = self.wom_read(args[0]);
        let in_b = self.wom_read(args[1]);
        let result = FpExt::from_u32(in_a.elems()[0].as_u32() & in_b.elems()[0].as_u32());
        tracing::trace!("{in_a:?} & {in_b:?} -> {result:?}");
        self.wom_write(write_addr, result)?;
        Ok(true)
    }

    fn bit_op_shorts(
        &mut self,
        args: [Fp; OPERANDS],
        args_u32: [u32; OPERANDS],
        write_addr: Fp,
    ) -> Result<bool> {
        let in_a = self.wom_read(args[0]);
        let in_b = self.wom_read(args[1]);
        let result = if args_u32[2] != 0 {
            // AND and combine [a, b, 0, 0] & [c, d, 0, 0] -> [(a&c) + ((b&d) << 16), 0, 0,
            // 0]
            let result = FpExt::new(
                Fp::new(
                    (in_a.elems()[0].as_u32() & in_b.elems()[0].as_u32())
                        + ((in_a.elems()[1].as_u32() & in_b.elems()[1].as_u32()) << 16),
                ),
                Fp::ZERO,
                Fp::ZERO,
                Fp::ZERO,
            );
            tracing::trace!("{in_a:?} & {in_b:?} (as shorts) -> {result:?}");
            result
        } else {
            // Xors and returns 2 shorts: [a, b, 0, 0] ^ [c, d, 0, 0] -> [a ^ c, b ^ d, 0,
            // 0]
            let result = FpExt::new(
                Fp::new(in_a.elems()[0].as_u32() ^ in_b.elems()[0].as_u32()),
                Fp::new(in_a.elems()[1].as_u32() ^ in_b.elems()[1].as_u32()),
                Fp::ZERO,
                Fp::ZERO,
            );
            tracing::trace!("{in_a:?} ^ {in_b:?} -> {result:?}");
            result
        };
        self.wom_write(write_addr, result)?;
        Ok(true)
    }

    fn sha_init(&mut self, _cycle: usize) -> Result<bool> {
        if self.sha_init_pos == 0 {
            tracing::trace!("sha_init");

            for i in 0..self.sha_state.len() {
                self.sha_state[i] = SHA256_INIT.as_words()[i].to_be();
            }
        }
        self.sha_init_pos = (self.sha_init_pos + 1) % 4;
        Ok(false)
    }

    fn sha_load(
        &mut self,
        _cycle: usize,
        args: [Fp; OPERANDS],
        arg_u32: [u32; OPERANDS],
    ) -> Result<bool> {
        let io0 = self.wom_read(args[0]);
        tracing::trace!("Reading sha from wom {:?}, got {io0:?}", args[0]);
        let subtype = arg_u32[2];
        let val: u32 = if subtype == 0 {
            // Montgomery encoded.
            io0.elems()[0].as_u32_montgomery()
        } else {
            io0.elems()[0].as_u32() + (io0.elems()[1].as_u32() << 16)
        };
        tracing::trace!(
            "sha_load {:x?} (from {:?}) (subtype = {subtype})",
            val.to_be(),
            io0
        );

        self.sha_load[self.sha_load_pos] = val;
        self.sha_load_pos = (self.sha_load_pos + 1) % 16;
        Ok(false)
    }

    fn sha_mix(&mut self, _cycle: usize) -> Result<bool> {
        Ok(false)
    }

    fn sha_fini(&mut self, _cycle: usize, args: [Fp; OPERANDS]) -> Result<bool> {
        if self.sha_fini_pos == 0 {
            tracing::trace!("sha_fini: state={:x?}", self.sha_state);
            tracing::trace!("sha_fini: in={:x?}", self.sha_load);
            let u8s: &[u8] = bytemuck::cast_slice(&self.sha_load);

            sha2::compress256(&mut self.sha_state, &[*GenericArray::from_slice(u8s)]);
            tracing::trace!("sha_fini: out={:x?}", self.sha_state);
            // for (size_t i = 0; i < 4; i++) {
            //   addMacro(/*outs=*/0, MacroOpcode::SHA_FINI, out + 3 - i, out + 7 - i);
            // So, if shaFiniPos is 0, we're at the first SHA_FINI of the bunch, and
            // args[0] = out + 3 - 0, so out = args[0] - 3.
            let out_addr = args[0].as_u32() - 3;

            for i in 0u32..8 {
                let out = self.sha_state[i as usize].to_be();
                tracing::trace!("sha_fini {out:?} -> wom[{out_addr} + {i}]");
                self.wom_write(
                    Fp::new(out_addr + i),
                    FpExt::new(
                        Fp::new(out & 0xFFFF),
                        Fp::new(out >> 16),
                        Fp::ZERO,
                        Fp::ZERO,
                    ),
                )?;
            }
        }
        self.sha_fini_pos = (self.sha_fini_pos + 1) % 4;
        Ok(false)
    }

    fn micro_ops(&mut self, cycle: usize, code: &[Fp]) -> Result<bool> {
        let write_addr = self.get(code, CODE_LAYOUT.code.write_addr);
        tracing::trace!("micros -> wom[{write_addr:?}]");
        let mut is_par_safe = true;
        for i in 0..OPS_PER_CYCLE {
            is_par_safe &= self.micro_op(
                cycle,
                code,
                write_addr + Fp::new(i as u32),
                CODE_LAYOUT.code.inst.micro_ops[i],
            )?;
        }
        Ok(is_par_safe)
    }

    fn micro_op(
        &mut self,
        cycle: usize,
        code: &[Fp],
        write_addr: Fp,
        inst: &RecursionMicroInstLayout,
    ) -> Result<bool> {
        let opcode = self.get(code, inst.opcode).as_u32();
        let args: [Fp; OPERANDS] = core::array::from_fn(|i| self.get(code, inst.operand[i]));
        let args_u32: [u32; OPERANDS] = core::array::from_fn(|i| args[i].into());

        let is_par_safe = match opcode {
            micro_op::CONST => {
                let result = FpExt::new(args[0], args[1], Fp::ZERO, Fp::ZERO);
                tracing::trace!("const {result:?}");
                self.wom_write(write_addr, result)?;
                true
            }
            micro_op::ADD => {
                let a = self.wom_read(args[0]);
                let b = self.wom_read(args[1]);
                let result = a + b;
                tracing::trace!("{a:?} + {b:?} -> {result:?}");
                self.wom_write(write_addr, result)?;
                if args[2] != Fp::ZERO {
                    self.output.push(a.elems()[0].into());
                }
                true
            }
            micro_op::SUB => {
                let a = self.wom_read(args[0]);
                let b = self.wom_read(args[1]);
                let result = a - b;
                tracing::trace!("{a:?} - {b:?} -> {result:?}");
                self.wom_write(write_addr, result)?;
                true
            }
            micro_op::MUL => {
                let a = self.wom_read(args[0]);
                let b = self.wom_read(args[1]);
                let result = a * b;
                tracing::trace!("{a:?} * {b:?} -> {result:?}");
                self.wom_write(write_addr, result)?;
                true
            }
            micro_op::INV => {
                let a = self.wom_read(args[0]);
                if args_u32[1] == 0 {
                    let result = if a.elems()[0] == Fp::ZERO {
                        Fp::ONE
                    } else {
                        Fp::ZERO
                    };
                    tracing::trace!("is_zero({a:?}) -> {result:?}");
                    self.wom_write(write_addr, FpExt::from(result))?;
                } else {
                    let result = a.inv();
                    tracing::trace!("inv({a:?}) -> {result:?}");
                    self.wom_write(write_addr, a.inv())?;
                }
                true
            }
            micro_op::EQ => {
                let a = self.wom_read(args[0]);
                let b = self.wom_read(args[1]);
                tracing::trace!("eq({a:?} ({:?}), {b:?}) ({:?})", args[0], args[1]);
                if a != b {
                    bail!("Equality check failed: Expecting {a:?} == {b:?}");
                }
                true
            }
            micro_op::READ_IOP_HEADER => {
                tracing::trace!("iop header {:?} {:?}", args[0], args[1]);
                self.read_iop_header(args[0], args[1])?;
                true
            }
            micro_op::READ_IOP_BODY => {
                let body = self.read_iop_body(args[2]);
                tracing::trace!("[{cycle}]: iop body -> {write_addr:?}, {body:?}");
                self.wom_write(write_addr, body)?;
                self.trace.iops.push(body);
                true
            }
            micro_op::MIX_RNG => {
                let mut val = args[2];
                let is_par_safe = if args_u32[2] != 0 {
                    // If non-zero, multiply in "extraPrev" which was the last
                    // instruction that wrote to the previous WOM address.
                    let prev = self.wom_read(write_addr - Fp::ONE).elems()[0];
                    tracing::trace!("mix_rng prev={prev:?}");
                    val *= prev;
                    false
                } else {
                    true
                };
                let a = self.wom_read(args[0]);
                let b = self.wom_read(args[1]);
                const SHIFT_WORD: Fp = Fp::new(1 << 16);
                val = val * SHIFT_WORD + a.elems()[1];
                val = val * SHIFT_WORD + a.elems()[0];
                val = val * SHIFT_WORD + b.elems()[1];
                val = val * SHIFT_WORD + b.elems()[0];
                tracing::trace!("mix_rng -> {val:?}");
                self.wom_write(write_addr, val.into())?;
                is_par_safe
            }
            micro_op::SELECT => {
                let a = self.wom_read(args[0]);
                let val = self.wom_read(args[1] + args[2] * a.elems()[0]);
                tracing::trace!("select {a:?} -> {val:?}");
                self.wom_write(write_addr, val)?;
                true
            }
            micro_op::EXTRACT => {
                let a = self.wom_read(args[0]);
                let val = args[1] * args[2] * a.elems()[3]
                    + args[1] * (Fp::ONE - args[2]) * a.elems()[2]
                    + (Fp::ONE - args[1]) * args[2] * a.elems()[1]
                    + (Fp::ONE - args[1]) * (Fp::ONE - args[2]) * a.elems()[0];
                tracing::trace!("extract {a:?} -> {val:?}");
                self.wom_write(write_addr, val.into())?;
                true
            }
            _ => bail!("Unknown opcode"),
        };
        Ok(is_par_safe)
    }
}
