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
use risc0_binfmt::MemoryImage;
use risc0_zkp::core::digest::Digest;
use test_log::test;

use crate::{TerminateState, MAX_INSN_CYCLES};

use super::{
    platform::*,
    syscall::{Syscall, SyscallContext},
    testutil::{self, Assembler},
    DEFAULT_SEGMENT_LIMIT_PO2,
};

#[test]
fn basic() {
    let program = testutil::kernel::basic();
    let expected_cycles = program.size_in_words();
    let mut image = MemoryImage::new_kernel(program);
    let pre_image_id = image.image_id();

    println!("image_id: {pre_image_id}");

    let session = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        MAX_INSN_CYCLES,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();

    let segments = session.segments;
    assert_eq!(segments.len(), 1);
    let segment = segments.first().unwrap();
    assert_eq!(segment.claim.pre_state, pre_image_id);
    assert_ne!(segment.claim.post_state, pre_image_id);
    assert_eq!(segment.claim.input, Digest::ZERO);
    assert_eq!(segment.claim.output, Some(Digest::ZERO));
    assert_eq!(
        segment.claim.terminate_state,
        Some(TerminateState::default())
    );
    assert!(segment.read_record.is_empty());
    assert!(segment.write_record.is_empty());
    assert_eq!(segment.suspend_cycle, expected_cycles as u32 + 1);
}

#[test]
fn system_split() {
    let program = testutil::kernel::simple_loop(2000);
    let mut image = MemoryImage::new_kernel(program);
    let pre_image_id = image.image_id();

    let session = testutil::execute(
        image,
        testutil::MIN_CYCLES_PO2,
        100,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();

    let segments = session.segments;
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].claim.pre_state, pre_image_id);
    assert_ne!(segments[0].claim.post_state, pre_image_id);
    assert_eq!(segments[0].claim.input, Digest::ZERO);
    assert_eq!(segments[0].claim.output, None);
    assert_eq!(segments[0].claim.terminate_state, None);

    assert_eq!(segments[1].claim.pre_state, segments[0].claim.post_state);
    assert_ne!(segments[1].claim.post_state, segments[1].claim.pre_state);
    assert_eq!(segments[1].claim.input, Digest::ZERO);
    assert_eq!(segments[1].claim.output, Some(Digest::ZERO));
    assert_eq!(
        segments[1].claim.terminate_state,
        Some(TerminateState::default())
    );

    assert!(segments[0].read_record.is_empty());
    assert!(segments[0].write_record.is_empty());
}

/// Executes `body` in machine mode, then halts: the error, if execution refuses it.
fn run_kernel(body: impl FnOnce(&mut Assembler)) -> Result<()> {
    struct Sink;
    impl Syscall for Sink {
        fn host_read(&self, _: &mut impl SyscallContext, _: u32, _: &mut [u8]) -> Result<u32> {
            Ok(0)
        }
        fn host_write(&self, _: &mut impl SyscallContext, _: u32, buf: &[u8]) -> Result<u32> {
            Ok(buf.len() as u32)
        }
    }
    let mut asm = Assembler::new();
    body(&mut asm);
    asm.host_terminate(0, 0);
    let image = MemoryImage::new_kernel(asm.program());
    testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        MAX_INSN_CYCLES,
        testutil::DEFAULT_SESSION_LIMIT,
        &Sink,
        None,
    )
    .map(|_| ())
}

fn assert_refused(r: Result<()>, needle: &str) {
    let err = r.expect_err("executed");
    assert!(format!("{err:#}").contains(needle), "{err:#}");
}

const ECALL: u32 = 0x0000_0073;
const MRET: u32 = 0x3020_0073;

/// The circuit decodes jalr only with func3 0, and mret and ecall only as their exact words: any
/// other encoding of them is an illegal instruction here too (where it ran, no proof of it could).
#[test]
fn only_the_circuits_encodings_execute() {
    run_kernel(|_| {}).unwrap();

    // A halt through `ecall` with rd (or rs1) set.
    for word in [ECALL | (1 << 7), ECALL | (1 << 15)] {
        assert_refused(
            run_kernel(|asm| {
                asm.li(REG_A7, HOST_ECALL_TERMINATE);
                asm.li(REG_A0, 0);
                asm.li(REG_A1, 0);
                asm.raw(word);
            }),
            "IllegalInstruction",
        );
    }
    // mret with rd set.
    assert_refused(
        run_kernel(|asm| asm.raw(MRET | (1 << 7))),
        "IllegalInstruction",
    );
    // A jalr to the next instruction (after li's two and its own), with func3 1; with 0, it runs.
    let jalr = |func3: u32| {
        let next = USER_START_ADDR.0 + WORD_SIZE as u32 + 3 * WORD_SIZE as u32;
        move |asm: &mut Assembler| {
            asm.li(REG_T0, next);
            asm.raw((REG_T0 as u32) << 15 | func3 << 12 | 0b1100111);
        }
    };
    run_kernel(jalr(0)).unwrap();
    assert_refused(run_kernel(jalr(1)), "IllegalInstruction");
}

const KERNEL_CODE: u32 = KERNEL_START_ADDR.0;
const USER_CODE: u32 = ZERO_PAGE_END_ADDR.0 + 0x1_0000;

/// `addi rd, x0, imm`, `lw rd, 0(rs1)` and `sw rs2, 0(rs1)`.
fn addi(rd: usize, imm: u32) -> u32 {
    (imm << 20) | ((rd as u32) << 7) | 0x13
}
fn lw(rd: usize, rs1: usize) -> u32 {
    ((rs1 as u32) << 15) | (0b010 << 12) | ((rd as u32) << 7) | 0x03
}
fn sw(rs2: usize, rs1: usize) -> u32 {
    ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (0b010 << 12) | 0x23
}

/// Drops to user mode at `target` (mret resumes at MEPC + 4).
fn mret_to(asm: &mut Assembler, target: u32) {
    asm.li(REG_T0, MEPC_ADDR.0);
    asm.li(REG_T1, target.wrapping_sub(WORD_SIZE as u32));
    asm.raw(sw(REG_T1, REG_T0));
    asm.raw(MRET);
}

/// The circuit has no trap step. A user-mode trap with a trap table pointing at a kernel handler
/// that halts cleanly used to execute; no proof of it could verify.
#[test]
fn a_trap_is_refused() {
    assert_refused(
        run_kernel(|asm| {
            // The handler: halt, exit code 0.
            for (i, word) in [
                addi(REG_A7, HOST_ECALL_TERMINATE),
                addi(REG_A0, 0),
                addi(REG_A1, 0),
            ]
            .into_iter()
            .chain([ECALL])
            .enumerate()
            {
                asm.word(KERNEL_CODE + (i * WORD_SIZE) as u32, word);
            }
            for cause in 0..16 {
                asm.word(TRAP_DISPATCH_ADDR.0 + cause * WORD_SIZE as u32, KERNEL_CODE);
            }
            // The user code: a load from the zero page.
            asm.word(USER_CODE, addi(REG_T0, 4));
            asm.word(USER_CODE + WORD_SIZE as u32, lw(REG_T1, REG_T0));
            mret_to(asm, USER_CODE);
        }),
        "Trap cannot be proven",
    );
}

/// The circuit decodes an instruction only at an aligned pc: an mret to a misaligned one is refused
/// (fetching rounded it down).
#[test]
fn a_misaligned_pc_is_refused() {
    assert_refused(
        run_kernel(|asm| {
            asm.word(USER_CODE, addi(REG_T0, 4));
            mret_to(asm, USER_CODE + 1);
        }),
        "Misaligned pc",
    );
}

/// An unknown machine ecall is an error, not a panic.
#[test]
fn an_unknown_machine_ecall_is_refused() {
    assert_refused(
        run_kernel(|asm| {
            asm.li(REG_A7, 6);
            asm.ecall();
        }),
        "Unknown machine ecall: 6",
    );
}

fn sha2_ecall(asm: &mut Assembler, state_in: u32) {
    let base = ZERO_PAGE_END_ADDR.0 + 0x1000;
    asm.li(REG_A7, HOST_ECALL_SHA2);
    asm.li(REG_A0, state_in);
    asm.li(REG_A1, base + 0x100);
    asm.li(REG_A2, base + 0x200);
    asm.li(REG_A3, 0);
    asm.li(REG_A4, base + 0x300);
    asm.ecall();
}

/// The circuit divides a sha2 pointer by four as a field element: a misaligned one is refused.
#[test]
fn a_misaligned_sha2_pointer_is_refused() {
    let base = ZERO_PAGE_END_ADDR.0 + 0x1000;
    run_kernel(|asm| sha2_ecall(asm, base)).unwrap();
    assert_refused(
        run_kernel(|asm| sha2_ecall(asm, base + 1)),
        "Misaligned pointer in sha2 ecall",
    );
}

/// The prover's preflight replays no host write correctly: refused, after the host has it.
#[test]
fn a_host_write_is_refused() {
    assert_refused(
        run_kernel(|asm| {
            asm.li(REG_A7, HOST_ECALL_WRITE);
            asm.li(REG_A0, 2);
            asm.li(REG_A1, ZERO_PAGE_END_ADDR.0 + 0x1000);
            asm.li(REG_A2, 4);
            asm.ecall();
        }),
        "Host write (fd 2, 4 bytes) cannot be proven",
    );
}

#[test]
fn final_padding_preserves_execution_and_claim() {
    let image = MemoryImage::new_kernel(testutil::kernel::basic());
    let run = |po2| {
        testutil::execute(
            image.clone(),
            po2,
            100,
            testutil::DEFAULT_SESSION_LIMIT,
            &testutil::NullSyscall,
            None,
        )
        .unwrap()
    };
    let before = run(13);
    let after = run(14);
    assert_eq!(before.segments.len(), 1);
    assert_eq!(before.segments[0].po2, 13);
    assert_eq!(after.segments[0].po2, 14);
    let a = &before.result;
    let b = &after.result;
    assert_eq!(a.user_cycles, b.user_cycles);
    assert_eq!(a.paging_cycles, b.paging_cycles);
    assert_eq!(a.claim.pre_state, b.claim.pre_state);
    assert_eq!(a.claim.post_state, b.claim.post_state);
    assert_eq!(a.claim.input, b.claim.input);
    assert_eq!(a.claim.output, b.claim.output);
    assert_eq!(a.claim.terminate_state, b.claim.terminate_state);
    assert_eq!(b.total_cycles - a.total_cycles, 8192);
    assert_eq!(b.reserved_cycles - a.reserved_cycles, 8192);
}

#[test]
fn final_padding_at_segment_boundaries() {
    for po2 in [14, 16, 18] {
        for count in [0, 1, 4000, 6000, 10000] {
            let image = MemoryImage::new_kernel(testutil::kernel::simple_loop(count));
            let session = testutil::execute(
                image,
                po2,
                100,
                testutil::DEFAULT_SESSION_LIMIT,
                &testutil::NullSyscall,
                None,
            )
            .unwrap();
            assert!(session
                .segments
                .iter()
                .all(|s| (14..=po2 as u32).contains(&s.po2)));
            assert_eq!(
                session.result.total_cycles,
                session.segments.iter().map(|s| 1u64 << s.po2).sum::<u64>()
            );
            assert_eq!(
                session.result.total_cycles,
                session.result.user_cycles
                    + session.result.paging_cycles
                    + session.result.reserved_cycles
            );
            for pair in session.segments.windows(2) {
                assert_eq!(pair[0].claim.post_state, pair[1].claim.pre_state);
            }
        }
    }
}
