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

use risc0_binfmt::MemoryImage;
use risc0_zkp::core::digest::Digest;
use test_log::test;

use crate::{TerminateState, MAX_INSN_CYCLES};

use super::{testutil, DEFAULT_SEGMENT_LIMIT_PO2};

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


#[test]
fn final_padding_preserves_execution_and_claim() {
    let image = MemoryImage::new_kernel(testutil::kernel::basic());
    let run = |po2| testutil::execute(image.clone(), po2, 100,
        testutil::DEFAULT_SESSION_LIMIT, &testutil::NullSyscall, None).unwrap();
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
            let session = testutil::execute(image, po2, 100,
                testutil::DEFAULT_SESSION_LIMIT, &testutil::NullSyscall, None).unwrap();
            assert!(session.segments.iter().all(|s| (14..=po2 as u32).contains(&s.po2)));
            assert_eq!(session.result.total_cycles,
                session.segments.iter().map(|s| 1u64 << s.po2).sum::<u64>());
            assert_eq!(session.result.total_cycles,
                session.result.user_cycles + session.result.paging_cycles + session.result.reserved_cycles);
            for pair in session.segments.windows(2) {
                assert_eq!(pair[0].claim.post_state, pair[1].claim.pre_state);
            }
        }
    }
}
