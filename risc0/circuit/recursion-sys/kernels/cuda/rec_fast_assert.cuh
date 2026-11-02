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

// ROCm only (build.rs includes this in the amalgam right before step_exec.cu, so it applies to the
// generated step_exec.cu, step_verify_*.cu that follow it; no include guard: it is meant to
// re-apply the definition after a later <assert.h>).
//
// The generated witgen steps check every trace read with `assert(x != Fp::invalid())` (4373 in
// step_exec.cu, 960 in step_verify_mem.cu). HIP's device assert is a call to __assert_fail, which
// the compiler does not treat as noreturn: each check became an if/else with exec-mask
// save/restore around an inline ~45-dword call block (four 64-bit string addresses, argument moves,
// s_swappc; on gfx12 also an s_wait_alu after nearly every scalar op), and every step function a
// non-leaf. The hot path was thus spread over ~4x the code (I-cache), with ~8 extra scalar/vector
// ops per check. Here a failing check traps (s_trap 2) as __assert_fail does after printing, so
// the device still aborts the dispatch (the host sees the kernel fault) on the same condition; only
// the message (file/line of the generated check) is lost. Valid traces take the same path and
// produce the same witness.
#if defined(__HIP_DEVICE_COMPILE__)
#undef assert
#define assert(e) (__builtin_expect(!!(e), 1) ? (void)0 : __builtin_trap())
#endif
