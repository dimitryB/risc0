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

// Included first in the Intel (SYCL) keccak eval_check translation unit that build.rs generates
// (see `write_intel_translation_unit` there) from the generated CUDA constraint code
// (kernels/cuda/eval_check_<n>.cu), unchanged but for the rewrites build.rs makes. This header
// gives that code what supra/fp.h and the CUDA launch give it, in a form SYCL device code accepts:
// risc0's portable Fp/FpExt, and the scratch and polynomial mix as arguments instead of
// `__constant__` globals.

#pragma once

#include <sycl/sycl.hpp>

#include "fp.h"
#include "fpext.h"

#include <cstddef>
#include <cstdint>
#include <tuple>

#define __device__

// What build.rs makes `__device__` mean for the generated functions: real calls. Inlined into
// their kernels, they make kernels too large to compile in reasonable time (as rv32im-sys found
// for its eval_check).
#define R0_KECCAK_OUTLINE __attribute__((noinline))

namespace risc0::circuit::keccak::cuda {

constexpr size_t EXT_SIZE = 4;

/// How many work-items run an eval_check stage at once: the stride of the scratch buffer, which
/// holds the constraint code's intermediate array (`poly_fp`'s `Fp xN[M]`) instead of private
/// memory. A compile-time stride keeps every access a constant offset from the work-item's base.
constexpr uint32_t kScratchStride = 32768;

/// One work-item's slice of the scratch buffer. Element `i` of work-item `t` is at
/// `base[i * kScratchStride + t]`, so the lanes of a sub-group touching the same element touch
/// consecutive words.
struct ScratchRef {
  Fp* base;

  inline Fp& operator[](uint32_t i) const { return base[size_t(i) * kScratchStride]; }
};

/// What build.rs threads through the constraint functions in place of CUDA's `__constant__`
/// globals: the powers of the polynomial mix, this work-item's scratch, and its carry (below).
struct EvalCtx {
  const FpExt* poly_mix;
  ScratchRef scratch;
  uint32_t* carry;
};

// The constraint code is one chain of tail calls (poly_fp -> keccak_56 -> ... -> keccak_0), which
// build.rs cuts into stages, each its own kernel: compiled as one, the chain takes the GPU
// compiler tens of gigabytes. At a cut, the call to the next stage's first function becomes
// `carry_out`, which saves its arguments in this work-item's carry; the next stage's kernel calls
// that function through `call_from_carry`, which reads them back in the same order. The scratch
// itself is in global memory, and so already where the next stage finds it.

struct CarryWriter {
  uint32_t* p;

  void put(const FpExt& v) {
    for (size_t i = 0; i < EXT_SIZE; i++) {
      *p++ = v.elems[i].asRaw();
    }
  }

  void put(const Fp* v) {
    uint64_t bits = reinterpret_cast<uintptr_t>(v);
    *p++ = uint32_t(bits);
    *p++ = uint32_t(bits >> 32);
  }

  void put(const ScratchRef&) {}
};

template <typename... Args>
inline FpExt carry_out(uint32_t idx, uint32_t size, EvalCtx ectx, Args... args) {
  CarryWriter w{ectx.carry};
  (w.put(args), ...);
  return FpExt();
}

struct CarryReader {
  uint32_t* p;
  EvalCtx ectx;

  void get(FpExt& v) {
    for (size_t i = 0; i < EXT_SIZE; i++) {
      v.elems[i] = sycl::bit_cast<Fp>(*p++);
    }
  }

  void get(const Fp*& v) {
    uint64_t lo = *p++;
    uint64_t hi = *p++;
    v = reinterpret_cast<const Fp*>(uintptr_t(lo | (hi << 32)));
  }

  void get(ScratchRef& v) { v = ectx.scratch; }

  template <typename T> T next() {
    T v;
    get(v);
    return v;
  }
};

template <typename F> struct StageParams;

template <typename... P> struct StageParams<FpExt (*)(uint32_t, uint32_t, EvalCtx, P...)> {
  template <auto Fn> static FpExt call(uint32_t idx, uint32_t size, EvalCtx ectx) {
    CarryReader r{ectx.carry, ectx};
    // A braced list evaluates left to right: the order carry_out wrote them in.
    std::tuple<P...> args{r.template next<P>()...};
    return std::apply([&](P... a) { return Fn(idx, size, ectx, a...); }, args);
  }
};

template <auto Fn> inline FpExt call_from_carry(uint32_t idx, uint32_t size, EvalCtx ectx) {
  return StageParams<decltype(Fn)>::template call<Fn>(idx, size, ectx);
}

} // namespace risc0::circuit::keccak::cuda
