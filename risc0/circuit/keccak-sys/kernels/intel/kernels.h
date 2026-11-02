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

// Included last in the Intel (SYCL) keccak eval_check translation unit that build.rs generates,
// after the rewritten constraint code and its `eval_stage<S>`: the stage kernels and their
// launcher, on the SYCL queue the zkp HAL uses. Mirrors kernels/cuda/eval_check_scratch_kernel.cuh.

#pragma once

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <utility>

namespace risc0::circuit::keccak::cuda {

/// Device memory freed when the launcher returns (after it has waited for its kernels).
template <typename T> struct DeviceArray {
  sycl::queue& q;
  T* ptr = nullptr;

  DeviceArray(sycl::queue& q, size_t count) : q(q) {
    if (count) {
      ptr = sycl::malloc_device<T>(count, q);
      if (!ptr) {
        throw std::runtime_error("keccak eval_check: device allocation failed");
      }
    }
  }

  DeviceArray(sycl::queue& q, const T* host, size_t count) : DeviceArray(q, count) {
    if (count) {
      q.memcpy(ptr, host, count * sizeof(T));
    }
  }

  ~DeviceArray() {
    if (ptr) {
      sycl::free(ptr, q);
    }
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
};

template <uint32_t S> class KeccakEvalStage;

constexpr uint32_t kEvalCheckGroup = 128;

/// What every eval_check stage kernel reads.
struct EvalArgs {
  Fp* check;
  const Fp* ctrl;
  const Fp* data;
  const Fp* accum;
  const Fp* mix;
  const Fp* out;
  const FpExt* polyMix;
  Fp* scratch;
  uint32_t* carry;
  uint32_t domain;
  // The vanishing quotient, by cycle mod 4.
  Fp quot[4];
};

/// Stage `S` of eval_check (see `eval_stage` in the generated code) for the cycles from `first`,
/// one per work-item. The last stage has the constraint polynomial and writes the check.
template <uint32_t S>
void launch_eval_stage(sycl::queue& q, const EvalArgs& a, uint32_t first, uint32_t items) {
  q.parallel_for<KeccakEvalStage<S>>(
      sycl::nd_range<1>(items, kEvalCheckGroup), [=](sycl::nd_item<1> item) {
        uint32_t t = item.get_global_linear_id();
        uint32_t cycle = first + t;
        if (cycle >= a.domain) {
          return;
        }
        EvalCtx ectx{a.polyMix, ScratchRef{a.scratch + t}, a.carry + size_t(t) * kCarryWords};
        FpExt tot = eval_stage<S>(cycle, a.domain, ectx, a.ctrl, a.out, a.data, a.mix, a.accum);
        if constexpr (S + 1 == kEvalStages) {
          Fp quot = a.quot[cycle & 3];
          for (uint32_t i = 0; i < 4; i++) {
            a.check[a.domain * i + cycle] = tot.elems[i] * quot;
          }
        }
      });
}

template <uint32_t... S>
void launch_eval_stages(sycl::queue& q,
                        const EvalArgs& a,
                        uint32_t first,
                        uint32_t items,
                        std::integer_sequence<uint32_t, S...>) {
  (launch_eval_stage<S>(q, a, first, items), ...);
}

} // namespace risc0::circuit::keccak::cuda

extern "C" {

using namespace risc0;
using namespace risc0::circuit::keccak::cuda;

const char* risc0_circuit_keccak_intel_eval_check(void* queue,
                                                  Fp* check,
                                                  const Fp* ctrl,
                                                  const Fp* data,
                                                  const Fp* accum,
                                                  const Fp* mix,
                                                  const Fp* out,
                                                  uint32_t rouRaw,
                                                  uint32_t po2,
                                                  uint32_t domain,
                                                  const FpExt* polyMixPows) {
  try {
    if (domain == 0) {
      return nullptr;
    }
    sycl::queue& q = *static_cast<sycl::queue*>(queue);

    // y = (3 * rou^cycle)^(2^po2) depends only on cycle mod 4, rou having order 2^(po2 + 2), so
    // the vanishing quotient takes 4 values: compute them here.
    Fp rou;
    std::memcpy(&rou, &rouRaw, sizeof(rou));
    if (pow(rou, size_t(4) << po2) != Fp(1)) {
      return strdup("keccak eval_check: rou's order does not divide 2^(po2 + 2)");
    }
    EvalArgs a{check, ctrl, data, accum, mix, out, nullptr, nullptr, nullptr, domain, {}};
    for (uint32_t k = 0; k < 4; k++) {
      Fp y = pow(Fp(3) * pow(rou, k), size_t(1) << po2);
      a.quot[k] = inv(y - Fp(1));
    }

    // The scratch and carry of kScratchStride work-items: each stage runs that many cycles, one
    // per work-item, and the cycles go through all the stages a chunk at a time.
    DeviceArray<FpExt> polyMix(q, polyMixPows, kNumPolyMixPows);
    DeviceArray<Fp> scratch(q, size_t(kScratchStride) * kKeccakScratchElems);
    DeviceArray<uint32_t> carry(q, size_t(kScratchStride) * kCarryWords);
    a.polyMix = polyMix.ptr;
    a.scratch = scratch.ptr;
    a.carry = carry.ptr;
    for (uint32_t first = 0; first < domain; first += kScratchStride) {
      uint32_t items = std::min(domain - first, kScratchStride);
      items = (items + kEvalCheckGroup - 1) / kEvalCheckGroup * kEvalCheckGroup;
      launch_eval_stages(q, a, first, items, std::make_integer_sequence<uint32_t, kEvalStages>());
    }
    q.wait_and_throw();
  } catch (const sycl::exception& err) {
    return strdup(err.what());
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

} // extern "C"
