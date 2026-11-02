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

#include "ffi.h"
#include "context.h"
#include "fp.h"
#include "fpext.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#endif

#include "vendor/nvtx3/nvtx3.hpp"
#include "vendor/poolstl.hpp"

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <vector>

using namespace risc0;
using namespace risc0::circuit::recursion;

constexpr size_t kStepModeParallel = 0;
constexpr size_t kStepModeSeqForward = 1;
constexpr size_t kStepModeSeqReverse = 2;

namespace risc0::circuit::recursion {

MachineContext::MachineContext(ExecBuffers* buffers, PreflightTrace* trace, uint32_t steps)
    : buffers(buffers)
    , trace(trace)
    , steps(steps)
    , womRows(trace->numCycles * kMaxWomRowsPerCycle,
              WomArgumentRow{kInvalidPattern, FpExt::invalid()})
    , womIndex(trace->numCycles) {}

void MachineContext::parStepExec(uint32_t cycle) {
  if (cycle == 0 || isParSafeExec(cycle)) {
    // printf("step_exec(%u)\n", cycle);
    step_exec(this, steps, cycle++, args().data());
    while (cycle < trace->numCycles && !isParSafeExec(cycle)) {
      // printf("step_exec(%u)\n", cycle);
      step_exec(this, steps, cycle++, args().data());
    }
  }
}

void MachineContext::doStepExec(uint32_t mode) {
  nvtx3::scoped_range range("stepExec");
  switch (mode) {
  case kStepModeParallel: {
    auto begin = poolstl::iota_iter<uint32_t>(0);
    auto end = poolstl::iota_iter<uint32_t>(trace->numCycles);
    std::for_each(poolstl::par, begin, end, [&](uint32_t cycle) { parStepExec(cycle); });
  } break;
  case kStepModeSeqForward: {
    for (size_t i = 0; i < trace->numCycles; i++) {
      // printf("step_exec(%zu)\n", i);
      step_exec(this, steps, i, args().data());
    }
  } break;
  case kStepModeSeqReverse: {
    for (size_t i = 0; i < trace->numCycles; i++) {
      parStepExec(trace->numCycles - i - 1);
    }
  } break;
  }
}

void MachineContext::doStepVerifyWom(uint32_t mode) {
  nvtx3::scoped_range range("stepVerifyWom");
  switch (mode) {
  case kStepModeParallel: {
    auto begin = poolstl::iota_iter<uint32_t>(0);
    auto end = poolstl::iota_iter<uint32_t>(trace->numCycles);
    std::for_each(poolstl::par, begin, end, [&](uint32_t cycle) {
      step_verify_mem(this, steps, cycle, args().data());
    });
  } break;
  case kStepModeSeqForward: {
    for (size_t i = 0; i < trace->numCycles; i++) {
      // printf("step_verify_mem(%zu)\n", i);
      step_verify_mem(this, steps, i, args().data());
    }
  } break;
  case kStepModeSeqReverse: {
    for (size_t i = 0; i < trace->numCycles; i++) {
      size_t cycle = trace->numCycles - i - 1;
      // printf("step_verify_mem(%zu)\n", cycle);
      step_verify_mem(this, steps, cycle, args().data());
    }
  } break;
  }
}

// Sort the write-once-memory rows by operator< (addr, then value). The
// written rows of cycle c occupy slots [0, womIndex[c]) of that cycle; every
// other slot is the identical invalid pattern (addr 0xffffffff), which sorts
// last. A counting sort on addr over the written rows, with runs of equal
// addr finished by operator<, therefore gives exactly the std::sort result.
// The circuit makes valid addresses dense (step_verify_mem allows address
// steps of 0 or 1 only), so the histogram is at most the number of written
// rows; anything else (or RISC0_RECURSION_FAST_SORT=0) uses std::sort.
// RISC0_RECURSION_FAST_SORT_CHECK=1 compares against std::sort.
static bool sortWomFast(std::vector<WomArgumentRow>& womRows,
                        const std::vector<uint32_t>& womIndex,
                        uint32_t numCycles) {
  size_t nv = 0;
  uint32_t maxAddr = 0;
  for (uint32_t c = 0; c < numCycles; c++) {
    uint32_t n = womIndex[c];
    if (n > kMaxWomRowsPerCycle) {
      return false;
    }
    nv += n;
    const WomArgumentRow* r = &womRows[size_t(c) * kMaxWomRowsPerCycle];
    for (uint32_t i = 0; i < n; i++) {
      maxAddr = std::max(maxAddr, r[i].addr);
    }
  }
  if (size_t(maxAddr) >= nv) {
    return false;
  }
  std::vector<uint32_t> cnt(size_t(maxAddr) + 2, 0);
  for (uint32_t c = 0; c < numCycles; c++) {
    const WomArgumentRow* r = &womRows[size_t(c) * kMaxWomRowsPerCycle];
    for (uint32_t i = 0; i < womIndex[c]; i++) {
      cnt[r[i].addr + 1]++;
    }
  }
  for (size_t d = 1; d < cnt.size(); d++) {
    cnt[d] += cnt[d - 1];
  }
  std::vector<WomArgumentRow> out(womRows.size(),
                                  WomArgumentRow{kInvalidPattern, FpExt::invalid()});
  for (uint32_t c = 0; c < numCycles; c++) {
    const WomArgumentRow* r = &womRows[size_t(c) * kMaxWomRowsPerCycle];
    for (uint32_t i = 0; i < womIndex[c]; i++) {
      out[cnt[r[i].addr]++] = r[i];
    }
  }
  for (size_t i = 0; i < nv;) {
    size_t j = i + 1;
    bool same = true;
    while (j < nv && out[j].addr == out[i].addr) {
      same = same && memcmp(&out[j], &out[i], sizeof(WomArgumentRow)) == 0;
      j++;
    }
    if (j - i > 1 && !same) {
      std::sort(out.begin() + i, out.begin() + j);
    }
    i = j;
  }
  womRows.swap(out);
  return true;
}

void MachineContext::verifyWom(uint32_t mode) {
  nvtx3::scoped_range range("verifyWom");

  {
    nvtx3::scoped_range range("sortWom");
    static const bool fast = []() {
      const char* e = std::getenv("RISC0_RECURSION_FAST_SORT");
      return !(e && e[0] == '0');
    }();
    static const bool check = std::getenv("RISC0_RECURSION_FAST_SORT_CHECK") != nullptr;
    std::vector<WomArgumentRow> ref;
    if (fast && check) {
      ref = womRows;
      std::sort(poolstl::par, ref.begin(), ref.end());
    }
    if (!fast || !sortWomFast(womRows, womIndex, trace->numCycles)) {
      std::sort(poolstl::par, womRows.begin(), womRows.end());
    }
    if (!ref.empty() &&
        memcmp(ref.data(), womRows.data(), ref.size() * sizeof(WomArgumentRow)) != 0) {
      throw std::runtime_error("recursion witgen: fast WOM sort differs from std::sort");
    }
  }

  {
    nvtx3::scoped_range range("scan");
    std::exclusive_scan(poolstl::par, womIndex.begin(), womIndex.end(), womIndex.begin(), 0);
  }

  injectWomBacks();
  doStepVerifyWom(mode);
}

void MachineContext::injectWomBacks() {
  nvtx3::scoped_range range("injectWomBacks");
  auto begin = poolstl::iota_iter<uint32_t>(1);
  auto end = poolstl::iota_iter<uint32_t>(trace->numCycles);
  Fp* data = buffers->data;
  std::for_each(poolstl::par, begin, end, [&](uint32_t cycle) {
    uint32_t idx = womIndex[cycle];
    if (idx) {
      const WomArgumentRow& prev = womRows[idx - 1];
      data[0 * steps + cycle - 1] = prev.addr;
      data[1 * steps + cycle - 1] = prev.value.elems[0];
      data[2 * steps + cycle - 1] = prev.value.elems[1];
      data[3 * steps + cycle - 1] = prev.value.elems[2];
      data[4 * steps + cycle - 1] = prev.value.elems[3];
    } else {
      data[0 * steps + cycle - 1] = 0;
      data[1 * steps + cycle - 1] = 0;
      data[2 * steps + cycle - 1] = 0;
      data[3 * steps + cycle - 1] = 0;
      data[4 * steps + cycle - 1] = 0;
    }
  });
}

AccumContext::AccumContext(AccumBuffers* buffers, uint32_t steps, uint32_t cycles)
    : buffers(buffers), steps(steps), cycles(cycles), accum(steps, FpExt(1)) {}

void AccumContext::computeAccum() {
  nvtx3::scoped_range range("computeAccum");
  auto begin = poolstl::iota_iter<uint32_t>(0);
  auto end = poolstl::iota_iter<uint32_t>(steps);
  std::for_each(poolstl::par, begin, end, [&](uint32_t cycle) {
    // printf("step_compute_accum(%zu)\n", cycle);
    step_compute_accum(this, cycles, cycle, args().data());
  });
}

void AccumContext::calcPrefixProducts() {
  nvtx3::scoped_range range("calcPrefixProducts");
  std::inclusive_scan(
      accum.begin(), accum.end(), accum.begin(), std::multiplies<FpExt>{}, FpExt(1));
}

void AccumContext::verifyAccum() {
  nvtx3::scoped_range range("verifyAccum");
  auto begin = poolstl::iota_iter<uint32_t>(0);
  auto end = poolstl::iota_iter<uint32_t>(steps);
  std::for_each(poolstl::par, begin, end, [&](uint32_t cycle) {
    // printf("step_verify_accum(%zu)\n", cycle);
    step_verify_accum(this, cycles, cycle, args().data());
  });
}

} // namespace risc0::circuit::recursion

extern "C" {

const char* risc0_circuit_recursion_cpu_witgen(uint32_t mode,
                                               ExecBuffers* buffers,
                                               PreflightTrace* trace,
                                               uint32_t steps) {
  try {
    MachineContext ctx(buffers, trace, steps);
    ctx.doStepExec(mode);
    ctx.verifyWom(mode);
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

const char*
risc0_circuit_recursion_cpu_accum(AccumBuffers* buffers, uint32_t steps, uint32_t cycles) {
  try {
    AccumContext ctx(buffers, steps, cycles);
    ctx.computeAccum();
    ctx.calcPrefixProducts();
    ctx.verifyAccum();
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

const char* risc0_circuit_recursion_cpu_eval_check(
    AccumBuffers* buffers, FpExt* poly_mix, Fp* check, Fp rou, uint32_t po2, uint32_t steps) {
  try {
    nvtx3::scoped_range range("evalCheck");
    std::array<Fp*, 5> args{
        buffers->ctrl,
        buffers->global,
        buffers->data,
        buffers->mix,
        buffers->accum,
    };
    auto begin = poolstl::iota_iter<uint32_t>(0);
    auto end = poolstl::iota_iter<uint32_t>(steps);
    std::for_each(poolstl::par, begin, end, [&](uint32_t cycle) {
      FpExt tot = poly_fp(cycle, steps, poly_mix, args.data());
      Fp x = pow(rou, cycle);
      Fp y = pow(Fp(3) * x, 1 << po2);
      FpExt ret = tot * inv(y - Fp(1));
      check[steps * 0 + cycle] = ret.elems[0];
      check[steps * 1 + cycle] = ret.elems[1];
      check[steps * 2 + cycle] = ret.elems[2];
      check[steps * 3 + cycle] = ret.elems[3];
    });
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

} // extern "C"

// Execute the sequential (non-par-safe) chains of length >= the caller's threshold on the
// host, so the GPU witgen does not run them one thread at a time (the longest lift chain, a
// 1439-cycle Poseidon2 sponge, is ~90% of the GPU stepExec time). A chain reads only its own
// rows, ctrl at its own rows, and the WOM/IOP preflight data, so each chain is executed in a
// small private window: local cycle i (1..len) <-> global cycle start + i - 1, with a
// power-of-two window size as the step count. (A global steps-sized column-major scratch is
// ~10x slower: every column is a 1 MB power-of-two stride apart and the rows thrash the
// caches.) step_exec uses `steps` and `cycle` only to index the buffers and the trace.
//
// code: row-major program code (numCycles x ctrlCols). chains: (start, len) pairs.
// Outputs, per chain row r in chain order: outCycles[r], outRows[r*dataCols + k] (raw Fp, every
// data column), outWomIdx[r] (number of plonkWrite_wom rows), outWomRows[(r*9 + j)*5 ..]
// ({addr as raw Fp, value[4]}). Errors if a chain wrote a global (the caller falls back).
extern "C" const char* risc0_circuit_recursion_cpu_chains(const Fp* code,
                                                         uint32_t ctrlCols,
                                                         const PreflightTrace* trace,
                                                         uint32_t /*steps*/,
                                                         uint32_t dataCols,
                                                         const uint32_t* chains,
                                                         uint32_t nchains,
                                                         uint32_t* outCycles,
                                                         uint32_t* outRows,
                                                         uint32_t* outWomIdx,
                                                         uint32_t* outWomRows) {
  try {
    std::vector<size_t> offset(nchains + 1, 0);
    for (uint32_t i = 0; i < nchains; i++) offset[i + 1] = offset[i] + chains[2 * i + 1];
    std::vector<uint32_t> order(nchains);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return chains[2 * a + 1] > chains[2 * b + 1];
    });
    std::atomic<bool> wroteGlobal{false};
    const uint32_t* code32 = reinterpret_cast<const uint32_t*>(code);
    std::for_each(poolstl::par, order.begin(), order.end(), [&](uint32_t ci) {
      uint32_t start = chains[2 * ci], len = chains[2 * ci + 1];
      uint32_t S = 2;
      while (S < len + 1) S <<= 1;
      std::vector<uint32_t> ctrl((size_t)S * ctrlCols, kInvalidPattern);
      std::vector<uint32_t> data((size_t)S * dataCols, kInvalidPattern);
      uint32_t global[64];
      for (auto& g : global) g = kInvalidPattern;
      for (uint32_t i = 1; i <= len; i++) {
        size_t c = start + i - 1;
        for (uint32_t k = 0; k < ctrlCols; k++) ctrl[(size_t)k * S + i] = code32[c * ctrlCols + k];
      }
      // Private copy of the window's cycles (readIOPBody advances iopIdx); slot 0 is unused.
      std::vector<PreflightCycle> cyc(len + 1);
      for (uint32_t i = 1; i <= len; i++) cyc[i] = trace->cycles[start + i - 1];
      PreflightTrace tr = *trace;
      tr.cycles = cyc.data();
      tr.numCycles = len + 1;
      ExecBuffers eb{reinterpret_cast<Fp*>(ctrl.data()),
                     reinterpret_cast<Fp*>(data.data()),
                     reinterpret_cast<Fp*>(global)};
      MachineContext mctx(&eb, &tr, S);
      Fp* args[3] = {eb.ctrl, eb.global, eb.data};
      for (uint32_t i = 1; i <= len; i++) step_exec(&mctx, S, i, args);
      for (auto g : global) {
        if (g != kInvalidPattern) wroteGlobal = true;
      }
      size_t r = offset[ci];
      for (uint32_t i = 1; i <= len; i++, r++) {
        outCycles[r] = start + i - 1;
        for (uint32_t k = 0; k < dataCols; k++) outRows[r * dataCols + k] = data[(size_t)k * S + i];
        outWomIdx[r] = mctx.womIndex[i];
        for (uint32_t j = 0; j < kMaxWomRowsPerCycle; j++) {
          const WomArgumentRow& w = mctx.womRows[(size_t)i * kMaxWomRowsPerCycle + j];
          uint32_t* o = &outWomRows[(r * kMaxWomRowsPerCycle + j) * 5];
          // The device rows keep the address as a raw Fp (the host decodes it).
          o[0] = w.addr == kInvalidPattern ? kInvalidPattern : Fp(w.addr).asRaw();
          for (int e = 0; e < 4; e++) o[1 + e] = w.value.elems[e].asRaw();
        }
      }
    });
    if (wroteGlobal) {
      throw std::runtime_error("host chain wrote a global");
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}
