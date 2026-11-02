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

#pragma once

#include "fp.h"

#include <cassert>
#include <cstdint>
#include <cstdio>

#ifdef __HIPCC__
namespace cuda {
  template<typename T> struct atomic {
    T val;
    __device__ void operator++()    { atomicAdd(&val, T(1)); }
    __device__ void operator++(int) { atomicAdd(&val, T(1)); }
    __device__ operator T() const   { return atomicAdd(const_cast<T*>(&val), T(0)); }
  };
}
using LookupCell = ::cuda::atomic<uint32_t>;
#else
// Plain counters, updated with relaxed device-scope atomicAdd. ::cuda::atomic's
// operators are seq_cst at system scope (a MEMBAR.SC.SYS per access); the
// counts are integer adds, which commute, so the totals are the same.
using LookupCell = uint32_t;
#endif

namespace risc0::circuit::rv32im_v2::cuda {

struct LookupTables {
  LookupCell* tableU8;
  LookupCell* tableU16;

  __device__ void lookupDelta(Fp table, Fp index, Fp count) {
    uint32_t tableU32 = table.asUInt32();
    uint32_t indexU32 = index.asUInt32();
    if (tableU32 == 0) {
      // tableCycle[index] += count;
      return;
    }
    if (tableU32 != 8 && tableU32 != 16) {
      assert(false && "Invalid lookup table");
    }
    if (indexU32 >= (1 << tableU32)) {
      printf("LOOKUP ERROR: table = %u, index = %u\n", tableU32, indexU32);
      assert(false && "u8/16 table error");
    }
    // printf("table = %u, index = %u\n", tableU32, indexU32);
#ifdef __HIPCC__
    if (tableU32 == 8) {
      tableU8[indexU32]++;
    } else {
      tableU16[indexU32]++;
    }
#else
    if (tableU32 == 8) {
      atomicAdd(&tableU8[indexU32], 1u);
    } else {
      atomicAdd(&tableU16[indexU32], 1u);
    }
#endif
  }

  __device__ Fp lookupCurrent(Fp table, Fp index) {
    uint32_t tableU32 = table.asUInt32();
    if (tableU32 != 8 && tableU32 != 16) {
      assert(false && "Invalid lookup table");
    }
    uint32_t indexU32 = index.asUInt32();
#ifdef __HIPCC__
    if (tableU32 == 8) {
      return Fp(tableU8[indexU32]);
    } else {
      return Fp(tableU16[indexU32]);
    }
#else
    // Read after the counting kernel finished: bypass L1, which may be stale.
    if (tableU32 == 8) {
      return Fp(__ldcg(&tableU8[indexU32]));
    } else {
      return Fp(__ldcg(&tableU16[indexU32]));
    }
#endif
  }
};

} // namespace risc0::circuit::rv32im_v2::cuda
