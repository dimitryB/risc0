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

// Pinned host staging buffers backed by transparent huge pages (include after "cuda.h").
//
// cudaHostAlloc pins 4 KB pages: 200 MB costs ~40 ms (30 ms faulting and zeroing the pages,
// 10 ms registering them) and holds the CUDA driver lock throughout, stalling every other
// thread's CUDA calls. Here the memory is a 2 MB-aligned anonymous mapping advised
// MADV_HUGEPAGE, populated one huge page per madvise call (so the process mmap lock is never held
// for long), then registered with cudaHostRegister: ~11 + 1.7 ms for 200 MB, with the same H2D
// bandwidth (28.7 GB/s). Falls back to cudaHostAlloc when THP or registration is unavailable, on
// HIP, or with RISC0_CUDA_THP=0. Either way the memory is portable (pinned for every CUDA
// context, not just the calling thread's current device's).

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#if !defined(__HIPCC__) && defined(__linux__)
#include <sys/mman.h>
#define RISC0_THP_PINNED 1
#endif

struct ThpPinned {
  void* ptr = nullptr;
  size_t bytes = 0;
  void* raw = nullptr; // mmap base (THP path only)
  size_t raw_bytes = 0;

  // Replace the buffer with one of at least `n` bytes. Frees the old one first (see free_pinned).
  void alloc(size_t n) {
    free_pinned();
#ifdef RISC0_THP_PINNED
    if (thp_enabled() && thp_alloc(n)) return;
#endif
    void* p = nullptr;
    CUDA_OK(cudaHostAlloc(&p, n, cudaHostAllocPortable));
    ptr = p;
    bytes = n;
  }

  // Unpins and frees the buffer. Invariant: the caller has waited (stream or event sync) for
  // every async copy from or to it; nothing may be queued that still uses it.
  //
  // Measured on driver 595 (opt-artifacts cuda/land/fix/unreg): cudaHostUnregister, like
  // cudaFreeHost, waits until the whole device is idle, not only for copies touching the buffer,
  // and holds a driver lock while it waits, so other threads' kernel launches block for as long
  // (252 ms behind a 280 ms kernel). The cudaDeviceSynchronize first waits WITHOUT that lock
  // (other threads keep launching), leaving the unregister only the work queued meanwhile. So
  // this is a device-wide wait either way: keep it off the proving path (the pools retire
  // buffers and unpin them from the HAL's release hooks).
  void free_pinned() {
    if (!ptr) return;
#ifdef RISC0_THP_PINNED
    if (raw) {
      cudaDeviceSynchronize();
      cudaHostUnregister(ptr);
      munmap(raw, raw_bytes);
    } else
#endif
    {
      cudaFreeHost(ptr);
    }
    *this = ThpPinned();
  }

private:
#ifdef RISC0_THP_PINNED
  static bool thp_enabled() {
    static const bool on = [] {
      const char* e = std::getenv("RISC0_CUDA_THP");
      return !(e && e[0] == '0');
    }();
    return on;
  }

  bool thp_alloc(size_t n) {
    const size_t huge = size_t(2) << 20;
    size_t len = (n + huge - 1) & ~(huge - 1);
    void* r = mmap(nullptr, len + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (r == MAP_FAILED) return false;
    char* p = (char*)(((uintptr_t)r + huge - 1) & ~(uintptr_t)(huge - 1));
    (void)madvise(p, len, MADV_HUGEPAGE);
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
    for (size_t off = 0; off < len; off += huge) {
      if (madvise(p + off, huge, MADV_POPULATE_WRITE) != 0) {
        for (size_t i = off; i < off + huge; i += 4096) p[i] = 0;
      }
    }
    if (cudaHostRegister(p, len, cudaHostRegisterPortable) != cudaSuccess) {
      (void)cudaGetLastError();
      munmap(r, len + huge);
      return false;
    }
    ptr = p;
    bytes = len;
    raw = r;
    raw_bytes = len + huge;
    return true;
  }
#endif
};
