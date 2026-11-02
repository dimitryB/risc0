// Copyright 2024 RISC Zero, Inc.
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

// The kernels use the driver API, and CUDA 13.1's CCCL v2 headers need its types,
// from the toolkit's own <cuda.h>, which this file shadows. MSVC has no
// `#include_next`; risc0-build-kernel generates a header that includes the toolkit's
// by absolute path (see `msvc_toolkit_include_dir`).
#if !defined(__HIPCC__)
#if defined(_MSC_VER)
#include <risc0_toolkit_cuda.h>
#else
#include_next <cuda.h>
#endif
#endif

#include <cstdint>
#include <cstring>
#ifndef __HIPCC__
#include <cuda_runtime.h>
#endif
#include <stdexcept>
#ifdef __HIPCC__
#include "hip_hop.h"
#endif

// When cuda2hip.hpp is force-included (sppark builds), these are already defined.
// Only provide them for standalone HIP compilation without cuda2hip.hpp.
#if defined(__HIPCC__) && !defined(WARP_SZ)
static const auto cudaStreamCreate         = hipStreamCreate;
static const auto cudaDeviceGetAttribute   = hipDeviceGetAttribute;
#define           cudaDevAttrMaxThreadsPerBlock hipDeviceAttributeMaxThreadsPerBlock
#endif

template <typename... Types> inline std::string fmt(const char* fmt, Types... args) {
  size_t len = std::snprintf(nullptr, 0, fmt, args...);
  std::string ret(++len, '\0');
  std::snprintf(&ret.front(), len, fmt, args...);
  ret.resize(--len);
  return ret;
}

#define CUDA_OK(expr)                                                                              \
  do {                                                                                             \
    cudaError_t code = expr;                                                                       \
    if (code != cudaSuccess) {                                                                     \
      auto file = std::strstr(__FILE__, "sppark");                                                 \
      auto msg = fmt("%s@%s:%d failed: \"%s\"",                                                    \
                     #expr,                                                                        \
                     file ? file : __FILE__,                                                       \
                     __LINE__,                                                                     \
                     cudaGetErrorString(code));                                                    \
      throw std::runtime_error{msg};                                                               \
    }                                                                                              \
  } while (0)

class CudaStream {
private:
  cudaStream_t stream;

public:
  CudaStream() { cudaStreamCreate(&stream); }
  ~CudaStream() { cudaStreamDestroy(stream); }

  inline operator cudaStream_t() const { return stream; }
};

struct LaunchConfig {
  dim3 grid;
  dim3 block;
  size_t shared;

  LaunchConfig(dim3 grid, dim3 block, size_t shared = 0)
      : grid(grid), block(block), shared(shared) {}
  LaunchConfig(int grid, int block, size_t shared = 0) : grid(grid), block(block), shared(shared) {}
};

inline int getCachedDevice() {
#ifdef __HIPCC__
  // HIP multi-GPU: different threads may use different devices,
  // so we must query each time (or use thread_local).
  int dev = 0;
  cudaGetDevice(&dev);
  return dev;
#else
  static int cached = -1;
  if (cached < 0) {
    cudaGetDevice(&cached);
  }
  return cached;
#endif
}

inline cudaStream_t getPersistentStream() {
  int device = getCachedDevice();
  static cudaStream_t streams[16] = {};
  if (!streams[device]) {
    CUDA_OK(cudaStreamCreate(&streams[device]));
  }
  return streams[device];
}

inline LaunchConfig getCachedSimpleConfig(uint32_t count) {
  int device = getCachedDevice();
  static int blocks[16] = {};
  if (blocks[device] == 0) {
    int maxThreads;
    CUDA_OK(cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, device));
    blocks[device] = maxThreads / 4;
  }
  int grid = (count + blocks[device] - 1) / blocks[device];
  return LaunchConfig{grid, blocks[device], 0};
}

#ifdef __HIPCC__
// launchKernel, with the kernel on a hop scope (hip_hop.h) when `hop`: for kernels expected to
// run >~1 ms.
template <typename... ExpTypes, typename... ActTypes>
const char* launchKernelHop(bool hop,
                            void (*kernel)(ExpTypes...),
                            uint32_t count,
                            uint32_t shared_size,
                            ActTypes&&... args) {
  try {
    risc0_hip_hop::Scope stream(getPersistentStream(), hop);
    LaunchConfig cfg = getCachedSimpleConfig(count);
    kernel<<<cfg.grid, cfg.block, shared_size, stream>>>(std::forward<ActTypes>(args)...);
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}
#endif

// Backward-compat alias used by circuit ffi files
inline LaunchConfig getSimpleConfig(uint32_t count) {
  return getCachedSimpleConfig(count);
}

template <typename... ExpTypes, typename... ActTypes>
const char* launchKernel(void (*kernel)(ExpTypes...),
                         uint32_t count,
                         uint32_t shared_size,
                         ActTypes&&... args) {
  try {
    cudaStream_t stream = getPersistentStream();
    LaunchConfig cfg = getCachedSimpleConfig(count);
#ifdef __HIPCC__
    kernel<<<cfg.grid, cfg.block, shared_size, stream>>>(std::forward<ActTypes>(args)...);
#else
    cudaLaunchConfig_t config;
    config.attrs = nullptr;
    config.numAttrs = 0;
    config.gridDim = cfg.grid;
    config.blockDim = cfg.block;
    config.dynamicSmemBytes = shared_size;
    config.stream = stream;
    CUDA_OK(cudaLaunchKernelEx(&config, kernel, std::forward<ActTypes>(args)...));
#endif
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}
