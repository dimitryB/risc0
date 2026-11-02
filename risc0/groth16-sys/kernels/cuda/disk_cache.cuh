#pragma once
// On-disk caches of the Groth16 prover's derived data (CUDA): the compact coefficients
// (groth16_coeffs.cuh) and the split-MSM corrections (groth16_prover.cuh). Each file carries a
// magic, kFormatVersion, the element size and a checksum of its payload, and is keyed by a hash
// that covers the format and the source file (sampled bytes, size, mtime and inode), so a file
// from another build, another proving key or a damaged write is rebuilt instead of used. The
// zkvm prover also deletes them when a Groth16 seal fails verification
// (risc0_groth16::prove::invalidate_caches).
//
// Directory: RISC0_G16_CACHE_DIR, else $XDG_CACHE_HOME/risc0-groth16, else
// $HOME/.cache/risc0-groth16 (the same as the Rust side's witness_len cache).

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace g16_cache {

// Bump when the layout or the meaning of a cache file changes.
constexpr uint32_t kFormatVersion = 1;

inline uint64_t fnv1a(uint64_t h, const void* p, size_t n) {
  const uint8_t* b = (const uint8_t*)p;
  for (size_t i = 0; i < n; i++) {
    h ^= b[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull;

// Checksum of a (large) payload: a multiply-xor over 64-bit words in 8 chunks on their own
// threads, combined in order. Detects damaged cache files; ~5 ms for the 280 MB compact file.
inline uint64_t payload_hash(const void* p, size_t n) {
  constexpr size_t kChunks = 8;
  size_t words = n / 8, per = (words + kChunks - 1) / kChunks;
  uint64_t part[kChunks] = {};
  auto run = [&](size_t c) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ c;
    size_t lo = c * per, hi = std::min(words, lo + per);
    for (size_t i = lo; i < hi; i++) {
      uint64_t w;
      memcpy(&w, (const uint8_t*)p + i * 8, 8);
      h = (h ^ w) * 0xff51afd7ed558ccdull;
      h ^= h >> 32;
    }
    part[c] = h;
  };
  std::vector<std::thread> th;
  for (size_t c = 1; c < kChunks; c++)
    th.emplace_back(run, c);
  run(0);
  for (auto& t : th)
    t.join();
  uint64_t h = fnv1a(kFnvBasis, part, sizeof part);
  h = fnv1a(h, (const uint8_t*)p + words * 8, n - words * 8);
  return fnv1a(h, &n, sizeof n);
}

// The cache directory (created), or "" if none can be determined.
inline std::string dir() {
  std::string d;
  if (const char* e = getenv("RISC0_G16_CACHE_DIR"))
    d = e;
  else if (const char* x = getenv("XDG_CACHE_HOME"))
    d = std::string(x) + "/risc0-groth16";
  else if (const char* h = getenv("HOME"))
    d = std::string(h) + "/.cache/risc0-groth16";
  else
    return "";
  for (size_t i = 1; i <= d.size(); i++) {
    if (i == d.size() || d[i] == '/' || d[i] == '\\') {
#ifdef _WIN32
      _mkdir(d.substr(0, i).c_str());
#else
      mkdir(d.substr(0, i).c_str(), 0755);
#endif
    }
  }
  return d;
}

// "<dir>/<prefix><key>.bin", or "" without a cache directory.
inline std::string path(const char* prefix, uint64_t key) {
  std::string d = dir();
  if (d.empty())
    return "";
  char name[96];
  snprintf(name, sizeof name, "/%s%016llx.bin", prefix, (unsigned long long)key);
  return d + name;
}

// A per-process temporary name next to `p`, renamed over it once written.
inline std::string tmp_path(const std::string& p) {
#ifdef _WIN32
  return p + ".tmp" + std::to_string(_getpid());
#else
  return p + ".tmp" + std::to_string(getpid());
#endif
}

} // namespace g16_cache
