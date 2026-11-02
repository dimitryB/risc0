// Groth16 prover for Intel GPUs (SYCL), the counterpart of the CUDA prover in ../cuda.
//
// It follows the CUDA prover step for step, on the same inputs (the proving key, the
// preprocessed coefficients and the fuzzed MSM results of the rzup `risc0-groth16` component)
// and with the same arithmetic (see bn254.hpp), so it computes the same proof up to the random
// blinding r, s:
//   a, b = witness into the QAP polynomials (evaluations over the domain); c = a * b;
//   a, b, c to evaluations over the coset (inverse NTT, times shift^k, forward NTT);
//   h = a * b - c;
//   A = MSM(A, w + f), B1 = MSM(B1, w + f), B2 = MSM(B2, w + f), C = MSM(C, w + f), H = MSM(H, h)
//   with f the deterministic ChaCha "fuzz" whose MSMs the fuzzed results file holds, negated;
//   the blinding with the verifying key's alpha, beta, delta and random r, s on the host.
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "bn254.hpp"
#include "ntt_params.hpp"

namespace r0g16 {

// sppark's coeff_t: {u32 m, c, s; fr_t value}, value 8-aligned (48 bytes).
struct Coeff {
  uint32_t m, c, s, pad;
  Fr value;
};
static_assert(sizeof(Coeff) == 48, "Coeff must match sppark's coeff_t");

struct VerifyingKey {
  G1Affine alpha_g1, beta_g1;
  G2Affine beta_g2, gamma_g2;
  G1Affine delta_g1;
  G2Affine delta_g2;
};

// The fuzzed results file: sppark's msm_results {point_t a, b_g1, c, h; point_fp2_t b_g2},
// already negated.
struct FuzzedResults {
  G1Jacobian a, b_g1, c, h;
  G2Jacobian b_g2;
};
static_assert(sizeof(FuzzedResults) == 576, "FuzzedResults must match the fuzzed results file");

struct Proof {
  G1Affine a, c;
  G2Affine b;
};

inline bool verbose() {
  static const bool v = std::getenv("RISC0_GROTH16_INTEL_VERBOSE") != nullptr;
  return v;
}

struct Timer {
  const char* what;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  explicit Timer(const char* w) : what(w) {}
  ~Timer() {
    if (verbose()) {
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                      .count();
      std::fprintf(stderr, "[groth16-intel] %s: %.1f ms\n", what, ms);
    }
  }
};

// ---------------------------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------------------------

class MappedFile {
public:
  const uint8_t* data = nullptr;
  size_t size = 0;

  explicit MappedFile(const char* path) {
#ifdef _WIN32
    HANDLE file = CreateFileA(
        path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE)
      throw std::runtime_error(std::string("open(\"") + path + "\") failed");
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(file, &sz)) {
      CloseHandle(file);
      throw std::runtime_error(std::string("stat(\"") + path + "\") failed");
    }
    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(file);
    if (!mapping)
      throw std::runtime_error(std::string("mmap(\"") + path + "\") failed");
    data = (const uint8_t*)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    if (!data)
      throw std::runtime_error(std::string("mmap(\"") + path + "\") failed");
    size = (size_t)sz.QuadPart;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0)
      throw std::runtime_error(std::string("open(\"") + path + "\") failed");
    struct stat st;
    fstat(fd, &st);
    size = st.st_size;
    void* p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED)
      throw std::runtime_error(std::string("mmap(\"") + path + "\") failed");
    data = (const uint8_t*)p;
#endif
  }
  ~MappedFile() {
#ifdef _WIN32
    if (data)
      UnmapViewOfFile(data);
#else
    if (data)
      munmap((void*)data, size);
#endif
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
};

// Device memory owned by a queue's context.
template <class T> class DeviceBuffer {
public:
  T* ptr = nullptr;
  size_t len = 0;
  sycl::queue* q = nullptr;

  DeviceBuffer() = default;
  DeviceBuffer(sycl::queue& queue, size_t n) : len(n), q(&queue) {
    ptr = sycl::malloc_device<T>(n ? n : 1, queue);
    if (!ptr)
      throw std::runtime_error("groth16-intel: device allocation of " +
                               std::to_string(n * sizeof(T)) + " bytes failed");
  }
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), len(o.len), q(o.q) { o.ptr = nullptr; }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    reset();
    ptr = o.ptr;
    len = o.len;
    q = o.q;
    o.ptr = nullptr;
    return *this;
  }
  ~DeviceBuffer() { reset(); }
  void reset() {
    if (ptr) {
      q->wait();
      sycl::free(ptr, *q);
      ptr = nullptr;
    }
  }
  void upload(const void* src, size_t n) { q->memcpy(ptr, src, n * sizeof(T)).wait(); }
};

// ---------------------------------------------------------------------------------------------
// Proving key (.zkey): verifying key and the A, B1, B2, C, H point vectors.
// ---------------------------------------------------------------------------------------------

struct Srs {
  VerifyingKey vk;
  size_t num_vars = 0, num_public = 0, domain_size = 0;
  DeviceBuffer<G1Affine> a, b_g1, c, h;
  DeviceBuffer<G2Affine> b_g2;

  Srs(sycl::queue& q, const char* path) {
    Timer t("load proving key");
    MappedFile f(path);
    const uint8_t* p = f.data;
    size_t cur = 0;
    auto need = [&](size_t n) {
      if (cur + n > f.size)
        throw std::runtime_error(std::string("proving key \"") + path + "\" is truncated");
    };
    auto u32 = [&]() {
      need(4);
      uint32_t v;
      std::memcpy(&v, p + cur, 4);
      cur += 4;
      return v;
    };
    auto u64 = [&]() {
      need(8);
      uint64_t v;
      std::memcpy(&v, p + cur, 8);
      cur += 8;
      return v;
    };
    need(4);
    if (std::memcmp(p, "zkey", 4) != 0)
      throw std::runtime_error(std::string("\"") + path + "\" is not a zkey file");
    cur = 4;
    (void)u32(); // version
    if (u32() != 10)
      throw std::runtime_error("zkey: expected 10 sections");
    size_t off[10] = {0}, size[10] = {0};
    for (uint32_t id = 1; id <= 9; id++) {
      if (u32() != id)
        throw std::runtime_error("zkey: sections out of order");
      size[id] = (size_t)u64();
      need(size[id]);
      off[id] = cur;
      if (id == 1) {
        if (u32() != 1)
          throw std::runtime_error("zkey: protocol is not groth16");
      } else if (id == 2) {
        uint32_t qb = u32();
        cur += qb;
        uint32_t rb = u32();
        cur += rb;
        num_vars = u32();
        num_public = u32();
        domain_size = u32();
        auto g1 = [&](G1Affine& pt) {
          need(64);
          std::memcpy(&pt, p + cur, 64);
          cur += 64;
        };
        auto g2 = [&](G2Affine& pt) {
          need(128);
          std::memcpy(&pt, p + cur, 128);
          cur += 128;
        };
        g1(vk.alpha_g1);
        g1(vk.beta_g1);
        g2(vk.beta_g2);
        g2(vk.gamma_g2);
        g1(vk.delta_g1);
        g2(vk.delta_g2);
      }
      cur = off[id] + size[id];
    }
    auto load_g1 = [&](DeviceBuffer<G1Affine>& d, int id) {
      size_t n = size[id] / sizeof(G1Affine);
      d = DeviceBuffer<G1Affine>(q, n);
      d.upload(p + off[id], n);
    };
    load_g1(a, 5);
    load_g1(b_g1, 6);
    {
      size_t n = size[7] / sizeof(G2Affine);
      b_g2 = DeviceBuffer<G2Affine>(q, n);
      b_g2.upload(p + off[7], n);
    }
    load_g1(c, 8);
    load_g1(h, 9);
    if (h.len != domain_size || (domain_size & (domain_size - 1)) != 0)
      throw std::runtime_error("zkey: H must have one point per domain element (a power of 2)");
  }
};

// Preprocessed coefficients: 4 x u64 header (coeff counts of a and b, index counts of a and b),
// the coefficients, then the per-row start indices.
struct Coeffs {
  DeviceBuffer<Coeff> coeffs;
  DeviceBuffer<uint32_t> indices;
  size_t count_a = 0, count_b = 0, ind_a = 0, ind_b = 0;

  Coeffs(sycl::queue& q, const char* path) {
    Timer t("load coefficients");
    MappedFile f(path);
    if (f.size < 32)
      throw std::runtime_error(std::string("\"") + path + "\" is truncated");
    uint64_t hdr[4];
    std::memcpy(hdr, f.data, 32);
    count_a = hdr[0];
    count_b = hdr[1];
    ind_a = hdr[2];
    ind_b = hdr[3];
    size_t need = 32 + (count_a + count_b) * sizeof(Coeff) + (ind_a + ind_b) * 4;
    if (f.size < need)
      throw std::runtime_error(std::string("\"") + path + "\" is truncated");
    coeffs = DeviceBuffer<Coeff>(q, count_a + count_b);
    coeffs.upload(f.data + 32, count_a + count_b);
    indices = DeviceBuffer<uint32_t>(q, ind_a + ind_b);
    indices.upload(f.data + 32 + (count_a + count_b) * sizeof(Coeff), ind_a + ind_b);
  }
};

inline FuzzedResults read_fuzzed_results(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error(std::string("open(\"") + path + "\") failed");
  FuzzedResults r;
  f.read((char*)&r, sizeof(r));
  if (f.gcount() != (std::streamsize)sizeof(r))
    throw std::runtime_error(std::string("\"") + path + "\" has the wrong size");
  return r;
}

// ---------------------------------------------------------------------------------------------
// Elementwise kernels
// ---------------------------------------------------------------------------------------------

// out[coeffs[j].c] = sum_j witness[coeffs[j].s] * coeffs[j].value over each row's coefficients
// [indices[i], indices[i + 1]) (the CUDA prover's witness_into_poly).
inline void witness_into_poly(sycl::queue& q,
                              Fr* out,
                              const Coeff* coeffs,
                              const uint32_t* indices,
                              size_t n_indices,
                              const Fr* witness) {
  if (n_indices < 2)
    return;
  q.parallel_for(sycl::range<1>(n_indices - 1), [=](sycl::id<1> id) {
    size_t i = id[0];
    uint32_t start = indices[i], end = indices[i + 1];
    Fr sum = Fr::zero();
    uint32_t row = 0;
    for (uint32_t j = start; j < end; j++) {
      Coeff c = coeffs[j];
      sum += witness[c.s] * c.value;
      row = c.c;
    }
    if (end > start)
      out[row] = sum;
  });
}

inline void fill_zero(sycl::queue& q, Fr* p, size_t n) { q.memset(p, 0, n * sizeof(Fr)); }

// ---------------------------------------------------------------------------------------------
// NTT over Fr (radix-2, iterative)
// ---------------------------------------------------------------------------------------------

inline uint32_t bit_rev(uint32_t x, uint32_t lg) {
  uint32_t r = 0;
  for (uint32_t i = 0; i < lg; i++) {
    r = (r << 1) | (x & 1);
    x >>= 1;
  }
  return r;
}

// out[j] = base^j for j < n, from base^(2^k) for k < 32.
inline void powers(sycl::queue& q, Fr* out, size_t n, Fr base) {
  Fr sq[32];
  sq[0] = base;
  for (int k = 1; k < 32; k++)
    sq[k] = sq[k - 1].sqr();
  struct Table {
    Fr v[32];
  } table;
  for (int k = 0; k < 32; k++)
    table.v[k] = sq[k];
  q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
    uint32_t j = (uint32_t)id[0];
    Fr r = Fr::one();
    for (int k = 0; j; k++, j >>= 1)
      if (j & 1)
        r *= table.v[k];
    out[id[0]] = r;
  });
}

struct NttPlan {
  uint32_t lg = 0;
  size_t n = 0;
  DeviceBuffer<Fr> tw_fwd, tw_inv; // omega^j, omega^-j for j < n / 2
  DeviceBuffer<Fr> coset;          // n^-1 * shift^k for k < n

  NttPlan(sycl::queue& q, uint32_t lg_n) : lg(lg_n), n((size_t)1 << lg_n) {
    if (lg + 1 > (uint32_t)NTT_S)
      throw std::runtime_error("groth16-intel: NTT domain too large");
    tw_fwd = DeviceBuffer<Fr>(q, n / 2);
    tw_inv = DeviceBuffer<Fr>(q, n / 2);
    coset = DeviceBuffer<Fr>(q, n);
    powers(q, tw_fwd.ptr, n / 2, Fr::from_limbs(FORWARD_ROOTS_OF_UNITY[lg]));
    powers(q, tw_inv.ptr, n / 2, Fr::from_limbs(INVERSE_ROOTS_OF_UNITY[lg]));
    // The coset shift is the 2^(lg+1)-th root, as in the CUDA prover.
    powers(q, coset.ptr, n, Fr::from_limbs(FORWARD_ROOTS_OF_UNITY[lg + 1]));
    Fr ninv = Fr::from_limbs(DOMAIN_SIZE_INVERSE[lg]);
    Fr* cs = coset.ptr;
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) { cs[id[0]] *= ninv; });
    q.wait();
  }

  void bit_reverse(sycl::queue& q, Fr* x) const {
    uint32_t l = lg;
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
      uint32_t i = (uint32_t)id[0];
      uint32_t j = bit_rev(i, l);
      if (i < j) {
        Fr t = x[i];
        x[i] = x[j];
        x[j] = t;
      }
    });
  }

  // Natural-order DFT with twiddles tw (omega^j, j < n / 2) of x after bit_reverse.
  void stages(sycl::queue& q, Fr* x, const Fr* tw) const {
    size_t half_n = n / 2;
    for (uint32_t s = 0; s < lg; s++) {
      uint32_t sh = s;
      uint32_t stride_lg = lg - 1 - s; // twiddle index scale: n / (2 * half)
      q.parallel_for(sycl::range<1>(half_n), [=](sycl::id<1> id) {
        size_t t = id[0];
        size_t half = (size_t)1 << sh;
        size_t j = t & (half - 1);
        size_t i0 = ((t >> sh) << (sh + 1)) + j;
        size_t i1 = i0 + half;
        Fr w = tw[j << stride_lg];
        Fr u = x[i0];
        Fr v = x[i1] * w;
        x[i0] = u + v;
        x[i1] = u - v;
      });
    }
  }

  // Evaluations over the domain to evaluations over the coset shift * domain.
  void to_coset(sycl::queue& q, Fr* x) const {
    bit_reverse(q, x);
    stages(q, x, tw_inv.ptr);
    const Fr* cs = coset.ptr;
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) { x[id[0]] *= cs[id[0]]; });
    bit_reverse(q, x);
    stages(q, x, tw_fwd.ptr);
  }
};

// ---------------------------------------------------------------------------------------------
// ChaCha fuzz (the CUDA prover's generate_random_scalars, bit for bit)
// ---------------------------------------------------------------------------------------------

struct ChachaState {
  uint32_t d[16];
};

inline void chacha_quarter(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  a += b;
  d ^= a;
  d = (d << 16) | (d >> 16);
  c += d;
  b ^= c;
  b = (b << 12) | (b >> 20);
  a += b;
  d ^= a;
  d = (d << 8) | (d >> 24);
  c += d;
  b ^= c;
  b = (b << 7) | (b >> 25);
}

// Fills out[0 .. 2 * blocks) (blocks = ceil(n * 32 / 64)); the caller uses the first n.
inline void generate_fuzz(sycl::queue& q, Fr* out, size_t blocks) {
  ChachaState init;
  init.d[0] = 0x61707865;
  init.d[1] = 0x3320646e;
  init.d[2] = 0x79622d32;
  init.d[3] = 0x6b206574;
  for (int i = 4; i < 16; i++)
    init.d[i] = 0; // zero key, counter, nonce
  q.parallel_for(sycl::range<1>(blocks), [=](sycl::id<1> id) {
    size_t i = id[0];
    ChachaState s = init;
    s.d[12] += (uint32_t)i;
    if (s.d[12] < (uint32_t)i)
      s.d[13]++;
    s.d[13] += (uint32_t)((uint64_t)i >> 32);
    ChachaState t = s;
    for (int r = 0; r < 4; r++) { // ChaCha8
      chacha_quarter(s.d[0], s.d[4], s.d[8], s.d[12]);
      chacha_quarter(s.d[1], s.d[5], s.d[9], s.d[13]);
      chacha_quarter(s.d[2], s.d[6], s.d[10], s.d[14]);
      chacha_quarter(s.d[3], s.d[7], s.d[11], s.d[15]);
      chacha_quarter(s.d[0], s.d[5], s.d[10], s.d[15]);
      chacha_quarter(s.d[1], s.d[6], s.d[11], s.d[12]);
      chacha_quarter(s.d[2], s.d[7], s.d[8], s.d[13]);
      chacha_quarter(s.d[3], s.d[4], s.d[9], s.d[14]);
    }
    for (int j = 0; j < 16; j++)
      s.d[j] += t.d[j];
    Fr v0 = Fr::from_limbs(&s.d[0]);
    Fr v1 = Fr::from_limbs(&s.d[8]);
    v0.l[7] &= 0x1FFFFFFFu;
    v1.l[7] &= 0x1FFFFFFFu;
    // The CUDA kernel's warp-interleaved layout (WARP_SZ = 32).
    size_t lane = i % 32, wid = i / 32;
    size_t idx = lane + wid * 64;
    out[idx] = v0;
    if (idx + 32 < blocks * 2)
      out[idx + 32] = v1;
  });
}

// ---------------------------------------------------------------------------------------------
// MSM (Pippenger)
// ---------------------------------------------------------------------------------------------

// Kernel properties for the MSM's point-arithmetic kernels: the large register file (256 GRF),
// since each work-item holds several 256-bit (512-bit for G2) values and spills otherwise.
#ifndef R0G16_GRF
#define R0G16_GRF 256
#endif
template <class K> inline void heavy_parallel_for(sycl::queue& q, size_t n, K kernel) {
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::parallel_for(
      q,
      syclex::launch_config{sycl::range<1>(n), syclex::properties{intelex::grf_size<R0G16_GRF>}},
      kernel);
}

constexpr uint32_t MSM_C = 16;                      // window bits
constexpr uint32_t MSM_WINDOWS = (256 + MSM_C - 1) / MSM_C;
constexpr uint32_t MSM_BUCKETS = 1u << MSM_C;      // bucket 0 unused
constexpr uint32_t MSM_SEG = 64;                    // buckets per reduction work-item
constexpr uint32_t MSM_SEGS = MSM_BUCKETS / MSM_SEG;

inline uint32_t scalar_digit(const Fr& s, uint32_t w) {
  uint32_t bit = w * MSM_C;
  return (s.l[bit / 32] >> (bit % 32)) & (MSM_BUCKETS - 1);
}

// Scratch reused across MSMs (sized for the largest).
struct MsmScratch {
  DeviceBuffer<uint32_t> hist;   // [window][bucket] counts, then fill cursors
  DeviceBuffer<uint32_t> offset; // [window][bucket] start in sorted
  DeviceBuffer<uint32_t> sorted; // [window][n] point indices by bucket
  DeviceBuffer<uint8_t> buckets; // [window][bucket] Xyzz (G1 or G2)
  DeviceBuffer<uint8_t> partial; // [window][segment] Xyzz
  size_t cap = 0;
  size_t sorted_n = 0; // the scalar count msm_sort last sorted

  MsmScratch(sycl::queue& q, size_t max_n)
      : hist(q, (size_t)MSM_WINDOWS * MSM_BUCKETS)
      , offset(q, (size_t)MSM_WINDOWS * MSM_BUCKETS)
      , sorted(q, (size_t)MSM_WINDOWS * max_n)
      , buckets(q, (size_t)MSM_WINDOWS * MSM_BUCKETS * sizeof(G2Xyzz))
      , partial(q, (size_t)MSM_WINDOWS * MSM_SEGS * sizeof(G2Xyzz))
      , cap(max_n) {}
};

// With RISC0_GROTH16_INTEL_VERBOSE, a step's time (waiting for the queue after each).
struct MsmSteps {
  sycl::queue& q;
  std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();

  void operator()(const char* name) {
    if (!verbose())
      return;
    q.wait();
    auto now = std::chrono::steady_clock::now();
    std::fprintf(stderr,
                 "[groth16-intel]   msm %s: %.1f ms\n",
                 name,
                 std::chrono::duration<double, std::milli>(now - t).count());
    t = now;
  }
};

// Sort the digits of scalars[0..n) (canonical, not Montgomery) by window and bucket, for msm_sorted:
// sc.sorted then holds, per window, the scalar indices of each bucket b in
// [sc.offset[b], sc.hist[b]). MSMs over the same scalars share one sort: A, B1, B2 and C use the
// witness. Points at infinity are not left out here (msm_sorted's additions skip them), so that
// the sort depends on the scalars alone; they are 6-8% of A, B1 and B2 and none of C and H.
inline void msm_sort(sycl::queue& q, MsmScratch& sc, const Fr* scalars, size_t n) {
  if (n > sc.cap)
    throw std::runtime_error("groth16-intel: MSM larger than its scratch");
  uint32_t* hist = sc.hist.ptr;
  uint32_t* offset = sc.offset.ptr;
  uint32_t* sorted = sc.sorted.ptr;
  MsmSteps step{q};

  // 1. Count the scalars per (window, bucket).
  q.memset(hist, 0, sizeof(uint32_t) * MSM_WINDOWS * MSM_BUCKETS);
  q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
    Fr s = scalars[id[0]];
    for (uint32_t w = 0; w < MSM_WINDOWS; w++) {
      uint32_t d = scalar_digit(s, w);
      if (d) {
        sycl::atomic_ref<uint32_t,
                         sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            a(hist[(size_t)w * MSM_BUCKETS + d]);
        a.fetch_add(1u);
      }
    }
  });
  step("count");
  // 2. Exclusive prefix sums per window (one work-item per window; 64K entries each).
  q.parallel_for(sycl::range<1>(MSM_WINDOWS), [=](sycl::id<1> id) {
    size_t base = id[0] * MSM_BUCKETS;
    uint32_t acc = 0;
    for (uint32_t b = 0; b < MSM_BUCKETS; b++) {
      uint32_t c = hist[base + b];
      offset[base + b] = acc;
      hist[base + b] = acc; // becomes the fill cursor
      acc += c;
    }
  });
  step("prefix sums");
  // 3. Scatter the scalar indices into their buckets, a window at a time: each launch writes into
  // its window's n entries (22 MB for the witness) rather than all 16 windows' at once.
  for (uint32_t w = 0; w < MSM_WINDOWS; w++) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
      size_t i = id[0];
      uint32_t d = scalar_digit(scalars[i], w);
      if (d) {
        sycl::atomic_ref<uint32_t,
                         sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            a(hist[(size_t)w * MSM_BUCKETS + d]);
        uint32_t pos = a.fetch_add(1u);
        sorted[(size_t)w * n + pos] = (uint32_t)i;
      }
    });
  }
  step("scatter");
  sc.sorted_n = n;
}

// sum_i scalars[i] * points[i - first] over the scalars msm_sort sorted last, for i >= first (C's
// scalars are the witness's from index 6 on).
template <class F>
Xyzz<F> msm_sorted(sycl::queue& q, MsmScratch& sc, const Affine<F>* points, size_t first) {
  using P = Xyzz<F>;
  const size_t n = sc.sorted_n;
  const uint32_t* hist = sc.hist.ptr;
  const uint32_t* offset = sc.offset.ptr;
  const uint32_t* sorted = sc.sorted.ptr;
  P* buckets = (P*)sc.buckets.ptr;
  P* partial = (P*)sc.partial.ptr;
  MsmSteps step{q};

  // 4. Sum each bucket. hist now holds each bucket's end.
  heavy_parallel_for(q, (size_t)MSM_WINDOWS * MSM_BUCKETS, [=](sycl::id<1> id) {
    size_t wb = id[0];
    size_t w = wb / MSM_BUCKETS;
    uint32_t start = offset[wb], end = hist[wb];
    P acc = P::inf();
    const uint32_t* idx = sorted + w * n;
    for (uint32_t k = start; k < end; k++) {
      size_t i = idx[k];
      if (i >= first)
        acc = acc.add_affine(points[i - first]);
    }
    buckets[wb] = acc;
  });
  step("buckets");
  // 5. Per segment of buckets [lo, lo + SEG): sum_b b * B_b, by running sums.
  heavy_parallel_for(q, (size_t)MSM_WINDOWS * MSM_SEGS, [=](sycl::id<1> id) {
    size_t ws = id[0];
    size_t w = ws / MSM_SEGS;
    uint32_t lo = (uint32_t)(ws % MSM_SEGS) * MSM_SEG;
    const P* b = buckets + w * MSM_BUCKETS;
    P running = P::inf(), total = P::inf();
    for (uint32_t k = lo + MSM_SEG; k-- > lo;) {
      if (k == 0)
        break;
      running = running.add(b[k]);
      total = total.add(running);
    }
    // total = sum_{b in seg} (b - lo + 1) * B_b (b >= max(lo, 1)); add (lo - 1) * sum B_b.
    if (lo > 1)
      total = total.add(running.mul_small(lo - 1));
    partial[ws] = total;
  });
  step("reduce");
  // 6. On the host: the windows' sums, then sum_w 2^(C w) S_w.
  std::vector<P> host((size_t)MSM_WINDOWS * MSM_SEGS);
  q.memcpy(host.data(), partial, host.size() * sizeof(P)).wait();
  P result = P::inf();
  for (uint32_t w = MSM_WINDOWS; w-- > 0;) {
    for (uint32_t k = 0; k < MSM_C; k++)
      result = result.dbl();
    for (uint32_t s = 0; s < MSM_SEGS; s++)
      result = result.add(host[(size_t)w * MSM_SEGS + s]);
  }
  return result;
}

// sum_i scalars[i] * points[i] for canonical (non-Montgomery) scalars.
template <class F>
Xyzz<F> msm(sycl::queue& q,
            MsmScratch& sc,
            const Affine<F>* points,
            const Fr* scalars,
            size_t n) {
  msm_sort(q, sc, scalars, n);
  return msm_sorted<F>(q, sc, points, 0);
}

// ---------------------------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------------------------

// Decimal string of a canonical 256-bit integer.
inline std::string to_decimal(const uint32_t v[8]) {
  uint32_t x[8];
  std::memcpy(x, v, sizeof(x));
  std::string s;
  for (;;) {
    bool zero = true;
    for (int i = 0; i < 8; i++)
      if (x[i])
        zero = false;
    if (zero)
      break;
    uint64_t rem = 0;
    for (int i = 7; i >= 0; i--) {
      uint64_t cur = (rem << 32) | x[i];
      x[i] = (uint32_t)(cur / 10);
      rem = cur % 10;
    }
    s.push_back((char)('0' + rem));
  }
  if (s.empty())
    s = "0";
  std::reverse(s.begin(), s.end());
  return s;
}
inline std::string dec(const Fp& a) { return to_decimal(a.from_mont().l); }

inline void write_proof_file(const char* path, const Proof& p) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error(std::string("open(\"") + path + "\") failed");
  f << "{ \"pi_a\":[\"" << dec(p.a.x) << "\",\"" << dec(p.a.y) << "\",\"1\"],  \"pi_b\": [[\""
    << dec(p.b.x.c0) << "\",\"" << dec(p.b.x.c1) << "\"],[\"" << dec(p.b.y.c0) << "\",\""
    << dec(p.b.y.c1) << "\"], [\"1\",\"0\"]],  \"pi_c\": [\"" << dec(p.c.x) << "\",\""
    << dec(p.c.y) << "\",\"1\"],  \"protocol\":\"groth16\" }";
  f.flush();
  if (f.fail())
    throw std::runtime_error(std::string("failed to write the proof to \"") + path + "\"");
}

// A random blinding scalar, drawn as the CUDA prover draws r and s: 31 random bytes read as a
// Montgomery-form Fr element, converted to the canonical integer it represents.
inline void random_scalar(uint32_t out[8]) {
  static thread_local std::random_device rd;
  Fr raw;
  for (int i = 0; i < 8; i++)
    raw.l[i] = (uint32_t)rd();
  raw.l[7] &= 0x00ffffffu;
  Fr v = raw.from_mont();
  std::memcpy(out, v.l, 32);
}

class Prover {
public:
  sycl::queue& q;
  Srs srs;
  Coeffs coeffs;
  FuzzedResults fuzzed;
  NttPlan ntt;
  DeviceBuffer<Fr> fuzz;    // the ChaCha fuzz, one per witness element
  DeviceBuffer<Fr> abc;     // a, b, c over the domain
  DeviceBuffer<Fr> witness; // witness + fuzz
  MsmScratch scratch;

  Prover(sycl::queue& queue, const char* srs_path, const char* coeffs_path, const char* fres_path)
      : q(queue)
      , srs(queue, srs_path)
      , coeffs(queue, coeffs_path)
      , fuzzed(read_fuzzed_results(fres_path))
      , ntt(queue, lg2(srs.domain_size))
      , fuzz(queue, ((srs.a.len * sizeof(Fr) + 63) / 64) * 2)
      , abc(queue, 3 * srs.domain_size)
      , witness(queue, srs.a.len)
      , scratch(queue, std::max(srs.domain_size, srs.a.len)) {
    {
      Timer t("generate fuzz");
      q.memset(fuzz.ptr, 0, fuzz.len * sizeof(Fr));
      generate_fuzz(q, fuzz.ptr, fuzz.len / 2);
      q.wait();
    }
    // Run each kernel once now, so that their JIT compilation (~30 s the first time on a
    // machine; the driver caches it) happens here, in the preload, not in the first proof.
    Timer t("compile kernels");
    (void)msm<Fp>(q, scratch, srs.a.ptr, fuzz.ptr, 1);
    (void)msm<Fp2>(q, scratch, srs.b_g2.ptr, fuzz.ptr, 1);
    fill_zero(q, abc.ptr, 3 * srs.domain_size);
    witness_into_poly(q, abc.ptr, coeffs.coeffs.ptr, coeffs.indices.ptr, 2, fuzz.ptr);
    ntt.to_coset(q, abc.ptr);
    q.wait();
  }

  static uint32_t lg2(size_t n) {
    uint32_t l = 0;
    while (((size_t)1 << l) < n)
      l++;
    return l;
  }

  // witness: the circuit's full witness, canonical little-endian, srs.a.len elements.
  Proof prove(const Fr* host_witness) {
    Timer total("prove");
    size_t n = srs.domain_size;
    size_t wn = srs.a.len;
    Fr* a = abc.ptr;
    Fr* b = a + n;
    Fr* c = b + n;
    Fr* w = witness.ptr;
    {
      Timer t("witness to polynomials");
      q.memcpy(w, host_witness, wn * sizeof(Fr));
      fill_zero(q, a, 3 * n);
      witness_into_poly(q, a, coeffs.coeffs.ptr, coeffs.indices.ptr, coeffs.ind_a, w);
      witness_into_poly(q,
                        b,
                        coeffs.coeffs.ptr + coeffs.count_a,
                        coeffs.indices.ptr + coeffs.ind_a,
                        coeffs.ind_b,
                        w);
      q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) { c[id[0]] = a[id[0]] * b[id[0]]; });
      q.wait();
    }
    {
      Timer t("NTTs");
      ntt.to_coset(q, a);
      ntt.to_coset(q, b);
      ntt.to_coset(q, c);
      // h = a * b - c, then out of Montgomery form for the MSM.
      q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        size_t i = id[0];
        a[i] = (a[i] * b[i] - c[i]).from_mont();
      });
      // witness + fuzz
      const Fr* f = fuzz.ptr;
      q.parallel_for(sycl::range<1>(wn), [=](sycl::id<1> id) { w[id[0]] += f[id[0]]; });
      q.wait();
    }
    G1Xyzz h, ra, rb1, rc;
    G2Xyzz rb2;
    {
      Timer t("MSM H");
      h = msm<Fp>(q, scratch, srs.h.ptr, a, n);
    }
    // A, B1, B2 and C have the same scalars, the witness (C's from wn - c.len on): sort them once.
    if (srs.a.len != wn || srs.b_g1.len != wn || srs.b_g2.len != wn || srs.c.len > wn)
      throw std::runtime_error("groth16-intel: A, B1, B2 or C does not match the witness");
    {
      Timer t("witness sort");
      msm_sort(q, scratch, w, wn);
    }
    {
      Timer t("MSM A");
      ra = msm_sorted<Fp>(q, scratch, srs.a.ptr, 0);
    }
    {
      Timer t("MSM B1");
      rb1 = msm_sorted<Fp>(q, scratch, srs.b_g1.ptr, 0);
    }
    {
      Timer t("MSM B2");
      rb2 = msm_sorted<Fp2>(q, scratch, srs.b_g2.ptr, 0);
    }
    {
      Timer t("MSM C");
      rc = msm_sorted<Fp>(q, scratch, srs.c.ptr, wn - srs.c.len);
    }

    // Remove the fuzz (the file's results are already negated) and blind.
    ra = ra.add(G1Xyzz::from_jacobian(fuzzed.a));
    rb1 = rb1.add(G1Xyzz::from_jacobian(fuzzed.b_g1));
    rb2 = rb2.add(G2Xyzz::from_jacobian(fuzzed.b_g2));
    rc = rc.add(G1Xyzz::from_jacobian(fuzzed.c));

    uint32_t r[8], s[8], rs[8];
    random_scalar(r);
    random_scalar(s);
    {
      // rs = r * s mod the group order, via Montgomery multiplication.
      Fr fr = Fr::from_limbs(r), fs = Fr::from_limbs(s);
      Fr prod = (fr * fs) * Fr::from_limbs(FrParams::RR); // r s R^-1 * R^2 R^-1 = r s
      std::memcpy(rs, prod.l, sizeof(rs));
    }
    const VerifyingKey& vk = srs.vk;
    G1Xyzz delta1 = G1Xyzz::from_affine(vk.delta_g1);
    G2Xyzz delta2 = G2Xyzz::from_affine(vk.delta_g2);

    ra = ra.add_affine(vk.alpha_g1).add(scalar_mul(delta1, r));
    rb2 = rb2.add_affine(vk.beta_g2).add(scalar_mul(delta2, s));
    rb1 = rb1.add_affine(vk.beta_g1).add(scalar_mul(delta1, s));
    rc = rc.add(h).add(scalar_mul(ra, s)).add(scalar_mul(rb1, r)).add(scalar_mul(delta1, rs).neg());

    return Proof{ra.to_affine(), rc.to_affine(), rb2.to_affine()};
  }
};

} // namespace r0g16
