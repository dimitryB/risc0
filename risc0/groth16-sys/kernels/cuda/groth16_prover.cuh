#pragma once

#include "groth16_coeffs.cuh"
#include "groth16_srs.cuh"

#if defined(__HIP_DEVICE_COMPILE__)
// Device pass: only the __global__ kernel is needed
__global__ __launch_bounds__(512) void witness_into_poly(fr_t* out,
                                                         const coeff_t* coeffs,
                                                         slice_t<uint32_t> indices,
                                                         const fr_t* witness) {
  size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

  for (size_t i = tid; i < indices.size() - 1; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t start_idx = indices[i];
    uint32_t coeff_count = indices[i + 1] - start_idx;

    fr_t sum;
    sum.zero();

    coeff_t coeff;
    for (size_t j = 0; j < coeff_count; j++) {
      coeff = coeffs[start_idx + j];
      fr_t val = witness[coeff.s];
      fr_t aux = val * coeff.value;
      sum += aux;
    }
    out[coeff.c] = sum;
  }
}

#else // Host pass

#include <cmath>
#include <fstream>
#include <random>
#ifndef __HIPCC__
#include "msm_bit_split.cuh"
#endif
// Tuned MSM batching (cold path, see g16_cold_enabled; msm_groth16 only).
static bool g16_tune() {
#ifdef __HIPCC__
  return false;
#else
  return g16_cold_enabled();
#endif
}
// Groth16 tail (msm_groth16 only): each MSM's host collect overlaps the next MSM's GPU work
// (deferred collect), 4 windows per sort launch, split integrate (K = 8). Group-element identical
// results. The same switch also gates, on the Rust side, the key stager and witness buffer pinned
// at session start, the witness computed while the STARK HAL's pool is released, and the
// succinct receipt verified beside the wrap: RISC0_CUDA_G16_TAIL=0 disables all of them.
static bool g16_tail_enabled() {
#ifdef __HIPCC__
  return false;
#else
  static const bool on = [] {
    const char* e = getenv("RISC0_CUDA_G16_TAIL");
    return !(e && e[0] == '0');
  }();
  return on;
#endif
}
// The bit set S of the split MSMs (risc0_groth16_cuda_set_bit_signals, derived from the witness
// graph by the Rust side).
static std::mutex g16_bits_mu;
static std::vector<uint32_t>& g16_bits() { static std::vector<uint32_t> v; return v; }

// from https://github.com/iden3/rapidsnark/blob/main/src/random_generator.hpp
static inline void randombytes_buf(void* buf, size_t size) {
  // Per thread: proves on different devices can run concurrently (each under its own
  // device's lock), and the standard library's engines aren't safe to share.
  static thread_local std::random_device engine;
  // Uniform over every byte value. Not `uniform_int_distribution<uint8_t>`: the
  // standard does not allow 8-bit types there, and MSVC's library rejects them.
  static thread_local std::uniform_int_distribution<unsigned int> distr(0, 255);

  uint8_t* buffer = static_cast<uint8_t*>(buf);

  for (size_t i = 0; i < size; i++) {
    buffer[i] = static_cast<uint8_t>(distr(engine));
  }
}

// msm_t::invoke reports a CUDA failure (e.g. out of memory for its scratch buffers) by
// returning it, not by throwing: check it, or the proof is assembled from a partial sum and
// returned as if it were valid.
static inline void msm_ok(const RustError& err, const char* what) {
  if (err.code != 0) {
    std::string msg = std::string("groth16 MSM failed: ") + what;
    // RustError has no destructor: its message (strdup'd, when set) is ours to free.
    if (err.message != nullptr) {
      msg += ": ";
      msg += err.message;
      free(err.message);
    }
    throw sppark_error(err.code, msg);
  }
}

class LDE_launcher : NTT {
public:
  static void LDE_powers(fr_t* inout,
                         bool bitrev,
                         uint32_t lg_dsz,
                         uint32_t lg_blowup,
                         stream_t& stream,
                         const fr_t (*powers)[WINDOW_SIZE]) {
    size_t domain_size = (size_t)1 << lg_dsz;
    const uint32_t warpSize = gpu_props(stream).warpSize;

    if (domain_size < warpSize)
      LDE_distribute_powers<<<1, domain_size, 0, stream>>>(
          inout, lg_dsz, lg_blowup, bitrev, powers);
    else if (lg_dsz < 32)
      LDE_distribute_powers<<<domain_size / warpSize, warpSize, 0, stream>>>(
          inout, lg_dsz, lg_blowup, bitrev, powers);
    else
      LDE_distribute_powers<<<stream.sm_count(), 1024, 0, stream>>>(
          inout, lg_dsz, lg_blowup, bitrev, powers);

    CUDA_OK(cudaGetLastError());
  }
};

#ifdef __HIPCC__
// HIP host pass: provide empty body to generate device launch stub.
// Real kernel body is compiled in the device pass above.
__global__ __launch_bounds__(512) void witness_into_poly(fr_t* out,
                                                         const coeff_t* coeffs,
                                                         slice_t<uint32_t> indices,
                                                         const fr_t* witness) {}
__global__ __launch_bounds__(64) void diag_g2_double(uint32_t* out,
                                                      const uint32_t* inp) {}
#else
__global__ __launch_bounds__(512) void witness_into_poly(fr_t* out,
                                                         const coeff_t* coeffs,
                                                         slice_t<uint32_t> indices,
                                                         const fr_t* witness) {

  size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

  for (size_t i = tid; i < indices.size() - 1; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t start_idx = indices[i];
    // There will always be some coeffs
    uint32_t coeff_count = indices[i + 1] - start_idx;

    fr_t sum;
    sum.zero();

    coeff_t coeff;
    for (size_t j = 0; j < coeff_count; j++) {
      coeff = coeffs[start_idx + j];
      fr_t val = witness[coeff.s];
      fr_t aux = val * coeff.value;
      sum += aux;
    }

    // "c" field of the coeffs gone over in above loop will be equal to each other
    out[coeff.c] = sum;
  }
}
#endif

#ifndef __HIPCC__
// witness_into_poly over the compact coefficients; same summation order, bit-exact.
__global__ __launch_bounds__(512) void witness_into_poly_compact(fr_t* out, const uint2* cs,
    const fr_t* table, slice_t<uint32_t> indices, const uint32_t* rowc, const fr_t* witness) {
  size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
  for (size_t i = tid; i < indices.size() - 1; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t start_idx = indices[i], cnt = indices[i + 1] - start_idx;
    fr_t sum;
    sum.zero();
    for (size_t j = 0; j < cnt; j++) {
      uint2 c = cs[start_idx + j];
      fr_t val = witness[c.x];
      fr_t aux = val * table[c.y];
      sum += aux;
    }
    out[rowc[i]] = sum;
  }
}
#endif

struct groth16_proof {
  affine_t a, c;
  affine_fp2_t b;
};

void write_public_file(const char* file_path, slice_t<fr_t> public_vars) {
  std::ofstream public_file(file_path);
  if (!public_file.is_open()) {
    throw sppark_error{errno, "open(\"%s\") failed: ", file_path};
  }

  public_file << "[";
  for (size_t i = 0; i < public_vars.size(); i++) {
    fr_t v = public_vars[i];
    v.to();
#ifndef __CUDA_ARCH__
    public_file << "\"" << v << "\"";
#endif
    if (i < public_vars.size() - 1) {
      public_file << ",";
    }
  }
  public_file << "]";

  public_file.flush();

  if (public_file.fail()) {
    throw sppark_error{EIO, "failed to write public inputs to \"%s\": ", file_path};
  }
}

void write_proof_file(const char* file_path, groth16_proof proof) {
  union proof_and_fp_t {
    groth16_proof proof;
    struct {
      fp_t a[2], c[2], b[4];
    };
  };

  std::ofstream proof_file(file_path);
  if (!proof_file.is_open()) {
    throw sppark_error{errno, "open(\"%s\") failed: ", file_path};
  }

#ifndef __CUDA_ARCH__
  proof_and_fp_t p = proof_and_fp_t{proof};

  proof_file << "{ \"pi_a\":[\"" << p.a[0] << "\",\"" << p.a[1] << "\",\"1\"],  \"pi_b\": [[\""
             << p.b[0] << "\",\"" << p.b[1] << "\"],[\"" << p.b[2] << "\",\"" << p.b[3]
             << "\"], [\"1\",\"0\"]],  \"pi_c\": [\"" << p.c[0] << "\",\"" << p.c[1]
             << "\",\"1\"],  \"protocol\":\"groth16\" }";
#endif

  proof_file.flush();

  if (proof_file.fail()) {
    throw sppark_error{EIO, "failed to write proof to \"%s\": ", file_path};
  }
}

class groth16_prover {
private:
  struct msm_results {
    point_t a, b_g1, c, h;
    point_fp2_t b_g2;
  };

  const gpu_t& gpu;

public:
  const SRS& srs;
  preprocessed_coeffs coeffs;
  size_t witness_size;

  uint32_t lg_domain_size;
  gpu_ptr_t<fr_t> d_poly_a_b_c, d_fuzz, _partial_shift_powers;

  fr_t *d_a, *d_b, *d_c, (*partial_shift_powers)[WINDOW_SIZE];

  msm_t<bucket_t, point_t, affine_t, fr_t> msm_g1;
  msm_t<bucket_fp2_t, point_fp2_t, affine_fp2_t, fr_t> msm_g2;
  msm_results fuzzed_results;

  gpu_ptr_t<fr_t> d_witness;

#ifndef __HIPCC__
  // Bit-split MSMs (msm_bit_split.cuh), with the bit set from risc0_groth16_cuda_set_bit_signals.
  // The SRS points of a, b_g1, b_g2 (over [0,n)) and c (over [0,c_size)) are permuted IN PLACE
  // to [non-bit | bit] order (no extra 1.8 GB copies); the corrections -MSM(P_nb, f_nb) are
  // cached in a small file (keyed by the permutation, the sizes, sampled SRS bytes and the fuzz),
  // so a cold init costs no MSMs. The fallback path runs on the permuted points with permuted
  // scalars.
  bool split_on = false;
  size_t split_nb = 0, split_nn = 0, split_nb_c = 0, split_nn_c = 0, split_off_c = 0;
  gpu_ptr_t<uint32_t> split_perm, split_perm_c;   // [nbidx | bidx]
  gpu_ptr_t<fr_t> split_sn, split_sn_c;           // permuted scalars (w + f); [0,nn) used by split
  gpu_ptr_t<int> split_bad;
  msm_split_t<bucket_t, point_t, affine_t> sp_a, sp_b1, sp_c;
  msm_split_t<bucket_fp2_t, point_fp2_t, affine_fp2_t> sp_b2;

  // The corrections cache file (disk_cache.cuh).
  struct corr_file_t {
    char magic[8];
    uint32_t version, point_size;
    uint64_t key;
    point_t a, b1, c;
    point_fp2_t b2;
    uint64_t sum;  // FNV-1a of magic..b2
  };
  static constexpr char kCorrMagic[8] = {'R', '0', 'G', '1', '6', 'C', 'O', 'R'};
  static uint64_t corr_sum(const corr_file_t& cf) {
    return g16_cache::fnv1a(g16_cache::kFnvBasis, &cf, (const char*)&cf.sum - (const char*)&cf);
  }
  bool load_corr(const std::string& path, uint64_t key) {
    if (path.empty()) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    corr_file_t cf;
    f.read((char*)&cf, sizeof cf);
    if (f.gcount() != (std::streamsize)sizeof cf || memcmp(cf.magic, kCorrMagic, 8) ||
        cf.version != g16_cache::kFormatVersion || cf.point_size != sizeof(point_t) ||
        cf.key != key || cf.sum != corr_sum(cf))
      return false;
    sp_a.corr = cf.a; sp_b1.corr = cf.b1; sp_c.corr = cf.c; sp_b2.corr = cf.b2;
    return true;
  }
  void store_corr(const std::string& path, uint64_t key) {
    if (path.empty()) return;
    corr_file_t cf;
    memset((void*)&cf, 0, sizeof cf);
    memcpy(cf.magic, kCorrMagic, 8);
    cf.version = g16_cache::kFormatVersion;
    cf.point_size = sizeof(point_t);
    cf.key = key; cf.a = sp_a.corr; cf.b1 = sp_b1.corr; cf.c = sp_c.corr; cf.b2 = sp_b2.corr;
    cf.sum = corr_sum(cf);
    std::string tmp = g16_cache::tmp_path(path);
    {
      std::ofstream f(tmp, std::ios::binary);
      f.write((const char*)&cf, sizeof cf);
      if (!f.good()) { f.close(); remove(tmp.c_str()); return; }
    }
    rename(tmp.c_str(), path.c_str());
  }

  void split_init() {
    // Cards with >= 28 GB only (the split raises the peak from 19.9 to 24.4 GB).
    if (!g16_cold_enabled()) return;
    {
      size_t free_b = 0, total_b = 0;
      if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b < (size_t(28) << 30)) return;
    }
    std::vector<uint32_t> bidx;
    {
      std::lock_guard<std::mutex> lk(g16_bits_mu);
      bidx = g16_bits();
    }
    if (bidx.empty())
      return;
    size_t n = witness_size;
    if (srs.get_a_size() != n || srs.get_b_g1_size() != n || srs.get_b_g2_size() != n ||
        srs.get_c_size() > n)
      return;
    if (srs.split_permuted)
      throw sppark_error{EINVAL, "groth16: SRS points already permuted by an earlier prover: "};
    // S only affects speed (every S scalar is checked at prove time): indices that cannot be in
    // it (out of range, the constant and public signals, duplicates) are dropped, not fatal.
    std::vector<uint8_t> isb(n, 0);
    size_t nbits = 0;
    for (uint32_t i : bidx) {
      if (i >= n || i <= srs.get_num_public() || isb[i])
        continue;
      isb[i] = 1;
      nbits++;
    }
    if (nbits != bidx.size() && g16_verbose())
      fprintf(stderr, "groth16: ignored %zu of %zu bit signal indices\n", bidx.size() - nbits,
              bidx.size());
    if (nbits == 0)
      return;
    split_off_c = n - srs.get_c_size();
    std::vector<uint32_t> b, nb, bc, nbc;
    for (uint32_t i = 0; i < n; i++) {
      (isb[i] ? b : nb).push_back(i);
      if (i >= split_off_c)
        (isb[i] ? bc : nbc).push_back(i - split_off_c);
    }
    split_nb = b.size(); split_nn = nb.size(); split_nb_c = bc.size(); split_nn_c = nbc.size();
    std::vector<uint32_t> perm(nb), perm_c(nbc);
    perm.insert(perm.end(), b.begin(), b.end());
    perm_c.insert(perm_c.end(), bc.begin(), bc.end());
    auto up = [&](gpu_ptr_t<uint32_t>& d, const std::vector<uint32_t>& v) {
      d = gpu_ptr_t<uint32_t>((uint32_t*)gpu.Dmalloc(v.size() * sizeof(uint32_t)));
      gpu.sync();
      if (g16_staged())
        key_stager::get().upload(&d[0], v.data(), v.size() * sizeof(uint32_t));
      else {
        gpu.HtoD(&d[0], v.data(), v.size());
        gpu.sync();
      }
    };
    up(split_perm, perm); up(split_perm_c, perm_c);
    auto fr_alloc = [&](size_t k) { return gpu_ptr_t<fr_t>((fr_t*)gpu.Dmalloc(k * sizeof(fr_t))); };
    split_sn = fr_alloc(n); split_sn_c = fr_alloc(srs.get_c_size());
    split_bad = gpu_ptr_t<int>((int*)gpu.Dmalloc(sizeof(int)));
    sp_a.template alloc<1>(gpu);
    sp_b1.template alloc<1>(gpu);
    sp_b2.template alloc<2>(gpu);
    sp_c.template alloc<1>(gpu);
    // The permutation's staging buffer, sized for the largest points (G2); freed (stream-ordered)
    // when this returns or throws.
    struct dev_scratch {
      const gpu_t& g;
      void* p;
      ~dev_scratch() { (void)cudaFreeAsync(p, g); }
    } tmp{gpu, gpu.Dmalloc(n * sizeof(affine_fp2_t))};

    // Cache key: the cache format, the permutation, the sizes, the SRS file (srs.points_salt:
    // sampled bytes, size, mtime, inode) and the fuzz.
    uint64_t key = g16_cache::kFnvBasis;
    uint32_t format_id[3] = {g16_cache::kFormatVersion, (uint32_t)sizeof(point_t),
                       (uint32_t)sizeof(point_fp2_t)};
    key = g16_cache::fnv1a(key, format_id, sizeof format_id);
    key = g16_cache::fnv1a(key, perm.data(), perm.size() * 4);
    key = g16_cache::fnv1a(key, perm_c.data(), perm_c.size() * 4);
    size_t sizes[3] = {n, srs.get_c_size(), srs.get_num_public()};
    key = g16_cache::fnv1a(key, sizes, sizeof sizes);
    key = g16_cache::fnv1a(key, &srs.points_salt, sizeof srs.points_salt);
    key = g16_cache::fnv1a(key, &fuzzed_results, sizeof fuzzed_results);
    std::string cpath = g16_cache::path("split_corr_", key);
    bool cached = load_corr(cpath, key);
    if (!cached && g16_verbose())
      fprintf(stderr, "groth16: computing split-MSM corrections (cache %s)\n", cpath.c_str());
    gpu.sync();  // the allocations

    // Permute in place: from here on the SRS points are in split order. Anything after this that
    // throws leaves them permuted; the caller then drops the SRS (build_cached_prover in ffi.cu),
    // so a later prover reloads it instead of building on the permuted points.
    srs.split_permuted = true;
    sp_a.template permute<64>(gpu, (affine_t*)&srs.get_a()[0], &split_perm[0], n, split_nn, tmp.p);
    sp_b1.template permute<64>(gpu, (affine_t*)&srs.get_b_g1()[0], &split_perm[0], n, split_nn, tmp.p);
    sp_b2.template permute<128>(gpu, (affine_fp2_t*)&srs.get_b_g2()[0], &split_perm[0], n, split_nn, tmp.p);
    sp_c.template permute<64>(gpu, (affine_t*)&srs.get_c()[0], &split_perm_c[0], srs.get_c_size(), split_nn_c, tmp.p);
    gpu.sync();

    if (!cached) {
      // corr = -MSM(P_nb, f_nb); split_sn / split_sn_c are scratch here.
      split_gather_add<<<gpu.sm_count() * 4, 256, 0, gpu>>>(&split_sn[0], &d_fuzz[0], nullptr, &split_perm[0], split_nn);
      split_gather_add<<<gpu.sm_count() * 4, 256, 0, gpu>>>(&split_sn_c[0], &d_fuzz[split_off_c], nullptr, &split_perm_c[0], split_nn_c);
      CUDA_OK(cudaGetLastError());
      gpu.sync();
      msm_ok(msm_g1.invoke(sp_a.corr, dev_ptr_t<affine_t>{sp_a.pn}, split_nn, split_sn, false), "split fuzz a");
      msm_ok(msm_g1.invoke(sp_b1.corr, dev_ptr_t<affine_t>{sp_b1.pn}, split_nn, split_sn, false), "split fuzz b_g1");
      msm_ok(msm_g2.invoke(sp_b2.corr, dev_ptr_t<affine_fp2_t>{sp_b2.pn}, split_nn, split_sn, false), "split fuzz b_g2");
      msm_ok(msm_g1.invoke(sp_c.corr, dev_ptr_t<affine_t>{sp_c.pn}, split_nn_c, split_sn_c, false), "split fuzz c");
      sp_a.corr.cneg(true); sp_b1.corr.cneg(true); sp_b2.corr.cneg(true); sp_c.corr.cneg(true);
      store_corr(cpath, key);
    }
    split_on = true;
  }

  // true if the split MSMs ran (all S scalars were 0/1); false: the caller runs permuted_full_msms
  bool split_msms(msm_results& results) {
    const uint32_t* bidx = &split_perm[split_nn];
    const uint32_t* bidx_c = &split_perm_c[split_nn_c];
    cudaMemsetAsync(&split_bad[0], 0, sizeof(int), gpu);
    sp_a.template bit_sum<1>(gpu, &split_bad[0], &d_witness[0], bidx, split_nb);
    sp_b1.template bit_sum<1>(gpu, &split_bad[0], &d_witness[0], bidx, split_nb);
    sp_b2.template bit_sum<2>(gpu, &split_bad[0], &d_witness[0], bidx, split_nb);
    sp_c.template bit_sum<1>(gpu, &split_bad[0], &d_witness[split_off_c], bidx_c, split_nb_c);
    split_gather_add<<<gpu.sm_count() * 4, 256, 0, gpu>>>(&split_sn[0], &d_witness[0], &d_fuzz[0], &split_perm[0], split_nn);
    split_gather_add<<<gpu.sm_count() * 4, 256, 0, gpu>>>(&split_sn_c[0], &d_witness[split_off_c], &d_fuzz[split_off_c], &split_perm_c[0], split_nn_c);
    CUDA_OK(cudaGetLastError());
    int bad = -1;
    bucket_t ba, bb1, bc;
    bucket_fp2_t bb2;
    gpu.DtoH(&bad, &split_bad[0], 1);
    gpu.DtoH(&ba, &sp_a.acc[0], 1);
    gpu.DtoH(&bb1, &sp_b1.acc[0], 1);
    gpu.DtoH(&bb2, &sp_b2.acc[0], 1);
    gpu.DtoH(&bc, &sp_c.acc[0], 1);
    gpu.sync();
    if (bad != 0) {
      if (g16_verbose())
        fprintf(stderr, "groth16: %d bit-split scalars are not 0/1; using the full MSMs\n", bad);
      return false;
    }
    if (g16_tail_enabled()) {
      // GPU: a, b_g1, b_g2, c back to back; host: each collect on its own thread meanwhile.
      typename decltype(msm_g1)::pending_t pa, pb1, pc;
      typename decltype(msm_g2)::pending_t pb2;
      std::vector<std::thread> th;
      auto join_all = [&] {
        for (auto& t : th)
          if (t.joinable()) t.join();
      };
      try {
        msm_ok(msm_g1.invoke_defer(pa, dev_ptr_t<affine_t>{sp_a.pn}, split_nn, split_sn, false), "a");
        th.emplace_back([&] { msm_g1.finish(results.a, pa); });
        msm_ok(msm_g1.invoke_defer(pb1, dev_ptr_t<affine_t>{sp_b1.pn}, split_nn, split_sn, false), "b_g1");
        th.emplace_back([&] { msm_g1.finish(results.b_g1, pb1); });
        msm_ok(msm_g2.invoke_defer(pb2, dev_ptr_t<affine_fp2_t>{sp_b2.pn}, split_nn, split_sn, false), "b_g2");
        th.emplace_back([&] { msm_g2.finish(results.b_g2, pb2); });
        msm_ok(msm_g1.invoke_defer(pc, dev_ptr_t<affine_t>{sp_c.pn}, split_nn_c, split_sn_c, false), "c");
        th.emplace_back([&] { msm_g1.finish(results.c, pc); });
      } catch (...) {
        join_all();
        throw;
      }
      join_all();
    } else {
      msm_ok(msm_g1.invoke(results.a, dev_ptr_t<affine_t>{sp_a.pn}, split_nn, split_sn, false), "a");
      msm_ok(msm_g1.invoke(results.b_g1, dev_ptr_t<affine_t>{sp_b1.pn}, split_nn, split_sn, false), "b_g1");
      msm_ok(msm_g2.invoke(results.b_g2, dev_ptr_t<affine_fp2_t>{sp_b2.pn}, split_nn, split_sn, false), "b_g2");
      msm_ok(msm_g1.invoke(results.c, dev_ptr_t<affine_t>{sp_c.pn}, split_nn_c, split_sn_c, false), "c");
    }
    point_t p;
    p = ba; results.a.add(p); results.a.add(sp_a.corr);
    p = bb1; results.b_g1.add(p); results.b_g1.add(sp_b1.corr);
    p = bc; results.c.add(p); results.c.add(sp_c.corr);
    point_fp2_t p2;
    p2 = bb2; results.b_g2.add(p2); results.b_g2.add(sp_b2.corr);
    return true;
  }

  // The unsplit MSMs over the permuted points: scalars (w + f) permuted the same way.
  void permuted_full_msms(msm_results& results) {
    size_t n = witness_size, nc = srs.get_c_size();
    split_gather_add<<<gpu.sm_count() * 4, 256, 0, gpu>>>(&split_sn[0], &d_witness[0], &d_fuzz[0], &split_perm[0], n);
    split_gather_add<<<gpu.sm_count() * 4, 256, 0, gpu>>>(&split_sn_c[0], &d_witness[split_off_c], &d_fuzz[split_off_c], &split_perm_c[0], nc);
    CUDA_OK(cudaGetLastError());
    gpu.sync();
    msm_ok(msm_g1.invoke(results.a, dev_ptr_t<affine_t>{sp_a.pn}, n, split_sn, false), "a");
    msm_ok(msm_g1.invoke(results.b_g1, dev_ptr_t<affine_t>{sp_b1.pn}, n, split_sn, false), "b_g1");
    msm_ok(msm_g2.invoke(results.b_g2, dev_ptr_t<affine_fp2_t>{sp_b2.pn}, n, split_sn, false), "b_g2");
    msm_ok(msm_g1.invoke(results.c, dev_ptr_t<affine_t>{sp_c.pn}, nc, split_sn_c, false), "c");
    results.a.add(fuzzed_results.a);
    results.b_g1.add(fuzzed_results.b_g1);
    results.b_g2.add(fuzzed_results.b_g2);
    results.c.add(fuzzed_results.c);
  }
#endif

  void NTT_sequence(stream_t& stream, fr_t* d_poly, uint32_t lg_domain_size) {
    NTT::Base_dev_ptr(stream,
                      d_poly,
                      lg_domain_size,
                      NTT::InputOutputOrder::NR,
                      NTT::Direction::inverse,
                      NTT::Type::standard);
    LDE_launcher::LDE_powers(d_poly, true, lg_domain_size, 0, stream, partial_shift_powers);
    NTT::Base_dev_ptr(stream,
                      d_poly,
                      lg_domain_size,
                      NTT::InputOutputOrder::RN,
                      NTT::Direction::forward,
                      NTT::Type::standard);
  }

  void write_fuzzed_results_to_file(const char* file_path) {
    std::ofstream file(file_path, std::ofstream::binary);
    if (!file.is_open()) {
      throw sppark_error{errno, "open(\"%s\") failed: ", file_path};
    }

    file.write((const char*)&fuzzed_results, sizeof(msm_results));
    file.flush();

    if (file.fail()) {
      throw sppark_error{EIO, "failed to write fuzzed results to \"%s\": ", file_path};
    }
  }

  msm_results read_fuzzed_results_from_file(const char* file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
      throw sppark_error{errno, "open(\"%s\") failed: ", file_path};
    }

    msm_results fuzzed_results;

    file.read((char*)&fuzzed_results, sizeof(fuzzed_results));

    if (file.gcount() != sizeof(msm_results)) {
      throw sppark_error{
          EINVAL,
          "fuzzed results file(\"%s\") is not the correct size. expected %zu, got %zu: ",
          file_path,
          sizeof(msm_results),
          file.gcount()};
    }

    return fuzzed_results;
  }

  void init_memory() {
    d_poly_a_b_c = gpu_ptr_t<fr_t>((fr_t*)gpu.Dmalloc(3 * srs.get_h_size() * sizeof(fr_t)));
    _partial_shift_powers =
        gpu_ptr_t<fr_t>((fr_t*)gpu.Dmalloc(WINDOW_NUM * WINDOW_SIZE * sizeof(fr_t)));
    d_witness = gpu_ptr_t<fr_t>((fr_t*)gpu.Dmalloc(witness_size * sizeof(fr_t)));

    size_t domain_size = srs.get_h_size();
    if ((domain_size & (domain_size - 1)) != 0) {
      throw sppark_error{
          EINVAL, "size of h is expected to be a power of 2, but is %zu: ", domain_size};
    }
    lg_domain_size = std::log2(domain_size);

    d_a = &d_poly_a_b_c[0];
    d_b = &d_a[domain_size];
    d_c = &d_b[domain_size];

    partial_shift_powers =
        reinterpret_cast<decltype(partial_shift_powers)>(&_partial_shift_powers[0]);

    fr_t shift = forward_roots_of_unity[lg_domain_size + 1];
    generate_partial_twiddles<<<WINDOW_SIZE / 32, 32, 0, gpu>>>(partial_shift_powers, shift);
    CUDA_OK(cudaGetLastError());

    size_t fuzz_size = srs.get_a_size();
    d_fuzz = generate_random_scalars(gpu, fuzz_size);
  }

  groth16_prover(const SRS& srs,
                 const char* preprocessed_coeffs_file,
                 const char* fuzzed_results_file)
      : gpu(srs.gpu)
      , srs(srs)
      , coeffs(gpu, preprocessed_coeffs_file)
      , witness_size(srs.get_a_size())
      , msm_g1(nullptr, g16_tune() ? 2 * srs.get_h_size() : srs.get_h_size())
      , msm_g2(nullptr, srs.get_b_g2_size()) {
    init_memory();
#ifndef __HIPCC__  // msm_groth16 only (g16_tune() is false under HIP)
    if (g16_tune()) {
      msm_g1.set_batch(1);
      msm_g2.set_batch(1);
    }
    if (g16_tail_enabled()) {
      msm_g1.set_tail(4, 8);
      msm_g2.set_tail(4, 8);
    }
#endif

    fuzzed_results = read_fuzzed_results_from_file(fuzzed_results_file);
#ifndef __HIPCC__
    split_init();
#endif
  }

#ifdef SRS_READ_COEFFS
  groth16_prover(const SRS& srs)
      : gpu(srs.gpu)
      , srs(srs)
      , coeffs(gpu, srs.get_coeffs_slice())
      , witness_size(srs.get_a_size())
      , msm_g1(nullptr, srs.get_h_size())
      , msm_g2(nullptr, srs.get_b_g2_size()) {
    init_memory();

    msm_ok(msm_g1.invoke(fuzzed_results.a, srs.get_a(), srs.get_a_size(), d_fuzz, false),
           "fuzz a");
    msm_ok(msm_g1.invoke(fuzzed_results.b_g1, srs.get_b_g1(), srs.get_b_g1_size(), d_fuzz, false),
           "fuzz b_g1");
    msm_ok(msm_g2.invoke(fuzzed_results.b_g2, srs.get_b_g2(), srs.get_b_g2_size(), d_fuzz, false),
           "fuzz b_g2");
    msm_ok(msm_g1.invoke(fuzzed_results.c,
                         srs.get_c(),
                         srs.get_c_size(),
                         dev_ptr_t<fr_t>{&d_fuzz[witness_size - srs.get_c_size()]},
                         false),
           "fuzz c");

    fuzzed_results.a.cneg(true);
    fuzzed_results.b_g1.cneg(true);
    fuzzed_results.b_g2.cneg(true);
    fuzzed_results.c.cneg(true);
  }
#endif

  groth16_proof prove(const char* public_file_path, const fr_t* witness) {
    size_t domain_size = (size_t)1 << lg_domain_size;

    msm_results results;
    gpu.HtoD(&d_witness[0], &witness[0], witness_size);

    std::exception_ptr eptr = nullptr;
    std::thread write_public_file_thread = std::thread([&] {
      try {
        write_public_file(public_file_path, slice_t<fr_t>{&witness[1], srs.get_num_public()});
      } catch (const sppark_error& e) {
        eptr = std::current_exception();
      }
    });

    try {
      event_t sync_event;

      // The witness is uploaded on `gpu` (sppark's zero stream), but witness_into_poly for d_a
      // runs on gpu[0], another non-blocking stream: make it wait for the upload. Without this
      // it can read d_witness before the copy lands (a stale witness from the previous proof);
      // from pageable memory cudaMemcpyAsync returns before its DMA completes.
      {
        event_t h2d_done;
        h2d_done.record(gpu);
        h2d_done.wait(gpu[0]);
      }

      // witness_into_poly: one row per thread in a grid-stride loop, so the launch shape does not
      // change the result; 12 x 256-thread blocks per SM beat 1 x 512 (measured: A+B
      // 4.03 -> 3.63 ms with the compact coefficients).
#ifndef __HIPCC__
      const unsigned wip_grid = 12 * gpu.sm_count(), wip_block = 256;
#else
      const unsigned wip_grid = gpu.sm_count(), wip_block = 512;
#endif
      gpu[0].bzero(&d_a[0], domain_size);
#ifndef __HIPCC__
      if (coeffs.compact)
        witness_into_poly_compact<<<wip_grid, wip_block, 0, gpu[0]>>>(
            &d_a[0], coeffs.get_cs_a(), &coeffs.d_table[0], coeffs.get_indices_a(),
            coeffs.get_rowc_a(), &d_witness[0]);
      else
#endif
      witness_into_poly<<<wip_grid, wip_block, 0, gpu[0]>>>(
          &d_a[0], coeffs.get_coeffs_a().data(), coeffs.get_indices_a(), &d_witness[0]);
      CUDA_OK(cudaGetLastError());
      sync_event.record(gpu[0]);

      gpu.bzero(&d_b[0], domain_size);
#ifndef __HIPCC__
      if (coeffs.compact)
        witness_into_poly_compact<<<wip_grid, wip_block, 0, gpu>>>(
            &d_b[0], coeffs.get_cs_b(), &coeffs.d_table[0], coeffs.get_indices_b(),
            coeffs.get_rowc_b(), &d_witness[0]);
      else
#endif
      witness_into_poly<<<wip_grid, wip_block, 0, gpu>>>(
          &d_b[0], coeffs.get_coeffs_b().data(), coeffs.get_indices_b(), &d_witness[0]);
      CUDA_OK(cudaGetLastError());

      sync_event.wait(gpu);

      gpu.bzero(&d_c[0], domain_size);
      coeff_wise_mul<<<gpu.sm_count(), 1024, 0, gpu>>>(&d_c[0], &d_a[0], &d_b[0], lg_domain_size);
      CUDA_OK(cudaGetLastError());
      NTT_sequence(gpu, d_a, lg_domain_size);
      NTT_sequence(gpu, d_b, lg_domain_size);
      NTT_sequence(gpu, d_c, lg_domain_size);
      coeff_wise_mul_and_sub<<<gpu.sm_count(), 1024, 0, gpu>>>(
          &d_a[0], &d_b[0], &d_c[0], lg_domain_size);
      CUDA_OK(cudaGetLastError());
      gpu.sync();
      // With the MSM tail, h's host collect runs on a thread while the next MSMs use the GPU;
      // joined before anything reads results.h (also on unwind).
      std::thread th_h;
      struct join_on_exit {
        std::thread& t;
        ~join_on_exit() {
          if (t.joinable()) t.join();
        }
      } th_h_guard{th_h};
#ifndef __HIPCC__
      typename decltype(msm_g1)::pending_t pend_h;
      if (g16_tail_enabled()) {
        msm_ok(msm_g1.invoke_defer(pend_h, srs.get_h(), domain_size, d_poly_a_b_c), "h");
        th_h = std::thread([&] { msm_g1.finish(results.h, pend_h); });
      } else
#endif
      msm_ok(msm_g1.invoke(results.h, srs.get_h(), domain_size, d_poly_a_b_c), "h");

      bool split_done = false;
#ifndef __HIPCC__
      if (split_on)
        split_done = split_msms(results);
#endif
      if (th_h.joinable()) th_h.join();
#ifndef __HIPCC__
      if (split_on && !split_done) {
        permuted_full_msms(results);
      } else
#endif
      if (!split_done) {
      coeff_wise_add<<<gpu.sm_count(), 1024, 0, gpu>>>(&d_witness[0], &d_fuzz[0], witness_size);
      CUDA_OK(cudaGetLastError());
      gpu.sync();

      msm_ok(msm_g1.invoke(results.a, srs.get_a(), srs.get_a_size(), d_witness, false), "a");
      msm_ok(msm_g1.invoke(results.b_g1, srs.get_b_g1(), srs.get_b_g1_size(), d_witness, false),
             "b_g1");
      msm_ok(msm_g2.invoke(results.b_g2, srs.get_b_g2(), srs.get_b_g2_size(), d_witness, false),
             "b_g2");
      msm_ok(msm_g1.invoke(results.c,
                           srs.get_c(),
                           srs.get_c_size(),
                           dev_ptr_t<fr_t>{&d_witness[witness_size - srs.get_c_size()]},
                           false),
             "c");
      results.a.add(fuzzed_results.a);
      results.b_g1.add(fuzzed_results.b_g1);
      results.b_g2.add(fuzzed_results.b_g2);
      results.c.add(fuzzed_results.c);
      }

      write_public_file_thread.join();
      if (eptr != nullptr) {
        std::rethrow_exception(eptr);
      }
    } catch (const sppark_error& e) {
      if (write_public_file_thread.joinable())
        write_public_file_thread.join();
      throw;
    }

    fr_t r, s;
    r.zero();
    s.zero();
    randombytes_buf((void*)&r, sizeof(fr_t) - 1);
    randombytes_buf((void*)&s, sizeof(fr_t) - 1);

    point_t p1;
    point_fp2_t p2;

    const verifying_key& vk = srs.get_vk();

    results.a.add(vk.alpha_g1);
    mult(p1, vk.delta_g1, r);
    results.a.add(p1);

    results.b_g2.add(vk.beta_g2);
    mult(p2, vk.delta_g2, s);
    results.b_g2.add(p2);

    results.b_g1.add(vk.beta_g1);
    mult(p1, vk.delta_g1, s);
    results.b_g1.add(p1);

    results.c.add(results.h);

    mult(p1, results.a, s);
    results.c.add(p1);

    mult(p1, results.b_g1, r);
    results.c.add(p1);

    fr_t rs = r * s;

    mult(p1, vk.delta_g1, rs);
    p1.cneg(true);
    results.c.add(p1);

    return groth16_proof{results.a, results.c, results.b_g2};
  }

  void write_precomputations_to_file(const char* fuzzed_results_path,
                                     const char* preprocessed_coeffs_path) {
    write_fuzzed_results_to_file(fuzzed_results_path);
    coeffs.write_to_file(preprocessed_coeffs_path);
  }
};

#endif // !__HIP_DEVICE_COMPILE__  (outer guard from top of file)
