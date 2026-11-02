#pragma once

#include "pinned_file.cuh"

#ifdef _WIN32
#include <cerrno>
#include <cstdio>
#include <sys/stat.h>
#include <vector>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "disk_cache.cuh"

typedef uint8_t byte;

struct coeff_t {
  uint32_t m, c, s;
  fr_t value;
};

#if !defined(__HIP_DEVICE_COMPILE__)
class preprocessed_coeffs {
private:
  const gpu_t& gpu;

  gpu_ptr_t<coeff_t> d_coeffs;
  slice_t<coeff_t> d_coeffs_a, d_coeffs_b;

  gpu_ptr_t<uint32_t> d_independent_indices;
  slice_t<uint32_t> d_ind_idx_a, d_ind_idx_b;

public:
  // Compact form (cold path, see g16_cold_enabled): per coefficient (s, value index) into a table of
  // the 16,413 distinct values, plus each row's output index c. 278 MB instead of 1.44 GB;
  // derived once from the preprocessed file and cached (keyed by sampled file bytes).
  bool compact = false;
  gpu_ptr_t<uint2> d_cs;
  gpu_ptr_t<fr_t> d_table;
  gpu_ptr_t<uint32_t> d_rowc;   // [rows a (ua entries) | rows b (ub entries)]
private:

public:
  preprocessed_coeffs(const gpu_t& gpu, const slice_t<coeff_t>& coeffs_slice) : gpu(gpu) {
    size_t coeff_count_a = 0, unique_count_a = 0, coeff_count_b = 0, unique_count_b = 0,
           prev_a = (size_t)-1, prev_b = (size_t)-1;

    for (size_t i = 0; i < coeffs_slice.size(); i++) {
      bool is_a = coeffs_slice[i].m == 0;

      if (is_a) {
        coeff_count_a++;
        if (coeffs_slice[i].c != prev_a) {
          prev_a = coeffs_slice[i].c;
          unique_count_a++;
        }
      } else {
        coeff_count_b++;
        if (coeffs_slice[i].c != prev_b) {
          prev_b = coeffs_slice[i].c;
          unique_count_b++;
        }
      }

      if (i > 0)
        if (coeffs_slice[i].c < coeffs_slice[i - 1].c) {
          throw sppark_error{EINVAL,
                             "coeff[%zu].c < coeff[%zu].c (%u < %u) coeffs are expected to be in "
                             "ascending order with respect to their \"c\" field: ",
                             i,
                             i - 1,
                             coeffs_slice[i].c,
                             coeffs_slice[i - 1].c};
        }
    }

    std::vector<coeff_t> coeffs(coeff_count_a + coeff_count_b);
    slice_mut<coeff_t> coeffs_a(&coeffs[0], coeff_count_a),
        coeffs_b(&coeffs[coeff_count_a], coeff_count_b);

    std::vector<uint32_t> independent_indices(2 + unique_count_a + unique_count_b);
    slice_mut<uint32_t> ind_idx_a(&independent_indices[0], 1 + unique_count_a),
        ind_idx_b(&independent_indices[1 + unique_count_a], 1 + unique_count_b);

    coeff_count_a = 0, unique_count_a = 0, coeff_count_b = 0, unique_count_b = 0,
    prev_a = (size_t)-1, prev_b = (size_t)-1;

    for (size_t i = 0; i < coeffs_slice.size(); i++) {
      bool is_a = coeffs_slice[i].m == 0;

      if (is_a) {
        coeffs_a[coeff_count_a] = coeffs_slice[i];

        if (coeffs_slice[i].c != prev_a) {
          prev_a = coeffs_slice[i].c;
          ind_idx_a[unique_count_a] = coeff_count_a;
          unique_count_a++;
        }
        coeff_count_a++;
      }
    }

    for (size_t i = 0; i < coeffs_slice.size(); i++) {
      bool is_a = coeffs_slice[i].m == 0;

      if (!is_a) {
        coeffs_b[coeff_count_b] = coeffs_slice[i];

        if (coeffs_slice[i].c != prev_b) {
          prev_b = coeffs_slice[i].c;
          ind_idx_b[unique_count_b] = coeff_count_b;
          unique_count_b++;
        }
        coeff_count_b++;
      }
    }

    ind_idx_a[unique_count_a] = coeff_count_a;
    ind_idx_b[unique_count_b] = coeff_count_b;

    d_coeffs = gpu_ptr_t<coeff_t>((coeff_t*)gpu.Dmalloc(coeffs.size() * sizeof(coeff_t)));
    d_independent_indices =
        gpu_ptr_t<uint32_t>((uint32_t*)gpu.Dmalloc(independent_indices.size() * sizeof(uint32_t)));

    gpu.HtoD(&d_coeffs[0], &coeffs[0], coeffs.size());
    gpu.HtoD(&d_independent_indices[0], &independent_indices[0], independent_indices.size());

    d_coeffs_a = slice_t<coeff_t>(&d_coeffs[0], coeffs_a.size());
    d_coeffs_b = slice_t<coeff_t>(&d_coeffs[coeffs_a.size()], coeffs_b.size());

    d_ind_idx_a = slice_t<uint32_t>(&d_independent_indices[0], ind_idx_a.size());
    d_ind_idx_b = slice_t<uint32_t>(&d_independent_indices[ind_idx_a.size()], ind_idx_b.size());

    gpu.sync();
  }

#ifdef _WIN32
  // Windows has no mmap(2): the file is read into, and written from, a host buffer.
  // The layout is the same as below: four size_t counts, the coefficients, then the
  // independent indices.
  static std::vector<byte> read_whole_file(const char* file_path) {
    std::FILE* file = std::fopen(file_path, "rb");
    if (!file) {
      throw sppark_error{errno, "open(\"%s\") failed: ", file_path};
    }
    struct _stat64 st;
    if (_fstat64(_fileno(file), &st) != 0) {
      int err = errno;
      std::fclose(file);
      throw sppark_error{err, "fstat(\"%s\") failed: ", file_path};
    }
    std::vector<byte> buf((size_t)st.st_size);
    size_t got = std::fread(buf.data(), 1, buf.size(), file);
    int err = errno;
    std::fclose(file);
    if (got != buf.size()) {
      throw sppark_error{err ? err : EIO, "read(\"%s\") failed: ", file_path};
    }
    return buf;
  }

  preprocessed_coeffs(const gpu_t& gpu, const char* file_path) : gpu(gpu) {
    std::vector<byte> owned;
    const byte* file_data = nullptr;
    size_t file_size = 0;
#ifdef GROTH16_PINNED_FILES
    file_data = (const byte*)groth16_pinned_file(file_path, &file_size);
#endif
    if (!file_data) {
      owned = read_whole_file(file_path);
      file_data = owned.data();
      file_size = owned.size();
    }
    if (file_size < 4 * sizeof(size_t)) {
      throw sppark_error{EINVAL, "\"%s\" is truncated: ", file_path};
    }

    try {
      {
        const size_t* ptr = (const size_t*)file_data;
        size_t coeff_count_a = ptr[0];
        size_t coeff_count_b = ptr[1];
        size_t unique_count_a = ptr[2];
        size_t unique_count_b = ptr[3];

        d_coeffs = gpu_ptr_t<coeff_t>(
            (coeff_t*)gpu.Dmalloc((coeff_count_a + coeff_count_b) * sizeof(coeff_t)));
        d_independent_indices = gpu_ptr_t<uint32_t>(
            (uint32_t*)gpu.Dmalloc((unique_count_a + unique_count_b) * sizeof(uint32_t)));

        d_coeffs_a = slice_t<coeff_t>(&d_coeffs[0], coeff_count_a);
        d_coeffs_b = slice_t<coeff_t>(&d_coeffs[coeff_count_a], coeff_count_b);

        d_ind_idx_a = slice_t<uint32_t>(&d_independent_indices[0], unique_count_a);
        d_ind_idx_b = slice_t<uint32_t>(&d_independent_indices[unique_count_a], unique_count_b);
      }

      const byte* ptr = file_data + 4 * sizeof(size_t);
      gpu.HtoD(&d_coeffs[0], (coeff_t*)&ptr[0], d_coeffs_a.size() + d_coeffs_b.size());

      ptr += (d_coeffs_a.size() + d_coeffs_b.size()) * sizeof(coeff_t);
      gpu.HtoD(
          &d_independent_indices[0], (uint32_t*)&ptr[0], d_ind_idx_a.size() + d_ind_idx_b.size());

      // The copies are asynchronous: the file's bytes must outlive them.
      gpu.sync();
    } catch (const sppark_error& e) {
      gpu.sync();
      throw;
    }
  }

  void write_to_file(const char* file_path) {
    size_t file_size = 4 * sizeof(size_t) +
                       (d_coeffs_a.size() + d_coeffs_b.size()) * sizeof(coeff_t) +
                       (d_ind_idx_a.size() + d_ind_idx_b.size()) * sizeof(uint32_t);
    std::vector<byte> file(file_size);

    try {
      {
        size_t* file_ptr = (size_t*)file.data();
        file_ptr[0] = d_coeffs_a.size();
        file_ptr[1] = d_coeffs_b.size();
        file_ptr[2] = d_ind_idx_a.size();
        file_ptr[3] = d_ind_idx_b.size();
      }

      byte* file_ptr = file.data() + 4 * sizeof(size_t);
      gpu.DtoH((coeff_t*)file_ptr, &d_coeffs[0], d_coeffs_a.size() + d_coeffs_b.size());

      file_ptr += (d_coeffs_a.size() + d_coeffs_b.size()) * sizeof(coeff_t);
      gpu.DtoH(
          (uint32_t*)file_ptr, &d_independent_indices[0], d_ind_idx_a.size() + d_ind_idx_b.size());

      gpu.sync();
    } catch (const sppark_error& e) {
      gpu.sync();
      throw;
    }

    std::FILE* out = std::fopen(file_path, "wb");
    if (!out) {
      throw sppark_error{errno, "open(\"%s\") failed: ", file_path};
    }
    size_t written = std::fwrite(file.data(), 1, file_size, out);
    int err = errno;
    if (std::fclose(out) != 0 || written != file_size) {
      throw sppark_error{err ? err : EIO, "write(\"%s\") failed: ", file_path};
    }
  }
#else
  preprocessed_coeffs(const gpu_t& gpu, const char* file_path) : gpu(gpu) {
    int pcoeffs_file = open(file_path, O_RDONLY);

    if (pcoeffs_file < 0) {
      throw sppark_error{errno, "open(\"%s\") failed: ", file_path};
    }

    struct stat st;
    if (fstat(pcoeffs_file, &st) != 0) {
      int err = errno;
      close(pcoeffs_file);
      throw sppark_error{err, "fstat(\"%s\") failed: ", file_path};
    }
    size_t file_size = st.st_size;

    const byte* _ptr = (const byte*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, pcoeffs_file, 0);
    {
      int err = errno;
      close(pcoeffs_file);
      if (_ptr == MAP_FAILED) {
        throw sppark_error{err, "mmap(\"%s\") failed: ", file_path};
      }
    }

    stream_t ld{gpu.id()};
    try {
#ifndef __HIPCC__
      if (g16_cold_enabled() && compact_load(ld, _ptr, file_size, st)) {
        munmap(const_cast<byte*>(_ptr), file_size);
        return;
      }
#endif
      {
        size_t* ptr = (size_t*)_ptr;
        size_t coeff_count_a = ptr[0];
        size_t coeff_count_b = ptr[1];
        size_t unique_count_a = ptr[2];
        size_t unique_count_b = ptr[3];

        d_coeffs = gpu_ptr_t<coeff_t>(
            (coeff_t*)ld.Dmalloc((coeff_count_a + coeff_count_b) * sizeof(coeff_t)));
        d_independent_indices = gpu_ptr_t<uint32_t>(
            (uint32_t*)ld.Dmalloc((unique_count_a + unique_count_b) * sizeof(uint32_t)));

        d_coeffs_a = slice_t<coeff_t>(&d_coeffs[0], coeff_count_a);
        d_coeffs_b = slice_t<coeff_t>(&d_coeffs[coeff_count_a], coeff_count_b);

        d_ind_idx_a = slice_t<uint32_t>(&d_independent_indices[0], unique_count_a);
        d_ind_idx_b = slice_t<uint32_t>(&d_independent_indices[unique_count_a], unique_count_b);
      }

      const byte* ptr = _ptr + 4 * sizeof(size_t);
      if (g16_staged()) {
        ld.sync();
        key_stager& sg = key_stager::get();
        sg.upload(&d_coeffs[0], ptr, (d_coeffs_a.size() + d_coeffs_b.size()) * sizeof(coeff_t));
      } else
      ld.HtoD(&d_coeffs[0], (coeff_t*)&ptr[0], d_coeffs_a.size() + d_coeffs_b.size());

      ptr += (d_coeffs_a.size() + d_coeffs_b.size()) * sizeof(coeff_t);
      if (g16_staged())
        key_stager::get().upload(&d_independent_indices[0], ptr,
                                (d_ind_idx_a.size() + d_ind_idx_b.size()) * sizeof(uint32_t));
      else
      ld.HtoD(
          &d_independent_indices[0], (uint32_t*)&ptr[0], d_ind_idx_a.size() + d_ind_idx_b.size());

      ld.sync();
      munmap(const_cast<byte*>(_ptr), file_size);
    } catch (const sppark_error& e) {
      ld.sync();
      munmap(const_cast<byte*>(_ptr), file_size);

      throw;
    }
  }

  void write_to_file(const char* file_path) {
    size_t file_size = 4 * sizeof(size_t) +
                       (d_coeffs_a.size() + d_coeffs_b.size()) * sizeof(coeff_t) +
                       (d_ind_idx_a.size() + d_ind_idx_b.size()) * sizeof(uint32_t);

    int pcoeffs_file = open(file_path, O_RDWR | O_CREAT, (mode_t)0666);
    if (pcoeffs_file == -1) {
      throw sppark_error{errno, "open(\"%s\") failed: ", file_path};
    }
    if (ftruncate(pcoeffs_file, file_size) == -1) {
      int err = errno;
      close(pcoeffs_file);
      throw sppark_error{err, "failed to set the size of \"%s\" to %zu: ", file_path, file_size};
    }
    byte* _file_ptr =
        (byte*)mmap(0, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, pcoeffs_file, 0);

    {
      int err = errno;
      close(pcoeffs_file);
      if (_file_ptr == MAP_FAILED) {
        throw sppark_error{err, "mmap(\"%s\") failed: ", file_path};
      }
    }

    try {
      {
        size_t* file_ptr = (size_t*)_file_ptr;
        file_ptr[0] = d_coeffs_a.size();
        file_ptr[1] = d_coeffs_b.size();
        file_ptr[2] = d_ind_idx_a.size();
        file_ptr[3] = d_ind_idx_b.size();
      }

      byte* file_ptr = _file_ptr;

      file_ptr += 4 * sizeof(size_t);
      gpu.DtoH((coeff_t*)file_ptr, &d_coeffs[0], d_coeffs_a.size() + d_coeffs_b.size());

      file_ptr += (d_coeffs_a.size() + d_coeffs_b.size()) * sizeof(coeff_t);
      gpu.DtoH(
          (uint32_t*)file_ptr, &d_independent_indices[0], d_ind_idx_a.size() + d_ind_idx_b.size());

      gpu.sync();
      munmap(_file_ptr, file_size);
    } catch (const sppark_error& e) {
      gpu.sync();
      munmap(_file_ptr, file_size);

      throw;
    }
  }
#endif // _WIN32

#if !defined(__HIPCC__) && !defined(_WIN32)
  // The compact cache file (disk_cache.cuh): this header, then (s, value index) per coefficient,
  // the value table, each row's output index c, and the independent indices.
  struct compact_hdr {
    char magic[8];
    uint32_t version, elem_size;
    uint64_t key, ca, cb, ua, ub, ntab, sum;
  };
  static constexpr char kCompactMagic[8] = {'R', '0', 'G', '1', '6', 'C', 'P', 'T'};
  void compact_upload(stream_t& ld, void* dst, const void* src, size_t bytes) {
    if (g16_staged()) {
      ld.sync();
      key_stager::get().upload(dst, src, bytes);
    } else {
      CUDA_OK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, ld));
      ld.sync();
    }
  }
  // Loads the compact form (building and caching it from the mmap'd original `orig`, whose
  // file is described by `st`, if needed). false: the original isn't in the expected layout.
  bool compact_load(stream_t& ld, const byte* orig, size_t orig_size, const struct stat& st) {
    if (orig_size < 4096) return false;
    const size_t* h = (const size_t*)orig;
    size_t ca = h[0], cb = h[1], ua = h[2], ub = h[3];
    if (ca > orig_size / sizeof(coeff_t) || cb > orig_size / sizeof(coeff_t) || ua > orig_size / 4 ||
        ub > orig_size / 4)
      return false;
    size_t n = ca + cb;
    if (orig_size != 32 + n * sizeof(coeff_t) + (ua + ub) * 4) return false;
    // Key: the cache format, sampled bytes of the original, and its size, mtime and inode.
    uint64_t key = g16_cache::kFnvBasis;
    uint32_t format_id[3] = {g16_cache::kFormatVersion, (uint32_t)sizeof(fr_t), (uint32_t)sizeof(coeff_t)};
    key = g16_cache::fnv1a(key, format_id, sizeof format_id);
    for (size_t k = 0; k < 64; k++)
      key = g16_cache::fnv1a(key, orig + (orig_size - 4096) / 63 * k, 4096);
    uint64_t fid[5] = {(uint64_t)orig_size, (uint64_t)st.st_mtim.tv_sec,
                       (uint64_t)st.st_mtim.tv_nsec, (uint64_t)st.st_ino, (uint64_t)st.st_dev};
    key = g16_cache::fnv1a(key, fid, sizeof fid);
    std::string path = g16_cache::path("coeffs_compact_", key);
    if (path.empty()) return false;
    const coeff_t* co = (const coeff_t*)(orig + 32);
    const uint32_t* ind = (const uint32_t*)(orig + 32 + n * sizeof(coeff_t));

    const byte* cp = nullptr;
    size_t csz = 0;
    int fd = open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
      struct stat cst;
      if (fstat(fd, &cst) == 0 && cst.st_size >= (off_t)sizeof(compact_hdr)) {
        csz = cst.st_size;
        cp = (const byte*)mmap(NULL, csz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (cp == MAP_FAILED) cp = nullptr;
      }
      close(fd);
    }
    const compact_hdr* ch = (const compact_hdr*)cp;
    bool ok = cp && !memcmp(ch->magic, kCompactMagic, 8) &&
              ch->version == g16_cache::kFormatVersion && ch->elem_size == sizeof(fr_t) &&
              ch->key == key && ch->ca == ca && ch->cb == cb && ch->ua == ua && ch->ub == ub &&
              ch->ntab <= n &&
              csz == sizeof(compact_hdr) + n * 8 + ch->ntab * sizeof(fr_t) + (ua + ub) * 8 &&
              ch->sum == g16_cache::payload_hash(cp + sizeof(compact_hdr), csz - sizeof(compact_hdr));
    std::vector<byte> built;
    if (!ok) {
      if (g16_verbose())
        fprintf(stderr, "groth16: %s compact coefficients cache %s\n", cp ? "rebuilding invalid" : "building",
                path.c_str());
      if (cp) munmap(const_cast<byte*>(cp), csz);
      cp = nullptr;
      // Build: value table (first-seen order), (s, vidx), row c, indices.
      struct H { size_t operator()(const std::array<uint64_t, 4>& k) const {
        return k[0] ^ (k[1] * 0x9e3779b97f4a7c15ull) ^ (k[2] << 7) ^ (k[3] >> 3); } };
      std::unordered_map<std::array<uint64_t, 4>, uint32_t, H> vm;
      std::vector<fr_t> table;
      std::vector<uint2> cs(n);
      for (size_t i = 0; i < n; i++) {
        std::array<uint64_t, 4> k;
        memcpy(k.data(), &co[i].value, 32);
        auto it = vm.find(k);
        uint32_t vi;
        if (it == vm.end()) { vi = table.size(); vm.emplace(k, vi); table.push_back(co[i].value); }
        else vi = it->second;
        cs[i] = make_uint2(co[i].s, vi);
      }
      std::vector<uint32_t> rowc(ua + ub, 0);
      for (size_t r = 0; r + 1 < ua; r++) rowc[r] = co[ind[r]].c;
      for (size_t r = 0; r + 1 < ub; r++) rowc[ua + r] = co[ca + ind[ua + r]].c;
      compact_hdr hd = {};
      memcpy(hd.magic, kCompactMagic, 8);
      hd.version = g16_cache::kFormatVersion; hd.elem_size = sizeof(fr_t);
      hd.key = key; hd.ca = ca; hd.cb = cb; hd.ua = ua; hd.ub = ub; hd.ntab = table.size();
      csz = sizeof hd + n * 8 + table.size() * sizeof(fr_t) + (ua + ub) * 8;
      built.resize(csz);
      byte* w = built.data() + sizeof hd;
      memcpy(w, cs.data(), n * 8); w += n * 8;
      memcpy(w, table.data(), table.size() * sizeof(fr_t)); w += table.size() * sizeof(fr_t);
      memcpy(w, rowc.data(), (ua + ub) * 4); w += (ua + ub) * 4;
      memcpy(w, ind, (ua + ub) * 4);
      hd.sum = g16_cache::payload_hash(built.data() + sizeof hd, csz - sizeof hd);
      memcpy(built.data(), &hd, sizeof hd);
      std::string tmp = g16_cache::tmp_path(path);
      if (FILE* f = fopen(tmp.c_str(), "wb")) {
        bool good = fwrite(built.data(), 1, csz, f) == csz;
        good = (fclose(f) == 0) && good;
        if (good) rename(tmp.c_str(), path.c_str()); else remove(tmp.c_str());
      }
    }
    const byte* base = cp ? cp : built.data();
    ch = (const compact_hdr*)base;
    const byte* p = base + sizeof(compact_hdr);
    d_cs = gpu_ptr_t<uint2>((uint2*)ld.Dmalloc(n * 8));
    d_table = gpu_ptr_t<fr_t>((fr_t*)ld.Dmalloc(ch->ntab * sizeof(fr_t)));
    d_rowc = gpu_ptr_t<uint32_t>((uint32_t*)ld.Dmalloc((ua + ub) * 4));
    d_independent_indices = gpu_ptr_t<uint32_t>((uint32_t*)ld.Dmalloc((ua + ub) * 4));
    compact_upload(ld, &d_cs[0], p, n * 8); p += n * 8;
    compact_upload(ld, &d_table[0], p, ch->ntab * sizeof(fr_t)); p += ch->ntab * sizeof(fr_t);
    compact_upload(ld, &d_rowc[0], p, (ua + ub) * 4); p += (ua + ub) * 4;
    compact_upload(ld, &d_independent_indices[0], p, (ua + ub) * 4);
    ld.sync();
    if (cp) munmap(const_cast<byte*>(cp), csz);
    d_ind_idx_a = slice_t<uint32_t>(&d_independent_indices[0], ua);
    d_ind_idx_b = slice_t<uint32_t>(&d_independent_indices[ua], ub);
    // Empty coefficient slices with the right sizes (the compact kernel indexes d_cs instead).
    d_coeffs_a = slice_t<coeff_t>((const coeff_t*)nullptr, ca);
    d_coeffs_b = slice_t<coeff_t>((const coeff_t*)nullptr, cb);
    compact = true;
    return true;
  }
#endif
#ifndef __HIPCC__
  const uint2* get_cs_a() const { return &d_cs[0]; }
  const uint2* get_cs_b() const { return &d_cs[d_coeffs_a.size()]; }
  const uint32_t* get_rowc_a() const { return &d_rowc[0]; }
  const uint32_t* get_rowc_b() const { return &d_rowc[d_ind_idx_a.size()]; }
#endif

  const slice_t<coeff_t> get_coeffs_a() const { return d_coeffs_a; }

  const slice_t<coeff_t> get_coeffs_b() const { return d_coeffs_b; }

  const slice_t<uint32_t> get_indices_a() const { return d_ind_idx_a; }

  const slice_t<uint32_t> get_indices_b() const { return d_ind_idx_b; }
};
#endif // !__HIP_DEVICE_COMPILE__
