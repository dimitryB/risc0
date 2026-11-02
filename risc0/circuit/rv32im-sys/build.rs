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

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use risc0_build_kernel::{KernelBuild, KernelType};

/// Build-cache stamp utilities for the Intel SYCL kernels.
///
/// Replaces the previous `"built"` literal stamp (which never invalidated on
/// source changes) with content-addressed SHA-256 hashing. Three independent
/// stamps cover the three icpx compilations: monolithic eval_check, two-way
/// multipass eval_check, and witgen. Each stamp is a small TOML record:
///
///     hash         = "<sha256 hex>"
///     built_at     = "<utc rfc3339 timestamp>"
///     icpx_version = "<first line of `icpx --version`>"
///     status       = "ok" | "failed"
///
/// Cache hit requires all of: artifact present, stamp parses, hash matches,
/// status is "ok". Anything else triggers a rebuild.
mod stamp {
    use sha2::{Digest, Sha256};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn update_len(h: &mut Sha256, n: u64) {
        h.update(n.to_le_bytes());
    }

    /// Hash a list of files (path + content) and extra string ingredients.
    /// Each file's path AND content go into the digest. Convenience wrapper:
    /// label == content_path, suitable for in-tree relative paths.
    #[allow(dead_code)] // kept as a convenience entry-point for callers that
                        // don't need label/content separation
    pub fn hash(files: &[&Path], extras: &[&str]) -> String {
        let pairs: Vec<(PathBuf, PathBuf)> = files
            .iter()
            .map(|p| (p.to_path_buf(), p.to_path_buf()))
            .collect();
        hash_labeled(&pairs, extras)
    }

    /// Hash a list of (label, content_path) pairs and extra string
    /// ingredients. The `label` participates in the digest as the file's
    /// identity; `content_path` is where bytes are read from. This lets
    /// callers strip absolute prefixes (e.g. `cxx_root` from external
    /// dependency headers) so the hash stays portable across machines with
    /// different workspace roots while still invalidating on content drift.
    ///
    /// Sorting is by label, deterministic byte-lex order. Each variable-
    /// length field is length-prefixed (8-byte LE) so distinct input sets
    /// cannot frame-collide. Present/missing files use distinct
    /// discriminators (`OK` / `MISS`) so a real file whose content happens
    /// to equal a sentinel cannot impersonate a missing file.
    pub fn hash_labeled(items: &[(PathBuf, PathBuf)], extras: &[&str]) -> String {
        let mut sorted: Vec<&(PathBuf, PathBuf)> = items.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let mut h = Sha256::new();
        h.update(b"STAMP-V2\0");
        for (label, content_path) in sorted {
            // Use OS-encoded bytes (lossless on Linux), not `to_string_lossy`,
            // so non-UTF8 labels still hash deterministically.
            let path_bytes = label.as_os_str().as_encoded_bytes();
            h.update(b"FILE");
            update_len(&mut h, path_bytes.len() as u64);
            h.update(path_bytes);
            match std::fs::read(content_path) {
                Ok(b) => {
                    h.update(b"OK");
                    update_len(&mut h, b.len() as u64);
                    h.update(&b);
                }
                Err(_) => {
                    h.update(b"MISS");
                    update_len(&mut h, 0);
                }
            }
        }
        for s in extras {
            let bytes = s.as_bytes();
            h.update(b"EXTRA");
            update_len(&mut h, bytes.len() as u64);
            h.update(bytes);
        }
        format!("{:x}", h.finalize())
    }

    /// First line of `icpx --version`, trimmed and stripped of control chars.
    /// On error, returns a discriminating string so distinct broken
    /// environments don't coalesce onto the same cache key:
    /// - `unknown-exec:<errno>` when the binary fails to spawn
    ///   (ENOENT / EACCES / ENOMEM / E2BIG / ETXTBSY etc.)
    /// - `unknown-exit:<code>` when icpx runs but exits non-zero
    /// - `unknown` for the (unreachable) zero-exit + empty-stdout case
    pub fn icpx_version(icpx: &Path) -> String {
        let out = match Command::new(icpx).arg("--version").output() {
            Ok(o) => o,
            Err(e) => return format!("unknown-exec:{}", e.raw_os_error().unwrap_or(-1)),
        };
        let stdout = String::from_utf8_lossy(&out.stdout);
        let first = stdout.lines().next().map(|s| s.trim()).unwrap_or("");
        let sanitized: String = first.chars().filter(|c| !c.is_control()).collect();
        if !out.status.success() {
            return format!("unknown-exit:{}", out.status.code().unwrap_or(-1));
        }
        if sanitized.is_empty() {
            return "unknown".into();
        }
        sanitized
    }

    /// Current UTC timestamp as RFC3339 via `date -u`. "unknown" on error.
    pub fn now() -> String {
        Command::new("date")
            .arg("-u")
            .arg("+%Y-%m-%dT%H:%M:%SZ")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".into())
    }

    /// Parsed stamp record. Only fields needed for cache decisions are exposed.
    pub struct Stamp {
        pub hash: String,
        pub status: String,
    }

    pub fn read(path: &Path) -> Option<Stamp> {
        let s = std::fs::read_to_string(path).ok()?;
        let mut hash = None;
        let mut status = None;
        for line in s.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                let k = k.trim();
                let v = v.trim().trim_matches('"');
                match k {
                    "hash" => hash = Some(v.to_string()),
                    "status" => status = Some(v.to_string()),
                    _ => {}
                }
            }
        }
        Some(Stamp {
            hash: hash?,
            status: status?,
        })
    }

    pub fn write(path: &Path, hash_value: &str, icpx_version: &str, status: &str) {
        let now = now();
        // Strip control characters first (so an icpx_version with embedded
        // newlines/tabs cannot break TOML line-orientation), then escape the
        // two TOML basic-string metacharacters in the order \\ → \\\\, " → \"
        // (backslash MUST come first or the second pass would re-escape).
        let escape = |s: &str| -> String {
            let cleaned: String = s.chars().filter(|c| !c.is_control()).collect();
            cleaned.replace('\\', "\\\\").replace('"', "\\\"")
        };
        let body = format!(
            "hash = \"{}\"\nbuilt_at = \"{}\"\nicpx_version = \"{}\"\nstatus = \"{}\"\n",
            escape(hash_value),
            escape(&now),
            escape(icpx_version),
            escape(status),
        );
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // Write to a per-process tempfile + rename so concurrent writers
        // can't clobber each other's tmp before the rename completes. The
        // suffix includes pid + nanos so two parallel `cargo build`s
        // sharing the same cache_dir each get a unique tempfile path.
        // `rename` is atomic on POSIX same-fs.
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = {
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            let name = path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "stamp".into());
            parent.join(format!("{name}.{pid}.{nanos}.tmp"))
        };
        std::fs::write(&tmp, body).expect("failed to write build-cache stamp tmp");
        std::fs::rename(&tmp, path).expect("failed to rename build-cache stamp");
    }

    /// Cache hit only when artifact exists AND stamp parses AND hash matches
    /// AND status == "ok".
    pub fn need_rebuild(artifact: &Path, stamp_path: &Path, expected_hash: &str) -> bool {
        need_rebuild_multi(&[artifact], stamp_path, expected_hash)
    }

    /// Multi-artifact variant: cache hit requires ALL artifacts present plus
    /// the same stamp invariants. Used by the multipass branch which must
    /// verify both pass1.so and pass2.so.
    pub fn need_rebuild_multi(artifacts: &[&Path], stamp_path: &Path, expected_hash: &str) -> bool {
        if artifacts.iter().any(|p| !p.exists()) {
            return true;
        }
        match read(stamp_path) {
            None => true,
            Some(s) => s.hash != expected_hash || s.status != "ok",
        }
    }
}

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("Features 'cuda' and 'rocm' are mutually exclusive. Enable only one GPU backend.");

fn main() {
    let cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let rocm = env::var("CARGO_FEATURE_ROCM").is_ok();

    // Set the GPU builds' environment before any thread starts: the CPU build below runs
    // concurrently and reads the environment.
    if cuda || rocm {
        env::set_var("SCCACHE_IDLE_TIMEOUT", "0");
    }
    if rocm {
        env::set_var("HIP_PLATFORM", "amd");
    }

    // The CPU kernels (rust_poly_fp_<n>.cpp and steps.cpp, 2-4 MB of generated C++ each) and
    // the GPU kernels share nothing, so build them concurrently instead of one after the other.
    std::thread::scope(|scope| {
        scope.spawn(build_cpu_kernels);

        if cuda {
            build_cuda_kernels();
        }

        if rocm {
            build_rocm_kernels();
        }

        if env::var("CARGO_FEATURE_INTEL").is_ok() {
            build_intel_kernels();
        }
    });
}

fn build_cpu_kernels() {
    rerun_if_changed("kernels/cxx");
    // The generated constraint polynomial (rust_poly_fp_<n>.cpp) and steps.cpp take far longer
    // to compile than the rest: start them at once, each on its own thread.
    let (big, rest): (Vec<PathBuf>, Vec<PathBuf>) =
        glob_paths("kernels/cxx/*.cpp").into_iter().partition(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("rust_poly_fp_") || n == "steps.cpp")
        });
    KernelBuild::new(KernelType::Cpp)
        .files(rest)
        .files_on_own_threads(big)
        .deps(glob_paths("kernels/cxx/*.h"))
        .deps(glob_paths("kernels/cxx/*.cpp.inc"))
        .deps(glob_paths("kernels/cxx/*.h.inc"))
        .include(env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap())
        .compile("risc0_rv32im_cpu");
}

fn build_cuda_kernels() {
    let output = "risc0_rv32im_cuda";

    println!("cargo:rerun-if-env-changed=NVCC_APPEND_FLAGS");
    println!("cargo:rerun-if-env-changed=NVCC_PREPEND_FLAGS");
    println!("cargo:rerun-if-env-changed=SCCACHE_RECACHE");
    rerun_if_changed("kernels/cuda");

    if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
        let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
        let out_path = out_dir.join(format!("lib{output}-skip.a"));
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&out_path)
            .unwrap();
        println!("cargo:{}={}", output, out_path.display());
        return;
    }

    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    rerun_if_include_roots_changed(&[&cuda_root, &cxx_root, &sppark_root]);
    let use_native_arch =
        env::var_os("NVCC_PREPEND_FLAGS").is_none() && env::var_os("NVCC_APPEND_FLAGS").is_none();

    // Step 1: Compile eval_check_ef.cu WITHOUT -dc (standalone mode). It is the
    // generated evalfast kernel (kernels/cuda/evalfast/regen.py); ROCm builds
    // eval_check_ef_hip.cu, its ROCm output. Standalone,
    // NVCC sees the whole poly_fp call chain in one translation unit; with -dc it would
    // compile relocatable code.
    //
    // CACHING: eval_check compilation takes a few minutes. We cache it under a key over
    // everything it can read (see `eval_check_ef_sources_key`) and the compiler, and skip
    // recompilation when only other .cu files changed.
    // Use a stable cache dir (target/release/) instead of OUT_DIR which changes
    // with each cargo build hash.
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let cache_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with("release") || p.ends_with("debug"))
        .map(|p| p.join("eval_check_cache"))
        .unwrap_or_else(|| out_dir.join("eval_check_cache"));
    std::fs::create_dir_all(&cache_dir).unwrap();
    let msvc = risc0_build_kernel::is_msvc_target();
    let obj_ext = risc0_build_kernel::object_extension();
    let eval_check_obj = out_dir.join(format!("eval_check_ef_standalone.{obj_ext}"));
    remove_legacy_cache_files(
        &cache_dir,
        &[
            &format!("eval_check_combined_standalone.{obj_ext}"),
            "eval_check_hash.stamp",
        ],
    );

    let mut cmd = risc0_build_kernel::nvcc_command();
    cmd.current_dir("kernels/cuda");
    if !msvc {
        cmd.arg(risc0_build_kernel::nvcc_ccbin_arg())
            .arg("-Xcompiler")
            .arg("-O3,-ffunction-sections,-fdata-sections,-fPIC")
            .arg("-Xcompiler")
            .arg("-Wno-unused-function,-Wno-unused-parameter");
    }
    cmd.arg(risc0_build_kernel::cuda_cxx_std_flag())
        .arg("-m64")
        .arg("-Xptxas")
        .arg("-O3")
        .arg("-diag-suppress=177")
        .arg("-diag-suppress=550")
        .arg("-diag-suppress=2922")
        .arg("--expt-relaxed-constexpr")
        .arg("-I")
        .arg(&cuda_root)
        .arg("-I")
        .arg(&cxx_root)
        .arg("-I")
        .arg(&sppark_root);
    if use_native_arch {
        cmd.arg("-arch=native");
    }
    cmd.arg("-c") // compile only, NO --device-c
        .arg("eval_check_ef.cu");

    // Key the cache on the sources, the whole command (but its output path), and what nvcc
    // reads from outside it (the NVCC_*_FLAGS that carry the GPU architecture list, the host
    // compiler, include-path variables, the nvcc version) -- and for -arch=native, this
    // machine's GPUs -- so a changed source, header, architecture list or toolchain doesn't
    // reuse an object built without it.
    let native_key = if use_native_arch {
        risc0_build_kernel::native_gpu_arch_key(false)
    } else {
        String::new()
    };
    let key = risc0_build_kernel::hash_key(&[
        "rv32im-cuda-eval-check-ef-v1",
        &eval_check_ef_sources_key(&cuda_root, &cxx_root, &sppark_root),
        &risc0_build_kernel::command_key(&cmd),
        &risc0_build_kernel::nvcc_environment_key(),
        &native_key,
    ]);
    let mut standalone = vec![("eval_check_ef_standalone", key, cmd, eval_check_obj.clone())];

    // Step 1b: accum_tu.cu and witgen_tu.cu (generated by kernels/cuda/accum_tu_gen.py and
    // witgen_tu_gen.py): the stepAccum / par_stepExec call graphs as whole-program TUs,
    // compiled WITHOUT -dc like eval_check_ef.cu and cached the same way.
    let accum_tu_obj = out_dir.join(format!("accum_tu_standalone.{obj_ext}"));
    let witgen_tu_obj = out_dir.join(format!("witgen_tu_standalone.{obj_ext}"));
    for (stem, tu_src, tu_obj) in [
        ("accum_tu_standalone", "accum_tu.cu", &accum_tu_obj),
        ("witgen_tu_standalone", "witgen_tu.cu", &witgen_tu_obj),
    ] {
        let mut cmd = risc0_build_kernel::nvcc_command();
        cmd.current_dir("kernels/cuda");
        if !msvc {
            cmd.arg(risc0_build_kernel::nvcc_ccbin_arg())
                .arg("-Xcompiler")
                .arg("-O3,-ffunction-sections,-fdata-sections,-fPIC")
                .arg("-Xcompiler")
                .arg("-Wno-unused-function,-Wno-unused-parameter");
        }
        cmd.arg(risc0_build_kernel::cuda_cxx_std_flag())
            .arg("-m64")
            .arg("-Xptxas")
            .arg("-O3")
            .arg("-diag-suppress=177")
            .arg("-diag-suppress=550")
            .arg("-diag-suppress=2922")
            .arg("-I")
            .arg(&cuda_root)
            .arg("-I")
            .arg(&cxx_root)
            .arg("-I")
            .arg(&sppark_root);
        if use_native_arch {
            cmd.arg("-arch=native");
        }
        cmd.arg("-c").arg(tu_src);
        let key = risc0_build_kernel::hash_key(&[
            "rv32im-cuda-standalone-tu-v1",
            &standalone_tu_sources_key(tu_src, &cuda_root, &cxx_root, &sppark_root),
            &risc0_build_kernel::command_key(&cmd),
            &risc0_build_kernel::nvcc_environment_key(),
            &native_key,
        ]);
        standalone.push((stem, key, cmd, tu_obj.clone()));
    }

    // The standalone compiles (each a single long ptxas run) go on their own threads, so they
    // overlap each other and the -dc compile of the remaining files below instead of running
    // one after another before it.
    std::thread::scope(|scope| {
        for (stem, key, mut cmd, obj) in standalone {
            let cache_dir = &cache_dir;
            scope.spawn(move || {
                risc0_build_kernel::cached_compile(cache_dir, stem, obj_ext, &key, &obj, |out| {
                    let status = cmd
                        .arg("-o")
                        .arg(out)
                        .status()
                        .unwrap_or_else(|e| panic!("failed to run nvcc for {stem}: {e}"));
                    assert!(status.success(), "nvcc failed for {stem} (standalone mode)");
                });
            });
        }

        // Step 2: Compile remaining .cu files with -dc (separate compilation) via cc crate.
        // Exclude eval_check files (compiled standalone above).
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .cudart("static")
            .debug(false)
            .flag("-diag-suppress=177")
            .flag("-diag-suppress=550")
            .flag("-diag-suppress=2922")
            .flag(risc0_build_kernel::cuda_cxx_std_flag())
            .flag("-Xptxas")
            .flag("-O3")
            .include(&cuda_root)
            .include(&cxx_root)
            .include(&sppark_root);
        // GCC/Clang host flags; on MSVC `cc` already passes cl.exe its optimization level.
        if msvc {
            risc0_build_kernel::msvc_cuda_build_setup(&mut build);
        } else {
            build
                .flag("-Xcompiler")
                .flag("-Wno-unused-function,-Wno-unused-parameter")
                .flag("-Xcompiler")
                .flag("-O3");
        }
        if use_native_arch {
            build.flag("-arch=native");
        }
        let cuda_files: Vec<PathBuf> = glob_paths("kernels/cuda/*.cu")
            .into_iter()
            .filter(|p| {
                let name = p.file_name().unwrap().to_str().unwrap();
                !matches!(
                    name,
                    "eval_check_0.cu"
                        | "eval_check_1.cu"
                        | "eval_check_2.cu"
                        | "eval_check_3.cu"
                        | "eval_check_combined.cu" // ROCm only
                        | "eval_check_ef.cu"
                        | "eval_check_ef_hip.cu" // ROCm only
                        | "accum_tu.cu" // standalone (Step 1b)
                        | "witgen_tu.cu" // standalone (Step 1b)
                        | "eval_check_kernel.cu"  // RDC-only, not used by NVIDIA
                        | "witgen_combined.cu"
                )
            })
            .collect();
        build.files(cuda_files).compile(output);
    });

    // Step 3: Add standalone objects to the archive.
    risc0_build_kernel::add_to_static_lib(
        &out_dir,
        output,
        &[eval_check_obj, accum_tu_obj, witgen_tu_obj],
    );
}

fn build_rocm_kernels() {
    let output = "risc0_rv32im_cuda";

    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=RISC0_HIP_ARCH");
    println!("cargo:rerun-if-env-changed=SCCACHE_RECACHE");
    // Tier-4 IGC experiment + T3.2 tree-reduce env vars must invalidate
    // build.rs cache; otherwise changing them silently no-ops because cargo
    // doesn't track unregistered env vars.
    println!("cargo:rerun-if-env-changed=RISC0_IGC_EXTRA_OPTS");
    println!("cargo:rerun-if-env-changed=RISC0_TREE_REDUCE");
    println!("cargo:rerun-if-env-changed=RISC0_POLY_FP_CSE");
    println!("cargo:rerun-if-env-changed=RISC0_POLY_FP_CSE_TOP");
    println!("cargo:rerun-if-env-changed=RISC0_POLY_FP_CSE_MIN_SHARE");
    println!("cargo:rerun-if-env-changed=RISC0_LSC_HINTS");
    rerun_if_changed("kernels/cuda");

    if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
        let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
        let out_path = out_dir.join(format!("lib{output}-skip.a"));
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&out_path)
            .unwrap();
        println!("cargo:{}={}", output, out_path.display());
        return;
    }

    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    rerun_if_include_roots_changed(&[&cuda_root, &cxx_root, &sppark_root]);
    let hipcc = risc0_build_kernel::find_hipcc();

    // Step 1: Compile eval_check standalone with hipcc: the evalfast kernel
    // (eval_check_ef_hip.cu, see its header) unless RISC0_HIP_EVAL_CHECK=combined selects the original
    // eval_check_combined.cu. A po2 20 eval_check takes ~235 ms on an RX 9070 XT (gfx1201; its poly_mix
    // chains on the matrix units, ~257 ms without) against the original's 845 ms, and ~214 ms on an
    // RX 7900 XTX (gfx1100).
    // For the original, -mllvm -amdgpu-early-inline-all=false prevents OOM on this large kernel
    // (21 functions, 2154 column reads).
    //
    // CACHING: eval_check compilation is very slow. We hash all eval_check
    // source files and skip recompilation when only other .cu files changed.
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let kernel_dir = std::fs::canonicalize("kernels/cuda").unwrap();
    let cache_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with("release") || p.ends_with("debug"))
        .map(|p| p.join("eval_check_cache_rocm"))
        .unwrap_or_else(|| out_dir.join("eval_check_cache_rocm"));
    std::fs::create_dir_all(&cache_dir).unwrap();
    remove_legacy_cache_files(
        &cache_dir,
        &[
            "eval_check_combined_standalone.o",
            "eval_check_hash_rocm.stamp",
            "remaining_kernels_rocm.o",
            "remaining_kernels_hash_rocm.stamp",
            "ffi_supra.o",
            "ffi_supra_hash_rocm.stamp",
        ],
    );
    let obj_ext = risc0_build_kernel::object_extension();
    let eval_check_obj = out_dir.join(format!("eval_check_combined_standalone.{obj_ext}"));
    // Every cache below is keyed on its sources, the hipcc command and flags, what hipcc
    // reads from outside its arguments (its version and environment), and the architectures
    // -- for `native`, this machine's GPUs -- so none can reuse an object built without them.
    let hip_arches = risc0_build_kernel::hip_arches();
    let hipcc_key = format!(
        "{hipcc}\n{}\narches={hip_arches}\n{}",
        risc0_build_kernel::hipcc_environment_key(&hipcc),
        if hip_arches.split(',').any(|a| a.trim() == "native") {
            risc0_build_kernel::native_gpu_arch_key(true)
        } else {
            String::new()
        }
    );
    let include_cuda2hip = format!("{}/util/cuda2hip.hpp", &sppark_root);
    println!("cargo:rerun-if-env-changed=RISC0_HIP_EVAL_CHECK");
    let evalfast = env::var("RISC0_HIP_EVAL_CHECK").as_deref() != Ok("combined");
    let mut eval_check_flags: Vec<&str> = vec![
        "-x",
        "hip",
        "-std=c++17",
        "-O3",
        "-Wno-unused-function",
        "-Wno-unused-parameter",
    ];
    if evalfast {
        // 64-wide waves and an occupancy target of 4 blocks (EF_MINB, the launch bounds' minimum):
        // the fastest at po2 20 on both an RX 7900 XTX and an RX 9070 XT (3 and 5 within 2%,
        // 32-wide waves 5-20% slower). On the previous kernel, ROCm 7.2 miscompiled EF_MINB 8:
        // nearly every call after the first few returned wrong values (5, 6 and 10 were correct
        // in 300 calls each). gfx11 runs 3 (EF_MINB_GFX11: 252 VGPRs, less scratch spilling) since
        // the inlined kernel with pmlag_hip.py: ~115.7 instead of ~123.6 ms on an RX 7900 XTX (ef
        // harness, po2 20, bit-exact); gfx12 keeps 4 (3 is ~13 ms slower on an RX 9070 XT).
        eval_check_flags.extend(["-mwavefrontsize64", "-DEF_MINB=4", "-DEF_MINB_GFX11=3"]);
    } else {
        eval_check_flags.extend([
            "-mllvm",
            "-amdgpu-early-inline-all=false",
            "-mllvm",
            "-amdgpu-use-aa-in-codegen",
            "-mllvm",
            "-amdgpu-schedule-metric-bias=0",
            "-mllvm",
            "-amdgpu-internalize-symbols",
            "-mllvm",
            "-amdgpu-schedule-relaxed-occupancy",
        ]);
    }
    eval_check_flags.extend([
        "-include",
        &include_cuda2hip,
        "-I",
        &cuda_root,
        "-I",
        &cxx_root,
        "-I",
        &sppark_root,
    ]);
    eval_check_flags.extend(risc0_build_kernel::hip_host_flags());
    let (eval_check_src, eval_check_sources) = if evalfast {
        (
            "eval_check_ef_hip.cu",
            eval_check_ef_hip_sources_key(&cuda_root, &cxx_root, &sppark_root),
        )
    } else {
        (
            "eval_check_combined.cu",
            eval_check_sources_key(&cuda_root, &cxx_root, &sppark_root),
        )
    };
    let eval_check_key = risc0_build_kernel::hash_key(&[
        "rv32im-rocm-eval-check-v1",
        eval_check_src,
        &eval_check_sources,
        &format!("{eval_check_flags:?}"),
        &hipcc_key,
    ]);
    // The compiles below run concurrently (each is one hipcc process; eval_check is the long
    // pole), see the thread scope after Step 2.
    let compile_eval_check = || {
        risc0_build_kernel::cached_compile(
            &cache_dir,
            "eval_check_combined_standalone",
            obj_ext,
            &eval_check_key,
            &eval_check_obj,
            |out| {
                risc0_build_kernel::hip_compile(
                    &hipcc,
                    &eval_check_flags,
                    Path::new(eval_check_src),
                    out,
                    &hip_arches,
                    Some(Path::new("kernels/cuda")),
                )
            },
        )
    };

    // Step 2: Single-TU amalgamation for remaining .cu files.
    // This avoids -fgpu-rdc and the problematic HIP device link step entirely.
    // Exclude eval_check files (compiled standalone above), witgen_combined.cu,
    // and ffi_supra.cu (uses sppark types that conflict with risc0's fpext.h).
    let cuda_files: Vec<PathBuf> = glob_paths("kernels/cuda/*.cu")
        .into_iter()
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap();
            !matches!(
                name,
                "eval_check_0.cu"
                    | "eval_check_1.cu"
                    | "eval_check_2.cu"
                    | "eval_check_3.cu"
                    | "eval_check_combined.cu"
                    | "eval_check_ef.cu" // CUDA only
                    | "eval_check_ef_hip.cu"
                    | "eval_check_kernel.cu"
                    | "witgen_combined.cu"
                    | "accum_tu.cu"
                    | "witgen_tu.cu"
                    | "ffi_supra.cu"
            )
        })
        .collect();
    // accum_tu.cu: stepAccum's call graph force-inlined in its own TU, as on CUDA (Step 1b).
    // witgen_tu.cu: par_stepExec's call graph in its own TU with global buffer accesses (ffi.cu,
    // hipWitgenTu): witgen_tu2 (the default) force-inlined with launch bounds (256, 1), and
    // witgen_tu (RISC0_HIP_WITGEN_TU=1) inlined at the compiler's discretion with (256, 8) (the
    // same 192-VGPR cap as par_stepExec).
    let separate_files: Vec<(PathBuf, String, Vec<&str>)> = [
        ("kernels/cuda/ffi_supra.cu", "ffi_supra", vec![]),
        ("kernels/cuda/accum_tu.cu", "accum_tu", vec![]),
        (
            "kernels/cuda/witgen_tu.cu",
            "witgen_tu",
            vec![
                "-DRISC0_HIP_GLOBAL_BUFS",
                "-DWITGEN_TU_LB=256,8",
                "-DWITGEN_TU_FNATTR=",
            ],
        ),
        (
            "kernels/cuda/witgen_tu.cu",
            "witgen_tu2",
            vec![
                "-DRISC0_HIP_GLOBAL_BUFS",
                "-DWITGEN_TU_LB=256,1",
                "-DWITGEN_TU_FNATTR=__forceinline__",
                "-DWITGEN_TU_KERNEL=par_stepExecW2",
                "-DWITGEN_TU_LAUNCH=risc0_rv32im_launch_par_stepExecTU2",
            ],
        ),
    ]
    .into_iter()
    .flat_map(|(p, stem, extra)| {
        glob_paths(p)
            .into_iter()
            .map(move |f| (f, stem.to_string(), extra.clone()))
    })
    .collect();

    let amalg_path = out_dir.join("remaining_kernels_rocm.cu");
    let mut amalg = String::new();
    for cu in &cuda_files {
        let abs = std::fs::canonicalize(cu).unwrap();
        amalg.push_str(&format!("#include \"{}\"\n", abs.display()));
    }
    std::fs::write(&amalg_path, &amalg).unwrap();

    // Both of the compiles below take every .cu and header under kernels/cuda (the
    // amalgamation by absolute path, and with `-I` to it), so their caches are keyed on all
    // of them.
    let kernel_dir_str = kernel_dir.to_str().unwrap();
    let mut local_flags: Vec<&str> = vec![
        "-x",
        "hip",
        "-std=c++17",
        "-O3",
        "-Wno-unused-function",
        "-Wno-unused-parameter",
        "-mllvm",
        "-amdgpu-early-inline-all=false",
        "-include",
        &include_cuda2hip,
        "-I",
        &cuda_root,
        "-I",
        &cxx_root,
        "-I",
        &sppark_root,
        "-I",
        kernel_dir_str,
    ];
    // witgen.h's non-virtual buffer objects (one indirect call less per witness access) on
    // Linux; Windows builds, and RISC0_HIP_WITGEN_VIRTUAL=1, keep the virtual ones.
    println!("cargo:rerun-if-env-changed=RISC0_HIP_WITGEN_VIRTUAL");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows")
        && env::var("RISC0_HIP_WITGEN_VIRTUAL").as_deref() != Ok("1")
    {
        local_flags.push("-DRISC0_HIP_DIRECT_BUFS");
    }
    local_flags.extend(risc0_build_kernel::hip_host_flags());
    let local_sources_key = risc0_build_kernel::kernel_sources_key(
        &[
            Path::new(&cuda_root),
            Path::new(&cxx_root),
            Path::new(&sppark_root),
            &kernel_dir,
        ],
        &[],
        |_| true,
    );

    // Cache remaining_kernels.
    let remaining_obj = out_dir.join(format!("remaining_kernels_rocm.{obj_ext}"));
    let remaining_key = risc0_build_kernel::hash_key(&[
        "rv32im-rocm-remaining-v1",
        &local_sources_key,
        &amalg,
        &format!("{local_flags:?}"),
        &hipcc_key,
    ]);

    // Compile ffi_supra.cu (uses sppark types, can't be in same TU) and the TUs above separately.
    // Also cached
    let separate_jobs: Vec<(PathBuf, String, PathBuf, Vec<&str>, String)> = separate_files
        .iter()
        .map(|(cu, stem, extra)| {
            let obj = out_dir.join(format!("{stem}.{obj_ext}"));
            let mut flags = local_flags.clone();
            flags.extend(extra.iter().copied());
            let key = risc0_build_kernel::hash_key(&[
                "rv32im-rocm-separate-v1",
                &local_sources_key,
                &cu.to_string_lossy(),
                &format!("{flags:?}"),
                &hipcc_key,
            ]);
            (cu.clone(), stem.clone(), obj, flags, key)
        })
        .collect();
    std::thread::scope(|sc| {
        sc.spawn(compile_eval_check);
        sc.spawn(|| {
            risc0_build_kernel::cached_compile(
                &cache_dir,
                "remaining_kernels_rocm",
                obj_ext,
                &remaining_key,
                &remaining_obj,
                |out| {
                    risc0_build_kernel::hip_compile(
                        &hipcc,
                        &local_flags,
                        &amalg_path,
                        out,
                        &hip_arches,
                        None,
                    )
                },
            )
        });
        for (cu, stem, obj, flags, key) in &separate_jobs {
            let (cache_dir, hipcc, hip_arches) = (&cache_dir, &hipcc, &hip_arches);
            sc.spawn(move || {
                risc0_build_kernel::cached_compile(cache_dir, stem, obj_ext, key, obj, |out| {
                    risc0_build_kernel::hip_compile(hipcc, flags, cu, out, hip_arches, None)
                })
            });
        }
    });
    let mut all_objs = vec![eval_check_obj.clone(), remaining_obj];
    all_objs.extend(separate_jobs.iter().map(|j| j.2.clone()));

    // Step 3: Archive all objects
    let _ = std::fs::remove_file(out_dir.join(format!("lib{output}.a")));
    let _ = std::fs::remove_file(out_dir.join(format!("{output}.lib")));
    risc0_build_kernel::add_to_static_lib(&out_dir, output, &all_objs);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={output}");

    // Link against HIP runtime
    risc0_build_kernel::emit_rocm_lib_link();
}

fn rerun_if_changed<P: AsRef<Path>>(path: P) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}

fn glob_paths(pattern: &str) -> Vec<PathBuf> {
    glob::glob(pattern).unwrap().map(|x| x.unwrap()).collect()
}

/// The files in kernels/cuda that eval_check_combined.cu reads: itself and what it includes
/// from there. The rest of kernels/cuda (witgen, steps, the layouts) is left out, so changing
/// it doesn't trigger this ~24-minute compile; add a file here if eval_check starts
/// including it.
const EVAL_CHECK_LOCAL_SOURCES: &[&str] = &[
    "kernels/cuda/eval_check_combined.cu",
    "kernels/cuda/eval_check_0.cu",
    "kernels/cuda/eval_check_1.cu",
    "kernels/cuda/eval_check_2.cu",
    "kernels/cuda/eval_check_3.cu",
    "kernels/cuda/eval_check.cuh",
];

/// The files in kernels/cuda that the CUDA eval_check_ef.cu reads.
const EVAL_CHECK_EF_LOCAL_SOURCES: &[&str] = &[
    "kernels/cuda/eval_check_ef.cu",
    "kernels/cuda/ef_fp.h",
    "kernels/cuda/ef_fpext.h",
];

/// The files in kernels/cuda that the ROCm eval_check_ef_hip.cu reads.
const EVAL_CHECK_EF_HIP_LOCAL_SOURCES: &[&str] = &[
    "kernels/cuda/eval_check_ef_hip.cu",
    "kernels/cuda/ef_fp.h",
    "kernels/cuda/ef_fpext.h",
];

/// [`eval_check_sources_key`] for the ROCm eval_check_ef_hip.cu.
fn eval_check_ef_hip_sources_key(cuda_root: &str, cxx_root: &str, sppark_root: &str) -> String {
    include_roots_sources_key(
        EVAL_CHECK_EF_HIP_LOCAL_SOURCES,
        cuda_root,
        cxx_root,
        sppark_root,
    )
}

/// [`eval_check_sources_key`] for the CUDA eval_check_ef.cu.
fn eval_check_ef_sources_key(cuda_root: &str, cxx_root: &str, sppark_root: &str) -> String {
    include_roots_sources_key(
        EVAL_CHECK_EF_LOCAL_SOURCES,
        cuda_root,
        cxx_root,
        sppark_root,
    )
}

/// [`eval_check_sources_key`] for a standalone TU in kernels/cuda (accum_tu.cu, witgen_tu.cu):
/// the TU and every header next to it (they include witgen.h and the generated layouts).
fn standalone_tu_sources_key(
    tu_src: &str,
    cuda_root: &str,
    cxx_root: &str,
    sppark_root: &str,
) -> String {
    let tu = format!("kernels/cuda/{tu_src}");
    let mut local: Vec<String> = ["h", "cuh", "inc"]
        .iter()
        .flat_map(|ext| glob_paths(&format!("kernels/cuda/*.{ext}")))
        .map(|p| p.to_str().unwrap().to_owned())
        .collect();
    local.sort();
    local.push(tu);
    let local: Vec<&str> = local.iter().map(String::as_str).collect();
    include_roots_sources_key(&local, cuda_root, cxx_root, sppark_root)
}

/// A key over every file the eval_check compile can read: [`EVAL_CHECK_LOCAL_SOURCES`], and
/// what it can include from the include roots -- all but their `.c`/`.cc`/`.cpp`/`.cu`
/// files, except sppark's `.cu` files, which its headers include (e.g. `ntt/ntt.cuh`
/// includes `kernels.cu`). Leaving out risc0-sys's own translation units means editing one of
/// them (e.g. its ffi.cu) doesn't trigger this ~24-minute compile.
fn eval_check_sources_key(cuda_root: &str, cxx_root: &str, sppark_root: &str) -> String {
    include_roots_sources_key(EVAL_CHECK_LOCAL_SOURCES, cuda_root, cxx_root, sppark_root)
}

fn include_roots_sources_key(
    local: &[&str],
    cuda_root: &str,
    cxx_root: &str,
    sppark_root: &str,
) -> String {
    let files: Vec<&Path> = local.iter().map(Path::new).collect();
    let sppark_root = Path::new(sppark_root);
    risc0_build_kernel::kernel_sources_key(
        &[Path::new(cuda_root), Path::new(cxx_root), sppark_root],
        &files,
        |path| {
            path.starts_with(sppark_root)
                || !matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("c" | "cc" | "cpp" | "cu")
                )
        },
    )
}

/// The include roots both GPU builds compile against: re-run this script when a file in them
/// changes (as a path dependency), so the cache keys are recomputed. File by file, as
/// a directory would make cargo recurse into sppark's `rust/sppark` symlink loop.
fn rerun_if_include_roots_changed(roots: &[&str]) {
    for root in roots {
        for file in risc0_build_kernel::kernel_source_files(Path::new(root)) {
            rerun_if_changed(file);
        }
    }
}

/// Delete what the stamp-file caches left in `cache_dir` before the caches were keyed by
/// file name, best effort, once a week old (by write time: the old code never touched them
/// on reuse): until then a build of an older version sharing this target directory can
/// still use them.
fn remove_legacy_cache_files(cache_dir: &Path, names: &[&str]) {
    let week = std::time::Duration::from_secs(7 * 24 * 60 * 60);
    for name in names {
        let path = cache_dir.join(name);
        let stale = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .is_ok_and(|t| t.elapsed().is_ok_and(|age| age > week));
        if stale {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Stamp version: bump when stamp inputs or layout change so older stamps
/// invalidate cleanly without manual cache wiping. Bumping this string
/// invalidates all three Intel stamps simultaneously.
const INTEL_STAMP_VERSION: &str = "rv32im-v3";

/// Default GRF mode used when `RISC0_INTEL_GRF_MODE` is unset. 256-GRF
/// mode doubles the per-thread register file (8 KB → 16 KB) at the cost
/// of halving occupancy (8 → 4 threads/EU). Phase 0c flips this to 128
/// to test whether the kernel is spill-bound vs occupancy-bound.
const DEFAULT_GRF_MODE: &str = "256";

/// Phase 0c A/B knobs. Read at build-script time; their values participate
/// in every Intel stamp hash so each (GRF, SIMD) combination caches
/// separately under `intel_rv32im_cache_<variant>/`. Setting either to
/// an unrecognized value is rejected by `intel_variant_tag` to avoid
/// silently caching garbage on a typo.
fn intel_grf_mode() -> String {
    let v = std::env::var("RISC0_INTEL_GRF_MODE").unwrap_or_else(|_| DEFAULT_GRF_MODE.into());
    match v.as_str() {
        "128" | "256" => v,
        other => panic!("RISC0_INTEL_GRF_MODE must be 128 or 256, got {:?}", other),
    }
}

/// `RISC0_INTEL_SIMD_WIDTH=16|32` forces the AOT compiler to emit a
/// specific sub-group width via `-cl-intel-force-simd-size`. Default
/// (unset OR empty string) lets the compiler choose. Phase 0c sweeps
/// 16 vs 32 to test whether ILP is the bottleneck.
///
/// Note: ab_experiments.sh sets `RISC0_INTEL_SIMD_WIDTH=""` (empty
/// string) for "auto" cells. Rust's `std::env::var` returns `Ok("")`
/// (not `Err(NotPresent)`) for set-but-empty, so we explicitly treat
/// empty as None to avoid panicking the build for every auto cell.
fn intel_simd_width() -> Option<String> {
    let v = std::env::var("RISC0_INTEL_SIMD_WIDTH").ok()?;
    if v.is_empty() {
        return None;
    }
    match v.as_str() {
        "16" | "32" => Some(v),
        other => panic!(
            "RISC0_INTEL_SIMD_WIDTH must be 16, 32, or unset, got {:?}",
            other
        ),
    }
}

/// Render the Phase 0c variant tag: `default` (no overrides), or a
/// dash-joined name like `grf128` / `grf128_simd16`. Used as the cache
/// subdirectory suffix and as a hash extra.
fn intel_variant_tag() -> String {
    let grf = intel_grf_mode();
    let simd = intel_simd_width();
    let mut parts: Vec<String> = Vec::new();
    if grf != DEFAULT_GRF_MODE {
        parts.push(format!("grf{grf}"));
    }
    if let Some(s) = &simd {
        parts.push(format!("simd{s}"));
    }
    if parts.is_empty() {
        "default".into()
    } else {
        parts.join("_")
    }
}

/// Render the icpx `-Xs -options "..."` argument from the Phase 0c knobs.
/// Concatenates `-cl-intel-{N}-GRF-per-thread` and (when set) the SIMD
/// override.
fn intel_xs_options() -> String {
    let grf = intel_grf_mode();
    let mut parts = vec![format!("-cl-intel-{grf}-GRF-per-thread")];
    if let Some(s) = intel_simd_width() {
        parts.push(format!("-cl-intel-force-simd-size={s}"));
    }
    // Tier-4 IGC experiment hook: RISC0_IGC_EXTRA_OPTS injects extra
    // -options tokens verbatim (e.g. for `-cl-intel-no-subgroup-ifp` or
    // `-igc_opts 'VISAOptions=-presched-rp 200'`). Cache stamp picks this
    // up via the env var read in compute_*_hash, so each setting gets
    // its own .so without forcing a manual clean.
    if let Ok(extra) = std::env::var("RISC0_IGC_EXTRA_OPTS") {
        if !extra.is_empty() {
            parts.push(extra);
        }
    }
    parts.join(" ")
}

/// RISC0_INTEL_EVAL_FAST (default on; "0"/"off" disables): build the mono
/// eval_check with the measured fast path (2319 -> 1510 ms per po2=20 kernel,
/// bit-identical):
/// - `-DRISC0_INTEL_EVAL_FAST`: 32-bit device Fp ops and an inlined
///   lazy-reduction FpExt multiply (risc0/sys/cxx/fp.h, fpext.h), plus the
///   host-supplied vanishing-quotient table (kernels/intel/eval_check.cpp).
/// - kernels/intel/evalfast/pipeline.py: the source pipeline (exact,
///   numerically verified algebraic rewrites; no per-row zero-fill, proven by
///   eval_nozero.py; lazy poly_mix chains; scheduling + scalar FpExt args; the
///   per-row arrays in a liveness-renumbered slot-major global slab via
///   eval_slab.py, which eval_check.cpp allocates and tiles; 32-bit indexing,
///   so po2 <= 22; lazy linear combinations). 2319 -> ~320 ms per po2=20 kernel.
/// - `-DRISC0_INTEL_EVAL_LOHI`: the lo/hi-word FpExt reduction (fpext.h),
///   fastest for this kernel only.
/// - IGC `DisableRecompilation=1`: mandatory with the above. Otherwise IGC's
///   retry heuristic picks a recompiled kernel with ~2.5 MB of private memory
///   per thread that runs ~2x slower than baseline.
/// Only the mono eval_check is affected; multipass, witgen, accum and
/// recursion keep the original arithmetic.
fn intel_eval_fast_enabled() -> bool {
    !matches!(
        std::env::var("RISC0_INTEL_EVAL_FAST").as_deref(),
        Ok("0") | Ok("false") | Ok("off") | Ok("OFF"),
    )
}

/// RISC0_INTEL_EVAL_XMX (default on; "0"/"off" disables): evalfast/xmxlink.py
/// moves the lazy poly_mix linking chains onto the XMX units (u8 DPAS, exact;
/// ~-52 ms per po2=20 segment). Needs the cl_intel_subgroup_matrix_multiply_
/// accumulate builtins, which IGC resolves when the AOT build is finalized.
fn intel_eval_xmx_enabled() -> bool {
    !matches!(
        std::env::var("RISC0_INTEL_EVAL_XMX").as_deref(),
        Ok("0") | Ok("false") | Ok("off") | Ok("OFF"),
    )
}

/// The `-fsycl-targets=` argument for the GPUs without Xe2's DPAS among the build's targets
/// (anything but Battlemage and Lunar Lake, e.g. spir64 or an Alchemist target), or None. The
/// XMX linking chains are written for Xe2's SIMD16 DPAS and gave wrong values on Alchemist (Arc
/// A770, SIMD8 DPAS), so with XMX on, eval_check is also built without it for these targets, as
/// risc0_rv32im_intel_noxmx, which risc0_rv32im_intel loads on such a GPU.
fn intel_non_xmx_targets() -> Option<String> {
    let arg = risc0_build_kernel::intel_sycl_targets_arg();
    let rest: Vec<&str> = arg
        .trim_start_matches("-fsycl-targets=")
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty() && !t.contains("bmg") && !t.contains("lnl"))
        .collect();
    (!rest.is_empty()).then(|| format!("-fsycl-targets={}", rest.join(",")))
}

/// Puts a library built here but not linked (risc0_rv32im_intel_noxmx, loaded at run time) next
/// to the linked ones: on Windows, where intel_link_library copies the DLLs to the profile and
/// deps directories; elsewhere it already sits beside them in the cache directory.
fn intel_place_unlinked_library(dir: &Path, name: &str) {
    if !cfg!(windows) {
        return;
    }
    let lib = risc0_build_kernel::intel_library_path(dir, name);
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    if let Some(profile_dir) = out_dir.ancestors().nth(3) {
        for dest in [profile_dir.to_path_buf(), profile_dir.join("deps")] {
            if dest.is_dir() {
                let target = dest.join(lib.file_name().unwrap());
                if let Err(e) = std::fs::copy(&lib, &target) {
                    println!(
                        "cargo:warning=could not copy {} to {}: {e}",
                        lib.display(),
                        target.display()
                    );
                }
            }
        }
    }
}

/// Cache_dir helper that suffixes the variant name so each (GRF, SIMD)
/// combination has its own .so + stamp set. Switching variants does NOT
/// re-trigger a 30-min icpx compile if that variant is already cached.
fn intel_cache_subdir() -> String {
    format!("intel_rv32im_cache_{}", intel_variant_tag())
}

/// Glob `*.h`/`*.hpp`/`*.cuh` headers from an absolute external include
/// root (typically `DEP_RISC0_SYS_CXX_ROOT`, e.g. `risc0/sys/cxx/`) so
/// their content participates in the input hash.
///
/// Returns (label, content_path) pairs where the label is a portable
/// virtual path of the form `cxx_root/<basename>` — this keeps the hash
/// invariant across machines that resolve the same dependency to
/// different absolute filesystem prefixes (e.g. developer-laptop vs CI
/// runner with workspace at /__w/...). Content is read from the real
/// filesystem path.
fn cxx_root_headers(cxx_root: &str) -> Vec<(PathBuf, PathBuf)> {
    let mut out = Vec::new();
    for ext in &["*.h", "*.hpp", "*.cuh"] {
        let pattern = format!("{cxx_root}/{ext}");
        if let Ok(paths) = glob::glob(&pattern) {
            for p in paths.filter_map(Result::ok) {
                if let Some(name) = p.file_name() {
                    let label = PathBuf::from("cxx_root").join(name);
                    out.push((label, p));
                }
            }
        }
    }
    out
}

/// Build a `(label, content_path)` pair list for in-tree files (label ==
/// content_path) plus cxx_root files (label == `cxx_root/<basename>`).
fn build_pairs(in_tree: Vec<PathBuf>, cxx_root: &str) -> Vec<(PathBuf, PathBuf)> {
    let mut pairs: Vec<(PathBuf, PathBuf)> = in_tree.into_iter().map(|p| (p.clone(), p)).collect();
    pairs.extend(cxx_root_headers(cxx_root));
    pairs
}

/// Inputs for the monolithic eval_check kernel: the four generated
/// `rust_poly_fp_*.cpp` halves of poly_fp + the SYCL kernel wrapper +
/// transitive headers from `cxx_root` (fp.h, fpext.h).
fn compute_mono_hash(cxx_root: &str, icpx_version: &str) -> String {
    let mut in_tree = vec![
        PathBuf::from("kernels/intel/eval_check.cpp"),
        PathBuf::from("kernels/cxx/rust_poly_fp_0.cpp"),
        PathBuf::from("kernels/cxx/rust_poly_fp_1.cpp"),
        PathBuf::from("kernels/cxx/rust_poly_fp_2.cpp"),
        PathBuf::from("kernels/cxx/rust_poly_fp_3.cpp"),
        PathBuf::from("build.rs"),
    ];
    // When RISC0_TREE_REDUCE=1, include the transform script in the input hash
    // so changes to the script invalidate the cache.
    if std::env::var_os("RISC0_TREE_REDUCE").is_some() {
        in_tree.push(PathBuf::from("kernels/intel/tree_reduce_fma.py"));
    }
    let tree_reduce_tag = if std::env::var_os("RISC0_TREE_REDUCE").is_some() {
        "tr1"
    } else {
        "tr0"
    };
    // T3.1 cross-function CSE: hoist shared args[group][col*steps+...] reads
    // from the 20 noinline sub-functions into the entry poly_fp.
    // DEFAULT ON (R3-A01 confirmed +11.5% E2E). Disable with RISC0_POLY_FP_CSE=0.
    let cse_enabled = !matches!(
        std::env::var("RISC0_POLY_FP_CSE").as_deref(),
        Ok("0") | Ok("false") | Ok("off") | Ok("OFF"),
    );
    if cse_enabled {
        in_tree.push(PathBuf::from("kernels/intel/poly_fp_cse.py"));
    }
    let cse_top = std::env::var("RISC0_POLY_FP_CSE_TOP").unwrap_or_default();
    let cse_min = std::env::var("RISC0_POLY_FP_CSE_MIN_SHARE").unwrap_or_default();
    // Normalize the env var into "1"/"0" so default-on and explicit "1" hash the same.
    let cse_state = if cse_enabled { "1" } else { "0" };
    // Tier B4: RISC0_LSC_HINTS toggles the cached_load() wrap; folded into
    // cse_tag so flipping it invalidates the eval_check stamp.
    let lsc_state = if matches!(
        std::env::var("RISC0_LSC_HINTS").as_deref(),
        Ok("1") | Ok("true") | Ok("on") | Ok("ON")
    ) {
        "1"
    } else {
        "0"
    };
    let cse_raw = format!("{}|{}|{}|lsc={}", cse_state, cse_top, cse_min, lsc_state);
    let cse_tag = format!("cse-{:x}", {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        cse_raw.hash(&mut h);
        (h.finish() & 0xffffffff) as u32
    });
    // Tier-4 IGC flag experiment: tag includes the RISC0_IGC_EXTRA_OPTS
    // value so changing it triggers rebuild; an empty/unset value collapses
    // to the canonical "iex-" tag (no extra opts).
    let iex_raw = std::env::var("RISC0_IGC_EXTRA_OPTS").unwrap_or_default();
    let iex_tag = format!("iex-{:x}", {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        iex_raw.hash(&mut h);
        (h.finish() & 0xffffffff) as u32
    });
    let fast_tag = if intel_eval_fast_enabled() {
        in_tree.push(PathBuf::from("kernels/intel/eval_nozero.py"));
        in_tree.push(PathBuf::from("kernels/intel/eval_slab.py"));
        let mut scripts: Vec<PathBuf> = std::fs::read_dir("kernels/intel/evalfast")
            .expect("kernels/intel/evalfast")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |x| x == "py"))
            .collect();
        scripts.sort();
        in_tree.extend(scripts);
        if intel_eval_xmx_enabled() {
            "fast3-xmx"
        } else {
            "fast3"
        }
    } else {
        "fast0"
    };
    let pairs = build_pairs(in_tree, cxx_root);
    let variant = intel_variant_tag();
    stamp::hash_labeled(
        &pairs,
        &[
            icpx_version,
            INTEL_STAMP_VERSION,
            "mono",
            &variant,
            tree_reduce_tag,
            &iex_tag,
            &cse_tag,
            fast_tag,
            &risc0_build_kernel::intel_sycl_targets_arg(),
        ],
    )
}

/// Inputs for the multipass eval_check kernels: the mono inputs plus the
/// `gen_multipass.py` generator script that produces pass1/pass2
/// amalgamations.
fn compute_multipass_hash(cxx_root: &str, icpx_version: &str) -> String {
    let in_tree = vec![
        PathBuf::from("kernels/intel/eval_check.cpp"),
        PathBuf::from("kernels/intel/gen_multipass.py"),
        PathBuf::from("kernels/cxx/rust_poly_fp_0.cpp"),
        PathBuf::from("kernels/cxx/rust_poly_fp_1.cpp"),
        PathBuf::from("kernels/cxx/rust_poly_fp_2.cpp"),
        PathBuf::from("kernels/cxx/rust_poly_fp_3.cpp"),
        PathBuf::from("build.rs"),
    ];
    let pairs = build_pairs(in_tree, cxx_root);
    let variant = intel_variant_tag();
    stamp::hash_labeled(
        &pairs,
        &[
            icpx_version,
            INTEL_STAMP_VERSION,
            "multipass",
            &variant,
            &risc0_build_kernel::intel_sycl_targets_arg(),
        ],
    )
}

/// Inputs for the witgen kernel: `ffi_witgen.cpp`, `steps.cpp`, all Intel
/// kernel headers, all shared cxx headers / `.inc` fragments, and the same
/// external `cxx_root` headers as mono/multipass.
fn compute_witgen_hash(cxx_root: &str, icpx_version: &str) -> String {
    let mut in_tree: Vec<PathBuf> = vec![
        PathBuf::from("kernels/intel/ffi_witgen.cpp"),
        PathBuf::from("kernels/intel/accum_batchinv.py"),
        PathBuf::from("kernels/cxx/steps.cpp"),
        PathBuf::from("build.rs"),
    ];
    in_tree.extend(glob_paths("kernels/intel/*.h"));
    in_tree.extend(glob_paths("kernels/cxx/*.h"));
    in_tree.extend(glob_paths("kernels/cxx/*.h.inc"));
    in_tree.extend(glob_paths("kernels/cxx/*.cpp.inc"));
    let pairs = build_pairs(in_tree, cxx_root);
    let variant = intel_variant_tag();
    stamp::hash_labeled(
        &pairs,
        &[
            icpx_version,
            INTEL_STAMP_VERSION,
            "witgen",
            &variant,
            &risc0_build_kernel::intel_sycl_targets_arg(),
        ],
    )
}

/// Recovery shim for the witgen kernel build.
///
/// The witgen amalgamation is large (~14k lines after `__attribute__((noinline))`
/// injection) and its IGC compile path triggers a known SIGSEGV in
/// `PreCompiledFuncImport::replaceFunc` on stock Intel IGC 2.30.1
/// (icpx exit 254 / ocloc exit 226). The fix lives upstream-ready as
/// `inteldebug/igc-bug-report/0001-PreCompiledFuncImport-fix-nullptr-arg-push-in-replac.patch`
/// and requires either: (a) a patched libigc on `LD_LIBRARY_PATH` at
/// build time, or (b) re-using an artifact from a build that DID have
/// the patched libigc.
///
/// **Recovery is intentionally witgen-only**: eval_check (mono) panics
/// on icpx failure because stale .so = wrong constraints = unsound
/// proofs; multipass is an optional optimization that degrades cleanly
/// at runtime. Witgen has a CPU fallback in the production prove path,
/// and its inputs (`steps.cpp`) are far more stable than `poly_fp.cpp`,
/// making artifact reuse safe.
///
/// **Same-variant only**: witgen icpx now uses `intel_xs_options()`
/// which embeds the GRF mode and (optional) SIMD width into the
/// emitted SPIR-V. Cross-variant fallback (e.g. copying the GRF=256
/// `_default` .so into a GRF=128 cache) would silently link a
/// wrong-flag binary, invalidating Phase 0c A/B measurements. So
/// recovery only matches an existing artifact at the same (GRF, SIMD)
/// variant — both in the current build's profile and the sibling
/// release profile.
///
/// Search order (most-current first): same-profile same-variant cache,
/// release same-variant cache (skipping self-pointer if current build
/// is itself release+same-variant), legacy unsuffixed release cache
/// only when running with the default variant.
///
/// Atomic write via tempfile + rename (matches Phase 0a stamp pattern):
/// an interrupt mid-copy leaves no partial .so on disk.
///
/// Visibility via `cargo:warning=` so operators see the recovery
/// message in default `cargo build` output (build-script `eprintln!`
/// is suppressed unless `-vv` or build failure).
fn try_recover_witgen_so(out_dir: &Path, dest_so: &Path) {
    // Resolve target/ root structurally from OUT_DIR layout
    // (`<target>/<profile>/build/<crate>-<hash>/out`) instead of
    // string-matching "target" — works under custom CARGO_TARGET_DIR
    // and `--target=<triple>` cross-compile dirs.
    let target_root = out_dir.ancestors().nth(4).map(Path::to_path_buf);
    let target_root = match target_root {
        Some(r) => r,
        None => {
            warn(&format!(
                "witgen-recovery: could not resolve target/ root from OUT_DIR={}",
                out_dir.display()
            ));
            return;
        }
    };
    let variant = intel_variant_tag();
    let so_name_path =
        risc0_build_kernel::intel_library_path(Path::new(""), "risc0_rv32im_intel_witgen");
    let so_name = so_name_path.to_str().unwrap();
    // Same-variant only. We DO include both same-profile and release
    // candidates — a debug build's recovery can pull from release
    // (icpx flags are profile-invariant for the same variant), but we
    // skip self-pointer (release-mode build pointing at itself).
    let candidates: Vec<PathBuf> = ["debug", "release"]
        .iter()
        .map(|profile| {
            target_root
                .join(profile)
                .join(format!("intel_rv32im_cache_{variant}"))
                .join(so_name)
        })
        // Legacy pre-Phase-0c release cache (no variant suffix). Only
        // valid when current build is itself the default variant —
        // the unsuffixed cache predates variant tagging and was
        // implicitly GRF=256/SIMD=auto. Including it for non-default
        // variants would re-introduce the cross-variant unsoundness.
        .chain(std::iter::once_with(|| {
            if variant == "default" {
                target_root
                    .join("release")
                    .join("intel_rv32im_cache")
                    .join(so_name)
            } else {
                // Sentinel that won't match `dest_so` and won't exist.
                target_root.join("__phase0c_unreachable__").join(so_name)
            }
        }))
        .collect();
    let dest_canon = std::fs::canonicalize(dest_so.parent().unwrap_or(dest_so))
        .ok()
        .map(|p| p.join(dest_so.file_name().unwrap_or_default()));
    for cand in &candidates {
        if !cand.exists() {
            continue;
        }
        // Skip self-pointer: when current build is release+same-variant,
        // candidate #1 (release/intel_rv32im_cache_<variant>/...) IS
        // dest_so. `std::fs::copy(p, p)` truncates the source on Linux,
        // producing a 0-byte .so. Compare canonicalized paths to also
        // catch symlink-equivalent targets.
        let cand_canon = std::fs::canonicalize(cand).ok();
        if let (Some(c), Some(d)) = (&cand_canon, &dest_canon) {
            if c == d {
                continue;
            }
        }
        // Atomic copy: write to a tempfile in the dest's parent dir,
        // then rename. POSIX rename is atomic same-fs.
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = dest_so.with_file_name(format!(
            "{}.{}.{}.tmp",
            dest_so
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| so_name.into()),
            pid,
            nanos
        ));
        match std::fs::copy(cand, &tmp).and_then(|_| std::fs::rename(&tmp, dest_so)) {
            Ok(_) => {
                warn(&format!(
                    "witgen-recovery: copied {} from {} (build patched-IGC unavailable). \
                     NOTE: artifact may be stale relative to current steps.cpp; \
                     install patched IGC per inteldebug/igc-bug-report/FIXED_INTEL_COMPILER_README.md \
                     for a fresh rebuild.",
                    so_name,
                    cand.display()
                ));
                return;
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                warn(&format!(
                    "witgen-recovery: found {} but copy/rename failed: {}",
                    cand.display(),
                    e
                ));
            }
        }
    }
    warn(&format!(
        "witgen-recovery: no same-variant fallback for variant '{}' under {}. \
         Linker will fail on `risc0_circuit_rv32im_intel_witgen` unless the \
         patched IGC build path is available — see \
         inteldebug/igc-bug-report/FIXED_INTEL_COMPILER_README.md.",
        variant,
        target_root.display()
    ));
}

/// Emit a build-script warning visible in default `cargo build` output.
/// `cargo:warning=...` is the only build-script directive surfaced
/// without `-vv`, so operator-actionable diagnostics use it.
fn warn(msg: &str) {
    println!("cargo:warning={msg}");
}

#[allow(dead_code)]
fn build_intel_kernels() {
    rerun_if_changed("kernels/intel");
    rerun_if_changed("kernels/cxx");

    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    // The Intel kernels' field arithmetic (incl. the RISC0_INTEL_EVAL_FAST
    // device paths) lives in these shared headers; the stamps hash them, but
    // cargo must also re-run this script when they change.
    for header in ["fp.h", "fpext.h"] {
        println!("cargo:rerun-if-changed={cxx_root}/{header}");
    }

    let icpx = risc0_build_kernel::find_icpx();

    // Cache directory for the expensive eval_check compilation. Anchor on the
    // <profile> directory under target/ regardless of profile name so custom
    // profiles (e.g. `cargo build --profile=ci-fast`) still get a stable
    // cache. OUT_DIR is always `<target>/<profile>/build/<crate>-<hash>/out`,
    // so the profile dir is the ancestor whose parent's filename is "target".
    let cache_dir = out_dir
        .ancestors()
        .find(|p| {
            p.parent()
                .and_then(|q| q.file_name())
                .map(|n| n == std::ffi::OsStr::new("target"))
                .unwrap_or(false)
        })
        .map(|p| p.join(intel_cache_subdir()))
        .unwrap_or_else(|| out_dir.join(intel_cache_subdir()));
    // Re-run build.rs when Phase 0c knobs change so the active cache_dir
    // and stamp inputs reflect the new variant.
    println!("cargo:rerun-if-env-changed=RISC0_INTEL_GRF_MODE");
    println!("cargo:rerun-if-env-changed=RISC0_INTEL_SIMD_WIDTH");
    // Same env vars that build_rocm_kernels declares; mirrored here so Intel
    // builds also invalidate the stamp when any of them flip. Without this,
    // toggling RISC0_LSC_HINTS / RISC0_POLY_FP_CSE* / RISC0_TREE_REDUCE /
    // RISC0_IGC_EXTRA_OPTS silently reuses the stale .so cache.
    println!("cargo:rerun-if-env-changed=RISC0_IGC_EXTRA_OPTS");
    println!("cargo:rerun-if-env-changed=RISC0_TREE_REDUCE");
    println!("cargo:rerun-if-env-changed=RISC0_POLY_FP_CSE");
    println!("cargo:rerun-if-env-changed=RISC0_POLY_FP_CSE_TOP");
    println!("cargo:rerun-if-env-changed=RISC0_POLY_FP_CSE_MIN_SHARE");
    println!("cargo:rerun-if-env-changed=RISC0_LSC_HINTS");
    println!("cargo:rerun-if-env-changed=RISC0_INTEL_EVAL_FAST");
    println!("cargo:rerun-if-env-changed=RISC0_INTEL_EVAL_XMX");
    std::fs::create_dir_all(&cache_dir).unwrap();

    // Compute SHA-256 hashes of the inputs to each of the three icpx
    // compilations. Each stamp covers a focused input set; all stamps
    // include build.rs, the resolved icpx version, and the cxx_root path
    // so toolchain or build-script changes invalidate the cache.
    let icpx_version = stamp::icpx_version(&icpx);
    let mono_hash = compute_mono_hash(&cxx_root, &icpx_version);
    let multipass_hash = compute_multipass_hash(&cxx_root, &icpx_version);
    let witgen_hash = compute_witgen_hash(&cxx_root, &icpx_version);

    let so_path = risc0_build_kernel::intel_library_path(&cache_dir, "risc0_rv32im_intel");

    // Check if we can skip rebuild: artifact present, stamp parses, hash
    // matches the recomputed-from-source hash, status was "ok" last build.
    let stamp_path = cache_dir.join("intel_eval_check.stamp");
    let noxmx_targets = (intel_eval_fast_enabled() && intel_eval_xmx_enabled())
        .then(intel_non_xmx_targets)
        .flatten();
    let noxmx_path = risc0_build_kernel::intel_library_path(&cache_dir, "risc0_rv32im_intel_noxmx");
    let need_rebuild = stamp::need_rebuild(&so_path, &stamp_path, &mono_hash)
        || (noxmx_targets.is_some() && !noxmx_path.exists());

    if need_rebuild {
        eprintln!("Building Intel SYCL eval_check kernel...");

        // T3.2: Optional source-codegen pass that transforms Horner-style
        // FpExt FMA chains into balanced tree reductions. Cuts the critical-
        // path add-dependency depth from O(N) to O(log N) per chain. Opt-in
        // via RISC0_TREE_REDUCE=1; controls which rust_poly_fp_*.cpp inputs
        // feed the amalgamation. The transform is mathematically exact
        // (commutativity + associativity of FpExt addition).
        let use_tree_reduce = std::env::var_os("RISC0_TREE_REDUCE").is_some();
        let mut current_src_dir: std::path::PathBuf = std::path::PathBuf::from("kernels/cxx");
        if use_tree_reduce {
            let dst = out_dir.join("poly_fp_tree_reduced");
            std::fs::create_dir_all(&dst).unwrap();
            let script = std::path::PathBuf::from("kernels/intel/tree_reduce_fma.py");
            let status = risc0_build_kernel::python_command()
                .arg(&script)
                .arg(&current_src_dir)
                .arg(&dst)
                .status()
                .expect("Failed to run tree_reduce_fma.py");
            if !status.success() {
                panic!("tree_reduce_fma.py failed");
            }
            eprintln!(
                "  RISC0_TREE_REDUCE=1: using tree-reduced poly_fp sources at {}",
                dst.display()
            );
            current_src_dir = dst;
        }

        // T3.1: Cross-function CSE — hoist shared args[buf][col*steps+...]
        // reads from the 20 noinline sub-functions into the entry poly_fp.
        // Bit-exact (same global loads, just relocated and forwarded by-value).
        // DEFAULT ON: R3-A01 controlled A/B measured +11.5% E2E vs baseline
        // (CSE median 16.876s vs baseline 19.060s on B70 BMG-G31, 5-seg fib).
        // Disable with RISC0_POLY_FP_CSE=0.
        // Tunables: RISC0_POLY_FP_CSE_TOP (default 16), RISC0_POLY_FP_CSE_MIN_SHARE (default 10).
        let cse_enabled = !matches!(
            std::env::var("RISC0_POLY_FP_CSE").as_deref(),
            Ok("0") | Ok("false") | Ok("off") | Ok("OFF"),
        );
        // Tier B4: RISC0_LSC_HINTS=1 wraps remaining argK[...] reads with
        // ::risc0::lsc::cached_load() for IGC LSC L1+L3-cached hints on the
        // hot eval_check loads. Requires CSE enabled (the wrap runs as a
        // CSE post-pass).
        let lsc_hints_enabled = matches!(
            std::env::var("RISC0_LSC_HINTS").as_deref(),
            Ok("1") | Ok("true") | Ok("on") | Ok("ON")
        );
        if cse_enabled {
            let dst = out_dir.join("poly_fp_cse");
            std::fs::create_dir_all(&dst).unwrap();
            let script = std::path::PathBuf::from("kernels/intel/poly_fp_cse.py");
            let cse_top =
                std::env::var("RISC0_POLY_FP_CSE_TOP").unwrap_or_else(|_| "16".to_string());
            let cse_min =
                std::env::var("RISC0_POLY_FP_CSE_MIN_SHARE").unwrap_or_else(|_| "10".to_string());
            let mut script_cmd = risc0_build_kernel::python_command();
            script_cmd
                .arg(&script)
                .arg(&current_src_dir)
                .arg(&dst)
                .arg("--top")
                .arg(&cse_top)
                .arg("--min-share")
                .arg(&cse_min);
            if lsc_hints_enabled {
                script_cmd.arg("--lsc-hints");
            }
            let status = script_cmd.status().expect("Failed to run poly_fp_cse.py");
            if !status.success() {
                panic!("poly_fp_cse.py failed");
            }
            eprintln!(
                "  RISC0_POLY_FP_CSE=1 (top={}, min-share={}, lsc_hints={}): hoisted-CSE sources at {}",
                cse_top, cse_min, lsc_hints_enabled, dst.display(),
            );
            current_src_dir = dst;
        }
        let poly_fp_dir = current_src_dir;

        // Collects the arguments for intel_build_library (not run itself).
        let mut cmd = Command::new(&icpx);
        cmd.arg("-fsycl")
            .arg("-std=c++17")
            .arg("-Os") // -Os + noinline + 256 GRF: best eval_check perf (18MB vs 30MB -O1, 2x fewer icache misses)
            // Tell ocloc to skip expensive optimization passes via -cl-opt-disable.
            // Without this, ocloc takes 3+ hours for the 52K-line kernel.
            // AOT compilation for BMG. With __noinline__ on the 20 sub-functions,
            // ocloc should handle this in reasonable time (est. 15-30 min).
            .arg("-Wno-unused-parameter")
            .arg("-Wno-unused-function")
            .arg("-Wno-unused-variable")
            .arg("-Wno-sign-compare")
            .arg(format!("-I{cxx_root}"))
            .arg("-Ikernels/cxx")
            .arg("-Ikernels/intel");

        // Create an amalgamation file that includes all poly_fp sources
        // in a single translation unit (required for SYCL device code).
        // We handle the kInvRate redefinition by including all files in
        // a namespace wrapper with the constant defined once.
        let amalg_path = out_dir.join("intel_eval_check_amalg.cpp");
        let mut amalg = String::new();
        amalg.push_str("// Auto-generated amalgamation for SYCL device code\n");
        amalg.push_str("#include \"fp.h\"\n");
        amalg.push_str("#include \"fpext.h\"\n");
        amalg.push_str("#include <cstdint>\n");
        if lsc_hints_enabled {
            // Tier B4: pull in the cached_load helper at amalgamation level so
            // the wrapped reads in rust_poly_fp_*.cpp resolve.
            amalg.push_str("#include \"cached_load.h\"\n");
        }
        amalg.push_str("namespace risc0::circuit::rv32im_v2 {\n");
        amalg.push_str("constexpr size_t kInvRate = 4;\n");
        // Include the function bodies but skip their preamble (includes + kInvRate).
        // CRITICAL: inject __attribute__((noinline)) before each rv32im_v2_* function
        // DEFINITION (not declarations). Without noinline, LLVM/IGC tries to inline
        // all 52K lines into one mega-function, causing ocloc to take 3+ hours.
        // Tested selective inlining (pairs) — SLOWER due to increased per-function
        // register pressure. All-noinline with 256 GRF is optimal.
        for i in 0..4 {
            let src =
                std::fs::read_to_string(poly_fp_dir.join(format!("rust_poly_fp_{i}.cpp"))).unwrap();
            if let Some(ns_start) = src.find("namespace risc0::circuit::rv32im_v2 {") {
                let body_start = ns_start + "namespace risc0::circuit::rv32im_v2 {".len();
                if let Some(body_end) = src.rfind('}') {
                    let body = &src[body_start..body_end];
                    let mut modified = String::new();
                    for line in body.lines() {
                        if line.starts_with("FpExt rv32im_v2_") && line.contains('{') {
                            modified.push_str("__attribute__((noinline)) ");
                        }
                        modified.push_str(line);
                        modified.push('\n');
                    }
                    amalg.push_str(&modified);
                }
            }
        }
        amalg.push_str("} // namespace risc0::circuit::rv32im_v2\n");
        // Append the monolithic kernel wrapper
        amalg.push_str(&std::fs::read_to_string("kernels/intel/eval_check.cpp").unwrap());
        std::fs::write(&amalg_path, &amalg).unwrap();

        // Also save the mono amalgamation (without kernel wrapper) for multipass generation
        let mono_path = out_dir.join("intel_eval_check_mono.cpp");
        let mono_content = amalg[..amalg
            .rfind("} // namespace risc0::circuit::rv32im_v2")
            .unwrap()
            + "} // namespace risc0::circuit::rv32im_v2\n".len()]
            .to_string();
        std::fs::write(&mono_path, &mono_content).unwrap();

        // RISC0_INTEL_EVAL_FAST post-passes run on the mono build's copy only
        // (the multipass generator keeps reading the untransformed mono_path).
        let eval_fast = intel_eval_fast_enabled();
        let mut xs_options = intel_xs_options();
        let compile_src = if eval_fast {
            let fast_path = out_dir.join("intel_eval_check_amalg_fast.cpp");
            let work_dir = out_dir.join("evalfast_work");
            let status = risc0_build_kernel::python_command()
                .arg("kernels/intel/evalfast/pipeline.py")
                .arg(&amalg_path)
                .arg(&fast_path)
                .arg(&work_dir)
                .status()
                .unwrap_or_else(|e| panic!("failed to run evalfast/pipeline.py: {e}"));
            if !status.success() {
                panic!(
                    "evalfast/pipeline.py failed (RISC0_INTEL_EVAL_FAST=0 builds the plain kernel)"
                );
            }
            cmd.arg("-DRISC0_INTEL_EVAL_FAST");
            cmd.arg("-DRISC0_INTEL_EVAL_LOHI");
            // LSC cache-control override 6 on loads and stores: -4.5 ms per
            // po2=20 segment on this kernel, bit-identical (7 gives wrong output).
            xs_options.push_str(" -igc_opts 'DisableRecompilation=1,LscLoadCacheControlOverride=6,LscStoreCacheControlOverride=6'");
            eprintln!(
                "  RISC0_INTEL_EVAL_FAST=1: compiling {}",
                fast_path.display()
            );
            fast_path
        } else {
            amalg_path.clone()
        };

        cmd.arg(risc0_build_kernel::intel_sycl_targets_arg()) // AOT with noinline sub-functions
            // Force 256 GRF mode: doubles register file from 8KB to 16KB per thread,
            // dramatically reducing the 42KB spill overhead. Trades occupancy (8→4 threads/EU)
            // for fewer spills — net win since kernel is spill-bound, not compute-bound.
            .arg("-Xs")
            .arg(format!("-options \"{xs_options}\""));

        let args: Vec<std::ffi::OsString> = cmd.get_args().map(|a| a.to_owned()).collect();
        let output = risc0_build_kernel::intel_build_library(
            &icpx,
            &args,
            &[compile_src],
            &cache_dir,
            "risc0_rv32im_intel",
        );
        // IGC/ocloc report warnings on stdout, the compiler driver on stderr.
        let log = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if !output.status.success() {
            stamp::write(&stamp_path, &mono_hash, &icpx_version, "failed");
            panic!("Intel eval_check compilation failed:\n{}", log);
        }
        // An unrecognised -igc_opts key is only a warning; with the fast path
        // that silently means the ~2x slower recompiled kernel.
        if log.contains("Invalid registry flag") {
            stamp::write(&stamp_path, &mono_hash, &icpx_version, "failed");
            panic!("IGC ignored an -igc_opts flag for eval_check:\n{}", log);
        }
        for line in log.lines().filter(|l| l.contains("spilled around")) {
            eprintln!("  {}", line.trim());
        }
        if let Some(targets) = &noxmx_targets {
            // The same kernel without the XMX linking, for the non-Xe2 targets only.
            let plain_path = out_dir.join("intel_eval_check_amalg_fast_noxmx.cpp");
            let status = risc0_build_kernel::python_command()
                .env("RISC0_INTEL_EVAL_XMX", "0")
                .arg("kernels/intel/evalfast/pipeline.py")
                .arg(&amalg_path)
                .arg(&plain_path)
                .arg(out_dir.join("evalfast_work_noxmx"))
                .status()
                .unwrap_or_else(|e| panic!("failed to run evalfast/pipeline.py: {e}"));
            if !status.success() {
                stamp::write(&stamp_path, &mono_hash, &icpx_version, "failed");
                panic!("evalfast/pipeline.py failed for the eval_check without XMX");
            }
            let mut plain_args: Vec<std::ffi::OsString> = args
                .iter()
                .filter(|a| !a.to_string_lossy().starts_with("-fsycl-targets="))
                .cloned()
                .collect();
            plain_args.push(targets.into());
            plain_args.push("-DRISC0_RV32IM_EVAL_NOXMX".into());
            let output = risc0_build_kernel::intel_build_library(
                &icpx,
                &plain_args,
                &[plain_path],
                &cache_dir,
                "risc0_rv32im_intel_noxmx",
            );
            if !output.status.success() {
                stamp::write(&stamp_path, &mono_hash, &icpx_version, "failed");
                panic!(
                    "Intel eval_check (without XMX) compilation failed:\n{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            eprintln!("  Built {}", noxmx_path.display());
        }
        stamp::write(&stamp_path, &mono_hash, &icpx_version, "ok");
        eprintln!("  Built {}", so_path.display());
    } else {
        eprintln!("Using cached Intel eval_check kernel");
    }

    // Link eval_check
    risc0_build_kernel::intel_link_library(&cache_dir, "risc0_rv32im_intel");
    if noxmx_targets.is_some() {
        intel_place_unlinked_library(&cache_dir, "risc0_rv32im_intel_noxmx");
    }

    // mono_path is the monolithic amalgamation without kernel wrapper (for multipass gen)
    let mono_path = out_dir.join("intel_eval_check_mono.cpp");

    // ========================================================================
    // Build 2-way multi-pass eval_check (two separate .so files)
    // Each .so has ~27K lines (half monolithic) → ~9MB GPU binary, fits in L2.
    // ========================================================================
    let pass1_so = risc0_build_kernel::intel_library_path(&cache_dir, "risc0_rv32im_intel_pass1");
    let pass2_so = risc0_build_kernel::intel_library_path(&cache_dir, "risc0_rv32im_intel_pass2");
    let multipass_stamp = cache_dir.join("intel_multipass.stamp");
    // Multipass cache is hit only when both pass-side .so files exist and the
    // stamp hash matches mono+gen_multipass.py inputs. Either .so missing or
    // hash drift triggers a full rebuild of both passes.
    let multipass_rebuild =
        stamp::need_rebuild_multi(&[&pass1_so, &pass2_so], &multipass_stamp, &multipass_hash);

    if multipass_rebuild {
        let multipass_script = PathBuf::from("kernels/intel/gen_multipass.py");
        if multipass_script.exists() {
            eprintln!("Building 2-way multi-pass eval_check kernels...");
            let pass1_amalg = out_dir.join("intel_eval_check_pass1.cpp");
            let pass2_amalg = out_dir.join("intel_eval_check_pass2.cpp");

            let mp_output = risc0_build_kernel::python_command()
                .arg(&multipass_script)
                .arg(&mono_path)
                .arg(&pass1_amalg)
                .arg(&pass2_amalg)
                .output()
                .expect("Failed to run gen_multipass.py");

            if mp_output.status.success() {
                let stderr = String::from_utf8_lossy(&mp_output.stderr);
                eprintln!("{}", stderr);

                // Compile pass1
                eprintln!("  Compiling pass1...");
                let p1_args: Vec<std::ffi::OsString> = [
                    "-fsycl".to_string(),
                    "-std=c++17".into(),
                    "-Os".into(),
                    "-Wno-unused-parameter".into(),
                    "-Wno-unused-function".into(),
                    "-Wno-unused-variable".into(),
                    "-Wno-sign-compare".into(),
                    format!("-I{cxx_root}"),
                    "-Ikernels/cxx".into(),
                    "-Ikernels/intel".into(),
                    risc0_build_kernel::intel_sycl_targets_arg(),
                    "-Xs".into(),
                    format!("-options \"{}\"", intel_xs_options()),
                ]
                .into_iter()
                .map(Into::into)
                .collect();
                let p1_output = risc0_build_kernel::intel_build_library(
                    &icpx,
                    &p1_args,
                    std::slice::from_ref(&pass1_amalg),
                    &cache_dir,
                    "risc0_rv32im_intel_pass1",
                );

                if !p1_output.status.success() {
                    let stderr = String::from_utf8_lossy(&p1_output.stderr);
                    eprintln!("  Pass1 compilation failed:\n{}", stderr);
                } else {
                    eprintln!("  Built {}", pass1_so.display());
                }

                // Compile pass2
                eprintln!("  Compiling pass2...");
                let p2_args: Vec<std::ffi::OsString> = [
                    "-fsycl".to_string(),
                    "-std=c++17".into(),
                    "-Os".into(),
                    "-Wno-unused-parameter".into(),
                    "-Wno-unused-function".into(),
                    "-Wno-unused-variable".into(),
                    "-Wno-sign-compare".into(),
                    format!("-I{cxx_root}"),
                    "-Ikernels/cxx".into(),
                    "-Ikernels/intel".into(),
                    risc0_build_kernel::intel_sycl_targets_arg(),
                    "-Xs".into(),
                    format!("-options \"{}\"", intel_xs_options()),
                ]
                .into_iter()
                .map(Into::into)
                .collect();
                let p2_output = risc0_build_kernel::intel_build_library(
                    &icpx,
                    &p2_args,
                    std::slice::from_ref(&pass2_amalg),
                    &cache_dir,
                    "risc0_rv32im_intel_pass2",
                );

                if !p2_output.status.success() {
                    let stderr = String::from_utf8_lossy(&p2_output.stderr);
                    eprintln!("  Pass2 compilation failed:\n{}", stderr);
                } else {
                    eprintln!("  Built {}", pass2_so.display());
                }

                if pass1_so.exists() && pass2_so.exists() {
                    stamp::write(&multipass_stamp, &multipass_hash, &icpx_version, "ok");
                } else {
                    stamp::write(&multipass_stamp, &multipass_hash, &icpx_version, "failed");
                }
            } else {
                stamp::write(&multipass_stamp, &multipass_hash, &icpx_version, "failed");
                let stderr = String::from_utf8_lossy(&mp_output.stderr);
                eprintln!("  gen_multipass.py failed:\n{}", stderr);
            }
        }
    } else {
        eprintln!("Using cached multi-pass eval_check kernels");
    }

    // Link multi-pass .so files if they exist
    if pass1_so.exists() {
        risc0_build_kernel::intel_link_library(&cache_dir, "risc0_rv32im_intel_pass1");
    }
    if pass2_so.exists() {
        risc0_build_kernel::intel_link_library(&cache_dir, "risc0_rv32im_intel_pass2");
    }

    // ========================================================================
    // Build Intel SYCL witgen kernel (separate .so, ~34s compile)
    // ========================================================================
    let witgen_so = risc0_build_kernel::intel_library_path(&cache_dir, "risc0_rv32im_intel_witgen");
    let witgen_stamp = cache_dir.join("intel_witgen.stamp");
    let witgen_rebuild = stamp::need_rebuild(&witgen_so, &witgen_stamp, &witgen_hash);

    if witgen_rebuild {
        eprintln!("Building Intel SYCL witgen kernel...");

        // Create witgen amalgamation: Intel headers + steps.cpp body + ffi_witgen.cpp
        let witgen_amalg_path = out_dir.join("intel_witgen_amalg.cpp");
        let mut witgen_amalg = String::new();
        witgen_amalg.push_str("// Auto-generated SYCL witgen amalgamation\n");
        witgen_amalg.push_str("#include \"steps.h\"\n");
        witgen_amalg.push_str("#include \"witgen.h\"\n");
        witgen_amalg.push_str("\n");
        witgen_amalg.push_str("namespace risc0::circuit::rv32im_v2::intel {\n");

        // Read steps.cpp, extract body, inject noinline
        let steps_src = std::fs::read_to_string("kernels/cxx/steps.cpp").unwrap();
        let ns_marker = "namespace risc0::circuit::rv32im_v2::cpu {";
        if let Some(ns_start) = steps_src.find(ns_marker) {
            let body_start = ns_start + ns_marker.len();
            if let Some(body_end) = steps_src.rfind('}') {
                let body = &steps_src[body_start..body_end];
                let mut noinline_count = 0;
                for line in body.lines() {
                    let s = line.trim_start();

                    // Inject noinline on function definitions
                    if !s.is_empty()
                        && !s.starts_with("//")
                        && !s.starts_with('#')
                        && !s.starts_with("namespace")
                        && !s.starts_with("using")
                        && !s.starts_with('}')
                        && !s.starts_with('{')
                        && !s.starts_with("if")
                        && !s.starts_with("for")
                        && !s.starts_with("while")
                        && !s.starts_with("switch")
                        && !s.starts_with("else")
                        && !s.starts_with("return")
                        && !s.starts_with("auto")
                        && !s.starts_with("Val ")
                        && !s.starts_with("ExtVal")
                        && !s.starts_with("size_t")
                        && s.contains('(')
                        && s.trim_end().ends_with('{')
                        && (s.contains("Struct ")
                            || s.starts_with("void step_")
                            || s.starts_with("ComponentStruct "))
                    {
                        witgen_amalg.push_str("__attribute__((noinline)) ");
                        noinline_count += 1;
                    }
                    witgen_amalg.push_str(line);
                    witgen_amalg.push('\n');
                }
                eprintln!("  Injected noinline on {} functions", noinline_count);
            }
        }
        witgen_amalg.push_str("} // namespace risc0::circuit::rv32im_v2::intel\n\n");
        // Append the kernel wrapper + extern implementations
        witgen_amalg.push_str(&std::fs::read_to_string("kernels/intel/ffi_witgen.cpp").unwrap());
        std::fs::write(&witgen_amalg_path, &witgen_amalg).unwrap();
        // Montgomery batch inversion of the FpExt inv_0 calls in each
        // exec_TopAccum mux arm (chunks of 8; 64 miscompiled under IGC spill
        // pressure, 8/16 verified bit-exact). Expect all but 4 of 411 batched.
        let st = risc0_build_kernel::python_command()
            .arg("kernels/intel/accum_batchinv.py")
            .arg(&witgen_amalg_path)
            .arg(&witgen_amalg_path)
            .arg("8")
            .output()
            .expect("accum_batchinv.py");
        let out = String::from_utf8_lossy(&st.stdout);
        assert!(
            st.status.success() && out.contains("remaining inv_0 in TopAccum 4"),
            "accum_batchinv.py failed or changed shape: {out} {}",
            String::from_utf8_lossy(&st.stderr)
        );

        // Collects the arguments for intel_build_library (not run itself).
        let mut cmd = Command::new(&icpx);
        cmd.arg("-fsycl")
            .arg("-std=c++17")
            .arg("-Os") // -Os: different opt passes to avoid icpx -O1 accum miscompilation
            // 32-bit Montgomery + inlined lazy FpExt mul (fp.h/fpext.h): accum 2x.
            .arg("-DRISC0_INTEL_EVAL_FAST")
            // 2026-04-21: patched IGC + no cl-opt-disable + explicit 256 GRF (committed setup).
            // 2026-04-22: -Xfinalizer presched-rp/spillAllowed flags tested via -options nesting,
            // ocloc rejected with exit 226. Need different syntax — deferred.
            .arg("-Xs")
            .arg(format!("-options \"{}\"", intel_xs_options()))
            .arg("-Wno-unused-parameter")
            .arg("-Wno-unused-function")
            .arg("-Wno-unused-variable")
            .arg("-Wno-sign-compare")
            .arg("-Wno-unused-but-set-variable")
            .arg(format!("-Ikernels/intel"))
            .arg(format!("-I{cxx_root}"))
            .arg("-Ikernels/cxx")
            .arg(risc0_build_kernel::intel_sycl_targets_arg());

        let args: Vec<std::ffi::OsString> = cmd.get_args().map(|a| a.to_owned()).collect();
        let output = risc0_build_kernel::intel_build_library(
            &icpx,
            &args,
            std::slice::from_ref(&witgen_amalg_path),
            &cache_dir,
            "risc0_rv32im_intel_witgen",
        );
        if !output.status.success() {
            stamp::write(&witgen_stamp, &witgen_hash, &icpx_version, "failed");
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Surface failure summary via cargo:warning so it's visible
            // in default cargo output; full stderr stays in eprintln
            // for operators who want to grep on -vv builds.
            warn(&format!(
                "Intel witgen compilation failed (icpx exit {:?}); attempting recovery...",
                output.status.code()
            ));
            eprintln!(
                "Intel witgen compilation failed (non-fatal): icpx exit {:?}.\n{}",
                output.status.code(),
                stderr
            );
            // The witgen kernel requires a patched IGC at build time
            // (see inteldebug/igc-bug-report/FIXED_INTEL_COMPILER_README.md).
            // If a previously-built `.so` exists in any sibling profile/
            // variant cache, copy it in so the linker can resolve
            // `risc0_circuit_rv32im_intel_witgen` and tests can run.
            // The .so is profile-and-variant invariant — same icpx flags
            // and inputs in release vs debug — so copying is sound.
            try_recover_witgen_so(&out_dir, &witgen_so);
        } else {
            stamp::write(&witgen_stamp, &witgen_hash, &icpx_version, "ok");
            eprintln!("  Built {}", witgen_so.display());
        }
    } else {
        eprintln!("Using cached Intel witgen kernel");
    }

    // Link witgen .so (if it exists)
    if witgen_so.exists() {
        risc0_build_kernel::intel_link_library(&cache_dir, "risc0_rv32im_intel_witgen");
    }

    // RPATH for runtime
    let intel_lib = PathBuf::from("/opt/intel/oneapi/compiler/latest/lib");
    if intel_lib.exists() {
        println!("cargo:rustc-link-search=native={}", intel_lib.display());
    }
}
