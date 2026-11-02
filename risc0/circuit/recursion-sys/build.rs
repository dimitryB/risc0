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

/// Build-cache stamp utilities for the Intel SYCL recursion kernel.
///
/// Replaces the previous `"built"` literal stamp (which never invalidated on
/// source changes) with content-addressed SHA-256 hashing. One stamp covers
/// the recursion eval_check compilation. Stamp file format is a small TOML:
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

    #[allow(dead_code)] // kept as a convenience entry-point for callers that
                        // don't need label/content separation
    pub fn hash(files: &[&Path], extras: &[&str]) -> String {
        let pairs: Vec<(PathBuf, PathBuf)> = files
            .iter()
            .map(|p| (p.to_path_buf(), p.to_path_buf()))
            .collect();
        hash_labeled(&pairs, extras)
    }

    pub fn hash_labeled(items: &[(PathBuf, PathBuf)], extras: &[&str]) -> String {
        let mut sorted: Vec<&(PathBuf, PathBuf)> = items.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let mut h = Sha256::new();
        h.update(b"STAMP-V2\0");
        for (label, content_path) in sorted {
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

    pub fn need_rebuild(artifact: &Path, stamp_path: &Path, expected_hash: &str) -> bool {
        need_rebuild_multi(&[artifact], stamp_path, expected_hash)
    }

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

const RECURSION_INTEL_STAMP_VERSION: &str = "recursion-v3-fast";

/// Glob `*.h`/`*.hpp`/`*.cuh` from an absolute external include root
/// (`DEP_RISC0_SYS_CXX_ROOT`) so upstream `fp.h`/`fpext.h` content edits
/// invalidate the stamp. Returns (label, content_path) pairs where the
/// label is a portable virtual `cxx_root/<basename>` path so the cache key
/// stays stable across machines with different absolute workspace roots.
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

/// Inputs for the recursion eval_check kernel: poly_fp + the SYCL kernel
/// wrapper + transitive cxx_root headers + local cxx headers/.inc fragments
/// (poly_fp.cpp transitively includes them).
fn compute_recursion_intel_hash(cxx_root: &str, icpx_version: &str) -> String {
    let mut in_tree: Vec<PathBuf> = vec![
        PathBuf::from("kernels/intel/eval_check.cpp"),
        PathBuf::from("kernels/cxx/poly_fp.cpp"),
        PathBuf::from("build.rs"),
    ];
    in_tree.push(PathBuf::from("kernels/intel/eval_lazy_chain.py"));
    in_tree.push(PathBuf::from("kernels/intel/eval_remat_loads.py"));
    in_tree.extend(glob_paths("kernels/cxx/*.h"));
    in_tree.extend(glob_paths("kernels/cxx/*.h.inc"));
    in_tree.extend(glob_paths("kernels/cxx/*.cpp.inc"));
    if std::env::var_os("RISC0_RECURSION_SPLIT_POLY_FP").is_some() {
        in_tree.push(PathBuf::from("kernels/intel/split_poly_fp.py"));
    }
    let mut pairs: Vec<(PathBuf, PathBuf)> = in_tree.into_iter().map(|p| (p.clone(), p)).collect();
    pairs.extend(cxx_root_headers(cxx_root));
    // R3-A06 follow-up: include RISC0_RECURSION_OPTIMIZE env in stamp so the
    // cache invalidates when the user flips the optimization mode.
    let opt_tag = if std::env::var_os("RISC0_RECURSION_OPTIMIZE").is_some() {
        "opt1"
    } else {
        "opt0"
    };
    let split_tag = if std::env::var_os("RISC0_RECURSION_SPLIT_POLY_FP").is_some() {
        let n = std::env::var("RISC0_RECURSION_SPLIT_N").unwrap_or_else(|_| "10".to_string());
        format!("split{n}")
    } else {
        "nosplit".to_string()
    };
    stamp::hash_labeled(
        &pairs,
        &[
            icpx_version,
            RECURSION_INTEL_STAMP_VERSION,
            "recursion",
            opt_tag,
            &split_tag,
            &remat_distance(),
            &risc0_build_kernel::intel_sycl_targets_arg(),
        ],
    )
}

/// The statement distance past which eval_remat_loads.py reloads a buffer value instead of
/// keeping it live (`RISC0_RECURSION_REMAT_G` overrides).
///
/// With icpx/ocloc 2026.0 and eval_check.cpp's loop over rows, 200 or less keeps poly_fp's
/// values out of private memory: the kernel's private size is 896 B per work item, against
/// 527,232 B at 300 and above (the B70-era 400) or without the loop. On an Arc B580 that took
/// eval_check from 172 to 68 ms per recursion proof (lift/join of 32 po2 20 segments 17.1 ->
/// 10.4 s); 100 was 72 ms, 200 69 ms, 300 and 800 172 ms. `warn_private_size` checks the result.
/// Warns if a kernel in the built library `lib` uses more than 64 KiB of private memory per work
/// item, as its embedded zeInfo reports (`private_size: N`): eval_check then runs ~2.5x slower
/// (see `remat_distance`), which a compiler or generated-code change can bring about silently.
fn warn_private_size(lib: &Path) {
    let Ok(bytes) = std::fs::read(lib) else {
        return;
    };
    let key = b"private_size:";
    let mut largest = 0u64;
    for i in 0..bytes.len().saturating_sub(key.len()) {
        if &bytes[i..i + key.len()] == key {
            let digits: String = bytes[i + key.len()..]
                .iter()
                .map(|&b| b as char)
                .skip_while(|c| *c == ' ')
                .take_while(|c| c.is_ascii_digit())
                .collect();
            largest = largest.max(digits.parse().unwrap_or(0));
        }
    }
    if largest > 64 << 10 {
        println!(
            "cargo:warning=recursion eval_check: a kernel uses {largest} B of private memory per \
             work item (expected under 64 KiB); it will run ~2.5x slower. See remat_distance in \
             risc0/circuit/recursion-sys/build.rs."
        );
    }
}

fn remat_distance() -> String {
    std::env::var("RISC0_RECURSION_REMAT_G").unwrap_or_else(|_| "150".into())
}

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("Features 'cuda' and 'rocm' are mutually exclusive. Enable only one GPU backend.");

fn main() {
    let rocm = env::var("CARGO_FEATURE_ROCM").is_ok();

    // Set the GPU builds' environment before any thread starts: the CPU build below runs
    // concurrently and reads the environment.
    if rocm {
        env::set_var("HIP_PLATFORM", "amd");
    }

    // The CPU and GPU kernels share nothing, so build them concurrently.
    std::thread::scope(|scope| {
        scope.spawn(build_cpu_kernels);

        if env::var("CARGO_FEATURE_CUDA").is_ok() {
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
    // step_exec.cpp and poly_fp.cpp (1-2 MB of generated C++) take far longer to compile than
    // the rest: start them at once, each on its own thread.
    let (big, rest): (Vec<PathBuf>, Vec<PathBuf>) =
        glob_paths("kernels/cxx/*.cpp").into_iter().partition(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n == "step_exec.cpp" || n == "poly_fp.cpp")
        });
    KernelBuild::new(KernelType::Cpp)
        .files(rest)
        .files_on_own_threads(big)
        .include(env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap())
        .compile("risc0_recursion_cpu");
}

fn build_cuda_kernels() {
    rerun_if_changed("kernels/cuda");

    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();

    // Split build: step/ffi kernels get -maxrregcount=96 for higher occupancy,
    // while eval_check/supra kernels keep unrestricted registers.
    KernelBuild::new(KernelType::Cuda)
        .files(glob_paths("kernels/cuda/step_*.cu"))
        .file("kernels/cuda/ffi.cu")
        .deps(["kernels/cuda"])
        .flag("-DFEATURE_BABY_BEAR")
        .flag("-maxrregcount=96")
        .include(&cuda_root)
        .include(&cxx_root)
        .include(&sppark_root)
        .compile("risc0_recursion_cuda_step");

    KernelBuild::new(KernelType::Cuda)
        .file("kernels/cuda/eval_check.cu")
        .file("kernels/cuda/accum_inl.cu")
        .file("kernels/cuda/ffi_supra.cu")
        .file("kernels/cuda/sppark.cu")
        .deps(["kernels/cuda"])
        .flag("-DFEATURE_BABY_BEAR")
        .include(&cuda_root)
        .include(&cxx_root)
        .include(&sppark_root)
        .compile("risc0_recursion_cuda");

    // eval_check_split.cu (generated by kernels/cuda/eval_check_split.py), compiled standalone
    // (NO --device-c), with its own __constant__ poly_mix: relocatable device code makes it ~25%
    // slower (6.3 vs 4.9 ms per proof). Like rv32im-sys's standalone TUs: skipped with
    // RISC0_SKIP_BUILD_KERNELS (as the builds above are), MSVC-aware, and cached under a key over
    // its sources, the command and the toolchain.
    if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
        return;
    }
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cache_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with("release") || p.ends_with("debug"))
        .map(|p| p.join("eval_check_cache"))
        .unwrap_or_else(|| out_dir.join("eval_check_cache"));
    std::fs::create_dir_all(&cache_dir).unwrap();
    let msvc = risc0_build_kernel::is_msvc_target();
    let obj_ext = risc0_build_kernel::object_extension();
    let obj = out_dir.join(format!("eval_check_split_standalone.{obj_ext}"));
    let use_native_arch =
        env::var_os("NVCC_PREPEND_FLAGS").is_none() && env::var_os("NVCC_APPEND_FLAGS").is_none();
    let mut cmd = risc0_build_kernel::nvcc_command();
    cmd.current_dir("kernels/cuda");
    if !msvc {
        cmd.arg(risc0_build_kernel::nvcc_ccbin_arg())
            .arg("-Xcompiler")
            .arg("-O3,-fPIC");
    }
    cmd.arg(risc0_build_kernel::cuda_cxx_std_flag())
        .arg("-m64")
        .arg("-Xptxas")
        .arg("-O3")
        .arg("-DFEATURE_BABY_BEAR")
        .arg("-diag-suppress=177,550,2922")
        .arg("-I")
        .arg(&cuda_root)
        .arg("-I")
        .arg(&cxx_root)
        .arg("-I")
        .arg(&sppark_root);
    if use_native_arch {
        cmd.arg("-arch=native");
    }
    cmd.arg("-c").arg("eval_check_split.cu");
    let sppark = Path::new(&sppark_root);
    let mut local: Vec<PathBuf> = ["h", "cuh", "inc"]
        .iter()
        .flat_map(|ext| glob_paths(&format!("kernels/cuda/*.{ext}")))
        .collect();
    local.sort();
    local.push(PathBuf::from("kernels/cuda/eval_check_split.cu"));
    let local: Vec<&Path> = local.iter().map(PathBuf::as_path).collect();
    let sources = risc0_build_kernel::kernel_sources_key(
        &[Path::new(&cuda_root), Path::new(&cxx_root), sppark],
        &local,
        |path| {
            path.starts_with(sppark)
                || !matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("c" | "cc" | "cpp" | "cu")
                )
        },
    );
    let native_key = if use_native_arch {
        risc0_build_kernel::native_gpu_arch_key(false)
    } else {
        String::new()
    };
    let key = risc0_build_kernel::hash_key(&[
        "recursion-cuda-eval-check-split-v1",
        &sources,
        &risc0_build_kernel::command_key(&cmd),
        &risc0_build_kernel::nvcc_environment_key(),
        &native_key,
    ]);
    risc0_build_kernel::cached_compile(
        &cache_dir,
        "eval_check_split_standalone",
        obj_ext,
        &key,
        &obj,
        |out| {
            let status = cmd
                .arg("-o")
                .arg(out)
                .status()
                .unwrap_or_else(|e| panic!("failed to run nvcc for eval_check_split.cu: {e}"));
            assert!(status.success(), "nvcc failed for eval_check_split.cu");
        },
    );
    risc0_build_kernel::add_to_static_lib(&out_dir, "risc0_recursion_cuda", &[obj]);
}

fn build_rocm_kernels() {
    let output = "risc0_recursion_cuda";
    rerun_if_changed("kernels/cuda");
    println!("cargo:rerun-if-env-changed=RISC0_HIP_ARCH");

    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    // The headers these kernels include: re-run when one changes (file by file for sppark, whose
    // `rust/sppark` symlink loop a directory would make cargo recurse).
    for root in [&sppark_root, &cuda_root, &cxx_root] {
        for file in risc0_build_kernel::kernel_source_files(Path::new(root)) {
            println!("cargo:rerun-if-changed={}", file.display());
        }
    }
    let hipcc = risc0_build_kernel::find_hipcc();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let kernel_dir = std::fs::canonicalize("kernels/cuda").unwrap();

    // Single-TU approach: amalgamate all .cu files EXCEPT ffi_supra.cu, eval_check.cu, sppark.cu
    // and eval_check_split.cu (which use sppark types that conflict with risc0's fpext.h in the
    // same TU).
    let cu_files = glob_paths("kernels/cuda/*.cu");
    // Three-group approach:
    // Group A (risc0 types): ffi.cu + step_*.cu — amalgamated
    // Group B (sppark eval): ffi_supra.cu + eval_check.cu — amalgamated
    // Group C, each compiled separately: sppark.cu (sppark host: gpu_t.cuh's fmt()
    //   can't compile for device when combined with device kernels) and
    //   eval_check_split.cu (sppark types, with its own poly_mix; as on CUDA)
    let mut risc0_amalg = String::new();
    let mut supra_amalg = String::new();
    let mut separate_files = Vec::new();
    // The generated witgen steps' trace-read asserts as plain traps (see rec_fast_assert.cuh).
    // RISC0_HIP_REC_ASSERT=1 keeps HIP's __assert_fail.
    println!("cargo:rerun-if-env-changed=RISC0_HIP_REC_ASSERT");
    let fast_assert = env::var("RISC0_HIP_REC_ASSERT").map_or(true, |v| v != "1");
    for cu in &cu_files {
        let name = cu.file_name().unwrap().to_str().unwrap();
        let abs = std::fs::canonicalize(cu).unwrap();
        let line = format!("#include \"{}\"\n", abs.display());
        if name == "sppark.cu" || name == "eval_check_split.cu" {
            separate_files.push(cu.clone());
        } else if name == "ffi_supra.cu" || name == "eval_check.cu" {
            supra_amalg.push_str(&line);
        } else {
            if fast_assert && name.starts_with("step_") {
                risc0_amalg.push_str(&format!(
                    "#include \"{}/rec_fast_assert.cuh\"\n",
                    kernel_dir.display()
                ));
            }
            risc0_amalg.push_str(&line);
        }
    }

    let risc0_amalg_path = out_dir.join("risc0_kernels_rocm.cu");
    std::fs::write(&risc0_amalg_path, &risc0_amalg).unwrap();
    let supra_amalg_path = out_dir.join("supra_kernels_rocm.cu");
    std::fs::write(&supra_amalg_path, &supra_amalg).unwrap();

    let include_cuda2hip = format!("{sppark_root}/util/cuda2hip.hpp");
    let kernel_dir_str = kernel_dir.to_str().unwrap();
    let arches = risc0_build_kernel::hip_arches();
    let mut base_flags: Vec<&str> = vec![
        "-x",
        "hip",
        "-std=c++17",
        "-O3",
        "-Wno-unused-function",
        "-Wno-unused-parameter",
        "-Wno-missing-braces",
        "-mllvm",
        "-amdgpu-early-inline-all=false",
        "-DFEATURE_BABY_BEAR",
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
    base_flags.extend(risc0_build_kernel::hip_host_flags());
    let obj_ext = risc0_build_kernel::object_extension();

    let mut obj_files = Vec::new();
    // Group A and eval_check_split.cu use the AMDGPU iterative max-occupancy scheduler: it keeps
    // the big unrolled kernels under the occupancy thresholds with fewer spills. Per recursion
    // proof (gfx1100 / gfx1201): ec_part<0..5> 19.3 -> 14.9 / 27.6 -> 17.1 ms (with the bounds in
    // eval_check_split.py), parStepComputeAccumInl+parStepVerifyAccumInl 2.36 -> 1.75 / 3.28 ->
    // 2.05 ms, parStepExec neutral / -3%. The Group B (supra) eval code has no such kernel.
    let mut maxocc_flags = base_flags.clone();
    maxocc_flags.extend(["-mllvm", "-amdgpu-sched-strategy=iterative-maxocc"]);
    for (amalg_path, obj_name) in [
        (&risc0_amalg_path, "risc0_kernels_rocm"),
        (&supra_amalg_path, "supra_kernels_rocm"),
    ] {
        let obj = out_dir.join(format!("{obj_name}.{obj_ext}"));
        let flags = if obj_name == "risc0_kernels_rocm" {
            &maxocc_flags
        } else {
            &base_flags
        };
        risc0_build_kernel::hip_compile(&hipcc, flags, amalg_path, &obj, &arches, None);
        obj_files.push(obj);
    }

    // Compile separate files individually: sppark.cu contains host-only code (gpu_t.cuh) that
    // can't be in the same TU as device kernels; eval_check_split.cu uses sppark types.
    for cu in &separate_files {
        let name = cu.file_name().unwrap().to_str().unwrap();
        let obj_name = format!("{}.{obj_ext}", name.trim_end_matches(".cu"));
        let obj = out_dir.join(&obj_name);
        let abs = std::fs::canonicalize(cu).unwrap();
        let flags = if name == "eval_check_split.cu" {
            &maxocc_flags
        } else {
            &base_flags
        };
        risc0_build_kernel::hip_compile(&hipcc, flags, &abs, &obj, &arches, None);
        obj_files.push(obj);
    }

    // On MSVC, where lld-link has no relocatable (`-r`) link, archive the objects themselves:
    // none is compiled with -fgpu-rdc, so each carries its own complete GPU code object and there
    // is no device code to link across them.
    if risc0_build_kernel::is_msvc_target() {
        let _ = std::fs::remove_file(out_dir.join(format!("{output}.lib")));
        risc0_build_kernel::add_to_static_lib(&out_dir, output, &obj_files);
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static={output}");
        risc0_build_kernel::emit_rocm_lib_link();
        return;
    }

    // Device-link: with -fgpu-rdc, device code is compiled as bitcode in each .o file.
    // We must link all device code together into a single GPU code object so that
    // cross-TU device function calls resolve and __noinline__ functions stay separate.
    // For the architectures the objects were compiled for (RISC0_HIP_ARCH, else `native`):
    // `native` alone fails on a machine without an AMD GPU, such as a CI runner, and links for
    // the wrong target on one whose GPU is not in RISC0_HIP_ARCH.
    let device_linked = out_dir.join("device_linked_rocm.o");
    {
        let mut cmd = Command::new(&hipcc);
        cmd.arg(format!("--offload-arch={arches}"))
            .arg("-fgpu-rdc")
            .arg("-fPIC")
            .arg("-r")
            .arg("-nostdlib");
        for obj in &obj_files {
            cmd.arg(obj);
        }
        cmd.arg("-o").arg(&device_linked);
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("failed to run hipcc device link: {e}"));
        assert!(status.success(), "hipcc device link failed");
    }

    // Archive: with -fgpu-rdc, the device-linked .o contains both the host stubs
    // and the fully linked GPU code, so we only archive it (not the individual .o files,
    // which would cause duplicate symbol errors).
    let lib_path = out_dir.join(format!("lib{output}.a"));
    let _ = std::fs::remove_file(&lib_path);
    let ar = risc0_build_kernel::find_ar_tool();
    let mut ar_cmd = Command::new(&ar);
    ar_cmd.arg("rcs").arg(&lib_path).arg(&device_linked);
    let status = ar_cmd.status().expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={output}");

    // Link against HIP runtime
    risc0_build_kernel::emit_rocm_lib_link();
}

#[allow(dead_code)]
fn build_intel_kernels() {
    rerun_if_changed("kernels/intel");
    rerun_if_changed("kernels/cxx");
    println!("cargo:rerun-if-env-changed=RISC0_RECURSION_OPTIMIZE");
    println!("cargo:rerun-if-env-changed=RISC0_RECURSION_REMAT_G");
    println!("cargo:rerun-if-env-changed=RISC0_RECURSION_SPLIT_POLY_FP");
    println!("cargo:rerun-if-env-changed=RISC0_RECURSION_SPLIT_N");

    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();

    let icpx = risc0_build_kernel::find_icpx();

    // Cache directory anchored on the <profile> dir under target/, regardless
    // of profile name (so custom profiles still cache). OUT_DIR layout is
    // `<target>/<profile>/build/<crate>-<hash>/out`.
    let cache_dir = out_dir
        .ancestors()
        .find(|p| {
            p.parent()
                .and_then(|q| q.file_name())
                .map(|n| n == std::ffi::OsStr::new("target"))
                .unwrap_or(false)
        })
        .map(|p| p.join("intel_recursion_cache"))
        .unwrap_or_else(|| out_dir.join("intel_recursion_cache"));
    std::fs::create_dir_all(&cache_dir).unwrap();

    let so_path = risc0_build_kernel::intel_library_path(&cache_dir, "risc0_recursion_intel");

    // Compute SHA-256 over the recursion eval_check inputs. Stamp invalidates
    // when any of poly_fp.cpp, eval_check.cpp, build.rs, or the icpx version
    // change.
    let icpx_version = stamp::icpx_version(&icpx);
    let recursion_hash = compute_recursion_intel_hash(&cxx_root, &icpx_version);

    // Check if we can skip rebuild: artifact present, stamp parses, hash
    // matches, status was "ok" last build.
    let stamp_path = cache_dir.join("intel_eval_check.stamp");
    let need_rebuild = stamp::need_rebuild(&so_path, &stamp_path, &recursion_hash);

    if need_rebuild {
        eprintln!("Building Intel SYCL eval_check kernel for recursion...");

        // R3-A06 follow-up: by default we use `-O1 -cl-opt-disable` for fast
        // builds (~3 min ocloc), trading runtime perf. Set
        // `RISC0_RECURSION_OPTIMIZE=1` to compile with full optimization (-O2,
        // no -cl-opt-disable). Compile time grows to ~30-50 min but the
        // GPU kernel may run 2-5× faster, saving 1-4% Succinct E2E.
        let recursion_optimize = std::env::var_os("RISC0_RECURSION_OPTIMIZE").is_some();
        // Collects the arguments for intel_build_library (not run itself).
        let mut cmd = Command::new(&icpx);
        cmd.arg("-fsycl").arg("-std=c++17");
        if recursion_optimize {
            cmd.arg("-O2")
                .arg("-Xs")
                .arg(format!("-options \"-cl-intel-256-GRF-per-thread\""));
        } else {
            // Fast path (same recipe as rv32im eval_check): -Os, 256 GRF,
            // RISC0_INTEL_EVAL_FAST device helpers (mandatory: without them the
            // monolith never finishes compiling), DisableRecompilation=1, and the
            // 4-entry vanishing-quotient table. Post-passes below: lazy poly_mix
            // chains + volatile load rematerialisation (see `remat_distance`).
            cmd.arg("-Os")
                .arg("-DRISC0_INTEL_EVAL_FAST")
                .arg("-DRECURSION_QTABLE")
                .arg("-Xs")
                .arg(
                    "-options \"-cl-intel-256-GRF-per-thread -igc_opts 'DisableRecompilation=1'\"",
                );
        }
        cmd.arg("-Wno-unused-parameter")
            .arg("-Wno-unused-function")
            .arg("-Wno-unused-variable")
            .arg("-Wno-sign-compare")
            .arg(format!("-I{cxx_root}"))
            .arg("-Ikernels/cxx");

        // Create an amalgamation file that includes poly_fp.cpp + kernel wrapper
        // in a single translation unit (required for SYCL device code).
        // The recursion poly_fp.cpp is a single monolithic function (~24K lines),
        // roughly half the size of rv32im's 52K. We mark the function itself as
        // noinline so it compiles as a separate device function.
        //
        // PR 2 / RISC0_RECURSION_SPLIT_POLY_FP=1 (default OFF — DO NOT FLIP):
        // attempts to split the monolith into N noinline sub-functions via
        // `kernels/intel/split_poly_fp.py`. The intent is to unblock
        // `RISC0_RECURSION_OPTIMIZE=1` (which hangs IGC on the 24K-line
        // monolith). Compiles cleanly but the resulting kernel triggers
        // `UR_RESULT_ERROR_DEVICE_LOST` at runtime — closure-based re-emission
        // inflates the .so 4.6× (13MB → 59.7MB) because recursion's codegen
        // front-loads ~689 buffer reads used throughout the function. The
        // right next-iteration design is to hoist buffer loads ONCE into
        // `Fp x33[N]` / `FpExt x34[M]` scratch arrays in the entry `poly_fp`
        // and pass pointers to sub-functions (rv32im's pattern). See the
        // header docstring of split_poly_fp.py for full details. Kept in
        // tree as a starting point for future work.
        let split_poly_fp = std::env::var_os("RISC0_RECURSION_SPLIT_POLY_FP").is_some();
        let amalg_path = out_dir.join("intel_recursion_eval_check_amalg.cpp");
        let mut amalg = String::new();
        amalg.push_str("// Auto-generated amalgamation for SYCL device code\n");
        amalg.push_str("#include \"fp.h\"\n");
        amalg.push_str("#include \"fpext.h\"\n");
        amalg.push_str("#include <cstdint>\n");
        amalg.push_str("namespace risc0::circuit::recursion {\n");
        amalg.push_str("constexpr size_t kInvRate = 4;\n");

        if split_poly_fp {
            // Run the splitter into OUT_DIR/poly_fp_split/, then include the
            // produced rust_poly_fp_{0,1}.cpp files' bodies.
            let split_dir = out_dir.join("poly_fp_split");
            std::fs::create_dir_all(&split_dir).unwrap();
            let split_script = PathBuf::from("kernels/intel/split_poly_fp.py");
            let split_n =
                std::env::var("RISC0_RECURSION_SPLIT_N").unwrap_or_else(|_| "10".to_string());
            let status = risc0_build_kernel::python_command()
                .arg(&split_script)
                .arg("kernels/cxx/poly_fp.cpp")
                .arg(&split_dir)
                .arg("--n")
                .arg(&split_n)
                .status()
                .expect("Failed to run split_poly_fp.py");
            if !status.success() {
                panic!("split_poly_fp.py failed");
            }
            eprintln!(
                "  RISC0_RECURSION_SPLIT_POLY_FP=1 (n={split_n}): split sources at {}",
                split_dir.display()
            );
            // Include both split files' bodies.
            for fname in &["rust_poly_fp_0.cpp", "rust_poly_fp_1.cpp"] {
                let src = std::fs::read_to_string(split_dir.join(fname)).unwrap();
                if let Some(ns_start) = src.find("namespace risc0::circuit::recursion {") {
                    let body_start = ns_start + "namespace risc0::circuit::recursion {".len();
                    if let Some(body_end) = src.rfind('}') {
                        amalg.push_str(&src[body_start..body_end]);
                        amalg.push('\n');
                    }
                }
            }
        } else {
            // Original monolithic path. Read poly_fp.cpp and extract the function
            // body, injecting __attribute__((noinline)) on the definition.
            let src = std::fs::read_to_string("kernels/cxx/poly_fp.cpp").unwrap();
            if let Some(ns_start) = src.find("namespace risc0::circuit::recursion {") {
                let body_start = ns_start + "namespace risc0::circuit::recursion {".len();
                if let Some(body_end) = src.rfind('}') {
                    let body = &src[body_start..body_end];
                    let mut modified = String::new();
                    let mut injected = false;
                    for line in body.lines() {
                        if !injected && line.starts_with("FpExt poly_fp(") && line.contains('{') {
                            modified.push_str("__attribute__((noinline)) ");
                            injected = true;
                        }
                        modified.push_str(line);
                        modified.push('\n');
                    }
                    amalg.push_str(&modified);
                }
            }
        }
        amalg.push_str("} // namespace risc0::circuit::recursion\n");
        // Append the kernel wrapper
        amalg.push_str(&std::fs::read_to_string("kernels/intel/eval_check.cpp").unwrap());
        if recursion_optimize || split_poly_fp {
            std::fs::write(&amalg_path, &amalg).unwrap();
        } else {
            // Post-passes on the default (fast) path; both scripts assert when
            // their patterns stop matching the generated code.
            let raw_path = out_dir.join("intel_recursion_eval_check_amalg_raw.cpp");
            let lazy_path = out_dir.join("intel_recursion_eval_check_amalg_lazy.cpp");
            std::fs::write(&raw_path, &amalg).unwrap();
            for (script, args) in [
                (
                    "kernels/intel/eval_lazy_chain.py",
                    vec![raw_path.clone(), lazy_path.clone()],
                ),
                (
                    "kernels/intel/eval_remat_loads.py",
                    vec![
                        lazy_path.clone(),
                        amalg_path.clone(),
                        PathBuf::from(remat_distance()),
                    ],
                ),
            ] {
                let status = risc0_build_kernel::python_command()
                    .arg(script)
                    .args(&args)
                    .status()
                    .unwrap_or_else(|e| panic!("failed to run {script}: {e}"));
                assert!(status.success(), "{script} failed");
            }
        }

        cmd.arg(risc0_build_kernel::intel_sycl_targets_arg()); // AOT for Battlemage

        let args: Vec<std::ffi::OsString> = cmd.get_args().map(|a| a.to_owned()).collect();
        let output = risc0_build_kernel::intel_build_library(
            &icpx,
            &args,
            std::slice::from_ref(&amalg_path),
            &cache_dir,
            "risc0_recursion_intel",
        );
        if !output.status.success() {
            stamp::write(&stamp_path, &recursion_hash, &icpx_version, "failed");
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("Intel recursion eval_check compilation failed:\n{}", stderr);
        }
        stamp::write(&stamp_path, &recursion_hash, &icpx_version, "ok");
        eprintln!("  Built {}", so_path.display());
        warn_private_size(&so_path);
    } else {
        eprintln!("Using cached Intel recursion eval_check kernel");
    }

    build_intel_accum(&icpx, &icpx_version, &cxx_root, &cache_dir, &out_dir);

    // Link
    risc0_build_kernel::intel_link_library(&cache_dir, "risc0_recursion_intel");
    risc0_build_kernel::intel_link_library(&cache_dir, "risc0_recursion_intel_accum");

    // RPATH for runtime
    let intel_lib = PathBuf::from("/opt/intel/oneapi/compiler/latest/lib");
    if intel_lib.exists() {
        println!("cargo:rustc-link-search=native={}", intel_lib.display());
    }
}

const RECURSION_ACCUM_STAMP_VERSION: &str = "recursion-accum-v1";

/// GPU accum kernel: kernels/intel/accum.cpp with the generated
/// step_compute_accum.cpp / step_verify_accum.cpp spliced in (their
/// extern.h/ffi.h includes dropped; accum.cpp defines the device-side
/// extern_plonk{Read,Write}Accum_wom). Separate library and stamp so that it
/// does not invalidate the eval_check build. -O2 is bit-exact here (checked
/// against the CPU accum with RISC0_RECURSION_GPU_ACCUM=check).
fn build_intel_accum(
    icpx: &Path,
    icpx_version: &str,
    cxx_root: &str,
    cache_dir: &Path,
    out_dir: &Path,
) {
    let so_path = risc0_build_kernel::intel_library_path(cache_dir, "risc0_recursion_intel_accum");
    let stamp_path = cache_dir.join("intel_accum.stamp");
    let mut pairs: Vec<(PathBuf, PathBuf)> = [
        "kernels/intel/accum.cpp",
        "kernels/cxx/step_compute_accum.cpp",
        "kernels/cxx/step_verify_accum.cpp",
        "build.rs",
    ]
    .iter()
    .map(|p| (PathBuf::from(p), PathBuf::from(p)))
    .collect();
    pairs.extend(cxx_root_headers(cxx_root));
    let targets = risc0_build_kernel::intel_sycl_targets_arg();
    let hash = stamp::hash_labeled(
        &pairs,
        &[icpx_version, RECURSION_ACCUM_STAMP_VERSION, &targets],
    );
    if !stamp::need_rebuild(&so_path, &stamp_path, &hash) {
        eprintln!("Using cached Intel recursion accum kernel");
        return;
    }
    eprintln!("Building Intel SYCL accum kernel for recursion...");
    let marker = "// @RECURSION_ACCUM_STEPS@";
    let wrapper = std::fs::read_to_string("kernels/intel/accum.cpp").unwrap();
    assert!(
        wrapper.contains(marker),
        "accum.cpp lost its {marker} marker"
    );
    let mut steps = String::new();
    for f in [
        "kernels/cxx/step_compute_accum.cpp",
        "kernels/cxx/step_verify_accum.cpp",
    ] {
        // CRLF in a Windows checkout: match the includes on LF.
        let src = std::fs::read_to_string(f).unwrap().replace("\r\n", "\n");
        for inc in ["#include \"extern.h\"\n", "#include \"ffi.h\"\n"] {
            assert!(src.contains(inc), "{f}: expected {inc:?}");
        }
        steps.push_str(
            &src.replace("#include \"extern.h\"\n", "")
                .replace("#include \"ffi.h\"\n", ""),
        );
        steps.push('\n');
    }
    let amalg_path = out_dir.join("intel_recursion_accum_amalg.cpp");
    std::fs::write(&amalg_path, wrapper.replace(marker, &steps)).unwrap();
    // Collects the arguments for intel_build_library (not run itself).
    let mut cmd = Command::new(icpx);
    cmd.arg("-fsycl")
        .arg("-std=c++17")
        .arg("-O2")
        .arg("-Wno-unused-parameter")
        .arg("-Wno-unused-function")
        .arg("-Wno-unused-variable")
        .arg("-Wno-sign-compare")
        .arg(format!("-I{cxx_root}"))
        .arg("-Ikernels/cxx")
        .arg(risc0_build_kernel::intel_sycl_targets_arg());
    let args: Vec<std::ffi::OsString> = cmd.get_args().map(|a| a.to_owned()).collect();
    let output = risc0_build_kernel::intel_build_library(
        icpx,
        &args,
        &[amalg_path],
        cache_dir,
        "risc0_recursion_intel_accum",
    );
    if !output.status.success() {
        stamp::write(&stamp_path, &hash, icpx_version, "failed");
        panic!(
            "Intel recursion accum compilation failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    stamp::write(&stamp_path, &hash, icpx_version, "ok");
    eprintln!("  Built {}", so_path.display());
}

fn rerun_if_changed<P: AsRef<Path>>(path: P) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}

fn glob_paths(pattern: &str) -> Vec<PathBuf> {
    glob::glob(pattern).unwrap().map(|x| x.unwrap()).collect()
}
