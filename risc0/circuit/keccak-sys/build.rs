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

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use risc0_build_kernel::{KernelBuild, KernelType};

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("Features 'cuda' and 'rocm' are mutually exclusive. Enable only one GPU backend.");

fn main() {
    let cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let rocm = env::var("CARGO_FEATURE_ROCM").is_ok();

    // Set the GPU builds' environment before any thread starts: the CPU build below runs
    // concurrently and reads the environment.
    if (cuda || rocm) && env::var("RISC0_SKIP_BUILD_KERNELS").is_err() {
        env::set_var("SCCACHE_IDLE_TIMEOUT", "0");
        if rocm {
            env::set_var("HIP_PLATFORM", "amd");
        }
    }

    // The CPU kernels (rust_poly_fp_<n>.cpp, ~7 MB of generated C++ each, up to ~20 min of -O3
    // per file) and the GPU kernels share nothing, so build them concurrently instead of one
    // after the other: each takes about as long as the other.
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
    // rust_poly_fp_<n>.cpp, the generated constraint polynomial, are ~7 MB each and take far
    // longer to compile than everything else here: start them at once, each on its own thread.
    let (poly_fp, rest): (Vec<PathBuf>, Vec<PathBuf>) =
        glob_paths("kernels/cxx/*.cpp").into_iter().partition(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("rust_poly_fp_"))
        });
    KernelBuild::new(KernelType::Cpp)
        .files(rest)
        .files_on_own_threads(poly_fp)
        .include(env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap())
        .compile("risc0_keccak_cpu");
}

fn build_cuda_kernels() {
    let output = "risc0_keccak_cuda";

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
    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    rerun_if_include_roots_changed(&[&cuda_root, &sppark_root]);
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    // `absolute`, not `canonicalize`: on Windows the latter gives a `\\?\` path, which cl.exe
    // (nvcc's host compiler there) can't search for the scratch TU's includes.
    let kernel_dir = std::path::absolute("kernels/cuda").unwrap();
    let msvc = risc0_build_kernel::is_msvc_target();
    let use_native_arch =
        env::var_os("NVCC_PREPEND_FLAGS").is_none() && env::var_os("NVCC_APPEND_FLAGS").is_none();

    // Step 1: eval_check as one translation unit compiled WITHOUT --device-c, so the generated
    // constraint chain inlines, and with its scratch array in global memory instead of on the
    // stack. See kernels/cuda/eval_check_scratch_kernel.cuh for why.
    let tu = out_dir.join("eval_check_scratch.cu");
    write_scratch_eval_check(&kernel_dir, &tu);
    let obj_ext = risc0_build_kernel::object_extension();
    let eval_check_obj = out_dir.join(format!("eval_check_scratch.{obj_ext}"));
    // A stable cache dir (target/<profile>/) rather than OUT_DIR, which changes with the crate's
    // fingerprint: compiling this translation unit takes minutes.
    let cache_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with("release") || p.ends_with("debug"))
        .map(|p| p.join("keccak_eval_check_cache"))
        .unwrap_or_else(|| out_dir.join("keccak_eval_check_cache"));
    std::fs::create_dir_all(&cache_dir).unwrap();

    let mut cmd = risc0_build_kernel::nvcc_command();
    cmd.current_dir(&kernel_dir);
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
        .arg(&kernel_dir)
        .arg("-I")
        .arg(&cuda_root)
        .arg("-I")
        .arg(&sppark_root);
    if use_native_arch {
        cmd.arg("-arch=native");
    }
    cmd.arg("-c") // compile only, NO --device-c
        .arg(&tu);

    // Key the cache on everything the compile can read (the generated translation unit, the
    // headers it includes, the include roots), the whole command and nvcc's environment, as
    // rv32im-sys does for its eval_check.
    let native_key = if use_native_arch {
        risc0_build_kernel::native_gpu_arch_key(false)
    } else {
        String::new()
    };
    let local_sources: Vec<PathBuf> = vec![
        tu.clone(),
        kernel_dir.join("eval_check_scratch_prelude.cuh"),
        kernel_dir.join("eval_check_scratch_kernel.cuh"),
    ];
    let local_sources: Vec<&Path> = local_sources.iter().map(PathBuf::as_path).collect();
    let sppark = Path::new(&sppark_root);
    let sources_key = risc0_build_kernel::kernel_sources_key(
        &[Path::new(&cuda_root), sppark],
        &local_sources,
        |path| {
            path.starts_with(sppark)
                || !matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("c" | "cc" | "cpp" | "cu")
                )
        },
    );
    let key = risc0_build_kernel::hash_key(&[
        "keccak-cuda-eval-check-scratch-v1",
        &sources_key,
        &risc0_build_kernel::command_key(&cmd),
        &risc0_build_kernel::nvcc_environment_key(),
        &native_key,
    ]);
    // eval_check is a single ptxas run of ~25 min, the -dc compile of the step files below
    // about 8: run it on its own thread, concurrently with them, instead of before them.
    std::thread::scope(|scope| {
        scope.spawn(|| {
            risc0_build_kernel::cached_compile(
                &cache_dir,
                "eval_check_scratch",
                obj_ext,
                &key,
                &eval_check_obj,
                |out| {
                    let status = cmd
                        .arg("-o")
                        .arg(out)
                        .status()
                        .expect("failed to run nvcc for eval_check_scratch.cu");
                    assert!(status.success(), "nvcc failed for eval_check_scratch.cu");
                },
            );
        });

        // Step 2: everything else with --device-c, as before. eval_check_*.cu and ffi_supra.cu
        // are in the translation unit above (ROCm still builds ffi_supra.cu).
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .cudart("static")
            .debug(false)
            .flag("-diag-suppress=177")
            .flag("-diag-suppress=550")
            .flag("-diag-suppress=2922")
            .flag(risc0_build_kernel::cuda_cxx_std_flag())
            .include(&cuda_root)
            .include(&sppark_root);
        // GCC/Clang warning flags; cl.exe, nvcc's host compiler on MSVC, rejects them.
        if msvc {
            risc0_build_kernel::msvc_cuda_build_setup(&mut build);
        } else {
            build
                .flag("-Xcompiler")
                .flag("-Wno-unused-function,-Wno-unused-parameter");
        }
        if use_native_arch {
            build.flag("-arch=native");
        }
        let cuda_files: Vec<PathBuf> = glob_paths("kernels/cuda/*.cu")
            .into_iter()
            .filter(|p| {
                let name = p.file_name().unwrap().to_str().unwrap();
                !(name == "ffi_supra.cu" || name.starts_with("eval_check_"))
            })
            .collect();
        build.files(cuda_files).compile(output);
    });

    // Step 3: add the eval_check object to the archive.
    risc0_build_kernel::add_to_static_lib(&out_dir, output, &[eval_check_obj]);
}

/// Write the CUDA eval_check translation unit to `dest`: the generated constraint code
/// (eval_check.cuh, then eval_check_<n>.cu) between eval_check_scratch_prelude.cuh and
/// eval_check_scratch_kernel.cuh, with its scratch array moved to global memory. Every writable
/// `Fp* argN` parameter becomes a `ScratchRef argN`, and `poly_fp`'s `Fp xN[M];` becomes this
/// thread's slice of the launch's scratch buffer, `M` elements long (`kKeccakScratchElems`).
/// The header's `extern` declaration of `poly_mix` becomes its definition.
///
/// Panics if the generated code doesn't have that shape (one array, and no writable `Fp*` but
/// the parameters that carry it), rather than build something that means something else.
fn write_scratch_eval_check(kernel_dir: &Path, dest: &Path) {
    let mut bodies: Vec<(u32, PathBuf)> = glob_paths("kernels/cuda/eval_check_*.cu")
        .into_iter()
        .filter_map(|p| {
            let name = p.file_name()?.to_str()?;
            let n = name
                .strip_prefix("eval_check_")?
                .strip_suffix(".cu")?
                .parse()
                .ok()?;
            Some((n, p))
        })
        .collect();
    bodies.sort();
    assert!(!bodies.is_empty(), "no kernels/cuda/eval_check_<n>.cu");
    let mut files = vec![kernel_dir.join("eval_check.cuh")];
    files.extend(bodies.into_iter().map(|(_, p)| p));

    let mut tu = String::from(
        "// Generated by keccak-sys/build.rs (`write_scratch_eval_check`) from kernels/cuda/\n\
         // eval_check.cuh and eval_check_<n>.cu. Do not edit.\n\
         #include \"eval_check_scratch_prelude.cuh\"\n",
    );
    let mut array: Option<(String, u64)> = None;
    let mut params = 0;
    let mut poly_mix = false;
    for file in &files {
        let text = std::fs::read_to_string(file).unwrap();
        let name = file.display();
        assert!(
            !text.contains("Fp *"),
            "{name}: unexpected `Fp *` spelling in the generated eval_check code"
        );
        tu.push_str(&format!(
            "// --- {} ---\n",
            file.file_name().unwrap().to_string_lossy()
        ));
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed == "#pragma once" || trimmed == "#include \"eval_check.cuh\"" {
                continue;
            }
            // The header declares `poly_mix` for --device-c builds; a whole-program build
            // defines it here, before its first use, and not again.
            if trimmed == "extern __constant__ FpExt poly_mix[kNumPolyMixPows];" {
                assert!(!poly_mix, "{name}: poly_mix declared twice");
                poly_mix = true;
                tu.push_str("__constant__ FpExt poly_mix[kNumPolyMixPows];\n");
                continue;
            }
            // `Fp x13[8237];`: the scratch array.
            let array_decl = trimmed
                .strip_prefix("Fp x")
                .and_then(|d| d.strip_suffix("];"))
                .and_then(|d| d.split_once('['))
                .filter(|(var, len)| {
                    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
                    digits(var) && digits(len)
                });
            if let Some((var, len)) = array_decl {
                assert!(
                    array.is_none(),
                    "{name}: more than one Fp array in eval_check"
                );
                array = Some((format!("x{var}"), len.parse().expect("array length")));
                tu.push_str(&format!(
                    "  ScratchRef x{var} = scratch_for_this_thread();\n"
                ));
                continue;
            }
            let mut out = String::with_capacity(line.len());
            let mut rest = line;
            while let Some(i) = rest.find("Fp*") {
                let (before, after) = rest.split_at(i);
                out.push_str(before);
                let ident_before = before
                    .chars()
                    .last()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
                if ident_before || out.ends_with("const ") {
                    out.push_str("Fp*");
                } else {
                    assert!(
                        after.starts_with("Fp* arg"),
                        "{name}: writable `Fp*` that isn't a scratch parameter: {line}"
                    );
                    out.push_str("ScratchRef");
                    params += 1;
                }
                rest = &after["Fp*".len()..];
            }
            out.push_str(rest);
            tu.push_str(&out);
            tu.push('\n');
        }
    }
    let (_, len) = array.expect("no Fp array in the generated eval_check code");
    assert!(
        params > 0,
        "no scratch parameters in the generated eval_check code"
    );
    assert!(
        poly_mix,
        "no poly_mix declaration in the generated eval_check code"
    );
    tu.push_str(&format!(
        "namespace risc0::circuit::keccak::cuda {{\n\
         constexpr uint32_t kKeccakScratchElems = {len};\n\
         }}\n\
         #include \"eval_check_scratch_kernel.cuh\"\n"
    ));
    if std::fs::read_to_string(dest).ok().as_deref() != Some(tu.as_str()) {
        std::fs::write(dest, tu).unwrap();
    }
}

/// The Intel (SYCL) eval_check: one translation unit, from the CUDA sources (see
/// `write_intel_translation_unit`), built into the `risc0_keccak_intel` shared library. (The
/// witness is generated on the CPU; see keccak's prove/hal/intel.rs.) Building it takes a long
/// time, so the library is kept in a stable directory under target/<profile>/, keyed on everything
/// the build reads.
fn build_intel_kernels() {
    const NAME: &str = "risc0_keccak_intel";
    rerun_if_changed("kernels/intel");
    rerun_if_changed("kernels/cuda");

    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    for header in ["fp.h", "fpext.h"] {
        println!("cargo:rerun-if-changed={cxx_root}/{header}");
    }
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let cuda_dir = std::fs::canonicalize("kernels/cuda").unwrap();
    let intel_dir = std::fs::canonicalize("kernels/intel").unwrap();

    let tu = out_dir.join("keccak_intel.cpp");
    write_intel_translation_unit(&cuda_dir, &tu);

    let icpx = risc0_build_kernel::find_icpx();
    let include = |dir: &Path| format!("-I{}", dir.display());
    // Compiled ahead of time for BMG, as rv32im-sys compiles its eval_check: compiled by the GPU
    // driver when a process first uses them instead, these kernels take it tens of minutes. With
    // the generated functions kept as calls (see prelude.h), -Os and 256 GRF, as there.
    let args: Vec<std::ffi::OsString> = vec![
        "-fsycl".into(),
        "-std=c++17".into(),
        "-Os".into(),
        "-w".into(),
        "-fsycl-device-code-split=per_kernel".into(),
        // The 32-bit-only field arithmetic in fp.h/fpext.h (see there).
        "-DRISC0_INTEL_EVAL_FAST".into(),
        include(&intel_dir).into(),
        include(&cuda_dir).into(),
        include(Path::new(&cxx_root)).into(),
        risc0_build_kernel::intel_sycl_targets_arg().into(),
        "-Xs".into(),
        "-options \"-cl-intel-256-GRF-per-thread -igc_opts 'DisableRecompilation=1'\"".into(),
    ];

    let icpx_version = Command::new(&icpx)
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    let read = |p: &Path| std::fs::read_to_string(p).unwrap_or_default();
    let mut parts = vec![
        "keccak-intel-v1".to_string(),
        icpx.display().to_string(),
        icpx_version,
        read(&tu),
        read(&intel_dir.join("prelude.h")),
        read(&intel_dir.join("kernels.h")),
        read(&Path::new(&cxx_root).join("fp.h")),
        read(&Path::new(&cxx_root).join("fpext.h")),
    ];
    parts.extend(args.iter().map(|a| a.to_string_lossy().into_owned()));
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
    let key = risc0_build_kernel::hash_key(&parts);

    let cache_root = out_dir
        .ancestors()
        .find(|p| p.ends_with("release") || p.ends_with("debug"))
        .map(|p| p.join("keccak_intel_cache"))
        .unwrap_or_else(|| out_dir.join("keccak_intel_cache"));
    let lib_dir = cache_root.join(&key[..16]);
    let lib = risc0_build_kernel::intel_library_path(&lib_dir, NAME);
    if !lib.is_file() {
        eprintln!("Building the Intel keccak eval_check (takes a long time)...");
        let _ = std::fs::remove_dir_all(&lib_dir);
        std::fs::create_dir_all(&lib_dir).unwrap();
        let output = risc0_build_kernel::intel_build_library(
            &icpx,
            &args,
            std::slice::from_ref(&tu),
            &lib_dir,
            NAME,
        );
        if !output.status.success() || !lib.is_file() {
            let _ = std::fs::remove_dir_all(&lib_dir);
            panic!(
                "icpx failed for the Intel keccak kernels:\n{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    risc0_build_kernel::intel_link_library(&lib_dir, NAME);
    risc0_build_kernel::intel_link_sycl_runtime();
}

/// Write the Intel (SYCL) translation unit to `dest`: kernels/intel/prelude.h, the constraint code
/// (kernels/cuda/eval_check.cuh and eval_check_<n>.cu, rewritten as below), then
/// kernels/intel/kernels.h.
///
/// The constraint code, like CUDA's (see `write_scratch_eval_check`), gets its scratch array in
/// global memory: `poly_fp`'s `Fp xN[M];` becomes this work-item's `ScratchRef`, and each writable
/// `Fp* argN` that carries it a `ScratchRef argN`. SYCL device code has no `__constant__` globals,
/// so the polynomial mix powers and the scratch come in an `EvalCtx` that every function takes after
/// `size` and passes on to the functions it calls. The functions are then cut into stages, each its
/// own kernel (see `cut_into_stages`).
///
/// Panics if the generated code doesn't have that shape, rather than build something that means
/// something else.
fn write_intel_translation_unit(cuda_dir: &Path, dest: &Path) {
    let numbered = |prefix: &str| -> Vec<PathBuf> {
        let mut files: Vec<(u32, PathBuf)> = glob_paths(&format!("kernels/cuda/{prefix}*.cu"))
            .into_iter()
            .filter_map(|p| {
                let n = p
                    .file_name()?
                    .to_str()?
                    .strip_prefix(prefix)?
                    .strip_suffix(".cu")?
                    .parse()
                    .ok()?;
                Some((n, p))
            })
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no kernels/cuda/{prefix}<n>.cu");
        files.into_iter().map(|(_, p)| p).collect()
    };
    let keep = |trimmed: &str| trimmed != "#pragma once" && !trimmed.starts_with("#include");

    let mut tu = String::from(
        "// Generated by keccak-sys/build.rs (`write_intel_translation_unit`) from kernels/cuda/ and\n\
         // kernels/intel/. Do not edit.\n\
         #include \"prelude.h\"\n\
         // The generated functions stay calls (see prelude.h).\n\
         #undef __device__\n\
         #define __device__ R0_KECCAK_OUTLINE\n",
    );

    let mut checks = vec![cuda_dir.join("eval_check.cuh")];
    checks.extend(numbered("eval_check_"));
    let mut definitions = String::new();
    let mut array: Option<u64> = None;
    let mut params = 0;
    let mut signatures = 0;
    let mut poly_mix = false;
    for (i, file) in checks.iter().enumerate() {
        let text = std::fs::read_to_string(file).unwrap();
        let name = file.display();
        assert!(
            !text.contains("Fp *"),
            "{name}: unexpected `Fp *` spelling in the generated eval_check code"
        );
        // The header goes straight in; the definitions after it, once cut into stages.
        let tu = if i == 0 { &mut tu } else { &mut definitions };
        tu.push_str(&format!(
            "// --- {} ---\n",
            file.file_name().unwrap().to_string_lossy()
        ));
        for line in text.lines() {
            let trimmed = line.trim();
            if !keep(trimmed) {
                continue;
            }
            if trimmed == "extern __constant__ FpExt poly_mix[kNumPolyMixPows];" {
                assert!(!poly_mix, "{name}: poly_mix declared twice");
                poly_mix = true;
                continue;
            }
            // `Fp x13[8237];`: the scratch array.
            let array_decl = trimmed
                .strip_prefix("Fp x")
                .and_then(|d| d.strip_suffix("];"))
                .and_then(|d| d.split_once('['))
                .filter(|(var, len)| {
                    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
                    digits(var) && digits(len)
                });
            if let Some((var, len)) = array_decl {
                assert!(
                    array.is_none(),
                    "{name}: more than one Fp array in eval_check"
                );
                array = Some(len.parse().expect("array length"));
                tu.push_str(&format!("  ScratchRef x{var} = ectx.scratch;\n"));
                continue;
            }
            let mut out = String::with_capacity(line.len() + 16);
            let mut rest = line;
            while let Some(i) = rest.find("Fp*") {
                let (before, after) = rest.split_at(i);
                out.push_str(before);
                let ident_before = before
                    .chars()
                    .last()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
                if ident_before || out.ends_with("const ") {
                    out.push_str("Fp*");
                } else {
                    assert!(
                        after.starts_with("Fp* arg"),
                        "{name}: writable `Fp*` that isn't a scratch parameter: {line}"
                    );
                    out.push_str("ScratchRef");
                    params += 1;
                }
                rest = &after["Fp*".len()..];
            }
            out.push_str(rest);
            signatures += out.matches("uint32_t size,").count();
            let out = out
                .replace("uint32_t size,", "uint32_t size, EvalCtx ectx,")
                .replace("(idx, size,", "(idx, size, ectx,")
                .replace("poly_mix[", "ectx.poly_mix[");
            tu.push_str(&out);
            tu.push('\n');
        }
    }
    let len = array.expect("no Fp array in the generated eval_check code");
    assert!(
        params > 0,
        "no scratch parameters in the generated eval_check code"
    );
    assert!(
        poly_mix,
        "no poly_mix declaration in the generated eval_check code"
    );
    // A function left without `ectx`, or a call that doesn't pass it, fails to compile.
    assert!(
        signatures > 0,
        "no `uint32_t size,` parameters in the generated eval_check code"
    );
    let (definitions, heads, carry_words) = cut_into_stages(&definitions, eval_stages());
    tu.push_str(&definitions);
    // `eval_stage<S>` runs stage S: poly_fp's run, then each later run from its carry.
    tu.push_str(&format!(
        "namespace risc0::circuit::keccak::cuda {{\n\
         constexpr uint32_t kKeccakScratchElems = {len};\n\
         constexpr uint32_t kEvalStages = {stages};\n\
         constexpr uint32_t kCarryWords = {carry_words};\n\
         template <uint32_t S>\n\
         FpExt eval_stage(uint32_t idx, uint32_t size, EvalCtx ectx, const Fp* ctrl, const Fp* out,\n\
         \x20                const Fp* data, const Fp* mix, const Fp* accum);\n\
         template <>\n\
         inline FpExt eval_stage<0>(uint32_t idx, uint32_t size, EvalCtx ectx, const Fp* ctrl,\n\
         \x20                          const Fp* out, const Fp* data, const Fp* mix, const Fp* accum) {{\n\
         \x20 return poly_fp(idx, size, ectx, ctrl, out, data, mix, accum);\n\
         }}\n",
        stages = heads.len() + 1,
    ));
    for (s, head) in heads.iter().enumerate() {
        tu.push_str(&format!(
            "template <>\n\
             inline FpExt eval_stage<{stage}>(uint32_t idx, uint32_t size, EvalCtx ectx, const Fp*,\n\
             \x20                          const Fp*, const Fp*, const Fp*, const Fp*) {{\n\
             \x20 return call_from_carry<{head}>(idx, size, ectx);\n\
             }}\n",
            stage = s + 1,
        ));
    }
    tu.push_str("} // namespace risc0::circuit::keccak::cuda\n#include \"kernels.h\"\n");
    if std::fs::read_to_string(dest).ok().as_deref() != Some(tu.as_str()) {
        std::fs::write(dest, tu).unwrap();
    }
}

/// How many kernels the Intel eval_check is cut into (see `cut_into_stages`).
fn eval_stages() -> usize {
    println!("cargo:rerun-if-env-changed=RISC0_KECCAK_INTEL_EVAL_STAGES");
    env::var("RISC0_KECCAK_INTEL_EVAL_STAGES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(8)
}

/// Cut the constraint code's chain of tail calls (`poly_fp` -> `keccak_56` -> ... -> `keccak_0`,
/// each function's last act being `auto xN = <next>(idx, size, ectx, ...); return xN;`) into
/// `stages` runs of about equal length. Returns the definitions with the call into each later run
/// made a `carry_out` (see kernels/intel/prelude.h), the functions that start those runs, and how
/// many words of carry the largest cut needs.
///
/// Panics if the definitions aren't such a chain.
fn cut_into_stages(definitions: &str, stages: usize) -> (String, Vec<String>, usize) {
    struct Function {
        name: String,
        lines: usize,
        callee: Option<String>,
    }
    let lines: Vec<&str> = definitions.lines().collect();
    let mut functions: Vec<Function> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        // `__device__ FpExt keccak_3(...`, or `__device__ FpExt` then `keccak_3(...`.
        let header = if let Some(rest) = lines[i].strip_prefix("__device__ FpExt ") {
            Some(rest)
        } else if lines[i] == "__device__ FpExt" && i + 1 < lines.len() {
            i += 1;
            Some(lines[i])
        } else {
            None
        };
        let Some(header) = header else {
            i += 1;
            continue;
        };
        let name = header.split('(').next().unwrap().trim().to_string();
        let start = i;
        while lines[i] != "}" {
            i += 1;
        }
        let callees: Vec<String> = lines[start..i]
            .iter()
            .filter_map(|l| {
                let (_, call) = l.split_once(" = ")?;
                let (callee, args) = call.split_once('(')?;
                (callee.starts_with("keccak_") && args.starts_with("idx, size, ectx,"))
                    .then(|| callee.to_string())
            })
            .collect();
        assert!(callees.len() <= 1, "eval_check: {name} calls {callees:?}");
        functions.push(Function {
            name,
            lines: i - start,
            callee: callees.into_iter().next(),
        });
        i += 1;
    }

    let mut chain: Vec<&Function> = Vec::new();
    let mut next = Some("poly_fp".to_string());
    while let Some(name) = next {
        let f = functions
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("eval_check: no definition of {name}"));
        chain.push(f);
        next = f.callee.clone();
    }
    assert_eq!(
        chain.len(),
        functions.len(),
        "eval_check: functions outside the chain from poly_fp"
    );

    let total: usize = chain.iter().map(|f| f.lines).sum();
    let mut heads = Vec::new();
    let mut done = 0;
    for f in &chain {
        done += f.lines;
        if heads.len() + 1 < stages && done >= total * (heads.len() + 1) / stages {
            if let Some(callee) = &f.callee {
                heads.push(callee.clone());
            }
        }
    }

    let mut out = definitions.to_string();
    let mut carry_words = 1;
    for head in &heads {
        let call = format!("= {head}(idx, size, ectx,");
        assert_eq!(out.matches(&call).count(), 1, "eval_check: calls to {head}");
        let at = out.find(&call).unwrap() + call.len();
        let args = out[at..].split(");").next().unwrap().split(',').count();
        // At most four words an argument: an FpExt (a pointer is two).
        carry_words = carry_words.max(4 * args);
        out = out.replace(&call, "= carry_out(idx, size, ectx,");
    }
    (out, heads, carry_words)
}

fn build_rocm_kernels() {
    let output = "risc0_keccak_cuda";

    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=RISC0_HIP_ARCH");
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

    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    // The headers these kernels include (see the CUDA build).
    rerun_if_include_roots_changed(&[&sppark_root, &cuda_root]);
    let hipcc = risc0_build_kernel::find_hipcc();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let kernel_dir = std::fs::canonicalize("kernels/cuda").unwrap();

    // Two-group single-TU approach:
    // Group A (risc0 types): ffi.cu + step_*.cu + empty.cu
    // Group B (sppark types): ffi_supra.cu + eval_check_*.cu
    // These can't share a TU because sppark's bb31_4_t conflicts with risc0's fpext.h.
    // Within each group, files share a TU so cross-TU __device__ calls resolve without -fgpu-rdc.
    let cu_files = glob_paths("kernels/cuda/*.cu");
    let mut risc0_amalg = String::new();
    let mut supra_amalg = String::new();
    for cu in &cu_files {
        let name = cu.file_name().unwrap().to_str().unwrap();
        let abs = std::fs::canonicalize(cu).unwrap();
        let line = format!("#include \"{}\"\n", abs.display());
        if name == "ffi_supra.cu" || name.starts_with("eval_check") {
            supra_amalg.push_str(&line);
        } else {
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
        "-O2",
        "-Wno-unused-function",
        "-Wno-unused-parameter",
        "-Wno-missing-braces",
        "-mllvm",
        "-amdgpu-early-inline-all=false",
        "-include",
        &include_cuda2hip,
        "-I",
        &cuda_root,
        "-I",
        &sppark_root,
        "-I",
        kernel_dir_str,
    ];
    base_flags.extend(risc0_build_kernel::hip_host_flags());
    let obj_ext = risc0_build_kernel::object_extension();

    let mut obj_files = Vec::new();
    for (amalg_path, obj_name) in [
        (&risc0_amalg_path, "risc0_kernels_rocm"),
        (&supra_amalg_path, "supra_kernels_rocm"),
    ] {
        let obj = out_dir.join(format!("{obj_name}.{obj_ext}"));
        risc0_build_kernel::hip_compile(&hipcc, &base_flags, amalg_path, &obj, &arches, None);
        obj_files.push(obj);
    }

    // Archive
    let _ = std::fs::remove_file(out_dir.join(format!("lib{output}.a")));
    let _ = std::fs::remove_file(out_dir.join(format!("{output}.lib")));
    risc0_build_kernel::add_to_static_lib(&out_dir, output, &obj_files);

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

/// The include roots the CUDA build compiles against: re-run this script when a file in them
/// changes (as a path dependency), so the cache key is recomputed. File by file, as a directory
/// would make cargo recurse into sppark's `rust/sppark` symlink loop.
fn rerun_if_include_roots_changed(roots: &[&str]) {
    for root in roots {
        for file in risc0_build_kernel::kernel_source_files(Path::new(root)) {
            rerun_if_changed(file);
        }
    }
}
