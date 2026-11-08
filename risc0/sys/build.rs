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

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("Features 'cuda' and 'rocm' are mutually exclusive. Enable only one GPU backend.");

#[cfg(all(feature = "cuda", feature = "intel"))]
compile_error!("Features 'cuda' and 'intel' are mutually exclusive. Enable only one GPU backend.");

#[cfg(all(feature = "rocm", feature = "intel"))]
compile_error!("Features 'rocm' and 'intel' are mutually exclusive. Enable only one GPU backend.");

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cxx_root = manifest_dir.join("cxx");
    println!("cargo:cxx_root={}", cxx_root.to_string_lossy());

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!(
            "cargo:cuda_root={}",
            manifest_dir.join("kernels/zkp/cuda").to_string_lossy()
        );
        build_cuda_kernels(&cxx_root);
    }

    if env::var("CARGO_FEATURE_ROCM").is_ok() {
        println!(
            "cargo:cuda_root={}",
            manifest_dir.join("kernels/zkp/cuda").to_string_lossy()
        );
        build_rocm_kernels(&cxx_root);
    }

    if env::var("CARGO_FEATURE_INTEL").is_ok() {
        build_intel_kernels();
    }

    // CPU proving on Apple platforms must not require the Metal compiler.
    if env::var("CARGO_FEATURE_METAL").is_ok()
        && env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos" || os == "ios")
    {
        println!(
            "cargo:metal_root={}",
            manifest_dir.join("kernels/zkp/metal").to_string_lossy()
        );
        if env::var_os("CARGO_FEATURE_METAL_RUNTIME").is_some() {
            embed_metal_sources();
        } else {
            build_metal_kernels();
        }
    }
}

fn build_cuda_kernels(cxx_root: &Path) {
    KernelBuild::new(KernelType::Cuda)
        .files([
            "kernels/zkp/cuda/combos.cu",
            "kernels/zkp/cuda/eltwise.cu",
            "kernels/zkp/cuda/ffi.cu",
            "kernels/zkp/cuda/kernels.cu",
            "kernels/zkp/cuda/sha.cu",
            "kernels/zkp/cuda/supra/api.cu",
            "kernels/zkp/cuda/supra/ntt.cu",
        ])
        .deps(["kernels/zkp/cuda", "kernels/zkp/cuda/supra"])
        .flag("-DFEATURE_BABY_BEAR")
        .include(cxx_root)
        .include(env::var("DEP_BLST_C_SRC").unwrap())
        .include(env::var("DEP_SPPARK_ROOT").unwrap())
        .compile("risc0_zkp_cuda");
}

fn build_rocm_kernels(cxx_root: &Path) {
    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    println!("cargo:rerun-if-env-changed=RISC0_HIP_ARCH");
    // `cc` tracks no sources itself, and the line above turns off cargo's default of re-running
    // on any change in the package: track the kernels and the headers they include. File by
    // file for sppark, whose `rust/sppark` symlink loop a directory would make cargo recurse.
    println!("cargo:rerun-if-changed=kernels/zkp/cuda");
    println!("cargo:rerun-if-changed={}", cxx_root.display());
    for file in risc0_build_kernel::kernel_source_files(Path::new(&sppark_root)) {
        println!("cargo:rerun-if-changed={}", file.display());
    }

    env::set_var("HIP_PLATFORM", "amd");

    let hipcc = risc0_build_kernel::find_hipcc();

    let mut build = cc::Build::new();
    build
        .compiler(&hipcc)
        .cpp(true)
        .debug(false)
        .flag("-x")
        .flag("hip")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-Wno-unused-function")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-missing-braces")
        .flag(format!(
            "--offload-arch={}",
            std::env::var("RISC0_HIP_ARCH").unwrap_or_else(|_| "native".to_string())
        ))
        .flag("-mllvm")
        .flag("-amdgpu-early-inline-all=false")
        .flag("-include")
        .flag("util/cuda2hip.hpp")
        .flag("-DFEATURE_BABY_BEAR")
        .include(cxx_root)
        .include(env::var("DEP_BLST_C_SRC").unwrap())
        .include(&sppark_root)
        .files([
            "kernels/zkp/cuda/combos.cu",
            "kernels/zkp/cuda/eltwise.cu",
            "kernels/zkp/cuda/ffi.cu",
            "kernels/zkp/cuda/kernels.cu",
            "kernels/zkp/cuda/sha.cu",
            "kernels/zkp/cuda/supra/api.cu",
            "kernels/zkp/cuda/supra/ntt.cu",
        ])
        .flags(risc0_build_kernel::hip_host_flags())
        .compile("risc0_zkp_cuda");

    // pv.cuh's Poseidon2 kernels in their own TU, with the ILP-first scheduler (see supra/pv_hip.cu).
    cc::Build::new()
        .compiler(&hipcc)
        .cpp(true)
        .debug(false)
        .flag("-x")
        .flag("hip")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-Wno-unused-function")
        .flag("-Wno-unused-parameter")
        .flag(format!(
            "--offload-arch={}",
            std::env::var("RISC0_HIP_ARCH").unwrap_or_else(|_| "native".to_string())
        ))
        .flag("-mllvm")
        .flag("-amdgpu-early-inline-all=false")
        .flag("-mllvm")
        .flag("-amdgpu-sched-strategy=max-ilp")
        .include(cxx_root)
        .file("kernels/zkp/cuda/supra/pv_hip.cu")
        .flags(risc0_build_kernel::hip_host_flags())
        .compile("risc0_zkp_cuda_pv");

    // Link against HIP runtime
    risc0_build_kernel::emit_rocm_lib_link();
}

fn build_intel_kernels() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let intel_dir = manifest_dir.join("kernels/zkp/intel");

    println!("cargo:rerun-if-changed={}", intel_dir.display());

    // The OpenCL sources are embedded as raw-string .inc files (JIT-compiled
    // at runtime); fail the build if an .inc no longer matches its .cl.
    for name in ["poseidon2_gate0", "poseidon2_commit"] {
        let cl = std::fs::read_to_string(intel_dir.join(format!("{name}.cl"))).unwrap();
        let inc = std::fs::read_to_string(intel_dir.join(format!("{name}_cl.inc"))).unwrap();
        let start = inc.find("R\"OPENCL(").map(|i| i + "R\"OPENCL(".len());
        let end = inc.rfind(")OPENCL\"");
        let embedded = match (start, end) {
            (Some(s), Some(e)) if s <= e => &inc[s..e],
            _ => panic!("{name}_cl.inc: no R\"OPENCL(...)OPENCL\" literal"),
        };
        if embedded.trim() != cl.trim() {
            panic!("{name}_cl.inc is out of date with {name}.cl; regenerate it");
        }
    }

    let icpx = risc0_build_kernel::find_icpx();

    // The ESIMD kernel sources. These are compiled into a single shared library
    // that provides the extern "C" FFI functions called from intel.rs.
    let sources = [
        intel_dir.join("intel_ffi.cpp"),
        intel_dir.join("ntt_kernel.cpp"),
        intel_dir.join("poseidon2.cpp"),
        intel_dir.join("poseidon254.cpp"),
        intel_dir.join("eltwise_ops.cpp"),
        intel_dir.join("fri_poly_ops.cpp"),
    ];

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // poseidon2.cpp's OpenCL kernels, compiled ahead of time for the build's Intel targets.
    poseidon2_ahead_of_time(&intel_dir, &out_dir);
    // The library's own SYCL/ESIMD kernels stay SPIR-V, which the driver compiles on the card at
    // the first proof: about 75 s on an Arc B580, 64 s of it the optimized poseidon254 image. The
    // runtime builds an ESIMD image with -vc-codegen (and -doubleGRF for grf_size<256>), from the
    // image's properties; icpx 2026.0 runs ocloc for an ahead-of-time image with no options, and
    // IGC 2.28.4 crashes on the poseidon254 images that way (SIGSEGV in its VLD path). One set of
    // options for every image would not do either: the library also has SPMD images, and two of
    // its ESIMD images need 256 GRF while the rest must not get it.
    let args: Vec<std::ffi::OsString> = [
        "-fsycl",
        "-std=c++17",
        "-O3",
        // ESIMD stateless mem: forces A64 addressing for correct large-buffer access.
        // NOTE: -fsycl-device-code-split=off and -flto are used in benchmarks but cause
        // massive JIT regression (~200s) when combined without AOT. Keep them out until
        // AOT (-fsycl-targets=intel_gpu_bmg_g31) is added.
        "-fsycl-esimd-force-stateless-mem",
        "-Wno-unused-parameter",
        "-Wno-unused-function",
        "-Wno-unused-variable",
    ]
    .iter()
    .map(Into::into)
    .chain([
        format!("-I{}", intel_dir.display()).into(),
        // poseidon2_aot.inc
        format!("-I{}", out_dir.display()).into(),
    ])
    .collect();

    let output = risc0_build_kernel::intel_build_library(
        &icpx,
        &args,
        &sources,
        &out_dir,
        "risc0_intel_kernels",
    );
    if !output.status.success() {
        panic!(
            "icpx compilation failed for Intel ESIMD kernels:
{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    risc0_build_kernel::intel_link_library(&out_dir, "risc0_intel_kernels");
    risc0_build_kernel::intel_link_sycl_runtime();
}

/// The Intel targets the build compiles ahead of time, as ocloc names them: `bmg_g21` for
/// `intel_gpu_bmg_g21` in `RISC0_INTEL_SYCL_TARGETS` (risc0_build_kernel::intel_sycl_targets_arg).
fn intel_aot_devices() -> Vec<String> {
    risc0_build_kernel::intel_sycl_targets_arg()
        .trim_start_matches("-fsycl-targets=")
        .split(',')
        .filter_map(|t| t.trim().strip_prefix("intel_gpu_"))
        // dg2_* are aliases of acm_*, the name poseidon2.cpp looks the device up by.
        .map(|d| d.replace("dg2_", "acm_"))
        .fold(Vec::new(), |mut v, d| {
            if !v.contains(&d) {
                v.push(d);
            }
            v
        })
}

/// Compile poseidon2.cpp's two OpenCL C kernel sources ahead of time, for every Intel target the
/// build compiles its SYCL kernels for, and write `poseidon2_aot.inc` into `out_dir`, which embeds
/// them for poseidon2.cpp to load natively. Without this the driver compiles them on the card at the
/// first proof (about 5 minutes on an Arc B580). Any other GPU still compiles the embedded source.
///
/// The options are those the run-time build uses (poseidon2.cpp, `ensure_poseidon2_opencl_kernels`),
/// with ocloc, which is also what the SYCL kernel compiler runs for OpenCL source, so the code is the
/// same: gate0 in 256-GRF mode except on Alchemist (whose 256-wide SIMD32 work-groups do not run in
/// that mode there, so it falls back to the default at run time too), commit in the default mode.
fn poseidon2_ahead_of_time(intel_dir: &Path, out_dir: &Path) {
    println!("cargo:rerun-if-env-changed=RISC0_OCLOC");
    let devices = intel_aot_devices();
    let aot_dir = out_dir.join("poseidon2_aot");
    let _ = std::fs::remove_dir_all(&aot_dir);
    std::fs::create_dir_all(&aot_dir).unwrap();
    let mut jobs = Vec::new();
    for device in &devices {
        let alchemist = device.starts_with("acm") || device.starts_with("dg2");
        let gate0_opts = if alchemist {
            "-cl-std=CL3.0 -cl-mad-enable"
        } else {
            "-cl-std=CL3.0 -cl-mad-enable -cl-intel-256-GRF-per-thread"
        };
        jobs.push((device.clone(), "gate0", "poseidon2_gate0.cl", gate0_opts));
        jobs.push((
            device.clone(),
            "commit",
            "poseidon2_commit.cl",
            "-cl-std=CL3.0 -cl-mad-enable",
        ));
    }
    let mut embedded = Vec::new();
    if !jobs.is_empty() {
        // The intel_gpu_* targets above need ocloc anyway (icpx runs it for them).
        let ocloc = env::var("RISC0_OCLOC").unwrap_or_else(|_| "ocloc".to_string());
        let results: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = jobs
                .iter()
                .map(|(device, source, file, opts)| {
                    let ocloc = &ocloc;
                    let aot_dir = &aot_dir;
                    scope.spawn(move || {
                        let name = format!("poseidon2_{source}_{device}");
                        let out = Command::new(ocloc)
                            .args(["compile", "-q", "-file"])
                            .arg(intel_dir.join(file))
                            .args(["-device", device, "-options", opts, "-output", &name])
                            .arg("-out_dir")
                            .arg(aot_dir)
                            .arg("-output_no_suffix")
                            .output()
                            .unwrap_or_else(|e| {
                                panic!("{ocloc} could not be run ({e}); it compiles the Intel AOT targets")
                            });
                        // ocloc names the binary <name>.bin.
                        let bin = aot_dir.join(format!("{name}.bin"));
                        if !out.status.success() || !bin.is_file() {
                            panic!(
                                "ocloc failed for {file} on {device}:\n{}{}",
                                String::from_utf8_lossy(&out.stdout),
                                String::from_utf8_lossy(&out.stderr)
                            );
                        }
                        (device.clone(), *source, bin)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        embedded = results;
    }
    let mut inc = String::from(
        "// Generated by risc0-sys build.rs (poseidon2_ahead_of_time): poseidon2.cpp's OpenCL kernels\n\
         // compiled ahead of time with ocloc, one native binary per Intel device and source.\n\
         #pragma once\n#include <cstddef>\n\
         struct Poseidon2Aot { const char* device; const char* source; const unsigned char* data; size_t size; };\n",
    );
    if embedded.is_empty() {
        inc.push_str("static const Poseidon2Aot* const POSEIDON2_AOT = nullptr;\nstatic const size_t POSEIDON2_AOT_COUNT = 0;\n");
    } else {
        inc.push_str("#if !defined(__SYCL_DEVICE_ONLY__)\n#pragma clang diagnostic push\n#pragma clang diagnostic ignored \"-Wc23-extensions\"\n");
        for (i, (_, _, bin)) in embedded.iter().enumerate() {
            let path = bin.to_string_lossy().replace('\\', "/");
            inc.push_str(&format!(
                "static const unsigned char poseidon2_aot_{i}[] = {{\n#embed \"{path}\"\n}};\n"
            ));
        }
        inc.push_str(
            "#pragma clang diagnostic pop\nstatic const Poseidon2Aot POSEIDON2_AOT[] = {\n",
        );
        for (i, (device, source, _)) in embedded.iter().enumerate() {
            inc.push_str(&format!(
                "    {{\"{device}\", \"{source}\", poseidon2_aot_{i}, sizeof(poseidon2_aot_{i})}},\n"
            ));
        }
        inc.push_str(&format!(
            "}};\nstatic const size_t POSEIDON2_AOT_COUNT = {};\n#else\n\
             static const Poseidon2Aot* const POSEIDON2_AOT = nullptr;\nstatic const size_t POSEIDON2_AOT_COUNT = 0;\n#endif\n",
            embedded.len()
        ));
    }
    std::fs::write(out_dir.join("poseidon2_aot.inc"), inc).unwrap();
}

fn build_metal_kernels() {
    const METAL_KERNELS: &[(&str, &[&str])] = &[(
        "zkp",
        &[
            "eltwise.metal",
            "fri.metal",
            "mix.metal",
            "ntt.metal",
            "poseidon2.metal",
            "poseidon254.metal",
            "sha.metal",
            "zk.metal",
        ],
    )];

    let inc_path = Path::new("kernels/zkp/metal");
    for (name, srcs) in METAL_KERNELS {
        let dir = Path::new("kernels").join(name).join("metal");
        let src_paths = srcs.iter().map(|x| dir.join(x));
        let out = format!("metal_kernels_{name}");
        KernelBuild::new(KernelType::Metal)
            .files(src_paths)
            .include(inc_path)
            .dep(inc_path.join("sha256.h"))
            .compile(&out);
    }
}

fn embed_metal_sources() {
    // Same source files and field headers as build_metal_kernels. This only
    // expands the known local includes; kernel arithmetic remains unchanged.
    // There is no implicit runtime fallback when an offline compiler fails.
    const SOURCES: &[&str] = &[
        "sha256.h",
        "eltwise.metal",
        "fri.metal",
        "mix.metal",
        "ntt.metal",
        "poseidon2.metal",
        "poseidon254.metal",
        "sha.metal",
        "zk.metal",
    ];
    fn append(source: &mut String, name: &str, contents: &str) {
        source.push_str(&format!("\n#line 1 \"{name}\"\n"));
        for line in contents.lines() {
            match line.trim() {
                "#pragma once"
                | "#include \"fp.h\""
                | "#include \"fpext.h\""
                | "#include \"sha256.h\"" => source.push('\n'),
                directive if directive.starts_with("#include \"") => {
                    panic!("unhandled runtime Metal include in {name}: {directive}");
                }
                _ => {
                    source.push_str(line);
                    source.push('\n');
                }
            }
        }
    }
    let mut source = String::new();
    for (name, contents) in risc0_build_kernel::metal_headers() {
        append(&mut source, name, contents);
    }
    for name in SOURCES {
        let path = Path::new("kernels/zkp/metal").join(name);
        println!("cargo:rerun-if-changed={}", path.display());
        append(&mut source, name, &std::fs::read_to_string(path).unwrap());
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("metal_kernels_zkp.metal");
    std::fs::write(&output, source).unwrap();
    println!("cargo:metal_source_zkp={}", output.display());
}
