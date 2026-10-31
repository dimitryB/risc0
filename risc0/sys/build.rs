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
        .flag(&format!(
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
        .compile("risc0_zkp_cuda");

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

    // Find icpx compiler
    let icpx = which_icpx();

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
    let lib_path = out_dir.join("librisc0_intel_kernels.so");

    // Compile with icpx
    let mut cmd = std::process::Command::new(&icpx);
    cmd.arg("-fsycl")
        .arg("-shared")
        .arg("-fPIC")
        .arg("-std=c++17")
        .arg("-O3")
        // ESIMD stateless mem: forces A64 addressing for correct large-buffer access.
        // NOTE: -fsycl-device-code-split=off and -flto are used in benchmarks but cause
        // massive JIT regression (~200s) when combined without AOT. Keep them out until
        // AOT (-fsycl-targets=intel_gpu_bmg_g31) is added.
        .arg("-fsycl-esimd-force-stateless-mem")
        .arg("-Wno-unused-parameter")
        .arg("-Wno-unused-function")
        .arg("-Wno-unused-variable")
        .arg("-o")
        .arg(&lib_path)
        .arg(format!("-I{}", intel_dir.display()));

    for src in &sources {
        cmd.arg(src);
    }

    let status = cmd
        .status()
        .expect("Failed to run icpx. Is oneAPI installed?");
    if !status.success() {
        panic!("icpx compilation failed for Intel ESIMD kernels");
    }

    // Tell cargo to link against the compiled library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=risc0_intel_kernels");

    // Also link the SYCL runtime
    if let Ok(oneapi_root) = env::var("ONEAPI_ROOT") {
        let lib_dir = PathBuf::from(&oneapi_root).join("lib");
        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
    }
    // Link sycl runtime (libsycl.so)
    println!("cargo:rustc-link-lib=dylib=sycl");
}

fn which_icpx() -> PathBuf {
    // Check RISC0_ICPX env var first, then PATH
    if let Ok(icpx) = env::var("RISC0_ICPX") {
        return PathBuf::from(icpx);
    }
    // Try to find icpx in PATH
    if let Ok(output) = std::process::Command::new("which").arg("icpx").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return PathBuf::from(path);
            }
        }
    }
    // Default: assume it's in PATH
    PathBuf::from("icpx")
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
