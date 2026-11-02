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

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda_kernels();
    }

    if env::var("CARGO_FEATURE_ROCM").is_ok() {
        build_rocm_kernels();
    }

    if env::var("CARGO_FEATURE_INTEL").is_ok() {
        build_intel_kernels();
    }
}

fn build_intel_kernels() {
    println!("cargo:rerun-if-changed=kernels/intel");
    let icpx = risc0_build_kernel::find_icpx();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let args: Vec<std::ffi::OsString> = ["-fsycl", "-std=c++17", "-O3", "-Ikernels/intel"]
        .iter()
        .map(Into::into)
        .collect();
    let output = risc0_build_kernel::intel_build_library(
        &icpx,
        &args,
        &[PathBuf::from("kernels/intel/ffi.cpp")],
        &out_dir,
        "risc0_groth16_intel",
    );
    if !output.status.success() {
        panic!(
            "icpx failed for the Intel Groth16 prover:
{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    risc0_build_kernel::intel_link_library(&out_dir, "risc0_groth16_intel");
}

fn build_cuda_kernels() {
    let mut build = KernelBuild::new(KernelType::Cuda);
    if env::var("CARGO_FEATURE_SETUP").is_ok() {
        build.flag("-DSRS_READ_COEFFS");
    }
    build
        .files(["kernels/cuda/ffi.cu"])
        .deps(["kernels/cuda"])
        .flag("-D__ADX__")
        .include(env::var("DEP_BLST_C_SRC").unwrap())
        .include(env::var("DEP_SPPARK_ROOT").unwrap())
        .flag(&thp_pinned_define())
        .compile("risc0_groth16_cuda");

    println!("cargo:rustc-link-lib=static=blst");
}

fn build_rocm_kernels() {
    let output = "risc0_groth16_cuda";

    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=RISC0_HIP_ARCH");
    for entry in std::fs::read_dir("kernels/cuda").unwrap() {
        let path = entry.unwrap().path();
        println!("cargo:rerun-if-changed={}", path.display());
    }

    env::set_var("HIP_PLATFORM", "amd");

    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    // The sppark headers these kernels include (MSM, NTT): re-run when one changes, file by file,
    // as a directory would make cargo recurse into sppark's `rust/sppark` symlink loop.
    for file in risc0_build_kernel::kernel_source_files(Path::new(&sppark_root)) {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    let hipcc = risc0_build_kernel::find_hipcc();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let kernel_dir = std::fs::canonicalize("kernels/cuda").unwrap();

    let include_cuda2hip = format!("{sppark_root}/util/cuda2hip.hpp");
    let blst_src = env::var("DEP_BLST_C_SRC").unwrap();
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
        "-include",
        &include_cuda2hip,
        "-I",
        kernel_dir.to_str().unwrap(),
        "-I",
        &blst_src,
        "-I",
        &sppark_root,
        "-D__ADX__",
    ];
    // Windows hipcc rebuilds the command line for clang and drops unescaped quotes, which would
    // leave the header name bare; escaped, they reach clang as quotes.
    let thp_define = if cfg!(windows) {
        thp_pinned_define().replace('"', "\\\"")
    } else {
        thp_pinned_define()
    };
    base_flags.push(&thp_define);

    if env::var("CARGO_FEATURE_SETUP").is_ok() {
        base_flags.push("-DSRS_READ_COEFFS");
    }
    base_flags.extend(risc0_build_kernel::hip_host_flags());

    let obj = out_dir.join(format!(
        "groth16_ffi.{}",
        risc0_build_kernel::object_extension()
    ));
    let ffi_cu = kernel_dir.join("ffi.cu");
    risc0_build_kernel::hip_compile(&hipcc, &base_flags, &ffi_cu, &obj, &arches, None);

    // On MSVC, whose linker resolves symbols across libraries in any order, archive the object
    // alone and link blst beside it, as the CUDA build does.
    if risc0_build_kernel::is_msvc_target() {
        let _ = std::fs::remove_file(out_dir.join(format!("{output}.lib")));
        risc0_build_kernel::add_to_static_lib(&out_dir, output, &[obj]);
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static={output}");
        println!("cargo:rustc-link-lib=static=blst");
        risc0_build_kernel::emit_rocm_lib_link();
        return;
    }

    // Merge blst objects into our archive so that the linker can resolve
    // blst symbols referenced by host-side fp2_t/xyzz_t/jacobian_t code
    // in the same archive pass (avoids link-order issues with static libs).
    let build_dir = out_dir.parent().unwrap().parent().unwrap();
    let blst_lib_names: &[&str] = if cfg!(target_os = "windows") {
        &["blst.lib", "libblst.a"]
    } else {
        &["libblst.a"]
    };
    let mut blst_lib = None;
    for entry in std::fs::read_dir(build_dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name().to_string_lossy().starts_with("blst-") {
            for lib_name in blst_lib_names {
                let candidate = entry.path().join("out").join(lib_name);
                if candidate.exists() {
                    blst_lib = Some(candidate);
                    break;
                }
            }
            if blst_lib.is_some() {
                break;
            }
        }
    }
    let blst_lib = blst_lib.expect("could not find blst library in build directory");

    let lib_path = out_dir.join(format!("lib{output}.a"));
    let _ = std::fs::remove_file(&lib_path);

    let ar = risc0_build_kernel::find_ar_tool();

    // Create archive with our HIP object
    let status = Command::new(&ar)
        .arg("rcs")
        .arg(&lib_path)
        .arg(&obj)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

    // Extract blst objects and merge them into our archive
    let blst_tmp = out_dir.join("blst_objs");
    let _ = std::fs::remove_dir_all(&blst_tmp);
    std::fs::create_dir_all(&blst_tmp).unwrap();
    let status = Command::new(&ar)
        .arg("x")
        .arg(&blst_lib)
        .current_dir(&blst_tmp)
        .status()
        .expect("failed to extract blst objects");
    assert!(status.success(), "ar extract failed");

    for entry in std::fs::read_dir(&blst_tmp).unwrap() {
        let obj_path = entry.unwrap().path();
        let status = Command::new(&ar)
            .arg("rcs")
            .arg(&lib_path)
            .arg(&obj_path)
            .status()
            .expect("failed to add blst object to archive");
        assert!(status.success(), "ar append failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={output}");

    // Link HIP runtime
    risc0_build_kernel::emit_rocm_lib_link();
}

/// `-DRISC0_THP_PINNED_H="<risc0-sys>/kernels/zkp/cuda/thp_pinned.h"` (see pinned_upload.cuh). That
/// directory can't go on the include path: its `cuda.h` would shadow the toolkit's.
fn thp_pinned_define() -> String {
    let root = PathBuf::from(env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap());
    let path = root.join("thp_pinned.h");
    println!("cargo:rerun-if-changed={}", path.display());
    // Forward slashes: inside the quoted header name a Windows backslash would be read as an
    // escape by the preprocessor (both work as separators there).
    let path = path.to_string_lossy().replace('\\', "/");
    format!("-DRISC0_THP_PINNED_H=\"{path}\"")
}
