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
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use rayon::prelude::*;
use sha2::{Digest, Sha256};
use tempfile::tempdir_in;

const METAL_INCS: &[(&str, &str)] = &[
    ("fp.h", include_str!("../kernels/metal/fp.h")),
    ("fpext.h", include_str!("../kernels/metal/fpext.h")),
];

/// Shared field headers used by the retained Metal kernels.
///
/// Exposing the embedded sources lets runtime-compiled libraries use the same
/// headers as offline builds without depending on sibling crate source paths.
pub fn metal_headers() -> &'static [(&'static str, &'static str)] {
    METAL_INCS
}

#[derive(Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum KernelType {
    Cpp,
    Cuda,
    Metal,
}

pub struct KernelBuild {
    kernel_type: KernelType,
    flags: Vec<String>,
    files: Vec<PathBuf>,
    inc_dirs: Vec<PathBuf>,
    deps: Vec<PathBuf>,
}

impl KernelBuild {
    pub fn new(kernel_type: KernelType) -> Self {
        Self {
            kernel_type,
            flags: Vec::new(),
            files: Vec::new(),
            inc_dirs: Vec::new(),
            deps: Vec::new(),
        }
    }

    /// Add a directory to the `-I` or include path for headers
    pub fn include<P: AsRef<Path>>(&mut self, dir: P) -> &mut KernelBuild {
        self.inc_dirs.push(dir.as_ref().to_path_buf());
        self
    }

    /// Add an arbitrary flag to the invocation of the compiler
    pub fn flag(&mut self, flag: &str) -> &mut KernelBuild {
        self.flags.push(flag.to_string());
        self
    }

    /// Add a file which will be compiled
    pub fn file<P: AsRef<Path>>(&mut self, p: P) -> &mut KernelBuild {
        self.files.push(p.as_ref().to_path_buf());
        self
    }

    /// Add files which will be compiled
    pub fn files<P>(&mut self, p: P) -> &mut KernelBuild
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        for file in p.into_iter() {
            self.file(file);
        }
        self
    }

    /// Add a file which will be compiled
    pub fn file_opt<P: AsRef<Path>>(&mut self, _p: P, _opt: usize) -> &mut KernelBuild {
        self
    }

    /// Add files which will be compiled
    pub fn files_opt<P>(&mut self, _p: P, _opt: usize) -> &mut KernelBuild
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        self
    }

    /// Add a dependency
    pub fn dep<P: AsRef<Path>>(&mut self, p: P) -> &mut KernelBuild {
        self.deps.push(p.as_ref().to_path_buf());
        self
    }

    /// Add dependencies
    pub fn deps<P>(&mut self, p: P) -> &mut KernelBuild
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        for file in p.into_iter() {
            self.dep(file);
        }
        self
    }

    pub fn compile(&mut self, output: &str) {
        println!("cargo:rerun-if-env-changed=RISC0_SKIP_BUILD_KERNELS");
        for src in self.files.iter() {
            rerun_if_changed(src);
        }
        for dep in self.deps.iter() {
            rerun_if_changed(dep);
        }
        match &self.kernel_type {
            KernelType::Cpp => self.compile_cpp(output),
            KernelType::Cuda => self.compile_cuda(output),
            KernelType::Metal => self.compile_metal(output),
        }
    }

    fn compile_cpp(&mut self, output: &str) {
        if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
            return;
        }

        // It's *highly* recommended to install `sccache` and use this combined with
        // `RUSTC_WRAPPER=/path/to/sccache` to speed up rebuilds of C++ kernels
        let mut build = cc::Build::new();
        if is_msvc_target() {
            // The vendored NVTX headers include windows.h, whose min/max macros
            // break std::min/std::max (e.g. in the vendored poolstl).
            build.define("NOMINMAX", None);
            // C++20 for the generated layouts' designated initializers; see
            // `cuda_cxx_std_flag`.
            build.flag("/std:c++20");
        } else {
            build.flag_if_supported("-std=c++17");
        }
        build
            .cpp(true)
            .debug(false)
            .files(&self.files)
            .includes(&self.inc_dirs)
            .flag_if_supported("-fno-var-tracking")
            .flag_if_supported("-fno-var-tracking-assignments")
            .flag_if_supported("-g0")
            .compile(output);
    }

    fn compile_cuda(&mut self, output: &str) {
        println!("cargo:rerun-if-env-changed=NVCC_APPEND_FLAGS");
        println!("cargo:rerun-if-env-changed=NVCC_PREPEND_FLAGS");
        println!("cargo:rerun-if-env-changed=RISC0_CUDART_LINKAGE");
        println!("cargo:rerun-if-env-changed=NVCC_CCBIN");

        for inc_dir in self.inc_dirs.iter() {
            rerun_if_changed(inc_dir);
        }

        if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
            let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
            let out_path = out_dir.join(format!("lib{output}-skip.a"));
            fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&out_path)
                .unwrap();
            println!("cargo:{}={}", output, out_path.display());
            return;
        }

        let mut build = cc::Build::new();

        for file in self.files.iter() {
            build.file(file);
        }

        for inc in self.inc_dirs.iter() {
            build.include(inc);
        }

        for flag in self.flags.iter() {
            build.flag(flag);
        }

        if env::var_os("NVCC_PREPEND_FLAGS").is_none() && env::var_os("NVCC_APPEND_FLAGS").is_none()
        {
            build.flag("-arch=native");
        }

        let cudart = env::var("RISC0_CUDART_LINKAGE").unwrap_or("static".to_string());

        build
            .cuda(true)
            .cudart(&cudart)
            .debug(false)
            .ccbin(env::var("NVCC_CCBIN").is_err())
            .flag("-diag-suppress=177")
            .flag("-diag-suppress=2922")
            .flag("-Xcudafe")
            .flag("--display_error_number");
        if is_msvc_target() {
            build.flag(cuda_cxx_std_flag());
            msvc_cuda_build_setup(&mut build);
        } else {
            // GCC/Clang warning flags; cl.exe, nvcc's host compiler on MSVC, rejects them.
            build
                .flag("-Xcompiler")
                .flag("-Wno-missing-braces,-Wno-unused-function");
        }
        build.compile(output);
    }

    fn compile_metal(&mut self, output: &str) {
        let target = env::var("TARGET").unwrap();
        let sdk_name = if target.ends_with("ios") {
            "iphoneos"
        } else if target.ends_with("ios-sim") {
            "iphonesimulator"
        } else if target.ends_with("darwin") {
            "macosx"
        } else {
            panic!("unsupported target: {target}")
        };

        self.cached_compile(
            output,
            "metallib",
            METAL_INCS,
            &[],
            &[sdk_name.to_string()],
            |out_dir, out_path, sys_inc_dir, _flags| {
                let files: Vec<_> = self.files.iter().map(|x| x.as_path()).collect();

                let air_paths: Vec<_> = files
                    .into_par_iter()
                    .map(|src| {
                        let air_path = out_dir.join(src).with_extension("").with_extension("air");
                        if let Some(parent) = air_path.parent() {
                            fs::create_dir_all(parent).unwrap();
                        }
                        let mut cmd = Command::new("xcrun");
                        cmd.args(["--sdk", sdk_name]);
                        cmd.arg("metal");
                        cmd.arg("-o").arg(&air_path);
                        cmd.arg("-c").arg(src);
                        cmd.arg("-I").arg(sys_inc_dir);
                        cmd.arg("-Wno-unused-variable");
                        for inc_dir in self.inc_dirs.iter() {
                            cmd.arg("-I").arg(inc_dir);
                        }
                        println!("Running: {cmd:?}");
                        let status = cmd.status().unwrap();
                        if !status.success() {
                            panic!("Could not build metal kernels");
                        }
                        air_path
                    })
                    .collect();

                let result = Command::new("xcrun")
                    .args(["--sdk", sdk_name])
                    .arg("metallib")
                    .args(air_paths)
                    .arg("-o")
                    .arg(out_path)
                    .status()
                    .unwrap();
                if !result.success() {
                    panic!("Could not build metal kernels");
                }
            },
        );
    }

    fn cached_compile<F: Fn(&Path, &Path, &Path, &[String])>(
        &self,
        output: &str,
        extension: &str,
        assets: &[(&str, &str)],
        flags: &[String],
        tags: &[String],
        inner: F,
    ) {
        let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
        if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
            let out_path = out_dir
                .join("skip-".to_string() + output)
                .with_extension(extension);
            fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&out_path)
                .unwrap();
            println!("cargo:{}={}", output, out_path.display());
            return;
        }

        let out_path = out_dir.join(output).with_extension(extension);
        let sys_inc_dir = out_dir.join("_sys_");

        let cache_dir = risc0_cache();
        if !cache_dir.is_dir() {
            fs::create_dir_all(&cache_dir).unwrap();
        }

        let temp_dir = tempdir_in(&cache_dir).unwrap();
        let mut hasher = Hasher::new();
        for flag in flags {
            hasher.add_flag(flag);
        }
        for tag in tags {
            hasher.add_flag(tag);
        }
        for src in self.files.iter() {
            hasher.add_file(src);
        }
        for (name, contents) in assets {
            let path = sys_inc_dir.join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&path, contents).unwrap();
            hasher.add_file(path);
        }
        for dep in self.deps.iter() {
            hasher.add_file(dep);
        }
        let digest = hasher.finalize();
        let cache_path = cache_dir.join(digest).with_extension(extension);
        if !cache_path.is_file() {
            let tmp_dir = temp_dir.path();
            let tmp_path = tmp_dir.join(output).with_extension(extension);
            inner(tmp_dir, &tmp_path, &sys_inc_dir, flags);
            fs::rename(tmp_path, &cache_path).unwrap();
        }
        fs::copy(cache_path, &out_path).unwrap();

        println!("cargo:{}={}", output, out_path.display());
    }
}

fn risc0_cache() -> PathBuf {
    directories::ProjectDirs::from("com.risczero", "RISC Zero", "risc0")
        .unwrap()
        .cache_dir()
        .into()
}

struct Hasher {
    sha: Sha256,
}

impl Hasher {
    pub fn new() -> Self {
        Self { sha: Sha256::new() }
    }

    pub fn add_flag(&mut self, flag: &str) {
        self.sha.update(flag);
    }

    pub fn add_file<P: AsRef<Path>>(&mut self, path: P) {
        let bytes = fs::read(path).unwrap();
        self.sha.update(bytes);
    }

    pub fn finalize(self) -> String {
        hex::encode(self.sha.finalize())
    }
}

fn rerun_if_changed<P: AsRef<Path>>(path: P) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}

// ---------------------------------------------------------------------------
// CUDA build utilities (cross-platform: Linux + Windows/MSVC)
// ---------------------------------------------------------------------------

/// Whether the target uses the MSVC toolchain, where nvcc's host compiler is
/// cl.exe and static libraries are `<name>.lib` archives built with lib.exe.
pub fn is_msvc_target() -> bool {
    env::var("CARGO_CFG_TARGET_ENV").is_ok_and(|e| e == "msvc")
}

fn crt_static() -> bool {
    env::var("CARGO_CFG_TARGET_FEATURE").is_ok_and(|f| f.split(',').any(|f| f == "crt-static"))
}

/// The nvcc for a direct invocation (one that bypasses `cc`): `NVCC` if set,
/// else `nvcc` on PATH.
pub fn find_nvcc() -> String {
    env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string())
}

/// A `Command` for a direct nvcc invocation (one that bypasses `cc`).
///
/// On MSVC it is set up the way `cc` sets up its own: nvcc is pointed at the
/// cl.exe `cc` found and given that compiler's environment (`INCLUDE`, `LIB`,
/// `PATH`), so it works outside a Visual Studio developer shell; the C runtime is
/// the one rustc links (`/MD`, or `/MT` under `+crt-static`; cl.exe's own default
/// is `/MT`, which does not link with a `/MD` binary); and it gets the setup in
/// [`msvc_cuda_build_setup`]. Elsewhere it is a bare nvcc, for the caller's
/// host-compiler flags.
pub fn nvcc_command() -> Command {
    let mut cmd = Command::new(find_nvcc());
    if is_msvc_target() {
        let cl = cc::Build::new().cpp(true).get_compiler();
        for (key, value) in cl.env() {
            cmd.env(key, value);
        }
        let crt = if crt_static() { "/MT" } else { "/MD" };
        cmd.arg(format!("-ccbin={}", cl.path().display()))
            .arg("-Xcompiler")
            .arg(format!("{crt},/O2"))
            .arg(format!("-I{}", msvc_toolkit_include_dir().display()))
            .arg("-DNOMINMAX");
    }
    cmd
}

/// The C++ standard flag for nvcc. The generated circuit layouts (e.g.
/// keccak-sys's `layout.cu.inc`) use designated initializers, a C++20 feature
/// that GCC/Clang also accept in C++17 mode; MSVC, and nvcc's front end in MSVC
/// mode, require C++20 for them.
pub fn cuda_cxx_std_flag() -> &'static str {
    if is_msvc_target() {
        "-std=c++20"
    } else {
        "-std=c++17"
    }
}

/// MSVC setup for a `cc` build of CUDA kernels: the toolkit forwarding headers
/// (see [`msvc_toolkit_include_dir`]) and `NOMINMAX`, without which the
/// `windows.h` the vendored NVTX headers include defines `min`/`max` macros that
/// break `std::min`/`std::max`.
pub fn msvc_cuda_build_setup(build: &mut cc::Build) {
    build
        .include(msvc_toolkit_include_dir())
        .define("NOMINMAX", None);
}

/// On MSVC, a directory to add to the include path holding headers that include
/// CUDA toolkit headers by absolute path:
///
/// * `risc0_toolkit_cuda.h` — the toolkit's `cuda.h`. risc0's kernels have a
///   `cuda.h` of their own, which shadows the toolkit's wherever the kernel
///   directory is on the include path, and reaches it with `#include_next`.
/// * `risc0_toolkit_cuda_std_array.h` — the toolkit's `<cuda/std/array>`, which
///   the hemilabs sppark fork's `cuda/std/array` (a HIP shim) shadows the same way.
///
/// `#include_next` is a GCC/Clang extension MSVC lacks; on MSVC those headers
/// include these instead. The paths are written into files rather than passed as
/// `-D` defines, so they need no shell quoting.
pub fn msvc_toolkit_include_dir() -> PathBuf {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    let include = cuda_toolkit_dir().join("include");
    let cuda_h = include.join("cuda.h");
    // CUDA 13 moved the CCCL headers (libcudacxx among them) under include/cccl.
    let cuda_std_array = [include.join("cuda/std/array"), include.join("cccl/cuda/std/array")]
        .into_iter()
        .find(|p| p.is_file());
    assert!(cuda_h.is_file(), "CUDA toolkit header not found at {}", cuda_h.display());
    let cuda_std_array = cuda_std_array.unwrap_or_else(|| {
        panic!("<cuda/std/array> not found under {}", include.display())
    });

    let dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("risc0_msvc_include");
    fs::create_dir_all(&dir).unwrap();
    for (name, target) in [
        ("risc0_toolkit_cuda.h", &cuda_h),
        ("risc0_toolkit_cuda_std_array.h", &cuda_std_array),
    ] {
        let path = target.to_string_lossy().replace('\\', "/");
        fs::write(dir.join(name), format!("#pragma once\n#include \"{path}\"\n")).unwrap();
    }
    dir
}

/// The CUDA toolkit root: from `NVCC` when it is a full path to nvcc (which lives
/// in `<root>/bin`), else `CUDA_PATH`, which the Windows installer sets.
fn cuda_toolkit_dir() -> PathBuf {
    if let Ok(nvcc) = env::var("NVCC") {
        let nvcc = PathBuf::from(nvcc);
        if nvcc.is_absolute() {
            if let Some(root) = nvcc.parent().and_then(Path::parent) {
                return root.to_path_buf();
            }
        }
    }
    env::var("CUDA_PATH").map(PathBuf::from).expect(
        "cannot locate the CUDA toolkit: set CUDA_PATH, or NVCC to the full path of nvcc",
    )
}

/// File extension for an object file on the target toolchain.
pub fn object_extension() -> &'static str {
    if is_msvc_target() {
        "obj"
    } else {
        "o"
    }
}

/// Add `objs` to the static library `name` in `out_dir`, creating it if absent.
///
/// The library is the one `cargo:rustc-link-lib=static=<name>` resolves to:
/// `lib<name>.a` built with `ar` elsewhere, `<name>.lib` built with lib.exe on
/// MSVC (which is also what `cc` names the libraries it compiles there).
pub fn add_to_static_lib(out_dir: &Path, name: &str, objs: &[PathBuf]) {
    if is_msvc_target() {
        let lib = out_dir.join(format!("{name}.lib"));
        let mut cmd = cc::Build::new().get_archiver();
        cmd.arg("/NOLOGO").arg(format!("/OUT:{}", lib.display()));
        if lib.exists() {
            cmd.arg(&lib);
        }
        cmd.args(objs);
        let status = cmd.status().expect("failed to run lib.exe");
        assert!(status.success(), "lib.exe failed to build {}", lib.display());
    } else {
        let lib = out_dir.join(format!("lib{name}.a"));
        let status = Command::new(find_ar_tool())
            .arg("rcs")
            .arg(&lib)
            .args(objs)
            .status()
            .expect("failed to run ar");
        assert!(status.success(), "ar failed to build {}", lib.display());
    }
}

/// Everything outside a direct nvcc command's own arguments that changes what it
/// produces: the `NVCC_{PREPEND,APPEND}_FLAGS` nvcc reads from the environment
/// (where the GPU architecture list is usually set) and the nvcc version.
/// For keying a cache of nvcc output.
pub fn nvcc_environment_key() -> String {
    // CL and _CL_ are cl.exe's own extra flags; CPATH and CPLUS_INCLUDE_PATH extend gcc's
    // include path.
    let mut key = env_key(&[
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "CL",
        "_CL_",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
    ]);
    if is_msvc_target() {
        // The toolkit the forwarding headers of `msvc_toolkit_include_dir` point into.
        key.push_str(&format!("toolkit={}\n", cuda_toolkit_dir().display()));
        // cl.exe prints its version (to stderr) when run without arguments.
        let cl = cc::Build::new().cpp(true).get_compiler();
        let mut cmd = Command::new(cl.path());
        cmd.envs(cl.env().iter().cloned());
        key.push_str(&format!("host={}\n", tool_output(cmd)));
    } else {
        // The host compiler nvcc runs (`-ccbin=c++` or its default, gcc), by version.
        for host in ["c++", "gcc"] {
            let mut cmd = Command::new(host);
            cmd.arg("--version");
            key.push_str(&format!("{host}={}\n", tool_output(cmd)));
        }
    }
    let mut nvcc = Command::new(find_nvcc());
    nvcc.arg("--version");
    key.push_str(&tool_output(nvcc));
    key
}

/// `NAME=value` lines for `vars` (empty when unset), each also registered with cargo as a
/// reason to re-run the build script, for keying a cache on the environment.
fn env_key(vars: &[&str]) -> String {
    let mut key = String::new();
    for var in vars {
        println!("cargo:rerun-if-env-changed={var}");
        key.push_str(&format!("{var}={}\n", env::var(var).unwrap_or_default()));
    }
    key
}

/// What `cmd` prints (stdout, then stderr), or "" if it can't run: for keying a cache on a
/// tool's version.
fn tool_output(mut cmd: Command) -> String {
    cmd.output()
        .map(|o| {
            format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            )
        })
        .unwrap_or_default()
}

/// The GPUs `-arch=native` (nvcc) or `--offload-arch=native` (hipcc) compiles for on this
/// machine, best effort (empty if the query tool is missing). For keying a cache of output
/// built for `native`, so moving the build to a different GPU can't reuse it. Includes the
/// `*_VISIBLE_DEVICES` variables, which pick the GPUs `native` sees (nvidia-smi ignores
/// CUDA_VISIBLE_DEVICES); only call it when compiling for `native`, as they don't matter
/// otherwise.
pub fn native_gpu_arch_key(hip: bool) -> String {
    let mut key = if hip {
        env_key(&["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"])
    } else {
        env_key(&["CUDA_VISIBLE_DEVICES"])
    };
    key.push_str(&native_gpus(hip));
    key
}

fn native_gpus(hip: bool) -> String {
    let mut queries: Vec<(String, &[&str])> = Vec::new();
    if hip {
        // Usually not on PATH: look in the ROCm install too.
        for root in ["HIP_PATH", "ROCM_PATH"]
            .iter()
            .filter_map(|v| env::var(v).ok())
        {
            for dir in ["bin", "llvm/bin"] {
                for tool in ["amdgpu-arch", "amdgpu-arch.exe", "rocm_agent_enumerator"] {
                    let p = Path::new(&root).join(dir).join(tool);
                    if p.is_file() {
                        queries.push((p.to_string_lossy().into_owned(), &[]));
                    }
                }
            }
        }
        if Path::new("/opt/rocm/llvm/bin/amdgpu-arch").is_file() {
            queries.push(("/opt/rocm/llvm/bin/amdgpu-arch".into(), &[]));
        }
        queries.push(("amdgpu-arch".into(), &[]));
        queries.push(("rocm_agent_enumerator".into(), &[]));
    } else {
        queries.push((
            "nvidia-smi".into(),
            &["--query-gpu=compute_cap", "--format=csv,noheader"],
        ));
    }
    for (tool, args) in &queries {
        if let Ok(o) = Command::new(tool).args(*args).output() {
            if o.status.success() {
                // The tool's path isn't part of the key, only what it found.
                return String::from_utf8_lossy(&o.stdout).trim().to_string();
            }
        }
    }
    String::new()
}

/// Everything outside a direct hipcc command's own arguments that changes what it produces:
/// the environment hipcc reads its compiler and extra flags from, and the hipcc version.
/// For keying a cache of hipcc output.
pub fn hipcc_environment_key(hipcc: &str) -> String {
    let mut key = env_key(&[
        "HIPCC_COMPILE_FLAGS_APPEND",
        "HIPCC_COMPILE_FLAGS_PREPEND",
        "HIP_CLANG_PATH",
        "HIP_PATH",
        "ROCM_PATH",
        "HIP_PLATFORM",
        "HIP_DEVICE_LIB_PATH",
        "DEVICE_LIB_PATH",
        "HIP_INCLUDE_PATH",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
    ]);
    let mut cmd = Command::new(hipcc);
    cmd.arg("--version");
    key.push_str(&tool_output(cmd));
    key
}

/// Every file under `root` that [`kernel_sources_key`] hashes, searched recursively, skipping
/// `target` directories, hidden files and directories and editor backups, and not following
/// directory symlinks. For emitting `cargo:rerun-if-changed` per file, which, unlike a
/// directory, can't recurse into a symlink loop.
pub fn kernel_source_files(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') {
                continue;
            }
            if file_type.is_dir() {
                if name != "target" {
                    walk(&path, out);
                }
            } else if path.is_file()
                && !name.ends_with('~')
                && ![".orig", ".rej", ".bak"].iter().any(|s| name.ends_with(s))
            {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out.sort();
    out
}

/// SHA-256 (hex) over every file under `roots` ([`kernel_source_files`]; not following
/// directory symlinks, since sppark's `rust/sppark` links back to its root) that `include`
/// accepts, plus `files`, by relative path and contents. A superset of what a compile with
/// these include roots can read, whatever the extension (sppark includes `.hip` files and an
/// extensionless `cuda/std/array`), for keying a cache of its output: unlike a list of known
/// files, it can't miss a header a change adds. `include` can narrow it to what the compile
/// can reach, e.g. leaving out a root's own translation units.
pub fn kernel_sources_key(
    roots: &[&Path],
    files: &[&Path],
    include: impl Fn(&Path) -> bool,
) -> String {
    let mut hasher = Sha256::new();
    let mut add = |label: &str, path: &Path| {
        // A file that can't be read hashes as missing rather than failing the build.
        let contents = fs::read(path).ok();
        hasher.update((label.len() as u64).to_le_bytes());
        hasher.update(label.as_bytes());
        match contents {
            Some(c) => {
                hasher.update((c.len() as u64).to_le_bytes());
                hasher.update(&c);
            }
            None => hasher.update(u64::MAX.to_le_bytes()),
        }
    };
    for (i, root) in roots.iter().enumerate() {
        let mut labeled: Vec<(String, PathBuf)> = kernel_source_files(root)
            .into_iter()
            .filter(|p| include(p))
            .map(|p| {
                let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy();
                (format!("{i}/{}", rel.replace('\\', "/")), p)
            })
            .collect();
        labeled.sort();
        for (label, path) in &labeled {
            add(label, path);
        }
    }
    for file in files {
        add(
            &format!("file/{}", file.to_string_lossy().replace('\\', "/")),
            file,
        );
    }
    hex::encode(hasher.finalize())
}

/// Copy `<cache_dir>/<stem>-<key>.<ext>`, a compiler output cached under `key` (which must
/// cover everything the output depends on), to `dest`, building it with `compile(path)` on a
/// miss.
///
/// The output goes to a temporary path and is renamed into place only once `compile`
/// returns, so an interrupted or failed build never leaves a file a later build would take
/// for finished, and builds of different keys sharing `cache_dir` can't overwrite each other's
/// output. After a build, outputs not written for a day are deleted, to bound the cache.
pub fn cached_compile(
    cache_dir: &Path,
    stem: &str,
    ext: &str,
    key: &str,
    dest: &Path,
    compile: impl FnOnce(&Path),
) {
    let key = &key[..key.len().min(32)];
    let cached = cache_dir.join(format!("{stem}-{key}.{ext}"));
    // Copy first, then mark it in use (so another build's pruning below leaves it alone): if
    // another build pruned it after all, the copy finds nothing and this builds it afresh.
    match fs::copy(&cached, dest) {
        Ok(_) => {
            eprintln!("{stem}: inputs unchanged, reused {}", cached.display());
            if let Ok(f) = fs::OpenOptions::new().write(true).open(&cached) {
                let _ = f.set_modified(std::time::SystemTime::now());
            }
            return;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !cached.exists() => {}
        Err(e) => panic!(
            "failed to copy {} to {}: {e}",
            cached.display(),
            dest.display()
        ),
    }
    eprintln!("{stem}: inputs changed (or first build), compiling...");
    let tmp = cache_dir.join(format!("{stem}-{key}.tmp{}.{ext}", std::process::id()));
    let _ = fs::remove_file(&tmp);
    // Removes the (possibly large, partial) temporary output if `compile` panics.
    struct RemoveOnDrop<'a>(&'a Path, bool);
    impl Drop for RemoveOnDrop<'_> {
        fn drop(&mut self) {
            if self.1 {
                let _ = fs::remove_file(self.0);
            }
        }
    }
    let mut guard = RemoveOnDrop(&tmp, true);
    compile(&tmp);
    assert!(
        tmp.exists(),
        "{stem}: the compiler produced no {}",
        tmp.display()
    );
    fs::copy(&tmp, dest).unwrap_or_else(|e| {
        panic!(
            "failed to copy {} to {}: {e}",
            tmp.display(),
            dest.display()
        )
    });
    // On Windows a scanner briefly holding the new file can make the rename fail: retry, and
    // if it still fails, leave it uncached (the temporary file is removed).
    let mut renamed = fs::rename(&tmp, &cached);
    for attempt in 1..=10 {
        if renamed.is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100 * attempt));
        renamed = fs::rename(&tmp, &cached);
    }
    match renamed {
        Ok(()) => guard.1 = false,
        Err(e) => println!(
            "cargo:warning=could not move {} into the cache ({e}); it will be rebuilt next time",
            tmp.display()
        ),
    }
    drop(guard);
    // Leave anything written in the last day: another build script running now may be about
    // to use it (two versions of a crate in one build share `cache_dir`).
    let day = std::time::Duration::from_secs(24 * 60 * 60);
    if let Ok(entries) = fs::read_dir(cache_dir) {
        let prefix = format!("{stem}-");
        for entry in entries.flatten() {
            let stale = entry
                .metadata()
                .and_then(|m| m.modified())
                .is_ok_and(|t| t.elapsed().is_ok_and(|age| age > day));
            if stale && entry.file_name().to_string_lossy().starts_with(&prefix) {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
}

/// A cache key for what `cmd` runs: its program, arguments, working directory and the
/// environment it sets. `OUT_DIR`, which changes with the crate's fingerprint (and appears in
/// arguments such as the MSVC forwarding-header include), is replaced by a placeholder, so
/// that a rebuild of the same sources reuses the cached output.
pub fn command_key(cmd: &Command) -> String {
    let out_dir = env::var("OUT_DIR").unwrap_or_default();
    let normalize = |s: &std::ffi::OsStr| {
        let s = s.to_string_lossy();
        if out_dir.is_empty() {
            s.into_owned()
        } else {
            s.replace(&out_dir, "<OUT_DIR>")
        }
    };
    let mut key = format!("program={}\n", normalize(cmd.get_program()));
    for arg in cmd.get_args() {
        key.push_str(&format!("arg={}\n", normalize(arg)));
    }
    if let Some(dir) = cmd.get_current_dir() {
        key.push_str(&format!("cwd={}\n", normalize(dir.as_os_str())));
    }
    // Not PATH, which `cc` passes whole for MSVC: it varies with the shell, and the tools it
    // finds are keyed by their versions.
    let mut envs: Vec<_> = cmd
        .get_envs()
        .filter(|(k, _)| !k.eq_ignore_ascii_case("PATH"))
        .collect();
    envs.sort();
    for (k, v) in envs {
        let v = v.map(normalize).unwrap_or_else(|| "<removed>".into());
        key.push_str(&format!("env {}={v}\n", k.to_string_lossy()));
    }
    key
}

/// SHA-256 (hex) of `parts`, each length-prefixed so different splits can't collide.
pub fn hash_key(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part.as_bytes());
    }
    hex::encode(hasher.finalize())
}

// ---------------------------------------------------------------------------
// HIP build utilities (cross-platform: Linux + Windows)
// ---------------------------------------------------------------------------

/// Find the hipcc compiler, handling Windows HIP SDK quirks.
///
/// On Windows, the HIP SDK ships `hipcc` as a Perl wrapper script and
/// `hipcc.bin.exe` as the actual compiler driver. This function checks
/// `HIPCC` env var first, then tries `hipcc.bin.exe` on Windows (avoids
/// Perl dependency), then falls back to `hipcc` on PATH.
pub fn find_hipcc() -> String {
    if let Ok(hipcc) = env::var("HIPCC") {
        return hipcc;
    }

    if cfg!(target_os = "windows") {
        // Try hipcc.bin.exe from HIP_PATH (avoids Perl dependency)
        if let Ok(hip_path) = env::var("HIP_PATH") {
            let candidate = PathBuf::from(&hip_path).join("bin").join("hipcc.bin.exe");
            if candidate.exists() {
                return candidate.to_string_lossy().into_owned();
            }
        }
        // Try standard Windows ROCm install paths
        if let Some(candidate) = find_in_windows_rocm("bin", "hipcc.bin.exe") {
            return candidate;
        }
    }

    "hipcc".to_string()
}

/// Find a platform-appropriate static archive tool (`ar` / `llvm-ar`).
///
/// On Windows, GNU `ar` is typically not available. This function looks for
/// `llvm-ar` in the HIP SDK installation and on PATH.
/// On Unix, returns `ar`.
///
/// Respects the `AR` environment variable.
pub fn find_ar_tool() -> String {
    if let Ok(ar) = env::var("AR") {
        return ar;
    }

    if cfg!(target_os = "windows") {
        // Try llvm-ar from HIP SDK
        if let Ok(hip_path) = env::var("HIP_PATH") {
            for subdir in &["bin", "llvm/bin"] {
                let candidate = PathBuf::from(&hip_path).join(subdir).join("llvm-ar.exe");
                if candidate.exists() {
                    return candidate.to_string_lossy().into_owned();
                }
            }
        }
        // Try standard Windows ROCm install paths
        for subdir in &["bin", "llvm\\bin"] {
            if let Some(candidate) = find_in_windows_rocm(subdir, "llvm-ar.exe") {
                return candidate;
            }
        }
        // Fall back to llvm-ar on PATH (better chance than ar on Windows)
        return "llvm-ar".to_string();
    }

    "ar".to_string()
}

/// Emit `cargo:rustc-link-search` for the HIP runtime library directory.
///
/// Checks `HIP_PATH` env var first, then platform-specific default locations
/// (`/opt/rocm/lib` on Linux, `C:\Program Files\AMD\ROCm\<ver>\lib` on Windows).
pub fn emit_rocm_lib_link() {
    if let Ok(hip_path) = env::var("HIP_PATH") {
        println!("cargo:rustc-link-search=native={}/lib", hip_path);
    } else if cfg!(target_os = "windows") {
        if let Some(lib_path) = find_in_windows_rocm("lib", "amdhip64.lib") {
            // Return the directory, not the file
            let p = PathBuf::from(&lib_path);
            if let Some(parent) = p.parent() {
                println!("cargo:rustc-link-search=native={}", parent.display());
            }
        }
    } else if Path::new("/opt/rocm/lib").exists() {
        println!("cargo:rustc-link-search=native=/opt/rocm/lib");
    }
    println!("cargo:rustc-link-lib=amdhip64");
}

/// Search Windows standard ROCm install paths for a file.
///
/// Looks in `C:\Program Files\AMD\ROCm\<version>\<subdir>\<filename>`,
/// trying the latest version first.
fn find_in_windows_rocm(subdir: &str, filename: &str) -> Option<String> {
    let program_files =
        env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".to_string());
    let rocm_dir = PathBuf::from(&program_files).join("AMD").join("ROCm");
    if let Ok(entries) = fs::read_dir(&rocm_dir) {
        let mut versions: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        versions.sort();
        // Try latest version first
        for ver in versions.iter().rev() {
            let candidate = ver.join(subdir).join(filename);
            if candidate.exists() {
                return Some(candidate.to_string_lossy().into_owned());
            }
        }
    }
    None
}

/// Resolve `RISC0_HIP_ARCH` from the environment, defaulting to `"native"`.
pub fn hip_arches() -> String {
    env::var("RISC0_HIP_ARCH").unwrap_or_else(|_| "native".to_string())
}

/// Compile a HIP source file to an object, potentially for multiple GPU
/// architectures in parallel.
///
/// * `hipcc`    — path to the hipcc compiler.
/// * `flags`    — all compiler flags **except** `--offload-arch`, `-c`, and `-o`.
/// * `source`   — the `.cu` / `.hip` source file.
/// * `output`   — destination object file.
/// * `arches`   — comma-separated arch list (e.g. `"gfx1100,gfx1201"`) or `"native"`.
/// * `work_dir` — optional working directory for hipcc.
///
/// When `arches` contains multiple targets, device code is compiled in parallel
/// (one hipcc process per arch) and merged with `clang-offload-bundler`.
/// Falls back to a single sequential hipcc invocation if the parallel pipeline
/// encounters an error.
pub fn hip_compile(
    hipcc: &str,
    flags: &[&str],
    source: &Path,
    output: &Path,
    arches: &str,
    work_dir: Option<&Path>,
) {
    // Always use single-invocation hipcc which correctly generates
    // __hip_fatbin_* symbols. The host/device split + clang-offload-bundler
    // approach doesn't produce proper fat binary symbols in ROCm 7.x.
    hip_compile_single(hipcc, flags, source, output, arches, work_dir);
}

/// Single-invocation hipcc compilation (original behaviour).
fn hip_compile_single(
    hipcc: &str,
    flags: &[&str],
    source: &Path,
    output: &Path,
    arches: &str,
    work_dir: Option<&Path>,
) {
    let mut cmd = Command::new(hipcc);
    if let Some(dir) = work_dir {
        cmd.current_dir(dir);
    }
    for flag in flags {
        cmd.arg(flag);
    }
    cmd.arg(format!("--offload-arch={arches}"));
    cmd.arg("-c").arg(source).arg("-o").arg(output);

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("hipcc failed to start: {e}"));
    assert!(status.success(), "hipcc failed for {}", source.display());
}

/// Parallel multi-arch compilation:
///   1. Host code compiled once (`--offload-host-only`).
///   2. Device code compiled per-arch in parallel (`--offload-device-only`).
///   3. Results merged with `clang-offload-bundler`.
#[allow(dead_code)]
fn hip_compile_parallel(
    hipcc: &str,
    flags: &[&str],
    source: &Path,
    output: &Path,
    arch_list: &[&str],
    work_dir: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = output
        .parent()
        .ok_or("output path has no parent directory")?;
    let stem = output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("hip_kernel");

    eprintln!(
        "hip_compile: parallel build for {} ({} arches: {})",
        source.display(),
        arch_list.len(),
        arch_list.join(", "),
    );

    // Collect flags into owned Strings so we can share across threads.
    let flags_owned: Vec<String> = flags.iter().map(|s| s.to_string()).collect();
    let hipcc_owned = hipcc.to_string();
    let source_owned = source.to_path_buf();
    let work_dir_owned = work_dir.map(|d| d.to_path_buf());

    // 1. Host-only compilation (fast — no device codegen).
    let host_obj = out_dir.join(format!("{stem}_host.o"));
    {
        let mut cmd = Command::new(hipcc);
        if let Some(dir) = work_dir {
            cmd.current_dir(dir);
        }
        for flag in flags {
            cmd.arg(flag);
        }
        cmd.arg("--offload-host-only");
        cmd.arg(format!("--offload-arch={}", arch_list[0]));
        cmd.arg("-c").arg(source).arg("-o").arg(&host_obj);

        let status = cmd.status()?;
        if !status.success() {
            return Err(
                format!("hipcc --offload-host-only failed for {}", source.display()).into(),
            );
        }
    }

    // 2. Device-only compilation — one per arch, in parallel via rayon.
    let dev_results: Vec<Result<PathBuf, String>> = arch_list
        .par_iter()
        .map(|arch| {
            let dev_obj = out_dir.join(format!("{stem}_dev_{arch}.o"));
            let mut cmd = Command::new(&hipcc_owned);
            if let Some(ref dir) = work_dir_owned {
                cmd.current_dir(dir);
            }
            for flag in &flags_owned {
                cmd.arg(flag);
            }
            cmd.arg("--offload-device-only");
            cmd.arg(format!("--offload-arch={arch}"));
            cmd.arg("-c").arg(&source_owned).arg("-o").arg(&dev_obj);

            let status = cmd
                .status()
                .map_err(|e| format!("hipcc device-only ({arch}) failed to start: {e}"))?;
            if !status.success() {
                return Err(format!(
                    "hipcc --offload-device-only --offload-arch={arch} failed"
                ));
            }
            Ok(dev_obj)
        })
        .collect();

    // Collect results, bail on any failure.
    let dev_objs: Vec<PathBuf> = dev_results
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // 3. Bundle host + all device objects into one fat binary.
    let bundler = find_offload_bundler(hipcc);
    let mut cmd = Command::new(&bundler);
    cmd.arg("-type=o");

    // Host input first.
    cmd.arg(format!("-input={}", host_obj.display()));
    let host_triple = if cfg!(target_os = "windows") {
        "host-x86_64-pc-windows-msvc"
    } else if cfg!(target_os = "macos") {
        "host-x86_64-apple-darwin"
    } else {
        "host-x86_64-unknown-linux-gnu"
    };
    let mut targets = host_triple.to_string();

    // Device inputs.
    for (arch, dev_obj) in arch_list.iter().zip(&dev_objs) {
        cmd.arg(format!("-input={}", dev_obj.display()));
        targets.push_str(&format!(",hipv4-amdgcn-amd-amdhsa--{arch}"));
    }

    cmd.arg(format!("-output={}", output.display()));
    cmd.arg(format!("-targets={targets}"));

    eprintln!(
        "hip_compile: bundling {} arches → {}",
        arch_list.len(),
        output.display()
    );
    let status = cmd.status()?;
    if !status.success() {
        return Err("clang-offload-bundler failed".into());
    }

    // Cleanup intermediate files.
    let _ = fs::remove_file(&host_obj);
    for obj in &dev_objs {
        let _ = fs::remove_file(obj);
    }

    Ok(())
}

/// Locate `clang-offload-bundler`, searching ROCm paths and hipcc's neighbourhood.
#[allow(dead_code)]
fn find_offload_bundler(hipcc: &str) -> String {
    let exe_name = if cfg!(target_os = "windows") {
        "clang-offload-bundler.exe"
    } else {
        "clang-offload-bundler"
    };

    // 1. Explicit env var.
    if let Ok(path) = env::var("CLANG_OFFLOAD_BUNDLER") {
        return path;
    }

    // 2. Platform-specific standard install paths.
    if cfg!(target_os = "windows") {
        // Windows: check HIP SDK paths
        if let Ok(hip_path) = env::var("HIP_PATH") {
            for subdir in &["bin", "llvm/bin"] {
                let candidate = PathBuf::from(&hip_path).join(subdir).join(exe_name);
                if candidate.is_file() {
                    return candidate.to_string_lossy().into_owned();
                }
            }
        }
        if let Some(path) = find_in_windows_rocm("llvm\\bin", exe_name) {
            return path;
        }
    } else {
        let rocm_path = "/opt/rocm/llvm/bin/clang-offload-bundler";
        if Path::new(rocm_path).is_file() {
            return rocm_path.to_string();
        }
    }

    // 3. Adjacent to hipcc.
    if let Ok(resolved) = fs::canonicalize(hipcc) {
        if let Some(bin_dir) = resolved.parent() {
            // hipcc is usually in <rocm>/bin/ ; bundler in <rocm>/llvm/bin/
            let candidate = bin_dir.join("../llvm/bin").join(exe_name);
            if let Ok(canon) = fs::canonicalize(&candidate) {
                if canon.is_file() {
                    return canon.to_string_lossy().into_owned();
                }
            }
        }
    }

    // 4. PATH fallback.
    "clang-offload-bundler".to_string()
}
