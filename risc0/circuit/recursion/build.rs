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

fn main() {
    #[cfg(feature = "metal")]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        embed_metal();
    }
    #[cfg(feature = "prove")]
    download_zkr();
}

#[cfg(feature = "prove")]
fn download_zkr() {
    use std::{
        env, fs,
        path::{Path, PathBuf},
        str::FromStr,
    };

    use downloader::{verify, Download, DownloadSummary, Downloader};
    use sha2::{Digest, Sha256};

    const FILENAME: &str = "recursion_zkr.zip";
    const SRC_PATH: &str = "src/recursion_zkr.zip";
    // NOTE: This can be calculated with:
    // shasum -a256 risc0/circuit/recursion/src/recursion_zkr.zip
    const SHA256_HASH: &str = "744b999f0a35b3c86753311c7efb2a0054be21727095cf105af6ee7d3f4d8849";

    fn check_sha2(path: &Path) -> bool {
        let data = fs::read(path).unwrap();
        hex::encode(Sha256::digest(data)) == SHA256_HASH
    }

    println!("cargo:rerun-if-env-changed=RECURSION_SRC_PATH");

    let src_path = env::var("RECURSION_SRC_PATH").unwrap_or(SRC_PATH.to_string());
    let src_path = PathBuf::from_str(src_path.as_str()).unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);
    let out_path = out_dir.join(FILENAME);

    if env::var("DOCS_RS").is_ok() && !out_path.exists() {
        fs::write(&out_path, b"").unwrap();
        return;
    }

    if out_path.exists() {
        if check_sha2(&out_path) {
            return;
        }
        fs::remove_file(&out_path).unwrap();
    }

    if src_path.exists() && check_sha2(&src_path) {
        fs::copy(&src_path, &out_path).unwrap();
        return;
    }

    let mut downloader = Downloader::builder()
        .download_folder(out_dir)
        .build()
        .unwrap();
    let url = format!("https://risc0-artifacts.s3.us-west-2.amazonaws.com/zkr/{SHA256_HASH}.zip");
    eprintln!("Downloading {url}");
    let dl = Download::new(&url)
        .file_name(&PathBuf::from_str(FILENAME).unwrap())
        .verify(verify::with_digest::<Sha256>(
            hex::decode(SHA256_HASH).unwrap(),
        ));
    let results = downloader.download(&[dl]).unwrap();
    for result in results {
        let summary: DownloadSummary = result.unwrap();
        eprintln!("{summary}");
    }
}

// Reuse the circuit's authoritative generated polynomial. A missing or changed
// wrapper fails the build; no stale evaluator or CPU fallback is substituted.
#[cfg(feature = "metal")]
fn embed_metal() {
    use std::{env, fs, path::PathBuf};
    let root = PathBuf::from("../recursion-sys/kernels");
    let cuda = root.join("cuda/eval_check.cu");
    let wrapper = root.join("metal/eval_check.metal");
    println!("cargo:rerun-if-changed={}", cuda.display());
    println!("cargo:rerun-if-changed={}", wrapper.display());
    let cuda = fs::read_to_string(cuda).unwrap();
    let function = cuda
        .split_once("__device__ FpExt poly_fp(")
        .expect("poly_fp signature changed")
        .1;
    let body = function
        .split_once('{')
        .unwrap()
        .1
        .split_once("\n}")
        .unwrap()
        .0;
    assert!(
        !body.contains('{') && body.contains("return "),
        "poly_fp structure changed"
    );
    let wrapper = fs::read_to_string(wrapper).unwrap();
    assert_eq!(wrapper.matches("/* GENERATED_POLY_FP_BODY */").count(), 1);
    let kernel = wrapper.replace("/* GENERATED_POLY_FP_BODY */", body);
    let mut source = String::new();
    for (name, contents) in
        risc0_build_kernel::metal_headers()
            .iter()
            .copied()
            .chain(std::iter::once((
                "recursion_eval_check.metal",
                kernel.as_str(),
            )))
    {
        source.push_str(&format!("\n#line 1 \"{name}\"\n"));
        for line in contents.lines() {
            match line.trim() {
                "#pragma once" | "#include \"fp.h\"" | "#include \"fpext.h\"" => source.push('\n'),
                s if s.starts_with("#include \"") => panic!("unknown Metal include: {s}"),
                _ => {
                    source.push_str(line);
                    source.push('\n');
                }
            }
        }
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("recursion.metal");
    fs::write(&output, source).unwrap();
    println!(
        "cargo:rustc-env=RECURSION_METAL_SOURCE={}",
        output.display()
    );
}
