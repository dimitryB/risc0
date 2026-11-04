fn main() {
    #[cfg(feature = "metal")]
    {
        use std::{env, path::PathBuf, process::Command};
        assert_eq!(env::var("CARGO_CFG_TARGET_OS").as_deref(), Ok("macos"));
        assert_eq!(env::var("CARGO_CFG_TARGET_ARCH").as_deref(), Ok("aarch64"));
        let package = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
        let root = package.ancestors().nth(3).unwrap();
        for path in [
            package.join("metal"),
            root.join("risc0/circuit/rv32im-sys/kernels"),
            root.join("risc0/build_kernel/kernels/metal"),
            package.join("src/zirgen/info.rs"),
        ] {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("rv32im.metal");
        let result = Command::new("python3")
            .arg(package.join("metal/generate.py"))
            .arg("--engine")
            .arg(root)
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "RV32IM Metal generation failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        println!("cargo:rustc-env=RV32IM_METAL_SOURCE={}", output.display());
        let evalfast = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("rv32im-evalfast.metal");
        let result = Command::new("python3")
            .arg(package.join("metal/port_evalfast.py"))
            .arg("--engine")
            .arg(root)
            .arg("--reference-metadata")
            .arg(output.with_extension("json"))
            .arg("--output")
            .arg(&evalfast)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "RV32IM evalfast Metal generation failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        println!(
            "cargo:rustc-env=RV32IM_EVALFAST_METAL_SOURCE={}",
            evalfast.display()
        );
    }
}
