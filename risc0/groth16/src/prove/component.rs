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

//! Where the rzup `risc0-groth16` component (proving key, witness graph, preprocessed
//! coefficients) is, for the GPU provers.

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use rzup::{Component, Rzup, Version};

/// The rzup `risc0-groth16` component's directory: `RISC0_GROTH16_DIR` if set (rzup does not
/// run on Windows, where the files are installed by hand), else rzup's, else rzup's layout
/// under `$RISC0_HOME` or `~/.risc0`.
pub(crate) fn component_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("RISC0_GROTH16_DIR") {
        return Ok(dir.into());
    }
    let version = Version::new(0, 1, 0);
    if let Ok(dir) = Rzup::new().and_then(|r| r.get_version_dir(&Component::Risc0Groth16, &version))
    {
        return Ok(dir);
    }
    let home = std::env::var_os("RISC0_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(|home| PathBuf::from(home).join(".risc0"))
        })
        .ok_or_else(|| anyhow!("cannot locate the risc0-groth16 component: no home directory"))?;
    let dir = home
        .join("extensions")
        .join(format!("v{version}-risc0-groth16"));
    if dir.is_dir() {
        Ok(dir)
    } else {
        Err(anyhow!(
            "missing the `risc0-groth16` component: install it with `rzup install risc0-groth16`, \
             copy its files to {}, or set RISC0_GROTH16_DIR",
            dir.display()
        ))
    }
}
