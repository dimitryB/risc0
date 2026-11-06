// Copyright 2026 RISC Zero, Inc.
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

//! Optional Metal binary archive for the exact retained shader source.
//! The archive is only a compilation cache; callers still compile the source
//! library and obtain functions from it on every process start.

use std::{
    cell::RefCell, collections::HashSet, fmt, fs, path::PathBuf, process::Command, sync::OnceLock,
};

use metal::{
    BinaryArchive, BinaryArchiveDescriptor, ComputePipelineDescriptorRef, ComputePipelineState,
    Device, URL,
};
use sha2::{Digest, Sha256};

pub struct PipelineArchive {
    archive: Option<BinaryArchive>,
    path: Option<PathBuf>,
    loaded: bool,
    added: RefCell<HashSet<String>>,
}

impl fmt::Debug for PipelineArchive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineArchive")
            .field("path", &self.path)
            .field("loaded", &self.loaded)
            .finish()
    }
}

fn safe_key(part: &str) -> String {
    part.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn os_build() -> Option<&'static str> {
    static BUILD: OnceLock<Option<String>> = OnceLock::new();
    BUILD
        .get_or_init(|| {
            let output = Command::new("sysctl")
                .args(["-n", "kern.osversion"])
                .output()
                .ok()?;
            output.status.success().then_some(())?;
            let build = String::from_utf8(output.stdout).ok()?;
            let build = build.trim();
            (!build.is_empty()).then(|| safe_key(build))
        })
        .as_deref()
}

fn file_url(path: &PathBuf) -> URL {
    // metal 0.29 wraps NSURL's autoreleased URLWithString: result as an
    // owned URL. Retain a separate wrapper and leave the original release
    // to the autorelease pool, otherwise the pool releases it twice.
    let autoreleased = URL::new_with_string(&format!("file://{}", path.display()));
    let retained = autoreleased.clone();
    std::mem::forget(autoreleased);
    retained
}

fn digest_file(path: &PathBuf) -> Option<String> {
    Some(hex::encode(Sha256::digest(fs::read(path).ok()?)))
}

impl PipelineArchive {
    pub fn new(device: &Device, source: &[u8]) -> Self {
        let disabled = || Self {
            archive: None,
            path: None,
            loaded: false,
            added: RefCell::new(HashSet::new()),
        };
        let Some(home) = std::env::var_os("HOME") else {
            return disabled();
        };
        let Some(build) = os_build() else {
            return disabled();
        };
        let directory = PathBuf::from(home).join(".zkminer/metal-cache");
        if fs::create_dir_all(&directory).is_err() {
            return disabled();
        }
        let source_sha = hex::encode(Sha256::digest(source));
        let filename = format!(
            "{}-{}-{}.metallib-archive",
            source_sha,
            safe_key(device.name()),
            build
        );
        let path = directory.join(filename);
        let index = path.with_extension("metallib-archive.index");
        let indexed = fs::read_to_string(&index).ok();
        let (indexed_sha, indexed_names) = indexed
            .as_deref()
            .and_then(|text| text.split_once('\n'))
            .unwrap_or(("", ""));
        let mut loaded =
            digest_file(&path).as_deref() == Some(indexed_sha) && !indexed_sha.is_empty();
        let descriptor = BinaryArchiveDescriptor::new();
        if loaded {
            descriptor.set_url(&file_url(&path));
        }
        let archive = match device.new_binary_archive_with_descriptor(&descriptor) {
            Ok(archive) => archive,
            Err(error) => {
                tracing::warn!(%error, path = %path.display(), "ignoring unreadable Metal binary archive");
                loaded = false;
                let empty = BinaryArchiveDescriptor::new();
                match device.new_binary_archive_with_descriptor(&empty) {
                    Ok(archive) => archive,
                    Err(error) => {
                        tracing::warn!(%error, "Metal binary archive unavailable; compiling normally");
                        return disabled();
                    }
                }
            }
        };
        tracing::info!(path = %path.display(), loaded, %source_sha, "Metal binary archive ready");
        Self {
            archive: Some(archive),
            path: Some(path),
            loaded,
            added: RefCell::new(if loaded {
                indexed_names.lines().map(str::to_owned).collect()
            } else {
                HashSet::new()
            }),
        }
    }

    pub fn pipeline(
        &self,
        device: &Device,
        descriptor: &ComputePipelineDescriptorRef,
    ) -> Result<ComputePipelineState, String> {
        let Some(archive) = &self.archive else {
            return device.new_compute_pipeline_state(descriptor);
        };
        descriptor.set_binary_archives(&[archive]);
        let name = descriptor
            .compute_function()
            .ok_or_else(|| "Metal compute pipeline has no function".to_owned())?
            .name()
            .to_owned();
        let needs_add = !self.added.borrow().contains(&name);
        let added = if needs_add {
            // Archive updates never block a valid proof. The pipeline remains
            // available through Metal's ordinary compilation path on failure.
            match archive.add_compute_pipeline_functions_with_descriptor(descriptor) {
                Ok(true) => true,
                Ok(false) => {
                    tracing::warn!(%name, "Metal declined to add pipeline to archive");
                    false
                }
                Err(error) => {
                    tracing::warn!(%error, %name, "cannot add Metal pipeline to archive");
                    false
                }
            }
        } else {
            false
        };
        let pipeline = device.new_compute_pipeline_state(descriptor)?;
        if added && self.added.borrow_mut().insert(name) {
            if let Some(path) = &self.path {
                let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
                let result = archive
                    .serialize_to_url(&file_url(&temporary))
                    .and_then(|ok| {
                        ok.then_some(())
                            .ok_or_else(|| "archive serialization returned false".to_owned())
                    })
                    .and_then(|()| fs::rename(&temporary, path).map_err(|e| e.to_string()));
                if let Err(error) = result {
                    tracing::warn!(%error, path = %path.display(), "cannot persist Metal binary archive");
                    let _ = fs::remove_file(temporary);
                } else if let Some(archive_sha) = digest_file(path) {
                    let index = path.with_extension("metallib-archive.index");
                    let index_temp = index.with_extension(format!("tmp-{}", std::process::id()));
                    let mut names: Vec<_> = self.added.borrow().iter().cloned().collect();
                    names.sort();
                    let text = format!("{archive_sha}\n{}\n", names.join("\n"));
                    if fs::write(&index_temp, text)
                        .and_then(|()| fs::rename(&index_temp, &index))
                        .is_err()
                    {
                        tracing::warn!(path = %index.display(), "cannot persist Metal archive index");
                        let _ = fs::remove_file(index_temp);
                    }
                }
            }
        }
        Ok(pipeline)
    }
}
