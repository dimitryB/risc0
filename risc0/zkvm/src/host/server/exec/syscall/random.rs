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

use anyhow::Result;
use risc0_zkvm_platform::WORD_SIZE;
use sha2::{Digest as _, Sha256};

use super::{Syscall, SyscallContext};

/// The guest's `sys_rand`: fresh host entropy, or a stream fixed by the
/// [ExecutorEnv][crate::ExecutorEnv]'s random seed.
pub(crate) enum SysRandom {
    Entropy,
    Seeded(SeedStream),
}

impl SysRandom {
    pub(crate) fn new(seed: Option<[u8; 32]>) -> Self {
        match seed {
            Some(seed) => Self::Seeded(SeedStream::new(seed)),
            None => Self::Entropy,
        }
    }

    fn fill(&mut self, to_guest: &mut [u32]) {
        match self {
            Self::Entropy => {
                let mut rand_buf = vec![0u8; to_guest.len() * WORD_SIZE];
                rand::fill(rand_buf.as_mut_slice());
                bytemuck::cast_slice_mut(to_guest).clone_from_slice(rand_buf.as_slice());
            }
            Self::Seeded(stream) => {
                for word in to_guest.iter_mut() {
                    let mut bytes = [0u8; WORD_SIZE];
                    stream.read(&mut bytes);
                    *word = u32::from_le_bytes(bytes);
                }
            }
        }
    }
}

impl Syscall for SysRandom {
    fn syscall(
        &mut self,
        _syscall: &str,
        _ctx: &mut dyn SyscallContext,
        to_guest: &mut [u32],
    ) -> Result<(u32, u32)> {
        tracing::debug!("SYS_RANDOM: {}", to_guest.len());
        self.fill(to_guest);
        Ok(((to_guest.len() * WORD_SIZE) as u32, 0))
    }
}

/// SHA-256 in counter mode: block `i` is `SHA-256(DOMAIN || seed || i)`, with `i` a little-endian
/// u64, and the stream is the blocks in order. Bytes are handed out in sequence, so what the guest
/// receives depends only on the seed and on how many bytes it asked for before, not on how it split
/// its requests.
pub(crate) struct SeedStream {
    seed: [u8; 32],
    counter: u64,
    block: [u8; 32],
    used: usize,
}

impl SeedStream {
    const DOMAIN: &'static [u8] = b"risc0-zkvm/sys_rand/v1";

    pub(crate) fn new(seed: [u8; 32]) -> Self {
        Self {
            seed,
            counter: 0,
            block: [0; 32],
            used: 32,
        }
    }

    pub(crate) fn read(&mut self, out: &mut [u8]) {
        for byte in out {
            if self.used == self.block.len() {
                self.block = Sha256::new()
                    .chain_update(Self::DOMAIN)
                    .chain_update(self.seed)
                    .chain_update(self.counter.to_le_bytes())
                    .finalize()
                    .into();
                self.counter += 1;
                self.used = 0;
            }
            *byte = self.block[self.used];
            self.used += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SeedStream, SysRandom};

    const SEED: [u8; 32] = {
        let mut seed = [0u8; 32];
        let mut i = 0;
        while i < 32 {
            seed[i] = i as u8;
            i += 1;
        }
        seed
    };

    #[test]
    fn seed_stream_known_answer() {
        // SHA-256("risc0-zkvm/sys_rand/v1" || 00 01 .. 1f || i as u64 LE), from Python's hashlib.
        let mut out = [0u8; 64];
        SeedStream::new(SEED).read(&mut out);
        assert_eq!(
            hex::encode(&out[..32]),
            "a08341db7e9bc2920b575b9456a7303495d680127eaf562806c9ecc084d1fdf8"
        );
        assert_eq!(
            hex::encode(&out[32..]),
            "4e52899ecec1b8bbb137cf1085857d0de30cd046ea931f846310798b7b89b8c6"
        );
    }

    #[test]
    fn seed_stream_ignores_request_boundaries() {
        let mut whole = [0u8; 100];
        SeedStream::new(SEED).read(&mut whole);

        let mut stream = SeedStream::new(SEED);
        let mut pieces = Vec::new();
        for len in [0, 3, 29, 1, 32, 35] {
            let mut piece = vec![0u8; len];
            stream.read(&mut piece);
            pieces.extend(piece);
        }
        assert_eq!(pieces, whole);
    }

    #[test]
    fn seeded_words_are_the_stream_little_endian() {
        let mut bytes = [0u8; 40];
        SeedStream::new(SEED).read(&mut bytes);
        let mut words = [0u32; 10];
        SysRandom::new(Some(SEED)).fill(&mut words);
        for (word, chunk) in words.iter().zip(bytes.chunks(4)) {
            assert_eq!(*word, u32::from_le_bytes(chunk.try_into().unwrap()));
        }
    }

    #[test]
    fn seeded_repeats_and_depends_on_the_seed() {
        let fill = |seed| {
            let mut words = [0u32; 16];
            SysRandom::new(seed).fill(&mut words);
            words
        };
        assert_eq!(fill(Some(SEED)), fill(Some(SEED)));
        let mut other = SEED;
        other[31] ^= 1;
        assert_ne!(fill(Some(SEED)), fill(Some(other)));
        // Without a seed, every call is fresh entropy.
        assert_ne!(fill(None), fill(None));
    }
}
