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

use risc0_binfmt::WordAddr;

const ENTRY_COUNT: usize = 1 << 11;
const PAGED_MAP_MASK: u32 = (1 << 10) - 1;

struct PagedMapEntry {
    words: Vec<Option<u32>>,
}

impl Default for PagedMapEntry {
    fn default() -> Self {
        Self {
            words: vec![None; ENTRY_COUNT],
        }
    }
}

struct PagedMapTable {
    entries: Vec<u32>,
}

impl Default for PagedMapTable {
    fn default() -> Self {
        Self {
            entries: vec![0; ENTRY_COUNT],
        }
    }
}

pub(crate) struct PagedMap {
    high: PagedMapTable,
    mid: Vec<PagedMapTable>,
    low: Vec<PagedMapEntry>,
}

impl Default for PagedMap {
    fn default() -> Self {
        Self {
            high: PagedMapTable::default(),
            mid: vec![PagedMapTable {
                entries: Vec::new(),
            }],
            low: vec![PagedMapEntry { words: Vec::new() }],
        }
    }
}

impl PagedMap {
    /// Returns the value corresponding to the key.
    #[inline(always)]
    pub fn get(&mut self, addr: &WordAddr) -> Option<u32> {
        let idx = addr.0 >> 20;
        let mid = self.high.entries.get(idx as usize).unwrap();
        if *mid != 0 {
            let mid_table = self.mid.get(*mid as usize).unwrap();
            let idx = (addr.0 >> 10) & PAGED_MAP_MASK;
            let low = mid_table.entries.get(idx as usize).unwrap();
            if *low != 0 {
                let low_table = self.low.get(*low as usize).unwrap();
                let idx = addr.0 & PAGED_MAP_MASK;
                return *low_table.words.get(idx as usize).unwrap();
            }
        }
        None
    }

    /// Returns a mutable reference to the value corresponding to the key.
    #[inline(always)]
    pub fn get_mut(&mut self, addr: &WordAddr) -> &mut Option<u32> {
        let high_idx = addr.0 >> 20;
        let mid_idx = (addr.0 >> 10) & PAGED_MAP_MASK;
        let low_idx = addr.0 & PAGED_MAP_MASK;
        // tracing::debug!("insert: {addr:?}, {high_idx:#010x}, {mid_idx:#010x}, {low_idx:#010x}");
        let mid = self.high.entries.get_mut(high_idx as usize).unwrap();
        if *mid == 0 {
            *mid = self.mid.len() as u32;
            self.mid.push(PagedMapTable::default());
        }
        let mid_table = self.mid.get_mut(*mid as usize).unwrap();
        let low = mid_table.entries.get_mut(mid_idx as usize).unwrap();
        if *low == 0 {
            *low = self.low.len() as u32;
            self.low.push(PagedMapEntry::default());
        }
        let low_table = self.low.get_mut(*low as usize).unwrap();
        low_table.words.get_mut(low_idx as usize).unwrap()
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, None is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old value is returned.
    pub fn insert(&mut self, addr: &WordAddr, word: u32) -> Option<u32> {
        self.get_mut(addr).replace(word)
    }
}

/// Per-word preflight bookkeeping in one map: the word's value at its first recorded
/// load (`orig`, the old `orig_words` map) and the cycle of its last transaction (`prev`, the
/// old `prev_cycle` map). The body touched both maps on every recorded load; one lookup now
/// serves both, and the wrap pass reads both with one lookup too.
#[derive(Clone, Copy)]
pub(crate) struct WordTxn {
    /// Last transaction cycle, `u32::MAX` if none (cycles are < 2^31).
    pub prev: u32,
    /// Value at the first recorded load, valid if `has_orig`.
    pub orig: u32,
    pub has_orig: bool,
}

impl Default for WordTxn {
    fn default() -> Self {
        Self {
            prev: u32::MAX,
            orig: 0,
            has_orig: false,
        }
    }
}

const TXN_LOW_BITS: u32 = 10;
const TXN_LOW: usize = 1 << TXN_LOW_BITS;

/// Two-level map from word address to [`WordTxn`]: a directly indexed table of low-table
/// indices (by address >> 10), then 1024-entry low tables. Maps are recycled across segments
/// ([`WordTxnMap::take`] / [`WordTxnMap::recycle`]): a fresh 16 MB index table per segment cost
/// the preflight thread page faults and an munmap (up to ~10 ms on a loaded host).
pub(crate) struct WordTxnMap {
    high: Vec<u32>,
    low: Vec<Box<[WordTxn; TXN_LOW]>>,
    /// Low tables in use (index 0 is the "absent" marker); the rest are kept for reuse.
    used: usize,
    /// High-table entries set since the last reset.
    touched: Vec<u32>,
}

static WORD_TXN_POOL: std::sync::Mutex<Vec<WordTxnMap>> = std::sync::Mutex::new(Vec::new());

impl WordTxnMap {
    fn new() -> Self {
        Self {
            high: vec![0; 1 << (32 - TXN_LOW_BITS)],
            low: vec![Box::new([WordTxn::default(); TXN_LOW])],
            used: 1,
            touched: Vec::new(),
        }
    }

    /// An empty map from the pool (or a new one).
    pub fn take() -> Self {
        WORD_TXN_POOL
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(Self::new)
    }

    /// A placeholder without allocations (only valid to drop or recycle).
    pub fn placeholder() -> Self {
        Self {
            high: Vec::new(),
            low: Vec::new(),
            used: 0,
            touched: Vec::new(),
        }
    }

    /// Drop the pooled maps (see `cuda_stage::release`). Never blocks.
    #[cfg_attr(not(any(feature = "cuda", feature = "rocm")), allow(dead_code))]
    pub fn trim_pool() {
        if let Ok(mut pool) = WORD_TXN_POOL.try_lock() {
            pool.clear();
        }
    }

    /// Reset this map and return it to the pool.
    pub fn recycle(mut self) {
        if self.high.is_empty() {
            return;
        }
        for hi in self.touched.drain(..) {
            self.high[hi as usize] = 0;
        }
        self.used = 1;
        let mut pool = WORD_TXN_POOL.lock().unwrap();
        if pool.len() < 4 {
            pool.push(self);
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, addr: &WordAddr) -> &mut WordTxn {
        let hi = (addr.0 >> TXN_LOW_BITS) as usize;
        let lo = addr.0 as usize & (TXN_LOW - 1);
        let mut idx = self.high[hi] as usize;
        if idx == 0 {
            idx = self.new_low(hi);
        }
        &mut self.low[idx][lo]
    }

    #[cold]
    fn new_low(&mut self, hi: usize) -> usize {
        let idx = self.used;
        if idx == self.low.len() {
            self.low.push(Box::new([WordTxn::default(); TXN_LOW]));
        } else {
            self.low[idx].fill(WordTxn::default());
        }
        self.used += 1;
        self.high[hi] = idx as u32;
        self.touched.push(hi as u32);
        idx
    }

    #[inline(always)]
    pub fn get(&self, addr: &WordAddr) -> Option<&WordTxn> {
        let idx = self.high[(addr.0 >> TXN_LOW_BITS) as usize] as usize;
        (idx != 0).then(|| &self.low[idx][addr.0 as usize & (TXN_LOW - 1)])
    }
}
