//! JIT code cache: maps physical PCs to compiled native code blocks.

use std::collections::HashMap;

/// A compiled native code block.
pub struct CompiledBlock {
    /// Function pointer to compiled native code.
    pub entry: *const u8,
    /// Physical address this block starts at.
    pub phys_addr: u64,
    /// Virtual address (for diagnostics).
    pub virt_addr: u64,
    /// Number of MIPS instructions in this block.
    pub len_mips: u32,
    /// Size of native code in bytes.
    pub len_native: u32,
}

// Safety: CompiledBlock is only accessed from the CPU thread.
unsafe impl Send for CompiledBlock {}

/// Code cache keyed by physical PC (aligned to 4 bytes).
pub struct CodeCache {
    blocks: HashMap<u64, CompiledBlock>,
}

impl CodeCache {
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
        }
    }

    pub fn lookup(&self, phys_pc: u64) -> Option<&CompiledBlock> {
        self.blocks.get(&phys_pc)
    }

    pub fn insert(&mut self, phys_pc: u64, block: CompiledBlock) {
        self.blocks.insert(phys_pc, block);
    }

    /// Invalidate all blocks that overlap a physical address range.
    /// Called when self-modifying code is detected or CACHE instruction executes.
    pub fn invalidate_range(&mut self, phys_start: u64, phys_end: u64) {
        self.blocks.retain(|&addr, block| {
            let block_end = addr + (block.len_mips as u64 * 4);
            addr >= phys_end || block_end <= phys_start
        });
    }

    /// Invalidate everything (used on TLB flush or mode change).
    pub fn invalidate_all(&mut self) {
        self.blocks.clear();
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }
}
