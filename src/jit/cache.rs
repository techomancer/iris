//! JIT code cache: maps physical PCs to compiled native code blocks.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum BlockTier {
    Alu   = 0,  // ALU + branches only, no memory helper calls
    Loads = 1,  // ALU + loads + branches
    Full  = 2,  // ALU + loads + stores + branches
}

impl BlockTier {
    pub fn promote(self) -> Option<BlockTier> {
        match self {
            BlockTier::Alu   => Some(BlockTier::Loads),
            BlockTier::Loads => Some(BlockTier::Full),
            BlockTier::Full  => None,
        }
    }
    pub fn demote(self) -> Option<BlockTier> {
        match self {
            BlockTier::Alu   => None,
            BlockTier::Loads => Some(BlockTier::Alu),
            BlockTier::Full  => Some(BlockTier::Loads),
        }
    }
}

// Defaults; overridden by IRIS_JIT_STABLE / IRIS_JIT_PROMOTE / IRIS_JIT_DEMOTE env vars.
pub const TIER_STABLE_THRESHOLD:  u32 = 50;   // consecutive clean exits → trusted
pub const TIER_PROMOTE_THRESHOLD: u32 = 200;  // trusted clean exits → try next tier
pub const TIER_DEMOTE_THRESHOLD:  u32 = 3;    // exceptions in trial period → demote

/// Runtime-configurable tier thresholds. Reads env vars once at init.
pub struct TierConfig {
    pub stable:  u32,
    pub promote: u32,
    pub demote:  u32,
}

impl TierConfig {
    pub fn from_env() -> Self {
        Self {
            stable:  std::env::var("IRIS_JIT_STABLE").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(TIER_STABLE_THRESHOLD),
            promote: std::env::var("IRIS_JIT_PROMOTE").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(TIER_PROMOTE_THRESHOLD),
            demote:  std::env::var("IRIS_JIT_DEMOTE").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(TIER_DEMOTE_THRESHOLD),
        }
    }
}

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
    /// Compilation tier for this block.
    pub tier:            BlockTier,
    /// Total number of times this block has been entered.
    pub hit_count:       u32,
    /// Number of exceptions that occurred during this block's execution.
    pub exception_count: u32,
    /// Consecutive clean (non-exception) exits since last exception or tier change.
    pub stable_hits:     u32,
    /// True when this block is in a trial period (not yet fully trusted at current tier).
    pub speculative:     bool,
    /// FNV-1a hash of the raw instruction words; used to detect stale profile
    /// entries when a different DSO is loaded at the same virtual address.
    pub content_hash:    u32,
}

// Safety: CompiledBlock is only accessed from the CPU thread.
unsafe impl Send for CompiledBlock {}

/// Code cache keyed by (physical PC, virtual PC).
///
/// Physical PC alone is insufficient: compiled blocks bake virtual PC constants
/// for exit PC and branch targets. When different virtual addresses map to the
/// same physical page (shared libraries, fork), a block compiled for virtual
/// address A would produce wrong exit PCs when executed at virtual address B.
/// Including the virtual PC in the key ensures each virtual mapping gets its
/// own correctly-compiled block.
pub struct CodeCache {
    blocks: HashMap<(u64, u64), CompiledBlock>,
}

impl CodeCache {
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
        }
    }

    pub fn lookup(&self, phys_pc: u64, virt_pc: u64) -> Option<&CompiledBlock> {
        self.blocks.get(&(phys_pc, virt_pc))
    }

    pub fn contains(&self, phys_pc: u64, virt_pc: u64) -> bool {
        self.blocks.contains_key(&(phys_pc, virt_pc))
    }

    pub fn lookup_mut(&mut self, phys_pc: u64, virt_pc: u64) -> Option<&mut CompiledBlock> {
        self.blocks.get_mut(&(phys_pc, virt_pc))
    }

    pub fn insert(&mut self, phys_pc: u64, virt_pc: u64, block: CompiledBlock) {
        self.blocks.insert((phys_pc, virt_pc), block);
    }

    pub fn replace(&mut self, phys_pc: u64, virt_pc: u64, block: CompiledBlock) {
        self.blocks.insert((phys_pc, virt_pc), block);
    }

    /// Invalidate all blocks that overlap a physical address range.
    /// Called when self-modifying code is detected or CACHE instruction executes.
    pub fn invalidate_range(&mut self, phys_start: u64, phys_end: u64) {
        self.blocks.retain(|&(addr, _), block| {
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

    pub fn iter(&self) -> impl Iterator<Item = (&(u64, u64), &CompiledBlock)> {
        self.blocks.iter()
    }
}
