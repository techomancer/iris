//! The IP27 processor bus: a node's memory, its uncached views and the Hub special windows.
//!
//! Two attribute bits beside an uncached address (XKPHYS bits 58:57) pick one of four spaces,
//! which [`crate::sn_nodes::UncachedBus`] indexes. Cached references, and uncached ones to memory
//! (attribute 3), use the node's `Physical` and its ppmem window; the Hub owns and remaps it.
//!
//! # What is approximated here
//!
//! - A node holding zero shows its memory at node-relative offsets. Once it holds another number it
//!   shows its memory there, keeps node zero's below the cached alias, and shows the memory of the
//!   node holding zero above it; on a two-node board each also shows the other's at its number.
//!   Each node reserves 4 GiB (`next_power_of_two(nodes) x 4 GiB` with two) of inaccessible host
//!   address space (`PROT_NONE` and `MAP_NORESERVE`, or a `VirtualAlloc2` placeholder), by reading
//!   the host code: only Linux was run.
//! - The low 128 KiB of the Hub-special memory view is not private to each processor of the pair.
//! - A 512 MiB bank 0 covers the boot flash's reset address: cached references reach the bank, the
//!   uncached reset fetch reaches the flash through the Hub-special view. That the hardware does
//!   the same is inferred (the PROM's memory test walks that address as RAM).
//! - A bank up to 128 MiB is one run at its 512 MiB stride; a larger one is four runs at 128 MiB.
//! - Unfitted memory fails every access with a logged bus error (a software expectation: the IRIX
//!   handler and SGI's event catalogue allow it, no capture shows it). The PROM's sizing reads the
//!   directory back door, which answers a fixed pattern for an absent bank, so boots log no access.
//!
//! Ported from hw/mips/sgi/sn0.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::dev::node_flash::NodeFlash;
pub use crate::dev::hub::{bank_mappings, BANKS, BANK_STRIDE, SLOT_STRIDE};
use crate::sn_nodes::RESET_WINDOW;
use crate::window_map::SparseWindowMap;
use crate::traits::*;
use std::sync::Arc;

/// The ranges the CPU model reports as memory for a node with `banks_mb`: the
/// fitted runs, and the boot flash's reset address unless a 512 MiB bank 0 holds it.
pub fn memory_map(banks_mb: [u32; BANKS]) -> crate::memory_map::MemoryMap {
    use crate::memory_map::{MemKind, MemRange, MemoryMap};
    const NAMES: [&str; BANKS] = ["bank 0", "bank 1", "bank 2", "bank 3"];
    let end = RESET_WINDOW + crate::dev::node_flash::SIZE as u64;
    let runs = crate::dev::hub::fitted_mappings(banks_mb);
    let mut ranges: Vec<_> = runs.iter().map(|r| MemRange::new(NAMES[(r.base / BANK_STRIDE) as usize], MemKind::Ram, r.base, r.end)).collect();
    if !runs.iter().any(|r| r.base < end && r.end > RESET_WINDOW) {
        ranges.push(MemRange::new("boot flash", MemKind::Rom, RESET_WINDOW, end));
    }
    MemoryMap::new(ranges)
}

/// The Hub-special space of a node: its own memory and boot flash and the
/// named node's, and the two back doors.
pub fn hub_special(memory: Arc<dyn BusDevice>, flash: &Arc<NodeFlash>, ecc: Arc<dyn BusDevice>, directory: Arc<dyn BusDevice>) -> Arc<dyn BusDevice> {
    const WINDOW: u64 = 256 << 20;
    const BACK_DOOR: u64 = 1 << 30;
    let mut map = SparseWindowMap::default();
    for base in [0, 0x2000_0000] {
        map.map(base, WINDOW, memory.clone());
    }
    for base in [0x1000_0000, 0x3000_0000] {
        map.map(base, WINDOW, flash.window());
    }
    map.map(0x8000_0000, BACK_DOOR, ecc);
    map.map(0xc000_0000, BACK_DOOR, directory);
    Arc::new(map)
}


#[cfg(test)]
mod tests;
