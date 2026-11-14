//! The IP35 node: the Bedrock chip with its memory and boot flash, the
//! processor bus over its uncached spaces ([`crate::sn_nodes::UncachedBus`])
//! and the reset PROM's window, shared by the Fuel, the Tezro and the
//! C-bricks of the Origin 3200 and Onyx 3000.
//!
//! Ported from hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::dev::bedrock::RamExtent;
use crate::xtalk::XtalkHost;

pub(crate) use crate::window_map::{map_device, SparseWindowMap, Window, CPU_PHYS_MASK};

pub(crate) use crate::sn_nodes::RESET_WINDOW;
/// The reset PROM's compatibility window: the processor's reset vector and
/// the PROM's own code, fetched uncached.
pub(crate) use crate::dev::bedrock::RESET_WINDOW_SIZE;

/// Bedrock's crossbar port: widget 10, taking interrupt packets at any
/// address (the address picks the processor interface).
pub(crate) const BEDROCK_PORT: XtalkHost =
    XtalkHost { widget: crate::dev::xbridge::HOST_WIDGET, int_packet: 0, int_packet_mask: 0, vectors: 256 };

/// Where a Bedrock node has memory: each 1 GiB software slot `extent` says
/// can hold RAM, as far as its largest bank reaches, and the reset PROM.
/// Which bank lands in which slot of a group of four depends on how the PROM
/// programs the memory controller; the Fuel's two 256 MiB banks can be in any
/// of the first four, and a Tezro's banks 4 and 5 in any of the next four. A
/// slot 0 over 508 MiB reaches the PROM's range, so its RAM is listed on either
/// side.
pub fn memory_map(extent: &RamExtent) -> crate::memory_map::MemoryMap {
    use crate::dev::bedrock::{BANK_SPAN, SLOTS};
    use crate::memory_map::{MemKind, MemRange, MemoryMap};
    const NAMES: [&str; SLOTS] =
        ["ram slot 0", "ram slot 1", "ram slot 2", "ram slot 3", "ram slot 4", "ram slot 5", "ram slot 6", "ram slot 7"];
    const PROM: (u64, u64) = (RESET_WINDOW, RESET_WINDOW + RESET_WINDOW_SIZE);
    let mut ranges = vec![MemRange::new("prom", MemKind::Rom, PROM.0, PROM.1)];
    for (slot, &name) in NAMES.iter().enumerate() {
        let (base, end) = (slot as u64 * BANK_SPAN, slot as u64 * BANK_SPAN + extent.prefix(slot));
        if base == end {
            continue;
        }
        if base < PROM.1 && end > PROM.0 {
            ranges.push(MemRange::new(name, MemKind::Ram, base, PROM.0));
            if end > PROM.1 {
                ranges.push(MemRange::new("ram slot 0 above the prom", MemKind::Ram, PROM.1, end));
            }
        } else {
            ranges.push(MemRange::new(name, MemKind::Ram, base, end));
        }
    }
    MemoryMap::new(ranges)
}

pub(crate) mod node;
pub(crate) mod presence;
pub(crate) mod serial;
#[cfg(test)]
mod tests;
