//! The system id and serial numbers of an emulated IP35 machine.
//!
//! A machine has one observable, the 32-bit system id that IRIX reports
//! through `sysinfo -s` and FLEXlm's `lmhostid`. `[machine] emu_id`, 0 to
//! 4095, mints it in the emulator layout, a pure function of the kind, the
//! `emu_id` and a small sub-index (a node or port), with no hashing. The kind
//! is a digit that marks an id as minted by an emulator, and iris's is 2:
//!
//! - The system serial every L1 of a Tezro answers with is `LEMU`, the kind
//!   digit, the `emu_id` as four digits and `0`.
//! - The default station address is `08:00:69:E<kind>:<emu_id >> 4>:
//!   <emu_id & 15, sub-index>`. The sub-index takes the low nibble of the
//!   last byte, so an address derived from the default by adding a node or
//!   port number up to 15 never reaches the id.
//!
//! IRIX builds the id from a text serial by taking the low nibble of each of
//! its first ten characters and keeping the low 32 bits, or from the last
//! four bytes of the station address.
//!
//! These are synthetic values, not any machine's factory identity.
//!
//! Ported from include/hw/mips/sgi/sgi-board-identity.h and
//! hw/mips/sgi/board-mac.c in jamesbraid/qemu-sgi-origami.

/// The kind of every machine iris builds.
pub const KIND: u32 = 2;

/// The `emu_id` is 0 (the default) or 1 to this.
pub const EMU_ID_MAX: u32 = 4095;

/// Nodes or ports a default station address can number, 0 to 15.
pub const SUB_INDEX_COUNT: u32 = 16;

/// `LEMU`, the kind digit, four `emu_id` digits and `0`: the system serial an
/// L1 answers with. It names the system, not a brick, so every L1 of a
/// machine gives the same text.
pub fn system_serial(kind: u32, emu_id: u32) -> String {
    assert!((1..=9).contains(&kind) && emu_id <= EMU_ID_MAX);
    format!("LEMU{kind}{emu_id:04}0")
}

/// The default station address of node or port `sub`: `08:00:69:E<kind>`, the
/// `emu_id`'s high 8 bits, then its low 4 bits over `sub`. `None` when `sub`
/// is 16 or more.
pub fn default_mac(kind: u32, emu_id: u32, sub: u32) -> Option<[u8; 6]> {
    assert!((1..=15).contains(&kind) && emu_id <= EMU_ID_MAX);
    (sub < SUB_INDEX_COUNT).then(|| [0x08, 0x00, 0x69, 0xe0 | kind as u8, (emu_id >> 4) as u8, ((emu_id & 0xf) << 4 | sub) as u8])
}

/// The id IRIX computes from a text serial: the low nibble of each of the
/// first ten characters shifted in, the low 32 bits kept. A shorter text folds
/// in fewer nibbles; an empty one gives 0.
pub fn hostid_from_text(text: &str) -> u32 {
    let id = text.bytes().take(10).fold(0u64, |id, c| id << 4 | u64::from(c & 0xf));
    id as u32
}

/// The id IRIX takes from a station address: its last four bytes.
pub fn hostid_from_mac(mac: &[u8; 6]) -> u32 {
    u32::from_be_bytes([mac[2], mac[3], mac[4], mac[5]])
}

#[cfg(test)]
mod tests;
