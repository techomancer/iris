//! A Bedrock node's fitted memory: banks owned by `Physical`, decoded by the memory configuration
//! register.
//!
//! Each bank answers in its own 1 GiB slot, so banks 4 to 7 sit at 4 to 8 GiB; a node with banks
//! there asks ppmem for its 16 GiB window. Accesses reach a bank without a lock.
//!
//! Ported from hw/misc/sgi-bedrock-md.c in jamesbraid/qemu-sgi-origami.

use super::regs::md::{self, MEMORY_CONFIG as cfg};
use crate::physical::{BankMapping, Physical, RamBank};
use crate::ppmem::{BITMAP_SHIFT, LARGE_WINDOW_SIZE, WINDOW_SIZE};
use crate::sn_nodes::{UnfittedMemory, SN1_NODE_SHIFT};
use crate::traits::*;
use std::sync::Arc;

pub const BANK_SPAN: u64 = md::SOFTWARE_BANK_BYTES;
/// Software slots of [`BANK_SPAN`] in the node's 8 GiB.
pub const SLOTS: usize = md::SOFTWARE_BANKS as usize;

/// MEMORY_CONFIG holds one group of bits per DIMM position, two banks to a
/// position, from position 0.
fn position(config: u64, bank: u64) -> u64 {
    config >> (bank / 2 * u64::from(cfg::POS1_SIZE::SHIFT - cfg::POS0_SIZE::SHIFT))
}
/// The bank select: software slot `s` reaches bank `s ^ bank_select(config)`.
pub(super) fn bank_select(config: u64) -> u64 {
    (config & cfg::BANK_SELECT::MASK) >> cfg::BANK_SELECT::SHIFT
}
/// The size code of `bank`'s DIMM position.
pub(super) fn size_code(config: u64, bank: u64) -> u64 {
    (position(config, bank) & cfg::POS0_SIZE::MASK) >> cfg::POS0_SIZE::SHIFT
}
/// Whether the size code names a size: a code the map lists, other than empty.
/// The list is the hardware's codes, so `None` is a real unused encoding.
pub(super) fn sized(config: u64, bank: u64) -> bool {
    cfg::POS0_SIZE::Value::from_bits(position(config, bank)).is_some_and(|v| v != cfg::POS0_SIZE::Value::EMPTY)
}
/// Whether `bank`'s enable bit is set.
pub(super) fn enabled(config: u64, bank: u64) -> bool {
    position(config, bank) & (1 << (u64::from(cfg::POS0_ENABLES::SHIFT) + (bank & 1))) != 0
}
/// The bytes a size code names: 64 MiB at code 1, doubling to 1 GiB.
pub(super) fn code_bytes(code: u64) -> u64 {
    (32 << 20) << code
}
/// The most memory a bank can hold: what the largest size code names.
const MAX_BANK_MIB: u32 = 1024;
/// Banks are mapped into the direct window in whole regions of this size.
const REGION_MIB: u32 = 1 << (BITMAP_SHIFT - 20);

/// Where fitted RAM can answer: per software slot, the most any bank the swizzle can place there
/// holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamExtent([u64; SLOTS]);
impl RamExtent {
    /// `banks` are the fitted sizes in MiB from locational bank 0.
    pub fn new(banks: &[u32]) -> Self {
        let fitted = |bank: usize| banks.get(bank).map_or(0, |&mib| u64::from(mib) << 20);
        Self(std::array::from_fn(|slot| {
            (0..4).map(|group_slot| fitted(slot / 4 * 4 + group_slot)).max().unwrap_or(0)
        }))
    }
    /// Whether `address` can be fitted RAM under some bank-select setting.
    pub fn contains(&self, address: u64) -> bool {
        let slot = address / BANK_SPAN;
        slot < SLOTS as u64 && address % BANK_SPAN < self.0[slot as usize]
    }
    /// Whether no slot can hold RAM.
    pub fn is_empty(&self) -> bool { self.0.iter().all(|&size| size == 0) }
    /// The most RAM software slot `slot` can hold, in bytes.
    pub fn prefix(&self, slot: usize) -> u64 { self.0[slot] }
}

/// The locational bank software slot `slot` holds under `config`.
pub(super) fn bank_of(config: u64, slot: u64) -> u64 {
    slot ^ bank_select(config)
}
/// Bytes bank `bank`'s lane enables under `config`, zero for a bank turned off.
pub(super) fn bank_bytes(config: u64, bank: u64) -> u64 {
    if bank < SLOTS as u64 && enabled(config, bank) && sized(config, bank) { code_bytes(size_code(config, bank)) } else { 0 }
}
/// Whether two words place and size the banks differently.
pub(super) fn layout_differs(old: u64, new: u64) -> bool {
    bank_select(old) != bank_select(new) || (0..SLOTS as u64).any(|bank| bank_bytes(old, bank) != bank_bytes(new, bank))
}

/// Whether `banks` are a memory the controller can decode: at most [`SLOTS`], none empty or over 1
/// GiB, a whole number of direct-window regions, and banks 2k and 2k+1 equal (one size code per
/// lane, from the PROM's decode).
pub fn check_banks(banks: &[u32]) -> Result<(), String> {
    if banks.is_empty() || banks.len() > SLOTS {
        return Err(format!("memory: {} banks; fit 1 to {SLOTS}", banks.len()));
    }
    if let Some((bank, mib)) = banks.iter().enumerate()
        .find(|(_, &mib)| mib == 0 || mib > MAX_BANK_MIB || mib % REGION_MIB != 0)
    {
        return Err(format!(
            "memory: bank {bank} is {mib} MiB; a bank is a multiple of {REGION_MIB} MiB up to {MAX_BANK_MIB} MiB"));
    }
    // The configuration word has one size code per byte lane, shared by
    // the lane's two banks, so the two cannot differ.
    if let Some(lane) = banks.chunks(2).position(|pair| pair.len() == 2 && pair[0] != pair[1]) {
        return Err(format!(
            "memory: banks {} and {} are {} and {} MiB; the two banks of a lane share one size",
            2 * lane, 2 * lane + 1, banks[2 * lane], banks[2 * lane + 1]));
    }
    Ok(())
}

pub struct Memory {
    physical: Arc<Physical>,
    /// Fitted bytes in each locational bank, bank 0 first. Physical's banks
    /// are these, in the same order.
    fitted: Vec<u64>,
    /// Each processor's window while its PROM cache lines stand.
    extent: RamExtent,
}
impl Memory {
    /// Memory with `banks` fitted, locational bank 0 upward, each a size in
    /// MiB that [`check_banks`] accepts, on a board of `nodes` nodes.
    pub fn new(banks: &[u32], nodes: usize) -> Result<Self, String> {
        check_banks(banks)?;
        // Bank n starts at n GiB; past bank 3 the node runs over 4 GiB.
        let window = if banks.len() as u64 * BANK_SPAN > WINDOW_SIZE { LARGE_WINDOW_SIZE } else { WINDOW_SIZE };
        // A board of several nodes shows every node's memory at its number,
        // above the node's own, so the window has room for the numbers.
        let window = window.max(if nodes > 1 { (nodes.next_power_of_two() as u64) << SN1_NODE_SHIFT } else { 0 });
        let physical = Physical::with_window(banks.iter().map(|&mib| RamBank::new(mib as usize)).collect(), window);
        physical.set_default(Arc::new(UnfittedMemory(crate::devlog::LogModule::Bedrock)), "unfitted memory");
        Ok(Self { physical, fitted: banks.iter().map(|&mib| u64::from(mib) << 20).collect(), extent: RamExtent::new(banks) })
    }
    /// Fitted bytes in locational bank `bank`; zero where nothing is fitted.
    pub(crate) fn fitted(&self, bank: u64) -> u64 {
        self.fitted.get(bank as usize).copied().unwrap_or(0)
    }
    /// Where the fitted banks start in a dense walk of all of them; sparse tables key by it.
    pub(crate) fn dense_base(&self, bank: u64) -> u64 {
        self.fitted.iter().take(bank as usize).sum()
    }
    pub fn extent(&self) -> RamExtent { self.extent }
    pub fn physical(&self) -> &Arc<Physical> {
        &self.physical
    }
    /// The mappings `config` places, for the remap: each software slot's bank,
    /// as far as both its lane and the fitted memory reach.
    pub(crate) fn mappings(&self, config: u64) -> Arc<[BankMapping]> {
        (0..SLOTS as u64)
            .filter_map(|slot| {
                let (bank, _, len) = self.run(config, slot * BANK_SPAN)?;
                Some(BankMapping { base: slot * BANK_SPAN, end: slot * BANK_SPAN + len, bank, src: 0 })
            })
            .collect()
    }
    pub(crate) fn decode(&self, config: u64, addr: u64, bytes: u64) -> Option<(usize, u64)> {
        self.run(config, addr).and_then(|(bank, offset, left)| (bytes <= left).then_some((bank, offset)))
    }
    /// The run of fitted memory holding `addr`: its bank, the offset in it and the bytes to its end
    /// (never crossing a slot).
    pub(crate) fn run(&self, config: u64, addr: u64) -> Option<(usize, u64, u64)> {
        let software = addr / BANK_SPAN;
        if software >= SLOTS as u64 {
            return None;
        }
        let bank = bank_of(config, software);
        let limit = self.fitted(bank).min(bank_bytes(config, bank));
        let offset = addr % BANK_SPAN;
        (offset < limit).then(|| (bank as usize, offset, limit - offset))
    }
    pub(crate) fn raw_read(&self, bank: usize, at: u64) -> u64 {
        self.physical.banks()[bank].read64(at).data
    }
    /// Self-test's zero fill of `size` bytes of bank `bank`. The chip runs it
    /// with every other processor parked, as a remap.
    pub(crate) fn zero(&self, bank: usize, size: u64) {
        assert!(size <= self.fitted(bank as u64));
        let bank = &self.physical.banks()[bank];
        // SAFETY: the bank owns at least `size` bytes from its host pointer, zero is zero in
        // either byte order, and no other processor runs meanwhile.
        unsafe {
            std::ptr::write_bytes(bank.host_ptr(), 0, size as usize);
        }
        #[cfg(feature = "jitv2")]
        bank.bump_gen_all();
    }
}
