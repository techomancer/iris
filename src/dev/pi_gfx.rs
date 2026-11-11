//! Graphics flow control in the Hub and Bedrock processor interfaces. The
//! Octane has no processor-interface link (its board's `pi` stays `None`),
//! so on that machine the credit calls are no-ops. `PiGfx` is the register
//! block a node's processor interface embeds; its functions are pure state.
//! The credit-link trait and the store-mark helpers the Fuel's graphics board
//! uses are not carried here.
//!
//! Ported from hw/mips/sgi/pi-gfx.c in jamesbraid/qemu-sgi-origami.

use crate::devlog::LogModule;

pub(super) const PI_GFX_SLOTS: usize = 2;
pub const PI_GFX_REGS: u32 = 10;

/// The five registers of one slot, in address order.
pub(super) const PI_GFX_PAGE: u32 = 0;
pub(super) const PI_GFX_CREDIT: u32 = 1;
pub(super) const PI_GFX_BIAS: u32 = 2;
pub(super) const PI_GFX_INT_COUNT: u32 = 3;
pub(super) const PI_GFX_SLOT_REGS: u32 = 5;

/// Page: bit 40 enables it, bits 39:17 name it; the rest is reserved.
pub(super) const PI_GFX_PAGE_ENABLE: u64 = 1 << 40;
pub(super) const PI_GFX_PAGE_ADDRESS: u64 = 0xfffffe0000;
pub(super) const PI_GFX_PAGE_MASK: u64 = PI_GFX_PAGE_ENABLE | PI_GFX_PAGE_ADDRESS;
pub(super) const PI_GFX_CREDIT_MASK: u32 = 0xfff;
pub(super) const PI_GFX_BIAS_MASK: u32 = 0xfff;
pub(super) const PI_GFX_INT_MASK: u32 = 0x3ffffff;

#[derive(Clone, Copy, Default)]
pub struct PiGfx {
    pub page: [u64; PI_GFX_SLOTS],
    pub credit: [u32; PI_GFX_SLOTS],
    pub bias: [u32; PI_GFX_SLOTS],
    pub int_count: [u32; PI_GFX_SLOTS],
    pub int_compare: [u32; PI_GFX_SLOTS],
    /// The slot's graphics interrupt has been raised for the counter falling
    /// below the bias, and not yet re-armed by the counter rising above it.
    pub starved: [bool; PI_GFX_SLOTS],
    /// Host-only: the zero-length stall has been logged for this slot.
    pub stall_logged: [bool; PI_GFX_SLOTS],
    /// Bedrock: a counted store left the counter below the bias, and the
    /// slot's next counted store waits until the processor has taken the
    /// graphics interrupt or the counter has risen above the bias.
    pub stopped: [bool; PI_GFX_SLOTS],
    /// Bedrock: the interrupt counter runs (the credit counter is at zero).
    pub int_running: [bool; PI_GFX_SLOTS],
    /// The clock tick at which int_count held its value.
    pub int_epoch: [u64; PI_GFX_SLOTS],
}

impl PiGfx {
    /// The counter above the bias re-arms the slot's graphics interrupt.
    fn rearm(&mut self, slot: usize) {
        if self.credit[slot] > self.bias[slot] {
            self.starved[slot] = false;
        }
    }

    /// Register n of the block, 0 to PI_GFX_REGS - 1, counted in
    /// doublewords from PI_GFX_BASE.
    pub fn read(&self, n: u32) -> u64 {
        let slot = (n / PI_GFX_SLOT_REGS) as usize;

        assert!(n < PI_GFX_REGS);
        match n % PI_GFX_SLOT_REGS {
            PI_GFX_PAGE => self.page[slot],
            PI_GFX_CREDIT => self.credit[slot] as u64,
            PI_GFX_BIAS => self.bias[slot] as u64,
            PI_GFX_INT_COUNT => self.int_count[slot] as u64,
            _ => self.int_compare[slot] as u64,
        }
    }

    /// A write keeps the field's bits; a write to the credit counter replaces
    /// the count.
    pub fn write(&mut self, n: u32, value: u64) {
        let slot = (n / PI_GFX_SLOT_REGS) as usize;

        assert!(n < PI_GFX_REGS);
        match n % PI_GFX_SLOT_REGS {
            PI_GFX_PAGE => self.page[slot] = value & PI_GFX_PAGE_MASK,
            PI_GFX_CREDIT => {
                self.credit[slot] = (value as u32) & PI_GFX_CREDIT_MASK;
                self.rearm(slot);
            }
            PI_GFX_BIAS => self.bias[slot] = (value as u32) & PI_GFX_BIAS_MASK,
            PI_GFX_INT_COUNT => self.int_count[slot] = (value as u32) & PI_GFX_INT_MASK,
            _ => self.int_compare[slot] = (value as u32) & PI_GFX_INT_MASK,
        }
    }

    /// Whether the slot's page is enabled and covers physical address `addr`.
    pub fn covers(&self, slot: usize, addr: u64) -> bool {
        assert!(slot < PI_GFX_SLOTS);
        let page = self.page[slot];
        (page & PI_GFX_PAGE_ENABLE) != 0
            && (addr & PI_GFX_PAGE_ADDRESS) == (page & PI_GFX_PAGE_ADDRESS)
    }

    /// An uncached I/O store by the processor in `slot` reached physical
    /// address bits 39:0 `addr`. Charges one credit when the slot's page is
    /// enabled and covers it, and returns whether it did. `who` names the
    /// device in log messages.
    pub fn count_store(&mut self, slot: usize, addr: u64, who: &str) -> bool {
        if !self.covers(slot, addr) {
            return false;
        }
        // Whether the twelve-bit counter wraps, saturates or goes negative is
        // unknown. It wraps, and says so.
        if self.credit[slot] == 0 {
            crate::dlog!(LogModule::Odyssey, "{}: graphics credit counter {} wrapped below zero", who, (b'A' + slot as u8) as char);
        }
        self.credit[slot] = self.credit[slot].wrapping_sub(1) & PI_GFX_CREDIT_MASK;
        true
    }

    /// After a counted store: whether the slot's graphics interrupt must be
    /// raised now. True once when the counter has fallen below the bias; the
    /// counter rising above the bias again re-arms it. The stall that
    /// accompanies it on the chip is modelled as zero length.
    pub fn take_starved(&mut self, slot: usize, who: &str) -> bool {
        assert!(slot < PI_GFX_SLOTS);
        if self.starved[slot] || self.credit[slot] >= self.bias[slot] {
            return false;
        }
        self.starved[slot] = true;
        if !self.stall_logged[slot] {
            self.stall_logged[slot] = true;
            crate::dlog!(
                LogModule::Odyssey,
                "{}: graphics credit counter {} fell below its bias; interrupt raised at once, the write stall and the interrupt counter are not modelled",
                who,
                (b'A' + slot as u8) as char
            );
        }
        true
    }

}

impl PiGfx {
    /// `credits` credits returned to `slot`.
    pub fn add_credits(&mut self, slot: usize, credits: u32, who: &str) {
        assert!(slot < PI_GFX_SLOTS);
        let sum = self.credit[slot] as u64 + credits as u64;
        if sum > PI_GFX_CREDIT_MASK as u64 {
            crate::dlog!(LogModule::Odyssey, "{}: graphics credit counter {} wrapped above {:#x}", who, (b'A' + slot as u8) as char, PI_GFX_CREDIT_MASK);
        }
        self.credit[slot] = (sum & PI_GFX_CREDIT_MASK as u64) as u32;
        self.rearm(slot);
    }
}

