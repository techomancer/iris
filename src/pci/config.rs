//! A function's configuration header and its base address registers.

use std::sync::Mutex;

use crate::snapshot::{get_field, load_u32_slice, u32_slice_to_toml};

/// Command register bits.
pub const PCI_COMMAND_IO: u32 = 0x1;
pub const PCI_COMMAND_MEMORY: u32 = 0x2;

/// Secondary status (dword 7 bit 29): received master abort.
const SEC_STATUS_MASTER_ABORT: u32 = 1 << 29;

/// Interrupt pins, as register 0x3d holds them.
pub const PCI_INTA: u8 = 1;

/// What one base address register decodes, and how much.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bar {
    Unused,
    /// I/O space, this many bytes (a power of two).
    Io(u32),
    /// 32-bit memory space, this many bytes (a power of two).
    Mem(u32),
    /// 64-bit memory space, this many bytes (a power of two, under 4 GiB).
    /// The BAR takes two header dwords: the next entry of the array must be
    /// [`Bar::Unused`], and the header makes it the upper half. Both host
    /// bridges generate 32-bit PCI addresses, so a base programmed above 4
    /// GiB (a non-zero upper half) is not reachable and decodes nothing.
    Mem64(u32),
}

/// What the header holds for each BAR dword: a BAR, or the upper half of
/// the 64-bit BAR before it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BarSlot {
    Bar(Bar),
    Upper,
}

/// Which address space an access is in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Space {
    Io,
    Mem,
}

/// Initial value and writable bits for a function-specific configuration dword.
#[derive(Clone, Copy, Debug)]
pub struct PciRegisterPolicy {
    /// Dword index. Header, BAR, subsystem identity, and interrupt dwords are managed separately.
    pub reg_dword: usize,
    /// Value restored by [`PciConfig::reset`].
    pub initial: u32,
    /// Bits that configuration writes may change.
    pub writable_mask: u32,
}

/// Function-specific write behavior for a type 0 PCI configuration header.
///
/// Register policies are copied into the header during construction; the
/// borrowed slice is not retained.
#[derive(Clone, Copy, Debug)]
pub struct PciConfigPolicy<'a> {
    /// Fixed Status bits that read as one.
    pub status: u16,
    /// Device-latched Status bits that writes acknowledge with one.
    pub status_w1c_mask: u16,
    /// Writable Command bits.
    pub command_mask: u16,
    /// Writable bits in the latency timer byte.
    pub latency_mask: u8,
    /// Writable bits in the cache-line-size byte.
    pub cache_line_mask: u8,
    /// Initial values and write masks for function-specific dwords.
    pub registers: &'a [PciRegisterPolicy],
}

impl PciConfigPolicy<'static> {
    /// The policy of a plain function, as a constant a board table can
    /// start from.
    pub const DEFAULT: Self = Self {
        status: 0x0200,
        status_w1c_mask: 0,
        command_mask: 0xffff,
        latency_mask: 0xff,
        cache_line_mask: 0xff,
        registers: &[],
    };
}

impl<'a> Default for PciConfigPolicy<'a> {
    fn default() -> Self {
        Self { registers: &[], ..PciConfigPolicy::DEFAULT }
    }
}

/// One function's type 0 configuration header.
///
/// Enough of it to be discovered and configured: identity, class, command
/// and status, the interrupt pin, and base address registers that answer a
/// size probe. Writing all ones to a BAR and reading it back is how firmware
/// learns how much space a device wants, so a BAR that stores whatever it is
/// given reports a size of four gigabytes and the bus layout falls apart.
pub struct PciConfig {
    regs: Mutex<[u32; 64]>,
    bars: [BarSlot; 6],
    /// The registers as the function powers up, for [`PciConfig::reset`].
    power_on: [u32; 64],
    /// Dwords `16..caps_end` hold a capability list, which firmware cannot
    /// overwrite. 0 when the function has none.
    caps_end: usize,
    status: u16,
    status_w1c_mask: u16,
    command_mask: u16,
    latency_mask: u8,
    cache_line_mask: u8,
    writable_masks: [u32; 64],
    /// A type 1 header (PCI-to-PCI bridge): no BARs, and dwords 6 to 12 are
    /// bus numbers and forwarding windows.
    type1: bool,
}

impl PciConfig {
    /// `class` is the 24-bit class/subclass/interface, as it sits in the top
    /// three bytes of register 0x08. `pin` is 0 for none, or [`PCI_INTA`]
    /// and on; the line register is left for firmware to assign.
    pub fn new(vendor: u16, device: u16, class: u32, revision: u8, bars: [Bar; 6], pin: u8) -> Self {
        Self::new_with_policy(
            vendor,
            device,
            class,
            revision,
            bars,
            pin,
            PciConfigPolicy::default(),
        )
    }

    /// Construct a type 0 header with function-specific register behavior.
    pub fn new_with_policy(
        vendor: u16,
        device: u16,
        class: u32,
        revision: u8,
        bars: [Bar; 6],
        pin: u8,
        policy: PciConfigPolicy<'_>,
    ) -> Self {
        let mut slots = bars.map(BarSlot::Bar);
        for i in 0..5 {
            if let Bar::Mem64(_) = bars[i] {
                assert!(bars[i + 1] == Bar::Unused, "a 64-bit BAR takes the next one");
                slots[i + 1] = BarSlot::Upper;
            }
        }
        assert!(!matches!(bars[5], Bar::Mem64(_)), "a 64-bit BAR needs a BAR after it");
        let mut r = [0u32; 64];
        r[0] = ((device as u32) << 16) | vendor as u32;
        r[2] = (class << 8) | revision as u32;
        r[15] = (pin as u32) << 8;
        for (i, b) in bars.iter().enumerate() {
            match b {
                Bar::Io(_) => r[4 + i] = 1,
                // Memory type 10b: decodes anywhere in 64-bit space.
                Bar::Mem64(_) => r[4 + i] = 0x4,
                _ => {}
            }
        }
        let mut writable_masks = [u32::MAX; 64];
        for register in policy.registers {
            assert!(
                matches!(register.reg_dword, 10 | 12..=14 | 16..=63),
                "pci: register policy requires a function-specific dword in 0..64");
            r[register.reg_dword] = register.initial;
            writable_masks[register.reg_dword] = register.writable_mask;
        }
        Self {
            regs: Mutex::new(r),
            bars: slots,
            power_on: r,
            status: policy.status,
            status_w1c_mask: policy.status_w1c_mask,
            command_mask: policy.command_mask,
            latency_mask: policy.latency_mask,
            cache_line_mask: policy.cache_line_mask,
            writable_masks,
            caps_end: 0,
            type1: false,
        }
    }

    /// A PCI-to-PCI bridge's type 1 header: class 0604, header type 1, no
    /// BARs and no interrupt pin. Dwords 6 (primary, secondary and
    /// subordinate bus numbers), 7 (16-bit I/O base and limit, bits 15:12 of
    /// each), 8 (memory base and limit, bits 31:20) and 9 (32-bit
    /// prefetchable memory) store what firmware writes, to the granularity
    /// the register has; dwords 10 to 12, the 64-bit and 32-bit-I/O upper
    /// halves, and dword 14, the expansion ROM base, read zero. The bridge
    /// control register is the top half of dword 15. Everything resets to
    /// zero, and the windows forward nothing until the command register
    /// enables them.
    pub fn type1(vendor: u16, device: u16, revision: u8) -> Self {
        let mut c = Self::new(vendor, device, 0x06_0400, revision, [Bar::Unused; 6], 0);
        c.type1 = true;
        let r = c.regs.get_mut().unwrap();
        r[3] = 1 << 16;
        c.power_on = *r;
        c
    }

    /// Whether this type 1 header forwards the range `base..base + size` of
    /// `space` to its secondary bus: the range lies wholly inside one
    /// enabled window (I/O, memory or prefetchable memory). A range that
    /// straddles a window's edge is not forwarded; what a bridge does with
    /// one is not defined.
    pub fn forwards(&self, space: Space, base: u32, size: u32) -> bool {
        let r = self.regs.lock().unwrap();
        let (lo, hi) = (base as u64, base as u64 + size as u64 - 1);
        let inside = |wlo: u64, whi: u64| wlo <= whi && wlo <= lo && hi <= whi;
        match space {
            Space::Io => {
                r[1] & PCI_COMMAND_IO != 0 && inside(((r[7] & 0xf0) as u64) << 8, ((r[7] >> 8 & 0xf0) as u64) << 8 | 0xfff)
            }
            Space::Mem => {
                r[1] & PCI_COMMAND_MEMORY != 0 && {
                    let window = |dword: u32| (((dword & 0xfff0) as u64) << 16, ((dword >> 16 & 0xfff0) as u64) << 16 | 0xf_ffff);
                    let (mem, pf) = (window(r[8]), window(r[9]));
                    inside(mem.0, mem.1) || inside(pf.0, pf.1)
                }
            }
        }
    }

    /// Give the function a capability list: `caps` are the raw dwords of the
    /// list, which starts at dword `first` (at least 16, the end of the
    /// header). They are read-only and survive a reset. The header's
    /// capability pointer and status bit are set to match.
    pub fn with_capabilities(mut self, first: usize, caps: &[u32]) -> Self {
        assert!(first >= 16 && first + caps.len() <= 64, "capabilities must fit in the 256-byte header");
        let r = self.regs.get_mut().unwrap();
        for (i, &d) in caps.iter().enumerate() {
            r[first + i] = d;
        }
        // Capabilities pointer (byte offset) in the low byte of dword 13.
        r[13] = (first * 4) as u32;
        self.power_on = *r;
        self.caps_end = first + caps.len();
        self
    }

    /// RST#: back to the power-up header. The command register clears, so
    /// the function stops decoding, and the BARs and line forget what
    /// firmware assigned.
    pub fn reset(&self) {
        let mut r = self.regs.lock().unwrap();
        // The subsystem identity is the board's, set once after
        // construction (IOC3's selects what a guest instantiates).
        let subsystem = r[11];
        *r = self.power_on;
        r[11] = subsystem;
    }

    /// A type 1 header's secondary status gains received master abort: a
    /// cycle forwarded to the bus behind the bridge went unclaimed.
    pub fn record_master_abort(&self) {
        let mut r = self.regs.lock().unwrap();
        if self.type1 {
            r[7] |= SEC_STATUS_MASTER_ABORT;
        }
    }

    /// Register `reg` (a dword index, 0..64).
    pub fn read(&self, reg: usize) -> u32 {
        let r = self.regs.lock().unwrap();
        match reg {
            // A function with a capability list also sets status bit 4.
            1 => (r[1] & (0xffff | ((self.status_w1c_mask as u32) << 16)))
                | ((self.status as u32) << 16)
                | if r[13] != 0 { 0x0010_0000 } else { 0 },
            _ => r[reg],
        }
    }

    pub fn write(&self, reg: usize, val: u32) {
        self.write_masked(reg, val, u32::MAX);
    }

    /// Write selected bits of dword `reg`. Both `val` and `mask` use the
    /// register's bit positions. Unselected Status events are not acknowledged.
    pub fn write_masked(&self, reg: usize, val: u32, mask: u32) {
        if mask == 0 {
            return;
        }
        let mut r = self.regs.lock().unwrap();
        match reg {
            // Identity, class and header type are read-only.
            0 | 2 => {}
            3 => {
                let mask = mask & (((self.latency_mask as u32) << 8) | self.cache_line_mask as u32);
                r[3] = (r[3] & !mask) | (val & mask);
            }
            // Command writes replace selected bits; Status events clear on ones.
            1 => {
                let clear = val & mask & ((self.status_w1c_mask as u32) << 16);
                r[1] = (r[1] & !(mask & 0xffff) & !clear)
                    | (val & mask & self.command_mask as u32);
            }
            // A bridge's bus numbers and forwarding windows. The low nibbles
            // of the base and limit registers are read-only zero (16-bit I/O
            // and 32-bit memory), which also makes the upper halves of the
            // prefetchable and I/O windows (dwords 10 to 12) read-only zero;
            // the I/O secondary status is not kept.
            6 if self.type1 => r[reg] = (r[reg] & !mask) | (val & mask),
            10..=12 if self.type1 => {}
            // I/O base and limit; the secondary status above them is
            // write-one-to-clear, and only received master abort is kept.
            7 if self.type1 => {
                r[reg] = (((r[reg] & !mask) | (val & mask)) & 0x0000_f0f0) | (r[reg] & !(val & mask) & SEC_STATUS_MASTER_ABORT)
            }
            8 | 9 if self.type1 => r[reg] = ((r[reg] & !mask) | (val & mask)) & 0xfff0_fff0,
            // A type 1 header's expansion ROM base is dword 14, and the
            // bridge has no ROM either.
            14 if self.type1 => {}
            4..=9 => {
                let val = (r[reg] & !mask) | (val & mask);
                r[reg] = match self.bars[reg - 4] {
                    BarSlot::Bar(Bar::Unused) => 0,
                    // The upper half of a 64-bit base takes every bit.
                    BarSlot::Upper => val,
                    // Address bits above the size take the write; those
                    // below read back as zero, which tells firmware the
                    // size. The type bits stay what the BAR is.
                    BarSlot::Bar(Bar::Io(n)) => (val & !(n - 1)) | 1,
                    BarSlot::Bar(Bar::Mem(n)) => val & !(n - 1),
                    BarSlot::Bar(Bar::Mem64(n)) => (val & !(n - 1)) | 0x4,
                };
            }
            // Expansion ROM base: no function has a ROM, so the register is
            // unimplemented and reads zero (PCI 2.2, 6.1 and 6.2.5.2). A size
            // probe that read back its ones would have firmware map a ROM
            // that is not there.
            12 => {}
            // The capability list is read-only.
            16.. if reg < self.caps_end => {}
            // Interrupt pin is read-only; the line is firmware's scratch.
            15 => {
                let mask = mask & !0xff00;
                r[15] = (r[15] & !mask) | (val & mask);
            }
            _ => {
                let mask = mask & self.writable_masks[reg];
                r[reg] = (r[reg] & !mask) | (val & mask);
            }
        }
    }

    /// Latch selected device Status events without replacing earlier events.
    pub fn set_status_bits(&self, bits: u16) {
        self.regs.lock().unwrap()[1] |= ((bits & self.status_w1c_mask) as u32) << 16;
    }

    /// Set the board-assigned Subsystem ID, which survives function reset.
    pub fn set_subsystem_identity(&self, value: u32) {
        self.regs.lock().unwrap()[11] = value;
    }

    /// The base address firmware programmed into BAR `i`, type bits masked.
    pub fn bar_base(&self, i: usize) -> u32 {
        let r = self.regs.lock().unwrap();
        let v = r[4 + i];
        match self.bars[i] {
            BarSlot::Bar(Bar::Io(_)) => v & !3,
            // Above 4 GiB: out of reach of a 32-bit PCI address.
            BarSlot::Bar(Bar::Mem64(_)) if r[5 + i] != 0 => 0,
            _ => v & !15,
        }
    }

    pub fn command(&self) -> u32 {
        self.regs.lock().unwrap()[1] & 0xffff
    }

    pub fn interrupt_pin(&self) -> u8 {
        (self.regs.lock().unwrap()[15] >> 8) as u8
    }

    /// The regions this function decodes: (space, BAR, base, size) for
    /// each BAR that has been assigned an address in a space the command
    /// register enables, in BAR order. The bus caches these and updates its
    /// mappings after a configuration write.
    pub fn bar_regions(&self) -> Vec<(Space, usize, u32, u32)> {
        let cmd = self.command();
        (0..6)
            .filter_map(|i| {
                let (space, size) = match self.bars[i] {
                    BarSlot::Bar(Bar::Io(n)) if cmd & PCI_COMMAND_IO != 0 => (Space::Io, n),
                    BarSlot::Bar(Bar::Mem(n) | Bar::Mem64(n)) if cmd & PCI_COMMAND_MEMORY != 0 => (Space::Mem, n),
                    _ => return None,
                };
                let base = self.bar_base(i);
                (base != 0).then_some((space, i, base, size))
            })
            .collect()
    }
}

/// The header's registers. Identity, class and the BARs' sizes come from
/// the function, so a header only loads into the same kind of function.
impl crate::traits::Saveable for PciConfig {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("regs".into(), u32_slice_to_toml(&*self.regs.lock().unwrap()));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let Some(r) = get_field(v, "regs") else { return Ok(()) };
        let mut saved = [0u32; 64];
        load_u32_slice(r, &mut saved);
        let mut regs = self.regs.lock().unwrap();
        if saved[0] != regs[0] {
            return Err(format!("pci: saved header is for {:#010x}, this function is {:#010x}", saved[0], regs[0]));
        }
        *regs = saved;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scsi_like() -> PciConfig {
        PciConfig::new(0x9004, 0x8078, 0x0100_00, 1,
            [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], PCI_INTA)
    }

    fn scsi_like_with_policy(policy: PciConfigPolicy<'_>) -> PciConfig {
        PciConfig::new_with_policy(
            0x9004, 0x8078, 0x0100_00, 1,
            [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], PCI_INTA,
            policy,
        )
    }

    /// Firmware learns how much space a device wants by writing all ones to
    /// a BAR and reading back which address bits stuck.
    #[test]
    fn save_load_round_trip() {
        use crate::traits::Saveable;
        let src = scsi_like();
        src.write(1, PCI_COMMAND_MEMORY | PCI_COMMAND_IO);
        src.write(4, 0x0000_1000);
        src.write(5, 0x8000_1000);
        src.write(15, 0x0000_0108);
        let v1 = src.save_state();
        let dst = scsi_like();
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "PciConfig save_state mismatch after load_state round-trip");
        assert_eq!(dst.bar_base(1), 0x8000_1000);
        let other = PciConfig::new(0x1234, 0x5678, 0, 0, [Bar::Unused; 6], 0);
        assert!(other.load_state(&v1).is_err(), "a header loads only into the same kind of function");
    }

    fn nvme_like() -> PciConfig {
        PciConfig::new(0x1b36, 0x0010, 0x01_0802, 1,
            [Bar::Mem64(0x2000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], PCI_INTA)
    }

    /// A 64-bit BAR reads as memory type 10b, sizes through its lower dword,
    /// and keeps whatever its upper dword is given.
    #[test]
    fn a_64_bit_bar_probes_and_stores_both_halves() {
        let c = nvme_like();
        assert_eq!(c.read(4), 0x4, "64-bit memory, not prefetchable");
        c.write(4, 0xffff_ffff);
        c.write(5, 0xffff_ffff);
        assert_eq!(c.read(4), 0xffff_e004, "8 KiB, type bits kept");
        assert_eq!(c.read(5), 0xffff_ffff, "the upper half is plain storage");
        c.write(5, 0);
        c.write(4, 0x8000_2000);
        assert_eq!(c.bar_base(0), 0x8000_2000);
        assert_eq!(c.read(4), 0x8000_2004);
        c.write(6, 0xffff_ffff);
        assert_eq!(c.read(6), 0, "the BAR after the pair is its own");
    }

    #[test]
    fn a_64_bit_bar_decodes_below_4_gib_only() {
        let c = nvme_like();
        c.write(4, 0x8000_2000);
        c.write(1, PCI_COMMAND_MEMORY);
        assert_eq!(c.bar_regions(), vec![(Space::Mem, 0, 0x8000_2000, 0x2000)]);
        c.write(5, 1);
        assert!(c.bar_regions().is_empty(), "a base above 4 GiB cannot be reached");
        c.write(1, 0);
        c.reset();
        assert_eq!((c.read(4), c.read(5)), (4, 0), "reset forgets both halves");
    }

    #[test]
    fn a_64_bit_bar_saves_with_its_upper_half() {
        use crate::traits::Saveable;
        let src = nvme_like();
        src.write(4, 0x8000_2000);
        src.write(5, 3);
        let dst = nvme_like();
        dst.load_state(&src.save_state()).unwrap();
        assert_eq!((dst.read(4), dst.read(5)), (0x8000_2004, 3));
    }

    #[test]
    fn a_bar_reports_its_size_to_a_probe() {
        let c = scsi_like();
        assert_eq!(c.read(0), 0x8078_9004, "device and vendor id");
        c.write(4, 0xffff_ffff);
        assert_eq!(c.read(4), 0xffff_ff01, "256 bytes of I/O space");
        c.write(5, 0xffff_ffff);
        assert_eq!(c.read(5), 0xffff_f000, "4 KiB of memory space");
        // An unimplemented BAR reads back as zero however hard it is probed.
        c.write(6, 0xffff_ffff);
        assert_eq!(c.read(6), 0);
        c.write(5, 0x8000_1000);
        assert_eq!(c.bar_base(1), 0x8000_1000, "and then holds the address assigned to it");
    }

    /// No function has an expansion ROM, so the ROM base register is
    /// unimplemented: a size probe reads back zero, which tells firmware
    /// there is no ROM to map (PCI 2.2, 6.1 and 6.2.5.2).
    #[test]
    fn the_expansion_rom_register_reads_zero_when_probed() {
        let c = scsi_like();
        c.write(12, 0xffff_f800);
        assert_eq!(c.read(12), 0);
        c.write(12, 0x00b0_2001);
        assert_eq!(c.read(12), 0, "an assigned address does not stick either");
    }

    /// A function decodes a space only once it has been told it may, and
    /// only the BARs that have been assigned an address.
    #[test]
    fn a_function_claims_only_its_own_region_and_only_when_enabled() {
        let c = scsi_like();
        c.write(5, 0x8000_1000);
        assert!(c.bar_regions().is_empty(), "memory decoding is off until enabled");
        c.write(1, PCI_COMMAND_MEMORY);
        assert_eq!(c.bar_regions(), vec![(Space::Mem, 1, 0x8000_1000, 0x1000)], "BAR 1 alone, and not the I/O BAR");
    }

    #[test]
    fn reset_returns_the_header_to_power_up() {
        let c = scsi_like();
        let fresh: Vec<u32> = (0..64).map(|i| c.read(i)).collect();
        c.write(1, PCI_COMMAND_MEMORY | PCI_COMMAND_IO);
        c.write(4, 0x0000_1000);
        c.write(5, 0x8000_1000);
        c.write(15, 0x0000_0009);
        c.reset();
        assert_eq!((0..64).map(|i| c.read(i)).collect::<Vec<_>>(), fresh);
        assert!(c.bar_regions().is_empty());
        assert_eq!(c.interrupt_pin(), PCI_INTA, "the pin is wiring, not state");
    }

    /// A subsystem identity set after construction, as IOC3 sets its own,
    /// is the board's and survives a reset.
    #[test]
    fn reset_keeps_the_subsystem_identity() {
        let c = scsi_like();
        c.set_subsystem_identity(0xc304_10a9);
        c.write(4, 0x0000_1000);
        c.reset();
        assert_eq!(c.read(11), 0xc304_10a9);
        assert_eq!(c.read(4), 1);
    }

    /// Firmware programs the latency timer and cache line size, and reads
    /// them back; the header type byte above them stays what it is.
    #[test]
    fn the_latency_timer_and_cache_line_size_are_stored() {
        let c = scsi_like();
        c.write(3, 0x00ff_2810);
        assert_eq!(c.read(3), 0x0000_2810);
        c.reset();
        assert_eq!(c.read(3), 0);
    }

    #[test]
    fn the_interrupt_pin_is_read_only() {
        let c = scsi_like();
        c.write(15, 0x0000_0009);
        assert_eq!(c.read(15) & 0xffff, 0x0109, "line 9 assigned, pin A kept");
        c.write(15, 0);
        assert_eq!(c.interrupt_pin(), PCI_INTA);
    }

    #[test]
    fn header_policies_mask_writes_and_restore_reset_values() {
        for (status, command, latency, cache) in [(0x0200, 0xffff, 0, 0), (0x02b0, 0x0156, 0xf0, 0x0f)] {
            let c = scsi_like_with_policy(PciConfigPolicy {
                status,
                status_w1c_mask: 0,
                command_mask: command,
                latency_mask: latency,
                cache_line_mask: cache,
                registers: &[PciRegisterPolicy {
                    reg_dword: 18,
                    initial: 0xaabb_0020,
                    writable_mask: 0xff00,
                }],
            });
            c.write(1, u32::MAX);
            c.write(3, u32::MAX);
            c.write(18, u32::MAX);
            assert_eq!(c.read(1), (u32::from(status) << 16) | u32::from(command));
            assert_eq!(c.read(3), (u32::from(latency) << 8) | u32::from(cache));
            assert_eq!(c.read(18), 0xaabb_ff20);
            c.reset();
            assert_eq!(c.read(1), u32::from(status) << 16);
            assert_eq!(c.read(3), 0);
            assert_eq!(c.read(18), 0xaabb_0020);
        }
    }

    #[test]
    fn masked_writes_preserve_unselected_fields_and_status_events() {
        let c = scsi_like_with_policy(PciConfigPolicy {
            status: 0x0200,
            status_w1c_mask: 0x8040,
            command_mask: 0x0157,
            latency_mask: 0xff,
            cache_line_mask: 0xff,
            registers: &[PciRegisterPolicy {
                reg_dword: 18,
                initial: 0xaabb_0000,
                writable_mask: 0xffff,
            }],
        });
        c.set_status_bits(0x8040);
        c.write(1, 0x0157);
        c.write(3, 0x1234);
        c.write(5, 0x1234_5000);
        c.write(4, 0x1201);
        for (reg, value, mask, expected) in [
            (1, 0xffff_0002, 0xff, 0x8240_0102),
            (1, 0xffff_0000, 0x8000_0000, 0x0240_0102),
            (1, 0x0040_0000, 0x00ff_0000, 0x0200_0102),
            (1, u32::MAX, 0, 0x0200_0102),
            (3, 0xffff_00ab, 0xff, 0x12ab),
            (5, 0xabcd_ffff, 0xffff_0000, 0xabcd_5000),
            (4, 0x3400, 0xff00, 0x3401),
            (15, u32::MAX, 0xff00, 0x0100),
            (18, u32::MAX, 0xff, 0xaabb_00ff),
        ] {
            c.write_masked(reg, value, mask);
            assert_eq!(c.read(reg), expected, "register {reg}");
        }
        c.set_status_bits(u16::MAX);
        c.reset();
        assert_eq!(c.read(1), 0x0200_0000);
        let default = scsi_like();
        default.set_status_bits(u16::MAX);
        assert_eq!(default.read(1), 0x0200_0000);
    }

    #[test]
    fn snapshots_restore_registers_with_destination_write_policies() {
        use crate::traits::Saveable;
        let src = scsi_like_with_policy(PciConfigPolicy {
            status: 0x02b0,
            status_w1c_mask: 0xc000,
            command_mask: 0x0157,
            latency_mask: 0xff,
            cache_line_mask: 0xff,
            registers: &[PciRegisterPolicy {
                reg_dword: 18,
                initial: 0,
                writable_mask: 0x100,
            }],
        });
        src.set_status_bits(0x8000);
        src.write(3, 0x1234);
        src.write(18, 0x100);
        let saved = src.save_state();
        for (events, command, cache) in [(0, 7, 0x0f), (0xc000, 0x0157, 0xff)] {
            let dst = scsi_like_with_policy(PciConfigPolicy {
                status: 0x0200,
                status_w1c_mask: events,
                command_mask: command,
                latency_mask: 0,
                cache_line_mask: cache,
                registers: &[PciRegisterPolicy {
                    reg_dword: 18,
                    initial: 0x20,
                    writable_mask: 0x200,
                }],
            });
            dst.load_state(&saved).unwrap();
            assert_eq!(dst.save_state(), saved);
            assert_eq!(dst.read(1), 0x0200_0000 | ((0x8000 & u32::from(events)) << 16));
            dst.write_masked(1, u32::MAX, u32::MAX);
            dst.write(3, u32::MAX);
            dst.write(18, u32::MAX);
            assert_eq!(dst.read(1), 0x0200_0000 | u32::from(command));
            assert_eq!(dst.read(3), (0x1234 & !u32::from(cache)) | u32::from(cache));
            assert_eq!(dst.read(18), 0x300);
            dst.reset();
            assert_eq!(dst.read(18), 0x20);
        }
    }
}
