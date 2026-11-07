//! What a PCI host bridge supplies to its root bus, and what it does to a
//! CPU access on its way to a function.
//!
//! The bus (`pci/`) has no byte order. A host chooses how bytes move per
//! window as a [`ByteOrder`] value, and the physical bus maps [`BarDevice`]
//! entries that apply it, so the processor reaches a function with no
//! search of the bus and no host lock. The hosts (`mace_pci`, `bridge`)
//! implement [`PciHost`] and use the rest; the core files (`bus`, `decode`,
//! `intx`, `config`, `function`, `pci_to_pci_bridge`) use only the trait.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::pci::{BarMapping, PciDma, PciFunction};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_OK};

/// What a root bus asks of its host bridge: the DMA each requester gets
/// and where each host IRQ goes. Nothing else.
pub trait PciHost: Send + Sync {
    /// The DMA identity of root slot `requester`.
    fn dma_for(&self, requester: u8) -> Arc<dyn PciDma>;
    /// Host IRQ `irq` is now `level`: the OR of every INTx line and board
    /// IRQ routed to it. Calls are serialised and in order per IRQ.
    fn set_irq(&self, irq: u8, level: bool);
}

/// How a CPU window orders bytes on their way to a function. `Stream`: each
/// byte is at its own address whatever the access size. `Values`: each access
/// carries a whole value of `width` bytes, as the host's swap bit says.
///
/// How a CPU access becomes bus cycles. Functions are little-endian and see
/// offsets and values as they would on the bus; the CPU is big-endian.
/// SGI's names for the two modes are byte stream and word values.
///
/// `n` is the bytes in one cycle; the CPU's value is the number it loaded
/// or stores, so a little-endian function reading it back sees the same
/// number except where the table says the bytes are reversed.
///
/// | Variant | Size | Address | Cycles | Placement |
/// |---|---|---|---|---|
/// | `Stream` | 1, 2, 4 | as given | one | PCI byte i is the value's i-th byte in memory order |
/// | `Stream` | 8 | as given | two, lower address first | the same, a word each |
/// | `Values{4}` | 1, 2 | `off ^ (4 - size)` | one | the value |
/// | `Values{4}` | 4 | as given | one | the value |
/// | `Values{4}` | 8 | as given | two, lower address first | the high word, then the low |
/// | `Values{8}` | 1, 2, 4 | `off ^ (8 - size)` | one | the value |
/// | `Values{8}` | 8 | as given | one (two word calls) | the low word, then the high |
///
/// MACE is `Values{4}`: bytes `^3`, halves `^2`, words unchanged
/// (O2 Device Driver Programming Guide 007-3443-002, pp. 397-398, Table
/// 15-3, established). Its 64-bit PIO is not documented there; the two-word
/// split above is iris's existing behaviour and is unknown. The Bridge picks
/// per window from Device(x): its programmed swap bit gives `Stream`,
/// and its 64-bit device bit gives the width.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ByteOrder {
    Stream,
    Values { width: u8 },
}

impl ByteOrder {
    /// Word values for a bus `width` bytes wide (4 or 8). Anything else is a
    /// machine bug.
    pub fn values(width: u8) -> Self {
        assert!(matches!(width, 4 | 8), "pci: no word-value byte order for a {width}-byte bus");
        ByteOrder::Values { width }
    }

    /// A CPU read of `size` bytes (1, 2, 4 or 8) whose first byte is at BAR
    /// offset `off`.
    pub fn read(self, f: &dyn PciFunction, bar: usize, off: u32, size: usize) -> u64 {
        match self {
            ByteOrder::Stream => match size {
                1 => f.bar_read(bar, off, 1) as u64,
                2 => (f.bar_read(bar, off, 2) as u16).swap_bytes() as u64,
                4 => f.bar_read(bar, off, 4).swap_bytes() as u64,
                _ => {
                    let first = f.bar_read(bar, off, 4).swap_bytes() as u64;
                    first << 32 | f.bar_read(bar, off.wrapping_add(4), 4).swap_bytes() as u64
                }
            },
            ByteOrder::Values { width } => {
                let width = width as usize;
                if size == 8 {
                    let (a, b) = (f.bar_read(bar, off, 4) as u64, f.bar_read(bar, off.wrapping_add(4), 4) as u64);
                    return if width == 8 { b << 32 | a } else { a << 32 | b };
                }
                f.bar_read(bar, off ^ (width - size) as u32, size) as u64
            }
        }
    }

    pub fn write(self, f: &dyn PciFunction, bar: usize, off: u32, size: usize, val: u64) {
        match self {
            ByteOrder::Stream => match size {
                1 => f.bar_write(bar, off, val as u32, 1),
                2 => f.bar_write(bar, off, (val as u16).swap_bytes() as u32, 2),
                4 => f.bar_write(bar, off, (val as u32).swap_bytes(), 4),
                _ => {
                    f.bar_write(bar, off, ((val >> 32) as u32).swap_bytes(), 4);
                    f.bar_write(bar, off.wrapping_add(4), (val as u32).swap_bytes(), 4);
                }
            },
            ByteOrder::Values { width } => {
                let width = width as usize;
                if size == 8 {
                    let (hi, lo) = ((val >> 32) as u32, val as u32);
                    let (a, b) = if width == 8 { (lo, hi) } else { (hi, lo) };
                    f.bar_write(bar, off, a, 4);
                    f.bar_write(bar, off.wrapping_add(4), b, 4);
                    return;
                }
                f.bar_write(bar, off ^ (width - size) as u32, val as u32, size);
            }
        }
    }
}

/// A run of a host's CPU window that one BAR decodes whole, as a bus device
/// the physical bus maps over the host's own window. `order` is by value:
/// the access is one `match` and the function's own call.
pub struct BarDevice {
    f: Arc<dyn PciFunction>,
    bar: usize,
    /// The CPU address bits the host decodes, and where in them the BAR's
    /// byte 0 sits (it may lie below the window: a BAR larger than it).
    mask: u64,
    base: u64,
    order: ByteOrder,
}

impl BarDevice {
    pub fn new(f: Arc<dyn PciFunction>, bar: usize, base: u64, mask: u64, order: ByteOrder) -> Self {
        Self { f, bar, mask, base, order }
    }

    #[inline]
    fn off(&self, addr: u64) -> u32 {
        (addr & self.mask).wrapping_sub(self.base) as u32
    }
}

impl BusDevice for BarDevice {
    fn read8(&self, a: u64) -> BusRead8 { BusRead8::ok(self.order.read(&*self.f, self.bar, self.off(a), 1) as u8) }
    fn read16(&self, a: u64) -> BusRead16 { BusRead16::ok(self.order.read(&*self.f, self.bar, self.off(a), 2) as u16) }
    fn read32(&self, a: u64) -> BusRead32 { BusRead32::ok(self.order.read(&*self.f, self.bar, self.off(a), 4) as u32) }
    fn read64(&self, a: u64) -> BusRead64 { BusRead64::ok(self.order.read(&*self.f, self.bar, self.off(a), 8)) }
    fn write8(&self, a: u64, v: u8) -> u32 { self.order.write(&*self.f, self.bar, self.off(a), 1, v as u64); BUS_OK }
    fn write16(&self, a: u64, v: u16) -> u32 { self.order.write(&*self.f, self.bar, self.off(a), 2, v as u64); BUS_OK }
    fn write32(&self, a: u64, v: u32) -> u32 { self.order.write(&*self.f, self.bar, self.off(a), 4, v as u64); BUS_OK }
    fn write64(&self, a: u64, v: u64) -> u32 { self.order.write(&*self.f, self.bar, self.off(a), 8, v); BUS_OK }
}

/// The [`BarDevice`]s a host has made, one per distinct configuration.
///
/// The physical bus keeps every device it is given for as long as it lives (its
/// decode table holds raw pointers), so a host that made a fresh `BarDevice` on
/// each publish would grow that list without bound, and every configuration
/// write updates the mappings. The same function, BAR, CPU base and order give
/// back the same `Arc`, so a configuration that is mapped again maps the entry
/// it already has. One cache per host: the CPU address mask is the host's and
/// is not part of the key.
#[derive(Default)]
pub struct BarDevices(Mutex<HashMap<(usize, usize, u64, ByteOrder), Arc<BarDevice>>>);

impl BarDevices {
    /// The device that answers `run`, a run of a window with byte `order`
    /// whose CPU address bits are `mask`.
    pub fn get(&self, run: &BarMapping, mask: u64, order: ByteOrder) -> Arc<dyn BusDevice> {
        let base = run.cpu.wrapping_sub(run.off as u64);
        let key = (Arc::as_ptr(&run.function) as *const () as usize, run.bar, base, order);
        self.0.lock().unwrap().entry(key).or_insert_with(|| Arc::new(BarDevice::new(run.function.clone(), run.bar, base, mask, order))).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pci::testing::TestCard;
    use crate::pci::Bar;

    const V: u64 = 0x0102_0304_0506_0708;

    fn card() -> Arc<TestCard> {
        TestCard::new([Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], 0)
    }

    /// The register file after a write of V's top `size` bytes at `off`, and
    /// the BAR calls it took.
    fn written(order: ByteOrder, off: u32, size: usize) -> (Vec<u8>, Vec<(u32, usize)>) {
        let c = card();
        order.write(&*c, 0, off, size, V >> (64 - 8 * size));
        let bytes = c.mem()[..16].to_vec();
        (bytes, c.calls())
    }

    fn at(bytes: &[u8], off: usize, n: usize) -> Vec<u8> {
        bytes[off..off + n].to_vec()
    }

    #[test]
    fn every_row_of_the_order_table() {
        let be = |n: usize| V.to_be_bytes()[..n].to_vec();
        let le = |n: usize| (V >> (64 - 8 * n)).to_le_bytes()[..n].to_vec();
        // The value stored is V's top `size` bytes: 0x01, 0x0102, ...
        for off in 0..8u32 {
            let o = off as usize;
            // Stream: address as given, bytes in memory order.
            for size in [1usize, 2, 4, 8] {
                if off as usize % size != 0 { continue }
                let (b, calls) = written(ByteOrder::Stream, off, size);
                assert_eq!(at(&b, o, size), be(size), "byte stream size {size} off {off}");
                let want: Vec<_> = if size == 8 { vec![(off, 4), (off + 4, 4)] } else { vec![(off, size)] };
                assert_eq!(calls, want, "cycles, byte stream size {size}");
            }
            // Values{4}: sub-word accesses move to off ^ (4 - size), values unchanged.
            for size in [1usize, 2, 4] {
                if off as usize % size != 0 { continue }
                let (b, calls) = written(ByteOrder::values(4), off, size);
                let a = (off ^ (4 - size) as u32) as usize;
                assert_eq!(at(&b, a, size), le(size), "word values 4 size {size} off {off}");
                assert_eq!(calls, vec![(a as u32, size)]);
            }
            if off % 8 == 0 {
                let (b, calls) = written(ByteOrder::values(4), off, 8);
                assert_eq!(at(&b, o, 4), ((V >> 32) as u32).to_le_bytes(), "high word first");
                assert_eq!(at(&b, o + 4, 4), (V as u32).to_le_bytes());
                assert_eq!(calls, vec![(off, 4), (off + 4, 4)]);
            }
            // Values{8}.
            for size in [1usize, 2, 4] {
                if off as usize % size != 0 { continue }
                let (b, calls) = written(ByteOrder::values(8), off, size);
                let a = (off ^ (8 - size) as u32) as usize;
                assert_eq!(at(&b, a, size), le(size), "word values 8 size {size} off {off}");
                assert_eq!(calls, vec![(a as u32, size)]);
            }
            if off == 0 {
                let (b, calls) = written(ByteOrder::values(8), 0, 8);
                assert_eq!(at(&b, 0, 8), V.to_le_bytes(), "one cycle of eight reversed bytes");
                assert_eq!(calls, vec![(0, 4), (4, 4)], "carried as two word calls, low first");
            }
        }
    }

    /// Whatever a write leaves, the same access reads back.
    #[test]
    fn a_read_returns_what_the_same_write_stored() {
        for order in [ByteOrder::Stream, ByteOrder::values(4), ByteOrder::values(8)] {
            for size in [1usize, 2, 4, 8] {
                for off in (0..16u32).step_by(size) {
                    let c = card();
                    let v = if size == 8 { V } else { V >> (64 - 8 * size) };
                    order.write(&*c, 0, off, size, v);
                    assert_eq!(order.read(&*c, 0, off, size), v, "{order:?} size {size} off {off}");
                }
            }
        }
    }

    /// MACE's old swizzle: bytes `^3`, halves `^2`, words as given.
    #[test]
    fn values_4_matches_mace_swizzle() {
        for size in [1usize, 2, 4] {
            for off in 0..32u32 {
                let c = card();
                ByteOrder::values(4).read(&*c, 0, off, size);
                let want = crate::dev::mace_pci::MacePci::lane_swizzle(off, size);
                assert_eq!(c.calls(), vec![(want, size)], "size {size} off {off}");
            }
        }
    }

    /// The Bridge's old bar_cycle for a CPU access at `off`, copied here as
    /// the reference: bytes in CPU order, swap programmed or not.
    fn old_bridge_write(width: usize, swap: bool, off: u32, size: usize, val: u64, f: &dyn PciFunction) {
        let mut host = [0u8; 8];
        for (k, b) in host.iter_mut().enumerate().take(size) {
            *b = (val >> (8 * (size - 1 - k))) as u8;
        }
        let cycles: Vec<(u32, usize)> = if size > width { vec![(off, 4), (off + 4, 4)] } else { vec![(off, size)] };
        for (o, n) in cycles {
            let start = if o == off { 0 } else { 4 };
            let h = &host[start..start + n];
            let pci: Vec<u8> = (0..n).map(|i| h[if swap { i } else { n - 1 - i }]).collect();
            let boff = if swap { o } else { o ^ (width - n) as u32 };
            let mut done = 0;
            while done < n {
                let chunk = (n - done).min(4);
                let v = pci[done..done + chunk].iter().rev().fold(0u32, |a, &b| a << 8 | b as u32);
                f.bar_write(0, boff + done as u32, v, chunk);
                done += chunk;
            }
        }
    }

    #[test]
    fn values_matches_the_bridge_bar_cycle() {
        for (order, width, swap) in [(ByteOrder::Stream, 4, true), (ByteOrder::Stream, 8, true), (ByteOrder::values(4), 4, false), (ByteOrder::values(8), 8, false)] {
            for size in [1usize, 2, 4, 8] {
                for off in (0..16u32).step_by(size) {
                    let (old, new) = (card(), card());
                    let v = if size == 8 { V } else { V >> (64 - 8 * size) };
                    old_bridge_write(width, swap, off, size, v, &*old);
                    order.write(&*new, 0, off, size, v);
                    assert_eq!(old.mem(), new.mem(), "{order:?} width {width} size {size} off {off}");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "no word-value byte order")]
    fn values_3_panics() {
        ByteOrder::values(3);
    }

    /// MACE's 64-bit PIO is not documented (the guide's table stops at 32
    /// bits); the two-word split, high word first, is iris's own choice and
    /// this test pins it so a change is deliberate.
    #[test]
    fn mace_64_bit_is_the_two_word_split_and_marked_unknown() {
        let c = card();
        ByteOrder::values(4).write(&*c, 0, 0x10, 8, 0x1122_3344_5566_7788);
        assert_eq!(c.mem()[0x10..0x18], [0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55]);
    }

    #[test]
    fn a_bar_device_applies_its_order_at_the_offset_from_its_base() {
        let c = card();
        let bar = BarDevice::new(c.clone(), 0, 0x1a00_1000, 0xffff_ffff, ByteOrder::values(4));
        bar.write8(0x1a00_1084, 0xaa);
        assert_eq!(c.calls(), vec![(0x87, 1)], "off 0x84, byte ^3");
        assert_eq!(bar.read32(0x1a00_1084).data, 0xaa00_0000, "the byte sits in the top byte of its word");
        let hi = BarDevice::new(c, 0, 0x1000, 0x01ff_ffff, ByteOrder::Stream);
        hi.write16(0x9a00_1010, 0x1234);
        assert_eq!(hi.read16(0x1a00_1010).data, 0x1234, "the host mask folds the aliases");
    }

    #[test]
    fn a_configuration_that_is_made_again_is_the_same_device() {
        let (c, cache) = (card(), BarDevices::default());
        let run = |cpu: u64, off: u32| BarMapping { cpu, len: 0x1000, function: c.clone(), bar: 0, off, window: 0 };
        let a = cache.get(&run(0x1a00_1000, 0), 0xffff_ffff, ByteOrder::values(4));
        // The same BAR placed at the same base, whichever page the run starts at.
        let b = cache.get(&run(0x1a00_2000, 0x1000), 0xffff_ffff, ByteOrder::values(4));
        assert!(Arc::ptr_eq(&a, &b));
        assert!(!Arc::ptr_eq(&a, &cache.get(&run(0x1a00_1000, 0), 0xffff_ffff, ByteOrder::Stream)), "other order");
        assert!(!Arc::ptr_eq(&a, &cache.get(&run(0x1a01_1000, 0), 0xffff_ffff, ByteOrder::values(4))), "moved");
        assert!(!Arc::ptr_eq(&a, &cache.get(&run(0x1a00_1000, 0), 0xffff_ffff, ByteOrder::values(8))), "other width");
    }
}
