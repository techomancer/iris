//! virtio-mmio transport, version 2 (virtio 1.x), for machines without PCI.
//!
//! The register file is the one in the virtio spec (§4.2.2) and in Linux's
//! `virtio_mmio.c`, at the same offsets. It sits in a 64 KB window of a GIO
//! expansion slot on the Indy (IP24) and Indigo2 (IP22/IP28), with the
//! interrupt on that slot's line.
//!
//! **Byte order.** Registers are 32-bit and read or written as numbers: a
//! `lw` of `MagicValue` returns `0x74726976` on this big-endian bus, the same
//! as it would on a little-endian one (this is what QEMU's virtio-mmio does
//! on a big-endian target). The config space from `0x100` is bytes in the
//! device's little-endian layout; a byte read at offset N returns config byte
//! N, and a wider read assembles the bytes in memory order, as any other
//! memory-like region would. Drivers read config a byte at a time, as Linux
//! does, or swap. Rings in guest memory are little-endian (see `queue.rs`).

use std::sync::Arc;

use crate::irq::IrqLine;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead8, Device, Resettable, BUS_OK};

use super::{GuestMemory, VirtioCore};

/// Bytes of address space the window decodes (one 64 KB decode slot).
pub const WINDOW: u64 = 0x1_0000;

pub const MAGIC: u32 = 0x7472_6976; // "virt"
pub const VERSION: u32 = 2;
/// Vendor ID the register reads back: "IRIS".
pub const VENDOR: u32 = 0x4952_4953;

pub const REG_MAGIC: u64 = 0x000;
pub const REG_VERSION: u64 = 0x004;
pub const REG_DEVICE_ID: u64 = 0x008;
pub const REG_VENDOR_ID: u64 = 0x00c;
pub const REG_DEVICE_FEATURES: u64 = 0x010;
pub const REG_DEVICE_FEATURES_SEL: u64 = 0x014;
pub const REG_DRIVER_FEATURES: u64 = 0x020;
pub const REG_DRIVER_FEATURES_SEL: u64 = 0x024;
pub const REG_QUEUE_SEL: u64 = 0x030;
pub const REG_QUEUE_NUM_MAX: u64 = 0x034;
pub const REG_QUEUE_NUM: u64 = 0x038;
pub const REG_QUEUE_READY: u64 = 0x044;
pub const REG_QUEUE_NOTIFY: u64 = 0x050;
pub const REG_INTERRUPT_STATUS: u64 = 0x060;
pub const REG_INTERRUPT_ACK: u64 = 0x064;
pub const REG_STATUS: u64 = 0x070;
pub const REG_QUEUE_DESC_LOW: u64 = 0x080;
pub const REG_QUEUE_DESC_HIGH: u64 = 0x084;
pub const REG_QUEUE_DRIVER_LOW: u64 = 0x090;
pub const REG_QUEUE_DRIVER_HIGH: u64 = 0x094;
pub const REG_QUEUE_DEVICE_LOW: u64 = 0x0a0;
pub const REG_QUEUE_DEVICE_HIGH: u64 = 0x0a4;
pub const REG_CONFIG_GENERATION: u64 = 0x0fc;
pub const REG_CONFIG: u64 = 0x100;

pub struct VirtioMmio {
    core: VirtioCore,
    /// Selector registers: the driver writes these, then reads or writes the
    /// register they steer.
    sel: parking_lot::Mutex<Selectors>,
}

#[derive(Default)]
struct Selectors {
    device_features: u32,
    driver_features: u32,
    queue: u32,
}

impl VirtioMmio {
    /// Put `core` behind the register file, interrupting on `irq` and using
    /// `mem` for the rings.
    pub fn new(core: VirtioCore, irq: IrqLine, mem: Arc<dyn GuestMemory>) -> Self {
        core.connect_dma(mem);
        core.connect_interrupt(irq);
        Self { core, sel: Default::default() }
    }

    pub fn core(&self) -> &VirtioCore {
        &self.core
    }

    fn off(addr: u64) -> u64 {
        addr & (WINDOW - 1)
    }

    fn read_reg(&self, off: u64) -> u32 {
        let c = &self.core;
        let s = self.sel.lock();
        match off {
            REG_MAGIC => MAGIC,
            REG_VERSION => VERSION,
            REG_DEVICE_ID => c.device_id(),
            REG_VENDOR_ID => VENDOR,
            REG_DEVICE_FEATURES => c.device_features_word(s.device_features),
            REG_QUEUE_NUM_MAX => c.queue_max(s.queue as usize) as u32,
            REG_QUEUE_READY => c.with_queue(s.queue as usize, |q| q.ready as u32).unwrap_or(0),
            REG_INTERRUPT_STATUS => c.isr(),
            REG_STATUS => c.status() as u32,
            REG_CONFIG_GENERATION => c.config_generation(),
            REG_QUEUE_DESC_LOW => c.with_queue(s.queue as usize, |q| q.desc as u32).unwrap_or(0),
            REG_QUEUE_DESC_HIGH => c.with_queue(s.queue as usize, |q| (q.desc >> 32) as u32).unwrap_or(0),
            REG_QUEUE_DRIVER_LOW => c.with_queue(s.queue as usize, |q| q.avail as u32).unwrap_or(0),
            REG_QUEUE_DRIVER_HIGH => c.with_queue(s.queue as usize, |q| (q.avail >> 32) as u32).unwrap_or(0),
            REG_QUEUE_DEVICE_LOW => c.with_queue(s.queue as usize, |q| q.used as u32).unwrap_or(0),
            REG_QUEUE_DEVICE_HIGH => c.with_queue(s.queue as usize, |q| (q.used >> 32) as u32).unwrap_or(0),
            // Write-only registers read as zero.
            _ => 0,
        }
    }

    fn write_reg(&self, off: u64, val: u32) {
        let c = &self.core;
        let mut s = self.sel.lock();
        let q = s.queue as usize;
        match off {
            REG_DEVICE_FEATURES_SEL => s.device_features = val,
            REG_DRIVER_FEATURES_SEL => s.driver_features = val,
            REG_DRIVER_FEATURES => c.set_driver_features_word(s.driver_features, val),
            REG_QUEUE_SEL => s.queue = val,
            REG_QUEUE_NUM => {
                // Sizes the device cannot use are stored as zero, which leaves
                // the queue unusable: the driver sees QueueReady stay 0.
                let n = if val <= super::queue::QUEUE_MAX as u32 { val as u16 } else { 0 };
                c.with_queue(q, |qu| qu.size = n);
            }
            REG_QUEUE_READY => {
                c.with_queue(q, |qu| qu.ready = val & 1 != 0);
            }
            REG_QUEUE_NOTIFY => {
                drop(s);
                c.notify(val as usize);
            }
            REG_INTERRUPT_ACK => {
                drop(s);
                c.isr_ack(val);
            }
            REG_STATUS => {
                drop(s);
                c.set_status(val as u8);
            }
            REG_QUEUE_DESC_LOW => { c.with_queue(q, |qu| qu.desc = (qu.desc & !0xffff_ffff) | val as u64); }
            REG_QUEUE_DESC_HIGH => { c.with_queue(q, |qu| qu.desc = (qu.desc & 0xffff_ffff) | ((val as u64) << 32)); }
            REG_QUEUE_DRIVER_LOW => { c.with_queue(q, |qu| qu.avail = (qu.avail & !0xffff_ffff) | val as u64); }
            REG_QUEUE_DRIVER_HIGH => { c.with_queue(q, |qu| qu.avail = (qu.avail & 0xffff_ffff) | ((val as u64) << 32)); }
            REG_QUEUE_DEVICE_LOW => { c.with_queue(q, |qu| qu.used = (qu.used & !0xffff_ffff) | val as u64); }
            REG_QUEUE_DEVICE_HIGH => { c.with_queue(q, |qu| qu.used = (qu.used & 0xffff_ffff) | ((val as u64) << 32)); }
            _ => {}
        }
    }

    /// `n` config bytes from window offset `off`, in memory order.
    fn config_bytes<const N: usize>(&self, off: u64) -> [u8; N] {
        let mut b = [0u8; N];
        self.core.config_read((off - REG_CONFIG) as usize, &mut b);
        b
    }
}

impl BusDevice for VirtioMmio {
    fn read32(&self, addr: u64) -> BusRead32 {
        let off = Self::off(addr) & !3;
        if off >= REG_CONFIG {
            return BusRead32::ok(u32::from_be_bytes(self.config_bytes::<4>(off)));
        }
        BusRead32::ok(self.read_reg(off))
    }

    fn write32(&self, addr: u64, val: u32) -> u32 {
        let off = Self::off(addr) & !3;
        if off >= REG_CONFIG {
            self.core.config_write((off - REG_CONFIG) as usize, &val.to_be_bytes());
        } else {
            self.write_reg(off, val);
        }
        BUS_OK
    }

    fn read16(&self, addr: u64) -> BusRead16 {
        let off = Self::off(addr) & !1;
        if off >= REG_CONFIG {
            return BusRead16::ok(u16::from_be_bytes(self.config_bytes::<2>(off)));
        }
        // A register read through a half-word lane: the lane of the word.
        let w = self.read_reg(off & !3);
        BusRead16::ok((w >> (16 * (1 - ((off >> 1) & 1)))) as u16)
    }

    fn write16(&self, addr: u64, val: u16) -> u32 {
        let off = Self::off(addr) & !1;
        if off >= REG_CONFIG {
            self.core.config_write((off - REG_CONFIG) as usize, &val.to_be_bytes());
        }
        BUS_OK
    }

    fn read8(&self, addr: u64) -> BusRead8 {
        let off = Self::off(addr);
        if off >= REG_CONFIG {
            return BusRead8::ok(self.config_bytes::<1>(off)[0]);
        }
        let w = self.read_reg(off & !3);
        BusRead8::ok((w >> (8 * (3 - (off & 3)))) as u8)
    }

    fn write8(&self, addr: u64, val: u8) -> u32 {
        let off = Self::off(addr);
        if off >= REG_CONFIG {
            self.core.config_write((off - REG_CONFIG) as usize, &[val]);
        }
        BUS_OK
    }
}

impl Device for VirtioMmio {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {
        self.core.stop();
    }
    fn start(&self) {
        self.core.start();
    }
    fn is_running(&self) -> bool {
        self.core.is_running()
    }
    fn get_clock(&self) -> u64 {
        0
    }
}

impl Resettable for VirtioMmio {
    fn power_on(&self) {
        *self.sel.lock() = Selectors::default();
        self.core.power_on();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::virtio::backend::MemPort;
    use crate::dev::virtio::console::{Console, PortConfig, F_MULTIPORT};
    use crate::dev::virtio::{VecMemory, Waker, F_VERSION_1};
    use crate::irq::LevelProbe;

    const BASE: u64 = 0x1F60_0000;

    struct Rig {
        dev: VirtioMmio,
        mem: Arc<VecMemory>,
        port: Arc<MemPort>,
        irq: Arc<LevelProbe>,
    }

    fn rig() -> Rig {
        let port = MemPort::new();
        let waker = Waker::new();
        let core = VirtioCore::new(
            Box::new(Console::new(vec![PortConfig { name: None, console: true, backend: port.clone() }], waker.clone())),
            waker,
        );
        let mem = Arc::new(VecMemory::new(0x2_0000));
        let (irq, line) = LevelProbe::line();
        Rig { dev: VirtioMmio::new(core, line, mem.clone()), mem, port, irq }
    }

    impl Rig {
        fn w(&self, off: u64, v: u32) {
            assert_eq!(self.dev.write32(BASE + off, v), BUS_OK);
        }
        fn r(&self, off: u64) -> u32 {
            let r = self.dev.read32(BASE + off);
            assert!(r.is_ok());
            r.data
        }
    }

    #[test]
    fn identifies_itself_as_a_virtio_console() {
        let r = rig();
        assert_eq!(r.r(REG_MAGIC), 0x7472_6976);
        assert_eq!(r.r(REG_VERSION), 2);
        assert_eq!(r.r(REG_DEVICE_ID), 3);
        assert_eq!(r.r(REG_VENDOR_ID), VENDOR);
        r.w(REG_DEVICE_FEATURES_SEL, 0);
        assert_eq!(r.r(REG_DEVICE_FEATURES) as u64 & F_MULTIPORT, F_MULTIPORT);
        r.w(REG_DEVICE_FEATURES_SEL, 1);
        assert_eq!(r.r(REG_DEVICE_FEATURES), 1, "VERSION_1 is bit 32");
        // A register is a number: its high byte sits at the lowest address.
        assert_eq!(r.dev.read8(BASE + REG_MAGIC).data, b't');
        assert_eq!(r.dev.read8(BASE + REG_MAGIC + 3).data, b'v');
    }

    #[test]
    fn config_space_is_bytes_in_memory_order() {
        let r = rig();
        // max_nr_ports = 1, little-endian at config offset 4.
        assert_eq!(r.dev.read8(BASE + REG_CONFIG + 4).data, 1);
        assert_eq!(r.dev.read8(BASE + REG_CONFIG + 5).data, 0);
        assert_eq!(r.r(REG_CONFIG + 4), 0x0100_0000);
        // Past the end of config reads zero.
        assert_eq!(r.r(REG_CONFIG + 0x80), 0);
    }

    #[test]
    fn a_driver_can_bring_up_the_console_and_exchange_bytes() {
        let r = rig();
        // Negotiate: VERSION_1 only, so a plain single-port console.
        r.w(REG_STATUS, 1 | 2);
        r.w(REG_DRIVER_FEATURES_SEL, 0);
        r.w(REG_DRIVER_FEATURES, 0);
        r.w(REG_DRIVER_FEATURES_SEL, 1);
        r.w(REG_DRIVER_FEATURES, (F_VERSION_1 >> 32) as u32);
        r.w(REG_STATUS, 1 | 2 | 8);
        assert_eq!(r.r(REG_STATUS) & 8, 8, "FEATURES_OK stuck");

        // Queue 1 (transmitq): 8 entries.
        r.w(REG_QUEUE_SEL, 1);
        assert_eq!(r.r(REG_QUEUE_NUM_MAX), 256);
        assert_eq!(r.r(REG_QUEUE_READY), 0);
        r.w(REG_QUEUE_NUM, 8);
        r.w(REG_QUEUE_DESC_LOW, 0x1000);
        r.w(REG_QUEUE_DRIVER_LOW, 0x1200);
        r.w(REG_QUEUE_DEVICE_LOW, 0x1400);
        r.w(REG_QUEUE_READY, 1);
        assert_eq!(r.r(REG_QUEUE_READY), 1);
        assert_eq!(r.r(REG_QUEUE_DESC_LOW), 0x1000);
        r.w(REG_STATUS, 1 | 2 | 8 | 4);

        // One descriptor: "hi" at 0x1800.
        r.mem.write(0x1800, b"hi");
        let mut d = [0u8; 16];
        d[0..8].copy_from_slice(&0x1800u64.to_le_bytes());
        d[8..12].copy_from_slice(&2u32.to_le_bytes());
        r.mem.write(0x1000, &d);
        r.mem.write(0x1204, &0u16.to_le_bytes());
        r.mem.write(0x1202, &1u16.to_le_bytes());
        assert!(!r.irq.level());
        r.w(REG_QUEUE_NOTIFY, 1);

        assert_eq!(r.port.take_tx(), b"hi");
        assert_eq!(r.r(REG_INTERRUPT_STATUS), 1);
        assert!(r.irq.level(), "line is up until the driver acknowledges");
        r.w(REG_INTERRUPT_ACK, 1);
        assert_eq!(r.r(REG_INTERRUPT_STATUS), 0);
        assert!(!r.irq.level());

        // Reset.
        r.w(REG_STATUS, 0);
        assert_eq!(r.r(REG_STATUS), 0);
        r.w(REG_QUEUE_SEL, 1);
        assert_eq!(r.r(REG_QUEUE_READY), 0);
    }

    #[test]
    fn oversized_queue_stays_unusable() {
        let r = rig();
        r.w(REG_QUEUE_SEL, 0);
        r.w(REG_QUEUE_NUM, 4096);
        r.w(REG_QUEUE_READY, 1);
        assert!(r.dev.core().with_queue(0, |q| !q.usable()).unwrap());
    }

    /// The whole path a guest takes, on a real physical bus: a driver on a
    /// big-endian CPU stores little-endian ring words (`swap_bytes`) with
    /// `sw`, rings the doorbell through the slot's window, and the device
    /// reads them back through `BusMemory` in memory order.
    #[test]
    fn end_to_end_on_the_physical_bus() {
        use crate::dev::virtio::BusMemory;
        use crate::physical::Physical;
        use crate::ppmem::devmap::SLOT_SHIFT;
        use crate::ppmem::MappedMemory;

        struct Absent;
        impl BusDevice for Absent {}
        let phys = Physical::new(vec![crate::physical::RamBank::new(1)]);
        phys.set_default(Arc::new(Absent), "nothing");
        phys.fill_slots(0x30_0000 >> SLOT_SHIFT, 0x10_0000 >> SLOT_SHIFT, phys.bank_view(0, 0x30_0000));
        let port = MemPort::new();
        let waker = Waker::new();
        let core = VirtioCore::new(
            Box::new(Console::new(vec![PortConfig { name: None, console: true, backend: port.clone() }], waker.clone())),
            waker,
        );
        let (irq, line) = LevelProbe::line();
        let dev = Arc::new(VirtioMmio::new(core, line, Arc::new(BusMemory(phys.clone()))));
        phys.map(BASE, BASE + WINDOW, dev.clone(), "virtio-serial");

        let reg = |off: u64, v: u32| assert_eq!(phys.write32(BASE + off, v), BUS_OK);
        // Rings at 0x300000: desc 0x0, avail 0x200, used 0x400, data 0x800.
        let ring = 0x30_0000u64;
        reg(REG_STATUS, 3);
        reg(REG_DRIVER_FEATURES_SEL, 1);
        reg(REG_DRIVER_FEATURES, 1);
        reg(REG_STATUS, 0xb);
        reg(REG_QUEUE_SEL, 1);
        reg(REG_QUEUE_NUM, 4);
        reg(REG_QUEUE_DESC_LOW, ring as u32);
        reg(REG_QUEUE_DRIVER_LOW, ring as u32 + 0x200);
        reg(REG_QUEUE_DEVICE_LOW, ring as u32 + 0x400);
        reg(REG_QUEUE_READY, 1);
        reg(REG_STATUS, 0xf);

        // "hello" at 0x800; descriptor 0: addr, len, flags|next.
        for (i, b) in b"hello\0\0\0".chunks(4).enumerate() {
            phys.write32(ring + 0x800 + 4 * i as u64, u32::from_be_bytes(b.try_into().unwrap()));
        }
        let le = |v: u32| v.swap_bytes();
        phys.write32(ring, le((ring + 0x800) as u32)); // addr low
        phys.write32(ring + 4, 0); // addr high
        phys.write32(ring + 8, le(5)); // len
        phys.write32(ring + 12, 0); // flags 0, next 0
        phys.write32(ring + 0x200, 0); // avail flags, idx 0
        phys.write32(ring + 0x204, 0); // ring[0] = 0, ring[1]
        // avail.idx = 1: u16 LE at +2, so the byte at +2 is 1.
        phys.write16(ring + 0x202, 1u16.swap_bytes());
        reg(REG_QUEUE_NOTIFY, 1);

        assert_eq!(port.take_tx(), b"hello");
        assert!(irq.level());
        // used.idx (u16 LE at +0x402) is 1 and used[0] = {id 0, len 0}: the
        // device wrote nothing into a transmit buffer.
        assert_eq!(phys.read16(ring + 0x402).data, 1u16.swap_bytes());
        assert_eq!(phys.read32(ring + 0x404).data, 0);
        assert_eq!(phys.read32(ring + 0x408).data, 0);
    }
}
