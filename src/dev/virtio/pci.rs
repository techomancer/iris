//! virtio-pci transport, modern layout (virtio 1.x §4.1), for machines with a
//! PCI host bridge: the O2 (IP32, MACE PCI) and the Octane (IP30, Bridge).
//!
//! One memory BAR (BAR0, 16 KB) holds the four structures the vendor
//! capabilities in the config header point at:
//!
//! | offset | structure | capability `cfg_type` |
//! |---|---|---|
//! | `0x0000` | common configuration | 1 |
//! | `0x1000` | ISR status (read clears) | 3 |
//! | `0x2000` | device-specific configuration | 4 |
//! | `0x3000` | notifications, one dword per queue | 2 |
//!
//! Everything in the BAR is little-endian as the device sees it; the host
//! bridge has already applied its byte-lane rules (see `PciBusDevice`), so a
//! guest driver swaps exactly as it does for any other PCI device on that
//! machine.
//!
//! Interrupts are INTx on pin A, level-sensitive, cleared by reading the ISR.
//! MSI-X is not offered.

use std::sync::Arc;

use crate::irq::IrqLine;
use crate::pci::{Bar, PciBusDevice, PciConfig, PciDma, PCI_INTA};
use crate::traits::Resettable;

use super::{PciMemory, VirtioCore};

pub const VENDOR_ID: u16 = 0x1af4;
/// Modern virtio PCI device IDs are `0x1040 + virtio device ID`.
pub const DEVICE_ID_BASE: u16 = 0x1040;
/// PCI class 0x07 (simple communication), subclass 0x80 (other).
const CLASS_COMM_OTHER: u32 = 0x07_80_00;

const BAR_SIZE: u32 = 0x4000;
const COMMON: u32 = 0x0000;
const ISR: u32 = 0x1000;
const DEVICE: u32 = 0x2000;
const NOTIFY: u32 = 0x3000;
const COMMON_LEN: u32 = 0x38;
/// Each queue's doorbell is a dword of its own.
const NOTIFY_MULTIPLIER: u32 = 4;

fn cap(cfg_type: u8, next: u8, len: u8, offset: u32, length: u32, extra: Option<u32>) -> Vec<u32> {
    // struct virtio_pci_cap: cap_vndr(9) cap_next cap_len cfg_type | bar id pad pad | offset | length
    let mut d = vec![
        u32::from_le_bytes([9, next, len, cfg_type]),
        u32::from_le_bytes([0, 0, 0, 0]), // BAR 0
        offset,
        length,
    ];
    d.extend(extra);
    d
}

/// A virtio device as a PCI function.
pub struct VirtioPci {
    core: VirtioCore,
    pci: PciConfig,
    sel: parking_lot::Mutex<Selectors>,
}

#[derive(Default)]
struct Selectors {
    device_features: u32,
    driver_features: u32,
    queue: u16,
}

impl VirtioPci {
    pub fn new(core: VirtioCore) -> Self {
        let id = core.device_id() as u16;
        // The capability list starts at 0x40: common, notify, ISR, device.
        let mut caps = Vec::new();
        caps.extend(cap(1, 0x50, 16, COMMON, COMMON_LEN, None));
        caps.extend(cap(2, 0x64, 20, NOTIFY, NOTIFY_MULTIPLIER * 64, Some(NOTIFY_MULTIPLIER)));
        caps.extend(cap(3, 0x74, 16, ISR, 4, None));
        caps.extend(cap(4, 0, 16, DEVICE, 0x1000, None));
        let pci = PciConfig::new(
            VENDOR_ID,
            DEVICE_ID_BASE + id,
            CLASS_COMM_OTHER,
            1,
            [Bar::Mem(BAR_SIZE), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused],
            PCI_INTA,
        )
        .with_capabilities(16, &caps);
        // Subsystem: vendor and the virtio device ID, dword 11.
        pci.write(11, ((id as u32) << 16) | VENDOR_ID as u32);
        Self { core, pci, sel: Default::default() }
    }

    pub fn core(&self) -> &VirtioCore {
        &self.core
    }

    fn read_common(&self, off: u32) -> u32 {
        let c = &self.core;
        let s = self.sel.lock();
        let q = s.queue as usize;
        match off {
            0x00 => s.device_features,
            0x04 => c.device_features_word(s.device_features),
            0x08 => s.driver_features,
            // 0x0c driver_feature reads back as zero: write-only in practice.
            0x10 => 0xffff | ((c.num_queues() as u32) << 16), // msix_config = NO_VECTOR, num_queues
            0x14 => c.status() as u32 | (c.config_generation() & 0xff) << 8 | (s.queue as u32) << 16,
            0x18 => {
                let size = c.with_queue(q, |qu| qu.size).unwrap_or(0);
                let max = c.queue_max(q);
                // queue_size reads the size in use, or the maximum before the driver chooses.
                (if size == 0 { max } else { size }) as u32 | 0xffff_0000 // queue_msix_vector = NO_VECTOR
            }
            0x1c => {
                let ready = c.with_queue(q, |qu| qu.ready as u32).unwrap_or(0);
                ready | (q as u32) << 16 // queue_enable, queue_notify_off = queue index
            }
            0x20 => c.with_queue(q, |qu| qu.desc as u32).unwrap_or(0),
            0x24 => c.with_queue(q, |qu| (qu.desc >> 32) as u32).unwrap_or(0),
            0x28 => c.with_queue(q, |qu| qu.avail as u32).unwrap_or(0),
            0x2c => c.with_queue(q, |qu| (qu.avail >> 32) as u32).unwrap_or(0),
            0x30 => c.with_queue(q, |qu| qu.used as u32).unwrap_or(0),
            0x34 => c.with_queue(q, |qu| (qu.used >> 32) as u32).unwrap_or(0),
            _ => 0,
        }
    }

    /// A write of `width` bytes at `off` of the common structure; `val` is
    /// the `width`-byte number the device sees, not positioned in a dword.
    fn write_common(&self, off: u32, val: u32, width: usize) {
        let c = &self.core;
        let mut s = self.sel.lock();
        let q = s.queue as usize;
        let v = if width == 4 { val } else { val & ((1u32 << (width * 8)) - 1) };
        match (off, width) {
            (0x00, 4) => s.device_features = v,
            (0x08, 4) => s.driver_features = v,
            (0x0c, 4) => c.set_driver_features_word(s.driver_features, v),
            (0x14, 1) => {
                drop(s);
                c.set_status(v as u8);
            }
            (0x16, 2) => s.queue = v as u16,
            (0x18, 2) => {
                let n = if v <= super::queue::QUEUE_MAX as u32 { v as u16 } else { 0 };
                c.with_queue(q, |qu| qu.size = n);
            }
            (0x1c, 2) => {
                c.with_queue(q, |qu| qu.ready = v & 1 != 0);
            }
            (0x20, 4) => { c.with_queue(q, |qu| qu.desc = (qu.desc & !0xffff_ffff) | v as u64); }
            (0x24, 4) => { c.with_queue(q, |qu| qu.desc = (qu.desc & 0xffff_ffff) | ((v as u64) << 32)); }
            (0x28, 4) => { c.with_queue(q, |qu| qu.avail = (qu.avail & !0xffff_ffff) | v as u64); }
            (0x2c, 4) => { c.with_queue(q, |qu| qu.avail = (qu.avail & 0xffff_ffff) | ((v as u64) << 32)); }
            (0x30, 4) => { c.with_queue(q, |qu| qu.used = (qu.used & !0xffff_ffff) | v as u64); }
            (0x34, 4) => { c.with_queue(q, |qu| qu.used = (qu.used & 0xffff_ffff) | ((v as u64) << 32)); }
            _ => {}
        }
    }
}

/// Bytes `off..off+width` of the little-endian dword stream `word(i)`
/// produces, as a number. Lets registers be defined as dwords and read at
/// any width the bus issues.
fn lanes(off: u32, width: usize, word: impl Fn(u32) -> u32) -> u32 {
    let w = word(off & !3);
    match width {
        4 => w,
        _ => (w >> ((off & 3) * 8)) & ((1u32 << (width * 8)) - 1),
    }
}

impl PciBusDevice for VirtioPci {
    fn config(&self) -> &PciConfig {
        &self.pci
    }

    fn bar_read(&self, _bar: usize, off: u32, width: usize) -> u32 {
        match off & (BAR_SIZE - 1) {
            o if o >= NOTIFY => 0,
            o if o >= DEVICE => {
                let mut b = [0u8; 4];
                self.core.config_read((o - DEVICE) as usize, &mut b[..width]);
                u32::from_le_bytes(b)
            }
            o if o >= ISR => {
                // Reading clears; only the first byte means anything.
                if o == ISR { self.core.isr_read_clear() & 0xff } else { 0 }
            }
            o => lanes(o, width, |w| self.read_common(w - COMMON)),
        }
    }

    fn bar_write(&self, _bar: usize, off: u32, val: u32, width: usize) {
        match off & (BAR_SIZE - 1) {
            o if o >= NOTIFY => {
                // The doorbell: queue number is the dword written, or the value.
                let q = (o - NOTIFY) / NOTIFY_MULTIPLIER;
                self.core.notify(q as usize);
            }
            o if o >= DEVICE => {
                let b = val.to_le_bytes();
                self.core.config_write((o - DEVICE) as usize, &b[..width]);
            }
            o if o >= ISR => {}
            o => self.write_common(o - COMMON, val, width),
        }
    }

    fn connect_interrupt(&self, line: IrqLine) {
        self.core.connect_interrupt(line);
    }

    fn connect_dma(&self, dma: Arc<dyn PciDma>) {
        self.core.connect_dma(Arc::new(PciMemory(dma)));
    }
}

impl crate::traits::Device for VirtioPci {
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

impl Resettable for VirtioPci {
    fn power_on(&self) {
        self.pci.reset();
        *self.sel.lock() = Selectors::default();
        self.core.power_on();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::virtio::backend::MemPort;
    use crate::dev::virtio::console::{Console, PortConfig};
    use crate::dev::virtio::{VecMemory, Waker, F_VERSION_1};
    use crate::irq::LevelProbe;
    use crate::pci::PciDmaStatus;

    struct Dma(Arc<VecMemory>);
    impl PciDma for Dma {
        fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
            use crate::dev::virtio::GuestMemory;
            if self.0.read(addr, buf) { PciDmaStatus::Ok } else { PciDmaStatus::MasterAbort }
        }
        fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
            use crate::dev::virtio::GuestMemory;
            if self.0.write(addr, data) { PciDmaStatus::Ok } else { PciDmaStatus::MasterAbort }
        }
    }

    fn rig() -> (Arc<VirtioPci>, Arc<VecMemory>, Arc<MemPort>, Arc<LevelProbe>) {
        let port = MemPort::new();
        let waker = Waker::new();
        let core = VirtioCore::new(
            Box::new(Console::new(vec![PortConfig { name: None, console: true, backend: port.clone() }], waker.clone())),
            waker,
        );
        let dev = Arc::new(VirtioPci::new(core));
        let mem = Arc::new(VecMemory::new(0x2_0000));
        let (probe, line) = LevelProbe::line();
        dev.connect_dma(Arc::new(Dma(mem.clone())));
        dev.connect_interrupt(line);
        (dev, mem, port, probe)
    }

    #[test]
    fn header_and_capability_list() {
        let (dev, ..) = rig();
        let c = dev.config();
        assert_eq!(c.read(0), (0x1043 << 16) | 0x1af4);
        assert_eq!(c.read(2) >> 8, 0x07_80_00);
        assert_eq!(c.read(1) & 0x0010_0000, 0x0010_0000, "capability list present");
        assert_eq!(c.read(13) & 0xff, 0x40);
        assert_eq!(c.read(11), (3 << 16) | 0x1af4, "subsystem");

        // Walk the list the way a driver does.
        let mut at = 0x40usize;
        let mut types = Vec::new();
        while at != 0 {
            let d = c.read(at / 4);
            assert_eq!(d & 0xff, 9, "vendor-specific capability");
            types.push(((d >> 24) as u8, c.read(at / 4 + 2), c.read(at / 4 + 3)));
            at = ((d >> 8) & 0xff) as usize;
        }
        assert_eq!(types, vec![(1, COMMON, COMMON_LEN), (2, NOTIFY, 256), (3, ISR, 4), (4, DEVICE, 0x1000)]);
        assert_eq!(c.read(0x50 / 4 + 4), NOTIFY_MULTIPLIER);

        // Firmware cannot scribble on the list, but the line register is writable.
        c.write(0x40 / 4, 0);
        assert_eq!(c.read(0x40 / 4) & 0xff, 9);
        c.write(15, 0x2a);
        assert_eq!(c.read(15) & 0xff, 0x2a);
        // BAR size probe.
        c.write(4, 0xffff_ffff);
        assert_eq!(c.read(4), !(BAR_SIZE - 1));
    }

    #[test]
    fn driver_brings_up_queue_and_sends() {
        let (dev, mem, port, irq) = rig();
        let w = |off: u32, v: u32, n: usize| dev.bar_write(0, off, v, n);
        let r = |off: u32, n: usize| dev.bar_read(0, off, n);

        assert_eq!(r(0x12, 2), 1 + 3, "num_queues: rx, tx, control rx/tx");
        w(0x14, 3, 1);
        w(0x08, 1, 4); // driver_feature_select = 1
        w(0x0c, (F_VERSION_1 >> 32) as u32, 4);
        w(0x14, 0xb, 1);
        assert_eq!(r(0x14, 1) & 8, 8);

        w(0x16, 1, 2); // queue_select
        assert_eq!(r(0x18, 2), 256);
        w(0x18, 8, 2);
        w(0x20, 0x1000, 4);
        w(0x28, 0x1200, 4);
        w(0x30, 0x1400, 4);
        w(0x1c, 1, 2);
        assert_eq!(r(0x1c, 2), 1);
        assert_eq!(r(0x1e, 2), 1, "queue_notify_off");
        w(0x14, 0xf, 1);

        mem.write(0x1800, b"pci");
        let mut d = [0u8; 16];
        d[0..8].copy_from_slice(&0x1800u64.to_le_bytes());
        d[8..12].copy_from_slice(&3u32.to_le_bytes());
        use crate::dev::virtio::GuestMemory;
        mem.write(0x1000, &d);
        mem.write(0x1202, &1u16.to_le_bytes());
        w(NOTIFY + NOTIFY_MULTIPLIER * 1, 1, 2);

        assert_eq!(port.take_tx(), b"pci");
        assert!(irq.level());
        assert_eq!(r(ISR, 1), 1);
        assert!(!irq.level(), "reading the ISR clears it");
        assert_eq!(r(ISR, 1), 0);
    }

    #[test]
    fn device_config_reads_little_endian_at_any_width() {
        let (dev, ..) = rig();
        // max_nr_ports = 1 at config offset 4.
        assert_eq!(dev.bar_read(0, DEVICE + 4, 4), 1);
        assert_eq!(dev.bar_read(0, DEVICE + 4, 2), 1);
        assert_eq!(dev.bar_read(0, DEVICE + 6, 2), 0);
        assert_eq!(dev.bar_read(0, DEVICE + 4, 1), 1);
    }
}
