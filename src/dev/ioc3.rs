//! The IOC3, the Octane's I/O controller: PCI device 2 behind the Bridge,
//! with the two serial ports (16550s in the SuperIO block), the DS1687
//! clock and NVRAM on its byte bus, the 1-wire master that reads the
//! MAC-address NIC, the GPIO latch, and (not yet) the keyboard and mouse,
//! the parallel port and the Ethernet.
//!
//! One 1 MB memory BAR; offsets are within it (`docs/Octane/ip30-address-map.md`
//! §7). Registers without a model read back what was written to them.

use std::collections::HashMap;

use parking_lot::Mutex;

use crate::dev::ns16550::Com16550;
use crate::dev::sgi_nic::{Ds2502, OneWire};
use crate::pci::{Bar, PciBusDevice, PciConfig, PCI_INTA};

const IOC3_MCR: u32 = 0x30;
const IOC3_GPDR: u32 = 0x3c;
/// The serial DMA control registers and the Ethernet MAC control: bit 31
/// is a reset that completes at once.
const IOC3_SSCR_A: u32 = 0xb8;
const IOC3_SSCR_B: u32 = 0xd4;
const IOC3_EMCR: u32 = 0xf0;
const IOC3_SSCR_A_ALIAS: u32 = 0x8000;
const IOC3_SSCR_B_ALIAS: u32 = 0xc000;
const SELF_CLEARING_RESET: u32 = 1 << 31;
/// The SuperIO control register's bit 22 reads 1 while no serial or
/// parallel DMA request is in flight; the PROM waits for it after
/// resetting the serial DMA engines.
const IOC3_SIO_CR: u32 = 0x28;
const SIO_CR_ARB_DIAG_IDLE: u32 = 1 << 22;
const IOC3_GPPR0: u32 = 0x40;
const IOC3_UARTA: u32 = 0x20178;
const IOC3_UARTB: u32 = 0x20170;
/// The DS1687's address and data ports: byte buses 1 and 2 (OpenBSD's
/// dsrtc.c for IP30), not the SuperIO's own RTC slot.
const IOC3_RTC_AD: u32 = 0xa0000;
const IOC3_RTC_DATA: u32 = 0xc0000;
const BYTEBUS_MASK: u32 = !0x1ffff;
/// GPIO pin 2 reads 1 on an original Octane, 0 on an Octane2.
const GPIO_CLASSIC: u32 = 2;

/// The DS1687 clock with 114 bytes of RAM in bank 0 and 128 more through
/// the extended registers of bank 1 (register A's DV0). IRIX keeps its
/// PROM environment here (sys/RACER/IP30nvram.h). The clock reads the
/// host's time; the RAM is saved to `nvram` on every change.
pub struct Ds1687 {
    reg: [u8; 128],
    ext: [u8; 128],
    index: u8,
    ext_addr: u8,
    path: Option<std::path::PathBuf>,
}

impl Ds1687 {
    pub fn new(path: Option<std::path::PathBuf>) -> Self {
        let mut d = Ds1687 { reg: [0; 128], ext: [0; 128], index: 0, ext_addr: 0, path };
        if let Some(saved) = d.path.as_ref().and_then(|p| std::fs::read(p).ok()) {
            if saved.len() == 256 {
                d.reg.copy_from_slice(&saved[..128]);
                d.ext.copy_from_slice(&saved[128..]);
            }
        }
        d.reg[0x0b] |= 0x02; // 24-hour
        d
    }

    pub fn save(&self) -> Option<&std::path::Path> {
        let p = self.path.as_deref()?;
        let mut all = self.reg.to_vec();
        all.extend_from_slice(&self.ext);
        if let Err(e) = std::fs::write(p, all) {
            eprintln!("iris: nvram: could not save {}: {e}", p.display());
        }
        Some(p)
    }

    fn bank1(&self) -> bool {
        self.reg[0x0a] & 0x10 != 0
    }

    /// Host time in the chip's format: BCD unless register B's DM.
    fn clock(&self, r: u8) -> u8 {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
        let days = t.div_euclid(86400);
        let secs = t.rem_euclid(86400);
        // civil from days (Howard Hinnant's algorithm)
        let z = days + 719468;
        let era = z.div_euclid(146097);
        let doe = z - era * 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = yoe + era * 400 + (month <= 2) as i64;
        let v = match r {
            0x00 => secs % 60,
            0x02 => secs / 60 % 60,
            0x04 => secs / 3600,
            0x06 => (days + 4).rem_euclid(7) + 1,
            0x07 => day,
            0x08 => month,
            0x09 => year % 100,
            _ => 0,
        } as u8;
        if self.reg[0x0b] & 0x04 != 0 { v } else { (v / 10) << 4 | v % 10 }
    }

    fn read(&mut self) -> u8 {
        let i = self.index & 0x7f;
        match i {
            0x00 | 0x02 | 0x04 | 0x06..=0x09 => self.clock(i),
            0x0a => self.reg[0x0a] & 0x7f, // never mid-update
            0x0c => 0,
            0x0d => 0x80, // VRT: the battery is fine
            0x40..=0x7f if self.bank1() => match i {
                0x40 => 0x71, // model: DS1687
                0x48 => if self.reg[0x0b] & 0x04 != 0 { 20 } else { 0x20 }, // century
                0x50 => self.ext_addr,
                0x53 => self.ext[(self.ext_addr & 0x7f) as usize],
                _ => 0,
            },
            _ => self.reg[i as usize],
        }
    }

    fn write(&mut self, val: u8) {
        let i = self.index & 0x7f;
        match i {
            0x00..=0x09 | 0x0c | 0x0d => {} // the clock follows the host
            0x40..=0x7f if self.bank1() => match i {
                0x50 => self.ext_addr = val,
                0x53 => {
                    self.ext[(self.ext_addr & 0x7f) as usize] = val;
                    self.save();
                }
                _ => {}
            },
            _ => {
                self.reg[i as usize] = val;
                if i >= 0x0e {
                    self.save();
                }
            }
        }
    }
}

struct Ioc3State {
    rtc: Ds1687,
    mcr: OneWire,
    regs: HashMap<u32, u32>,
}

pub struct Ioc3 {
    cfg: PciConfig,
    pub uart_a: Com16550,
    pub uart_b: Com16550,
    s: Mutex<Ioc3State>,
}

impl Ioc3 {
    /// `mac` goes into the MAC NIC; `nvram` is where the DS1687's RAM is kept.
    pub fn new(mac: [u8; 6], nvram: Option<std::path::PathBuf>) -> Self {
        let cfg = PciConfig::new(0x10a9, 0x0003, 0xff_0000, 1,
            [Bar::Mem(0x10_0000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], PCI_INTA);
        let mut regs = HashMap::new();
        regs.insert(IOC3_GPDR, 1 << GPIO_CLASSIC);
        regs.insert(IOC3_GPPR0 + 4 * GPIO_CLASSIC, 1);
        Ioc3 {
            cfg,
            uart_a: Com16550::named("ioc3-uart-a"),
            uart_b: Com16550::named("ioc3-uart-b"),
            // The IOC3's wire carries the MAC part and, as `hinv` places
            // them under pci/2, the front plane's and the power supply's
            // identification parts (snorlax's).
            s: Mutex::new(Ioc3State {
                rtc: Ds1687::new(nvram),
                mcr: OneWire::with_all(vec![
                    Ds2502::mac_nic(mac),
                    Ds2502::board_nic(0x0030_0891, "FP1", "030-0891-003", "A", "EDJ162"),
                    Ds2502::board_nic(0x0060_0035, "PWR.SPPLY.ER", "060-0035-003", "A", "AAE3230158"),
                ]),
                regs,
            }),
        }
    }

    /// Which UART a SuperIO offset is in, and the 16550 register.
    fn uart(&self, off: u32) -> Option<(&Com16550, u32)> {
        if (IOC3_UARTA..IOC3_UARTA + 8).contains(&off) {
            Some((&self.uart_a, off - IOC3_UARTA))
        } else if (IOC3_UARTB..IOC3_UARTB + 8).contains(&off) {
            Some((&self.uart_b, off - IOC3_UARTB))
        } else {
            None
        }
    }

    /// Pull host input into the UARTs; the service thread's job.
    pub fn service(&self) {
        self.uart_a.pump();
        self.uart_b.pump();
    }

    /// Write the clock's RAM to its file; returns the path.
    pub fn save_nvram(&self) -> Option<std::path::PathBuf> {
        self.s.lock().rtc.save().map(|p| p.to_path_buf())
    }

    pub fn power_on(&self) {
        use crate::traits::Resettable;
        self.uart_a.power_on();
        self.uart_b.power_on();
        let mut s = self.s.lock();
        s.regs.clear();
        s.regs.insert(IOC3_GPDR, 1 << GPIO_CLASSIC);
        s.regs.insert(IOC3_GPPR0 + 4 * GPIO_CLASSIC, 1);
    }

    fn read8(&self, off: u32) -> u8 {
        if let Some((u, r)) = self.uart(off) {
            // Com16550 decodes MACE's ISA spacing: register n at (n << 8) | 7.
            return u.read_reg((r << 8) | 7);
        }
        let mut s = self.s.lock();
        match off & BYTEBUS_MASK {
            IOC3_RTC_AD => return s.rtc.index,
            IOC3_RTC_DATA => return s.rtc.read(),
            _ => {}
        }
        let w = *s.regs.get(&(off & !3)).unwrap_or(&0);
        (w >> (8 * (3 - (off & 3)))) as u8
    }

    fn write8(&self, off: u32, val: u8) {
        if let Some((u, r)) = self.uart(off) {
            return u.write_reg((r << 8) | 7, val);
        }
        let mut s = self.s.lock();
        match off & BYTEBUS_MASK {
            IOC3_RTC_AD => return s.rtc.index = val,
            IOC3_RTC_DATA => return s.rtc.write(val),
            _ => {}
        }
        let sh = 8 * (3 - (off & 3));
        let w = s.regs.entry(off & !3).or_insert(0);
        *w = (*w & !(0xff << sh)) | (val as u32) << sh;
    }

    fn read32(&self, off: u32) -> u32 {
        let s = self.s.lock();
        match off & !3 {
            IOC3_MCR => s.mcr.read(),
            // No DMA engine is modelled, so none is ever busy.
            IOC3_SIO_CR => s.regs.get(&IOC3_SIO_CR).copied().unwrap_or(0) | SIO_CR_ARB_DIAG_IDLE,
            o => *s.regs.get(&o).unwrap_or(&0),
        }
    }

    fn write32(&self, off: u32, val: u32) {
        let mut s = self.s.lock();
        match off & !3 {
            IOC3_MCR => s.mcr.write(val),
            // A reset bit that software polls until it drops: it drops.
            o @ (IOC3_SSCR_A | IOC3_SSCR_B | IOC3_EMCR | IOC3_SSCR_A_ALIAS | IOC3_SSCR_B_ALIAS) => {
                s.regs.insert(o, val & !SELF_CLEARING_RESET);
            }
            o => {
                s.regs.insert(o, val);
            }
        }
    }
}

impl PciBusDevice for Ioc3 {
    fn config(&self) -> &PciConfig {
        &self.cfg
    }

    fn bar_read(&self, bar: usize, off: u32, width: usize) -> u32 {
        if bar != 0 {
            return 0;
        }
        match width {
            1 => self.read8(off) as u32,
            2 => (self.read32(off) >> (8 * (2 - (off & 2)))) as u16 as u32,
            _ => self.read32(off),
        }
    }

    fn bar_write(&self, bar: usize, off: u32, val: u32, width: usize) {
        if bar != 0 {
            return;
        }
        match width {
            1 => self.write8(off, val as u8),
            2 => {
                let sh = 8 * (2 - (off & 2));
                let old = self.read32(off);
                self.write32(off, (old & !(0xffff << sh)) | ((val & 0xffff) << sh));
            }
            _ => self.write32(off, val),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtc_nvram_and_extended_ram_hold_what_is_written() {
        let h = Ioc3::new([8, 0, 0x69, 1, 2, 3], None);
        let rtc = |r: u8, v: Option<u8>| {
            h.bar_write(0, IOC3_RTC_AD, r as u32, 1);
            match v {
                Some(v) => {
                    h.bar_write(0, IOC3_RTC_DATA, v as u32, 1);
                    0
                }
                None => h.bar_read(0, IOC3_RTC_DATA, 1) as u8,
            }
        };
        rtc(0x0e, Some(0x5a));
        assert_eq!(rtc(0x0e, None), 0x5a);
        assert_eq!(rtc(0x0d, None), 0x80); // valid RAM and time
        rtc(0x0a, Some(0x30)); // bank 1
        rtc(0x50, Some(0x07));
        rtc(0x53, Some(0xa5));
        rtc(0x50, Some(0x07));
        assert_eq!(rtc(0x53, None), 0xa5);
        assert!(rtc(0x00, None) < 0x60); // BCD seconds
    }

    #[test]
    fn the_uarts_are_in_the_superio_block() {
        let h = Ioc3::new([8, 0, 0x69, 1, 2, 3], None);
        // LSR: always ready to transmit.
        assert_eq!(h.bar_read(0, IOC3_UARTA + 5, 1) & 0x20, 0x20);
        h.bar_write(0, IOC3_UARTA + 7, 0x5a, 1); // scratch
        assert_eq!(h.bar_read(0, IOC3_UARTA + 7, 1), 0x5a);
        assert_eq!(h.bar_read(0, IOC3_UARTB + 7, 1), 0);
        // GPIO pin 2: an original Octane.
        assert_eq!(h.bar_read(0, IOC3_GPPR0 + 8, 4), 1);
    }
}
