//! The SGI Octane (IP30): its physical bus. HEART, the Xbow and its
//! widgets, the Bridge with the PCI devices and the PROM flash behind it,
//! and memory where HEART's MEMCFG puts it. The map is
//! `docs/Octane/ip30-address-map.md`; this is its §11.
//!
//! Like the O2, the Octane is not wired into `Physical`: its map has
//! nothing in common with the Indy's, and `MipsCpu::new` takes any
//! `Arc<dyn BusDevice>`. A short chain of range checks decodes it: the
//! low memory alias, HEART's processor-side registers, the sixteen widget
//! windows, memory. `ip30_machine` drives it.
//!
//! The bus takes 32-bit addresses, as on the other machines. Everything
//! the PROM and an early kernel touch is below 4 GB; the large and huge
//! widget windows and memory beyond 4 GB wait for a wider bus.
//!
//! Set IRIS_IP30_TRACE=1 to log the first access to each register; that
//! is how the next thing to model gets found.

use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::dev::heart::{Heart, HEART_WIDGET};
use crate::dev::heart_mem::{Decode, HeartMemory, LOW_ALIAS_SIZE, RAM_BASE};
use crate::dev::ioc3::Ioc3;
use crate::dev::sflash::Flash;
use crate::dev::xbow::Xbow;
use crate::dev::xbridge::{octane_slots, Bridge};
use crate::ip32::Stub;
use crate::pci::PciBusDevice;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_ERR, BUS_OK};

pub const HEART_PIU_BASE: u32 = 0x0ff0_0000;
pub const HEART_PIU_END: u32 = 0x1000_0000;
pub const WIDGET_SPACE_BASE: u32 = 0x1000_0000;
pub const WIDGET_SPACE_END: u32 = 0x2000_0000;
const WIDGET_SIZE: u32 = 0x0100_0000;
pub const XBOW_WIDGET: u32 = 0;
pub const GFX_WIDGET: u32 = 0xc;
pub const BRIDGE_WIDGET: u32 = 0xf;
/// The PROM, as the CPU sees it: the Bridge's flash select.
pub const PROM_BASE: u32 = 0x1fc0_0000;

/// What is plugged into the Octane beyond its memory, and where the
/// files it keeps between runs are.
#[derive(Clone, Debug, Default)]
pub struct Ip30Options {
    /// The flash file: a copy of the PROM image whose last segment, the
    /// PROM's data store, the guest rewrites.
    pub flash_file: Option<std::path::PathBuf>,
    /// The DS1687's RAM, where the PROM keeps its environment.
    pub nvram_file: Option<std::path::PathBuf>,
    /// The Ethernet address in the MAC NIC.
    pub mac: [u8; 6],
}

pub struct Ip30Bus {
    pub ram: Mutex<HeartMemory>,
    pub heart: Heart,
    pub xbow: Xbow,
    pub bridge: Arc<Bridge>,
    pub ioc3: Arc<Ioc3>,
    /// The graphics widget: a register file until HQ4 and IMPACT sit there.
    pub gfx: Stub,
    /// Accesses that hit nothing, first 64 kept, as `(addr, is_write)`.
    unmapped: Mutex<Vec<(u32, bool)>>,
    /// The ppmem window's base and the RAM bitmap the memory space keeps,
    /// once the machine has wired them: aligned RAM reads then skip the
    /// RAM lock.
    fast_ram: std::sync::OnceLock<(usize, usize)>,
    /// IRIS_IP30_TRACE=1: the first access to each register. =2: also
    /// every access to the IOC3's window, in order, for following a
    /// driver's conversation with it.
    trace: bool,
    trace_ioc3: bool,
    seen: Mutex<HashSet<(u32, bool)>>,
}

impl Ip30Bus {
    /// `banks_mb` is the DIMM pair in each of HEART's four banks; `prom`
    /// the flash image (1 MB, the reset vector at its start).
    pub fn new(banks_mb: [u32; 4], prom: &[u8], opts: Ip30Options) -> Self {
        let heart = Heart::new(banks_mb);
        let ram = HeartMemory::new(banks_mb, heart.memcfg());
        let ioc3 = Arc::new(Ioc3::new(opts.mac, opts.nvram_file.clone()));
        let flash = Flash::new(prom, opts.flash_file.clone());
        let bridge = Arc::new(Bridge::new(octane_slots(ioc3.clone()), flash));
        // The IOC3's interrupt pins: Ethernet on 2, serial on 4 (IP30.h).
        let ctl: Arc<dyn crate::irq::InterruptController> = bridge.clone();
        ioc3.uart_a.connect_irq(crate::irq::IrqLine::new(ctl.clone(), 4));
        ioc3.uart_b.connect_irq(crate::irq::IrqLine::new(ctl, 4));
        Ip30Bus {
            ram: Mutex::new(ram),
            heart,
            // No graphics widget on the Xbow's links yet: the PROM takes
            // the serial console. The register file at widget 12 still
            // answers if something looks.
            xbow: Xbow::new(&[HEART_WIDGET, BRIDGE_WIDGET]),
            bridge,
            ioc3,
            gfx: Stub::new("gfx"),
            unmapped: Mutex::new(Vec::new()),
            fast_ram: std::sync::OnceLock::new(),
            trace: std::env::var_os("IRIS_IP30_TRACE").is_some(),
            trace_ioc3: std::env::var("IRIS_IP30_TRACE").is_ok_and(|v| v == "2"),
            seen: Mutex::new(HashSet::new()),
        }
    }

    /// Where HEART's interrupt levels go: the CPU's interrupt word. Also
    /// wires the Bridge's interrupt packets to HEART's ISR.
    pub fn connect_cpu_irq(self: &Arc<Self>, word: *const AtomicU64) {
        self.heart.set_interrupts(word);
        let me = Arc::downgrade(self);
        self.bridge.connect_interrupts(Arc::new(move |vec, level| {
            if let Some(bus) = me.upgrade() {
                bus.heart.set_vector(vec, level);
            }
        }));
    }

    /// Let aligned RAM reads go straight to the ppmem window: its base and
    /// the bitmap the memory space publishes (the CPU's, as `ppmem_bitmap_ptr`).
    pub fn set_fast_ram(&self, window: *mut u8, bitmap: *mut u64) {
        if !window.is_null() && !bitmap.is_null() {
            let _ = self.fast_ram.set((window as usize, bitmap as usize));
        }
    }

    pub fn unmapped(&self) -> Vec<(u32, bool)> {
        self.unmapped.lock().clone()
    }

    /// What the service thread does between instructions: host input into
    /// the UARTs, HEART's compare.
    pub fn service(&self) {
        self.ioc3.service();
        self.heart.poll_timer();
    }

    /// Write the flash and the clock's RAM out, if they changed.
    pub fn save(&self) {
        self.bridge.flash.save();
    }

    /// A cold reset: the chips back to power-on; memory and the flash keep
    /// their contents, as on the hardware.
    pub fn power_on(&self) {
        self.heart.power_on();
        self.xbow.power_on();
        self.bridge.power_on();
        self.ioc3.power_on();
        let cfg = self.heart.memcfg();
        self.ram.lock().set_memcfg(cfg);
    }

    fn miss(&self, addr: u32, write: bool) {
        let mut u = self.unmapped.lock();
        if u.len() < 64 && !u.iter().any(|&(a, w)| a == addr && w == write) {
            u.push((addr, write));
            if self.trace {
                eprintln!("ip30: {} {addr:08x}: nothing there", if write { "W" } else { "R" });
            }
        }
    }

    fn note(&self, addr: u32, write: bool, size: u32, val: u64) {
        if !self.trace || addr >= RAM_BASE as u32 {
            return;
        }
        if self.trace_ioc3 && (0x1f60_0000..0x1f70_0000).contains(&addr) {
            eprintln!("ioc3: {} {:05x} size {} {:#x}", if write { "W" } else { "R" }, addr - 0x1f60_0000, size, val);
            return;
        }
        let memcfg = (HEART_PIU_BASE + 0x20..HEART_PIU_BASE + 0x40).contains(&addr);
        if memcfg || self.seen.lock().insert((addr & !7, write)) {
            eprintln!("ip30: {} {:08x} size {} {:#x}", if write { "W" } else { "R" }, addr, size, val);
        }
    }

    fn is_ram(addr: u32) -> bool {
        (addr as u64) < LOW_ALIAS_SIZE || addr >= RAM_BASE as u32
    }

    fn read_ram(&self, addr: u32, n: u32) -> Option<u64> {
        // Mapped RAM, straight from the window: an aligned word is a host
        // u32 in the window's layout (see HeartMemory::get_be).
        if addr >= RAM_BASE as u32 {
            if let Some(&(win, bm)) = self.fast_ram.get() {
                let bits = unsafe { std::ptr::read_volatile(bm as *const u64) };
                if bits & (1u64 << (addr >> crate::ppmem::BITMAP_SHIFT)) != 0 && addr % n == 0 {
                    let p = win + addr as usize;
                    let v = unsafe {
                        match n {
                            1 => *((p ^ 3) as *const u8) as u64,
                            2 => ((p ^ 2) as *const u16).read_unaligned() as u64,
                            4 => (p as *const u32).read_unaligned() as u64,
                            _ => ((p as *const u32).read_unaligned() as u64) << 32 | ((p + 4) as *const u32).read_unaligned() as u64,
                        }
                    };
                    return Some(v);
                }
            }
        }
        let r = self.ram.lock();
        match r.decode(addr as u64) {
            Decode::Ram(off) if off + n as usize <= r.len() && addr % n == 0 => Some(r.get_be(off, n)),
            // A valid bank with no DIMMs: what the memory bus last carried.
            Decode::Empty => Some(0),
            _ => None,
        }
    }

    fn write_ram(&self, addr: u32, n: u32, val: u64) -> bool {
        let mut r = self.ram.lock();
        match r.decode(addr as u64) {
            Decode::Ram(off) if off + n as usize <= r.len() && addr % n == 0 => {
                r.put_be(off, n, val);
                true
            }
            // Swallowed by the memory controller.
            Decode::Empty => true,
            _ => false,
        }
    }

    fn read(&self, addr: u32, size: u32) -> Option<u64> {
        if Self::is_ram(addr) {
            return self.read_ram(addr, size);
        }
        let v = if (HEART_PIU_BASE..HEART_PIU_END).contains(&addr) {
            Some(narrow(self.heart.piu_read((addr - HEART_PIU_BASE) & !7), addr, size))
        } else if (WIDGET_SPACE_BASE..WIDGET_SPACE_END).contains(&addr) {
            let w = (addr - WIDGET_SPACE_BASE) / WIDGET_SIZE;
            let off = (addr - WIDGET_SPACE_BASE) % WIDGET_SIZE;
            match w {
                XBOW_WIDGET => Some(match size {
                    8 => self.xbow.read32((off & !7) | 4) as u64,
                    _ => crate::dev::xbridge::narrow32(self.xbow.read32(off & !3), off, size),
                }),
                HEART_WIDGET => Some(narrow(self.heart.widget_read(off), addr, size)),
                GFX_WIDGET => Some(match size {
                    1 => self.gfx.read8(off as u64).data as u64,
                    2 => self.gfx.read16(off as u64).data as u64,
                    4 => self.gfx.read32(off as u64).data as u64,
                    _ => self.gfx.read64(off as u64).data,
                }),
                BRIDGE_WIDGET => self.bridge.read(off, size),
                _ => None,
            }
        } else {
            None
        };
        self.note(addr, false, size, v.unwrap_or(u64::MAX));
        v
    }

    fn write(&self, addr: u32, size: u32, val: u64) -> bool {
        if Self::is_ram(addr) {
            return self.write_ram(addr, size, val);
        }
        self.note(addr, true, size, val);
        if (HEART_PIU_BASE..HEART_PIU_END).contains(&addr) {
            let off = (addr - HEART_PIU_BASE) & !7;
            let old = if size == 8 { 0 } else { self.heart.piu_read(off) };
            if self.heart.piu_write(off, widen(old, addr, size, val)) {
                // MEMCFG: place the banks again. The write is the CPU's,
                // which is the window's contract.
                let cfg = self.heart.memcfg();
                self.ram.lock().set_memcfg(cfg);
            }
            return true;
        }
        if (WIDGET_SPACE_BASE..WIDGET_SPACE_END).contains(&addr) {
            let w = (addr - WIDGET_SPACE_BASE) / WIDGET_SIZE;
            let off = (addr - WIDGET_SPACE_BASE) % WIDGET_SIZE;
            return match w {
                XBOW_WIDGET => {
                    if size == 8 {
                        self.xbow.write32((off & !7) | 4, val as u32);
                    } else {
                        let old = self.xbow.read32(off & !3);
                        self.xbow.write32(off & !3, crate::dev::xbridge::widen32(old, off, size, val as u32));
                    }
                    true
                }
                HEART_WIDGET => {
                    let old = self.heart.widget_read(off);
                    self.heart.widget_write(off, widen(old, addr, size, val));
                    true
                }
                GFX_WIDGET => {
                    match size {
                        1 => self.gfx.write8(off as u64, val as u8),
                        2 => self.gfx.write16(off as u64, val as u16),
                        4 => self.gfx.write32(off as u64, val as u32),
                        _ => self.gfx.write64(off as u64, val),
                    };
                    true
                }
                BRIDGE_WIDGET => self.bridge.write(off, size, val),
                _ => false,
            };
        }
        false
    }
}

/// The `size` bytes at `addr` of the big-endian doubleword `v`.
fn narrow(v: u64, addr: u32, size: u32) -> u64 {
    if size == 8 {
        return v;
    }
    let sh = 8 * (8 - size - (addr & 7 & !(size - 1)));
    (v >> sh) & ((1u64 << (8 * size)) - 1)
}

/// `old` with the `size` bytes at `addr` replaced by `val`.
fn widen(old: u64, addr: u32, size: u32, val: u64) -> u64 {
    if size == 8 {
        return val;
    }
    let sh = 8 * (8 - size - (addr & 7 & !(size - 1)));
    let m = ((1u64 << (8 * size)) - 1) << sh;
    (old & !m) | ((val << sh) & m)
}

impl BusDevice for Ip30Bus {
    /// Memory pages have generations jitv2 can compile against; the PROM
    /// and devices have none, so nothing there is compiled.
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, addr: u64) -> *const AtomicU64 {
        let addr = addr as u32;
        if Self::is_ram(addr) {
            let r = self.ram.lock();
            if let Decode::Ram(off) = r.decode(addr as u64) {
                return r.gen_ptr(off);
            }
        }
        std::ptr::null()
    }

    fn read8(&self, addr: u64) -> BusRead8 {
        let addr = addr as u32;
        match self.read(addr, 1) {
            Some(v) => BusRead8::ok(v as u8),
            None => { self.miss(addr, false); BusRead8::err() }
        }
    }
    fn read16(&self, addr: u64) -> BusRead16 {
        let addr = addr as u32;
        match self.read(addr, 2) {
            Some(v) => BusRead16::ok(v as u16),
            None => { self.miss(addr, false); BusRead16::err() }
        }
    }
    fn read32(&self, addr: u64) -> BusRead32 {
        let addr = addr as u32;
        match self.read(addr, 4) {
            Some(v) => BusRead32::ok(v as u32),
            None => { self.miss(addr, false); BusRead32::err() }
        }
    }
    fn read64(&self, addr: u64) -> BusRead64 {
        let addr = addr as u32;
        match self.read(addr, 8) {
            Some(v) => BusRead64::ok(v),
            None => { self.miss(addr, false); BusRead64::err() }
        }
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        let addr = addr as u32;
        if self.write(addr, 1, val as u64) { BUS_OK } else { self.miss(addr, true); BUS_ERR }
    }
    fn write16(&self, addr: u64, val: u16) -> u32 {
        let addr = addr as u32;
        if self.write(addr, 2, val as u64) { BUS_OK } else { self.miss(addr, true); BUS_ERR }
    }
    fn write32(&self, addr: u64, val: u32) -> u32 {
        let addr = addr as u32;
        if self.write(addr, 4, val as u64) { BUS_OK } else { self.miss(addr, true); BUS_ERR }
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        let addr = addr as u32;
        if self.write(addr, 8, val) { BUS_OK } else { self.miss(addr, true); BUS_ERR }
    }
}

impl crate::traits::Device for Ip30Bus {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("ip30".to_string(), "Octane: ip30 unmapped | memcfg | heart | bridge | gfx".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "ip30" {
            return Err("Command not found".to_string());
        }
        let out: String = match args.first().copied().unwrap_or("") {
            "unmapped" => {
                let u = self.unmapped();
                let list: Vec<String> = u.iter().map(|(a, wr)| format!("{} {a:#010x}", if *wr { "w" } else { "r" })).collect();
                format!("{} unmapped accesses: {}", u.len(), list.join("; "))
            }
            "memcfg" => {
                let r = self.ram.lock();
                let cfg = r.memcfg();
                let mut s = format!("MEMCFG {:08x} {:08x} {:08x} {:08x}; {} MB fitted\n", cfg[0], cfg[1], cfg[2], cfg[3], r.len() >> 20);
                for (base, span, real) in r.layout() {
                    s.push_str(&format!("  {base:#011x} span {} MB, DIMMs {} MB\n", span >> 20, real >> 20));
                }
                s
            }
            "heart" => format!(
                "count {:#x} compare {:#x} isr {:#018x} imr0 {:#018x} mode {:#x}",
                self.heart.count(), self.heart.piu_read(0x30000), self.heart.piu_read(0x10030),
                self.heart.piu_read(0x10000), self.heart.piu_read(0)
            ),
            "bridge" => {
                let (wid, addr) = self.bridge.interrupt_destination();
                let mut s = format!("interrupt destination widget {wid} address {addr:#x}\n");
                for n in 0..8 {
                    s.push_str(&format!("  device {n}: {:#010x} int addr {:#010x}\n",
                        self.bridge.read(0x204 + 8 * n, 4).unwrap_or(0), self.bridge.read(0x134 + 8 * n, 4).unwrap_or(0)));
                }
                s
            }
            "gfx" => {
                let (r, wr) = self.gfx.counts();
                let hot: Vec<String> = self.gfx.hottest().iter().take(8).map(|(o, r, w)| format!("{o:#x}: {r}r {w}w")).collect();
                format!("graphics widget: {r} reads, {wr} writes; {}", hot.join(", "))
            }
            _ => return Err("Usage: ip30 unmapped | memcfg | heart | bridge | gfx".to_string()),
        };
        writeln!(w, "{out}").map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bus() -> Ip30Bus {
        let mut prom = vec![0u8; 0x400];
        prom[..4].copy_from_slice(&[0x0b, 0xf0, 0x00, 0x40]); // j 0xbfc00100
        Ip30Bus::new([256, 0, 0, 0], &prom, Ip30Options { mac: [8, 0, 0x69, 1, 2, 3], ..Default::default() })
    }

    #[test]
    fn the_reset_vector_is_the_flash_and_memory_starts_at_512m() {
        let b = bus();
        assert_eq!(b.read32(PROM_BASE as u64).data, 0x0bf0_0040);
        assert_eq!(b.write32(RAM_BASE + 0x1000, 0xdead_beef), BUS_OK);
        assert_eq!(b.read32(RAM_BASE + 0x1000).data, 0xdead_beef);
        assert_eq!(b.read32(0x1000).data, 0xdead_beef, "the low alias");
        assert_eq!(b.read8(0x1003).data, 0xef);
        assert_ne!(b.read32(0x0100_0000).status, BUS_OK, "nothing between the alias and HEART");
        assert_ne!(b.read32(RAM_BASE + (256 << 20)).status, BUS_OK, "past the DIMMs");
    }

    #[test]
    fn widgets_answer_in_their_windows() {
        let b = bus();
        let part = |a| (b.read32(a).data >> 12) & 0xffff;
        assert_eq!(part(0x1000_0004), 0x0000); // Xbow
        assert_eq!(part(0x1800_0004), 0xc001); // HEART
        assert_eq!(part(0x1f00_0004), 0xc002); // Bridge
        assert_ne!(b.read32(0x1100_0004).status, BUS_OK); // nothing on widget 1
        // The Xbow sees HEART and the Bridge; no graphics yet.
        let alive = |port: u32| b.read32((0x1000_0000 + 0x100 + (port & 7) * 0x40 + 0x14) as u64).data >> 31;
        assert_eq!((alive(8), alive(0xc), alive(0xf), alive(9)), (1, 0, 1, 0));
    }

    #[test]
    fn a_memcfg_write_moves_the_banks() {
        let b = bus();
        b.write32(RAM_BASE, 0x1234_5678);
        // The PROM's probe: bank 0 as 1 GB at 2 GB.
        b.write32((HEART_PIU_BASE + 0x20) as u64, 0x819f_0040);
        assert_ne!(b.read32(RAM_BASE).status, BUS_OK);
        assert_eq!(b.read32(RAM_BASE + (2 << 30)).data, 0x1234_5678);
        assert_eq!(b.read32(RAM_BASE + (2 << 30) + (256 << 20)).data, 0x1234_5678, "repeats every 256 MB");
    }

    #[test]
    fn heart_drives_the_cpu_pins_from_the_bridge() {
        let b = Arc::new(bus());
        let word = AtomicU64::new(0);
        b.connect_cpu_irq(&word);
        b.write64((HEART_PIU_BASE + 0x10000) as u64, 1 << 16); // IMR0: vector 16
        b.write32(0x1f00_0134 + 8 * 4, 0x10); // Bridge pin 4 -> vector 16
        b.write32(0x1f00_010c, 1 << 4); // enable pin 4
        crate::irq::InterruptController::set_line(&*b.bridge, 4, true);
        assert_eq!(word.load(std::sync::atomic::Ordering::SeqCst), 1 << 11, "level 1: hardware interrupt 1");
        b.write64((HEART_PIU_BASE + 0x10028) as u64, 1 << 16); // CLR_ISR
        assert_eq!(word.load(std::sync::atomic::Ordering::SeqCst), 0);
    }
}
