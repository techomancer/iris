//! The SGI Octane (IP30): its processor-side physical bus. HEART, the
//! Crossbow and the widgets behind it, low memory, and SDRAM where HEART's
//! bank descriptors put it. `ip30_machine` builds and drives it.
//!
//! The map, from OpenBSD's `ip30.h` and Linux's `ip30-xtalk.c` (also
//! `docs/Octane/ip30-address-map.md`):
//!
//! | Range | What |
//! |---|---|
//! | `0x0000_0000`, 16 KiB | low memory, the first 16 KiB of SDRAM |
//! | `0x0ff0_0000` | HEART's processor-interface registers |
//! | `0x1000_0000 + (w << 24)` | widget `w`'s 16 MiB small window |
//! | `0x2000_0000` up | SDRAM through HEART's bank descriptors (`heart_mem::HeartRam`); its first 16 KiB bus-error here |
//! | `w << 36` | widget `w`'s 64 GiB large window (8 to 15) |
//!
//! Widget 0 is the Crossbow, 8 HEART, 15 the onboard Bridge (whose window
//! holds the boot flash at 0x1fc0_0000, the reset vector), 9 to 12 the
//! graphics ports.
//!
//! The decode is the machine's `Physical`, configured here the way
//! `ip22_bus` configures it for the Indy and `ip32` for the O2: one table
//! of 4 KiB slots over the R10000's 40-bit physical space, each pointing at
//! the device that answers it.
//!
//! Set IRIS_IP30_TRACE=1 to log unmapped accesses as they happen.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::dev::heart_mem::HeartRam;
use crate::pci::BarRun;
use crate::physical::Physical;
use crate::ppmem::devmap::{entry_eq, SLOT_SHIFT};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_ERR, BUS_OK};

/// Widget numbers on the Octane's Crossbow.
pub const WIDGET_XBOW: u8 = 0x0;
pub const WIDGET_HEART: u8 = 0x8;
pub const WIDGET_BRIDGE: u8 = 0xf;

/// Widget `w`'s small window: 16 MiB at `0x1000_0000 + (w << 24)`.
pub const fn widget_small_base(w: u8) -> u64 {
    0x1000_0000 | (w as u64) << 24
}
pub const WIDGET_SMALL_SIZE: u64 = 0x0100_0000;
/// Widget `w`'s large window (widgets 8 to 15): 64 GiB at `w << 36`.
pub const fn widget_large_base(w: u8) -> u64 {
    (w as u64) << 36
}
pub const WIDGET_LARGE_SIZE: u64 = 1 << 36;
/// Where SDRAM space ends: the first large window.
pub const SDRAM_SPACE_END: u64 = crate::dev::heart_mem::SDRAM_END;
const _: () = assert!(SDRAM_SPACE_END == widget_large_base(8));
/// The boot flash window, both chip selects, as the processor addresses it.
pub const BOOT_FLASH: std::ops::Range<u64> = 0x1fc0_0000..0x2000_0000;

/// Where the CPU was at the end of its last batch, for the unmapped log.
/// The machine's batch hook stores it.
pub static LAST_PC: [AtomicU64; crate::cpu::smp::MAX_CPUS] = [const { AtomicU64::new(0) }; crate::cpu::smp::MAX_CPUS];

/// What answers an address nothing claims: a bus error, noted with the PC
/// for the monitor's `ip30 unmapped`.
#[derive(Default)]
struct Unmapped {
    /// First 64 distinct: `(addr, write, pc)`.
    misses: Mutex<Vec<(u64, bool, u64)>>,
    /// `misses` holds 64: a further miss costs one relaxed load, so a guest
    /// polling an absent device does not lock and scan the list each time.
    full: AtomicBool,
}

impl Unmapped {
    #[cold]
    fn miss(&self, addr: u64, write: bool) {
        if self.full.load(Ordering::Relaxed) {
            return;
        }
        let pc = LAST_PC[crate::cpu::smp::current_cpu()].load(Ordering::Relaxed);
        let mut u = self.misses.lock();
        if !u.iter().any(|&(a, w, _)| a == addr && w == write) {
            if std::env::var_os("IRIS_IP30_TRACE").is_some() {
                eprintln!("ip30: unmapped {} at {addr:#x} (pc {pc:#x})", if write { "write" } else { "read" });
            }
            u.push((addr, write, pc));
            if u.len() == 64 {
                self.full.store(true, Ordering::Relaxed);
            }
        }
    }
}

impl BusDevice for Unmapped {
    fn read8(&self, a: u64) -> BusRead8 { self.miss(a, false); BusRead8::err() }
    fn read16(&self, a: u64) -> BusRead16 { self.miss(a, false); BusRead16::err() }
    fn read32(&self, a: u64) -> BusRead32 { self.miss(a, false); BusRead32::err() }
    fn read64(&self, a: u64) -> BusRead64 { self.miss(a, false); BusRead64::err() }
    fn write8(&self, a: u64, _: u8) -> u32 { self.miss(a, true); BUS_ERR }
    fn write16(&self, a: u64, _: u16) -> u32 { self.miss(a, true); BUS_ERR }
    fn write32(&self, a: u64, _: u32) -> u32 { self.miss(a, true); BUS_ERR }
    fn write64(&self, a: u64, _: u64) -> u32 { self.miss(a, true); BUS_ERR }
}

/// The Octane's processor-side physical bus: the machine's `Physical`,
/// configured with what HEART and the Crossbow decode, as `Ip22Bus` and
/// `Ip32Bus` configure it for the Indy and the O2. The CPU, the JIT and the
/// monitor use `phys`; bus masters use [`Self::memory`].
///
/// Devices see full physical addresses (masked to 40 bits), not offsets,
/// and mask to their own window. HEART places memory (`HeartRam`, through
/// [`Self::remap_banks`]); the Crossbow's link changes switch a widget's
/// windows between the widget and the bus error (`set_widget_link`).
/// Either replaces whole table pages; readers on device threads see the old
/// entry or the new one, never a torn one (`docs/devmap-design.md`).
pub struct Ip30Bus {
    pub phys: Arc<Physical>,
    ram: Arc<HeartRam>,
    unmapped: Arc<Unmapped>,
    /// The widgets on the Crossbow's ports 9 to 15, and whether each also
    /// takes its large window.
    widgets: Mutex<[Option<(Arc<dyn BusDevice>, bool)>; 16]>,
    /// Per port, the runs of its small window that a PCI BAR decodes whole
    /// (the Bridge's [`crate::pci::BarRun`]s, offsets in the window), mapped
    /// over the widget while its link is up.
    runs: Mutex<[Vec<BarRun>; 16]>,
}

impl Ip30Bus {
    /// The bus `ram` built over its banks, with every address a bus error
    /// until something is mapped (and memory where the descriptors in
    /// force put it).
    pub fn new(ram: Arc<HeartRam>) -> Self {
        let phys = ram.phys().clone();
        let unmapped = Arc::new(Unmapped::default());
        phys.set_default(unmapped.clone(), "unmapped (bus error)");
        ram.rebuild();
        Self { phys, ram, unmapped, widgets: Mutex::new(Default::default()), runs: Mutex::new(Default::default()) }
    }

    /// Put `dev` at physical `start..end` (`end` exclusive, rounded up to a
    /// 4 KiB slot): HEART's PIU registers, the Crossbow's window (widget 0)
    /// and HEART's widget registers (widget 8), none of which a link gates.
    /// Takes effect at once; a later `map` over the same slots wins.
    pub fn map(&self, start: u64, end: u64, dev: Arc<dyn BusDevice>, name: &str) {
        self.phys.map(start, end, dev, name);
    }

    /// Put `dev` on the Crossbow's port `w` (9 to 15): its small window and,
    /// with `large`, its large window. Routed only while the link is alive;
    /// until `set_widget_link(w, true)` both windows bus-error.
    pub fn attach_widget(&self, w: u8, dev: Arc<dyn BusDevice>, large: bool) {
        assert!((9..16).contains(&w), "ip30: widget {w} is not a port the Crossbow switches");
        self.widgets.lock()[w as usize] = Some((dev, large));
    }

    /// Fill the widget windows as each link stands (dead). Call once, after
    /// the devices are registered and before the CPU runs.
    pub fn init(&self) {
        for w in 9..16 {
            self.set_widget_link(w, false);
        }
    }

    /// HEART's bank descriptors changed: memory decodes against them from
    /// now on. HEART calls this (through the machine) from the descriptor
    /// write, on the CPU thread.
    pub fn remap_banks(&self, memcfg: [u32; 4]) {
        self.ram.set_memcfg(memcfg);
    }

    /// The Crossbow brought widget `w`'s link up or took it down: point its
    /// windows at the widget or at the bus error. The Crossbow's link-change
    /// callback lands here, outside the Crossbow's own lock. Ports other
    /// than 9 to 15 have no windows to switch.
    pub fn set_widget_link(&self, w: u8, alive: bool) {
        if !(9..16).contains(&w) {
            return;
        }
        let widgets = self.widgets.lock();
        let Some((dev, large)) = &widgets[w as usize] else {
            self.unmap(widget_small_base(w), WIDGET_SMALL_SIZE);
            return;
        };
        let mut windows = vec![(widget_small_base(w), WIDGET_SMALL_SIZE)];
        if *large {
            windows.push((widget_large_base(w), WIDGET_LARGE_SIZE));
        }
        for (base, size) in windows {
            if alive {
                self.phys.map(base, base + size, dev.clone(), &format!("widget {w:#x}"));
            } else {
                self.unmap(base, size);
            }
        }
        if alive {
            self.map_runs(w, &self.runs.lock()[w as usize]);
        }
    }

    /// Widget `w` says which runs of its small window a PCI BAR decodes
    /// whole (the Bridge's `on_bar_runs`): map them over the widget while
    /// its link is up, giving the runs it named before back to the widget.
    pub fn set_widget_runs(&self, w: u8, new: Vec<BarRun>) {
        if !(9..16).contains(&w) {
            return;
        }
        let widgets = self.widgets.lock();
        let mut runs = self.runs.lock();
        let old = std::mem::replace(&mut runs[w as usize], new);
        let Some((dev, _)) = &widgets[w as usize] else { return };
        let base = widget_small_base(w);
        // The window's first page is the widget's own registers, never a
        // run: it answers as the widget exactly while the link is up.
        if !entry_eq(self.phys.entry_at(base), Arc::as_ptr(dev)) {
            return; // link down: `set_widget_link` maps them when it comes up
        }
        for r in &old {
            self.phys.map(base + r.start, base + r.end, dev.clone(), &format!("widget {w:#x}"));
        }
        self.map_runs(w, &runs[w as usize]);
    }

    fn map_runs(&self, w: u8, runs: &[BarRun]) {
        let base = widget_small_base(w);
        for r in runs {
            self.phys.map(base + r.start, base + r.end, r.dev.clone(), &format!("widget {w:#x} pci bar"));
        }
    }

    fn unmap(&self, base: u64, size: u64) {
        self.phys.fill_slots(base >> SLOT_SHIFT, size >> SLOT_SHIFT, self.phys.default_entry());
    }

    /// Memory as the bus masters (the Bridge and its `PciDma`s, IMPACT's
    /// HQ4) reach it: `HeartRam::dma_view`.
    pub fn memory(&self) -> Arc<dyn BusDevice> {
        self.ram.dma_view()
    }

    /// Accesses nothing answered, first 64 distinct, as `(addr, is_write, pc)`.
    pub fn unmapped(&self) -> Vec<(u64, bool, u64)> {
        self.unmapped.misses.lock().clone()
    }
}

/// An R10000 secondary cache line.
const CACHE_LINE: u64 = 128;

/// The CPU's cached view of the bus: what the cache model reaches, while
/// uncached accesses go to the bus directly. Bus masters never see it.
///
/// The PROM runs cached from the boot flash and keeps its initialised data
/// there, storing to its own variables at their flash addresses. On an
/// R10000 a cached store allocates the line in the write-back cache, dirty
/// exclusive, and later cached loads hit it; the PROM never has the line
/// written back, the same pattern as firmware that keeps its working data
/// in the cache before memory exists. Nothing reaches the Bridge, whose
/// flash port refuses anything but a 16-bit write.
///
/// Approximation: iris's R10000 cache model passes every cached access to
/// memory, so the dirty lines live here instead. The first cached store to a
/// flash line copies the line in; from then on cached loads and stores of it
/// use the copy, which is never written back or evicted. Uncached accesses
/// always reach the flash. Reset empties it. What the hardware does if such
/// a line is ever written back is not modelled. Outside the flash window
/// every access, the fast paths included, goes straight to the bus.
///
/// With several CPUs each has its own, as each has its own cache (the PROM runs
/// non-coherent, `Config.K0` = 3), and cached reads hold their lines too,
/// filled from the flash's array: while one CPU has the part in status mode
/// another keeps executing from flash, and on the hardware that CPU hits its
/// instruction cache. Reading through instead hands it the status byte as an
/// instruction.
pub struct CachedFlashLines {
    bus: Arc<dyn BusDevice>,
    lines: Mutex<std::collections::HashMap<u64, [u8; CACHE_LINE as usize]>>,
    hold_reads: bool,
}

impl CachedFlashLines {
    pub fn new(bus: Arc<dyn BusDevice>) -> Self {
        Self { bus, lines: Mutex::new(std::collections::HashMap::new()), hold_reads: false }
    }

    /// One CPU of several: see the type's doc.
    pub fn for_one_of_several_cpus(bus: Arc<dyn BusDevice>) -> Self {
        Self { hold_reads: true, ..Self::new(bus) }
    }

    /// The line at `base` as the flash's array holds it.
    fn fill(&self, base: u64) -> Result<[u8; CACHE_LINE as usize], u32> {
        let mut words = [0u64; (CACHE_LINE / 8) as usize];
        let r = crate::dev::sflash::reading_the_array(|| self.bus.read_block(base, &mut words));
        if r != BUS_OK {
            return Err(r);
        }
        let mut line = [0u8; CACHE_LINE as usize];
        for (chunk, w) in line.chunks_mut(8).zip(words) {
            chunk.copy_from_slice(&w.to_be_bytes());
        }
        Ok(line)
    }

    /// Forget every line, as a reset loses the cache's contents.
    pub fn clear(&self) {
        self.lines.lock().clear();
    }

    /// Bytes `addr..addr + n` of a held line, big-endian, or `None` if the
    /// line is not held.
    fn load(&self, addr: u64, n: usize) -> Option<u64> {
        if !BOOT_FLASH.contains(&addr) {
            return None;
        }
        let base = addr & !(CACHE_LINE - 1);
        let mut lines = self.lines.lock();
        if self.hold_reads && !lines.contains_key(&base) {
            // A failed fill goes on to the bus, which reports it.
            lines.insert(base, self.fill(base).ok()?);
        }
        let line = lines.get(&base)?;
        let at = (addr & (CACHE_LINE - 1)) as usize;
        Some(line[at..at + n].iter().fold(0, |v, &b| v << 8 | b as u64))
    }

    /// Store into the line, filling it from the flash first. `None`: not a
    /// flash address.
    fn store(&self, addr: u64, n: usize, val: u64) -> Option<u32> {
        if !BOOT_FLASH.contains(&addr) {
            return None;
        }
        let base = addr & !(CACHE_LINE - 1);
        let mut lines = self.lines.lock();
        let line = match lines.entry(base) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let line = match self.fill(base) {
                    Ok(line) => line,
                    Err(r) => return Some(r),
                };
                e.insert(line)
            }
        };
        let at = (addr & (CACHE_LINE - 1)) as usize;
        line[at..at + n].copy_from_slice(&val.to_be_bytes()[8 - n..]);
        Some(BUS_OK)
    }

    /// Whether `n` doublewords at `addr` touch the flash window.
    fn in_flash(addr: u64, n: usize) -> bool {
        addr < BOOT_FLASH.end && addr.saturating_add(8 * n as u64) > BOOT_FLASH.start
    }
}

macro_rules! cached_access {
    ($read:ident, $write:ident, $r:ident, $t:ty, $n:expr) => {
        fn $read(&self, addr: u64) -> $r {
            match self.load(addr, $n) {
                Some(v) => $r::ok(v as $t),
                None => self.bus.$read(addr),
            }
        }
        fn $write(&self, addr: u64, val: $t) -> u32 {
            self.store(addr, $n, val as u64).unwrap_or_else(|| self.bus.$write(addr, val))
        }
    };
}

impl BusDevice for CachedFlashLines {
    cached_access!(read8, write8, BusRead8, u8, 1);
    cached_access!(read16, write16, BusRead16, u16, 2);
    cached_access!(read32, write32, BusRead32, u32, 4);
    cached_access!(read64, write64, BusRead64, u64, 8);

    fn describe_fault(&self, addr: u64, size: usize) -> Option<String> {
        self.bus.describe_fault(addr, size)
    }

    fn mem_ptr(&self, addr: u64) -> Option<*const u64> {
        if BOOT_FLASH.contains(&addr) {
            return None;
        }
        self.bus.mem_ptr(addr)
    }

    /// A line fill from the flash goes word by word, so held lines answer.
    fn read_block(&self, addr: u64, buf: &mut [u64]) -> u32 {
        if !Self::in_flash(addr, buf.len()) {
            return self.bus.read_block(addr, buf);
        }
        for (i, slot) in buf.iter_mut().enumerate() {
            let r = self.read64(addr + 8 * i as u64);
            if r.status != BUS_OK {
                return r.status;
            }
            *slot = r.data;
        }
        BUS_OK
    }

    fn write_block(&self, addr: u64, buf: &[u64]) -> u32 {
        if !Self::in_flash(addr, buf.len()) {
            return self.bus.write_block(addr, buf);
        }
        for (i, &v) in buf.iter().enumerate() {
            let s = self.write64(addr + 8 * i as u64, v);
            if s != BUS_OK {
                return s;
            }
        }
        BUS_OK
    }

    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, addr: u64) -> *const AtomicU64 {
        self.bus.gen_ptr(addr)
    }
}

/// Cross-device tests: the chips wired as the machine wires them, no CPU.
#[cfg(test)]
mod tests;
