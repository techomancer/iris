//! The SGI Octane (IP30): its processor-side physical bus. HEART, the
//! Crossbow and the widgets behind it, low memory, and SDRAM where HEART's
//! bank descriptors put it. `ip30_machine` builds and drives it.
//!
//! The map, from OpenBSD's `ip30.h` and Linux's `ip30-xtalk.c` (also
//! `docs/Octane/ip30-address-map.md`):
//!
//! | Range | What |
//! |---|---|
//! | `0x0000_0000`, 16 KiB | low memory, the first 16 KiB of SDRAM (`heart_mem::LowAlias`) |
//! | `0x0ff0_0000` | HEART's processor-interface registers |
//! | `0x1000_0000 + (w << 24)` | widget `w`'s 16 MiB small window |
//! | `0x2000_0000` up | SDRAM through HEART's bank descriptors (`heart_mem::HeartRam`); its first 16 KiB bus-error here |
//! | `w << 36` | widget `w`'s 64 GiB large window (8 to 15) |
//!
//! Widget 0 is the Crossbow, 8 HEART, 15 the onboard Bridge (whose window
//! holds the boot flash at 0x1fc0_0000, the reset vector), 9 to 12 the
//! graphics ports.
//!
//! The decode is one table, built the way `Physical` is for the Indy: a
//! `SparseDeviceMap` of 4 KiB slots over the R10000's 40-bit physical
//! space, each pointing at the device that answers it.
//!
//! Set IRIS_IP30_TRACE=1 to log unmapped accesses as they happen.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::dev::heart_mem::{HeartRam, LowAlias, RAM_BASE};
use crate::ppmem::devmap::{Entry, PHYS_MASK, SLOT_SHIFT};
use crate::ppmem::SparseDeviceMap;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_ERR, BUS_OK};

/// Widget numbers on the Octane's Crossbow.
pub const WIDGET_XBOW: u8 = 0x0;
pub const WIDGET_HEART: u8 = 0x8;
pub const WIDGET_BRIDGE: u8 = 0xf;
/// The four ports graphics and expansion boards take.
pub const WIDGET_GFX_FIRST: u8 = 0x9;

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
/// Where SDRAM space ends: the first large window. Everything from
/// `RAM_BASE` to here goes to `HeartRam`, which decodes the descriptors.
pub const SDRAM_SPACE_END: u64 = widget_large_base(8);
/// The boot flash window, both chip selects, as the processor addresses it.
pub const BOOT_FLASH: std::ops::Range<u64> = 0x1fc0_0000..0x2000_0000;

/// Where the CPU was at the end of its last batch, for the unmapped log.
/// The machine's batch hook stores it.
pub static LAST_PC: AtomicU64 = AtomicU64::new(0);

/// What answers an address nothing claims: a bus error.
#[derive(Default)]
struct Unmapped;

impl BusDevice for Unmapped {
    fn read8(&self, _: u64) -> BusRead8 { BusRead8::err() }
    fn read16(&self, _: u64) -> BusRead16 { BusRead16::err() }
    fn read32(&self, _: u64) -> BusRead32 { BusRead32::err() }
    fn read64(&self, _: u64) -> BusRead64 { BusRead64::err() }
    fn write8(&self, _: u64, _: u8) -> u32 { BUS_ERR }
    fn write16(&self, _: u64, _: u16) -> u32 { BUS_ERR }
    fn write32(&self, _: u64, _: u32) -> u32 { BUS_ERR }
    fn write64(&self, _: u64, _: u64) -> u32 { BUS_ERR }
}

/// The Octane's processor-side physical bus: a `SparseDeviceMap` of 4 KiB
/// slots over the 40-bit space, each pointing at the device that answers
/// it. Every access is one mask, one table load and a call; there is no
/// lock and no decode chain.
///
/// Devices see full physical addresses (masked to 40 bits), not offsets,
/// and mask to their own window. A device that fills only part of a slot
/// answers the rest of it itself (low memory is mapped over 64 KiB and
/// bus-errors 0x4000..0x10000).
///
/// Built in three steps, as `Physical` is: `new`, then the machine
/// registers devices (`map`, `attach_widget`), then `init` fills the table
/// and from then on the CPU may run. Table changes after that
/// (`set_widget_link`) replace whole slots; readers on device threads see
/// the old slot or the new one, never a torn one (`docs/devmap-design.md`).
pub struct Ip30Physical {
    devmap: SparseDeviceMap,
    /// `devmap.base()`, cached for the hot path.
    map_base: *const Entry,
    ram: Arc<HeartRam>,
    low: Arc<LowAlias>,
    unmapped: Arc<Unmapped>,
    /// The table holds raw pointers; these keep what they point at alive.
    keep: Mutex<Vec<Arc<dyn BusDevice>>>,
    /// The widgets on the Crossbow's ports 9 to 15, and whether each also
    /// takes its large window.
    widgets: Mutex<[Option<(Arc<dyn BusDevice>, bool)>; 16]>,
    /// Accesses that failed, first 64 distinct: `(addr, write, pc)`.
    misses: Mutex<Vec<(u64, bool, u64)>>,
}

// SAFETY: the table's pointers are into devices `keep` (or the fields
// above) hold for the life of the bus; every device is Send + Sync, and the
// table itself is only changed through `SparseDeviceMap`'s own lock.
unsafe impl Send for Ip30Physical {}
unsafe impl Sync for Ip30Physical {}

impl Ip30Physical {
    /// An empty map over `ram` (SDRAM space, and physical 0 as the window onto
    /// its start).
    pub fn new(ram: Arc<HeartRam>) -> std::io::Result<Self> {
        let devmap = SparseDeviceMap::new()?;
        let map_base = devmap.base();
        let me = Self {
            devmap,
            map_base,
            low: Arc::new(LowAlias::new(ram.clone())),
            ram,
            unmapped: Arc::new(Unmapped),
            keep: Mutex::new(Vec::new()),
            widgets: Mutex::new(Default::default()),
            misses: Mutex::new(Vec::new()),
        };
        // Bus error by default, so `map` may fill part of a page before
        // `init`.
        me.devmap.set_default(me.unmapped_entry())?;
        Ok(me)
    }

    fn fill(&self, start: u64, end: u64, dev: Entry) {
        self.devmap
            .fill_bytes(start, end, dev)
            .unwrap_or_else(|e| panic!("ip30: mapping {start:#x}..{end:#x}: {e}"));
    }

    fn unmapped_entry(&self) -> Entry {
        Arc::as_ptr(&self.unmapped) as *const dyn BusDevice
    }

    /// Put `dev` at physical `start..end` (`end` exclusive, rounded up to a
    /// 4 KiB slot): HEART's PIU registers, the Crossbow's window (widget 0)
    /// and HEART's widget registers (widget 8), none of which a link gates.
    /// Takes effect at once; a later `map` over the same slots wins.
    pub fn map(&self, start: u64, end: u64, dev: Arc<dyn BusDevice>) {
        self.fill(start, end, Arc::as_ptr(&dev));
        self.keep.lock().push(dev);
    }

    /// Put `dev` on the Crossbow's port `w` (9 to 15): its small window and,
    /// with `large`, its large window. Routed only while the link is alive;
    /// until `set_widget_link(w, true)` both windows bus-error.
    pub fn attach_widget(&self, w: u8, dev: Arc<dyn BusDevice>, large: bool) {
        assert!((9..16).contains(&w), "ip30: widget {w} is not a port the Crossbow switches");
        self.keep.lock().push(dev.clone());
        self.widgets.lock()[w as usize] = Some((dev, large));
    }

    /// Fill the table: low memory, SDRAM space, and
    /// every attached widget's windows as its link stands (dead). Call once,
    /// after the devices are registered and before the CPU runs.
    pub fn init(&self) {
        self.fill(0, 0x1_0000, Arc::as_ptr(&self.low) as *const dyn BusDevice);
        self.fill(RAM_BASE, SDRAM_SPACE_END, Arc::as_ptr(&self.ram) as *const dyn BusDevice);
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
        let unmapped = self.unmapped_entry();
        let (dev, large) = match &self.widgets.lock()[w as usize] {
            Some((d, large)) if alive => (Arc::as_ptr(d), *large),
            Some((_, large)) => (unmapped, *large),
            None => (unmapped, false),
        };
        let small = widget_small_base(w);
        self.fill(small, small + WIDGET_SMALL_SIZE, dev);
        if large {
            let base = widget_large_base(w);
            self.fill(base, base + WIDGET_LARGE_SIZE, dev);
        }
    }

    /// Memory as the bus masters (the Bridge and its `PciDma`s, IMPACT's
    /// HQ4) reach it: `HeartRam::dma_view`.
    pub fn memory(&self) -> Arc<dyn BusDevice> {
        self.ram.dma_view()
    }

    /// Accesses that failed, first 64 distinct, as `(addr, is_write, pc)`.
    pub fn unmapped(&self) -> Vec<(u64, bool, u64)> {
        self.misses.lock().clone()
    }

    #[cold]
    fn miss(&self, addr: u64, write: bool) {
        let pc = LAST_PC.load(Ordering::Relaxed);
        let mut u = self.misses.lock();
        if u.len() < 64 && !u.iter().any(|&(a, w, _)| a == addr && w == write) {
            if std::env::var_os("IRIS_IP30_TRACE").is_some() {
                eprintln!("ip30: unmapped {} at {addr:#x} (pc {pc:#x})", if write { "write" } else { "read" });
            }
            u.push((addr, write, pc));
        }
    }

    /// The device behind `addr`: one indexed load.
    #[inline(always)]
    fn dev_at(&self, addr: u64) -> &dyn BusDevice {
        // SAFETY: `map_base` spans the whole table for the life of
        // `devmap`; a page nobody filled faults in as the default device,
        // and every entry points at a device this bus keeps alive.
        unsafe { &**self.map_base.add(((addr & PHYS_MASK) >> SLOT_SHIFT) as usize) }
    }
}

macro_rules! forward_rw {
    ($read:ident, $write:ident, $r:ident, $t:ty) => {
        #[inline]
        fn $read(&self, a: u64) -> $r {
            let a = a & PHYS_MASK;
            let r = self.dev_at(a).$read(a);
            if r.status == BUS_ERR {
                self.miss(a, false);
            }
            r
        }
        #[inline]
        fn $write(&self, a: u64, v: $t) -> u32 {
            let a = a & PHYS_MASK;
            let r = self.dev_at(a).$write(a, v);
            if r == BUS_ERR {
                self.miss(a, true);
            }
            r
        }
    };
}

/// Every access forwarded to the slot's device, the fast paths included.
/// A failed width access is noted for the monitor's `ip30 unmapped`.
impl BusDevice for Ip30Physical {
    forward_rw!(read8, write8, BusRead8, u8);
    forward_rw!(read16, write16, BusRead16, u16);
    forward_rw!(read32, write32, BusRead32, u32);
    forward_rw!(read64, write64, BusRead64, u64);
    fn write64_masked(&self, a: u64, v: u64, m: u64) -> u32 {
        let a = a & PHYS_MASK;
        self.dev_at(a).write64_masked(a, v, m)
    }
    fn dma_read64(&self, a: u64) -> BusRead64 {
        let a = a & PHYS_MASK;
        self.dev_at(a).dma_read64(a)
    }
    fn dma_write64(&self, a: u64, v: u64) -> u32 {
        let a = a & PHYS_MASK;
        self.dev_at(a).dma_write64(a, v)
    }
    #[inline]
    fn mem_ptr(&self, a: u64) -> Option<*const u64> {
        let a = a & PHYS_MASK;
        self.dev_at(a).mem_ptr(a)
    }
    #[inline]
    fn read_block(&self, a: u64, buf: &mut [u64]) -> u32 {
        let a = a & PHYS_MASK;
        self.dev_at(a).read_block(a, buf)
    }
    #[inline]
    fn write_block(&self, a: u64, buf: &[u64]) -> u32 {
        let a = a & PHYS_MASK;
        self.dev_at(a).write_block(a, buf)
    }
    #[cfg(feature = "jitv2")]
    #[inline]
    fn gen_ptr(&self, a: u64) -> *const AtomicU64 {
        let a = a & PHYS_MASK;
        self.dev_at(a).gen_ptr(a)
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
pub struct CachedFlashLines {
    bus: Arc<dyn BusDevice>,
    lines: Mutex<std::collections::HashMap<u64, [u8; CACHE_LINE as usize]>>,
}

impl CachedFlashLines {
    pub fn new(bus: Arc<dyn BusDevice>) -> Self {
        Self { bus, lines: Mutex::new(std::collections::HashMap::new()) }
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
        let lines = self.lines.lock();
        let line = lines.get(&(addr & !(CACHE_LINE - 1)))?;
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
                let mut words = [0u64; (CACHE_LINE / 8) as usize];
                let r = self.bus.read_block(base, &mut words);
                if r != BUS_OK {
                    return Some(r);
                }
                let mut line = [0u8; CACHE_LINE as usize];
                for (chunk, w) in line.chunks_mut(8).zip(words) {
                    chunk.copy_from_slice(&w.to_be_bytes());
                }
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
