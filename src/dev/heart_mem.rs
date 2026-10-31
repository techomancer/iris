//! Octane main memory: the DIMMs that are fitted, where HEART's bank
//! descriptors (MEMCFG) put them, and the alias of their start at physical 0.
//!
//! HEART has four banks (a DIMM pair each) and a 32-bit descriptor per
//! bank: bit 31 valid, bits 21:16 the span less one and bits 8:0 the base,
//! both in 32 MiB units from 0x2000_0000. When two descriptors claim an
//! address the higher-numbered bank answers.
//!
//! Each fitted bank is one of the machine's `Physical` RAM banks, and HEART
//! is the memory controller that places them, as CRIME is on the O2: a
//! descriptor write points every 32 MiB unit it claims at the bank run that
//! answers there (`Physical::bank_view`), so the CPU and every bus master
//! reach memory through the decode table with no lock, and the ppmem window
//! the CPU's cache and the JIT read is laid out from the same placement.
//! Memory that runs past 4 GiB gets a 16 GiB window; a descriptor placing
//! memory beyond the window leaves it reachable through the bus only.
//!
//! The first 16 KiB of SDRAM are the processor's physical 0..0x4000, and it
//! reaches them only there: its access to 0x2000_0000..+16 KiB is a bus
//! error, and SDRAM is reachable at its own address from 0x2000_4000. The
//! stock IDE diagnostics check exactly this. Low memory is a view of the
//! same bank, so it has the same bytes and the same jitv2 page generations.
//! Whether a bus master reaches 0x2000_0000..+16 KiB is not known; the
//! `dma_view` does, as the same bytes. Public software uses physical 0x400
//! (the GDA) and 0x600 (MPCONF).
//!
//! The ppmem window maps whole 32 MiB units, the guard included: the CPU's
//! cached accesses through the window reach those 16 KiB. Only uncached
//! accesses, which go through the table, see the bus error.
//!
//! Ported from hw/misc/sgi-heart.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{fence, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::physical::{Physical, RamBank};
use crate::ppmem::devmap::{entry_eq, Entry, SLOT_SHIFT};
use crate::ppmem::{MappedMemory, LARGE_WINDOW_SIZE, WINDOW_SIZE};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_ERR, BUS_OK};

/// Where SDRAM space starts. Descriptors count from here in [`UNIT`]s.
pub const RAM_BASE: u64 = 0x2000_0000;
/// The descriptors' unit, and the window's mapping granule.
pub const UNIT: u64 = 32 << 20;
pub const BANKS: usize = 4;
/// Low memory at physical 0: the first `LOW_SIZE` bytes of SDRAM, which the
/// processor reaches only there.
pub const LOW_SIZE: u64 = 0x4000;
const GUARD: std::ops::Range<u64> = RAM_BASE..RAM_BASE + LOW_SIZE;
/// Where SDRAM space ends: the first large widget window. A bus master's
/// address outside `RAM_BASE..SDRAM_END` is not memory.
pub const SDRAM_END: u64 = 8 << 36;
/// What the processor reaches at physical 0: low memory, then bus errors to
/// the end of the first 64 KiB.
const LOW_WINDOW: u64 = 0x1_0000;

const DESC_VALID: u32 = 0x8000_0000;

/// Where a byte offset into a fitted bank lands for an address `offset` into
/// a descriptor's span.
///
/// The stock PROM sizes a bank by writing through a temporary 2 GiB density-6
/// descriptor (`0x81bf0040`) and checking which address bits alias. Real
/// HEART's address wiring for that probe is not documented; this projection
/// is an approximation that makes each fitted size give the PROM the answer
/// it needs to pick the right descriptor. Every other descriptor wraps at
/// the bank's size. All of them keep the low 25 bits, so a 32 MiB unit maps
/// to one 32 MiB run of the bank.
pub fn bank_offset(descriptor: u32, capacity: u64, offset: u64) -> u64 {
    const MIB: u64 = 1 << 20;
    if descriptor & 0x803f_0000 == 0x803f_0000 && (descriptor >> 22) & 7 == 6 {
        match capacity {
            c if c == 64 * MIB => return (offset & (32 * MIB - 1)) | ((offset & (64 * MIB)) >> 1),
            c if c == 128 * MIB => {
                return (offset & (32 * MIB - 1)) | ((offset & (64 * MIB)) >> 1) | ((offset & (1024 * MIB)) >> 4)
            }
            c if c == 256 * MIB => return (offset & (128 * MIB - 1)) | ((offset & (256 * MIB)) >> 1),
            c if c == 512 * MIB => {
                return (offset & (128 * MIB - 1)) | ((offset & (256 * MIB)) >> 1) | ((offset & (1024 * MIB)) >> 2)
            }
            _ => {}
        }
    }
    offset % capacity
}

fn desc_base(d: u32) -> u64 {
    RAM_BASE + (d & 0x1ff) as u64 * UNIT
}

fn desc_span(d: u32) -> u64 {
    (((d >> 16) & 0x3f) as u64 + 1) * UNIT
}

/// Where a physical address in SDRAM space lands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decode {
    /// Byte `offset` of fitted bank `bank`.
    Ram { bank: usize, offset: u64 },
    /// A valid descriptor for a bank with nothing fitted: from the
    /// processor, reads are zero and writes go nowhere.
    Absent,
    /// No descriptor claims it: a bus error.
    None,
}

/// The four descriptors, published whole: a seqlock, so an access on any
/// thread decodes against one MEMCFG or the next, never half of each.
/// Writers are serialised by `HeartRam::remap`. Kept for the monitor and
/// for `decode`; accesses go through the decode table.
#[derive(Default)]
struct Placement {
    seq: AtomicU64,
    words: [AtomicU32; BANKS],
}

impl Placement {
    #[inline]
    fn load(&self) -> [u32; BANKS] {
        loop {
            let s = self.seq.load(Ordering::Acquire);
            if s & 1 == 0 {
                let w = std::array::from_fn(|i| self.words[i].load(Ordering::Relaxed));
                fence(Ordering::Acquire);
                if self.seq.load(Ordering::Relaxed) == s {
                    return w;
                }
            }
            std::hint::spin_loop();
        }
    }

    fn store(&self, w: [u32; BANKS]) {
        let s = self.seq.load(Ordering::Relaxed);
        self.seq.store(s + 1, Ordering::Relaxed);
        fence(Ordering::Release);
        for (a, v) in self.words.iter().zip(w) {
            a.store(v, Ordering::Relaxed);
        }
        self.seq.store(s + 2, Ordering::Release);
    }
}

/// What nothing claims answers until the machine sets its own default: a
/// bus error.
struct NoMemory;

impl BusDevice for NoMemory {
    fn read8(&self, _: u64) -> BusRead8 { BusRead8::err() }
    fn read16(&self, _: u64) -> BusRead16 { BusRead16::err() }
    fn read32(&self, _: u64) -> BusRead32 { BusRead32::err() }
    fn read64(&self, _: u64) -> BusRead64 { BusRead64::err() }
    fn write8(&self, _: u64, _: u8) -> u32 { BUS_ERR }
    fn write16(&self, _: u64, _: u16) -> u32 { BUS_ERR }
    fn write32(&self, _: u64, _: u32) -> u32 { BUS_ERR }
    fn write64(&self, _: u64, _: u64) -> u32 { BUS_ERR }
}

/// HEART's memory: the banks fitted, their descriptors, and the placement
/// they make on the machine's physical bus.
///
/// Placed per 32 MiB unit (every descriptor keeps the low 25 bits, so a unit
/// is one run of one bank): the unit's slots point at a view of the bank
/// counted from where that run starts. A bank a descriptor marks valid with
/// no DIMMs reads zero and drops writes; an address no descriptor claims,
/// or the first 16 KiB of SDRAM space, gets the bus's default.
pub struct HeartRam {
    phys: Arc<Physical>,
    /// Each bank's capacity in bytes (0 for nothing fitted) and its index
    /// among the bus's RAM banks.
    size: [u64; BANKS],
    index: [Option<usize>; BANKS],
    desc: Placement,
    /// Units placed by the last remap, to put back to the default.
    placed: Mutex<Vec<u64>>,
}

impl HeartRam {
    /// `banks_mb` is the DIMM pair in each of the four banks, in MB (0 for
    /// an empty bank). No descriptor is valid until HEART writes one.
    /// Panics on a size `try_new` refuses.
    pub fn new(banks_mb: [u32; 4]) -> Self {
        Self::try_new(banks_mb).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Each bank 0 or a power of two from 64 MiB to 2 GiB.
    /// Builds the machine's physical bus over the fitted banks, with a bus
    /// error as its default until the machine sets its own.
    pub fn try_new(banks_mb: [u32; 4]) -> Result<Self, String> {
        let mut size = [0u64; BANKS];
        let mut index = [None; BANKS];
        let mut banks = Vec::new();
        for (i, &mb) in banks_mb.iter().enumerate() {
            let s = (mb as u64) << 20;
            if s != 0 && (!(64 << 20..=2 << 30).contains(&s) || !s.is_power_of_two()) {
                return Err(format!("HEART bank {i} capacity must be 0, 64M, 128M, 256M, 512M, 1G or 2G"));
            }
            if s != 0 {
                let bank = RamBank::try_new(mb as usize).map_err(|e| format!("ppmem: {e}"))?;
                index[i] = Some(banks.len());
                banks.push(bank);
            }
            size[i] = s;
        }
        // Back to back from RAM_BASE is where the PROM leaves them.
        let window = if RAM_BASE + size.iter().sum::<u64>() > WINDOW_SIZE { LARGE_WINDOW_SIZE } else { WINDOW_SIZE };
        let phys = Physical::with_window(banks, window);
        phys.set_default(Arc::new(NoMemory), "bus error");
        let m = Self { phys, size, index, desc: Placement::default(), placed: Mutex::new(Vec::new()) };
        m.rebuild();
        Ok(m)
    }

    /// The bus the banks are on.
    pub fn phys(&self) -> &Arc<Physical> {
        &self.phys
    }

    /// HEART's descriptors changed: place the banks as they now say.
    /// Called on the CPU thread (a MEMCFG write, which the PROM makes before
    /// anything else runs) or with the CPU stopped.
    ///
    /// A write that leaves the decode as it was does nothing to the window
    /// or the page generations: the PROM rewrites the same descriptors, and
    /// a rebuild empties the window under a running CPU.
    pub fn set_memcfg(&self, words: [u32; 4]) {
        let same = Self::decoded(&self.desc.load()) == Self::decoded(&words);
        self.desc.store(words);
        if !same {
            self.rebuild();
        }
    }

    /// The bits of each descriptor the decode reads: all of a valid one
    /// (valid, density, span, base), nothing of an invalid one.
    fn decoded(words: &[u32; BANKS]) -> [u32; BANKS] {
        words.map(|d| if d & DESC_VALID == 0 { 0 } else { d & (DESC_VALID | 0x7 << 22 | 0x3f << 16 | 0x1ff) })
    }

    /// Place the banks as the descriptors in force say: after `set_memcfg`,
    /// and after the machine sets the bus's default device.
    pub fn rebuild(&self) {
        let mut placed = self.placed.lock();
        let phys = &*self.phys;
        let words = self.desc.load();
        let default = phys.default_entry();
        let sp = phys.ppmem_space();
        if let Some(sp) = sp {
            sp.clear_mappings();
        }
        // A bank now answering where another did must look changed to the
        // JIT even if the two pages' counters happen to be equal.
        #[cfg(feature = "jitv2")]
        for b in phys.banks() {
            b.bump_gen_all();
        }
        // Unit address -> (entry, the bank run for the window). Lower banks
        // first, so a higher one overrides them as in the decode.
        let mut units: std::collections::BTreeMap<u64, (Entry, Option<(usize, u64)>)> = Default::default();
        for (bank, &d) in words.iter().enumerate() {
            if d & DESC_VALID == 0 {
                continue;
            }
            let base = desc_base(d);
            for unit in 0..desc_span(d) / UNIT {
                let at = base + unit * UNIT;
                let v = match self.index[bank] {
                    Some(i) => {
                        let src = bank_offset(d, self.size[bank], unit * UNIT);
                        (phys.bank_view(i, at - src), Some((i, src)))
                    }
                    None => (phys.unmapped_entry(), None),
                };
                units.insert(at, v);
            }
        }
        for &at in placed.iter() {
            if !units.contains_key(&at) {
                phys.fill_slots(at >> SLOT_SHIFT, UNIT >> SLOT_SHIFT, default);
            }
        }
        placed.clear();
        for (&at, &(entry, run)) in &units {
            phys.fill_slots(at >> SLOT_SHIFT, UNIT >> SLOT_SHIFT, entry);
            placed.push(at);
            if let (Some(sp), Some((i, src))) = (sp, run) {
                if at + UNIT <= sp.window_size() {
                    if let Err(e) = sp.map_range(i, src, at, UNIT) {
                        eprintln!("ip30: HEART unit at {at:#x}: window map failed: {e}");
                    }
                }
            }
        }
        // The processor's low memory is the start of SDRAM space; SDRAM's
        // own first 16 KiB are a bus error to it.
        let low = match units.get(&RAM_BASE) {
            Some(&(_, Some((i, src)))) => phys.bank_view(i, src.wrapping_neg()),
            Some(&(entry, None)) => entry,
            None => default,
        };
        phys.fill_slots(0, LOW_SIZE >> SLOT_SHIFT, low);
        phys.fill_slots(LOW_SIZE >> SLOT_SHIFT, (LOW_WINDOW - LOW_SIZE) >> SLOT_SHIFT, default);
        phys.fill_slots(GUARD.start >> SLOT_SHIFT, LOW_SIZE >> SLOT_SHIFT, default);
    }

    /// The descriptors in force, for the monitor.
    pub fn memcfg(&self) -> [u32; 4] {
        self.desc.load()
    }

    /// Each bank's capacity in bytes.
    pub fn bank_sizes(&self) -> [u64; BANKS] {
        self.size
    }

    /// Bytes fitted.
    pub fn total(&self) -> usize {
        self.size.iter().sum::<u64>() as usize
    }

    /// The valid descriptors as placed: (physical base, span, capacity), a
    /// capacity of 0 for a bank with nothing fitted.
    pub fn layout(&self) -> Vec<(u64, u64, u64)> {
        let d = self.desc.load();
        (0..BANKS)
            .filter(|&b| d[b] & DESC_VALID != 0)
            .map(|b| (desc_base(d[b]), desc_span(d[b]), self.size[b]))
            .collect()
    }

    /// Memory as an XIO bus master reaches it through HEART: what the
    /// Bridge (and its per-slot `PciDma`) and IMPACT's HQ4 are given.
    /// Fitted memory only, at its physical addresses, the 16 KiB the
    /// processor cannot reach at SDRAM's start included: a bank with no
    /// DIMMs, low memory, HEART's registers and the widget windows are all
    /// bus errors here. An access stays within one 4 KiB page, as
    /// `physical::read_bytes` and `write_bytes` make them.
    pub fn dma_view(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(DmaView(self.clone()))
    }

    /// The highest-numbered bank whose descriptor claims `addr` answers.
    pub fn decode(&self, addr: u64) -> Decode {
        let d = self.desc.load();
        for bank in (0..BANKS).rev() {
            let w = d[bank];
            if w & DESC_VALID == 0 {
                continue;
            }
            let base = desc_base(w);
            if addr >= base && addr - base < desc_span(w) {
                if self.size[bank] == 0 {
                    return Decode::Absent;
                }
                return Decode::Ram { bank, offset: bank_offset(w, self.size[bank], addr - base) };
            }
        }
        Decode::None
    }
}

/// Memory as a bus master sees it: [`HeartRam::dma_view`].
struct DmaView(Arc<HeartRam>);

impl DmaView {
    /// The memory behind `addr`, and the address to give it: the guard's
    /// bytes through low memory, which is a view of the same bank.
    #[inline]
    fn at(&self, addr: u64) -> Option<(&dyn BusDevice, u64)> {
        if !(RAM_BASE..SDRAM_END).contains(&addr) {
            return None;
        }
        let a = if GUARD.contains(&addr) { addr - RAM_BASE } else { addr };
        let phys = &self.0.phys;
        let e = phys.entry_at(a);
        if entry_eq(e, phys.unmapped_entry()) || entry_eq(e, phys.default_entry()) {
            return None;
        }
        // SAFETY: a table entry points at a device the bus keeps alive.
        Some((unsafe { &*e }, a))
    }
}

macro_rules! dma_rw {
    ($read:ident, $write:ident, $r:ident, $t:ty) => {
        fn $read(&self, addr: u64) -> $r {
            self.at(addr).map_or($r::err(), |(d, a)| d.$read(a))
        }
        fn $write(&self, addr: u64, v: $t) -> u32 {
            self.at(addr).map_or(BUS_ERR, |(d, a)| d.$write(a, v))
        }
    };
}

impl BusDevice for DmaView {
    dma_rw!(read8, write8, BusRead8, u8);
    dma_rw!(read16, write16, BusRead16, u16);
    dma_rw!(read32, write32, BusRead32, u32);
    dma_rw!(read64, write64, BusRead64, u64);
    fn write64_masked(&self, addr: u64, v: u64, m: u64) -> u32 {
        self.at(addr).map_or(BUS_ERR, |(d, a)| d.write64_masked(a, v, m))
    }
    fn mem_ptr(&self, addr: u64) -> Option<*const u64> {
        self.at(addr).and_then(|(d, a)| d.mem_ptr(a))
    }
    fn read_block(&self, addr: u64, buf: &mut [u64]) -> u32 {
        self.at(addr).map_or(BUS_ERR, |(d, a)| d.read_block(a, buf))
    }
    fn write_block(&self, addr: u64, buf: &[u64]) -> u32 {
        self.at(addr).map_or(BUS_ERR, |(d, a)| d.write_block(a, buf))
    }
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, addr: u64) -> *const AtomicU64 {
        self.at(addr).map_or(std::ptr::null(), |(d, a)| d.gen_ptr(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical::{read_bytes, write_bytes};

    const MIB: u64 = 1 << 20;
    const PROBE_DESC: u32 = 0x81bf_0040;
    const MODULO_DESC: u32 = 0x817f_0040;
    const PROBE_RAM: u64 = RAM_BASE + 0x40 * UNIT;

    fn ram(mib: u64) -> Arc<HeartRam> {
        Arc::new(HeartRam::new([mib as u32, 0, 0, 0]))
    }

    fn bank0(m: &HeartRam, d: u32) {
        m.set_memcfg([d, 0, 0, 0]);
    }

    fn w(m: &HeartRam, addr: u64, v: u64) {
        assert_eq!(m.phys().write64(addr, v), BUS_OK, "write {addr:#x}");
    }

    fn r(m: &HeartRam, addr: u64) -> u64 {
        let v = m.phys().read64(addr);
        assert_eq!(v.status, BUS_OK, "read {addr:#x}");
        v.data
    }

    #[test]
    fn descriptors_find_the_fitted_size() {
        for mib in [64u64, 128, 256] {
            let m = ram(mib);
            assert_eq!(m.memcfg(), [0; 4], "nothing valid at power-on");
            bank0(&m, MODULO_DESC);
            w(&m, PROBE_RAM + 0x100, 0x1234_5678_9abc_def0);
            let units = (1..32u64).find(|u| r(&m, PROBE_RAM + 0x100 + u * UNIT) == 0x1234_5678_9abc_def0);
            assert_eq!(units, Some(mib / 32), "{mib} MiB");
            // An absent bank's descriptor still claims probe space but keeps
            // nothing.
            m.set_memcfg([0, PROBE_DESC, 0, 0]);
            assert_eq!(m.phys().write64(PROBE_RAM + 0x100, u64::MAX), BUS_OK);
            assert_eq!(r(&m, PROBE_RAM + 0x100), 0);
            m.set_memcfg([0; 4]);
            assert_eq!(m.phys().read64(PROBE_RAM).status, BUS_ERR);
        }
    }

    fn projected(mib: u64, offset: u64) -> u64 {
        match mib {
            64 => (offset & (UNIT - 1)) | ((offset & (2 * UNIT)) >> 1),
            128 => (offset & (UNIT - 1)) | ((offset & (2 * UNIT)) >> 1) | ((offset & (1 << 30)) >> 4),
            256 => (offset & (4 * UNIT - 1)) | ((offset & (8 * UNIT)) >> 1),
            _ => unreachable!(),
        }
    }

    #[test]
    fn the_probe_descriptor_projects_onto_the_bank() {
        for mib in [64u64, 128, 256] {
            let m = ram(mib);
            let dma = m.dma_view();
            let final_desc = DESC_VALID | (((mib * MIB / UNIT) as u32 - 1) << 16);
            for off in [0u64, 8, 0x100, UNIT, 2 * UNIT, 4 * UNIT, 8 * UNIT] {
                bank0(&m, PROBE_DESC);
                w(&m, PROBE_RAM + off, 0x98a9_bacb_dced_fe0f);
                // DMA reaches the same bytes through the same descriptor.
                let mut b = [0u8; 8];
                assert_eq!(read_bytes(&*dma, PROBE_RAM + off, &mut b), BUS_OK);
                assert_eq!(u64::from_be_bytes(b), 0x98a9_bacb_dced_fe0f);
                assert_eq!(write_bytes(&*dma, PROBE_RAM + off, &0x1021_3243_5465_7687u64.to_be_bytes()), BUS_OK);
                bank0(&m, final_desc);
                // Read back through DMA: the first projected offsets fall in
                // the 16 KiB the processor cannot reach.
                let mut b = [0u8; 8];
                assert_eq!(read_bytes(&*dma, RAM_BASE + projected(mib, off), &mut b), BUS_OK);
                assert_eq!(u64::from_be_bytes(b), 0x1021_3243_5465_7687, "{mib} {off:#x}");
            }
        }
    }

    /// The stock PROM's bank sizing, replayed: every size reaches the
    /// descriptor the PROM's table names for it.
    #[test]
    fn the_prom_probe_reaches_the_right_descriptor() {
        const TABLE: [u32; 16] = [
            0x0000_0040, 0x0001_0040, 0x0043_0040, 0x0047_0040, 0x008f_0040, 0x009f_0040, 0x0000_0040, 0x0000_0040,
            0x0101_0040, 0x0103_0040, 0x0147_0040, 0x014f_0040, 0x019f_0040, 0x01bf_0040, 0x0000_0040, 0x0000_0040,
        ];
        fn distinct(m: &HeartRam, base: u64, step: u64, pattern: u64) -> bool {
            w(m, base, pattern);
            w(m, base + step, !pattern);
            r(m, base) == pattern
        }
        fn capacity(m: &HeartRam, base: u64) -> u32 {
            let pattern = 0x1111_1111_1111_1111u64;
            let mut bits = 0;
            w(m, base + 0x100, pattern);
            w(m, base, !pattern);
            if r(m, base + 0x100) == pattern {
                bits |= 0x20;
            }
            if distinct(m, base, 4 * UNIT, pattern << 1) {
                bits |= 0x10;
            } else if distinct(m, base, UNIT, pattern << 2) {
                bits |= 0x08;
            }
            if distinct(m, base, 32 * UNIT, pattern << 3) {
                bits |= 0x04;
            }
            TABLE[bits / 4]
        }
        // Memory addresses here run past 4 GiB: the probe descriptor's base
        // is 0x40 units up.
        let probe = PROBE_RAM;
        for (mib, want) in [
            (64u64, 0x8101_0040u32),
            (128, 0x8103_0040),
            (256, 0x8147_0040),
            (512, 0x814f_0040),
            (1024, 0x819f_0040),
            (2048, 0x81bf_0040),
        ] {
            let m = ram(mib);
            bank0(&m, PROBE_DESC);
            let word = capacity(&m, probe) | DESC_VALID;
            bank0(&m, word);
            if (word >> 22) & 1 != 0 {
                w(&m, probe + 0x80, 0);
                w(&m, probe + UNIT + 0x80, !0);
                assert_ne!(r(&m, probe + 0x80), r(&m, probe + UNIT + 0x80));
            }
            assert_eq!(word, want, "{mib} MiB");
            for unit in 0..mib / 32 {
                w(&m, probe + unit * UNIT + 0x100, 0x1021_3243_5465_7687 + unit);
            }
            for unit in 0..mib / 32 {
                assert_eq!(r(&m, probe + unit * UNIT + 0x100), 0x1021_3243_5465_7687 + unit);
            }
        }
    }

    #[test]
    fn the_window_follows_the_descriptors() {
        let m = ram(128);
        bank0(&m, DESC_VALID | (3 << 16));
        w(&m, RAM_BASE + 0x8000, 0x0123_4567_89ab_cdef);
        let sp = m.phys().ppmem_space().expect("a window");
        assert_eq!(sp.mapped_bitmap(), 0b11 << (RAM_BASE >> 26), "128 MiB from 0x20000000 is two 64 MiB regions");
        let p = unsafe { sp.window_base().add(RAM_BASE as usize + 0x8000) } as *const u32;
        assert_eq!(unsafe { p.read_unaligned() }, 0x0123_4567, "the window holds host words");
        m.set_memcfg([0; 4]);
        assert_eq!(sp.mapped_bitmap(), 0);
        assert_eq!(m.phys().read64(RAM_BASE + 0x8000).status, BUS_ERR, "and the table forgets the units");
    }

    #[test]
    fn a_write_that_leaves_the_decode_alone_does_not_touch_the_window() {
        let m = ram(128);
        let d = DESC_VALID | (3 << 16);
        bank0(&m, d);
        let space = m.phys().ppmem_space().unwrap();
        let mapped = space.mapped_bitmap();
        assert_ne!(mapped, 0);
        // Drop the mappings behind its back: a rebuild would put them back.
        space.clear_mappings();
        bank0(&m, d);
        assert_eq!(space.mapped_bitmap(), 0, "the same descriptors: no rebuild");
        // Bits the decode ignores, and the rest of an invalid descriptor.
        m.set_memcfg([d | 1 << 29, 0x7fff_ffff, 0, 0]);
        assert_eq!(space.mapped_bitmap(), 0, "ignored bits: no rebuild");
        assert_eq!(m.memcfg()[1], 0x7fff_ffff, "but the monitor sees what was written");
        bank0(&m, DESC_VALID | (1 << 16));
        assert_eq!(space.mapped_bitmap(), 1 << (RAM_BASE >> 26), "a change lays it out");
        assert_ne!(mapped, space.mapped_bitmap());
    }

    /// A rebuild bumps every page's generation, so compiled code from a page
    /// that now holds other bytes is dropped; a repeated write does not.
    #[cfg(feature = "jitv2")]
    #[test]
    fn a_changed_descriptor_bumps_the_page_generations() {
        let gen = |p: *const AtomicU64| unsafe { (*p).load(Ordering::Acquire) };
        let m = ram(128);
        let d = DESC_VALID | (3 << 16);
        bank0(&m, d);
        let g = m.phys().gen_ptr(RAM_BASE + 0x8000);
        let v = gen(g);
        bank0(&m, d);
        assert_eq!(gen(g), v, "the same descriptors leave it");
        bank0(&m, DESC_VALID | (1 << 16));
        assert_ne!(gen(g), v, "a change advances it");
        let v = gen(g);
        m.set_memcfg([0; 4]);
        assert_ne!(gen(g), v, "so does taking the memory away");
    }

    #[test]
    fn the_higher_bank_answers_where_two_overlap() {
        let m = HeartRam::new([64, 64, 0, 0]);
        let both = DESC_VALID | (1 << 16);
        m.set_memcfg([both, 0, 0, 0]);
        w(&m, RAM_BASE + 0x8000, 1);
        m.set_memcfg([both, both, 0, 0]);
        w(&m, RAM_BASE + 0x8000, 2);
        assert_eq!(r(&m, RAM_BASE + 0x8000), 2);
        assert_eq!(m.decode(RAM_BASE + 0x8000), Decode::Ram { bank: 1, offset: 0x8000 });
        m.set_memcfg([both, 0, 0, 0]);
        assert_eq!(r(&m, RAM_BASE + 0x8000), 1);
    }

    #[test]
    fn a_unit_is_one_run_of_its_bank() {
        let m = ram(64);
        bank0(&m, DESC_VALID | (1 << 16));
        let a = RAM_BASE + UNIT - 16;
        w(&m, a, 0x0011_2233_4455_6677);
        w(&m, a + 8, 0x8899_aabb_ccdd_eeff);
        w(&m, a + 16, 1);
        let bank = &m.phys().banks()[0];
        assert_eq!(bank.read64(UNIT - 16).data, 0x0011_2233_4455_6677);
        assert_eq!(bank.read64(UNIT).data, 1, "the second unit is the bank's second 32 MiB");
        assert_eq!(m.phys().read32(a + 4).data, 0x4455_6677);
        assert_eq!(m.phys().read16(a + 2).data, 0x2233);
        assert_eq!(m.phys().read8(a + 1).data, 0x11);
        let mut buf = [0u64; 2];
        assert_eq!(m.phys().read_block(a, &mut buf), BUS_OK);
        assert_eq!(buf, [0x0011_2233_4455_6677, 0x8899_aabb_ccdd_eeff]);
        assert_eq!(m.phys().read64(RAM_BASE + 64 * MIB).status, BUS_ERR, "past the bank");
    }

    #[test]
    fn an_empty_bank_reads_zero_from_the_processor_but_faults_dma() {
        let m = Arc::new(HeartRam::new([64, 0, 0, 0]));
        m.set_memcfg([0, PROBE_DESC, 0, 0]);
        assert_eq!(m.phys().write64(PROBE_RAM, 1), BUS_OK);
        assert_eq!(r(&m, PROBE_RAM), 0);
        let dma = m.dma_view();
        assert_eq!(dma.read64(PROBE_RAM).status, BUS_ERR);
        let mut buf = [5u64; 2];
        assert_eq!(dma.read_block(PROBE_RAM, &mut buf), BUS_ERR);
        assert_ne!(write_bytes(&*dma, PROBE_RAM, &[1; 3]), BUS_OK);
    }

    #[test]
    fn dma_reaches_fitted_memory_only() {
        let m = ram(64);
        bank0(&m, DESC_VALID | (1 << 16));
        let dma = m.dma_view();
        let end = RAM_BASE + 64 * MIB;
        assert_eq!(write_bytes(&*dma, end - 8, &[1; 8]), BUS_OK);
        assert_ne!(write_bytes(&*dma, end - 4, &[2; 8]), BUS_OK, "off the end");
        let mut b = [9u8; 8];
        assert_ne!(read_bytes(&*dma, end - 4, &mut b), BUS_OK);
        assert_eq!(read_bytes(&*dma, end - 8, &mut b), BUS_OK);
        assert_eq!(b, [1, 1, 1, 1, 2, 2, 2, 2], "what fitted memory there was took the write");
        // Across a unit boundary, DMA and the CPU agree.
        assert_eq!(write_bytes(&*dma, RAM_BASE + UNIT - 4, &[1, 2, 3, 4, 5, 6, 7, 8]), BUS_OK);
        assert_eq!(r(&m, RAM_BASE + UNIT - 8) & 0xffff_ffff, 0x0102_0304);
        assert_eq!(r(&m, RAM_BASE + UNIT) >> 32, 0x0506_0708);
        assert_eq!(dma.read8(RAM_BASE - 1).status, BUS_ERR, "below SDRAM space");
        assert_eq!(dma.read8(0x100).status, BUS_ERR, "low memory is the processor's");
    }

    /// A store-conditional's compare-and-swap reaches memory the window does not.
    #[test]
    fn compare_and_swap_works_above_the_window() {
        let m = ram(64);
        let high = (WINDOW_SIZE - RAM_BASE) / UNIT;
        bank0(&m, DESC_VALID | (1 << 16) | high as u32);
        let at = WINDOW_SIZE + 0x8000;
        let p = m.phys();
        assert_eq!(p.write64(at, 0x0102_0304_0506_0708), BUS_OK);
        assert_eq!(p.cas64(at, 1, 2), Some(false));
        assert_eq!(p.read64(at).data, 0x0102_0304_0506_0708, "a failed swap stores nothing");
        assert_eq!(p.cas64(at, 0x0102_0304_0506_0708, 0x1112_1314_1516_1718), Some(true));
        assert_eq!(p.read64(at).data, 0x1112_1314_1516_1718);
        assert_eq!(p.cas32(at, 0x2222_2222, 3), Some(false));
        assert_eq!(p.cas32(at, 0x1112_1314, 0xa1a2_a3a4), Some(true));
        assert_eq!(p.read64(at).data, 0xa1a2_a3a4_1516_1718, "the same word a 32-bit store writes");
        assert_eq!(p.cas32(WINDOW_SIZE * 2, 0, 1), None, "not memory: the caller stores as usual");
    }

    /// All eight DIMMs: the window reaches the top of memory.
    #[cfg(feature = "jitv2")]
    #[test]
    fn eight_gigabytes_are_all_in_the_window() {
        let m = HeartRam::new([2048; 4]);
        const TWO_G: u32 = 0x3f << 16;
        m.set_memcfg([0, 0x40, 0x80, 0xc0].map(|base| DESC_VALID | TWO_G | base));
        let sp = m.phys().ppmem_space().expect("a window");
        assert_eq!(sp.window_size(), LARGE_WINDOW_SIZE);
        let top = RAM_BASE + (8u64 << 30);
        for (at, mapped) in [(RAM_BASE, true), (WINDOW_SIZE, true), (top - 8, true), (top, false), (RAM_BASE - 8, false)] {
            let bits = sp.mapped_bitmap();
            assert_eq!(crate::ppmem::bitmap_maps_at(bits, crate::ppmem::window_shift(LARGE_WINDOW_SIZE), at), mapped, "{at:#x}");
        }
        let at = top - 0x1000;
        w(&m, at, 0x0123_4567_89ab_cdef);
        let p = unsafe { sp.window_base().add(at as usize) } as *const u32;
        assert_eq!(unsafe { p.read_unaligned() }, 0x0123_4567, "the window holds the bus's words");
        let g = m.phys().gen_ptr(at);
        let before = unsafe { (*g).load(Ordering::Relaxed) };
        w(&m, at, 1);
        assert_eq!(unsafe { (*g).load(Ordering::Relaxed) }, before + 1);
        let gen = unsafe { sp.gen_window_base().add((at >> 12) as usize) };
        assert_eq!(unsafe { (*gen).load(Ordering::Relaxed) }, before + 1, "the generation window follows");
    }

    #[test]
    fn invalid_bank_sizes_are_refused() {
        assert!(HeartRam::try_new([96, 0, 0, 0]).is_err());
        assert!(HeartRam::try_new([32, 0, 0, 0]).is_err());
        assert!(HeartRam::try_new([4096, 0, 0, 0]).is_err());
    }

    #[test]
    fn low_memory_aliases_the_start_of_sdram() {
        let m = ram(128);
        bank0(&m, DESC_VALID | (3 << 16));
        let (p, dma) = (m.phys(), m.dma_view());
        assert_eq!(p.read64(0).data, 0);
        assert_eq!(p.write32(0x100, 0x1234_5678), BUS_OK);
        assert_eq!(dma.read32(RAM_BASE + 0x100).data, 0x1234_5678, "a low store is in SDRAM");
        assert_eq!(p.read32(RAM_BASE + 0x100).status, BUS_ERR, "and not visible to the processor there");
        assert_eq!(p.read16(0x102).data, 0x5678);
        assert_eq!(p.read8(0x100).data, 0x12);
        assert_eq!(dma.write64(RAM_BASE + 0x2000, 0xa5a5_a5a5_5a5a_5a5a), BUS_OK);
        assert_eq!(p.read64(0x2000).data, 0xa5a5_a5a5_5a5a_5a5a, "and the reverse");
        assert_eq!(p.write8(0x3fff, 7), BUS_OK);
        assert_eq!(dma.read8(RAM_BASE + 0x3fff).data, 7);
        assert_eq!(p.write64_masked(0x2000, 0, 0xffff_ffff_0000_0000), BUS_OK);
        assert_eq!(dma.read64(RAM_BASE + 0x2000).data, 0x5a5a_5a5a);
        assert_eq!(p.read8(0x4000).status, BUS_ERR);
        assert_eq!(p.write32(0xfffc, 0), BUS_ERR, "the rest of the first 64 KiB");
        let mut line = [0u64; 2];
        assert_eq!(p.read_block(0x2000, &mut line), BUS_OK);
        assert_eq!(line[0], 0x5a5a_5a5a);
        assert_eq!(p.mem_ptr(0x100), dma.mem_ptr(RAM_BASE + 0x100), "one word");
        assert!(p.mem_ptr(0x4000).is_none());
        // With no descriptor at SDRAM's start, low memory is a bus error.
        m.set_memcfg([0; 4]);
        assert_eq!(p.read64(0).status, BUS_ERR);
    }

    #[test]
    fn sdram_starts_past_the_guard() {
        let m = ram(128);
        bank0(&m, DESC_VALID | (3 << 16));
        let p = m.phys();
        assert_eq!(p.read64(RAM_BASE).status, BUS_ERR);
        assert_eq!(p.read8(RAM_BASE + 0x3fff).status, BUS_ERR);
        assert_eq!(p.write32(RAM_BASE + 0x10, 1), BUS_ERR);
        w(&m, RAM_BASE + 0x4000, 0x0102_0304_0506_0708);
        assert_eq!(p.read32(RAM_BASE + 0x4004).data, 0x0506_0708);
        // A bus master reaches the guard's bytes, and low memory is the
        // processor's way to them.
        assert_eq!(m.dma_view().write64(RAM_BASE, 7), BUS_OK);
        assert_eq!(m.dma_view().read64(RAM_BASE).data, 7);
        assert_eq!(p.read64(0).data, 7);
    }

    /// Every write path advances the page's generation, so code the JIT
    /// compiled from it is dropped; low memory shares SDRAM's counters.
    #[cfg(feature = "jitv2")]
    #[test]
    fn low_memory_shares_the_page_generation() {
        let gen = |p: *const AtomicU64| unsafe { (*p).load(Ordering::Acquire) };
        let m = ram(64);
        bank0(&m, DESC_VALID | (1 << 16));
        let (p, dma) = (m.phys(), m.dma_view());
        assert_eq!(p.gen_ptr(0x180), dma.gen_ptr(RAM_BASE + 0x180));
        assert_ne!(p.gen_ptr(0x180), p.gen_ptr(0x1000));
        let (l0, a0) = (p.gen_ptr(0x180), gen(p.gen_ptr(0x180)));
        assert_eq!(p.write64(0x180, 0xdf5a_c000_0340_0008), BUS_OK);
        assert_ne!(gen(l0), a0, "a low-memory store advances the page");
        let v = gen(l0);
        assert_eq!(dma.write64(RAM_BASE + 0x180, 1), BUS_OK);
        assert_ne!(gen(l0), v, "and so does a bus master's at its SDRAM address");
        assert!(p.gen_ptr(0x4000).is_null());
    }
}
