//! Octane main memory: the DIMMs that are fitted, where HEART's bank
//! descriptors (MEMCFG) put them, and the alias of their start at physical 0.
//!
//! HEART has four banks (a DIMM pair each) and a 32-bit descriptor per
//! bank: bit 31 valid, bits 21:16 the span less one and bits 8:0 the base,
//! both in 32 MiB units from 0x2000_0000. When two descriptors claim an
//! address the higher-numbered bank answers.
//!
//! The fitted banks live packed, in bank order, in one ppmem object (byte
//! `o` at `host + (o ^ 3)`, the layout every iris RAM keeps), reached
//! through that object's own mapping by every master with no lock, as the
//! Indy's banks are. The 4 GiB ppmem window the CPU's cache and the JIT
//! read is laid out from the descriptors, 32 MiB at a time; memory a
//! descriptor puts above 4 GiB is reachable through the bus only.
//!
//! The first 16 KiB of SDRAM are the processor's physical 0..0x4000, and it
//! reaches them only there: its access to 0x2000_0000..+16 KiB is a bus
//! error, and SDRAM is reachable at its own address from 0x2000_4000. The
//! stock IDE diagnostics check exactly this. [`LowAlias`] is the low window,
//! the same bytes and the same jitv2 page generations. Whether a bus master
//! reaches 0x2000_0000..+16 KiB is not known; the `dma_view` does, as the
//! same bytes. Public software uses physical 0x400 (the GDA) and 0x600
//! (MPCONF).

use std::cell::UnsafeCell;
use std::sync::atomic::{fence, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::ppmem::{MappedMemory, PpMemSpace, PpMemory, WINDOW_SIZE};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Resettable, BUS_ERR, BUS_OK};

/// Where SDRAM space starts. Descriptors count from here in [`UNIT`]s.
pub const RAM_BASE: u64 = 0x2000_0000;
/// The descriptors' unit, and the window's mapping granule.
pub const UNIT: u64 = 32 << 20;
pub const BANKS: usize = 4;
/// Low memory at physical 0: the first `LOW_SIZE` bytes of SDRAM, which the
/// processor reaches only there.
pub const LOW_SIZE: u64 = 0x4000;
const GUARD: std::ops::Range<u64> = RAM_BASE..RAM_BASE + LOW_SIZE;
/// The page jitv2 tracks code by.
const PAGE: usize = 4096;

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
    /// A byte offset into the fitted memory.
    Ram(usize),
    /// A valid descriptor for a bank with nothing fitted: from the
    /// processor, reads are zero and writes go nowhere.
    Absent,
    /// No descriptor claims it: a bus error.
    None,
}

// Host-word layout shared by `HeartRam` and its views: what `PpMemory`
// keeps, so an aligned word is a host u32 and a doubleword is two of them
// swapped (`rotate_left(32)`). Offsets are aligned to the access.
mod layout {
    #[inline(always)]
    pub unsafe fn ld8(h: *mut u8, o: usize) -> u8 {
        *h.add(o ^ 3)
    }
    #[inline(always)]
    pub unsafe fn ld16(h: *mut u8, o: usize) -> u16 {
        *(h as *const u16).add((o >> 1) ^ 1)
    }
    #[inline(always)]
    pub unsafe fn ld32(h: *mut u8, o: usize) -> u32 {
        *(h as *const u32).add(o >> 2)
    }
    #[inline(always)]
    pub unsafe fn ld64(h: *mut u8, o: usize) -> u64 {
        (*(h as *const u64).add(o >> 3)).rotate_left(32)
    }
    #[inline(always)]
    pub unsafe fn st8(h: *mut u8, o: usize, v: u8) {
        *h.add(o ^ 3) = v
    }
    #[inline(always)]
    pub unsafe fn st16(h: *mut u8, o: usize, v: u16) {
        *(h as *mut u16).add((o >> 1) ^ 1) = v
    }
    #[inline(always)]
    pub unsafe fn st32(h: *mut u8, o: usize, v: u32) {
        *(h as *mut u32).add(o >> 2) = v
    }
    #[inline(always)]
    pub unsafe fn st64(h: *mut u8, o: usize, v: u64) {
        *(h as *mut u64).add(o >> 3) = v.rotate_left(32)
    }
    #[inline(always)]
    pub unsafe fn st64_masked(h: *mut u8, o: usize, v: u64, m: u64) {
        let p = (h as *mut u64).add(o >> 3);
        let (v, m) = (v.rotate_left(32), m.rotate_left(32));
        *p = (*p & !m) | (v & m);
    }
}
use layout::*;

/// The four descriptors, published whole: a seqlock, so an access on any
/// thread decodes against one MEMCFG or the next, never half of each.
/// Writers are serialised by `HeartRam::remap`.
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

/// The fitted DIMMs where HEART's descriptors put them: the device
/// `Ip30Physical` maps over SDRAM space, and the processor's view of it.
///
/// No lock on an access. Every master reaches the same host bytes, as on
/// the Indy: aligned 32- and 64-bit host accesses are atomic, and the
/// descriptors an access decodes against are published whole by
/// `set_memcfg`. Addresses are full physical addresses (up to 40 bits).
/// A bank a descriptor marks valid with no DIMMs reads zero and drops
/// writes; an address no descriptor claims, or the first 16 KiB of SDRAM
/// space, is a bus error.
///
/// Fast paths, as `Physical`'s banks give them: `mem_ptr`, `read_block` and
/// `write_block` go straight to the host bytes; `gen_ptr` returns the 4 KiB
/// page's generation counter, bumped by every write path, DMA included.
pub struct HeartRam {
    store: PpMemory,
    host: *mut u8,
    space: PpMemSpace,
    /// Each bank's capacity in bytes and where it starts in the store.
    size: [u64; BANKS],
    start: [u64; BANKS],
    desc: Placement,
    /// Held across a descriptor change: the window may be laid out by one
    /// caller at a time.
    remap: Mutex<()>,
}

// `host` points into `store`, which lives as long as this does. Every master
// reads and writes the bytes through it with no lock, as the hardware's own
// masters race; the window (`space`) is changed only under `remap`.
unsafe impl Send for HeartRam {}
unsafe impl Sync for HeartRam {}

impl HeartRam {
    /// `banks_mb` is the DIMM pair in each of the four banks, in MB (0 for
    /// an empty bank), packed back to back in the store. No descriptor is
    /// valid until HEART writes one. Panics on a size `try_new` refuses.
    pub fn new(banks_mb: [u32; 4]) -> Self {
        Self::try_new(banks_mb).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Each bank 0 or a power of two from 64 MiB to 2 GiB, 2 GiB in all.
    pub fn try_new(banks_mb: [u32; 4]) -> Result<Self, String> {
        let mut size = [0u64; BANKS];
        let mut start = [0u64; BANKS];
        let mut total = 0u64;
        for (i, &mb) in banks_mb.iter().enumerate() {
            let s = (mb as u64) << 20;
            if s != 0 && (!(64 << 20..=2 << 30).contains(&s) || !s.is_power_of_two()) {
                return Err(format!("HEART bank {i} capacity must be 0, 64M, 128M, 256M, 512M, 1G or 2G"));
            }
            size[i] = s;
            start[i] = total;
            total += s;
        }
        if total > 2 << 30 {
            return Err(format!("HEART banks hold {} MiB; an Octane takes at most 2048", total >> 20));
        }
        let store = PpMemory::try_new(((total >> 20) as usize).max(1)).map_err(|e| format!("ppmem: {e}"))?;
        let space = PpMemSpace::over(std::slice::from_ref(&store)).map_err(|e| format!("ppmem: {e}"))?;
        let host = store.host_ptr();
        Ok(Self { store, host, space, size, start, desc: Placement::default(), remap: Mutex::new(()) })
    }

    /// HEART's descriptors changed: decode against them from now on and lay
    /// the ppmem window out again. Called on the CPU thread (a MEMCFG write)
    /// or with the CPU stopped (reset, the machine publishing the window).
    pub fn set_memcfg(&self, words: [u32; 4]) {
        let _g = self.remap.lock();
        self.desc.store(words);
        self.space.clear_mappings();
        // Lower banks first, so higher ones overlay them as in the decode.
        for (bank, &d) in words.iter().enumerate() {
            if d & DESC_VALID == 0 || self.size[bank] == 0 {
                continue;
            }
            let base = desc_base(d);
            for unit in 0..desc_span(d) / UNIT {
                let at = base + unit * UNIT;
                if at + UNIT > WINDOW_SIZE {
                    break;
                }
                let src = self.start[bank] + bank_offset(d, self.size[bank], unit * UNIT);
                if let Err(e) = self.space.map_range(0, src, at, UNIT) {
                    eprintln!("ip30: HEART bank {bank} unit {unit}: window map failed: {e}");
                }
            }
        }
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

    /// The ppmem window the CPU and the JIT reach memory through.
    pub fn space(&self) -> &PpMemSpace {
        &self.space
    }

    /// Memory as an XIO bus master reaches it through HEART: what the
    /// Bridge (and its per-slot `PciDma`) and IMPACT's HQ4 are given.
    /// Fitted memory only, at 64-bit physical addresses: a bank with no
    /// DIMMs, low memory, HEART's registers and the widget windows are all
    /// bus errors here. `read_block` and `write_block` move nothing unless
    /// every doubleword is fitted memory; use `dma_read_bytes` and
    /// `dma_write_bytes` for byte transfers.
    pub fn dma_view(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(DmaView(self.clone()))
    }

    /// The highest-numbered bank whose descriptor claims `addr` answers.
    pub fn decode(&self, addr: u64) -> Decode {
        Self::decode_in(&self.desc.load(), &self.size, &self.start, addr)
    }

    #[inline]
    fn decode_in(d: &[u32; BANKS], size: &[u64; BANKS], start: &[u64; BANKS], addr: u64) -> Decode {
        for bank in (0..BANKS).rev() {
            let w = d[bank];
            if w & DESC_VALID == 0 {
                continue;
            }
            let base = desc_base(w);
            if addr >= base && addr - base < desc_span(w) {
                if size[bank] == 0 {
                    return Decode::Absent;
                }
                return Decode::Ram((start[bank] + bank_offset(w, size[bank], addr - base)) as usize);
            }
        }
        Decode::None
    }

    /// Where the processor's access at `addr` lands: `Some(Some(o))` memory,
    /// `Some(None)` an empty bank, `None` a bus error.
    #[inline]
    fn cpu_at(&self, addr: u64) -> Option<Option<usize>> {
        if GUARD.contains(&addr) {
            return None;
        }
        self.unguarded_at(addr)
    }

    /// `cpu_at` without the guard: how low memory reaches those bytes.
    #[inline]
    fn unguarded_at(&self, addr: u64) -> Option<Option<usize>> {
        match self.decode(addr) {
            Decode::Ram(o) => Some(Some(o)),
            Decode::Absent => Some(None),
            Decode::None => None,
        }
    }

    #[inline]
    fn dma_at(&self, addr: u64) -> Option<usize> {
        match self.decode(addr) {
            Decode::Ram(o) => Some(o),
            _ => None,
        }
    }

    /// Byte `o` of the store was written: bump its page's generation.
    #[inline(always)]
    fn touch(&self, o: usize) {
        #[cfg(feature = "jitv2")]
        self.store.bump_gen_off(o);
        #[cfg(not(feature = "jitv2"))]
        let _ = o;
    }

    /// The doublewords `addr..addr + 8 * n`, decoded a 32 MiB unit at a time
    /// (within a unit every byte follows the first): calls `f(offset, first,
    /// count)` for each run, `offset` None for an empty bank. False, before
    /// any call, if any of it is not reachable (`dma`: not fitted memory).
    fn runs(&self, addr: u64, n: usize, dma: bool, mut f: impl FnMut(Option<usize>, usize, usize)) -> bool {
        let d = self.desc.load();
        let end = addr.wrapping_add(8 * n as u64);
        let walk = |f: &mut dyn FnMut(Option<usize>, usize, usize)| -> bool {
            let (mut a, mut i) = (addr, 0usize);
            while i < n {
                // The guard ends inside a unit; a run never straddles it.
                let unit_end = (a / UNIT + 1) * UNIT;
                let stop = if !dma && a < GUARD.end && GUARD.end < unit_end { GUARD.end } else { unit_end };
                let words = (((stop.min(end).wrapping_sub(a)) / 8) as usize).clamp(1, n - i);
                if !dma && GUARD.contains(&a) {
                    return false;
                }
                match Self::decode_in(&d, &self.size, &self.start, a) {
                    Decode::Ram(o) => f(Some(o), i, words),
                    Decode::Absent if !dma => f(None, i, words),
                    _ => return false,
                }
                i += words;
                a += 8 * words as u64;
            }
            true
        };
        // All or nothing: check every run before moving anything.
        walk(&mut |_, _, _| {}) && walk(&mut f)
    }
}

/// Width accesses for one view: `$at` decodes an address to
/// `Option<Option<usize>>` (bus error, empty bank, memory).
macro_rules! ram_widths {
    ($at:expr) => {
        fn read8(&self, addr: u64) -> BusRead8 {
            match $at(self, addr) {
                Some(Some(o)) => BusRead8::ok(unsafe { ld8(self.ram().host, o) }),
                Some(None) => BusRead8::ok(0),
                None => BusRead8::err(),
            }
        }
        fn read16(&self, addr: u64) -> BusRead16 {
            match $at(self, addr) {
                Some(Some(o)) => BusRead16::ok(unsafe { ld16(self.ram().host, o) }),
                Some(None) => BusRead16::ok(0),
                None => BusRead16::err(),
            }
        }
        fn read32(&self, addr: u64) -> BusRead32 {
            match $at(self, addr) {
                Some(Some(o)) => BusRead32::ok(unsafe { ld32(self.ram().host, o) }),
                Some(None) => BusRead32::ok(0),
                None => BusRead32::err(),
            }
        }
        fn read64(&self, addr: u64) -> BusRead64 {
            match $at(self, addr) {
                Some(Some(o)) => BusRead64::ok(unsafe { ld64(self.ram().host, o) }),
                Some(None) => BusRead64::ok(0),
                None => BusRead64::err(),
            }
        }
        fn write8(&self, addr: u64, val: u8) -> u32 {
            match $at(self, addr) {
                Some(Some(o)) => {
                    unsafe { st8(self.ram().host, o, val) };
                    self.ram().touch(o);
                    BUS_OK
                }
                Some(None) => BUS_OK,
                None => BUS_ERR,
            }
        }
        fn write16(&self, addr: u64, val: u16) -> u32 {
            match $at(self, addr) {
                Some(Some(o)) => {
                    unsafe { st16(self.ram().host, o, val) };
                    self.ram().touch(o);
                    BUS_OK
                }
                Some(None) => BUS_OK,
                None => BUS_ERR,
            }
        }
        fn write32(&self, addr: u64, val: u32) -> u32 {
            match $at(self, addr) {
                Some(Some(o)) => {
                    unsafe { st32(self.ram().host, o, val) };
                    self.ram().touch(o);
                    BUS_OK
                }
                Some(None) => BUS_OK,
                None => BUS_ERR,
            }
        }
        fn write64(&self, addr: u64, val: u64) -> u32 {
            match $at(self, addr) {
                Some(Some(o)) => {
                    unsafe { st64(self.ram().host, o, val) };
                    self.ram().touch(o);
                    BUS_OK
                }
                Some(None) => BUS_OK,
                None => BUS_ERR,
            }
        }
        fn write64_masked(&self, addr: u64, val: u64, mask: u64) -> u32 {
            match $at(self, addr) {
                Some(Some(o)) => {
                    unsafe { st64_masked(self.ram().host, o, val, mask) };
                    self.ram().touch(o);
                    BUS_OK
                }
                Some(None) => BUS_OK,
                None => BUS_ERR,
            }
        }
    };
}

impl HeartRam {
    #[inline(always)]
    fn ram(&self) -> &HeartRam {
        self
    }

    fn block_read(&self, addr: u64, buf: &mut [u64], dma: bool) -> u32 {
        let host = self.host as *const u64;
        let ok = self.runs(addr, buf.len(), dma, |o, first, count| {
            let out = &mut buf[first..first + count];
            match o {
                // Same layout as `PpMemory::read_block`.
                Some(o) => unsafe {
                    let p = host.add(o >> 3);
                    for (j, slot) in out.iter_mut().enumerate() {
                        *slot = (*p.add(j)).rotate_left(32);
                    }
                },
                None => out.fill(0),
            }
        });
        if ok {
            BUS_OK
        } else {
            BUS_ERR
        }
    }

    fn block_write(&self, addr: u64, buf: &[u64], dma: bool) -> u32 {
        let host = self.host as *mut u64;
        let ok = self.runs(addr, buf.len(), dma, |o, first, count| {
            let Some(o) = o else { return };
            unsafe {
                let p = host.add(o >> 3);
                for (j, &v) in buf[first..first + count].iter().enumerate() {
                    *p.add(j) = v.rotate_left(32);
                }
            }
            // One bump per page touched.
            let mut page = usize::MAX;
            for j in 0..count {
                let b = o + 8 * j;
                if b / PAGE != page {
                    page = b / PAGE;
                    self.touch(b);
                }
            }
        });
        if ok {
            BUS_OK
        } else {
            BUS_ERR
        }
    }
}

/// The processor's view.
impl BusDevice for HeartRam {
    ram_widths!(HeartRam::cpu_at);

    fn mem_ptr(&self, addr: u64) -> Option<*const u64> {
        match self.cpu_at(addr) {
            Some(Some(o)) => Some(unsafe { self.host.add(o & !7) } as *const u64),
            _ => None,
        }
    }

    fn read_block(&self, addr: u64, buf: &mut [u64]) -> u32 {
        self.block_read(addr, buf, false)
    }

    fn write_block(&self, addr: u64, buf: &[u64]) -> u32 {
        self.block_write(addr, buf, false)
    }

    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, addr: u64) -> *const AtomicU64 {
        match self.cpu_at(addr) {
            Some(Some(o)) => self.store.gen_ptr(o as u64),
            _ => std::ptr::null(),
        }
    }
}

/// `HeartRam::dma_view`: fitted memory only.
struct DmaView(Arc<HeartRam>);

impl DmaView {
    #[inline(always)]
    fn ram(&self) -> &HeartRam {
        &self.0
    }

    #[inline(always)]
    fn at(&self, addr: u64) -> Option<Option<usize>> {
        self.0.dma_at(addr).map(Some)
    }
}

impl BusDevice for DmaView {
    ram_widths!(DmaView::at);

    fn mem_ptr(&self, addr: u64) -> Option<*const u64> {
        self.0.dma_at(addr).map(|o| unsafe { self.0.host.add(o & !7) } as *const u64)
    }

    fn read_block(&self, addr: u64, buf: &mut [u64]) -> u32 {
        self.0.block_read(addr, buf, true)
    }

    fn write_block(&self, addr: u64, buf: &[u64]) -> u32 {
        self.0.block_write(addr, buf, true)
    }

    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, addr: u64) -> *const AtomicU64 {
        self.0.dma_at(addr).map_or(std::ptr::null(), |o| self.0.store.gen_ptr(o as u64))
    }
}

/// The first 16 KiB of the processor's physical space: the same bytes as
/// SDRAM at `RAM_BASE`, with the same jitv2 page generations, so code the
/// kernel rewrites at its exception vectors is recompiled. This is the only
/// way the processor reaches them, so it takes `HeartRam`'s unguarded path.
/// It is mapped over the first 64 KiB, so it answers 0x4000..0x10000 with
/// a bus error itself, and where no bank descriptor covers `RAM_BASE` it
/// answers as SDRAM does there.
pub struct LowAlias(Arc<HeartRam>);

impl LowAlias {
    pub fn new(ram: Arc<HeartRam>) -> Self {
        Self(ram)
    }

    #[inline(always)]
    fn ram(&self) -> &HeartRam {
        &self.0
    }

    /// Where `addr` lands, as `cpu_at`: SDRAM's first 16 KiB, past the guard.
    #[inline(always)]
    fn at(&self, addr: u64) -> Option<Option<usize>> {
        let o = addr & 0xffff;
        if o >= LOW_SIZE {
            return None;
        }
        self.0.unguarded_at(RAM_BASE + o)
    }
}

impl BusDevice for LowAlias {
    ram_widths!(LowAlias::at);

    fn mem_ptr(&self, addr: u64) -> Option<*const u64> {
        match self.at(addr) {
            Some(Some(o)) => Some(unsafe { self.0.host.add(o & !7) } as *const u64),
            _ => None,
        }
    }

    /// The same counter SDRAM's page has: without it a page reached through
    /// the alias reads a fixed generation, and the JIT keeps running code
    /// compiled before the kernel rewrote it.
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, addr: u64) -> *const AtomicU64 {
        match self.at(addr) {
            Some(Some(o)) => self.0.store.gen_ptr(o as u64),
            _ => std::ptr::null(),
        }
    }
}

/// `buf.len()` bytes from physical `addr` of `mem` (a `dma_view`), for a
/// master that moves bytes. All or nothing: false if any byte is not
/// memory, and then `buf` is zero.
pub fn dma_read_bytes(mem: &dyn BusDevice, addr: u64, buf: &mut [u8]) -> bool {
    if buf.is_empty() {
        return true;
    }
    let Some((first, words)) = covering(addr, buf.len()) else {
        buf.fill(0);
        return false;
    };
    let mut w = vec![0u64; words];
    if mem.read_block(first, &mut w) != BUS_OK {
        buf.fill(0);
        return false;
    }
    let skip = (addr - first) as usize;
    for (i, b) in buf.iter_mut().enumerate() {
        let k = skip + i;
        *b = w[k / 8].to_be_bytes()[k % 8];
    }
    true
}

/// `data` to physical `addr` of `mem` (a `dma_view`). All or nothing:
/// false if any byte is not memory, and then nothing is written. Partial
/// doublewords at the ends go as masked writes, so the bytes around them
/// are left alone.
pub fn dma_write_bytes(mem: &dyn BusDevice, addr: u64, data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }
    let Some((first, words)) = covering(addr, data.len()) else { return false };
    // The check: a read of the whole span fails exactly when a write would.
    let mut probe = vec![0u64; words];
    if mem.read_block(first, &mut probe) != BUS_OK {
        return false;
    }
    let mut vals = vec![0u64; words];
    let mut masks = vec![0u64; words];
    let skip = (addr - first) as usize;
    for (i, &b) in data.iter().enumerate() {
        let k = skip + i;
        let sh = 56 - 8 * (k % 8);
        vals[k / 8] |= (b as u64) << sh;
        masks[k / 8] |= 0xff << sh;
    }
    let mut i = 0;
    while i < words {
        let at = first + 8 * i as u64;
        if masks[i] != u64::MAX {
            if mem.write64_masked(at, vals[i], masks[i]) != BUS_OK {
                return false;
            }
            i += 1;
            continue;
        }
        let j = (i..words).find(|&j| masks[j] != u64::MAX).unwrap_or(words);
        if mem.write_block(at, &vals[i..j]) != BUS_OK {
            return false;
        }
        i = j;
    }
    true
}

/// The aligned doubleword span covering `len` bytes at `addr`: its first
/// address and how many doublewords. `None` if it would wrap.
fn covering(addr: u64, len: usize) -> Option<(u64, usize)> {
    let end = addr.checked_add(len as u64)?;
    let first = addr & !7;
    let last = end.checked_add(7)? & !7;
    Some((first, ((last - first) / 8) as usize))
}

#[cfg(test)]
mod dma_tests {
    use super::*;
    use crate::traits::BUS_ERR;

    /// 64 bytes of memory at 0x1000, big-endian, 8- and 32-bit only.
    struct Bytes(parking_lot::Mutex<[u8; 64]>);

    impl Bytes {
        fn at(&self, a: u64) -> Option<usize> {
            (0x1000..0x1040).contains(&a).then(|| (a - 0x1000) as usize)
        }
    }

    impl BusDevice for Bytes {
        fn read8(&self, a: u64) -> BusRead8 {
            self.at(a).map_or(BusRead8::err(), |o| BusRead8::ok(self.0.lock()[o]))
        }
        fn write8(&self, a: u64, v: u8) -> u32 {
            self.at(a).map_or(BUS_ERR, |o| { self.0.lock()[o] = v; BUS_OK })
        }
        fn read32(&self, a: u64) -> BusRead32 {
            match self.at(a) {
                Some(o) if o + 4 <= 64 => {
                    let m = self.0.lock();
                    BusRead32::ok(u32::from_be_bytes(m[o..o + 4].try_into().unwrap()))
                }
                _ => BusRead32::err(),
            }
        }
        fn write32(&self, a: u64, v: u32) -> u32 {
            match self.at(a) {
                Some(o) if o + 4 <= 64 => {
                    self.0.lock()[o..o + 4].copy_from_slice(&v.to_be_bytes());
                    BUS_OK
                }
                _ => BUS_ERR,
            }
        }
    }

    fn mem() -> Bytes {
        Bytes(parking_lot::Mutex::new(std::array::from_fn(|i| i as u8)))
    }

    #[test]
    fn unaligned_bytes_read_and_write_in_place() {
        let m = mem();
        let mut buf = [0u8; 11];
        assert!(dma_read_bytes(&m, 0x1003, &mut buf));
        assert_eq!(buf, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
        assert!(dma_write_bytes(&m, 0x1005, &[0xaa; 20]));
        let got = *m.0.lock();
        assert_eq!(&got[..5], &[0, 1, 2, 3, 4]);
        assert_eq!(&got[5..25], &[0xaa; 20]);
        assert_eq!(&got[25..27], &[25, 26], "the bytes after are left alone");
    }

    #[test]
    fn a_transfer_off_the_end_of_memory_moves_nothing() {
        let m = mem();
        let mut buf = [0x55u8; 8];
        assert!(!dma_read_bytes(&m, 0x103c, &mut buf));
        assert_eq!(buf, [0; 8]);
        assert!(!dma_write_bytes(&m, 0x103c, &[0xee; 8]));
        assert_eq!(m.0.lock()[0x3c..], [0x3c, 0x3d, 0x3e, 0x3f]);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    const PROBE_DESC: u32 = 0x81bf_0040;
    const MODULO_DESC: u32 = 0x817f_0040;
    const PROBE_RAM: u64 = RAM_BASE + 0x40 * UNIT;

    fn ram(mib: u64) -> HeartRam {
        HeartRam::new([mib as u32, 0, 0, 0])
    }

    fn bank0(m: &HeartRam, d: u32) {
        m.set_memcfg([d, 0, 0, 0]);
    }

    fn w(m: &HeartRam, addr: u64, v: u64) {
        assert_eq!(m.write64(addr, v), BUS_OK, "write {addr:#x}");
    }

    fn r(m: &HeartRam, addr: u64) -> u64 {
        let v = m.read64(addr);
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
            assert_eq!(m.write64(PROBE_RAM + 0x100, u64::MAX), BUS_OK);
            assert_eq!(r(&m, PROBE_RAM + 0x100), 0);
            m.set_memcfg([0; 4]);
            assert_eq!(m.read64(PROBE_RAM).status, BUS_ERR);
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
            let m = Arc::new(ram(mib));
            let dma = m.dma_view();
            let final_desc = DESC_VALID | (((mib * MIB / UNIT) as u32 - 1) << 16);
            for off in [0u64, 8, 0x100, UNIT, 2 * UNIT, 4 * UNIT, 8 * UNIT] {
                bank0(&m, PROBE_DESC);
                w(&m, PROBE_RAM + off, 0x98a9_bacb_dced_fe0f);
                // DMA reaches the same bytes through the same descriptor.
                let mut b = [0u8; 8];
                assert!(dma_read_bytes(&*dma, PROBE_RAM + off, &mut b));
                assert_eq!(u64::from_be_bytes(b), 0x98a9_bacb_dced_fe0f);
                assert!(dma_write_bytes(&*dma, PROBE_RAM + off, &0x1021_3243_5465_7687u64.to_be_bytes()));
                bank0(&m, final_desc);
                // Read back through DMA: the first projected offsets fall in
                // the 16 KiB the processor cannot reach.
                let mut b = [0u8; 8];
                assert!(dma_read_bytes(&*dma, RAM_BASE + projected(mib, off), &mut b));
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
        let bits = m.space().mapped_bitmap();
        assert_eq!(bits, 0b11 << (RAM_BASE >> 26), "128 MiB from 0x20000000 is two 64 MiB regions");
        let p = unsafe { m.space().window_base().add(RAM_BASE as usize + 0x8000) } as *const u32;
        assert_eq!(unsafe { p.read_unaligned() }, 0x0123_4567, "the window holds host words");
        m.set_memcfg([0; 4]);
        assert_eq!(m.space().mapped_bitmap(), 0);
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
        m.set_memcfg([both, 0, 0, 0]);
        assert_eq!(r(&m, RAM_BASE + 0x8000), 1);
    }

    #[test]
    fn the_fast_paths_see_what_the_width_accesses_see() {
        let m = ram(64);
        bank0(&m, DESC_VALID | (1 << 16));
        let a = RAM_BASE + UNIT - 16;
        w(&m, a, 0x0011_2233_4455_6677);
        w(&m, a + 8, 0x8899_aabb_ccdd_eeff);
        let p = m.mem_ptr(a).expect("memory has a pointer");
        assert_eq!(unsafe { *p }.rotate_left(32), 0x0011_2233_4455_6677, "read64's layout");
        assert_eq!(m.read32(a + 4).data, 0x4455_6677);
        assert_eq!(m.read16(a + 2).data, 0x2233);
        assert_eq!(m.read8(a + 1).data, 0x11);
        // A block across a unit boundary: the second unit is the bank's
        // second 32 MiB.
        assert_eq!(m.write_block(a + 16, &[1, 2]), BUS_OK);
        let mut buf = [0u64; 4];
        assert_eq!(m.read_block(a, &mut buf), BUS_OK);
        assert_eq!(buf, [0x0011_2233_4455_6677, 0x8899_aabb_ccdd_eeff, 1, 2]);
        assert_eq!(m.write64_masked(a, 0xffff_0000_0000_ffff, 0xff00_0000_0000_00ff), BUS_OK);
        assert_eq!(r(&m, a), 0xff11_2233_4455_66ff);
        // Past the end: nothing moves.
        let end = RAM_BASE + (64 << 20);
        assert_eq!(m.write_block(end - 8, &[9, 9]), BUS_ERR);
        assert_eq!(r(&m, end - 8), 0, "an all-or-nothing block wrote nothing");
    }

    #[test]
    fn an_empty_bank_reads_zero_from_the_processor_but_faults_dma() {
        let m = Arc::new(HeartRam::new([64, 0, 0, 0]));
        m.set_memcfg([0, PROBE_DESC, 0, 0]);
        let mut buf = [5u64; 2];
        assert_eq!(m.read_block(PROBE_RAM, &mut buf), BUS_OK);
        assert_eq!(buf, [0, 0]);
        assert_eq!(m.write_block(PROBE_RAM, &[1]), BUS_OK);
        let dma = m.dma_view();
        assert_eq!(dma.read64(PROBE_RAM).status, BUS_ERR);
        assert_eq!(dma.read_block(PROBE_RAM, &mut buf), BUS_ERR);
        assert!(!dma_write_bytes(&*dma, PROBE_RAM, &[1; 3]));
    }

    #[test]
    fn dma_needs_every_byte_fitted() {
        let m = Arc::new(ram(64));
        bank0(&m, DESC_VALID | (1 << 16));
        let dma = m.dma_view();
        let end = RAM_BASE + 64 * MIB;
        assert!(dma_write_bytes(&*dma, end - 8, &[1; 8]));
        assert!(!dma_write_bytes(&*dma, end - 4, &[2; 8]));
        let mut b = [9u8; 8];
        assert!(!dma_read_bytes(&*dma, end - 4, &mut b));
        assert_eq!(b, [0; 8]);
        assert!(dma_read_bytes(&*dma, end - 8, &mut b));
        assert_eq!(b, [1; 8]);
        // Across a unit boundary, DMA and the CPU agree.
        assert!(dma_write_bytes(&*dma, RAM_BASE + UNIT - 4, &[1, 2, 3, 4, 5, 6, 7, 8]));
        assert_eq!(r(&m, RAM_BASE + UNIT - 8) & 0xffff_ffff, 0x0102_0304);
        assert_eq!(r(&m, RAM_BASE + UNIT) >> 32, 0x0506_0708);
        assert_eq!(dma.read8(RAM_BASE - 1).status, BUS_ERR, "below SDRAM space");
    }

    #[test]
    fn invalid_bank_sizes_are_refused() {
        assert!(HeartRam::try_new([96, 0, 0, 0]).is_err());
        assert!(HeartRam::try_new([32, 0, 0, 0]).is_err());
        assert!(HeartRam::try_new([2048, 64, 0, 0]).is_err());
    }

    #[test]
    fn low_memory_aliases_the_start_of_sdram() {
        let m = Arc::new(ram(128));
        bank0(&m, DESC_VALID | (3 << 16));
        let l = LowAlias::new(m.clone());
        // The processor reaches the bytes only through `l`; a bus master's
        // view shows what it did.
        let dma = m.dma_view();
        assert_eq!(l.read64(0).data, 0);
        assert_eq!(l.write32(0x100, 0x1234_5678), BUS_OK);
        assert_eq!(dma.read32(RAM_BASE + 0x100).data, 0x1234_5678, "a low store is in SDRAM");
        assert_eq!(m.read32(RAM_BASE + 0x100).status, BUS_ERR, "and not visible to the processor there");
        assert_eq!(l.read16(0x102).data, 0x5678);
        assert_eq!(l.read8(0x100).data, 0x12);
        assert_eq!(dma.write64(RAM_BASE + 0x2000, 0xa5a5_a5a5_5a5a_5a5a), BUS_OK);
        assert_eq!(l.read64(0x2000).data, 0xa5a5_a5a5_5a5a_5a5a, "and the reverse");
        assert_eq!(l.write8(0x3fff, 7), BUS_OK);
        assert_eq!(dma.read8(RAM_BASE + 0x3fff).data, 7);
        assert_eq!(l.write64_masked(0x2000, 0, 0xffff_ffff_0000_0000), BUS_OK);
        assert_eq!(dma.read64(RAM_BASE + 0x2000).data, 0x5a5a_5a5a);
        assert_eq!(l.read8(0x4000).status, BUS_ERR);
        assert_eq!(l.write32(0xfffc, 0), BUS_ERR, "the rest of the slot");
        let mut line = [0u64; 2];
        assert_eq!(l.read_block(0x2000, &mut line), BUS_OK);
        assert_eq!(line[0], 0x5a5a_5a5a);
        assert_eq!(l.mem_ptr(0x100), dma.mem_ptr(RAM_BASE + 0x100), "one word");
        assert!(l.mem_ptr(0x4000).is_none());
    }

    #[test]
    fn sdram_starts_past_the_guard() {
        let m = Arc::new(ram(128));
        bank0(&m, DESC_VALID | (3 << 16));
        assert_eq!(m.read64(RAM_BASE).status, BUS_ERR);
        assert_eq!(m.read8(RAM_BASE + 0x3fff).status, BUS_ERR);
        assert_eq!(m.write32(RAM_BASE + 0x10, 1), BUS_ERR);
        assert!(m.mem_ptr(RAM_BASE).is_none());
        let mut line = [0u64; 16];
        assert_eq!(m.read_block(RAM_BASE + 0x3f80, &mut line), BUS_ERR);
        assert_eq!(m.read_block(RAM_BASE + 0x4000, &mut line), BUS_OK);
        w(&m, RAM_BASE + 0x4000, 0x0102_0304_0506_0708);
        assert_eq!(m.read32(RAM_BASE + 0x4004).data, 0x0506_0708);
        assert_eq!(m.read64(RAM_BASE + (128 << 20)).status, BUS_ERR, "past the bank");
        // A bus master reaches the guard's bytes, and low memory is the
        // processor's way to them.
        assert_eq!(m.dma_view().write64(RAM_BASE, 7), BUS_OK);
        assert_eq!(m.dma_view().read64(RAM_BASE).data, 7);
        assert_eq!(LowAlias::new(m.clone()).read64(0).data, 7);
    }

    /// Every write path advances the page's generation, so code the JIT
    /// compiled from it is dropped; reads do not.
    #[cfg(feature = "jitv2")]
    #[test]
    fn stores_advance_the_page_generation() {
        let gen = |p: *const AtomicU64| unsafe { (*p).load(Ordering::Acquire) };
        let m = Arc::new(ram(64));
        bank0(&m, DESC_VALID | (1 << 16));
        let a = RAM_BASE + 0x10_0000;
        let (g0, g1) = (m.gen_ptr(a), m.gen_ptr(a + 0x1000));
        assert!(!g0.is_null() && g0 != g1);
        assert_eq!(m.gen_ptr(a + 0xff8), g0);
        let v = gen(g0);
        let _ = m.read64(a);
        assert_eq!(gen(g0), v, "a read leaves it");
        w(&m, a, 1);
        assert_ne!(gen(g0), v, "a store advances it");
        let (v0, v1) = (gen(g0), gen(g1));
        assert_eq!(m.write_block(a + 0xff8, &[1, 2]), BUS_OK);
        assert!(gen(g0) != v0 && gen(g1) != v1, "a block across pages advances both");
        let v = gen(g1);
        assert!(dma_write_bytes(&*m.dma_view(), a + 0x1003, &[1]));
        assert_ne!(gen(g1), v, "DMA too");
        // The window maps the store's counters: the CPU's fast path sees
        // the bus's stores.
        let win = unsafe { m.space().gen_window_base().add((a >> 12) as usize) } as *const AtomicU64;
        assert_eq!(gen(win), gen(g0));
        w(&m, a + 8, 2);
        assert_eq!(gen(win), gen(g0));
        // Low memory shares SDRAM's counters.
        let l = LowAlias::new(m.clone());
        assert!(m.gen_ptr(RAM_BASE).is_null(), "the guard has none");
        let dma = m.dma_view();
        assert_eq!(l.gen_ptr(0x180), dma.gen_ptr(RAM_BASE + 0x180));
        assert_ne!(l.gen_ptr(0x180), l.gen_ptr(0x1000));
        let (l0, a0) = (l.gen_ptr(0x180), gen(l.gen_ptr(0x180)));
        assert_eq!(l.write64(0x180, 0xdf5a_c000_0340_0008), BUS_OK);
        assert_ne!(gen(l0), a0, "a low-memory store advances the page");
        let v = gen(l0);
        assert_eq!(dma.write64(RAM_BASE + 0x180, 1), BUS_OK);
        assert_ne!(gen(l0), v, "and so does a bus master's at its SDRAM address");
        assert!(l.gen_ptr(0x4000).is_null());
    }
}
