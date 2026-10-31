//! The physical bus: a machine's address decode and its RAM, tied together.
//!
//! `Physical` belongs to no machine. It holds:
//!
//! - the decode table, one device per 4 KB slot of a 40-bit physical space
//!   (`SparseDeviceMap`), so a bus access is one indexed load and a call;
//! - the RAM banks (`PpMemory`, each a `BusDevice`) and ppmem's 4 GB window
//!   over them, which the CPU's cache and the JIT read and write directly, and
//!   which `mem_ptr`, `read_block`, `write_block` and `gen_ptr` here serve
//!   without a device dispatch;
//! - the plain devices every machine uses: memory that reads zero, a black
//!   hole, an error device, and aliases that re-dispatch an address range
//!   somewhere else on this bus.
//!
//! A machine builds its devices, makes a `Physical` over its banks and maps
//! everything where its hardware decodes it: `ip22_bus` for the Indy and
//! Indigo2, `ip32` for the O2. Where a memory controller places RAM, the
//! machine's remap fills the table with banks (`fill_slots`) and maps the
//! window to match (`ram_space`). DMA masters reach memory through this bus
//! as a `BusDevice`, like the CPU.

use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};
use std::io::Write as IoWrite;

use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BusDevice, Device, BUS_OK};
use crate::devlog::LogModule;
use crate::exp::eval_const_expr;
use crate::cpu::mips_dis;
use crate::dev::mem::{BlackHoleRegion, UnmappedRam};
use crate::ppmem::{MappedMemory, PpMemSpace, PpMemory};
use crate::ppmem::devmap::{entry_eq, Entry, SparseDeviceMap, PHYS_MASK, SLOT_SHIFT};

/// The RAM bank implementation: `PpMemory`, host-MMU-backed (see
/// `docs/ppmem-design.md`). It is still a `BusDevice`, so DMA and every other
/// bus-path access keep working whether or not the 4GB window was reserved.
pub type RamBank = PpMemory;

// Error device: a bus error for every access, optionally logged.
struct ErrorBus {
    debug: AtomicBool,
}

impl ErrorBus {
    fn new() -> Self {
        Self {
            debug: AtomicBool::new(false),
        }
    }

    fn set_debug(&self, val: bool) {
        self.debug.store(val, Ordering::Relaxed);
    }
}

impl BusDevice for ErrorBus {
    fn read8(&self, addr: u64) -> BusRead8 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Read8 {:08x}", addr); }
        BusRead8::err()
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Write8 {:08x} val {:02x}", addr, val); }
        crate::traits::BUS_ERR
    }
    fn read16(&self, addr: u64) -> BusRead16 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Read16 {:08x}", addr); }
        BusRead16::err()
    }
    fn write16(&self, addr: u64, val: u16) -> u32 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Write16 {:08x} val {:04x}", addr, val); }
        crate::traits::BUS_ERR
    }
    fn read32(&self, addr: u64) -> BusRead32 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Read32 {:08x}", addr); }
        BusRead32::err()
    }
    fn write32(&self, addr: u64, val: u32) -> u32 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Write32 {:08x} val {:08x}", addr, val); }
        crate::traits::BUS_ERR
    }
    fn read64(&self, addr: u64) -> BusRead64 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Read64 {:08x}", addr); }
        BusRead64::err()
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Write64 {:08x} val {:016x}", addr, val); }
        crate::traits::BUS_ERR
    }
}

// Alias device - wraps another device and translates addresses
struct AliasBus {
    target: *const dyn BusDevice,
    offset: u32,
}

unsafe impl Send for AliasBus {}
unsafe impl Sync for AliasBus {}

impl AliasBus {
    fn new(target: *const dyn BusDevice, offset: u32) -> Self {
        Self { target, offset }
    }
}

impl AliasBus {
    /// Aliases live in the 32-bit space and rely on its wrap-around (the VINO
    /// GIO alias adds 0xE1000000 to reach 0x00080000).
    #[inline(always)]
    fn target_addr(&self, addr: u64) -> u64 {
        (addr as u32).wrapping_add(self.offset) as u64
    }
}

impl BusDevice for AliasBus {
    fn read8(&self, addr: u64) -> BusRead8   { unsafe { (*self.target).read8(self.target_addr(addr)) } }
    fn write8(&self, addr: u64, val: u8) -> u32  { unsafe { (*self.target).write8(self.target_addr(addr), val) } }
    fn read16(&self, addr: u64) -> BusRead16  { unsafe { (*self.target).read16(self.target_addr(addr)) } }
    fn write16(&self, addr: u64, val: u16) -> u32 { unsafe { (*self.target).write16(self.target_addr(addr), val) } }
    fn read32(&self, addr: u64) -> BusRead32  { unsafe { (*self.target).read32(self.target_addr(addr)) } }
    fn write32(&self, addr: u64, val: u32) -> u32 { unsafe { (*self.target).write32(self.target_addr(addr), val) } }
    fn read64(&self, addr: u64) -> BusRead64  { unsafe { (*self.target).read64(self.target_addr(addr)) } }
    fn write64(&self, addr: u64, val: u64) -> u32 { unsafe { (*self.target).write64(self.target_addr(addr), val) } }
    fn describe_fault(&self, addr: u64, size: usize) -> Option<String> { unsafe { (*self.target).describe_fault(self.target_addr(addr), size) } }

    /// Forward the generation counter lookup too — an alias is the same
    /// physical memory, so it must resolve to the same counter.
    ///
    /// Without this the trait default returns null, and `PhysicalCodePage::
    /// claim` maps a null `gen_ptr` onto the shared `NEVER_COMPILABLE_GEN`
    /// (initialised to 0, never bumped). A page reached through the alias then
    /// reads `gen=0` forever: the JIT compiles it, the guest rewrites it via the
    /// real address, no invalidation is ever observed, and stale compiled code
    /// keeps running. Seen live as `j2 pcp` reporting `pfn=0 gen=0 entry_gen=0`
    /// on the TLB refill vector page after the kernel had patched it many times.
    ///
    /// This is the bus-path twin of the ppmem-window bug fixed in
    /// `PpMemSpace::map_alias` — both had to be wrong for the symptom to appear
    /// in every build, and fixing only one leaves the other configuration broken.
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, addr: u64) -> *const std::sync::atomic::AtomicU64 {
        unsafe { (*self.target).gen_ptr(self.target_addr(addr)) }
    }
}

/// A RAM bank answering `delta` bytes up the bus from where its own address
/// mask would put it: `addr - delta` reaches the bank, which wraps it at its
/// size. For a memory controller that places a bank at a base and counts
/// offsets from there (CRIME), where the base is not a multiple of the bank's
/// size.
struct BankView {
    bank: *const RamBank,
    delta: u64,
}

unsafe impl Send for BankView {}
unsafe impl Sync for BankView {}

impl BankView {
    #[inline(always)]
    fn bank(&self) -> &RamBank {
        unsafe { &*self.bank }
    }
    #[inline(always)]
    fn at(&self, addr: u64) -> u64 {
        addr.wrapping_sub(self.delta)
    }
}

impl BusDevice for BankView {
    #[inline] fn read8(&self, a: u64) -> BusRead8 { self.bank().read8(self.at(a)) }
    #[inline] fn write8(&self, a: u64, v: u8) -> u32 { self.bank().write8(self.at(a), v) }
    #[inline] fn read16(&self, a: u64) -> BusRead16 { self.bank().read16(self.at(a)) }
    #[inline] fn write16(&self, a: u64, v: u16) -> u32 { self.bank().write16(self.at(a), v) }
    #[inline] fn read32(&self, a: u64) -> BusRead32 { self.bank().read32(self.at(a)) }
    #[inline] fn write32(&self, a: u64, v: u32) -> u32 { self.bank().write32(self.at(a), v) }
    #[inline] fn read64(&self, a: u64) -> BusRead64 { self.bank().read64(self.at(a)) }
    #[inline] fn write64(&self, a: u64, v: u64) -> u32 { self.bank().write64(self.at(a), v) }
    #[inline] fn write64_masked(&self, a: u64, v: u64, m: u64) -> u32 { self.bank().write64_masked(self.at(a), v, m) }
    #[inline] fn mem_ptr(&self, a: u64) -> Option<*const u64> { self.bank().mem_ptr(self.at(a)) }
    #[inline] fn read_block(&self, a: u64, buf: &mut [u64]) -> u32 { self.bank().read_block(self.at(a), buf) }
    #[inline] fn write_block(&self, a: u64, buf: &[u64]) -> u32 { self.bank().write_block(self.at(a), buf) }
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, a: u64) -> *const std::sync::atomic::AtomicU64 { self.bank().gen_ptr(self.at(a)) }
}

/// Guest memory from `addr` into `buf`, through `bus` as a DMA master sees
/// it: whole qwords with `read_block` a page at a time, single bytes at the
/// ragged ends. Returns `BUS_OK` or the first other status.
pub fn read_bytes(bus: &dyn BusDevice, addr: u64, buf: &mut [u8]) -> u32 {
    const PAGE: u64 = 4096;
    let mut done = 0usize;
    let mut words = [0u64; (PAGE / 8) as usize];
    while done < buf.len() {
        let a = addr.wrapping_add(done as u64);
        let left = buf.len() - done;
        if a & 7 != 0 || left < 8 {
            let r = bus.read8(a);
            if !r.is_ok() {
                return r.status;
            }
            buf[done] = r.data;
            done += 1;
            continue;
        }
        let n = (((PAGE - (a & (PAGE - 1))) / 8) as usize).min(left / 8);
        let st = bus.read_block(a, &mut words[..n]);
        if st != BUS_OK {
            return st;
        }
        for (k, w) in words[..n].iter().enumerate() {
            buf[done + 8 * k..done + 8 * k + 8].copy_from_slice(&w.to_be_bytes());
        }
        done += 8 * n;
    }
    BUS_OK
}

/// `data` into guest memory from `addr`, through `bus`: the write twin of
/// [`read_bytes`].
pub fn write_bytes(bus: &dyn BusDevice, addr: u64, data: &[u8]) -> u32 {
    const PAGE: u64 = 4096;
    let mut done = 0usize;
    let mut words = [0u64; (PAGE / 8) as usize];
    while done < data.len() {
        let a = addr.wrapping_add(done as u64);
        let left = data.len() - done;
        if a & 7 != 0 || left < 8 {
            let st = bus.write8(a, data[done]);
            if st != BUS_OK {
                return st;
            }
            done += 1;
            continue;
        }
        let n = (((PAGE - (a & (PAGE - 1))) / 8) as usize).min(left / 8);
        for (k, w) in words[..n].iter_mut().enumerate() {
            *w = u64::from_be_bytes(data[done + 8 * k..done + 8 * k + 8].try_into().unwrap());
        }
        let st = bus.write_block(a, &words[..n]);
        if st != BUS_OK {
            return st;
        }
        done += 8 * n;
    }
    BUS_OK
}

/// Physical Bus (Physical)
///
/// Routes every bus access to the device the decode table holds for its
/// 4 KB slot: one indexed load from a table that is sparse and
/// host-MMU-backed (`SparseDeviceMap`), so it spans a 40-bit space while
/// costing a few pages. Built by [`Physical::new`] over a machine's RAM banks
/// and filled in by the machine; see the module notes.
pub struct Physical {
    /// The RAM banks, in the order the machine's memory controller numbers
    /// them. Placed in the table and the window by the machine's remap.
    banks: Vec<RamBank>,
    /// ppmem's 4GB window over `banks`. The machine maps banks into it in
    /// addition to placing them in the decode table, so the mapping-based fast
    /// path and the bus path stay in agreement. `None` if the window could not
    /// be reserved, in which case only the bus path is used.
    ppmem_space: Option<PpMemSpace>,
    /// Base of ppmem's 4GB data window, or null when ppmem is unavailable.
    /// Cached here so the hot path is a load + shift + test with no `Option`
    /// unwrapping and no walk through `PpMemSpace`.
    ppmem_base: *mut u8,
    /// Base of ppmem's 8MB generation window (jitv2 only).
    #[cfg(feature = "jitv2")]
    ppmem_gen_base: *mut std::sync::atomic::AtomicU64,
    /// The live mapped-region bitmap. Points at the CPU's inline
    /// `MipsCore::ppmem_bitmap` once the CPU claims it (see
    /// [`Self::resync_ppmem_bitmap`]), at `PpMemSpace`'s own `u64` before
    /// that; null only without a window.
    ppmem_bitmap: AtomicPtr<u64>,

    /// The decode table: one device per 4 KB slot of a 40-bit physical
    /// space, sparse and host-MMU-backed (`src/ppmem/devmap.rs`). Slots the
    /// machine fills in are filled at configuration; a slot nobody filled is
    /// faulted in as the default device the first time it is touched.
    devmap: SparseDeviceMap,
    /// `devmap.base()`, cached so the hot path is a load off `self` and one
    /// indexed load.
    map_base: *const Entry,

    /// Devices the table points at that this bus keeps alive, by name for
    /// `phys map`.
    devices: Mutex<Vec<(Arc<dyn BusDevice>, String)>>,
    /// Devices the table points at that the machine keeps alive (fields of
    /// its own long-lived bus structure), by name.
    borrowed: Mutex<Vec<(Entry, String)>>,
    /// Address ranges that re-dispatch into this bus at an offset. Boxed so
    /// the table's pointers survive the vector growing.
    aliases: Mutex<Vec<(Box<AliasBus>, String)>>,
    /// Bank views handed out by [`Self::bank_view`], kept for the life of
    /// the bus: a device thread may be dispatching through one a remap has
    /// just replaced.
    bank_views: Mutex<Vec<Box<BankView>>>,
    /// What [`Self::set_default`] installed, for a remap to put back.
    default: Mutex<Option<Entry>>,

    unmapped_ram: UnmappedRam,
    black_hole: BlackHoleRegion,
    error_bus: ErrorBus,

    trace: AtomicBool,
    start_tick: u64,
    host_freq: u64,
}

// Safety: the table's pointers point at devices this bus or its machine keeps
// alive for as long as the bus is used, every device is Send + Sync, and the
// window state follows `PpMemSpace`'s CPU-thread contract.
unsafe impl Send for Physical {}
unsafe impl Sync for Physical {}

impl Physical {
    /// A bus over `banks`, with an empty table: nothing answers until the
    /// machine sets a default and maps its devices. Returned in an `Arc`
    /// because aliases re-dispatch into the bus by address, so it must be in
    /// its final place before anything is mapped.
    pub fn new(banks: Vec<RamBank>) -> Arc<Self> {
        let host_freq = crate::platform::get_host_tick_frequency();
        let start_tick = crate::platform::get_host_ticks();

        let devmap = SparseDeviceMap::new().unwrap_or_else(|e| {
            panic!("iris: cannot reserve the physical address decode table: {e}")
        });
        let map_base = devmap.base();

        // ppmem: reserve the 4GB window over these banks. A failure here is
        // not fatal — the bus path works regardless — so log and carry on
        // rather than refusing to boot.
        let ppmem_space = match PpMemSpace::over(&banks) {
            Ok(sp) => Some(sp),
            Err(e) => {
                dlog_dev!(LogModule::Mc, "ppmem: could not reserve 4GB window ({e}); \
                                          falling back to bus-only access");
                None
            }
        };
        let (ppmem_base, ppmem_bitmap) = match &ppmem_space {
            Some(sp) => (sp.window_base(), sp.bitmap_ptr() as *mut u64),
            None => (std::ptr::null_mut(), std::ptr::null_mut()),
        };
        #[cfg(feature = "jitv2")]
        let ppmem_gen_base = match &ppmem_space {
            Some(sp) => sp.gen_window_base(),
            None => std::ptr::null_mut(),
        };

        Arc::new(Self {
            banks,
            ppmem_space,
            ppmem_base,
            #[cfg(feature = "jitv2")]
            ppmem_gen_base,
            ppmem_bitmap: AtomicPtr::new(ppmem_bitmap),
            devmap,
            map_base,
            devices: Mutex::new(Vec::new()),
            borrowed: Mutex::new(Vec::new()),
            aliases: Mutex::new(Vec::new()),
            bank_views: Mutex::new(Vec::new()),
            default: Mutex::new(None),
            unmapped_ram: UnmappedRam,
            black_hole: BlackHoleRegion::new(),
            error_bus: ErrorBus::new(),
            trace: AtomicBool::new(false),
            start_tick,
            host_freq,
        })
    }

    fn fill_range(&self, base: u64, end: u64, dev: Entry) {
        self.devmap.fill_bytes(base, end, dev)
            .unwrap_or_else(|e| panic!("physical: mapping {base:#010x}..{end:#010x}: {e}"));
    }

    /// Keep `dev` alive and named, once however often it is mapped.
    fn keep(&self, dev: Arc<dyn BusDevice>, name: &str) -> Entry {
        let p: Entry = Arc::as_ptr(&dev);
        let mut d = self.devices.lock().unwrap();
        if !d.iter().any(|(k, _)| entry_eq(Arc::as_ptr(k), p)) {
            d.push((dev, name.to_string()));
        }
        p
    }

    /// What answers wherever nothing else is mapped: the whole 32-bit space
    /// now, and any slot above it the first time the guest reaches it.
    pub fn set_default(&self, dev: Arc<dyn BusDevice>, name: &str) {
        let p = self.keep(dev, name);
        *self.default.lock().unwrap() = Some(p);
        self.devmap.set_default(p)
            .unwrap_or_else(|e| panic!("physical: setting the default device: {e}"));
        self.devmap.fill(0, 1 << (32 - SLOT_SHIFT), p)
            .unwrap_or_else(|e| panic!("physical: filling the 32-bit space: {e}"));
    }

    /// `dev` answers at `base..end` (4 KB granular). The bus keeps it alive.
    pub fn map(&self, base: u64, end: u64, dev: Arc<dyn BusDevice>, name: &str) {
        let p = self.keep(dev, name);
        self.fill_range(base, end, p);
    }

    /// `dev` answers at `base..end`, owned by the machine rather than the bus.
    ///
    /// # Safety
    /// `dev` must outlive every access through this bus: a field of a
    /// structure the machine keeps in an `Arc` alongside it, for instance.
    pub unsafe fn map_borrowed(&self, base: u64, end: u64, dev: Entry, name: &str) {
        {
            let mut b = self.borrowed.lock().unwrap();
            if !b.iter().any(|(k, _)| entry_eq(*k, dev)) {
                b.push((dev, name.to_string()));
            }
        }
        self.fill_range(base, end, dev);
    }

    /// `base..end` answers as this bus does at the address plus `offset`
    /// (wrapping in 32 bits), so an alias follows whatever is mapped at its
    /// target, RAM included.
    pub fn map_alias(&self, base: u64, end: u64, offset: u32, name: &str) {
        let me: *const dyn BusDevice = self as *const Physical;
        let alias = Box::new(AliasBus::new(me, offset));
        let p: Entry = &*alias;
        self.aliases.lock().unwrap().push((alias, name.to_string()));
        self.fill_range(base, end, p);
    }

    /// `base..end` reads as zero and ignores writes.
    pub fn map_unmapped(&self, base: u64, end: u64) {
        self.fill_range(base, end, self.unmapped_entry());
    }

    /// `base..end` reads as zero and ignores writes, logging both under
    /// `phys hole on`.
    pub fn map_black_hole(&self, base: u64, end: u64) {
        let p: Entry = &self.black_hole;
        self.fill_range(base, end, p);
    }

    /// The table entry for RAM bank `i`, for a remap.
    pub fn bank_entry(&self, i: usize) -> Entry {
        &self.banks[i]
    }

    /// The table entry for RAM bank `i` answering `delta` bytes up the bus
    /// (`addr - delta` reaches the bank), for a remap. One view per
    /// `(i, delta)`, kept for the life of the bus.
    pub fn bank_view(&self, i: usize, delta: u64) -> Entry {
        let bank: *const RamBank = &self.banks[i];
        let mut v = self.bank_views.lock().unwrap();
        if let Some(b) = v.iter().find(|b| std::ptr::eq(b.bank, bank) && b.delta == delta) {
            return &**b as Entry;
        }
        v.push(Box::new(BankView { bank, delta }));
        &**v.last().unwrap() as Entry
    }

    /// The table entry for what answers where nothing is mapped, for a
    /// remap to put back. Panics before [`Self::set_default`].
    pub fn default_entry(&self) -> Entry {
        self.default.lock().unwrap().expect("physical: no default device set")
    }

    /// The table entry for memory that reads zero, for a remap.
    pub fn unmapped_entry(&self) -> Entry {
        &self.unmapped_ram
    }

    /// Point `count` slots from `first` (slot numbers, `addr >> SLOT_SHIFT`) at `dev`, unless
    /// they already all answer that way. A slot that does change is swapped
    /// in whole table pages, so a device thread dispatching through the table
    /// meanwhile sees the old entry or the new one, never a torn pointer.
    pub fn fill_slots(&self, first: u64, count: u64, dev: Entry) {
        if !self.devmap.all_are(first, count, dev) {
            self.devmap.fill(first, count, dev).unwrap_or_else(|e| {
                panic!("physical: remapping slots {first:#x}+{count:#x}: {e}")
            });
        }
    }

    /// The RAM banks.
    pub fn banks(&self) -> &[RamBank] {
        &self.banks
    }

    /// Save a bank's contents to a binary file.
    pub fn save_bank(&self, bank: usize, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.banks[bank].save_bin(path)
    }

    /// Load a bank's contents from a binary file.
    pub fn load_bank(&self, bank: usize, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.banks[bank].load_bin(path)
    }

    /// Reset all banks to zero (power-on state).
    pub fn reset_memory(&self) {
        use crate::traits::Resettable;
        for bank in &self.banks {
            bank.power_on();
        }
    }

    /// Snapshot bank `bank` into a native-endian Vec<u32>. Used by the
    /// in-memory rollback checkpoint to skip the disk byte-shuffle.
    pub fn snapshot_bank_inmem(&self, bank: usize) -> Vec<u32> {
        self.banks[bank].snapshot_words()
    }

    /// Restore bank `bank` from a buffer produced by `snapshot_bank_inmem`.
    pub fn restore_bank_inmem(&self, bank: usize, src: &[u32]) {
        self.banks[bank].restore_words(src);
    }

    /// ppmem: the 4GB window banks are mapped into, if one was reserved.
    pub fn ppmem_space(&self) -> Option<&PpMemSpace> {
        self.ppmem_space.as_ref()
    }

    /// Re-read the bitmap sink from the space.
    ///
    /// `PpMemSpace::set_bitmap_sink` moves publication from the space's own
    /// `u64` to the CPU's inline field; without this, the bus's cached
    /// pointer would keep reading the abandoned one and never see another
    /// remap. Call immediately after handing the CPU's pointer over.
    pub fn resync_ppmem_bitmap(&self) {
        if let Some(sp) = &self.ppmem_space {
            self.ppmem_bitmap.store(sp.bitmap_ptr() as *mut u64, Ordering::Relaxed);
        }
    }

    /// The table entry answering `addr` (masked to the table's bits), for
    /// a view that routes by what the machine mapped there: compare it with
    /// [`Self::unmapped_entry`] or [`Self::default_entry`].
    pub fn entry_at(&self, addr: u64) -> Entry {
        self.dev_at(addr & PHYS_MASK)
    }

    /// The device behind `addr`: one indexed load from the decode table.
    #[inline(always)]
    fn dev_at(&self, addr: u64) -> Entry {
        // SAFETY: `map_base` spans the whole table for the life of `devmap`;
        // a page nobody filled is faulted in by devmap's handler, so every
        // load lands on a readable page holding a live device pointer.
        unsafe { *self.map_base.add((addr >> SLOT_SHIFT) as usize) }
    }

    /// Name a decode-table entry for the monitor.
    fn device_name(&self, dev: Entry) -> String {
        fn e<T: BusDevice + 'static>(t: &T) -> Entry { t }
        let mut known: Vec<(Entry, String)> = vec![
            (e(&self.unmapped_ram), "unmapped ram (reads 0)".into()),
            (e(&self.black_hole), "black hole".into()),
            (e(&self.error_bus), "bus error".into()),
        ];
        for (i, b) in self.banks.iter().enumerate() {
            known.push((e(b), format!("bank {i}")));
        }
        for (d, n) in self.devices.lock().unwrap().iter() {
            known.push((Arc::as_ptr(d), n.clone()));
        }
        for (d, n) in self.borrowed.lock().unwrap().iter() {
            known.push((*d, n.clone()));
        }
        for (a, n) in self.aliases.lock().unwrap().iter() {
            known.push((&**a as Entry, n.clone()));
        }
        for v in self.bank_views.lock().unwrap().iter() {
            let i = self.banks.iter().position(|b| std::ptr::eq(b, v.bank)).unwrap_or(usize::MAX);
            known.push((&**v as Entry, format!("bank {i} at +{:#x}", v.delta)));
        }
        known
            .into_iter()
            .find(|(p, _)| entry_eq(*p, dev))
            .map(|(_, n)| n)
            .unwrap_or_else(|| format!("{:p}", dev as *const ()))
    }

    /// The live bitmap, or 0 without a window.
    #[inline(always)]
    fn bitmap(&self) -> u64 {
        let p = self.ppmem_bitmap.load(Ordering::Relaxed);
        if p.is_null() { 0 } else { unsafe { *p } }
    }

    /// Is `addr` backed by a whole directly-mapped 64MB region?
    ///
    /// This is the quick test the design calls for: one load of the bitmap,
    /// one shift of the top bits of the physical address, one AND. A set bit
    /// means the entire 64MB region is mapped RAM, so a direct host access is
    /// equivalent to going through the bus.
    #[inline(always)]
    fn ppmem_mapped(&self, addr: u64) -> bool {
        crate::ppmem::bitmap_maps(self.bitmap(), addr)
    }

    /// Host pointer for a directly-mapped physical address, or `None` if the
    /// address is not in a fully-mapped region (MMIO, unmapped RAM, or ppmem
    /// unavailable).
    #[inline(always)]
    fn ppmem_ptr(&self, addr: u64) -> Option<*mut u64> {
        if !self.ppmem_mapped(addr) {
            return None;
        }
        Some(unsafe { self.ppmem_base.add(addr as usize) as *mut u64 })
    }

    /// Generation counter for a directly-mapped physical address.
    #[cfg(feature = "jitv2")]
    #[inline(always)]
    fn ppmem_gen_ptr(&self, addr: u64) -> Option<*const std::sync::atomic::AtomicU64> {
        if !self.ppmem_mapped(addr) || self.ppmem_gen_base.is_null() {
            return None;
        }
        let page = (addr >> 12) as usize;
        Some(unsafe { self.ppmem_gen_base.add(page) as *const std::sync::atomic::AtomicU64 })
    }

    /// Bump generation counters for every page a block write touches — the
    /// direct-path equivalent of `PpMemory::write_block`'s per-page cursor.
    #[cfg(feature = "jitv2")]
    #[inline]
    fn ppmem_bump_gen_range(&self, addr: u64, qwords: usize) {
        if self.ppmem_gen_base.is_null() {
            return;
        }
        let first = (addr >> 12) as usize;
        let last = ((addr + (qwords.max(1) - 1) as u64 * 8) >> 12) as usize;
        for page in first..=last.max(first) {
            unsafe {
                (*(self.ppmem_gen_base.add(page))).fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Device for Physical {
    fn step(&self, _cycles: u64) {
        // Timers are now updated on read/write based on host clock
    }

    fn stop(&self) {
    }

    fn start(&self) {
    }

    fn is_running(&self) -> bool {
        true
    }

    fn get_clock(&self) -> u64 {
        let now = crate::platform::get_host_ticks();
        let diff = now.wrapping_sub(self.start_tick);
        ((diff as u128 * 50_000_000) / (self.host_freq as u128)) as u64
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        let cmds = vec![
            ("phys".to_string(), "Physical Bus commands: mem, dis, map, trace, error <on|off>, hole <on|off>, bench".to_string()),
            ("mm".to_string(), "Alias for phys mem".to_string()),
            ("md".to_string(), "Alias for phys dis".to_string()),
            ("trace".to_string(), "Enable/disable tracing: trace <on|off>".to_string()),
            ("bench".to_string(), "Benchmark memory write performance".to_string()),
        ];
        cmds
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn IoWrite + Send>) -> Result<(), String> {
        // Handle "phys" prefix
        let (actual_cmd, actual_args) = if cmd == "phys" {
            if args.is_empty() {
                return Err("Usage: phys <command> [args...]".to_string());
            }
            (args[0], &args[1..])
        } else {
            (cmd, args)
        };

        match actual_cmd {
            "help" => {
                writeln!(writer, "Physical Bus Commands:").unwrap();
                for (c, h) in self.register_commands() {
                    writeln!(writer, "  {:12} - {}", c, h).unwrap();
                }
            }
            "mem" | "m" | "mm" => {
                if actual_args.is_empty() {
                    return Err("Usage: mem <addr>".to_string());
                }
                let addr = eval_const_expr(actual_args[0])
                    .map_err(|e| format!("mem: {}", e))?;

                let r = BusDevice::read32(self, addr);
                if r.is_ok() { writeln!(writer, "{:08x}: {:08x}", addr, r.data).unwrap(); }
                else { writeln!(writer, "{:08x}: Error/Busy", addr).unwrap(); }
            }
            "dis" | "d" | "md" => {
                if actual_args.is_empty() {
                    return Err("Usage: dis <addr>".to_string());
                }
                let addr = eval_const_expr(actual_args[0])
                    .map_err(|e| format!("dis: {}", e))?;

                let r = BusDevice::read32(self, addr);
                if r.is_ok() { writeln!(writer, "{}", mips_dis::disassemble(r.data, addr, None)).unwrap(); }
                else { writeln!(writer, "Could not fetch instruction at {:016x}", addr).unwrap(); }
            }
            "map" => {
                let s = self.devmap.stats();
                writeln!(writer,
                    "decode table: {} pages of {} KB ({} MB of guest space each): {} untouched, {} filled, not yet mapped, {} shared, {} private",
                    s.pages, s.page_bytes >> 10, ((s.page_bytes as u64 / 16) << SLOT_SHIFT) >> 20,
                    s.unmapped, s.lazy, s.shared, s.private).unwrap();
                writeln!(writer,
                    "  {} constant page(s), {} pool page(s) ({} free), {} KB in use, {} fault(s)",
                    s.const_pages, s.pool_pages, s.pool_free, s.in_use_bytes() >> 10, s.faults).unwrap();
                for a in &s.first_faults {
                    writeln!(writer, "  faulted in: {:#012x}", a).unwrap();
                }
                for r in self.devmap.runs() {
                    let name = match r.dev {
                        Some(d) => self.device_name(d),
                        None => "(untouched)".to_string(),
                    };
                    writeln!(writer, "  {:#012x}-{:#012x} {}",
                        r.start << SLOT_SHIFT, (r.end << SLOT_SHIFT) - 1, name).unwrap();
                }
            }
            "error" => {
                if actual_args.is_empty() {
                    return Err("Usage: error <on|off>".to_string());
                }
                let val = match actual_args[0] {
                    "on" | "1" => true,
                    "off" | "0" => false,
                    _ => return Err("Usage: error <on|off>".to_string()),
                };
                self.error_bus.set_debug(val);
                writeln!(writer, "Bus Error debug {}", if val { "enabled" } else { "disabled" }).unwrap();
            }
            "hole" => {
                if actual_args.is_empty() {
                    return Err("Usage: hole <on|off>".to_string());
                }
                let val = match actual_args[0] {
                    "on" | "1" => true,
                    "off" | "0" => false,
                    _ => return Err("Usage: hole <on|off>".to_string()),
                };
                self.black_hole.set_debug(val);
                writeln!(writer, "Black Hole debug {}", if val { "enabled" } else { "disabled" }).unwrap();
            }
            "trace" => {
                if actual_args.is_empty() {
                    let state = if self.trace.load(Ordering::Relaxed) { "on" } else { "off" };
                    writeln!(writer, "MC trace is {}", state).unwrap();
                } else {
                    match actual_args[0] {
                        "on" | "1" => {
                            self.trace.store(true, Ordering::Relaxed);
                            writeln!(writer, "MC trace enabled").unwrap();
                        }
                        "off" | "0" => {
                            self.trace.store(false, Ordering::Relaxed);
                            writeln!(writer, "MC trace disabled").unwrap();
                        }
                        _ => return Err("Usage: trace <on|off|1|0>".to_string()),
                    }
                }
            }
            "bench" => {
                // The first 64 MB region the window maps whole: RAM on every
                // machine, wherever its memory controller put it. Overwrites it.
                let bm = self.bitmap();
                if bm == 0 {
                    return Err("bench: no RAM region is mapped yet".to_string());
                }
                let size = 1u64 << crate::ppmem::BITMAP_SHIFT;
                let base = (bm.trailing_zeros() as u64) << crate::ppmem::BITMAP_SHIFT;
                writeln!(writer, "Benchmarking Physical Bus write32 performance at {base:#x}...").unwrap();
                let bench_start = crate::platform::get_host_ticks();
                let mut error_count = 0;
                for addr in (base..base + size).step_by(4) {
                    if self.write32(addr, 0) != BUS_OK {
                        error_count += 1;
                    }
                }
                let bench_end = crate::platform::get_host_ticks();
                if error_count > 0 {
                    writeln!(writer, "  (had {} errors)", error_count).unwrap();
                }
                let bench_elapsed = bench_end.wrapping_sub(bench_start);
                let bench_freq = crate::platform::get_host_tick_frequency();
                let bench_elapsed_us = (bench_elapsed as f64 / bench_freq as f64) * 1_000_000.0;
                let bench_elapsed_s = bench_elapsed_us / 1_000_000.0;
                let mb_per_s = (size as f64 / (1024.0 * 1024.0)) / bench_elapsed_s;
                let cycles_per_word = (bench_freq as f64 * bench_elapsed_s) / ((size / 4) as f64);
                writeln!(writer, "Physical Bus: Filled {}MB in {:.3} us ({} ticks) = {:.2} MB/s, {:.1} cycles/word",
                    size / (1024 * 1024), bench_elapsed_us, bench_elapsed, mb_per_s, cycles_per_word).unwrap();
            }
            _ => return Err(format!("Unknown Physical command: {}", cmd)),
        }
        Ok(())
    }
}

impl BusDevice for Physical {
    /// Ask the device the decode table names for `addr`, as an access to it
    /// would have been routed, and name that device in the answer.
    fn describe_fault(&self, addr: u64, size: usize) -> Option<String> {
        let addr = addr & PHYS_MASK;
        let dev = self.dev_at(addr);
        let why = unsafe { (*dev).describe_fault(addr, size) }?;
        Some(format!("{}: {why}", self.device_name(dev)))
    }

    #[inline(always)]
    fn read8(&self, addr: u64) -> BusRead8 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        let r = unsafe { (*device_ptr).read8(addr) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) {
            if r.is_ok() { println!("PHYS8 Read {:08x} -> {:02x}", addr, r.data); }
            else { println!("PHYS8 Read {:08x} -> err {:08x}", addr, r.status); }
        }
        r
    }

    #[inline(always)]
    fn write8(&self, addr: u64, val: u8) -> u32 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        let ws = unsafe { (*device_ptr).write8(addr, val) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) { println!("PHYS8 Write {:08x} val={:02x} -> {:08x}", addr, val, ws); }
        ws
    }

    #[inline(always)]
    fn read16(&self, addr: u64) -> BusRead16 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        let r = unsafe { (*device_ptr).read16(addr) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) {
            if r.is_ok() { println!("PHYS16 Read {:08x} -> {:04x}", addr, r.data); }
            else { println!("PHYS16 Read {:08x} -> err {:08x}", addr, r.status); }
        }
        r
    }

    #[inline(always)]
    fn write16(&self, addr: u64, val: u16) -> u32 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        let ws = unsafe { (*device_ptr).write16(addr, val) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) { println!("PHYS16 Write {:08x} val={:04x} -> {:08x}", addr, val, ws); }
        ws
    }

    #[inline(always)]
    fn read32(&self, addr: u64) -> BusRead32 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        let r = unsafe { (*device_ptr).read32(addr) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) {
            if r.is_ok() { println!("PHYS32 Read {:08x} -> {:08x}", addr, r.data); }
            else { println!("PHYS32 Read {:08x} -> err {:08x}", addr, r.status); }
        }
        r
    }

    #[inline(always)]
    fn write32(&self, addr: u64, val: u32) -> u32 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        let ws = unsafe { (*device_ptr).write32(addr, val) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) { println!("PHYS32 Write {:08x} val={:08x} -> {:08x}", addr, val, ws); }
        ws
    }

    #[inline(always)]
    fn read64(&self, addr: u64) -> BusRead64 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        let r = unsafe { (*device_ptr).read64(addr) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) {
            if r.is_ok() { println!("PHYS64 Read {:08x} -> {:016x}", addr, r.data); }
            else { println!("PHYS64 Read {:08x} -> err {:08x}", addr, r.status); }
        }
        r
    }

    #[inline(always)]
    fn write64(&self, addr: u64, val: u64) -> u32 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        let ws = unsafe { (*device_ptr).write64(addr, val) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) { println!("PHYS64 Write {:08x} val={:016x} -> {:08x}", addr, val, ws); }
        ws
    }

    #[inline(always)]
    fn write64_masked(&self, addr: u64, val: u64, mask: u64) -> u32 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        unsafe { (*device_ptr).write64_masked(addr, val, mask) }
    }

    // Route to the target device's dma_read64/dma_write64, not its read64/write64.
    // Without these overrides, BusDevice's default dma_read64/dma_write64 (which
    // call self.read64/self.write64) resolve against Physical itself — reaching
    // the target device's plain read64/write64 one layer too early and skipping
    // any DMA-specific override (e.g. Rex3::dma_read64) entirely.
    #[inline(always)]
    fn dma_read64(&self, addr: u64) -> BusRead64 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        unsafe { (*device_ptr).dma_read64(addr) }
    }

    #[inline(always)]
    fn dma_write64(&self, addr: u64, val: u64) -> u32 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        unsafe { (*device_ptr).dma_write64(addr, val) }
    }

    // Same reason as the scalar pair above: without these, the trait's default
    // bulk loop would run against Physical and call *its* dma_write64 per word,
    // which forwards correctly but one word at a time — losing the entire point
    // of batching. Forward the whole slice so the device's own bulk override
    // (Rex3's single-token push) actually gets a chance to run.
    #[inline(always)]
    fn dma_write64_bulk(&self, addr: u64, vals: &[u64]) -> u32 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        unsafe { (*device_ptr).dma_write64_bulk(addr, vals) }
    }

    #[inline(always)]
    fn dma_read64_bulk(&self, addr: u64, out: &mut [u64]) -> u32 {
        let addr = addr & PHYS_MASK;
        let device_ptr = self.dev_at(addr);
        unsafe { (*device_ptr).dma_read64_bulk(addr, out) }
    }

    #[cfg(feature = "jitv2")]
    #[inline(always)]
    fn gen_ptr(&self, addr: u64) -> *const std::sync::atomic::AtomicU64 {
        let addr = addr & PHYS_MASK;
        // ppmem: the counter for a directly-mapped page is at a constant
        // offset in the gen window — a pure shift off the physical address,
        // no bank lookup. See docs/ppmem-design.md §6.2.
        #[cfg(feature = "jitv2")]
        if let Some(p) = self.ppmem_gen_ptr(addr) {
            return p;
        }
        let device_ptr = self.dev_at(addr);
        unsafe { (*device_ptr).gen_ptr(addr) }
    }

    /// Direct pointer into ppmem's window for a directly-mapped physical
    /// address, so the cache's line-fill fast path (`mips_cache_v2.rs`) reads
    /// RAM without a device dispatch.
    ///
    /// Falls through to the device's own `mem_ptr` — and thus to `None` for
    /// MMIO — whenever the address is not backed by a whole mapped region.
    #[inline]
    fn mem_ptr(&self, addr: u64) -> Option<*const u64> {
        let addr = addr & PHYS_MASK;
        if let Some(p) = self.ppmem_ptr(addr) {
            return Some(p as *const u64);
        }
        let device_ptr = self.dev_at(addr);
        unsafe { (*device_ptr).mem_ptr(addr) }
    }

    #[inline]
    fn read_block(&self, addr: u64, buf: &mut [u64]) -> u32 {
        let addr = addr & PHYS_MASK;
        if let Some(p) = self.ppmem_ptr(addr) {
            // Same layout as PpMemory::read_block — storage keeps qwords
            // rotate_left(32).
            unsafe {
                for (i, slot) in buf.iter_mut().enumerate() {
                    *slot = (*p.add(i)).rotate_left(32);
                }
            }
            return BUS_OK;
        }
        let device_ptr = self.dev_at(addr);
        unsafe { (*device_ptr).read_block(addr, buf) }
    }

    #[inline]
    fn write_block(&self, addr: u64, buf: &[u64]) -> u32 {
        let addr = addr & PHYS_MASK;
        if let Some(p) = self.ppmem_ptr(addr) {
            unsafe {
                for (i, &val) in buf.iter().enumerate() {
                    *p.add(i) = val.rotate_left(32);
                }
            }
            // The gen bump still has to happen: a cache writeback mutates RAM
            // under any compiled artifact for those pages.
            #[cfg(feature = "jitv2")]
            self.ppmem_bump_gen_range(addr, buf.len());
            return BUS_OK;
        }
        let device_ptr = self.dev_at(addr);
        unsafe { (*device_ptr).write_block(addr, buf) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bus with one 1 MB bank placed at 1 MB, an alias of it at 0, a black
    /// hole and a bus-error default: each answers where it was mapped.
    #[test]
    fn mapped_ranges_answer_as_configured() {
        let phys = Physical::new(vec![RamBank::new(1)]);
        phys.set_default(Arc::new(ErrorBus::new()), "error");
        phys.fill_slots(0x10_0000 >> SLOT_SHIFT, 0x10_0000 >> SLOT_SHIFT, phys.bank_entry(0));
        phys.map_alias(0, 0x10_0000, 0x10_0000, "alias");
        phys.map_black_hole(0x20_0000, 0x21_0000);

        assert_eq!(phys.write32(0x10_0040, 0x1234_5678), BUS_OK);
        assert_eq!(phys.read32(0x10_0040).data, 0x1234_5678, "the bank");
        assert_eq!(phys.read32(0x40).data, 0x1234_5678, "the alias reaches the bank");
        assert!(phys.read32(0x20_0000).is_ok(), "the black hole answers");
        assert!(!phys.read32(0x30_0000).is_ok(), "the default is a bus error");
        assert!(!phys.read32(0x1_0000_0000).is_ok(), "above 4 GB too");
    }

    /// A bank view reaches the bank counting from its base, and the byte
    /// copies agree with word accesses across a page and at ragged ends.
    #[test]
    fn bank_views_and_byte_copies() {
        let phys = Physical::new(vec![RamBank::new(1)]);
        phys.set_default(Arc::new(ErrorBus::new()), "error");
        // The 1 MB bank at 3 MB: offsets count from there, not from its mask.
        phys.fill_slots(0x30_0000 >> SLOT_SHIFT, 0x10_0000 >> SLOT_SHIFT, phys.bank_view(0, 0x30_0000));
        assert_eq!(phys.bank_view(0, 0x30_0000), phys.bank_view(0, 0x30_0000), "one view per placement");
        assert_eq!(phys.write32(0x30_0010, 0xdead_beef), BUS_OK);
        assert_eq!(phys.banks()[0].read32(0x10).data, 0xdead_beef);

        let data: Vec<u8> = (0..5000u32).map(|i| (i * 7 + 3) as u8).collect();
        assert_eq!(write_bytes(&*phys, 0x30_0ffd, &data), BUS_OK);
        let mut back = vec![0u8; data.len()];
        assert_eq!(read_bytes(&*phys, 0x30_0ffd, &mut back), BUS_OK);
        assert_eq!(back, data);
        assert_eq!(phys.read8(0x30_0ffd).data, data[0]);
        assert_eq!(phys.read32(0x30_1000).data, u32::from_be_bytes(data[3..7].try_into().unwrap()));
        let mut none = [0u8; 4];
        assert_ne!(read_bytes(&*phys, 0x50_0000, &mut none), BUS_OK, "nothing there");
    }

    /// The low-512KB alias must resolve to the SAME generation counter as
    /// the memory it aliases — through the bus as through the window.
    ///
    /// `AliasBus` forwarded all eight read/write methods but not `gen_ptr`,
    /// so it fell through to the trait default (null). A null `gen_ptr` makes
    /// `PhysicalCodePage::claim` use the shared `NEVER_COMPILABLE_GEN`, which
    /// is initialised to 0 and never bumped: `j2 pcp` reported
    /// `pfn=0 gen=0 entry_gen=0` for the TLB refill vector page after the
    /// kernel had patched it repeatedly, and the compiled code was never
    /// invalidated. (The window's twin, `PpMemSpace::map_alias`, had to be
    /// fixed as well.)
    #[cfg(feature = "jitv2")]
    #[test]
    fn alias_gen_counter_agrees_with_its_target() {
        const TARGET: u32 = 0x0800_0000;
        struct GenBank { ctr: std::sync::atomic::AtomicU64, base: u64 }
        impl BusDevice for GenBank {
            fn read8(&self, _a: u64) -> BusRead8 { BusRead8::ok(0) }
            fn write8(&self, _a: u64, _v: u8) -> u32 { BUS_OK }
            fn read16(&self, _a: u64) -> BusRead16 { BusRead16::ok(0) }
            fn write16(&self, _a: u64, _v: u16) -> u32 { BUS_OK }
            fn read32(&self, _a: u64) -> BusRead32 { BusRead32::ok(0) }
            fn write32(&self, _a: u64, _v: u32) -> u32 { BUS_OK }
            fn read64(&self, _a: u64) -> BusRead64 { BusRead64::ok(0) }
            fn write64(&self, _a: u64, _v: u64) -> u32 { BUS_OK }
            #[cfg(feature = "jitv2")]
            fn gen_ptr(&self, addr: u64) -> *const std::sync::atomic::AtomicU64 {
                // Only the page at `base` has a counter — anything else is a
                // different page and must NOT resolve here.
                if addr & !0xFFF == self.base { &self.ctr as *const _ } else { std::ptr::null() }
            }
        }

        let target = GenBank { ctr: std::sync::atomic::AtomicU64::new(0), base: TARGET as u64 };
        let target_ptr: *const dyn BusDevice = &target;
        let alias = AliasBus::new(target_ptr, TARGET);

        let via_alias = alias.gen_ptr(0);
        let via_direct = target.gen_ptr(TARGET as u64);

        assert!(!via_alias.is_null(),
            "AliasBus must forward gen_ptr — the trait default returns null, and a null \
             gen_ptr sends the page to NEVER_COMPILABLE_GEN (always 0), so stale JIT code \
             is never invalidated");
        assert_eq!(via_alias, via_direct,
            "the alias and the real address must resolve to ONE counter");
        assert!(alias.gen_ptr(0x1000).is_null(),
            "forwarding must apply the offset, not blanket-return the first counter");
    }
}
