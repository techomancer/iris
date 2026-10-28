//! A sparse, host-MMU-backed physical address decode table.
//!
//! `Physical` finds the device behind a physical address with one indexed
//! load: `table[addr >> 12]`, a `*const dyn BusDevice` per 4 KB, the
//! guest's page size and the smallest window a device or a PCI BAR gets. As
//! a plain array that is 16 MB for a 32-bit space and 4 GB for the 40-bit
//! space an Octane has, almost all of it the same few pointers repeated.
//! This keeps the one-load lookup and makes the array virtual:
//!
//! - **Reserved, not allocated.** The whole table is one `AddrSpace`
//!   reservation (`PROT_NONE` / a placeholder). A page nobody has written is
//!   not backed by anything.
//! - **Mapped when used.** Each table page is one host mapping, and hosts
//!   limit how many a process has (`vm.max_map_count`, 65530 by default on
//!   Linux). A whole page filled with one device is only recorded; the
//!   fault handler maps it on the first access. An Octane's 64 GiB widget
//!   windows cost nothing until the guest reaches into them.
//! - **Shared where it repeats.** A table page whose entries are all the same
//!   pointer is a view of one shared "constant page" for that pointer. Four
//!   banks and three error sinks cost seven pages however much address space
//!   they cover.
//! - **Built off-line, swapped in whole.** A mixed page lives in a pool of
//!   shared-memory pages. To change any entry the new contents are written
//!   into a fresh pool page through a private staging view, then mapped over
//!   the table page with `MAP_FIXED`. Readers on other threads (the MC's DMA
//!   worker dispatches through this table) see the old page or the new one,
//!   never a torn 16-byte fat pointer: the table is never written in place,
//!   and every table view is read-only so that cannot happen by accident.
//! - **Faults fill in the rest.** A load from a page that was never written
//!   is a host access violation. The handler installed here recognises the
//!   reservation, maps that page to what was filled there (or, where
//!   nothing was, to the table's default device, the bus-error sink, and
//!   prints the guest address once), and resumes the load.
//!   So an unmapped region costs nothing until the guest probes it, and a
//!   probe costs one fault.
//!
//! One table page is 64 KB (the Windows allocation granularity, and a
//! multiple of every Unix page size), 4096 entries, 16 MB of guest address
//! space. Every mapping is exactly one page, which is what the Windows
//! placeholder backend in `map_windows.rs` can split and replace.
//!
//! Who may call what: `fill` and `set_default` are for the owner's setup and
//! MEMCFG remaps, on the CPU thread; the fault path may run on any thread
//! that dispatches through the table. Both take the table's spin lock, which
//! is held only across mapping calls, never across anything that can block
//! or fault.

use super::map::{granularity, AddrSpace, Prot, SharedMem};
use crate::traits::BusDevice;
use std::cell::UnsafeCell;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};

/// One table entry: the device that answers a 4 KB slot.
pub type Entry = *const dyn BusDevice;

/// Width of the physical address space the table covers.
pub const PHYS_BITS: u32 = 40;
/// The bits of a physical address the table decodes. `Physical` masks every
/// address with this before dispatch, so a device never sees bits above it.
pub const PHYS_MASK: u64 = (1 << PHYS_BITS) - 1;
/// Bytes of guest address space per entry: 4 KB.
pub const SLOT_SHIFT: u32 = 12;
/// Entries in the table.
pub const SLOTS: u64 = 1 << (PHYS_BITS - SLOT_SHIFT);

const ENTRY_BYTES: usize = std::mem::size_of::<Entry>();
const TABLE_BYTES: usize = SLOTS as usize * ENTRY_BYTES;
/// A table page is at least this, whatever the host page size.
const MIN_PAGE_BYTES: usize = 64 * 1024;
/// Pool pages come in objects of this many pages.
const POOL_PAGES: usize = 16;
/// How many faulting addresses are remembered for the monitor.
const FIRST_FAULTS: usize = 8;

const _: () = assert!(ENTRY_BYTES == 2 * std::mem::size_of::<usize>(), "fat pointer layout");

/// Compare both halves of a fat pointer. Comparing only the data address is
/// wrong here: a zero-sized device (`UnmappedRam`) shares its address with
/// whatever field follows it, and only the vtable tells them apart.
#[inline]
fn same(a: Entry, b: Entry) -> bool {
    // SAFETY: `Entry` is two words, asserted above; reading them as integers
    // is the comparison `std::ptr::eq` would do on a stable `ptr_metadata`.
    unsafe {
        std::mem::transmute::<Entry, [usize; 2]>(a) == std::mem::transmute::<Entry, [usize; 2]>(b)
    }
}

/// Do two entries name the same device through the same vtable?
pub fn entry_eq(a: Entry, b: Entry) -> bool {
    same(a, b)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageState {
    /// Never written: reserved, faults on access.
    Unmapped,
    /// Filled whole with constant page `consts[k]` but never touched: still
    /// reserved, and the fault handler maps the view on first access. A
    /// table page is one host mapping, so mapping only what the guest uses
    /// keeps a 40-bit table with small slots well inside the host's mapping
    /// limit (`vm.max_map_count`).
    Lazy(u32),
    /// A view of constant page `consts[k]`.
    Const(u32),
    /// A view of pool page `slot` (object `slot / POOL_PAGES`, page within it
    /// `slot % POOL_PAGES`).
    Private(u32),
}

struct ConstObj {
    dev: Entry,
    mem: SharedMem,
}

struct State {
    pages: Vec<PageState>,
    consts: Vec<ConstObj>,
    pool: Vec<SharedMem>,
    free: Vec<u32>,
    /// One page of private address space the mutator writes new page contents
    /// through. Mapped read-write over whichever pool or constant page is
    /// being built; the table's own views stay read-only.
    staging: AddrSpace,
    /// The device an untouched slot resolves to: what the fault handler maps,
    /// and what fills the rest of a page that is written only in part. The
    /// constant page for it is created by `set_default`, so the fault path
    /// allocates nothing.
    default: Option<u32>,
}

struct Inner {
    table: AddrSpace,
    page_bytes: usize,
    per_page: usize,
    lock: AtomicBool,
    st: UnsafeCell<State>,
    faults: AtomicU64,
    /// Guest addresses of the first faults, `u64::MAX` where empty.
    first_faults: [AtomicU64; FIRST_FAULTS],
    /// Index in the fault registry, so `Drop` can clear it.
    reg: usize,
}

/// The table. See the module documentation.
pub struct SparseDeviceMap {
    // Boxed so the fault registry can hold a stable pointer while the owner
    // moves the map into place.
    inner: Box<Inner>,
}

// SAFETY: the raw pointers inside are device addresses the owner keeps alive
// for as long as the table; `State` is guarded by `lock`.
unsafe impl Send for SparseDeviceMap {}
unsafe impl Sync for SparseDeviceMap {}

/// One run of the table with one answer, for the monitor.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    /// First slot (guest address `>> SLOT_SHIFT`).
    pub start: u64,
    /// One past the last slot.
    pub end: u64,
    /// The device, or `None` for a run of pages nobody has written.
    pub dev: Option<Entry>,
}

/// Counts for the monitor.
#[derive(Clone, Debug, Default)]
pub struct MapStats {
    pub page_bytes: usize,
    pub pages: usize,
    pub unmapped: usize,
    pub shared: usize,
    /// Filled with one device but not yet touched, so not yet mapped.
    pub lazy: usize,
    pub private: usize,
    pub const_pages: usize,
    pub pool_pages: usize,
    pub pool_free: usize,
    pub faults: u64,
    pub first_faults: Vec<u64>,
}

impl MapStats {
    /// Host memory the table has in use: every constant page, every pool
    /// page a table page views, and the staging page. Free pool pages are
    /// address space in a shared object, not memory, until written.
    pub fn in_use_bytes(&self) -> usize {
        (self.const_pages + self.pool_pages - self.pool_free + 1) * self.page_bytes
    }
}

struct Guard<'a>(&'a AtomicBool);

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl Inner {
    fn lock(&self) -> Guard<'_> {
        while self
            .lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
        Guard(&self.lock)
    }

    /// As `lock`, but gives up: the fault handler must not hang a thread
    /// forever if it ever runs with the lock already held.
    fn try_lock_spinning(&self, spins: u32) -> Option<Guard<'_>> {
        for _ in 0..spins {
            if self
                .lock
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return Some(Guard(&self.lock));
            }
            std::hint::spin_loop();
        }
        None
    }

    #[inline]
    fn entries(&self) -> *const Entry {
        self.table.base() as *const Entry
    }

    fn npages(&self) -> usize {
        TABLE_BYTES / self.page_bytes
    }

    /// Map table page `page` as a read-only view of `mem[off..off+page]`.
    unsafe fn view(&self, page: usize, mem: &SharedMem, off: usize) -> io::Result<()> {
        unsafe { self.table.map(page * self.page_bytes, self.page_bytes, mem, off as u64, Prot::ReadOnly) }
    }

    /// Write `buf` into `mem[off..]` through the staging view.
    unsafe fn write_through_staging(&self, st: &State, mem: &SharedMem, off: usize, buf: &[Entry]) -> io::Result<()> {
        debug_assert_eq!(buf.len(), self.per_page);
        unsafe {
            st.staging.map(0, self.page_bytes, mem, off as u64, Prot::ReadWrite)?;
            std::ptr::copy_nonoverlapping(buf.as_ptr(), st.staging.base() as *mut Entry, self.per_page);
        }
        Ok(())
    }

    /// Index of the constant page for `dev`, created on first use.
    fn const_for(&self, st: &mut State, dev: Entry) -> io::Result<u32> {
        if let Some(k) = st.consts.iter().position(|c| same(c.dev, dev)) {
            return Ok(k as u32);
        }
        let mem = SharedMem::new(self.page_bytes)?;
        let buf = vec![dev; self.per_page];
        unsafe { self.write_through_staging(st, &mem, 0, &buf)? };
        st.consts.push(ConstObj { dev, mem });
        Ok((st.consts.len() - 1) as u32)
    }

    fn alloc_slot(&self, st: &mut State) -> io::Result<u32> {
        if let Some(s) = st.free.pop() {
            return Ok(s);
        }
        let obj = st.pool.len() as u32;
        st.pool.push(SharedMem::new(POOL_PAGES * self.page_bytes)?);
        // Hand out low pages first; the order only matters for the tests.
        for p in (0..POOL_PAGES as u32).rev() {
            st.free.push(obj * POOL_PAGES as u32 + p);
        }
        Ok(st.free.pop().unwrap())
    }

    fn release(&self, st: &mut State, old: PageState) {
        if let PageState::Private(s) = old {
            st.free.push(s);
        }
    }

    /// Point a whole page at `dev`.
    fn map_const(&self, st: &mut State, page: usize, dev: Entry) -> io::Result<()> {
        let k = self.const_for(st, dev)?;
        let old = st.pages[page];
        match old {
            PageState::Const(j) | PageState::Lazy(j) if j == k => return Ok(()),
            // Nothing mapped yet: leave it to the first access.
            PageState::Unmapped | PageState::Lazy(_) => st.pages[page] = PageState::Lazy(k),
            // Replace the view in one step, so a reader sees old or new.
            PageState::Const(_) | PageState::Private(_) => {
                unsafe { self.view(page, &st.consts[k as usize].mem, 0)? };
                st.pages[page] = PageState::Const(k);
                self.release(st, old);
            }
        }
        Ok(())
    }

    /// Current contents of `page` as the mutator sees them: what is mapped,
    /// or the default for a page nobody has written.
    fn contents(&self, st: &State, page: usize) -> io::Result<Vec<Entry>> {
        Ok(match st.pages[page] {
            PageState::Unmapped => {
                let Some(k) = st.default else {
                    return Err(io::Error::other(
                        "devmap: a partial write to an untouched page needs a default device",
                    ));
                };
                vec![st.consts[k as usize].dev; self.per_page]
            }
            PageState::Const(k) | PageState::Lazy(k) => vec![st.consts[k as usize].dev; self.per_page],
            PageState::Private(_) => {
                let mut v = Vec::with_capacity(self.per_page);
                // SAFETY: a private page is mapped and readable; only this
                // thread (lock held) replaces views.
                unsafe {
                    let p = self.entries().add(page * self.per_page);
                    v.extend_from_slice(std::slice::from_raw_parts(p, self.per_page));
                }
                v
            }
        })
    }

    /// Set entries `lo..hi` of `page` to `dev`.
    fn modify(&self, st: &mut State, page: usize, lo: usize, hi: usize, dev: Entry) -> io::Result<()> {
        let mut buf = self.contents(st, page)?;
        if buf[lo..hi].iter().all(|&e| same(e, dev)) && st.pages[page] != PageState::Unmapped {
            return Ok(());
        }
        buf[lo..hi].fill(dev);
        let first = buf[0];
        if buf.iter().all(|&e| same(e, first)) {
            return self.map_const(st, page, first);
        }
        let slot = self.alloc_slot(st)?;
        let (obj, idx) = (slot as usize / POOL_PAGES, slot as usize % POOL_PAGES);
        let off = idx * self.page_bytes;
        let r = unsafe {
            self.write_through_staging(st, &st.pool[obj], off, &buf)
                .and_then(|()| self.view(page, &st.pool[obj], off))
        };
        if let Err(e) = r {
            st.free.push(slot);
            return Err(e);
        }
        let old = st.pages[page];
        st.pages[page] = PageState::Private(slot);
        self.release(st, old);
        Ok(())
    }

    /// The fault path. `addr` is the host address that faulted. Returns true
    /// if it was inside this table and the page is now mapped, so the access
    /// can be retried.
    fn fault(&self, addr: usize) -> bool {
        let base = self.table.base() as usize;
        if addr < base || addr >= base + TABLE_BYTES {
            return false;
        }
        let page = (addr - base) / self.page_bytes;
        let slot = ((addr - base) / ENTRY_BYTES) as u64;
        // The lock is only ever held across mapping calls, which do not
        // touch unmapped table pages, so it is never held by the faulting
        // thread; the bound is a backstop against hanging in a handler.
        let Some(_g) = self.try_lock_spinning(50_000_000) else {
            fault_message(b"devmap: fault while the table lock is held; giving up\n", slot << SLOT_SHIFT);
            return false;
        };
        // SAFETY: lock held.
        let st = unsafe { &mut *self.st.get() };
        match st.pages[page] {
            PageState::Unmapped => {}
            PageState::Lazy(k) => {
                // A filled page's first access: no message, it is not a miss.
                if unsafe { self.view(page, &st.consts[k as usize].mem, 0) }.is_err() {
                    fault_message(b"devmap: could not map a filled page at ", slot << SLOT_SHIFT);
                    return false;
                }
                st.pages[page] = PageState::Const(k);
                return true;
            }
            // Another thread mapped it between the fault and the lock.
            PageState::Const(_) | PageState::Private(_) => return true,
        }
        let Some(k) = st.default else {
            fault_message(b"devmap: unmapped physical region and no default device at ", slot << SLOT_SHIFT);
            return false;
        };
        if unsafe { self.view(page, &st.consts[k as usize].mem, 0) }.is_err() {
            fault_message(b"devmap: could not map the default device at ", slot << SLOT_SHIFT);
            return false;
        }
        st.pages[page] = PageState::Const(k);
        drop(_g);
        let n = self.faults.fetch_add(1, Ordering::Relaxed) as usize;
        if n < FIRST_FAULTS {
            self.first_faults[n].store(slot << SLOT_SHIFT, Ordering::Relaxed);
        }
        fault_message(b"iris: physical access to an unmapped region at ", slot << SLOT_SHIFT);
        true
    }
}

impl SparseDeviceMap {
    /// Reserve the table. Nothing is mapped yet; set a default before the
    /// first dispatch.
    pub fn new() -> io::Result<Self> {
        let page_bytes = granularity().max(MIN_PAGE_BYTES);
        let table = AddrSpace::reserve(TABLE_BYTES)?;
        let staging = AddrSpace::reserve(page_bytes)?;
        let npages = TABLE_BYTES / page_bytes;
        let inner = Box::new(Inner {
            table,
            page_bytes,
            per_page: page_bytes / ENTRY_BYTES,
            lock: AtomicBool::new(false),
            st: UnsafeCell::new(State {
                pages: vec![PageState::Unmapped; npages],
                consts: Vec::new(),
                pool: Vec::new(),
                free: Vec::new(),
                staging,
                default: None,
            }),
            faults: AtomicU64::new(0),
            first_faults: [const { AtomicU64::new(u64::MAX) }; FIRST_FAULTS],
            reg: usize::MAX,
        });
        let mut map = Self { inner };
        fault::install();
        map.inner.reg = fault::register(&*map.inner)?;
        Ok(map)
    }

    /// Base of the table: entry `i` is at `base().add(i)`. Valid for the
    /// life of the map; the owner caches it for the hot path.
    #[inline]
    pub fn base(&self) -> *const Entry {
        self.inner.entries()
    }

    /// Entries per table page.
    pub fn page_slots(&self) -> u64 {
        self.inner.per_page as u64
    }

    /// The device an untouched slot answers with. Required before a partial
    /// write to a page nobody has written, and before any dispatch may fault.
    pub fn set_default(&self, dev: Entry) -> io::Result<()> {
        let _g = self.inner.lock();
        let st = unsafe { &mut *self.inner.st.get() };
        let k = self.inner.const_for(st, dev)?;
        st.default = Some(k);
        Ok(())
    }

    /// Point slots `first..first + count` at `dev`.
    pub fn fill(&self, first: u64, count: u64, dev: Entry) -> io::Result<()> {
        let end = first.checked_add(count).filter(|&e| e <= SLOTS).unwrap_or_else(|| {
            panic!("devmap: slots [{first:#x}, +{count:#x}) escape the {PHYS_BITS}-bit space")
        });
        let per = self.inner.per_page as u64;
        let _g = self.inner.lock();
        let st = unsafe { &mut *self.inner.st.get() };
        let mut slot = first;
        while slot < end {
            let page = (slot / per) as usize;
            let page_start = page as u64 * per;
            let hi = end.min(page_start + per);
            if slot == page_start && hi == page_start + per {
                self.inner.map_const(st, page, dev)?;
            } else {
                self.inner.modify(st, page, (slot - page_start) as usize, (hi - page_start) as usize, dev)?;
            }
            slot = hi;
        }
        Ok(())
    }

    /// Point the slots covering guest bytes `start..end` at `dev`. `end` is
    /// exclusive and rounded up to a slot boundary.
    pub fn fill_bytes(&self, start: u64, end: u64, dev: Entry) -> io::Result<()> {
        assert!(end > start, "devmap: empty range {start:#x}..{end:#x}");
        let first = start >> SLOT_SHIFT;
        let last = (end - 1) >> SLOT_SHIFT;
        self.fill(first, last - first + 1, dev)
    }

    /// Whether the page holding `slot` has been written (or faulted in).
    pub fn is_mapped(&self, slot: u64) -> bool {
        let _g = self.inner.lock();
        let st = unsafe { &*self.inner.st.get() };
        st.pages[(slot / self.inner.per_page as u64) as usize] != PageState::Unmapped
    }

    /// The entry at `slot`, without faulting an untouched page in.
    pub fn get(&self, slot: u64) -> Option<Entry> {
        let _g = self.inner.lock();
        let st = unsafe { &*self.inner.st.get() };
        match st.pages[(slot / self.inner.per_page as u64) as usize] {
            PageState::Unmapped => None,
            PageState::Lazy(k) => Some(st.consts[k as usize].dev),
            // SAFETY: mapped pages stay mapped (views are replaced, never
            // removed), and the load is one the hot path does without a lock.
            PageState::Const(_) | PageState::Private(_) => Some(unsafe { *self.inner.entries().add(slot as usize) }),
        }
    }

    /// Whether every slot in `first..first + count` already answers with
    /// `dev`. Untouched pages count as not matching.
    pub fn all_are(&self, first: u64, count: u64, dev: Entry) -> bool {
        let per = self.inner.per_page as u64;
        let end = first + count;
        let _g = self.inner.lock();
        let st = unsafe { &*self.inner.st.get() };
        let mut slot = first;
        while slot < end {
            let page = (slot / per) as usize;
            let hi = end.min((page as u64 + 1) * per);
            match st.pages[page] {
                PageState::Unmapped => return false,
                PageState::Const(k) | PageState::Lazy(k) => {
                    if !same(st.consts[k as usize].dev, dev) {
                        return false;
                    }
                }
                PageState::Private(_) => unsafe {
                    let p = self.inner.entries();
                    for s in slot..hi {
                        if !same(*p.add(s as usize), dev) {
                            return false;
                        }
                    }
                },
            }
            slot = hi;
        }
        true
    }

    /// The table as runs of equal entries, untouched pages included as
    /// `dev: None`.
    pub fn runs(&self) -> Vec<Run> {
        let per = self.inner.per_page as u64;
        let _g = self.inner.lock();
        let st = unsafe { &*self.inner.st.get() };
        let mut out: Vec<Run> = Vec::new();
        let mut push = |start: u64, end: u64, dev: Option<Entry>| {
            if let Some(last) = out.last_mut() {
                let joins = match (last.dev, dev) {
                    (None, None) => true,
                    (Some(a), Some(b)) => same(a, b),
                    _ => false,
                };
                if joins && last.end == start {
                    last.end = end;
                    return;
                }
            }
            out.push(Run { start, end, dev });
        };
        for (page, &state) in st.pages.iter().enumerate() {
            let start = page as u64 * per;
            match state {
                PageState::Unmapped => push(start, start + per, None),
                PageState::Const(k) | PageState::Lazy(k) => push(start, start + per, Some(st.consts[k as usize].dev)),
                PageState::Private(_) => unsafe {
                    let p = self.inner.entries().add(start as usize);
                    let mut run_start = start;
                    let mut cur = *p;
                    for i in 1..per {
                        let e = *p.add(i as usize);
                        if !same(e, cur) {
                            push(run_start, start + i, Some(cur));
                            run_start = start + i;
                            cur = e;
                        }
                    }
                    push(run_start, start + per, Some(cur));
                },
            }
        }
        out
    }

    pub fn stats(&self) -> MapStats {
        let _g = self.inner.lock();
        let st = unsafe { &*self.inner.st.get() };
        let mut s = MapStats {
            page_bytes: self.inner.page_bytes,
            pages: st.pages.len(),
            const_pages: st.consts.len(),
            pool_pages: st.pool.len() * POOL_PAGES,
            pool_free: st.free.len(),
            faults: self.inner.faults.load(Ordering::Relaxed),
            ..Default::default()
        };
        for p in &st.pages {
            match p {
                PageState::Unmapped => s.unmapped += 1,
                PageState::Const(_) => s.shared += 1,
                PageState::Lazy(_) => s.lazy += 1,
                PageState::Private(_) => s.private += 1,
            }
        }
        s.first_faults = self
            .inner
            .first_faults
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .take_while(|&a| a != u64::MAX)
            .collect();
        s
    }
}

impl Drop for SparseDeviceMap {
    fn drop(&mut self) {
        fault::unregister(self.inner.reg);
    }
}

/// Print `what` followed by the guest address, from a fault handler: a fixed
/// buffer and one raw write, no allocation and no locks.
fn fault_message(what: &[u8], phys: u64) {
    let mut buf = [0u8; 160];
    let mut n = 0;
    let mut push = |b: &[u8]| {
        let take = b.len().min(buf.len() - n);
        buf[n..n + take].copy_from_slice(&b[..take]);
        n += take;
    };
    push(what);
    push(b"0x");
    let digits = (PHYS_BITS as usize).div_ceil(4);
    let mut hex = [0u8; 16];
    for (i, d) in hex.iter_mut().take(digits).enumerate() {
        let nib = (phys >> (4 * (digits - 1 - i))) & 0xF;
        *d = b"0123456789abcdef"[nib as usize];
    }
    push(&hex[..digits]);
    push(b" (64 KB slot now answers as the bus-error device)\n");
    fault::emit(&buf[..n]);
}

/// Process-wide: the host fault handler and the registry of live tables it
/// consults.
mod fault {
    use super::Inner;
    use std::io;
    use std::sync::atomic::{AtomicPtr, Ordering};
    use std::sync::Once;

    // Generous: a test binary builds many machines at once, each with a table.
    const MAX_MAPS: usize = 64;
    static REGISTRY: [AtomicPtr<Inner>; MAX_MAPS] = [const { AtomicPtr::new(std::ptr::null_mut()) }; MAX_MAPS];
    static INSTALL: Once = Once::new();

    pub(super) fn register(inner: *const Inner) -> io::Result<usize> {
        for (i, slot) in REGISTRY.iter().enumerate() {
            if slot
                .compare_exchange(std::ptr::null_mut(), inner as *mut Inner, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(i);
            }
        }
        Err(io::Error::other(format!("devmap: more than {MAX_MAPS} tables alive")))
    }

    pub(super) fn unregister(i: usize) {
        if i < MAX_MAPS {
            REGISTRY[i].store(std::ptr::null_mut(), Ordering::Release);
        }
    }

    /// Offer a faulting host address to every live table.
    pub(super) fn dispatch(addr: usize) -> bool {
        REGISTRY.iter().any(|slot| {
            let p = slot.load(Ordering::Acquire);
            // SAFETY: a registered table stays alive until it unregisters in
            // `Drop`, and nothing dispatches through a table being dropped.
            !p.is_null() && unsafe { (*p).fault(addr) }
        })
    }

    pub(super) fn install() {
        INSTALL.call_once(imp::install);
    }

    pub(super) fn emit(bytes: &[u8]) {
        imp::emit(bytes);
    }

    #[cfg(unix)]
    mod imp {
        //! `SIGSEGV` (Linux, and macOS for unmapped pages) and `SIGBUS`
        //! (macOS for `PROT_NONE` pages). Whatever handler was installed
        //! before — Rust's own stack-overflow detector — is chained to for
        //! every fault that is not ours, so nothing else changes.
        use std::cell::UnsafeCell;

        struct Prev(UnsafeCell<[libc::sigaction; 2]>);
        // SAFETY: written once under `INSTALL`, read only from the handler.
        unsafe impl Sync for Prev {}
        static PREV: Prev = Prev(UnsafeCell::new([unsafe { std::mem::zeroed() }; 2]));

        const SIGNALS: [libc::c_int; 2] = [libc::SIGSEGV, libc::SIGBUS];

        unsafe extern "C" fn handler(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
            if !info.is_null() {
                let addr = unsafe { (*info).si_addr() } as usize;
                if super::dispatch(addr) {
                    return; // the load re-executes against the new page
                }
            }
            let idx = SIGNALS.iter().position(|&s| s == sig).unwrap_or(0);
            let prev = unsafe { (*PREV.0.get())[idx] };
            if prev.sa_flags & libc::SA_SIGINFO != 0 {
                let f: unsafe extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                    unsafe { std::mem::transmute(prev.sa_sigaction) };
                unsafe { f(sig, info, ctx) };
            } else if prev.sa_sigaction == libc::SIG_DFL || prev.sa_sigaction == libc::SIG_IGN {
                // Restore the default and return: the access repeats and the
                // default action (a core dump) takes it from there.
                unsafe { libc::sigaction(sig, &prev, std::ptr::null_mut()) };
            } else {
                let f: unsafe extern "C" fn(libc::c_int) = unsafe { std::mem::transmute(prev.sa_sigaction) };
                unsafe { f(sig) };
            }
        }

        pub(super) fn install() {
            unsafe {
                let mut sa: libc::sigaction = std::mem::zeroed();
                // Through `*const ()`: see `hwwatch.rs` on why not a direct cast.
                sa.sa_sigaction = handler as *const () as usize;
                // ONSTACK so a thread with an alternate stack keeps using it,
                // as Rust's handler expects; NODEFER so a fault inside the
                // chained handler is not silently blocked.
                sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_NODEFER;
                libc::sigemptyset(&mut sa.sa_mask);
                for (i, &sig) in SIGNALS.iter().enumerate() {
                    libc::sigaction(sig, &sa, &mut (*PREV.0.get())[i]);
                }
            }
        }

        pub(super) fn emit(bytes: &[u8]) {
            unsafe { libc::write(2, bytes.as_ptr() as *const libc::c_void, bytes.len()) };
        }
    }

    #[cfg(windows)]
    mod imp {
        //! A vectored exception handler, added at the head of the chain.
        //! `crash_diag` adds its own at the head earlier in `main`, so this
        //! one runs first and a handled fault never reaches its log; every
        //! other exception continues down the chain untouched.
        use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};

        const EXCEPTION_ACCESS_VIOLATION: i32 = 0xC000_0005u32 as i32;
        const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
        const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

        unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
            if info.is_null() {
                return EXCEPTION_CONTINUE_SEARCH;
            }
            let rec = unsafe { (*info).ExceptionRecord };
            if rec.is_null() {
                return EXCEPTION_CONTINUE_SEARCH;
            }
            let rec = unsafe { &*rec };
            if rec.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && rec.NumberParameters >= 2 {
                // ExceptionInformation[0] is the access kind, [1] the address.
                let addr = rec.ExceptionInformation[1];
                if super::dispatch(addr) {
                    return EXCEPTION_CONTINUE_EXECUTION;
                }
            }
            EXCEPTION_CONTINUE_SEARCH
        }

        pub(super) fn install() {
            unsafe {
                let _ = AddVectoredExceptionHandler(1, Some(handler));
            }
        }

        pub(super) fn emit(bytes: &[u8]) {
            use std::io::Write;
            let _ = std::io::stderr().write_all(bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device with a body, so two of them never share an address.
    struct Dummy(#[allow(dead_code)] u64);
    impl BusDevice for Dummy {}

    fn dev(tag: u64) -> Entry {
        Box::leak(Box::new(Dummy(tag))) as *const Dummy as Entry
    }

    fn entry_at(m: &SparseDeviceMap, slot: u64) -> Entry {
        unsafe { std::ptr::read_volatile(m.base().add(slot as usize)) }
    }

    #[test]
    fn a_uniform_fill_shares_one_constant_page() {
        let m = SparseDeviceMap::new().unwrap();
        let (d0, d1) = (dev(0), dev(1));
        m.set_default(d0).unwrap();
        let per = m.page_slots();
        m.fill(0, 4 * per, d1).unwrap();
        let s = m.stats();
        assert_eq!((s.lazy, s.shared, s.private, s.const_pages, s.pool_pages), (4, 0, 0, 2, 0), "nothing mapped yet");
        for slot in [0, per - 1, per, 3 * per + 7, 4 * per - 1] {
            assert!(same(entry_at(&m, slot), d1), "slot {slot:#x}");
        }
        let s = m.stats();
        assert_eq!((s.lazy, s.shared, s.faults), (1, 3, 0), "the pages read were mapped, silently; page 2 waits");
        assert!(same(m.get(2 * per + 5).unwrap(), d1), "get answers a filled page without mapping it");
        assert!(!m.is_mapped(4 * per));
        // A second identical fill maps nothing new.
        m.fill(0, 4 * per, d1).unwrap();
        assert_eq!(m.stats().const_pages, 2);
    }

    #[test]
    fn a_partial_fill_makes_a_private_page_and_uniformity_frees_it() {
        let m = SparseDeviceMap::new().unwrap();
        let (d0, d1, d2) = (dev(0), dev(1), dev(2));
        m.set_default(d0).unwrap();
        let per = m.page_slots();
        m.fill(0, per, d1).unwrap();
        m.fill(5, 4, d2).unwrap();
        let s = m.stats();
        assert_eq!((s.shared, s.private, s.pool_pages, s.pool_free), (0, 1, POOL_PAGES, POOL_PAGES - 1));
        assert!(same(entry_at(&m, 4), d1));
        assert!(same(entry_at(&m, 5), d2));
        assert!(same(entry_at(&m, 8), d2));
        assert!(same(entry_at(&m, 9), d1));
        assert!(m.all_are(5, 4, d2));
        assert!(!m.all_are(4, 4, d2));
        // Changing one entry of a private page swaps in a fresh pool page.
        m.fill(7, 1, d1).unwrap();
        assert!(same(entry_at(&m, 7), d1));
        assert_eq!(m.stats().pool_free, POOL_PAGES - 1, "the old pool page was released");
        // Back to uniform: the page becomes a constant view again.
        m.fill(5, 4, d1).unwrap();
        let s = m.stats();
        assert_eq!((s.shared, s.private, s.pool_free), (1, 0, POOL_PAGES));
        assert!(same(entry_at(&m, 6), d1));
    }

    #[test]
    fn a_partial_fill_of_an_untouched_page_is_padded_with_the_default() {
        let m = SparseDeviceMap::new().unwrap();
        let (d0, d1) = (dev(0), dev(1));
        assert!(m.fill(10, 2, d1).is_err(), "no default yet");
        m.set_default(d0).unwrap();
        m.fill(10, 2, d1).unwrap();
        let per = m.page_slots();
        assert!(same(entry_at(&m, 0), d0));
        assert!(same(entry_at(&m, 9), d0));
        assert!(same(entry_at(&m, 10), d1));
        assert!(same(entry_at(&m, 11), d1));
        assert!(same(entry_at(&m, per - 1), d0));
        assert!(!m.is_mapped(per));
        assert_eq!(m.get(per), None);
        assert!(same(m.get(10).unwrap(), d1));
    }

    #[test]
    fn a_load_from_an_untouched_page_faults_in_the_default() {
        let m = SparseDeviceMap::new().unwrap();
        let (d0, d1) = (dev(0), dev(1));
        m.set_default(d0).unwrap();
        let per = m.page_slots();
        m.fill(0, per, d1).unwrap();
        // Well above anything written, as an Octane probe above 4 GB would be.
        let slot = (1u64 << (PHYS_BITS - SLOT_SHIFT - 1)) + 3 * per + 17;
        assert!(!m.is_mapped(slot));
        let e = entry_at(&m, slot);
        assert!(same(e, d0), "the faulted-in page answers with the default");
        assert!(m.is_mapped(slot));
        let s = m.stats();
        assert_eq!(s.faults, 1);
        assert_eq!(s.first_faults, vec![slot << SLOT_SHIFT]);
        // The page is a shared view of the default's constant page; the
        // filled page was never read, so it is still waiting.
        assert_eq!((s.shared, s.lazy, s.private), (1, 1, 0));
        // Loading it again is an ordinary load.
        assert!(same(entry_at(&m, slot + 1), d0));
        assert_eq!(m.stats().faults, 1);
    }

    #[test]
    fn runs_coalesce_across_pages_and_report_gaps() {
        let m = SparseDeviceMap::new().unwrap();
        let (d0, d1, d2) = (dev(0), dev(1), dev(2));
        m.set_default(d0).unwrap();
        let per = m.page_slots();
        m.fill(0, 2 * per, d1).unwrap();
        m.fill(2 * per - 3, 5, d2).unwrap(); // straddles pages 1 and 2
        m.fill(0x7c0, 0x40, d0).unwrap(); // 64 slots inside page 0
        let runs = m.runs();
        let mapped: Vec<_> = runs.iter().filter(|r| r.dev.is_some()).collect();
        let starts: Vec<(u64, u64)> = mapped.iter().map(|r| (r.start, r.end)).collect();
        assert_eq!(
            starts,
            vec![
                (0, 0x7c0),
                (0x7c0, 0x800),
                (0x800, 2 * per - 3),
                (2 * per - 3, 2 * per + 2),
                (2 * per + 2, 3 * per),
            ]
        );
        assert!(same(mapped[1].dev.unwrap(), d0));
        assert!(same(mapped[3].dev.unwrap(), d2));
        assert!(same(mapped[4].dev.unwrap(), d0), "rest of page 2 is the default");
        // Everything after page 2 is one untouched run to the end.
        let last = runs.last().unwrap();
        assert!(last.dev.is_none());
        assert_eq!((last.start, last.end), (3 * per, SLOTS));
    }

    #[test]
    fn same_tells_zero_sized_devices_apart() {
        struct A;
        struct B;
        impl BusDevice for A {}
        impl BusDevice for B {}
        #[repr(C)]
        struct Both(A, B);
        let both = Both(A, B);
        let a: Entry = &both.0;
        let b: Entry = &both.1;
        assert!(std::ptr::addr_eq(a, b), "the test needs them to share an address");
        assert!(!same(a, b));
        assert!(same(a, a));
    }
}
