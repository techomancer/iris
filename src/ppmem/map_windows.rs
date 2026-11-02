//! Windows backend for ppmem's virtual-memory primitives — see
//! `docs/ppmem-design.md` §7.3.
//!
//! Placeholders keep the address range reserved throughout a replacement, but
//! UnmapViewOfFile2 and MapViewOfFile3 are separate calls. The handler below
//! retries data accesses caught between them. `scrub` leaves demand-committed
//! zero pages, matching the Unix backend. This provides address accessibility;
//! guest memory ordering and JIT invalidation remain the caller's responsibility.
//!
//! ## Why the placeholder API and not plain VirtualAlloc
//!
//! The intuitive approach — `VirtualAlloc(MEM_RESERVE)` the whole range, then
//! map views inside it — does not work. `MapViewOfFileEx` fails with
//! `ERROR_INVALID_ADDRESS` if any part of the target is already reserved, and
//! a plain `MEM_RESERVE` region counts as reserved. Nor can the sub-range be
//! freed first: `MEM_RELEASE` only accepts an entire original allocation, so
//! carving 8MB out of a 4GB reservation is impossible.
//!
//! The placeholder API (Win10 1803+) exists for exactly this. It is the direct
//! analogue of the Unix sequence:
//!
//! | | Unix | Windows |
//! |---|---|---|
//! | reserve | `mmap(PROT_NONE)` | `VirtualAlloc2(MEM_RESERVE_PLACEHOLDER)` |
//! | map | `mmap(MAP_SHARED\|MAP_FIXED)` | split, then `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)` |
//! | unmap | `mmap(PROT_NONE\|MAP_FIXED)` | `UnmapViewOfFile2(MEM_PRESERVE_PLACEHOLDER)` |
//!
//! `MEM_PRESERVE_PLACEHOLDER` is the load-bearing flag: without it the address
//! becomes free the instant a view is unmapped and another thread can take it,
//! which is precisely why the Unix side re-maps `PROT_NONE` instead of calling
//! `munmap`.

use super::{oserr, Prot};
use std::io;
use std::sync::{Arc, Mutex};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile3, UnmapViewOfFile2, VirtualAlloc2, VirtualFree,
    VirtualQuery, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
    MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS,
    PAGE_READONLY, PAGE_READWRITE, SEC_RESERVE,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

// windows-sys exposes this SDK flag under SystemServices, which otherwise
// is not needed by this backend.
const MEM_COALESCE_PLACEHOLDERS: u32 = 0x0000_0001;

/// The current-process pseudo-handle. Not a real handle — never closed.
fn current_process() -> HANDLE {
    unsafe { GetCurrentProcess() }
}

/// An unnamed pagefile-backed section, shared by its views. Retaining the
/// handle lets a partial replacement re-map the untouched parts of a view.
#[derive(Clone)]
pub struct SharedMem {
    handle: Arc<Section>,
    len: usize,
}

struct Section(HANDLE);
unsafe impl Send for Section {}
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        unsafe { CloseHandle(self.0) };
    }
}

impl SharedMem {
    pub fn new(len: usize) -> io::Result<Self> {
        Self::create(len, 0)
    }

    fn create(len: usize, flags: u32) -> io::Result<Self> {
        assert!(len > 0, "ppmem: SharedMem::new with zero length");
        assert!(
            super::is_aligned(len),
            "ppmem: SharedMem length {len:#x} is not granularity-aligned"
        );

        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE, // pagefile-backed, not a real file
                std::ptr::null(),
                PAGE_READWRITE | flags,
                (len >> 32) as u32,
                (len & 0xFFFF_FFFF) as u32,
                std::ptr::null(), // unnamed
            )
        };
        if handle.is_null() {
            return Err(oserr("CreateFileMappingW"));
        }
        Ok(Self { handle: Arc::new(Section(handle)), len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Windows has no section-level hole punch; callers zero the bank instead.
    pub fn discard(&self) -> bool {
        false
    }
}

#[derive(Clone)]
struct View {
    at: usize,
    len: usize,
    mem: SharedMem,
    offset: u64,
    prot: Prot,
}

impl View {
    fn end(&self) -> usize { self.at + self.len }

    fn slice(&self, at: usize, end: usize) -> Self {
        Self { at, len: end - at, offset: self.offset + (at - self.at) as u64, ..self.clone() }
    }
}

/// A reservation kept from creation to drop. `views` records actual Windows
/// view boundaries, which need not match a caller's logical bank/mirror range.
pub struct AddrSpace {
    base: *mut u8,
    size: usize,
    views: Mutex<Vec<View>>,
}

unsafe impl Send for AddrSpace {}
unsafe impl Sync for AddrSpace {}

impl AddrSpace {
    /// Reserve `size` bytes as a single placeholder.
    pub fn reserve(size: usize) -> io::Result<Self> {
        assert!(size > 0, "ppmem: AddrSpace::reserve with zero size");
        assert!(
            super::is_aligned(size),
            "ppmem: reservation size {size:#x} is not granularity-aligned"
        );
        fault::install()?;

        let p = unsafe {
            VirtualAlloc2(
                current_process(),
                std::ptr::null(),
                size,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        };
        if p.is_null() {
            return Err(oserr("VirtualAlloc2(MEM_RESERVE_PLACEHOLDER)"));
        }
        fault::lock().spaces.push((p as usize, size));
        Ok(Self {
            base: p.cast(),
            size,
            views: Mutex::new(Vec::new()),
        })
    }

    pub fn base(&self) -> *mut u8 {
        self.base
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn check_range(&self, at: usize, len: usize) {
        assert!(len > 0, "ppmem: range has zero length");
        assert!(super::is_aligned(at), "ppmem: offset {at:#x} misaligned");
        assert!(super::is_aligned(len), "ppmem: length {len:#x} misaligned");
        assert!(
            at.checked_add(len).is_some_and(|end| end <= self.size),
            "ppmem: [{at:#x}, +{len:#x}) escapes the {:#x}-byte reservation",
            self.size
        );
    }

    /// Carve exactly this range from one or more adjacent placeholders.
    /// Called with the backend lock held and no views in the requested range.
    unsafe fn placeholder(&self, at: usize, len: usize) -> io::Result<()> {
        let start = self.base as usize + at;
        let end = start + len;
        let mut pos = start;
        while pos < end {
            let info = query(pos)?;
            let region_start = info.AllocationBase as usize;
            let region_end = info.BaseAddress as usize + info.RegionSize;
            if info.State != MEM_RESERVE || region_end <= pos {
                return Err(io::Error::other("ppmem: expected a placeholder"));
            }
            if region_start < pos && unsafe {
                VirtualFree(region_start as *mut _, pos - region_start, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)
            } == 0 {
                return Err(oserr("VirtualFree(split prefix)"));
            }
            let next = region_end.min(end);
            if next < region_end && unsafe {
                VirtualFree(pos as *mut _, next - pos, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)
            } == 0 {
                return Err(oserr("VirtualFree(split suffix)"));
            }
            pos = next;
        }
        if query(start)?.RegionSize != len && unsafe {
            VirtualFree(start as *mut _, len, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)
        } == 0 {
            return Err(oserr("VirtualFree(coalesce placeholders)"));
        }
        Ok(())
    }

    unsafe fn map_view(&self, view: &View) -> io::Result<()> {
        unsafe { self.placeholder(view.at, view.len)? };
        #[cfg(test)]
        if FAIL_MAP_AFTER.with(|fail| match fail.get() {
            None => false,
            Some(0) => { fail.set(None); true }
            Some(n) => { fail.set(Some(n - 1)); false }
        }) {
            return Err(io::Error::other("injected mapping failure"));
        }
        let want = unsafe { self.base.add(view.at) };
        let prot = match view.prot { Prot::ReadWrite => PAGE_READWRITE, Prot::ReadOnly => PAGE_READONLY };
        let got = unsafe {
            MapViewOfFile3(view.mem.handle.0, current_process(), want.cast(), view.offset,
                view.len, MEM_REPLACE_PLACEHOLDER, prot, std::ptr::null_mut(), 0)
        };
        if got.Value.is_null() {
            return Err(oserr("MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)"));
        }
        assert_eq!(got.Value, want.cast(), "ppmem: misplaced view");
        Ok(())
    }

    unsafe fn unmap_view(&self, at: usize) -> io::Result<()> {
        if unsafe {
            UnmapViewOfFile2(current_process(), MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.base.add(at).cast(),
            }, MEM_PRESERVE_PLACEHOLDER)
        } == 0 {
            return Err(oserr("UnmapViewOfFile2(MEM_PRESERVE_PLACEHOLDER)"));
        }
        Ok(())
    }

    /// Replace whole Windows views, restoring untouched prefixes/suffixes.
    /// No device callback or access through the window occurs under this lock.
    unsafe fn replace(&self, at: usize, len: usize, new: Option<View>) -> io::Result<()> {
        self.check_range(at, len);
        let mut state = fault::lock();
        let mut views = self.views.lock().unwrap();
        let end = at + len;
        let old: Vec<_> = views.iter().filter(|v| v.at < end && at < v.end()).cloned().collect();
        let mut next = Vec::with_capacity(3);
        for v in &old {
            if v.at < at { next.push(v.slice(v.at, at)); }
            if v.end() > end { next.push(v.slice(end, v.end())); }
        }
        next.extend(new);
        next.sort_unstable_by_key(|v| v.at);
        // Finish all allocations before taking any view away.
        views.reserve(next.len());
        state.version = state.version.wrapping_add(1);
        let (mut removed, mut installed) = (0, 0);
        let result = (|| {
            for v in &old {
                unsafe { self.unmap_view(v.at)? };
                removed += 1;
            }
            for v in &next {
                unsafe { self.map_view(v)? };
                installed += 1;
            }
            Ok(())
        })();
        if let Err(error) = result {
            // clear_mappings ignores a scrub error. Restore the old views so
            // a failed replacement cannot strand its readers on placeholders.
            for v in &next[..installed] {
                unsafe { self.unmap_view(v.at) }.expect("ppmem: rollback unmap failed");
            }
            for v in &old[..removed] {
                unsafe { self.map_view(v) }.expect("ppmem: rollback map failed");
            }
            return Err(error);
        }
        views.retain(|v| v.at >= end || v.end() <= at);
        views.extend(next);
        Ok(())
    }

    /// Map a shared section at a fixed address. Concurrent window accesses can
    /// see old or new storage; the backend retries the intervening host fault.
    ///
    /// # Safety
    /// The caller owns the range and is responsible for guest synchronisation.
    pub unsafe fn map(&self, at: usize, len: usize, mem: &SharedMem, offset: u64, prot: Prot) -> io::Result<()> {
        self.check_range(at, len);
        assert!(super::is_aligned(offset as usize));
        assert!(offset.checked_add(len as u64).is_some_and(|end| end <= mem.len() as u64));
        unsafe { self.replace(at, len, Some(View { at, len, mem: mem.clone(), offset, prot })) }
    }

    /// Revert a range to inaccessible placeholders without releasing it.
    ///
    /// # Safety
    /// The caller must ensure nobody continues to access the removed range.
    pub unsafe fn unmap(&self, at: usize, len: usize) -> io::Result<()> {
        unsafe { self.replace(at, len, None) }
    }

    /// Replace a range with independent read-write zero pages. SEC_RESERVE
    /// charges only pages the handler commits on access, not the whole bank.
    ///
    /// # Safety
    /// As `map`: old pointers now refer to different storage.
    pub unsafe fn scrub(&self, at: usize, len: usize) -> io::Result<()> {
        self.check_range(at, len);
        let zero = SharedMem::create(len, SEC_RESERVE)?;
        unsafe { self.map(at, len, &zero, 0, Prot::ReadWrite) }
    }
}

impl Drop for AddrSpace {
    fn drop(&mut self) {
        let mut state = fault::lock();
        state.spaces.retain(|&(base, _)| base != self.base as usize);
        state.version = state.version.wrapping_add(1);
        for view in self.views.get_mut().unwrap().iter() {
            unsafe {
                UnmapViewOfFile2(current_process(), MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.base.add(view.at).cast(),
                }, MEM_PRESERVE_PLACEHOLDER);
            }
        }
        unsafe {
            // Split placeholders are separate reservations: coalesce before
            // releasing, otherwise VirtualFree(base, 0) frees only the first.
            let _ = self.placeholder(0, self.size);
            VirtualFree(self.base.cast(), 0, MEM_RELEASE);
        }
    }
}

fn query(addr: usize) -> io::Result<MEMORY_BASIC_INFORMATION> {
    let mut info = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQuery(addr as *const _, &mut info, std::mem::size_of_val(&info)) } == 0 {
        return Err(oserr("VirtualQuery"));
    }
    Ok(info)
}

/// Numeric ranges only: an exception can never retain a pointer to a dropped
/// AddrSpace. Mutation, membership checks and teardown use the same lock.
mod fault {
    use super::*;
    use std::cell::Cell;
    use std::sync::{MutexGuard, OnceLock};
    use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
    use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, MEM_MAPPED};

    pub(super) struct State {
        pub spaces: Vec<(usize, usize)>,
        pub version: u64,
    }
    static STATE: Mutex<State> = Mutex::new(State { spaces: Vec::new(), version: 0 });
    static INSTALL: OnceLock<Result<(), i32>> = OnceLock::new();
    thread_local! {
        static LAST_RETRY: Cell<(usize, usize, u64)> = const { Cell::new((0, 0, 0)) };
    }

    pub(super) fn lock() -> MutexGuard<'static, State> { STATE.lock().unwrap() }

    pub(super) fn install() -> io::Result<()> {
        match *INSTALL.get_or_init(|| {
            if unsafe { AddVectoredExceptionHandler(1, Some(handler)) }.is_null() {
                Err(io::Error::last_os_error().raw_os_error().unwrap_or(1))
            } else { Ok(()) }
        }) {
            Ok(()) => Ok(()),
            Err(code) => Err(io::Error::from_raw_os_error(code)),
        }
    }

    fn retry(addr: usize, ip: usize, write: bool) -> bool {
        // No allocation, logging, device callbacks or blocking lock acquisition
        // in the handler. A bound also prevents a recursive fault hanging it.
        let mut guard = None;
        for _ in 0..50_000_000 {
            match STATE.try_lock() {
                Ok(g) => { guard = Some(g); break; }
                Err(std::sync::TryLockError::Poisoned(_)) => return false,
                Err(std::sync::TryLockError::WouldBlock) => std::hint::spin_loop(),
            }
        }
        let Some(state) = guard else { return false };
        if !state.spaces.iter().any(|&(base, size)| addr >= base && addr - base < size) {
            return false;
        }
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(addr as *const _, &mut info, std::mem::size_of_val(&info)) } == 0 {
            return false;
        }
        let ready = if info.State == MEM_RESERVE && info.Type == MEM_MAPPED
            && info.AllocationProtect == PAGE_READWRITE
        {
            // Only scrub's SEC_RESERVE views have this state. Never commit an
            // untouched placeholder or a deliberately unmapped address.
            !unsafe { VirtualAlloc(addr as *const _, 1, MEM_COMMIT, PAGE_READWRITE) }.is_null()
        } else {
            info.State == MEM_COMMIT
                && (info.Protect == PAGE_READWRITE || (!write && info.Protect == PAGE_READONLY))
        };
        if !ready { return false; }
        // The fault may arrive after replacement has finished. Permit that
        // retry, but not an endless loop at the same instruction with no map
        // progress. ExceptionAddress works on both x64 and ARM64.
        first_retry((ip, addr, state.version))
    }

    pub(super) fn first_retry(cookie: (usize, usize, u64)) -> bool {
        LAST_RETRY.with(|last| last.replace(cookie) != cookie)
    }

    unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
        if info.is_null() { return 0; }
        let rec = unsafe { (*info).ExceptionRecord };
        if rec.is_null() { return 0; }
        let rec = unsafe { &*rec };
        if rec.ExceptionCode != 0xC000_0005u32 as i32 || rec.ExceptionFlags & 1 != 0
            || rec.NumberParameters < 2 || rec.ExceptionInformation[0] > 1
        {
            return 0;
        }
        let addr = rec.ExceptionInformation[1];
        #[cfg(test)]
        if WATCH_ADDR.load(std::sync::atomic::Ordering::Relaxed) == addr {
            FAULT_SEEN.store(true, std::sync::atomic::Ordering::Release);
        }
        if retry(addr, rec.ExceptionAddress as usize, rec.ExceptionInformation[0] == 1) { -1 } else { 0 }
    }

    #[cfg(test)]
    pub(super) static WATCH_ADDR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    #[cfg(test)]
    pub(super) static FAULT_SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    #[cfg(test)]
    pub(super) fn test_retry(addr: usize, ip: usize, write: bool) -> bool { retry(addr, ip, write) }
}

pub fn granularity() -> usize {
    // Windows placeholders split only on allocation-granularity boundaries
    // (64KB), and views must be aligned to it in both address and section
    // offset — so this is the number that matters here, not the 4KB page size.
    unsafe {
        let mut si: SYSTEM_INFO = std::mem::zeroed();
        GetSystemInfo(&mut si);
        let g = si.dwAllocationGranularity as usize;
        if g == 0 {
            64 * 1024
        } else {
            g
        }
    }
}

#[cfg(test)]
thread_local! {
    static FAIL_MAP_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    #[test]
    fn scrub_zeroes_detaches_and_can_be_remapped() {
        let g = granularity();
        let mem = SharedMem::new(2 * g).unwrap();
        let space = AddrSpace::reserve(4 * g).unwrap();
        unsafe {
            space.map(0, 2 * g, &mem, 0, Prot::ReadWrite).unwrap();
            space.map(2 * g, 2 * g, &mem, 0, Prot::ReadWrite).unwrap();
            *space.base() = 17;
            *space.base().add(2 * g - 1) = 29;
            space.scrub(0, 2 * g).unwrap();
            assert_eq!(query(space.base() as usize).unwrap().State, MEM_RESERVE);
            assert_eq!(*space.base(), 0);
            assert_eq!(*space.base().add(2 * g - 1), 0);
            assert_eq!(*space.base().add(2 * g), 17);
            assert_eq!(*space.base().add(4 * g - 1), 29);
            let gen = &*space.base().add(g).cast::<AtomicU64>();
            assert_eq!(gen.fetch_add(7, Ordering::Relaxed), 0);
            assert_eq!(gen.load(Ordering::Relaxed), 7);
            assert_eq!(*space.base().add(3 * g).cast::<u64>(), 0);
            space.scrub(0, 2 * g).unwrap();
            assert_eq!((*space.base().add(g).cast::<AtomicU64>()).load(Ordering::Relaxed), 0);
            space.map(0, 2 * g, &mem, 0, Prot::ReadWrite).unwrap();
            assert_eq!(*space.base(), 17);
        }
    }

    #[test]
    fn first_view_can_start_in_the_middle_of_a_reservation() {
        let g = granularity();
        let mem = SharedMem::new(g).unwrap();
        let space = AddrSpace::reserve(8 * g).unwrap();
        unsafe {
            space.map(3 * g, g, &mem, 0, Prot::ReadWrite).unwrap();
            *space.base().add(3 * g) = 45;
            space.map(0, g, &mem, 0, Prot::ReadWrite).unwrap();
            space.map(7 * g, g, &mem, 0, Prot::ReadWrite).unwrap();
            assert_eq!(*space.base(), 45);
            assert_eq!(*space.base().add(7 * g), 45);
        }
    }

    #[test]
    fn replacements_preserve_tails_and_merge_mirrors() {
        let g = granularity();
        let a = SharedMem::new(4 * g).unwrap();
        let b = SharedMem::new(g).unwrap();
        let space = AddrSpace::reserve(8 * g).unwrap();
        unsafe {
            space.map(0, 4 * g, &a, 0, Prot::ReadWrite).unwrap();
            *space.base() = 11;
            *space.base().add(3 * g) = 33;
            drop(a); // The remaining view must keep its section handle alive.
            space.map(g, g, &b, 0, Prot::ReadWrite).unwrap();
            assert_eq!(*space.base(), 11);
            assert_eq!(*space.base().add(3 * g), 33);
            *space.base().add(g) = 22;
            for i in 4..8 { space.map(i * g, g, &b, 0, Prot::ReadWrite).unwrap(); }
            space.scrub(4 * g, 4 * g).unwrap();
            assert_eq!(*space.base().add(4 * g), 0);
            assert_eq!(*space.base().add(8 * g - 1), 0);
            *space.base().add(7 * g) = 77;
            space.map(5 * g, g, &b, 0, Prot::ReadWrite).unwrap();
            assert_eq!(*space.base().add(5 * g), 22);
            assert_eq!(*space.base().add(7 * g), 77);
            space.unmap(g, 2 * g).unwrap();
            assert_eq!(*space.base(), 11);
            assert_eq!(*space.base().add(3 * g), 33);
            assert!(!fault::test_retry(space.base() as usize + g, 1, false));
        }
    }

    #[test]
    fn failed_replacement_restores_old_views() {
        let g = granularity();
        let mem = SharedMem::new(g).unwrap();
        let space = AddrSpace::reserve(2 * g).unwrap();
        unsafe {
            for i in 0..2 { space.map(i * g, g, &mem, 0, Prot::ReadWrite).unwrap(); }
            *space.base() = 91;
            FAIL_MAP_AFTER.with(|f| f.set(Some(0)));
            assert!(space.scrub(0, 2 * g).is_err());
            assert_eq!(*space.base(), 91);
            assert_eq!(*space.base().add(g), 91);
            space.scrub(0, 2 * g).unwrap();
            assert_eq!(*space.base(), 0);
            *space.base() = 42;
            *space.base().add(g) = 67;
            FAIL_MAP_AFTER.with(|f| f.set(Some(1)));
            assert!(space.map(g, g, &mem, 0, Prot::ReadWrite).is_err());
            assert_eq!(*space.base(), 42);
            assert_eq!(*space.base().add(g), 67);
        }
    }

    #[test]
    fn a_reader_retries_a_real_gap_once() {
        let g = granularity();
        let mem = SharedMem::new(g).unwrap();
        let space = AddrSpace::reserve(g).unwrap();
        unsafe { space.map(0, g, &mem, 0, Prot::ReadWrite).unwrap() };
        let addr = space.base() as usize;
        unsafe { (*(addr as *const AtomicU64)).store(40, Ordering::Relaxed) };
        fault::WATCH_ADDR.store(addr, Ordering::Relaxed);
        fault::FAULT_SEEN.store(false, Ordering::Relaxed);
        let mut state = fault::lock();
        state.version = state.version.wrapping_add(1);
        unsafe { space.unmap_view(0).unwrap() };
        let reader = std::thread::spawn(move || unsafe {
            (*(addr as *const AtomicU64)).fetch_add(1, Ordering::Relaxed)
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !fault::FAULT_SEEN.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        let seen = fault::FAULT_SEEN.load(Ordering::Acquire);
        let view = space.views.lock().unwrap()[0].clone();
        unsafe { space.map_view(&view).unwrap() };
        drop(state);
        assert_eq!(reader.join().unwrap(), 40);
        assert!(seen, "reader must fault inside the deliberately open gap");
        assert_eq!(unsafe { (*(addr as *const AtomicU64)).load(Ordering::Relaxed) }, 41);
        fault::WATCH_ADDR.store(0, Ordering::Relaxed);
        // The exception can also reach its handler after the map is complete.
        assert!(fault::test_retry(addr, 123, false));
    }

    #[test]
    fn an_unchanged_fault_is_not_retried_forever() {
        assert!(fault::first_retry((123, 456, 789)));
        assert!(!fault::first_retry((123, 456, 789)));
        assert!(fault::first_retry((123, 456, 790)));
    }

    #[test]
    fn faults_outside_readable_views_are_not_handled() {
        let g = granularity();
        let space = AddrSpace::reserve(2 * g).unwrap();
        let mem = SharedMem::new(g).unwrap();
        let addr = space.base() as usize;
        assert!(!fault::test_retry(addr, 1, false));
        assert!(!fault::test_retry(1, 1, false));
        unsafe { space.map(0, g, &mem, 0, Prot::ReadOnly).unwrap() };
        assert!(!fault::test_retry(addr, 1, true));
        assert!(!fault::test_retry(addr + g, 1, false));
    }

    #[test]
    fn concurrent_data_and_generation_access_survive_remaps() {
        let g = granularity();
        let a = SharedMem::new(g).unwrap();
        let b = SharedMem::new(g).unwrap();
        let space = Arc::new(AddrSpace::reserve(g).unwrap());
        unsafe { space.map(0, g, &a, 0, Prot::ReadWrite).unwrap() };
        let stop = Arc::new(AtomicBool::new(false));
        let reader_space = space.clone();
        let reader_stop = stop.clone();
        let reader = std::thread::spawn(move || {
            let p = reader_space.base();
            while !reader_stop.load(Ordering::Acquire) {
                unsafe {
                    assert_eq!(std::ptr::read_volatile(p.add(g - 1)), 0);
                    assert_eq!((*p.cast::<AtomicU64>()).load(Ordering::Relaxed), 0);
                }
            }
        });
        let result: io::Result<()> = (|| {
            for i in 0..1000 {
                unsafe {
                    space.scrub(0, g)?;
                    space.map(0, g, if i % 2 == 0 { &a } else { &b }, 0, Prot::ReadWrite)?;
                }
            }
            Ok(())
        })();
        stop.store(true, Ordering::Release);
        reader.join().unwrap();
        result.unwrap();
    }
}
