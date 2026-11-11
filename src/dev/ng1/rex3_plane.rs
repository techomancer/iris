//! Opt-in, page-isolated CPU shadows for the resident raster's unit tests.

use std::io;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AccessCounts {
    pub(crate) reads: usize,
    pub(crate) writes: usize,
}

enum Storage {
    Boxed(Box<[u32]>),
    Isolated(Arc<Mapping>),
}

/// Slice-compatible with the normal boxed plane. Existing tests start with
/// boxed storage; only a test that explicitly isolates it gets OS pages.
pub struct CpuPlane {
    storage: Storage,
    reads: AtomicUsize,
    writes: AtomicUsize,
}

impl From<Box<[u32]>> for CpuPlane {
    fn from(storage: Box<[u32]>) -> Self {
        Self { storage: Storage::Boxed(storage), reads: AtomicUsize::new(0), writes: AtomicUsize::new(0) }
    }
}

impl CpuPlane {
    /// Replace the box only while the board is exclusively owned and stopped.
    /// Failure leaves the original plane intact.
    pub(crate) fn isolate(&mut self) -> io::Result<()> {
        if matches!(self.storage, Storage::Isolated(_)) {
            return Err(io::Error::other("CPU plane is already isolated"));
        }
        let mapping = Mapping::copy_from(self)?;
        self.storage = Storage::Isolated(Arc::new(mapping));
        Ok(())
    }

    pub(crate) fn reset_access_counts(&self) {
        self.reads.store(0, Ordering::Relaxed);
        self.writes.store(0, Ordering::Relaxed);
    }

    pub(crate) fn access_counts(&self) -> AccessCounts {
        AccessCounts {
            reads: self.reads.load(Ordering::Relaxed),
            writes: self.writes.load(Ordering::Relaxed),
        }
    }

    /// Mapping address, reserved byte length and OS page size, without
    /// obtaining a slice into the mapping.
    pub(crate) fn mapping_range(&self) -> Option<(usize, usize, usize)> {
        match &self.storage {
            Storage::Boxed(_) => None,
            Storage::Isolated(mapping) => Some((mapping.base.as_ptr() as usize, mapping.bytes, mapping.page_size)),
        }
    }

    /// No CPU slice may be live, or in use by another thread, when protection
    /// is changed. Raw pointers may be retained, but must not be dereferenced
    /// until the guard is dropped (except in the fault-probe child process).
    pub(crate) unsafe fn protect(&self) -> io::Result<PlaneProtection> {
        let Storage::Isolated(mapping) = &self.storage else {
            return Err(io::Error::other("CPU plane must be isolated before protection"));
        };
        mapping.protect()?;
        Ok(PlaneProtection { mapping: Arc::clone(mapping) })
    }
}

impl Deref for CpuPlane {
    type Target = [u32];

    fn deref(&self) -> &[u32] {
        self.reads.fetch_add(1, Ordering::Relaxed);
        match &self.storage {
            Storage::Boxed(words) => words,
            Storage::Isolated(mapping) => {
                assert!(!mapping.protected.load(Ordering::Acquire), "access to protected CPU plane");
                // SAFETY: the isolated allocation contains initialized u32s,
                // and protection changes require quiescent slice users.
                unsafe { std::slice::from_raw_parts(mapping.base.as_ptr(), mapping.words) }
            }
        }
    }
}

impl DerefMut for CpuPlane {
    fn deref_mut(&mut self) -> &mut [u32] {
        self.writes.fetch_add(1, Ordering::Relaxed);
        match &mut self.storage {
            Storage::Boxed(words) => words,
            Storage::Isolated(mapping) => {
                assert!(!mapping.protected.load(Ordering::Acquire), "access to protected CPU plane");
                // SAFETY: the mutable plane borrow is exclusive; a protection
                // guard only owns mapping metadata, never a slice.
                unsafe { std::slice::from_raw_parts_mut(mapping.base.as_ptr(), mapping.words) }
            }
        }
    }
}

impl AsRef<[u32]> for CpuPlane {
    fn as_ref(&self) -> &[u32] { self }
}

impl AsMut<[u32]> for CpuPlane {
    fn as_mut(&mut self) -> &mut [u32] { self }
}

impl Clone for CpuPlane {
    fn clone(&self) -> Self { self.to_vec().into_boxed_slice().into() }
}

impl PartialEq for CpuPlane {
    fn eq(&self, other: &Self) -> bool { self.deref() == other.deref() }
}

impl Eq for CpuPlane {}

impl std::fmt::Debug for CpuPlane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.deref().fmt(f) }
}

struct Mapping {
    base: NonNull<u32>,
    words: usize,
    bytes: usize,
    page_size: usize,
    protected: AtomicBool,
}

// The mapping owns its pages; slice access is governed by CpuPlane's borrows
// and protection changes are explicitly coordinated by the test.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    fn copy_from(words: &[u32]) -> io::Result<Self> {
        let page_size = os::page_size()?;
        let content_bytes = words.len().checked_mul(size_of::<u32>())
            .ok_or_else(|| io::Error::other("CPU plane size overflow"))?;
        if content_bytes == 0 {
            return Err(io::Error::other("cannot isolate an empty CPU plane"));
        }
        let bytes = content_bytes.checked_add(page_size - 1)
            .ok_or_else(|| io::Error::other("CPU plane mapping size overflow"))?
            / page_size * page_size;
        let base = os::allocate(bytes)?;
        let mapping = Self {
            base, words: words.len(), bytes, page_size, protected: AtomicBool::new(false),
        };
        // SAFETY: the new private mapping is writable and large enough, and
        // cannot overlap the still-live source box.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), mapping.base.as_ptr(), words.len()) };
        Ok(mapping)
    }

    fn protect(&self) -> io::Result<()> {
        self.protected.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| io::Error::other("CPU plane is already protected"))?;
        if let Err(error) = os::protect(self.base, self.bytes, true) {
            // Restore even if an OS failed after changing part of the range.
            if os::protect(self.base, self.bytes, false).is_err() {
                std::process::abort();
            }
            self.protected.store(false, Ordering::Release);
            return Err(error);
        }
        Ok(())
    }

    fn unprotect(&self) {
        if self.protected.load(Ordering::Acquire) {
            // A failed restoration cannot safely leave a live plane behind.
            if os::protect(self.base, self.bytes, false).is_err() {
                std::process::abort();
            }
            self.protected.store(false, Ordering::Release);
        }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        self.unprotect();
        if os::release(self.base, self.bytes).is_err() {
            std::process::abort();
        }
    }
}

/// Keeps the mapping alive even if its owner is dropped first. Unwinding drops
/// this guard and restores read/write access before the mapping can be freed.
pub(crate) struct PlaneProtection {
    mapping: Arc<Mapping>,
}

impl Drop for PlaneProtection {
    fn drop(&mut self) { self.mapping.unprotect(); }
}

#[cfg(windows)]
mod os {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
        PAGE_NOACCESS, PAGE_READWRITE,
    };
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

    pub(super) fn page_size() -> io::Result<usize> {
        let mut info = std::mem::MaybeUninit::<SYSTEM_INFO>::uninit();
        // SAFETY: GetSystemInfo initializes the whole output structure.
        unsafe { GetSystemInfo(info.as_mut_ptr()) };
        let size = unsafe { info.assume_init() }.dwPageSize as usize;
        if size == 0 { Err(io::Error::other("Windows returned a zero page size")) } else { Ok(size) }
    }

    pub(super) fn allocate(bytes: usize) -> io::Result<NonNull<u32>> {
        // SAFETY: a private reservation is committed read/write for its owner.
        let base = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        NonNull::new(base.cast()).ok_or_else(io::Error::last_os_error)
    }

    pub(super) fn protect(base: NonNull<u32>, bytes: usize, none: bool) -> io::Result<()> {
        let mut old = 0;
        let flags = if none { PAGE_NOACCESS } else { PAGE_READWRITE };
        // SAFETY: base and bytes describe our complete VirtualAlloc reservation.
        if unsafe { VirtualProtect(base.as_ptr().cast(), bytes, flags, &mut old) } == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    pub(super) fn release(base: NonNull<u32>, _bytes: usize) -> io::Result<()> {
        // SAFETY: base is the original reservation, released exactly once.
        if unsafe { VirtualFree(base.as_ptr().cast(), 0, MEM_RELEASE) } == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

#[cfg(unix)]
mod os {
    use super::*;

    pub(super) fn page_size() -> io::Result<usize> {
        // SAFETY: sysconf has no pointer arguments.
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if size <= 0 { Err(io::Error::last_os_error()) } else { Ok(size as usize) }
    }

    pub(super) fn allocate(bytes: usize) -> io::Result<NonNull<u32>> {
        // SAFETY: anonymous private pages do not overlap an existing allocation.
        let base = unsafe {
            libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0)
        };
        if base == libc::MAP_FAILED {
            Err(io::Error::last_os_error())
        } else {
            match NonNull::new(base.cast()) {
                Some(base) => Ok(base),
                None => {
                    // A mapping at address zero cannot back a Rust slice.
                    unsafe { libc::munmap(base, bytes) };
                    Err(io::Error::other("mmap returned a null address"))
                }
            }
        }
    }

    pub(super) fn protect(base: NonNull<u32>, bytes: usize, none: bool) -> io::Result<()> {
        let flags = if none { libc::PROT_NONE } else { libc::PROT_READ | libc::PROT_WRITE };
        // SAFETY: base and bytes describe our complete private mmap.
        if unsafe { libc::mprotect(base.as_ptr().cast(), bytes, flags) } != 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    pub(super) fn release(base: NonNull<u32>, bytes: usize) -> io::Result<()> {
        // SAFETY: base and bytes describe the original mapping, released once.
        if unsafe { libc::munmap(base.as_ptr().cast(), bytes) } != 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

#[cfg(not(any(windows, unix)))]
mod os {
    use super::*;

    fn unsupported<T>() -> io::Result<T> {
        Err(io::Error::new(io::ErrorKind::Unsupported, "CPU plane protection needs Windows or Unix"))
    }

    pub(super) fn page_size() -> io::Result<usize> { unsupported() }
    pub(super) fn allocate(_: usize) -> io::Result<NonNull<u32>> { unsupported() }
    pub(super) fn protect(_: NonNull<u32>, _: usize, _: bool) -> io::Result<()> { unsupported() }
    pub(super) fn release(_: NonNull<u32>, _: usize) -> io::Result<()> { unsupported() }
}
