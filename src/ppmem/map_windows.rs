//! Windows backend for ppmem's virtual-memory primitives — see
//! `docs/ppmem-design.md` §7.3.
//!
//! **Status: written against the API contract, not yet run on a Windows host.**
//! It must pass the shared test suite in `map.rs`, which the Unix backend
//! already passes; that suite is the definition of "done" here.
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
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile3, UnmapViewOfFile2, VirtualAlloc2, VirtualFree,
    MEMORY_MAPPED_VIEW_ADDRESS, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
    MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS,
    PAGE_READONLY, PAGE_READWRITE,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// The current-process pseudo-handle. Not a real handle — never closed.
fn current_process() -> HANDLE {
    unsafe { GetCurrentProcess() }
}

/// Anonymous shared memory that can be mapped at several addresses at once.
///
/// A pagefile-backed section object — Windows' equivalent of a Unix shm
/// object. Unnamed, so it is private to this process and reclaimed when the
/// handle closes.
pub struct SharedMem {
    handle: HANDLE,
    len: usize,
}

unsafe impl Send for SharedMem {}
unsafe impl Sync for SharedMem {}

impl SharedMem {
    /// Create a shared memory object of `len` bytes, zero-filled.
    pub fn new(len: usize) -> io::Result<Self> {
        assert!(len > 0, "ppmem: SharedMem::new with zero length");
        assert!(
            super::is_aligned(len),
            "ppmem: SharedMem length {len:#x} is not granularity-aligned"
        );

        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE, // pagefile-backed, not a real file
                std::ptr::null(),
                PAGE_READWRITE,
                (len >> 32) as u32,
                (len & 0xFFFF_FFFF) as u32,
                std::ptr::null(), // unnamed
            )
        };
        if handle.is_null() {
            return Err(oserr("CreateFileMappingW"));
        }
        Ok(Self { handle, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Windows has no section-level hole punch equivalent to
    /// `FALLOC_FL_PUNCH_HOLE`, so callers must zero the mapping themselves.
    /// See the design doc's [Q4].
    pub fn discard(&self) -> bool {
        false
    }
}

impl Drop for SharedMem {
    fn drop(&mut self) {
        unsafe { CloseHandle(self.handle) };
    }
}

/// A contiguous placeholder reservation of host address space.
///
/// Held from creation to drop. Sub-ranges are carved out of it as needed, but
/// the address range never stops being ours.
pub struct AddrSpace {
    base: *mut u8,
    size: usize,
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
        Ok(Self {
            base: p as *mut u8,
            size,
        })
    }

    pub fn base(&self) -> *mut u8 {
        self.base
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn check_range(&self, at: usize, len: usize) {
        assert!(super::is_aligned(at), "ppmem: offset {at:#x} misaligned");
        assert!(super::is_aligned(len), "ppmem: length {len:#x} misaligned");
        assert!(
            at.checked_add(len).is_some_and(|end| end <= self.size),
            "ppmem: [{at:#x}, +{len:#x}) escapes the {:#x}-byte reservation",
            self.size
        );
    }

    /// Split a bank-sized placeholder out of the enclosing placeholder, so a
    /// view can replace exactly that range.
    ///
    /// Splitting a placeholder that is already exactly this range fails with
    /// `ERROR_INVALID_PARAMETER`; that is benign and means the slot is already
    /// carved, so it is not treated as an error.
    unsafe fn split(&self, at: usize, len: usize) {
        // If this range is the whole reservation there is nothing to split.
        if at == 0 && len == self.size {
            return;
        }
        unsafe {
            VirtualFree(
                self.base.add(at) as *mut _,
                len,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
        }
        // Deliberately ignoring the return: the only expected failure is
        // "already split to these bounds", and a genuine problem surfaces at
        // the MapViewOfFile3 below with a far more useful error.
    }

    /// Map `mem[offset .. offset+len]` at `base() + at`.
    ///
    /// # Safety
    ///
    /// Replaces mappings in place; see the Unix backend's `map`.
    pub unsafe fn map(
        &self,
        at: usize,
        len: usize,
        mem: &SharedMem,
        offset: u64,
        prot: Prot,
    ) -> io::Result<()> {
        self.check_range(at, len);
        assert!(
            super::is_aligned(offset as usize),
            "ppmem: object offset {offset:#x} misaligned"
        );
        assert!(
            offset as usize + len <= mem.len(),
            "ppmem: [{offset:#x}, +{len:#x}) escapes the {:#x}-byte object",
            mem.len()
        );

        // A view already occupying this range must go before it can be
        // replaced, and it must revert to a placeholder rather than to free
        // space. Failure here just means nothing was mapped.
        unsafe {
            let _ = UnmapViewOfFile2(
                current_process(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.base.add(at) as *mut _,
                },
                MEM_PRESERVE_PLACEHOLDER,
            );
            self.split(at, len);
        }

        let page_prot = match prot {
            Prot::ReadWrite => PAGE_READWRITE,
            Prot::ReadOnly => PAGE_READONLY,
        };

        let want = unsafe { self.base.add(at) };
        let got = unsafe {
            MapViewOfFile3(
                mem.handle,
                current_process(),
                want as *const _,
                offset,
                len,
                MEM_REPLACE_PLACEHOLDER,
                page_prot,
                std::ptr::null_mut(),
                0,
            )
        };
        if got.Value.is_null() {
            return Err(oserr("MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)"));
        }
        assert_eq!(
            got.Value as usize, want as usize,
            "ppmem: MapViewOfFile3 landed at {:p}, wanted {want:p}",
            got.Value
        );
        Ok(())
    }

    /// Revert `[at, at+len)` to placeholder, retaining the claim.
    ///
    /// # Safety
    ///
    /// Same as the Unix backend's `unmap`.
    pub unsafe fn unmap(&self, at: usize, len: usize) -> io::Result<()> {
        self.check_range(at, len);
        let ok = unsafe {
            UnmapViewOfFile2(
                current_process(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.base.add(at) as *mut _,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        if ok == 0 {
            return Err(oserr("UnmapViewOfFile2(MEM_PRESERVE_PLACEHOLDER)"));
        }
        Ok(())
    }
}

impl Drop for AddrSpace {
    fn drop(&mut self) {
        unsafe {
            // Unmap every view still outstanding, preserving placeholders, then
            // release the whole reservation. Walking in granularity steps is
            // crude but runs once at shutdown; a view that isn't there simply
            // fails and is skipped.
            let g = super::granularity();
            let mut at = 0;
            while at < self.size {
                UnmapViewOfFile2(
                    current_process(),
                    MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: self.base.add(at) as *mut _,
                    },
                    MEM_PRESERVE_PLACEHOLDER,
                );
                at += g;
            }
            VirtualFree(self.base as *mut _, 0, MEM_RELEASE);
        }
    }
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
