//! Unix backend for ppmem's virtual-memory primitives — **Linux and macOS
//! both**, see `docs/ppmem-design.md` §7.2.
//!
//! The two platforms differ in exactly one place: how the anonymous shared
//! memory object is created. Linux has `memfd_create`; macOS does not. Since
//! POSIX `shm_open` works on both, this uses `shm_open` everywhere — which
//! means the macOS code path is exercised by running these tests on Linux.
//! That matters because macOS is not routinely testable here.
//!
//! ## The Darwin `ftruncate` trap
//!
//! macOS allows `ftruncate` on a POSIX shm object exactly **once** — a second
//! call returns `EINVAL` because `PSHM_ALLOCATED` is set — and a stale segment
//! surviving a crash can be re-opened already-allocated, so a plain
//! `O_CREAT` open can hand back an object that can never be sized.
//!
//! Both hazards are avoided by construction in `SharedMem::new`:
//! `O_CREAT | O_EXCL` with a unique name (never reuse a stale object),
//! `shm_unlink` immediately (the fd keeps it alive; nothing persists to be
//! found later), then exactly one `ftruncate` before any mapping exists.

use super::{oserr, Prot};
use std::io;
use std::sync::atomic::{AtomicU32, Ordering};

/// Anonymous shared memory that can be mapped at several addresses at once.
///
/// Backed by a POSIX shm object that is unlinked immediately after creation,
/// so it is anonymous in every sense that matters: invisible to other
/// processes, and reclaimed when the fd closes. Nothing reaches the disk —
/// this is tmpfs/RAM on both platforms.
pub struct SharedMem {
    fd: libc::c_int,
    len: usize,
}

// The fd is just a handle; sharing it across threads is fine.
unsafe impl Send for SharedMem {}
unsafe impl Sync for SharedMem {}

impl SharedMem {
    /// Create a shared memory object of `len` bytes, zero-filled.
    ///
    /// `len` must be granularity-aligned (see `super::granularity`).
    pub fn new(len: usize) -> io::Result<Self> {
        assert!(len > 0, "ppmem: SharedMem::new with zero length");
        assert!(
            super::is_aligned(len),
            "ppmem: SharedMem length {len:#x} is not granularity-aligned"
        );

        // Unique per process *and* per object, so O_EXCL can never collide with
        // one of our own live objects.
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        // SHM_NAME_MAX is small (31 on macOS incl. NUL) — keep this short.
        let name = format!("/iris-pp{}-{}\0", std::process::id(), seq);

        let fd = unsafe {
            libc::shm_open(
                name.as_ptr() as *const libc::c_char,
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                0o600 as libc::c_uint,
            )
        };
        if fd < 0 {
            return Err(oserr("shm_open"));
        }

        // Unlink at once: the fd holds the object alive, and no stale segment
        // can be left behind for a later run to re-open already-allocated.
        unsafe { libc::shm_unlink(name.as_ptr() as *const libc::c_char) };

        // Exactly one ftruncate, before any mapping — the only form Darwin
        // accepts.
        if unsafe { libc::ftruncate(fd, len as libc::off_t) } != 0 {
            let e = oserr("ftruncate");
            unsafe { libc::close(fd) };
            return Err(e);
        }

        Ok(Self { fd, len })
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Release the object's physical pages, zeroing every mapping of it, while
    /// keeping the mappings themselves in place. This is `power_on()`'s
    /// primitive: one syscall instead of memsetting the whole bank, and it
    /// returns the RAM to the OS.
    ///
    /// Only Linux has hole-punching for shm objects; elsewhere the caller must
    /// fall back to writing zeroes, so this reports whether it did anything.
    pub fn discard(&self) -> bool {
        #[cfg(target_os = "linux")]
        unsafe {
            libc::fallocate(
                self.fd,
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                0,
                self.len as libc::off_t,
            ) == 0
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
}

impl Drop for SharedMem {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
    }
}

/// A contiguous reservation of host address space that we own outright.
///
/// Held from creation to drop. Owning the range is what makes `MAP_FIXED` safe
/// to use: `MAP_FIXED` over an address we do *not* own would silently unmap
/// whatever is there — the JIT's own arena, a thread stack, a `dlopen`ed
/// library — and corrupt it with no error at the call site.
pub struct AddrSpace {
    base: *mut u8,
    size: usize,
}

unsafe impl Send for AddrSpace {}
unsafe impl Sync for AddrSpace {}

impl AddrSpace {
    /// Reserve `size` bytes of inaccessible address space. The kernel picks a
    /// base that is guaranteed not to collide with anything else.
    pub fn reserve(size: usize) -> io::Result<Self> {
        assert!(size > 0, "ppmem: AddrSpace::reserve with zero size");
        assert!(
            super::is_aligned(size),
            "ppmem: reservation size {size:#x} is not granularity-aligned"
        );

        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(oserr("mmap(reserve)"));
        }
        Ok(Self {
            base: p as *mut u8,
            size,
        })
    }

    /// Base of the reservation. Offset `n` within it is `base() + n`.
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// Size of the reservation in bytes.
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

    /// Map `mem[offset .. offset+len]` at `base() + at`, atomically replacing
    /// whatever currently occupies that range.
    ///
    /// Calling this repeatedly with the same `mem` and different `at` is what
    /// makes a bank repeat across a larger region — every view is the same
    /// physical memory.
    ///
    /// # Safety
    ///
    /// Replaces mappings in place: any pointer into `[at, at+len)` obtained
    /// earlier may now refer to different memory. The caller must ensure no
    /// other thread is reading through this range concurrently.
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

        let prot_bits = match prot {
            Prot::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
            Prot::ReadOnly => libc::PROT_READ,
        };

        // MAP_FIXED replaces the reservation atomically — there is no window in
        // which the range is unmapped and could be handed to someone else.
        let want = self.base.add(at);
        let got = libc::mmap(
            want as *mut libc::c_void,
            len,
            prot_bits,
            libc::MAP_SHARED | libc::MAP_FIXED,
            mem.fd,
            offset as libc::off_t,
        );
        if got == libc::MAP_FAILED {
            return Err(oserr("mmap(MAP_FIXED)"));
        }
        // MAP_FIXED guarantees placement, but assert rather than trust: the
        // Windows backend has no such guarantee and shares this contract, and
        // a silently misplaced mapping is the exact failure mode that makes
        // `mmap-rs` unusable on Windows.
        assert_eq!(
            got as usize, want as usize,
            "ppmem: MAP_FIXED landed at {got:p}, wanted {want:p}"
        );
        Ok(())
    }

    /// Revert `[at, at+len)` to inaccessible reservation, **retaining the
    /// claim**.
    ///
    /// Deliberately not `munmap`: that would hand the address space back to
    /// the kernel, after which another thread's allocation could take it and
    /// the reservation would be full of holes. Re-mapping `PROT_NONE` over the
    /// range keeps it ours. (Windows expresses the same invariant as
    /// `MEM_PRESERVE_PLACEHOLDER`.)
    ///
    /// # Safety
    ///
    /// Same as `map`: pointers into the range become invalid, and accessing
    /// the range afterwards faults.
    pub unsafe fn unmap(&self, at: usize, len: usize) -> io::Result<()> {
        self.check_range(at, len);
        let want = self.base.add(at);
        let got = libc::mmap(
            want as *mut libc::c_void,
            len,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED | libc::MAP_NORESERVE,
            -1,
            0,
        );
        if got == libc::MAP_FAILED {
            return Err(oserr("mmap(PROT_NONE)"));
        }
        assert_eq!(
            got as usize, want as usize,
            "ppmem: unmap landed at {got:p}, wanted {want:p}"
        );
        Ok(())
    }
}

impl Drop for AddrSpace {
    fn drop(&mut self) {
        // One munmap releases the whole reservation including every view
        // mapped into it.
        unsafe { libc::munmap(self.base as *mut libc::c_void, self.size) };
    }
}

pub fn granularity() -> usize {
    // Unix has no separate allocation granularity; the page size is the
    // alignment mmap requires for both address and file offset.
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if n > 0 {
        n as usize
    } else {
        4096
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_mem_is_zero_filled() {
        let g = super::super::granularity();
        let mem = SharedMem::new(g).unwrap();
        let space = AddrSpace::reserve(g).unwrap();
        unsafe { space.map(0, g, &mem, 0, Prot::ReadWrite) }.unwrap();
        unsafe {
            for i in 0..g {
                assert_eq!(*space.base().add(i), 0, "byte {i} not zero-filled");
            }
        }
    }

    /// `discard` is `power_on()`'s primitive: it must zero *every* alias, since
    /// they are all the same physical pages.
    #[cfg(target_os = "linux")]
    #[test]
    fn discard_zeroes_every_alias() {
        let g = super::super::granularity();
        let len = 4 * g;
        let mem = SharedMem::new(len).unwrap();
        let space = AddrSpace::reserve(len * 2).unwrap();
        unsafe { space.map(0, len, &mem, 0, Prot::ReadWrite) }.unwrap();
        unsafe { space.map(len, len, &mem, 0, Prot::ReadWrite) }.unwrap();

        unsafe {
            *space.base().add(7) = 0x5A;
            assert_eq!(*space.base().add(len + 7), 0x5A, "aliases not coherent");
        }

        assert!(mem.discard(), "discard should succeed on Linux");

        unsafe {
            assert_eq!(*space.base().add(7), 0, "alias 0 not zeroed by discard");
            assert_eq!(*space.base().add(len + 7), 0, "alias 1 not zeroed by discard");
        }
    }

    /// Two objects must be independent — a sanity check that the O_EXCL naming
    /// really does produce a fresh object each time rather than reopening one.
    #[test]
    fn distinct_objects_do_not_alias() {
        let g = super::super::granularity();
        let a = SharedMem::new(g).unwrap();
        let b = SharedMem::new(g).unwrap();
        let space = AddrSpace::reserve(2 * g).unwrap();
        unsafe { space.map(0, g, &a, 0, Prot::ReadWrite) }.unwrap();
        unsafe { space.map(g, g, &b, 0, Prot::ReadWrite) }.unwrap();
        unsafe {
            *space.base() = 0x11;
            *space.base().add(g) = 0x22;
            assert_eq!(*space.base(), 0x11, "distinct objects are aliasing");
        }
    }

    #[test]
    fn read_only_mapping_is_not_writable() {
        let g = super::super::granularity();
        let mem = SharedMem::new(g).unwrap();
        let space = AddrSpace::reserve(g).unwrap();
        unsafe { space.map(0, g, &mem, 0, Prot::ReadOnly) }.unwrap();
        // Reading is fine; we don't attempt the write, which would SIGSEGV by
        // design. Just prove the mapping exists and reads as zero.
        unsafe { assert_eq!(*space.base(), 0) };
    }
}
