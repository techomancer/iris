//! Host virtual-memory primitives for ppmem (see `docs/ppmem-design.md` §7).
//!
//! Three operations, and only three: reserve a large range of address space,
//! map a shared-memory object at a fixed address inside it (possibly several
//! times over, which is how bank mirroring works), and revert a sub-range to
//! reservation without ever releasing the claim.
//!
//! No mapping crate does this portably — `memmap2` has no fixed-address
//! mapping at all, and `mmap-rs` silently ignores the requested address for
//! file-backed mappings on Windows (see
//! `rules/build/mmap-rs-fixed-address-aliasing.md`). So this is hand-written
//! over `libc` / `windows-sys`, both of which are already dependencies.
//!
//! # Why a shared-memory handle rather than anonymous memory
//!
//! Aliasing — the same physical page visible at several addresses — requires a
//! *named* backing object. Two anonymous mappings are two unrelated
//! allocations; there is nothing to refer back to. `MAP_SHARED` shares across
//! `fork()`, not across mappings. Hence `SharedMem`: a POSIX shm object on
//! Unix, a section object on Windows. Nothing touches disk in either case.

use std::io;

#[cfg_attr(unix, path = "map_unix.rs")]
#[cfg_attr(windows, path = "map_windows.rs")]
mod imp;

pub use imp::{AddrSpace, SharedMem};

/// Page protection for a mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prot {
    /// Readable and writable — normal RAM.
    ReadWrite,
    /// Readable only — for immutable regions (ROM/PROM, if ppmem ever covers
    /// them; see the design doc's [Q6]).
    ReadOnly,
}

/// Host allocation granularity: the alignment every `map`/`unmap` offset and
/// length must satisfy.
///
/// 4KB-ish on Unix, but **64KB on Windows** — `VirtualAlloc2` placeholders can
/// only be split on 64KB boundaries and `MapViewOfFile3` views must be 64KB
/// aligned in both address and section offset. Callers should use this rather
/// than assuming the page size; ppmem's 8MB minimum bank satisfies it either
/// way.
pub fn granularity() -> usize {
    imp::granularity()
}

/// Round `n` up to a multiple of the host granularity.
pub fn align_up(n: usize) -> usize {
    let g = granularity();
    (n + g - 1) & !(g - 1)
}

/// Check that `n` is granularity-aligned.
pub fn is_aligned(n: usize) -> bool {
    n & (granularity() - 1) == 0
}

/// Error context helper — every failure here is an OS error worth reporting
/// with what we were trying to do.
pub(crate) fn oserr(what: &str) -> io::Error {
    let e = io::Error::last_os_error();
    io::Error::new(e.kind(), format!("ppmem: {what}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full lifecycle the design depends on, run against whichever backend
    /// is compiled. This is the contract the Windows implementation must meet
    /// to be considered done — it is the same test the Unix one passes.
    #[test]
    fn reserve_map_alias_unmap_remap() {
        let g = granularity();
        let bank_len = 8 * g;
        let region = 4 * bank_len;

        let mem = SharedMem::new(bank_len).expect("SharedMem::new");
        let space = AddrSpace::reserve(region).expect("reserve");
        let base = space.base();

        // Map the one bank four times back to back: an undersized bank
        // repeating to fill its slot, which is how SIMM mirroring is expressed.
        for i in 0..4 {
            unsafe { space.map(i * bank_len, bank_len, &mem, 0, Prot::ReadWrite) }
                .unwrap_or_else(|e| panic!("map alias {i}: {e}"));
        }

        // All four views must be the same physical memory.
        unsafe {
            let p = base;
            *p.add(0) = 0xAB;
            *p.add(bank_len - 1) = 0xCD;
            for i in 1..4 {
                assert_eq!(*p.add(i * bank_len), 0xAB, "alias {i} not coherent at 0");
                assert_eq!(
                    *p.add(i * bank_len + bank_len - 1),
                    0xCD,
                    "alias {i} not coherent at end"
                );
            }
            // ...and coherent in the other direction too.
            *p.add(3 * bank_len + 4) = 0x77;
            assert_eq!(*p.add(4), 0x77, "aliasing is not bidirectional");
        }

        // Revert alias 1 to reservation. The claim must be retained, and the
        // other aliases must be untouched.
        unsafe { space.unmap(bank_len, bank_len) }.expect("unmap");
        unsafe {
            assert_eq!(*base.add(0), 0xAB, "unmapping alias 1 damaged alias 0");
            assert_eq!(*base.add(2 * bank_len), 0xAB, "unmapping alias 1 damaged alias 2");
        }

        // The reverted slot must be re-mappable, and still alias.
        unsafe { space.map(bank_len, bank_len, &mem, 0, Prot::ReadWrite) }
            .expect("remap into reverted slot");
        unsafe {
            assert_eq!(*base.add(bank_len + 4), 0x77, "remapped view lost aliasing");
        }
    }

    /// Mapping at a non-zero offset within the shared object.
    #[test]
    fn map_at_object_offset() {
        let g = granularity();
        let half = 4 * g;
        let mem = SharedMem::new(half * 2).expect("SharedMem::new");
        let space = AddrSpace::reserve(half * 2).expect("reserve");

        unsafe { space.map(0, half, &mem, 0, Prot::ReadWrite) }.expect("map low half");
        unsafe { space.map(half, half, &mem, half as u64, Prot::ReadWrite) }
            .expect("map high half");

        unsafe {
            // Two different windows onto the same object: writes must not alias.
            *space.base().add(0) = 0x11;
            *space.base().add(half) = 0x22;
            assert_eq!(*space.base().add(0), 0x11);
            assert_eq!(*space.base().add(half), 0x22);
        }
    }

    #[test]
    fn granularity_is_a_power_of_two() {
        let g = granularity();
        assert!(g >= 4096, "granularity {g} unexpectedly small");
        assert_eq!(g & (g - 1), 0, "granularity {g} is not a power of two");
    }

    #[test]
    fn alignment_helpers() {
        let g = granularity();
        assert!(is_aligned(0));
        assert!(is_aligned(g));
        assert!(!is_aligned(g + 1));
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), g);
        assert_eq!(align_up(g), g);
        assert_eq!(align_up(g + 1), 2 * g);
    }
}
