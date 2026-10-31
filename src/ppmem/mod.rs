//! ppmem — physical paged memory backed by the host MMU.
//!
//! An alternative to [`crate::dev::mem::Memory`] that presents the same interface
//! (`BusDevice`, `Resettable`, and `Memory`'s inherent methods) but backs the
//! guest's physical address space with real host virtual memory, so that
//! SIMM mirroring and the low-512KB alias become *mappings* rather than
//! address arithmetic on every access. `devmap` applies the same idea to
//! the physical address decode table itself.
//!
//! Full design and rationale: `docs/ppmem-design.md`.
//! Verified platform findings: `clanker/build/mmap-rs-fixed-address-aliasing.md`.

pub mod devmap;
pub mod map;
#[allow(clippy::module_inception)]
pub mod ppmem;

pub use map::{align_up, granularity, is_aligned, AddrSpace, Prot, SharedMem};
pub use devmap::SparseDeviceMap;
pub use ppmem::{bitmap_maps, bitmap_maps_at, window_shift, MappedMemory, PpMemSpace, PpMemory, BITMAP_SHIFT, LARGE_WINDOW_SIZE, WINDOW_SIZE};
