//! A page's generation: how writers move it and how a compile reads it.
//!
//! Every write to a page stores its bytes and then bumps one of the page's
//! two counters (`GenPtr`); the page's generation is their sum. Compiled
//! stores bump the inline counter, which on one CPU has no other writer, so a
//! plain load, add and store is enough. Everything else bumps the shared
//! counter with [`bump`], an atomic add, so writers on any thread never lose
//! each other's bumps.

/// One of a page's counters. Implemented for std's atomics and, under
/// `cfg(loom)`, loom's, so the models run these functions.
pub(crate) trait Counter {
    fn value(&self) -> u64;
    fn add(&self, n: u64);
}

impl Counter for std::sync::atomic::AtomicU64 {
    #[inline(always)]
    fn value(&self) -> u64 {
        self.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[inline(always)]
    fn add(&self, n: u64) {
        self.fetch_add(n, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(loom)]
impl Counter for loom::sync::atomic::AtomicU64 {
    fn value(&self) -> u64 {
        self.load(loom::sync::atomic::Ordering::Relaxed)
    }
    fn add(&self, n: u64) {
        self.fetch_add(n, loom::sync::atomic::Ordering::Release);
    }
}

/// The page's generation. Relaxed: a compile orders its reads with fences
/// (`comp::snapshot_page`), and dispatch with `entry_gen`'s acquire.
///
/// The sum moves whenever either counter does and never returns to a value
/// it left: each counter only rises, except a release taking `GEN_HELD` off
/// the shared one, which still leaves the sum one above where it was before
/// the hold.
#[inline(always)]
pub(crate) fn generation(shared: &impl Counter, inline: &impl Counter) -> u64 {
    shared.value().wrapping_add(inline.value())
}

/// Bump the shared counter after a write's bytes. Release: a compile that
/// sees the new generation sees the bytes too.
#[inline(always)]
pub(crate) fn bump(shared: &impl Counter) {
    shared.add(1);
}
