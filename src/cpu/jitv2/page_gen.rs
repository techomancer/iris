//! A page's generation: how writers move it and how a compile reads it.
//!
//! Every write to a page stores its bytes and then bumps one of the page's
//! two counters (`GenPtr`); the page's generation is their sum. Compiled
//! stores bump the inline counter, which on one CPU has no other writer, so a
//! plain load, add and store is enough. Everything else bumps the shared
//! counter with [`bump`], an atomic add, so writers on any thread never lose
//! each other's bumps.
//!
//! `loom_tests` checks the protocol against a compiled store racing a DMA
//! write and a compile: `RUSTFLAGS="--cfg loom" cargo test --release
//! --features jitv2 --lib page_gen::loom_tests`.

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

/// Publish's rule: a compile snapshotted at `snapshot` is refused once the
/// page has moved past it.
#[inline(always)]
pub(crate) fn outdated(current: u64, snapshot: u64) -> bool {
    current > snapshot
}

/// A seqlock snapshot of a code page's words: the generation the words
/// were read under, or why there is none.
pub(crate) enum Snapshot {
    Taken(u64),
    /// A word could not be read (not RAM right now).
    Unreadable,
    /// A CPU's cache holds lines in the page, under this generation.
    Held(u64),
    /// The generation moved under every attempt.
    Busy,
}

/// §13.3 step 2, the seqlock reader: read the page (`read`; `false` if it is
/// unreadable) between two reads of its generation (`gen`), and keep what it
/// read if the generation held. Bounded retries — a page under sustained
/// concurrent SMC just keeps losing the race to itself, correctly punted to
/// the next arrival's request rather than compiled against torn bytes.
pub(crate) fn seqlock_read(gen: impl Fn() -> u64, mut read: impl FnMut() -> bool) -> Snapshot {
    #[cfg(loom)]
    use loom::sync::atomic::{fence, Ordering};
    #[cfg(not(loom))]
    use std::sync::atomic::{fence, Ordering};
    const MAX_SEQLOCK_RETRIES: u32 = 8;
    for _ in 0..MAX_SEQLOCK_RETRIES {
        let gen0 = gen();
        // A CPU's cache holds lines in the page (`GEN_HELD`).
        if gen0 >= crate::cpu::jitv2::jitv2::GEN_HELD {
            return Snapshot::Held(gen0);
        }
        // A seqlock reader: the words are read after `gen0` and before
        // `gen1` (acquire), and every writer bumps the generation after its
        // bytes, with release. Then a generation that held means these are
        // its bytes. Relaxed alone lets a weakly ordered host (aarch64) read
        // a word before `gen0`, or see a bump before the bytes it follows,
        // and the snapshot pairs a generation with older bytes.
        fence(Ordering::Acquire);
        if !read() {
            return Snapshot::Unreadable;
        }
        fence(Ordering::Acquire);
        let gen1 = gen();
        if gen0 == gen1 {
            return Snapshot::Taken(gen1);
        }
    }
    Snapshot::Busy
}

/// The protocol under every interleaving of a CPU store, a DMA write and a
/// compile, checked by loom.
///
/// The compile runs [`seqlock_read`] over [`generation`] and publishes by
/// [`outdated`], and the DMA write bumps with [`bump`], as `PpMemory` and
/// Bedrock's memory do. The CPU store is the one transcription, and an
/// unavoidable one: it is the inline store `codegen::emit_mem_write_fast`
/// emits, and loom cannot run generated code. With one CPU that is a plain
/// load, add and store on the inline counter (a release store on a weakly
/// ordered host; on x86_64 a plain store is one). A page's two words stand
/// for its bytes.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    use loom::sync::Arc;
    use loom::thread;

    struct Page {
        words: [AtomicU32; 2],
        shared: AtomicU64,
        inline: AtomicU64,
    }

    impl Page {
        fn generation(&self) -> u64 {
            generation(&self.shared, &self.inline)
        }
        /// The compiled store: the bytes, then the inline counter.
        fn cpu_store(&self) {
            self.words[0].store(1, Ordering::Relaxed);
            let g = self.inline.load(Ordering::Relaxed);
            self.inline.store(g + 1, Ordering::Release);
        }
        fn dma_write(&self) {
            self.words[1].store(1, Ordering::Relaxed);
            bump(&self.shared);
        }
    }

    /// Snapshot the page and publish what it read.
    fn compile(p: &Page) -> Option<(u64, [u32; 2])> {
        let mut words = [0; 2];
        let Snapshot::Taken(g) = seqlock_read(|| p.generation(), || {
            words = [p.words[0].load(Ordering::Relaxed), p.words[1].load(Ordering::Relaxed)];
            true
        }) else {
            return None;
        };
        if outdated(p.generation(), g) {
            return None;
        }
        Some((g, words))
    }

    /// Once the writers are done, code that dispatch would still run must
    /// have been compiled from the bytes memory holds.
    #[test]
    fn a_compile_racing_a_cpu_store_and_a_dma_write_leaves_no_stale_code() {
        loom::model(|| {
            let p = Arc::new(Page {
                words: [AtomicU32::new(0), AtomicU32::new(0)],
                shared: AtomicU64::new(0),
                inline: AtomicU64::new(0),
            });
            let cpu = { let p = p.clone(); thread::spawn(move || p.cpu_store()) };
            let dma = { let p = p.clone(); thread::spawn(move || p.dma_write()) };
            let published = compile(&p);
            cpu.join().unwrap();
            dma.join().unwrap();
            if let Some((entry_gen, words)) = published {
                if entry_gen == p.generation() {
                    assert_eq!(words, [1, 1], "stale code is runnable: compiled from {words:?} at generation {entry_gen}");
                }
            }
        });
    }
}
