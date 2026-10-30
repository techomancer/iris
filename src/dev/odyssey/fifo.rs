//! The command FIFO between the processor and the board's thread.
//!
//! The PROM's and IRIX's textports send hundreds of thousands of
//! command-FIFO words per scrolled line, so a store must cost little. A
//! store goes into the shared graphics ring (`GFifo`, gfifo.rs) without
//! taking the host interface's lock; the board's thread is woken only when
//! it is asleep, and otherwise finds the store when it next looks.
//!
//! The guest sees the board's FIFO, never the host ring. The FIFO's depth,
//! in any unit, is UNKNOWN; the
//! model gives it a size of 32,768 words (`cfifo::WORDS`), a MODEL
//! CHOICE independent of the `0x6bX` values the kernel stores. The level
//! STATUS0 reports is the 32-bit words of the stores pushed and not yet
//! taken out, a doubleword store counting two, clamped to that 32,768
//! (`level`); the high-water bit compares it against the `0x6b2` value
//! (a model choice: the register's role is unknown), and the low-water bit
//! against the model's 16-word threshold. Level 0 is the FIFO empty and
//! every store run.
//!
//! Where a store is refused is a model choice, not a recorded hardware
//! fact. What a full FIFO does to the processor is not known (software is
//! inferred to avoid overflow by credits and descheduling, and a board-side
//! overflow to be a flagged error), and the FIFO has no wrap protection of
//! its own, the kernel's flow control keeping it from filling. On the
//! Octane that flow control (the node's counted graphics credits and the
//! kernel's high-water path) is not modelled. IMPACT and GR2 do not stall
//! at their FIFO's size either: their host ring is deep, a store is refused
//! with the bus busy only when the ring is full, and the guest-visible
//! level is the model's, clamped to the model's FIFO size (a model choice,
//! `cfifo::WORDS`). The VPro does the same: the ring holds
//! `ODSY_WORK_QUEUE_MAX` entries, a store is one entry whatever its width,
//! so it takes at least twice the model's FIFO in words, and the processor
//! runs ahead of the board by that much. Everything the guest can read
//! comes from the clamped level, so it never sees a level above the
//! model's size: past the brim it reads a full FIFO (and a high-water bit,
//! once a `0x6b2` threshold is set) while its stores still succeed. A store
//! the full ring refuses is retried by the processor while the board's
//! thread makes room (`BoardLocks::bus_write`). The ring is 64K entries as
//! IMPACT's and GR2's are.
//!
//! The processor's stores are the producer; whoever runs the queue (the
//! board's thread, or an access that drains it with that thread stopped)
//! is the consumer.


use super::consts::{cfifo, intr, ODSY_WORK_QUEUE_MAX};
use super::interface::HostInterface;
use super::worker::GAP;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::{Relaxed, SeqCst}};
use std::thread::JoinHandle;

use crossbeam_utils::CachePadded;
use parking_lot::Mutex;

use crate::gfifo::GFifo;

/// The host ring under the command FIFO.
pub(super) type CommandRing = GFifo<{ ODSY_WORK_QUEUE_MAX as usize }>;

/// A command-FIFO store waiting for the board's thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct QueuedStore {
    pub value: u64,
    pub addr: u32,
    pub size: u8,
}

// A ring entry is (address word, value). The address word carries the
// aperture offset in its low 24 bits and the store's size above them.
const ENTRY_OFFSET_MASK: u32 = 0x00ff_ffff;
const ENTRY_SIZE_SHIFT: u32 = 24;
const ENTRY_SIZE_MASK: u32 = 0xf;

impl QueuedStore {
    fn entry(&self) -> u32 {
        debug_assert!(self.addr <= ENTRY_OFFSET_MASK && self.size as u32 <= ENTRY_SIZE_MASK);
        self.addr | (self.size as u32) << ENTRY_SIZE_SHIFT
    }

    fn from_entry(addr: u32, value: u64) -> Self {
        QueuedStore {
            value,
            addr: addr & ENTRY_OFFSET_MASK,
            size: ((addr >> ENTRY_SIZE_SHIFT) & ENTRY_SIZE_MASK) as u8,
        }
    }

    /// The FIFO words the store fills: two for a doubleword.
    pub fn words(&self) -> u32 {
        store_words(self.size as u32)
    }
}

/// The 32-bit FIFO words a store of `size` bytes fills.
pub(super) fn store_words(size: u32) -> u32 {
    if size == 8 { 2 } else { 1 }
}

/// The most stores the board's thread takes out of the FIFO at a time.
pub(super) const RUN_MAX: usize = 32;

/// Stores the board's thread has looked at and is running (`take_run`).
pub(super) struct Run {
    stores: [QueuedStore; RUN_MAX],
    len: usize,
}

impl Run {
    pub fn new() -> Self {
        Run { stores: [QueuedStore { value: 0, addr: 0, size: 0 }; RUN_MAX], len: 0 }
    }

    /// The stores to run, oldest first.
    pub fn stores(&self) -> &[QueuedStore] {
        &self.stores[..self.len]
    }
}

/// The stores the host ring holds: it keeps one slot empty.
pub(super) const RING_STORES: u32 = ODSY_WORK_QUEUE_MAX - 1;

/// The counts the processor's pushes raise and the board's thread's
/// retirements lower.
struct Levels {
    /// The level: FIFO words pushed and not yet taken out.
    words: AtomicU32,
    /// Stores pushed and not yet taken out.
    stores: AtomicU32,
}

pub(super) struct CommandFifo {
    ring: Box<CommandRing>,
    /// Serialises the consumers' accounting against reset's clear. The
    /// producers need no lock of their own: the ring's spinlock orders them.
    pop: Mutex<()>,
    /// What both sides count, on a line of its own.
    levels: CachePadded<Levels>,
    /// The threshold for status bit 15 and flag 25 from `0x6b2`, in words
    /// (a model choice: the register's role is unknown); zero until set up. This
    /// and `running` are read on every store and written rarely, so each has
    /// a line no counter shares.
    threshold_6b2: CachePadded<AtomicU32>,
    /// The board's thread runs, so command stores queue.
    pub running: CachePadded<AtomicBool>,
    /// The board's thread is parked (`wait_for_work`): a producer unparks it
    /// only then, so a store reads a line nobody writes instead of locking.
    asleep: CachePadded<AtomicBool>,
    /// The processor waits for the board's thread, for a refused access: its
    /// next slice starts without the pause for a burst of stores, and a
    /// pause under way ends.
    hurry: CachePadded<AtomicBool>,
    /// The board's thread is running stores.
    pub slice: CachePadded<AtomicBool>,
    exit: AtomicBool,
    sleeper: Mutex<Option<std::thread::Thread>>,
    pub thread: Mutex<Option<JoinHandle<()>>>,
    /// Edge filter for the traced level; zero-sized without the feature.
    pub pf_level: crate::PfValue,
    /// Tests: queue every command store even with no thread running, and
    /// run this many per slice.
    #[cfg(test)]
    pub test_words: AtomicU32,
    /// Tests: the board's thread's pause in milliseconds instead of `GAP`.
    #[cfg(test)]
    pub test_gap_ms: AtomicU32,
    /// Tests: how many times the board's thread has gone to sleep in `gap`,
    /// counted after its last look at the hurry request.
    #[cfg(test)]
    pub test_parked: AtomicU32,
    /// Tests: the next push calls this, once, as soon as its
    /// store is in the ring: a consumer that takes the store at that
    /// moment, before the push has returned.
    #[cfg(test)]
    pub test_after_ring_push: Mutex<Option<fn(&CommandFifo, &Mutex<HostInterface>)>>,
}

impl Default for CommandFifo {
    fn default() -> Self {
        CommandFifo {
            // SAFETY: a zeroed GFifo is an empty one (GFifo::new); built in
            // place because the ring is too big for the stack.
            ring: unsafe { Box::new_zeroed().assume_init() },
            pop: Mutex::new(()),
            levels: CachePadded::new(Levels { words: AtomicU32::new(0), stores: AtomicU32::new(0) }),
            threshold_6b2: CachePadded::new(AtomicU32::new(0)),
            running: CachePadded::new(AtomicBool::new(false)),
            asleep: CachePadded::new(AtomicBool::new(false)),
            hurry: CachePadded::new(AtomicBool::new(false)),
            slice: CachePadded::new(AtomicBool::new(false)),
            exit: AtomicBool::new(false),
            sleeper: Mutex::new(None),
            thread: Mutex::new(None),
            pf_level: crate::PfValue::new(),
            #[cfg(test)]
            test_words: AtomicU32::new(0),
            #[cfg(test)]
            test_gap_ms: AtomicU32::new(0),
            #[cfg(test)]
            test_parked: AtomicU32::new(0),
            #[cfg(test)]
            test_after_ring_push: Mutex::new(None),
        }
    }
}

impl CommandFifo {
    /// The FIFO words waiting, as the guest sees them: the model's FIFO
    /// holds no more than `cfifo::WORDS` (a model choice), so a deeper
    /// backlog in the host ring reads as a full FIFO. Zero only when nothing waits.
    pub fn level(&self) -> u32 {
        self.queued_words().min(cfifo::WORDS)
    }

    /// The FIFO words waiting in the host ring, unclamped.
    pub fn queued_words(&self) -> u32 {
        self.levels.words.load(SeqCst)
    }

    /// Whether the host ring has room for another store.
    fn has_room(&self) -> bool {
        self.stores() < RING_STORES
    }

    /// The stores waiting.
    pub fn stores(&self) -> u32 {
        self.levels.stores.load(SeqCst)
    }

    pub fn pending(&self) -> bool {
        self.level() != 0
    }

    /// Whether the level is at or above the model's high-water threshold.
    pub fn high_water(&self) -> bool {
        let mark = self.threshold_6b2.load(SeqCst);
        mark != 0 && self.level() >= mark
    }

    /// The stored `0x6b2`, applied as the high-water threshold in words. MODEL
    /// CHOICE (see `reg::FIFO_SETUP_FIRST`).
    pub fn set_threshold_6b2(&self, words: u32) {
        self.threshold_6b2.store(words, SeqCst);
    }

    /// Whether a store of `size` bytes finds the board's FIFO full while
    /// the board's thread runs, so it must wait for room. With two
    /// processors the room it sees can go to the other one's store first;
    /// `push` then refuses, and the store is not lost.
    pub fn full_for(&self, size: u32) -> bool {
        self.running.load(SeqCst) && !self.has_room()
    }

    /// The processor waits for the board's thread to run what is queued, as
    /// a refused access does. A thread already in its pause (`gap`) is woken
    /// to run at once.
    pub fn hurry(&self) {
        // A refused store retries in a loop; write the line only when it
        // changes, as IMPACT does for its asleep flag.
        if !self.hurry.load(SeqCst) {
            self.hurry.store(true, SeqCst);
            if let Some(t) = self.sleeper.lock().as_ref() {
                t.unpark();
            }
        }
    }

    /// The board's thread's pause before a slice (`GAP`), cut short when the
    /// processor asks it to hurry.
    pub fn gap(&self) {
        #[cfg(test)]
        let gap = match self.test_gap_ms.load(SeqCst) {
            0 => GAP,
            ms => std::time::Duration::from_millis(ms as u64),
        };
        #[cfg(not(test))]
        let gap = GAP;
        let end = std::time::Instant::now() + gap;
        loop {
            if self.take_hurry() {
                return;
            }
            let now = std::time::Instant::now();
            if now >= end {
                return;
            }
            #[cfg(test)]
            self.test_parked.fetch_add(1, SeqCst);
            std::thread::park_timeout(end - now);
        }
    }

    /// Whether the board's thread's next slice is awaited, clearing it.
    pub fn take_hurry(&self) -> bool {
        self.hurry.swap(false, SeqCst)
    }

    /// Tests: every command store queues, to be run by hand.
    pub fn held(&self) -> bool {
        #[cfg(test)]
        return self.test_words.load(SeqCst) != 0;
        #[cfg(not(test))]
        false
    }

    /// Whether a command store waits in the queue: always while the board's
    /// thread runs, so decoding and drawing never run on the processor's
    /// thread, and otherwise behind any store that already waits.
    pub fn queues(&self) -> bool {
        self.running.load(SeqCst) || self.pending() || self.held()
    }

    /// Queue a store. False when the host ring has no room for it.
    pub fn push(&self, front: &Mutex<HostInterface>, addr: u64, value: u64, size: u32) -> bool {
        let words = store_words(size);
        if !self.has_room() {
            return false;
        }
        let q = QueuedStore { value, addr: addr as u32, size: size as u8 };
        // Count the store before the ring shows it. The board's thread can
        // take it and retire it the moment it is in the ring, and a retire
        // must find every store it takes already counted, or the level falls
        // below zero. Until the push lands the counts are a store high,
        // which reads as a store still waiting, never as one gone.
        self.levels.stores.fetch_add(1, SeqCst);
        let before = self.levels.words.fetch_add(words, SeqCst);
        if !self.ring.try_push(q.entry(), q.value) {
            // With one processor this cannot happen: has_room counted the
            // ring's stores and nothing else pushes. With two, the other
            // processor's store can hold the ring's lock, or take the last
            // place between has_room and here. Refused, the store is pushed
            // again (`write_command`). Nothing took it out, so its counts
            // come back off.
            self.levels.words.fetch_sub(words, SeqCst);
            self.levels.stores.fetch_sub(1, SeqCst);
            return false;
        }
        #[cfg(test)]
        if let Some(consumer) = self.test_after_ring_push.lock().take() {
            consumer(self, front);
        }
        let after = before + words;
        let mark = self.threshold_6b2.load(SeqCst);
        if mark != 0 && before < mark && after >= mark {
            // Reaching the model's 0x6b2 threshold raises flag 25, whatever
            // the enable mask says.
            let mut f = front.lock();
            f.intr_flags |= intr::FIFO_HIGH_WATER;
            f.update_irq();
        }
        self.wake();
        true
    }

    /// Look at the oldest waiting stores, up to `RUN_MAX`, without taking
    /// them out; the caller runs them, then `retire`s the ones it ran.
    pub fn take_run(&self, run: &mut Run) {
        let mut raw = [(0u32, 0u64); RUN_MAX];
        run.len = self.ring.peek_run_at(0, &mut raw);
        for (store, (addr, value)) in run.stores.iter_mut().zip(&raw[..run.len]) {
            *store = QueuedStore::from_entry(*addr, *value);
        }
    }

    /// Take the first `n` stores of `run` out of the FIFO: they have run. The
    /// level and the stores waiting move once for the run, not
    /// once per store, which is what the board's thread spends most of its
    /// time on when each store costs little to decode. A store the thread has
    /// run but not retired still counts in the level, so the level never
    /// reads lower than the words the board has yet to take.
    ///
    /// The low-water flag latches if the run takes the level from the mark
    /// or above to below it. The flag is set before the level falls, under the
    /// front lock a status read also takes, so a read sees the flag whenever
    /// it sees the level below the mark.
    pub fn retire(&self, front: &Mutex<HostInterface>, run: &Run, n: usize) {
        if n == 0 {
            return;
        }
        let ran = &run.stores[..n];
        let words: u32 = ran.iter().map(QueuedStore::words).sum();
        let _c = self.pop.lock();
        self.ring.consume_n_at(0, n);
        if self.levels.words.load(SeqCst) < cfifo::LOW_WATER + words {
            let mut f = front.lock();
            let before = self.levels.words.fetch_sub(words, SeqCst);
            debug_assert!(before >= words, "the level counts every store before the ring shows it (push)");
            if before >= cfifo::LOW_WATER && before - words < cfifo::LOW_WATER {
                f.intr_flags |= intr::FIFO_LOW_WATER;
                crate::pf_instant!(crate::perfetto::events::cfifo::LOW_WATER, before - words);
                f.update_irq();
            }
        } else {
            self.levels.words.fetch_sub(words, SeqCst);
        }
        self.levels.stores.fetch_sub(n as u32, SeqCst);
    }

    /// Abandon every waiting store.
    pub fn clear(&self) {
        let _c = self.pop.lock();
        while let Some((addr, value)) = self.ring.peek() {
            self.ring.consume();
            let q = QueuedStore::from_entry(addr, value);
            self.levels.words.fetch_sub(q.words(), SeqCst);
            self.levels.stores.fetch_sub(1, SeqCst);
        }
    }

    /// After a push or a stop: unpark the board's thread if it is asleep.
    /// The fence pairs with `wait_for_work`'s: either this sees the thread
    /// asleep, or the thread sees the push before it parks.
    fn wake(&self) {
        std::sync::atomic::fence(SeqCst);
        if self.asleep.load(Relaxed) {
            if let Some(t) = self.sleeper.lock().as_ref() {
                t.unpark();
            }
        }
    }

    /// The board's thread: sleep until a store waits or `stop` asks.
    /// Returns false when the thread is to finish. The timeout is only a
    /// backstop for a wake the fence protocol should never lose.
    pub fn wait_for_work(&self) -> bool {
        *self.sleeper.lock() = Some(std::thread::current());
        while !self.pending() && !self.exit.load(SeqCst) {
            self.asleep.store(true, Relaxed);
            std::sync::atomic::fence(SeqCst);
            if !self.pending() && !self.exit.load(SeqCst) {
                std::thread::park_timeout(std::time::Duration::from_millis(50));
            }
            self.asleep.store(false, Relaxed);
        }
        !self.exit.load(SeqCst)
    }

    pub fn exiting(&self) -> bool {
        self.exit.load(SeqCst)
    }

    /// Ask the board's thread to finish.
    pub fn stop(&self) {
        self.exit.store(true, SeqCst);
        self.wake();
    }

    pub fn start(&self) {
        self.exit.store(false, SeqCst);
    }
}

thread_local! {
    /// The command words waiting, as the display thread last sampled them
    /// for its status bar (Odyssey::frame). Per thread, because each board
    /// presents from a display thread of its own.
    static DISPLAY_FIFO_PENDING: Cell<usize> = const { Cell::new(0) };
}

/// The display thread's sample of the words waiting.
pub(super) fn display_fifo_pending() -> usize {
    DISPLAY_FIFO_PENDING.with(Cell::get)
}

/// Record the words waiting for this display thread's next present.
pub(super) fn set_display_fifo_pending(words: usize) {
    DISPLAY_FIFO_PENDING.with(|p| p.set(words));
}
