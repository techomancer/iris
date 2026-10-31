//! The command FIFO between the processor and the board's thread.
//!
//! The PROM's and IRIX's textports send hundreds of thousands of
//! command-FIFO words per scrolled line, so a store must cost little. A
//! store goes into the shared graphics ring (`GFifo`, gfifo.rs) without
//! taking the host interface's lock; the board's thread is woken only when
//! it is asleep, and otherwise finds the store when it next looks.
//!
//! The guest sees the board's FIFO, never the host ring. The board keeps
//! its command FIFO in its own memory, 32,768 words as the kernel sets it
//! up (`ODSY_CFIFO_WORDS`). The level STATUS0 reports is the 32-bit words
//! of the stores pushed and not yet taken out, a doubleword store counting
//! two, clamped to that 32,768 (`level`); the high-water bit compares it
//! against the mark the kernel sends in `0x6b2`, and the low-water bit
//! against the model's 16-word mark. Level 0 is the FIFO empty and every
//! store run.
//!
//! Where a store is refused is a model choice, not a recorded hardware
//! fact. The datasheet establishes the FIFO's size and marks; it does not
//! say what a full FIFO does to the processor, and it gives the FIFO no
//! wrap protection of its own, the kernel's flow control keeping it from
//! filling. On the Octane that flow control (the node's counted graphics
//! credits and the kernel's high-water path) is not modelled, so the old
//! stall at 32,768 words stood in for it. IMPACT and GR2 do not stall
//! there. Their host ring is deep, a store is refused with the bus busy
//! only when the ring is full, and the guest-visible level is the board's,
//! clamped to what the real FIFO can hold. The VPro follows them by
//! default (`FifoDepth::Deep`): the ring holds `ODSY_WORK_QUEUE_MAX`
//! entries, a store is one entry whatever its width, so it takes at least
//! twice the board's FIFO in words, and the processor runs ahead of the
//! board by that much. Everything the guest can read comes from the
//! clamped level, so it never sees a value the board cannot produce: past
//! the brim it reads a full, high-water FIFO while its stores still
//! succeed. `IRIS_VPRO_FIFO=shallow` restores the old stall: a store that
//! would overfill the board's 32,768 words is refused, and the processor
//! retries it while the board's thread makes room (`BoardLocks::bus_write`).
//! The ring is 64K entries as IMPACT's and GR2's are.
//!
//! The processor's stores are the producer; whoever runs the queue (the
//! board's thread, or an access that drains it) holds the board lock and is
//! the consumer. Graphics credits, which only machines whose processor
//! interface counts stores use, are settled as a store enters the FIFO's
//! first `ODSY_CFIFO_DEPTH` places: at the store when there is room, or
//! when the stores ahead of it have run.

use super::*;

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::{Relaxed, SeqCst}};
use std::thread::JoinHandle;

use crossbeam_utils::CachePadded;
use parking_lot::Mutex;

use crate::gfifo::GFifo;

/// The host ring under the command FIFO.
pub type CommandRing = GFifo<{ ODSY_WORK_QUEUE_MAX as usize }>;

/// A command-FIFO store waiting for the board's thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueuedStore {
    pub value: u64,
    pub addr: u32,
    pub size: u8,
    /// The processor interface counted it: its credit is due.
    pub counted: bool,
    /// It waited outside the FIFO, so the node holds its credit as owed.
    pub owed: bool,
}

// A ring entry is (address word, value). The address word carries the
// aperture offset in its low 24 bits and the store's size and credit
// state above them.
const ENTRY_OFFSET_MASK: u32 = 0x00ff_ffff;
const ENTRY_SIZE_SHIFT: u32 = 24;
const ENTRY_SIZE_MASK: u32 = 0xf;
const ENTRY_COUNTED: u32 = 1 << 28;
const ENTRY_OWED: u32 = 1 << 29;

impl QueuedStore {
    fn entry(&self) -> u32 {
        debug_assert!(self.addr <= ENTRY_OFFSET_MASK && self.size as u32 <= ENTRY_SIZE_MASK);
        self.addr
            | (self.size as u32) << ENTRY_SIZE_SHIFT
            | if self.counted { ENTRY_COUNTED } else { 0 }
            | if self.owed { ENTRY_OWED } else { 0 }
    }

    fn from_entry(addr: u32, value: u64) -> Self {
        QueuedStore {
            value,
            addr: addr & ENTRY_OFFSET_MASK,
            size: ((addr >> ENTRY_SIZE_SHIFT) & ENTRY_SIZE_MASK) as u8,
            counted: addr & ENTRY_COUNTED != 0,
            owed: addr & ENTRY_OWED != 0,
        }
    }

    /// The FIFO words the store fills: two for a doubleword.
    pub fn words(&self) -> u32 {
        store_words(self.size as u32)
    }
}

/// The 32-bit FIFO words a store of `size` bytes fills.
pub fn store_words(size: u32) -> u32 {
    if size == 8 { 2 } else { 1 }
}

/// The most stores the board's thread takes out of the FIFO at a time.
pub const RUN_MAX: usize = 32;

/// Stores the board's thread has looked at and is running (`take_run`).
pub struct Run {
    stores: [QueuedStore; RUN_MAX],
    len: usize,
}

impl Run {
    pub fn new() -> Self {
        Run { stores: [QueuedStore { value: 0, addr: 0, size: 0, counted: false, owed: false }; RUN_MAX], len: 0 }
    }

    /// The stores to run, oldest first.
    pub fn stores(&self) -> &[QueuedStore] {
        &self.stores[..self.len]
    }
}

/// Where a command store is refused with the bus busy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FifoDepth {
    /// Only when the host ring is full, as IMPACT and GR2 do.
    Deep,
    /// When the board's 32,768-word FIFO would overfill: the old stall.
    Shallow,
}

impl FifoDepth {
    /// `IRIS_VPRO_FIFO`: `shallow` or `32768` for the old stall, anything
    /// else, or unset, for the deep ring.
    pub fn from_env() -> Self {
        match std::env::var("IRIS_VPRO_FIFO").as_deref() {
            Ok("shallow") | Ok("32768") => FifoDepth::Shallow,
            _ => FifoDepth::Deep,
        }
    }
}

/// The stores the host ring holds: it keeps one slot empty.
pub const RING_STORES: u32 = ODSY_WORK_QUEUE_MAX - 1;

/// The counts the processor's pushes raise and the board's thread's
/// retirements lower.
struct Levels {
    /// The level: FIFO words pushed and not yet taken out.
    words: AtomicU32,
    /// Stores pushed and not yet taken out.
    stores: AtomicU32,
    /// Counted stores waiting; while zero no store needs credit work.
    counted: AtomicU32,
}

pub struct CommandFifo {
    ring: Box<CommandRing>,
    /// Serialises the consumers (the board lock already does for the queue's
    /// runners; reset's clear takes it too). The producers need no lock of
    /// their own: the ring's spinlock orders them.
    pop: Mutex<()>,
    /// What both sides count, on a line of its own.
    levels: CachePadded<Levels>,
    /// The high-water mark from `0x6b2`, in words; zero until set up. This
    /// and `running` are read on every store and written rarely, so each has
    /// a line no counter shares.
    high_mark: CachePadded<AtomicU32>,
    /// Where a store is refused (`FifoDepth`); read on every store, set at
    /// construction.
    deep: CachePadded<AtomicBool>,
    /// The board's thread runs, so command stores queue.
    pub running: CachePadded<AtomicBool>,
    /// The board's thread is parked (`wait_for_work`): a producer unparks it
    /// only then, so a store reads a line nobody writes instead of locking.
    asleep: CachePadded<AtomicBool>,
    /// The processor waits for the board's thread: its next slice starts
    /// without the pause for a burst of stores.
    hurry: CachePadded<AtomicBool>,
    /// The board's thread is running stores under the board lock.
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
}

impl Default for CommandFifo {
    fn default() -> Self {
        CommandFifo {
            // SAFETY: a zeroed GFifo is an empty one (GFifo::new); built in
            // place because the ring is too big for the stack.
            ring: unsafe { Box::new_zeroed().assume_init() },
            pop: Mutex::new(()),
            levels: CachePadded::new(Levels { words: AtomicU32::new(0), stores: AtomicU32::new(0), counted: AtomicU32::new(0) }),
            high_mark: CachePadded::new(AtomicU32::new(0)),
            deep: CachePadded::new(AtomicBool::new(FifoDepth::from_env() == FifoDepth::Deep)),
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
        }
    }
}

impl CommandFifo {
    /// The FIFO words waiting, as the guest sees them: the board's FIFO
    /// holds no more than `ODSY_CFIFO_WORDS`, so a deeper backlog in the
    /// host ring reads as a full FIFO. Zero only when nothing waits.
    pub fn level(&self) -> u32 {
        self.queued_words().min(ODSY_CFIFO_WORDS)
    }

    /// The FIFO words waiting in the host ring, unclamped.
    pub fn queued_words(&self) -> u32 {
        self.levels.words.load(SeqCst)
    }

    /// Choose where a store is refused (`FifoDepth`).
    pub fn set_depth(&self, depth: FifoDepth) {
        self.deep.store(depth == FifoDepth::Deep, SeqCst);
    }

    /// Whether the FIFO has room for a store of `words` words.
    fn has_room(&self, words: u32) -> bool {
        if self.deep.load(Relaxed) {
            self.stores() < RING_STORES
        } else {
            self.queued_words() + words <= ODSY_CFIFO_WORDS
        }
    }

    /// The stores waiting.
    pub fn stores(&self) -> u32 {
        self.levels.stores.load(SeqCst)
    }

    pub fn pending(&self) -> bool {
        self.level() != 0
    }

    /// Whether the level is at or above a set-up high-water mark.
    pub fn high_water(&self) -> bool {
        let mark = self.high_mark.load(SeqCst);
        mark != 0 && self.level() >= mark
    }

    /// The kernel's `0x6b2`: the high-water mark, in words.
    pub fn set_high_mark(&self, words: u32) {
        self.high_mark.store(words, SeqCst);
    }

    /// Whether a store of `size` bytes finds the board's FIFO full while
    /// the board's thread runs, so it must wait for room. With two
    /// processors the room it sees can go to the other one's store first;
    /// `push` then refuses, and the store is not lost.
    pub fn full_for(&self, size: u32) -> bool {
        self.running.load(SeqCst) && !self.has_room(store_words(size))
    }

    /// The processor waits for the board's thread to run what is queued.
    pub fn hurry(&self) {
        // A refused store retries in a loop; write the line only when it
        // changes, as IMPACT does for its asleep flag.
        if !self.hurry.load(SeqCst) {
            self.hurry.store(true, SeqCst);
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

    /// The stores in the FIFO's first places, credited.
    pub fn in_fifo(&self) -> u32 {
        self.stores().min(ODSY_CFIFO_DEPTH)
    }

    /// Queue a store. A counted store is credited at once when it lands
    /// inside the FIFO's first places and owed otherwise. False when the
    /// board's FIFO has no room for it.
    pub fn push(&self, front: &Mutex<HostInterface>, addr: u64, value: u64, size: u32, counted: bool) -> bool {
        let words = store_words(size);
        if !self.has_room(words) {
            return false;
        }
        let before = self.queued_words();
        let mut q = QueuedStore { value, addr: addr as u32, size: size as u8, counted, owed: false };
        // A counted store's place is settled against the consumer's pops.
        let _c = counted.then(|| self.pop.lock());
        let credited = counted && self.stores() < ODSY_CFIFO_DEPTH;
        q.owed = counted && !credited;
        if !self.ring.try_push(q.entry(), q.value) {
            // With one processor this cannot happen: has_room counted the
            // ring's stores and nothing else pushes. With two, the other
            // processor's store can hold the ring's lock, or take the last
            // place between has_room and here. Refused, the store goes the
            // board-lock way (`write_command`), which runs what waits and
            // then the store.
            return false;
        }
        if counted {
            let mut f = front.lock();
            if credited {
                f.return_credit();
            } else if let Some(pi) = &f.pi {
                pi_gfx_owe_credits(pi.as_ref(), f.slot, 1);
            }
            self.levels.counted.fetch_add(1, SeqCst);
        }
        self.levels.stores.fetch_add(1, SeqCst);
        // The board's thread may retire this store before the level counts it,
        // which wraps the level for a moment; the sum wraps back.
        let after = self.levels.words.fetch_add(words, SeqCst).wrapping_add(words);
        let mark = self.high_mark.load(SeqCst);
        if mark != 0 && before < mark && after >= mark {
            // Reaching the high-water mark raises its flag, whatever the
            // enable mask says.
            let mut f = front.lock();
            f.intr_flags |= ODSY_INTR_FIFO_HIGH_WATER;
            f.update_irq();
        }
        self.wake();
        true
    }

    /// Look at the oldest waiting stores, up to `RUN_MAX`, without taking
    /// them out; the caller runs them, then `retire`s the ones it ran. While
    /// a counted store waits the run is one store, so each credit is settled
    /// as that store leaves, as it always was.
    pub fn take_run(&self, run: &mut Run) {
        let max = if self.levels.counted.load(SeqCst) != 0 { 1 } else { RUN_MAX };
        let mut raw = [(0u32, 0u64); RUN_MAX];
        run.len = self.ring.peek_run_at(0, &mut raw[..max]);
        for (store, (addr, value)) in run.stores.iter_mut().zip(&raw[..run.len]) {
            *store = QueuedStore::from_entry(*addr, *value);
        }
    }

    /// Take the first `n` stores of `run` out of the FIFO: they have run. The
    /// level, the stores waiting and the credits move once for the run, not
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
        let counted_ran = ran.iter().filter(|q| q.counted).count() as u32;
        let _c = self.pop.lock();
        self.ring.consume_n_at(0, n);
        if self.levels.words.load(SeqCst) < ODSY_CFIFO_LOW_WATER + words {
            let mut f = front.lock();
            let before = self.levels.words.fetch_sub(words, SeqCst);
            if before >= ODSY_CFIFO_LOW_WATER && before - words < ODSY_CFIFO_LOW_WATER {
                f.intr_flags |= ODSY_INTR_FIFO_LOW_WATER;
                crate::pf_instant!(crate::perfetto::events::ODSY_CFIFO_LOW_WATER, before - words);
                f.update_irq();
            }
        } else {
            self.levels.words.fetch_sub(words, SeqCst);
        }
        let stores_before = self.levels.stores.fetch_sub(n as u32, SeqCst);
        if counted_ran != 0 {
            self.levels.counted.fetch_sub(counted_ran, SeqCst);
        }
        if self.levels.counted.load(SeqCst) == 0 {
            return;
        }
        // The next waiting store moves into the FIFO's last credited place
        // and is credited.
        for k in 1..=n as u32 {
            if stores_before - k < ODSY_CFIFO_DEPTH {
                continue;
            }
            // The run is already out of the ring, so count back from its head.
            let place = ODSY_CFIFO_DEPTH as usize - 1 - (n - k as usize);
            if let Some((addr, value)) = self.ring.peek_nth_at(0, place) {
                let entered = QueuedStore::from_entry(addr, value);
                let mut f = front.lock();
                if entered.owed {
                    if let Some(pi) = &f.pi {
                        pi_gfx_repay_credits(pi.as_ref(), f.slot, 1);
                    }
                } else if entered.counted {
                    f.return_credit();
                }
            }
        }
    }

    /// Abandon every waiting store; an owed one outside the FIFO gives its
    /// owed credit back.
    pub fn clear(&self, front: &Mutex<HostInterface>) {
        let _c = self.pop.lock();
        let mut index = 0u32;
        while let Some((addr, value)) = self.ring.peek() {
            self.ring.consume();
            let q = QueuedStore::from_entry(addr, value);
            self.levels.words.fetch_sub(q.words(), SeqCst);
            self.levels.stores.fetch_sub(1, SeqCst);
            if q.counted {
                self.levels.counted.fetch_sub(1, SeqCst);
            }
            if q.owed && index >= ODSY_CFIFO_DEPTH {
                let f = front.lock();
                if let Some(pi) = &f.pi {
                    pi_gfx_owe_credits(pi.as_ref(), f.slot, -1);
                }
            }
            index += 1;
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
pub fn display_fifo_pending() -> usize {
    DISPLAY_FIFO_PENDING.with(Cell::get)
}

/// Record the words waiting for this display thread's next present.
pub fn set_display_fifo_pending(words: usize) {
    DISPLAY_FIFO_PENDING.with(|p| p.set(words));
}
