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
//! two; the high-water bit compares it against the mark the kernel sends
//! in `0x6b2`, and the low-water bit against the model's 16-word mark. The
//! ring under it holds `ODSY_WORK_QUEUE_MAX` entries, more than the board's
//! FIFO can, so it never fills: a processor store that would overfill the
//! board's FIFO is refused with the bus busy, and the processor retries it
//! while the board's thread makes room (`BoardLocks::bus_write`).
//!
//! The processor's stores are the producer; whoever runs the queue (the
//! board's thread, or an access that drains it) holds the board lock and is
//! the consumer. Graphics credits, which only machines whose processor
//! interface counts stores use, are settled as a store enters the FIFO's
//! first `ODSY_CFIFO_DEPTH` places: at the store when there is room, or
//! when the stores ahead of it have run.

use super::*;

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::SeqCst};
use std::thread::JoinHandle;

use parking_lot::{Condvar, Mutex};

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

pub struct CommandFifo {
    ring: Box<CommandRing>,
    /// Serialises the producers: the processor's front stores and stores
    /// made under the board lock.
    push: Mutex<()>,
    /// Serialises the consumers (the board lock already does for the queue's
    /// runners; reset's clear takes it too). Taken after `push` when both are.
    pop: Mutex<()>,
    /// The level: FIFO words pushed and not yet taken out.
    words: AtomicU32,
    /// Stores pushed and not yet taken out.
    stores: AtomicU32,
    /// Counted stores waiting; while zero no store needs credit work.
    counted: AtomicU32,
    /// The high-water mark from `0x6b2`, in words; zero until set up.
    high_mark: AtomicU32,
    /// The board's thread runs, so command stores queue.
    pub running: AtomicBool,
    /// The board's thread sleeps on `kick`.
    waiting: AtomicBool,
    /// The processor waits for the board's thread: its next slice starts
    /// without the pause for a burst of stores.
    hurry: AtomicBool,
    /// The board's thread is running stores under the board lock.
    pub slice: AtomicBool,
    exit: AtomicBool,
    wake: Mutex<()>,
    kick: Condvar,
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
            push: Mutex::new(()),
            pop: Mutex::new(()),
            words: AtomicU32::new(0),
            stores: AtomicU32::new(0),
            counted: AtomicU32::new(0),
            high_mark: AtomicU32::new(0),
            running: AtomicBool::new(false),
            waiting: AtomicBool::new(false),
            hurry: AtomicBool::new(false),
            slice: AtomicBool::new(false),
            exit: AtomicBool::new(false),
            wake: Mutex::new(()),
            kick: Condvar::new(),
            thread: Mutex::new(None),
            pf_level: crate::PfValue::new(),
            #[cfg(test)]
            test_words: AtomicU32::new(0),
        }
    }
}

impl CommandFifo {
    /// The FIFO words waiting: the level STATUS0 reports.
    pub fn level(&self) -> u32 {
        self.words.load(SeqCst)
    }

    /// The stores waiting.
    pub fn stores(&self) -> u32 {
        self.stores.load(SeqCst)
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
    /// the board's thread runs, so it must wait for room. The processor is
    /// the only producer, so room it sees stays until it stores.
    pub fn full_for(&self, size: u32) -> bool {
        self.running.load(SeqCst) && self.level() + store_words(size) > ODSY_CFIFO_WORDS
    }

    /// The processor waits for the board's thread to run what is queued.
    pub fn hurry(&self) {
        self.hurry.store(true, SeqCst);
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
        let _p = self.push.lock();
        let words = store_words(size);
        let before = self.level();
        if before + words > ODSY_CFIFO_WORDS {
            return false;
        }
        let mut q = QueuedStore { value, addr: addr as u32, size: size as u8, counted, owed: false };
        // A counted store's place is settled against the consumer's pops.
        let _c = counted.then(|| self.pop.lock());
        let credited = counted && self.stores() < ODSY_CFIFO_DEPTH;
        q.owed = counted && !credited;
        if !self.ring.try_push(q.entry(), q.value) {
            // The ring is deeper than the board's FIFO, and `push` keeps
            // this the only producer.
            debug_assert!(false, "the command ring refused a store the board's FIFO had room for");
            return false;
        }
        if counted {
            let mut f = front.lock();
            if credited {
                f.return_credit();
            } else if let Some(pi) = &f.pi {
                pi_gfx_owe_credits(pi.as_ref(), f.slot, 1);
            }
            self.counted.fetch_add(1, SeqCst);
        }
        self.stores.fetch_add(1, SeqCst);
        let after = self.words.fetch_add(words, SeqCst) + words;
        let mark = self.high_mark.load(SeqCst);
        if mark != 0 && before < mark && after >= mark {
            // Reaching the high-water mark raises its flag, whatever the
            // enable mask says.
            let mut f = front.lock();
            f.intr_flags |= ODSY_INTR_FIFO_HIGH_WATER;
            f.update_irq();
        }
        if self.waiting.load(SeqCst) {
            let _g = self.wake.lock();
            self.kick.notify_one();
        }
        true
    }

    /// The oldest waiting store, and the level after it left. The next
    /// waiting store moves into the FIFO's last credited place and is
    /// credited.
    pub fn pop(&self, front: &Mutex<HostInterface>) -> Option<(QueuedStore, u32)> {
        let _c = self.pop.lock();
        let (addr, value) = self.ring.peek()?;
        self.ring.consume();
        let q = QueuedStore::from_entry(addr, value);
        let after = self.words.fetch_sub(q.words(), SeqCst) - q.words();
        let stores_after = self.stores.fetch_sub(1, SeqCst) - 1;
        if q.counted {
            self.counted.fetch_sub(1, SeqCst);
        }
        if stores_after >= ODSY_CFIFO_DEPTH && self.counted.load(SeqCst) != 0 {
            if let Some((addr, value)) = self.ring.peek_nth_at(0, ODSY_CFIFO_DEPTH as usize - 1) {
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
        Some((q, after))
    }

    /// Abandon every waiting store; an owed one outside the FIFO gives its
    /// owed credit back.
    pub fn clear(&self, front: &Mutex<HostInterface>) {
        let _c = self.pop.lock();
        let mut index = 0u32;
        while let Some((addr, value)) = self.ring.peek() {
            self.ring.consume();
            let q = QueuedStore::from_entry(addr, value);
            self.words.fetch_sub(q.words(), SeqCst);
            self.stores.fetch_sub(1, SeqCst);
            if q.counted {
                self.counted.fetch_sub(1, SeqCst);
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

    /// The board's thread: sleep until a store waits or `stop` asks.
    /// Returns false when the thread is to finish.
    pub fn wait_for_work(&self) -> bool {
        let mut g = self.wake.lock();
        self.waiting.store(true, SeqCst);
        while !self.pending() && !self.exit.load(SeqCst) {
            self.kick.wait(&mut g);
        }
        self.waiting.store(false, SeqCst);
        !self.exit.load(SeqCst)
    }

    pub fn exiting(&self) -> bool {
        self.exit.load(SeqCst)
    }

    /// Ask the board's thread to finish.
    pub fn stop(&self) {
        self.exit.store(true, SeqCst);
        let _g = self.wake.lock();
        self.kick.notify_one();
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
