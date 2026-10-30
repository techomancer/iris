//! The command FIFO between the processor and the board's thread.
//!
//! The PROM's and IRIX's textports send hundreds of thousands of
//! command-FIFO words per scrolled line, so a store must cost little. A
//! store goes into a single-producer, single-consumer ring without taking
//! the host interface's lock; the board's thread is woken only when it is
//! asleep, and otherwise finds the store when it next looks. The level the
//! status register reports is the number of stores pushed and not yet
//! taken out, as the board's FIFO counts them.
//!
//! The processor's stores are the producer; whoever runs the queue (the
//! board's thread, or an access that drains it) holds the board lock and is
//! the consumer. Graphics credits, which only machines whose processor
//! interface counts stores use, are settled as a store enters the FIFO's
//! first `ODSY_CFIFO_DEPTH` places: at the store when there is room, or
//! when the stores ahead of it have run.

use super::*;

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::SeqCst};
use std::thread::JoinHandle;

use parking_lot::{Condvar, Mutex};
use rtrb::{Consumer, Producer, RingBuffer};

/// A command-FIFO store waiting for the board's thread.
#[derive(Clone, Copy)]
pub struct QueuedStore {
    pub value: u64,
    pub addr: u32,
    pub size: u8,
    /// The processor interface counted it: its credit is due.
    pub counted: bool,
    /// It waited outside the FIFO, so the node holds its credit as owed.
    pub owed: bool,
}

pub struct CommandFifo {
    producer: Mutex<Producer<QueuedStore>>,
    consumer: Mutex<Consumer<QueuedStore>>,
    /// Stores pushed and not yet taken out.
    level: AtomicU32,
    /// Counted stores waiting; while zero no store needs credit work.
    counted: AtomicU32,
    /// The board's thread runs, so command stores queue.
    pub running: AtomicBool,
    /// The board's thread sleeps on `kick`.
    waiting: AtomicBool,
    /// The board's thread is running stores under the board lock.
    pub slice: AtomicBool,
    exit: AtomicBool,
    wake: Mutex<()>,
    kick: Condvar,
    pub thread: Mutex<Option<JoinHandle<()>>>,
    /// Tests: queue every command store even with no thread running, and
    /// run this many per slice.
    #[cfg(test)]
    pub test_words: AtomicU32,
}

impl Default for CommandFifo {
    fn default() -> Self {
        let (producer, consumer) = RingBuffer::new(ODSY_WORK_QUEUE_MAX as usize);
        CommandFifo {
            producer: Mutex::new(producer),
            consumer: Mutex::new(consumer),
            level: AtomicU32::new(0),
            counted: AtomicU32::new(0),
            running: AtomicBool::new(false),
            waiting: AtomicBool::new(false),
            slice: AtomicBool::new(false),
            exit: AtomicBool::new(false),
            wake: Mutex::new(()),
            kick: Condvar::new(),
            thread: Mutex::new(None),
            #[cfg(test)]
            test_words: AtomicU32::new(0),
        }
    }
}

impl CommandFifo {
    /// Stores waiting: the level STATUS0 reports.
    pub fn level(&self) -> u32 {
        self.level.load(SeqCst)
    }

    pub fn pending(&self) -> bool {
        self.level() != 0
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
        self.level().min(ODSY_CFIFO_DEPTH)
    }

    /// Queue a store. A counted store is credited at once when it lands
    /// inside the FIFO and owed otherwise. False when the queue is full.
    pub fn push(&self, front: &Mutex<HostInterface>, addr: u64, value: u64, size: u32, counted: bool) -> bool {
        let mut p = self.producer.lock();
        let before = self.level();
        let mut q = QueuedStore { value, addr: addr as u32, size: size as u8, counted, owed: false };
        if counted {
            let mut f = front.lock();
            if before < ODSY_CFIFO_DEPTH {
                f.return_credit();
            } else if let Some(pi) = &f.pi {
                q.owed = true;
                pi_gfx_owe_credits(pi.as_ref(), f.slot, 1);
            }
            self.counted.fetch_add(1, SeqCst);
        }
        if p.push(q).is_err() {
            return false;
        }
        drop(p);
        self.level.fetch_add(1, SeqCst);
        if self.waiting.load(SeqCst) {
            let _g = self.wake.lock();
            self.kick.notify_one();
        }
        true
    }

    /// The oldest waiting store, and the level after it left. The next
    /// waiting store moves into the FIFO's last place and is credited.
    pub fn pop(&self, front: &Mutex<HostInterface>) -> Option<(QueuedStore, u32)> {
        let mut c = self.consumer.lock();
        let q = c.pop().ok()?;
        let after = self.level.fetch_sub(1, SeqCst) - 1;
        if q.counted {
            self.counted.fetch_sub(1, SeqCst);
        }
        if after >= ODSY_CFIFO_DEPTH && self.counted.load(SeqCst) != 0 {
            let place = ODSY_CFIFO_DEPTH as usize - 1;
            if let Ok(chunk) = c.read_chunk(place + 1) {
                let (a, b) = chunk.as_slices();
                let entered = if place < a.len() { a[place] } else { b[place - a.len()] };
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
        let mut c = self.consumer.lock();
        let mut index = 0u32;
        while let Ok(q) = c.pop() {
            self.level.fetch_sub(1, SeqCst);
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
