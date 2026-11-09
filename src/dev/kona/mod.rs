//! The InfiniteReality (KONA) graphics pipe's front end: the host interface.
//!
//! The pipe is a crosstalk-style widget. The processor stores commands to a
//! command page; a Host Interface Processor on the board takes them, answers
//! the ones that are about the board itself and passes the ones that draw to
//! the Geometry Distributor, Geometry Engines, Raster Managers and Display
//! Generator behind it. This module is the first part: the widget header and
//! identity registers, the one-wire identity record, the board's flash
//! EEPROM and the operation the graphics install tool programs it with,
//! reset, the command FIFO, command framing, DMA and display-list storage.
//! Drawing is behind the typed job interface in `job`.
//!
//! The machine gives the board an `InterruptController` (each interrupt packet
//! is one `set_line(vector, true)`, a set of the pending bit and never an edge)
//! and a `BusDevice` for host memory; it reaches the board through the
//! `BusDevice` impl, with offsets inside the 16 MiB aperture.
//!
//! - `front`: the registers a processor reaches without waiting for the
//!   command stream, behind one short lock.
//! - `regs`: which accesses join the stream, which are served at once and
//!   which wait; the registers the Host Interface Processor owns.
//! - `pipe`: the Host Interface Processor's state and the command stream it
//!   runs (framers, pages, contexts, reasons, swaps).
//! - `frame`, `tables`: command-word framing and the selector tables.
//! - `dma`: DMA packets, display-list memory and list invocation.
//! - `flash`: the flash EEPROM and its programming operation.
//! - `ident`: widget id, one-wire record, configuration block.
//! - `job`: the back end's interface.
//! - `back`: the back end that draws and displays.
//! - `handle`: what a back end's display thread may call on the front end.
//!
//! # Threads
//!
//! One thread, `kona-hip`, stands for the Host Interface Processor. The
//! processor stores into a command FIFO and goes on, as it does on the
//! board, and a display list the processor invokes can run for as long as
//! its words take to decode. A store to the command page (and the few other
//! stream registers) goes into the shared graphics ring (`GFifo`) with no
//! lock beyond the ring's; a full ring answers the bus busy, never a spin on
//! the processor's thread. The thread takes a run of stores at a time and
//! runs them in order. A register the stream changes or the stream reads
//! (`regs::classify`) is answered busy until the ring is empty and the thread
//! idle; a register the driver polls while the pipe works is served from the
//! register file at once.
//!
//! Without the thread running (before `start`) a stream store waits in the ring;
//! only the test-only `test_write` runs it on the caller's thread.
//!
//! # Locks
//!
//! `Host::front` guards the register file and is held for a few instructions,
//! never across a call out of the board. `Shared::pipe` is a mutex over the Host
//! Interface Processor's state and the back end: `kona-hip` holds it for each
//! batch of up to 32 stores, across calls to the machine's memory and interrupt
//! controller (nothing the machine calls takes it back), and a processor only
//! `try_lock`s it, for a read that waits for the stream, answering busy when it
//! is out. Stores, including the ones that wait for the stream and the register
//! writes that change the pipe's state, go into the ring and are applied by the
//! thread in order. The mutex is registered with the lock monitor.
//!
//! Ported from hw/misc/sgi-kona.c, sgi-kona-ports.c, sgi-kona-dma.c and
//! sgi-command-port.c in jamesbraid/qemu-sgi-origami.

// The crate allows dead code; this board's code does not hide any.
#![warn(dead_code, unused_imports)]

mod back;
pub mod stream;
mod defs;
mod dma;
mod flash;
mod frame;
mod front;
mod handle;
mod ident;
mod job;
mod pipe;
mod regs;
mod tables;
#[cfg(test)]
mod tests;

// What a machine needs: the board, its configuration and the flash's size.
pub use back::snapshot::KonaChunk;
pub use flash::{flash_file, FLASH_SIZE};
pub use ident::{KonaConfig, Product as KonaProduct};
use back::KonaBackend;

#[cfg(test)]
use job::{ContextEvent, Job, Reply};

use defs::APERTURE;
use handle::DisplayHandle;
use pipe::LoadSection;

use crate::gfifo::GFifo;
use crate::irq::InterruptController;
use crate::traits::{BusDevice, BusRead32, BusRead64, Device, Resettable, BUS_BUSY, BUS_ERR, BUS_OK};
use front::{Front, Host};
use parking_lot::Mutex;
use pipe::Core;
use regs::{classify, Access};
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::{Relaxed, SeqCst}};
use std::sync::Arc;
use std::thread::JoinHandle;

/// The host ring under the command stream: 64K stores, as the other
/// graphics boards' rings are.
type Ring = GFifo<{ 1 << 16 }>;

/// The most stores the thread looks at before running them.
const RUN_MAX: usize = 32;

/// The Host Interface Processor's state with the back end it hands jobs to.
struct Pipe {
    core: Core,
    backend: KonaBackend,
}

/// The longest a monitor command that reads state waits for the command stream.
const MONITOR_WAIT: std::time::Duration = std::time::Duration::from_millis(500);

struct Shared {
    /// Its own `Arc` so a back end's display thread can hold it weakly: the
    /// register file does not own the back end, the pipe does.
    host: Arc<Host>,
    /// The Host Interface Processor's state with the back end. `kona-hip` holds
    /// it for each batch of stores, across calls to the machine's memory and
    /// interrupt controller; that cannot deadlock, because processor accesses
    /// only `try_lock` it, the machine never calls into the board, and nothing
    /// takes it while holding another lock. A plain mutex because no processor
    /// access needs the pipe for more than a pure read.
    pipe: Mutex<Pipe>,
    ring: Box<Ring>,
    /// The back end's pixel reads, which need not wait for the state.
    pixels: back::display::Pixels,
    /// The window the back end draws the screen into.
    display: Arc<dyn crate::gfx_display::GfxDisplay>,
    /// How long a monitor command that reads state waits for it.
    monitor_wait: AtomicU64,
    /// The thread died (its back end panicked): bus accesses fail.
    dead: AtomicBool,
    /// The thread runs, so stream stores are taken from the ring.
    running: AtomicBool,
    exit: AtomicBool,
    /// The thread is parked in `wait_for_work`: a producer wakes it only
    /// then, so a store reads a line nobody writes instead of locking.
    asleep: AtomicBool,
    sleeper: Mutex<Option<std::thread::Thread>>,
    /// A cause-register access found completion reasons waiting behind an
    /// unread one: the thread posts them, not the processor.
    reason_poke: AtomicBool,
}

impl Shared {
    fn wake(&self) {
        // Store the entry, then look at `asleep`; the thread stores `asleep`,
        // then looks at the ring. Both sides need the fence for either to
        // see the other (a store-then-load handshake).
        std::sync::atomic::fence(SeqCst);
        if self.asleep.load(SeqCst) {
            if let Some(t) = self.sleeper.lock().as_ref() {
                t.unpark();
            }
        }
    }

    /// The thread: sleep until a store waits or `stop` asks. False when the
    /// thread is to finish. The timeout is only a backstop for a wake the
    /// fences should never lose.
    fn wait_for_work(&self) -> bool {
        *self.sleeper.lock() = Some(std::thread::current());
        while self.ring.is_empty() && !self.exit.load(SeqCst) && !self.reason_poke.load(SeqCst) {
            self.asleep.store(true, Relaxed);
            std::sync::atomic::fence(SeqCst);
            if self.ring.is_empty() && !self.exit.load(SeqCst) && !self.reason_poke.load(SeqCst) {
                std::thread::park_timeout(std::time::Duration::from_millis(50));
            }
            self.asleep.store(false, Relaxed);
        }
        !self.exit.load(SeqCst)
    }

    /// Run the stores waiting, a run at a time, in order, until the ring is
    /// empty.
    fn run_stores(&self) -> bool {
        let mut run = [(0u32, 0u64); RUN_MAX];
        let mut worked = false;
        loop {
            let n = self.ring.peek_run_at(0, &mut run);
            if n == 0 {
                return worked;
            }
            worked = true;
            {
                let mut guard = self.pipe.lock();
                let pipe = &mut *guard;
                for &(off, val) in &run[..n] {
                    // A waiting write (a widget register, the credit or tag) is applied here, in stream order, with the stores.
                    match classify(off, true) {
                        Access::Waits => pipe.core.write_wait(&self.host, &mut pipe.backend, off, val as u32),
                        _ => pipe.core.stream_store(&self.host, &mut pipe.backend, off, val as u32),
                    }
                }
                pipe.core.reason_drain(&self.host);
                self.publish(&pipe.core);
                // A tile worker that panicked leaves the pixels unknowable: the board is dead,
                // as for a panic of this thread.
                if pipe.backend.is_poisoned() && !self.dead.swap(true, SeqCst) {
                    crate::dlog!(crate::devlog::LogModule::Kona, "kona: a tile worker panicked; the board is dead");
                }
                // Taken out of the ring only now that they have run: a
                // register that waits for the ring to be empty then finds the
                // state the stores left.
                self.ring.consume_n_at(0, n);
                self.ring.flush_head();
            }
        }
    }

    /// What the polled registers read of the stream.
    fn publish(&self, core: &Core) {
        self.host.held_words.store(core.held_words(), SeqCst);
        self.host.reasons_waiting.store(core.reasons_waiting(), SeqCst);
    }

    /// The ring is empty and the thread has finished its work: the FIFO-low
    /// cause, if the driver enabled it, and the machine's signal.
    fn drained(&self) {
        self.host.with_front(|f| ((), f.lowater()));
    }

    fn thread_main(self: &Arc<Self>) {
        let _dead = DeadOnPanic(self);
        while self.wait_for_work() {
            if self.run_stores() {
                self.drained();
            }
            // Reasons a cause-register access found waiting are posted here.
            if self.reason_poke.load(SeqCst) {
                // The poke is cleared with the pipe held, so a test that waits for the
                // poke and then for the pipe cannot see both clear mid-drain.
                let mut guard = self.pipe.lock();
                self.reason_poke.store(false, SeqCst);
                guard.core.reason_drain(&self.host);
                self.publish(&guard.core);
            }
        }
    }

    /// Run whatever is left in the ring on the calling thread (a stop with stores waiting).
    fn drain_inline(&self) {
        if self.run_stores() {
            self.drained();
        }
    }
}

/// Marks the board dead if the thread unwinds: its back end panicked. The
/// bus then fails visibly instead of answering busy forever to a thread that
/// is gone.
struct DeadOnPanic<'a>(&'a Shared);

impl Drop for DeadOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.dead.store(true, SeqCst);
            self.0.running.store(false, SeqCst);
            crate::dlog!(crate::devlog::LogModule::Kona, "kona: the Host Interface Processor thread panicked; the board is dead");
        }
    }
}

/// The InfiniteReality pipe's host interface.
pub struct Kona {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Kona {
    /// A pipe of configuration `cfg`: its interrupt packets reach `irq`, its DMA
    /// reaches `mem`, and its display shows the machine's `status`. With `flash`, the
    /// flash is held in that file, which must be exactly `FLASH_SIZE` bytes:
    /// every byte the pipe programs is written through, so the contents outlast
    /// the process as they outlast power-off on a pipe.
    pub fn new(cfg: KonaConfig, irq: Arc<dyn InterruptController>, mem: Arc<dyn BusDevice>, status: Arc<crate::disp::StatusSource>, flash: Option<File>) -> Result<Arc<Kona>, String> {
        let flash = match flash {
            Some(f) => flash::Flash::backed(f)?,
            None => flash::Flash::fresh(),
        };
        let backend = KonaBackend::new(cfg);
        backend.set_status(status);
        let pixels = backend.pixels();
        let display: Arc<dyn crate::gfx_display::GfxDisplay> = backend.display();
        let mut front = Front::new(&cfg);
        front.probe_value = cfg.probe;
        let host = Arc::new(Host {
            front: Mutex::new(front),
            irq,
            mem,
            unrouted: Default::default(),
            held_words: Default::default(),
            reasons_waiting: Default::default(),
            list_running: Default::default(),
            activity: Default::default(),
            video_acks: Default::default(),
        });
        // SAFETY: a zeroed GFifo is an empty one (`GFifo::new`); built in
        // place because the ring is too big for the stack.
        let ring: Box<Ring> = unsafe { Box::new_zeroed().assume_init() };
        let shared = Arc::new(Shared {
                host,
                pipe: Mutex::new(Pipe { core: Core::new(cfg, flash), backend }),
                ring,
                pixels,
                display,
                monitor_wait: AtomicU64::new(MONITOR_WAIT.as_nanos() as u64),
                dead: AtomicBool::new(false),
                running: AtomicBool::new(false),
                exit: AtomicBool::new(false),
                asleep: AtomicBool::new(false),
                sleeper: Mutex::new(None),
                reason_poke: AtomicBool::new(false),
            });
        crate::locks::register_mutex_fn("kona pipe", &shared, |s| s.pipe.is_locked());
        Ok(Arc::new(Kona { shared, thread: Mutex::new(None) }))
    }

    /// A read of `size` bytes at aperture offset `off`: the bus status and
    /// the data. The registers are word-sized: an 8-byte read is two word
    /// reads, high word at the lower address.
    fn bus_read(&self, off: u64, size: u32) -> (u32, u64) {
        if self.shared.dead.load(Relaxed) {
            return (BUS_ERR, 0);
        }
        if off >= APERTURE {
            return (BUS_ERR, 0);
        }
        let off = off as u32;
        match size {
            4 => match self.read_word(off) {
                Some(v) => (BUS_OK, v as u64),
                None => (BUS_BUSY, 0),
            },
            8 => {
                crate::dlog!(crate::devlog::LogModule::Kona, "kona: 8-byte load at offset {off:#x} split into two 4-byte accesses");
                match (self.read_word(off), self.read_word(off.wrapping_add(4))) {
                    (Some(hi), Some(lo)) => (BUS_OK, (hi as u64) << 32 | lo as u64),
                    _ => (BUS_BUSY, 0),
                }
            }
            _ => (BUS_ERR, 0),
        }
    }

    /// A store of `size` bytes at aperture offset `off`. BUS_BUSY: the caller
    /// retries the same store.
    ///
    /// An 8-byte store to the command port is two words, high first, each as
    /// a 4-byte store at the same offset (the X driver module's bulk data
    /// helpers store aligned pairs of data words that way, and the GL client's
    /// vector senders store a selector and its first argument as one
    /// doubleword at any stream position). Any other 8-byte store is split
    /// into two word stores at `off` and `off + 4`, logged: no evidence says
    /// what a real pipe does with one.
    fn bus_write(&self, off: u64, size: u32, val: u64) -> u32 {
        if self.shared.dead.load(Relaxed) {
            return BUS_ERR;
        }
        if off >= APERTURE {
            return BUS_ERR;
        }
        let off = off as u32;
        match size {
            4 => self.write_word(off, val as u32),
            8 => {
                let (hi, lo) = ((val >> 32) as u32, val as u32);
                if off == defs::COMMAND_PAGE {
                    self.write_words(&[(off, hi), (off, lo)])
                } else {
                    crate::dlog!(crate::devlog::LogModule::Kona, "kona: 8-byte store at offset {off:#x} split into two 4-byte accesses");
                    self.write_words(&[(off, hi), (off.wrapping_add(4), lo)])
                }
            }
            _ => BUS_ERR,
        }
    }

    fn read_word(&self, off: u32) -> Option<u32> {
        match classify(off, false) {
            Access::Now => Some(self.read_now(off)),
            _ => {
                // A read that waits for the stream answers busy while stores wait
                // in the ring or the thread is mid-batch; otherwise it takes the
                // pipe, for a pure read that cannot reach memory or an interrupt.
                // The ring is checked first, which is sound because `run_stores`
                // takes entries out of the ring only after the batch has run and
                // still under the pipe lock: an empty ring means every earlier
                // store has finished.
                let s = &*self.shared;
                if !s.ring.is_empty() {
                    s.wake();
                    return None;
                }
                let mut pipe = s.pipe.try_lock()?;
                Some(pipe.core.read_wait(&s.host.front, off))
            }
        }
    }

    fn read_now(&self, off: u32) -> u32 {
        use defs::*;
        let s = &*self.shared;
        let pending = !s.ring.is_empty();
        match off {
            INT_CAUSE => {
                // Reasons waiting behind an unread one are looked for when
                // the cause register is accessed.
                if s.host.reasons_waiting.load(SeqCst) {
                    s.reason_poke.store(true, SeqCst);
                    s.wake();
                }
            }
            ACTIVITY => {
                // The latch reads set while work waits, the pipe being busy,
                // and the work sets it again as it consumes words.
                return if s.host.activity.load(Relaxed) || pending { ACTIVITY_BUS } else { 0 };
            }
            _ => {}
        }
        // The register is the graphics FIFO's fill level: the words the pipe
        // holds, those of commands still arriving and the stores waiting in
        // the ring, rounded up to two (the field's bit 0 never sets). It
        // reads full while a display list runs: one store can stand for a
        // long list, so a count of words would understate the wait. Software
        // expectation: users wait for it to fall (to zero, below 0x5dd, or
        // to 0x300) before storing more. Inferred: the unit is 32-bit words.
        // Unknown: the FIFO's depth (levels of 0x70c have been read from real
        // pipes, so it holds at least that).
        let credit = if s.host.list_running.load(SeqCst) {
            CREDIT_MASK
        } else {
            let words = s.host.held_words.load(SeqCst) + s.ring.len() as u32;
            ((words + 1) & !1).min(CREDIT_MASK)
        };
        s.host.front.lock().read(off, credit).unwrap_or_else(|| {
            crate::dlog!(crate::devlog::LogModule::Kona, "kona: unimplemented read of offset {off:#x}");
            0
        })
    }

    /// One word of a store. A store that joins the stream, or waits for it, goes
    /// into the ring and nothing more.
    #[inline]
    fn write_word(&self, off: u32, val: u32) -> u32 {
        self.write_words(&[(off, val)])
    }

    /// The words of one store, together: they all go in or none does, so a busy
    /// answer can be retried. Words that join the stream or wait for it go into
    /// the ring for `kona-hip` to apply in order (stores wait there until the
    /// thread starts); words served at once are applied here, after the ring
    /// took the others.
    fn write_words(&self, words: &[(u32, u32)]) -> u32 {
        let s = &*self.shared;
        // A map RAM store with nothing ahead of it in the stream is applied
        // here, as a waiting read already is: the driver tests the map RAM
        // word by word, a store and a load each, a million times at start-up,
        // and a hand-off to the thread for every store costs it a thread wake.
        if let &[(o, v)] = words {
            if (defs::MAPRAM_BASE..defs::MAPRAM_BASE + 4 * defs::MAPRAM_ENTRIES as u32).contains(&o) && s.ring.is_empty() {
                if let Some(mut pipe) = s.pipe.try_lock() {
                    if s.ring.is_empty() {
                        let p = &mut *pipe;
                        p.core.stream_store(&s.host, &mut p.backend, o, v);
                        return BUS_OK;
                    }
                }
            }
        }
        let mut queued = [(0u32, 0u64); 2];
        let mut n = 0;
        for &(o, v) in words {
            if classify(o, true) != Access::Now {
                queued[n] = (o, v as u64);
                n += 1;
            }
        }
        let ok = match &queued[..n] {
            [] => true,
            [(o, v)] => s.ring.try_push(*o, *v),
            [(o0, v0), (o1, v1)] => s.ring.try_push2(*o0, *v0, *o1, *v1),
            _ => unreachable!(),
        };
        if n > 0 {
            s.wake();
        }
        if !ok {
            return BUS_BUSY;
        }
        for &(o, v) in words {
            if classify(o, true) == Access::Now {
                self.write_now(o, v);
            }
        }
        BUS_OK
    }

    fn write_now(&self, off: u32, value: u32) {
        let s = &*self.shared;
        let host = &s.host;
        if off == defs::ACTIVITY {
            if value & defs::ACTIVITY_BUS != 0 {
                host.activity.store(false, Relaxed);
            }
            if value & defs::ACTIVITY_VIDEO != 0 {
                host.video_acks.fetch_add(1, Relaxed);
            }
            if value & !(defs::ACTIVITY_BUS | defs::ACTIVITY_VIDEO) != 0 {
                crate::dlog!(crate::devlog::LogModule::Kona, "kona: unimplemented write of offset {off:#x} (value {value:#x})");
            }
            return;
        }
        let mut packet = None;
        let known = host.front.lock().write(off, value, &mut packet);
        if let Some(p) = packet {
            host.deliver(p);
        }
        if !known && off != defs::INTR_END {
            crate::dlog!(crate::devlog::LogModule::Kona, "kona: unimplemented write of offset {off:#x} (value {value:#x})");
        }
        if off == defs::INT_CAUSE && host.reasons_waiting.load(SeqCst) {
            s.reason_poke.store(true, SeqCst);
            s.wake();
        }
    }

    /// The window the pipe shows the guest, for the machine's `get_display`.
    pub fn display(&self) -> Arc<dyn crate::gfx_display::GfxDisplay> {
        self.shared.display.clone()
    }

    /// The widget the pipe's DMA requests go to: bits 25:22 of the widget
    /// control register, which host software fills with its interrupt
    /// target. A machine's DMA handle may choose the destination with it.
    pub fn dma_widget(&self) -> u32 {
        (self.shared.host.front.lock().widget_control >> 22) & 0xf
    }

    /// Why the pipe stopped taking commands, if it did.
    /// How many times the kernel has written the video-interrupt bit of the activity register.
    pub fn video_acks(&self) -> u64 {
        self.shared.host.video_acks.load(Relaxed)
    }

    pub fn failure(&self) -> Option<String> {
        if self.shared.dead.load(Relaxed) {
            return Some("kona: the back end panicked on the Host Interface Processor thread; the board is dead".into());
        }
        let pipe = self.shared.pipe.lock();
        if pipe.backend.is_poisoned() {
            return Some("kona: a tile worker panicked; the board is dead".into());
        }
        pipe.core.failed.clone()
    }

    /// The flash's contents.
    pub fn flash_image(&self) -> Vec<u8> {
        self.shared.pipe.lock().core.flash.image().to_vec()
    }

    /// The sections of the last completed firmware load.
    pub fn load_record(&self) -> Vec<LoadSection> {
        self.shared.pipe.lock().core.load_record().to_vec()
    }

    /// A store with no board thread: the bus write, then the stores it queued
    /// run on this thread.
    #[cfg(test)]
    pub(super) fn test_write(&self, off: u64, size: u32, val: u64) -> u32 {
        let r = self.bus_write(off, size, val);
        self.run_stores();
        r
    }

    /// Run the stores in the ring on this thread, with no board thread: a test
    /// driver that wants each store to have run before the next goes in (a
    /// replay's records are numbered as they arrive).
    #[cfg(test)]
    pub(super) fn run_stores(&self) {
        self.shared.drain_inline();
    }

    /// Wait until every store taken has been run: the ring empty and the pipe
    /// between batches. (With no thread running, stores stay in the ring.)
    #[cfg(test)]
    pub(super) fn settle(&self) {
        let s = &*self.shared;
        while s.running.load(SeqCst) && !s.dead.load(SeqCst) && !(s.ring.is_empty() && !s.reason_poke.load(SeqCst) && s.pipe.try_lock().is_some()) {
            s.wake();
            std::thread::yield_now();
        }
    }

    /// Run `f` on the back end, with the thread's work finished first.
    pub fn with_backend<R>(&self, f: impl FnOnce(&mut KonaBackend) -> R) -> R {
        let s = &*self.shared;
        // The stores in the ring run first; the thread then holds the pipe only
        // for a batch.
        while s.running.load(SeqCst) && !s.dead.load(SeqCst) && !s.ring.is_empty() {
            s.wake();
            std::thread::yield_now();
        }
        let mut guard = s.pipe.lock();
        f(&mut guard.backend)
    }

    /// Bring the back end to rest for a snapshot. The machine's processors are
    /// stopped: the stores in the ring run to completion, then the back end
    /// finishes its queued work and parks its threads. `resume` undoes it.
    pub fn quiesce(&self) {
        let s = &*self.shared;
        while s.running.load(SeqCst) && !s.dead.load(SeqCst) && !s.ring.is_empty() {
            s.wake();
            std::thread::yield_now();
        }
        s.pipe.lock().backend.quiesce();
    }

    /// The board's state as chunks, for a snapshot: the pixel store, the cursor memory and the
    /// texture arena, and the small state (the register file, the pipe's registers, pages,
    /// contexts and memories, the display tables, the textport and every rendering context
    /// with its level map) as one more list named `state`. The board is quiesced, and the
    /// stream is between commands: a snapshot taken in the middle of one is refused, so is
    /// one with a primitive or a pixel transfer open. Counters are not saved.
    pub fn save_chunks(&self, store: &crate::chunk_store::ChunkStore) -> std::io::Result<Vec<KonaChunk>> {
        let io = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidInput, e);
        let mut pipe = self.shared.pipe.lock();
        let p = &mut *pipe;
        let state = self.state_bytes(p).map_err(io)?;
        let mut chunks = p.backend.save_chunks(store)?;
        let n = state.len();
        chunks.push(KonaChunk { name: format!("state-{n}"), hashes: back::snapshot::put_bytes(store, &state)? });
        Ok(chunks)
    }

    /// Restore what `save_chunks` wrote, into a board of the same configuration, all or nothing.
    /// Every chunk is read, and the small state is parsed once over throwaway objects, before
    /// the board is touched; only then does the real pass run (which cannot fail on the bytes,
    /// since a visit is a function of them alone), the bulk state go in, and the flash image,
    /// the one effect outside the board, be written through to its file last.
    pub fn load_chunks(&self, store: &crate::chunk_store::ChunkStore, list: &[KonaChunk]) -> Result<(), String> {
        let (state, rest): (Vec<_>, Vec<_>) = list.iter().cloned().partition(|c| c.name.starts_with("state-"));
        let [c] = &state[..] else { return Err("a snapshot holds one state list".into()) };
        let n: usize = c.name["state-".len()..].parse().map_err(|_| "the state list's size is not a number")?;
        let blob = back::snapshot::get_bytes(store, &c.hashes, n)?;
        let mut pipe = self.shared.pipe.lock();
        let p = &mut *pipe;
        let prepared = p.backend.prepare_chunks(store, &rest)?;
        self.load_state(p, &blob)?;
        p.backend.apply_chunks(prepared)?;
        p.core.flash.write_through();
        self.shared.host.activity.store(false, Relaxed);
        self.shared.publish(&p.core);
        Ok(())
    }

    /// The small state as the bytes `load_state` reads: the register file, the pipe's registers,
    /// pages, queues, contexts and memories, and the back end's small state. Refused between the
    /// words of a command and with a primitive or a pixel transfer open.
    fn state_bytes(&self, p: &mut Pipe) -> Result<Vec<u8>, String> {
        p.core.at_boundary()?;
        let mut w = stream::Writer::default();
        self.shared.host.front.lock().visit(&mut w);
        p.core.visit(&mut w);
        p.backend.visit_small(&mut w)?;
        Ok(w.out)
    }

    /// Read the small state from `blob`: parsed once over throwaway objects, which changes
    /// nothing and refuses a blob that is cut short or has bytes left, then read into the board.
    fn load_state(&self, p: &mut Pipe, blob: &[u8]) -> Result<(), String> {
        {
            let cfg = p.core.cfg.clone();
            let mut r = stream::Reader::new(blob);
            Front::new(&cfg).visit(&mut r);
            Core::new(cfg, flash::Flash::fresh()).visit(&mut r);
            KonaBackend::check_small(&mut r);
            r.finish()?;
        }
        let mut r = stream::Reader::new(blob);
        self.shared.host.front.lock().visit(&mut r);
        p.core.visit(&mut r);
        p.backend.visit_small(&mut r)?;
        r.finish()
    }

    /// Undo `quiesce`.
    pub fn resume(&self) {
        self.shared.pipe.lock().backend.resume();
    }

    fn reset(&self) {
        let was_running = self.is_running();
        Device::stop(self);
        let s = &*self.shared;
        s.ring.reset();
        {
            let mut pipe = s.pipe.lock();
            pipe.core.reset();
            // A new epoch before the back end cancels its tickets, so a call
            // a thread of its made in the old one is dropped.
            s.host.front.lock().advance_epoch();
            pipe.backend.reset();
            s.publish(&pipe.core);
        }
        s.host.front.lock().reset();
        s.host.activity.store(false, Relaxed);
        if was_running {
            Device::start(self);
        }
    }
}

/// The processor's access to the widget's aperture. Only the register-sized
/// widths are served; the others answer `BUS_ERR`.
impl BusDevice for Kona {
    fn read32(&self, addr: u64) -> BusRead32 {
        match self.bus_read(addr, 4) {
            (BUS_OK, v) => BusRead32::ok(v as u32),
            (BUS_BUSY, _) => BusRead32::busy(),
            _ => BusRead32::err(),
        }
    }

    fn read64(&self, addr: u64) -> BusRead64 {
        match self.bus_read(addr, 8) {
            (BUS_OK, v) => BusRead64::ok(v),
            (BUS_BUSY, _) => BusRead64::busy(),
            _ => BusRead64::err(),
        }
    }

    fn write32(&self, addr: u64, val: u32) -> u32 {
        self.bus_write(addr, 4, val as u64)
    }

    fn write64(&self, addr: u64, val: u64) -> u32 {
        self.bus_write(addr, 8, val)
    }
}

impl Device for Kona {
    fn step(&self, _cycles: u64) {}

    fn start(&self) {
        let mut t = self.thread.lock();
        if t.is_some() {
            return;
        }
        let s = self.shared.clone();
        s.pipe.lock().backend.attach(DisplayHandle::new(&s.host));
        s.exit.store(false, SeqCst);
        s.running.store(true, SeqCst);
        *t = Some(
            std::thread::Builder::new()
                .name("kona-hip".into())
                .spawn(move || s.thread_main())
                .expect("kona thread"),
        );
    }

    fn stop(&self) {
        let Some(t) = self.thread.lock().take() else { return };
        let s = &*self.shared;
        s.exit.store(true, SeqCst);
        s.wake();
        if let Some(th) = s.sleeper.lock().as_ref() {
            th.unpark();
        }
        let _ = t.join();
        s.running.store(false, SeqCst);
        // What the thread had not taken runs here, unless it died: its back
        // end would only panic again.
        if !s.dead.load(SeqCst) {
            s.drain_inline();
        }
        // The back end's own threads join with the slot taken and no
        // register-file lock held: one of them may be inside a handle call,
        // which takes only that lock and sends its packet after dropping it,
        // and never asks for the slot.
        // Always, even for a dead board: `start` attaches again, and the
        // first run's threads must be gone by then. A back end that panics
        // here is logged, not rethrown, since this runs from `Drop`.
        let stopped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.pipe.lock().backend.stop()));
        if stopped.is_err() {
            crate::dlog!(crate::devlog::LogModule::Kona, "kona: the back end panicked while stopping");
        }
    }

    fn is_running(&self) -> bool {
        self.shared.running.load(SeqCst)
    }

    fn get_clock(&self) -> u64 {
        0
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("kona".into(), "InfiniteReality pipe: status|counters|hash|pix X Y|shot FILE.png (kona help)".into())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "kona" {
            return Err(format!("unknown kona command: {cmd}"));
        }
        // Pixel reads never wait. The commands that read the back end's state
        // wait for the command stream at most this long, then say so.
        match self.shared.pixels.read(args, &mut *w) {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(e) => return Err(e.to_string()),
        }
        let Some(mut guard) = self.shared.pipe.try_lock_for(std::time::Duration::from_nanos(self.shared.monitor_wait.load(Relaxed))) else {
            return Err("kona: board busy".into());
        };
        match guard.backend.monitor(args, &mut *w) {
            Ok(true) => Ok(()),
            Ok(false) => Err("kona: status|counters|hash|pix X Y|shot FILE.png".into()),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Drop for Kona {
    fn drop(&mut self) {
        Device::stop(self);
    }
}

impl Resettable for Kona {
    fn power_on(&self) {
        self.reset();
    }
}
