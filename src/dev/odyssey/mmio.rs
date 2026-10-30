//! The MMIO aperture: write dispatch, which accesses wait for drawing, and the
//! locks the board is reached through.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.


use crate::devlog::LogModule;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use super::board::Board;
use super::consts::{
    backend, cmd, ctx, cursor, flctl, host, i2c, intr, mem, nic, route, stats, swap, ODSY_BUZZ_1,
    ODSY_BUZZ_2, ODSY_BUZZ_3, ODSY_BUZZ_4, ODSY_BUZZ_5, ODSY_BUZZ_6, ODSY_CONTROL,
    ODSY_DISPLAYED_BUFFER, ODSY_ERROR_ADDR_LOWER, ODSY_ERROR_ADDR_UPPER, ODSY_ERROR_COMMAND,
    ODSY_IDENTITY, ODSY_INT_DEST_LOWER, ODSY_INT_DEST_UPPER, ODSY_RESET_CONTROL, ODSY_STATUS,
    ODSY_STATUS0, ODSY_SW_TOKEN, ODSY_USER_CMD_FIFO, ODSY_XT_REQ_TIMEOUT,
};
use super::fifo::CommandFifo;
use super::interface::{front_mmio_read, front_mmio_write, front_register, HostInterface};
use super::pi_gfx::pi_gfx_take_mark;
use super::raster_unit::RasterQueue;
use super::reader::DisplayShared;
use super::registers::reg;
impl Board {
    pub fn write_dispatch(&mut self, addr: u64, value: u64, size: u32) {
        let sequence = self.trace_sequence.fetch_add(1, Ordering::Relaxed);
        crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: MMIO write seq {} offset {:#x} size {} value {:#x}", sequence, addr, size, value);

        if addr == backend::SELECT && (size == 8 || size == 4) {
            self.backend_write(value, size);
            return;
        }
        if addr == ctx::SWITCH_PORT && size == 8 {
            self.ctx_switch(value);
            return;
        }
        if addr == cmd::FIFO {
            self.selftest_track(cmd::FIFO, size, value);
            self.cmd.from_kernel = true;
            self.cmd_fifo_write(value, size);
            return;
        }
        if addr == ODSY_USER_CMD_FIFO {
            // The user-mapped command FIFO has the kernel port's grammar but
            // never carries the self-test preamble, so it skips the tracker.
            self.cmd.from_kernel = false;
            self.cmd_fifo_write(value, size);
            return;
        }

        // Registers that do not wait for the board's work live in the front
        // (interface.rs).
        if front_register(addr, size, true) {
            self.front.lock().write(addr, value, size);
            return;
        }

        match reg(addr, size) {
            Some(ODSY_RESET_CONTROL) => {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: write of {:#x} to the reset control register accepted; its effect is unknown",
                    value
                );
            }
            Some(mem::CONTROL) => self.memory_control(value as u32),
            Some(flctl::CONFIG) => self.widget.flctl_config = value as u32,
            Some(ODSY_BUZZ_1) => self.widget.buzz[0] = value as u32,
            Some(ODSY_BUZZ_2) => self.widget.buzz[1] = value as u32,
            Some(ODSY_BUZZ_3) => self.widget.buzz[2] = value as u32,
            Some(ODSY_BUZZ_4) => self.widget.buzz[3] = value as u32,
            Some(ODSY_BUZZ_5) => self.widget.buzz[4] = value as u32,
            Some(ODSY_BUZZ_6) => self.widget.buzz[5] = value as u32,
            Some(nic::PROM | nic::KERNEL) => self.nic_slot(value as u32),
            Some(host::WRITE_ADDR_UPPER) => {
                self.query_cancel();
                self.widget.host_write_addr_upper = value as u32;
            }
            Some(host::WRITE_ADDR_LOWER) => {
                self.query_cancel();
                self.widget.host_write_addr_lower = value as u32;
            }
            Some(stats::ADDR_UPPER | stats::ADDR_LOWER | stats::CONTROL) => {
                if addr == stats::ADDR_UPPER {
                    self.widget.stats_addr_upper = value as u32;
                } else if addr == stats::ADDR_LOWER {
                    self.widget.stats_addr_lower = value as u32;
                } else {
                    self.widget.stats_control = value as u32;
                }
                if value != 0 && !self.diag.stats_reported {
                    self.diag.stats_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: statistics region word {:#x} at {:#x} stored; the board's writes into the region are not modeled",
                        value as u32,
                        addr
                    );
                }
            }
            Some(ctx::SHADOW_ADDR_UPPER) => {
                self.query_cancel();
                self.widget.ctx_shadow_addr_upper = value as u32;
            }
            Some(ctx::SHADOW_ADDR_LOWER) => {
                self.query_cancel();
                self.widget.ctx_shadow_addr_lower = value as u32;
            }
            Some(mem::CONFIG
            | i2c::MAIN_CMD_READ
            | i2c::MAIN_STATUS_READ
            | i2c::OPT_CMD_READ
            | i2c::OPT_STATUS_READ
            | backend::VERSION) => {
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: write of {:#x} to the read-only register at {:#x}", value, addr);
            }
            _ => self.unimplemented(addr, size, true),
        }
    }

    // Command work after the store.
    //
    // The board draws a primitive after the store that completes it has
    // finished: the processor goes on, and the driver learns of completion from
    // a host-write fence, the context-switch word in guest memory, or an
    // interrupt. A model that draws inside the store keeps the processor from
    // its scheduler tick for as long as the draw takes, and the guest does not
    // make up lost ticks.
    //
    // So every store that joins the command stream waits in a queue, in
    // order, and the board's thread (worker.rs) runs it. Accesses that must
    // see the drawing are answered busy until the queue is empty
    // (BoardLocks::bus_read) or, with no thread running (unit tests), run what
    // is queued first (Board::work_drain).
    //
    // The Octane has no PI link (only Hub and Bedrock machines have graphics
    // credits), so the front's `pi` stays None there and every credit call is
    // skipped.

}

pub(super) fn fifo_port(addr: u64) -> bool {
    addr == ODSY_USER_CMD_FIFO || addr == cmd::FIFO
}

/// The stores that join the command stream: the two command-FIFO ports, and
/// the kernel's context switch, the shadow address it names and the switch
/// store itself.
pub(super) fn command_store(addr: u64, size: u32) -> bool {
    fifo_port(addr)
        || (addr == ctx::SWITCH_PORT && size == 8)
        || matches!(reg(addr, size), Some(ctx::SHADOW_ADDR_UPPER | ctx::SHADOW_ADDR_LOWER))
}

/// A read of the level word `0x106c`, which never waits: it is a plain load
/// of live levels. The make-current and detach readers discard it, and the
/// drain waits poll bits 15:7 until they read 0, which the live level
/// does once the board has run the queue.
pub(super) fn level_word(addr: u64, size: u32) -> bool {
    reg(addr, size) == Some(mem::CONFIG)
}

/// Accesses that need not wait for the queued stores, and act on the board
/// as it stands. Inferred from what a driver does while the board works,
/// not measured on a board; every other access waits.
pub(super) fn access_waits(addr: u64, size: u32, write: bool) -> bool {
    let cursor = backend::READ_BASE + ((cursor::BASE as u64) << 3);

    if fifo_port(addr) {
        return !write;
    }
    let Some(r) = reg(addr, size) else {
        // Outside the modeled registers only the back-end and
        // context-switch ports and the cursor do anything; the rest is
        // logged and answers zero, so it need not wait.
        if write {
            return addr == backend::SELECT || addr == ctx::SWITCH_PORT;
        }
        return false;
    };
    if !write && addr >= cursor && addr < cursor + ((cursor::ENTRIES as u64) << 3) {
        return false;
    }
    !matches!(
        r,
        ODSY_IDENTITY
            | ODSY_STATUS
            | ODSY_ERROR_ADDR_UPPER
            | ODSY_ERROR_ADDR_LOWER
            | ODSY_CONTROL
            | ODSY_XT_REQ_TIMEOUT
            | ODSY_INT_DEST_UPPER
            | ODSY_INT_DEST_LOWER
            | ODSY_ERROR_COMMAND
            | ODSY_STATUS0
            | intr::FLAGS
            | intr::CLEAR
            | intr::ENABLE_SET
            | intr::ENABLE_CLEAR
            | route::DEST_UPPER
            | route::DEST_LOWER
            | route::VECTORS
            | ODSY_SW_TOKEN
            // The word is stored, not routed, with no wait; making the processor wait on it modelled nothing.
            | flctl::TARGET
            | swap::ARM
            | swap::PENDING
            | swap::CONTROL
            | swap::CONTROL_COMPANION
            | ODSY_DISPLAYED_BUFFER
    )
}

impl Board {
    /// A read of the board's state.
    pub fn mmio_read(&mut self, addr: u64, size: u32) -> u64 {
        if access_waits(addr, size, false) && !level_word(addr, size) {
            // The stores before it and the raster work they handed on.
            if self.work_pending() {
                self.work_drain(addr);
            }
            self.raster_wait();
        }
        self.read(addr, size)
    }
}

/// Whether this store is counted: the processor interface's mark.
pub(super) fn store_counted(front: &parking_lot::Mutex<HostInterface>) -> bool {
    // The Octane has no PI link, so nothing sets the mark there.
    #[cfg(test)]
    if front.lock().test_counted_stores != 0 {
        return true;
    }
    let _ = front;
    pi_gfx_take_mark()
}

impl Board {
    /// A store to the board's state.
    pub fn mmio_write(&mut self, addr: u64, value: u64, size: u32) {
        let counted = store_counted(&self.front);
        self.mmio_write_counted(addr, value, size, counted);
    }

    pub fn mmio_write_counted(&mut self, addr: u64, value: u64, size: u32, mut counted: bool) {
        let fifo = self.fifo.clone();
        let queued = command_store(addr, size) && fifo.queues() && fifo.push(&self.front, addr, value, size, counted);
        if queued {
            // Credited as it enters the FIFO (CommandFifo::push). With no
            // board thread running the storing caller runs it.
            counted = false;
            if !fifo.running.load(std::sync::atomic::Ordering::SeqCst) && !fifo.held() {
                self.work_drain(addr);
            }
        } else if command_store(addr, size) {
            // The queue is full and nobody waits for room (a caller other
            // than the bus): run what waits and then this store, in order.
            self.work_drain(addr);
            self.write_dispatch(addr, value, size);
        } else {
            if access_waits(addr, size, true) && self.work_pending() {
                self.work_drain(addr);
            }
            self.write_dispatch(addr, value, size);
        }
        if counted {
            // It went through the FIFO inside the store.
            self.front.lock().return_credit();
        }
    }
}

/// The board's state, which the front end's thread owns. There is no lock
/// between the processor and the front end: the board's units reach their
/// memory and registers at the same time, and software orders them. Here
/// that order is the command FIFO (stores run in arrival order on the
/// board's thread), the waiting accesses (a processor's read or store that
/// must see the drawing is answered busy until the FIFO and the raster queue
/// are empty, so the board's thread is idle when the processor touches the
/// state), and the host interface's registers, which the processor reaches
/// without waiting (`HostInterface`).
///
/// Processors are serialised against each other by `BoardLocks::submit`, as
/// MGRAS's producers are by its `submit` lock: a processor holds it across
/// its busy check and its access, and across a command push, so a second
/// processor's push cannot wake the board's thread under a first
/// processor's access. The board's thread and the raster thread never take
/// it, so it adds no wait for drawing.
///
/// A processor access that does not wait and reaches the board (the level
/// word, the back-end read window, an unmodelled offset) still borrows the
/// whole board while the board's thread may run: it reads words the thread
/// writes only in whole stores, or atomics, and shows the old or the new
/// value. That is the other boards' tolerated racy read, not a lock.
pub(super) struct BoardCell(std::cell::UnsafeCell<Box<Board>>);

// SAFETY: see the type's comment: the board's thread owns the state while it
// runs stores; a processor touches it holding `submit` after the waiting
// accesses have drained the FIFO and the raster queue, with the machine
// stopped, or for the whole-word reads above. This is the other boards'
// contract between a CPU thread and an engine's FIFO.
unsafe impl Sync for BoardCell {}
unsafe impl Send for BoardCell {}

impl BoardCell {
    pub fn new(b: Box<Board>) -> Self {
        BoardCell(std::cell::UnsafeCell::new(b))
    }

    /// The board, for the thread that may touch it now: the board's thread
    /// running stores, a processor through `BoardLocks::board()`, or the
    /// only thread using the board (machine stopped, tests). Dereferencing
    /// it is the caller's claim to be that thread.
    pub(super) fn get(&self) -> *mut Board {
        // The box never moves, so the pointer is the board's for its life.
        // SAFETY: a place, not a reference: nothing is borrowed here.
        unsafe { std::ptr::addr_of_mut!(**self.0.get()) }
    }
}

/// The board for a processor, with the other processors held off
/// (`BoardLocks::submit`), as MGRAS's `FrontGuard` holds its front lock.
pub(super) struct BoardGuard<'a> {
    _cpu: crate::locks::SpinGuard<'a, ()>,
    board: &'a mut Board,
}

impl std::ops::Deref for BoardGuard<'_> {
    type Target = Board;
    fn deref(&self) -> &Board { self.board }
}

impl std::ops::DerefMut for BoardGuard<'_> {
    fn deref_mut(&mut self) -> &mut Board { self.board }
}

/// What a processor's access goes through: the host interface's lock, the
/// board's state, and the queues behind them.
#[derive(Clone)]
pub(super) struct BoardLocks {
    pub(super) board: Arc<BoardCell>,
    /// Serialises the processors' accesses to the board against each other
    /// (see `BoardCell`). Never held by the board's thread or the raster
    /// thread.
    submit: Arc<crate::locks::SpinLock<()>>,
    pub front: Arc<Mutex<HostInterface>>,
    pub fifo: Arc<CommandFifo>,
    pub raster: Arc<RasterQueue>,
    pub refusals: Arc<RefusalStats>,
    /// What the display thread reads of the board.
    pub shared: Arc<DisplayShared>,
}

/// Why the bus answered busy.
#[derive(Clone, Copy)]
pub(super) enum Refusal {
    /// A read behind command stores still in the FIFO.
    ReadFifo,
    /// A read behind raster work still queued.
    ReadRaster,
    /// A command store with the host ring full.
    StoreFull,
    /// A store that waits for the FIFO to drain.
    StoreWaits,
}

impl Refusal {
    const NAMES: [&'static str; 4] = ["read-fifo-pending", "read-raster-pending", "store-ring-full", "store-waits-fifo"];
}

const REFUSAL_SLOTS: usize = 512;

/// Standing counters of the accesses the bus answered busy, by reason and
/// by the aperture offset and width: lock-free, so the processor's threads
/// count without waiting. Read by the monitor's `vpro refusals`. The table
/// is open addressed; a key that finds it full is counted in `lost`.
pub(super) struct RefusalStats {
    keys: [AtomicU64; REFUSAL_SLOTS],
    counts: [AtomicU64; REFUSAL_SLOTS],
    lost: AtomicU64,
}

impl RefusalStats {
    pub fn new() -> Self {
        Self { keys: std::array::from_fn(|_| AtomicU64::new(0)), counts: std::array::from_fn(|_| AtomicU64::new(0)), lost: AtomicU64::new(0) }
    }

    /// Count one refusal. The key is the reason, width and offset, plus one
    /// so that zero means an empty slot.
    pub fn note(&self, why: Refusal, addr: u64, size: u32) {
        let key = (((why as u64) << 56) | ((size as u64) << 48) | addr) + 1;
        let mut i = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as usize % REFUSAL_SLOTS;
        for _ in 0..REFUSAL_SLOTS {
            let k = self.keys[i].load(Ordering::Relaxed);
            if k == key || (k == 0 && self.keys[i].compare_exchange(0, key, Ordering::Relaxed, Ordering::Relaxed).map_or_else(|c| c == key, |_| true)) {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
            i = (i + 1) % REFUSAL_SLOTS;
        }
        self.lost.fetch_add(1, Ordering::Relaxed);
    }

    pub fn reset(&self) {
        for c in &self.counts {
            c.store(0, Ordering::Relaxed);
        }
        self.lost.store(0, Ordering::Relaxed);
    }

    /// Totals by reason, then each offset by count, largest first.
    pub fn report(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let mut rows: Vec<(u64, u64)> = (0..REFUSAL_SLOTS)
            .filter_map(|i| {
                let (k, c) = (self.keys[i].load(Ordering::Relaxed), self.counts[i].load(Ordering::Relaxed));
                (k != 0 && c != 0).then(|| (k - 1, c))
            })
            .collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        let mut totals = [0u64; Refusal::NAMES.len()];
        for (k, c) in &rows {
            totals[(k >> 56) as usize] += c;
        }
        for (name, t) in Refusal::NAMES.iter().zip(totals) {
            writeln!(w, "reason {name} {t}")?;
        }
        writeln!(w, "lost {}", self.lost.load(Ordering::Relaxed))?;
        for (k, c) in rows {
            writeln!(w, "{} offset {:#x} size {} count {}", Refusal::NAMES[(k >> 56) as usize], k & 0xffff_ffff_ffff, (k >> 48) & 0xff, c)?;
        }
        Ok(())
    }
}

impl Default for RefusalStats {
    fn default() -> Self {
        Self::new()
    }
}

impl BoardLocks {
    /// The board for a processor holding `submit`, which `cpu` proves.
    ///
    /// # Safety
    /// The FIFO and the raster queue are empty (the waiting accesses), or
    /// the board's thread is stopped, or the access is a whole-word read
    /// `BoardCell` describes.
    unsafe fn board<'a>(&'a self, cpu: crate::locks::SpinGuard<'a, ()>) -> BoardGuard<'a> {
        // SAFETY: the caller's contract; `cpu` keeps the other processors out.
        BoardGuard { _cpu: cpu, board: unsafe { &mut *self.board.get() } }
    }

    pub fn new(s: Box<Board>) -> Self {
        let (front, fifo, raster, shared) = (s.front.clone(), s.fifo.clone(), s.raster.clone(), s.scanout.shared.clone());
        BoardLocks {
            board: Arc::new(BoardCell::new(s)),
            submit: Arc::new(crate::locks::SpinLock::new(())),
            front,
            fifo,
            raster,
            refusals: Arc::new(RefusalStats::new()),
            shared,
        }
    }

    /// A read for a caller other than the bus (tests, the monitor): through the
    /// front alone when it can, else it runs what waits and reads the state.
    /// With the board's thread running, call `bus_read` and retry instead.
    pub fn read(&self, addr: u64, size: u32) -> u64 {
        match front_mmio_read(&self.front, addr, size) {
            Some(v) => v,
            None => {
                // SAFETY: the caller owns the board with its thread stopped,
                // or accepts the race of a whole-word read.
                unsafe { self.board(self.submit.lock()) }.mmio_read(addr, size)
            }
        }
    }

    /// A processor's read as the bus takes it. None, for the bus to answer
    /// busy, while the work the read must see is still queued on the
    /// board's threads: the command stores before it and the raster work
    /// they handed on. The level word never waits. The processor retries
    /// the load, taking its interrupts meanwhile, while the board's threads
    /// run that work, as IMPACT's waiting reads do; the read then finds
    /// nothing to drain. Without the board's thread the read drains the
    /// queue itself.
    pub fn bus_read(&self, addr: u64, size: u32) -> Option<u64> {
        if let Some(v) = front_mmio_read(&self.front, addr, size) {
            return Some(v);
        }
        // From the busy check to the end of the access no other processor
        // pushes a store that would wake the board's thread under it.
        let cpu = self.submit.lock();
        if access_waits(addr, size, false) && !level_word(addr, size) && self.fifo.running.load(std::sync::atomic::Ordering::SeqCst) {
            if self.fifo.pending() {
                self.fifo.hurry();
                self.refusals.note(Refusal::ReadFifo, addr, size);
                return None;
            }
            if self.raster.pending() != 0 {
                self.refusals.note(Refusal::ReadRaster, addr, size);
                return None;
            }
        }
        // SAFETY: the FIFO and the raster queue are empty, or the access is
        // a whole-word read.
        Some(unsafe { self.board(cpu) }.mmio_read(addr, size))
    }

    /// A processor's store as the bus takes it. False, for the bus to
    /// answer busy, when it waits for queued stores the board's thread will
    /// run, or when it is a command store and the host ring is full while
    /// the board's thread runs: the processor retries the store, taking its
    /// interrupts meanwhile, while the board's thread makes room. The
    /// processor interface's mark stays untaken until the store goes in.
    ///
    /// A store that waits is held back for the command FIFO only, not the
    /// raster queue: the raster unit never shared a lock with the board's
    /// state, and a store that touches the planes waits for the raster queue
    /// itself (`Board::raster_wait`). A read, which reads the planes' results,
    /// checks both.
    pub fn bus_write(&self, addr: u64, value: u64, size: u32) -> bool {
        if front_register(addr, size, true) {
            // The host interface's registers: no board state, no wait.
            self.write(addr, value, size);
            return true;
        }
        let cpu = self.submit.lock();
        // Classified once: a command store goes straight to the FIFO.
        if command_store(addr, size) {
            if self.fifo.full_for(size) {
                self.fifo.hurry();
                self.refusals.note(Refusal::StoreFull, addr, size);
                return false;
            }
            self.write_command(cpu, addr, value, size);
            return true;
        }
        // A store that waits for the work, as a waiting read does.
        if access_waits(addr, size, true) && self.fifo.running.load(std::sync::atomic::Ordering::SeqCst) && self.fifo.pending() {
            self.fifo.hurry();
            self.refusals.note(Refusal::StoreWaits, addr, size);
            return false;
        }
        self.write_board(cpu, addr, value, size);
        true
    }

    /// A command store the caller has classified, holding `submit`: it joins
    /// the FIFO for the board's thread, else, with no thread, runs here.
    fn write_command(&self, cpu: crate::locks::SpinGuard<'_, ()>, addr: u64, value: u64, size: u32) {
        let counted = store_counted(&self.front);
        while self.fifo.running.load(std::sync::atomic::Ordering::SeqCst) {
            // The ring refuses a push that finds the last place taken between
            // the room check and the push: the board's thread owns the state,
            // so the store goes in behind it rather than around it.
            if self.fifo.push(&self.front, addr, value, size, counted) {
                return;
            }
            std::thread::yield_now();
        }
        // SAFETY: no board thread is running.
        unsafe { self.board(cpu) }.mmio_write_counted(addr, value, size, counted);
    }

    /// A store to the board's state that is not a command store, holding
    /// `submit`.
    fn write_board(&self, cpu: crate::locks::SpinGuard<'_, ()>, addr: u64, value: u64, size: u32) {
        let counted = store_counted(&self.front);
        // SAFETY: the waiting accesses have drained the FIFO, or the board's
        // thread is stopped.
        unsafe { self.board(cpu) }.mmio_write_counted(addr, value, size, counted);
    }

    /// A store for a caller other than the bus: through the front alone when it
    /// can, a command store into the FIFO, else it runs what waits and then
    /// the store. With the board's thread running, call `bus_write` and retry
    /// for a store that waits.
    pub fn write(&self, addr: u64, value: u64, size: u32) {
        if front_register(addr, size, true) {
            let counted = store_counted(&self.front);
            front_mmio_write(&self.front, &self.fifo, addr, value, size, counted);
            return;
        }
        let cpu = self.submit.lock();
        if command_store(addr, size) {
            return self.write_command(cpu, addr, value, size);
        }
        self.write_board(cpu, addr, value, size);
    }
}

#[cfg(test)]
impl BoardLocks {
    /// The board, for a test that is the only thread using it, or whose
    /// board thread is idle.
    #[allow(clippy::mut_from_ref)]
    pub(super) fn test_board(&self) -> &mut Board {
        // SAFETY: the test's contract.
        unsafe { &mut *self.board.get() }
    }
}

#[cfg(test)]
impl BoardLocks {
    /// Hold the processors' lock, as a processor in the middle of an access
    /// does.
    pub(super) fn test_submit(&self) -> crate::locks::SpinGuard<'_, ()> {
        self.submit.lock()
    }
}
