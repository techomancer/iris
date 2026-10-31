//! The MMIO aperture: write dispatch, which accesses wait for drawing, and the
//! locks the board is reached through.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.

use super::*;

impl Board {
    pub fn write_dispatch(&mut self, addr: u64, value: u64, size: u32) {
        crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: MMIO write seq {} offset {:#x} size {} value {:#x}", self.trace_sequence, addr, size, value);
        self.trace_sequence = self.trace_sequence.wrapping_add(1);

        if addr == ODSY_BACKEND_SELECT && (size == 8 || size == 4) {
            self.backend_write(value, size);
            return;
        }
        if addr == ODSY_CTX_SWITCH_PORT && size == 8 {
            self.ctx_switch(value);
            return;
        }
        if addr == ODSY_CMD_FIFO {
            self.selftest_track(ODSY_CMD_FIFO, size, value);
            self.cmd_from_kernel = true;
            self.cmd_fifo_write(value, size);
            return;
        }
        if addr == ODSY_USER_CMD_FIFO {
            // The user-mapped command FIFO has the kernel port's grammar but
            // never carries the self-test preamble, so it skips the tracker.
            self.cmd_from_kernel = false;
            self.cmd_fifo_write(value, size);
            return;
        }

        // Registers that do not wait for the board's work live in the front
        // (interface.rs).
        if front_register(addr, size, true) {
            self.front.lock().write(addr, value, size);
            return;
        }

        // -1 (no register) becomes u64::MAX, which no constant matches.
        match reg(addr, size) as u64 {
            ODSY_RESET_CONTROL => {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: write of {:#x} to the reset control register accepted; its effect is unknown",
                    value
                );
            }
            ODSY_MEMORY_CONTROL => self.memory_control(value as u32),
            ODSY_FLCTL_CONFIG => self.flctl_config = value as u32,
            ODSY_BUZZ_1 => self.buzz[0] = value as u32,
            ODSY_BUZZ_2 => self.buzz[1] = value as u32,
            ODSY_BUZZ_3 => self.buzz[2] = value as u32,
            ODSY_BUZZ_4 => self.buzz[3] = value as u32,
            ODSY_BUZZ_5 => self.buzz[4] = value as u32,
            ODSY_BUZZ_6 => self.buzz[5] = value as u32,
            ODSY_NIC_PROM | ODSY_NIC_KERNEL => self.nic_slot(value as u32),
            ODSY_HOST_WRITE_ADDR_UPPER => {
                self.query_cancel();
                self.host_write_addr_upper = value as u32;
            }
            ODSY_HOST_WRITE_ADDR_LOWER => {
                self.query_cancel();
                self.host_write_addr_lower = value as u32;
            }
            ODSY_STATS_ADDR_UPPER | ODSY_STATS_ADDR_LOWER | ODSY_STATS_CONTROL => {
                if addr == ODSY_STATS_ADDR_UPPER {
                    self.stats_addr_upper = value as u32;
                } else if addr == ODSY_STATS_ADDR_LOWER {
                    self.stats_addr_lower = value as u32;
                } else {
                    self.stats_control = value as u32;
                }
                if value != 0 && !self.stats_reported {
                    self.stats_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: statistics region word {:#x} at {:#x} stored; the board's writes into the region are not modeled",
                        value as u32,
                        addr
                    );
                }
            }
            ODSY_CTX_SHADOW_ADDR_UPPER => {
                self.query_cancel();
                self.ctx_shadow_addr_upper = value as u32;
            }
            ODSY_CTX_SHADOW_ADDR_LOWER => {
                self.query_cancel();
                self.ctx_shadow_addr_lower = value as u32;
            }
            ODSY_MEMORY_CONFIG
            | ODSY_I2C_MAIN_CMD_READ
            | ODSY_I2C_MAIN_STATUS_READ
            | ODSY_I2C_OPT_CMD_READ
            | ODSY_I2C_OPT_STATUS_READ
            | ODSY_BACKEND_VERSION => {
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
    // order, and the board's thread (worker.rs) runs the queue in slices under
    // the board lock. Accesses that must see the drawing take the board lock
    // and run what is still queued first (Board::work_drain). With no thread
    // running (unit tests) the storing caller runs the store itself.
    //
    // The Octane has no PI link (only Hub and Bedrock machines have graphics
    // credits), so the front's `pi` stays None there and every credit call is
    // skipped.

}

pub fn fifo_port(addr: u64) -> bool {
    addr == ODSY_USER_CMD_FIFO || addr == ODSY_CMD_FIFO
}

/// The stores that join the command stream: the two command-FIFO ports, and
/// the kernel's context switch, the shadow address it names and the switch
/// store itself.
pub fn command_store(addr: u64, size: u32) -> bool {
    fifo_port(addr)
        || (addr == ODSY_CTX_SWITCH_PORT && size == 8)
        || reg(addr, size) == ODSY_CTX_SHADOW_ADDR_UPPER as i64
        || reg(addr, size) == ODSY_CTX_SHADOW_ADDR_LOWER as i64
}

/// A read of the level word `0x106c`, which never waits: it is a plain load
/// of live levels (Odyssey datasheet, "Flags, status and flow control", the
/// level word). The make-current and detach readers discard it, and the
/// drain waits poll bits 15:7 until they read 0, which the live level
/// does once the board has run the queue.
pub fn level_word(addr: u64, size: u32) -> bool {
    reg(addr, size) == ODSY_MEMORY_CONFIG as i64
}

/// Accesses that need not wait for the queued stores, and act on the board
/// as it stands. Inferred from what a driver does while the board works,
/// not measured on a board; every other access waits.
pub fn access_waits(addr: u64, size: u32, write: bool) -> bool {
    let cursor = ODSY_BACKEND_READ_BASE + ((ODSY_CURSOR_BASE as u64) << 3);

    if fifo_port(addr) {
        return !write;
    }
    if reg(addr, size) < 0 {
        // Outside the modeled registers only the back-end and
        // context-switch ports and the cursor do anything; the rest is
        // logged and answers zero, so it need not wait.
        if write {
            return addr == ODSY_BACKEND_SELECT || addr == ODSY_CTX_SWITCH_PORT;
        }
        return false;
    }
    if !write && addr >= cursor && addr < cursor + ((ODSY_CURSOR_ENTRIES as u64) << 3) {
        return false;
    }
    !matches!(
        reg(addr, size) as u64,
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
            | ODSY_INTR_FLAGS
            | ODSY_INTR_CLEAR
            | ODSY_INTR_ENABLE_SET
            | ODSY_INTR_ENABLE_CLEAR
            | ODSY_ROUTE_DEST_UPPER
            | ODSY_ROUTE_DEST_LOWER
            | ODSY_ROUTE_VECTORS
            | ODSY_SW_TOKEN
            // The datasheet records the word as stored, not routed, with
            // no wait; making the processor wait on it modelled nothing.
            | ODSY_FLCTL_TARGET
            | ODSY_SWAP_ARM
            | ODSY_SWAP_PENDING
            | ODSY_SWAP_CONTROL
            | ODSY_SWAP_CONTROL_COMPANION
            | ODSY_DISPLAYED_BUFFER
    )
}

impl Board {
    /// A read with the board lock held.
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
pub fn store_counted(front: &parking_lot::Mutex<HostInterface>) -> bool {
    // The Octane has no PI link, so nothing sets the mark there.
    #[cfg(test)]
    if front.lock().test_counted_stores != 0 {
        return true;
    }
    let _ = front;
    pi_gfx_take_mark()
}

impl Board {
    /// A store with the board lock held.
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

/// The two locks a processor's access goes through, and the queues behind
/// them.
#[derive(Clone)]
pub struct BoardLocks {
    pub board: Arc<Mutex<Box<Board>>>,
    pub front: Arc<Mutex<HostInterface>>,
    pub fifo: Arc<CommandFifo>,
    pub raster: Arc<RasterQueue>,
    pub refusals: Arc<RefusalStats>,
    /// What the display thread reads of the board.
    pub shared: Arc<DisplayShared>,
}

/// Why the bus answered busy.
#[derive(Clone, Copy)]
pub enum Refusal {
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
pub struct RefusalStats {
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
        for i in 0..REFUSAL_SLOTS {
            self.counts[i].store(0, Ordering::Relaxed);
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
    pub fn new(s: Box<Board>) -> Self {
        let (front, fifo, raster, shared) = (s.front.clone(), s.fifo.clone(), s.raster.clone(), s.scanout.shared.clone());
        BoardLocks { board: Arc::new(Mutex::new(s)), front, fifo, raster, refusals: Arc::new(RefusalStats::new()), shared }
    }

    /// A processor's read: through the front alone when it can, else under the
    /// board lock.
    pub fn read(&self, addr: u64, size: u32) -> u64 {
        match front_mmio_read(&self.front, addr, size) {
            Some(v) => v,
            None => self.board.lock().mmio_read(addr, size),
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
        Some(self.board.lock().mmio_read(addr, size))
    }

    /// A processor's store as the bus takes it. False, for the bus to
    /// answer busy, when it waits for queued stores the board's thread will
    /// run, or when it is a command store and the board's FIFO has no
    /// room for it while the board's thread runs: the processor retries
    /// the store, taking its interrupts meanwhile, while the board's thread
    /// makes room, as a full FIFO holds the store back on the board. The
    /// processor interface's mark stays untaken until the store goes in.
    pub fn bus_write(&self, addr: u64, value: u64, size: u32) -> bool {
        // Classified once: a command store goes straight to the FIFO.
        if command_store(addr, size) {
            if self.fifo.full_for(size) {
                self.fifo.hurry();
                self.refusals.note(Refusal::StoreFull, addr, size);
                return false;
            }
            self.write_command(addr, value, size);
            return true;
        }
        // A store that waits for the work, as a waiting read does.
        if access_waits(addr, size, true) && self.fifo.running.load(std::sync::atomic::Ordering::SeqCst) && self.fifo.pending() {
            self.fifo.hurry();
            self.refusals.note(Refusal::StoreWaits, addr, size);
            return false;
        }
        self.write(addr, value, size);
        true
    }

    /// A command store the caller has classified: it joins the FIFO for the
    /// board's thread, else takes the board lock.
    fn write_command(&self, addr: u64, value: u64, size: u32) {
        let counted = store_counted(&self.front);
        let queued = self.fifo.running.load(std::sync::atomic::Ordering::SeqCst) && self.fifo.push(&self.front, addr, value, size, counted);
        if !queued {
            self.board.lock().mmio_write_counted(addr, value, size, counted);
        }
    }

    /// A processor's store: through the front alone when it can, else under the
    /// board lock.
    pub fn write(&self, addr: u64, value: u64, size: u32) {
        let counted = store_counted(&self.front);
        if !front_mmio_write(&self.front, &self.fifo, addr, value, size, counted) {
            self.board.lock().mmio_write_counted(addr, value, size, counted);
        }
    }
}
