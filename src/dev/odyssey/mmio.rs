//! The MMIO aperture: write dispatch, which accesses wait for drawing, and the
//! locks the board is reached through.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.

use super::*;

impl Board {
    pub fn write_dispatch(&mut self, addr: u64, value: u64, size: u32) {
        crate::dlog_dev!(LogModule::Odyssey, "sgi_odyssey_mmio: seq {} offset {:#x} size {} value {:#x} write=1", self.trace_sequence, addr, size, value);
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
            ODSY_FLCTL_TARGET => {
                self.flctl_target = value as u32;
                if !self.flctl_target_reported {
                    self.flctl_target_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: flow-control interrupt target {} and credit target {} stored; credits still return through the node interface",
                        (value as u32) >> ODSY_FLCTL_INTR_TARGET_SHIFT,
                        ((value >> ODSY_FLCTL_TARGET_SHIFT) as u32) & ODSY_FLCTL_TARGET_MASK
                    );
                }
            }
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
        if access_waits(addr, size, false) && self.work_pending() {
            self.work_drain(addr);
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
            // The queue is full: run what waits and then this store, in order.
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

/// The two locks a processor's access goes through.
#[derive(Clone)]
pub struct BoardLocks {
    pub board: Arc<Mutex<Box<Board>>>,
    pub front: Arc<Mutex<HostInterface>>,
    pub fifo: Arc<CommandFifo>,
}

impl BoardLocks {
    pub fn new(s: Box<Board>) -> Self {
        let (front, fifo) = (s.front.clone(), s.fifo.clone());
        BoardLocks { board: Arc::new(Mutex::new(s)), front, fifo }
    }

    /// A processor's read: through the front alone when it can, else under the
    /// board lock.
    pub fn read(&self, addr: u64, size: u32) -> u64 {
        match front_mmio_read(&self.front, addr, size) {
            Some(v) => v,
            None => self.board.lock().mmio_read(addr, size),
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
