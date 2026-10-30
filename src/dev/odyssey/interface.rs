//! The board's host interface, under a lock of its own.
//!
//! The board takes command-FIFO stores into a FIFO and draws behind them:
//! the processor goes on while a primitive is drawn. Meanwhile the driver
//! stores more commands and polls the registers that do not wait for the
//! drawing: identity, control, the interrupt bank, the routing, the swap
//! bank and the FIFO status (`access_waits`). Every other access
//! waits until the drawing behind the stores before it is done.
//!
//! So the board is two states under two locks. The host interface
//! (`HostInterface`) holds those registers, the interrupt state and the
//! queue of stores the board has not run yet. The rest of the board
//! (`Board`) sits behind the board lock, which the board's thread
//! holds for a whole slice of stores (worker.rs). A processor's front
//! access and its command stores take only the front lock and never wait
//! for a draw; every other access takes the board lock, which waits out the
//! slice, and then runs whatever is still queued before it.
//!
//! Lock order: board, then front. Nothing takes the board lock while it
//! holds the front lock.
//!
//! Ported from hw/misc/sgi-odyssey.c in jamesbraid/qemu-sgi-origami.
use super::*;

use parking_lot::Mutex;
use std::sync::Arc;

pub struct HostInterface {
    /// The crosstalk widget number.
    pub slot: u32,
    /// Where interrupt packets go (the iris glue's HEART).
    pub irq: Option<Arc<dyn crate::irq::InterruptController>>,
    /// The master node's processor interface, for graphics credits.
    pub pi: Option<Arc<dyn PiGfxLink>>,
    /// The widget number of HEART, the packets' destination.
    pub heart_widget: u32,
    /// ODSY_IDENTITY, composed from the revision at reset.
    pub identity: u32,

    pub widget_control: u32,
    pub xt_req_timeout: u32,
    pub int_dest_upper: u32,
    pub int_dest_lower: u32,
    pub error_command: u32,
    pub intr_flags: u32,
    pub intr_enable: u32,
    /// The flag source's transition state, for the packet edge.
    pub interrupt_asserted: bool,
    pub sw_token: u32,
    pub route_dest_upper: u32,
    pub route_dest_lower: u32,
    pub route_vectors: u32,
    pub swap_arm: u32,
    pub swap_pending: u32,
    pub swap_control: u32,
    pub swap_control_companion: u32,
    pub swap_set: u32,
    pub swap_clear: u32,
    pub displayed_buffer: u32,
    /// A processor wrote ODSY_DISPLAYED_BUFFER: the scanout, which lives
    /// under the board lock, repaints at its next update.
    pub displayed_written: bool,

    // Command-FIFO stores waiting for the board's thread (worker.rs).
    pub work_queue: Vec<QueuedStore>,
    /// The oldest waiting stores, this many, are in the FIFO and credited.
    pub work_in_fifo: u32,
    /// The board's thread, while it runs.
    pub worker: Option<std::thread::JoinHandle<()>>,
    /// Wakes the board's thread when a store joins an empty queue.
    pub kick: Arc<parking_lot::Condvar>,
    /// The board's thread is asleep on `kick`.
    pub worker_waiting: bool,
    /// The board's thread is running stores under the board lock.
    pub slice: bool,
    /// The board's thread is to finish.
    pub worker_exit: bool,
    /// Tests: stores count as counted by a processor interface.
    #[cfg(test)]
    pub test_counted_stores: u8,
    /// Tests: queue every command store even with no thread running, and
    /// run this many per slice.
    #[cfg(test)]
    pub test_words: u32,
    /// The front accesses' own trace numbering; the board's accesses keep
    /// the state's.
    pub trace_sequence: u64,
}

impl Default for HostInterface {
    fn default() -> Self {
        HostInterface {
            slot: 0,
            irq: None,
            pi: None,
            heart_widget: 8,
            identity: 0,
            widget_control: 0,
            xt_req_timeout: 0,
            int_dest_upper: 0,
            int_dest_lower: 0,
            error_command: 0,
            intr_flags: 0,
            intr_enable: 0,
            interrupt_asserted: false,
            sw_token: 0,
            route_dest_upper: 0,
            route_dest_lower: 0,
            route_vectors: 0,
            swap_arm: 0,
            swap_pending: 0,
            swap_control: 0,
            swap_control_companion: 0,
            swap_set: 0,
            swap_clear: 0,
            displayed_buffer: 0,
            displayed_written: false,
            work_queue: Vec::new(),
            work_in_fifo: 0,
            worker: None,
            kick: Default::default(),
            worker_waiting: false,
            slice: false,
            worker_exit: false,
            #[cfg(test)]
            test_counted_stores: 0,
            #[cfg(test)]
            test_words: 0,
            trace_sequence: 0,
        }
    }
}

// Interrupts. Every source is delivered from the front, which holds the
// routing and the flag bank.

impl HostInterface {
    /// An interrupt packet into the widget-addressed fabric.
    pub fn send_packet(&mut self, where_: u64, vector: u8, set: bool) {
        self.send_interrupt(where_, vector, set);
    }
}

/// The fabric address of a widget/address pair: the widget in bits 51:48.
pub fn odyssey_xio_compose(widget: u32, address: u64, fabric: &mut u64) -> bool {
    sgi_xio_compose("sgi-odyssey", widget, address, fabric)
}

impl HostInterface {
    /// One routed source's interrupt. Every routed source shares one
    /// destination (ODSY_ROUTE_DEST_UPPER/_LOWER), distinct from
    /// int_dest_upper/int_dest_lower, which the kernel driver gives its
    /// crosstalk error handler; the sources differ only in their byte of
    /// ODSY_ROUTE_VECTORS. The vectors are read from the register, never
    /// assumed. The flag source is a level the guest clears through
    /// ODSY_INTR_CLEAR, so it sends a clearing packet as it falls
    /// (HostInterface::update_irq()); the softsync, retrace and DMA-done handlers ack
    /// nothing, and those sources send a single set pulse.
    pub fn deliver_routed(&mut self, shift: u32, what: &str, set: bool) {
        let widget = (self.route_dest_upper >> ODSY_ROUTE_DEST_WIDGET_SHIFT) & ((1u32 << ODSY_INT_DEST_WIDGET_BITS) - 1);
        let addr = (((self.route_dest_upper & 0xffff) as u64) << 32) | self.route_dest_lower as u64;
        let vector = self.route_vectors.checked_shr(shift).unwrap_or(0) as u8;
        let mut where_ = 0u64;

        if self.route_dest_upper == 0 && self.route_dest_lower == 0 {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-odyssey: {} interrupt {} with no destination programmed; no packet sent",
                what,
                if set { "raised" } else { "lowered" }
            );
            return;
        }
        if !odyssey_xio_compose(widget, addr, &mut where_) {
            return;
        }
        self.send_packet(where_, vector, set);
    }

    /// Whether the routed destination is programmed.
    pub fn routed(&self) -> bool {
        self.route_dest_upper != 0 || self.route_dest_lower != 0
    }

    /// The softsync source ODSY_REG_HOST_WRITE (0x0a5a) raises.
    pub fn deliver_softsync(&mut self) {
        self.deliver_routed(ODSY_ROUTE_VECTOR_SOFTSYNC_SHIFT, "softsync", true);
    }

    pub fn update_irq(&mut self) {
        let level = (self.intr_flags & self.intr_enable) != 0;

        if level != self.interrupt_asserted {
            self.interrupt_asserted = level;
            self.deliver_routed(ODSY_ROUTE_VECTOR_FLAG_SHIFT, "flag", level);
        }
    }

    /// The command FIFO's level: the stores waiting for the board's work.
    pub fn fifo_level(&self) -> u32 {
        self.work_queue.len() as u32
    }

    /// The FIFO's level went from 'before' to its current value. Falling below
    /// the low-water mark latches the low-water flag whatever the enable mask
    /// says; it reaches the driver only while the source is enabled.
    pub fn fifo_level_changed(&mut self, before: u32) {
        if before >= ODSY_CFIFO_LOW_WATER && self.fifo_level() < ODSY_CFIFO_LOW_WATER {
            self.intr_flags |= ODSY_INTR_FIFO_LOW_WATER;
            self.update_irq();
        }
    }

    // The work queue.

    pub fn pending(&self) -> bool {
        !self.work_queue.is_empty()
    }

    /// A counted store's credit, to the processor interface the node names.
    pub fn return_credit(&mut self) {
        if let Some(pi) = &self.pi {
            pi_gfx_return_credits(pi.as_ref(), self.slot, 1);
        } else {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: counted graphics store has no processor interface for its return");
        }
    }

    /// Waiting stores that have moved up into the FIFO get their credits. One
    /// that waited outside it repays a credit the node holds as owed
    /// (pi_gfx_owe_credits).
    pub fn work_admit(&mut self) {
        while self.work_in_fifo < self.work_queue.len() as u32 && self.work_in_fifo < ODSY_CFIFO_DEPTH {
            let q = self.work_queue[self.work_in_fifo as usize];

            if q.owed {
                if let Some(pi) = &self.pi {
                    pi_gfx_repay_credits(pi.as_ref(), self.slot, 1);
                }
                self.work_queue[self.work_in_fifo as usize].owed = false;
            } else if q.counted {
                self.return_credit();
            }
            self.work_in_fifo += 1;
        }
    }

    /// Queue a store and have the board's thread run it. With no thread
    /// started the caller, which holds the board lock, drains it.
    pub fn work_append(&mut self, addr: u64, value: u64, size: u32, counted: bool) {
        let q = QueuedStore {
            value,
            addr: addr as u32,
            size: size as u8,
            counted,
            owed: false,
            queued_at: std::time::Instant::now(),
        };

        self.work_queue.push(q);
        self.work_admit();
        if counted && self.work_in_fifo < self.work_queue.len() as u32 && self.pi.is_some() {
            let last = self.work_queue.len() - 1;
            self.work_queue[last].owed = true;
            pi_gfx_owe_credits(self.pi.as_ref().unwrap().as_ref(), self.slot, 1);
        }
        if self.worker_waiting {
            self.kick.notify_one();
        }
    }

    /// The oldest waiting store, out of the queue and the FIFO.
    pub fn work_take(&mut self) -> Option<QueuedStore> {
        if self.work_queue.is_empty() {
            return None;
        }
        let q = self.work_queue.remove(0);
        // The store leaves the FIFO, and the next waiting one moves up.
        self.work_in_fifo = self.work_in_fifo.wrapping_sub(1);
        self.work_admit();
        Some(q)
    }

    /// Whether a store that joins the command stream waits in the queue: always
    /// while the board's thread runs, so that decoding and drawing never run on
    /// the processor's thread, and otherwise behind any store that already
    /// waits.
    pub fn store_queues(&self, addr: u64, size: u32) -> bool {
        if !command_store(addr, size) {
            return false;
        }
        self.worker.is_some() || self.pending() || self.queue_held()
    }

    /// Tests: every command store queues, to be run by hand.
    pub fn queue_held(&self) -> bool {
        #[cfg(test)]
        return self.test_words != 0;
        #[cfg(not(test))]
        {
            let _ = self;
            false
        }
    }
}

// The front registers.

/// Whether `addr` is a front register for this access: one of the
/// registers that do not wait for the board's work. The other accesses
/// that do not wait (the back end's read window, unmodeled offsets) report
/// through state the board lock covers, so they take it.
pub fn front_register(addr: u64, size: u32, write: bool) -> bool {
    if fifo_port(addr) || reg(addr, size) < 0 || access_waits(addr, size, write) {
        return false;
    }
    if !write {
        // Reads of these two fall to the unmodeled-access report.
        return !matches!(reg(addr, size) as u64, ODSY_INTR_CLEAR | ODSY_INTR_ENABLE_CLEAR);
    }
    true
}

impl HostInterface {
    pub fn read(&mut self, addr: u64, size: u32) -> u64 {
        let mut value: u64 = 0;
        let sequence = self.trace_sequence;
        self.trace_sequence = self.trace_sequence.wrapping_add(1);

        match reg(addr, size) as u64 {
            ODSY_IDENTITY => value = self.identity as u64,
            ODSY_CONTROL => value = self.widget_control as u64,
            ODSY_XT_REQ_TIMEOUT => value = self.xt_req_timeout as u64,
            ODSY_INT_DEST_UPPER => value = self.int_dest_upper as u64,
            ODSY_INT_DEST_LOWER => value = self.int_dest_lower as u64,
            ODSY_ERROR_COMMAND => value = self.error_command as u64,
            ODSY_STATUS | ODSY_ERROR_ADDR_UPPER | ODSY_ERROR_ADDR_LOWER => {
                // No crosstalk error is ever posted, so the error class is clear.
            }
            ODSY_INTR_FLAGS => value = self.intr_flags as u64,
            ODSY_INTR_ENABLE_SET => value = self.intr_enable as u64,
            ODSY_STATUS0 => {
                // The waiting stores are its level (HostInterface::fifo_level()).
                let level = self.fifo_level();
                value = (ODSY_STATUS0_FIFO_ENABLED
                    | if level != 0 { 0 } else { ODSY_STATUS0_FIFO_EMPTY }
                    | if level < ODSY_CFIFO_LOW_WATER { ODSY_STATUS0_FIFO_LOW_WATER } else { 0 }) as u64;
            }
            ODSY_SW_TOKEN => value = self.sw_token as u64,
            ODSY_SWAP_ARM => value = self.swap_arm as u64,
            ODSY_SWAP_PENDING => value = self.swap_pending as u64,
            ODSY_SWAP_CONTROL => value = self.swap_control as u64,
            ODSY_SWAP_CONTROL_COMPANION => value = self.swap_control_companion as u64,
            // The buffer the last latched swap put on screen (retrace()).
            ODSY_DISPLAYED_BUFFER => value = self.displayed_buffer as u64,
            ODSY_ROUTE_DEST_UPPER | ODSY_ROUTE_DEST_LOWER | ODSY_ROUTE_VECTORS => {
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: read of the write-only register at {:#x}", addr);
            }
            _ => unreachable!("odyssey_front_register() admits only these"),
        }
        crate::dlog_dev!(LogModule::Odyssey, "sgi_odyssey_mmio: front seq {} offset {:#x} size {} value {:#x} write=0", sequence, addr, size, value);
        value
    }

    pub fn write(&mut self, addr: u64, value: u64, size: u32) {
        crate::dlog_dev!(LogModule::Odyssey, "sgi_odyssey_mmio: front seq {} offset {:#x} size {} value {:#x} write=1", self.trace_sequence, addr, size, value);
        self.trace_sequence = self.trace_sequence.wrapping_add(1);

        match reg(addr, size) as u64 {
            ODSY_CONTROL => self.widget_control = value as u32,
            ODSY_XT_REQ_TIMEOUT => self.xt_req_timeout = value as u32,
            ODSY_INT_DEST_UPPER => self.int_dest_upper = value as u32,
            ODSY_INT_DEST_LOWER => self.int_dest_lower = value as u32,
            // Written to acknowledge; the data itself is discarded.
            ODSY_ERROR_COMMAND => self.error_command = 0,
            ODSY_INTR_CLEAR => {
                self.intr_flags &= !(value as u32);
                self.update_irq();
            }
            ODSY_INTR_ENABLE_SET => {
                self.intr_enable |= value as u32;
                self.update_irq();
            }
            ODSY_INTR_ENABLE_CLEAR => {
                self.intr_enable &= !(value as u32);
                self.update_irq();
            }
            ODSY_INTR_FLAGS => {
                // The flags take the value written: an inferred model choice.
                // A driver's flow control writes the low-water bit here before
                // enabling that source, and its DMA diagnostic clears one flag
                // by writing back the flags it read without it; only a
                // replacing write serves the second. The board's paired
                // registers (set enables, clear enables, clear flags) make a
                // set-only write the competing reading.
                self.intr_flags = value as u32;
                self.update_irq();
            }
            ODSY_SW_TOKEN => self.sw_token = value as u32,
            ODSY_SWAP_ARM => {
                // Queue flips for the next retrace. Writes accumulate: the
                // driver can issue several windows or swap groups between two
                // retraces, and its retrace handler re-writes one window's bit
                // when a flip has not shown after two retraces. Whether the
                // board ORs or replaces is unknown; ORing loses no flip.
                self.swap_arm |= value as u32;
            }
            ODSY_SWAP_PENDING => self.swap_pending = value as u32,
            ODSY_SWAP_CONTROL => {
                // The windows that must show buffer 1 from the next retrace on.
                // A later write to either register for the same window replaces
                // this one before the retrace latches it.
                self.swap_control = value as u32;
                self.swap_set |= value as u32;
                self.swap_clear &= !(value as u32);
            }
            ODSY_SWAP_CONTROL_COMPANION => {
                self.swap_control_companion = value as u32;
                self.swap_clear |= value as u32;
                self.swap_set &= !(value as u32);
            }
            ODSY_DISPLAYED_BUFFER => {
                // Board-produced status; a guest write is not seen in the
                // capture, but store it rather than drop it.
                self.displayed_buffer = value as u32;
                self.displayed_written = true;
            }
            ODSY_ROUTE_DEST_UPPER => self.route_dest_upper = value as u32,
            ODSY_ROUTE_DEST_LOWER => self.route_dest_lower = value as u32,
            ODSY_ROUTE_VECTORS => self.route_vectors = value as u32,
            ODSY_IDENTITY | ODSY_STATUS | ODSY_ERROR_ADDR_UPPER | ODSY_ERROR_ADDR_LOWER => {
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: write of {:#x} to the read-only register at {:#x}", value, addr);
            }
            _ => unreachable!("odyssey_front_register() admits only these"),
        }
    }
}

/// A processor's read through the front alone: Some when it is a front
/// register, None when the access needs the board lock.
pub fn front_mmio_read(f: &Mutex<HostInterface>, addr: u64, size: u32) -> Option<u64> {
    if !front_register(addr, size, false) {
        return None;
    }
    Some(f.lock().read(addr, size))
}

/// A processor's store through the front alone: true when it went to a
/// front register or joined the queue for the board's thread, false when
/// it needs the board lock. `counted`: the processor interface counted it.
pub fn front_mmio_write(f: &Mutex<HostInterface>, addr: u64, value: u64, size: u32, counted: bool) -> bool {
    let mut f = f.lock();
    if front_register(addr, size, true) {
        f.write(addr, value, size);
        if counted {
            f.return_credit();
        }
        return true;
    }
    // A queued store needs the board's thread to run it, and the queue to
    // have room; otherwise the store drains the queue under the board lock.
    if f.worker.is_none() || !f.store_queues(addr, size) || f.work_queue.len() >= ODSY_WORK_QUEUE_MAX as usize {
        return false;
    }
    f.work_append(addr, value, size, counted);
    true
}

/// The fabric packet's address bits; the widget sits above them.
pub const SGI_XIO_PACKET_ADDRESS_BITS: u32 = 48;

pub const SGI_XIO_WIDGETS: u32 = 16;

/// The fabric address of a widget/address pair: the widget in bits 51:48.
pub fn sgi_xio_compose(who: &str, widget: u32, address: u64, packet: &mut u64) -> bool {
    if widget >= SGI_XIO_WIDGETS || address >> SGI_XIO_PACKET_ADDRESS_BITS != 0 {
        crate::dlog!(LogModule::Odyssey, "{who}: packet fabric cannot encode widget {widget} address {address:#x}");
        return false;
    }
    *packet = (widget as u64) << SGI_XIO_PACKET_ADDRESS_BITS | address;
    true
}

/// HEART's interrupt-packet register in its widget space.
const HEART_PACKET: u64 = 0x80;

impl HostInterface {
    /// `HostInterface::send_interrupt(&s->fabric_as, packet, number, set)`: the
    /// interrupt-modify write. HEART takes it at 0x80 of widget 8.
    pub fn send_interrupt(&mut self, packet: u64, number: u8, set: bool) {
        let widget = (packet >> SGI_XIO_PACKET_ADDRESS_BITS) as u32;
        let address = packet & ((1u64 << SGI_XIO_PACKET_ADDRESS_BITS) - 1);
        if widget != self.heart_widget || address != HEART_PACKET {
            crate::dlog!(LogModule::Odyssey, "odyssey: interrupt packet to widget {widget} address {address:#x} has no receiver");
            return;
        }
        if let Some(irq) = &self.irq {
            irq.set_line(number as u32, set);
        }
    }
}
