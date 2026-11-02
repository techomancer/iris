//! The board's host interface: the registers a processor reaches without waiting.
//!
//! The board takes command-FIFO stores into a FIFO and draws behind them:
//! the processor goes on while a primitive is drawn. Meanwhile the driver
//! stores more commands and polls the registers that do not wait for the
//! drawing: identity, control, the interrupt bank, the routing, the swap
//! bank and the FIFO status (`access_waits`). Every other access
//! waits until the drawing behind the stores before it is done.
//!
//! So the board is two states. The host interface (`HostInterface`) holds
//! those registers and the interrupt state; its lock only serialises the
//! register file, and is held for a few instructions. The rest of the board
//! (`Board`) is the board's thread's (worker.rs), with no lock: a processor's
//! front access and its command stores never wait for a draw, and every
//! other access is answered busy until the stores before it and the raster
//! work they handed on are done (`BoardLocks::bus_read`), so the processor
//! touches the state only while the board's thread is idle.
//!
//! Ported from hw/misc/sgi-odyssey.c in jamesbraid/qemu-sgi-origami.

use crate::devlog::LogModule;
use super::consts::{
    cfifo, flctl, intr, route, status0, swap, ODSY_CONTROL, ODSY_DISPLAYED_BUFFER,
    ODSY_ERROR_ADDR_LOWER, ODSY_ERROR_ADDR_UPPER, ODSY_ERROR_COMMAND, ODSY_IDENTITY,
    ODSY_INT_DEST_LOWER, ODSY_INT_DEST_UPPER, ODSY_INT_DEST_WIDGET_BITS, ODSY_STATUS, ODSY_STATUS0,
    ODSY_SW_TOKEN, ODSY_XT_REQ_TIMEOUT,
};
use super::fifo::CommandFifo;
use super::mmio::{access_waits, command_store, fifo_port};
use super::pi_gfx::{pi_gfx_return_credits, PiGfxLink};
use super::registers::reg;
use super::PacketSender;
use parking_lot::Mutex;
use std::sync::Arc;

pub(super) struct HostInterface {
    /// The crosstalk widget number.
    pub slot: u32,
    /// Where interrupt packets go on the Octane (the iris glue's HEART).
    pub irq: Option<Arc<dyn crate::irq::InterruptController>>,
    /// Where interrupt packets go on a machine whose fabric carries them to
    /// any widget (the Fuel's XBridge crossbar); `irq` is unused then.
    pub packets: Option<PacketSender>,
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
    /// The flag level rose with no destination programmed and its set
    /// packet waits for one (`HostInterface::deliver_held_flag`).
    pub flag_held: bool,
    pub sw_token: u32,
    /// The flow-control target word (+0x1074), as last written. Stored, not
    /// routed: nothing consumes it, so it is a front register that neither
    /// waits for the command FIFO nor touches the board's state.
    pub flctl_target: u32,
    pub flctl_target_reported: bool,
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
    /// A processor wrote ODSY_DISPLAYED_BUFFER: the scanout repaints at its
    /// next update.
    pub displayed_written: bool,

    /// The command FIFO (fifo.rs), whose level STATUS0 reports.
    pub fifo: Arc<CommandFifo>,
    /// Tests: stores count as counted by a processor interface.
    #[cfg(test)]
    pub test_counted_stores: u8,
    /// The front accesses' own trace numbering; the board's accesses keep
    /// the state's.
    pub trace_sequence: u64,
}

impl Default for HostInterface {
    fn default() -> Self {
        HostInterface {
            slot: 0,
            irq: None,
            packets: None,
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
            flag_held: false,
            sw_token: 0,
            flctl_target: 0,
            flctl_target_reported: false,
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
            fifo: Default::default(),
            #[cfg(test)]
            test_counted_stores: 0,
            trace_sequence: 0,
        }
    }
}

// Interrupts. Every source is delivered from the front, which holds the
// routing and the flag bank.

impl HostInterface {
    /// A vertical retrace, which the scanout's clock runs at the mode's rate
    /// (ScanoutReader::poll()). The driver issues a swap by writing
    /// swap::ARM, a toggle mask with one bit per window whose displayed
    /// buffer must flip at the next retrace, then waits for the retrace
    /// interrupt. At each retrace the board first latches the pending flips
    /// into the displayed-buffer bits and then raises the retrace source,
    /// whose vector is byte 23:16 of route::VECTORS. The interrupt fires
    /// at every retrace once the routing is programmed, not only when a swap
    /// is armed; before the driver programs the routing nothing is
    /// delivered. Whether the board can mask this source is unknown and it is
    /// left unmasked.
    ///
    /// Approximation: the board's encoding of ODSY_DISPLAYED_BUFFER is
    /// inferred from the driver's compare, and the latch's timing within the
    /// retrace is unknown. The model latches at its retrace callback and
    /// repaints the rows the new selection changes (ScanoutReader::flip_rows())
    /// in the same frame's update; a frame is never presented half from each
    /// buffer.
    ///
    /// Whether a swap latched. The latch needs this lock alone.
    pub fn retrace(&mut self) -> bool {
        let flips = self.swap_arm;

        if self.swap_set != 0 || self.swap_clear != 0 {
            self.displayed_buffer = (self.displayed_buffer | self.swap_set) & !self.swap_clear;
            self.swap_set = 0;
            self.swap_clear = 0;
        }
        if flips != 0 {
            self.swap_arm = 0;
            self.displayed_buffer ^= flips;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: swap latch: arm {:#x} displayed {:#x}", flips, self.displayed_buffer);
        }
        crate::pf_instant!(crate::perfetto::events::ODSY_RETRACE, self.displayed_buffer);
        if self.routed() {
            self.deliver_routed(route::VECTOR_RETRACE_SHIFT, "retrace", true);
        }
        flips != 0
    }
}

impl HostInterface {
    /// An interrupt packet into the widget-addressed fabric.
    pub fn send_packet(&mut self, where_: u64, vector: u8, set: bool) {
        self.send_interrupt(where_, vector, set);
    }
}

/// The fabric address of a widget/address pair: the widget in bits 51:48.
pub(super) fn odyssey_xio_compose(widget: u32, address: u64, fabric: &mut u64) -> bool {
    sgi_xio_compose("sgi-odyssey", widget, address, fabric)
}

impl HostInterface {
    /// One routed source's interrupt. Every routed source shares one
    /// destination (route::DEST_UPPER/_LOWER), distinct from
    /// int_dest_upper/int_dest_lower, which the kernel driver gives its
    /// crosstalk error handler; the sources differ only in their byte of
    /// route::VECTORS. The vectors are read from the register, never
    /// assumed. The flag source is a level the guest clears through
    /// intr::CLEAR, so it sends a clearing packet as it falls
    /// (HostInterface::update_irq()); the softsync, retrace and DMA-done handlers ack
    /// nothing, and those sources send a single set pulse.
    ///
    /// MODEL CHOICE: bits 18:16 of
    /// ROUTE_DEST_UPPER are the per-source enables of the DMA-done, retrace
    /// and softsync sources, in the order of ODSY_ROUTE_ENABLE_*. A source
    /// raised while its bit is clear sends nothing and is dropped. Which
    /// bit enables which source is unknown, and what the board does with a
    /// source raised while disabled is approximated;
    /// IRIX sets and clears all three together, so it cannot tell. The flag
    /// source has no bit here: intr::ENABLE_SET gates it.
    pub fn deliver_routed(&mut self, shift: u32, what: &str, set: bool) {
        let widget = (self.route_dest_upper >> route::DEST_WIDGET_SHIFT) & ((1u32 << ODSY_INT_DEST_WIDGET_BITS) - 1);
        let addr = (((self.route_dest_upper & 0xffff) as u64) << 32) | self.route_dest_lower as u64;
        let vector = self.route_vectors.checked_shr(shift).unwrap_or(0) as u8;
        let mut where_ = 0u64;

        if !self.routed() {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-odyssey: {} interrupt {} with no destination programmed; no packet sent",
                what,
                if set { "raised" } else { "lowered" }
            );
            return;
        }
        let enable = match shift {
            route::VECTOR_DMA_SHIFT => route::ENABLE_DMA,
            route::VECTOR_RETRACE_SHIFT => route::ENABLE_RETRACE,
            route::VECTOR_SOFTSYNC_SHIFT => route::ENABLE_SOFTSYNC,
            _ => 0,
        };
        if self.route_dest_upper & enable != enable {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: {} interrupt with its enable bit clear; dropped", what);
            return;
        }
        if !odyssey_xio_compose(widget, addr, &mut where_) {
            return;
        }
        crate::pf_instant!(crate::perfetto::events::ODSY_IRQ, (shift as u64) << 9 | (vector as u64) << 1 | set as u64);
        self.send_packet(where_, vector, set);
    }

    /// Whether the routing is programmed: both halves of the destination and
    /// the vector table written. The kernel's interrupt set-up writes all
    /// three, the vectors first and the upper half, which carries the target
    /// widget, last, so a routing with one of them still zero is not yet the
    /// one it means, and no source sends to it. Limitation: a routing with a
    /// zero lower half or vector table never takes an interrupt.
    pub fn routed(&self) -> bool {
        self.route_dest_upper != 0 && self.route_dest_lower != 0 && self.route_vectors != 0
    }

    /// The softsync source reg::HOST_WRITE (0x0a5a) raises.
    pub fn deliver_softsync(&mut self) {
        self.deliver_routed(route::VECTOR_SOFTSYNC_SHIFT, "softsync", true);
    }

    pub fn update_irq(&mut self) {
        let level = (self.intr_flags & self.intr_enable) != 0;

        if level != self.interrupt_asserted {
            self.interrupt_asserted = level;
            self.flag_held = level && !self.routed();
            if self.flag_held {
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: flag interrupt raised with no destination programmed; held until one is");
            } else {
                // A level that falls while held was never sent, and with no
                // routing programmed there is no packet for its clear.
                self.deliver_routed(route::VECTOR_FLAG_SHIFT, "flag", level);
            }
        }
    }

    /// A routing write: send the flag level that rose before the destination
    /// was programmed, if it is still raised. MODEL CHOICE: the kernel enables
    /// flow control, and so can raise low water, before its interrupt set-up
    /// programs 0x1044, 0x104c and 0x1054, so the model holds the level
    /// pending and sends it once the destination is written. What the board
    /// does with such a flag (drop it, hold it, or send it to address zero)
    /// is unknown.
    fn deliver_held_flag(&mut self) {
        if self.flag_held && self.routed() {
            self.flag_held = false;
            self.deliver_routed(route::VECTOR_FLAG_SHIFT, "held flag", true);
        }
    }

    /// The command FIFO's level: the words waiting for the board's work.
    pub fn fifo_level(&self) -> u32 {
        self.fifo.level()
    }

    /// The FIFO's level went from 'before' to its current value. Falling below
    /// the model's low-water threshold (cfifo::LOW_WATER, a model choice)
    /// latches the low-water flag whatever the enable mask
    /// says; it reaches the driver only while the source is enabled.
    pub fn fifo_level_changed(&mut self, before: u32) {
        if before >= cfifo::LOW_WATER && self.fifo_level() < cfifo::LOW_WATER {
            self.intr_flags |= intr::FIFO_LOW_WATER;
            self.update_irq();
        }
    }

    /// A counted store's credit, to the processor interface the node names.
    pub fn return_credit(&mut self) {
        if let Some(pi) = &self.pi {
            pi_gfx_return_credits(pi.as_ref(), self.slot, 1);
        } else {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: counted graphics store has no processor interface for its return");
        }
    }
}

// The front registers.

/// Whether `addr` is a front register for this access: one of the
/// registers that do not wait for the board's work. The other accesses
/// that do not wait (the back end's read window, unmodeled offsets) read
/// the board's state directly (`BoardCell`).
pub(super) fn front_register(addr: u64, size: u32, write: bool) -> bool {
    if fifo_port(addr) || reg(addr, size).is_none() || access_waits(addr, size, write) {
        return false;
    }
    if !write {
        // Reads of these two fall to the unmodeled-access report.
        return !matches!(reg(addr, size), Some(intr::CLEAR | intr::ENABLE_CLEAR));
    }
    true
}

impl HostInterface {
    pub fn read(&mut self, addr: u64, size: u32) -> u64 {
        let mut value: u64 = 0;
        let sequence = self.trace_sequence;
        self.trace_sequence = self.trace_sequence.wrapping_add(1);

        match reg(addr, size) {
            Some(ODSY_IDENTITY) => value = self.identity as u64,
            Some(ODSY_CONTROL) => value = self.widget_control as u64,
            Some(ODSY_XT_REQ_TIMEOUT) => value = self.xt_req_timeout as u64,
            Some(ODSY_INT_DEST_UPPER) => value = self.int_dest_upper as u64,
            Some(ODSY_INT_DEST_LOWER) => value = self.int_dest_lower as u64,
            Some(ODSY_ERROR_COMMAND) => value = self.error_command as u64,
            Some(ODSY_STATUS | ODSY_ERROR_ADDR_UPPER | ODSY_ERROR_ADDR_LOWER) => {
                // No crosstalk error is ever posted, so the error class is clear.
            }
            Some(intr::FLAGS) => value = self.intr_flags as u64,
            Some(intr::ENABLE_SET) => value = self.intr_enable as u64,
            Some(ODSY_STATUS0) => {
                // The waiting stores are its level (HostInterface::fifo_level()).
                // Bit 15 and bit 17 use model-chosen thresholds, see cfifo::LOW_WATER
                // and reg::FIFO_SETUP_FIRST.
                let level = self.fifo_level();
                crate::pf_value!(self.fifo.pf_level, crate::perfetto::events::ODSY_CFIFO_LEVEL, level);
                value = (status0::FIFO_ENABLED
                    | if self.fifo.high_water() { status0::FIFO_HIGH_WATER } else { 0 }
                    | if level != 0 { 0 } else { status0::FIFO_EMPTY }
                    | if level < cfifo::LOW_WATER { status0::FIFO_LOW_WATER } else { 0 }) as u64;
            }
            Some(ODSY_SW_TOKEN) => value = self.sw_token as u64,
            Some(flctl::TARGET) => value = self.flctl_target as u64,
            Some(swap::ARM) => value = self.swap_arm as u64,
            Some(swap::PENDING) => value = self.swap_pending as u64,
            Some(swap::CONTROL) => value = self.swap_control as u64,
            Some(swap::CONTROL_COMPANION) => value = self.swap_control_companion as u64,
            // The buffer the last latched swap put on screen (retrace()).
            Some(ODSY_DISPLAYED_BUFFER) => value = self.displayed_buffer as u64,
            Some(route::DEST_UPPER | route::DEST_LOWER | route::VECTORS) => {
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: read of the write-only register at {:#x}", addr);
            }
            _ => unreachable!("odyssey_front_register() admits only these"),
        }
        crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: front MMIO read seq {} offset {:#x} size {} value {:#x}", sequence, addr, size, value);
        value
    }

    pub fn write(&mut self, addr: u64, value: u64, size: u32) {
        crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: front MMIO write seq {} offset {:#x} size {} value {:#x}", self.trace_sequence, addr, size, value);
        self.trace_sequence = self.trace_sequence.wrapping_add(1);

        match reg(addr, size) {
            Some(ODSY_CONTROL) => self.widget_control = value as u32,
            Some(ODSY_XT_REQ_TIMEOUT) => self.xt_req_timeout = value as u32,
            Some(ODSY_INT_DEST_UPPER) => self.int_dest_upper = value as u32,
            Some(ODSY_INT_DEST_LOWER) => self.int_dest_lower = value as u32,
            // Written to acknowledge; the data itself is discarded.
            Some(ODSY_ERROR_COMMAND) => self.error_command = 0,
            Some(intr::CLEAR) => {
                self.intr_flags &= !(value as u32);
                self.update_irq();
            }
            Some(intr::ENABLE_SET) => {
                self.intr_enable |= value as u32;
                self.update_irq();
            }
            Some(intr::ENABLE_CLEAR) => {
                self.intr_enable &= !(value as u32);
                self.update_irq();
            }
            Some(intr::FLAGS) => {
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
            Some(ODSY_SW_TOKEN) => self.sw_token = value as u32,
            Some(flctl::TARGET) => {
                self.flctl_target = value as u32;
                if !self.flctl_target_reported {
                    self.flctl_target_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: flow-control interrupt target {} and credit target {} stored; credits still return through the node interface",
                        (value as u32) >> flctl::INTR_TARGET_SHIFT,
                        ((value >> flctl::TARGET_SHIFT) as u32) & flctl::TARGET_MASK
                    );
                }
            }
            Some(swap::ARM) => {
                // Queue flips for the next retrace. Writes accumulate: the
                // driver can issue several windows or swap groups between two
                // retraces, and its retrace handler re-writes one window's bit
                // when a flip has not shown after two retraces. Whether the
                // board ORs or replaces is unknown; ORing loses no flip.
                self.swap_arm |= value as u32;
                crate::pf_instant!(crate::perfetto::events::swap::ARM, value);
            }
            Some(swap::PENDING) => self.swap_pending = value as u32,
            Some(swap::CONTROL) => {
                // The windows that must show buffer 1 from the next retrace on.
                // A later write to either register for the same window replaces
                // this one before the retrace latches it.
                self.swap_control = value as u32;
                self.swap_set |= value as u32;
                self.swap_clear &= !(value as u32);
            }
            Some(swap::CONTROL_COMPANION) => {
                self.swap_control_companion = value as u32;
                self.swap_clear |= value as u32;
                self.swap_set &= !(value as u32);
            }
            Some(ODSY_DISPLAYED_BUFFER) => {
                // Board-produced status; a guest write is not seen in the
                // capture, but store it rather than drop it.
                self.displayed_buffer = value as u32;
                self.displayed_written = true;
            }
            Some(route::DEST_UPPER) => {
                self.route_dest_upper = value as u32;
                self.deliver_held_flag();
            }
            Some(route::DEST_LOWER) => {
                self.route_dest_lower = value as u32;
                self.deliver_held_flag();
            }
            Some(route::VECTORS) => {
                self.route_vectors = value as u32;
                self.deliver_held_flag();
            }
            Some(ODSY_IDENTITY | ODSY_STATUS | ODSY_ERROR_ADDR_UPPER | ODSY_ERROR_ADDR_LOWER) => {
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: write of {:#x} to the read-only register at {:#x}", value, addr);
            }
            _ => unreachable!("odyssey_front_register() admits only these"),
        }
    }
}

/// A processor's read through the front alone: Some when it is a front
/// register, None when the access needs the board's state.
pub(super) fn front_mmio_read(f: &Mutex<HostInterface>, addr: u64, size: u32) -> Option<u64> {
    if !front_register(addr, size, false) {
        return None;
    }
    Some(f.lock().read(addr, size))
}

/// A processor's store through the front alone: true when it went to a
/// front register or joined the command FIFO for the board's thread, false
/// when it needs the board's state. `counted`: the processor interface counted
/// it.
pub(super) fn front_mmio_write(f: &Mutex<HostInterface>, fifo: &CommandFifo, addr: u64, value: u64, size: u32, counted: bool) -> bool {
    if front_register(addr, size, true) {
        let mut f = f.lock();
        f.write(addr, value, size);
        if counted {
            f.return_credit();
        }
        return true;
    }
    // A queued store needs the board's thread to run it, and the queue to
    // have room; otherwise the caller runs it.
    command_store(addr, size) && fifo.running.load(std::sync::atomic::Ordering::SeqCst) && fifo.push(f, addr, value, size, counted)
}

/// The fabric packet's address bits; the widget sits above them.
pub(super) const XIO_PACKET_ADDRESS_BITS: u32 = 48;

pub(super) const XIO_WIDGETS: u32 = 16;

/// The fabric address of a widget/address pair: the widget in bits 51:48.
pub(super) fn sgi_xio_compose(who: &str, widget: u32, address: u64, packet: &mut u64) -> bool {
    if widget >= XIO_WIDGETS || address >> XIO_PACKET_ADDRESS_BITS != 0 {
        crate::dlog!(LogModule::Odyssey, "{who}: packet fabric cannot encode widget {widget} address {address:#x}");
        return false;
    }
    *packet = (widget as u64) << XIO_PACKET_ADDRESS_BITS | address;
    true
}

/// HEART's interrupt-packet register in its widget space.
const HEART_PACKET: u64 = 0x80;

impl HostInterface {
    /// The interrupt-modify write to the interrupt target. HEART takes it at
    /// 0x80 of widget 8.
    pub fn send_interrupt(&mut self, packet: u64, number: u8, set: bool) {
        let widget = (packet >> XIO_PACKET_ADDRESS_BITS) as u32;
        let address = packet & ((1u64 << XIO_PACKET_ADDRESS_BITS) - 1);
        if let Some(send) = &self.packets {
            if !send(widget as u8, address, number, set) {
                crate::dlog!(LogModule::Odyssey, "odyssey: interrupt packet to widget {widget} address {address:#x} was refused");
            }
            return;
        }
        if widget != self.heart_widget || address != HEART_PACKET {
            crate::dlog!(LogModule::Odyssey, "odyssey: interrupt packet to widget {widget} address {address:#x} has no receiver");
            return;
        }
        if let Some(irq) = &self.irq {
            irq.set_line(number as u32, set);
        }
    }
}
