//! The serial ring-DMA engine of the SGI IOC3 and IOC4.
//!
//! Both chips put the same engine in front of each of their 16550-compatible
//! ports: seven 32-bit registers per port, a ring of eight-byte entries in
//! host memory for each direction, and a 64-bit ring base shared by a pair of
//! ports. The transmit walk writes the port's 16550; the receive side drains
//! its FIFO into a packer that becomes one ring entry. A ring's address is
//! derived from its port and direction, and no register reads it back.
//!
//! The chip around the engine owns its register window, its interrupt status
//! words and the 16550s. The engine reaches them through [`SerialDmaHost`], so the same
//! walk serves the IOC3 (two ports, nine interrupt bits each) and the IOC4
//! (four ports, eight bits each, DMA errors reported elsewhere).
//!
//! # What differs between the chips
//!
//! [`SerialDmaParams`] holds what the two do differently and nothing else:
//!
//! * IOC3 reports the packer's held-character count in the receive producer
//!   index. IOC4's register map does not define the field, so whether the
//!   chip has it is not known.
//! * IOC4 stops a port after a failed transfer until a channel reset (its
//!   error sources are named "serial halted by mem err"). Whether an IOC3
//!   port halts is not known.
//!
//! # What is approximated here
//!
//! * The receive timer and a transmit delay run on host time. A pause freezes
//!   both where they stand and a resume lets each finish the time it had
//!   left. No source says what the part does.
//! * Receive-ring-full is a level that follows a held entry; what the part
//!   does is not known.
//!
//! # Mapping to QEMU
//!
//! The split is QEMU's: this file is the shared engine, `ioc3.rs` and
//! `ioc4.rs` are the chips that use it.
//!
//! | QEMU (`hw/char/sgi-ioc-serial-dma.c`) | here |
//! |---|---|
//! | `SGIIOCSerialDMA`, `SGIIOCSerialDMAPort` | [`SerialDma`], [`SerialDmaPort`] |
//! | `SGIIOCSerialDMAParams` | [`SerialDmaParams`] |
//! | `SGIIOCSerialDMAOps` and its `opaque` | [`SerialDmaHost`], implemented by the chip |
//! | `SGIIOCSerialDMASource` | [`SerialDmaSource`] |
//! | the port's `SerialState *uart` | [`SerialDmaUart`] |
//! | `sgi_ioc_sdma_port_read`, `_port_write`, `_kick`, `_catch_up`, `_reset` | `port_read`, `port_write`, `run`, `catch_up`, `reset` |
//! | `sgi_ioc_sdma_base_read`, `_base_write` | `base_read`, `base_write` |
//!
//! Ported from hw/char/sgi-ioc-serial-dma.c in jamesbraid/qemu-sgi-origami,
//! in the form the IOC3 had before the engine was shared.

use std::time::{Duration, Instant};

use crate::dev::ioc3::{due, span, unspan, until, usize_field};
use crate::snapshot::{field, field_bool, field_u32, field_u32s, field_u64, field_u8s, StateTable};

use crate::dev::ns16550::Ns16550;

use crate::dev::ns16550::com_reg::{
    LSR_BI, LSR_DR, LSR_FE, LSR_OE, LSR_PE, MCR_DTR, MCR_LOOP, MCR_OUT1, MCR_OUT2, MCR_RTS, MSR_CTS, MSR_DCD,
    MSR_DCTS, MSR_DDCD,
};
use crate::dev::ns16550::com_reg::{LSR as UART_LSR, MCR as UART_MCR, MSR as UART_MSR, THR as UART_THR};

/// Port DMA register offsets (bytes, from the port's first register).
pub mod off {
    pub const CONTROL: u32 = 0x00;
    pub const TX_PRODUCE: u32 = 0x04;
    pub const TX_CONSUME: u32 = 0x08;
    pub const RX_PRODUCE: u32 = 0x0c;
    pub const RX_CONSUME: u32 = 0x10;
    pub const RX_TIMER: u32 = 0x14;
    pub const SHADOW: u32 = 0x18;
    /// Registers per port.
    pub const REGS: u32 = 7;
}

/// Bits of the port's control register.
pub mod ctl {
    pub const HIGH_WATER: u32 = 0x1ff;
    pub const TX_TIMER_RUN: u32 = 1 << 16;
    pub const FLOW_CONTROL: u32 = 1 << 17;
    pub const RX_TO_MEMORY: u32 = 1 << 27;
    pub const ENABLE: u32 = 1 << 28;
    pub const PAUSE: u32 = 1 << 29;
    pub const PAUSED: u32 = 1 << 30;
    pub const RESET: u32 = 1 << 31;
    /// Status and commands: these never read back as written.
    pub const UNSTORED: u32 = RESET | RX_TO_MEMORY | PAUSED | TX_TIMER_RUN;
}

/// The most ports one engine drives.
pub const MAX_PORTS: usize = 4;
/// Ports sharing one ring base.
const PAIR: usize = 2;

pub const ENTRY: usize = 8;
pub const ENTRY_CHARS: usize = 4;
/// Bit 31 of the receive consumer index arms the receive timer.
pub const ARM_RX_TIMER: u32 = 1 << 31;
/// The packer's held count, in the receive producer index (IOC3 only).
pub const HELD_SHIFT: u32 = 24;
const TIMER_RELOAD: u32 = 0xfff;
const TIMER_COUNT_SHIFT: u32 = 16;
pub const TIMER_TICK_NS: u64 = 62_500; // 16 kHz
const RING_4K: u32 = 1;
const RING_ADDRESS: u32 = 0xffff_f000;
const RING_1K_SIZE: u32 = 1024;
const RING_4K_SIZE: u32 = 4096;
const TX_TYPE: u8 = 0xc0;
const TX_CHARACTER: u8 = 0x40;
const TX_MODEM: u8 = 0x80;
pub const TX_DELAY: u8 = 0xc0;
pub const TX_INTERRUPT: u8 = 0x20;
const TX_UNDEFINED: u8 = 0x1f;
const RX_OVERRUN: u8 = 0x01;
const RX_PARITY: u8 = 0x02;
const RX_FRAMING: u8 = 0x04;
const RX_BREAK: u8 = 0x08;
const RX_CTS: u8 = 0x10;
const RX_CD: u8 = 0x20;
const RX_MODEM_OK: u8 = 0x40;
const RX_DATA_OK: u8 = 0x80;

/// A port's interrupt sources, numbered as the IOC3 lays them out in its
/// status word. The IOC4 places all but the DMA error the same way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SerialDmaSource {
    TxEmpty = 0,
    RxFull = 1,
    RxHighWater = 2,
    RxTimer = 3,
    TxExplicit = 7,
    DmaError = 8,
}

/// What differs between the chips, fixed per chip.
#[derive(Clone, Copy, Debug)]
pub struct SerialDmaParams {
    /// The receive producer index reports the packer's held count.
    pub held_counts: bool,
    /// A failed transfer stops the port until a channel reset.
    pub halt_on_dma_error: bool,
}

/// A 16550 as the engine drives it.
pub trait SerialDmaUart {
    fn loopback(&self) -> bool;
    /// The port can take a character from the engine. The walk waits while
    /// it cannot, and goes on when the chip next runs the engine.
    fn tx_ready(&self) -> bool;
    /// How long a character takes on the line at the programmed rate.
    fn char_time(&self) -> Duration;
    /// A read with the chip's side effects left out: the 16550 proper.
    fn read_core(&mut self, reg: u32) -> u8;
    fn write(&mut self, reg: u32, val: u8);
    fn lsr(&self) -> u8;
    fn mcr(&self) -> u8;
    fn msr(&self) -> u8;
}

/// The chip around the engine.
pub trait SerialDmaHost {
    fn uart(&mut self, port: usize) -> &mut dyn SerialDmaUart;
    /// Latch a source; the chip's write-one-to-clear acknowledges it.
    fn raise(&mut self, port: usize, src: SerialDmaSource);
    /// Drive a source that follows a level and ignores acknowledgement.
    fn set_level(&mut self, port: usize, src: SerialDmaSource, level: bool);
    /// The port's 16550 is fitted and the engine is aimed at it.
    fn reachable(&self, port: usize) -> bool;
    /// The 16550's interrupt line may have moved: reflect it in the chip's
    /// pass-through source.
    fn uart_line_changed(&mut self, port: usize);
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool;
    fn dma_write(&self, addr: u64, data: &[u8]) -> bool;
    /// A transfer failed at `addr`: a receive post or a transmit fetch.
    fn dma_error(&mut self, port: usize, _rx: bool, _addr: u64) {
        self.raise(port, SerialDmaSource::DmaError);
    }
}

#[derive(Default)]
pub struct SerialDmaPort {
    reg: [u32; off::REGS as usize],
    packer: [u8; ENTRY],
    packed: usize,
    tx_entry: [u8; ENTRY],
    tx_fetched: bool,
    tx_slot: usize,
    armed: bool,
    drain_pending: bool,
    busy: bool,
    halted: bool,
    /// The last post was turned away for lack of ring room.
    post_refused: bool,
    paused: bool,
    pub rx_deadline: Option<Instant>,
    rx_idle_count: u32,
    /// Time a frozen timer had left when a pause began.
    rx_frozen: Option<Duration>,
    tx_delay_deadline: Option<Instant>,
    tx_delay_frozen: Option<Duration>,
    tx_delay_irq: bool,
    /// A looped-back character still on the line: when it reaches the
    /// receiver. The 16550 core loops it back the moment it is written, and
    /// the engine does not look at the receiver until this time.
    loop_due: Option<Instant>,
}

pub struct SerialDma {
    params: SerialDmaParams,
    base_hi: [u32; MAX_PORTS / PAIR],
    base_lo: [u32; MAX_PORTS / PAIR],
    /// Transfers under way, for the chip's idle status. Nonzero only inside
    /// a call: a chip cannot read its own status mid-walk, so software sees
    /// idle unless a bus master it reaches re-enters the chip.
    in_flight: u32,
    pub port: Vec<SerialDmaPort>,
}

impl SerialDma {
    pub fn new(ports: usize, params: SerialDmaParams) -> SerialDma {
        assert!((1..=MAX_PORTS).contains(&ports));
        SerialDma {
            params,
            base_hi: [0; MAX_PORTS / PAIR],
            base_lo: [0; MAX_PORTS / PAIR],
            in_flight: 0,
            port: (0..ports).map(|_| SerialDmaPort::default()).collect(),
        }
    }

    /// Back to the power-on state: the ring bases and every port.
    pub fn reset(&mut self) {
        *self = SerialDma::new(self.port.len(), self.params);
    }

    pub fn base_read(&self, pair: usize, high: bool) -> u32 {
        if high { self.base_hi[pair] } else { self.base_lo[pair] }
    }

    pub fn base_write(&mut self, pair: usize, high: bool, val: u32) {
        if high {
            self.base_hi[pair] = val;
        } else {
            self.base_lo[pair] = val;
        }
    }

    /// A port's control register as stored.
    pub fn control(&self, p: usize) -> u32 {
        self.port[p].reg[0]
    }

    /// A register as stored, without the status the read adds.
    pub fn stored(&self, p: usize, offset: u32) -> u32 {
        self.port[p].reg[(offset / 4) as usize]
    }

    /// When the port's receive timer runs out, if it is running.
    pub fn rx_deadline(&self, p: usize) -> Option<Instant> {
        self.port[p].rx_deadline
    }

    /// When the port's transmit delay ends. A paused port keeps it frozen.
    pub fn tx_delay_deadline(&self, p: usize) -> Option<Instant> {
        let port = &self.port[p];
        if port.paused { None } else { port.tx_delay_deadline }
    }

    /// No transfer is under way: what a chip's DMA idle status reports.
    pub fn idle(&self) -> bool {
        self.in_flight == 0
    }

    /// A ring read, counted as in flight while the bus master has it.
    fn dma_read(&mut self, host: &mut dyn SerialDmaHost, addr: u64, buf: &mut [u8]) -> bool {
        self.in_flight += 1;
        let ok = host.dma_read(addr, buf);
        self.in_flight -= 1;
        ok
    }

    /// A ring write, counted as in flight while the bus master has it.
    fn dma_write(&mut self, host: &mut dyn SerialDmaHost, addr: u64, data: &[u8]) -> bool {
        self.in_flight += 1;
        let ok = host.dma_write(addr, data);
        self.in_flight -= 1;
        ok
    }

    /// When a looped-back character reaches the receiver, if one is on the
    /// line.
    pub fn loop_deadline(&self, p: usize) -> Option<Instant> {
        self.port[p].loop_due
    }

    /// Nothing queued, held or pending on the port: what a chip that wants
    /// the rings left alone while they move checks before it lets a guest
    /// change a mode.
    pub fn port_quiescent(&self, p: usize) -> bool {
        let port = &self.port[p];
        port.reg[1] == port.reg[2]
            && port.reg[3] == port.reg[4]
            && !port.tx_fetched
            && port.tx_delay_deadline.is_none()
            && port.packed == 0
            && !port.armed
            && !port.drain_pending
    }

    /// Both ports of a pair are off, with their indices at zero and nothing
    /// pending.
    pub fn pair_quiescent(&self, pair: usize) -> bool {
        (pair * PAIR..pair * PAIR + PAIR).all(|p| {
            self.control(p) & (ctl::ENABLE | ctl::PAUSE) == 0
                && self.port[p].reg[1..=4].iter().all(|&r| r == 0)
                && self.port_quiescent(p)
        })
    }

    /// Whether the port is stopped by a failed transfer.
    pub fn halted(&self, p: usize) -> bool {
        self.port[p].halted
    }

    /// Looped back with the engine's hardware flow control on, the port
    /// reports clear-to-send asserted even with request-to-send clear.
    pub fn flow_cts(&self, p: usize, loopback: bool) -> bool {
        loopback && self.control(p) & ctl::FLOW_CONTROL != 0
    }

    fn ring_size(&self, p: usize) -> u32 {
        if self.base_lo[p / PAIR] & RING_4K != 0 { RING_4K_SIZE } else { RING_1K_SIZE }
    }

    /// The selected ring size is also every index register's pointer width.
    fn mask(&self, p: usize) -> u32 {
        (self.ring_size(p) - 1) & !(ENTRY as u32 - 1)
    }

    /// The size-select bit is not address.
    fn ring(&self, p: usize, dir: u32) -> u64 {
        let pair = p / PAIR;
        let base = (u64::from(self.base_hi[pair]) << 32) | u64::from(self.base_lo[pair] & RING_ADDRESS);
        base.saturating_add(((p % PAIR) as u64 * 2 + u64::from(dir)) * u64::from(self.ring_size(p)))
    }

    fn running(&self, p: usize) -> bool {
        let c = self.control(p);
        !self.port[p].halted && c & ctl::ENABLE != 0 && c & ctl::PAUSE == 0
    }

    /// A transfer the fabric refused: the chip reports it, and may halt.
    fn dma_failed(&mut self, p: usize, rx: bool, addr: u64, host: &mut dyn SerialDmaHost) {
        if self.params.halt_on_dma_error {
            self.port[p].halted = true;
        }
        host.dma_error(p, rx, addr);
    }

    /// Transmit-ring-empty is a level: ring DMA on, nothing left in the ring
    /// or held by the engine.
    fn update_tx_empty(&self, p: usize, host: &mut dyn SerialDmaHost) {
        let mask = self.mask(p);
        let port = &self.port[p];
        let empty = self.control(p) & ctl::ENABLE != 0
            && (port.reg[1] ^ port.reg[2]) & mask == 0
            && !port.tx_fetched
            && port.tx_delay_deadline.is_none();
        host.set_level(p, SerialDmaSource::TxEmpty, empty);
    }

    /// The receive high-water mark counts ring entries; a level. So is
    /// receive-ring-full: set while ring DMA is on and an entry waits, turned
    /// away for lack of ring room. It does not come back with every refused
    /// retry after software has acknowledged it, and it falls once the
    /// consumer makes room, whether or not the held entry has posted yet. A
    /// ring filled exactly, with nothing waiting, does not raise it. A model
    /// choice; what the part does is not known.
    fn update_high_water(&mut self, p: usize, host: &mut dyn SerialDmaHost) {
        let mask = self.mask(p);
        let hw = self.control(p) & ctl::HIGH_WATER;
        let enabled = self.control(p) & ctl::ENABLE != 0;
        let port = &mut self.port[p];
        let waiting = port.reg[3].wrapping_sub(port.reg[4]) & mask;
        host.set_level(p, SerialDmaSource::RxHighWater, hw != 0 && waiting / ENTRY as u32 >= hw);
        if (port.reg[3] + ENTRY as u32) & mask != port.reg[4] & mask {
            port.post_refused = false;
        }
        host.set_level(p, SerialDmaSource::RxFull, enabled && port.post_refused);
    }

    /// Post the packer as one entry. Fails, keeping the characters, when the
    /// ring is full or the write is refused.
    fn post(&mut self, p: usize, host: &mut dyn SerialDmaHost) -> bool {
        let mask = self.mask(p);
        let ring = self.ring(p, 1);
        let (produce, consume, packer) = {
            let port = &self.port[p];
            (port.reg[3], port.reg[4], port.packer)
        };
        let next = (produce + ENTRY as u32) & mask;
        if next == consume & mask {
            // Full: refuse. The characters keep in the packer and the FIFO.
            self.port[p].post_refused = true;
            return false;
        }
        let at = ring.saturating_add(u64::from(produce & mask));
        if !self.dma_write(host, at, &packer) {
            self.dma_failed(p, true, at, host);
            return false;
        }
        let port = &mut self.port[p];
        port.reg[3] = next;
        port.post_refused = false;
        port.packed = 0;
        port.packer = [0; ENTRY];
        true
    }

    fn rx_timer_start(&mut self, p: usize, since: Instant) {
        let port = &mut self.port[p];
        let reload = port.reg[5] & TIMER_RELOAD;
        port.rx_idle_count = 0;
        let dur = Duration::from_nanos(u64::from(reload) * TIMER_TICK_NS);
        if port.paused {
            // Started while paused, it waits frozen at its full count.
            port.rx_frozen = Some(dur);
        } else {
            port.rx_deadline = Some(since + dur);
        }
    }

    fn rx_timer_expired(&mut self, p: usize, host: &mut dyn SerialDmaHost) {
        self.port[p].rx_idle_count = 0;
        self.port[p].rx_deadline = None;
        if self.port[p].packed != 0 {
            self.post(p, host);
        }
        // Raised at every expiry, even over an empty ring; the arm is spent.
        if self.port[p].armed {
            host.raise(p, SerialDmaSource::RxTimer);
        }
        self.port[p].armed = false;
        self.update_high_water(p, host);
    }

    /// Expiries already due happen before an access.
    pub fn catch_up(&mut self, now: Instant, host: &mut dyn SerialDmaHost) {
        for p in 0..self.port.len() {
            if let Some(d) = self.port[p].tx_delay_deadline {
                if !self.port[p].paused && d <= now {
                    self.tx_delay_ended(p, d, host);
                }
            }
            // Each arrival can send the next character; the ring bounds it.
            while let Some(d) = self.port[p].loop_due.filter(|&d| d <= now) {
                self.loop_arrived(p, d, host);
            }
            if let Some(d) = self.port[p].rx_deadline {
                if d <= now {
                    self.rx_timer_expired(p, host);
                }
            }
        }
    }

    /// A looped-back character reaches the receiver; the walk goes on from
    /// then.
    fn loop_arrived(&mut self, p: usize, when: Instant, host: &mut dyn SerialDmaHost) {
        self.port[p].loop_due = None;
        self.run_from(p, when, host);
    }

    fn tx_delay_ended(&mut self, p: usize, when: Instant, host: &mut dyn SerialDmaHost) {
        let irq = {
            let port = &mut self.port[p];
            port.tx_delay_deadline = None;
            std::mem::take(&mut port.tx_delay_irq)
        };
        if irq {
            host.raise(p, SerialDmaSource::TxExplicit);
        }
        self.run_from(p, when, host);
    }

    fn receive(&mut self, p: usize, since: Instant, host: &mut dyn SerialDmaHost) {
        // The 16550 core loops a character back the moment it is written; on
        // the part it takes a character time on the line to arrive.
        if self.port[p].loop_due.is_some() {
            return;
        }
        while self.running(p) {
            if self.port[p].packed == ENTRY_CHARS && !self.post(p, host) {
                break;
            }
            // Line status before the character: reading it clears the latches.
            let uart = host.uart(p);
            let lsr = uart.read_core(UART_LSR);
            if lsr & LSR_DR == 0 {
                break;
            }
            let data = uart.read_core(UART_THR);
            let mut msr = uart.read_core(UART_MSR);
            let looped = uart.loopback();
            if self.flow_cts(p, looped) {
                msr |= MSR_CTS;
            }
            let mut status = RX_DATA_OK | RX_MODEM_OK;
            for (bit, flag) in [(LSR_OE, RX_OVERRUN), (LSR_PE, RX_PARITY), (LSR_FE, RX_FRAMING), (LSR_BI, RX_BREAK)] {
                if lsr & bit != 0 {
                    status |= flag;
                }
            }
            if msr & MSR_CTS != 0 {
                status |= RX_CTS;
            }
            if msr & MSR_DCD != 0 {
                status |= RX_CD;
            }
            let start_timer = {
                let port = &mut self.port[p];
                port.packer[port.packed] = data;
                port.packer[ENTRY_CHARS + port.packed] = status;
                port.packed += 1;
                port.armed && port.rx_deadline.is_none() && port.rx_frozen.is_none()
            };
            // Started by the first character after the arm; later ones leave
            // a running timer alone.
            if start_timer {
                self.rx_timer_start(p, since);
            }
            // A fourth character completes the entry; fewer wait for the
            // timer or a drain.
            if self.port[p].packed == ENTRY_CHARS {
                self.post(p, host);
            }
        }
        host.uart_line_changed(p);
    }

    /// The type field and bit 5 decide what a slot does; the low five bits
    /// have no defined meaning and are reported, not obeyed. Report them once,
    /// when the entry is fetched, for the slots up to the one that ends it.
    fn check_entry(p: usize, entry: &[u8; ENTRY]) {
        let mut undefined = 0;
        for &control in &entry[ENTRY_CHARS..] {
            undefined |= control & TX_UNDEFINED;
            if control & TX_TYPE == 0 {
                break;
            }
        }
        if undefined != 0 {
            crate::dlog!(
                crate::devlog::LogModule::Ioc,
                "serial DMA: undefined TX control bits {undefined:#04x} in an entry on port {p}"
            );
        }
    }

    /// Each control byte types its data byte; the empty type ends the entry,
    /// wherever it sits.
    fn transmit(&mut self, p: usize, since: Instant, host: &mut dyn SerialDmaHost) {
        let mask = self.mask(p);
        let ring = self.ring(p, 0);
        if self.port[p].tx_delay_deadline.is_some() {
            return;
        }
        loop {
            let (produce, consume) = {
                let port = &self.port[p];
                (port.reg[1], port.reg[2])
            };
            if !(self.running(p) && consume & mask != produce & mask) {
                break;
            }
            let was_fetched = self.port[p].tx_fetched;
            if !was_fetched {
                let mut entry = [0u8; ENTRY];
                let at = ring.saturating_add(u64::from(consume & mask));
                if !self.dma_read(host, at, &mut entry) {
                    self.dma_failed(p, false, at, host);
                    break;
                }
                let port = &mut self.port[p];
                port.tx_entry = entry;
                port.tx_fetched = true;
            }
            let entry = self.port[p].tx_entry;
            if !was_fetched {
                Self::check_entry(p, &entry);
            }
            let mut slot = self.port[p].tx_slot;
            while slot < ENTRY_CHARS {
                let control = entry[ENTRY_CHARS + slot];
                let ty = control & TX_TYPE;
                if ty == 0 {
                    break;
                }
                match ty {
                    TX_CHARACTER => {
                        // In loopback the line carries one character at a time.
                        if !host.uart(p).tx_ready() || self.port[p].loop_due.is_some() {
                            return;
                        }
                        let looped = host.uart(p).loopback();
                        // What has already arrived goes ahead of this one.
                        if looped {
                            self.receive(p, since, host);
                        }
                        host.uart(p).write(UART_THR, entry[slot]);
                        if looped {
                            self.port[p].loop_due = Some(since + host.uart(p).char_time());
                        }
                        // Drain per character or loopback overruns the FIFO.
                        self.receive(p, since, host);
                    }
                    TX_MODEM => host.uart(p).write(UART_MCR, entry[slot]),
                    TX_DELAY => {
                        // The data byte is milliseconds of quiet.
                        if entry[slot] != 0 {
                            let port = &mut self.port[p];
                            port.tx_slot = slot + 1;
                            port.tx_delay_deadline = Some(since + Duration::from_millis(u64::from(entry[slot])));
                            port.tx_delay_irq = control & TX_INTERRUPT != 0;
                            return;
                        }
                    }
                    _ => unreachable!("the type field has two bits"),
                }
                slot += 1;
                self.port[p].tx_slot = slot;
                if control & TX_INTERRUPT != 0 {
                    host.raise(p, SerialDmaSource::TxExplicit);
                }
            }
            let port = &mut self.port[p];
            port.tx_slot = 0;
            port.tx_fetched = false;
            port.reg[2] = (port.reg[2] + ENTRY as u32) & mask;
        }
        host.uart_line_changed(p);
    }

    /// The drain posts what the packer holds and reads back set until it has;
    /// it waits while ring DMA is off or paused and while the ring is full.
    fn drain(&mut self, p: usize, host: &mut dyn SerialDmaHost) {
        if !self.port[p].drain_pending || !self.running(p) {
            return;
        }
        if self.port[p].packed != 0 && !self.post(p, host) {
            return;
        }
        self.port[p].drain_pending = false;
    }

    /// Walk the port's rings from `since`: where a transmit delay ended, or
    /// now.
    pub fn run_from(&mut self, p: usize, since: Instant, host: &mut dyn SerialDmaHost) {
        if !host.reachable(p) || self.port[p].busy {
            return;
        }
        self.port[p].busy = true;
        self.transmit(p, since, host);
        self.receive(p, since, host);
        self.drain(p, host);
        self.update_high_water(p, host);
        self.update_tx_empty(p, host);
        self.port[p].busy = false;
    }

    pub fn run(&mut self, p: usize, host: &mut dyn SerialDmaHost) {
        self.run_from(p, Instant::now(), host);
    }

    /// A passive mirror of the 16550's lines; no read side effects.
    fn shadow(&self, p: usize, host: &mut dyn SerialDmaHost) -> u32 {
        let u = host.uart(p);
        let (lsr, mcr, mut msr) = (u.lsr(), u.mcr(), u.msr());
        if self.flow_cts(p, u.loopback()) {
            msr |= MSR_CTS;
        }
        let mut s = u32::from(lsr);
        for (m, bit) in [(MSR_DCTS, 16), (MSR_DDCD, 19), (MSR_CTS, 20), (MSR_DCD, 23)] {
            if msr & m != 0 {
                s |= 1 << bit;
            }
        }
        for (m, bit) in [(MCR_DTR, 24), (MCR_RTS, 25), (MCR_OUT1, 26), (MCR_OUT2, 27), (MCR_LOOP, 28)] {
            if mcr & m != 0 {
                s |= 1 << bit;
            }
        }
        s
    }

    pub fn port_read(&mut self, p: usize, offset: u32, host: &mut dyn SerialDmaHost) -> u32 {
        let now = Instant::now();
        self.catch_up(now, host);
        if offset == off::SHADOW {
            return self.shadow(p, host);
        }
        let port = &self.port[p];
        let mut v = port.reg[(offset / 4) as usize];
        match offset {
            off::CONTROL => {
                if v & ctl::PAUSE != 0 {
                    v |= ctl::PAUSED;
                }
                if port.tx_delay_deadline.is_some() {
                    v |= ctl::TX_TIMER_RUN;
                }
                if port.drain_pending {
                    v |= ctl::RX_TO_MEMORY;
                }
            }
            off::RX_CONSUME => {
                if port.armed {
                    v |= ARM_RX_TIMER;
                }
            }
            off::RX_PRODUCE if self.params.held_counts => v |= (port.packed as u32) << HELD_SHIFT,
            off::RX_TIMER => {
                // Ticks left while running or frozen; idle, whatever a reload
                // write last loaded.
                let ticks_in = |d: Duration| (d.as_nanos() as u64 / TIMER_TICK_NS).min(u64::from(TIMER_RELOAD)) as u32;
                let mut ticks = port.rx_idle_count;
                if let (true, Some(left)) = (port.paused, port.rx_frozen) {
                    ticks = ticks_in(left);
                } else if let Some(d) = port.rx_deadline {
                    ticks = ticks_in(d.saturating_duration_since(now));
                }
                v |= ticks << TIMER_COUNT_SHIFT;
            }
            _ => {}
        }
        v
    }

    /// Resets the four indices and the work in progress; the settings
    /// survive, and a port a failed transfer stopped runs again.
    fn reset_port(&mut self, p: usize) {
        let port = &mut self.port[p];
        port.halted = false;
        port.reg[1..=4].fill(0);
        port.armed = false;
        port.drain_pending = false;
        port.post_refused = false;
        port.packed = 0;
        port.packer = [0; ENTRY];
        port.tx_slot = 0;
        port.tx_fetched = false;
        port.tx_delay_deadline = None;
        port.tx_delay_irq = false;
        port.rx_deadline = None;
        port.rx_frozen = None;
        port.rx_idle_count = 0;
    }

    pub fn port_write(&mut self, p: usize, offset: u32, val: u32, host: &mut dyn SerialDmaHost) {
        let now = Instant::now();
        self.catch_up(now, host);
        let mask = self.mask(p);
        match offset {
            off::CONTROL => {
                self.port[p].reg[0] = val & !ctl::UNSTORED;
                if val & ctl::RESET != 0 {
                    self.reset_port(p);
                }
                if val & ctl::RX_TO_MEMORY != 0 {
                    self.port[p].drain_pending = true;
                }
                let port = &mut self.port[p];
                // A pause freezes the receive timer and a transmit delay;
                // resuming lets each finish the time it had left. A model
                // choice; no source says what the part does.
                if val & ctl::PAUSE != 0 && !port.paused {
                    port.paused = true;
                    port.rx_frozen = port.rx_deadline.take().map(|d| d.saturating_duration_since(now));
                    port.tx_delay_frozen = port.tx_delay_deadline.take().map(|d| d.saturating_duration_since(now));
                } else if val & ctl::PAUSE == 0 && port.paused {
                    port.paused = false;
                    port.rx_deadline = port.rx_frozen.take().map(|l| now + l);
                    port.tx_delay_deadline = port.tx_delay_frozen.take().map(|l| now + l);
                }
                if val & ctl::ENABLE == 0 {
                    port.rx_deadline = None;
                    port.rx_frozen = None;
                }
                self.run(p, host);
            }
            off::TX_PRODUCE | off::TX_CONSUME | off::RX_PRODUCE => {
                self.port[p].reg[(offset / 4) as usize] = val & mask;
                self.run(p, host);
            }
            off::RX_CONSUME => {
                // Bit 31 sets the arm; a write with it clear leaves it.
                self.port[p].reg[4] = val & mask;
                if val & ARM_RX_TIMER != 0 {
                    self.port[p].armed = true;
                }
                self.run(p, host);
            }
            off::RX_TIMER => {
                let port = &mut self.port[p];
                port.reg[5] = val & TIMER_RELOAD;
                port.rx_idle_count = val & TIMER_RELOAD;
            }
            // The shadow discards writes silently.
            _ => {}
        }
        self.update_tx_empty(p, host);
        self.update_high_water(p, host);
    }
}

impl SerialDmaUart for Ns16550 {
    fn loopback(&self) -> bool {
        Ns16550::loopback(self)
    }

    fn tx_ready(&self) -> bool {
        Ns16550::tx_ready(self)
    }

    fn char_time(&self) -> Duration {
        Ns16550::char_time(self)
    }

    fn read_core(&mut self, r: u32) -> u8 {
        self.read(r)
    }

    fn write(&mut self, r: u32, v: u8) {
        Ns16550::write(self, r, v);
    }

    fn lsr(&self) -> u8 {
        Ns16550::lsr(self)
    }

    fn mcr(&self) -> u8 {
        Ns16550::mcr(self)
    }

    fn msr(&self) -> u8 {
        Ns16550::msr(self)
    }
}

#[cfg(test)]
mod tests {
    //! The engine against a stand-in chip. The IOC3 and IOC4 register tests
    //! cover the engine through each chip; these cover what only the second
    //! chip's parameters change.

    use super::*;

    #[derive(Default)]
    struct FakeUart {
        sent: Vec<u8>,
        rx: std::collections::VecDeque<u8>,
    }

    impl SerialDmaUart for FakeUart {
        fn loopback(&self) -> bool {
            false
        }
        fn tx_ready(&self) -> bool {
            true
        }
        fn char_time(&self) -> Duration {
            Duration::ZERO
        }
        fn read_core(&mut self, reg: u32) -> u8 {
            match reg {
                UART_LSR => u8::from(!self.rx.is_empty()),
                UART_THR => self.rx.pop_front().unwrap_or(0),
                _ => 0,
            }
        }
        fn write(&mut self, reg: u32, val: u8) {
            if reg == UART_THR {
                self.sent.push(val);
            }
        }
        fn lsr(&self) -> u8 {
            0
        }
        fn mcr(&self) -> u8 {
            0
        }
        fn msr(&self) -> u8 {
            0
        }
    }

    struct Chip {
        uart: [FakeUart; MAX_PORTS],
        mem: Vec<u8>,
        /// Transfers below this address are refused.
        refuse_below: u64,
        raised: Vec<(usize, SerialDmaSource)>,
        errors: Vec<(usize, bool, u64)>,
        /// Whether the chip reports errors itself, as IOC4 does.
        reports_errors: bool,
    }

    impl Chip {
        fn new() -> Chip {
            Chip {
                uart: Default::default(),
                mem: vec![0; 0x4000],
                refuse_below: 0,
                raised: Vec::new(),
                errors: Vec::new(),
                reports_errors: true,
            }
        }
    }

    impl SerialDmaHost for Chip {
        fn uart(&mut self, port: usize) -> &mut dyn SerialDmaUart {
            &mut self.uart[port]
        }
        fn raise(&mut self, port: usize, src: SerialDmaSource) {
            self.raised.push((port, src));
        }
        fn set_level(&mut self, _port: usize, _src: SerialDmaSource, _level: bool) {}
        fn reachable(&self, _port: usize) -> bool {
            true
        }
        fn uart_line_changed(&mut self, _port: usize) {}
        fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
            if addr < self.refuse_below {
                return false;
            }
            buf.copy_from_slice(&self.mem[addr as usize..addr as usize + buf.len()]);
            true
        }
        fn dma_write(&self, _addr: u64, _data: &[u8]) -> bool {
            true
        }
        fn dma_error(&mut self, port: usize, rx: bool, addr: u64) {
            if self.reports_errors {
                self.errors.push((port, rx, addr));
            } else {
                self.raise(port, SerialDmaSource::DmaError);
            }
        }
    }

    const FOUR: SerialDmaParams = SerialDmaParams { held_counts: false, halt_on_dma_error: true };
    const TWO: SerialDmaParams = SerialDmaParams { held_counts: true, halt_on_dma_error: false };

    /// One transmit entry holding `byte`, in the port's transmit ring.
    fn queue_char(sdma: &mut SerialDma, chip: &mut Chip, p: usize, byte: u8) {
        let at = sdma.ring(p, 0) as usize;
        chip.mem[at..at + ENTRY].copy_from_slice(&[byte, 0, 0, 0, TX_CHARACTER, 0, 0, 0]);
        let produce = sdma.stored(p, off::TX_PRODUCE);
        sdma.port_write(p, off::TX_PRODUCE, produce + ENTRY as u32, chip);
    }

    fn enable(sdma: &mut SerialDma, chip: &mut Chip, p: usize) {
        sdma.port_write(p, off::CONTROL, ctl::ENABLE, chip);
    }

    #[test]
    fn a_port_in_the_second_pair_uses_the_second_ring_base() {
        let mut sdma = SerialDma::new(4, FOUR);
        let mut chip = Chip::new();
        sdma.base_write(0, false, 0x1000);
        sdma.base_write(1, false, 0x2000);
        assert_eq!(sdma.ring(0, 0), 0x1000);
        assert_eq!(sdma.ring(1, 0), 0x1000 + 2 * 1024);
        assert_eq!(sdma.ring(2, 0), 0x2000);
        assert_eq!(sdma.ring(3, 1), 0x2000 + 3 * 1024);
        enable(&mut sdma, &mut chip, 3);
        queue_char(&mut sdma, &mut chip, 3, b'x');
        assert_eq!(chip.uart[3].sent, b"x");
        assert!(chip.uart[0].sent.is_empty(), "the other ports saw nothing");
    }

    #[test]
    fn a_failed_fetch_halts_the_port_until_a_channel_reset() {
        let mut sdma = SerialDma::new(4, FOUR);
        let mut chip = Chip::new();
        sdma.base_write(0, false, 0x1000);
        enable(&mut sdma, &mut chip, 0);
        chip.refuse_below = 0x2000;
        queue_char(&mut sdma, &mut chip, 0, b'a');
        assert_eq!(chip.errors, [(0, false, 0x1000)], "the chip is told where the fetch failed");
        assert!(sdma.halted(0));
        chip.refuse_below = 0;
        sdma.run(0, &mut chip);
        assert!(chip.uart[0].sent.is_empty(), "a halted port does not resume by itself");
        sdma.port_write(0, off::CONTROL, ctl::ENABLE | ctl::RESET, &mut chip);
        assert!(!sdma.halted(0));
        assert_eq!(sdma.stored(0, off::TX_PRODUCE), 0, "the reset cleared the indices");
        queue_char(&mut sdma, &mut chip, 0, b'b');
        assert_eq!(chip.uart[0].sent, b"b");
    }

    #[test]
    fn a_failed_fetch_leaves_a_chip_that_does_not_halt_running() {
        let mut sdma = SerialDma::new(2, TWO);
        let mut chip = Chip::new();
        chip.reports_errors = false;
        sdma.base_write(0, false, 0x1000);
        enable(&mut sdma, &mut chip, 0);
        chip.refuse_below = 0x2000;
        queue_char(&mut sdma, &mut chip, 0, b'a');
        assert_eq!(chip.raised.last(), Some(&(0, SerialDmaSource::DmaError)), "the default report is the port's own source");
        assert!(!sdma.halted(0));
        chip.refuse_below = 0;
        sdma.run(0, &mut chip);
        assert_eq!(chip.uart[0].sent, b"a", "the walk retries once the transfer works");
    }

    #[test]
    fn only_a_chip_that_reports_held_counts_shows_them() {
        for (params, shown) in [(TWO, true), (FOUR, false)] {
            let mut sdma = SerialDma::new(2, params);
            let mut chip = Chip::new();
            sdma.base_write(0, false, 0x1000);
            enable(&mut sdma, &mut chip, 0);
            chip.uart[0].rx.extend([1, 2]);
            sdma.run(0, &mut chip);
            let produce = sdma.port_read(0, off::RX_PRODUCE, &mut chip);
            assert_eq!(produce >> HELD_SHIFT & 7, if shown { 2 } else { 0 });
        }
    }

    #[test]
    fn the_engine_is_idle_between_calls_whether_a_transfer_worked_or_not() {
        let mut sdma = SerialDma::new(4, FOUR);
        let mut chip = Chip::new();
        assert!(sdma.idle());
        sdma.base_write(0, false, 0x1000);
        enable(&mut sdma, &mut chip, 0);
        queue_char(&mut sdma, &mut chip, 0, b'a');
        assert!(sdma.idle(), "after a good transfer");
        chip.refuse_below = 0x2000;
        queue_char(&mut sdma, &mut chip, 0, b'b');
        assert!(sdma.halted(0));
        assert!(sdma.idle(), "after a failed one");
    }
}

// The engine's state for a snapshot of the chip that carries it. Each timer
// is saved as the time it had left (see the IOC3's snapshot helpers).

impl SerialDmaPort {
    fn save_state(&self, now: Instant) -> toml::Value {
        StateTable::new()
            .u32s("reg", &self.reg)
            .u8s("packer", &self.packer)
            .u64("packed", self.packed as u64)
            .u8s("tx_entry", &self.tx_entry)
            .bool("tx_fetched", self.tx_fetched)
            .u64("tx_slot", self.tx_slot as u64)
            .bool("armed", self.armed)
            .bool("drain_pending", self.drain_pending)
            .bool("busy", self.busy)
            .bool("paused", self.paused)
            .u64("rx_due", until(self.rx_deadline, now))
            .u32("rx_idle_count", self.rx_idle_count)
            .u64("rx_frozen", span(self.rx_frozen))
            .u64("tx_delay_due", until(self.tx_delay_deadline, now))
            .u64("tx_delay_frozen", span(self.tx_delay_frozen))
            .bool("tx_delay_irq", self.tx_delay_irq)
            .bool("post_refused", self.post_refused)
            .bool("halted", self.halted)
            .u64("loop_due", until(self.loop_due, now))
            .build()
    }
    fn load_state(v: &toml::Value, now: Instant) -> Result<Self, String> {
        Ok(Self {
            reg: field_u32s(v, "reg")?,
            packer: field_u8s(v, "packer")?,
            packed: usize_field(v, "packed", ENTRY)?,
            tx_entry: field_u8s(v, "tx_entry")?,
            tx_fetched: field_bool(v, "tx_fetched")?,
            tx_slot: usize_field(v, "tx_slot", ENTRY)?,
            armed: field_bool(v, "armed")?,
            drain_pending: field_bool(v, "drain_pending")?,
            busy: field_bool(v, "busy")?,
            paused: field_bool(v, "paused")?,
            rx_deadline: due(field_u64(v, "rx_due")?, now),
            rx_idle_count: field_u32(v, "rx_idle_count")?,
            rx_frozen: unspan(field_u64(v, "rx_frozen")?),
            tx_delay_deadline: due(field_u64(v, "tx_delay_due")?, now),
            tx_delay_frozen: unspan(field_u64(v, "tx_delay_frozen")?),
            tx_delay_irq: field_bool(v, "tx_delay_irq")?,
            post_refused: field_bool(v, "post_refused")?,
            halted: field_bool(v, "halted")?,
            loop_due: due(field_u64(v, "loop_due")?, now),
        })
    }
}

impl SerialDma {
    /// The ring bases and each port's state, timers as the time left after
    /// `now`.
    pub fn save_state(&self, now: Instant) -> toml::Value {
        let mut t = StateTable::new();
        t.u32s("base_hi", &self.base_hi).u32s("base_lo", &self.base_lo);
        for (i, port) in self.port.iter().enumerate() {
            t.value(&format!("port{i}"), port.save_state(now));
        }
        t.build()
    }

    /// The engine `save_state` described, with this engine's parameters
    /// and port count, timers set that far after `now`.
    pub fn loaded(&self, v: &toml::Value, now: Instant) -> Result<SerialDma, String> {
        let port = (0..self.port.len())
            .map(|i| SerialDmaPort::load_state(field(v, &format!("port{i}"))?, now))
            .collect::<Result<Vec<_>, String>>()?;
        // Transfers are under way only inside a call, never at a save.
        Ok(SerialDma { params: self.params, base_hi: field_u32s(v, "base_hi")?, base_lo: field_u32s(v, "base_lo")?, in_flight: 0, port })
    }
}
