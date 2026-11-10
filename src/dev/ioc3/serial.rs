//! One serial port's ring-DMA engine, the IOC3 logic in front of a 16550.
//!
//! The engine does not hold the UART: it reaches it through the Super I/O
//! window (`superio.rs`) at the offset chip control gives it, as the part's
//! own bus master would, and aims a status bit at the chip's interrupt word.
//! Software either drives the UART directly (pass-through, the mode Linux's
//! 8250 driver and OpenBSD's `com` use) or hands the engine rings in memory.
//!
//! Nothing is timed to a baud rate: serial transmit completes inside the
//! write and a loopback character arrives at once (the part takes a
//! character time). Ring DMA, the receive timer and transmit delays are
//! evaluated lazily on every register access, and an `hptimer` one-shot per
//! deadline runs the same catch-up while the guest is idle.
//!
//! Ported from hw/misc/sgi-ioc3.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use super::superio::{
    uart_reg, SuperIo, LSR_BI, LSR_DR, LSR_FE, LSR_OE, LSR_PE, MCR_DTR, MCR_LOOP, MCR_OUT1, MCR_OUT2, MCR_RTS,
    MSR_CTS, MSR_DCD, MSR_DCTS,
};
use super::{decode, due, encode, int, reg, span, unspan, until, usize_field, Core};
use crate::pci::PciDma;
use crate::snapshot::{field_bool, field_u32, field_u32s, field_u64, field_u8s, StateTable};

/// Port DMA register offsets (bytes, from the port's first register).
pub(super) mod sdma_off {
    pub const CONTROL: u32 = 0x00;
    pub const TX_PRODUCE: u32 = 0x04;
    pub const TX_CONSUME: u32 = 0x08;
    pub const RX_PRODUCE: u32 = 0x0c;
    pub const RX_CONSUME: u32 = 0x10;
    pub const RX_TIMER: u32 = 0x14;
    pub const SHADOW: u32 = 0x18;
    pub const REGS: u32 = 7;
}

/// Registers a page-aliased write cannot reach: the two indexes the engine
/// advances and the 16550 shadow.
pub(super) fn alias_read_only(off: u32) -> bool {
    matches!(off, sdma_off::TX_CONSUME | sdma_off::RX_PRODUCE | sdma_off::SHADOW)
}
pub(super) mod sdma_ctl {
    pub const HIGH_WATER: u32 = 0x1ff;
    pub const TX_TIMER_RUN: u32 = 1 << 16;
    pub const FLOW_CONTROL: u32 = 1 << 17;
    pub const RX_TO_MEMORY: u32 = 1 << 27;
    pub const ENABLE: u32 = 1 << 28;
    pub const PAUSE: u32 = 1 << 29;
    pub const PAUSED: u32 = 1 << 30;
    pub const RESET: u32 = 1 << 31;
    pub const UNSTORED: u32 = RESET | RX_TO_MEMORY | PAUSED | TX_TIMER_RUN;
}
pub(super) const ENTRY: usize = 8;
pub(super) const ENTRY_CHARS: usize = 4;
pub(super) const ARM_RX_TIMER: u32 = 1 << 31;
pub(super) const HELD_SHIFT: u32 = 24;
pub(super) const TIMER_RELOAD: u32 = 0xfff;
pub(super) const TIMER_COUNT_SHIFT: u32 = 16;
pub(super) const TIMER_TICK_NS: u64 = 62_500; // 16 kHz
pub(super) const TX_TYPE: u8 = 0xc0;
pub(super) const TX_CHARACTER: u8 = 0x40;
pub(super) const TX_MODEM: u8 = 0x80;
pub(super) const TX_DELAY: u8 = 0xc0;
pub(super) const TX_INTERRUPT: u8 = 0x20;
pub(super) const RX_OVERRUN: u8 = 0x01;
pub(super) const RX_PARITY: u8 = 0x02;
pub(super) const RX_FRAMING: u8 = 0x04;
pub(super) const RX_BREAK: u8 = 0x08;
pub(super) const RX_CTS: u8 = 0x10;
pub(super) const RX_CD: u8 = 0x20;
pub(super) const RX_MODEM_OK: u8 = 0x40;
pub(super) const RX_DATA_OK: u8 = 0x80;
/// Chip control: where each engine finds its 16550 (byte offset / 8).
pub(super) const CHIP_CTRL_PORT_A_SHIFT: u32 = 1;
pub(super) const CHIP_CTRL_PORT_B_SHIFT: u32 = 8;
pub(super) const CHIP_CTRL_PORT_FIELD: u32 = 0x7f;

/// One port's engine state, under one lock with the engine that advances it.
#[derive(Default)]
pub(super) struct SdmaPort {
    reg: [u32; 7],
    packer: [u8; ENTRY],
    packed: usize,
    tx_entry: [u8; ENTRY],
    tx_fetched: bool,
    tx_slot: usize,
    armed: bool,
    drain_pending: bool,
    busy: bool,
    paused: bool,
    rx_deadline: Option<Instant>,
    rx_idle_count: u32,
    /// Time a frozen timer had left when a pause began.
    rx_frozen: Option<Duration>,
    tx_delay_deadline: Option<Instant>,
    tx_delay_frozen: Option<Duration>,
    tx_delay_irq: bool,
    /// An entry waits that the receive ring had no room for; drives
    /// receive-ring-full while ring DMA is on.
    post_refused: bool,
}


pub(super) struct Serial {
    p: usize,
    core: Arc<Core>,
    sio: Arc<SuperIo>,
    st: Mutex<SdmaPort>,
    /// The deadlines in `st`, kept where a timer can read them without the
    /// lock: nanoseconds after the process epoch plus one, zero for none.
    rx_due: AtomicU64,
    tx_due: AtomicU64,
}

impl Serial {
    pub(super) fn new(p: usize, core: Arc<Core>, sio: Arc<SuperIo>) -> Serial {
        Serial { p, core, sio, st: Mutex::new(SdmaPort::default()), rx_due: AtomicU64::new(0), tx_due: AtomicU64::new(0) }
    }

    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.core.dma().is_some_and(|d| d.dma_read(addr, buf).is_ok())
    }

    fn dma_write(&self, addr: u64, data: &[u8]) -> bool {
        self.core.dma().is_some_and(|d| d.dma_write(addr, data).is_ok())
    }

    fn ring_size(&self) -> u32 {
        if self.core.ring_lo.load(Ordering::SeqCst) & 1 != 0 { 4096 } else { 1024 }
    }

    fn mask(&self) -> u32 {
        (self.ring_size() - 1) & !(ENTRY as u32 - 1)
    }

    /// The ring in `dir` (0 transmit, 1 receive) for this port.
    fn ring(&self, _st: &SdmaPort, dir: u32) -> u64 {
        let base = ((self.core.ring_hi.load(Ordering::SeqCst) as u64) << 32)
            | (self.core.ring_lo.load(Ordering::SeqCst) & 0xffff_f000) as u64;
        base.saturating_add(((self.p as u64 % 2) * 2 + dir as u64) * self.ring_size() as u64)
    }

    fn raise(&self, src: u32) {
        self.core.set_bits(int::serial(self.p, src));
    }

    fn level(&self, src: u32, level: bool) {
        self.core.level(int::serial(self.p, src), level);
    }

    /// Chip control places each port's 16550 for its engine; aimed anywhere
    /// else, the engine does not reach the port.
    fn reachable(&self, _st: &SdmaPort) -> bool {
        let (shift, base) = if self.p == 1 {
            (CHIP_CTRL_PORT_B_SHIFT, reg::SERIAL_B)
        } else {
            (CHIP_CTRL_PORT_A_SHIFT, reg::SERIAL_A)
        };
        (self.core.chip_control.load(Ordering::SeqCst) >> shift) & CHIP_CTRL_PORT_FIELD == (base & 0xfff) / 8
    }

    /// The deadlines the engine holds, for the timers: the receive timer and
    /// the transmit delay (frozen, so none, while paused).
    fn due_of(st: &SdmaPort) -> (Option<Instant>, Option<Instant>) {
        (st.rx_deadline, if st.paused { None } else { st.tx_delay_deadline })
    }

    fn publish_due(&self, st: &SdmaPort) {
        let (rx, tx) = Self::due_of(st);
        self.rx_due.store(encode(rx), Ordering::SeqCst);
        self.tx_due.store(encode(tx), Ordering::SeqCst);
    }

    /// The receive-timer and transmit-delay deadlines as of the last access,
    /// without taking the engine's lock.
    pub(super) fn due(&self) -> (Option<Instant>, Option<Instant>) {
        (decode(self.rx_due.load(Ordering::SeqCst)), decode(self.tx_due.load(Ordering::SeqCst)))
    }

    /// Run `f` with the engine locked and publish its deadlines after.
    fn with<R>(&self, f: impl FnOnce(&Self, &mut SdmaPort) -> R) -> R {
        let mut st = self.st.lock();
        let r = f(self, &mut st);
        self.publish_due(&st);
        r
    }

    #[cfg(test)]
    pub(super) fn rx_deadline(&self) -> Option<Instant> {
        self.st.lock().rx_deadline
    }

    #[cfg(test)]
    pub(super) fn set_rx_deadline(&self, due: Option<Instant>) {
        self.with(|_, st| st.rx_deadline = due);
    }

    pub(super) fn is_locked(&self) -> bool {
        self.st.is_locked()
    }

    /// Run the engine on whatever work is waiting.
    pub(super) fn kick(&self) {
        self.with(|s, st| s.run(st));
    }

    /// Expiries already due happen before an access.
    pub(super) fn catch_up(&self, now: Instant) {
        self.with(|s, st| s.catch_up_locked(st, now));
    }

    /// A register of this port's engine, at `off` bytes from its first.
    pub(super) fn read(&self, off: u32) -> u32 {
        self.with(|s, st| s.port_read(st, off))
    }

    pub(super) fn write(&self, off: u32, val: u32) {
        self.with(|s, st| s.port_write(st, off, val));
    }

    /// A guest's byte read of the 16550 through the Super I/O window, then
    /// whatever the engine has to do about it.
    ///
    /// This takes the engine's lock and then the Super I/O's. The part's two
    /// channels are independent and the second lock is not the hardware's; it
    /// is there because the Super I/O also holds the parallel port's ECP
    /// registers under the same lock, and it is uncontended in practice.
    pub(super) fn pio_read(&self, r: u32) -> u8 {
        self.with(|s, st| {
            let flow = st.reg[0] & sdma_ctl::FLOW_CONTROL != 0;
            let v = s.sio.uart(s.p, |u| {
                let mut v = u.read(r);
                if r == uart_reg::MSR && u.flow_cts(flow) {
                    v |= MSR_CTS;
                }
                v
            });
            // On every level, not on assertion alone: the ring console
            // receives with the 16550's own interrupts disabled.
            s.run(st);
            v
        })
    }

    pub(super) fn pio_write(&self, r: u32, v: u8) {
        self.with(|s, st| {
            s.sio.uart(s.p, |u| u.write(r, v));
            s.run(st);
        });
    }

    /// Host input the Super I/O's UART has taken, and the engine's run on
    /// it. Returns whether anything changed.
    pub(super) fn pump(&self) -> bool {
        self.with(|s, st| {
            let moved = s.sio.uart(s.p, |u| u.pump());
            if moved {
                s.run(st);
            }
            moved
        })
    }

    /// A pass of what the receive thread and the timers do between guest
    /// accesses.
    #[cfg(test)]
    pub(super) fn service(&self, now: Instant) {
        self.with(|s, st| {
            if s.sio.uart(s.p, |u| u.pump()) {
                s.run(st);
            }
            s.catch_up_locked(st, now);
            s.run(st);
        });
    }

    pub(super) fn reset(&self) {
        self.with(|_, st| *st = SdmaPort::default());
    }

    pub(super) fn save_state(&self, now: Instant) -> toml::Value {
        self.st.lock().save_state(now)
    }

    pub(super) fn load_state(&self, v: &toml::Value, now: Instant) -> Result<(), String> {
        self.with(|_, st| {
            *st = SdmaPort::load_state(v, now)?;
            Ok(())
        })
    }

    fn running(&self, st: &SdmaPort) -> bool {
        let c = st.reg[0];
        c & sdma_ctl::ENABLE != 0 && c & sdma_ctl::PAUSE == 0
    }

    /// Transmit-ring-empty is a level: ring DMA on, nothing left in the ring
    /// or held by the engine.
    fn update_tx_empty(&self, st: &mut SdmaPort) {
        let mask = self.mask();
        let port = &*st;
        let empty = st.reg[0] & sdma_ctl::ENABLE != 0
            && (port.reg[1] ^ port.reg[2]) & mask == 0
            && !port.tx_fetched
            && port.tx_delay_deadline.is_none();
        self.level(int::TX_RING_EMPTY, empty);
    }

    /// The receive high-water mark counts ring entries; a level. So is
    /// receive-ring-full: set while ring DMA is on and an entry waits that
    /// the ring had no room for, so an acknowledgement does not bring it back
    /// with every refused retry, and it falls once the consumer makes room,
    /// posted yet or not. A ring filled exactly, with nothing waiting, does
    /// not raise it. A model choice; what the part does is not known.
    fn update_high_water(&self, st: &mut SdmaPort) {
        let mask = self.mask();
        let port = &*st;
        let hw = st.reg[0] & sdma_ctl::HIGH_WATER;
        let waiting = port.reg[3].wrapping_sub(port.reg[4]) & mask;
        let level = hw != 0 && waiting / ENTRY as u32 >= hw;
        if (port.reg[3] + ENTRY as u32) & mask != port.reg[4] & mask {
            st.post_refused = false;
        }
        let full = st.reg[0] & sdma_ctl::ENABLE != 0 && st.post_refused;
        self.level(int::RX_HIGH_WATER, level);
        self.level(int::RX_RING_FULL, full);
    }

    /// Post the packer as one entry. Fails, keeping the characters, when the
    /// ring is full or the write is refused.
    fn post(&self, st: &mut SdmaPort) -> bool {
        let mask = self.mask();
        let ring = self.ring(st, 1);
        let (produce, consume, packer) = {
            let port = &*st;
            (port.reg[3], port.reg[4], port.packer)
        };
        let next = (produce + ENTRY as u32) & mask;
        if next == consume & mask {
            st.post_refused = true;
            return false;
        }
        if !self.dma_write(ring.saturating_add((produce & mask) as u64), &packer) {
            self.raise(int::DMA_ERROR);
            return false;
        }
        let port = &mut *st;
        port.reg[3] = next;
        port.post_refused = false;
        port.packed = 0;
        port.packer = [0; ENTRY];
        true
    }

    fn rx_timer_start(&self, st: &mut SdmaPort, since: Instant) {
        let port = &mut *st;
        let reload = port.reg[5] & TIMER_RELOAD;
        port.rx_idle_count = 0;
        let dur = Duration::from_nanos(reload as u64 * TIMER_TICK_NS);
        if port.paused {
            // Started while paused, it waits frozen at its full count.
            port.rx_frozen = Some(dur);
        } else {
            port.rx_deadline = Some(since + dur);
        }
    }

    fn rx_timer_expired(&self, st: &mut SdmaPort) {
        st.rx_idle_count = 0;
        st.rx_deadline = None;
        if st.packed != 0 {
            self.post(st);
        }
        // Raised at every expiry, even over an empty ring; the arm is spent.
        if st.armed {
            self.raise(int::RX_TIMER);
        }
        st.armed = false;
        self.update_high_water(st);
    }

    /// Expiries already due happen before an access.
    fn catch_up_locked(&self, st: &mut SdmaPort, now: Instant) {
        let delay = st.tx_delay_deadline;
        if let Some(d) = delay {
            if !st.paused && d <= now {
                self.tx_delay_ended(st, d);
            }
        }
        if let Some(d) = st.rx_deadline {
            if d <= now {
                self.rx_timer_expired(st);
            }
        }
    }

    fn tx_delay_ended(&self, st: &mut SdmaPort, when: Instant) {
        let irq = {
            let port = &mut *st;
            port.tx_delay_deadline = None;
            std::mem::take(&mut port.tx_delay_irq)
        };
        if irq {
            self.raise(int::TX_EXPLICIT);
        }
        self.run_from(st, when);
    }

    fn receive(&self, st: &mut SdmaPort, since: Instant) {
        while self.running(st) {
            if st.packed == ENTRY_CHARS && !self.post(st) {
                break;
            }
            // Line status before the character: reading it clears the latches.
            let flow = st.reg[0] & sdma_ctl::FLOW_CONTROL != 0;
            let Some((lsr, data, msr)) = self.sio.uart(self.p, |u| {
                let lsr = u.read_core(uart_reg::LSR);
                if lsr & LSR_DR == 0 {
                    return None;
                }
                let data = u.read_core(uart_reg::RBR_THR);
                let mut msr = u.read_core(uart_reg::MSR);
                if u.flow_cts(flow) {
                    msr |= MSR_CTS;
                }
                Some((lsr, data, msr))
            }) else {
                break;
            };
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
                let port = &mut *st;
                port.packer[port.packed] = data;
                port.packer[ENTRY_CHARS + port.packed] = status;
                port.packed += 1;
                port.armed && port.rx_deadline.is_none() && port.rx_frozen.is_none()
            };
            // Started by the first character after the arm; later ones leave
            // a running timer alone.
            if start_timer {
                self.rx_timer_start(st, since);
            }
            // A fourth character completes the entry; fewer wait for the
            // timer or a drain.
            if st.packed == ENTRY_CHARS {
                self.post(st);
            }
        }
    }

    /// Each control byte types its data byte; the empty type ends the entry,
    /// wherever it sits.
    fn transmit(&self, st: &mut SdmaPort, since: Instant) {
        let mask = self.mask();
        let ring = self.ring(st, 0);
        if st.tx_delay_deadline.is_some() {
            return;
        }
        loop {
            let (produce, consume) = {
                let port = &*st;
                (port.reg[1], port.reg[2])
            };
            if !(self.running(st) && consume & mask != produce & mask) {
                break;
            }
            if !st.tx_fetched {
                let mut entry = [0u8; ENTRY];
                if !self.dma_read(ring.saturating_add((consume & mask) as u64), &mut entry) {
                    self.raise(int::DMA_ERROR);
                    break;
                }
                let port = &mut *st;
                port.tx_entry = entry;
                port.tx_fetched = true;
            }
            let entry = st.tx_entry;
            let mut slot = st.tx_slot;
            while slot < ENTRY_CHARS {
                let control = entry[ENTRY_CHARS + slot];
                let ty = control & TX_TYPE;
                if ty == 0 {
                    break;
                }
                match ty {
                    TX_CHARACTER => {
                        let looped = self.sio.uart(self.p, |u| u.loopback());
                        // What has already arrived goes ahead of this one.
                        if looped {
                            self.receive(st, since);
                        }
                        self.sio.uart(self.p, |u| u.write(uart_reg::RBR_THR, entry[slot]));
                        // Approximation: a looped-back character arrives at
                        // once, where the part takes a character time.
                        self.receive(st, since);
                    }
                    TX_MODEM => self.sio.uart(self.p, |u| u.write(uart_reg::MCR, entry[slot])),
                    TX_DELAY => {
                        // The data byte is milliseconds of quiet.
                        if entry[slot] != 0 {
                            let port = &mut *st;
                            port.tx_slot = slot + 1;
                            port.tx_delay_deadline = Some(since + Duration::from_millis(entry[slot] as u64));
                            port.tx_delay_irq = control & TX_INTERRUPT != 0;
                            return;
                        }
                    }
                    _ => unreachable!(),
                }
                slot += 1;
                st.tx_slot = slot;
                if control & TX_INTERRUPT != 0 {
                    self.raise(int::TX_EXPLICIT);
                }
            }
            let port = &mut *st;
            port.tx_slot = 0;
            port.tx_fetched = false;
            port.reg[2] = (port.reg[2] + ENTRY as u32) & mask;
        }
    }

    /// The drain posts what the packer holds and reads back set until it has;
    /// it waits while ring DMA is off or paused and while the ring is full.
    fn drain(&self, st: &mut SdmaPort) {
        if !st.drain_pending || !self.running(st) {
            return;
        }
        if st.packed != 0 && !self.post(st) {
            return;
        }
        st.drain_pending = false;
    }

    fn run_from(&self, st: &mut SdmaPort, since: Instant) {
        if !self.reachable(st) || st.busy {
            return;
        }
        st.busy = true;
        self.transmit(st, since);
        self.receive(st, since);
        self.drain(st);
        self.update_high_water(st);
        self.update_tx_empty(st);
        st.busy = false;
    }

    fn run(&self, st: &mut SdmaPort) {
        self.run_from(st, Instant::now());
    }

    /// A passive mirror of the 16550's lines; no read side effects.
    fn shadow(&self, st: &SdmaPort) -> u32 {
        let flow = st.reg[0] & sdma_ctl::FLOW_CONTROL != 0;
        let (lsr, mcr, msr) = self.sio.uart(self.p, |u| {
            let msr = u.core.msr();
            (u.core.lsr(), u.core.mcr(), if u.flow_cts(flow) { msr | MSR_CTS } else { msr })
        });
        let mut s = lsr as u32;
        for (m, bit) in [(MSR_DCTS, 16), (0x08, 19), (MSR_CTS, 20), (MSR_DCD, 23)] {
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

    fn port_read(&self, st: &mut SdmaPort, off: u32) -> u32 {
        let now = Instant::now();
        self.catch_up_locked(st, now);
        if off == sdma_off::SHADOW {
            return self.shadow(st);
        }
        let port = &*st;
        let mut v = port.reg[(off / 4) as usize];
        match off {
            sdma_off::CONTROL => {
                if v & sdma_ctl::PAUSE != 0 {
                    v |= sdma_ctl::PAUSED;
                }
                if port.tx_delay_deadline.is_some() {
                    v |= sdma_ctl::TX_TIMER_RUN;
                }
                if port.drain_pending {
                    v |= sdma_ctl::RX_TO_MEMORY;
                }
            }
            sdma_off::RX_CONSUME => {
                if port.armed {
                    v |= ARM_RX_TIMER;
                }
            }
            sdma_off::RX_PRODUCE => v |= (port.packed as u32) << HELD_SHIFT,
            sdma_off::RX_TIMER => {
                // Ticks left while running or frozen; idle, whatever a reload
                // write last loaded.
                let mut ticks = port.rx_idle_count;
                if let (true, Some(left)) = (port.paused, port.rx_frozen) {
                    ticks = (left.as_nanos() as u64 / TIMER_TICK_NS).min(TIMER_RELOAD as u64) as u32;
                } else if let Some(d) = port.rx_deadline {
                    let left = d.saturating_duration_since(now);
                    ticks = (left.as_nanos() as u64 / TIMER_TICK_NS).min(TIMER_RELOAD as u64) as u32;
                }
                v |= ticks << TIMER_COUNT_SHIFT;
            }
            _ => {}
        }
        v
    }

    fn port_write(&self, st: &mut SdmaPort, off: u32, val: u32) {
        let now = Instant::now();
        self.catch_up_locked(st, now);
        let mask = self.mask();
        match off {
            sdma_off::CONTROL => {
                st.reg[0] = val & !sdma_ctl::UNSTORED;
                if val & sdma_ctl::RESET != 0 {
                    // Resets the four indices and nothing else.
                    let port = &mut *st;
                    port.reg[1..=4].fill(0);
                    port.armed = false;
                    port.drain_pending = false;
                    port.packed = 0;
                    port.post_refused = false;
                    port.packer = [0; ENTRY];
                    port.tx_slot = 0;
                    port.tx_fetched = false;
                    port.tx_delay_deadline = None;
                    port.tx_delay_irq = false;
                    port.rx_deadline = None;
                    port.rx_frozen = None;
                    port.rx_idle_count = 0;
                }
                if val & sdma_ctl::RX_TO_MEMORY != 0 {
                    st.drain_pending = true;
                }
                let port = &mut *st;
                // A pause freezes the receive timer and a transmit delay;
                // resuming lets each finish the time it had left. A model
                // choice; no source says what the part does.
                if val & sdma_ctl::PAUSE != 0 && !port.paused {
                    port.paused = true;
                    port.rx_frozen = port.rx_deadline.take().map(|d| d.saturating_duration_since(now));
                    port.tx_delay_frozen = port.tx_delay_deadline.take().map(|d| d.saturating_duration_since(now));
                } else if val & sdma_ctl::PAUSE == 0 && port.paused {
                    port.paused = false;
                    port.rx_deadline = port.rx_frozen.take().map(|l| now + l);
                    port.tx_delay_deadline = port.tx_delay_frozen.take().map(|l| now + l);
                }
                if val & sdma_ctl::ENABLE == 0 {
                    port.rx_deadline = None;
                    port.rx_frozen = None;
                }
                self.run(st);
            }
            sdma_off::TX_PRODUCE | sdma_off::TX_CONSUME | sdma_off::RX_PRODUCE => {
                st.reg[(off / 4) as usize] = val & mask;
                self.run(st);
            }
            sdma_off::RX_CONSUME => {
                // Bit 31 sets the arm; a write with it clear leaves it.
                st.reg[4] = val & mask;
                if val & ARM_RX_TIMER != 0 {
                    st.armed = true;
                }
                self.run(st);
            }
            sdma_off::RX_TIMER => {
                let port = &mut *st;
                port.reg[5] = val & TIMER_RELOAD;
                port.rx_idle_count = val & TIMER_RELOAD;
            }
            // The shadow discards writes silently.
            _ => {}
        }
        self.update_tx_empty(st);
        self.update_high_water(st);
    }
}

impl SdmaPort {
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
        })
    }
}

