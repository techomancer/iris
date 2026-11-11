//! The IOC3's two serial ports: the shared ring-DMA engine in front of the
//! Super I/O's 16550s.
//!
//! The engine itself is `dev/ioc_serial_dma.rs`, which the IOC4 uses too.
//! This file is the IOC3 around it: where its registers and interrupt bits
//! sit, how chip control aims each port at its UART, and how the engine
//! reaches the UARTs. Software either drives a UART directly (pass-through,
//! the mode Linux's 8250 driver and OpenBSD's `com` use) or hands the engine
//! rings in memory.
//!
//! Nothing is timed to a baud rate: serial transmit completes inside the
//! write and a loopback character arrives at once (the part takes a
//! character time). Ring DMA, the receive timer and transmit delays are
//! evaluated lazily on every register access, and an `hptimer` one-shot per
//! deadline runs the same catch-up while the guest is idle.
//!
//! Both ports share one engine and so one lock, taken before the Super
//! I/O's, which the engine holds for the length of a pass: it reaches the
//! UARTs as the part's own bus master would, through the window the Super
//! I/O owns.
//!
//! Ported from hw/misc/sgi-ioc3.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Mutex, MutexGuard};

use super::superio::{Sio, SuperIo};
use crate::dev::ns16550::com_reg::{self, MSR_CTS};
use super::{decode, encode, int, reg, Core};
use crate::dev::ioc_serial_dma::{ctl, SerialDma, SerialDmaHost, SerialDmaParams, SerialDmaSource, SerialDmaUart};
use crate::snapshot::StateTable;

// What the chip's register map and its tests name.
#[cfg(test)]
pub(super) use crate::dev::ioc_serial_dma::{ctl as sdma_ctl, ARM_RX_TIMER, ENTRY_CHARS, HELD_SHIFT, TIMER_TICK_NS, TX_DELAY, TX_INTERRUPT};
pub(super) use crate::dev::ioc_serial_dma::off as sdma_off;

/// Registers a page-aliased write cannot reach: the two indexes the engine
/// advances and the 16550 shadow.
pub(super) fn alias_read_only(off: u32) -> bool {
    matches!(off, sdma_off::TX_CONSUME | sdma_off::RX_PRODUCE | sdma_off::SHADOW)
}

/// Chip control: where each engine finds its 16550 (byte offset / 8).
pub(super) const CHIP_CTRL_PORT_A_SHIFT: u32 = 1;
pub(super) const CHIP_CTRL_PORT_B_SHIFT: u32 = 8;
pub(super) const CHIP_CTRL_PORT_FIELD: u32 = 0x7f;

/// The receive producer index reports the packer's held count; a failed
/// transfer raises the error source and the port goes on; and a looped-back
/// character arrives inside the write.
const PARAMS: SerialDmaParams =
    SerialDmaParams { held_counts: true, halt_on_dma_error: false, loopback_at_once: true };

/// The chip's two ports and the one engine that runs them.
struct Engine {
    core: Arc<Core>,
    sio: Arc<SuperIo>,
    st: Mutex<SerialDma>,
    /// The deadlines in `st`, kept where a timer can read them without the
    /// lock: per port the receive timer then the transmit delay, as
    /// nanoseconds after the process epoch plus one, zero for none.
    due: [AtomicU64; 4],
}

/// The chip as the engine sees it for the length of one pass. The Super I/O
/// is locked when the pass first reaches a UART, and held from then to the
/// pass's end: a ring register access that touches no port leaves it alone.
struct Host<'a> {
    e: &'a Engine,
    sio: Option<MutexGuard<'a, Sio>>,
}

impl Host<'_> {
    fn sio(&mut self) -> &mut Sio {
        let e = self.e;
        self.sio.get_or_insert_with(|| e.sio.lock())
    }
}

impl SerialDmaHost for Host<'_> {
    fn uart(&mut self, port: usize) -> &mut dyn SerialDmaUart {
        &mut self.sio().uart[port]
    }

    fn raise(&mut self, port: usize, src: SerialDmaSource) {
        self.e.core.set_bits(int::serial(port, src as u32));
    }

    fn set_level(&mut self, port: usize, src: SerialDmaSource, level: bool) {
        self.e.core.level(int::serial(port, src as u32), level);
    }

    /// Chip control places each port's 16550 for its engine; aimed anywhere
    /// else, the engine does not reach the port.
    fn reachable(&self, port: usize) -> bool {
        let (shift, base) = if port == 1 {
            (CHIP_CTRL_PORT_B_SHIFT, reg::SERIAL_B)
        } else {
            (CHIP_CTRL_PORT_A_SHIFT, reg::SERIAL_A)
        };
        (self.e.core.chip_control.load(Ordering::SeqCst) >> shift) & CHIP_CTRL_PORT_FIELD == (base & 0xfff) / 8
    }

    /// The part's interrupt output goes on its pass-through bit in the
    /// status word: the bit follows the 16550's level, set or cleared under
    /// the part's lock so two threads cannot leave it stale.
    fn uart_line_changed(&mut self, port: usize) {
        let level = self.sio().uart[port].irq();
        self.e.core.level(int::serial(port, int::PASSTHROUGH), level);
    }

    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.e.core.dma().is_some_and(|d| d.dma_read(addr, buf).is_ok())
    }

    fn dma_write(&self, addr: u64, data: &[u8]) -> bool {
        self.e.core.dma().is_some_and(|d| d.dma_write(addr, data).is_ok())
    }
}

impl Engine {
    /// Run `f` with the engine locked, and the Super I/O locked once the
    /// pass reaches a UART, and publish the deadlines after. Both stay held
    /// across the pass's DMA: no host's DMA path calls back into a function
    /// (the Octane's Bridge and the O2's host memory reach only memory, the
    /// PIC bridge only its peer, which
    /// `the_bridge_does_not_route_dma_to_a_function` shows), so nothing it
    /// does comes back to take either lock.
    fn with<R>(&self, f: impl FnOnce(&mut SerialDma, &mut Host<'_>) -> R) -> R {
        let mut st = self.st.lock();
        let mut host = Host { e: self, sio: None };
        let r = f(&mut st, &mut host);
        self.publish_due(&st);
        r
    }

    fn publish_due(&self, st: &SerialDma) {
        for p in 0..2 {
            for (i, d) in [st.rx_deadline(p), st.tx_delay_deadline(p)].into_iter().enumerate() {
                let (slot, d) = (&self.due[p * 2 + i], encode(d));
                // Most passes move no deadline; skip the store then.
                if slot.load(Ordering::SeqCst) != d {
                    slot.store(d, Ordering::SeqCst);
                }
            }
        }
    }
}

/// One of the chip's two serial ports, as the registers see it.
pub(super) struct Serial {
    p: usize,
    e: Arc<Engine>,
}

impl Serial {
    /// Both ports of a chip, over one engine.
    pub(super) fn pair(core: Arc<Core>, sio: Arc<SuperIo>) -> [Serial; 2] {
        let e = Arc::new(Engine {
            core,
            sio,
            st: Mutex::new(SerialDma::new(2, PARAMS)),
            due: std::array::from_fn(|_| AtomicU64::new(0)),
        });
        [0, 1].map(|p| Serial { p, e: e.clone() })
    }

    /// The receive-timer and transmit-delay deadlines as of the last access,
    /// without taking the engine's lock.
    pub(super) fn due(&self) -> (Option<Instant>, Option<Instant>) {
        let at = self.p * 2;
        (decode(self.e.due[at].load(Ordering::SeqCst)), decode(self.e.due[at + 1].load(Ordering::SeqCst)))
    }

    #[cfg(test)]
    pub(super) fn rx_deadline(&self) -> Option<Instant> {
        self.e.st.lock().rx_deadline(self.p)
    }

    #[cfg(test)]
    pub(super) fn set_rx_deadline(&self, due: Option<Instant>) {
        self.e.with(|st, _| st.port[self.p].rx_deadline = due);
    }

    pub(super) fn is_locked(&self) -> bool {
        self.e.st.is_locked()
    }

    /// Run the engine on whatever work is waiting.
    pub(super) fn kick(&self) {
        self.e.with(|st, h| st.run(self.p, h));
    }

    /// Expiries already due happen before an access.
    pub(super) fn catch_up(&self, now: Instant) {
        self.e.with(|st, h| st.catch_up(now, h));
    }

    /// A register of this port's engine, at `off` bytes from its first.
    pub(super) fn read(&self, off: u32) -> u32 {
        self.e.with(|st, h| st.port_read(self.p, off, h))
    }

    pub(super) fn write(&self, off: u32, val: u32) {
        self.e.with(|st, h| st.port_write(self.p, off, val, h));
    }

    /// The ring base register the two ports share, high word or low.
    pub(super) fn ring_base(&self, high: bool) -> u32 {
        self.e.st.lock().base_read(0, high)
    }

    pub(super) fn set_ring_base(&self, high: bool, val: u32) {
        self.e.st.lock().base_write(0, high, val);
    }

    /// A guest's byte read of the 16550 through the Super I/O window, then
    /// whatever the engine has to do about it.
    pub(super) fn pio_read(&self, r: u32) -> u8 {
        let p = self.p;
        self.e.with(|st, h| {
            let flow = st.control(p) & ctl::FLOW_CONTROL != 0;
            let u = &mut h.sio().uart[p];
            let mut v = u.read(r);
            if r == com_reg::MSR && u.flow_cts(flow) {
                v |= MSR_CTS;
            }
            // On every level, not on assertion alone: the ring console
            // receives with the 16550's own interrupts disabled.
            st.run(p, h);
            h.uart_line_changed(p);
            v
        })
    }

    pub(super) fn pio_write(&self, r: u32, v: u8) {
        let p = self.p;
        self.e.with(|st, h| {
            h.sio().uart[p].write(r, v);
            st.run(p, h);
            h.uart_line_changed(p);
        });
    }

    /// Host input the Super I/O's UART has taken, and the engine's run on
    /// it. Returns whether anything changed.
    pub(super) fn pump(&self) -> bool {
        let p = self.p;
        self.e.with(|st, h| {
            let moved = h.sio().uart[p].pump();
            h.uart_line_changed(p);
            if moved {
                st.run(p, h);
            }
            moved
        })
    }

    /// A pass of what the receive thread and the timers do between guest
    /// accesses.
    #[cfg(test)]
    pub(super) fn service(&self, now: Instant) {
        let p = self.p;
        self.e.with(|st, h| {
            if h.sio().uart[p].pump() {
                st.run(p, h);
            }
            h.uart_line_changed(p);
            st.catch_up(now, h);
            st.run(p, h);
        });
    }

    /// This port back to power-on. The ring bases belong to the pair and
    /// are reset with [`Serial::reset_rings`].
    pub(super) fn reset(&self) {
        self.e.with(|st, _| st.clear_port(self.p));
    }

    /// The ring base registers back to zero.
    pub(super) fn reset_rings(&self) {
        self.set_ring_base(true, 0);
        self.set_ring_base(false, 0);
    }

    pub(super) fn save_state(&self, now: Instant) -> toml::Value {
        self.e.st.lock().save_port(self.p, now)
    }

    pub(super) fn load_state(&self, v: &toml::Value, now: Instant) -> Result<(), String> {
        self.e.with(|st, _| st.load_port(self.p, v, now))
    }
}

/// The ring base registers and each port's table for a snapshot, under the
/// keys the chip has always written.
pub(super) fn save_rings(ports: &[Option<Serial>; 2], now: Instant) -> Option<toml::Value> {
    let first = ports.iter().flatten().next()?;
    let mut sdma = StateTable::new();
    sdma.u32("base_hi", first.ring_base(true)).u32("base_lo", first.ring_base(false));
    for s in ports.iter().flatten() {
        sdma.value(["port0", "port1"][s.p], s.save_state(now));
    }
    Some(sdma.build())
}
