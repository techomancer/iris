// Ported from hw/char/serial.c in jamesbraid/qemu-sgi-origami, which carries
// this notice:
//
// QEMU 16550A UART emulation
//
// Copyright (c) 2003-2004 Fabrice Bellard
// Copyright (c) 2008 Citrix Systems, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//! The 16550A register interface, as copied by the O2's TI TL16PIR552 (behind
//! MACE) and the Octane's IOC3.
//!
//! This is a port of QEMU's `hw/char/serial.c`: the register file, the
//! 16-byte receive FIFO and its trigger levels, interrupt identification in
//! priority order, line and modem status, loopback, receive overrun, the
//! FIFO character timeout and the divisor latch. Both chips wrap it; this
//! file knows nothing of where its registers sit or what drives its
//! interrupt. [`crate::dev::mace_uart`] and [`crate::dev::ioc3`] are the shells.
//!
//! # Differences from the QEMU source
//!
//! * Transmit is instant. A write of the transmit register hands the byte to
//!   the host backend (or, in loopback, back to the receiver) before the
//!   write returns, so the holding register and the transmitter always read
//!   empty and QEMU's transmit timer, retry and hold-off machinery is not
//!   here. Nothing is paced to the baud rate; the divisor and line control
//!   only set how long the receive timeout is.
//! * The host end is a [`SerialBackend`] polled by [`Ns16550::pump`], where
//!   QEMU's chardev calls back. The receive timeout is a host-time deadline
//!   that [`Ns16550::check_timeout`] resolves, where QEMU arms a timer.
//! * A receive overrun with the FIFOs off keeps the character already in the
//!   receive register and drops the new one, which is what the part does.
//!   QEMU overwrites the register.
//! * Reading the receive register with nothing received returns zero, where
//!   QEMU returns the last character again.
//! * Modem status changes in loopback latch the delta bits, as the part's
//!   datasheet has them and MAME's `ins8250` does; QEMU's loopback read returns
//!   the levels alone. The
//!   levels outside loopback are fixed at carrier, DSR and CTS asserted;
//!   there is no real modem to poll.
//! * Not ported: break detection and set-break, the transmit FIFO (there is
//!   nothing to queue), the live modem-line polling of a host serial port,
//!   and the host baud-rate control.
//! * The TI part's extra register at the scratch position with the divisor
//!   latch set is not modelled: there it is the scratch register.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::dev::serial_host::{null_serial_backend, SerialBackend};
use crate::snapshot::{get_field, hex_u8, toml_bool, toml_u32, toml_u8};

/// The receive FIFO's depth.
pub const FIFO_LENGTH: usize = 16;

/// How much of a UART's output it keeps for the monitor.
pub const OUT_KEEP: usize = 64 * 1024;

/// Register numbers and bits.
pub mod com_reg {
    /// Transmit holding / receive buffer (and divisor low when DLAB is set).
    pub const THR: u32 = 0;
    pub const IER: u32 = 1;
    pub const IIR_FCR: u32 = 2;
    pub const LCR: u32 = 3;
    pub const MCR: u32 = 4;
    pub const LSR: u32 = 5;
    pub const MSR: u32 = 6;
    pub const SCR: u32 = 7;

    pub const LCR_DLAB: u8 = 0x80;
    pub const LCR_PEN: u8 = 0x08;
    pub const LCR_NSTB: u8 = 0x04;
    pub const LCR_WLS: u8 = 0x03;

    /// IER: interrupt on received data, transmitter drained, line status
    /// and modem status.
    pub const IER_RX: u8 = 0x01;
    pub const IER_TX: u8 = 0x02;
    pub const IER_RLS: u8 = 0x04;
    pub const IER_MS: u8 = 0x08;

    /// IIR: bit 0 clear means "something to service"; the cause is above it.
    pub const IIR_NONE: u8 = 0x01;
    pub const IIR_ID: u8 = 0x06;
    pub const IIR_MSI: u8 = 0x00;
    pub const IIR_TX_EMPTY: u8 = 0x02;
    pub const IIR_RX_DATA: u8 = 0x04;
    pub const IIR_RLSI: u8 = 0x06;
    /// Character timeout (bit 3 set beside the receive-data cause).
    pub const IIR_CTI: u8 = 0x0c;
    /// The two high bits a 16550 sets once its FIFOs are enabled.
    pub const IIR_FIFO: u8 = 0xc0;

    pub const FCR_FE: u8 = 0x01;
    pub const FCR_RFR: u8 = 0x02;
    pub const FCR_XFR: u8 = 0x04;

    /// LSR: a received character is waiting.
    pub const LSR_DR: u8 = 0x01;
    pub const LSR_OE: u8 = 0x02;
    pub const LSR_BI: u8 = 0x10;
    /// LSR: transmit holding register empty.
    pub const LSR_THRE: u8 = 0x20;
    /// LSR: transmitter completely empty.
    pub const LSR_TEMT: u8 = 0x40;
    /// The status bits that raise a line-status interrupt.
    pub const LSR_INT_ANY: u8 = 0x1e;

    pub const MCR_DTR: u8 = 0x01;
    pub const MCR_RTS: u8 = 0x02;
    pub const MCR_OUT1: u8 = 0x04;
    pub const MCR_OUT2: u8 = 0x08;
    /// MCR bit 4: loopback. The transmitter is wired back to the receiver and
    /// the modem control outputs to the modem status inputs. The O2 PROM's
    /// serial self-test runs entirely in this mode.
    pub const MCR_LOOP: u8 = 0x10;

    pub const MSR_DCTS: u8 = 0x01;
    pub const MSR_DDSR: u8 = 0x02;
    pub const MSR_TERI: u8 = 0x04;
    pub const MSR_DDCD: u8 = 0x08;
    pub const MSR_ANY_DELTA: u8 = 0x0f;
    pub const MSR_CTS: u8 = 0x10;
    pub const MSR_DSR: u8 = 0x20;
    pub const MSR_RI: u8 = 0x40;
    pub const MSR_DCD: u8 = 0x80;
}
use com_reg::*;

/// A character time is ten bit times at the reset divisor of 9600 baud.
const RESET_CHAR_TIME: Duration = Duration::from_nanos(1_000_000_000 / 9600 * 10);

/// The modem inputs of a port with nothing plugged in: carrier, data set
/// ready and clear to send asserted.
const MSR_IDLE_LEVELS: u8 = MSR_DCD | MSR_DSR | MSR_CTS;

pub struct Ns16550 {
    backend: Arc<dyn SerialBackend>,
    baudbase: u32,
    divider: u16,
    rbr: u8,
    ier: u8,
    iir: u8,
    lcr: u8,
    mcr: u8,
    lsr: u8,
    msr: u8,
    scr: u8,
    fcr: u8,
    recv_fifo: VecDeque<u8>,
    recv_fifo_itl: usize,
    thr_ipending: bool,
    timeout_ipending: bool,
    /// When the receive timeout fires, while characters sit unread.
    timeout_deadline: Option<Instant>,
    char_transmit_time: Duration,
    /// Characters a script or the monitor has typed, ahead of what the
    /// backend delivers. They reach the receiver the way the backend's do,
    /// as it will take them.
    typed: VecDeque<u8>,
    /// The last [`OUT_KEEP`] bytes sent, and how many were sent in all.
    out: VecDeque<u8>,
    sent: u64,
}

impl Ns16550 {
    /// A port in its reset state. `baudbase` is the reference clock over 16
    /// (115200 for a 1.8432 MHz crystal): the divisor divides it.
    pub fn new(baudbase: u32) -> Self {
        let mut s = Self {
            backend: null_serial_backend(),
            baudbase,
            divider: 0,
            rbr: 0,
            ier: 0,
            iir: IIR_NONE,
            lcr: 0,
            mcr: 0,
            lsr: 0,
            msr: 0,
            scr: 0,
            fcr: 0,
            recv_fifo: VecDeque::with_capacity(FIFO_LENGTH),
            recv_fifo_itl: 1,
            thr_ipending: false,
            timeout_ipending: false,
            timeout_deadline: None,
            char_transmit_time: RESET_CHAR_TIME,
            typed: VecDeque::new(),
            out: VecDeque::new(),
            sent: 0,
        };
        s.reset();
        s
    }

    /// Reset to 9600 baud, 8N1, the FIFOs off and empty, the
    /// transmitter empty, carrier up. The host end, the clock and the record
    /// of what was sent stay.
    pub fn reset(&mut self) {
        self.rbr = 0;
        self.ier = 0;
        self.iir = IIR_NONE;
        self.lcr = 0;
        self.lsr = LSR_TEMT | LSR_THRE;
        self.msr = MSR_IDLE_LEVELS;
        self.divider = 0x0c;
        self.mcr = MCR_OUT2;
        self.scr = 0;
        self.fcr = 0;
        self.recv_fifo_itl = 1;
        self.char_transmit_time = RESET_CHAR_TIME;
        self.timeout_ipending = false;
        self.timeout_deadline = None;
        self.recv_fifo.clear();
        self.thr_ipending = false;
    }

    pub fn set_backend(&mut self, backend: Arc<dyn SerialBackend>) {
        self.backend = backend;
    }

    /// Change the reference clock the divisor divides (the IOC3 derives it
    /// from its pre-divider).
    pub fn set_baudbase(&mut self, baudbase: u32) {
        self.baudbase = baudbase;
        self.update_parameters();
    }

    // ----- state, read without side effects -----

    pub fn baudbase(&self) -> u32 { self.baudbase }
    pub fn ier(&self) -> u8 { self.ier }
    pub fn lcr(&self) -> u8 { self.lcr }
    pub fn mcr(&self) -> u8 { self.mcr }
    pub fn lsr(&self) -> u8 { self.lsr }
    pub fn msr(&self) -> u8 { self.msr }
    pub fn scr(&self) -> u8 { self.scr }
    pub fn fifos_enabled(&self) -> bool { self.fcr & FCR_FE != 0 }

    pub fn dlab(&self) -> bool {
        self.lcr & LCR_DLAB != 0
    }

    pub fn loopback(&self) -> bool {
        self.mcr & MCR_LOOP != 0
    }

    /// Received characters not yet read, and typed ones not yet received.
    pub fn rx_pending(&self) -> usize {
        self.typed.len()
            + if self.fcr & FCR_FE != 0 { self.recv_fifo.len() } else { (self.lsr & LSR_DR != 0) as usize }
    }

    /// Type characters at the port, as if the host had sent them. They wait
    /// until the port will take them, as the backend's do.
    pub fn type_in(&mut self, bytes: &[u8]) {
        self.typed.extend(bytes.iter().copied());
        self.pump();
    }

    /// True while the port is asking for attention.
    pub fn irq(&self) -> bool {
        self.iir & IIR_NONE == 0
    }

    /// The last `n` bytes sent (at most [`OUT_KEEP`]).
    pub fn output_tail(&self, n: usize) -> String {
        let tail: Vec<u8> = self.out.iter().skip(self.out.len().saturating_sub(n)).copied().collect();
        String::from_utf8_lossy(&tail).to_string()
    }

    /// Bytes sent in all, including those no longer kept.
    pub fn bytes_out(&self) -> u64 {
        self.sent
    }

    // ----- interrupt identification -----

    /// Pick the highest-priority cause.
    fn update_irq(&mut self) {
        let tmp = if self.ier & IER_RLS != 0 && self.lsr & LSR_INT_ANY != 0 {
            IIR_RLSI
        } else if self.ier & IER_RX != 0 && self.timeout_ipending {
            // The enable bit masks this too: not in the specification but
            // seen on existing parts.
            IIR_CTI
        } else if self.ier & IER_RX != 0
            && self.lsr & LSR_DR != 0
            && (self.fcr & FCR_FE == 0 || self.recv_fifo.len() >= self.recv_fifo_itl)
        {
            IIR_RX_DATA
        } else if self.ier & IER_TX != 0 && self.thr_ipending {
            IIR_TX_EMPTY
        } else if self.ier & IER_MS != 0 && self.msr & MSR_ANY_DELTA != 0 {
            IIR_MSI
        } else {
            IIR_NONE
        };
        self.iir = tmp | (self.iir & 0xf0);
    }

    /// Only the character time survives; there
    /// is no host line to set.
    fn update_parameters(&mut self) {
        let mut frame = 1u32;
        if self.lcr & LCR_PEN != 0 {
            frame += 1;
        }
        let stop = if self.lcr & LCR_NSTB != 0 { 2 } else { 1 };
        let data = (self.lcr & LCR_WLS) as u32 + 5;
        frame += data + stop;
        // A zero divisor gives about 3500 baud.
        let speed = if self.divider == 0 { 3500.0 } else { self.baudbase as f64 / self.divider as f64 };
        let speed = if speed > 0.0 { speed } else { 3500.0 };
        self.char_transmit_time = Duration::from_nanos((1e9 / speed * frame as f64) as u64);
    }

    // ----- the register file -----

    /// A byte read of register `reg` (0..8). Reading status acknowledges what
    /// a 16550 acknowledges on a read.
    pub fn read(&mut self, reg: u32) -> u8 {
        match reg & 7 {
            THR => {
                if self.dlab() {
                    self.divider as u8
                } else {
                    // The host has typed, perhaps, since the last look.
                    self.pump();
                    let v = self.read_rbr();
                    // The guest made room: the backend may deliver more.
                    self.pump();
                    v
                }
            }
            IER => {
                if self.dlab() {
                    (self.divider >> 8) as u8
                } else {
                    self.ier
                }
            }
            IIR_FCR => {
                self.check_timeout(Instant::now());
                let ret = self.iir;
                if ret & IIR_ID == IIR_TX_EMPTY {
                    self.thr_ipending = false;
                    self.update_irq();
                }
                ret
            }
            LCR => self.lcr,
            MCR => self.mcr,
            LSR => {
                self.pump();
                self.check_timeout(Instant::now());
                let ret = self.lsr;
                // Reading clears the break and overrun latches.
                if self.lsr & (LSR_BI | LSR_OE) != 0 {
                    self.lsr &= !(LSR_BI | LSR_OE);
                    self.update_irq();
                }
                ret
            }
            MSR => {
                let ret = self.msr;
                if self.msr & MSR_ANY_DELTA != 0 {
                    self.msr &= 0xf0;
                    self.update_irq();
                }
                ret
            }
            _ => self.scr,
        }
    }

    /// A byte write of register `reg` (0..8).
    pub fn write(&mut self, reg: u32, val: u8) {
        match reg & 7 {
            THR => {
                if self.dlab() {
                    self.divider = (self.divider & 0xff00) | val as u16;
                    self.update_parameters();
                } else {
                    self.write_thr(val);
                }
            }
            IER => {
                if self.dlab() {
                    self.divider = (self.divider & 0x00ff) | (val as u16) << 8;
                    self.update_parameters();
                } else {
                    let changed = (self.ier ^ val) & 0x0f;
                    self.ier = val & 0x0f;
                    // Turning the transmit interrupt on with the holding
                    // register empty raises it, even if reading the cause
                    // masked it before. Not in the datasheet, but drivers
                    // toggle the enable and rely on it.
                    if changed & IER_TX != 0 {
                        self.thr_ipending = self.ier & IER_TX != 0 && self.lsr & LSR_THRE != 0;
                    }
                    if changed != 0 {
                        self.update_irq();
                    }
                }
            }
            IIR_FCR => self.write_fcr(val),
            LCR => {
                self.lcr = val;
                self.update_parameters();
            }
            MCR => {
                self.mcr = val & 0x1f;
                self.update_modem_levels();
            }
            // Line and modem status are read-only; the IOC3 shell has test
            // writes of its own.
            LSR | MSR => {}
            _ => self.scr = val,
        }
    }

    fn write_fcr(&mut self, mut val: u8) {
        // Flush both sides when the enable changes.
        if (val ^ self.fcr) & FCR_FE != 0 {
            val |= FCR_XFR | FCR_RFR;
        }
        if val & FCR_RFR != 0 {
            self.lsr &= !(LSR_DR | LSR_BI);
            self.timeout_deadline = None;
            self.timeout_ipending = false;
            self.recv_fifo.clear();
        }
        if val & FCR_XFR != 0 {
            self.lsr |= LSR_THRE;
            self.thr_ipending = true;
        }
        // Only the bits that stick.
        let val = val & 0xc9;
        self.fcr = val;
        if val & FCR_FE != 0 {
            self.iir |= IIR_FIFO;
            self.recv_fifo_itl = match val & 0xc0 {
                0x00 => 1,
                0x40 => 4,
                0x80 => 8,
                _ => 14,
            };
        } else {
            self.iir &= !IIR_FIFO;
        }
        self.update_irq();
    }

    /// The modem status inputs follow the outputs in loopback and sit at
    /// their idle levels otherwise; a change latches the delta bits.
    fn update_modem_levels(&mut self) {
        let new = if self.loopback() {
            ((self.mcr & (MCR_OUT1 | MCR_OUT2)) << 4) | ((self.mcr & MCR_RTS) << 3) | ((self.mcr & MCR_DTR) << 5)
        } else {
            MSR_IDLE_LEVELS
        };
        let old = self.msr & 0xf0;
        // CTS, DSR and carrier changes latch straight; ring only on its
        // trailing edge.
        let mut delta = ((new ^ old) >> 4) & (MSR_DCTS | MSR_DDSR | MSR_DDCD);
        if old & MSR_RI != 0 && new & MSR_RI == 0 {
            delta |= MSR_TERI;
        }
        self.msr = new | (self.msr & MSR_ANY_DELTA) | delta;
        self.update_irq();
    }

    // ----- transmit -----

    /// A write of the holding register and the transmit it starts, which here completes
    /// at once: the byte goes to the host (or round to the receiver in
    /// loopback), and the holding register is empty again with its interrupt
    /// raised.
    fn write_thr(&mut self, val: u8) {
        self.thr_ipending = false;
        self.lsr &= !(LSR_THRE | LSR_TEMT);
        self.update_irq();
        if self.loopback() {
            self.receive(&[val]);
        } else {
            if self.out.len() == OUT_KEEP {
                self.out.pop_front();
            }
            self.out.push_back(val);
            self.sent += 1;
            self.backend.send_byte(val);
        }
        self.lsr |= LSR_THRE | LSR_TEMT;
        self.thr_ipending = true;
        self.update_irq();
    }

    // ----- receive -----

    fn read_rbr(&mut self) -> u8 {
        let ret = if self.fcr & FCR_FE != 0 {
            let ret = self.recv_fifo.pop_front().unwrap_or(0);
            if self.recv_fifo.is_empty() {
                self.lsr &= !(LSR_DR | LSR_BI);
                self.timeout_deadline = None;
            } else {
                self.timeout_deadline = Some(Instant::now() + self.char_transmit_time * 4);
            }
            self.timeout_ipending = false;
            ret
        } else {
            // With nothing received the register reads zero, as both models
            // that preceded this one answered, where QEMU returns the last
            // character again.
            let ret = if self.lsr & LSR_DR != 0 { self.rbr } else { 0 };
            self.lsr &= !(LSR_DR | LSR_BI);
            ret
        };
        self.update_irq();
        ret
    }

    /// How many characters the port will take now. With
    /// the FIFOs on, only up to the trigger level, so the guest gets its
    /// interrupt before the FIFO is full.
    fn can_receive(&self) -> usize {
        if self.fcr & FCR_FE != 0 {
            let used = self.recv_fifo.len();
            if used >= FIFO_LENGTH {
                0
            } else if used <= self.recv_fifo_itl {
                self.recv_fifo_itl - used
            } else {
                1
            }
        } else {
            (self.lsr & LSR_DR == 0) as usize
        }
    }

    /// Characters arrive from the line (or from loopback).
    pub fn receive(&mut self, buf: &[u8]) {
        if buf.is_empty() {
            return;
        }
        if self.fcr & FCR_FE != 0 {
            for &b in buf {
                // Overruns do not overwrite FIFO contents.
                if self.recv_fifo.len() < FIFO_LENGTH {
                    self.recv_fifo.push_back(b);
                } else {
                    self.lsr |= LSR_OE;
                }
            }
            self.lsr |= LSR_DR;
            self.timeout_deadline = Some(Instant::now() + self.char_transmit_time * 4);
        } else {
            if self.lsr & LSR_DR != 0 {
                // The register keeps the character it has.
                self.lsr |= LSR_OE;
            } else {
                self.rbr = buf[0];
            }
            self.lsr |= LSR_DR;
        }
        self.update_irq();
    }

    /// Take what the host has typed, as far as the port will accept it.
    /// Returns whether anything arrived. Called by the shell's service loop
    /// and on status and data reads, so a driver polling for data sees it at
    /// once.
    pub fn pump(&mut self) -> bool {
        if self.loopback() {
            // Loopback disconnects the serial input: what arrives on the
            // line goes nowhere and only the looped-back characters reach
            // the receiver. It is dropped when the host offers it, rather
            // than a character time later as on the line, an approximation.
            let mut dropped = self.typed.len();
            self.typed.clear();
            while self.backend.recv_byte().is_ok() {
                dropped += 1;
            }
            if dropped != 0 {
                log::debug!("16550: dropped {dropped} input characters while in loopback");
            }
            return false;
        }
        let mut moved = false;
        loop {
            let n = self.can_receive();
            if n == 0 {
                break;
            }
            let mut buf = [0u8; FIFO_LENGTH];
            let mut got = 0;
            while got < n {
                match self.typed.pop_front().map(Ok).unwrap_or_else(|| self.backend.recv_byte()) {
                    Ok(b) => {
                        buf[got] = b;
                        got += 1;
                    }
                    Err(_) => break,
                }
            }
            if got == 0 {
                break;
            }
            self.receive(&buf[..got]);
            moved = true;
        }
        moved
    }

    /// Resolve the receive timeout at `now`: characters left unread for four
    /// character times raise the timeout cause. Returns whether it did.
    pub fn check_timeout(&mut self, now: Instant) -> bool {
        match self.timeout_deadline {
            Some(d) if d <= now => {
                self.timeout_deadline = None;
                if !self.recv_fifo.is_empty() {
                    self.timeout_ipending = true;
                    self.update_irq();
                    return true;
                }
                false
            }
            _ => false,
        }
    }

    #[cfg(test)]
    fn timeout_deadline(&self) -> Option<Instant> {
        self.timeout_deadline
    }

    // ----- hooks for a DMA engine in front of the port -----

    /// A DMA engine's transmit: the byte goes as a write of the holding
    /// register does. The transmitter is never busy, so it always takes it.
    pub fn dma_write(&mut self, byte: u8) {
        self.write_thr(byte);
    }

    /// A DMA engine's read: the character and the line status that came with
    /// it (reading the status clears its latches), or `None` if nothing is
    /// waiting.
    pub fn dma_read(&mut self) -> Option<(u8, u8)> {
        // The host's characters arrive between the engine's looks.
        self.pump();
        if self.lsr & LSR_DR == 0 {
            return None;
        }
        let status = self.read(LSR);
        let byte = self.read_rbr();
        Some((byte, status))
    }

    pub fn dma_modem_control(&mut self, val: u8) {
        self.write(MCR, val);
    }

    // ----- snapshots -----

    pub fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("divider".into(), toml::Value::Integer(self.divider as i64));
        for (k, v) in [
            ("rbr", self.rbr),
            ("ier", self.ier),
            ("iir", self.iir),
            ("lcr", self.lcr),
            ("mcr", self.mcr),
            ("lsr", self.lsr),
            ("msr", self.msr),
            ("scr", self.scr),
            ("fcr", self.fcr),
        ] {
            t.insert(k.into(), hex_u8(v));
        }
        t.insert("thr_ipending".into(), toml::Value::Boolean(self.thr_ipending));
        t.insert("timeout_ipending".into(), toml::Value::Boolean(self.timeout_ipending));
        t.insert("recv_fifo".into(), toml::Value::Array(self.recv_fifo.iter().map(|&b| hex_u8(b)).collect()));
        t.insert("typed".into(), toml::Value::Array(self.typed.iter().map(|&b| hex_u8(b)).collect()));
        toml::Value::Table(t)
    }

    pub fn load_state(&mut self, v: &toml::Value) {
        let b = |k: &str| get_field(v, k).and_then(toml_u8).unwrap_or(0);
        self.divider = get_field(v, "divider").and_then(toml_u32).unwrap_or(0x0c) as u16;
        self.rbr = b("rbr");
        self.ier = b("ier");
        self.iir = b("iir");
        self.lcr = b("lcr");
        self.mcr = b("mcr");
        self.lsr = b("lsr");
        self.msr = b("msr");
        self.scr = b("scr");
        self.fcr = b("fcr");
        self.thr_ipending = get_field(v, "thr_ipending").and_then(toml_bool).unwrap_or(false);
        self.timeout_ipending = get_field(v, "timeout_ipending").and_then(toml_bool).unwrap_or(false);
        self.recv_fifo = match get_field(v, "recv_fifo") {
            Some(toml::Value::Array(a)) => a.iter().take(FIFO_LENGTH).map(|x| toml_u8(x).unwrap_or(0)).collect(),
            _ => VecDeque::new(),
        };
        self.typed = match get_field(v, "typed") {
            Some(toml::Value::Array(a)) => a.iter().map(|x| toml_u8(x).unwrap_or(0)).collect(),
            _ => VecDeque::new(),
        };
        self.recv_fifo_itl = match self.fcr & 0xc0 {
            0x40 => 4,
            0x80 => 8,
            0xc0 => 14,
            _ => 1,
        };
        self.update_parameters();
        // The timeout restarts from the load.
        self.timeout_deadline = (!self.recv_fifo.is_empty() && !self.timeout_ipending)
            .then(|| Instant::now() + self.char_transmit_time * 4);
        self.update_irq();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Wire {
        out: Mutex<Vec<u8>>,
        inp: Mutex<VecDeque<u8>>,
    }

    impl SerialBackend for Wire {
        fn send_byte(&self, b: u8) {
            self.out.lock().unwrap().push(b);
        }
        fn recv_byte(&self) -> io::Result<u8> {
            self.inp.lock().unwrap().pop_front().ok_or_else(|| io::Error::new(io::ErrorKind::WouldBlock, "empty"))
        }
    }

    fn uart() -> (Ns16550, Arc<Wire>) {
        let w = Arc::new(Wire::default());
        let mut u = Ns16550::new(115200);
        u.set_backend(w.clone());
        (u, w)
    }

    /// Loopback, with the FIFOs on or off, so `write(THR)` is also a receive.
    fn looped(fcr: u8) -> Ns16550 {
        let (mut u, _) = uart();
        u.write(IIR_FCR, fcr);
        u.write(MCR, MCR_LOOP);
        u.read(MSR);
        u
    }

    #[test]
    fn a_port_comes_up_idle_with_carrier_at_9600_baud() {
        let (mut u, _) = uart();
        assert_eq!((u.read(IER), u.read(IIR_FCR), u.read(LCR), u.read(SCR)), (0, 1, 0, 0));
        assert_eq!((u.read(MCR), u.read(LSR), u.read(MSR)), (0x08, 0x60, 0xb0));
        u.write(LCR, 0x80);
        assert_eq!((u.read(THR), u.read(IER)), (0x0c, 0), "divisor 12 of 115200");
    }

    #[test]
    fn the_divisor_latch_hides_data_and_enable_and_sets_the_character_time() {
        let (mut u, w) = uart();
        u.write(IER, 0x05);
        u.write(LCR, 0x83);
        u.write(THR, 0x01);
        u.write(IER, 0x00);
        assert_eq!((u.read(THR), u.read(IER)), (1, 0));
        assert!(w.out.lock().unwrap().is_empty(), "the divisor is not transmitted");
        u.write(LCR, 0x03);
        assert_eq!(u.read(IER), 0x05, "the enable register was not touched");
        assert_eq!(u.char_transmit_time.as_micros(), 86, "ten bits at 115200 baud");
        u.write(LCR, 0x1b);
        assert_eq!(u.char_transmit_time.as_micros(), 95, "parity adds a bit");
    }

    #[test]
    fn without_fifos_a_second_byte_waits_in_the_backend_until_the_first_is_read() {
        let (mut u, w) = uart();
        w.inp.lock().unwrap().extend(b"ab");
        assert_eq!(u.read(LSR) & 3, 1, "one byte in, no overrun");
        assert_eq!(w.inp.lock().unwrap().len(), 1, "the other is still the backend's");
        assert_eq!(u.read(THR), b'a');
        assert_eq!(u.read(LSR) & 3, 1, "reading made room for it");
        assert_eq!(u.read(THR), b'b');
        assert_eq!(u.read(LSR) & 1, 0);
    }

    /// With modem-control bit 4 set the part disconnects its serial input: a
    /// character from the line goes nowhere, whether the host typed it or its
    /// backend offered it, and only looped-back ones reach the receiver.
    ///
    /// QEMU: sn0/ioc3-serial/loopback-discards-external-input.
    #[test]
    fn loopback_discards_input_from_the_line() {
        let (mut u, w) = uart();
        u.write(MCR, MCR_LOOP);
        w.inp.lock().unwrap().extend(b"o");
        u.type_in(b"t");
        assert!(!u.pump());
        assert_eq!(u.read(LSR) & 1, 0, "nothing reached the receiver");
        assert!(w.inp.lock().unwrap().is_empty(), "the backend's character was taken and dropped");
        u.write(THR, b'l');
        assert_eq!((u.read(LSR) & 1, u.read(THR)), (1, b'l'), "a looped-back one still lands");
        u.write(MCR, 0);
        u.type_in(b"x");
        assert_eq!(u.read(THR), b'x', "and the line works again outside loopback");
    }

    #[test]
    fn the_fifo_interrupts_at_its_trigger_level_and_a_flush_empties_it() {
        let (mut u, w) = uart();
        u.write(IER, IER_RX);
        u.write(IIR_FCR, 0xc1);
        assert_eq!(u.read(IIR_FCR) & 0xc0, 0xc0);
        w.inp.lock().unwrap().extend(0..40u8);
        u.pump();
        assert_eq!(u.rx_pending(), 14, "the port takes the host's bytes up to its trigger level");
        assert_eq!(u.iir & 0x0f, IIR_RX_DATA);
        u.write(IIR_FCR, 0xc3);
        assert_eq!((u.rx_pending(), u.irq()), (0, false));
        u.write(IIR_FCR, 0);
        assert_eq!(u.read(IIR_FCR) & 0xc0, 0, "FCR bit 0 turns the FIFOs off");
    }

    #[test]
    fn interrupt_causes_come_in_priority_order_and_clear_on_their_read() {
        let mut u = looped(0);
        u.write(IER, 0x0f);
        u.read(IIR_FCR);
        u.write(THR, b'1');
        u.write(THR, b'2');
        u.write(MCR, MCR_LOOP | MCR_RTS);
        assert_eq!(u.iir & 0x0f, IIR_RLSI, "line status first");
        u.read(LSR);
        assert_eq!(u.iir & 0x0f, IIR_RX_DATA);
        assert_eq!(u.read(THR), b'1');
        assert_eq!(u.read(IIR_FCR) & 0x0f, IIR_TX_EMPTY, "then the empty holding register");
        assert_eq!(u.read(IIR_FCR) & 0x0f, IIR_MSI, "reading the transmit cause cleared it; modem last");
        u.read(MSR);
        assert_eq!(u.read(IIR_FCR) & 0x0f, IIR_NONE);
    }

    #[test]
    fn a_cause_needs_its_enable_bit_and_the_high_nibble_is_dropped() {
        let mut u = looped(0);
        u.write(THR, b'x');
        u.write(MCR, MCR_LOOP | MCR_RTS);
        assert_eq!(u.read(IIR_FCR), IIR_NONE, "data, transmit-empty and a modem change pending, all masked");
        u.write(IER, 0xf1);
        assert_eq!(u.read(IER), 0x01);
        assert_eq!(u.read(IIR_FCR) & 0x0f, IIR_RX_DATA);
        u.write(IER, IER_TX);
        assert_eq!(u.read(IIR_FCR) & 0x0f, IIR_TX_EMPTY, "turning the enable on raises the cause");
    }

    #[test]
    fn line_status_reports_overrun_and_an_empty_transmitter() {
        let mut u = looped(0);
        assert_eq!(u.read(LSR), 0x60);
        u.write(THR, b'a');
        u.write(THR, b'b');
        assert_eq!(u.read(LSR) & 3, 3, "a second byte overruns the one-byte register");
        assert_eq!(u.read(LSR) & 3, 1, "the overrun clears on read");
        assert_eq!((u.read(THR), u.read(THR)), (b'a', 0), "the first is kept; an empty register reads zero");
        let mut u = looped(1);
        for b in 0..17u8 {
            u.write(THR, b);
        }
        assert_eq!(u.read(LSR) & 2, 2, "the seventeenth overruns the FIFO");
        assert_eq!((0..16u8).filter(|&b| u.read(THR) == b).count(), 16, "and the sixteen are kept");
    }

    #[test]
    fn loopback_wires_the_modem_outputs_to_the_inputs_and_latches_changes() {
        let (mut u, _) = uart();
        u.write(IER, IER_MS);
        u.write(MCR, MCR_LOOP | MCR_OUT1 | MCR_RTS);
        assert_eq!(u.iir & 0x0f, IIR_MSI);
        assert_eq!(u.read(MSR), 0x5a, "ring and CTS up; DSR and carrier moved");
        u.write(MCR, MCR_LOOP | MCR_DTR | MCR_OUT2);
        let msr = u.read(MSR);
        assert_eq!(msr & 0xf0, 0xa0, "DSR and carrier");
        assert_eq!(msr & 0x04, 0x04, "ring fell: the trailing edge");
        assert_eq!(u.read(MSR) & 0x0f, 0, "a read clears the changes");
        assert_eq!(u.read(IIR_FCR), IIR_NONE);
    }

    #[test]
    fn the_timeout_restarts_after_a_partial_read_and_a_flush_clears_it() {
        let (mut u, _) = uart();
        u.write(IER, IER_RX);
        u.write(IIR_FCR, 0xc1);
        u.receive(&[1, 2]);
        let first = u.timeout_deadline().expect("armed");
        assert!(!u.check_timeout(first - Duration::from_micros(1)));
        assert_eq!(u.read(THR), 1);
        let second = u.timeout_deadline().expect("a partial read starts it again");
        assert!(second >= first);
        assert!(u.check_timeout(second));
        assert_eq!(u.iir & 0x0f, IIR_CTI);
        u.write(IIR_FCR, 0xc7);
        assert_eq!((u.irq(), u.timeout_deadline()), (false, None), "a flush clears it");
        assert_eq!(u.read(IIR_FCR) & 0x0f, IIR_NONE);
        u.write(IER, IER_TX);
        u.write(IIR_FCR, 0xc7);
        assert_eq!(u.iir & 0x0f, IIR_TX_EMPTY, "resetting the transmit FIFO raises the empty cause");
    }

    #[test]
    fn transmit_is_instant_and_the_kept_output_is_bounded() {
        let (mut u, w) = uart();
        for _ in 0..OUT_KEEP + 5 {
            u.write(THR, b'z');
        }
        u.write(THR, b'!');
        assert_eq!(u.read(LSR) & 0x60, 0x60, "the transmitter is never busy");
        assert_eq!(w.out.lock().unwrap().len(), OUT_KEEP + 6, "every byte reached the backend");
        assert_eq!((u.bytes_out(), u.output_tail(usize::MAX).len()), (OUT_KEEP as u64 + 6, OUT_KEEP));
        assert!(u.output_tail(2).ends_with("z!"));
    }
}
