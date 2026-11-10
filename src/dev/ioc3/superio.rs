//! The Super I/O behind the IOC3: two 16550 UARTs and a parallel port, in the
//! 0x20000 register window.
//!
//! The IOC3 does not contain these. Linux lays the window out as
//! `ioc3_sioregs` (UART B at 0x170, UART A at 0x178, the parallel port's
//! 1284 ECP registers around 0x150, and two configuration bytes), and drives
//! the UARTs through the stock 8250 driver at `offset ^ 3`. OpenBSD attaches
//! its generic `com` and `lpt` drivers at `IOC3_UARTA_BASE` and
//! `IOC3_UARTB_BASE` in the same window. The chip's own logic sits in front of
//! them: the ring-DMA engines in `serial.rs` and `parallel.rs`, the status and
//! enable words that gate the parts' interrupt outputs, and the PS/2
//! controller in `kbd.rs`, which Linux and OpenBSD reach through the main
//! register block, not this window.
//!
//! Nothing in the drivers' view tells the BaseIO's Super I/O from the
//! MediaIO's, or from the Octane's: every board gets the same 16550 and ECP
//! register files, so there is one model. The UARTs and the parallel port share one lock
//! because they are one part.
//!
//! Ported from hw/misc/sgi-ioc3.c and its serial and parallel siblings in
//! jamesbraid/qemu-sgi-origami.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

use super::{int, Core};
use crate::dev::ns16550::Ns16550;
use crate::dev::serial_host::SerialBackend;
use crate::snapshot::{field, field_bytes, field_u8, StateTable};

pub(super) mod uart_reg {
    pub const RBR_THR: u32 = 0;
    pub const IIR_FCR: u32 = 2;
    pub const MCR: u32 = 4;
    pub const LSR: u32 = 5;
    pub const MSR: u32 = 6;
    pub const SCR: u32 = 7;
}
pub(super) const LSR_DR: u8 = 0x01;
pub(super) const LSR_OE: u8 = 0x02;
pub(super) const LSR_PE: u8 = 0x04;
pub(super) const LSR_FE: u8 = 0x08;
pub(super) const LSR_BI: u8 = 0x10;
pub(super) const MCR_DTR: u8 = 0x01;
pub(super) const MCR_RTS: u8 = 0x02;
pub(super) const MCR_OUT1: u8 = 0x04;
pub(super) const MCR_OUT2: u8 = 0x08;
pub(super) const MCR_LOOP: u8 = 0x10;
pub(super) const MCR_AFE: u8 = 0x20;
pub(super) const MSR_DCTS: u8 = 0x01;
pub(super) const MSR_CTS: u8 = 0x10;
pub(super) const MSR_DCD: u8 = 0x80;
pub(super) const LSR_TEST_MASK: u8 = 0x7f;
pub(super) const MSR_TEST_MASK: u8 = 0x0f;
/// The reference clock of the UART crystal: the pre-divider divides it, and
/// the divisor latch divides the result by 16 again.
pub(super) const UART_CRYSTAL_HZ: u32 = 22_118_400;

/// A 16550 as IOC3 fits it: the shared core (`ns16550.rs`) with the chip's
/// own additions around it. Transmit is instantaneous (the byte goes to the
/// backend inside the write), so the transmitter always reads empty.
///
/// The additions: the auto-flow bit in modem control, which the core's modem
/// control masks away; factory-test writes to line and modem status; and the
/// scratch register doubling as the clock pre-divider while the divisor latch
/// is open. Ported from hw/misc/sgi-ioc3.c in jamesbraid/qemu-sgi-origami.
pub(super) struct Uart {
    pub(super) core: Ns16550,
    // IOC3 specifics, kept beside the 16550.
    pub(super) predivider: u8,
    mcr_afe: u8,
    lsr_test: u8,
    msr_test: u8,
}

impl Uart {
    fn new() -> Uart {
        Uart { core: Ns16550::new(115200), predivider: 0, mcr_afe: 0, lsr_test: 0, msr_test: 0 }
    }

    fn reset(&mut self) {
        self.core.reset();
        self.mcr_afe = 0;
        self.lsr_test = 0;
        self.msr_test = 0;
        // The pre-divider keeps its value across a reset; no reset value is
        // specified for it.
    }

    pub(super) fn loopback(&self) -> bool {
        self.core.loopback()
    }

    /// The interrupt line: low IIR bit clear.
    pub(super) fn irq(&self) -> bool {
        self.core.irq()
    }

    /// Move whatever the host end has typed into the receive FIFO and let
    /// the receive timeout run out. Returns whether anything changed.
    pub(super) fn pump(&mut self) -> bool {
        let moved = self.core.pump();
        self.core.check_timeout(Instant::now()) || moved
    }

    /// The 16550 proper: what the ring engine sees.
    pub(super) fn read_core(&mut self, r: u32) -> u8 {
        self.core.read(r)
    }

    /// A guest's byte-wide read, with the chip's additions: the auto-flow
    /// bit in modem control, and a factory-test write to line or modem
    /// status showing in the read that follows it and no later one.
    pub(super) fn read(&mut self, r: u32) -> u8 {
        let mut v = self.read_core(r);
        match r {
            uart_reg::MCR => v |= self.mcr_afe,
            uart_reg::LSR => {
                v |= self.lsr_test;
                self.lsr_test = 0;
            }
            uart_reg::MSR => {
                v |= self.msr_test;
                self.msr_test = 0;
            }
            _ => {}
        }
        v
    }

    pub(super) fn write(&mut self, r: u32, v: u8) {
        match r {
            uart_reg::MCR => self.mcr_afe = v & MCR_AFE,
            uart_reg::LSR => self.lsr_test = v & LSR_TEST_MASK,
            uart_reg::MSR => self.msr_test = v & MSR_TEST_MASK,
            uart_reg::SCR => {
                // Scratch writes with the divisor latch up program the clock
                // pre-divider; twice the divide wanted. 0 and 1 mean nothing.
                if self.core.dlab() && v / 2 != 0 {
                    self.predivider = v;
                    self.core.set_baudbase(UART_CRYSTAL_HZ / (v / 2) as u32 / 16);
                }
            }
            _ => {}
        }
        self.core.write(r, v);
    }

    /// Modem status as the ring engine reads it with flow control looped
    /// back: `flow` is the engine's flow-control bit.
    pub(super) fn flow_cts(&self, flow: bool) -> bool {
        self.loopback() && flow
    }

    fn save_state(&self) -> toml::Value {
        StateTable::new()
            .value("core", self.core.save_state())
            .u8("predivider", self.predivider)
            .u8("mcr_afe", self.mcr_afe)
            .u8("lsr_test", self.lsr_test)
            .u8("msr_test", self.msr_test)
            .build()
    }

    fn load_state(&mut self, v: &toml::Value) -> Result<(), String> {
        self.core.load_state(field(v, "core")?);
        self.predivider = field_u8(v, "predivider")?;
        self.mcr_afe = field_u8(v, "mcr_afe")?;
        self.lsr_test = field_u8(v, "lsr_test")?;
        self.msr_test = field_u8(v, "msr_test")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Parallel port (TL16PIR552 population, 16-byte test FIFO)
// ---------------------------------------------------------------------------

pub(super) const PAR_ECR_FIFO_EMPTY: u8 = 0x01;
pub(super) const PAR_ECR_FIFO_FULL: u8 = 0x02;
pub(super) const PAR_ECR_DMA_ENABLE: u8 = 0x08;
pub(super) const PAR_ECR_STORED: u8 = 0xfc;
pub(super) const PAR_MODE_TEST: u8 = 6;
pub(super) const PAR_MODE_CONFIG: u8 = 7;
pub(super) const PAR_FIFO_DEPTH: usize = 16;
/// The part's fixed configuration: implementation id and reserved bits;
/// IRQ7 and DMA3.
pub(super) const PAR_CONFIG_A: u8 = 0x1f;
pub(super) const PAR_CONFIG_B: u8 = 0x0b;
// The status lines as the external SuperIO part reads them: bit 7 is the
// inverse of the Busy pin, bits 6 to 3 are nAck, PError, Select and nFault as
// the pins stand (nAck and nFault are active low). Bit 2, the part's PRINT
// indicator, is not modelled and reads clear.
pub(super) const PAR_STATUS_NFAULT: u8 = 1 << 3;
pub(super) const PAR_STATUS_SELECT: u8 = 1 << 4;
pub(super) const PAR_STATUS_NACK: u8 = 1 << 6;
pub(super) const PAR_STATUS_BUSY: u8 = 1 << 7;
/// Nothing on the connector: each of the five inputs sits at the level a
/// pull-up gives it, high, and Busy high reads as a clear bit 7. The pull
/// direction on the real boards is inferred; nothing measures an unplugged
/// port.
pub(super) const PAR_STATUS_EMPTY: u8 = PAR_STATUS_NFAULT | PAR_STATUS_SELECT | 1 << 5 | PAR_STATUS_NACK;
/// A ready printer: Busy low (bit 7 set), nAck, Select and nFault high, no
/// paper error. Every host backend is taken as one that always accepts what
/// is sent, an approximation: a file or pty never holds Busy.
pub(super) const PAR_STATUS_READY: u8 = PAR_STATUS_BUSY | PAR_STATUS_NACK | PAR_STATUS_SELECT | PAR_STATUS_NFAULT;

#[derive(Default)]
struct Ecp {
    data: u8,
    control: u8,
    config_b: u8,
    extended: u8,
    fifo: VecDeque<u8>,
    backend: Option<Arc<dyn SerialBackend>>,
}

impl Ecp {
    fn mode(&self) -> u8 {
        self.extended >> 5
    }

    fn read(&mut self, r: u32) -> u8 {
        match r {
            0 => self.data,
            // The lines follow the cable: a peripheral, or pull-ups on nothing.
            1 if self.backend.is_some() => PAR_STATUS_READY,
            1 => PAR_STATUS_EMPTY,
            2 => self.control,
            8 => {
                if self.mode() == PAR_MODE_CONFIG {
                    PAR_CONFIG_A
                } else if self.mode() == PAR_MODE_TEST {
                    self.fifo.pop_front().unwrap_or(0)
                } else {
                    0
                }
            }
            9 => if self.mode() == PAR_MODE_CONFIG { PAR_CONFIG_B } else { self.config_b },
            0xa => {
                if self.mode() == PAR_MODE_TEST {
                    let mut s = self.extended;
                    if self.fifo.is_empty() {
                        s |= PAR_ECR_FIFO_EMPTY;
                    }
                    if self.fifo.len() >= PAR_FIFO_DEPTH {
                        s |= PAR_ECR_FIFO_FULL;
                    }
                    s
                } else {
                    self.extended | PAR_ECR_FIFO_EMPTY
                }
            }
            _ => 0,
        }
    }

    fn write(&mut self, r: u32, v: u8) {
        match r {
            0 => self.data = v,
            2 => self.control = v,
            8 => match self.mode() {
                // The fitted part's configuration is read-only.
                PAR_MODE_CONFIG => {}
                PAR_MODE_TEST => {
                    // The part's behaviour on overrun is unspecified;
                    // dropping the oldest byte is a model choice.
                    if self.fifo.len() >= PAR_FIFO_DEPTH {
                        self.fifo.pop_front();
                    }
                    self.fifo.push_back(v);
                }
                _ => {
                    if let Some(b) = &self.backend {
                        b.send_byte(v);
                    }
                }
            },
            9 => {
                if self.mode() != PAR_MODE_CONFIG {
                    self.config_b = v;
                }
            }
            0xa => {
                self.extended = v & PAR_ECR_STORED;
                let m = self.mode();
                if m == 0 || m == 1 {
                    self.fifo.clear();
                }
            }
            _ => {}
        }
    }
}

struct Sio {
    uart: [Uart; 2],
    ecp: Ecp,
}

/// The Super I/O part. Which of its functions a board brings out is data
/// in the board row; an absent one reads zero.
pub(super) struct SuperIo {
    core: Arc<Core>,
    uarts: [bool; 2],
    parallel: bool,
    st: Mutex<Sio>,
}

impl SuperIo {
    pub(super) fn new(core: Arc<Core>, uarts: [bool; 2], parallel: bool) -> SuperIo {
        SuperIo { core, uarts, parallel, st: Mutex::new(Sio { uart: [Uart::new(), Uart::new()], ecp: Ecp::default() }) }
    }

    pub(super) fn has_uart(&self, p: usize) -> bool {
        self.uarts[p]
    }

    pub(super) fn has_parallel(&self) -> bool {
        self.parallel
    }

    /// Run `f` on UART `p`, then put the part's interrupt output on its
    /// pass-through bit in the status word: the bit follows the 16550's
    /// level, set or cleared under the part's lock so two threads cannot
    /// leave it stale.
    pub(super) fn uart<R>(&self, p: usize, f: impl FnOnce(&mut Uart) -> R) -> R {
        let mut st = self.st.lock();
        let r = f(&mut st.uart[p]);
        self.core.level(int::serial(p, int::PASSTHROUGH), st.uart[p].irq());
        r
    }

    pub(super) fn set_serial_backend(&self, p: usize, backend: Arc<dyn SerialBackend>) {
        self.st.lock().uart[p].core.set_backend(backend);
    }

    pub(super) fn set_parallel_backend(&self, backend: Arc<dyn SerialBackend>) {
        self.st.lock().ecp.backend = Some(backend);
    }

    /// Whether the port's ECR has DMA enabled.
    pub(super) fn parallel_dma_enabled(&self) -> bool {
        self.st.lock().ecp.extended & PAR_ECR_DMA_ENABLE != 0
    }

    /// Bytes the engine moved, to the host end if there is one.
    pub(super) fn parallel_send(&self, bytes: &[u8]) {
        if let Some(b) = &self.st.lock().ecp.backend {
            for &x in bytes {
                b.send_byte(x);
            }
        }
    }

    pub(super) fn is_locked(&self) -> bool {
        self.st.is_locked()
    }

    /// A byte-wide read of the parallel port's register `r`.
    pub(super) fn ecp_read(&self, r: u32) -> u8 {
        self.st.lock().ecp.read(r)
    }

    pub(super) fn reset(&self) {
        let mut st = self.st.lock();
        for u in &mut st.uart {
            u.reset();
        }
        let backend = st.ecp.backend.take();
        st.ecp = Ecp { backend, ..Ecp::default() };
    }

    pub(super) fn save_state(&self, p: usize) -> toml::Value {
        self.st.lock().uart[p].save_state()
    }

    pub(super) fn load_state(&self, p: usize, v: &toml::Value) -> Result<(), String> {
        self.st.lock().uart[p].load_state(v)
    }

    /// The parallel port's register file for a snapshot, as the keys of the
    /// `parallel` table.
    pub(super) fn save_ecp(&self, t: &mut StateTable) {
        let st = self.st.lock();
        let fifo: Vec<u8> = st.ecp.fifo.iter().copied().collect();
        t.u8("data", st.ecp.data)
            .u8("control", st.ecp.control)
            .u8("config_b", st.ecp.config_b)
            .u8("extended", st.ecp.extended)
            .u8s("fifo", &fifo);
    }

    pub(super) fn load_ecp(&self, p: &toml::Value) -> Result<(), String> {
        let fifo = field_bytes(p, "fifo")?;
        if fifo.len() > PAR_FIFO_DEPTH {
            return Err(format!("parallel fifo: {} bytes", fifo.len()));
        }
        let mut st = self.st.lock();
        st.ecp.data = field_u8(p, "data")?;
        st.ecp.control = field_u8(p, "control")?;
        st.ecp.config_b = field_u8(p, "config_b")?;
        st.ecp.extended = field_u8(p, "extended")?;
        st.ecp.fifo = fifo.into();
        Ok(())
    }
}

impl SuperIo {
    /// A byte-wide write of the parallel port's register `r`. A write of the
    /// ECR is the caller's to follow with the DMA engine.
    pub(super) fn ecp_write(&self, r: u32, v: u8) {
        self.st.lock().ecp.write(r, v);
    }
}
