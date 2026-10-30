// Derived from MAME's xbox_usb.cpp and xbox_usb.h (the OHCI host controller),
// BSD-3-Clause, copyright Samuele Zannoli.

//! The Lucent USS-312 USB controller: two root ports, OHCI registers and
//! descriptor processing through the PCI bridge's DMA interface.
//!
//! Functions plug into a root port with `attach`: the Fuel fits a keyboard and
//! a mouse (`usb_hid`), and the tests drive the descriptor and packet
//! scheduling engine through a test-only function. Isochronous
//! traffic, suspend/resume, external hubs, USB storage and legacy keyboard/SMI
//! routing are unsupported. Scheduling uses synthetic work units rather than
//! wire time. Register behavior follows OpenHCI 1.0a and the USS-312 revision 5
//! data sheet.
//!
//! Ported from src/mame/shared/xbox_usb.cpp and xbox_usb.h in MAME,
//! revision 2ee9b1e30299b122b131cab0bd7912abc2b095ba.

use crate::dev::heart::{host_clock, Clock};
use crate::irq::IrqLine;
use crate::pci::{Bar, PciBusDevice, PciConfig, PciConfigPolicy, PciDma, PCI_INTA};
use crate::traits::{Device, Resettable};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Pid {
    Setup = 0,
    Out = 1,
    In = 2,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsbPacket {
    pub endpoint: u8,
    pub pid: Pid,
    pub toggle: bool,
    /// IN requests name capacity; OUT and SETUP carry their bytes.
    pub capacity: usize,
    pub data: Vec<u8>,
    pub time_ms: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UsbOutcome {
    Ack(Vec<u8>),
    Nak,
    Stall,
    NoResponse,
    CrcError,
    BitStuffError,
}
pub trait UsbFunction: Send + Sync {
    fn reset(&self);
    fn address(&self) -> u8;
    fn low_speed(&self) -> bool {
        false
    }
    fn transact(&self, packet: UsbPacket) -> UsbOutcome;
}

mod engine;
mod schedule;

const EVENTS: u32 = 0x4000_007f;
const WDH: u32 = 2;
const UE: u32 = 16;
const RHSC: u32 = 64;
const PPS: u32 = 0x100;
const CHANGES: u32 = 0x1f0000;
#[derive(Clone)]
struct State {
    r: [u32; 23],
    ports: [Option<Arc<dyn UsbFunction>>; 2],
    low_speed: [bool; 2],
    reset_ms: [u8; 2],
    done_delay: u8,
    now_ms: u64,
    active_fi: u32,
    bit_credit: u64,
}
impl State {
    fn new() -> Self {
        let mut r = [0; 23];
        r[0] = 0x110;
        r[13] = 0x2edf;
        r[17] = 0x628;
        r[18] = 0x1000_0902;
        r[19] = 0x60000;
        Self {
            r,
            ports: [None, None],
            low_speed: [false; 2],
            reset_ms: [0; 2],
            done_delay: 7,
            now_ms: 0,
            active_fi: 0x2edf,
            bit_credit: 0,
        }
    }
    fn soft_reset(&mut self) {
        let control = (self.r[1] & 0x300) | 0xc0;
        let keep = [self.r[18], self.r[19], self.r[20], self.r[21], self.r[22]];
        let ports = self.ports.clone();
        let low_speed = self.low_speed;
        let resets = self.reset_ms;
        let time = self.now_ms;
        *self = Self::new();
        self.r[1] = control;
        self.r[18..23].copy_from_slice(&keep);
        self.ports = ports;
        self.low_speed = low_speed;
        self.reset_ms = resets;
        self.now_ms = time;
    }
    fn irq(&self) -> bool {
        self.r[1] & 0x100 == 0 && self.r[4] & 0x80000000 != 0 && self.r[3] & self.r[4] & EVENTS != 0
    }
    fn powered(&self, p: usize) -> bool {
        self.r[18] & 0x200 != 0 || self.r[21 + p] & PPS != 0
    }
    fn individual(&self, p: usize) -> bool {
        self.r[18] & 0x100 != 0 && self.r[19] & (1 << (17 + p)) != 0
    }
    fn power(&mut self, p: usize, on: bool) {
        if self.r[18] & 0x200 != 0 || on {
            self.r[21 + p] |= PPS;
            if self.ports[p].is_some() && self.r[21 + p] & 1 == 0 {
                self.r[21 + p] |= 1 | 0x10000 | if self.low_speed[p] { 0x200 } else { 0 };
                self.r[3] |= RHSC;
            }
        } else {
            let previous = self.r[21 + p];
            self.r[21 + p] &= !(PPS | 0x17);
            if previous & 1 != 0 {
                self.r[21 + p] |= 0x10000;
            }
            if previous & 2 != 0 {
                self.r[21 + p] |= 0x20000;
            }
            if previous & 3 != 0 {
                self.r[3] |= RHSC;
            }
            self.reset_ms[p] = 0;
        }
    }
}

pub struct Ohci {
    config: PciConfig,
    state: Mutex<State>,
    pass: Mutex<()>,
    dma: Mutex<Option<Arc<dyn PciDma>>>,
    irq: Mutex<IrqLine>,
    running: AtomicBool,
    /// Source of the frame count: the machine's own clock, so the guest's
    /// idea of a millisecond and the controller's agree.
    time: OnceLock<Clock>,
    /// `time` reading at the last frame boundary delivered; `None` while stopped.
    clock: Mutex<Option<u64>>,
}
impl Ohci {
    pub fn new() -> Arc<Self> {
        let config = PciConfig::new_with_policy(
            0x11c1,
            0x5802,
            0x0c0310,
            0x10,
            [
                Bar::Mem(0x1000),
                Bar::Unused,
                Bar::Unused,
                Bar::Unused,
                Bar::Unused,
                Bar::Unused,
            ],
            PCI_INTA,
            PciConfigPolicy {
                status: 0x0200,
                status_w1c_mask: 0x8000,
                command_mask: 0x0006,
                latency_mask: 0,
                cache_line_mask: 0,
                registers: &[],
            },
        );
        config.set_subsystem_identity(0xffff_ffff);
        Arc::new(Self {
            config,
            state: Mutex::new(State::new()),
            pass: Mutex::new(()),
            dma: Mutex::new(None),
            irq: Mutex::new(IrqLine::unconnected()),
            running: AtomicBool::new(false),
            time: OnceLock::new(),
            clock: Mutex::new(None),
        })
    }
    /// Use `clock` for frame timing. Call before `start`.
    pub fn set_clock(&self, clock: Clock) {
        assert!(
            !self.running.load(Ordering::Acquire) && self.time.set(clock).is_ok(),
            "connect the OHCI clock once, before start"
        );
    }
    fn now(&self) -> u64 {
        (self.time.get_or_init(host_clock))()
    }
    /// Deliver every frame boundary the clock has passed. Every register
    /// access calls this, so a read sees the frame count the guest's own
    /// clock implies, however late the machine's poll or timer ran.
    pub fn sync_clock(&self) {
        // Held through delivery: a second caller must not find the boundary
        // consumed and read the frame count before the first has applied it.
        let mut clock = self.clock.lock();
        let Some(at) = clock.as_mut() else { return };
        let now = self.now();
        let elapsed = now.saturating_sub(*at) / 1_000_000;
        *at += elapsed * 1_000_000;
        self.advance_frames(elapsed);
    }
    fn drive_irq(&self) {
        loop {
            let level = self.state.lock().irq();
            let line = self.irq.lock().clone();
            line.set(level);
            // A callback may acknowledge the interrupt or another writer may
            // change it while delivery is outside the state lock.
            if self.state.lock().irq() == level {
                break;
            }
        }
    }
    pub fn attach(&self, port: usize, function: Arc<dyn UsbFunction>) -> Result<(), String> {
        if port >= 2 {
            return Err("OHCI has two ports".into());
        }
        if self.state.lock().ports[port].is_some() {
            return Err("OHCI port occupied".into());
        }
        let low_speed = function.low_speed();
        function.reset();
        {
            let _pass = self.pass.lock();
            let mut s = self.state.lock();
            if s.ports[port].is_some() {
                return Err("OHCI port occupied".into());
            }
            if s.powered(port) {
                s.r[21 + port] |= 1 | 0x10000;
            }
            s.low_speed[port] = low_speed;
            if low_speed {
                s.r[21 + port] |= 0x200;
            }
            s.ports[port] = Some(function);
            s.r[3] |= RHSC;
        }
        self.drive_irq();
        Ok(())
    }
    #[cfg(test)]
    pub fn detach(&self, port: usize) -> Result<(), String> {
        if port >= 2 {
            return Err("OHCI has two ports".into());
        }
        {
            let _pass = self.pass.lock();
            let mut s = self.state.lock();
            let enabled = s.r[21 + port] & 2;
            s.ports[port] = None;
            s.low_speed[port] = false;
            s.reset_ms[port] = 0;
            s.r[21 + port] =
                (s.r[21 + port] & PPS) | 0x10000 | if enabled != 0 { 0x20000 } else { 0 };
            s.r[3] |= RHSC;
        }
        self.drive_irq();
        Ok(())
    }
    /// Deterministic millisecond ticks; `sync_clock` delivers the same ticks from the clock.
    pub fn advance_frames(&self, count: u64) {
        for _ in 0..count {
            if !self.running.load(Ordering::Acquire) {
                break;
            }
            {
                let _pass = self.pass.lock();
                if !self.running.load(Ordering::Acquire) {
                    break;
                }
                let mut s = self.state.lock().clone();
                self.frame(&mut s);
                *self.state.lock() = s;
            }
            self.drive_irq();
        }
    }
    fn frame(&self, s: &mut State) {
        s.now_ms = s.now_ms.wrapping_add(1);
        for p in 0..2 {
            if s.reset_ms[p] != 0 {
                s.reset_ms[p] -= 1;
                if s.reset_ms[p] == 0 {
                    s.r[21 + p] = (s.r[21 + p] & !16) | 2 | (1 << 20);
                    s.r[3] |= RHSC;
                }
            }
        }
        if s.r[1] & 0xc0 != 0x80 || s.r[3] & UE != 0 {
            return;
        }
        s.bit_credit += 12000;
        while s.bit_credit >= u64::from((s.active_fi & 0x3fff) + 1) {
            s.bit_credit -= u64::from((s.active_fi & 0x3fff) + 1);
            s.active_fi = s.r[13];
            if self.schedule(s).is_err() {
                s.r[3] |= UE;
                s.bit_credit = 0;
                self.config.set_status_bits(0x8000);
                break;
            }
        }
    }
}
impl PciBusDevice for Ohci {
    fn config(&self) -> &PciConfig {
        &self.config
    }
    fn connect_dma(&self, dma: Arc<dyn PciDma>) {
        *self.dma.lock() = Some(dma);
    }
    fn connect_interrupt(&self, line: IrqLine) {
        *self.irq.lock() = line;
        self.drive_irq();
    }
    fn bar_read(&self, bar: usize, off: u32, width: usize) -> u32 {
        if bar != 0 || width != 4 || off & 3 != 0 || off > 0x58 {
            crate::dlog!(
                crate::devlog::LogModule::Usb,
                "ohci unsupported read at register offset {off:#x}, width {width}"
            );
            return 0;
        }
        self.sync_clock();
        let s = self.state.lock();
        let i = (off / 4) as usize;
        if i == 5 {
            s.r[4]
        } else if i == 21 || i == 22 {
            s.r[i] | if s.r[18] & 0x200 != 0 { PPS } else { 0 }
        } else {
            s.r[i]
        }
    }
    fn bar_write(&self, bar: usize, off: u32, val: u32, width: usize) {
        if bar != 0 || width != 4 || off & 3 != 0 || off > 0x58 {
            crate::dlog!(
                crate::devlog::LogModule::Usb,
                "ohci unsupported write at register offset {off:#x}, width {width}"
            );
            return;
        }
        self.sync_clock();
        // Taken before `pass`, as `sync_clock` orders them. Only a control
        // write can restart the frame phase.
        let mut anchor = (off == 4).then(|| self.clock.lock());
        let mut reset = None;
        {
            let _pass = self.pass.lock();
            let mut s = self.state.lock();
            match off / 4 {
                1 => {
                    let old = s.r[1];
                    s.r[1] = val & 0x7ff;
                    if val & 0xc0 == 0x80 && old & 0xc0 != 0x80 {
                        // OpenHCI 1.0a 6.2.1, 7.3.2, 7.3.3: entering
                        // UsbOperational loads HcFmRemaining from the frame
                        // interval and increments HcFmNumber, with no SOF at
                        // the load; the first follows one frame later. The
                        // free-running clock's phase therefore restarts here.
                        s.bit_credit = 0;
                        s.active_fi = s.r[13];
                        s.r[14] = (s.r[13] & 0x8000_0000) | (s.r[13] & 0x3fff);
                        let old_frame = s.r[15] as u16;
                        let frame = old_frame.wrapping_add(1);
                        s.r[15] = u32::from(frame);
                        if (old_frame ^ frame) & 0x8000 != 0 {
                            s.r[3] |= 32;
                        }
                        if let Some(Some(at)) = anchor.as_deref_mut() {
                            *at = self.now();
                        }
                    }
                    if val & 0xc0 == 0 && old & 0xc0 != 0 {
                        s.r[18] = 0x10000902;
                        for p in 0..2 {
                            s.power(p, false);
                        }
                    }
                    if val & 0xc0 == 0x40 || val & 0xc0 == 0xc0 {
                        crate::dlog!(
                            crate::devlog::LogModule::Usb,
                            "ohci suspend/resume timing unsupported at control register offset 0x04"
                        );
                    }
                }
                2 => {
                    if val & 1 != 0 {
                        s.soft_reset();
                    } else {
                        s.r[2] |= val & 6;
                        if val & 8 != 0 {
                            s.r[3] |= 1 << 30;
                        }
                    }
                }
                3 => s.r[3] &= !(val & EVENTS),
                4 => s.r[4] |= val & (EVENTS | 0x80000000),
                5 => s.r[4] &= !(val & (EVENTS | 0x80000000)),
                6 => s.r[6] = val & !255,
                7..=11 => s.r[(off / 4) as usize] = val & !15,
                13 => s.r[13] = val & 0xffff3fff,
                16 => s.r[16] = val & 0x3fff,
                17 => s.r[17] = val & 0xfff,
                18 => {
                    s.r[18] = (s.r[18] & !0xff001b00) | (val & 0xff001b00);
                    if val & 0x200 != 0 {
                        for p in 0..2 {
                            s.power(p, true);
                        }
                    }
                }
                19 => s.r[19] = val & 0x60006,
                20 => {
                    if val & (1 << 15) != 0 {
                        s.r[20] |= 1 << 15;
                    }
                    if val & (1 << 31) != 0 {
                        s.r[20] &= !(1 << 15);
                    }
                    if val & (1 << 17) != 0 {
                        s.r[20] &= !(1 << 17);
                    }
                    if s.r[18] & 0x200 == 0 {
                        for p in 0..2 {
                            if !s.individual(p) {
                                if val & 1 != 0 {
                                    s.power(p, false);
                                }
                                if val & (1 << 16) != 0 {
                                    s.power(p, true);
                                }
                            }
                        }
                    }
                }
                21 | 22 => {
                    let p = (off / 4 - 21) as usize;
                    let i = 21 + p;
                    s.r[i] &= !(val & CHANGES);
                    if val & 1 != 0 {
                        s.r[i] &= !2;
                    }
                    if val & 2 != 0 {
                        if s.r[i] & 1 != 0 && s.powered(p) {
                            s.r[i] |= 2;
                        } else {
                            s.r[i] |= 1 << 16;
                        }
                    }
                    if val & 12 != 0 {
                        crate::dlog!(
                            crate::devlog::LogModule::Usb,
                            "ohci port suspend/resume unsupported at port register offset {:#x}",
                            0x54 + p * 4
                        );
                    }
                    if val & 16 != 0 {
                        if s.r[i] & 1 != 0 && s.powered(p) {
                            s.r[i] |= 16;
                            s.r[i] &= !2;
                            s.reset_ms[p] = 10;
                            reset = s.ports[p].clone();
                        } else {
                            s.r[i] |= 1 << 16;
                        }
                    }
                    if s.individual(p) && s.r[18] & 0x200 == 0 {
                        if val & PPS != 0 {
                            s.power(p, true);
                        }
                        if val & 0x200 != 0 {
                            s.power(p, false);
                        }
                    }
                    if s.r[i] & CHANGES != 0 {
                        s.r[3] |= RHSC;
                    }
                }
                _ => {
                    crate::dlog!(
                        crate::devlog::LogModule::Usb,
                        "ohci write to read-only register offset {off:#x}"
                    );
                }
            }
        }
        drop(anchor);
        if let Some(f) = reset {
            f.reset();
        }
        self.drive_irq();
    }
}
impl Resettable for Ohci {
    fn power_on(&self) {
        let ports = {
            let _pass = self.pass.lock();
            let mut s = self.state.lock();
            let ports = s.ports.clone();
            let low_speed = s.low_speed;
            *s = State::new();
            s.ports = ports.clone();
            s.low_speed = low_speed;
            self.config.reset();
            ports
        };
        for f in ports.into_iter().flatten() {
            f.reset();
        }
        self.drive_irq();
    }
}
impl Device for Ohci {
    fn start(&self) {
        if self.running.swap(true, Ordering::AcqRel) {
            return;
        }
        let now = self.now();
        *self.clock.lock() = Some(now);
    }
    fn stop(&self) {
        self.sync_clock();
        self.running.store(false, Ordering::Release);
        *self.clock.lock() = None;
        drop(self.pass.lock());
    }
    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
    fn step(&self, _: u64) {}
    fn get_clock(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod register_tests {
    use super::*;
    use crate::pci::PciBusDevice;
    use crate::traits::Resettable;

    #[test]
    fn lucent_identity_and_reset_contract() {
        let hc = Ohci::new();
        assert_eq!(hc.config().read(0), 0x5802_11c1);
        assert_eq!(hc.config().read(2), 0x0c03_1010);
        assert_eq!(hc.config().read(11), 0xffff_ffff);
        assert_eq!(hc.config().read(1), 0x0200_0000);
        assert_eq!(hc.bar_read(0, 0, 4), 0x110);
        assert_eq!(hc.bar_read(0, 0x10, 4), 0);
        assert_eq!(hc.bar_read(0, 0x34, 4), 0x2edf);
        assert_eq!(hc.bar_read(0, 0x48, 4), 0x1000_0902);
        assert_eq!(hc.bar_read(0, 0x4c, 4), 0x0006_0000);
        for p in [0x54, 0x58] {
            assert_eq!(hc.bar_read(0, p, 4), 0);
        }
        hc.bar_write(0, 0x10, 0xffff_ffff, 4);
        hc.bar_write(0, 0x08, 1, 4);
        assert_eq!(hc.bar_read(0, 0x10, 4), 0);
        assert_eq!(hc.bar_read(0, 0x04, 4) & 0xc0, 0xc0);
        hc.power_on();
        assert_eq!(hc.bar_read(0, 0x04, 4), 0);
    }
}

#[cfg(test)]
#[path = "ohci/tests.rs"]
mod tests;
