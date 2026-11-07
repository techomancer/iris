//! Philips PCF8584 I2C-bus controller, as the Origin 200's Hub wires it: two
//! registers eight bytes apart (data and its neighbours at A0 low, control on
//! write and status on read at A0 high) in front of a private untimed bus.
//!
//! A byte completes inside the register access that starts it and no time
//! passes. Several adapters sharing one clocked wire, as on an Origin 2000
//! module, are not modelled; `Pcf8584` owns its `I2cBus` and a shared wire
//! would replace that bus.
//!
//! Model choices, not measured behaviour: the control register reads back its
//! bits plus the live PIN with the serial interface off; ES1 and ES2 both set
//! with ESO clear selects the clock register; the general call is
//! acknowledged without raising AAS; a PIN write clears BER, LAB and AAS
//! without disturbing a transfer; the last unmatched address byte in
//! `0xe0`-`0xef` is copied into the data register when ESO is set and the
//! own address is nonzero, which is how a processor reads its arbitration
//! grant.
//!
//! The chip's slave half (the general call, and the programmed own address)
//! is handled inside this file, not as a target on the bus.
//!
//! Migration and snapshot state are not saved yet.
//!
//! Ported from hw/i2c/pcf8584.c in jamesbraid/qemu-sgi-origami.

use crate::dev::i2c::{BusBusy, I2cBus, Service};
use crate::devlog::LogModule;
use crate::dlog;
use crate::irq::IrqLine;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Resettable, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::sync::atomic::Ordering;

/// The two addresses the A0 pin selects, spaced as the Hub wires them.
pub const A0_STRIDE: u64 = 8;
pub const MMIO_SIZE: u64 = 2 * A0_STRIDE;

pub mod ctl {
    pub const PIN: u8 = 0x80;
    pub const ESO: u8 = 0x40;
    pub const ES1: u8 = 0x20;
    pub const ES2: u8 = 0x10;
    pub const ENI: u8 = 0x08;
    pub const STA: u8 = 0x04;
    pub const STO: u8 = 0x02;
    pub const ACK: u8 = 0x01;
}

pub mod sts {
    pub const PIN: u8 = 0x80;
    pub const STS: u8 = 0x20;
    pub const BER: u8 = 0x10;
    pub const LRB: u8 = 0x08;
    pub const AAS: u8 = 0x04;
    pub const LAB: u8 = 0x02;
    /// Set means the bus is not busy.
    pub const BB: u8 = 0x01;
}

enum Reg {
    OwnAddress,
    Clock,
    Vector,
    Data,
}

struct State {
    bus: I2cBus,
    /// S0', seven bits, right-justified.
    own_address: u8,
    /// S2, oscillator divider and bus speed. Stored only.
    clock: u8,
    /// S3, interrupt vector. Stored only.
    vector: u8,
    /// S0, the byte shifted in or out.
    data: u8,
    control: u8,
    status: u8,
    /// The core accepted the start and the open transfer must be ended.
    transfer: bool,
    /// Current direction is slave to master.
    recv: bool,
    /// A repeated start awaits its address byte.
    start_pending: bool,
    /// An idle data-register write awaits an immediate first start.
    start_address_staged: bool,
    /// Outbound address byte kept apart from the readable S0.
    start_address: u8,
    /// The matched seven-bit address while `transfer` is set.
    cur_addr7: u8,
    /// The chip's own slave half is part of the open transfer, and whether
    /// that is a directed hit on the own address rather than a general call.
    rx_current: bool,
    rx_directed: bool,
    /// Interrupt output level last reported to the line.
    irq_level: bool,
}

impl State {
    fn new() -> Self {
        Self {
            bus: I2cBus::new(),
            own_address: 0,
            clock: 0,
            vector: 0,
            data: 0,
            control: 0,
            // Out of reset the chip is idle and the bus is free.
            status: sts::PIN | sts::BB,
            transfer: false,
            recv: false,
            start_pending: false,
            start_address_staged: false,
            start_address: 0,
            cur_addr7: 0,
            rx_current: false,
            rx_directed: false,
            irq_level: false,
        }
    }

    /// Which register the A0-low address selects. ES2 keeps selecting even
    /// with the serial interface enabled; only ES1 stops meaning anything.
    fn selected(&self) -> Reg {
        let c = self.control;
        if c & ctl::ES2 != 0 && c & ctl::ES1 == 0 {
            Reg::Vector
        } else if c & ctl::ESO != 0 {
            if c & ctl::ES2 != 0 {
                Reg::Vector
            } else {
                Reg::Data
            }
        } else if c & ctl::ES1 != 0 {
            Reg::Clock
        } else {
            Reg::OwnAddress
        }
    }

    /// PIN is active low in the chip's naming: clear means completed.
    fn irq_wanted(&self) -> bool {
        self.control & ctl::ENI != 0 && self.status & sts::PIN == 0
    }

    /// Mark the byte transfer finished, recording whether the far end acked.
    fn complete(&mut self, nacked: bool) {
        self.status &= !sts::PIN;
        if nacked {
            self.status |= sts::LRB;
        } else {
            self.status &= !sts::LRB;
        }
    }

    fn bus_busy(&self) -> bool {
        self.bus.busy() || self.rx_current
    }

    fn bus_end(&mut self) {
        self.bus.end();
        self.rx_current = false;
        self.rx_directed = false;
    }

    /// Copy an address byte no target answered into S0, when the grant
    /// protocol asks for it.
    fn observe_address(&mut self, addr_byte: u8) {
        if self.control & ctl::ESO == 0
            || self.own_address == 0
            || addr_byte >> 1 == self.own_address
            || addr_byte == 0
        {
            return;
        }
        self.data = addr_byte;
    }

    /// Start a transfer on the bus, the way the I2C core does. Answers
    /// whether the address was acknowledged.
    fn bus_start(&mut self, addr7: u8, recv: bool) -> bool {
        let fresh = !self.bus_busy();
        if fresh {
            self.bus.scan(addr7);
            // The chip's slave half joins a general call, and answers its own
            // address when no target claimed it first.
            if self.bus.is_broadcast() {
                self.rx_current = true;
                self.rx_directed = false;
            } else if !self.bus.busy() && addr7 == self.own_address {
                self.rx_current = true;
                self.rx_directed = true;
            }
        }
        if !self.bus.busy() && !self.rx_current {
            if fresh && (0xe0..=0xef).contains(&((addr7 << 1) | recv as u8)) {
                self.observe_address((addr7 << 1) | recv as u8);
            }
            return false;
        }
        if !self.bus.start_selected(addr7, recv, fresh) {
            if fresh {
                self.rx_current = false;
                self.rx_directed = false;
            }
            return false;
        }
        // Only a directed hit on the programmed address raises AAS.
        if self.rx_directed {
            self.status |= sts::AAS;
        }
        true
    }

    fn bus_send(&mut self, byte: u8) -> bool {
        let ok = self.bus.send(byte);
        if self.rx_current {
            self.data = byte;
        }
        ok
    }

    fn bus_recv(&mut self) -> u8 {
        if self.bus.busy() {
            self.bus.recv()
        } else if self.rx_current && !self.bus.is_broadcast() {
            self.data
        } else {
            0xff
        }
    }

    fn stop(&mut self) {
        if self.transfer {
            self.bus_end();
        }
        self.transfer = false;
        self.recv = false;
        self.start_pending = false;
        self.start_address_staged = false;
        self.status |= sts::BB;
    }

    /// Issue a start with `addr` as the address-and-direction byte.
    fn start(&mut self, addr: u8) {
        let recv = addr & 1 != 0;
        let addr7 = addr >> 1;

        self.start_address_staged = false;
        // The core rescans only when nothing is selected, so an address change
        // must end the old selection here; ending unconditionally would tell a
        // target mid-command it was finished.
        if self.transfer && addr7 != self.cur_addr7 {
            self.bus_end();
            self.transfer = false;
        }
        self.cur_addr7 = addr7;
        self.recv = recv;
        self.status &= !sts::BB;

        let acked = self.bus_start(addr7, recv);
        if !acked && self.transfer {
            // A refused same-address repeated start is left open for the
            // caller to end; held open it would keep the bus busy for ever.
            self.bus_end();
        }
        self.transfer = acked;
        // Normal: firmware probes an invalid address in order to be refused.
        self.complete(!acked);
    }

    fn write_data(&mut self, val: u8) {
        if self.start_pending {
            self.start_pending = false;
            self.start(val);
            return;
        }
        if self.status & sts::BB != 0 {
            // Bus free: the address byte staged ahead of the control write.
            self.data = val;
            self.start_address = val;
            self.start_address_staged = true;
            return;
        }
        if self.recv {
            dlog!(LogModule::Bedrock, "pcf8584: data written during a receive transfer, ignored");
            return;
        }
        self.status |= sts::PIN;
        // A refused start leaves nothing selected; the refusal is remembered
        // here because sending over an empty selection would report success.
        let acked = self.transfer && self.bus_send(val);
        self.data = val;
        self.complete(!acked);
    }

    fn read_data(&mut self) -> u8 {
        let val = self.data;
        if self.transfer && self.recv {
            // The held byte goes out first, then the fetch: fetching first
            // would be off by one for the whole packet.
            if self.control & ctl::ACK == 0 {
                // ACK cleared means the driver wants this to be the last byte.
                self.bus.nack();
            }
            self.data = self.bus_recv();
            self.complete(false);
        }
        val
    }

    fn write_control(&mut self, val: u8) {
        self.control = val;

        if val & ctl::PIN != 0 {
            // PIN from the host is the idle request. It is also written alone
            // to select a configuration register, so it must not disturb a
            // transfer.
            self.status |= sts::PIN;
            self.status &= !(sts::BER | sts::LAB | sts::AAS);
        }
        if val & ctl::STA != 0 {
            if self.status & sts::BB == 0 {
                // Repeated start: the address byte has not been written yet.
                self.start_pending = true;
                self.status |= sts::PIN;
            } else {
                let addr = if self.start_address_staged {
                    self.start_address
                } else {
                    self.data
                };
                // First start: the address is already staged in the register.
                self.status |= sts::PIN;
                self.start(addr);
            }
            return;
        }
        if val & ctl::STO != 0 {
            self.stop();
            self.status |= sts::PIN;
        }
    }

    // The Hub decodes only bit 3 of the offset (the A0 pin): the bus window
    // bounds the offset, and bit 3 alone picks the register pair.
    fn read(&mut self, addr: u64) -> u8 {
        if addr & A0_STRIDE != 0 {
            // Write-only in the documentation with the serial interface off;
            // real parts answer with the control bits plus the PIN state.
            if self.control & ctl::ESO != 0 {
                return self.status;
            }
            return (self.control & !sts::PIN) | (self.status & sts::PIN);
        }
        match self.selected() {
            Reg::OwnAddress => self.own_address,
            Reg::Clock => self.clock,
            Reg::Vector => self.vector,
            Reg::Data => self.read_data(),
        }
    }

    fn write(&mut self, addr: u64, val: u8) {
        if addr & A0_STRIDE != 0 {
            self.write_control(val);
            return;
        }
        match self.selected() {
            // Seven bits, right-justified; stored as written so the
            // firmware's read-back liveness check sees what it wrote.
            Reg::OwnAddress => self.own_address = val & 0x7f,
            Reg::Clock => self.clock = val & 0x1f,
            Reg::Vector => self.vector = val,
            Reg::Data => self.write_data(val),
        }
    }

    /// Reset releases an untimed transfer before forgetting that it owns it.
    fn reset(&mut self) {
        if self.transfer && self.bus_busy() {
            self.bus_end();
        }
        // The line level survives so the access that follows reports a drop.
        let (bus, irq_level) = (std::mem::take(&mut self.bus), self.irq_level);
        *self = Self { bus, irq_level, ..Self::new() };
    }
}

pub struct Pcf8584 {
    state: Mutex<State>,
    busy: BusBusy,
    irq: Mutex<IrqLine>,
}

impl Pcf8584 {
    /// An adapter on a bus of its own. `busy` is the flag the bus's targets
    /// read; the adapter keeps it current.
    pub fn new(bus: I2cBus, busy: BusBusy) -> Self {
        let mut state = State::new();
        state.bus = bus;
        Self {
            state: Mutex::new(state),
            busy,
            irq: Mutex::new(IrqLine::unconnected()),
        }
    }

    /// Where the chip's interrupt output leads. The Origin 200 Hub does not
    /// connect it; software polls.
    pub fn set_irq(&self, irq: IrqLine) {
        *self.irq.lock() = irq;
    }

    /// Run `f` with the bus servicing around it: targets are let advance
    /// first (a grant they publish lands before the access), and what they
    /// ask afterwards (releasing the bus) is carried out once it is done.
    fn access<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        let (value, level) = {
            let mut st = self.state.lock();
            Self::service(&mut st);
            let value = f(&mut st);
            Self::service(&mut st);
            self.busy.store(st.bus_busy(), Ordering::Relaxed);
            let level = st.irq_wanted();
            let changed = level != st.irq_level;
            st.irq_level = level;
            (value, changed.then_some(level))
        };
        // Drive the line with no lock held, ours or the line slot's: the
        // controller behind it may call back into the Hub.
        if let Some(level) = level {
            let irq = self.irq.lock().clone();
            irq.set(level);
        }
        value
    }

    fn service(st: &mut State) {
        for i in 0..st.bus.targets().len() {
            let service: Service = st.bus.targets()[i].service();
            if service.release_bus && st.bus_busy() {
                st.bus_end();
            }
            if let Some(byte) = service.grant {
                // An address-only transaction nobody acknowledges; it cannot
                // open while a transfer holds the bus.
                if !st.bus_busy() {
                    st.bus_start(byte >> 1, byte & 1 != 0);
                    st.bus_end();
                }
            }
        }
    }

    pub fn read_register(&self, addr: u64) -> u8 {
        self.access(|st| st.read(addr))
    }

    pub fn write_register(&self, addr: u64, val: u8) {
        self.access(|st| st.write(addr, val));
    }
}

impl Resettable for Pcf8584 {
    fn power_on(&self) {
        self.access(|st| st.reset());
    }
}

/// Every width is one register access: the firmware reads the data register
/// with 8-byte loads, and splitting one would advance the receive pipeline
/// twice. The register value is the low byte of the access.
impl BusDevice for Pcf8584 {
    fn read8(&self, addr: u64) -> BusRead8 {
        if addr >= MMIO_SIZE {
            return BusRead8::err();
        }
        BusRead8::ok(self.read_register(addr))
    }

    fn write8(&self, addr: u64, val: u8) -> u32 {
        self.write_checked(addr, val)
    }

    fn read16(&self, addr: u64) -> BusRead16 {
        if addr >= MMIO_SIZE {
            return BusRead16::err();
        }
        BusRead16::ok(self.read_register(addr) as u16)
    }

    fn write16(&self, addr: u64, val: u16) -> u32 {
        self.write_checked(addr, val as u8)
    }

    fn read32(&self, addr: u64) -> BusRead32 {
        if addr >= MMIO_SIZE {
            return BusRead32::err();
        }
        BusRead32::ok(self.read_register(addr) as u32)
    }

    fn write32(&self, addr: u64, val: u32) -> u32 {
        self.write_checked(addr, val as u8)
    }

    fn read64(&self, addr: u64) -> BusRead64 {
        if addr >= MMIO_SIZE {
            return BusRead64::err();
        }
        BusRead64::ok(self.read_register(addr) as u64)
    }

    fn write64(&self, addr: u64, val: u64) -> u32 {
        self.write_checked(addr, val as u8)
    }
}

impl Pcf8584 {
    fn write_checked(&self, addr: u64, val: u8) -> u32 {
        if addr >= MMIO_SIZE {
            return BUS_ERR;
        }
        self.write_register(addr, val);
        BUS_OK
    }
}

#[cfg(test)]
mod tests;
