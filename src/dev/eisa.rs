//! The Indigo2's EISA bus, present and empty.
//!
//! Every Indigo2 (IP22 fullhouse and IP28) has an EISA Interface Unit (EIU)
//! and an Intel 82357 ISP (interrupt controllers, timers, DMA, NMI control)
//! in the low physical window `0x00080000..0x000A0000`:
//!
//! - `0x00080000 + port`: EISA I/O space. Ports below `0x1000` are the
//!   system board (the ISP); slot `n` owns `n << 12` up, with its ID bytes at
//!   `+0xC80`.
//! - `0x0009FFC0..0x0009FFD0`: the EIU's mode, status, preempt and quiet
//!   registers; `0x00090004` is the interrupt acknowledge.
//!
//! IRIX decides whether EISA exists by probing the EIU mode register, then
//! programs the ISP's interrupt controllers and reads each slot's ID. The PROM
//! probes the ISP's timer to turn EISA refresh off and to drive the keyboard
//! bell. No EISA card is modelled, so this answers just enough for those
//! probes to find a bus with every slot empty:
//!
//! - The EIU registers read back what was written.
//! - System-board ports are a byte latch: a read returns the last value
//!   written there, which is what the ISP's mask registers do.
//! - Slot space floats high: reads return all ones, which IRIX takes as "no
//!   EISA board found".
//! - Nothing raises the EISA interrupt; the acknowledge cycle returns 7, the
//!   8259's spurious vector.
//!
//! The Indy has no EISA: its VINO sits at `0x00080000` and the EIU probe
//! bus-errors, as on hardware.

use parking_lot::Mutex;

use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK};

/// Where the EISA window starts on the Indigo2's physical bus.
pub const EISA_BASE: u32 = 0x0008_0000;
/// One past its end.
pub const EISA_END: u32 = 0x000A_0000;

/// I/O ports below this belong to the system board (the ISP); slots above.
const SYSTEM_PORTS: usize = 0x1000;
/// The EIU registers (mode, status, preempt, quiet), from the window base.
const EIU_REGS: u64 = 0x1_FFC0;
const EIU_REG_COUNT: usize = 4;
/// The EIU's interrupt acknowledge address, from the window base.
const EIU_INTRPT_ACK: u64 = 0x1_0004;
/// The vector an 8259 hands out when nothing is pending.
const SPURIOUS_VECTOR: u8 = 7;

struct State {
    system_ports: Box<[u8; SYSTEM_PORTS]>,
    eiu: [u32; EIU_REG_COUNT],
}

pub struct Eisa {
    state: Mutex<State>,
}

impl Default for Eisa {
    fn default() -> Self {
        Self::new()
    }
}

impl Eisa {
    pub fn new() -> Self {
        Self { state: Mutex::new(State { system_ports: Box::new([0; SYSTEM_PORTS]), eiu: [0; EIU_REG_COUNT] }) }
    }

    fn offset(addr: u64) -> u64 {
        addr.wrapping_sub(EISA_BASE as u64)
    }

    fn eiu_index(off: u64) -> Option<usize> {
        let i = off.checked_sub(EIU_REGS)? / 4;
        (i < EIU_REG_COUNT as u64).then_some(i as usize)
    }

    fn byte(&self, off: u64) -> u8 {
        if off == EIU_INTRPT_ACK {
            return SPURIOUS_VECTOR;
        }
        if let Some(i) = Self::eiu_index(off) {
            // Big-endian lanes: byte 0 is the register's top byte.
            return (self.state.lock().eiu[i] >> ((3 - (off & 3)) * 8)) as u8;
        }
        match off as usize {
            p if p < SYSTEM_PORTS => self.state.lock().system_ports[p],
            // Empty slots (and the rest of the window) float high.
            _ => 0xFF,
        }
    }

    fn set_byte(&self, off: u64, v: u8) {
        if let Some(i) = Self::eiu_index(off) {
            let shift = (3 - (off & 3)) * 8;
            let r = &mut self.state.lock().eiu[i];
            *r = (*r & !(0xFF << shift)) | (u32::from(v) << shift);
        } else if (off as usize) < SYSTEM_PORTS {
            self.state.lock().system_ports[off as usize] = v;
        }
    }

    fn read_be(&self, addr: u64, n: u64) -> u64 {
        let off = Self::offset(addr);
        (0..n).fold(0, |acc, i| (acc << 8) | u64::from(self.byte(off + i)))
    }

    fn write_be(&self, addr: u64, n: u64, v: u64) {
        let off = Self::offset(addr);
        for i in 0..n {
            self.set_byte(off + i, (v >> ((n - 1 - i) * 8)) as u8);
        }
    }
}

impl BusDevice for Eisa {
    fn read8(&self, addr: u64) -> BusRead8 { BusRead8::ok(self.read_be(addr, 1) as u8) }
    fn write8(&self, addr: u64, v: u8) -> u32 { self.write_be(addr, 1, v.into()); BUS_OK }
    fn read16(&self, addr: u64) -> BusRead16 { BusRead16::ok(self.read_be(addr, 2) as u16) }
    fn write16(&self, addr: u64, v: u16) -> u32 { self.write_be(addr, 2, v.into()); BUS_OK }
    fn read32(&self, addr: u64) -> BusRead32 { BusRead32::ok(self.read_be(addr, 4) as u32) }
    fn write32(&self, addr: u64, v: u32) -> u32 { self.write_be(addr, 4, v.into()); BUS_OK }
    fn read64(&self, addr: u64) -> BusRead64 { BusRead64::ok(self.read_be(addr, 8)) }
    fn write64(&self, addr: u64, v: u64) -> u32 { self.write_be(addr, 8, v); BUS_OK }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EIU_MODE: u64 = 0x0009_FFC0;

    #[test]
    fn eiu_registers_read_back() {
        let e = Eisa::new();
        assert_eq!(e.write32(EIU_MODE, 0x40f3_c07f), BUS_OK);
        assert_eq!(e.read32(EIU_MODE).data, 0x40f3_c07f);
        assert_eq!(e.read8(EIU_MODE).data, 0x40);
        assert_eq!(e.read32(0x0009_FFC8).data, 0);
    }

    #[test]
    fn system_ports_latch_and_slots_are_empty() {
        let e = Eisa::new();
        // The ISP's master interrupt mask (port 0x21).
        e.write8(0x0008_0021, 0xFB);
        assert_eq!(e.read8(0x0008_0021).data, 0xFB);
        // Every slot's compressed ID reads all ones: no board.
        for slot in 1..=4u64 {
            assert_eq!(e.read32(EISA_BASE as u64 + (slot << 12) + 0xC80).data, 0xFFFF_FFFF);
        }
        assert_eq!(e.read8(0x0009_0004).data, SPURIOUS_VECTOR);
    }
}
