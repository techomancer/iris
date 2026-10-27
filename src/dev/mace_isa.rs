//! MACE's ISA interrupt status and mask: the O2's second-level interrupt
//! controller. The ISA peripherals drive bits of its status, and it
//! cascades them into three of CRIME's inputs: serial and parallel (bits
//! 16..31) into 4, the miscellaneous group (8..15: the PS/2 ports, RTC, UST
//! timers) into 5, and audio (0..7) into 6.

use std::sync::{Arc, Mutex};

use crate::dev::mace_sdma::sdma;
use crate::irq::{InterruptController, IrqLine};
use crate::ip32::*;
use crate::snapshot::{get_field, hex_u64, toml_u64};
use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, Device, Resettable, Saveable, BUS_OK};

/// MACE's own interrupt status, which CRIME input 4 cascades from. NetBSD's
/// handler reads this register and matches each device's mask against it, so
/// raising CRIME alone is not enough -- the demultiplexer has to find
/// something.
pub const MACE_ISA_INT_STATUS: u32 = MACE_BASE + 0x0031_0010;
/// MACE ISA interrupt mask, next to [`MACE_ISA_INT_STATUS`].
pub const MACE_ISA_INT_MASK: u32 = MACE_BASE + 0x0031_0018;
/// CRIME input 4 is the ISA cascade; `com0` sits behind it.
pub const IP32_ISA_CRIME_INT: u32 = 4;
/// CRIME input for MACE ISA status bits 8..15: RTC, the PS/2 ports, UST timers.
pub const IP32_MISC_CRIME_INT: u32 = 5;
/// CRIME input for the audio bits (0..7).
pub const IP32_AUDIO_CRIME_INT: u32 = 6;
/// The PS/2 bits of the ISA status: 9 and 11 the ports' requests, 10 and 12
/// their poll flags.
pub const MACE_ISA_PS2_MASK: u64 = 0x1e00;
/// The bits NetBSD's `com0` attaches with, as it prints them:
/// "com0 at mace0 offset 0x390000 intr 4 intrmask 0x3f00000".
pub const MACE_ISA_SERIAL0_MASK: u64 = 0x03f0_0000;
/// The input a console UART driven by programmed I/O raises: not a status
/// bit of its own but port 0's whole group (see [`MaceIsa`]).
pub const ISA_SERIAL0_PIO: u32 = 64;
/// ISA ring base and reset register; bit 4 reads as the MACE revision.
pub const MACE_ISA_RING_BASE: u32 = MACE_BASE + 0x0031_0000;
pub const MACE_REV2_BIT: u64 = 0x10;

#[derive(Default)]
struct State {
    /// What software last wrote to the status register; it shows in the
    /// bits no device drives.
    guest: u64,
    mask: u64,
    /// The status bits devices drive, and their levels.
    driven: u64,
    levels: u64,
    /// The console UART's interrupt, while it is driven by PIO.
    serial0_pio: bool,
    /// The serial DMA engine: which ports it owns and their status bits,
    /// before the mask.
    serial_dma: ([bool; 2], u64),
}

impl State {
    /// The status register as software reads it.
    ///
    /// A port the guest drives by PIO keeps the behaviour the PROM and the
    /// BSDs boot with: all six of port 0's bits follow the 16550's
    /// interrupt, whatever the mask says (port 1 has never contributed). A
    /// port in DMA mode has the datasheet's bits, read through the mask, so
    /// the cascade follows them only while unmasked: IRIX acknowledges a
    /// level by masking it and leaving the work to a thread, and a line that
    /// ignored the mask would bring it straight back into the dispatcher,
    /// never letting that thread run.
    fn status(&self) -> u64 {
        let mut s = (self.guest & !self.driven) | (self.levels & self.driven);
        for p in 0..2 {
            let g = sdma::group(p);
            if self.serial_dma.0[p] {
                s = (s & !g) | (self.serial_dma.1 & g & self.mask);
            } else if p == 0 {
                s = if self.serial0_pio { s | g } else { s & !g };
            }
        }
        s
    }
}

pub struct MaceIsa {
    st: Mutex<State>,
    /// CRIME inputs 4 (serial), 5 (miscellaneous) and 6 (audio).
    crime: [IrqLine; 3],
}

impl MaceIsa {
    pub fn new(crime: Arc<dyn InterruptController>) -> Self {
        let line = |n| IrqLine::new(crime.clone(), n);
        Self {
            st: Mutex::new(State::default()),
            crime: [line(IP32_ISA_CRIME_INT), line(IP32_MISC_CRIME_INT), line(IP32_AUDIO_CRIME_INT)],
        }
    }

    pub fn contains(addr: u32) -> bool {
        matches!(addr & !7, MACE_ISA_INT_STATUS | MACE_ISA_INT_MASK)
    }

    pub fn status(&self) -> u64 {
        self.st.lock().unwrap().status()
    }

    pub fn mask(&self) -> u64 {
        self.st.lock().unwrap().mask
    }

    /// What software last wrote to the status register (the serial DMA
    /// engine's pair-request latches clear on it).
    pub fn status_written(&self) -> u64 {
        self.st.lock().unwrap().guest
    }

    /// The serial DMA engine's ports and their status bits, before the mask.
    pub fn set_serial_dma(&self, owned: [bool; 2], bits: u64) {
        let mut st = self.st.lock().unwrap();
        st.serial_dma = (owned, bits);
        self.drive(&st);
    }

    fn update(&self) {
        self.drive(&self.st.lock().unwrap());
    }

    /// Drive the cascade into CRIME from `st`, with `st` locked: two threads
    /// computing levels under the lock and driving after it could deliver
    /// them out of order and leave a stale one standing. The miscellaneous
    /// group reaches CRIME only once a device drives one of its bits: a
    /// serial-console O2 has no PS/2 behind it, and software's own writes
    /// there never interrupted.
    fn drive(&self, st: &State) {
        let (s, mask) = (st.status(), st.mask);
        self.crime[0].set(s & 0xffff_0000 != 0);
        if st.driven & 0xff00 != 0 {
            self.crime[1].set(s & mask & 0xff00 != 0);
        }
        self.crime[2].set(s & mask & 0xff != 0);
    }

    fn reg(&self, addr: u32) -> u64 {
        match addr & !7 {
            MACE_ISA_INT_STATUS => self.status(),
            _ => self.mask(),
        }
    }

    /// Every width writes the whole register, as MACE's other cells take
    /// them.
    fn put(&self, addr: u32, val: u64) {
        let mut st = self.st.lock().unwrap();
        match addr & !7 {
            MACE_ISA_INT_STATUS => st.guest = val,
            _ => st.mask = val,
        }
        self.drive(&st);
    }
}

/// A device's status bit (0..31) follows its level; [`ISA_SERIAL0_PIO`] is
/// the console UART's interrupt while it is driven by PIO.
impl InterruptController for MaceIsa {
    fn set_line(&self, line: u32, level: bool) {
        let mut st = self.st.lock().unwrap();
        if line == ISA_SERIAL0_PIO {
            st.serial0_pio = level;
        } else {
            let bit = 1u64 << line;
            st.driven |= bit;
            if level { st.levels |= bit } else { st.levels &= !bit }
        }
        self.drive(&st);
    }
}

/// Every width reads the register's low bits, as MACE's other cells do.
impl BusDevice for MaceIsa {
    fn read8(&self, addr: u64) -> BusRead8 { BusRead8::ok(self.reg(addr as u32) as u8) }
    fn read16(&self, addr: u64) -> BusRead16 { BusRead16::ok(self.reg(addr as u32) as u16) }
    fn read32(&self, addr: u64) -> BusRead32 { BusRead32::ok(self.reg(addr as u32) as u32) }
    fn read64(&self, addr: u64) -> BusRead64 { BusRead64::ok(self.reg(addr as u32)) }
    fn write8(&self, addr: u64, val: u8) -> u32 { self.put(addr as u32, val as u64); BUS_OK }
    fn write16(&self, addr: u64, val: u16) -> u32 { self.put(addr as u32, val as u64); BUS_OK }
    fn write32(&self, addr: u64, val: u32) -> u32 { self.put(addr as u32, val as u64); BUS_OK }
    fn write64(&self, addr: u64, val: u64) -> u32 { self.put(addr as u32, val); BUS_OK }
}

impl Device for MaceIsa {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("isa".to_string(), "MACE ISA interrupts: isa status".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write;
        if cmd != "isa" {
            return Err("Command not found".to_string());
        }
        match args.first().copied().unwrap_or("status") {
            "status" => {
                let st = self.st.lock().unwrap();
                writeln!(w, "MACE ISA: status {:#010x} mask {:#010x} (devices drive {:#010x}); serial 0 PIO {}, DMA owns {:?} bits {:#010x}",
                    st.status(), st.mask, st.driven, st.serial0_pio, st.serial_dma.0, st.serial_dma.1).ok();
                Ok(())
            }
            _ => Err("Usage: isa status".to_string()),
        }
    }
}

/// The registers, the levels devices drive, and the serial engine's view;
/// loading drives CRIME's cascade lines from them again.
impl Saveable for MaceIsa {
    fn save_state(&self) -> toml::Value {
        let st = self.st.lock().unwrap();
        let mut t = toml::map::Map::new();
        t.insert("status".into(), hex_u64(st.guest));
        t.insert("mask".into(), hex_u64(st.mask));
        t.insert("driven".into(), hex_u64(st.driven));
        t.insert("levels".into(), hex_u64(st.levels));
        t.insert("serial0_pio".into(), toml::Value::Boolean(st.serial0_pio));
        t.insert("serial_dma_owned".into(),
            toml::Value::Array(st.serial_dma.0.iter().map(|&b| toml::Value::Boolean(b)).collect()));
        t.insert("serial_dma_bits".into(), hex_u64(st.serial_dma.1));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        {
            let u = |k: &str| get_field(v, k).and_then(toml_u64).unwrap_or(0);
            let mut st = self.st.lock().unwrap();
            st.guest = u("status");
            st.mask = u("mask");
            st.driven = u("driven");
            st.levels = u("levels");
            st.serial0_pio = get_field(v, "serial0_pio").and_then(|b| b.as_bool()).unwrap_or(false);
            let mut owned = [false; 2];
            if let Some(toml::Value::Array(a)) = get_field(v, "serial_dma_owned") {
                for (o, b) in owned.iter_mut().zip(a) {
                    *o = b.as_bool().unwrap_or(false);
                }
            }
            st.serial_dma = (owned, u("serial_dma_bits"));
        }
        self.update();
        Ok(())
    }
}

impl Resettable for MaceIsa {
    fn power_on(&self) {
        *self.st.lock().unwrap() = State::default();
        self.update();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lines(Mutex<[bool; 8]>);
    impl InterruptController for Lines {
        fn set_line(&self, line: u32, level: bool) {
            self.0.lock().unwrap()[line as usize] = level;
        }
    }

    #[test]
    fn the_isa_status_cascades_into_crime_as_it_always_did() {
        let crime = Arc::new(Lines(Mutex::new([false; 8])));
        let isa = MaceIsa::new(crime.clone());
        let lv = |n: usize| crime.0.lock().unwrap()[n];
        // The console UART by PIO: port 0's group, whatever the mask.
        isa.set_line(ISA_SERIAL0_PIO, true);
        assert_eq!(isa.status() & MACE_ISA_SERIAL0_MASK, MACE_ISA_SERIAL0_MASK);
        assert!(lv(4));
        isa.set_line(ISA_SERIAL0_PIO, false);
        assert!(!lv(4));
        // Audio and PS/2 only through the mask.
        isa.set_line(2, true);
        assert!(!lv(6));
        isa.write64(MACE_ISA_INT_MASK as u64, 0xff | MACE_ISA_PS2_MASK);
        assert!(lv(6));
        isa.set_line(9, true);
        assert!(lv(5));
        // A port in DMA mode: its bits through the mask.
        isa.set_serial_dma([true, false], sdma::group(0));
        assert!(!lv(4), "masked");
        isa.write64(MACE_ISA_INT_MASK as u64, sdma::group(0));
        assert!(lv(4));
        // Software's writes do not stick on bits a device drives.
        isa.write64(MACE_ISA_INT_STATUS as u64, 0);
        assert_eq!(isa.status() & 1 << 2, 1 << 2);
    }

    #[test]
    fn save_load_round_trip() {
        let src = MaceIsa::new(Arc::new(Lines(Mutex::new([false; 8]))));
        src.set_line(2, true);
        src.set_line(9, true);
        src.set_line(ISA_SERIAL0_PIO, true);
        src.write64(MACE_ISA_INT_MASK as u64, 0xff | MACE_ISA_PS2_MASK);
        src.write64(MACE_ISA_INT_STATUS as u64, 0x4000_0000);
        src.set_serial_dma([false, true], sdma::group(1));
        let v1 = src.save_state();

        let crime = Arc::new(Lines(Mutex::new([false; 8])));
        let dst = MaceIsa::new(crime.clone());
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "MaceIsa save_state mismatch after load_state round-trip");
        let lv = |n: usize| crime.0.lock().unwrap()[n];
        assert!(lv(4) && lv(5) && lv(6), "the cascade is driven again after a load");
    }
}
