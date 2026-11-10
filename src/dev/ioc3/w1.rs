//! The IOC3's MicroLAN master: one register, one slot per write, on a
//! one-wire bus the board populates (the station-address part, and on some
//! boards the identity records).
//!
//! Linux binds `sgi_w1` to this register; OpenBSD bit-bangs it from `ioc.c`
//! to read the part number that says which board it is on. The engine is
//! routed onto its pin by bit 21 of pin control; with that clear the bus is
//! disconnected and reads high.
//!
//! Ported from hw/misc/sgi-ioc3.c in jamesbraid/qemu-sgi-origami.

use parking_lot::Mutex;

use crate::dev::w1;

/// Pin control bit that routes the MicroLAN engine onto its pin.
const PIN_CONTROL_MICROLAN: u32 = 1 << 21;

struct Wire {
    microlan: u32,
    bus: w1::Bus,
}

pub(super) struct W1 {
    st: Mutex<Wire>,
}

impl W1 {
    pub(super) fn new(bus: w1::Bus) -> W1 {
        W1 { st: Mutex::new(Wire { microlan: 0, bus }) }
    }

    /// Replace the wire the chip was built with.
    pub(super) fn set_bus(&self, bus: w1::Bus) {
        self.st.lock().bus = bus;
    }

    pub(super) fn read(&self) -> u32 {
        self.st.lock().microlan
    }

    /// One write is one slot, finished inside the write.
    pub(super) fn write(&self, val: u32, pin_control: u32) {
        let mut st = self.st.lock();
        let level = if pin_control & PIN_CONTROL_MICROLAN == 0 {
            true
        } else {
            let (pulse, sample) = w1::slot::fields(val as u64);
            w1::slot::slot(&mut st.bus, pulse, sample)
        };
        st.microlan = (val & !3) | 2 | level as u32;
    }

    pub(super) fn reset(&self) {
        self.st.lock().microlan = 0;
    }

    #[cfg(test)]
    pub(super) fn roms(&self) -> Vec<[u8; 8]> {
        self.st.lock().bus.roms()
    }

    pub(super) fn save_state(&self) -> (u32, toml::Value) {
        let st = self.st.lock();
        (st.microlan, st.bus.save_state())
    }

    pub(super) fn load_state(&self, microlan: u32, wire: &toml::Value) -> Result<(), String> {
        let mut st = self.st.lock();
        match wire.as_bool() {
            // A snapshot from before the wire was built eagerly: none had
            // been built, so the one this chip was given stands.
            Some(false) => {}
            Some(true) => return Err("wire: not a bus".into()),
            None => st.bus.load_state(wire).map_err(|e| format!("MicroLAN: {e}"))?,
        }
        st.microlan = microlan;
        Ok(())
    }
}
