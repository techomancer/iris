//! MACE's miscellaneous control register at 0x1f310008: the flash write
//! enable, the identity line to the board's DS2502 and the two front LEDs.
//! The line is two bits, not a slot register: the master clears RELEASE to
//! pull it low and sets it to let go, and SAMPLE reads it back. What a pulse
//! means depends on how long the line stayed low, so the device times it on
//! the guest clock and hands each finished reset or bit slot to the shared
//! [`w1::Bus`].
//!
//! The QEMU model keeps the DS2502 datasheet's absolute windows (reset
//! 480-960 us, zero 60-120, one under 24, 480 high before a reset). The
//! PROM's delay loops are counted in emulated instructions, so a fast
//! emulator (jitv2, a fast host) runs them short, and a pulse outside a
//! window was ignored, leaving the Ethernet address reading as all ff. So
//! the classes are only ordered: a low of at least 300 us is a reset, one of
//! at least 30 a zero, anything shorter a one. The widths the PROM produces,
//! about 500, 90 and 8, land in the same classes.
//!
//! Ported from hw/mips/sgi/mace-id.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::dev::ng1::rex3::Rex3;

use crate::dev::w1::{self, sgi};
use crate::snapshot::{get_field, toml_u64};

pub const FLASH_ENABLE: u8 = 1;
const PASSWORD_CLEAR: u8 = 1 << 1;
/// The master lets the line float high.
pub const RELEASE: u8 = 1 << 2;
/// The line as it reads back.
pub const SAMPLE: u8 = 1 << 3;
const RED_LED: u8 = 1 << 4;
const GREEN_LED: u8 = 1 << 5;
const WRITABLE: u8 = FLASH_ENABLE | RELEASE | RED_LED | GREEN_LED;

/// Shortest low, in microseconds of guest time, that is a reset, and the
/// shortest that is a zero.
const RESET_MIN: u64 = 300;
const ZERO_MIN: u64 = 30;
/// After a reset releases, the part answers presence from this long after
/// until this long after.
const PRESENCE_DELAY: u64 = 30;
const PRESENCE_END: u64 = 150;

/// The station address the board carries when none is configured.
pub const DEFAULT_MAC: [u8; 6] = [0x08, 0x00, 0x69, 0x12, 0x34, 0x56];

/// Where the PROM's start-up handshake stands: its first low, however long,
/// is a reset when the line is first released.
#[derive(Default)]
enum Start {
    /// Waiting for the master to pull the line down (`None`), or since when
    /// it has held it down.
    #[default]
    Pending,
    Low(u64),
    Done,
}

#[derive(Default)]
struct State {
    bus: w1::Bus,
    low_start: u64,
    start: Start,
    /// The window in which the part answers presence, once a reset has
    /// been seen; none before.
    presence: Option<(u64, u64)>,
    released: bool,
    flash_write_enabled: bool,
    led_control: u8,
    /// The part is holding the line low for a zero it is sending.
    slot_pull: bool,
}

pub struct MaceId {
    st: Mutex<State>,
    /// The LEDs as the status bar shows them: `crate::status::hb::LED_RED` and
    /// `HB_LED_GREEN`.
    pub heartbeat: Arc<AtomicU64>,
}

impl State {
    /// Reset changes the drive outputs, not the connected board input.
    fn reset(&mut self) {
        let bus = std::mem::take(&mut self.bus);
        *self = State { bus, led_control: GREEN_LED, ..Default::default() };
        self.bus.reset();
    }

    fn presence_at(&self, now: u64) -> bool {
        self.presence.is_some_and(|(from, to)| now >= from && now < to)
    }

    fn value(&self, now: u64) -> u8 {
        let level = self.released && !self.presence_at(now) && !self.slot_pull;
        self.led_control
            | if self.flash_write_enabled { FLASH_ENABLE } else { 0 }
            | if self.released { RELEASE } else { 0 }
            | if level { SAMPLE } else { 0 }
    }

    fn reset_pulse(&mut self, now: u64) {
        self.bus.reset();
        self.slot_pull = false;
        self.presence = Some((now + PRESENCE_DELAY, now + PRESENCE_END));
    }

    fn write(&mut self, now: u64, value: u8) {
        let release = value & RELEASE != 0;
        self.led_control = value & (RED_LED | GREEN_LED);
        self.flash_write_enabled = value & FLASH_ENABLE != 0;
        if !release && matches!(self.start, Start::Pending) {
            self.start = Start::Low(now);
        }
        if release == self.released {
            return;
        }
        self.released = release;
        if !release {
            // The slot ends where the next one starts.
            self.slot_pull = false;
            self.low_start = now;
            return;
        }
        if let Start::Low(since) = self.start {
            self.start = Start::Done;
            if now.saturating_sub(since) >= RESET_MIN {
                self.reset_pulse(now);
            }
            return;
        }
        let low = now.saturating_sub(self.low_start);
        if low >= RESET_MIN {
            self.reset_pulse(now);
            return;
        }
        let Some((_, end)) = self.presence else { return };
        if now < end {
            return;
        }
        if low >= ZERO_MIN {
            self.bus.slot(false);
        } else {
            // A one is also a read slot: a zero from the part holds the line
            // low until the next slot starts.
            self.slot_pull = !self.bus.slot(true);
        }
    }
}

impl MaceId {
    /// The board's Ethernet-address chip for station address `mac`.
    pub fn new(mac: [u8; 6]) -> Self {
        let mut st = State::default();
        st.bus.add(Box::new(sgi::o2_ethernet_address_part(mac)));
        st.reset();
        let id = Self { st: Mutex::new(st), heartbeat: Arc::new(AtomicU64::new(0)) };
        id.show_leds(GREEN_LED);
        id
    }

    /// The register at guest time `now` (microseconds). The password-clear
    /// input is not wired, so bit 1 reads zero.
    pub fn read(&self, now: u64) -> u8 {
        self.st.lock().unwrap().value(now) & !PASSWORD_CLEAR
    }

    pub fn write(&self, now: u64, value: u8) {
        let led = {
            let mut st = self.st.lock().unwrap();
            st.write(now, value & WRITABLE);
            st.led_control
        };
        self.show_leds(led);
    }

    /// An LED lights when its bit is clear: writing zero enables both
    /// outputs, and the register resets with only the green one disabled.
    fn show_leds(&self, bits: u8) {
        for (bit, led) in [(RED_LED, crate::status::hb::LED_RED), (GREEN_LED, crate::status::hb::LED_GREEN)] {
            if bits & bit == 0 {
                self.heartbeat.fetch_or(led, Ordering::Relaxed);
            } else {
                self.heartbeat.fetch_and(!led, Ordering::Relaxed);
            }
        }
    }
}

impl crate::traits::Device for MaceId {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool {
        true
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("onewire".to_string(), "MACE identity line: onewire status".to_string())]
    }

    fn execute_command(&self, cmd: &str, _args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write;
        if cmd != "onewire" {
            return Err("Command not found".to_string());
        }
        let st = self.st.lock().unwrap();
        let rom: String = st.bus.roms()[0].iter().map(|b| format!("{b:02x}")).collect();
        writeln!(w, "ID chip ROM {rom}, register {:#04x}", st.value(0)).ok();
        Ok(())
    }
}

/// The register's state only, as the QEMU model keeps it. The part comes
/// from the configured MAC, and a restored guest is past the PROM's start-up
/// handshake.
impl crate::traits::Saveable for MaceId {
    fn save_state(&self) -> toml::Value {
        let st = self.st.lock().unwrap();
        let mut t = toml::map::Map::new();
        t.insert("register".into(), toml::Value::Integer((st.value(0) & !SAMPLE) as i64));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let reg = get_field(v, "register").and_then(toml_u64).unwrap_or(0) as u8;
        {
            let mut st = self.st.lock().unwrap();
            st.reset();
            st.start = Start::Done;
            st.released = reg & RELEASE != 0;
            st.flash_write_enabled = reg & FLASH_ENABLE != 0;
            st.led_control = reg & (RED_LED | GREEN_LED);
        }
        self.show_leds(reg & (RED_LED | GREEN_LED));
        Ok(())
    }
}

impl crate::traits::Resettable for MaceId {
    fn power_on(&self) {
        self.st.lock().unwrap().reset();
        self.show_leds(GREEN_LED);
    }
}
