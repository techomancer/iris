/// CDMC — SGI Camera Digital Multistandard Codec
///
/// The camera controller chip on the IndyCam module.  Connected to VINO's
/// master I2C bus alongside the SAA7191 (DMSD).  IRIX's `vlcam` and
/// `videopanel` clients write CDMC registers to adjust brightness, hue,
/// saturation, gamma, etc.
///
/// Fake device: register storage + I2C state machine.  `apply_uyvy_field` applies
/// gain/balance/saturation/shutter exposure to host camera pixels via
/// `CdmcAdjustedSource` in `video_source.rs`.
///
/// I2C address: 0x56 write / 0x57 read (7-bit address 0x2B).
///
/// Note: a previous comment here said "0xAE/0xAF" based on a stale read of
/// the IRIX 6.5 `indycam.h`. The actual I2C address bytes the IRIX 5.3
/// vino driver puts on the bus are 0x56 (write) / 0x57 (read) — verified
/// by tracing `vino.write_reg(I2C_DATA, …)` while running `videod` +
/// `vidtomem`, and by literal scan of `vino_i2c.o` / `vino_input.o` /
/// `vino_ctrls.o` (the 0x56 / 0x57 / 0x2b immediates appear dozens of
/// times; 0xAE appears exactly once in `vino_input.o`).
///
/// References:
///   IRIX 5.3 vino driver `.text` literals (vino_i2c.o, vino_input.o)
///   IRIX 6.5 indycam.h (kernel header) for register layout

use parking_lot::Mutex;

use crate::devlog::LogModule;

// ─── Register subaddresses ────────────────────────────────────────────────────
//
// Layout and power-on values follow the IndyCam ("Guinness camera") register
// map as documented by the Linux `indycam` driver (drivers/media/video/
// indycam.h, Ladislav Michl / Mikael Nousiainen) — same silicon the IRIX vino
// driver talks to. Low addresses are the exposure/colour controls, VERSION
// sits at 0x0E, and RESET at 0x0F.

pub mod reg {
    pub const CONTROL:     u8 = 0x00; // rw  AGC / AWB enables + EVNFLD status
    pub const SHUTTER:     u8 = 0x01; // rw  Shutter speed
    pub const GAIN:        u8 = 0x02; // rw  Analog gain
    pub const BRIGHTNESS:  u8 = 0x03; // r   Measured scene brightness
    pub const RED_BAL:     u8 = 0x04; // rw  Red balance
    pub const BLUE_BAL:    u8 = 0x05; // rw  Blue balance
    pub const RED_SAT:     u8 = 0x06; // rw  Red saturation
    pub const BLUE_SAT:    u8 = 0x07; // rw  Blue saturation
    pub const GAMMA:       u8 = 0x08; // rw  Gamma

    // 0x09–0x0D unused (silently ignored)

    pub const VERSION:     u8 = 0x0E; // r   Camera model/version byte
    //
    // The IRIX 5.3 vino driver's `vinoCameraAttached()` reads this byte
    // and considers the camera "present" iff the value is exactly 0x10.
    // Disassembly of vinoCameraAttached in vino_main.o:
    //   ...
    //   addiu $a1, $zero, 0x56     ; CDMC I2C write addr
    //   addiu $a2, $zero, 0x0e     ; subaddr
    //   jal   vinoI2cReadReg
    //   ...
    //   addiu $at, $zero, 0x10
    //   bnel  $v0, $at, not_attached
    //
    // Without this byte present at the expected value, the kernel prints
    // "IndyCam not attached. [HELP=VINONOCAMERA_WARN]" and refuses to
    // start frame capture even though videod / vlinfo have already
    // enumerated the device.

    pub const RESET:       u8 = 0x0F; // w   Write triggers a device reset

    // Total register slots — 0x00..=0x0F inclusive = 16.
    pub const COUNT: usize = 0x10;

    // ── CONTROL bits ──
    pub const CONTROL_AGCENA: u8 = 1 << 0; // automatic gain control
    pub const CONTROL_AWBCTL: u8 = 1 << 1; // automatic white balance
    pub const CONTROL_EVNFLD: u8 = 1 << 4; // read-only: current field is even

    // ── Power-on values ──
    //
    // `apply_uyvy_field` treats exactly this set as "no adjustment", so a
    // guest that leaves the camera at its defaults gets the host frame
    // through unaltered.
    pub const CONTROL_DEFAULT:    u8 = CONTROL_AGCENA;
    pub const SHUTTER_DEFAULT:    u8 = 0xFF;
    pub const GAIN_DEFAULT:       u8 = 0x80;
    pub const BRIGHTNESS_DEFAULT: u8 = 0x80;
    pub const RED_BAL_DEFAULT:    u8 = 0x18;
    pub const BLUE_BAL_DEFAULT:   u8 = 0xA4;
    pub const RED_SAT_DEFAULT:    u8 = 0x80;
    pub const BLUE_SAT_DEFAULT:   u8 = 0xC0;
    pub const GAMMA_DEFAULT:      u8 = 0x80;

    /// Value returned at VERSION (0x0E). Must be exactly 0x10 (IndyCam v1.0)
    /// for the IRIX 5.3 vino driver's `vinoCameraAttached()` check to pass.
    pub const VERSION_VAL: u8 = 0x10;
}

// ─── I2C state machine ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum I2cState {
    Idle,
    SubaddrWrite,
    SubaddrRead,
    DataWrite,
    DataRead,
}

struct CdmcState {
    regs:           [u8; reg::COUNT],

    i2c_write_addr: u8, // 0x56 (= 7-bit 0x2B << 1, R/W=0)
    i2c_read_addr:  u8, // 0x57
    i2c_subaddr:    u8,
    i2c_state:      I2cState,
}

impl Default for CdmcState {
    fn default() -> Self {
        let mut regs = [0u8; reg::COUNT];
        regs[reg::CONTROL as usize]    = reg::CONTROL_DEFAULT;
        regs[reg::SHUTTER as usize]    = reg::SHUTTER_DEFAULT;
        regs[reg::GAIN as usize]       = reg::GAIN_DEFAULT;
        regs[reg::BRIGHTNESS as usize] = reg::BRIGHTNESS_DEFAULT;
        regs[reg::RED_BAL as usize]    = reg::RED_BAL_DEFAULT;
        regs[reg::BLUE_BAL as usize]   = reg::BLUE_BAL_DEFAULT;
        regs[reg::RED_SAT as usize]    = reg::RED_SAT_DEFAULT;
        regs[reg::BLUE_SAT as usize]   = reg::BLUE_SAT_DEFAULT;
        regs[reg::GAMMA as usize]      = reg::GAMMA_DEFAULT;
        regs[reg::VERSION as usize]    = reg::VERSION_VAL;
        Self {
            regs,
            i2c_write_addr: 0x56,
            i2c_read_addr:  0x57,
            i2c_subaddr:    0x00,
            i2c_state:      I2cState::Idle,
        }
    }
}

// ─── Public handle ────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Cdmc {
    state: std::sync::Arc<Mutex<CdmcState>>,
}

impl Cdmc {
    pub fn new() -> Self {
        Self { state: std::sync::Arc::new(Mutex::new(CdmcState::default())) }
    }

    pub fn power_on(&self) {
        *self.state.lock() = CdmcState::default();
    }

    /// True when the device has accepted an address byte and is mid-transfer.
    /// VINO uses this to route subsequent bytes to the correct I2C target.
    pub fn is_active(&self) -> bool {
        self.state.lock().i2c_state != I2cState::Idle
    }

    // ── I2C interface ─────────────────────────────────────────────────────

    pub fn i2c_write(&self, data: u8) {
        let mut st = self.state.lock();
        // REPEATED START: a slave-address byte arriving in any non-Idle
        // state means the master issued repeated-start and is re-addressing
        // the bus. Recognise our own write/read addresses and transition
        // appropriately, preserving subaddr if we already had one set.
        if st.i2c_state != I2cState::Idle {
            if data == st.i2c_write_addr {
                st.i2c_state = I2cState::SubaddrWrite;
                return;
            }
            if data == st.i2c_read_addr {
                // Caller set subaddr earlier (typical IndyCam probe:
                // write 0x56, subaddr 0x00, then repeated start + 0x57 to
                // read VERSION). Jump straight to DataRead — i2c_read()
                // returns regs[subaddr] then auto-increments.
                st.i2c_state = I2cState::DataRead;
                return;
            }
        }
        match st.i2c_state {
            I2cState::Idle => {
                if data == st.i2c_write_addr {
                    st.i2c_state = I2cState::SubaddrWrite;
                } else if data == st.i2c_read_addr {
                    // Standalone read without prior subaddr write: use the
                    // current subaddr (zero on reset, or whatever the last
                    // transaction left it at). IRIX's vino driver always
                    // pairs subaddr-write + repeated-start + read, so this
                    // path mostly matters for fall-throughs.
                    st.i2c_state = I2cState::DataRead;
                }
                // Address didn't match — silently stay idle.  Another device
                // on the shared bus may pick it up.
            }
            I2cState::SubaddrWrite => {
                st.i2c_subaddr = data;
                st.i2c_state   = I2cState::DataWrite;
            }
            I2cState::SubaddrRead => {
                st.i2c_subaddr = data;
                st.i2c_state   = I2cState::DataRead;
            }
            I2cState::DataWrite => {
                Self::reg_w(&mut st, data);
                st.i2c_subaddr = st.i2c_subaddr.wrapping_add(1) % reg::COUNT as u8;
            }
            I2cState::DataRead => {
                dlog_dev!(LogModule::Vino, "CDMC: I2C expected read but got write, returning to idle");
                st.i2c_state = I2cState::Idle;
            }
        }
    }

    pub fn i2c_read(&self) -> u8 {
        let mut st = self.state.lock();
        if st.i2c_state != I2cState::DataRead {
            dlog_dev!(LogModule::Vino, "CDMC: i2c_read called in state {:?}, returning to idle", st.i2c_state);
            st.i2c_state = I2cState::Idle;
            return 0;
        }
        let sub = st.i2c_subaddr as usize;
        let val = if sub < reg::COUNT { st.regs[sub] } else { 0 };
        st.i2c_subaddr = st.i2c_subaddr.wrapping_add(1) % reg::COUNT as u8;
        val
    }

    pub fn i2c_stop(&self) {
        self.state.lock().i2c_state = I2cState::Idle;
    }

    /// Snapshot of image-control registers for the video pixel pipeline.
    pub fn regs_copy(&self) -> [u8; reg::COUNT] {
        self.state.lock().regs
    }

    /// Apply CDMC exposure / colour balance to a packed UYVY field in-place.
    ///
    /// The host camera already delivers an exposed, white-balanced frame, so
    /// the transfer functions here are anchored at the IndyCam's power-on
    /// register values: with the defaults in place every term is unity and the
    /// field passes through untouched. A guest that moves a control away from
    /// its default (the video panel's sliders) gets a proportional, bounded
    /// change in that direction — the response curve is an approximation of
    /// the real camera's, not a measured match.
    pub fn apply_uyvy_field(pixels: &mut [u8], regs: &[u8; reg::COUNT]) {
        /// Deviation of a control from its default, as a multiplier centred on
        /// 1.0 and bounded to roughly 0.5×..1.5× across the full 0x00..0xFF range.
        fn factor(val: u8, default: u8) -> f32 {
            1.0 + (val as f32 - default as f32) / 255.0
        }

        let control = regs[reg::CONTROL as usize];

        // With automatic gain control enabled the camera runs its own exposure
        // loop and the GAIN/SHUTTER registers don't drive the picture, which
        // matches the host camera doing its own auto-exposure.
        let luma_scale = if control & reg::CONTROL_AGCENA != 0 {
            1.0
        } else {
            factor(regs[reg::GAIN as usize], reg::GAIN_DEFAULT)
                * factor(regs[reg::SHUTTER as usize], reg::SHUTTER_DEFAULT)
        };

        // Balance shifts a colour-difference channel's white point; saturation
        // scales its amplitude about neutral. Automatic white balance takes the
        // balance registers out of the picture the same way AGC does for gain.
        let awb = control & reg::CONTROL_AWBCTL != 0;
        let red_shift = if awb { 0.0 } else {
            (regs[reg::RED_BAL as usize] as f32 - reg::RED_BAL_DEFAULT as f32) / 4.0
        };
        let blue_shift = if awb { 0.0 } else {
            (regs[reg::BLUE_BAL as usize] as f32 - reg::BLUE_BAL_DEFAULT as f32) / 4.0
        };
        let red_scale = factor(regs[reg::RED_SAT as usize], reg::RED_SAT_DEFAULT);
        let blue_scale = factor(regs[reg::BLUE_SAT as usize], reg::BLUE_SAT_DEFAULT);

        for c in pixels.chunks_mut(4) {
            let u = c[0] as f32;
            let y = c[1] as f32;
            let v = c[2] as f32;
            let y2 = c[3] as f32;

            let y_adj = ((y - 16.0) * luma_scale + 16.0).clamp(0.0, 235.0);
            let y2_adj = ((y2 - 16.0) * luma_scale + 16.0).clamp(0.0, 235.0);
            let u_adj = (128.0 + (u - 128.0) * blue_scale + blue_shift).clamp(0.0, 255.0);
            let v_adj = (128.0 + (v - 128.0) * red_scale + red_shift).clamp(0.0, 255.0);

            c[0] = u_adj as u8;
            c[1] = y_adj as u8;
            c[2] = v_adj as u8;
            c[3] = y2_adj as u8;
        }
    }

    // ── Register write ────────────────────────────────────────────────────

    fn reg_w(st: &mut CdmcState, data: u8) {
        let sub = st.i2c_subaddr;
        match sub {
            // A write to RESET returns every control to its power-on value.
            // The I2C transaction itself carries on.
            reg::RESET => st.regs = CdmcState::default().regs,
            // BRIGHTNESS reports the measured scene level and VERSION identifies
            // the camera; both are read-only.
            reg::BRIGHTNESS | reg::VERSION => {}
            _ if (sub as usize) < reg::COUNT => st.regs[sub as usize] = data,
            _ => {}
        }
        let name = Self::reg_name(sub);
        dlog_dev!(LogModule::Vino, "CDMC: write reg {:#04x} ({}) = {:#04x}", sub, name, data);
    }

    fn reg_name(subaddr: u8) -> &'static str {
        match subaddr {
            reg::CONTROL    => "CONTROL",
            reg::SHUTTER    => "SHUTTER",
            reg::GAIN       => "GAIN",
            reg::BRIGHTNESS => "BRIGHTNESS",
            reg::RED_BAL    => "RED_BAL",
            reg::BLUE_BAL   => "BLUE_BAL",
            reg::RED_SAT    => "RED_SAT",
            reg::BLUE_SAT   => "BLUE_SAT",
            reg::GAMMA      => "GAMMA",
            reg::VERSION    => "VERSION",
            reg::RESET      => "RESET",
            _               => "(unknown)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A guest that never touches the camera controls gets the host frame
    /// through unaltered — any drift between the power-on register values and
    /// the transfer functions' neutral point shows up as a colour cast on
    /// every captured frame.
    #[test]
    fn power_on_defaults_pass_the_field_through_untouched() {
        let regs = Cdmc::new().regs_copy();
        // Luma spans the legal 16..=235 range; chroma spans the full byte.
        let original: Vec<u8> = (0..64)
            .flat_map(|i| {
                let luma = 16 + (i * 219 / 63) as u8;
                [(i * 4) as u8, luma, (i * 4 + 2) as u8, luma]
            })
            .collect();
        let mut pixels = original.clone();
        Cdmc::apply_uyvy_field(&mut pixels, &regs);
        assert_eq!(pixels, original);
    }

    #[test]
    fn neutral_chroma_survives_the_defaults() {
        let regs = Cdmc::new().regs_copy();
        let mut pixels = vec![128, 100, 128, 120];
        Cdmc::apply_uyvy_field(&mut pixels, &regs);
        assert_eq!(pixels, vec![128, 100, 128, 120]);
    }

    #[test]
    fn red_balance_above_default_shifts_cr_up() {
        let mut regs = Cdmc::new().regs_copy();
        regs[reg::RED_BAL as usize] = reg::RED_BAL_DEFAULT + 0x40;
        let mut pixels = vec![128, 100, 128, 100];
        Cdmc::apply_uyvy_field(&mut pixels, &regs);
        assert_eq!(pixels[0], 128, "blue-difference channel untouched");
        assert!(pixels[2] > 128, "Cr shifted up, got {}", pixels[2]);
    }

    /// Automatic white balance is the camera's own loop; the balance registers
    /// stop driving the picture while it is enabled.
    #[test]
    fn awb_suppresses_the_balance_registers() {
        let mut regs = Cdmc::new().regs_copy();
        regs[reg::CONTROL as usize] |= reg::CONTROL_AWBCTL;
        regs[reg::RED_BAL as usize] = 0xFF;
        let mut pixels = vec![128, 100, 128, 100];
        Cdmc::apply_uyvy_field(&mut pixels, &regs);
        assert_eq!(pixels, vec![128, 100, 128, 100]);
    }

    #[test]
    fn version_is_read_only_and_identifies_an_indycam() {
        let cdmc = Cdmc::new();
        assert_eq!(cdmc.regs_copy()[reg::VERSION as usize], reg::VERSION_VAL);
        cdmc.i2c_write(0x56);
        cdmc.i2c_write(reg::VERSION);
        cdmc.i2c_write(0x00);
        cdmc.i2c_stop();
        assert_eq!(cdmc.regs_copy()[reg::VERSION as usize], reg::VERSION_VAL);
    }

    #[test]
    fn reset_restores_the_power_on_values() {
        let cdmc = Cdmc::new();
        cdmc.i2c_write(0x56);
        cdmc.i2c_write(reg::RED_BAL);
        cdmc.i2c_write(0xFF);
        cdmc.i2c_stop();
        assert_eq!(cdmc.regs_copy()[reg::RED_BAL as usize], 0xFF);

        cdmc.i2c_write(0x56);
        cdmc.i2c_write(reg::RESET);
        cdmc.i2c_write(0x01);
        cdmc.i2c_stop();
        assert_eq!(cdmc.regs_copy()[reg::RED_BAL as usize], reg::RED_BAL_DEFAULT);
    }
}
