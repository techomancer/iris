/// SAA7191B — Philips Digital Multistandard Colour Decoder (DMSD)
///
/// Fake device: full register storage and proper I2C identification.
/// No actual video decode — we just accept writes and return status on read.
///
/// I2C address: 0x8A write / 0x8B read  (IICSA=LOW, default)
///              0x8E write / 0x8F read  (IICSA=HIGH)
///
/// References:
///   saa7191.{h,cpp}  — MAME reference (Ryan Holtz)
///   Philips SAA7191B data sheet

use parking_lot::Mutex;

use crate::devlog::LogModule;

// ─── Register subaddresses ────────────────────────────────────────────────────

pub mod reg {
    pub const IDEL: u8 = 0x00; // Increment delay
    pub const HSYB: u8 = 0x01; // H sync begin, 50 Hz
    pub const HSYS: u8 = 0x02; // H sync stop, 50 Hz
    pub const HCLB: u8 = 0x03; // H clamp begin, 50 Hz
    pub const HCLS: u8 = 0x04; // H clamp stop, 50 Hz
    pub const HPHI: u8 = 0x05; // H sync after PHI1, 50 Hz
    pub const LUMC: u8 = 0x06; // Luminance control
    pub const HUEC: u8 = 0x07; // Hue control
    pub const CKTQ: u8 = 0x08; // Colour killer threshold QAM
    pub const CKTS: u8 = 0x09; // Colour killer threshold SECAM
    pub const PLSE: u8 = 0x0A; // PAL switch sensitivity
    pub const SESE: u8 = 0x0B; // SECAM switch sensitivity
    pub const GAIN: u8 = 0x0C; // Chroma gain control settings
    pub const STDC: u8 = 0x0D; // Standard/mode control
    pub const IOCK: u8 = 0x0E; // I/O and clock control
    pub const CTL1: u8 = 0x0F; // Control #1
    pub const CTL2: u8 = 0x10; // Control #2
    pub const CHCV: u8 = 0x11; // Chroma gain reference
    // 0x12, 0x13: not used, acknowledged (stored but ignored)
    pub const HS6B: u8 = 0x14; // H sync begin, 60 Hz
    pub const HS6S: u8 = 0x15; // H sync stop, 60 Hz
    pub const HC6B: u8 = 0x16; // H clamp begin, 60 Hz
    pub const HC6S: u8 = 0x17; // H clamp stop, 60 Hz
    pub const HP6I: u8 = 0x18; // H sync after PHI1, 60 Hz

    /// Total number of register slots (0x00–0x18 inclusive = 25).
    pub const COUNT: usize = 0x19;

    // ── LUMC bitfields ────────────────────────────────────────────────────
    pub const LUMC_APER_SHIFT: u8 = 0; pub const LUMC_APER_MASK: u8 = 0x03;
    pub const LUMC_CORI_SHIFT: u8 = 2; pub const LUMC_CORI_MASK: u8 = 0x0C;
    pub const LUMC_BPSS_SHIFT: u8 = 4; pub const LUMC_BPSS_MASK: u8 = 0x30;
    pub const LUMC_PREF_BIT:   u8 = 6; pub const LUMC_PREF_MASK: u8 = 0x40;
    pub const LUMC_BYPS_BIT:   u8 = 7; pub const LUMC_BYPS_MASK: u8 = 0x80;

    // ── GAIN bitfields ────────────────────────────────────────────────────
    pub const GAIN_LFIS_SHIFT: u8 = 5; pub const GAIN_LFIS_MASK: u8 = 0x60;
    pub const GAIN_COLO_BIT:   u8 = 7; pub const GAIN_COLO_MASK: u8 = 0x80;

    // ── STDC bitfields ────────────────────────────────────────────────────
    pub const STDC_SECS_BIT:  u8 = 0; // SECAM select
    pub const STDC_GPSW0_BIT: u8 = 1; // General purpose switch 0
    pub const STDC_HRMV_BIT:  u8 = 2; // Horizontal reference move
    pub const STDC_NFEN_BIT:  u8 = 3; // Noise filter enable
    pub const STDC_VTRC_BIT:  u8 = 7; // VTR mode (60 Hz) select

    // ── IOCK bitfields ────────────────────────────────────────────────────
    pub const IOCK_GPSW1_BIT: u8 = 0; // General purpose switch 1
    pub const IOCK_GPSW2_BIT: u8 = 1; // General purpose switch 2
    pub const IOCK_CHRS_BIT:  u8 = 2; // Chroma reset
    pub const IOCK_OEDY_BIT:  u8 = 3; // Output enable DY
    pub const IOCK_OEVS_BIT:  u8 = 4; // Output enable VS
    pub const IOCK_OEHS_BIT:  u8 = 5; // Output enable HS
    pub const IOCK_OEDC_BIT:  u8 = 6; // Output enable DC
    pub const IOCK_HPLL_BIT:  u8 = 7; // H-PLL open/closed

    // ── CTL1 bitfields ────────────────────────────────────────────────────
    pub const CTL1_YDEL_SHIFT: u8 = 0; pub const CTL1_YDEL_MASK: u8 = 0x07; // Y delay
    pub const CTL1_OFTS_BIT:   u8 = 3; // Output format select
    pub const CTL1_SCEN_BIT:   u8 = 4; // Subcarrier to ENC
    pub const CTL1_SXCR_BIT:   u8 = 5; // Subcarrier crystal
    pub const CTL1_FSEL_BIT:   u8 = 6; // Field select
    pub const CTL1_AUFD_BIT:   u8 = 7; // Automatic field detection

    // ── CTL2 bitfields ────────────────────────────────────────────────────
    pub const CTL2_VNOI_SHIFT: u8 = 0; pub const CTL2_VNOI_MASK: u8 = 0x03; // Video noise
    pub const CTL2_HRFS_BIT:   u8 = 2; // H-ref frequency select
}

// ─── I2C state machine ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum I2cState {
    Idle,
    SubaddrWrite, // received write address, awaiting subaddress
    SubaddrRead,  // received read address, awaiting subaddress
    DataWrite,    // streaming register writes
    DataRead,     // streaming register reads
}

// ─── Device state ─────────────────────────────────────────────────────────────

struct Saa7191State {
    regs:         [u8; reg::COUNT],
    status:       u8,   // hardware status byte (returned for subaddr 0x01 reads)

    i2c_write_addr: u8, // 0x8A or 0x8E depending on IICSA pin
    i2c_read_addr:  u8, // write_addr | 1
    i2c_subaddr:    u8,
    i2c_state:      I2cState,
}

impl Default for Saa7191State {
    fn default() -> Self {
        Self {
            regs:           [0u8; reg::COUNT],
            status:         0,
            i2c_write_addr: 0x8A,
            i2c_read_addr:  0x8B,
            i2c_subaddr:    0x00,
            i2c_state:      I2cState::Idle,
        }
    }
}

// ─── Public handle ────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Saa7191 {
    state: std::sync::Arc<Mutex<Saa7191State>>,
}

impl Saa7191 {
    pub fn new() -> Self {
        Self { state: std::sync::Arc::new(Mutex::new(Saa7191State::default())) }
    }

    pub fn power_on(&self) {
        *self.state.lock() = Saa7191State::default();
    }

    /// True when the device has accepted an address byte and is mid-transfer.
    /// VINO uses this to route subsequent bytes to the correct I2C target.
    pub fn is_active(&self) -> bool {
        self.state.lock().i2c_state != I2cState::Idle
    }

    /// Set the IICSA pin state (false=LOW → addr 0x8A, true=HIGH → addr 0x8E).
    pub fn set_iicsa(&self, high: bool) {
        let mut st = self.state.lock();
        st.i2c_write_addr = if high { 0x8E } else { 0x8A };
        st.i2c_read_addr  = st.i2c_write_addr | 1;
    }

    // ── I2C interface ─────────────────────────────────────────────────────

    /// Write one byte over I2C.  VINO calls this after assembling a byte
    /// from its own I2C_CONTROL / I2C_DATA register pair.
    pub fn i2c_write(&self, data: u8) {
        let mut st = self.state.lock();
        match st.i2c_state {
            I2cState::Idle => {
                if data == st.i2c_write_addr {
                    st.i2c_state = I2cState::SubaddrWrite;
                } else if data == st.i2c_read_addr {
                    st.i2c_state = I2cState::SubaddrRead;
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
                dlog_dev!(LogModule::Vino, "SAA7191: I2C expected read but got write, returning to idle");
                st.i2c_state = I2cState::Idle;
            }
        }
    }

    /// Read one byte over I2C.  Only subaddress 0x01 (status) is defined;
    /// all others return 0x00
    pub fn i2c_read(&self) -> u8 {
        let mut st = self.state.lock();
        if st.i2c_state != I2cState::DataRead {
            dlog_dev!(LogModule::Vino, "SAA7191: i2c_read called in state {:?}, returning to idle", st.i2c_state);
            st.i2c_state = I2cState::Idle;
            return 0;
        }
        let subaddr = st.i2c_subaddr;
        st.i2c_subaddr = subaddr.wrapping_add(1) % reg::COUNT as u8;

        if subaddr == 0x01 {
            // Status register — only readable byte
            st.status
        } else {
            dlog_dev!(LogModule::Vino, "SAA7191: i2c_read subaddr {:#04x} not readable, returning 0x00", subaddr);
            0x00
        }
    }

    /// I2C STOP condition — returns state machine to idle.
    pub fn i2c_stop(&self) {
        self.state.lock().i2c_state = I2cState::Idle;
    }

    // ── Register write ────────────────────────────────────────────────────

    fn reg_w(st: &mut Saa7191State, data: u8) {
        let sub = st.i2c_subaddr as usize;
        if sub < reg::COUNT {
            st.regs[sub] = data;
        }
        // Log writes for debugging
        let name = Self::reg_name(st.i2c_subaddr);
        dlog_dev!(LogModule::Vino, "SAA7191: write reg {:#04x} ({}) = {:#04x}", st.i2c_subaddr, name, data);
    }

    fn reg_name(subaddr: u8) -> &'static str {
        match subaddr {
            reg::IDEL => "IDEL: increment delay",
            reg::HSYB => "HSYB: H sync begin 50Hz",
            reg::HSYS => "HSYS: H sync stop 50Hz",
            reg::HCLB => "HCLB: H clamp begin 50Hz",
            reg::HCLS => "HCLS: H clamp stop 50Hz",
            reg::HPHI => "HPHI: H sync after PHI1 50Hz",
            reg::LUMC => "LUMC: luminance control",
            reg::HUEC => "HUEC: hue control",
            reg::CKTQ => "CKTQ: colour killer threshold QAM",
            reg::CKTS => "CKTS: colour killer threshold SECAM",
            reg::PLSE => "PLSE: PAL switch sensitivity",
            reg::SESE => "SESE: SECAM switch sensitivity",
            reg::GAIN => "GAIN: chroma gain control",
            reg::STDC => "STDC: standard/mode control",
            reg::IOCK => "IOCK: I/O and clock control",
            reg::CTL1 => "CTL1: control #1",
            reg::CTL2 => "CTL2: control #2",
            reg::CHCV => "CHCV: chroma gain reference",
            0x12      => "(not used)",
            0x13      => "(not used)",
            reg::HS6B => "HS6B: H sync begin 60Hz",
            reg::HS6S => "HS6S: H sync stop 60Hz",
            reg::HC6B => "HC6B: H clamp begin 60Hz",
            reg::HC6S => "HC6S: H clamp stop 60Hz",
            reg::HP6I => "HP6I: H sync after PHI1 60Hz",
            _         => "(unknown)",
        }
    }
}
