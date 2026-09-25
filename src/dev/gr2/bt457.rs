//! Brooktree Bt457 RAMDAC — one per colour channel (red, green, blue).
//!
//! Storage only. The display compositor reads `readmask` and `palette` to
//! apply the per-channel gamma ramp. See `ignore/gr2/GR2.h` (Bt457 section)
//! for the register protocol and the PROM/kernel init values.

/// Internal control register addresses (selected by `addr`, accessed via `cmd2`).
pub const DAC_READMASK: u8 = 0x04;
pub const DAC_BLINKMASK: u8 = 0x05;
pub const DAC_CMD: u8 = 0x06;
pub const DAC_TEST: u8 = 0x07;

/// Register offsets within a DAC's 0x20-byte window.
pub const DAC_ADDR: u32 = 0x00;
pub const DAC_PALETTE: u32 = 0x04;
pub const DAC_CTRL: u32 = 0x08;
pub const DAC_OVERLAY: u32 = 0x0c;

/// One Bt457. Plain data; all-zero is a valid power-on state.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Bt457 {
    /// Address pointer: palette/overlay index, or control register select.
    pub addr: u8,
    pub readmask: u8,
    pub blinkmask: u8,
    pub cmd: u8,
    pub test: u8,
    /// 256-entry gamma ramp for this channel.
    pub palette: [u8; 256],
    /// 4 overlay colour entries for this channel.
    pub overlay: [u8; 4],
}

impl Bt457 {
    pub fn write(&mut self, reg: u32, val: u8) {
        match reg {
            DAC_ADDR => self.addr = val,
            DAC_PALETTE => {
                self.palette[self.addr as usize] = val;
                self.addr = self.addr.wrapping_add(1);
            }
            DAC_CTRL => match self.addr & 7 {
                4 => self.readmask = val,
                5 => self.blinkmask = val,
                6 => self.cmd = val,
                7 => self.test = val,
                _ => {}
            },
            DAC_OVERLAY => {
                self.overlay[(self.addr & 3) as usize] = val;
                self.addr = self.addr.wrapping_add(1);
            }
            _ => {}
        }
    }

    pub fn read(&mut self, reg: u32) -> u8 {
        match reg {
            DAC_ADDR => self.addr,
            DAC_PALETTE => {
                let v = self.palette[self.addr as usize];
                self.addr = self.addr.wrapping_add(1);
                v
            }
            DAC_CTRL => match self.addr & 7 {
                4 => self.readmask,
                5 => self.blinkmask,
                6 => self.cmd,
                7 => self.test,
                _ => 0,
            },
            DAC_OVERLAY => {
                let v = self.overlay[(self.addr & 3) as usize];
                self.addr = self.addr.wrapping_add(1);
                v
            }
            _ => 0,
        }
    }

    /// Output level for an 8-bit channel value: read mask, then gamma ramp.
    #[inline]
    pub fn lookup(&self, v: u8) -> u8 {
        self.palette[(v & self.readmask) as usize]
    }
}
