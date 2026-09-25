//! GE7 geometry engines — storage only.
//!
//! GE microcode is stored and read back but never executed; HQ2 commands are
//! emulated at a high level (`hq2.rs`). The load/verify protocol is described
//! in `ignore/gr2/GE7.h`.

pub const GE_WINDOW_WORDS: usize = 256;
pub const GE_MAX: usize = 8;
pub const GE_UCODE_WORDS: usize = 64 * 1024;

/// Staging registers in GE #0's window.
pub const STAGE0: usize = 0xf8;
pub const STAGE3: usize = 0xfb;
/// RAM bank select / revision register.
pub const BANKSEL: usize = 0xfd;

/// GE7 revision reported in ram0[0xFD] bits 7:5.
pub const GE7_REV: u32 = 1;

/// Mask applied when `ge7loaducode` is read back (PROM/kernel verify mask).
pub const LOADUCODE_MASK: u32 = 0x03df_ffff;

/// One stored 160-bit microword: bits 127:0 from the staging registers, bits
/// 159:128 held by the HQ2.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GeMicroword {
    pub stage: [u32; 4],
    pub hq: u32,
}

/// Plain data; all-zero is a valid power-on state.
#[repr(C)]
pub struct Ge7 {
    /// Number of installed engines (1, 2, 4 or 8); windows past it do not respond.
    pub count: u32,
    pub win: [[u32; GE_WINDOW_WORDS]; GE_MAX],
    pub ucode: [GeMicroword; GE_UCODE_WORDS],
}

impl Ge7 {
    pub fn read(&self, ge: usize, word: usize) -> u32 {
        if ge >= self.count as usize {
            return 0;
        }
        if ge == 0 && word == BANKSEL {
            return (GE7_REV << 5) | (self.win[0][BANKSEL] & 0x1f);
        }
        self.win[ge][word]
    }

    pub fn write(&mut self, ge: usize, word: usize, val: u32) {
        if ge < self.count as usize {
            self.win[ge][word] = val;
        }
    }

    /// `ge7loaducode` write: commit staging + HQ bits at `pc`.
    pub fn commit(&mut self, pc: u32, hq_bits: u32) {
        let mut w = GeMicroword { stage: [0; 4], hq: hq_bits };
        w.stage.copy_from_slice(&self.win[0][STAGE0..=STAGE3]);
        self.ucode[pc as usize & (GE_UCODE_WORDS - 1)] = w;
    }

    /// `gepc` write: reload the staging registers from the stored word and
    /// return the HQ bits for the `ge7loaducode` readback latch.
    pub fn reload(&mut self, pc: u32) -> u32 {
        let w = self.ucode[pc as usize & (GE_UCODE_WORDS - 1)];
        self.win[0][STAGE0..=STAGE3].copy_from_slice(&w.stage);
        w.hq
    }
}
