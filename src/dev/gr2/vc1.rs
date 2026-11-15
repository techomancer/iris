//! VC1 video controller — storage and table decode.
//!
//! The emulator does not run the VC1 timing program. Only the state the
//! compositor needs is used: the DID tables, cursor glyph, cursor position
//! and `sysctl`. The PROM/kernel programming sequence is documented in
//! `ignore/gr2/VC1.h`.

pub const VC1_CMD0: u32 = 0x00;
pub const VC1_CMD1: u32 = 0x04;
pub const VC1_SRAM: u32 = 0x08;
pub const VC1_TESTREG: u32 = 0x0c;
pub const VC1_ADDRLO: u32 = 0x10;
pub const VC1_ADDRHI: u32 = 0x14;
pub const VC1_SYSCTL: u32 = 0x18;

// Internal register (byte) addresses.
pub const VID_EP: usize = 0x00;
pub const CUR_EP: usize = 0x20;
pub const CUR_XL: usize = 0x22;
pub const CUR_YL: usize = 0x24;
pub const DID_EP: usize = 0x40;
pub const DID_END_EP: usize = 0x42;

pub const SYSCTL_CURSOR: u8 = 0x10;
pub const SYSCTL_CURSOR_DISPLAY: u8 = 0x20;

/// VC1 revision reported through `testreg`.
pub const VC1_REV: u8 = 1;

pub const SRAM_SIZE: usize = 0x10000;

/// [`Vc1::totals`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vc1Totals {
    /// Pixel clocks a line and lines a frame, blanking included.
    pub h_total: u32,
    pub v_total: u32,
    /// Lines of the picture, and the runs they come in (fields).
    pub visible_lines: u32,
    pub fields: u32,
}

/// Plain data; all-zero is a valid power-on state.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Vc1 {
    /// Internal address pointer (addrhi:addrlo), shared by cmd0 and SRAM.
    pub addr: u16,
    pub sysctl: u8,
    pub cmd1: u8,
    /// Internal registers, byte-addressed, big-endian 16-bit pairs.
    pub regs: [u8; 256],
    /// External SRAM, byte-addressed, big-endian 16-bit words.
    pub sram: [u8; SRAM_SIZE],
}

impl Vc1 {
    /// A cmd0/SRAM write carries 16 bits and advances the pointer by 2.
    pub fn write(&mut self, reg: u32, val: u32) {
        match reg {
            VC1_CMD0 => {
                let a = (self.addr & 0xff) as usize;
                self.regs[a] = (val >> 8) as u8;
                self.regs[(a + 1) & 0xff] = val as u8;
                self.addr = self.addr.wrapping_add(2);
            }
            VC1_CMD1 => self.cmd1 = val as u8,
            VC1_SRAM => {
                let a = self.addr as usize;
                self.sram[a] = (val >> 8) as u8;
                self.sram[(a + 1) & (SRAM_SIZE - 1)] = val as u8;
                self.addr = self.addr.wrapping_add(2);
            }
            VC1_ADDRLO => self.addr = (self.addr & 0xff00) | (val as u16 & 0xff),
            VC1_ADDRHI => self.addr = ((val as u16 & 0xff) << 8) | (self.addr & 0xff),
            VC1_SYSCTL => self.sysctl = val as u8,
            _ => {}
        }
    }

    pub fn read(&mut self, reg: u32) -> u32 {
        match reg {
            VC1_CMD0 => {
                let a = (self.addr & 0xff) as usize;
                let v = ((self.regs[a] as u32) << 8) | self.regs[(a + 1) & 0xff] as u32;
                self.addr = self.addr.wrapping_add(2);
                v
            }
            VC1_CMD1 => self.cmd1 as u32,
            VC1_SRAM => {
                let a = self.addr as usize;
                let v = self.sram16(a) as u32;
                self.addr = self.addr.wrapping_add(2);
                v
            }
            VC1_TESTREG => VC1_REV as u32,
            VC1_ADDRLO => (self.addr & 0xff) as u32,
            VC1_ADDRHI => (self.addr >> 8) as u32,
            VC1_SYSCTL => self.sysctl as u32,
            _ => 0,
        }
    }

    #[inline]
    /// What the video timing tables add up to. The frame table at VID_EP
    /// is a word, the lines of a frame, then three bytes a line type (the
    /// address of its line-state table, and how many lines repeat it) until
    /// the repeats reach that word. A line-state table is runs: state
    /// bytes, the last of them with bit 7 set, then a byte whose low seven
    /// bits are the run's length in units of five pixel clocks and whose bit
    /// 7 ends the line. A line is in the picture when the last state byte
    /// of one of its runs has bits 6, 1 and 0 all set. Read from the PROM's
    /// tables and checked on IRIX's fourteen format files: in each, every
    /// line type has the same length, and the picture lines come to the
    /// format's height (1024 or 1025 of 60 Hz's 1065) in one run, or in two
    /// for the interlaced formats and full-screen stereo. None while no
    /// table is loaded.
    pub fn totals(&self) -> Option<Vc1Totals> {
        let frame = self.reg16(VID_EP) as usize;
        let v_total = self.sram16(frame) as u32;
        let mut t = Vc1Totals { h_total: 0, v_total, visible_lines: 0, fields: 0 };
        let (mut lines, mut in_picture) = (0u32, false);
        // The tables are the guest's: every walk is bounded.
        for k in 0..256 {
            if lines >= v_total {
                break;
            }
            let at = frame + 2 + 3 * k;
            let line = self.sram16(at) as usize;
            let repeats = self.sram[(at + 2) & (SRAM_SIZE - 1)] as u32;
            lines += repeats;
            let (mut a, mut units, mut visible) = (line, 0u32, false);
            for _ in 0..512 {
                // State bytes up to the one with bit 7, then the length.
                while self.sram[a & (SRAM_SIZE - 1)] & 0x80 == 0 && a < line + 2048 {
                    a += 1;
                }
                visible |= self.sram[a & (SRAM_SIZE - 1)] & 0x43 == 0x43;
                let n = self.sram[(a + 1) & (SRAM_SIZE - 1)];
                a += 2;
                units += (n & 0x7f) as u32;
                if n & 0x80 != 0 {
                    break;
                }
            }
            t.h_total = t.h_total.max(5 * units);
            if visible {
                t.visible_lines += repeats;
                t.fields += !in_picture as u32;
            }
            in_picture = visible;
        }
        (v_total != 0 && lines == v_total && t.h_total != 0).then_some(t)
    }

    /// Lines the window-ID frame table covers: the picture's height.
    pub fn did_lines(&self) -> u32 {
        (self.reg16(DID_END_EP).saturating_sub(self.reg16(DID_EP)) / 2) as u32
    }

    pub fn reg16(&self, a: usize) -> u16 {
        ((self.regs[a & 0xff] as u16) << 8) | self.regs[(a + 1) & 0xff] as u16
    }

    #[inline]
    pub fn sram16(&self, a: usize) -> u16 {
        ((self.sram[a & (SRAM_SIZE - 1)] as u16) << 8) | self.sram[(a + 1) & (SRAM_SIZE - 1)] as u16
    }

    pub fn cursor_visible(&self) -> bool {
        self.sysctl & (SYSCTL_CURSOR | SYSCTL_CURSOR_DISPLAY) == (SYSCTL_CURSOR | SYSCTL_CURSOR_DISPLAY)
    }

    /// Cursor top-left in display coordinates (0,0 = top-left), undoing the
    /// host's timing-table offsets. The offsets are inferred from the PROM and
    /// kernel cursor code; see VC1.h.
    pub fn cursor_pos(&self) -> (i32, i32) {
        const X_ADJ: i32 = 250;
        const Y_ADJ: i32 = 35;
        let xl = self.reg16(CUR_XL) as i32;
        let yl = self.reg16(CUR_YL) as i32;
        let x = if self.reg16(VID_EP) == 0x800 { xl - X_ADJ } else { xl };
        (x, yl - Y_ADJ)
    }

    /// 2-bit cursor pixel at glyph position (cx, cy), 32x32. Plane 0 lives at
    /// words 0..63 (2 words per row) and plane 1 at words 64..127.
    #[inline]
    pub fn cursor_pixel(&self, cx: usize, cy: usize) -> u8 {
        let base = self.reg16(CUR_EP) as usize;
        let word = cy * 2 + (cx >> 4);
        let bit = 15 - (cx & 15);
        let p0 = (self.sram16(base + word * 2) >> bit) & 1;
        let p1 = (self.sram16(base + (64 + word) * 2) >> bit) & 1;
        (p0 | (p1 << 1)) as u8
    }

    /// Maximum DID transitions per line the compositor follows.
    pub const MAX_DID_RUNS: usize = 64;

    /// DID transitions for display line `y` (0 = top), as (x, did) with x
    /// ascending; returns how many. The frame table at DID_EP holds one line
    /// table pointer per scanline; a line table is [count, entry * count]
    /// with entry = (x << 5) | did (VC1.h, confirmed against Xsgi's tables:
    /// [3, 0x0003, 0x28c9, 0x7583] = DID 3, DID 9 from x 326, DID 3 from 940).
    pub fn line_did_runs(&self, y: usize, runs: &mut [(u16, u8); Self::MAX_DID_RUNS]) -> usize {
        let frame = self.reg16(DID_EP) as usize;
        if frame == 0 {
            runs[0] = (0, 0);
            return 1;
        }
        let line = self.sram16(frame + y * 2) as usize;
        let count = (self.sram16(line) as usize).clamp(1, Self::MAX_DID_RUNS);
        for (k, r) in runs.iter_mut().enumerate().take(count) {
            let e = self.sram16(line + 2 + k * 2);
            *r = ((e >> 5) & 0x7ff, (e & 0x1f) as u8);
        }
        runs[0].0 = 0;
        count
    }

    /// DID at display pixel (x, y).
    pub fn did_at(&self, x: usize, y: usize) -> u8 {
        let mut runs = [(0u16, 0u8); Self::MAX_DID_RUNS];
        let n = self.line_did_runs(y, &mut runs);
        runs[..n].iter().rev().find(|r| r.0 as usize <= x).map_or(runs[0].1, |r| r.1)
    }

    /// DID at the left edge of display line `y`.
    pub fn line_did(&self, y: usize) -> u8 {
        self.did_at(0, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_timing_tables_total_a_frame() {
        // SAFETY: plain data, valid zeroed.
        let mut v: Box<Vc1> = unsafe { Box::new_zeroed().assume_init() };
        assert_eq!(v.totals(), None, "nothing loaded");
        // Two line types: 336 units of five clocks each, in runs of one and
        // of two or three state bytes, the last run's count with the end bit.
        let blank = [0x2c, 0x64, 0xcc, 7, 0x24, 0xe0, 100, 0xac, 124, 0xa4, 0x80 | 105];
        let active = [0x2c, 0xe5, 61, 0xdf, 100, 0xac, 100, 0x2c, 0xed, 0x80 | 75];
        v.sram[0x10..0x10 + blank.len()].copy_from_slice(&blank);
        v.sram[0x40..0x40 + active.len()].copy_from_slice(&active);
        // The frame table: 1065 lines as 33 + 1024 + 8.
        let frame = [0x04, 0x29, 0x00, 0x10, 33, 0x00, 0x40, 126, 0x00, 0x40, 126, 0x00, 0x40, 126, 0x00, 0x40, 126, 0x00, 0x40, 126, 0x00, 0x40, 126, 0x00, 0x40, 126, 0x00, 0x40, 126, 0x00, 0x40, 16, 0x00, 0x10, 8, 0xff, 0xff];
        v.sram[0x900..0x900 + frame.len()].copy_from_slice(&frame);
        v.regs[VID_EP] = 0x09;
        assert_eq!(v.totals(), Some(Vc1Totals { h_total: 1680, v_total: 1065, visible_lines: 1024, fields: 1 }));
        // Entries past the frame's lines (IRIX's tables carry an alternate
        // ending there) are not part of it.
        v.sram[0x900 + frame.len() - 2..0x900 + frame.len() + 4].copy_from_slice(&[0x00, 0xff, 0xff, 0x00, 0x10, 3]);
        assert_eq!(v.totals().map(|t| t.v_total), Some(1065));
        // 60 Hz at the board's 107.352 MHz.
        assert_eq!(107_352_000 / (1680 * 1065), 60);
        // Repeats that do not add up to the stated lines are a table half
        // loaded.
        v.sram[0x904] = 32;
        assert_eq!(v.totals(), None);
        // The picture's height from the window-ID frame table's extent.
        v.regs[DID_EP] = 0x0b;
        v.regs[DID_END_EP] = 0x13;
        assert_eq!(v.did_lines(), 1024);
    }
}
