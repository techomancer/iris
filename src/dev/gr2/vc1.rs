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

pub const SYSCTL_CURSOR: u8 = 0x10;
pub const SYSCTL_CURSOR_DISPLAY: u8 = 0x20;

/// VC1 revision reported through `testreg`.
pub const VC1_REV: u8 = 1;

pub const SRAM_SIZE: usize = 0x10000;

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
