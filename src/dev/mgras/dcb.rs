//! The display control bus (DCB): the board's slow side bus to its video
//! chips, reached through a 32 KB window at slot offset `0x60000`.
//!
//! Each device owns a 1 KB window (`0x60000 + dev * 0x400`), and the address
//! within it encodes the transaction: bits 9:7 select the chip's register
//! (its "CRS" line), bits 4:3 the transfer width in bytes (0 = 4), bit 5 asks
//! the chip to increment its register select, bit 6 packs data. Data rides in
//! the most significant bytes of a 32-bit bus access, so a one-byte register
//! read with a word load comes back in bits 31:24.

use super::plain::RegMap;

/// Device numbers on the bus.
pub const DEV_CMAP_ALL: u32 = 3;
pub const DEV_CMAP0: u32 = 4;
pub const DEV_CMAP1: u32 = 5;
pub const DEV_DAC: u32 = 6;
pub const DEV_XMAP: u32 = 7;
pub const DEV_PP1_XMAP: u32 = DEV_XMAP;
pub const DEV_VC3: u32 = 8;
pub const DEV_BDVERS: u32 = 9;
pub const DEV_I2C: u32 = 11;

/// One decoded bus transaction.
#[derive(Clone, Copy, Debug)]
pub struct Txn {
    pub dev: u32,
    pub crs: u32,
    /// Transfer width in bytes, 1-4.
    pub width: u32,
}

impl Txn {
    /// `off` is the offset within the slot, in `0x60000..0x68000`.
    pub fn decode(off: u32) -> Self {
        let w = off & 0x3FF;
        let width = match (w >> 3) & 3 { 0 => 4, n => n };
        Txn { dev: (off - 0x60000) >> 10, crs: (w >> 7) & 7, width }
    }

    /// The data a CPU store of `bits` carries for this transaction.
    pub fn store_data(&self, bits: u32, val: u64) -> u32 {
        match bits {
            8 => val as u32 & 0xFF,
            16 => val as u32 & 0xFFFF,
            32 => (val as u32) >> (8 * (4 - self.width)),
            _ => ((val >> 32) as u32) >> (8 * (4 - self.width)),
        }
    }

    /// Place `data` where a CPU load of `bits` expects it.
    pub fn load_value(&self, bits: u32, data: u32) -> u64 {
        match bits {
            8 => (data & 0xFF) as u64,
            16 => (data & 0xFFFF) as u64,
            32 => (data << (8 * (4 - self.width))) as u64 & 0xFFFF_FFFF,
            _ => ((data << (8 * (4 - self.width))) as u64) << 32,
        }
    }
}

/// A colormap chip: 8192 entries of 24-bit RGB.
#[repr(C)]
pub struct Cmap {
    pub pal: [u32; 8192],
    addr: u32,
    rev: u32,
    cmd: u32,
}

impl Cmap {
    fn write(&mut self, t: Txn, d: u32) {
        match (t.crs, t.width) {
            // Address: one byte at a time (low, then high), or 16 bits sent low
            // byte first.
            (0, 1) => self.addr = (self.addr & 0x1F00) | d,
            (0, _) => self.addr = (((d & 0xFF) << 8) | (d >> 8 & 0xFF)) & 0x1FFF,
            (1, _) => self.addr = ((d & 0x1F) << 8) | (self.addr & 0xFF),
            // Palette entry: red, green, blue; the address advances.
            (2, _) => {
                self.pal[self.addr as usize] = d & 0xFF_FFFF;
                self.addr = (self.addr + 1) & 0x1FFF;
            }
            (3, _) => self.cmd = d,
            _ => {}
        }
    }

    fn read(&self, t: Txn) -> u32 {
        match t.crs {
            0 => self.addr & 0xFF,
            1 => self.addr >> 8,
            2 => self.pal[self.addr as usize],
            3 => self.cmd,
            4 => 0x08, // status: ready for writes
            6 => self.rev,
            _ => 0,
        }
    }
}

/// The RAMDAC: an address register selecting internal registers, and a
/// 256-entry, 10-bit gamma table written as red, green, blue in turn.
#[repr(C)]
pub struct Dac {
    addr: u32,
    regs: RegMap<256>,
    pub gamma: [[u16; 3]; 256],
    gamma_comp: usize,
    mode: u32,
}

/// DAC register: the pixel read mask; zero blanks the screen.
pub const DAC_PIXMASK: u32 = 4;

impl Dac {
    /// Power-on: an identity gamma ramp.
    fn init(&mut self) {
        for (i, g) in self.gamma.iter_mut().enumerate() {
            let v = (i << 2) as u16;
            *g = [v, v, v];
        }
    }

    fn write(&mut self, t: Txn, d: u32) {
        match t.crs {
            // 16 bits, low byte first; a single byte sets the low half.
            0 if t.width >= 2 => { self.addr = ((d & 0xFF) << 8) | (d >> 8 & 0xFF); self.gamma_comp = 0; }
            0 => { self.addr = d & 0xFF; self.gamma_comp = 0; }
            1 => {
                let i = (self.addr & 0xFF) as usize;
                // Ten-bit entries: a byte write gives the top eight bits; a
                // 16-bit write gives bits 9:2 in its first byte and 1:0 in
                // the second.
                let v = if t.width >= 2 { (((d >> 8 & 0xFF) << 2) | (d & 3)) as u16 } else { (d as u16) << 2 };
                self.gamma[i][self.gamma_comp] = v & 0x3FF;
                self.gamma_comp += 1;
                if self.gamma_comp == 3 {
                    self.gamma_comp = 0;
                    self.addr = self.addr.wrapping_add(1);
                }
            }
            2 => self.regs.insert(self.addr, d & 0xFF),
            3 => self.mode = d,
            _ => {}
        }
    }

    fn read(&self, t: Txn) -> u32 {
        match t.crs {
            0 => self.addr,
            2 => self.regs.get(self.addr),
            3 => self.mode,
            _ => 0,
        }
    }

    /// Registers written so far, (address, value).
    pub fn regs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.regs.iter()
    }

    pub fn pixmask(&self) -> u32 {
        self.regs.lookup(DAC_PIXMASK).unwrap_or(0xFF)
    }
}

/// The XMAP: the pixel processors' display side (display modes per window ID,
/// buffer selects, scanout pointers), reached as an index register (`INDEX`)
/// plus register files on the selects after it.
#[repr(C)]
pub struct Xmap {
    index: u32,
    /// The config register (CONFIG select, indices 0-3): one 32-bit
    /// register whose bytes a one-byte transfer at index n reaches, byte 0
    /// being bits 31:24. Bit 19 makes the index auto-increment; bits 10:0
    /// hold the cursor colormap base / 4.
    config: u32,
    /// Register files, keyed by `key(select, index)`.
    regs: RegMap<1024>,
}

/// An XMAP register file key: the select in the top byte, the index below.
fn key(crs: u32, index: u32) -> u32 {
    crs << 24 | (index & 0xFF_FFFF)
}

/// XMAP registers, by select (named as in OpenBSD's impact(4)).
mod xmap {
    /// Which pixel processor hears writes.
    pub const PP1SELECT: u32 = 0;
    pub const INDEX: u32 = 1;
    pub const CONFIG: u32 = 2;
    pub const BUF_SELECT: u32 = 3;
    /// Display mode per window ID, at index `did * 4`.
    pub const MAIN_MODE: u32 = 4;
    pub const OVERLAY_MODE: u32 = 5;
    /// Display interface buffer registers (scanout), index 0 the pointers.
    pub const DIB: u32 = 6;
    /// The raster engine to pixel processor link.
    pub const RE_RAC: u32 = 7;
}

/// `CONFIG` index of the byte whose bit 3 makes the index auto-increment.
const XMAP_AUTOINC: u32 = 1 << 19;
/// `CONFIG` index 4: the pixel processor revision.
const XMAP_REV_INDEX: u32 = 4;
/// `CONFIG` index read before each display-mode write.
const XMAP_MODE_ROOM_INDEX: u32 = 8;

impl Xmap {
    fn autoinc(&mut self, t: Txn) {
        if t.crs >= xmap::BUF_SELECT && self.config & XMAP_AUTOINC != 0 {
            self.index = self.index.wrapping_add(t.width);
        }
    }

    fn write(&mut self, t: Txn, d: u32) {
        match t.crs {
            xmap::PP1SELECT => self.regs.insert(key(xmap::PP1SELECT, 0), d),
            xmap::INDEX => self.index = d,
            xmap::CONFIG if self.index < 4 => {
                // Bytes index..index+width of the register, most significant
                // first.
                let n = t.width.min(4 - self.index);
                let shift = 8 * (4 - self.index - n);
                let mask = (if n == 4 { u32::MAX } else { (1 << (8 * n)) - 1 }) << shift;
                self.config = (self.config & !mask) | ((d << shift) & mask);
            }
            crs => {
                self.regs.insert(key(crs, self.index), d);
                self.autoinc(t);
            }
        }
    }

    fn read(&mut self, t: Txn) -> u32 {
        let v = match t.crs {
            xmap::INDEX => self.index,
            // The raster engine to pixel processor link reports its sync
            // signature here; 1 in each nibble lane means "in sync".
            xmap::RE_RAC => 0x0001_0101,
            xmap::CONFIG if self.index < 4 => {
                let n = t.width.min(4 - self.index);
                let v = self.config >> (8 * (4 - self.index - n));
                if n == 4 { v } else { v & ((1 << (8 * n)) - 1) }
            }
            xmap::CONFIG if self.index == XMAP_REV_INDEX => 1,
            // Polled nonzero before every display-mode write: room for a
            // mode update.
            xmap::CONFIG if self.index == XMAP_MODE_ROOM_INDEX => 0x10,
            crs => self.regs.get(key(crs, self.index)),
        };
        if t.crs >= xmap::CONFIG { self.autoinc(t); }
        v
    }

    /// Display mode of overlay window ID `did` (`OVERLAY_MODE`, index `did * 4`);
    /// 0 is "overlay off".
    pub fn overlay_mode(&self, did: u32) -> u32 {
        self.regs.get(key(xmap::OVERLAY_MODE, (did & 0x1F) << 2))
    }

    /// Colormap address of cursor colour 0: the config register (`CONFIG`,
    /// index 0) holds it divided by four.
    pub fn cursor_cmap_base(&self) -> usize {
        (self.config as usize & 0x7FF) << 2
    }

    /// Register file entries written so far, (select << 24 | index, value).
    pub fn regs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.regs.iter()
    }

    /// Displayed buffer per window ID (`BUF_SELECT`, index 0): bit d set =
    /// window ID d shows buffer B (the DIB pointer's second field) instead
    /// of A. The kernel's retrace handler flips the bits for swapbuffers.
    pub fn buf_select(&self) -> u32 {
        self.regs.get(key(xmap::BUF_SELECT, 0))
    }

    /// Scanout page pointers (`DIB`, index 0): bits 9:0 the main buffer
    /// (A), 19:10 the second buffer (B, double buffering), 31:20 the
    /// overlay. Until written, the PROM's 1280x1024 value.
    pub fn dib_pointers(&self) -> u32 {
        match self.regs.get(key(xmap::DIB, 0)) {
            0 => 0x1C0C_8240,
            v => v,
        }
    }

    /// Display mode of window ID `did` (`MAIN_MODE`, index `did * 4`).
    pub fn main_mode(&self, did: u32) -> u32 {
        self.regs.get(key(xmap::MAIN_MODE, did << 2))
    }
}

/// The video timing chip: indexed 16-bit registers and a 32K x 16 SRAM holding
/// line and frame tables, the cursor glyph and window-ID tables.
/// VC3 cursor registers: glyph address, position, and control (bit 0
/// enable, bit 1 display, bit 3 64x64 glyph).
const CURSOR_GLYPH_REG: usize = 1;
const CURSOR_X_REG: usize = 2;
const CURSOR_Y_REG: usize = 3;
const CURSOR_CONTROL_REG: usize = 0x1D;
/// VC3 register: SRAM address of the main planes' window-ID frame table.
const MAIN_WID_FRAME_REG: usize = 4;
const OVERLAY_WID_FRAME_REG: usize = 5;

#[repr(C)]
pub struct Vc3 {
    index: u32,
    pub regs: [u16; 32],
    pub sram: [u16; 0x8000],
    line_counter: u16,
}

pub const VC3_SRAM_POINTER: usize = 7;
const VC3_CURRENT_LINE: usize = 0xB;

impl Vc3 {
    /// The hardware cursor, when shown: the screen position of its top-left
    /// corner, its size (32 or 64), and the SRAM address of its glyph. The
    /// glyph is two bit planes, one after the other, a row at a time, most
    /// significant bit leftmost; the planes make a two-bit colour, 0 being
    /// transparent. The position registers are offset by 31.
    pub fn cursor(&self) -> Option<(i32, i32, usize, usize)> {
        let ctl = self.regs[CURSOR_CONTROL_REG];
        if ctl & 3 != 3 {
            return None;
        }
        let size = if ctl & 8 != 0 { 64 } else { 32 };
        Some((
            self.regs[CURSOR_X_REG] as i32 - 31,
            self.regs[CURSOR_Y_REG] as i32 - 31,
            size,
            self.regs[CURSOR_GLYPH_REG] as usize,
        ))
    }

    /// Visible display size from the video timing tables (register 0 points
    /// at them; the layout is the VC2's, so Newport's decoder reads them).
    /// None while no timing is loaded.
    pub fn timing_size(&self) -> Option<(usize, usize)> {
        let (w, h, _) = crate::disp::decode_vc2_timings(&self.regs, &self.sram);
        (w > 0 && h > 0).then_some((w, h))
    }

    /// Scanlines in the main window-ID frame table: one entry per visible
    /// line, then an 0xFFFF end-of-frame entry.
    pub fn did_lines(&self) -> usize {
        let frame = self.regs[MAIN_WID_FRAME_REG] as usize;
        // No end marker (the table not loaded yet): 0.
        (0..2048).find(|&y| self.sram[(frame + y) & 0x7FFF] == 0xFFFF).unwrap_or(0)
    }

    /// Window-ID runs of scanline `y` (0 is the top) in the main planes, as
    /// `(first x, did)`. Register 4 points at the frame table, one line-table
    /// address per scanline; a line table lists `(x << 5) | did` entries,
    /// each starting a run, and ends at x = 0x7FF.
    pub fn main_did_runs(&self, y: usize, out: &mut Vec<(u16, u8)>) {
        self.did_runs(MAIN_WID_FRAME_REG, y, out)
    }

    /// The same for the overlay planes, from register 5's frame table.
    pub fn overlay_did_runs(&self, y: usize, out: &mut Vec<(u16, u8)>) {
        self.did_runs(OVERLAY_WID_FRAME_REG, y, out)
    }

    fn did_runs(&self, reg: usize, y: usize, out: &mut Vec<(u16, u8)>) {
        out.clear();
        let frame = self.regs[reg] as usize;
        let line = self.sram[(frame + y) & 0x7FFF];
        if line == 0xFFFF {
            return;
        }
        for k in 0..64 {
            let e = self.sram[(line as usize + k) & 0x7FFF];
            let x = e >> 5;
            if x == 0x7FF {
                break;
            }
            out.push((x, (e & 0x1F) as u8));
        }
    }

    fn sram_write(&mut self, v: u16) {
        let a = self.regs[VC3_SRAM_POINTER] as usize & 0x7FFF;
        self.sram[a] = v;
        self.regs[VC3_SRAM_POINTER] = self.regs[VC3_SRAM_POINTER].wrapping_add(1);
    }

    fn sram_read(&mut self) -> u16 {
        let a = self.regs[VC3_SRAM_POINTER] as usize & 0x7FFF;
        self.regs[VC3_SRAM_POINTER] = self.regs[VC3_SRAM_POINTER].wrapping_add(1);
        self.sram[a]
    }

    fn write(&mut self, t: Txn, d: u32) {
        match (t.crs, t.width) {
            (0, 1) => self.index = d & 0x1F,
            // Index and 16-bit value in one three-byte transfer.
            (0, _) => {
                self.index = (d >> 16) & 0x1F;
                self.regs[self.index as usize] = d as u16;
            }
            (1, _) => self.regs[self.index as usize] = d as u16,
            (3, 4) => { self.sram_write((d >> 16) as u16); self.sram_write(d as u16); }
            (3, _) => self.sram_write(d as u16),
            _ => {}
        }
    }

    fn read(&mut self, t: Txn) -> u32 {
        match (t.crs, t.width) {
            (0, _) => self.index,
            (1, _) if self.index as usize == VC3_CURRENT_LINE => {
                // Keep moving so vertical-blank polls terminate.
                self.line_counter = (self.line_counter + 1) % 1066;
                self.line_counter as u32
            }
            (1, _) => self.regs[self.index as usize] as u32,
            (3, 4) => { let hi = self.sram_read() as u32; (hi << 16) | self.sram_read() as u32 }
            (3, _) => self.sram_read() as u32,
            _ => 0,
        }
    }
}

/// The whole bus: all devices of one board. Plain data, valid zeroed; call
/// `init` once.
#[repr(C)]
pub struct Dcb {
    pub cmap: [Cmap; 2],
    pub dac: Dac,
    pub xmap: Xmap,
    pub vc3: Vc3,
    bdvers: [u32; 2],
    bc1: u32,
    pub dcbctrl: [u32; 16],
    /// Devices not modelled, by (device, select).
    other: [[u32; 8]; 32],
}

impl Dcb {
    /// `bdvers` are the two board-version bytes; `cmap_rev` the colormaps'
    /// revision registers.
    pub fn init(&mut self, bdvers: [u32; 2], cmap_rev: [u32; 2]) {
        self.cmap[0].rev = cmap_rev[0];
        self.cmap[1].rev = cmap_rev[1];
        self.dac.init();
        self.bdvers = bdvers;
    }

    pub fn write_dcbctrl(&mut self, dev: usize, val: u32) {
        if dev < 16 {
            self.dcbctrl[dev] = val;
        }
    }

    pub fn read_dcbctrl(&self, dev: usize) -> u32 {
        if dev < 16 {
            self.dcbctrl[dev]
        } else {
            0
        }
    }

    pub fn write(&mut self, t: Txn, d: u32) {
        match t.dev {
            DEV_CMAP_ALL => { self.cmap[0].write(t, d); self.cmap[1].write(t, d); }
            DEV_CMAP0 => self.cmap[0].write(t, d),
            DEV_CMAP1 => self.cmap[1].write(t, d),
            DEV_DAC => self.dac.write(t, d),
            DEV_XMAP => self.xmap.write(t, d),
            DEV_VC3 => self.vc3.write(t, d),
            DEV_BDVERS => {
                if t.crs == 1 { self.bc1 = d; }
            }
            DEV_I2C => {}
            dev => self.other[dev as usize & 31][t.crs as usize] = d,
        }
    }

    pub fn read(&mut self, t: Txn) -> u32 {
        match t.dev {
            DEV_CMAP_ALL | DEV_CMAP0 => self.cmap[0].read(t),
            DEV_CMAP1 => self.cmap[1].read(t),
            DEV_DAC => self.dac.read(t),
            DEV_XMAP => self.xmap.read(t),
            DEV_VC3 => self.vc3.read(t),
            DEV_BDVERS => self.bdvers.get(t.crs as usize).copied().unwrap_or(0),
            // No flat panel: the I2C controller reads back nothing.
            DEV_I2C => 0,
            dev => self.other[dev as usize & 31][t.crs as usize],
        }
    }
}
