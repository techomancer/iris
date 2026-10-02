//! The display control bus (DCB): the board's slow side bus to its video
//! chips, reached through a 32 KB window at slot offset `0x60000`.
//!
//! Each device owns a 1 KB window (`0x60000 + dev * 0x400`), and the address
//! within it encodes the transaction: bits 9:7 select the chip's register
//! (its "CRS" line), bits 4:3 the transfer width in bytes (0 = 4), bit 5 asks
//! the chip to increment its register select, bit 6 packs data. Data rides in
//! the most significant bytes of a 32-bit bus access, so a one-byte register
//! read with a word load comes back in bits 31:24.

use std::collections::HashMap;

/// Device numbers on the bus.
pub const DEV_CMAP_ALL: u32 = 3;
pub const DEV_CMAP0: u32 = 4;
pub const DEV_CMAP1: u32 = 5;
pub const DEV_DAC: u32 = 6;
pub const DEV_XMAP: u32 = 7;
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
pub struct Cmap {
    pub pal: Vec<u32>,
    addr: u32,
    rev: u32,
    cmd: u32,
}

impl Cmap {
    fn new(rev: u32) -> Self {
        Cmap { pal: vec![0; 8192], addr: 0, rev, cmd: 0 }
    }

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
pub struct Dac {
    addr: u32,
    regs: HashMap<u32, u32>,
    pub gamma: Vec<[u16; 3]>,
    gamma_comp: usize,
    mode: u32,
}

/// DAC register: the pixel read mask; zero blanks the screen.
pub const DAC_PIXMASK: u32 = 4;

impl Dac {
    fn new() -> Self {
        let gamma = (0..256).map(|i| { let v = (i << 2) as u16; [v, v, v] }).collect();
        Dac { addr: 0, regs: HashMap::new(), gamma, gamma_comp: 0, mode: 0 }
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
            2 => { self.regs.insert(self.addr, d & 0xFF); }
            3 => self.mode = d,
            _ => {}
        }
    }

    fn read(&self, t: Txn) -> u32 {
        match t.crs {
            0 => self.addr,
            2 => self.regs.get(&self.addr).copied().unwrap_or(0),
            3 => self.mode,
            _ => 0,
        }
    }

    pub fn pixmask(&self) -> u32 {
        self.regs.get(&DAC_PIXMASK).copied().unwrap_or(0xFF)
    }
}

/// The XMAP: the pixel processors' display side (display modes per window ID,
/// buffer selects, scanout pointers), reached as an index register (`INDEX`)
/// plus register files on the selects after it.
pub struct Xmap {
    index: u32,
    regs: HashMap<(u32, u32), u32>,
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
    /// The raster engine to pixel processor link.
    pub const RE_RAC: u32 = 7;
}

/// `CONFIG` index of the byte whose bit 3 makes the index auto-increment.
const XMAP_CONFIG_BYTE: u32 = 1;
const XMAP_AUTOINC: u32 = 0x08;
/// `CONFIG` index 4: the pixel processor revision.
const XMAP_REV_INDEX: u32 = 4;
/// `CONFIG` index read before each display-mode write.
const XMAP_MODE_ROOM_INDEX: u32 = 8;

impl Xmap {
    fn new() -> Self {
        Xmap { index: 0, regs: HashMap::new() }
    }

    fn autoinc(&mut self, t: Txn) {
        let cfg = self.regs.get(&(xmap::CONFIG, XMAP_CONFIG_BYTE)).copied().unwrap_or(0);
        if t.crs >= xmap::BUF_SELECT && cfg & XMAP_AUTOINC != 0 {
            self.index = self.index.wrapping_add(t.width);
        }
    }

    fn write(&mut self, t: Txn, d: u32) {
        match t.crs {
            xmap::PP1SELECT => { self.regs.insert((xmap::PP1SELECT, 0), d); }
            xmap::INDEX => self.index = d,
            crs => {
                self.regs.insert((crs, self.index), d);
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
            xmap::CONFIG if self.index == XMAP_REV_INDEX => 1,
            // Polled nonzero before every display-mode write: room for a
            // mode update.
            xmap::CONFIG if self.index == XMAP_MODE_ROOM_INDEX => 0x10,
            crs => self.regs.get(&(crs, self.index)).copied().unwrap_or(0),
        };
        if t.crs >= xmap::CONFIG { self.autoinc(t); }
        v
    }

    /// Display mode of overlay window ID `did` (`OVERLAY_MODE`, index `did * 4`);
    /// 0 is "overlay off".
    pub fn overlay_mode(&self, did: u32) -> u32 {
        self.regs.get(&(xmap::OVERLAY_MODE, (did & 0x1F) << 2)).copied().unwrap_or(0)
    }

    /// Colormap address of cursor colour 0: the config register (`CONFIG`,
    /// index 0) holds it divided by four.
    pub fn cursor_cmap_base(&self) -> usize {
        (self.regs.get(&(xmap::CONFIG, 0)).copied().unwrap_or(0) as usize & 0x7FF) << 2
    }

    /// Display mode of window ID `did` (`MAIN_MODE`, index `did * 4`).
    pub fn main_mode(&self, did: u32) -> u32 {
        self.regs.get(&(xmap::MAIN_MODE, did << 2)).copied().unwrap_or(0)
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

pub struct Vc3 {
    index: u32,
    pub regs: [u16; 32],
    pub sram: Vec<u16>,
    line_counter: u16,
}

pub const VC3_SRAM_POINTER: usize = 7;
const VC3_CURRENT_LINE: usize = 0xB;

impl Vc3 {
    fn new() -> Self {
        Vc3 { index: 0, regs: [0; 32], sram: vec![0; 0x8000], line_counter: 0 }
    }

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

/// The whole bus: all devices of one board.
pub struct Dcb {
    pub cmap: [Cmap; 2],
    pub dac: Dac,
    pub xmap: Xmap,
    pub vc3: Vc3,
    bdvers: [u32; 2],
    bc1: u32,
    other: HashMap<(u32, u32), u32>,
}

impl Dcb {
    /// `bdvers` are the two board-version bytes; `cmap_rev` the colormaps'
    /// revision registers.
    pub fn new(bdvers: [u32; 2], cmap_rev: [u32; 2]) -> Self {
        Dcb {
            cmap: [Cmap::new(cmap_rev[0]), Cmap::new(cmap_rev[1])],
            dac: Dac::new(),
            xmap: Xmap::new(),
            vc3: Vc3::new(),
            bdvers,
            bc1: 0,
            other: HashMap::new(),
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
            dev => { self.other.insert((dev, t.crs), d); }
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
            dev => self.other.get(&(dev, t.crs)).copied().unwrap_or(0),
        }
    }
}
