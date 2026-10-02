//! The raster subsystem: the raster engine's register file, its indirect
//! device space, pixel transfers, and the framebuffer it draws into.
//!
//! Registers are numbered 0..0x3FF. A register write may carry an "execute"
//! flag, which runs the primitive held in the instruction register (IR) once
//! the write has landed.
//!
//! Coordinates: primitives give block corners in window coordinates. The
//! window origin (`xywin`: y in the high half, x in the low) is added, and
//! with Y-flip set in `config` y runs downward from it. The PROM draws with no
//! origin and no flip; the X server sets the origin to the top row and flips,
//! so it draws top-down. The framebuffer has row 0 at the bottom.
//!
//! A block runs one of several ways, chosen by the fill mode's block type:
//! fill it (fast fill uses the fill colour registers, others the red
//! iterator), stipple it with character data, or move pixels in or out of it
//! (by PIO through the character registers, or by DMA).

use std::collections::{HashMap, VecDeque};

pub const WIDTH: usize = 1280;
pub const HEIGHT: usize = 1024;

/// Raster registers. Those OpenBSD's impact(4) driver also uses carry its
/// names; the rest are named for what they do here.
pub mod reg {
    /// The instruction register: the primitive a write with "execute" runs.
    pub const IR: u32 = 0x013;
    pub const LINE_START: u32 = 0x040;
    pub const LINE_END: u32 = 0x041;
    pub const IR_ALIAS: u32 = 0x045;
    pub const BLOCKXYSTARTI: u32 = 0x046;
    pub const BLOCKXYENDI: u32 = 0x047;
    /// Packed RGB colour for character and line drawing in RGB modes.
    pub const PACKEDCOLOR: u32 = 0x05B;
    pub const RED: u32 = 0x05C;
    pub const CHAR_H: u32 = 0x070;
    pub const CHAR_L: u32 = 0x071;
    pub const XFRCONTROL: u32 = 0x102;
    pub const FILLMODE: u32 = 0x110;
    pub const CONFIG: u32 = 0x112;
    pub const XYWIN: u32 = 0x115;
    /// The clip rectangle: x and y ranges, each `min << 16 | max`, and its
    /// control (bit 0 enable, bit 4 keep the inside rather than the outside).
    pub const CLIP_X: u32 = 0x147;
    pub const CLIP_Y: u32 = 0x148;
    pub const CLIP_MODE: u32 = 0x14F;
    pub const XFRSIZE: u32 = 0x153;
    pub const XFRMODE: u32 = 0x159;
    pub const LINE_STIPPLE: u32 = 0x15A;
    /// The indirect device space: an address, then its data.
    pub const INDIRECT_ADDR: u32 = 0x15C;
    pub const INDIRECT_DATA: u32 = 0x15D;
    pub const STATUS: u32 = 0x15E;
    pub const PP1FILLMODE: u32 = 0x161;
    /// Plane write mask (low planes, buffer A).
    pub const COLORMASKLSBSA: u32 = 0x163;
    pub const DRBPOINTERS: u32 = 0x16D;
    /// Fast-fill colour: one 12-bit component each, or the index in R.
    pub const FILL_COLOR_R: u32 = 0x176;
    pub const FILL_COLOR_G: u32 = 0x177;
    pub const FILL_COLOR_B: u32 = 0x178;
}

/// IR opcodes: a line between two points, and a block (rectangle).
const OP_LINE: u32 = 0x5;
const OP_BLOCK: u32 = 0x8;
/// Fill mode: lines follow the 32-bit line stipple pattern.
const FILL_LINE_STIPPLE: u32 = 1 << 5;
/// Status: command FIFO empty, engine and pixel processors idle, revision 1.
const STATUS_IDLE: u32 = 0x100 | (1 << 4);
/// Config: Y-flip.
const CONFIG_YFLIP: u32 = 1 << 3;
/// Fill mode: fast fill (solid, from the fill colour registers).
const FILL_FAST: u32 = 1 << 20;
/// Pixel processor fill mode used when drawing window IDs, which live in
/// their own planes, not the colour planes.
const PP1_DRAW_CID: u32 = 0x14_2600;
/// Scanout pointers: the low nine bits say which planes are drawn, and this
/// value means the overlay planes (the main planes read 0x240).
const DRB_PLANES: u32 = 0x1FF;
const DRB_OVERLAY: u32 = 0x1C0;

/// Block types (fill mode bits 24:22).
mod block {
    pub const NORMAL: u32 = 1;
    pub const PIO_READ: u32 = 2;
    pub const PIO_WRITE: u32 = 3;
    pub const DMA_READ: u32 = 4;
    pub const DMA_WRITE: u32 = 5;
}

/// Whether the pixel processors' pixel type (fill mode bits 10:8) is an RGB
/// one; the others are colour index.
fn rgb_pixtype(pp1fillmode: u32) -> bool {
    matches!((pp1fillmode >> 8) & 7, 0 | 1 | 4)
}

/// Pixel processor logic op (fill mode bit 2 enables it; bits 29:26 hold
/// the X11 function number) applied to source `s` and destination `d`.
fn logic_op(op: u32, s: u32, d: u32) -> u32 {
    match op & 0xF {
        0x0 => 0,
        0x1 => s & d,
        0x2 => s & !d,
        0x3 => s,
        0x4 => !s & d,
        0x5 => d,
        0x6 => s ^ d,
        0x7 => s | d,
        0x8 => !(s | d),
        0x9 => !(s ^ d),
        0xA => !d,
        0xB => s | !d,
        0xC => !s,
        0xD => !s | d,
        0xE => !(s & d),
        _ => !0,
    }
}
const PP1_LOGIC_OP_ENABLE: u32 = 1 << 2;

/// Framebuffer pixels: colour indices as they are, RGB as `0x00BBGGRR` with
/// eight bits per component.
fn pack_rgb(r: u32, g: u32, b: u32) -> u32 {
    (r & 0xFF) | (g & 0xFF) << 8 | (b & 0xFF) << 16
}

/// A host pixel of transfer format (PixelFormat, CompType) to a framebuffer
/// pixel, and back. Only the RGB formats convert.
fn from_host(format: (u32, u32), v: u32) -> u32 {
    let c4 = |s: u32| ((v >> s) & 0xF) * 0x11;
    let c5 = |s: u32| ((v >> s) & 0x1F) << 3 | ((v >> s) & 0x1F) >> 2;
    match format {
        (8, 8) => pack_rgb(c4(0), c4(4), c4(8)),
        (8, 10) => pack_rgb(c5(0), c5(5), c5(10)),
        (8, 0) => v & 0xFF_FFFF,
        (0, 1) => v & 0xFFF,
        _ => v,
    }
}

fn to_host(format: (u32, u32), v: u32) -> u32 {
    let c = |s: u32| (v >> s) & 0xFF;
    match format {
        (8, 8) => c(0) / 0x11 | (c(8) / 0x11) << 4 | (c(16) / 0x11) << 8,
        (8, 10) => c(0) >> 3 | (c(8) >> 3) << 5 | (c(16) >> 3) << 10,
        _ => v,
    }
}

fn signed16(v: u32) -> i32 {
    v as u16 as i16 as i32
}

/// A block, in window coordinates, with its colour.
#[derive(Clone, Copy)]
struct Block {
    xs: i32,
    ys: i32,
    xe: i32,
    ye: i32,
    color: u32,
}

impl Block {
    fn dx(&self) -> i32 {
        if self.xe < self.xs { -1 } else { 1 }
    }
    fn dy(&self) -> i32 {
        if self.ye < self.ys { -1 } else { 1 }
    }
    fn rows(&self) -> i32 {
        (self.ye - self.ys).abs() + 1
    }
    fn cols(&self) -> i32 {
        (self.xe - self.xs).abs() + 1
    }
}

/// A character block being filled with stipple data, one row at a time.
#[derive(Clone, Copy)]
struct Stipple {
    block: Block,
    col: i32,
    row: i32,
}

/// An armed pixel transfer.
struct Xfer {
    block: Block,
    read: bool,
    /// Pixels per line, bytes per pixel, and (PixelFormat, CompType).
    width: u32,
    bpp: u32,
    format: (u32, u32),
    begin_skip: u32,
    stride_skip: u32,
    /// PIO write stream: the line being assembled, its byte offset within
    /// its first doubleword, the bytes collected, and filler still to skip.
    line: u32,
    line_begin: u32,
    pending: Vec<u8>,
    skip: u32,
    /// PIO read: doublewords waiting to be read, and the low half of the one
    /// the last high read took.
    out: VecDeque<u64>,
    out_lo: u32,
}

impl Xfer {
    fn line_bytes(&self) -> u32 {
        self.width * self.bpp
    }

    /// Where the line after one starting at `b` starts within its first
    /// doubleword.
    fn next_begin(&self, b: u32) -> u32 {
        (b + self.line_bytes() + self.stride_skip) & 7
    }
}

/// Bytes per pixel for an xfrmode (PixelFormat, PixelCompType) pair.
fn bytes_per_pixel(xfrmode: u32) -> u32 {
    match ((xfrmode >> 4) & 0xF, xfrmode & 0xF) {
        (0, 0) => 1,
        (0, 1) | (8, 8) | (8, 10) => 2,
        (8, 0) => 4,
        (7, 1) => 6,
        _ => 1,
    }
}

pub struct Raster {
    regs: Vec<u32>,
    device: HashMap<u32, u32>,
    /// Colour planes, one value per pixel, row 0 at the bottom. In
    /// colour-index modes the low byte is the index.
    pub fb: Vec<u32>,
    /// Overlay planes (kept, not yet displayed).
    pub overlay: Vec<u32>,
    pending: Option<Block>,
    /// Bumped for every new pending block, so the refresh thread can tell a
    /// block that has waited a whole frame (see `flush_if_stale`).
    pending_id: u64,
    stipple: Option<Stipple>,
    xfer: Option<Xfer>,
    /// Fill modes seen with a block, for bring-up logging.
    pub fillmodes_seen: Vec<u32>,
}

impl Default for Raster {
    fn default() -> Self {
        Raster {
            regs: vec![0; 0x400],
            device: HashMap::new(),
            fb: vec![0; WIDTH * HEIGHT],
            overlay: vec![0; WIDTH * HEIGHT],
            pending: None,
            pending_id: 0,
            stipple: None,
            xfer: None,
            fillmodes_seen: Vec::new(),
        }
    }
}

impl Raster {
    pub fn read(&self, r: u32) -> u32 {
        match r & 0x3FF {
            reg::STATUS => STATUS_IDLE,
            reg::INDIRECT_DATA => {
                self.device.get(&self.regs[reg::INDIRECT_ADDR as usize]).copied().unwrap_or(0)
            }
            r => self.regs[r as usize],
        }
    }

    fn reg(&self, r: u32) -> u32 {
        self.regs[r as usize]
    }

    /// Write register `r`; `exec` runs the primitive afterwards. Returns true
    /// when the framebuffer changed.
    pub fn write(&mut self, r: u32, val: u32, exec: bool) -> bool {
        let r = r & 0x3FF;
        self.regs[r as usize] = val;
        match r {
            reg::INDIRECT_DATA => {
                self.device.insert(self.regs[reg::INDIRECT_ADDR as usize], val);
            }
            reg::IR_ALIAS => self.regs[reg::IR as usize] = val,
            reg::XFRCONTROL if val == 0 => self.xfer = None,
            _ => {}
        }
        if !exec {
            return false;
        }
        let pio_write = self.xfer.as_ref().is_some_and(|x| !x.read);
        match r {
            reg::CHAR_L if pio_write => {
                let dw = ((self.reg(reg::CHAR_H) as u64) << 32) | val as u64;
                self.pio_write(dw)
            }
            reg::CHAR_H if pio_write => self.pio_write((val as u64) << 32),
            reg::CHAR_H => self.stipple_bits((val as u64) << 32, 32),
            reg::CHAR_L => {
                let bits = ((self.reg(reg::CHAR_H) as u64) << 32) | val as u64;
                self.stipple_bits(bits, 64)
            }
            _ => self.execute(),
        }
    }

    /// Whether pixels are RGB rather than colour indices: an RGB pixel type,
    /// or a write mask of exactly the 24 RGB planes (24-bit windows draw
    /// with other pixel types; colour-index drawing masks 8 or 12 planes, or
    /// all 32 when the PROM and kernel draw).
    fn rgb_mode(&self) -> bool {
        rgb_pixtype(self.reg(reg::PP1FILLMODE)) || self.reg(reg::COLORMASKLSBSA) == 0xFF_FFFF
    }

    /// Window coordinates to framebuffer coordinates.
    fn to_fb(&self, x: i32, y: i32) -> (i32, i32) {
        let win = self.reg(reg::XYWIN);
        let (ox, oy) = (signed16(win), signed16(win >> 16));
        if self.reg(reg::CONFIG) & CONFIG_YFLIP != 0 {
            (ox + x, oy - y)
        } else {
            (ox + x, oy + y)
        }
    }

    /// Whether a framebuffer pixel may be written: on screen, and inside
    /// the clip rectangle when it is enabled (bit 4 chooses inside or outside).
    fn visible(&self, x: i32, y: i32) -> bool {
        if !(0..WIDTH as i32).contains(&x) || !(0..HEIGHT as i32).contains(&y) {
            return false;
        }
        let clip = self.reg(reg::CLIP_MODE);
        if clip & 1 != 0 {
            let (mx, my) = (self.reg(reg::CLIP_X), self.reg(reg::CLIP_Y));
            let inside = (signed16(mx >> 16)..=signed16(mx)).contains(&x)
                && (signed16(my >> 16)..=signed16(my)).contains(&y);
            if inside != (clip & 0x10 != 0) {
                return false;
            }
        }
        true
    }

    /// Store `v` at framebuffer `(x, y)` in the planes the pixel processors
    /// are drawing to. Window-ID drawing is dropped.
    fn put(&mut self, x: i32, y: i32, v: u32) {
        if self.reg(reg::PP1FILLMODE) == PP1_DRAW_CID || !self.visible(x, y) {
            return;
        }
        let i = y as usize * WIDTH + x as usize;
        let pp1 = self.reg(reg::PP1FILLMODE);
        // A write through all planes (window moves copy the screen that way,
        // 24 bits a pixel) keeps the whole value; so does RGB.
        let wide = self.rgb_mode() || self.reg(reg::COLORMASKLSBSA) == 0xFFFF_FFFF;
        let plane = if self.reg(reg::DRBPOINTERS) & DRB_PLANES == DRB_OVERLAY {
            &mut self.overlay
        } else {
            &mut self.fb
        };
        plane[i] = if pp1 & PP1_LOGIC_OP_ENABLE != 0 {
            let width = if wide { 0xFF_FFFF } else { 0xFFF };
            logic_op(pp1 >> 26, v, plane[i]) & width
        } else {
            v
        };
    }

    fn get(&self, x: i32, y: i32) -> u32 {
        if !(0..WIDTH as i32).contains(&x) || !(0..HEIGHT as i32).contains(&y) {
            return 0;
        }
        let i = y as usize * WIDTH + x as usize;
        if self.reg(reg::DRBPOINTERS) & DRB_PLANES == DRB_OVERLAY { self.overlay[i] } else { self.fb[i] }
    }

    /// Block pixel `(col, row)` in framebuffer coordinates.
    fn block_px(&self, b: &Block, col: i32, row: i32) -> (i32, i32) {
        self.to_fb(b.xs + col * b.dx(), b.ys + row * b.dy())
    }

    fn current_block(&self) -> Block {
        let s = self.reg(reg::BLOCKXYSTARTI);
        let e = self.reg(reg::BLOCKXYENDI);
        // Colour index modes: the index, from the fill colour register for
        // fast fills and from the red iterator (12 fraction bits) otherwise.
        // RGB modes: three 12-bit components for fast fills, a packed
        // 8-8-8 colour otherwise.
        let rgb = self.rgb_mode();
        let fast = self.reg(reg::FILLMODE) & FILL_FAST != 0;
        let color = match (rgb, fast) {
            (false, true) => self.reg(reg::FILL_COLOR_R),
            (false, false) => (self.reg(reg::RED) >> 12) & 0xFFF,
            (true, true) => pack_rgb(
                self.reg(reg::FILL_COLOR_R) >> 4,
                self.reg(reg::FILL_COLOR_G) >> 4,
                self.reg(reg::FILL_COLOR_B) >> 4,
            ),
            (true, false) => self.reg(reg::PACKEDCOLOR) & 0xFF_FFFF,
        };
        Block { xs: signed16(s >> 16), ys: signed16(s), xe: signed16(e >> 16), ye: signed16(e), color }
    }

    /// Run the primitive in the IR.
    fn execute(&mut self) -> bool {
        let changed = self.flush_pending();
        match self.reg(reg::IR) & 0xF {
            OP_BLOCK => {}
            OP_LINE => return self.line() | changed,
            _ => return changed,
        }
        let fm = self.reg(reg::FILLMODE);
        if !self.fillmodes_seen.contains(&fm) && self.fillmodes_seen.len() < 32 {
            self.fillmodes_seen.push(fm);
        }
        let b = self.current_block();
        // A new block ends whatever the last one was doing; a write transfer
        // in particular is never disarmed explicitly.
        self.stipple = None;
        self.xfer = None;
        let kind = (fm >> 22) & 7;
        if fm & FILL_FAST != 0 {
            self.fill(&b);
            return true;
        }
        match kind {
            block::NORMAL => {
                // Character block: filled by the stipple data that follows.
                self.stipple = Some(Stipple { block: b, col: 0, row: 0 });
                changed
            }
            block::PIO_READ | block::PIO_WRITE | block::DMA_READ | block::DMA_WRITE => {
                let read = kind == block::PIO_READ || kind == block::DMA_READ;
                self.arm_transfer(b, read, kind == block::PIO_READ);
                changed
            }
            _ => {
                // A block followed by character data is a stipple; anything
                // else, a fill.
                self.pending_id += 1;
                self.pending = Some(b);
                changed
            }
        }
    }

    /// A line from the start to the end point, both included, in the current
    /// colour; with line stipple on, pixel `k` is drawn only where bit
    /// `31 - k % 32` of the pattern is set.
    fn line(&mut self) -> bool {
        let s = self.reg(reg::LINE_START);
        let e = self.reg(reg::LINE_END);
        let (mut x, mut y) = (signed16(s >> 16), signed16(s));
        let (x1, y1) = (signed16(e >> 16), signed16(e));
        let color = self.current_block().color;
        let stipple = (self.reg(reg::FILLMODE) & FILL_LINE_STIPPLE != 0).then(|| self.reg(reg::LINE_STIPPLE));
        let (dx, dy) = ((x1 - x).abs(), -(y1 - y).abs());
        let (sx, sy) = (if x < x1 { 1 } else { -1 }, if y < y1 { 1 } else { -1 });
        let mut err = dx + dy;
        let mut k = 0u32;
        loop {
            if stipple.map_or(true, |p| p & (1 << (31 - k % 32)) != 0) {
                let (fx, fy) = self.to_fb(x, y);
                self.put(fx, fy, color);
            }
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
            k += 1;
        }
        true
    }

    fn fill(&mut self, b: &Block) {
        for row in 0..b.rows() {
            for col in 0..b.cols() {
                let (x, y) = self.block_px(b, col, row);
                self.put(x, y, b.color);
            }
        }
    }

    /// Called once per displayed frame: a block that was already pending at
    /// the previous frame never got character data, so draw it as a fill.
    pub fn flush_if_stale(&mut self, last_seen: &mut u64) -> bool {
        if self.pending.is_none() {
            return false;
        }
        if *last_seen == self.pending_id {
            return self.flush_pending();
        }
        *last_seen = self.pending_id;
        false
    }

    /// A pending block that got no character data is a solid fill.
    fn flush_pending(&mut self) -> bool {
        let Some(b) = self.pending.take() else { return false };
        self.fill(&b);
        true
    }

    /// Consume `n` stipple bits (most significant first) into the current
    /// character block: 1 bits take the colour, 0 bits leave the pixel. A row
    /// ends at the block's edge, discarding the rest of the chunk.
    fn stipple_bits(&mut self, bits: u64, n: u32) -> bool {
        if let Some(b) = self.pending.take() {
            self.stipple = Some(Stipple { block: b, col: 0, row: 0 });
        }
        let Some(mut s) = self.stipple else { return false };
        if s.row >= s.block.rows() {
            return false;
        }
        for i in 0..n {
            if bits & (1u64 << (63 - i)) != 0 {
                let (x, y) = self.block_px(&s.block, s.col, s.row);
                self.put(x, y, s.block.color);
            }
            s.col += 1;
            if s.col >= s.block.cols() {
                s.col = 0;
                s.row += 1;
                break;
            }
        }
        self.stipple = Some(s);
        true
    }

    // ---- pixel transfers ----

    fn arm_transfer(&mut self, block: Block, read: bool, pio_read: bool) {
        let mode = self.reg(reg::XFRMODE);
        let begin = (mode >> 8) & 7;
        let mut x = Xfer {
            block,
            read,
            width: self.reg(reg::XFRSIZE) & 0xFFFF,
            bpp: bytes_per_pixel(mode),
            format: ((mode >> 4) & 0xF, mode & 0xF),
            begin_skip: begin,
            stride_skip: (mode >> 14) & 0x1FF,
            line: 0,
            line_begin: begin,
            pending: Vec::new(),
            skip: begin,
            out: VecDeque::new(),
            out_lo: 0,
        };
        if pio_read {
            x.out = self.pio_read_stream(&x);
        }
        self.xfer = Some(x);
    }

    /// `Some(read)` while a transfer is armed.
    pub fn transfer_armed(&self) -> Option<bool> {
        self.xfer.as_ref().map(|x| x.read)
    }

    /// Lines and bytes per line of the armed transfer.
    pub fn transfer_shape(&self) -> Option<(u32, u32)> {
        self.xfer.as_ref().map(|x| (x.block.rows() as u32, x.line_bytes()))
    }

    fn put_line(&mut self, x: &Xfer, line: u32, bytes: &[u8]) {
        let bpp = x.bpp as usize;
        for (k, px) in bytes.chunks(bpp).enumerate().take(x.width as usize) {
            // Big-endian bytes; pixels wider than 32 bits keep their low word.
            let v = px.iter().fold(0u64, |a, &b| (a << 8) | b as u64) as u32;
            let (fx, fy) = self.block_px(&x.block, k as i32, line as i32);
            self.put(fx, fy, from_host(x.format, v));
        }
    }

    fn get_line(&self, x: &Xfer, line: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.line_bytes() as usize);
        for k in 0..x.width as i32 {
            let (fx, fy) = self.block_px(&x.block, k, line as i32);
            let v = to_host(x.format, self.get(fx, fy)) as u64;
            for i in (0..x.bpp).rev() {
                out.push((v >> (8 * i)) as u8);
            }
        }
        out
    }

    /// A PIO write doubleword. Each line starts in a fresh doubleword, after
    /// its begin offset; the rest of a line's last doubleword is dropped.
    fn pio_write(&mut self, dw: u64) -> bool {
        let Some(mut x) = self.xfer.take() else { return false };
        let mut changed = false;
        for i in 0..8 {
            if x.line >= x.block.rows() as u32 {
                break;
            }
            if x.skip > 0 {
                x.skip -= 1;
                continue;
            }
            x.pending.push((dw >> (56 - 8 * i)) as u8);
            if x.pending.len() as u32 == x.line_bytes() {
                let bytes = std::mem::take(&mut x.pending);
                self.put_line(&x, x.line, &bytes);
                changed = true;
                x.line += 1;
                x.line_begin = x.next_begin(x.line_begin);
                x.skip = x.line_begin;
                break;
            }
        }
        self.xfer = Some(x);
        changed
    }

    /// The whole PIO read stream: per line, filler up to its begin offset, the
    /// pixels, and padding to the doubleword.
    fn pio_read_stream(&self, x: &Xfer) -> VecDeque<u64> {
        let mut out = VecDeque::new();
        let mut b = x.begin_skip;
        for line in 0..x.block.rows() as u32 {
            let mut bytes = vec![0u8; b as usize];
            bytes.extend(self.get_line(x, line));
            while bytes.len() % 8 != 0 {
                bytes.push(0);
            }
            for c in bytes.chunks(8) {
                out.push_back(c.iter().fold(0u64, |a, &v| (a << 8) | v as u64));
            }
            b = x.next_begin(b);
        }
        out
    }

    /// PIO read, high half: takes the next doubleword.
    pub fn pio_read_hi(&mut self) -> u32 {
        let Some(x) = self.xfer.as_mut() else { return 0 };
        let dw = x.out.pop_front().unwrap_or(0);
        x.out_lo = dw as u32;
        (dw >> 32) as u32
    }

    /// PIO read, low half of the doubleword the last high read took.
    pub fn pio_read_lo(&self) -> u32 {
        self.xfer.as_ref().map(|x| x.out_lo).unwrap_or(0)
    }

    /// DMA into the armed block: line `line`'s pixel bytes.
    pub fn dma_write_line(&mut self, line: u32, bytes: &[u8]) {
        if let Some(x) = self.xfer.take() {
            self.put_line(&x, line, bytes);
            self.xfer = Some(x);
        }
    }

    /// DMA out of the armed block: line `line`'s pixel bytes.
    pub fn dma_read_line(&self, line: u32) -> Vec<u8> {
        self.xfer.as_ref().map(|x| self.get_line(x, line)).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(r: &Raster, x: usize, y_top: usize) -> u32 {
        r.fb[(HEIGHT - 1 - y_top) * WIDTH + x]
    }

    /// The X server's setup: window origin at the top row, Y-flip on, so
    /// block coordinates are X's top-down ones.
    fn x_server() -> Raster {
        let mut r = Raster::default();
        r.write(reg::CONFIG, 0xCAC, false);
        r.write(reg::XYWIN, 1023 << 16, false);
        r.write(reg::PP1FILLMODE, 0x0C00_4504, false);
        r
    }

    fn block(r: &mut Raster, x0: u32, y0: u32, x1: u32, y1: u32) {
        r.write(reg::IR_ALIAS, 0x18, false);
        r.write(reg::BLOCKXYSTARTI, x0 << 16 | y0, false);
        r.write(reg::BLOCKXYENDI, x1 << 16 | y1, true);
    }

    #[test]
    fn fast_fill_lands_top_down_with_yflip() {
        let mut r = x_server();
        r.write(reg::FILLMODE, FILL_FAST, false);
        r.write(reg::FILL_COLOR_R, 0x13, false);
        block(&mut r, 10, 20, 12, 21);
        assert_eq!(px(&r, 10, 20), 0x13);
        assert_eq!(px(&r, 12, 21), 0x13);
        assert_eq!(px(&r, 13, 21), 0);
        assert_eq!(px(&r, 10, 22), 0);
    }

    #[test]
    fn rgb_fast_fill_packs_components() {
        let mut r = x_server();
        r.write(reg::PP1FILLMODE, 3 << 26 | 0x104, false); // RGB pixel type, copy
        r.write(reg::FILLMODE, FILL_FAST, false);
        r.write(reg::FILL_COLOR_R, 0xF00, false);
        r.write(reg::FILL_COLOR_G, 0x800, false);
        r.write(reg::FILL_COLOR_B, 0x100, false);
        block(&mut r, 0, 0, 0, 0);
        assert_eq!(px(&r, 0, 0), 0x10_80F0);
    }

    #[test]
    fn pio_write_frames_each_line_in_a_fresh_doubleword() {
        let mut r = x_server();
        // Three 1-byte pixels per line, two lines, begin skip 2: line 0 is
        // bytes 2..5 of its doubleword; line 1 starts at (2 + 3) & 7 = 5.
        r.write(reg::FILLMODE, 3 << 22, false);
        r.write(reg::XFRMODE, 2 << 8, false);
        r.write(reg::XFRSIZE, 2 << 16 | 3, false);
        block(&mut r, 100, 50, 102, 51);
        assert_eq!(r.transfer_armed(), Some(false));
        r.write(reg::CHAR_H, 0x0000_0102, false);
        r.write(reg::CHAR_L, 0x03FF_FFFF, true);
        r.write(reg::CHAR_H, 0x0000_0000, false);
        r.write(reg::CHAR_L, 0x0004_0506, true);
        assert_eq!([px(&r, 100, 50), px(&r, 101, 50), px(&r, 102, 50)], [1, 2, 3]);
        assert_eq!([px(&r, 100, 51), px(&r, 101, 51), px(&r, 102, 51)], [4, 5, 6]);
    }

    #[test]
    fn a_new_block_disarms_a_finished_write_transfer() {
        let mut r = x_server();
        r.write(reg::FILLMODE, 5 << 22, false);
        r.write(reg::XFRSIZE, 1 << 16 | 1, false);
        block(&mut r, 0, 0, 0, 0);
        assert!(r.transfer_armed().is_some());
        // A glyph block: its char data must stipple, not feed the transfer.
        r.write(reg::FILLMODE, 1 << 22, false);
        r.write(reg::RED, 0x7 << 12, false);
        block(&mut r, 200, 10, 203, 10);
        assert_eq!(r.transfer_armed(), None);
        r.write(reg::CHAR_H, 0xA000_0000, true);
        assert_eq!([px(&r, 200, 10), px(&r, 201, 10), px(&r, 202, 10)], [7, 0, 7]);
    }

    #[test]
    fn rgb_host_formats_round_trip() {
        for (fmt, v) in [((8, 8), 0x0ABCu32), ((8, 10), 0x7FFF), ((8, 0), 0x00C0_FFEE), ((0, 1), 0xFFF)] {
            assert_eq!(to_host(fmt, from_host(fmt, v)), v, "{fmt:?}");
        }
        assert_eq!(from_host((8, 8), 0x0F0), pack_rgb(0, 0xFF, 0));
        // 8-8-8 host pixels are X pixel values of the visuals, red in 7:0.
        assert_eq!(from_host((8, 0), 0x00_00FF), pack_rgb(0xFF, 0, 0));
    }

    #[test]
    fn a_24_plane_mask_means_rgb_fills() {
        let mut r = x_server();
        r.write(reg::PP1FILLMODE, 0x0C00_6304, false);
        r.write(reg::COLORMASKLSBSA, 0xFF_FFFF, false);
        r.write(reg::FILLMODE, FILL_FAST, false);
        r.write(reg::FILL_COLOR_R, 0x380, false);
        r.write(reg::FILL_COLOR_G, 0x8E0, false);
        r.write(reg::FILL_COLOR_B, 0x8E0, false);
        block(&mut r, 1, 1, 1, 1);
        assert_eq!(px(&r, 1, 1), 0x8E_8E38);
    }

    #[test]
    fn an_all_planes_copy_keeps_24_bit_pixels() {
        let mut r = x_server();
        // A window move's write-back: pixel type 2, every plane enabled,
        // 4-byte host pixels.
        r.write(reg::PP1FILLMODE, 0x0C00_6204, false);
        r.write(reg::COLORMASKLSBSA, 0xFFFF_FFFF, false);
        r.write(reg::FILLMODE, 5 << 22, false);
        r.write(reg::XFRMODE, 0x80, false);
        r.write(reg::XFRSIZE, 1 << 16 | 1, false);
        block(&mut r, 7, 7, 7, 7);
        r.dma_write_line(0, &[0x00, 0x50, 0x50, 0x50]);
        assert_eq!(px(&r, 7, 7), 0x50_5050);
    }

    #[test]
    fn xor_fill_toggles_and_restores() {
        let mut r = x_server();
        r.write(reg::FILLMODE, FILL_FAST, false);
        r.write(reg::FILL_COLOR_R, 0x13, false);
        block(&mut r, 5, 5, 5, 5);
        r.write(reg::PP1FILLMODE, 0x0C00_4504 & !(0xF << 26) | 6 << 26, false);
        r.write(reg::FILL_COLOR_R, 0x0F, false);
        block(&mut r, 5, 5, 5, 5);
        assert_eq!(px(&r, 5, 5), 0x13 ^ 0x0F);
        block(&mut r, 5, 5, 5, 5);
        assert_eq!(px(&r, 5, 5), 0x13);
    }

    #[test]
    fn stippled_line_skips_zero_bits() {
        let mut r = x_server();
        r.write(reg::FILLMODE, FILL_LINE_STIPPLE, false);
        r.write(reg::RED, 0x9 << 12, false);
        r.write(reg::LINE_STIPPLE, 0xAAAA_AAAA, false);
        r.write(reg::IR_ALIAS, 0x15, false);
        r.write(reg::LINE_START, 300 << 16 | 40, false);
        r.write(reg::LINE_END, 303 << 16 | 40, true);
        assert_eq!((300..304).map(|x| px(&r, x, 40)).collect::<Vec<_>>(), [9, 0, 9, 0]);
    }
}
