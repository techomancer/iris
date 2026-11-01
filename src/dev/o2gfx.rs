//! O2 graphics: GBE, the display back end, and CRIME's rendering engine.
//!
//! The O2 has no video memory. The frame buffer is ordinary RAM in 64 KB
//! tiles; GBE fetches it through a tile list and turns it into pixels, and
//! the rendering engine (RE) draws into it through its own tile tables. Both
//! are DMA masters on the memory controller, so neither sees what the CPU
//! still holds in its caches, exactly as on the hardware.
//!
//! Everything here is the plain CPU version: scan-out is a pure function of
//! RAM and the GBE registers ([`Gbe::render`]), and each RE primitive runs to
//! completion when it is started, writing RAM through [`O2Memory::write`] so
//! jitv2's page generations see it. RAM is the only store. That keeps this a
//! usable oracle for a later GPU (compute shader) version of the same thing.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::dev::crime_mem::{Draw, O2Memory, RamWindow};
use crate::snapshot::{get_field, hex_u64, load_u32_slice, toml_u32, toml_u64, u32_slice_to_toml};
use crate::traits::{BusDevice, BusRead32, BusRead64, Saveable, BUS_BUSY, BUS_OK};

/// GBE register offsets from its base, `0x1600_0000`.
pub mod gbe_reg {
    pub const CTRLSTAT: u32 = 0x00000;
    pub const DOTCLOCK: u32 = 0x00004;
    pub const I2C: u32 = 0x00008;
    pub const I2CFP: u32 = 0x00010;
    pub const DEVICE_ID: u32 = 0x00014;
    pub const VT_XY: u32 = 0x10000;
    pub const VT_XYMAX: u32 = 0x10004;
    pub const VT_FLAGS: u32 = 0x10018;
    pub const VT_INTR01: u32 = 0x10020;
    pub const VT_INTR23: u32 = 0x10024;
    pub const VT_HCMAP: u32 = 0x1003c;
    pub const VT_VCMAP: u32 = 0x10040;
    pub const OVR_WIDTH_TILE: u32 = 0x20000;
    pub const OVR_CONTROL: u32 = 0x20004;
    pub const OVR_CONTROL_SHADOW: u32 = 0x20008;
    pub const FRM_SIZE_TILE: u32 = 0x30000;
    pub const FRM_SIZE_PIXEL: u32 = 0x30004;
    pub const FRM_CONTROL: u32 = 0x30008;
    pub const FRM_CONTROL_SHADOW: u32 = 0x3000c;
    pub const DID_CONTROL: u32 = 0x40000;
    pub const DID_CONTROL_SHADOW: u32 = 0x40004;
    pub const MODE: u32 = 0x48000;
    pub const CMAP: u32 = 0x50000;
    pub const CM_FIFO: u32 = 0x58000;
    pub const GMAP: u32 = 0x60000;
    pub const CRS_POS: u32 = 0x70000;
    pub const CRS_CTL: u32 = 0x70004;
    pub const CRS_CMAP1: u32 = 0x70008;
    pub const CRS_GLYPH: u32 = 0x78000;
}

use gbe_reg::*;

/// What scan-out reads memory through: a [`ScanSnapshot`] of the pages one
/// frame needs.
pub trait ScanMem {
    fn dma_offset(&self, addr: u32) -> Option<usize>;
    fn len(&self) -> usize;
}

/// The 64 KB pages one frame of scan-out reads (tile lists, tiles, the DID
/// table), copied out of RAM under its lock so the scan itself runs without
/// it. Scanning straight out of locked RAM kept the CPU, which takes the same
/// lock for every load and store, waiting most of the time.
pub struct ScanSnapshot {
    slot: std::collections::HashMap<u32, usize>,
    bytes: Vec<u8>,
}

impl ScanMem for ScanSnapshot {
    fn dma_offset(&self, addr: u32) -> Option<usize> {
        self.slot.get(&(addr >> 16)).map(|s| s * 0x10000 + (addr & 0xffff) as usize)
    }
    fn len(&self) -> usize {
        self.bytes.len()
    }
}

/// The pages are copied as RAM keeps them, so byte `o` is at `o ^ 3`.
impl std::ops::Index<usize> for ScanSnapshot {
    type Output = u8;
    fn index(&self, o: usize) -> &u8 {
        &self.bytes[o ^ 3]
    }
}

/// Offsets 0 to the end of the video-capture block.
const GBE_WORDS: usize = 0x81000 / 4;
const VT_FREEZE: u32 = 1 << 31;
const DOTCLOCK_RUN: u32 = 1 << 20;
/// Dots the beam moves on every read of VT_XY, on top of guest time, so a
/// poll for blanking always sees it move even if guest time stands still.
const DOTS_PER_READ: u64 = 32;

struct GbeState {
    r: Vec<u32>,
    /// Guest microseconds when the beam was last unfrozen.
    t0: u64,
    /// VT_XY reads since then.
    reads: u64,
    /// Beam position (in dots since t0) when vt_crossings last looked.
    seen_dots: u64,
    /// Shadow writes waiting for vertical blanking: (live offset, value,
    /// frame number when written).
    pending: Vec<(u32, u32, u64)>,
}

pub struct Gbe {
    s: Mutex<GbeState>,
    /// Reads per register offset, for finding a driver spinning on one.
    pub hot: Mutex<BTreeMap<u32, u64>>,
    /// Guest time in microseconds (MACE UST).
    now: Arc<dyn Fn() -> u64 + Send + Sync>,
    /// The beam's clock when it should follow the CPU instead: microseconds
    /// from the instruction count (see `use_cpu_clock`).
    cpu_clock: std::sync::OnceLock<(crate::cpu::mips_core::CyclesPtr, u64)>,
    /// The control port's `pixel X Y`: the point to describe on the next
    /// frame, and what the scan-out found there.
    pub probe: Mutex<(Option<(usize, usize)>, String)>,
    /// The four vertical-timing interrupts, into CRIME's latched inputs
    /// 16..19 ([`Self::connect_irq`]).
    irq: std::sync::OnceLock<[crate::irq::IrqLine; 4]>,
}

impl Gbe {
    /// Wire the vertical-timing interrupts into CRIME.
    pub fn connect_irq(&self, crime: Arc<dyn crate::irq::InterruptController>) {
        let _ = self.irq.set(std::array::from_fn(|n| crate::irq::IrqLine::new(crime.clone(), 16 + n as u32)));
    }

    /// Raise the timing interrupts whose line the beam has crossed since the
    /// last call. The beam runs on guest time, so the machine's service
    /// thread calls this as its clock.
    pub fn service(&self) {
        let Some(lines) = self.irq.get() else { return };
        let crossed = self.vt_crossings();
        for (n, line) in lines.iter().enumerate() {
            if crossed & (1 << n) != 0 {
                line.set(true);
            }
        }
    }

    pub fn new(now: Arc<dyn Fn() -> u64 + Send + Sync>) -> Self {
        let mut r = vec![0u32; GBE_WORDS];
        // GPIO pins unconfigured (inputs), chip id 1.
        r[(CTRLSTAT / 4) as usize] = 0x03ff_ffc1;
        r[(VT_XY / 4) as usize] = VT_FREEZE;
        Self { s: Mutex::new(GbeState { r, t0: 0, reads: 0, seen_dots: 0, pending: Vec::new() }), hot: Mutex::new(BTreeMap::new()), now, cpu_clock: std::sync::OnceLock::new(), probe: Mutex::new((None, String::new())), irq: std::sync::OnceLock::new() }
    }

    /// Run the beam on the CPU's instruction count at `mhz` rather than on
    /// the host clock. The vertical-timing interrupts then come at the rate
    /// the guest would see on its own hardware: on the host clock a slow
    /// emulated CPU gets sixty a real second whatever it manages to do, and
    /// IRIX's interrupt dispatcher -- which re-reads CRIME until nothing is
    /// pending -- found GBE latched again before each pass was out and never
    /// returned.
    pub fn use_cpu_clock(&self, cycles: crate::cpu::mips_core::CyclesPtr, mhz: u32) {
        let _ = self.cpu_clock.set((cycles, mhz.max(1) as u64));
    }

    fn now(&self) -> u64 {
        match self.cpu_clock.get() {
            Some((c, mhz)) => c.get() / mhz,
            None => (self.now)(),
        }
    }

    fn get(r: &[u32], off: u32) -> u32 {
        r.get((off / 4) as usize).copied().unwrap_or(0)
    }

    /// The dot clock in Hz: 20 MHz x (M+1) / ((N+1) x 2^P).
    fn dot_hz(r: &[u32]) -> u64 {
        let d = Self::get(r, DOTCLOCK);
        let m = (d & 0xff) as u64 + 1;
        let n = ((d >> 8) & 0x3f) as u64 + 1;
        let p = (d >> 14) & 3;
        20_000_000 * m / (n << p)
    }

    fn beam(&self, s: &mut GbeState) -> u32 {
        let xy = Self::get(&s.r, VT_XY);
        if xy & VT_FREEZE != 0 {
            return VT_FREEZE;
        }
        let max = Self::get(&s.r, VT_XYMAX);
        let (maxx, maxy) = ((max & 0xfff) as u64, ((max >> 12) & 0xfff) as u64);
        if Self::get(&s.r, DOTCLOCK) & DOTCLOCK_RUN == 0 || maxx == 0 || maxy == 0 {
            return xy & 0x00ff_ffff;
        }
        s.reads += 1;
        let us = self.now().saturating_sub(s.t0);
        let dots = us * Self::dot_hz(&s.r) / 1_000_000 + s.reads * DOTS_PER_READ;
        let x = dots % (maxx + 1);
        let y = (dots / (maxx + 1)) % (maxy + 1);
        ((y as u32) << 12) | x as u32
    }

    /// Frames the beam has completed since t0, or None while the timing
    /// generator is stopped (frozen, no dot clock, or no timing set).
    fn frame(&self, s: &GbeState) -> Option<u64> {
        if Self::get(&s.r, VT_XY) & VT_FREEZE != 0 || Self::get(&s.r, DOTCLOCK) & DOTCLOCK_RUN == 0 {
            return None;
        }
        let max = Self::get(&s.r, VT_XYMAX);
        let (maxx, maxy) = ((max & 0xfff) as u64, ((max >> 12) & 0xfff) as u64);
        if maxx == 0 || maxy == 0 {
            return None;
        }
        let us = self.now().saturating_sub(s.t0);
        let dots = us * Self::dot_hz(&s.r) / 1_000_000 + s.reads * DOTS_PER_READ;
        Some(dots / ((maxx + 1) * (maxy + 1)))
    }

    /// Copy every shadow written before the current frame into its live
    /// register, as the start of vertical blanking does; all of them at once
    /// if the timing generator has stopped.
    fn latch(&self, s: &mut GbeState) {
        if s.pending.is_empty() {
            return;
        }
        let now = self.frame(s);
        let pending = std::mem::take(&mut s.pending);
        for (live, val, at) in pending {
            if now.is_none_or(|f| f > at) {
                s.r[(live / 4) as usize] = val;
            } else {
                s.pending.push((live, val, at));
            }
        }
    }

    /// Which of the four vertical-timing interrupts fired since the last call:
    /// bit n when the beam reached line VT_INTRn (CRIME latches them as its
    /// bits 16..19). A line beyond the frame never fires.
    pub fn vt_crossings(&self) -> u32 {
        let mut s = self.s.lock().unwrap();
        if Self::get(&s.r, VT_XY) & VT_FREEZE != 0 || Self::get(&s.r, DOTCLOCK) & DOTCLOCK_RUN == 0 {
            return 0;
        }
        let max = Self::get(&s.r, VT_XYMAX);
        let (maxx, maxy) = ((max & 0xfff) as u64, ((max >> 12) & 0xfff) as u64);
        if maxx == 0 || maxy == 0 {
            return 0;
        }
        let us = self.now().saturating_sub(s.t0);
        let now = us * Self::dot_hz(&s.r) / 1_000_000 + s.reads * DOTS_PER_READ;
        let last = std::mem::replace(&mut s.seen_dots, now);
        if now <= last {
            return 0;
        }
        let (i01, i23) = (Self::get(&s.r, VT_INTR01), Self::get(&s.r, VT_INTR23));
        let lines = [(i01 >> 12) & 0xfff, i01 & 0xfff, (i23 >> 12) & 0xfff, i23 & 0xfff];
        let period = (maxx + 1) * (maxy + 1);
        // Crossings of `off` (mod period) at or before `p`.
        let count = |p: u64, off: u64| if p < off { 0 } else { (p - off) / period + 1 };
        let mut fired = 0;
        for (n, &line) in lines.iter().enumerate() {
            let line = line as u64;
            if line > maxy {
                continue;
            }
            let off = line * (maxx + 1);
            if count(now, off) != count(last, off) {
                fired |= 1 << n;
            }
        }
        fired
    }

    fn read_reg(&self, off: u32) -> u32 {
        if crate::ip32::hot_counters() {
            *self.hot.lock().unwrap().entry(off).or_insert(0) += 1;
        }
        let mut s = self.s.lock().unwrap();
        self.latch(&mut s);
        match off {
            CTRLSTAT => {
                let v = Self::get(&s.r, CTRLSTAT);
                // Chip id 1, monitor sense 0, and board revision 1 on the
                // strap pin io_3 while it is an input: with revision 0 the
                // PROM loads a bit-reversed gamma ramp for early boards.
                let strap = if v & (1 << 13) != 0 { 1 << 12 } else { 0 };
                (v & !0x1f) | 1 | strap
            }
            // Inverted I2C with nothing on the bus: the lines read as driven.
            I2C | I2CFP => Self::get(&s.r, off) & 3,
            DEVICE_ID => 0x666,
            VT_XY => self.beam(&mut s),
            // An idle rev-1 colour-map FIFO, the one value every poller takes.
            CM_FIFO => 0,
            _ => Self::get(&s.r, off),
        }
    }

    fn write_reg(&self, off: u32, val: u32) {
        let mut s = self.s.lock().unwrap();
        let i = (off / 4) as usize;
        if i >= s.r.len() {
            return;
        }
        if off == VT_XY && val & VT_FREEZE == 0 && s.r[i] & VT_FREEZE != 0 {
            // Frame numbers restart with the beam: nothing pending may wait
            // on the old count.
            for (live, v, _) in std::mem::take(&mut s.pending) {
                s.r[(live / 4) as usize] = v;
            }
            s.t0 = self.now();
            s.reads = 0;
            s.seen_dots = 0;
        }
        s.r[i] = val;
        // The hardware copies each DMA control's shadow into the live
        // register at the start of vertical blanking, so until then the live
        // one still reads what it held. IRIX depends on that: its mode set
        // writes the overlay shadow with DMA on and then waits for the live
        // bit to read the old state, which it would never see if the copy
        // were immediate. Pollers that wait for the new value get it a frame
        // later. With the timing generator stopped there is no blanking to
        // wait for, and the copy is immediate.
        let live = match off {
            OVR_CONTROL_SHADOW => Some(OVR_CONTROL),
            FRM_CONTROL_SHADOW => Some(FRM_CONTROL),
            DID_CONTROL_SHADOW => Some(DID_CONTROL),
            _ => None,
        };
        if let Some(l) = live {
            self.latch(&mut s);
            match self.frame(&s) {
                Some(f) => {
                    s.pending.retain(|p| p.0 != l);
                    s.pending.push((l, val, f));
                }
                None => s.r[(l / 4) as usize] = val,
            }
        }
    }

    /// The registers that decide what is on screen, for a report.
    pub fn summary(&self) -> String {
        let s = self.s.lock().unwrap();
        let g = |o: u32| Self::get(&s.r, o);
        format!(
            "ctrlstat {:#010x} dotclock {:#010x} vt_xy {:#010x} xymax {:#010x} hcmap {:#010x} vcmap {:#010x} \
             flags {:#x} tile {:#010x} pix {:#010x} frm {:#010x}/{:#010x} did {:#010x} mode0 {:#x}",
            g(CTRLSTAT), g(DOTCLOCK), g(VT_XY), g(VT_XYMAX), g(VT_HCMAP), g(VT_VCMAP), g(VT_FLAGS),
            g(FRM_SIZE_TILE), g(FRM_SIZE_PIXEL), g(FRM_CONTROL), g(FRM_CONTROL_SHADOW), g(DID_CONTROL), g(MODE)
        )
    }

    /// Render from `ram`: copy the frame's pages, then scan them.
    pub fn render_from(&self, ram: &O2Memory, out: &mut [u8]) -> Option<(u32, u32)> {
        let snap = {
            let m = &*ram;
            self.snapshot(&m)
        };
        self.render(&snap, out)
    }

    /// Copy the pages the next [`Self::render`] reads.
    pub fn snapshot(&self, mem: &O2Memory) -> ScanSnapshot {
        let r = {
            let mut s = self.s.lock().unwrap();
            self.latch(&mut s);
            s.r.clone()
        };
        let g = |off: u32| Self::get(&r, off);
        let h = ((g(VT_VCMAP) >> 12) & 0xfff) as u32;
        let mut pages: Vec<u32> = Vec::new();
        let bus = mem.phys();
        let rd16 = |a: u32| mem.dma_offset(a).map(|p| bus.read16(p as u64).data).unwrap_or(0);
        // A tile list and its tiles: `ctl` holds the list address and the
        // DMA enable, `tile` the width in tiles and the partial tile.
        let list = |ctl: u32, tile: u32, rows: u32, pages: &mut Vec<u32>| {
            let per_row = ((tile >> 5) & 0xff) + ((tile & 0x1f) != 0) as u32;
            if ctl & 1 == 0 || per_row == 0 {
                return;
            }
            let base = ctl & 0xffff_ffe0;
            let n = (per_row * rows.div_ceil(128)).min(65536);
            for p in (base >> 16)..=((base + 2 * n) >> 16) {
                pages.push(p);
            }
            for i in 0..n {
                let e = rd16(base + 2 * i) as u32;
                if e != 0 {
                    pages.push(e);
                }
            }
        };
        list(g(FRM_CONTROL), g(FRM_SIZE_TILE), g(FRM_SIZE_PIXEL) >> 16, &mut pages);
        list(g(OVR_CONTROL), g(OVR_WIDTH_TILE), h, &mut pages);
        let did = g(DID_CONTROL);
        if did & (1 << 16) != 0 {
            pages.push(did & 0xffff);
        }
        pages.sort_unstable();
        pages.dedup();
        let mut snap = ScanSnapshot { slot: std::collections::HashMap::new(), bytes: Vec::with_capacity(pages.len() * 0x10000) };
        let mut words = vec![0u64; 0x10000 / 8];
        for p in pages {
            if let Some(o) = mem.dma_offset(p << 16) {
                // A page never crosses a bank. Kept as RAM keeps it: each
                // word a host u32, so byte `o` is at `o ^ 3`.
                if bus.read_block(o as u64, &mut words) != BUS_OK {
                    continue;
                }
                snap.slot.insert(p, snap.slot.len());
                for w in &words {
                    snap.bytes.extend_from_slice(&((w >> 32) as u32).to_ne_bytes());
                    snap.bytes.extend_from_slice(&(*w as u32).to_ne_bytes());
                }
            }
        }
        snap
    }

    /// CRS_POS: x in bits 11:0 and y in bits 27:16, each a 12-bit signed
    /// number (the X server puts the glyph at -6 to keep a hotspot on the
    /// edge).
    fn cursor_xy(crs: u32) -> (i32, i32) {
        let sext12 = |v: u32| ((v & 0xfff) as i32) << 20 >> 20;
        (sext12(crs), sext12(crs >> 16))
    }

    /// The hardware cursor when enabled: its glyph's top-left on screen and
    /// one opaque-pixel mask word per row (bit `c` = column `c`), 32x32.
    pub fn cursor(&self) -> Option<(i32, i32, Vec<u64>)> {
        let r = {
            let mut s = self.s.lock().unwrap();
            self.latch(&mut s);
            [Self::get(&s.r, CRS_CTL), Self::get(&s.r, CRS_POS)]
                .into_iter()
                .chain((0..64).map(|i| Self::get(&s.r, CRS_GLYPH + 4 * i)))
                .collect::<Vec<u32>>()
        };
        if r[0] & 1 == 0 {
            return None;
        }
        let (x, y) = Self::cursor_xy(r[1]);
        // Two bits a pixel, 16 pixels a word, leftmost in the top bits; as
        // `render` draws it.
        let mask = (0..32usize).map(|dy| (0..32usize).fold(0u64, |m, dx| {
            let word = r[2 + 2 * dy + dx / 16];
            m | (((word >> (30 - 2 * (dx % 16))) & 3 != 0) as u64) << dx
        })).collect();
        Some((x, y, mask))
    }

    /// Scan the frame out of RAM into `out` (R, G, B, A bytes, row 0 at the
    /// top), as GBE would. Returns the size drawn, or None when the display
    /// is off. `out` must hold the largest mode (2048 x 2048 pixels).
    pub fn render<M: ScanMem + std::ops::Index<usize, Output = u8>>(&self, mem: &M, out: &mut [u8]) -> Option<(u32, u32)> {
        let r = {
            let mut s = self.s.lock().unwrap();
            self.latch(&mut s);
            s.r.clone()
        };
        let g = |off: u32| Self::get(&r, off);
        let w = (g(VT_HCMAP) >> 12) & 0xfff;
        let h = (g(VT_VCMAP) >> 12) & 0xfff;
        if (g(CTRLSTAT) >> 28) & 3 != 3
            || g(DOTCLOCK) & DOTCLOCK_RUN == 0
            || g(VT_XY) & VT_FREEZE != 0
            || w == 0
            || h == 0
            || g(VT_FLAGS) & 0xa == 0xa
            || (w as usize * h as usize * 4) > out.len()
        {
            return None;
        }
        let (w, h) = (w as usize, h as usize);

        // The normal planes as one byte stream (section 2.3 of the spec).
        let tile = g(FRM_SIZE_TILE);
        let rhs = (tile & 0x1f) as usize;
        let wt = ((tile >> 5) & 0xff) as usize;
        let bpp = 1usize << ((tile >> 13) & 3).min(2);
        let fb_h = (g(FRM_SIZE_PIXEL) >> 16) as usize;
        let per_row = wt + (rhs != 0) as usize;
        let line_bytes = wt * 512 + rhs * 32;
        let frm = g(FRM_CONTROL);
        let dma = frm & 1 != 0 && per_row != 0 && line_bytes != 0;
        // Each tile's offset in `mem`, resolved once per frame.
        let mut tiles: Vec<Option<usize>> = Vec::new();
        if dma {
            let list = frm & 0xffff_ffe0;
            let n = per_row * fb_h.div_ceil(128);
            tiles.reserve(n);
            for i in 0..n.min(65536) {
                let e = mem
                    .dma_offset(list + 2 * i as u32)
                    .filter(|&o| o + 1 < mem.len())
                    .map(|o| u16::from_be_bytes([mem[o], mem[o + 1]]))
                    .unwrap_or(0);
                tiles.push(if e == 0 { None } else { mem.dma_offset((e as u32) << 16).filter(|&o| o + 0x10000 <= mem.len()) });
            }
        }
        let fetch = |s: usize| -> u32 {
            if !dma {
                return 0;
            }
            let (line, col) = (s / line_bytes, s % line_bytes);
            if line >= fb_h {
                return 0;
            }
            let Some(Some(base)) = tiles.get((line / 128) * per_row + col / 512) else { return 0 };
            let o = base + (line % 128) * 512 + col % 512;
            match bpp {
                1 => mem[o] as u32,
                2 => u16::from_be_bytes([mem[o], mem[o + 1]]) as u32,
                _ => u32::from_be_bytes([mem[o], mem[o + 1], mem[o + 2], mem[o + 3]]),
            }
        };

        // The overlay planes: 8-bit pixels over the normal ones, in their own
        // tile list (same width-in-tiles layout as the normal planes). Zero is
        // clear; anything else is colour map entry 0x1100 + the pixel, always
        // through the gamma map (GBE spec). The X server draws its pop-up
        // menus here.
        let ovr = g(OVR_CONTROL);
        let ovr_tile = g(OVR_WIDTH_TILE);
        let ovr_rhs = (ovr_tile & 0x1f) as usize;
        let ovr_per_row = ((ovr_tile >> 5) & 0xff) as usize + (ovr_rhs != 0) as usize;
        let ovr_line = ((ovr_tile >> 5) & 0xff) as usize * 512 + ovr_rhs * 32;
        let ovr_dma = ovr & 1 != 0 && ovr_per_row != 0 && ovr_line != 0;
        let ovr_tiles: Vec<Option<usize>> = if ovr_dma {
            let list = ovr & 0xffff_ffe0;
            (0..(ovr_per_row * h.div_ceil(128)).min(4096))
                .map(|i| {
                    let e = mem
                        .dma_offset(list + 2 * i as u32)
                        .filter(|&o| o + 1 < mem.len())
                        .map(|o| u16::from_be_bytes([mem[o], mem[o + 1]]))
                        .unwrap_or(0);
                    if e == 0 { None } else { mem.dma_offset((e as u32) << 16).filter(|&o| o + 0x10000 <= mem.len()) }
                })
                .collect()
        } else {
            Vec::new()
        };
        let ovr_at = |x: usize, y: usize| -> u32 {
            if !ovr_dma || x >= ovr_line {
                return 0;
            }
            let Some(Some(base)) = ovr_tiles.get((y / 128) * ovr_per_row + x / 512) else { return 0 };
            mem[base + (y % 128) * 512 + x % 512] as u32
        };

        // Each pixel's WID (mode-table entry) comes from the DID tables when
        // DID DMA is on and the 32 WIDs differ; otherwise WID 0, which is
        // all the PROM and the BSD consoles use. IRIX's X server runs 8-bit
        // indexed windows (the colour index in a pixel's low byte) beside
        // 24-bit ones, and only the DIDs say which a pixel belongs to.
        let did = g(DID_CONTROL);
        let wids: Vec<u32> = (0..32).map(|i| g(MODE + 4 * i)).collect();
        let use_did = did & (1 << 16) != 0 && wids.iter().any(|&m| m != wids[0]);
        let mut line_did = vec![0u8; if use_did { w } else { 0 }];
        let table = (did & 0xffff) << 16;
        let rd16 = |a: u32| mem.dma_offset(a).filter(|&o| o + 1 < mem.len()).map(|o| u16::from_be_bytes([mem[o], mem[o + 1]])).unwrap_or(0);
        let rd32 = |a: u32| (rd16(a) as u32) << 16 | rd16(a + 2) as u32;
        let mut frame_i = 0u32;
        let cmap = |i: u32| g(CMAP + 4 * (i % 6144));
        let gmap = |v: u32, shift: u32| (g(GMAP + 4 * (v & 0xff)) >> shift) & 0xff;
        let crs_on = g(CRS_CTL) & 1 != 0;
        let crs = g(CRS_POS);
        let probe_at = self.probe.lock().unwrap().0;
        let (cx, cy) = Self::cursor_xy(crs);

        for y in 0..h {
            if use_did {
                // Frame entry: 25:19 block, 18:11 offset, 10:0 last line. Its
                // line table (16-bit entries, 15:11 DID, 10:0 last x) sits in
                // one 512-byte block of the 64 KB table.
                let fe = rd32(table + 4 * frame_i.min(16383));
                let (block, off, yend) = ((fe >> 19) & 0x7f, (fe >> 11) & 0xff, fe & 0x7ff);
                let lt = table + 512 * block + 2 * off;
                let (mut x, mut k) = (0usize, 0u32);
                while x < w && k < 256 - off {
                    let le = rd16(lt + 2 * k);
                    let (d, xend) = ((le >> 11) as u8, (le & 0x7ff) as usize);
                    while x < w && x <= xend {
                        line_did[x] = d;
                        x += 1;
                    }
                    k += 1;
                }
                // Anything past the table's last span keeps that span's DID.
                let last = if x > 0 { line_did[x - 1] } else { 0 };
                line_did[x..].fill(last);
                if y as u32 >= yend {
                    frame_i += 1;
                }
            }
            for x in 0..w {
                let wid = if use_did { wids[line_did[x] as usize & 31] } else { wids[0] };
                let typ = (wid >> 2) & 7;
                let cm = (wid >> 5) & 0x1f;
                let gamma = wid & (1 << 10) == 0;
                let p = fetch((y * w + x) * bpp);
                // A type narrower than the frame buffer reads one half of the
                // word: buf 10 the upper half, otherwise the lower.
                let p = if typ <= 4 && bpp > 1 {
                    let half = bpp as u32 * 4;
                    let mask = (1u32 << half) - 1;
                    if wid & 3 == 2 { (p >> half) & mask } else { p & mask }
                } else {
                    p
                };
                let (mut rr, mut gg, mut bb) = match typ {
                    0 => {
                        let c = cmap((cm << 8) | (p & 0xff));
                        (c >> 24, (c >> 16) & 0xff, (c >> 8) & 0xff)
                    }
                    1 => {
                        let c = cmap(p & 0xfff);
                        (c >> 24, (c >> 16) & 0xff, (c >> 8) & 0xff)
                    }
                    2 => {
                        let (r3, g3, b2) = ((p >> 5) & 7, (p >> 2) & 7, p & 3);
                        ((r3 << 5) | (r3 << 2) | (r3 >> 1), (g3 << 5) | (g3 << 2) | (g3 >> 1), b2 * 0x55)
                    }
                    3 => (((p >> 8) & 0xf) * 0x11, ((p >> 4) & 0xf) * 0x11, (p & 0xf) * 0x11),
                    4 => {
                        let e = |v: u32| (v << 3) | (v >> 2);
                        (e((p >> 10) & 0x1f), e((p >> 5) & 0x1f), e(p & 0x1f))
                    }
                    _ => (p >> 24, (p >> 16) & 0xff, (p >> 8) & 0xff),
                };
                if typ >= 2 && cm != 0 {
                    rr = cmap((cm << 8) | rr) >> 24;
                    gg = (cmap((cm << 8) | gg) >> 16) & 0xff;
                    bb = (cmap((cm << 8) | bb) >> 8) & 0xff;
                }
                if gamma {
                    (rr, gg, bb) = (gmap(rr, 24), gmap(gg, 16), gmap(bb, 8));
                }
                let ov = ovr_at(x, y);
                if probe_at == Some((x, y)) {
                    let raw = fetch((y * w + x) * bpp);
                    self.probe.lock().unwrap().1 = format!(
                        "raw {raw:#010x} did {} wid {wid:#x} type {typ} cmap {cm} gamma {gamma} pixel {p:#x} rgb {rr:02x}{gg:02x}{bb:02x} overlay {ov:#x}",
                        if use_did { line_did[x] as i32 } else { -1 }
                    );
                }
                if ov != 0 {
                    let c = cmap(0x1100 + ov);
                    (rr, gg, bb) = (gmap(c >> 24, 24), gmap((c >> 16) & 0xff, 16), gmap((c >> 8) & 0xff, 8));
                }
                if crs_on {
                    let (dx, dy) = (x as i32 - cx, y as i32 - cy);
                    if (0..32).contains(&dx) && (0..32).contains(&dy) {
                        let word = g(CRS_GLYPH + 4 * (2 * dy as u32 + (dx as u32 / 16)));
                        let v = (word >> (30 - 2 * (dx as u32 % 16))) & 3;
                        if v != 0 {
                            let c = g(CRS_CMAP1 + 4 * (v - 1));
                            (rr, gg, bb) = (c >> 24, (c >> 16) & 0xff, (c >> 8) & 0xff);
                        }
                    }
                }
                let o = (y * w + x) * 4;
                out[o] = rr as u8;
                out[o + 1] = gg as u8;
                out[o + 2] = bb as u8;
                out[o + 3] = 255;
            }
        }
        Some((w as u32, h as u32))
    }
}

/// Narrower accesses reach the 32-bit register holding them.
impl BusDevice for Gbe {
    fn read8(&self, addr: u64) -> crate::traits::BusRead8 {
        crate::traits::BusRead8::ok(self.read32(addr & !3).data as u8)
    }
    fn read16(&self, addr: u64) -> crate::traits::BusRead16 {
        crate::traits::BusRead16::ok(self.read32(addr & !3).data as u16)
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        self.write32(addr & !3, val as u32)
    }
    fn write16(&self, addr: u64, val: u16) -> u32 {
        self.write32(addr & !3, val as u32)
    }
    fn read32(&self, addr: u64) -> BusRead32 {
        BusRead32::ok(self.read_reg((addr as u32) & 0xffffc))
    }
    fn read64(&self, addr: u64) -> BusRead64 {
        let a = (addr as u32) & 0xffff8;
        BusRead64::ok((self.read_reg(a) as u64) << 32 | self.read_reg(a + 4) as u64)
    }
    fn write32(&self, addr: u64, val: u32) -> u32 {
        self.write_reg((addr as u32) & 0xffffc, val);
        BUS_OK
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        let a = (addr as u32) & 0xffff8;
        self.write_reg(a, (val >> 32) as u32);
        self.write_reg(a + 4, val as u32);
        BUS_OK
    }
}

/// A texture as the rendering engine samples it: its base level's size, how
/// it wraps, and where its texels are. Worked out from perfly, whose hubcap
/// tile matches hubcap.rgb texel for texel this way (the spec leaves texel
/// look-up "TBD"):
///
/// - s and t are 1.14 fixed point once divided by q (16384 is the whole
///   map: perfly's hubcap centre is s 24399, t 24900, 0.49 and 0.52 once
///   repeated); u = s * width / 16384, width = 1 << the format's u wrap shift.
/// - Texels sit in 64 KB tiles laid out like frame-buffer tiles, 128 lines
///   of 512 bytes, in 32-byte blocks of four rows: see `subtile_offset`.
///   16-bit texels are A1R5G5B5, 32-bit ones RGBA8 with R high (o2gltest's
///   self-describing 64 x 64 map, 2026-10-19). Bits past those for maps over
///   128 rows are a guess.
/// - The base level is at the first valid entry of the texture TLB (88 for
///   perfly's 128 x 128 and 128 x 64 maps; levels 1 and up follow at 92 and
///   93, unused here: no mipmapping yet).
struct Texture {
    w: u32,
    h: u32,
    repeat_s: bool,
    repeat_t: bool,
    linear: bool,
    base: usize,
    /// Bytes a texel: 2 (see `texel16`) or 4 (RGBA8, R high).
    bpt: u32,
    /// Texture.mode 23:20: 1 RGB, 2 RGBA, 4 RGBA4, 5 ALPHA, 6 INTENSITY,
    /// 7 LUMINANCE, 8 LUMINANCE_ALPHA.
    typ: u32,
    /// The base level counted from 1024 x 1024 (Texture.mode 17:14; 4 for
    /// 64 x 64), the last level below it (13:10), and the minification
    /// filter (9:7: NEAREST, LINEAR, then the four MIPMAP ones).
    map_level: u32,
    max_level: u32,
    min_filter: u32,
}

/// Where absolute mip level `a` (0 = 1024 x 1024) starts, as a byte address
/// in texture-TLB space (entry << 16 | offset). Levels are not packed after
/// their texture: each size has its own place, the same for every texture.
/// 128 is at entry 88 (perfly); 64 at entry 92 and the smaller ones in 92
/// and 93 at fixed offsets, read off o2gltest's map with a solid colour a
/// level (2026-10-19). 1024, 512 and 256 (entries 0, 64, 80) are where a
/// 32-bit level of that size would fit before 88: unconfirmed.
fn mip_level_addr(a: u32) -> u64 {
    let at = |entry: u64, off: u64| entry << 16 | off;
    match a {
        0 => at(0, 0),
        1 => at(64, 0),
        2 => at(80, 0),
        3 => at(88, 0),
        4 => at(92, 0),
        5 => at(93, 0x8000),
        6 => at(92, 0xc000),
        7 => at(93, 0xe000),
        8 => at(93, 0xf000),
        9 => at(93, 0xf800),
        _ => at(93, 0xfc00),
    }
}

/// Byte offset within its 64 KB tile of the texel at byte column `bx`
/// (x times the texel size) and row `y` of a subtiled texture map. One
/// layout for 16- and 32-bit texels: byte column bits 2:0, then y 1:0,
/// byte column 6:3, y 5:4, y 3:2, byte column 8:7, y 6. A tile is 512 bytes
/// by 128 rows; past that, `subtile_step` says which tile. Solved from
/// IRIX's textures (the spec leaves texel look-up "TBD"): every texel of a
/// 64x64 RGBA8 map that encodes its own coordinates, and the 16-bit maps
/// before.
fn subtile_offset(bx: u32, y: u32) -> u32 {
    (bx & 7) | (y & 3) << 3 | ((bx >> 3) & 15) << 5 | ((y >> 4) & 3) << 9 | ((y >> 2) & 3) << 11
        | ((bx >> 7) & 3) << 13 | ((y >> 6) & 1) << 15
}

/// The tile, counted in texture-TLB entries from a level's first, holding
/// byte column `bx` and row `y`: a level's tiles form a grid, the next
/// 512-byte column one entry on and the next 128 rows `stride` on. Quake 2's
/// maps show it: 256 x 128 RGBA8 in entries 80 and 81, 256 x 256 RGBA8 in
/// 80, 81, 84 and 85, 256 x 256 16-bit in 80 and 84.
fn subtile_step(bx: u32, y: u32, stride: u32) -> u32 {
    (bx >> 9) + (y >> 7) * stride
}

/// A 16-bit texel of type `typ` (Texture.mode 23:20) as R, G, B, A, 0-255
/// (spec figure 7-7): RGBA4 has R in the top nibble, LUMINANCE_ALPHA L in
/// the high byte and A in the low, and RGB and RGBA are A1R5G5B5. The
/// spec draws no 16-bit luminance, intensity or alpha texel; they are taken
/// to sit where LA8's L and A do (a guess: L and I high, A low). A channel
/// the type lacks reads as 255; `Texture::has_rgb` and `has_alpha` say
/// which the texture function uses.
fn texel16(typ: u32, p: u32) -> [f64; 4] {
    let (hi, lo) = ((p >> 8) as f64, (p & 0xff) as f64);
    match typ {
        4 => [12, 8, 4, 0].map(|sh| ((p >> sh) & 15) as f64 * 17.0),
        5 => [255.0, 255.0, 255.0, lo],
        6 => [hi, hi, hi, hi],
        7 => [hi, hi, hi, 255.0],
        8 => [hi, hi, hi, lo],
        _ => {
            let c = |v: u32| ((v << 3) | (v >> 2)) as f64;
            [c((p >> 10) & 31), c((p >> 5) & 31), c(p & 31), if p & 0x8000 != 0 { 255.0 } else { 0.0 }]
        }
    }
}

impl Texture {
    fn from_regs(r: &[u32]) -> Option<Texture> {
        let g = |off: u32| r[(off / 4) as usize];
        let mode = g(TEX_MODE);
        let (typ, depth) = ((mode >> 20) & 15, (mode >> 18) & 3);
        // RGBA4 and LA8 come only 16 bits wide (spec figure 7-7); so do
        // alpha, intensity and luminance alone, as IRIX uses them (bzflag's
        // and Macau's maps), laid out as LA8 (see `texel16`).
        if !matches!((typ, depth), (1 | 2, 1 | 2) | (4..=8, 1)) {
            return None;
        }
        let fmt_hi = g(TEX_FORMAT);
        let (uw, vw) = ((fmt_hi >> 4) & 15, fmt_hi & 15);
        let base = (0..112).find(|&i| {
            let word = TLB_TEX + 8 * (i as u32 / 4);
            let w64 = ((g(word) as u64) << 32) | g(word + 4) as u64;
            (w64 >> (48 - 16 * (i % 4))) & 0x8000 != 0
        })?;
        Some(Texture {
            w: 1 << uw,
            h: 1 << vw,
            repeat_s: (mode >> 4) & 3 == 1,
            repeat_t: (mode >> 2) & 3 == 1,
            linear: mode & (1 << 6) != 0,
            base,
            bpt: if depth == 2 { 4 } else { 2 },
            typ,
            map_level: (mode >> 14) & 15,
            max_level: (mode >> 10) & 15,
            min_filter: (mode >> 7) & 7,
        })
    }

    /// Whether the minification filter uses mip levels.
    fn mipmapped(&self) -> bool {
        self.min_filter >= 2
    }

    /// The texel at integer (u, v) of mip level `level` (0 = the base),
    /// wrapped or clamped, as R, G, B, A.
    fn texel(&self, r: &[u32], mem: &RamWindow, level: u32, u: i64, v: i64) -> [f64; 4] {
        let (w, h) = ((self.w >> level).max(1), (self.h >> level).max(1));
        let wrap = |c: i64, n: u32, rep: bool| if rep { c.rem_euclid(n as i64) as u32 } else { c.clamp(0, n as i64 - 1) as u32 };
        let (u, v) = (wrap(u, w, self.repeat_s), wrap(v, h, self.repeat_t));
        // A map of 64 x 64 or less sits where its size's level of a full
        // pyramid would, entry and offset (a 32 x 32 map at 0x8000 of entry
        // 93, as IRIX loads it), whichever of the two entries are valid.
        // Larger maps start at their first valid entry.
        let start = if level == 0 && self.map_level < 4 {
            (self.base as u64) << 16
        } else {
            mip_level_addr(self.map_level + level)
        };
        // Grid rows are 4 entries for 256 and 512 (Quake 2's 256 maps);
        // 1024 would need 8 (64 entries for 8 rows of tiles): unconfirmed.
        let stride = if self.map_level + level == 0 { 8 } else { 4 };
        let bx = u * self.bpt;
        let addr = start + ((subtile_step(bx, v, stride) as u64) << 16) + subtile_offset(bx & 511, v & 127) as u64;
        let (entry, byte) = ((addr >> 16) as usize, (addr & 0xffff) as usize);
        let word = TLB_TEX + 8 * (entry as u32 / 4);
        let w64 = ((r[(word / 4) as usize] as u64) << 32) | r[(word / 4 + 1) as usize] as u64;
        let e = (w64 >> (48 - 16 * (entry % 4))) & 0xffff;
        if e & 0x8000 == 0 {
            return [0.0; 4];
        }
        let n = self.bpt as usize;
        let Some(o) = mem.dma_offset((((e & 0x7fff) as u32) << 16) + (byte & 0xffff) as u32).filter(|&o| mem.holds(o, n)) else {
            return [0.0; 4];
        };
        if n == 4 {
            return [mem[o], mem[o + 1], mem[o + 2], mem[o + 3]].map(|b| b as f64);
        }
        texel16(self.typ, ((mem[o] as u32) << 8) | mem[o + 1] as u32)
    }

    /// Whether the map's texels carry alpha (RGBA, RGBA4, ALPHA,
    /// INTENSITY, LUMINANCE_ALPHA): without, the fragment's alpha stays, as
    /// in OpenGL.
    fn has_alpha(&self) -> bool {
        matches!(self.typ, 2 | 4 | 5 | 6 | 8)
    }

    /// Whether the map's texels carry colour: an ALPHA map leaves the
    /// fragment's colour alone.
    fn has_rgb(&self) -> bool {
        self.typ != 5
    }

    /// Level `level` at (s, t) in 1.14 units, bilinear or nearest. IRIX's
    /// GL hands the engine coordinates 1.0 (16384) past the map's own: the
    /// map runs from 1.0 to 2.0. Repeating maps never showed it; a clamped
    /// one read every texel off its far edge (distort's whole picture one
    /// tan, Quake 3's lightmaps black), and perfly's hubcap centre is s
    /// 1.49, not 0.49.
    fn sample_level(&self, r: &[u32], mem: &RamWindow, level: u32, linear: bool, s: f64, t: f64) -> [f64; 4] {
        let (w, h) = ((self.w >> level).max(1), (self.h >> level).max(1));
        let (s, t) = (s - 16384.0, t - 16384.0);
        let (u, v) = (s * w as f64 / 16384.0 - 0.5, t * h as f64 / 16384.0 - 0.5);
        if !linear {
            return self.texel(r, mem, level, u.round() as i64, v.round() as i64);
        }
        let (u0, v0) = (u.floor(), v.floor());
        let (fu, fv) = (u - u0, v - v0);
        let (u0, v0) = (u0 as i64, v0 as i64);
        let tx = |du: i64, dv: i64| self.texel(r, mem, level, u0 + du, v0 + dv);
        let (a, b, c, d) = (tx(0, 0), tx(1, 0), tx(0, 1), tx(1, 1));
        let mut out = [0.0; 4];
        for k in 0..4 {
            out[k] = (a[k] * (1.0 - fu) + b[k] * fu) * (1.0 - fv) + (c[k] * (1.0 - fu) + d[k] * fu) * fv;
        }
        out
    }

    /// The map at (s, t) in 1.14 units with level of detail `lod`
    /// (log2 of texels per pixel at the base level, the spec's rho): the
    /// magnification filter at or below 0, the minification filter above,
    /// choosing or blending mip levels as OpenGL does.
    fn sample(&self, r: &[u32], mem: &RamWindow, s: f64, t: f64, lod: f64) -> [f64; 4] {
        if lod <= 0.0 || !self.mipmapped() {
            let linear = if lod <= 0.0 { self.linear } else { self.min_filter == 1 };
            return self.sample_level(r, mem, 0, linear, s, t);
        }
        let last = self.max_level as f64;
        // 2, 4: NEAREST within a level; 3, 5: LINEAR. 2, 3: the nearest
        // level; 4, 5: the two either side, blended.
        let linear = matches!(self.min_filter, 3 | 5);
        if self.min_filter <= 3 {
            let level = (lod + 0.5).floor().clamp(0.0, last) as u32;
            return self.sample_level(r, mem, level, linear, s, t);
        }
        let l0 = lod.floor().clamp(0.0, last);
        let l1 = (l0 + 1.0).min(last);
        let f = (lod - l0).clamp(0.0, 1.0);
        let (a, b) = (self.sample_level(r, mem, l0 as u32, linear, s, t), self.sample_level(r, mem, l1 as u32, linear, s, t));
        [0, 1, 2, 3].map(|k| a[k] * (1.0 - f) + b[k] * f)
    }
}

/// A GL vertex coordinate: 13.6 fixed point in the low 23 bits (CRIME 1.5
/// spec). IRIX's GL writes floats biased by 2^17, whose mantissa holds
/// exactly that; perfly's panel text writes the fixed-point value itself
/// (0x00844140 for 4357.0), which read as a float is a denormal.
fn gl_coord(w: u32) -> f64 {
    (w & 0x7f_ffff) as f64 / 64.0
}

/// How Gouraud gradients are read: their fraction bits and which vertex the
/// start colour belongs to (3: the traversal's start point, the default).
/// The replay test can try others (RE_GOURAUD).
static GOURAUD: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(12 << 8 | 3);

fn gouraud_model() -> (u32, usize) {
    let v = GOURAUD.load(std::sync::atomic::Ordering::Relaxed);
    (v >> 8, (v & 0xff) as usize)
}

/// CRIME rendering-engine register offsets from `0x1500_0000`.
pub mod re_reg {
    pub const IB_CONTROL: u32 = 0x0400;
    pub const TLB_A: u32 = 0x1000;
    pub const TLB_LINEAR_A: u32 = 0x1700;
    pub const BUFMODE_SRC: u32 = 0x2000;
    pub const BUFMODE_DST: u32 = 0x2008;
    pub const DRAWMODE: u32 = 0x2018;
    /// Clip mode (bits 9:5 enable screen masks 0-4, bits 4:0 inside or
    /// outside) and the five screen-mask rectangles.
    pub const CLIP_MODE: u32 = 0x2010;
    pub const CLIP_RECT0: u32 = 0x2020;
    pub const WINOFFSET_SRC: u32 = 0x2050;
    pub const WINOFFSET_DST: u32 = 0x2058;
    pub const PRIMITIVE: u32 = 0x2060;
    pub const VERTEX_0: u32 = 0x2070;
    pub const VERTEX_1: u32 = 0x2074;
    pub const XFER_SRC: u32 = 0x20a0;
    /// The linear end's address steps, in bytes: from one pixel to the
    /// next, and from a row's last pixel to the next row's first.
    pub const XFER_STEP_X: u32 = 0x20a8;
    pub const XFER_STEP_ROW: u32 = 0x20ac;
    /// Linear destination byte address.
    pub const XFER_DST: u32 = 0x20b0;
    pub const XFER_DST_STRIDE: u32 = 0x20b4;
    pub const STIPPLE_MODE: u32 = 0x20c0;
    pub const STIPPLE_PATTERN: u32 = 0x20c4;
    pub const FG: u32 = 0x20d0;
    /// Triangle vertices, 64 bits each: x float high, y float low.
    pub const TRI_V0: u32 = 0x2080;
    /// Shade colour R, G, B, A (8.12 fixed), then their gradients.
    pub const SHADE_R: u32 = 0x20e0;
    pub const SHADE_GRAD: u32 = 0x20f0;
    pub const TEX_MODE: u32 = 0x2110;
    pub const TEX_FORMAT: u32 = 0x2118;
    pub const TEX_SQ0: u32 = 0x2120;
    pub const TEX_TQ0: u32 = 0x2128;
    pub const TEX_Q0: u32 = 0x2130;
    /// Bit 0: s was scaled down by 16 to fit; bit 1: t. IRIX sets them per
    /// triangle once a coordinate passes about 16 maps (a 64-map floor sets
    /// 2, a 1024-map triangle 3); ignored, far floors took too fine a mip
    /// level and repeated 16 times too few.
    pub const TEX_COORD_SCALE: u32 = 0x2134;
    pub const TEX_DSQDX: u32 = 0x2138;
    pub const TEX_DSQDY: u32 = 0x2140;
    pub const TEX_DTQDX: u32 = 0x2148;
    pub const TEX_DTQDY: u32 = 0x2150;
    pub const TEX_DQDX: u32 = 0x2158;
    pub const TEX_DQDY: u32 = 0x215c;
    pub const TLB_TEX: u32 = 0x1600;
    pub const DEPTH_FUNC: u32 = 0x21c0;
    /// Stencil test: reference 31:24, value mask 23:16, function 15:12
    /// (the depth functions' order), then the ops for a stencil failure,
    /// a depth failure and a pass in 11:8, 7:4 and 3:0. Read off IRIX's
    /// writes (the spec's table is empty): mirrors marks its floor with
    /// 0x01ff7333 (ALWAYS, ref 1, REPLACE) and draws the reflection with
    /// 0x01ff2000 (EQUAL 1, KEEP).
    pub const STENCIL_MODE: u32 = 0x21e0;
    /// The stencil write mask, low 8 bits.
    pub const STENCIL_MASK: u32 = 0x21e8;
    pub const DEPTH_Z0: u32 = 0x21c8;
    pub const DEPTH_DZDX: u32 = 0x21d0;
    pub const DEPTH_DZDY: u32 = 0x21d8;
    pub const BG: u32 = 0x20d8;
    /// Alpha test: function in 11:8 (NEVER, LESS, EQUAL, LEQUAL, GREATER,
    /// NOTEQUAL, GEQUAL, ALWAYS), reference in 7:0.
    pub const ALPHA_TEST: u32 = 0x2198;
    /// Blend constant colour, RGBA with R high.
    pub const BLEND_CONST: u32 = 0x21a0;
    /// Blend op in 11:8, source factor in 7:4, destination factor in 3:0
    /// (OpenGL's, in the spec's order; see `CrimeRe::blend`).
    pub const BLEND_FUNC: u32 = 0x21a8;
    pub const LOGIC_OP: u32 = 0x21b0;
    pub const COLOR_MASK: u32 = 0x21b8;
    pub const MTE_MODE: u32 = 0x3000;
    pub const MTE_BYTEMASK: u32 = 0x3008;
    pub const MTE_STIPPLE: u32 = 0x3010;
    pub const MTE_FG: u32 = 0x3018;
    pub const MTE_SRC0: u32 = 0x3020;
    pub const MTE_DST0: u32 = 0x3030;
    pub const MTE_DST1: u32 = 0x3038;
    pub const MTE_SRC_STEP: u32 = 0x3040;
    pub const MTE_DST_STEP: u32 = 0x3048;
    pub const STATUS: u32 = 0x4000;
}

use re_reg::*;

const RE_WORDS: usize = 0x6000 / 4;
/// Idle everywhere, interface buffer empty.
const RE_STATUS_IDLE: u32 = 0x1e00_0000;
/// Entries the interface buffer holds here: far more than CRIME's, so the
/// CPU seldom waits on the engine (as the other boards' GFIFOs do).
const IB_DEPTH: usize = 65536;
/// CRIME's own interface buffer depth, the most software is shown.
const IB_HW_DEPTH: usize = 64;
/// The interface buffer level field of the status word (24:18).
const RE_STATUS_IB_LEVEL_SHIFT: u32 = 18;
/// How long a status read waits for a busy engine before it answers.
///
/// A guest polls the status word in a loop timed in its own microseconds
/// (`us_delay`), and the engine runs on its own host thread. Without a pause
/// the guest's clock races ahead of the engine: IRIX's early console setup
/// waits for the interface buffer to empty for about 4096 polls, which a CPU
/// that is not drawing anything gets through long before the engine drains a
/// startup queue, and the kernel then panics (a read TLB miss at
/// 0xff800000). A poll is the guest waiting for the engine anyway, so
/// spending that wait on the engine costs no overlap.
///
/// The first read after a write waits this long: software that only checks
/// for room in the buffer goes on drawing while the engine works.
const RE_STATUS_POLL_WAIT: std::time::Duration = std::time::Duration::from_micros(100);
/// How long a status read waits when the read before it also found the
/// engine busy and nothing was written since: the guest is in a loop waiting
/// for the engine.
///
/// The first wait alone gave a waiting loop 4096 x 100 us, about 0.4 s of
/// host time. That covered the engine on an idle host, but not one whose
/// thread was kept off a processor: on a loaded host IRIX panicked in early
/// console setup on a 400 MHz R12000 O2 in five boots of eleven, and holding
/// the engine for 2 to 3 s while the kernel waited brought the same panic
/// with either R12000 revision. A wait that ends when the engine goes idle
/// costs nothing on an idle host; the bound only keeps a wedged engine from
/// holding the CPU for good.
const RE_STATUS_LOOP_WAIT: std::time::Duration = std::time::Duration::from_millis(20);

mod draw {
    pub const GL: u32 = 1 << 22;
    pub const PIXEL_XFER: u32 = 1 << 21;
    pub const LINE_STIPPLE: u32 = 1 << 19;
    pub const POLY_STIPPLE: u32 = 1 << 18;
    pub const OPAQUE_STIPPLE: u32 = 1 << 17;
    pub const SHADE: u32 = 1 << 16;
    pub const TEXTURE: u32 = 1 << 15;
    pub const ANTIALIAS_LINE: u32 = 1 << 12;
    pub const ALPHA_TEST: u32 = 1 << 11;
    pub const BLEND: u32 = 1 << 10;
    pub const DEPTH_TEST: u32 = 1 << 2;
    pub const STENCIL_TEST: u32 = 1 << 0;
    pub const DEPTH_MASK: u32 = 1 << 1;
    pub const LOGIC_OP: u32 = 1 << 9;
    /// Dithering, set by GL (on by default in OpenGL). The spec's table
    /// swaps this with LOGIC_OP: see clanker/devices/ip32-o2-bringup-log.md.
    pub const DITHER: u32 = 1 << 8;
    pub const PLANE_MASK: u32 = 1 << 7;
}

/// CRIME's host interface buffer: register writes in order from the CPU to
/// the engine's thread, `(offset | WIDE for a 64-bit write, value)`. Far
/// deeper than CRIME's 64 entries, with the status word showing software no
/// more than those (see `read_reg`). Software that reuses memory the engine
/// reads waits for the idle bits, which stay clear until everything queued
/// is done. (An early asynchronous engine with a 64K queue but no busy
/// status drew icons from a staging buffer the X server had reused.)
type IntfBuf = crate::gfifo::GFifo<IB_DEPTH>;

/// Marks a 64-bit write in the interface buffer (offsets are below 0x8000).
const WIDE: u32 = 1 << 31;

pub struct CrimeRe {
    regs: Mutex<Vec<u32>>,
    /// Reads per register offset, for finding a driver spinning on one.
    pub hot: Mutex<BTreeMap<u32, u64>>,
    /// Memory, through the ppmem window: the engine draws without the
    /// memory's lock, as the CPU reads and writes it. Only the engine's
    /// thread takes this.
    win: Mutex<RamWindow>,
    /// What the guest asked for that is not modelled, with counts, so a
    /// missing primitive shows up instead of drawing nothing silently.
    unhandled: Mutex<BTreeMap<String, u64>>,
    /// Primitives run, for the end-of-run report.
    pub drawn: std::sync::atomic::AtomicU64,
    /// Register writes as they happen, `(offset, value, width)`: the last
    /// `trace_cap` (IRIS_IP32_RE_TRACE) of them, the evidence for what a
    /// driver does with parts the model does not know yet. The most recent,
    /// so what was just drawn is in it.
    trace: Mutex<std::collections::VecDeque<(u32, u64, u8)>>,
    trace_cap: usize,
    /// The engine runs on its own thread ([`Self::start`]), working
    /// through the writes the CPU leaves in `ib`, as the real one works
    /// through its interface buffer while the CPU runs on. A write that
    /// finds the buffer full, and a register read while the engine is busy,
    /// report `BUS_BUSY`: the CPU retries them, taking interrupts meanwhile.
    ib: Box<IntfBuf>,
    /// The engine is carrying out a write it has taken from `ib`.
    busy: std::sync::atomic::AtomicBool,
    /// The last status read found the engine busy, and the CPU has written
    /// nothing since: a further status read is a guest waiting in a loop.
    /// Host pacing only, not CRIME state, so a snapshot does not keep it.
    polled_busy: std::sync::atomic::AtomicBool,
    asleep: std::sync::atomic::AtomicBool,
    started: std::sync::atomic::AtomicBool,
    engine: std::sync::OnceLock<std::thread::Thread>,
}

/// A buffer the RE draws through: which TLB, and bytes per pixel word.
#[derive(Clone, Copy)]
struct Buf {
    tlb: u32,
    bd: u32,
}

/// Pixel offsets along a row of a buffer, located once per tile: a tile row
/// is contiguous, so the rest of it is the first pixel's offset plus a
/// stride. Locating every pixel was a sixth of the engine's time under GL.
struct RowCursor {
    buf: Buf,
    y: u32,
    x_lo: u32,
    x_hi: u32,
    base: Option<usize>,
}

impl RowCursor {
    fn new(buf: Buf) -> Self {
        RowCursor { buf, y: u32::MAX, x_lo: 1, x_hi: 0, base: None }
    }

    #[inline]
    fn at(&mut self, r: &[u32], mem: &RamWindow, x: u32, y: u32) -> Option<usize> {
        if y != self.y || x < self.x_lo || x > self.x_hi {
            let tw = 512 / self.buf.bd;
            self.y = y;
            self.x_lo = x - x % tw;
            self.x_hi = self.x_lo + tw - 1;
            self.base = CrimeRe::locate(r, mem, self.buf, self.x_lo, y);
        }
        self.base.map(|b| b + ((x - self.x_lo) * self.buf.bd) as usize)
    }
}

/// The screen masks of [`CrimeRe::clip_ok`], decoded once per primitive:
/// testing every pixel against the registers was a sixth of the engine's
/// time under GL.
struct Clip {
    inside: [[i32; 4]; 5],
    n_inside: usize,
    outside: [[i32; 4]; 5],
    n_outside: usize,
}

impl Clip {
    fn new(r: &[u32]) -> Self {
        let mode = r[(CLIP_MODE / 4) as usize];
        let mut c = Clip { inside: [[0; 4]; 5], n_inside: 0, outside: [[0; 4]; 5], n_outside: 0 };
        for i in 0..5 {
            if mode & (1 << (5 + i)) == 0 {
                continue;
            }
            let (x0, y0) = CrimeRe::xy(r[((CLIP_RECT0 + 8 * i) / 4) as usize]);
            let (x1, y1) = CrimeRe::xy(r[((CLIP_RECT0 + 8 * i) / 4 + 1) as usize]);
            if mode & (1 << i) != 0 {
                c.inside[c.n_inside] = [x0, y0, x1, y1];
                c.n_inside += 1;
            } else {
                c.outside[c.n_outside] = [x0, y0, x1, y1];
                c.n_outside += 1;
            }
        }
        c
    }

    #[inline]
    fn ok(&self, x: i32, y: i32) -> bool {
        let hit = |b: &[i32; 4]| x >= b[0] && x < b[2] && y >= b[1] && y < b[3];
        (self.n_inside == 0 || self.inside[..self.n_inside].iter().any(hit)) && !self.outside[..self.n_outside].iter().any(hit)
    }
}

impl CrimeRe {
    pub fn new(ram: Arc<O2Memory>) -> Self {
        // byteMask resets to all ones: NetBSD's MTE clears never write it.
        let mut regs = vec![0; RE_WORDS];
        regs[(MTE_BYTEMASK / 4) as usize] = 0xffff_ffff;
        Self {
            regs: Mutex::new(regs),
            // Built in place: by way of the stack, a debug build's 1 MB
            // ring overflows a 2 MB test thread.
            // SAFETY: a zeroed GFifo is what `GFifo::new` returns.
            ib: unsafe { Box::<IntfBuf>::new_zeroed().assume_init() },
            busy: std::sync::atomic::AtomicBool::new(false),
            polled_busy: std::sync::atomic::AtomicBool::new(false),
            asleep: std::sync::atomic::AtomicBool::new(false),
            started: std::sync::atomic::AtomicBool::new(false),
            engine: std::sync::OnceLock::new(),
            hot: Mutex::new(BTreeMap::new()),
            win: Mutex::new(ram.window()),
            unhandled: Mutex::new(BTreeMap::new()),
            drawn: std::sync::atomic::AtomicU64::new(0),
            trace: Mutex::new(std::collections::VecDeque::new()),
            trace_cap: std::env::var("IRIS_IP32_RE_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
        }
    }

    /// Start the engine's thread. `owner` keeps this engine alive for the
    /// thread, which reaches it through `get`. Nothing is drawn before.
    pub fn start<O: Send + Sync + 'static>(&self, owner: Arc<O>, get: fn(&O) -> &CrimeRe) {
        if self.started.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        std::thread::Builder::new()
            .name("CRIME-RE".into())
            .spawn(move || {
                let re = get(&owner);
                let _ = re.engine.set(std::thread::current());
                re.run();
            })
            .expect("CRIME-RE thread");
    }

    /// The engine's thread: carry out each write in order. With none left
    /// it spins for `IDLE_SPIN` before it sleeps: drivers draw a little and
    /// wait for idle, over and over, and waking a parked thread (tens of
    /// microseconds on macOS) cost a GL clear a fifth of its speed. The spin
    /// backs off between looks: looking without pause took the buffer's
    /// tail away from the CPU on every write and lost a fifth of small
    /// triangles.
    fn run(&self) -> ! {
        use std::sync::atomic::Ordering::*;
        const IDLE_SPIN: std::time::Duration = std::time::Duration::from_micros(250);
        let backoff = crossbeam_utils::Backoff::new();
        let mut busy = false;
        let mut idle_since = None;
        loop {
            let Some((a, val)) = self.ib.peek() else {
                self.ib.flush_head();
                if busy {
                    busy = false;
                    self.busy.store(false, Release);
                }
                // Stays set across sleeps, so an idle engine wakes only for
                // the park backstop and goes straight back to sleep.
                let since = *idle_since.get_or_insert_with(std::time::Instant::now);
                if since.elapsed() < IDLE_SPIN {
                    backoff.spin();
                } else {
                    self.sleep();
                }
                continue;
            };
            idle_since = None;
            backoff.reset();
            if !busy {
                busy = true;
                self.busy.store(true, Release);
            }
            if a & WIDE != 0 {
                self.apply64(a & !WIDE, val);
            } else {
                self.write_reg(a, val as u32);
            }
            self.ib.consume();
        }
    }

    /// Park the engine's thread for up to 50 ms, unless a write came in
    /// before `asleep` was raised. The fence pairs with [`Self::wake`]'s.
    /// The timeout is only a backstop for a wake the fence protocol should
    /// never lose: long enough that an idle engine costs ~20 wakeups/s
    /// rather than 500.
    fn sleep(&self) {
        use std::sync::atomic::Ordering::*;
        self.asleep.store(true, Relaxed);
        std::sync::atomic::fence(SeqCst);
        if self.ib.is_empty() {
            std::thread::park_timeout(std::time::Duration::from_millis(50));
        }
        self.asleep.store(false, Relaxed);
    }

    /// How long a status read may wait for a busy engine: see
    /// [`RE_STATUS_POLL_WAIT`] and [`RE_STATUS_LOOP_WAIT`].
    fn status_wait(&self) -> std::time::Duration {
        if self.polled_busy.load(std::sync::atomic::Ordering::Relaxed) {
            RE_STATUS_LOOP_WAIT
        } else {
            RE_STATUS_POLL_WAIT
        }
    }

    /// After a write: unpark the engine's thread if it is asleep.
    fn wake(&self) {
        use std::sync::atomic::Ordering::*;
        std::sync::atomic::fence(SeqCst);
        if self.asleep.load(Relaxed) {
            if let Some(t) = self.engine.get() {
                t.unpark();
            }
        }
    }

    /// The engine has carried out every write it was given. Whatever it
    /// was working on is in `ib` until it is done (the head moves after
    /// `busy` is raised), so an empty buffer with `busy` clear is idle.
    fn idle(&self) -> bool {
        self.ib.is_empty() && !self.busy.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Wait up to `limit` for the engine to finish what is queued: a
    /// snapshot of its registers means nothing with writes still in the
    /// interface buffer.
    pub fn drain(&self, limit: std::time::Duration) -> bool {
        let until = std::time::Instant::now() + limit;
        let backoff = crossbeam_utils::Backoff::new();
        while !self.idle() {
            if std::time::Instant::now() >= until {
                return false;
            }
            backoff.snooze();
        }
        true
    }

    /// Wait until the engine is idle (tests and replays; the CPU retries).
    #[cfg(test)]
    fn wait_idle(&self) {
        let backoff = crossbeam_utils::Backoff::new();
        while !self.idle() {
            backoff.snooze();
        }
    }

    /// A CPU write, into the interface buffer: `BUS_BUSY` while it is full.
    fn cpu_write(&self, off: u32, val: u64, width: u8) -> u32 {
        if !self.ib.try_push(if width == 64 { off | WIDE } else { off }, val) {
            return BUS_BUSY;
        }
        self.polled_busy.store(false, std::sync::atomic::Ordering::Relaxed);
        self.record(off, val, width);
        self.wake();
        BUS_OK
    }

    fn record(&self, off: u32, val: u64, width: u8) {
        if self.trace_cap == 0 {
            return;
        }
        let mut t = self.trace.lock().unwrap();
        if t.len() >= self.trace_cap {
            t.pop_front();
        }
        t.push_back((off, val, width));
    }

    /// The recorded writes, one per line: `w32 OFFSET VALUE` (offsets from
    /// the RE base, 0x1500_0000).
    pub fn trace_text(&self) -> String {
        let t = self.trace.lock().unwrap();
        let mut s = String::with_capacity(t.len() * 28);
        for &(off, val, w) in t.iter() {
            s.push_str(&format!("w{w} {off:05x} {val:0pad$x}\n", pad = w as usize / 4));
        }
        s
    }

    pub fn unhandled(&self) -> BTreeMap<String, u64> {
        self.unhandled.lock().unwrap().clone()
    }

    fn note(&self, what: String) {
        let mut u = self.unhandled.lock().unwrap();
        let n = u.entry(what.clone()).or_insert(0);
        *n += 1;
        if *n == 1 {
            eprintln!("crime-re: not modelled: {what}");
        }
    }

    /// A CPU read: `None` (the CPU retries) for a register while the
    /// engine is busy.
    fn read_reg(&self, off: u32) -> Option<u32> {
        if crate::ip32::hot_counters() {
            *self.hot.lock().unwrap().entry(off & !3).or_insert(0) += 1;
        }
        match off & !3 {
            // Busy: the four idle bits clear, and the interface buffer's
            // level as CRIME's own 64 entries would show it: what is
            // queued while that fits, one entry past it, so software that
            // waits for room (the kernel's MTE copies wait for 64 - level)
            // always finds some and the CPU runs ahead of the engine.
            STATUS | 0x4004 => Some(if self.idle() || self.drain(self.status_wait()) {
                self.polled_busy.store(false, std::sync::atomic::Ordering::Relaxed);
                RE_STATUS_IDLE
            } else {
                self.polled_busy.store(true, std::sync::atomic::Ordering::Relaxed);
                let n = self.ib.len();
                let shown = if n < IB_HW_DEPTH { n } else { 1 };
                (shown as u32) << RE_STATUS_IB_LEVEL_SHIFT
            }),
            IB_CONTROL | 0x0404 => Some(0),
            // Any other read waits for the engine to catch up, as a read
            // through CRIME's interface buffer would: drivers can use one as
            // a barrier before touching what the engine draws.
            _ if !self.idle() => None,
            o => Some(self.regs.lock().unwrap().get((o / 4) as usize).copied().unwrap_or(0)),
        }
    }

    fn write_reg(&self, off: u32, val: u32) {
        let page = off & 0xf000;
        let start = (page == 0x2000 || page == 0x3000) && off & 0x800 != 0;
        let reg = if start { off & !0x800 } else { off };
        {
            let mut r = self.regs.lock().unwrap();
            if let Some(slot) = r.get_mut((reg / 4) as usize) {
                *slot = val;
            }
        }
        if start {
            if page == 0x2000 {
                self.pixel_pipe();
            } else {
                self.mte();
            }
        }
    }

    fn buf(&self, r: &[u32], mode: u32, what: &str) -> Option<Buf> {
        let b = (mode >> 10) & 7;
        let bd = 1 << ((mode >> 8) & 3).min(2);
        if b > 2 {
            self.note(format!("{what} buffer {b} (linear/other)"));
            return None;
        }
        let _ = r;
        Some(Buf { tlb: TLB_A + 0x200 * b, bd })
    }

    /// Byte offset in `mem` of pixel (x, y) of `buf`, through its tile TLB.
    fn locate(r: &[u32], mem: &RamWindow, buf: Buf, x: u32, y: u32) -> Option<usize> {
        let tw = 512 / buf.bd;
        let entry = (y / 128) * 16 + x / tw;
        if entry >= 256 {
            return None;
        }
        let word = buf.tlb + 8 * (entry / 4);
        let w64 = ((r[(word / 4) as usize] as u64) << 32) | r[(word / 4 + 1) as usize] as u64;
        let e = (w64 >> (48 - 16 * (entry % 4))) & 0xffff;
        if e & 0x8000 == 0 {
            return None;
        }
        // The tile's offset in memory, from a one-entry cache: a primitive
        // stays in one tile for many pixels, and resolving the tile through
        // the memory decode every pixel was a fifth of the engine's time
        // under perfly.
        // (In the memory, not a thread-local: on macOS each thread-local
        // access is a call.)
        let tile = e & 0x7fff;
        let key = tile | (mem.decode_gen() << 16);
        let (k, b) = mem.tile.get();
        let base = if k == key {
            b
        } else {
            let b = mem.dma_offset((tile as u32) << 16).filter(|&o| mem.holds(o, 0x10000))?;
            mem.tile.set((key, b));
            b
        };
        Some(base + ((y % 128) * 512 + (x % tw) * buf.bd) as usize)
    }

        fn get_px(mem: &RamWindow, o: usize, bd: u32) -> u32 {
        mem.get_be(o, bd)
    }

    /// A register colour as a pixel of `bd` bytes: its low bytes, wherever
    /// the pixel sits in its word. IRIX's X server writes an 8-bit index as
    /// is (FG 0x00000002); the BSDs repeat it in every byte, which reads
    /// the same either way.
    fn lane(v: u32, _o: usize, bd: u32) -> u32 {
        if bd >= 4 { v } else { v & ((1 << (8 * bd)) - 1) }
    }

    fn logic(op: u32, s: u32, d: u32) -> u32 {
        match op & 0xf {
            0 => 0,
            1 => s & d,
            2 => s & !d,
            3 => s,
            4 => !s & d,
            5 => d,
            6 => s ^ d,
            7 => s | d,
            8 => !(s | d),
            9 => !(s ^ d),
            10 => !d,
            11 => s | !d,
            12 => !s,
            13 => !s | d,
            14 => !(s & d),
            _ => u32::MAX,
        }
    }

    /// Write `src` to the pixel at `o`, through the logic op, plane mask and
    /// byte-lane enables.
        fn put(r: &[u32], mem: &mut RamWindow, o: usize, bd: u32, src: u32) {
        let dm = r[(DRAWMODE / 4) as usize];
        let reads = dm & (draw::LOGIC_OP | draw::PLANE_MASK) != 0;
        let old = if reads { Self::get_px(mem, o, bd) } else { 0 };
        let mut v = if dm & draw::LOGIC_OP != 0 { Self::logic(r[(LOGIC_OP / 4) as usize], src, old) } else { src };
        if dm & draw::PLANE_MASK != 0 {
            let m = Self::lane(r[(COLOR_MASK / 4) as usize], o, bd);
            v = (v & m) | (old & !m);
        }
        let lanes = (dm >> 3) & 0xf;
        if lanes == 0xf {
            mem.put_be(o, bd, v);
        } else {
            // The enabled byte lanes (lane bit 3 - (address & 3) for each
            // byte) as a mask over the pixel: one read-modify-write.
            let mut m = 0u32;
            for k in 0..bd as usize {
                if lanes & (1 << (3 - ((o + k) & 3))) != 0 {
                    m |= 0xff << (8 * (bd as usize - 1 - k));
                }
            }
            if m != 0 {
                let cur = if reads { old } else { Self::get_px(mem, o, bd) };
                mem.put_be(o, bd, (cur & !m) | (v & m));
            }
        }
    }

    fn xy(v: u32) -> (i32, i32) {
        ((v >> 16) as i32, (v & 0xffff) as i32)
    }

    /// Whether the screen masks let pixel (x, y) be drawn. CLIP_MODE bit
    /// 5 + i enables ScrMask[i] (CLIP_RECT0 + 8i, 64 bits: x0, y0 high, x1,
    /// y1 low, the max edges exclusive) and bit i says whether a pixel must
    /// be inside it (1) or outside it (0). A pixel must lie in one of the
    /// enabled inside masks, if there are any, and in none of the outside
    /// ones. The spec's table numbers the masks the other way round; IRIX
    /// pairs bit 5 with ScrMask[0] (mode 0x21 clips to the Icon Catalog's
    /// rectangle in mask 0, the other four zero). 4Dwm fills a minimized
    /// window's frame with masks 0-3 (mode 0x1ef), which keeps the fill off
    /// the icon picture, and redraws a whole frame with all five (0x3ff):
    /// mask 4 is the title bar, so reading only four left titles blank.
    fn clip_ok(r: &[u32], x: i32, y: i32) -> bool {
        Clip::new(r).ok(x, y)
    }

    /// A window offset: two signed 16-bit halves. IRIX's GL windows sit at
    /// offsets like (-3443, -3477) -- the window origin less a 4096 guard --
    /// which read unsigned put every triangle 62,000 pixels off screen.
    fn offset_xy(v: u32) -> (i32, i32) {
        ((v >> 16) as u16 as i16 as i32, v as u16 as i16 as i32)
    }

    /// Run the pixel pipe's current primitive.
    /// A 32-bit pixel moved between buffers whose pixel types (BufMode
    /// 7:4) differ: through RGBA. RGB keeps its colour in the low three
    /// bytes (0x00RRGGBB) and ABGR is RGBA reversed. IRIX's X server keeps
    /// pixmaps as RGB in linear B and the screen as RGBA; copying the word
    /// unchanged gave 0x00c0c0c0's grey back as 0x00c0c0 -- cyan.
    fn convert(v: u32, from_mode: u32, to_mode: u32, bd: u32) -> u32 {
        let (from, to) = ((from_mode >> 4) & 0xf, (to_mode >> 4) & 0xf);
        if from == to || bd != 4 {
            return v;
        }
        let rgba = match from {
            1 => (v << 8) | 0xff,
            3 => v.swap_bytes(),
            _ => v,
        };
        match to {
            1 => rgba >> 8,
            3 => rgba.swap_bytes(),
            _ => rgba,
        }
    }

    /// A rectangle pixel transfer with a linear buffer at either end: the
    /// X server keeps icons and saved screen areas in linear B and copies
    /// them in and out this way. The linear end starts at a byte address
    /// (XFER_SRC for a source, XFER_DST for a destination) for the
    /// rectangle's top-left pixel and moves XFER_STEP_X bytes per pixel and
    /// XFER_STEP_ROW from the end of one row to the start of the next. The
    /// login screen's copies out of a 16-pixel-wide pixmap all have steps 4
    /// and 4 + (16 - width) x 4: a 64-byte pitch whatever the width.
    fn xfer_linear(&self, r: &[u32]) {
        let g = |off: u32| r[(off / 4) as usize];
        let clip = Clip::new(r);
        let (sm, dmode) = (g(BUFMODE_SRC), g(BUFMODE_DST));
        let (sb, db) = ((sm >> 10) & 7, (dmode >> 10) & 7);
        let (sbd, dbd) = (1u32 << ((sm >> 8) & 3).min(2), 1u32 << ((dmode >> 8) & 3).min(2));
        let lin = |b: u32| b == 4 || b == 5;
        if !lin(sb) && sb <= 2 && lin(db) {
            return self.xfer_to_linear(r);
        }
        if (!lin(sb) && sb > 2) || (!lin(db) && db > 2) {
            self.note(format!("xfer buffers src {sb} dst {db}"));
            return;
        }
        let (ox, oy) = Self::offset_xy(g(WINOFFSET_DST));
        let (x0, y0) = Self::xy(g(VERTEX_0));
        let (x1, y1) = Self::xy(g(VERTEX_1));
        let (x0, y0, x1, y1) = (x0 + ox, y0 + oy, x1 + ox, y1 + oy);
        let (xa, xb) = (x0.min(x1).max(0), x0.max(x1).max(0));
        let (ya, yb) = (y0.min(y1).max(0), y0.max(y1).max(0));
        if xb - xa > 4096 || yb - ya > 4096 {
            self.note(format!("xfer rect {xa},{ya}..{xb},{yb}"));
            return;
        }
        let (step_x, step_row) = (g(XFER_STEP_X), g(XFER_STEP_ROW));
        // A linear end is addressed in bytes, but its pixels are words of
        // the buffer's depth; an address off that alignment is taken down
        // to it (demograph's transfers aborted the emulator on 0x...001).
        let skew = |b: u32, bd: u32, a: u32| lin(b) && (a | step_x | step_row) & (bd - 1) != 0;
        if skew(sb, sbd, g(XFER_SRC)) || skew(db, dbd, g(XFER_DST)) {
            self.note(format!("xfer unaligned linear end (src {sb}/{sbd}, dst {db}/{dbd})"));
        }
        let width = (xb - xa) as u32;
        // Offset of pixel (x, y) from the top-left one, along the linear end.
        let lin_off = |x: i32, y: i32| -> u32 {
            let row = width.wrapping_mul(step_x).wrapping_add(step_row);
            ((y - ya) as u32).wrapping_mul(row).wrapping_add(((x - xa) as u32).wrapping_mul(step_x))
        };
        let (sx0, sy0) = Self::xy(g(XFER_SRC));
        let src_tiled = Buf { tlb: TLB_A + 0x200 * sb.min(2), bd: sbd };
        let dst_tiled = Buf { tlb: TLB_A + 0x200 * db.min(2), bd: dbd };
        let mut ram = Draw::new(self.win.lock().unwrap());
        let mem: &mut RamWindow = &mut ram;
        let src_at = |mem: &RamWindow, x: i32, y: i32| -> Option<usize> {
            if lin(sb) {
                let a = g(XFER_SRC).wrapping_add(lin_off(x, y));
                Self::linear(r, mem, sb - 4, a & !(sbd - 1))
            } else {
                let (px, py) = (x + sx0 - x0, y + sy0 - y0);
                if px < 0 || py < 0 {
                    return None;
                }
                Self::locate(r, mem, src_tiled, px as u32, py as u32)
            }
        };
        let dst_at = |mem: &RamWindow, x: i32, y: i32| -> Option<usize> {
            if lin(db) {
                let a = g(XFER_DST).wrapping_add(lin_off(x, y));
                Self::linear(r, mem, db - 4, a & !(dbd - 1))
            } else {
                Self::locate(r, mem, dst_tiled, x as u32, y as u32)
            }
        };
        // Read the whole source first, as the tiled copy does.
        let blend = Self::xfer_blends(r, dbd);
        let mut tmp = Vec::with_capacity(((xb - xa + 1) * (yb - ya + 1)) as usize);
        for y in ya..=yb {
            for x in xa..=xb {
                let v = src_at(mem, x, y).map(|o| Self::get_px(mem, o, sbd)).unwrap_or(0);
                tmp.push(if blend { v } else { Self::convert(v, sm, dmode, sbd.min(dbd)) });
            }
        }
        let mut it = tmp.into_iter();
        for y in ya..=yb {
            for x in xa..=xb {
                let v = it.next().unwrap_or(0);
                if !clip.ok(x, y) {
                    continue;
                }
                if let Some(o) = dst_at(mem, x, y) {
                    let v = if blend { Self::blend_xfer(r, v, Self::get_px(mem, o, 4)) } else { v };
                    Self::put(r, mem, o, dbd, v);
                }
            }
        }
    }

    /// A rectangle read out of a tiled buffer into a linear one: how IRIX's
    /// OpenGL reads pixels back (the kernel transfers them into a buffer the
    /// library then copies out).
    ///
    /// The rectangle (Primitive op-code 3) has its corners in the GL vertex
    /// registers (13.6 fixed point) under DrawMode's GL bit, else in the X
    /// ones, both ends included; only its size matters here. The source end
    /// starts at XFER_SRC, a pixel in window coordinates that the source
    /// window offset places on the screen modulo 4096 (GL windows carry the
    /// 4096-pixel guard band in the offset), and moves XFER_STEP_X /
    /// XFER_STEP_ROW pixels per column and row, in the directions the
    /// Primitive register's edge type gives (bottom to top is up the
    /// screen). The destination end starts at XFER_DST, a byte address in
    /// the linear buffer, moves a pixel's width per column and
    /// XFER_DST_STRIDE bytes from a row's last pixel to the next row's
    /// first, as a linear source's XFER_STEP_ROW does: rows are packed when
    /// it equals the pixel width (IRIX's GL: 4 at 32 bits; its X server: 1
    /// at 8).
    fn xfer_to_linear(&self, r: &[u32]) {
        let g = |off: u32| r[(off / 4) as usize];
        let (sm, dmode) = (g(BUFMODE_SRC), g(BUFMODE_DST));
        let (sb, db) = ((sm >> 10) & 7, (dmode >> 10) & 7);
        let (sbd, dbd) = (1u32 << ((sm >> 8) & 3).min(2), 1u32 << ((dmode >> 8) & 3).min(2));
        let ((x0, y0), (x1, y1)) = if g(DRAWMODE) & draw::GL != 0 {
            let fx = |off: u32| (g(off) as i32) >> 6;
            ((fx(TRI_V0), fx(TRI_V0 + 4)), (fx(TRI_V0 + 8), fx(TRI_V0 + 12)))
        } else {
            (Self::xy(g(VERTEX_0)), Self::xy(g(VERTEX_1)))
        };
        let (w, h) = ((x1 - x0).abs() + 1, (y1 - y0).abs() + 1);
        if w > 4096 || h > 4096 {
            self.note(format!("xfer to linear {w}x{h}"));
            return;
        }
        let edge = (g(PRIMITIVE) >> 16) & 3;
        let (dx, dy) = (if edge & 1 != 0 { -1 } else { 1 }, if edge & 2 != 0 { 1 } else { -1 });
        let (sx0, sy0) = Self::xy(g(XFER_SRC));
        let (step_x, step_y) = (g(XFER_STEP_X) as i32 * dx, g(XFER_STEP_ROW) as i32 * dy);
        let (ox, oy) = Self::offset_xy(g(WINOFFSET_SRC));
        let src = Buf { tlb: TLB_A + 0x200 * sb, bd: sbd };
        let dst0 = g(XFER_DST);
        let pitch = ((w - 1) as u32).wrapping_mul(dbd).wrapping_add(g(XFER_DST_STRIDE));
        let mut ram = Draw::new(self.win.lock().unwrap());
        let mem: &mut RamWindow = &mut ram;
        for j in 0..h {
            let row = dst0.wrapping_add((j as u32).wrapping_mul(pitch));
            for i in 0..w {
                let sx = ((sx0 + i * step_x + ox) & 0xfff) as u32;
                let sy = ((sy0 + j * step_y + oy) & 0xfff) as u32;
                let v = Self::locate(r, mem, src, sx, sy).map(|o| Self::get_px(mem, o, sbd)).unwrap_or(0);
                let a = row.wrapping_add(i as u32 * dbd);
                if let Some(o) = Self::linear(r, mem, db - 4, a & !(dbd - 1)) {
                    Self::put(r, mem, o, dbd, Self::convert(v, sm, dmode, sbd.min(dbd)));
                }
            }
        }
    }

    /// Whether a transfer into a buffer of `dbd`-byte words blends: with
    /// DrawMode.enBlend set and an RGB(A) destination of 32-bit words, not
    /// split for double buffering. NetBSD's X server draws antialiased text
    /// and other Render composites this way, a linear buffer of pixels with
    /// alpha blended onto the screen.
    fn xfer_blends(r: &[u32], dbd: u32) -> bool {
        let (dm, dmode) = (r[(DRAWMODE / 4) as usize], r[(BUFMODE_DST / 4) as usize]);
        dm & draw::BLEND != 0 && dbd == 4 && (dmode >> 4) & 0xf != 0 && dmode & 2 == 0
    }

    /// Source pixel `s`, as read from its buffer, blended over destination
    /// pixel `d` by the Blend register, in the destination's pixel type. A
    /// pixel narrower than its 32-bit type arrives zero-extended, so an
    /// 8-bit buffer of RGBA pixels gives alpha only: NetBSD blends black
    /// text from its 8-bit glyph masks so, with a ZERO source factor.
    fn blend_xfer(r: &[u32], s: u32, d: u32) -> u32 {
        let g = |off: u32| r[(off / 4) as usize];
        let (smode, dmode) = (g(BUFMODE_SRC), g(BUFMODE_DST));
        let split = |v: u32, mode: u32| {
            let v = Self::convert(v, mode, 0x20, 4);
            [v >> 24, (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff]
        };
        let [r8, g8, b8, a8] = Self::blend(g(BLEND_FUNC), g(BLEND_CONST), split(s, smode), split(d, dmode));
        Self::convert(r8 << 24 | g8 << 16 | b8 << 8 | a8, 0x20, dmode, 4)
    }

    fn pixel_pipe(&self) {
        self.drawn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The registers stay locked for the primitive: copying all 24 KB of
        // them per primitive was a tenth of the engine's time under X.
        let r = self.regs.lock().unwrap();
        let g = |off: u32| r[(off / 4) as usize];
        let dm = g(DRAWMODE);
        let prim = g(PRIMITIVE);
        let clip = Clip::new(&r);
        // Clip-ID testing (ClipMode bit 11) is not modelled: the buffer's
        // layout is unknown (TLB.cid maps 8 tiles, the boot-time clear spans
        // 256 x 1024 bytes) and nothing seen so far draws with it on, only
        // sets the bit between primitives. Say so the first time it matters.
        if g(CLIP_MODE) & (1 << 11) != 0 {
            self.note("clip-ID test (drawn unclipped)".into());
        }
        if dm & draw::PIXEL_XFER != 0 && (prim >> 24) & 3 == 3 {
            let (sb, db) = ((g(BUFMODE_SRC) >> 10) & 7, (g(BUFMODE_DST) >> 10) & 7);
            if matches!(sb, 4 | 5) || matches!(db, 4 | 5) {
                self.xfer_linear(&r);
                return;
            }
        }
        let Some(dst) = self.buf(&r, g(BUFMODE_DST), "dst") else { return };
        let (ox, oy) = Self::offset_xy(g(WINOFFSET_DST));
        // In GL mode points and rectangles take their corners from the GL
        // vertex registers too (perfly's panel text: one stippled GL
        // rectangle per glyph row).
        let ((x0, y0), (x1, y1)) = if dm & draw::GL != 0 {
            let v = |off: u32| (gl_coord(g(off)).floor() as i32, gl_coord(g(off + 4)).floor() as i32);
            (v(TRI_V0), v(TRI_V0 + 8))
        } else {
            (Self::xy(g(VERTEX_0)), Self::xy(g(VERTEX_1)))
        };
        let (x0, y0, x1, y1) = (x0 + ox, y0 + oy, x1 + ox, y1 + oy);
        let fg = g(FG);
        let bg = g(BG);
        let smode = g(STIPPLE_MODE);
        let (sstart, smax) = ((smode >> 24) & 0x1f, (smode >> 16) & 0x1f);
        let pattern = g(STIPPLE_PATTERN);
        let stipple = |k: u32| -> Option<bool> {
            let span = smax + 1;
            let idx = (sstart + k) % span.max(1);
            Some((pattern >> (31 - idx)) & 1 != 0)
        };
        let mut ram = Draw::new(self.win.lock().unwrap());
        let mem: &mut RamWindow = &mut ram;

        // Which colour register a destination pixel takes, if any.
        let colour_at = |mem: &RamWindow, _o: usize, k: u32, stippled: bool| -> Option<u32> {
            let _ = mem;
            if stippled {
                match stipple(k) {
                    Some(true) => Some(fg),
                    _ if dm & draw::OPAQUE_STIPPLE != 0 => Some(bg),
                    _ => None,
                }
            } else {
                Some(fg)
            }
        };
        // Write a register colour: its low bytes for a narrow pixel, or, in a
        // double-pix window (as for triangles), the colour (RGBA, R high) as
        // A1R5G5B5 into the half the buffer mode picks. perfly's panel text
        // is FG 0x190c19ff into such a window. A 32-bit pixel takes the
        // colour in the destination's pixel type, as triangles do: X fills
        // its ABGR windows with FG 0x00d4d4d4, and storing that unchanged
        // showed the grey as 0x00d4d4 (the cyan Motif bevels).
        let dmode = g(BUFMODE_DST);
        let half = dmode & 2 != 0 && (dmode >> 2) & 3 == 1 && dst.bd == 4;
        let half_off = if dmode & 1 == 0 { 2 } else { 0 };
        let paint = |mem: &mut RamWindow, o: usize, c: u32| {
            if half {
                let v = (((c & 0xff) >= 0x80) as u32) << 15 | ((c >> 27) & 31) << 10 | ((c >> 19) & 31) << 5 | ((c >> 11) & 31);
                Self::put(&r, mem, o + half_off, 2, v);
            } else {
                Self::put(&r, mem, o, dst.bd, Self::lane(Self::convert(c, 0x20, dmode, dst.bd), o, dst.bd));
            }
        };

        match (prim >> 24) & 3 {
            // A shaded GL point: a line from the vertex to itself, so it
            // takes the shade colour, blending and a double-pix half as GL
            // lines do. bz's world is shaded GL points; painted with FG
            // they all came out black. (A flat one, or one into an 8-bit
            // colour-index buffer such as flight's, keeps FG as it stands.)
            0 if dm & draw::GL != 0 && dm & draw::SHADE != 0 && dst.bd == 4 => {
                Self::gl_line(&r, mem, dst, ox, oy, false, true);
            }
            // Point.
            0 => {
                if !clip.ok(x0, y0) {
                    return;
                }
                if let Some(o) = Self::locate(&r, mem, dst, x0.max(0) as u32, y0.max(0) as u32) {
                    paint(mem, o, fg);
                }
            }
            // Line, Bresenham, the end pixel unless told to skip it.
            1 if dm & draw::GL != 0 => {
                let skip_last = prim & (1 << 18) != 0;
                Self::gl_line(&r, mem, dst, ox, oy, skip_last, false);
            }
            1 => {
                let stippled = dm & draw::LINE_STIPPLE != 0;
                let skip_last = prim & (1 << 18) != 0;
                let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
                let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
                let (mut x, mut y, mut err, mut k) = (x0, y0, dx + dy, 0u32);
                loop {
                    let last = x == x1 && y == y1;
                    if !(last && skip_last) && x >= 0 && y >= 0 && clip.ok(x, y) {
                        if let Some(o) = Self::locate(&r, mem, dst, x as u32, y as u32) {
                            if let Some(v) = colour_at(mem, o, k, stippled) {
                                paint(mem, o, v);
                            }
                        }
                    }
                    if last || k > 8192 {
                        break;
                    }
                    k += 1;
                    let e2 = 2 * err;
                    if e2 >= dy {
                        err += dy;
                        x += sx;
                    }
                    if e2 <= dx {
                        err += dx;
                        y += sy;
                    }
                }
            }
            // Rectangle: both corners inclusive.
            3 => {
                let (xa, xb) = (x0.min(x1).max(0) as u32, x0.max(x1).max(0) as u32);
                let (ya, yb) = (y0.min(y1).max(0) as u32, y0.max(y1).max(0) as u32);
                if (xb - xa) > 4096 || (yb - ya) > 4096 {
                    self.note(format!("rect {xa},{ya}..{xb},{yb}"));
                    return;
                }
                if dm & draw::PIXEL_XFER != 0 {
                    // A copy, read whole before any write, so overlaps come
                    // out as the drivers' traversal order intends.
                    let Some(src) = self.buf(&r, g(BUFMODE_SRC), "src") else { return };
                    let (sx0, sy0) = Self::xy(g(XFER_SRC));
                    let (dxs, dys) = (sx0 - x0, sy0 - y0);
                    let blend = Self::xfer_blends(&r, dst.bd);
                    let mut tmp = Vec::with_capacity(((xb - xa + 1) * (yb - ya + 1)) as usize);
                    for y in ya..=yb {
                        for x in xa..=xb {
                            let (px, py) = (x as i32 + dxs, y as i32 + dys);
                            let v = if px >= 0 && py >= 0 {
                                Self::locate(&r, mem, src, px as u32, py as u32).map(|o| Self::get_px(mem, o, src.bd)).unwrap_or(0)
                            } else {
                                0
                            };
                            tmp.push(if blend { v } else { Self::convert(v, g(BUFMODE_SRC), g(BUFMODE_DST), src.bd.min(dst.bd)) });
                        }
                    }
                    let mut it = tmp.into_iter();
                    for y in ya..=yb {
                        for x in xa..=xb {
                            let v = it.next().unwrap_or(0);
                            if !clip.ok(x as i32, y as i32) {
                                continue;
                            }
                            if let Some(o) = Self::locate(&r, mem, dst, x, y) {
                                let v = if blend { Self::blend_xfer(&r, v, Self::get_px(mem, o, 4)) } else { v };
                                Self::put(&r, mem, o, dst.bd, v);
                            }
                        }
                    }
                } else {
                    let stippled = dm & draw::POLY_STIPPLE != 0;
                    for y in ya..=yb {
                        for (k, x) in (xa..=xb).enumerate() {
                            if !clip.ok(x as i32, y as i32) {
                                continue;
                            }
                            if let Some(o) = Self::locate(&r, mem, dst, x, y) {
                                if let Some(v) = colour_at(mem, o, k as u32, stippled) {
                                    paint(mem, o, v);
                                }
                            }
                        }
                    }
                }
            }
            // Triangle: three float vertices at 0x2080/0x2088/0x2090 (x in
            // the high word, y in the low), biased by 2^17 so the mantissa
            // holds 1/64-pixel fixed point, plus the signed window offset.
            // Colour from the shade registers, 8.12 fixed RGBA. IRIX's
            // desktop fills its icon backgrounds and vector icons this way,
            // flat-shaded and opaque.
            2 => {
                let fv = |off: u32| -> (f64, f64) {
                    let f = gl_coord;
                    (f(g(off)) + ox as f64, f(g(off + 4)) + oy as f64)
                };
                let v = [fv(TRI_V0), fv(TRI_V0 + 8), fv(TRI_V0 + 16)];
                // Gouraud: each channel is its 8.12 start value plus its x and
                // y gradients times the distance from the reference point.
                // Register order: R0 G0 B0 A0, then dR/dx dG/dx dR/dy dG/dy
                // dB/dx dA/dx dB/dy dA/dy.
                let (frac, refv) = gouraud_model();
                let grad = |i: u32| g(SHADE_GRAD + 4 * i) as i32 as f64 / (1u64 << frac) as f64;
                let (ddx, ddy) = ([grad(0), grad(1), grad(4), grad(5)], [grad(2), grad(3), grad(6), grad(7)]);
                // Shading off (draw mode bit 16 clear): flat FG, as points and
                // rectangles always are; the shade registers then hold
                // whatever the last shaded primitive left (perfly's textured
                // hubcaps: FG white, R0 negative).
                let smooth = dm & draw::SHADE != 0;
                let shaded = smooth && (0..8).any(|i| g(SHADE_GRAD + 4 * i) != 0);
                // The start point the driver gives every attribute (colour,
                // depth, texture) for: the vertex the traversal begins at,
                // the bottom one for bottom-to-top edge types (0, 1) and the
                // top one otherwise, the left or right one of two by the
                // horizontal direction, at its pixel's corner. Fitted on GL
                // triangles with known vertex colours and texture
                // coordinates: every vertex comes out exact there (an
                // earlier fit to perfly put it at (1/16, 19/64) into the
                // pixel, before the texture scale flags below were known,
                // and was 2-7% out at a far vertex).
                let (rx, ry) = if refv < 3 {
                    v[refv]
                } else {
                    let et = (prim >> 16) & 3;
                    let rightward = et & 1 == 0;
                    let start = v.iter().copied().fold(None::<(f64, f64)>, |best, p| match best {
                        None => Some(p),
                        Some(b) => {
                            let better_y = if et < 2 { p.1 > b.1 } else { p.1 < b.1 };
                            let tie_x = p.1 == b.1 && if rightward { p.0 < b.0 } else { p.0 > b.0 };
                            Some(if better_y || tie_x { p } else { b })
                        }
                    }).unwrap();
                    (start.0.floor(), start.1.floor())
                };
                let fgc = g(FG);
                let c0 = if smooth {
                    [0u32, 4, 8, 12].map(|o| g(SHADE_R + o) as i32 as f64 / 4096.0)
                } else {
                    [24u32, 16, 8, 0].map(|sh| ((fgc >> sh) & 0xff) as f64)
                };
                let channel = |k: usize, px: f64, py: f64| -> u32 {
                    let c = if shaded { c0[k] + ddx[k] * (px - rx) + ddy[k] * (py - ry) } else { c0[k] };
                    c.clamp(0.0, 255.0) as u32
                };
                let (cr, cg, cb, ca) = (channel(0, rx, ry), channel(1, rx, ry), channel(2, rx, ry), channel(3, rx, ry));
                let dmode = g(BUFMODE_DST);
                // Double-pix (BufMode bit 1) with 16-bit pixels (3:2 = 1): a
                // double-buffered GL window keeps front and back in the two
                // halves of each 32-bit word, bit 0 choosing the half (0 the
                // low one, which IRIX's WID for it, 0x11, shows). The 16-bit
                // pixel is A1R5G5B5, as GBE decodes it.
                let half = dmode & 2 != 0 && (dmode >> 2) & 3 == 1 && dst.bd == 4;
                // Dithering into the 5-bit channels of a double-pix window
                // (Quake 2's): without it a lightmapped wall falls into a few
                // flat steps of 8 levels each.
                let dither = dm & draw::DITHER != 0 && half;
                let pack_at = |x: i32, y: i32, cr: u32, cg: u32, cb: u32, ca: u32| -> u32 {
                    let d = |c: u32| Self::dither5(c, x, y);
                    (u32::from(ca >= 0x80) << 15) | (d(cr) << 10) | (d(cg) << 5) | d(cb)
                };
                let pack = |cr: u32, cg: u32, cb: u32, ca: u32| -> u32 {
                    if half {
                        let c5 = |c: u32| c >> 3;
                        (u32::from(ca >= 0x80) << 15) | (c5(cr) << 10) | (c5(cg) << 5) | c5(cb)
                    } else {
                        let rgba = (cr << 24) | (cg << 16) | (cb << 8) | ca;
                        match (dmode >> 4) & 0xf {
                            0 => cr,
                            1 => rgba >> 8,
                            3 => rgba.swap_bytes(),
                            _ => rgba,
                        }
                    }
                };
                // A flat colour-index triangle (pixel type 0) takes its index
                // from FG as it stands, the low byte, as points, lines and
                // rectangles do: read as RGBA it is the red channel, FG >> 24,
                // which drew gr_osview's index-2 bars as index 0.
                let flat = if (dmode >> 4) & 0xf == 0 && !smooth { fgc } else { pack(cr, cg, cb, ca) };
                // Texture (draw mode bit 15): s/q and t/q from the same start
                // point as the colours, modulating them (perfly's texture
                // function; the others are not modelled yet).
                let tex = if dm & draw::TEXTURE != 0 {
                    let t = Texture::from_regs(&r);
                    if t.is_none() {
                        self.note(format!("texture mode {:#x} (drawn untextured)", g(TEX_MODE)));
                    }
                    t
                } else {
                    None
                };
                let s64r = |off: u32| (((g(off) as u64) << 32) | g(off + 4) as u64) as i64 as f64;
                let s32r = |off: u32| g(off) as i32 as f64;
                let (sq0, tq0, q0) = (s64r(TEX_SQ0), s64r(TEX_TQ0), s32r(TEX_Q0));
                let (dsx, dsy, dtx, dty) = (s64r(TEX_DSQDX), s64r(TEX_DSQDY), s64r(TEX_DTQDX), s64r(TEX_DTQDY));
                let (dqx, dqy) = (s32r(TEX_DQDX), s32r(TEX_DQDY));
                // A coordinate the driver scaled down to fit the engine's
                // range (s, t past about 16 maps) is flagged here, bit 0 for
                // s and bit 1 for t, and read 16 times its register value.
                let flags = g(TEX_COORD_SCALE);
                let (ks, kt) = (if flags & 1 != 0 { 16.0 } else { 1.0 }, if flags & 2 != 0 { 16.0 } else { 1.0 });
                // The source colour at a pixel as RGBA, before packing: what
                // the alpha test and blending work on.
                let rgba_at = |mem: &RamWindow, px: f64, py: f64| -> [u32; 4] {
                    let base = |px: f64, py: f64| [0, 1, 2, 3].map(|k| channel(k, px, py));
                    let Some(t) = &tex else {
                        return if shaded { base(px, py) } else { [cr, cg, cb, ca] };
                    };
                    let (dx, dy) = (px - rx, py - ry);
                    let q = q0 + dqx * dx + dqy * dy;
                    if q == 0.0 {
                        return [cr, cg, cb, ca];
                    }
                    let st = |dx: f64, dy: f64| {
                        let q = q0 + dqx * dx + dqy * dy;
                        (ks * (sq0 + dsx * dx + dsy * dy) / q, kt * (tq0 + dtx * dx + dty * dy) / q)
                    };
                    let (s0, t0) = st(dx, dy);
                    // Level of detail from the texel steps to the next pixel
                    // in x and y, at the base level (spec 7.3.7.5).
                    let lod = {
                        let ((sx, tx_), (sy, ty)) = (st(dx + 1.0, dy), st(dx, dy + 1.0));
                        let (kw, kh) = (t.w as f64 / 16384.0, t.h as f64 / 16384.0);
                        let rho = ((sx - s0) * kw).abs().max(((tx_ - t0) * kh).abs()).max(((sy - s0) * kw).abs()).max(((ty - t0) * kh).abs());
                        if rho > 0.0 { rho.log2() } else { 0.0 }
                    };
                    let tx = t.sample(&r, mem, s0, t0, lod);
                    // Texture function (Texture.mode 1:0), as OpenGL's for
                    // maps with and without alpha.
                    let f = [0, 1, 2, 3].map(|k| channel(k, px, py) as f64);
                    let rgba_map = t.has_alpha();
                    let out = |v: f64| (v.round() as u32).min(255);
                    match g(TEX_MODE) & 3 {
                        // REPLACE
                        3 => {
                            let c = |k: usize| out(if t.has_rgb() { tx[k] } else { f[k] });
                            [c(0), c(1), c(2), out(if rgba_map { tx[3] } else { f[3] })]
                        }
                        // DECAL: the map over the fragment by the map's alpha.
                        1 => {
                            let ta = if rgba_map { tx[3] / 255.0 } else { 1.0 };
                            let mix = |k: usize| out(f[k] * (1.0 - ta) + tx[k] * ta);
                            [mix(0), mix(1), mix(2), out(f[3])]
                        }
                        // MODULATE, and BLEND until its colour is found.
                        _ => {
                            let m = |k: usize| out(f[k] * tx[k] / 255.0);
                            [m(0), m(1), m(2), if rgba_map { m(3) } else { out(f[3]) }]
                        }
                    }
                };
                let colour_at = |mem: &RamWindow, px: f64, py: f64| -> u32 {
                    if tex.is_none() && !shaded {
                        return flat;
                    }
                    let [r8, g8, b8, a8] = rgba_at(mem, px, py);
                    pack(r8, g8, b8, a8)
                };
                // Alpha test and blending need the colour as RGBA; neither
                // applies to colour index (pixel type 0).
                let rgb_dst = (dmode >> 4) & 0xf != 0 || half;
                let atest = dm & draw::ALPHA_TEST != 0 && rgb_dst;
                let blend = dm & draw::BLEND != 0 && rgb_dst && dst.bd == 4;
                let (afunc, aref) = ((g(ALPHA_TEST) >> 8) & 7, g(ALPHA_TEST) & 0xff);
                let (bfunc, bconst) = (g(BLEND_FUNC), g(BLEND_CONST));
                let unpack = |v: u32| -> [u32; 4] {
                    if half {
                        let e = |c: u32| (c << 3) | (c >> 2);
                        [e((v >> 10) & 0x1f), e((v >> 5) & 0x1f), e(v & 0x1f), if v & 0x8000 != 0 { 255 } else { 0 }]
                    } else {
                        let v = match (dmode >> 4) & 0xf {
                            1 => (v << 8) | 0xff,
                            3 => v.swap_bytes(),
                            _ => v,
                        };
                        [v >> 24, (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff]
                    }
                };
                let half_off = if dmode & 1 == 0 { 2 } else { 0 };
                // Depth: Z0 and its gradients are 64-bit 25.12 (high word
                // first), from the same start point as the colours, truncated
                // and clamped to 24 bits. The stencil/depth buffer is 32-bit
                // (stencil 31:24, Z 23:0) at the same x, y through TLB C: GL
                // contexts load C with their depth buffer, the MTE clears it
                // to 0xffffff, and the X server uses C for the overlay.
                let depth = dm & draw::DEPTH_TEST != 0;
                let zmask = dm & draw::DEPTH_MASK != 0;
                let r64 = |off: u32| (((g(off) as u64) << 32) | g(off + 4) as u64) as i64 as f64 / 4096.0;
                let (z0, dzdx, dzdy) = (r64(DEPTH_Z0), r64(DEPTH_DZDX), r64(DEPTH_DZDY));
                let zfunc = (g(DEPTH_FUNC) >> 25) & 7;
                let zbuf = Buf { tlb: TLB_A + 0x400, bd: 4 };
                // The stencil buffer is the depth buffer's top byte.
                let stencil = dm & draw::STENCIL_TEST != 0;
                let smode = g(STENCIL_MODE);
                let (sref, svmask, sfunc) = (smode >> 24, (smode >> 16) & 0xff, (smode >> 12) & 7);
                let swmask = g(STENCIL_MASK) & 0xff;
                if stencil {
                    for op in [(smode >> 8) & 15, (smode >> 4) & 15, smode & 15] {
                        if op != 0 && op != 3 {
                            self.note(format!("stencil op {op} (taken as KEEP)"));
                        }
                    }
                    if matches!(sfunc, 1 | 3 | 4 | 6) {
                        self.note(format!("stencil function {sfunc} (order unconfirmed)"));
                    }
                }
                // Op 0 keeps, 3 replaces with the reference; no other has been
                // seen, so they keep too (and say so above).
                let sop = |op: u32, s: u32| -> u32 {
                    let v = if op == 3 { sref & 0xff } else { s };
                    (s & !swmask) | (v & swmask)
                };
                // Signed area, so the edge functions agree in sign inside.
                let area = (v[1].0 - v[0].0) * (v[2].1 - v[0].1) - (v[1].1 - v[0].1) * (v[2].0 - v[0].0);
                if area == 0.0 {
                    return;
                }
                let s = area.signum();
                let edge = |a: (f64, f64), b: (f64, f64), px: f64, py: f64| s * ((b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0));
                // Top-left rule: a pixel centre exactly on an edge belongs to
                // the triangle only for a top or left edge.
                let top_left = |a: (f64, f64), b: (f64, f64)| {
                    let (dx, dy) = (s * (b.0 - a.0), s * (b.1 - a.1));
                    (dy == 0.0 && dx < 0.0) || dy > 0.0
                };
                let xmin = v.iter().map(|p| p.0).fold(f64::MAX, f64::min).floor().max(0.0) as i32;
                let xmax = v.iter().map(|p| p.0).fold(f64::MIN, f64::max).ceil().min(4095.0) as i32;
                let ymin = v.iter().map(|p| p.1).fold(f64::MAX, f64::min).floor().max(0.0) as i32;
                let ymax = v.iter().map(|p| p.1).fold(f64::MIN, f64::max).ceil().min(4095.0) as i32;
                let edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])];
                let (mut crow, mut zrow) = (RowCursor::new(dst), RowCursor::new(zbuf));
                for y in ymin..=ymax {
                    // Narrow the row to where every edge function can be
                    // non-negative (each is linear in x), a pixel wider each
                    // side for rounding, and test exactly only there:
                    // scanning the whole bounding box tested twice the pixels
                    // a triangle covers.
                    let py = y as f64;
                    let (mut lo, mut hi) = (xmin as f64, xmax as f64);
                    for &(a, b) in &edges {
                        let c1 = -s * (b.1 - a.1);
                        let c0 = s * ((b.0 - a.0) * (py - a.1) + (b.1 - a.1) * a.0);
                        if c1 > 0.0 {
                            lo = lo.max(-c0 / c1);
                        } else if c1 < 0.0 {
                            hi = hi.min(-c0 / c1);
                        } else if c0 < 0.0 {
                            hi = lo - 4.0;
                        }
                    }
                    if hi + 2.0 < lo {
                        continue;
                    }
                    let (xs, xe) = ((lo.floor() as i32 - 1).max(xmin), (hi.ceil() as i32 + 1).min(xmax));
                    // A triangle covers one run of each row: find its ends
                    // with the exact test and fill between them, rather than
                    // testing three edges at every pixel.
                    let covered = |x: i32| {
                        let px = x as f64;
                        edges.iter().all(|&(a, b)| {
                            let e = edge(a, b, px, py);
                            e > 0.0 || (e == 0.0 && top_left(a, b))
                        })
                    };
                    let (mut xs, mut xe) = (xs, xe);
                    while xs <= xe && !covered(xs) {
                        xs += 1;
                    }
                    while xe >= xs && !covered(xe) {
                        xe -= 1;
                    }
                    for x in xs..=xe {
                        let px = x as f64;
                        if clip.ok(x, y) {
                            // GL order: the alpha test before the depth test,
                            // so a rejected fragment leaves depth alone.
                            let src = if atest || blend || dither { Some(rgba_at(mem, px, py)) } else { None };
                            if atest && !Self::alpha_pass(afunc, aref, src.unwrap()[3]) {
                                continue;
                            }
                            // GL order: the stencil test, then depth; the
                            // stencil op is chosen by which of them failed.
                            if depth || stencil {
                                let zs = (z0 + dzdx * (px - rx) + dzdy * (py - ry)).clamp(0.0, 16_777_215.0) as u32;
                                if let Some(zo) = zrow.at(&r, mem, x as u32, y as u32) {
                                    let old = Self::get_px(mem, zo, 4);
                                    let (sd, zd) = (old >> 24, old & 0xff_ffff);
                                    let cmp = |f: u32, a: u32, b: u32| match f {
                                        0 => false,
                                        1 => a < b,
                                        2 => a == b,
                                        3 => a <= b,
                                        4 => a > b,
                                        5 => a != b,
                                        6 => a >= b,
                                        _ => true,
                                    };
                                    if stencil && !cmp(sfunc, sd & svmask, sref & svmask) {
                                        let s = sop((smode >> 8) & 15, sd);
                                        if s != sd {
                                            mem.put_be(zo, 4, s << 24 | zd);
                                        }
                                        continue;
                                    }
                                    let zpass = !depth || cmp(zfunc, zs, zd);
                                    let s = if stencil { sop(if zpass { smode & 15 } else { (smode >> 4) & 15 }, sd) } else { sd };
                                    let z = if zpass && depth && zmask { zs } else { zd };
                                    if s != sd || z != zd {
                                        mem.put_be(zo, 4, s << 24 | z);
                                    }
                                    if !zpass {
                                        continue;
                                    }
                                }
                            }
                            if let Some(o) = crow.at(&r, mem, x as u32, y as u32) {
                                let colour = match src {
                                    Some(c) if blend => {
                                        let (at, n) = if half { (o + half_off, 2) } else { (o, 4) };
                                        let [r8, g8, b8, a8] = Self::blend(bfunc, bconst, c, unpack(Self::get_px(mem, at, n)));
                                        if dither { pack_at(x, y, r8, g8, b8, a8) } else { pack(r8, g8, b8, a8) }
                                    }
                                    Some([r8, g8, b8, a8]) if dither => pack_at(x, y, r8, g8, b8, a8),
                                    Some([r8, g8, b8, a8]) => pack(r8, g8, b8, a8),
                                    None => colour_at(mem, px, py),
                                };
                                if half {
                                    Self::put(&r, mem, o + half_off, 2, colour);
                                } else {
                                    Self::put(&r, mem, o, dst.bd, colour);
                                }
                            }
                        }
                    }
                }
            }
            _ => self.note("primitive type 3?".into()),
        }
    }

    /// OpenGL blending of source `s` over destination `d` (RGBA, 0-255)
    /// by the Blend register: op in 11:8 (ADD, MIN, MAX, SUBTRACT,
    /// REVERSE_SUBTRACT), source factor in 7:4 and destination factor in
    /// 3:0, numbered as the spec lists them; `konst` is the constant colour,
    /// RGBA with R high. MIN and MAX ignore the factors, as in OpenGL.
    fn blend(func: u32, konst: u32, s: [u32; 4], d: [u32; 4]) -> [u32; 4] {
        let f = |v: u32| v as f64 / 255.0;
        let (s, d) = (s.map(f), d.map(f));
        let k = [konst >> 24, (konst >> 16) & 0xff, (konst >> 8) & 0xff, konst & 0xff].map(f);
        // Factor `code` for channel `i`; `other` is the colour the
        // COLOR factors name (the destination's for the source factor, the
        // source's for the destination factor).
        let factor = |code: u32, other: &[f64; 4], i: usize| -> f64 {
            match code {
                0 => 0.0,
                1 => 1.0,
                2 => other[i],
                3 => 1.0 - other[i],
                4 => s[3],
                5 => 1.0 - s[3],
                6 => d[3],
                7 => 1.0 - d[3],
                8 => k[i],
                9 => 1.0 - k[i],
                10 => k[3],
                11 => 1.0 - k[3],
                12 if i < 3 => s[3].min(1.0 - d[3]),
                12 => 1.0,
                _ => 1.0,
            }
        };
        let (op, sf, df) = ((func >> 8) & 0xf, (func >> 4) & 0xf, func & 0xf);
        [0, 1, 2, 3].map(|i| {
            let (a, b) = (s[i] * factor(sf, &d, i), d[i] * factor(df, &s, i));
            let v = match op {
                1 => s[i].min(d[i]),
                2 => s[i].max(d[i]),
                3 => a - b,
                4 => b - a,
                _ => a + b,
            };
            (v.clamp(0.0, 1.0) * 255.0).round() as u32
        })
    }

    /// An 8-bit channel reduced to 5 bits with a 4 x 4 ordered dither at
    /// pixel (x, y): the threshold spreads the 8 levels each 5-bit step
    /// drops over the cell. The spec does not give CRIME's matrix; this is
    /// the standard Bayer one.
    fn dither5(c: u32, x: i32, y: i32) -> u32 {
        const BAYER: [[u32; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
        ((c + BAYER[(y & 3) as usize][(x & 3) as usize] / 2) >> 3).min(31)
    }

    /// The alpha test: whether alpha `a` passes function `func` against `aref`.
    fn alpha_pass(func: u32, aref: u32, a: u32) -> bool {
        match func {
            0 => false,
            1 => a < aref,
            2 => a == aref,
            3 => a <= aref,
            4 => a > aref,
            5 => a != aref,
            6 => a >= aref,
            _ => true,
        }
    }

    /// A GL line: float ends at 0x2080/0x2088 (as the triangle's, biased by
    /// 2^17), coloured by the shade registers. With antialiasing on (draw
    /// mode bit 12) each step along the major axis covers the two pixels
    /// straddling the line in proportion, blended by coverage times alpha.
    /// 4Dwm and the desktop outline their vector icons this way, black at
    /// full alpha into the 32-bit window.
    fn gl_line(r: &[u32], mem: &mut RamWindow, dst: Buf, ox: i32, oy: i32, skip_last: bool, point: bool) {
        let clip = Clip::new(r);
        let g = |off: u32| r[(off / 4) as usize];
        let dm = g(DRAWMODE);
        let fv = |off: u32| -> (f64, f64) {
            let f = gl_coord;
            (f(g(off)) + ox as f64, f(g(off + 4)) + oy as f64)
        };
        let (a, b) = (fv(TRI_V0), if point { fv(TRI_V0) } else { fv(TRI_V0 + 8) });
        let ch = |off: u32| (g(off) >> 12) & 0xff;
        let (cr, cg, cb, ca) = if dm & draw::SHADE != 0 {
            (ch(SHADE_R), ch(SHADE_R + 4), ch(SHADE_R + 8), ch(SHADE_R + 12))
        } else {
            let fg = g(FG);
            (fg >> 24, (fg >> 16) & 0xff, (fg >> 8) & 0xff, fg & 0xff)
        };
        let fmt = (g(BUFMODE_DST) >> 4) & 0xf;
        // Pack and unpack the destination's pixel type (as for triangles).
        let pack = |r8: u32, g8: u32, b8: u32, a8: u32| -> u32 {
            let rgba = (r8 << 24) | (g8 << 16) | (b8 << 8) | a8;
            match fmt {
                0 => r8,
                1 => rgba >> 8,
                3 => rgba.swap_bytes(),
                _ => rgba,
            }
        };
        let unpack = |v: u32| -> (u32, u32, u32) {
            let v = match fmt {
                1 => v << 8,
                3 => v.swap_bytes(),
                _ => v,
            };
            (v >> 24, (v >> 16) & 0xff, (v >> 8) & 0xff)
        };
        // Double-pix, as for triangles: a double-buffered GL window's 16-bit
        // A1R5G5B5 pixel in one half of each word (the icon catalog's).
        let dmode = g(BUFMODE_DST);
        let half = dmode & 2 != 0 && (dmode >> 2) & 3 == 1 && dst.bd == 4;
        let half_off = if dmode & 1 == 0 { 2 } else { 0 };
        let blend = dm & draw::BLEND != 0 && dst.bd == 4 && (fmt != 0 || half);
        let mut plot = |x: i32, y: i32, cov: f64| {
            if x < 0 || y < 0 || cov <= 0.0 || !clip.ok(x, y) {
                return;
            }
            let Some(o) = Self::locate(r, mem, dst, x as u32, y as u32) else { return };
            // Coverage scales the fragment's alpha; with blending off a
            // fragment is drawn whole once at least half covered, whatever
            // its colour's alpha: IRIS GL's RGB colours carry alpha 0, and
            // gating on the product drew none of amesh's lines.
            if !blend && cov < 0.5 {
                return;
            }
            let alpha = cov.min(1.0) * ca as f64 / 255.0;
            let sa = (alpha * 255.0).round() as u32;
            let (func, konst) = (g(BLEND_FUNC), g(BLEND_CONST));
            if half {
                let o = o + half_off;
                let d = Self::get_px(mem, o, 2);
                let e = |v: u32| (v << 3) | (v >> 2);
                let dc = [e((d >> 10) & 0x1f), e((d >> 5) & 0x1f), e(d & 0x1f), if d & 0x8000 != 0 { 255 } else { 0 }];
                let [r8, g8, b8, _] = if blend { Self::blend(func, konst, [cr, cg, cb, sa], dc) } else { [cr, cg, cb, ca] };
                let v = (d & 0x8000) | ((r8 >> 3) << 10) | ((g8 >> 3) << 5) | (b8 >> 3);
                Self::put(r, mem, o, 2, v);
            } else {
                // The destination is read only to blend, which needs 32-bit
                // pixels: an 8-bit one read as a word is off its alignment
                // (flight's shaded points into its colour-index window
                // stopped the emulator).
                let [r8, g8, b8, _] = if blend {
                    let (dr, dg, db) = unpack(Self::get_px(mem, o, 4));
                    Self::blend(func, konst, [cr, cg, cb, sa], [dr, dg, db, 255])
                } else {
                    [cr, cg, cb, ca]
                };
                Self::put(r, mem, o, dst.bd, pack(r8, g8, b8, ca));
            }
        };
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let steep = dy.abs() > dx.abs();
        // Walk the major axis from a's end to b's, one pixel centre a step.
        let (ma, mb, na, nb) = if steep { (a.1, b.1, a.0, b.0) } else { (a.0, b.0, a.1, b.1) };
        let (s0, s1) = (ma.round() as i32, mb.round() as i32);
        let step = if s1 >= s0 { 1 } else { -1 };
        let slope = if mb != ma { (nb - na) / (mb - ma) } else { 0.0 };
        let aa = dm & draw::ANTIALIAS_LINE != 0;
        // Line stipple, as for X lines: one pattern bit per step from the
        // mode's start, repeating every max + 1. perfly draws its panel text
        // a glyph row at a time this way.
        let stippled = dm & draw::LINE_STIPPLE != 0;
        let smode = g(STIPPLE_MODE);
        let (sstart, span) = ((smode >> 24) & 0x1f, ((smode >> 16) & 0x1f) + 1);
        let pattern = g(STIPPLE_PATTERN);
        let mut m = s0;
        loop {
            if m == s1 && skip_last && s1 != s0 {
                break;
            }
            let k = (m - s0).unsigned_abs();
            if stippled && (pattern >> (31 - (sstart + k) % span)) & 1 == 0 {
                if m == s1 || (m - s0).abs() > 8192 {
                    break;
                }
                m += step;
                continue;
            }
            let n = na + slope * (m as f64 - ma);
            let mut put2 = |mm: i32, nn: i32, cov: f64| if steep { plot(nn, mm, cov) } else { plot(mm, nn, cov) };
            if aa {
                let lo = n.floor();
                let f = n - lo;
                put2(m, lo as i32, 1.0 - f);
                put2(m, lo as i32 + 1, f);
            } else {
                put2(m, n.round() as i32, 1.0);
            }
            if m == s1 || (m - s0).abs() > 8192 {
                break;
            }
            m += step;
        }
    }

    /// Run the memory transfer engine's current transfer (NetBSD's clears
    /// and scrolls).
    /// RAM offset of byte `addr` of linear buffer `which` (0 = A, 1 = B).
    ///
    /// A linear buffer is 32 pages of 4 KB: entry `(addr >> 12) & 31`, kept
    /// in pairs, the even page in a word's high half and the odd one in its
    /// low half. An entry is bit 31 valid and the page number below it, so
    /// the byte is at `page << 12 | addr & 0xfff`. IRIX maps a buffer onto
    /// the pages it is about to clear or copy, with the page number equal to
    /// the address it then hands the MTE.
    fn linear(r: &[u32], mem: &RamWindow, which: u32, addr: u32) -> Option<usize> {
        let idx = (addr >> 12) & 31;
        let word = TLB_LINEAR_A + 0x80 * which + 8 * (idx / 2);
        let e = r[(word / 4 + (idx & 1)) as usize];
        if e & (1 << 31) == 0 {
            return None;
        }
        mem.dma_offset(((e & 0x000f_ffff) << 12) | (addr & 0xfff))
    }

    /// An MTE clear or copy whose buffers are linear: DST0..=DST1 (and SRC0
    /// on) are byte addresses, not x/y. The kernel zeroes and copies pages
    /// this way; doing nothing left them holding whatever was there before.
    fn mte_linear(&self, r: &[u32], copy: bool, sbuf: u32, dbuf: u32) {
        let g = |off: u32| r[(off / 4) as usize];
        let (d0, d1) = (g(MTE_DST0), g(MTE_DST1));
        if d1 < d0 || d1 - d0 >= 32 * 4096 {
            self.note(format!("mte linear span {d0:#x}..{d1:#x}"));
            return;
        }
        let s0 = g(MTE_SRC0);
        let fg = g(MTE_FG);
        let mask = g(MTE_BYTEMASK);
        let mut ram = Draw::new(self.win.lock().unwrap());
        let mem: &mut RamWindow = &mut ram;
        // A 4 KB page at a time: each is contiguous behind its TLB entry.
        let mut a = d0;
        let mut src = Vec::new();
        while a <= d1 {
            let end = d1.min(a | 0xfff);
            let n = (end - a + 1) as usize;
            let Some(o) = Self::linear(r, mem, dbuf - 4, a) else {
                a = end + 1;
                continue;
            };
            let wanted = |k: usize| (mask >> (31 - ((a as usize + k) & 31))) & 1 != 0;
            if copy {
                // The source byte for each, None where its page is unmapped.
                src.clear();
                let mut page = (u32::MAX, None);
                for k in 0..n {
                    let sa = s0.wrapping_add(a - d0 + k as u32);
                    if page.0 != sa >> 12 {
                        page = (sa >> 12, Self::linear(r, mem, sbuf - 4, sa & !0xfff));
                    }
                    src.push(page.1.map(|so| mem[so + (sa & 0xfff) as usize]));
                }
                mem.write_span(o, n, |k| if wanted(k) { src[k] } else { None });
            } else {
                mem.write_span(o, n, |k| wanted(k).then(|| (fg >> (24 - 8 * ((a as usize + k) & 3))) as u8));
            }
            a = end + 1;
        }
    }

    fn mte(&self) {
        self.drawn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The registers stay locked for the primitive: copying all 24 KB of
        // them per primitive was a tenth of the engine's time under X.
        let r = self.regs.lock().unwrap();
        let g = |off: u32| r[(off / 4) as usize];
        let mode = g(MTE_MODE);
        let copy = mode & (1 << 11) != 0;
        let dbuf = (mode >> 2) & 7;
        let sbuf = (mode >> 5) & 7;
        let linear = |b: u32| b == 4 || b == 5;
        if linear(dbuf) && (!copy || linear(sbuf)) {
            self.mte_linear(&r, copy, sbuf, dbuf);
            return;
        }
        if dbuf > 2 || (copy && sbuf > 2) {
            // dst 6 is the clip-ID buffer: IRIX clears it once at boot, and
            // without its layout (see pixel_pipe's clip-ID note) the clear
            // is skipped rather than aimed at the wrong tiles.
            self.note(format!("mte buffers src {sbuf} dst {dbuf}"));
            return;
        }
        // MTE x is in bytes: one byte per "pixel" through the tile TLB.
        let (dst, src) = (Buf { tlb: TLB_A + 0x200 * dbuf, bd: 1 }, Buf { tlb: TLB_A + 0x200 * sbuf, bd: 1 });
        let (dx0, dy0) = Self::xy(g(MTE_DST0));
        let (dx1, dy1) = Self::xy(g(MTE_DST1));
        let (sx0, sy0) = Self::xy(g(MTE_SRC0));
        let dstep = g(MTE_DST_STEP) as i32;
        let sstep = g(MTE_SRC_STEP) as i32;
        let (dstep, sstep) = (if dstep < 0 { -1 } else { 1 }, if sstep < 0 { -1 } else { 1 });
        let rows = (dy1 - dy0).unsigned_abs() + 1;
        let (xa, xb) = (dx0.min(dx1).max(0), dx0.max(dx1).max(0));
        if rows > 4096 || xb - xa > 8192 {
            self.note(format!("mte span {xa}..{xb} x {rows}"));
            return;
        }
        let fg = g(MTE_FG);
        let bytemask = g(MTE_BYTEMASK);
        let stippled = mode & (1 << 10) != 0;
        let stipple = g(MTE_STIPPLE);
        let bpp = 1u32 << ((mode >> 8) & 3).min(2);
        let mut ram = Draw::new(self.win.lock().unwrap());
        let mem: &mut RamWindow = &mut ram;
        let mut row = Vec::new();
        for i in 0..rows as i32 {
            let dy = dy0 + i * dstep;
            if dy < 0 {
                continue;
            }
            if copy {
                // The source row first, a tile's worth at a time.
                let sy = sy0 + i * sstep;
                row.clear();
                let mut x = xa;
                while x <= xb {
                    let sx = sx0 + (x - xa);
                    let n = if sx < 0 { 1 } else { (xb - x + 1).min(512 - (sx & 511)) };
                    let base = if sx < 0 || sy < 0 { None } else { Self::locate(&r, mem, src, sx as u32, sy as u32) };
                    for k in 0..n as usize {
                        row.push(base.map(|o| mem[o + k]).unwrap_or(0));
                    }
                    x += n;
                }
            }
            // The destination a tile's 512-byte row at a time.
            let mut x = xa;
            while x <= xb {
                let n = (xb - x + 1).min(512 - (x & 511));
                let Some(o) = Self::locate(&r, mem, dst, x as u32, dy as u32) else {
                    x += n;
                    continue;
                };
                let x0 = x;
                if copy {
                    let from = (x0 - xa) as usize;
                    mem.write_span(o, n as usize, |k| Some(row[from + k]));
                } else if !stippled && bytemask == (bytemask >> 28).wrapping_mul(0x1111_1111) {
                    // No stipple and the same byte mask in every word (page
                    // clears, depth clears, the X server's low-byte fills):
                    // whole words through a mask, ragged ends a byte at a
                    // time. The memory offset and x agree mod 4.
                    let nib = bytemask >> 28;
                    let wm = (0..4).fold(0u32, |m, j| if nib & (1 << (3 - j)) != 0 { m | (0xff << (8 * (3 - j))) } else { m });
                    let (start, end) = (o, o + n as usize);
                    let (a0, a1) = ((start + 3) & !3, end & !3);
                    let byte = |at: usize| (nib & (1 << (3 - (at & 3))) != 0).then(|| (fg >> (24 - 8 * (at & 3))) as u8);
                    if a0 >= a1 {
                        mem.write_span(start, n as usize, |k| byte(start + k));
                    } else {
                        mem.write_span(start, a0 - start, |k| byte(start + k));
                        mem.fill_words_masked(a0, (a1 - a0) / 4, fg, wm);
                        mem.write_span(a1, end - a1, |k| byte(a1 + k));
                    }
                } else {
                    mem.write_span(o, n as usize, |k| {
                        let x = x0 + k as i32;
                        // A stippled clear writes only the pixels whose mask
                        // bit is set: one bit per pixel, most significant
                        // first, counted from the 32-byte block holding the
                        // row's first byte (eight pixels at 32 bpp, 32 at 8
                        // bpp) and wrapping after bit 31. IRIX's X server
                        // draws text a row and a mask at a time: from the row
                        // start, the login screen's 8-bit labels come out
                        // shifted; without the wrap, the 32-pixel spans of a
                        // 32-bit terminal lose their last seven pixels, most
                        // of every fourth character.
                        if stippled {
                            let b = ((x - (xa & !31)) as u32 / bpp) % 32;
                            if (stipple >> (31 - b)) & 1 == 0 {
                                return None;
                            }
                        }
                        // byteMask: one bit per byte of a 32-byte block, most
                        // significant first. The kernel's page clears pass all
                        // ones; IRIX's X server passes 0x11111111 to clear
                        // only the low byte of each 32-bit pixel, and writing
                        // the whole pixel instead blacked out what it had
                        // drawn.
                        if (bytemask >> (31 - (x as u32 & 31))) & 1 == 0 {
                            return None;
                        }
                        Some((fg >> (24 - 8 * ((o + k) as u32 & 3))) as u8)
                    });
                }
                x += n;
            }
        }
    }
}

impl BusDevice for CrimeRe {
    fn read32(&self, addr: u64) -> BusRead32 {
        self.read_reg((addr as u32) & 0x7ffc).map_or(BusRead32::busy(), BusRead32::ok)
    }
    fn read64(&self, addr: u64) -> BusRead64 {
        let a = (addr as u32) & 0x7ff8;
        if a == STATUS {
            return self.read_reg(STATUS).map_or(BusRead64::busy(), |v| BusRead64::ok(v as u64));
        }
        match (self.read_reg(a), self.read_reg(a + 4)) {
            (Some(hi), Some(lo)) => BusRead64::ok((hi as u64) << 32 | lo as u64),
            _ => BusRead64::busy(),
        }
    }
    fn write32(&self, addr: u64, val: u32) -> u32 {
        self.cpu_write((addr as u32) & 0x7ffc, val as u64, 32)
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        self.cpu_write((addr as u32) & 0x7ff8, val, 64)
    }
}

impl CrimeRe {
    /// Both halves land before a start acts, so a 64-bit write to a start
    /// address (vertices 0 and 1 at once) draws with both.
    fn apply64(&self, a: u32, val: u64) {
        let page = a & 0xf000;
        let start = (page == 0x2000 || page == 0x3000) && a & 0x800 != 0;
        let base = if start { a & !0x800 } else { a };
        {
            let mut r = self.regs.lock().unwrap();
            if let Some(s) = r.get_mut((base / 4) as usize) {
                *s = (val >> 32) as u32;
            }
            if let Some(s) = r.get_mut((base / 4 + 1) as usize) {
                *s = val as u32;
            }
        }
        if start {
            if page == 0x2000 {
                self.pixel_pipe();
            } else {
                self.mte();
            }
        }
    }
}

/// The registers (TLBs, colour maps and cursor among them), the beam's
/// timing reference, and the shadow writes still waiting for blanking. The
/// timing is in guest time, which the UST restores.
impl Saveable for Gbe {
    fn save_state(&self) -> toml::Value {
        let s = self.s.lock().unwrap();
        let mut t = toml::map::Map::new();
        t.insert("regs".into(), u32_slice_to_toml(&s.r));
        t.insert("t0".into(), hex_u64(s.t0));
        t.insert("reads".into(), hex_u64(s.reads));
        t.insert("seen_dots".into(), hex_u64(s.seen_dots));
        let pending = s.pending.iter().map(|&(off, val, frame)| {
            toml::Value::Array(vec![
                toml::Value::Integer(off as i64),
                toml::Value::Integer(val as i64),
                hex_u64(frame),
            ])
        }).collect();
        t.insert("pending".into(), toml::Value::Array(pending));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut s = self.s.lock().unwrap();
        if let Some(r) = get_field(v, "regs") {
            load_u32_slice(r, &mut s.r);
        }
        let u = |k: &str| get_field(v, k).and_then(toml_u64).unwrap_or(0);
        s.t0 = u("t0");
        s.reads = u("reads");
        s.seen_dots = u("seen_dots");
        s.pending.clear();
        if let Some(toml::Value::Array(a)) = get_field(v, "pending") {
            for e in a {
                if let Some([off, val, frame]) = e.as_array().map(|x| x.as_slice()) {
                    s.pending.push((
                        toml_u32(off).unwrap_or(0),
                        toml_u32(val).unwrap_or(0),
                        toml_u64(frame).unwrap_or(0),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The engine's registers, once what is queued has been drawn: the interface
/// buffer is empty in every snapshot, so a load only sets registers.
impl Saveable for CrimeRe {
    fn save_state(&self) -> toml::Value {
        if !self.drain(std::time::Duration::from_secs(2)) {
            eprintln!("crime-re: saving with {} writes still queued; they are lost", self.ib.len());
        }
        let mut t = toml::map::Map::new();
        t.insert("regs".into(), u32_slice_to_toml(&self.regs.lock().unwrap()));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        if !self.drain(std::time::Duration::from_secs(2)) {
            return Err("crime-re: the engine is still busy".into());
        }
        if let Some(r) = get_field(v, "regs") {
            load_u32_slice(r, &mut self.regs.lock().unwrap());
        }
        Ok(())
    }
}

impl crate::traits::Device for Gbe {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("gbe".to_string(), "GBE display engine: gbe status".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write;
        if cmd != "gbe" {
            return Err("Command not found".to_string());
        }
        match args.first().copied().unwrap_or("status") {
            "status" => {
                writeln!(w, "GBE: {}", self.summary()).ok();
                Ok(())
            }
            _ => Err("Usage: gbe status".to_string()),
        }
    }
}

impl crate::traits::Device for CrimeRe {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("re".to_string(), "CRIME rendering engine: re status | re notes".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write;
        use std::sync::atomic::Ordering::Relaxed;
        if cmd != "re" {
            return Err("Command not found".to_string());
        }
        match args.first().copied().unwrap_or("status") {
            "status" => {
                writeln!(w, "CRIME RE: {} primitives drawn, {}, {} of {} interface buffer entries; trace {} of {} writes",
                    self.drawn.load(Relaxed),
                    if self.idle() { "idle" } else { "busy" },
                    self.ib.len(), IB_DEPTH - 1,
                    self.trace.lock().unwrap().len(), self.trace_cap).ok();
            }
            "notes" => {
                for (what, n) in self.unhandled() {
                    writeln!(w, "  {n:8}  {what}").ok();
                }
            }
            _ => return Err("Usage: re status | notes".to_string()),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An engine on its own thread, for tests: `write32`/`write64` retry
    /// while the interface buffer is full and then wait for the engine, so
    /// a test sees each write carried out; `put32`/`put64` only retry.
    struct TestRe(Arc<CrimeRe>);

    impl TestRe {
        fn new(ram: Arc<O2Memory>) -> Self {
            let re = Arc::new(CrimeRe::new(ram));
            re.start(re.clone(), |r| r);
            TestRe(re)
        }
        fn put32(&self, addr: u32, val: u32) {
            while self.0.write32(addr as u64, val) == BUS_BUSY {
                std::hint::spin_loop();
            }
        }
        fn put64(&self, addr: u32, val: u64) {
            while self.0.write64(addr as u64, val) == BUS_BUSY {
                std::hint::spin_loop();
            }
        }
        fn write32(&self, addr: u32, val: u32) {
            self.put32(addr, val);
            self.0.wait_idle();
        }
        fn write64(&self, addr: u32, val: u64) {
            self.put64(addr, val);
            self.0.wait_idle();
        }
    }

    impl std::ops::Deref for TestRe {
        type Target = CrimeRe;
        fn deref(&self) -> &CrimeRe {
            &self.0
        }
    }

    fn gbe() -> Gbe {
        let t = Arc::new(std::sync::atomic::AtomicU64::new(0));
        Gbe::new(Arc::new(move || t.load(std::sync::atomic::Ordering::Relaxed)))
    }

    #[test]
    fn gbe_save_load_round_trip() {
        let src = gbe();
        src.write32(DOTCLOCK as u64, DOTCLOCK_RUN | 0x0123);
        src.write32((CMAP + 4 * 7) as u64, 0x00ff_8000);
        {
            let mut s = src.s.lock().unwrap();
            s.t0 = 1_000_000;
            s.reads = 42;
            s.seen_dots = 9999;
            s.pending.push((FRM_CONTROL, 0x8000_0001, 17));
        }
        let v1 = src.save_state();

        let dst = gbe();
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "Gbe save_state mismatch after load_state round-trip");
    }

    #[test]
    fn re_save_load_round_trip() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let src = TestRe::new(ram.clone());
        src.write32(0x1500_0000 + FG, 0x1234_5678);
        src.write32(0x1500_0000 + BG, 0x9abc_def0);
        src.write64(0x1500_0000 + TLB_A, 0x0123_4567_89ab_cdef);
        let v1 = src.save_state();

        let dst = CrimeRe::new(ram);
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "CrimeRe save_state mismatch after load_state round-trip");
    }

    #[test]
    fn the_polls_the_prom_and_the_bsds_run_terminate() {
        let g = gbe();
        assert_eq!(g.read32(GBE_CTRLSTAT_ADDR as u64).data & 0xf, 1, "chip id 1");
        assert_ne!(g.read32(GBE_CTRLSTAT_ADDR as u64).data & (1 << 12), 0, "board revision strap");
        assert_eq!(g.read32((0x1600_0000 + CM_FIFO) as u64).data, 0);
        g.write32((0x1600_0000 + FRM_CONTROL_SHADOW) as u64, 0x00d6_0001);
        assert_eq!(g.read32((0x1600_0000 + FRM_CONTROL) as u64).data, 0x00d6_0001, "live follows shadow exactly");
        g.write32((0x1600_0000 + DOTCLOCK) as u64, 0x0010_020f);
        assert_ne!(g.read32((0x1600_0000 + DOTCLOCK) as u64).data & DOTCLOCK_RUN, 0);
        assert_eq!(g.read32((0x1600_0000 + VT_XY) as u64).data, VT_FREEZE, "frozen at reset");
        g.write32((0x1600_0000 + VT_XYMAX) as u64, 0x0042_9690);
        g.write32((0x1600_0000 + VT_XY) as u64, 0);
        // The PROM's wait for blanking: x in 1280..=1480 or y in 1024..=1044.
        let mut seen = false;
        for _ in 0..100_000 {
            let xy = g.read32((0x1600_0000 + VT_XY) as u64).data;
            let (x, y) = (xy & 0xfff, (xy >> 12) & 0xfff);
            if (1280..=1480).contains(&x) || (1024..=1044).contains(&y) {
                seen = true;
                break;
            }
        }
        assert!(seen, "the beam reaches blanking with guest time standing still");
    }

    const GBE_CTRLSTAT_ADDR: u32 = 0x1600_0000;

    /// How fast the engine replays a recorded trace, for comparing changes
    /// without a guest: RE_SPEED=trace [RE_SPEED_PASSES=n]. The trace is
    /// parsed first; only the register writes are timed.
    #[test]
    #[ignore]
    fn replay_speed() {
        let Ok(path) = std::env::var("RE_SPEED") else { return };
        let passes: u32 = std::env::var("RE_SPEED_PASSES").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        let writes: Vec<(u32, u64, bool)> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| {
                let mut f = line.split_whitespace();
                let (w, off, val) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
                (u32::from_str_radix(off, 16).unwrap(), u64::from_str_radix(val, 16).unwrap(), w == "w64")
            })
            .collect();
        let m = O2Memory::from_kits(&[128, 128]).unwrap();
        m.set_ctrl(0, 0x100);
        m.set_ctrl(2, 0x100 | 4);
        let ram = Arc::new(m);
        let re = TestRe::new(ram.clone());
        let mut best = f64::MAX;
        for _ in 0..passes {
            let t = std::time::Instant::now();
            for &(off, val, w64) in &writes {
                if w64 {
                    re.put64(0x1500_0000 + off, val);
                } else {
                    re.put32(0x1500_0000 + off, val as u32);
                }
            }
            re.wait_idle();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let drawn = re.drawn.load(std::sync::atomic::Ordering::Relaxed) / passes as u64;
        // A hash of all of memory, to check a change draws exactly the same.
        let hash = {
            use std::hash::{Hash, Hasher};
            let m = &*ram;
            let mut h = std::collections::hash_map::DefaultHasher::new();
            m.raw(0, m.len()).hash(&mut h);
            h.finish()
        };
        eprintln!("replay_speed: {} writes, {drawn} primitives, best of {passes}: {:.3} s ({:.0} ns/write), memory {hash:016x}", writes.len(), best, best * 1e9 / writes.len() as f64);
    }

    /// Replay a recorded register trace (IRIS_IP32_RE_TRACE, saved with the
    /// control port's `retrace`) into a fresh RE and save what TLB A shows
    /// at 32 bpp as a PNG: RE_REPLAY=trace RE_REPLAY_PNG=out.png. CPU writes
    /// to the frame buffer are not in the trace, so only drawing shows.
    #[test]
    #[ignore]
    fn replay_a_recorded_trace() {
        let Ok(path) = std::env::var("RE_REPLAY") else { return };
        // RE_GOURAUD="frac,vertex": gradient fraction bits and reference vertex.
        if let Some((f, v)) = std::env::var("RE_GOURAUD").ok().as_deref().and_then(|s| s.split_once(',')) {
            GOURAUD.store(f.parse::<u32>().unwrap() << 8 | v.parse::<u32>().unwrap(), std::sync::atomic::Ordering::Relaxed);
        }
        let out = std::env::var("RE_REPLAY_PNG").unwrap_or("replay.png".into());
        // 256 MB as two 128 MB banks, placed the way the PROM places them.
        let m = O2Memory::from_kits(&[128, 128]).unwrap();
        m.set_ctrl(0, 0x100);
        m.set_ctrl(2, 0x100 | 4);
        // RE_REPLAY_PRELOAD="0849:tex-0849.bin,...": 64 KB tiles (by tile
        // number) to fill first, for what the CPU wrote that a register
        // trace cannot hold (textures).
        // The kernel's MTE page clears in the trace zero pages the CPU later
        // refilled, so the tiles are put back every couple of thousand writes.
        let preload: Vec<(usize, Vec<u8>)> = std::env::var("RE_REPLAY_PRELOAD")
            .unwrap_or_default()
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|spec| {
                let (tile, file) = spec.split_once(':').unwrap();
                (m.dma_offset(u32::from_str_radix(tile, 16).unwrap() << 16).unwrap(), std::fs::read(file).unwrap())
            })
            .collect();
        let ram = Arc::new(m);
        let re = TestRe::new(ram.clone());
        // RE_REPLAY_PRELOAD_EVERY=N: put the tiles back every N writes
        // (2000 by default; a large trace with many tiles wants fewer).
        let every: usize = std::env::var("RE_REPLAY_PRELOAD_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
        for (n, line) in std::fs::read_to_string(path).unwrap().lines().enumerate() {
            if n % every == 0 {
                re.wait_idle();
                let m = &*ram;
                for (o, data) in &preload {
                    m.write(*o, data);
                }
            }
            let mut f = line.split_whitespace();
            let (w, off, val) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
            let (off, val) = (u32::from_str_radix(off, 16).unwrap(), u64::from_str_radix(val, 16).unwrap());
            if w == "w64" {
                re.put64(0x1500_0000 + off, val);
            } else {
                re.put32(0x1500_0000 + off, val as u32);
            }
        }
        re.wait_idle();
        let r = re.regs.lock().unwrap().clone();
        let mut mem = ram.window();
        mem.begin();
        let (w, h) = (1280u32, 1024u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        // RE_REPLAY_BUF=1 or 2: an 8-bit buffer through TLB B or C (the
        // overlay is C) instead, each index as its own grey.
        let low = std::env::var("RE_REPLAY_LOW").is_ok();
        let half = std::env::var("RE_REPLAY_HALF").ok().map(|h| if h == "hi" { 0usize } else { 2 });
        let eight: u32 = std::env::var("RE_REPLAY_BUF").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                if eight != 0 {
                    if let Some(o) = CrimeRe::locate(&r, &mem, Buf { tlb: TLB_A + 0x200 * eight, bd: 1 }, x, y) {
                        let v = mem[o].wrapping_mul(85);
                        rgba[i..i + 4].copy_from_slice(&[v, v, v, 255]);
                    }
                } else if let Some(o) = CrimeRe::locate(&r, &mem, Buf { tlb: TLB_A, bd: 4 }, x, y) {
                    rgba[i..i + 3].copy_from_slice(&mem.bytes(o, (o + 3) - (o)));
                    rgba[i + 3] = 255;
                    // RE_REPLAY_LOW=1: the low byte (an 8-bit window's index) as grey.
                    if low {
                        rgba[i..i + 3].copy_from_slice(&[mem[o + 3]; 3]);
                    }
                    // RE_REPLAY_HALF=lo or hi: a double-pix window's A1R5G5B5 half.
                    if let Some(off) = half {
                        let p = u16::from_be_bytes([mem[o + off], mem[o + off + 1]]) as u32;
                        let e = |v: u32| ((v << 3) | (v >> 2)) as u8;
                        rgba[i..i + 3].copy_from_slice(&[e((p >> 10) & 31), e((p >> 5) & 31), e(p & 31)]);
                    }
                }
            }
        }
        crate::disp::save_png_rgba8(&out, &rgba, w, h).unwrap();
        // RE_REPLAY_PX="x,y x,y": the raw 32-bit words there.
        for xy in std::env::var("RE_REPLAY_PX").unwrap_or_default().split_whitespace() {
            let (x, y) = xy.split_once(',').unwrap();
            let (x, y) = (x.parse::<u32>().unwrap(), y.parse::<u32>().unwrap());
            if let Some(o) = CrimeRe::locate(&r, &mem, Buf { tlb: TLB_A, bd: 4 }, x, y) {
                eprintln!("replay px {x},{y} = {:02x}{:02x}{:02x}{:02x}", mem[o], mem[o + 1], mem[o + 2], mem[o + 3]);
            }
        }
    }

    /// 4Dwm's minimized-window icon: an 8-bit pixmap in linear B copied to
    /// the 8-bit root window (32-bit words, index in the low byte), with
    /// the registers IRIX used.
    #[test]
    fn an_8bit_pixmap_copies_out_of_linear_b_into_the_root_window() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        // Screen: TLB A entry 0 at 0x0100_0000. Pixmap page at 0x0050_0000.
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        let src = 0x1038_fb60u32;
        let page = 0x0050_0000u32;
        let idx = (src >> 12) & 31;
        re.write64(base + TLB_LINEAR_A + 0x80 + 8 * (idx / 2), if idx & 1 == 1 { 0x8000_0000 | (page >> 12) as u64 } else { (0x8000_0000 | (page >> 12) as u64) << 32 });
        for i in 0..88u32 {
            ram.write((page | (src & 0xfff)) as usize + i as usize, &[0x21]);
        }
        re.write32(base + BUFMODE_SRC, 0x1400);
        re.write32(base + BUFMODE_DST, 0x8000_0200);
        re.write32(base + DRAWMODE, 0x0020_0208);
        re.write32(base + LOGIC_OP, 3);
        re.write32(base + PRIMITIVE, 0x0302_0000);
        re.write32(base + VERTEX_0, (10 << 16) | 0);
        re.write32(base + VERTEX_1, (20 << 16) | 0);
        re.write32(base + XFER_STEP_X, 1);
        re.write32(base + XFER_STEP_ROW, 4);
        re.write32(base + (XFER_SRC | 0x800), src);
        let m = &*ram;
        // Pixel (10, 0): word 10 of the first tile row.
        assert_eq!(&m.bytes(0x0100_0000 + 40, (0x0100_0000 + 44) - (0x0100_0000 + 40)), &[0, 0, 0, 0x21]);
    }

    /// NetBSD's Render text: an 8-bit glyph mask in linear B blended onto a
    /// 32-bit RGBA window with ZERO, ONE_MINUS_SRC_ALPHA (black text), and
    /// a 32-bit coloured one with SRC_ALPHA, ONE_MINUS_SRC_ALPHA, with the
    /// registers its crime driver used.
    #[test]
    fn transfers_blend_onto_the_screen() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        let src = 0x1038_f000u32;
        let page = 0x0050_0000u32;
        let idx = (src >> 12) & 31;
        re.write64(base + TLB_LINEAR_A + 0x80 + 8 * (idx / 2), if idx & 1 == 1 { 0x8000_0000 | (page >> 12) as u64 } else { (0x8000_0000 | (page >> 12) as u64) << 32 });
        let screen = 0x0100_0000 + 10 * 4;
        let xfer = |smode: u32, func: u32, step: u32, data: &[u8]| {
            ram.write(screen, &[0xff; 12]);
            ram.write(page as usize, data);
            re.write32(base + BUFMODE_SRC, smode);
            re.write32(base + BUFMODE_DST, 0x228);
            re.write32(base + BLEND_CONST, 0);
            re.write32(base + BLEND_FUNC, func);
            re.write32(base + DRAWMODE, 0x0020_0478);
            re.write32(base + PRIMITIVE, 0x0302_0000);
            re.write32(base + VERTEX_0, 10 << 16);
            re.write32(base + VERTEX_1, 12 << 16);
            re.write32(base + XFER_STEP_X, step);
            re.write32(base + XFER_STEP_ROW, 0);
            re.write32(base + (XFER_SRC | 0x800), src);
            ram.bytes(screen, 12)
        };
        // Alpha 0, half and whole: white, grey and black.
        assert_eq!(xfer(0x1428, 0x05, 1, &[0, 0x80, 0xff]), [255, 255, 255, 255, 127, 127, 127, 127, 0, 0, 0, 0]);
        // Red at alpha 0, half and whole.
        let red = [0xff, 0, 0, 0, 0xff, 0, 0, 0x80, 0xff, 0, 0, 0xff];
        assert_eq!(xfer(0x1628, 0x45, 4, &red), [255, 255, 255, 255, 255, 127, 127, 191, 255, 0, 0, 255]);
    }

    /// The subtile layout is the 16-bit one perfly's maps were solved with,
    /// and places 32-bit texels as o2gltest's self-describing RGBA8 map
    /// showed (index bits x0 y0 y1 x1 x2 x3 x4 y4 y5 y2 y3 x5).
    #[test]
    fn texels_sit_in_the_subtile_layout() {
        for v in 0..128u32 {
            for u in 0..256u32 {
                let idx = (u & 3) | (v & 3) << 2 | ((u >> 2) & 15) << 4 | ((v >> 4) & 3) << 8 | ((v >> 2) & 3) << 10
                    | ((u >> 6) & 1) << 12 | ((u >> 7) & 1) << 13 | ((v >> 6) & 1) << 14;
                assert_eq!(subtile_offset(u * 2, v), idx * 2, "16-bit ({u}, {v})");
            }
        }
        // Past one tile: Quake 2's 256 x 256 RGBA8 map uses entries 80, 81,
        // 84, 85 from 80; its 16-bit one 80 and 84.
        assert_eq!([(0, 0), (1023, 0), (0, 255), (1023, 255)].map(|(bx, y)| subtile_step(bx, y, 4)), [0, 1, 4, 5]);
        assert_eq!([(0, 0), (511, 255)].map(|(bx, y)| subtile_step(bx, y, 4)), [0, 4]);
        let bit = |v: u32, b: u32| (v >> b) & 1;
        for y in 0..64u32 {
            for x in 0..64u32 {
                let idx = bit(x, 0) | bit(y, 0) << 1 | bit(y, 1) << 2 | bit(x, 1) << 3 | bit(x, 2) << 4 | bit(x, 3) << 5
                    | bit(x, 4) << 6 | bit(y, 4) << 7 | bit(y, 5) << 8 | bit(y, 2) << 9 | bit(y, 3) << 10 | bit(x, 5) << 11;
                assert_eq!(subtile_offset(x * 4, y), idx * 4, "32-bit ({x}, {y})");
            }
        }
    }

    /// A 64 x 64 mipmapped RGBA8 map, each level a solid colour where
    /// o2gltest's was: drawn at 32 and 16 pixels wide it shows levels 1
    /// and 2 (NEAREST_MIPMAP_NEAREST), at 128 the base magnified.
    #[test]
    fn mip_levels_come_from_their_places() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        // Texture TLB entries 92 and 93 -> tiles 0x60 and 0x61.
        re.write64(base + TLB_TEX + 8 * 23, (0x8060u64 << 48) | (0x8061u64 << 32));
        let cols: [[u8; 4]; 3] = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        {
            let m = &*ram;
            for (level, (tile, off)) in [(0x60usize, 0usize), (0x61, 0x8000), (0x60, 0xc000)].into_iter().enumerate() {
                let n = 64u32 >> level;
                for y in 0..n {
                    for x in 0..n {
                        m.write((tile << 16) + off + subtile_offset(x * 4, y) as usize, &cols[level]);
                    }
                }
            }
        }
        // Tiled, RGBA, 32-bit, base level 4, 6 below it,
        // NEAREST_MIPMAP_NEAREST, REPEAT, REPLACE.
        re.write32(base + TEX_MODE, 0x0129_1917);
        re.write64(base + TEX_FORMAT, 0x0000_4466_03ff_03ff);
        re.write32(base + BUFMODE_DST, 0x220);
        re.write32(base + PRIMITIVE, 0x0200_0000);
        let gl = |c: u32| 0x4800_0000 | (c * 64);
        // A right triangle `n` pixels on a side at (0, 0) whose s runs 0..1
        // across it (1.14), so the level of detail is log2(64 / n).
        let tri = |n: u32| {
            re.write32(base + DRAWMODE, 0x0040_8078);
            re.write64(base + TRI_V0, (gl(0) as u64) << 32 | gl(0) as u64);
            re.write64(base + TRI_V0 + 8, (gl(n) as u64) << 32 | gl(0) as u64);
            re.write64(base + TRI_V0 + 16, (gl(0) as u64) << 32 | gl(n) as u64);
            re.write64(base + TEX_SQ0, 0);
            re.write64(base + TEX_TQ0, 0);
            re.write32(base + TEX_Q0, 1);
            let step = ((16384 / n) as u64) << 0;
            re.write64(base + TEX_DSQDX, step);
            re.write64(base + TEX_DSQDY, 0);
            re.write64(base + TEX_DTQDX, 0);
            re.write64(base + TEX_DTQDY, step);
            re.write32(base + TEX_DQDX, 0);
            re.write32(base + (TEX_DQDY | 0x800), 0);
        };
        let px = |ram: &Arc<O2Memory>| ram.bytes(0x0100_0000 + 512 + 4, 3);
        tri(128);
        assert_eq!(px(&ram), &[255, 0, 0]);
        tri(32);
        assert_eq!(px(&ram), &[0, 255, 0]);
        tri(16);
        assert_eq!(px(&ram), &[0, 0, 255]);
    }

    /// With texture entries 92 and 93 both valid, a 32 x 32 map is still
    /// read from 0x8000 into 93, its size's place, not from the first valid
    /// entry: Quake 3's 32 x 32 HUD digits were missing.
    #[test]
    fn a_small_map_is_found_whichever_entries_are_valid() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        re.write64(base + TLB_TEX + 8 * 23, (0x8060u64 << 48) | (0x8061u64 << 32));
        {
            let m = &*ram;
            for y in 0..32 {
                for x in 0..32 {
                    m.write((0x61 << 16) + 0x8000 + subtile_offset(x * 4, y) as usize, &[30, 160, 90, 255]);
                }
            }
        }
        // Tiled, RGBA, 32-bit, level 5 of a full pyramid, NEAREST, REPLACE.
        re.write32(base + TEX_MODE, 0x0129_5417);
        re.write64(base + TEX_FORMAT, 0x0000_5555_01ff_01ff);
        re.write32(base + BUFMODE_DST, 0x220);
        re.write32(base + PRIMITIVE, 0x0200_0000);
        let gl = |c: u32| 0x4800_0000 | (c * 64);
        re.write32(base + DRAWMODE, 0x0040_8078);
        re.write64(base + TRI_V0, (gl(0) as u64) << 32 | gl(0) as u64);
        re.write64(base + TRI_V0 + 8, (gl(32) as u64) << 32 | gl(0) as u64);
        re.write64(base + TRI_V0 + 16, (gl(0) as u64) << 32 | gl(32) as u64);
        re.write64(base + TEX_SQ0, 16384);
        re.write64(base + TEX_TQ0, 16384);
        re.write32(base + TEX_Q0, 1);
        re.write64(base + TEX_DSQDX, 512);
        re.write64(base + TEX_DSQDY, 0);
        re.write64(base + TEX_DTQDX, 0);
        re.write64(base + TEX_DTQDY, 512);
        re.write32(base + TEX_DQDX, 0);
        re.write32(base + (TEX_DQDY | 0x800), 0);
        assert_eq!(ram.bytes(0x0100_0000 + 512 + 4, 3), &[30, 160, 90]);
    }

    /// A map smaller than 64 x 64 without mip levels is read from where its
    /// size's level sits in its tile, not from the tile's start: a 16 x 16
    /// RGBA8 map in entry 92 at 0xc000, with the registers IRIX's GL wrote
    /// for one (textest, 2026-10-21). Read from offset 0, it drew black.
    #[test]
    fn a_small_map_is_read_from_its_place_in_the_tile() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        re.write64(base + TLB_TEX + 8 * 23, 0x8060u64 << 48);
        {
            let m = &*ram;
            for y in 0..16 {
                for x in 0..16 {
                    m.write((0x60 << 16) + 0xc000 + subtile_offset(x * 4, y) as usize, &[200, 100, 50, 255]);
                }
            }
        }
        // Tiled, RGBA, 32-bit, level 6 of a full pyramid, NEAREST, REPLACE.
        re.write32(base + TEX_MODE, 0x0129_9017);
        re.write64(base + TEX_FORMAT, 0x0000_6644_00ff_00ff);
        re.write32(base + BUFMODE_DST, 0x220);
        re.write32(base + PRIMITIVE, 0x0200_0000);
        let gl = |c: u32| 0x4800_0000 | (c * 64);
        re.write32(base + DRAWMODE, 0x0040_8078);
        re.write64(base + TRI_V0, (gl(0) as u64) << 32 | gl(0) as u64);
        re.write64(base + TRI_V0 + 8, (gl(16) as u64) << 32 | gl(0) as u64);
        re.write64(base + TRI_V0 + 16, (gl(0) as u64) << 32 | gl(16) as u64);
        re.write64(base + TEX_SQ0, 16384);
        re.write64(base + TEX_TQ0, 16384);
        re.write32(base + TEX_Q0, 1);
        re.write64(base + TEX_DSQDX, 1024);
        re.write64(base + TEX_DSQDY, 0);
        re.write64(base + TEX_DTQDX, 0);
        re.write64(base + TEX_DTQDY, 1024);
        re.write32(base + TEX_DQDX, 0);
        re.write32(base + (TEX_DQDY | 0x800), 0);
        assert_eq!(ram.bytes(0x0100_0000 + 512 + 4, 3), &[200, 100, 50]);
    }

    /// Ordered dithering to 5 bits: exact steps stay exact, and a value
    /// between two steps comes out as a mix of both over a 4 x 4 cell, in
    /// proportion (12 of 8.5 levels up from 96: 4/8 of the cell at 13).
    #[test]
    fn sixteen_bit_texels_follow_their_type() {
        // RGBA4: R in the top nibble (GLQuake's world and status bar).
        assert_eq!(texel16(4, 0xf84c), [255.0, 136.0, 68.0, 204.0]);
        // LUMINANCE_ALPHA: L high, A low; the lone ones where LA8 has them.
        assert_eq!(texel16(8, 0x8040), [128.0, 128.0, 128.0, 64.0]);
        assert_eq!(texel16(7, 0x8040), [128.0, 128.0, 128.0, 255.0]);
        assert_eq!(texel16(6, 0x8040), [128.0; 4]);
        assert_eq!(texel16(5, 0x8040), [255.0, 255.0, 255.0, 64.0]);
        // RGB and RGBA: A1R5G5B5.
        assert_eq!(texel16(2, 0xfc00), [255.0, 0.0, 0.0, 255.0]);
        assert_eq!(texel16(1, 0x001f), [0.0, 0.0, 255.0, 0.0]);
    }

    #[test]
    fn dithering_to_five_bits_mixes_neighbouring_steps() {
        let cell = |c: u32| (0..16).map(|i| CrimeRe::dither5(c, i % 4, i / 4)).collect::<Vec<_>>();
        assert!(cell(96).iter().all(|&v| v == 12));
        assert!(cell(255).iter().all(|&v| v == 31));
        let mid = cell(100);
        assert_eq!(mid.iter().filter(|&&v| v == 13).count(), 8);
        assert!(mid.iter().all(|&v| v == 12 || v == 13));
    }

    #[test]
    fn blending_follows_the_blend_register() {
        let (s, d) = ([200, 100, 0, 128], [0, 100, 200, 255]);
        // ADD, SRC_ALPHA, ONE_MINUS_SRC_ALPHA: IRIX's 0x45.
        assert_eq!(CrimeRe::blend(0x45, 0, s, d), [100, 100, 100, 191]);
        // ADD, ONE, ONE saturates.
        assert_eq!(CrimeRe::blend(0x11, 0, s, d), [200, 200, 200, 255]);
        // MIN and MAX ignore the factors.
        assert_eq!(CrimeRe::blend(0x100, 0, s, d), [0, 100, 0, 128]);
        assert_eq!(CrimeRe::blend(0x200, 0, s, d), [200, 100, 200, 255]);
        // REVERSE_SUBTRACT, ONE, ONE: d - s.
        assert_eq!(CrimeRe::blend(0x411, 0, s, d), [0, 0, 200, 127]);
        // CONSTANT_COLOR, ZERO.
        assert_eq!(CrimeRe::blend(0x080, 0x8040_2000, [255; 4], d), [128, 64, 32, 0]);
    }

    /// A smooth GL triangle with blending 0x45 into a 32-bit RGBA buffer
    /// mixes by source alpha; with the alpha test GREATER 0x80 a
    /// half-transparent one draws nothing.
    #[test]
    fn triangles_blend_and_alpha_test() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        let o = 0x0100_0000 + 512 + 2 * 4;
        ram.write(o, &[0, 0, 200, 255]);
        re.write32(base + BUFMODE_DST, 0x220);
        re.write32(base + PRIMITIVE, 0x0200_0000);
        re.write32(base + BLEND_FUNC, 0x45);
        let gl = |c: u32| 0x4800_0000 | (c * 64);
        let tri = |dm: u32| {
            re.write32(base + DRAWMODE, dm);
            re.write64(base + TRI_V0, (gl(0) as u64) << 32 | gl(0) as u64);
            re.write64(base + TRI_V0 + 8, (gl(16) as u64) << 32 | gl(0) as u64);
            re.write64(base + TRI_V0 + 16, (gl(0) as u64) << 32 | gl(8) as u64);
            // Smooth shading, flat values: R 200, G 0, B 0, A 128 (8.12).
            re.write64(base + SHADE_R, (200u64 << 12) << 32);
            re.write64(base + SHADE_R + 8, 128u64 << 12);
            re.write32(base + (FG | 0x800), 0);
        };
        tri(0x0041_0478);
        assert_eq!(&ram.bytes(o, 4), &[100, 0, 100, 191]);
        re.write32(base + ALPHA_TEST, (4 << 8) | 0x80);
        ram.write(o, &[0, 0, 200, 255]);
        tri(0x0041_0878);
        assert_eq!(&ram.bytes(o, 4), &[0, 0, 200, 255]);
    }

    /// A shaded GL point takes its colour from the shade registers and
    /// lands in its double-pix half, as bz's world needs.
    #[test]
    fn a_shaded_gl_point_takes_the_shade_colour() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        // Double-pix A1R5G5B5, the half at offset 2.
        re.write32(base + BUFMODE_DST, 0x226);
        re.write32(base + DRAWMODE, 0x0051_0178);
        re.write32(base + PRIMITIVE, 0x0000_0000);
        let gl = |c: u32| 0x4800_0000 | (c * 64);
        re.write64(base + TRI_V0, (gl(2) as u64) << 32 | gl(1) as u64);
        re.write64(base + SHADE_R, (248u64 << 12) << 32);
        re.write64(base + SHADE_R + 8, 0);
        re.write32(base + (FG | 0x800), 0);
        let o = 0x0100_0000 + 512 + 2 * 4;
        let m = &*ram;
        assert_eq!(&m.bytes(o + 2, 2), &[0x7c, 0x00], "red 31 in the half at offset 2");
    }

    /// A shaded GL point into an 8-bit colour-index buffer stores FG's
    /// index and reads nothing as a word: flight's points read a 32-bit
    /// destination at an odd byte and stopped the emulator.
    #[test]
    fn a_shaded_gl_point_into_an_8bit_buffer_keeps_fg() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        re.write32(base + BUFMODE_DST, 0x000);
        re.write32(base + DRAWMODE, 0x0051_0178);
        re.write32(base + PRIMITIVE, 0x0000_0000);
        let gl = |c: u32| 0x4800_0000 | (c * 64);
        re.write64(base + TRI_V0, (gl(3) as u64) << 32 | gl(1) as u64);
        re.write64(base + SHADE_R, (248u64 << 12) << 32);
        re.write32(base + (FG | 0x800), 7);
        assert_eq!(ram.bytes(0x0100_0000 + 512 + 3, 1), vec![7]);
    }

    /// An unblended GL line is drawn whatever its colour's alpha: amesh's
    /// mesh, Gouraud lines with alpha 0 (IRIS GL RGB colours), drew nothing.
    #[test]
    fn an_unblended_gl_line_ignores_its_alpha() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        re.write32(base + BUFMODE_DST, 0x220);
        // GL, shaded, no blend; a line from (0, 1) to (8, 1).
        re.write32(base + DRAWMODE, 0x0051_0178);
        re.write32(base + PRIMITIVE, 0x0104_0000);
        let gl = |c: u32| 0x4800_0000 | (c * 64);
        re.write64(base + TRI_V0, (gl(0) as u64) << 32 | gl(1) as u64);
        re.write64(base + TRI_V0 + 8, (gl(8) as u64) << 32 | gl(1) as u64);
        // R 200, G 100, B 50, A 0 (8.12), flat.
        re.write64(base + SHADE_R, (200u64 << 12) << 32 | 100u64 << 12);
        re.write64(base + SHADE_R + 8, (50u64 << 12) << 32);
        re.write32(base + (FG | 0x800), 0);
        let o = 0x0100_0000 + 512 + 2 * 4;
        assert_eq!(&ram.bytes(o, 3), &[200, 100, 50]);
    }

    /// A flat colour-index GL triangle stores FG's index as it stands, as
    /// gr_osview's bars ask (FG 2 into a pixel-type-0 buffer).
    #[test]
    fn a_flat_colour_index_triangle_stores_fg_as_its_index() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        re.write32(base + BUFMODE_DST, 0x200);
        re.write32(base + DRAWMODE, 0x0040_0178);
        re.write32(base + PRIMITIVE, 0x0200_0000);
        let gl = |c: u32| 0x4800_0000 | (c * 64);
        re.write64(base + TRI_V0, (gl(0) as u64) << 32 | gl(0) as u64);
        re.write64(base + TRI_V0 + 8, (gl(16) as u64) << 32 | gl(0) as u64);
        re.write64(base + TRI_V0 + 16, (gl(0) as u64) << 32 | gl(8) as u64);
        re.write32(base + (FG | 0x800), 2);
        let m = &*ram;
        // Pixel (2, 1): row 1 of the first tile, 512 bytes a row.
        let o = 0x0100_0000 + 512 + 2 * 4;
        assert_eq!(&m.bytes(o, 4), &[0, 0, 0, 2]);
    }

    /// The stencil test as mirrors uses it: a triangle marks the stencil
    /// (ALWAYS, ref 1, REPLACE), then a larger one drawn where it EQUALs 1
    /// lands only where the first was. Without the test the reflection was
    /// drawn over the whole scene.
    #[test]
    fn a_stencilled_triangle_draws_only_where_the_stencil_was_marked() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        // TLB C (the stencil-depth buffer) entry 0 -> 0x0110_0000, zeroed.
        re.write64(base + TLB_A + 0x400, (0x8000u64 | 0x110) << 48);
        re.write32(base + BUFMODE_DST, 0x200);
        re.write32(base + PRIMITIVE, 0x0200_0000);
        re.write32(base + STENCIL_MASK, 0xff);
        let gl = |c: u32| 0x4800_0000 | (c * 64);
        let tri = |n: u32, fg: u32| {
            re.write64(base + TRI_V0, (gl(0) as u64) << 32 | gl(0) as u64);
            re.write64(base + TRI_V0 + 8, (gl(n) as u64) << 32 | gl(0) as u64);
            re.write64(base + TRI_V0 + 16, (gl(0) as u64) << 32 | gl(n) as u64);
            re.write32(base + (FG | 0x800), fg);
        };
        // Flat, colour index, stencil test on, depth test off.
        re.write32(base + DRAWMODE, 0x0040_0179);
        re.write32(base + STENCIL_MODE, 0x01ff_7333);
        tri(8, 1);
        re.write32(base + STENCIL_MODE, 0x01ff_2000);
        tri(32, 2);
        let m = &*ram;
        let px = |x: usize, y: usize| m.bytes(0x0100_0000 + y * 512 + x * 4 + 3, 1)[0];
        let st = |x: usize, y: usize| m.bytes(0x0110_0000 + y * 512 + x * 4, 1)[0];
        assert_eq!((px(2, 1), st(2, 1)), (2, 1), "inside the mark: drawn, stencil 1");
        assert_eq!((px(20, 1), st(20, 1)), (0, 0), "outside it: neither");
    }

    /// A 32-bit linear end at a byte address (a start off the word) is taken
    /// down to its word, not stored unaligned: opening demograph aborted the
    /// emulator on a write to 0x...001. Its pixels then follow a word apart.
    #[test]
    fn a_32bit_linear_end_off_its_word_does_not_abort() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        {
            let m = &*ram;
            for x in 0..8u32 {
                m.write((0x0100_0000 + x * 4) as usize, &(0x1100_0000 | x).to_be_bytes());
            }
        }
        let dst = 0x1038_f001u32;
        let page = 0x0050_0000u32;
        let idx = (dst >> 12) & 31;
        re.write64(base + TLB_LINEAR_A + 0x80 + 8 * (idx / 2), if idx & 1 == 1 { 0x8000_0000 | (page >> 12) as u64 } else { (0x8000_0000 | (page >> 12) as u64) << 32 });
        re.write32(base + BUFMODE_SRC, 0x0200);
        re.write32(base + BUFMODE_DST, 0x1600);
        re.write32(base + DRAWMODE, 0x0020_0078);
        re.write32(base + PRIMITIVE, 0x0302_0000);
        re.write32(base + VERTEX_0, 0);
        re.write32(base + VERTEX_1, 7 << 16);
        re.write32(base + XFER_STEP_X, 1);
        re.write32(base + XFER_STEP_ROW, 0);
        re.write32(base + XFER_DST, dst);
        re.write32(base + (XFER_SRC | 0x800), 0);
        let m = &*ram;
        let at = (page + (dst & 0xfff & !3)) as usize;
        for x in 0..8u32 {
            let v = u32::from_be_bytes(m.bytes(at + 4 * x as usize, 4).try_into().unwrap());
            assert_eq!(v, 0x1100_0000 | x, "pixel {x}, from the word the address falls in");
        }
    }

    #[test]
    fn rgb_pixmaps_become_rgba_on_the_way_to_the_screen() {
        // BufMode pixel types: 1 RGB, 2 RGBA, 3 ABGR, in bits 7:4.
        assert_eq!(CrimeRe::convert(0x00c0_c0c0, 0x1618, 0x228, 4), 0xc0c0_c0ff);
        assert_eq!(CrimeRe::convert(0x1122_3344, 0x228, 0x1618, 4), 0x0011_2233);
        assert_eq!(CrimeRe::convert(0x1122_3344, 0x228, 0x238, 4), 0x4433_2211);
        assert_eq!(CrimeRe::convert(0x1122_3344, 0x228, 0x228, 4), 0x1122_3344);
    }

    /// 4Dwm redraws jot's frame with all five screen masks (mode 0x3ff, as
    /// IRIX set them): the four border strips and the title bar in mask 4.
    #[test]
    fn all_five_screen_masks_clip_and_their_max_edges_are_exclusive() {
        let mut r = vec![0u32; 0x4000 / 4];
        let rects = [(24, 676, 378, 684), (24, 294, 32, 676), (866, 32, 874, 294), (24, 32, 32, 294), (24, 0, 874, 32)];
        for (i, (x0, y0, x1, y1)) in rects.into_iter().enumerate() {
            r[(CLIP_RECT0 / 4) as usize + 2 * i] = (x0 << 16) | y0;
            r[(CLIP_RECT0 / 4) as usize + 2 * i + 1] = (x1 << 16) | y1;
        }
        r[(CLIP_MODE / 4) as usize] = 0x3ff;
        assert!(CrimeRe::clip_ok(&r, 400, 15), "the title bar, in mask 4");
        assert!(CrimeRe::clip_ok(&r, 24, 100), "the left border");
        assert!(!CrimeRe::clip_ok(&r, 874, 15), "max x is exclusive");
        assert!(!CrimeRe::clip_ok(&r, 400, 100), "the client area is not in any mask");
        // Mode 0x21: mask 0 alone, inside; 0x20: mask 0 alone, outside.
        r[(CLIP_MODE / 4) as usize] = 0x21;
        assert!(CrimeRe::clip_ok(&r, 30, 680) && !CrimeRe::clip_ok(&r, 400, 15));
        r[(CLIP_MODE / 4) as usize] = 0x20;
        assert!(!CrimeRe::clip_ok(&r, 30, 680) && CrimeRe::clip_ok(&r, 400, 15));
    }

    /// Writes left in the interface buffer as fast as it takes them draw
    /// what writes carried out one at a time do, and STATUS says busy
    /// until they are done.
    #[test]
    fn a_pipelined_engine_draws_the_same_and_reports_idle_after() {
        let draw = |re: &TestRe, wait: bool| {
            let base = 0x1500_0000;
            let w = |a: u32, v: u32| if wait { re.write32(a, v) } else { re.put32(a, v) };
            re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
            w(base + DRAWMODE, 0x0000_0278);
            w(base + LOGIC_OP, 3);
            w(base + PRIMITIVE, 0x0300_0000);
            w(base + BUFMODE_DST, 0x228);
            for i in 0..200u32 {
                w(base + FG, 0x1020_3000 | i);
                w(base + VERTEX_0, (i % 100) << 16 | (i / 100));
                w(base + (VERTEX_1 | 0x800), ((i % 100) + 3) << 16 | (i / 100));
            }
        };
        let one_ram = Arc::new(O2Memory::single(32 << 20));
        draw(&TestRe::new(one_ram.clone()), true);
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        draw(&re, false);
        let mut spins = 0;
        while re.read32((0x1500_0000 + STATUS) as u64).data != RE_STATUS_IDLE {
            spins += 1;
            assert!(spins < 50_000_000, "never idle");
        }
        assert_eq!(re.read32((0x1500_0000 + FG) as u64).data, 0x1020_3000 | 199, "a read when idle sees the last write");
        assert_eq!(ram.bytes(0x0100_0000, 0x1000), one_ram.bytes(0x0100_0000, 0x1000));
    }

    /// A guest that polls STATUS waiting for idle, a fixed number of times
    /// as the kernel's early console setup does, sees the engine finish
    /// even when the engine's thread is kept from running for far longer
    /// than one poll's wait. The first read after a write still answers
    /// busy without waiting long, so a driver that only checks for room
    /// keeps drawing while the engine works.
    #[test]
    fn a_status_wait_loop_outlasts_an_engine_held_off_the_processor() {
        let re = Arc::new(CrimeRe::new(Arc::new(O2Memory::single(32 << 20))));
        let status = |re: &CrimeRe| re.read32((0x1500_0000 + STATUS) as u64).data;
        for i in 0..1000u32 {
            assert_eq!(re.write32((0x1500_0000 + FG) as u64, i), BUS_OK);
        }
        assert_eq!(re.status_wait(), RE_STATUS_POLL_WAIT, "the first poll after a write waits briefly");
        assert_ne!(status(&re), RE_STATUS_IDLE, "nothing has run yet");
        assert_eq!(re.status_wait(), RE_STATUS_LOOP_WAIT, "a second poll is a waiting loop");
        assert_eq!(re.write32((0x1500_0000 + FG) as u64, 1000), BUS_OK);
        assert_eq!(re.status_wait(), RE_STATUS_POLL_WAIT, "a write ends the loop");
        assert_ne!(status(&re), RE_STATUS_IDLE);
        // The engine's thread starts 200 ms from now. 256 polls of the
        // first wait alone give up after about 26 ms.
        let late = re.clone();
        let starter = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            late.start(late.clone(), |r| r);
        });
        let idle = (0..256).any(|_| status(&re) == RE_STATUS_IDLE);
        starter.join().unwrap();
        assert!(idle, "the waiting loop ran out before the engine finished");
        assert_eq!(re.read32((0x1500_0000 + FG) as u64).data, 1000);
    }

    /// A full interface buffer holds the CPU's write back (BUS_BUSY) with
    /// nothing lost, and the status word shows the level as CRIME's 64
    /// entries would: what is queued below that, one entry past it.
    #[test]
    fn a_full_interface_buffer_asks_the_cpu_to_retry() {
        let re = CrimeRe::new(Arc::new(O2Memory::single(32 << 20)));
        let level = |re: &CrimeRe| re.read32((0x1500_0000 + STATUS) as u64).data;
        for i in 0..(IB_DEPTH as u32 - 1) {
            assert_eq!(re.write32((0x1500_0000 + FG) as u64, i), BUS_OK, "write {i}");
            if i == 9 {
                assert_eq!(level(&re), 10 << RE_STATUS_IB_LEVEL_SHIFT, "ten queued, busy");
            }
        }
        assert_eq!(re.write32((0x1500_0000 + FG) as u64, 999), BUS_BUSY);
        assert_eq!(level(&re), 1 << RE_STATUS_IB_LEVEL_SHIFT, "full: busy, one entry shown");
        assert_eq!(re.read32((0x1500_0000 + FG) as u64).status, BUS_BUSY);
        let re = Arc::new(re);
        re.start(re.clone(), |r| r);
        while re.write32((0x1500_0000 + FG) as u64, 999) == BUS_BUSY {
            std::hint::spin_loop();
        }
        re.wait_idle();
        assert_eq!(re.read32((0x1500_0000 + FG) as u64).data, 999);
    }

    /// The X server's Motif bevel: a rectangle into the 32-bit ABGR screen
    /// with FG 0x00d4d4d4 must land as the grey 0xd4d4d400 the RGB8 window
    /// shows, and into an RGBA window FG goes in unchanged.
    #[test]
    fn rectangles_store_fg_in_the_destination_pixel_type() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        re.write64(base + TLB_A, (0x8000u64 | 0x100) << 48);
        re.write32(base + DRAWMODE, 0x0000_0278);
        re.write32(base + LOGIC_OP, 3);
        re.write32(base + PRIMITIVE, 0x0300_0000);
        for (mode, fg, want) in [(0x238, 0x00d4_d4d4u32, [0xd4, 0xd4, 0xd4, 0]), (0x228, 0xc0c0_c000, [0xc0, 0xc0, 0xc0, 0])] {
            re.write32(base + BUFMODE_DST, mode);
            re.write32(base + FG, fg);
            re.write32(base + VERTEX_0, (4 << 16) | 0);
            re.write32(base + (VERTEX_1 | 0x800), (5 << 16) | 0);
            let m = &*ram;
            assert_eq!(&m.bytes(0x0100_0000 + 16, 4), &want, "mode {mode:#x}");
        }
    }

    /// glReadPixels as IRIX's OpenGL asks for it: a GL rectangle (13.6
    /// vertices, both ends included) from the window's frame buffer into
    /// linear A, bottom row first, packed (XFER_DST_STRIDE 4). The
    /// source pixel is window-relative and the source window offset, which
    /// carries the 4096 guard band, places it modulo 4096.
    #[test]
    fn a_read_back_fills_the_linear_buffer_bottom_row_first() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        // One 64 KB tile at 0x0010_0000: 128 x 128 pixels of 32 bits.
        let tile = 0x0010_0000u32;
        re.write64(base + TLB_A, (0x8000u64 | (tile >> 16) as u64) << 48);
        let px = |x: u32, y: u32| 0xaa00_0000 | x << 8 | y;
        {
            let m = &*ram;
            for y in 60..70u32 {
                for x in 60..70u32 {
                    m.write((tile + y * 512 + x * 4) as usize, &px(x, y).to_be_bytes());
                }
            }
        }
        // Linear A page 2 (byte address 0x2000) onto 0x0030_0000.
        let buf = 0x0030_0000u32;
        re.write64(base + TLB_LINEAR_A + 8, (1u64 << 31 | (buf >> 12) as u64) << 32);
        // A window at (32, 32): offset 32 - 4096 in each half.
        re.write32(base + WINOFFSET_SRC, 0xf020_f020);
        re.write32(base + BUFMODE_SRC, 0x228);
        re.write32(base + BUFMODE_DST, 0x1228);
        re.write32(base + DRAWMODE, draw::GL | draw::PIXEL_XFER | 0xf << 3);
        re.write32(base + PRIMITIVE, 3 << 24);
        // GL rows 30 and 31 of a 64-row window: rows 33 and 32 from its top.
        let v = |x: u32, y: u32| ((4096 + x) << 6, (4096 + y) << 6);
        let ((x0, y0), (x1, y1)) = (v(30, 33), v(33, 32));
        re.write32(base + TRI_V0, x0);
        re.write32(base + TRI_V0 + 4, y0);
        re.write32(base + TRI_V0 + 8, x1);
        re.write32(base + TRI_V0 + 12, y1);
        re.write32(base + XFER_SRC, 30 << 16 | 33);
        re.write32(base + XFER_STEP_X, 1);
        re.write32(base + XFER_STEP_ROW, 1);
        re.write32(base + XFER_DST_STRIDE, 4);
        re.write32(base + (XFER_DST | 0x800), 0x2000);
        let m = &*ram;
        let got = |i: u32| u32::from_be_bytes(m.bytes((buf + 4 * i) as usize, 4).try_into().unwrap());
        for (j, row) in [65u32, 64].into_iter().enumerate() {
            for i in 0..4u32 {
                assert_eq!(got(4 * j as u32 + i), px(62 + i, row), "pixel {i} of row {j} is screen ({}, {row})", 62 + i);
            }
        }
    }

    /// IRIX zeroes pages with the MTE: linear A mapped onto the pages in
    /// pairs, then a clear from DST0 to DST1 with the fill colour.
    #[test]
    fn the_mte_clears_pages_through_linear_a_as_irix_asks() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        ram.write(0x0011_3fff, &[0x55; 0x2_0002]);
        let re = TestRe::new(ram.clone());
        let base = 0x1500_0000;
        let addr = 0x0011_4000u32;
        let page = addr >> 12;
        for i in 0..16u32 {
            let p = page + 2 * i;
            let pair = ((1u64 << 31 | p as u64) << 32) | (1u64 << 31 | (p + 1) as u64);
            re.write64(base + TLB_LINEAR_A + 8 * ((p & 31) >> 1), pair);
        }
        re.write32(base + MTE_DST0, addr);
        re.write32(base + MTE_DST1, addr + 32 * 4096 - 1);
        re.write32(base + MTE_FG, 0);
        re.write32(base + MTE_BYTEMASK, 0xffff_ffff);
        re.write32(base + (MTE_MODE | 0x800), 0x11);
        let m = &*ram;
        assert!(m.bytes(addr as usize, ((addr + 32 * 4096) as usize) - (addr as usize)).iter().all(|&b| b == 0), "all 128 KB cleared");
        assert_eq!(m[addr as usize - 1], 0x55, "the byte before is untouched");
        assert_eq!(m[(addr + 32 * 4096) as usize], 0x55, "the byte after is untouched");
    }

    /// IRIX's mode set writes the overlay shadow with DMA on and then waits
    /// for the live register to read the old state; it must see it until
    /// vertical blanking, and the new value after.
    #[test]
    fn a_running_beam_latches_shadow_controls_at_the_next_frame() {
        let t = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let tc = t.clone();
        let g = Gbe::new(Arc::new(move || tc.load(std::sync::atomic::Ordering::Relaxed)));
        let base = 0x1600_0000;
        g.write32((base + DOTCLOCK) as u64, 0x0010_020f);
        g.write32((base + VT_XYMAX) as u64, 0x0042_9690);
        g.write32((base + VT_XY) as u64, 0);
        g.write32((base + OVR_CONTROL_SHADOW) as u64, 0x0035_b9c1);
        assert_eq!(g.read32((base + OVR_CONTROL) as u64).data & 1, 0, "the old state until blanking");
        // A frame at 108 MHz is about 15 ms of guest time.
        t.store(40_000, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(g.read32((base + OVR_CONTROL) as u64).data, 0x0035_b9c1, "the new value after it");
    }

    /// Fill a rectangle through TLB A, then scan it out through a tile list.
    #[test]
    fn a_filled_rectangle_comes_out_of_scanout_in_its_cmap_colour() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let g = gbe();
        // One 64 KB tile at 0x0010_0000, so a 512 x 128 screen at 8 bpp.
        let tile = 0x0010_0000u32;
        let list = 0x0020_0000u32;
        ram.write(list as usize, &((tile >> 16) as u16).to_be_bytes());
        // TLB A entry 0: valid, tile address >> 16.
        re.write64(0x1500_0000 + TLB_A, (0x8000u64 | (tile >> 16) as u64) << 48);
        re.write32(0x1500_0000 + BUFMODE_DST, 0);
        re.write32(0x1500_0000 + DRAWMODE, 0xf << 3);
        re.write32(0x1500_0000 + PRIMITIVE, 3 << 24);
        re.write32(0x1500_0000 + FG, 0x0505_0505);
        re.write32(0x1500_0000 + VERTEX_0, (10 << 16) | 20);
        re.write32(0x1500_0000 + (VERTEX_1 | 0x800), (19 << 16) | 29);
        {
            let m = &*ram;
            assert_eq!(m[(tile + 20 * 512 + 10) as usize], 5);
            assert_eq!(m[(tile + 29 * 512 + 19) as usize], 5, "inclusive corner");
            assert_eq!(m[(tile + 29 * 512 + 20) as usize], 0);
        }
        // GBE: 512 x 128, one whole tile per row, 8 bpp, WID 0 = I8, gamma bypass.
        for (off, v) in [
            (CTRLSTAT, 0x300a_a000u32),
            (DOTCLOCK, 0x0010_020f),
            (VT_XYMAX, 0x0042_9690),
            // Before the beam runs, so the copy to live is immediate: guest
            // time stands still here and no blanking would come.
            (FRM_CONTROL_SHADOW, list | 1),
            (VT_XY, 0),
            (VT_HCMAP, 512 << 12),
            (VT_VCMAP, 128 << 12),
            (FRM_SIZE_TILE, 1 << 5),
            (FRM_SIZE_PIXEL, 128 << 16),
            (MODE, 0x403),
            (CMAP + 4 * 5, 0x11223300),
        ] {
            g.write32((0x1600_0000 + off) as u64, v);
        }
        let mut out = vec![0u8; 2048 * 2048 * 4];
        let (w, h) = g.render_from(&ram, &mut out).expect("display on");
        assert_eq!((w, h), (512, 128));
        let px = |x: usize, y: usize| &out[(y * 512 + x) * 4..(y * 512 + x) * 4 + 3];
        assert_eq!(px(10, 20), [0x11, 0x22, 0x33]);
        assert_eq!(px(9, 20), [0, 0, 0]);
    }

    /// IRIX draws menu text into the 8-bit overlay as stippled lines with
    /// FG 0x00000002: every set bit is index 2, whatever its byte in the word.
    #[test]
    fn an_8bit_stippled_line_takes_its_colour_from_the_low_byte() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let re = TestRe::new(ram.clone());
        let tile = 0x0010_0000u32;
        let b = 0x1500_0000;
        // TLB C (buffer 2), entry 16: the second row of tiles, y 128-255.
        re.write64(b + TLB_A + 0x400 + 0x20, (0x8000u64 | (tile >> 16) as u64) << 48);
        re.write32(b + BUFMODE_DST, 0x800);
        re.write32(b + DRAWMODE, 0x0008_0278);
        re.write32(b + PRIMITIVE, 0x0104_0020);
        re.write32(b + FG, 2);
        re.write32(b + LOGIC_OP, 3);
        re.write32(b + STIPPLE_MODE, 0x001f_0000);
        re.write64(b + VERTEX_0, 0x0081_00b4_0089_00b4);
        re.write32(b + (STIPPLE_PATTERN | 0x800), 0xff00_0000);
        let m = &*ram;
        let row = (tile + (0xb4 - 128) * 512 + 0x81) as usize;
        assert_eq!(&m.bytes(row, (row + 9) - (row)), &[2, 2, 2, 2, 2, 2, 2, 2, 0], "eight pixels, the end one skipped");
    }

    #[test]
    fn overlay_pixels_cover_the_normal_planes_and_zero_is_clear() {
        let ram = Arc::new(O2Memory::single(32 << 20));
        let g = gbe();
        // Normal planes: one tile of 5s. Overlay: its own tile, one pixel 7.
        let (tile, list, otile, olist) = (0x0010_0000u32, 0x0020_0000u32, 0x0030_0000u32, 0x0020_0100u32);
        {
            let m = &*ram;
            m.write(list as usize, &((tile >> 16) as u16).to_be_bytes());
            m.write(olist as usize, &((otile >> 16) as u16).to_be_bytes());
            m.write(tile as usize, &[5u8; 0x10000]);
            m.write((otile + 4 * 512 + 3) as usize, &[7]);
        }
        for (off, v) in [
            (CTRLSTAT, 0x300a_a000u32),
            (DOTCLOCK, 0x0010_020f),
            (VT_XYMAX, 0x0042_9690),
            (FRM_CONTROL_SHADOW, list | 1),
            (OVR_CONTROL_SHADOW, olist | 1),
            (VT_XY, 0),
            (VT_HCMAP, 512 << 12),
            (VT_VCMAP, 128 << 12),
            (FRM_SIZE_TILE, 1 << 5),
            (OVR_WIDTH_TILE, 1 << 5),
            (FRM_SIZE_PIXEL, 128 << 16),
            (MODE, 0x403),
            (CMAP + 4 * 5, 0x11223300),
            (CMAP + 4 * 0x1107, 0xaabbcc00),
            // Identity gamma for the overlay colour's three bytes.
            (GMAP + 4 * 0xaa, 0xaaaaaa00),
            (GMAP + 4 * 0xbb, 0xbbbbbb00),
            (GMAP + 4 * 0xcc, 0xcccccc00),
        ] {
            g.write32((0x1600_0000 + off) as u64, v);
        }
        let mut out = vec![0u8; 2048 * 2048 * 4];
        g.render_from(&ram, &mut out).expect("display on");
        let px = |x: usize, y: usize| &out[(y * 512 + x) * 4..(y * 512 + x) * 4 + 3];
        assert_eq!(px(3, 4), [0xaa, 0xbb, 0xcc], "overlay colour from entry 0x1100 + 7");
        assert_eq!(px(4, 4), [0x11, 0x22, 0x33], "overlay 0 shows the normal planes");
    }
}
