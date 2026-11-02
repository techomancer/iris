//! O2 graphics annotated trace (`re trace <file> [re,cpu,gbe,crime,irq|all]`, also
//! as `gbe trace` and `crime trace`), in the REX3, GR2 and IMPACT format.
//!
//! Trace lines look like
//!   `    1.234567 RE  #000123 DRAWMODE = 0x00408178  ; GL TEXTURE DITHER LANES=f`
//!   `    1.234570 RE  #000124 TRI_V2 GO = 0x48004000480a8000 -> TRI edge 2 ...`
//!   `    1.234580 CPU #000071 rd STATUS = 0x1e000000  (x5123)`
//!   `    1.234590 GBE #000072 wr FRM_CONTROL_SHADOW = 0x03297201  ; list 0x3297200 dma`
//!   `    1.234595 GBE #000073 wr CMAP[0..255] = 0xffffff00  (x256)`
//!   `    1.234600 CRM #000074 rd INTSTAT = 0x00010000`
//! with the time in seconds since the trace started. RE lines are interface
//! buffer entries as the engine's thread applies them, a primitive's start
//! decoded with the registers it draws with; CPU lines are reads of the RE's
//! registers; GBE and CRM lines are accesses to those blocks, repeats of one
//! register collapsed and runs through a table merged.
//!
//! All three devices share one sink so that their lines interleave in time.
//! `IRIS_O2_TRACE=<file>` starts a full trace at power-on. All of it needs
//! the `gputrace` feature; without it `on` is a constant false.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::dev::o2gfx::{gbe_reg, gl_coord, re_reg};
use crate::trace_writer::TraceWriter;

/// Interface buffer entries as the engine applies them.
pub const TRACE_RE: u32 = 1;
/// CPU reads of the RE's registers.
pub const TRACE_CPU: u32 = 2;
/// GBE register accesses, shadow latches and timing interrupts.
pub const TRACE_GBE: u32 = 4;
/// CRIME register accesses, other than the interrupt registers.
pub const TRACE_CRIME: u32 = 8;
/// CRIME's interrupt registers: IRIX writes the mask on every spl change, so
/// this outweighs everything else and `all` leaves it out.
pub const TRACE_IRQ: u32 = 16;
/// The categories `all` and `IRIS_O2_TRACE` mean.
pub const TRACE_ALL: u32 = TRACE_RE | TRACE_CPU | TRACE_GBE | TRACE_CRIME;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Block {
    Cpu,
    Gbe,
    Crime,
}

/// Pending run of similar accesses, collapsed into one line.
struct Run {
    block: Block,
    write: bool,
    bits: u32,
    first: u32,
    last: u32,
    val: u64,
    count: u64,
}

struct Sink {
    out: Option<TraceWriter>,
    start: Option<Instant>,
    re_seq: u64,
    seq: u64,
    run: Option<Run>,
    last_flush: Option<Instant>,
}

impl Sink {
    fn line(&mut self, text: std::fmt::Arguments) {
        let t = self.start.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
        if let Some(f) = self.out.as_mut() {
            f.line(format_args!("{t:12.6} {text}"));
        }
    }

    fn close_run(&mut self) {
        let Some(r) = self.run.take() else { return };
        let seq = self.seq;
        let dir = if r.write { "wr" } else { "rd" };
        let (tag, name, note) = match r.block {
            Block::Cpu => ("CPU", re_name(r.first), String::new()),
            Block::Gbe => ("GBE", gbe_name(r.first, r.last), if r.write { gbe_note(r.first, r.val as u32) } else { String::new() }),
            Block::Crime => ("CRM", crime_name(r.first), String::new()),
        };
        let val = if r.bits > 32 { format!("{:#018x}", r.val) } else { format!("{:#010x}", r.val) };
        let times = if r.count > 1 { format!("  (x{})", r.count) } else { String::new() };
        self.line(format_args!("{tag:3} #{seq:06} {dir} {name} = {val}{times}{note}"));
    }
}

pub struct O2Trace {
    mask: AtomicU32,
    s: Mutex<Sink>,
}

impl Default for O2Trace {
    fn default() -> Self {
        Self { mask: AtomicU32::new(0), s: Mutex::new(Sink { out: None, start: None, re_seq: 0, seq: 0, run: None, last_flush: None }) }
    }
}

/// Parse a trace category list: `re,cpu,gbe,crime,irq` or `all`.
pub fn trace_mask(s: &str) -> u32 {
    s.split(',').fold(0, |m, w| m | match w {
        "re" => TRACE_RE,
        "cpu" => TRACE_CPU,
        "gbe" => TRACE_GBE,
        "crime" => TRACE_CRIME,
        "irq" => TRACE_IRQ,
        _ => TRACE_ALL,
    })
}

impl O2Trace {
    /// Whether a trace of `what` runs. A constant false without `gputrace`,
    /// so every `if tr.on(..)` on a bus or engine path compiles away,
    /// formatting and all.
    #[inline(always)]
    pub fn on(&self, what: u32) -> bool {
        cfg!(feature = "gputrace") && self.mask.load(Ordering::Relaxed) & what != 0
    }

    /// One access to a register block. Repeats collapse; polled registers
    /// (`poll`) collapse whatever they read, and writes through a table
    /// merge into one range (`table`).
    pub fn access(&self, block: Block, write: bool, bits: u32, off: u32, val: u64, poll: bool, table: bool) {
        let mut t = self.s.lock().unwrap();
        if let Some(r) = t.run.as_mut() {
            if r.block == block && r.write == write && r.bits == bits {
                let same = r.first == off && (poll || r.val == val);
                let next = table && off == r.last + 4;
                if same || next {
                    r.count += 1;
                    r.last = off;
                    r.val = val;
                    return;
                }
            }
        }
        t.close_run();
        t.seq += 1;
        t.run = Some(Run { block, write, bits, first: off, last: off, val, count: 1 });
    }

    /// A free-form event line from a block.
    pub fn event(&self, tag: &str, text: std::fmt::Arguments) {
        let mut t = self.s.lock().unwrap();
        t.close_run();
        t.seq += 1;
        let seq = t.seq;
        t.line(format_args!("{tag:3} #{seq:06} {text}"));
    }

    /// One interface buffer entry, before it is applied. `regs` holds the
    /// registers it will run with, the entry itself included.
    pub fn re_entry(&self, a: u32, val: u64, wide: bool, regs: &[u32]) {
        let start = (a & 0xf000 == 0x2000 || a & 0xf000 == 0x3000) && a & 0x800 != 0;
        let base = if start { a & !0x800 } else { a };
        let v = if wide { format!("{val:#018x}") } else { format!("{:#010x}", val as u32) };
        let go = if start { " GO" } else { "" };
        let run = if start {
            if a & 0xf000 == 0x2000 { format!(" -> {}", decode_prim(regs)) } else { format!(" -> {}", decode_mte(regs)) }
        } else {
            let note = re_note(base, val as u32);
            if note.is_empty() { note } else { format!("  ; {note}") }
        };
        let name = re_name(base);
        let mut t = self.s.lock().unwrap();
        t.close_run();
        t.re_seq += 1;
        let seq = t.re_seq;
        t.line(format_args!("RE  #{seq:06} {name}{go} = {v}{run}"));
    }

    /// Hand the trace to its writer thread (called from the machine's service
    /// thread; it ships at most twice a second unless forced).
    pub fn flush(&self, force: bool) {
        if !cfg!(feature = "gputrace") || (self.mask.load(Ordering::Relaxed) == 0 && !force) {
            return;
        }
        let mut t = self.s.lock().unwrap();
        if force || t.last_flush.map_or(true, |l| l.elapsed().as_millis() >= 500) {
            t.close_run();
            if let Some(f) = t.out.as_mut() {
                f.flush();
            }
            t.last_flush = Some(Instant::now());
        }
    }

    /// Start the annotated trace into `path` for the categories in `mask`.
    pub fn start(&self, path: &str, mask: u32) -> std::io::Result<()> {
        if !cfg!(feature = "gputrace") {
            return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "needs a build with the gputrace feature (on with developer)"));
        }
        let f = TraceWriter::create(path)?;
        let mut t = self.s.lock().unwrap();
        t.out = Some(f);
        t.start = Some(Instant::now());
        t.re_seq = 0;
        t.seq = 0;
        t.run = None;
        t.line(format_args!("O2 graphics trace started (mask {mask:#x})"));
        drop(t);
        self.mask.store(mask, Ordering::SeqCst);
        Ok(())
    }

    /// Stop the trace, writing out everything. Returns a summary, or None
    /// when no trace was running.
    pub fn stop(&self) -> Option<String> {
        self.mask.store(0, Ordering::SeqCst);
        let mut t = self.s.lock().unwrap();
        t.close_run();
        let f = t.out.take()?;
        let path = f.path.clone();
        let res = f.finish();
        Some(match res {
            Ok(bytes) => format!("{path} (re {} / other {} entries, {bytes} bytes)", t.re_seq, t.seq),
            Err(e) => format!("{path}: write failed: {e}"),
        })
    }

    /// `IRIS_O2_TRACE=<file>`: a full trace from power-on.
    pub fn start_from_env(&self) {
        if let Ok(path) = std::env::var("IRIS_O2_TRACE") {
            if let Err(e) = self.start(&path, TRACE_ALL) {
                eprintln!("IRIS_O2_TRACE {path}: {e}");
            }
        }
    }

    /// `re trace ...`, `gbe trace ...`, `crime trace ...`.
    pub fn command(&self, args: &[&str], w: &mut dyn std::io::Write) -> Result<(), String> {
        let e = |r: std::io::Result<()>| r.map_err(|e| e.to_string());
        match args {
            ["flush"] => {
                self.flush(true);
                if let Some(f) = self.s.lock().unwrap().out.as_mut() {
                    f.sync();
                }
                e(writeln!(w, "trace flushed"))
            }
            [] | ["off"] => e(match self.stop() {
                Some(s) => writeln!(w, "trace closed: {s}"),
                None => writeln!(w, "trace not active"),
            }),
            [path, rest @ ..] => {
                let mask = trace_mask(rest.first().copied().unwrap_or("all"));
                self.start(path, mask).map_err(|e| format!("trace {path}: {e}"))?;
                e(writeln!(w, "tracing to {path} (mask {mask:#x})"))
            }
        }
    }
}

// ── decoding ─────────────────────────────────────────────────────────────────

/// An RE register's name (`off` without the start bit).
pub fn re_name(off: u32) -> String {
    use re_reg::*;
    let tlb = |name: &str, base: u32| format!("{name}[{}]", (off - base) / 8 * 4);
    match off {
        IB_CONTROL => "IB_CONTROL".into(),
        0x1000..=0x11ff => tlb("TLB_A", TLB_A),
        0x1200..=0x13ff => tlb("TLB_B", 0x1200),
        0x1400..=0x15ff => tlb("TLB_C", 0x1400),
        0x1600..=0x16ff => tlb("TLB_TEX", TLB_TEX),
        0x1700..=0x177f => format!("TLB_LINEAR_A[{}]", (off - TLB_LINEAR_A) / 4),
        0x1780..=0x17ff => format!("TLB_LINEAR_B[{}]", (off - 0x1780) / 4),
        BUFMODE_SRC => "BUFMODE_SRC".into(),
        BUFMODE_DST => "BUFMODE_DST".into(),
        CLIP_MODE => "CLIP_MODE".into(),
        DRAWMODE => "DRAWMODE".into(),
        0x2020..=0x2047 => format!("CLIP_RECT{}{}", (off - CLIP_RECT0) / 8, if off & 4 != 0 { ".lo" } else { "" }),
        0x2048 => "SCISSOR".into(),
        0x204c => "SCISSOR.lo".into(),
        WINOFFSET_SRC => "WINOFFSET_SRC".into(),
        WINOFFSET_DST => "WINOFFSET_DST".into(),
        PRIMITIVE => "PRIMITIVE".into(),
        VERTEX_0 => "VERTEX_0".into(),
        VERTEX_1 => "VERTEX_1".into(),
        0x2080..=0x2097 => format!("GL_V{}.{}", (off - TRI_V0) / 8, if off & 4 != 0 { "y" } else { "x" }),
        XFER_SRC => "XFER_SRC".into(),
        0x20a4 => "XFER_SRC_STRIDE".into(),
        XFER_STEP_X => "XFER_STEP_X".into(),
        XFER_STEP_ROW => "XFER_STEP_ROW".into(),
        XFER_DST => "XFER_DST".into(),
        XFER_DST_STRIDE => "XFER_DST_STRIDE".into(),
        STIPPLE_MODE => "STIPPLE_MODE".into(),
        STIPPLE_PATTERN => "STIPPLE_PATTERN".into(),
        FG => "FG".into(),
        BG => "BG".into(),
        0x20e0..=0x20ef => format!("SHADE_{}", ["R", "G", "B", "A"][((off - SHADE_R) / 4) as usize]),
        0x20f0..=0x210f => format!("SHADE_{}", ["DRDX", "DGDX", "DRDY", "DGDY", "DBDX", "DADX", "DBDY", "DADY"][((off - SHADE_GRAD) / 4) as usize]),
        TEX_MODE => "TEX_MODE".into(),
        TEX_FORMAT => "TEX_FORMAT".into(),
        0x211c => "TEX_FORMAT.lo".into(),
        TEX_SQ0 => "TEX_SQ0".into(),
        TEX_TQ0 => "TEX_TQ0".into(),
        TEX_Q0 => "TEX_Q0".into(),
        TEX_COORD_SCALE => "TEX_COORD_SCALE".into(),
        TEX_DSQDX => "TEX_DSQDX".into(),
        TEX_DSQDY => "TEX_DSQDY".into(),
        TEX_DTQDX => "TEX_DTQDX".into(),
        TEX_DTQDY => "TEX_DTQDY".into(),
        TEX_DQDX => "TEX_DQDX".into(),
        TEX_DQDY => "TEX_DQDY".into(),
        0x2160 => "TEX_BORDER".into(),
        0x2168 => "TEX_ENVCOLOR".into(),
        0x2170 => "FOG_COLOR".into(),
        ALPHA_TEST => "ALPHA_TEST".into(),
        BLEND_CONST => "BLEND_CONST".into(),
        BLEND_FUNC => "BLEND_FUNC".into(),
        LOGIC_OP => "LOGIC_OP".into(),
        COLOR_MASK => "COLOR_MASK".into(),
        DEPTH_FUNC => "DEPTH_FUNC".into(),
        DEPTH_Z0 => "DEPTH_Z0".into(),
        DEPTH_DZDX => "DEPTH_DZDX".into(),
        DEPTH_DZDY => "DEPTH_DZDY".into(),
        STENCIL_MODE => "STENCIL_MODE".into(),
        STENCIL_MASK => "STENCIL_MASK".into(),
        0x21f0 => "PP_NULL".into(),
        0x21f8 => "PP_FLUSH".into(),
        MTE_MODE => "MTE_MODE".into(),
        MTE_BYTEMASK => "MTE_BYTEMASK".into(),
        MTE_STIPPLE => "MTE_STIPPLE".into(),
        MTE_FG => "MTE_FG".into(),
        MTE_SRC0 => "MTE_SRC0".into(),
        0x3028 => "MTE_SRC1".into(),
        MTE_DST0 => "MTE_DST0".into(),
        MTE_DST1 => "MTE_DST1".into(),
        MTE_SRC_STEP => "MTE_SRC_STEP".into(),
        MTE_DST_STEP => "MTE_DST_STEP".into(),
        0x4000 | 0x4004 => "STATUS".into(),
        0x4008 => "IB_DISCARD".into(),
        _ => format!("[{off:#06x}]"),
    }
}

/// What a register value means, for the registers worth spelling out.
fn re_note(off: u32, v: u32) -> String {
    use re_reg::*;
    match off {
        DRAWMODE => decode_drawmode(v),
        BUFMODE_SRC | BUFMODE_DST => decode_bufmode(v),
        VERTEX_0 | VERTEX_1 | WINOFFSET_SRC | WINOFFSET_DST => format!("({}, {})", (v >> 16) as u16 as i16, v as u16 as i16),
        0x2080..=0x2097 => format!("{:.3}", gl_coord(v)),
        0x20e0..=0x20ef => format!("{:.3}", v as i32 as f64 / 4096.0),
        BLEND_FUNC => format!("op {} src {} dst {}", (v >> 8) & 15, (v >> 4) & 15, v & 15),
        ALPHA_TEST => format!("func {} ref {}", (v >> 8) & 15, v & 255),
        LOGIC_OP => format!("op {}", v & 15),
        _ => String::new(),
    }
}

pub fn decode_drawmode(v: u32) -> String {
    const BITS: &[(u32, &str)] = &[
        (22, "GL"), (21, "XFER"), (20, "SCISSOR"), (19, "LINE_STIPPLE"), (18, "POLY_STIPPLE"), (17, "OPAQUE_STIPPLE"),
        (16, "SHADE"), (15, "TEXTURE"), (14, "FOG"), (13, "COVERAGE"), (12, "AA_LINE"), (11, "ALPHA_TEST"), (10, "BLEND"),
        (9, "LOGIC_OP"), (8, "DITHER"), (7, "COLORMASK"), (2, "DEPTH_TEST"), (1, "DEPTH_MASK"), (0, "STENCIL"),
    ];
    let mut s: Vec<String> = BITS.iter().filter(|(b, _)| v & (1 << b) != 0).map(|(_, n)| n.to_string()).collect();
    s.push(format!("LANES={:x}", (v >> 3) & 15));
    s.join(" ")
}

pub fn decode_bufmode(v: u32) -> String {
    let buf = match (v >> 10) & 7 {
        0 => "tlbA",
        1 => "tlbB",
        2 => "tlbC",
        4 => "linA",
        5 => "linB",
        6 => "cid",
        _ => "?",
    };
    let bytes = 1 << ((v >> 8) & 3).min(2);
    let ty = match (v >> 4) & 15 {
        0 => "CI",
        1 => "RGB",
        2 => "RGBA",
        3 => "ABGR",
        _ => "type?",
    };
    let size = match (v >> 2) & 3 {
        0 => "8",
        1 => "16",
        2 => "32",
        _ => "?",
    };
    let dp = if v & 2 != 0 { if v & 1 != 0 { " doublepix hi" } else { " doublepix lo" } } else { "" };
    format!("{buf} {bytes}B {ty} {size}{dp}")
}

/// A primitive as the pixel pipe will run it, from the registers it draws with.
pub fn decode_prim(r: &[u32]) -> String {
    use re_reg::*;
    let g = |off: u32| r[(off / 4) as usize];
    let dm = g(DRAWMODE);
    let prim = g(PRIMITIVE);
    let op = (prim >> 24) & 3;
    let mut s = format!("{} edge {}", ["POINT", "LINE", "TRI", "RECT"][op as usize], (prim >> 16) & 3);
    if op == 1 && prim & (1 << 18) != 0 {
        s += " skiplast";
    }
    let off = |o: u32| ((g(o) >> 16) as u16 as i16, g(o) as u16 as i16);
    if dm & (1 << 22) != 0 {
        let n = if op == 2 { 3 } else if op == 0 { 1 } else { 2 };
        for i in 0..n {
            s += &format!(" ({:.3}, {:.3})", gl_coord(g(TRI_V0 + 8 * i)), gl_coord(g(TRI_V0 + 8 * i + 4)));
        }
    } else {
        s += &format!(" ({}, {})", (g(VERTEX_0) >> 16) as i16, g(VERTEX_0) as u16 as i16);
        if op != 0 {
            s += &format!(" ({}, {})", (g(VERTEX_1) >> 16) as i16, g(VERTEX_1) as u16 as i16);
        }
    }
    let (ox, oy) = off(WINOFFSET_DST);
    s += &format!(" off ({ox}, {oy}) [{}] dst [{}] clip {:#x}", decode_drawmode(dm), decode_bufmode(g(BUFMODE_DST)), g(CLIP_MODE));
    if dm & (1 << 21) != 0 {
        let (sx, sy) = off(XFER_SRC);
        s += &format!(" src [{}] from {:#x} ({sx}, {sy}) steps {} {} dst {:#x} stride {}", decode_bufmode(g(BUFMODE_SRC)), g(XFER_SRC), g(XFER_STEP_X) as i32, g(XFER_STEP_ROW) as i32, g(XFER_DST), g(XFER_DST_STRIDE) as i32);
    }
    if dm & (1 << 16) != 0 {
        let c = |i: u32| g(SHADE_R + 4 * i) as i32 as f64 / 4096.0;
        s += &format!(" shade {:.3} {:.3} {:.3} {:.3}", c(0), c(1), c(2), c(3));
        if (0..8).any(|i| g(SHADE_GRAD + 4 * i) != 0) {
            let d = |i: u32| g(SHADE_GRAD + 4 * i) as i32;
            s += &format!(" grad {:x} {:x} {:x} {:x} / {:x} {:x} {:x} {:x}", d(0), d(1), d(4), d(5), d(2), d(3), d(6), d(7));
        }
    } else {
        s += &format!(" fg {:#010x}", g(FG));
    }
    if dm & (1 << 17) != 0 {
        s += &format!(" bg {:#010x}", g(BG));
    }
    if dm & ((1 << 19) | (1 << 18)) != 0 {
        s += &format!(" stipple {:#010x}/{:#x}", g(STIPPLE_PATTERN), g(STIPPLE_MODE));
    }
    if dm & (1 << 15) != 0 {
        s += &format!(" tex mode {:#x} fmt {:#x} sq {:#x} tq {:#x} q {:#x}", g(TEX_MODE), g(TEX_FORMAT), g(TEX_SQ0 + 4), g(TEX_TQ0 + 4), g(TEX_Q0));
    }
    if dm & (1 << 10) != 0 {
        s += &format!(" blend {:#x}", g(BLEND_FUNC));
    }
    if dm & (1 << 11) != 0 {
        s += &format!(" atest {:#x}", g(ALPHA_TEST));
    }
    if dm & (1 << 9) != 0 {
        s += &format!(" logic {}", g(LOGIC_OP) & 15);
    }
    if dm & (1 << 7) != 0 {
        s += &format!(" mask {:#x}", g(COLOR_MASK));
    }
    if dm & 5 != 0 {
        s += &format!(" zfunc {:#x} stencil {:#x}", g(DEPTH_FUNC), g(STENCIL_MODE));
    }
    s
}

/// A memory transfer as the MTE will run it.
pub fn decode_mte(r: &[u32]) -> String {
    use re_reg::*;
    let g = |off: u32| r[(off / 4) as usize];
    let mode = g(MTE_MODE);
    let buf = |b: u32| ["tlbA", "tlbB", "tlbC", "?", "linA", "linB", "cid", "?"][(b & 7) as usize];
    let mut s = format!("MTE {} dst {} {:#x}..{:#x}", if mode & (1 << 11) != 0 { "COPY" } else { "CLEAR" }, buf(mode >> 2), g(MTE_DST0), g(MTE_DST1));
    if mode & (1 << 11) != 0 {
        s += &format!(" src {} {:#x} sstep {}", buf(mode >> 5), g(MTE_SRC0), g(MTE_SRC_STEP) as i32);
    } else {
        s += &format!(" fg {:#010x}", g(MTE_FG));
    }
    s += &format!(" dstep {} bytemask {:#010x} mode {mode:#x}", g(MTE_DST_STEP) as i32, g(MTE_BYTEMASK));
    if mode & (1 << 10) != 0 {
        s += &format!(" stipple {:#010x}", g(MTE_STIPPLE));
    }
    s
}

fn gbe_table(off: u32) -> Option<(&'static str, u32)> {
    use gbe_reg::*;
    match off {
        0x48000..=0x4807f => Some(("MODE", MODE)),
        0x50000..=0x57fff => Some(("CMAP", CMAP)),
        0x60000..=0x603ff => Some(("GMAP", GMAP)),
        0x78000..=0x780ff => Some(("CRS_GLYPH", CRS_GLYPH)),
        0x70008..=0x70013 => Some(("CRS_CMAP", CRS_CMAP1)),
        _ => None,
    }
}

/// Whether writes through `off` merge into one range line.
pub fn gbe_is_table(off: u32) -> bool {
    gbe_table(off).is_some()
}

fn gbe_name(first: u32, last: u32) -> String {
    use gbe_reg::*;
    if let Some((name, base)) = gbe_table(first) {
        let (a, b) = ((first - base) / 4, (last - base) / 4);
        return if a == b { format!("{name}[{a}]") } else { format!("{name}[{a}..{b}]") };
    }
    let name = match first {
        CTRLSTAT => "CTRLSTAT",
        DOTCLOCK => "DOTCLOCK",
        I2C => "I2C",
        I2CFP => "I2CFP",
        DEVICE_ID => "DEVICE_ID",
        VT_XY => "VT_XY",
        VT_XYMAX => "VT_XYMAX",
        VT_FLAGS => "VT_FLAGS",
        VT_INTR01 => "VT_INTR01",
        VT_INTR23 => "VT_INTR23",
        VT_HCMAP => "VT_HCMAP",
        VT_VCMAP => "VT_VCMAP",
        OVR_WIDTH_TILE => "OVR_WIDTH_TILE",
        OVR_CONTROL => "OVR_CONTROL",
        OVR_CONTROL_SHADOW => "OVR_CONTROL_SHADOW",
        FRM_SIZE_TILE => "FRM_SIZE_TILE",
        FRM_SIZE_PIXEL => "FRM_SIZE_PIXEL",
        FRM_CONTROL => "FRM_CONTROL",
        FRM_CONTROL_SHADOW => "FRM_CONTROL_SHADOW",
        DID_CONTROL => "DID_CONTROL",
        DID_CONTROL_SHADOW => "DID_CONTROL_SHADOW",
        CM_FIFO => "CM_FIFO",
        CRS_POS => "CRS_POS",
        CRS_CTL => "CRS_CTL",
        _ => return format!("[{first:#07x}]"),
    };
    name.to_string()
}

/// Whether a read of `off` is a poll: repeats collapse whatever they read.
pub fn gbe_is_poll(off: u32) -> bool {
    off == gbe_reg::VT_XY || off == gbe_reg::CTRLSTAT || off == gbe_reg::CM_FIFO
}

/// The name of a live or table GBE register, for an event line.
pub fn gbe_reg_name(off: u32) -> String {
    gbe_name(off, off)
}

fn gbe_note(off: u32, v: u32) -> String {
    use gbe_reg::*;
    let body = match off {
        FRM_CONTROL | FRM_CONTROL_SHADOW | OVR_CONTROL | OVR_CONTROL_SHADOW => {
            format!("list {:#x} {}", v & 0xffff_ffe0, if v & 1 != 0 { "dma" } else { "off" })
        }
        DID_CONTROL | DID_CONTROL_SHADOW => format!("table {:#x}0000 {}", v & 0xffff, if v & (1 << 16) != 0 { "dma" } else { "off" }),
        FRM_SIZE_TILE => format!("{} tiles + {} x 32 bytes, {}B", (v >> 5) & 0xff, v & 0x1f, 1 << ((v >> 13) & 3).min(2)),
        FRM_SIZE_PIXEL => format!("height {}", v >> 16),
        CRS_POS => format!("({}, {})", ((v & 0xfff) as i32) << 20 >> 20, ((v >> 16 & 0xfff) as i32) << 20 >> 20),
        0x48000..=0x4807f => format!("buf {} type {} cmap {} gamma {}", v & 3, (v >> 2) & 7, (v >> 5) & 0x1f, if v & (1 << 10) == 0 { "on" } else { "off" }),
        _ => return String::new(),
    };
    format!("  ; {body}")
}

fn crime_name(off: u32) -> String {
    use crate::dev::crime::crime_reg::*;
    match off {
        REVISION => "REVISION".into(),
        CONTROL => "CONTROL".into(),
        INTSTAT => "INTSTAT".into(),
        INTMASK => "INTMASK".into(),
        SOFTINT => "SOFTINT".into(),
        HARDINT => "HARDINT".into(),
        WATCHDOG => "WATCHDOG".into(),
        TIME => "TIME".into(),
        CPU_ERROR_ADDR => "CPU_ERROR_ADDR".into(),
        CPU_ERROR_STAT => "CPU_ERROR_STAT".into(),
        CPU_ERROR_ENA => "CPU_ERROR_ENA".into(),
        _ => format!("[{off:#06x}]"),
    }
}
