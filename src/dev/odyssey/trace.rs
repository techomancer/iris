//! The VPro annotated trace: `vpro trace <file> [fifo,draw,cpu,dma|all]`,
//! `vpro trace flush|off`, or `IRIS_ODSY_TRACE=<file>` from power-on.
//!
//! Lines look like
//!   `    3.214711 FIFO #000412 DEST_SELECT 0x1017 [4] 00000000 00000400 00000000 00000000`
//!   `    3.214712 FIFO #000413 VERTEX_3F 0x8080cc03 [3] 3f800000 40000000 00000000  (1, 2, 0)`
//!   `    3.214720 DRAW #000208 draw TRIANGLES n=3 ctx=1 queued cost 96`
//!   `    3.214801 CPU  #000031 rd 0x1064 status0 [4] = 0x10010000  (x218)`
//!   `    3.230004 DMA  #000002 upload table 0xbf0a000 dest 0x1234 origin 0x0 width 64 size 0x40 fmt 0x1 skip 0 rowbytes 256 ctl 0x80000000`
//! with the time in seconds since the trace started. FIFO lines are command
//! words as the parser completes them (the tag, its name, the operands as
//! words and, where the word carries floats, as floats), plus the parser's
//! other completions (copy rect, pattern load, image box and burst, matrix
//! load, rect, query, the words it does not know). DRAW lines are the
//! primitives assembly sends to the drawing pipeline and the clears. CPU
//! lines are every bus access that is not a command store, runs of one
//! access repeated collapsed into one line. DMA lines are the page-table
//! descriptors as the board runs them.
//!
//! FIFO, DRAW and DMA lines are written by the board's thread, CPU lines by the
//! processors' threads, so tracing slows both. The sink is a leaf lock:
//! nothing is taken while it is held.
//!
//! All of it needs the `gputrace` feature (on with `developer`); without it
//! `Trace::on` is a constant false and every hook compiles away.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use parking_lot::Mutex;

use super::board::Board;
use super::consts::{
    backend, cfifo, cmd, ctx, dma, gl_cmd, host, i2c, intr, mem, mode, nic, prog, query, reg, route,
    stats, swap, tex, ODSY_BUZZ_1, ODSY_BUZZ_2, ODSY_BUZZ_3, ODSY_BUZZ_4, ODSY_BUZZ_5,
    ODSY_BUZZ_6, ODSY_CONTROL, ODSY_DISPLAYED_BUFFER, ODSY_ERROR_ADDR_LOWER,
    ODSY_ERROR_ADDR_UPPER, ODSY_ERROR_COMMAND, ODSY_IDENTITY, ODSY_INT_DEST_LOWER,
    ODSY_INT_DEST_UPPER, ODSY_RESET_CONTROL, ODSY_STATUS, ODSY_STATUS0, ODSY_SW_TOKEN,
    ODSY_USER_CMD_FIFO, ODSY_XT_REQ_TIMEOUT, ODSY_BLOCK_END,
};
use super::debug::begin_name;
use super::assembly::Prim;
use super::draw::{RasterOp, RASTER_CLEAR};

pub const TRACE_FIFO: u32 = 1;
pub const TRACE_DRAW: u32 = 2;
pub const TRACE_CPU: u32 = 4;
pub const TRACE_DMA: u32 = 8;
pub const TRACE_ALL: u32 = TRACE_FIFO | TRACE_DRAW | TRACE_CPU | TRACE_DMA;

const CLASSES: [(&str, u32); 4] = [("fifo", TRACE_FIFO), ("draw", TRACE_DRAW), ("cpu", TRACE_CPU), ("dma", TRACE_DMA)];

/// `all`, or a comma list of class names.
pub(super) fn trace_mask(spec: &str) -> Result<u32, String> {
    if spec == "all" {
        return Ok(TRACE_ALL);
    }
    spec.split(',').try_fold(0, |m, c| match CLASSES.iter().find(|(n, _)| *n == c) {
        Some((_, bit)) => Ok(m | bit),
        None => Err(format!("unknown trace class '{c}' (fifo, gl, cpu, dma, all)")),
    })
}

fn mask_names(mask: u32) -> String {
    if mask == TRACE_ALL {
        return "all".into();
    }
    CLASSES.iter().filter(|(_, b)| mask & b != 0).map(|(n, _)| *n).collect::<Vec<_>>().join(",")
}

/// A run of one CPU access repeated, collapsed into one line.
struct CpuRun {
    write: bool,
    off: u64,
    size: u32,
    val: u64,
    count: u64,
}

/// The sink, behind `Trace::sink`.
pub(super) struct TraceSink {
    out: Option<BufWriter<File>>,
    path: String,
    start: Instant,
    /// Lines per class, in `CLASSES` order.
    seq: [u64; 4],
    run: Option<CpuRun>,
    last_flush: Instant,
}

impl TraceSink {
    fn line(&mut self, class: usize, text: std::fmt::Arguments) {
        self.seq[class] += 1;
        let (seq, t) = (self.seq[class], self.start.elapsed().as_secs_f64());
        if let Some(f) = self.out.as_mut() {
            let _ = writeln!(f, "{t:12.6} {:<4} #{seq:06} {text}", CLASSES[class].0.to_ascii_uppercase());
        }
    }

    /// Emit the pending collapsed CPU run, if any.
    fn close_run(&mut self) {
        let Some(r) = self.run.take() else { return };
        let dir = if r.write { "wr" } else { "rd" };
        let name = offset_name(r.off).map(|n| format!(" {n}")).unwrap_or_default();
        let times = if r.count > 1 { format!("  (x{})", r.count) } else { String::new() };
        self.line(2, format_args!("{dir} {:#x}{name} [{}] = {:#x}{times}", r.off, r.size, r.val));
    }
}

/// The trace: its mask, read on every hook, and the sink.
pub(super) struct Trace {
    mask: AtomicU32,
    sink: Mutex<TraceSink>,
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

impl Trace {
    pub fn new() -> Self {
        let now = Instant::now();
        Trace {
            mask: AtomicU32::new(0),
            sink: Mutex::new(TraceSink { out: None, path: String::new(), start: now, seq: [0; 4], run: None, last_flush: now }),
        }
    }

    /// Whether a trace of `what` runs. A constant false without `gputrace`,
    /// so every `if trace.on(..)` on a bus or engine path compiles away,
    /// formatting and all.
    #[inline(always)]
    pub fn on(&self, what: u32) -> bool {
        cfg!(feature = "gputrace") && self.mask.load(Ordering::Relaxed) & what != 0
    }

    /// Start tracing the classes in `mask` into `path`.
    pub fn start(&self, path: &str, mask: u32) -> std::io::Result<()> {
        if !cfg!(feature = "gputrace") {
            return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "needs a build with the gputrace feature (on with developer)"));
        }
        let f = File::create(path)?;
        let mut t = self.sink.lock();
        t.out = Some(BufWriter::with_capacity(1 << 20, f));
        t.path = path.to_string();
        t.start = Instant::now();
        t.last_flush = t.start;
        t.seq = [0; 4];
        t.run = None;
        if let Some(f) = t.out.as_mut() {
            let _ = writeln!(f, "# VPro trace ({})", mask_names(mask));
        }
        drop(t);
        self.mask.store(mask, Ordering::SeqCst);
        Ok(())
    }

    /// Stop and close the file; what was written, or None when no trace ran.
    pub fn stop(&self) -> Option<String> {
        self.mask.store(0, Ordering::SeqCst);
        let mut t = self.sink.lock();
        t.close_run();
        let mut f = t.out.take()?;
        let _ = f.flush();
        Some(format!("{} ({})", t.path, Self::counts(&t)))
    }

    fn counts(t: &TraceSink) -> String {
        CLASSES.iter().zip(t.seq).map(|((n, _), c)| format!("{n} {c}")).collect::<Vec<_>>().join(" / ")
    }

    /// "tracing to <path> (<classes>): <counts>", or None.
    pub fn status(&self) -> Option<String> {
        let mask = self.mask.load(Ordering::Relaxed);
        let t = self.sink.lock();
        t.out.as_ref()?;
        Some(format!("tracing to {} ({}): {}", t.path, mask_names(mask), Self::counts(&t)))
    }

    /// Write the buffered lines out: at most twice a second unless forced.
    /// Called every display frame.
    pub fn flush(&self, force: bool) {
        if self.mask.load(Ordering::Relaxed) == 0 && !force {
            return;
        }
        let mut t = self.sink.lock();
        if force || t.last_flush.elapsed().as_millis() >= 500 {
            t.close_run();
            if let Some(f) = t.out.as_mut() {
                let _ = f.flush();
            }
            t.last_flush = Instant::now();
        }
    }

    fn line(&self, class: usize, text: std::fmt::Arguments) {
        let mut t = self.sink.lock();
        t.close_run();
        t.line(class, text);
    }

    pub fn fifo(&self, text: std::fmt::Arguments) {
        self.line(0, text);
    }

    pub fn draw(&self, text: std::fmt::Arguments) {
        self.line(1, text);
    }

    pub fn dma(&self, text: std::fmt::Arguments) {
        self.line(3, text);
    }

    /// One bus access that is not a command store; repeats of one access
    /// collapse into a run.
    pub fn cpu(&self, write: bool, off: u64, size: u32, val: u64) {
        let mut t = self.sink.lock();
        if let Some(r) = t.run.as_mut() {
            if r.write == write && r.off == off && r.size == size && r.val == val {
                r.count += 1;
                return;
            }
        }
        t.close_run();
        t.run = Some(CpuRun { write, off, size, val, count: 1 });
    }
}

// ── names ────────────────────────────────────────────────────────────────────

/// The name of a modelled aperture offset, for CPU lines. A configuration
/// block register answers at its base and at base+4 (`registers::reg`), so
/// both spellings get its name.
pub(super) fn offset_name(off: u64) -> Option<&'static str> {
    let off = if off < ODSY_BLOCK_END { off & !7 } else { off };
    Some(match off {
        ODSY_IDENTITY => "identity",
        ODSY_STATUS => "status",
        ODSY_ERROR_ADDR_UPPER => "error addr upper",
        ODSY_ERROR_ADDR_LOWER => "error addr lower",
        ODSY_CONTROL => "control",
        ODSY_XT_REQ_TIMEOUT => "xt req timeout",
        ODSY_INT_DEST_UPPER => "int dest upper",
        ODSY_INT_DEST_LOWER => "int dest lower",
        ODSY_ERROR_COMMAND => "error command",
        mem::CONTROL => "mem control",
        mem::CONFIG => "mem config",
        ODSY_RESET_CONTROL => "reset control",
        intr::FLAGS => "intr flags",
        intr::CLEAR => "intr clear",
        intr::ENABLE_SET => "intr enable set",
        intr::ENABLE_CLEAR => "intr enable clear",
        route::DEST_UPPER => "route dest upper",
        route::DEST_LOWER => "route dest lower",
        route::VECTORS => "route vectors",
        ODSY_STATUS0 => "status0",
        ODSY_SW_TOKEN => "sw token",
        nic::PROM => "nic (prom)",
        nic::KERNEL => "nic (kernel)",
        host::WRITE_ADDR_UPPER => "host write addr upper",
        host::WRITE_ADDR_LOWER => "host write addr lower",
        stats::ADDR_UPPER => "stats addr upper",
        stats::ADDR_LOWER => "stats addr lower",
        stats::CONTROL => "stats control",
        ctx::SHADOW_ADDR_UPPER => "ctx shadow addr upper",
        ctx::SHADOW_ADDR_LOWER => "ctx shadow addr lower",
        ctx::SWITCH_PORT => "ctx switch port",
        ODSY_BUZZ_1 => "buzz1",
        ODSY_BUZZ_2 => "buzz2",
        ODSY_BUZZ_3 => "buzz3",
        ODSY_BUZZ_4 => "buzz4",
        ODSY_BUZZ_5 => "buzz5",
        ODSY_BUZZ_6 => "buzz6",
        swap::ARM => "swap arm",
        swap::PENDING => "swap pending",
        swap::CONTROL => "swap control",
        swap::CONTROL_COMPANION => "swap control companion",
        ODSY_DISPLAYED_BUFFER => "displayed buffer",
        cfifo::SELFTEST => "cfifo selftest",
        backend::SELECT => "backend select",
        backend::VERSION => "backend version",
        i2c::MAIN_CMD_READ => "i2c main cmd",
        i2c::MAIN_STATUS_READ => "i2c main status",
        i2c::OPT_CMD_READ => "i2c opt cmd",
        i2c::OPT_STATUS_READ => "i2c opt status",
        cmd::FIFO => "cmd fifo",
        ODSY_USER_CMD_FIFO => "user cmd fifo",
        o if (backend::READ_BASE..backend::READ_BASE + 0x10000).contains(&o) => "backend read window",
        _ => return None,
    })
}

/// The name of a completed command word's tag: a register-select's
/// register, a GL-class word, or one of the parser's own words.
pub(super) fn tag_name(tag: u32) -> Option<&'static str> {
    Some(match tag {
        reg::DRAW_MASK => "DRAW_MASK",
        0x6b0 => "FIFO_SETUP_6B0",
        0x6b1 => "FIFO_SETUP_6B1",
        reg::FIFO_SETUP_6B2 => "FIFO_SETUP_6B2",
        0x6b3 => "FIFO_SETUP_6B3",
        reg::FIFO_SETUP_7B5 => "FIFO_SETUP_7B5",
        reg::SETUP_7F0 => "SETUP_7F0",
        reg::QUICK_FLUSH_ARG => "QUICK_FLUSH_ARG",
        reg::STROBE => "STROBE",
        reg::FULL_FLUSH_EXTRA => "FULL_FLUSH_EXTRA",
        reg::GL_BEGIN => "GL_BEGIN",
        reg::GL_END => "GL_END",
        reg::PIXEL_MODE_EXT => "PIXEL_MODE_EXT",
        reg::BLIT_PREFIX => "BLIT_PREFIX",
        reg::HOST_WRITE => "HOST_WRITE",
        reg::READ_ORIGIN => "READ_ORIGIN",
        swap::SYNC_A => "DRAWABLE_0608/SYNC_A",
        swap::SYNC_B => "DRAWABLE_0609/SYNC_B",
        0x60a | 0x60c..=0x60f => "DRAWABLE",
        reg::DEPTH_FUNC => "DEPTH_FUNC",
        reg::WRITE_MASK => "WRITE_MASK",
        reg::TEX_MODE => "TEX_MODE",
        reg::TEX_SAMPLER => "TEX_SAMPLER",
        reg::TEX_ENV => "TEX_ENV",
        t if (reg::TEX_LEVEL..reg::TEX_LEVEL + 0x40).contains(&t) => "TEX_LEVEL",
        reg::TEX_LEVEL0 => "TEX_LEVEL0",
        reg::TEX_LOD => "TEX_LOD",
        reg::TEX_SCALE_BIAS => "TEX_SCALE_BIAS",
        reg::POST_FILTER_DEFAULT => "POST_FILTER_DEFAULT",
        t if (reg::TEX_COLOUR_TABLE0..reg::TEX_COLOUR_TABLE0 + 0x10).contains(&t) => "TEX_COLOUR_TABLE0",
        t if (reg::TEX_COLOUR_TABLE1..reg::TEX_COLOUR_TABLE1 + 0x10).contains(&t) => "TEX_COLOUR_TABLE1",
        t if (reg::TEX_TABLE3..reg::TEX_TABLE3 + 0x10).contains(&t) => "TEX_TABLE3",
        t if (reg::TEX_DETAIL_FUNCTION..reg::TEX_DETAIL_FUNCTION + tex::DETAIL_FUNCTION_BLOCKS).contains(&t) => "TEX_DETAIL_FUNCTION",
        reg::TEX_OFFSET => "TEX_OFFSET",
        reg::FOG => "FOG",
        reg::PATTERN_LOW => "PATTERN_LOW",
        reg::PATTERN_HIGH => "PATTERN_HIGH",
        reg::PATTERN_SHAPE => "PATTERN_SHAPE",
        reg::PATTERN_MODE => "PATTERN_MODE",
        reg::LINE_STIPPLE => "LINE_STIPPLE",
        reg::DEST_SELECT => "DEST_SELECT",
        reg::PLANE_MASK => "PLANE_MASK",
        reg::ALPHA_FUNC => "ALPHA_FUNC",
        reg::BLEND => "BLEND",
        reg::LOGIC_OP => "LOGIC_OP",
        reg::STENCIL_FUNC => "STENCIL_FUNC",
        reg::POLYGON_OFFSET => "POLYGON_OFFSET",
        reg::POLYGON_MODE => "POLYGON_MODE",
        reg::LINE_POINT_SIZE => "LINE_POINT_SIZE",
        reg::CLIP_BOX => "CLIP_BOX",
        reg::WINDOW_ORIGIN => "WINDOW_ORIGIN",
        reg::ID_COMPARE => "ID_COMPARE",
        reg::WINDOW_CLIP => "WINDOW_CLIP",
        0x4c0 => "QUERY",
        t if t == 0x4c0 + query::KIND_RASTER => "QUERY_RASTER",
        0x740 => "WORD_740",
        0x741 => "WORD_741",
        0x2f00c02e | 0x3000c02e => "SYNC",
        cmd::COPY_RECT_TRIGGER => "COPY_RECT",
        cmd::IMAGE_BOX => "IMAGE_BOX",
        cmd::IMAGE_HEADER => "IMAGE_HEADER",
        cmd::IMAGE_HEADER_TAIL => "IMAGE_HEADER_TAIL",
        cmd::WINDOW_RESTORE => "WINDOW_RESTORE",
        dma::DESC_UPLOAD => "DMA_DESC_UPLOAD",
        dma::DESC_READBACK => "DMA_DESC_READBACK",
        prog::RUN_ONCE => "PROG_RUN_ONCE",
        gl_cmd::COLOR_3UB => "COLOR_3UB",
        mode::INDEX_ZEROS => "INDEX_ZEROS",
        gl_cmd::VERTEX_2I => "VERTEX_2I",
        gl_cmd::COLOR_INDEX => "COLOR_INDEX",
        gl_cmd::COLOR_INDEX_F => "COLOR_INDEX_F",
        gl_cmd::REPLAY_INDEX => "REPLAY_INDEX",
        gl_cmd::REPLAY_INDEX_F => "REPLAY_INDEX_F",
        gl_cmd::INDEX_MODE_COLOR_3F => "INDEX_MODE_COLOR_3F",
        gl_cmd::INDEX_MODE_COLOR_4F => "INDEX_MODE_COLOR_4F",
        gl_cmd::INDEX_MODE_COLOR_3I => "INDEX_MODE_COLOR_3I",
        gl_cmd::COLOR_3I => "COLOR_3I",
        gl_cmd::COLOR_4F => "COLOR_4F",
        gl_cmd::COLOR_3US => "COLOR_3US",
        gl_cmd::COLOR_3UI => "COLOR_3UI",
        gl_cmd::COLOR_3B => "COLOR_3B",
        gl_cmd::COLOR_3S => "COLOR_3S",
        gl_cmd::COLOR_4UB => "COLOR_4UB",
        gl_cmd::COLOR_4US => "COLOR_4US",
        gl_cmd::COLOR_4UI => "COLOR_4UI",
        gl_cmd::COLOR_4B => "COLOR_4B",
        gl_cmd::COLOR_4S => "COLOR_4S",
        gl_cmd::COLOR_4I => "COLOR_4I",
        gl_cmd::NORMAL_3B => "NORMAL_3B",
        gl_cmd::NORMAL_3S => "NORMAL_3S",
        gl_cmd::NORMAL_3I => "NORMAL_3I",
        gl_cmd::COLOR_3F => "COLOR_3F",
        gl_cmd::VERTEX_2F => "VERTEX_2F",
        gl_cmd::VERTEX_3F => "VERTEX_3F",
        gl_cmd::VERTEX_3I => "VERTEX_3I",
        gl_cmd::VERTEX_4F => "VERTEX_4F",
        gl_cmd::NORMAL_3F => "NORMAL_3F",
        gl_cmd::MATERIAL_AMBIENT => "MATERIAL_AMBIENT",
        gl_cmd::MATERIAL_DIFFUSE => "MATERIAL_DIFFUSE",
        gl_cmd::MATERIAL_SPECULAR => "MATERIAL_SPECULAR",
        gl_cmd::MATERIAL_EMISSION => "MATERIAL_EMISSION",
        gl_cmd::BACK_AMBIENT => "BACK_AMBIENT",
        gl_cmd::BACK_DIFFUSE => "BACK_DIFFUSE",
        gl_cmd::BACK_SPECULAR => "BACK_SPECULAR",
        gl_cmd::BACK_EMISSION => "BACK_EMISSION",
        gl_cmd::SHININESS_SQRT => "SHININESS_SQRT",
        gl_cmd::BACK_SHININESS_SQRT => "BACK_SHININESS_SQRT",
        gl_cmd::SHININESS_DERIVED => "SHININESS_DERIVED",
        gl_cmd::LIGHT_COLOUR_LOAD => "LIGHT_COLOUR_LOAD",
        gl_cmd::LIGHT_VEC_LOAD => "LIGHT_VEC_LOAD",
        gl_cmd::CLIP_PLANE_LOAD => "CLIP_PLANE_LOAD",
        gl_cmd::TEXCOORD_1F => "TEXCOORD_1F",
        gl_cmd::TEXCOORD_2F => "TEXCOORD_2F",
        gl_cmd::TEXCOORD_3F => "TEXCOORD_3F",
        gl_cmd::TEXCOORD_4F => "TEXCOORD_4F",
        gl_cmd::TEXCOORD_1I => "TEXCOORD_1I",
        gl_cmd::TEXCOORD_2I => "TEXCOORD_2I",
        gl_cmd::TEXCOORD_3I => "TEXCOORD_3I",
        gl_cmd::TEXCOORD_4I => "TEXCOORD_4I",
        gl_cmd::TEX_RECT => "TEX_RECT",
        gl_cmd::MATRIX_LOAD => "MATRIX_LOAD",
        gl_cmd::LOAD_4F => "LOAD_4F",
        gl_cmd::RECT => "RECT",
        gl_cmd::CLEAR_BOX => "CLEAR_BOX",
        t if t & gl_cmd::MATRIX_SELECT_SHAPE == gl_cmd::MATRIX_SELECT_TAG => "MATRIX_SELECT",
        t if t & cmd::STATE_WORD_MASK == cmd::STATE_WORD => "STATE_WORD",
        _ => return None,
    })
}

/// Whether a word's operands are IEEE floats.
fn tag_floats(tag: u32) -> bool {
    matches!(
        tag,
        gl_cmd::COLOR_3F
            | gl_cmd::COLOR_4F
            | gl_cmd::COLOR_INDEX_F
            | gl_cmd::REPLAY_INDEX_F
            | gl_cmd::INDEX_MODE_COLOR_3F
            | gl_cmd::INDEX_MODE_COLOR_4F
            | gl_cmd::VERTEX_2F
            | gl_cmd::VERTEX_3F
            | gl_cmd::VERTEX_4F
            | gl_cmd::NORMAL_3F
            | gl_cmd::MATERIAL_AMBIENT
            | gl_cmd::MATERIAL_DIFFUSE
            | gl_cmd::MATERIAL_SPECULAR
            | gl_cmd::MATERIAL_EMISSION
            | gl_cmd::BACK_AMBIENT
            | gl_cmd::BACK_DIFFUSE
            | gl_cmd::BACK_SPECULAR
            | gl_cmd::BACK_EMISSION
            | gl_cmd::SHININESS_SQRT
            | gl_cmd::BACK_SHININESS_SQRT
            | gl_cmd::SHININESS_DERIVED
            | gl_cmd::LIGHT_COLOUR_LOAD
            | gl_cmd::LIGHT_VEC_LOAD
            | gl_cmd::CLIP_PLANE_LOAD
            | gl_cmd::TEXCOORD_1F
            | gl_cmd::TEXCOORD_2F
            | gl_cmd::TEXCOORD_3F
            | gl_cmd::TEXCOORD_4F
            | gl_cmd::TEX_RECT
            | gl_cmd::MATRIX_LOAD
            | gl_cmd::LOAD_4F
            | gl_cmd::RECT
            | gl_cmd::CLEAR_BOX
            | cmd::IMAGE_BOX
    )
}


fn words_hex(words: &[u32]) -> String {
    words.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" ")
}

fn words_f32(words: &[u32]) -> String {
    words.iter().map(|&w| format!("{}", f32::from_bits(w))).collect::<Vec<_>>().join(", ")
}

// ── hooks ────────────────────────────────────────────────────────────────────

impl Board {
    /// A command word the parser completed, about to be dispatched: the tag
    /// (a register number for a register-select word, else the word) and
    /// the operands gathered for it.
    pub fn trace_command(&self, tag: u32) {
        let n = (self.cmd.operand_count as usize).min(self.cmd.operand.len());
        let ops = &self.cmd.operand[..n];
        let name = match tag_name(tag) {
            Some(n) => n.to_string(),
            None if (tag & prog::SHAPE) == prog::TAG => format!("PROG class {:#04x}", tag >> prog::CLASS_SHIFT),
            None => "?".to_string(),
        };
        let floats = if tag_floats(tag) && n > 0 { format!("  ({})", words_f32(ops)) } else { String::new() };
        let kernel = if self.cmd.from_kernel { "" } else { " user" };
        self.trace.fifo(format_args!("{name} {tag:#x} [{}] {}{floats}{kernel}", self.cmd.operand_count, words_hex(ops)));
    }

    /// One of the parser's other completions (copy rect, pattern load, image
    /// box or burst, matrix load, rect, query) or a word it does not know.
    pub fn trace_note(&self, text: std::fmt::Arguments) {
        self.trace.fifo(text);
    }

    /// A primitive assembly sent, with how the board took it, and the
    /// fragment enables of the raster enable word.
    pub fn trace_draw(&self, d: &Prim, textured: bool, how: &str) {
        let m = self.stream.pattern_mode;
        let mut flags = String::new();
        for (on, name) in [
            (textured, "textured"),
            (d.back_colours, "two-sided"),
            (m & mode::DEPTH_TEST != 0, "depth"),
            (m & mode::BLEND != 0, "blend"),
            (m & mode::STENCIL_TEST != 0, "stencil"),
            (m & mode::ALPHA_TEST != 0, "alpha"),
        ] {
            if on {
                flags += " ";
                flags += name;
            }
        }
        self.trace.draw(format_args!("draw {} n={} ctx={}{flags} {how}", begin_name(d.mode), d.count, d.context));
    }

    /// A clear (the rectangle words, the pattern rectangle).
    pub fn trace_raster_op(&self, op: &RasterOp) {
        let kind = if op.kind == RASTER_CLEAR { "clear".to_string() } else { format!("raster kind {}", op.kind) };
        let detail = if op.kind == RASTER_CLEAR {
            format!(
                " buffers {:#x} rgba ({}, {}, {}, {}) mask {:#x} depth {:#08x} stencil {:#x}/{:#x}",
                op.clear_buffers, op.clear_rgba[0], op.clear_rgba[1], op.clear_rgba[2], op.clear_rgba[3], op.clear_colour_mask, op.clear_depth24, op.clear_stencil, op.stencil_mask
            )
        } else {
            String::new()
        };
        self.trace.draw(format_args!("{kind} ({}, {})-({}, {}) targets {}{detail}", op.x0, op.y0, op.x1, op.y1, op.ntargets));
    }

    /// A page-table DMA descriptor as the board runs it.
    pub fn trace_dma(&self, seq: u64, what: &str, table: u64, hdr: &[u32]) {
        self.trace.dma(format_args!(
            "{what} #{seq} table {table:#x} dest {:#x} origin {:#x} width {} size {:#x} fmt {:#010x} skip {} rowbytes {} ctl {:#010x}",
            hdr[dma::HDR_DEST], hdr[dma::HDR_ORIGIN], hdr[dma::HDR_WIDTH], hdr[dma::HDR_SIZE], hdr[dma::HDR_FORMAT], hdr[dma::HDR_SKIP], hdr[dma::HDR_ROWBYTES], hdr[dma::HDR_CONTROL]
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_classes_parse() {
        assert_eq!(trace_mask("all"), Ok(TRACE_ALL));
        assert_eq!(trace_mask("fifo,cpu"), Ok(TRACE_FIFO | TRACE_CPU));
        assert!(trace_mask("hq").is_err());
        assert_eq!(mask_names(TRACE_DRAW | TRACE_DMA), "draw,dma");
    }

    #[test]
    fn words_read_as_what_they_most_likely_are() {
        assert_eq!(tag_name(reg::DEST_SELECT), Some("DEST_SELECT"));
        assert_eq!(tag_name(gl_cmd::VERTEX_3F), Some("VERTEX_3F"));
        assert_eq!(tag_name(0x34008024), Some("MATRIX_SELECT"));
        assert_eq!(tag_name(0x12345678), None);
        assert!(tag_floats(gl_cmd::VERTEX_3F) && !tag_floats(reg::DEST_SELECT));
        assert_eq!(offset_name(ODSY_STATUS0), Some("status0"));
        assert_eq!(offset_name(0x4), Some("identity"));
        assert_eq!(offset_name(0x24), Some("control"));
        assert_eq!(offset_name(0x1234), None);
    }
}
