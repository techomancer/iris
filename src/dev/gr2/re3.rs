//! RE3 raster engine: register state, VRAM and generic (unoptimised) drawing.
//!
//! Register writes arrive through the RE3 FIFO, from the CPU (0x6c200..) or
//! from the HQ2 thread. The RE3 thread applies them in order, and an `IR`
//! write executes. Semantics follow MAME's RE2 model (`sgi_re2.cpp`), the RE3's
//! predecessor; see `ignore/gr2/RE3.h` for what is and isn't verified.
//!
//! VRAM is one u32 per pixel: `cid[31:28] | aux[27:24] | rgb/ci[23:0]`.
//! Rows are stored bottom-up (y = 0 is the bottom scanline).
//! X registers hold the linear pixel column (not bank-coded; see RE3.h).

pub const FB_W: usize = 1280;
pub const FB_H: usize = 1024;

// Buffered registers.
pub const REG_ENABRGB: usize = 0x04;
pub const REG_FUNC: usize = 0x06;
pub const REG_NOPUP: usize = 0x08;
/// Subpixel start fraction (4 bits; see load_iterators).
pub const REG_XYFRAC: usize = 0x09;
pub const REG_RGB: usize = 0x0a;
pub const REG_YX: usize = 0x0b;
pub const REG_PUPDATA: usize = 0x0c;
pub const REG_PATL: usize = 0x0d;
pub const REG_PATH: usize = 0x0e;
pub const REG_DZI: usize = 0x0f;
pub const REG_DZF: usize = 0x10;
pub const REG_DR: usize = 0x11;
pub const REG_DG: usize = 0x12;
pub const REG_DB: usize = 0x13;
pub const REG_Z: usize = 0x14;
pub const REG_R: usize = 0x15;
pub const REG_G: usize = 0x16;
pub const REG_B: usize = 0x17;
pub const REG_STIP: usize = 0x18;
pub const REG_STIPCOUNT: usize = 0x19;
pub const REG_DX: usize = 0x1a;
pub const REG_DY: usize = 0x1b;
pub const REG_NUMPIX: usize = 0x1c;
pub const REG_X: usize = 0x1d;
pub const REG_Y: usize = 0x1e;
pub const REG_IR: usize = 0x1f;
// Unbuffered control registers.
pub const REG_RWDATA: usize = 0x20;
pub const REG_PIXMASK: usize = 0x21;
pub const REG_AUXMASK: usize = 0x22;
pub const REG_WIDDATA: usize = 0x23;
pub const REG_UAUXDATA: usize = 0x24;
pub const REG_RWMODE: usize = 0x25;
pub const REG_ALIGNPAT: usize = 0x29;
pub const REG_ENABPAT: usize = 0x2a;
pub const REG_ENABDITH: usize = 0x2c;
pub const REG_ENABWID: usize = 0x2d;
pub const REG_CURWID: usize = 0x2e;
pub const REG_DEPTHFN: usize = 0x2f;
pub const REG_ENABLWID: usize = 0x31;
pub const REG_FBOPTION: usize = 0x32;
pub const REG_TOPSCAN: usize = 0x33;
pub const REG_UPACMODE: usize = 0x38;
pub const REG_YMIN: usize = 0x39;
pub const REG_YMAX: usize = 0x3a;
pub const REG_XMIN: usize = 0x3b;
pub const REG_XMAX: usize = 0x3c;

pub const IR_SHADED: u32 = 1;
pub const IR_FLAT: u32 = 2;
pub const IR_FLAT4: u32 = 3;
pub const IR_TOPLINE: u32 = 4;
pub const IR_BOTLINE: u32 = 5;
pub const IR_READBUF: u32 = 6;
pub const IR_WRITEBUF: u32 = 7;

pub const RWMODE_FB: u32 = 0;
pub const RWMODE_PUP: u32 = 1;
pub const RWMODE_UAUX: u32 = 2;
pub const RWMODE_ZB: u32 = 3;
pub const RWMODE_WID: u32 = 4;
pub const RWMODE_FB_P: u32 = 6;
pub const RWMODE_ZB_P: u32 = 7;

pub const ROP_COPY: u32 = 3;

/// Register write masks (MAME `sgi_re2.cpp` regmask[]).
pub const REGMASK: [u32; 64] = [
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00000001, 0x0000000f, 0x00000003,
    0x00000001, 0x0000000f, 0x07ffffff, 0x003fffff, 0x00000003, 0x0000ffff, 0x0000ffff, 0x00ffffff,
    0x00003fff, 0x00ffffff, 0x000fffff, 0x000fffff, 0x00ffffff, 0x007fffff, 0x0007ffff, 0x0007ffff,
    0x0000ffff, 0x000000ff, 0x0000ffff, 0x0000ffff, 0x000007ff, 0x0000ffff, 0x000007ff, 0x00000007,
    0xffffffff, 0x00ffffff, 0x000001ff, 0x0000000f, 0x0000000f, 0x00000007, 0x00000001, 0x00000003,
    0x0000003f, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x0000000f, 0x0000000f,
    0x000000ff, 0x00000001, 0x00000003, 0x0003ffff, 0x00000001, 0x00003fff, 0x00000001, 0x000000ff,
    0x00000003, 0x000007ff, 0x000007ff, 0x00000fff, 0x00000fff, 0x00000001, 0x00000001, 0x00000000,
];

/// Register names (MAME `sgi_re2.cpp` regname[]), for the monitor and traces.
pub const REG_NAMES: [&str; 64] = [
    "R00", "R01", "R02", "R03", "ENABRGB", "BIGENDIAN", "FUNC", "HADDR",
    "NOPUP", "XYFRAC", "RGB", "YX", "PUPDATA", "PATL", "PATH", "DZI",
    "DZF", "DR", "DG", "DB", "Z", "R", "G", "B",
    "STIP", "STIPCOUNT", "DX", "DY", "NUMPIX", "X", "Y", "IR",
    "RWDATA", "PIXMASK", "AUXMASK", "WIDDATA", "UAUXDATA", "RWMODE", "READBUF", "PIXTYPE",
    "ASELECT", "ALIGNPAT", "ENABPAT", "ENABSTIP", "ENABDITH", "ENABWID", "CURWID", "DEPTHFN",
    "REPSTIP", "ENABLWID", "FBOPTION", "TOPSCAN", "TESTMODE", "TESTDATA", "ZBOPTION", "XZOOM",
    "UPACMODE", "YMIN", "YMAX", "XMIN", "XMAX", "COLORCMP", "MEGOPTION", "R3F",
];

pub fn ir_name(ir: u32) -> &'static str {
    match ir {
        IR_SHADED => "SHADED",
        IR_FLAT => "FLAT",
        IR_FLAT4 => "FLAT4",
        IR_TOPLINE => "TOPLINE",
        IR_BOTLINE => "BOTLINE",
        IR_READBUF => "READBUF",
        IR_WRITEBUF => "WRITEBUF",
        _ => "IR?",
    }
}

pub fn rwmode_name(m: u32) -> &'static str {
    match m {
        RWMODE_FB => "FB",
        RWMODE_PUP => "PUP",
        RWMODE_UAUX => "UAUX",
        RWMODE_ZB => "ZB",
        RWMODE_WID => "WID",
        RWMODE_FB_P => "FB_P",
        RWMODE_ZB_P => "ZB_P",
        _ => "?",
    }
}

/// One-line decode of the primitive an `IR` write is about to execute, from
/// the register file as it stands (before `execute` loads the iterators).
pub fn describe_ir(reg: &[u32; 64], ir: u32) -> String {
    let x = reg[REG_X] as i32;
    let rgb = |r: u32| r >> 11;
    let mut s = format!("{} x={} y={} n={} rgb=({},{},{}) func={} rwmode={}",
        ir_name(ir), x, reg[REG_Y], reg[REG_NUMPIX],
        rgb(reg[REG_R]), rgb(reg[REG_G]), rgb(reg[REG_B]), reg[REG_FUNC], rwmode_name(reg[REG_RWMODE]));
    if reg[REG_ENABPAT] != 0 {
        s += &format!(" pat={:#010x}{}", (reg[REG_PATH] << 16) | reg[REG_PATL],
            if reg[REG_ALIGNPAT] != 0 { "/aligned" } else { "" });
    }
    if ir == IR_SHADED {
        s += &format!(" dx={:#x} dy={:#x} drgb=({:#x},{:#x},{:#x}) z={:#x} dz={:#x}.{:#x}",
            reg[REG_DX], reg[REG_DY], reg[REG_DR], reg[REG_DG], reg[REG_DB], reg[REG_Z], reg[REG_DZI], reg[REG_DZF]);
    }
    if reg[REG_ENABWID] != 0 {
        s += &format!(" wid={}", reg[REG_CURWID]);
    }
    s
}

// X registers (X, XMIN, XMAX, the X half of YX) hold the linear pixel column.
// Guest software never writes RE3 registers directly (only the HQ2/GE
// microcode, which is HLE'd), so this is internal. The bank-coded form
// (x / 5) * 8 + x % 5 from MAME's RE2 model is not used (RE3.h).

// ── Private FIFO operations (emulator-only; what the real HQ2/GE7 microcode
//    did with longer register sequences) ────────────────────────────────────
/// Screen-to-screen copy, first half: val = sx | sy<<16 | w<<32 | h<<48.
pub const RE3_OP_COPY_A: u32 = 0xFFFF_1001;
/// Second half (executes): val = dx | dy<<16. Coordinates are linear, y bottom-up.
pub const RE3_OP_COPY_B: u32 = 0xFFFF_1002;
/// RE3_OP_COPY_B bits 63:32 for a GL copy (glCopyPixels; the microcode's
/// copy, emulated): zoom x (7:0) and y (15:8), integer >= 1; the pixel
/// depth of the visual's buffers / 4 (17:16: 0 = 24-bit, no banks) and the
/// read buffer's bank (18); bit 19 set (nonzero = GL copy). The source
/// pixel is taken from the read bank and replicated into every bank, so
/// PIXMASK picks the draw buffer, as for drawing.
pub fn copy_mode(zx: u32, zy: u32, bits: u32, bank: u32) -> u32 {
    (zx & 0xff) | ((zy & 0xff) << 8) | (((bits / 4) & 3) << 16) | ((bank & 1) << 18) | 1 << 19
}
/// CPU consumed RWDATA during a READBUF stream; fetch the next pixel.
pub const RE3_OP_READ_ADVANCE: u32 = 0xFFFF_1003;
/// A host-visible finish/readback boundary, queued so telemetry observes it
/// in exact RE3 order.
pub const RE3_OP_SYNC_BOUNDARY: u32 = 0xFFFF_100B;
/// Emulator-private stand-in for RE3's (unidentified) packing-mode register:
/// the GE sends neutral 8-bit-per-channel colour and RE3 packs each pixel
/// into the window's frame-buffer format (RE3.h "Colour packing"). 0 = 24-bit / colour index as the iterators
/// produce it; PIXFMT_RGB12 = 12-bit 4:4:4 RGB (R in bits 3:0) duplicated
/// into both 12-bit buffers, the plane mask selecting the buffer.
pub const RE3_OP_PIXFMT: u32 = 0xFFFF_1004;
pub const PIXFMT_RGB12: u32 = 1;
/// 12-bit colour index: the R iterator (12.11) carries the index, written
/// into both 12-bit banks like PIXFMT_RGB12 (the plane mask picks the buffer).
pub const PIXFMT_CI12: u32 = 2;
/// 8-bit colour index: the R iterator carries the index (0..255), written
/// as i | i << 8 so either 8-bit buffer mask works (libglcore builds the
/// double-buffer masks by shifting by the depth: 0x00FF / 0xFF00 for 8
/// bits; the bank-1 position is unverified).
pub const PIXFMT_CI8: u32 = 3;
/// 8-bit 3:3:2 TrueColor (Xsgi visual masks: R 7:5, B 4:3, G 2:0), packed
/// from 8:8:8 by truncation and written as v | v << 8 like PIXFMT_CI8.
pub const PIXFMT_RGB8: u32 = 4;
/// 4-bit 1:2:1 TrueColor (Xsgi visual 0x24 masks: R 3, G 2:1, B 0), packed
/// from 8:8:8 by truncation and written as v | v << 4 (bank 1 = bits 7:4,
/// libglcore's write masks shifted by the depth).
pub const PIXFMT_RGB4: u32 = 5;
/// Emulator-private depth control (the enables and write mask have no
/// documented RE3 register): val = test enable (bit 0) | func (bits 3:1,
/// GL order NEVER..ALWAYS) | Z write mask << 8 (24 bits). Depth is tested
/// and written by SHADED spans (the only spans that iterate Z per pixel).
pub const RE3_OP_ZCTL: u32 = 0xFFFF_1005;
/// Emulator-private stencil control: val = enable (bit 0) | func << 1 |
/// ref << 4 (8 bits) | mask << 12 (8) | write mask << 20 (8) | fail << 28 |
/// zfail << 32 | zpass << 36 (ops 0 KEEP, 1 INVERT, 2 ZERO, 3 REPLACE,
/// 4 INCR, 5 DECR; libglcore __glExpPassStencilMode codes).
pub const RE3_OP_STENCIL: u32 = 0xFFFF_1006;
/// Emulator-private Z/stencil rectangle fill: RE3_OP_ZFILL_A = x0 | y0 << 16
/// | x1 << 32 | y1 << 48 (exclusive), RE3_OP_ZFILL_B = value (bits 31:0) |
/// mask << 32: zbuf = (zbuf & !mask) | (value & mask). Z is bits 23:0,
/// stencil bits 31:24 of the emulator's zbuf word (layout unverified).
pub const RE3_OP_ZFILL_A: u32 = 0xFFFF_1007;
pub const RE3_OP_ZFILL_B: u32 = 0xFFFF_1008;
pub const ZBUF_STENCIL_SHIFT: u32 = 24;
/// EMULATOR EXTENSION, not RE3 hardware. Real RE3 has no blend unit and no
/// alpha iterator: on GR2 the GE7 microcode does blending (reading the
/// destination back through RE3 and writing the result). Doing it here, per
/// pixel inside RE3, gives the same pixels without a span read-modify-write
/// round trip (GR2.h "BLENDING AND ALPHA TEST"; verified against a real Indy
/// XZ with glprim --scene blendsmooth). There is deliberately no alpha test:
/// libglcore sends only the fragments that pass.
///
/// Emulator-private blend control: val = enable (bit 0) | src factor << 1 |
/// dst factor << 4 (libglcore __glExpPassBlendFunc codes: 0 ZERO, 1 ONE,
/// 2 DST_COLOR (src) / SRC_COLOR (dst), 3 ONE_MINUS of that, 4 SRC_ALPHA,
/// 5 ONE_MINUS_SRC_ALPHA). Applied by SHADED spans; the destination is the
/// frame-buffer colour expanded to 8 bits per channel (no alpha planes).
pub const RE3_OP_BLEND: u32 = 0xFFFF_1009;
/// Emulator-private source alpha iterator: val = start (8.11, bits 31:0) |
/// per-pixel step (signed 8.11) << 32.
pub const RE3_OP_ALPHA: u32 = 0xFFFF_100A;

/// Stop the RE3 thread.
pub const RE3_OP_EXIT: u32 = 0xFFFF_1FFF;
/// Set on register entries pushed by the HQ2 thread (vs. CPU writes to the
/// RE3 window), so traces can show where each write came from.
pub const RE3_SRC_HQ: u32 = 0x100;

/// Streaming state of READBUF/WRITEBUF.
pub const STREAM_IDLE: u32 = 0;
pub const STREAM_READ: u32 = 1;
pub const STREAM_WRITE: u32 = 2;

/// RE3 register file plus the working state of the current primitive.
/// C-style and `Copy` so a future JIT can address it by offset.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Re3Context {
    pub reg: [u32; 64],
    /// Iterators for the executing primitive (14-bit fraction, like MAME).
    pub x: i64,
    pub y: i64,
    pub z: i64,
    pub r: i64,
    pub g: i64,
    pub b: i64,
    pub numpix: u32,
    pub stream: u32,
    /// RE3_OP_PIXFMT value.
    pub pixfmt: u32,
    /// RE3_OP_ZCTL / RE3_OP_STENCIL values.
    pub zctl: u32,
    pub stencil: u64,
    /// RE3_OP_ZFILL_A, until RE3_OP_ZFILL_B arrives.
    pub zfill_rect: u64,
    /// RE3_OP_BLEND value; alpha iterator (8.11) and its per-pixel step.
    pub blend: u32,
    pub a: i64,
    pub da: i64,
}

pub const SPAN_BUCKET_LIMITS: [u32; 8] = [4, 8, 16, 32, 64, 256, 1024, u32::MAX];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Re3TelemetrySnapshot {
    pub spans: u64,
    pub pixels: u64,
    pub span_buckets: [u64; 8],
    pub batchable_spans: u64,
    pub max_batch: u32,
    pub sync_boundaries: u64,
    pub framebuffer_read_words: u64,
    pub framebuffer_write_words: u64,
    pub z_read_words: u64,
    pub z_write_words: u64,
    pub host_read_pixels: u64,
}

/// Fixed-size, runtime-only evidence counters. They are deliberately outside
/// `Re3Context`, so snapshots restore rendering state without restoring a
/// measurement window.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Re3Telemetry {
    snapshot: Re3TelemetrySnapshot,
    last_pipeline: u64,
    current_batch: u32,
    have_pipeline: u32,
}

/// RE3 state owned by the RE3 thread. Plain data, zero-valid.
#[repr(C)]
pub struct Re3 {
    pub ctx: Re3Context,
    pub vram: [u32; FB_W * FB_H],
    pub zbuf: [u32; FB_W * FB_H],
    /// The span JIT's state (`re3_jit`): mode, memoised shaders, counts.
    #[cfg(feature = "gr2-jit")]
    pub jit: super::re3_jit::JitState,
    pub telemetry: Re3Telemetry,
}

impl Re3 {
    pub fn telemetry_snapshot(&self) -> Re3TelemetrySnapshot {
        self.telemetry.snapshot
    }

    pub fn reset_telemetry(&mut self) {
        self.telemetry = Re3Telemetry::default();
        #[cfg(feature = "gr2-jit")]
        {
            self.jit.hits = 0;
            self.jit.misses = 0;
        }
    }

    pub fn record_sync_boundary(&mut self) {
        self.telemetry.snapshot.sync_boundaries =
            self.telemetry.snapshot.sync_boundaries.saturating_add(1);
        self.telemetry.have_pipeline = 0;
        self.telemetry.current_batch = 0;
    }

    pub fn record_host_read(&mut self, pixels: u64) {
        self.telemetry.snapshot.host_read_pixels =
            self.telemetry.snapshot.host_read_pixels.saturating_add(pixels);
    }

    fn record_span(&mut self) {
        let c = &self.ctx;
        let pixels = c.numpix as u64;
        let bucket = SPAN_BUCKET_LIMITS
            .iter()
            .position(|&limit| c.numpix <= limit)
            .unwrap_or(SPAN_BUCKET_LIMITS.len() - 1);
        let s = &mut self.telemetry.snapshot;
        s.spans = s.spans.saturating_add(1);
        s.pixels = s.pixels.saturating_add(pixels);
        s.span_buckets[bucket] = s.span_buckets[bucket].saturating_add(1);

        let mut signature = 0xcbf2_9ce4_8422_2325u64;
        for value in [
            c.reg[REG_IR],
            c.reg[REG_FUNC],
            c.reg[REG_NOPUP],
            c.reg[REG_ENABPAT],
            c.reg[REG_ALIGNPAT],
            c.reg[REG_ENABDITH],
            c.reg[REG_ENABWID],
            c.reg[REG_ENABLWID],
            c.reg[REG_FBOPTION],
            c.reg[REG_DEPTHFN],
            c.pixfmt,
            c.zctl & 0xff,
            (c.stencil & 0x000f_ffff_fff) as u32,
            (c.stencil >> 32) as u32,
            c.blend,
        ] {
            signature ^= value as u64;
            signature = signature.wrapping_mul(0x100_0000_01b3);
        }
        if self.telemetry.have_pipeline != 0 && self.telemetry.last_pipeline == signature {
            s.batchable_spans = s.batchable_spans.saturating_add(1);
            self.telemetry.current_batch = self.telemetry.current_batch.saturating_add(1);
        } else {
            self.telemetry.current_batch = 1;
        }
        self.telemetry.have_pipeline = 1;
        self.telemetry.last_pipeline = signature;
        s.max_batch = s.max_batch.max(self.telemetry.current_batch);

        let fb_reads_per_pixel = 1
            + u64::from(c.reg[REG_ENABWID] != 0)
            + u64::from(c.blend & 1 != 0);
        s.framebuffer_read_words = s
            .framebuffer_read_words
            .saturating_add(pixels.saturating_mul(fb_reads_per_pixel));
        s.framebuffer_write_words = s.framebuffer_write_words.saturating_add(pixels);
        if c.zctl & 1 != 0 || c.stencil & 1 != 0 {
            s.z_read_words = s.z_read_words.saturating_add(pixels);
            s.z_write_words = s.z_write_words.saturating_add(pixels);
        }
    }

    /// Apply one register write (masked), with MAME's fan-out side effects.
    /// Returns true when the write executed something that touches VRAM.
    pub fn write_reg(&mut self, reg: usize, val: u32) -> bool {
        let reg = reg & 63;
        let v = val & REGMASK[reg];
        #[cfg(feature = "gr2-jit")]
        if super::re3_jit::shapes_key(reg) && self.ctx.reg[reg] != v {
            self.jit.invalidate();
        }
        self.ctx.reg[reg] = v;
        match reg {
            REG_RGB => {
                // 27-bit packed 9:9:9 (RE3.h). Each 9-bit component lands in
                // the integer part of the 8.11 R/G/B iterators.
                self.ctx.reg[REG_R] = ((v >> 18) & 0x1ff) << 11;
                self.ctx.reg[REG_G] = (((v >> 9) & 0x1ff) << 11) & REGMASK[REG_G];
                self.ctx.reg[REG_B] = ((v & 0x1ff) << 11) & REGMASK[REG_B];
                false
            }
            REG_YX => {
                self.ctx.reg[REG_X] = v & 0xfff;
                self.ctx.reg[REG_Y] = (v >> 12) & 0x7ff;
                false
            }
            REG_IR => {
                self.execute();
                true
            }
            REG_RWDATA => {
                if self.ctx.stream == STREAM_WRITE {
                    self.write_buffer();
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// The private operations that shape span pipelines (RE3_OP_PIXFMT,
    /// RE3_OP_ZCTL, RE3_OP_STENCIL, RE3_OP_BLEND).
    pub fn set_pixfmt(&mut self, v: u32) {
        #[cfg(feature = "gr2-jit")]
        if self.ctx.pixfmt != v {
            self.jit.invalidate();
        }
        self.ctx.pixfmt = v;
    }

    pub fn set_zctl(&mut self, v: u32) {
        #[cfg(feature = "gr2-jit")]
        if self.ctx.zctl & 0xf != v & 0xf {
            self.jit.invalidate();
        }
        self.ctx.zctl = v;
    }

    pub fn set_stencil(&mut self, v: u64) {
        #[cfg(feature = "gr2-jit")]
        {
            // Enable, function and the three ops; the rest is span data.
            let shape = |s: u64| s & (0xf | 0xfff << 28);
            if shape(self.ctx.stencil) != shape(v) {
                self.jit.invalidate();
            }
        }
        self.ctx.stencil = v;
    }

    pub fn set_blend(&mut self, v: u32) {
        #[cfg(feature = "gr2-jit")]
        if self.ctx.blend != v {
            self.jit.invalidate();
        }
        self.ctx.blend = v;
    }

    fn load_iterators(&mut self) {
        let c = &mut self.ctx;
        c.x = (c.reg[REG_X] as i64) << 14;
        c.y = (c.reg[REG_Y] as i64) << 14;
        // XYFRAC (inferred): the minor axis's subpixel start in 1/16 pixel,
        // the GE's vertex precision. The major axis steps whole pixels from
        // a pixel start (its step is +-1.0), so one 4-bit fraction suffices;
        // it becomes the top 4 bits of the minor iterator's 14-bit fraction.
        let frac = ((c.reg[REG_XYFRAC] & 0xf) as i64) << 10;
        let dx = (c.reg[REG_DX] as u16 as i16) as i32;
        if dx.abs() == 0x4000 {
            c.y |= frac;
        } else {
            c.x |= frac;
        }
        c.z = ((c.reg[REG_Z] as i64) << 40) >> 26;
        c.r = c.reg[REG_R] as i64;
        c.g = c.reg[REG_G] as i64;
        c.b = c.reg[REG_B] as i64;
        c.numpix = c.reg[REG_NUMPIX];
    }

    fn step_iterators(&mut self) {
        let c = &mut self.ctx;
        c.x += (c.reg[REG_DX] as u16 as i16) as i64;
        c.y += (c.reg[REG_DY] as u16 as i16) as i64;
        c.z += (((c.reg[REG_DZI] as i64) << 40) >> 26) | c.reg[REG_DZF] as i64;
        c.r += sext(c.reg[REG_DR], 24);
        c.g += sext(c.reg[REG_DG], 20);
        c.b += sext(c.reg[REG_DB], 20);
        c.a += c.da;
    }

    fn execute(&mut self) {
        self.load_iterators();
        if matches!(self.ctx.reg[REG_IR], IR_SHADED | IR_FLAT | IR_FLAT4) {
            self.record_span();
        }
        #[cfg(feature = "gr2-jit")]
        if super::re3_jit::run(self, self.ctx.reg[REG_IR]) {
            return;
        }
        match self.ctx.reg[REG_IR] {
            IR_SHADED => self.draw_shaded_span(),
            IR_FLAT => self.draw_flat_span(5),
            IR_FLAT4 => self.draw_flat_span(20),
            IR_READBUF => {
                self.ctx.stream = STREAM_READ;
                self.read_buffer();
            }
            IR_WRITEBUF => {
                self.ctx.stream = if self.ctx.numpix > 0 { STREAM_WRITE } else { STREAM_IDLE };
            }
            // TOPLINE / BOTLINE (antialiased lines): not needed for the textport yet.
            _ => {}
        }
    }

    /// Current colour from the iterators: 8.11 R/G/B -> pixel bits at
    /// (x, y). In colour-index mode G/B are zero and R carries the (up to
    /// 12-bit) index. 12-bit RGB packs 4:4:4 (R low, same convention as
    /// REX3 DRAWDEPTH 12) into both buffers, dithered 8 -> 4 bits with the
    /// REX3 4x4 Bayer matrix when ENABDITH is set.
    #[inline]
    fn color(&self, x: i32, y: i32) -> u32 {
        let c = &self.ctx;
        if c.pixfmt == PIXFMT_CI12 {
            let i = (c.r >> 11).clamp(0, 0xfff) as u32;
            return i | (i << 12);
        }
        if c.pixfmt == PIXFMT_CI8 {
            let i = (c.r >> 11).clamp(0, 0xff) as u32;
            return i | (i << 8);
        }
        let ch = |v: i64| (v >> 11).clamp(0, 255) as u32;
        self.pack(ch(c.r) | (ch(c.g) << 8) | (ch(c.b) << 16), x, y)
    }

    /// Colour for a SHADED span pixel at `off`, blended with the frame
    /// buffer when RE3_OP_BLEND is on.
    #[inline]
    fn color_blended(&self, x: i32, y: i32, off: usize) -> u32 {
        let c = &self.ctx;
        // No blending in colour-index mode (GL).
        if c.blend & 1 == 0 || c.pixfmt == PIXFMT_CI12 || c.pixfmt == PIXFMT_CI8 {
            return self.color(x, y);
        }
        let ch = |v: i64| (v >> 11).clamp(0, 255) as u32;
        let s = [ch(c.r), ch(c.g), ch(c.b)];
        let a = ch(c.a);
        let d = self.dst_rgb8(off);
        let factor = |code: u32, k: usize, other: [u32; 3]| -> u32 {
            match code {
                0 => 0,
                1 => 255,
                2 => other[k],
                3 => 255 - other[k],
                4 => a,
                5 => 255 - a,
                _ => 255,
            }
        };
        let (sf, df) = ((c.blend >> 1) & 7, (c.blend >> 4) & 7);
        let mut out = 0;
        for k in 0..3 {
            let v = (s[k] * factor(sf, k, d) + d[k] * factor(df, k, s) + 127) / 255;
            out |= v.min(255) << (8 * k);
        }
        self.pack(out, x, y)
    }

    /// Frame-buffer colour at `off` as 8-bit R, G, B for blending. For 12-bit
    /// RGB, the buffer being written (PIXMASK) is read and nibbles expanded.
    fn dst_rgb8(&self, off: usize) -> [u32; 3] {
        let p = self.vram[off];
        if self.ctx.pixfmt == PIXFMT_RGB12 {
            let buf = if self.ctx.reg[REG_PIXMASK] & 0xfff == 0 { (p >> 12) & 0xfff } else { p & 0xfff };
            return [(buf & 0xf) * 17, ((buf >> 4) & 0xf) * 17, ((buf >> 8) & 0xf) * 17];
        }
        if self.ctx.pixfmt == PIXFMT_RGB8 {
            let v = if self.ctx.reg[REG_PIXMASK] & 0xff == 0 { (p >> 8) & 0xff } else { p & 0xff };
            return rgb332_to_888(v);
        }
        if self.ctx.pixfmt == PIXFMT_RGB4 {
            let v = if self.ctx.reg[REG_PIXMASK] & 0xf == 0 { (p >> 4) & 0xf } else { p & 0xf };
            return rgb121_to_888(v);
        }
        [p & 0xff, (p >> 8) & 0xff, (p >> 16) & 0xff]
    }

    /// Pack an 8-bit-per-channel colour (R in bits 7:0) into the frame-buffer
    /// format selected by RE3_OP_PIXFMT.
    #[inline]
    fn pack(&self, rgb: u32, x: i32, y: i32) -> u32 {
        let c = &self.ctx;
        if c.pixfmt == PIXFMT_RGB12 {
            let v = if c.reg[REG_ENABDITH] != 0 {
                crate::dev::ng1::rex3_generic::rgb24_to_rgb12_dither(crate::dev::ng1::rex3_generic::bayer_pack(rgb, x, y))
            } else {
                crate::dev::ng1::rex3_generic::rgb24_to_rgb12(rgb)
            };
            return v | (v << 12);
        }
        if c.pixfmt == PIXFMT_RGB8 {
            let v = (rgb & 0xe0) | ((rgb >> 19) & 0x18) | ((rgb >> 13) & 0x07);
            return v | (v << 8);
        }
        if c.pixfmt == PIXFMT_RGB4 {
            let v = ((rgb >> 4) & 0x8) | ((rgb >> 13) & 0x6) | ((rgb >> 23) & 0x1);
            return v | (v << 4);
        }
        rgb & 0x00ff_ffff
    }

    /// aux/cid bits written with a span, and the combined write mask.
    #[inline]
    fn aux_and_mask(&self) -> (u32, u32) {
        let r = &self.ctx.reg;
        let aux = if r[REG_NOPUP] != 0 {
            (r[REG_WIDDATA] << 28) | (r[REG_UAUXDATA] << 24)
        } else {
            (r[REG_WIDDATA] << 28) | ((r[REG_UAUXDATA] & 3) << 26) | (r[REG_PUPDATA] << 24)
        };
        let mask = ((r[REG_AUXMASK] & 0xff) << 24) | r[REG_PIXMASK];
        (aux, mask)
    }

    #[inline]
    fn clip(&self, x: i32, y: i32) -> bool {
        let r = &self.ctx.reg;
        x >= 0 && y >= 0 && (x as usize) < FB_W && (y as usize) < FB_H
            && x >= (r[REG_XMIN] as i32) && x <= (r[REG_XMAX] as i32)
            && y >= r[REG_YMIN] as i32 && y <= r[REG_YMAX] as i32
    }

    #[inline]
    fn pattern(&self, x: i32, n: u32) -> bool {
        let r = &self.ctx.reg;
        if r[REG_ENABPAT] == 0 {
            return true;
        }
        let idx = if r[REG_ALIGNPAT] != 0 { x as u32 } else { n } & 31;
        let pat = (r[REG_PATH] << 16) | r[REG_PATL];
        (pat >> (31 - idx)) & 1 != 0
    }

    #[inline]
    fn wid_ok(&self, ir: u32, offset: usize) -> bool {
        let r = &self.ctx.reg;
        if r[REG_ENABWID] == 0 {
            return true;
        }
        if (ir == IR_TOPLINE || ir == IR_BOTLINE) && r[REG_ENABLWID] == 0 {
            return true;
        }
        let wid = self.vram[offset] >> 28;
        if r[REG_FBOPTION] & 1 != 0 {
            if r[REG_DEPTHFN] & 8 != 0 {
                (wid & 0xe) == (r[REG_CURWID] & 0xe)
            } else {
                (wid & 0xf) == (r[REG_CURWID] & 0xf)
            }
        } else {
            (wid & 3) == (r[REG_CURWID] & 3)
        }
    }

    /// Raster op on the colour/aux bits, then a masked store.
    #[inline]
    fn store(&mut self, offset: usize, src: u32, mask: u32) {
        let dst = self.vram[offset];
        let v = rop(self.ctx.reg[REG_FUNC], src, dst);
        self.vram[offset] = (dst & !mask) | (v & mask);
    }

    /// Stencil and depth tests for one fragment at `off`, applying the
    /// stencil ops and the masked Z write. True if the colour may be written.
    #[inline]
    fn zs_test(&mut self, off: usize) -> bool {
        let (zctl, st) = (self.ctx.zctl, self.ctx.stencil);
        if zctl & 1 == 0 && st & 1 == 0 {
            return true;
        }
        let cmp = |func: u32, a: u32, b: u32| match func & 7 {
            0 => false,
            1 => a < b,
            2 => a == b,
            3 => a <= b,
            4 => a > b,
            5 => a != b,
            6 => a >= b,
            _ => true,
        };
        let word = self.zbuf[off];
        let mut new = word;
        let mut pass = true;
        let mut sop = 0u64;
        if st & 1 != 0 {
            let (func, sref, smask) = ((st >> 1) & 7, (st >> 4) & 0xff, (st >> 12) & 0xff);
            let sv = (word >> ZBUF_STENCIL_SHIFT) as u64 & 0xff;
            if !cmp(func as u32, (sref & smask) as u32, (sv & smask) as u32) {
                pass = false;
                sop = (st >> 28) & 0xf;
            }
        }
        if pass && zctl & 1 != 0 {
            // Z is a signed 24-bit value: IRIS GL uses the full range
            // (zclear -0x800000, lsetdepth ZMIN..ZMAX, ZF_GEQUAL), OpenGL
            // only 0..0x7FFFFF (23 bits). Bias both sides so the unsigned
            // compare orders them as signed.
            let z = ((self.ctx.z >> 14).clamp(-0x80_0000, 0x7f_ffff) as u32) & 0x00ff_ffff;
            if !cmp((zctl >> 1) & 7, z ^ 0x80_0000, (word & 0x00ff_ffff) ^ 0x80_0000) {
                pass = false;
                sop = (st >> 32) & 0xf;
            } else {
                let wm = zctl >> 8 & 0x00ff_ffff;
                new = (new & !wm) | (z & wm);
            }
        }
        if pass {
            sop = (st >> 36) & 0xf;
        }
        if st & 1 != 0 {
            let (sref, wmask) = ((st >> 4) & 0xff, ((st >> 20) & 0xff) as u32);
            let sv = (new >> ZBUF_STENCIL_SHIFT) & 0xff;
            let nv = match sop {
                1 => !sv,
                2 => 0,
                3 => sref as u32,
                4 => (sv + 1).min(0xff),
                5 => sv.saturating_sub(1),
                _ => sv,
            } & 0xff;
            let sv2 = (sv & !wmask) | (nv & wmask);
            new = (new & 0x00ff_ffff) | (sv2 << ZBUF_STENCIL_SHIFT);
        }
        self.zbuf[off] = new;
        pass
    }

    /// RE3_OP_ZFILL: fill a rectangle of the Z/stencil buffer.
    pub fn zfill(&mut self, rect: u64, val: u64) {
        let f = |s: u32| ((rect >> s) & 0xffff) as usize;
        let (x0, y0, x1, y1) = (f(0), f(16), f(32).min(FB_W), f(48).min(FB_H));
        let (value, mask) = (val as u32, (val >> 32) as u32);
        let pixels = y1.saturating_sub(y0) as u64 * x1.saturating_sub(x0) as u64;
        self.telemetry.snapshot.z_read_words =
            self.telemetry.snapshot.z_read_words.saturating_add(pixels);
        self.telemetry.snapshot.z_write_words =
            self.telemetry.snapshot.z_write_words.saturating_add(pixels);
        for y in y0..y1 {
            for z in &mut self.zbuf[y * FB_W + x0..y * FB_W + x1.max(x0)] {
                *z = (*z & !mask) | (value & mask);
            }
        }
    }

    fn draw_shaded_span(&mut self) {
        let (aux, mask) = self.aux_and_mask();
        let mut n = 0u32;
        while self.ctx.numpix > 0 {
            self.ctx.numpix -= 1;
            let x = (self.ctx.x >> 14) as i32;
            let y = (self.ctx.y >> 14) as i32;
            if self.clip(x, y) && self.pattern(x, n) {
                let off = y as usize * FB_W + x as usize;
                if self.wid_ok(IR_SHADED, off) && self.zs_test(off) {
                    let c = self.color_blended(x, y, off);
                    self.store(off, aux | c, mask);
                }
            }
            self.step_iterators();
            n += 1;
        }
    }

    /// Flat span: `numpix` consecutive pixels from (x, y). Colour steps once
    /// per `n` pixels.
    fn draw_flat_span(&mut self, n: u32) {
        let (aux, mask) = self.aux_and_mask();
        let x0 = (self.ctx.x >> 14) as i32;
        let y = (self.ctx.y >> 14) as i32;
        let count = self.ctx.numpix;
        // Dithering makes the colour depend on the pixel position.
        let per_pixel = self.ctx.pixfmt == PIXFMT_RGB12 && self.ctx.reg[REG_ENABDITH] != 0;
        let mut c = self.color(x0, y);
        for i in 0..count {
            let x = x0 + i as i32;
            if self.clip(x, y) && self.pattern(x, i) {
                let off = y as usize * FB_W + x as usize;
                if self.wid_ok(IR_FLAT, off) {
                    let c = if per_pixel { self.color(x, y) } else { c };
                    self.store(off, aux | c, mask);
                }
            }
            if (i + 1) % n == 0 {
                self.step_iterators();
                c = self.color(x + 1, y);
            }
        }
        self.ctx.numpix = 0;
    }

    /// Plane value at `offset` for the current RWMODE.
    fn plane_read(&self, offset: usize) -> u32 {
        let p = self.vram[offset];
        match self.ctx.reg[REG_RWMODE] {
            RWMODE_PUP => (p >> 24) & 0x3,
            RWMODE_UAUX => (p >> 24) & 0xf,
            RWMODE_WID => p >> 28,
            RWMODE_ZB | RWMODE_ZB_P => self.zbuf[offset],
            RWMODE_FB_P => p,
            _ => p & 0x00ff_ffff,
        }
    }

    fn plane_write(&mut self, offset: usize, data: u32) {
        let r = &self.ctx.reg;
        let aux_mask = r[REG_AUXMASK];
        match r[REG_RWMODE] {
            RWMODE_PUP => {
                let m = (aux_mask & 0x3) << 24;
                self.store(offset, (data & 0x3) << 24, m);
            }
            RWMODE_UAUX => {
                let m = (aux_mask & if r[REG_NOPUP] != 0 { 0xf } else { 0xc }) << 24;
                self.store(offset, (data & 0xf) << 24, m);
            }
            RWMODE_WID => {
                let m = ((aux_mask >> 4) & 0xf) << 28;
                self.store(offset, (data & 0xf) << 28, m);
            }
            RWMODE_ZB | RWMODE_ZB_P => self.zbuf[offset] = data & 0x00ff_ffff,
            RWMODE_FB_P => self.vram[offset] = data,
            _ => {
                let m = r[REG_PIXMASK];
                self.store(offset, data & 0x00ff_ffff, m);
            }
        }
    }

    /// READBUF: latch the next pixel into RWDATA, or finish the stream.
    pub fn read_buffer(&mut self) {
        if self.ctx.numpix == 0 {
            self.ctx.stream = STREAM_IDLE;
            return;
        }
        let x = (self.ctx.x >> 14) as i32;
        let y = (self.ctx.y >> 14) as i32;
        self.record_host_read(1);
        if matches!(self.ctx.reg[REG_RWMODE], RWMODE_ZB | RWMODE_ZB_P) {
            self.telemetry.snapshot.z_read_words =
                self.telemetry.snapshot.z_read_words.saturating_add(1);
        } else {
            self.telemetry.snapshot.framebuffer_read_words =
                self.telemetry.snapshot.framebuffer_read_words.saturating_add(1);
        }
        self.ctx.reg[REG_RWDATA] = if x >= 0 && y >= 0 && (x as usize) < FB_W && (y as usize) < FB_H {
            self.plane_read(y as usize * FB_W + x as usize)
        } else {
            0
        };
        self.step_iterators();
        self.ctx.numpix -= 1;
    }

    /// WRITEBUF: consume RWDATA (unpacked per UPACMODE) into VRAM.
    fn write_buffer(&mut self) {
        let mode = self.ctx.reg[REG_UPACMODE];
        let data = self.ctx.reg[REG_RWDATA];
        for i in 0..=mode {
            if self.ctx.numpix == 0 {
                break;
            }
            let x = (self.ctx.x >> 14) as i32;
            let y = (self.ctx.y >> 14) as i32;
            if self.clip(x, y) {
                let off = y as usize * FB_W + x as usize;
                if self.wid_ok(IR_WRITEBUF, off) {
                    self.plane_write(off, unpack(data, i, mode));
                }
            }
            self.step_iterators();
            self.ctx.numpix -= 1;
        }
        if self.ctx.numpix == 0 {
            self.ctx.stream = STREAM_IDLE;
        }
    }

    /// GL copy (see copy_mode): source rectangle at (sx, sy) to (dx, dy),
    /// each pixel repeated zx x zy times, through the logic op, write masks,
    /// scissor and WID test. The source is read first, so overlap is safe.
    pub fn copy_rect_gl(&mut self, sx: i32, sy: i32, w: i32, h: i32, dx: i32, dy: i32, mode: u32) {
        if w <= 0 || h <= 0 {
            return;
        }
        let (zx, zy) = ((mode & 0xff).max(1) as i32, ((mode >> 8) & 0xff).max(1) as i32);
        let source = (w as u64).saturating_mul(h as u64);
        let dest = source.saturating_mul(zx as u64).saturating_mul(zy as u64);
        self.telemetry.snapshot.framebuffer_read_words = self
            .telemetry
            .snapshot
            .framebuffer_read_words
            .saturating_add(source.saturating_add(dest));
        self.telemetry.snapshot.framebuffer_write_words = self
            .telemetry
            .snapshot
            .framebuffer_write_words
            .saturating_add(dest);
        let bits = ((mode >> 16) & 3) * 4;
        let bank = (mode >> 18) & 1;
        let pix = |v: u32| -> u32 {
            if bits == 0 {
                return v;
            }
            let m = (1u32 << bits) - 1;
            let c = (v >> (bits * bank)) & m;
            (v & 0xff00_0000) | c | (c << bits)
        };
        let mut src = vec![0u32; (w * h) as usize];
        for j in 0..h {
            for i in 0..w {
                let (x, y) = (sx + i, sy + j);
                if x >= 0 && y >= 0 && (x as usize) < FB_W && (y as usize) < FB_H {
                    src[(j * w + i) as usize] = pix(self.vram[y as usize * FB_W + x as usize]);
                }
            }
        }
        let mask = ((self.ctx.reg[REG_AUXMASK] & 0xff) << 24) | self.ctx.reg[REG_PIXMASK];
        for j in 0..h * zy {
            for i in 0..w * zx {
                let (x, y) = (dx + i, dy + j);
                if !self.clip(x, y) {
                    continue;
                }
                let off = y as usize * FB_W + x as usize;
                if self.wid_ok(IR_WRITEBUF, off) {
                    self.store(off, src[((j / zy) * w + i / zx) as usize], mask);
                }
            }
        }
    }

    /// Emulator-private screen-to-screen copy of the planes selected by
    /// PIXMASK/AUXMASK. Linear coordinates, y bottom-up; overlap-safe.
    pub fn copy_rect(&mut self, sx: i32, sy: i32, w: i32, h: i32, dx: i32, dy: i32) {
        if w <= 0 || h <= 0 {
            return;
        }
        let pixels = (w as u64).saturating_mul(h as u64);
        self.telemetry.snapshot.framebuffer_read_words = self
            .telemetry
            .snapshot
            .framebuffer_read_words
            .saturating_add(pixels.saturating_mul(2));
        self.telemetry.snapshot.framebuffer_write_words = self
            .telemetry
            .snapshot
            .framebuffer_write_words
            .saturating_add(pixels);
        let mask = ((self.ctx.reg[REG_AUXMASK] & 0xff) << 24) | self.ctx.reg[REG_PIXMASK];
        // Walk away from the overlap: bottom-up rows when moving down in
        // memory, right-to-left columns when moving right.
        for jj in 0..h {
            let j = if dy > sy { h - 1 - jj } else { jj };
            let (ys, yd) = (sy + j, dy + j);
            if ys < 0 || yd < 0 || ys as usize >= FB_H || yd as usize >= FB_H {
                continue;
            }
            for ii in 0..w {
                let i = if dx > sx { w - 1 - ii } else { ii };
                let (xs, xd) = (sx + i, dx + i);
                if xs < 0 || xs as usize >= FB_W || !self.clip(xd, yd) {
                    continue;
                }
                let src = self.vram[ys as usize * FB_W + xs as usize];
                let off = yd as usize * FB_W + xd as usize;
                let dst = self.vram[off];
                self.vram[off] = (dst & !mask) | (src & mask);
            }
        }
    }
}

#[inline]
fn sext(v: u32, bits: u32) -> i64 {
    let s = 64 - bits;
    ((v as i64) << s) >> s
}

/// 16 raster ops, X11/GL numbering (RE3.h).
#[inline]
pub fn rop(func: u32, s: u32, d: u32) -> u32 {
    match func & 0xf {
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
        _ => !0,
    }
}

/// Extract pixel `n` of a packed RWDATA word (UPACMODE 0 = 32, 1 = 2x16, 3 = 4x8).
#[inline]
fn unpack(data: u32, n: u32, mode: u32) -> u32 {
    match mode {
        1 => (data >> (16 * (1 - n))) & 0xffff,
        3 => (data >> (8 * (3 - n))) & 0xff,
        _ => data,
    }
}

/// 4-bit 1:2:1 (R 3, G 2:1, B 0) to 8:8:8 values [R, G, B].
pub fn rgb121_to_888(v: u32) -> [u32; 3] {
    [((v >> 3) & 1) * 255, ((v >> 1) & 3) * 85, (v & 1) * 255]
}

/// 3:3:2 (R 7:5, B 4:3, G 2:0) to 8 bits a component by bit replication,
/// which is also GL's rounded fixed-point conversion for every 3- and
/// 2-bit value (4 -> 0x92, not 0x91).
pub fn rgb332_to_888(v: u32) -> [u32; 3] {
    let c3 = |c: u32| (c << 5) | (c << 2) | (c >> 1);
    [c3((v >> 5) & 7), c3(v & 7), ((v >> 3) & 3) * 0x55]
}
