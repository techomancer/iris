//! GR2 introspection: monitor commands (`gr2 ...`, `re3 ...`) and the
//! annotated FIFO trace (`gr2 trace <file> [hq|re3|all]`).
//!
//! Trace lines look like
//!   `  12.345678 HQ  #000123 fifo[0x198 PUC_DRAWCHAR] = 0x00000008`
//!   `  12.345690 HQ  #000147 exec PUC_DRAWCHAR 8x16 mode=1 ...`
//!   `  12.345702 RE3 #004410 hq  NUMPIX = 0x8`
//!   `  12.345705 RE3 #004411 hq  IR = 0x2  -> FLAT x=100 y=200 n=8 ...`
//! with the time in seconds since the trace started. Written from the HQ2 and
//! RE3 threads, so tracing slows the pipeline; that's fine for debugging.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::hq2::{index_label, HQ_TOKEN_GEDMA};
use super::re3::{self, describe_ir, REG_NAMES, RE3_SRC_HQ};
use super::{gr2comp, vc1, Gr2, TP_PROBE_ID};

pub const TRACE_HQ: u32 = 1;
pub const TRACE_RE3: u32 = 2;
/// CPU register/memory accesses outside the two FIFOs.
pub const TRACE_CPU: u32 = 4;

/// Pending run of similar CPU accesses, collapsed into one line.
struct CpuRun {
    write: bool,
    /// Region kind; bulk regions merge any offsets, registers merge only
    /// identical (offset, value) repeats.
    kind: &'static str,
    bulk: bool,
    first: u32,
    last: u32,
    val: u32,
    count: u64,
}

/// Trace sink state; lives in `Gr2::trace` behind a mutex.
pub struct Gr2Trace {
    out: Option<BufWriter<File>>,
    path: String,
    start: Option<Instant>,
    hq_seq: u64,
    re3_seq: u64,
    cpu_seq: u64,
    run: Option<CpuRun>,
    last_flush: Option<Instant>,
}

impl Gr2Trace {
    pub const fn new() -> Self {
        Self { out: None, path: String::new(), start: None, hq_seq: 0, re3_seq: 0, cpu_seq: 0, run: None, last_flush: None }
    }

    /// Emit the pending collapsed CPU run, if any.
    fn close_run(&mut self) {
        let Some(r) = self.run.take() else { return };
        let dir = if r.write { "wr" } else { "rd" };
        let seq = self.cpu_seq;
        if r.count == 1 {
            let d = region_detail(r.first);
            self.line(format_args!("CPU #{:06} {dir} {d} = {:#010x}", seq, r.val));
        } else if r.bulk {
            self.line(format_args!("CPU #{:06} {dir} {} x{} ({} .. {}) last={:#010x}",
                seq, r.kind, r.count, region_detail(r.first), region_detail(r.last), r.val));
        } else {
            let d = region_detail(r.first);
            self.line(format_args!("CPU #{:06} {dir} {d} = {:#010x}  (x{})", seq, r.val, r.count));
        }
    }

    fn line(&mut self, text: std::fmt::Arguments) {
        let t = self.start.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
        if let Some(f) = self.out.as_mut() {
            let _ = writeln!(f, "{t:12.6} {text}");
        }
    }
}

impl Gr2 {
    #[inline]
    pub(super) fn tracing(&self, what: u32) -> bool {
        self.trace_mask.load(Ordering::Relaxed) & what != 0
    }

    /// One CPU access outside the FIFOs (board offset, value read/written).
    pub(super) fn trace_cpu(&self, write: bool, off: u32, val: u32) {
        let (kind, bulk) = region_kind(off);
        let mut t = self.trace.lock();
        if let Some(r) = t.run.as_mut() {
            let same = r.write == write && r.kind == kind
                && if bulk { true } else { r.first == off && r.val == val };
            if same {
                r.count += 1;
                r.last = off;
                r.val = val;
                return;
            }
        }
        t.close_run();
        t.cpu_seq += 1;
        t.run = Some(CpuRun { write, kind, bulk, first: off, last: off, val, count: 1 });
    }

    /// Flush the trace file (called ~2x/s from the display thread).
    pub(super) fn trace_flush(&self, force: bool) {
        if self.trace_mask.load(Ordering::Relaxed) == 0 && !force {
            return;
        }
        let mut t = self.trace.lock();
        let due = t.last_flush.map_or(true, |l| l.elapsed().as_millis() >= 500);
        if force || due {
            t.close_run();
            if let Some(f) = t.out.as_mut() {
                let _ = f.flush();
            }
            t.last_flush = Some(Instant::now());
        }
    }

    /// Free-form annotation line (e.g. decoded PLL programming).
    pub(super) fn trace_note(&self, text: &str) {
        let mut t = self.trace.lock();
        t.close_run();
        t.line(format_args!("NOTE {text}"));
    }

    /// One HQ FIFO entry, as consumed by the HQ2 thread.
    pub(super) fn trace_hq_entry(&self, index: u32, val: u32) {
        let mut t = self.trace.lock();
        t.close_run();
        t.hq_seq += 1;
        let seq = t.hq_seq;
        let name = index_label(index);
        if index == super::hq2::HQ_TOKEN_UNSTALL {
            t.line(format_args!("HQ  #{seq:06} UNSTALL (microcode restart)"));
        } else if index == HQ_TOKEN_GEDMA {
            t.line(format_args!("HQ  #{seq:06} GEDMA data = {val:#010x}"));
        } else {
            t.line(format_args!("HQ  #{seq:06} fifo[{index:#05x} {name}] = {val:#010x}"));
        }
    }

    /// A command the HQ2 interpreter just completed, decoded.
    pub(super) fn trace_hq_exec(&self, desc: &str) {
        let mut t = self.trace.lock();
        let seq = t.hq_seq;
        t.line(format_args!("HQ  #{seq:06} exec {desc}"));
    }

    /// One RE3 FIFO entry, before it is applied (so `IR` decodes the
    /// register state the primitive will use).
    pub(super) fn trace_re3(&self, addr: u32, val: u64, regs: &[u32; 64]) {
        let mut t = self.trace.lock();
        t.close_run();
        t.re3_seq += 1;
        let seq = t.re3_seq;
        match addr {
            re3::RE3_OP_COPY_A => {
                let f = |s: u32| (val >> s) as u16 as i16;
                t.line(format_args!("RE3 #{seq:06} op  COPYRECT src=({}, {}) size={}x{}", f(0), f(16), f(32), f(48)));
            }
            re3::RE3_OP_COPY_B => {
                let f = |s: u32| (val >> s) as u16 as i16;
                t.line(format_args!("RE3 #{seq:06} op  COPYRECT dst=({}, {})  [GL y, bottom-up]", f(0), f(16)));
            }
            re3::RE3_OP_READ_ADVANCE => t.line(format_args!("RE3 #{seq:06} op  READBUF advance")),
            re3::RE3_OP_PIXFMT => t.line(format_args!("RE3 #{seq:06} op  PIXFMT {}", match val { 1 => "RGB12", 2 => "CI12", _ => "native" })),
            re3::RE3_OP_ZCTL => t.line(format_args!("RE3 #{seq:06} op  ZCTL test={} func={} zmask={:#08x}", val & 1, (val >> 1) & 7, (val >> 8) & 0xff_ffff)),
            re3::RE3_OP_STENCIL => t.line(format_args!("RE3 #{seq:06} op  STENCIL on={} func={} ref={} mask={:#x} wmask={:#x} ops fail={} zfail={} zpass={}",
                val & 1, (val >> 1) & 7, (val >> 4) & 0xff, (val >> 12) & 0xff, (val >> 20) & 0xff, (val >> 28) & 0xf, (val >> 32) & 0xf, (val >> 36) & 0xf)),
            re3::RE3_OP_BLEND => t.line(format_args!("RE3 #{seq:06} op  BLEND on={} src={} dst={}", val & 1, (val >> 1) & 7, (val >> 4) & 7)),
            re3::RE3_OP_ALPHA => t.line(format_args!("RE3 #{seq:06} op  ALPHA start={:.3} step={:.5}", (val as u32) as f32 / 2048.0, ((val >> 32) as u32 as i32) as f32 / 2048.0)),
            re3::RE3_OP_ZFILL_A => t.line(format_args!("RE3 #{seq:06} op  ZFILL rect ({}, {})-({}, {})", val & 0xffff, (val >> 16) & 0xffff, (val >> 32) & 0xffff, val >> 48)),
            re3::RE3_OP_ZFILL_B => t.line(format_args!("RE3 #{seq:06} op  ZFILL value={:#010x} mask={:#010x}", val as u32, val >> 32)),
            _ => {
                let src = if addr & RE3_SRC_HQ != 0 { "hq " } else { "cpu" };
                let reg = (addr & !RE3_SRC_HQ) as usize & 63;
                let v = val as u32;
                let extra = match reg {
                    re3::REG_IR => format!("  -> {}", describe_ir(regs, v)),
                    re3::REG_X | re3::REG_XMIN | re3::REG_XMAX => format!("  (x={})", v),
                    re3::REG_YX => format!("  (x={} y={})", v & 0xfff, (v >> 12) & 0x7ff),
                    re3::REG_R | re3::REG_G | re3::REG_B => format!("  ({})", v >> 11),
                    re3::REG_RWMODE => format!("  ({})", re3::rwmode_name(v)),
                    _ => String::new(),
                };
                t.line(format_args!("RE3 #{seq:06} {src} {} = {v:#x}{extra}", REG_NAMES[reg]));
            }
        }
    }

    // ── monitor commands ─────────────────────────────────────────────────────

    pub(super) fn cmd_gr2(&self, args: &[&str], w: &mut dyn Write) -> std::io::Result<()> {
        let r = self.regs();
        let num = |i: usize, d: u32| -> u32 {
            args.get(i).and_then(|s| {
                let s = s.trim_start_matches("0x");
                u32::from_str_radix(s, if args[i].starts_with("0x") { 16 } else { 10 }).ok()
            }).unwrap_or(d)
        };
        match args.first().copied().unwrap_or("status") {
            "status" => self.gr2_status(w)?,
            "hq" => {
                let h = &r.hq;
                writeln!(w, "HQ2: running={} gepc={:#06x} loaducode={:#010x} numge={:#x} refresh={:#x}",
                    h.running, h.gepc, h.loaducode, h.numge, h.refresh)?;
                writeln!(w, "  fifo_full={} fifo_empty={} timeouts full={} empty={}  intr={:#x} dmasync={:#x}",
                    h.fifo_full, h.fifo_empty, h.fifo_full_timeout, h.fifo_empty_timeout, h.intr, h.dmasync)?;
                let f = |i: usize| h.fin[i].load(Ordering::Relaxed);
                writeln!(w, "  fin1..3 = {:#x} {:#x} {:#x}  version={:#010x}", f(0), f(1), f(2), h.read(super::hq2::HQ_VERSION))?;
                let aj: Vec<String> = h.attrjmp.iter().map(|v| format!("{v:x}")).collect();
                writeln!(w, "  attrjmp: {}", aj.join(" "))?;
                // SAFETY: read-only peek at the HQ2 thread's state (may tear).
                let eng = unsafe { &*self.hq_engine.get() };
                writeln!(w, "  interpreter: {}", eng.summary())?;
                writeln!(w, "  fifo: {} pending, busy={}", self.hq_fifo.len(), self.hq_busy.load(Ordering::Relaxed))?;
            }
            "ge" => {
                let ge = num(1, 0) as usize & 7;
                let start = num(2, 0) as usize & 0xff;
                let n = (num(3, 64) as usize).min(256 - start);
                writeln!(w, "GE{ge} ({} installed) ram0[{start:#x}..{:#x}]:", r.ge.count, start + n)?;
                hexdump(w, start, &r.ge.win[ge][start..start + n])?;
            }
            "shram" => {
                let start = num(1, 0) as usize & 0x7fff;
                let n = (num(2, 64) as usize).min(0x8000 - start);
                hexdump(w, start, &r.shram[start..start + n])?;
            }
            "ucode" => {
                let start = num(1, 0x191) as usize & 0x1fff;
                let n = (num(2, 16) as usize).min(0x2000 - start);
                hexdump(w, start, &r.hq.ucode[start..start + n])?;
            }
            "vc1" => {
                let v = &r.vc1;
                if args.get(1) == Some(&"sram") {
                    let start = num(2, 0) as usize & 0xfffe;
                    let n = (num(3, 64) as usize).min((0x10000 - start) / 2);
                    let words: Vec<u32> = (0..n).map(|i| v.sram16(start + i * 2) as u32).collect();
                    return hexdump16(w, start, &words);
                }
                let (cx, cy) = v.cursor_pos();
                writeln!(w, "VC1: addr={:#06x} sysctl={:#04x} [{}{}{}{}{}]",
                    v.addr, v.sysctl,
                    if v.sysctl & 0x04 != 0 { "VC1 " } else { "" },
                    if v.sysctl & 0x08 != 0 { "DID " } else { "" },
                    if v.sysctl & 0x10 != 0 { "CURSOR " } else { "" },
                    if v.sysctl & 0x20 != 0 { "CURDISP " } else { "" },
                    if v.sysctl & 0x80 != 0 { "VIDEO" } else { "" })?;
                writeln!(w, "  VID_EP={:#06x} VID_ENAB={:#06x}", v.reg16(vc1::VID_EP), v.reg16(0x14))?;
                writeln!(w, "  CUR_EP={:#06x} CUR_XL={} CUR_YL={} CUR_MODE={:#04x} -> screen ({cx}, {cy}) visible={}",
                    v.reg16(vc1::CUR_EP), v.reg16(vc1::CUR_XL), v.reg16(vc1::CUR_YL), v.regs[0x26], v.cursor_visible())?;
                writeln!(w, "  DID_EP={:#06x} DID_END={:#06x} HOR={:#06x}  line0 DID={} line1023 DID={}",
                    v.reg16(vc1::DID_EP), v.reg16(0x42), v.reg16(0x44), v.line_did(0), v.line_did(1023))?;
                writeln!(w, "  BLKOUT={:#06x}", v.reg16(0x60))?;
            }
            "xmap" => {
                let ch = num(1, 0) as usize % 5;
                let x = &r.xmap[ch];
                writeln!(w, "XMAP5 #{ch}: addr={:#06x} misc={:02x?}", x.addr, x.misc)?;
                for (did, &m) in x.mode.iter().enumerate() {
                    if m != 0 || did == 0 {
                        writeln!(w, "  DID {did:2}: {m:#010x}  {}", decode_mode(m))?;
                    }
                }
            }
            "clut" => {
                let start = num(1, 0x1000) as usize & 0x1fff;
                let n = (num(2, 16) as usize).min(0x2000 - start);
                let ch = num(3, 0) as usize % 5;
                for i in start..start + n {
                    let c = r.xmap[ch].clut[i];
                    writeln!(w, "  [{i:#06x}] r={:3} g={:3} b={:3}", c >> 16, (c >> 8) & 0xff, c & 0xff)?;
                }
            }
            "dac" => {
                for (name, d) in ["red", "green", "blue"].iter().zip(r.dac.iter()) {
                    let ident = d.palette.iter().enumerate().all(|(i, &v)| v as usize == i);
                    writeln!(w, "  {name:5}: addr={:#04x} readmask={:#04x} blink={:#04x} cmd={:#04x} test={:#04x} ramp={} overlay={:02x?}",
                        d.addr, d.readmask, d.blinkmask, d.cmd, d.test, if ident { "identity" } else { "custom" }, d.overlay)?;
                }
            }
            "pix" => {
                // gr2 pix x y   (display coordinates: y = 0 is the top row)
                let (x, y) = (num(1, 0) as usize, num(2, 0) as usize);
                if x >= re3::FB_W || y >= re3::FB_H {
                    return writeln!(w, "out of range");
                }
                let p = self.vram()[(re3::FB_H - 1 - y) * re3::FB_W + x];
                let did = r.vc1.did_at(x, y) as usize;
                let xm = &r.xmap[x % 5];
                let rgb = gr2comp::pixel_rgb(p, xm.mode[did], xm);
                writeln!(w, "({x}, {y}) [vram row {}]: word={p:#010x} cid={} aux={:#x} pixel={:#08x}  DID={did} mode={:#010x} -> rgb={rgb:06x}",
                    re3::FB_H - 1 - y, p >> 28, (p >> 24) & 0xf, p & 0xffffff, xm.mode[did])?;
            }
            "trace" => return self.cmd_trace(&args[1..], w),
            "fbdump" => {
                let dir = std::path::PathBuf::from(args.get(1).copied().unwrap_or("gr2dump"));
                match self.dump_framebuffer(&dir) {
                    Ok(()) => writeln!(w, "framebuffer dumped to {}/ (screen rgb ci aux cid .png, vram.bin z.bin)", dir.display())?,
                    Err(e) => writeln!(w, "fbdump failed: {e}")?,
                }
            }
            _ => {
                writeln!(w, "gr2 [status]                   board summary")?;
                writeln!(w, "gr2 hq                         HQ2 registers + interpreter state")?;
                writeln!(w, "gr2 ge <n> [start] [count]     GE7 window words")?;
                writeln!(w, "gr2 shram [start] [count]      shared RAM words")?;
                writeln!(w, "gr2 ucode [start] [count]      HQ2 microcode RAM")?;
                writeln!(w, "gr2 vc1 | vc1 sram <a> [n]     VC1 registers / SRAM (byte address)")?;
                writeln!(w, "gr2 xmap [ch]                  XMAP5 mode table + misc")?;
                writeln!(w, "gr2 clut [start] [n] [ch]      CLUT entries")?;
                writeln!(w, "gr2 dac                        Bt457 state")?;
                writeln!(w, "gr2 pix <x> <y>                pixel at display (x, y), through the XMAP")?;
                writeln!(w, "gr2 fbdump [dir]               VRAM planes + composed screen to PNG/raw files")?;
                writeln!(w, "gr2 trace <file> [hq,re3,cpu|all]  capture annotated traffic; gr2 trace flush|off")?;
                writeln!(w, "re3 [regs] | re3 pix <x> <y> [w] [h]")?;
            }
        }
        Ok(())
    }

    /// `gr2 fbdump`: raw VRAM/Z plus PNG views of each plane group and of the
    /// composed screen. PNGs are top-down (display orientation).
    pub(super) fn dump_framebuffer(&self, dir: &std::path::Path) -> std::io::Result<()> {
        use re3::{FB_H, FB_W};
        std::fs::create_dir_all(dir)?;
        // SAFETY: read-only views; tearing tolerated.
        let re = unsafe { &*self.re3.get() };
        let be = |words: &[u32]| words.iter().flat_map(|w| w.to_be_bytes()).collect::<Vec<u8>>();
        std::fs::write(dir.join("vram.bin"), be(&re.vram))?;
        std::fs::write(dir.join("z.bin"), be(&re.zbuf))?;

        // Rows in display order (VRAM is bottom-up).
        let rows = || (0..FB_H).rev().map(|y| &re.vram[y * FB_W..(y + 1) * FB_W]);
        let mut rgb = Vec::with_capacity(FB_W * FB_H * 3);
        let (mut ci, mut aux, mut cid) = (Vec::with_capacity(FB_W * FB_H), Vec::with_capacity(FB_W * FB_H), Vec::with_capacity(FB_W * FB_H));
        for row in rows() {
            for &p in row {
                rgb.extend_from_slice(&[p as u8, (p >> 8) as u8, (p >> 16) as u8]);
                ci.push(p as u8);
                aux.push((((p >> 24) & 0xf) * 17) as u8);
                cid.push(((p >> 28) * 17) as u8);
            }
        }
        write_png(&dir.join("rgb.png"), &rgb, png::ColorType::Rgb)?;
        write_png(&dir.join("ci.png"), &ci, png::ColorType::Grayscale)?;
        write_png(&dir.join("aux.png"), &aux, png::ColorType::Grayscale)?;
        write_png(&dir.join("cid.png"), &cid, png::ColorType::Grayscale)?;

        let r = self.regs();
        let mut out = vec![0u32; gr2comp::OUT_STRIDE * FB_H];
        gr2comp::compose(&re.vram, &r.vc1, &r.xmap, &r.dac, &mut out);
        let mut scr = Vec::with_capacity(FB_W * FB_H * 3);
        for y in 0..FB_H {
            for &p in &out[y * gr2comp::OUT_STRIDE..y * gr2comp::OUT_STRIDE + FB_W] {
                // Screen format 0xAABBGGRR (R in the low byte).
                scr.extend_from_slice(&[p as u8, (p >> 8) as u8, (p >> 16) as u8]);
            }
        }
        write_png(&dir.join("screen.png"), &scr, png::ColorType::Rgb)
    }

    pub(super) fn gr2_status(&self, w: &mut dyn Write) -> std::io::Result<()> {
        let r = self.regs();
        let v = self.variant;
        writeln!(w, "GR2 {} ({} GE7, board rev {})", v.name(), v.ges(), v.board_rev())?;
        writeln!(w, "  hq: running={} gepc={:#06x} numge={:#x}  shram[TP_PROBE_ID]={}",
            r.hq.running, r.hq.gepc, r.hq.numge, r.shram[TP_PROBE_ID])?;
        writeln!(w, "  fifo: hq {}  re3 {}  idle={}", self.hq_fifo.len(), self.re3_fifo.len(), self.idle())?;
        writeln!(w, "  vc1: sysctl={:#04x} vid_ep={:#06x} did_ep={:#06x}  xmap0.mode[0]={:#010x}",
            r.vc1.sysctl, r.vc1.reg16(vc1::VID_EP), r.vc1.reg16(vc1::DID_EP), r.xmap[0].mode[0])?;
        writeln!(w, "  dac readmask r/g/b = {:#04x}/{:#04x}/{:#04x}",
            r.dac[0].readmask, r.dac[1].readmask, r.dac[2].readmask)?;
        if r.pll_programs > 0 {
            writeln!(w, "  clock: {} (programmed {} times)", pll_describe(&r.pll_last), r.pll_programs)?;
        }
        let t = self.trace.lock();
        let mask = self.trace_mask.load(Ordering::Relaxed);
        if mask != 0 {
            writeln!(w, "  trace: {} ({}{}{}) hq={} re3={} cpu={} entries", t.path,
                if mask & TRACE_HQ != 0 { "hq " } else { "" }, if mask & TRACE_RE3 != 0 { "re3 " } else { "" },
                if mask & TRACE_CPU != 0 { "cpu" } else { "" }, t.hq_seq, t.re3_seq, t.cpu_seq)?;
        }
        Ok(())
    }

    fn cmd_trace(&self, args: &[&str], w: &mut dyn Write) -> std::io::Result<()> {
        match args.first().copied() {
            Some("flush") => {
                self.trace_flush(true);
                writeln!(w, "trace flushed")?;
            }
            None | Some("off") => {
                self.trace_mask.store(0, Ordering::SeqCst);
                let mut t = self.trace.lock();
                t.close_run();
                if let Some(mut f) = t.out.take() {
                    let _ = f.flush();
                    writeln!(w, "trace closed: {} (hq {} / re3 {} entries)", t.path, t.hq_seq, t.re3_seq)?;
                } else {
                    writeln!(w, "trace not active")?;
                }
            }
            Some(path) => {
                // Categories: hq, re3, cpu, or combinations like hq,cpu; default all.
                let mut mask = 0;
                for c in args.get(1).copied().unwrap_or("all").split(',') {
                    mask |= match c {
                        "hq" => TRACE_HQ,
                        "re3" => TRACE_RE3,
                        "cpu" => TRACE_CPU,
                        _ => TRACE_HQ | TRACE_RE3 | TRACE_CPU,
                    };
                }
                let f = match File::create(path) {
                    Ok(f) => f,
                    Err(e) => return writeln!(w, "cannot create {path}: {e}"),
                };
                let mut t = self.trace.lock();
                t.out = Some(BufWriter::with_capacity(1 << 20, f));
                t.path = path.to_string();
                t.start = Some(Instant::now());
                t.hq_seq = 0;
                t.re3_seq = 0;
                t.cpu_seq = 0;
                t.run = None;
                t.line(format_args!("GR2 {} trace started (mask {mask:#x})", self.variant.name()));
                drop(t);
                self.trace_mask.store(mask, Ordering::SeqCst);
                writeln!(w, "tracing to {path}")?;
            }
        }
        Ok(())
    }

    pub(super) fn cmd_re3(&self, args: &[&str], w: &mut dyn Write) -> std::io::Result<()> {
        // SAFETY: read-only peek at the RE3 thread's state (may tear while drawing).
        let re = unsafe { &*self.re3.get() };
        let num = |i: usize, d: u32| args.get(i).and_then(|s| s.parse::<u32>().ok()).unwrap_or(d);
        match args.first().copied().unwrap_or("regs") {
            "regs" => {
                let c = &re.ctx;
                writeln!(w, "RE3: fifo {} pending, busy={}  stream={}  numpix left={}",
                    self.re3_fifo.len(), self.re3_busy.load(Ordering::Relaxed),
                    ["idle", "READ", "WRITE"].get(c.stream as usize).unwrap_or(&"?"), c.numpix)?;
                writeln!(w, "  iterators: x={:.3} y={:.3} r={:.3} g={:.3} b={:.3} z={:#x}",
                    c.x as f64 / 16384.0, c.y as f64 / 16384.0,
                    c.r as f64 / 2048.0, c.g as f64 / 2048.0, c.b as f64 / 2048.0, c.z)?;
                for (i, chunk) in c.reg.chunks(4).enumerate() {
                    let mut line = String::new();
                    for (j, v) in chunk.iter().enumerate() {
                        let n = i * 4 + j;
                        line += &format!("  {:02x} {:<9} {:#010x}", n, REG_NAMES[n], v);
                    }
                    writeln!(w, "{line}")?;
                }
                let g = |n: usize| c.reg[n];
                writeln!(w, "  decoded: X={} Y={} scissor x {}..{} y {}..{} rwmode={} ir={}",
                    ((g(re3::REG_X)) as i32), g(re3::REG_Y),
                    ((g(re3::REG_XMIN)) as i32), ((g(re3::REG_XMAX)) as i32), g(re3::REG_YMIN), g(re3::REG_YMAX),
                    re3::rwmode_name(g(re3::REG_RWMODE)), re3::ir_name(g(re3::REG_IR)))?;
            }
            "pix" => {
                // re3 pix x y [w] [h]   RE3 coordinates: linear x, y = 0 at the bottom
                let (x0, y0) = (num(1, 0) as usize, num(2, 0) as usize);
                let (wd, ht) = (num(3, 8) as usize, num(4, 1) as usize);
                for y in (y0..(y0 + ht).min(re3::FB_H)).rev() {
                    let row: Vec<String> = (x0..(x0 + wd).min(re3::FB_W))
                        .map(|x| format!("{:08x}", re.vram[y * re3::FB_W + x])).collect();
                    writeln!(w, "  y={y:4} x={x0:4}: {}", row.join(" "))?;
                }
            }
            _ => writeln!(w, "re3 [regs] | re3 pix <x> <y> [w] [h]   (y = 0 is the bottom row)")?,
        }
        Ok(())
    }
}

fn decode_mode(m: u32) -> String {
    let pd = ["CI", "RGB", "CIMM", "BLACKOUT"][((m >> 16) & 3) as usize];
    let pix = ["4/0", "4/1", "8/0", "8/1", "12/0", "12/1", "16", "24"][((m >> 24) & 7) as usize];
    format!("{pd} {pix} pix_pg={} olayen={:#x} aux_pg={} pou_mode={} pou_pg={}",
        m >> 27, (m >> 20) & 0xf, (m >> 9) & 0x1f, (m >> 5) & 3, m & 0x1f)
}

fn hexdump(w: &mut dyn Write, start: usize, words: &[u32]) -> std::io::Result<()> {
    for (i, chunk) in words.chunks(8).enumerate() {
        let line: Vec<String> = chunk.iter().map(|v| format!("{v:08x}")).collect();
        writeln!(w, "  {:#06x}: {}", start + i * 8, line.join(" "))?;
    }
    Ok(())
}

fn hexdump16(w: &mut dyn Write, start: usize, words: &[u32]) -> std::io::Result<()> {
    for (i, chunk) in words.chunks(8).enumerate() {
        let line: Vec<String> = chunk.iter().map(|v| format!("{v:04x}")).collect();
        writeln!(w, "  {:#06x}: {}", start + i * 16, line.join(" "))?;
    }
    Ok(())
}

/// Region of a board offset for CPU tracing: (kind, bulk). Bulk regions are
/// memories whose runs of accesses collapse into one line.
fn region_kind(off: u32) -> (&'static str, bool) {
    match off {
        0x00000..0x20000 => ("shram", !matches!(off >> 2, 0x7fff | 0x7ff8 | 0x302 | 0x304 | 0x4822..=0x4825)),
        0x60000..0x68000 => ("hqucode", true),
        0x68000..0x6a000 => {
            let w = (off >> 2) & 0xff;
            if (0xf8..=0xfb).contains(&w) { ("ge.stage", true) } else { ("ge", w != 0xfd) }
        }
        0x6a000..0x6a200 => ("hq", false),
        0x6c000..0x6c020 => ("bdvers", false),
        0x6c020..0x6c040 => ("clock", true),
        0x6c040..0x6c060 => if off & 0x1c == 0x08 { ("vc1.sram", true) } else { ("vc1", false) },
        0x6c0a0..0x6c100 => if off & 0x1c == 0x04 { ("dac.palette", true) } else { ("dac", false) },
        0x6c100..0x6c1c0 => if off & 0x1c == 0x08 { ("xmap.clut", true) } else { ("xmap", false) },
        0x6c1c0..0x6c200 => ("ab1/cc1", false),
        0x6c200..0x6c300 | 0x6c600..0x6c680 => ("re3", false),
        0x6b000 => ("fin3", false),
        _ => ("other", false),
    }
}

/// Human-readable name for a board offset (register names where known).
fn region_detail(off: u32) -> String {
    const HQ: [(u32, &str); 18] = [
        (0x40, "version"), (0x44, "numge"), (0x48, "fin1"), (0x4c, "fin2"), (0x50, "dmasync"),
        (0x54, "fifo_full_timeout"), (0x58, "fifo_empty_timeout"), (0x5c, "fifo_full"),
        (0x60, "fifo_empty"), (0x64, "ge7loaducode"), (0x68, "gedma"), (0x6c, "hq_gepc"),
        (0x70, "gepc"), (0x74, "intr"), (0x78, "unstall"), (0x7c, "mystery"), (0x80, "refresh"),
        (0x100, "fin3"),
    ];
    const VC1: [&str; 8] = ["cmd0", "cmd1", "sram", "testreg", "addrlo", "addrhi", "sysctl", "pad"];
    const DAC: [&str; 4] = ["addr", "palette", "ctrl", "overlay"];
    const XMAP: [&str; 8] = ["misc", "mode", "clut", "crc", "addrlo", "addrhi", "bytecnt", "fifostatus"];
    match off {
        0x00000..0x20000 => {
            let w = off >> 2;
            let tag = match w {
                0x7fff => " TP_PROBE_ID",
                0x7ff8 => " (info+0x32)",
                0x302 => " CX_SIZE_MAIN",
                0x304 => " CX_SIZE_EXT",
                0x4822..=0x4825 => " READBACK",
                _ => "",
            };
            format!("shram[{w:#06x}]{tag}")
        }
        0x60000..0x68000 => format!("hqucode[{:#06x}]", (off - 0x60000) >> 2),
        0x68000..0x6a000 => format!("ge[{}].ram0[{:#04x}]", (off - 0x68000) >> 10, (off >> 2) & 0xff),
        0x6a000..0x6a040 => format!("hq.attrjmp[{}]", (off - 0x6a000) >> 2),
        0x6a040..0x6a200 => {
            let o = off - 0x6a000;
            HQ.iter().find(|(a, _)| *a == o).map(|(_, n)| format!("hq.{n}")).unwrap_or(format!("hq[{o:#x}]"))
        }
        0x6c000..0x6c020 => format!("bdvers.rd{}", (off - 0x6c000) >> 2),
        0x6c020..0x6c040 => "clock".into(),
        0x6c040..0x6c060 => format!("vc1.{}", VC1[((off >> 2) & 7) as usize]),
        0x6c0a0..0x6c100 => format!("dac{}.{}", ["R", "G", "B"][((off - 0x6c0a0) >> 5) as usize], DAC[((off >> 2) & 3) as usize]),
        0x6c100..0x6c1a0 => format!("xmap{}.{}", (off - 0x6c100) >> 5, XMAP[((off >> 2) & 7) as usize]),
        0x6c1a0..0x6c1c0 => format!("xmapall.{}", XMAP[((off >> 2) & 7) as usize]),
        0x6c1c0..0x6c1e0 => "ab1".into(),
        0x6c1e0..0x6c200 => "cc1".into(),
        0x6c200..0x6c300 => format!("re3.{}", REG_NAMES[((off - 0x6c200) >> 2) as usize & 63]),
        0x6c600..0x6c680 => "re3.RWDATA32".into(),
        0x6b000 => "fin3_port(0x6b000)".into(),
        _ => format!("[{off:#07x}]"),
    }
}

fn write_png(path: &std::path::Path, data: &[u8], color: png::ColorType) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut enc = png::Encoder::new(BufWriter::new(file), re3::FB_W as u32, re3::FB_H as u32);
    enc.set_color(color);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(std::io::Error::other)?;
    w.write_image_data(data).map_err(std::io::Error::other)
}

/// Decode a recorded PLL programming (7 bytes + bit count in byte 7) and
/// name the tables the PROM/kernel use (GR2.h "Clock").
pub(super) fn pll_describe(b: &[u8; 8]) -> String {
    let bytes = &b[..7];
    let name = match bytes {
        [0x00, 0x10, 0x00, 0x15, 0x18, 0x01, 0x0f] => "60 Hz, 107.352 MHz (PLLclk107)",
        [0x00, 0x00, 0x00, 0x04, 0x05, 0x04, 0x06] => "72 Hz, 132 MHz (PLLclk132)",
        _ => "unknown table",
    };
    let hex: Vec<String> = bytes.iter().map(|v| format!("{v:02x}")).collect();
    format!("[{}] {} bits, {name}", hex.join(" "), b[7])
}
