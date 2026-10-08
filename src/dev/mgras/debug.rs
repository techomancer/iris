//! IMPACT introspection: monitor commands (`mgras ...`, `rss ...`) and the
//! annotated trace (`mgras trace <file> [hq,rss,cpu,tex|all]`).
//!
//! Trace lines look like
//!   `   12.345678 HQ  #000123 u fillmode = 0x100000`
//!   `   12.345690 HQ  #000124 u block_xyendi = 0x1f003ff +exec`
//!   `   12.345702 RSS #004410 hq  block_xyendi = 0x1f003ff  (496, 1023) -> BLOCK (0, 0)-(496, 1023) fast fill ...`
//!   `   12.345705 CPU #000071 rd flag_set = 0x00010000  (x5123)`
//! with the time in seconds since the trace started. HQ lines are command
//! FIFO commands as the HQ3 dispatches them; RSS lines are raster register
//! writes and ops as the RSS applies them (`hq` from the command FIFO, `cpu`
//! from the direct register window); CPU lines are every other board access,
//! with repeats of one register and runs through a memory collapsed. Written
//! from the CPU, HQ3 and RSS threads, so tracing slows the pipeline.
//!
//! `IRIS_MGRAS_TRACE=<file>` starts a full trace at power-on.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::hq3::host;
use super::rss::{self, Rss, REG_NAMES};
use super::{dcb, disp, tag, Mgras, MGRAS_SLOT_GFX_BASE};

pub const TRACE_HQ: u32 = 1;
pub const TRACE_RSS: u32 = 2;
/// CPU accesses outside the command FIFO and the direct raster window.
pub const TRACE_CPU: u32 = 4;
/// One line per texture load (cheap enough for a running game).
pub const TRACE_TEX: u32 = 8;
/// The categories `IRIS_MGRAS_TRACE` starts with.
pub const TRACE_ALL: u32 = TRACE_HQ | TRACE_RSS | TRACE_CPU;

/// Pending run of similar CPU accesses, collapsed into one line.
struct CpuRun {
    write: bool,
    /// Region kind; bulk regions merge any offsets, registers merge only
    /// identical (offset, value) repeats.
    kind: &'static str,
    bulk: bool,
    first: u32,
    last: u32,
    val: u64,
    count: u64,
}

/// Trace sink state; lives in `Mgras::trace` behind a mutex.
pub struct MgrasTrace {
    out: Option<BufWriter<File>>,
    path: String,
    start: Option<Instant>,
    hq_seq: u64,
    rss_seq: u64,
    cpu_seq: u64,
    run: Option<CpuRun>,
    last_flush: Option<Instant>,
}

impl MgrasTrace {
    pub const fn new() -> Self {
        Self { out: None, path: String::new(), start: None, hq_seq: 0, rss_seq: 0, cpu_seq: 0, run: None, last_flush: None }
    }

    /// Emit the pending collapsed CPU run, if any.
    fn close_run(&mut self) {
        let Some(r) = self.run.take() else { return };
        let dir = if r.write { "wr" } else { "rd" };
        let seq = self.cpu_seq;
        if r.count == 1 {
            self.line(format_args!("CPU #{seq:06} {dir} {} = {:#010x}", region_detail(r.first), r.val));
        } else if r.bulk {
            self.line(format_args!("CPU #{seq:06} {dir} {} x{} ({} .. {}) last={:#010x}",
                r.kind, r.count, region_detail(r.first), region_detail(r.last), r.val));
        } else {
            self.line(format_args!("CPU #{seq:06} {dir} {} = {:#010x}  (x{})", region_detail(r.first), r.val, r.count));
        }
    }

    fn line(&mut self, text: std::fmt::Arguments) {
        let t = self.start.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
        if let Some(f) = self.out.as_mut() {
            let _ = writeln!(f, "{t:12.6} {text}");
        }
    }
}

impl Mgras {
    #[inline]
    pub(super) fn tracing(&self, what: u32) -> bool {
        self.trace_mask.load(Ordering::Relaxed) & what != 0
    }

    /// One CPU access outside the FIFOs (board offset, value read/written).
    pub(super) fn trace_cpu(&self, write: bool, off: u32, val: u64) {
        let (kind, bulk) = region_kind(off);
        let mut t = self.trace.lock();
        if let Some(r) = t.run.as_mut() {
            let same = r.write == write && r.kind == kind && if bulk { true } else { r.first == off && r.val == val };
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

    /// One command the HQ3 dispatched (or another frontend event).
    pub(super) fn trace_hq(&self, desc: &str) {
        let mut t = self.trace.lock();
        t.close_run();
        t.hq_seq += 1;
        let seq = t.hq_seq;
        t.line(format_args!("HQ  #{seq:06} {desc}"));
    }

    /// One `rss_fifo` entry, after the RSS applied it (so an executed
    /// primitive decodes with the state it ran with).
    pub(super) fn trace_rss(&self, entry: u32, val: u64, rss: &Rss, changed: bool) {
        let text = match entry {
            tag::PIO_ADVANCE => "op  PIO read advance".to_string(),
            tag::DMA_LINE => format!("op  DMA line {} ({} bytes)", val as u32, val >> 32),
            e if e < 0x1000 => {
                let src = if e & super::RSS_SRC_CPU != 0 { "cpu" } else { "hq " };
                let r = (e & 0x7FF) >> 1;
                let v = val as u32;
                let exec = e & 1 != 0;
                let name = match REG_NAMES.get(r as usize).copied().unwrap_or("") {
                    "" => format!("rss[{r:#x}]"),
                    n => n.to_string(),
                };
                let extra = match r {
                    0x40 | 0x41 | 0x46 | 0x47 | 0x48 => format!("  ({}, {})", v >> 16 & 0xFFFF, v & 0xFFFF),
                    // The window origin and screen masks hold y (or min) in
                    // the high half.
                    0x115 => format!("  (x {}, y {})", v & 0xFFFF, v >> 16),
                    0x147..=0x14E => format!("  ({}..{})", v >> 16, v & 0xFFFF),
                    _ => String::new(),
                };
                let run = if exec { format!(" -> {}", rss.describe_ir()) } else { String::new() };
                format!("{src} {name} = {v:#x}{extra}{run}")
            }
            e => format!("op  {e:#x} = {val:#x}"),
        };
        let mut t = self.trace.lock();
        t.close_run();
        t.rss_seq += 1;
        let seq = t.rss_seq;
        t.line(format_args!("RSS #{seq:06} {text}"));
    }

    /// Flush the trace file (called every frame from the display thread; it
    /// writes out at most twice a second unless forced).
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

    /// Start the annotated trace into `path` for the categories in `mask`.
    pub(super) fn start_trace(&self, path: &str, mask: u32) -> std::io::Result<()> {
        let f = File::create(path)?;
        let mut t = self.trace.lock();
        t.out = Some(BufWriter::with_capacity(1 << 20, f));
        t.path = path.to_string();
        t.start = Some(Instant::now());
        t.hq_seq = 0;
        t.rss_seq = 0;
        t.cpu_seq = 0;
        t.run = None;
        t.line(format_args!("IMPACT {:?} trace started (mask {mask:#x})", self.kind));
        drop(t);
        self.trace_mask.store(mask, Ordering::SeqCst);
        Ok(())
    }

    fn stop_trace(&self) -> Option<String> {
        self.trace_mask.store(0, Ordering::SeqCst);
        let mut t = self.trace.lock();
        t.close_run();
        let mut f = t.out.take()?;
        let _ = f.flush();
        Some(format!("{} (hq {} / rss {} / cpu {} entries)", t.path, t.hq_seq, t.rss_seq, t.cpu_seq))
    }

    // ── monitor commands ─────────────────────────────────────────────────────

    pub(super) fn cmd_mgras(&self, args: &[&str], w: &mut dyn Write) -> Result<(), String> {
        let e = |r: std::io::Result<()>| r.map_err(|e| e.to_string());
        let num = |i: usize, d: u32| -> u32 {
            args.get(i).and_then(|s| match s.strip_prefix("0x") {
                Some(h) => u32::from_str_radix(h, 16).ok(),
                None => s.parse().ok(),
            }).unwrap_or(d)
        };
        match args {
            [] | ["status"] => return e(self.mgras_status(w)),
            ["rec", "off"] => {
                let what = self.submit.lock().as_ref().map(|r| { let r = r.lock(); format!("{} ({} records)", r.path, r.records) });
                self.set_recording(None).map_err(|e| format!("mgras rec: {e}"))?;
                return e(writeln!(w, "recording stopped: {}", what.unwrap_or_else(|| "none running".into())));
            }
            ["rec", "mark"] => {
                let Some((h, n)) = self.record_mark() else { return Err("mgras rec: no recording running".into()) };
                return e(writeln!(w, "checkpoint {} at record {n}", blake3::Hash::from(h).to_hex()));
            }
            ["rec", path] => {
                self.set_recording(Some(path)).map_err(|e| format!("mgras rec: {e}"))?;
                return e(writeln!(w, "recording to {path}"));
            }
            ["trace", "flush"] => {
                self.trace_flush(true);
                return e(writeln!(w, "trace flushed"));
            }
            ["trace"] | ["trace", "off"] => {
                return e(match self.stop_trace() {
                    Some(s) => writeln!(w, "trace closed: {s}"),
                    None => writeln!(w, "trace not active"),
                });
            }
            ["trace", path, rest @ ..] => {
                let mask = trace_mask(rest.first().copied().unwrap_or("all"));
                self.start_trace(path, mask).map_err(|e| format!("mgras trace: cannot create {path}: {e}"))?;
                return e(writeln!(w, "tracing to {path} (mask {mask:#x})"));
            }
            ["dump", path] => {
                let bytes = {
                    let _sub = self.submit.lock();
                    self.wait_idle();
                    self.state_bytes()
                };
                std::fs::write(path, bytes).map_err(|e| format!("mgras dump: {e}"))?;
                return e(writeln!(w, "dumped {path}"));
            }
            ["shot", path] => {
                self.save_shot(path).map_err(|e| format!("mgras shot: {e}"))?;
                return e(writeln!(w, "saved {path}"));
            }
            ["fbdump", rest @ ..] => {
                let dir = std::path::PathBuf::from(rest.first().copied().unwrap_or("mgrasdump"));
                return e(match self.dump_framebuffer(&dir) {
                    Ok(()) => writeln!(w, "framebuffer dumped to {}/ (screen rgb ci overlay .png, fb.bin overlay.bin)", dir.display()),
                    Err(err) => writeln!(w, "fbdump failed: {err}"),
                });
            }
            ["stats"] => return e(self.stats(w)),
            #[cfg(feature = "gr4-jit")]
            ["jit", rest @ ..] => return self.cmd_jit(rest, w),
            _ => {}
        }
        let f = self.front();
        match args[0] {
            "hq" => {
                let flags = self.all_flags(&f);
                e(writeln!(w, "HQ3: flags {flags:#010x} [{}]", flag_names(flags)))?;
                e(writeln!(w, "  flag enable {:#010x}  interrupt enable {:#010x}  GE readback {:#x} {:#x}",
                    f.hq.flag_enable, f.hq.interrupt_enable, f.hq.ge_readback[0], f.hq.ge_readback[1]))?;
                let regs: Vec<String> = sorted(f.hq.regs.iter()).iter().map(|(k, v)| format!("{}={v:#x}", region_detail(*k))).collect();
                e(writeln!(w, "  other registers: {}", regs.join(" ")))?;
                drop(f);
                // SAFETY: read-only peek at the HQ3 thread's state (may tear).
                let eng = unsafe { &*self.eng.get() };
                e(writeln!(w, "  engine: {}", eng.summary()))?;
                let dma: Vec<String> = eng.dma_regs.iter().enumerate().filter(|(_, v)| **v != 0).map(|(i, v)| format!("{i:#x}={v:#x}")).collect();
                e(writeln!(w, "  DMA registers: {}", dma.join(" ")))?;
                let rif: Vec<String> = eng.raster_if_regs.iter().enumerate().filter(|(_, v)| **v != 0).map(|(i, v)| format!("{i:#x}={v:#x}")).collect();
                e(writeln!(w, "  raster interface: {}  formatter {:#x}", rif.join(" "), eng.formatter))?;
                e(writeln!(w, "  fifo: hq {} (busy {})  rss {} (busy {})  idle={}", self.hq_fifo.len(),
                    self.hq_busy.load(Ordering::Relaxed), self.rss_fifo.len(), self.rss_busy.load(Ordering::Relaxed), self.idle()))
            }
            "gl" => {
                drop(f);
                // SAFETY: read-only peek at the HQ3 thread's state (may tear).
                let eng = unsafe { &*self.eng.get() };
                e(writeln!(w, "{}", eng.gl_summary()))
            }
            "ge" => {
                let ge = num(1, 0) as usize & 1;
                let start = num(2, 0) as usize;
                let n = num(3, 16) as usize;
                e(writeln!(w, "GE11 #{ge} diagnostic address {:#x}; microcode lines (72 bits as byte2:word1:word0):", f.ge.diag_addr(ge)))?;
                for i in start..start + n {
                    let Some(l) = f.ge.ucode_line(ge, i) else { break };
                    e(writeln!(w, "  {i:#07x}: {:02x}:{:08x}:{:08x}", l[2] & 0xFF, l[1], l[0]))?;
                }
                Ok(())
            }
            "ucode" => {
                let start = (num(1, 0) as usize).min(f.hq.ucode.len());
                let n = (num(2, 32) as usize).min(f.hq.ucode.len() - start);
                e(hexdump(w, start, &f.hq.ucode[start..start + n]))
            }
            "vc3" => {
                let v = &f.dcb.vc3;
                if args.get(1) == Some(&"sram") {
                    let start = num(2, 0) as usize & 0x7FFF;
                    let n = (num(3, 64) as usize).min(0x8000 - start);
                    let words: Vec<u32> = v.sram[start..start + n].iter().map(|&x| x as u32).collect();
                    return e(hexdump16(w, start, &words));
                }
                e(writeln!(w, "VC3 registers:"))?;
                for (i, chunk) in v.regs.chunks(8).enumerate() {
                    let line: Vec<String> = chunk.iter().map(|r| format!("{r:04x}")).collect();
                    e(writeln!(w, "  {:#04x}: {}", i * 8, line.join(" ")))?;
                }
                e(writeln!(w, "  cursor: {:?}  (x, y, size, glyph)", v.cursor()))?;
                e(writeln!(w, "  display: timing tables {:?}, main DID frame table {} lines", v.timing_size(), v.did_lines()))?;
                let mut runs = Vec::new();
                for y in [0usize, 512, 1023] {
                    v.main_did_runs(y, &mut runs);
                    e(writeln!(w, "  line {y} main DID runs: {runs:?}"))?;
                    v.overlay_did_runs(y, &mut runs);
                    e(writeln!(w, "  line {y} overlay DID runs: {runs:?}"))?;
                }
                Ok(())
            }
            "xmap" => {
                for did in 0..32u32 {
                    let (m, o) = (f.dcb.xmap.main_mode(did), f.dcb.xmap.overlay_mode(did));
                    if m != 0 || o != 0 || did == 0 {
                        e(writeln!(w, "  DID {did:2}: main {m:#010x} ({})  overlay {o:#010x}", decode_mode(m)))?;
                    }
                }
                e(writeln!(w, "  cursor colormap base {:#x}", f.dcb.xmap.cursor_cmap_base()))?;
                let regs: Vec<String> = sorted(f.dcb.xmap.regs()).iter().map(|(k, v)| format!("sel{}[{:#x}]={v:#x}", k >> 24, k & 0xFF_FFFF)).collect();
                e(writeln!(w, "  register files: {}", regs.join(" ")))
            }
            "cmap" => {
                let start = num(1, 0) as usize & 0x1FFF;
                let n = (num(2, 16) as usize).min(0x2000 - start);
                let which = num(3, 0) as usize & 1;
                for i in start..start + n {
                    let c = f.dcb.cmap[which].pal[i];
                    e(writeln!(w, "  cmap{which}[{i:#06x}] r={:3} g={:3} b={:3}", (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF))?;
                }
                Ok(())
            }
            "dac" => {
                let d = &f.dcb.dac;
                let ident = d.gamma.iter().enumerate().all(|(i, g)| g.iter().all(|&v| v as usize == i << 2));
                e(writeln!(w, "DAC: pixmask {:#04x}  gamma {}", d.pixmask(), if ident { "identity" } else { "custom" }))?;
                let regs: Vec<String> = sorted(d.regs()).iter().map(|(k, v)| format!("[{k:#x}]={v:#x}")).collect();
                e(writeln!(w, "  registers: {}", regs.join(" ")))?;
                e(writeln!(w, "  gamma[0..8]: {:x?}", &d.gamma[..8]))
            }
            "pix" => {
                // mgras pix x y   (display coordinates: y = 0 is the top row)
                let (x, y) = (num(1, 0) as usize, num(2, 0) as usize);
                drop(f);
                let fr = self.snapshot_frame();
                if x >= fr.width || y >= fr.height {
                    return e(writeln!(w, "out of range ({}x{} displayed)", fr.width, fr.height));
                }
                let i = y * rss::WIDTH + x;
                let (did, odid) = (fr.did_main[i] as usize & 31, fr.did_overlay[i] as usize & 31);
                let px = fr.pixel(x, y);
                e(writeln!(w, "({x}, {y}) [fb row {}]: main={:#010x} overlay={:#x}  DID main {did} {:?} overlay {odid} {:?} -> screen r={} g={} b={}",
                    fr.height - 1 - y, fr.main[i], fr.overlay[i], fr.main_mode[did], fr.overlay_mode[odid],
                    px & 0xFF, (px >> 8) & 0xFF, (px >> 16) & 0xFF))
            }
            _ => e(help(w)),
        }
    }

    pub(super) fn cmd_rss(&self, args: &[&str], w: &mut dyn Write) -> Result<(), String> {
        let e = |r: std::io::Result<()>| r.map_err(|e| e.to_string());
        // SAFETY: read-only peek at the RSS thread's state (may tear).
        let rss = unsafe { &*self.rss.get() };
        let num = |i: usize, d: u32| args.get(i).and_then(|s| s.parse::<u32>().ok()).unwrap_or(d);
        match args.first().copied().unwrap_or("regs") {
            "regs" => {
                e(writeln!(w, "RSS: fifo {} pending, busy={}", self.rss_fifo.len(), self.rss_busy.load(Ordering::Relaxed)))?;
                for r in 0..0x400u32 {
                    let v = rss.reg(r);
                    let name = REG_NAMES.get(r as usize).copied().unwrap_or("");
                    if v != 0 {
                        e(writeln!(w, "  {r:#05x} {name:<16} {v:#010x}"))?;
                    }
                }
                e(writeln!(w, "  IR decodes as: {}", rss.describe_ir()))
            }
            "pix" => {
                // rss pix x y [w] [h]   framebuffer coordinates: y = 0 is the bottom row
                let (x0, y0) = (num(1, 0) as usize, num(2, 0) as usize);
                let (wd, ht) = (num(3, 8) as usize, num(4, 1) as usize);
                let b = rss.target();
                e(writeln!(w, "  drawing buffer: page {} {:?}", b.ptr, b.kind))?;
                for y in (y0..(y0 + ht).min(rss::HEIGHT)).rev() {
                    let row: Vec<String> = (x0..(x0 + wd).min(rss::WIDTH))
                        .map(|x| format!("{:08x}", rss.mem.get(&b, x as u32, y as u32))).collect();
                    e(writeln!(w, "  y={y:4} x={x0:4}: {}", row.join(" ")))?;
                }
                Ok(())
            }
            _ => e(writeln!(w, "rss [regs] | rss pix <x> <y> [w] [h]   (the buffer being drawn; y = 0 is the bottom row)")),
        }
    }

    fn mgras_status(&self, w: &mut dyn Write) -> std::io::Result<()> {
        // SAFETY: read-only peek at the RSS (may tear while drawing).
        let rss = unsafe { &*self.rss.get() };
        let f = self.front();
        let flags = self.all_flags(&f);
        writeln!(w, "IMPACT {:?} at {:#010x}", self.kind, MGRAS_SLOT_GFX_BASE)?;
        writeln!(w, "  flags {flags:#010x} [{}]  interrupt enable {:#010x}", flag_names(flags), f.hq.interrupt_enable)?;
        writeln!(w, "  fifo: hq {} rss {}  idle={}", self.hq_fifo.len(), self.rss_fifo.len(), self.idle())?;
        writeln!(w, "  DAC pixmask {:#04x}  XMAP DID0 mode {:#x}", f.dcb.dac.pixmask(), f.dcb.xmap.main_mode(0))?;
        match f.dcb.vc3.cursor() {
            Some((x, y, size, _)) => writeln!(w, "  cursor at ({x}, {y}) size {size}")?,
            None => writeln!(w, "  cursor hidden")?,
        }
        writeln!(w, "  fill modes seen: {:x?}", rss.fillmodes_seen())?;
        let mut hist: HashMap<u32, usize> = HashMap::new();
        let (main, _) = super::frame::scanout_buffers(rss, &f.dcb);
        for y in 0..1024 {
            for x in 0..1280 {
                *hist.entry(rss.mem.get(&main, x, y) as u32).or_default() += 1;
            }
        }
        let mut top: Vec<_> = hist.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1));
        writeln!(w, "  main buffer (page {}) values over 1280x1024 (value, pixels): {:x?}", main.ptr, &top[..top.len().min(8)])?;
        let pal = &f.dcb.cmap[0].pal;
        let blocks: Vec<usize> = (0..pal.len() / 256).filter(|k| pal[k * 256..(k + 1) * 256].iter().any(|c| *c != 0)).collect();
        writeln!(w, "  colormap blocks in use: {blocks:?}")?;
        drop(f);
        if let Some(r) = self.submit.lock().as_ref() {
            let r = r.lock();
            writeln!(w, "  recording: {} ({} records)", r.path, r.records)?;
        }
        let t = self.trace.lock();
        let mask = self.trace_mask.load(Ordering::Relaxed);
        if mask != 0 {
            writeln!(w, "  trace: {} (mask {mask:#x}) hq={} rss={} cpu={} entries", t.path, t.hq_seq, t.rss_seq, t.cpu_seq)?;
        }
        drop(t);
        for u in self.unhandled.lock().iter() {
            writeln!(w, "  not modelled: {u}")?;
        }
        Ok(())
    }

    fn stats(&self, w: &mut dyn Write) -> std::io::Result<()> {
        // SAFETY: read-only peeks at the engines' counters (may tear).
        let h = unsafe { &*self.eng.get() }.stats;
        let r = unsafe { &*self.rss.get() }.stats;
        writeln!(w, "HQ3: {} words; raster writes {} (exec {}), done flags {}, swaps {}, context switches {}",
            h.words, h.raster_writes, h.raster_execs, h.set_done, h.swaps, h.context_switches)?;
        writeln!(w, "  DMA: register writes {}, writes {}, reads {}, {} bytes; formatter {}, raster interface {}, pixel commands {}, other {}, display lists {} ({} segments)",
            h.dma_reg_writes, h.dma_writes, h.dma_reads, h.dma_bytes, h.formatter_writes, h.raster_if_writes, h.pixel_cmds, h.other_cmds, h.dl_calls, h.dl_segments)?;
        let cp: Vec<String> = h.cp_tokens.iter().enumerate().filter(|(_, n)| **n != 0).map(|(t, n)| format!("{t:#x}:{n}")).collect();
        writeln!(w, "  command-processor tokens (token:count): {}", if cp.is_empty() { "none".into() } else { cp.join(" ") })?;
        let prims: Vec<String> = r.prims.iter().enumerate().filter(|(_, n)| **n != 0).map(|(o, n)| format!("{o:#x}:{n}")).collect();
        writeln!(w, "RSS: primitives by IR opcode {}", prims.join(" "))?;
        let blocks: Vec<String> = r.blocks.iter().enumerate().filter(|(_, n)| **n != 0).map(|(k, n)| format!("{k}:{n}")).collect();
        writeln!(w, "  blocks by type {}  fast fills {}", blocks.join(" "), r.fast_fills)?;
        writeln!(w, "  stipple chunks {}  PIO dw written {} read {}  DMA lines in {}",
            r.stipple_chunks, r.pio_write_dw, r.pio_read_dw, r.dma_lines_in)?;
        #[cfg(feature = "gr4-jit")]
        {
            // SAFETY: as above.
            let j = unsafe { &*self.rss.get() }.jit;
            let (compiled, queued, failed, bytes) = super::rss_jit::store().summary();
            writeln!(w, "JIT: {} shader runs, {} while compiling, {} not covered; {compiled} shaders ({bytes} bytes), {queued} queued, {failed} failed",
                j.hits, j.misses, j.declined)?;
        }
        Ok(())
    }

    /// `mgras jit [on|off|sync|list]`: the raster JIT's mode and shaders.
    #[cfg(feature = "gr4-jit")]
    fn cmd_jit(&self, args: &[&str], w: &mut dyn Write) -> Result<(), String> {
        use super::rss_jit::{self, PipeKey, MODE_ASYNC, MODE_OFF, MODE_SYNC};
        let e = |r: std::io::Result<()>| r.map_err(|e| e.to_string());
        let mode = match args {
            ["on"] | ["async"] => Some(MODE_ASYNC),
            ["off"] => Some(MODE_OFF),
            ["sync"] => Some(MODE_SYNC),
            ["list"] => {
                for (k, bytes) in rss_jit::store().compiled() {
                    e(writeln!(w, "{k:#018x} {bytes:6}B  {}", PipeKey::unpack(k)))?;
                }
                return Ok(());
            }
            [] => None,
            _ => return Err("usage: mgras jit [on|off|sync|list]".into()),
        };
        if let Some(m) = mode {
            // The RSS thread reads the mode: change it with the board idle.
            let _sub = self.submit.lock();
            self.wait_idle();
            // SAFETY: the board is idle and `submit` keeps it so.
            unsafe { &mut *self.rss.get() }.jit.mode = m;
        }
        // SAFETY: read-only peek (may tear).
        let j = unsafe { &*self.rss.get() }.jit;
        let name = match j.mode {
            MODE_OFF => "off",
            MODE_SYNC => "sync",
            _ => "on",
        };
        let (compiled, queued, failed, bytes) = rss_jit::store().summary();
        e(writeln!(w, "raster JIT {name}: {} shader runs, {} while compiling, {} not covered; {compiled} shaders ({bytes} bytes), {queued} queued, {failed} failed",
            j.hits, j.misses, j.declined))
    }

    /// `mgras fbdump`: the raw page memory (pixmem.bin, big-endian u64s),
    /// the displayed main and overlay buffers as big-endian u32s (fb.bin,
    /// overlay.bin: stride 2048, display-height rows, row 0 at the bottom),
    /// and PNG views of them at the displayed size, top-down.
    fn dump_framebuffer(&self, dir: &std::path::Path) -> std::io::Result<()> {
        use rss::WIDTH;
        std::fs::create_dir_all(dir)?;
        // SAFETY: read-only views; tearing tolerated.
        let rss = unsafe { &*self.rss.get() };
        let f = self.front();
        let (dw, dh) = super::frame::display_size(&f.dcb);
        let (main, overlay) = super::frame::scanout_buffers(rss, &f.dcb);
        drop(f);
        std::fs::write(dir.join("pixmem.bin"), rss.mem.words.iter().flat_map(|w| w.to_be_bytes()).collect::<Vec<u8>>())?;
        let view = |b: Option<super::pixmem::Buffer>| {
            let mut out = vec![0u32; WIDTH * dh];
            if let Some(b) = b {
                for y in 0..dh {
                    rss.mem.read_row(&b, y as u32, &mut out[y * WIDTH..(y + 1) * WIDTH]);
                }
            }
            out
        };
        let (fb, ov_plane) = (view(Some(main)), view(overlay));
        let be = |words: &[u32]| words.iter().flat_map(|w| w.to_be_bytes()).collect::<Vec<u8>>();
        std::fs::write(dir.join("fb.bin"), be(&fb))?;
        std::fs::write(dir.join("overlay.bin"), be(&ov_plane))?;
        let (mut rgb, mut ci, mut ov) = (Vec::new(), Vec::new(), Vec::new());
        for y in (0..dh).rev() {
            for &p in &fb[y * WIDTH..y * WIDTH + dw] {
                rgb.extend_from_slice(&[p as u8, (p >> 8) as u8, (p >> 16) as u8]);
                ci.push(p as u8);
            }
            ov.extend(ov_plane[y * WIDTH..y * WIDTH + dw].iter().map(|&p| p as u8));
        }
        write_png(&dir.join("rgb.png"), dw, dh, &rgb, png::ColorType::Rgb)?;
        write_png(&dir.join("ci.png"), dw, dh, &ci, png::ColorType::Grayscale)?;
        write_png(&dir.join("overlay.png"), dw, dh, &ov, png::ColorType::Grayscale)?;
        self.save_shot(dir.join("screen.png").to_str().unwrap_or("screen.png")).map_err(std::io::Error::other)
    }
}

fn trace_mask(spec: &str) -> u32 {
    spec.split(',').fold(0, |m, c| m | match c {
        "hq" => TRACE_HQ,
        "rss" => TRACE_RSS,
        "cpu" => TRACE_CPU,
        "tex" => TRACE_TEX,
        _ => TRACE_ALL,
    })
}

fn help(w: &mut dyn Write) -> std::io::Result<()> {
    for l in [
        "mgras [status]                    board summary",
        "mgras hq                          HQ3 flags, enables, engine, DMA/RE-interface registers, FIFOs",
        "mgras ge [n] [start] [count]      GE11 microcode lines",
        "mgras ucode [start] [count]       HQ3 microcode RAM",
        "mgras vc3 | vc3 sram <a> [n]      VC3 registers, cursor, DID runs / SRAM words",
        "mgras xmap                        XMAP display modes per DID and register files",
        "mgras cmap [start] [n] [0|1]      colormap entries",
        "mgras dac                         DAC registers and gamma",
        "mgras pix <x> <y>                 one pixel through the compositor (y = 0 is the top)",
        "mgras stats                       command, primitive and DMA counters",
        "mgras fbdump [dir]                planes and the screen as PNG + raw files",
        "mgras shot <file.png> | dump <file>",
        "mgras trace <file> [hq,rss,cpu,tex|all] | trace flush | trace off",
        "mgras rec <file> | rec mark | rec off   replayable recording (see record.rs)",
        "rss [regs] | rss pix <x> <y> [w] [h]",
    ] {
        writeln!(w, "{l}")?;
    }
    Ok(())
}

fn sorted(it: impl Iterator<Item = (u32, u32)>) -> Vec<(u32, u32)> {
    let mut v: Vec<_> = it.collect();
    v.sort();
    v
}

/// Names of the flag bits this model knows.
fn flag_names(flags: u32) -> String {
    const NAMES: [(u32, &str); 6] = [
        (host::FLAG_CONTEXT_LOADED, "ctx_loaded"), (host::FLAG_CP0, "cp0_swap"), (host::FLAG_DONE, "done"),
        (host::FLAG_GE_DATA, "ge_data"), (host::FLAG_GE_DIAG, "ge_diag"), (host::FLAG_CONTEXT_SAVED, "ctx_saved"),
    ];
    let mut out: Vec<String> = NAMES.iter().filter(|(b, _)| flags & b != 0).map(|(_, n)| n.to_string()).collect();
    let rest = flags & !NAMES.iter().fold(0, |m, (b, _)| m | b);
    if rest != 0 {
        out.push(format!("{rest:#x}"));
    }
    out.join(" ")
}

/// An XMAP display mode as this model reads it (pixel format in bits 4:0,
/// colormap block in 9:5).
fn decode_mode(m: u32) -> String {
    let fmt = m & 0x1F;
    if fmt >= 4 { format!("RGB fmt {fmt}") } else { format!("CI fmt {fmt} cmap block {}", (m >> 5) & 0x1F) }
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
        writeln!(w, "  {:#06x}: {}", start + i * 8, line.join(" "))?;
    }
    Ok(())
}

fn write_png(path: &std::path::Path, width: usize, height: usize, data: &[u8], color: png::ColorType) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut enc = png::Encoder::new(BufWriter::new(file), width as u32, height as u32);
    enc.set_color(color);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(std::io::Error::other)?;
    w.write_image_data(data).map_err(std::io::Error::other)
}

/// Display control bus device names, and each device's register selects.
fn dcb_names(dev: u32) -> (&'static str, [&'static str; 8]) {
    match dev {
        dcb::DEV_CMAP_ALL | dcb::DEV_CMAP0 | dcb::DEV_CMAP1 => (
            ["cmap_all", "cmap0", "cmap1"][(dev - dcb::DEV_CMAP_ALL) as usize],
            ["addr", "addrhi", "pal", "cmd", "status", "sel5", "rev", "reserved"],
        ),
        dcb::DEV_DAC => ("dac", ["addr", "pal", "reg", "mode", "sel4", "sel5", "sel6", "sel7"]),
        dcb::DEV_XMAP => ("xmap", ["pp1select", "index", "config", "buf_select", "main_mode", "overlay_mode", "dib", "re_rac"]),
        dcb::DEV_VC3 => ("vc3", ["index", "data", "sel2", "ram", "sel4", "sel5", "sel6", "sel7"]),
        dcb::DEV_BDVERS => ("bdvers", ["bdvers0", "bdvers1", "sel2", "sel3", "sel4", "sel5", "sel6", "sel7"]),
        dcb::DEV_I2C => ("i2c", ["sel0", "sel1", "sel2", "sel3", "sel4", "sel5", "sel6", "sel7"]),
        _ => ("dcb?", ["sel0", "sel1", "sel2", "sel3", "sel4", "sel5", "sel6", "sel7"]),
    }
}

/// Region of a board offset for CPU tracing: (kind, bulk). Bulk regions are
/// memories (colormap and gamma tables, VC3 SRAM, microcode) whose runs of
/// accesses collapse into one line; any other repeat collapses only while
/// it hits the same register with the same value.
fn region_kind(off: u32) -> (&'static str, bool) {
    match off {
        host::UCODE..host::UCODE_END => ("hq.ucode", true),
        0x60000..0x68000 => {
            let t = dcb::Txn::decode(off);
            match (t.dev, t.crs) {
                (dcb::DEV_CMAP_ALL..=dcb::DEV_CMAP1, 2) => ("cmap.pal", true),
                (dcb::DEV_DAC, 1) => ("dac.pal", true),
                (dcb::DEV_VC3, 3) => ("vc3.ram", true),
                (dcb::DEV_XMAP, 6) => ("xmap.dib", true),
                _ => ("dcb", false),
            }
        }
        0x50200..0x50500 => ("hq.context", true),
        _ => ("reg", false),
    }
}

/// Human-readable name for a board offset (register names where known).
fn region_detail(off: u32) -> String {
    const HOST: [(u32, &str); 40] = [
        (0x00000, "gio_id"), (0x46000, "hqpc"), (0x47000, "cp_data"),
        (0x50000, "hq_config"), (0x50004, "flag_ctx"), (0x50008, "flag_set_priv"), (0x5000C, "flag_clr_priv"),
        (0x50010, "flag_enab_set"), (0x50014, "flag_enab_clr"), (0x50018, "intr_enab_set"), (0x5001C, "intr_enab_clr"),
        (0x50020, "cfifo_hw"), (0x50024, "cfifo_lw"), (0x50028, "cfifo_delay"), (0x5002C, "dfifo_hw"),
        (0x50030, "dfifo_lw"), (0x50034, "dfifo_delay"), (0x50040, "ge0_diag_d"), (0x50044, "ge0_diag_a"),
        (0x50048, "ge1_diag_d"), (0x5004C, "ge1_diag_a"), (0x50050, "context_switch"),
        (0x50108, "bfifo_hw"), (0x5010C, "bfifo_lw"), (0x50110, "bfifo_delay"), (0x50114, "gio_config"),
        (0x5022C, "ge_diag_read"), (0x50230, "ge_diag_read_pad"), (0x50880, "tlb_curr_addr"), (0x50900, "tlb_valids"),
        (0x70000, "status"), (0x70004, "fifo_status"), (0x70008, "flag_set"), (0x7000C, "flag_clr"),
        (0x70010, "ge_readback_hi"), (0x70014, "ge_readback_lo"), (0x70100, "gio_status"), (0x70104, "dma_busy"),
        (host::PIO_READ_HI, "pio_read_hi"), (host::PIO_READ_LO, "pio_read_lo"),
    ];
    if let Some((_, n)) = HOST.iter().find(|(a, _)| *a == off) {
        return n.to_string();
    }
    match off {
        host::UCODE..host::UCODE_END => format!("hq.ucode[{:#06x}]", (off - host::UCODE) >> 2),
        0x4C000..0x50000 => format!("rss_diag[{:#x}]", (off - 0x4C000) >> 2),
        0x50200..0x50300 => format!("reif_ctx[{:#x}]", (off - 0x50200) >> 2),
        0x50300..0x50500 => format!("hag_ctx[{:#x}]", (off - 0x50300) >> 2),
        0x50800..0x50880 => format!("tlb[{}]", (off - 0x50800) >> 2),
        0x60000..0x68000 => {
            let t = dcb::Txn::decode(off);
            let (dev, sel) = dcb_names(t.dev);
            format!("{dev}.{} (w{})", sel[t.crs as usize], t.width)
        }
        0x68000..0x68040 => {
            let dev = (off - 0x68000) >> 2;
            format!("dcbctrl[{dev}] ({})", dcb_names(dev).0)
        }
        host::RASTER..host::RASTER_END => {
            let r = (off & 0xFFC) >> 2;
            match REG_NAMES.get(r as usize).copied().unwrap_or("") {
                "" => format!("rss[{r:#x}]"),
                n => format!("rss.{n}"),
            }
        }
        _ => format!("[{off:#07x}]"),
    }
}
