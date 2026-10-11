//! REX3 annotated trace (`rex trace <file> [gfifo,cpu|all]`), in the GR2 and
//! IMPACT trace format.
//!
//! Trace lines look like
//!   `    1.234567 GF  #000123 DRAWMODE0 = 0x00000006  ; NOOP BLOCK DOSETUP`
//!   `    1.234570 GF  #000124 XYENDI = 0x13ff13ff GO -> DRAW BLOCK DOSETUP (0, 0)-(1023, 1023)`
//!   `    1.234575 GF  #000125 DMA_BATCH_W GO  ; batch write 600 qwords, first=... -> ...`
//!   `    1.234580 CPU #000071 rd STATUS = 0x00000000  (x5123)`
//!   `    1.234590 CPU #000072 wr DCBDATA0 [VC2 crs 3 dw2] = 0x00001000`
//!   `    1.234600 DMA #000073 BATCH_R done: 600 qwords, first=... last=...`
//! with the time in seconds since the trace started. GF lines are GFIFO
//! entries as the processor thread applies them, a GO decoded with the
//! registers it runs with; CPU lines are every other bus access, repeats of
//! one register collapsed; DMA lines are the MC DMA engine's transfers.
//!
//! Lines are formatted into memory and written by a `TraceWriter` thread, so
//! a trace costs the processor thread a format, not a syscall.
//! `IRIS_REX3_TRACE=<file>` starts a full trace when the board starts.

use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::trace_writer::TraceWriter;

use super::*;

/// GFIFO entries as the processor applies them.
pub const TRACE_GFIFO: u32 = 1;
/// CPU and DMA accesses outside the GFIFO.
pub const TRACE_CPU: u32 = 2;
pub const TRACE_ALL: u32 = TRACE_GFIFO | TRACE_CPU;

/// Pending run of identical CPU accesses, collapsed into one line.
struct CpuRun {
    write: bool,
    bits: u32,
    off: u32,
    val: u64,
    /// DCB target, for DCBDATA accesses.
    dcb: Option<u32>,
    count: u64,
}

/// Trace sink state; lives in `Rex3::trace`.
pub struct Rex3Trace {
    out: Option<TraceWriter>,
    start: Option<Instant>,
    gf_seq: u64,
    cpu_seq: u64,
    run: Option<CpuRun>,
    last_flush: Option<Instant>,
}

impl Rex3Trace {
    pub const fn new() -> Self {
        Self { out: None, start: None, gf_seq: 0, cpu_seq: 0, run: None, last_flush: None }
    }

    /// Emit the pending collapsed CPU run, if any.
    fn close_run(&mut self) {
        let Some(r) = self.run.take() else { return };
        let dir = if r.write { "wr" } else { "rd" };
        let seq = self.cpu_seq;
        let name = reg_label(r.off & !0x0800, r.dcb);
        let go = if r.off & 0x0800 != 0 { " GO" } else { "" };
        let val = if r.bits > 32 { format!("{:#018x}", r.val) } else { format!("{:#010x}", r.val) };
        if r.count == 1 {
            self.line(format_args!("CPU #{seq:06} {dir} {name}{go} = {val}"));
        } else {
            self.line(format_args!("CPU #{seq:06} {dir} {name}{go} = {val}  (x{})", r.count));
        }
    }

    fn line(&mut self, text: std::fmt::Arguments) {
        let t = self.start.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
        if let Some(f) = self.out.as_mut() {
            f.line(format_args!("{t:12.6} {text}"));
        }
    }
}

/// Register name, with the DCB target for DCBDATA accesses.
fn reg_label(reg: u32, dcbmode: Option<u32>) -> String {
    let name = match rex3_reg_name(reg) {
        "UNKNOWN" => format!("[{reg:#06x}]"),
        n => n.to_string(),
    };
    match dcbmode {
        Some(m) => format!("{name} [{}]", dcb_target(m)),
        None => name,
    }
}

/// The DCB device, register select and width a DCBMODE value addresses.
pub(crate) fn dcb_target(mode: u32) -> String {
    let addr = (mode & DCBMODE_DCBADDR_MASK) >> DCBMODE_DCBADDR_SHIFT;
    let crs = (mode & DCBMODE_DCBCRS_MASK) >> DCBMODE_DCBCRS_SHIFT;
    let width = match mode & DCBMODE_DATAWIDTH_MASK {
        DCBMODE_DATAWIDTH_1 => 1,
        DCBMODE_DATAWIDTH_2 => 2,
        DCBMODE_DATAWIDTH_3 => 3,
        _ => 4,
    };
    let dev = match addr {
        0 => "VC2".to_string(),
        1 => "CMAP01".to_string(),
        2 => "CMAP0".to_string(),
        3 => "CMAP1".to_string(),
        4 => "XMAP01".to_string(),
        5 => "XMAP0".to_string(),
        6 => "XMAP1".to_string(),
        7 => "RAMDAC".to_string(),
        12 => "BACKBUSY".to_string(),
        n => format!("dev{n}"),
    };
    format!("{dev} crs {crs} dw{width}")
}

/// Parse a trace category list: `gfifo,cpu` or `all`.
pub fn trace_mask(s: &str) -> u32 {
    s.split(',').fold(0, |m, w| m | match w {
        "gfifo" | "gf" => TRACE_GFIFO,
        "cpu" | "dma" => TRACE_CPU,
        _ => TRACE_ALL,
    })
}

impl Rex3 {
    #[inline]
    pub(crate) fn tracing(&self, what: u32) -> bool {
        self.tap.load(Ordering::Relaxed) & what != 0
    }

    /// One CPU access outside the GFIFO (board offset with the GO bit,
    /// value read or written; `dcbmode` as it was before a DCBDATA access).
    pub(crate) fn trace_cpu(&self, write: bool, bits: u32, off: u32, val: u64, dcbmode: Option<u32>) {
        let dcb = dcbmode.map(|m| m & (DCBMODE_DCBADDR_MASK | DCBMODE_DCBCRS_MASK | DCBMODE_DATAWIDTH_MASK));
        let mut t = self.trace.lock();
        if let Some(r) = t.run.as_mut() {
            if (r.write, r.bits, r.off, r.val, r.dcb) == (write, bits, off, val, dcb) {
                r.count += 1;
                return;
            }
        }
        t.close_run();
        t.cpu_seq += 1;
        t.run = Some(CpuRun { write, bits, off, val, dcb, count: 1 });
    }

    /// A DMA engine transfer.
    pub(crate) fn trace_dma(&self, text: std::fmt::Arguments) {
        let mut t = self.trace.lock();
        t.close_run();
        t.cpu_seq += 1;
        let seq = t.cpu_seq;
        t.line(format_args!("DMA #{seq:06} {text}"));
    }

    /// One GFIFO entry, after `process_register` applied it and before its
    /// GO (if any) runs, so the GO decodes with the registers it draws with.
    pub(crate) fn trace_gfifo(&self, addr: u32, val: u64) {
        // SAFETY: called on the processor thread, which owns the context.
        let ctx = unsafe { &*self.context.get() };
        let reg = addr & !0x0800;
        let is_go = addr & 0x0800 != 0;
        let go = || {
            let dm0 = ctx.drawmode0.0;
            format!(" -> {} ({}, {})-({}, {})", decode_dm0(dm0), ctx.xstart >> 11, ctx.ystart >> 11, ctx.xend >> 11, ctx.yend >> 11)
        };
        let text = match reg {
            GFIFO_EXIT => "op  EXIT".to_string(),
            GFIFO_DISP_SYNC => format!("op  DISP_SYNC fence={val}"),
            GFIFO_PURE_GO_REG => format!("PURE_GO{}", go()),
            REX3_DMA_PURE_GO_REG => format!("DMA_PURE_GO{}", go()),
            REX3_DMA_BATCH_W_REG => format!(
                "DMA_BATCH_W GO  ; batch write {} qwords, first={:016x}{}",
                ctx.host_len, ctx.hostrw[0], go()
            ),
            REX3_DMA_BATCH_R_REG => format!("DMA_BATCH_R GO  ; batch read {val} qwords (filled after GO){}", go()),
            _ => {
                let name = reg_label(reg, None);
                let gs = if is_go { " GO" } else { "" };
                let v = val as u32;
                let extra = match reg {
                    REX3_DRAWMODE0 => format!("  ; {}", decode_dm0(v)),
                    REX3_DRAWMODE1 => format!("  ; {}", decode_dm1(v)),
                    REX3_XYSTARTI | REX3_XYENDI => format!("  ({}, {})", (v >> 16) as u16 as i16, v as u16 as i16),
                    _ => String::new(),
                };
                let run = if is_go { go() } else { String::new() };
                if reg == REX3_HOSTRW64 {
                    format!("{name}{gs} = {val:#018x}{run}")
                } else {
                    format!("{name}{gs} = {v:#010x}{extra}{run}")
                }
            }
        };
        let mut t = self.trace.lock();
        t.close_run();
        t.gf_seq += 1;
        let seq = t.gf_seq;
        t.line(format_args!("GF  #{seq:06} {text}"));
    }

    /// Hand the trace to its writer thread (called every frame from the
    /// refresh thread; it ships at most twice a second unless forced).
    pub(crate) fn trace_flush(&self, force: bool) {
        if self.tap.load(Ordering::Relaxed) & TRACE_ALL == 0 && !force {
            return;
        }
        let mut t = self.trace.lock();
        let due = t.last_flush.map_or(true, |l| l.elapsed().as_millis() >= 500);
        if force || due {
            t.close_run();
            if let Some(f) = t.out.as_mut() {
                f.flush();
            }
            t.last_flush = Some(Instant::now());
        }
    }

    /// Start the annotated trace into `path` for the categories in `mask`.
    pub fn start_trace(&self, path: &str, mask: u32) -> std::io::Result<()> {
        let f = TraceWriter::create(path)?;
        let mut t = self.trace.lock();
        t.out = Some(f);
        t.start = Some(Instant::now());
        t.gf_seq = 0;
        t.cpu_seq = 0;
        t.run = None;
        t.line(format_args!("REX3 trace started (mask {mask:#x})"));
        drop(t);
        self.tap.fetch_and(!TRACE_ALL, Ordering::SeqCst);
        self.tap.fetch_or(mask & TRACE_ALL, Ordering::SeqCst);
        Ok(())
    }

    /// Stop the trace, writing out everything. Returns a summary, or None
    /// when no trace was running.
    pub fn stop_trace(&self) -> Option<String> {
        self.tap.fetch_and(!TRACE_ALL, Ordering::SeqCst);
        let mut t = self.trace.lock();
        t.close_run();
        let f = t.out.take()?;
        let path = f.path.clone();
        let res = f.finish();
        Some(match res {
            Ok(bytes) => format!("{path} (gfifo {} / cpu {} entries, {bytes} bytes)", t.gf_seq, t.cpu_seq),
            Err(e) => format!("{path}: write failed: {e}"),
        })
    }

    /// `rex trace ...`.
    pub(super) fn cmd_trace(&self, args: &[&str], w: &mut dyn std::io::Write) -> Result<(), String> {
        let e = |r: std::io::Result<()>| r.map_err(|e| e.to_string());
        match args {
            ["flush"] => {
                self.trace_flush(true);
                if let Some(f) = self.trace.lock().out.as_mut() {
                    f.sync();
                }
                e(writeln!(w, "trace flushed"))
            }
            [] | ["off"] => e(match self.stop_trace() {
                Some(s) => writeln!(w, "trace closed: {s}"),
                None => writeln!(w, "trace not active"),
            }),
            [path, rest @ ..] => {
                let mask = trace_mask(rest.first().copied().unwrap_or("all"));
                self.start_trace(path, mask).map_err(|e| format!("rex trace: cannot create {path}: {e}"))?;
                e(writeln!(w, "tracing to {path} (mask {mask:#x})"))
            }
        }
    }
}
