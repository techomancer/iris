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
//! `--startup "rex trace <file>"` starts a full trace before the CPU runs.

use std::sync::atomic::Ordering;

use crate::trace_sink::{trace_command, Classes, TraceFile, TraceRun};

use super::*;

/// GFIFO entries as the processor applies them.
pub const TRACE_GFIFO: u32 = 1;
/// CPU and DMA accesses outside the GFIFO.
pub const TRACE_CPU: u32 = 2;
pub const TRACE_ALL: u32 = TRACE_GFIFO | TRACE_CPU;

/// Pending run of identical CPU accesses, collapsed into one line.
pub struct CpuRun {
    write: bool,
    bits: u32,
    off: u32,
    val: u64,
    /// DCB target, for DCBDATA accesses.
    dcb: Option<u32>,
    count: u64,
}

impl TraceRun for CpuRun {
    fn merge(&mut self, n: &Self) -> bool {
        let same = (self.write, self.bits, self.off, self.val, self.dcb) == (n.write, n.bits, n.off, n.val, n.dcb);
        self.count += same as u64;
        same
    }

    fn emit(&self, seq: u64, line: impl FnOnce(std::fmt::Arguments)) {
        let dir = if self.write { "wr" } else { "rd" };
        let name = reg_label(self.off & !0x0800, self.dcb);
        let go = if self.off & 0x0800 != 0 { " GO" } else { "" };
        let val = if self.bits > 32 { format!("{:#018x}", self.val) } else { format!("{:#010x}", self.val) };
        if self.count == 1 {
            line(format_args!("CPU #{seq:06} {dir} {name}{go} = {val}"));
        } else {
            line(format_args!("CPU #{seq:06} {dir} {name}{go} = {val}  (x{})", self.count));
        }
    }
}

/// Trace state; lives in `Rex3::trace`. Counters: GFIFO entries, then CPU
/// and DMA lines.
pub type Rex3Trace = TraceFile<CpuRun, 2>;

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

/// `gfifo,cpu` or `all`.
const CLASSES: Classes = Classes {
    names: &[("gfifo", TRACE_GFIFO), ("gf", TRACE_GFIFO), ("cpu", TRACE_CPU), ("dma", TRACE_CPU)],
    all: TRACE_ALL,
};

impl Rex3 {
    #[inline]
    pub(crate) fn tracing(&self, what: u32) -> bool {
        self.tap.load(Ordering::Relaxed) & what != 0
    }

    /// One CPU access outside the GFIFO (board offset with the GO bit,
    /// value read or written; `dcbmode` as it was before a DCBDATA access).
    pub(crate) fn trace_cpu(&self, write: bool, bits: u32, off: u32, val: u64, dcbmode: Option<u32>) {
        let dcb = dcbmode.map(|m| m & (DCBMODE_DCBADDR_MASK | DCBMODE_DCBCRS_MASK | DCBMODE_DATAWIDTH_MASK));
        self.trace.lock().access(CpuRun { write, bits, off, val, dcb, count: 1 });
    }

    /// A DMA engine transfer.
    pub(crate) fn trace_dma(&self, text: std::fmt::Arguments) {
        self.trace.lock().event(1, "DMA", text);
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
        self.trace.lock().event(0, "GF ", text);
    }

    /// Hand the trace to its writer thread (called every frame from the
    /// refresh thread; it ships at most twice a second unless forced).
    pub(crate) fn trace_flush(&self, force: bool) {
        if self.tap.load(Ordering::Relaxed) & TRACE_ALL == 0 && !force {
            return;
        }
        self.trace.lock().flush(force);
    }

    /// Start the annotated trace into `path` for the categories in `mask`.
    pub fn start_trace(&self, path: &str, mask: u32) -> std::io::Result<()> {
        #[cfg(feature = "wgpu")]
        self.reject_resident_operation("tracing").map_err(std::io::Error::other)?;
        let mut t = self.trace.lock();
        t.open(path)?;
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
        self.trace.lock().stop()
    }

    /// `rex trace ...`.
    pub(super) fn cmd_trace(&self, args: &[&str], w: &mut dyn std::io::Write) -> Result<(), String> {
        trace_command(args, "rex trace", &CLASSES, w, || self.trace_flush(true), || self.stop_trace(), |p, m| self.start_trace(p, m))
    }
}
