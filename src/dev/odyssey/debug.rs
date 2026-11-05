//! VPro introspection: the monitor's `vpro ...` commands.
//!
//! Two kinds of command, by what they read:
//!
//! - State commands (`status`, `regs`, `fifo`, `gl`, `geom`, `tex`, `win`,
//!   `mem`, `backend`, `unimp`, `hash`) read the board's state,
//!   which the board's thread owns while it runs stores (interface.rs). They
//!   take it through `BoardLocks::quiesce`: the processors held off and the
//!   FIFO and raster queue drained, the state a waiting read sees. A board
//!   that stays busy past the timeout answers "busy" rather than a torn
//!   read. The text is formatted under the guard and written after it.
//! - Pixel commands (`pix`, `raw`, `peek`, `shot`, `fbdump`, `dispmode`,
//!   `lut`, `cursor`) read board memory and the back end's registers the way
//!   the display thread does (frame.rs): lock-free, while the board goes on
//!   drawing. They never stall the guest; a pixel read under a draw in
//!   flight may be one frame stale.
//!
//! `refusals` reads standing counters (mmio.rs).

use std::fmt::Write as _;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

use super::board::{Board, ODSY_TEX_LEVELS};
use super::consts::{
    backend, cursor, dispmode, lut, overlay, ODSY_CMAP_UNVERIFIED_GROUP, ODSY_DID_MAIN_SHIFT,
    ODSY_DID_OVERLAY_MASK, ODSY_GAMMA_GROUP,
};
use super::frame::Frame;
use super::reader::DisplayRegs;
use super::vram::{VramPtr, UNIT};
use super::record::Recorder;
use super::trace::trace_mask;
use super::Odyssey;
use super::assembly::CLIP_PLANES;

/// How long a state command waits for the board to go idle before it
/// answers busy. A guest that keeps the FIFO fed still drains it at every
/// waiting read, so the wait is normally microseconds.
const QUIESCE: Duration = Duration::from_millis(500);

/// What a replay checkpoint hashes (`golden/replay.rs`) and `vpro hash`
/// prints: the whole of board memory.
pub(super) fn memory_hash(b: &Board) -> String {
    blake3::Hash::from(memory_hash_bytes(b)).to_hex().to_string()
}

/// `memory_hash` as bytes: what a recording's mark holds.
pub(super) fn memory_hash_bytes(b: &Board) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    let mem = b.vram.ptr_shared();
    let mut buf = Vec::with_capacity(1 << 16);
    for byte in mem.address_bytes() {
        buf.push(byte);
        if buf.len() == buf.capacity() {
            h.update(&buf);
            buf.clear();
        }
    }
    h.update(&buf);
    *h.finalize().as_bytes()
}

pub(super) const HELP: &[&str] = &[
    "vpro [status]                   board summary",
    "vpro regs                       host interface and widget registers",
    "vpro fifo                       command FIFO, raster queue, parser",
    "vpro state                      primitive assembly, transform and the stages' decoded words",
    "vpro geom                       geometry engine: transforms, lighting, program",
    "vpro tex                        texture state words, decoded",
    "vpro win                        window origin, clip rectangles, id compare",
    "vpro mem                        board memory: the pixel mode, the destination's fields, the scan-out",
    "vpro backend [start] [n]        back-end register file entries (0x2000-0x2fff)",
    "vpro lut [start] [n]            colour lookup RAM",
    "vpro dispmode                   display and overlay modes; drawable ids on screen",
    "vpro cursor                     the hardware cursor",
    "vpro pix <x> <y>                one pixel through the scanout (y = 0 is the top)",
    "vpro raw <x> <y> [w] [h]        frame-buffer pixels' bytes as the scan-out reads them",
    "vpro peek <addr> [bytes]        board memory at an address in 16-byte units",
    "vpro shot <file.png>            the screen composed from board memory now",
    "vpro fbdump [dir]               the screen as PNG, board memory as board.bin, layout.txt",
    "vpro hash                       the memory hash a replay checkpoint takes",
    "vpro unimp                      offsets and command words reported as not modelled",
    "vpro refusals [reset]           bus refusals by reason and register",
    "vpro trace <file> [fifo,draw,cpu,dma|all]   annotated trace (gputrace builds); alone: its state",
    "vpro trace flush|off            write the trace out / close it",
    "vpro rec [<file>|mark|off]      replayable recording: start, mark, stop; alone: its state",
];

/// A number as the monitor takes it: hex with 0x, else decimal.
fn num(s: &str) -> Option<u64> {
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => s.parse().ok(),
    }
}

fn arg(args: &[&str], i: usize, what: &str) -> Result<u64, String> {
    let s = args.get(i).ok_or_else(|| format!("vpro: missing {what}"))?;
    num(s).ok_or_else(|| format!("vpro: bad {what} '{s}'"))
}

fn arg_or(args: &[&str], i: usize, what: &str, default: u64) -> Result<u64, String> {
    match args.get(i) {
        Some(_) => arg(args, i, what),
        None => Ok(default),
    }
}

/// A primitive type's name.
pub(super) fn begin_name(mode: u32) -> &'static str {
    const NAMES: [&str; 10] = ["POINTS", "LINES", "LINE_LOOP", "LINE_STRIP", "TRIANGLES", "TRIANGLE_STRIP", "TRIANGLE_FAN", "QUADS", "QUAD_STRIP", "POLYGON"];
    NAMES.get(mode as usize).copied().unwrap_or("?")
}

fn f32s(bits: &[u32]) -> String {
    bits.iter().map(|&b| format!("{}", f32::from_bits(b))).collect::<Vec<_>>().join(" ")
}

fn fmt_mat4(out: &mut String, name: &str, m: &[f32; 16], column_major: bool) {
    let _ = writeln!(out, "  {name}:");
    for r in 0..4 {
        let row: Vec<String> = (0..4).map(|c| format!("{:12.6}", if column_major { m[c * 4 + r] } else { m[r * 4 + c] })).collect();
        let _ = writeln!(out, "    {}", row.join(" "));
    }
}

fn fmt_mat4_bits(out: &mut String, name: &str, m: &[u32; 16], column_major: bool) {
    fmt_mat4(out, name, &m.map(f32::from_bits), column_major);
}

fn on(b: bool) -> &'static str {
    if b { "on" } else { "off" }
}

/// A display mode decoded: its colour map group and how a plane word
/// resolves through it (reader.rs, `resolve_generic`).
fn dispmode_text(mode: u32) -> String {
    let group = (mode >> dispmode::GROUP_SHIFT) & dispmode::GROUP_MASK;
    if group >= ODSY_GAMMA_GROUP {
        let width = match (mode >> dispmode::WIDTH_SHIFT) & dispmode::WIDTH_MASK {
            dispmode::WIDTH_4 => "4-bit channels",
            dispmode::WIDTH_5 => "5-bit channels",
            _ => "8-bit channels",
        };
        format!("group {group} RGB through the gamma ramp, {width}")
    } else if group == 0 {
        "group 0: index, lut[word & 0xfff]".to_string()
    } else {
        format!("group {group}: index, lut[{:#x} + (word & 0xff)]{}", group * 256, if group == ODSY_CMAP_UNVERIFIED_GROUP { " (unverified)" } else { "" })
    }
}

fn write_png(path: &Path, width: usize, height: usize, data: &[u8], color: png::ColorType) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    enc.set_color(color);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(std::io::Error::other)?;
    w.write_image_data(data).map_err(std::io::Error::other)
}

impl Odyssey {
    /// The frame the scan-out registers describe now, and board memory.
    fn screen(&self) -> Result<(Frame, VramPtr, DisplayRegs), String> {
        let regs = self.state.shared.regs.read();
        let frame = Frame::snapshot(&regs, self.displayed()).ok_or_else(|| "vpro: the scan-out is not set up (PB&J 0x2800-0x2809)".to_string())?;
        let mem = self.state.shared.vram.get().ok_or_else(|| "vpro: board memory is not allocated".to_string())?;
        // Read only, racing the board's units as the display does.
        Ok((frame, *mem, regs))
    }

    fn displayed(&self) -> u32 {
        self.state.front.lock().displayed_buffer
    }

    /// A state command's view of the board, or the busy answer.
    fn quiesced(&self) -> Result<super::mmio::BoardGuard<'_>, String> {
        self.state.quiesce(QUIESCE).ok_or_else(|| {
            format!(
                "vpro: board busy for {} ms (FIFO {} words in {} stores, raster {} jobs); try again",
                QUIESCE.as_millis(),
                self.state.fifo.queued_words(),
                self.state.fifo.stores(),
                self.state.raster.pending()
            )
        })
    }

    pub(super) fn cmd_vpro(&self, args: &[&str], w: &mut dyn Write) -> Result<(), String> {
        let e = |r: std::io::Result<()>| r.map_err(|e| e.to_string());
        let text = match args {
            [] | ["status"] => self.status_text()?,
            ["help"] => HELP.join("\n") + "\n",
            ["trace"] => self.state.trace.status().unwrap_or_else(|| "trace not active".into()) + "\n",
            ["trace", "off"] => match self.state.trace.stop() {
                Some(s) => format!("trace closed: {s}\n"),
                None => "trace not active\n".to_string(),
            },
            ["trace", "flush"] => {
                self.state.trace.flush(true);
                "trace flushed\n".to_string()
            }
            ["trace", path, rest @ ..] => {
                let mask = trace_mask(rest.first().copied().unwrap_or("all"))?;
                self.state.trace.start(path, mask).map_err(|e| format!("vpro trace {path}: {e}"))?;
                format!("tracing to {path} (mask {mask:#x})\n")
            }
            ["rec", rest @ ..] => self.rec_text(rest)?,
            ["regs"] => self.regs_text()?,
            ["fifo"] => self.fifo_text()?,
            ["state"] => self.state_text()?,
            ["geom"] => self.geom_text()?,
            ["tex"] => self.tex_text()?,
            ["win"] => self.win_text()?,
            ["mem"] => self.mem_text()?,
            ["backend", rest @ ..] => {
                let start = arg_or(rest, 0, "start", backend::CTL_BASE as u64)? as u32;
                let n = arg_or(rest, 1, "count", backend::CTL_ENTRIES as u64)? as u32;
                self.backend_text(start, n)?
            }
            ["lut", rest @ ..] => {
                let start = arg_or(rest, 0, "start", 0)? as u32;
                let n = arg_or(rest, 1, "count", 16)? as u32;
                self.lut_text(start, n)
            }
            ["dispmode"] => self.dispmode_text()?,
            ["cursor"] => self.cursor_text(),
            ["pix", rest @ ..] => {
                let (x, y) = (arg(rest, 0, "x")? as u32, arg(rest, 1, "y")? as u32);
                self.pix_text(x, y)?
            }
            ["raw", rest @ ..] => {
                let (x, y) = (arg(rest, 0, "x")? as u32, arg(rest, 1, "y")? as u32);
                let wd = arg_or(rest, 2, "width", 4)? as u32;
                let ht = arg_or(rest, 3, "height", 1)? as u32;
                self.raw_text(x, y, wd, ht)?
            }
            ["peek", rest @ ..] => {
                let addr = arg(rest, 0, "address")?;
                let n = arg_or(rest, 1, "bytes", 64)?;
                self.peek_text(addr, n)?
            }
            ["shot", path] => self.shot(path)?,
            ["fbdump", rest @ ..] => self.fbdump(Path::new(rest.first().copied().unwrap_or("vprodump")))?,
            ["hash"] => {
                let g = self.quiesced()?;
                format!("{}\n", memory_hash(&g))
            }
            ["unimp"] => self.unimp_text()?,
            ["refusals"] => {
                let mut v = Vec::new();
                self.state.refusals.report(&mut v).map_err(|e| e.to_string())?;
                String::from_utf8_lossy(&v).into_owned()
            }
            ["refusals", "reset"] => {
                self.state.refusals.reset();
                String::new()
            }
            _ => return Err(format!("Usage:\n{}", HELP.join("\n"))),
        };
        e(w.write_all(text.as_bytes()))
    }

    // ── state commands ───────────────────────────────────────────────────────

    fn status_text(&self) -> Result<String, String> {
        let mut out = String::new();
        let fifo = &self.state.fifo;
        let raster = &self.state.raster;
        {
            let f = self.state.front.lock();
            let _ = writeln!(out, "VPro identity {:#010x}, widget control {:#010x}", f.identity, f.widget_control);
            let _ = writeln!(
                out,
                "  interrupts: flags {:#010x} enable {:#010x} asserted {} held {}  routed {} dest {:#x}:{:#x} vectors {:#010x}",
                f.intr_flags, f.intr_enable, f.interrupt_asserted, f.flag_held, f.routed(), f.route_dest_upper, f.route_dest_lower, f.route_vectors
            );
            let _ = writeln!(
                out,
                "  swap: arm {:#x} pending {:#x} set {:#x} clear {:#x}  displayed buffer bits {:#010x}",
                f.swap_arm, f.swap_pending, f.swap_set, f.swap_clear, f.displayed_buffer
            );
        }
        let _ = writeln!(
            out,
            "  command FIFO: level {} words ({} queued in {} stores), high water {}, thread {}",
            fifo.level(),
            fifo.queued_words(),
            fifo.stores(),
            fifo.high_water(),
            if fifo.running.load(std::sync::atomic::Ordering::SeqCst) { "running" } else { "stopped" }
        );
        let _ = writeln!(
            out,
            "  raster unit: {} jobs pending, {} pixels of work queued, thread {}",
            raster.pending(),
            raster.queued(),
            if raster.threaded() { "running" } else { "stopped" }
        );
        {
            let r = self.display.reader.lock();
            let _ = writeln!(out, "  display: frame {}x{}, {} retraces", r.frame_width, r.frame_height, r.clock.frames);
        }
        let regs = self.state.shared.regs.read();
        if regs.cursor_shown() {
            let _ = writeln!(out, "  cursor at ({}, {}) control {:#x}", regs.cursor_x(), regs.cursor_y(), regs.cursor[cursor::CONTROL as usize]);
        } else {
            let _ = writeln!(out, "  cursor off (control {:#x})", regs.cursor[cursor::CONTROL as usize]);
        }
        match self.state.quiesce(QUIESCE) {
            Some(mut g) => {
                let id = g.identity();
                let _ = writeln!(
                    out,
                    "  board: {} revision {:#x} at widget {}, {} MB in {} banks, identity {:#010x}, back end version {}",
                    g.widget.product, g.widget.revision, g.slot, g.widget.memory_size, g.widget.memory_banks, id, g.widget.backend_version
                );
                let _ = writeln!(
                    out,
                    "  scanout {}x{}, {} clocks a frame at {} Hz",
                    g.scanout.mode.width, g.scanout.mode.height, g.scanout.mode.frame_clocks, g.scanout.mode.clock_hz
                );
                let _ = writeln!(
                    out,
                    "  drawing: draw mask {:#x} draw word {:#010x} read word {:#010x} dest {:#010x} plane write mask {:#010x}",
                    g.draw_mask, g.draw_word, g.read_word, g.dest_buffer_word, g.plane_write_mask
                );
                let _ = writeln!(
                    out,
                    "  pixel mode {:#x} {:#x} {:#010x} {:#x}; window origin ({}, {})",
                    g.pixel_mode[0], g.pixel_mode[1], g.pixel_mode[2], g.pixel_mode[3], g.win.origin[0], g.win.origin[1]
                );
                let _ = writeln!(
                    out,
                    "  assembly: context {:#x}, {}, {} vertices held, texture path {:#x}, dma sequence {}",
                    g.assembly.hw_id,
                    if g.assembly.in_begin { format!("in {}", begin_name(g.assembly.begin_mode)) } else { "no primitive open".to_string() },
                    g.assembly.nheld,
                    g.tex_regs().path,
                    g.dma_seq
                );
                let _ = writeln!(
                    out,
                    "  not modelled: {} offsets, {} command words reported (vpro unimp)",
                    g.diag.reported_addr_count.load(std::sync::atomic::Ordering::Relaxed),
                    g.diag.reported_cmd_count
                );
            }
            None => {
                let _ = writeln!(out, "  board busy: state not read (FIFO {} words, raster {} jobs)", fifo.queued_words(), raster.pending());
            }
        }
        let _ = writeln!(out, "  {}", self.state.trace.status().unwrap_or_else(|| "trace not active".into()));
        let _ = writeln!(out, "  {}", self.rec_status());
        Ok(out)
    }

    // ── recording ────────────────────────────────────────────────────────────

    fn rec_status(&self) -> String {
        match self.state.rec.lock().as_ref() {
            Some(r) => format!("recording to {} ({} records, {} marks)", r.path, r.records, r.marks),
            None => "not recording".to_string(),
        }
    }

    /// `vpro rec`: start, mark or stop the recording (record.rs). A start
    /// and a stop mark the memory they start from and end with, a mark adds
    /// one; all three want the board idle (`quiesce`), with the processors
    /// held off so no store lands between the hash and the record. Lock
    /// order: the processors' lock (the guard), then the recorder.
    fn rec_text(&self, args: &[&str]) -> Result<String, String> {
        match args {
            [] => Ok(self.rec_status() + "\n"),
            ["off"] => {
                let g = self.state.quiesce(QUIESCE);
                let closing = g.as_deref().map(memory_hash_bytes);
                if let Some(h) = closing {
                    if let Some(r) = self.state.rec.lock().as_mut() {
                        r.mark(h);
                    }
                }
                let Some(r) = self.state.rec.set(None) else { return Err("vpro rec: no recording running".into()) };
                let note = if closing.is_some() { "" } else { "; board busy: no closing mark" };
                Ok(format!("recording stopped: {} ({} records, {} marks){note}\n", r.path, r.records, r.marks))
            }
            ["mark"] => {
                let g = self.state.quiesce(QUIESCE).ok_or("vpro rec mark: board busy, not marked")?;
                let h = memory_hash_bytes(&g);
                let mut slot = self.state.rec.lock();
                let r = slot.as_mut().ok_or("vpro rec: no recording running")?;
                r.mark(h);
                Ok(format!("mark {} at record {}: {}\n", r.marks, r.records, blake3::Hash::from(h).to_hex()))
            }
            [path] => {
                if self.state.rec.active() {
                    return Err(format!("vpro rec: already {} (vpro rec off first)", self.rec_status()));
                }
                let g = self.state.quiesce(QUIESCE).ok_or("vpro rec: board busy, not started")?;
                let mut r = Recorder::create(path).map_err(|e| format!("vpro rec {path}: {e}"))?;
                r.mark(memory_hash_bytes(&g));
                self.state.rec.set(Some(r));
                Ok(format!("recording to {path} (start state marked)\n"))
            }
            _ => Err("vpro rec [<file>|mark|off]".into()),
        }
    }

    fn regs_text(&self) -> Result<String, String> {
        let mut out = String::new();
        {
            let f = self.state.front.lock();
            let _ = writeln!(out, "host interface (no wait):");
            let _ = writeln!(out, "  identity {:#010x}  widget control {:#010x}  xt req timeout {:#x}  error command {:#x}", f.identity, f.widget_control, f.xt_req_timeout, f.error_command);
            let _ = writeln!(out, "  int dest {:#010x}:{:#010x}  flags {:#010x}  enable {:#010x}  asserted {}  held {}", f.int_dest_upper, f.int_dest_lower, f.intr_flags, f.intr_enable, f.interrupt_asserted, f.flag_held);
            let _ = writeln!(out, "  route dest {:#010x}:{:#010x}  vectors {:#010x} (softsync {} retrace {} dma {} flag {})  routed {}",
                f.route_dest_upper, f.route_dest_lower, f.route_vectors, (f.route_vectors >> 24) & 0xff, (f.route_vectors >> 16) & 0xff, (f.route_vectors >> 8) & 0xff, f.route_vectors & 0xff, f.routed());
            let _ = writeln!(out, "  swap arm {:#x} pending {:#x} control {:#x} companion {:#x} set {:#x} clear {:#x}  displayed {:#010x}{}",
                f.swap_arm, f.swap_pending, f.swap_control, f.swap_control_companion, f.swap_set, f.swap_clear, f.displayed_buffer, if f.displayed_written { " (written, repaint pending)" } else { "" });
            let _ = writeln!(out, "  sw token {:#x}  flow-control target {:#x}  heart widget {}", f.sw_token, f.flctl_target, f.heart_widget);
        }
        let g = self.quiesced()?;
        let wg = &g.widget;
        let _ = writeln!(out, "widget ({} rev {:#x}):", wg.product, wg.revision);
        let _ = writeln!(out, "  memory config {:#010x}  memory control {:#010x}  flctl config {:#010x}  cas latency {}", wg.memory_config, wg.memory_control, wg.flctl_config, wg.cas_latency);
        let _ = writeln!(out, "  fifo setup 0x6b0-0x6b3,0x7b5: {:x?}  0x7f0 {:#x}", wg.fifo_setup, wg.setup_7f0);
        let _ = writeln!(out, "  host write addr {:#x}:{:#x}  stats addr {:#x}:{:#x} control {:#x}  ctx shadow {:#x}:{:#x}",
            wg.host_write_addr_upper, wg.host_write_addr_lower, wg.stats_addr_upper, wg.stats_addr_lower, wg.stats_control, wg.ctx_shadow_addr_upper, wg.ctx_shadow_addr_lower);
        let _ = writeln!(out, "  buzz {:x?}  nic {:#x} (serial {:#x})  i2c main {:#x} opt {:#x}", wg.buzz, wg.nic, wg.nic_serial, wg.i2c_main_cmd, wg.i2c_opt_cmd);
        let _ = writeln!(out, "  back end: index {:#x} count {}  version {}  selftest progress {} ready {}", wg.backend_index, wg.backend_count, wg.backend_version, wg.selftest_progress, wg.cfifo_selftest_ready);
        let _ = writeln!(out, "  product record: {}", if wg.nic_record.is_empty() { &wg.product_record } else { &wg.nic_record });
        Ok(out)
    }

    fn fifo_text(&self) -> Result<String, String> {
        let mut out = String::new();
        let fifo = &self.state.fifo;
        let raster = &self.state.raster;
        let _ = writeln!(out, "command FIFO: level {} words, {} queued in {} stores, high water {}, pending {}, thread {}",
            fifo.level(), fifo.queued_words(), fifo.stores(), fifo.high_water(), fifo.pending(),
            if fifo.running.load(std::sync::atomic::Ordering::SeqCst) { "running" } else { "stopped" });
        let _ = writeln!(out, "raster queue: {} jobs pending, {} pixels queued, thread {}", raster.pending(), raster.queued(), if raster.threaded() { "running" } else { "stopped" });
        let g = self.quiesced()?;
        let c = &g.cmd;
        let _ = writeln!(out, "parser: state {} tag {:#x} operands {}/{} from kernel {}", c.state, c.tag, c.operand_index, c.operand_count, c.from_kernel);
        if c.operand_count != 0 {
            let _ = writeln!(out, "  operands: {:x?}", &c.operand[..(c.operand_count as usize).min(c.operand.len())]);
        }
        let s = &g.stream;
        let _ = writeln!(out, "2D stream: primitive {:#x} colour {:#x} {} vertices held, pattern shape {:#x} mode {:#x}", s.primitive, s.color, s.vertex_count, s.pattern_shape, s.pattern_mode);
        let p = &g.prim;
        let _ = writeln!(out, "open primitive: in_prim {} begin_open {} integer {} held {} restart armed {}", p.in_prim, p.begin_open, p.integer_prim, p.open_held, p.restart_armed);
        let im = &g.image;
        if im.active {
            let _ = writeln!(out, "image raster: ({}, {})-({}, {}) at ({}, {}) fmt {:#x} code {:#x} group {}/{}", im.x1, im.y1, im.x2, im.y2, im.cur_x, im.cur_y, im.fmt, im.write_code, im.group_painted, im.group_len);
        }
        let q = &g.query;
        let _ = writeln!(out, "query: stage {} kind {} count {} epoch {} base {:#x} sync base {:#x}", q.stage, q.kind, q.count, q.epoch, q.base, q.sync_base);
        Ok(out)
    }

    fn state_text(&self) -> Result<String, String> {
        use super::pipe::backend::BackendState;
        use super::pipe::shader::ShaderState;

        let g = self.quiesced()?;
        let a = &g.assembly;
        let xf = &a.xf;
        let w = g.fragment_words();
        let r = &w.raster;
        let mut out = String::new();
        let _ = writeln!(out, "assembly: context {:#x}, {}, {} vertices held{}", a.hw_id,
            if a.in_begin { format!("in {}", begin_name(a.begin_mode)) } else { "no primitive open".to_string() },
            a.nheld, if a.strip_odd { ", odd strip triangle" } else { "" });
        let _ = writeln!(out, "  current colour {:?} texture coordinates {:?} edge {}", a.colour, a.texcoord, a.edge);
        let _ = writeln!(out, "transform: viewport {:?} depth range {:?} user clip planes {:#x}", xf.viewport, xf.depth_range, xf.clip_planes);
        for i in (0..CLIP_PLANES as usize).filter(|&i| xf.clip_planes & (1 << i) != 0) {
            let _ = writeln!(out, "  clip plane {i}: {:?}", xf.clip_plane[i]);
        }
        fmt_mat4(&mut out, "projection (column-major)", &xf.projection, true);
        fmt_mat4(&mut out, "texture (column-major)", &a.texture, true);
        let _ = writeln!(out, "raster words: mode {:#010x} cull {:?} front cw {} flat {}", r.mode, r.cull(), r.front_cw(), r.flat());
        let _ = writeln!(out, "  smooth points {} lines {} polygons {}  polygon modes {:?}/{:?}  offset {} steps (fill {} line {} point {})",
            on(r.point_smooth()), on(r.line_smooth()), on(r.polygon_smooth()), r.polygon_mode_front(), r.polygon_mode_back(),
            r.offset_steps(), on(r.offset_fill()), on(r.offset_line()), on(r.offset_point()));
        let _ = writeln!(out, "  line width {} point size {} (x128)  line stipple {} {:#06x} x{}  polygon stipple {}  scissor {:?}",
            r.line_width_x128(), r.point_size_x128(), on(r.line_stipple_on()), r.line_stipple_pattern(), r.line_stipple_repeat(),
            on(r.polygon_stipple().is_some()), r.scissor());
        let _ = writeln!(out, "  window origin {:?}, {} clip rects, id compare {:#x}", r.origin, r.clip_kept, r.id_compare);
        let _ = writeln!(out, "shader: {:?}", ShaderState::of(&w.shader, g.tex_regs()));
        let _ = writeln!(out, "back end: {:?}", BackendState::of(&w.backend, true, !g.gl_index_mode()));
        let _ = writeln!(out, "board words:");
        let _ = writeln!(out, "  draw mask {:#x}  dest {:#010x}  plane write mask {:#010x}  colour mode {:#010x} {:#010x}", g.draw_mask, g.dest_buffer_word, g.plane_write_mask, g.colour_mode[0], g.colour_mode[1]);
        let _ = writeln!(out, "  depth func {:#x} write mask {:#010x}  blend word {:#010x} constant colour {:#010x} {:#010x}", g.gl_depth_func, g.gl_write_mask, g.blend_word, g.const_colour[0], g.const_colour[1]);
        let _ = writeln!(out, "  clip box {:#010x} {:#010x}  fog words {:x?}", g.gl_clip_box[0], g.gl_clip_box[1], g.gl_fog);
        let _ = writeln!(out, "  colour index {:#x} ({})", g.gl_color_index, if g.gl_color_is_index { "index fills" } else { "RGB fills" });
        Ok(out)
    }

    fn geom_text(&self) -> Result<String, String> {
        let g = self.quiesced()?;
        let ge = &g.geom;
        let mut out = String::new();
        let _ = writeln!(out, "geometry engine: lit {} local viewer {} normalize {} colour overridden {} colour material {:#x}", ge.lit, ge.local_viewer, ge.normalize, ge.colour_overridden, ge.colour_material);
        let _ = writeln!(out, "  program addr {:#x} half {} entries {:x?} constant index {:#x} linear texgen {:#x}", ge.program_addr, ge.program_half, ge.entry, ge.constant_index, ge.linear_texgen);
        let _ = writeln!(out, "  light colour select {:#x} light vector select {:#x} shininess sqrt {}", ge.light_colour_sel, ge.light_vec_sel, f32::from_bits(ge.shininess_sqrt));
        let _ = writeln!(out, "  current colour [{}] normal [{}]", f32s(&ge.colour), f32s(&ge.normal));
        fmt_mat4_bits(&mut out, "modelview (row-major as sent)", &ge.modelview, false);
        fmt_mat4_bits(&mut out, "normal matrix", &ge.normal_matrix, false);
        let _ = writeln!(out, "  material: [{}]", f32s(&ge.material));
        let _ = writeln!(out, "  classes 0x20-0x35: {:x?}", ge.regs_20_35);
        let nonzero = ge.constants.iter().filter(|&&c| c != 0).count();
        let _ = writeln!(out, "  constants: {nonzero} of {} nonzero", ge.constants.len());
        // The non-zero entries by index: the program lists the walk reads.
        for (i, &c) in ge.constants.iter().enumerate().filter(|(_, &c)| c != 0).take(64) {
            let _ = writeln!(out, "    [{i:#05x}] {c:#06x}");
        }
        let _ = writeln!(out, "  eye clip planes: {}", (0..ge.clip_eye.len() / 4).map(|i| format!("[{}]", f32s(&ge.clip_eye[i * 4..i * 4 + 4]))).collect::<Vec<_>>().join(" "));
        Ok(out)
    }

    fn tex_text(&self) -> Result<String, String> {
        let g = self.quiesced()?;
        let t = g.tex_regs();
        let mut out = String::new();
        let _ = writeln!(out, "path {:#010x}: target {:?}, clip planes {}", t.path, t.target(), t.clip_planes());
        let _ = writeln!(out, "  sampler {:08x?}", t.sampler);
        let _ = writeln!(out, "    log2 size {:?} wrap {:?} {:?} {:?} min {:?} mag {:?} class {:?} stored border {} colour table {:?} border {:03x?}",
            t.log2(), t.wrap(0), t.wrap(1), t.wrap(2), t.min(), t.mag(), t.class(), t.stored_border(), t.colour_table(), t.border());
        let _ = writeln!(out, "  env {:08x?} ({:?}, colour {:03x?})  level0 {:#x}  lod {:08x?}", t.env, t.env(), t.env_colour(), t.level0, t.lod);
        let _ = writeln!(out, "  scale/bias {:08x?}  offset {:08x?}", t.scale_bias, t.offset);
        let v = &g.tex;
        let _ = writeln!(out, "  vertex: coord {:08x?} r {:#x} q {:#x}  linear valid {:#x}", v.coord, v.coord_r, v.coord_q, v.linear_valid);
        let l = &g.slots.cur().latch;
        let _ = writeln!(out, "  latch{}: {} levels, lod {} [{}, {}] mag below {} scale {:?} bias scale {:?} clamp {:?} unimplemented {:#x}",
            if g.slots.cur().dirty { " (stale)" } else { "" }, l.count, l.lod, l.min_lod, l.max_lod, l.mag_below, l.scale, l.lod_scale, l.clamp, l.unimplemented);
        for (i, lv) in l.level[..l.count as usize].iter().enumerate() {
            let _ = writeln!(out, "    level {i:2}: {lv:x?}");
        }
        let _ = writeln!(out, "  level table (0x0e80 + 3n; entries with any word set):");
        for i in 0..ODSY_TEX_LEVELS {
            let e = t.entry(i);
            if e != [0; 3] {
                let _ = writeln!(out, "    {i:2}: {:08x} {:08x} {:08x}  index {} size code {}", e[0], e[1], e[2], (e[1] >> 20) & 0x3f, e[1] >> 30);
            }
        }
        Ok(out)
    }

    fn win_text(&self) -> Result<String, String> {
        let g = self.quiesced()?;
        let wn = &g.win;
        let mut out = String::new();
        let _ = writeln!(out, "window: origin ({}, {}) read origin ({}, {}){}", wn.origin[0], wn.origin[1], wn.read_origin[0], wn.read_origin[1], if wn.origin_pending { " (origin pending)" } else { "" });
        let _ = writeln!(out, "  id compare {:#010x}: {} mask {:#04x} value {:#04x}", wn.id_compare,
            if wn.id_compare & super::consts::ODSY_ID_COMPARE_ENABLE != 0 { "enabled" } else { "disabled" },
            (wn.id_compare >> super::consts::ODSY_ID_COMPARE_MASK_SHIFT) & 0xff, wn.id_compare & 0xff);
        let _ = writeln!(out, "  clip rectangles kept {:#x} (bit i: rectangle i):", wn.clip_kept);
        for (i, r) in wn.clip_rect.iter().enumerate() {
            let _ = writeln!(out, "    {i}: ({}, {})-({}, {}) words {:#010x} {:#010x}{}", r.x0, r.y0, r.x1, r.y1, wn.clip_words[i * 2], wn.clip_words[i * 2 + 1], if wn.clip_kept & (1 << i) != 0 { "" } else { " (not kept)" });
        }
        let _ = writeln!(out, "  committed: origin ({}, {}) id compare {:#010x} clip words {:08x?}", wn.committed_origin[0], wn.committed_origin[1], wn.committed_id_compare, wn.committed_clip);
        let _ = writeln!(out, "  prior origin ({}, {}) after tail {}", wn.prior_origin[0], wn.prior_origin[1], wn.after_tail);
        Ok(out)
    }

    /// Board memory as the registers name it, for `vpro mem` and `fbdump`'s
    /// layout.txt.
    fn layout_text(b: &Board) -> String {
        let mut out = String::new();
        let pm = &b.pixel_mode;
        let f = super::vram::Format::decode(pm[2]);
        let d = b.dest_views();
        let _ = writeln!(out, "board memory: {} MiB", b.vram.size() >> 20);
        let _ = writeln!(out, "pixel mode (0x017): origin {:#x} stride {} tiles status {:#010x} zcull {:#x}", pm[0], pm[1] & 0xfff, pm[2], pm[3]);
        let _ = writeln!(out, "  FB_DEPTH {} bytes, WIN_DEPTH {} bytes, PIXEL_FMT {}, FIELD_FMT {}, index {}, scratch {}", 1u32 << f.fb_log2, 1u32 << f.win_log2, f.pixel_fmt, f.field_fmt, f.index, f.scratch);
        let _ = writeln!(out, "  fields: A {:?}  B {:?}  depth {:?}  stencil {:?}", d.colour[0].kind, d.colour[1].kind, d.depth.kind, d.stencil.kind);
        let _ = writeln!(out, "buffer select (0x024) {:#x}, draw word {:#010x}, read word {:#010x}", b.draw_mask, b.draw_word, b.read_word);
        let ctl = |i: u32| b.widget.backend_ctl[(i - backend::CTL_BASE) as usize];
        let _ = writeln!(out, "scan-out (PB&J): window-id base {:#x}, pixel base {:#x}, {}x{}, stride {} tiles, depth code {} ({} bytes)",
            ctl(0x2800), ctl(0x2801), ctl(0x2802), ctl(0x2803), ctl(0x2807), ctl(0x2809), 2u32 << (ctl(0x2809) & 7));
        let _ = writeln!(out, "scanout {}x{}, {} clocks a frame at {} Hz; displayed-buffer bits {:#010x}",
            b.scanout.mode.width, b.scanout.mode.height, b.scanout.mode.frame_clocks, b.scanout.mode.clock_hz, b.front.lock().displayed_buffer);
        out
    }

    fn mem_text(&self) -> Result<String, String> {
        let g = self.quiesced()?;
        Ok(Self::layout_text(&g))
    }

    fn backend_text(&self, start: u32, n: u32) -> Result<String, String> {
        let g = self.quiesced()?;
        let ctl = &g.widget.backend_ctl;
        let first = start.saturating_sub(backend::CTL_BASE).min(backend::CTL_ENTRIES) as usize;
        let last = (first + n as usize).min(ctl.len());
        let mut out = String::new();
        let _ = writeln!(out, "back-end register file (PB&J), index {:#x} count {}; entries {:#x}..{:#x} that are set:", g.widget.backend_index, g.widget.backend_count, backend::CTL_BASE + first as u32, backend::CTL_BASE + last as u32);
        let mut shown = 0;
        for i in first..last {
            if ctl[i] == 0 {
                continue;
            }
            if shown == 256 {
                let _ = writeln!(out, "  ... more; narrow the range");
                break;
            }
            shown += 1;
            let index = backend::CTL_BASE + i as u32;
            let what = match index {
                i if (dispmode::BASE..dispmode::BASE + dispmode::COUNT).contains(&i) => format!("  display mode {} ({})", i - dispmode::BASE, dispmode_text(ctl[i as usize - backend::CTL_BASE as usize])),
                i if (overlay::MODE_BASE..overlay::MODE_BASE + overlay::IDS).contains(&i) => format!("  overlay mode {}", i - overlay::MODE_BASE),
                i if (cursor::BASE..cursor::BASE + cursor::ENTRIES).contains(&i) => format!("  cursor {:#x}", i - cursor::BASE),
                _ => String::new(),
            };
            let _ = writeln!(out, "  {index:#06x} = {:#010x}{what}", ctl[i]);
        }
        if shown == 0 {
            let _ = writeln!(out, "  (none)");
        }
        let _ = writeln!(out, "the lookup RAM, display modes, overlay modes and cursor the display reads: vpro lut, dispmode, cursor");
        Ok(out)
    }

    fn unimp_text(&self) -> Result<String, String> {
        let g = self.quiesced()?;
        let d = &g.diag;
        let mut out = String::new();
        let n = (d.reported_addr_count.load(std::sync::atomic::Ordering::Relaxed) as usize).min(d.reported_addr.len());
        let _ = writeln!(out, "offsets reported as not modelled ({n}; W = write):");
        for a in &d.reported_addr[..n] {
            let t = a.load(std::sync::atomic::Ordering::Relaxed);
            let _ = writeln!(out, "  {:#x}{}", t & !super::consts::ODSY_UNIMP_WRITE, if t & super::consts::ODSY_UNIMP_WRITE != 0 { " W" } else { "" });
        }
        let m = (d.reported_cmd_count as usize).min(d.reported_cmd.len());
        let _ = writeln!(out, "command words reported as not modelled ({m}):");
        for w in &d.reported_cmd[..m] {
            let _ = writeln!(out, "  {w:#010x}");
        }
        Ok(out)
    }

    // ── pixel commands ───────────────────────────────────────────────────────

    fn lut_text(&self, start: u32, n: u32) -> String {
        let regs = self.state.shared.regs.read();
        let start = start.min(lut::SIZE);
        let end = (start + n).min(lut::SIZE);
        let mut out = String::new();
        let _ = writeln!(out, "colour lookup RAM {start:#x}..{end:#x} (groups of 256: 0 index, 1-24 indexed windows, 25 unverified, 26+ the gamma ramp at {:#x}); ten-bit channels, the top eight shown:", ODSY_GAMMA_GROUP * 256);
        for i in start..end {
            let w = regs.lut[i as usize];
            let _ = writeln!(out, "  [{i:#06x}] {w:#010x}  r {:4} g {:4} b {:4}  -> {:06x}", (w >> (lut::RED_SHIFT - 2)) & 0x3ff, (w >> (lut::GREEN_SHIFT - 2)) & 0x3ff, (w >> (lut::BLUE_SHIFT - 2)) & 0x3ff, regs.lut_rgb(i));
        }
        out
    }

    fn dispmode_text(&self) -> Result<String, String> {
        let (frame, mem, regs) = self.screen()?;
        let displayed = self.displayed();
        let (width, height) = (frame.width, frame.height);
        let (_, wid) = frame.surfaces();
        // The id bytes on screen: count and box per distinct byte.
        let mut count = [0u64; 256];
        let mut bbox = [(u32::MAX, u32::MAX, 0u32, 0u32); 256];
        for y in 0..height {
            for x in 0..width {
                let d = mem.load(wid.offset(x, y), 1) as usize;
                count[d] += 1;
                let b = &mut bbox[d];
                *b = (b.0.min(x), b.1.min(y), b.2.max(x), b.3.max(y));
            }
        }
        let mut main_pixels = [0u64; 32];
        for d in 0..256 {
            main_pixels[d >> ODSY_DID_MAIN_SHIFT] += count[d];
        }
        let mut out = String::new();
        let _ = writeln!(out, "display modes ({:#x} + id), displayed-buffer bits {displayed:#010x}:", dispmode::BASE);
        for (id, &m) in regs.dispmode.iter().enumerate() {
            if m == 0 && main_pixels[id] == 0 {
                continue;
            }
            let _ = writeln!(out, "  id {id:2}: {m:#010x}  {}  shows buffer {}  {} pixels", dispmode_text(m), (displayed >> id) & 1, main_pixels[id]);
        }
        let _ = writeln!(out, "overlay modes ({:#x} + id):", overlay::MODE_BASE);
        for id in 0..overlay::IDS as usize {
            let m = regs.overlay_mode[id];
            if m == 0 {
                continue;
            }
            let _ = writeln!(out, "  id {id}: {m:#010x}  {}  lut group {}", if m & overlay::MODE_ENABLE != 0 { "enabled" } else { "disabled" }, (m >> overlay::MODE_GROUP_SHIFT) & overlay::MODE_GROUP_MASK);
        }
        let _ = writeln!(out, "id bytes on screen ({width}x{height}):");
        for d in 0..256 {
            if count[d] == 0 {
                continue;
            }
            let b = bbox[d];
            let _ = writeln!(out, "  {d:#04x}: main {:2} overlay {}  {:>8} pixels  ({}, {})-({}, {})", d >> ODSY_DID_MAIN_SHIFT, d as u32 & ODSY_DID_OVERLAY_MASK, count[d], b.0, b.1, b.2, b.3);
        }
        Ok(out)
    }

    fn cursor_text(&self) -> String {
        let regs = self.state.shared.regs.read();
        let control = regs.cursor[cursor::CONTROL as usize];
        let mut out = String::new();
        let _ = writeln!(out, "cursor control {control:#010x}: enable {} crosshair {} channel 1 {}  position ({}, {})",
            control & cursor::CTL_ENABLE != 0, control & cursor::CTL_CROSSHAIR != 0, control & cursor::CTL_CHANNEL1 != 0, regs.cursor_x(), regs.cursor_y());
        let colours: Vec<String> = (0..cursor::COLOUR_COUNT as usize).map(|i| format!("{:06x}", super::registers::cursor_rgb(regs.cursor[cursor::COLOURS as usize + i]))).collect();
        let _ = writeln!(out, "  colours: {}", colours.join(" "));
        match self.state.shared.regs.hw_cursor() {
            Some(c) => {
                let _ = writeln!(out, "  glyph ({}x{}, # opaque):", c.size, c.size);
                for m in &c.mask {
                    let row: String = (0..c.size).map(|x| if m >> x & 1 != 0 { '#' } else { '.' }).collect();
                    let _ = writeln!(out, "    {row}");
                }
            }
            None => {
                let _ = writeln!(out, "  no glyph drawn (off, on channel 1, or a crosshair)");
            }
        }
        out
    }

    fn pix_text(&self, x: u32, y: u32) -> Result<String, String> {
        let (frame, mem, regs) = self.screen()?;
        if x >= frame.width || y >= frame.height {
            return Err(format!("vpro pix: ({x}, {y}) is outside the {}x{} screen", frame.width, frame.height));
        }
        let p = frame.parts(mem, x, y).ok_or_else(|| format!("vpro pix: ({x}, {y}) is outside board memory"))?;
        let displayed = self.displayed();
        let mode = regs.dispmode[p.id as usize];
        let fields = super::frame::mode_fields(mode);
        let omode = regs.overlay_mode[(p.did & ODSY_DID_OVERLAY_MASK) as usize];
        let (fb, _) = frame.surfaces();
        let mut out = String::new();
        let _ = writeln!(out, "({x}, {y}) window-id byte {:#04x}: main id {}, overlay id {}; overlay byte {:#04x}", p.did, p.id, p.did & ODSY_DID_OVERLAY_MASK, p.overlay);
        let _ = writeln!(out, "  frame-buffer pixel at {:#x}: {}", fb.offset(x, y), p.bytes.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "));
        let _ = writeln!(out, "  display mode[{}] {mode:#010x}: {}; A field byte {} {:#010x}, B field byte {} {:#010x}; displayed bits {displayed:#010x} -> {} shows",
            p.id, dispmode_text(mode), fields[0].at, p.fields[0], fields[1].at, p.fields[1], if p.shown == 0 { "A" } else { "B" });
        let _ = writeln!(out, "  overlay mode {omode:#010x}: {}", if omode & overlay::MODE_ENABLE != 0 { "enabled" } else { "disabled" });
        let covered = self.state.shared.regs.hw_cursor().is_some_and(|c| {
            let (dx, dy) = (x as i32 - c.x, y as i32 - c.y);
            (0..c.size as i32).contains(&dx) && (0..c.size as i32).contains(&dy) && c.mask[dy as usize] >> dx & 1 != 0
        });
        let _ = writeln!(out, "  screen {:06x}{}", p.rgb & 0xff_ffff, if covered { " (under the cursor)" } else { "" });
        Ok(out)
    }

    fn raw_text(&self, x: u32, y: u32, wd: u32, ht: u32) -> Result<String, String> {
        let (frame, mem, _) = self.screen()?;
        let (fb, wid) = frame.surfaces();
        let mut out = String::new();
        let _ = writeln!(out, "frame-buffer pixels ({} bytes each) | window-id buffer pixels (id byte, overlay byte):", fb.bytes_per_pixel());
        for yy in y..y + ht.max(1) {
            for xx in x..x + wd.max(1) {
                let get = |s: &super::vram::Surface| {
                    let o = s.offset(xx, yy);
                    (0..s.bytes_per_pixel() as u64).map(|i| format!("{:02x}", mem.load(o + i, 1))).collect::<String>()
                };
                let _ = writeln!(out, "  ({xx:4}, {yy:4}) {} | {}", get(&fb), get(&wid));
            }
        }
        Ok(out)
    }

    /// `vpro peek`: board memory bytes from an address in 16-byte units.
    fn peek_text(&self, addr: u64, n: u64) -> Result<String, String> {
        let mem = self.state.shared.vram.get().ok_or_else(|| "vpro: board memory is not allocated".to_string())?;
        // Read only, as the display reads.
        let start = addr * UNIT;
        if start >= mem.size() {
            return Err(format!("vpro peek: {addr:#x} is past the end of board memory"));
        }
        let end = (start + n).min(mem.size());
        let bytes: Vec<u8> = (start..end).map(|a| mem.load(a, 1) as u8).collect();
        let mut out = String::new();
        for (i, chunk) in bytes.chunks(16).enumerate() {
            let _ = writeln!(out, "  {:#010x}: {}", start + 16 * i as u64, chunk.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "));
        }
        Ok(out)
    }

    /// `vpro shot`: the screen composed from board memory now.
    fn shot(&self, path: &str) -> Result<String, String> {
        let (frame, mem, regs) = self.screen()?;
        let (width, height) = (frame.width as usize, frame.height as usize);
        let mut px = vec![0u32; width * height];
        frame.compose(mem, &regs, &mut px, width, height);
        // save_png takes 0x00BBGGRR.
        let px: Vec<u32> = px.iter().map(|&p| (p & 0xff) << 16 | (p & 0xff00) | p >> 16).collect();
        let (w, h) = crate::hostinput::save_png(path, &px, width, height, None)?;
        Ok(format!("saved {path} {w}x{h}\n"))
    }

    /// `vpro fbdump`: the composed screen as PNG, the whole of board memory
    /// as board.bin, and the registers that name it in layout.txt.
    fn fbdump(&self, dir: &Path) -> Result<String, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("vpro fbdump {}: {e}", dir.display()))?;
        let io = |e: std::io::Error| format!("vpro fbdump: {e}");
        let (frame, mem, regs) = self.screen()?;
        let (w, h) = (frame.width as usize, frame.height as usize);
        let mut px = vec![0u32; w * h];
        frame.compose(mem, &regs, &mut px, w, h);
        let mut rgb = Vec::with_capacity(w * h * 3);
        for &p in &px {
            rgb.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8]);
        }
        write_png(&dir.join("screen.png"), w, h, &rgb, png::ColorType::Rgb).map_err(io)?;
        std::fs::write(dir.join("board.bin"), mem.address_bytes().collect::<Vec<u8>>()).map_err(io)?;
        let mut layout = match self.state.quiesce(QUIESCE) {
            Some(g) => Self::layout_text(&g),
            None => "board busy: the registers were not read; vpro mem\n".to_string(),
        };
        let _ = writeln!(layout, "files: screen.png (composed), board.bin (board memory, addresses x 16)");
        layout.push_str(&self.dispmode_text()?);
        std::fs::write(dir.join("layout.txt"), layout).map_err(io)?;
        Ok(format!("dumped {}/ (screen.png, board.bin, layout.txt) at {w}x{h}\n", dir.display()))
    }
}

#[cfg(test)]
/// Board memory as a file in `dir`, for the replay's `VPRO_REPLAY_DUMP`.
pub(super) fn dump_memory(mem: VramPtr, dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("board.bin"), mem.address_bytes().collect::<Vec<u8>>())
}
