//! HQ3: the host interface and command processor.
//!
//! `Hq3Regs` is the CPU-side register state (microcode RAM, flags and
//! interrupt enables). `Hq3Engine` is the frontend: it parses the command
//! FIFO word stream, runs the command-processor tokens this model handles,
//! the host DMA engine and the formatter, and turns raster commands into
//! raster register writes through an `Hq3Sink`.
//!
//! Drawing arrives through the command FIFO as (command, data) pairs.
//! Commands at `0x1000` and up write raster registers directly, with bit
//! `0x400` meaning "execute"; lower numbers go to the command processor's
//! microcode, which this model does not run (the PROM, the kernel's console
//! and the X server's 2D paths never need it).

use std::sync::Arc;

use super::plain::RegMap;

use crate::traits::BusDevice;

/// Host interface register offsets.
pub mod host {
    pub const UCODE: u32 = 0x40000;
    pub const UCODE_END: u32 = 0x46000;
    pub const SET_FLAGS_PRIVILEGED: u32 = 0x50008;
    pub const CLEAR_FLAGS_PRIVILEGED: u32 = 0x5000C;
    pub const CFIFO_PRIVILEGED: u32 = 0x50080;
    pub const STATUS: u32 = 0x70000;
    pub const FIFOSTATUS: u32 = 0x70004;
    pub const SET_FLAGS: u32 = 0x70008;
    pub const CLEAR_FLAGS: u32 = 0x7000C;
    pub const GE_READBACK_HI: u32 = 0x70010;
    pub const GE_READBACK_LO: u32 = 0x70014;
    pub const CFIFO: u32 = 0x70080;
    /// libGLcore's immediate-mode path stores command words here (HQ3.h
    /// HQ3_CFIFO_GL); taken as another user FIFO port.
    pub const CFIFO_GL: u32 = 0x74080;
    pub const GIOSTATUS: u32 = 0x70100;
    pub const DMABUSY: u32 = 0x70104;
    pub const RASTER: u32 = 0x7C000;
    pub const RASTER_END: u32 = 0x80000;

    /// Status: raster idle and host idle, command and data FIFOs at or below
    /// their low-water marks.
    pub const STATUS_IDLE: u32 = 0x01 | 0x02 | 0x10 | 0x40;
    pub const STATUS_VBLANK: u32 = 0x04;

    /// Flag set by the "set flag" command (0xE04): DMA/sync completion.
    pub const FLAG_DONE: u32 = 1 << 16;
    /// Flag set when the geometry engine has data waiting to be read back.
    pub const FLAG_GE_DATA: u32 = 1 << 17;
    /// Flag set while a geometry engine diagnostic readback has data waiting.
    pub const FLAG_GE_DIAG: u32 = 1 << 18;
    /// Context switch: outgoing context saved (phase 1) and incoming context
    /// loaded (phase 2).
    pub const FLAG_CONTEXT_SAVED: u32 = 1 << 19;
    pub const FLAG_CONTEXT_LOADED: u32 = 1 << 6;
    /// Command-processor flag 0: a scheduled buffer swap has happened.
    pub const FLAG_CP0: u32 = 1 << 10;
    /// Flags that can raise the general interrupt.
    pub const INTR_CAUSES: u32 = 0x7F_FFFF;

    /// Flag and interrupt enables: read at the first address, write-1-to-set
    /// there, write-1-to-clear at the second.
    pub const FLAG_ENABLE_SET: u32 = 0x50010;
    pub const FLAG_ENABLE_CLEAR: u32 = 0x50014;
    pub const INTERRUPT_ENABLE_SET: u32 = 0x50018;
    pub const INTERRUPT_ENABLE_CLEAR: u32 = 0x5001C;
    /// Context switch request: starts the switch routine at the written
    /// microcode address.
    pub const CONTEXT_SWITCH: u32 = 0x50050;
    /// Words of incoming context the host pushes after the save phase.
    pub const CONTEXT_SWITCH_WORDS: u32 = 63;

    /// Geometry engine diagnostic ports, per engine: data, then address.
    pub const GE_DIAG: [(u32, u32); 2] = [(0x50040, 0x50044), (0x50048, 0x5004C)];
    /// Diagnostic readback words: a discarded word, then the data word.
    pub const GE_DIAG_READ_PAD: u32 = 0x50230;
    pub const GE_DIAG_READ: u32 = 0x5022C;

    /// Host DMA engine and raster-interface context, read back per register.
    pub const DMA_CONTEXT: u32 = 0x50300;
    pub const RASTER_IF_CONTEXT: u32 = 0x50200;
    /// PIO pixel reads: the raster char registers, high word at the execute
    /// alias (which takes the next doubleword), low word at the plain one.
    pub const PIO_READ_HI: u32 = 0x7D1C0;
    pub const PIO_READ_LO: u32 = 0x7C1C4;
}

/// Context image word 1: the window state in words 2-13 is new.
const CTX_WINDOW_CHANGED: u32 = 1 << 29;
/// Context image word 1: the context's first load since it was created.
const CTX_FIRST_LOAD: u32 = 1 << 31;

/// Words of external RAM (ERAM) behind each GE11.
pub const ERAM_WORDS: usize = 0x10000;
/// The external RAM of the two GE11s. Plain data, valid zeroed.
#[repr(C)]
pub struct Eram {
    pub words: [[u32; ERAM_WORDS]; 2],
}

/// A context's saved state in its ERAM slot (image word 0), our layout: the
/// GE HLE's `Gl` as raw words, then the user-port parser. The real
/// microcode's layout is unknown; only the kernel's view matters (a slot
/// is an opaque block of words it may copy to host memory and back, maybe
/// into another slot).
const GL_WORDS: usize = std::mem::size_of::<super::gl::Gl>().div_ceil(4);
const PARSER_WORDS: usize = 6 + CFIFO_KEEP;
/// The host DMA engine's and raster interface's registers are context
/// state too (the HAG and REIF context RAMs): the kernel installs a page
/// table for one process's transfer, and another process's switch-in must
/// not leave it in place when the first comes back (snoop's pixel
/// transfers ran through a parked table and scribbled on low memory).
const IF_WORDS: usize = 0x80 + 0x10;
const SLOT_RECORD_WORDS: usize = GL_WORDS + PARSER_WORDS + IF_WORDS;
/// IRIX 6.5.22's slot size (MgrasGlContextAlloc; slots traced 0x17C7 words
/// apart): our record must fit.
const KERNEL_SLOT_WORDS: usize = 0x17C7;
const _: () = assert!(SLOT_RECORD_WORDS <= KERNEL_SLOT_WORDS);
/// Control word data conversion (bits 29:23), as libGLcore's vertex,
/// colour, normal and index entry points use it (glColor4ub = 0x29,
/// glVertex3s = 0x72, ...): bit 5 converts to float, bit 4 signed, bit 3
/// normalise (OpenGL: unsigned c / (2^b - 1), signed (2c + 1) / (2^b - 1)),
/// bits 2:0 the source size (0 32-bit, 1 byte, 2 short, 4 double); bit 6
/// appends a fourth component of 1.0 (the three-component forms, floats
/// included).
const CONV_ON: u32 = 1 << 5;
const CONV_PAD: u32 = 1 << 6;

/// A command's data words converted to floats (bit patterns), per `conv`.
fn convert(conv: u32, bytes: u32, data: &[u32]) -> Vec<u32> {
    let mut out: Vec<f32> = Vec::new();
    if conv & CONV_ON == 0 {
        out.extend(data.iter().map(|w| f32::from_bits(*w)));
    } else {
        let signed = conv & 0x10 != 0;
        let norm = conv & 0x08 != 0;
        let scale = |v: f64, bits: u32| -> f32 {
            let max = ((1u64 << bits) - 1) as f64;
            (if !norm { v } else if signed { (2.0 * v + 1.0) / max } else { v / max }) as f32
        };
        match conv & 7 {
            4 => {
                for p in data.chunks(2).filter(|p| p.len() == 2) {
                    out.push(f64::from_bits((p[0] as u64) << 32 | p[1] as u64) as f32);
                }
            }
            1 => {
                for i in 0..bytes as usize {
                    let b = (data.get(i / 4).copied().unwrap_or(0) >> (24 - 8 * (i % 4))) & 0xFF;
                    out.push(scale(if signed { b as u8 as i8 as f64 } else { b as f64 }, 8));
                }
            }
            2 => {
                for i in 0..bytes as usize / 2 {
                    let h = (data.get(i / 2).copied().unwrap_or(0) >> (16 - 16 * (i % 2))) & 0xFFFF;
                    out.push(scale(if signed { h as u16 as i16 as f64 } else { h as f64 }, 16));
                }
            }
            _ => {
                for w in data.iter().take(bytes as usize / 4) {
                    out.push(scale(if signed { *w as i32 as f64 } else { *w as f64 }, 32));
                }
            }
        }
    }
    if conv & CONV_PAD != 0 {
        out.push(1.0);
    }
    out.into_iter().map(f32::to_bits).collect()
}

/// Command FIFO command numbers.
mod cmd {
    /// Command-processor token: schedule a buffer swap for the next retrace.
    pub const CP_SCHEDULE_SWAP: u32 = 0x37;
    /// Command-processor token: return the state word at a GE address
    /// (`__Mgr_ReturnMode(addr, 1, 1)`: glGet*, glIsEnabled, and glFinish,
    /// which is a glGetBooleanv(GL_DITHER) round trip). The host clears
    /// FLAG_GE_DATA, sends this, polls for the flag, then reads the answer
    /// at GE_READBACK_LO.
    pub const CP_RETURN_MODE: u32 = 0xA2;
    /// Command-processor token: wait for the pipeline, then return the
    /// state word at the argument's GE address (same readback protocol as
    /// CP_RETURN_MODE). Native IRIS GL getcolor() and clear() request address
    /// 4: the current index as a float, not the address itself.
    pub const CP_SPIN_AND_RETURN: u32 = 0xA1;
    /// Kernel token (MgrasValidateClip, current context): a GL window's
    /// raster state, 15 words: origin, window mode, PP1 window mode (the
    /// kernel stores them as one doubleword, window mode high),
    /// screen masks 4..1 as (x, y) pairs, DRB pointers, 0, then the same
    /// origin and pointers for the second buffer set.
    /// The kernel's and X server's way into GE11 ERAM (MgrasEramRead /
    /// MgrasEramWrite; SGI's diagnostic names CP_PASS_THROUGH_GE_*). Write:
    /// ERAM_WRITE (word count, rounded up to even), the words (FIFO pixel
    /// data, or a host to board DMA), ERAM_WRITE_AT (address, count), and
    /// ERAM_WRITE_END after the last chunk. Read: a board to host DMA armed
    /// first, then ERAM_READ (address, count) feeds it.
    pub const CP_ERAM_WRITE: u32 = 0x0FD;
    pub const CP_ERAM_WRITE_AT: u32 = 0x0FB;
    pub const CP_ERAM_WRITE_END: u32 = 0x0FA;
    pub const CP_ERAM_READ: u32 = 0x0FC;
    /// Kernel, switching to its own context: skip the next GE save.
    pub const CP_GE_NOSAVE_NEXT_SWITCH: u32 = 0x0F4;
    pub const CP_WINDOW: u32 = 0xE4;
    pub const SET_DONE_FLAG: u32 = 0xE04;
    pub const RASTER_BASE: u32 = 0x1000;
    pub const RASTER_EXECUTE: u32 = 0x400;
    pub const DMA_BASE: u32 = 0x800;
    pub const RASTER_IF_BASE: u32 = 0xA00;
    pub const FORMATTER: u32 = 0xC00;
    /// Below this, commands are command-processor microcode tokens.
    pub const CP_LIMIT: u32 = 0x200;
}

/// Host DMA engine registers.
mod dma {
    pub const PAGE_LIST: usize = 0x00;
    pub const STRIDE: usize = 0x04;
    pub const ROW_OFFSET: usize = 0x05;
    pub const ROW_START: usize = 0x06;
    pub const LINES: usize = 0x07;
    pub const LINE_BYTES: usize = 0x08;
    /// In list mode, the bytes taken from the last page.
    pub const LAST_BYTES: usize = 0x09;
    /// The start word: bit 0 run, bits 2:1 pool, bit 3 board to host.
    pub const START: usize = 0x0B;
    /// Display-list branches: eight bytes, the list's offset in the
    /// display-list space (high word) and its length in words (low word).
    /// NO_WAIT branches at once, WAIT once the pipe has drained, PUSH is a
    /// call (the FIFO resumes after the list in every case here).
    pub const DL_JUMP_NO_WAIT: usize = 0x0C;
    pub const DL_JUMP_WAIT: usize = 0x0E;
    pub const DL_PUSH: usize = 0x37;
    pub const DL_RETURN: usize = 0x39;
    /// Page-table base of pool `p`: eight bytes at `TABLE_BASE + 2p`, the
    /// address in the low word.
    pub const TABLE_BASE: usize = 0x20;
    /// The display-list space's page table (eight bytes, address in the low
    /// word) and its last valid offset.
    pub const DL_TABLE_BASE: usize = 0x28;
    pub const MAX_DL_PTR_OFFSET: usize = 0x34;
}

/// The parser slot that display-list words run through.
const DL_PORT: usize = 2;
/// Display lists calling display lists deeper than this are cut off (GL's
/// own limit, GL_MAX_LIST_NESTING, is 64), and so is a list running more
/// segments than this (a loop).
const DL_MAX_DEPTH: usize = 64;
const DL_MAX_SEGMENTS: u32 = 1 << 20;

/// A display-list segment's words from host memory: `offset` in the
/// display-list space, through `table`.
fn fetch_dl_segment(mem: &dyn BusDevice, table: u32, offset: u32, words: u32) -> Option<Vec<u32>> {
    let mut out = Vec::with_capacity(words as usize);
    let mut cached: Option<(u32, u32)> = None;
    for i in 0..words {
        let l = offset.wrapping_add(4 * i);
        let page = l >> 12;
        let frame = match cached {
            Some((p, f)) if p == page => f,
            _ => {
                let r = mem.read32(table.wrapping_add(4 * page));
                if !r.is_ok() {
                    return None;
                }
                cached = Some((page, r.data << 12));
                r.data << 12
            }
        };
        let w = mem.read32(frame | (l & 0xFFF));
        out.push(if w.is_ok() { w.data } else { 0 });
    }
    Some(out)
}

/// Raster instruction: the transfer control register, and the value that
/// starts a board-to-host transfer.
const RSS_XFRCONTROL: u32 = 0x102;
const XFRCONTROL_READ_START: u32 = 9;

/// CPU-side HQ3 registers (the flags are atomic, in `Mgras`). Plain data,
/// valid zeroed.
#[repr(C)]
pub struct Hq3Regs {
    /// Registers not modelled otherwise, read back as written.
    pub regs: RegMap<256>,
    pub ucode: [u32; ((host::UCODE_END - host::UCODE) / 4) as usize],
    pub flag_enable: u32,
    pub interrupt_enable: u32,
    pub ge_readback: [u32; 2],
}

/// What the frontend drives: the raster subsystem, the flags, system memory.
pub trait Hq3Sink {
    /// Write raster register `r`; `exec` runs the IR afterwards.
    fn rss_write(&mut self, r: u32, val: u32, exec: bool);
    /// The armed raster transfer, once everything queued ahead has run:
    /// `Some(read)`, and its (lines, bytes per line).
    fn transfer(&mut self) -> Option<(bool, (u32, u32))>;
    /// DMA into the armed transfer: one line of pixel bytes.
    fn dma_write_line(&mut self, line: u32, bytes: &[u8]);
    /// DMA out of the armed transfer: one line of pixel bytes.
    fn dma_read_line(&mut self, line: u32) -> Vec<u8>;
    fn set_flags(&mut self, bits: u32);
    /// A GL batch starts (true) or ends: the RSS saves / restores the raster
    /// registers the batch loads.
    fn gl_bracket(&mut self, _enter: bool) {}
    /// System memory for DMA, if connected.
    fn sys_mem(&mut self) -> Option<Arc<dyn BusDevice>>;
    /// Something not modelled yet (reported once).
    fn note(&mut self, what: String);
    /// Whether the annotated trace wants HQ3 lines.
    fn tracing(&self) -> bool {
        false
    }
    /// Whether the annotated trace wants texture-load lines.
    fn tracing_tex(&self) -> bool {
        false
    }
    /// One annotated trace line.
    fn trace(&mut self, _line: String) {}
}

/// What the frontend has done, for `mgras stats`.
#[derive(Clone, Copy)]
pub struct Hq3Stats {
    pub words: u64,
    pub raster_writes: u64,
    pub raster_execs: u64,
    /// Command-processor tokens by number (below `CP_LIMIT`).
    pub cp_tokens: [u64; cmd::CP_LIMIT as usize],
    pub set_done: u64,
    pub swaps: u64,
    pub dma_reg_writes: u64,
    pub dma_writes: u64,
    pub dma_reads: u64,
    pub dma_bytes: u64,
    pub formatter_writes: u64,
    pub raster_if_writes: u64,
    pub pixel_cmds: u64,
    pub other_cmds: u64,
    pub context_switches: u64,
    /// Display lists run from the FIFO, and their segments.
    pub dl_calls: u64,
    pub dl_segments: u64,
}

impl Hq3Stats {
    const ZERO: Hq3Stats = Hq3Stats {
        words: 0, raster_writes: 0, raster_execs: 0, cp_tokens: [0; cmd::CP_LIMIT as usize], set_done: 0, swaps: 0,
        dma_reg_writes: 0, dma_writes: 0, dma_reads: 0, dma_bytes: 0, formatter_writes: 0, raster_if_writes: 0,
        pixel_cmds: 0, other_cmds: 0, context_switches: 0, dl_calls: 0, dl_segments: 0,
    };
}

/// Names of the host DMA engine registers this model uses.
fn dma_reg_name(n: usize) -> &'static str {
    match n {
        dma::PAGE_LIST => "page_list",
        dma::STRIDE => "stride",
        dma::ROW_OFFSET => "row_offset",
        dma::ROW_START => "row_start",
        dma::LINES => "lines",
        dma::LINE_BYTES => "line_bytes",
        dma::START => "start",
        dma::DL_JUMP_NO_WAIT => "dl_jump",
        dma::DL_JUMP_WAIT => "dl_jump_wait",
        dma::DL_PUSH => "dl_push",
        dma::DL_RETURN => "dl_return",
        dma::DL_TABLE_BASE => "dl_table_base",
        dma::MAX_DL_PTR_OFFSET => "max_dl_offset",
        n if (dma::TABLE_BASE..dma::TABLE_BASE + 8).contains(&n) => "table_base",
        _ => "",
    }
}

/// Data words a command keeps (the rest of a longer one are counted, not kept).
const CFIFO_KEEP: usize = 64;

/// The command FIFO's word-stream parser: a command word, then the data words
/// its byte count announces.
#[derive(Clone, Copy)]
struct Cfifo {
    cmd: u32,
    /// The control word's data conversion (bits 29:23) and byte count.
    conv: u32,
    bytes: u32,
    pixel: bool,
    need: u32,
    data: [u32; CFIFO_KEEP],
    len: usize,
}

impl Cfifo {
    const EMPTY: Cfifo = Cfifo { cmd: 0, conv: 0, bytes: 0, pixel: false, need: 0, data: [0; CFIFO_KEEP], len: 0 };

    fn data(&self) -> &[u32] {
        &self.data[..self.len]
    }

    /// Into a context's ERAM record, `PARSER_WORDS` words.
    fn store(&self, out: &mut [u32]) {
        out[..6].copy_from_slice(&[self.cmd, self.conv, self.bytes, self.pixel as u32, self.need, self.len as u32]);
        out[6..PARSER_WORDS].copy_from_slice(&self.data);
    }

    /// From a context's ERAM record (any words: clamped to sense).
    fn load(w: &[u32]) -> Self {
        let mut f = Cfifo {
            cmd: w[0], conv: w[1], bytes: w[2], pixel: w[3] != 0, need: w[4],
            len: (w[5] as usize).min(CFIFO_KEEP), data: [0; CFIFO_KEEP],
        };
        f.data.copy_from_slice(&w[6..PARSER_WORDS]);
        f
    }
}

/// The words of a context's record in ERAM, if the slot holds it.
fn slot_range(slot: u32) -> Option<std::ops::Range<usize>> {
    let s = slot as usize;
    (s + SLOT_RECORD_WORDS <= ERAM_WORDS).then(|| s..s + SLOT_RECORD_WORDS)
}

/// The frontend: everything behind the command FIFO.
pub struct Hq3Engine {
    /// Context-switch packet words still to swallow from the FIFO.
    context_words: u32,
    /// The incoming context image being loaded (`CONTEXT_SWITCH_WORDS`).
    context_image: [u32; host::CONTEXT_SWITCH_WORDS as usize],
    /// One parser per FIFO port (user, privileged): the two may interleave.
    /// The third runs display lists.
    cfifo: [Cfifo; 3],
    /// Display-list branches logged so far (bring-up).
    dl_logged: u32,
    /// Host DMA engine registers as 32-bit words; an eight-byte register
    /// takes two, high word first.
    pub dma_regs: [u32; 0x80],
    pub raster_if_regs: [u32; 0x10],
    pub formatter: u32,
    /// A board-to-host DMA started on the host side, waiting for the raster
    /// engine to be started (xfrcontrol = 9).
    dma_read_pending: Option<u32>,
    /// DMA transfers logged so far (bring-up).
    dma_logged: u32,
    /// The GE11s' OpenGL state (HLE) for the current context, the context
    /// it belongs to (image word 0, constant per context), and the other
    /// contexts' state, parked until they are switched back in (the real
    /// GE saves and restores its state with the context).
    pub gl: super::gl::Gl,
    /// The loaded context's ERAM slot (image word 0); none before the
    /// first context load.
    slot: Option<u32>,
    /// CP_GE_NOSAVE_NEXT_SWITCH seen: the next save skips the GE state.
    nosave_next: bool,
    /// An ERAM write in progress (CP_PASS_THROUGH_GE_WRAM): the words
    /// that came for it so far, by FIFO pixel data or host DMA, until
    /// CP_PASS_THROUGH_GE_FIFO_3 says where they go.
    eram_in: Option<Vec<u32>>,
    /// glBitmap / glDrawPixels rows arriving as FIFO pixel data, and those
    /// of switched-out contexts, by slot (a switch can land halfway
    /// through a client's pixel data; with two clients drawing text, their
    /// rows mixed into broken glyphs).
    bitmap_in: Vec<u32>,
    bitmap_parked: Vec<(u32, Vec<u32>)>,
    pub eram: Box<Eram>,
    /// The geometry engine's last answer to a state readback
    /// (`__MGR_RETURN_MODE`), read at the GE readback registers once set.
    pub ge_return: Option<[u32; 2]>,
    last_token: Option<u32>,
    pub stats: Hq3Stats,
}

impl Hq3Engine {
    pub fn new() -> Self {
        Hq3Engine {
            context_words: 0,
            context_image: [0; host::CONTEXT_SWITCH_WORDS as usize],
            cfifo: [Cfifo::EMPTY; 3],
            dl_logged: 0,
            dma_regs: [0; 0x80],
            raster_if_regs: [0; 0x10],
            formatter: 0,
            dma_read_pending: None,
            dma_logged: 0,
            ge_return: None,
            last_token: None,
            // SAFETY: plain data, valid zeroed (see `Gl`).
            gl: unsafe { std::mem::zeroed() },
            slot: None,
            nosave_next: false,
            eram_in: None,
            bitmap_in: Vec::new(),
            bitmap_parked: Vec::new(),
            eram: super::plain::boxed_zeroed(),
            stats: Hq3Stats::ZERO,
        }
    }

    /// The host requested a context switch: the save phase completes at once
    /// (the caller raises the flag). The outgoing context's GE state and
    /// user-port parser go to its ERAM slot, as the HQ microcode saves them
    /// on the board: a switch can preempt a process halfway through a
    /// command, whose remaining words come when it runs again (otherwise
    /// the next context's first words, such as the kernel's SCHEDULE_SWAP,
    /// were swallowed as the old command's data). The incoming context
    /// follows through the FIFO as raw words (see `push`).
    pub fn begin_context_switch(&mut self, sink: &mut dyn Hq3Sink) {
        self.gl.end_raster(sink);
        if let Some(slot) = self.slot {
            self.bitmap_parked.retain(|(s, _)| *s != slot);
            if !self.bitmap_in.is_empty() {
                self.bitmap_parked.push((slot, std::mem::take(&mut self.bitmap_in)));
            }
        }
        self.bitmap_in.clear();
        let parser = std::mem::replace(&mut self.cfifo[0], Cfifo::EMPTY);
        match self.slot.and_then(slot_range) {
            Some(r) => {
                let rec = &mut self.eram.words[0][r];
                if !std::mem::take(&mut self.nosave_next) {
                    // SAFETY: `Gl` is plain numbers (no padding-sensitive
                    // or invalid-bit-pattern fields) and fits the words.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            &self.gl as *const super::gl::Gl as *const u8,
                            rec.as_mut_ptr() as *mut u8,
                            std::mem::size_of::<super::gl::Gl>(),
                        );
                    }
                }
                parser.store(&mut rec[GL_WORDS..]);
                let ifr = &mut rec[GL_WORDS + PARSER_WORDS..];
                ifr[..0x80].copy_from_slice(&self.dma_regs);
                ifr[0x80..IF_WORDS].copy_from_slice(&self.raster_if_regs);
            }
            None => sink.note(format!("context switch from slot {:x?}: nowhere to save", self.slot)),
        }
        self.context_words = host::CONTEXT_SWITCH_WORDS;
        self.stats.context_switches += 1;
        if sink.tracing() {
            sink.trace(format!("context switch: saved slot {:x?}; loading {} words", self.slot, host::CONTEXT_SWITCH_WORDS));
        }
    }

    /// Load the context in ERAM slot `slot`: its GE state and user-port
    /// parser, or OpenGL's initial state on its first load (the kernel
    /// recycles slots, so a fresh context finds a dead one's state there).
    fn load_context(&mut self, slot: u32, first: bool, sink: &mut dyn Hq3Sink) {
        self.slot = Some(slot);
        self.bitmap_in = match self.bitmap_parked.iter().position(|(s, _)| *s == slot) {
            Some(i) if !first => self.bitmap_parked.remove(i).1,
            _ => Vec::new(),
        };
        // SAFETY (both): plain numbers, valid zeroed and for any bit pattern.
        self.gl = unsafe { std::mem::zeroed() };
        self.cfifo[0] = Cfifo::EMPTY;
        // Another context may have used the TE meanwhile.
        self.gl.te_dirty = 1;
        if first {
            return;
        }
        let Some(r) = slot_range(slot) else {
            sink.note(format!("context load from slot {slot:#x}: outside ERAM"));
            return;
        };
        let rec = &self.eram.words[0][r];
        unsafe {
            std::ptr::copy_nonoverlapping(
                rec.as_ptr() as *const u8,
                &mut self.gl as *mut super::gl::Gl as *mut u8,
                std::mem::size_of::<super::gl::Gl>(),
            );
        }
        self.cfifo[0] = Cfifo::load(&rec[GL_WORDS..]);
        let ifr = &rec[GL_WORDS + PARSER_WORDS..];
        self.dma_regs.copy_from_slice(&ifr[..0x80]);
        self.raster_if_regs.copy_from_slice(&ifr[0x80..IF_WORDS]);
        self.gl.te_dirty = 1;
    }

    /// Load a GL window's raster state (from a context image or
    /// CP_WINDOW): origin (x | y << 16, y bottom-up), window mode (masks
    /// enabled, bits 0-3, and kept inside, 4-7), screen masks 1-4 as (x, y)
    /// ranges `min << 16 | max`, the buffer page pointers, and the PP1
    /// window mode (the clip ID to match).
    fn apply_window(&mut self, origin: u32, mode: u32, masks: [[u32; 2]; 4], drb: u32, pp1winmode: u32, sink: &mut dyn Hq3Sink) {
        if sink.tracing() {
            sink.trace(format!(
                "GL window: origin {origin:#x} mode {mode:#x} masks {masks:x?} DRBpointers {drb:#x} pp1winmode {pp1winmode:#x}"
            ));
        }
        self.gl.window = super::gl::Window { valid: 1, origin, mode, masks, drb, pp1winmode };
    }

    /// The current context's GL state and the parked contexts, for the
    /// monitor.
    pub fn gl_summary(&self) -> String {
        format!("context in ERAM slot {:x?} (record {SLOT_RECORD_WORDS} words, GE state {GL_WORDS})\n{}", self.slot, self.gl.describe())
    }

    /// Words still owed to a context switch, and the parsers' state, for the
    /// monitor.
    pub fn summary(&self) -> String {
        let p = |f: &Cfifo| if f.need == 0 { "idle".to_string() } else { format!("cmd {:#x} needs {} more", f.cmd, f.need) };
        format!("context words owed {}, user parser {}, privileged parser {}, DMA read pending {:?}",
            self.context_words, p(&self.cfifo[0]), p(&self.cfifo[1]), self.dma_read_pending)
    }

    /// One 32-bit word from command FIFO `port` (0 user, 1 privileged).
    pub fn push(&mut self, port: usize, w: u32, sink: &mut dyn Hq3Sink) {
        self.stats.words += 1;
        // A context switch's incoming state follows the save phase as raw
        // words; the command processor would load it. Consume it here, and
        // report the load done after the last word.
        if self.context_words > 0 {
            let i = (host::CONTEXT_SWITCH_WORDS - self.context_words) as usize;
            self.context_image[i] = w;
            self.context_words -= 1;
            if self.context_words == 0 {
                let img = self.context_image;
                if sink.tracing() {
                    sink.trace(format!("context switch: loaded {:x?}", &img[..]));
                }
                // Word 0 is the context's ERAM slot; a first load (word 1
                // bit 31, cleared by the kernel afterwards) starts from
                // OpenGL's initial state. Every image also carries its
                // context's window (words 2-13: origin, window mode, PP1
                // window mode, masks 4..1, DRB pointers); the kernel
                // (MgrasValidateClip) rewrites it and sets bit 29 of word 1
                // when the window changed.
                self.load_context(img[0], img[1] & CTX_FIRST_LOAD != 0, sink);
                let masks = [[img[11], img[12]], [img[9], img[10]], [img[7], img[8]], [img[5], img[6]]];
                if sink.tracing() && img[1] & CTX_WINDOW_CHANGED != 0 {
                    sink.trace(format!("context {:#x}: window changed", img[0]));
                }
                self.apply_window(img[2], img[3], masks, img[13], img[4], sink);
                // Words 16-17: the banks drawn into (MgrasValidateBanks).
                self.gl.set_draw_bank(img[16], sink);
                sink.set_flags(host::FLAG_CONTEXT_LOADED);
            }
            return;
        }
        let f = &mut self.cfifo[port];
        if f.need == 0 {
            if w & 0x8000_0000 != 0 {
                // Pixel data: byte count in bits 19:0, sent as doublewords.
                f.pixel = true;
                f.cmd = w;
                f.need = (((w & 0xF_FFFF) + 7) / 8) * 2;
            } else {
                f.pixel = false;
                f.cmd = (w >> 8) & 0x1FFF;
                f.conv = (w >> 23) & 0x7F;
                f.bytes = w & 0xFF;
                f.need = ((w & 0xFF) + 3) / 4;
            }
            f.len = 0;
            if f.need == 0 {
                self.dispatch(port, sink);
            }
            return;
        }
        f.need -= 1;
        if f.pixel && !(f.need == 0 && (f.cmd & 0xF_FFFF).div_ceil(4) & 1 != 0) {
            if let Some(buf) = self.eram_in.as_mut() {
                buf.push(w);
            } else if self.gl.wants_bitmap_rows() {
                self.bitmap_in.push(w);
            }
        }
        let f = &mut self.cfifo[port];
        if f.len < CFIFO_KEEP {
            f.data[f.len] = w;
            f.len += 1;
        }
        if f.need == 0 {
            self.dispatch(port, sink);
        }
    }

    fn dispatch(&mut self, port: usize, sink: &mut dyn Hq3Sink) {
        let f = self.cfifo[port];
        let (cmd, data) = (f.cmd, f.data());
        if sink.tracing() {
            sink.trace(describe_cmd(port, f.pixel, cmd, data));
        }
        if f.pixel {
            self.stats.pixel_cmds += 1;
            if self.eram_in.is_none() && self.gl.wants_bitmap_rows() {
                // An image bigger than one pixel command (255 bytes) comes
                // in several: wait for all of it.
                if self.gl.wants_pixel_rows() {
                    let n = (cmd & 0xF_FFFF).div_ceil(4) as usize;
                    let start = self.bitmap_in.len().saturating_sub(n);
                    let tail = &mut self.bitmap_in[start..];
                    let mut bytes: Vec<u8> = tail.iter().flat_map(|w| w.to_be_bytes()).collect();
                    format_pixels(self.formatter, &mut bytes[..(cmd & 0xF_FFFF) as usize]);
                    for (w, b) in tail.iter_mut().zip(bytes.chunks_exact(4)) {
                        *w = u32::from_be_bytes(b.try_into().unwrap());
                    }
                }
                if self.bitmap_in.len() >= self.gl.pixel_words_needed() {
                    let rows = std::mem::take(&mut self.bitmap_in);
                    self.gl.bitmap_rows(&rows, sink);
                }
            } else if self.eram_in.is_none() {
                // libGLcore __glMgrFlushImageRegs: RSS_REG_SET (0x07F)
                // carries (raster register, value) pairs, RSS_REG_SHADOW
                // (0x07C, __glMgrRSSRegShadow) (register, value, GE shadow
                // address) triples, padded to whole doublewords.
                let group = match self.last_token {
                    Some(0x07F) => 2,
                    Some(0x07C) => 3,
                    _ => 0,
                };
                if group != 0 {
                    // Inside a GL pixel operation (SAVE_RSS .. RESTORE_RSS)
                    // the registers are written, within the GL raster
                    // bracket. Outside one they are the GE's shadow of a GL
                    // context's raster state (libGLcore sends 33 triples at
                    // context creation, CONFIG = 0x4 among them): kept for
                    // the pixel path, not written over the X server's state.
                    let live = self.gl.pixel_op != 0;
                    if live {
                        self.gl.begin_raster(sink);
                    }
                    for g in data.chunks_exact(group) {
                        // Texture registers belong to the context in any
                        // case: the GE keeps them and loads the TE.
                        if self.gl.te_write(g[0] & 0x3FF, g[1], sink) {
                            continue;
                        }
                        match g[0] & 0x3FF {
                            0x159 => self.gl.xfrmode = g[1],
                            0x046 => self.gl.pixel_block[0] = g[1],
                            0x047 => self.gl.pixel_block[1] = g[1],
                            _ => {}
                        }
                        if live {
                            sink.rss_write(g[0] & 0x3FF, g[1], false);
                        }
                    }
                } else if !matches!(self.last_token, Some(0x082)) {
                    sink.note(format!("pixel data command {cmd:#010x}"));
                }
            }
            return;
        }
        self.last_token = Some(cmd);
        if cmd >= cmd::RASTER_BASE {
            let r = cmd & 0x3FF;
            let exec = cmd & cmd::RASTER_EXECUTE != 0;
            self.stats.raster_writes += 1;
            self.stats.raster_execs += exec as u64;
            match data {
                [] => sink.rss_write(r, 0, exec),
                [d] => sink.rss_write(r, *d, exec),
                [hi, lo, ..] => {
                    sink.rss_write(r, *hi, false);
                    sink.rss_write(r + 1, *lo, exec);
                }
            }
            if r == RSS_XFRCONTROL && data.first() == Some(&XFRCONTROL_READ_START) {
                self.start_read_dma(sink);
            }
        } else if cmd == cmd::SET_DONE_FLAG {
            self.stats.set_done += 1;
            sink.set_flags(host::FLAG_DONE);
        } else if cmd >= cmd::DMA_BASE {
            let n = (cmd & 0x1FF) as usize;
            let v = data.first().copied().unwrap_or(0);
            if cmd >= cmd::FORMATTER {
                self.stats.formatter_writes += 1;
                self.formatter = v;
            } else if cmd >= cmd::RASTER_IF_BASE {
                self.stats.raster_if_writes += 1;
                self.raster_if_regs[n & 0xF] = v;
                // Register 5 starts the raster engine's side of a read (the
                // host DMA was armed first): libGLcore's glReadPixels
                // writes 0x4009, glGetTexImage 0x400D, low bits the
                // xfrcontrol read start (bit 2: the texture side).
                if n & 0xF == 5 && v & XFRCONTROL_READ_START == XFRCONTROL_READ_START {
                    self.start_read_dma(sink);
                }
            } else {
                self.stats.dma_reg_writes += 1;
                // Eight-byte registers arrive as a high word, then a low word,
                // and fill two register slots.
                let n = n & 0x7F;
                // The page list is four words (eight page indices).
                let words = if n == dma::PAGE_LIST { 4 } else { 2 };
                for (i, d) in data.iter().take(words).enumerate() {
                    self.dma_regs[(n + i) & 0x7F] = *d;
                }
                if data.is_empty() {
                    self.dma_regs[n] = 0;
                }
                match n {
                    dma::START => self.dma_start(v, sink),
                    // From the FIFO, a branch starts a display list; inside
                    // one, the links are the list engine's business.
                    dma::DL_JUMP_NO_WAIT | dma::DL_JUMP_WAIT if port != DL_PORT => {
                        let (offset, words) = (self.dma_regs[n], self.dma_regs[n + 1]);
                        self.run_display_list(offset, words, sink);
                    }
                    _ => {}
                }
            }
        } else if cmd == cmd::CP_SCHEDULE_SWAP {
            // No retrace wait: the swap is reported done at once. The GE's
            // front and back trade places for the drawing that follows.
            self.gl.swap_buffers(sink);
            self.stats.swaps += 1;
            self.stats.cp_tokens[cmd as usize] += 1;
            sink.set_flags(host::FLAG_CP0);
        } else if cmd == cmd::CP_WINDOW && data.len() >= 12 {
            self.stats.cp_tokens[cmd as usize] += 1;
            let d = data;
            let masks = [[d[9], d[10]], [d[7], d[8]], [d[5], d[6]], [d[3], d[4]]];
            self.apply_window(d[0], d[1], masks, d[11], d[2], sink);
        } else if matches!(cmd, cmd::CP_ERAM_WRITE | cmd::CP_ERAM_WRITE_AT | cmd::CP_ERAM_WRITE_END | cmd::CP_ERAM_READ) {
            self.stats.cp_tokens[cmd as usize] += 1;
            self.eram_pass_through(cmd, data, sink);
        } else if cmd == cmd::CP_GE_NOSAVE_NEXT_SWITCH {
            // The kernel, on switching to its own context: it changes no GE
            // state, so the next switch away need not save it.
            self.stats.cp_tokens[cmd as usize] += 1;
            self.nosave_next = true;
        } else if cmd == cmd::CP_SPIN_AND_RETURN {
            self.stats.cp_tokens[cmd as usize] += 1;
            self.gl.end_raster(sink);
            let addr = data.first().copied().unwrap_or(0);
            let v = self.gl.state_word(addr).unwrap_or(addr);
            self.ge_return = Some([0, v]);
            sink.set_flags(host::FLAG_GE_DATA);
        } else if cmd == cmd::CP_RETURN_MODE {
            self.stats.cp_tokens[cmd as usize] += 1;
            // Addresses the GE HLE knows answer from its state; others read
            // 0 and are reported once each, so the layout can be mapped.
            let addr = data.first().copied().unwrap_or(0);
            let v = self.gl.state_word(addr).unwrap_or_else(|| {
                sink.note(format!("GE state readback at {addr:#x} ({:x?}) answered 0", data.get(1..).unwrap_or(&[])));
                0
            });
            self.gl.end_raster(sink);
            self.ge_return = Some([0, v]);
            sink.set_flags(host::FLAG_GE_DATA);
        } else if cmd < cmd::CP_LIMIT && {
            let conv = f.conv;
            if conv & (CONV_ON | CONV_PAD) != 0 && f.bytes > 0 {
                let floats = convert(conv, f.bytes, data);
                if sink.tracing() {
                    sink.trace(format!("  converted ({conv:#x}): {:?}", floats.iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>()));
                }
                self.gl.token(cmd, &floats, sink)
            } else {
                self.gl.token(cmd, data, sink)
            }
        } {
            self.stats.cp_tokens[cmd as usize] += 1;
        } else if cmd < cmd::CP_LIMIT {
            self.stats.cp_tokens[cmd as usize] += 1;
            // The kernel's GE11 plumbing (diagnostic CP_* tokens, the HQ DMA
            // setup, its own tokens) changes nothing this model draws; the
            // rest is the OpenGL protocol, for the GE11s.
            if !is_ge_plumbing(cmd) {
                sink.note(format!("command-processor token {cmd:#x} {} ({} data words)", token_name(cmd).unwrap_or("?"), data.len()));
            }
        } else {
            self.stats.other_cmds += 1;
            sink.note(format!("command {cmd:#x}"));
        }
    }

    /// The host DMA engine's start word. Host to board runs now: the raster
    /// engine was armed first. Board to host waits for the raster engine.
    fn dma_start(&mut self, word: u32, sink: &mut dyn Hq3Sink) {
        if word & 1 == 0 {
            return;
        }
        if word & 8 != 0 {
            self.dma_read_pending = Some(word);
            return;
        }
        if self.eram_in.is_some() {
            // An ERAM write's data: host to board, into the pending words.
            let mut buf = self.eram_in.take().unwrap_or_default();
            self.dma(word, DmaPeer::EramIn(&mut buf), sink);
            self.eram_in = Some(buf);
            return;
        }
        if self.gl.wants_bitmap_rows() {
            // SEND_PIXELS whose rows come by host DMA (snoop's zoomed
            // image, distort's textures) rather than as FIFO pixel data;
            // possibly in several DMAs.
            let mut buf = std::mem::take(&mut self.bitmap_in);
            self.dma(word, DmaPeer::EramIn(&mut buf), sink);
            if buf.len() >= self.gl.pixel_words_needed() {
                self.gl.bitmap_rows(&buf, sink);
            } else {
                self.bitmap_in = buf;
            }
            return;
        }
        match sink.transfer() {
            Some((false, _)) => self.dma(word, DmaPeer::Raster, sink),
            _ => sink.note(format!("host DMA start {word:#x} with no write transfer armed")),
        }
    }

    /// The ERAM pass-through commands (see `cmd::CP_ERAM_*`), on GE0's
    /// ERAM (a board with two GE11s gets the same transfer once per GE).
    fn eram_pass_through(&mut self, cmd: u32, data: &[u32], sink: &mut dyn Hq3Sink) {
        let arg = |i: usize| data.get(i).copied().unwrap_or(0) as usize;
        match cmd {
            cmd::CP_ERAM_WRITE => self.eram_in = Some(Vec::with_capacity(arg(0))),
            cmd::CP_ERAM_WRITE_AT => {
                let (addr, n) = (arg(0), arg(1));
                let buf = self.eram_in.replace(Vec::new()).unwrap_or_default();
                let n = n.min(buf.len()).min(ERAM_WORDS.saturating_sub(addr));
                self.eram.words[0][addr..addr + n].copy_from_slice(&buf[..n]);
                if sink.tracing() {
                    sink.trace(format!("ERAM write {n} words at {addr:#x} ({} came)", buf.len()));
                }
            }
            cmd::CP_ERAM_WRITE_END => self.eram_in = None,
            _ => {
                let (addr, n) = (arg(0).min(ERAM_WORDS), arg(1));
                let n = n.min(ERAM_WORDS - addr);
                let Some(word) = self.dma_read_pending.take() else {
                    sink.note(format!("ERAM read at {addr:#x} with no host DMA armed"));
                    return;
                };
                if sink.tracing() {
                    sink.trace(format!("ERAM read {n} words at {addr:#x}"));
                }
                let words: Vec<u32> = self.eram.words[0][addr..addr + n].to_vec();
                self.dma(word, DmaPeer::EramOut(&words, 0), sink);
            }
        }
    }

    fn start_read_dma(&mut self, sink: &mut dyn Hq3Sink) {
        let Some(word) = self.dma_read_pending.take() else {
            sink.note("raster DMA read started with no host DMA pending".into());
            return;
        };
        match sink.transfer() {
            Some((true, _)) => self.dma(word, DmaPeer::Raster, sink),
            _ => sink.note(format!("host DMA read {word:#x} with no read transfer armed")),
        }
    }

    /// Run a display list starting with the segment of `words` words at
    /// byte `offset` in the display-list space, fetched from host memory
    /// through the display-list page table (one 32-bit frame number per
    /// 4 KB page, as for images). libGLcore compiles a list into segments
    /// whose first commands are the segment's link, the body following:
    /// the engine runs the body, then the link. Links (host DMA engine
    /// writes): DL_JUMP to the next segment, DL_PUSH (a return segment)
    /// followed by DL_JUMP to a called list, DL_RETURN to the last pushed
    /// segment, or back to the FIFO when none is left. ideas keeps its
    /// materials in lists; long lists chain segments of about 1 K words.
    fn run_display_list(&mut self, offset: u32, words: u32, sink: &mut dyn Hq3Sink) {
        let Some(mem) = sink.sys_mem() else { return };
        let table = self.dma_regs[dma::DL_TABLE_BASE + 1] & !3;
        let saved = std::mem::replace(&mut self.cfifo[DL_PORT], Cfifo::EMPTY);
        let mut stack: Vec<(u32, u32)> = Vec::new();
        let mut cur = Some((offset, words));
        let mut segments = 0u32;
        while let Some((offset, words)) = cur.take() {
            segments += 1;
            if segments > DL_MAX_SEGMENTS {
                sink.note(format!("display list runs past {DL_MAX_SEGMENTS} segments"));
                break;
            }
            self.stats.dl_segments += 1;
            let Some(seg) = fetch_dl_segment(&*mem, table, offset, words) else {
                sink.note(format!("display list segment at {offset:#x} unreadable"));
                break;
            };
            // The link: the DL engine commands at the head of the segment.
            let mut links: Vec<(usize, u32, u32)> = Vec::new();
            let mut at = 0;
            while at < seg.len() {
                let w = seg[at];
                let c = (w >> 8) & 0x1FFF;
                let n = ((w & 0xFF) as usize).div_ceil(4);
                let r = (c & 0x7F) as usize;
                if w & 0x8000_0000 != 0 || !(cmd::DMA_BASE..cmd::RASTER_IF_BASE).contains(&c)
                    || !matches!(r, dma::DL_JUMP_NO_WAIT | dma::DL_JUMP_WAIT | dma::DL_PUSH | dma::DL_RETURN)
                {
                    break;
                }
                let arg = |k: usize| seg.get(at + 1 + k).copied().unwrap_or(0);
                links.push((r, arg(0), arg(1)));
                at += 1 + n;
            }
            if sink.tracing() {
                sink.trace(format!("display list segment {offset:#x}: {words} words, links {links:x?}, stack depth {}", stack.len()));
            }
            for w in &seg[at.min(seg.len())..] {
                self.push(DL_PORT, *w, sink);
            }
            for (r, a, b) in links {
                match r {
                    dma::DL_PUSH => {
                        if stack.len() >= DL_MAX_DEPTH {
                            sink.note("display list calls nest too deep".into());
                            break;
                        }
                        stack.push((a, b));
                    }
                    dma::DL_RETURN => cur = stack.pop(),
                    _ => cur = Some((a, b)),
                }
            }
        }
        if self.dl_logged < 2 {
            self.dl_logged += 1;
            eprintln!("mgras: display list at {offset:#x} ({words} words) ran {segments} segments");
        }
        self.stats.dl_calls += 1;
        self.cfifo[DL_PORT] = saved;
    }

    /// Run a DMA between host memory and the armed raster transfer, a line at
    /// a time. Host addresses are logical within the pool and translate
    /// through its page table: one 32-bit frame number per 4 KB page.
    fn dma(&mut self, word: u32, mut peer: DmaPeer, sink: &mut dyn Hq3Sink) {
        let Some(mem) = sink.sys_mem() else { return };
        let read = word & 8 != 0;
        let pool = ((word >> 1) & 3) as usize;
        let table = self.dma_regs[dma::TABLE_BASE + 2 * pool + 1] & !3;
        // Between transfers the kernel parks a pool's table at a dummy
        // value (0x1, 0x5, 0x9: no page table). A DMA through it would take
        // frame numbers from low memory and scribble over the kernel.
        if table < 0x1000 {
            sink.note(format!("host DMA {word:#x} through pool {pool} with no page table ({table:#x}): dropped"));
            return;
        }
        // List mode (the texture manager's restores): the page list holds
        // up to eight 16-bit indices into the pool's table, bit 15 on the
        // last; one run from ROW_OFFSET in the first page, LINE_BYTES of it
        // there, whole pages between, LAST_BYTES of the last. Otherwise
        // (the list just [0 | last]) the pool's pages are consecutive and
        // the transfer is LINES lines of LINE_BYTES, STRIDE apart.
        let mut list: Vec<u32> = Vec::new();
        for w in &self.dma_regs[dma::PAGE_LIST..dma::PAGE_LIST + 4] {
            for h in [w >> 16, w & 0xFFFF] {
                if list.last().is_some_and(|l| l & 0x8000 != 0) {
                    break;
                }
                list.push(h);
            }
        }
        if list.last().is_none_or(|l| l & 0x8000 == 0) || list == [0x8000] {
            list.clear();
        }
        let (base, stride, lines, len) = if list.is_empty() {
            (
                self.dma_regs[dma::ROW_START].wrapping_add(self.dma_regs[dma::ROW_OFFSET]),
                self.dma_regs[dma::STRIDE],
                self.dma_regs[dma::LINES],
                self.dma_regs[dma::LINE_BYTES],
            )
        } else {
            let (first, last) = (self.dma_regs[dma::LINE_BYTES], self.dma_regs[dma::LAST_BYTES]);
            let n = list.len() as u32;
            let len = if n == 1 { first } else { first + 0x1000 * (n - 2) + last };
            (self.dma_regs[dma::ROW_OFFSET] & 0xFFF, 0, 1, len)
        };
        let list: Vec<u32> = list.iter().map(|h| h & 0x7FFF).collect();
        if read { self.stats.dma_reads += 1 } else { self.stats.dma_writes += 1 }
        self.stats.dma_bytes += lines as u64 * len as u64;
        let desc = || format!(
            "DMA {} pool {pool} table {table:#x} base {base:#x} stride {stride} lines {lines} bytes {len} pglist {:#x} with {}",
            if read { "read" } else { "write" },
            self.dma_regs[dma::PAGE_LIST],
            peer.name()
        );
        if sink.tracing() {
            sink.trace(desc());
        }
        if self.dma_logged < 16 {
            self.dma_logged += 1;
            eprintln!("mgras: {}", desc());
        }
        let frame_of = |page: u32| -> Option<u32> {
            let page = if list.is_empty() { page } else { *list.get(page as usize)? };
            let r = mem.read32(table.wrapping_add(4 * page));
            r.is_ok().then_some(r.data << 12)
        };
        let mut cached: Option<(u32, u32)> = None;
        let mut phys = |l: u32| -> Option<u32> {
            let page = l >> 12;
            let f = match cached {
                Some((p, f)) if p == page => f,
                _ => {
                    let f = frame_of(page)?;
                    cached = Some((page, f));
                    f
                }
            };
            Some(f | (l & 0xFFF))
        };
        // A read from the raster engine is a byte stream across its lines:
        // the host may take it a row per DMA line, or all rows in one
        // (glGetTexImage of a small texture reads 8 rows as one line).
        let raster_lines = if read && matches!(peer, DmaPeer::Raster) { sink.transfer().map_or(0, |(_, (n, _))| n) } else { 0 };
        let (mut stream, mut next_line) = (std::collections::VecDeque::<u8>::new(), 0u32);
        for i in 0..lines {
            let a = base.wrapping_add(i.wrapping_mul(stride));
            if read {
                let mut bytes = match &mut peer {
                    DmaPeer::EramOut(words, pos) => {
                        let mut v = Vec::with_capacity(len as usize);
                        for _ in 0..len {
                            let w = words.get(*pos / 4).copied().unwrap_or(0);
                            v.push((w >> (24 - 8 * (*pos % 4))) as u8);
                            *pos += 1;
                        }
                        v
                    }
                    _ => {
                        while stream.len() < len as usize && next_line < raster_lines.max(i + 1) {
                            stream.extend(sink.dma_read_line(next_line));
                            next_line += 1;
                        }
                        let n = (len as usize).min(stream.len());
                        stream.drain(..n).collect()
                    }
                };
                // The RSS exposes X's packed ABGR byte pixels. A GE
                // framebuffer read returns RGBA components before the HQ
                // formatter (as RGB/16-bit reads already do in to_host).
                // Without this, glCopyPixels rotates alpha into red each
                // time it draws the saved image back.
                if matches!(peer, DmaPeer::Raster) && self.gl.pixel_op == 2 && self.gl.xfrmode & 0xFF == 0x80 {
                    for px in bytes.chunks_exact_mut(4) {
                        px.reverse();
                    }
                }
                format_pixels(self.formatter, &mut bytes);
                for (k, b) in bytes.iter().take(len as usize).enumerate() {
                    let Some(pa) = phys(a + k as u32) else { return };
                    mem.write8(pa, *b);
                }
            } else {
                let mut bytes = Vec::with_capacity(len as usize);
                for k in 0..len {
                    let Some(pa) = phys(a + k) else { return };
                    let r = mem.read8(pa);
                    bytes.push(if r.is_ok() { r.data } else { 0 });
                }
                if i == 0 && sink.tracing() {
                    sink.trace(format!("  DMA line 0 from {a:#x}: {:02x?}", &bytes[..bytes.len().min(32)]));
                }
                format_pixels(self.formatter, &mut bytes);
                match &mut peer {
                    DmaPeer::EramIn(buf) => buf.extend(bytes.chunks(4).map(|c| {
                        c.iter().enumerate().fold(0u32, |w, (j, b)| w | (*b as u32) << (24 - 8 * j))
                    })),
                    _ => sink.dma_write_line(i, &bytes),
                }
            }
        }
    }
}

/// The formatter's work on DMA'd pixel bytes (mode register 0xC00, see
/// HQ3.h): within each pixel of N components of 1, 2 or 4 bytes, swap the
/// bytes of each component (bit 6) and reverse the components (bit 7,
/// "swizzle": libGLcore loads GL_ABGR_EXT textures with 0xE8D, byte
/// components, and 0xE8E, shorts). The same reordering undoes itself, so
/// reads use it too. Nibble components and the endian modes are not done.
pub(super) fn format_pixels(mode: u32, bytes: &mut [u8]) {
    const SWAP_BYTES: u32 = 1 << 6;
    const SWIZZLE: u32 = 1 << 7;
    if mode & (SWAP_BYTES | SWIZZLE) == 0 {
        return;
    }
    let size = match mode & 3 {
        1 => 1,
        2 => 2,
        3 => 4,
        _ => return,
    };
    let n = ((mode >> 2) & 3) as usize + 1;
    for px in bytes.chunks_exact_mut(size * n) {
        if mode & SWAP_BYTES != 0 {
            for c in px.chunks_exact_mut(size) {
                c.reverse();
            }
        }
        if mode & SWIZZLE != 0 {
            px.reverse();
            if size > 1 {
                // Reversing the pixel reversed each component's bytes too.
                for c in px.chunks_exact_mut(size) {
                    c.reverse();
                }
            }
        }
    }
}

/// What a host DMA moves data to or from: the raster engine's armed
/// transfer, or GE11 ERAM (words big-endian in host memory; the read keeps
/// its byte position across lines).
enum DmaPeer<'a> {
    Raster,
    EramIn(&'a mut Vec<u32>),
    EramOut(&'a [u32], usize),
}

impl DmaPeer<'_> {
    fn name(&self) -> &'static str {
        match self {
            DmaPeer::Raster => "raster",
            DmaPeer::EramIn(_) => "ERAM write",
            DmaPeer::EramOut(..) => "ERAM read",
        }
    }
}

/// Tokens the kernel sends to set up and save the GE11s (and the HQ DMA
/// setup token), which nothing in this model needs.
fn is_ge_plumbing(t: u32) -> bool {
    matches!(t, 0x006 | 0x07C | 0x07F | 0x082 | 0x08C | 0x0EC | 0x0ED | 0x0F2 | 0x0F3..=0x0FF)
}

/// One command FIFO command, decoded for the trace.
fn describe_cmd(port: usize, pixel: bool, cmd: u32, data: &[u32]) -> String {
    use super::rss::REG_NAMES;
    let p = ["u", "p", "dl"][port.min(2)];
    let words = |d: &[u32]| d.iter().take(16).map(|v| format!("{v:#x}")).collect::<Vec<_>>().join(" ");
    if pixel {
        return format!("{p} pixel data: {} bytes, addr {}, align {}{}{} ({} words) {}",
            cmd & 0xF_FFFF, (cmd >> 20) & 3, (cmd >> 22) & 7,
            if cmd & 1 << 25 != 0 { ", packed" } else { "" }, if cmd & 1 << 26 != 0 { ", last" } else { "" }, data.len(), words(data));
    }
    let name = |r: u32| match REG_NAMES.get(r as usize).copied().unwrap_or("") {
        "" => format!("rss[{r:#x}]"),
        n => n.to_string(),
    };
    if cmd >= cmd::RASTER_BASE {
        let r = cmd & 0x3FF;
        let exec = if cmd & cmd::RASTER_EXECUTE != 0 { " +exec" } else { "" };
        return match data {
            [] => format!("{p} {} (no data){exec}", name(r)),
            [v] => format!("{p} {} = {v:#x}{exec}", name(r)),
            [hi, lo, ..] => format!("{p} {} = {hi:#x}, {} = {lo:#x}{exec}", name(r), name(r + 1)),
        };
    }
    let v = data.first().copied().unwrap_or(0);
    match cmd {
        cmd::SET_DONE_FLAG => format!("{p} SET_DONE_FLAG"),
        c if c >= cmd::FORMATTER => format!("{p} formatter = {v:#x}"),
        c if c >= cmd::RASTER_IF_BASE => format!("{p} raster_if[{:#x}] = {v:#x}", c & 0xF),
        c if c >= cmd::DMA_BASE => {
            let n = (c & 0x7F) as usize;
            format!("{p} dma[{n:#x}] {} = {}", dma_reg_name(n), words(data))
        }
        cmd::CP_SCHEDULE_SWAP => format!("{p} CP SCHEDULE_SWAP {}", words(data)),
        c if c < cmd::CP_LIMIT => format!("{p} CP {} [{c:#x}] ({} words) {}", token_name(c).unwrap_or("token"), data.len(), words(data)),
        c => format!("{p} command {c:#x} ({} words) {}", data.len(), words(data)),
    }
}

/// Command-processor token names (ignore/gr4/HQ3.h): the OpenGL protocol as
/// libGLcore sends it, SGI's diagnostic CP_* tokens, and kernel tokens.
pub fn token_name(t: u32) -> Option<&'static str> {
    Some(match t {
        0x000 => "VERTEX4F",
        0x002 => "COLOR4F",
        0x005 => "SIM_OPEN_GRAPHICS",
        0x006 => "MAKE_CURRENT_READ",
        0x008 => "TEX_COORD4F",
        0x009 => "INIT_RGB",
        0x00D => "ENABLE_LIGHTING",
        0x00E => "DISABLE_LIGHTING",
        0x00F => "EDGE_FLAG_ENABLE",
        0x010 => "EDGE_FLAG",
        0x011 => "MATERIAL",
        0x012 => "LIGHT_AMBIENT",
        0x015 => "CLEAR",
        0x016 => "BEGIN_POINTS",
        0x017 => "BEGIN_LINES",
        0x018 => "BEGIN_LSTRIP",
        0x019 => "BEGIN_LLOOP",
        0x01A => "BEGIN_TRIANGLES",
        0x01B => "BEGIN_TSTRIP",
        0x01C => "BEGIN_TFAN",
        0x01D => "BEGIN_QUADS",
        0x01E => "BEGIN_QSTRIP",
        0x01F => "BEGIN_POLYGON",
        0x020 => "END_POINTS",
        0x021 => "END_LINES",
        0x022 => "END_LSTRIP",
        0x023 => "END_LLOOP",
        0x024 => "END_TRIANGLES",
        0x025 => "END_TSTRIP",
        0x026 => "END_TFAN",
        0x027 => "END_QUADS",
        0x028 => "END_QSTRIP",
        0x029 => "END_POLYGON",
        0x02A => "MATRIX_MODE",
        0x02B => "LOAD_MATRIXD",
        0x02C => "LOAD_IDENTITY",
        0x02D => "MULT_MATRIXD",
        0x02E => "POP_MATRIX",
        0x02F => "PUSH_MATRIX",
        0x030 => "ROTATED",
        0x031 => "SCALED",
        0x032 => "TRANSLATED",
        0x033 => "VIEWPORT",
        0x034 => "FRUSTUM",
        0x035 => "ORTHO",
        0x036 => "FLUSH",
        0x037 => "SCHEDULE_SWAP",
        0x038 => "RASTER_POS4F",
        0x039 => "SCISSOR",
        0x03A => "STENCIL_MASK",
        0x03B => "COLOR_MASK",
        0x03C => "DEPTH_MASK",
        0x03D => "INDEX_MASK",
        0x03E => "ALPHA_FUNC",
        0x03F => "BLEND_FUNC",
        0x040 => "LOGIC_OP",
        0x041 => "STENCIL_FUNC",
        0x042 => "STENCIL_OP",
        0x043 => "DEPTH_FUNC",
        0x044 => "READ_BUFFER",
        0x045 => "LINE_WIDTH",
        0x046 => "LINE_STIPPLE",
        0x047 => "SHADE_MODEL",
        0x048 => "DEPTH_RANGE",
        0x049 => "DRAW_BUFFER",
        0x098 => "VALIDATE_BANKS",
        0x04A => "CLEAR_DEPTH_BUFFER",
        0x04C => "CLEAR_STENCIL_BUFFER",
        0x04D => "POINT_SIZE",
        0x04E => "LOAD_POLYGON_STIPPLE_RAM",
        0x055 => "LIGHT_QUADRATIC_ATTENUATION",
        0x056 => "FRONT_FACE",
        0x057 => "LIGHT_MODEL_AMBIENT",
        0x059 => "LIGHT_MODEL_TWO_SIDE",
        0x05A => "TEX_GENIV",
        0x05B => "TEX_GEN_PLANE_HEADER",
        0x05C => "TEX_GEN_PLANE_DATA",
        0x05D => "PASS_THROUGH",
        0x05F => "DISABLE",
        0x060 => "BLEND_EQUATION_EXT",
        0x061 => "GET_FEEDBACK_DATA",
        0x063 => "GET_ERROR",
        0x066 => "DEPTH_TEST_ENABLE",
        0x067 => "DITHER_ENABLE",
        0x06E => "SOFT_DISABLE",
        0x072 => "CULL_FACE_ENABLE",
        0x073 => "CULL_FACE_DISABLE",
        0x071 => "ENABLE_LIGHT_SOURCE",
        0x074 => "CULL_FACE",
        0x07C => "RSS_REG_SHADOW",
        0x07F => "RSS_REG_SET",
        0x082 => "INIT_TABLES",
        0x083 => "RESET_HISTOGRAM_MEMORY",
        0x085 => "INIT_CONVOLVE2_DSTATE",
        0x086 => "FAST_COPY_PIXELS",
        0x088 => "NEW_INIT_FAST_PATH",
        0x089 => "INIT_COLOR_TABLE",
        0x08B => "__MGR_RETURN_HIST_PIO",
        0x08C => "KERN_8C",
        0x08E => "COPY_TILE",
        0x091 => "GENERAL_BITMAP",
        0x094 => "BITMAP",
        0x096 => "COPY_COLOR_TABLE_SGI",
        0x09A => "INIT_FORMAT_VALUES",
        0x09B => "TL_NO_TABLE",
        0x09D => "_WRITE_DMAGESETUP",
        0x0A0 => "DISABLE_LIGHT_SOURCE",
        0x0A1 => "SPIN_AND_RETURN",
        0x0A2 => "__MGR_RETURN_MODE",
        0x0A5 => "CLIP_PLANE",
        0x0A8 => "POLYGON_MODE",
        0x0A9 => "EVAL_MESH1_DGE11",
        0x0AA => "EVAL_MESH2_DGE11",
        0x0AB => "EVAL_COORD1_DGE11",
        0x0AC => "EVAL_COORD2_DGE11",
        0x0AE => "MAP_GRID1D",
        0x0AF => "MAP_GRID2D",
        0x0B0 => "CLEAN_UP1_D",
        0x0B1 => "CLEAN_UP2_D",
        0x0B2 => "SEND_MAP_DATA1_D",
        0x0B3 => "SEND_MAP_DATA2_D",
        0x0B7 => "FOG_DENSITY",
        0x0B9 => "FOG_MODE",
        0x0BA => "CLEAR_COLOR",
        0x0BB => "CLEAR_ACCUM",
        0x0BC => "CLEAR_DEPTH",
        0x0BD => "CLEAR_INDEX",
        0x0BE => "CLEAR_STENCIL",
        0x0C1 => "COLOR_MATERIAL_BOTH",
        0x0C4 => "INIT_NAMES",
        0x0C5 => "LOAD_NAME",
        0x0C6 => "POP_NAME",
        0x0C7 => "PUSH_NAME",
        0x0C9 => "RENDER_MODE",
        0x0CC => "END_GENERIC",
        0x0D2 => "SAVE_RSS",
        0x0D3 => "RESTORE_RSS",
        0x0D4 => "SEND_HISTOGRAM_FILLS",
        0x0D5 => "SEND_MINMAX_FILLS",
        0x0D6 => "UPDATE_RASTER_POS",
        0x0D7 => "LOAD_RASTER_POS_INFO",
        0x0D9 => "BLEND_COLOR_EXT",
        0x0DB => "GET_PIXELS_BITMAP_PIO",
        0x0DC => "SWAPTMESH",
        0x0E4 => "KERN_WINDOW",
        0x0DE => "LMCOLOR",
        0x0E0 => "DRAW_DUMB_POINT",
        0x0E7 => "POLYGON_OFFSET",
        0x0E9 => "SAVE_TEXMODE",
        0x0EA => "RESTORE_TEXMODE",
        0x0EC => "RSS_REG_SHADOW_IDX",
        0x0ED => "SET_DATA",
        0x0EE => "GET_MATERIALFV",
        0x0EF => "GET_LIGHTFV",
        0x0F2 => "GET_HISTOGRAM_EXT",
        0x0F0 => "READ_TEXTURE",
        0x0F3 => "CP_DUMP_FOREVER",
        0x0F4 => "CP_GE_NOSAVE_NEXT_SWITCH",
        0x0F5 => "CP_GE_FTOF",
        0x0F6 => "CP_GE_DMA_HEADER_SIZE",
        0x0F7 => "CP_DELAY",
        0x0F8 => "CP_SET_CP_FLAGS",
        0x0F9 => "CP_PASS_THROUGH_GE_WRAM_5",
        0x0FA => "CP_PASS_THROUGH_GE_WRAM_4",
        0x0FB => "CP_PASS_THROUGH_GE_FIFO_3",
        0x0FC => "CP_PASS_THROUGH_GE_WRAM_DELAYED",
        0x0FD => "CP_PASS_THROUGH_GE_WRAM",
        0x0FE => "CP_PASS_THROUGH_GE_FIFO",
        0x0FF => "CP_GE_SAVE_IDLE_HQ_STATE",
        0x100 => "COPY_PIXELS",
        0x007 => "NORMAL3F",
        0x013 => "LIGHT_DIFFUSE",
        0x014 => "LIGHT_SPECULAR",
        0x04F => "LIGHT_POSITION",
        0x050 => "LIGHT_SPOT_DIRECTION",
        0x051 => "LIGHT_SPOT_EXPONENT",
        0x052 => "LIGHT_SPOT_CUTOFF",
        0x053 => "LIGHT_CONSTANT_ATTENUATION",
        0x054 => "LIGHT_LINEAR_ATTENUATION",
        0x058 => "LIGHT_MODEL_LOCAL_VIEWER",
        0x064 => "ALPHA_TEST_ENABLE",
        0x065 => "BLEND_ENABLE",
        0x068 => "FOG_ENABLE",
        0x06A => "LINE_STIPPLE_ENABLE",
        0x06B => "LOGIC_OP_ENABLE",
        0x06D => "POLYGON_STIPPLE_ENABLE",
        0x06F => "STENCIL_TEST_ENABLE",
        0x0A3 => "NORMALIZE_ENABLE",
        0x0A6 => "CLIP_PLANE_ENABLE",
        0x0A7 => "CLIP_PLANE_DISABLE",
        0x0B4 => "FOG_COLOR",
        0x0B5 => "FOG_START",
        0x0B6 => "FOG_END",
        0x0B8 => "FOG_INDEX",
        0x0BF => "COLOR_MATERIAL_FRONT",
        0x0C0 => "COLOR_MATERIAL_BACK",
        0x0C2 => "COLOR_MATERIAL_ENABLE",
        0x0C3 => "COLOR_MATERIAL_DISABLE",
        _ => return None,
    })
}
