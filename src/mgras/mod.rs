//! IMPACT (MGRAS) graphics for the Indigo2.
//!
//! One board occupies the GIO graphics slot (`0x1F000000`) and decodes a 1 MB
//! window there. The host talks to one chip on it, the host interface,
//! which offers:
//!
//! | offset            | what                                              |
//! |-------------------|---------------------------------------------------|
//! | `0x00000`         | GIO ID                                            |
//! | `0x40000-0x45FFF` | command-processor microcode RAM (24-bit words)    |
//! | `0x50000-0x5FFFF` | privileged registers, privileged command FIFO     |
//! | `0x60000-0x67FFF` | display control bus devices (see `dcb`)           |
//! | `0x68000-0x6FFFF` | per-device bus protocol registers                 |
//! | `0x70000-0x7BFFF` | user registers: status, flags, command FIFO       |
//! | `0x7C000-0x7FFFF` | raster registers, direct access (see `raster`);   |
//! |                   | the alias at `+0x1000` also executes the IR       |
//!
//! Drawing arrives through the command FIFO as (command, data) pairs. Commands
//! at `0x1000` and up write raster registers directly, with bit `0x400`
//! meaning "execute"; lower numbers go to the command processor's microcode,
//! which this model does not run (the PROM, the kernel's console and the X
//! server's 2D paths never need it). The FIFO is drained as it is written, so
//! it always reads as empty.
//!
//! The model scans its framebuffer out through the colormap and DAC gamma into
//! the same window and status bar Newport uses (`GfxDisplay`).
//!
//! `IRIS_MGRAS_TRACE=<file>` logs every access to the board.

mod dcb;
mod raster;

use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::io::Write as IoWrite;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::config::{ImpactSection, ImpactSlot};
use crate::dev::ng1::rex3::Renderer;
use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, Device, Saveable};

/// The graphics slot the board decodes.
pub const MGRAS_SLOT_GFX_BASE: u32 = 0x1F00_0000;
pub const MGRAS_SLOT_GFX_SIZE: u32 = 0x0040_0000;
/// The board's register window within the slot.
const MAP_SIZE: u32 = 0x10_0000;

/// GIO ID: product 0x10, 32-bit ID, revision 1, GIO64, no ROM, manufacturer 1.
pub const GIO_ID: u32 = 0x0005_0190;

/// Host interface register offsets.
mod host {
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

/// A geometry engine as its diagnostic port sees it. The engine itself does
/// not run; it only has to hold downloaded microcode, read it back, and
/// answer "started".
#[derive(Default)]
struct GeDiag {
    /// Current diagnostic address: a microcode line (from `UCODE_BASE`) or an
    /// internal register.
    addr: u32,
    /// Which 32-bit word of the current 72-bit microcode line comes next.
    word: usize,
    ucode: HashMap<u32, [u32; 3]>,
}

impl GeDiag {
    const UCODE_BASE: u32 = 0x20_0000;
    /// Internal register: execution control; bit 0 starts the engine.
    const EXEC_CONTROL: u32 = 0x4_0000;

    /// Words a readback starting at `addr` delivers, in order: for each pair
    /// of lines, word 0 and the top byte of the first, then word 1 of the
    /// second. A readback starting on an odd line is preceded by two words
    /// that carry nothing. (The driver's verifier reads exactly this; the
    /// order is inferred from it.)
    fn readback(&self, lines: u32) -> Vec<u32> {
        let w = |l: u32, i: usize| self.ucode.get(&l).map(|x| x[i]).unwrap_or(0);
        let mut out = Vec::new();
        let mut l = self.addr;
        if l.wrapping_sub(Self::UCODE_BASE) & 1 == 1 {
            out.extend([0, 0]);
        }
        for _ in 0..lines / 2 {
            out.extend([w(l, 0), w(l, 2) & 0xFF, w(l + 1, 1)]);
            l += 2;
        }
        out
    }
}

/// Command FIFO command numbers.
mod cmd {
    /// Command-processor token: schedule a buffer swap for the next retrace.
    pub const CP_SCHEDULE_SWAP: u32 = 0x37;
    pub const SET_DONE_FLAG: u32 = 0xE04;
    pub const RASTER_BASE: u32 = 0x1000;
    pub const RASTER_EXECUTE: u32 = 0x400;
    pub const DMA_BASE: u32 = 0x800;
    pub const RASTER_IF_BASE: u32 = 0xA00;
    pub const FORMATTER: u32 = 0xC00;
    /// Below this, commands are command-processor microcode tokens.
    pub const CP_LIMIT: u32 = 0x200;
}

/// The command FIFO's word-stream parser: a command word, then the data words
/// its byte count announces.
#[derive(Default)]
struct Cfifo {
    cmd: u32,
    pixel: bool,
    need: u32,
    data: Vec<u32>,
}

/// Host DMA engine registers.
mod dma {
    pub const PAGE_LIST: usize = 0x00;
    pub const STRIDE: usize = 0x04;
    pub const ROW_OFFSET: usize = 0x05;
    pub const ROW_START: usize = 0x06;
    pub const LINES: usize = 0x07;
    pub const LINE_BYTES: usize = 0x08;
    /// The start word: bit 0 run, bits 2:1 pool, bit 3 board to host.
    pub const START: usize = 0x0B;
    /// Page-table base of pool `p`: eight bytes at `TABLE_BASE + 2p`, the
    /// address in the low word.
    pub const TABLE_BASE: usize = 0x20;
}

/// Board state behind one lock.
struct Board {
    regs: HashMap<u32, u32>,
    ucode: Vec<u32>,
    flags: u32,
    flag_enable: u32,
    interrupt_enable: u32,
    /// Context-switch packet words still to swallow from the FIFO.
    context_words: u32,
    ge_readback: [u32; 2],
    ge: [GeDiag; 2],
    /// Pending diagnostic readback words, oldest first.
    ge_out: std::collections::VecDeque<u32>,
    /// One parser per FIFO port (user, privileged): the two may interleave.
    cfifo: [Cfifo; 2],
    /// Host DMA engine registers as 32-bit words; an eight-byte register
    /// takes two, high word first.
    dma_regs: [u32; 0x80],
    raster_if_regs: [u32; 0x10],
    formatter: u32,
    /// A board-to-host DMA started on the host side, waiting for the raster
    /// engine to be started (xfrcontrol = 9).
    dma_read_pending: Option<u32>,
    /// System memory, for DMA.
    mem: Option<Arc<dyn BusDevice>>,
    /// DMA transfers logged so far (bring-up).
    dma_logged: u32,
    dcb: dcb::Dcb,
    raster: raster::Raster,
    /// Commands and registers not modelled yet, reported once each.
    unhandled: HashSet<String>,
}

impl Board {
    fn new(kind: ImpactSlot) -> Self {
        // Board version bytes: [RA/RB boards + TRAMs, product + GE count].
        let bdvers = match kind {
            ImpactSlot::Solid => [0x70, 0x21],
            ImpactSlot::High => [0x70, 0x01],
            ImpactSlot::Max => [0x00, 0x02],
            ImpactSlot::None => [0, 0],
        };
        Board {
            regs: HashMap::new(),
            ucode: vec![0; ((host::UCODE_END - host::UCODE) / 4) as usize],
            flags: 0,
            flag_enable: 0,
            interrupt_enable: 0,
            context_words: 0,
            ge_readback: [0; 2],
            ge: [GeDiag::default(), GeDiag::default()],
            ge_out: std::collections::VecDeque::new(),
            cfifo: [Cfifo::default(), Cfifo::default()],
            dma_regs: [0; 0x80],
            raster_if_regs: [0; 0x10],
            formatter: 0,
            dma_read_pending: None,
            mem: None,
            dma_logged: 0,
            dcb: dcb::Dcb::new(bdvers, [0xFB, 0xFB]),
            raster: raster::Raster::default(),
            unhandled: HashSet::new(),
        }
    }

    fn note(&mut self, what: String) {
        if self.unhandled.len() < 256 && self.unhandled.insert(what.clone()) {
            eprintln!("mgras: not modelled yet: {what}");
        }
    }

    /// Push one 32-bit word into the command FIFO. Returns true when the
    /// framebuffer changed.
    fn cfifo_word(&mut self, port: usize, w: u32) -> bool {
        // A context switch's incoming state follows the save phase as raw
        // words; the command processor would load it. Consume it here, and
        // report the load done after the last word.
        if self.context_words > 0 {
            self.context_words -= 1;
            if self.context_words == 0 {
                self.flags |= host::FLAG_CONTEXT_LOADED;
            }
            return false;
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
                f.need = ((w & 0xFF) + 3) / 4;
            }
            f.data.clear();
            if f.need == 0 {
                return self.dispatch(port);
            }
            return false;
        }
        f.need -= 1;
        if f.data.len() < 64 {
            f.data.push(w);
        }
        if f.need == 0 { self.dispatch(port) } else { false }
    }

    fn dispatch(&mut self, port: usize) -> bool {
        let cmd = self.cfifo[port].cmd;
        let data = std::mem::take(&mut self.cfifo[port].data);
        if self.cfifo[port].pixel {
            self.note(format!("pixel data command {cmd:#010x}"));
            return false;
        }
        let changed = if cmd >= cmd::RASTER_BASE {
            let r = cmd & 0x3FF;
            let exec = cmd & cmd::RASTER_EXECUTE != 0;
            let changed = match data.as_slice() {
                [] => self.raster.write(r, 0, exec),
                [d] => self.raster.write(r, *d, exec),
                [hi, lo, ..] => {
                    self.raster.write(r, *hi, false);
                    self.raster.write(r + 1, *lo, exec)
                }
            };
            if r == raster::reg::XFRCONTROL && data.first() == Some(&9) {
                changed | self.start_read_dma()
            } else {
                changed
            }
        } else if cmd == cmd::SET_DONE_FLAG {
            self.flags |= host::FLAG_DONE;
            false
        } else if cmd >= cmd::DMA_BASE {
            let n = (cmd & 0x1FF) as usize;
            let v = data.first().copied().unwrap_or(0);
            if cmd >= cmd::FORMATTER {
                self.formatter = v;
                false
            } else if cmd >= cmd::RASTER_IF_BASE {
                self.raster_if_regs[n & 0xF] = v;
                false
            } else {
                // Eight-byte registers arrive as a high word, then a low word,
                // and fill two register slots.
                let n = n & 0x7F;
                for (i, d) in data.iter().take(2).enumerate() {
                    self.dma_regs[(n + i) & 0x7F] = *d;
                }
                if data.is_empty() {
                    self.dma_regs[n] = 0;
                }
                if n == dma::START { self.dma_start(v) } else { false }
            }
        } else if cmd == cmd::CP_SCHEDULE_SWAP {
            // No retrace wait: the swap is reported done at once.
            self.flags |= host::FLAG_CP0;
            false
        } else if cmd < cmd::CP_LIMIT {
            self.note(format!("command-processor token {cmd:#x} ({} data words)", data.len()));
            false
        } else {
            self.note(format!("command {cmd:#x}"));
            false
        };
        self.cfifo[port].data = data;
        changed
    }

    /// The host DMA engine's start word. Host to board runs now: the raster
    /// engine was armed first. Board to host waits for the raster engine.
    fn dma_start(&mut self, word: u32) -> bool {
        if word & 1 == 0 {
            return false;
        }
        if word & 8 != 0 {
            self.dma_read_pending = Some(word);
            return false;
        }
        match self.raster.transfer_armed() {
            Some(false) => self.dma(word, false),
            _ => {
                self.note(format!("host DMA start {word:#x} with no write transfer armed"));
                false
            }
        }
    }

    fn start_read_dma(&mut self) -> bool {
        let Some(word) = self.dma_read_pending.take() else {
            self.note("raster DMA read started with no host DMA pending".into());
            return false;
        };
        if self.raster.transfer_armed() != Some(true) {
            self.note(format!("host DMA read {word:#x} with no read transfer armed"));
            return false;
        }
        self.dma(word, true)
    }

    /// Run a DMA between host memory and the armed raster transfer, a line at
    /// a time. Host addresses are logical within the pool and translate
    /// through its page table: one 32-bit frame number per 4 KB page.
    fn dma(&mut self, word: u32, read: bool) -> bool {
        let Some(mem) = self.mem.clone() else { return false };
        let pool = ((word >> 1) & 3) as usize;
        let table = self.dma_regs[dma::TABLE_BASE + 2 * pool + 1] & !3;
        let base = self.dma_regs[dma::ROW_START].wrapping_add(self.dma_regs[dma::ROW_OFFSET]);
        let stride = self.dma_regs[dma::STRIDE];
        let lines = self.dma_regs[dma::LINES];
        let len = self.dma_regs[dma::LINE_BYTES];
        if self.dma_logged < 16 {
            self.dma_logged += 1;
            eprintln!(
                "mgras: DMA {} pool {pool} table {table:#x} base {base:#x} stride {stride} lines {lines} bytes {len} pglist {:#x} shape {:?}",
                if read { "read" } else { "write" },
                self.dma_regs[dma::PAGE_LIST],
                self.raster.transfer_shape()
            );
        }
        let frame_of = |page: u32| -> Option<u32> {
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
        let mut changed = false;
        for i in 0..lines {
            let a = base.wrapping_add(i.wrapping_mul(stride));
            if read {
                let bytes = self.raster.dma_read_line(i);
                for (k, b) in bytes.iter().take(len as usize).enumerate() {
                    let Some(pa) = phys(a + k as u32) else { return changed };
                    mem.write8(pa, *b);
                }
            } else {
                let mut bytes = Vec::with_capacity(len as usize);
                for k in 0..len {
                    let Some(pa) = phys(a + k) else { return changed };
                    let r = mem.read8(pa);
                    bytes.push(if r.is_ok() { r.data } else { 0 });
                }
                self.raster.dma_write_line(i, &bytes);
                changed = true;
            }
        }
        changed
    }

    /// Flags as the host reads them, including those derived from state.
    fn all_flags(&self) -> u32 {
        self.flags | if self.ge_out.is_empty() { 0 } else { host::FLAG_GE_DIAG }
    }

    /// Whether the general interrupt (GIO line 1) is asserted: some enabled
    /// flag is set. It is a level, held until the handler clears the flag or
    /// its enable.
    fn general_irq(&self) -> bool {
        self.all_flags() & self.interrupt_enable & host::INTR_CAUSES != 0
    }

    /// Video timing chip display control bit 0: vertical retrace interrupts
    /// enabled.
    fn retrace_enabled(&self) -> bool {
        self.dcb.vc3.regs[0x1E] & 1 != 0
    }

    /// A write to a geometry engine's diagnostic data or address port.
    fn ge_diag_write(&mut self, off: u32, val: u32) {
        let n = host::GE_DIAG.iter().position(|&(d, a)| off == d || off == a).unwrap();
        let (data_port, _) = host::GE_DIAG[n];
        let ge = &mut self.ge[n];
        if off != data_port {
            if val & 0x8000_0000 != 0 {
                // Read request for `val & 0x7FFF_FFFF` lines from `addr`; it
                // replaces anything a previous request left unread.
                let words = ge.readback(val & 0x7FFF_FFFF);
                self.ge_out.clear();
                self.ge_out.extend(words);
            } else {
                ge.addr = val;
                ge.word = 0;
            }
            return;
        }
        if ge.addr >= GeDiag::UCODE_BASE {
            let line = ge.ucode.entry(ge.addr).or_insert([0; 3]);
            line[ge.word] = val;
            ge.word += 1;
            if ge.word == 3 {
                ge.word = 0;
                ge.addr += 1;
            }
        } else if ge.addr == GeDiag::EXEC_CONTROL && val & 1 != 0 {
            // Started: the version program's answer is waiting (revision 1).
            self.ge_readback = [0, 1];
            self.flags |= host::FLAG_GE_DATA;
        }
    }

    fn status(&self) -> u32 {
        // A vertical blank of about 1 ms in every 60 Hz frame.
        let us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64 % 16_667)
            .unwrap_or(0);
        host::STATUS_IDLE | if us < 1000 { host::STATUS_VBLANK } else { 0 }
    }

    fn read(&mut self, off: u32, bits: u32) -> u64 {
        match off {
            0 => GIO_ID as u64,
            host::UCODE..=0x45FFF => self.ucode[((off - host::UCODE) / 4) as usize] as u64,
            0x60000..=0x67FFF => {
                let t = dcb::Txn::decode(off);
                let d = self.dcb.read(t);
                t.load_value(bits, d)
            }
            host::STATUS => self.status() as u64,
            host::FIFOSTATUS | host::GIOSTATUS | host::DMABUSY => 0,
            host::SET_FLAGS | host::CLEAR_FLAGS | host::SET_FLAGS_PRIVILEGED | host::CLEAR_FLAGS_PRIVILEGED => self.all_flags() as u64,
            host::FLAG_ENABLE_SET | host::FLAG_ENABLE_CLEAR => self.flag_enable as u64,
            host::INTERRUPT_ENABLE_SET | host::INTERRUPT_ENABLE_CLEAR => self.interrupt_enable as u64,
            host::GE_DIAG_READ => self.ge_out.pop_front().unwrap_or(0) as u64,
            host::RASTER_IF_CONTEXT..=0x50228 => self.raster_if_regs[((off - host::RASTER_IF_CONTEXT) / 4) as usize] as u64,
            host::DMA_CONTEXT..=0x504FF => self.dma_regs[((off - host::DMA_CONTEXT) / 4) as usize] as u64,
            host::PIO_READ_HI => self.raster.pio_read_hi() as u64,
            host::PIO_READ_LO => self.raster.pio_read_lo() as u64,
            host::GE_READBACK_HI => self.ge_readback[0] as u64,
            host::GE_READBACK_LO => self.ge_readback[1] as u64,
            host::RASTER..=0x7FFFF => {
                let r = (off & 0xFFC) >> 2;
                if bits == 64 {
                    ((self.raster.read(r) as u64) << 32) | self.raster.read(r + 1) as u64
                } else {
                    self.raster.read(r) as u64
                }
            }
            _ => self.regs.get(&off).copied().unwrap_or(0) as u64,
        }
    }

    /// Returns true when the framebuffer changed.
    fn write(&mut self, off: u32, bits: u32, val: u64) -> bool {
        match off {
            host::UCODE..=0x45FFF => {
                self.ucode[((off - host::UCODE) / 4) as usize] = val as u32 & 0xFF_FFFF;
                false
            }
            0x60000..=0x67FFF => {
                let t = dcb::Txn::decode(off);
                self.dcb.write(t, t.store_data(bits, val));
                // Display-side state (colormaps, modes, the cursor) changes
                // what is shown without touching the framebuffer. Index
                // writes (select 1 and below) change nothing by themselves.
                t.crs >= 2 || t.dev == dcb::DEV_VC3
            }
            off if host::GE_DIAG.iter().any(|&(d, a)| off == d || off == a) => {
                self.ge_diag_write(off, val as u32);
                false
            }
            host::FLAG_ENABLE_SET => { self.flag_enable |= val as u32; false }
            host::FLAG_ENABLE_CLEAR => { self.flag_enable &= !(val as u32); false }
            host::INTERRUPT_ENABLE_SET => { self.interrupt_enable |= val as u32; false }
            host::INTERRUPT_ENABLE_CLEAR => { self.interrupt_enable &= !(val as u32); false }
            host::CONTEXT_SWITCH => {
                // The save phase completes at once; the incoming context
                // follows through the FIFO (see `cfifo_word`).
                self.flags |= host::FLAG_CONTEXT_SAVED;
                self.context_words = host::CONTEXT_SWITCH_WORDS;
                false
            }
            host::SET_FLAGS | host::SET_FLAGS_PRIVILEGED => { self.flags |= val as u32; false }
            host::CLEAR_FLAGS | host::CLEAR_FLAGS_PRIVILEGED => { self.flags &= !(val as u32); false }
            host::CFIFO | 0x70084 | host::CFIFO_PRIVILEGED | 0x50084 => {
                let port = if off >= host::STATUS { 0 } else { 1 };
                if bits == 64 {
                    let a = self.cfifo_word(port, (val >> 32) as u32);
                    let b = self.cfifo_word(port, val as u32);
                    a | b
                } else {
                    self.cfifo_word(port, val as u32)
                }
            }
            host::RASTER..=0x7FFFF => {
                let r = (off & 0xFFC) >> 2;
                // Registers at 0x7C000 + 4r; the same register at +0x1000
                // also executes the primitive in the IR after the write.
                let exec = off & 0x1000 != 0;
                if bits == 64 {
                    self.raster.write(r, (val >> 32) as u32, false);
                    self.raster.write(r + 1, val as u32, exec)
                } else {
                    self.raster.write(r, val as u32, exec)
                }
            }
            0x80000..=0xFFFFF => {
                self.note(format!("fast-path command window write at {off:#x}"));
                false
            }
            _ => {
                self.regs.insert(off, val as u32);
                false
            }
        }
    }

    /// The window ID a frame for the window at (`x`, `y`), `w` x `h` (screen
    /// coordinates, top-down) should be painted through: the commonest ID
    /// over a grid of samples inside it that has an RGB display mode. None
    /// if no sampled pixel is in an RGB window.
    fn window_did(&self, x: i32, y: i32, w: usize, h: usize) -> Option<u8> {
        let mut counts = [0u32; 32];
        let mut runs = Vec::new();
        for sy in 0..16 {
            let py = y + (h as i32 * (2 * sy + 1)) / 32;
            if !(0..raster::HEIGHT as i32).contains(&py) {
                continue;
            }
            self.dcb.vc3.main_did_runs(py as usize, &mut runs);
            for sx in 0..16 {
                let px = x + (w as i32 * (2 * sx + 1)) / 32;
                if !(0..raster::WIDTH as i32).contains(&px) {
                    continue;
                }
                let did = runs.iter().rev().find(|r| r.0 as i32 <= px).map_or(0, |r| r.1);
                if self.dcb.xmap.main_mode(did as u32) & 0x1F >= 4 {
                    counts[did as usize & 31] += 1;
                }
            }
        }
        let (did, n) = counts.iter().enumerate().max_by_key(|&(_, n)| *n)?;
        (*n > 0).then_some(did as u8)
    }

    /// Paint a host GL frame (`bgra`: `h` rows, top first, `stride` bytes
    /// each) into the framebuffer at screen position (`x`, `y`), wherever the
    /// pixel belongs to the frame's window ID, so windows over it stay over it.
    /// The pixels become part of the framebuffer, as GL's would on the board,
    /// for anything that reads them back. False when no window ID fits.
    fn composite(&mut self, x: i32, y: i32, bgra: &[u8], stride: usize, w: usize, h: usize) -> bool {
        let Some(target) = self.window_did(x, y, w, h) else { return false };
        let mut runs = Vec::new();
        for row in 0..h {
            let sy = y + row as i32;
            if !(0..raster::HEIGHT as i32).contains(&sy) {
                continue;
            }
            self.dcb.vc3.main_did_runs(sy as usize, &mut runs);
            if runs.is_empty() || runs[0].0 != 0 {
                runs.insert(0, (0, 0));
            }
            let fb_row = (raster::HEIGHT - 1 - sy as usize) * raster::WIDTH;
            for (k, &(x0, did)) in runs.iter().enumerate() {
                if did != target {
                    continue;
                }
                let x1 = runs.get(k + 1).map_or(raster::WIDTH as i32, |r| r.0 as i32);
                let lo = (x0 as i32).max(x).max(0);
                let hi = x1.min(x + w as i32).min(raster::WIDTH as i32);
                for sx in lo..hi {
                    let i = row * stride + (sx - x) as usize * 4;
                    let Some(p) = bgra.get(i..i + 4) else { break };
                    self.raster.fb[fb_row + sx as usize] = p[2] as u32 | (p[1] as u32) << 8 | (p[0] as u32) << 16;
                }
            }
        }
        true
    }

    /// Scan the framebuffer out to `0xFF_BB_GG_RR` (the compositor's order,
    /// red in the low byte), stride 2048, top row first. Each pixel's window
    /// ID (from the video timing chip's tables) picks its display mode. Modes
    /// with a pixel format (bits 4:0) of 4 and up are RGB; the others are
    /// colour index, into the colormap block that bits 9:5 choose. Both go
    /// through the DAC gamma.
    fn scanout(&self, out: &mut [u32]) {
        if self.dcb.dac.pixmask() == 0 {
            out.iter_mut().for_each(|p| *p = 0xFF00_0000);
            return;
        }
        let pal = &self.dcb.cmap[0].pal;
        let gamma = &self.dcb.dac.gamma;
        let gamma_rgb = |r: u32, g: u32, b: u32| -> u32 {
            let g1 = |v: u32, comp: usize| (gamma[(v & 0xFF) as usize][comp] >> 2) as u32;
            0xFF00_0000 | (g1(b, 2) << 16) | (g1(g, 1) << 8) | g1(r, 0)
        };
        // Per window ID: None for an RGB mode, else the colormap block's
        // entries through the gamma tables.
        let luts: Vec<Option<Vec<u32>>> = (0..32u32)
            .map(|did| {
                let mode = self.dcb.xmap.main_mode(did);
                if mode & 0x1F >= 4 {
                    return None;
                }
                let base = ((mode >> 5) & 0x1F) as usize * 256;
                Some(
                    (0..4096usize)
                        .map(|i| {
                            let c = pal.get((base + i) % pal.len().max(1)).copied().unwrap_or(0);
                            gamma_rgb(c >> 16, c >> 8, c)
                        })
                        .collect(),
                )
            })
            .collect();
        let mut runs = Vec::new();
        for row in 0..raster::HEIGHT {
            let y = raster::HEIGHT - 1 - row;
            let src = &self.raster.fb[y * raster::WIDTH..(y + 1) * raster::WIDTH];
            let dst = &mut out[row * 2048..row * 2048 + raster::WIDTH];
            self.dcb.vc3.main_did_runs(row, &mut runs);
            if runs.is_empty() || runs[0].0 != 0 {
                runs.insert(0, (0, 0));
            }
            for (k, &(x0, did)) in runs.iter().enumerate() {
                let x0 = (x0 as usize).min(raster::WIDTH);
                let x1 = runs.get(k + 1).map(|r| (r.0 as usize).min(raster::WIDTH)).unwrap_or(raster::WIDTH);
                if x1 <= x0 {
                    continue;
                }
                match &luts[did as usize & 31] {
                    Some(lut) => {
                        for x in x0..x1 {
                            dst[x] = lut[(src[x] & 0xFFF) as usize];
                        }
                    }
                    None => {
                        for x in x0..x1 {
                            let v = src[x];
                            dst[x] = gamma_rgb(v, v >> 8, v >> 16);
                        }
                    }
                }
            }
        }
        self.draw_overlay(out, &gamma_rgb);
        self.draw_cursor(out, &gamma_rgb);
    }

    /// Overlay planes over the main scanout: a nonzero pixel is a colour
    /// index into the block its overlay mode names (bits 7:3); zero, or a
    /// window ID whose overlay is off, shows the main planes.
    fn draw_overlay(&self, out: &mut [u32], gamma_rgb: &dyn Fn(u32, u32, u32) -> u32) {
        let pal = &self.dcb.cmap[0].pal;
        let bases: Vec<Option<usize>> = (0..32u32)
            .map(|did| {
                let mode = self.dcb.xmap.overlay_mode(did);
                (mode != 0).then(|| ((mode >> 3) & 0x1F) as usize * 256)
            })
            .collect();
        if bases.iter().all(Option::is_none) {
            return;
        }
        let mut runs = Vec::new();
        for row in 0..raster::HEIGHT {
            let y = raster::HEIGHT - 1 - row;
            let src = &self.raster.overlay[y * raster::WIDTH..(y + 1) * raster::WIDTH];
            let dst = &mut out[row * 2048..row * 2048 + raster::WIDTH];
            self.dcb.vc3.overlay_did_runs(row, &mut runs);
            if runs.is_empty() || runs[0].0 != 0 {
                runs.insert(0, (0, 0));
            }
            for (k, &(x0, did)) in runs.iter().enumerate() {
                let Some(base) = bases[did as usize & 31] else { continue };
                let x0 = (x0 as usize).min(raster::WIDTH);
                let x1 = runs.get(k + 1).map(|r| (r.0 as usize).min(raster::WIDTH)).unwrap_or(raster::WIDTH);
                for x in x0..x1 {
                    let v = src[x] & 0xFF;
                    if v != 0 {
                        let c = pal.get((base + v as usize) % pal.len().max(1)).copied().unwrap_or(0);
                        dst[x] = gamma_rgb(c >> 16, c >> 8, c);
                    }
                }
            }
        }
    }

    /// Overlay the hardware cursor, its colours from the cursor colormap.
    fn draw_cursor(&self, out: &mut [u32], gamma_rgb: &dyn Fn(u32, u32, u32) -> u32) {
        let Some((cx, cy, size, glyph)) = self.dcb.vc3.cursor() else { return };
        let sram = &self.dcb.vc3.sram;
        let pal = &self.dcb.cmap[0].pal;
        let base = self.dcb.xmap.cursor_cmap_base();
        let words_per_row = size / 16;
        let plane_words = size * words_per_row;
        let bit = |plane: usize, row: usize, col: usize| -> u32 {
            let w = sram[(glyph + plane * plane_words + row * words_per_row + col / 16) & 0x7FFF];
            (w >> (15 - col % 16)) as u32 & 1
        };
        for row in 0..size {
            let y = cy + row as i32;
            if !(0..raster::HEIGHT as i32).contains(&y) {
                continue;
            }
            for col in 0..size {
                let x = cx + col as i32;
                if !(0..raster::WIDTH as i32).contains(&x) {
                    continue;
                }
                let c = bit(0, row, col) | bit(1, row, col) << 1;
                if c != 0 {
                    let rgb = pal.get(base + c as usize).copied().unwrap_or(0xFF_FFFF);
                    out[y as usize * 2048 + x as usize] = gamma_rgb(rgb >> 16, rgb >> 8, rgb);
                }
            }
        }
    }
}

/// Access trace for bring-up: every access to the board, one line each
/// (`R`/`W`, width in bits, physical address, value). Started from
/// `IRIS_MGRAS_TRACE=<file>` or the monitor (`mgras trace <file>`, `mgras
/// trace off`). Off, it costs one relaxed load per access.
mod trace {
    use std::io::Write;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::OnceLock;
    use parking_lot::Mutex;

    static ON: AtomicBool = AtomicBool::new(false);

    fn sink() -> &'static Mutex<Option<std::io::BufWriter<std::fs::File>>> {
        static SINK: OnceLock<Mutex<Option<std::io::BufWriter<std::fs::File>>>> = OnceLock::new();
        SINK.get_or_init(|| {
            let w = std::env::var_os("IRIS_MGRAS_TRACE").and_then(|p| open(&p).ok());
            ON.store(w.is_some(), Ordering::Relaxed);
            Mutex::new(w)
        })
    }

    fn open(path: &std::ffi::OsStr) -> std::io::Result<std::io::BufWriter<std::fs::File>> {
        let f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        Ok(std::io::BufWriter::new(f))
    }

    /// Start tracing to `path`, or stop with `None`.
    pub fn set(path: Option<&str>) -> std::io::Result<()> {
        let mut s = sink().lock();
        if let Some(mut w) = s.take() {
            let _ = w.flush();
        }
        if let Some(p) = path {
            *s = Some(open(std::ffi::OsStr::new(p))?);
        }
        ON.store(s.is_some(), Ordering::Relaxed);
        Ok(())
    }

    pub fn init() {
        let _ = sink();
    }

    pub fn note(dir: char, bits: u32, addr: u32, val: u64) {
        if !ON.load(Ordering::Relaxed) {
            return;
        }
        if let Some(w) = sink().lock().as_mut() {
            let _ = writeln!(w, "{dir}{bits} {addr:08x} {val:x}");
        }
    }
}

/// The board's three interrupt lines to the GIO slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Line {
    /// GIO interrupt 0: command FIFO high/low water.
    Fifo,
    /// GIO interrupt 1: the general graphics interrupt.
    General,
    /// GIO interrupt 2: vertical retrace.
    Retrace,
}

pub struct Mgras {
    kind: ImpactSlot,
    ioc: crate::ioc::Ioc,
    /// System memory, for the board's DMA (it is a GIO bus master).
    sys_mem: Mutex<Option<Arc<dyn BusDevice>>>,
    /// Current level of the general interrupt line.
    general: AtomicBool,
    board: Mutex<Board>,
    dirty: AtomicBool,
    running: AtomicBool,
    refresh: Mutex<Option<std::thread::JoinHandle<()>>>,
    renderer: Mutex<Option<Box<dyn Renderer>>>,
    /// The finished frame, handed to the renderer as a prebuilt picture
    /// (`Rex3Screen::prebuilt`), as GR2 does.
    screen: Mutex<crate::disp::Rex3Screen>,
    screenshot_pending: AtomicBool,
    heartbeat: Arc<AtomicU64>,
    fasttick: Arc<AtomicU64>,
    cycles: Mutex<crate::mips_core::CyclesPtr>,
}

impl Mgras {
    pub fn new(cfg: &ImpactSection, ioc: crate::ioc::Ioc, heartbeat: Arc<AtomicU64>, fasttick: Arc<AtomicU64>) -> Self {
        trace::init();
        Mgras {
            kind: cfg.gfx,
            ioc,
            sys_mem: Mutex::new(None),
            general: AtomicBool::new(false),
            board: Mutex::new(Board::new(cfg.gfx)),
            dirty: AtomicBool::new(true),
            running: AtomicBool::new(false),
            refresh: Mutex::new(None),
            renderer: Mutex::new(None),
            screen: Mutex::new(crate::disp::Rex3Screen::new()),
            screenshot_pending: AtomicBool::new(false),
            heartbeat,
            fasttick,
            cycles: Mutex::new(crate::mips_core::CyclesPtr::dangling()),
        }
    }

    /// Composite a host GL frame at a screen position (see `Board::composite`).
    pub fn composite(&self, x: i32, y: i32, bgra: &[u8], stride: usize, w: usize, h: usize) -> bool {
        let ok = self.board.lock().composite(x, y, bgra, stride, w, h);
        if ok {
            self.dirty.store(true, Ordering::Release);
        }
        ok
    }

    /// Give the board its path to system memory, for DMA.
    pub fn set_phys(&self, mem: Arc<dyn BusDevice>) {
        self.board.lock().mem = Some(mem.clone());
        *self.sys_mem.lock() = Some(mem);
    }

    /// Drive one of the board's interrupt lines (graphics slot wiring).
    fn set_line(&self, line: Line, active: bool) {
        use crate::ioc::IocInterrupt;
        let src = match line {
            Line::Fifo => IocInterrupt::GioSgFifo,
            Line::General => IocInterrupt::GioSgGraphics,
            Line::Retrace => IocInterrupt::GioSgRetrace,
        };
        self.ioc.set_interrupt(src, active);
    }

    /// Wire up the CPU cycle counter for the status bar's MIPS figure.
    pub fn set_cpu_cycles(&self, ptr: crate::mips_core::CyclesPtr) {
        *self.cycles.lock() = ptr;
    }

    fn offset(&self, addr: u32) -> Option<u32> {
        let off = addr.wrapping_sub(MGRAS_SLOT_GFX_BASE);
        (off < MAP_SIZE).then_some(off)
    }

    /// Bring the general interrupt line in line with the board's flags.
    fn update_general(&self, level: bool) {
        if self.general.swap(level, Ordering::AcqRel) != level {
            self.set_line(Line::General, level);
        }
    }

    fn do_read(&self, addr: u32, bits: u32) -> u64 {
        let v = match self.offset(addr) {
            Some(off) => {
                let mut b = self.board.lock();
                let v = b.read(off, bits);
                let irq = b.general_irq();
                drop(b);
                self.update_general(irq);
                v
            }
            None => 0,
        };
        trace::note('R', bits, addr, v);
        v
    }

    fn do_write(&self, addr: u32, bits: u32, val: u64) {
        trace::note('W', bits, addr, val);
        if let Some(off) = self.offset(addr) {
            let mut b = self.board.lock();
            let changed = b.write(off, bits, val);
            let irq = b.general_irq();
            drop(b);
            if changed {
                self.dirty.store(true, Ordering::Release);
            }
            self.update_general(irq);
        }
    }

    fn refresh_loop(self: &Arc<Self>) {
        let frame = std::time::Duration::from_micros(16_667);
        {
            let mut screen = self.screen.lock();
            screen.width = raster::WIDTH;
            screen.height = raster::HEIGHT;
            screen.fb_rgb.fill(0xFF00_0000);
            screen.prebuilt = true;
        }
        let mut overlay = crate::debug_overlay::DebugOverlay::new();
        let mut status_bar = crate::disp::StatusBar::new();
        let mut sbtex = crate::disp::StatusBarTexture::new();
        let mut sized = false;
        let mut last_pending = 0u64;
        let mut idle_frames = 0u32;
        const PERSISTENT: u64 = crate::dev::ng1::rex3::Rex3::HB_LED_RED | crate::dev::ng1::rex3::Rex3::HB_LED_GREEN;

        while self.running.load(Ordering::Relaxed) {
            let start = std::time::Instant::now();
            let stats = crate::disp::BarStats {
                now: start,
                hb: self.heartbeat.fetch_and(PERSISTENT, Ordering::Relaxed),
                cycles: self.cycles.lock().get(),
                fasttick: self.fasttick.load(Ordering::Relaxed),
                decoded_delta: 0,
                l1i_hits: 0,
                l1i_fetches: 0,
                uncached: 0,
                count_hz: 0,
                gfifo_pending: 0,
            };
            let retrace = {
                let mut b = self.board.lock();
                if b.raster.flush_if_stale(&mut last_pending) {
                    self.dirty.store(true, Ordering::Release);
                }
                b.retrace_enabled()
            };
            // Vertical retrace: a pulse per frame. The handler acknowledges
            // nothing on the board, so the line must drop again by itself.
            if retrace {
                self.set_line(Line::Retrace, true);
                std::thread::sleep(std::time::Duration::from_micros(500));
                self.set_line(Line::Retrace, false);
            }
            let dirty = self.dirty.swap(false, Ordering::AcqRel);
            let shot = self.screenshot_pending.swap(false, Ordering::Relaxed);
            idle_frames += 1;
            if dirty || shot || idle_frames >= 6 {
                idle_frames = 0;
                let mut screen = self.screen.lock();
                if dirty || shot {
                    let screen = &mut *screen;
                    self.board.lock().scanout(&mut screen.fb_rgb);
                    // The frame is already final RGB, so keep `rgba` (what CI
                    // screenshots read) current without a renderer readback,
                    // as GR2 does. This also makes screenshots work headless.
                    for y in 0..raster::HEIGHT {
                        let row = y * 2048;
                        screen.rgba[row..row + raster::WIDTH].copy_from_slice(&screen.fb_rgb[row..row + raster::WIDTH]);
                    }
                    screen.status_bar_only = false;
                } else {
                    screen.status_bar_only = true;
                }
                if let Some(r) = self.renderer.lock().as_mut() {
                    if !sized {
                        r.resize(raster::WIDTH, raster::HEIGHT);
                        sized = true;
                    }
                    r.present(&mut screen, &mut overlay, &mut status_bar, &mut sbtex, &stats, shot, None, None);
                }
            }
            if let Some(rest) = frame.checked_sub(start.elapsed()) {
                std::thread::sleep(rest);
            }
        }
        // The renderer's GL state belongs to this thread (its context is
        // current here), so it is torn down here and nowhere else.
        if let Some(r) = self.renderer.lock().as_mut() {
            r.stop();
        }
    }

    /// Start the display refresh thread.
    pub fn start_display(self: &Arc<Self>) {
        if self.running.swap(true, Ordering::AcqRel) {
            return;
        }
        let me = Arc::clone(self);
        *self.refresh.lock() = Some(
            std::thread::Builder::new()
                .name("MGRAS-Refresh".into())
                .spawn(move || me.refresh_loop())
                .expect("spawn MGRAS refresh thread"),
        );
    }

    pub fn stop_display(&self) {
        self.running.store(false, Ordering::Release);
        if let Some(h) = self.refresh.lock().take() {
            let _ = h.join();
        }
    }
}

impl crate::gfx_display::GfxDisplay for Mgras {
    fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>> { &self.renderer }
    fn screen(&self) -> &Mutex<crate::disp::Rex3Screen> { &self.screen }
    fn request_screenshot(&self) { self.screenshot_pending.store(true, Ordering::Relaxed); }
    fn cycles(&self) -> crate::mips_core::CyclesPtr { *self.cycles.lock() }
}

impl Device for Mgras {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { self.running.load(Ordering::Relaxed) }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("mgras".into(), "IMPACT graphics: mgras (board state) | mgras shot <file.png> (save the displayed frame) | mgras dump <file> (raw display state) | mgras trace <file>|off".into())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn IoWrite + Send>) -> Result<(), String> {
        if cmd != "mgras" {
            return Err(format!("unknown command: {cmd}"));
        }
        if let ["trace", what] = args {
            let r = if *what == "off" { trace::set(None) } else { trace::set(Some(what)) };
            r.map_err(|e| format!("mgras trace: {e}"))?;
            return writeln!(w, "trace {what}").map_err(|e| e.to_string());
        }
        if let ["dump", path] = args {
            let b = self.board.lock();
            dump_state(path, &b).map_err(|e| format!("mgras dump: {e}"))?;
            return writeln!(w, "dumped {path}").map_err(|e| e.to_string());
        }
        if let ["shot", path] = args {
            let mut frame = vec![0u32; 2048 * raster::HEIGHT];
            self.board.lock().scanout(&mut frame);
            save_png(path, &frame).map_err(|e| format!("mgras shot: {e}"))?;
            return writeln!(w, "saved {path}").map_err(|e| e.to_string());
        }
        let b = self.board.lock();
        let e = |r: std::io::Result<()>| r.map_err(|e| e.to_string());
        e(writeln!(w, "IMPACT {:?} at {:#010x}", self.kind, MGRAS_SLOT_GFX_BASE))?;
        e(writeln!(w, "  flags {:#010x}  DAC pixmask {:#04x}  XMAP DID0 mode {:#x}",
            b.flags, b.dcb.dac.pixmask(), b.dcb.xmap.main_mode(0)))?;
        e(writeln!(w, "  fill modes seen: {:x?}", b.raster.fillmodes_seen))?;
        let mut hist: HashMap<u32, usize> = HashMap::new();
        for v in &b.raster.fb {
            *hist.entry(*v).or_default() += 1;
        }
        let mut top: Vec<_> = hist.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1));
        e(writeln!(w, "  framebuffer values (value, pixels): {:x?}", &top[..top.len().min(12)]))?;
        let pal = &b.dcb.cmap[0].pal;
        let blocks: Vec<usize> = (0..pal.len() / 256)
            .filter(|k| pal[k * 256..(k + 1) * 256].iter().any(|c| *c != 0))
            .collect();
        e(writeln!(w, "  colormap blocks in use (of {}): {:?}", pal.len() / 256, blocks))?;
        for k in blocks.iter().take(4) {
            e(writeln!(w, "    block {k}: {:06x?}", &pal[k * 256..k * 256 + 8]))?;
        }
        let modes: Vec<(u32, u32)> = (0..32).map(|d| (d, b.dcb.xmap.main_mode(d))).filter(|m| m.1 != 0).collect();
        e(writeln!(w, "  XMAP main modes (did, mode): {:x?}", modes))?;
        let mut runs = Vec::new();
        for y in [0usize, 100, 400, 700, 1023] {
            b.dcb.vc3.main_did_runs(y, &mut runs);
            e(writeln!(w, "  scanline {y} DID runs: {:?}", runs))?;
        }
        for u in &b.unhandled {
            e(writeln!(w, "  not modelled: {u}"))?;
        }
        Ok(())
    }
}

/// Save the display state for offline study: little-endian u32 sections, in
/// order: framebuffer (`WIDTH * HEIGHT`, row 0 at the bottom), overlay (same
/// size), colormap 0, the 32 main XMAP modes, then the video timing chip's
/// registers and SRAM as u32s.
fn dump_state(path: &str, b: &Board) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut put = |v: u32| f.write_all(&v.to_le_bytes());
    for v in b.raster.fb.iter().chain(b.raster.overlay.iter()).chain(b.dcb.cmap[0].pal.iter()) {
        put(*v)?;
    }
    for d in 0..32 {
        put(b.dcb.xmap.main_mode(d))?;
    }
    for v in b.dcb.vc3.regs.iter().chain(b.dcb.vc3.sram.iter()) {
        put(*v as u32)?;
    }
    Ok(())
}

/// Write a scanned-out frame (`0xFF_BB_GG_RR`, stride 2048) as a PNG.
fn save_png(path: &str, frame: &[u32]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), raster::WIDTH as u32, raster::HEIGHT as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut out = enc.write_header().map_err(|e| e.to_string())?;
    let mut rows = Vec::with_capacity(raster::WIDTH * raster::HEIGHT * 3);
    for y in 0..raster::HEIGHT {
        for px in &frame[y * 2048..y * 2048 + raster::WIDTH] {
            rows.extend_from_slice(&[*px as u8, (px >> 8) as u8, (px >> 16) as u8]);
        }
    }
    out.write_image_data(&rows).map_err(|e| e.to_string())
}

impl Saveable for Mgras {
    // Bring-up: board state is not snapshotted yet.
    fn save_state(&self) -> toml::Value {
        toml::Value::Table(toml::map::Map::new())
    }

    fn load_state(&self, _v: &toml::Value) -> Result<(), String> {
        Ok(())
    }
}

impl BusDevice for Mgras {
    fn read32(&self, addr: u32) -> BusRead32 { BusRead32::ok(self.do_read(addr, 32) as u32) }
    fn write32(&self, addr: u32, val: u32) -> u32 { self.do_write(addr, 32, val as u64); BUS_OK }
    fn read8(&self, addr: u32) -> BusRead8 { BusRead8::ok(self.do_read(addr, 8) as u8) }
    fn write8(&self, addr: u32, val: u8) -> u32 { self.do_write(addr, 8, val as u64); BUS_OK }
    fn read16(&self, addr: u32) -> BusRead16 { BusRead16::ok(self.do_read(addr, 16) as u16) }
    fn write16(&self, addr: u32, val: u16) -> u32 { self.do_write(addr, 16, val as u64); BUS_OK }
    fn read64(&self, addr: u32) -> BusRead64 { BusRead64::ok(self.do_read(addr, 64)) }
    fn write64(&self, addr: u32, val: u64) -> u32 { self.do_write(addr, 64, val); BUS_OK }
}

/// The board as the display host GL presents into: frames land in the
/// framebuffer under their window (see `Board::composite`). It cannot know the
/// guest X server's window ids, so only frames that say where their window is
/// are taken; the rest go back to the program to put up itself.
#[cfg(feature = "hostgl")]
pub struct ImpactScreen(pub Arc<Mgras>);

#[cfg(feature = "hostgl")]
impl iris_hostcall::Display for ImpactScreen {
    fn present(&self, _window: u32, _bgra: &[u8], _stride: usize, _width: usize, _height: usize) -> bool {
        false
    }

    fn present_at(&self, _window: u32, x: i32, y: i32, bgra: &[u8], stride: usize, width: usize, height: usize) -> bool {
        self.0.composite(x, y, bgra, stride, width, height)
    }
}
