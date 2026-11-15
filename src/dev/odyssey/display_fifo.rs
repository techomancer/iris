//! The display FIFO (board +0x400000): the processor's path to PB&J's
//! indexed register file, a stream of records, and the read window and I2C
//! words that read it back.
//!
//! It is the board's own path to the back end, apart from the command FIFO.
//! The driver paces it by its own level (`0x106c` bits 6:0, which the model
//! keeps at 0) and never waits for the command FIFO before a store. Some
//! stores it makes with its interrupts masked: a cursor position left
//! pending while another holder had the display FIFO is written by that
//! holder under the display FIFO's spin lock (`ignore/odsy`
//! `docs/kernel.md`). So the state has its own lock, and a store lands at
//! once, whatever the command FIFO and the raster unit hold; an access the
//! bus answered busy there would keep the guest's interrupts off for as long
//! as the queued drawing takes.
//!
//! The colour lookup RAM, the display modes, the overlay modes, the cursor
//! and the scan-out geometry go on to the display's atomic words
//! (`SharedRegs`); drawing reads the geometry from there, never through
//! this lock.

use std::sync::Arc;

use crate::devlog::LogModule;
use super::board::{ODSY_HEIGHT, ODSY_WIDTH};
use super::consts::{backend, cursor, dispmode, i2c, lut, overlay, ODSY_BLANK_LINES, ODSY_CHANNEL_HEIGHT, ODSY_CHANNEL_WIDTH, ODSY_FRAME_CLOCKS, ODSY_LINE_CLOCKS, ODSY_PIXEL_CLOCK_HZ};
use super::reader::{DisplayShared, SharedRegs};
use super::registers::reg;
use super::scanout::{Scanout, ScanoutMode};

/// The display FIFO's state.
pub(super) struct DisplayFifo {
    /// Back-end entries 0x2000-0x2fff as last written (0x1000 entries), for
    /// the read window.
    pub ctl: Vec<u32>,
    /// The open record's next entry, and the data words it still takes.
    pub index: u32,
    pub count: u32,
    /// The two I2C engines' command words.
    pub i2c_main_cmd: u32,
    pub i2c_opt_cmd: u32,
    /// What the control words spell on each port, and the clock parts'
    /// registers as written through them.
    two_wire: [TwoWire; 2],
    clock: ClockParts,
    /// The scan-out mode the channel-size entries set, and the display's
    /// shared state.
    pub scanout: Scanout,
    /// Once-only reports: an overlay mode with undecoded bits, the cursor's
    /// (bits of consts::cursor::REPORT_*), and a word the model does not
    /// decode.
    overlay_mode_reported: bool,
    cursor_reported: u32,
    word_reported: bool,
}

/// A two-wire port's transfer as its control words spell it. The kernel
/// drives the bus a byte at a time: a word with start, transfer and send
/// carries the address byte, one with transfer and send a data byte, and a
/// stop or either force bit ends the transfer. A write to a device is its
/// address, a register number, then that register's byte and its
/// successors'.
#[derive(Clone, Copy, Default)]
struct TwoWire {
    /// The address byte of the transfer open, when it is a write.
    address: Option<u8>,
    /// The register the next data byte goes to.
    register: Option<u8>,
}

impl TwoWire {
    const START: u32 = 1 << 8;
    const TRANSFER: u32 = 1 << 9;
    const STOP: u32 = 1 << 11;
    const SEND: u32 = 1 << 12;
    const FORCE: u32 = 1 << 13 | 1 << 14;

    /// Take a control word; a register byte it completes, as (device's
    /// seven-bit address, register, value).
    fn control(&mut self, word: u32) -> Option<(u8, u8, u8)> {
        if word & Self::FORCE != 0 {
            *self = TwoWire::default();
        } else if word & Self::TRANSFER != 0 && word & Self::SEND != 0 {
            let byte = word as u8;
            if word & Self::START != 0 {
                *self = TwoWire { address: (byte & 1 == 0).then_some(byte), register: None };
            } else if let Some(address) = self.address {
                match self.register {
                    None => self.register = Some(byte),
                    Some(r) => {
                        self.register = Some(r.wrapping_add(1));
                        return Some((address >> 1, r, byte));
                    }
                }
            }
        } else if word & Self::STOP != 0 {
            *self = TwoWire::default();
        }
        None
    }
}

/// The two clock parts a board can carry, as the kernel programs them: the
/// first revision an ICS1523 at 0x26, later boards a part at 0x58. Both
/// take a 12 MHz reference (the video format compiler's constant, which
/// every shipped format's achieved frequency follows from).
#[derive(Clone, Copy, Default)]
struct ClockParts {
    ics: [u8; 8],
    ami: [u8; 8],
    /// The part written last, which is the one the board has.
    last: Option<u8>,
}

impl ClockParts {
    const ICS1523: u8 = 0x26;
    const AMI: u8 = 0x58;
    const REFERENCE_HZ: u64 = 12_000_000;

    fn write(&mut self, device: u8, register: u8, value: u8) {
        let part = match device {
            Self::ICS1523 => &mut self.ics,
            Self::AMI => &mut self.ami,
            _ => return,
        };
        if let Some(r) = part.get_mut(register as usize) {
            *r = value;
            self.last = Some(device);
        }
    }

    /// The pixel clock the part written last makes; None before either is
    /// programmed, or while its dividers are not whole.
    fn pixel_clock_hz(&self) -> Option<u32> {
        let (feedback, reference) = match self.last? {
            // Feedback divider registers 2 and 3 (twelve bits, plus 8) over
            // the oscillator divider, register 7 (seven bits, plus 2).
            Self::ICS1523 => (self.ics[2] as u64 + 8 + 256 * (self.ics[3] & 0xf) as u64, (self.ics[7] & 0x7f) as u64 + 2),
            // Registers 4:3 over registers 1:0.
            _ => (self.ami[4] as u64 * 256 + self.ami[3] as u64, self.ami[1] as u64 * 256 + self.ami[0] as u64),
        };
        if reference == 0 || feedback == 0 {
            return None;
        }
        u32::try_from((Self::REFERENCE_HZ * feedback + reference / 2) / reference).ok()
    }
}

/// The timing generator's entries.
mod vtg {
    /// The line table: 0x400 words of two 16-bit entries.
    pub const LINE_TABLE: u32 = 0x2000;
    pub const LINE_WORDS: usize = 0x400;
    /// The frame table: 0x80 words, a (position, repeats) pair each.
    pub const FRAME_TABLE: u32 = 0x2400;
    pub const FRAME_ENTRIES: usize = 0x80;
    pub const CONTROL: u32 = 0x2481;
    /// A run's states: the line is in the picture.
    pub const STATE_PICTURE: u16 = 1 << 6;
}

/// `DisplayFifo::table_totals`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct VtgTotals {
    line_clocks: u32,
    lines: u32,
    visible_lines: u32,
    fields: u32,
}

/// Whether an access is the display FIFO's: a store to its port, or a load
/// of the read window (the back-end entries and the cursor) or of an I2C
/// engine's command or status word.
pub(super) fn display_fifo_access(addr: u64, size: u32, write: bool) -> bool {
    if write {
        return addr == backend::SELECT && (size == 8 || size == 4);
    }
    // The back end's version word sits in the window (entry 0x2701) but is
    // the board's own register (registers.rs).
    cursor_slot(addr, size).is_some()
        || (ctl_slot(addr, size).is_some() && reg(addr, size) != Some(backend::VERSION))
        || matches!(reg(addr, size), Some(i2c::MAIN_CMD_READ | i2c::MAIN_STATUS_READ | i2c::OPT_CMD_READ | i2c::OPT_STATUS_READ))
}

/// The cursor entry a 32-bit load at the start of its 8-byte read-window
/// slot names.
fn cursor_slot(addr: u64, size: u32) -> Option<usize> {
    let first = backend::READ_BASE + ((cursor::BASE as u64) << 3);

    (size == 4 && addr >= first && addr & 7 == 0 && addr < first + ((cursor::ENTRIES as u64) << 3)).then(|| ((addr - first) >> 3) as usize)
}

/// The back-end entry a 32-bit load at offset 4 of its 8-byte read-window
/// slot names.
fn ctl_slot(addr: u64, size: u32) -> Option<usize> {
    let first = backend::READ_BASE + ((backend::CTL_BASE as u64) << 3);

    (size == 4 && addr >= first && addr & 7 == 4 && addr < first + ((backend::CTL_ENTRIES as u64) << 3)).then(|| ((addr - first) >> 3) as usize)
}

// Back-end entries 0x2000-0x2fff: the line and frame tables, timing, cursor,
// DAC, clock, I2C, genlock, scanout and display-mode entries. Every word
// written there is kept, and a 32-bit load at offset 4 of the entry's 8-byte
// slot in the read window returns it (established by the kernel's
// read-modify-writes of 0x26c0, 0x2482, 0x2483 and 0x2700; that the board
// returns the last word written is inferred). Entries the board itself
// updates answer from their own models. Reset values are unknown; the model
// resets them to zero.
//
// The line table (0x2000-0x23ff) and frame table (0x2400-0x247f) are stored,
// and totalled for the retrace period (`DisplayFifo::table_totals`); the
// 1280x1024 60 Hz period stands until they are loaded.
//
// The scanout entries 0x2800 and 0x2801, 0x2807 and 0x2809 are stored and
// not used: APPROXIMATION, the model scans out its own colour buffers
// (resolve_row()) rather than locating the displayed buffer in board
// memory by these entries. The 16 x 16-pixel tiling of board memory is not
// modeled.

/// The entries the kernel writes.
pub(super) fn backend_ctl_known(index: u32) -> bool {
    (0x2000..=0x247f).contains(&index)
        || (0x2481..=0x2483).contains(&index)
        || index == 0x2600
        || index == 0x2680
        || index == 0x26c0
        || index == 0x2700
        || index == 0x2741
        || index == 0x2781
        || (0x27c0..=0x27cd).contains(&index)
        || (0x2800..=0x2809).contains(&index)
        || (0x2929..=0x292c).contains(&index)
        || index == 0x2b00
}

pub(super) fn i2c_status(command: u32) -> u32 {
    i2c::STATUS_READY | if command != 0 { i2c::STATUS_DONE } else { 0 }
}

impl DisplayFifo {
    pub fn new(shared: Arc<DisplayShared>) -> Self {
        DisplayFifo {
            ctl: vec![0; backend::CTL_ENTRIES as usize],
            index: 0,
            count: 0,
            i2c_main_cmd: 0,
            i2c_opt_cmd: 0,
            two_wire: Default::default(),
            clock: ClockParts::default(),
            scanout: Scanout::new(shared),
            overlay_mode_reported: false,
            cursor_reported: 0,
            word_reported: false,
        }
    }

    fn regs(&self) -> &SharedRegs {
        &self.scanout.shared.regs
    }

    /// The state a board reset leaves (the display's words are reset with
    /// the board's, `SharedRegs::reset`).
    pub fn reset(&mut self) {
        self.ctl.fill(0);
        self.index = 0;
        self.count = 0;
        self.i2c_main_cmd = 0;
        self.i2c_opt_cmd = 0;
        self.two_wire = Default::default();
        self.clock = ClockParts::default();
        self.overlay_mode_reported = false;
        self.cursor_reported = 0;
        self.word_reported = false;
    }

    /// A load display_fifo_access() names.
    pub fn read(&self, addr: u64, size: u32) -> u64 {
        if let Some(i) = cursor_slot(addr, size) {
            return self.regs().cursor(i) as u64;
        }
        if let Some(i) = ctl_slot(addr, size) {
            return self.ctl[i] as u64;
        }
        match reg(addr, size) {
            Some(i2c::MAIN_CMD_READ) => self.i2c_main_cmd as u64,
            Some(i2c::MAIN_STATUS_READ) => i2c_status(self.i2c_main_cmd) as u64,
            Some(i2c::OPT_CMD_READ) => self.i2c_opt_cmd as u64,
            Some(i2c::OPT_STATUS_READ) => i2c_status(self.i2c_opt_cmd) as u64,
            _ => 0,
        }
    }

    /// The port's store: an 8-byte store is two words, upper half first, a
    /// 4-byte store is one word (see backend::SELECT).
    pub fn write(&mut self, value: u64, size: u32) {
        if size == 8 {
            self.word((value >> 32) as u32);
            self.word(value as u32);
        } else {
            self.word(value as u32);
        }
    }

    /// One 32-bit word of the stream. While no record is open the word is a
    /// header: its 0b11 tag opens a record of `count` data words starting at
    /// `index` (a zero count is a pad). Any word without the tag at a header
    /// position has no established meaning and is reported rather than
    /// guessed. While a record is open the word is a data word for the
    /// running index, which then advances (see DisplayFifo::data).
    fn word(&mut self, word: u32) {
        if self.count == 0 {
            if ((word >> backend::TAG_SHIFT) & backend::TAG_MASK) != backend::TAG_HEADER {
                self.report_word(word);
                return;
            }
            self.index = (word >> backend::SELECTOR_INDEX_SHIFT) & backend::SELECTOR_INDEX_MASK;
            self.count = word & backend::COUNT_MASK;
            return;
        }
        self.data(self.index, word);
        self.index = (self.index + 1) & backend::SELECTOR_INDEX_MASK;
        self.count -= 1;
    }

    /// One data word of a record, delivered to the entry the record's header
    /// selected. The colour lookup RAM and the per-drawable display-mode
    /// registers take their words here; the two I2C command indices keep their
    /// existing dispatch. Entries 0x2000-0x2fff are also kept for the read
    /// window; one the kernel is not known to write is still reported.
    pub fn data(&mut self, index: u32, data: u32) {
        if (backend::CTL_BASE..backend::CTL_BASE + backend::CTL_ENTRIES).contains(&index) {
            self.ctl[(index - backend::CTL_BASE) as usize] = data;
        }
        if (super::frame::pbj::BASE..super::frame::pbj::BASE + 16).contains(&index) {
            self.regs().set_pbj((index - super::frame::pbj::BASE) as usize, data);
        }
        if index < lut::SIZE {
            self.regs().set_lut(index as usize, data & lut::ENTRY_MASK);
            return;
        }
        if (dispmode::BASE..dispmode::BASE + dispmode::COUNT).contains(&index) {
            self.regs().set_dispmode((index - dispmode::BASE) as usize, data);
            return;
        }
        if (cursor::BASE..cursor::BASE + cursor::ENTRIES).contains(&index) {
            self.regs().set_cursor((index - cursor::BASE) as usize, data);
            self.cursor_check(index - cursor::BASE, data);
            return;
        }
        if (overlay::MODE_BASE..overlay::MODE_BASE + overlay::IDS).contains(&index) {
            self.regs().set_overlay_mode((index - overlay::MODE_BASE) as usize, data);
            if (data & !overlay::MODE_KNOWN) != 0 && !self.overlay_mode_reported {
                self.overlay_mode_reported = true;
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: overlay mode {:#x} has bits this model does not decode", data);
            }
            return;
        }
        match index {
            ODSY_CHANNEL_WIDTH | ODSY_CHANNEL_HEIGHT => self.channel_size(index, data),
            i2c::MAIN_CMD_INDEX => {
                self.i2c_main_cmd = data;
                self.two_wire_control(0, data);
            }
            i2c::OPT_CMD_INDEX => {
                self.i2c_opt_cmd = data;
                self.two_wire_control(1, data);
            }
            // The timing generator's control: the kernel's mode load ends
            // with a write here, after the tables and the clock.
            vtg::CONTROL => self.retime(),
            _ => {
                if !backend_ctl_known(index) {
                    self.report_word(data);
                }
            }
        }
    }

    /// A word the model does not decode, reported once.
    fn report_word(&mut self, word: u32) {
        if !self.word_reported {
            self.word_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: display FIFO word {:#010x} at {:#x} is not modeled; further words stay silent", word, backend::SELECT);
        }
    }

    // The display back end's 32 by 32 hardware cursor, drawn over the resolved picture: after the lookup RAM
    // and the gamma table, over the main and overlay planes alike. Entries:
    //   0x2500  128 words  glyph: 32 rows of four words, eight 4-bit pixels a
    //                      word, the leftmost pixel in bits 3:0
    //   0x2580             position: x in 31:16, y in 15:0
    //   0x2581             control
    //   0x2590  15 words   colours, ten bits a channel, as in the lookup RAM
    //   0x25c0  64 words   plane: 32 rows of two words, sixteen 2-bit pixels a
    //                      word, the leftmost pixel in bits 1:0
    // The layouts are established; what the board makes of the values is
    // inferred. APPROXIMATION: under plane 3 a glyph value k shows colour word k
    // and glyph 0 shows word 0; under plane 0 the pixel is transparent. Glyph
    // value 15 and plane values 1 and 2 have no known meaning: reported once and
    // drawn transparent. A channel shows the top eight bits of its ten. Reset
    // values are unknown; the model resets every entry to zero, cursor off.

    fn cursor_report(&mut self, bit: u32, what: &str, index: u32, data: u32) {
        if self.cursor_reported & bit != 0 {
            return;
        }
        self.cursor_reported |= bit;
        crate::dlog!(LogModule::Odyssey, "sgi-odyssey: cursor entry {:#x} word {:#010x}: {}", cursor::BASE + index, data, what);
    }

    /// Report the values a stored cursor word carries that have no known
    /// meaning.
    fn cursor_check(&mut self, index: u32, data: u32) {
        if index < cursor::POSITION {
            if (0..32).step_by(4).any(|i| (data >> i) & 0xf == 0xf) {
                self.cursor_report(cursor::REPORT_GLYPH15, "glyph value 15 has no colour word; drawn transparent", index, data);
            }
        } else if index >= cursor::PLANE {
            if (0..32).step_by(2).any(|i| matches!((data >> i) & 3, 1 | 2)) {
                self.cursor_report(cursor::REPORT_PLANE, "plane values 1 and 2 are not modeled; drawn transparent", index, data);
            }
        } else if index == cursor::CONTROL {
            if (data & cursor::CTL_ENABLE) != 0 && (data & cursor::CTL_CROSSHAIR) != 0 {
                self.cursor_report(cursor::REPORT_CROSSHAIR, "crosshair drawn as full-raster lines in colour word 1, an approximation", index, data);
            }
            if data & cursor::CTL_CHANNEL1 != 0 {
                self.cursor_report(cursor::REPORT_CHANNEL1, "second display channel is not modeled; nothing drawn", index, data);
            }
            if data & !cursor::CTL_KNOWN != 0 {
                self.cursor_report(cursor::REPORT_CONTROL, "control bits outside 0-5 and 31:30 have no known meaning", index, data);
            }
        } else if index != cursor::POSITION && !(cursor::COLOURS..cursor::COLOURS + cursor::COLOUR_COUNT).contains(&index) {
            self.cursor_report(cursor::REPORT_ENTRY, "no known function; stored", index, data);
        }
    }

    /// Present the plane's top-left width x height, at the rate the
    /// timing tables and the clock part give.
    pub fn set_scanout_size(&mut self, width: u32, height: u32) {
        let mode = self.timed_mode(width, height);
        self.scanout.set_mode(&mode);
    }

    /// The timing changed (a mode load): publish the mode again.
    fn retime(&mut self) {
        let mode = self.timed_mode(self.scanout.mode.width, self.scanout.mode.height);
        let old = self.scanout.mode;
        if (mode.frame_clocks, mode.clock_hz, mode.line_clocks, mode.blank_lines, mode.fields) != (old.frame_clocks, old.clock_hz, old.line_clocks, old.blank_lines, old.fields) {
            self.scanout.set_mode(&mode);
        }
    }

    /// The mode of a width x height channel under the timing loaded: the
    /// tables' totals at the clock part's frequency, and the 1280x1024
    /// 60 Hz timing for whichever of the two is not loaded.
    fn timed_mode(&self, width: u32, height: u32) -> ScanoutMode {
        let clock_hz = self.clock.pixel_clock_hz().unwrap_or(ODSY_PIXEL_CLOCK_HZ);
        let (line_clocks, lines, blank_lines, fields) = match self.table_totals() {
            // Each run of picture lines is a field with its own retrace:
            // two for field-sequential stereo and the two-field formats.
            Some(t) if t.fields != 0 => (t.line_clocks, t.lines, (t.lines - t.visible_lines) / t.fields, t.fields),
            Some(t) => (t.line_clocks, t.lines, ODSY_BLANK_LINES.min(t.lines / 2), 1),
            None => (ODSY_LINE_CLOCKS, ODSY_FRAME_CLOCKS / ODSY_LINE_CLOCKS, ODSY_BLANK_LINES, 1),
        };
        ScanoutMode { width, height, frame_clocks: line_clocks as u64 * lines as u64, clock_hz, line_clocks, blank_lines, fields }
    }

    /// What the timing generator's tables add up to, as the kernel loads
    /// them: pixel clocks a line, lines a frame, the lines in the picture
    /// and the fields (runs of picture lines) a frame. The frame table
    /// (0x2400) is a word a line type: its repeat count in the high half,
    /// its position in the line table (in words) in the low, up to a zero
    /// word. The line table (0x2000) is 16-bit entries, two a word with
    /// the low half first, in the VC2's scheme: a run's first entry holds
    /// its length in pixel clocks in bits 14:8 and signal states in bits
    /// 6:0; with bit 7 clear a second entry of states follows, which takes
    /// no time; bit 15 of either ends the line. Every line type of every
    /// format checked adds up to the format's line length that way
    /// (1280x1024_60: 1688, of 1066 lines). A line is in the picture when
    /// a run of it has state bit 6 set: 1024 lines of that format's 1066,
    /// and two runs of 492 in the full-screen stereo format. None while no
    /// table is loaded.
    fn table_totals(&self) -> Option<VtgTotals> {
        let frame = &self.ctl[(vtg::FRAME_TABLE - backend::CTL_BASE) as usize..][..vtg::FRAME_ENTRIES];
        let line = &self.ctl[(vtg::LINE_TABLE - backend::CTL_BASE) as usize..][..vtg::LINE_WORDS];
        let entry = |i: usize| line.get(i / 2).map(|w| if i & 1 == 0 { *w as u16 } else { (w >> 16) as u16 });
        let mut t = VtgTotals::default();
        let mut in_picture = false;
        for word in frame.iter().take_while(|w| **w != 0) {
            let (repeats, mut at) = (word >> 16, 2 * (word & 0xffff) as usize);
            let (mut clocks, mut visible) = (0u32, false);
            loop {
                let first = entry(at)?;
                at += 1;
                let mut end = first & 0x8000 != 0;
                if first & 0x80 == 0 {
                    end |= entry(at)? & 0x8000 != 0;
                    at += 1;
                }
                clocks += ((first >> 8) & 0x7f) as u32;
                visible |= first & vtg::STATE_PICTURE != 0;
                if end {
                    break;
                }
            }
            t.line_clocks = t.line_clocks.max(clocks);
            t.lines += repeats;
            if visible {
                t.visible_lines += repeats;
                t.fields += !in_picture as u32;
            }
            in_picture = visible;
        }
        (t.lines != 0 && t.line_clocks != 0).then_some(t)
    }

    /// A control word pushed to two-wire port `port`: follow the transfer,
    /// and keep a clock part's register when one is written.
    fn two_wire_control(&mut self, port: usize, word: u32) {
        if let Some((device, register, value)) = self.two_wire[port].control(word) {
            self.clock.write(device, register, value);
        }
    }

    /// One of the display channel's two size entries. A zero is the guest
    /// blanking the channel while it changes format and leaves the picture as it
    /// was. A size past the modeled planes is reported and kept to them.
    fn channel_size(&mut self, index: u32, data: u32) {
        let mut width = self.scanout.mode.width;
        let mut height = self.scanout.mode.height;

        if data == 0 {
            return;
        }
        if index == ODSY_CHANNEL_WIDTH {
            width = data;
        } else {
            height = data;
        }
        if width > ODSY_WIDTH || height > ODSY_HEIGHT {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-odyssey: display channel {}x{} is larger than the modeled {}x{} planes",
                width,
                height,
                ODSY_WIDTH,
                ODSY_HEIGHT
            );
            width = width.min(ODSY_WIDTH);
            height = height.min(ODSY_HEIGHT);
        }
        if width != self.scanout.mode.width || height != self.scanout.mode.height {
            self.set_scanout_size(width, height);
        }
    }
}

#[cfg(test)]
mod timing_tests {
    use super::*;

    fn fifo() -> DisplayFifo {
        let front: Arc<parking_lot::Mutex<super::super::interface::HostInterface>> = Default::default();
        DisplayFifo::new(Arc::new(DisplayShared::new(front)))
    }

    /// The control words of one register write, as the kernel sends them:
    /// the address with start, then the register and its byte, then stop.
    fn write_register(f: &mut DisplayFifo, device: u8, register: u8, value: u8) {
        let send = TwoWire::TRANSFER | TwoWire::SEND;
        for word in [1 << 14, 0, send | TwoWire::START | (device as u32) << 1, send | register as u32, send | value as u32, TwoWire::STOP] {
            f.data(i2c::MAIN_CMD_INDEX, word);
        }
    }

    #[test]
    fn the_clock_parts_make_the_formats_clocks() {
        // The first part: 12 MHz x (feedback + 8) / (oscillator - 126).
        let mut f = fifo();
        assert_eq!(f.clock.pixel_clock_hz(), None);
        // 1920x1200_60: feedback 191 + 256, oscillator 154: 195 MHz.
        for (register, value) in [(2, 191), (3, 1), (7, 154)] {
            write_register(&mut f, 0x26, register, value);
        }
        assert_eq!(f.clock.pixel_clock_hz(), Some(195_000_000));
        // A read of the part (an odd address byte) writes nothing.
        let send = TwoWire::TRANSFER | TwoWire::SEND;
        for word in [send | TwoWire::START | 0x4d, send | 7, send | 0] {
            f.data(i2c::MAIN_CMD_INDEX, word);
        }
        assert_eq!(f.clock.pixel_clock_hz(), Some(195_000_000));
        // The second part: 12 MHz x registers 4:3 / registers 1:0. The
        // built-in 1280x1024 72 Hz timing's bytes: 0x30, 0, 0, 5, 2, ...
        // give 12 x 517 / 48 = 129.25 MHz.
        for (register, value) in [(0, 0x30), (1, 0), (2, 0), (3, 5), (4, 2)] {
            write_register(&mut f, 0x58, register, value);
        }
        assert_eq!(f.clock.pixel_clock_hz(), Some(129_250_000));
        // Another device's registers are not a clock's.
        write_register(&mut f, 0x38, 4, 0xff);
        assert_eq!(f.clock.pixel_clock_hz(), Some(129_250_000));
    }

    #[test]
    fn a_mode_load_sets_the_rate() {
        let mut f = fifo();
        f.set_scanout_size(1280, 1024);
        let t = f.scanout.mode.video_timing().unwrap();
        assert_eq!((t.h_total, t.v_total, t.vblank_lines, t.fields, t.refresh_mhz()), (1688, 1066, 42, 1, 59_999));
        // Two line types of 200 clocks: a blank one at position 3 (runs of
        // 100 and 60, the second with a states entry after it, and the end
        // run's 40), and a picture one at position 6 (its runs' bit 6).
        let entries = |f: &mut DisplayFifo, at: u32, e: [u16; 4]| {
            f.data(vtg::LINE_TABLE + at, (e[1] as u32) << 16 | e[0] as u32);
            f.data(vtg::LINE_TABLE + at + 1, (e[3] as u32) << 16 | e[2] as u32);
        };
        entries(&mut f, 3, [0x6488, 0x3c08, 0x45cf, 0xa888]);
        entries(&mut f, 6, [0x64c8, 0x3c48, 0x45cf, 0xa8c8]);
        // 20 blank lines, 480 of the picture, 25 blank.
        for (k, (repeats, at)) in [(20u32, 3u32), (480, 6), (25, 3)].into_iter().enumerate() {
            f.data(vtg::FRAME_TABLE + k as u32, repeats << 16 | at);
        }
        assert_eq!(f.table_totals(), Some(VtgTotals { line_clocks: 200, lines: 525, visible_lines: 480, fields: 1 }));
        // 12 MHz x 105 / 126 = 10 MHz over 200 x 525: 95.238 Hz.
        for (register, value) in [(0, 126), (1, 0), (3, 105), (4, 0)] {
            write_register(&mut f, 0x58, register, value);
        }
        f.set_scanout_size(640, 480);
        // Nothing changes until the kernel ends the load.
        f.data(vtg::CONTROL, 1);
        let t = f.scanout.mode.video_timing().unwrap();
        assert_eq!((t.h_total, t.v_total, t.pixel_clock_hz, t.vblank_lines, t.fields), (200, 525, 10_000_000, 45, 1));
        assert_eq!(t.refresh_mhz(), 95_238);
        // Two runs of picture lines: two fields, as full-screen stereo.
        for (k, (repeats, at)) in [(20u32, 3u32), (240, 6), (25, 3), (240, 6)].into_iter().enumerate() {
            f.data(vtg::FRAME_TABLE + k as u32, repeats << 16 | at);
        }
        f.data(vtg::CONTROL, 1);
        let t = f.scanout.mode.video_timing().unwrap();
        assert_eq!((t.v_total, t.fields, t.vblank_lines), (525, 2, 22));
    }
}
