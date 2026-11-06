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
use super::consts::{backend, cursor, dispmode, i2c, lut, overlay, ODSY_CHANNEL_HEIGHT, ODSY_CHANNEL_WIDTH, ODSY_FRAME_CLOCKS, ODSY_PIXEL_CLOCK_HZ};
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
// The line table (0x2000-0x23ff) and frame table (0x2400-0x247f) are only
// stored. APPROXIMATION: the retrace period stays the 1280x1024 60 Hz one
// (ODSY_FRAME_CLOCKS) rather than following the frame table.
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
            i2c::MAIN_CMD_INDEX => self.i2c_main_cmd = data,
            i2c::OPT_CMD_INDEX => self.i2c_opt_cmd = data,
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

    /// Present the plane's top-left width x height at the 1280x1024 rate.
    pub fn set_scanout_size(&mut self, width: u32, height: u32) {
        self.scanout.set_mode(&ScanoutMode { width, height, frame_clocks: ODSY_FRAME_CLOCKS as u64, clock_hz: ODSY_PIXEL_CLOCK_HZ });
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
