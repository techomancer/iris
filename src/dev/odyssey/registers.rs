//! Register-level behaviour: identity, cursor, channel and back-end registers,
//! and the reports made for offsets and commands the board does not implement.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.

use super::*;

impl Board {
    pub fn identity(&mut self) -> u32 {
        (self.revision << 28) | (ODSY_PART << 12) | (ODSY_MANUFACTURER << 1) | 1
    }

    pub fn unimplemented(&mut self, addr: u64, size: u32, write: bool) {
        let token = (addr as u32) | if write { ODSY_UNIMP_WRITE } else { 0 };

        if self.reported_addr[..self.reported_addr_count as usize].contains(&token) {
            return;
        }
        if self.reported_addr_count == SGI_ODYSSEY_UNIMP_REPORTS {
            return;
        }
        self.reported_addr[self.reported_addr_count as usize] = token;
        self.reported_addr_count += 1;
        crate::dlog!(
            LogModule::Odyssey,
            "sgi-odyssey: {}-byte {} at {:#x} is not modeled{}",
            size,
            if write { "write" } else { "read" },
            addr,
            if self.reported_addr_count == SGI_ODYSSEY_UNIMP_REPORTS { "; further offsets stay silent" } else { "" }
        );
    }

    /// Report an unmodeled command word once. note follows the message: the
    /// parser's framing decision for a word it does not know, or "".
    pub fn cmd_report(&mut self, word: u32, note: &str) {
        if self.reported_cmd[..self.reported_cmd_count as usize].contains(&word) {
            return;
        }
        if self.reported_cmd_count == SGI_ODYSSEY_UNIMP_REPORTS {
            return;
        }
        self.reported_cmd[self.reported_cmd_count as usize] = word;
        self.reported_cmd_count += 1;
        crate::dlog!(
            LogModule::Odyssey,
            "sgi-odyssey: command FIFO word {:#010x} is not modeled{}{}",
            word,
            note,
            if self.reported_cmd_count == SGI_ODYSSEY_UNIMP_REPORTS { "; further words stay silent" } else { "" }
        );
    }

    pub fn cmd_unimplemented(&mut self, word: u32) {
        self.cmd_report(word, "");
    }
}

/// The modeled register an access names, or -1. A configuration-block
/// register answers its 64-bit address and the 32-bit spelling of either
/// half identically (a live V12 board reads the same value at a register's
/// base and at base+4 across the block). Every other register has only been
/// seen as a 32-bit access. Other widths, alignments and offsets are
/// unmodeled rather than an error.
pub fn reg(addr: u64, size: u32) -> i64 {
    if addr < ODSY_BLOCK_END {
        if (size == 8 || size == 4) && (addr & 7) == 0 {
            return addr as i64;
        }
        if size == 4 && (addr & 7) == 4 {
            return (addr & !7) as i64;
        }
        return -1;
    }
    if size != 4 {
        return -1;
    }
    match addr {
        ODSY_INTR_FLAGS
        | ODSY_INTR_CLEAR
        | ODSY_INTR_ENABLE_SET
        | ODSY_INTR_ENABLE_CLEAR
        | ODSY_ROUTE_DEST_UPPER
        | ODSY_ROUTE_DEST_LOWER
        | ODSY_ROUTE_VECTORS
        | ODSY_STATUS0
        | ODSY_CFIFO_SELFTEST
        | ODSY_MEMORY_CONFIG
        | ODSY_FLCTL_CONFIG
        | ODSY_FLCTL_TARGET
        | ODSY_SW_TOKEN
        | ODSY_NIC_PROM
        | ODSY_NIC_KERNEL
        | ODSY_BUZZ_1
        | ODSY_BUZZ_2
        | ODSY_BUZZ_3
        | ODSY_BUZZ_4
        | ODSY_BUZZ_5
        | ODSY_BUZZ_6
        | ODSY_SWAP_ARM
        | ODSY_SWAP_PENDING
        | ODSY_SWAP_CONTROL
        | ODSY_SWAP_CONTROL_COMPANION
        | ODSY_DISPLAYED_BUFFER
        | ODSY_I2C_MAIN_CMD_READ
        | ODSY_I2C_MAIN_STATUS_READ
        | ODSY_I2C_OPT_CMD_READ
        | ODSY_I2C_OPT_STATUS_READ
        | ODSY_HOST_WRITE_ADDR_UPPER
        | ODSY_HOST_WRITE_ADDR_LOWER
        | ODSY_STATS_ADDR_UPPER
        | ODSY_STATS_ADDR_LOWER
        | ODSY_STATS_CONTROL
        | ODSY_CTX_SHADOW_ADDR_UPPER
        | ODSY_CTX_SHADOW_ADDR_LOWER
        | ODSY_RESET_CONTROL
        | ODSY_MEMORY_CONTROL
        | ODSY_BACKEND_VERSION => addr as i64,
        _ => -1,
    }
}

impl Board {
    /// One bit slot, reported finished inside the write the guest polls after.
    pub fn nic_slot(&mut self, value: u32) {
        let pulse = (value & ODSY_NIC_PULSE_MASK) >> ODSY_NIC_PULSE_SHIFT;
        let sample = (value & ODSY_NIC_SAMPLE_MASK) >> ODSY_NIC_SAMPLE_SHIFT;
        let level = match self.nic_bus.as_mut() {
            Some(bus) => crate::dev::w1::slot::slot(bus, pulse, sample),
            None => false,
        };

        self.nic = (value & !(ODSY_NIC_LEVEL | ODSY_NIC_DONE)) | ODSY_NIC_DONE | if level { ODSY_NIC_LEVEL } else { 0 };
    }

    // The display back end's 32 by 32 hardware cursor (the datasheet's
    // "Hardware cursor"), drawn over the resolved picture: after the lookup RAM
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

    pub fn cursor_report(&mut self, bit: u32, what: &str, index: u32, data: u32) {
        if self.cursor_reported & bit != 0 {
            return;
        }
        self.cursor_reported |= bit;
        crate::dlog!(LogModule::Odyssey, "sgi-odyssey: cursor entry {:#x} word {:#010x}: {}", ODSY_CURSOR_BASE + index, data, what);
    }

    /// The position's components, each a signed 16-bit value (inferred).
    pub fn cursor_x(&mut self) -> i32 {
        (self.cursor[ODSY_CURSOR_POSITION as usize] >> 16) as i16 as i32
    }

    pub fn cursor_y(&mut self) -> i32 {
        self.cursor[ODSY_CURSOR_POSITION as usize] as i16 as i32
    }

    /// Whether anything is drawn: bit 0 set and bit 5 clear. APPROXIMATION: on
    /// the second channel the cursor is on a display the model does not have.
    pub fn cursor_shown(&mut self) -> bool {
        let ctl = self.cursor[ODSY_CURSOR_CONTROL as usize];

        (ctl & ODSY_CURSOR_CTL_ENABLE) != 0 && (ctl & ODSY_CURSOR_CTL_CHANNEL1) == 0
    }

    /// The display lines the cursor covers as it stands: first > last if none.
    pub fn cursor_span(&mut self, first: &mut i32, last: &mut i32) {
        *first = 0;
        *last = -1;
        if !self.cursor_shown() {
            return;
        }
        if self.cursor[ODSY_CURSOR_CONTROL as usize] & ODSY_CURSOR_CTL_CROSSHAIR != 0 {
            // The vertical line crosses every line.
            *last = self.scanout.mode.height as i32 - 1;
            return;
        }
        *first = self.cursor_y().max(0);
        *last = (self.cursor_y() + ODSY_CURSOR_SIZE as i32 - 1).min(self.scanout.mode.height as i32 - 1);
    }

    pub fn cursor_invalidate(&mut self, first: i32, last: i32) {
        if first <= last {
            self.scanout.invalidate(first, last - first + 1);
        }
    }
}

/// A colour word, ten bits a channel, as 0x00RRGGBB from the top eight.
pub fn cursor_rgb(word: u32) -> u32 {
    (((word >> ODSY_LUT_RED_SHIFT) & 0xff) << 16) | (((word >> ODSY_LUT_GREEN_SHIFT) & 0xff) << 8) | ((word >> ODSY_LUT_BLUE_SHIFT) & 0xff)
}

impl Board {
    /// Report the values a stored word carries that have no known meaning.
    pub fn cursor_check(&mut self, index: u32, data: u32) {
        if index < ODSY_CURSOR_POSITION {
            let mut i = 0;
            while i < 32 {
                if (data >> i) & 0xf == 0xf {
                    self.cursor_report(
                        ODSY_CURSOR_REPORT_GLYPH15,
                        "glyph value 15 has no colour word; drawn transparent",
                        index,
                        data,
                    );
                    break;
                }
                i += 4;
            }
        } else if index >= ODSY_CURSOR_PLANE {
            let mut i = 0;
            while i < 32 {
                let v = (data >> i) & 3;

                if v == 1 || v == 2 {
                    self.cursor_report(
                        ODSY_CURSOR_REPORT_PLANE,
                        "plane values 1 and 2 are not modeled; drawn transparent",
                        index,
                        data,
                    );
                    break;
                }
                i += 2;
            }
        } else if index == ODSY_CURSOR_CONTROL {
            if (data & ODSY_CURSOR_CTL_ENABLE) != 0 && (data & ODSY_CURSOR_CTL_CROSSHAIR) != 0 {
                self.cursor_report(
                    ODSY_CURSOR_REPORT_CROSSHAIR,
                    "crosshair drawn as full-raster lines in colour word 1, an approximation",
                    index,
                    data,
                );
            }
            if data & ODSY_CURSOR_CTL_CHANNEL1 != 0 {
                self.cursor_report(
                    ODSY_CURSOR_REPORT_CHANNEL1,
                    "second display channel is not modeled; nothing drawn",
                    index,
                    data,
                );
            }
            if data & !ODSY_CURSOR_CTL_KNOWN != 0 {
                self.cursor_report(
                    ODSY_CURSOR_REPORT_CONTROL,
                    "control bits outside 0-5 and 31:30 have no known meaning",
                    index,
                    data,
                );
            }
        } else if index != ODSY_CURSOR_POSITION && !(ODSY_CURSOR_COLOURS..ODSY_CURSOR_COLOURS + ODSY_CURSOR_COLOUR_COUNT).contains(&index) {
            self.cursor_report(ODSY_CURSOR_REPORT_ENTRY, "no known function; stored", index, data);
        }
    }

    /// One backend data word for a cursor entry: store it, report what it
    /// carries that has no known meaning, and repaint the lines the cursor
    /// covered before and covers now.
    pub fn cursor_write(&mut self, index: u32, data: u32) {
        let (mut old_first, mut old_last, mut first, mut last) = (0, 0, 0, 0);

        self.cursor_span(&mut old_first, &mut old_last);
        self.cursor[index as usize] = data;
        self.cursor_check(index, data);
        self.cursor_span(&mut first, &mut last);
        self.cursor_invalidate(old_first, old_last);
        self.cursor_invalidate(first, last);
    }

    /// Draw the cursor over display line y, already resolved into dst. With bit
    /// 2 it is a crosshair: APPROXIMATION, a one-pixel horizontal and vertical
    /// line across the whole raster through the position, in colour word 1. The
    /// crosshair's extent, width and colour are unknown.
    pub fn cursor_row(&mut self, y: u32, dst: &mut [u32], width: u32) {
        let cx = self.cursor_x();
        let cy = self.cursor_y();
        let row = (y as i32).wrapping_sub(cy);

        if !self.cursor_shown() {
            return;
        }
        if self.cursor[ODSY_CURSOR_CONTROL as usize] & ODSY_CURSOR_CTL_CROSSHAIR != 0 {
            let colour = cursor_rgb(self.cursor[ODSY_CURSOR_COLOURS as usize + 1]);

            if row == 0 {
                dst[..width as usize].fill(colour);
            } else if cx >= 0 && cx < width as i32 {
                dst[cx as usize] = colour;
            }
            return;
        }
        if row < 0 || row >= ODSY_CURSOR_SIZE as i32 {
            return;
        }
        for col in 0..ODSY_CURSOR_SIZE as i32 {
            let x = cx.wrapping_add(col);
            let plane = (self.cursor[(ODSY_CURSOR_PLANE as i32 + row * 2 + col / 16) as usize] >> ((col % 16) * 2)) & 3;
            let glyph = (self.cursor[(ODSY_CURSOR_GLYPH as i32 + row * 4 + col / 8) as usize] >> ((col % 8) * 4)) & 0xf;

            if x < 0 || x >= width as i32 || plane != 3 || glyph == 0xf {
                continue;
            }
            dst[x as usize] = cursor_rgb(self.cursor[ODSY_CURSOR_COLOURS as usize + glyph as usize]);
        }
    }

    /// A backend read-window offset that names a cursor entry: its stored word.
    pub fn cursor_read(&mut self, addr: u64, size: u32, value: &mut u64) -> bool {
        let first = ODSY_BACKEND_READ_BASE + ((ODSY_CURSOR_BASE as u64) << 3);

        if size != 4 || addr < first || (addr & 7) != 0 || addr >= first + ((ODSY_CURSOR_ENTRIES as u64) << 3) {
            return false;
        }
        *value = self.cursor[((addr - first) >> 3) as usize] as u64;
        true
    }
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

/// The entries the kernel writes (the datasheet's back-end entry table).
pub fn backend_ctl_known(index: u32) -> bool {
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

impl Board {
    pub fn backend_ctl_read(&mut self, addr: u64, size: u32, value: &mut u64) -> bool {
        let first = ODSY_BACKEND_READ_BASE + ((ODSY_BACKEND_CTL_BASE as u64) << 3);

        if size != 4 || addr < first || (addr & 7) != 4 || addr >= first + ((ODSY_BACKEND_CTL_ENTRIES as u64) << 3) {
            return false;
        }
        *value = self.backend_ctl[((addr - first) >> 3) as usize] as u64;
        true
    }

    /// +0x1004 (ODSY_MEMORY_CONTROL): kept, and checked against the strap.
    pub fn memory_control(&mut self, value: u32) {
        let strap = self.memory_config;
        let size = (value >> ODSY_MEMORY_CONTROL_SIZE_SHIFT) & ODSY_MEMORY_CONTROL_SIZE_MASK;
        // The kernel programs strap code 3 like 128 MB, code 2.
        let strap_size = (strap >> ODSY_CONFIG_SIZE_SHIFT).min(2);

        self.memory_control = value;
        if !self.memory_control_reported {
            self.memory_control_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: memory controller configuration {:#x} stored; the controller is not modeled", value);
        }
        // The master id goes in first, with the strap fields still zero.
        if value & !(0xfu32 << 19) == 0 {
            return;
        }
        // MODEL CHOICE: a mismatch is reported; see ODSY_MEMORY_CONTROL.
        if size != strap_size
            || ((value & ODSY_MEMORY_CONTROL_FOUR_BANKS) != 0) != ((strap & ODSY_CONFIG_FOUR_BANKS) != 0)
            || ((value & ODSY_MEMORY_CONTROL_LATENCY_2) != 0) != ((strap & ODSY_CONFIG_LATENCY_2) != 0)
        {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: memory controller configuration {:#x} disagrees with the strap word {:#x}", value, strap);
        }
    }
}

pub fn i2c_status(command: u32) -> u32 {
    ODSY_I2C_STATUS_READY | if command != 0 { ODSY_I2C_STATUS_DONE } else { 0 }
}

impl Board {
    pub fn read(&mut self, addr: u64, size: u32) -> u64 {
        // Registers that do not wait for the board's work live in the front
        // (interface.rs).
        if front_register(addr, size, false) {
            return self.front.lock().read(addr, size);
        }
        let mut value: u64 = 0;
        let sequence = self.trace_sequence;
        self.trace_sequence = self.trace_sequence.wrapping_add(1);

        let reg = reg(addr, size);
        let key = if reg < 0 { u64::MAX } else { reg as u64 };
        match key {
            ODSY_CFIFO_SELFTEST => {
                // Answers OK only after the kernel driver's command-FIFO
                // initialisation preamble; zero before.
                value = if self.cfifo_selftest_ready { ODSY_CFIFO_SELFTEST_OK as u64 } else { 0 };
            }
            ODSY_MEMORY_CONFIG => value = self.memory_config as u64,
            ODSY_FLCTL_CONFIG => value = self.flctl_config as u64,
            ODSY_FLCTL_TARGET => value = self.flctl_target as u64,
            ODSY_BUZZ_1 => value = self.buzz[0] as u64,
            ODSY_BUZZ_2 => value = self.buzz[1] as u64,
            ODSY_BUZZ_3 => value = self.buzz[2] as u64,
            ODSY_BUZZ_4 => value = self.buzz[3] as u64,
            ODSY_BUZZ_5 => value = self.buzz[4] as u64,
            ODSY_BUZZ_6 => value = self.buzz[5] as u64,
            ODSY_NIC_PROM | ODSY_NIC_KERNEL => value = self.nic as u64,
            ODSY_I2C_MAIN_CMD_READ => value = self.i2c_main_cmd as u64,
            ODSY_I2C_MAIN_STATUS_READ => value = i2c_status(self.i2c_main_cmd) as u64,
            ODSY_I2C_OPT_CMD_READ => value = self.i2c_opt_cmd as u64,
            ODSY_I2C_OPT_STATUS_READ => value = i2c_status(self.i2c_opt_cmd) as u64,
            ODSY_BACKEND_VERSION => value = (self.backend_version & 7) as u64,
            ODSY_HOST_WRITE_ADDR_UPPER
            | ODSY_HOST_WRITE_ADDR_LOWER
            | ODSY_STATS_ADDR_UPPER
            | ODSY_STATS_ADDR_LOWER
            | ODSY_STATS_CONTROL
            | ODSY_CTX_SHADOW_ADDR_UPPER
            | ODSY_CTX_SHADOW_ADDR_LOWER
            | ODSY_RESET_CONTROL
            | ODSY_MEMORY_CONTROL => {
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: read of the write-only register at {:#x}", addr);
            }
            _ => {
                if self.cursor_read(addr, size, &mut value) || self.backend_ctl_read(addr, size, &mut value) {
                    crate::dlog_dev!(LogModule::Odyssey, "sgi_odyssey_mmio: seq {} offset {:#x} size {} value {:#x} write=0", sequence, addr, size, value);
                    return value;
                }
                self.unimplemented(addr, size, false);
            }
        }
        crate::dlog_dev!(LogModule::Odyssey, "sgi_odyssey_mmio: seq {} offset {:#x} size {} value {:#x} write=0", sequence, addr, size, value);
        value
    }

    /// Present the plane's top-left width x height at the 1280x1024 rate.
    pub fn set_scanout_size(&mut self, width: u32, height: u32) {
        self.scanout.set_mode(
            &ScanoutMode { width, height, frame_clocks: ODSY_FRAME_CLOCKS as u64, clock_hz: ODSY_PIXEL_CLOCK_HZ },
        );
    }

    /// One of the display channel's two size entries. A zero is the guest
    /// blanking the channel while it changes format and leaves the picture as it
    /// was. A size past the modeled planes is reported and kept to them.
    pub fn channel_size(&mut self, index: u32, data: u32) {
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
        if width > SGI_ODYSSEY_WIDTH || height > SGI_ODYSSEY_HEIGHT {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-odyssey: display channel {}x{} is larger than the modeled {}x{} planes",
                width,
                height,
                SGI_ODYSSEY_WIDTH,
                SGI_ODYSSEY_HEIGHT
            );
            width = width.min(SGI_ODYSSEY_WIDTH);
            height = height.min(SGI_ODYSSEY_HEIGHT);
        }
        if width != self.scanout.mode.width || height != self.scanout.mode.height {
            self.set_scanout_size(width, height);
        }
    }

    /// One data word of a backend record, delivered to the entry the record's
    /// header selected. The colour lookup RAM and the per-drawable display-mode
    /// registers take their words here; the two I2C command indices keep their
    /// existing dispatch. Entries 0x2000-0x2fff are also kept for the read
    /// window (Board::backend_ctl_read()); one the kernel is not known to write
    /// is still reported.
    pub fn backend_data(&mut self, index: u32, data: u32) {
        if (ODSY_BACKEND_CTL_BASE..ODSY_BACKEND_CTL_BASE + ODSY_BACKEND_CTL_ENTRIES).contains(&index) {
            self.backend_ctl[(index - ODSY_BACKEND_CTL_BASE) as usize] = data;
        }
        // Scanout resolves through both, so either change repaints.
        if index < ODSY_LUT_SIZE {
            self.lut[index as usize] = data & ODSY_LUT_ENTRY_MASK;
            self.scanout.invalidate_all();
            return;
        }
        if (ODSY_DISPMODE_BASE..ODSY_DISPMODE_BASE + ODSY_DISPMODE_COUNT).contains(&index) {
            self.dispmode[(index - ODSY_DISPMODE_BASE) as usize] = data;
            self.scanout.invalidate_all();
            return;
        }
        if (ODSY_CURSOR_BASE..ODSY_CURSOR_BASE + ODSY_CURSOR_ENTRIES).contains(&index) {
            self.cursor_write(index - ODSY_CURSOR_BASE, data);
            return;
        }
        if (ODSY_OVERLAY_MODE_BASE..ODSY_OVERLAY_MODE_BASE + ODSY_OVERLAY_IDS).contains(&index) {
            self.overlay_mode[(index - ODSY_OVERLAY_MODE_BASE) as usize] = data;
            if (data & !ODSY_OVERLAY_MODE_KNOWN) != 0 && !self.overlay_mode_reported {
                self.overlay_mode_reported = true;
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: overlay mode {:#x} has bits this model does not decode", data);
            }
            self.scanout.invalidate_all();
            return;
        }
        match index {
            ODSY_CHANNEL_WIDTH | ODSY_CHANNEL_HEIGHT => self.channel_size(index, data),
            ODSY_I2C_MAIN_CMD_INDEX => self.i2c_main_cmd = data,
            ODSY_I2C_OPT_CMD_INDEX => self.i2c_opt_cmd = data,
            _ => {
                if !backend_ctl_known(index) {
                    self.unimplemented(ODSY_BACKEND_SELECT, 8, true);
                }
            }
        }
    }

    /// One 32-bit word of the backend word stream. While no record is open the
    /// word is a header: its 0b11 tag opens a record of `count` data words
    /// starting at `index` (a zero count is a pad). Any word without the tag at a
    /// header position has no established meaning and is reported rather than
    /// guessed. While a record is open the word is a data word for the running
    /// index, which then advances (see Board::backend_data).
    pub fn backend_word(&mut self, word: u32) {
        if self.backend_count == 0 {
            if ((word >> ODSY_BACKEND_TAG_SHIFT) & ODSY_BACKEND_TAG_MASK) != ODSY_BACKEND_TAG_HEADER {
                self.unimplemented(ODSY_BACKEND_SELECT, 8, true);
                return;
            }
            self.backend_index = (word >> ODSY_BACKEND_SELECTOR_INDEX_SHIFT) & ODSY_BACKEND_SELECTOR_INDEX_MASK;
            self.backend_count = word & ODSY_BACKEND_COUNT_MASK;
            return;
        }
        self.backend_data(self.backend_index, word);
        self.backend_index = (self.backend_index + 1) & ODSY_BACKEND_SELECTOR_INDEX_MASK;
        self.backend_count -= 1;
    }

    /// The indexed register file's write port: an 8-byte store is two words,
    /// upper half first, a 4-byte store is one word (see ODSY_BACKEND_SELECT).
    pub fn backend_write(&mut self, value: u64, size: u32) {
        if size == 8 {
            self.backend_word((value >> 32) as u32);
            self.backend_word(value as u32);
        } else {
            self.backend_word(value as u32);
        }
    }
}

/// Deterministic model defaults; physical back-material reset is unknown.
/// (The ODSY_QUERY_* enum and the hardware shadow-page constants are in
/// draw.rs and consts.rs.)
pub const BACK_MATERIAL_DEFAULTS: [u32; 16] = [
    ODSY_F32_POINT_TWO,
    ODSY_F32_POINT_TWO,
    ODSY_F32_POINT_TWO,
    ODSY_F32_ONE,
    ODSY_F32_POINT_EIGHT,
    ODSY_F32_POINT_EIGHT,
    ODSY_F32_POINT_EIGHT,
    ODSY_F32_ONE,
    0,
    0,
    0,
    ODSY_F32_ONE,
    0,
    0,
    0,
    ODSY_F32_ONE,
];
