//! Register-level behaviour: identity, the widget registers, and the reports
//! made for offsets and commands the board does not implement. The back end's
//! registers are the display FIFO's (display_fifo.rs).
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.


use crate::devlog::LogModule;
use std::sync::atomic::Ordering;
use super::board::{Board, ODSY_UNIMP_REPORTS};
use super::consts::{
    backend, cfifo, config, ctx, flctl, host, i2c, intr, lut, mem, nic, route, stats, swap,
    ODSY_BLOCK_END, ODSY_BUZZ_1, ODSY_BUZZ_2, ODSY_BUZZ_3, ODSY_BUZZ_4, ODSY_BUZZ_5, ODSY_BUZZ_6,
    ODSY_DISPLAYED_BUFFER, ODSY_F32_ONE, ODSY_F32_POINT_EIGHT, ODSY_F32_POINT_TWO,
    ODSY_MANUFACTURER, ODSY_PART, ODSY_RESET_CONTROL, ODSY_STATUS0, ODSY_SW_TOKEN,
    ODSY_UNIMP_WRITE,
};
use super::display_fifo::display_fifo_access;
use super::interface::front_register;
use super::reader::SharedRegs;
use super::scanout::ScanoutMode;
impl Board {
    pub fn identity(&mut self) -> u32 {
        (self.widget.revision << 28) | (ODSY_PART << 12) | (ODSY_MANUFACTURER << 1) | 1
    }

    pub fn unimplemented(&self, addr: u64, size: u32, write: bool) {
        let token = (addr as u32) | if write { ODSY_UNIMP_WRITE } else { 0 };

        // A processor's access to an unmodelled offset reaches this while the
        // board's thread may be running a store of its own, so the report
        // table is atomic and the count bounded on every use; a race can only
        // log an offset twice.
        let len = self.diag.reported_addr.len();
        let seen = (self.diag.reported_addr_count.load(Ordering::Relaxed) as usize).min(len);
        if self.diag.reported_addr[..seen].iter().any(|a| a.load(Ordering::Relaxed) == token) || seen == len {
            return;
        }
        let slot = self.diag.reported_addr_count.fetch_add(1, Ordering::Relaxed) as usize;
        if slot >= len {
            return;
        }
        self.diag.reported_addr[slot].store(token, Ordering::Relaxed);
        crate::dlog!(
            LogModule::Odyssey,
            "sgi-odyssey: {}-byte {} at {:#x} is not modeled{}",
            size,
            if write { "write" } else { "read" },
            addr,
            if slot + 1 == len { "; further offsets stay silent" } else { "" }
        );
    }

    /// Report an unmodeled command word once. note follows the message: the
    /// parser's framing decision for a word it does not know, or "".
    pub fn cmd_report(&mut self, word: u32, note: &str) {
        if self.trace.on(super::trace::TRACE_FIFO) {
            self.trace_note(format_args!("unmodelled word {word:#010x}{note}"));
        }
        if self.diag.reported_cmd[..self.diag.reported_cmd_count as usize].contains(&word) {
            return;
        }
        if self.diag.reported_cmd_count == ODSY_UNIMP_REPORTS {
            return;
        }
        self.diag.reported_cmd[self.diag.reported_cmd_count as usize] = word;
        self.diag.reported_cmd_count += 1;
        crate::dlog!(
            LogModule::Odyssey,
            "sgi-odyssey: command FIFO word {:#010x} is not modeled{}{}",
            word,
            note,
            if self.diag.reported_cmd_count == ODSY_UNIMP_REPORTS { "; further words stay silent" } else { "" }
        );
    }

    pub fn cmd_unimplemented(&mut self, word: u32) {
        self.cmd_report(word, "");
    }
}

/// The modeled register an access names, or None. A configuration-block
/// register answers its 64-bit address and the 32-bit spelling of either
/// half identically (a live V12 board reads the same value at a register's
/// base and at base+4 across the block). Every other register has only been
/// seen as a 32-bit access. Other widths, alignments and offsets are
/// unmodeled rather than an error.
pub(super) fn reg(addr: u64, size: u32) -> Option<u64> {
    if addr < ODSY_BLOCK_END {
        if (size == 8 || size == 4) && (addr & 7) == 0 {
            return Some(addr);
        }
        if size == 4 && (addr & 7) == 4 {
            return Some(addr & !7);
        }
        return None;
    }
    if size != 4 {
        return None;
    }
    match addr {
        intr::FLAGS
        | intr::CLEAR
        | intr::ENABLE_SET
        | intr::ENABLE_CLEAR
        | route::DEST_UPPER
        | route::DEST_LOWER
        | route::VECTORS
        | ODSY_STATUS0
        | cfifo::SELFTEST
        | mem::CONFIG
        | flctl::CONFIG
        | flctl::TARGET
        | ODSY_SW_TOKEN
        | nic::PROM
        | nic::KERNEL
        | ODSY_BUZZ_1
        | ODSY_BUZZ_2
        | ODSY_BUZZ_3
        | ODSY_BUZZ_4
        | ODSY_BUZZ_5
        | ODSY_BUZZ_6
        | swap::ARM
        | swap::PENDING
        | swap::CONTROL
        | swap::CONTROL_COMPANION
        | ODSY_DISPLAYED_BUFFER
        | i2c::MAIN_CMD_READ
        | i2c::MAIN_STATUS_READ
        | i2c::OPT_CMD_READ
        | i2c::OPT_STATUS_READ
        | host::WRITE_ADDR_UPPER
        | host::WRITE_ADDR_LOWER
        | stats::ADDR_UPPER
        | stats::ADDR_LOWER
        | stats::CONTROL
        | ctx::SHADOW_ADDR_UPPER
        | ctx::SHADOW_ADDR_LOWER
        | ODSY_RESET_CONTROL
        | mem::CONTROL
        | backend::VERSION => Some(addr),
        _ => None,
    }
}

impl Board {
    /// One bit slot, reported finished inside the write the guest polls after.
    pub fn nic_slot(&mut self, value: u32) {
        let pulse = (value & nic::PULSE_MASK) >> nic::PULSE_SHIFT;
        let sample = (value & nic::SAMPLE_MASK) >> nic::SAMPLE_SHIFT;
        let level = match self.nic_bus.as_mut() {
            Some(bus) => crate::dev::w1::slot::slot(bus, pulse, sample),
            None => false,
        };

        self.widget.nic = (value & !(nic::LEVEL | nic::DONE)) | nic::DONE | if level { nic::LEVEL } else { 0 };
    }

    /// The registers the display resolves through (DisplayRegs).
    pub fn regs(&self) -> &SharedRegs {
        &self.display.regs
    }
}

/// A colour word, ten bits a channel, as 0x00RRGGBB from the top eight.
pub(super) fn cursor_rgb(word: u32) -> u32 {
    (((word >> lut::RED_SHIFT) & 0xff) << 16) | (((word >> lut::GREEN_SHIFT) & 0xff) << 8) | ((word >> lut::BLUE_SHIFT) & 0xff)
}


impl Board {
    /// +0x1004 (mem::CONTROL): kept, and checked against the strap.
    pub fn memory_control(&mut self, value: u32) {
        let strap = self.widget.memory_config;
        let size = (value >> mem::CONTROL_SIZE_SHIFT) & mem::CONTROL_SIZE_MASK;
        // The kernel programs strap code 3 like 128 MB, code 2.
        let strap_size = (strap >> config::SIZE_SHIFT).min(2);

        self.widget.memory_control = value;
        if !self.diag.memory_control_reported {
            self.diag.memory_control_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: memory controller configuration {:#x} stored; the controller is not modeled", value);
        }
        // The master id goes in first, with the strap fields still zero.
        if value & !(0xfu32 << 19) == 0 {
            return;
        }
        // MODEL CHOICE: a mismatch is reported; see mem::CONTROL.
        if size != strap_size
            || ((value & mem::CONTROL_FOUR_BANKS) != 0) != ((strap & config::FOUR_BANKS) != 0)
            || ((value & mem::CONTROL_LATENCY_2) != 0) != ((strap & config::LATENCY_2) != 0)
        {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: memory controller configuration {:#x} disagrees with the strap word {:#x}", value, strap);
        }
    }
}

impl Board {
    pub fn read(&mut self, addr: u64, size: u32) -> u64 {
        // Registers that do not wait for the board's work live in the front
        // (interface.rs).
        if front_register(addr, size, false) {
            return self.front.lock().read(addr, size);
        }
        let mut value: u64 = 0;
        let sequence = self.trace_sequence.fetch_add(1, Ordering::Relaxed);

        if display_fifo_access(addr, size, false) {
            value = self.display_fifo.lock().read(addr, size);
            crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: MMIO read seq {} offset {:#x} size {} value {:#x}", sequence, addr, size, value);
            return value;
        }

        match reg(addr, size) {
            Some(cfifo::SELFTEST) => {
                // Answers OK only after the kernel driver's command-FIFO
                // initialisation preamble; zero before.
                value = if self.widget.cfifo_selftest_ready { cfifo::SELFTEST_OK as u64 } else { 0 };
            }
            Some(mem::CONFIG) => {
                // Bits 15:7: the field and the drain waits are ESTABLISHED,
                // its unit UNKNOWN, and a count of unreturned credits
                // INFERRED. MODEL CHOICE: pending doublewords, the model's
                // pending words rounded up. Only reading 0 once the board has
                // run the queue is needed, by the start-up and flow-reset
                // drain waits.
                let doublewords = self.fifo.level().div_ceil(2).min(0x1ff);
                value = (self.widget.memory_config | (doublewords << 7)) as u64;
            }
            Some(flctl::CONFIG) => value = self.widget.flctl_config as u64,
            Some(ODSY_BUZZ_1) => value = self.widget.buzz[0] as u64,
            Some(ODSY_BUZZ_2) => value = self.widget.buzz[1] as u64,
            Some(ODSY_BUZZ_3) => value = self.widget.buzz[2] as u64,
            Some(ODSY_BUZZ_4) => value = self.widget.buzz[3] as u64,
            Some(ODSY_BUZZ_5) => value = self.widget.buzz[4] as u64,
            Some(ODSY_BUZZ_6) => value = self.widget.buzz[5] as u64,
            Some(nic::PROM | nic::KERNEL) => value = self.widget.nic as u64,
            Some(backend::VERSION) => value = (self.widget.backend_version & 7) as u64,
            Some(host::WRITE_ADDR_UPPER
            | host::WRITE_ADDR_LOWER
            | stats::ADDR_UPPER
            | stats::ADDR_LOWER
            | stats::CONTROL
            | ctx::SHADOW_ADDR_UPPER
            | ctx::SHADOW_ADDR_LOWER
            | ODSY_RESET_CONTROL
            | mem::CONTROL) => {
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: read of the write-only register at {:#x}", addr);
            }
            _ => self.unimplemented(addr, size, false),
        }
        crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: MMIO read seq {} offset {:#x} size {} value {:#x}", sequence, addr, size, value);
        value
    }

    /// Present the plane's top-left width x height at the 1280x1024 rate.
    pub fn set_scanout_size(&mut self, width: u32, height: u32) {
        self.display_fifo.lock().set_scanout_size(width, height);
    }

    /// The scan-out mode the display FIFO last set.
    pub fn scanout_mode(&self) -> ScanoutMode {
        self.display_fifo.lock().scanout.mode
    }
}

/// Deterministic model defaults; physical back-material reset is unknown.
/// (The query:: constants and the hardware shadow-page constants are in
/// draw.rs and consts.rs.)
pub(super) const BACK_MATERIAL_DEFAULTS: [u32; 16] = [
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
