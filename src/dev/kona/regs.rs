//! The registers the board's thread owns and the stores that join the
//! command stream.
//!
//! Which accesses wait for the stream is `classify`: a store that belongs to
//! the command stream queues behind the stores before it; a register the
//! driver reads and acknowledges while the pipe works is served at once; every
//! other access reads what the stream writes or changes what it reads, so it
//! waits for the stream (the bus answers busy meanwhile). Inferred from the
//! driver's use, not measured: which registers the pipe serves while busy.
//!
//! The indirect port reaches a pipe-internal 32-bit address space: the
//! address port selects a word, the data port reads or writes it, and the
//! advancing write port writes it and adds 4 to the address. The firmware and
//! driver read the flash EEPROM's header through it; a user-space loader
//! writes the pipe's firmware through it. Approximated: the data port does not
//! advance the address. Every reader selects a new address before each access,
//! so no evidence says whether the hardware advances. The advancing write's
//! advance is what the loader depends on (its sections abut at address plus
//! four times their word count), inferred for the hardware.
//!
//! Ported from hw/misc/sgi-kona.c in jamesbraid/qemu-sgi-origami.

use super::defs::*;
use super::flash::FLASH_SIZE;
use super::front::{Front, Host};
use parking_lot::Mutex;
use super::ident;
use super::back::KonaBackend;
use super::job::{Job};
use super::pipe::{Core, LoadSection};
use crate::devlog::LogModule;

/// How an access meets the command stream.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Access {
    /// Served at once from the register file, whatever the stream is doing.
    Now,
    /// A store that joins the stream, in order.
    Queues,
    /// Waits for the stream to have run.
    Waits,
}

pub(super) fn classify(off: u32, write: bool) -> Access {
    let stream = if write { Access::Queues } else { Access::Waits };
    if (COMMAND_PAGE..COMMAND_PAGE + COMMAND_PAGE_SIZE).contains(&off) {
        return stream;
    }
    match off {
        DMA_FIFO_PUSH | DMA_FIFO_KICK | PORT_WRITE_NEXT => stream,
        INT_CAUSE | INT_ENABLE | MAILBOX => Access::Now,
        CREDIT => {
            if write {
                Access::Waits
            } else {
                Access::Now
            }
        }
        RETRACE_TAG | WIDGET_ID => {
            if write {
                Access::Waits
            } else {
                Access::Now
            }
        }
        // Software expectation (from the ported model): host software reads and
        // writes the low-window word and acknowledges video interrupts while
        // the pipe works, and polls the activity latch to decide whether the
        // pipe has hung. The command stream reaches none of them.
        LOW_WINDOW_WRITE | ACTIVITY => Access::Now,
        INTR_END => {
            if write {
                Access::Now
            } else {
                Access::Waits
            }
        }
        _ if (MAPRAM_BASE..MAPRAM_BASE + 4 * MAPRAM_ENTRIES as u32).contains(&off) => stream,
        _ => Access::Waits,
    }
}

impl Core {
    /// One word of a store that joined the stream.
    pub fn stream_store(&mut self, h: &Host, be: &mut KonaBackend, off: u32, value: u32) {
        match off {
            o if (COMMAND_PAGE..COMMAND_PAGE + COMMAND_PAGE_SIZE).contains(&o) => {
                self.command_page_write(h, be, o - COMMAND_PAGE, value)
            }
            DMA_FIFO_PUSH => {
                h.activity.store(true, std::sync::atomic::Ordering::Relaxed);
                self.push_dma_fifo(h, be, value);
            }
            DMA_FIFO_KICK => self.bracket_write(value),
            PORT_WRITE_NEXT => {
                h.activity.store(true, std::sync::atomic::Ordering::Relaxed);
                self.port_write_next(value);
            }
            o if (MAPRAM_BASE..MAPRAM_BASE + 4 * MAPRAM_ENTRIES as u32).contains(&o) => {
                let index = ((o - MAPRAM_BASE) / 4) as usize;
                if index < h.front.lock().mapram_count() {
                    self.map_ram[index] = value;
                } else {
                    crate::dlog!(LogModule::Kona, "kona: map RAM write at {o:#x} beyond the {} entries the probe register selects", h.front.lock().mapram_count());
                }
            }
            _ => unreachable!("offset {off:#x} is not a stream store"),
        }
    }

    /// A read that waits for the stream.
    pub fn read_wait(&mut self, front: &Mutex<Front>, off: u32) -> u32 {
        match off {
            PORT_ADDR => return self.port_addr,
            // Does not advance; see the module header.
            PORT_DATA => return self.internal_read(self.port_addr),
            INDIRECT_SELECT => return self.indirect_select,
            INDIRECT_DATA => {
                if let Some(w) = ident::class_word(&self.cfg, self.indirect_select) {
                    return w;
                }
            }
            o if (MAPRAM_BASE..MAPRAM_BASE + 4 * MAPRAM_ENTRIES as u32).contains(&o) => {
                let index = ((o - MAPRAM_BASE) / 4) as usize;
                let count = front.lock().mapram_count();
                if index < count {
                    return self.map_ram[index];
                }
                crate::dlog!(LogModule::Kona, "kona: map RAM read at {o:#x} beyond the {count} entries the probe register selects");
                return 0;
            }
            _ => {}
        }
        if let Some(v) = front.lock().read(off, 0) {
            return v;
        }
        crate::dlog!(LogModule::Kona, "kona: unimplemented read of offset {off:#x}");
        0
    }

    /// A write that waits for the stream.
    pub fn write_wait(&mut self, h: &Host, be: &mut KonaBackend, off: u32, value: u32) {
        match off {
            PORT_ADDR => {
                self.port_addr = value;
                self.load_open = false;
                return;
            }
            // The firmware and driver write 0 here to prime an EEPROM read;
            // outside the backed ranges that write is dropped.
            PORT_DATA => return self.internal_write(self.port_addr, value),
            INDIRECT_SELECT => {
                self.indirect_select = value;
                return;
            }
            INDIRECT_DATA => {
                self.job(h, be, Job::Indirect { selector: self.indirect_select, value });
                return;
            }
            RESET_STROBE => {
                // A write with bit 0 set resets what host software rewrites
                // after its 1, read, 0, read pulse (inferred from that
                // rewrite): the enable,
                // the global enable, the credit configuration, the group word
                // and every map RAM entry, which becomes invalid. It leaves
                // the DMA registers, the low-water state, the command framers
                // and the interrupt routing alone: the kernel never rewrites
                // them after a pulse and what the pulse does to them is
                // unknown. Bit 1 (ARM start) starts nothing: the model has no
                // ARM.
                h.front.lock().reset_strobe = value;
                if value & 1 != 0 {
                    h.front.lock().hip_reset();
                    self.map_ram.fill(0);
                }
                return;
            }
            LOADER_328 => {
                h.front.lock().loader_328 = value;
                if value == 1 {
                    self.firmware_start(h, be);
                }
                return;
            }
            _ => {}
        }
        let mut packet = None;
        let known = h.front.lock().write(off, value, &mut packet);
        if let Some(p) = packet {
            h.deliver(p);
        }
        if !known {
            crate::dlog!(LogModule::Kona, "kona: unimplemented write of offset {off:#x} (value {value:#x})");
        }
    }

    /// The loader released the firmware: the stored video format combination
    /// goes to the back end, as the released ARM would run it. The stored
    /// words are video commands framed as on the command page after 0x1a6.
    fn firmware_start(&mut self, h: &Host, be: &mut KonaBackend) {
        let count = self.flash.load(super::flash::COMBO + 4) as usize;
        let words: Vec<u32> = if self.flash.load(super::flash::COMBO) == super::flash::COMBO_MARKER && count < 0x8000 {
            (0..count).map(|i| self.flash.load(super::flash::COMBO + 8 + 4 * i as u32)).collect()
        } else {
            Vec::new()
        };
        self.job(h, be, Job::FirmwareStart { combination: &words });
    }

    // ----- the indirect port ------------------------------------------------

    pub(super) fn internal_read(&self, addr: u32) -> u32 {
        if addr.wrapping_sub(FLASH_BASE) < FLASH_SIZE as u32 && addr & 3 == 0 {
            return self.flash.load(addr - FLASH_BASE);
        }
        if addr == ICU_CONTROL {
            return self.icu_control;
        }
        self.internal_word(addr).copied().unwrap_or(0)
    }

    pub(super) fn internal_write(&mut self, addr: u32, value: u32) {
        if addr == ICU_CONTROL {
            let old = self.icu_control;
            self.icu_control = value;
            // The load record is closed where the loader releases the ARM's
            // memory: the 1-to-0 edge.
            if old == 1 && value == 0 {
                self.load_record = std::mem::take(&mut self.load_sections);
                self.load_open = false;
                self.load_count = 0;
            }
            return;
        }
        match self.internal_word_mut(addr) {
            Some(w) => *w = value,
            None => crate::dlog!(LogModule::Kona, "kona: internal write of {value:#x} to {addr:#x} dropped"),
        }
    }

    /// One word of the pipe-internal space the indirect port reaches, or
    /// None outside the two ranges the loader writes: the ARM's reset code at
    /// address 0 and its image, data and the geometry-engine microcode from
    /// 0x4000000. Backed with plain storage the model never executes or
    /// interprets.
    fn internal_word(&self, addr: u32) -> Option<&u32> {
        if addr & 3 != 0 {
            return None;
        }
        if ((addr - ARM_LOW_BASE) as usize) < ARM_LOW_WORDS * 4 {
            return self.arm_low.get(((addr - ARM_LOW_BASE) / 4) as usize);
        }
        if addr >= ARM_HIGH_BASE && ((addr - ARM_HIGH_BASE) as usize) < ARM_HIGH_WORDS * 4 {
            return self.arm_high.get(((addr - ARM_HIGH_BASE) / 4) as usize);
        }
        None
    }

    fn internal_word_mut(&mut self, addr: u32) -> Option<&mut u32> {
        if addr & 3 != 0 {
            return None;
        }
        if ((addr - ARM_LOW_BASE) as usize) < ARM_LOW_WORDS * 4 {
            return self.arm_low.get_mut(((addr - ARM_LOW_BASE) / 4) as usize);
        }
        if addr >= ARM_HIGH_BASE && ((addr - ARM_HIGH_BASE) as usize) < ARM_HIGH_WORDS * 4 {
            return self.arm_high.get_mut(((addr - ARM_HIGH_BASE) / 4) as usize);
        }
        None
    }

    /// The advancing write: a write after a select opens a new section of the
    /// load record, and each write extends the open one. The address wraps at
    /// 32 bits like the port register it lives in.
    fn port_write_next(&mut self, value: u32) {
        if !self.load_open {
            if self.load_sections.len() < LOAD_SECTIONS_MAX {
                self.load_sections.push(LoadSection { addr: self.port_addr, words: 0 });
            }
            self.load_count += 1;
            self.load_open = true;
        }
        // A section past the recorded ones keeps counting but is not kept.
        if let Some(s) = self.load_sections.get_mut(self.load_count - 1) {
            s.words += 1;
        }
        self.internal_write(self.port_addr, value);
        self.port_addr = self.port_addr.wrapping_add(4);
    }

    /// The sections of the last completed firmware load.
    pub fn load_record(&self) -> &[LoadSection] {
        &self.load_record
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stream_registers_queue_and_the_polled_ones_do_not_wait() {
        assert_eq!(classify(COMMAND_PAGE, true), Access::Queues);
        assert_eq!(classify(COMMAND_PAGE + 0xc, true), Access::Queues);
        assert_eq!(classify(COMMAND_PAGE, false), Access::Waits);
        assert_eq!(classify(DMA_FIFO_PUSH, true), Access::Queues);
        assert_eq!(classify(MAPRAM_BASE + 8, true), Access::Queues);
        assert_eq!(classify(MAPRAM_BASE + 8, false), Access::Waits);
        for off in [INT_CAUSE, INT_ENABLE, MAILBOX, LOW_WINDOW_WRITE, ACTIVITY] {
            assert_eq!(classify(off, false), Access::Now);
            assert_eq!(classify(off, true), Access::Now);
        }
        assert_eq!(classify(CREDIT, false), Access::Now);
        assert_eq!(classify(CREDIT, true), Access::Waits);
        assert_eq!(classify(WIDGET_ID, false), Access::Now);
        assert_eq!(classify(INTR_END, true), Access::Now);
        assert_eq!(classify(INTR_END, false), Access::Waits);
        assert_eq!(classify(PORT_DATA, false), Access::Waits);
        assert_eq!(classify(ONEWIRE, true), Access::Waits);
    }
}
