//! The host interface registers a processor reaches without waiting for the
//! command stream: the widget header, the interrupt registers, the mailbox
//! and the plain-storage registers the driver and the firmware write.
//!
//! The mutex around `Front` only serialises the register file; it is held
//! for a few instructions and never across a call out of the board. A
//! method that raises an interrupt returns the packet and the caller sends
//! it after dropping the lock.
//!
//! Ported from hw/misc/sgi-kona.c in jamesbraid/qemu-sgi-origami.

use crate::irq::InterruptController;
use crate::physical;
use crate::traits::{BusDevice, BUS_OK};
use super::defs::*;
use super::ident::{self, Identity, KonaConfig};
use crate::devlog::LogModule;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

pub(super) struct Front {
    pub widget_control: u32,
    /// What it resets to: the host widget in bits 25:22 (the model's choice).
    widget_control_reset: u32,
    pub identity: Identity,
    pub onewire_reg: u32,
    pub probe_value: u32,
    pub low_window_write: u32,
    pub int_dest_upper: u32,
    pub int_dest_lower: u32,
    pub int_vectors: u32,
    pub reset_arg: u32,
    pub int_enable: u32,
    pub credit_config: u32,
    pub int_cause: u32,
    /// Reply words queued for the driver, `mailbox[pos..len]` unread. Bit
    /// `i` of `completions` marks word `i` (counted from `pos`) as an
    /// unsolicited completion rather than a reply: clearing the mailbox
    /// cause bit consumes it.
    mailbox: [u32; MAILBOX_MAX_WORDS],
    mailbox_len: usize,
    mailbox_pos: usize,
    completions: u32,
    pub retrace_tag: u32,
    pub reset_strobe: u32,
    pub eeprom_enable: u32,
    pub loader_3b0: u32,
    pub loader_328: u32,
    /// A video cause has been delivered and not yet acknowledged.
    video_delivered: bool,
    /// The missing interrupt destination has been logged for the video
    /// cause: it rises every frame, so it says so once.
    video_unrouted_logged: bool,
    /// Which run of the board a back end's call belongs to. A reset moves it
    /// on, so a call a back end thread made from before the reset (a retrace
    /// it completes, a frame it announces) is dropped instead of raising a
    /// cause the reset cleared. Not itself reset.
    epoch: u64,
}

impl Front {
    pub fn new(cfg: &KonaConfig) -> Self {
        Front {
            widget_control: cfg.widget_control_reset(),
            widget_control_reset: cfg.widget_control_reset(),
            identity: Identity::new(cfg),
            onewire_reg: 0,
            probe_value: 0,
            low_window_write: 0,
            int_dest_upper: 0,
            int_dest_lower: 0,
            int_vectors: 0,
            reset_arg: 0,
            int_enable: 0,
            credit_config: 0,
            int_cause: 0,
            mailbox: [0; MAILBOX_MAX_WORDS],
            mailbox_len: 0,
            mailbox_pos: 0,
            completions: 0,
            retrace_tag: 0,
            reset_strobe: 0,
            eeprom_enable: 0,
            loader_3b0: 0,
            loader_328: 0,
            video_delivered: false,
            video_unrouted_logged: false,
            epoch: 0,
        }
    }

    /// The registers, as the stream module reads and writes them. The configuration
    /// (the widget's reset value, the probe) and the epoch are the board's, not the
    /// guest's, and stay. The identity chain's state goes as the text its own save makes.
    pub fn visit<S: super::stream::Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { widget_control: _, widget_control_reset: _, identity: _, onewire_reg: _, probe_value: _, low_window_write: _, int_dest_upper: _, int_dest_lower: _, int_vectors: _, reset_arg: _, int_enable: _, credit_config: _, int_cause: _, mailbox: _, mailbox_len: _, mailbox_pos: _, completions: _, retrace_tag: _, reset_strobe: _, eeprom_enable: _, loader_3b0: _, loader_328: _, video_delivered: _, video_unrouted_logged: _, epoch: _ } = self;
        s.u32(&mut self.widget_control);
        let mut bus = self.identity.bus.save_state().to_string();
        s.string(&mut bus);
        if s.loading() {
            match bus.parse::<toml::Value>().map_err(|e| e.to_string()).and_then(|v| self.identity.bus.load_state(&v)) {
                Ok(()) => {}
                Err(e) => s.fail(&format!("the one-wire chain: {e}")),
            }
        }
        for v in [
            &mut self.onewire_reg,
            &mut self.low_window_write,
            &mut self.int_dest_upper,
            &mut self.int_dest_lower,
            &mut self.int_vectors,
            &mut self.reset_arg,
            &mut self.int_enable,
            &mut self.credit_config,
            &mut self.int_cause,
            &mut self.completions,
            &mut self.retrace_tag,
            &mut self.reset_strobe,
            &mut self.eeprom_enable,
            &mut self.loader_3b0,
            &mut self.loader_328,
        ] {
            s.u32(v);
        }
        s.u32s(&mut self.mailbox);
        s.usize(&mut self.mailbox_len);
        s.usize(&mut self.mailbox_pos);
        s.bool(&mut self.video_delivered);
        s.bool(&mut self.video_unrouted_logged);
    }

    /// Power-on state. The identity chain and the probe value are the
    /// board's, not the driver's, and stay.
    pub fn reset(&mut self) {
        self.widget_control = self.widget_control_reset;
        self.onewire_reg = 0;
        self.low_window_write = 0;
        self.int_dest_upper = 0;
        self.int_dest_lower = 0;
        self.int_vectors = 0;
        self.reset_arg = 0;
        self.int_enable = 0;
        self.credit_config = 0;
        self.int_cause = 0;
        self.mailbox = [0; MAILBOX_MAX_WORDS];
        self.mailbox_len = 0;
        self.mailbox_pos = 0;
        self.completions = 0;
        self.retrace_tag = 0;
        self.reset_strobe = 0;
        self.eeprom_enable = 0;
        self.loader_3b0 = 0;
        self.loader_328 = 0;
        self.video_delivered = false;
        self.video_unrouted_logged = false;
    }

    /// The current epoch: a ticket made now is good until the next reset.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Start a new epoch, so calls carrying an older one are dropped.
    pub fn advance_epoch(&mut self) {
        self.epoch += 1;
    }

    /// The map RAM's entry count: `N << 15` with `N` from the probe
    /// register's bits 5:3, zero meaning 8.
    pub fn mapram_count(&self) -> usize {
        let n = (self.probe_value & PROBE_MAPRAM_COUNT_MASK) >> PROBE_MAPRAM_COUNT_SHIFT;
        (if n == 0 { 8 } else { n as usize }) << 15
    }

    /// Whether mailbox words are unread.
    fn mailbox_unread(&self) -> usize {
        self.mailbox_len - self.mailbox_pos
    }

    /// Replace the queue with a reply and raise the mailbox cause.
    pub fn mailbox_reply(&mut self, words: &[u32]) {
        assert!(words.len() <= MAILBOX_MAX_WORDS);
        self.mailbox[..words.len()].copy_from_slice(words);
        self.mailbox_len = words.len();
        self.mailbox_pos = 0;
        self.completions = 0;
        if !words.is_empty() {
            self.int_cause |= CAUSE_MAILBOX;
        }
    }

    /// One unsolicited mailbox word behind any reply words still unread. A
    /// queue with no room left loses it, logged.
    pub fn mailbox_post(&mut self, word: u32) {
        let unread = self.mailbox_unread();
        if unread >= MAILBOX_MAX_WORDS {
            crate::dlog!(LogModule::Kona, "kona: mailbox full; completion word {word:#x} lost");
            return;
        }
        self.mailbox.copy_within(self.mailbox_pos..self.mailbox_len, 0);
        self.completions >>= self.mailbox_pos;
        self.completions |= 1 << unread;
        self.mailbox[unread] = word;
        self.mailbox_len = unread + 1;
        self.mailbox_pos = 0;
        self.int_cause |= CAUSE_MAILBOX;
    }

    /// The interrupt packet for the general or the geometry-engine vector,
    /// to the destination the driver programmed. Assert-only. None, logged,
    /// when no destination was programmed.
    pub fn raise_vector(&self, general: bool) -> Option<InterruptPacket> {
        let (vector, what) = if general {
            ((self.int_vectors & INT_VECTORS_GENERAL_MASK) as u8, "general")
        } else {
            ((self.int_vectors >> INT_VECTORS_GE_SHIFT) as u8, "geometry-engine")
        };
        if self.int_dest_upper == 0 && self.int_dest_lower == 0 {
            crate::dlog!(LogModule::Kona, "kona: {what} interrupt with no destination programmed; no packet sent");
            return None;
        }
        Some(InterruptPacket {
            widget: ((self.int_dest_upper >> INT_DEST_WIDGET_SHIFT) & INT_DEST_WIDGET_MASK) as u8,
            address: ((self.int_dest_upper & 0xffff) as u64) << 32 | self.int_dest_lower as u64,
            vector,
        })
    }

    /// Deliver the video cause once, if the driver enabled it.
    pub fn video_deliver(&mut self) -> Option<InterruptPacket> {
        if self.int_cause & self.int_enable & CAUSE_VIDEO != 0 && !self.video_delivered {
            // Checked here, not left to `raise_vector`, which logs every call:
            // the video cause rises each frame and logs once.
            if self.int_dest_upper == 0 && self.int_dest_lower == 0 {
                if !self.video_unrouted_logged {
                    self.video_unrouted_logged = true;
                    crate::dlog!(LogModule::Kona, "kona: video interrupt with no destination programmed; no packet sent");
                }
                return None;
            }
            let p = self.raise_vector(true);
            self.video_delivered = p.is_some();
            return p;
        }
        None
    }

    /// A retrace the back end counted: the tag register reads the request's
    /// tag, and the video cause rises.
    pub fn retrace(&mut self, tag: u32) -> Option<InterruptPacket> {
        self.retrace_tag = tag;
        self.int_cause |= CAUSE_VIDEO;
        self.video_deliver()
    }

    /// A retrace the back end counted in `epoch`: as `retrace`, unless a
    /// reset has begun a later epoch.
    pub fn retrace_in(&mut self, epoch: u64, tag: u32) -> Option<InterruptPacket> {
        if epoch != self.epoch {
            return None;
        }
        self.retrace(tag)
    }

    /// A frame the display announced in `epoch`: the video cause rises, with
    /// no tag, unless a reset has begun a later epoch.
    pub fn video_frame(&mut self, epoch: u64) -> Option<InterruptPacket> {
        if epoch != self.epoch {
            return None;
        }
        self.int_cause |= CAUSE_VIDEO;
        self.video_deliver()
    }

    /// The pipe has drained: the low-watermark cause, if the driver has
    /// enabled it. The driver enables it only while a thread waits for the
    /// pipe to drain below its threshold. Model choice: raised when the work
    /// and the words behind it are all done; when a real pipe raises it,
    /// against which level, is unknown.
    pub fn lowater(&mut self) -> Option<InterruptPacket> {
        if self.int_enable & CAUSE_LOWATER != 0 {
            self.int_cause |= CAUSE_LOWATER;
            return self.raise_vector(true);
        }
        None
    }

    /// A read of a register this file owns. `None` for any other offset.
    pub fn read(&mut self, off: u32, credit: u32) -> Option<u32> {
        Some(match off {
            WIDGET_ID => ident::widget_id(0),
            WIDGET_CONTROL => self.widget_control,
            PROBE => self.probe_value,
            ONEWIRE | NIC => self.onewire_reg,
            LOW_WINDOW_WRITE => self.low_window_write,
            INT_DEST_UPPER => self.int_dest_upper,
            INT_DEST_LOWER => self.int_dest_lower,
            INT_VECTORS => self.int_vectors,
            CREDIT => credit,
            CREDIT_CONFIG => self.credit_config,
            RESET_ARG => self.reset_arg,
            // Software never reads it (inferred); the bits it holds are those
            // of the status register.
            INT_ENABLE => self.int_enable,
            INT_CAUSE => self.int_cause,
            RETRACE_TAG => self.retrace_tag,
            MAILBOX => {
                if self.mailbox_pos < self.mailbox_len {
                    self.mailbox[self.mailbox_pos]
                } else {
                    0
                }
            }
            RESET_STROBE => self.reset_strobe,
            EEPROM_ENABLE => self.eeprom_enable,
            LOADER_3B0 => self.loader_3b0,
            LOADER_328 => self.loader_328,
            _ => return None,
        })
    }

    /// A write of a register this file owns: false for any other offset.
    /// `packet` receives a packet the write raised.
    pub fn write(&mut self, off: u32, value: u32, packet: &mut Option<InterruptPacket>) -> bool {
        match off {
            // Unknown control fields are stored without interpreted effects.
            WIDGET_CONTROL => self.widget_control = value,
            ONEWIRE | NIC => self.onewire_reg = self.identity.write(value),
            LOW_WINDOW_WRITE => self.low_window_write = value,
            INT_DEST_UPPER => self.int_dest_upper = value,
            INT_DEST_LOWER => self.int_dest_lower = value,
            INT_VECTORS => self.int_vectors = value,
            RESET_ARG => self.reset_arg = value,
            CREDIT_CONFIG => self.credit_config = value,
            INT_ENABLE => {
                self.int_enable = value & INT_ENABLE_MASK;
                *packet = self.video_deliver();
            }
            INT_CAUSE => {
                // Write-1-to-clear, the geometry-engine bit included.
                self.int_cause &= !value;
                if value & CAUSE_VIDEO != 0 {
                    self.video_delivered = false;
                }
                // A completion word is acknowledged by this write alone: the
                // driver's DMA-write wait reads the mailbox once and writes
                // nothing back before its handler clears the cause bit,
                // where the reply protocol consumes each word with a mailbox
                // write first. So clearing the bit consumes an unread
                // completion at the head; left queued, completions would
                // fill the queue.
                if value & CAUSE_MAILBOX != 0 {
                    if self.mailbox_pos < self.mailbox_len && self.completions & (1 << self.mailbox_pos) != 0 {
                        self.mailbox_pos += 1;
                    }
                    // Re-arm synchronously if a next word is queued: the
                    // model has no device-side timing to simulate the real
                    // interrupt's re-assertion latency.
                    if self.mailbox_pos < self.mailbox_len {
                        self.int_cause |= CAUSE_MAILBOX;
                    }
                }
            }
            MAILBOX => {
                // Any value (the driver always writes 0) consumes the
                // current reply word; the cause register is not touched.
                if self.mailbox_pos < self.mailbox_len {
                    self.mailbox_pos += 1;
                }
            }
            RESET_STROBE => self.reset_strobe = value,
            EEPROM_ENABLE => self.eeprom_enable = value,
            LOADER_3B0 => self.loader_3b0 = value,
            LOADER_328 => self.loader_328 = value,
            _ => return false,
        }
        true
    }

    /// The host-interface reset bit of the reset strobe: what the kernel
    /// rewrites after its 1, read, 0, read pulse. (The map RAM, which the
    /// pulse also clears, is the board's thread's.)
    pub fn hip_reset(&mut self) {
        self.int_enable = 0;
        self.reset_arg = 0;
        self.credit_config = 0;
        self.low_window_write = 0;
    }
}

/// A crosstalk interrupt packet as the driver programmed it: a write that sets
/// one bit in the destination's pending register. The processor clears the bit
/// itself; the pipe never sends a clear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct InterruptPacket {
    /// The destination widget number (bits 19:16 of the upper destination
    /// register).
    pub widget: u8,
    /// The 48-bit address within that widget.
    pub address: u64,
    /// The eight-bit vector.
    pub vector: u8,
}

/// The register file with its connections: what the processor's thread and
/// the board's thread both hold.
pub(super) struct Host {
    pub front: Mutex<Front>,
    /// Where interrupt packets go: each is one `set_line(vector, true)`, which
    /// the controller takes as setting pending bit `vector`, never as an edge.
    pub irq: Arc<dyn InterruptController>,
    /// Host physical memory, as the board's DMA reaches it.
    pub mem: Arc<dyn BusDevice>,
    /// Packets with no receiver. A packet is delivered when it names the widget
    /// the pipe's DMA goes to (the widget control register's field, which host
    /// software fills with its interrupt target); any other is counted and
    /// dropped. A multi-node machine needs a destination-to-controller map in the
    /// machine, not in the board.
    pub unrouted: AtomicU64,
    /// Words the framers hold of commands still arriving, published by the
    /// board's thread at the end of each run: the credit register's count.
    pub held_words: AtomicU32,
    /// Completion reasons wait behind an unread one: the next access to the
    /// cause register looks again.
    pub reasons_waiting: AtomicBool,
    /// A display list is running: the credit register reads full.
    pub list_running: AtomicBool,
    /// The watchdog activity latch: set when a command word or a firmware
    /// load word arrives, cleared by the driver writing the bit back. A
    /// latch read and written on every word, so not behind the register
    /// file's lock.
    pub activity: AtomicBool,
    /// Writes of the activity register's bit 0 (the kernel's once-per-video-interrupt
    /// acknowledgement of the status bit; nothing reads it back).
    pub video_acks: AtomicU64,
}

impl Host {
    /// Run `f` on the register file, then send the packet it returns, with
    /// the lock dropped.
    pub fn with_front<R>(&self, f: impl FnOnce(&mut Front) -> (R, Option<InterruptPacket>)) -> R {
        let (r, packet) = f(&mut self.front.lock());
        if let Some(p) = packet {
            self.deliver(p);
        }
        r
    }

    /// Send `p` to the interrupt controller, with no board lock held. A packet
    /// for a widget other than the host's is counted, and logged once.
    pub fn deliver(&self, p: InterruptPacket) {
        let host_widget = ((self.front.lock().widget_control >> 22) & 0xf) as u8;
        if p.widget == host_widget {
            self.irq.set_line(p.vector as u32, true);
        } else if self.unrouted.fetch_add(1, Relaxed) == 0 {
            crate::dlog!(LogModule::Kona, "kona: interrupt packet for widget {} (host widget {host_widget}), vector {:#x}: no receiver; not delivered", p.widget, p.vector);
        }
    }

    /// Read `buf.len()` bytes of host memory at `addr`; false if the bus fails.
    pub fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
        physical::read_bytes(&*self.mem, addr, buf) == BUS_OK
    }

    /// Write `data` to host memory at `addr`; false if the bus fails.
    pub fn dma_write(&self, addr: u64, data: &[u8]) -> bool {
        physical::write_bytes(&*self.mem, addr, data) == BUS_OK
    }

    /// The general vector's packet for a mailbox post.
    pub fn post_mailbox(&self, word: u32) {
        self.with_front(|f| {
            f.mailbox_post(word);
            ((), f.raise_vector(true))
        });
    }

    /// Queue a reply and raise the mailbox cause. No vector: the driver
    /// polls for the replies the PROM and the textport protocol use.
    pub fn mailbox_reply(&self, words: &[u32]) {
        self.front.lock().mailbox_reply(words);
    }
}
