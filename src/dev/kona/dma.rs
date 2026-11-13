//! The pipe's DMA and display lists.
//!
//! A DMA packet is the selector 0x1eb and six words:
//!
//! 1. `io >> 12`
//! 2. `(io & 0xfff) | 0x9000`
//! 3. the descriptor's total byte count
//! 4. its per-line byte count
//! 5. its gap, the bytes between one line's end and the next line
//! 6. `0x100` when the sender wants a completion, else 0
//!
//! where `io` is the device address of the pinned buffer; the kick register
//! takes literals around the packet as a bracket and is not a word count.
//! Before a packet, host software sends a record, 0x1c8 with
//! a pipe-memory address and a byte count, when a display list is copied into
//! pipe memory. A record says where the next packet's bytes go, so the engine
//! holds it and places that packet's transfer in the list memory. With no
//! record the packet's bytes feed whatever the back end is waiting for
//! (an image, a texture, a bitmap); with neither, the packet is logged
//! without transferring bytes or posting a completion.
//!
//! The completion's mailbox word comes from the packet's word 6; that the
//! pipe posts the packet's own last word is inferred.
//!
//! Display lists live in pipe memory (the list memory) or in the DMA pool
//! the driver registered with 0x1ed. An invocation (0x16d, 0x16e, 0x170,
//! 0x171) fetches (offset, word count) ranges and feeds the words to the
//! command page's decoder, as a compiled list holds the words immediate mode
//! writes there. Pipe-memory offsets count from the start of the context's
//! display-list region, which the kernel's 0x1e4 packets place (inferred).
//!
//! Approximated: a pipe-memory address is a byte offset into the list memory;
//! how the pipe places the display-list region the kernel reserves is
//! unknown, so the list memory is that region, from offset 0. The packet
//! names no destination widget; the machine's DMA handle chooses one.
//!
//! Ported from hw/misc/sgi-kona-dma.c in jamesbraid/qemu-sgi-origami.

use super::defs::*;
use super::front::Host;
use super::back::KonaBackend;
use super::job::{Job, Reply};
use super::pipe::{Core, Run};
use crate::devlog::LogModule;

pub(super) struct Dma {
    /// The display-list memory, big-endian words as the guest wrote them.
    pub list: Box<[u8]>,
    record_pending: bool,
    record_dest: u32,
    record_bytes: u32,
    pool: Option<(u64, u32)>,
    /// Where the current context's display-list region starts in the list
    /// memory, in bytes.
    pub list_base: u32,
    /// A list is running: an invocation from inside it fails.
    in_list: bool,
    /// The words of the list being invoked, and the bytes a fetch reads,
    /// kept between invocations.
    words: Vec<u32>,
    bytes: Vec<u8>,
}

impl Dma {
    pub fn new() -> Self {
        Dma { list: vec![0u8; LIST_DEFAULT].into_boxed_slice(), record_pending: false, record_dest: 0, record_bytes: 0, pool: None, list_base: 0, in_list: false, words: Vec::new(), bytes: Vec::new() }
    }

    /// The display-list memory and the registration the guest made, as the stream
    /// module reads and writes them (a list is not running when it is saved).
    /// Whether a display list is running.
    pub fn running(&self) -> bool {
        self.in_list
    }

    pub fn visit<S: super::stream::Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { list: _, record_pending: _, record_dest: _, record_bytes: _, pool: _, list_base: _, in_list: _, words: _, bytes: _ } = self;
        s.bytes(&mut self.list);
        s.bool(&mut self.record_pending);
        s.u32(&mut self.record_dest);
        s.u32(&mut self.record_bytes);
        s.opt(&mut self.pool, |s, p| {
            s.u64(&mut p.0);
            s.u32(&mut p.1);
        });
        s.u32(&mut self.list_base);
        if s.loading() {
            self.in_list = false;
        }
    }

    pub fn reset(&mut self) {
        // Fresh, zeroed pages: nothing a list left before the reset can run.
        self.list = vec![0u8; LIST_DEFAULT].into_boxed_slice();
        self.record_pending = false;
        self.record_dest = 0;
        self.record_bytes = 0;
        self.pool = None;
        self.list_base = 0;
        self.in_list = false;
    }
}

impl Core {
    /// A command handed to the engine: the packet, the list-write record,
    /// the pool registration or a display-list packet.
    pub(super) fn dma_command(&mut self, h: &Host, be: &mut KonaBackend, selector: u32, a: &[u32]) {
        match selector {
            START_INPUT_DMA => self.dma_packet(h, be, a),
            DESCRIBE_DMA_PAGE_POOL => self.dma_register_pool(a),
            EXECUTE_LIST_BY_LIST_DMA => self.dma_invoke(h, be, false, &a[..2]),
            EXECUTE_LIST_BY_INPUT_DMA => self.dma_invoke(h, be, true, &a[..2]),
            EXECUTE_LIST_BY_LIST_DMA_REPEATED | EXECUTE_LIST_BY_INPUT_DMA_REPEATED => {
                // 0x170 n or 0x171 n: n pairs follow. An n outside 1 to the
                // packet's limit is logged and opens nothing, so the pairs
                // are read as fresh words.
                let max = if selector == EXECUTE_LIST_BY_LIST_DMA_REPEATED { LIST_CALLS_MAX } else { LIST_CALLS_POOL_MAX };
                if a[0] == 0 || a[0] > max {
                    crate::dlog!(LogModule::Kona, "kona: list invocation {selector:#x} names {} ranges, not 1 to {max}", a[0]);
                    return;
                }
                self.input.run = Run::Calls { token: selector, need: 2 * a[0] as usize, words: Vec::new() };
            }
            COPY_USER_LIST_BY_LIST_DMA => self.dma_pull(h, a),
            MOVE_LIST_BY_LIST_DMA => self.dma_move(a),
            COPY_LIST_BY_INPUT_DMA_THEN_LIST_DMA => {
                if self.dma.record_pending {
                    crate::dlog!(LogModule::Kona, "kona: dma list write {:#x} bytes at {:#x} replaces one no packet took", a[1], a[0]);
                }
                self.dma.record_pending = true;
                self.dma.record_dest = a[0].saturating_add(self.dma.list_base);
                self.dma.record_bytes = a[1];
            }
            _ => unreachable!("selector {selector:#x} is not a DMA command"),
        }
    }

    /// The packet, words 1 to 6. Every size is the guest's: the per-line
    /// count must be whole words and divide the total, and the buffer the
    /// lines and gaps span must fit the model's bound, or the packet is
    /// refused with its completion withheld.
    fn dma_packet(&mut self, h: &Host, be: &mut KonaBackend, w: &[u32]) {
        let record = self.dma.record_pending;
        self.dma.record_pending = false;
        let done = match w[5] {
            DMA_DONE_WORD => true,
            0 => false,
            other => {
                crate::dlog!(LogModule::Kona, "kona: dma word 6 is {other:#x}, not 0 or {DMA_DONE_WORD:#x}; no completion posted");
                false
            }
        };
        if w[1] & !0xfff != DMA_ADDR_TAG {
            crate::dlog!(LogModule::Kona, "kona: dma word 2 ({:#010x}) carries tag {:#x}, expected {DMA_ADDR_TAG:#x}; refusing", w[1], w[1] & !0xfff);
            return;
        }
        // A high word with bit 28 clear is `io >> 12` of a device address the
        // map RAM translates (inferred); with bit 28 set it names an absolute
        // host frame, the form the kernel's own pages use.
        let physical = w[0] & PAGE_WORD_PHYSICAL != 0;
        let io = if physical { ((w[0] & PAGE_WORD_FRAME) as u64) << 12 } else { (w[0] as u64) << 12 } | (w[1] & 0xfff) as u64;
        let (total, per_line, gap) = (w[2], w[3], w[4]);
        if per_line == 0 || total == 0 || total % per_line != 0 || (record && per_line % 4 != 0) {
            crate::dlog!(LogModule::Kona, "kona: dma of {total:#x} bytes in lines of {per_line:#x} is not whole lines of whole words; refusing");
            return;
        }
        let lines = total / per_line;
        let pitch = per_line as u64 + gap as u64;
        let span = (io & (DMA_PAGE_SIZE - 1)) + (lines as u64 - 1) * pitch + per_line as u64;
        if span > (64 << 20) {
            crate::dlog!(LogModule::Kona, "kona: dma of {lines} lines at a pitch of {pitch:#x} spans more than 64 MiB; refusing");
            return;
        }
        let completed = if record {
            self.dma_to_list(h, io, physical, lines, total)
        } else {
            self.dma_fault = false;
            match self.job(h, be, Job::DmaIn { io, physical, lines, per_line, pitch }) {
                Reply::Unclaimed => {
                    crate::dlog!(LogModule::Kona, "kona: dma packet with no list record and nothing waiting for pixels; no completion");
                    false
                }
                _ => !self.dma_fault,
            }
        };
        if completed && done {
            h.post_mailbox(DMA_DONE_WORD);
        }
    }

    /// The list end of a packet: its bytes, one line of them, at the pending
    /// record's pipe-memory address. The kernel's one DMA-write sender sends
    /// the record's count as the packet's total and per-line count with no
    /// gap, so anything else is refused rather than guessing how a pipe joins
    /// lines or reconciles counts.
    fn dma_to_list(&mut self, h: &Host, io: u64, physical: bool, lines: u32, total: u32) -> bool {
        let (dest, bytes) = (self.dma.record_dest as usize, self.dma.record_bytes);
        if lines != 1 || bytes != total {
            crate::dlog!(LogModule::Kona, "kona: dma of {lines} lines, {total:#x} bytes, for a list write of {bytes:#x} bytes; refusing");
            return false;
        }
        let len = self.dma.list.len();
        if dest % 4 != 0 || total % 4 != 0 || dest > len || total as usize > len - dest {
            crate::dlog!(
                LogModule::Kona,
                "kona: {total:#x} bytes at pipe-memory {dest:#x} are not whole words inside the {len:#x}-byte list memory; refusing"
            );
            return false;
        }
        let mut buf = vec![0u8; total as usize];
        let read = if physical { h.dma_read(io, &mut buf) } else { self.device_read(h, io, &mut buf) };
        if !read {
            crate::dlog!(LogModule::Kona, "kona: host read of {total:#x} bytes at {io:#x} failed; completion withheld");
            return false;
        }
        self.dma.list[dest..dest + total as usize].copy_from_slice(&buf);
        true
    }

    /// 0x1ed page tagged-offset length: the DMA pool the kernel pinned, which
    /// list invocations 0x16e and 0x171 and the pull 0x167 read by byte
    /// offset. The kernel's pool setup requires base and length to be
    /// multiples of the pool page, and its pool update sends a length of 1
    /// for a context with no pool, so a range that is not whole pages leaves
    /// no pool registered, silently. A word 2 without the 0x9000 tag, or a
    /// pool longer than the model's bound, is logged as well. The pool is one
    /// range of the device address space; each page goes through its map RAM
    /// entry when the pool is read, which is the model's choice: nothing shows
    /// when the pipe reads the entries.
    fn dma_register_pool(&mut self, a: &[u32]) {
        let base = (a[0] as u64) << 12 | (a[1] & 0xfff) as u64;
        self.dma.pool = None;
        if a[1] & !0xfff != DMA_ADDR_TAG || a[2] > DMA_POOL_MAX {
            crate::dlog!(
                LogModule::Kona,
                "kona: dma pool {:#x} {:#x} of {:#x} bytes is not tagged, or is longer than {DMA_POOL_MAX:#x} bytes; no pool",
                a[0], a[1], a[2]
            );
        } else if base.is_multiple_of(DMA_POOL_PAGE as u64) && a[2] != 0 && a[2].is_multiple_of(DMA_POOL_PAGE) {
            self.dma.pool = Some((base, a[2]));
        }
    }

    /// Fetch `nwords` words at byte `offset` of the pool or of the list
    /// memory (relative to the region base).
    fn fetch_words(&mut self, h: &Host, pool: bool, offset: u32, nwords: u32, out: &mut Vec<u32>) -> bool {
        let mut buf = std::mem::take(&mut self.dma.bytes);
        let ok = self.fetch_into(h, pool, offset, nwords, &mut buf);
        if ok {
            out.extend(buf.chunks_exact(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())));
        }
        self.dma.bytes = buf;
        ok
    }

    fn fetch_into(&self, h: &Host, pool: bool, offset: u32, nwords: u32, buf: &mut Vec<u8>) -> bool {
        let bytes = nwords as usize * 4;
        buf.clear();
        buf.resize(bytes, 0);
        if pool {
            let Some((base, len)) = self.dma.pool else {
                crate::dlog!(LogModule::Kona, "kona: list invoked from the pool with no pool registered");
                return false;
            };
            if offset as u64 + bytes as u64 > len as u64 {
                crate::dlog!(LogModule::Kona, "kona: pool range {offset:#x}+{bytes:#x} leaves the {len:#x}-byte pool");
                return false;
            }
            if !self.device_read(h, base + offset as u64, buf) {
                return false;
            }
        } else {
            let at = self.dma.list_base as u64 + offset as u64;
            if at % 4 != 0 || at + bytes as u64 > self.dma.list.len() as u64 {
                crate::dlog!(LogModule::Kona, "kona: list range {at:#x}+{bytes:#x} leaves the list memory");
                return false;
            }
            buf.copy_from_slice(&self.dma.list[at as usize..at as usize + bytes]);
        }
        true
    }

    /// Run n (offset, word count) ranges as one list: from the pool, or from
    /// pipe memory. The words are the list's as of this invocation. A list
    /// that invokes a list finds the engine busy and fails; the outer list
    /// goes on. What the driver observes at a list's end is unknown, so
    /// nothing is posted.
    pub(super) fn dma_invoke(&mut self, h: &Host, be: &mut KonaBackend, pool: bool, pairs: &[u32]) {
        if self.dma.in_list {
            crate::dlog!(LogModule::Kona, "kona: a list invoked from inside a list fails");
            return;
        }
        let mut words = std::mem::take(&mut self.dma.words);
        words.clear();
        for p in pairs.chunks_exact(2) {
            if p[1] > LIST_MAX_WORDS || words.len() as u64 + p[1] as u64 > LIST_MAX_WORDS as u64 {
                crate::dlog!(LogModule::Kona, "kona: list of {} words is longer than the display-list class; refusing", p[1]);
                self.dma.words = words;
                return;
            }
            if !self.fetch_words(h, pool, p[0], p[1], &mut words) {
                self.dma.words = words;
                return;
            }
        }
        self.dma.in_list = true;
        h.list_running.store(true, std::sync::atomic::Ordering::SeqCst);
        for &w in &words {
            self.command_word(h, be, w);
        }
        self.dma.words = words;
        h.list_running.store(false, std::sync::atomic::Ordering::SeqCst);
        self.dma.in_list = false;
    }

    /// 0x167 dest src bytes 0: the client library's copy of a list block from
    /// the pool into pipe memory. What its fourth word does is unknown; the
    /// one sender sends 0.
    fn dma_pull(&mut self, h: &Host, a: &[u32]) {
        if self.dma.pool.is_none() {
            crate::dlog!(LogModule::Kona, "kona: list pull with no pool registered");
            return;
        }
        let (dest, bytes) = (self.dma.list_base as u64 + a[0] as u64, a[2] as usize);
        if dest % 4 != 0 || bytes % 4 != 0 || dest + bytes as u64 > self.dma.list.len() as u64 {
            crate::dlog!(LogModule::Kona, "kona: {bytes:#x} bytes at pipe-memory {dest:#x} are not whole words inside the list memory; refusing");
            return;
        }
        let mut words = Vec::new();
        if !self.fetch_words(h, true, a[1], (bytes / 4) as u32, &mut words) {
            return;
        }
        for (i, w) in words.iter().enumerate() {
            self.dma.list[dest as usize + 4 * i..][..4].copy_from_slice(&w.to_be_bytes());
        }
    }

    /// 0x16f dest src length: the client library's pipe-memory block
    /// allocator's move, within the list memory. That the length counts bytes
    /// is inferred from its caller's other sizes; overlapping ranges move as
    /// `memmove` does.
    fn dma_move(&mut self, a: &[u32]) {
        let base = self.dma.list_base as u64;
        let (dest, src, len) = (base + a[0] as u64, base + a[1] as u64, a[2] as u64);
        let total = self.dma.list.len() as u64;
        if dest + len > total || src + len > total {
            crate::dlog!(LogModule::Kona, "kona: list move of {len:#x} bytes from {src:#x} to {dest:#x} leaves the list memory; dropped");
            return;
        }
        self.dma.list.copy_within(src as usize..(src + len) as usize, dest as usize);
    }

    /// 0x1e4 class key offset size: the kernel's placement of one of the
    /// current context's pipe-memory regions, offset and size in granules.
    /// The library's display-list offsets count from the start of its
    /// display-list allocation, class 1 key 0, so two contexts each using
    /// offset 0 must not meet in pipe memory. The model therefore adds that
    /// region's offset to every list-memory address. Inferred: that the pipe
    /// relocates the offsets by the region the kernel placed. A freed region
    /// (size 0) leaves no base.
    pub(super) fn dma_list_region(&mut self, a: &[u32]) {
        if a[0] != 1 || a[1] != 0 {
            return;
        }
        if a[3] == 0 || a[2] > LIST_MAX_WORDS * 4 / LIST_GRANULE {
            self.dma.list_base = 0;
            return;
        }
        self.dma.list_base = a[2] * LIST_GRANULE;
    }
}

impl Core {
    /// For tests and the monitor: the list memory word at byte `at`.
    #[cfg(test)]
    pub(super) fn list_word(&self, at: usize) -> u32 {
        u32::from_be_bytes(self.dma.list[at..at + 4].try_into().unwrap())
    }
}
