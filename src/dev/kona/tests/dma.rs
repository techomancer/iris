//! DMA packets, display-list memory and list invocation. Ported from
//! tests/qtest/sn0-origin2000-kona-dma-test.c: the cases about the list path
//! and the packet's framing. Those that place pixels in a plane or read them
//! back are the back end's.
//!
//! A display list holds the words immediate mode writes to the command page,
//! so a list of textport commands shows what it fed the decoder by the jobs
//! the back end is handed.

use super::*;

const DMA_FIFO_PUSH: u64 = 0x80007c;
const DMA_FIFO_KICK: u64 = 0xc00108;
const INT_CAUSE: u64 = 0x800238;
const MAILBOX: u64 = 0x800240;
const INT_DEST_UPPER: u64 = 0x34;
const INT_DEST_LOWER: u64 = 0x3c;
const INT_VECTORS: u64 = 0x5c;
const CAUSE_MAILBOX: u32 = 0x10;

const LIST_ROW: u32 = 4096;
const LIST_BYTES: u32 = 15 << 20;
const SRC: u64 = 0x40_0000;
const POOL_DEVICE: u64 = 0x34000;
const POOL_HOST: u64 = 0x80_0000;
const POOL_LEN: u32 = 0x8000;

/// A colour map load, a colour select and a point: the chain every glyph
/// stroke uses, thirteen words.
const LIST: [u32; 13] = [1, 3, 5, 0x10, 0x20, 0x30, 1, 2, 5, 1, 6, 7, 8];
const LIST_JOBS: [&str; 3] = [
    "Textport 0x3 [5, 10, 20, 30] packed=0x0 handled=false",
    "Textport 0x2 [5] packed=0x0 handled=false",
    "Textport 0x6 [7, 8] packed=0x0 handled=false",
];

impl Fixture {
    fn dma(&self, words: &[u32]) {
        for w in words {
            self.w(DMA_FIFO_PUSH, *w);
        }
    }

    fn list_write(&self, dest: u32, bytes: u32) {
        self.dma(&[0x1c8, dest, bytes]);
    }

    fn packet(&self, io: u64, total: u32, per_line: u32, gap: u32, done: u32) {
        self.dma(&[0x1eb, 0x1000_0000 | (io >> 12) as u32, 0x9000 | (io & 0xfff) as u32, total, per_line, gap, done]);
    }

    fn list_word(&self, at: usize) -> u32 {
        self.kona.shared.pipe.lock().core.list_word(at)
    }

    /// Map `len` bytes of the device address space at `device` onto `host`.
    fn map_range(&self, device: u64, host: u64, len: u64) {
        for off in (0..len).step_by(0x4000) {
            self.map((device + off) >> 14, host + off);
        }
    }

    /// The pool the kernel pinned, and its registration.
    fn pool(&self) {
        self.map_range(POOL_DEVICE, POOL_HOST, POOL_LEN as u64);
        self.cmd(&[0x1ed, (POOL_DEVICE >> 12) as u32, 0x9000 | (POOL_DEVICE & 0xfff) as u32, POOL_LEN]);
    }

    /// The decoder's jobs: the textport commands a list fed it.
    fn textport_jobs(&self) -> Vec<String> {
        self.jobs().into_iter().filter(|j| j.starts_with("Textport")).collect()
    }

    /// Put `words` into list memory at byte `at`, as a list write would.
    fn put_list(&self, at: u32, words: &[u32]) {
        self.host.put_words(SRC, words);
        self.list_write(at, 4 * words.len() as u32);
        self.packet(SRC, 4 * words.len() as u32, 4 * words.len() as u32, 0, 0);
    }
}

/// The driver's display-list copy: a
/// 0x1c8 record naming pipe-memory address 0xffc and 4,104 bytes, then a packet
/// of that many bytes in one line. The bytes land in the list memory from that
/// address, and the packet completes once with the mailbox word its word 6 asks
/// for.
#[test]
fn list_write_lands_in_list_memory() {
    let f = Fixture::hooked();
    let words: Vec<u32> = (0..1026).map(|i| 0x16d0_0000 + i).collect();
    f.host.put_words(SRC, &words);
    f.list_write(0xffc, 4104);
    f.packet(SRC, 4104, 4104, 0, 0x100);
    assert_eq!(f.list_word(0xff8), 0);
    assert_eq!(f.list_word(0xffc), words[0]);
    assert_eq!(f.list_word(LIST_ROW as usize), words[1]);
    assert_eq!(f.list_word(2 * LIST_ROW as usize - 4), words[1024]);
    assert_eq!(f.list_word(2 * LIST_ROW as usize), words[1025]);
    assert_eq!(f.list_word(2 * LIST_ROW as usize + 4), 0);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, CAUSE_MAILBOX);
    assert_eq!(f.r(MAILBOX), 0x100);
    // The record is spent: the next packet is not a list write.
    f.take_jobs();
    f.packet(SRC, 16, 16, 0, 0);
    assert_eq!(f.take_jobs(), ["dma-in 0x400000 1 0x10 0x10 physical"]);
    assert_eq!(f.list_word(0xffc), words[0]);
}

/// Lists sit past the first MiB of the
/// display-list class: a list write there lands, as anywhere inside the class's
/// 15 MiB.
#[test]
fn list_write_past_the_first_mib() {
    let f = Fixture::new();
    f.host.put_words(SRC, &[0xa, 0xb, 0xc, 0xd]);
    f.list_write(0x243000, 16);
    f.packet(SRC, 16, 16, 0, 0x100);
    assert_eq!(f.list_word(0x243000), 0xa);
    assert_eq!(f.list_word(0x24300c), 0xd);
    assert_eq!(f.r(MAILBOX), 0x100);
    // The last word of the class.
    f.list_write(LIST_BYTES - 4, 4);
    f.packet(SRC, 4, 4, 0, 0);
    assert_eq!(f.list_word(LIST_BYTES as usize - 4), 0xa);
}

/// A list write that would run past the
/// list memory's end, or whose record does not match the packet's byte count,
/// is refused, completes nothing, and writes nothing.
#[test]
fn list_write_outside_the_memory_is_refused() {
    let f = Fixture::new();
    f.host.put_words(SRC, &[1, 2, 3, 4]);
    f.list_write(LIST_BYTES - 8, 16);
    f.packet(SRC, 16, 16, 0, 0x100);
    assert_eq!(f.list_word(LIST_BYTES as usize - 8), 0);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0);
    // A record that does not match the packet's byte count.
    f.list_write(0x100, 8);
    f.packet(SRC, 16, 16, 0, 0x100);
    assert_eq!(f.list_word(0x100), 0);
    // Two lines for a list write.
    f.list_write(0x100, 16);
    f.packet(SRC, 16, 8, 0, 0x100);
    assert_eq!(f.list_word(0x100), 0);
    // An address that is not whole words.
    f.list_write(0x102, 16);
    f.packet(SRC, 16, 16, 0, 0x100);
    assert_eq!(f.list_word(0x100), 0);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0);
}

/// Words that are not a selector the port
/// frames are dropped one at a time, and the kick's values frame nothing, so a
/// packet after them completes normally.
#[test]
fn stray_words_resynchronize() {
    let f = Fixture::new();
    f.host.put_words(SRC, &[7, 8, 9, 10]);
    f.dma(&[0xdead_beef, 0x1234, 0]);
    f.w(DMA_FIFO_KICK, 0x30000);
    f.w(DMA_FIFO_KICK, 0x10000);
    f.list_write(0x40, 16);
    f.w(DMA_FIFO_KICK, 0);
    f.packet(SRC, 16, 16, 0, 0x100);
    assert_eq!(f.list_word(0x40), 7);
    assert_eq!(f.r(MAILBOX), 0x100);
}

/// A packet pushed in two
/// parts moves nothing and completes nothing until its last word arrives.
#[test]
fn a_partial_packet_holds_state_across_pushes() {
    let f = Fixture::new();
    f.host.put_words(SRC, &[7, 8, 9, 10]);
    f.list_write(0x40, 16);
    f.dma(&[0x1eb, 0x1000_0000 | (SRC >> 12) as u32, 0x9000, 16]);
    assert_eq!(f.list_word(0x40), 0);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0);
    assert_eq!(f.r(0x800208), 4, "four words held");
    f.dma(&[16, 0, 0x100]);
    assert_eq!(f.list_word(0x40), 7);
    assert_eq!(f.r(MAILBOX), 0x100);
    assert_eq!(f.r(0x800208), 0);
}

/// Software is expected to send the
/// kernel's packets on the +0xc10000 command page, so the record and the
/// packet reach the engine there as plain words too, token first, with the
/// kick literals around the packet.
#[test]
fn the_command_page_reaches_the_engine() {
    let f = Fixture::new();
    f.host.put_words(SRC, &[7, 8, 9, 10]);
    f.w(KICK, 0x30000);
    f.cmd(&[0x1c8, 0x40, 16]);
    f.w(KICK, 0x10000);
    f.cmd(&[0x1eb, 0x1000_0000 | (SRC >> 12) as u32, 0x9000, 16, 16, 0, 0x100]);
    f.w(KICK, 0);
    assert_eq!(f.list_word(0x40), 7);
    assert_eq!(f.list_word(0x4c), 10);
    assert_eq!(f.r(MAILBOX), 0x100);
}

/// Word 6 decides the completion:
/// 0x100 posts the mailbox word, 0 posts nothing. Both packets still transfer.
#[test]
fn completion_follows_word_6() {
    let f = Fixture::new();
    f.host.put_words(SRC, &[1, 2, 3, 4]);
    f.list_write(0x80, 16);
    f.packet(SRC, 16, 16, 0, 0);
    assert_eq!(f.list_word(0x80), 1);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0);
    f.list_write(0x90, 16);
    f.packet(SRC, 16, 16, 0, 0x100);
    assert_eq!(f.list_word(0x90), 1);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, CAUSE_MAILBOX);
    // Any other value asks for no completion.
    f.w(MAILBOX, 0);
    f.w(INT_CAUSE, CAUSE_MAILBOX);
    f.list_write(0xa0, 16);
    f.packet(SRC, 16, 16, 0, 0x55);
    assert_eq!(f.list_word(0xa0), 1);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0);
}

/// A packet whose word 6 is 0x100 completes
/// as the driver's interrupt arming expects: the mailbox holds 0x100 under
/// cause bit 0x10, one interrupt goes out on the general vector, and the
/// read-clear-acknowledge sequence clears both. A packet whose word 6 is 0
/// posts nothing, and neither does a display list's completion.
#[test]
fn a_completion_posts_the_mailbox_and_one_interrupt() {
    let f = Fixture::new();
    f.w(INT_DEST_UPPER, 0x98000);
    f.w(INT_DEST_LOWER, 0x180_0090);
    f.w(INT_VECTORS, 0x3e2210);
    f.host.put_words(SRC, &[1, 2, 3, 4]);
    f.list_write(0x80, 16);
    f.packet(SRC, 16, 16, 0, 0x100);
    assert_eq!(f.host.sent().len(), 1);
    assert_eq!(f.host.sent()[0].vector, 0x10);
    assert_eq!(f.r(MAILBOX), 0x100);
    f.w(MAILBOX, 0);
    f.w(INT_CAUSE, CAUSE_MAILBOX);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0);
    assert_eq!(f.r(MAILBOX), 0);
    // A list's completion posts nothing.
    f.put_list(0, &LIST);
    f.cmd(&[0x16d, 0, 13]);
    assert_eq!(f.host.sent().len(), 1);
}

/// Word 2 must carry the fixed 0x9000 tag: a
/// packet whose word 2 carries another is refused, moves nothing and completes
/// nothing, withholding the word the sender waits for. The port stays in step
/// for the next packet.
#[test]
fn a_tag_mismatch_is_refused() {
    let f = Fixture::new();
    f.host.put_words(SRC, &[1, 2, 3, 4]);
    f.list_write(0x80, 16);
    f.dma(&[0x1eb, 0x1000_0000 | (SRC >> 12) as u32, 0x8000, 16, 16, 0, 0x100]);
    assert_eq!(f.list_word(0x80), 0);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0);
    // The record was spent by the refused packet; send both again.
    f.list_write(0x80, 16);
    f.packet(SRC, 16, 16, 0, 0x100);
    assert_eq!(f.list_word(0x80), 1);
    assert_eq!(f.r(MAILBOX), 0x100);
}

/// A descriptor that is not whole lines of
/// whole words is refused: a zero per-line count, a per-line count that is not
/// whole words, and a total that is not a whole number of lines.
#[test]
fn shape_constraints_are_refused() {
    let f = Fixture::hooked();
    f.host.put_words(SRC, &[1, 2, 3, 4]);
    for (total, per_line) in [(16, 0), (16, 6), (16, 12), (0, 4)] {
        f.list_write(0x80, total);
        f.packet(SRC, total, per_line, 0, 0x100);
        assert_eq!(f.list_word(0x80), 0, "total {total:#x} in lines of {per_line:#x}");
        assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0);
    }
    f.packet(SRC, 16, 16, 0, 0x100);
    assert_eq!(f.r(MAILBOX), 0x100, "the port is in step");
}

/// A host DMA that cannot complete is reported the
/// one way the model can without inventing a cause bit: its completion is
/// withheld, so the driver's timeout fires. No cause bit and no mailbox word
/// appear, and the pipe still completes a good packet.
#[test]
fn a_failed_host_dma_withholds_its_completion() {
    let f = Fixture::hooked();
    f.list_write(0x80, 16);
    f.packet(0x7fff_f000, 16, 16, 0, 0x100);
    assert_eq!(f.list_word(0x80), 0);
    assert_eq!(f.r(INT_CAUSE), 0);
    assert_eq!(f.r(MAILBOX), 0);
    f.host.put_words(SRC, &[1, 2, 3, 4]);
    f.list_write(0x80, 16);
    f.packet(SRC, 16, 16, 0, 0x100);
    assert_eq!(f.r(MAILBOX), 0x100);
}

/// A packet with no list record is the back end's: lines and gaps reach it as
/// the descriptor names them, and it completes as the packet asks unless
/// nothing was waiting for it.
#[test]
fn a_packet_without_a_record_goes_to_the_back_end() {
    let f = Fixture::hooked();
    f.packet(SRC + 4, 16, 8, 8, 0x100);
    assert_eq!(f.take_jobs(), ["dma-in 0x400004 2 0x8 0x10 physical"]);
    assert_eq!(f.r(MAILBOX), 0x100);

    let rec = Recorder { unclaimed: true, ..Default::default() };
    let g = Fixture::with(KonaConfig::default(), rec);
    g.packet(SRC, 16, 16, 0, 0x100);
    assert_eq!(g.r(INT_CAUSE) & CAUSE_MAILBOX, 0, "nothing was waiting: no completion");
}

/// 0x1ed registers a pool; 0x171 invokes
/// one list from it as two ranges, the first six words at pool offset 0x100 and
/// the other seven at 0x800. Every word reaches the command page's decoder in
/// order, across the break between ranges.
#[test]
fn a_list_from_the_pool_reaches_the_decoder() {
    let f = Fixture::hooked();
    f.pool();
    f.host.put_words(POOL_HOST + 0x100, &LIST[..6]);
    f.host.put_words(POOL_HOST + 0x800, &LIST[6..]);
    f.cmd(&[0x171, 2, 0x100, 6, 0x800, 7]);
    assert_eq!(f.textport_jobs(), LIST_JOBS);
    // One range, 0x16e.
    f.take_jobs();
    f.host.put_words(POOL_HOST + 0x2000, &LIST);
    f.cmd(&[0x16e, 0x2000, 13]);
    assert_eq!(f.textport_jobs(), LIST_JOBS);
}

/// The pool's pages go through the map RAM
/// when the pool is read: a page whose entry maps nothing fails the whole read.
#[test]
fn the_pool_is_read_through_the_map_ram() {
    let f = Fixture::hooked();
    f.pool();
    f.host.put_words(POOL_HOST + 0x100, &LIST);
    f.w(0x100000 + 4 * (POOL_DEVICE >> 14), 0);
    f.cmd(&[0x16e, 0x100, 13]);
    assert!(f.textport_jobs().is_empty());
    // Mapped again, it reads.
    f.map(POOL_DEVICE >> 14, POOL_HOST);
    f.cmd(&[0x16e, 0x100, 13]);
    assert_eq!(f.textport_jobs(), LIST_JOBS);
    // The pool is not a single page: a range into the second one.
    f.take_jobs();
    f.host.put_words(POOL_HOST + 0x4000 + 0x10, &LIST);
    f.cmd(&[0x16e, 0x4010, 13]);
    assert_eq!(f.textport_jobs(), LIST_JOBS);
}

/// The pipe-memory path: 0x167
/// pulls the list from the pool into pipe memory at 0x2000, and 0x170 invokes
/// it from there as two ranges. 0x16f then moves the last four words, the point,
/// to 0x3000, and 0x16d invokes them alone, redrawing the point.
#[test]
fn a_list_from_pipe_memory_reaches_the_decoder() {
    let f = Fixture::hooked();
    f.pool();
    f.host.put_words(POOL_HOST + 0x100, &LIST);
    f.cmd(&[0x167, 0x2000, 0x100, 52, 0]);
    assert_eq!(f.list_word(0x2000), 1);
    assert_eq!(f.list_word(0x2000 + 48), 8);
    f.cmd(&[0x170, 2, 0x2000, 6, 0x2018, 7]);
    assert_eq!(f.textport_jobs(), LIST_JOBS);
    f.take_jobs();
    f.cmd(&[0x16f, 0x3000, 0x2000 + 36, 16]);
    assert_eq!(f.list_word(0x3000), 1);
    assert_eq!(f.list_word(0x300c), 8);
    f.cmd(&[0x16d, 0x3000, 4]);
    assert_eq!(f.textport_jobs(), [LIST_JOBS[2]]);
    // An overlapping move goes as memmove does.
    f.cmd(&[0x16f, 0x3004, 0x3000, 16]);
    assert_eq!(f.list_word(0x3004), 1);
    assert_eq!(f.list_word(0x3010), 8);
    // A move that leaves the memory is dropped.
    f.cmd(&[0x16f, LIST_BYTES - 8, 0x3000, 16]);
    assert_eq!(f.list_word(LIST_BYTES as usize - 8), 0);
}

/// The library's display-list
/// offsets count from the start of the context's display-list region, which the
/// kernel's 0x1e4 packets place at the model's 0x400-byte granule: two contexts
/// each using offset 0 do not meet in pipe memory. A freed region leaves no
/// base.
#[test]
fn list_offsets_follow_the_context_region() {
    let f = Fixture::hooked();
    f.pool();
    f.host.put_words(POOL_HOST, &LIST);
    // Class 1 key 0 at granule 4: a base of 0x1000.
    f.cmd(&[0x1e4, 1, 0, 4, 8]);
    f.cmd(&[0x167, 0, 0, 52, 0]);
    assert_eq!(f.list_word(0x1000), 1);
    assert_eq!(f.list_word(0), 0);
    f.cmd(&[0x16d, 0, 13]);
    assert_eq!(f.textport_jobs(), LIST_JOBS);
    // Another class or key places nothing.
    f.cmd(&[0x1e4, 2, 0, 9, 8]);
    f.cmd(&[0x1e4, 1, 3, 9, 8]);
    f.take_jobs();
    f.cmd(&[0x16d, 0, 13]);
    assert_eq!(f.textport_jobs(), LIST_JOBS);
    // Freed: the base is gone.
    f.cmd(&[0x1e4, 1, 0, 4, 0]);
    f.take_jobs();
    f.cmd(&[0x16d, 0, 13]);
    assert!(f.textport_jobs().is_empty());
    // A context activation clears the base before its regions are placed.
    f.cmd(&[0x1e4, 1, 0, 4, 8]);
    f.cmd(&[0x1fd, 0x1000_0000 | (0x50_0000 >> 12), 0, 0x10000]);
    f.take_jobs();
    f.cmd(&[0x16d, 0, 13]);
    assert!(f.textport_jobs().is_empty());
}

/// 0x170 runs up to eight ranges of a list in
/// pipe memory as one list. Nine is past the packet's limit and opens nothing:
/// the sync probe after it frames as a probe and is answered.
#[test]
fn a_list_call_takes_up_to_eight_ranges() {
    let f = Fixture::hooked();
    f.put_list(0x2000, &LIST);
    let counts = [2u32, 2, 2, 2, 2, 1, 1, 1];
    let mut cmd = vec![0x170, 8];
    let mut at = 0x2000;
    for c in counts {
        cmd.extend([at, c]);
        at += 4 * c;
    }
    f.cmd(&cmd);
    assert_eq!(f.textport_jobs(), LIST_JOBS);
    f.take_jobs();
    f.cmd(&[0x170, 9]);
    f.cmd(&[1, 9, 0]);
    assert_eq!(f.r(MAILBOX), 0x1234abcd);
    f.cmd(&[0x171, 5]);
    f.w(MAILBOX, 0);
    f.cmd(&[1, 9, 0]);
    assert_eq!(f.r(MAILBOX), 0x1234abcd);
    f.cmd(&[0x170, 0]);
    assert!(f.textport_jobs().is_empty());
}

/// A resident list may be as long as the
/// display-list class, 15 MiB of words, and runs whole. One word longer than the
/// class is refused before anything is read.
#[test]
fn a_list_may_be_as_long_as_the_class() {
    let f = Fixture::new();
    f.put_list(0, &[1, 9, 0]);
    let words = LIST_BYTES / 4;
    f.cmd(&[0x16d, 0, words]);
    assert_eq!(f.r(MAILBOX), 0x1234abcd, "the whole class ran");
    f.w(MAILBOX, 0);
    f.w(INT_CAUSE, CAUSE_MAILBOX);
    f.cmd(&[0x16d, 0, words + 1]);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0, "one word more is refused whole");
}

/// A list that invokes a list: the inner
/// invocation finds the engine running and fails, and the outer list goes on
/// after it. Without a pool, a pool invocation is refused.
#[test]
fn a_nested_list_fails_and_the_outer_list_goes_on() {
    let f = Fixture::hooked();
    // The inner list is a point; the outer list sets a colour, calls the inner
    // one and draws another point.
    f.put_list(0x100, &[1, 6, 1, 1]);
    f.put_list(0x0, &[1, 2, 5, 0x16d, 0x100, 4, 1, 6, 7, 8]);
    f.take_jobs();
    f.cmd(&[0x16d, 0, 10]);
    let jobs = f.jobs();
    assert_eq!(
        f.textport_jobs(),
        ["Textport 0x2 [5] packed=0x0 handled=false", "Textport 0x6 [7, 8] packed=0x0 handled=false"],
        "{jobs:?}"
    );
    assert!(jobs.iter().any(|j| j.starts_with("Kernel 0x16d [100, 4]")), "the call itself is seen");
    // No pool registered.
    f.take_jobs();
    f.cmd(&[0x16e, 0, 3]);
    assert!(f.textport_jobs().is_empty());
}

/// A list range that leaves the list memory, or a pool range that leaves the
/// pool, is refused with nothing fed.
#[test]
fn a_range_outside_its_memory_is_refused() {
    let f = Fixture::hooked();
    f.pool();
    f.host.put_words(POOL_HOST + POOL_LEN as u64 - 8, &[1, 6]);
    f.cmd(&[0x16e, POOL_LEN - 8, 4]);
    assert!(f.textport_jobs().is_empty());
    f.cmd(&[0x16d, LIST_BYTES - 8, 4]);
    f.cmd(&[0x16d, 2, 1]);
    assert!(f.textport_jobs().is_empty());
}

/// A pool that is not whole 16 KiB pages leaves no pool registered, silently;
/// one without the tag, or past the model's bound, is refused too.
#[test]
fn a_pool_must_be_whole_pages_and_tagged() {
    let f = Fixture::hooked();
    f.map_range(POOL_DEVICE, POOL_HOST, POOL_LEN as u64);
    f.host.put_words(POOL_HOST, &LIST);
    let page = (POOL_DEVICE >> 12) as u32;
    for words in [
        [0x1ed, page, 0x9000, 1],
        [0x1ed, page, 0x9000, 0x1000],
        [0x1ed, page, 0x0000, POOL_LEN],
        [0x1ed, page + 1, 0x9000, POOL_LEN],
        [0x1ed, page, 0x9000, 0x1000_4000],
    ] {
        f.cmd(&words);
        f.cmd(&[0x16e, 0, 13]);
        assert!(f.textport_jobs().is_empty(), "{words:x?}");
    }
    f.cmd(&[0x1ed, page, 0x9000, POOL_LEN]);
    f.cmd(&[0x16e, 0, 13]);
    assert_eq!(f.textport_jobs(), LIST_JOBS);
}

/// A reset forgets the pool, the region base and a pending record.
#[test]
fn reset_forgets_the_pool_the_region_and_the_record() {
    let f = Fixture::hooked();
    f.pool();
    f.cmd(&[0x1e4, 1, 0, 4, 8]);
    f.cmd(&[0x1c8, 0x40, 16]);
    f.kona.power_on();
    f.host.put_words(POOL_HOST, &LIST);
    f.cmd(&[0x16e, 0, 13]);
    assert!(f.textport_jobs().is_empty());
    // The record is gone: a packet is the back end's.
    f.host.put_words(SRC, &[1, 2, 3, 4]);
    f.packet(SRC, 16, 16, 0, 0);
    assert_eq!(f.list_word(0x40), 0);
    assert!(f.jobs().last().unwrap().starts_with("dma-in"));
}

/// A DMA completion goes out on the general vector.
#[test]
fn a_dma_completion_goes_out_on_the_general_vector() {
    let f = Fixture::new();
    f.w(INT_DEST_UPPER, 0x98000);
    f.w(INT_DEST_LOWER, 0x180_0090);
    f.w(INT_VECTORS, 0x3e2210);
    f.host.put_words(0x40_0000, &[1, 2, 3, 4]);
    f.cmd(&[0x1c8, 0x100, 16]);
    f.cmd(&[0x1eb, 0x1000_0400, 0x9000, 16, 16, 0, 0x100]);
    assert_eq!(f.host.sent().len(), 1);
    assert_eq!(f.host.sent()[0].vector, 0x10);
}


/// A packet whose high word has bit 28 clear carries `io >> 12` of a device
/// address: the map RAM entry `word >> 2` translates it, and the low two bits
/// of the word are bits 13:12 of the offset into its 16 KiB page. A packet
/// into an unmapped page is refused and completes nothing.
#[test]
fn a_mapped_packet_goes_through_the_map_ram() {
    let f = Fixture::hooked();
    // Device address 0x1b234: entry 6, offset 0x3234 in the page.
    f.map(6, 0x60_0000);
    f.host.put_words(0x60_3234, &[0x11, 0x22, 0x33, 0x44]);
    f.list_write(0x80, 16);
    f.dma(&[0x1eb, 0x1b, 0x9234, 16, 16, 0, 0x100]);
    assert_eq!(f.list_word(0x80), 0x11);
    assert_eq!(f.list_word(0x8c), 0x44);
    assert_eq!(f.r(MAILBOX), 0x100);
    // Entry 7 is not mapped.
    f.w(MAILBOX, 0);
    f.w(INT_CAUSE, CAUSE_MAILBOX);
    f.list_write(0x90, 16);
    f.dma(&[0x1eb, 0x1c, 0x9000, 16, 16, 0, 0x100]);
    assert_eq!(f.list_word(0x90), 0);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0);
    // With no record the back end gets the device address, not a physical one.
    f.dma(&[0x1eb, 0x1b, 0x9234, 16, 16, 0, 0]);
    assert_eq!(f.take_jobs(), ["dma-in 0x1b234 1 0x10 0x10"]);
}
