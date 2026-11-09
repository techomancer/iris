//! The flash EEPROM operation, command 0x1ec. Ported from
//! tests/qtest/sn0-origin2000-kona-eeprom-test.c.
//!
//! The pipe's flash holds its boot image, a video combination and a 48-byte
//! header. The graphics install tool reads and programs it with command 0x1ec,
//! an operation number and that operation's words, and reads the pipe's answer
//! from a reply buffer: word 0 goes non-zero last, word 1 is a word count and
//! the data follows. The PROM and the kernel read the same header through the
//! indirect port at internal 0x87ff00.

use super::*;

const PORT_ADDR: u64 = 0x8003e8;
const PORT_DATA: u64 = 0x8003f0;
const INT_CAUSE: u64 = 0x800238;
const MAILBOX: u64 = 0x800240;

const EEPROM_OPERATION: u32 = 0x1ec;
const OP_WRITE: u32 = 1;
const OP_READ: u32 = 2;
const OP_REPLY_BUFFER: u32 = 3;
const OP_CHECK: u32 = 4;
const OP_IDENTIFY: u32 = 5;
const OP_ERASE_COMBO: u32 = 6;
const TYPE_IMAGE: u32 = 0;
const TYPE_COMBO: u32 = 1;
const TYPE_HEADER: u32 = 2;

const FLASH_COMBO: u32 = 0x20000;
const FLASH_HEADER: u32 = 0x7ff00;
const HEADER_WORDS: usize = 12;
const HEADER_STATE: usize = 7;
/// The indirect port reaches the flash's read side here.
const INTERNAL_FLASH: u32 = 0x800000;
const FLASH_IDENTITY: u32 = 0x1fc4_0000;
const REPLY_PAGE: u64 = 0x32_0000;
const OTHER_PAGE: u64 = 0x33_0000;

impl Fixture {
    pub(super) fn peek(&self, addr: u32) -> u32 {
        self.w(PORT_ADDR, addr);
        self.r(PORT_DATA)
    }

    pub(super) fn clear(&self, addr: u64, len: usize) {
        self.host.put(addr, &vec![0; len]);
    }

    /// The unmanaged path's reply buffer: 0x1ec 3, the page word with the
    /// host-physical bit, the offset word and the buffer length.
    pub(super) fn reply_buffer(&self, addr: u64) {
        self.cmd(&[EEPROM_OPERATION, OP_REPLY_BUFFER, 0x1000_0000 | (addr >> 12) as u32, (addr & 0xfff) as u32, 0x4000]);
    }

    /// The reply at `addr`: word 0 must be set; returns the count in word 1.
    pub(super) fn reply_count(&self, addr: u64) -> u32 {
        assert_eq!(self.host.word(addr), 1, "word 0 goes non-zero last");
        self.host.word(addr + 4)
    }

    pub(super) fn reply_data(&self, addr: u64, i: u64) -> u32 {
        self.host.word(addr + 8 + 4 * i)
    }

    /// Op 1: a type, the offset for type 0 only, a count, then the words.
    pub(super) fn flash_write(&self, ty: u32, offset: u32, words: &[u32]) {
        self.cmd(&[EEPROM_OPERATION, OP_WRITE, ty]);
        if ty == TYPE_IMAGE {
            self.cmd(&[offset]);
        }
        self.cmd(&[words.len() as u32]);
        self.cmd(words);
    }

    pub(super) fn flash_read(&self, offset: u32, bytes: u32) {
        self.cmd(&[EEPROM_OPERATION, OP_READ, offset, bytes]);
    }

    /// The PROM's sync probe, 1 9 0: its mailbox reply shows the command page
    /// is between commands, so no word of an earlier operation is still owed.
    pub(super) fn assert_in_step(&self) {
        self.cmd(&[1, 9, 0]);
        // Wait for the stream: the polled registers below do not.
        self.r(PORT_DATA);
        assert_eq!(self.r(INT_CAUSE) & 0x10, 0x10);
        assert_eq!(self.r(MAILBOX), 0x1234abcd);
        self.w(MAILBOX, 0);
        self.w(INT_CAUSE, 0x10);
    }

    pub(super) fn assert_check_answers(&self, at: u64) {
        self.clear(at, 0x10);
        self.cmd(&[EEPROM_OPERATION, OP_CHECK, 0]);
        assert_eq!(self.reply_count(at), 1);
        assert_eq!(self.reply_data(at, 0), 1);
    }
}

/// The check, 0x1ec 4 0, answers count 1 and a
/// non-zero status in the buffer 0x1ec 3 named, at its byte offset: the tool
/// polls word 0 and then needs word 2 non-zero, and without it prints
/// "graphics not responding". Fails on a model that does not frame 0x1ec, does
/// not post, or drops the offset's low 12 bits as 0x1f2 does on the pipe.
#[test]
fn check_answers_at_offset() {
    let f = Fixture::new();
    let at = REPLY_PAGE + 0x100;
    f.clear(REPLY_PAGE, 0x200);
    f.reply_buffer(at);
    f.cmd(&[EEPROM_OPERATION, OP_CHECK, 0]);
    assert_eq!(f.reply_count(at), 1);
    assert_eq!(f.reply_data(at, 0), 1);
    assert_eq!(f.host.word(REPLY_PAGE), 0);
    f.assert_in_step();
}

/// 0x1ec 5 0: count 1, the identity word.
#[test]
fn identify() {
    let f = Fixture::new();
    f.reply_buffer(REPLY_PAGE);
    f.cmd(&[EEPROM_OPERATION, OP_IDENTIFY, 0]);
    assert_eq!(f.reply_count(REPLY_PAGE), 1);
    assert_eq!(f.reply_data(REPLY_PAGE, 0), FLASH_IDENTITY);
    f.assert_in_step();
}

/// The default flash is zero, but the header's state
/// reads 2, through op 2 and through the indirect port the PROM and kernel
/// use. A model that kept a fixed answer at one port address only would fail
/// the op 2 half.
#[test]
fn default_header() {
    let f = Fixture::new();
    f.reply_buffer(REPLY_PAGE);
    f.flash_read(FLASH_HEADER, HEADER_WORDS as u32 * 4);
    assert_eq!(f.reply_count(REPLY_PAGE), HEADER_WORDS as u32);
    for i in 0..HEADER_WORDS {
        let want = if i == HEADER_STATE { 2 } else { 0 };
        assert_eq!(f.reply_data(REPLY_PAGE, i as u64), want);
        assert_eq!(f.peek(INTERNAL_FLASH + FLASH_HEADER + 4 * i as u32), want);
    }
}

/// The tool's header write, 0x1ec 1 2 12 and
/// twelve words: status 1, and the words read back through op 2 and through
/// the indirect port at 0x87ff00, so a header written with state 3 is what the
/// PROM's check sees on the next boot.
#[test]
fn header_write_reaches_port() {
    let f = Fixture::new();
    let mut header: Vec<u32> = (0..HEADER_WORDS as u32).map(|i| 0x1111_1111 * (i + 1)).collect();
    header[HEADER_STATE] = 3;

    f.reply_buffer(REPLY_PAGE);
    f.flash_write(TYPE_HEADER, 0, &header);
    assert_eq!(f.reply_count(REPLY_PAGE), 1);
    assert_eq!(f.reply_data(REPLY_PAGE, 0), 1);

    f.clear(REPLY_PAGE, 0x100);
    f.flash_read(FLASH_HEADER, 4 * HEADER_WORDS as u32);
    assert_eq!(f.reply_count(REPLY_PAGE), HEADER_WORDS as u32);
    for (i, want) in header.iter().enumerate() {
        assert_eq!(f.reply_data(REPLY_PAGE, i as u64), *want);
        assert_eq!(f.peek(INTERNAL_FLASH + FLASH_HEADER + 4 * i as u32), *want);
    }
    f.assert_in_step();
}

/// An image group, 0x1ec 1 0 a n and n words,
/// reads back from a with op 2, including at an offset op 2 reads unaligned to
/// the group.
#[test]
fn image_write_reads_back() {
    let words: Vec<u32> = (0..64u32).map(|i| 0xa500_0000 | (i * 0x10203)).collect();
    let f = Fixture::new();
    f.reply_buffer(REPLY_PAGE);
    f.flash_write(TYPE_IMAGE, 0x4000, &words);
    assert_eq!(f.reply_count(REPLY_PAGE), 1);
    assert_eq!(f.reply_data(REPLY_PAGE, 0), 1);

    f.clear(REPLY_PAGE, 0x400);
    f.flash_read(0x4000, words.len() as u32 * 4);
    assert_eq!(f.reply_count(REPLY_PAGE), words.len() as u32);
    for (i, w) in words.iter().enumerate() {
        assert_eq!(f.reply_data(REPLY_PAGE, i as u64), *w);
    }

    f.clear(REPLY_PAGE, 0x400);
    f.flash_read(0x4000 + 8, 8);
    assert_eq!(f.reply_count(REPLY_PAGE), 2);
    assert_eq!(f.reply_data(REPLY_PAGE, 0), words[2]);
    assert_eq!(f.reply_data(REPLY_PAGE, 1), words[3]);
    f.assert_in_step();
}

/// A count above 0x7ffb fails before any data
/// word is read: status 0, and the next word starts a new command. A model
/// that took the count's words would swallow the sync probe that follows.
#[test]
fn oversized_count_refused() {
    let f = Fixture::new();
    f.reply_buffer(REPLY_PAGE);
    f.cmd(&[EEPROM_OPERATION, OP_WRITE, TYPE_IMAGE, 0, 0x7ffc]);
    assert_eq!(f.reply_count(REPLY_PAGE), 1);
    assert_eq!(f.reply_data(REPLY_PAGE, 0), 0);
    f.assert_in_step();
}

/// A combination, 0x1ec 1 1 n and n words,
/// lands at 0x20000 after the marker 0xfacefeed and n; 0x1ec 6, which takes no
/// word and posts nothing, zeroes the marker.
#[test]
fn combination_and_its_erase() {
    let words = [0xc0ffee01, 0xc0ffee02, 0xc0ffee03];
    let f = Fixture::new();
    f.reply_buffer(REPLY_PAGE);
    f.flash_write(TYPE_COMBO, 0, &words);
    assert_eq!(f.reply_data(REPLY_PAGE, 0), 1);

    assert_eq!(f.peek(INTERNAL_FLASH + FLASH_COMBO), 0xfacefeed);
    assert_eq!(f.peek(INTERNAL_FLASH + FLASH_COMBO + 4), 3);
    for (i, w) in words.iter().enumerate() {
        assert_eq!(f.peek(INTERNAL_FLASH + FLASH_COMBO + 8 + 4 * i as u32), *w);
    }

    f.clear(REPLY_PAGE, 0x100);
    f.cmd(&[EEPROM_OPERATION, OP_ERASE_COMBO]);
    assert_eq!(f.host.word(REPLY_PAGE), 0);
    assert_eq!(f.peek(INTERNAL_FLASH + FLASH_COMBO), 0);
    assert_eq!(f.peek(INTERNAL_FLASH + FLASH_COMBO + 4), 3);
    f.assert_in_step();
}

/// An operation number outside 1 to 6 takes
/// no more words: the ARM returns having read nothing more. The check that
/// follows answers, which fails on a model that swallowed its 0x1ec as the
/// unknown operation's word.
#[test]
fn unknown_op_takes_nothing() {
    let f = Fixture::new();
    f.reply_buffer(REPLY_PAGE);
    f.cmd(&[EEPROM_OPERATION, 7]);
    f.assert_check_answers(REPLY_PAGE);
    f.cmd(&[EEPROM_OPERATION, 0]);
    f.assert_check_answers(REPLY_PAGE);
    f.assert_in_step();
}

/// A write of an unknown type takes its
/// count's data words, stores none of them and answers 0.
#[test]
fn unknown_type_consumes_data() {
    let words = [0x0bad_da7a, 0x0bad_da7b];
    let f = Fixture::new();
    f.reply_buffer(REPLY_PAGE);
    f.cmd(&[EEPROM_OPERATION, OP_WRITE, 3, 2]);
    f.cmd(&words);
    assert_eq!(f.reply_count(REPLY_PAGE), 1);
    assert_eq!(f.reply_data(REPLY_PAGE, 0), 0);
    for off in (0..0x10).step_by(4) {
        let w = f.peek(INTERNAL_FLASH + off);
        assert!(w != words[0] && w != words[1]);
    }
    f.assert_check_answers(REPLY_PAGE);
    f.assert_in_step();
}

/// Reads the model refuses: a length that
/// is not whole words, which would never end the ARM's copy loop, and a reply
/// longer than the 16 KiB buffer. Each takes its two words, posts nothing, and
/// leaves the page in step.
#[test]
fn refused_reads_post_nothing() {
    let f = Fixture::new();
    f.reply_buffer(REPLY_PAGE);
    for len in [6, 0x4000 - 4, 0x4000] {
        f.clear(REPLY_PAGE, 0x10);
        f.flash_read(0, len);
        assert_eq!(f.host.word(REPLY_PAGE), 0, "length {len:#x}");
        f.assert_check_answers(REPLY_PAGE);
    }
    // The longest read that fits: count plus 0xff8 data words.
    f.clear(REPLY_PAGE, 0x10);
    f.flash_read(0, 0x4000 - 8);
    assert_eq!(f.reply_count(REPLY_PAGE), (0x4000 - 8) / 4);
    f.assert_in_step();
}

/// A read running past the 512 KiB answers
/// the words inside it and, as the model's choice, zero past it; the count is
/// the whole request.
#[test]
fn read_past_end_answers_zero() {
    let words = [0xe0d0_f000, 0xe0d0_f004];
    let f = Fixture::new();
    f.reply_buffer(OTHER_PAGE);
    f.flash_write(TYPE_IMAGE, 0x7fff8, &words);
    f.clear(REPLY_PAGE, 0x100);
    f.reply_buffer(REPLY_PAGE);
    f.flash_read(0x7fff8, 16);
    assert_eq!(f.reply_count(REPLY_PAGE), 4);
    assert_eq!(f.reply_data(REPLY_PAGE, 0), words[0]);
    assert_eq!(f.reply_data(REPLY_PAGE, 1), words[1]);
    assert_eq!(f.reply_data(REPLY_PAGE, 2), 0);
    assert_eq!(f.reply_data(REPLY_PAGE, 3), 0);
    // An offset op 2 rounds down to a word.
    f.clear(REPLY_PAGE, 0x100);
    f.flash_read(0x7fff8 + 3, 4);
    assert_eq!(f.reply_data(REPLY_PAGE, 0), words[0]);
    f.assert_in_step();
}

/// A write running past the 512 KiB stores the
/// bytes inside it, drops the rest, answers 1, and does not wrap round to
/// offset 0 even from an offset near 2^32.
#[test]
fn write_past_end_dropped() {
    let words = [0x5a5a_0001, 0x5a5a_0002];
    let f = Fixture::new();
    f.reply_buffer(REPLY_PAGE);
    f.flash_write(TYPE_IMAGE, 0x7fffc, &words);
    assert_eq!(f.reply_data(REPLY_PAGE, 0), 1);
    assert_eq!(f.peek(INTERNAL_FLASH + 0x7fffc), words[0]);

    f.clear(REPLY_PAGE, 0x100);
    f.flash_write(TYPE_IMAGE, 0xffff_fffc, &words);
    assert_eq!(f.reply_data(REPLY_PAGE, 0), 1);
    assert_ne!(f.peek(INTERNAL_FLASH), words[1]);
    f.assert_in_step();
}

/// The flash is non-volatile: a written word
/// survives a system reset, where the reply buffer does not.
#[test]
fn flash_survives_reset() {
    let f = Fixture::new();
    f.reply_buffer(REPLY_PAGE);
    f.flash_write(TYPE_IMAGE, 0x100, &[0x5eed_1e55]);
    f.kona.power_on();
    assert_eq!(f.peek(INTERNAL_FLASH + 0x100), 0x5eed_1e55);
    // The reply buffer did not survive: the check answers nowhere.
    f.clear(REPLY_PAGE, 0x10);
    f.cmd(&[EEPROM_OPERATION, OP_CHECK, 0]);
    assert_eq!(f.host.word(REPLY_PAGE), 0);
}

/// The EEPROM operation is interrupted mid-write by a slot store: the partial
/// operation is dropped and the new command opens.
#[test]
fn a_slot_store_drops_a_partial_operation() {
    let f = Fixture::new();
    f.reply_buffer(REPLY_PAGE);
    f.cmd(&[EEPROM_OPERATION, OP_WRITE, TYPE_IMAGE, 0x100, 4, 0x1111]);
    // Token 0x172 stored in its slot, with its argument.
    f.w(0xc1000c + 0x20 * 0x172, 0);
    f.assert_in_step();
    assert_eq!(f.peek(INTERNAL_FLASH + 0x100), 0x1111, "the word taken before the drop stays");
    assert_eq!(f.peek(INTERNAL_FLASH + 0x104), 0);
}

/// The backed flash survives the board: what the pipe programmed is in the
/// file, and a new board over the file reads it.
#[test]
fn a_flash_file_outlasts_the_board() {
    let dir = ScratchDir::new("eeprom-flash");
    let path = dir.join("flash.bin");
    std::fs::write(&path, vec![0u8; crate::dev::kona::FLASH_SIZE]).unwrap();
    let open = || std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
    let build = |host: &Arc<TestHost>| Kona::new(test_cfg(), host.clone(), host.clone(), crate::dev::kona::tests::test_status(), Some(open())).unwrap();
    {
        let host = TestHost::new();
        let f = Fixture::of(host.clone(), build(&host));
        f.reply_buffer(REPLY_PAGE);
        f.flash_write(TYPE_IMAGE, 0x2000, &[0xfeed_f00d, 0xcafe_0001]);
        assert_eq!(f.reply_data(REPLY_PAGE, 0), 1);
    }
    let host = TestHost::new();
    let f = Fixture::of(host.clone(), build(&host));
    assert_eq!(f.peek(INTERNAL_FLASH + 0x2000), 0xfeed_f00d);
    assert_eq!(f.peek(INTERNAL_FLASH + 0x2004), 0xcafe_0001);
}

/// A snapshot that does not parse is refused with the board and its flash file as they were: the state
/// blob cut short (every byte of the first 64, then at a growing stride, then the last bytes) or given a
/// trailing byte leaves the board's state unchanged and the flash file unwritten. The cuts go to the state
/// loader as bytes in memory; one refusal and the whole snapshot go through the chunk store, the path a
/// machine takes, and the file takes the image.
#[test]
fn a_snapshot_that_does_not_parse_leaves_the_board_and_its_flash_file_alone() {
    use crate::dev::kona::back::snapshot::put_bytes;
    let dir = ScratchDir::new("eeprom-snapshot");
    let file = |name: &str| {
        let path = dir.join(name);
        std::fs::write(&path, vec![0u8; crate::dev::kona::FLASH_SIZE]).unwrap();
        path
    };
    let (pa, pb) = (file("a.bin"), file("b.bin"));
    let make = |path: &std::path::Path| {
        let host = TestHost::new();
        let f = std::fs::OpenOptions::new().read(true).write(true).open(path).unwrap();
        (host.clone(), Kona::new(test_cfg(), host.clone(), host, crate::dev::kona::tests::test_status(), Some(f)).unwrap())
    };
    let cs = crate::chunk_store::ChunkStore::new(dir.join("saves"));
    // A board with a programmed flash image, snapshotted.
    let (host, kona) = make(&pa);
    let a = Fixture::of(host, kona);
    a.reply_buffer(REPLY_PAGE);
    a.flash_write(TYPE_IMAGE, 0x2000, &[0xfeed_f00d, 0xcafe_0001]);
    a.settle();
    let saved = a.kona.save_chunks(&cs).unwrap();
    let blob = a.kona.state_bytes(&mut a.kona.shared.pipe.lock()).unwrap();
    // Another board, its flash file untouched.
    // Not started, so nothing moves under the comparison.
    let (_host, b) = make(&pb);
    let before = b.save_chunks(&cs).unwrap();
    let state_before = b.state_bytes(&mut b.shared.pipe.lock()).unwrap();
    let flash_before = std::fs::read(&pb).unwrap();
    // Every byte of the first 64, a doubling stride after, and the last bytes: the cuts fall inside every field the blob has.
    let cuts: Vec<usize> = (0..64).chain(std::iter::successors(Some(97usize), |c| Some(c * 3 / 2 + 1)).take_while(|&c| c < blob.len())).chain(blob.len() - 8..blob.len()).collect();
    for cut in cuts {
        assert!(b.load_state(&mut b.shared.pipe.lock(), &blob[..cut]).is_err(), "a blob cut at {cut} loaded");
    }
    let mut long = blob.clone();
    long.push(0);
    assert!(b.load_state(&mut b.shared.pipe.lock(), &long).is_err(), "a blob with a trailing byte loaded");
    // The same refusal through the chunk store, with the bulk chunks the snapshot has.
    let cut = &blob[..blob.len() / 2];
    let mut list: Vec<_> = saved.iter().filter(|c| !c.name.starts_with("state-")).cloned().collect();
    list.push(crate::dev::kona::KonaChunk { name: format!("state-{}", cut.len()), hashes: put_bytes(&cs, cut).unwrap() });
    assert!(b.load_chunks(&cs, &list).is_err());
    assert_eq!(b.state_bytes(&mut b.shared.pipe.lock()).unwrap(), state_before, "a refused load changed the board");
    assert_eq!(b.save_chunks(&cs).unwrap(), before, "a refused load changed the board");
    assert_eq!(std::fs::read(&pb).unwrap(), flash_before, "a refused load wrote the flash file");
    // The whole snapshot loads, and the file takes the image.
    b.load_chunks(&cs, &saved).unwrap();
    assert_eq!(&std::fs::read(&pb).unwrap()[0x2000..0x2008], &[0xfe, 0xed, 0xf0, 0x0d, 0xca, 0xfe, 0x00, 0x01]);
    assert_eq!(b.save_chunks(&cs).unwrap(), saved);
}
