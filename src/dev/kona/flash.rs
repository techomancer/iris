//! The pipe's flash EEPROM and the operation that programs it.
//!
//! The flash is 512 KiB: the boot image from offset 0, a video combination at
//! `COMBO` behind a marker and a word count, and a twelve-word header at
//! `HEADER` whose word 7 is the state the PROM and the kernel accept only as
//! 2. The graphics install tool reads and programs it with command 0x1ec, an
//! operation number and that operation's words, and reads the pipe's answer
//! from a reply buffer: word 0 goes non-zero last, word 1 is a word count and
//! the data follows. The PROM and the kernel read the same header through the
//! indirect port at internal 0x87ff00.
//!
//! No factory image is known, so a new flash is zero with header
//! state 2 (approximated). The flash keeps its contents across reset, as flash
//! does; given a backing file, every byte the pipe programs is written through
//! so the contents outlast the process as they outlast power-off on a pipe.
//!
//! Ported from hw/misc/sgi-kona.c and hw/misc/sgi-kona-flash-image.c in
//! jamesbraid/qemu-sgi-origami.

use crate::devlog::LogModule;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

pub const FLASH_SIZE: usize = 0x80000;
pub(super) const COMBO: u32 = 0x20000;
pub(super) const COMBO_MARKER: u32 = 0xface_feed;
pub(super) const HEADER: u32 = 0x7ff00;
pub(super) const HEADER_STATE: u32 = HEADER + 7 * 4;
pub(super) const STATE_GOOD: u32 = 2;
/// The identification word op 5 answers: the manufacturer and device codes
/// in bytes 0 and 1, from the example readout in the InfiniteReality upgrade
/// manual (a documentation example, not a measurement). Bytes 2 and 3 are
/// unknown and answered zero.
pub(super) const IDENTITY: u32 = 0x1fc4_0000;
/// The largest count op 1 accepts; a larger one fails before its data.
pub(super) const MAX_WORDS: u32 = 0x7ffb;
/// The longest reply the board posts, in bytes from word 0: the 16 KiB
/// buffer both of the programming tool's paths name.
pub(super) const REPLY_MAX: u32 = 0x4000;

const OP_WRITE: u32 = 1;
const OP_READ: u32 = 2;
const OP_REPLY_BUFFER: u32 = 3;
const OP_CHECK: u32 = 4;
const OP_IDENTIFY: u32 = 5;
const OP_ERASE_COMBO: u32 = 6;

const TYPE_IMAGE: u32 = 0;
const TYPE_COMBO: u32 = 1;
const TYPE_HEADER: u32 = 2;

/// Words after the operation number, before any data. A write's first word is
/// its type, which decides the rest.
const OP_WORDS: [u8; 7] = [0, 1, 2, 3, 1, 1, 0];

/// The file a pipe's flash is kept in: opened for reading and writing, and
/// made as a new flash when it does not exist.
pub fn flash_file(path: &std::path::Path) -> Result<File, String> {
    let err = |e: std::io::Error| format!("kona flash {}: {e}", path.display());
    if !path.exists() {
        std::fs::write(path, Flash::fresh().image()).map_err(err)?;
    }
    std::fs::OpenOptions::new().read(true).write(true).open(path).map_err(err)
}

pub(super) struct Flash {
    image: Box<[u8]>,
    backing: Option<File>,
    /// A write-through failed: reported once, the bytes stay in the flash.
    sync_failed: bool,
}

impl Flash {
    /// The flash a new pipe starts with.
    pub fn fresh() -> Self {
        let mut image = vec![0u8; FLASH_SIZE].into_boxed_slice();
        image[HEADER_STATE as usize..][..4].copy_from_slice(&STATE_GOOD.to_be_bytes());
        Flash { image, backing: None, sync_failed: false }
    }

    /// A flash held in `file`, which must be exactly `FLASH_SIZE` bytes.
    pub fn backed(mut file: File) -> Result<Self, String> {
        let len = file.metadata().map_err(|e| format!("kona: cannot read the size of the flash image: {e}"))?.len();
        if len != FLASH_SIZE as u64 {
            return Err(format!("kona: the flash image is {len} bytes, not the required {FLASH_SIZE}"));
        }
        let mut f = Flash::fresh();
        file.seek(SeekFrom::Start(0)).and_then(|_| file.read_exact(&mut f.image))
            .map_err(|e| format!("kona: cannot read the flash image: {e}"))?;
        f.backing = Some(file);
        Ok(f)
    }

    /// The contents, as the board's own reads see them.
    pub fn image(&self) -> &[u8] {
        &self.image
    }

    /// The big-endian word at flash offset `offset`, or 0 past the end.
    pub fn load(&self, offset: u32) -> u32 {
        let o = offset as usize;
        if o > FLASH_SIZE - 4 {
            return 0;
        }
        u32::from_be_bytes(self.image[o..o + 4].try_into().unwrap())
    }

    /// Program four bytes at `at`; bytes past the end are dropped. True if
    /// all four landed.
    fn program(&mut self, at: u64, word: u32) -> bool {
        let bytes = word.to_be_bytes();
        let mut inside = true;
        for (i, b) in bytes.iter().enumerate() {
            match self.image.get_mut(at as usize + i) {
                Some(slot) => *slot = *b,
                None => inside = false,
            }
        }
        if (at as usize) < FLASH_SIZE {
            let end = (at as usize + 4).min(FLASH_SIZE);
            self.sync(at as usize, end);
        }
        inside
    }

    /// The image, as the stream module reads and writes it. A loaded image is not written to
    /// the backing file here: `write_through` does that, once the whole snapshot has parsed.
    pub fn visit<S: super::stream::Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { image: _, backing: _, sync_failed: _ } = self;
        s.bytes(&mut self.image);
    }

    /// The whole image to the backing file, as every programmed byte is.
    pub fn write_through(&mut self) {
        self.sync(0, self.image.len());
    }

    fn sync(&mut self, from: usize, to: usize) {
        let Some(file) = self.backing.as_mut() else { return };
        let r = file.seek(SeekFrom::Start(from as u64)).and_then(|_| file.write_all(&self.image[from..to]));
        if r.is_err() && !self.sync_failed {
            self.sync_failed = true;
            crate::dlog!(LogModule::Kona, "kona: write to the flash image failed; later writes may be lost from the image too");
        }
    }
}

/// What a completed step of an operation asks the board to do.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Action {
    /// Post these words on the return page at byte 4: the count, then them.
    Reply(Vec<u32>),
    /// Take the page and offset words as the new reply buffer.
    ReplyBuffer { page: u32, offset: u32 },
}

/// An operation taking its words from the command stream.
#[derive(Clone, Default, Debug)]
pub(super) struct FlashOp {
    op: u32,
    need: usize,
    have: usize,
    args: [u32; 3],
    /// Data words still to take (a write).
    data: u32,
    /// Whether they are stored (an unknown type takes its data and drops it).
    store: bool,
    addr: u64,
    outside: bool,
}

impl FlashOp {
    /// Whether an operation is open: every word is its.
    pub fn open(&self) -> bool {
        self.op != 0
    }

    /// The operation number arrived. Op 6 acts at once; an unknown number
    /// takes no more words.
    pub fn begin(&mut self, flash: &mut Flash, op: u32) -> Option<Action> {
        *self = FlashOp::default();
        if op == OP_ERASE_COMBO {
            // Four zero bytes over the combination's marker. Nothing posted.
            flash.program(COMBO as u64, 0);
            return None;
        }
        if op == 0 || op as usize >= OP_WORDS.len() {
            crate::dlog!(LogModule::Kona, "kona: EEPROM operation {op} is not one of 1 to 6; it takes no more words");
            return None;
        }
        self.op = op;
        self.need = OP_WORDS[op as usize] as usize;
        None
    }

    /// A word on the command page while the operation is open.
    pub fn word(&mut self, flash: &mut Flash, word: u32) -> Option<Action> {
        if self.have < self.need {
            self.args[self.have] = word;
            self.have += 1;
            if self.op == OP_WRITE && self.have == 1 {
                // An offset and a count for the image, a count otherwise.
                self.need = if word == TYPE_IMAGE { 3 } else { 2 };
            }
            if self.have == self.need {
                return self.ready(flash);
            }
            return None;
        }
        if self.store {
            self.store_word(flash, word);
        }
        self.data -= 1;
        if self.data == 0 {
            let status = self.store as u32;
            *self = FlashOp::default();
            return Some(Action::Reply(vec![status]));
        }
        None
    }

    fn store_word(&mut self, flash: &mut Flash, word: u32) {
        if !flash.program(self.addr, word) && !self.outside {
            crate::dlog!(
                LogModule::Kona,
                "kona: EEPROM write reaches flash offset {:#x}, past its 512 KiB; the bytes past it are dropped",
                self.addr
            );
            self.outside = true;
        }
        self.addr += 4;
    }

    /// The fixed words are in.
    fn ready(&mut self, flash: &mut Flash) -> Option<Action> {
        if self.op == OP_WRITE {
            return self.write_open(flash);
        }
        let done = std::mem::take(self);
        match done.op {
            OP_READ => read(flash, done.args[0], done.args[1]),
            OP_REPLY_BUFFER => Some(Action::ReplyBuffer { page: done.args[0], offset: done.args[1] }),
            // Approximated: the ARM sends a value round the transaction bus
            // and answers 1 when it comes back intact. The model has no such
            // bus; a working pipe's check passes.
            OP_CHECK => Some(Action::Reply(vec![1])),
            OP_IDENTIFY => Some(Action::Reply(vec![IDENTITY])),
            _ => unreachable!("operation {} has no fixed words to finish", done.op),
        }
    }

    /// A write's fixed words are in. A count above `MAX_WORDS` fails before
    /// any data word, so the next word opens a command. Otherwise
    /// `max(count, 1)` data words follow (the ARM always reads a last word),
    /// stored from the type's place: the image at its offset, the combination
    /// behind its marker and count, the header at its own place. An unknown
    /// type takes its data and stores none.
    fn write_open(&mut self, flash: &mut Flash) -> Option<Action> {
        let ty = self.args[0];
        let count = self.args[self.need - 1];
        if count > MAX_WORDS {
            crate::dlog!(LogModule::Kona, "kona: EEPROM write of {count} words, more than {MAX_WORDS}; refused");
            *self = FlashOp::default();
            return Some(Action::Reply(vec![0]));
        }
        self.data = count.max(1);
        self.store = true;
        match ty {
            TYPE_IMAGE => self.addr = self.args[1] as u64,
            TYPE_COMBO => {
                self.addr = COMBO as u64;
                self.store_word(flash, COMBO_MARKER);
                self.store_word(flash, count);
            }
            TYPE_HEADER => self.addr = HEADER as u64,
            _ => {
                crate::dlog!(LogModule::Kona, "kona: EEPROM write of unknown type {ty}; its words are stored nowhere");
                self.store = false;
            }
        }
        None
    }
}

/// Op 2: `bytes` bytes of flash from `offset`, rounded down to a word as the
/// ARM's word index does. A length that is not whole words would keep the ARM's
/// copy loop from ending, and a reply longer than the buffer would run past
/// it: both are logged and answered with nothing. Approximated: words past
/// the 512 KiB are answered zero.
fn read(flash: &Flash, offset: u32, bytes: u32) -> Option<Action> {
    if bytes % 4 != 0 {
        crate::dlog!(LogModule::Kona, "kona: EEPROM read of {bytes:#x} bytes, not whole words; nothing posted");
        return None;
    }
    if 8 + bytes as u64 > REPLY_MAX as u64 {
        crate::dlog!(LogModule::Kona, "kona: EEPROM read of {bytes:#x} bytes is longer than the reply buffer; nothing posted");
        return None;
    }
    let offset = offset & !3;
    if offset as u64 + bytes as u64 > FLASH_SIZE as u64 {
        crate::dlog!(LogModule::Kona, "kona: EEPROM read at flash offset {offset:#x} runs past its 512 KiB; read as zero there");
    }
    Some(Action::Reply((0..bytes / 4).map(|i| flash.load(offset.wrapping_add(4 * i))).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(op: &mut FlashOp, f: &mut Flash, words: &[u32]) -> Vec<Action> {
        let mut out = Vec::new();
        let mut it = words.iter();
        let first = *it.next().unwrap();
        out.extend(op.begin(f, first));
        for w in it {
            out.extend(op.word(f, *w));
        }
        out
    }

    #[test]
    fn a_new_flash_is_zero_with_the_header_state_the_prom_accepts() {
        let f = Flash::fresh();
        assert_eq!(f.load(HEADER_STATE), STATE_GOOD);
        assert_eq!(f.load(0), 0);
        assert_eq!(f.load(HEADER), 0);
    }

    #[test]
    fn an_image_write_lands_at_its_offset_and_answers_one() {
        let mut f = Flash::fresh();
        let mut op = FlashOp::default();
        let a = run(&mut op, &mut f, &[OP_WRITE, TYPE_IMAGE, 0x100, 2, 0xaabbccdd, 0x11223344]);
        assert_eq!(a, vec![Action::Reply(vec![1])]);
        assert_eq!(f.load(0x100), 0xaabbccdd);
        assert_eq!(f.load(0x104), 0x11223344);
        assert!(!op.open());
    }

    #[test]
    fn a_combination_write_is_framed_by_its_marker_and_count() {
        let mut f = Flash::fresh();
        let mut op = FlashOp::default();
        run(&mut op, &mut f, &[OP_WRITE, TYPE_COMBO, 1, 0x55]);
        assert_eq!(f.load(COMBO), COMBO_MARKER);
        assert_eq!(f.load(COMBO + 4), 1);
        assert_eq!(f.load(COMBO + 8), 0x55);
    }

    #[test]
    fn erasing_the_combination_zeroes_its_marker_and_posts_nothing() {
        let mut f = Flash::fresh();
        let mut op = FlashOp::default();
        run(&mut op, &mut f, &[OP_WRITE, TYPE_COMBO, 1, 0x55]);
        assert!(run(&mut op, &mut f, &[OP_ERASE_COMBO]).is_empty());
        assert_eq!(f.load(COMBO), 0);
        assert_eq!(f.load(COMBO + 8), 0x55);
        assert!(!op.open());
    }

    #[test]
    fn an_oversized_count_is_refused_before_any_data() {
        let mut f = Flash::fresh();
        let mut op = FlashOp::default();
        let a = run(&mut op, &mut f, &[OP_WRITE, TYPE_HEADER, MAX_WORDS + 1]);
        assert_eq!(a, vec![Action::Reply(vec![0])]);
        assert!(!op.open());
    }

    #[test]
    fn an_unknown_type_takes_its_data_and_stores_none() {
        let mut f = Flash::fresh();
        let mut op = FlashOp::default();
        let a = run(&mut op, &mut f, &[OP_WRITE, 9, 2, 0xdead, 0xbeef]);
        assert_eq!(a, vec![Action::Reply(vec![0])]);
        assert_eq!(f.load(0xdead), 0);
        assert!(!op.open());
    }

    #[test]
    fn an_unknown_operation_takes_nothing() {
        let mut f = Flash::fresh();
        let mut op = FlashOp::default();
        assert!(run(&mut op, &mut f, &[0]).is_empty());
        assert!(!op.open());
        assert!(run(&mut op, &mut f, &[7]).is_empty());
        assert!(!op.open());
    }

    #[test]
    fn a_read_past_the_end_answers_zero_and_a_write_past_it_is_dropped() {
        let mut f = Flash::fresh();
        let mut op = FlashOp::default();
        let a = run(&mut op, &mut f, &[OP_READ, FLASH_SIZE as u32 - 4, 8]);
        assert_eq!(a, vec![Action::Reply(vec![0, 0])]);
        let a = run(&mut op, &mut f, &[OP_WRITE, TYPE_IMAGE, FLASH_SIZE as u32 - 4, 2, 1, 2]);
        assert_eq!(a, vec![Action::Reply(vec![1])]);
        assert_eq!(f.load(FLASH_SIZE as u32 - 4), 1);
    }

    #[test]
    fn a_read_longer_than_the_reply_buffer_or_not_whole_words_posts_nothing() {
        let mut f = Flash::fresh();
        let mut op = FlashOp::default();
        assert!(run(&mut op, &mut f, &[OP_READ, 0, 3]).is_empty());
        assert!(run(&mut op, &mut f, &[OP_READ, 0, REPLY_MAX]).is_empty());
    }

    #[test]
    fn check_and_identify_answer_without_taking_effect() {
        let mut f = Flash::fresh();
        let mut op = FlashOp::default();
        assert_eq!(run(&mut op, &mut f, &[OP_CHECK, 0]), vec![Action::Reply(vec![1])]);
        assert_eq!(run(&mut op, &mut f, &[OP_IDENTIFY, 0]), vec![Action::Reply(vec![IDENTITY])]);
    }

    #[test]
    fn the_backed_flash_writes_through_and_reloads() {
        let dir = crate::dev::kona::tests::ScratchDir::new("flash");
        let path = dir.join("flash.bin");
        std::fs::write(&path, vec![0u8; FLASH_SIZE]).unwrap();
        let open = || std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
        let mut f = Flash::backed(open()).unwrap();
        let mut op = FlashOp::default();
        run(&mut op, &mut f, &[OP_WRITE, TYPE_IMAGE, 0x40, 1, 0x1234_5678]);
        drop(f);
        let again = Flash::backed(open()).unwrap();
        assert_eq!(again.load(0x40), 0x1234_5678);
        // A file of the wrong size is refused.
        std::fs::write(&path, vec![0u8; 10]).unwrap();
        assert!(Flash::backed(open()).is_err());
    }

    /// An image group reads back from where it was
    /// written, including from an offset inside the group.
    #[test]
    fn a_written_image_reads_back_from_any_word() {
        let mut f = Flash::fresh();
        let mut op = FlashOp::default();
        let words: Vec<u32> = (0..64u32).map(|i| 0xa500_0000 | (i * 0x10203)).collect();
        let mut req = vec![OP_WRITE, TYPE_IMAGE, 0x4000, 64];
        req.extend(&words);
        assert_eq!(run(&mut op, &mut f, &req), vec![Action::Reply(vec![1])]);
        assert_eq!(run(&mut op, &mut f, &[OP_READ, 0x4000, 256]), vec![Action::Reply(words.clone())]);
        assert_eq!(run(&mut op, &mut f, &[OP_READ, 0x4008, 8]), vec![Action::Reply(vec![words[2], words[3]])]);
    }

    /// A header written with state 3 is the
    /// state the PROM's check reads next time, at the address the indirect port
    /// reads it from.
    #[test]
    fn a_written_header_is_what_the_next_read_returns() {
        let mut f = Flash::fresh();
        let mut op = FlashOp::default();
        let mut header: Vec<u32> = (0..12u32).map(|i| 0x1111_1111 * (i + 1)).collect();
        header[7] = 3;
        let mut req = vec![OP_WRITE, TYPE_HEADER, 12];
        req.extend(&header);
        assert_eq!(run(&mut op, &mut f, &req), vec![Action::Reply(vec![1])]);
        assert_eq!(run(&mut op, &mut f, &[OP_READ, HEADER, 48]), vec![Action::Reply(header)]);
        assert_eq!(f.load(HEADER_STATE), 3);
    }
}
