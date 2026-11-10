//! The pixel store: one array for the whole screen, held word by word.
//!
//! A pixel owns a record of up to 64 32-bit words (the planes of its window,
//! laid out by `layout`). The store keeps one two-dimensional array of `u32`
//! per word position, one entry per pixel of the managed area, in rows of the
//! managed area's pitch. A fill that touches one or two words of a record
//! then writes contiguous memory, and a window that uses 13 of the 16 words
//! of its class allocates 13 arrays, not 16.
//!
//! The physical address function from pixel and word to memory is unknown and
//! nothing a guest sees depends on it, so it stays behind `word_slot`. Cold
//! paths (read-back, snapshots, region swaps, monitors) use `word_slot`; a
//! draw resolves each array's base pointer and pitch once when it is set up
//! and its kernels index from those.
//!
//! # Allocation and publication
//!
//! An array is allocated whole, never grown, zero-filled, on the raster side
//! at a draw's or clear's set-up and before any tile worker sees the job: the
//! set-up allocates every array the job's planes read or write that is still
//! null (a depth test with depth writes off still reads depth). Each array is
//! published through a fixed slot, an `AtomicPtr` per word position, stored
//! once with release ordering and never changed until the store is dropped. A
//! reset zeroes the arrays in place and frees nothing. A worker loads its base
//! pointers when its job's state is resolved, so a store on the pixel path
//! checks nothing. The display loads a word position once per row run with
//! acquire ordering and treats null as a run of zeros. Because a pointer never
//! moves, a display read that races a worker's write sees the old or the new
//! value, as the lock-free display accepts.
//!
//! The elements are `AtomicU32` and accessed relaxed: a racing read and write
//! is then defined, and costs a plain load or store.

pub mod layout;
pub mod texture;

use std::alloc::{alloc_zeroed, dealloc, Layout as MemLayout};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering::{Acquire, Relaxed, Release}};

/// Words in a pixel's record: the largest class.
pub const MAX_WORDS: usize = 64;

/// Planes held beside the pixel record, each in a position past it: the
/// display-ID plane, whose 13-bit words name a pixel's window for the scan-out,
/// and the three planes the X server draws that are not colour (overlay 0,
/// overlay 1 and popup). Where the board keeps them is not known and nothing a
/// guest sees depends on it; they take positions of their own so a draw
/// resolves them as it does a colour word.
pub const DID_PLANE: usize = MAX_WORDS;
pub const OVERLAY0_PLANE: usize = MAX_WORDS + 1;
pub const OVERLAY1_PLANE: usize = MAX_WORDS + 2;
pub const POPUP_PLANE: usize = MAX_WORDS + 3;
/// Word positions in all: the record's and the planes beside it.
pub const POSITIONS: usize = MAX_WORDS + 4;

/// One word position's array as a draw resolves it: its base, its length and
/// the pitch of a row. It borrows the store, so it cannot outlive it, and the
/// array never moves while the store lives.
#[derive(Clone, Copy)]
pub struct WordArray<'a> {
    base: *const AtomicU32,
    len: usize,
    pitch: usize,
    _store: PhantomData<&'a PixelStore>,
}

// SAFETY: the base points into an array the store owns until it drops, and
// `'a` keeps the store borrowed; the elements are atomics.
unsafe impl Send for WordArray<'_> {}
unsafe impl Sync for WordArray<'_> {}

impl WordArray<'_> {
    /// The word of the pixel at `index` (row * pitch + column).
    ///
    /// # Safety
    ///
    /// `index` must be less than `len()`. A draw meets this by clipping to the
    /// managed area before it indexes, so the pixel path carries no check; a
    /// debug build asserts it.
    #[inline]
    pub unsafe fn load(&self, index: usize) -> u32 {
        debug_assert!(index < self.len, "pixel index {index} past an array of {}", self.len);
        // SAFETY: `index` is in range by the caller's contract, and the array
        // is live for `'a`.
        unsafe { (*self.base.add(index)).load(Relaxed) }
    }

    /// Store `value` in the word of the pixel at `index`.
    ///
    /// # Safety
    ///
    /// As `load`: `index` must be less than `len()`.
    #[inline]
    pub unsafe fn store(&self, index: usize, value: u32) {
        debug_assert!(index < self.len, "pixel index {index} past an array of {}", self.len);
        // SAFETY: as `load`.
        unsafe { (*self.base.add(index)).store(value, Relaxed) }
    }

    /// Entries in the array: pitch times rows.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn pitch(&self) -> usize {
        self.pitch
    }
}

pub struct PixelStore {
    width: usize,
    height: usize,
    slots: [AtomicPtr<AtomicU32>; POSITIONS],
    /// Texture memory, which a draw reads beside the planes.
    tex: texture::TextureStore,
}

impl PixelStore {
    /// A store for a managed area of `width` x `height` pixels. Nothing is
    /// allocated until a word position is first written.
    #[cfg(test)]
    pub fn new(width: u32, height: u32) -> PixelStore {
        Self::with_texture(width, height, Default::default())
    }

    /// A store whose texture memory is `tex`: a product's own size.
    pub fn with_texture(width: u32, height: u32, tex: texture::TextureStore) -> PixelStore {
        PixelStore { width: width as usize, height: height as usize, slots: std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())), tex }
    }

    /// Texture memory.
    pub fn texture(&self) -> &texture::TextureStore {
        &self.tex
    }

    pub fn width(&self) -> u32 {
        self.width as u32
    }

    pub fn height(&self) -> u32 {
        self.height as u32
    }

    /// Entries in one word position's array.
    fn entries(&self) -> usize {
        self.width * self.height
    }

    /// Where the word `w` of pixel (`x`, `y`) lives: the word position and the
    /// index in its array. The one definition of the address function; None
    /// outside the managed area or past the record.
    pub fn word_slot(&self, x: u32, y: u32, w: usize) -> Option<(usize, usize)> {
        (w < POSITIONS && (x as usize) < self.width && (y as usize) < self.height).then(|| (w, y as usize * self.width + x as usize))
    }

    /// Allocate word position `w`'s array if it is still null, zero-filled, and
    /// publish it. The set-up step of a draw or clear calls this for every word
    /// its planes touch, before any worker runs. Safe from more than one
    /// thread: the loser of a race frees its array.
    pub fn ensure(&self, w: usize) {
        assert!(w < POSITIONS);
        if !self.slots[w].load(Acquire).is_null() {
            return;
        }
        let n = self.entries();
        if n == 0 {
            return;
        }
        let layout = MemLayout::array::<AtomicU32>(n).expect("the managed area fits memory");
        // SAFETY: the layout has a nonzero size; an all-zero bit pattern is a
        // valid `AtomicU32`.
        let p = unsafe { alloc_zeroed(layout) } as *mut AtomicU32;
        assert!(!p.is_null(), "pixel store allocation of {} bytes failed", layout.size());
        if self.slots[w].compare_exchange(std::ptr::null_mut(), p, Release, Acquire).is_err() {
            // SAFETY: allocated just above with this layout and never shared.
            unsafe { dealloc(p as *mut u8, layout) };
        }
    }

    /// The array at word position `w`, for a draw that resolves its state once;
    /// None while it is unallocated (every entry reads zero).
    pub fn array(&self, w: usize) -> Option<WordArray<'_>> {
        let base = self.slots[w].load(Acquire);
        (!base.is_null()).then_some(WordArray { base, len: self.entries(), pitch: self.width, _store: PhantomData })
    }

    /// Word `w` of pixel (`x`, `y`): zero while its array is unallocated or
    /// the pixel is outside the managed area. A cold path.
    pub fn read(&self, x: u32, y: u32, w: usize) -> u32 {
        let Some((w, i)) = self.word_slot(x, y, w) else { return 0 };
        // SAFETY: `word_slot` bounds the index to the managed area.
        self.array(w).map_or(0, |a| unsafe { a.load(i) })
    }

    #[cfg(test)]
    /// Write word `w` of pixel (`x`, `y`), allocating its array. False outside
    /// the managed area or past the record. A cold path.
    pub fn write(&self, x: u32, y: u32, w: usize, value: u32) -> bool {
        let Some((w, i)) = self.word_slot(x, y, w) else { return false };
        self.ensure(w);
        // SAFETY: `word_slot` bounds the index to the managed area.
        unsafe { self.array(w).expect("ensured").store(i, value) };
        true
    }

    /// The word positions that have an array.
    pub fn allocated(&self) -> Vec<usize> {
        (0..POSITIONS).filter(|&w| !self.slots[w].load(Acquire).is_null()).collect()
    }

    /// A hash of the store's contents, to compare a store with a recorded or
    /// remembered one: FNV-1a over each allocated word position's index and
    /// entries. An unallocated position and an all-zero one differ, so replays
    /// that allocate the same arrays compare equal.
    pub fn checksum(&self) -> u64 {
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: u64| h = (h ^ v).wrapping_mul(PRIME);
        for w in self.allocated() {
            mix(w as u64 | 1 << 32);
            let a = self.array(w).expect("allocated");
            // SAFETY: `i` is below `len()`.
            (0..a.len()).for_each(|i| mix(unsafe { a.load(i) } as u64));
        }
        h
    }

    /// Zero every allocated array in place. Frees nothing and moves nothing,
    /// so a worker's or the display's pointer stays good. Each entry is a
    /// relaxed atomic store rather than a `memset`, which may write a word in
    /// pieces: a reader racing a reset must see the old value or zero, never a
    /// torn one. It is a cold path.
    pub fn reset(&self) {
        self.tex.reset();
        for w in self.allocated() {
            let a = self.array(w).expect("allocated");
            // SAFETY: the array has `a.len()` entries and is live for `&self`.
            let all = unsafe { std::slice::from_raw_parts(a.base, a.len()) };
            all.iter().for_each(|e| e.store(0, Relaxed));
        }
    }
}

impl Drop for PixelStore {
    fn drop(&mut self) {
        let n = self.entries();
        for s in &self.slots {
            let p = s.load(Relaxed);
            if !p.is_null() {
                // SAFETY: allocated in `ensure` with this layout and freed once,
                // here; `&mut self` means no draw or reader holds it.
                unsafe { dealloc(p as *mut u8, MemLayout::array::<AtomicU32>(n).expect("sized when allocated")) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::sync::Arc;

    #[test]
    fn nothing_is_allocated_until_a_word_is_written() {
        let s = PixelStore::new(64, 16);
        assert!(s.allocated().is_empty());
        assert_eq!(s.read(3, 4, 0), 0);
        assert!(s.allocated().is_empty(), "a read allocates nothing");
        assert!(s.write(3, 4, 5, 0xdead_beef));
        assert_eq!(s.allocated(), [5]);
        assert_eq!(s.read(3, 4, 5), 0xdead_beef);
        assert_eq!(s.read(4, 4, 5), 0);
        assert_eq!(s.read(3, 4, 6), 0);
    }

    #[test]
    fn word_slot_is_row_major_by_word_and_bounded() {
        let s = PixelStore::new(10, 4);
        assert_eq!(s.word_slot(0, 0, 0), Some((0, 0)));
        assert_eq!(s.word_slot(3, 2, 7), Some((7, 23)));
        assert_eq!(s.word_slot(10, 0, 0), None);
        assert_eq!(s.word_slot(0, 4, 0), None);
        assert_eq!(s.word_slot(0, 0, POSITIONS), None);
        assert!(!s.write(10, 0, 0, 1));
        assert_eq!(s.read(10, 0, 0), 0);
    }

    /// A draw resolves the base once and indexes from it; a clear is
    /// contiguous in one array.
    #[test]
    fn a_resolved_array_reads_and_writes_by_index() {
        let s = PixelStore::new(8, 8);
        for w in [1, 2] {
            s.ensure(w);
        }
        let a = s.array(1).unwrap();
        assert_eq!(a.pitch(), 8);
        for i in 0..64 {
            // SAFETY: i < 64 = len.
            unsafe { a.store(i, i as u32 + 1) };
        }
        assert_eq!(s.read(2, 3, 1), 3 * 8 + 2 + 1);
        assert_eq!(s.read(2, 3, 2), 0);
        assert!(s.array(3).is_none());
    }

    /// The pointer of an array never moves: a second `ensure` keeps it, and a
    /// reset zeroes in place.
    #[test]
    fn an_array_never_moves_and_reset_zeroes_in_place() {
        let s = PixelStore::new(4, 4);
        s.ensure(9);
        let before = s.array(9).unwrap().base;
        s.write(1, 1, 9, 7);
        s.ensure(9);
        assert_eq!(s.array(9).unwrap().base, before);
        s.reset();
        assert_eq!(s.array(9).unwrap().base, before);
        assert_eq!(s.read(1, 1, 9), 0);
        assert_eq!(s.allocated(), [9]);
    }

    /// A reader racing the first write sees zero or the value, never anything
    /// else, whatever the stage of the allocation. The reader runs until it
    /// has seen the value, with a bound on the iterations. A smoke check: the
    /// release and acquire pairing is argued in the module doc, and the test
    /// runs on whatever the host's memory model allows.
    #[test]
    fn a_reader_racing_a_first_write_sees_zero_or_the_value() {
        const VALUE: u32 = 0x1234_5678;
        for _ in 0..200 {
            let s = Arc::new(PixelStore::new(256, 64));
            let go = Arc::new(AtomicBool::new(false));
            let reader = {
                let (s, go) = (s.clone(), go.clone());
                std::thread::spawn(move || {
                    while !go.load(Relaxed) {
                        std::hint::spin_loop();
                    }
                    let mut seen = false;
                    for _ in 0..50_000_000u32 {
                        let v = s.read(200, 40, 3);
                        assert!(v == 0 || v == VALUE, "read {v:#x}");
                        if v == VALUE {
                            seen = true;
                            break;
                        }
                        std::thread::yield_now();
                    }
                    assert!(seen, "the value never showed");
                })
            };
            go.store(true, Relaxed);
            assert!(s.write(200, 40, 3, VALUE));
            reader.join().unwrap();
        }
    }

    /// A reset zeroes in place while a reader runs: the reader sees the value
    /// or zero, never a torn word, and its pointer stays good.
    #[test]
    fn a_reset_racing_a_reader_leaves_zero_or_the_value() {
        const VALUE: u32 = 0xa5a5_5a5a;
        let s = Arc::new(PixelStore::new(128, 128));
        for x in 0..128 {
            s.write(x, 7, 2, VALUE);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let passes = Arc::new(AtomicU64::new(0));
        let reader = {
            let (s, stop, passes) = (s.clone(), stop.clone(), passes.clone());
            std::thread::spawn(move || {
                while !stop.load(Relaxed) {
                    for x in 0..128 {
                        let v = s.read(x, 7, 2);
                        assert!(v == 0 || v == VALUE, "read {v:#x}");
                    }
                    passes.fetch_add(1, Relaxed);
                }
            })
        };
        for _ in 0..200 {
            s.reset();
            for x in 0..128 {
                s.write(x, 7, 2, VALUE);
            }
        }
        // The writer can finish before the reader thread is scheduled: let it make one pass before it is stopped.
        while passes.load(Relaxed) == 0 && !reader.is_finished() {
            std::thread::yield_now();
        }
        stop.store(true, Relaxed);
        reader.join().unwrap();
        assert!(passes.load(Relaxed) > 0);
    }

    #[test]
    fn the_checksum_follows_contents_and_allocation() {
        let a = PixelStore::new(8, 8);
        let b = PixelStore::new(8, 8);
        assert_eq!(a.checksum(), b.checksum());
        a.write(1, 1, 3, 5);
        assert_ne!(a.checksum(), b.checksum());
        b.write(1, 1, 3, 5);
        assert_eq!(a.checksum(), b.checksum());
        b.write(2, 2, 3, 0);
        assert_eq!(a.checksum(), b.checksum(), "writing a zero changes nothing");
        b.ensure(9);
        assert_ne!(a.checksum(), b.checksum(), "an allocated array of zeros counts");
        a.reset();
        assert_ne!(a.checksum(), PixelStore::new(8, 8).checksum());
    }

    /// Two threads allocating the same word position end with one array.
    #[test]
    fn two_threads_ensuring_one_word_share_one_array() {
        for _ in 0..100 {
            let s = Arc::new(PixelStore::new(32, 32));
            let ts: Vec<_> = (0..4)
                .map(|_| {
                    let s = s.clone();
                    std::thread::spawn(move || {
                        s.ensure(0);
                        s.array(0).unwrap().base as usize
                    })
                })
                .collect();
            let bases: Vec<_> = ts.into_iter().map(|t| t.join().unwrap()).collect();
            assert!(bases.windows(2).all(|p| p[0] == p[1]));
        }
    }
}
