use super::*;

impl Board {
    /// The index of the object at board base base, or -1. A base names at most
    /// one live object (Board::object_select()), so the index's entry is the one
    /// a search of obj_base would find.
    pub fn object_find(&mut self, base: u32) -> i32 {
        if base == 0 {
            return -1;
        }
        match self.obj_index.get(&base) {
            Some(slot) if *slot != 0 => (*slot - 1) as i32,
            _ => -1,
        }
    }

    /// Give object i the base base, keeping obj_index in step.
    pub fn object_set_base(&mut self, i: i32, base: u32) {
        let i = i as usize;

        if self.obj_base[i] != 0 {
            self.obj_index.remove(&self.obj_base[i]);
        }
        self.obj_base[i] = base;
        if base != 0 {
            self.obj_index.insert(base, i as u32 + 1);
        }
    }

    pub fn object_word_valid(&self, word: u64) -> bool {
        word < ODSY_OBJ_POOL_WORDS as u64 && (self.obj_valid[(word >> 3) as usize] & (1u8 << (word & 7))) != 0
    }

    pub fn object_word_set_valid(&mut self, word: u64, valid: bool) {
        let mask = 1u8 << (word & 7);

        assert!(word < ODSY_OBJ_POOL_WORDS as u64);
        if valid {
            self.obj_valid[(word >> 3) as usize] |= mask;
        } else {
            self.obj_valid[(word >> 3) as usize] &= !mask;
        }
    }

    pub fn object_valid_range(&mut self, first: u64, count: u64, valid: bool) {
        for i in 0..count {
            self.object_word_set_valid(first + i, valid);
        }
    }

    pub fn object_valid_move(&mut self, source: u64, dest: u64, count: u64) {
        if dest > source {
            for i in (1..=count).rev() {
                let v = self.object_word_valid(source + i - 1);

                self.object_word_set_valid(dest + i - 1, v);
            }
        } else {
            for i in 0..count {
                let v = self.object_word_valid(source + i);

                self.object_word_set_valid(dest + i, v);
            }
        }
    }

    pub fn object_region_written(&self, object: i32, x: u32, y: u32, width: u32, height: u32) -> bool {
        let o = object as usize;

        if x > self.obj_width[o] || width > self.obj_width[o] - x || y > self.obj_rows[o] || height > self.obj_rows[o] - y {
            return false;
        }
        for dy in 0..height {
            let row = self.obj_offset[o] as u64 + (y + dy) as u64 * self.obj_width[o] as u64 + x as u64;

            for dx in 0..width {
                if !self.object_word_valid(row + dx as u64) {
                    return false;
                }
            }
        }
        true
    }

    /// Mark object i as the most recently used.
    pub fn object_touch(&mut self, i: i32) {
        self.obj_clock += 1;
        self.obj_stamp[i as usize] = self.obj_clock;
    }

    /// Forget object i: its slot stays, with base 0, for a later object.
    pub fn object_forget(&mut self, i: i32) {
        self.object_set_base(i, 0);
        let i = i as usize;
        self.obj_width[i] = 0;
        self.obj_rows[i] = 0;
        self.obj_offset[i] = 0;
        self.obj_stamp[i] = 0;
    }

    /// The least recently used live object other than keep, or -1.
    pub fn object_oldest(&mut self, keep: i32) -> i32 {
        let mut oldest: i32 = -1;

        for i in 0..self.obj_count as usize {
            if i as i32 != keep && self.obj_base[i] != 0 && (oldest < 0 || self.obj_stamp[i] < self.obj_stamp[oldest as usize]) {
                oldest = i as i32;
            }
        }
        oldest
    }

    /// Pack the objects that hold rows to the front of the store, in store order,
    /// so the space between them is reclaimed, then object last (when not -1)
    /// after them all, so that it can grow in place. Indices do not change.
    pub fn object_compact(&mut self, last: i32) {
        // A queued textured draw may still read the store.
        self.raster_wait();
        let last_size: u64 = if last >= 0 { self.obj_width[last as usize] as u64 * self.obj_rows[last as usize] as u64 } else { 0 };
        let mut held: Vec<u8> = Vec::new();
        let mut held_valid: Vec<bool> = Vec::new();
        let mut next: u64 = 0;

        // Packing the others may overwrite last's words: keep them aside.
        if last_size != 0 {
            let off = self.obj_offset[last as usize] as u64;

            held = self.obj_mem.bytes[(off * 4) as usize..((off + last_size) * 4) as usize].to_vec();
            for k in 0..last_size {
                held_valid.push(self.object_word_valid(off + k));
            }
        }
        loop {
            let mut pick: i32 = -1;

            // The lowest-placed object not yet packed: at or past next.
            for i in 0..self.obj_count as usize {
                if i as i32 != last
                    && self.obj_base[i] != 0
                    && self.obj_rows[i] != 0
                    && self.obj_offset[i] as u64 >= next
                    && (pick < 0 || self.obj_offset[i] < self.obj_offset[pick as usize])
                {
                    pick = i as i32;
                }
            }
            if pick < 0 {
                break;
            }
            let p = pick as usize;
            let size = self.obj_width[p] as u64 * self.obj_rows[p] as u64;
            if self.obj_offset[p] as u64 != next {
                let from = self.obj_offset[p] as usize * 4;
                self.obj_mem.bytes.copy_within(from..from + size as usize * 4, next as usize * 4);
                self.object_valid_move(self.obj_offset[p] as u64, next, size);
                self.obj_offset[p] = next as u32;
            }
            next += size;
        }
        if last_size != 0 {
            let at = next as usize * 4;
            self.obj_mem.bytes[at..at + held.len()].copy_from_slice(&held);
            for k in 0..last_size {
                self.object_word_set_valid(next + k, held_valid[k as usize]);
            }
            self.obj_offset[last as usize] = next as u32;
            next += last_size;
        }
        // Growth in place assumes the words past the store's end are zero.
        let tail = (self.obj_used as u64).saturating_sub(next);
        let at = next as usize * 4;
        self.obj_mem.bytes[at..at + tail as usize * 4].fill(0);
        self.object_valid_range(next, tail, false);
        self.obj_used = next as u32;
    }

    /// The object at base, created with the given row length in pixels when it
    /// does not exist; a new row length for an existing object discards its
    /// storage (its layout changed). When the table is full, the least recently
    /// used object is forgotten, reported once.
    pub fn object_select(&mut self, base: u32, width: u32) -> i32 {
        let mut i = self.object_find(base);

        if i < 0 {
            if let Some(j) = self.obj_base[..self.obj_count as usize].iter().position(|&b| b == 0) {
                i = j as i32;
            }
        }
        if i < 0 && self.obj_count == SGI_ODYSSEY_OBJECTS {
            i = self.object_oldest(-1);
            if !self.obj_full_reported {
                self.obj_full_reported = true;
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: more than {} board-memory objects; the least recently used, {:#x}, is forgotten",
                    SGI_ODYSSEY_OBJECTS,
                    self.obj_base[i as usize]
                );
            }
            self.object_forget(i);
        }
        if i < 0 || self.obj_base[i as usize] == 0 {
            if i < 0 {
                i = self.obj_count as i32;
                self.obj_count += 1;
            }
            self.object_set_base(i, base);
            self.obj_width[i as usize] = width;
            self.obj_offset[i as usize] = 0;
            self.obj_rows[i as usize] = 0;
        } else if width != 0 && self.obj_width[i as usize] != width {
            self.obj_width[i as usize] = width;
            self.obj_rows[i as usize] = 0;
        }
        self.object_touch(i);
        i
    }

    /// Make room for rows rows in object i: grow it in place when it is the last
    /// in the store, else move it to the end. When there is no room, forget the
    /// least recently used other objects until the live objects and i's new size
    /// fit, reporting the first object forgotten, then pack the store once with i
    /// last. False, reported once, when i alone does not fit.
    pub fn object_rows(&mut self, i: i32, rows: u32) -> bool {
        let iu = i as usize;
        let pool = ODSY_OBJ_POOL_WORDS as u64;
        let need = self.obj_width[iu] as u64 * rows as u64;
        let have = self.obj_width[iu] as u64 * self.obj_rows[iu] as u64;

        if rows <= self.obj_rows[iu] {
            return true;
        }
        self.raster_wait();
        self.object_touch(i);
        if need <= pool
            && !(have != 0 && self.obj_offset[iu] as u64 + have == self.obj_used as u64 && self.obj_offset[iu] as u64 + need <= pool)
            && self.obj_used as u64 + need > pool
        {
            let mut live: u64 = 0;

            for j in 0..self.obj_count as usize {
                if j != iu && self.obj_base[j] != 0 {
                    live += self.obj_width[j] as u64 * self.obj_rows[j] as u64;
                }
            }
            while live + need > pool {
                let victim = self.object_oldest(i);

                if victim < 0 {
                    break;
                }
                let v = victim as usize;
                live -= self.obj_width[v] as u64 * self.obj_rows[v] as u64;
                if !self.obj_forget_reported {
                    self.obj_forget_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: the model's {} MiB board-memory store is full; the least recently used object, {:#x}, is forgotten",
                        ODSY_OBJ_POOL_BYTES / (1024 * 1024),
                        self.obj_base[v]
                    );
                }
                self.object_forget(victim);
            }
            self.object_compact(if have != 0 { i } else { -1 });
        }
        if have != 0 && self.obj_offset[iu] as u64 + have == self.obj_used as u64 && self.obj_offset[iu] as u64 + need <= pool {
            self.obj_used = (self.obj_offset[iu] as u64 + need) as u32;
            self.object_valid_range(self.obj_offset[iu] as u64 + have, need - have, false);
        } else if self.obj_used as u64 + need <= pool {
            let new_offset = self.obj_used;
            let from = self.obj_offset[iu] as usize * 4;

            self.obj_mem.bytes.copy_within(from..from + have as usize * 4, new_offset as usize * 4);
            let z = (new_offset as usize + have as usize) * 4;
            self.obj_mem.bytes[z..z + (need - have) as usize * 4].fill(0);
            self.object_valid_move(self.obj_offset[iu] as u64, new_offset as u64, have);
            self.object_valid_range(new_offset as u64 + have, need - have, false);
            self.obj_offset[iu] = new_offset;
            self.obj_used += need as u32;
        } else {
            if !self.obj_full_reported {
                self.obj_full_reported = true;
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: board-memory object {:#x} ({} x {}) exceeds the model's {} MiB store",
                    self.obj_base[iu],
                    self.obj_width[iu],
                    rows,
                    ODSY_OBJ_POOL_BYTES / (1024 * 1024)
                );
            }
            return false;
        }
        self.obj_rows[iu] = rows;
        true
    }

    /// Object i's texel (x, y) as 0xRRGGBBAA; outside its rows or width, dropped.
    pub fn object_store(&mut self, i: i32, x: u32, y: u32, rgba: u32) {
        if x >= self.obj_width[i as usize] || !self.object_rows(i, y.wrapping_add(1)) {
            return;
        }
        let word = self.obj_offset[i as usize] as u64 + y as u64 * self.obj_width[i as usize] as u64 + x as u64;
        self.raster_wait();
        self.obj_mem.bytes[(word * 4) as usize..][..4].copy_from_slice(&rgba.to_be_bytes());
        self.object_word_set_valid(word, true);
    }

    /// object_store() for a transfer the raster unit lands: the bookkeeping
    /// now (rows, validity), the word into `staged` for object_flush() to
    /// hand on. A queued textured draw may still read the texels this
    /// replaces, and the raster unit runs its jobs in order, so the write
    /// queues behind that draw instead of waiting for it.
    pub fn object_stage(&mut self, staged: &mut Vec<(u32, u32)>, i: i32, x: u32, y: u32, rgba: u32) {
        let iu = i as usize;

        if x >= self.obj_width[iu] {
            return;
        }
        if y.wrapping_add(1) > self.obj_rows[iu] {
            // Growing the object may move it in the store, and the words
            // staged so far name their places as they are now.
            self.object_flush(staged);
            if !self.object_rows(i, y.wrapping_add(1)) {
                return;
            }
        }
        let word = self.obj_offset[iu] as u64 + y as u64 * self.obj_width[iu] as u64 + x as u64;
        self.object_word_set_valid(word, true);
        staged.push((word as u32, rgba));
    }

    /// Hand the staged object words to the raster unit, behind the jobs
    /// queued before them.
    pub fn object_flush(&mut self, staged: &mut Vec<(u32, u32)>) {
        if staged.is_empty() {
            return;
        }
        let words = std::mem::take(staged);
        let spans = self.plane_spans();
        self.raster.submit_work(RASTER_JOB_OVERHEAD.saturating_add(words.len() as u32), move |_| {
            // SAFETY: the store outlives the job, and the front end waits
            // for the raster unit before it touches the store (PlaneSpans).
            // The whole PlaneSpans moves in, which is Send; its pointer alone
            // is not.
            let spans = spans;
            let bytes = unsafe { spans.obj.bytes() };
            for (word, rgba) in words {
                bytes[word as usize * 4..][..4].copy_from_slice(&rgba.to_be_bytes());
            }
        });
    }

    pub fn object_load(&mut self, i: i32, x: u32, y: u32) -> u32 {
        let iu = i as usize;

        if x >= self.obj_width[iu] || y >= self.obj_rows[iu] {
            return 0;
        }
        // A staged transfer may still be queued for the raster unit.
        self.raster_wait();
        u32::from_be_bytes(self.obj_mem.bytes[((self.obj_offset[iu] as u64 + y as u64 * self.obj_width[iu] as u64 + x as u64) * 4) as usize..][..4].try_into().unwrap())
    }

    /// A GL pixel copy (block 0x2031), decoded as the datasheet's "Copy
    /// words" gives its eight operands: from the read drawable or an object,
    /// to the draw drawable or an object. Any other block is the X server's
    /// rectangle copy (Board::copy_rect()). False when an operand names an
    /// object that is not there or does not fit, so the block is reported.
    ///
    /// The source is staged first, so an overlapping copy needs no direction.
    /// Words move as the planes hold them: a colour pixel keeps its alpha and
    /// an index copies as the raw index. The read header's class, code and
    /// fields and the write header's format and overlap bits say how the board
    /// converts and in which order it walks; the model's planes need neither.
    /// A window source takes the buffer the header's front bit names (see
    /// dma_read_pixel()), and a window destination the buffers the drawing
    /// mask selects, through the window's clip rectangles and id test.
    pub fn copy_rect_gl(&mut self, c: &CopyOperands) -> bool {
        let from_window = c.src_base == ODSY_DMA_SENTINEL_READ;
        let to_window = c.dest_base == ODSY_DMA_SENTINEL_DRAW;
        let (w, h) = c.size;
        let src = if from_window { -1 } else { self.object_find(c.src_base) };
        let dest = if to_window { -1 } else { self.object_find(c.dest_base) };

        if (!from_window && src < 0) || (!to_window && dest < 0) {
            return false;
        }
        if dest >= 0 && (c.dest.0 as u64 + w as u64 > self.obj_width[dest as usize] as u64 || !self.object_rows(dest, c.dest.1.wrapping_add(h))) {
            return false;
        }
        if w == 0 || h == 0 {
            return true;
        }
        let mut words = Vec::with_capacity(w as usize * h as usize);

        for y in 0..h {
            for x in 0..w {
                words.push(if from_window {
                    let (px, py) = ((c.src.0 + x) as i64 + self.read_origin[0] as i64, (c.src.1 + y) as i64 + self.read_origin[1] as i64);

                    if px < 0 || py < 0 || px >= SGI_ODYSSEY_WIDTH as i64 || py >= SGI_ODYSSEY_HEIGHT as i64 {
                        0
                    } else {
                        let buffer = self.read_buffer(c.read_front(), px, py);

                        self.buffer_load(buffer, px, py)
                    }
                } else {
                    self.object_load(src, c.src.0 + x, c.src.1 + y)
                });
            }
        }
        for y in 0..h {
            for x in 0..w {
                let word = words[(y * w + x) as usize];

                if to_window {
                    self.paint_pixel(
                        (c.dest.0 + x) as i64 + self.win_origin[0] as i64,
                        (c.dest.1 + y) as i64 + self.win_origin[1] as i64,
                        word,
                    );
                } else {
                    self.object_store(dest, c.dest.0 + x, c.dest.1 + y, word);
                }
            }
        }
        true
    }
}

/// The eight operands of the GL pixel copy, block 0x2031.
///
/// Datasheet: "Pixel transfer headers from the OpenGL library", "Copy
/// words" (established as what the library sends; what the board does with
/// the overlap bits is inferred).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyOperands {
    /// Word 0: 0x30000019 for the read drawable, else an object's address.
    pub src_base: u32,
    /// Word 1: source x and y.
    pub src: (u32, u32),
    /// Word 2 bit 31: the destination is the front buffer.
    pub dest_front: bool,
    /// Word 2 bits 27:16 and 11:0: the destination's and the source's
    /// 12-bit drawable words.
    pub dest_drawable: u32,
    pub src_drawable: u32,
    /// Word 2 bits 15:12: the destination's second field code.
    pub dest_field: u32,
    /// Word 3: width and height.
    pub size: (u32, u32),
    /// Word 4: the read header's word 4 (class, code, fields, front).
    pub read_header: u32,
    /// Word 5: 0x30000010 for the draw drawable, else an object's address.
    pub dest_base: u32,
    /// Word 6: destination x and y.
    pub dest: (u32, u32),
    /// Word 7: the write header's word 7: format and type, expand, size.
    pub write_header: u32,
}

impl CopyOperands {
    pub fn decode(op: &[u32]) -> Self {
        CopyOperands {
            src_base: op[0],
            src: (op[1] & 0xffff, op[1] >> 16),
            dest_front: op[2] & (1 << 31) != 0,
            dest_drawable: (op[2] >> 16) & 0xfff,
            src_drawable: op[2] & 0xfff,
            dest_field: (op[2] >> 12) & 0xf,
            size: (op[3] & 0xffff, op[3] >> 16),
            read_header: op[4],
            dest_base: op[5],
            dest: (op[6] & 0xffff, op[6] >> 16),
            write_header: op[7],
        }
    }

    /// The read header's front bit: logical front of the read buffer.
    pub fn read_front(&self) -> bool {
        self.read_header & ODSY_DMA_FMT_FRONT != 0
    }

    /// Write header bit 31: the destination x lies inside the source's x
    /// range; bit 30 the same for y. INFERRED to pick the copy direction.
    pub fn overlap(&self) -> (bool, bool) {
        (self.write_header & (1 << 31) != 0, self.write_header & (1 << 30) != 0)
    }
}
