
use crate::devlog::LogModule;
use super::bits::{be32, sextract32};
use super::board::{Board, ODSY_HEIGHT, ODSY_OBJECTS, ODSY_OBJ_POOL_BYTES, ODSY_OBJ_POOL_WORDS, ODSY_WIDTH};
use super::consts::{dma, tex};
use super::raster_unit::RASTER_JOB_OVERHEAD;
impl Board {
    /// The index of the object at board base base. A base names at most one
    /// live object (Board::object_select()), so the index's entry is the one a
    /// search of obj_base would find.
    pub fn object_find(&mut self, base: u32) -> Option<usize> {
        if base == 0 {
            return None;
        }
        match self.objects.index.get(&base) {
            Some(slot) if *slot != 0 => Some((*slot - 1) as usize),
            _ => None,
        }
    }

    /// Give object i the base base, keeping obj_index in step.
    pub fn object_set_base(&mut self, i: usize, base: u32) {
        if self.objects.base[i] != 0 {
            self.objects.index.remove(&self.objects.base[i]);
        }
        self.objects.base[i] = base;
        if base != 0 {
            self.objects.index.insert(base, i as u32 + 1);
        }
    }

    pub fn object_word_valid(&self, word: u64) -> bool {
        word < ODSY_OBJ_POOL_WORDS as u64 && (self.objects.valid[(word >> 3) as usize] & (1u8 << (word & 7))) != 0
    }

    pub fn object_word_set_valid(&mut self, word: u64, valid: bool) {
        let mask = 1u8 << (word & 7);

        assert!(word < ODSY_OBJ_POOL_WORDS as u64);
        if valid {
            self.objects.valid[(word >> 3) as usize] |= mask;
        } else {
            self.objects.valid[(word >> 3) as usize] &= !mask;
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

    pub fn object_region_written(&self, o: usize, x: u32, y: u32, width: u32, height: u32) -> bool {

        if x > self.objects.width[o] || width > self.objects.width[o] - x || y > self.objects.rows[o] || height > self.objects.rows[o] - y {
            return false;
        }
        for dy in 0..height {
            let row = self.objects.offset[o] as u64 + (y + dy) as u64 * self.objects.width[o] as u64 + x as u64;

            for dx in 0..width {
                if !self.object_word_valid(row + dx as u64) {
                    return false;
                }
            }
        }
        true
    }

    /// Mark object i as the most recently used. Use is an upload or a grow
    /// that targets the object, never a draw that samples it: a texture drawn
    /// every frame but uploaded once is the first to go when the store or
    /// the table fills. Model policy, not board behaviour.
    pub fn object_touch(&mut self, i: usize) {
        self.objects.clock += 1;
        self.objects.stamp[i] = self.objects.clock;
    }

    /// Forget object i: its slot stays, with base 0, for a later object.
    pub fn object_forget(&mut self, i: usize) {
        self.object_set_base(i, 0);
        self.objects.width[i] = 0;
        self.objects.rows[i] = 0;
        self.objects.offset[i] = 0;
        self.objects.stamp[i] = 0;
    }

    /// The least recently used live object other than keep.
    pub fn object_oldest(&mut self, keep: Option<usize>) -> Option<usize> {
        let mut oldest: Option<usize> = None;

        for i in 0..self.objects.count as usize {
            if Some(i) != keep && self.objects.base[i] != 0 && oldest.is_none_or(|o| self.objects.stamp[i] < self.objects.stamp[o]) {
                oldest = Some(i);
            }
        }
        oldest
    }

    /// Pack the objects that hold rows to the front of the store, in store order,
    /// so the space between them is reclaimed, then object last (when given)
    /// after them all, so that it can grow in place. Indices do not change.
    pub fn object_compact(&mut self, last: Option<usize>) {
        // A queued textured draw may still read the store.
        self.raster_wait();
        let last_size: u64 = last.map_or(0, |l| self.objects.width[l] as u64 * self.objects.rows[l] as u64);
        let mut held: Vec<u8> = Vec::new();
        let mut held_valid: Vec<bool> = Vec::new();
        let mut next: u64 = 0;

        // Packing the others may overwrite last's words: keep them aside.
        if last_size != 0 {
            let off = last.map_or(0, |l| self.objects.offset[l]) as u64;

            held = self.objects.mem.bytes[(off * 4) as usize..((off + last_size) * 4) as usize].to_vec();
            for k in 0..last_size {
                held_valid.push(self.object_word_valid(off + k));
            }
        }
        // The others in store order, once: each is then the lowest-placed
        // one not yet packed. One whose words a packed object now covers is
        // left where it is, as a search from `next` would leave it.
        let mut order: Vec<usize> = (0..self.objects.count as usize)
            .filter(|&i| Some(i) != last && self.objects.base[i] != 0 && self.objects.rows[i] != 0)
            .collect();
        order.sort_unstable_by_key(|&i| (self.objects.offset[i], i));
        for p in order {
            if (self.objects.offset[p] as u64) < next {
                continue;
            }
            let size = self.objects.width[p] as u64 * self.objects.rows[p] as u64;
            if self.objects.offset[p] as u64 != next {
                let from = self.objects.offset[p] as usize * 4;
                self.objects.mem.bytes.copy_within(from..from + size as usize * 4, next as usize * 4);
                self.object_valid_move(self.objects.offset[p] as u64, next, size);
                self.objects.offset[p] = next as u32;
            }
            next += size;
        }
        if let (Some(last), true) = (last, last_size != 0) {
            let at = next as usize * 4;
            self.objects.mem.bytes[at..at + held.len()].copy_from_slice(&held);
            for k in 0..last_size {
                self.object_word_set_valid(next + k, held_valid[k as usize]);
            }
            self.objects.offset[last] = next as u32;
            next += last_size;
        }
        // Growth in place assumes the words past the store's end are zero.
        let tail = (self.objects.used as u64).saturating_sub(next);
        let at = next as usize * 4;
        self.objects.mem.bytes[at..at + tail as usize * 4].fill(0);
        self.object_valid_range(next, tail, false);
        self.objects.used = next as u32;
    }

    /// The object at base, created with the given row length in pixels when it
    /// does not exist; a new row length for an existing object discards its
    /// storage (its layout changed). When the table is full, the least recently
    /// used object is forgotten, reported once.
    pub fn object_select(&mut self, base: u32, width: u32) -> usize {
        if let Some(i) = self.object_find(base) {
            if width != 0 && self.objects.width[i] != width {
                self.objects.width[i] = width;
                self.objects.rows[i] = 0;
            }
            self.object_touch(i);
            return i;
        }
        let free = self.objects.base[..self.objects.count as usize].iter().position(|&b| b == 0);
        let i = match free {
            Some(j) => j,
            None if self.objects.count == ODSY_OBJECTS => {
                // A full table has no free slot, so every entry is live.
                let i = self.object_oldest(None).expect("a full table has a live object");
                if !self.diag.obj_full_reported {
                    self.diag.obj_full_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: more than {} board-memory objects; the least recently used, {:#x}, is forgotten",
                        ODSY_OBJECTS,
                        self.objects.base[i]
                    );
                }
                self.object_forget(i);
                i
            }
            None => {
                self.objects.count += 1;
                self.objects.count as usize - 1
            }
        };
        self.object_set_base(i, base);
        self.objects.width[i] = width;
        self.objects.offset[i] = 0;
        self.objects.rows[i] = 0;
        self.object_touch(i);
        i
    }

    /// Make room for rows rows in object i: grow it in place when it is the last
    /// in the store, else move it to the end. When there is no room, forget the
    /// least recently used other objects until the live objects and i's new size
    /// fit, reporting the first object forgotten, then pack the store once with i
    /// last. False, reported once, when i alone does not fit.
    pub fn object_rows(&mut self, i: usize, rows: u32) -> bool {
        let iu = i;
        let pool = ODSY_OBJ_POOL_WORDS as u64;
        let need = self.objects.width[iu] as u64 * rows as u64;
        let have = self.objects.width[iu] as u64 * self.objects.rows[iu] as u64;

        if rows <= self.objects.rows[iu] {
            return true;
        }
        self.raster_wait();
        self.object_touch(i);
        if need <= pool
            && !(have != 0 && self.objects.offset[iu] as u64 + have == self.objects.used as u64 && self.objects.offset[iu] as u64 + need <= pool)
            && self.objects.used as u64 + need > pool
        {
            let mut live: u64 = 0;

            for j in 0..self.objects.count as usize {
                if j != iu && self.objects.base[j] != 0 {
                    live += self.objects.width[j] as u64 * self.objects.rows[j] as u64;
                }
            }
            while live + need > pool {
                let Some(v) = self.object_oldest(Some(i)) else {
                    break;
                };
                live -= self.objects.width[v] as u64 * self.objects.rows[v] as u64;
                if !self.diag.obj_forget_reported {
                    self.diag.obj_forget_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: the model's {} MiB board-memory store is full; the least recently used object, {:#x}, is forgotten",
                        ODSY_OBJ_POOL_BYTES / (1024 * 1024),
                        self.objects.base[v]
                    );
                }
                self.object_forget(v);
            }
            self.object_compact((have != 0).then_some(i));
        }
        if have != 0 && self.objects.offset[iu] as u64 + have == self.objects.used as u64 && self.objects.offset[iu] as u64 + need <= pool {
            self.objects.used = (self.objects.offset[iu] as u64 + need) as u32;
            self.object_valid_range(self.objects.offset[iu] as u64 + have, need - have, false);
        } else if self.objects.used as u64 + need <= pool {
            let new_offset = self.objects.used;
            let from = self.objects.offset[iu] as usize * 4;

            self.objects.mem.bytes.copy_within(from..from + have as usize * 4, new_offset as usize * 4);
            let z = (new_offset as usize + have as usize) * 4;
            self.objects.mem.bytes[z..z + (need - have) as usize * 4].fill(0);
            self.object_valid_move(self.objects.offset[iu] as u64, new_offset as u64, have);
            self.object_valid_range(new_offset as u64 + have, need - have, false);
            self.objects.offset[iu] = new_offset;
            self.objects.used += need as u32;
        } else {
            if !self.diag.obj_full_reported {
                self.diag.obj_full_reported = true;
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: board-memory object {:#x} ({} x {}) exceeds the model's {} MiB store",
                    self.objects.base[iu],
                    self.objects.width[iu],
                    rows,
                    ODSY_OBJ_POOL_BYTES / (1024 * 1024)
                );
            }
            return false;
        }
        self.objects.rows[iu] = rows;
        true
    }

    /// Object i's texel (x, y) as 0xRRGGBBAA; outside its rows or width, dropped.
    pub fn object_store(&mut self, i: usize, x: u32, y: u32, rgba: u32) {
        if x >= self.objects.width[i] || !self.object_rows(i, y.wrapping_add(1)) {
            return;
        }
        let word = self.objects.offset[i] as u64 + y as u64 * self.objects.width[i] as u64 + x as u64;
        self.raster_wait();
        self.objects.mem.bytes[(word * 4) as usize..][..4].copy_from_slice(&rgba.to_be_bytes());
        self.object_word_set_valid(word, true);
    }

    /// object_store() for a transfer the raster unit lands: the bookkeeping
    /// now (rows, validity), the word into `staged` for object_flush() to
    /// hand on. A queued textured draw may still read the texels this
    /// replaces, and the raster unit runs its jobs in order, so the write
    /// queues behind that draw instead of waiting for it.
    pub fn object_stage(&mut self, staged: &mut Vec<(u32, u32)>, i: usize, x: u32, y: u32, rgba: u32) {
        let iu = i;

        if x >= self.objects.width[iu] {
            return;
        }
        if y.wrapping_add(1) > self.objects.rows[iu] {
            // Growing the object may move it in the store, and the words
            // staged so far name their places as they are now.
            self.object_flush(staged);
            if !self.object_rows(i, y.wrapping_add(1)) {
                return;
            }
        }
        let word = self.objects.offset[iu] as u64 + y as u64 * self.objects.width[iu] as u64 + x as u64;
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

    pub fn object_load(&mut self, i: usize, x: u32, y: u32) -> u32 {
        let iu = i;

        if x >= self.objects.width[iu] || y >= self.objects.rows[iu] {
            return 0;
        }
        // A staged transfer may still be queued for the raster unit.
        self.raster_wait();
        be32(&self.objects.mem.bytes[((self.objects.offset[iu] as u64 + y as u64 * self.objects.width[iu] as u64 + x as u64) * 4) as usize..][..4])
    }

    /// A GL pixel copy (block 0x2031), decoded as its eight
    /// operands (CopyOperands): from the read drawable or an object,
    /// to the draw drawable or an object. Any other block is the X server's
    /// rectangle copy (Board::copy_rect()). False when an operand names an
    /// object that is not there or does not fit, so the block is reported.
    ///
    /// A copy from the window into an object lists the rectangle's rows
    /// bottom first: the library reads such an object top row first, with no
    /// bottom-up bit, and expects the lowest row of the rectangle first (the
    /// way a window read asks for it). OBSERVED in guest traces of
    /// glReadPixels of 16-bit types and glCopyTexImage, where the host got
    /// the rectangle's top row. INFERRED to be the board's copy rule: no
    /// known header bit selects it. The other directions
    /// keep the rows in the order they are held.
    ///
    /// The opposite copy, from an object to the window, lists the rows
    /// bottom first as well: the burst that filled the object put the
    /// image's first row (GL's lowest) first, and a window row grows
    /// downwards, so the object's first row lands on the rectangle's last.
    /// OBSERVED in a glDrawPixels of the signed types, whose raw burst goes
    /// to an object and then to the window: the host read back the rows in
    /// reverse (row 4's bytes where row 0's belonged, which differs from
    /// row 0's by 4 * 29 = 0x74 in the test pattern). INFERRED to be the
    /// board's copy rule, as above.
    ///
    /// The programmed post-filter (0x203b) is applied to the copied words
    /// (see copy_post_filter()); the identity leaves them alone.
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
        let from_window = c.src_base == dma::SENTINEL_READ;
        let to_window = c.dest_base == dma::SENTINEL_DRAW;
        let (w, h) = c.size;
        let src = if from_window { None } else { self.object_find(c.src_base) };
        let dest = if to_window { None } else { self.object_find(c.dest_base) };

        if (!from_window && src.is_none()) || (!to_window && dest.is_none()) {
            return false;
        }
        if let Some(d) = dest {
            if c.dest.0 as u64 + w as u64 > self.objects.width[d] as u64 || !self.object_rows(d, c.dest.1.wrapping_add(h)) {
                return false;
            }
        }
        if w == 0 || h == 0 {
            return true;
        }
        let mut words = Vec::with_capacity(w as usize * h as usize);

        for y in 0..h {
            for x in 0..w {
                words.push(if let Some(src) = src {
                    self.object_load(src, c.src.0 + x, c.src.1 + y)
                } else {
                    // Into an object the window's rows go bottom first.
                    let row = if dest.is_some() { h - 1 - y } else { y };
                    let (px, py) = ((c.src.0 + x) as i64 + self.win.read_origin[0] as i64, (c.src.1 + row) as i64 + self.win.read_origin[1] as i64);

                    if px < 0 || py < 0 || px >= ODSY_WIDTH as i64 || py >= ODSY_HEIGHT as i64 {
                        0
                    } else {
                        let buffer = self.read_buffer(c.read_front(), px, py);

                        self.buffer_load(buffer, px, py)
                    }
                });
            }
        }
        self.copy_post_filter(&mut words, Self::copy_reads_signed(from_window));
        // From an object to the window the rows go bottom first.
        let flip = !from_window && to_window;

        for y in 0..h {
            for x in 0..w {
                let word = words[(y * w + x) as usize];
                let y = if flip { h - 1 - y } else { y };

                if let Some(dest) = dest {
                    self.object_store(dest, c.dest.0 + x, c.dest.1 + y, word);
                } else {
                    self.paint_pixel(
                        (c.dest.0 + x) as i64 + self.win.origin[0] as i64,
                        (c.dest.1 + y) as i64 + self.win.origin[1] as i64,
                        word,
                    );
                }
            }
        }
        true
    }
}

impl Board {
    /// Whether the board reads the bytes of a copy's source as signed values
    /// when it filters them. INFERRED, and nothing board-visible establishes
    /// it: the library sends a signed type's burst and write code as the
    /// unsigned type's, and programs the final pass as ordinary scale and
    /// bias words. The pass the library builds for a signed type is 2x + 1/4096
    /// from an object the burst filled, which gives GL's result only if the
    /// bits are read as signed. So the model reads an object's bytes as signed
    /// and a window's as unsigned. This is the one place the choice is made.
    fn copy_reads_signed(from_window: bool) -> bool {
        !from_window
    }

    /// Apply the post-filter words (block 0x203b) to a copy's words, lane by
    /// lane in R, G, B, A order: the byte b of a lane becomes
    /// clamp(x * scale + bias, 0, 1) as a byte, with x = b / 255 (or, for a
    /// signed source, the byte as a signed value over 255), scale and bias
    /// signed 3.12 fixed point (ESTABLISHED as the layout of the words). The
    /// bias is used as programmed: the library halves the application's bias
    /// before it programs the step (ESTABLISHED), so
    /// the board does not halve again. That the filter governs a copy at all
    /// is INFERRED from the library's use of the block; the identity (scale
    /// 1.0, bias 0) leaves every word unchanged, as does a lane that is
    /// the identity.
    ///
    /// An alpha lane whose bias is full scale (0x1000) is the library's
    /// opaque overwrite for a source without alpha (ESTABLISHED): it stores
    /// 1.0, since the object's stored alpha is the model's 255, not source
    /// bits.
    pub fn copy_post_filter(&self, words: &mut [u32], signed: bool) {
        let sb = &self.tex.scale_bias;
        let lane = |c: usize| {
            (sextract32(sb[c * 2], 0, tex::SCALE_BIAS_BITS) as f64 / tex::SCALE_BIAS_ONE as f64,
             sextract32(sb[c * 2 + 1], 0, tex::SCALE_BIAS_BITS) as f64 / tex::SCALE_BIAS_ONE as f64)
        };

        if self.texture_scale_bias_identity() {
            return;
        }
        for word in words.iter_mut() {
            let mut out = *word;

            for c in 0..4usize {
                let (scale, bias) = lane(c);
                let shift = 24 - 8 * c as u32;
                let byte = (*word >> shift) as u8;
                let x = if signed { byte as i8 as f64 } else { byte as f64 } / 255.0;
                let v = if c == 3 && bias >= 1.0 { 1.0 } else { (x * scale + bias).clamp(0.0, 1.0) };

                out = (out & !(0xff << shift)) | (((v * 255.0 + 0.5).floor() as u32) << shift);
            }
            *word = out;
        }
    }
}

/// The eight operands of the GL pixel copy, block 0x2031.
///
/// Established as what the library sends; what the board does with the
/// overlap bits is inferred.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CopyOperands {
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
        self.read_header & dma::FMT_FRONT != 0
    }

    /// Write header bit 31: the destination x lies inside the source's x
    /// range; bit 30 the same for y. INFERRED to pick the copy direction.
    pub fn overlap(&self) -> (bool, bool) {
        (self.write_header & (1 << 31) != 0, self.write_header & (1 << 30) != 0)
    }
}
