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
        self.obj_mem.bytes[(word * 4) as usize..][..4].copy_from_slice(&rgba.to_be_bytes());
        self.object_word_set_valid(word, true);
    }

    pub fn object_load(&mut self, i: i32, x: u32, y: u32) -> u32 {
        let iu = i as usize;

        if x >= self.obj_width[iu] || y >= self.obj_rows[iu] {
            return 0;
        }
        u32::from_be_bytes(self.obj_mem.bytes[((self.obj_offset[iu] as u64 + y as u64 * self.obj_width[iu] as u64 + x as u64) * 4) as usize..][..4].try_into().unwrap())
    }

    /// Copy a recognised GL read-drawable rectangle into its scratch object.
    pub fn copy_rect_to_object(&mut self) -> bool {
        let w = self.copy_size & 0xffff;
        let h = self.copy_size >> 16;
        let dx = self.copy_dst & 0xffff;
        let dy = self.copy_dst >> 16;
        let sx = (self.copy_src & 0xffff) as i64 + self.read_origin[0] as i64;
        let sy = (self.copy_src >> 16) as i64 + self.read_origin[1] as i64;
        let fmt = self.cmd_operand[4];
        let tail = self.cmd_operand[7];
        let rgba8 = fmt == 0x92233002 && tail == 0x3801210b;
        let rgba16 = (fmt & !ODSY_DMA_FMT_FRONT) == 0x12633042 && (tail == 0x3802204b || tail == 0xf802204b);
        let obj = self.object_find(self.cmd_operand[5]);

        if self.cmd_operand[0] != ODSY_DMA_SENTINEL_READ
            || self.cmd_operand[2] != 0x80200050
            || (!rgba8 && !rgba16)
            || self.cmd_operand[5] != self.dest_base
            || obj < 0
            || w == 0
            || h == 0
            || dx as u64 + w as u64 > self.obj_width[obj as usize] as u64
            || sx < 0
            || sy < 0
            || sx + w as i64 > SGI_ODYSSEY_WIDTH as i64
            || sy + h as i64 > SGI_ODYSSEY_HEIGHT as i64
            || !self.object_rows(obj, dy.wrapping_add(h))
        {
            return false;
        }
        let displayed = self.front.lock().displayed_buffer;
        for y in 0..h {
            for x in 0..w {
                let shown = (displayed >> self.main_id((sx + x as i64) as u32, (sy + y as i64) as u32)) & 1;
                let source = if fmt & ODSY_DMA_FMT_FRONT != 0 { shown } else { shown ^ 1 };
                let px = self.buffer_load(source, sx + x as i64, sy + y as i64);

                self.object_store(obj, dx + x, dy + y, px);
            }
        }
        true
    }
}
