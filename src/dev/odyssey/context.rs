use super::*;

// Anonymous enum: current-value query kinds.
pub const ODSY_QUERY_INDEX: u32 = 0;

pub const ODSY_QUERY_COLOUR: u32 = 1;

pub const ODSY_QUERY_NORMAL: u32 = 2;

pub const ODSY_QUERY_TEXCOORD: u32 = 3;

pub const ODSY_QUERY_EDGE: u32 = 4;

impl Board {
    pub fn back_material_reset(&mut self) {
        self.gl_back_material.copy_from_slice(&BACK_MATERIAL_DEFAULTS);
        self.gl_back_shininess_sqrt = 0;
    }

    pub fn shadow_put(&mut self, base: u64, off: u32, word: u32) {
        let at = base.wrapping_add(off as u64);
        if !write_guest(&self.guest, at, &word.to_be_bytes()) && !self.ctx_shadow_fail_reported {
            self.ctx_shadow_fail_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: context shadow store to {:#x} failed", at);
        }
    }

    /// One selected ring entry: attribute a's word and its raw operands.
    pub fn shadow_ring(&mut self, base: u64, a: u32, slot: u32, word: u32, ops: &[u32]) {
        self.shadow_put(base, ODSY_HW_RING_WORD + ODSY_HW_RING_WORD_STRIDE * a + 4 * slot, word);
        for (i, op) in ops.iter().enumerate() {
            self.shadow_put(
                base,
                ODSY_HW_RING_OPERANDS + ODSY_HW_RING_OPERAND_STRIDE * a + 16 * slot + 4 * i as u32,
                *op,
            );
        }
    }

    /// One two-word descriptor of an open-primitive save. Attribute A's 2-bit
    /// ring-slot selector is at bit 2 * (16 - A) of the first word for A 1..16
    /// and at bit 2 * (32 - A) of the second for A 17 and 18. An attribute the
    /// model keeps no record of (record[A][0] zero) selects slot 0, which this
    /// save never writes for it, so the replay sends back what that slot already
    /// holds. This is an approximation: the board's values for these attributes,
    /// and the slots it saves them in, are unknown.
    pub fn shadow_descriptor(&mut self, base: u64, off: u32, record: &[[u32; 5]; 19], slot: u32) {
        let mut word = [0u32; 2];

        for a in 1..19usize {
            if record[a][0] == 0 {
                continue;
            }
            if a <= 16 {
                word[0] |= slot << (2 * (16 - a));
            } else {
                word[1] |= slot << (2 * (32 - a));
            }
        }
        self.shadow_put(base, off, word[0]);
        self.shadow_put(base, off + 4, word[1]);
    }

    pub fn open_current_records(&mut self, record: &mut [[u32; 5]; 19]) {
        const MATERIAL: [u32; 4] = [
            ODSY_GL_MATERIAL_AMBIENT,
            ODSY_GL_MATERIAL_DIFFUSE,
            ODSY_GL_MATERIAL_SPECULAR,
            ODSY_GL_MATERIAL_EMISSION,
        ];
        const BACK_MATERIAL: [u32; 4] = [ODSY_GL_BACK_AMBIENT, ODSY_GL_BACK_DIFFUSE, ODSY_GL_BACK_SPECULAR, ODSY_GL_BACK_EMISSION];

        *record = [[0u32; 5]; 19];
        record[ODSY_HW_ATTR_INDEX] = self.query_current[ODSY_QUERY_INDEX as usize];
        record[ODSY_HW_ATTR_COLOUR] = self.query_current[ODSY_QUERY_COLOUR as usize];
        record[ODSY_HW_ATTR_NORMAL] = self.query_current[ODSY_QUERY_NORMAL as usize];
        record[ODSY_HW_ATTR_TEXTURE] = self.query_current[ODSY_QUERY_TEXCOORD as usize];
        record[ODSY_HW_ATTR_EDGE] = self.query_current[ODSY_QUERY_EDGE as usize];
        for i in 0..4usize {
            record[ODSY_HW_ATTR_MATERIAL + i][0] = MATERIAL[i];
            record[ODSY_HW_ATTR_MATERIAL + i][1..5].copy_from_slice(&self.gl_material[4 * i..4 * i + 4]);
            record[ODSY_HW_ATTR_BACK_MATERIAL + i][0] = BACK_MATERIAL[i];
            record[ODSY_HW_ATTR_BACK_MATERIAL + i][1..5].copy_from_slice(&self.gl_back_material[4 * i..4 * i + 4]);
        }
        record[ODSY_HW_ATTR_SHININESS][0] = ODSY_GL_SHININESS_SQRT;
        record[ODSY_HW_ATTR_SHININESS][1] = self.gl_shininess_sqrt;
        record[ODSY_HW_ATTR_BACK_SHININESS][0] = ODSY_GL_BACK_SHININESS_SQRT;
        record[ODSY_HW_ATTR_BACK_SHININESS][1] = self.gl_back_shininess_sqrt;
    }

    pub fn open_record_vertex(&mut self, word: u32) {
        let mut v = OpenVertex { word, slot: self.gl_open_r, ..Default::default() };
        let mut n = self.gl_open_held as usize;
        let arity = (cmd_arity(word) as usize).min(4);

        v.operands[..arity].copy_from_slice(&self.cmd_operand[..arity]);
        self.open_current_records(&mut v.attribute);
        if self.gl_primitive == 2 || self.gl_primitive == 6 {
            self.gl_open_r = if self.gl_open_r == 3 { 1 } else { self.gl_open_r + 1 };
        } else {
            self.gl_open_r = (self.gl_open_r + 1) & 3;
        }
        self.gl_open_c = (self.gl_open_c + 1).min(7);
        self.gl_open_m3 = (self.gl_open_m3 + 1) % 3;
        self.gl_open_m4 = (self.gl_open_m4 + 1) & 3;

        match self.gl_primitive {
            0 => {
                // Points retain no vertices.
                n = 0;
            }
            1 => {
                // Lines retain one of each pair.
                if n != 0 {
                    n = 0;
                } else {
                    self.gl_open_vertex[0] = v;
                    n = 1;
                }
            }
            2 | 6 => {
                // Anchor and latest vertex.
                self.gl_open_vertex[if n != 0 { 1 } else { 0 }] = v;
                n = (n + 1).min(2);
            }
            3 => {
                // Line strip.
                self.gl_open_vertex[0] = v;
                n = 1;
            }
            4 | 7 => {
                // Triangles and quads.
                if n == (if self.gl_primitive == 4 { 2 } else { 3 }) {
                    n = 0;
                } else {
                    self.gl_open_vertex[n] = v;
                    n += 1;
                }
            }
            5 => {
                // Triangle strip.
                if n == 2 {
                    self.gl_open_vertex[0] = self.gl_open_vertex[1];
                    self.gl_open_vertex[1] = v;
                } else {
                    self.gl_open_vertex[n] = v;
                    n += 1;
                }
            }
            8 => {
                // Quad strip.
                if n == 3 {
                    self.gl_open_vertex[0] = self.gl_open_vertex[2];
                    self.gl_open_vertex[1] = v;
                    n = 2;
                } else {
                    self.gl_open_vertex[n] = v;
                    n += 1;
                }
            }
            _ => {}
        }
        self.gl_open_held = n as u32;
    }

    pub fn restart_finish(&mut self) {
        let n = self.gl_open_held;
        let mut count;

        if !self.gl_restart_armed || self.gl_restart_absorb != 0 {
            return;
        }
        if self.gl_restart_m3 >= 3 || n != self.gl_restart_total {
            self.cmd_unimplemented(0x741);
            self.gl_restart_armed = false;
            return;
        }
        self.gl_open_c = if self.gl_restart_c == 7 { 7 } else { (self.gl_restart_c + n).min(7) };
        self.gl_open_m3 = (self.gl_restart_m3 + n) % 3;
        self.gl_open_m4 = (self.gl_restart_m4 + n) & 3;
        count = self.gl_open_c;
        if count == 7 {
            while (count % 3 != self.gl_open_m3 || count % 4 != self.gl_open_m4) && count < 19 {
                count += 1;
            }
        }
        self.gl_in_prim = self.gl_ctx.resume_primitive(self.gl_restart_type, Some(&self.gl_restart_vertex[..]), n, count);
        self.gl_restart_armed = false;
    }

    /// The board's save of the outgoing context's geometry-engine state into the
    /// hardware shadow, before the completion word. Register state needs no save:
    /// the kernel re-pushes it from the owner's software shadow on every
    /// switch-in, so the model keeps its single copy. What the kernel cannot
    /// restore itself is the current vertex attributes, which it replays from this
    /// page. Each record goes out as the last command word it retained with its
    /// raw operands; nothing is converted between a setter, this save and a later
    /// query. This is an approximation: the physical board's saved words and
    /// operand encodings are unknown. Every save writes all ten descriptor words.
    /// Idle selects ring slot 0. A float primitive of type 0..8 packs its retained
    /// raw attribute records into ring slots 0..N-1 at save time, and current
    /// records into slot 3. Integer 2D, type 10 and type 9 polygon assembly still
    /// report an unsupported switch and save idle. This producer and its restart
    /// decode are model approximations, not established board behaviour.
    pub fn ctx_save(&mut self, base: u64) {
        const MATERIAL: [u32; 4] = [
            ODSY_GL_MATERIAL_AMBIENT,
            ODSY_GL_MATERIAL_DIFFUSE,
            ODSY_GL_MATERIAL_SPECULAR,
            ODSY_GL_MATERIAL_EMISSION,
        ];
        const BACK_MATERIAL: [u32; 4] = [ODSY_GL_BACK_AMBIENT, ODSY_GL_BACK_DIFFUSE, ODSY_GL_BACK_SPECULAR, ODSY_GL_BACK_EMISSION];
        let index = self.query_current[ODSY_QUERY_INDEX as usize];
        let colour = self.query_current[ODSY_QUERY_COLOUR as usize];
        let normal = self.query_current[ODSY_QUERY_NORMAL as usize];
        let edge = self.query_current[ODSY_QUERY_EDGE as usize][0];
        let open = self.gl_begin_open && !self.gl_integer_prim && self.gl_primitive <= 8;
        let ar = |w: u32| (cmd_arity(w) as usize).min(4);

        if self.gl_begin_open && !open && !self.ctx_open_prim_logged {
            self.ctx_open_prim_logged = true;
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-odyssey: context switch inside primitive {:#x}; its buffered vertices are not saved",
                self.gl_primitive
            );
        }
        self.shadow_put(base, ODSY_HW_PIPE_STATE, ODSY_HW_PIPE_IDLE);
        for i in 0..ODSY_HW_DESCRIPTOR_WORDS {
            self.shadow_put(base, ODSY_HW_CURRENT_ATTRIBUTES + 4 * i, 0);
        }
        self.shadow_put(base, ODSY_HW_INDEX_WORD, index[0]);
        for i in 0..cmd_arity(index[0]) {
            self.shadow_put(base, ODSY_HW_INDEX_OPERANDS + 4 * i, index[1 + (i as usize).min(3)]);
        }
        self.shadow_put(base, ODSY_HW_COLOUR_WORD, colour[0]);
        for i in 0..cmd_arity(colour[0]) {
            self.shadow_put(base, ODSY_HW_COLOUR_OPERANDS + 4 * i, colour[1 + (i as usize).min(3)]);
        }
        self.shadow_ring(base, ODSY_HW_ATTR_INDEX as u32, 0, index[0], &index[1..1 + ar(index[0])]);
        self.shadow_ring(base, ODSY_HW_ATTR_COLOUR as u32, 0, colour[0], &colour[1..1 + ar(colour[0])]);
        self.shadow_ring(base, ODSY_HW_ATTR_NORMAL as u32, 0, normal[0], &normal[1..1 + ar(normal[0])]);
        self.shadow_ring(base, ODSY_HW_ATTR_EDGE as u32, 0, edge, &[]);
        if self.tex_coord_q != ODSY_F32_ONE {
            let coordinate = [self.tex_coord[0], self.tex_coord[1], self.tex_coord_r, self.tex_coord_q];

            self.shadow_ring(base, ODSY_HW_ATTR_TEXTURE as u32, 0, ODSY_GL_TEXCOORD_4F, &coordinate);
        } else {
            let three = (self.tex_mode & ODSY_TEX_MODE_3D) != 0 || self.tex_coord_r != 0;
            let three_ops = [self.tex_coord[0], self.tex_coord[1], self.tex_coord_r];
            let two_ops = self.tex_coord;

            self.shadow_ring(
                base,
                ODSY_HW_ATTR_TEXTURE as u32,
                0,
                if three { ODSY_GL_TEXCOORD_3F } else { ODSY_GL_TEXCOORD_2F },
                if three { &three_ops[..] } else { &two_ops[..] },
            );
        }
        for i in 0..4usize {
            let m: [u32; 4] = self.gl_material[i * 4..i * 4 + 4].try_into().unwrap();

            self.shadow_ring(base, (ODSY_HW_ATTR_MATERIAL + i) as u32, 0, MATERIAL[i], &m);
        }
        let sh = [self.gl_shininess_sqrt];
        self.shadow_ring(base, ODSY_HW_ATTR_SHININESS as u32, 0, ODSY_GL_SHININESS_SQRT, &sh);
        for i in 0..4usize {
            let m: [u32; 4] = self.gl_back_material[i * 4..i * 4 + 4].try_into().unwrap();

            self.shadow_ring(base, (ODSY_HW_ATTR_BACK_MATERIAL + i) as u32, 0, BACK_MATERIAL[i], &m);
        }
        let bsh = [self.gl_back_shininess_sqrt];
        self.shadow_ring(base, ODSY_HW_ATTR_BACK_SHININESS as u32, 0, ODSY_GL_BACK_SHININESS_SQRT, &bsh);
        if open {
            let mut current = [[0u32; 5]; 19];
            let mut state = self.gl_primitive
                | (self.gl_open_held << 4)
                | (self.gl_open_r << 6)
                | (self.gl_open_m3 << 12)
                | (self.gl_open_m4 << 14)
                | (self.gl_open_c << 21);

            // After c >= 2 vertices the next triangle is number c - 2.
            if self.gl_primitive == ODSY_GL_TRIANGLE_STRIP && self.gl_open_c >= 2 && (self.gl_open_m4 & 1) != 0 {
                state |= ODSY_HW_PIPE_STRIP_ODD;
            }

            self.open_current_records(&mut current);
            self.shadow_put(base, ODSY_HW_PIPE_STATE, state);
            for i in 0..self.gl_open_held.min(3) {
                let v = self.gl_open_vertex[i as usize];
                let slot = v.slot;

                self.shadow_ring(base, 0, slot, v.word, &v.operands[..ar(v.word)]);
                self.shadow_descriptor(base, 0x8c + 8 * slot, &v.attribute, i);
                for a in 1..19usize {
                    if v.attribute[a][0] == 0 {
                        continue;
                    }
                    self.shadow_ring(
                        base,
                        a as u32,
                        i,
                        v.attribute[a][0],
                        &v.attribute[a][1..1 + ar(v.attribute[a][0])],
                    );
                }
            }
            self.shadow_descriptor(base, ODSY_HW_CURRENT_ATTRIBUTES, &current, 3);
            for i in 0..19usize {
                if current[i][0] == 0 {
                    continue;
                }
                self.shadow_ring(base, i as u32, 3, current[i][0], &current[i][1..1 + ar(current[i][0])]);
            }
        }
    }

    /// The kernel driver's context switch: a single 8-byte store to
    /// ODSY_CTX_SWITCH_PORT, not a command-FIFO word. The new context id in bits
    /// 31:16 is not modelled: the board learns the context from the shadow address
    /// pair, which names where the outgoing state is saved (Board::ctx_save()),
    /// and the kernel restores the rest. A store naming context 0x7f is the
    /// kernel driver's shadow snapshot and saves the same way. Any other shape at
    /// this port has no established meaning and is reported unimplemented instead
    /// of guessed.
    pub fn ctx_switch(&mut self, value: u64) {
        if ((value >> ODSY_CTX_SWITCH_OPCODE_SHIFT) & 0xff) != ODSY_CTX_SWITCH_OPCODE as u64
            || (value & ODSY_CTX_SWITCH_VALID) == 0
        {
            self.unimplemented(ODSY_CTX_SWITCH_PORT, 8, true);
            return;
        }
        if self.ctx_shadow_addr_upper == 0 && self.ctx_shadow_addr_lower == 0 {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-odyssey: context switch ({:#x}) with no shadow address programmed; no completion written",
                value
            );
            return;
        }
        self.query_cancel();
        let shadow_base = (((self.ctx_shadow_addr_upper & 0xffff) as u64) << 32) | self.ctx_shadow_addr_lower as u64;
        // A plain guest-physical address: interrupt packets go elsewhere.
        self.ctx_save(shadow_base);
        self.shadow_put(shadow_base, ODSY_CTX_SWITCH_DONE_OFFSET as u32, 1);
    }
}
