
use crate::devlog::LogModule;
use super::board::{Board, OpenVertex};
use super::consts::{cmd_arity, ctx, gl_cmd, hw, tex, ODSY_F32_ONE};
use super::pixel_dma::write_guest;
use super::registers::BACK_MATERIAL_DEFAULTS;
// Anonymous enum: current-value query kinds.
pub(super) const ODSY_QUERY_INDEX: u32 = 0;

pub(super) const ODSY_QUERY_COLOUR: u32 = 1;

pub(super) const ODSY_QUERY_NORMAL: u32 = 2;

pub(super) const ODSY_QUERY_TEXCOORD: u32 = 3;

pub(super) const ODSY_QUERY_EDGE: u32 = 4;

impl Board {
    pub fn back_material_reset(&mut self) {
        self.geom.back_material.copy_from_slice(&BACK_MATERIAL_DEFAULTS);
        self.geom.back_shininess_sqrt = 0;
    }

    pub fn shadow_put(&mut self, base: u64, off: u32, word: u32) {
        let at = base.wrapping_add(off as u64);
        if !write_guest(&self.guest, at, &word.to_be_bytes()) && !self.diag.ctx_shadow_fail_reported {
            self.diag.ctx_shadow_fail_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: context shadow store to {:#x} failed", at);
        }
    }

    /// One selected ring entry: attribute a's word and its raw operands.
    pub fn shadow_ring(&mut self, base: u64, a: u32, slot: u32, word: u32, ops: &[u32]) {
        self.shadow_put(base, hw::RING_WORD + hw::RING_WORD_STRIDE * a + 4 * slot, word);
        for (i, op) in ops.iter().enumerate() {
            self.shadow_put(
                base,
                hw::RING_OPERANDS + hw::RING_OPERAND_STRIDE * a + 16 * slot + 4 * i as u32,
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

        for (a, rec) in record.iter().enumerate().skip(1) {
            if rec[0] == 0 {
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
            gl_cmd::MATERIAL_AMBIENT,
            gl_cmd::MATERIAL_DIFFUSE,
            gl_cmd::MATERIAL_SPECULAR,
            gl_cmd::MATERIAL_EMISSION,
        ];
        const BACK_MATERIAL: [u32; 4] = [gl_cmd::BACK_AMBIENT, gl_cmd::BACK_DIFFUSE, gl_cmd::BACK_SPECULAR, gl_cmd::BACK_EMISSION];

        *record = [[0u32; 5]; 19];
        record[hw::ATTR_INDEX] = self.query.current[ODSY_QUERY_INDEX as usize];
        record[hw::ATTR_COLOUR] = self.query.current[ODSY_QUERY_COLOUR as usize];
        record[hw::ATTR_NORMAL] = self.query.current[ODSY_QUERY_NORMAL as usize];
        record[hw::ATTR_TEXTURE] = self.query.current[ODSY_QUERY_TEXCOORD as usize];
        record[hw::ATTR_EDGE] = self.query.current[ODSY_QUERY_EDGE as usize];
        for i in 0..4usize {
            record[hw::ATTR_MATERIAL + i][0] = MATERIAL[i];
            record[hw::ATTR_MATERIAL + i][1..5].copy_from_slice(&self.geom.material[4 * i..4 * i + 4]);
            record[hw::ATTR_BACK_MATERIAL + i][0] = BACK_MATERIAL[i];
            record[hw::ATTR_BACK_MATERIAL + i][1..5].copy_from_slice(&self.geom.back_material[4 * i..4 * i + 4]);
        }
        record[hw::ATTR_SHININESS][0] = gl_cmd::SHININESS_SQRT;
        record[hw::ATTR_SHININESS][1] = self.geom.shininess_sqrt;
        record[hw::ATTR_BACK_SHININESS][0] = gl_cmd::BACK_SHININESS_SQRT;
        record[hw::ATTR_BACK_SHININESS][1] = self.geom.back_shininess_sqrt;
    }

    pub fn open_record_vertex(&mut self, word: u32) {
        let mut v = OpenVertex { word, slot: self.prim.open_r, ..Default::default() };
        let mut n = self.prim.open_held as usize;
        let arity = (cmd_arity(word) as usize).min(4);

        v.operands[..arity].copy_from_slice(&self.cmd.operand[..arity]);
        self.open_current_records(&mut v.attribute);
        if self.stream.primitive == 2 || self.stream.primitive == 6 {
            self.prim.open_r = if self.prim.open_r == 3 { 1 } else { self.prim.open_r + 1 };
        } else {
            self.prim.open_r = (self.prim.open_r + 1) & 3;
        }
        self.prim.open_c = (self.prim.open_c + 1).min(7);
        self.prim.open_m3 = (self.prim.open_m3 + 1) % 3;
        self.prim.open_m4 = (self.prim.open_m4 + 1) & 3;

        match self.stream.primitive {
            0 => {
                // Points retain no vertices.
                n = 0;
            }
            1 => {
                // Lines retain one of each pair.
                if n != 0 {
                    n = 0;
                } else {
                    self.prim.open_vertex[0] = v;
                    n = 1;
                }
            }
            2 | 6 => {
                // Anchor and latest vertex.
                self.prim.open_vertex[if n != 0 { 1 } else { 0 }] = v;
                n = (n + 1).min(2);
            }
            3 => {
                // Line strip.
                self.prim.open_vertex[0] = v;
                n = 1;
            }
            4 | 7 => {
                // Triangles and quads.
                if n == (if self.stream.primitive == 4 { 2 } else { 3 }) {
                    n = 0;
                } else {
                    self.prim.open_vertex[n] = v;
                    n += 1;
                }
            }
            5 => {
                // Triangle strip.
                if n == 2 {
                    self.prim.open_vertex[0] = self.prim.open_vertex[1];
                    self.prim.open_vertex[1] = v;
                } else {
                    self.prim.open_vertex[n] = v;
                    n += 1;
                }
            }
            8 => {
                // Quad strip.
                if n == 3 {
                    self.prim.open_vertex[0] = self.prim.open_vertex[2];
                    self.prim.open_vertex[1] = v;
                    n = 2;
                } else {
                    self.prim.open_vertex[n] = v;
                    n += 1;
                }
            }
            _ => {}
        }
        self.prim.open_held = n as u32;
    }

    pub fn restart_finish(&mut self) {
        let n = self.prim.open_held;
        let mut count;

        if !self.prim.restart_armed || self.prim.restart_absorb != 0 {
            return;
        }
        if self.prim.restart_m3 >= 3 || n != self.prim.restart_total {
            self.cmd_unimplemented(0x741);
            self.prim.restart_armed = false;
            return;
        }
        self.prim.open_c = if self.prim.restart_c == 7 { 7 } else { (self.prim.restart_c + n).min(7) };
        self.prim.open_m3 = (self.prim.restart_m3 + n) % 3;
        self.prim.open_m4 = (self.prim.restart_m4 + n) & 3;
        count = self.prim.open_c;
        if count == 7 {
            while (count % 3 != self.prim.open_m3 || count % 4 != self.prim.open_m4) && count < 19 {
                count += 1;
            }
        }
        self.prim.in_prim = self.gl_ctx.resume_primitive(self.prim.restart_type, Some(&self.prim.restart_vertex[..]), n, count);
        self.prim.restart_armed = false;
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
            gl_cmd::MATERIAL_AMBIENT,
            gl_cmd::MATERIAL_DIFFUSE,
            gl_cmd::MATERIAL_SPECULAR,
            gl_cmd::MATERIAL_EMISSION,
        ];
        const BACK_MATERIAL: [u32; 4] = [gl_cmd::BACK_AMBIENT, gl_cmd::BACK_DIFFUSE, gl_cmd::BACK_SPECULAR, gl_cmd::BACK_EMISSION];
        let index = self.query.current[ODSY_QUERY_INDEX as usize];
        let colour = self.query.current[ODSY_QUERY_COLOUR as usize];
        let normal = self.query.current[ODSY_QUERY_NORMAL as usize];
        let edge = self.query.current[ODSY_QUERY_EDGE as usize][0];
        let open = self.prim.begin_open && !self.prim.integer_prim && self.stream.primitive <= 8;
        let ar = |w: u32| (cmd_arity(w) as usize).min(4);

        if self.prim.begin_open && !open && !self.diag.ctx_open_prim_logged {
            self.diag.ctx_open_prim_logged = true;
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-odyssey: context switch inside primitive {:#x}; its buffered vertices are not saved",
                self.stream.primitive
            );
        }
        self.shadow_put(base, hw::PIPE_STATE, hw::PIPE_IDLE);
        for i in 0..hw::DESCRIPTOR_WORDS {
            self.shadow_put(base, hw::CURRENT_ATTRIBUTES + 4 * i, 0);
        }
        self.shadow_put(base, hw::INDEX_WORD, index[0]);
        for i in 0..cmd_arity(index[0]) {
            self.shadow_put(base, hw::INDEX_OPERANDS + 4 * i, index[1 + (i as usize).min(3)]);
        }
        self.shadow_put(base, hw::COLOUR_WORD, colour[0]);
        for i in 0..cmd_arity(colour[0]) {
            self.shadow_put(base, hw::COLOUR_OPERANDS + 4 * i, colour[1 + (i as usize).min(3)]);
        }
        self.shadow_ring(base, hw::ATTR_INDEX as u32, 0, index[0], &index[1..1 + ar(index[0])]);
        self.shadow_ring(base, hw::ATTR_COLOUR as u32, 0, colour[0], &colour[1..1 + ar(colour[0])]);
        self.shadow_ring(base, hw::ATTR_NORMAL as u32, 0, normal[0], &normal[1..1 + ar(normal[0])]);
        self.shadow_ring(base, hw::ATTR_EDGE as u32, 0, edge, &[]);
        if self.tex.coord_q != ODSY_F32_ONE {
            let coordinate = [self.tex.coord[0], self.tex.coord[1], self.tex.coord_r, self.tex.coord_q];

            self.shadow_ring(base, hw::ATTR_TEXTURE as u32, 0, gl_cmd::TEXCOORD_4F, &coordinate);
        } else {
            let three = (self.tex.mode & tex::MODE_3D) != 0 || self.tex.coord_r != 0;
            let three_ops = [self.tex.coord[0], self.tex.coord[1], self.tex.coord_r];
            let two_ops = self.tex.coord;

            self.shadow_ring(
                base,
                hw::ATTR_TEXTURE as u32,
                0,
                if three { gl_cmd::TEXCOORD_3F } else { gl_cmd::TEXCOORD_2F },
                if three { &three_ops[..] } else { &two_ops[..] },
            );
        }
        for i in 0..4usize {
            let m: [u32; 4] = self.geom.material[i * 4..i * 4 + 4].try_into().unwrap();

            self.shadow_ring(base, (hw::ATTR_MATERIAL + i) as u32, 0, MATERIAL[i], &m);
        }
        let sh = [self.geom.shininess_sqrt];
        self.shadow_ring(base, hw::ATTR_SHININESS as u32, 0, gl_cmd::SHININESS_SQRT, &sh);
        for i in 0..4usize {
            let m: [u32; 4] = self.geom.back_material[i * 4..i * 4 + 4].try_into().unwrap();

            self.shadow_ring(base, (hw::ATTR_BACK_MATERIAL + i) as u32, 0, BACK_MATERIAL[i], &m);
        }
        let bsh = [self.geom.back_shininess_sqrt];
        self.shadow_ring(base, hw::ATTR_BACK_SHININESS as u32, 0, gl_cmd::BACK_SHININESS_SQRT, &bsh);
        if open {
            let mut current = [[0u32; 5]; 19];
            let mut state = self.stream.primitive
                | (self.prim.open_held << 4)
                | (self.prim.open_r << 6)
                | (self.prim.open_m3 << 12)
                | (self.prim.open_m4 << 14)
                | (self.prim.open_c << 21);

            // After c >= 2 vertices the next triangle is number c - 2.
            if self.stream.primitive == gl_cmd::TRIANGLE_STRIP && self.prim.open_c >= 2 && (self.prim.open_m4 & 1) != 0 {
                state |= hw::PIPE_STRIP_ODD;
            }

            self.open_current_records(&mut current);
            self.shadow_put(base, hw::PIPE_STATE, state);
            for i in 0..self.prim.open_held.min(3) {
                let v = self.prim.open_vertex[i as usize];
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
            self.shadow_descriptor(base, hw::CURRENT_ATTRIBUTES, &current, 3);
            for i in 0..19usize {
                if current[i][0] == 0 {
                    continue;
                }
                self.shadow_ring(base, i as u32, 3, current[i][0], &current[i][1..1 + ar(current[i][0])]);
            }
        }
    }

    /// The kernel driver's context switch: a single 8-byte store to
    /// ctx::SWITCH_PORT, not a command-FIFO word. The new context id in bits
    /// 31:16 is not modelled: the board learns the context from the shadow address
    /// pair, which names where the outgoing state is saved (Board::ctx_save()),
    /// and the kernel restores the rest. A store naming context 0x7f is the
    /// kernel driver's shadow snapshot and saves the same way. Any other shape at
    /// this port has no established meaning and is reported unimplemented instead
    /// of guessed.
    pub fn ctx_switch(&mut self, value: u64) {
        if ((value >> ctx::SWITCH_OPCODE_SHIFT) & 0xff) != ctx::SWITCH_OPCODE as u64
            || (value & ctx::SWITCH_VALID) == 0
        {
            self.unimplemented(ctx::SWITCH_PORT, 8, true);
            return;
        }
        // The done word says the board has finished the work before the
        // switch, the raster unit's included.
        self.raster_wait();
        if self.widget.ctx_shadow_addr_upper == 0 && self.widget.ctx_shadow_addr_lower == 0 {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-odyssey: context switch ({:#x}) with no shadow address programmed; no completion written",
                value
            );
            return;
        }
        self.query_cancel();
        let shadow_base = (((self.widget.ctx_shadow_addr_upper & 0xffff) as u64) << 32) | self.widget.ctx_shadow_addr_lower as u64;
        // A plain guest-physical address: interrupt packets go elsewhere.
        self.ctx_save(shadow_base);
        self.shadow_put(shadow_base, ctx::SWITCH_DONE_OFFSET as u32, 1);
    }
}
