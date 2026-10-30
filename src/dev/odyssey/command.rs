//! The command-FIFO interpreter: word classification, operand gathering and
//! the one big dispatch of a completed command.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.

use super::*;

impl Board {
    pub fn cmd_dispatch(&mut self, tag: u32) {
        let mut after_tail = false;

        // Only the kernel command right after its window-origin write can be its tail.
        if self.cmd_from_kernel && tag != ODSY_REG_WINDOW_ORIGIN && tag != ODSY_REG_WINDOW_CLIP {
            self.win_origin_pending = false;
        }
        // And only the one right after the tail's 0x204b is its 0x840.
        if self.cmd_from_kernel {
            after_tail = self.win_after_tail;
            self.win_after_tail = false;
        }
        if is_program_word(tag) {
            self.gl_program_word(tag);
            return;
        }
        if (ODSY_REG_TEX_DETAIL_FUNCTION..ODSY_REG_TEX_DETAIL_FUNCTION + ODSY_TEX_DETAIL_FUNCTION_BLOCKS).contains(&tag)
        {
            let block = tag - ODSY_REG_TEX_DETAIL_FUNCTION;

            if self.cmd_operand_count != ODSY_TEX_DETAIL_FUNCTION_BLOCK_WORDS {
                self.tex_detail_function_valid &= !(1u32 << block);
                self.cmd_unimplemented(tag);
                return;
            }
            let w = ODSY_TEX_DETAIL_FUNCTION_BLOCK_WORDS as usize;
            let at = block as usize * w;
            self.tex_detail_function[at..at + w].copy_from_slice(&self.cmd_operand[..w]);
            self.tex_detail_function_valid |= 1u32 << block;
            return;
        }
        if (ODSY_REG_TEX_TABLE3..ODSY_REG_TEX_TABLE3 + ODSY_TEX_TABLE3_BLOCKS).contains(&tag) {
            self.cmd_unimplemented(tag);
            return;
        }
        if (ODSY_REG_TEX_COLOUR_TABLE0..ODSY_REG_TEX_COLOUR_TABLE0 + ODSY_TEX_COLOUR_TABLE_BLOCKS).contains(&tag)
            || (ODSY_REG_TEX_COLOUR_TABLE1..ODSY_REG_TEX_COLOUR_TABLE1 + ODSY_TEX_COLOUR_TABLE_BLOCKS).contains(&tag)
        {
            let plane = (tag >= ODSY_REG_TEX_COLOUR_TABLE1) as usize;
            let first_tag = if plane != 0 { ODSY_REG_TEX_COLOUR_TABLE1 } else { ODSY_REG_TEX_COLOUR_TABLE0 };
            let block = tag - first_tag;

            if self.cmd_operand_count != ODSY_TEX_COLOUR_TABLE_BLOCK_WORDS {
                self.cmd_unimplemented(tag);
                return;
            }
            let w = ODSY_TEX_COLOUR_TABLE_BLOCK_WORDS as usize;
            let at = block as usize * w;
            self.tex_colour_table[plane][at..at + w].copy_from_slice(&self.cmd_operand[..w]);
            self.tex_colour_table_valid[plane] |= 1u32 << block;
            return;
        }
        if (ODSY_REG_TEX_LEVEL..ODSY_REG_TEX_LEVEL + SGI_ODYSSEY_TEX_LEVELS).contains(&tag) {
            let at = (tag - ODSY_REG_TEX_LEVEL) as usize * 3;
            self.tex_levels[at..at + 3].copy_from_slice(&self.cmd_operand[..3]);
            return;
        }
        if tag == ODSY_GL_TEXCOORD_3F || tag == ODSY_GL_TEXCOORD_4F {
            self.tex_coord.copy_from_slice(&self.cmd_operand[..2]);
            self.tex_coord_r = self.cmd_operand[2];
            self.tex_coord_q = if tag == ODSY_GL_TEXCOORD_4F { self.cmd_operand[3] } else { ODSY_F32_ONE };
            return;
        }
        if tag == ODSY_GL_TEXCOORD_1I
            || tag == ODSY_GL_TEXCOORD_2I
            || tag == ODSY_GL_TEXCOORD_3I
            || tag == ODSY_GL_TEXCOORD_4I
        {
            let n = cmd_arity(tag);
            let mut c: [u32; 4] = [0, 0, 0, ODSY_F32_ONE];

            for i in 0..n as usize {
                c[i] = f32_bits((self.cmd_operand[i] as i32) as f32);
            }
            self.tex_coord[0] = c[0];
            self.tex_coord[1] = c[1];
            self.tex_coord_r = c[2];
            self.tex_coord_q = c[3];
            return;
        }
        if (tag & ODSY_GL_TEXCOORD_SHAPE) == ODSY_GL_TEXCOORD_TAG && tag != ODSY_GL_TEXCOORD_2F {
            // Other two-coordinate forms remain observable and change nothing.
            self.cmd_unimplemented(tag);
            return;
        }
        if (tag & ODSY_GL_MATRIX_SELECT_SHAPE) == ODSY_GL_MATRIX_SELECT_TAG {
            self.gl_seq[((ODSY_GL_MATRIX_SELECT_TAG >> ODSY_PROG_CLASS_SHIFT) - ODSY_PROG_SEQ_FIRST) as usize] =
                prog_value(tag);
            // Pick the shared matrix slot the following load targets.
            self.gl_matrix_slot = match tag & 0xff {
                ODSY_GL_MATRIX_SELECT_COMBINED => ODSY_GL_SLOT_COMBINED,
                ODSY_GL_MATRIX_SELECT_TEXTURE => ODSY_GL_SLOT_TEXTURE,
                ODSY_GL_MATRIX_SELECT_MODELVIEW => SGI_GL_MODELVIEW,
                ODSY_GL_MATRIX_SELECT_BLOCK16 => ODSY_GL_SLOT_BLOCK16,
                ODSY_GL_MATRIX_SELECT_NORMAL => ODSY_GL_SLOT_NORMAL_MATRIX,
                // Any other block: no home.
                _ => ODSY_GL_SLOT_NORMAL,
            };
            return;
        }
        match tag {
            ODSY_GL_COLOR_3UB => {
                self.gl_color = pack_pixel(self.cmd_operand[0], self.cmd_operand[1], self.cmd_operand[2]);
                self.gl_color_is_index = false;
                if !self.gl_index_mode() {
                    let mut colour = [0u32; SGI_GL_GET_WORDS];
                    let mut n = 0u32;
                    let opd = self.cmd_operand;

                    self.gl_feed(SGI_GL_OP_COLOR3UB as u16, &opd[..3]);
                    let queried = self.gl().get(SGI_GL_CURRENT_COLOR, &mut colour, &mut n);
                    assert!(queried && n as usize == self.gl_colour.len());
                    let l = self.gl_colour.len();
                    self.gl_colour.copy_from_slice(&colour[..l]);
                    self.gl_colour_overridden = false;
                }
            }
            ODSY_GL_COLOR_INDEX | ODSY_GL_REPLAY_INDEX => {
                // The user's index word, or the kernel's replay of a saved one.
                self.gl_color_index = self.cmd_operand[0];
                self.gl_color_is_index = true;
            }
            ODSY_GL_COLOR_INDEX_F | ODSY_GL_REPLAY_INDEX_F => {
                self.gl_color_index = float_index(self.cmd_operand[0]);
                self.gl_color_is_index = true;
            }
            ODSY_REG_DEST_SELECT => {
                // Destination-buffer select: its third operand marks an id-plane
                // write when it is ODSY_DEST_ID_PLANES. A colour select's non-zero
                // first operand names the colour buffer, the base a DMA header's
                // destination word is matched against (the all-zero blocks are GL
                // context setup, not a buffer).
                self.dest_buffer_word = self.cmd_operand[ODSY_DEST_SELECT_VALUE_INDEX];
                self.dest_base = self.cmd_operand[ODSY_DEST_SELECT_BASE_INDEX];
                if self.dest_base == 0 || self.dest_is_id_planes() {
                    return;
                }
                let base = self.dest_base;
                if !self.board_address_valid(base) || (self.dest_buffer_word & !ODSY_DEST_FLAGS) == 0 {
                    return;
                }
                if (self.dest_buffer_word >> ODSY_DEST_FORMAT_SHIFT) == ODSY_DEST_FORMAT_SCREEN
                    || self.dest_base == self.dma_colour_base
                {
                    self.dma_colour_base = self.dest_base;
                } else {
                    // An off-screen object: create it, or update its row length.
                    let width = self.cmd_operand[ODSY_DEST_WIDTH_INDEX].wrapping_mul(ODSY_DEST_WIDTH_UNIT);
                    self.object_select(base, width);
                }
            }
            ODSY_REG_PLANE_MASK => {
                // Plane write mask (mask << 12), consumed by an id-plane write.
                self.plane_write_mask = self.cmd_operand[0];
            }
            ODSY_MODE_INDEX_ZEROS
            | ODSY_GL_INDEX_MODE_COLOR_3I
            | ODSY_GL_INDEX_MODE_COLOR_3F
            | ODSY_GL_INDEX_MODE_COLOR_4F => {
                // An RGBA colour sent in index mode: GL does not draw with it.
            }
            ODSY_GL_VERTEX_2I => {
                if self.gl_begin_open {
                    self.gl_integer_prim = true;
                }
                // Bit 13 distinguishes the observed GL and X11 integer streams.
                // Its hardware meaning is unknown. This is a model routing policy,
                // not a claim that the bit enables hardware vertex transforms.
                if self.gl_begin_open
                    && !self.image_active
                    && ((self.pattern_mode & (1u32 << 13)) != 0
                        || (self.tex_mode & (ODSY_TEX_MODE_1D | ODSY_TEX_MODE_2D | ODSY_TEX_MODE_3D)) != 0
                        || self.gl_primitive == ODSY_GL_TRIANGLES
                        || self.gl_primitive == ODSY_GL_TRIANGLE_STRIP
                        || self.gl_primitive == ODSY_GL_TRIANGLE_FAN
                        || self.gl_primitive == ODSY_GL_QUAD_STRIP)
                {
                    // Lighting takes the signed integers as object coordinates.
                    let xyz: [f64; 3] = [
                        (self.cmd_operand[0] as i32) as f64,
                        (self.cmd_operand[1] as i32) as f64,
                        0.0,
                    ];
                    let opd = self.cmd_operand;

                    if !self.gl_in_prim {
                        let mode = self.gl_primitive;

                        self.gl_feed(SGI_GL_OP_BEGIN as u16, &[mode]);
                        self.gl_in_prim = true;
                    }
                    self.gl_vertex_colour(&xyz);
                    self.texture_vertex();
                    self.gl_feed(SGI_GL_OP_VERTEX2I as u16, &opd[..2]);
                } else {
                    // The X path uses window-relative integer coordinates.
                    let x = self.cmd_operand[0].wrapping_add(self.win_origin[0] as u32);
                    let y = self.cmd_operand[1].wrapping_add(self.win_origin[1] as u32);
                    self.gl_vertex(x, y);
                }
            }
            ODSY_GL_COLOR_3F
            | ODSY_GL_COLOR_3I
            | ODSY_GL_COLOR_4F
            | ODSY_GL_COLOR_3US
            | ODSY_GL_COLOR_3UI
            | ODSY_GL_COLOR_3B
            | ODSY_GL_COLOR_3S
            | ODSY_GL_COLOR_4UB
            | ODSY_GL_COLOR_4US
            | ODSY_GL_COLOR_4UI
            | ODSY_GL_COLOR_4B
            | ODSY_GL_COLOR_4S
            | ODSY_GL_COLOR_4I => {
                // The current colour of the true-colour GL stream (glColor3f,
                // glColor4f and the integer forms), or the kernel's replay of one
                // Board::ctx_save() saved. Kept so an unlit vertex after lit ones
                // takes it back, and so colour material lights with it.
                let n = cmd_arity(tag);
                let mut rgba = [0u32; 4];

                rgba.copy_from_slice(&self.cmd_operand[..4]);
                if tag != ODSY_GL_COLOR_3F && tag != ODSY_GL_COLOR_4F {
                    for i in 0..n as usize {
                        rgba[i] = integer_colour(tag, self.cmd_operand[i]);
                    }
                }
                if n == 3 {
                    rgba[3] = ODSY_F32_ONE;
                }
                self.gl_colour.copy_from_slice(&rgba);
                self.gl_colour_overridden = false;
                if n == 4 {
                    self.gl_feed(SGI_GL_OP_COLOR4F as u16, &rgba[..4]);
                } else {
                    self.gl_feed(SGI_GL_OP_COLOR3F as u16, &rgba[..3]);
                }
                if !self.gl_index_mode() {
                    self.gl_color = pack_pixel(
                        float_channel(rgba[0]),
                        float_channel(rgba[1]),
                        float_channel(rgba[2]),
                    );
                    self.gl_color_is_index = false;
                }
            }
            ODSY_GL_NORMAL_3B | ODSY_GL_NORMAL_3S | ODSY_GL_NORMAL_3I => {
                for i in 0..3 {
                    self.gl_normal[i] = signed_to_float(self.cmd_operand[i], tag & 0x3f);
                }
            }
            ODSY_GL_NORMAL_3F => {
                self.gl_normal.copy_from_slice(&self.cmd_operand[..3]);
            }
            ODSY_GL_MATERIAL_AMBIENT
            | ODSY_GL_MATERIAL_DIFFUSE
            | ODSY_GL_MATERIAL_SPECULAR
            | ODSY_GL_MATERIAL_EMISSION => {
                // The low byte counts 0x2f, 0x30, 0x31, 0x32 through the four.
                let at = (((tag & 0xff) - (ODSY_GL_MATERIAL_AMBIENT & 0xff)) * 4) as usize;
                self.gl_material[at..at + 4].copy_from_slice(&self.cmd_operand[..4]);
            }
            ODSY_GL_SHININESS_SQRT => {
                self.gl_shininess_sqrt = self.cmd_operand[0];
            }
            ODSY_GL_BACK_AMBIENT | ODSY_GL_BACK_DIFFUSE | ODSY_GL_BACK_SPECULAR | ODSY_GL_BACK_EMISSION => {
                let at = (((tag & 0xff) - (ODSY_GL_BACK_AMBIENT & 0xff)) * 4) as usize;
                self.gl_back_material[at..at + 4].copy_from_slice(&self.cmd_operand[..4]);
            }
            ODSY_GL_BACK_SHININESS_SQRT => {
                self.gl_back_shininess_sqrt = self.cmd_operand[0];
            }
            ODSY_GL_SHININESS_DERIVED | ODSY_PROG_RUN_ONCE => {
                // Consumed, not kept: the derived shininess floats and the one-shot
                // run have no modelled effect. Block 0x1ea (ODSY_GL_SHININESS_DERIVED)
                // is also the second read path of attribute slots 448 and 544: every
                // full context replay sends it with the selected slot-448 entry's
                // first operand, that with bit 31 flipped, and the same for slot 544
                // (ESTABLISHED as the kernel's replay). What the block does on the
                // board is UNKNOWN; the model gives neither use an effect.
            }
            ODSY_GL_CLIP_PLANE_LOAD => {
                // Public controls identify the effective S, T, R and Q planes.
                if self.gl_light_vec_sel == 0x10c {
                    self.tex_linear_plane_q.copy_from_slice(&self.cmd_operand[..4]);
                    self.tex_linear_valid |= 8;
                    return;
                }
                if self.gl_light_vec_sel == 0x108 {
                    self.tex_linear_plane_r.copy_from_slice(&self.cmd_operand[..4]);
                    self.tex_linear_valid |= 4;
                    return;
                }
                if self.gl_light_vec_sel == 0x100 || self.gl_light_vec_sel == 0x104 {
                    let axis = ((self.gl_light_vec_sel - 0x100) / 4) as usize;

                    self.tex_linear_plane[axis].copy_from_slice(&self.cmd_operand[..4]);
                    self.tex_linear_valid |= 1u32 << axis;
                    return;
                }
                // See ODSY_GL_CLIP_PLANE_BASE's comment: accept and pass through
                // loads into the shared layer's six slots, silently drop loads past
                // that budget (expected library traffic, not an error), and report
                // only a selector this model cannot place at all.
                if self.gl_light_vec_sel >= ODSY_GL_CLIP_PLANE_BASE
                    && (self.gl_light_vec_sel - ODSY_GL_CLIP_PLANE_BASE).is_multiple_of(ODSY_GL_CLIP_PLANE_STRIDE)
                {
                    let plane = (self.gl_light_vec_sel - ODSY_GL_CLIP_PLANE_BASE) / ODSY_GL_CLIP_PLANE_STRIDE;

                    if plane < SGI_GL_CLIP_PLANES {
                        let at = plane as usize * 4;
                        self.gl_clip_eye[at..at + 4].copy_from_slice(&self.cmd_operand[..4]);
                        self.gl_apply_clip_plane(plane);
                    }
                } else if !self.gl_block_logged {
                    self.gl_block_logged = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: class 0x31 clip-plane load at index {:#x} is not modeled",
                        self.gl_light_vec_sel
                    );
                }
            }
            ODSY_GL_LIGHT_COLOUR_LOAD => {
                if self.gl_light_colour_sel < SGI_ODYSSEY_GL_LIGHT_COLOURS {
                    let at = self.gl_light_colour_sel as usize * 4;
                    self.gl_light_colour[at..at + 4].copy_from_slice(&self.cmd_operand[..4]);
                } else if !self.gl_block_logged {
                    self.gl_block_logged = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: class 0x32 load at index {:#x} is not modeled",
                        self.gl_light_colour_sel
                    );
                }
            }
            ODSY_GL_LIGHT_VEC_LOAD => {
                if self.gl_light_vec_sel == 0x100 {
                    self.tex_linear_plane[0].copy_from_slice(&self.cmd_operand[0..4]);
                    self.tex_linear_plane[1].copy_from_slice(&self.cmd_operand[4..8]);
                    self.tex_linear_plane_r.copy_from_slice(&self.cmd_operand[8..12]);
                    self.tex_linear_plane_q.copy_from_slice(&self.cmd_operand[12..16]);
                    self.tex_linear_valid |= 0xf;
                    return;
                }
                // Index 16 i is light i. Other context setup blocks at 0x80
                // to 0x1e0 remain unknown and are reported once, not per index.
                if self.gl_light_vec_sel.is_multiple_of(ODSY_GL_LIGHT_VEC_FLOATS)
                    && self.gl_light_vec_sel / ODSY_GL_LIGHT_VEC_FLOATS < SGI_ODYSSEY_GL_LIGHTS
                {
                    let at = self.gl_light_vec_sel as usize;
                    let n = ODSY_GL_LIGHT_VEC_FLOATS as usize;
                    self.gl_light_vec[at..at + n].copy_from_slice(&self.cmd_operand[..n]);
                } else if !self.gl_block_logged {
                    self.gl_block_logged = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: class 0x31 load at index {:#x} is not modeled",
                        self.gl_light_vec_sel
                    );
                }
            }
            ODSY_GL_VERTEX_2F | ODSY_GL_VERTEX_3F | ODSY_GL_VERTEX_3I | ODSY_GL_VERTEX_4F => {
                // A float vertex uses the shared GL path, and so does an integer 3D
                // vertex once converted to float (ODSY_GL_VERTEX_3I). Open the
                // primitive lazily on its first vertex, as for shared integer
                // geometry above. The open-primitive record keeps the word and its
                // operands as sent. glVertex4f (ODSY_GL_VERTEX_4F) passes its W on.
                let components: u32 = if tag == ODSY_GL_VERTEX_4F {
                    4
                } else if tag == ODSY_GL_VERTEX_2F {
                    2
                } else {
                    3
                };
                let mut xyz = [0u32; 4];
                xyz.copy_from_slice(&self.cmd_operand[..4]);
                let mut obj = [0f64; 3];

                if tag == ODSY_GL_VERTEX_3I {
                    for k in 0..3 {
                        xyz[k] = f32_bits((self.cmd_operand[k] as i32) as f32);
                    }
                }
                if !self.gl_in_prim && self.gl_restart_absorb == 0 {
                    let mode = self.gl_primitive;

                    self.gl_feed(SGI_GL_OP_BEGIN as u16, &[mode]);
                    self.gl_in_prim = true;
                }
                if self.gl_begin_open && !self.gl_integer_prim && self.gl_primitive <= 8 {
                    self.open_record_vertex(tag);
                }
                obj[0] = odyssey_f32(xyz[0]);
                obj[1] = odyssey_f32(xyz[1]);
                obj[2] = if components > 2 { odyssey_f32(xyz[2]) } else { 0.0 };
                if components == 4 && odyssey_f32(xyz[3]) != 0.0 {
                    // (x, y, z, w) and (x/w, y/w, z/w, 1) are the same point.
                    for c in &mut obj[..3] {
                        *c /= odyssey_f32(xyz[3]);
                    }
                }
                self.gl_vertex_colour(&obj);
                self.texture_vertex();
                if self.gl_restart_absorb != 0 {
                    let slot = (self.gl_restart_total - self.gl_restart_absorb) as usize;

                    if slot < self.gl_restart_vertex.len() {
                        self.gl_ctx.as_mut().expect("the GL context").capture_vertex(
                            self.gl_restart_type,
                            &xyz,
                            components,
                            &mut self.gl_restart_vertex[slot],
                        );
                    }
                    self.gl_restart_absorb -= 1;
                    self.restart_finish();
                    return;
                }
                let op = if components == 4 {
                    SGI_GL_OP_VERTEX4F
                } else if components == 3 {
                    SGI_GL_OP_VERTEX3F
                } else {
                    SGI_GL_OP_VERTEX2F
                };
                self.gl_feed(op as u16, &xyz[..components as usize]);
            }
            ODSY_GL_LOAD_4F => {
                self.gl_load_block16();
            }
            ODSY_REG_DEPTH_FUNC => {
                if (self.cmd_operand[0] & !ODSY_DEPTH_FUNC_MASK) != 0 {
                    // Only bits 2:0 are known; every capture leaves the rest 0.
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: depth function word {:#x} has bits beyond 2:0, not modeled",
                        self.cmd_operand[0]
                    );
                }
                self.gl_depth_func = self.cmd_operand[0];
                self.gl_apply_depth();
            }
            ODSY_REG_WRITE_MASK => {
                let stencil_mask = (self.cmd_operand[0] >> ODSY_WRITE_STENCIL_SHIFT) & ODSY_WRITE_STENCIL_MASK;

                self.gl_write_mask = self.cmd_operand[0];
                self.gl_apply_depth();
                // The library forces it to 0 for a drawable with no stencil.
                self.gl_feed(SGI_GL_OP_STENCIL_MASK as u16, &[stencil_mask]);
            }
            ODSY_REG_ALPHA_FUNC => {
                let word = self.cmd_operand[0];
                let mut args = [0u32; 2];

                args[0] = SGI_GL_NEVER_ENUM + ((word >> ODSY_ALPHA_FUNC_SHIFT) & ODSY_ALPHA_FUNC_MASK);
                args[1] = f32_bits((word & ODSY_ALPHA_REF_MASK) as f32 / ODSY_ALPHA_REF_SCALE);
                self.gl_feed(SGI_GL_OP_ALPHA_FUNC as u16, &args);
            }
            ODSY_REG_STENCIL_FUNC => {
                let word = self.cmd_operand[0];
                let mut args = [0u32; 3];

                args[0] = SGI_GL_NEVER_ENUM + ((word >> ODSY_STENCIL_FUNC_SHIFT) & ODSY_STENCIL_FUNC_MASK);
                args[1] = word & ODSY_STENCIL_REF_MASK;
                args[2] = (word >> ODSY_STENCIL_VALUE_MASK_SHIFT) & ODSY_STENCIL_VALUE_MASK_MASK;
                self.gl_feed(SGI_GL_OP_STENCIL_FUNC as u16, &args);

                let fail = stencil_op((word >> ODSY_STENCIL_OP_FAIL_SHIFT) & ODSY_STENCIL_OP_MASK);
                let zfail = stencil_op((word >> ODSY_STENCIL_OP_ZFAIL_SHIFT) & ODSY_STENCIL_OP_MASK);
                let zpass = stencil_op((word >> ODSY_STENCIL_OP_ZPASS_SHIFT) & ODSY_STENCIL_OP_MASK);
                if fail == u32::MAX || zfail == u32::MAX || zpass == u32::MAX {
                    crate::dlog!(LogModule::Odyssey, "sgi-odyssey: stencil op word {:#x} is not modeled", word);
                } else {
                    args[0] = fail;
                    args[1] = zfail;
                    args[2] = zpass;
                    self.gl_feed(SGI_GL_OP_STENCIL_OP as u16, &args);
                }
            }
            ODSY_REG_BLEND => {
                let mut args = [0u32; 4];

                // Every operand is kept whatever bit 17 says (ODSY_MODE_BLEND): the
                // library sends glBlendFunc's factors, and the X server its
                // opaque-stipple background, before the 0x0404 that sets the bit.
                self.const_colour[0] = self.cmd_operand[1];
                self.const_colour[1] = self.cmd_operand[2];
                self.blend_word = self.cmd_operand[0];
                let word = self.cmd_operand[0];
                let eq = (word >> ODSY_BLEND_EQ_SHIFT) & ODSY_BLEND_EQ_MASK;
                let src = blend_factor((word >> ODSY_BLEND_SRC_SHIFT) & ODSY_BLEND_SRC_MASK, false);
                let dst = blend_factor(word & ODSY_BLEND_DST_MASK, true);
                if src == u32::MAX || dst == u32::MAX {
                    crate::dlog!(LogModule::Odyssey, "sgi-odyssey: blend factor word {:#x} is not modeled", word);
                } else {
                    args[0] = src;
                    args[1] = dst;
                    self.gl_feed(SGI_GL_OP_BLEND_FUNC as u16, &args[..2]);
                }
                if eq == 3 || eq == 4 {
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: blend equation {} (the SGIX alpha min/max extension) is not modeled",
                        eq
                    );
                } else {
                    args[0] = BLEND_EQUATION[eq as usize];
                    self.gl_feed(SGI_GL_OP_BLEND_EQUATION as u16, &args[..1]);
                }
                let c1 = self.cmd_operand[1];
                let c2 = self.cmd_operand[2];
                args[0] = f32_bits(((c1 >> ODSY_BLEND_COLOUR_SHIFT) & ODSY_BLEND_COLOUR_MASK) as f32 / ODSY_BLEND_COLOUR_SCALE);
                args[1] = f32_bits((c1 & ODSY_BLEND_COLOUR_MASK) as f32 / ODSY_BLEND_COLOUR_SCALE);
                args[2] = f32_bits(((c2 >> ODSY_BLEND_COLOUR_SHIFT) & ODSY_BLEND_COLOUR_MASK) as f32 / ODSY_BLEND_COLOUR_SCALE);
                args[3] = f32_bits((c2 & ODSY_BLEND_COLOUR_MASK) as f32 / ODSY_BLEND_COLOUR_SCALE);
                self.gl_feed(SGI_GL_OP_BLEND_COLOR as u16, &args);
            }
            ODSY_REG_LOGIC_OP => {
                let mode = SGI_GL_CLEAR + (self.cmd_operand[0] & ODSY_LOGIC_OP_MASK);

                self.gl_feed(SGI_GL_OP_LOGIC_OP as u16, &[mode]);
            }
            ODSY_REG_LINE_POINT_SIZE => {
                let mut args = [0u32; 2];

                args[0] = f32_bits(
                    (self.cmd_operand[0] & ODSY_LINE_POINT_SIZE_MASK) as f32 / ODSY_LINE_POINT_SIZE_SCALE,
                );
                self.gl_feed_raster_state(SGI_GL_OP_LINE_WIDTH as u16, &args[0..1]);
                args[1] = f32_bits(
                    (self.cmd_operand[1] & ODSY_LINE_POINT_SIZE_MASK) as f32 / ODSY_LINE_POINT_SIZE_SCALE,
                );
                self.gl_feed_raster_state(SGI_GL_OP_POINT_SIZE as u16, &args[1..2]);
            }
            ODSY_REG_POLYGON_MODE => {
                let mut front = [0u32; 2];
                let mut back = [0u32; 2];

                front[0] = SGI_GL_FRONT;
                front[1] = polygon_mode(
                    (self.cmd_operand[0] >> ODSY_POLYGON_MODE_FRONT_SHIFT) & ODSY_POLYGON_MODE_MASK,
                );
                self.gl_feed_raster_state(SGI_GL_OP_POLYGON_MODE as u16, &front);
                back[0] = SGI_GL_BACK;
                back[1] = polygon_mode(
                    (self.cmd_operand[0] >> ODSY_POLYGON_MODE_BACK_SHIFT) & ODSY_POLYGON_MODE_MASK,
                );
                self.gl_feed_raster_state(SGI_GL_OP_POLYGON_MODE as u16, &back);
            }
            ODSY_REG_POLYGON_OFFSET => {
                let mut args = [0u32; 2];

                args[0] = f32_bits(0.0f32);
                args[1] = f32_bits(
                    sextract32(self.cmd_operand[0], 0, ODSY_POLYGON_OFFSET_BITS) as f32 * ODSY_POLYGON_OFFSET_SCALE,
                );
                self.gl_feed(SGI_GL_OP_POLYGON_OFFSET_EXT as u16, &args);
            }
            ODSY_REG_CLIP_BOX => {
                self.gl_clip_box[0] = self.cmd_operand[0];
                self.gl_clip_box[1] = self.cmd_operand[1];
                self.gl_apply_clip_box();
            }
            ODSY_REG_FOG => {
                let n = self.gl_fog.len();
                self.gl_fog.copy_from_slice(&self.cmd_operand[..n]);
            }
            ODSY_REG_TEX_MODE => {
                self.tex_mode = self.cmd_operand[0];
                self.gl_apply_clip_planes();
            }
            ODSY_REG_TEX_SAMPLER => {
                let n = self.tex_sampler.len();
                self.tex_sampler.copy_from_slice(&self.cmd_operand[..n]);
            }
            ODSY_REG_TEX_ENV => {
                let n = self.tex_env.len();
                self.tex_env.copy_from_slice(&self.cmd_operand[..n]);
            }
            ODSY_REG_TEX_LEVEL0 => {
                self.tex_level0 = self.cmd_operand[0];
            }
            ODSY_REG_TEX_LOD => {
                let n = self.tex_lod.len();
                self.tex_lod.copy_from_slice(&self.cmd_operand[..n]);
            }
            ODSY_REG_TEX_SCALE_BIAS => {
                let n = self.tex_scale_bias.len();
                self.tex_scale_bias.copy_from_slice(&self.cmd_operand[..n]);
            }
            ODSY_REG_TEX_OFFSET => {
                let n = self.tex_offset.len();
                self.tex_offset.copy_from_slice(&self.cmd_operand[..n]);
            }
            ODSY_REG_POST_FILTER_DEFAULT => {
                self.cmd_unimplemented(tag);
            }
            ODSY_REG_FIFO_SETUP_FIRST..=ODSY_REG_FIFO_SETUP_LAST => {
                self.fifo_setup[(tag - ODSY_REG_FIFO_SETUP_FIRST) as usize] = self.cmd_operand[0];
            }
            ODSY_REG_FIFO_SETUP_7B5 => {
                self.fifo_setup[ODSY_FIFO_SETUP_7B5_INDEX] = self.cmd_operand[0];
            }
            ODSY_GL_TEXCOORD_1F => {
                self.tex_coord[0] = self.cmd_operand[0];
                self.tex_coord[1] = 0;
                self.tex_coord_r = 0;
                self.tex_coord_q = ODSY_F32_ONE;
            }
            ODSY_GL_TEXCOORD_2F => {
                self.tex_coord.copy_from_slice(&self.cmd_operand[..2]);
                self.tex_coord_r = 0;
                self.tex_coord_q = ODSY_F32_ONE;
            }
            ODSY_GL_TEX_RECT => {
                if !self.texture_build_level() && !self.tex_rect_reported {
                    self.tex_rect_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: textured rectangle into buffer {:#x} has unsupported construction state or missing backing",
                        self.dest_base
                    );
                }
            }
            ODSY_REG_GL_BEGIN => {
                // Record the primitive and, for an image, open the raster path. The
                // shared GL path opens on a float vertex, an integer GL stream,
                // a textured integer vertex or an assembly absent from the 2D
                // interpreter. Integer X11 geometry retains the 2D path.
                self.gl_primitive = self.cmd_operand[0];
                self.gl_vertex_count = 0;
                self.gl_begin_open = true;
                self.gl_integer_prim = false;
                self.gl_open_r = 0;
                self.gl_open_c = 0;
                self.gl_open_m3 = 0;
                self.gl_open_m4 = 0;
                self.gl_open_held = 0;
                // A polygon left open by a BEGIN without END encloses nothing.
                self.gl_polygon_close(false);
                self.image_active = self.cmd_operand[0] == ODSY_GL_PRIMITIVE_IMAGE;
                if self.image_active {
                    self.image_x1 = 0;
                    self.image_y1 = 0;
                    self.image_x2 = 0;
                    self.image_y2 = 0;
                    self.image_cur_x = 0;
                    self.image_cur_y = 0;
                    self.image_fmt = 0;
                    self.image_group_len = 0;
                    self.image_group_painted = 0;
                    self.image_bytes = 0;
                    self.image_nbytes = 0;
                }
            }
            ODSY_REG_GL_END => {
                // A GL geometry primitive closes on the shared context; the greeter's
                // QUADS fill (and the console text path's stray END) close on the 2D path.
                self.gl_begin_open = false;
                self.gl_open_held = 0;
                if self.gl_in_prim {
                    self.gl_feed(SGI_GL_OP_END as u16, &[]);
                    self.gl_in_prim = false;
                } else {
                    self.gl_end();
                }
                self.gl_integer_prim = false;
            }
            0x0400 => {
                // Pending primitive for context restore.
                self.gl_restart_type = self.cmd_operand[0];
            }
            0x0444 => {
                // Drop an outgoing primitive, without draw.
                if self.cmd_operand[0] == 0 && !self.gl_integer_prim && !(self.gl_begin_open && self.gl_primitive == 9) {
                    self.gl().discard_primitive();
                    self.gl_in_prim = false;
                    self.gl_begin_open = false;
                    self.gl_open_held = 0;
                    self.gl_restart_absorb = 0;
                    self.gl_restart_armed = false;
                } else if self.cmd_operand[0] != 0 {
                    self.cmd_unimplemented(0x10444);
                }
            }
            0x0850 => {
                let type_ = self.cmd_operand[0] & 0xf;

                if type_ == 0xf {
                    if !self.gl_integer_prim && !(self.gl_begin_open && self.gl_primitive == 9) {
                        self.gl().discard_primitive();
                        self.gl_in_prim = false;
                        self.gl_begin_open = false;
                        self.gl_open_held = 0;
                        self.gl_restart_absorb = 0;
                        self.gl_restart_armed = false;
                    }
                } else if type_ <= 8 {
                    self.gl_primitive = type_;
                    self.gl_restart_type = type_;
                    self.gl_restart_absorb = (self.cmd_operand[0] >> 19) & 3;
                    self.gl_restart_total = self.gl_restart_absorb;
                    self.gl_restart_armed = true;
                    self.gl_restart_select = 0;
                    self.gl_begin_open = true;
                    self.gl_integer_prim = false;
                    self.gl_open_r = 0;
                    self.gl_open_c = 0;
                    self.gl_open_m3 = 0;
                    self.gl_open_m4 = 0;
                    self.gl_open_held = 0;
                } else {
                    self.cmd_unimplemented(0x10850);
                }
            }
            0x0740 => {
                self.gl_restart_select = self.cmd_operand[0];
                if self.gl_restart_select != 0x1e0 {
                    self.cmd_unimplemented(0x740);
                }
            }
            0x0741 => {
                if self.gl_restart_select == 0x1e0 && self.gl_restart_armed {
                    let word = self.cmd_operand[0];

                    self.gl_restart_c = (word >> 21) & 7;
                    self.gl_restart_m3 = (word >> 12) & 3;
                    self.gl_restart_m4 = (word >> 14) & 3;
                    self.restart_finish();
                } else {
                    self.cmd_unimplemented(0x741);
                }
            }
            ODSY_CMD_IMAGE_HEADER => {
                // The transfer header opens a pixel group: its count (bits 23:20, 15
                // standing for 16, 0 meaning a full burst) is how many byte-format
                // pixels or SHORT-format words the group's first burst carries; any
                // more words in that burst are padding, and the bursts after it are
                // full, running on to the box edge. Reset the per-group count and
                // any partial pixel.
                //
                // That the count covers only the first burst is INFERRED from the
                // X server's emitter: a colour-index image 56 pixels wide is sent
                // as a header with count 0 and three 16-pixel bursts, then for each
                // later group a header with count 8 and bit 16 set, an 8-pixel
                // burst finishing the previous row, and three 16-pixel bursts. A
                // count that bounded the whole group drew 8 pixels of every 56 and
                // left the root window's background in grey bands.
                let glen = (self.cmd_operand[0] >> ODSY_IMAGE_HEADER_GROUP_SHIFT) & ODSY_IMAGE_HEADER_GROUP_MASK;

                self.image_group_len = if glen == ODSY_IMAGE_HEADER_GROUP_MASK {
                    ODSY_IMAGE_HEADER_GROUP_FULL
                } else {
                    glen
                };
                self.image_group_painted = 0;
                self.image_code = self.cmd_operand[0] & ODSY_IMAGE_CODE_MASK;
                self.image_nbytes = 0;
                self.image_bytes = 0;
            }
            ODSY_REG_PATTERN_SHAPE => {
                self.pattern_shape = self.cmd_operand[0];
                self.gl_apply_polygon_stipple();
            }
            ODSY_REG_PATTERN_MODE => {
                self.pattern_mode = self.cmd_operand[0];
                self.gl_apply_mode();
            }
            ODSY_REG_WINDOW_ORIGIN => {
                self.win_origin[0] = sextract32(self.cmd_operand[0], 0, ODSY_ORIGIN_BITS);
                self.win_origin[1] = sextract32(self.cmd_operand[1], 0, ODSY_ORIGIN_BITS);
                if self.cmd_from_kernel {
                    self.win_prior_origin = self.win_committed_origin;
                    self.win_committed_origin = self.win_origin;
                    self.win_origin_pending = true;
                }
            }
            ODSY_REG_ID_COMPARE => {
                self.id_compare = self.cmd_operand[1];
                if self.cmd_from_kernel && !after_tail {
                    self.win_committed_id_compare = self.id_compare;
                }
                // A non-zero pair still ends a pending query (which clears the
                // operands), as when the word was undecoded: whether the board does
                // is unknown, and no traced query has one inside it.
                if self.cmd_operand[0] != 0 || self.cmd_operand[1] != 0 {
                    self.query_cancel();
                }
            }
            ODSY_REG_WINDOW_CLIP => {
                let n = self.clip_words.len();
                self.clip_words.copy_from_slice(&self.cmd_operand[..n]);
                if self.cmd_from_kernel
                    && self.win_origin_pending
                    && self.win_origin[0] == 0
                    && self.win_origin[1] == 0
                    && self.clip_is_offscreen()
                {
                    // The off-screen tail: keep the window pushed before it.
                    self.win_committed_origin = self.win_prior_origin;
                    self.win_after_tail = true;
                } else if self.cmd_from_kernel {
                    self.win_committed_clip = self.clip_words;
                }
                if self.cmd_from_kernel {
                    self.win_origin_pending = false;
                }
                self.clip_update();
            }
            ODSY_CMD_WINDOW_RESTORE => {
                // The committed window back (ODSY_CMD_WINDOW_RESTORE; INFERRED).
                self.win_origin = self.win_committed_origin;
                self.clip_words = self.win_committed_clip;
                self.id_compare = self.win_committed_id_compare;
                self.clip_update();
            }
            ODSY_REG_LINE_STIPPLE => {
                let n = self.line_stipple.len();
                self.line_stipple.copy_from_slice(&self.cmd_operand[..n]);
                self.gl_apply_line_stipple();
                if self.line_stipple[ODSY_LINE_STIPPLE_CONTROL] != ODSY_LINE_STIPPLE_CONTROL_32
                    && (self.line_stipple[ODSY_LINE_STIPPLE_CONTROL] & !0xffu32) != 0
                    && !self.line_stipple_reported
                {
                    self.line_stipple_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: line stipple control {:#x} has unclassified high bits",
                        self.line_stipple[ODSY_LINE_STIPPLE_CONTROL]
                    );
                }
            }
            ODSY_REG_QUICK_FLUSH_ARG
            | ODSY_REG_STROBE
            | ODSY_REG_FULL_FLUSH_EXTRA
            | ODSY_REG_PIXEL_MODE_EXT
            | ODSY_REG_BLIT_PREFIX => {}
            ODSY_SWAP_SYNC_A => {
                // The bound drawing and reading drawable words, kept raw and
                // independently (ODSY_REG_DRAW_MASK): neither changes where the
                // model draws. The swap completes on the retrace hook
                // (retrace()).
                self.draw_word = self.cmd_operand[0];
                crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: buffer word: reg {:#x} word {:#x}", tag, self.draw_word);
            }
            ODSY_SWAP_SYNC_B => {
                self.read_word = self.cmd_operand[0];
                crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: buffer word: reg {:#x} word {:#x}", tag, self.read_word);
            }
            ODSY_REG_READ_ORIGIN => {
                self.read_origin[0] = sextract32(self.cmd_operand[0], 0, 16);
                self.read_origin[1] = sextract32(self.cmd_operand[0], 16, 16);
            }
            0x0000060a | 0x0000060c..=ODSY_REG_DRAWABLE_LAST => {
                if !self.drawable_word_reported {
                    self.drawable_word_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: drawable word {:#x} with operand {:#x} is consumed but not modeled",
                        tag,
                        self.cmd_operand[0]
                    );
                }
            }
            ODSY_REG_DRAW_MASK => {
                // The drawing mask: which colour buffers the next writes reach.
                self.draw_mask = self.cmd_operand[0];
                crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: draw mask: mask {:#x}", self.draw_mask);
                if (self.draw_mask & !ODSY_DRAW_MASK_BITS) != 0 && !self.draw_mask_high_reported {
                    self.draw_mask_high_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: drawing mask {:#x} bits above 3 are not modeled; ignored",
                        self.draw_mask
                    );
                }
                if (self.draw_mask & ODSY_DRAW_MASK_RIGHT) != 0 && !self.draw_mask_stereo_reported {
                    self.draw_mask_stereo_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: stereo is not modeled; drawing mask {:#x} draws its right-eye buffers into the left-eye ones",
                        self.draw_mask
                    );
                }
            }
            ODSY_REG_HOST_WRITE => {
                let op0 = self.cmd_operand[0];
                let op1 = self.cmd_operand[1];
                // The standalone host-write/sync command is the board's own DMA
                // completion signal, so it raises the same fence the DMA engine's
                // completion does, through the one fence implementation: op0's slot
                // and interrupt bit and op1's token become the completion cookie,
                // raised by Board::fence(). op0's interrupt
                // bit and op1's token already sit at the fence word's own positions
                // (ODSY_HOST_WRITE_INTERRUPT is the fence word's bit 15).
                let word: u32 = ((op1 & 0xffff) << ODSY_DMA_FENCE_TOKEN_SHIFT)
                    | (op0 & (ODSY_DMA_FENCE_INTERRUPT | ODSY_DMA_FENCE_SLOT_MASK));

                self.fence(ODSY_FENCE_VALID | word as u64);
            }
            ODSY_DMA_DESC_UPLOAD => {
                self.dma_descriptor(false);
            }
            ODSY_DMA_DESC_READBACK => {
                self.dma_descriptor(true);
            }
            _ => {
                // A recognized register-select word (valid shape, field-derived
                // arity) whose side effect this model does not track -- a mode, mask
                // or state register. Consumed like the flush and pixel-mode registers
                // above rather than treated as impossible.
            }
        }
    }

    pub fn cmd_start(&mut self, tag: u32, operands: u32) {
        self.cmd_tag = tag;
        self.cmd_operand_count = operands;
        self.cmd_operand_index = 0;
        if operands == 0 {
            self.query_update(tag);
            if tag != 0x2f00c02e && tag != 0x3000c02e {
                self.cmd_dispatch(tag);
            }
        } else {
            self.cmd_state = ODSY_CMD_OPERANDS;
        }
    }

    // A register-select word (0x0001rrrr): the most common command word, since
    // every flush and state write is one. No other word classify matches has
    // this shape, so it is tested first rather than after every by-value match.
    pub fn cmd_classify_register(&mut self, word: u32) {
        let reg = word & !ODSY_CMD_REGISTER_SHAPE;
        let operands = cmd_arity(reg);
        let mut fmt = 0u32;

        // A pixel burst while an image is open: its data words paint into the
        // box rather than streaming through the generic operand path. Outside
        // an image the same register-select is an ordinary state word.
        if self.image_active && operands > 0 && image_burst_fmt(reg, &mut fmt) {
            self.image_fmt = fmt;
            if !image_pairing_supported(self.image_code, fmt) {
                self.cmd_unimplemented(word);
            }
            self.cmd_state = ODSY_CMD_IMAGE_BURST;
            self.cmd_operand_count = operands;
            self.cmd_operand_index = 0;
            return;
        }

        // The 32x32-bit pattern loads: a 16-word burst each, more than
        // cmd_operand[] holds, so they stream straight into the pattern buffer
        // (0x3c4e -> words 0-15, 0x3c4f -> words 16-31) rather than through the
        // generic operand path.
        if reg == ODSY_REG_PATTERN_LOW || reg == ODSY_REG_PATTERN_HIGH {
            self.cmd_state = ODSY_CMD_PATTERN_LOAD;
            self.cmd_operand_count = operands;
            self.cmd_operand_index = 0;
            self.pattern_load_offset = if reg == ODSY_REG_PATTERN_HIGH { ODSY_PATTERN_WORDS } else { 0 };
            return;
        }
        self.cmd_start(reg, operands);
    }

    /// Classify a word arriving while the parser is idle: a new command.
    pub fn cmd_classify(&mut self, word: u32) {
        if (word & ODSY_CMD_REGISTER_SHAPE) == ODSY_CMD_REGISTER_TAG {
            self.cmd_classify_register(word);
            return;
        }
        if word == 0x4c0 || word == 0x4c0 + ODSY_QUERY_KIND_RASTER {
            self.query_begin();
            self.query_kind = word - 0x4c0;
            return;
        }
        if word == 0x2f00c02e || word == 0x3000c02e {
            self.cmd_start(word, 0);
            return;
        }
        if word == 0x740 || word == 0x741 {
            self.cmd_start(word, 1);
            return;
        }
        if (ODSY_REG_FIFO_SETUP_FIRST..=ODSY_REG_FIFO_SETUP_LAST).contains(&word)
            || word == ODSY_REG_FIFO_SETUP_7B5
        {
            self.cmd_start(word, 1);
            return;
        }

        if word == ODSY_CMD_COPY_RECT_TRIGGER {
            self.cmd_state = ODSY_CMD_COPY_RECT;
            self.cmd_operand_index = 0;
            return;
        }
        // A page-table DMA descriptor: the opcode word, then three more words the
        // generic operand path gathers before Board::cmd_dispatch() runs the
        // transfer (the two-doubleword descriptor reaches here as four words).
        if word == ODSY_DMA_DESC_UPLOAD || word == ODSY_DMA_DESC_READBACK {
            self.cmd_start(word, ODSY_DMA_DESC_OPERANDS);
            return;
        }
        // The two buffer-select doublewords: each opcode word takes one operand
        // word. Consumed cleanly so the parser stays word-aligned; the swap
        // completion is delivered from the retrace hook, not here (see the
        // ODSY_SWAP_SYNC_* constants and retrace()).
        if word == ODSY_SWAP_SYNC_A || word == ODSY_SWAP_SYNC_B {
            self.cmd_start(word, ODSY_SWAP_SYNC_OPERANDS);
            return;
        }
        if word == ODSY_REG_READ_ORIGIN {
            self.cmd_start(word, 1);
            return;
        }
        if (ODSY_REG_DRAWABLE_FIRST..=ODSY_REG_DRAWABLE_LAST).contains(&word) {
            // 0x60a, 0x60c to 0x60f; the three known ones are matched above.
            self.cmd_start(word, 1);
            return;
        }
        if word == ODSY_GL_COLOR_3UB {
            self.cmd_start(word, 3);
            return;
        }
        if word == ODSY_GL_VERTEX_2I {
            self.cmd_start(word, 2);
            return;
        }
        if word == ODSY_GL_TEXCOORD_1F
            || word == ODSY_GL_TEXCOORD_3F
            || word == ODSY_GL_TEXCOORD_4F
            || word == ODSY_GL_TEXCOORD_1I
            || word == ODSY_GL_TEXCOORD_3I
            || word == ODSY_GL_TEXCOORD_4I
            || (word & ODSY_GL_TEXCOORD_SHAPE) == ODSY_GL_TEXCOORD_TAG
            || word == ODSY_GL_TEX_RECT
        {
            self.cmd_start(word, cmd_arity(word));
            return;
        }
        if word == ODSY_CMD_WINDOW_RESTORE {
            self.cmd_start(word, 0);
            return;
        }
        if word == ODSY_GL_COLOR_INDEX
            || word == ODSY_GL_REPLAY_INDEX
            || word == ODSY_GL_COLOR_INDEX_F
            || word == ODSY_GL_REPLAY_INDEX_F
            || word == ODSY_GL_INDEX_MODE_COLOR_3I
            || word == ODSY_GL_INDEX_MODE_COLOR_3F
            || word == ODSY_GL_INDEX_MODE_COLOR_4F
            || word == ODSY_GL_COLOR_3I
            || word == ODSY_GL_COLOR_4F
            || is_integer_colour(word)
            || word == ODSY_GL_NORMAL_3B
            || word == ODSY_GL_NORMAL_3S
            || word == ODSY_GL_NORMAL_3I
        {
            self.cmd_start(word, cmd_arity(word));
            return;
        }
        // The true-colour GL words, matched by value: the float colour and vertex
        // siblings, the three matrix target-selects (0 operands each) and the
        // matrix load (16 floats, more than cmd_operand[] holds, so it streams into
        // gl_matrix[] through its own state).
        if word == ODSY_GL_COLOR_3F {
            self.cmd_start(word, 3);
            return;
        }
        if word == ODSY_GL_VERTEX_2F {
            self.cmd_start(word, 2);
            return;
        }
        if word == ODSY_GL_VERTEX_3F || word == ODSY_GL_VERTEX_3I {
            self.cmd_start(word, 3);
            return;
        }
        if word == ODSY_GL_VERTEX_4F {
            self.cmd_start(word, 4);
            return;
        }
        if word == ODSY_GL_LOAD_4F {
            self.cmd_start(word, ODSY_GL_BLOCK16_FLOATS);
            return;
        }
        // The lighting words, matched by value; each carries its operand count
        // in bits 13:10 like every other word. The program words carry none and
        // are matched by class ahead of the image path's 0x29 state word, which
        // is the same class (is_program_word()).
        if GL_LIGHT_WORDS.contains(&word) {
            self.cmd_start(word, cmd_arity(word));
            return;
        }
        if is_program_word(word) {
            self.cmd_start(word, 0);
            return;
        }
        if (word & ODSY_GL_MATRIX_SELECT_SHAPE) == ODSY_GL_MATRIX_SELECT_TAG {
            self.cmd_start(word, 0);
            return;
        }
        if word == ODSY_GL_MATRIX_LOAD {
            // 16 floats, more than cmd_operand[] holds, so gathered in gl_matrix[].
            self.cmd_state = ODSY_CMD_MATRIX_LOAD;
            self.cmd_operand_count = ODSY_GL_MATRIX_FLOATS;
            self.cmd_operand_index = 0;
            return;
        }
        if word == ODSY_GL_RECT || word == ODSY_GL_CLEAR_BOX {
            // 10 floats, more than cmd_operand[] holds: gathered in gl_rect[].
            self.cmd_state = ODSY_CMD_GL_RECT;
            self.cmd_operand_count = ODSY_GL_RECT_FLOATS;
            self.cmd_operand_index = 0;
            return;
        }
        if word == ODSY_MODE_INDEX_ZEROS {
            self.cmd_start(word, 3);
            return;
        }
        // The BEGIN-0xa image raster path's structural words, matched by value
        // rather than as register-selects. The raster-box floats need indices 4,
        // 5, 7 and 8 -- past what cmd_operand[] holds -- so they stream through a
        // dedicated state; the state word 0x29008000 | v and its teardown twin (0
        // operands), the transfer header (1) and its tail (0) run the generic
        // path, the header setting the group length as it dispatches.
        if word == ODSY_CMD_IMAGE_BOX {
            self.cmd_state = ODSY_CMD_IMAGE_BOX_LOAD;
            self.cmd_operand_count = cmd_arity(word);
            self.cmd_operand_index = 0;
            return;
        }
        if (word & ODSY_CMD_STATE_WORD_MASK) == ODSY_CMD_STATE_WORD
            || word == ODSY_CMD_IMAGE_HEADER
            || word == ODSY_CMD_IMAGE_HEADER_TAIL
        {
            self.cmd_start(word, cmd_arity(word));
            return;
        }
        // A word this model does not know still carries its operand count in
        // bits 13:10 (cmd_arity()), so pass over that many words and
        // keep the parser on the stream's command positions. The kernel's
        // context restore replays words of every class this way: 0x3cf8 to
        // 0x3cfb and 0x3f41 with 16 zero operands each, 0x0f41 with 3, 0x04b0
        // and 0xc480c426 with one, 0x2400800a with none. Taking their operands
        // as commands instead read zeros and floats as command words.
        //
        // A word of the GL classes (a non-zero top byte) with bit 23 clear takes
        // none, whatever its count field says: the restore's 0x9000bda2, whose
        // field reads 16, is followed at once by a flush, the window origin
        // (0x0841) and the clip list, which it would otherwise swallow, leaving
        // the next client drawing at the previous one's origin. Every bit-23-
        // clear GL word the model knows has a zero field and no operands, and
        // every one with operands has the bit set (0x9080bda2, 0x90808800).
        // INFERRED from that pattern; the bit's meaning on the board is UNKNOWN.
        if cmd_arity(word) != 0 && ((word >> 24) == 0 || (word & ODSY_CMD_GL_OPERANDS) != 0) {
            let note = format!("; skipping {} operands", cmd_arity(word));
            self.cmd_report(word, &note);
            self.cmd_state = ODSY_CMD_SKIP;
            self.cmd_operand_count = cmd_arity(word);
            self.cmd_operand_index = 0;
        } else if cmd_arity(word) != 0 {
            self.cmd_report(word, "; bit 23 clear, no operands");
        } else {
            self.cmd_report(word, "; no operands");
        }
    }

    /// One command-FIFO word, wherever the parser currently is.
    pub fn cmd_word(&mut self, word: u32) {
        match self.query_stage {
            1 => {
                self.query_count = word;
                self.query_stage = 2;
                return;
            }
            2 => {
                if word == 0x10a5a {
                    self.query_stage = 3;
                    return;
                }
                self.query_stage = 0;
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: unpaired current-query sequence");
                self.cmd_classify(word);
                return;
            }
            3 => {
                self.query_sync_index = word;
                self.query_stage = 4;
                return;
            }
            4 => {
                self.query_finish(word);
                return;
            }
            _ => {}
        }
        match self.cmd_state {
            ODSY_CMD_SKIP => {
                self.cmd_operand_index += 1;
                if self.cmd_operand_index == self.cmd_operand_count {
                    self.cmd_state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_OPERANDS => {
                // cmd_operand[] holds a whole burst; the bound is belt and braces.
                if (self.cmd_operand_index as usize) < self.cmd_operand.len() {
                    self.cmd_operand[self.cmd_operand_index as usize] = word;
                }
                self.cmd_operand_index += 1;
                if self.cmd_operand_index == self.cmd_operand_count {
                    let tag = self.cmd_tag;
                    self.query_update(tag);
                    self.cmd_dispatch(tag);
                    self.cmd_state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_COPY_RECT => {
                self.cmd_operand[self.cmd_operand_index as usize] = word;
                match self.cmd_operand_index {
                    ODSY_CMD_COPY_RECT_SRC_INDEX => self.copy_src = word,
                    ODSY_CMD_COPY_RECT_SIZE_INDEX => self.copy_size = word,
                    ODSY_CMD_COPY_RECT_DST_INDEX => self.copy_dst = word,
                    _ => {}
                }
                self.cmd_operand_index += 1;
                if self.cmd_operand_index == ODSY_CMD_COPY_RECT_WORDS {
                    self.copy_rect();
                    self.cmd_state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_PATTERN_LOAD => {
                // A 16-word burst into the pattern buffer at the load's own offset.
                if self.cmd_operand_index < ODSY_PATTERN_WORDS {
                    self.pattern_buffer[(self.pattern_load_offset + self.cmd_operand_index) as usize] = word;
                }
                self.cmd_operand_index += 1;
                if self.cmd_operand_index == self.cmd_operand_count {
                    self.gl_apply_polygon_stipple();
                    self.cmd_state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_IMAGE_BOX_LOAD => {
                // The ten raster-box floats. Only 4, 5, 7 and 8 carry the destination
                // corners; the rest are the zero the DDX's private was allocated with.
                // Set the cursor to the top-left corner once all ten have arrived.
                match self.cmd_operand_index {
                    ODSY_IMAGE_BOX_X1_INDEX => self.image_x1 = float_to_u32(word),
                    ODSY_IMAGE_BOX_Y1_INDEX => self.image_y1 = float_to_u32(word),
                    ODSY_IMAGE_BOX_X2_INDEX => self.image_x2 = float_to_u32(word),
                    ODSY_IMAGE_BOX_Y2_INDEX => self.image_y2 = float_to_u32(word),
                    _ => {}
                }
                self.cmd_operand_index += 1;
                if self.cmd_operand_index == self.cmd_operand_count {
                    self.image_cur_x = if self.image_x2 < self.image_x1 { self.image_x1.wrapping_sub(1) } else { self.image_x1 };
                    self.image_cur_y = if self.image_y2 < self.image_y1 { self.image_y1.wrapping_sub(1) } else { self.image_y1 };
                    self.cmd_state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_IMAGE_BURST => {
                // Each data word paints into the open image at the raster cursor.
                self.image_burst_word(word);
                self.cmd_operand_index += 1;
                if self.cmd_operand_index == self.cmd_operand_count {
                    // The header's count bounds only the group's first burst; the
                    // bursts after it are full (ODSY_CMD_IMAGE_HEADER).
                    self.image_group_len = 0;
                    self.cmd_state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_MATRIX_LOAD => {
                // A board-side matrix: 16 floats, loaded once all have arrived.
                if self.cmd_operand_index < ODSY_GL_MATRIX_FLOATS {
                    self.gl_matrix[self.cmd_operand_index as usize] = word;
                }
                self.cmd_operand_index += 1;
                if self.cmd_operand_index == self.cmd_operand_count {
                    self.gl_load_matrix();
                    self.cmd_state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_GL_RECT => {
                // The rectangle's colour and corners, painted once all have arrived.
                self.gl_rect[self.cmd_operand_index as usize] = word;
                self.cmd_operand_index += 1;
                if self.cmd_operand_index == self.cmd_operand_count {
                    self.gl_rect();
                    self.cmd_state = ODSY_CMD_IDLE;
                }
            }
            _ => {
                self.cmd_classify(word);
            }
        }
    }

    /// The command FIFO accepts both 32-bit and 64-bit stores. An 8-byte store
    /// carries two sequential 32-bit command words; which half is logically
    /// first is not independently established by any allowed source, so it is
    /// modeled by analogy to the backend-selector port's own
    /// (selector << 32) | data convention and this device's big-endian byte
    /// order, upper half first. Other widths are not established for this
    /// port, so they are reported unimplemented rather than fed into the
    /// command stream and risking desync.
    pub fn cmd_fifo_write(&mut self, value: u64, size: u32) {
        match size {
            4 => self.cmd_word(value as u32),
            8 => {
                self.cmd_word((value >> 32) as u32);
                self.cmd_word(value as u32);
            }
            _ => self.unimplemented(ODSY_CMD_FIFO, size, true),
        }
    }
}

/// One step of the command-FIFO self-test preamble.
pub struct CfifoPreamble {
    pub port: u64,
    pub size: u32,
    pub value: u64,
}

/// The kernel driver's command-FIFO initialisation self-test preamble, in
/// order, all through the command FIFO: three 4-byte command words, five
/// 8-byte pushes (a 32-bit selector in the upper half, data in the lower),
/// and one closing 4-byte word. Seeing the whole sequence makes
/// ODSY_CFIFO_SELFTEST answer OK. (The real driver sends the five 8-byte
/// words to 0x110000, not to the backend-select port, confirmed by a boot
/// trace.)
pub static CFIFO_PREAMBLE: [CfifoPreamble; 9] = [
    CfifoPreamble { port: ODSY_CMD_FIFO, size: 4, value: 0x00010854 },
    CfifoPreamble { port: ODSY_CMD_FIFO, size: 4, value: 0x00000000 },
    CfifoPreamble { port: ODSY_CMD_FIFO, size: 4, value: 0x0fffffff },
    CfifoPreamble { port: ODSY_CMD_FIFO, size: 8, value: 0x000006b000000000 },
    CfifoPreamble { port: ODSY_CMD_FIFO, size: 8, value: 0x000006b100000100 },
    CfifoPreamble { port: ODSY_CMD_FIFO, size: 8, value: 0x000006b300005e50 },
    CfifoPreamble { port: ODSY_CMD_FIFO, size: 8, value: 0x000006b200007dc0 },
    CfifoPreamble { port: ODSY_CMD_FIFO, size: 8, value: 0x000007b500000001 },
    CfifoPreamble { port: ODSY_CMD_FIFO, size: 4, value: 0x00010046 },
];

impl Board {
    pub fn selftest_track(&mut self, port: u64, size: u32, value: u64) {
        let next = self.selftest_progress as usize;

        if next < CFIFO_PREAMBLE.len()
            && CFIFO_PREAMBLE[next].port == port
            && CFIFO_PREAMBLE[next].size == size
            && CFIFO_PREAMBLE[next].value == value
        {
            self.selftest_progress = next as u32 + 1;
            if self.selftest_progress as usize == CFIFO_PREAMBLE.len() {
                self.cfifo_selftest_ready = true;
                self.selftest_progress = 0;
            }
            return;
        }
        // A break in the sequence restarts it, counting a fresh first word.
        self.selftest_progress = if CFIFO_PREAMBLE[0].port == port
            && CFIFO_PREAMBLE[0].size == size
            && CFIFO_PREAMBLE[0].value == value
        {
            1
        } else {
            0
        };
    }
}
