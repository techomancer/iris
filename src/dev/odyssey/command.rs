//! The command-FIFO interpreter: word classification, operand gathering and
//! the one big dispatch of a completed command.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.


use crate::devlog::LogModule;
use super::bits::sextract32;
use super::board::{Board, ODSY_GL_LIGHTS, ODSY_GL_LIGHT_COLOURS, ODSY_TEX_LEVELS};
use super::consts::{
    alpha, blend, cmd, cmd_arity, dest, dma, gl_cmd, image, line, mode, pattern, polygon, prog,
    query, reg, stencil, swap, tex, write, ODSY_DEPTH_FUNC_MASK, ODSY_DRAW_MASK_BITS,
    ODSY_DRAW_MASK_RIGHT, ODSY_F32_ONE, ODSY_FENCE_VALID, ODSY_FIFO_SETUP_7B5_INDEX,
    ODSY_LOGIC_OP_MASK, ODSY_ORIGIN_BITS, ODSY_RASTER_POS_LATCH_WORD,
};
use super::dma::{
    float_channel, integer_colour, is_integer_colour, signed_to_float, ODSY_CMD_COPY_RECT,
    ODSY_CMD_GL_RECT, ODSY_CMD_IDLE, ODSY_CMD_IMAGE_BOX_LOAD, ODSY_CMD_IMAGE_BURST,
    ODSY_CMD_MATRIX_LOAD, ODSY_CMD_OPERANDS, ODSY_CMD_PATTERN_LOAD, ODSY_CMD_SKIP,
};
use super::draw::{float_to_u32, image_burst_fmt, pack_pixel};
use super::geometry::{f32_bits, odyssey_f32};
use super::gl::{
    blend_factor, float_index, is_program_word, polygon_mode, prog_value, stencil_op,
    BLEND_EQUATION, GL_LIGHT_WORDS,
};
use super::gl_context::{
    op, GL_BACK, GL_CLEAR, GL_CLIP_PLANES, GL_FRONT, GL_MODELVIEW,
    GL_NEVER_ENUM,
};
impl Board {
    pub fn cmd_dispatch(&mut self, tag: u32) {
        let mut after_tail = false;

        // Only the kernel command right after its window-origin write can be its tail.
        if self.cmd.from_kernel && tag != reg::WINDOW_ORIGIN && tag != reg::WINDOW_CLIP {
            self.win.origin_pending = false;
        }
        // And only the one right after the tail's 0x204b is its 0x840.
        if self.cmd.from_kernel {
            after_tail = self.win.after_tail;
            self.win.after_tail = false;
        }
        if is_program_word(tag) {
            self.gl_program_word(tag);
            return;
        }
        if (reg::TEX_DETAIL_FUNCTION..reg::TEX_DETAIL_FUNCTION + tex::DETAIL_FUNCTION_BLOCKS).contains(&tag)
        {
            let block = tag - reg::TEX_DETAIL_FUNCTION;

            if self.cmd.operand_count != tex::DETAIL_FUNCTION_BLOCK_WORDS {
                self.tex.detail_function_valid &= !(1u32 << block);
                self.cmd_unimplemented(tag);
                return;
            }
            let w = tex::DETAIL_FUNCTION_BLOCK_WORDS as usize;
            let at = block as usize * w;
            self.tex.detail_function[at..at + w].copy_from_slice(&self.cmd.operand[..w]);
            self.tex.detail_function_valid |= 1u32 << block;
            return;
        }
        if (reg::TEX_TABLE3..reg::TEX_TABLE3 + tex::TABLE3_BLOCKS).contains(&tag) {
            self.cmd_unimplemented(tag);
            return;
        }
        if (reg::TEX_COLOUR_TABLE0..reg::TEX_COLOUR_TABLE0 + tex::COLOUR_TABLE_BLOCKS).contains(&tag)
            || (reg::TEX_COLOUR_TABLE1..reg::TEX_COLOUR_TABLE1 + tex::COLOUR_TABLE_BLOCKS).contains(&tag)
        {
            let plane = (tag >= reg::TEX_COLOUR_TABLE1) as usize;
            let first_tag = if plane != 0 { reg::TEX_COLOUR_TABLE1 } else { reg::TEX_COLOUR_TABLE0 };
            let block = tag - first_tag;

            if self.cmd.operand_count != tex::COLOUR_TABLE_BLOCK_WORDS {
                self.cmd_unimplemented(tag);
                return;
            }
            let w = tex::COLOUR_TABLE_BLOCK_WORDS as usize;
            let at = block as usize * w;
            self.tex.colour_table[plane][at..at + w].copy_from_slice(&self.cmd.operand[..w]);
            self.tex.colour_table_valid[plane] |= 1u32 << block;
            return;
        }
        if (reg::TEX_LEVEL..reg::TEX_LEVEL + ODSY_TEX_LEVELS).contains(&tag) {
            let at = (tag - reg::TEX_LEVEL) as usize * 3;
            self.tex.levels[at..at + 3].copy_from_slice(&self.cmd.operand[..3]);
            return;
        }
        if tag == gl_cmd::TEXCOORD_3F || tag == gl_cmd::TEXCOORD_4F {
            self.tex.coord.copy_from_slice(&self.cmd.operand[..2]);
            self.tex.coord_r = self.cmd.operand[2];
            self.tex.coord_q = if tag == gl_cmd::TEXCOORD_4F { self.cmd.operand[3] } else { ODSY_F32_ONE };
            return;
        }
        if tag == gl_cmd::TEXCOORD_1I
            || tag == gl_cmd::TEXCOORD_2I
            || tag == gl_cmd::TEXCOORD_3I
            || tag == gl_cmd::TEXCOORD_4I
        {
            let n = cmd_arity(tag);
            let mut c: [u32; 4] = [0, 0, 0, ODSY_F32_ONE];

            for i in 0..n as usize {
                c[i] = f32_bits((self.cmd.operand[i] as i32) as f32);
            }
            self.tex.coord[0] = c[0];
            self.tex.coord[1] = c[1];
            self.tex.coord_r = c[2];
            self.tex.coord_q = c[3];
            return;
        }
        if (tag & gl_cmd::TEXCOORD_SHAPE) == gl_cmd::TEXCOORD_TAG && tag != gl_cmd::TEXCOORD_2F {
            // Other two-coordinate forms remain observable and change nothing.
            self.cmd_unimplemented(tag);
            return;
        }
        if (tag & gl_cmd::MATRIX_SELECT_SHAPE) == gl_cmd::MATRIX_SELECT_TAG {
            self.geom.regs_20_35[((gl_cmd::MATRIX_SELECT_TAG >> prog::CLASS_SHIFT) - prog::REG_FIRST) as usize] =
                prog_value(tag);
            // Pick the shared matrix slot the following load targets.
            self.gl_matrix_slot = match tag & 0xff {
                gl_cmd::MATRIX_SELECT_COMBINED => gl_cmd::SLOT_COMBINED,
                gl_cmd::MATRIX_SELECT_TEXTURE => gl_cmd::SLOT_TEXTURE,
                gl_cmd::MATRIX_SELECT_MODELVIEW => GL_MODELVIEW,
                gl_cmd::MATRIX_SELECT_BLOCK16 => gl_cmd::SLOT_BLOCK16,
                gl_cmd::MATRIX_SELECT_NORMAL => gl_cmd::SLOT_NORMAL_MATRIX,
                // Any other block: no home.
                _ => gl_cmd::SLOT_NORMAL,
            };
            return;
        }
        match tag {
            gl_cmd::COLOR_3UB => {
                self.stream.color = pack_pixel(self.cmd.operand[0], self.cmd.operand[1], self.cmd.operand[2]);
                self.gl_color_is_index = false;
                if !self.gl_index_mode() {
                    let opd = self.cmd.operand;

                    self.gl_feed(op::COLOR3UB as u16, &opd[..3]);
                    self.geom.colour = self.gl().current_color();
                    self.geom.colour_overridden = false;
                }
            }
            gl_cmd::COLOR_INDEX | gl_cmd::REPLAY_INDEX => {
                // The user's index word, or the kernel's replay of a saved one.
                self.gl_color_index = self.cmd.operand[0];
                self.gl_color_is_index = true;
            }
            gl_cmd::COLOR_INDEX_F | gl_cmd::REPLAY_INDEX_F => {
                self.gl_color_index = float_index(self.cmd.operand[0]);
                self.gl_color_is_index = true;
            }
            reg::DEST_SELECT => {
                // Destination-buffer select: its third operand marks an id-plane
                // write when it is dest::ID_PLANES. A colour select's non-zero
                // first operand names the colour buffer, the base a DMA header's
                // destination word is matched against (the all-zero blocks are GL
                // context setup, not a buffer).
                self.dest_buffer_word = self.cmd.operand[dest::SELECT_VALUE_INDEX];
                self.dest_base = self.cmd.operand[dest::SELECT_BASE_INDEX];
                if self.dest_base == 0 || self.dest_is_id_planes() {
                    return;
                }
                let base = self.dest_base;
                if !self.board_address_valid(base) || (self.dest_buffer_word & !dest::FLAGS) == 0 {
                    return;
                }
                if (self.dest_buffer_word >> dest::FORMAT_SHIFT) == dest::FORMAT_SCREEN
                    || self.dest_base == self.dma_colour_base
                {
                    self.dma_colour_base = self.dest_base;
                    if self.screen_base == 0 {
                        self.screen_base = self.dest_base;
                    }
                } else {
                    // An off-screen object: create it, or update its row length.
                    let width = self.cmd.operand[dest::WIDTH_INDEX].wrapping_mul(dest::WIDTH_UNIT);
                    self.object_select(base, width);
                }
            }
            reg::PLANE_MASK => {
                // Plane write mask (mask << 12), consumed by an id-plane write.
                self.plane_write_mask = self.cmd.operand[0];
            }
            mode::INDEX_ZEROS
            | gl_cmd::INDEX_MODE_COLOR_3I
            | gl_cmd::INDEX_MODE_COLOR_3F
            | gl_cmd::INDEX_MODE_COLOR_4F => {
                // An RGBA colour sent in index mode: GL does not draw with it.
            }
            gl_cmd::VERTEX_2I => {
                if self.prim.begin_open {
                    self.prim.integer_prim = true;
                }
                // Bit 13 distinguishes the observed GL and X11 integer streams.
                // Its hardware meaning is unknown. This is a model routing policy,
                // not a claim that the bit enables hardware vertex transforms.
                if self.prim.begin_open
                    && !self.image.active
                    && ((self.stream.pattern_mode & (1u32 << 13)) != 0
                        || (self.tex.mode & (tex::MODE_1D | tex::MODE_2D | tex::MODE_3D)) != 0
                        || self.stream.primitive == gl_cmd::TRIANGLES
                        || self.stream.primitive == gl_cmd::TRIANGLE_STRIP
                        || self.stream.primitive == gl_cmd::TRIANGLE_FAN
                        || self.stream.primitive == gl_cmd::QUAD_STRIP)
                {
                    // Lighting takes the signed integers as object coordinates.
                    let xyz: [f32; 3] = [
                        (self.cmd.operand[0] as i32) as f32,
                        (self.cmd.operand[1] as i32) as f32,
                        0.0,
                    ];
                    let opd = self.cmd.operand;

                    if !self.prim.in_prim {
                        let mode = self.stream.primitive;

                        self.gl_feed(op::BEGIN as u16, &[mode]);
                        self.prim.in_prim = true;
                    }
                    self.gl_vertex_colour(&xyz);
                    self.texture_vertex();
                    self.gl_feed(op::VERTEX2I as u16, &opd[..2]);
                } else {
                    // The X path uses window-relative integer coordinates.
                    let x = self.cmd.operand[0].wrapping_add(self.win.origin[0] as u32);
                    let y = self.cmd.operand[1].wrapping_add(self.win.origin[1] as u32);
                    self.gl_vertex(x, y);
                }
            }
            gl_cmd::COLOR_3F
            | gl_cmd::COLOR_3I
            | gl_cmd::COLOR_4F
            | gl_cmd::COLOR_3US
            | gl_cmd::COLOR_3UI
            | gl_cmd::COLOR_3B
            | gl_cmd::COLOR_3S
            | gl_cmd::COLOR_4UB
            | gl_cmd::COLOR_4US
            | gl_cmd::COLOR_4UI
            | gl_cmd::COLOR_4B
            | gl_cmd::COLOR_4S
            | gl_cmd::COLOR_4I => {
                // The current colour of the true-colour GL stream (glColor3f,
                // glColor4f and the integer forms), or the kernel's replay of one
                // Board::ctx_save() saved. Kept so an unlit vertex after lit ones
                // takes it back, and so colour material lights with it.
                let n = cmd_arity(tag);
                let mut rgba = [0u32; 4];

                rgba.copy_from_slice(&self.cmd.operand[..4]);
                if tag != gl_cmd::COLOR_3F && tag != gl_cmd::COLOR_4F {
                    for i in 0..n as usize {
                        rgba[i] = integer_colour(tag, self.cmd.operand[i]);
                    }
                }
                if n == 3 {
                    rgba[3] = ODSY_F32_ONE;
                }
                self.geom.colour.copy_from_slice(&rgba);
                self.geom.colour_overridden = false;
                if n == 4 {
                    self.gl_feed(op::COLOR4F as u16, &rgba[..4]);
                } else {
                    self.gl_feed(op::COLOR3F as u16, &rgba[..3]);
                }
                if !self.gl_index_mode() {
                    self.stream.color = pack_pixel(
                        float_channel(rgba[0]),
                        float_channel(rgba[1]),
                        float_channel(rgba[2]),
                    );
                    self.gl_color_is_index = false;
                }
            }
            gl_cmd::NORMAL_3B | gl_cmd::NORMAL_3S | gl_cmd::NORMAL_3I => {
                for i in 0..3 {
                    self.geom.normal[i] = signed_to_float(self.cmd.operand[i], tag & 0x3f);
                }
            }
            gl_cmd::NORMAL_3F => {
                self.geom.normal.copy_from_slice(&self.cmd.operand[..3]);
            }
            gl_cmd::MATERIAL_AMBIENT
            | gl_cmd::MATERIAL_DIFFUSE
            | gl_cmd::MATERIAL_SPECULAR
            | gl_cmd::MATERIAL_EMISSION => {
                // The low byte counts 0x2f, 0x30, 0x31, 0x32 through the four.
                let at = (((tag & 0xff) - (gl_cmd::MATERIAL_AMBIENT & 0xff)) * 4) as usize;
                self.geom.material[at..at + 4].copy_from_slice(&self.cmd.operand[..4]);
            }
            gl_cmd::SHININESS_SQRT => {
                self.geom.shininess_sqrt = self.cmd.operand[0];
            }
            gl_cmd::BACK_AMBIENT | gl_cmd::BACK_DIFFUSE | gl_cmd::BACK_SPECULAR | gl_cmd::BACK_EMISSION => {
                let at = (((tag & 0xff) - (gl_cmd::BACK_AMBIENT & 0xff)) * 4) as usize;
                self.geom.back_material[at..at + 4].copy_from_slice(&self.cmd.operand[..4]);
            }
            gl_cmd::BACK_SHININESS_SQRT => {
                self.geom.back_shininess_sqrt = self.cmd.operand[0];
            }
            gl_cmd::SHININESS_DERIVED | prog::RUN_ONCE => {
                // Consumed, not kept: the derived shininess floats and the one-shot
                // run have no modelled effect. Block 0x1ea (gl_cmd::SHININESS_DERIVED)
                // is also the second read path of attribute slots 448 and 544: every
                // full context replay sends it with the selected slot-448 entry's
                // first operand, that with bit 31 flipped, and the same for slot 544
                // (ESTABLISHED as the kernel's replay). What the block does on the
                // board is UNKNOWN; the model gives neither use an effect.
            }
            gl_cmd::CLIP_PLANE_LOAD => {
                // Public controls identify the effective S, T, R and Q planes.
                if self.geom.light_vec_sel == 0x10c {
                    self.tex.linear_plane_q.copy_from_slice(&self.cmd.operand[..4]);
                    self.tex.linear_valid |= 8;
                    return;
                }
                if self.geom.light_vec_sel == 0x108 {
                    self.tex.linear_plane_r.copy_from_slice(&self.cmd.operand[..4]);
                    self.tex.linear_valid |= 4;
                    return;
                }
                if self.geom.light_vec_sel == 0x100 || self.geom.light_vec_sel == 0x104 {
                    let axis = ((self.geom.light_vec_sel - 0x100) / 4) as usize;

                    self.tex.linear_plane[axis].copy_from_slice(&self.cmd.operand[..4]);
                    self.tex.linear_valid |= 1u32 << axis;
                    return;
                }
                // See gl_cmd::CLIP_PLANE_BASE's comment: accept and pass through
                // loads into the shared layer's six slots, silently drop loads past
                // that budget (expected library traffic, not an error), and report
                // only a selector this model cannot place at all.
                if self.geom.light_vec_sel >= gl_cmd::CLIP_PLANE_BASE
                    && (self.geom.light_vec_sel - gl_cmd::CLIP_PLANE_BASE).is_multiple_of(gl_cmd::CLIP_PLANE_STRIDE)
                {
                    let plane = (self.geom.light_vec_sel - gl_cmd::CLIP_PLANE_BASE) / gl_cmd::CLIP_PLANE_STRIDE;

                    if plane < GL_CLIP_PLANES {
                        let at = plane as usize * 4;
                        self.geom.clip_eye[at..at + 4].copy_from_slice(&self.cmd.operand[..4]);
                        self.gl_apply_clip_plane(plane);
                    }
                } else if !self.diag.gl_block_logged {
                    self.diag.gl_block_logged = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: class 0x31 clip-plane load at index {:#x} is not modeled",
                        self.geom.light_vec_sel
                    );
                }
            }
            gl_cmd::LIGHT_COLOUR_LOAD => {
                if self.geom.light_colour_sel < ODSY_GL_LIGHT_COLOURS {
                    let at = self.geom.light_colour_sel as usize * 4;
                    self.geom.light_colour[at..at + 4].copy_from_slice(&self.cmd.operand[..4]);
                } else if !self.diag.gl_block_logged {
                    self.diag.gl_block_logged = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: class 0x32 load at index {:#x} is not modeled",
                        self.geom.light_colour_sel
                    );
                }
            }
            gl_cmd::LIGHT_VEC_LOAD => {
                if self.geom.light_vec_sel == 0x100 {
                    self.tex.linear_plane[0].copy_from_slice(&self.cmd.operand[0..4]);
                    self.tex.linear_plane[1].copy_from_slice(&self.cmd.operand[4..8]);
                    self.tex.linear_plane_r.copy_from_slice(&self.cmd.operand[8..12]);
                    self.tex.linear_plane_q.copy_from_slice(&self.cmd.operand[12..16]);
                    self.tex.linear_valid |= 0xf;
                    return;
                }
                // Index 16 i is light i. Other context setup blocks at 0x80
                // to 0x1e0 remain unknown and are reported once, not per index.
                if self.geom.light_vec_sel.is_multiple_of(gl_cmd::LIGHT_VEC_FLOATS)
                    && self.geom.light_vec_sel / gl_cmd::LIGHT_VEC_FLOATS < ODSY_GL_LIGHTS
                {
                    let at = self.geom.light_vec_sel as usize;
                    let n = gl_cmd::LIGHT_VEC_FLOATS as usize;
                    self.geom.light_vec[at..at + n].copy_from_slice(&self.cmd.operand[..n]);
                } else if !self.diag.gl_block_logged {
                    self.diag.gl_block_logged = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: class 0x31 load at index {:#x} is not modeled",
                        self.geom.light_vec_sel
                    );
                }
            }
            gl_cmd::VERTEX_2F | gl_cmd::VERTEX_3F | gl_cmd::VERTEX_3I | gl_cmd::VERTEX_4F => {
                // A float vertex uses the shared GL path, and so does an integer 3D
                // vertex once converted to float (gl_cmd::VERTEX_3I). Open the
                // primitive lazily on its first vertex, as for shared integer
                // geometry above. The open-primitive record keeps the word and its
                // operands as sent. glVertex4f (gl_cmd::VERTEX_4F) passes its W on.
                let components: u32 = if tag == gl_cmd::VERTEX_4F {
                    4
                } else if tag == gl_cmd::VERTEX_2F {
                    2
                } else {
                    3
                };
                let mut xyz = [0u32; 4];
                xyz.copy_from_slice(&self.cmd.operand[..4]);
                let mut obj = [0f32; 3];

                if tag == gl_cmd::VERTEX_3I {
                    for k in 0..3 {
                        xyz[k] = f32_bits((self.cmd.operand[k] as i32) as f32);
                    }
                }
                if !self.prim.in_prim && self.prim.restart_absorb == 0 {
                    let mode = self.stream.primitive;

                    self.gl_feed(op::BEGIN as u16, &[mode]);
                    self.prim.in_prim = true;
                }
                if self.prim.begin_open && !self.prim.integer_prim && self.stream.primitive <= 8 {
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
                if self.prim.restart_absorb != 0 {
                    let slot = (self.prim.restart_total - self.prim.restart_absorb) as usize;

                    if slot < self.prim.restart_vertex.len() {
                        self.prim.restart_vertex[slot] =
                            self.gl_ctx.capture_vertex(self.prim.restart_type, &xyz, components);
                    }
                    self.prim.restart_absorb -= 1;
                    self.restart_finish();
                    return;
                }
                let op = if components == 4 {
                    op::VERTEX4F
                } else if components == 3 {
                    op::VERTEX3F
                } else {
                    op::VERTEX2F
                };
                self.gl_feed(op as u16, &xyz[..components as usize]);
            }
            gl_cmd::LOAD_4F => {
                self.gl_load_block16();
            }
            reg::DEPTH_FUNC => {
                if (self.cmd.operand[0] & !ODSY_DEPTH_FUNC_MASK) != 0 {
                    // Only bits 2:0 are known; every capture leaves the rest 0.
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: depth function word {:#x} has bits beyond 2:0, not modeled",
                        self.cmd.operand[0]
                    );
                }
                self.gl_depth_func = self.cmd.operand[0];
                self.gl_apply_depth();
            }
            reg::WRITE_MASK => {
                let stencil_mask = (self.cmd.operand[0] >> write::STENCIL_SHIFT) & write::STENCIL_MASK;

                self.gl_write_mask = self.cmd.operand[0];
                self.gl_apply_depth();
                // The library forces it to 0 for a drawable with no stencil.
                self.gl_feed(op::STENCIL_MASK as u16, &[stencil_mask]);
            }
            reg::ALPHA_FUNC => {
                let word = self.cmd.operand[0];
                let mut args = [0u32; 2];

                args[0] = GL_NEVER_ENUM + ((word >> alpha::FUNC_SHIFT) & alpha::FUNC_MASK);
                // The reference is a 12-bit field, x4096 with 1.0 sent as 4095
                // (ESTABLISHED). Decoding it as field / 4096 is INFERRED; the
                // vendor frames rule out field / 4095.
                let field = word & alpha::REF_MASK;
                let reference = if field == alpha::REF_ONE { 1.0 } else { field as f32 / alpha::REF_SCALE };
                args[1] = f32_bits(reference);
                self.gl_feed(op::ALPHA_FUNC as u16, &args);
            }
            reg::STENCIL_FUNC => {
                let word = self.cmd.operand[0];
                let mut args = [0u32; 3];

                args[0] = GL_NEVER_ENUM + ((word >> stencil::FUNC_SHIFT) & stencil::FUNC_MASK);
                args[1] = word & stencil::REF_MASK;
                args[2] = (word >> stencil::VALUE_MASK_SHIFT) & stencil::VALUE_MASK_MASK;
                self.gl_feed(op::STENCIL_FUNC as u16, &args);

                let fail = stencil_op((word >> stencil::OP_FAIL_SHIFT) & stencil::OP_MASK);
                let zfail = stencil_op((word >> stencil::OP_ZFAIL_SHIFT) & stencil::OP_MASK);
                let zpass = stencil_op((word >> stencil::OP_ZPASS_SHIFT) & stencil::OP_MASK);
                if fail == u32::MAX || zfail == u32::MAX || zpass == u32::MAX {
                    crate::dlog!(LogModule::Odyssey, "sgi-odyssey: stencil op word {:#x} is not modeled", word);
                } else {
                    args[0] = fail;
                    args[1] = zfail;
                    args[2] = zpass;
                    self.gl_feed(op::STENCIL_OP as u16, &args);
                }
            }
            reg::BLEND => {
                let mut args = [0u32; 4];

                // Every operand is kept whatever bit 17 says (mode::BLEND): the
                // library sends glBlendFunc's factors, and the X server its
                // opaque-stipple background, before the 0x0404 that sets the bit.
                self.const_colour[0] = self.cmd.operand[1];
                self.const_colour[1] = self.cmd.operand[2];
                self.blend_word = self.cmd.operand[0];
                let word = self.cmd.operand[0];
                let eq = (word >> blend::EQ_SHIFT) & blend::EQ_MASK;
                let src = blend_factor((word >> blend::SRC_SHIFT) & blend::SRC_MASK, false);
                let dst = blend_factor(word & blend::DST_MASK, true);
                if src == u32::MAX || dst == u32::MAX {
                    crate::dlog!(LogModule::Odyssey, "sgi-odyssey: blend factor word {:#x} is not modeled", word);
                } else {
                    args[0] = src;
                    args[1] = dst;
                    self.gl_feed(op::BLEND_FUNC as u16, &args[..2]);
                }
                if eq == 3 || eq == 4 {
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: blend equation {} (the SGIX alpha min/max extension) is not modeled",
                        eq
                    );
                } else {
                    args[0] = BLEND_EQUATION[eq as usize];
                    self.gl_feed(op::BLEND_EQUATION as u16, &args[..1]);
                }
                let c1 = self.cmd.operand[1];
                let c2 = self.cmd.operand[2];
                args[0] = f32_bits(((c1 >> blend::COLOUR_SHIFT) & blend::COLOUR_MASK) as f32 / blend::COLOUR_SCALE);
                args[1] = f32_bits((c1 & blend::COLOUR_MASK) as f32 / blend::COLOUR_SCALE);
                args[2] = f32_bits(((c2 >> blend::COLOUR_SHIFT) & blend::COLOUR_MASK) as f32 / blend::COLOUR_SCALE);
                args[3] = f32_bits((c2 & blend::COLOUR_MASK) as f32 / blend::COLOUR_SCALE);
                self.gl_feed(op::BLEND_COLOR as u16, &args);
            }
            reg::LOGIC_OP => {
                let mode = GL_CLEAR + (self.cmd.operand[0] & ODSY_LOGIC_OP_MASK);

                self.gl_feed(op::LOGIC_OP as u16, &[mode]);
            }
            reg::LINE_POINT_SIZE => {
                let mut args = [0u32; 2];

                args[0] = f32_bits(
                    (self.cmd.operand[0] & line::POINT_SIZE_MASK) as f32 / line::POINT_SIZE_SCALE,
                );
                self.gl_feed_raster_state(op::LINE_WIDTH as u16, &args[0..1]);
                args[1] = f32_bits(
                    (self.cmd.operand[1] & line::POINT_SIZE_MASK) as f32 / line::POINT_SIZE_SCALE,
                );
                self.gl_feed_raster_state(op::POINT_SIZE as u16, &args[1..2]);
            }
            reg::POLYGON_MODE => {
                let mut front = [0u32; 2];
                let mut back = [0u32; 2];

                front[0] = GL_FRONT;
                front[1] = polygon_mode(
                    (self.cmd.operand[0] >> polygon::MODE_FRONT_SHIFT) & polygon::MODE_MASK,
                );
                self.gl_feed_raster_state(op::POLYGON_MODE as u16, &front);
                back[0] = GL_BACK;
                back[1] = polygon_mode(
                    (self.cmd.operand[0] >> polygon::MODE_BACK_SHIFT) & polygon::MODE_MASK,
                );
                self.gl_feed_raster_state(op::POLYGON_MODE as u16, &back);
            }
            reg::POLYGON_OFFSET => {
                let mut args = [0u32; 2];

                args[0] = f32_bits(0.0f32);
                args[1] = f32_bits(
                    sextract32(self.cmd.operand[0], 0, polygon::OFFSET_BITS) as f32 * polygon::OFFSET_SCALE,
                );
                self.gl_feed(op::POLYGON_OFFSET_EXT as u16, &args);
            }
            reg::CLIP_BOX => {
                self.gl_clip_box[0] = self.cmd.operand[0];
                self.gl_clip_box[1] = self.cmd.operand[1];
                self.gl_apply_clip_box();
            }
            reg::FOG => {
                let n = self.gl_fog.len();
                self.gl_fog.copy_from_slice(&self.cmd.operand[..n]);
            }
            reg::TEX_MODE => {
                self.tex.mode = self.cmd.operand[0];
                self.gl_apply_clip_planes();
            }
            reg::TEX_SAMPLER => {
                let n = self.tex.sampler.len();
                self.tex.sampler.copy_from_slice(&self.cmd.operand[..n]);
            }
            reg::TEX_ENV => {
                let n = self.tex.env.len();
                self.tex.env.copy_from_slice(&self.cmd.operand[..n]);
            }
            reg::TEX_LEVEL0 => {
                self.tex.level0 = self.cmd.operand[0];
            }
            reg::TEX_LOD => {
                let n = self.tex.lod.len();
                self.tex.lod.copy_from_slice(&self.cmd.operand[..n]);
            }
            reg::TEX_SCALE_BIAS => {
                let n = self.tex.scale_bias.len();
                self.tex.scale_bias.copy_from_slice(&self.cmd.operand[..n]);
            }
            reg::TEX_OFFSET => {
                let n = self.tex.offset.len();
                self.tex.offset.copy_from_slice(&self.cmd.operand[..n]);
            }
            reg::POST_FILTER_DEFAULT => {
                self.cmd_unimplemented(tag);
            }
            reg::FIFO_SETUP_FIRST..=reg::FIFO_SETUP_LAST => {
                self.widget.fifo_setup[(tag - reg::FIFO_SETUP_FIRST) as usize] = self.cmd.operand[0];
                if tag == reg::FIFO_SETUP_6B2 {
                    self.fifo.set_threshold_6b2(self.cmd.operand[0]);
                }
            }
            reg::FIFO_SETUP_7B5 => {
                self.widget.fifo_setup[ODSY_FIFO_SETUP_7B5_INDEX] = self.cmd.operand[0];
            }
            reg::SETUP_7F0 => {
                self.widget.setup_7f0 = self.cmd.operand[0];
                if self.widget.setup_7f0 != 0 && !self.diag.setup_7f0_reported {
                    self.diag.setup_7f0_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: register 0x7f0 written with {}; the value is stored and its role is unknown, so no timer runs",
                        self.widget.setup_7f0
                    );
                }
            }
            gl_cmd::TEXCOORD_1F => {
                self.tex.coord[0] = self.cmd.operand[0];
                self.tex.coord[1] = 0;
                self.tex.coord_r = 0;
                self.tex.coord_q = ODSY_F32_ONE;
            }
            gl_cmd::TEXCOORD_2F => {
                self.tex.coord.copy_from_slice(&self.cmd.operand[..2]);
                self.tex.coord_r = 0;
                self.tex.coord_q = ODSY_F32_ONE;
            }
            gl_cmd::TEX_RECT => {
                if !self.texture_build_level() && !self.texture_zoom_rect() && !self.diag.tex_rect_reported {
                    self.diag.tex_rect_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: textured rectangle into buffer {:#x} has unsupported construction state or missing backing",
                        self.dest_base
                    );
                }
            }
            reg::GL_BEGIN => {
                // Record the primitive and, for an image, open the raster path. The
                // shared GL path opens on a float vertex, an integer GL stream,
                // a textured integer vertex or an assembly absent from the 2D
                // interpreter. Integer X11 geometry retains the 2D path.
                self.stream.primitive = self.cmd.operand[0];
                self.stream.vertex_count = 0;
                self.prim.begin_open = true;
                self.prim.integer_prim = false;
                self.prim.open_r = 0;
                self.prim.open_c = 0;
                self.prim.open_m3 = 0;
                self.prim.open_m4 = 0;
                self.prim.open_held = 0;
                // A polygon left open by a BEGIN without END encloses nothing.
                self.gl_polygon_close(false);
                self.image.active = self.cmd.operand[0] == gl_cmd::PRIMITIVE_IMAGE;
                if self.image.active {
                    self.image.x1 = 0;
                    self.image.y1 = 0;
                    self.image.x2 = 0;
                    self.image.y2 = 0;
                    self.image.cur_x = 0;
                    self.image.cur_y = 0;
                    self.image.fmt = 0;
                    self.image.group_len = 0;
                    self.image.group_painted = 0;
                    self.image.bytes = 0;
                    self.image.nbytes = 0;
                }
            }
            reg::GL_END => {
                // A GL geometry primitive closes on the shared context; the greeter's
                // QUADS fill (and the console text path's stray END) close on the 2D path.
                self.prim.begin_open = false;
                self.prim.open_held = 0;
                if self.prim.in_prim {
                    self.gl_feed(op::END as u16, &[]);
                    self.prim.in_prim = false;
                } else {
                    self.gl_end();
                }
                self.prim.integer_prim = false;
            }
            0x0400 => {
                // Pending primitive for context restore.
                self.prim.restart_type = self.cmd.operand[0];
            }
            0x0444 => {
                // Drop an outgoing primitive, without draw.
                if self.cmd.operand[0] == 0 && !self.prim.integer_prim && !(self.prim.begin_open && self.stream.primitive == 9) {
                    self.gl().discard_primitive();
                    self.prim.in_prim = false;
                    self.prim.begin_open = false;
                    self.prim.open_held = 0;
                    self.prim.restart_absorb = 0;
                    self.prim.restart_armed = false;
                } else if self.cmd.operand[0] != 0 {
                    self.cmd_unimplemented(0x10444);
                }
            }
            0x0850 => {
                let type_ = self.cmd.operand[0] & 0xf;

                if type_ == 0xf {
                    if !self.prim.integer_prim && !(self.prim.begin_open && self.stream.primitive == 9) {
                        self.gl().discard_primitive();
                        self.prim.in_prim = false;
                        self.prim.begin_open = false;
                        self.prim.open_held = 0;
                        self.prim.restart_absorb = 0;
                        self.prim.restart_armed = false;
                    }
                } else if type_ <= 8 {
                    self.stream.primitive = type_;
                    self.prim.restart_type = type_;
                    self.prim.restart_absorb = (self.cmd.operand[0] >> 19) & 3;
                    self.prim.restart_total = self.prim.restart_absorb;
                    self.prim.restart_armed = true;
                    self.prim.restart_select = 0;
                    self.prim.begin_open = true;
                    self.prim.integer_prim = false;
                    self.prim.open_r = 0;
                    self.prim.open_c = 0;
                    self.prim.open_m3 = 0;
                    self.prim.open_m4 = 0;
                    self.prim.open_held = 0;
                } else {
                    self.cmd_unimplemented(0x10850);
                }
            }
            0x0740 => {
                self.prim.restart_select = self.cmd.operand[0];
                // The kernel selects 0x1ed as well as 0x1e0. What 0x1ed selects is
                // UNKNOWN: it is accepted, and 0x0741 is still reported.
                if self.prim.restart_select != 0x1e0 && self.prim.restart_select != 0x1ed {
                    self.cmd_unimplemented(0x740);
                }
            }
            0x0741 => {
                if self.prim.restart_select == 0x1e0 && self.prim.restart_armed {
                    let word = self.cmd.operand[0];

                    self.prim.restart_c = (word >> 21) & 7;
                    self.prim.restart_m3 = (word >> 12) & 3;
                    self.prim.restart_m4 = (word >> 14) & 3;
                    self.restart_finish();
                } else {
                    self.cmd_unimplemented(0x741);
                }
            }
            cmd::IMAGE_HEADER => {
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
                let glen = (self.cmd.operand[0] >> image::HEADER_GROUP_SHIFT) & image::HEADER_GROUP_MASK;

                self.image.group_len = if glen == image::HEADER_GROUP_MASK {
                    image::HEADER_GROUP_FULL
                } else {
                    glen
                };
                self.image.group_painted = 0;
                self.image.code = self.cmd.operand[0] & image::CODE_MASK;
                self.image.write_code = self.cmd.operand[0] & write::CODE_MASK;
                self.image.nbytes = 0;
                self.image.bytes = 0;
            }
            reg::PATTERN_SHAPE => {
                self.stream.pattern_shape = self.cmd.operand[0];
                self.gl_apply_polygon_stipple();
            }
            reg::PATTERN_MODE => {
                self.stream.pattern_mode = self.cmd.operand[0];
                self.gl_apply_mode();
            }
            reg::WINDOW_ORIGIN => {
                self.win.origin[0] = sextract32(self.cmd.operand[0], 0, ODSY_ORIGIN_BITS);
                self.win.origin[1] = sextract32(self.cmd.operand[1], 0, ODSY_ORIGIN_BITS);
                if self.cmd.from_kernel {
                    self.win.prior_origin = self.win.committed_origin;
                    self.win.committed_origin = self.win.origin;
                    self.win.origin_pending = true;
                }
            }
            reg::ID_COMPARE => {
                self.win.id_compare = self.cmd.operand[1];
                if self.cmd.from_kernel && !after_tail {
                    self.win.committed_id_compare = self.win.id_compare;
                }
                // A non-zero pair still ends a pending query (which clears the
                // operands), as when the word was undecoded: whether the board does
                // is unknown, and no traced query has one inside it.
                if self.cmd.operand[0] != 0 || self.cmd.operand[1] != 0 {
                    self.query_cancel();
                }
            }
            reg::WINDOW_CLIP => {
                let n = self.win.clip_words.len();
                self.win.clip_words.copy_from_slice(&self.cmd.operand[..n]);
                if self.cmd.from_kernel
                    && self.win.origin_pending
                    && self.win.origin[0] == 0
                    && self.win.origin[1] == 0
                    && self.clip_is_offscreen()
                {
                    // The off-screen tail: keep the window pushed before it.
                    self.win.committed_origin = self.win.prior_origin;
                    self.win.after_tail = true;
                } else if self.cmd.from_kernel {
                    self.win.committed_clip = self.win.clip_words;
                }
                if self.cmd.from_kernel {
                    self.win.origin_pending = false;
                }
                self.clip_update();
            }
            cmd::WINDOW_RESTORE => {
                // The committed window back (cmd::WINDOW_RESTORE; INFERRED).
                self.win.origin = self.win.committed_origin;
                self.win.clip_words = self.win.committed_clip;
                self.win.id_compare = self.win.committed_id_compare;
                self.clip_update();
            }
            reg::LINE_STIPPLE => {
                let n = self.stream.line_stipple.len();
                self.stream.line_stipple.copy_from_slice(&self.cmd.operand[..n]);
                self.gl_apply_line_stipple();
                if self.stream.line_stipple[line::STIPPLE_CONTROL] != line::STIPPLE_CONTROL_32
                    && (self.stream.line_stipple[line::STIPPLE_CONTROL] & !0xffu32) != 0
                    && !self.diag.line_stipple_reported
                {
                    self.diag.line_stipple_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: line stipple control {:#x} has unclassified high bits",
                        self.stream.line_stipple[line::STIPPLE_CONTROL]
                    );
                }
            }
            reg::QUICK_FLUSH_ARG
            | reg::STROBE
            | reg::FULL_FLUSH_EXTRA
            | reg::PIXEL_MODE_EXT
            | reg::BLIT_PREFIX => {}
            swap::SYNC_A => {
                // The bound drawing and reading drawable words, kept raw and
                // independently (reg::DRAW_MASK): neither changes where the
                // model draws. The swap completes on the retrace hook
                // (retrace()).
                self.draw_word = self.cmd.operand[0];
                crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: buffer word: reg {:#x} word {:#x}", tag, self.draw_word);
            }
            swap::SYNC_B => {
                self.read_word = self.cmd.operand[0];
                crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: buffer word: reg {:#x} word {:#x}", tag, self.read_word);
            }
            reg::READ_ORIGIN => {
                self.win.read_origin[0] = sextract32(self.cmd.operand[0], 0, 16);
                self.win.read_origin[1] = sextract32(self.cmd.operand[0], 16, 16);
            }
            0x0000060a | 0x0000060c..=reg::DRAWABLE_LAST => {
                if !self.diag.drawable_word_reported {
                    self.diag.drawable_word_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: drawable word {:#x} with operand {:#x} is consumed but not modeled",
                        tag,
                        self.cmd.operand[0]
                    );
                }
            }
            reg::DRAW_MASK => {
                // The drawing mask: which colour buffers the next writes reach.
                self.draw_mask = self.cmd.operand[0];
                crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: draw mask: mask {:#x}", self.draw_mask);
                if (self.draw_mask & !ODSY_DRAW_MASK_BITS) != 0 && !self.diag.draw_mask_high_reported {
                    self.diag.draw_mask_high_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: drawing mask {:#x} bits above 3 are not modeled; ignored",
                        self.draw_mask
                    );
                }
                if (self.draw_mask & ODSY_DRAW_MASK_RIGHT) != 0 && !self.diag.draw_mask_stereo_reported {
                    self.diag.draw_mask_stereo_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: stereo is not modeled; drawing mask {:#x} draws its right-eye buffers into the left-eye ones",
                        self.draw_mask
                    );
                }
            }
            reg::HOST_WRITE => {
                let op0 = self.cmd.operand[0];
                let op1 = self.cmd.operand[1];
                // The host-write command's word and the halfword it writes are
                // ESTABLISHED as the kernel's accesses; that the board treats it
                // as its DMA completion signal is INFERRED. The model raises the same fence the DMA engine's
                // completion does, through the one fence implementation: op0's slot
                // and interrupt bit and op1's token become the completion cookie,
                // raised by Board::fence(). op0's interrupt
                // bit and op1's token already sit at the fence word's own positions
                // (dma::FENCE_INTERRUPT is the fence word's bit 15).
                let word: u32 = ((op1 & 0xffff) << dma::FENCE_TOKEN_SHIFT)
                    | (op0 & (dma::FENCE_INTERRUPT | dma::FENCE_SLOT_MASK));

                self.fence(ODSY_FENCE_VALID | word as u64);
            }
            dma::DESC_UPLOAD => {
                self.dma_descriptor(false);
            }
            dma::DESC_READBACK => {
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
        self.cmd.tag = tag;
        self.cmd.operand_count = operands;
        self.cmd.operand_index = 0;
        if operands == 0 {
            self.query_update(tag);
            if tag != 0x2f00c02e && tag != 0x3000c02e {
                self.cmd_dispatch(tag);
            }
        } else {
            self.cmd.state = ODSY_CMD_OPERANDS;
        }
    }

    // A register-select word (0x0001rrrr): the most common command word, since
    // every flush and state write is one. No other word classify matches has
    // this shape, so it is tested first rather than after every by-value match.
    pub fn cmd_classify_register(&mut self, word: u32) {
        let reg = word & !cmd::REGISTER_SHAPE;
        let operands = cmd_arity(reg);

        // A pixel burst while an image is open: its data words paint into the
        // box rather than streaming through the generic operand path. Outside
        // an image the same register-select is an ordinary state word.
        if let Some(fmt) = image_burst_fmt(reg).filter(|_| self.image.active && operands > 0) {
            self.image.fmt = fmt;
            if !self.image_burst_supported(fmt) {
                self.cmd_unimplemented(word);
            }
            self.cmd.state = ODSY_CMD_IMAGE_BURST;
            self.cmd.operand_count = operands;
            self.cmd.operand_index = 0;
            return;
        }

        // The 32x32-bit pattern loads: a 16-word burst each, more than
        // cmd_operand[] holds, so they stream straight into the pattern buffer
        // (0x3c4e -> words 0-15, 0x3c4f -> words 16-31) rather than through the
        // generic operand path.
        if reg == reg::PATTERN_LOW || reg == reg::PATTERN_HIGH {
            self.cmd.state = ODSY_CMD_PATTERN_LOAD;
            self.cmd.operand_count = operands;
            self.cmd.operand_index = 0;
            self.stream.pattern_load_offset = if reg == reg::PATTERN_HIGH { pattern::WORDS } else { 0 };
            return;
        }
        self.cmd_start(reg, operands);
    }

    /// Classify a word arriving while the parser is idle: a new command.
    pub fn cmd_classify(&mut self, word: u32) {
        if (word & cmd::REGISTER_SHAPE) == cmd::REGISTER_TAG {
            self.cmd_classify_register(word);
            return;
        }
        if word == 0x4c0 || word == 0x4c0 + query::KIND_RASTER {
            self.query_begin();
            self.query.kind = word - 0x4c0;
            return;
        }
        if word == ODSY_RASTER_POS_LATCH_WORD {
            self.query_raster_latch();
        }
        if word == 0x2f00c02e || word == 0x3000c02e {
            self.cmd_start(word, 0);
            return;
        }
        if word == 0x740 || word == 0x741 {
            self.cmd_start(word, 1);
            return;
        }
        if (reg::FIFO_SETUP_FIRST..=reg::FIFO_SETUP_LAST).contains(&word)
            || word == reg::FIFO_SETUP_7B5
            || word == reg::SETUP_7F0
        {
            self.cmd_start(word, 1);
            return;
        }

        if word == cmd::COPY_RECT_TRIGGER {
            self.cmd.state = ODSY_CMD_COPY_RECT;
            self.cmd.operand_index = 0;
            return;
        }
        // A page-table DMA descriptor: the opcode word, then three more words the
        // generic operand path gathers before Board::cmd_dispatch() runs the
        // transfer (the two-doubleword descriptor reaches here as four words).
        if word == dma::DESC_UPLOAD || word == dma::DESC_READBACK {
            self.cmd_start(word, dma::DESC_OPERANDS);
            return;
        }
        // The two buffer-select doublewords: each opcode word takes one operand
        // word. Consumed cleanly so the parser stays word-aligned; the swap
        // completion is delivered from the retrace hook, not here (see the
        // swap::SYNC_* constants and retrace()).
        if word == swap::SYNC_A || word == swap::SYNC_B {
            self.cmd_start(word, swap::SYNC_OPERANDS);
            return;
        }
        if word == reg::READ_ORIGIN {
            self.cmd_start(word, 1);
            return;
        }
        if (reg::DRAWABLE_FIRST..=reg::DRAWABLE_LAST).contains(&word) {
            // 0x60a, 0x60c to 0x60f; the three known ones are matched above.
            self.cmd_start(word, 1);
            return;
        }
        if word == gl_cmd::COLOR_3UB {
            self.cmd_start(word, 3);
            return;
        }
        if word == gl_cmd::VERTEX_2I {
            self.cmd_start(word, 2);
            return;
        }
        if word == gl_cmd::TEXCOORD_1F
            || word == gl_cmd::TEXCOORD_3F
            || word == gl_cmd::TEXCOORD_4F
            || word == gl_cmd::TEXCOORD_1I
            || word == gl_cmd::TEXCOORD_3I
            || word == gl_cmd::TEXCOORD_4I
            || (word & gl_cmd::TEXCOORD_SHAPE) == gl_cmd::TEXCOORD_TAG
            || word == gl_cmd::TEX_RECT
        {
            self.cmd_start(word, cmd_arity(word));
            return;
        }
        if word == cmd::WINDOW_RESTORE {
            self.cmd_start(word, 0);
            return;
        }
        if word == gl_cmd::COLOR_INDEX
            || word == gl_cmd::REPLAY_INDEX
            || word == gl_cmd::COLOR_INDEX_F
            || word == gl_cmd::REPLAY_INDEX_F
            || word == gl_cmd::INDEX_MODE_COLOR_3I
            || word == gl_cmd::INDEX_MODE_COLOR_3F
            || word == gl_cmd::INDEX_MODE_COLOR_4F
            || word == gl_cmd::COLOR_3I
            || word == gl_cmd::COLOR_4F
            || is_integer_colour(word)
            || word == gl_cmd::NORMAL_3B
            || word == gl_cmd::NORMAL_3S
            || word == gl_cmd::NORMAL_3I
        {
            self.cmd_start(word, cmd_arity(word));
            return;
        }
        // The true-colour GL words, matched by value: the float colour and vertex
        // siblings, the three matrix target-selects (0 operands each) and the
        // matrix load (16 floats, more than cmd_operand[] holds, so it streams into
        // gl_matrix[] through its own state).
        if word == gl_cmd::COLOR_3F {
            self.cmd_start(word, 3);
            return;
        }
        if word == gl_cmd::VERTEX_2F {
            self.cmd_start(word, 2);
            return;
        }
        if word == gl_cmd::VERTEX_3F || word == gl_cmd::VERTEX_3I {
            self.cmd_start(word, 3);
            return;
        }
        if word == gl_cmd::VERTEX_4F {
            self.cmd_start(word, 4);
            return;
        }
        if word == gl_cmd::LOAD_4F {
            self.cmd_start(word, gl_cmd::BLOCK16_FLOATS);
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
        if (word & gl_cmd::MATRIX_SELECT_SHAPE) == gl_cmd::MATRIX_SELECT_TAG {
            self.cmd_start(word, 0);
            return;
        }
        if word == gl_cmd::MATRIX_LOAD {
            // 16 floats, more than cmd_operand[] holds, so gathered in gl_matrix[].
            self.cmd.state = ODSY_CMD_MATRIX_LOAD;
            self.cmd.operand_count = gl_cmd::MATRIX_FLOATS;
            self.cmd.operand_index = 0;
            return;
        }
        if word == gl_cmd::RECT || word == gl_cmd::CLEAR_BOX {
            // 10 floats, more than cmd_operand[] holds: gathered in gl_rect[].
            self.cmd.state = ODSY_CMD_GL_RECT;
            self.cmd.operand_count = gl_cmd::RECT_FLOATS;
            self.cmd.operand_index = 0;
            return;
        }
        if word == mode::INDEX_ZEROS {
            self.cmd_start(word, 3);
            return;
        }
        // The BEGIN-0xa image raster path's structural words, matched by value
        // rather than as register-selects. The raster-box floats need indices 4,
        // 5, 7 and 8 -- past what cmd_operand[] holds -- so they stream through a
        // dedicated state; the state word 0x29008000 | v and its teardown twin (0
        // operands), the transfer header (1) and its tail (0) run the generic
        // path, the header setting the group length as it dispatches.
        if word == cmd::IMAGE_BOX {
            self.cmd.state = ODSY_CMD_IMAGE_BOX_LOAD;
            self.cmd.operand_count = cmd_arity(word);
            self.cmd.operand_index = 0;
            return;
        }
        if (word & cmd::STATE_WORD_MASK) == cmd::STATE_WORD
            || word == cmd::IMAGE_HEADER
            || word == cmd::IMAGE_HEADER_TAIL
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
        if cmd_arity(word) != 0 && ((word >> 24) == 0 || (word & cmd::GL_OPERANDS) != 0) {
            let note = format!("; skipping {} operands", cmd_arity(word));
            self.cmd_report(word, &note);
            self.cmd.state = ODSY_CMD_SKIP;
            self.cmd.operand_count = cmd_arity(word);
            self.cmd.operand_index = 0;
        } else if cmd_arity(word) != 0 {
            self.cmd_report(word, "; bit 23 clear, no operands");
        } else {
            self.cmd_report(word, "; no operands");
        }
    }

    /// One command-FIFO word, wherever the parser currently is.
    pub fn cmd_word(&mut self, word: u32) {
        match self.query.stage {
            1 => {
                self.query.count = word;
                self.query.stage = 2;
                return;
            }
            2 => {
                if word == 0x10a5a {
                    self.query.stage = 3;
                    return;
                }
                self.query.stage = 0;
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: unpaired current-query sequence");
                self.cmd_classify(word);
                return;
            }
            3 => {
                self.query.sync_index = word;
                self.query.stage = 4;
                return;
            }
            4 => {
                self.query_finish(word);
                return;
            }
            _ => {}
        }
        match self.cmd.state {
            ODSY_CMD_SKIP => {
                self.cmd.operand_index += 1;
                if self.cmd.operand_index == self.cmd.operand_count {
                    self.cmd.state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_OPERANDS => {
                // cmd_operand[] holds a whole burst; the bound is belt and braces.
                if (self.cmd.operand_index as usize) < self.cmd.operand.len() {
                    self.cmd.operand[self.cmd.operand_index as usize] = word;
                }
                self.cmd.operand_index += 1;
                if self.cmd.operand_index == self.cmd.operand_count {
                    let tag = self.cmd.tag;
                    self.query_update(tag);
                    self.cmd_dispatch(tag);
                    self.cmd.state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_COPY_RECT => {
                self.cmd.operand[self.cmd.operand_index as usize] = word;
                match self.cmd.operand_index {
                    cmd::COPY_RECT_SRC_INDEX => self.copy_src = word,
                    cmd::COPY_RECT_SIZE_INDEX => self.copy_size = word,
                    cmd::COPY_RECT_DST_INDEX => self.copy_dst = word,
                    _ => {}
                }
                self.cmd.operand_index += 1;
                if self.cmd.operand_index == cmd::COPY_RECT_WORDS {
                    self.copy_rect();
                    self.cmd.state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_PATTERN_LOAD => {
                // A 16-word burst into the pattern buffer at the load's own offset.
                if self.cmd.operand_index < pattern::WORDS {
                    self.stream.pattern_buffer[(self.stream.pattern_load_offset + self.cmd.operand_index) as usize] = word;
                }
                self.cmd.operand_index += 1;
                if self.cmd.operand_index == self.cmd.operand_count {
                    self.gl_apply_polygon_stipple();
                    self.cmd.state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_IMAGE_BOX_LOAD => {
                // The ten raster-box floats. Only 4, 5, 7 and 8 carry the destination
                // corners; the rest are the zero the DDX's private was allocated with.
                // Set the cursor to the top-left corner once all ten have arrived.
                match self.cmd.operand_index {
                    image::BOX_X1_INDEX => self.image.x1 = float_to_u32(word),
                    image::BOX_Y1_INDEX => self.image.y1 = float_to_u32(word),
                    image::BOX_X2_INDEX => self.image.x2 = float_to_u32(word),
                    image::BOX_Y2_INDEX => self.image.y2 = float_to_u32(word),
                    _ => {}
                }
                self.cmd.operand_index += 1;
                if self.cmd.operand_index == self.cmd.operand_count {
                    self.image.cur_x = if self.image.x2 < self.image.x1 { self.image.x1.wrapping_sub(1) } else { self.image.x1 };
                    self.image.cur_y = if self.image.y2 < self.image.y1 { self.image.y1.wrapping_sub(1) } else { self.image.y1 };
                    self.cmd.state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_IMAGE_BURST => {
                // Each data word paints into the open image at the raster cursor.
                self.image_burst_word(word);
                self.cmd.operand_index += 1;
                if self.cmd.operand_index == self.cmd.operand_count {
                    // The header's count bounds only the group's first burst; the
                    // bursts after it are full (cmd::IMAGE_HEADER).
                    self.image.group_len = 0;
                    self.cmd.state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_MATRIX_LOAD => {
                // A board-side matrix: 16 floats, loaded once all have arrived.
                if self.cmd.operand_index < gl_cmd::MATRIX_FLOATS {
                    self.gl_matrix[self.cmd.operand_index as usize] = word;
                }
                self.cmd.operand_index += 1;
                if self.cmd.operand_index == self.cmd.operand_count {
                    self.gl_load_matrix();
                    self.cmd.state = ODSY_CMD_IDLE;
                }
            }
            ODSY_CMD_GL_RECT => {
                // The rectangle's colour and corners, painted once all have arrived.
                self.gl_rect[self.cmd.operand_index as usize] = word;
                self.cmd.operand_index += 1;
                if self.cmd.operand_index == self.cmd.operand_count {
                    self.gl_rect();
                    self.cmd.state = ODSY_CMD_IDLE;
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
            _ => self.unimplemented(cmd::FIFO, size, true),
        }
    }
}

/// One step of the command-FIFO self-test preamble.
pub(super) struct CfifoPreamble {
    pub port: u64,
    pub size: u32,
    pub value: u64,
}

/// The kernel driver's command-FIFO initialisation self-test preamble, in
/// order, all through the command FIFO: three 4-byte command words, five
/// 8-byte pushes (a 32-bit selector in the upper half, data in the lower),
/// and one closing 4-byte word. Seeing the whole sequence makes
/// cfifo::SELFTEST answer OK. The values are ESTABLISHED as the
/// kernel's first stores; the role of the three command words (0x00010854,
/// 0 and 0x0fffffff) is UNKNOWN, and nothing found names them a
/// memory-protect command. The tracker only matches the sequence. (The real
/// driver sends the five 8-byte
/// words to 0x110000, not to the backend-select port, confirmed by a boot
/// trace.)
pub(super) static CFIFO_PREAMBLE: [CfifoPreamble; 9] = [
    CfifoPreamble { port: cmd::FIFO, size: 4, value: 0x00010854 },
    CfifoPreamble { port: cmd::FIFO, size: 4, value: 0x00000000 },
    CfifoPreamble { port: cmd::FIFO, size: 4, value: 0x0fffffff },
    CfifoPreamble { port: cmd::FIFO, size: 8, value: 0x000006b000000000 },
    CfifoPreamble { port: cmd::FIFO, size: 8, value: 0x000006b100000100 },
    CfifoPreamble { port: cmd::FIFO, size: 8, value: 0x000006b300005e50 },
    CfifoPreamble { port: cmd::FIFO, size: 8, value: 0x000006b200007dc0 },
    CfifoPreamble { port: cmd::FIFO, size: 8, value: 0x000007b500000001 },
    CfifoPreamble { port: cmd::FIFO, size: 4, value: 0x00010046 },
];

impl Board {
    pub fn selftest_track(&mut self, port: u64, size: u32, value: u64) {
        let next = self.widget.selftest_progress as usize;

        if next < CFIFO_PREAMBLE.len()
            && CFIFO_PREAMBLE[next].port == port
            && CFIFO_PREAMBLE[next].size == size
            && CFIFO_PREAMBLE[next].value == value
        {
            self.widget.selftest_progress = next as u32 + 1;
            if self.widget.selftest_progress as usize == CFIFO_PREAMBLE.len() {
                self.widget.cfifo_selftest_ready = true;
                self.widget.selftest_progress = 0;
            }
            return;
        }
        // A break in the sequence restarts it, counting a fresh first word.
        self.widget.selftest_progress = if CFIFO_PREAMBLE[0].port == port
            && CFIFO_PREAMBLE[0].size == size
            && CFIFO_PREAMBLE[0].value == value
        {
            1
        } else {
            0
        };
    }
}
