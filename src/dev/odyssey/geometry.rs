//! GL vertex work for the board: the vertex-program upload and its lit and
//! window records, per-vertex lighting and colour, the depth scale and the
//! combined matrix loads.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.

use super::*;

use crate::dev::gl::light::Lighting;

/// What a program asks for, as far as the model recognises it.
#[derive(Clone, Copy, Default)]
pub struct Program {
    pub known: bool,
    pub lit: bool,
    pub local_viewer: bool,
    /// The program renormalises transformed normals (GL_NORMALIZE).
    pub normalize: bool,
    pub two_sided: bool,
    pub texgen: bool,
    pub clip: bool,
    pub texture: bool,
    pub linear_texgen: u32,
}

impl Board {
    /// Walk the list the engine continues through, constants[k] up to the
    /// terminator. With strict, every word must be a routine this model names
    /// (an assembled lit list); otherwise only the texgen and clip-plane stages
    /// are picked out (a precompiled program's list, which its entries already
    /// identify, so the result is not needed). False when the terminator is not
    /// reached or, when strict, a word is not recognised.
    pub fn gl_walk_list(&mut self, k: u32, strict: bool, p: &mut Program) -> bool {
        let nc = SGI_ODYSSEY_GL_CONSTANTS;
        let at = |s: &Board, i: u32| -> u32 { s.gl_const[(k.wrapping_add(i) % nc) as usize] };
        let mut n: u32 = 0;
        while n < ODSY_PROG_LIST_MAX {
            let a = at(self, n);
            let next = at(self, n + 1);

            if a == ODSY_UCODE_END {
                return true;
            }
            if a == 0 && (next == ODSY_UCODE_CLIP || next == ODSY_UCODE_CLIP_TEXTURED || next == ODSY_UCODE_CLIP_TEXGEN) {
                p.clip = true;
                n += 2;
                continue;
            }
            // Matched public S, T, R and Q controls identify these numeric list
            // selectors as generation stages, an experimental model policy.
            // Other fragments retain their diagnostic; no whole program is matched.
            if a == 0x033 {
                let third = at(self, n + 2);
                let fourth = at(self, n + 3);

                if next == 0x04d && third == 0x057 && fourth == 0x061 && at(self, n + 4) == 0x066 && at(self, n + 5) == 0x06c && at(self, n + 6) == 0x077 {
                    p.linear_texgen = 0xf;
                    n += 7;
                    continue;
                }
                if next == 0x04d && third == 0x061 && fourth == 0x066 && at(self, n + 4) == 0x06c && at(self, n + 5) == 0x077 {
                    p.linear_texgen = 0xd;
                    n += 6;
                    continue;
                }
                if next == 0x06b && third == 0x077 {
                    p.linear_texgen = 3;
                    n += 3;
                    continue;
                }
                if (next == 0x04d || next == 0x057 || next == 0x061 || next == 0x066) && third == 0x06c && fourth == 0x077 {
                    p.linear_texgen = if next == 0x04d {
                        1
                    } else if next == 0x057 {
                        2
                    } else if next == 0x061 {
                        4
                    } else {
                        8
                    };
                    n += 4;
                    continue;
                }
            }
            if a == ODSY_UCODE_NORMALIZE {
                p.normalize = true;
            }
            if ucode_is_texgen(a) {
                p.texgen = true;
                n += 1;
                continue;
            }
            match a {
                ODSY_UCODE_FINISH_A_TWO_SIDED | ODSY_UCODE_FINISH_TWO_SIDED | ODSY_UCODE_END_TWO_SIDED => {
                    p.two_sided = true;
                }
                ODSY_UCODE_FINISH_A_TEXTURE | ODSY_UCODE_FINISH_TEXTURE | ODSY_UCODE_UNLIT_TEXTURE => {
                    p.texture = true;
                }
                ODSY_UCODE_NORMAL
                | ODSY_UCODE_NORMALIZE
                | ODSY_UCODE_LIGHT_DIR
                | ODSY_UCODE_LIGHT_POS
                | ODSY_UCODE_LIGHT_PLAIN
                | ODSY_UCODE_LIGHT_ATTEN
                | ODSY_UCODE_FINISH_A
                | ODSY_UCODE_FINISH_ONE_SIDED
                | ODSY_UCODE_END_ONE_SIDED
                | ODSY_UCODE_TAIL_ONE_SHOT
                | ODSY_UCODE_TAIL
                | ODSY_UCODE_UNLIT_FINISH => {}
                _ => {
                    if strict {
                        return false;
                    }
                }
            }
            n += 1;
        }
        false
    }

    /// One program-memory halfword (ODSY_PROG_SHAPE): halfwords 5 down to 0 of
    /// the selected entry, then the next entry, kept in the image's order.
    pub fn gl_program_halfword(&mut self, word: u32) {
        let h: u16 = ((word & ODSY_PROG_LOW_MASK) | (((word >> ODSY_PROG_HIGH_SHIFT) & ODSY_PROG_HALFWORD_HIGH_MASK) << 10)) as u16;

        if self.gl_program_addr >= SGI_ODYSSEY_GL_PROGRAM_ENTRIES {
            if !self.gl_program_mem_reported {
                self.gl_program_mem_reported = true;
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: program-memory store past entry {:#x} is not modeled", SGI_ODYSSEY_GL_PROGRAM_ENTRIES - 1);
            }
            return;
        }
        let idx = self.gl_program_addr * ODSY_PROG_ENTRY_HALFWORDS + ODSY_PROG_ENTRY_HALFWORDS - 1 - self.gl_program_half;
        self.gl_program_mem[idx as usize] = h;
        self.gl_program_half += 1;
        if self.gl_program_half == ODSY_PROG_ENTRY_HALFWORDS {
            self.gl_program_half = 0;
            self.gl_program_addr += 1;
        }
    }
}

pub fn prog_in_window(a: u32) -> bool {
    (ODSY_PROG_WINDOW_BASE..ODSY_PROG_WINDOW_BASE + ODSY_PROG_WINDOW_ENTRIES).contains(&a)
}

/// The GL library's precompiled lit programs whose three entries are all
/// resident. The library picks a precompiled program from a 24-record table
/// by a 64-bit feature key and assembles a list only when no record covers
/// the context (ESTABLISHED from the library's table and a capture). Each key
/// below has lighting on, the infinite viewer and only directional lights,
/// and no fragment lighting; a key covers every context whose features are a
/// subset of it, so the texture, texture-generation and two-sided bits it
/// also carries do not say the context uses them. The model lights every
/// program here from the light blocks as it does an assembled list
/// (Board::gl_light()).
pub static UCODE_LIT_RECORDS: [[u32; 3]; 6] = [
    [0x5cb, 0x5d8, 0x5e5], // record 3: one light
    [0x62c, 0x63d, 0x64d], // record 6: two lights, texgen
    [0x6d8, 0x6eb, 0x6fd], // record 7: one light, two-sided, texgen
    [0x5f0, 0x605, 0x619], // record 9: one light, GL_NORMALIZE
    [0x657, 0x66e, 0x684], // record 10: two lights, GL_NORMALIZE
    [0x699, 0x6b4, 0x6ce], // record 11: four lights, texgen
];

/// The lit records above that renormalise normals (GL_NORMALIZE).
pub static UCODE_NORMALIZE_RECORDS: [[u32; 3]; 2] = [[0x5f0, 0x605, 0x619], [0x657, 0x66e, 0x684]];

pub fn gl_lit_record(e: &[u32; 3]) -> bool {
    for r in UCODE_LIT_RECORDS.iter() {
        if e == r {
            return true;
        }
    }
    false
}

/// The precompiled lit programs the GL library runs from the per-context
/// program window. The library holds the geometry engine's whole shipped
/// program image, 12 bytes an entry; the table below fits only the 2,794
/// entry image of the IP30 library, SHA-256
/// a0201d88cf23752cd35ac60c25f2c1b05cfab3253e5991db09006bd1b85b0375. For a
/// record it takes the lowest address b among the record's entries and list
/// that reaches the window (a + 0x32 > 0x738), uploads min(image length - b,
/// 0xc7) image entries from b into the window at 0x738, and when b >= 0x738
/// rebases each address a >= b to a - b + 0x738. When b is lower, nothing is
/// rebased and the window gets the image from 0x738 (record 13). ESTABLISHED
/// from the library's program selection and upload. Rebased entries do not
/// identify a record -- records 17, 18 and 19 all arrive as 738 75b 77d -- so
/// a record is recognised only when its entries arrive and the window holds
/// exactly the body the library uploads for it, checked by the body's
/// CRC-64/ECMA-182 over its halfwords in image order. Any other window body
/// stays unrecognised and reported.
///
/// Lit: records whose list ends in the finish word 0x4ca or 0x50f, as the
/// resident lit records' do, and whose key has no fragment-lighting bit
/// (INFERRED from the resident records: the unlit ones end in 0x4c5, the
/// fragment-lighting ones in 0x518). Records 12, 20, 21 (fragment lighting)
/// and 22 (finish 0x4c5) are left out.
///
/// Each record also carries what its key says about the viewer and the
/// faces: key bit 0x2 is the light model's local viewer, and only a
/// local-viewer context selects records 18 and 19; key bit 0x1 is two-sided
/// lighting (INFERRED: it is set in resident record 7, the two-sided one).
#[derive(Clone, Copy, Default)]
pub struct WindowRecord {
    pub record: u8,
    pub local_viewer: bool,
    pub two_sided: bool,
    pub entry: [u16; 3],
    pub entries: u16,
    pub crc: u64,
}

pub static WINDOW_RECORDS: [WindowRecord; 8] = [
    WindowRecord { record: 13, local_viewer: false, two_sided: true, entry: [0x706, 0x720, 0x73b], entries: 199, crc: 0xde5dd7f66d4e70cf },
    WindowRecord { record: 14, local_viewer: false, two_sided: false, entry: [0x4c5, 0x4c5, 0x4c5], entries: 199, crc: 0x5aa6880e8d77c0b2 },
    WindowRecord { record: 15, local_viewer: false, two_sided: true, entry: [0x4c5, 0x4c5, 0x4c5], entries: 199, crc: 0x9391fc6eed0accdc },
    WindowRecord { record: 16, local_viewer: false, two_sided: false, entry: [0x738, 0x75a, 0x77b], entries: 199, crc: 0xb1da870f5523467f },
    WindowRecord { record: 17, local_viewer: false, two_sided: false, entry: [0x738, 0x75b, 0x77d], entries: 199, crc: 0xc4bf2cd3423da137 },
    WindowRecord { record: 18, local_viewer: true, two_sided: false, entry: [0x738, 0x75b, 0x77d], entries: 199, crc: 0xe15a6186144f9887 },
    WindowRecord { record: 19, local_viewer: true, two_sided: false, entry: [0x738, 0x75b, 0x77d], entries: 199, crc: 0xa419c456a8238b77 },
    WindowRecord { record: 23, local_viewer: false, two_sided: false, entry: [0x738, 0x760, 0x787], entries: 199, crc: 0x8d6174822a99856f },
];


impl Board {
    /// CRC-64/ECMA-182 (no reflection, zero initial value) of n window entries.
    pub fn window_crc(&self, n: u32) -> u64 {
        let base = (ODSY_PROG_WINDOW_BASE * ODSY_PROG_ENTRY_HALFWORDS) as usize;
        let h = &self.gl_program_mem[base..];
        let mut crc: u64 = 0;

        for &half in &h[..(n * ODSY_PROG_ENTRY_HALFWORDS) as usize] {
            crc ^= (half as u64) << 48;
            for _bit in 0..16 {
                crc = if crc & (1u64 << 63) != 0 { (crc << 1) ^ ODSY_CRC64_ECMA_POLY } else { crc << 1 };
            }
        }
        crc
    }

    /// The lit window record the entries and window body identify, or None.
    pub fn gl_window_record(&self, e: &[u32; 3]) -> Option<&'static WindowRecord> {
        WINDOW_RECORDS.iter().find(|w| {
            e[0] == w.entry[0] as u32
                && e[1] == w.entry[1] as u32
                && e[2] == w.entry[2] as u32
                && self.window_crc(w.entries as u32) == w.crc
        })
    }
}

/// Uploaded program bodies have no reviewed decoder.
pub fn gl_resident(e: &[u32; 3]) -> bool {
    !e.iter().any(|&w| prog_in_window(w))
}

impl Board {
    /// Recognise the program the library just set (on every 0x2c write, the last
    /// word of each program set). APPROXIMATION, see
    /// ODSY_UCODE_END: a precompiled program is recognised by its three entries,
    /// an assembled one (0x2a and 0x2c both the terminator) by the routines in
    /// its list, the head (0x29) giving the viewer and saying lighting is on, or,
    /// for the unlit head 0x4ae, that it is off.
    /// Which lights are enabled and of what kind is not taken from the program:
    /// the library zeroes a disabled light's colours, so lighting all eight
    /// blocks gives the same colour, and how the engine maps a sparse set of
    /// enabled lights onto the list is unknown. All-zero entries are the state
    /// the GL library's context default-state setup pushes before any program
    /// exists.
    pub fn gl_program(&mut self) {
        let r: [u32; 3] = [self.gl_entry[ODSY_ENTRY_29], self.gl_entry[ODSY_ENTRY_2A], self.gl_entry[ODSY_ENTRY_2C]];
        let (e29, e2a, e2c) = (r[0], r[1], r[2]);
        let k = self.gl_entry[ODSY_ENTRY_33];
        let mut p = Program::default();
        let resident = gl_resident(&r);
        let window = self.gl_window_record(&r);

        if let Some(window) = window {
            // A precompiled lit record the library ran from the window.
            self.gl_walk_list(k, false, &mut p);
            p.known = true;
            p.lit = true;
            p.local_viewer = window.local_viewer;
            p.two_sided |= window.two_sided;
        } else if !resident {
            // Any other uploaded program is unsupported and reported below.
        } else if r[0] == 0 && r[1] == 0 && r[2] == 0 {
            p.known = true;
        } else if r == [ODSY_UCODE_UNLIT, ODSY_UCODE_UNLIT_2A, ODSY_UCODE_UNLIT_2C]
            || r == [ODSY_UCODE_UNLIT_CLIP, ODSY_UCODE_UNLIT_CLIP_2A, ODSY_UCODE_UNLIT_CLIP_2C]
        {
            self.gl_walk_list(k, false, &mut p);
            p.known = true;
        } else if gl_lit_record(&r) {
            self.gl_walk_list(k, false, &mut p);
            p.known = true;
            p.lit = true;
            p.normalize |= UCODE_NORMALIZE_RECORDS.contains(&r);
        } else if r[1] == ODSY_UCODE_END && r[2] == ODSY_UCODE_END {
            match r[0] {
                ODSY_UCODE_HEAD_LOCAL_DIR | ODSY_UCODE_HEAD_LOCAL_POS => {
                    p.local_viewer = true;
                    p.known = self.gl_walk_list(k, true, &mut p);
                    p.lit = true;
                }
                ODSY_UCODE_HEAD_INF_DIR | ODSY_UCODE_HEAD_INF_POS => {
                    p.known = self.gl_walk_list(k, true, &mut p);
                    p.lit = true;
                }
                ODSY_UCODE_HEAD_UNLIT => {
                    // Lighting off: unlit is the correct rendering, not a guess.
                    p.known = self.gl_walk_list(k, true, &mut p);
                }
                _ => {}
            }
        }

        if !p.known {
            p.lit = false;
            let mut i = 0u32;
            while i < self.gl_program_logged_count {
                let b = (i * 3) as usize;
                if self.gl_program_logged[b] == e29 && self.gl_program_logged[b + 1] == e2a && self.gl_program_logged[b + 2] == e2c {
                    break;
                }
                i += 1;
            }
            if i == self.gl_program_logged_count && i < SGI_ODYSSEY_GL_PROGRAM_REPORTS {
                let m = SGI_ODYSSEY_GL_CONSTANTS - 1;
                let c = |j: u32| -> u32 { self.gl_const[(k.wrapping_add(j) & m) as usize] };
                let (c0, c1, c2, c3, c4, c5, c6, c7) = (c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7));

                let b = (i * 3) as usize;
                self.gl_program_logged[b] = e29;
                self.gl_program_logged[b + 1] = e2a;
                self.gl_program_logged[b + 2] = e2c;
                self.gl_program_logged_count += 1;
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: geometry program {:#x} {:#x} {:#x} from list index {} ({:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}) is not recognised{}; drawn unlit",
                    e29, e2a, e2c, k, c0, c1, c2, c3, c4, c5, c6, c7,
                    if resident { "" } else { " (uploaded program is unsupported)" }
                );
            }
        }
        if p.two_sided && !self.gl_two_sided_logged {
            self.gl_two_sided_logged = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: two-sided lighting is not modeled; back faces are lit with the front material");
        }
        if p.texgen && !self.gl_texgen_logged {
            self.gl_texgen_logged = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: texture-coordinate generation in geometry program {:#x} is not modeled", e29);
        }
        if p.clip && !self.gl_clip_logged {
            self.gl_clip_logged = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: clip planes in geometry program {:#x} are not modeled; geometry is not clipped to them", e29);
        }
        if p.texture && !self.gl_texture_stage_logged {
            self.gl_texture_stage_logged = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: texture-mode lighting finish in geometry program {:#x} is not modeled", e29);
        }
        self.gl_linear_texgen = if p.known { p.linear_texgen } else { 0 };
        self.gl_lit = p.lit;
        self.gl_local_viewer = p.local_viewer;
        self.gl_normalize = p.normalize;
    }

    /// A colour-mode word (ODSY_PROG_COLOUR_MODE). Only the captured pairs are
    /// known: 0x22 = 0 or 1, 0x23 = 2 or 3 (0 from the kernel before a context
    /// has drawn). Anything else, or a pair whose index bits disagree once
    /// 0x23, which follows 0x22 in every capture, has arrived, is kept and
    /// reported once.
    pub fn colour_mode(&mut self, cls: u32, v: u32) {
        let known: u32 = if cls == ODSY_PROG_COLOUR_MODE { 0x1 } else { 0x3 };
        let index = self.gl_index_mode();

        self.colour_mode[(cls - ODSY_PROG_COLOUR_MODE) as usize] = v;
        if self.gl_index_mode() != index && !self.gl_in_prim {
            self.gl_apply_blend();
        }
        let disagree = cls == ODSY_PROG_COLOUR_MODE_B && ((self.colour_mode[0] ^ self.colour_mode[1]) & ODSY_COLOUR_MODE_INDEX) != 0;
        if ((v & !known) != 0 || disagree) && !self.colour_mode_reported {
            self.colour_mode_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: colour mode word class {:#x} value {:#x} (0x22 value {:#x}) is not modeled", cls, v, self.colour_mode[0]);
        }
    }

    /// One program word (is_program_word()).
    pub fn gl_program_word(&mut self, word: u32) {
        let v = prog_value(word);
        let cls = word >> ODSY_PROG_CLASS_SHIFT;

        if (ODSY_PROG_SEQ_FIRST..=ODSY_PROG_SEQ_LAST).contains(&cls) {
            self.gl_seq[(cls - ODSY_PROG_SEQ_FIRST) as usize] = v;
        }
        match cls {
            ODSY_PROG_COLOUR_MODE | ODSY_PROG_COLOUR_MODE_B => {
                self.colour_mode(word >> ODSY_PROG_CLASS_SHIFT, v);
            }
            ODSY_PROG_CONST_SELECT => {
                self.gl_const_index = v;
            }
            ODSY_PROG_CONST_STORE => {
                self.gl_const[(self.gl_const_index % SGI_ODYSSEY_GL_CONSTANTS) as usize] = v;
                self.gl_const_index = (self.gl_const_index + 1) % SGI_ODYSSEY_GL_CONSTANTS;
            }
            ODSY_PROG_ENTRY_29 => {
                // Alone, 0x29 selects the stream for light loads (0x19a) or pixel
                // transfers (0x12f) and runs the one-shot: not a program change. It
                // is kept for the next 0x2c.
                self.gl_entry[ODSY_ENTRY_29] = v;
            }
            ODSY_PROG_ENTRY_2A => {
                self.gl_entry[ODSY_ENTRY_2A] = v;
            }
            ODSY_PROG_ENTRY_2C => {
                self.gl_entry[ODSY_ENTRY_2C] = v;
                self.gl_program();
            }
            ODSY_PROG_LIST_INDEX => {
                self.gl_entry[ODSY_ENTRY_33] = v;
            }
            ODSY_PROG_COLOUR_MATERIAL => {
                self.gl_colour_material = v & ODSY_COLOUR_MATERIAL_MASK;
                if (v & !ODSY_COLOUR_MATERIAL_MASK) != 0 {
                    crate::dlog!(LogModule::Odyssey, "sgi-odyssey: colour-material word value {:#x} has bits beyond the eight known ones", v);
                }
            }
            ODSY_PROG_LIGHT_VEC_SELECT => {
                self.gl_light_vec_sel = v;
            }
            ODSY_PROG_LIGHT_COLOUR_SELECT => {
                self.gl_light_colour_sel = v;
            }
            ODSY_PROG_PROGRAM_ADDR => {
                self.gl_program_addr = v;
                self.gl_program_half = 0;
            }
            _ => {
                if (word >> ODSY_PROG_CLASS_SHIFT) >= ODSY_PROG_PROGRAM_HALFWORD {
                    self.gl_program_halfword(word);
                }
                // The other sequencer classes: kept in gl_seq[] only.
            }
        }
    }
}

pub fn odyssey_f32(bits: u32) -> f64 {
    f32::from_bits(bits) as f64
}

pub fn f32_bits(f: f32) -> u32 {
    f.to_bits()
}

// MAX/MIN as the draw maths needs them: the first operand wins ties and NaN comparisons.
fn cmax(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

fn cmin(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

impl Board {
    /// OpenGL's lighting equation (section 2.13.1) at one vertex, front
    /// material, from the kept light, material and matrix words, evaluated by
    /// dev::gl's lighting core. Under colour material (section 2.13.3) the
    /// current colour replaces each front material colour the class 0x35 word
    /// names (ODSY_PROG_COLOUR_MATERIAL); the back half of that word is kept
    /// but, like the back material, unused.
    ///
    /// APPROXIMATION: the board's geometry engine computes this in microcode
    /// the model does not run, with a table function for the specular term
    /// whose exact falloff is unknown; GL's equation reproduces both lighting
    /// captures. The eye position is the model-view (slot 0x24) applied to the
    /// vertex, the normal the upper 3x3 of slot 0x34 applied to the current
    /// normal, renormalised only when the program is one of the GL_NORMALIZE
    /// ones. The library precomputes a half vector (floats 3-5) for each
    /// infinite light and leaves it zero for a positional one, so a light
    /// block is directional when its half vector is not zero (INFERRED; a
    /// directional light straight down -z would have a zero half vector and
    /// is then read as positional). A light with zero ambient, diffuse and
    /// specular colour, every disabled one as the library sends it, adds
    /// nothing and is left out. Shininess arrives as its square root.
    pub fn gl_light(&mut self, obj: &[f64; 4], out: &mut [f32; 4]) {
        let f = |w: u32| f32::from_bits(w);
        let obj = [obj[0] as f32, obj[1] as f32, obj[2] as f32, obj[3] as f32];
        // The model-view's first three rows; the board ignores w.
        let mv = &self.gl_modelview;
        let mut eye = [0f32; 4];
        for r in 0..3usize {
            eye[r] = (0..4usize).map(|k| f(mv[r * 4 + k]) * obj[k]).sum();
        }
        eye[3] = 1.0;

        let mut lt = Lighting::default();
        lt.init();
        lt.on = 1;
        lt.local_viewer = self.gl_local_viewer as u32;
        lt.normalize = self.gl_normalize as u32;
        for r in 0..3usize {
            for c in 0..3usize {
                lt.normal_matrix[c * 3 + r] = f(self.gl_normal_matrix[r * 4 + c]);
            }
        }
        let colour = self.gl_colour.map(f);
        let cm = self.gl_colour_material;
        let pick = |bit: u32, base: usize| -> [f32; 4] {
            if cm & bit != 0 { colour } else { std::array::from_fn(|k| f(self.gl_material[base + k])) }
        };
        let (ma, md, ms, me) = (
            pick(ODSY_COLOUR_MATERIAL_AMBIENT, 0),
            pick(ODSY_COLOUR_MATERIAL_DIFFUSE, 4),
            pick(ODSY_COLOUR_MATERIAL_SPECULAR, 8),
            pick(ODSY_COLOUR_MATERIAL_EMISSION, 12),
        );
        let m = &mut lt.mat[0];
        m.ambient = [ma[0], ma[1], ma[2]];
        m.diffuse = md;
        m.specular = [ms[0], ms[1], ms[2]];
        m.emission = [me[0], me[1], me[2]];
        let shininess = f(self.gl_shininess_sqrt);
        m.shininess = shininess * shininess;
        let ambient = ODSY_GL_LIGHT_MODEL_AMBIENT as usize * 4;
        lt.amb_sum = std::array::from_fn(|k| f(self.gl_light_colour[ambient + k]));

        let mut last = -1i32;
        for i in 0..SGI_ODYSSEY_GL_LIGHTS as usize {
            let blk = &self.gl_light_vec[i * ODSY_GL_LIGHT_VEC_FLOATS as usize..];
            let col = &self.gl_light_colour[i * 3 * 4..];
            let rgb = |base: usize| -> [f32; 3] { std::array::from_fn(|k| f(col[base + k])) };
            let (la, ld, ls) = (rgb(0), rgb(4), rgb(8));
            if la.iter().chain(&ld).chain(&ls).all(|&v| v == 0.0) {
                continue;
            }
            let l = &mut lt.lights[i];
            l.ambient = la;
            l.diffuse = ld;
            l.specular = ls;
            let pos: [f32; 3] = std::array::from_fn(|k| f(blk[ODSY_LIGHT_POS + k]));
            let directional = (0..3).any(|k| f(blk[ODSY_LIGHT_HALF + k]) != 0.0);
            if directional {
                l.pos = [pos[0], pos[1], pos[2], 0.0];
            } else {
                l.pos = [pos[0], pos[1], pos[2], 1.0];
                // The board's order is constant, linear, quadratic.
                l.atten = [f(blk[ODSY_LIGHT_ATTEN + 1]), f(blk[ODSY_LIGHT_ATTEN + 2]), f(blk[ODSY_LIGHT_ATTEN])];
                // The board keeps minus the cutoff cosine, minus the spot
                // direction and minus the exponent.
                let cut = -f(blk[ODSY_LIGHT_SPOT_COS]);
                if cut > -1.0 {
                    l.spot_on = 1;
                    l.spot_cos_cutoff = cut;
                    l.spot_dir = std::array::from_fn(|k| -f(blk[ODSY_LIGHT_SPOT_DIR + k]));
                    l.spot_exp = -f(blk[ODSY_LIGHT_SPOT_EXP]);
                }
            }
            if last < 0 {
                lt.head = i as i32;
            } else {
                lt.lights[last as usize].next = i as i32;
            }
            last = i as i32;
            lt.count += 1;
        }
        let normal = self.gl_normal.map(f);
        let (front, _) = lt.light_vertex(eye, normal, colour);
        *out = front;
        // The board passes the diffuse alpha through unclamped.
        out[3] = md[3];
    }

    /// The colour a GL vertex takes, handed to the shared context just before
    /// the vertex: in index mode the current index (glIndex, not glColor); under
    /// a lit program the lit colour; otherwise the current glColor, restored if
    /// an index or lit vertex replaced it. xyz is the vertex's object position,
    /// already converted from whatever form its word carried, z 0 for a 2D one;
    /// lighting needs it for positional lights and the local viewer.
    pub fn gl_vertex_colour(&mut self, xyz: &[f64; 3]) {
        if self.gl_dither && !self.gl_dither_reported {
            self.gl_dither_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: dither not modelled; drawing undithered");
        }
        if self.gl_index_mode() {
            let mut rgba = [0f32; 4];

            if self.gl_lit && !self.gl_index_lit_reported {
                self.gl_index_lit_reported = true;
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: colour-index lighting is not modeled; drawn with the current index");
            }
            gl_index_rgba(self.gl_color_index, &mut rgba);
            let words = rgba.map(f32::to_bits);
            self.gl_feed(SGI_GL_OP_COLOR4F as u16, &words);
            self.gl_colour_overridden = true;
        } else if self.gl_lit {
            let obj = [xyz[0], xyz[1], xyz[2], 1.0];
            let mut rgba = [0f32; 4];

            self.gl_light(&obj, &mut rgba);
            let words = rgba.map(f32::to_bits);
            self.gl_feed(SGI_GL_OP_COLOR4F as u16, &words);
            self.gl_colour_overridden = true;
        } else if self.gl_colour_overridden {
            let words = self.gl_colour;
            self.gl_feed(SGI_GL_OP_COLOR4F as u16, &words);
            self.gl_colour_overridden = false;
        }
    }

    /// The combined matrix (slot 0x00): the client's model-view and projection
    /// composed with its window map, so it takes object coordinates straight to
    /// window pixels with row 0 at the window's top. That it carries the
    /// model-view is ESTABLISHED from a capture: its fourth row is minus the
    /// third row of the model-view it loads into slot 0x24, as a frustum times
    /// that model-view gives. So the model-view (slot 0x24) is kept only for
    /// lighting's eye position (Board::gl_light()), and the shared context's own
    /// model-view stays the identity. For a 128x128 window under an identity
    /// model-view and projection it is x' = 64x + 64, y' = -64y + 63.996, and
    /// glOrtho(-2, 2, -2, 2, -1, 1) halves the x and y scales. The two cannot be
    /// pulled apart, so the model applies the product whole, as the board must.
    ///
    /// The shared context's pipeline is projection, divide, then viewport map. It
    /// is given a fixed viewport of +-ODSY_GL_WINDOW_REACH about the window
    /// origin, xw = R x_ndc and yw = R y_ndc, and as its projection the combined
    /// matrix with the x and y rows divided by R, so its output is the combined
    /// matrix's window position unchanged, the library's 1/256 row bias included.
    /// The rows run top-down like the window origin, rectangles and id plane, so
    /// no flip is needed. Clipping is then the window's own rectangles; where the
    /// board clips geometry that leaves the window is unknown.
    ///
    /// Depth: the library builds the depth row and the four block-16 floats from
    /// the same rows of its composite transform (ESTABLISHED from the library's
    /// combined-matrix validation). Block 16 is the eye-space w row times d, the
    /// determinant of the projection's lower-right 2x2; the depth row is the
    /// eye-space z and w rows, mapped so that the near and far eye depths give 0
    /// and 2S, times the same d (S = ODSY_GL_DEPTH_S). Their ratio
    /// (row2 . v) / (block16 . v) is then the eye-linear depth in board units for
    /// any projection, and under an affine model-view block 16 is (0, 0, 0, d).
    /// That the board divides by block 16 and not by w is INFERRED: it is the one
    /// reading under which the identity, glOrtho and glFrustum captures, the
    /// orthographic controls and a perspective scene's rows all span the depth
    /// range. That it divides by the whole row, and not by its fourth float
    /// alone, is INFERRED the same way: under an affine model-view the two
    /// agree, and under a model-view that carries the perspective (gluPerspective
    /// in model-view mode) block 16 equals the combined matrix's fourth row.
    /// Dividing by the fourth float alone puts such a mesh's rows behind the
    /// front one, all clipped, where the whole row puts every row just inside
    /// the far plane. Malformed zero or non-finite words keep the sign-only
    /// fallback, which divides by a constant.
    ///
    /// The depth reaches the shared rasterizer as its rules' depth row,
    /// row2 / (2S block16[3]), and depth divisor, block16 / block16[3], each
    /// applied to the object coordinates and interpolated with perspective
    /// correction (Board::gl_operation_rules()). Under perspective the shared
    /// projection's z row is 0, so no depth is carried as a difference from w,
    /// which a float row could not resolve; a fragment outside depth [0, 1] by
    /// more than the rounding slack (ODSY_GL_DEPTH_SLACK) is dropped, which is
    /// the near and far clip. Under an orthographic matrix the z row stays
    /// row2 / (block16[3] S) - row3, whose window depth is the same. The plane
    /// holds the board's 24-bit float depth word of 2S times that depth
    /// (depth_word.rs; datasheet, "Depth word"). The word's layout is
    /// ESTABLISHED as the library's reading; that the plane stores exactly it is
    /// INFERRED, and the rounding of a rasterised depth into it (truncation here)
    /// is UNKNOWN. Whether the board interpolates depth with perspective
    /// correction is INFERRED from SGI's raster patents; the model does.
    ///
    /// Polygon offset (block 0x058) is a signed count of steps of the encoded
    /// word, added to it after encoding (RasterRules::split_offset()). The
    /// field is ESTABLISHED; adding it to the word, not to eye-linear depth, is
    /// INFERRED, so the offset shrinks toward the near plane in proportion to
    /// d - n. The far-plane kill is by depth value in the raster stage
    /// (INFERRED), and this model cuts at the slack ODSY_GL_DEPTH_SLACK
    /// (APPROXIMATION: the board's cut point is UNKNOWN).
    ///
    /// APPROXIMATION, raster position: under perspective the shared projection's
    /// z row is 0, so glRasterPos takes window z 0.5 whatever its eye distance,
    /// and a bitmap or pixel rectangle placed there is depth-tested at 0.5.
    /// Where the board puts a perspective raster position's depth is UNKNOWN.
    ///
    /// A non-default glDepthRange changes the row in a way this rule does not
    /// fit: it is drawn with the default [0, 1] range, and the near and far clip
    /// planes the library adds for it are reported as a program stage
    /// (Board::gl_program()).
    ///
    /// A matrix whose x or y row is zero -- the library's load before its window
    /// exists, or the kernel's zero restore at a context switch -- leaves an empty
    /// viewport, so nothing is drawn.
    pub fn gl_depth_scale(&mut self) -> f64 {
        let sign_word = f32::from_bits(self.gl_block16[ODSY_GL_BLOCK16_SIGN as usize]);

        if sign_word.is_finite() && sign_word != 0.0 {
            return sign_word as f64;
        }
        if sign_word < 0.0 { -1.0 } else { 1.0 }
    }

    pub fn gl_load_combined(&mut self, m: &[f32; 16]) {
        let perspective = m[3] != 0.0 || m[7] != 0.0 || m[11] != 0.0;
        let depth_scale = self.gl_depth_scale();
        let mut p = [0f32; 16];

        self.gl().cur.viewport = [0; 4];
        if (m[0] == 0.0 && m[4] == 0.0 && m[8] == 0.0) || (m[1] == 0.0 && m[5] == 0.0 && m[9] == 0.0) {
            return;
        }
        for col in 0..4usize {
            p[col * 4] = m[col * 4] / ODSY_GL_WINDOW_REACH as f32;
            p[col * 4 + 1] = m[col * 4 + 1] / ODSY_GL_WINDOW_REACH as f32;
            p[col * 4 + 2] = if perspective {
                0.0
            } else {
                (m[col * 4 + 2] as f64 / (depth_scale * ODSY_GL_DEPTH_S) - m[col * 4 + 3] as f64) as f32
            };
            p[col * 4 + 3] = m[col * 4 + 3];
        }
        self.gl().set_matrix(SGI_GL_PROJECTION, &p);
        let vp = &mut self.gl().cur.viewport;
        vp[0] = -ODSY_GL_WINDOW_REACH;
        vp[1] = -ODSY_GL_WINDOW_REACH;
        vp[2] = 2 * ODSY_GL_WINDOW_REACH;
        vp[3] = 2 * ODSY_GL_WINDOW_REACH;
    }
}

/// A board-side matrix load: 16 IEEE-754 floats, ROW-major on the wire.
/// GlContext::set_matrix() takes column-major, so transpose before the call.
/// gl_matrix_slot is the slot the last matrix-select chose; the normal slot
/// (ODSY_GL_SLOT_NORMAL) is one the context refuses, and its false return is
/// expected. The texture matrix is kept as sent and applied to the current
/// ST by the shared context.
pub fn gl_transpose(rows: &[u32], m: &mut [f32; 16]) {
    for row in 0..4usize {
        for col in 0..4usize {
            m[col * 4 + row] = f32::from_bits(rows[row * 4 + col]);
        }
    }
}

impl Board {
    /// Apply the kept slot-0x00 matrix under the kept block-16 sign.
    pub fn gl_reload_combined(&mut self) {
        let mut m = [0f32; 16];

        gl_transpose(&self.gl_combined, &mut m);
        self.gl_load_combined(&m);
    }

    pub fn gl_load_matrix(&mut self) {
        let mut m = [0f32; 16];

        if self.gl_matrix_slot == ODSY_GL_SLOT_TEXTURE {
            self.gl_texture_matrix = self.gl_matrix;
            return;
        }
        if self.gl_matrix_slot == ODSY_GL_SLOT_NORMAL_MATRIX {
            self.gl_normal_matrix = self.gl_matrix;
            return;
        }
        if self.gl_matrix_slot == ODSY_GL_SLOT_COMBINED {
            self.gl_combined = self.gl_matrix;
            self.gl_reload_combined();
            return;
        }
        if self.gl_matrix_slot == SGI_GL_MODELVIEW {
            // The combined matrix already carries the model-view
            // (Board::gl_load_combined()), so the shared context keeps its
            // identity model-view; this copy serves lighting's eye position.
            self.gl_modelview = self.gl_matrix;
            for i in 0..SGI_GL_CLIP_PLANES {
                self.gl_apply_clip_plane(i);
            }
            return;
        }
        gl_transpose(&self.gl_matrix, &mut m);
        let slot = self.gl_matrix_slot;
        self.gl().set_matrix(slot, &m);
    }

    /// The four floats behind index 16, gathered in cmd_operand[]. Their sign
    /// word orients the combined matrix's depth row, which arrived just before
    /// them, so the projection is rebuilt. A four-float load after any other
    /// select has no known meaning and is reported.
    pub fn gl_load_block16(&mut self) {
        if self.gl_matrix_slot != ODSY_GL_SLOT_BLOCK16 {
            self.cmd_unimplemented(ODSY_GL_LOAD_4F);
            return;
        }
        let n = self.gl_block16.len();
        self.gl_block16.copy_from_slice(&self.cmd_operand[..n]);
        self.gl_reload_combined();
    }

    /// The screen-aligned rectangle (ODSY_GL_RECT): its corners are window pixels,
    /// so add the window origin, then fill the pixels whose centres fall inside the
    /// half-open box between the two corners -- sx0 <= p + 0.5 < sx1 -- each
    /// through the pixel test. It goes through the shared rasterizer's clear so
    /// the float colour is quantized the same way a triangle's is; the clear's
    /// rectangle is the box within the plane and its clip list the kept 0x204b
    /// rectangles.
    ///
    /// glClear of the depth buffer is the same rectangle: the library sets the
    /// 0x0452 depth write bit and puts the clear depth, d x 2S, in both corners'
    /// z, and for a depth-only clear also zeroes the colour write mask, leaving
    /// the colour operands uninitialised. So the rectangle writes depth, from the
    /// first corner's z scaled to the 24-bit plane, when the depth write bit is
    /// set, and writes colour only when the colour mask is not zero (ESTABLISHED
    /// for the words). The library forces GL_ALWAYS around every rectangle it
    /// sends; how the board would test a rectangle under another comparison is not
    /// known, so that case is reported and the rectangle written unconditionally.
    /// In index mode the first float is the clear index, not red (glClearIndex(3)
    /// sends 3.0 there, ESTABLISHED from a colour-index capture), and the
    /// rectangle stores it like any index fragment. glClear of the stencil
    /// planes is the rectangle with the stencil test on under GL_ALWAYS and
    /// GL_REPLACE (ODSY_MODE_STENCIL_TEST), so it writes the reference under the
    /// stencil write mask; another stencil comparison or operation around a
    /// rectangle is reported and leaves the stencil unchanged.
    pub fn gl_rect(&mut self) {
        let mut f = [0f32; ODSY_GL_RECT_FLOATS as usize];
        let mut anchor_x: u32 = 0;
        let mut anchor_y: u32 = 0;
        let patterned = (self.pattern_mode & ODSY_GL_RECT_PATTERN_ENABLE) != 0;
        let rules = self.gl_operation_rules();
        let mut op = RasterOp {
            kind: SGI_RASTER_CLEAR,
            clear_colour_mask: u32::MAX,
            ..Default::default()
        };

        for i in 0..f.len() {
            f[i] = f32::from_bits(self.gl_rect[i]);
        }
        if patterned {
            if self.pattern_shape != ODSY_PATTERN_SHAPE_32X32
                || !f[4].is_finite()
                || !f[5].is_finite()
                || !f[7].is_finite()
                || !f[8].is_finite()
                || f[4].trunc() != f[4]
                || f[5].trunc() != f[5]
                || f[7].trunc() != f[7]
                || f[8].trunc() != f[8]
            {
                if !self.gl_rect_pattern_logged {
                    self.gl_rect_pattern_logged = true;
                    crate::dlog!(LogModule::Odyssey, "sgi-odyssey: patterned rectangle requires shape 4 and finite integral corners");
                }
                return;
            }
            // Reduce before conversion: integral floats can exceed int range.
            anchor_x = ((self.win_origin[0] % 32).wrapping_add(((f[4] as f64) % 32.0) as i32) & 31) as u32;
            anchor_y = ((self.win_origin[1] % 32).wrapping_add(((f[5] as f64) % 32.0) as i32) & 31) as u32;
        }
        if ((self.gl_write_mask >> ODSY_WRITE_COLOUR_SHIFT) & ODSY_WRITE_COLOUR_MASK) != 0 {
            op.clear_buffers |= SGI_GL_COLOR_BUFFER_BIT;
        }
        if (self.gl_write_mask & ODSY_WRITE_DEPTH) != 0 {
            let mut d = f[6] as f64 / (2.0 * ODSY_GL_DEPTH_S);
            d = cmin(cmax(d, 0.0), 1.0);
            op.clear_buffers |= SGI_GL_DEPTH_BUFFER_BIT;
            op.clear_depth24 = rules.depth_code(d);
        }
        if (self.pattern_mode & ODSY_MODE_STENCIL_TEST) != 0 && self.gl_ref().cur.stencil_write_mask != 0 {
            let st = &self.gl_ref().cur;

            if st.stencil_func == SGI_GL_ALWAYS && st.stencil_op[2] == SGI_GL_REPLACE {
                // glClear's form (ODSY_MODE_STENCIL_TEST): write the reference.
                op.clear_buffers |= SGI_GL_STENCIL_BUFFER_BIT;
                op.clear_stencil = st.stencil_ref as u32;
                op.stencil_mask = st.stencil_write_mask;
            } else if !self.gl_rect_stencil_logged {
                let (func, sop) = (st.stencil_func, st.stencil_op[2]);
                self.gl_rect_stencil_logged = true;
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: stencil-tested rectangle under comparison {} and operation {:#x} is not modeled; stencil left unchanged", func, sop);
            }
        }
        if (self.pattern_mode & ODSY_MODE_DEPTH_TEST) != 0 && (self.gl_depth_func & ODSY_DEPTH_FUNC_MASK) != SGI_GL_ALWAYS && !self.gl_rect_depth_logged {
            self.gl_rect_depth_logged = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: depth-tested rectangle under comparison {} is not modeled; written unconditionally", self.gl_depth_func & ODSY_DEPTH_FUNC_MASK);
        }
        // No kept rectangle means every 0x204b rectangle is empty: the window is
        // fully covered and nothing may be drawn. The shared clear reads a clip
        // list with n = 0 as "no clip", so the rectangle is skipped here. (Reset
        // is not this case: rectangle 0 then spans the whole range.)
        if self.clip_kept == 0 || op.clear_buffers == 0 {
            return;
        }
        let mut x0 = self.win_origin[0] as f64 + f[4] as f64;
        let mut y0 = self.win_origin[1] as f64 + f[5] as f64;
        let mut x1 = self.win_origin[0] as f64 + f[7] as f64;
        let mut y1 = self.win_origin[1] as f64 + f[8] as f64;
        if x0 > x1 {
            std::mem::swap(&mut x0, &mut x1);
        }
        if y0 > y1 {
            std::mem::swap(&mut y0, &mut y1);
        }
        // Pixel p is covered when x0 <= p + 0.5 < x1, clamped to the plane.
        x0 = cmax((x0 - 0.5).ceil(), 0.0);
        y0 = cmax((y0 - 0.5).ceil(), 0.0);
        x1 = cmin((x1 - 0.5).ceil(), SGI_ODYSSEY_WIDTH as f64) - 1.0;
        y1 = cmin((y1 - 0.5).ceil(), SGI_ODYSSEY_HEIGHT as f64) - 1.0;
        if x0 > x1 || y0 > y1 {
            return;
        }
        // The buffers the drawing mask selects; none leaves only depth.
        op.ntargets = 0;
        let buffers = self.draw_buffers();
        for b in 0..ODSY_BUFFERS {
            if buffers & (1u32 << b) != 0 {
                op.targets[op.ntargets as usize] = b;
                op.ntargets += 1;
            }
        }
        op.x0 = x0 as i32;
        op.y0 = y0 as i32;
        op.x1 = x1 as i32;
        op.y1 = y1 as i32;
        for i in 0..self.clip_kept as usize {
            op.clip.r[i] = self.clip_rect[i];
        }
        op.clip.n = self.clip_kept;
        if !self.gl_index_mode() {
            op.clear_colour_mask = (self.gl_write_mask >> ODSY_WRITE_COLOUR_SHIFT) & ODSY_WRITE_COLOUR_MASK;
        }
        if (op.clear_buffers & SGI_GL_COLOR_BUFFER_BIT) != 0 {
            if self.gl_index_mode() {
                // glClearIndex travels as the first float (the probe's 3.0).
                gl_index_rgba(float_index(self.gl_rect[0]), &mut op.clear_rgba);
            } else {
                op.clear_rgba.copy_from_slice(&f[..4]);
            }
        }
        if !patterned {
            self.gl_raster(&op);
            return;
        }
        let target = self.gl_target();
        self.gl_report_refusals(op.kind != SGI_RASTER_CLEAR || (op.clear_buffers & SGI_GL_COLOR_BUFFER_BIT) != 0);
        // Proposed shape-4 policy: the first corner anchors a repeating bitmap,
        // independent of corner order. Admit runs through the existing clear so
        // clear bits preserve both colour and depth, with clipping unchanged.
        let (xs, xe) = (x0 as i32, x1 as i32);
        let mut y = y0 as i32;
        while y <= y1 as i32 {
            let row = self.pattern_buffer[((y as u32).wrapping_sub(anchor_y) & 31) as usize];
            let mut x = xs;

            op.y0 = y;
            op.y1 = y;
            while x <= xe {
                while x <= xe && ((row >> (31 - ((x as u32).wrapping_sub(anchor_x) & 31))) & 1) == 0 {
                    x += 1;
                }
                if x > xe {
                    break;
                }
                op.x0 = x;
                loop {
                    x += 1;
                    if !(x <= xe && ((row >> (31 - ((x as u32).wrapping_sub(anchor_x) & 31))) & 1) != 0) {
                        break;
                    }
                }
                op.x1 = x - 1;
                rules.raster(&target, &mut self.gl_parts(None).0, &op);
            }
            y += 1;
        }
        self.scanout.invalidate_all();
    }

    /// One QUADS rectangle: the bounding box of the four vertices from first,
    /// painted through the id-plane, pattern or solid path.
    pub fn gl_quad(&mut self, first: u32) {
        let f = first as usize;
        let mut minx = self.gl_vertex_x[f] as i32;
        let mut maxx = minx;
        let mut miny = self.gl_vertex_y[f] as i32;
        let mut maxy = miny;

        for i in f + 1..f + 4 {
            let vx = self.gl_vertex_x[i] as i32;
            let vy = self.gl_vertex_y[i] as i32;

            minx = minx.min(vx);
            maxx = maxx.max(vx);
            miny = miny.min(vy);
            maxy = maxy.max(vy);
        }

        let x0 = (minx as i64).max(0);
        let x1 = (maxx as i64).min(SGI_ODYSSEY_WIDTH as i64);
        let y0 = (miny as i64).max(0);
        let y1 = (maxy as i64).min(SGI_ODYSSEY_HEIGHT as i64);
        if x0 >= x1 || y0 >= y1 {
            return;
        }

        // The X server's drawable-id plane write, marked by the 0x1017
        // destination word: it updates the did plane under the rectangle and does
        // not touch the colour framebuffer. The id value is the raw COLOR_INDEX
        // operand (id << 3), the mask the 0x1023 operand shifted down from its <<
        // 12 position. A pattern quad there is overlay text and goes through
        // the pattern path, which Board::paint_integer() sends to the same
        // bytes.
        if self.dest_is_id_planes() && !self.pattern_enabled() {
            let value = self.gl_color_index as u8;
            let mask = ((self.plane_write_mask >> ODSY_PLANE_MASK_SHIFT) & 0xff) as u8;
            self.clear_planes(x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32, value, mask);
            return;
        }

        let value = if self.gl_color_is_index { self.gl_color_index } else { self.gl_color };
        if self.pattern_enabled() {
            let is_index = self.gl_color_is_index;
            self.pattern_fill(x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32, (x0 - minx as i64) as u32, (y0 - miny as i64) as u32, is_index, value);
        } else {
            let is_index = self.gl_color_is_index;
            self.fill_rect(x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32, is_index, value);
        }
    }

    /// One line or polygon pixel in the current colour, clipped to the plane.
    pub fn gl_plot(&mut self, x: i64, y: i64) {
        let is_index = self.gl_color_is_index;
        let value = if is_index { self.gl_color_index } else { self.gl_color };
        self.paint_integer(x, y, is_index, value);
    }

    /// One line segment. The DDX pushes an X span (x, y, w) as the vertices (x, y),
    /// (x + w, y) for a solid span and in the reverse order for a stippled one, and
    /// both are right only if a line paints the pixels whose centres lie strictly
    /// between its endpoints along the major axis -- [min, max) -- whichever way it
    /// runs (INFERRED from the two DDX routines; the same is what lets the X
    /// server's line routine add a one-pixel stub to draw a polyline's last point).
    /// Along the major axis pixel i is painted at the minor-axis position of its
    /// centre, rounded down. With the stipple enabled (0x0404 bits 3 and 4) the k-th
    /// covered pixel from the first vertex, k = 1 next to it, is painted only where
    /// bit (k & 31) of the 0x0c48 pattern is set, and where it is clear, with 0x0404
    /// bit 17 set, in the 0x0c21 constant colour (Board::pattern_background()),
    /// when 0x0c21's factors are ONE, ZERO. The X server's stippled-span and
    /// dashed-line routines write no 0x043f, so its shape is whatever an earlier
    /// fill left; they set bit 17 and send 0x0c21 = ONE, ZERO, background exactly
    /// for FillOpaqueStippled spans and LineDoubleDash lines, the two forms whose
    /// clear bits X paints in the background (ESTABLISHED as software behaviour
    /// from the X driver module). Requiring ONE, ZERO keeps a GL client's blended
    /// integer stippled line, bit 17 with its own factors, from painting clear
    /// bits. That the board then paints the clear bits in the constant colour is
    /// INFERRED as for the opaque quads.
    pub fn gl_segment_pixel(&mut self, x: i64, y: i64, clear: bool) {
        if !clear {
            self.gl_plot(x, y);
        } else if (self.pattern_mode & ODSY_MODE_BLEND) != 0 && self.blend_word == ODSY_BLEND_X_OPAQUE {
            let is_index = self.gl_color_is_index;
            let value = self.pattern_background(is_index);
            self.paint_integer(x, y, is_index, value);
        }
    }

    pub fn gl_segment(&mut self, x1: i32, y1: i32, x2: i32, y2: i32, stippled: bool) {
        let pattern = self.line_stipple[ODSY_LINE_STIPPLE_PATTERN];
        let dx = x2 as f64 - x1 as f64;
        let dy = y2 as f64 - y1 as f64;

        if stippled && self.line_stipple[ODSY_LINE_STIPPLE_CONTROL] != ODSY_LINE_STIPPLE_CONTROL_32 && !self.line_stipple_reported {
            self.line_stipple_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: integer line stipple control {:#x} is not modeled; drawn as a 32-pixel stipple", self.line_stipple[ODSY_LINE_STIPPLE_CONTROL]);
        }

        // These captured Xlib modes distinguish inclusion of the final vertex.
        // The bit's physical meaning is unknown. Limit this model policy to
        // solid axial independent lines; other integer paths retain their rules.
        if !stippled
            && self.gl_primitive == SGI_GL_LINES
            && ((self.pattern_mode & !(1u32 << 25)) == 0x00080000 || (self.pattern_mode & !(1u32 << 25)) == 0x00100000)
            && (x1 == x2 || y1 == y2)
        {
            let omit_last = (self.pattern_mode & (1u32 << 25)) != 0;

            if y1 == y2 {
                let lo = (x1.min(x2) as i64).max(0);
                let hi = (x1.max(x2) as i64).min(SGI_ODYSSEY_WIDTH as i64 - 1);
                let mut i = lo;
                while i <= hi {
                    if !omit_last || i != x2 as i64 {
                        self.gl_plot(i, y1 as i64);
                    }
                    i += 1;
                }
            } else {
                let lo = (y1.min(y2) as i64).max(0);
                let hi = (y1.max(y2) as i64).min(SGI_ODYSSEY_HEIGHT as i64 - 1);
                let mut i = lo;
                while i <= hi {
                    if !omit_last || i != y2 as i64 {
                        self.gl_plot(x1 as i64, i);
                    }
                    i += 1;
                }
            }
            return;
        }
        if dx == 0.0 && dy == 0.0 {
            return;
        }
        if dx.abs() >= dy.abs() {
            let lo = (x1.min(x2) as i64).max(0);
            let hi = (x1.max(x2) as i64).min(SGI_ODYSSEY_WIDTH as i64);
            let mut i = lo;
            while i < hi {
                let c = i as f64 + 0.5;
                let k = ((c - x1 as f64).abs() + 0.5) as u32;

                self.gl_segment_pixel(i, (y1 as f64 + (c - x1 as f64) * dy / dx).floor() as i64, stippled && ((pattern >> (k & 31)) & 1) == 0);
                i += 1;
            }
        } else {
            let lo = (y1.min(y2) as i64).max(0);
            let hi = (y1.max(y2) as i64).min(SGI_ODYSSEY_HEIGHT as i64);
            let mut i = lo;
            while i < hi {
                let c = i as f64 + 0.5;
                let k = ((c - y1 as f64).abs() + 0.5) as u32;

                self.gl_segment_pixel((x1 as f64 + (c - y1 as f64) * dx / dy).floor() as i64, i, stippled && ((pattern >> (k & 31)) & 1) == 0);
                i += 1;
            }
        }
    }

    /// A POLYGON is filled by the even-odd rule, the pixels whose centres are
    /// inside. It streams like every other integer primitive, so instead of a
    /// vertex list it keeps crossing marks (poly_marks): each edge toggles, on
    /// every row whose centre lies in [its lower y, its upper y) -- so a shared
    /// vertex is crossed once -- the mark at the first pixel whose centre is at or
    /// right of the crossing. At END a pixel is inside when an odd number of marks
    /// lie at or left of it, which is the same pixel set as pairing the sorted
    /// crossings of each row. INFERRED: the two traced polygons are four-vertex
    /// convex trapezoids, for which every inside rule agrees.
    pub fn gl_polygon_edge(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
        let lo = (y1.min(y2) as i64).max(0);
        let hi = (y1.max(y2) as i64).min(SGI_ODYSSEY_HEIGHT as i64);
        let mut row = lo;

        while row < hi {
            let c = row as f64 + 0.5;
            let px = (x1 as f64 + (c - y1 as f64) * (x2 as f64 - x1 as f64) / (y2 as f64 - y1 as f64) - 0.5).ceil();

            // A crossing right of the plane leaves every plane pixel outside.
            if px >= SGI_ODYSSEY_WIDTH as f64 {
                row += 1;
                continue;
            }
            let col: u32 = if px > 0.0 { px as u32 } else { 0 };
            self.poly_marks[(row as usize) * SGI_ODYSSEY_WIDTH as usize + col as usize] ^= 1;
            self.poly_row_lo = (self.poly_row_lo as i64).min(row) as u32;
            self.poly_row_hi = (self.poly_row_hi as i64).max(row + 1) as u32;
            self.poly_col_lo = self.poly_col_lo.min(col);
            row += 1;
        }
    }

    /// Clear the polygon's marks; with paint, first fill what they enclose.
    pub fn gl_polygon_close(&mut self, paint: bool) {
        let mut row = self.poly_row_lo;

        while row < self.poly_row_hi {
            let base = row as usize * SGI_ODYSSEY_WIDTH as usize;
            let mut inside: u8 = 0;

            let mut col = self.poly_col_lo;
            while col < SGI_ODYSSEY_WIDTH {
                inside ^= self.poly_marks[base + col as usize];
                self.poly_marks[base + col as usize] = 0;
                if paint && inside != 0 {
                    self.gl_plot(col as i64, row as i64);
                }
                col += 1;
            }
            row += 1;
        }
        self.poly_row_lo = SGI_ODYSSEY_HEIGHT;
        self.poly_row_hi = 0;
        self.poly_col_lo = SGI_ODYSSEY_WIDTH;
    }

    pub fn gl_hold(&mut self, slot: u32, x: u32, y: u32) {
        self.gl_vertex_x[slot as usize] = x;
        self.gl_vertex_y[slot as usize] = y;
    }

    /// One integer vertex, window origin already added. The command FIFO is a
    /// stream with no batch size, so each piece is drawn as its last vertex
    /// arrives, the way the shared GL context assembles the float path, holding
    /// only what a later piece needs: a QUADS rectangle's corners, a LINES span's
    /// first vertex, a strip's first and previous vertex (slots 0 and 1). A
    /// trailing partial piece -- a QUADS remainder, an odd LINES vertex -- is
    /// never drawn, as glEnd drops it. A primitive this path does not draw is
    /// reported at END.
    ///
    /// A POINTS vertex paints pixel (x, y). INFERRED: a line's pixels lie at the
    /// minor-axis position rounded down (Board::gl_segment()), so an integer
    /// vertex names the top-left corner of its pixel, and a size-one point lands
    /// on the pixel GL's rule (floor of each coordinate) picks. The point size is
    /// not modeled.
    pub fn gl_vertex(&mut self, x: u32, y: u32) {
        let n = self.gl_vertex_count;
        self.gl_vertex_count = n.wrapping_add(1);
        let stippled = (self.pattern_mode & ODSY_PATTERN_MODE_ENABLE) == ODSY_PATTERN_MODE_ENABLE;

        // The (0, 0) vertex a BEGIN-0xa image carries is not geometry.
        if self.image_active {
            return;
        }
        match self.gl_primitive {
            ODSY_GL_POINTS => {
                self.gl_plot(x as i32 as i64, y as i32 as i64);
            }
            ODSY_GL_QUADS => {
                // One rectangle per four vertices (the greeter's frame fills).
                self.gl_hold(n % 4, x, y);
                if n % 4 == 3 {
                    self.gl_quad(0);
                }
            }
            ODSY_GL_LINES => {
                if n & 1 != 0 {
                    let (hx, hy) = (self.gl_vertex_x[0], self.gl_vertex_y[0]);
                    self.gl_segment(hx as i32, hy as i32, x as i32, y as i32, stippled);
                } else {
                    self.gl_hold(0, x, y);
                }
            }
            ODSY_GL_LINE_STRIP | ODSY_GL_LINE_LOOP => {
                if n == 0 {
                    self.gl_hold(0, x, y);
                } else {
                    let (hx, hy) = (self.gl_vertex_x[1], self.gl_vertex_y[1]);
                    self.gl_segment(hx as i32, hy as i32, x as i32, y as i32, stippled);
                }
                self.gl_hold(1, x, y);
            }
            ODSY_GL_POLYGON => {
                if n == 0 {
                    self.gl_hold(0, x, y);
                } else {
                    let (hx, hy) = (self.gl_vertex_x[1], self.gl_vertex_y[1]);
                    self.gl_polygon_edge(hx as i32, hy as i32, x as i32, y as i32);
                }
                self.gl_hold(1, x, y);
            }
            _ => {}
        }
    }

    pub fn gl_end(&mut self) {
        let count = self.gl_vertex_count;
        let stippled = (self.pattern_mode & ODSY_PATTERN_MODE_ENABLE) == ODSY_PATTERN_MODE_ENABLE;

        // END always closes the current primitive, the same as a real
        // immediate-mode glEnd(): the next BEGIN starts a fresh vertex count,
        // and a stray END with nothing newly accumulated since -- exactly
        // what the console text path issues to close out its per-row pixel runs,
        // which never call BEGIN again -- paints nothing.
        self.gl_vertex_count = 0;

        // END also closes a BEGIN-0xa image. The box and bursts already painted
        // as they streamed, so there is nothing more to do but leave image mode.
        if self.image_active {
            self.image_active = false;
            self.image_bytes = 0;
            self.image_nbytes = 0;
            return;
        }

        // Every piece but a loop's closing edge was drawn as it streamed.
        match self.gl_primitive {
            ODSY_GL_POINTS | ODSY_GL_QUADS | ODSY_GL_LINES | ODSY_GL_LINE_STRIP => {}
            ODSY_GL_LINE_LOOP => {
                if count >= 2 {
                    let (ax, ay, bx, by) = (self.gl_vertex_x[1], self.gl_vertex_y[1], self.gl_vertex_x[0], self.gl_vertex_y[0]);
                    self.gl_segment(ax as i32, ay as i32, bx as i32, by as i32, stippled);
                }
            }
            ODSY_GL_POLYGON => {
                // Fewer than three vertices enclose nothing.
                if count >= 3 {
                    let (ax, ay, bx, by) = (self.gl_vertex_x[1], self.gl_vertex_y[1], self.gl_vertex_x[0], self.gl_vertex_y[0]);
                    self.gl_polygon_edge(ax as i32, ay as i32, bx as i32, by as i32);
                }
                self.gl_polygon_close(count >= 3);
            }
            _ => {
                // A glBegin/glEnd pair with no vertex between draws nothing in GL
                // (2.6.1), whatever the primitive. Performer's town and pointblank
                // send such pairs, a texture coordinate and no vertex.
                if count == 0 {
                    return;
                }
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: command FIFO OpenGL primitive {:#x} is not modeled", self.gl_primitive);
                return;
            }
        }
        self.scanout.invalidate_all();
    }
}
