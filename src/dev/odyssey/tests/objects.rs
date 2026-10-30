//! Board-memory objects as texture levels: the model's object store keeps
//! what a game uploads, and texture setup finds a level's storage there.

use crate::dev::odyssey::board::{Board, ODSY_OBJECTS};
use crate::dev::odyssey::consts::tex;
use crate::dev::odyssey::gl::{texture_detail_sampler, volume_slice_stride, BoardTexels, TextureChain, TextureDetailWeight};
use crate::dev::odyssey::gl_context::{
    GL_ADD, GL_ALPHA, GL_CLAMP, GL_CLAMP_TO_EDGE, GL_INTENSITY, GL_LINEAR, GL_LINEAR_DETAIL_COLOR_SGIS, GL_LUMINANCE,
    GL_LUMINANCE_ALPHA, GL_MODULATE, GL_RGB, GL_RGBA,
};
use crate::dev::odyssey::sampler::{
    texel_format, texture_class_format, texture_level_texel_format, Placement, SamplerContext, Target, Wrap,
};
use crate::dev::odyssey::raster::GlTexImage;
use crate::dev::odyssey::tests::board_with_ram;

/// TyrQuake on IRIX 6.5.30 keeps some 1,500 texture levels on the board at
/// once, each its own board-memory object, and samples the first ones it
/// loaded, the console font and the menu pictures, every frame. A table of
/// 1024 objects forgot the oldest and those draws went untextured, white.
/// The table holds ODSY_OBJECTS, and only the one past that forgets.
#[test]
fn the_object_table_keeps_every_level_a_game_loads() {
    let (mut s, _, _) = board_with_ram();
    let base = |i: u32| 0x16b000 + i * 0x100;

    for i in 0..ODSY_OBJECTS {
        let o = s.object_select(base(i), 16);
        assert!(s.object_rows(o, 1), "object {i}");
    }
    assert!(s.object_find(base(0)).is_some(), "the first object loaded was forgotten");
    s.object_select(base(ODSY_OBJECTS), 16);
    assert!(s.object_find(base(0)).is_none(), "a full table forgets the oldest");
    assert!(s.object_find(base(1)).is_some() && s.object_find(base(ODSY_OBJECTS)).is_some());
}

/// The texture state of a plain 2D RGBA level, as chocolate-doom's SDL
/// renderer sends it for its 512x256 frame texture: clamp to edge on both
/// axes, nearest, replace, one level at `base` (sampler word 4 0x13001400,
/// level entry 0x41100020, replayed from a guest trace).
fn rgba_512x256(s: &mut Board, base: u32) {
    s.tex.mode = tex::MODE_2D;
    s.tex.level0 = 0;
    s.tex.levels[..3].copy_from_slice(&[base, 0x4110_0020, 0]);
    s.tex.sampler = [0x0800_0000 | (9 << 10) | (8 << 5), 0x0004_0000 | 512, 0x0004_0000 | 256, 1, 0x1300_1400, 0, 0];
    s.tex.env = [1, 0, 0, 0, 0];
    s.tex.lod = [0; 4];
}

/// glTexImage2D with no data and then glTexSubImage2D of a 320x200 frame,
/// SDL's way of streaming a texture: the object holds only the 200 rows the
/// subimage wrote of the level's 256. The level is still the client's
/// texture, so setup must take it, with the rows nobody wrote reading 0.
/// Before, setup called the level missing and drew the frame untextured,
/// a white window.
#[test]
fn a_level_written_only_in_part_is_still_the_texture() {
    let (mut s, _, _) = board_with_ram();
    let base = 0x16b800;
    let obj = s.object_select(base, 512);

    for y in 0..200 {
        s.object_store(obj, 0, y, 0x1122_33ff);
    }
    rgba_512x256(&mut s, base);
    let mut image = GlTexImage::default();
    let mut chain = Box::<TextureChain>::default();

    assert!(s.texture_setup(&mut image, &mut chain));
    assert!(chain.levels[0].read_texel, "level 0 is missing");
    assert_eq!((chain.storage[0].width, chain.storage[0].height), (512, 256));
    assert_eq!(s.object_load(obj, 0, 199), 0x1122_33ff);
    assert_eq!(s.object_load(obj, 0, 255), 0, "a row the client never wrote");
}

/// The rows an object holds after texture setup under `word` (sampler word
/// 4), `sampler6` (word 6) and level entry word `entry1`, for the 512x256
/// level of an object `width` wide that holds 200 rows. The rest of the
/// state is rgba_512x256()'s.
fn rows_after_setup(width: u32, word: u32, sampler6: u32, entry1: u32) -> u32 {
    let (mut s, _, _) = board_with_ram();
    let base = 0x16b800;
    let obj = s.object_select(base, width);

    assert!(s.object_rows(obj, 200));
    rgba_512x256(&mut s, base);
    s.tex.sampler[4] = word;
    s.tex.sampler[6] = sampler6;
    s.tex.levels[1] = entry1;
    if word & 0x3000 == 0 {
        // No clamp to edge, so no clamp mark in the size words.
        s.tex.sampler[1] = 512;
        s.tex.sampler[2] = 256;
    }
    let mut image = GlTexImage::default();
    let mut chain = Box::<TextureChain>::default();
    s.texture_setup(&mut image, &mut chain);
    s.objects.rows[obj as usize]
}

/// Only a level of the client's own row length grows: an object narrower
/// than the level, or one whose row length is not the level entry's, is
/// some other buffer and stays as it is.
#[test]
fn a_level_grows_only_into_an_object_of_its_row_length() {
    const ENTRY: u32 = 0x4110_0020;
    // A one-level entry may leave its row length 0 (no stride to check).
    const NO_STRIDE: u32 = 0x4110_0000;

    assert_eq!(rows_after_setup(512, 0x1300_1400, 0, ENTRY), 256, "the control");
    assert_eq!(rows_after_setup(512, 0x1300_1400, 0, NO_STRIDE), 256, "the control with no row length");
    assert_eq!(rows_after_setup(256, 0x1300_1400, 0, NO_STRIDE), 200, "narrower than the level");
    assert_eq!(rows_after_setup(1024, 0x1300_1400, 0, ENTRY), 200, "not the entry's row length");
}

/// A level with its border stored around it keeps its own checks: its
/// object needs rows past the level's height, so growing it to that height
/// would claim a level the border check then reads as written.
#[test]
fn a_bordered_level_does_not_grow() {
    // Word 4 bit 28 clear and word 6's top byte 1: the stored-border marks.
    assert_eq!(rows_after_setup(512, 0x0300_0000, 0x0100_0000, 0x4110_0020), 200);
}

/// The texture move draws only the rows it wrote, from a scratch object
/// sized to them; it must not grow that object to the power of two the
/// sampler names.
#[test]
fn a_texture_move_does_not_grow_its_object() {
    let (mut s, _, _) = board_with_ram();
    let base = 0x16b000;
    let obj = s.object_select(base, 512);

    assert!(s.object_rows(obj, 2));
    s.tex.mode = tex::MODE_2D;
    s.tex.level0 = 60;
    s.tex.levels[180..183].copy_from_slice(&[base, 0x8210_0020, 0]);
    s.tex.sampler = [(9 << 10) | (6 << 5), 512, 64, 0, 0x0364_0000, 0, 0];
    s.tex.env = [1, 0, 0, 0, 0];
    let mut image = GlTexImage::default();
    let mut chain = Box::<TextureChain>::default();
    assert!(s.texture_setup(&mut image, &mut chain));
    assert_eq!(s.objects.rows[obj as usize], 2);
}

/// Packing the store keeps every live object's texels and its place in
/// store order, closes the holes forgotten objects left, and puts them all
/// at the front. With a full table of 16384 objects, some moved to the end
/// by growing and a third forgotten, this is also the case where a packing
/// that searched the table for each object took a third of a second.
#[test]
fn packing_a_full_table_keeps_each_object_in_store_order() {
    let (mut s, _, _) = board_with_ram();
    let n = ODSY_OBJECTS as usize;

    for i in 0..n {
        let o = s.object_select(0x10_0000 + i as u32 * 0x100, 16);
        assert_eq!(o, i);
        s.object_store(o, 0, 0, i as u32);
    }
    for i in (0..n).step_by(7) {
        // Growth moves the object to the store's end.
        s.object_store(i, 0, 1, !(i as u32));
    }
    for i in (1..n).step_by(3) {
        s.object_forget(i);
    }
    let mut live: Vec<usize> = (0..n).filter(|&i| s.objects.base[i] != 0).collect();
    live.sort_by_key(|&i| s.objects.offset[i]);

    s.object_compact(None);
    let mut next = 0u32;
    for &i in &live {
        let o = i;
        assert_eq!(s.objects.offset[o], next, "object {i} is not packed in store order");
        next += s.objects.width[o] * s.objects.rows[o];
        assert_eq!(s.object_load(i, 0, 0), i as u32, "object {i}'s texel");
    }
    assert_eq!(s.objects.used, next);
}

/// A stored volume's slices keep their own check: every row and texel the
/// volume needs must be written. A slice written in part leaves the volume
/// unusable, and is not grown to look complete as a 2D level is. The
/// control, with every slice whole, shows the state is a usable volume.
#[test]
fn a_stored_volume_slice_written_in_part_does_not_grow() {
    for partial in [false, true] {
        let (mut s, _, _) = board_with_ram();
        let base = 0x10_0000;
        // A 4x4x4 RGBA volume with its border stored: each slice an object
        // 64 texels wide, the level 15 rows and columns in, plus the two
        // border slices.
        let (w, h, d) = (4u32, 4u32, 4u32);
        let stride = volume_slice_stride(h + 2, texel_format(17).unwrap()) as u32;
        let mut rows = Vec::new();

        // The border slice sits fifteen slice strides up, as rows and
        // columns sit 15 in.
        for z in 0..d + 2 {
            let o = s.object_select(base + (15 + z) * stride, 64);
            let n = if partial && z == 2 { 18 } else { 15 + h + 2 };
            for y in 0..n {
                for x in 0..64 {
                    s.object_store(o, x, y, 0x1020_30ff);
                }
            }
            rows.push((o, n));
        }
        s.tex.mode = tex::MODE_3D;
        s.tex.level0 = 0;
        s.tex.levels[..3].copy_from_slice(&[base, 0x4110_0004, h + 2]);
        s.tex.sampler = [0x0800_0000 | (2 << 10) | (2 << 5) | 2, w, h, d, 0x0300_2a00, 0, 0x0100_0000];
        s.tex.env = [1, 0, 0, 0, 0];
        let mut image = GlTexImage::default();
        let mut chain = Box::<TextureChain>::default();
        let taken = s.texture_setup(&mut image, &mut chain) && chain.levels[0].read_texel_3d;

        assert_eq!(taken, !partial, "partial {partial}");
        for (o, n) in rows {
            assert_eq!(s.objects.rows[o as usize], n, "slice object {o}, partial {partial}");
        }
    }
}

/// Sampler word 0's detail fields: the six captured detail controls' words
/// decode to the level, size, entry and combine those controls were built
/// with, and SGI Quake's world word (0x0873a500, a 512x256 base) to the
/// 128x128 detail image at the default level -4, which sits in entry 60.
#[test]
fn detail_sampler_words_name_the_detail_image() {
    let cases = [
        (0x0210_8000, 63, 2, 2, -1, GL_ADD),
        (0x0410_8000, 62, 2, 2, -2, GL_ADD),
        (0x0610_8000, 61, 2, 2, -3, GL_ADD),
        (0x0420_8000, 62, 4, 2, -2, GL_ADD),
        (0x0411_0000, 62, 2, 4, -2, GL_ADD),
        (0x8410_8000, 62, 2, 2, -2, GL_MODULATE),
        (0x0873_a500, 60, 128, 128, -4, GL_ADD),
    ];
    for (word, descriptor, width, height, level, mode) in cases {
        let d = texture_detail_sampler(word).unwrap_or_else(|| panic!("{word:#x} refused"));
        assert_eq!((d.descriptor, d.width, d.height, d.level, d.mode), (descriptor, width, height, level, mode), "{word:#x}");
    }
    assert!(texture_detail_sampler(0x0010_8000).is_none(), "level 0 is no detail level");
    assert!(texture_detail_sampler(0x4410_8000).is_none(), "bit 30 has no known meaning");
}

/// The library's default ramp, weight 0 at LOD 0 to 1 at LOD -4: the weight
/// is the end entry's beyond the table and interpolated between entries.
#[test]
fn the_detail_weight_follows_the_function_between_entries() {
    let mut w = TextureDetailWeight::default();
    for (i, e) in w.table.iter_mut().enumerate() {
        *e = (i as u16 * 32).min(4095);
    }
    assert_eq!(w.at(0.5), Some(0.0));
    assert_eq!(w.at(-1.0), Some(1024.0 / 4096.0));
    assert_eq!(w.at(-1.0 / 64.0), Some(16.0 / 4096.0));
    assert_eq!(w.at(-2.5), Some(2560.0 / 4096.0));
    assert_eq!(w.at(-9.0), Some(4095.0 / 4096.0));
    assert_eq!(w.at(f64::NAN), None);
}

/// SGI Quake's world textures as the library left them: a 512x256 RGB base
/// with ten mip levels, sampler word 4 `word4`, modulate, and a 128x128
/// detail image in entry 60 whose word 1 is `detail_word1`.
fn quake_world_state(s: &mut Board, word4: u32, detail_word1: u32) {
    let base = 0x1d90c0;
    let mut level_base = base;

    s.tex.mode = 0x2410;
    s.tex.level0 = 0;
    for i in 0..10u32 {
        let (w, h) = ((512u32 >> i).max(1), (256u32 >> i).max(1));
        let units = (w.div_ceil(16) + 1) & !1;
        let o = s.object_select(level_base, units * 16);
        assert!(s.object_rows(o, h));
        s.tex.levels[i as usize * 3..][..3].copy_from_slice(&[level_base, 0x00d0_0000 | units, 0]);
        level_base += units * 16 * h;
    }
    let detail = 0x16b800;
    let o = s.object_select(detail, 128);
    assert!(s.object_rows(o, 128));
    for y in 0..128 {
        for x in 0..128 {
            s.object_store(o, x, y, 0x8080_80ff);
        }
    }
    s.tex.levels[60 * 3..][..3].copy_from_slice(&[detail, detail_word1, 0]);
    s.tex.sampler = [0x0873_a500, 0x200, 0x100, 1, word4, 0, 0];
    s.tex.lod = [0, 0, 0x3100, 0xf00];
    s.tex.env = [2, 0, 0, 0, 0];
    for (i, e) in s.tex.detail_function.iter_mut().enumerate() {
        *e = ((i as u32 * 32).min(4095)) << 12;
    }
    s.tex.detail_function_valid = 0xffff;
}

/// SGI Quake's world textures, replayed from its texture state: a 512x256
/// RGB base with ten mip levels under GL_LINEAR_DETAIL_COLOR_SGIS (sampler
/// word 4 0x128000a8: minification code 5, magnification code 4), modulate,
/// and a 128x128 detail image in entry 60. Setup must accept it: a
/// refused draw leaves the whole world undrawn, black.
#[test]
fn sgi_quake_world_texture_takes_its_detail_image() {
    let (mut s, _, _) = board_with_ram();
    quake_world_state(&mut s, 0x1280_00a8, 0x00d0_0008);
    let mut image = GlTexImage::default();
    let mut chain = Box::<TextureChain>::default();

    assert!(s.texture_setup(&mut image, &mut chain), "the draw was refused");
    assert_eq!(image.mag_filter, GL_LINEAR_DETAIL_COLOR_SGIS);
    assert_eq!((image.width, image.height, image.base_format), (512, 256, GL_RGB));
    assert_eq!(chain.pyramid.map(|p| p.count), Some(10));
    assert_eq!((chain.detail_storage.width, chain.detail_storage.height), (128, 128));
    assert_eq!((image.detail_level, image.detail_mode), (-4, GL_ADD));
    assert_eq!(chain.detail_weight.table[128], 4095);
}

/// Word 0 bits 24:15 name a detail image only while word 4 carries a detail
/// magnification code; the library leaves them behind after a change to a
/// plain filter. Quake's state with magnification code 1 (linear) is a plain
/// mipmapped image and takes no detail image, where code 4 takes one.
#[test]
fn stale_detail_size_bits_are_ignored_without_a_detail_magnification_code() {
    let (mut s, _, _) = board_with_ram();
    quake_world_state(&mut s, 0x1280_00a2, 0x00d0_0008);
    let mut image = GlTexImage::default();
    let mut chain = Box::<TextureChain>::default();

    assert!(s.texture_setup(&mut image, &mut chain), "the plain draw was refused");
    assert_eq!(image.mag_filter, GL_LINEAR);
    assert!(image.detail_image.is_none());
    assert_eq!((image.width, image.height, image.base_format), (512, 256, GL_RGB));
    assert_eq!(chain.detail_storage.width, 0);
}


/// A detail level names its own format index, which must be a table row
/// with its own size code, but its components are the primary's class: only
/// the bound texture's mode word reaches the board. An index the table does
/// not list, or one whose size code is not its own, is refused.
#[test]
fn the_detail_image_takes_its_index_and_the_primary_class() {
    const RGB_WORD4: u32 = 0x1280_00a8;
    const RGBA_WORD4: u32 = 0x1300_00a8;
    let cases = [
        // (sampler word 4, detail entry word 1, takes the draw)
        (RGB_WORD4, 0x4110_0008, true),
        (RGBA_WORD4, 0x4110_0008, true),
        (RGB_WORD4, 0x00c0_0008, true),
        (RGBA_WORD4, 0x00d0_0008, true),
        (RGB_WORD4, 0x8210_0008, true),
        (RGB_WORD4, 0x00b0_0008, true),
        (RGBA_WORD4, 0x4100_0008, true),
        (RGB_WORD4, 0x0040_0008, true),
        (RGB_WORD4, 0x0000_0008, true),
        (RGBA_WORD4, 0x0020_0008, true),
        (RGBA_WORD4, 0x0300_0008, true),
        (RGB_WORD4, 0x4310_0008, true),
        // Indices nothing emits, and a size code that is not the index's.
        (RGB_WORD4, 0x0230_0008, false),
        (RGB_WORD4, 0x00e0_0008, false),
        (RGB_WORD4, 0x0110_0008, false),
        (RGB_WORD4, 0x4040_0008, false),
    ];
    for (word4, word1, takes) in cases {
        let (mut s, _, _) = board_with_ram();
        quake_world_state(&mut s, word4, word1);
        let mut image = GlTexImage::default();
        let mut chain = Box::<TextureChain>::default();
        let drew = s.texture_setup(&mut image, &mut chain);
        let primary = if word4 == RGB_WORD4 { GL_RGB } else { GL_RGBA };

        assert_eq!(drew, takes, "word 4 {word4:#x} entry {word1:#x}");
        if takes {
            assert_eq!(chain.detail_storage.base_format, primary, "word 4 {word4:#x} entry {word1:#x}");
            assert_eq!(chain.detail_image.base_format, primary);
        }
    }
}

/// The format index table: sixteen indices are emitted, each with its own
/// size code and classes, and every other index is refused. Index 12 holds
/// 4/4/4/4 whether its class says RGB or RGBA.
#[test]
fn the_format_index_table_lists_the_sixteen_emitted_indices() {
    // (index, size code, classes as bits 1 to 6)
    let rows = [
        (0, 0, 1 << 1),
        (2, 0, 1 << 4),
        (4, 0, 1 << 2),
        (8, 0, 1 << 1),
        (9, 0, 1 << 4),
        (10, 0, 1 << 2),
        (11, 0, 1 << 3),
        (12, 0, 3 << 5),
        (13, 0, 3 << 5),
        (16, 1, 1 << 3),
        (17, 1, 3 << 5),
        (18, 1, 3 << 5),
        (32, 2, 3 << 5),
        (33, 2, 3 << 5),
        (48, 0, 1 << 5),
        (49, 1, 1 << 6),
    ];
    for index in 0..64u32 {
        let want = rows.iter().find(|r| r.0 == index);

        assert_eq!(texel_format(index).map(|f| (f.size, f.classes)), want.map(|r| (r.1, r.2)), "index {index}");
    }
    // The size code is part of the word: an index with another size is no format.
    assert_eq!(texture_level_texel_format(0x4110_0020).map(|f| f.size), Some(1));
    assert!(texture_level_texel_format(0x0110_0020).is_none());
    assert!(texture_level_texel_format(0x8110_0020).is_none());
}

/// Sampler word 4 bits 25:23 are the class: 1 alpha, 2 luminance, 3
/// luminance-alpha, 4 intensity, 5 RGB, 6 RGBA. Class 0 and 7 name no
/// components.
#[test]
fn the_sampler_class_names_the_components() {
    let want = [None, Some(GL_ALPHA), Some(GL_LUMINANCE), Some(GL_LUMINANCE_ALPHA), Some(GL_INTENSITY), Some(GL_RGB), Some(GL_RGBA), None];

    for (class, format) in want.iter().enumerate() {
        let word = (class as u32) << 23;

        assert_eq!(texture_class_format(word & tex::COMPONENTS_MASK), *format, "class {class}");
    }
}

/// Sampler word 4 bits 27:26 are not part of the class and the library never
/// writes them, so a word that sets either is refused, not ignored.
#[test]
fn sampler_word_4_bits_27_26_are_refused() {
    for bit in [26u32, 27] {
        let (mut s, _, _) = board_with_ram();
        let base = 0x16b800;
        let obj = s.object_select(base, 512);

        assert!(s.object_rows(obj, 256));
        rgba_512x256(&mut s, base);
        s.tex.levels[1] = 0x4110_0020;
        assert!(s.texture_setup(&mut GlTexImage::default(), &mut Box::<TextureChain>::default()), "the plain word is taken");
        s.tex.sampler[4] |= 1 << bit;
        assert!(!s.texture_setup(&mut GlTexImage::default(), &mut Box::<TextureChain>::default()), "bit {bit} is refused");
    }
}

/// Whether texture setup takes a plain 512x256 level whose sampler class is
/// `class` and whose entry word 1 is `entry1`.
fn level_draws(class: u32, entry1: u32) -> bool {
    let (mut s, _, _) = board_with_ram();
    let base = 0x16b800;
    let obj = s.object_select(base, 512);

    assert!(s.object_rows(obj, 256));
    rgba_512x256(&mut s, base);
    s.tex.sampler[4] = (s.tex.sampler[4] & !tex::COMPONENTS_MASK) | (class << 23);
    s.tex.levels[1] = entry1;
    let mut image = GlTexImage::default();
    let mut chain = Box::<TextureChain>::default();
    s.texture_setup(&mut image, &mut chain)
}

/// A level must name an index the library emits, with that index's own size
/// code. Index 12 is 4/4/4/4 under class 5 or 6, so class does not choose
/// the layout, and no class and index pair is refused: the texture move
/// sends class 6 with index 10. A listed index with a wrong size code, or an
/// index nothing emits (1, 3, 5 to 7, 14, 15, 19 to 31, 34 to 47, 50 to 63),
/// drops the draw.
#[test]
fn a_level_needs_an_emitted_index_and_its_size_code() {
    // (class, entry word 1, takes the draw)
    let cases = [
        (6, 0x00a0_0020, true),
        (2, 0x4110_0020, true),
        (1, 0x8200_0020, true),
        (6, 0x4110_0020, true),
        (5, 0x4110_0020, true),
        (6, 0x00c0_0020, true),
        (5, 0x00c0_0020, true),
        (3, 0x00b0_0020, true),
        (1, 0x0000_0020, true),
        (2, 0x0040_0020, true),
        (4, 0x0020_0020, true),
        (1, 0x0080_0020, true),
        (6, 0x8210_0020, true),
        (5, 0x4310_0020, true),
        (6, 0x0300_0020, true),
        (6, 0x0010_0020, false),
        (2, 0x0030_0020, false),
        (6, 0x0050_0020, false),
        (6, 0x0070_0020, false),
        (6, 0x00e0_0020, false),
        (6, 0x00f0_0020, false),
        (6, 0x0130_0020, false),
        (6, 0x0230_0020, false),
        (6, 0x03f0_0020, false),
        (6, 0x0110_0020, false),
        (6, 0x8110_0020, false),
        (6, 0x4040_0020, false),
        (6, 0xc110_0020, false),
    ];
    for (class, entry1, want) in cases {
        assert_eq!(level_draws(class, entry1), want, "class {class} entry {entry1:#x}");
    }
}

/// The library's texture move stores class 6 whatever the pixel class's
/// index, so the move takes any emitted index, and only those. Its size code
/// is the same quantity texture levels send, and is checked the same way.
#[test]
fn a_texture_move_takes_any_emitted_index_with_its_size_code() {
    for (index, wrong_size) in (0..64u32).flat_map(|i| [(i, false), (i, true)]) {
        let (mut s, _, _) = board_with_ram();
        let base = 0x16b000;
        let obj = s.object_select(base, 512);

        assert!(s.object_rows(obj, 2));
        s.tex.mode = tex::MODE_2D;
        s.tex.level0 = 60;
        let size = texel_format(index).map_or(0, |f| f.size);
        let size = if wrong_size { (size + 1) & 3 } else { size };
        s.tex.levels[180..183].copy_from_slice(&[base, (size << 30) | (index << 20) | 0x20, 0]);
        s.tex.sampler = [(9 << 10) | (6 << 5), 512, 64, 0, 0x0364_0000, 0, 0];
        s.tex.env = [1, 0, 0, 0, 0];
        let emitted = texel_format(index).is_some() && !wrong_size;

        assert_eq!(
            s.texture_setup(&mut GlTexImage::default(), &mut Box::<TextureChain>::default()),
            emitted,
            "index {index}, wrong size code {wrong_size}"
        );
    }
}

/// The sampler words decoded field by field: chocolate-doom's 512x256 RGBA
/// frame texture, clamp to edge, nearest.
#[test]
fn the_sampler_words_decode_into_fields() {
    let (mut s, _, _) = board_with_ram();
    rgba_512x256(&mut s, 0x16b800);
    let d = s.texture_sampler().expect("the control decodes");

    assert_eq!(d.target, Target::TwoD);
    assert_eq!(d.size(), [512, 256, 1]);
    assert_eq!(d.max_clamp, [512, 256, 1]);
    assert_eq!(d.wrap, [Wrap::ClampToEdge, Wrap::ClampToEdge, Wrap::Repeat]);
    assert_eq!((d.base_format, d.context, d.placement), (GL_RGBA, SamplerContext::Plain, Placement::Corner));
    assert_eq!((d.min, d.mag, d.first, d.env), (0, 0, 0, 1));
    assert!(d.detail.is_none());
}

/// Each state field the library never writes, or writes only in agreement
/// with another, is unknown state when it disagrees, and refuses the draw.
#[test]
fn words_the_library_never_writes_refuse_the_draw() {
    let control = |s: &mut Board| rgba_512x256(s, 0x16b800);
    let cases: [(&str, fn(&mut Board)); 12] = [
        ("size word bit 30", |s| s.tex.sampler[0] |= 0x4000_0000),
        ("an R size on a 2D image", |s| s.tex.sampler[0] |= 1),
        ("a size word's mark without clamp to edge", |s| s.tex.sampler[1] = 512),
        ("a size word bit past the mark", |s| s.tex.sampler[2] |= 0x0100_0000),
        ("a maximum clamp past the size", |s| s.tex.sampler[1] = 0x0004_0000 | 513),
        ("class 7", |s| s.tex.sampler[4] |= 0x0380_0000),
        ("word 4 bit 31", |s| s.tex.sampler[4] |= 0x8000_0000),
        ("bits 22:18 the model has not seen", |s| s.tex.sampler[4] |= 0x0004_0000),
        ("word 4 bit 28 and word 6 bit 24 both set", |s| s.tex.sampler[6] |= 0x0100_0000),
        ("minification code 6", |s| s.tex.sampler[4] |= 0xc0),
        ("an R wrap on a 2D image", |s| s.tex.sampler[4] |= 0x100),
        ("an offset word", |s| s.tex.offset[1] = 1),
    ];
    let (mut s, _, _) = board_with_ram();
    control(&mut s);
    assert!(s.texture_sampler().is_ok(), "the control");
    for (what, change) in cases {
        let (mut s, _, _) = board_with_ram();
        control(&mut s);
        change(&mut s);
        assert!(s.texture_sampler().is_err(), "{what} decoded");
    }
}

/// What the model cannot draw is refused by the support check, after the
/// words decode. GL_CLAMP on one axis of a 2D image and not the other is one
/// such state; on both it is drawn.
#[test]
fn clamp_on_one_axis_of_two_is_not_drawn() {
    let (mut s, _, _) = board_with_ram();
    rgba_512x256(&mut s, 0x16b800);
    s.tex.sampler[1] = 512;
    s.tex.sampler[4] = 0x1300_2400;
    let d = s.texture_sampler().expect("mixed wraps decode");
    assert!(s.texture_support(&d).is_err());

    s.tex.sampler[2] = 256;
    s.tex.sampler[4] = 0x1300_2800;
    let d = s.texture_sampler().expect("GL_CLAMP decodes");
    assert!(s.texture_support(&d).is_ok());
}

/// A level that holds no rows at all stays missing: the partial-level rule
/// grows only an object some transfer has written. Model policy.
#[test]
fn a_level_whose_object_holds_no_rows_is_missing() {
    let (mut s, _, _) = board_with_ram();
    let base = 0x16b800;
    let obj = s.object_select(base, 512);
    rgba_512x256(&mut s, base);
    let mut image = GlTexImage::default();
    let mut chain = Box::<TextureChain>::default();

    assert!(s.texture_setup(&mut image, &mut chain));
    assert!(!chain.levels[0].read_texel, "an object with no rows was read");
    assert_eq!(s.objects.rows[obj as usize], 0);
}

/// Level descriptor word 2 is not written for a 2D level, so a value left by
/// a volume that used the entry before is not this texture's and does not
/// refuse it.
#[test]
fn a_2d_level_ignores_a_stale_word_2() {
    assert!(level_draws(6, 0x4110_0020));
    let (mut s, _, _) = board_with_ram();
    let base = 0x16b800;
    let obj = s.object_select(base, 512);
    assert!(s.object_rows(obj, 256));
    rgba_512x256(&mut s, base);
    s.tex.levels[2] = 4;
    let mut chain = Box::<TextureChain>::default();

    assert!(s.texture_setup(&mut GlTexImage::default(), &mut chain));
    assert!(chain.levels[0].read_texel);
}

/// A 1D image of any width stored with a texel offset: its texels 16 in
/// and its two border texels at 15 and past its end, all of it written. The
/// border is read only when it was written.
#[test]
fn a_one_dimensional_stored_border_of_any_width_is_read() {
    for (width, written) in [(16u32, true), (32, true), (16, false)] {
        let (mut s, _, _) = board_with_ram();
        let base = 0x16b800;
        let units = ((width + 32).div_ceil(16) + 1) & !1;
        let o = s.object_select(base, units * 16);
        let last = if written { 16 + width } else { 16 + width - 1 };
        for x in 15..=last {
            s.object_store(o, x, 0, 0x1020_30ff + x);
        }
        s.tex.mode = tex::MODE_1D;
        s.tex.level0 = 0;
        s.tex.levels[..3].copy_from_slice(&[base, 0x4110_0000 | units, 0]);
        s.tex.sampler = [0x0800_0000 | (width.trailing_zeros() << 10), width, 1, 1, 0x0300_2000, 0, 0x0100_0000];
        s.tex.env = [1, 0, 0, 0, 0];
        let mut image = GlTexImage::default();
        let mut chain = Box::<TextureChain>::default();

        assert!(s.texture_setup(&mut image, &mut chain), "width {width}");
        assert_eq!(image.border, 1);
        assert_eq!(image.wrap_t, GL_CLAMP_TO_EDGE, "a 1D image takes no T border tap");
        assert_eq!(chain.levels[0].read_border_texel, written, "width {width} written {written}");
        if written {
            let texels = BoardTexels {
                chain: &chain,
                obj: &s.objects.mem.bytes,
                obj_words: s.objects.mem.size / 4,
                slice_offset: &s.tex.slice_offset,
                slice_width: &s.tex.slice_width,
                colour_table: &s.tex.colour_table,
                scale_bias: &s.tex.scale_bias,
            };
            assert_eq!(texels.read_border_level(0, -1, 0), Some(0x1020_30ff + 15));
            assert_eq!(texels.read_border_level(0, width as i32, 0), Some(0x1020_30ff + 16 + width));
            assert_eq!(texels.read(&chain.storage[0], 0, 0), Some(0x1020_30ff + 16));
        }
    }
}

/// The captured stored-border volume: a 2x2x2 RGBA interior with its border,
/// level 0 at 0x16a800 with slices 0x400 apart, its four slice objects
/// uploaded from 0x16e400, fifteen slice strides up, each 64 texels wide
/// with the interior from (16, 16). At the other observed level base,
/// 0x16b800, the same gap. A volume whose slices start at the level base
/// itself has no object where the rule looks, and is refused.
#[test]
fn the_captured_stored_volume_reads_its_slices_fifteen_strides_up() {
    for (level_base, first_slice, takes) in [(0x16a800u32, 0x16e400u32, true), (0x16b800, 0x16f400, true), (0x16a800, 0x16a800, false)] {
        let (mut s, _, _) = board_with_ram();
        for z in 0..4u32 {
            let o = s.object_select(first_slice + z * 0x400, 64);
            for y in 15..19 {
                for x in 15..19 {
                    s.object_store(o, x, y, (z << 24) | (y << 16) | (x << 8) | 0xff);
                }
            }
        }
        s.tex.mode = 0x2420;
        s.tex.level0 = 0;
        s.tex.levels[..3].copy_from_slice(&[level_base, 0x4110_0004, 4]);
        s.tex.sampler = [0x0800_0421, 2, 2, 2, 0x0300_2a00, 0x404, 0x01c0_cfff];
        s.tex.env = [1, 0, 0, 0, 0];
        let mut image = GlTexImage::default();
        let mut chain = Box::<TextureChain>::default();

        assert_eq!(s.texture_setup(&mut image, &mut chain), takes, "level base {level_base:#x} slices from {first_slice:#x}");
        if takes {
            assert_eq!((image.border, image.wrap_r, chain.volume.slices), (1, GL_CLAMP, 4));
            assert_eq!(chain.volume.base, level_base + 15 * 0x400);
        }
    }
}

/// A volume's slices sit 2^(word 2) tiles of 256 texels apart at the bytes
/// the library books for the format, in 16-byte units: 0x80 for a 4-byte
/// format at word 2 1, as the captured 8x8x8 levels sit, and half or twice
/// that for a 2- or 8-byte format.
#[test]
fn the_volume_slice_stride_follows_the_booked_bytes() {
    let f = |index| texel_format(index).unwrap();

    assert_eq!(volume_slice_stride(1, f(17)), 0x80);
    assert_eq!(volume_slice_stride(4, f(17)), 0x400);
    assert_eq!(volume_slice_stride(6, f(17)), 0x1000);
    assert_eq!(volume_slice_stride(1, f(12)), 0x40);
    assert_eq!(volume_slice_stride(1, f(33)), 0x100);
}
