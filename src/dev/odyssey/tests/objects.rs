//! Texture levels in board memory: the latch takes a level where its
//! descriptor places it, the unit reads what memory holds there, and the
//! draw state slots keep a queued draw's registers.

use crate::dev::odyssey::board::Board;
use crate::dev::odyssey::consts::tex;
use crate::dev::odyssey::field::level_surface;
use crate::dev::odyssey::pipe::slot::DrawState;
use crate::dev::odyssey::pipe::texture::{Class, Mag, TexQuad, Unit, Wrap};
use crate::dev::odyssey::tests::board_with_ram;
use crate::dev::odyssey::view::{Rows, View, MAX_DIM};

/// Store texel (x, y) of the level descriptor words (word0, word1) names.
fn put(s: &Board, word0: u32, word1: u32, x: u32, y: u32, texel: u32) {
    let (surface, field) = level_surface(word0, word1).expect("a level the library emits");
    Rows::new(s.vram_ptr(), View::colour(&surface, field, MAX_DIM)).store_word(x as i32, y as i32, texel);
}

/// A packed RRGGBBAA texel as the unit's 12-bit components.
fn c12(w: u32) -> [u32; 4] {
    [24, 16, 8, 0].map(|s| {
        let v = (w >> s) & 0xff;
        (v << 4) | (v >> 4)
    })
}

/// Run f on the texture unit a draw issued now would read, then let the
/// slot go.
fn with_unit<R>(s: &mut Board, f: impl FnOnce(&Unit) -> R) -> R {
    let st = s.issue_state();
    let mem = s.vram.ptr();
    // SAFETY: issued above and released below.
    let d = unsafe { &*st };
    let r = f(&Unit { regs: &d.tex, latch: &d.latch, mem });
    unsafe { DrawState::release(st) };
    r
}

/// The texture state of a plain 2D RGBA level, as chocolate-doom's SDL
/// renderer sends it for its 512x256 frame texture: clamp to edge on both
/// axes, nearest, replace, one level at `base` (sampler word 4 0x13001400,
/// level entry 0x41100020, replayed from a guest trace).
fn rgba_512x256(s: &mut Board, base: u32) {
    let t = s.tex_mut();
    t.path = tex::MODE_2D;
    t.level0 = 0;
    t.levels[..3].copy_from_slice(&[base, 0x4110_0020, 0]);
    t.sampler = [0x0800_0000 | (9 << 10) | (8 << 5), 0x0004_0000 | 512, 0x0004_0000 | 256, 1, 0x1300_1400, 0, 0];
    t.env = [1, 0, 0, 0, 0];
    t.lod = [0; 4];
}

/// glTexImage2D with no data and then glTexSubImage2D of a 320x200 frame,
/// SDL's way of streaming a texture: only 200 of the level's 256 rows were
/// written. The rows nobody wrote read what memory holds (0 after a reset).
#[test]
fn a_level_written_only_in_part_is_still_the_texture() {
    let (mut s, _, _) = board_with_ram();
    let base = 0x16b800;

    for y in 0..200 {
        put(&s, base, 0x4110_0020, 0, y, 0x1122_33ff);
    }
    rgba_512x256(&mut s, base);
    with_unit(&mut s, |u| {
        assert_eq!((u.latch.count, u.latch.level[0].wl, u.latch.level[0].hl), (1, 9, 8));
        assert_eq!(u.fetch(0, 0, 199, 0), c12(0x1122_33ff));
        assert_eq!(u.fetch(0, 0, 255, 0), [0; 4], "a row the client never wrote");
    });
}

/// A quad sampled NEAREST at its texel centres reads each pixel's texel, in
/// 12.12 over 4095.
#[test]
fn a_quad_samples_each_pixels_texel() {
    let (mut s, _, _) = board_with_ram();
    let base = 0x16b800;

    for (x, y, w) in [(3, 5, 0x1122_33ff), (4, 5, 0x4455_66ff), (3, 6, 0x7788_99ff), (4, 6, 0xaabb_ccff)] {
        put(&s, base, 0x4110_0020, x, y, w);
    }
    rgba_512x256(&mut s, base);
    // Texel (x, y)'s centre: s = (x + 0.5) / 512, t = (y + 0.5) / 256, with
    // q 2 to divide out.
    let at = |x: f32, y: f32| [2.0 * (x + 0.5) / 512.0, 2.0 * (y + 0.5) / 256.0, 0.0, 2.0];
    let q = TexQuad { c: [at(3.0, 5.0), at(4.0, 5.0), at(3.0, 6.0), at(4.0, 6.0)] };
    let mut out = [[0i32; 4]; 4];

    with_unit(&mut s, |u| u.sample_quad(&q, 0xf, &mut out));
    for (i, w) in [0x1122_33ffu32, 0x4455_66ff, 0x7788_99ff, 0xaabb_ccff].into_iter().enumerate() {
        assert_eq!(out[i], c12(w).map(|c| (c * 4097) as i32), "pixel {i}");
    }
}

/// A register write while a queued draw holds the current slot goes to a
/// copy: the draw keeps the registers it was issued under.
#[test]
fn a_write_after_a_draw_copies_the_slot() {
    let (mut s, _, _) = board_with_ram();

    s.tex_mut().env = [2, 0, 0, 0, 0];
    let held = s.issue_state();
    s.tex_mut().env = [4, 0, 0, 0, 0];
    // SAFETY: issued above and released below.
    assert_eq!(unsafe { (*held).tex.env[0] }, 2, "the queued draw's state");
    assert_eq!(s.tex_regs().env[0], 4);
    assert!(!std::ptr::eq(held, s.slots.cur()), "the write went to another slot");
    unsafe { DrawState::release(held) };
    // A free slot is written in place.
    let cur: *const DrawState = s.slots.cur();
    s.tex_mut().env = [3, 0, 0, 0, 0];
    assert!(std::ptr::eq(cur, s.slots.cur()));
}

/// SGI Quake's world textures as the library left them: a 512x256 RGB base
/// with ten mip levels, sampler word 4 `word4`, modulate, and a 128x128
/// detail image in entry 60 whose word 1 is `detail_word1`.
fn quake_world_state(s: &mut Board, word4: u32, detail_word1: u32) {
    let base = 0x1d90c0;
    let mut level_base = base;
    let detail = 0x16b800;

    for y in 0..128 {
        for x in 0..128 {
            put(s, detail, detail_word1, x, y, 0x8080_80ff);
        }
    }
    let t = s.tex_mut();
    t.path = 0x2410;
    t.level0 = 0;
    for i in 0..10u32 {
        let (w, h) = ((512u32 >> i).max(1), (256u32 >> i).max(1));
        let units = (w.div_ceil(16) + 1) & !1;
        t.levels[i as usize * 3..][..3].copy_from_slice(&[level_base, 0x00d0_0000 | units, 0]);
        // 2 bytes a texel (index 13), 16 rows a tile row: in 16-byte units.
        level_base += units * 256 * 2 * h.div_ceil(16) / 16;
    }
    t.levels[60 * 3..][..3].copy_from_slice(&[detail, detail_word1, 0]);
    t.sampler = [0x0873_a500, 0x200, 0x100, 1, word4, 0, 0];
    t.lod = [0, 0, 0x3100, 0xf00];
    t.env = [2, 0, 0, 0, 0];
    for (i, e) in t.detail.iter_mut().enumerate() {
        *e = ((i as u32 * 32).min(4095)) << 12;
    }
}

/// SGI Quake's world textures, replayed from its texture state: a 512x256
/// RGB base with ten mip levels under GL_LINEAR_DETAIL_COLOR_SGIS (sampler
/// word 4 0x128000a8: minification code 5, magnification code 4), and the
/// 128x128 detail image at the default level -4, entry 60, added in.
#[test]
fn sgi_quake_world_texture_takes_its_detail_image() {
    let (mut s, _, _) = board_with_ram();
    quake_world_state(&mut s, 0x1280_00a8, 0x00d0_0008);

    assert_eq!((s.tex_regs().mag(), s.tex_regs().class(), s.tex_regs().log2()), (Mag::DetailColour, Class::Rgb, [9, 8, 0]));
    with_unit(&mut s, |u| {
        let l = u.latch;
        assert_eq!(l.count, 10);
        assert_eq!((l.detail.addr, l.detail.wl, l.detail.hl), (0x16b800 * 16, 7, 7));
        // 2^4 times as fine as the 512x256 base, in 1/256 of its texels.
        assert_eq!(l.detail_scale, [512.0 * 16.0 * 256.0, 256.0 * 16.0 * 256.0]);
        assert_eq!(u.regs.sampler[0] & 0x8000_0000, 0, "added");
    });
}

/// The library's default ramp, weight 0 at LOD 0 to 1 at LOD -4, 12.12:
/// the end entry's beyond the table and linear between entries.
#[test]
fn the_detail_weight_follows_the_function_between_entries() {
    let (mut s, _, _) = board_with_ram();
    for (i, e) in s.tex_mut().detail.iter_mut().enumerate() {
        *e = ((i as u32 * 32).min(4095)) << 12;
    }
    let t = s.tex_regs();
    assert_eq!(t.detail_weight(128), 0);
    assert_eq!(t.detail_weight(-256), 1024 << 12);
    assert_eq!(t.detail_weight(-4), 16 << 12);
    assert_eq!(t.detail_weight(-640), 2560 << 12);
    assert_eq!(t.detail_weight(-9 * 256), 4095 << 12);
}

/// A 1D image of any width stored with a texel offset: its texels 16 in
/// and its two border texels at 15 and just past its end.
#[test]
fn a_one_dimensional_stored_border_of_any_width_is_read() {
    for width in [16u32, 32] {
        let (mut s, _, _) = board_with_ram();
        let base = 0x16b800;
        let units = ((width + 32).div_ceil(16) + 1) & !1;
        for x in 15..=16 + width {
            put(&s, base, 0x4110_0000 | units, x, 0, 0x1020_30ff + x);
        }
        let t = s.tex_mut();
        t.path = tex::MODE_1D;
        t.level0 = 0;
        t.levels[..3].copy_from_slice(&[base, 0x4110_0000 | units, 0]);
        t.sampler = [0x0800_0000 | (width.trailing_zeros() << 10), width, 1, 1, 0x0300_2000, 0, 0x0100_0000];
        t.env = [1, 0, 0, 0, 0];

        assert!(s.tex_regs().stored_border());
        with_unit(&mut s, |u| {
            assert_eq!(u.fetch(0, -1, 0, 0), c12(0x1020_30ff + 15), "width {width}");
            assert_eq!(u.fetch(0, width as i64, 0, 0), c12(0x1020_30ff + 16 + width));
            assert_eq!(u.fetch(0, 0, 0, 0), c12(0x1020_30ff + 16));
        });
    }
}

/// The captured stored-border volume: a 2x2x2 RGBA interior with its border,
/// level 0 at 0x16a800 with slices 0x400 apart, its four slice objects
/// uploaded from 0x16e400, fifteen slice strides up, each 64 texels wide
/// with the interior from (16, 16). At the other observed level base,
/// 0x16b800, the same gap.
#[test]
fn the_captured_stored_volume_reads_its_slices_fifteen_strides_up() {
    for (level_base, first_slice) in [(0x16a800u32, 0x16e400u32), (0x16b800, 0x16f400)] {
        let (mut s, _, _) = board_with_ram();
        for z in 0..4u32 {
            for y in 15..19 {
                for x in 15..19 {
                    put(&s, first_slice + z * 0x400, 0x4110_0004, x, y, (z << 24) | (y << 16) | (x << 8) | 0xff);
                }
            }
        }
        let t = s.tex_mut();
        t.path = 0x2420;
        t.level0 = 0;
        t.levels[..3].copy_from_slice(&[level_base, 0x4110_0004, 4]);
        t.sampler = [0x0800_0421, 2, 2, 2, 0x0300_2a00, 0x404, 0x01c0_cfff];
        t.env = [1, 0, 0, 0, 0];

        assert_eq!(s.tex_regs().wrap(2), Wrap::Clamp);
        // Interior texel (0, 0, 0) is slice 1 at (16, 16); the border slice
        // and corner are slice 0 at (15, 15).
        with_unit(&mut s, |u| {
            assert_eq!(u.fetch(0, 0, 0, 0), c12((1 << 24) | (16 << 16) | (16 << 8) | 0xff), "level base {level_base:#x}");
            assert_eq!(u.fetch(0, -1, -1, -1), c12((15 << 16) | (15 << 8) | 0xff));
        });
    }
}

/// A volume's slices sit 2^(word 2) tiles of 256 texels apart at the bytes
/// the size code books: 0x80 16-byte units for a 4-byte format at word 2
/// 1, as the captured 8x8x8 levels sit, and half or twice that for a 2- or
/// 8-byte format.
#[test]
fn the_volume_slice_stride_follows_the_booked_bytes() {
    for (word1, word2, units) in [(0x4110_0002u32, 1u32, 0x80u64), (0x4110_0002, 4, 0x400), (0x4110_0002, 6, 0x1000), (0x00c0_0002, 1, 0x40), (0x8210_0002, 1, 0x100)] {
        let (mut s, _, _) = board_with_ram();
        let t = s.tex_mut();
        t.path = tex::MODE_3D;
        t.levels[..3].copy_from_slice(&[0x10_0000, word1, word2]);
        t.sampler = [(3 << 10) | (3 << 5) | 3, 8, 8, 8, 0x0300_0000, 0, 0];
        with_unit(&mut s, |u| assert_eq!(u.latch.level[0].slice, units * 16, "word 1 {word1:#x} word 2 {word2}"));
    }
}
