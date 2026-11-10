//! Golden scenes: GL triangles and clears through set-up, the tile scheduler and the
//! per-pixel module, against pinned hashes, and a tiled run against an inline one.

use super::raster::ops::{BlendEq, BlendFactor, Compare, LogicOp, Ownership, StencilOp};
use super::store::DID_PLANE;
use super::raster::{ClearReq, DrawState, Prims, Rect, WVertex};
use super::store::PixelStore;
use super::KonaBackend;
use crate::dev::kona::KonaConfig;

const N: u32 = 64;

/// A board with an `N` x `N` managed area, `pool.0` tile workers and `pool.1`-pixel tiles.
fn board(pool: (usize, u32)) -> KonaBackend {
    KonaBackend::new(KonaConfig { managed: (N, N), ..KonaConfig::default() }.with_pool(pool.0, pool.1))
}

/// A draw state drawing into word 1 (and 2 for the back buffer), depth in
/// word 3, over the whole 64 x 64 screen.
fn state(f: impl FnOnce(&mut DrawState)) -> DrawState {
    let full = Rect { x0: 0, y0: 0, x1: N as i32, y1: N as i32 };
    let mut d = DrawState { clip: full, viewport: full, colour: [1, 0], colours: 1, depth_word: 3, depth_bits: 32, depth_func: Compare::Less, ..DrawState::default() };
    f(&mut d);
    d
}

fn v(x: f32, y: f32, c: [u16; 4], z: u32) -> WVertex {
    WVertex { x: (x * 256.0) as i32, y: (y * 256.0) as i32, z, c, t: [[0.0; 4]; 2], d: 0.0, fog: [0.0; 2] }
}

const WHITE: [u16; 4] = [8191; 4];

/// A triangle as GL window coordinates.
fn tri(p: [(f32, f32); 3], c: [[u16; 4]; 3], z: [u32; 3]) -> [WVertex; 3] {
    [v(p[0].0, p[0].1, c[0], z[0]), v(p[1].0, p[1].1, c[1], z[1]), v(p[2].0, p[2].1, c[2], z[2])]
}

/// Flush `tris` under `ds` through set-up and the tiles.
fn draw(st: &mut KonaBackend, ds: &DrawState, tris: &[[WVertex; 3]]) {
    let p = Prims { tris: tris.to_vec(), clears: Vec::new(), state: Some(*ds), ..Default::default() };
    st.prims = p;
    st.flush();
}

fn clear(st: &mut KonaBackend, ds: &DrawState, colour: Option<u32>, depth: Option<u32>) {
    let p = Prims { tris: Vec::new(), clears: vec![ClearReq { rect: ds.clip, colour, depth, ..Default::default() }], state: Some(*ds), ..Default::default() };
    st.prims = p;
    st.flush();
}

fn pixels(s: &PixelStore, word: usize) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    for y in 0..N {
        for x in 0..N {
            let w = s.read(x, y, word);
            if w != 0 {
                out.push((x, y, w));
            }
        }
    }
    out
}

fn hash(s: &PixelStore) -> u64 {
    s.checksum()
}

/// A GL unit square lights exactly the pixel GL specifies, in two triangles
/// sharing a diagonal, with no gap and no pixel twice.
#[test]
fn a_gl_unit_square_lights_one_pixel_and_a_quad_is_watertight() {
    let mut st = board((0, 32));
    let s = st.store.clone();
    let ds = state(|_| {});
    let quad = |x0: f32, y0: f32, x1: f32, y1: f32| vec![tri([(x0, y0), (x1, y0), (x1, y1)], [[8191; 4]; 3], [0; 3]), tri([(x0, y0), (x1, y1), (x0, y1)], [[8191; 4]; 3], [0; 3])];
    draw(&mut st, &ds, &quad(3.0, 4.0, 4.0, 5.0));
    assert_eq!(pixels(&s, 1).iter().map(|&(x, y, _)| (x, y)).collect::<Vec<_>>(), [(3, 4)]);
    let mut st = board((0, 32));
    let s = st.store.clone();
    draw(&mut st, &ds, &quad(10.0, 20.0, 18.0, 26.0));
    let lit = pixels(&s, 1);
    assert_eq!(lit.len(), 8 * 6);
    assert!(lit.iter().all(|&(x, y, _)| (10..18).contains(&x) && (20..26).contains(&y)));
    // A quad of odd corners, cut by a diagonal that passes through samples, still has no gap.
    let mut st = board((0, 32));
    let s = st.store.clone();
    draw(&mut st, &ds, &quad(0.0, 0.0, 17.0, 17.0));
    assert_eq!(pixels(&s, 1).len(), 17 * 17);
}

/// Colour follows the plane from the vertices; the word is `0xAARRGGBB`.
#[test]
fn colour_is_interpolated_across_the_triangle() {
    let mut st = board((0, 32));
    let s = st.store.clone();
    let (r, g, b) = ([8191, 0, 0, 8191], [0, 8191, 0, 8191], [0, 0, 8191, 8191]);
    draw(&mut st, &state(|_| {}), &[tri([(0.5, 0.5), (32.5, 0.5), (0.5, 32.5)], [r, g, b], [0; 3])]);
    let at = |x, y| s.read(x, y, 1);
    // Next to the red vertex it is nearly red; on the bottom edge's middle it is half red, half green.
    assert_eq!(at(1, 1) >> 24, 0xff);
    assert!((at(1, 1) >> 16) & 0xff > 0xe0);
    let mid = at(16, 1);
    assert!(((mid >> 16) & 0xff).abs_diff((mid >> 8) & 0xff) < 12 && (mid >> 16) & 0xff > 100, "{mid:#x}");
    assert_eq!(at(40, 40), 0, "outside");
}

/// A nearer triangle wins where two cross, whichever was drawn first; with the
/// test off the later one wins.
#[test]
fn depth_decides_which_triangle_shows() {
    let near = 0x1000_0000;
    let far = 0xc000_0000;
    let red = [8191, 0, 0, 8191];
    let blue = [0, 0, 8191, 8191];
    let a = tri([(0.5, 0.5), (40.5, 0.5), (0.5, 40.5)], [red; 3], [near; 3]);
    let b = tri([(0.5, 0.5), (40.5, 0.5), (0.5, 40.5)], [blue; 3], [far; 3]);
    let test = state(|d| {
        d.depth_test = true;
        d.depth_func = Compare::Less;
    });
    for first_near in [true, false] {
        let mut st = board((0, 32));
        let s = st.store.clone();
        clear(&mut st, &test, None, Some(0xffff_ff00));
        if first_near {
            draw(&mut st, &test, &[a]);
            draw(&mut st, &test, &[b]);
        } else {
            draw(&mut st, &test, &[b]);
            draw(&mut st, &test, &[a]);
        }
        assert_eq!(s.read(5, 5, 1) & 0xffffff, 0xff0000, "near is red, drawn {} first", if first_near { "near" } else { "far" });
    }
    let mut st = board((0, 32));
    let s = st.store.clone();
    let off = state(|_| {});
    draw(&mut st, &off, &[a]);
    draw(&mut st, &off, &[b]);
    assert_eq!(s.read(5, 5, 1) & 0xffffff, 0x0000ff, "no test: the later wins");
    assert_eq!(s.read(5, 5, 3), 0, "and no depth is written");
}

/// The scissor, in window coordinates, narrows a draw and a clear.
#[test]
fn the_scissor_narrows_draws_and_clears() {
    let mut st = board((0, 32));
    let s = st.store.clone();
    // The GL path puts the scissor into the clip rectangle when it builds the draw state.
    let scissored = state(|d| d.clip = Rect { x0: 10, y0: 10, x1: 15, y1: 15 });
    draw(&mut st, &scissored, &[tri([(0.5, 0.5), (60.5, 0.5), (0.5, 60.5)], [WHITE; 3], [0; 3])]);
    let lit = pixels(&s, 1);
    assert!(!lit.is_empty() && lit.iter().all(|&(x, y, _)| (10..15).contains(&x) && (10..15).contains(&y)));
    clear(&mut st, &scissored, Some(0x1122_3344), None);
    let lit = pixels(&s, 1);
    assert_eq!(lit.len(), 25);
    assert!(lit.iter().all(|&(_, _, w)| w == 0x1122_3344));
}

/// A clear writes the colour word of every buffer drawn to and the depth word.
#[test]
fn a_clear_writes_colour_and_depth_to_the_targets() {
    let mut st = board((0, 32));
    let s = st.store.clone();
    let both = state(|d| {
        d.colour = [1, 2];
        d.colours = 2;
    });
    clear(&mut st, &both, Some(0xff00_ff00), Some(0xffff_ff00));
    assert_eq!((s.read(0, 0, 1), s.read(63, 63, 2), s.read(30, 30, 3)), (0xff00_ff00, 0xff00_ff00, 0xffff_ff00));
    // No target: no colour, no write at all.
    let none = state(|d| d.colours = 0);
    clear(&mut st, &none, Some(0xabab_abab), None);
    assert_eq!(s.read(0, 0, 1), 0xff00_ff00);
    assert_eq!(st.raster.off_target, 1);
}

/// The window's origin moves a draw and cuts it at the window's edge.
#[test]
fn the_window_origin_places_the_draw_and_bounds_it() {
    let mut st = board((0, 32));
    let s = st.store.clone();
    let win = state(|d| {
        d.origin = [20, 30];
        d.clip = Rect { x0: 20, y0: 30, x1: 25, y1: 40 };
    });
    draw(&mut st, &win, &[tri([(0.5, 0.5), (30.5, 0.5), (0.5, 30.5)], [WHITE; 3], [0; 3])]);
    let lit = pixels(&s, 1);
    assert!(lit.iter().all(|&(x, y, _)| (20..25).contains(&x) && (30..40).contains(&y)));
    // The triangle's own left and bottom edges lie on window column 0 and row 0: those samples are out.
    assert_eq!(lit.len(), 4 * 9);
}

/// The store hash of the 150-triangle scene below, drawn inline: a regression net
/// for set-up and the per-pixel module that moves only if they do (or the scene).
/// Re-pinned when depth became a 32.16 integer plane: a depth that rounded one way
/// in `f32` now rounds to the plane's own integer, so a few depth-test results moved.
const SCENE_150: u64 = 0xa949_d8e0_9bb7_4571;

/// A tiled run, with workers, small tiles and a small ring, is bit-exact with
/// the inline one, depth test and overlap included.
#[test]
fn a_tiled_run_is_bit_exact_with_the_inline_one() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move |m: u32| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as u32 % m
    };
    let draws: Vec<[WVertex; 3]> = (0..150)
        .map(|_| {
            let p = |next: &mut dyn FnMut(u32) -> u32| (next(90) as f32 - 12.5, next(90) as f32 - 12.5);
            let c = |next: &mut dyn FnMut(u32) -> u32| [next(8192) as u16, next(8192) as u16, next(8192) as u16, 8191];
            let z = next(0xffff_ff00);
            tri([p(&mut next), p(&mut next), p(&mut next)], [c(&mut next), c(&mut next), c(&mut next)], [z, z.saturating_add(next(1 << 20)), z])
        })
        .collect();
    let run = |pool: (usize, u32)| {
        let mut st = board(pool);
        let s = st.store.clone();
        let ds = state(|d| {
            d.depth_test = true;
            d.depth_func = Compare::Lequal;
        });
        clear(&mut st, &ds, Some(0xff10_2030), Some(0xffff_ff00));
        for d in &draws {
            draw(&mut st, &ds, &[*d]);
        }
        st.drain();
        assert!(!st.is_poisoned());
        hash(&s)
    };
    let inline = run((0, 32));
    assert_eq!(inline, run((0, 32)), "the inline run is deterministic");
    assert_eq!(inline, SCENE_150, "the pinned scene hash: {inline:#018x}");
    for pool in [(1, 32), (3, 16), (4, 64), (2, 8)] {
        assert_eq!(run(pool), inline, "pool {pool:?}");
    }
}

/// A tile size that is not a power of two is refused and the default used: the
/// draw still lands.
#[test]
fn a_bad_tile_size_falls_back_to_the_default() {
    let mut st = board((1, 24));
    let s = st.store.clone();
    draw(&mut st, &state(|_| {}), &[tri([(0.5, 0.5), (20.5, 0.5), (0.5, 20.5)], [WHITE; 3], [0; 3])]);
    st.drain();
    assert!(!pixels(&s, 1).is_empty());
}

/// A triangle covering pixel (5, 5) and a good area about it, with colour `c` and depth `z`.
fn big(c: [u16; 4], z: u32) -> [WVertex; 3] {
    tri([(0.5, 0.5), (40.5, 0.5), (0.5, 40.5)], [c; 3], [z; 3])
}

const RED: [u16; 4] = [8191, 0, 0, 8191];

/// Fill colour word 1 with `word` over the whole screen.
fn fill(st: &mut KonaBackend, ds: &DrawState, word: u32) {
    clear(st, ds, Some(word), None);
}

/// The alpha test drops a fragment whose alpha fails the comparison against the
/// reference, in 13-bit units.
#[test]
fn the_alpha_test_drops_failing_fragments() {
    let half = [8191, 8191, 8191, 4096];
    for (func, reference, drawn) in [(Compare::Greater, 6000, false), (Compare::Less, 6000, true), (Compare::Gequal, 4096, true), (Compare::Never, 0, false)] {
        let mut st = board((0, 32));
        let s = st.store.clone();
        let ds = state(|d| d.alpha = super::raster::AlphaTest { enabled: true, func, reference });
        draw(&mut st, &ds, &[big(half, 0)]);
        assert_eq!(s.read(5, 5, 1) != 0, drawn, "{func:?} against {reference}");
    }
    // Off, the same fragment draws whatever the function says.
    let mut st = board((0, 32));
    let s = st.store.clone();
    draw(&mut st, &state(|d| d.alpha = super::raster::AlphaTest { enabled: false, func: Compare::Never, reference: 0 }), &[big(half, 0)]);
    assert_ne!(s.read(5, 5, 1), 0);
}

/// A depth mask of off tests depth without writing it; colour is written.
#[test]
fn the_depth_mask_keeps_the_depth_plane() {
    for (mask, depth) in [(true, 0x1000_0000u32), (false, 0xffff_ff00)] {
        let mut st = board((0, 32));
        let s = st.store.clone();
        let ds = state(|d| {
            d.depth_test = true;
            d.depth_func = Compare::Less;
            d.depth_mask = mask;
        });
        clear(&mut st, &ds, None, Some(0xffff_ff00));
        draw(&mut st, &ds, &[big(RED, 0x1000_0000)]);
        assert_eq!(s.read(5, 5, 3), depth, "depth mask {mask}");
        assert_ne!(s.read(5, 5, 1), 0, "the colour is written either way");
    }
}

/// The stencil test, each outcome's operation and the write mask: a clear to 1; a
/// test of equal to 1 passes and increments on depth pass; a failing test applies
/// its own operation; a depth fail its own; the write mask keeps the other bits.
#[test]
fn the_stencil_test_and_its_operations() {
    let stencil = |f: &dyn Fn(&mut super::raster::StencilState)| {
        state(|d| {
            (d.stencil_word, d.stencil_bits) = (4, 8);
            d.stencil.enabled = true;
            f(&mut d.stencil);
        })
    };
    let mut st = board((0, 32));
    let s = st.store.clone();
    let on = stencil(&|t| {
        t.func = Compare::Equal;
        t.reference = 1;
        t.zpass = StencilOp::Incr;
        t.fail = StencilOp::Zero;
    });
    let mut p = Prims { clears: vec![ClearReq { rect: on.clip, stencil: Some(1), ..Default::default() }], state: Some(on), ..Default::default() };
    st.prims = std::mem::take(&mut p);
    st.flush();
    draw(&mut st, &on, &[big(RED, 0)]);
    assert_eq!(s.read(5, 5, 4), 2, "passed the test, depth always passes: incremented");
    assert_ne!(s.read(5, 5, 1), 0, "and drawn");
    // The value is now 2: equal-to-1 fails, the fail operation zeroes it, and nothing is drawn.
    let before = s.read(5, 5, 1);
    draw(&mut st, &on, &[big([0, 8191, 0, 8191], 0)]);
    assert_eq!((s.read(5, 5, 4), s.read(5, 5, 1)), (0, before), "failed: zeroed, not drawn");
    // A depth failure takes the zfail operation, and the write mask keeps the other bits.
    let mut st = board((0, 32));
    let s = st.store.clone();
    let deep = stencil(&|t| {
        t.zfail = StencilOp::Replace;
        t.reference = 0xf5;
        t.write_mask = 0x0f;
    });
    let deep = DrawState { depth_test: true, depth_func: Compare::Less, ..deep };
    let mut p = Prims { clears: vec![ClearReq { rect: deep.clip, depth: Some(0x1000_0000), stencil: Some(0xa0), stencil_mask: 0xff, ..Default::default() }], state: Some(deep), ..Default::default() };
    st.prims = std::mem::take(&mut p);
    st.flush();
    draw(&mut st, &deep, &[big(RED, 0x2000_0000)]);
    assert_eq!(s.read(5, 5, 4), 0xa5, "the depth test failed: replaced under the write mask 0x0f, the high nibble kept");
    assert_eq!(s.read(5, 5, 1), 0, "and nothing drawn");
}

/// Source-over blending against a cleared buffer, and the constant colour.
#[test]
fn blending_mixes_with_the_buffer() {
    let mut st = board((0, 32));
    let s = st.store.clone();
    let ds = state(|d| {
        d.blend = super::raster::BlendState { enabled: true, src: BlendFactor::SrcAlpha, dst: BlendFactor::OneMinusSrcAlpha, eq: BlendEq::Add, constant: [0; 4] };
    });
    fill(&mut st, &ds, 0xff00_00ff); // opaque blue
    draw(&mut st, &ds, &[big([8191, 0, 0, 4096], 0)]);
    let w = s.read(5, 5, 1);
    // Red at half alpha over blue: red about 0x80, blue about 0x7f, alpha the blend of 0.5 and 1.
    assert_eq!((w >> 16 & 0xff, w & 0xff), (0x80, 0x7f), "{w:#x}");
    assert_eq!(s.read(60, 60, 1), 0xff00_00ff, "outside the triangle the buffer is as filled");
    // A constant colour of (0, 1, 0) takes the destination's green and nothing else.
    let ds = state(|d| {
        d.blend = super::raster::BlendState { enabled: true, src: BlendFactor::ConstantColour, dst: BlendFactor::Zero, eq: BlendEq::Add, constant: [0, 8191, 0, 8191] };
    });
    draw(&mut st, &ds, &[big([8191, 8191, 8191, 8191], 0)]);
    assert_eq!(s.read(5, 5, 1), 0xff00_ff00, "white times a constant of (0, 1, 0, 1)");
}

/// A triangle with fog coordinates `c` (eye distance) at its corners and w of 1 there, so a vertex's
/// words are (c, 1).
fn fogged(c: [f32; 3]) -> [WVertex; 3] {
    let mut t = big(WHITE, 0);
    for (v, c) in t.iter_mut().zip(c) {
        v.fog = [c, 1.0];
    }
    t
}

/// Fog blends toward the fog colour by the factor of the fragment's coordinate; with fog off the
/// fragment is untouched. The fragment is white, the fog colour black.
#[test]
fn fog_darkens_a_fragment_by_its_distance() {
    use super::raster::fog::Fog;
    use crate::dev::kona::defs::gl;
    let red_of = |s: &PixelStore, x, y| s.read(x, y, 1) >> 16 & 0xff;
    // Constant coordinate 5, linear from 0 to 10: half way.
    let on = state(|d| d.fog = Fog::new(true, gl::LINEAR, 1.0, 0.0, 10.0, [0.0; 3]));
    let mut st = board((0, 32));
    let s = st.store.clone();
    draw(&mut st, &on, &[fogged([5.0; 3])]);
    assert_eq!(s.read(5, 5, 1), 0xff80_8080, "half way to black, alpha untouched");
    let mut st = board((0, 32));
    let s = st.store.clone();
    draw(&mut st, &state(|_| {}), &[fogged([5.0; 3])]);
    assert_eq!(s.read(5, 5, 1), 0xffff_ffff, "fog off: untouched");
    // The coordinate is interpolated across the triangle: 0 at the corner (0.5, 0.5), 20 at the others,
    // so the fragment darkens with distance from the first corner and is black past an end of 10.
    let mut st = board((0, 32));
    let s = st.store.clone();
    draw(&mut st, &on, &[fogged([0.0, 20.0, 20.0])]);
    assert!(red_of(&s, 1, 1) > 0xe0 && red_of(&s, 8, 1) < red_of(&s, 4, 1) && red_of(&s, 4, 1) < red_of(&s, 2, 1), "darker with distance");
    assert_eq!(red_of(&s, 30, 5), 0, "past the end it is all fog");
    // The exponential curves, density 0.2 at distance 5: exp(-1) and exp(-1) again.
    for mode in [gl::EXP, gl::EXP2] {
        let ds = state(|d| d.fog = Fog::new(true, mode, 0.2, 0.0, 1.0, [0.0; 3]));
        let mut st = board((0, 32));
        let s = st.store.clone();
        draw(&mut st, &ds, &[fogged([5.0; 3])]);
        assert_eq!(red_of(&s, 5, 5), 94, "255 exp(-1) = 93.8, mode {mode:#x}");
    }
    // The fog colour replaces the fragment's colour where the factor is 0.
    let blue = state(|d| d.fog = Fog::new(true, gl::LINEAR, 1.0, 0.0, 10.0, [0.0, 0.0, 1.0]));
    let mut st = board((0, 32));
    let s = st.store.clone();
    draw(&mut st, &blue, &[fogged([50.0; 3])]);
    assert_eq!(s.read(5, 5, 1), 0xff00_00ff);
}

/// The coordinate is perspective-correct: it is the quotient of two screen-linear words, so a vertex of
/// larger w pulls the coordinate toward the others, not along the screen.
#[test]
fn the_fog_coordinate_is_interpolated_perspective_correct() {
    use super::raster::fog::Fog;
    use crate::dev::kona::defs::gl;
    let ds = state(|d| d.fog = Fog::new(true, gl::LINEAR, 1.0, 0.0, 100.0, [0.0; 3]));
    let mut st = board((0, 32));
    let s = st.store.clone();
    // Corner 0 at distance 1 (w 1), the others at distance 9 (w 9): words (c / w, 1 / w).
    let mut t = big(WHITE, 0);
    t[0].fog = [1.0, 1.0];
    t[1].fog = [1.0, 1.0 / 9.0];
    t[2].fog = [1.0, 1.0 / 9.0];
    draw(&mut st, &ds, &[t]);
    // At pixel (20, 1) the weights are .475 on the near corner and .525 on the far two: 1/w is .533 and
    // c/w is 1, so the distance is 1.875 and linear fog over 0..100 leaves 98.1 percent (250 of 255).
    // A screen-linear coordinate would be 5.2 and leave 94.8 percent (242).
    let mid = (s.read(20, 1, 1) >> 16 & 0xff) as i32;
    assert!((mid - 250).abs() <= 2, "{mid}");
}

/// A logic operation combines the written word with the buffer's, and a colour
/// mask keeps the masked components of the buffer.
#[test]
fn a_logic_operation_and_a_colour_mask_write_what_they_allow() {
    let mut st = board((0, 32));
    let s = st.store.clone();
    let xor = state(|d| d.logic = Some(LogicOp::Xor));
    fill(&mut st, &xor, 0xff12_3456);
    draw(&mut st, &xor, &[big([8191, 8191, 8191, 8191], 0)]);
    assert_eq!(s.read(5, 5, 1), 0xffff_ffff ^ 0xff12_3456, "white xor the buffer");
    let masked = state(|d| d.colour_mask = [true, false, false, true]);
    fill(&mut st, &masked, 0x0011_2233);
    draw(&mut st, &masked, &[big(RED, 0)]);
    assert_eq!(s.read(5, 5, 1), 0xffff_2233 & !0x0000_ff00 & !0x0000_00ff | 0x0000_2233 & 0xffff, "red and alpha written, green and blue kept");
}

/// A clear writes only what its masks allow.
#[test]
fn a_clear_obeys_the_write_masks() {
    let mut st = board((0, 32));
    let s = st.store.clone();
    let ds = state(|d| (d.stencil_word, d.stencil_bits) = (4, 8));
    fill(&mut st, &ds, 0x0011_2233);
    let p = Prims { clears: vec![ClearReq { rect: ds.clip, colour: Some(0xffaa_bbcc), colour_mask: [true, false, false, true], stencil: Some(0xff), stencil_mask: 0x0f, ..Default::default() }], state: Some(ds), ..Default::default() };
    st.prims = p;
    st.flush();
    assert_eq!(s.read(3, 3, 1), 0xffaa_2233, "alpha and red cleared, green and blue kept");
    assert_eq!(s.read(3, 3, 4), 0x0f, "the stencil's low nibble");
    // A depth mask of off skips the depth clear.
    let ds = state(|_| {});
    let p = Prims { clears: vec![ClearReq { rect: ds.clip, depth: Some(0x1234_5600), depth_mask: false, ..Default::default() }], state: Some(ds), ..Default::default() };
    st.prims = p;
    st.flush();
    assert_eq!(s.read(3, 3, 3), 0);
}

/// A board whose ID plane holds ID 5 on the left half of the screen and ID 6 on the right.
fn two_windows() -> KonaBackend {
    let st = board((0, 32));
    st.store.ensure(DID_PLANE);
    for y in 0..N {
        for x in 0..N {
            st.store.write(x, y, DID_PLANE, (if x < N / 2 { 5 } else { 6 }) << 2);
        }
    }
    st
}

/// With the enable set, a draw writes only the pixels its window ID owns: ID 5 of
/// ordinary window 5 across a triangle that spans pixels holding 5 and 6.
#[test]
fn ownership_drops_the_fragments_of_other_windows() {
    let mut st = two_windows();
    let s = st.store.clone();
    let ds = state(|d| d.ownership = Some(Ownership::encode(5).0));
    draw(&mut st, &ds, &[big(RED, 0)]);
    let lit = pixels(&s, 1);
    assert!(!lit.is_empty() && lit.iter().all(|&(x, _, _)| x < N / 2), "only ID 5's half");
    // The same triangle with the test off covers both halves.
    let mut st = two_windows();
    let s = st.store.clone();
    draw(&mut st, &state(|_| {}), &[big(RED, 0)]);
    assert!(pixels(&s, 1).iter().any(|&(x, _, _)| x >= N / 2));
    // A mask of 0 compares no bit, so every pixel passes (the polarity the model infers).
    let mut st = two_windows();
    let s = st.store.clone();
    let open = state(|d| d.ownership = Some(Ownership { reference: 5, mask: 0 }));
    draw(&mut st, &open, &[big(RED, 0)]);
    assert!(pixels(&s, 1).iter().any(|&(x, _, _)| x >= N / 2));
}

/// The ID plane holds the painted value, the main ID in bits 7:2 and an overlay group in bits 12:8, so a window
/// in the group range owns the pixels painted with its group, whatever main ID is beneath: group 4 is the
/// middle band of `two_windows` here.
#[test]
fn a_group_range_window_owns_the_pixels_painted_with_its_group() {
    let mut st = two_windows();
    for y in 0..N {
        for x in N / 4..3 * N / 4 {
            let v = st.store.read(x, y, DID_PLANE);
            st.store.write(x, y, DID_PLANE, v | 4 << 8);
        }
    }
    let s = st.store.clone();
    draw(&mut st, &state(|d| d.ownership = Some(Ownership::encode(0x44).0)), &[big(RED, 0)]);
    let lit = pixels(&s, 1);
    assert!(!lit.is_empty() && lit.iter().all(|&(x, _, _)| (N / 4..3 * N / 4).contains(&x)), "group 4's band, across IDs 5 and 6");
    let mut st2 = two_windows();
    let s2 = st2.store.clone();
    draw(&mut st2, &state(|d| d.ownership = Some(Ownership::encode(0x40).0)), &[big(RED, 0)]);
    assert!(pixels(&s2, 1).iter().any(|&(x, _, _)| x < N / 2) && pixels(&s2, 1).iter().any(|&(x, _, _)| x >= N / 2), "group 0 is every pixel with no overlay painted");
}

/// A window ID of all ones owns no pixel while the test is on, for draws and clears.
#[test]
fn an_all_ones_window_id_writes_nothing() {
    let mut st = two_windows();
    let s = st.store.clone();
    let ds = state(|d| d.ownership = Some(Ownership::encode(-1).0));
    draw(&mut st, &ds, &[big(RED, 0)]);
    clear(&mut st, &ds, Some(0xff11_2233), None);
    assert!(pixels(&s, 1).is_empty());
}

/// A GL clear obeys the test too, per pixel, for colour, depth and stencil.
#[test]
fn a_clear_writes_only_the_pixels_the_window_owns() {
    let mut st = two_windows();
    let s = st.store.clone();
    let ds = state(|d| {
        d.ownership = Some(Ownership::encode(6).0);
        (d.stencil_word, d.stencil_bits) = (4, 8);
    });
    let p = Prims { clears: vec![ClearReq { rect: ds.clip, colour: Some(0xff11_2233), depth: Some(0x1000_0000), stencil: Some(7), ..Default::default() }], state: Some(ds), ..Default::default() };
    st.prims = p;
    st.flush();
    assert_eq!((s.read(5, 5, 1), s.read(5, 5, 3), s.read(5, 5, 4)), (0, 0, 0), "ID 5 is not the window's");
    assert_eq!((s.read(50, 5, 1), s.read(50, 5, 3), s.read(50, 5, 4)), (0xff11_2233, 0x1000_0000, 7), "ID 6 is");
}

/// The lit pixels of triangles drawn in white each into its own store.
fn coverage(tris: &[[WVertex; 3]]) -> Vec<Vec<(u32, u32)>> {
    tris.iter()
        .map(|t| {
            let mut st = board((0, 32));
            let s = st.store.clone();
            draw(&mut st, &state(|_| {}), &[*t]);
            pixels(&s, 1).into_iter().map(|(x, y, _)| (x, y)).collect()
        })
        .collect()
}

/// Triangles that share an edge cover every pixel once: a convex polygon fanned
/// from one vertex or from another is lit the same, and no pixel is lit twice.
#[test]
fn shared_edges_are_watertight() {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f32 / (1u64 << 53) as f32
    };
    for _ in 0..200 {
        // Five corners on a circle about (32, 32), each in its own fifth of it, so the polygon is convex.
        let pts: Vec<(f32, f32)> = (0..5)
            .map(|k| {
                let a = (k as f32 + 0.15 + 0.7 * next()) * std::f32::consts::TAU / 5.0;
                (32.0 + 20.0 * a.cos(), 32.0 + 20.0 * a.sin())
            })
            .collect();
        let fan = |root: usize| -> Vec<[WVertex; 3]> {
            (1..4)
                .map(|i| {
                    let at = |j: usize| pts[(root + j) % 5];
                    tri([at(0), at(i), at(i + 1)], [[8191; 4]; 3], [0; 3])
                })
                .collect()
        };
        let mut sets: Vec<Vec<(u32, u32)>> = Vec::new();
        for root in [0, 2] {
            let parts = coverage(&fan(root));
            let mut all: Vec<(u32, u32)> = parts.iter().flatten().copied().collect();
            let n = all.len();
            all.sort();
            all.dedup();
            assert_eq!(all.len(), n, "a pixel lit by two triangles of one fan, polygon {pts:?}");
            sets.push(all);
        }
        assert_eq!(sets[0], sets[1], "the fans from two roots light different pixels, polygon {pts:?}");
    }
}

/// A triangle far larger than the screen keeps its slopes and so its colour
/// where it crosses the screen: set-up does not cut it in window space, and
/// values its planes at the visible corner.
#[test]
fn a_guard_band_triangle_keeps_its_colour_at_the_screen_edge() {
    let mut st = board((0, 32));
    let s = st.store.clone();
    let (r, g, b) = ([8191, 0, 0, 8191], [0, 8191, 0, 8191], [0, 0, 8191, 8191]);
    // Red at (-1000, -1000), green at (3000, -1000), blue at (-1000, 3000): the colour at (x, y)
    // is (1 - u - w, u, w) with u = (x + 1000) / 4000 and w = (y + 1000) / 4000.
    draw(&mut st, &state(|_| {}), &[tri([(-1000.0, -1000.0), (3000.0, -1000.0), (-1000.0, 3000.0)], [r, g, b], [0; 3])]);
    for (x, y) in [(0u32, 0u32), (63, 0), (0, 63), (63, 63), (30, 20)] {
        let word = s.read(x, y, 1);
        let (u, w) = ((x as f64 + 1000.5) / 4000.0, (y as f64 + 1000.5) / 4000.0);
        let want = [(1.0 - u - w, 16u32), (u, 8), (w, 0)];
        for (weight, shift) in want {
            let got = (word >> shift & 0xff) as f64;
            assert!((got - weight * 255.0).abs() <= 1.5, "pixel ({x}, {y}): channel {shift} is {got}, wanted {}", weight * 255.0);
        }
    }
}

/// A triangle crossing a small viewport inside a large window draws only
/// inside the viewport; a clear is cut to the clip rectangle alone.
#[test]
fn a_triangle_crossing_a_small_viewport_draws_only_inside_it() {
    let mut st = board((0, 32));
    let s = st.store.clone();
    let ds = state(|d| d.viewport = Rect { x0: 20, y0: 10, x1: 40, y1: 30 });
    draw(&mut st, &ds, &[tri([(0.5, 0.5), (200.5, 0.5), (0.5, 200.5)], [WHITE; 3], [0; 3])]);
    let lit = pixels(&s, 1);
    assert!(!lit.is_empty() && lit.iter().all(|&(x, y, _)| (20..40).contains(&x) && (10..30).contains(&y)), "{lit:?}");
    assert_eq!(lit.len(), 20 * 20, "the viewport is wholly inside the triangle");
    clear(&mut st, &ds, Some(0x0102_0304), None);
    assert_eq!(pixels(&s, 1).len(), (N * N) as usize, "the viewport does not apply to a clear");
}

/// The triangles' counts: a sliver with area and no sample, one outside the
/// clip rectangle, and one of no area.
#[test]
fn skipped_triangles_are_counted_apart() {
    let mut st = board((0, 32));
    let s = st.store.clone();
    let ds = state(|d| d.clip = Rect { x0: 0, y0: 0, x1: 10, y1: 10 });
    draw(
        &mut st,
        &ds,
        &[
            tri([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], [[0; 4]; 3], [0; 3]),
            tri([(1.625, 1.5), (1.875, 1.5), (1.625, 5.5)], [[0; 4]; 3], [0; 3]),
            tri([(30.5, 30.5), (40.5, 30.5), (30.5, 40.5)], [[0; 4]; 3], [0; 3]),
        ],
    );
    let c = st.raster;
    assert_eq!((c.triangles, c.no_area, c.no_sample, c.off_target), (3, 1, 1, 1));
}

/// The per-pixel module has no `f64`: the edge functions are `i64` and the
/// planes `f32`, so a GPU kernel could reproduce them. This reads the file's
/// text, so its limits are the text's: it finds the type's name and a float
/// literal with no `f32` suffix (which a context could turn into a double),
/// and any write other than the one `store` call or a compound atomic; it does
/// not see a write made by a helper in another module, or a double that comes
/// from a function's return type.
#[test]
fn the_per_pixel_module_has_no_f64() {
    let src = include_str!("../raster/pixel.rs");
    assert!(!src.contains(concat!("f", "64")), "pixel.rs uses a double");
    assert!(include_str!("../raster/setup.rs").contains(concat!("f", "64")), "set-up is where the doubles are");
    // Every float literal carries its type. Comments are skipped.
    for (n, line) in src.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        let b: Vec<char> = code.chars().collect();
        for i in 1..b.len().saturating_sub(1) {
            let literal = b[i] == '.' && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit();
            if literal {
                let end = (i + 1..b.len()).find(|&j| !(b[j].is_ascii_digit() || b[j] == '_' || b[j] == 'e')).unwrap_or(b.len());
                let suffix: String = b[end..].iter().take(3).collect();
                assert_eq!(suffix, "f32", "pixel.rs line {}: a float literal with no f32 suffix: {line}", n + 1);
            }
        }
    }
    // Every write is `WordArray::store`, inside an unsafe block, and no other way into the planes.
    for (n, line) in src.lines().enumerate().filter(|(_, l)| l.contains(".store(")) {
        assert!(line.contains("unsafe"), "pixel.rs line {}: a write outside an unsafe block that says why: {line}", n + 1);
    }
    // Every `unsafe` block says why it is sound.
    assert_eq!(src.matches("unsafe {").count(), src.matches("// SAFETY").count(), "an unsafe block with no SAFETY comment");
    for other in ["fetch_", "compare_exchange", "write_volatile", "ptr::write", "copy_nonoverlapping", ".swap(", ".write(", "mul_add"] {
        assert!(!src.contains(other), "pixel.rs writes through {other}");
    }
}

/// The files besides the per-pixel module that do per-fragment work keep its one-store-point property:
/// texel sampling, the sample patterns and the pixel operations write only through `WordArray::store`
/// inside an `unsafe` block that says why, and use none of the other ways into the planes. The two
/// that are float kernels (`texel.rs`, `multisample.rs`) also have no `f64`. `gl/pixels.rs` does: its
/// unpacking and packing of host images runs once a transfer, on `kona-hip` and not on a tile
/// worker, and takes 32-bit integer components and the raster position's double forms through doubles.
#[test]
fn the_other_per_fragment_files_keep_the_store_point_rule() {
    for (name, src, no_f64) in [("fog.rs", include_str!("../raster/fog.rs"), true), ("smooth.rs", include_str!("../raster/smooth.rs"), true), ("texel.rs", include_str!("../raster/texel.rs"), true), ("multisample.rs", include_str!("../raster/multisample.rs"), true), ("gl/pixels.rs", include_str!("../gl/pixels.rs"), false)] {
        // Tests in the files are not the kernel.
        let src = src.split("#[cfg(test)]").next().unwrap_or(src);
        if no_f64 {
            assert!(!src.contains(concat!("f", "64")), "{name} uses a double");
        }
        for (n, line) in src.lines().enumerate().filter(|(_, l)| l.contains(".store(")) {
            assert!(line.contains("unsafe"), "{name} line {}: a write outside an unsafe block that says why: {line}", n + 1);
        }
        assert_eq!(src.matches("unsafe {").count(), src.matches("// SAFETY").count(), "{name}: an unsafe block with no SAFETY comment");
        for other in ["fetch_", "compare_exchange", "write_volatile", "ptr::write", "copy_nonoverlapping", ".swap(", ".write(", "mul_add"] {
            assert!(!src.contains(other), "{name} writes through {other}");
        }
    }
}
