//! Draw-level golden frames for the VPro rasteriser and its vertex side.
//!
//! Each scene drives `Board::gl_draw()` the way the GL context does (and the
//! fill-rate benchmark's scenes do): vertices and a `GlDrawState` into a v12
//! board's own planes, the board's texture, fog and depth registers set the
//! way the command stream sets them. No guest runs. A scene's colour, depth
//! and stencil planes over its 128 x 128 window are hashed, and the hash is
//! kept in `testdata/golden/scenes.tsv`; the planes themselves are kept as
//! PNGs in `testdata/golden/frames/`, so a deliberate change can be judged
//! by how far the frame moved, not only that it did.
//!
//! The default check is bit-exact. `VPRO_GOLDEN=tolerance` lets a scene whose
//! hash changed pass while it stays inside the limits `scenes.tsv` keeps for
//! it, measured against the reference images: for each plane, the number of
//! pixels that differ by more than the scene's per-channel delta. That is the
//! check for a change that moves numerics on purpose (the vertex side moving
//! to 32-bit floats); a change that must not move a pixel uses the default.
//!
//! `scripts/vpro-goldens.sh` wraps the commands, and
//! `src/dev/odyssey/testdata/README.md` says how to re-take the goldens.

use super::raster_bench;
use super::board::Board;
use super::draw::gl_rules;
pub mod replay;
pub mod scenes;
pub mod vertices;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};


/// A scene's window: the frame is the square from (0, 0) to SIDE - 1.
pub const SIDE: i32 = 128;

pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/dev/odyssey/testdata/golden")
}

/// The planes of a finished scene.
#[derive(Clone, PartialEq, Eq)]
pub struct Frame {
    pub colour: Vec<u32>,
    pub depth: Vec<u32>,
    pub stencil: Vec<u32>,
}

impl Frame {
    /// The planes of `b` over the scene window.
    pub fn of(b: &Board) -> Frame {
        let depth = b.planes.depth.plane(b.planes.depth_plane).expect("the depth plane");
        let stencil = b.planes.stencil.plane(b.planes.stencil_plane).expect("the stencil plane");
        let mut f = Frame { colour: vec![], depth: vec![], stencil: vec![] };
        for y in 0..SIDE {
            for x in 0..SIDE {
                f.colour.push(b.buffer_load(0, x as i64, y as i64));
                f.depth.push(b.planes.depth.plane_load(&depth, x, y));
                f.stencil.push(b.planes.stencil.plane_load(&stencil, x, y));
            }
        }
        f
    }

    /// blake3 over the three planes, big-endian words, in a fixed order.
    pub fn hash(&self) -> String {
        let mut h = blake3::Hasher::new();
        for plane in [&self.colour, &self.depth, &self.stencil] {
            for w in plane.iter() {
                h.update(&w.to_be_bytes());
            }
        }
        h.finalize().to_hex().to_string()
    }

    /// Pixels that differ from `other` in any plane.
    pub fn differing(&self, other: &Frame) -> usize {
        (0..self.colour.len())
            .filter(|&i| self.colour[i] != other.colour[i] || self.depth[i] != other.depth[i] || self.stencil[i] != other.stencil[i])
            .count()
    }
}

/// How far a frame is from a reference, plane by plane. A pixel is "bad" in
/// a plane when it differs by more than the scene's delta there.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Distance {
    pub colour_max: u32,
    pub colour_differ: usize,
    pub colour_bad: usize,
    pub depth_max: u32,
    pub depth_differ: usize,
    pub depth_bad: usize,
    pub stencil_differ: usize,
}

/// A scene's limits in tolerance mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tolerance {
    /// Colour: a channel difference above this makes a pixel bad.
    pub colour_delta: u32,
    pub colour_bad: usize,
    /// Depth: a 24-bit word difference above this makes a pixel bad.
    pub depth_delta: u32,
    pub depth_bad: usize,
    /// Stencil: pixels that differ at all.
    pub stencil_bad: usize,
}

impl Default for Tolerance {
    /// What blessing gives a new scene: a few least-significant steps of
    /// colour and a sliver of edge pixels may move.
    fn default() -> Self {
        Tolerance { colour_delta: 2, colour_bad: 64, depth_delta: 0x40, depth_bad: 64, stencil_bad: 64 }
    }
}

impl Distance {
    pub fn within(&self, t: &Tolerance) -> bool {
        self.colour_bad <= t.colour_bad && self.depth_bad <= t.depth_bad && self.stencil_differ <= t.stencil_bad
    }
}

fn channel_max(a: u32, b: u32) -> u32 {
    (0..4).map(|k| ((a >> (8 * k)) & 0xff).abs_diff((b >> (8 * k)) & 0xff)).max().unwrap()
}

pub fn distance(f: &Frame, r: &Frame, t: &Tolerance) -> Distance {
    let mut d = Distance::default();
    for i in 0..f.colour.len() {
        let c = channel_max(f.colour[i], r.colour[i]);
        d.colour_max = d.colour_max.max(c);
        d.colour_differ += (c != 0) as usize;
        d.colour_bad += (c > t.colour_delta) as usize;
        let z = (f.depth[i] & 0xff_ffff).abs_diff(r.depth[i] & 0xff_ffff);
        d.depth_max = d.depth_max.max(z);
        d.depth_differ += (z != 0) as usize;
        d.depth_bad += (z > t.depth_delta) as usize;
        d.stencil_differ += (f.stencil[i] != r.stencil[i]) as usize;
    }
    d
}

// The reference images: colour as RGBA, depth as RGB (the 24-bit word's
// bytes), stencil as grey.

fn write_png(path: &Path, w: u32, h: u32, colour: png::ColorType, data: &[u8]) {
    let file = std::fs::File::create(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(colour);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::High);
    enc.write_header().unwrap().write_image_data(data).unwrap();
}

fn read_png(path: &Path) -> Option<Vec<u8>> {
    let file = std::fs::File::open(path).ok()?;
    let mut r = png::Decoder::new(std::io::BufReader::new(file)).read_info().ok()?;
    let mut buf = vec![0; r.output_buffer_size()?];
    let info = r.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());
    Some(buf)
}

pub fn save_reference(name: &str, f: &Frame) {
    let dir = data_dir().join("frames");
    std::fs::create_dir_all(&dir).unwrap();
    let n = SIDE as u32;
    let rgba: Vec<u8> = f.colour.iter().flat_map(|w| w.to_be_bytes()).collect();
    write_png(&dir.join(format!("{name}.colour.png")), n, n, png::ColorType::Rgba, &rgba);
    let depth: Vec<u8> = f.depth.iter().flat_map(|w| [(w >> 16) as u8, (w >> 8) as u8, *w as u8]).collect();
    write_png(&dir.join(format!("{name}.depth.png")), n, n, png::ColorType::Rgb, &depth);
    let stencil: Vec<u8> = f.stencil.iter().map(|&w| w as u8).collect();
    write_png(&dir.join(format!("{name}.stencil.png")), n, n, png::ColorType::Grayscale, &stencil);
}

pub fn load_reference(name: &str) -> Option<Frame> {
    let dir = data_dir().join("frames");
    let rgba = read_png(&dir.join(format!("{name}.colour.png")))?;
    let depth = read_png(&dir.join(format!("{name}.depth.png")))?;
    let stencil = read_png(&dir.join(format!("{name}.stencil.png")))?;
    Some(Frame {
        colour: rgba.chunks_exact(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())).collect(),
        depth: depth.chunks_exact(3).map(|c| (c[0] as u32) << 16 | (c[1] as u32) << 8 | c[2] as u32).collect(),
        stencil: stencil.iter().map(|&b| b as u32).collect(),
    })
}

// scenes.tsv: name, hash, then the tolerance. `#` starts a comment.

pub struct Entry {
    pub hash: String,
    pub tol: Tolerance,
}

pub fn load_entries() -> BTreeMap<String, Entry> {
    let text = std::fs::read_to_string(data_dir().join("scenes.tsv")).unwrap_or_default();
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 7, "scenes.tsv: {line:?}");
        let n = |i: usize| f[i].parse::<u64>().unwrap_or_else(|_| panic!("scenes.tsv: {line:?}"));
        out.insert(
            f[0].to_string(),
            Entry {
                hash: f[1].to_string(),
                tol: Tolerance { colour_delta: n(2) as u32, colour_bad: n(3) as usize, depth_delta: n(4) as u32, depth_bad: n(5) as usize, stencil_bad: n(6) as usize },
            },
        );
    }
    out
}

pub fn save_entries(e: &BTreeMap<String, Entry>) {
    let mut s = String::from(
        "# VPro draw-level goldens (src/dev/odyssey/golden). One scene a line:\n\
         #   name  hash(blake3 of colour, depth, stencil)  colour_delta colour_bad depth_delta depth_bad stencil_bad\n\
         # The hash is the default, bit-exact check. The five numbers are the limits for\n\
         # VPRO_GOLDEN=tolerance: pixels past colour_delta (per channel) or depth_delta (24-bit\n\
         # word) may number at most colour_bad / depth_bad, and at most stencil_bad stencil pixels\n\
         # may differ. Edit them by hand per scene; blessing keeps what is there.\n",
    );
    for (k, v) in e {
        let t = v.tol;
        s += &format!("{k}\t{}\t{}\t{}\t{}\t{}\t{}\n", v.hash, t.colour_delta, t.colour_bad, t.depth_delta, t.depth_bad, t.stencil_bad);
    }
    std::fs::write(data_dir().join("scenes.tsv"), s).unwrap();
}

/// Run a scene on a fresh board and return its frame and how many pixels
/// it changed from the cleared window.
pub fn render_counted(name: &str) -> (Frame, usize) {
    let mut c = scenes::Ctx::new();
    assert!(scenes::run_scene(name, &mut c), "no scene {name}");
    let f = Frame::of(&c.board);
    let n = f.differing(&c.blank());
    (f, n)
}

/// A scene that draws nothing would pin nothing.
const MIN_DRAWN: usize = 200;

pub fn render(name: &str) -> Frame {
    let (f, n) = render_counted(name);
    assert!(n >= MIN_DRAWN, "scene {name} drew almost nothing ({n} pixels)");
    f
}

/// The check one scene's test makes.
pub fn check(name: &str) {
    let f = render(name);
    let entries = load_entries();
    let Some(e) = entries.get(name) else {
        panic!("scene {name} has no golden: run scripts/vpro-goldens.sh bless");
    };
    let hash = f.hash();
    if hash == e.hash {
        return;
    }
    let msg = match load_reference(name) {
        Some(r) => {
            let d = distance(&f, &r, &e.tol);
            let within = d.within(&e.tol);
            if within && std::env::var("VPRO_GOLDEN").is_ok_and(|v| v == "tolerance") {
                eprintln!("{name}: hash moved, within tolerance: {d:?}");
                return;
            }
            format!("{d:?}; {} of the limits {:?}", if within { "inside" } else { "OUTSIDE" }, e.tol)
        }
        None => "no reference image".to_string(),
    };
    if let Ok(dir) = std::env::var("VPRO_GOLDEN_OUT") {
        // The failing frame's colour, for looking at; the references stay put.
        std::fs::create_dir_all(&dir).unwrap();
        let p = Path::new(&dir).join(format!("{name}.colour.png"));
        let rgba: Vec<u8> = f.colour.iter().flat_map(|w| w.to_be_bytes()).collect();
        write_png(&p, SIDE as u32, SIDE as u32, png::ColorType::Rgba, &rgba);
    }
    panic!("scene {name}: hash {hash} is not the golden {} ({msg})", e.hash);
}

/// Re-take every scene's hash and reference images, keeping tolerances.
/// `VPRO_GOLDEN_BLESS=1 cargo test --release --lib odyssey::golden::bless -- --ignored`
#[test]
#[ignore]
fn bless() {
    if std::env::var("VPRO_GOLDEN_BLESS").is_err() {
        panic!("set VPRO_GOLDEN_BLESS=1: blessing rewrites the goldens");
    }
    let mut entries = load_entries();
    let only = std::env::var("VPRO_GOLDEN_SCENE").ok();
    let mut blank = vec![];
    for name in scenes::NAMES {
        if only.as_deref().is_some_and(|o| o != *name) {
            continue;
        }
        let (f, n) = render_counted(name);
        eprintln!("{name}: {n} pixels changed");
        if n < MIN_DRAWN {
            blank.push(*name);
            continue;
        }
        let tol = entries.get(*name).map(|e| e.tol).unwrap_or_default();
        entries.insert(name.to_string(), Entry { hash: f.hash(), tol });
        save_reference(name, &f);
    }
    assert!(blank.is_empty(), "scenes that drew almost nothing: {blank:?}");
    // Scenes dropped from the list leave no golden behind.
    entries.retain(|k, _| scenes::NAMES.contains(&k.as_str()));
    save_entries(&entries);
    vertices::bless();
    eprintln!("blessed {} scenes", entries.len());
}

/// Every scene in the list has a hash and reference images, and no golden is
/// left for a scene that is gone.
#[test]
fn every_scene_has_a_golden_and_no_golden_is_orphaned() {
    let entries = load_entries();
    for name in scenes::NAMES {
        assert!(entries.contains_key(*name), "scene {name} has no line in scenes.tsv");
        assert!(load_reference(name).is_some(), "scene {name} has no reference images");
    }
    for k in entries.keys() {
        assert!(scenes::NAMES.contains(&k.as_str()), "scenes.tsv has {k}, which is no scene");
    }
}

/// The comparison itself: a changed pixel is seen, and tolerance counts it.
#[test]
fn the_distance_counts_what_moved() {
    let f = Frame { colour: vec![0x1020_3040; 4], depth: vec![100; 4], stencil: vec![0; 4] };
    let mut g = f.clone();
    g.colour[1] = 0x1020_3045;
    g.depth[2] = 100 + 0x8000;
    g.stencil[3] = 1;
    let t = Tolerance { colour_delta: 2, colour_bad: 0, depth_delta: 0x4000, depth_bad: 0, stencil_bad: 0 };
    let d = distance(&g, &f, &t);
    assert_eq!((d.colour_max, d.colour_differ, d.colour_bad), (5, 1, 1));
    assert_eq!((d.depth_max, d.depth_differ, d.depth_bad), (0x8000, 1, 1));
    assert_eq!(d.stencil_differ, 1);
    assert!(!d.within(&t));
    assert!(distance(&f, &f, &t).within(&t));
    assert_ne!(f.hash(), g.hash());
}

fn render_without(name: &str, off: &'static str) -> Frame {
    let mut c = scenes::Ctx::new();
    c.off = Some(off);
    assert!(scenes::run_scene(name, &mut c));
    Frame::of(&c.board)
}

/// The goldens are only worth keeping if a scene's frame depends on the
/// feature it is about: leave the feature out and the frame changes, by more
/// than the odd pixel.
#[test]
fn the_features_the_scenes_cover_change_their_frames() {
    let pairs = [
        ("fog_ramp_triangles", "fog"),
        ("fog_lines_and_points", "fog"),
        ("fog_over_blend", "fog"),
        ("tex_with_fog_blend_depth", "fog"),
        ("tex_with_fog_blend_depth", "texture"),
        ("blend_triangles_over", "blend"),
        ("blend_factor_grid", "blend"),
        ("polygon_smooth", "blend"),
        ("lines_smooth", "blend"),
        ("depth_less", "depth"),
        ("depth_range_wide", "depth"),
        ("tex_format_rgba", "texture"),
        ("tex_env_blend_rgba", "texture"),
        ("tex_mip_linear_linear", "texture"),
        ("perspective_correct_colour", "texture"),
        ("light_lit_blended_textured", "texture"),
    ];
    for (scene, feature) in pairs {
        let on = render(scene);
        let off = render_without(scene, feature);
        assert!(on.differing(&off) >= 50, "{scene} looks the same with {feature} left out ({} pixels differ)", on.differing(&off));
    }
}

/// Scenes that vary one thing at a time produce different frames: a texture
/// format, environment, filter or light that drew like another would be a
/// scene that does not test what it names.
#[test]
fn scenes_of_one_family_are_told_apart() {
    let families: [&[&str]; 5] = [
        &["tex_format_rgba", "tex_format_rgb", "tex_format_luminance", "tex_format_luminance_alpha", "tex_format_alpha", "tex_format_intensity"],
        &["tex_env_replace_rgba", "tex_env_modulate_rgba", "tex_env_decal_rgba", "tex_env_decal_rgb", "tex_env_blend_rgba", "tex_env_blend_luminance", "tex_env_modulate_luminance_alpha", "tex_env_replace_alpha"],
        &["tex_mip_nearest_nearest", "tex_mip_linear_nearest", "tex_mip_nearest_linear", "tex_mip_linear_linear", "tex_nearest_repeat", "tex_clamp_to_edge_linear"],
        &["light_directional", "light_positional_attenuated", "light_spot", "light_local_viewer_specular", "light_three_lights_colour_material", "light_unlit_emission_only"],
        &["depth_less", "depth_lequal_greater", "depth_always_masked", "depth_equal_notequal", "blend_equations", "blend_constant_colour"],
    ];
    for family in families {
        let hashes: Vec<String> = family.iter().map(|n| render(n).hash()).collect();
        for i in 0..hashes.len() {
            for j in 0..i {
                assert_ne!(hashes[i], hashes[j], "{} and {} draw the same frame", family[i], family[j]);
            }
        }
    }
}

/// A clip vector scaled by a tiny positive w is the same point: the
/// mapping keeps VPro's w > 0 rule rather than the shared core's cut at
/// w <= 1e-6 (the board's rule for a small w is unknown).
#[test]
fn a_small_positive_w_maps_as_the_same_point() {
    let st = scenes::Ctx::new().st;
    let sub = (1i32 << gl_rules().subpixel_bits) as f32;
    let at = |w: f32| st.clip_to_window(&[0.5 * w, -0.25 * w, 0.125 * w, w], sub);
    let want = at(1.0).expect("a window position");

    for w in [1e-6, 5e-7, 1e-7, 1e-12] {
        assert_eq!(at(w), Some(want), "w {w}");
    }
    assert_eq!(at(0.0), None);
    assert_eq!(at(-1e-7), None);
}

/// A triangle drawn at w = 1e-7 covers the pixels the same triangle does at
/// w = 1.
#[test]
fn a_triangle_at_a_small_w_draws_as_at_one() {
    let draw = |w: f32| {
        let mut c = scenes::Ctx::new();
        let g = [0.0, 1.0, 0.0, 1.0];
        // The same point, its whole clip vector scaled by w.
        let at = |x: f32, y: f32| {
            let mut v = scenes::v(x, y, 0.25, g);

            v.p = v.p.map(|c| c * w);
            v
        };

        c.tri(at(10.0, 10.0), at(110.0, 20.0), at(40.0, 110.0));
        Frame::of(&c.board)
    };
    let (one, small) = (draw(1.0), draw(1e-7));
    let blank = scenes::Ctx::new().blank();

    assert!(one.colour.iter().zip(&blank.colour).filter(|(a, b)| a != b).count() > 1000, "the triangle draws");
    assert_eq!(small.colour, one.colour);
}

/// amesh's block 16 is the clip w row itself and its depth row the same row
/// times 0.99999999. The depth row and divisor are now
/// taken on eye and w rounded to single precision, which moves the ratio by
/// about 6e-8: it must stay inside the 2^-20 slack, so a wall from the floor
/// to a peak still draws, and the depth it writes is the far plane's give or
/// take the steps that error is worth.
#[test]
fn the_depth_row_on_single_precision_eye_stays_in_the_slack() {
    let block16 = [-0.0f32, -6.84, -2.31, 65.93];
    let draw = |rows: bool| {
        let mut c = scenes::Ctx::new();
        let g = [0.0, 1.0, 0.0, 1.0];

        if rows {
            for k in 0..4 {
                c.board.gl_block16[k] = block16[k].to_bits();
            }
            // The board's divisor is block 16 over its own w.
            c.depth_row(block16.map(|b| b as f64 * 0.99999999 / block16[3] as f64));
        } else {
            c.depth_row([0.0, 0.0, 0.0, 1.0]);
        }
        c.clear([0.0; 4], 0, 0);
        c.tri(scenes::v(10.0, 10.0, -0.9, g), scenes::v(110.0, 20.0, 0.9, g), scenes::v(40.0, 110.0, 0.3, g));
        Frame::of(&c.board)
    };
    let (rows, far) = (draw(true), draw(false));

    assert!(far.colour.iter().filter(|&&p| p != 0).count() > 1000, "the reference draws");
    assert_eq!(rows.colour, far.colour, "every pixel the far plane draws is drawn");
    let worst = rows.depth.iter().zip(&far.depth).map(|(a, b)| (a & 0xff_ffff).abs_diff(b & 0xff_ffff)).max().unwrap();
    eprintln!("worst depth steps from the far plane: {worst}");
    assert!(worst <= 2, "{worst} steps from the far plane");
}
