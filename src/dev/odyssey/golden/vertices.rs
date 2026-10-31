//! The vertex side on its own: where object-space points land.
//!
//! The scene frames pin what the vertex side does only as far as it shows in
//! pixels. This pins the numbers: a corpus of points through several
//! model-view, projection, viewport and window-matrix setups (eye position,
//! clip coordinates, window position and depth), and through the board's
//! lighting under several light setups (the vertex's front colour). The file
//! `testdata/golden/vertices.txt` holds them as the shortest decimal that
//! reads back exactly. The default check is exact; `VPRO_GOLDEN=tolerance`
//! compares numerically against the limits in the file's header, in window
//! pixels for x and y, in window depth, and in colour units, which is the
//! check for a move to 32-bit floats.

use crate::dev::odyssey::consts::colour;
use crate::dev::odyssey::draw::gl_rules;
use crate::dev::odyssey::gl_context::{vtx, GlDrawState};
use crate::dev::odyssey::raster_bench::Bench;
use super::scenes::{load_lights, Light, MATERIAL};
use super::*;

use super::raster_bench::Lcg;

pub struct Row {
    pub name: String,
    pub kind: &'static str,
    pub vals: Vec<f64>,
}

fn rows() -> Vec<Row> {
    let mut out = Vec::new();
    let ident = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let (l, r, b, t, n, f) = (-0.6f32, 0.4f32, -0.5f32, 0.5f32, 1.0f32, 30.0f32);
    let frustum = [2.0 * n / (r - l), 0.0, 0.0, 0.0, 0.0, 2.0 * n / (t - b), 0.0, 0.0, (r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1.0, 0.0, 0.0, -2.0 * f * n / (f - n), 0.0];
    let ortho = [0.5f32, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, -0.1, 0.2, -0.9, 1.0];
    let (s, k) = (0.6f32, 0.8f32);
    let rot = [k, s, 0.0, 0.0, -s, k, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.7, -0.4, -6.0, 1.0];
    let tilt = [k, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, k, 0.0, 0.0, 0.3, -9.0, 1.0];
    let window = [60.0 * k, 60.0 * s, 0.0, 0.0, -60.0 * s, 60.0 * k, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 400.5, 300.25, 0.0, 1.0];
    // (name, modelview, projection, viewport, window matrix)
    let setups: [(&str, [f32; 16], [f32; 16], [i32; 4], Option<[f32; 16]>); 6] = [
        ("identity", ident, ident, [0, 0, 128, 128], None),
        ("ortho_rot", rot, ortho, [10, 20, 600, 480], None),
        ("frustum_rot", rot, frustum, [0, 0, 1280, 1024], None),
        ("frustum_tilt_vp", tilt, frustum, [-30, 15, 333, 257], None),
        ("window_matrix", ident, ident, [0, 0, 1280, 1024], Some(window)),
        ("window_matrix_rot", rot, ident, [0, 0, 1280, 1024], Some(window)),
    ];
    let mut rng = Lcg(2024);
    let points: Vec<[f32; 4]> = (0..40)
        .map(|i| match i {
            0 => [0.0, 0.0, 0.0, 1.0],
            1 => [1.0, 1.0, 1.0, 1.0],
            2 => [-1.0, 0.5, -2.0, 1.0],
            3 => [0.3, -0.3, 5.0, 1.0],
            _ => [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), rng.range(-6.0, 4.0), if i % 5 == 0 { rng.range(0.5, 2.0) } else { 1.0 }],
        })
        .collect();
    for (name, mv, proj, vp, wm) in setups {
        let mut st = GlDrawState { modelview: mv, projection: proj, viewport: vp, ..Default::default() };
        if let Some(w) = wm {
            st.window_matrix_given = true;
            st.window_matrix = w;
            st.window_depth_scale = 0.75;
            st.window_rows_down = name == "window_matrix_rot";
        }
        for (i, p) in points.iter().enumerate() {
            let mut vertex = [0.0f32; vtx::FLOATS];
            vertex[..4].copy_from_slice(p);
            let vv = st.volume_from_draw_vertex(&vertex);
            out.push(Row { name: format!("{name}/{i}"), kind: "eye", vals: vv.eye.iter().map(|&x| f64::from(x)).collect() });
            out.push(Row { name: format!("{name}/{i}"), kind: "clip", vals: vv.clip.iter().map(|&x| f64::from(x)).collect() });
            // A vertex at or behind the eye has no window position.
            if vv.clip[3] > 0.0 {
                let win = st.clip_to_window(&vv.clip, (1i32 << gl_rules().subpixel_bits) as f32).expect("a window position");
                out.push(Row { name: format!("{name}/{i}"), kind: "win", vals: win.iter().map(|&c| f64::from(c)).collect() });
            }
        }
    }

    // Lighting: the board's front colour at a spread of positions and normals.
    let dir = |pos: [f32; 3], colour: [f32; 3]| Light { pos, directional: true, colour, atten: [1.0, 0.0, 0.0], spot: None };
    let setups: [(&str, Vec<(usize, Light)>, [f32; 3], bool, f32, u32); 6] = [
        ("directional", vec![(0, dir([0.5, 0.6, -0.62], [1.0, 0.9, 0.8]))], [0.2; 3], false, 20.0, 0),
        ("positional", vec![(1, Light { pos: [1.5, 1.0, -2.0], colour: [1.0; 3], atten: [1.0, 0.3, 0.15], ..Default::default() })], [0.1; 3], false, 8.0, 0),
        ("spot", vec![(0, Light { pos: [0.0, 1.5, -2.5], colour: [1.0, 1.0, 0.8], atten: [1.0, 0.0, 0.0], spot: Some(([0.0, -0.5, 0.85], 0.8, 4.0)), ..Default::default() })], [0.05; 3], false, 30.0, 0),
        ("local_viewer", vec![(2, Light { pos: [-1.0, 1.0, -1.5], colour: [0.9, 0.9, 1.0], atten: [1.0, 0.0, 0.0], ..Default::default() })], [0.15; 3], true, 64.0, 0),
        ("three", vec![(0, dir([1.0, 0.0, -1.0], [1.0, 0.1, 0.1])), (3, dir([-1.0, 0.5, -1.0], [0.1, 0.1, 1.0])), (7, Light { pos: [0.0, -2.0, -1.5], colour: [0.2, 1.0, 0.2], atten: [1.0, 0.1, 0.0], spot: Some(([0.0, 1.0, 0.5], 0.7, 2.0)), ..Default::default() })], [0.1; 3], false, 12.0, colour::MATERIAL_AMBIENT | colour::MATERIAL_DIFFUSE),
        ("none", vec![], [0.3; 3], false, 1.0, colour::MATERIAL_EMISSION),
    ];
    let mut board = Bench::empty().board;
    for (name, lights, ambient, local, shininess, cm) in setups {
        load_lights(&mut board, &lights, ambient, local, shininess, MATERIAL, cm);
        board.geom.colour = [0.9f32, 0.7, 0.3, 0.75].map(f32::to_bits);
        let mut rng = Lcg(99);
        for i in 0..24 {
            let mut nrm = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt().max(1e-3);
            nrm = nrm.map(|x| x / len);
            board.geom.normal = nrm.map(f32::to_bits);
            let obj = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), 1.0];
            let rgba = board.gl_light(&obj);
            out.push(Row { name: format!("light_{name}/{i}"), kind: "rgba", vals: rgba.iter().map(|&c| c as f64).collect() });
        }
    }
    out
}

const HEADER: &str = "# VPro vertex-side goldens (src/dev/odyssey/golden/vertices.rs).\n\
# name  kind  values, as the shortest decimal that reads back exactly: kind eye is the eye\n\
# position, clip the clip coordinates, win the window x, y and depth in [0, 1], rgba the\n\
# lit front colour. Limits for VPRO_GOLDEN=tolerance:\n";
const TOLERANCE_LINE: &str = "# tolerance eye 1e-5 clip 1e-5 win_xy 0.01 win_z 1e-5 rgba 0.004\n";

fn render() -> String {
    let mut s = String::from(HEADER);
    s += TOLERANCE_LINE;
    for r in rows() {
        s += &format!("{}\t{}", r.name, r.kind);
        for v in &r.vals {
            s += &format!("\t{v:e}");
        }
        s.push('\n');
    }
    s
}

fn path() -> std::path::PathBuf {
    data_dir().join("vertices.txt")
}

pub fn bless() {
    std::fs::write(path(), render()).unwrap();
}

struct Limits {
    eye: f64,
    clip: f64,
    win_xy: f64,
    win_z: f64,
    rgba: f64,
}

fn limits(text: &str) -> Limits {
    let line = text.lines().find(|l| l.starts_with("# tolerance")).expect("a tolerance line");
    let f: Vec<&str> = line.split_whitespace().collect();
    let get = |k: &str| -> f64 { f.iter().position(|x| *x == k).and_then(|i| f.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or_else(|| panic!("tolerance {k}")) };
    Limits { eye: get("eye"), clip: get("clip"), win_xy: get("win_xy"), win_z: get("win_z"), rgba: get("rgba") }
}

#[test]
fn the_vertex_side_gives_the_golden_numbers() {
    let want = std::fs::read_to_string(path()).expect("vertices.txt: run scripts/vpro-goldens.sh bless");
    let got = render();
    if got == want {
        return;
    }
    let lim = limits(&want);
    let (mut worst, mut bad, mut moved) = ([0.0f64; 5], 0usize, 0usize);
    let (wl, gl): (Vec<&str>, Vec<&str>) = (want.lines().filter(|l| !l.starts_with('#')).collect(), got.lines().filter(|l| !l.starts_with('#')).collect());
    assert_eq!(wl.len(), gl.len(), "the corpus changed size: {} rows against the golden {}", gl.len(), wl.len());
    for (w, g) in wl.iter().zip(&gl) {
        if w == g {
            continue;
        }
        moved += 1;
        let (wf, gf): (Vec<&str>, Vec<&str>) = (w.split('\t').collect(), g.split('\t').collect());
        assert_eq!((wf[0], wf[1], wf.len()), (gf[0], gf[1], gf.len()), "row shape changed: {w:?} against {g:?}");
        for i in 2..wf.len() {
            let (a, b): (f64, f64) = (wf[i].parse().unwrap(), gf[i].parse().unwrap());
            let err = (a - b).abs();
            let (slot, limit) = match (wf[1], i) {
                ("eye", _) => (0, lim.eye),
                ("clip", _) => (1, lim.clip),
                ("win", 2 | 3) => (2, lim.win_xy),
                ("win", _) => (3, lim.win_z),
                _ => (4, lim.rgba),
            };
            worst[slot] = worst[slot].max(err);
            if err > limit {
                bad += 1;
                eprintln!("{}: {} value {} is {b:e}, the golden {a:e}", wf[0], wf[1], i - 2);
            }
        }
    }
    let msg = format!("{moved} rows moved; worst error eye {:e} clip {:e} win_xy {:e} win_z {:e} rgba {:e}; {bad} values outside the limits", worst[0], worst[1], worst[2], worst[3], worst[4]);
    if bad == 0 && std::env::var("VPRO_GOLDEN").is_ok_and(|v| v == "tolerance") {
        eprintln!("vertex goldens: {msg}");
        return;
    }
    panic!("vertex side differs from vertices.txt: {msg}");
}
