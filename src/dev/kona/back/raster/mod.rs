//! Drawing into the pixel store.
//!
//! The integer primitives the textport and the X server use: points, boxes
//! and bitmaps written into one word position of the store, clipped to the
//! managed area. The store is bottom-origin, y up, and the textport's
//! coordinates are the board's own, written as given. The X server's run down
//! from the top of the screen, so its fills mirror the row (`x::fill`); both were
//! settled by comparing the store rendered from the recordings with their
//! screen captures. The display then flips the rows once.
//!
//! # The fill rule
//!
//! For a box with corners `x1 < x2`, `y1 < y2` the board lights columns
//! `x1 + 1` to `x2` and rows `y1 + 1` to `y2`: the left column and bottom row of
//! a whole-pixel box are left out (established per the hardware documents for boxes sent
//! as four unchanged corners; the grade of that rests on a ruling still to
//! come). `lit` is the one place that rule lives, so a ruling changes a function and its grade.
//!
//! # The GL stage, and what it leaves out
//!
//! `setup` and `pixel` draw the GL client's triangles on the tile scheduler
//! (`back::tiles`): the 1/8-pixel set-up, the fill rule, colour, depth, the
//! scissor and polygon offset; then, per fragment, window-ID ownership, the
//! polygon and line stipples, the alpha test, stencil, depth, blend, the logic
//! operation and the write masks, and, for a bound 1D or 2D texture, sampling
//! with a level of detail taken from the 2 x 2 quad. A multisampled window runs
//! the same tests per sample (`multisample`). Points and lines reach set-up as
//! triangles (`gl`: a point is a square or a disc, a line a rectangle about its
//! segment). What the stage does not do yet, each a gap and not a decision:
//!
//! - nothing for the board's fog table or its depth-offset register: fog is
//!   applied per fragment from the interpolated eye distance (`fog`);
//! - depth is the 32.16 fixed-point plane of `pixel`, clamped to the 32-bit
//!   fraction, not the board's float or compressed depth forms;
//! - the order of the fragment operations, the blend and dither arithmetic and
//!   the line and point shapes are the model's own, not read from the board.

pub mod fog;
pub mod multisample;
pub mod ops;
pub mod pixel;
pub mod texel;
pub mod setup;
pub mod smooth;
use super::store::{PixelStore, WordArray};
use super::tiles::Tiles;
use ops::{BlendEq, BlendFactor, Compare, LogicOp, Ownership, StencilOp};

/// A rectangle in pixels, far corner exclusive: the only rectangle type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    pub fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    pub fn intersect(&self, o: &Rect) -> Rect {
        Rect { x0: self.x0.max(o.x0), y0: self.y0.max(o.y0), x1: self.x1.min(o.x1), y1: self.y1.min(o.y1) }
    }
}

/// A vertex handed to set-up: relative to the window origin, at the vertex
/// bus's precision. `x` and `y` are on a 1/256-pixel grid (the bus carries 20
/// bits, unsigned; the model keeps the sign, since a window origin can be
/// negative and set-up works in integers), `z` is a 32-bit fraction of the
/// depth range, `c` four 13-bit components (8191 is full intensity) and `t`
/// two sets of the texture words `s/w, t/w, r/w, q/w`, each linear in screen
/// space (zero when untextured). That a textured vertex carries a fourth word
/// is inferred, and so is its form (1/w, q/w or 1/q): the model uses q/w,
/// which is 1/w for the usual q of 1 and keeps the divide right when q is not.
///
/// Positions are rounded onto the 1/256 grid once, to the nearest with ties to even (`vertex::grid`).
/// The half-pixel GL offset is set-up's conversion and is not applied here.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WVertex {
    pub x: i32,
    pub y: i32,
    pub z: u32,
    pub c: [u16; 4],
    pub t: [[f32; 4]; 2],
    /// How far along its line the vertex is, in pixels (zero for a triangle's).
    pub d: f32,
    /// The fog coordinate over w, and 1 / w: both linear in screen space, so the fragment's
    /// coordinate is their quotient (zero when fog is off).
    pub fog: [f32; 2],
}

/// A glClear, over `rect` (the draw state's clip rectangle; the viewport does
/// not apply). Each plane the request names is cleared under its write mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClearReq {
    pub rect: Rect,
    /// The colour word to write, `0xAARRGGBB` from the clear colour, or None.
    pub colour: Option<u32>,
    /// The depth fraction to write, or None.
    pub depth: Option<u32>,
    /// The stencil value to write, or None.
    pub stencil: Option<u8>,
    /// Which of red, green, blue and alpha the clear writes, whether depth is
    /// written, and which stencil bits.
    pub colour_mask: [bool; 4],
    pub depth_mask: bool,
    pub stencil_mask: u8,
}

impl Default for ClearReq {
    fn default() -> Self {
        ClearReq { rect: Rect::default(), colour: None, depth: None, stencil: None, colour_mask: [true; 4], depth_mask: true, stencil_mask: 0xff }
    }
}

/// The alpha test: a fragment passes when its alpha, in 13-bit units, compares
/// with `reference` (`func.passes(alpha, reference)`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AlphaTest {
    pub enabled: bool,
    pub func: Compare,
    pub reference: u32,
}

/// The stencil test and its operations: the test compares `reference & read_mask`
/// with the stored value under `read_mask`; each operation's result is written
/// under `write_mask`. One state serves front and back faces: a model limit, since
/// the tokens name no face and no recording sends them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilState {
    pub enabled: bool,
    pub func: Compare,
    pub reference: u8,
    pub read_mask: u8,
    pub write_mask: u8,
    pub fail: StencilOp,
    pub zfail: StencilOp,
    pub zpass: StencilOp,
}

impl Default for StencilState {
    fn default() -> Self {
        StencilState { enabled: false, func: Compare::Always, reference: 0, read_mask: 0xff, write_mask: 0xff, fail: StencilOp::Keep, zfail: StencilOp::Keep, zpass: StencilOp::Keep }
    }
}

/// Blending: `src * src_factor + dst * dst_factor` under `eq` (add is all there is), with
/// the constant colour in 13-bit units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendState {
    pub enabled: bool,
    pub src: BlendFactor,
    pub dst: BlendFactor,
    pub eq: BlendEq,
    pub constant: [u32; 4],
}

impl Default for BlendState {
    fn default() -> Self {
        BlendState { enabled: false, src: BlendFactor::One, dst: BlendFactor::Zero, eq: BlendEq::Add, constant: [0; 4] }
    }
}

/// What a draw runs under, as plain data: it is copied into the scheduler's
/// draw-state table and read there by the workers. Built from the context's GL
/// state when that state changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawState {
    /// The window's origin in plane pixels; a vertex is relative to it.
    pub origin: [i32; 2],
    /// Where a draw can write: the window cut to the scissor when it is on and
    /// to the screen.
    pub clip: Rect,
    /// The viewport in plane pixels. A primitive is cut to it as well.
    pub viewport: Rect,
    /// The colour words of the buffers drawn to, and how many.
    pub colour: [u32; 2],
    pub colours: u8,
    /// The depth word and the bits of depth in it; 0 bits: no depth plane.
    pub depth_word: u32,
    pub depth_bits: u32,
    pub depth_test: bool,
    pub depth_func: Compare,
    /// The depth plane is written by a passing fragment.
    pub depth_mask: bool,
    /// The stencil word, and its bits (8 or 0: none).
    pub stencil_word: u32,
    pub stencil_bits: u32,
    pub stencil: StencilState,
    pub alpha: AlphaTest,
    pub blend: BlendState,
    /// A logic operation applied to the colour written and the buffer's, or None.
    pub logic: Option<LogicOp>,
    /// Which of red, green, blue and alpha a draw writes.
    pub colour_mask: [bool; 4],
    /// Dither is carried and applied as off: no source gives the matrix.
    pub dither: bool,
    /// Polygon offset, added to the depth plane in set-up: `factor` times the
    /// larger depth slope plus `units` times the smallest depth step.
    pub offset: Option<[f32; 2]>,
    /// The window-ID test, when the window's enable is set. A fragment whose pixel
    /// fails it is dropped. X drawing does not take it (approximated: the X server paints
    /// the display-ID plane itself, and no source says the pipe tests its fills).
    pub ownership: Option<Ownership>,
    /// The line stipple: the 16-bit pattern and the repeat count of each bit.
    pub line_stipple: Option<(u16, u32)>,
    /// The polygon stipple: 32 rows of 32 bits, the most significant bit leftmost.
    pub polygon_stipple: Option<[u32; 32]>,
    /// The bound texture and its environment, when texturing is on.
    pub texture: Option<texel::TexDesc>,
    /// The samples of a multisampled window, when multisampling is on.
    pub samples: Option<multisample::Samples>,
    /// Fog, when it is on.
    pub fog: Option<fog::Fog>,
    /// Which primitives are smoothed.
    pub smooth: Smooth,
}

/// The smoothing switches a draw runs under. A multisampled window smooths lines only (see `smooth`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Smooth {
    pub points: bool,
    pub lines: bool,
    pub polygons: bool,
}

impl Default for DrawState {
    fn default() -> Self {
        DrawState {
            origin: [0; 2],
            clip: Rect::default(),
            viewport: Rect::default(),
            colour: [0; 2],
            colours: 0,
            depth_word: 0,
            depth_bits: 0,
            depth_test: false,
            depth_func: Compare::Always,
            depth_mask: true,
            stencil_word: 0,
            stencil_bits: 0,
            stencil: StencilState::default(),
            alpha: AlphaTest::default(),
            blend: BlendState::default(),
            logic: None,
            colour_mask: [true; 4],
            dither: false,
            offset: None,
            ownership: None,
            line_stipple: None,
            polygon_stipple: None,
            texture: None,
            samples: None,
            fog: None,
            smooth: Smooth::default(),
        }
    }
}

/// What the GL path made in one job, for `KonaBackend` to flush into set-up:
/// triangles and clears, and the draw state they run under.
#[derive(Default)]
pub struct Prims {
    pub tris: Vec<[WVertex; 3]>,
    /// Triangles that make up lines: the rectangle about a segment, each vertex
    /// carrying `d`, so the line stipple can index its pattern.
    pub lines: Vec<[WVertex; 3]>,
    /// The smoothed polygons and lines, each as a run of `aa_verts`.
    pub aa: Vec<SmoothPoly>,
    pub aa_verts: Vec<WVertex>,
    /// The smoothed points.
    pub discs: Vec<SmoothPoint>,
    pub clears: Vec<ClearReq>,
    /// The state the above run under; set whenever either is not empty.
    pub state: Option<DrawState>,
}

/// A smoothed polygon or line: `n` convex vertices of `Prims::aa_verts` from `start`, drawn as a fan
/// from the first. A line is the rectangle about its segment, with its distance along the line in
/// each vertex, and takes the line stipple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmoothPoly {
    pub start: u32,
    pub n: u32,
    pub line: bool,
}

/// A smoothed point: the square about its disc as four corners (counter-clockwise), and the disc's
/// centre and radius in pixels, in GL's window coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothPoint {
    pub quad: [WVertex; 4],
    pub x: f32,
    pub y: f32,
    pub r: f32,
}

impl Prims {
    /// Empty, keeping the capacity.
    pub fn clear(&mut self) {
        self.tris.clear();
        self.lines.clear();
        self.discs.clear();
        self.aa.clear();
        self.aa_verts.clear();
        self.clears.clear();
        self.state = None;
    }
}

/// The coordinates a box from `lo` to `hi` lights along one axis, as a
/// half-open range: the left column and bottom row are left out and the right
/// column and top row lit (the rule the hardware documents give for the board).
/// Empty when `hi < lo`.
pub fn lit(lo: i64, hi: i64) -> std::ops::Range<i64> {
    lo + 1..hi + 1
}

/// A surface onto word `word` of the store, after every queued raster job has
/// run: whatever reads or writes the store's pixels comes through here, so the
/// tile workers never race it. The drain lives in the way to the pixels, not in a
/// list of the jobs that need it.
pub fn surface<'a>(tiles: &mut Option<Tiles>, store: &'a PixelStore, word: usize) -> Surface<'a> {
    if let Some(t) = tiles {
        // A poisoned scheduler is reported by `Tiles::is_poisoned`, which the
        // board reads; the drain itself still ends.
        let _ = t.drain();
    }
    Surface::new(store, word)
}

/// A word position of the store to draw into, and where drawing stops.
pub struct Surface<'a> {
    store: &'a PixelStore,
    array: WordArray<'a>,
}

impl<'a> Surface<'a> {
    /// Draw into word `word` of the store's pixels. Allocates the array: this
    /// is the draw's set-up, and resolves it once for every call after.
    pub fn new(store: &'a PixelStore, word: usize) -> Surface<'a> {
        store.ensure(word);
        Surface { store, array: store.array(word).expect("allocated by the set-up") }
    }

    /// A surface onto another word position of the same store, allocated
    /// now: the display-ID and overlay planes a draw reaches by its target.
    pub fn sibling(&self, word: usize) -> Surface<'a> {
        Surface::new(self.store, word)
    }

    /// The managed area's size.
    pub fn size(&self) -> (u32, u32) {
        (self.store.width(), self.store.height())
    }

    /// The word array drawn into.
    pub fn array(&self) -> WordArray<'a> {
        self.array
    }

    /// Set the pixel (`x`, `y`) to `value`; false if it is off the managed area.
    pub fn plot(&self, x: i64, y: i64, value: u32) -> bool {
        let (w, h) = self.size();
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            return false;
        }
        // SAFETY: the pixel is inside the managed area, which the array covers.
        unsafe { self.array.store(y as usize * self.array.pitch() + x as usize, value) };
        true
    }

    /// Fill the box with corners (`x1`, `y1`) and (`x2`, `y2`) by the fill rule (`lit`),
    /// clipped to the managed area. A box whose far corner is below or left of
    /// its near one draws nothing (what a reversed box does is unknown).
    pub fn fill_box(&self, x1: i64, y1: i64, x2: i64, y2: i64, value: u32) {
        let (xs, ys) = (lit(x1, x2), lit(y1, y2));
        let xs = xs.start.max(0)..xs.end.min(self.store.width() as i64);
        let ys = ys.start.max(0)..ys.end.min(self.store.height() as i64);
        let pitch = self.array.pitch();
        for y in ys {
            let row = y as usize * pitch;
            for x in xs.clone() {
                // SAFETY: both ranges are clipped to the managed area.
                unsafe { self.array.store(row + x as usize, value) };
            }
        }
    }

    /// Set the pixels of a bitmap whose bits are set, most significant bit
    /// first, row-major, `width` bits a row, with its row 0 at `y0` and its
    /// column 0 at `x0`. Rows and columns off the managed area are skipped
    /// without being looked at, and a clear word is skipped whole.
    pub fn bitmap(&self, x0: i64, y0: i64, width: u32, height: u32, bits: &[u32], value: u32) {
        let (w, h) = (self.store.width() as i64, self.store.height() as i64);
        let pitch = self.array.pitch();
        // The columns of the bitmap that land on the store.
        let (lo, hi) = ((-x0).max(0), (w - x0).min(width as i64));
        for r in 0..height as u64 {
            let y = y0 + r as i64;
            if y < 0 || y >= h || lo >= hi {
                continue;
            }
            let (row_bit, row_store) = (r * width as u64, y as usize * pitch);
            let (mut bit, end) = (row_bit + lo as u64, row_bit + hi as u64);
            while bit < end {
                let word = bits.get((bit / 32) as usize).copied().unwrap_or(0);
                if word == 0 {
                    bit = (bit / 32 + 1) * 32;
                    continue;
                }
                if word >> (31 - bit % 32) & 1 != 0 {
                    // SAFETY: the column is inside `lo..hi`, clipped to the store.
                    unsafe { self.array.store(row_store + (x0 + (bit - row_bit) as i64) as usize, value) };
                }
                bit += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(s: &PixelStore, w: usize) -> Vec<(u32, u32)> {
        let mut v = Vec::new();
        for y in 0..s.height() {
            for x in 0..s.width() {
                if s.read(x, y, w) != 0 {
                    v.push((x, y));
                }
            }
        }
        v
    }

    /// The four-corner packet of the hardware documents: x1 < x2, y1 < y2, no
    /// fraction, lights columns x1+1..=x2 and rows y1+1..=y2.
    #[test]
    fn a_box_lights_columns_and_rows_after_the_near_corner() {
        let s = PixelStore::new(16, 16);
        Surface::new(&s, 1).fill_box(2, 3, 5, 6, 7);
        let got = lit(&s, 1);
        assert_eq!(got.len(), 3 * 3);
        assert!(got.iter().all(|&(x, y)| (3..=5).contains(&x) && (4..=6).contains(&y)));
        assert_eq!(s.read(2, 4, 1), 0, "the left column is out");
        assert_eq!(s.read(3, 3, 1), 0, "the bottom row is out");
        assert_eq!(s.read(5, 6, 1), 7, "the right column and top row are in");
    }

    /// Adjacent boxes that share an edge meet with neither a gap nor an
    /// overlap, which is why the rule leaves the near edge out.
    #[test]
    fn boxes_sharing_an_edge_neither_gap_nor_overlap() {
        let s = PixelStore::new(8, 12);
        let d = Surface::new(&s, 0);
        d.fill_box(0, 0, 7, 5, 1);
        d.fill_box(0, 5, 7, 10, 2);
        for y in 1..=10u32 {
            for x in 1..=7u32 {
                assert_eq!(s.read(x, y, 0), if y <= 5 { 1 } else { 2 }, "({x},{y})");
            }
        }
    }

    /// The rule leaves out the left column and bottom row, and a reversed box is empty.
    #[test]
    fn the_rule_omits_the_left_and_bottom_edges() {
        assert_eq!(super::lit(2, 5), 3..6);
        assert!(super::lit(5, 2).is_empty());
    }

    /// A box at the left or bottom edge of the screen leaves column or row 0
    /// unlit; a box past the far edges is clipped; a reversed box draws nothing.
    #[test]
    fn boxes_are_clipped_and_reversed_ones_draw_nothing() {
        let s = PixelStore::new(8, 8);
        let d = Surface::new(&s, 0);
        d.fill_box(0, 0, 100, 100, 1);
        assert_eq!(s.read(0, 4, 0), 0);
        assert_eq!(s.read(4, 0, 0), 0);
        assert_eq!(lit(&s, 0).len(), 7 * 7);
        let t = PixelStore::new(8, 8);
        Surface::new(&t, 0).fill_box(5, 5, 2, 2, 1);
        Surface::new(&t, 0).fill_box(-50, -50, -10, -10, 1);
        assert!(lit(&t, 0).is_empty());
    }

    #[test]
    fn plot_refuses_off_screen_points() {
        let s = PixelStore::new(4, 4);
        let d = Surface::new(&s, 0);
        assert!(d.plot(3, 3, 9));
        assert!(!d.plot(4, 0, 9) && !d.plot(0, 4, 9) && !d.plot(-1, 0, 9) && !d.plot(i64::MAX, 0, 9));
        assert_eq!(lit(&s, 0), [(3, 3)]);
    }

    /// Bits go most significant first, row-major, and a missing word reads as
    /// clear.
    #[test]
    fn a_bitmap_sets_its_set_bits() {
        let s = PixelStore::new(16, 16);
        // 5 x 3: row 0 = 10101, row 1 = 01010, row 2 = 11111.
        let bits = [0b10101_01010_11111u32 << 17];
        Surface::new(&s, 2).bitmap(4, 8, 5, 3, &bits, 3);
        assert_eq!(lit(&s, 2).len(), 3 + 2 + 5);
        assert_eq!(s.read(4, 8, 2), 3);
        assert_eq!(s.read(5, 8, 2), 0);
        assert_eq!(s.read(5, 9, 2), 3);
        assert_eq!(s.read(8, 10, 2), 3);
        // A short payload: the missing words read as clear.
        let wide = PixelStore::new(32, 4);
        Surface::new(&wide, 4).bitmap(0, 0, 32, 2, &[u32::MAX], 1);
        assert_eq!(lit(&wide, 4).len(), 32);
    }
}
