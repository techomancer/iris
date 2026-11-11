//! The X server's 2D fills (Board::fill_rect, clear_planes) store what
//! Board::paint_integer() stores a pixel at a time: the same window tests
//! (plane bound, clip list, id test), logical operations, plane mask and
//! buffers, into the colour planes and into the window-id buffer's bytes.
//! With `VPRO_GPU=1` the fills run as one record per buffer on the GPU
//! unit and are checked against the per-pixel path on the same unit.

use crate::dev::odyssey::board::Board;
use crate::dev::odyssey::clip::ClipRect;
use crate::dev::odyssey::consts::{
    mode, ODSY_DRAW_MASK_BUFFER0, ODSY_DRAW_MASK_BUFFER1, ODSY_ID_COMPARE_ENABLE, ODSY_ID_COMPARE_MASK_SHIFT, ODSY_PLANE_MASK_SHIFT,
};
use crate::dev::odyssey::raster_bench::Bench;

/// The window-id buffer as the destination (the B format).
const WID_STATUS: u32 = 0x80;
const WID_BASE: u32 = 0x14_2000;

/// The corner of the screen the cases draw in.
const W: i32 = 72;
const H: i32 = 40;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// One case's state, set on a board the same way twice.
struct Case {
    id_planes: bool,
    seed_words: Vec<(u32, u32, u32)>,
    seed_ids: Vec<(u32, u8, u8)>,
    clips: Vec<ClipRect>,
    id_compare: u32,
    draw_mask: u32,
    plane_mask: u8,
    logic: u32,
    pattern_mode: u32,
    is_index: bool,
    value: u32,
    rect: (u32, u32, u32, u32),
}

impl Case {
    fn new(rng: &mut Rng) -> Case {
        let mut seed_words = Vec::new();
        let mut seed_ids = Vec::new();
        for _ in 0..400 {
            seed_words.push((rng.below((W * H) as u32), rng.below(2), rng.next()));
            seed_ids.push((rng.below((W * H) as u32), rng.below(2) as u8, rng.below(4) as u8));
        }
        let clips = (0..1 + rng.below(4))
            .map(|_| {
                let (x0, y0) = (rng.below(W as u32) as i32 - 4, rng.below(H as u32) as i32 - 4);
                ClipRect { x0, y0, x1: x0 + rng.below(40) as i32, y1: y0 + rng.below(24) as i32 }
            })
            .collect();
        let id_compare = match rng.below(3) {
            0 => 0,
            _ => ODSY_ID_COMPARE_ENABLE | (1 + rng.below(3)) << ODSY_ID_COMPARE_MASK_SHIFT | rng.below(4),
        };
        let draw_mask = [ODSY_DRAW_MASK_BUFFER0, ODSY_DRAW_MASK_BUFFER1, ODSY_DRAW_MASK_BUFFER0 | ODSY_DRAW_MASK_BUFFER1][rng.below(3) as usize];
        let (x, y) = (rng.below(W as u32), rng.below(H as u32));
        Case {
            id_planes: rng.below(3) == 0,
            seed_words,
            seed_ids,
            clips,
            id_compare,
            draw_mask,
            plane_mask: if rng.below(2) == 0 { 0xff } else { rng.next() as u8 },
            logic: rng.below(16),
            pattern_mode: [0, mode::INDEX_LOGIC_OP, mode::COLOR_LOGIC_OP, mode::INDEX_LOGIC_OP | mode::COLOR_LOGIC_OP][rng.below(4) as usize],
            is_index: rng.below(2) == 0,
            value: rng.next(),
            rect: (x, y, 1 + rng.below(W as u32 - x + 8), 1 + rng.below(H as u32 - y + 8)),
        }
    }

    fn board(&self) -> Box<Board> {
        let mut s = Bench::empty().board;
        {
            let planes = s.dest_mem();
            for &(i, b, w) in &self.seed_words {
                planes.colour[b as usize].store_word((i % W as u32) as i32, (i / W as u32) as i32, w);
            }
            for &(i, b, v) in &self.seed_ids {
                super::set_aux(&mut s, b as u32, (i / W as u32) * 1280 + i % W as u32, v);
            }
        }
        if self.id_planes {
            s.pixel_mode = [WID_BASE, 0x50, WID_STATUS, 0];
        }
        s.win.clip_kept = self.clips.len() as u32;
        for (i, r) in self.clips.iter().enumerate() {
            s.win.clip_rect[i] = *r;
        }
        s.win.id_compare = self.id_compare;
        s.draw_mask = self.draw_mask;
        s.plane_write_mask = (self.plane_mask as u32) << ODSY_PLANE_MASK_SHIFT;
        s.logic_op_word = self.logic;
        s.stream.pattern_mode = self.pattern_mode;
        s
    }
}

/// The two boards' memory is the same, all 128 MB of it. Compared as two
/// slices rather than through `memory_hash`: hashing both boards in an
/// unoptimised build took most of each case's time.
fn assert_same_memory(fill: &Board, pixels: &Board, what: impl Fn() -> String) {
    let (fill, pixels) = (fill.vram.ptr_shared(), pixels.vram.ptr_shared());
    let (a, b) = (fill.words_at(0, fill.size()), pixels.words_at(0, pixels.size()));
    if a != b {
        let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap();
        panic!("{}: the word at {:#x} is {:#018x} after the fill, {:#018x} pixel by pixel", what(), 8 * i, a[i], b[i]);
    }
}

#[test]
fn a_rectangle_fill_stores_what_its_pixels_would() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for case in 0..48 {
        let c = Case::new(&mut rng);
        let (x, y, w, h) = c.rect;

        let mut fill = c.board();
        fill.fill_rect(x, y, w, h, c.is_index, c.value);

        let mut pixels = c.board();
        for py in y..y + h {
            for px in x..x + w {
                pixels.paint_integer(px as i64, py as i64, c.is_index, c.value);
            }
        }
        fill.gpu_sync();
        pixels.gpu_sync();
        assert_same_memory(&fill, &pixels, || format!("case {case}: id planes {}, clips {}, id test {:#x}, draw mask {:#x}, plane mask {:#x}, op {}, mode {:#x}, index {}", c.id_planes, c.clips.len(), c.id_compare, c.draw_mask, c.plane_mask, c.logic, c.pattern_mode, c.is_index));
    }
}

#[test]
fn an_id_plane_write_changes_the_masked_bits_of_the_pixels_the_window_passes() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for case in 0..32 {
        let mut c = Case::new(&mut rng);
        c.id_planes = true;
        let (x, y, w, h) = c.rect;
        let (value, mask) = (c.value as u8, (c.value >> 8) as u8);

        let mut fill = c.board();
        fill.clear_planes(x, y, w, h, value, mask);

        let pixels = c.board();
        let buffers = pixels.draw_buffers();
        for py in y as i64..(y + h) as i64 {
            for px in x as i64..(x + w) as i64 {
                if !pixels.pixel_allowed(px, py) {
                    continue;
                }
                let planes = pixels.dest_mem();
                for b in 0..2 {
                    if buffers & (1 << b) != 0 {
                        let rows = planes.colour[b];
                        let old = rows.load1(px as i32, py as i32).unwrap();
                        rows.store1(px as i32, py as i32, (old & !mask) | (value & mask));
                    }
                }
            }
        }
        fill.gpu_sync();
        pixels.gpu_sync();
        assert_same_memory(&fill, &pixels, || format!("case {case}: clips {}, id test {:#x}, draw mask {:#x}", c.clips.len(), c.id_compare, c.draw_mask));
    }
}
