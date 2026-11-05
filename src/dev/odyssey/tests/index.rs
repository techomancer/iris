//! Colour-index and overlay GL drawing through the fragment pipeline: the
//! index lands in the destination's index field under the index mask and
//! the index logical operation, and an overlay context's draws and clears
//! reach the overlay byte of the window-id buffer and nothing else.

use crate::dev::odyssey::board::{Board, ODSY_HEIGHT, ODSY_WIDTH};
use crate::dev::odyssey::consts::{colour, mode, write, ODSY_DRAW_MASK_BUFFER0, ODSY_DRAW_MASK_BUFFER1};
use crate::dev::odyssey::gl::gl_index_rgba;
use crate::dev::odyssey::pipe::encode::{vtx, GL_COLOR_BUFFER_BIT, GL_FLAT, GL_TRIANGLES, LibDraw, LibState};
use crate::dev::odyssey::raster_bench::Bench;
use crate::dev::odyssey::tests::{aux_at, set_aux};

/// The test screen's window as a colour-index visual: the same fields,
/// holding an index.
const INDEX_STATUS: u32 = 0x0040_6d50 | (1 << 19);
/// The window-id buffer's pixel mode status (the B format).
const WID_STATUS: u32 = 0x80;
const WID_BASE: u32 = 0x14_2000;

fn index_board() -> (Box<Board>, LibState) {
    let mut b = Bench::empty();
    b.board.pixel_mode[2] = INDEX_STATUS;
    b.board.colour_mode[0] |= colour::MODE_INDEX;
    b.board.gl_write_mask |= write::INDEX_MASK;
    b.st.shade_model = GL_FLAT;
    (b.board, b.st)
}

/// The board as an overlay context draws: the window-id buffer selected
/// and bound as the draw drawable (Board::gl_to_aux).
fn overlay_board() -> (Box<Board>, LibState) {
    let (mut s, st) = index_board();
    s.pixel_mode = [WID_BASE, 0x50, WID_STATUS, 0];
    s.dest_base = WID_BASE;
    s.draw_word = 0x8000_0000 | WID_BASE;
    assert!(s.gl_to_aux());
    (s, st)
}

/// The id the window's id test passes (any, with the test off).
fn window_id(s: &Board) -> u8 {
    s.win.id_compare as u8
}

/// A flat triangle over the screen's top-left half, in index `index`.
fn index_triangle(s: &mut Board, st: &LibState, index: u32) {
    index_triangle_then(s, st, index, |_| {});
}

/// index_triangle, with words the draw state does not carry set by
/// `words` once it is encoded.
fn index_triangle_then(s: &mut Board, st: &LibState, index: u32, words: impl FnOnce(&mut Board)) {
    let mut v = vec![0.0f32; 3 * vtx::FLOATS];
    for (i, (x, y)) in [(-1.0f32, -1.0f32), (1.0, -1.0), (-1.0, 1.0)].into_iter().enumerate() {
        let o = &mut v[i * vtx::FLOATS..][..vtx::FLOATS];
        o[vtx::POSITION..][..4].copy_from_slice(&[x, y, 0.0, 1.0]);
        o[vtx::COLOR..][..4].copy_from_slice(&gl_index_rgba(index));
    }
    let stipple = [0u32; 32];
    let mut d = LibDraw::new(GL_TRIANGLES, 3, &v, st, &stipple);
    s.encode_lib_draw(&d);
    words(s);
    s.lib_draw_encoded(&mut d);
}

fn index_at(s: &Board, buffer: usize, x: i32, y: i32) -> u32 {
    s.raster_wait();
    s.dest_mem().colour[buffer].load_word(x, y).unwrap_or(0)
}

/// Clear the colour buffers to index `index` the way glClear does.
fn index_clear(s: &mut Board, index: u32) {
    let mut op = Bench::clear_op(GL_COLOR_BUFFER_BIT);
    op.clear_rgba = gl_index_rgba(index);
    op.ntargets = 0;
    for b in 0..2 {
        if s.draw_buffers() & (1 << b) != 0 {
            op.targets[op.ntargets as usize] = b;
            op.ntargets += 1;
        }
    }
    s.gl_raster(&op);
}

#[test]
fn a_colour_index_triangle_stores_its_index() {
    let (mut s, st) = index_board();
    index_triangle(&mut s, &st, 0x5a3);
    assert_eq!(index_at(&s, 0, 10, 10), 0x5a3);
    assert_eq!(index_at(&s, 0, ODSY_WIDTH as i32 - 10, ODSY_HEIGHT as i32 - 10), 0, "outside the triangle");
}

#[test]
fn the_index_mask_keeps_the_bits_it_clears() {
    let (mut s, st) = index_board();
    index_triangle(&mut s, &st, 0x123);
    s.gl_write_mask = (s.gl_write_mask & !write::INDEX_MASK) | 0x0f0;
    index_triangle(&mut s, &st, 0xabc);
    assert_eq!(index_at(&s, 0, 10, 10), 0x1b3);
}

#[test]
fn the_index_logical_operation_combines_with_the_stored_index() {
    let (mut s, st) = index_board();
    index_triangle(&mut s, &st, 0x0ff);
    index_triangle_then(&mut s, &st, 0x00f, |s| {
        s.stream.pattern_mode |= mode::INDEX_LOGIC_OP;
        s.logic_op_word = 6; // XOR
    });
    assert_eq!(index_at(&s, 0, 10, 10), 0x0f0);
}

#[test]
fn an_index_clear_stores_the_index_under_the_index_mask() {
    let (mut s, _) = index_board();
    index_clear(&mut s, 0x456);
    assert_eq!(index_at(&s, 0, 600, 900), 0x456);
    s.gl_write_mask = (s.gl_write_mask & !write::INDEX_MASK) | 0x00f;
    index_clear(&mut s, 0x789);
    assert_eq!(index_at(&s, 0, 600, 900), 0x459);
}

#[test]
fn an_overlay_draw_reaches_the_overlay_byte_and_not_the_id_byte() {
    let (mut s, st) = overlay_board();
    let i = 10 * 1280 + 10;
    let id = window_id(&s);
    set_aux(&mut s, 0, i, id);
    // Both buffers: the id byte's share is refused.
    index_triangle_then(&mut s, &st, 0x3c5, |s| s.draw_mask = ODSY_DRAW_MASK_BUFFER0 | ODSY_DRAW_MASK_BUFFER1);
    assert_eq!(aux_at(&s, 1, i), 0xc5, "the index's low byte in the overlay byte");
    assert_eq!(aux_at(&s, 0, i), id, "the id byte is the X server's");
}

#[test]
fn an_overlay_clear_reaches_the_overlay_byte_and_not_the_id_byte() {
    let (mut s, _) = overlay_board();
    let i = 900 * 1280 + 600;
    let id = window_id(&s);
    set_aux(&mut s, 0, i, id);
    s.draw_mask = ODSY_DRAW_MASK_BUFFER0 | ODSY_DRAW_MASK_BUFFER1;
    index_clear(&mut s, 0x7);
    assert_eq!(aux_at(&s, 1, i), 0x07);
    assert_eq!(aux_at(&s, 0, i), id);
}
