//! The current-value and raster-position queries (words 0x4c0 and 0x4c3).
//!
//! The expectations come from the odyssey datasheet's "Raster-position
//! query and glyph colour": the raster-position stream carries no colour,
//! the library may send its query after later glColor calls, so the board
//! is modelled as latching index, colour and texture coordinate when the
//! last block of that stream (0x3cfb) arrives (*inferred*), answering
//! kind 3 from the latch and kind 0 live (*established* as software
//! behaviour).

use super::*;

const SHADOW: u64 = 0x2000;
const SYNC: u64 = 0x1000;
const F_ONE: u32 = ODSY_F32_ONE;
const F_HALF: u32 = 0x3f00_0000;
const F_QUARTER: u32 = 0x3e80_0000;

fn word(ram: &Ram, at: u64) -> u32 {
    u32::from_be_bytes(ram.get(at, 4).try_into().unwrap())
}

fn colour(s: &mut Board, r: u32, g: u32, b: u32) {
    for w in [ODSY_GL_COLOR_3F, r, g, b] {
        cmd(s, w);
    }
}

fn index(s: &mut Board, i: u32) {
    cmd(s, ODSY_GL_COLOR_INDEX);
    cmd(s, i);
}

/// The tail of glRasterPos's stream: four class-0 blocks of zero operands.
fn raster_pos_stream(s: &mut Board) {
    for block in 0x3cf8..=0x3cfb {
        cmd(s, block);
        for _ in 0..16 {
            cmd(s, 0);
        }
    }
}

/// Send query word `0x4c0 + kind` with its count, fence and sequence.
fn query(s: &mut Board, kind: u32) {
    for w in [0x4c0 + kind, 7, 0x10a5a, 3, 0x5a5a] {
        cmd(s, w);
    }
}

fn query_board() -> (Box<Board>, Arc<Ram>) {
    let (mut s, _, ram) = board_with_ram();
    s.ctx_shadow_addr_lower = SHADOW as u32;
    s.host_write_addr_lower = SYNC as u32;
    (s, ram)
}

fn raster_colour(ram: &Ram) -> [u32; 4] {
    let at = SHADOW + ODSY_QUERY_RASTER_COLOUR as u64;
    [0, 4, 8, 12].map(|o| word(ram, at + o))
}

/// Colour A, glRasterPos, colour B, then the deferred query: the library
/// reads A, the colour from glRasterPos time. Reading the colour current
/// at query time returns B (what the glstate RASTER_COLOR case saw).
#[test]
fn the_raster_query_returns_the_colour_latched_at_raster_position_time() {
    let (mut s, ram) = query_board();

    colour(&mut s, F_HALF, F_QUARTER, F_ONE);
    index(&mut s, 9);
    raster_pos_stream(&mut s);
    colour(&mut s, 0, 0, 0);
    index(&mut s, 4);
    query(&mut s, ODSY_QUERY_KIND_RASTER);

    assert_eq!(raster_colour(&ram), [ODSY_GL_COLOR_3F, F_HALF, F_QUARTER, F_ONE]);
    let at = SHADOW + ODSY_QUERY_RASTER_INDEX as u64;
    assert_eq!([word(&ram, at), word(&ram, at + 4)], [ODSY_GL_COLOR_INDEX, 9]);
    assert_eq!(ram.half(SYNC + 6), 0x5a5a, "the query still completes");
}

/// A colour change after the latch does not move the latch, and a second
/// raster position takes the new colour.
#[test]
fn a_later_raster_position_replaces_the_latch() {
    let (mut s, ram) = query_board();

    colour(&mut s, F_HALF, F_HALF, F_HALF);
    raster_pos_stream(&mut s);
    colour(&mut s, F_ONE, 0, 0);
    raster_pos_stream(&mut s);
    colour(&mut s, 0, 0, F_ONE);
    query(&mut s, ODSY_QUERY_KIND_RASTER);

    assert_eq!(raster_colour(&ram), [ODSY_GL_COLOR_3F, F_ONE, 0, 0]);
}

/// The latch is taken at the last block, not the first: a colour sent
/// before 0x3cfb but after 0x3cf8 would be in it, one sent after is not.
/// Only the 0x3cfb block latches.
#[test]
fn only_the_last_position_block_latches() {
    let (mut s, ram) = query_board();

    colour(&mut s, F_HALF, F_HALF, F_HALF);
    raster_pos_stream(&mut s);
    colour(&mut s, F_ONE, 0, 0);
    cmd(&mut s, 0x3cf8);
    for _ in 0..16 {
        cmd(&mut s, 0);
    }
    query(&mut s, ODSY_QUERY_KIND_RASTER);

    assert_eq!(raster_colour(&ram), [ODSY_GL_COLOR_3F, F_HALF, F_HALF, F_HALF]);
}

/// Before any glRasterPos the raster records hold the GL initial values:
/// colour (1, 1, 1, 1) and index 1, whatever the current colour is.
#[test]
fn the_raster_query_before_any_raster_position_returns_the_initial_values() {
    let (mut s, ram) = query_board();

    colour(&mut s, F_HALF, F_QUARTER, 0);
    index(&mut s, 7);
    query(&mut s, ODSY_QUERY_KIND_RASTER);

    assert_eq!(raster_colour(&ram), [ODSY_GL_COLOR_3F, F_ONE, F_ONE, F_ONE]);
    let at = SHADOW + ODSY_QUERY_RASTER_INDEX as u64;
    assert_eq!([word(&ram, at), word(&ram, at + 4)], [ODSY_GL_COLOR_INDEX, 1]);
}

/// The current-value query (kind 0) stays live: it reports the colour and
/// index current at the query, not the raster latch.
#[test]
fn the_current_value_query_stays_live() {
    let (mut s, ram) = query_board();

    colour(&mut s, F_HALF, F_QUARTER, F_ONE);
    index(&mut s, 9);
    raster_pos_stream(&mut s);
    colour(&mut s, F_ONE, F_ONE, 0);
    index(&mut s, 4);
    query(&mut s, 0);

    let live = SHADOW + 0x914;
    assert_eq!([0, 4, 8, 12].map(|o| word(&ram, live + o)), [ODSY_GL_COLOR_3F, F_ONE, F_ONE, 0]);
    let idx = SHADOW + 0x900;
    assert_eq!([word(&ram, idx), word(&ram, idx + 4)], [ODSY_GL_COLOR_INDEX, 4]);
    assert_eq!(word(&ram, SHADOW + ODSY_QUERY_RASTER_COLOUR as u64), 0, "kind 0 writes no raster record");
}
