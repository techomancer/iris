//! What the display shows: swaps and the displayed-buffer bank, the two
//! colour buffers, colour resolved at scanout, the cursor, the channel
//! size and what a reset clears.

use super::*;

const WIDTH: i32 = SGI_ODYSSEY_WIDTH as i32;
/// The display mode of a drawable showing 24-bit RGB through the gamma
/// ramp, and of one showing colour map group 17 (the X server's modes).
const DISPMODE_RGB: u32 = 0x0052_15a6;
const DISPMODE_GROUP17: u32 = 0x004b_5f10;
const LUT_RED: u32 = 0x3ff0_0000;
const LUT_BLUE: u32 = 0x0000_03ff;

fn backend(s: &mut Board, word: u32) {
    s.mmio_write(ODSY_BACKEND_SELECT, word as u64, 4);
}

/// A record of one back-end word at `index`.
fn backend_entry(s: &mut Board, index: u32, word: u32) {
    backend(s, 0x3000_0000 | index << 14 | 1);
    backend(s, word);
}

fn dispmode(s: &mut Board, did: u32, mode: u32) {
    backend_entry(s, ODSY_DISPMODE_BASE + did, mode);
}

/// What a guest loads before any RGB pixel is visible: the identity gamma
/// ramp at lookup-RAM 0x1a00 and drawable `did` in an RGB mode. Out of
/// reset the lookup RAM is zero and the screen black.
fn display_identity(s: &mut Board, did: u32) {
    dispmode(s, did, DISPMODE_RGB);
    backend(s, 0x3000_0000 | 0x1a00 << 14 | 0x600);
    for i in 0..0x600u32 {
        let v = if i < 0x200 {
            i >> 2
        } else if i < 0x400 {
            ((i - 0x200) >> 1) + 0x80
        } else {
            (i - 0x300) + 0x100
        };
        backend(s, v << 20 | v << 10 | v);
    }
}

/// The display's pixel at (x, y) as 0x00RRGGBB, after the dirty lines
/// have been resolved.
fn shown(s: &mut Board, x: i32, y: i32) -> u32 {
    s.scanout_update();
    s.scanout.frame[(y * s.scanout.frame_width as i32 + x) as usize] & 0xff_ffff
}

/// Put main drawable `id` (bits 7:3 of a pixel's id byte) on every pixel of the box [x0, x1) x [y0, y1).
fn drawable(s: &mut Board, id: u8, x0: i32, y0: i32, x1: i32, y1: i32) {
    for y in y0..y1 {
        for x in x0..x1 {
            s.did[(y * WIDTH + x) as usize] = id << ODSY_DID_MAIN_SHIFT;
        }
    }
    s.scanout.invalidate(y0, y1 - y0);
}

/// Colour index `index` on every pixel of the box, as the plane keeps it.
fn index_box(s: &mut Board, index: u32, x0: i32, y0: i32, x1: i32, y1: i32) {
    for y in y0..y1 {
        for x in x0..x1 {
            pixel_write(s, x, y, index);
        }
    }
}

/// One solid RGB QUADS fill of the half-open box [x0, x1) x [y0, y1).
#[allow(clippy::too_many_arguments, reason = "a colour and a box")]
fn rgb_fill(s: &mut Board, r: u32, g: u32, b: u32, x0: u32, y0: u32, x1: u32, y1: u32) {
    cmd(s, select(ODSY_REG_GL_BEGIN));
    cmd(s, ODSY_GL_QUADS);
    cmd(s, ODSY_GL_COLOR_3UB);
    for c in [r, g, b] {
        cmd(s, c);
    }
    for (x, y) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] {
        cmd(s, ODSY_GL_VERTEX_2I);
        cmd(s, x);
        cmd(s, y);
    }
    cmd(s, select(ODSY_REG_GL_END));
}

fn draw_mask(s: &mut Board, mask: u32) {
    cmd(s, select(ODSY_REG_DRAW_MASK));
    cmd(s, mask);
}

/// The vertical retrace, as the scanout's timer fires it.
fn retrace(s: &mut Board) {
    s.retrace();
}

/// The retrace interrupt carries the retrace byte's vector, not the
/// softsync byte's, and fires at every retrace once routing is programmed
/// whether or not a swap is armed: the driver's vblank counters advance
/// only in that handler. Before routing, a retrace sends nothing.
///
/// QEMU: swap-retrace-vector.
#[test]
fn a_retrace_interrupts_on_its_own_vector_once_routed() {
    let (mut s, rec) = board("v12");
    retrace(&mut s);
    assert!(rec.silent());

    route_to_heart(&mut s);
    retrace(&mut s);
    assert_eq!(rec.events(RETRACE_VECTOR), [true]);
    assert!(rec.events(SOFTSYNC_VECTOR).is_empty());
    retrace(&mut s);
    assert_eq!(rec.events(RETRACE_VECTOR), [true, true], "every retrace, armed or not");
}

/// The arm is a toggle mask latched at the retrace: the window-24 bit
/// flips bit 24 of the displayed-buffer status, a second swap flips it
/// back, and a two-window mask flips both. A retrace with nothing armed
/// flips nothing, and the buffer-select doublewords the driver pushes
/// ahead of the arm neither flip a buffer nor raise an interrupt.
///
/// QEMU: swap-arm-toggles-displayed-buffer.
#[test]
fn an_armed_swap_toggles_the_displayed_buffer_at_the_retrace() {
    const ARM_BIT: u32 = 0x0100_0000;
    let (mut s, rec) = board("v12");
    route_to_heart(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0);

    cmd64(&mut s, ODSY_SWAP_SYNC_A, 0x8000_2000);
    cmd64(&mut s, ODSY_SWAP_SYNC_B, 0x8000_2000);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0);
    assert!(rec.silent());
    assert_eq!(s.reported_cmd_count, 0);

    reg_write(&mut s, ODSY_SWAP_ARM, ARM_BIT);
    assert_eq!(reg_read(&mut s, ODSY_SWAP_ARM), ARM_BIT);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0, "nothing flips before the retrace");
    retrace(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_SWAP_ARM), 0);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), ARM_BIT);
    assert!(rec.level(RETRACE_VECTOR));

    retrace(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), ARM_BIT, "an idle retrace flips nothing");

    reg_write(&mut s, ODSY_SWAP_ARM, ARM_BIT);
    retrace(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0);

    reg_write(&mut s, ODSY_SWAP_ARM, 0x5);
    retrace(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0x5);
}

/// Arm writes between two retraces accumulate: separate windows issued
/// before one retrace all flip there, and the interrupt handler's re-arm
/// of a window still pending does not cancel the others.
///
/// QEMU: swap-arm-accumulates.
#[test]
fn arm_writes_between_two_retraces_accumulate() {
    let (mut s, _) = board("v12");
    route_to_heart(&mut s);

    reg_write(&mut s, ODSY_SWAP_ARM, 0x1);
    reg_write(&mut s, ODSY_SWAP_ARM, 0x0100_0000);
    assert_eq!(reg_read(&mut s, ODSY_SWAP_ARM), 0x0100_0001);
    reg_write(&mut s, ODSY_SWAP_ARM, 0x0100_0000);
    assert_eq!(reg_read(&mut s, ODSY_SWAP_ARM), 0x0100_0001, "a re-arm leaves the mask as it was");

    retrace(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_SWAP_ARM), 0);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0x0100_0001);
}

/// For the windows whose swap is due the kernel writes those that must
/// show buffer 1 to the set register and the rest to the companion. The
/// selects latch at the next retrace and not before; the last write for a
/// window before the retrace wins.
///
/// QEMU: swap-set-and-clear-at-retrace.
#[test]
fn set_and_clear_selects_latch_at_the_next_retrace() {
    let (mut s, _) = board("v12");

    reg_write(&mut s, ODSY_SWAP_ARM, 0x3);
    retrace(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0x3);

    reg_write(&mut s, ODSY_SWAP_CONTROL_COMPANION, 0xffff_ffff);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0x3);
    retrace(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0);

    reg_write(&mut s, ODSY_SWAP_CONTROL, 0x8000_0010);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0);
    retrace(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0x8000_0010);
    retrace(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0x8000_0010, "an idle retrace keeps the selects");

    reg_write(&mut s, ODSY_SWAP_CONTROL_COMPANION, 0x10);
    reg_write(&mut s, ODSY_SWAP_CONTROL, 0x11);
    reg_write(&mut s, ODSY_SWAP_CONTROL_COMPANION, 0x8000_0001);
    retrace(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 0x10);
}

/// The drawable-register block as a context switch sends it: eight
/// doublewords of a word and its one operand, 0x608 to 0x60f. Each takes
/// its operand, so none is parsed as a command and a fill sent after them
/// still draws.
///
/// QEMU: command-fifo-drawable-words.
#[test]
fn the_drawable_register_block_takes_its_operands() {
    let (mut s, _) = board("v12");
    for (word, operand) in [
        (0x608, 0xff00_ff00),
        (0x60a, 0x0268_0008),
        (0x609, 0xff00_ff00),
        (0x60b, 0x0268_0008),
        (0x60c, 0x004c_aef3),
        (0x60d, 0x004b_6a8b),
        (0x60e, 0),
        (0x60f, 0),
    ] {
        cmd64(&mut s, word, operand);
    }
    rgb_fill(&mut s, 0, 0, 0xff, 64, 100, 80, 110);

    assert_eq!(pixel(&s, 0, 64, 100), 0x0000_ffff);
    assert_eq!(s.reported_cmd_count, 0);
}

/// The bound drawing (0x608) and reading (0x609) drawable words are kept
/// independently for the readback path, and neither moves drawing: their
/// decoding is not known, so the drawing mask alone selects the buffer.
/// Reset clears both.
///
/// QEMU: buffers-read-and-draw-words-are-independent.
#[test]
fn the_draw_and_read_drawable_words_are_kept_apart() {
    let (mut s, _) = board("v12");

    cmd64(&mut s, ODSY_SWAP_SYNC_A, 0x8000_2000);
    assert_eq!((s.draw_word, s.read_word), (0x8000_2000, 0));
    cmd64(&mut s, ODSY_SWAP_SYNC_B, 0x0000_2000);
    assert_eq!((s.draw_word, s.read_word), (0x8000_2000, 0x0000_2000));

    rgb_fill(&mut s, 0x12, 0x34, 0x56, 0, 0, 1, 1);
    assert_eq!(pixel(&s, 0, 0, 0), 0x1234_56ff);
    assert_eq!(pixel(&s, 1, 0, 0), 0, "the words do not select the buffer");

    s.reset();
    assert_eq!((s.draw_word, s.read_word), (0, 0));
}

/// The same box drawn red into buffer 0 and blue into buffer 1 holds both
/// colours at once. The window shows buffer 0 until its swap latches at a
/// retrace, then buffer 1. A mask naming both buffers writes both, and a
/// zero mask writes neither.
///
/// QEMU: buffers-hold-opposite-colours.
#[test]
fn the_two_colour_buffers_hold_opposite_colours() {
    let (mut s, _) = board("v12");
    display_identity(&mut s, 0);

    rgb_fill(&mut s, 0xff, 0, 0, 100, 100, 110, 110);
    draw_mask(&mut s, 0x4);
    rgb_fill(&mut s, 0, 0, 0xff, 100, 100, 110, 110);
    assert_eq!(pixel(&s, 0, 105, 105), 0xff00_00ff);
    assert_eq!(pixel(&s, 1, 105, 105), 0x0000_ffff);
    assert_eq!(shown(&mut s, 105, 105), 0xff0000);

    reg_write(&mut s, ODSY_SWAP_ARM, 0x1);
    assert_eq!(shown(&mut s, 105, 105), 0xff0000, "an armed swap shows nothing yet");
    retrace(&mut s);
    assert_eq!(shown(&mut s, 105, 105), 0x0000ff);

    draw_mask(&mut s, 0x5);
    rgb_fill(&mut s, 0, 0xff, 0, 100, 100, 102, 102);
    assert_eq!(pixel(&s, 0, 100, 100), 0x00ff_00ff);
    assert_eq!(pixel(&s, 1, 100, 100), 0x00ff_00ff);

    draw_mask(&mut s, 0);
    rgb_fill(&mut s, 0xff, 0xff, 0xff, 100, 100, 110, 110);
    assert_eq!(pixel(&s, 0, 105, 105), 0xff00_00ff);
    assert_eq!(pixel(&s, 1, 105, 105), 0x0000_ffff);
}

/// A double-buffered client draws each frame in two halves into the buffer
/// the window does not show, arms the swap, and only the retrace shows it.
/// The kernel exchanges the mask's pairs as the drawable's buffer state
/// changes, so the client's back buffer alternates. Neither a half-drawn
/// frame nor an armed swap reaches the display before the retrace, over
/// four swaps.
///
/// QEMU: buffers-swap-shows-only-complete-frames.
#[test]
fn a_swap_shows_only_complete_frames() {
    let frames = [(0xff, 0, 0), (0, 0xff, 0), (0, 0, 0xff), (0xff, 0xff, 0)];
    let (mut s, _) = board("v12");
    display_identity(&mut s, 0);
    route_to_heart(&mut s);
    let mut visible = 0u32;

    for (i, (r, g, b)) in frames.into_iter().enumerate() {
        let back = if i & 1 != 0 { 0 } else { 1 };
        let colour = r << 16 | g << 8 | b;
        draw_mask(&mut s, if back == 1 { 0x4 } else { 0x1 });
        rgb_fill(&mut s, r, g, b, 0, 0, 64, 32);
        assert_eq!(shown(&mut s, 10, 10), visible, "frame {i}: first half");
        rgb_fill(&mut s, r, g, b, 0, 32, 64, 64);
        assert_eq!(shown(&mut s, 10, 40), visible, "frame {i}: second half");

        reg_write(&mut s, ODSY_SWAP_ARM, 0x1);
        assert_eq!(shown(&mut s, 10, 10), visible, "frame {i}: armed");
        assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), 1 - back);
        retrace(&mut s);
        assert_eq!(reg_read(&mut s, ODSY_DISPLAYED_BUFFER), back);
        assert_eq!(shown(&mut s, 10, 10), colour, "frame {i}: swapped");
        assert_eq!(shown(&mut s, 10, 40), colour);

        retrace(&mut s);
        assert_eq!(shown(&mut s, 10, 40), colour, "an idle retrace keeps the frame");
        visible = colour;
    }
}

/// The colour plane keeps the index or RGB the guest drew; what the
/// display shows follows the pixel's drawable, its display mode and the
/// lookup RAM as they stand now. The X server's blank zeroes colour map
/// group 17; a pixel painted during the blank must come back when the
/// unblank reloads it, with no repaint.
///
/// QEMU: scanout-wipe-repaint-reload.
#[test]
fn colour_is_resolved_at_scanout_so_a_reloaded_map_brings_pixels_back() {
    let (mut s, _) = board("v12");
    dispmode(&mut s, 1, DISPMODE_GROUP17);
    backend_entry(&mut s, 0x1105, LUT_RED);
    drawable(&mut s, 1, 100, 100, 116, 116);
    index_box(&mut s, 5, 100, 100, 116, 116);
    assert_eq!(shown(&mut s, 108, 108), 0xff0000);

    // The blank: group 17's 256 entries zeroed, no repaint.
    backend(&mut s, 0x3000_0000 | 0x1100 << 14 | 0x100);
    for _ in 0..0x100 {
        backend(&mut s, 0);
    }
    assert_eq!(shown(&mut s, 108, 108), 0);

    // Painted during the blank: black too.
    index_box(&mut s, 5, 100, 100, 116, 116);
    assert_eq!(shown(&mut s, 108, 108), 0);

    // The unblank reloads the entry: red again.
    backend_entry(&mut s, 0x1105, LUT_RED);
    assert_eq!(shown(&mut s, 108, 108), 0xff0000);
    assert_eq!(shown(&mut s, 100, 115), 0xff0000);
}

/// The X server's colour map install rewrites the drawable's display mode;
/// its pixels change colour with no repaint and the plane keeps the index.
///
/// QEMU: scanout-display-mode-recolours.
#[test]
fn rewriting_a_display_mode_recolours_its_pixels() {
    const GROUP18_CI: u32 = 0x004b_5f20;
    let (mut s, _) = board("v12");
    dispmode(&mut s, 1, DISPMODE_GROUP17);
    dispmode(&mut s, 2, GROUP18_CI);
    backend_entry(&mut s, 0x1105, LUT_RED);
    backend_entry(&mut s, 0x1205, LUT_BLUE);
    drawable(&mut s, 1, 100, 100, 116, 116);
    drawable(&mut s, 2, 200, 100, 216, 116);
    index_box(&mut s, 5, 100, 100, 116, 116);
    index_box(&mut s, 5, 200, 100, 216, 116);
    assert_eq!(shown(&mut s, 108, 108), 0xff0000);
    assert_eq!(shown(&mut s, 208, 108), 0x0000ff);

    dispmode(&mut s, 2, DISPMODE_GROUP17);

    assert_eq!(shown(&mut s, 208, 108), 0xff0000);
    assert_eq!(pixel(&s, 0, 208, 108), 5);
}

/// An RGB pixel passes through the gamma table at 0x1a00, which the RGB
/// display modes name as their group: zeroing the table blackens it,
/// reloading brings it back.
///
/// QEMU: scanout-rgb-through-gamma.
#[test]
fn an_rgb_pixel_passes_through_the_gamma_table() {
    let (mut s, _) = board("v12");
    display_identity(&mut s, 8);
    drawable(&mut s, 8, 100, 100, 116, 116);
    rgb_fill(&mut s, 0x11, 0x22, 0x33, 100, 100, 116, 116);
    assert_eq!(pixel(&s, 0, 108, 108), 0x1122_33ff);
    assert_eq!(shown(&mut s, 108, 108), 0x112233);

    backend(&mut s, 0x3000_0000 | 0x1a00 << 14 | 0x600);
    for _ in 0..0x600 {
        backend(&mut s, 0);
    }
    assert_eq!(shown(&mut s, 108, 108), 0);

    display_identity(&mut s, 8);
    assert_eq!(shown(&mut s, 108, 108), 0x112233);
}

/// Pixels painted first and the drawable and colour map programmed after
/// still show their colour.
///
/// QEMU: scanout-paint-order.
#[test]
fn pixels_painted_before_the_map_is_programmed_show_it_afterwards() {
    let (mut s, _) = board("v12");
    drawable(&mut s, 1, 100, 100, 116, 116);
    index_box(&mut s, 5, 100, 100, 116, 116);
    assert_eq!(shown(&mut s, 108, 108), 0);

    dispmode(&mut s, 1, DISPMODE_GROUP17);
    backend_entry(&mut s, 0x1105, LUT_RED);

    assert_eq!(shown(&mut s, 108, 108), 0xff0000);
}

/// A raw 0xRRGGBBAA written to the colour plane reaches the display with
/// the alpha byte dropped, through an RGB mode on the screen's drawable.
/// An untouched pixel stays at its post-reset value.
///
/// QEMU: framebuffer-pixel-reaches-scanout.
#[test]
fn a_plane_pixel_reaches_the_display_without_its_alpha() {
    let (mut s, _) = board("v12");
    display_identity(&mut s, 0);
    assert_eq!(pixel(&s, 0, 5, 5), 0);
    assert_eq!(shown(&mut s, 5, 5), 0);

    pixel_write(&mut s, 5, 5, 0x1122_3344);

    assert_eq!(pixel(&s, 0, 5, 5), 0x1122_3344);
    assert_eq!(shown(&mut s, 5, 5), 0x112233);
    assert_eq!(shown(&mut s, 6, 5), 0);
}

/// The display channel's active size: the kernel and the X server write
/// the selected format's pixels and lines to back-end entries 0x2802 and
/// 0x2803. The scanout presents the latest nonzero size, kept within the
/// modelled planes, zeros (a format change) leave the picture as it was,
/// and a reset returns it to 1280x1024.
///
/// QEMU: display-channel-size-sets-scanout.
#[test]
fn the_display_channel_size_sets_the_scanout_mode() {
    let (mut s, _) = board("v12");
    let size = |s: &Board| (s.scanout.mode.width, s.scanout.mode.height);
    assert_eq!(size(&s), (1280, 1024));

    backend(&mut s, 0x3000_0000 | 0x2802 << 14 | 5);
    for w in [1280, 720, 0, 0, 0] {
        backend(&mut s, w);
    }
    assert_eq!(size(&s), (1280, 720));
    pixel_write(&mut s, 5, 5, 0x1122_3344);
    display_identity(&mut s, 0);
    assert_eq!(shown(&mut s, 5, 5), 0x112233);

    // The zeros of a format change leave the picture as it was.
    backend(&mut s, 0x3000_0000 | 0x2802 << 14 | 2);
    backend(&mut s, 0);
    backend(&mut s, 0);
    assert_eq!(size(&s), (1280, 720));

    // A narrower channel presents the left of each line.
    backend(&mut s, 0x3000_0000 | 0x2802 << 14 | 2);
    backend(&mut s, 1024);
    backend(&mut s, 768);
    assert_eq!(size(&s), (1024, 768));
    assert_eq!(shown(&mut s, 5, 5), 0x112233);

    // Larger than the modelled planes: kept to them.
    backend(&mut s, 0x3000_0000 | 0x2802 << 14 | 2);
    backend(&mut s, 1600);
    backend(&mut s, 1200);
    assert_eq!(size(&s), (1280, 1024));

    backend_entry(&mut s, ODSY_CHANNEL_HEIGHT, 720);
    s.reset();
    assert_eq!(size(&s), (1280, 1024));
}

/// A reset clears the framebuffer along with the registers.
///
/// QEMU: reset-clears-the-framebuffer.
#[test]
fn a_reset_clears_the_framebuffer() {
    let (mut s, _) = board("v12");
    pixel_write(&mut s, 5, 5, 0x1122_3344);
    assert_eq!(pixel(&s, 0, 5, 5), 0x1122_3344);

    s.reset();

    assert_eq!(pixel(&s, 0, 5, 5), 0);
    assert_eq!(shown(&mut s, 5, 5), 0);
}

/// A guest reset that lands with a primitive still open (a BEGIN with no
/// END) completes and leaves the board responsive.
///
/// QEMU: reset-mid-primitive-does-not-hang.
#[test]
fn a_reset_with_a_primitive_open_leaves_the_board_responsive() {
    let (mut s, _) = board("v12");
    for w in [select(ODSY_REG_GL_BEGIN), ODSY_GL_TRIANGLES, ODSY_GL_VERTEX_2F, 0x4220_0000, 0x4220_0000] {
        s.mmio_write(ODSY_USER_CMD_FIFO, w as u64, 4);
    }

    s.reset();

    assert_eq!(reg_read(&mut s, CONTROL), VPRO_WIDGET);
    rgb_fill(&mut s, 1, 2, 3, 0, 0, 4, 4);
    assert_eq!(pixel(&s, 0, 1, 1), 0x0102_03ff, "the next primitive draws");
}

// The hardware cursor.

const CURSOR_BACKGROUND: u32 = 0x4060_80ff;
const WORD0_RGB: u32 = 0xff0000;
const WORD1_RGB: u32 = 0x00b100;
const WORD2_RGB: u32 = 0x0000ff;

fn cursor_entry(s: &mut Board, index: u32, word: u32) {
    backend_entry(s, ODSY_CURSOR_BASE + index, word);
}

fn cursor_readback(s: &mut Board, index: u32) -> u32 {
    s.mmio_read(ODSY_BACKEND_READ_BASE + (((ODSY_CURSOR_BASE + index) as u64) << 3), 4) as u32
}

/// An RGB screen with a known colour over the corner the cursor uses, the
/// test glyph, plane and colours, and the cursor at (x, y), still off.
fn cursor_load(s: &mut Board, x: u32, y: u32) {
    display_identity(s, 0);
    for py in 0..8 {
        for px in 0..12 {
            pixel_write(s, px, py, CURSOR_BACKGROUND);
        }
    }
    backend(s, 0x3000_0000 | ODSY_CURSOR_BASE << 14 | 128);
    for i in 0..128 {
        backend(s, if i == 0 { 0x1021 } else { 0 });
    }
    backend(s, 0x3000_0000 | (ODSY_CURSOR_BASE + ODSY_CURSOR_PLANE) << 14 | 64);
    for i in 0..64 {
        backend(s, if i == 0 { 0x3f } else { 0 });
    }
    backend(s, 0x3000_0000 | (ODSY_CURSOR_BASE + ODSY_CURSOR_COLOURS) << 14 | 15);
    for w in [0x3ff << 20, 0x2c4 << 10, 0x3ff] {
        backend(s, w);
    }
    for _ in 3..15 {
        backend(s, 0);
    }
    cursor_entry(s, ODSY_CURSOR_POSITION, (x & 0xffff) << 16 | (y & 0xffff));
}

/// Plane 3 shows glyph value k in colour word k and glyph 0 in word 0;
/// plane 0 is transparent whatever the glyph. The cursor is drawn after
/// colour resolution, and every entry reads back as written.
///
/// QEMU: cursor/glyph-and-plane.
#[test]
fn the_cursor_draws_its_glyph_through_the_plane_over_the_picture() {
    let (mut s, _) = board("v12");
    cursor_load(&mut s, 2, 3);
    let bg = shown(&mut s, 11, 7);
    assert_eq!(shown(&mut s, 2, 3), bg, "the cursor is off");

    cursor_entry(&mut s, ODSY_CURSOR_CONTROL, 1);
    assert_eq!(shown(&mut s, 2, 3), WORD1_RGB);
    assert_eq!(shown(&mut s, 3, 3), WORD2_RGB);
    assert_eq!(shown(&mut s, 4, 3), WORD0_RGB);
    // Glyph 1 and glyph 0 under plane 0 are transparent.
    assert_eq!(shown(&mut s, 5, 3), bg);
    assert_eq!(shown(&mut s, 6, 3), bg);
    assert_eq!(shown(&mut s, 2, 4), bg);
    assert_eq!(shown(&mut s, 2, 2), bg);
    assert_eq!(shown(&mut s, 1, 3), bg);
    assert_eq!(cursor_readback(&mut s, ODSY_CURSOR_GLYPH), 0x1021);
    assert_eq!(cursor_readback(&mut s, ODSY_CURSOR_PLANE), 0x3f);
    assert_eq!(cursor_readback(&mut s, ODSY_CURSOR_COLOURS + 1), 0x2c4 << 10);
    assert_eq!(cursor_readback(&mut s, ODSY_CURSOR_POSITION), 0x0002_0003);
    assert_eq!(cursor_readback(&mut s, ODSY_CURSOR_CONTROL), 1);

    cursor_entry(&mut s, ODSY_CURSOR_CONTROL, 0);
    for x in 2..7 {
        assert_eq!(shown(&mut s, x, 3), bg, "x {x}");
    }
}

/// The position is the glyph's top-left pixel, read as signed 16-bit
/// components and clipped to the raster. A move repaints the old place.
///
/// QEMU: cursor/moves-and-clips.
#[test]
fn the_cursor_position_is_signed_and_clipped_to_the_raster() {
    let (mut s, _) = board("v12");
    cursor_load(&mut s, 2, 3);
    let bg = shown(&mut s, 11, 7);
    cursor_entry(&mut s, ODSY_CURSOR_CONTROL, 1);
    assert_eq!(shown(&mut s, 2, 3), WORD1_RGB);

    // x = -1: column 0 of the glyph is off the left edge.
    cursor_entry(&mut s, ODSY_CURSOR_POSITION, 0xffff_0005);
    assert_eq!(shown(&mut s, 0, 5), WORD2_RGB);
    assert_eq!(shown(&mut s, 1, 5), WORD0_RGB);
    assert_eq!(shown(&mut s, 2, 5), bg);
    assert_eq!(shown(&mut s, 2, 3), bg, "the old place is repainted");
    assert_eq!(shown(&mut s, 3, 3), bg);

    // y = -2: glyph row 2 lands on line 0. Read unsigned, 0xfffe would put
    // the whole cursor below the raster and draw nothing.
    cursor_entry(&mut s, ODSY_CURSOR_GLYPH + 2 * 4, 0x1);
    cursor_entry(&mut s, ODSY_CURSOR_PLANE + 2 * 2, 0x3);
    cursor_entry(&mut s, ODSY_CURSOR_POSITION, 0x0002_fffe);
    assert_eq!(shown(&mut s, 2, 0), WORD1_RGB);
    assert_eq!(shown(&mut s, 3, 0), bg);
    assert_eq!(shown(&mut s, 2, 1), bg);
    assert_eq!(shown(&mut s, 0, 5), bg);

    // The PROM's first position after its enable, x 1280 y 0, is the first
    // column right of the raster and draws nothing.
    cursor_entry(&mut s, ODSY_CURSOR_POSITION, 0x0500_0000);
    assert_eq!(shown(&mut s, 0, 5), bg);
    assert_eq!(shown(&mut s, 2, 0), bg);
    assert_eq!(shown(&mut s, 1279, 0), 0);
}

/// The cursor request writes the position through a type 7 header when it
/// defers it. A deferred position flushed when control bit 5 changes
/// arrives as four records, each applying in turn, so the cursor ends
/// hidden on the second channel with every entry reading back as written.
///
/// QEMU: cursor/deferred-records.
#[test]
fn deferred_cursor_records_apply_in_turn() {
    let (mut s, _) = board("v12");
    cursor_load(&mut s, 2, 3);
    let bg = shown(&mut s, 11, 7);
    let deferred = |s: &mut Board, index: u32, word: u32| {
        backend(s, 0x3000_0000 | (ODSY_CURSOR_BASE + index) << 14 | 1 | 0x4000_0000);
        backend(s, word);
    };

    cursor_entry(&mut s, ODSY_CURSOR_CONTROL, 3);
    deferred(&mut s, ODSY_CURSOR_POSITION, 0x0004_0005);
    assert_eq!(shown(&mut s, 4, 5), WORD1_RGB);
    assert_eq!(shown(&mut s, 2, 3), bg);
    deferred(&mut s, ODSY_CURSOR_CONTROL, 1);
    assert_eq!(shown(&mut s, 4, 5), WORD1_RGB);

    deferred(&mut s, ODSY_CURSOR_CONTROL, 0x21);
    deferred(&mut s, ODSY_CURSOR_POSITION, 0x0002_0003);
    deferred(&mut s, ODSY_CURSOR_CONTROL, 0x20);
    assert_eq!(shown(&mut s, 2, 3), bg);
    assert_eq!(shown(&mut s, 4, 5), bg);
    assert_eq!(cursor_readback(&mut s, ODSY_CURSOR_POSITION), 0x0002_0003);
    assert_eq!(cursor_readback(&mut s, ODSY_CURSOR_CONTROL), 0x20);
}

/// Control bits 0 and 2 are the X server's crosshair: a one-pixel line
/// across the raster each way through the position, in colour word 1. Bit
/// 2 alone draws nothing, and neither does the second display channel.
///
/// QEMU: cursor/crosshair.
#[test]
fn the_crosshair_draws_a_full_raster_line_each_way() {
    let (mut s, _) = board("v12");
    cursor_load(&mut s, 2, 3);
    let bg = shown(&mut s, 11, 7);

    cursor_entry(&mut s, ODSY_CURSOR_CONTROL, 4);
    assert_eq!(shown(&mut s, 2, 3), bg);
    assert_eq!(shown(&mut s, 9, 3), bg);

    cursor_entry(&mut s, ODSY_CURSOR_CONTROL, 5);
    for (x, y) in [(0, 3), (9, 3), (1279, 3), (2, 0), (2, 1023)] {
        assert_eq!(shown(&mut s, x, y), WORD1_RGB, "({x}, {y})");
    }
    assert_eq!(shown(&mut s, 3, 4), bg, "no glyph beside the point");

    cursor_entry(&mut s, ODSY_CURSOR_CONTROL, 0x25);
    assert_eq!(shown(&mut s, 9, 3), bg);
    assert_eq!(shown(&mut s, 2, 6), bg);
}

/// A reset clears every cursor entry, so the cursor is off and its glyph,
/// plane and colours are gone: turning it on draws nothing.
///
/// QEMU: cursor/reset-leaves-it-off.
#[test]
fn a_reset_leaves_the_cursor_off_and_empty() {
    let (mut s, _) = board("v12");
    cursor_load(&mut s, 2, 3);
    cursor_entry(&mut s, ODSY_CURSOR_CONTROL, 1);
    assert_eq!(shown(&mut s, 2, 3), WORD1_RGB);

    s.reset();

    assert_eq!(shown(&mut s, 2, 3), 0);
    for e in [ODSY_CURSOR_CONTROL, ODSY_CURSOR_POSITION, ODSY_CURSOR_GLYPH, ODSY_CURSOR_COLOURS + 1] {
        assert_eq!(cursor_readback(&mut s, e), 0, "{e:#x}");
    }
    cursor_entry(&mut s, ODSY_CURSOR_CONTROL, 1);
    assert_eq!(shown(&mut s, 2, 3), 0);
}
