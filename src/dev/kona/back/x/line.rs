//! The X server's lines: segments and polylines.
//!
//! Segments (0x159 with one, 0x15a with six) are independent pairs of points,
//! `x << 16 | y` with signed halves, each point added to the drawable's position.
//! Polylines (0x15b with one word, 0x15c with seven) join their points in order, a
//! longer one arriving as n - 1 words of data preload and the one-word token for its last
//! point (the sender counts are established, the pipe's consumption is inferred);
//! relative polylines (0x15d with two, 0x15e with seven) take each point as a step from the
//! previous one, the first from the drawable's origin, as relative points do. The senders
//! do not zero the position (386 of the 1192 position loads in the desktop recording are
//! nonzero); the position loads are established and the add to every vertex, absolute
//! and relative, is inferred.
//! The token rows and their word counts are established except for the burst
//! of six segments (0x15a), which is inferred. What the segment token (0x159)
//! draws is unknown and what the polyline tokens draw is inferred; the framing of
//! the two polyline bursts is the shipped X server's wire contract (seven words),
//! established.
//! Lines draw under the context as it stands, through the clip, scissor,
//! stipple, write mask and logic operation, as a request checked whole before
//! any pixel is written. A polyline takes the points of a waiting data run
//! before its own, as the X server sends a long one (the sender counts are established, the
//! pipe's consumption inferred); a segment token
//! does not, and ends a run. A relative polyline starts from the drawable's
//! origin with every command, as relative points do (established).
//!
//! Which pixels a line lights is unknown at the device level: the engine's
//! line drawer is known only to operation counts, so the tie and rounding rule, whether the last end point is lit, whether a
//! segment of no length lights a pixel and whether a vertex shared by two
//! segments is lit once are all unread. The evidence once offered for a
//! one-pixel segment of no length is a line of width zero, not a segment of
//! no length. What the model does is therefore a labelled approximation, not a
//! reading: a segment draws both end points, a segment of no length draws one
//! pixel, a vertex shared by two segments of a polyline is drawn once, and a
//! polyline is not closed unless the caller's last point is its first (that
//! it has no implicit close is inferred). The line is the integer line through
//! pixel centres, and where it falls exactly between two pixels of the minor
//! axis it takes the one nearer the end point. A real pipe's choice would show in a short line of odd slope drawn
//! by the X server and read back, or in the pipe's line routine if it were
//! read. Also not known: whether the last vertex of a polyline is drawn or is
//! left to the next command that shares it (the model draws it), whether a
//! closing vertex that is the first point is drawn twice (the model does).
//!
//! Ported from sgi_kona_x_lines in hw/misc/sgi-kona.c in
//! jamesbraid/qemu-sgi-origami, which draws the same shapes by an error term
//! and a coordinate wrap; this walks the major axis and keeps the minor axis's
//! remainder.
//!
//! # Cost
//!
//! A line is cut into boxes before the request is planned: a run of pixels
//! along a row or column is one box, and pixels outside the screen are not
//! made. The pixel loops and the destination check are the fills' own.

use crate::dev::kona::defs::*;
use super::fill::Box4;
use super::state::XState;
use crate::dev::kona::back::raster::Surface;

pub const X_SEGMENT: u32 = DRAW_LINE_SEGMENTS;
pub const X_SEGMENT6: u32 = DRAW_LINE_SEGMENTS_BURST_OF_6;
pub const X_LINE: u32 = DRAW_CONNECTED_POLYLINE;
pub const X_LINE6: u32 = DRAW_CONNECTED_POLYLINE_BURST_OF_6;
pub const X_RLINE: u32 = DRAW_CONNECTED_POLYLINE_RELATIVE_POINTS;
pub const X_RLINE6: u32 = DRAW_CONNECTED_POLYLINE_RELATIVE_POINTS_BURST_OF_6;

/// Whether `selector` is a segment or polyline token.
pub fn is_line(selector: u32) -> bool {
    matches!(selector, X_SEGMENT | X_SEGMENT6 | X_LINE | X_LINE6 | X_RLINE | X_RLINE6)
}

/// Whether `selector` is one of the four polyline tokens (absolute or relative, one segment or a burst).
pub fn is_polyline(selector: u32) -> bool {
    matches!(selector, X_LINE | X_LINE6 | X_RLINE | X_RLINE6)
}

fn point(w: u32) -> (i64, i64) {
    ((w >> 16) as u16 as i16 as i64, w as u16 as i16 as i64)
}

/// Add the pixel (`x`, `y`) to `boxes`, widening the last box when it is a
/// row or column the pixel continues, and not at all when it is off screen.
fn plot(boxes: &mut Vec<Box4>, bounds: &super::state::Bounds, x: i64, y: i64) {
    if x < bounds.x1 || x >= bounds.x2 || y < bounds.y1 || y >= bounds.y2 {
        return;
    }
    if let Some(last) = boxes.last_mut() {
        let column = last.0 == x && last.2 == x + 1;
        let row = last.1 == y && last.3 == y + 1;
        if column && y == last.3 {
            last.3 += 1;
            return;
        }
        if column && y + 1 == last.1 {
            last.1 = y;
            return;
        }
        if row && x == last.2 {
            last.2 += 1;
            return;
        }
        if row && x + 1 == last.0 {
            last.0 = x;
            return;
        }
    }
    boxes.push((x, y, x + 1, y + 1));
}

/// The pixels of the line from `a` to `b`, in order from the one after `a`
/// (and `a` too when `first`). The major axis steps once a pixel; the minor
/// axis takes `floor((n + 2 i m) / 2n)` steps by pixel `i`, with `n` and `m`
/// the major and minor extents, which rounds an exact half toward `b`.
fn walk(boxes: &mut Vec<Box4>, bounds: &super::state::Bounds, a: (i64, i64), b: (i64, i64), first: bool) {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let (n, m) = (dx.abs().max(dy.abs()), dx.abs().min(dy.abs()));
    if first {
        plot(boxes, bounds, a.0, a.1);
    }
    let x_major = dx.abs() >= dy.abs();
    let (major_step, minor_step) = if x_major { (dx.signum(), dy.signum()) } else { (dy.signum(), dx.signum()) };
    let (mut major, mut minor, mut acc) = (0i64, 0i64, n);
    for _ in 0..n {
        major += major_step;
        acc += 2 * m;
        if acc >= 2 * n {
            acc -= 2 * n;
            minor += minor_step;
        }
        if x_major {
            plot(boxes, bounds, a.0 + major, a.1 + minor);
        } else {
            plot(boxes, bounds, a.0 + minor, a.1 + major);
        }
    }
}

impl XState {
    /// A segment or polyline token. The words of a waiting data run are not
    /// part of a line: the arrival of this token has already dropped them.
    pub(super) fn lines(&mut self, surface: &Surface<'_>, selector: u32, args: &[u32]) {
        let (width, height) = surface.size();
        let bounds = self.bounds(width, height);
        let mut boxes = std::mem::take(&mut self.boxes);
        boxes.clear();
        if matches!(selector, X_SEGMENT | X_SEGMENT6) {
            let (ox, oy) = self.origin();
            for pair in args.chunks_exact(2) {
                // Both ends, and a single pixel when they are the same point.
                let (a, b) = (point(pair[0]), point(pair[1]));
                walk(&mut boxes, &bounds, (a.0 + ox, a.1 + oy), (b.0 + ox, b.1 + oy), true);
            }
        } else {
            let relative = matches!(selector, X_RLINE | X_RLINE6);
            // A relative polyline restarts at the drawable's origin with each command, as relative
            // points do (established); an absolute one adds the origin to each point.
            let (ox, oy) = self.origin();
            let (mut sx, mut sy) = if relative { (ox, oy) } else { (0, 0) };
            let mut previous = None;
            // The points of a waiting data run come first, then the token's: the X server sends all
            // but the last of a long polyline as a run (inferred from the sender).
            let mut run = std::mem::take(&mut self.run);
            for &w in run.iter().chain(args) {
                let (px, py) = point(w);
                let p = if relative {
                    sx += px;
                    sy += py;
                    (sx, sy)
                } else {
                    (px + ox, py + oy)
                };
                // The first point, then each segment without its start pixel,
                // which the segment before it has drawn.
                match previous {
                    None => plot(&mut boxes, &bounds, p.0, p.1),
                    Some(q) => walk(&mut boxes, &bounds, q, p, false),
                }
                previous = Some(p);
            }
            run.clear();
            self.run = run;
        }
        self.run_request(surface, boxes.iter().copied(), false);
        self.boxes = boxes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::kona::back::store::PixelStore;

    /// An index-16 context: foreground 0x5a00, full mask.
    const GC: [u32; 13] = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];

    fn rig() -> (XState, PixelStore) {
        let mut x = XState::default();
        x.command(0x129, &GC);
        (x, PixelStore::new(32, 32))
    }

    fn pack(a: i32, b: i32) -> u32 {
        (a as u32) << 16 | (b as u32 & 0xffff)
    }

    /// The lit pixels, as X coordinates, in row order.
    fn lit(s: &PixelStore) -> Vec<(u32, u32)> {
        (0..32).flat_map(|y| (0..32).map(move |x| (x, y))).filter(|&(x, y)| s.read(x, 31 - y, 1) != 0).collect()
    }

    #[test]
    fn a_segment_draws_both_ends_and_runs_along_a_row() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        assert!(x.draw_command(&d, X_SEGMENT, &[pack(2, 3), pack(6, 3)]));
        assert_eq!(lit(&s), [(2, 3), (3, 3), (4, 3), (5, 3), (6, 3)]);
        // Backwards is the same pixels.
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_SEGMENT, &[pack(6, 3), pack(2, 3)]);
        assert_eq!(lit(&s), [(2, 3), (3, 3), (4, 3), (5, 3), (6, 3)]);
    }

    #[test]
    fn a_segment_of_no_length_is_one_pixel() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_SEGMENT, &[pack(7, 9), pack(7, 9)]);
        assert_eq!(lit(&s), [(7, 9)]);
    }

    /// A diagonal lights its diagonal; where a line is exactly half a pixel
    /// along the minor axis it takes the pixel on the end point's side
    /// (approximated).
    #[test]
    fn diagonal_and_sloped_lines_follow_the_pixel_centres() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_SEGMENT, &[pack(1, 1), pack(4, 4)]);
        assert_eq!(lit(&s), [(1, 1), (2, 2), (3, 3), (4, 4)]);
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        // Four columns over two rows: the line is half a pixel along the
        // minor axis at the first column and at the third, and each exact half
        // goes to the end point's side.
        x.draw_command(&d, X_SEGMENT, &[pack(0, 0), pack(4, 2)]);
        assert_eq!(lit(&s), [(0, 0), (1, 1), (2, 1), (3, 2), (4, 2)]);
        // A steep line is the same shape on the other axis.
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_SEGMENT, &[pack(0, 0), pack(2, 4)]);
        assert_eq!(lit(&s), [(0, 0), (1, 1), (1, 2), (2, 3), (2, 4)]);
    }

    /// Each segment of a burst is its own, and segments that meet draw the
    /// shared pixel for each.
    #[test]
    fn a_burst_draws_six_independent_segments() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        let mut w = Vec::new();
        for i in 0..6 {
            w.extend([pack(0, 2 * i), pack(3, 2 * i)]);
        }
        x.draw_command(&d, X_SEGMENT6, &w);
        assert_eq!(lit(&s).len(), 6 * 4);
        // Two segments through one pixel under XOR clear it.
        let (mut x, s) = rig();
        let mut gc = GC;
        gc[9] = 12; // logic operation 6, xor
        x.command(0x129, &gc);
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_SEGMENT, &[pack(0, 5), pack(4, 5)]);
        x.draw_command(&d, X_SEGMENT, &[pack(4, 5), pack(4, 8)]);
        assert!(!lit(&s).contains(&(4, 5)), "the meeting pixel was drawn by both");
        assert_eq!(lit(&s).len(), 4 + 3);
    }

    /// A polyline draws a vertex shared by two of its segments once, even
    /// where the write would undo itself, and does not close itself.
    #[test]
    fn a_polyline_draws_its_interior_vertices_once_and_does_not_close() {
        let (mut x, s) = rig();
        let mut gc = GC;
        gc[9] = 12;
        x.command(0x129, &gc);
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_LINE6, &[pack(0, 0), pack(4, 0), pack(4, 4), pack(0, 4), pack(0, 1), pack(0, 1), pack(0, 1)]);
        let got = lit(&s);
        assert!(got.contains(&(4, 0)) && got.contains(&(4, 4)) && got.contains(&(0, 4)), "{got:?}");
        assert!(got.contains(&(0, 0)), "the start");
        // Row 0 (5), column 4 (4 more), row 4 (4 more), column 0 up from row 4 to row 1 (3 more).
        assert_eq!(got.len(), 5 + 4 + 4 + 3);
        // The last point of the line is (0, 1): (0, 0) is lit only as the start.
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_LINE, &[pack(3, 3), pack(8, 3)]);
        assert_eq!(lit(&s).len(), 6);
    }

    /// A polyline of two to five points arrives as one to four words of data preload and the one-word token
    /// (the sender counts), and a single word alone is a single pixel.
    #[test]
    fn a_polyline_of_two_to_five_points_is_a_run_and_the_one_word_token() {
        for points in 2..=5 {
            let (mut x, s) = rig();
            let d = Surface::new(&s, 1);
            let run: Vec<u32> = (0..points - 1).map(|i| pack(2 * i, 4)).collect();
            x.draw_command(&d, 0x140 + run.len() as u32, &run);
            x.draw_command(&d, X_LINE, &[pack(2 * (points - 1), 4)]);
            let want: Vec<(u32, u32)> = (0..=2 * (points as u32 - 1)).map(|c| (c, 4)).collect();
            assert_eq!(lit(&s), want, "{points} points");
            assert_eq!(x.runs_dropped, 0);
        }
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_LINE, &[pack(5, 6)]);
        assert_eq!(lit(&s), [(5, 6)]);
    }

    /// The drawable's position is added to every vertex of a segment and of an absolute polyline, with or
    /// without a data run, as it is to a relative polyline's start.
    #[test]
    fn the_position_is_added_to_every_vertex() {
        let (mut x, s) = rig();
        x.command(0x12a, &[10, 20, 100, 100]);
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_SEGMENT, &[pack(1, 1), pack(3, 1)]);
        assert_eq!(lit(&s), [(11, 21), (12, 21), (13, 21)]);
        x.draw_command(&d, X_SEGMENT6, &[pack(0, 2), pack(1, 2), pack(0, 3), pack(1, 3), pack(0, 4), pack(1, 4), pack(0, 5), pack(1, 5), pack(0, 6), pack(1, 6), pack(0, 7), pack(1, 7)]);
        assert_eq!(lit(&s).len(), 3 + 12);
        assert!(lit(&s).contains(&(11, 27)));
        let (mut x, s) = rig();
        x.command(0x12a, &[10, 20, 100, 100]);
        let d = Surface::new(&s, 1);
        x.draw_command(&d, 0x141, &[pack(0, 0)]);
        x.draw_command(&d, X_LINE, &[pack(2, 0)]);
        assert_eq!(lit(&s), [(10, 20), (11, 20), (12, 20)]);
        x.draw_command(&d, X_LINE6, &[pack(0, 3), pack(2, 3), pack(2, 3), pack(2, 3), pack(2, 3), pack(2, 3), pack(2, 3)]);
        assert_eq!(lit(&s).len(), 3 + 3);
        assert!(lit(&s).contains(&(10, 23)) && lit(&s).contains(&(12, 23)));
    }

    /// Relative polylines step from the previous point, the first from the
    /// drawable's origin.
    #[test]
    fn a_relative_polyline_steps_from_the_origin_and_the_previous_point() {
        let (mut x, s) = rig();
        x.command(0x12a, &[10, 20, 100, 100]);
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_RLINE, &[pack(1, 1), pack(3, 0)]);
        assert_eq!(lit(&s), [(11, 21), (12, 21), (13, 21), (14, 21)]);
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_RLINE6, &[pack(2, 2), pack(2, 0), pack(0, 2), pack(-2, 0), pack(0, 0), pack(0, 0), pack(0, 0)]);
        // From (2, 2) right to (4, 2), down to (4, 4), left to (2, 4).
        assert_eq!(lit(&s).len(), 3 + 2 + 2);
    }

    /// Pixels off the screen are not drawn, and a line cut by the clip draws
    /// the part inside.
    #[test]
    fn a_line_is_clipped_to_the_screen_and_the_clip_list() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_SEGMENT, &[pack(-5, 2), pack(3, 2)]);
        assert_eq!(lit(&s), [(0, 2), (1, 2), (2, 2), (3, 2)]);
        x.command(0x12b, &[1, 10, 0, 12, 32]);
        x.draw_command(&d, X_SEGMENT, &[pack(8, 6), pack(14, 6)]);
        assert_eq!(lit(&s).iter().filter(|p| p.1 == 6).count(), 2);
        x.draw_command(&d, X_SEGMENT, &[pack(-30000, -30000), pack(30000, 30000)]);
    }

    /// A line whose destination does not fit the target draws nothing, as
    /// every other fill is checked before it is drawn.
    #[test]
    fn a_line_over_a_wide_destination_is_refused_whole() {
        let (mut x, s) = rig();
        let mut masked = GC;
        masked[4] = 0xf0053f;
        x.command(0x129, &masked);
        s.write(4, 31 - 3, 1, 0x12_3456);
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_SEGMENT, &[pack(0, 3), pack(8, 3)]);
        assert_eq!((x.fills.bad_destination, s.read(0, 31 - 3, 1)), (1, 0));
    }

    /// A segment token ends a waiting data run, which no segment uses.
    #[test]
    fn a_segment_token_drops_a_waiting_run() {
        let (mut x, _s) = rig();
        x.command(0x141, &[pack(1, 1)]);
        x.collect_command(0x141, &[pack(1, 1)]);
        assert!(!x.command(X_SEGMENT, &[]));
        assert_eq!(x.runs_dropped, 1);
    }

    /// A polyline takes the points of a waiting run before its own, as the X server sends a long one
    /// (inferred), and the run is spent; a relative polyline starts at the origin with every command.
    #[test]
    fn a_polyline_joins_a_waiting_data_run() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        for p in [pack(2, 2), pack(6, 2)] {
            x.command(0x141, &[p]);
            x.collect_command(0x141, &[p]);
        }
        x.end_runs(X_LINE);
        assert_eq!(x.runs_dropped, 0, "a polyline keeps the run");
        x.draw_command(&d, X_LINE, &[pack(6, 5), pack(2, 5)]);
        assert_eq!(lit(&s).len(), 5 + 3 + 4, "(2,2)-(6,2), then down to (6,5), then to (2,5), shared vertices once");
        assert!(lit(&s).contains(&(2, 2)) && lit(&s).contains(&(2, 5)));
        // Spent: a second polyline starts from its own first point.
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.command(0x141, &[pack(2, 2)]);
        x.collect_command(0x141, &[pack(2, 2)]);
        x.draw_command(&d, X_LINE, &[pack(4, 4), pack(4, 6)]);
        x.draw_command(&d, X_LINE, &[pack(9, 9), pack(9, 9)]);
        assert!(!lit(&s).contains(&(2, 2)) || lit(&s).contains(&(3, 3)), "the first run joined the first polyline only");
        assert_eq!(lit(&s).iter().filter(|p| p.0 == 9).count(), 1);
    }

    /// A relative polyline's first point is a step from the drawable's origin, every command.
    #[test]
    fn a_relative_polyline_restarts_at_the_origin_each_command() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_RLINE, &[pack(3, 3), pack(2, 0)]);
        x.draw_command(&d, X_RLINE, &[pack(3, 3), pack(0, 2)]);
        let l = lit(&s);
        assert!(l.contains(&(3, 3)) && l.contains(&(5, 3)) && l.contains(&(3, 5)));
        assert!(!l.contains(&(6, 6)), "the second command did not continue from the first's end");
    }
}
