//! The host's hands on the guest: look at the screen, put the pointer on a
//! pixel, click, press keys. Shared by the VNC server and the monitor's
//! `screen`, `ptr` and `key` commands, for every machine and board.
//!
//! The guest's mouse is relative (PS/2), and the X server accelerates what
//! it reads, so a position cannot be sent. Instead the pointer is steered:
//! send a move, read back where the board's hardware cursor went (IRIX always
//! uses it), and correct until the pointer sits on the target. The hardware
//! knows the glyph's top-left, not its hotspot. A hotspot is known exactly
//! when the pointer is pinned in the top-left corner (X keeps the pointer on
//! screen, so it is at 0,0), which the first move does once, or when a new
//! shape appears after a move small enough that X does not accelerate it
//! (the pointer then moved exactly that far). Until then a new shape's
//! hotspot is guessed as its tip plus the offset the known shapes showed.
//! Only the top and left edges pin reliably: the size X uses can differ
//! from the board's visible size by a few pixels.

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use winit::keyboard::KeyCode;

use crate::dev::ps2::Ps2Controller;
use crate::gfx_display::{GfxDisplay, HwCursor};

/// PS/2 button bits.
pub const BUTTON_LEFT: u8 = 1;
pub const BUTTON_RIGHT: u8 = 2;
pub const BUTTON_MIDDLE: u8 = 4;

/// How long a move waits for the cursor to answer one packet.
const MOVE_ANSWER: Duration = Duration::from_millis(250);
/// How long the cursor must stay put after a move before the next step.
const SETTLE: Duration = Duration::from_millis(20);
/// How long the cursor may be off before a move gives up on it.
const CURSOR_GAP: Duration = Duration::from_millis(400);
/// How long the pointer may ignore the mouse (the guest busy) before a move
/// gives up.
const STALL_LIMIT: Duration = Duration::from_millis(3000);
/// Moves this small (mickeys) are sent exactly: under X's acceleration
/// threshold, they move the pointer as far as they say.
const EXACT_STEP: i32 = 3;

pub struct HostInput {
    display: Option<Arc<dyn GfxDisplay>>,
    ps2: Arc<Ps2Controller>,
    /// Buttons held now (PS/2 bits), sent with every move.
    buttons: AtomicU8,
    /// Hotspots seen exactly, by glyph shape.
    hotspots: Mutex<HashMap<u64, (i32, i32)>>,
    /// Hotspot minus tip of the last shape learnt: the guess for a new one.
    bias: Mutex<(i32, i32)>,
    /// Pixels the pointer moves per mickey on a large move (X's
    /// acceleration), learnt as moves are made.
    gain: Mutex<f32>,
    /// One steering loop at a time: two would fight over the pointer.
    seek: Mutex<()>,
}

/// Where a move ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveResult {
    /// The pointer is on the target.
    Reached,
    /// Time ran out this far (x, y) from the target.
    Short(i32, i32),
}

impl HostInput {
    pub fn new(display: Option<Arc<dyn GfxDisplay>>, ps2: Arc<Ps2Controller>) -> Self {
        Self {
            display,
            ps2,
            buttons: AtomicU8::new(0),
            hotspots: Mutex::new(HashMap::new()),
            bias: Mutex::new((0, 0)),
            gain: Mutex::new(1.0),
            seek: Mutex::new(()),
        }
    }

    /// Build one for a machine and add its commands to the machine's monitor.
    pub fn attach(machine: &dyn crate::workstation::Workstation) -> Arc<Self> {
        let hi = Arc::new(Self::new(machine.get_display(), machine.get_ps2()));
        machine.register_monitor_device(hi.clone());
        hi
    }

    pub fn display(&self) -> Option<&Arc<dyn GfxDisplay>> {
        self.display.as_ref()
    }

    pub fn ps2(&self) -> &Arc<Ps2Controller> {
        &self.ps2
    }

    /// The display's visible size, (0, 0) before it has a mode.
    pub fn screen_size(&self) -> (usize, usize) {
        self.display.as_ref().map(|d| {
            let s = d.screen().lock();
            (s.width, s.height)
        }).unwrap_or((0, 0))
    }

    fn cursor(&self) -> Option<HwCursor> {
        self.display.as_ref()?.hw_cursor()
    }

    /// The cursor, waiting out a short gap: X servers turn the hardware
    /// cursor off for a moment while they draw under it or change it.
    fn cursor_settled(&self) -> Option<HwCursor> {
        let start = Instant::now();
        loop {
            if let Some(c) = self.cursor() {
                return Some(c);
            }
            if start.elapsed() >= CURSOR_GAP {
                return None;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// The hotspot of a glyph: seen exactly, or else guessed.
    pub fn hotspot(&self, c: &HwCursor) -> (i32, i32) {
        self.known_hotspot(c).unwrap_or_else(|| {
            let (tx, ty) = c.tip();
            let (bx, by) = *self.bias.lock();
            (tx + bx, ty + by)
        })
    }

    fn known_hotspot(&self, c: &HwCursor) -> Option<(i32, i32)> {
        self.hotspots.lock().get(&c.shape_key()).copied()
    }

    fn learn(&self, c: &HwCursor, hot: (i32, i32)) {
        if !hotspot_plausible(c, hot) {
            return;
        }
        self.hotspots.lock().insert(c.shape_key(), hot);
        let (tx, ty) = c.tip();
        *self.bias.lock() = (hot.0 - tx, hot.1 - ty);
    }

    /// Where the pointer is, by the hardware cursor; None while it is off.
    pub fn pointer(&self) -> Option<(i32, i32)> {
        let c = self.cursor()?;
        let (hx, hy) = self.hotspot(&c);
        Some((c.x + hx, c.y + hy))
    }

    /// Steer the pointer onto (x, y), clamped to the screen. `retarget` is
    /// asked between steps for a newer target (a VNC viewer still moving).
    /// Errors when there is no cursor to steer by or it does not move.
    pub fn move_to(&self, x: i32, y: i32, timeout: Duration,
                   retarget: &mut dyn FnMut() -> Option<(i32, i32)>) -> Result<MoveResult, String> {
        let _one = self.seek.lock();
        if self.hotspots.lock().is_empty() {
            // Once: pin the pointer in the corner to know a hotspot exactly.
            self.calibrate_locked()?;
        }
        let deadline = Instant::now() + timeout;
        let (mut tx, mut ty) = (x, y);
        let mut stalls = [0u32; 2];
        let mut last_motion = Instant::now();
        // How far right/down the pointer goes, once seen pinned there: a
        // board that draws the cursor a pixel off leaves the last column or
        // row out of reach, and a move there ends at the edge.
        let mut far_edge: [Option<i32>; 2] = [None; 2];
        loop {
            if let Some(t) = retarget() {
                (tx, ty) = t;
                stalls = [0; 2];
            }
            let (w, h) = self.screen_size();
            if w == 0 {
                return Err("the display has no video mode yet".into());
            }
            let (tx, ty) = (tx.clamp(0, w as i32 - 1), ty.clamp(0, h as i32 - 1));
            let (tx, ty) = (far_edge[0].map_or(tx, |e| tx.min(e)), far_edge[1].map_or(ty, |e| ty.min(e)));
            let c = self.cursor_settled().ok_or("the hardware cursor is off (no X server yet?)")?;
            let (hx, hy) = self.hotspot(&c);
            let (ex, ey) = (tx - (c.x + hx), ty - (c.y + hy));
            if ex == 0 && ey == 0 {
                return Ok(MoveResult::Reached);
            }
            if Instant::now() >= deadline {
                return Ok(MoveResult::Short(ex, ey));
            }
            let gain = *self.gain.lock();
            let step = |e: i32| -> i32 {
                if e.abs() <= EXACT_STEP {
                    e
                } else {
                    match ((e as f32 / gain).round() as i32).clamp(-127, 127) {
                        0 => e.signum(),
                        s => s,
                    }
                }
            };
            let (sx, sy) = (step(ex), step(ey));
            // Steer from where the cursor is, not from moves still queued:
            // let a busy guest catch up before sending more.
            let queued = Instant::now();
            while self.ps2.mouse_pending() > 0 && queued.elapsed() < MOVE_ANSWER {
                std::thread::sleep(Duration::from_millis(1));
            }
            if self.ps2.mouse_pending() > 0 {
                if last_motion.elapsed() >= STALL_LIMIT {
                    return Err("the guest is not reading the mouse".into());
                }
                continue;
            }
            self.ps2.push_mouse_input(self.buttons.load(Ordering::Relaxed), sx, sy, 0);

            // Wait for the X server to read it and move the cursor, then
            // for the cursor to settle: a board may show a move a frame late,
            // and steering from a position before the last step oscillates.
            let sent = Instant::now();
            let mut now_at: Option<HwCursor> = None;
            let mut changed_at = sent;
            while sent.elapsed() < MOVE_ANSWER {
                std::thread::sleep(Duration::from_millis(1));
                match self.cursor() {
                    Some(n) if (n.x, n.y) != (c.x, c.y) || n.mask != c.mask => {
                        if now_at.as_ref().is_none_or(|m| (m.x, m.y, &m.mask) != (n.x, n.y, &n.mask)) {
                            changed_at = Instant::now();
                            now_at = Some(n);
                        } else if changed_at.elapsed() >= SETTLE {
                            break;
                        }
                    }
                    Some(_) | None => {}
                }
            }
            let (mx, my) = match &now_at {
                Some(n) if n.mask == c.mask => (n.x - c.x, n.y - c.y),
                Some(n) => {
                    // A new shape. After an unaccelerated step from a known
                    // shape the pointer moved exactly (sx, sy): that gives
                    // the new hotspot.
                    if sx.abs() <= EXACT_STEP && sy.abs() <= EXACT_STEP && self.known_hotspot(n).is_none() {
                        if let Some((ohx, ohy)) = self.known_hotspot(&c) {
                            // Let X finish the move before reading the glyph.
                            std::thread::sleep(Duration::from_millis(5));
                            if let Some(n) = self.cursor().filter(|m| m.mask == n.mask) {
                                let (px, py) = (c.x + ohx + sx, c.y + ohy + sy);
                                self.learn(&n, (px - n.x, py - n.y));
                            }
                        }
                    }
                    stalls = [0; 2];
                    continue;
                }
                None => (0, 0),
            };
            // Learn the acceleration from large steps.
            for (m, s) in [(mx, sx), (my, sy)] {
                if s.abs() > EXACT_STEP * 2 && m != 0 && m.signum() == s.signum() {
                    let mut g = self.gain.lock();
                    *g = (*g * 0.5 + (m as f32 / s as f32) * 0.5).clamp(0.25, 8.0);
                }
            }
            // An axis that will not move although pushed: pinned at an edge
            // (X keeps the pointer on screen). At the left or top edge the
            // pointer is at 0, which tells the hotspot. Or the guest is not
            // reading the mouse.
            let size = c.size as i32;
            let mut learnt = false;
            for (axis, (m, s, pos)) in [(mx, sx, c.x), (my, sy, c.y)].into_iter().enumerate() {
                if s == 0 || m != 0 {
                    stalls[axis] = 0;
                    continue;
                }
                stalls[axis] += 1;
                if stalls[axis] < 2 {
                    continue;
                }
                if s > 0 {
                    // Within a glyph of the right or bottom edge: that is
                    // the edge.
                    let p = pos + [hx, hy][axis];
                    if p >= [w, h][axis] as i32 - size {
                        far_edge[axis] = Some(p);
                        learnt = true;
                    }
                    continue;
                }
                let hot = -pos;
                if (-size..size).contains(&hot) && hot != [hx, hy][axis] {
                    let mut both = (hx, hy);
                    if axis == 0 { both.0 = hot } else { both.1 = hot }
                    self.learn(&c, both);
                    stalls[axis] = 0;
                    learnt = true;
                }
            }
            if mx != 0 || my != 0 || learnt {
                last_motion = Instant::now();
            }
            if last_motion.elapsed() >= STALL_LIMIT {
                return Err(if self.ps2.mouse_enabled() {
                    "the cursor does not follow the mouse (guest busy, or no X server?)".into()
                } else {
                    "the guest has not enabled the mouse".into()
                });
            }
        }
    }

    /// Move, then check the move arrived.
    pub fn move_to_exact(&self, x: i32, y: i32) -> Result<(), String> {
        match self.move_to(x, y, Duration::from_secs(3), &mut || None)? {
            MoveResult::Reached => Ok(()),
            MoveResult::Short(ex, ey) => Err(format!("pointer stopped {ex},{ey} short of {x},{y}")),
        }
    }

    /// Push the pointer into the top-left corner and learn the current
    /// glyph's hotspot from where the cursor stops. Returns the hotspot.
    pub fn calibrate(&self) -> Result<(i32, i32), String> {
        let _one = self.seek.lock();
        self.calibrate_locked()
    }

    fn calibrate_locked(&self) -> Result<(i32, i32), String> {
        // Keep pushing up-left until the cursor has stayed put for a while
        // although the guest read every packet: then X has it pinned.
        const PINNED: Duration = Duration::from_millis(150);
        let start = Instant::now();
        let mut still: Option<((i32, i32), Instant)> = None;
        while start.elapsed() < Duration::from_secs(10) {
            let drained = Instant::now();
            while self.ps2.mouse_pending() > 0 && drained.elapsed() < Duration::from_millis(500) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let c = self.cursor_settled().ok_or("the hardware cursor is off")?;
            match still {
                Some((at, since)) if at == (c.x, c.y) && self.ps2.mouse_pending() == 0 => {
                    if since.elapsed() >= PINNED {
                        let hot = (-c.x, -c.y);
                        if !hotspot_plausible(&c, hot) {
                            return Err(format!("the cursor stopped at {},{}, not in the corner", c.x, c.y));
                        }
                        self.learn(&c, hot);
                        return Ok(hot);
                    }
                }
                _ => still = Some(((c.x, c.y), Instant::now())),
            }
            self.ps2.push_mouse_input(self.buttons.load(Ordering::Relaxed), -100, -100, 0);
            std::thread::sleep(Duration::from_millis(20));
        }
        Err("the cursor never settled in the corner".into())
    }

    /// A raw relative move, as a mouse would send it.
    pub fn nudge(&self, dx: i32, dy: i32) {
        let b = self.buttons.load(Ordering::Relaxed);
        // Packets carry 9-bit deltas.
        let (mut dx, mut dy) = (dx, dy);
        while dx != 0 || dy != 0 {
            let (sx, sy) = (dx.clamp(-200, 200), dy.clamp(-200, 200));
            self.ps2.push_mouse_input(b, sx, sy, 0);
            (dx, dy) = (dx - sx, dy - sy);
        }
    }

    /// Hold exactly `mask` (PS/2 bits) from now on.
    pub fn set_buttons(&self, mask: u8) {
        if self.buttons.swap(mask, Ordering::Relaxed) != mask {
            self.ps2.push_mouse_input(mask, 0, 0, 0);
        }
    }

    pub fn buttons(&self) -> u8 {
        self.buttons.load(Ordering::Relaxed)
    }

    /// Wheel clicks (positive away from the user); only reaches a guest
    /// that has switched the mouse to its wheel mode.
    pub fn wheel(&self, dz: i32) {
        self.ps2.push_mouse_input(self.buttons.load(Ordering::Relaxed), 0, 0, dz);
    }

    /// Press and release `button` `count` times where the pointer is.
    pub fn click(&self, button: u8, count: u32) {
        let held = self.buttons.load(Ordering::Relaxed);
        for i in 0..count {
            if i > 0 {
                std::thread::sleep(Duration::from_millis(60));
            }
            self.set_buttons(held | button);
            std::thread::sleep(Duration::from_millis(40));
            self.set_buttons(held & !button);
        }
    }

    pub fn key(&self, code: KeyCode, down: bool) {
        self.ps2.push_kb(code, down);
    }

    /// Press a chord such as `ctrl+alt+F1`, or with `down` = Some, only
    /// press or only release it.
    pub fn chord(&self, chord: &str, down: Option<bool>) -> Result<(), String> {
        let keys = chord.split('+').filter(|s| !s.is_empty())
            .map(|n| crate::keysym::from_name(n).ok_or_else(|| format!("unknown key '{n}'")))
            .collect::<Result<Vec<_>, _>>()?;
        if keys.is_empty() {
            return Err("no key named".into());
        }
        if down != Some(false) {
            for &k in &keys {
                self.key(k, true);
            }
        }
        if down.is_none() {
            std::thread::sleep(Duration::from_millis(30));
        }
        if down != Some(true) {
            for &k in keys.iter().rev() {
                self.key(k, false);
            }
        }
        Ok(())
    }

    /// Type text on a US keyboard; returns how many characters had a key.
    pub fn type_text(&self, text: &str) -> usize {
        text.chars().filter(|&ch| match crate::keysym::char_key(ch) {
            Some((k, shifted)) => {
                if shifted { self.key(KeyCode::ShiftLeft, true); }
                self.key(k, true);
                self.key(k, false);
                if shifted { self.key(KeyCode::ShiftLeft, false); }
                true
            }
            None => self.ps2.type_char(ch),
        }).count()
    }

    /// A fresh frame, tight, with its size.
    pub fn frame(&self) -> Result<(usize, usize, Vec<u32>), String> {
        let d = self.display.as_ref().ok_or("no display (headless)")?;
        let mut px = Vec::new();
        let (w, h, _) = crate::gfx_display::fresh_frame(&**d, &mut px, Duration::from_millis(250))
            .ok_or("the display has no video mode yet")?;
        Ok((w, h, px))
    }

    /// Wait until the screen has not changed for `quiet`, at most `timeout`.
    /// Returns whether it settled.
    pub fn wait_still(&self, quiet: Duration, timeout: Duration) -> Result<bool, String> {
        let start = Instant::now();
        let mut last = self.frame()?;
        let mut since = Instant::now();
        while start.elapsed() < timeout {
            std::thread::sleep(Duration::from_millis(50));
            let f = self.frame()?;
            if f != last {
                last = f;
                since = Instant::now();
            } else if since.elapsed() >= quiet {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// A hotspot is inside its glyph, give or take how far the board draws the
/// cursor from where the guest put it: a board whose cursor offset we have
/// slightly wrong shows a hotspot a few pixels outside.
fn hotspot_plausible(c: &HwCursor, (hx, hy): (i32, i32)) -> bool {
    let size = c.size as i32;
    (-8..size + 8).contains(&hx) && (-8..size + 8).contains(&hy)
}

/// Save tight `0xAABBGGRR` pixels (row length `w`) as an RGB PNG, cropped to
/// `crop` (x, y, w, h) when given.
pub fn save_png(path: &str, px: &[u32], w: usize, h: usize, crop: Option<(usize, usize, usize, usize)>) -> Result<(usize, usize), String> {
    let (cx, cy, cw, ch) = crop.unwrap_or((0, 0, w, h));
    if cw == 0 || ch == 0 || cx + cw > w || cy + ch > h {
        return Err(format!("crop {cx},{cy} {cw}x{ch} is outside the {w}x{h} screen"));
    }
    let mut rgb = Vec::with_capacity(cw * ch * 3);
    for y in cy..cy + ch {
        for &p in &px[y * w + cx..y * w + cx + cw] {
            rgb.extend_from_slice(&[p as u8, (p >> 8) as u8, (p >> 16) as u8]);
        }
    }
    let tmp = format!("{path}.tmp");
    {
        let f = std::fs::File::create(&tmp).map_err(|e| format!("{path}: {e}"))?;
        let mut enc = png::Encoder::new(std::io::BufWriter::new(f), cw as u32, ch as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().map_err(|e| e.to_string())?;
        wr.write_image_data(&rgb).map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("{path}: {e}"))?;
    Ok((cw, ch))
}

fn parse_button(s: Option<&&str>) -> Result<u8, String> {
    Ok(match s.copied().unwrap_or("left") {
        "left" | "l" | "1" => BUTTON_LEFT,
        "middle" | "m" | "2" => BUTTON_MIDDLE,
        "right" | "r" | "3" => BUTTON_RIGHT,
        b => return Err(format!("unknown button '{b}' (left, middle, right)")),
    })
}

fn num<T: std::str::FromStr>(args: &[&str], i: usize, what: &str) -> Result<T, String> {
    args.get(i).ok_or(format!("missing {what}"))?.parse().map_err(|_| format!("bad {what} '{}'", args[i]))
}

const SCREEN_HELP: &str = "screen info | screen cursor | screen shot <file.png> [x y w h] | screen pix <x> <y> | screen wait [quiet_ms] [timeout_ms]";
const PTR_HELP: &str = "ptr pos | ptr move <x> <y> | ptr click <x> <y> [left|middle|right] [count] | ptr down|up [button] | ptr drag <x1> <y1> <x2> <y2> [button] | ptr rel <dx> <dy> | ptr calibrate";
const KEY_HELP: &str = "key <name>[+<name>...] [down|up] | key type <text> | key names";

impl crate::traits::Device for HostInput {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("screen".into(), format!("The guest's screen, any board: {SCREEN_HELP}")),
            ("ptr".into(), format!("Pointer at screen pixels, steered by the hardware cursor: {PTR_HELP}")),
            ("key".into(), format!("Keyboard by key name (X keysym names, ctrl/alt/shift/enter/esc..., KeyA): {KEY_HELP}")),
        ]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn Write + Send>) -> Result<(), String> {
        let sub = args.first().copied().unwrap_or("");
        let out = |w: &mut Box<dyn Write + Send>, s: String| { let _ = writeln!(w, "{s}"); };
        match (cmd, sub) {
            ("screen", "info") | ("screen", "") => {
                let (sw, sh) = self.screen_size();
                let seq = self.display.as_ref().map(|d| d.screen().lock().frame_seq).unwrap_or(0);
                out(&mut w, format!("screen {sw}x{sh} frame {seq}"));
                match self.cursor() {
                    Some(c) => {
                        let (hx, hy) = self.hotspot(&c);
                        out(&mut w, format!("cursor glyph at {},{} size {} hotspot {hx},{hy} -> pointer {},{}",
                            c.x, c.y, c.size, c.x + hx, c.y + hy));
                    }
                    None => out(&mut w, "cursor off".into()),
                }
                out(&mut w, format!("mouse {}, buttons held {:#x}",
                    if self.ps2.mouse_enabled() { "enabled" } else { "not enabled by the guest" }, self.buttons()));
                Ok(())
            }
            ("screen", "cursor") => {
                let c = self.cursor().ok_or("cursor off")?;
                let (hx, hy) = self.hotspot(&c);
                out(&mut w, format!("glyph at {},{} size {} shape {:016x} hotspot {hx},{hy} (tip {:?})",
                    c.x, c.y, c.size, c.shape_key(), c.tip()));
                for (y, m) in c.mask.iter().enumerate() {
                    let row: String = (0..c.size).map(|x| match (m >> x & 1 != 0, (x as i32, y as i32) == (hx, hy)) {
                        (_, true) => '@',
                        (true, _) => '#',
                        _ => '.',
                    }).collect();
                    out(&mut w, row);
                }
                Ok(())
            }
            ("screen", "shot") => {
                let path = args.get(1).ok_or("Usage: screen shot <file.png> [x y w h]")?;
                let crop = if args.len() >= 6 {
                    Some((num(args, 2, "x")?, num(args, 3, "y")?, num(args, 4, "w")?, num(args, 5, "h")?))
                } else { None };
                let (fw, fh, px) = self.frame()?;
                let (cw, ch) = save_png(path, &px, fw, fh, crop)?;
                out(&mut w, format!("saved {path} {cw}x{ch} (screen {fw}x{fh})"));
                Ok(())
            }
            ("screen", "pix") => {
                let (x, y): (usize, usize) = (num(args, 1, "x")?, num(args, 2, "y")?);
                let (fw, fh, px) = self.frame()?;
                if x >= fw || y >= fh {
                    return Err(format!("{x},{y} is outside the {fw}x{fh} screen"));
                }
                let p = px[y * fw + x];
                out(&mut w, format!("{x},{y} = {:02x}{:02x}{:02x}", p & 0xff, (p >> 8) & 0xff, (p >> 16) & 0xff));
                Ok(())
            }
            ("screen", "wait") => {
                let quiet = args.get(1).map(|_| num::<u64>(args, 1, "quiet_ms")).transpose()?.unwrap_or(500);
                let timeout = args.get(2).map(|_| num::<u64>(args, 2, "timeout_ms")).transpose()?.unwrap_or(10_000);
                let t = Instant::now();
                let still = self.wait_still(Duration::from_millis(quiet), Duration::from_millis(timeout))?;
                out(&mut w, if still {
                    format!("still for {quiet} ms after {} ms", t.elapsed().as_millis())
                } else {
                    format!("still changing after {timeout} ms")
                });
                Ok(())
            }
            ("screen", _) => Err(format!("Usage: {SCREEN_HELP}")),

            ("ptr", "pos") | ("ptr", "") => {
                match self.pointer() {
                    Some((x, y)) => out(&mut w, format!("{x} {y}")),
                    None => out(&mut w, "cursor off".into()),
                }
                Ok(())
            }
            ("ptr", "move") => {
                let (x, y) = (num(args, 1, "x")?, num(args, 2, "y")?);
                self.move_to_exact(x, y)?;
                let (px, py) = self.pointer().unwrap_or((x, y));
                out(&mut w, format!("pointer at {px} {py}"));
                Ok(())
            }
            ("ptr", "click") => {
                let (x, y) = (num(args, 1, "x")?, num(args, 2, "y")?);
                let b = parse_button(args.get(3))?;
                let n = args.get(4).map(|_| num::<u32>(args, 4, "count")).transpose()?.unwrap_or(1);
                self.move_to_exact(x, y)?;
                self.click(b, n.clamp(1, 3));
                let (px, py) = self.pointer().unwrap_or((x, y));
                out(&mut w, format!("clicked at {px} {py}"));
                Ok(())
            }
            ("ptr", "down") | ("ptr", "up") => {
                let b = parse_button(args.get(1))?;
                let held = self.buttons();
                self.set_buttons(if sub == "down" { held | b } else { held & !b });
                out(&mut w, format!("buttons held {:#x}", self.buttons()));
                Ok(())
            }
            ("ptr", "drag") => {
                let (x1, y1, x2, y2) = (num(args, 1, "x1")?, num(args, 2, "y1")?, num(args, 3, "x2")?, num(args, 4, "y2")?);
                let b = parse_button(args.get(5))?;
                self.move_to_exact(x1, y1)?;
                let held = self.buttons();
                self.set_buttons(held | b);
                std::thread::sleep(Duration::from_millis(40));
                let r = self.move_to_exact(x2, y2);
                std::thread::sleep(Duration::from_millis(40));
                self.set_buttons(held);
                r?;
                out(&mut w, format!("dragged {x1} {y1} -> {x2} {y2}"));
                Ok(())
            }
            ("ptr", "rel") => {
                let (dx, dy) = (num(args, 1, "dx")?, num(args, 2, "dy")?);
                self.nudge(dx, dy);
                out(&mut w, format!("sent {dx} {dy}"));
                Ok(())
            }
            ("ptr", "calibrate") => {
                let (hx, hy) = self.calibrate()?;
                out(&mut w, format!("hotspot {hx},{hy}; pointer at 0 0"));
                Ok(())
            }
            ("ptr", _) => Err(format!("Usage: {PTR_HELP}")),

            ("key", "type") => {
                let text = args[1..].join(" ");
                let n = self.type_text(&text);
                out(&mut w, format!("typed {n} of {} chars", text.chars().count()));
                Ok(())
            }
            ("key", "names") => {
                out(&mut w, crate::keysym::names().collect::<Vec<_>>().join(" "));
                Ok(())
            }
            ("key", "") => Err(format!("Usage: {KEY_HELP}")),
            ("key", chord) => {
                let down = match args.get(1).copied() {
                    None => None,
                    Some("down") => Some(true),
                    Some("up") => Some(false),
                    Some(o) => return Err(format!("'{o}': down or up")),
                };
                self.chord(chord, down)?;
                out(&mut w, format!("key {chord}{}", match down { Some(true) => " down", Some(false) => " up", None => "" }));
                Ok(())
            }
            _ => Err("Command not found".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tip_and_shape() {
        let mut mask = vec![0u64; 32];
        mask[2] = 0b1100;
        mask[3] = 0b1110;
        let c = HwCursor { x: 10, y: 20, size: 32, mask: mask.clone() };
        assert_eq!(c.tip(), (2, 2));
        let mut other = mask;
        other[4] = 1;
        let d = HwCursor { x: 0, y: 0, size: 32, mask: other };
        assert_ne!(c.shape_key(), d.shape_key());
    }

    #[test]
    fn png_crop_bounds() {
        let px = vec![0u32; 4 * 3];
        assert!(save_png("/nonexistent/x.png", &px, 4, 3, Some((3, 0, 2, 1))).is_err());
    }
}
