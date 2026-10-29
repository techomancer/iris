//! The built-in VNC (RFB 3.8) server.
//!
//! It serves whatever the board composes, for every board, and knows nothing
//! about drawing: a grabber thread copies the composed frame (`screen.rgba`)
//! when its sequence number moves, compares it with the last one in 64x64
//! tiles, and stamps each changed tile with the frame's generation. Each
//! viewer remembers the generation it last saw and is sent the tiles newer
//! than that, so a slow viewer skips frames instead of queueing them.
//!
//! Input goes to the guest's PS/2 keyboard and mouse through
//! [`HostInput`]: absolute pointer positions are steered onto the hardware
//! cursor. A window, the monitor and VNC viewers can all type at once.
//!
//! Threads: one accept thread, one grabber, one pointer driver, and two per
//! viewer (reading its messages; sending its updates). All std::net.

pub mod encode;
mod session;

use std::collections::VecDeque;
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::gfx_display::GfxDisplay;
use crate::hostinput::{HostInput, MoveResult};

/// Tile side for damage tracking.
pub const TILE: usize = 64;
/// Viewers served at once.
const MAX_CLIENTS: usize = 8;

/// The last frame and which of its tiles changed when.
#[derive(Clone)]
pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub px: Arc<Vec<u32>>,
    /// Per tile (row-major, `tiles_x` a row), the generation it last changed in.
    pub tile_gen: Arc<Vec<u64>>,
    pub tiles_x: usize,
    /// Generation of this frame; bumped when any tile changes.
    pub gen: u64,
    /// Bumped when the size changes.
    pub size_gen: u64,
}

pub(crate) struct Shared {
    pub input: Arc<HostInput>,
    display: Arc<dyn GfxDisplay>,
    pub frame: Mutex<Frame>,
    /// Signalled on a new frame, and by viewers' readers when they want
    /// their sender to look again.
    pub changed: Condvar,
    pub clients: AtomicUsize,
    pub stop: AtomicBool,
    max_fps: u32,
    pub pointer: PointerDriver,
    pub name: String,
}

pub struct VncServer {
    shared: Arc<Shared>,
    addr: SocketAddr,
}

impl VncServer {
    /// Bind `listen` and start serving `input`'s display.
    pub fn start(input: Arc<HostInput>, listen: &str, max_fps: Option<u32>, name: &str) -> Result<VncServer, String> {
        let display = input.display().cloned()
            .ok_or("VNC needs a display: this machine runs without one (--headless, or no graphics board)")?;
        let listen = if listen.parse::<u16>().is_ok() { format!("127.0.0.1:{listen}") } else { listen.to_string() };
        let listener = TcpListener::bind(&listen).map_err(|e| format!("VNC: bind {listen}: {e}"))?;
        let addr = listener.local_addr().map_err(|e| e.to_string())?;
        if !addr.ip().is_loopback() {
            eprintln!("iris: WARNING: the VNC server on {addr} has no password: anyone who can reach it controls the guest");
        }
        let shared = Arc::new(Shared {
            input: input.clone(),
            display,
            frame: Mutex::new(Frame {
                w: 0, h: 0, px: Arc::new(Vec::new()), tile_gen: Arc::new(Vec::new()),
                tiles_x: 0, gen: 0, size_gen: 0,
            }),
            changed: Condvar::new(),
            clients: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            max_fps: max_fps.unwrap_or(30).clamp(1, 120),
            pointer: PointerDriver::new(),
            name: name.to_string(),
        });
        {
            let s = shared.clone();
            std::thread::Builder::new().name("VNC-Grab".into()).spawn(move || grab_loop(&s)).unwrap();
        }
        {
            let s = shared.clone();
            std::thread::Builder::new().name("VNC-Pointer".into()).spawn(move || s.pointer.run(&s.input, &s.stop)).unwrap();
        }
        {
            let s = shared.clone();
            std::thread::Builder::new().name("VNC-Accept".into()).spawn(move || {
                for conn in listener.incoming() {
                    if s.stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let Ok(stream) = conn else { continue };
                    if s.clients.load(Ordering::Relaxed) >= MAX_CLIENTS {
                        drop(stream);
                        continue;
                    }
                    let s = s.clone();
                    let _ = std::thread::Builder::new().name("VNC-Client".into()).spawn(move || {
                        let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
                        s.clients.fetch_add(1, Ordering::SeqCst);
                        s.changed.notify_all();
                        let r = session::serve(&s, stream);
                        s.clients.fetch_sub(1, Ordering::SeqCst);
                        if let Err(e) = r {
                            eprintln!("iris: VNC {peer}: {e}");
                        }
                    });
                }
            }).unwrap();
        }
        eprintln!("iris: VNC server on {addr}");
        Ok(VncServer { shared, addr })
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn clients(&self) -> usize {
        self.shared.clients.load(Ordering::Relaxed)
    }
}

impl Drop for VncServer {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.shared.changed.notify_all();
        let _ = std::net::TcpStream::connect(self.addr); // wake accept()
    }
}

/// Follow the composed frame while anyone watches.
fn grab_loop(s: &Shared) {
    let period = Duration::from_micros(1_000_000 / s.max_fps as u64);
    let mut capturing = false;
    let mut last_seq = None;
    let mut cur = Vec::new();
    while !s.stop.load(Ordering::Relaxed) {
        let start = Instant::now();
        let watching = s.clients.load(Ordering::Relaxed) > 0;
        if watching != capturing {
            s.display.set_capture(watching);
            capturing = watching;
            last_seq = None;
        }
        if !watching {
            // Sleep until a viewer connects (or a poll period passes).
            let mut f = s.frame.lock();
            s.changed.wait_for(&mut f, Duration::from_millis(200));
            continue;
        }
        let seq = s.display.screen().lock().frame_seq;
        if last_seq != Some(seq) {
            if let Some((w, h, seq)) = crate::gfx_display::copy_frame(&*s.display, &mut cur) {
                last_seq = Some(seq);
                publish(s, w, h, &mut cur);
            }
        }
        let spent = start.elapsed();
        if spent < period {
            std::thread::sleep(period - spent);
        }
    }
    if capturing {
        s.display.set_capture(false);
    }
}

/// Compare `cur` with the published frame and publish it if anything moved.
/// `cur` comes back holding a spare buffer.
fn publish(s: &Shared, w: usize, h: usize, cur: &mut Vec<u32>) {
    let old = s.frame.lock().clone();
    let tiles_x = w.div_ceil(TILE);
    let tiles_y = h.div_ceil(TILE);
    let resized = (old.w, old.h) != (w, h);
    let mut dirty = vec![resized; tiles_x * tiles_y];
    if !resized {
        let prev = &old.px;
        for ty in 0..tiles_y {
            let row_tiles = &mut dirty[ty * tiles_x..(ty + 1) * tiles_x];
            for y in ty * TILE..(ty * TILE + TILE).min(h) {
                let (a, b) = (&cur[y * w..(y + 1) * w], &prev[y * w..(y + 1) * w]);
                if a == b {
                    continue;
                }
                for (tx, d) in row_tiles.iter_mut().enumerate() {
                    if !*d {
                        let x0 = tx * TILE;
                        let x1 = (x0 + TILE).min(w);
                        *d = a[x0..x1] != b[x0..x1];
                    }
                }
            }
        }
        if !dirty.iter().any(|&d| d) {
            return;
        }
    }
    let gen = old.gen + 1;
    let tile_gen: Vec<u64> = if resized {
        vec![gen; tiles_x * tiles_y]
    } else {
        old.tile_gen.iter().zip(&dirty).map(|(&g, &d)| if d { gen } else { g }).collect()
    };
    let px = Arc::new(std::mem::take(cur));
    {
        let mut f = s.frame.lock();
        *f = Frame {
            w, h, px, tile_gen: Arc::new(tile_gen), tiles_x, gen,
            size_gen: old.size_gen + resized as u64,
        };
    }
    // Reuse the frame before last when no viewer still holds it.
    if let Ok(v) = Arc::try_unwrap(old.px) {
        *cur = v;
    }
    s.changed.notify_all();
}

/// Turns viewers' pointer events into steered moves and button changes, in
/// order, on one thread: a click is sent once the pointer is where the
/// viewer clicked.
pub(crate) struct PointerDriver {
    queue: Mutex<VecDeque<(i32, i32, u8)>>,
    ready: Condvar,
}

impl PointerDriver {
    fn new() -> Self {
        Self { queue: Mutex::new(VecDeque::new()), ready: Condvar::new() }
    }

    /// A viewer's pointer event: position and PS/2 buttons.
    pub fn push(&self, x: i32, y: i32, buttons: u8) {
        let mut q = self.queue.lock();
        // Moves with no button change only matter as the latest target.
        if let Some(last) = q.back_mut() {
            if last.2 == buttons && q.len() > 1 && q[q.len() - 2].2 == buttons {
                *q.back_mut().unwrap() = (x, y, buttons);
                drop(q);
                self.ready.notify_one();
                return;
            }
        }
        if q.len() < 1024 {
            q.push_back((x, y, buttons));
        }
        drop(q);
        self.ready.notify_one();
    }

    fn run(&self, input: &HostInput, stop: &AtomicBool) {
        // Where the viewer last put the pointer: for moving by difference
        // when there is no cursor to steer by.
        let mut last: Option<(i32, i32)> = None;
        while !stop.load(Ordering::Relaxed) {
            let (x, y, buttons) = {
                let mut q = self.queue.lock();
                if q.is_empty() {
                    self.ready.wait_for(&mut q, Duration::from_millis(200));
                }
                match q.pop_front() {
                    Some(e) => e,
                    None => continue,
                }
            };
            let held = input.buttons();
            // Take newer moves with the same buttons as the new target; not
            // while a press or release waits for this move to arrive.
            let mut target = (x, y);
            let mut retarget = || {
                let mut q = self.queue.lock();
                match q.front() {
                    Some(&(nx, ny, b)) if b == buttons && buttons == held => {
                        q.pop_front();
                        target = (nx, ny);
                        Some((nx, ny))
                    }
                    _ => None,
                }
            };
            let steered = input.move_to(x, y, Duration::from_millis(400), &mut retarget);
            let target = target;
            match steered {
                Ok(MoveResult::Reached) | Ok(MoveResult::Short(..)) => {}
                Err(_) => {
                    // No cursor to steer by (the X server is not up, or a
                    // program hid it): move by the viewer's difference.
                    if let Some((lx, ly)) = last {
                        input.nudge(target.0 - lx, target.1 - ly);
                    }
                }
            }
            last = Some(target);
            if buttons != held {
                input.set_buttons(buttons);
            }
        }
    }
}
