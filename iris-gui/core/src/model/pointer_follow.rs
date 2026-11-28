//! Follow the host pointer: steer the guest's relative mouse until its
//! hardware cursor is on the pixel the host pointer is over, as the VNC
//! server does (clanker/vnc/pointer-steering-and-hotspots.md).
//!
//! This is the display's no-capture mode. Remote-desktop tools (RustDesk,
//! Screen Sharing) move the host cursor to absolute places and report no
//! motion, so capture, which forwards motion, gets nothing; steering only
//! needs where the pointer is.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use iris::hostinput::{HostInput, MoveResult};
use parking_lot::{Condvar, Mutex};

/// What the follower does next, in the order the host did it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ev {
    /// Put the pointer on this guest pixel.
    Move(i32, i32),
    /// The buttons held now (bit 0 left, 1 right, 2 middle).
    Buttons(u8),
    Wheel(i32),
}

struct Shared {
    queue: Mutex<VecDeque<Ev>>,
    ready: Condvar,
    stop: AtomicBool,
}

/// The steering thread for one running machine. Dropping it stops the thread.
pub struct PointerFollower {
    shared: Arc<Shared>,
}

impl PointerFollower {
    pub fn start(input: Arc<HostInput>) -> Self {
        let shared = Arc::new(Shared { queue: Mutex::new(VecDeque::new()), ready: Condvar::new(), stop: AtomicBool::new(false) });
        let s = shared.clone();
        let spawned = std::thread::Builder::new().name("iris-pointer-follow".into()).spawn(move || run(&s, &input));
        if spawned.is_err() {
            log::warn!("could not start the pointer follower");
        }
        Self { shared }
    }

    fn push(&self, ev: Ev) {
        let mut q = self.shared.queue.lock();
        // A newer position replaces a move not yet started.
        if let (Ev::Move(..), Some(Ev::Move(..))) = (ev, q.back()) {
            q.pop_back();
        }
        q.push_back(ev);
        drop(q);
        self.shared.ready.notify_one();
    }

    /// The host pointer is over guest pixel (x, y).
    pub fn move_to(&self, x: i32, y: i32) {
        self.push(Ev::Move(x, y));
    }

    pub fn buttons(&self, held: u8) {
        self.push(Ev::Buttons(held));
    }

    pub fn wheel(&self, dz: i32) {
        if dz != 0 {
            self.push(Ev::Wheel(dz));
        }
    }
}

impl Drop for PointerFollower {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.ready.notify_all();
    }
}

fn run(s: &Shared, input: &HostInput) {
    // Where the host last put the pointer, for moving by difference when
    // there is no cursor to steer by (no X server yet, or a hidden cursor).
    let mut last: Option<(i32, i32)> = None;
    while !s.stop.load(Ordering::Relaxed) {
        let ev = {
            let mut q = s.queue.lock();
            if q.is_empty() {
                s.ready.wait_for(&mut q, Duration::from_millis(200));
            }
            match q.pop_front() {
                Some(e) => e,
                None => continue,
            }
        };
        match ev {
            Ev::Move(x, y) => {
                let target = std::cell::Cell::new((x, y));
                // Take newer positions as the new target while steering;
                // a button change waits for this move to arrive.
                let mut retarget = || {
                    let mut q = s.queue.lock();
                    match q.front() {
                        Some(&Ev::Move(nx, ny)) => {
                            q.pop_front();
                            target.set((nx, ny));
                            Some((nx, ny))
                        }
                        _ => None,
                    }
                };
                match input.move_to(x, y, Duration::from_millis(400), &mut retarget) {
                    Ok(MoveResult::Reached) | Ok(MoveResult::Short(..)) => {}
                    Err(_) => {
                        if let Some((lx, ly)) = last {
                            let (tx, ty) = target.get();
                            input.nudge(tx - lx, ty - ly);
                        }
                    }
                }
                last = Some(target.get());
            }
            Ev::Buttons(b) => input.set_buttons(b),
            Ev::Wheel(dz) => input.wheel(dz),
        }
    }
}
