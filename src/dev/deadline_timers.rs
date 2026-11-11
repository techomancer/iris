//! One-shot timers for the host-time deadlines a device keeps.
//!
//! A device evaluates its deadlines lazily, on every access, and arms one
//! timer per deadline so the work still happens when the guest is idle. The
//! IOC3 and IOC4 do this for their serial ring timers and transmit delays,
//! and the IOC3 for its interrupt output. This holds the part they share:
//! which timer serves which deadline, replacing one whose deadline moved,
//! and taking them all away on stop.
//!
//! QEMU has no counterpart: its timers belong to the virtual clock and are
//! armed where they are set.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::hptimer::{TimerId, TimerManager, TimerReturn};

/// The timers of a device with `N` kinds of deadline.
pub struct DeadlineTimers<const N: usize> {
    manager: OnceLock<Arc<TimerManager>>,
    /// The timer pending for each kind, and the deadline it serves.
    armed: Mutex<[Option<(TimerId, Instant)>; N]>,
}

impl<const N: usize> DeadlineTimers<N> {
    pub fn new() -> Self {
        Self { manager: OnceLock::new(), armed: Mutex::new([None; N]) }
    }

    /// The timer manager to arm on. Set it before the device starts.
    pub fn set_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.manager.set(tm);
    }

    /// Make the pending timers match `want`: one one-shot per deadline. A
    /// timer whose deadline moved is replaced. With no manager set, or the
    /// device not `running`, deadlines are only evaluated lazily, by the next
    /// access. `floor` is the least delay for a kind, for a deadline that can
    /// recur faster than a timer should fire; `fire` runs on the timer thread
    /// with the device and the kind and deadline it was armed for.
    ///
    /// `held` is the device lock `want` was read under. It is let go only
    /// once the table is taken, so passes set the timers in the order they
    /// saw the device. Otherwise a pass that read the device just before a
    /// register write could arm after that write, from the older deadlines,
    /// and remove the timer the write set.
    pub fn arm<S: Send + Sync + 'static, G>(
        &self,
        want: &[Option<Instant>; N],
        held: G,
        running: &AtomicBool,
        device: &Weak<S>,
        floor: fn(usize) -> Duration,
        fire: fn(&Weak<S>, usize, Instant) -> TimerReturn,
    ) {
        let Some(tm) = self.manager.get() else { return };
        let mut armed = self.armed.lock();
        drop(held);
        // Checked under the lock `cancel_all` clears the slots with: a pass
        // that sees `running` set has its timers removed by that stop.
        if !running.load(Ordering::SeqCst) {
            return;
        }
        for (kind, w) in want.iter().enumerate() {
            if armed[kind].map(|(_, d)| d) == *w {
                continue;
            }
            if let Some((id, _)) = armed[kind].take() {
                tm.remove(id);
            }
            let Some(due) = *w else { continue };
            let delay = due.saturating_duration_since(Instant::now()).max(floor(kind));
            let id = tm.add_one_shot(delay, device.clone(), move |d| fire(d, kind, due));
            armed[kind] = Some((id, due));
        }
    }

    /// A timer for `kind` and `due` has fired: forget it, unless a newer
    /// deadline has replaced it.
    pub fn fired(&self, kind: usize, due: Instant) {
        let mut armed = self.armed.lock();
        if armed[kind].map(|(_, d)| d) == Some(due) {
            armed[kind] = None;
        }
    }

    /// Whether a timer is pending for `kind`.
    #[cfg(test)]
    pub fn is_armed(&self, kind: usize) -> bool {
        self.armed.lock()[kind].is_some()
    }

    /// Whether no timer is pending.
    #[cfg(test)]
    pub fn none_armed(&self) -> bool {
        self.armed.lock().iter().all(|a| a.is_none())
    }

    /// Remove every pending timer.
    pub fn cancel_all(&self) {
        let mut armed = self.armed.lock();
        for slot in armed.iter_mut() {
            if let (Some((id, _)), Some(tm)) = (slot.take(), self.manager.get()) {
                tm.remove(id);
            }
        }
    }
}

impl<const N: usize> Default for DeadlineTimers<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct Device {
        timers: DeadlineTimers<2>,
        running: AtomicBool,
        fired: [AtomicUsize; 2],
    }

    fn fire(me: &Weak<Device>, kind: usize, due: Instant) -> TimerReturn {
        if let Some(d) = me.upgrade() {
            d.timers.fired(kind, due);
            d.fired[kind].fetch_add(1, Ordering::SeqCst);
        }
        TimerReturn::Delete
    }

    fn device() -> Arc<Device> {
        let d = Arc::new(Device {
            timers: DeadlineTimers::new(),
            running: AtomicBool::new(true),
            fired: Default::default(),
        });
        d.timers.set_manager(Arc::new(TimerManager::new()));
        d
    }

    fn wait_for(f: impl Fn() -> bool) -> bool {
        for _ in 0..500 {
            if f() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        false
    }

    #[test]
    fn a_deadline_fires_once_and_its_slot_empties() {
        let d = device();
        let due = Instant::now() + Duration::from_millis(5);
        d.timers.arm(&[Some(due), None], (), &d.running, &Arc::downgrade(&d), |_| Duration::ZERO, fire);
        assert!(d.timers.is_armed(0));
        assert!(!d.timers.is_armed(1));
        assert!(wait_for(|| d.fired[0].load(Ordering::SeqCst) == 1));
        assert!(d.timers.none_armed());
        assert_eq!(d.fired[1].load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_moved_deadline_replaces_its_timer() {
        let d = device();
        let me = Arc::downgrade(&d);
        let far = Instant::now() + Duration::from_secs(3600);
        d.timers.arm(&[Some(far), None], (), &d.running, &me, |_| Duration::ZERO, fire);
        let near = Instant::now() + Duration::from_millis(5);
        d.timers.arm(&[Some(near), None], (), &d.running, &me, |_| Duration::ZERO, fire);
        assert!(wait_for(|| d.fired[0].load(Ordering::SeqCst) == 1));
        assert_eq!(d.fired[0].load(Ordering::SeqCst), 1, "the far timer was removed, not left to fire");
    }

    #[test]
    fn a_stopped_device_arms_nothing_and_cancel_clears_what_was_armed() {
        let d = device();
        let me = Arc::downgrade(&d);
        let far = Some(Instant::now() + Duration::from_secs(3600));
        d.running.store(false, Ordering::SeqCst);
        d.timers.arm(&[far, far], (), &d.running, &me, |_| Duration::ZERO, fire);
        assert!(d.timers.none_armed());
        d.running.store(true, Ordering::SeqCst);
        d.timers.arm(&[far, far], (), &d.running, &me, |_| Duration::ZERO, fire);
        assert!(d.timers.is_armed(0) && d.timers.is_armed(1));
        d.timers.cancel_all();
        assert!(d.timers.none_armed());
    }

    #[test]
    fn the_floor_delays_a_kind_that_recurs_quickly() {
        let d = device();
        let now = Instant::now();
        d.timers.arm(&[Some(now), Some(now)], (), &d.running, &Arc::downgrade(&d), |k| {
            if k == 0 { Duration::from_millis(40) } else { Duration::ZERO }
        }, fire);
        assert!(wait_for(|| d.fired[1].load(Ordering::SeqCst) == 1));
        assert_eq!(d.fired[0].load(Ordering::SeqCst), 0, "kind 0 is held back by its floor");
        assert!(wait_for(|| d.fired[0].load(Ordering::SeqCst) == 1));
    }

    /// A pass that read the device before a write cannot arm after that
    /// write from what it read, and so cannot remove the timer the write
    /// set. The test holds the table so the pass stops just short of arming;
    /// the write must then wait for it.
    #[test]
    fn a_pass_that_read_the_device_first_arms_first() {
        let d = device();
        let me = Arc::downgrade(&d);
        let chip: Mutex<Option<Instant>> = Mutex::new(None);
        let due = Instant::now() + Duration::from_secs(3600);
        let pass = |set: Option<Instant>| {
            let mut st = chip.lock();
            if set.is_some() {
                *st = set;
            }
            d.timers.arm(&[*st, None], st, &d.running, &me, |_| Duration::ZERO, fire);
        };
        std::thread::scope(|s| {
            let table = d.timers.armed.lock();
            let old = s.spawn(|| pass(None));
            assert!(wait_for(|| chip.is_locked()), "the pass keeps the device until it has the table");
            let write = s.spawn(|| pass(Some(due)));
            drop(table);
            old.join().unwrap();
            write.join().unwrap();
        });
        assert_eq!(d.timers.armed.lock()[0].map(|(_, at)| at), Some(due));
        d.timers.cancel_all();
    }
}
