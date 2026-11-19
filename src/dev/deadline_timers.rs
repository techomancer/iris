//! One-shot timers for the host-time deadlines a device keeps.
//!
//! A device evaluates its deadlines lazily, on every access, and arms one
//! timer per deadline so the work still happens when the guest is idle. The
//! IOC3 and IOC4 do this for their serial ring timers and transmit delays
//! (and the IOC3 for its interrupt output and NVRAM debounce). This holds the
//! bookkeeping: which timer serves which deadline, replacing one whose
//! deadline moved, and taking them all away on stop.
//!
//! Most accesses move no deadline, so the deadlines armed are mirrored in
//! atomics and a pass that would change nothing returns without taking the
//! table's lock or touching the timer manager.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::dev::deadline::{encode, start_epoch};
use crate::hptimer::{TimerId, TimerManager, TimerReturn};

/// The timers of a device with `N` kinds of deadline.
pub struct DeadlineTimers<const N: usize> {
    manager: OnceLock<Arc<TimerManager>>,
    /// The timer pending for each kind, and the deadline it serves.
    armed: Mutex<[Option<(TimerId, Instant)>; N]>,
    /// The deadlines in `armed`, `deadline::encode`d, readable without its
    /// lock. Written only under it.
    armed_due: [AtomicU64; N],
}

impl<const N: usize> DeadlineTimers<N> {
    pub fn new() -> Self {
        // The encoding counts from a process epoch; fix it before any
        // deadline a device holds, so none falls before it.
        start_epoch();
        Self { manager: OnceLock::new(), armed: Mutex::new([None; N]), armed_due: std::array::from_fn(|_| AtomicU64::new(0)) }
    }

    /// Whether the timers pending already serve `want` exactly. Read without
    /// the table's lock: a pass that gets a stale answer here was raced by a
    /// writer that runs a pass of its own.
    fn unchanged(&self, want: &[Option<Instant>; N]) -> bool {
        want.iter().zip(&self.armed_due).all(|(w, a)| encode(*w) == a.load(Ordering::SeqCst))
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
        // With the device still held no write can move a deadline, so a
        // pass that would change nothing can let go here.
        if self.unchanged(want) {
            return;
        }
        let mut armed = self.armed.lock();
        drop(held);
        // Checked under the lock `cancel_all` clears the slots with: a pass
        // that sees `running` set has its timers removed by that stop.
        if !running.load(Ordering::SeqCst) {
            return;
        }
        self.apply(&mut armed, want, tm, device, floor, fire);
    }

    /// `arm` for a device whose deadlines are read from several parts, each
    /// publishing its own under its own lock, with no lock over all of them:
    /// `want` reads them. They are read again once the table is held, never
    /// only before, so a pass held up between reading and arming cannot arm
    /// from deadlines older than those a later access left and remove the
    /// timer that access set.
    pub fn arm_with<S: Send + Sync + 'static>(
        &self,
        want: impl Fn() -> [Option<Instant>; N],
        running: &AtomicBool,
        device: &Weak<S>,
        floor: fn(usize) -> Duration,
        fire: fn(&Weak<S>, usize, Instant) -> TimerReturn,
    ) {
        let Some(tm) = self.manager.get() else { return };
        if self.unchanged(&want()) {
            return;
        }
        let mut armed = self.armed.lock();
        if !running.load(Ordering::SeqCst) {
            return;
        }
        let want = want();
        self.apply(&mut armed, &want, tm, device, floor, fire);
    }

    fn apply<S: Send + Sync + 'static>(
        &self,
        armed: &mut [Option<(TimerId, Instant)>; N],
        want: &[Option<Instant>; N],
        tm: &TimerManager,
        device: &Weak<S>,
        floor: fn(usize) -> Duration,
        fire: fn(&Weak<S>, usize, Instant) -> TimerReturn,
    ) {
        for (kind, w) in want.iter().enumerate() {
            if armed[kind].map(|(_, d)| d) == *w {
                continue;
            }
            if let Some((id, _)) = armed[kind].take() {
                tm.remove(id);
            }
            self.armed_due[kind].store(0, Ordering::SeqCst);
            let Some(due) = *w else { continue };
            let delay = due.saturating_duration_since(Instant::now()).max(floor(kind));
            let id = tm.add_one_shot(delay, device.clone(), move |d| fire(d, kind, due));
            armed[kind] = Some((id, due));
            self.armed_due[kind].store(encode(Some(due)), Ordering::SeqCst);
        }
    }

    /// A timer for `kind` and `due` has fired: forget it, unless a newer
    /// deadline has replaced it.
    pub fn fired(&self, kind: usize, due: Instant) {
        let mut armed = self.armed.lock();
        if armed[kind].map(|(_, d)| d) == Some(due) {
            armed[kind] = None;
            self.armed_due[kind].store(0, Ordering::SeqCst);
        }
    }

    /// The deadline the timer pending for `kind` serves.
    #[cfg(test)]
    pub fn armed_for(&self, kind: usize) -> Option<Instant> {
        self.armed.lock()[kind].map(|(_, d)| d)
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
        for (slot, due) in armed.iter_mut().zip(&self.armed_due) {
            if let (Some((id, _)), Some(tm)) = (slot.take(), self.manager.get()) {
                tm.remove(id);
            }
            due.store(0, Ordering::SeqCst);
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

    #[test]
    fn a_pass_that_moves_nothing_leaves_the_table_alone() {
        let d = device();
        let me = Arc::downgrade(&d);
        let far = Some(Instant::now() + Duration::from_secs(3600));
        d.timers.arm(&[far, None], (), &d.running, &me, |_| Duration::ZERO, fire);
        let table = d.timers.armed.lock();
        // Would block on the table if it went past the unlocked check.
        d.timers.arm(&[far, None], (), &d.running, &me, |_| Duration::ZERO, fire);
        d.timers.arm_with(|| [far, None], &d.running, &me, |_| Duration::ZERO, fire);
        drop(table);
        d.timers.arm_with(|| [None, far], &d.running, &me, |_| Duration::ZERO, fire);
        assert_eq!(d.timers.armed_for(0), None);
        assert_eq!(d.timers.armed_for(1), far);
        d.timers.cancel_all();
        assert!(d.timers.none_armed());
        // Cleared mirrors: the same deadlines arm again.
        d.timers.arm_with(|| [None, far], &d.running, &me, |_| Duration::ZERO, fire);
        assert_eq!(d.timers.armed_for(1), far);
        d.timers.cancel_all();
    }

    /// `arm_with` reads the deadlines again under the table: what it arms is
    /// what the device holds once it has the table, not what it read first.
    #[test]
    fn arm_with_arms_what_the_device_holds_under_the_table() {
        let d = device();
        let me = Arc::downgrade(&d);
        let chip: Mutex<Option<Instant>> = Mutex::new(Some(Instant::now() + Duration::from_secs(3600)));
        let newer = Instant::now() + Duration::from_secs(7200);
        std::thread::scope(|s| {
            let table = d.timers.armed.lock();
            let pass = s.spawn(|| d.timers.arm_with(|| [*chip.lock(), None], &d.running, &me, |_| Duration::ZERO, fire));
            std::thread::sleep(Duration::from_millis(20));
            *chip.lock() = Some(newer);
            drop(table);
            pass.join().unwrap();
        });
        assert_eq!(d.timers.armed_for(0), Some(newer));
        d.timers.cancel_all();
    }

    /// A pass that read the device before a write cannot arm after that
    /// write from what it read, and so cannot remove the timer the write
    /// set. The test holds the table so the pass stops just short of arming;
    /// the write must then wait for it.
    #[test]
    fn a_pass_that_read_the_device_first_arms_first() {
        let d = device();
        let me = Arc::downgrade(&d);
        // The chip holds a deadline no timer serves yet, so the older pass
        // has something to arm and goes past the unlocked check.
        let chip: Mutex<Option<Instant>> = Mutex::new(Some(Instant::now() + Duration::from_secs(1800)));
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
