//! Interrupt lines between devices and the controllers they report to.
//!
//! A device holds an [`IrqLine`]: which controller, and which of its inputs.
//! It drives the line to its level when its state may have changed, and the
//! controller decides what that means: a status bit behind a mask, an input
//! ORed with others, or a cascade into a controller further up (the IOC's
//! local and mappable sources into the CPU's interrupt pins). Devices no
//! longer need a callback trait of their own for each chip that collects
//! their interrupt.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use parking_lot::{Condvar, Mutex, MutexGuard};

/// Something interrupt lines lead into.
pub trait InterruptController: Send + Sync {
    /// Drive input `line` to `level`. An input the controller latches takes
    /// a rising level as its edge.
    fn set_line(&self, line: u32, level: bool);

    /// A crosstalk interrupt packet that widget `source` sent to `address`,
    /// naming vector `line`; whether the host took it. HEART and the Hub
    /// decode the vector alone, as this default does. A host whose packet
    /// address picks the destination overrides it.
    fn packet(&self, _source: u8, _address: u64, line: u32, level: bool) -> bool {
        self.set_line(line, level);
        true
    }
}

/// One input of one controller, as a device sees it.
#[derive(Clone)]
pub struct IrqLine {
    ctrl: Arc<dyn InterruptController>,
    line: u32,
}

impl IrqLine {
    pub fn new(ctrl: Arc<dyn InterruptController>, line: u32) -> Self {
        Self { ctrl, line }
    }

    /// A line that leads nowhere, for a device built on its own (tests).
    pub fn unconnected() -> Self {
        Self::new(Arc::new(Unconnected), 0)
    }

    pub fn set(&self, level: bool) {
        self.ctrl.set_line(self.line, level);
    }

    pub fn line(&self) -> u32 {
        self.line
    }
}

struct Unconnected;

/// A device state behind a lock, with the interrupt line it drives. The
/// guard drives the line from the state before the lock is let go (drop,
/// condvar wait, unlock), and only when the level changed. So the
/// controller sees every edge, in order, and never a fall and a rise merged
/// into one steady high.
///
/// Delivery runs under the device's lock. That is safe only while the
/// controller never calls back into the device while holding a lock the
/// delivery takes.
pub struct IrqMutex<S> {
    inner: Mutex<(S, Last)>,
    /// The level the state gives.
    level: fn(&S) -> bool,
}

/// The level last written to the line, or `None` when the controller's input
/// may not be what was last written, as after power-on.
#[derive(Default)]
struct Last(Option<bool>);

impl<S> IrqMutex<S> {
    pub fn new(state: S, level: fn(&S) -> bool) -> Self {
        Self { inner: Mutex::new((state, Last::default())), level }
    }

    /// Lock the state. `line` is where the device's line is set, which the
    /// bus may do while the guard is held: the guard looks it up each time it
    /// drives. A device without a line has nothing to drive and keeps what it
    /// remembers.
    pub fn lock<'a>(&'a self, line: &'a OnceLock<IrqLine>) -> IrqGuard<'a, S> {
        IrqGuard { guard: self.inner.lock(), line, level: self.level }
    }

    pub fn is_locked(&self) -> bool {
        self.inner.is_locked()
    }
}

pub struct IrqGuard<'a, S> {
    guard: MutexGuard<'a, (S, Last)>,
    line: &'a OnceLock<IrqLine>,
    level: fn(&S) -> bool,
}

impl<S> IrqGuard<'_, S> {
    fn drive(&mut self) {
        let Some(line) = self.line.get() else { return };
        let level = (self.level)(&self.guard.0);
        if self.guard.1.0 != Some(level) {
            self.guard.1.0 = Some(level);
            line.set(level);
        }
    }

    /// The controller's input may differ from what was last written: the
    /// device was wired to a new line, or its state was loaded over the run
    /// the controller knew. The next drive writes the line even when its level
    /// has not changed.
    pub fn forget(&mut self) {
        self.guard.1.0 = None;
    }

    /// Wait on `cv`, which lets go of the lock.
    pub fn wait(&mut self, cv: &Condvar) {
        self.drive();
        cv.wait(&mut self.guard);
    }

    /// Like [`IrqGuard::wait`], giving up after `timeout`.
    pub fn wait_for(&mut self, cv: &Condvar, timeout: Duration) {
        self.drive();
        cv.wait_for(&mut self.guard, timeout);
    }

    /// Run `f` with the lock let go.
    pub fn unlocked<R>(&mut self, f: impl FnOnce() -> R) -> R {
        self.drive();
        MutexGuard::unlocked(&mut self.guard, f)
    }
}

impl<S> std::ops::Deref for IrqGuard<'_, S> {
    type Target = S;
    fn deref(&self) -> &S { &self.guard.0 }
}

impl<S> std::ops::DerefMut for IrqGuard<'_, S> {
    fn deref_mut(&mut self) -> &mut S { &mut self.guard.0 }
}

impl<S> Drop for IrqGuard<'_, S> {
    /// Runs before the field `guard` drops, so before the lock is released.
    fn drop(&mut self) {
        self.drive();
    }
}

/// A controller with one input, for tests: the level last driven, and every
/// level driven.
#[cfg(test)]
#[derive(Default)]
pub struct LevelProbe(std::sync::Mutex<(bool, Vec<bool>)>);

#[cfg(test)]
impl LevelProbe {
    /// A probe and a line into it.
    pub fn line() -> (Arc<Self>, IrqLine) {
        let p = Arc::new(Self::default());
        (p.clone(), IrqLine::new(p, 0))
    }

    pub fn level(&self) -> bool {
        self.0.lock().unwrap().0
    }

    /// Every level the device drove, in order, repeats included.
    pub fn writes(&self) -> Vec<bool> {
        self.0.lock().unwrap().1.clone()
    }
}

#[cfg(test)]
impl InterruptController for LevelProbe {
    fn set_line(&self, _line: u32, level: bool) {
        let mut probe = self.0.lock().unwrap();
        probe.0 = level;
        probe.1.push(level);
    }
}

/// A [`LevelProbe`] that can hold one delivery on its way in, as a slow
/// parent would: [`StallingProbe::arm`] makes the next `set_line` report that
/// it has started and wait to be let go before it records its level. A test
/// uses it to see whether a second thread can change the level, and have its
/// own change delivered, while a first delivery is in flight.
#[cfg(test)]
pub struct StallingProbe {
    pub probe: Arc<LevelProbe>,
    stall: std::sync::Mutex<Option<(std::sync::mpsc::SyncSender<()>, std::sync::mpsc::Receiver<()>)>>,
}

#[cfg(test)]
impl StallingProbe {
    /// A stalling probe and a line into it.
    pub fn line() -> (Arc<Self>, IrqLine) {
        let s = Arc::new(Self { probe: Arc::new(LevelProbe::default()), stall: Default::default() });
        (s.clone(), IrqLine::new(s, 0))
    }

    /// Hold the next delivery. Returns the signal that it has begun and the
    /// release; a delivery nobody releases goes on after five seconds, so a
    /// failing test does not hang.
    pub fn arm(&self) -> (std::sync::mpsc::Receiver<()>, std::sync::mpsc::Sender<()>) {
        let (begun_tx, begun_rx) = std::sync::mpsc::sync_channel(1);
        let (go_tx, go_rx) = std::sync::mpsc::channel();
        *self.stall.lock().unwrap() = Some((begun_tx, go_rx));
        (begun_rx, go_tx)
    }
}

#[cfg(test)]
impl InterruptController for StallingProbe {
    fn set_line(&self, line: u32, level: bool) {
        let stall = self.stall.lock().unwrap().take();
        if let Some((begun, go)) = stall {
            begun.send(()).unwrap();
            let _ = go.recv_timeout(std::time::Duration::from_secs(5));
        }
        self.probe.set_line(line, level);
    }
}

impl InterruptController for Unconnected {
    fn set_line(&self, _line: u32, _level: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Recorder(Mutex<Vec<(u32, bool)>>);
    impl InterruptController for Recorder {
        fn set_line(&self, line: u32, level: bool) {
            self.0.lock().unwrap().push((line, level));
        }
    }

    #[test]
    fn a_line_drives_its_controllers_input() {
        let r = Arc::new(Recorder(Mutex::new(Vec::new())));
        let l = IrqLine::new(r.clone(), 5);
        l.set(true);
        l.clone().set(false);
        assert_eq!(*r.0.lock().unwrap(), vec![(5, true), (5, false)]);
        IrqLine::unconnected().set(true);
    }
}

#[cfg(test)]
mod irq_mutex_tests {
    use super::*;

    fn device() -> (Arc<LevelProbe>, OnceLock<IrqLine>, IrqMutex<bool>) {
        let (probe, line) = LevelProbe::line();
        (probe, OnceLock::from(line), IrqMutex::new(false, |s| *s))
    }

    #[test]
    fn dropping_the_guard_drives_a_change_and_only_a_change() {
        let (probe, line, dev) = device();
        *dev.lock(&line) = true;
        *dev.lock(&line) = true;
        drop(dev.lock(&line));
        *dev.lock(&line) = false;
        assert_eq!(probe.writes(), [true, false]);
    }

    #[test]
    fn a_forgotten_line_is_written_again() {
        let (probe, line, dev) = device();
        *dev.lock(&line) = true;
        dev.lock(&line).forget();
        drop(dev.lock(&line));
        assert_eq!(probe.writes(), [true, true]);
    }

    #[test]
    fn a_device_without_a_line_keeps_what_it_remembers() {
        let (probe, line, dev) = device();
        let unwired = OnceLock::new();
        *dev.lock(&unwired) = true;
        let _ = unwired.set(line.get().unwrap().clone());
        drop(dev.lock(&unwired));
        assert_eq!(probe.writes(), [true]);
    }

    /// A change is on the line while the lock is let go, not after.
    #[test]
    fn unlocking_for_a_while_drives_first() {
        let (probe, line, dev) = device();
        let mut g = dev.lock(&line);
        *g = true;
        assert!(g.unlocked(|| probe.level()));
    }

    #[test]
    fn waiting_drives_first() {
        let (probe, line, dev) = device();
        let cv = Condvar::new();
        std::thread::scope(|scope| {
            let mut g = dev.lock(&line);
            *g = true;
            scope.spawn(|| {
                let end = std::time::Instant::now() + Duration::from_secs(5);
                while !probe.level() && std::time::Instant::now() < end {
                    std::thread::yield_now();
                }
                assert!(probe.level(), "the line was not driven while the lock was waited out");
                cv.notify_all();
            });
            g.wait_for(&cv, Duration::from_secs(5));
        });
        assert_eq!(probe.writes(), [true]);
    }
}
