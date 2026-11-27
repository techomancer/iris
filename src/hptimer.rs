use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::thread;
use std::time::{Duration, Instant};

use spin::Mutex;

use crate::clock::Clock;

/// Unique identifier for a timer.
pub type TimerId = u64;

/// Action returned by a timer callback upon expiration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimerReturn {
    /// Keep the timer as is. If it's recurring, it schedules the next expiration by adding its period.
    /// If it's a one-shot timer, it gets deleted.
    Continue,
    /// Remove the timer from the manager completely.
    Delete,
    /// Reschedule the timer to trigger exactly once at the given relative Duration from now.
    RescheduleOneShot(Duration),
}

/// How far short of a deadline the worker stops sleeping and starts spinning.
/// A park overshoots by the kernel's timer slack plus scheduling latency, so
/// the margin has to exceed that for the deadline to be met.
const SPIN_MARGIN: Duration = Duration::from_micros(50);
/// Waits at or under this are spun outright: a sleep would end after them.
const SPIN_THRESHOLD: Duration = Duration::from_micros(80);

/// A Timer entry in the manager.
struct TimerEntry {
    next_expiration: Instant,
    period: Option<Duration>,
    in_fifo: bool,
    callback: Option<Box<dyn FnMut() -> TimerReturn + Send>>,
}

/// Represents a slot in our arena.
struct Slot {
    generation: u32,
    timer: Option<TimerEntry>,
}

/// Internal state shielded by a Spinlock Mutex.
struct TimerManagerInner {
    slots: Vec<Slot>,
    free_indices: Vec<usize>,
    fifo: VecDeque<(usize, u32)>, // Stores (index, generation) of expired timers waiting to be called
    stop: bool,
    /// Wakes the worker out of its wait (`Waiter`), once the worker has made one.
    waker: Option<Waker>,
}

impl TimerManagerInner {
    fn allocate(&mut self, entry: TimerEntry) -> TimerId {
        let index = if let Some(idx) = self.free_indices.pop() {
            idx
        } else {
            let idx = self.slots.len();
            self.slots.push(Slot {
                generation: 0,
                timer: None,
            });
            idx
        };

        // Increment generation on each allocation to prevent ABA problems.
        self.slots[index].generation = self.slots[index].generation.wrapping_add(1);
        let generation = self.slots[index].generation;
        self.slots[index].timer = Some(entry);

        ((generation as u64) << 32) | (index as u64)
    }

    fn deallocate(&mut self, id: TimerId) -> bool {
        let (generation, idx) = ((id >> 32) as u32, (id & 0xFFFFFFFF) as usize);
        if let Some(slot) = self.slots.get_mut(idx) {
            if slot.generation == generation && slot.timer.is_some() {
                slot.timer = None;
                self.free_indices.push(idx);
                return true;
            }
        }
        false
    }
}

/// A high precision timer manager driven by a background spin-thread.
pub struct TimerManager {
    inner: Arc<Mutex<TimerManagerInner>>,
    thread_handle: Option<thread::JoinHandle<()>>,
    new_timer_added: Arc<AtomicBool>,
    clock: Clock,
    /// Every `Timer` built on this manager, for `run_due`.
    #[cfg(test)]
    live: Mutex<Vec<Weak<TimerInner>>>,
    /// Host timers never fire: only `run_due` does, so a test sees one order and no race.
    #[cfg(test)]
    pub(crate) manual: AtomicBool,
}

impl TimerManager {
    /// Create a new `TimerManager` and start its worker thread.
    pub fn new() -> Self {
        Self::with_clock(Clock::new())
    }

    pub fn clock(&self) -> &Clock {
        &self.clock
    }

    /// The manager owns the machine clock that every guest-visible clock reads; its worker waits on host time.
    pub fn with_clock(clock: Clock) -> Self {
        let inner = Arc::new(Mutex::new(TimerManagerInner {
            slots: Vec::new(),
            free_indices: Vec::new(),
            fifo: VecDeque::new(),
            stop: false,
            waker: None,
        }));

        let new_timer_added = Arc::new(AtomicBool::new(false));

        let thread_inner = inner.clone();
        let thread_added = new_timer_added.clone();

        let thread_handle = thread::Builder::new().name("hptimer".into()).spawn(move || {
            let waiter = Waiter::new();
            {
                let mut guard = thread_inner.lock();
                guard.waker = Some(waiter.waker());
            }
            timer_thread_loop(thread_inner, thread_added, &waiter);
        }).expect("spawn hptimer thread");

        Self {
            inner,
            thread_handle: Some(thread_handle),
            new_timer_added,
            clock,
            #[cfg(test)]
            live: Mutex::new(Vec::new()),
            #[cfg(test)]
            manual: AtomicBool::new(false),
        }
    }

    /// Run the callback of every started timer the clock has reached, oldest deadline first and
    /// on this thread, until none is left: what the host timer does, without waiting for it.
    #[cfg(test)]
    pub(crate) fn run_due(&self) {
        loop {
            let now = self.clock.now();
            let mut live = self.live.lock();
            live.retain(|w| w.strong_count() > 0);
            let next = live
                .iter()
                .filter_map(Weak::upgrade)
                .filter_map(|t| {
                    let due = {
                        let st = t.state.lock();
                        st.due.filter(|d| *d <= now && st.running)?
                    };
                    Some((due, t))
                })
                .min_by_key(|(due, _)| *due);
            drop(live);
            let Some((_, timer)) = next else { return };
            timer.expire();
        }
    }

    /// Stop the background timer thread and clean up.
    pub fn stop(&mut self) {
        {
            let mut inner = self.inner.lock();
            inner.stop = true;
        }
        self.wake_up();
        if let Some(h) = self.thread_handle.take() {
            let _ = h.join();
        }
    }

    /// Add a new one-shot timer to the manager counting from now.
    pub fn add_one_shot<F>(&self, expire_in: Duration, cb: F) -> TimerId
    where
        F: FnMut() -> TimerReturn + Send + 'static,
    {
        let mut inner = self.inner.lock();
        let id = inner.allocate(TimerEntry {
            next_expiration: Instant::now() + expire_in,
            period: None,
            in_fifo: false,
            callback: Some(Box::new(cb)),
        });
        drop(inner);
        self.wake_up();
        id
    }

    /// Add a recurring timer to the manager.
    pub fn add_recurring<F>(&self, expire_at: Instant, period: Duration, cb: F) -> TimerId
    where
        F: FnMut() -> TimerReturn + Send + 'static,
    {
        let mut inner = self.inner.lock();
        let id = inner.allocate(TimerEntry {
            next_expiration: expire_at,
            period: Some(period),
            in_fifo: false,
            callback: Some(Box::new(cb)),
        });
        drop(inner);
        self.wake_up();
        id
    }

    /// Remove a timer entirely.
    pub fn remove(&self, id: TimerId) -> bool {
        let mut inner = self.inner.lock();
        inner.deallocate(id)
    }

    fn wake_up(&self) {
        self.new_timer_added.store(true, Ordering::Release);
        let inner = self.inner.lock();
        if let Some(w) = &inner.waker {
            w.wake();
        }
    }
}

impl Default for TimerManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TimerManager {
    fn drop(&mut self) {
        self.stop();
    }
}

/// One deadline in machine nanoseconds, owned by one device.
pub struct Timer(Arc<TimerInner>);

struct TimerInner {
    tm: Weak<TimerManager>,
    clock: Clock,
    callback: Box<dyn Fn(u64) + Send + Sync>,
    state: parking_lot::Mutex<TimerState>,
    /// `state.due` as a word (see `word`), so a repeated deadline skips the lock.
    due_word: AtomicU64,
    /// Held while the callback runs, so `stop` can wait one out.
    gate: parking_lot::Mutex<()>,
}

struct TimerState {
    due: Option<u64>,
    /// The host timer pending for `due`.
    id: Option<TimerId>,
    running: bool,
    /// The callback is running: a re-arm from it reuses the host timer.
    firing: bool,
}

impl Timer {
    /// `callback(device, due)` runs on the timer thread once the machine clock reaches `due`;
    /// it resyncs under the device's lock.
    pub fn new<S, F>(tm: &Arc<TimerManager>, me: &Weak<S>, callback: F) -> Self
    where
        S: Send + Sync + 'static,
        F: Fn(&S, u64) + Send + Sync + 'static,
    {
        let me = me.clone();
        let inner = Arc::new(TimerInner {
            tm: Arc::downgrade(tm),
            clock: tm.clock().clone(),
            callback: Box::new(move |due| {
                if let Some(s) = me.upgrade() {
                    callback(&s, due);
                }
            }),
            state: parking_lot::Mutex::new(TimerState { due: None, id: None, running: false, firing: false }),
            due_word: AtomicU64::new(crate::clock::encode(None)),
            gate: parking_lot::Mutex::new(()),
        });
        #[cfg(test)]
        tm.live.lock().push(Arc::downgrade(&inner));
        Self(inner)
    }

    /// Arm, or re-arm, for absolute machine time `Some(when)`; `None` forgets the
    /// deadline, and a callback already running is not waited for.
    pub fn set_due(&self, due: Option<u64>) {
        if self.0.due_word.load(Ordering::Relaxed) == crate::clock::encode(due) {
            return;
        }
        let mut st = self.0.state.lock();
        self.0.due_word.store(crate::clock::encode(due), Ordering::Relaxed);
        let old = std::mem::replace(&mut st.due, due);
        // A later deadline keeps the pending host timer, which `expire` re-arms for the rest;
        // re-timing costs the storing CPU microseconds, and the guest sees them pass.
        let later = st.id.is_some() && matches!((old, due), (Some(o), Some(n)) if n > o);
        if old != due && st.running && !st.firing && !later {
            self.0.retime(&mut st);
        }
    }

    /// The deadline armed, started or not.
    #[cfg(test)]
    pub fn due(&self) -> Option<u64> {
        self.0.state.lock().due
    }

    #[cfg(test)]
    pub fn is_armed(&self) -> bool {
        self.due().is_some()
    }

    /// The manager this timer runs on, while it lives.
    pub fn manager(&self) -> Option<Arc<TimerManager>> {
        self.0.tm.upgrade()
    }

    /// The machine clock, in nanoseconds.
    pub fn now(&self) -> u64 {
        self.0.clock.now()
    }

    /// From `Device::start`: arm the deadline `stop` kept or `set_due` set.
    pub fn start(&self) {
        let mut st = self.0.state.lock();
        st.running = true;
        self.0.retime(&mut st);
    }

    /// Remove the host timer, keep the deadline, wait out a running callback.
    /// Never from a callback or under a lock it takes.
    pub fn stop(&self) {
        {
            let mut st = self.0.state.lock();
            st.running = false;
            self.0.unarm(&mut st);
        }
        drop(self.0.gate.lock());
    }
}

impl TimerInner {
    /// Make the host timer serve `st.due`.
    fn retime(self: &Arc<Self>, st: &mut TimerState) {
        self.unarm(st);
        let (Some(due), Some(tm)) = (st.due, self.tm.upgrade()) else { return };
        let delay = Duration::from_nanos(due.saturating_sub(self.clock.now()));
        let w = Arc::downgrade(self);
        st.id = Some(tm.add_one_shot(delay, move || {
            w.upgrade().map_or(TimerReturn::Delete, |t| {
                #[cfg(test)]
                if t.tm.upgrade().is_some_and(|tm| tm.manual.load(Ordering::Relaxed)) {
                    t.state.lock().id = None;
                    return TimerReturn::Delete;
                }
                t.expire()
            })
        }));
    }

    fn unarm(&self, st: &mut TimerState) {
        if let (Some(id), Some(tm)) = (st.id.take(), self.tm.upgrade()) {
            tm.remove(id);
        }
    }

    /// A host timer is up. A cancelled deadline or a stopped timer does
    /// nothing; one that ran ahead of the machine clock waits out the rest.
    fn expire(self: &Arc<Self>) -> TimerReturn {
        let _gate = self.gate.lock();
        let due = {
            let mut st = self.state.lock();
            let Some(due) = st.due.filter(|_| st.running) else { return TimerReturn::Delete };
            let now = self.clock.now();
            if now < due {
                return TimerReturn::RescheduleOneShot(Duration::from_nanos(due - now));
            }
            st.due = None;
            self.due_word.store(crate::clock::encode(None), Ordering::Relaxed);
            st.firing = true;
            due
        };
        (self.callback)(due);
        let mut st = self.state.lock();
        st.firing = false;
        match st.due {
            Some(next) if st.running => {
                TimerReturn::RescheduleOneShot(Duration::from_nanos(next.saturating_sub(self.clock.now())))
            }
            _ => {
                st.id = None;
                TimerReturn::Delete
            }
        }
    }
}

impl Drop for TimerInner {
    fn drop(&mut self) {
        self.unarm(&mut self.state.lock());
    }
}

/// The core loop tracking timeouts precisely.
fn timer_thread_loop(inner: Arc<Mutex<TimerManagerInner>>, new_timer_added: Arc<AtomicBool>, waiter: &Waiter) {
    loop {
        let now = Instant::now();
        let mut next_wakeup = None;
        let mut to_call = None;

        {
            let mut guard = inner.lock();
            if guard.stop {
                break;
            }
            new_timer_added.store(false, Ordering::Relaxed);

            // 1. Scan for newly expired timers and register them in the FIFO to keep priority.
            for index in 0..guard.slots.len() {
                let slot = &mut guard.slots[index];
                if let Some(t) = &mut slot.timer {
                    if !t.in_fifo && t.next_expiration <= now {
                        t.in_fifo = true;
                        let generation = slot.generation;
                        guard.fifo.push_back((index, generation));
                    }
                }
            }

            // 2. Fetch the highest priority (oldest expired) timer callback.
            if let Some((idx, generation)) = guard.fifo.pop_front() {
                if let Some(slot) = guard.slots.get_mut(idx) {
                    if slot.generation == generation {
                        if let Some(t) = &mut slot.timer {
                            let cb = t.callback.take();
                            to_call = Some((idx, generation, cb, t.period, t.next_expiration));
                            // Processed, take it out of the fifo queue logic
                            t.in_fifo = false;
                        }
                    }
                }
            }

            // 3. Evaluate next wakeup if we don't have something to call right away.
            if to_call.is_none() {
                for slot in guard.slots.iter() {
                    if let Some(t) = &slot.timer {
                        if !t.in_fifo {
                            if let Some(w) = next_wakeup {
                                if t.next_expiration < w {
                                    next_wakeup = Some(t.next_expiration);
                                }
                            } else {
                                next_wakeup = Some(t.next_expiration);
                            }
                        }
                    }
                }
            }
        }

        // 4. Run the fetched callback (with late catch-up for recurring timers).
        const MAX_CATCHUP: u32 = 8;
        if let Some((idx, generation, cb_opt, _og_period, _og_expiration)) = to_call {
            if let Some(mut cb) = cb_opt {
                let mut result = cb();
                let mut catch_up = 0u32;

                loop {
                    let mut guard = inner.lock();
                    if guard.stop {
                        break;
                    }

                    if let Some(slot) = guard.slots.get_mut(idx) {
                        if slot.generation != generation || slot.timer.is_none() {
                            break;
                        }
                        let t = slot.timer.as_mut().unwrap();
                        t.callback = Some(cb);

                        let period = t.period;
                        match result {
                            TimerReturn::Continue => {
                                if let Some(period) = period {
                                    t.next_expiration += period;
                                    let now = Instant::now();
                                    if catch_up + 1 < MAX_CATCHUP
                                        && t.next_expiration <= now
                                    {
                                        catch_up += 1;
                                        cb = t.callback.take().unwrap();
                                        drop(guard);
                                        result = cb();
                                        continue;
                                    }
                                } else {
                                    guard.deallocate(((generation as u64) << 32) | (idx as u64));
                                }
                            }
                            TimerReturn::Delete => {
                                guard.deallocate(((generation as u64) << 32) | (idx as u64));
                            }
                            TimerReturn::RescheduleOneShot(new_delay) => {
                                t.next_expiration = Instant::now() + new_delay;
                                t.period = None;
                            }
                        }
                    }
                    break;
                }
            }
            continue; // Re-evaluate time continuously
        }

        // 5. Spin or park depending on how long we have to wait
        if let Some(target) = next_wakeup {
            loop {
                let sleep_now = Instant::now();
                if sleep_now >= target || new_timer_added.load(Ordering::Acquire) {
                    break; // Ready to wake or there's intermediate activity
                }

                let delay = target - sleep_now;

                // Sleep to just short of the deadline, then spin the rest.
                // The margin covers the kernel's timer slack and wake-up
                // latency (tens of microseconds), so the spin is bounded by
                // it and a periodic timer at 1 kHz costs a few percent of a
                // core rather than all of it. A waiter whose sleeps land on
                // a grid (Windows, 0.5 ms) may shorten the sleep to the grid
                // point before the deadline and leave the rest to the spin
                // (`Waiter::plan`), or ask for a spin outright.
                let request = if delay > SPIN_THRESHOLD { waiter.plan(delay - SPIN_MARGIN) } else { Duration::ZERO };
                if !request.is_zero() {
                    waiter.wait(request);
                } else {
                    // Audio-scale waits (~23 µs at 44.1 kHz): spin to avoid 50 µs sleep floor.
                    let deadline = Instant::now() + delay;
                    while Instant::now() < deadline
                        && !new_timer_added.load(Ordering::Acquire)
                    {
                        std::hint::spin_loop();
                    }
                }
            }
        } else {
            // Nothing to wake up for, indefinite sleep until woken.
            if !new_timer_added.load(Ordering::Acquire) {
                waiter.wait_forever();
            }
        }
    }
}

/// The longest the worker spins to land a deadline that falls between two
/// points of a gridded waiter's sleep (`Waiter::plan`). Past it the deadline
/// is taken late at the next grid point instead. Measured on the 0.5 ms
/// Windows grid (clanker/perf/windows-timer-waits-round-to-the-system-tick.md):
/// a 1 kHz recurring timer runs a mean 62 µs late for 4.7% of a core at
/// this budget, 207 µs late for 0.8% at 250 µs, 253 µs late with no
/// planning, and on time for 38% of a core when the whole grid step may be
/// spun; a one-shot re-armed from its callback (the guest's Compare
/// interrupt) runs 35-60 µs late at any budget.
#[cfg_attr(not(windows), allow(dead_code))] // only a gridded waiter plans
const SPIN_MAX: Duration = Duration::from_micros(400);

// The worker's wait. A bounded wait that a `Waker` can end early, from any
// thread, and that remembers a wake delivered while the worker is not
// waiting (the token `unpark` keeps). On Windows `park_timeout` is
// `WaitOnAddress`, whose timeout is rounded up to the system timer's tick:
// 15.625 ms unless some process raised the resolution. A 1 kHz guest clock
// slept to 50 µs short of each deadline then ran at 64-80 Hz, and every
// tick-paced wait in the guest took ten times as long (IRIX on the Octane
// after a Quake timedemo looked hung). The Windows waiter is a
// high-resolution waitable timer (Windows 10 1803 and later), which keeps
// its own deadline off the system tick, waited on together with an event
// the waker signals. Measured on Windows 11 (clanker/perf/
// windows-timer-waits-round-to-the-system-tick.md): it fires on a 0.5 ms
// grid counted from the request, 25-80 µs after the grid point, whatever
// the system tick; `plan` works around the grid.

#[cfg(not(windows))]
pub(crate) struct Waiter {
    thread: thread::Thread,
}

#[cfg(not(windows))]
#[derive(Clone)]
pub(crate) struct Waker {
    thread: thread::Thread,
}

#[cfg(not(windows))]
impl Waiter {
    fn new() -> Self {
        Self { thread: thread::current() }
    }

    fn waker(&self) -> Waker {
        Waker { thread: self.thread.clone() }
    }

    /// How long to ask `wait` for, to be awake `d` from now: all of it. A
    /// futex or condition-variable wait ends within tens of microseconds of
    /// its timeout on Linux and macOS; the loop's margin covers that.
    fn plan(&self, d: Duration) -> Duration {
        d
    }

    /// Wait up to `d`, or until woken.
    fn wait(&self, d: Duration) {
        thread::park_timeout(d);
    }

    fn wait_forever(&self) {
        thread::park();
    }
}

#[cfg(not(windows))]
impl Waker {
    fn wake(&self) {
        self.thread.unpark();
    }
}

#[cfg(windows)]
mod windows_waiter {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, WAIT_FAILED};
    use windows_sys::Win32::Media::{timeBeginPeriod, timeEndPeriod};
    use windows_sys::Win32::System::Threading::{
        CreateEventW, CreateWaitableTimerExW, SetEvent, SetWaitableTimer, WaitForMultipleObjects, WaitForSingleObject,
        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, INFINITE, TIMER_ALL_ACCESS,
    };

    use super::SPIN_MAX;

    /// The system timer resolution asked for while a waiter lives, in ms.
    /// It does nothing for the high-resolution timer, which keeps its own
    /// deadline, but every other timed wait in the process (`park_timeout`,
    /// `Condvar::wait_timeout`, the CPU's idle slices, the display clocks)
    /// then ends within about a millisecond instead of 15.6.
    const SYSTEM_TIMER_PERIOD_MS: u32 = 1;

    /// Calibration waits: a request this short sleeps one grid step.
    const PROBE: Duration = Duration::from_micros(100);
    const PROBES: usize = 7;
    /// A grid coarser than this is the system tick, with no high-resolution
    /// timer behind it: planning around it would spin for milliseconds, so
    /// the waiter asks for the whole wait and takes the lateness.
    const GRID_MAX: Duration = Duration::from_millis(2);
    /// Requested this far under a grid point, the timer fires at that point
    /// rather than the next (measured: 450 µs asked, 525 µs slept).
    const UNDER_GRID: Duration = Duration::from_micros(50);

    /// An auto-reset event: a `SetEvent` before the wait is kept until a wait
    /// consumes it, as `unpark`'s token is.
    struct Event(HANDLE);
    // A kernel handle may be signalled and waited on from any thread.
    unsafe impl Send for Event {}
    unsafe impl Sync for Event {}
    impl Drop for Event {
        fn drop(&mut self) {
            unsafe { CloseHandle(self.0) };
        }
    }

    pub(crate) struct Waiter {
        event: Arc<Event>,
        /// The high-resolution timer, or the ordinary one where the flag is
        /// refused (before Windows 10 1803); null if neither could be made,
        /// in which case a wait is a `thread::sleep` the waker cannot end.
        timer: HANDLE,
        /// The sleep grid `plan` works around, measured at start; None when
        /// the timer's sleeps are not gridded finely enough to plan for.
        grid: Option<Duration>,
    }

    #[derive(Clone)]
    pub(crate) struct Waker {
        event: Arc<Event>,
    }

    impl Waiter {
        pub(super) fn new() -> Self {
            // SAFETY: plain Win32 calls with null security attributes and names.
            let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
            assert!(!event.is_null(), "hptimer: CreateEventW failed");
            let mut timer = unsafe {
                CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)
            };
            if timer.is_null() {
                log::warn!("hptimer: no high-resolution waitable timer on this Windows; guest clocks follow the system tick");
                timer = unsafe { CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), 0, TIMER_ALL_ACCESS) };
            }
            unsafe { timeBeginPeriod(SYSTEM_TIMER_PERIOD_MS) };
            let mut w = Self { event: Arc::new(Event(event)), timer, grid: None };
            w.grid = w.measure_grid();
            match w.grid {
                Some(g) => log::info!("hptimer: Windows waitable timer sleeps on a {} µs grid", g.as_micros()),
                None => log::warn!("hptimer: Windows waitable timer has no fine sleep grid; deadlines run late by the system tick"),
            }
            w
        }

        /// The step the timer's sleeps round up to: the median of a few
        /// short waits, rounded to 50 µs.
        fn measure_grid(&self) -> Option<Duration> {
            if self.timer.is_null() {
                return None;
            }
            let mut slept = [Duration::ZERO; PROBES];
            for s in slept.iter_mut() {
                let t0 = Instant::now();
                self.wait(PROBE);
                *s = t0.elapsed();
            }
            slept.sort();
            let median = slept[PROBES / 2];
            let step = Duration::from_micros(50);
            let grid = step * ((median.as_micros() as u32 + 25) / 50);
            (grid >= step && grid <= GRID_MAX).then_some(grid)
        }

        pub(super) fn waker(&self) -> Waker {
            Waker { event: self.event.clone() }
        }

        /// How long to ask `wait` for, to be awake `d` from now. Zero means
        /// spin the whole of `d`. With a grid: the grid point at or under
        /// `d` when the rest fits `SPIN_MAX`, else `d` itself, which wakes at
        /// the next grid point, late by less than the grid minus `SPIN_MAX`.
        pub(super) fn plan(&self, d: Duration) -> Duration {
            let Some(grid) = self.grid else { return d };
            let steps = d.as_nanos() / grid.as_nanos();
            let under = grid * steps as u32;
            let rest = d - under;
            if rest > SPIN_MAX {
                d
            } else if steps == 0 {
                Duration::ZERO
            } else {
                under - UNDER_GRID
            }
        }

        /// Wait up to `d`, or until woken.
        pub(super) fn wait(&self, d: Duration) {
            if self.timer.is_null() {
                std::thread::sleep(d);
                return;
            }
            // A negative due time is relative, in 100 ns units. Setting the
            // timer also clears a signal left from a wait the event ended.
            let due = -i64::try_from(d.as_nanos() / 100).unwrap_or(i64::MAX).max(1);
            // SAFETY: both handles are open for the life of `self`; the
            // completion routine and its argument are null.
            let ok = unsafe { SetWaitableTimer(self.timer, &due, 0, None, std::ptr::null(), 0) };
            if ok == 0 {
                std::thread::sleep(d);
                return;
            }
            let handles = [self.event.0, self.timer];
            let r = unsafe { WaitForMultipleObjects(handles.len() as u32, handles.as_ptr(), 0, INFINITE) };
            if r == WAIT_FAILED {
                std::thread::sleep(d);
            }
        }

        pub(super) fn wait_forever(&self) {
            // SAFETY: the event handle is open for the life of `self`.
            unsafe { WaitForSingleObject(self.event.0, INFINITE) };
        }
    }

    impl Drop for Waiter {
        fn drop(&mut self) {
            unsafe { timeEndPeriod(SYSTEM_TIMER_PERIOD_MS) };
            if !self.timer.is_null() {
                unsafe { CloseHandle(self.timer) };
            }
        }
    }

    impl Waker {
        pub(super) fn wake(&self) {
            // SAFETY: the event handle is open while any `Arc<Event>` lives.
            unsafe { SetEvent(self.event.0) };
        }
    }
}

#[cfg(windows)]
pub(crate) use windows_waiter::{Waiter, Waker};

#[cfg(test)]
mod timer_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct Dev {
        timer: Timer,
        fired: Mutex<Vec<u64>>,
        rearm: AtomicUsize,
    }

    fn dev() -> (Arc<Dev>, crate::test_board::Stepped) {
        let clock = crate::test_board::Stepped::new();
        let d = Arc::new_cyclic(|me| Dev {
            timer: Timer::new(&clock.tm, me, |d: &Dev, due| {
                d.fired.lock().push(due);
                if d.rearm.load(Ordering::SeqCst) > 0 {
                    d.rearm.fetch_sub(1, Ordering::SeqCst);
                    d.timer.set_due(Some(due + 10));
                }
            }),
            fired: Mutex::new(Vec::new()),
            rearm: AtomicUsize::new(0),
        });
        (d, clock)
    }

    fn fired(d: &Dev) -> Vec<u64> {
        d.fired.lock().clone()
    }

    #[test]
    fn a_timer_fires_once_when_the_machine_clock_reaches_it() {
        let (d, clock) = dev();
        d.timer.start();
        d.timer.set_due(Some(1_000));
        assert!(fired(&d).is_empty(), "the host timer ran ahead of the stepped clock");
        clock.step(1_000);
        assert_eq!(fired(&d), [1_000]);
        assert!(!d.timer.is_armed());
    }

    /// A manual host timer is deleted without running, so the timer must stop pointing at it.
    #[test]
    fn a_manual_host_timer_is_forgotten_when_it_goes() {
        let (d, _clock) = dev();
        d.timer.start();
        d.timer.set_due(Some(1_000));
        assert!(crate::test_board::until(|| d.timer.0.state.lock().id.is_none()));
    }

    #[test]
    fn a_replaced_or_cancelled_deadline_does_not_fire() {
        let (d, clock) = dev();
        d.timer.start();
        d.timer.set_due(Some(1_000_000));
        d.timer.set_due(Some(50));
        clock.step(50);
        assert_eq!(fired(&d), [50]);
        d.timer.set_due(Some(60));
        d.timer.set_due(None);
        clock.step(1_000);
        assert_eq!(fired(&d), [50]);
    }

    #[test]
    fn stop_keeps_the_deadline_and_start_arms_it() {
        let (d, clock) = dev();
        d.timer.set_due(Some(10));
        clock.step(10);
        assert!(fired(&d).is_empty(), "a timer never started stays quiet");
        d.timer.start();
        clock.tm.run_due();
        assert_eq!(fired(&d), [10]);
        d.timer.set_due(Some(30));
        d.timer.stop();
        clock.step(100);
        assert_eq!(d.timer.due(), Some(30));
        assert_eq!(fired(&d), [10]);
        d.timer.start();
        clock.tm.run_due();
        assert_eq!(fired(&d), [10, 30]);
    }

    #[test]
    fn a_callback_can_arm_the_next_deadline() {
        let (d, clock) = dev();
        d.rearm.store(2, Ordering::SeqCst);
        d.timer.start();
        d.timer.set_due(Some(5));
        clock.step(100);
        assert_eq!(fired(&d), [5, 15, 25]);
        assert!(!d.timer.is_armed());
    }

    #[test]
    fn a_deadline_that_did_not_move_or_moved_later_keeps_its_host_timer() {
        // An hour out: the manual host timer's real delay is the stepped distance, so a near
        // deadline would let it run and drop its id mid-test.
        const T: u64 = 3_600_000_000_000;
        let (d, clock) = dev();
        d.timer.start();
        d.timer.set_due(Some(T + 1_000));
        let id = d.timer.0.state.lock().id;
        assert!(id.is_some());
        d.timer.set_due(Some(T + 1_000));
        assert_eq!(d.timer.0.state.lock().id, id);
        d.timer.set_due(Some(T + 5_000));
        assert_eq!(d.timer.0.state.lock().id, id);
        clock.step(T + 1_000);
        assert!(fired(&d).is_empty(), "fired at the deadline it moved away from");
        clock.step(4_000);
        assert_eq!(fired(&d), [T + 5_000]);
        d.timer.set_due(Some(T + 9_000));
        let id = d.timer.0.state.lock().id;
        d.timer.set_due(Some(T + 6_000));
        assert_ne!(d.timer.0.state.lock().id, id);
    }

    #[test]
    fn a_repeated_deadline_does_not_take_the_lock() {
        let (d, _clock) = dev();
        d.timer.set_due(Some(1_000));
        let held = d.timer.0.state.lock();
        let done = Arc::new(AtomicBool::new(false));
        let t = {
            let (d, done) = (d.clone(), done.clone());
            thread::spawn(move || {
                d.timer.set_due(Some(1_000));
                done.store(true, Ordering::SeqCst);
            })
        };
        assert!(crate::test_board::until(|| done.load(Ordering::SeqCst)), "blocked on the timer lock");
        drop(held);
        t.join().unwrap();
    }
}
