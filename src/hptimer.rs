use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use spin::Mutex;

/// Unique identifier for a timer.
pub type TimerId = u64;

/// Action returned by a timer callback upon expiration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimerReturn {
    /// Keep the timer as is. If it's recurring, it schedules the next expiration by adding its period.
    /// If it's a one-shot timer, it gets deleted.
    Continue,
    /// Keep the timer in the manager, but mark as disabled.
    Disable,
    /// Remove the timer from the manager completely.
    Delete,
    /// Reschedule the timer to trigger exactly once at the given relative Duration from now.
    RescheduleOneShot(Duration),
    /// Convert (or keep) the timer as recurring with a new period. The next expiration will be the old expiration + the new period.
    RescheduleRecurring(Duration),
}

/// A type-erased closure and context pair.
pub trait TimerCallback: Send {
    fn call(&mut self) -> TimerReturn;
}

struct CallbackWithContext<C, F> {
    context: C,
    cb: F,
}

impl<C, F> TimerCallback for CallbackWithContext<C, F>
where
    C: Send,
    F: FnMut(&mut C) -> TimerReturn + Send,
{
    fn call(&mut self) -> TimerReturn {
        (self.cb)(&mut self.context)
    }
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
    enabled: bool,
    in_fifo: bool,
    callback: Option<Box<dyn TimerCallback>>,
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
    worker_thread: Option<thread::Thread>,
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
}

impl TimerManager {
    /// Create a new `TimerManager` and start its worker thread.
    pub fn new() -> Self {
        let inner = Arc::new(Mutex::new(TimerManagerInner {
            slots: Vec::new(),
            free_indices: Vec::new(),
            fifo: VecDeque::new(),
            stop: false,
            worker_thread: None,
        }));

        let new_timer_added = Arc::new(AtomicBool::new(false));

        let thread_inner = inner.clone();
        let thread_added = new_timer_added.clone();

        let thread_handle = thread::Builder::new().name("hptimer".into()).spawn(move || {
            {
                let mut guard = thread_inner.lock();
                guard.worker_thread = Some(thread::current());
            }
            timer_thread_loop(thread_inner, thread_added);
        }).expect("spawn hptimer thread");

        Self {
            inner,
            thread_handle: Some(thread_handle),
            new_timer_added,
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

    /// Add a new one-shot timer to the manager counting from now, with its own context.
    pub fn add_one_shot<C, F>(&self, expire_in: Duration, context: C, cb: F) -> TimerId
    where
        C: Send + 'static,
        F: FnMut(&mut C) -> TimerReturn + Send + 'static,
    {
        let mut inner = self.inner.lock();
        let id = inner.allocate(TimerEntry {
            next_expiration: Instant::now() + expire_in,
            period: None,
            enabled: true,
            in_fifo: false,
            callback: Some(Box::new(CallbackWithContext { context, cb })),
        });
        drop(inner);
        self.wake_up();
        id
    }

    /// Add a recurring timer to the manager, with its own context.
    pub fn add_recurring<C, F>(&self, expire_at: Instant, period: Duration, context: C, cb: F) -> TimerId
    where
        C: Send + 'static,
        F: FnMut(&mut C) -> TimerReturn + Send + 'static,
    {
        let mut inner = self.inner.lock();
        let id = inner.allocate(TimerEntry {
            next_expiration: expire_at,
            period: Some(period),
            enabled: true,
            in_fifo: false,
            callback: Some(Box::new(CallbackWithContext { context, cb })),
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

    /// Turn on a timer that was previously disabled.
    pub fn enable(&self, id: TimerId) -> bool {
        let mut inner = self.inner.lock();
        let (generation, idx) = ((id >> 32) as u32, (id & 0xFFFFFFFF) as usize);
        if let Some(slot) = inner.slots.get_mut(idx) {
            if slot.generation == generation {
                if let Some(t) = &mut slot.timer {
                    t.enabled = true;
                    drop(inner);
                    self.wake_up();
                    return true;
                }
            }
        }
        false
    }

    /// Temporarily disable a timer.
    pub fn disable(&self, id: TimerId) -> bool {
        let mut inner = self.inner.lock();
        let (generation, idx) = ((id >> 32) as u32, (id & 0xFFFFFFFF) as usize);
        if let Some(slot) = inner.slots.get_mut(idx) {
            if slot.generation == generation {
                if let Some(t) = &mut slot.timer {
                    t.enabled = false;
                    return true;
                }
            }
        }
        false
    }

    fn wake_up(&self) {
        self.new_timer_added.store(true, Ordering::Release);
        let inner = self.inner.lock();
        if let Some(t) = &inner.worker_thread {
            t.unpark();
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

/// The core loop tracking timeouts precisely.
fn timer_thread_loop(inner: Arc<Mutex<TimerManagerInner>>, new_timer_added: Arc<AtomicBool>) {
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
                    if t.enabled && !t.in_fifo && t.next_expiration <= now {
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
                            if t.enabled {
                                let cb = t.callback.take();
                                to_call = Some((idx, generation, cb, t.period, t.next_expiration));
                            }
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
                        if t.enabled && !t.in_fifo {
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
                let mut result = cb.call();
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
                        let enabled = t.enabled;
                        match result {
                            TimerReturn::Continue => {
                                if let Some(period) = period {
                                    t.next_expiration += period;
                                    let now = Instant::now();
                                    if catch_up + 1 < MAX_CATCHUP
                                        && enabled
                                        && t.next_expiration <= now
                                    {
                                        catch_up += 1;
                                        cb = t.callback.take().unwrap();
                                        drop(guard);
                                        result = cb.call();
                                        continue;
                                    }
                                } else {
                                    guard.deallocate(((generation as u64) << 32) | (idx as u64));
                                }
                            }
                            TimerReturn::Disable => {
                                t.enabled = false;
                            }
                            TimerReturn::Delete => {
                                guard.deallocate(((generation as u64) << 32) | (idx as u64));
                            }
                            TimerReturn::RescheduleOneShot(new_delay) => {
                                t.next_expiration = Instant::now() + new_delay;
                                t.period = None;
                            }
                            TimerReturn::RescheduleRecurring(new_period) => {
                                t.period = Some(new_period);
                                t.next_expiration += new_period;
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

                if delay > SPIN_THRESHOLD {
                    // Sleep to just short of the deadline, then spin the
                    // rest. The margin covers the kernel's timer slack and
                    // wake-up latency (tens of microseconds), so the spin is
                    // bounded by it and a periodic timer at 1 kHz costs a
                    // few percent of a core rather than all of it.
                    thread::park_timeout(delay - SPIN_MARGIN);
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
            // Nothing to wake up for, indefinite sleep until unparked.
            if !new_timer_added.load(Ordering::Acquire) {
                thread::park();
            }
        }
    }
}
