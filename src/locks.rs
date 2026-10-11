/// Global named lock registry.
///
/// Every parking_lot Mutex/RwLock in the system registers itself here via
/// `register_lock()`. The `locks` monitor command calls `is_locked()` on
/// each one without acquiring anything, giving a live snapshot of which
/// locks are held at the moment of the query.

use std::cell::UnsafeCell;
use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use parking_lot::Mutex as PLMutex;
use crate::traits::Device;

/// One registered lock entry.
pub struct LockEntry {
    pub name: String,
    /// Returns true if the lock is currently held by any thread.
    pub is_locked: Box<dyn Fn() -> bool + Send + Sync>,
    /// Who holds it, when the lock knows (`SpinLock` does).
    pub holder: Option<Box<dyn Fn() -> Option<String> + Send + Sync>>,
}

// ── SpinLock ──────────────────────────────────────────────────────────────────

/// Small per-thread number for lock holders (0 = nobody), with the thread's
/// name kept in `THREAD_NAMES` for the monitor.
fn thread_tag() -> u32 {
    static NEXT: AtomicU32 = AtomicU32::new(1);
    thread_local! {
        static TAG: u32 = {
            let tag = NEXT.fetch_add(1, Ordering::Relaxed);
            let name = std::thread::current().name().unwrap_or("?").to_string();
            thread_names().lock().push((tag, name));
            tag
        };
    }
    TAG.with(|t| *t)
}

fn thread_names() -> &'static PLMutex<Vec<(u32, String)>> {
    static NAMES: OnceLock<PLMutex<Vec<(u32, String)>>> = OnceLock::new();
    NAMES.get_or_init(|| PLMutex::new(Vec::new()))
}

fn thread_name(tag: u32) -> String {
    thread_names().lock().iter().find(|(t, _)| *t == tag).map_or_else(|| format!("thread {tag}"), |(_, n)| n.clone())
}

/// How long a `SpinLock::lock` waits before it reports a suspected deadlock
/// (once per wait, on stderr, naming both sides).
const STALL_REPORT: std::time::Duration = std::time::Duration::from_secs(5);

/// A light mutex: one atomic flag, spinning with backoff (yielding once the
/// backoff is spent) while contended. For the short critical sections the
/// device threads share, where parking_lot's parking costs more than the
/// wait. It remembers who holds it (thread and call site) for the `locks`
/// monitor command, and a waiter stuck for `STALL_REPORT` prints both sides
/// once: a deadlock names itself instead of just hanging.
///
/// Not reentrant: locking it again on the holding thread spins forever (and
/// is reported as a stall).
pub struct SpinLock<T: ?Sized> {
    locked: AtomicBool,
    /// Holder's thread tag and call site (diagnostics only; written after
    /// the flag is taken, so a reader may briefly see the previous holder).
    owner: AtomicU32,
    site: AtomicPtr<Location<'static>>,
    data: UnsafeCell<T>,
}

// SAFETY: access to `data` is serialized by `locked`.
unsafe impl<T: ?Sized + Send> Send for SpinLock<T> {}
unsafe impl<T: ?Sized + Send> Sync for SpinLock<T> {}

pub struct SpinGuard<'a, T: ?Sized> {
    lock: &'a SpinLock<T>,
}

impl<T> SpinLock<T> {
    pub const fn new(v: T) -> Self {
        SpinLock {
            locked: AtomicBool::new(false),
            owner: AtomicU32::new(0),
            site: AtomicPtr::new(std::ptr::null_mut()),
            data: UnsafeCell::new(v),
        }
    }
}

impl<T: ?Sized> SpinLock<T> {
    #[inline]
    #[track_caller]
    pub fn lock(&self) -> SpinGuard<'_, T> {
        if self.locked.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            self.lock_slow(Location::caller());
        }
        self.note_holder(Location::caller());
        SpinGuard { lock: self }
    }

    #[cold]
    fn lock_slow(&self, site: &'static Location<'static>) {
        let backoff = crossbeam_utils::Backoff::new();
        let mut since: Option<std::time::Instant> = None;
        let mut reported = false;
        loop {
            while self.locked.load(Ordering::Relaxed) {
                backoff.snooze();
                if backoff.is_completed() && !reported {
                    let t0 = *since.get_or_insert_with(std::time::Instant::now);
                    if t0.elapsed() >= STALL_REPORT {
                        reported = true;
                        eprintln!(
                            "locks: {} at {site} has waited {}s for a lock held by {}",
                            thread_name(thread_tag()),
                            t0.elapsed().as_secs(),
                            self.holder().unwrap_or_else(|| "nobody (just released)".into())
                        );
                    }
                }
            }
            if self.locked.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                if reported {
                    eprintln!("locks: {} at {site} got the lock after all", thread_name(thread_tag()));
                }
                return;
            }
        }
    }

    #[inline]
    fn note_holder(&self, site: &'static Location<'static>) {
        self.owner.store(thread_tag(), Ordering::Relaxed);
        self.site.store(site as *const _ as *mut _, Ordering::Relaxed);
    }

    #[inline]
    #[track_caller]
    pub fn try_lock(&self) -> Option<SpinGuard<'_, T>> {
        if self.locked.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
            self.note_holder(Location::caller());
            Some(SpinGuard { lock: self })
        } else {
            None
        }
    }

    #[inline]
    pub fn is_locked(&self) -> bool {
        self.locked.load(Ordering::Relaxed)
    }

    /// "thread at file:line" of the holder, or None when free.
    pub fn holder(&self) -> Option<String> {
        if !self.is_locked() {
            return None;
        }
        let site = self.site.load(Ordering::Relaxed);
        let owner = self.owner.load(Ordering::Relaxed);
        // SAFETY: `site` only ever holds null or a `&'static Location`.
        let site = unsafe { site.as_ref() }.map_or_else(|| "?".to_string(), |s| s.to_string());
        Some(format!("{} at {site}", thread_name(owner)))
    }

    /// The data, through exclusive access to the lock itself.
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    /// The data without taking the lock.
    ///
    /// # Safety
    /// The caller must guarantee nobody mutates it meanwhile (or tolerate
    /// tearing of plain data, as the display thread does).
    pub unsafe fn data_ptr(&self) -> *mut T {
        self.data.get()
    }
}

impl<T: Default> Default for SpinLock<T> {
    fn default() -> Self {
        SpinLock::new(T::default())
    }
}

impl<T: ?Sized> std::ops::Deref for SpinGuard<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: the guard holds the lock.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T: ?Sized> std::ops::DerefMut for SpinGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: the guard holds the lock.
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T: ?Sized> Drop for SpinGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.lock.owner.store(0, Ordering::Relaxed);
        self.lock.locked.store(false, Ordering::Release);
    }
}

/// Register any lock inside an `Arc`ed parent by a "held?" probe on it.
pub fn register_mutex_fn<P: Send + Sync + 'static>(name: impl Into<String>, parent: &Arc<P>, probe: fn(&P) -> bool) {
    let p = parent.clone();
    register_lock(name, move || probe(&p));
}

/// Register a `SpinLock` that lives inside an `Arc`ed parent: `get` maps the
/// parent to the lock.
pub fn register_spin<P: Send + Sync + 'static, T: ?Sized + 'static>(
    name: impl Into<String>,
    parent: &Arc<P>,
    get: fn(&P) -> &SpinLock<T>,
) {
    let (a, b) = (parent.clone(), parent.clone());
    registry().lock().push(LockEntry {
        name: name.into(),
        is_locked: Box::new(move || get(&a).is_locked()),
        holder: Some(Box::new(move || get(&b).holder())),
    });
}

/// Global registry. Populated at startup; read-only thereafter.
static REGISTRY: OnceLock<PLMutex<Vec<LockEntry>>> = OnceLock::new();

fn registry() -> &'static PLMutex<Vec<LockEntry>> {
    REGISTRY.get_or_init(|| PLMutex::new(Vec::new()))
}

/// Register a named lock. `is_locked` should call `.is_locked()` on the
/// underlying parking_lot Mutex/RwLock. Call this once per lock at init time.
pub fn register_lock(name: impl Into<String>, is_locked: impl Fn() -> bool + Send + Sync + 'static) {
    registry().lock().push(LockEntry {
        name: name.into(),
        is_locked: Box::new(is_locked),
        holder: None,
    });
}

// ── Convenience helpers ───────────────────────────────────────────────────────

/// Register a parking_lot Mutex by cloning its Arc.
pub fn register_mutex<T: Send + 'static>(name: impl Into<String>, m: &Arc<PLMutex<T>>) {
    let m = m.clone();
    register_lock(name, move || m.is_locked());
}

/// Register a parking_lot Mutex that is not Arc-wrapped but lives in an Arc
/// of its parent struct — use a closure that captures whatever you need.
pub fn register_lock_fn(name: impl Into<String>, f: impl Fn() -> bool + Send + Sync + 'static) {
    register_lock(name, f);
}

// ── Monitor device ─────────────────────────────────────────────────────────

pub struct LockMonitor;

impl Device for LockMonitor {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { false }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("locks".into(), "Show status of all registered system locks".into()),
        ]
    }

    fn execute_command(&self, cmd: &str, _args: &[&str], mut writer: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "locks" {
            return Err(format!("Unknown command: {}", cmd));
        }

        let reg = registry().lock();
        writeln!(writer, "=== System Lock Status ({} registered) ===", reg.len()).unwrap();
        let mut any_locked = false;
        for entry in reg.iter() {
            let locked = (entry.is_locked)();
            if locked { any_locked = true; }
            let who = entry.holder.as_ref().and_then(|h| h()).map(|h| format!(" by {h}")).unwrap_or_default();
            writeln!(writer, "  {:50} {}{who}", entry.name, if locked { "LOCKED" } else { "free" }).unwrap();
        }
        if !any_locked {
            writeln!(writer, "  (all locks free)").unwrap();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uncontended lock/unlock cost: SpinLock (with holder tracking), the
    /// bare flag it is built on, and parking_lot's Mutex.
    /// `cargo test --release --lib spinlock_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark"]
    fn spinlock_cost() {
        const N: u64 = 50_000_000;
        let spin = SpinLock::new(0u64);
        let pl = PLMutex::new(0u64);
        let flag = AtomicBool::new(false);
        let mut bare = 0u64;
        for round in 0..2 {
            let t = std::time::Instant::now();
            for _ in 0..N {
                *std::hint::black_box(&spin).lock() += 1;
            }
            let ts = t.elapsed().as_secs_f64() * 1e9 / N as f64;
            let t = std::time::Instant::now();
            for _ in 0..N {
                *std::hint::black_box(&pl).lock() += 1;
            }
            let tp = t.elapsed().as_secs_f64() * 1e9 / N as f64;
            let t = std::time::Instant::now();
            for _ in 0..N {
                let f = std::hint::black_box(&flag);
                while f.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {}
                bare += 1;
                f.store(false, Ordering::Release);
            }
            let tb = t.elapsed().as_secs_f64() * 1e9 / N as f64;
            println!("round {round}: SpinLock {ts:.2} ns, bare flag {tb:.2} ns, parking_lot {tp:.2} ns");
        }
        assert!(*spin.lock() == 2 * N && *pl.lock() == 2 * N && bare == 2 * N);
    }
}
