//! One wait abstraction for every slow operation.
//!
//! A [`Job`] runs its body on a worker thread under a panic guard, so it always
//! ends in [`Status::Done`], [`Status::Failed`] or [`Status::Cancelled`]. The body
//! reports through a [`JobCtx`] (phase, progress, cancel checks); a controller
//! reads [`Job::snapshot`] and the shell is woken through its [`Notify`], so no
//! shell needs a per-frame timer. Progress wakes are throttled to about 10 Hz;
//! phase changes and the end always wake.
//!
//! A [`JobRegistry`] tracks which [`Resource`]s running jobs touch, refuses a
//! second job on a busy resource and says which job holds it.
//!
//! The pattern is rusty-backup's `model::job`, cut down to what IRIS needs.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Something that can wake the shell's main thread. egui requests a repaint;
/// AppKit queues a block on the main queue (always queued, never run inline).
pub trait Notify: Send + Sync {
    fn wake(&self);
}

/// A [`Notify`] that does nothing, for tests and headless use.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoWake;

impl Notify for NoWake {
    fn wake(&self) {}
}

/// What a determinate progress count measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Bytes,
    Items,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    Indeterminate,
    Determinate { done: u64, total: u64, unit: Unit },
    /// Work done so far when the total is unknown.
    Counting { done: u64, unit: Unit },
}

impl Progress {
    /// Completed fraction in `0.0..=1.0`, or `None` when there is no total.
    pub fn fraction(&self) -> Option<f32> {
        match *self {
            Progress::Indeterminate | Progress::Counting { .. } => None,
            Progress::Determinate { total: 0, .. } => Some(0.0),
            Progress::Determinate { done, total, .. } => Some((done.min(total) as f64 / total as f64) as f32),
        }
    }
}

/// Where a job is; its result is taken with [`Job::take_result`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    Running,
    Done,
    Failed(String),
    Cancelled,
}

/// A cheap copy of a job's visible state, for drawing.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    pub label: String,
    pub phase: String,
    pub progress: Progress,
    pub can_cancel: bool,
    pub cancel_requested: bool,
    pub status: Status,
    /// Bumped on every visible change, so a shell can skip an unchanged redraw.
    pub generation: u64,
    pub elapsed: Duration,
}

impl Snapshot {
    pub fn is_running(&self) -> bool {
        self.status == Status::Running
    }
}

struct State<R> {
    phase: String,
    progress: Progress,
    status: Status,
    result: Option<R>,
    generation: u64,
    last_progress_wake: Option<Instant>,
}

struct Shared<R> {
    label: String,
    can_cancel: bool,
    cancel: AtomicBool,
    started: Instant,
    state: Mutex<State<R>>,
    notify: Arc<dyn Notify>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panicking job body is caught before it can poison anything we read, but
    // never let a poisoned lock take the UI down with it.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a job's body uses to report and to check for cancellation.
pub struct JobCtx<R> {
    shared: Arc<Shared<R>>,
}

impl<R> JobCtx<R> {
    /// Whether the user asked to stop. Check this between units of work.
    pub fn cancelled(&self) -> bool {
        self.shared.cancel.load(Ordering::Acquire)
    }

    /// Name the step the job is on; always wakes the shell.
    pub fn phase(&self, phase: impl Into<String>) {
        let mut s = lock(&self.shared.state);
        s.phase = phase.into();
        s.generation += 1;
        drop(s);
        self.shared.notify.wake();
    }

    /// Report progress; wakes the shell at most about ten times a second.
    pub fn progress(&self, progress: Progress) {
        let now = Instant::now();
        let mut s = lock(&self.shared.state);
        s.progress = progress;
        s.generation += 1;
        let due = s.last_progress_wake.map_or(true, |t| now.duration_since(t) >= Duration::from_millis(100));
        if due {
            s.last_progress_wake = Some(now);
        }
        drop(s);
        if due {
            self.shared.notify.wake();
        }
    }
}

/// A slow operation running on its own thread.
pub struct Job<R> {
    shared: Arc<Shared<R>>,
}

impl<R: Send + 'static> Job<R> {
    /// Start `body` on a new thread. A body returns `Err` to fail; a panic is
    /// caught and reported as a failure. A body that sees [`JobCtx::cancelled`]
    /// should return promptly; whatever it returns then, the job ends as
    /// [`Status::Cancelled`].
    pub fn spawn<F>(label: impl Into<String>, can_cancel: bool, notify: Arc<dyn Notify>, body: F) -> Self
    where
        F: FnOnce(&JobCtx<R>) -> Result<R, String> + Send + 'static,
    {
        let shared = Arc::new(Shared {
            label: label.into(),
            can_cancel,
            cancel: AtomicBool::new(false),
            started: Instant::now(),
            state: Mutex::new(State {
                phase: String::new(),
                progress: Progress::Indeterminate,
                status: Status::Running,
                result: None,
                generation: 0,
                last_progress_wake: None,
            }),
            notify,
        });
        let ctx = JobCtx { shared: shared.clone() };
        let thread_name = format!("iris-gui-job: {}", shared.label);
        let run = move || {
            let outcome = catch_unwind(AssertUnwindSafe(|| body(&ctx)));
            let mut s = lock(&ctx.shared.state);
            s.status = if ctx.cancelled() {
                Status::Cancelled
            } else {
                match outcome {
                    Ok(Ok(r)) => {
                        s.result = Some(r);
                        Status::Done
                    }
                    Ok(Err(e)) => Status::Failed(e),
                    Err(p) => Status::Failed(panic_message(&p)),
                }
            };
            s.generation += 1;
            drop(s);
            ctx.shared.notify.wake();
        };
        if let Err(e) = std::thread::Builder::new().name(thread_name).spawn(run) {
            let mut s = lock(&shared.state);
            s.status = Status::Failed(format!("could not start a worker thread: {e}"));
            s.generation += 1;
        }
        Self { shared }
    }

    /// A job that has already finished with `result`, for tests and for work
    /// that turned out to be instant.
    pub fn finished(label: impl Into<String>, result: Result<R, String>) -> Self {
        let (status, result) = match result {
            Ok(r) => (Status::Done, Some(r)),
            Err(e) => (Status::Failed(e), None),
        };
        Self {
            shared: Arc::new(Shared {
                label: label.into(),
                can_cancel: false,
                cancel: AtomicBool::new(false),
                started: Instant::now(),
                state: Mutex::new(State {
                    phase: String::new(),
                    progress: Progress::Indeterminate,
                    status,
                    result,
                    generation: 1,
                    last_progress_wake: None,
                }),
                notify: Arc::new(NoWake),
            }),
        }
    }
}

impl<R> Job<R> {
    pub fn snapshot(&self) -> Snapshot {
        let s = lock(&self.shared.state);
        Snapshot {
            label: self.shared.label.clone(),
            phase: s.phase.clone(),
            progress: s.progress,
            can_cancel: self.shared.can_cancel,
            cancel_requested: self.shared.cancel.load(Ordering::Acquire),
            status: s.status.clone(),
            generation: s.generation,
            elapsed: self.shared.started.elapsed(),
        }
    }

    pub fn is_running(&self) -> bool {
        lock(&self.shared.state).status == Status::Running
    }

    /// Ask the body to stop. Wakes the shell so it can show "cancelling".
    pub fn cancel(&self) {
        if self.shared.can_cancel {
            self.shared.cancel.store(true, Ordering::Release);
            self.shared.notify.wake();
        }
    }

    /// The body's result, once, after [`Status::Done`].
    pub fn take_result(&self) -> Option<R> {
        lock(&self.shared.state).result.take()
    }

    /// The job's outcome once it ended: its result, or why it failed or was
    /// cancelled. `None` while it runs.
    pub fn outcome(&self) -> Option<Result<R, String>> {
        let status = lock(&self.shared.state).status.clone();
        match status {
            Status::Running => None,
            Status::Done => Some(self.take_result().ok_or_else(|| "no result".to_string())),
            Status::Failed(e) => Some(Err(e)),
            Status::Cancelled => Some(Err("cancelled".to_string())),
        }
    }

    /// Block until the job ends. For tests and shutdown paths only; a
    /// controller never waits on a job.
    pub fn wait(&self, timeout: Duration) -> bool {
        let until = Instant::now() + timeout;
        while self.is_running() {
            if Instant::now() >= until {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        true
    }
}

/// The text of a caught panic.
pub fn panic_message(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    if let Some(s) = p.downcast_ref::<String>() {
        return s.clone();
    }
    "unknown error".to_string()
}

/// Something a job touches that another job must not touch at the same time.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Resource {
    /// A disk image, by its path.
    Image(String),
    /// A machine, by its name.
    Machine(String),
}

/// Which resources running jobs hold, so controls that would conflict can be
/// disabled with a reason.
#[derive(Debug, Default)]
pub struct JobRegistry {
    held: Vec<(Resource, String, bool)>,
}

impl JobRegistry {
    /// Claim `resources` for the job labelled `label`. `writes` says whether it
    /// changes them. Fails with the label of the job already holding one.
    pub fn claim(&mut self, label: &str, resources: &[Resource], writes: bool) -> Result<(), String> {
        for r in resources {
            if let Some((_, by, _)) = self.held.iter().find(|(h, _, _)| h == r) {
                return Err(by.clone());
            }
        }
        self.held.extend(resources.iter().map(|r| (r.clone(), label.to_string(), writes)));
        Ok(())
    }

    /// Release everything the job labelled `label` holds.
    pub fn release(&mut self, label: &str) {
        self.held.retain(|(_, by, _)| by != label);
    }

    /// The label of the job holding `r`, if any.
    pub fn holder(&self, r: &Resource) -> Option<&str> {
        self.held.iter().find(|(h, _, _)| h == r).map(|(_, by, _)| by.as_str())
    }

    /// The label of a job writing to `r`, if any.
    pub fn writer(&self, r: &Resource) -> Option<&str> {
        self.held.iter().find(|(h, _, w)| h == r && *w).map(|(_, by, _)| by.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[derive(Default)]
    struct Count(AtomicUsize);
    impl Notify for Count {
        fn wake(&self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn a_job_ends_done_with_its_result_and_wakes() {
        let wake = Arc::new(Count::default());
        let job = Job::spawn("add", false, wake.clone(), |ctx| {
            ctx.phase("adding");
            Ok(2 + 2)
        });
        assert!(job.wait(Duration::from_secs(5)));
        assert_eq!(job.snapshot().status, Status::Done);
        assert_eq!(job.outcome(), Some(Ok(4)));
        assert!(wake.0.load(Ordering::Relaxed) >= 2, "phase and end both wake");
    }

    #[test]
    fn a_panic_is_a_failure_not_a_crash() {
        let job: Job<()> = Job::spawn("boom", false, Arc::new(NoWake), |_| panic!("disk on fire"));
        assert!(job.wait(Duration::from_secs(5)));
        assert_eq!(job.snapshot().status, Status::Failed("disk on fire".into()));
    }

    #[test]
    fn cancel_ends_cancelled() {
        let job: Job<()> = Job::spawn("spin", true, Arc::new(NoWake), |ctx| {
            while !ctx.cancelled() {
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(())
        });
        job.cancel();
        assert!(job.wait(Duration::from_secs(5)));
        assert_eq!(job.outcome(), Some(Err("cancelled".into())));
    }

    #[test]
    fn registry_refuses_a_busy_resource_and_names_the_holder() {
        let mut reg = JobRegistry::default();
        let disk = Resource::Image("/d/irix.img".into());
        reg.claim("Import", &[disk.clone()], true).unwrap();
        assert_eq!(reg.claim("Export", &[disk.clone()], false), Err("Import".into()));
        assert_eq!(reg.writer(&disk), Some("Import"));
        reg.release("Import");
        assert_eq!(reg.holder(&disk), None);
    }

    #[test]
    fn progress_fraction() {
        assert_eq!(Progress::Determinate { done: 5, total: 10, unit: Unit::Items }.fraction(), Some(0.5));
        assert_eq!(Progress::Determinate { done: 0, total: 0, unit: Unit::Bytes }.fraction(), Some(0.0));
        assert_eq!(Progress::Counting { done: 3, unit: Unit::Items }.fraction(), None);
    }
}
