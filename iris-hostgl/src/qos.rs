//! What the scheduler is told about a thread of ours.
//!
//! This Mac has eight performance cores and four efficiency ones, and macOS
//! decides which a thread may run on largely from its quality-of-service
//! class. A thread left at the default class is eligible for an efficiency
//! core; a `USER_INTERACTIVE` one is kept on the performance cores. Threads
//! spawned by Rust get whatever `pthread_create` gives them, which here is the
//! default class -- `ps -M` shows the emulator's main thread at priority 46
//! and *every* thread it spawns at 31.
//!
//! That matters because two of the threads on a GL frame's path are created
//! per guest process: the channel's service thread and the device's ring
//! threads. Where they land is decided when the process starts and does not
//! change for its life -- which is the shape of the fast/slow states we have
//! been chasing, a roughly twofold difference fixed for a process's lifetime,
//! invisible to every counter inside the guest and to every resource check
//! outside it.
//!
//! So the threads that carry a frame say what they are. They are latency
//! critical and tiny: a service thread is idle until a message arrives and
//! then must answer at once. Nothing else is raised -- this is not a licence
//! to mark everything important.
//!
//! `IRIS_HOSTGL_QOS=default` leaves them alone, so the same binary can be
//! measured both ways. That is the point: a difference between two builds
//! proves much less than a difference between two runs of one.

#[cfg(target_os = "macos")]
use std::sync::OnceLock;

/// `QOS_CLASS_USER_INTERACTIVE` from `<sys/qos.h>`.
#[cfg(target_os = "macos")]
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    fn pthread_get_qos_class_np(thread: libc::pthread_t, qos_class: *mut u32, relative_priority: *mut i32) -> i32;
}

/// The class this thread has now, for logging.
#[cfg(target_os = "macos")]
fn current() -> u32 {
    let mut class = 0u32;
    let mut relative = 0i32;
    // SAFETY: both out pointers are valid; pthread_self is always a live thread.
    unsafe { pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative) };
    class
}

#[cfg(target_os = "macos")]
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        // One switch for the whole emulator, so a controlled comparison is
        // one variable on one binary. `IRIS_HOSTGL_QOS=default` predates
        // `IRIS_QOS` and means the same thing, so runs already recorded
        // against that name stay comparable.
        if std::env::var("IRIS_HOSTGL_QOS").as_deref() == Ok("default") {
            return false;
        }
        !matches!(std::env::var("IRIS_QOS").as_deref(), Ok("default") | Ok("off"))
    })
}

/// Whether the report is wanted as well as the policy.
#[cfg(target_os = "macos")]
fn report() -> bool {
    std::env::var("IRIS_QOS").as_deref() != Ok("quiet")
}

/// Say that this thread is on a frame's path: it should be scheduled where a
/// frame can be answered promptly, not wherever there is room. Logs what the
/// thread was and what it became, because "what class did this thread have"
/// is otherwise not answerable from outside the process.
#[cfg(target_os = "macos")]
pub fn latency_critical(what: &str) {
    let before = current();
    if !enabled() {
        eprintln!("iris: qos {what}: left at {before:#x} (IRIS_QOS=default)");
        return;
    }
    // SAFETY: sets the calling thread's own class; no pointers involved.
    let r = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
    let after = current();
    if r != 0 {
        if report() { eprintln!("iris: qos {what}: could NOT become user-interactive ({r}); it stays at {before:#x}"); }
    } else {
        if report() { eprintln!("iris: qos {what}: {before:#x} -> {after:#x}"); }
    }
}

#[cfg(not(target_os = "macos"))]
pub fn latency_critical(_what: &str) {}
