use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::hptimer::{TimerManager, TimerReturn};

#[test]
#[ignore = "timing-sensitive: requires a quiet system, run with -- --ignored"]
fn test_precision_one_shot_and_recurring() {
    let manager = TimerManager::new();
    let start = Instant::now();

    let one_shot_fired = Arc::new(AtomicBool::new(false));
    let recurring_count = Arc::new(AtomicUsize::new(0));

    let fired_clone = one_shot_fired.clone();
    manager.add_one_shot(Duration::from_millis(50), (), move |_| {
        let diff = start.elapsed().as_secs_f64() - 0.050;
        assert!(
            diff.abs() < 0.015,
            "One shot diff {} out of bounds",
            diff
        );
        fired_clone.store(true, Ordering::SeqCst);
        TimerReturn::Continue // dies automatically since one-shot
    });

    let r_count = recurring_count.clone();
    manager.add_recurring(
        start + Duration::from_millis(20),
        Duration::from_millis(20),
        (),
        move |_| {
            let c = r_count.fetch_add(1, Ordering::SeqCst) + 1;
            let expected = 0.020 * (c as f64);
            let diff = start.elapsed().as_secs_f64() - expected;
            assert!(diff.abs() < 0.015, "Recurring off at {}: {}", c, diff);
            if c >= 3 {
                TimerReturn::Delete
            } else {
                TimerReturn::Continue
            }
        },
    );

    thread::sleep(Duration::from_millis(150));
    assert!(one_shot_fired.load(Ordering::SeqCst));
    assert_eq!(recurring_count.load(Ordering::SeqCst), 3);
}

#[test]
#[ignore = "timing-sensitive: requires a quiet system, run with -- --ignored"]
fn test_multiple_overlapping_timers() {
    let manager = TimerManager::new();
    let start = Instant::now();
    let fired_count = Arc::new(AtomicUsize::new(0));

    // Schedule 50 timers to fire at exactly 50ms from now
    for _ in 0..50 {
        let fc = fired_count.clone();
        manager.add_one_shot(Duration::from_millis(50), (), move |_| {
            let diff = start.elapsed().as_secs_f64() - 0.050;
            // Allow a bit more variance due to loop execution time and lock contention
            assert!(
                diff.abs() < 0.025,
                "Overlapping timer diff {} out of bounds",
                diff
            );
            fc.fetch_add(1, Ordering::SeqCst);
            TimerReturn::Continue
        });
    }

    thread::sleep(Duration::from_millis(100));
    assert_eq!(fired_count.load(Ordering::SeqCst), 50);
}

#[test]
#[ignore = "timing-sensitive: requires a quiet system, run with -- --ignored"]
fn test_extended_callback_recovery() {
    let manager = TimerManager::new();
    let start = Instant::now();
    let long_timer_done = Arc::new(AtomicBool::new(false));
    let short_timer_runs = Arc::new(AtomicUsize::new(0));

    // A timer that runs ONCE and blocks for 50ms, starving others.
    let ltd = long_timer_done.clone();
    manager.add_one_shot(Duration::from_millis(30), (), move |_| {
        thread::sleep(Duration::from_millis(50));
        ltd.store(true, Ordering::SeqCst);
        TimerReturn::Continue
    });

    // A fast recurring timer every 10ms
    let str = short_timer_runs.clone();
    manager.add_recurring(
        start + Duration::from_millis(10),
        Duration::from_millis(10),
        (),
        move |_| {
            str.fetch_add(1, Ordering::SeqCst);
            TimerReturn::Continue
        },
    );

    // Wait 150ms in total.
    // The short timer should trigger 14-15 times in 150ms.
    // However, it will be delayed during the 50ms block, but since we advance
    // by its period `period`, it will fire rapidly to catch up!
    thread::sleep(Duration::from_millis(150));
    
    assert!(long_timer_done.load(Ordering::SeqCst));
    let runs = short_timer_runs.load(Ordering::SeqCst);
    assert!(
        runs >= 13 && runs <= 16,
        "Expected around 14-15 runs, got {}",
        runs
    );
}

#[test]
#[ignore = "timing-sensitive: requires a quiet system, run with -- --ignored"]
fn test_rescheduling_timers() {
    let manager = TimerManager::new();
    let start = Instant::now();
    let states = Arc::new(std::sync::Mutex::new(Vec::new()));

    // 1. One-shot extending to another one-shot
    let st1 = states.clone();
    let mut step = 0;
    manager.add_one_shot(Duration::from_millis(20), (), move |_| {
        step += 1;
        let elapsed = start.elapsed().as_millis();
        st1.lock()
            .unwrap()
            .push(format!("oneshot_{} at ~{}", step, elapsed));
        if step == 1 {
            // Extend by another 30ms from now
            TimerReturn::RescheduleOneShot(Duration::from_millis(30))
        } else {
            TimerReturn::Delete
        }
    });

    // 2. One-shot upgrading to recurring
    let st2 = states.clone();
    let mut r_step = 0;
    manager.add_one_shot(Duration::from_millis(40), (), move |_| {
        r_step += 1;
        let elapsed = start.elapsed().as_millis();
        st2.lock()
            .unwrap()
            .push(format!("upgraded_{} at ~{}", r_step, elapsed));
        if r_step == 1 {
            // Upgrade to 20ms recurring
            TimerReturn::RescheduleRecurring(Duration::from_millis(20))
        } else if r_step == 3 {
            TimerReturn::Delete
        } else {
            TimerReturn::Continue
        }
    });

    thread::sleep(Duration::from_millis(150));

    let hist = states.lock().unwrap().clone();
    // Expected rough timings:
    // oneshot_1 at ~20,
    // oneshot_2 at ~50,
    // upgraded_1 at ~40,
    // upgraded_2 at ~60,
    // upgraded_3 at ~80

    assert_eq!(hist.len(), 5, "Expected 5 total timer executions");

    assert!(hist.iter().any(|s| s.starts_with("oneshot_1")));
    assert!(hist.iter().any(|s| s.starts_with("oneshot_2")));
    assert!(hist.iter().any(|s| s.starts_with("upgraded_1")));
    assert!(hist.iter().any(|s| s.starts_with("upgraded_2")));
    assert!(hist.iter().any(|s| s.starts_with("upgraded_3")));
}

#[test]
#[ignore = "timing-sensitive: requires a quiet system, run with -- --ignored"]
fn test_add_remove_multiple_order() {
    let manager = TimerManager::new();
    
    // We'll add 4 timers and remove 2 of them before they fire.
    // They are added out of order (by expiration time)
    // T1: 100ms
    // T2: 20ms
    // T3: 60ms
    // T4: 40ms
    
    let fired_t1 = Arc::new(AtomicBool::new(false));
    let fired_t2 = Arc::new(AtomicBool::new(false));
    let fired_t3 = Arc::new(AtomicBool::new(false));
    let fired_t4 = Arc::new(AtomicBool::new(false));

    let f1 = fired_t1.clone();
    let id1 = manager.add_one_shot(Duration::from_millis(100), (), move |_| {
        f1.store(true, Ordering::SeqCst);
        TimerReturn::Continue
    });

    let f2 = fired_t2.clone();
    let _id2 = manager.add_one_shot(Duration::from_millis(20), (), move |_| {
        f2.store(true, Ordering::SeqCst);
        TimerReturn::Continue
    });

    let f3 = fired_t3.clone();
    let id3 = manager.add_one_shot(Duration::from_millis(60), (), move |_| {
        f3.store(true, Ordering::SeqCst);
        TimerReturn::Continue
    });

    let f4 = fired_t4.clone();
    let _id4 = manager.add_one_shot(Duration::from_millis(40), (), move |_| {
        f4.store(true, Ordering::SeqCst);
        TimerReturn::Continue
    });

    // Let's remove T1 (100ms) and T3 (60ms) before they can fire. 
    // This leaves only T2 (20ms) and T4 (40ms) to fire.
    manager.remove(id3);
    manager.remove(id1);

    // Sleep for 120ms to give all timers a chance to fire
    thread::sleep(Duration::from_millis(120));

    assert_eq!(fired_t1.load(Ordering::SeqCst), false, "T1 was removed, should not fire");
    assert_eq!(fired_t2.load(Ordering::SeqCst), true, "T2 should fire");
    assert_eq!(fired_t3.load(Ordering::SeqCst), false, "T3 was removed, should not fire");
    assert_eq!(fired_t4.load(Ordering::SeqCst), true, "T4 should fire");
}

/// A 1 kHz recurring timer keeps its rate. Not ignored: it guards the
/// host's sleep granularity (Windows' 15.625 ms system tick made
/// `park_timeout` round every 1 ms wait up to it, and the guest's clock
/// ran at 64-80 Hz), so the bound is lenient enough for a loaded machine
/// and far above that failure.
#[test]
fn recurring_timer_at_1khz_keeps_its_rate() {
    let manager = TimerManager::new();
    let fired = Arc::new(AtomicUsize::new(0));
    let late_ns = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    let period = Duration::from_millis(1);
    let (f, l) = (fired.clone(), late_ns.clone());
    manager.add_recurring(start + period, period, 0u32, move |n| {
        *n += 1;
        f.fetch_add(1, Ordering::SeqCst);
        let due = start + period * *n;
        l.fetch_add(Instant::now().saturating_duration_since(due).as_nanos() as usize, Ordering::SeqCst);
        TimerReturn::Continue
    });
    thread::sleep(Duration::from_millis(500));
    let n = fired.load(Ordering::SeqCst);
    let mean_late_us = late_ns.load(Ordering::SeqCst) as f64 / n.max(1) as f64 / 1000.0;
    eprintln!("1 kHz recurring timer: {n} fires in 500 ms, mean lateness {mean_late_us:.0} us");
    assert!(n >= 300, "{n} fires in 500 ms: the host wait is rounding to its timer tick");
}
