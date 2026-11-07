// Cases follow tests/qtest/sn1-bedrock-pi-test.c in jamesbraid/qemu-sgi-origami:
// power-on-values, local-reset, pending-reflection, real-time-comparators
// and profile-counter-wrap.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
fn setup() -> (Arc<BedrockPi>, Arc<AtomicU64>) {
    let now = Arc::new(AtomicU64::new(0));
    let clock = now.clone();
    let pi = BedrockPi::new(
        BedrockPiConfig::default(),
        Arc::new(move || clock.load(Ordering::SeqCst)),
    );
    pi.start();
    (pi, now)
}
fn r(pi: &BedrockPi, off: u64) -> u64 {
    pi.read64(off).data
}
/// The PROM reads PI slot presence and relies on the documented reset survivors.
#[test]
fn the_pi_reports_interface_state_and_retains_registers_across_reset() {
    {
        let (pi, _) = setup();
        assert_eq!(r(&pi, reg::INTERFACE_ID), 0);
        assert_eq!(r(&pi, reg::INTERFACE_SLOT_A), 1);
        assert_eq!(r(&pi, reg::INTERFACE_SLOT_B), 0);
        pi.write64(reg::NODE_REGION, 0);
        pi.write64(reg::PROTECTION_KEY, 0x53474972756c6573);
        assert_eq!(r(&pi, reg::NODE_REGION), 1);
        let absent = BedrockPi::new(
            BedrockPiConfig {
                interface: 1,
                request_region: 0,
            },
            Arc::new(|| 0),
        );
        assert_eq!(r(&absent, reg::INTERFACE_ID), 2);
        assert_eq!(r(&absent, reg::INTERFACE_SLOT_A), 0);
        assert_eq!(pi.read32(reg::INTERFACE_ID).status, BUS_ERR);
        assert_eq!(pi.write32(reg::COUNTER, 0), BUS_ERR);
    }
    {
        let (pi, _) = setup();
        pi.write64(reg::PENDING_MODIFY, 0x10c);
        assert_eq!(r(&pi, reg::PENDING_STATUS_A), 1 << 12);
        pi.write64(reg::PENDING_MODIFY_B, 12);
        assert_eq!(r(&pi, reg::PENDING_STATUS_A), 0);
        pi.set_lb_error(true);
        pi.write64(reg::PENDING_MODIFY, 64 + 55);
        assert_eq!(r(&pi, reg::PENDING_STATUS_B), 1 << 55);
        pi.write64(reg::PENDING_MODIFY, 0x100 | 64 | 63);
        assert_eq!(r(&pi, reg::PENDING_STATUS_B), 1 << 55);
        pi.write64(reg::REQUEST_A, 0);
        pi.write64(reg::OFFSET_00E8, 1);
        assert_eq!(r(&pi, reg::PENDING_STATUS_A), 1 << 5);
        pi.write64(reg::PENDING_A_CLEAR, 1);
        assert_eq!(r(&pi, reg::PENDING_STATUS_A), 0);
        pi.write64(reg::OFFSET_0010, 0);
        assert!(!pi.io_modify(0x108, 0));
        pi.write64(reg::OFFSET_0010, 1);
        assert!(pi.io_modify(0x108, 0));
        assert_eq!(r(&pi, reg::PENDING_STATUS_A), 1 << 8);
    }
    {
        let (pi, _) = setup();
        pi.write64(reg::CACHED_ALIAS, 9);
        pi.write64(reg::CPU_A_ENABLE, 1);
        pi.write64(reg::ERROR_ADDRESS_A, !0);
        pi.set_memory_uncorrectable();
        pi.write64(reg::ERROR_SUMMARY_ALIAS, 3);
        pi.write64(reg::RT_COMPARE_A, 0x24);
        pi.power_on();
        assert_eq!(r(&pi, reg::ERROR_SUMMARY), MEMORY_ERROR | 3);
        assert_eq!(r(&pi, reg::CACHED_ALIAS), 9);
        assert_eq!(r(&pi, reg::CPU_A_ENABLE), 1);
        assert_eq!(r(&pi, reg::ERROR_ADDRESS_A), 0x1fffffff8);
        assert_eq!(r(&pi, reg::RT_COMPARE_A), 0);
        assert_eq!(r(&pi, reg::COUNTER), 0);
        pi.write64(reg::ERROR_SUMMARY_ALIAS, 0);
        assert_eq!(r(&pi, reg::ERROR_SUMMARY), MEMORY_ERROR);
        pi.write64(reg::ERROR_SUMMARY, MEMORY_ERROR);
        assert_eq!(r(&pi, reg::ERROR_SUMMARY), 0);
    }
}
/// The guest timer drives interrupts even while the processor is stopped.
/// The processor timer can wrap, pause while stopped, and reissue its guest interrupt.
#[test]
fn the_pi_timer_wraps_pauses_and_reissues_interrupts() {
    {
        let (pi, now) = setup();
        pi.write64(reg::COUNTER, 0x1_0000_0000);
        pi.write64(reg::RT_COMPARE_A, 0x1_0000_0002);
        now.store(1599, Ordering::SeqCst);
        assert_eq!(r(&pi, reg::RT_STATUS_A), 0);
        now.store(1600, Ordering::SeqCst);
        assert_eq!(r(&pi, reg::RT_STATUS_A), 1);
        assert_eq!(pi.read32(reg::COUNTER).data, 1);
        assert_eq!(pi.read32(reg::COUNTER + 4).data, 2);
        pi.write64(reg::RT_STATUS_A, !0);
        assert_eq!(r(&pi, reg::RT_STATUS_A), 0);
        pi.write64(reg::COUNTER, RT_MASK);
        pi.write64(reg::PROFILE_COMPARE, 0);
        now.store(2400, Ordering::SeqCst);
        assert_eq!(r(&pi, reg::COUNTER), 0);
        assert_eq!(r(&pi, reg::PROFILE_STATUS_A), 1);
        assert_eq!(r(&pi, reg::PROFILE_STATUS_B), 1);
        pi.write64(reg::PROFILE_STATUS_A, 0);
        assert_eq!(r(&pi, reg::PROFILE_STATUS_A), 0);
        assert_eq!(r(&pi, reg::PROFILE_STATUS_B), 1);
    }
    {
        let (pi, now) = setup();
        pi.set_level_source(0, 12, true);
        pi.write64(reg::PENDING_MODIFY, 0x10c);
        pi.write64(reg::PENDING_MODIFY, 12);
        now.store(999999, Ordering::SeqCst);
        assert_eq!(r(&pi, reg::PENDING_STATUS_A), 0);
        now.store(1000000, Ordering::SeqCst);
        assert_eq!(r(&pi, reg::PENDING_STATUS_A), 1 << 12);
        pi.write64(reg::COUNTER, 0);
        pi.write64(reg::RT_COMPARE_A, 2);
        now.store(1000400, Ordering::SeqCst);
        pi.stop();
        now.store(2000000, Ordering::SeqCst);
        assert_eq!(r(&pi, reg::COUNTER), 0);
        pi.start();
        now.store(2001200, Ordering::SeqCst);
        assert_eq!(r(&pi, reg::RT_STATUS_A), 1);
    }
}
/// An interrupt callback may inspect PI while an output changes.
/// An interrupt callback may inspect PI while an output changes.
#[test]
fn pi_interrupt_outputs_follow_masks_and_allow_reentrant_reads() {
    struct Probe {
        pi: Weak<BedrockPi>,
        bits: AtomicU64,
    }
    impl InterruptController for Probe {
        fn set_line(&self, line: u32, level: bool) {
            if let Some(pi) = self.pi.upgrade() {
                let _ = pi.read64(reg::PENDING_STATUS_A);
            }
            if level {
                self.bits.fetch_or(1 << line, Ordering::SeqCst);
            } else {
                self.bits.fetch_and(!(1 << line), Ordering::SeqCst);
            }
        }
    }
    let (pi, now) = setup();
    let probe = Arc::new(Probe {
        pi: Arc::downgrade(&pi),
        bits: AtomicU64::new(0),
    });
    pi.set_interrupt_controller(probe.clone());
    pi.write64(reg::OFFSET_00A8, 1 << 12);
    pi.write64(reg::OFFSET_00B0, 1 << 55);
    pi.write64(reg::PENDING_MODIFY, 0x10c);
    pi.set_lb_error(true);
    pi.write64(reg::RT_COMPARE_A, 2);
    pi.write64(reg::PROFILE_COMPARE, 2);
    pi.write64(reg::RT_ENABLE_A, 1);
    pi.write64(reg::PROFILE_ENABLE_A, 1);
    pi.write64(reg::ERROR_CAUSE_A, MEMORY_ERROR);
    pi.set_memory_uncorrectable();
    now.store(1600, Ordering::SeqCst);
    pi.poll();
    assert_eq!(probe.bits.load(Ordering::SeqCst), 31);
    pi.write64(reg::RT_STATUS_A, 0);
    assert_eq!(probe.bits.load(Ordering::SeqCst), 27);
    pi.write64(reg::PENDING_MODIFY, 12);
    pi.set_lb_error(false);
    pi.write64(reg::PROFILE_STATUS_A, 0);
    pi.write64(reg::ERROR_SUMMARY, MEMORY_ERROR);
    assert_eq!(probe.bits.load(Ordering::SeqCst), 0);
}
/// A guest can wait for a timer interrupt without polling its compare register.
/// The guest may wait for a timer interrupt without polling the compare register.
#[test]
fn the_timer_manager_delivers_expirations_without_mmio_polling() {
    struct Probe(std::sync::mpsc::Sender<()>);
    impl InterruptController for Probe {
        fn set_line(&self, line: u32, level: bool) {
            if line == 2 && level {
                self.0.send(()).unwrap();
            }
        }
    }
    let pi = BedrockPi::new(BedrockPiConfig::default(), crate::dev::heart::host_clock());
    let tm = Arc::new(TimerManager::new());
    pi.set_timer_manager(tm.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    pi.set_interrupt_controller(Arc::new(Probe(tx)));
    pi.write64(reg::RT_COMPARE_A, 12500);
    pi.write64(reg::RT_ENABLE_A, 1);
    pi.start();
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    pi.stop();
    assert!(!pi.is_running());
}
/// Node callbacks may reenter PI, so concurrent writes must keep their observed order.
/// Node callbacks may reenter PI, so concurrent writes must preserve observed order.
#[test]
fn concurrent_pi_updates_publish_callbacks_in_register_order() {
    use std::sync::mpsc;
    use std::thread;

    struct Hooks {
        pi: Weak<BedrockPi>,
        offset: u64,
        first_value: u64,
        entered: mpsc::Sender<()>,
        release: Mutex<mpsc::Receiver<()>>,
        observed: AtomicU64,
    }
    impl Hooks {
        fn publish(&self, value: u64) {
            // A node may inspect PI registers while processing a change.
            let _ = self.pi.upgrade().unwrap().read64(self.offset);
            if value == self.first_value {
                self.entered.send(()).unwrap();
                self.release.lock().recv().unwrap();
            }
            self.observed.store(value, Ordering::SeqCst);
        }
    }
    impl NodeHooks for Hooks {
        fn cached_alias_changed(&self, size: u8) {
            self.publish(u64::from(size));
        }
        fn cpu_enable(&self, _: u8, _: usize, enabled: bool) {
            self.publish(u64::from(enabled));
        }
    }

    for (offset, first_value, second_value) in [(reg::CACHED_ALIAS, 1, 2), (reg::CPU_A_ENABLE, 1, 0), (reg::CPU_B_ENABLE, 1, 0)] {
        let (pi, _) = setup();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let hooks = Arc::new(Hooks {
            pi: Arc::downgrade(&pi),
            offset,
            first_value,
            entered: entered_tx,
            release: Mutex::new(release_rx),
            observed: AtomicU64::new(99),
        });
        pi.set_node_hooks(hooks.clone());
        let first_pi = pi.clone();
        let first = thread::spawn(move || first_pi.write64(offset, first_value));
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let second_pi = pi.clone();
        let second = thread::spawn(move || {
            started_tx.send(()).unwrap();
            second_pi.write64(offset, second_value);
            done_tx.send(()).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let premature = done_rx.recv_timeout(Duration::from_millis(100));
        // Always release and join both writers before checking results.
        release_tx.send(()).unwrap();
        assert_eq!(first.join().unwrap(), BUS_OK);
        second.join().unwrap();
        assert_eq!(
            hooks.observed.load(Ordering::SeqCst),
            r(&pi, offset),
            "offset {offset:#x}"
        );
        assert!(matches!(premature, Err(mpsc::RecvTimeoutError::Timeout)));
        assert_eq!(r(&pi, offset), second_value);
    }
}
