// Cases follow tests/qtest/sn1-bedrock-pi-test.c in jamesbraid/qemu-sgi-origami:
// power-on-values, local-reset, pending-reflection, real-time-comparators,
// profile-counter-wrap, processor-presence, sparse-processor-presence,
// invalid-processor-masks, cpu-number-fallback, sparse-irq-and-nmi-slots and nmi.

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
                population: Population::SINGLE,
                running: Population::SINGLE,
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

/// One PI of a node holding the processors `count` names, in the board's order.
fn node(count: usize) -> [Arc<BedrockPi>; 2] {
    let population = Population::fitted_cpus(count).unwrap();
    std::array::from_fn(|interface| {
        BedrockPi::new(
            BedrockPiConfig { interface: interface as u8, request_region: 0, population, running: Population::SINGLE },
            Arc::new(|| 0),
        )
    })
}

/// The PROM counts processors from the present bits. The board fills PI0/A,
/// then PI1/A, then PI0/B and PI1/B, so two processors sit one per
/// interface (the qtest's sparse-processor-presence mask 5), where the
/// qtest's processor-count shortcut fills interface 0 first.
#[test]
fn every_fitted_slot_reads_present_in_the_boards_order() {
    for (count, present) in [(1, [1, 0, 0, 0]), (2, [1, 0, 1, 0]), (3, [1, 1, 1, 0]), (4, [1, 1, 1, 1])] {
        let pis = node(count);
        let read = |interface: usize, off| r(&pis[interface], off);
        assert_eq!(
            [read(0, reg::INTERFACE_SLOT_A), read(0, reg::INTERFACE_SLOT_B),
             read(1, reg::INTERFACE_SLOT_A), read(1, reg::INTERFACE_SLOT_B)],
            present, "{count} processors");
    }
    assert_eq!(Population::fitted_cpus(2).unwrap().mask(), 5);
    assert_eq!(Population::fitted_cpus(4).unwrap().count(), 4);
}

/// A population that cannot hold CPU 0, or names a fifth slot, or a count the
/// node cannot fit, is refused rather than built.
#[test]
fn a_population_without_cpu_0_or_beyond_four_slots_is_refused() {
    for mask in [0, 0b0010, 0b1100, 0b1_0001, 0xff] {
        assert!(Population::new(mask).is_err(), "{mask:#x}");
    }
    for count in [0, 5] {
        assert!(Population::fitted_cpus(count).is_err(), "{count}");
    }
    assert_eq!(Population::new(0b0101).unwrap(), Population::fitted_cpus(2).unwrap());
}

/// Only CPU 0 begins with its bus grant; the other fitted processors are held
/// disabled, and a power-on of the PI leaves that as it was.
#[test]
fn only_cpu_0_starts_enabled() {
    let pis = node(4);
    let enables = |pis: &[Arc<BedrockPi>; 2]| {
        [r(&pis[0], reg::CPU_A_ENABLE), r(&pis[0], reg::CPU_B_ENABLE),
         r(&pis[1], reg::CPU_A_ENABLE), r(&pis[1], reg::CPU_B_ENABLE)]
    };
    assert_eq!(enables(&pis), [1, 0, 0, 0]);
    pis[1].write64(reg::CPU_B_ENABLE, 1);
    for pi in &pis {
        pi.power_on();
    }
    assert_eq!(enables(&pis), [1, 0, 0, 1], "enables survive the PI's local reset");
    // An empty slot is not granted either.
    assert_eq!(r(&node(1)[0], reg::CPU_B_ENABLE), 0);
}

/// The CPU-number register carries the interface in bit 1; the reader is
/// CPU 0, which sits in slot A, so bit 0 is clear. Follows cpu-number-fallback.
#[test]
fn the_cpu_number_names_the_interface_and_slot_a() {
    let pis = node(4);
    assert_eq!((r(&pis[0], reg::INTERFACE_ID), r(&pis[1], reg::INTERFACE_ID)), (0, 2));
    for pi in &pis {
        pi.power_on();
    }
    assert_eq!((r(&pis[0], reg::INTERFACE_ID), r(&pis[1], reg::INTERFACE_ID)), (0, 2));
}

/// Enabling a processor that never runs is stored, as the register holds it,
/// and recorded so that the guest's attempt is visible; CPU 0's own enable
/// is not a refusal.
#[test]
fn enabling_another_cpu_is_stored_and_recorded_but_never_started() {
    let pis = node(4);
    pis[0].write64(reg::CPU_A_ENABLE, 1);
    assert!(pis[0].refused().is_empty(), "CPU 0 runs");
    pis[0].write64(reg::CPU_B_ENABLE, 1);
    pis[1].write64(reg::CPU_A_ENABLE, 3);
    pis[1].write64(reg::CPU_A_ENABLE, 1);
    pis[1].write64(reg::CPU_B_ENABLE, 0);
    assert_eq!(r(&pis[0], reg::CPU_B_ENABLE), 1);
    assert_eq!(pis[0].refused(), [(Refused::Start { interface: 0, slot: 1 }, 1)]);
    assert_eq!(pis[1].refused(), [(Refused::Start { interface: 1, slot: 0 }, 2)],
        "counted, and a clear bit is not an attempt");
}

/// An NMI for a slot with no running processor, or for CPU 0 with no node to
/// take it, is recorded. A node that takes CPU 0's still sees it. Follows nmi.
#[test]
fn an_nmi_no_running_cpu_can_take_is_recorded() {
    struct Nmis(std::sync::Mutex<Vec<(u8, usize)>>);
    impl NodeHooks for Nmis {
        fn nmi(&self, interface: u8, slot: usize) -> bool {
            self.0.lock().unwrap().push((interface, slot));
            true
        }
    }
    let pis = node(4);
    pis[0].write64(reg::NMI_SLOT_A, 1);
    pis[0].write64(reg::NMI_SLOT_B, u64::MAX);
    pis[1].write64(reg::NMI_SLOT_A, 1);
    assert_eq!(pis[0].refused(), [(Refused::Nmi { interface: 0, slot: 0 }, 1), (Refused::Nmi { interface: 0, slot: 1 }, 1)]);
    assert_eq!(pis[1].refused(), [(Refused::Nmi { interface: 1, slot: 0 }, 1)]);

    let nmis = Arc::new(Nmis(Default::default()));
    let pis = node(4);
    pis[0].set_node_hooks(nmis.clone());
    pis[0].write64(reg::NMI_SLOT_A, 1);
    assert!(pis[0].refused().is_empty());
    pis[0].write64(reg::NMI_SLOT_B, 1);
    assert_eq!(*nmis.0.lock().unwrap(), [(0, 0), (0, 1)], "a node still sees what it is sent");
    assert_eq!(pis[0].refused(), [(Refused::Nmi { interface: 0, slot: 1 }, 1)]);
}

/// An interrupt output that a processor in another slot would take, such as an
/// interprocessor interrupt aimed at slot A of PI1, goes high on its line and
/// is recorded; CPU 0's own lines are delivered without a record.
/// Follows sparse-irq-and-nmi-slots.
#[test]
fn interrupts_for_slots_without_a_running_cpu_are_recorded() {
    let pis = node(2);
    pis[1].write64(reg::PENDING_MODIFY, 0x107);
    pis[1].write64(reg::OFFSET_00A8, 1 << 7);
    assert_eq!(pis[1].refused(), [(Refused::Interrupt { interface: 1, line: 0 }, 1)]);
    pis[1].write64(reg::OFFSET_00A8, 0);
    assert_eq!(pis[1].refused().len(), 1, "the line going low is not a new event");

    pis[0].write64(reg::PENDING_MODIFY, 0x107);
    pis[0].write64(reg::OFFSET_00A8, 1 << 7);
    assert!(pis[0].refused().is_empty(), "PI0 slot A is CPU 0's");
    // Slot B of PI0 shares the pending word through its own mask.
    pis[0].write64(reg::OFFSET_00B8, 1 << 7);
    assert_eq!(pis[0].refused(), [(Refused::Interrupt { interface: 0, line: 5 }, 1)]);
}

/// Per-processor timers of a slot that never runs can be set but not
/// taken: enabling one is recorded.
#[test]
fn timers_for_other_slots_are_recorded_when_enabled() {
    let pis = node(4);
    pis[0].write64(reg::RT_ENABLE_A, 1);
    pis[0].write64(reg::PROFILE_ENABLE_A, 1);
    assert!(pis[0].refused().is_empty());
    pis[0].write64(reg::RT_ENABLE_B, 1);
    pis[0].write64(reg::PROFILE_ENABLE_B, 1);
    pis[1].write64(reg::RT_ENABLE_A, 1);
    assert_eq!(pis[0].refused(), [(Refused::Timer { interface: 0, slot: 1 }, 2)]);
    assert_eq!(pis[1].refused(), [(Refused::Timer { interface: 1, slot: 0 }, 1)]);
}

/// A node whose `count` fitted processors all run.
fn running_node(count: usize) -> [Arc<BedrockPi>; 2] {
    let population = Population::fitted_cpus(count).unwrap();
    std::array::from_fn(|interface| {
        BedrockPi::new(
            BedrockPiConfig { interface: interface as u8, request_region: 0, population, running: population },
            Arc::new(|| 0),
        )
    })
}

/// The board puts CPU n in the nth slot of its order: A, C, B, D.
#[test]
fn each_cpu_has_the_slot_the_boards_order_gives_it() {
    let four = Population::fitted_cpus(4).unwrap();
    assert_eq!((0..4).map(|cpu| four.slot_of(cpu).unwrap()).collect::<Vec<_>>(), [(0, 0), (1, 0), (0, 1), (1, 1)]);
    assert_eq!(four.slot_of(4), None);
    assert_eq!(four.cpu_in(0, 1), Some(2));
    let two = Population::fitted_cpus(2).unwrap();
    assert_eq!((two.slot_of(1), two.cpu_in(0, 1)), (Some((1, 0)), None));
}

/// Each processor reads its own slice from the CPU-number register of its
/// interface, and a reader that is no processor reads slot A. Follows
/// four-processors-know-their-slots in tests/qtest/sn1-cpu-identity-test.c
/// and cpu-number-fallback.
#[test]
fn four_processors_know_their_slots() {
    let pis = Arc::new(running_node(4));
    assert_eq!((r(&pis[0], reg::INTERFACE_ID), r(&pis[1], reg::INTERFACE_ID)), (0, 2), "no issuing processor");
    let slices: Vec<u64> = (0..4u8)
        .map(|cpu| {
            let pis = pis.clone();
            std::thread::spawn(move || {
                crate::cpu::smp::set_current_cpu(cpu);
                let (interface, _) = Population::fitted_cpus(4).unwrap().slot_of(cpu as usize).unwrap();
                r(&pis[interface as usize], reg::INTERFACE_ID)
            })
            .join()
            .unwrap()
        })
        .collect();
    assert_eq!(slices, [0, 2, 1, 3], "slices A, C, B, D");
}

/// Every running slot starts granted. A write is stored either way and
/// starts or stops nothing; clearing a running processor's bit is recorded.
/// Follows cpu-enable-lines and the identity test's enable writes.
#[test]
fn running_slots_start_enabled_and_a_clear_stops_nothing() {
    let pis = running_node(4);
    let enables = |pis: &[Arc<BedrockPi>; 2]| {
        [r(&pis[0], reg::CPU_A_ENABLE), r(&pis[0], reg::CPU_B_ENABLE),
         r(&pis[1], reg::CPU_A_ENABLE), r(&pis[1], reg::CPU_B_ENABLE)]
    };
    assert_eq!(enables(&pis), [1, 1, 1, 1]);
    pis[1].write64(reg::CPU_A_ENABLE, 0);
    assert_eq!(enables(&pis), [1, 1, 0, 1]);
    assert_eq!(pis[1].refused(), [(Refused::Stop { interface: 1, slot: 0 }, 1)]);
    pis[1].write64(reg::CPU_A_ENABLE, 1);
    assert_eq!(pis[1].refused().len(), 1, "re-enabling a running slot is no refusal");
    assert_eq!(enables(&running_node(2)), [1, 0, 1, 0]);
}

/// The other processors' interrupt outputs and timers are theirs to take:
/// nothing about them is refused once their slots run.
#[test]
fn a_running_slots_interrupts_and_timers_are_not_refused() {
    let pis = running_node(4);
    pis[1].write64(reg::PENDING_MODIFY, 0x107);
    pis[1].write64(reg::OFFSET_00A8, 1 << 7);
    pis[0].write64(reg::OFFSET_00B8, 1 << 7);
    pis[0].write64(reg::PENDING_MODIFY, 0x107);
    pis[0].write64(reg::RT_ENABLE_B, 1);
    pis[1].write64(reg::PROFILE_ENABLE_A, 1);
    assert!(pis[0].refused().is_empty() && pis[1].refused().is_empty());
}

/// An NMI to a running processor that the node cannot take is recorded:
/// the processor has no NMI entry. Follows nmi.
#[test]
fn an_nmi_to_a_running_processor_is_recorded() {
    let pis = running_node(2);
    pis[1].write64(reg::NMI_SLOT_A, 1);
    assert_eq!(pis[1].refused(), [(Refused::Nmi { interface: 1, slot: 0 }, 1)]);
}
