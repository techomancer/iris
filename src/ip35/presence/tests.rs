use super::*;
use crate::cpu::smp::HOST_PARK_BIT;
use crate::dev::bedrock::{reg, BedrockConfig};
use crate::traits::{BusDevice, Device, BUS_OK};
use std::sync::atomic::AtomicU64;

/// A node of `count` processors with its register window, each processor a
/// park slot and a retired-cycle counter the test drives.
struct Rig {
    hold: Arc<PresenceHold>,
    regs: Arc<dyn BusDevice>,
    words: Vec<Arc<AtomicU64>>,
    cycles: Vec<Arc<AtomicU64>>,
    slots: Vec<Arc<ParkSlot>>,
    now: Arc<AtomicU64>,
}

impl Rig {
    fn new(count: usize) -> Self {
        let now = Arc::new(AtomicU64::new(0));
        let clock_now = now.clone();
        let clock: Clock = Arc::new(move || clock_now.load(Ordering::SeqCst));
        let processors = Population::fitted_cpus(count).unwrap();
        let bedrock = Bedrock::new(BedrockConfig { name: "test", clock: clock.clone(), processors, banks_mib: &[64], events: None }).unwrap();
        bedrock.start();
        let hold = PresenceHold::new(processors, clock, &bedrock);
        let words: Vec<_> = (0..count).map(|_| Arc::new(AtomicU64::new(0))).collect();
        let cycles: Vec<_> = (0..count).map(|_| Arc::new(AtomicU64::new(0))).collect();
        let slots: Vec<_> = words.iter().map(|w| Arc::new(ParkSlot::new(Arc::as_ptr(w)))).collect();
        hold.set_processors(slots.iter().zip(&cycles).map(|(slot, c)| {
            let c = c.clone();
            Cpu { slot: slot.clone(), cycles: Box::new(move || c.load(Ordering::SeqCst)) }
        }).collect());
        Self { hold, regs: bedrock.registers(), words, cycles, slots, now }
    }
    /// II scratch 0, as a publisher's store leaves it.
    fn publish(&self, bits: u64) {
        assert_eq!(self.regs.write64(reg::IO_BASE + 0x150, bits), BUS_OK);
    }
    /// Take the hub lock through scratch 3's test-and-set twin, or release it.
    fn hub_lock(&self, take: bool) {
        if take {
            assert_eq!(self.regs.read64(reg::LB_BASE + 0x8058).data, 0, "the lock was free");
        } else {
            assert_eq!(self.regs.write64(reg::LB_BASE + 0x8018, 0), BUS_OK);
        }
    }
    /// A store of `value` at `offset` in the register window by processor
    /// `cpu`'s thread.
    fn store_as(&self, cpu: usize, offset: u64, value: u64) {
        let regs = self.regs.clone();
        let status = std::thread::spawn(move || {
            crate::cpu::smp::set_current_cpu(cpu as u8);
            regs.write64(offset, value)
        })
        .join()
        .unwrap();
        assert_eq!(status, BUS_OK);
    }
    /// The PROM's reset barrier on LB scratch 1: every processor clears it,
    /// CPU 0 releases, and then `joins` store again, their presence join.
    fn barrier(&self, joins: &[usize]) {
        for cpu in 0..self.slots.len() {
            self.store_as(cpu, reg::LB_BASE + 0x8008, 0);
        }
        self.store_as(0, reg::LB_BASE + 0x8008, 0x07f8_0000);
        for &cpu in joins {
            self.store_as(cpu, reg::LB_BASE + 0x8008, 0);
        }
    }
    /// Processor `cpu` stores `value` to the real-time compare register of
    /// `slot` (0 = A, 1 = B) on PI `interface`.
    fn compare(&self, cpu: usize, interface: u64, slot: u64, value: u64) {
        self.store_as(cpu, interface * reg::PI1_BASE + 0x108 + 8 * slot, value);
    }
    /// Processor `cpu` posts its liveness value in its own compare register.
    fn post(&self, cpu: usize) {
        let (interface, slot) = Population::fitted_cpus(self.slots.len()).unwrap().slot_of(cpu).unwrap();
        self.compare(cpu, u64::from(interface), slot as u64, 0x24);
    }
    fn parked(&self, cpu: usize) -> bool {
        self.words[cpu].load(Ordering::SeqCst) & HOST_PARK_BIT != 0 && self.slots[cpu].holds() & HOLD_RENDEZVOUS != 0
    }
}

/// PI0's publisher, having published with the lock free while PI1 has not,
/// is held at its batch boundary; its partner on PI0 and PI1's processors
/// are not. PI1's publication releases it at the next boundary any
/// processor reaches, and the join is over.
#[test]
fn a_publisher_waits_for_the_other_interface() {
    let rig = Rig::new(4);
    rig.hub_lock(true);
    rig.publish(0x30);
    rig.hold.at_batch(0);
    assert!(rig.hold.held().is_empty(), "not while the hub lock is held");
    rig.hub_lock(false);
    for cpu in [2, 1, 3] {
        rig.hold.at_batch(cpu);
    }
    assert!(rig.hold.held().is_empty(), "only a publisher whose interface has published");
    rig.hold.at_batch(0);
    assert_eq!(rig.hold.held(), [0]);
    assert!(rig.parked(0));
    rig.publish(0xf0);
    rig.hold.at_batch(1);
    assert!(rig.hold.held().is_empty() && !rig.parked(0));
    assert!(rig.hold.is_done());
}

/// The bound counts the waited-for processors' cycles: a PI1 that runs
/// HOLD_CYCLES without publishing ends the hold, and no later hold starts
/// until a reset, which starts the join over.
#[test]
fn a_hold_ends_once_the_other_interface_has_run_its_bound() {
    let rig = Rig::new(2);
    rig.publish(0x10);
    rig.hold.at_batch(0);
    assert_eq!(rig.hold.held(), [0]);
    rig.cycles[1].store(HOLD_CYCLES - 1, Ordering::SeqCst);
    rig.hold.at_batch(1);
    assert_eq!(rig.hold.held(), [0], "not yet");
    rig.cycles[0].store(u64::MAX, Ordering::SeqCst);
    rig.hold.at_batch(1);
    assert_eq!(rig.hold.held(), [0], "the held processor's own cycles do not count");
    rig.cycles[1].store(HOLD_CYCLES, Ordering::SeqCst);
    rig.hold.at_batch(1);
    assert!(rig.hold.held().is_empty() && !rig.parked(0));
    assert!(rig.hold.is_done(), "abandoned");
    rig.hold.reset();
    assert!(!rig.hold.is_done());
    rig.hold.at_batch(0);
    assert_eq!(rig.hold.held(), [0], "a reset starts the join over");
    rig.hold.reset();
    assert!(rig.hold.held().is_empty() && !rig.parked(0));
}

/// Host time backs the bound only as a safety net.
#[test]
fn the_safety_net_ends_a_hold_whatever_the_cycles() {
    let rig = Rig::new(2);
    rig.publish(0x10);
    rig.hold.at_batch(0);
    rig.now.store(SAFETY_NS - 1, Ordering::SeqCst);
    rig.hold.at_batch(1);
    assert_eq!(rig.hold.held(), [0]);
    rig.now.store(SAFETY_NS, Ordering::SeqCst);
    rig.hold.at_batch(1);
    assert!(rig.hold.held().is_empty());
}

/// One processor has nobody to wait for.
#[test]
fn one_processor_is_never_held() {
    let rig = Rig::new(1);
    assert!(rig.hold.is_done());
    rig.publish(0x10);
    rig.hold.at_batch(0);
    assert!(rig.hold.held().is_empty());
}

/// A boundary with nothing held and no hold to start takes no lock: CPU 2,
/// not a publisher, passes its boundary while the state is locked, and so
/// does a publisher whose partner interface has published.
#[test]
fn a_boundary_with_nothing_to_do_takes_no_lock() {
    let rig = Rig::new(4);
    rig.publish(0x30);
    let held = rig.hold.state.lock();
    rig.hold.at_batch(2);
    rig.hold.at_batch(1);
    drop(held);
    rig.publish(0xf0);
    let held = rig.hold.state.lock();
    rig.hold.at_batch(0);
    drop(held);
    assert!(rig.hold.is_done());
}

/// The first of a pair to post its liveness value after its join is held
/// until its partner posts: CPU 0 (PI0/A) waits for CPU 2 (PI0/B), and CPU 3
/// (PI1/B) for CPU 1 (PI1/A). Nobody else is held, and the partner's post
/// releases the hold at the next boundary any processor reaches.
#[test]
fn the_first_of_a_pair_to_post_waits_for_its_partner() {
    let rig = Rig::new(4);
    rig.barrier(&[0, 1, 2, 3]);
    rig.post(0);
    for cpu in [1, 2, 3, 0] {
        rig.hold.at_batch(cpu);
    }
    assert_eq!(rig.hold.held(), [0]);
    assert!(rig.parked(0));
    rig.post(3);
    rig.hold.at_batch(3);
    assert_eq!(rig.hold.held(), [0, 3]);
    rig.post(2);
    rig.hold.at_batch(1);
    assert_eq!(rig.hold.held(), [3], "CPU 2's post released CPU 0");
    assert!(!rig.parked(0));
    rig.post(1);
    rig.hold.at_batch(2);
    assert!(rig.hold.held().is_empty() && !rig.parked(3));
    for cpu in 0..4 {
        rig.hold.at_batch(cpu);
    }
    assert!(rig.hold.held().is_empty(), "both pairs posted");
}

/// Only a processor's first store to its own compare register after its
/// join is a post: earlier stores are progress codes, and a store to the
/// partner's register is not the processor's.
#[test]
fn only_a_store_to_its_own_register_after_its_join_posts() {
    let rig = Rig::new(4);
    rig.post(0);
    rig.hold.at_batch(0);
    assert!(rig.hold.held().is_empty(), "a progress code before the join");
    rig.barrier(&[0, 2]);
    rig.compare(0, 0, 1, 0x24);
    rig.hold.at_batch(0);
    assert!(rig.hold.held().is_empty(), "CPU 0 wrote CPU 2's register");
    rig.post(0);
    rig.hold.at_batch(0);
    assert_eq!(rig.hold.held(), [0]);
}

/// A partner that runs HOLD_CYCLES without posting ends the hold, and the
/// processor is not held for it again: the PROM's own check then finds the
/// partner dead. The presence join goes on.
#[test]
fn a_liveness_hold_runs_out_on_the_partners_cycles_and_ends_only_itself() {
    let rig = Rig::new(4);
    rig.barrier(&[0, 1, 2, 3]);
    rig.post(0);
    rig.hold.at_batch(0);
    assert_eq!(rig.hold.held(), [0]);
    rig.cycles[1].store(u64::MAX, Ordering::SeqCst);
    rig.hold.at_batch(1);
    assert_eq!(rig.hold.held(), [0], "only the partner's cycles count");
    rig.cycles[2].store(HOLD_CYCLES, Ordering::SeqCst);
    rig.hold.at_batch(2);
    assert!(rig.hold.held().is_empty() && !rig.parked(0));
    rig.hold.at_batch(0);
    assert!(rig.hold.held().is_empty(), "not held for that partner again");
    rig.publish(0x10);
    rig.hold.at_batch(0);
    assert_eq!(rig.hold.held(), [0], "the publisher's hold still applies");
}

/// Two processors sit one per interface: neither has a partner to wait for.
#[test]
fn two_processors_have_no_pair() {
    let rig = Rig::new(2);
    rig.barrier(&[0, 1]);
    rig.post(0);
    rig.post(1);
    rig.hold.at_batch(0);
    rig.hold.at_batch(1);
    assert!(rig.hold.held().is_empty());
}
