// Cases follow hw/misc/sgi-bedrock.c's holds and tests/qtest/sn1-bedrock-lb-test.c
// scratch1-second-write-no-hold in jamesbraid/qemu-sgi-origami. Each hold
// case fails with that hold removed.

use super::*;
use crate::cpu::smp::{set_current_cpu, HOST_PARK_BIT};
use crate::traits::BusDevice;
use std::sync::atomic::{AtomicU64, Ordering};

const SCRATCH1: u64 = 0x8008;
const SCRATCH3: u64 = 0x8018;

struct Node {
    r: Arc<Rendezvous>,
    lb: Arc<BedrockLb>,
    words: Vec<Arc<AtomicU64>>,
    now: Arc<AtomicU64>,
    /// Cycles each processor has retired.
    cycles: Vec<Arc<AtomicU64>>,
}

/// A node running `count` processors, with nothing between it and the LB.
fn node(count: usize) -> Node {
    let now = Arc::new(AtomicU64::new(0));
    let clock: Clock = {
        let now = now.clone();
        Arc::new(move || now.load(Ordering::SeqCst))
    };
    let lb = Arc::new(BedrockLb::new(clock.clone()));
    let r = Rendezvous::new(Population::fitted_cpus(count).unwrap(), clock, &lb);
    let words: Vec<_> = (0..count).map(|_| Arc::new(AtomicU64::new(0))).collect();
    let cycles: Vec<_> = (0..count).map(|_| Arc::new(AtomicU64::new(0))).collect();
    r.set_processors(
        words
            .iter()
            .zip(&cycles)
            .map(|(w, c)| {
                let c = c.clone();
                Cpu {
                    slot: Arc::new(ParkSlot::new(Arc::as_ptr(w))),
                    progress: Box::new(move || c.load(Ordering::SeqCst)),
                    running: Box::new(|| true),
                }
            })
            .collect(),
    );
    lb.connect_scratch_hooks(r.clone());
    Node { r, lb, words, now, cycles }
}

impl Node {
    /// Run `f` as processor `cpu`.
    fn on<T: Send + 'static>(&self, cpu: u8, f: impl FnOnce(&Arc<Rendezvous>, &Arc<BedrockLb>) -> T + Send + 'static) -> T {
        let (r, lb) = (self.r.clone(), self.lb.clone());
        std::thread::spawn(move || {
            set_current_cpu(cpu);
            f(&r, &lb)
        })
        .join()
        .unwrap()
    }
    fn store(&self, cpu: u8, offset: u64, value: u64) {
        self.on(cpu, move |_, lb| lb.write64(offset, value));
    }
    fn held(&self, cpu: usize) -> bool {
        self.words[cpu].load(Ordering::SeqCst) & HOST_PARK_BIT != 0
    }
    /// Scratch 1 as processor 1 reads it, which lands nothing withheld.
    fn scratch1(&self) -> u64 {
        self.on(1, |_, lb| lb.read64(SCRATCH1).data)
    }
    /// Every processor through the reset barrier: each clears scratch 1,
    /// CPU 0 posts the release, and each makes the store after it that joins.
    fn join(&self, count: u8) {
        for cpu in 0..count {
            self.store(cpu, SCRATCH1, 0);
        }
        self.store(0, SCRATCH1, 0x7f8_0000);
        for cpu in 0..count {
            self.store(cpu, SCRATCH1, 0x100 + u64::from(cpu));
        }
        assert!(self.r.held().is_empty());
    }
}

/// A store from no processor, as a qtest's, is stored at once however many
/// times it comes. Follows scratch1-second-write-no-hold.
#[test]
fn a_store_from_no_processor_is_never_withheld() {
    let n = node(2);
    for value in [0x7f8_0000, 0, 0x7f8_0000] {
        n.lb.write64(SCRATCH1, value);
        assert_eq!(n.scratch1(), value);
    }
    assert!(n.r.held().is_empty());
}

/// One processor alone is never held.
#[test]
fn one_processor_is_never_held() {
    let n = node(1);
    n.store(0, SCRATCH1, 0);
    n.store(0, SCRATCH1, 0x7f8_0000);
    assert_eq!(n.scratch1(), 0x7f8_0000);
    assert!(n.r.held().is_empty() && !n.held(0));
}

/// The master's release posted before the other processor's clear is
/// withheld, and its writer held, until the clear lands; then the release
/// lands over it, so the clear cannot erase it.
#[test]
fn the_reset_barrier_lands_the_release_after_every_clear() {
    let n = node(2);
    n.store(0, SCRATCH1, 0);
    n.store(0, SCRATCH1, 0x7f8_0000);
    assert_eq!(n.r.held(), [(0, Kind::Scratch1)]);
    assert!(n.held(0));
    assert_eq!(n.scratch1(), 0, "withheld");
    // CPU 1 (slot C) makes its first store, its clear.
    n.store(1, SCRATCH1, 0);
    assert!(n.r.held().is_empty() && !n.held(0));
    assert_eq!(n.scratch1(), 0x7f8_0000, "the release survives the clear");
}

/// A read from no processor sees every withheld store first.
#[test]
fn an_outside_read_lands_the_withheld_stores() {
    let n = node(2);
    n.store(0, SCRATCH1, 0);
    n.store(0, SCRATCH1, 0x55);
    assert_eq!(n.scratch1(), 0, "withheld from the other processor");
    assert_eq!(n.lb.read64(SCRATCH1).data, 0x55);
    assert!(n.held(0), "the processor stays held");
}

/// The first of a pair to post its liveness status in its own compare
/// register is held until its partner posts.
#[test]
fn the_first_of_a_pair_to_post_waits_for_its_partner() {
    let n = node(4);
    n.join(4);
    // CPU 0 (A) posts; its partner is CPU 2 (B).
    n.on(0, |r, _| r.rt_compare_written(0, 0));
    assert_eq!(n.r.held(), [(0, Kind::Liveness)]);
    // Another processor's register is not its post.
    n.on(1, |r, _| r.rt_compare_written(0, 1));
    assert_eq!(n.r.held(), [(0, Kind::Liveness)]);
    n.on(2, |r, _| r.rt_compare_written(0, 1));
    assert!(n.r.held().is_empty() && !n.held(0));
}

/// A store to the compare register before the barrier is a progress code,
/// not a post, and holds nothing.
#[test]
fn a_compare_store_before_the_join_is_not_a_post() {
    let n = node(4);
    n.on(0, |r, _| r.rt_compare_written(0, 0));
    assert!(n.r.held().is_empty());
}

/// Once both of a pair have posted, the partner's counter read holds it
/// until its publisher has published and released the hub lock.
#[test]
fn the_partner_waits_at_the_counter_for_its_publisher() {
    let n = node(4);
    n.join(4);
    n.on(0, |r, _| r.rt_compare_written(0, 0));
    n.on(2, |r, _| r.rt_compare_written(0, 1));
    // The publisher's own counter read holds nothing.
    n.on(0, |r, _| r.rt_counter_read());
    assert!(n.r.held().is_empty());
    n.on(2, |r, _| r.rt_counter_read());
    assert_eq!(n.r.held(), [(1, Kind::Partner)]);
    n.on(0, |r, _| r.presence_published());
    assert_eq!(n.r.held(), [(1, Kind::Partner)], "published, lock still held");
    n.store(0, SCRATCH3, 0);
    // The publisher now waits for PI1's publication, and the partner goes.
    assert_eq!(n.r.held(), [(0, Kind::Presence)]);
    assert!(n.held(0) && !n.held(2));
}

/// A publisher releasing the hub lock after its own publication is held
/// until the other populated interface has published.
#[test]
fn a_publisher_waits_for_the_other_interface() {
    let n = node(2);
    n.join(2);
    n.on(0, |r, _| r.presence_published());
    n.store(0, SCRATCH3, 0);
    assert_eq!(n.r.held(), [(0, Kind::Presence)]);
    n.on(1, |r, _| r.presence_published());
    assert!(n.r.held().is_empty() && !n.held(0));
    // CPU 1 releases with nothing left to wait for.
    n.store(1, SCRATCH3, 0);
    assert!(n.r.held().is_empty());
}

/// The joins are over once every populated interface has released its hub
/// lock after publishing: from then on a scratch 1 store is plain storage and
/// nothing is held, until a reset starts them again.
#[test]
fn the_joins_end_once_every_interface_has_released_its_lock() {
    let n = node(2);
    n.join(2);
    n.on(0, |r, _| r.presence_published());
    n.store(0, SCRATCH3, 0);
    n.on(1, |r, _| r.presence_published());
    assert!(!n.r.is_over(), "PI1 still holds its lock");
    n.store(1, SCRATCH3, 0);
    assert!(n.r.is_over());
    n.store(1, SCRATCH1, 0x55);
    assert_eq!(n.scratch1(), 0x55, "a store after the joins lands at once");
    n.on(1, |r, _| r.rt_counter_read());
    assert!(n.r.held().is_empty());
    n.r.reset();
    assert!(!n.r.is_over(), "a reset starts the joins again");
}

/// A hold ends once the processors it waits for have run their bound
/// without coming; a publisher's that runs out ends the join, so later
/// releases hold nothing.
#[test]
fn every_hold_ends_at_its_bound() {
    let n = node(2);
    n.join(2);
    n.on(0, |r, _| r.presence_published());
    n.store(0, SCRATCH3, 0);
    // CPU 0 is a publisher; CPU 1 sits on the other interface.
    n.cycles[1].store(HOLD_CYCLES - 1, Ordering::SeqCst);
    n.r.expire();
    assert!(n.held(0));
    n.cycles[1].store(HOLD_CYCLES, Ordering::SeqCst);
    n.r.expire();
    assert!(n.r.held().is_empty() && !n.held(0));
    n.store(0, SCRATCH3, 0);
    assert!(n.r.held().is_empty(), "the join is over");
}

/// A processor the host does not run retires nothing, so no amount of host
/// time up to the safety net ends a hold waiting for it. Fails with a bound
/// in host time, which let a starved thread drop out of the PROM's count.
#[test]
fn a_starved_processor_cannot_run_a_hold_out() {
    let n = node(2);
    n.join(2);
    n.on(0, |r, _| r.presence_published());
    n.store(0, SCRATCH3, 0);
    n.now.store(10 * 2_000_000_000, Ordering::SeqCst);
    n.r.expire();
    assert!(n.held(0), "ten seconds of host time with no progress is no timeout");
    n.now.store(SAFETY_NS, Ordering::SeqCst);
    n.r.expire();
    assert!(!n.held(0), "the safety net still ends a hold");
}

/// Cycles run by processors the hold does not wait for do not count.
#[test]
fn only_the_processors_waited_for_count() {
    let n = node(4);
    n.join(4);
    n.on(0, |r, _| r.rt_compare_written(0, 0));
    assert_eq!(n.r.held(), [(0, Kind::Liveness)]);
    // Slot A waits for its partner B (CPU 2) alone.
    n.cycles[1].store(HOLD_CYCLES * 2, Ordering::SeqCst);
    n.cycles[3].store(HOLD_CYCLES * 2, Ordering::SeqCst);
    n.r.expire();
    assert!(n.held(0));
    n.cycles[2].store(HOLD_CYCLES, Ordering::SeqCst);
    n.r.expire();
    assert!(!n.held(0));
}

/// A reset lets every held processor go and forgets the barrier.
#[test]
fn a_reset_releases_every_hold() {
    let n = node(2);
    n.store(0, SCRATCH1, 0);
    n.store(0, SCRATCH1, 1);
    assert!(n.held(0));
    n.r.reset();
    assert!(!n.held(0) && n.r.held().is_empty());
    n.store(0, SCRATCH1, 0);
    assert_eq!(n.scratch1(), 0, "a first store again");
}
