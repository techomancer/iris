//! The ticket word that makes compare-timer (IP7) delivery exactly-once per
//! armed Compare deadline.
//!
//! Every Compare write arms a fresh ticket. Whoever delivers the deadline
//! (the CPU thread's own Count check or the hptimer callback) claims the
//! ticket first, then asserts IP7. A Compare write is also the guest's
//! acknowledgement of IP7, so it must not be possible for a delivery that
//! claimed the *old* ticket to assert IP7 after the acknowledgement has
//! cleared it. Claiming and asserting are two steps on two words, so the word
//! has a third state, `CLAIMING`, held from the claim until the assertion is
//! done, and `arm` waits it out. The wait is only as long as the claimer's
//! `deliver` closure, which is a few atomic operations and an unpark.
//!
//! The per-instruction Count path never touches this word: it runs only when
//! Count crosses Compare or Compare is written.
//!
//! `loom_tests` checks every interleaving of one delivery against one
//! acknowledgement. Run them with
//! `RUSTFLAGS="--cfg loom" cargo test --release --lib ip7_ticket::loom_tests`.

#[cfg(loom)]
use loom::sync::atomic::AtomicU64;
#[cfg(not(loom))]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::SeqCst;
#[cfg(loom)]
use loom::thread::yield_now;
#[cfg(not(loom))]
use std::thread::yield_now;

/// The armed deadline has been delivered. Never a valid ticket.
pub const CONSUMED: u64 = 0;

/// A claimer has won the ticket and is asserting IP7. Never a valid ticket.
pub const CLAIMING: u64 = u64::MAX;

/// True for the two values that are states, not tickets.
#[inline]
pub fn is_sentinel(ticket: u64) -> bool {
    ticket == CONSUMED || ticket == CLAIMING
}

pub struct Ip7Ticket(AtomicU64);

impl Ip7Ticket {
    pub fn new() -> Self {
        Ip7Ticket(AtomicU64::new(CONSUMED))
    }

    /// The word: the armed ticket, or `CONSUMED`, or `CLAIMING` mid-delivery.
    pub fn load(&self) -> u64 {
        self.0.load(SeqCst)
    }

    /// Claim the right to deliver `ticket`, and run `deliver` (which asserts
    /// IP7) before anyone can arm a new one. Returns false, running nothing,
    /// if the ticket is no longer the armed one.
    #[inline]
    pub fn claim(&self, ticket: u64, deliver: impl FnOnce()) -> bool {
        if is_sentinel(ticket) {
            return false;
        }
        if self.0.compare_exchange(ticket, CLAIMING, SeqCst, SeqCst).is_err() {
            return false;
        }
        deliver();
        self.0.store(CONSUMED, SeqCst);
        true
    }

    /// Replace the armed ticket with `ticket`, after any delivery in flight
    /// has asserted its IP7. The caller clears IP7 next: nothing that claimed
    /// an older ticket can assert it after this returns.
    pub fn arm(&self, ticket: u64) {
        debug_assert!(!is_sentinel(ticket));
        loop {
            let cur = self.0.load(SeqCst);
            if cur != CLAIMING && self.0.compare_exchange(cur, ticket, SeqCst, SeqCst).is_ok() {
                return;
            }
            yield_now();
        }
    }

    /// A Compare write's acknowledgement: arm `ticket`, then run `clear`,
    /// which removes the pending IP7. The order is the point. Arming first
    /// means a delivery of the old deadline either lost its claim, or has
    /// finished asserting IP7 for `clear` to remove.
    #[inline]
    pub fn acknowledge(&self, ticket: u64, clear: impl FnOnce()) {
        self.arm(ticket);
        clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ticket_is_claimed_once_and_only_while_armed() {
        let t = Ip7Ticket::new();
        t.arm(1);
        assert!(!t.claim(2, || panic!("stale ticket delivered")));
        let mut n = 0;
        assert!(t.claim(1, || n += 1));
        assert!(!t.claim(1, || n += 1));
        assert_eq!((n, t.load()), (1, CONSUMED));
        t.arm(2);
        assert!(t.claim(2, || n += 1));
        assert!(!t.claim(CONSUMED, || n += 1));
        assert!(!t.claim(CLAIMING, || n += 1));
        assert_eq!(n, 2);
    }
}

#[cfg(loom)]
mod loom_tests {
    use super::*;
    use loom::sync::atomic::AtomicU64;
    use loom::sync::Arc;
    use loom::thread;

    const IP7: u64 = 1 << 15;
    const OTHER: u64 = 1 << 10;

    /// The timer callback delivers deadline 1 while the CPU thread's Compare
    /// write arms deadline 2 and clears IP7. `post` runs on the CPU thread
    /// after the acknowledgement.
    fn run(post: bool) -> u64 {
        let t = Arc::new(Ip7Ticket::new());
        let irq = Arc::new(AtomicU64::new(OTHER));
        t.arm(1);

        let cb = {
            let (t, irq) = (t.clone(), irq.clone());
            thread::spawn(move || {
                t.claim(1, || {
                    irq.fetch_or(IP7, SeqCst);
                });
            })
        };

        // The Compare write, through the function `MipsCore::write_cp0` calls.
        t.acknowledge(2, || {
            irq.fetch_and(!IP7, SeqCst);
        });
        if post {
            // A deadline the write found already missed: the CPU thread
            // delivers it itself.
            assert!(t.claim(2, || {
                irq.fetch_or(IP7, SeqCst);
            }));
        }
        cb.join().unwrap();
        irq.load(SeqCst)
    }

    /// After the guest's acknowledgement, a delivery of the superseded
    /// deadline must not leave IP7 set.
    #[test]
    fn an_acknowledged_deadline_is_not_delivered_afterwards() {
        loom::model(|| {
            let irq = run(false);
            assert_eq!(irq, OTHER, "IP7 asserted after the Compare write acknowledged it");
        });
    }

    /// The new deadline's own delivery survives the old one's.
    #[test]
    fn the_new_deadline_is_still_delivered() {
        loom::model(|| {
            let irq = run(true);
            assert_eq!(irq, OTHER | IP7, "the new deadline's IP7 was lost");
        });
    }
}
