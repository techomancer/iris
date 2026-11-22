// Cases follow tests/qtest/sn0-hub-test.c, sn0-pi-error-test.c and
// sn0-pi-gfx-test.c in jamesbraid/qemu-sgi-origami: reset-values,
// guest-writes, processor-number, cpu-enable, interrupt-modify,
// interrupt-number-selects-bank, masked-pending-raises-nothing,
// pending-banks-masked-per-processor, pending-banks-reject-writes, the rt-*
// counter and comparator cases, the prof-* profiling comparator cases,
// profiling-comparator-raises-cause-ip5, pi-rt-local-ctrl, pi-error-register-shell,
// pi-timedout-buffers, the graphics-flow-control register cases and the
// processor-error status cases.

use super::*;

struct Fixture {
    pi: Arc<Pi>,
    clock: crate::clock::Clock,
    _tm: Arc<TimerManager>,
    lines: Arc<Lines>,
    /// What the writes so far asked the Hub to do, as text.
    effects: Mutex<Vec<String>>,
}
/// The processors' interrupt words, as a plain word each.
#[derive(Default)]
struct Lines([AtomicU64; 2]);
impl Lines {
    fn get(&self, line: usize) -> bool {
        self.0[line / LINES_PER_CPU].load(Ordering::SeqCst) >> (CAUSE_SHIFT as usize + line % LINES_PER_CPU) & 1 == 1
    }
}
impl Fixture {
    fn new(cpus: u8) -> Self {
        let clock = crate::clock::Clock::stepped();
        clock.step(1_000_000);
        let tm = Arc::new(TimerManager::with_clock(clock.clone()));
        let pi = Pi::new(&tm, cpus, 0);
        let lines = Arc::new(Lines::default());
        for (slice, word) in lines.0.iter().enumerate() {
            pi.set_cpu_interrupts(slice, word);
        }
        Self { pi, clock, _tm: tm, lines, effects: Mutex::default() }
    }
    fn r(&self, off: u64) -> u64 {
        self.pi.read(off, true)
    }
    fn w(&self, off: u64, value: u64) {
        let text = match self.pi.write_sized(off, value, 8) {
            Effect::None => return,
            Effect::CachedAlias(bytes) => vec![format!("calias {bytes:#x}")],
            other => panic!("the processor interface does not return {other:?}"),
        };
        self.effects.lock().extend(text);
    }
    /// Let `ticks` counter ticks pass and run the timers that fell due.
    fn advance(&self, ticks: u64) {
        self.clock.step(ticks * TICK_NS);
        self.settle();
    }
    /// Fire the comparator timers whose deadline has come, as the timer thread would.
    fn settle(&self) {
        let now = self.clock.now();
        for kind in 0..3 {
            let due = {
                let s = self.pi.state.lock();
                [s.due[0], s.due[1], s.prof_due][kind]
            };
            if let Some(due) = due.filter(|&d| d <= now) {
                self.pi.expire(kind, due);
            }
        }
    }
    /// Whether slice `slice`'s word carries the soft-reset request.
    fn reset_requested(&self, slice: usize) -> bool {
        self.lines.0[slice].load(Ordering::SeqCst) >> 63 == 1
    }
    fn take_effects(&self) -> Vec<String> {
        std::mem::take(&mut *self.effects.lock())
    }
}

/// A reading taken on another socket can sit behind the last one; the counter does not go back.
#[test]
fn the_rt_counter_never_reads_below_the_last_reading() {
    let f = Fixture::new(1);
    f.clock.step(8_000);
    let first = f.r(reg::PI_RT_COUNT::OFFSET);
    let now = f.clock.now();
    f.clock.restore(now - 1_600);
    assert_eq!(f.r(reg::PI_RT_COUNT::OFFSET), first);
    f.clock.restore(now + 1_600);
    assert!(f.r(reg::PI_RT_COUNT::OFFSET) > first);
}

/// The clamp starts over when the guest writes the counter.
#[test]
fn a_counter_write_clears_the_rt_clamp() {
    let f = Fixture::new(1);
    f.clock.step(8_000_000);
    f.r(reg::PI_RT_COUNT::OFFSET);
    f.clock.restore(1_000_000);
    f.w(reg::PI_RT_COUNT::OFFSET, 1000);
    assert_eq!(f.r(reg::PI_RT_COUNT::OFFSET), 1000);
}

/// An unprogrammed Hub claims every region exists.
#[test]
fn the_pi_resets_and_stores_what_the_guest_writes() {
    let t = Fixture::new(2);
    assert_eq!(t.r(reg::PI_REGION_PRESENT::OFFSET), !0);
    assert_eq!(t.r(reg::PI_CALIAS_SIZE::OFFSET), 15);
    assert_eq!((t.r(reg::PI_INT_PEND0::OFFSET), t.r(reg::PI_INT_PEND1::OFFSET)), (0, 0));
    t.w(reg::PI_REGION_PRESENT::OFFSET, 0x1234);
    assert_eq!(t.r(reg::PI_REGION_PRESENT::OFFSET), 0x1234);
    t.w(reg::PI_INT_MASK0_A::OFFSET, !0);
    t.w(reg::PI_INT_MASK1_B::OFFSET, 0x55);
    assert_eq!((t.r(reg::PI_INT_MASK0_A::OFFSET), t.r(reg::PI_INT_MASK1_B::OFFSET)), (!0, 0x55));
    // The error stack address keeps bits 31:3 and survives a reset.
    t.w(reg::PI_ERR_STACK_ADDR_A::OFFSET, !0);
    t.w(reg::PI_ERR_STACK_ADDR_B::OFFSET, 0x1_2345_6789);
    assert_eq!(t.r(reg::PI_ERR_STACK_ADDR_A::OFFSET), 0xffff_fff8);
    assert_eq!(t.r(reg::PI_ERR_STACK_ADDR_B::OFFSET), 0x2345_6788);
    t.pi.power_on();
    assert_eq!(t.r(reg::PI_REGION_PRESENT::OFFSET), !0);
    assert_eq!(t.r(reg::PI_INT_MASK0_A::OFFSET), 0);
    assert_eq!(t.r(reg::PI_ERR_STACK_ADDR_A::OFFSET), 0xffff_fff8);
    assert_eq!(t.r(reg::PI_ERR_STACK_ADDR_B::OFFSET), 0x2345_6788);
}

#[test]
fn the_processor_number_names_the_reading_processor() {
    let t = Fixture::new(2);
    let pi = t.pi.clone();
    std::thread::spawn(move || {
        crate::cpu::smp::set_current_cpu(1);
        assert_eq!(pi.read(reg::PI_CPU_NUM::OFFSET, true), 1);
        crate::cpu::smp::set_current_cpu(0);
        assert_eq!(pi.read(reg::PI_CPU_NUM::OFFSET, true), 0);
    })
    .join()
    .unwrap();
    // A device thread counts as processor 0, and a store changes nothing.
    t.w(reg::PI_CPU_NUM::OFFSET, 1);
    assert_eq!(t.r(reg::PI_CPU_NUM::OFFSET), 0);
    // A node whose slice A is global CPU 2.
    let node = Pi::new(&Arc::new(TimerManager::new()), 2, 2);
    std::thread::spawn(move || {
        crate::cpu::smp::set_current_cpu(3);
        assert_eq!(node.read(reg::PI_CPU_NUM::OFFSET, true), 1);
        crate::cpu::smp::set_current_cpu(0);
        assert_eq!(node.read(reg::PI_CPU_NUM::OFFSET, true), 0);
    })
    .join()
    .unwrap();
}

#[test]
fn processor_presence_enable_nmi_and_soft_reset() {
    let t = Fixture::new(2);
    assert_eq!((t.r(reg::PI_CPU_PRESENT_A::OFFSET), t.r(reg::PI_CPU_PRESENT_B::OFFSET)), (1, 1));
    assert_eq!((t.r(reg::PI_CPU_ENABLE_A::OFFSET), t.r(reg::PI_CPU_ENABLE_B::OFFSET)), (1, 1));
    t.w(reg::PI_CPU_ENABLE_A::OFFSET, 0);
    t.w(reg::PI_CPU_ENABLE_B::OFFSET, 0xff);
    assert_eq!(t.r(reg::PI_CPU_ENABLE_B::OFFSET), 1);
    // The enable and the NMI are logged and ask nothing of the machine.
    assert!(t.take_effects().is_empty());
    assert_eq!((t.reset_requested(0), t.reset_requested(1)), (false, false));
    // The soft reset reaches only the enabled processors.
    t.w(reg::PI_SOFTRESET::OFFSET, 1);
    assert_eq!((t.reset_requested(0), t.reset_requested(1)), (false, true));
    assert!(t.take_effects().is_empty());
    t.w(reg::PI_NMI_A::OFFSET, 1);
    t.w(reg::PI_NMI_B::OFFSET, 0);
    assert!(t.take_effects().is_empty());
    assert_eq!(t.lines.0[0].load(Ordering::SeqCst), 0, "NMI does not reach the processor");
    // Reset enables the populated slices again.
    t.pi.power_on();
    assert_eq!((t.r(reg::PI_CPU_ENABLE_A::OFFSET), t.r(reg::PI_CPU_ENABLE_B::OFFSET)), (1, 1));
}

#[test]
fn an_absent_processors_enable_stores_and_reaches_nothing() {
    let t = Fixture::new(1);
    assert_eq!((t.r(reg::PI_CPU_PRESENT_A::OFFSET), t.r(reg::PI_CPU_PRESENT_B::OFFSET)), (1, 0));
    assert_eq!((t.r(reg::PI_CPU_ENABLE_A::OFFSET), t.r(reg::PI_CPU_ENABLE_B::OFFSET)), (1, 0));
    t.w(reg::PI_CPU_ENABLE_A::OFFSET, 0);
    t.w(reg::PI_CPU_ENABLE_B::OFFSET, 1);
    assert_eq!(t.r(reg::PI_CPU_ENABLE_B::OFFSET), 1);
    t.w(reg::PI_NMI_B::OFFSET, 1);
    t.w(reg::PI_SOFTRESET::OFFSET, 1);
    assert!(t.take_effects().is_empty());
    assert_eq!((t.reset_requested(0), t.reset_requested(1)), (false, false));
    t.w(reg::PI_CPU_ENABLE_A::OFFSET, 1);
    t.w(reg::PI_SOFTRESET::OFFSET, 1);
    assert_eq!((t.reset_requested(0), t.reset_requested(1)), (true, false));
    // Nor does the absent slot get an interrupt line.
    t.w(reg::PI_INT_MASK0_B::OFFSET, !0);
    t.w(reg::PI_INT_PEND_MOD::OFFSET, MODIFY_SET | 3);
    assert!(!t.lines.get(4));
}

#[test]
fn the_cached_alias_size_codes_double_from_4_kib() {
    let t = Fixture::new(2);
    for (code, bytes) in [(0, 0), (1, 4 << 10), (2, 8 << 10), (15, 64 << 20)] {
        t.w(reg::PI_CALIAS_SIZE::OFFSET, code);
        assert_eq!(t.take_effects(), [format!("calias {bytes:#x}")]);
    }
    // Four bits store; a wider value wraps into them.
    t.w(reg::PI_CALIAS_SIZE::OFFSET, 0x31);
    assert_eq!(t.r(reg::PI_CALIAS_SIZE::OFFSET), 1);
}

#[test]
fn interrupt_modify_sets_and_clears_one_bit_of_one_bank() {
    let t = Fixture::new(2);
    t.w(reg::PI_INT_PEND_MOD::OFFSET, MODIFY_SET | 12);
    assert_eq!((t.r(reg::PI_INT_PEND0::OFFSET), t.r(reg::PI_INT_PEND1::OFFSET)), (1 << 12, 0));
    t.w(reg::PI_INT_PEND_MOD::OFFSET, 12);
    assert_eq!(t.r(reg::PI_INT_PEND0::OFFSET), 0);
    // Number 70 is bit 6 of the second bank.
    t.w(reg::PI_INT_PEND_MOD::OFFSET, MODIFY_SET | 70);
    assert_eq!((t.r(reg::PI_INT_PEND0::OFFSET), t.r(reg::PI_INT_PEND1::OFFSET)), (0, 1 << 6));
    // The modify register is an instruction.
    assert_eq!(t.r(reg::PI_INT_PEND_MOD::OFFSET), 0);
    t.w(reg::PI_INT_PEND0::OFFSET, !0);
    t.w(reg::PI_INT_PEND1::OFFSET, !0);
    assert_eq!((t.r(reg::PI_INT_PEND0::OFFSET), t.r(reg::PI_INT_PEND1::OFFSET)), (0, 1 << 6));
}

#[test]
fn pending_bits_raise_a_line_only_through_a_mask_and_follow_it_down() {
    let t = Fixture::new(2);
    // Masked pending is invisible.
    t.w(reg::PI_INT_PEND_MOD::OFFSET, MODIFY_SET | 12);
    assert!(!t.lines.get(0));
    t.w(reg::PI_INT_MASK0_A::OFFSET, 1 << 12);
    assert!(t.lines.get(0));
    // Level-triggered: clearing the last visible bit drops it.
    t.w(reg::PI_INT_PEND_MOD::OFFSET, 12);
    assert!(!t.lines.get(0));
    // Bank 1 has its own line: number 70 does not raise bank 0's.
    t.w(reg::PI_INT_MASK0_A::OFFSET, !0);
    t.w(reg::PI_INT_MASK1_A::OFFSET, !0);
    t.w(reg::PI_INT_PEND_MOD::OFFSET, MODIFY_SET | 70);
    assert!(!t.lines.get(0) && t.lines.get(1));
}

#[test]
fn the_two_processors_have_their_own_masks() {
    let t = Fixture::new(2);
    t.w(reg::PI_INT_MASK0_A::OFFSET, 0);
    t.w(reg::PI_INT_MASK0_B::OFFSET, 1 << 12);
    t.w(reg::PI_INT_PEND_MOD::OFFSET, MODIFY_SET | 12);
    assert!(!t.lines.get(0) && t.lines.get(4));
    t.w(reg::PI_INT_PEND_MOD::OFFSET, 12);
    assert!(!t.lines.get(4));
    // The PROM's order for B: pend while masked, then unmask.
    t.w(reg::PI_INT_MASK0_B::OFFSET, 0);
    t.w(reg::PI_INT_PEND_MOD::OFFSET, MODIFY_SET | 12);
    assert!(!t.lines.get(4));
    t.w(reg::PI_INT_MASK0_B::OFFSET, 1 << 12);
    assert!(t.lines.get(4));
    // B's second bank must not follow the first bank's pending bit.
    t.w(reg::PI_INT_MASK1_B::OFFSET, !0);
    assert!(!t.lines.get(5));
}

#[test]
fn the_counter_runs_at_1_25_mhz_in_52_bits_and_can_be_written() {
    let t = Fixture::new(1);
    t.w(reg::PI_RT_COUNT::OFFSET, 0);
    assert_eq!(t.r(reg::PI_RT_COUNT::OFFSET), 0);
    // 800 ns is one tick, and the counter moves between two reads.
    t.clock.step(799);
    assert_eq!(t.r(reg::PI_RT_COUNT::OFFSET), 0);
    t.clock.step(1);
    assert_eq!(t.r(reg::PI_RT_COUNT::OFFSET), 1);
    // A whole second is 1,250,000 ticks.
    t.advance(1_250_000 - 1);
    assert_eq!(t.r(reg::PI_RT_COUNT::OFFSET), 1_250_000);
    // A write moves the origin and the counter keeps running from it.
    t.w(reg::PI_RT_COUNT::OFFSET, 0x000a_bcde_f012_3456);
    assert_eq!(t.r(reg::PI_RT_COUNT::OFFSET), 0x000a_bcde_f012_3456);
    t.advance(10);
    assert_eq!(t.r(reg::PI_RT_COUNT::OFFSET), 0x000a_bcde_f012_3460);
    // Fifty-two bits: the top of the range wraps to zero.
    t.w(reg::PI_RT_COUNT::OFFSET, RT_MASK);
    t.advance(1);
    assert_eq!(t.r(reg::PI_RT_COUNT::OFFSET), 0);
    t.w(reg::PI_RT_COUNT::OFFSET, !0);
    assert_eq!(t.r(reg::PI_RT_COUNT::OFFSET), RT_MASK);
    // Hours of elapsed time stay exact.
    t.w(reg::PI_RT_COUNT::OFFSET, 0);
    t.advance(1_250_000 * 3600 * 5);
    assert_eq!(t.r(reg::PI_RT_COUNT::OFFSET), 1_250_000 * 3600 * 5);
}

#[test]
fn a_comparator_pends_when_the_counter_reaches_it_and_the_enable_gates_the_line() {
    let t = Fixture::new(2);
    t.w(reg::PI_RT_COUNT::OFFSET, 100);
    t.w(reg::PI_RT_COMPARE_A::OFFSET, 150);
    t.advance(49);
    assert_eq!((t.r(reg::PI_RT_PEND_A::OFFSET), t.lines.get(2)), (0, false));
    // A match pends whether or not enabled.
    t.advance(1);
    assert_eq!(t.r(reg::PI_RT_PEND_A::OFFSET), 1);
    assert!(!t.lines.get(2));
    t.w(reg::PI_RT_EN_A::OFFSET, 1);
    assert!(t.lines.get(2));
    // Writing zero acknowledges and drops the line.
    t.w(reg::PI_RT_PEND_A::OFFSET, 0);
    assert!(!t.lines.get(2));
    // A match fires once; nothing re-arms it until software writes again.
    t.advance(10_000);
    assert_eq!(t.r(reg::PI_RT_PEND_A::OFFSET), 0);
    t.w(reg::PI_RT_COMPARE_A::OFFSET, 10_200);
    t.advance(49);
    assert_eq!(t.r(reg::PI_RT_PEND_A::OFFSET), 0);
    t.advance(1);
    assert_eq!(t.r(reg::PI_RT_PEND_A::OFFSET), 1);
    // The second processor has its own comparator and line.
    t.w(reg::PI_RT_EN_B::OFFSET, 1);
    t.w(reg::PI_RT_COMPARE_B::OFFSET, t.r(reg::PI_RT_COUNT::OFFSET) + 5);
    t.advance(5);
    assert!(t.lines.get(6));
    assert_eq!(t.r(reg::PI_RT_PEND_B::OFFSET), 1);
}

#[test]
fn a_comparator_behind_the_counter_waits_a_whole_revolution() {
    let t = Fixture::new(1);
    t.w(reg::PI_RT_EN_A::OFFSET, 1);
    // The guest kernel parks a processor's tick by writing a value behind the counter.
    t.w(reg::PI_RT_COUNT::OFFSET, 1000);
    t.w(reg::PI_RT_COMPARE_A::OFFSET, 0);
    t.advance(500_000);
    assert_eq!((t.r(reg::PI_RT_PEND_A::OFFSET), t.lines.get(2)), (0, false));
    // The counter wrapping round reaches the zeroed comparator.
    t.w(reg::PI_RT_COMPARE_A::OFFSET, 0);
    t.w(reg::PI_RT_COUNT::OFFSET, RT_MASK - 2);
    t.advance(2);
    assert_eq!(t.r(reg::PI_RT_PEND_A::OFFSET), 0);
    t.advance(1);
    assert_eq!(t.r(reg::PI_RT_PEND_A::OFFSET), 1);
}

/// The running timer, not a read, pends the comparator once the clock reaches it.
#[test]
fn a_started_comparator_timer_pends_the_bit_at_the_deadline() {
    let t = Fixture::new(1);
    t.pi.start();
    t.w(reg::PI_RT_COMPARE_A::OFFSET, t.r(reg::PI_RT_COUNT::OFFSET) + 10);
    t.clock.step(9 * TICK_NS);
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert_eq!(t.pi.rt_pending[0].load(Ordering::SeqCst), false);
    t.clock.step(TICK_NS);
    assert!(crate::test_board::until(std::time::Duration::from_secs(10), || t.pi.rt_pending[0].load(Ordering::SeqCst)));
    t.pi.stop();
}

/// Equal is not behind.
#[test]
fn a_comparator_written_equal_to_the_counter_matches_at_once() {
    let t = Fixture::new(2);
    t.w(reg::PI_RT_EN_A::OFFSET, 1);
    t.w(reg::PI_RT_COUNT::OFFSET, 1_250_000);
    t.w(reg::PI_RT_COMPARE_A::OFFSET, t.r(reg::PI_RT_COUNT::OFFSET));
    assert_eq!((t.r(reg::PI_RT_PEND_A::OFFSET), t.lines.get(2)), (1, true));
    // Once: acknowledged, it stays quiet.
    t.w(reg::PI_RT_PEND_A::OFFSET, 0);
    t.advance(4 * 1250);
    assert_eq!((t.r(reg::PI_RT_PEND_A::OFFSET), t.lines.get(2)), (0, false));
    // Moving the counter onto a comparator is not a write of it: no match.
    t.w(reg::PI_RT_COMPARE_A::OFFSET, 5000);
    t.w(reg::PI_RT_COUNT::OFFSET, 5000);
    t.advance(4 * 1250);
    assert_eq!(t.r(reg::PI_RT_PEND_A::OFFSET), 0);
}

#[test]
fn moving_the_counter_pends_nothing_but_rearms_the_deadline() {
    let t = Fixture::new(1);
    t.w(reg::PI_RT_COUNT::OFFSET, 0);
    t.w(reg::PI_RT_COMPARE_A::OFFSET, 100);
    // Jumping the counter past the compare value is not a match.
    t.w(reg::PI_RT_COUNT::OFFSET, 500);
    t.advance(1000);
    assert_eq!(t.r(reg::PI_RT_PEND_A::OFFSET), 0);
    // Jumping to just before it re-arms against the new counter.
    t.w(reg::PI_RT_COUNT::OFFSET, 90);
    t.advance(9);
    assert_eq!(t.r(reg::PI_RT_PEND_A::OFFSET), 0);
    t.advance(1);
    assert_eq!(t.r(reg::PI_RT_PEND_A::OFFSET), 1);
    // The longest interval is a revolution minus a tick.
    t.w(reg::PI_RT_PEND_A::OFFSET, 0);
    t.w(reg::PI_RT_COUNT::OFFSET, 0);
    t.w(reg::PI_RT_COMPARE_A::OFFSET, RT_MASK);
    t.advance(RT_MASK - 1);
    assert_eq!(t.r(reg::PI_RT_PEND_A::OFFSET), 0);
    t.advance(1);
    assert_eq!(t.r(reg::PI_RT_PEND_A::OFFSET), 1);
}

/// One profiling comparator for the node.
#[test]
fn the_profiling_comparator_pends_both_processors_once() {
    let t = Fixture::new(2);
    t.w(reg::PI_RT_COUNT::OFFSET, 1_250_000);
    let count = t.r(reg::PI_RT_COUNT::OFFSET);
    t.w(reg::PI_PROFILE_COMPARE::OFFSET, count + 1250);
    assert_eq!(t.r(reg::PI_PROFILE_COMPARE::OFFSET), count + 1250);
    assert_eq!((t.r(reg::PI_PROF_PEND_A::OFFSET), t.r(reg::PI_PROF_PEND_B::OFFSET)), (0, 0));
    t.advance(1249);
    assert_eq!((t.r(reg::PI_PROF_PEND_A::OFFSET), t.r(reg::PI_PROF_PEND_B::OFFSET)), (0, 0));
    t.advance(1);
    assert_eq!((t.r(reg::PI_PROF_PEND_A::OFFSET), t.r(reg::PI_PROF_PEND_B::OFFSET)), (1, 1));
    // Each processor acknowledges its own bit.
    t.w(reg::PI_PROF_PEND_A::OFFSET, 0);
    assert_eq!((t.r(reg::PI_PROF_PEND_A::OFFSET), t.r(reg::PI_PROF_PEND_B::OFFSET)), (0, 1));
    t.w(reg::PI_PROF_PEND_B::OFFSET, 0);
    // Once: nothing pends again until the comparator is written.
    t.advance(2 * 1250);
    assert_eq!((t.r(reg::PI_PROF_PEND_A::OFFSET), t.r(reg::PI_PROF_PEND_B::OFFSET)), (0, 0));
    // The re-arm the fast clock makes from its handler fires again.
    t.w(reg::PI_PROFILE_COMPARE::OFFSET, t.r(reg::PI_RT_COUNT::OFFSET) + 1250);
    t.advance(1250);
    assert_eq!(t.r(reg::PI_PROF_PEND_A::OFFSET), 1);
    // It is not the processors' comparators: those stayed quiet.
    assert_eq!((t.r(reg::PI_RT_PEND_A::OFFSET), t.r(reg::PI_RT_PEND_B::OFFSET)), (0, 0));
}

/// Behind the counter, the profiling comparator waits for the wrap too.
#[test]
fn a_profiling_comparator_behind_the_counter_does_not_match() {
    let t = Fixture::new(2);
    t.w(reg::PI_RT_COUNT::OFFSET, 1_250_000);
    t.w(reg::PI_PROFILE_COMPARE::OFFSET, t.r(reg::PI_RT_COUNT::OFFSET) - 1250);
    t.advance(4 * 1250);
    assert_eq!((t.r(reg::PI_PROF_PEND_A::OFFSET), t.r(reg::PI_PROF_PEND_B::OFFSET)), (0, 0));
}

/// The guest kernel's fast clock re-arms one interval past the counter only when the comparator is
/// strictly behind.
#[test]
fn a_profiling_comparator_written_equal_to_the_counter_matches_at_once() {
    let t = Fixture::new(2);
    t.w(reg::PI_RT_COUNT::OFFSET, 1_250_000);
    t.w(reg::PI_PROFILE_COMPARE::OFFSET, t.r(reg::PI_RT_COUNT::OFFSET));
    assert_eq!((t.r(reg::PI_PROF_PEND_A::OFFSET), t.r(reg::PI_PROF_PEND_B::OFFSET)), (1, 1));
}

/// Reset clears the profiling comparator, its bits and a match standing armed.
#[test]
fn a_reset_clears_the_profiling_comparator() {
    let t = Fixture::new(2);
    t.w(reg::PI_RT_COUNT::OFFSET, 1_250_000);
    t.w(reg::PI_PROF_EN_A::OFFSET, 1);
    t.w(reg::PI_PROF_EN_B::OFFSET, 1);
    t.w(reg::PI_PROFILE_COMPARE::OFFSET, t.r(reg::PI_RT_COUNT::OFFSET));
    assert_eq!(t.r(reg::PI_PROF_PEND_A::OFFSET), 1);
    assert!(t.lines.get(3) && t.lines.get(7));
    // Armed again a millisecond out, so a surviving deadline would show.
    t.w(reg::PI_PROFILE_COMPARE::OFFSET, t.r(reg::PI_RT_COUNT::OFFSET) + 1250);
    t.pi.power_on();
    assert!(!t.lines.get(3) && !t.lines.get(7));
    for off in [
        reg::PI_PROFILE_COMPARE::OFFSET,
        reg::PI_PROF_PEND_A::OFFSET,
        reg::PI_PROF_PEND_B::OFFSET,
        reg::PI_PROF_EN_A::OFFSET,
        reg::PI_PROF_EN_B::OFFSET,
    ] {
        assert_eq!(t.r(off), 0, "{off:#x}");
    }
    t.advance(4 * 1250);
    assert_eq!(t.r(reg::PI_PROF_PEND_A::OFFSET), 0);
}

/// The profiling comparator reaches each processor as hardware interrupt 3.
#[test]
fn the_profiling_comparator_raises_cause_ip5() {
    const IP_MASK: u64 = 0x3f00;
    const IP5: u64 = 0x2000;
    let t = Fixture::new(2);
    let cause = |slice: usize| t.lines.0[slice].load(Ordering::SeqCst) & IP_MASK;
    t.w(reg::PI_RT_COUNT::OFFSET, 0);
    t.w(reg::PI_PROF_EN_A::OFFSET, 1);
    assert_eq!(t.r(reg::PI_PROF_EN_A::OFFSET), 1);
    t.w(reg::PI_PROFILE_COMPARE::OFFSET, 1250);
    assert_eq!((cause(0), cause(1)), (0, 0));
    t.advance(1250);
    assert_eq!((cause(0), cause(1)), (IP5, 0));
    t.w(reg::PI_PROF_PEND_A::OFFSET, 0);
    assert_eq!(cause(0), 0);
    // The second processor's bit stood pending; enabling it raises IP5.
    t.w(reg::PI_PROF_EN_B::OFFSET, 1);
    assert_eq!((cause(0), cause(1)), (0, IP5));
}

#[test]
fn the_local_control_register_stores_bit_16_and_the_low_byte() {
    let t = Fixture::new(1);
    assert_eq!(t.r(reg::PI_RT_LOCAL_CTRL::OFFSET), 0x1_0000);
    t.w(reg::PI_RT_LOCAL_CTRL::OFFSET, 0x1_0047);
    assert_eq!(t.r(reg::PI_RT_LOCAL_CTRL::OFFSET), 0x1_0047);
    t.w(reg::PI_RT_LOCAL_CTRL::OFFSET, !0);
    assert_eq!(t.r(reg::PI_RT_LOCAL_CTRL::OFFSET), 0x1_00ff);
    // Nothing written there changes the rate.
    t.w(reg::PI_RT_COUNT::OFFSET, 0);
    t.advance(10);
    assert_eq!(t.r(reg::PI_RT_COUNT::OFFSET), 10);
    t.pi.power_on();
    assert_eq!(t.r(reg::PI_RT_LOCAL_CTRL::OFFSET), 0x1_0000);
}

#[test]
fn the_error_registers_store_masked_values_and_nothing_times_out() {
    let t = Fixture::new(2);
    assert_eq!(t.r(reg::PI_ERR_INT_PEND::OFFSET), 0);
    t.w(reg::PI_ERR_INT_PEND::OFFSET, !0);
    assert_eq!(t.r(reg::PI_ERR_INT_PEND::OFFSET), 0);
    t.w(reg::PI_ERR_INT_MASK_A::OFFSET, !0);
    t.w(reg::PI_ERR_INT_MASK_B::OFFSET, 0x0123456);
    assert_eq!((t.r(reg::PI_ERR_INT_MASK_A::OFFSET), t.r(reg::PI_ERR_INT_MASK_B::OFFSET)), (0x01ff_ffff, 0x0123456));
    t.w(reg::PI_ERR_STACK_SIZE::OFFSET, !0);
    assert_eq!(t.r(reg::PI_ERR_STACK_SIZE::OFFSET), 0xf);
    for off in [reg::PI_CRB_TIMEOUT_A::OFFSET, reg::PI_CRB_TIMEOUT_B::OFFSET] {
        t.w(off, !0);
        assert_eq!(t.r(off), 0);
    }
    t.pi.power_on();
    assert_eq!((t.r(reg::PI_ERR_INT_MASK_A::OFFSET), t.r(reg::PI_ERR_INT_MASK_B::OFFSET), t.r(reg::PI_ERR_STACK_SIZE::OFFSET)), (0, 0, 0));
    for off in (reg::PI_ERR_STATUS0_A::OFFSET..=reg::PI_ERR_STATUS1_B_RCLR::OFFSET).step_by(8) {
        assert_eq!(t.r(off), 0, "{off:#x}");
    }
}

#[test]
fn a_failed_read_latches_the_first_address_and_later_ones_only_set_overrun() {
    let t = Fixture::new(2);
    // The PROM's probe: a failed doubleword read at 0x0240_0208 of node 0.
    t.pi.read_error(1, 0x0240_0208);
    let first = STATUS0_VALID | ((0x0240_0208u64 >> 3) << 25);
    assert_eq!(t.r(reg::PI_ERR_STATUS0_B::OFFSET), first);
    assert_eq!(t.r(reg::PI_ERR_STATUS0_A::OFFSET), 0);
    t.pi.read_error(1, 0x1234_5678);
    assert_eq!(t.r(reg::PI_ERR_STATUS0_B::OFFSET), first | STATUS0_OVERRUN);
    // The kernel checks bits 61:25 and a 40-bit address; the type fields stay zero.
    t.pi.read_error(0, 0xff_0000_0008);
    assert_eq!(t.r(reg::PI_ERR_STATUS0_A::OFFSET), STATUS0_VALID | ((0xff_0000_0008u64 >> 3) << 25));
    // A slice the node does not populate records nothing.
    Fixture::new(1).pi.read_error(1, 8);
}

#[test]
fn a_read_to_clear_alias_clears_only_on_a_full_width_read() {
    let t = Fixture::new(1);
    t.pi.read_error(0, 0x240_0208);
    let value = t.r(reg::PI_ERR_STATUS0_A::OFFSET);
    // The 32-bit half path reads without clearing.
    assert_eq!(t.pi.read(reg::PI_ERR_STATUS0_A_RCLR::OFFSET, false), value);
    assert_eq!(t.r(reg::PI_ERR_STATUS0_A::OFFSET), value);
    assert_eq!(t.r(reg::PI_ERR_STATUS0_A_RCLR::OFFSET), value);
    assert_eq!(t.r(reg::PI_ERR_STATUS0_A::OFFSET), 0);
}

#[test]
fn a_full_width_zero_store_to_either_alias_ends_the_record() {
    let t = Fixture::new(2);
    for alias in [reg::PI_ERR_STATUS0_A_RCLR::OFFSET, reg::PI_ERR_STATUS1_A_RCLR::OFFSET] {
        t.pi.read_error(0, 0x240_0208);
        // A narrow or nonzero store does not clear.
        t.pi.write_sized(alias, 0, 4);
        t.pi.write_sized(alias, 1, 8);
        assert_ne!(t.r(reg::PI_ERR_STATUS0_A::OFFSET), 0);
        t.pi.write_sized(alias, 0, 8);
        assert_eq!(t.r(reg::PI_ERR_STATUS0_A::OFFSET), 0);
    }
    // B's aliases clear B alone.
    t.pi.read_error(0, 8);
    t.pi.read_error(1, 8);
    t.pi.write_sized(reg::PI_ERR_STATUS1_B_RCLR::OFFSET, 0, 8);
    assert_eq!((t.r(reg::PI_ERR_STATUS0_B::OFFSET), t.r(reg::PI_ERR_STATUS0_A::OFFSET) != 0), (0, true));
    // Reset clears both slices.
    t.pi.read_error(1, 8);
    t.pi.power_on();
    assert_eq!((t.r(reg::PI_ERR_STATUS0_A::OFFSET), t.r(reg::PI_ERR_STATUS0_B::OFFSET)), (0, 0));
}

#[test]
fn the_graphics_flow_control_registers_keep_their_field_widths() {
    let t = Fixture::new(2);
    const PAGE_MASK: u64 = (1 << 40) | 0xff_fffe_0000;
    let slot = |a: u64| reg::PI_GFX_PAGE_A::OFFSET + a * 8 * 5;
    for s in 0..2 {
        t.w(slot(s), 0x0123_4567_89ab_cdef);
        assert_eq!(t.r(slot(s)), 0x0123_4567_89ab_cdef & PAGE_MASK);
        // Credit and bias are twelve bits, the interrupt pair 26.
        for (index, mask) in [(1, 0xfff), (2, 0xfff), (3, 0x3ff_ffff), (4, 0x3ff_ffff)] {
            t.w(slot(s) + index * 8, !0);
            assert_eq!(t.r(slot(s) + index * 8), mask);
        }
    }
    // A and B are independent.
    t.w(reg::PI_GFX_PAGE_A::OFFSET + 8, 0x65);
    t.w(slot(1) + 8, 0xab);
    assert_eq!((t.r(reg::PI_GFX_PAGE_A::OFFSET + 8), t.r(slot(1) + 8)), (0x65, 0xab));
    // The block shares nothing with the comparator.
    t.w(reg::PI_RT_COMPARE_A::OFFSET, 0x1_2345_6789_abcd);
    t.w(reg::PI_GFX_PAGE_A::OFFSET, !0);
    assert_eq!(t.r(reg::PI_RT_COMPARE_A::OFFSET), 0x1_2345_6789_abcd);
    // All ten reset to zero.
    t.pi.power_on();
    for n in 0..10 {
        assert_eq!(t.r(reg::PI_GFX_PAGE_A::OFFSET + n * 8), 0);
    }
}

/// The guest kernel attaches a pipe by writing its page with the enable set.
#[test]
fn the_graphics_attach_sequence_leaves_the_credit_as_written() {
    let t = Fixture::new(2);
    let (page, credit, bias) = (reg::PI_GFX_PAGE_A::OFFSET, reg::PI_GFX_PAGE_A::OFFSET + 8, reg::PI_GFX_PAGE_A::OFFSET + 16);
    t.w(bias, 0);
    t.w(credit, 0x65);
    t.w(page, 0x1234_0000 | (1 << 40));
    t.w(bias, 0x44);
    assert_eq!((t.r(credit), t.r(bias), t.r(page)), (0x65, 0x44, 0x1234_0000 | (1 << 40)));
    assert!(t.r(credit) >= t.r(bias));
    // No pending bit follows from the bias: the interrupt is not modelled.
    assert_eq!(t.r(reg::PI_INT_PEND0::OFFSET), 0);
}

#[test]
fn the_hard_reset_bit_and_unimplemented_offsets_take_stores_and_read_zero() {
    let t = Fixture::new(1);
    for (off, value) in [(reg::PI_HARDRESET_BIT::OFFSET, 0), (reg::PI_HARDRESET_BIT::OFFSET, 1), (0x1f8, 0x5555)] {
        t.w(off, value);
        assert_eq!(t.r(off), 0, "{off:#x}");
    }
}

/// A write that changes no line writes no edge.
#[test]
fn a_write_that_changes_no_line_makes_no_edge() {
    let t = Fixture::new(1);
    t.w(reg::PI_INT_MASK0_A::OFFSET, !0);
    t.pi.int_modify(MODIFY_SET | 5);
    assert!(t.lines.get(0));
    let edges = |t: &Fixture| (t.pi.rises[0].load(Ordering::SeqCst), t.pi.falls[0].load(Ordering::SeqCst));
    assert_eq!(edges(&t), (1, 0));
    // The bit again, the mask again, a drive: the line stays up.
    t.pi.int_modify(MODIFY_SET | 5);
    t.pi.int_modify(MODIFY_SET | 6);
    t.w(reg::PI_INT_MASK0_A::OFFSET, !0);
    t.pi.change(|| {});
    assert_eq!(edges(&t), (1, 0));
    // One bit clears and the other holds the line.
    t.pi.int_modify(5);
    assert_eq!(edges(&t), (1, 0));
    t.pi.int_modify(6);
    assert!(!t.lines.get(0));
    assert_eq!(edges(&t), (1, 1));
}

/// Four threads pulse four inputs while two drive the lines: one rise and one fall per pulse.
#[test]
fn threads_pulsing_and_driving_make_one_rise_and_one_fall_per_pulse() {
    const PULSES: u64 = 4000;
    let t = Arc::new(Fixture::new(1));
    t.w(reg::PI_INT_MASK0_A::OFFSET, !0);
    t.w(reg::PI_RT_EN_A::OFFSET, 1);
    t.w(reg::PI_PROF_EN_A::OFFSET, 1);
    let stop = Arc::new(AtomicBool::new(false));
    let noisy: Vec<_> = (0..2)
        .map(|n| {
            let (t, stop) = (t.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    t.pi.change(|| {});
                    // The same values the registers hold.
                    t.pi.write_sized(reg::PI_INT_MASK0_A::OFFSET, !0, 8);
                    t.pi.write_sized(reg::PI_RT_EN_A::OFFSET, 1, 8);
                    t.pi.write_sized(reg::PI_PROF_EN_A::OFFSET, 1, 8);
                    if n == 0 {
                        t.pi.write_sized(reg::PI_INT_MASK1_A::OFFSET, 0, 8);
                    }
                }
            })
        })
        .collect();
    let bank = {
        let t = t.clone();
        std::thread::spawn(move || {
            for _ in 0..PULSES {
                t.pi.int_modify(MODIFY_SET | 5);
                t.pi.int_modify(5);
            }
        })
    };
    let comparator = {
        let t = t.clone();
        std::thread::spawn(move || {
            for _ in 0..PULSES {
                t.pi.write_sized(reg::PI_RT_PEND_A::OFFSET, 1, 8);
                t.pi.write_sized(reg::PI_RT_PEND_A::OFFSET, 0, 8);
            }
        })
    };
    let profiling = {
        let t = t.clone();
        std::thread::spawn(move || {
            for _ in 0..PULSES {
                t.pi.write_sized(reg::PI_PROF_PEND_A::OFFSET, 1, 8);
                t.pi.write_sized(reg::PI_PROF_PEND_A::OFFSET, 0, 8);
            }
        })
    };
    bank.join().unwrap();
    comparator.join().unwrap();
    profiling.join().unwrap();
    stop.store(true, Ordering::SeqCst);
    for n in noisy {
        n.join().unwrap();
    }
    let count = |a: &[AtomicU64; 2 * LINES_PER_CPU]| [0, 1, 2, 3].map(|l| a[l].load(Ordering::SeqCst));
    assert_eq!(count(&t.pi.rises), [PULSES, 0, PULSES, PULSES]);
    assert_eq!(count(&t.pi.falls), [PULSES, 0, PULSES, PULSES]);
    assert_eq!((t.lines.get(0), t.lines.get(1), t.lines.get(2), t.lines.get(3)), (false, false, false, false));
}

/// Writers that race an input against a mask.
#[test]
fn racing_writers_leave_the_word_matching_the_registers() {
    for round in 0..40 {
        let t = Arc::new(Fixture::new(1));
        let (a, b, c) = (t.clone(), t.clone(), t.clone());
        let setter = std::thread::spawn(move || {
            for _ in 0..2000 {
                a.pi.int_modify(MODIFY_SET | 5);
                a.pi.int_modify(5);
            }
        });
        let masker = std::thread::spawn(move || {
            for i in 0..2000u64 {
                b.pi.write_sized(reg::PI_INT_MASK0_A::OFFSET, if i % 2 == 0 { !0 } else { 0 }, 8);
            }
        });
        let enabler = std::thread::spawn(move || {
            for i in 0..2000u64 {
                c.pi.write_sized(reg::PI_RT_PEND_A::OFFSET, 1, 8);
                c.pi.write_sized(reg::PI_RT_EN_A::OFFSET, i & 1, 8);
            }
        });
        for h in [setter, masker, enabler] {
            h.join().unwrap();
        }
        let bank0 = t.r(reg::PI_INT_PEND0::OFFSET) & t.r(reg::PI_INT_MASK0_A::OFFSET) != 0;
        let rt = t.r(reg::PI_RT_PEND_A::OFFSET) != 0 && t.r(reg::PI_RT_EN_A::OFFSET) != 0;
        assert_eq!((t.lines.get(0), t.lines.get(1), t.lines.get(2)), (bank0, false, rt), "round {round}");
    }
}

/// Let a thread that has started get as far as the lock it should be waiting on.
fn give_a_blocked_thread_the_chance_to_run() {
    for _ in 0..20_000 {
        std::thread::yield_now();
    }
}

/// A writer that stalls in the middle of writing the lines holds off every change of their inputs.
#[test]
fn a_stalled_line_write_holds_off_changes_to_its_input() {
    use std::sync::Barrier;
    let t = Arc::new(Fixture::new(1));
    t.w(reg::PI_INT_MASK0_A::OFFSET, !0);
    let (entered, release) = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
    {
        let (entered, release) = (entered.clone(), release.clone());
        *t.pi.stall.lock() = Some(Arc::new(move || {
            entered.wait();
            release.wait();
        }));
    }
    let first = {
        let t = t.clone();
        std::thread::spawn(move || t.pi.int_modify(MODIFY_SET | 5))
    };
    // The first writer has set the bit and is stalled before writing the line.
    entered.wait();
    assert_eq!(t.pi.pending(0), 1 << 5);
    assert!(!t.lines.get(0), "the stalled writer has not written the line");
    let started = Arc::new(Barrier::new(2));
    let second = {
        let (t, started) = (t.clone(), started.clone());
        std::thread::spawn(move || {
            started.wait();
            t.pi.int_modify(5);
            t.pi.int_modify(MODIFY_SET | 5);
        })
    };
    started.wait();
    give_a_blocked_thread_the_chance_to_run();
    assert_eq!(t.pi.pending(0), 1 << 5, "the input changed while another writer held the lines");
    assert!(!second.is_finished());
    release.wait();
    first.join().unwrap();
    second.join().unwrap();
    assert!(t.lines.get(0));
    // The bit rose, fell and rose: three edges in the order the registers made them.
    assert_eq!((t.pi.rises[0].load(Ordering::SeqCst), t.pi.falls[0].load(Ordering::SeqCst)), (2, 1));
}

/// The comparator timer's callback runs outside the timer manager's lock.
#[test]
fn clearing_the_words_waits_for_a_publish_in_flight() {
    use std::sync::Barrier;
    let t = Arc::new(Fixture::new(1));
    t.w(reg::PI_INT_MASK0_A::OFFSET, !0);
    let (entered, release) = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
    {
        let (entered, release) = (entered.clone(), release.clone());
        *t.pi.stall.lock() = Some(Arc::new(move || {
            entered.wait();
            release.wait();
        }));
    }
    // A publish that will raise line 0, as the timer's callback does for the comparator's line.
    let publisher = {
        let t = t.clone();
        std::thread::spawn(move || t.pi.int_modify(MODIFY_SET | 5))
    };
    // It has loaded the word and is stalled before writing to it.
    entered.wait();
    let clearer = {
        let t = t.clone();
        // How many rises the word had seen by the time the clear returned.
        std::thread::spawn(move || {
            t.pi.clear_cpu_interrupts();
            t.pi.rises[0].load(Ordering::SeqCst)
        })
    };
    // The clear must not return while the publish is stalled.
    give_a_blocked_thread_the_chance_to_run();
    assert!(!clearer.is_finished(), "the clear returned while a publish held the word");
    release.wait();
    publisher.join().unwrap();
    assert_eq!(clearer.join().unwrap(), 1, "the words were let go before the publish in flight finished");
    assert!(!t.pi.publishes_to(0));
    // Later publishes find no word.
    t.pi.int_modify(5);
    t.pi.int_modify(MODIFY_SET | 5);
    assert_eq!(t.pi.rises[0].load(Ordering::SeqCst), 1);
}

/// A reset writes the lines with the registers it clears.
#[test]
fn a_reset_drops_the_lines_that_were_up() {
    let t = Fixture::new(1);
    t.w(reg::PI_INT_MASK0_A::OFFSET, !0);
    t.w(reg::PI_RT_EN_A::OFFSET, 1);
    t.pi.int_modify(MODIFY_SET | 5);
    t.w(reg::PI_RT_PEND_A::OFFSET, 1);
    assert!(t.lines.get(0) && t.lines.get(2));
    Resettable::power_on(&*t.pi);
    assert!(!t.lines.get(0) && !t.lines.get(2), "the lines stayed up through the reset");
    let falls = |l: usize| t.pi.falls[l].load(Ordering::SeqCst);
    assert_eq!((falls(0), falls(2)), (1, 1));
}

/// A failed uncached load sets its processor's "uncached uncorrectable" bit of the error pending
/// register (bit 21 for A.
#[test]
fn a_failed_read_sets_the_error_pending_bit_a_store_of_ones_clears() {
    let t = Fixture::new(2);
    assert_eq!(t.r(reg::PI_ERR_INT_PEND::OFFSET), 0);
    t.pi.read_error(0, 0x240_0208);
    assert_eq!(t.r(reg::PI_ERR_INT_PEND::OFFSET), 1 << 21);
    t.pi.read_error(1, 0x240_0208);
    assert_eq!(t.r(reg::PI_ERR_INT_PEND::OFFSET), 0x30_0000);
    // Writing zero, or the other processor's bit, clears nothing of A's.
    t.pi.write_sized(reg::PI_ERR_INT_PEND::OFFSET, 0, 8);
    t.pi.write_sized(reg::PI_ERR_INT_PEND::OFFSET, 1 << 20, 8);
    assert_eq!(t.r(reg::PI_ERR_INT_PEND::OFFSET), 1 << 21);
    // The status record is a separate latch and survives the clear.
    assert_ne!(t.r(reg::PI_ERR_STATUS0_A::OFFSET), 0);
    t.pi.write_sized(reg::PI_ERR_INT_PEND::OFFSET, 0x30_0000, 8);
    assert_eq!(t.r(reg::PI_ERR_INT_PEND::OFFSET), 0);
    assert_ne!(t.r(reg::PI_ERR_STATUS0_A::OFFSET), 0);
    // Bit 24 is the directory unit's; a store cannot touch it.
    t.pi.write_sized(reg::PI_ERR_INT_PEND::OFFSET, 1 << 24, 8);
    assert_eq!(t.r(reg::PI_ERR_INT_PEND::OFFSET), 0);
    // Reset clears it.
    t.pi.read_error(0, 8);
    t.pi.power_on();
    assert_eq!(t.r(reg::PI_ERR_INT_PEND::OFFSET), 0);
}
