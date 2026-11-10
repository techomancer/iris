// Cases follow tests/qtest/sn0-hub-test.c, sn0-pi-error-test.c and
// sn0-pi-gfx-test.c in jamesbraid/qemu-sgi-origami: reset-values,
// guest-writes, processor-number, cpu-enable, interrupt-modify,
// interrupt-number-selects-bank, masked-pending-raises-nothing,
// pending-banks-masked-per-processor, pending-banks-reject-writes, the rt-*
// counter and comparator cases, pi-rt-local-ctrl, pi-error-register-shell,
// pi-timedout-buffers, the graphics-flow-control register cases and the
// processor-error status cases.

use super::*;
use std::cell::RefCell;

struct Fixture {
    pi: Arc<Pi>,
    now: Arc<AtomicU64>,
    lines: Arc<Lines>,
    /// What the writes so far asked the Hub to do, as text.
    effects: Mutex<Vec<String>>,
}
/// The processors' interrupt words, as a plain word each: line 0 to 2 are
/// slice A's Cause bits 10 to 12 and 3 to 5 slice B's.
#[derive(Default)]
struct Lines([AtomicU64; 2]);
impl Lines {
    fn get(&self, line: usize) -> bool {
        self.0[line / LINES_PER_CPU].load(Ordering::SeqCst) >> (CAUSE_SHIFT as usize + line % LINES_PER_CPU) & 1 == 1
    }
}
impl Fixture {
    fn new(cpus: u8) -> Self {
        let now = Arc::new(AtomicU64::new(1_000_000));
        let clock = now.clone();
        let pi = Pi::new(cpus, 0, Arc::new(move || clock.load(Ordering::SeqCst)), Routing::new());
        let lines = Arc::new(Lines::default());
        for (slice, word) in lines.0.iter().enumerate() {
            pi.set_cpu_interrupts(slice, word);
        }
        Self { pi, now, lines, effects: Mutex::default() }
    }
    fn r(&self, off: u64) -> u64 {
        self.pi.read64(off).data
    }
    fn w(&self, off: u64, value: u64) {
        let text = match self.pi.write(off, value) {
            Effect::None => return,
            Effect::CachedAlias => vec![format!("calias {:#x}", self.pi.calias_bytes())],
            other => panic!("the processor interface does not return {other:?}"),
        };
        self.effects.lock().extend(text);
    }
    /// Let `ticks` counter ticks pass and bring the comparators up to date.
    fn advance(&self, ticks: u64) {
        self.now.fetch_add(ticks * TICK_NS, Ordering::SeqCst);
        self.pi.poll();
    }
    /// Whether slice `slice`'s word carries the soft-reset request.
    fn reset_requested(&self, slice: usize) -> bool {
        self.lines.0[slice].load(Ordering::SeqCst) >> 63 == 1
    }
    fn take_effects(&self) -> Vec<String> {
        std::mem::take(&mut *self.effects.lock())
    }
}

/// An unprogrammed Hub claims every region exists, resets its cached alias to
/// the largest size, and keeps the registers firmware uses as markers.
#[test]
fn the_pi_resets_and_stores_what_the_guest_writes() {
    let t = Fixture::new(2);
    assert_eq!(t.r(reg::REGION_PRESENT), !0);
    assert_eq!(t.r(reg::CALIAS_SIZE), 15);
    assert_eq!((t.r(reg::INT_PEND0), t.r(reg::INT_PEND1)), (0, 0));
    t.w(reg::REGION_PRESENT, 0x1234);
    assert_eq!(t.r(reg::REGION_PRESENT), 0x1234);
    t.w(reg::INT_MASK0_A, !0);
    t.w(reg::INT_MASK1_B, 0x55);
    assert_eq!((t.r(reg::INT_MASK0_A), t.r(reg::INT_MASK1_B)), (!0, 0x55));
    // The error stack address keeps bits 31:3 and survives a reset.
    t.w(reg::ERR_STACK_ADDR_A, !0);
    t.w(reg::ERR_STACK_ADDR_B, 0x1_2345_6789);
    assert_eq!(t.r(reg::ERR_STACK_ADDR_A), 0xffff_fff8);
    assert_eq!(t.r(reg::ERR_STACK_ADDR_B), 0x2345_6788);
    t.pi.power_on();
    assert_eq!(t.r(reg::REGION_PRESENT), !0);
    assert_eq!(t.r(reg::INT_MASK0_A), 0);
    assert_eq!(t.r(reg::ERR_STACK_ADDR_A), 0xffff_fff8);
    assert_eq!(t.r(reg::ERR_STACK_ADDR_B), 0x2345_6788);
}

#[test]
fn the_processor_number_names_the_reading_processor() {
    let t = Fixture::new(2);
    let pi = t.pi.clone();
    std::thread::spawn(move || {
        crate::cpu::smp::set_current_cpu(1);
        assert_eq!(pi.read64(reg::CPU_NUM).data, 1);
        crate::cpu::smp::set_current_cpu(0);
        assert_eq!(pi.read64(reg::CPU_NUM).data, 0);
    })
    .join()
    .unwrap();
    // A device thread counts as processor 0, and a store changes nothing.
    t.w(reg::CPU_NUM, 1);
    assert_eq!(t.r(reg::CPU_NUM), 0);
    // A node whose slice A is global CPU 2.
    let node = Pi::new(2, 2, Arc::new(|| 0), Routing::new());
    std::thread::spawn(move || {
        crate::cpu::smp::set_current_cpu(3);
        assert_eq!(node.read64(reg::CPU_NUM).data, 1);
        crate::cpu::smp::set_current_cpu(0);
        assert_eq!(node.read64(reg::CPU_NUM).data, 0);
    })
    .join()
    .unwrap();
}

#[test]
fn processor_presence_enable_nmi_and_soft_reset() {
    let t = Fixture::new(2);
    assert_eq!((t.r(reg::CPU_PRESENT_A), t.r(reg::CPU_PRESENT_B)), (1, 1));
    assert_eq!((t.r(reg::CPU_ENABLE_A), t.r(reg::CPU_ENABLE_B)), (0, 0));
    t.w(reg::CPU_ENABLE_B, 0xff);
    assert_eq!(t.r(reg::CPU_ENABLE_B), 1);
    // The enable and the NMI are logged and ask nothing of the machine.
    assert!(t.take_effects().is_empty());
    assert_eq!((t.reset_requested(0), t.reset_requested(1)), (false, false));
    // The soft reset reaches only the enabled processors, as bit 63 of the
    // processor's word, and asks nothing else.
    t.w(reg::SOFTRESET, 1);
    assert_eq!((t.reset_requested(0), t.reset_requested(1)), (false, true));
    assert!(t.take_effects().is_empty());
    t.w(reg::NMI_A, 1);
    t.w(reg::NMI_B, 0);
    assert!(t.take_effects().is_empty());
    assert_eq!(t.lines.0[0].load(Ordering::SeqCst), 0, "NMI does not reach the processor");
    // Reset halts both processors again; the enables read zero.
    t.pi.power_on();
    assert_eq!(t.r(reg::CPU_ENABLE_B), 0);
}

#[test]
fn an_absent_processors_enable_stores_and_reaches_nothing() {
    let t = Fixture::new(1);
    assert_eq!((t.r(reg::CPU_PRESENT_A), t.r(reg::CPU_PRESENT_B)), (1, 0));
    t.w(reg::CPU_ENABLE_B, 1);
    assert_eq!(t.r(reg::CPU_ENABLE_B), 1);
    t.w(reg::NMI_B, 1);
    t.w(reg::SOFTRESET, 1);
    assert!(t.take_effects().is_empty());
    assert_eq!((t.reset_requested(0), t.reset_requested(1)), (false, false));
    t.w(reg::CPU_ENABLE_A, 1);
    t.w(reg::SOFTRESET, 1);
    assert_eq!((t.reset_requested(0), t.reset_requested(1)), (true, false));
    // Nor does the absent slot get an interrupt line.
    t.w(reg::INT_MASK0_B, !0);
    t.w(reg::INT_PEND_MOD, MODIFY_SET | 3);
    assert!(!t.lines.get(3));
}

#[test]
fn the_cached_alias_size_codes_double_from_4_kib() {
    let t = Fixture::new(2);
    for (code, bytes) in [(0, 0), (1, 4 << 10), (2, 8 << 10), (15, 64 << 20)] {
        t.w(reg::CALIAS_SIZE, code);
        assert_eq!(t.take_effects(), [format!("calias {bytes:#x}")]);
        assert_eq!(t.pi.calias_bytes(), bytes);
    }
    // Four bits store; a wider value wraps into them.
    t.w(reg::CALIAS_SIZE, 0x31);
    assert_eq!(t.r(reg::CALIAS_SIZE), 1);
}

#[test]
fn interrupt_modify_sets_and_clears_one_bit_of_one_bank() {
    let t = Fixture::new(2);
    t.w(reg::INT_PEND_MOD, MODIFY_SET | 12);
    assert_eq!((t.r(reg::INT_PEND0), t.r(reg::INT_PEND1)), (1 << 12, 0));
    t.w(reg::INT_PEND_MOD, 12);
    assert_eq!(t.r(reg::INT_PEND0), 0);
    // Number 70 is bit 6 of the second bank.
    t.w(reg::INT_PEND_MOD, MODIFY_SET | 70);
    assert_eq!((t.r(reg::INT_PEND0), t.r(reg::INT_PEND1)), (0, 1 << 6));
    // The modify register is an instruction: it reads zero, and the pending
    // banks refuse writes.
    assert_eq!(t.r(reg::INT_PEND_MOD), 0);
    t.w(reg::INT_PEND0, !0);
    t.w(reg::INT_PEND1, !0);
    assert_eq!((t.r(reg::INT_PEND0), t.r(reg::INT_PEND1)), (0, 1 << 6));
}

#[test]
fn pending_bits_raise_a_line_only_through_a_mask_and_follow_it_down() {
    let t = Fixture::new(2);
    // Masked pending is invisible; unmasking raises the line with nothing
    // written to the pending bank, which separates recomputing from
    // raise-on-set.
    t.w(reg::INT_PEND_MOD, MODIFY_SET | 12);
    assert!(!t.lines.get(0));
    t.w(reg::INT_MASK0_A, 1 << 12);
    assert!(t.lines.get(0));
    // Level-triggered: clearing the last visible bit drops it.
    t.w(reg::INT_PEND_MOD, 12);
    assert!(!t.lines.get(0));
    // Bank 1 has its own line: number 70 does not raise bank 0's.
    t.w(reg::INT_MASK0_A, !0);
    t.w(reg::INT_MASK1_A, !0);
    t.w(reg::INT_PEND_MOD, MODIFY_SET | 70);
    assert!(!t.lines.get(0) && t.lines.get(1));
}

#[test]
fn the_two_processors_have_their_own_masks() {
    let t = Fixture::new(2);
    t.w(reg::INT_MASK0_A, 0);
    t.w(reg::INT_MASK0_B, 1 << 12);
    t.w(reg::INT_PEND_MOD, MODIFY_SET | 12);
    assert!(!t.lines.get(0) && t.lines.get(3));
    t.w(reg::INT_PEND_MOD, 12);
    assert!(!t.lines.get(3));
    // The PROM's order for B: pend while masked, then unmask.
    t.w(reg::INT_MASK0_B, 0);
    t.w(reg::INT_PEND_MOD, MODIFY_SET | 12);
    assert!(!t.lines.get(3));
    t.w(reg::INT_MASK0_B, 1 << 12);
    assert!(t.lines.get(3));
    // B's second bank must not follow the first bank's pending bit.
    t.w(reg::INT_MASK1_B, !0);
    assert!(!t.lines.get(4));
}

#[test]
fn the_counter_runs_at_1_25_mhz_in_52_bits_and_can_be_written() {
    let t = Fixture::new(1);
    t.w(reg::RT_COUNT, 0);
    assert_eq!(t.r(reg::RT_COUNT), 0);
    // 800 ns is one tick, and the counter moves between two reads.
    t.now.fetch_add(799, Ordering::SeqCst);
    assert_eq!(t.r(reg::RT_COUNT), 0);
    t.now.fetch_add(1, Ordering::SeqCst);
    assert_eq!(t.r(reg::RT_COUNT), 1);
    // A whole second is 1,250,000 ticks.
    t.advance(1_250_000 - 1);
    assert_eq!(t.r(reg::RT_COUNT), 1_250_000);
    // A write moves the origin and the counter keeps running from it.
    t.w(reg::RT_COUNT, 0xabcde_f012_3456);
    assert_eq!(t.r(reg::RT_COUNT), 0xabcde_f012_3456);
    t.advance(10);
    assert_eq!(t.r(reg::RT_COUNT), 0xabcde_f012_3460);
    // Fifty-two bits: the top of the range wraps to zero.
    t.w(reg::RT_COUNT, RT_MASK);
    t.advance(1);
    assert_eq!(t.r(reg::RT_COUNT), 0);
    t.w(reg::RT_COUNT, !0);
    assert_eq!(t.r(reg::RT_COUNT), RT_MASK);
    // Hours of elapsed time stay exact.
    t.w(reg::RT_COUNT, 0);
    t.advance(1_250_000 * 3600 * 5);
    assert_eq!(t.r(reg::RT_COUNT), 1_250_000 * 3600 * 5);
}

#[test]
fn a_comparator_pends_when_the_counter_reaches_it_and_the_enable_gates_the_line() {
    let t = Fixture::new(2);
    t.w(reg::RT_COUNT, 100);
    t.w(reg::RT_COMPARE_A, 150);
    t.advance(49);
    assert_eq!((t.r(reg::RT_PEND_A), t.lines.get(2)), (0, false));
    // A match pends whether or not enabled.
    t.advance(1);
    assert_eq!(t.r(reg::RT_PEND_A), 1);
    assert!(!t.lines.get(2));
    t.w(reg::RT_EN_A, 1);
    assert!(t.lines.get(2));
    // Writing zero acknowledges and drops the line.
    t.w(reg::RT_PEND_A, 0);
    assert!(!t.lines.get(2));
    // A match fires once; nothing re-arms it until software writes again.
    t.advance(10_000);
    assert_eq!(t.r(reg::RT_PEND_A), 0);
    t.w(reg::RT_COMPARE_A, 10_200);
    t.advance(49);
    assert_eq!(t.r(reg::RT_PEND_A), 0);
    t.advance(1);
    assert_eq!(t.r(reg::RT_PEND_A), 1);
    // The second processor has its own comparator and line.
    t.w(reg::RT_EN_B, 1);
    t.w(reg::RT_COMPARE_B, t.r(reg::RT_COUNT) + 5);
    t.advance(5);
    assert!(t.lines.get(5));
    assert_eq!(t.r(reg::RT_PEND_B), 1);
}

#[test]
fn a_comparator_behind_the_counter_waits_a_whole_revolution() {
    let t = Fixture::new(1);
    t.w(reg::RT_EN_A, 1);
    // The guest kernel parks a processor's tick by writing a value behind the counter
    // and depends on the silence.
    t.w(reg::RT_COUNT, 1000);
    t.w(reg::RT_COMPARE_A, 0);
    t.advance(500_000);
    assert_eq!((t.r(reg::RT_PEND_A), t.lines.get(2)), (0, false));
    // Equal to the counter is a revolution away too, not a match now.
    t.w(reg::RT_COMPARE_A, t.r(reg::RT_COUNT));
    assert_eq!(t.r(reg::RT_PEND_A), 0);
    // The counter wrapping round reaches the zeroed comparator.
    t.w(reg::RT_COMPARE_A, 0);
    t.w(reg::RT_COUNT, RT_MASK - 2);
    t.advance(2);
    assert_eq!(t.r(reg::RT_PEND_A), 0);
    t.advance(1);
    assert_eq!(t.r(reg::RT_PEND_A), 1);
}

#[test]
fn moving_the_counter_pends_nothing_but_rearms_the_deadline() {
    let t = Fixture::new(1);
    t.w(reg::RT_COUNT, 0);
    t.w(reg::RT_COMPARE_A, 100);
    // Jumping the counter past the compare value is not a match.
    t.w(reg::RT_COUNT, 500);
    t.advance(1000);
    assert_eq!(t.r(reg::RT_PEND_A), 0);
    // Jumping to just before it re-arms against the new counter. This is
    // Linux's guarded rewrite: read, add a delta, write, re-read.
    t.w(reg::RT_COUNT, 90);
    t.advance(9);
    assert_eq!(t.r(reg::RT_PEND_A), 0);
    t.advance(1);
    assert_eq!(t.r(reg::RT_PEND_A), 1);
    // The longest interval is a revolution minus a tick.
    t.w(reg::RT_PEND_A, 0);
    t.w(reg::RT_COUNT, 0);
    t.w(reg::RT_COMPARE_A, RT_MASK);
    t.advance(RT_MASK - 1);
    assert_eq!(t.r(reg::RT_PEND_A), 0);
    t.advance(1);
    assert_eq!(t.r(reg::RT_PEND_A), 1);
}

#[test]
fn the_local_control_register_stores_bit_16_and_the_low_byte() {
    let t = Fixture::new(1);
    assert_eq!(t.r(reg::RT_LOCAL_CTRL), 0x1_0000);
    t.w(reg::RT_LOCAL_CTRL, 0x1_0047);
    assert_eq!(t.r(reg::RT_LOCAL_CTRL), 0x1_0047);
    t.w(reg::RT_LOCAL_CTRL, !0);
    assert_eq!(t.r(reg::RT_LOCAL_CTRL), 0x1_00ff);
    // Nothing written there changes the rate.
    t.w(reg::RT_COUNT, 0);
    t.advance(10);
    assert_eq!(t.r(reg::RT_COUNT), 10);
    t.pi.power_on();
    assert_eq!(t.r(reg::RT_LOCAL_CTRL), 0x1_0000);
}

#[test]
fn the_error_registers_store_masked_values_and_nothing_times_out() {
    let t = Fixture::new(2);
    assert_eq!(t.r(reg::ERR_INT_PEND), 0);
    t.w(reg::ERR_INT_PEND, !0);
    assert_eq!(t.r(reg::ERR_INT_PEND), 0);
    t.w(reg::ERR_INT_MASK_A, !0);
    t.w(reg::ERR_INT_MASK_B, 0x0123456);
    assert_eq!((t.r(reg::ERR_INT_MASK_A), t.r(reg::ERR_INT_MASK_B)), (0x01ff_ffff, 0x0123456));
    t.w(reg::ERR_STACK_SIZE, !0);
    assert_eq!(t.r(reg::ERR_STACK_SIZE), 0xf);
    for off in [reg::TIMEDOUT_A, reg::TIMEDOUT_B] {
        t.w(off, !0);
        assert_eq!(t.r(off), 0);
    }
    t.pi.power_on();
    assert_eq!((t.r(reg::ERR_INT_MASK_A), t.r(reg::ERR_INT_MASK_B), t.r(reg::ERR_STACK_SIZE)), (0, 0, 0));
    for off in (reg::ERR_STATUS0_A..=reg::ERR_STATUS1_B_RCLR).step_by(8) {
        assert_eq!(t.r(off), 0, "{off:#x}");
    }
}

#[test]
fn a_failed_read_latches_the_first_address_and_later_ones_only_set_overrun() {
    let t = Fixture::new(2);
    // The PROM's probe: a failed doubleword read at 0x0240_0208 of node 0.
    t.pi.read_error(1, 0x0240_0208);
    let first = STATUS0_VALID | ((0x0240_0208u64 >> 3) << 25);
    assert_eq!(t.r(reg::ERR_STATUS0_B), first);
    assert_eq!(t.r(reg::ERR_STATUS0_A), 0);
    t.pi.read_error(1, 0x1234_5678);
    assert_eq!(t.r(reg::ERR_STATUS0_B), first | STATUS0_OVERRUN);
    // The kernel checks bits 61:25 and a 40-bit address; the type fields stay zero.
    t.pi.read_error(0, 0xff_0000_0008);
    assert_eq!(t.r(reg::ERR_STATUS0_A), STATUS0_VALID | ((0xff_0000_0008u64 >> 3) << 25));
    // A slice the node does not fit records nothing.
    Fixture::new(1).pi.read_error(1, 8);
}

#[test]
fn a_read_to_clear_alias_clears_only_on_a_full_width_read() {
    let t = Fixture::new(1);
    t.pi.read_error(0, 0x240_0208);
    let value = t.r(reg::ERR_STATUS0_A);
    // The 32-bit half path reads without clearing.
    assert_eq!(t.pi.read(reg::ERR_STATUS0_A_RCLR, false), value);
    assert_eq!(t.r(reg::ERR_STATUS0_A), value);
    assert_eq!(t.r(reg::ERR_STATUS0_A_RCLR), value);
    assert_eq!(t.r(reg::ERR_STATUS0_A), 0);
}

#[test]
fn a_full_width_zero_store_to_either_alias_ends_the_record() {
    let t = Fixture::new(2);
    for alias in [reg::ERR_STATUS0_A_RCLR, reg::ERR_STATUS1_A_RCLR] {
        t.pi.read_error(0, 0x240_0208);
        // A narrow or nonzero store does not clear.
        t.pi.write_sized(alias, 0, 4);
        t.pi.write_sized(alias, 1, 8);
        assert_ne!(t.r(reg::ERR_STATUS0_A), 0);
        t.pi.write_sized(alias, 0, 8);
        assert_eq!(t.r(reg::ERR_STATUS0_A), 0);
    }
    // B's aliases clear B alone.
    t.pi.read_error(0, 8);
    t.pi.read_error(1, 8);
    t.pi.write_sized(reg::ERR_STATUS1_B_RCLR, 0, 8);
    assert_eq!((t.r(reg::ERR_STATUS0_B), t.r(reg::ERR_STATUS0_A) != 0), (0, true));
    // Reset clears both slices.
    t.pi.read_error(1, 8);
    t.pi.power_on();
    assert_eq!((t.r(reg::ERR_STATUS0_A), t.r(reg::ERR_STATUS0_B)), (0, 0));
}

#[test]
fn the_graphics_flow_control_registers_keep_their_field_widths() {
    let t = Fixture::new(2);
    const PAGE_MASK: u64 = (1 << 40) | 0xff_fffe_0000;
    let slot = |a: u64| reg::GFX_BASE + a * 8 * 5;
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
    t.w(reg::GFX_BASE + 8, 0x65);
    t.w(slot(1) + 8, 0xab);
    assert_eq!((t.r(reg::GFX_BASE + 8), t.r(slot(1) + 8)), (0x65, 0xab));
    // The block shares nothing with the comparator.
    t.w(reg::RT_COMPARE_A, 0x1_2345_6789_abcd);
    t.w(reg::GFX_BASE, !0);
    assert_eq!(t.r(reg::RT_COMPARE_A), 0x1_2345_6789_abcd);
    // All ten reset to zero.
    t.pi.power_on();
    for n in 0..10 {
        assert_eq!(t.r(reg::GFX_BASE + n * 8), 0);
    }
}

/// The guest kernel attaches a pipe by writing its page with the enable set, the saved
/// count, and then the bias. No store is ever counted: the credit stays as
/// written.
#[test]
fn the_graphics_attach_sequence_leaves_the_credit_as_written() {
    let t = Fixture::new(2);
    let (page, credit, bias) = (reg::GFX_BASE, reg::GFX_BASE + 8, reg::GFX_BASE + 16);
    t.w(bias, 0);
    t.w(credit, 0x65);
    t.w(page, 0x1234_0000 | (1 << 40));
    t.w(bias, 0x44);
    assert_eq!((t.r(credit), t.r(bias), t.r(page)), (0x65, 0x44, 0x1234_0000 | (1 << 40)));
    assert!(t.r(credit) >= t.r(bias));
    // No pending bit follows from the bias: the interrupt is not modelled.
    assert_eq!(t.r(reg::INT_PEND0), 0);
}

#[test]
fn the_hard_reset_bit_takes_a_zero_store_and_reads_zero() {
    let t = Fixture::new(1);
    t.w(reg::HARDRESET_BIT, 0);
    assert_eq!(t.r(reg::HARDRESET_BIT), 0);
    t.w(reg::HARDRESET_BIT, 1);
    assert_eq!(t.r(reg::HARDRESET_BIT), 0);
}

#[test]
fn unimplemented_offsets_read_zero() {
    let t = Fixture::new(1);
    assert_eq!(t.r(0x1f8), 0);
    t.w(0x1f8, 0x5555);
    assert_eq!(t.r(0x1f8), 0);
}

/// A write that changes no line writes no edge: the lines are driven only on a
/// change.
#[test]
fn a_write_that_changes_no_line_makes_no_edge() {
    let t = Fixture::new(1);
    t.w(reg::INT_MASK0_A, !0);
    t.pi.int_modify(MODIFY_SET | 5);
    assert!(t.lines.get(0));
    let edges = |t: &Fixture| (t.pi.rises[0].load(Ordering::SeqCst), t.pi.falls[0].load(Ordering::SeqCst));
    assert_eq!(edges(&t), (1, 0));
    // The bit again, the mask again, a drive: the line stays up.
    t.pi.int_modify(MODIFY_SET | 5);
    t.pi.int_modify(MODIFY_SET | 6);
    t.w(reg::INT_MASK0_A, !0);
    t.pi.drive();
    t.pi.poll();
    assert_eq!(edges(&t), (1, 0));
    // One bit clears and the other holds the line.
    t.pi.int_modify(5);
    assert_eq!(edges(&t), (1, 0));
    t.pi.int_modify(6);
    assert!(!t.lines.get(0));
    assert_eq!(edges(&t), (1, 1));
}

/// Four threads change three inputs and two others write the same values and
/// drive the lines over and over. A line rises exactly once for each pulse of
/// its input and falls once, and a line nobody pulsed never moves: no stale
/// sample writes an edge the registers did not make.
#[test]
fn threads_pulsing_and_driving_make_one_rise_and_one_fall_per_pulse() {
    const PULSES: u64 = 4000;
    let t = Arc::new(Fixture::new(1));
    t.w(reg::INT_MASK0_A, !0);
    t.w(reg::RT_EN_A, 1);
    let stop = Arc::new(AtomicBool::new(false));
    let noisy: Vec<_> = (0..2)
        .map(|n| {
            let (t, stop) = (t.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    t.pi.drive();
                    // The same values the registers hold.
                    t.pi.write(reg::INT_MASK0_A, !0);
                    t.pi.write(reg::RT_EN_A, 1);
                    if n == 0 {
                        t.pi.write(reg::INT_MASK1_A, 0);
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
                t.pi.write(reg::RT_PEND_A, 1);
                t.pi.write(reg::RT_PEND_A, 0);
            }
        })
    };
    bank.join().unwrap();
    comparator.join().unwrap();
    stop.store(true, Ordering::SeqCst);
    for n in noisy {
        n.join().unwrap();
    }
    let count = |a: &[AtomicU64; 6]| [0, 1, 2].map(|l| a[l].load(Ordering::SeqCst));
    assert_eq!(count(&t.pi.rises), [PULSES, 0, PULSES]);
    assert_eq!(count(&t.pi.falls), [PULSES, 0, PULSES]);
    assert_eq!((t.lines.get(0), t.lines.get(1), t.lines.get(2)), (false, false, false));
}

/// Writers that race an input against a mask and an enable leave the word as
/// the registers say, whatever the interleaving.
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
                b.pi.write(reg::INT_MASK0_A, if i % 2 == 0 { !0 } else { 0 });
            }
        });
        let enabler = std::thread::spawn(move || {
            for i in 0..2000u64 {
                c.pi.write(reg::RT_PEND_A, 1);
                c.pi.write(reg::RT_EN_A, i & 1);
            }
        });
        for h in [setter, masker, enabler] {
            h.join().unwrap();
        }
        let bank0 = t.r(reg::INT_PEND0) & t.r(reg::INT_MASK0_A) != 0;
        let rt = t.r(reg::RT_PEND_A) != 0 && t.r(reg::RT_EN_A) != 0;
        assert_eq!((t.lines.get(0), t.lines.get(1), t.lines.get(2)), (bank0, false, rt), "round {round}");
    }
}

/// A writer that stalls in the middle of writing the lines holds off every
/// change of their inputs: the input changes under the same hold of `lines`
/// as the write. Were an input changed before the lock, the second thread
/// would clear and set the bit while the first still held the stale sample,
/// and the first's rise would land after the second's fall and rise, one
/// edge where the registers made two.
#[test]
fn a_stalled_line_write_holds_off_changes_to_its_input() {
    use std::sync::Barrier;
    let t = Arc::new(Fixture::new(1));
    t.w(reg::INT_MASK0_A, !0);
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
    let second = {
        let t = t.clone();
        std::thread::spawn(move || {
            t.pi.int_modify(5);
            t.pi.int_modify(MODIFY_SET | 5);
        })
    };
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(t.pi.pending(0), 1 << 5, "the input changed while another writer held the lines");
    assert!(!second.is_finished());
    release.wait();
    first.join().unwrap();
    second.join().unwrap();
    assert!(t.lines.get(0));
    // The bit rose, fell and rose: three edges in the order the registers made them.
    assert_eq!((t.pi.rises[0].load(Ordering::SeqCst), t.pi.falls[0].load(Ordering::SeqCst)), (2, 1));
}

/// A reset writes the lines with the registers it clears: a line that was up
/// falls, once, with the reset and not at the next change of an input.
#[test]
fn a_reset_drops_the_lines_that_were_up() {
    let t = Fixture::new(1);
    t.w(reg::INT_MASK0_A, !0);
    t.w(reg::RT_EN_A, 1);
    t.pi.int_modify(MODIFY_SET | 5);
    t.w(reg::RT_PEND_A, 1);
    assert!(t.lines.get(0) && t.lines.get(2));
    Resettable::power_on(&*t.pi);
    assert!(!t.lines.get(0) && !t.lines.get(2), "the lines stayed up through the reset");
    let falls = |l: usize| t.pi.falls[l].load(Ordering::SeqCst);
    assert_eq!((falls(0), falls(2)), (1, 1));
}
