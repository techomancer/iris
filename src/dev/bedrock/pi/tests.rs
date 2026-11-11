// Cases follow tests/qtest/sn1-bedrock-pi-test.c in jamesbraid/qemu-sgi-origami:
// power-on-values, local-reset, pending-reflection, real-time-comparators,
// profile-counter-wrap, cpu-number-fallback, cpu-enable-lines and nmi.

use super::super::reg::PI1_BASE;
use super::super::Rig;
use super::*;
use crate::traits::{BusDevice, Device, BUS_ERR, BUS_OK};
use std::sync::atomic::Ordering;

/// Drive the reflected bank 1 bit 55, as LB's error level does.
fn reflect(rig: &Rig, level: bool) {
    if rig.chip.levels.set((1, 55), level) {
        rig.chip.publish_all();
    }
}
use std::sync::Arc;
use std::time::Duration;

/// The PROM reads PI slot presence and relies on the documented reset survivors.
#[test]
fn the_pi_reports_interface_state_and_retains_registers_across_reset() {
    {
        let rig = Rig::new();
        assert_eq!(rig.r(reg::INTERFACE_ID), 0);
        assert_eq!(rig.r(reg::INTERFACE_SLOT_A), 1);
        assert_eq!(rig.r(reg::INTERFACE_SLOT_B), 0);
        rig.w(reg::NODE_REGION, 0);
        rig.w(reg::PROTECTION_KEY, 0x53474972756c6573);
        assert_eq!(rig.r(reg::NODE_REGION), 1);
        assert_eq!(rig.r(PI1_BASE + reg::INTERFACE_ID), 2);
        assert_eq!(rig.r(PI1_BASE + reg::INTERFACE_SLOT_A), 0);
        assert_eq!(rig.regs.read32(reg::INTERFACE_ID).status, BUS_ERR);
        assert_eq!(rig.regs.write32(reg::COUNTER, 0), BUS_ERR);
    }
    {
        let rig = Rig::new();
        rig.w(reg::PENDING_MODIFY, 0x10c);
        assert_eq!(rig.r(reg::PENDING_STATUS_A), 1 << 12);
        rig.w(reg::PENDING_MODIFY_B, 12);
        assert_eq!(rig.r(reg::PENDING_STATUS_A), 0);
        reflect(&rig, true);
        rig.w(reg::PENDING_MODIFY, 64 + 55);
        assert_eq!(rig.r(reg::PENDING_STATUS_B), 1 << 55);
        rig.w(reg::PENDING_MODIFY, 0x100 | 64 | 63);
        assert_eq!(rig.r(reg::PENDING_STATUS_B), 1 << 55);
        rig.w(reg::REQUEST_A, 0);
        rig.w(reg::OFFSET_00E8, 1);
        assert_eq!(rig.r(reg::PENDING_STATUS_A), 1 << 5);
        rig.w(reg::PENDING_A_CLEAR, 1);
        assert_eq!(rig.r(reg::PENDING_STATUS_A), 0);
    }
    {
        let rig = Rig::new();
        rig.w(reg::CACHED_ALIAS, 9);
        rig.w(reg::CPU_A_ENABLE, 1);
        rig.w(reg::ERROR_ADDRESS_A, !0);
        rig.chip.pi[0].error_summary.fetch_or(MEMORY_ERROR, Ordering::SeqCst);
        rig.w(reg::ERROR_SUMMARY_ALIAS, 3);
        rig.w(reg::RT_COMPARE_A, 0x24);
        rig.chip.pi[0].power_on(0);
        assert_eq!(rig.r(reg::ERROR_SUMMARY), MEMORY_ERROR | 3);
        assert_eq!(rig.r(reg::CACHED_ALIAS), 9);
        assert_eq!(rig.r(reg::CPU_A_ENABLE), 1);
        assert_eq!(rig.r(reg::ERROR_ADDRESS_A), 0x1fffffff8);
        assert_eq!(rig.r(reg::RT_COMPARE_A), 0);
        assert_eq!(rig.r(reg::COUNTER), 0);
        rig.w(reg::ERROR_SUMMARY_ALIAS, 0);
        assert_eq!(rig.r(reg::ERROR_SUMMARY), MEMORY_ERROR);
        rig.w(reg::ERROR_SUMMARY, MEMORY_ERROR);
        assert_eq!(rig.r(reg::ERROR_SUMMARY), 0);
    }
}

/// The counter wraps, the comparators latch at their tick, and the counter
/// pauses while the machine is stopped.
#[test]
fn the_pi_timer_wraps_and_pauses() {
    {
        let rig = Rig::new();
        rig.w(reg::COUNTER, 0x1_0000_0000);
        rig.w(reg::RT_COMPARE_A, 0x1_0000_0002);
        rig.set_time(1599);
        assert_eq!(rig.r(reg::RT_STATUS_A), 0);
        rig.set_time(1600);
        assert_eq!(rig.r(reg::RT_STATUS_A), 1);
        assert_eq!(rig.regs.read32(reg::COUNTER).data, 1);
        assert_eq!(rig.regs.read32(reg::COUNTER + 4).data, 2);
        rig.w(reg::RT_STATUS_A, !0);
        assert_eq!(rig.r(reg::RT_STATUS_A), 0);
        rig.w(reg::COUNTER, RT_MASK);
        rig.w(reg::PROFILE_COMPARE, 0);
        rig.set_time(2400);
        assert_eq!(rig.r(reg::COUNTER), 0);
        assert_eq!(rig.r(reg::PROFILE_STATUS_A), 1);
        assert_eq!(rig.r(reg::PROFILE_STATUS_B), 1);
        rig.w(reg::PROFILE_STATUS_A, 0);
        assert_eq!(rig.r(reg::PROFILE_STATUS_A), 0);
        assert_eq!(rig.r(reg::PROFILE_STATUS_B), 1);
    }
    {
        let rig = Rig::new();
        rig.set_time(1_000_000);
        rig.w(reg::COUNTER, 0);
        rig.w(reg::RT_COMPARE_A, 2);
        rig.set_time(1_000_400);
        rig.chip.stop();
        rig.set_time(2_000_000);
        assert_eq!(rig.r(reg::COUNTER), 0);
        assert_eq!(rig.r(reg::RT_STATUS_A), 0, "no time passes while stopped");
        rig.chip.start();
        rig.set_time(2_001_199);
        assert_eq!(rig.r(reg::RT_STATUS_A), 0);
        rig.set_time(2_001_200);
        assert_eq!(rig.r(reg::RT_STATUS_A), 1);
    }
}

/// Each of a slot's five outputs follows its own inputs into Cause bits
/// 10-14 of the processor's word, and nothing else in the word moves.
#[test]
fn pi_interrupt_outputs_follow_masks_into_the_cpus_word() {
    let rig = Rig::new();
    rig.word.store(0xffff_0000_0000_83ff & !(0x1f << 10), Ordering::SeqCst);
    rig.w(reg::OFFSET_00A8, 1 << 12);
    rig.w(reg::OFFSET_00B0, 1 << 55);
    rig.w(reg::PENDING_MODIFY, 0x10c);
    assert_eq!(rig.cause(), 1);
    reflect(&rig, true);
    assert_eq!(rig.cause(), 3);
    rig.w(reg::RT_COMPARE_A, 2);
    rig.w(reg::PROFILE_COMPARE, 2);
    rig.w(reg::RT_ENABLE_A, 1);
    rig.w(reg::PROFILE_ENABLE_A, 1);
    rig.w(reg::ERROR_CAUSE_A, 1);
    rig.w(reg::ERROR_SUMMARY_ALIAS, 1);
    assert_eq!(rig.cause(), 0b10011);
    rig.set_time(1600);
    assert_eq!(rig.r(reg::RT_STATUS_A), 1);
    assert_eq!(rig.cause(), 31);
    rig.w(reg::RT_STATUS_A, 0);
    assert_eq!(rig.cause(), 27);
    rig.w(reg::PENDING_MODIFY, 12);
    reflect(&rig, false);
    rig.w(reg::PROFILE_STATUS_A, 0);
    rig.w(reg::ERROR_SUMMARY, 1);
    assert_eq!(rig.cause(), 0);
    assert_eq!(rig.word.load(Ordering::SeqCst), 0xffff_0000_0000_83ff & !(0x1f << 10), "only bits 10-14 are the PI's");
}

/// A guest can wait for a timer interrupt without touching a register: the
/// host timer latches the comparator and raises IP4.
#[test]
fn the_timer_manager_delivers_expirations_without_mmio_polling() {
    let config = super::super::test_config(crate::dev::heart::host_clock());
    let chip = super::super::Bedrock::new(config).unwrap();
    let word = Arc::new(std::sync::atomic::AtomicU64::new(0));
    chip.set_cpu_interrupts(0, &*word);
    let tm = Arc::new(crate::hptimer::TimerManager::new());
    chip.set_timer_manager(&tm);
    let regs = chip.registers();
    assert_eq!(regs.write64(reg::RT_COMPARE_A, 12500), BUS_OK);
    assert_eq!(regs.write64(reg::RT_ENABLE_A, 1), BUS_OK);
    chip.start();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while word.load(Ordering::SeqCst) & (1 << 12) == 0 {
        assert!(std::time::Instant::now() < deadline, "IP4 never rose");
        std::thread::sleep(Duration::from_millis(1));
    }
    chip.stop();
    assert!(!chip.is_running());
}

/// The PROM counts processors from the present bits: the one processor
/// sits in PI0 slot A, present and enabled, and reads slice A of PI0 from
/// the CPU-number register.
#[test]
fn the_one_processor_sits_in_pi0_slot_a() {
    let rig = Rig::new();
    let read = |interface: u64, off| rig.r(interface * PI1_BASE + off);
    assert_eq!([read(0, reg::INTERFACE_SLOT_A), read(0, reg::INTERFACE_SLOT_B),
        read(1, reg::INTERFACE_SLOT_A), read(1, reg::INTERFACE_SLOT_B)], [1, 0, 0, 0]);
    assert_eq!([read(0, reg::CPU_A_ENABLE), read(0, reg::CPU_B_ENABLE),
        read(1, reg::CPU_A_ENABLE), read(1, reg::CPU_B_ENABLE)], [1, 0, 0, 0], "enables");
    assert_eq!((read(0, reg::INTERFACE_ID), read(1, reg::INTERFACE_ID)), (0, 2));
}

/// Enabling a processor that is not fitted, or clearing the fitted one's
/// bit, is stored as the register holds it and starts or stops nothing. An
/// NMI reaches no processor.
#[test]
fn enables_and_nmis_the_model_does_not_carry_out_are_stored() {
    let rig = Rig::new();
    rig.w(reg::OFFSET_00A8, 1 << 9);
    rig.w(reg::CPU_B_ENABLE, 1);
    rig.w(PI1_BASE + reg::CPU_A_ENABLE, 3);
    assert_eq!((rig.r(reg::CPU_B_ENABLE), rig.r(PI1_BASE + reg::CPU_A_ENABLE)), (1, 1));
    rig.w(reg::CPU_A_ENABLE, 0);
    assert_eq!(rig.r(reg::CPU_A_ENABLE), 0);
    rig.w(reg::NMI_SLOT_A, 1);
    rig.w(PI1_BASE + reg::NMI_SLOT_B, u64::MAX);
    // CPU 0 still takes its interrupts.
    rig.w(reg::PENDING_MODIFY, 0x109);
    assert_eq!(rig.cause(), 1);
}

/// A nonzero mask, timer enable or error cause for a slot with no processor
/// is stored and reaches no processor; an interprocessor interrupt to PI1/A
/// lands in its pending bank and goes nowhere.
#[test]
fn arming_a_slot_with_no_processor_reaches_no_processor() {
    let rig = Rig::new();
    rig.w(reg::OFFSET_00C0, 1 << 3);
    rig.w(PI1_BASE + reg::OFFSET_00A8, 1 << 7);
    rig.w(reg::RT_ENABLE_B, 1);
    rig.w(reg::PROFILE_ENABLE_B, 1);
    rig.w(PI1_BASE + reg::ERROR_CAUSE_B, MEMORY_ERROR);
    assert_eq!((rig.r(reg::OFFSET_00C0), rig.r(PI1_BASE + reg::OFFSET_00A8)), (1 << 3, 1 << 7));
    assert_eq!(rig.r(reg::RT_ENABLE_B), 1);
    rig.w(PI1_BASE + reg::PENDING_MODIFY, 0x107);
    assert_eq!(rig.r(PI1_BASE + reg::PENDING_STATUS_A) & (1 << 7), 1 << 7);
    assert_eq!(rig.cause(), 0);
}

impl Pi {
    /// Hold both locks, for tests of the paths that must not take them.
    pub(in crate::dev::bedrock) fn hold(&self) -> impl Sized + '_ {
        (self.state.lock(), self.timer.lock())
    }
}
