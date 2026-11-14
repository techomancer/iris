// Cases follow tests/qtest/sn1-bedrock-pi-test.c in jamesbraid/qemu-sgi-origami:
// power-on-values, local-reset, pending-reflection, real-time-comparators,
// profile-counter-wrap, cpu-number-fallback, cpu-enable-lines and nmi, and
// sn1-cpu-identity-test's four-processors-know-their-slots.

use super::super::regs::PI1_BASE;
use super::super::Rig;
use super::*;
use crate::sn_nodes::IO_REGISTERS_LOCAL;
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
        assert_eq!(rig.r(reg::CPU_NUM::OFFSET), 0);
        assert_eq!(rig.r(reg::CPU_PRESENT_A::OFFSET), 1);
        assert_eq!(rig.r(reg::CPU_PRESENT_B::OFFSET), 0);
        rig.w(reg::CPU_PROTECT::OFFSET, 0);
        rig.w(reg::PROT_OVERRIDE::OFFSET, IIO_PROTECT_OVRRD_KEY);
        assert_eq!(rig.r(reg::CPU_PROTECT::OFFSET), 1);
        assert_eq!(rig.r(PI1_BASE + reg::CPU_NUM::OFFSET), 2);
        assert_eq!(rig.r(PI1_BASE + reg::CPU_PRESENT_A::OFFSET), 0);
        assert_eq!(rig.regs.read32(reg::CPU_NUM::OFFSET).status, BUS_ERR);
        assert_eq!(rig.regs.write32(reg::RT_COUNTER::OFFSET, 0), BUS_ERR);
    }
    {
        let rig = Rig::new();
        rig.w(reg::INT_PEND_MOD::OFFSET, 0x10c);
        assert_eq!(rig.r(reg::INT_PEND0::OFFSET), 1 << 12);
        rig.w(reg::INT_PEND_MOD_ALIAS::OFFSET, 12);
        assert_eq!(rig.r(reg::INT_PEND0::OFFSET), 0);
        reflect(&rig, true);
        rig.w(reg::INT_PEND_MOD::OFFSET, 64 + 55);
        assert_eq!(rig.r(reg::INT_PEND1::OFFSET), 1 << 55);
        rig.w(reg::INT_PEND_MOD::OFFSET, 0x100 | 64 | 63);
        assert_eq!(rig.r(reg::INT_PEND1::OFFSET), 1 << 55);
        rig.w(reg::CC_PEND_SET_A::OFFSET, 0);
        rig.w(reg::CC_MASK::OFFSET, 1);
        assert_eq!(rig.r(reg::INT_PEND0::OFFSET), 1 << 5);
        rig.w(reg::CC_PEND_CLR_A::OFFSET, 1);
        assert_eq!(rig.r(reg::INT_PEND0::OFFSET), 0);
    }
    {
        let rig = Rig::new();
        rig.w(reg::CALIAS_SIZE::OFFSET, 9);
        rig.w(reg::CPU_ENABLE_A::OFFSET, 1);
        rig.w(reg::ERR_STACK_ADDR_A::OFFSET, !0);
        rig.chip.pi[0].error_summary.fetch_or(MEMORY_ERROR, Ordering::SeqCst);
        rig.w(reg::ERR_INT_PEND_WR::OFFSET, 3);
        rig.w(reg::RT_COMPARE_A::OFFSET, 0x24);
        rig.chip.pi[0].power_on(0, &rig.chip.levels);
        assert_eq!(rig.r(reg::ERR_INT_PEND::OFFSET), MEMORY_ERROR | 3);
        assert_eq!(rig.r(reg::CALIAS_SIZE::OFFSET), 9);
        assert_eq!(rig.r(reg::CPU_ENABLE_A::OFFSET), 1);
        assert_eq!(rig.r(reg::ERR_STACK_ADDR_A::OFFSET), 0x1fffffff8);
        assert_eq!(rig.r(reg::RT_COMPARE_A::OFFSET), 0);
        assert_eq!(rig.r(reg::RT_COUNTER::OFFSET), 0);
        rig.w(reg::ERR_INT_PEND_WR::OFFSET, 0);
        assert_eq!(rig.r(reg::ERR_INT_PEND::OFFSET), MEMORY_ERROR);
        rig.w(reg::ERR_INT_PEND::OFFSET, MEMORY_ERROR);
        assert_eq!(rig.r(reg::ERR_INT_PEND::OFFSET), 0);
    }
}

/// The counter wraps, the comparators latch at their tick, and the counter
/// pauses while the machine is stopped.
#[test]
fn the_pi_timer_wraps_and_pauses() {
    {
        let rig = Rig::new();
        rig.w(reg::RT_COUNTER::OFFSET, 0x1_0000_0000);
        rig.w(reg::RT_COMPARE_A::OFFSET, 0x1_0000_0002);
        rig.set_time(1599);
        assert_eq!(rig.r(reg::RT_INT_PEND_A::OFFSET), 0);
        rig.set_time(1600);
        assert_eq!(rig.r(reg::RT_INT_PEND_A::OFFSET), 1);
        assert_eq!(rig.regs.read32(reg::RT_COUNTER::OFFSET).data, 1);
        assert_eq!(rig.regs.read32(reg::RT_COUNTER::OFFSET + 4).data, 2);
        rig.w(reg::RT_INT_PEND_A::OFFSET, !0);
        assert_eq!(rig.r(reg::RT_INT_PEND_A::OFFSET), 0);
        rig.w(reg::RT_COUNTER::OFFSET, RT_MASK);
        rig.w(reg::PROFILE_COMPARE::OFFSET, 0);
        rig.set_time(2400);
        assert_eq!(rig.r(reg::RT_COUNTER::OFFSET), 0);
        assert_eq!(rig.r(reg::PROF_INT_PEND_A::OFFSET), 1);
        assert_eq!(rig.r(reg::PROF_INT_PEND_B::OFFSET), 1);
        rig.w(reg::PROF_INT_PEND_A::OFFSET, 0);
        assert_eq!(rig.r(reg::PROF_INT_PEND_A::OFFSET), 0);
        assert_eq!(rig.r(reg::PROF_INT_PEND_B::OFFSET), 1);
    }
    {
        let rig = Rig::new();
        rig.set_time(1_000_000);
        rig.w(reg::RT_COUNTER::OFFSET, 0);
        rig.w(reg::RT_COMPARE_A::OFFSET, 2);
        rig.set_time(1_000_400);
        rig.chip.stop();
        rig.set_time(2_000_000);
        assert_eq!(rig.r(reg::RT_COUNTER::OFFSET), 0);
        assert_eq!(rig.r(reg::RT_INT_PEND_A::OFFSET), 0, "no time passes while stopped");
        rig.chip.start();
        rig.set_time(2_001_199);
        assert_eq!(rig.r(reg::RT_INT_PEND_A::OFFSET), 0);
        rig.set_time(2_001_200);
        assert_eq!(rig.r(reg::RT_INT_PEND_A::OFFSET), 1);
    }
}

/// Each of a slot's five outputs follows its own inputs into Cause bits
/// 10-14 of the processor's word, and nothing else in the word moves.
#[test]
fn pi_interrupt_outputs_follow_masks_into_the_cpus_word() {
    let rig = Rig::new();
    rig.word.store(0xffff_0000_0000_83ff & !(0x1f << 10), Ordering::SeqCst);
    rig.w(reg::INT_MASK0_A::OFFSET, 1 << 12);
    rig.w(reg::INT_MASK1_A::OFFSET, 1 << 55);
    rig.w(reg::INT_PEND_MOD::OFFSET, 0x10c);
    assert_eq!(rig.cause(), 1);
    reflect(&rig, true);
    assert_eq!(rig.cause(), 3);
    rig.w(reg::RT_COMPARE_A::OFFSET, 2);
    rig.w(reg::PROFILE_COMPARE::OFFSET, 2);
    rig.w(reg::RT_INT_EN_A::OFFSET, 1);
    rig.w(reg::PROF_INT_EN_A::OFFSET, 1);
    rig.w(reg::ERR_INT_MASK_A::OFFSET, 1);
    rig.w(reg::ERR_INT_PEND_WR::OFFSET, 1);
    assert_eq!(rig.cause(), 0b10011);
    rig.set_time(1600);
    assert_eq!(rig.r(reg::RT_INT_PEND_A::OFFSET), 1);
    assert_eq!(rig.cause(), 31);
    rig.w(reg::RT_INT_PEND_A::OFFSET, 0);
    assert_eq!(rig.cause(), 27);
    rig.w(reg::INT_PEND_MOD::OFFSET, 12);
    reflect(&rig, false);
    rig.w(reg::PROF_INT_PEND_A::OFFSET, 0);
    rig.w(reg::ERR_INT_PEND::OFFSET, 1);
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
    assert_eq!(regs.write64(reg::RT_COMPARE_A::OFFSET, 12500), BUS_OK);
    assert_eq!(regs.write64(reg::RT_INT_EN_A::OFFSET, 1), BUS_OK);
    chip.start();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while word.load(Ordering::SeqCst) & (1 << 12) == 0 {
        assert!(std::time::Instant::now() < deadline, "IP4 never rose");
        std::thread::sleep(Duration::from_millis(1));
    }
    chip.stop();
    assert!(!chip.is_running());
}

/// The PROM counts processors from the present bits. A board fills PI0/A,
/// then PI1/A, then PI0/B and PI1/B, so two processors sit one per interface.
#[test]
fn every_fitted_slot_reads_present_and_starts_enabled() {
    for (mask, present) in [(0b0001, [1, 0, 0, 0]), (0b0101, [1, 0, 1, 0]), (0b0111, [1, 1, 1, 0]), (0b1111, [1, 1, 1, 1])] {
        let rig = Rig::with(Population::new(mask).unwrap());
        let read = |interface: u64, off| rig.r(interface * PI1_BASE + off);
        assert_eq!([read(0, reg::CPU_PRESENT_A::OFFSET), read(0, reg::CPU_PRESENT_B::OFFSET),
            read(1, reg::CPU_PRESENT_A::OFFSET), read(1, reg::CPU_PRESENT_B::OFFSET)], present, "{mask:#06b}");
        assert_eq!([read(0, reg::CPU_ENABLE_A::OFFSET), read(0, reg::CPU_ENABLE_B::OFFSET),
            read(1, reg::CPU_ENABLE_A::OFFSET), read(1, reg::CPU_ENABLE_B::OFFSET)], present, "{mask:#06b}: enables");
        assert_eq!((read(0, reg::CPU_NUM::OFFSET), read(1, reg::CPU_NUM::OFFSET)), (0, 2), "{mask:#06b}: no requester");
    }
}

/// A population that cannot hold CPU 0, or names a fifth slot, is refused
/// rather than built; the board's order puts CPU n in the nth fitted slot,
/// and one, two and four processors fill PI0/A, then PI1/A, then the B slots.
#[test]
fn a_population_without_cpu_0_or_beyond_four_slots_is_refused() {
    for mask in [0, 0b0010, 0b1100, 0b1_0001, 0xff] {
        assert!(Population::new(mask).is_err(), "{mask:#x}");
    }
    let four = Population::new(0b1111).unwrap();
    assert_eq!((0..4).map(|cpu| four.slot_of(cpu).unwrap()).collect::<Vec<_>>(), [(0, 0), (1, 0), (0, 1), (1, 1)]);
    assert_eq!(four.slot_of(4), None);
    assert_eq!(Population::new(0b0101).unwrap().slot_of(1), Some((1, 0)));
    assert_eq!(Population::SINGLE.count(), 1);
    assert_eq!([1, 2, 4].map(|n| Population::fitted_cpus(n).unwrap().mask()), [0b0001, 0b0101, 0b1111]);
    assert!(Population::fitted_cpus(0).is_err() && Population::fitted_cpus(5).is_err());
}

/// Each processor reads its own slice from the CPU-number register of its
/// interface, and a reader that is no processor reads slot A. The local
/// register form of the I/O space answers the requester's own PI.
/// (sn1-cpu-identity-test four-processors-know-their-slots)
#[test]
fn four_processors_know_their_slots_and_their_own_pi() {
    let rig = Arc::new(Rig::with(Population::new(0b1111).unwrap()));
    assert_eq!((rig.r(reg::CPU_NUM::OFFSET), rig.r(PI1_BASE + reg::CPU_NUM::OFFSET)), (0, 2), "no issuing processor");
    let local = rig.chip.io_space();
    let slices: Vec<(u64, u64)> = (0..4u8)
        .map(|cpu| {
            let (rig, local) = (rig.clone(), local.clone());
            std::thread::spawn(move || {
                crate::cpu::smp::set_current_cpu(cpu);
                let (interface, _) = Population::new(0b1111).unwrap().slot_of(cpu as usize).unwrap();
                (rig.r(u64::from(interface) * PI1_BASE + reg::CPU_NUM::OFFSET), local.read64(IO_REGISTERS_LOCAL + reg::CPU_NUM::OFFSET).data)
            })
            .join()
            .unwrap()
        })
        .collect();
    assert_eq!(slices, [(0, 0), (2, 2), (1, 1), (3, 3)], "slices A, C, B, D");
    // With processors on PI0 alone, the local form is the node form.
    let fuel = Rig::new();
    std::thread::spawn(move || {
        crate::cpu::smp::set_current_cpu(0);
        assert_eq!(fuel.chip.io_space().read64(IO_REGISTERS_LOCAL + reg::CPU_NUM::OFFSET).data, 0);
    })
    .join()
    .unwrap();
}

/// Each processor's IP2..IP6 come from its own slot: an interrupt pending
/// in PI1's bank 0 under slot A's mask reaches CPU 1 and no other.
#[test]
fn each_processor_takes_the_interrupts_of_its_own_slot() {
    let rig = Rig::with(Population::new(0b1111).unwrap());
    let words: Vec<Arc<std::sync::atomic::AtomicU64>> = (0..4).map(|_| Arc::new(Default::default())).collect();
    for (cpu, word) in words.iter().enumerate() {
        rig.chip.set_cpu_interrupts(cpu, &**word);
    }
    rig.w(PI1_BASE + reg::INT_MASK0_A::OFFSET, 1 << 9);
    rig.w(PI1_BASE + reg::INT_PEND_MOD::OFFSET, 0x109);
    let ip2 = |cpu: usize| (words[cpu].load(Ordering::SeqCst) >> 10) & 1;
    assert_eq!([ip2(0), ip2(1), ip2(2), ip2(3)], [0, 1, 0, 0]);
}

/// Enabling a processor that is not fitted, or clearing the fitted one's
/// bit, is stored as the register holds it and starts or stops nothing. An
/// NMI reaches no processor.
#[test]
fn enables_and_nmis_the_model_does_not_carry_out_are_stored() {
    let rig = Rig::new();
    rig.w(reg::INT_MASK0_A::OFFSET, 1 << 9);
    rig.w(reg::CPU_ENABLE_B::OFFSET, 1);
    rig.w(PI1_BASE + reg::CPU_ENABLE_A::OFFSET, 3);
    assert_eq!((rig.r(reg::CPU_ENABLE_B::OFFSET), rig.r(PI1_BASE + reg::CPU_ENABLE_A::OFFSET)), (1, 1));
    rig.w(reg::CPU_ENABLE_A::OFFSET, 0);
    assert_eq!(rig.r(reg::CPU_ENABLE_A::OFFSET), 0);
    rig.w(reg::NMI_A::OFFSET, 1);
    rig.w(PI1_BASE + reg::NMI_B::OFFSET, u64::MAX);
    // CPU 0 still takes its interrupts.
    rig.w(reg::INT_PEND_MOD::OFFSET, 0x109);
    assert_eq!(rig.cause(), 1);
}

/// A nonzero mask, timer enable or error cause for a slot with no processor
/// is stored and reaches no processor; an interprocessor interrupt to PI1/A
/// lands in its pending bank and goes nowhere.
#[test]
fn arming_a_slot_with_no_processor_reaches_no_processor() {
    let rig = Rig::new();
    rig.w(reg::INT_MASK1_B::OFFSET, 1 << 3);
    rig.w(PI1_BASE + reg::INT_MASK0_A::OFFSET, 1 << 7);
    rig.w(reg::RT_INT_EN_B::OFFSET, 1);
    rig.w(reg::PROF_INT_EN_B::OFFSET, 1);
    rig.w(PI1_BASE + reg::ERR_INT_MASK_B::OFFSET, MEMORY_ERROR);
    assert_eq!((rig.r(reg::INT_MASK1_B::OFFSET), rig.r(PI1_BASE + reg::INT_MASK0_A::OFFSET)), (1 << 3, 1 << 7));
    assert_eq!(rig.r(reg::RT_INT_EN_B::OFFSET), 1);
    rig.w(PI1_BASE + reg::INT_PEND_MOD::OFFSET, 0x107);
    assert_eq!(rig.r(PI1_BASE + reg::INT_PEND0::OFFSET) & (1 << 7), 1 << 7);
    assert_eq!(rig.cause(), 0);
}

impl Pi {
    /// Hold every lock, for tests of the paths that must not take them.
    pub(in crate::dev::bedrock) fn hold(&self) -> impl Sized + '_ {
        (self.out.lock(), self.state.lock(), self.timer.lock())
    }
}

thread_local! {
    static SAMPLED: std::cell::RefCell<Option<Box<dyn FnMut(u64)>>> = const { std::cell::RefCell::new(None) };
}

/// Run `hook` on this thread each time a publish has sampled a slot's
/// IP2..IP6 and not yet written them, so a test can hold a publisher there.
pub(in crate::dev::bedrock) fn on_sample(hook: impl FnMut(u64) + 'static) {
    SAMPLED.with(|h| *h.borrow_mut() = Some(Box::new(hook)));
}

/// The publish's side of [`on_sample`].
pub(super) fn sampled(want: u64) {
    SAMPLED.with(|h| {
        if let Some(hook) = h.borrow_mut().as_mut() {
            hook(want);
        }
    });
}
