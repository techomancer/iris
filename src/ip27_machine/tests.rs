use super::*;
use crate::test_temp::TempPath;
use crate::config::GraphicsBoard;
use super::test_support::{self, big_stack};
use crate::dev::hub::regs as hub_regs;
use crate::irq::InterruptController;
use crate::traits::BUS_OK;

fn config(dir: &Path) -> MachineConfig {
    test_support::config(MachineProfile::Origin200Ip29, dir, 1)
}

/// A module's Config word, part table and built cache agree on the secondary cache (SS, bits 18:16,
/// 512 KB << SS).
#[test]
fn a_modules_config_word_part_and_cache_type_agree() {
    for (part, l2_kb) in [(&parts::R10000_180, 1024), (&parts::R12000_270, 4096)] {
        let config = parts::cpu_module(part).unwrap().config();
        assert_eq!(512u32 << ((config >> 16) & 7), l2_kb, "{}", part.name());
        assert_eq!(part.l2_kb, l2_kb, "{}", part.name());
        let model = if l2_kb == 4096 {
            <crate::cpu::mips_cache_shadow::R12000Module4MShadowCache as crate::cpu::mips_cache_v2::MipsCache>::L2_SIZE as u32 / 1024
        } else {
            crate::with_cpu!(part, true, false, C => <C as crate::cpu::mips_cache_v2::MipsCache>::L2_SIZE as u32 / 1024)
        };
        assert_eq!(model, l2_kb, "{}", part.name());
    }
    assert_eq!(parts::R10000_180.count_hz(), 90_000_000);
    assert!(parts::cpu_module(&parts::R14000_600).is_none());
}

#[test]
fn the_r12000_module_has_no_record_on_its_one_wire_part() {
    let dir = TempPath::dir("sn0-o200-node-record");
    let mut cfg = config(&dir);
    cfg.set_cpu(&parts::R12000_270);
    assert!(origin200_specs(&cfg).unwrap()[0].node_record.is_none());
    cfg.set_cpu(&parts::R10000_180);
    assert_eq!(origin200_specs(&cfg).unwrap()[0].node_record.unwrap().name, "PIMM_1XT5_1MB");
}

#[test]
fn the_profile_takes_an_r10000_or_r12000_and_fills_banks_from_zero() {
    let dir = TempPath::dir("sn0-o200-config");
    let mut cfg = config(&dir);
    assert_eq!(cfg.validate(), Ok(()));
    cfg.set_cpu(&parts::R14000_600);
    assert!(cfg.validate().unwrap_err().contains("takes [machine] cpu"));
    cfg.set_cpu(&parts::R12000_270);
    assert_eq!(cfg.validate(), Ok(()));
    cfg.banks = [64, 0, 64, 0, 0, 0, 0, 0];
    assert!(cfg.validate().unwrap_err().contains("fill the banks from 0"));
    cfg.banks = [32, 0, 0, 0, 0, 0, 0, 0];
    assert!(cfg.validate().unwrap_err().contains("invalid for an Origin 200"));
    // The PROM disables 256 MB banks on the 180 MHz module only.
    cfg.banks = [256, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(cfg.validate(), Ok(()));
    cfg.set_cpu(&parts::R10000_180);
    assert!(cfg.validate().unwrap_err().contains("disables 256 MB banks"));
    cfg.banks = [128, 128, 0, 0, 0, 0, 0, 0];
    assert_eq!(cfg.validate(), Ok(()));
    cfg.banks = [64, 0, 0, 0, 0, 0, 0, 0];
    cfg.graphics.board = Some(GraphicsBoard::V12);
    assert!(cfg.validate().unwrap_err().contains("not modelled on the Origin 200"));
}

#[test]
fn the_reset_vector_reaches_the_flash_and_the_processor_is_enabled() {
    big_stack(|| {
        let dir = TempPath::dir("sn0-o200-reset");
        let mut m = Ip27Machine::new(&config(&dir)).unwrap();
        // Node memory view, the Hub-special views of the flash (own and named),
        // all show the PROM's first word at the reset address and the window.
        let memory = m.node_at(0).chip.decode.physical();
        assert_eq!(memory.read32(0x1fc0_0000).data, 0x3c08_a600);
        let special = m.node_at(0).chip.cpu_view(crate::sn_nodes::UncachedSpace::HubSpecial);
        assert_eq!(special.read32(0x1000_0000).data, 0x3c08_a600);
        assert_eq!(special.read32(0x3000_0000).data, 0x3c08_a600);
        // The Hub's own register: PI_CPU_ENABLE_A reads one without the PROM
        // having stored it.
        let io = m.node_at(0).chip.cpu_view(crate::sn_nodes::UncachedSpace::Io);
        let enable = crate::dev::hub::IO_REGISTERS_LOCAL + crate::dev::hub::regs::PI_CPU_ENABLE_A::OFFSET;
        assert_eq!(io.read64(enable).data, 1);
        // And its slice B, which a one-processor module does not fit, is clear.
        assert_eq!(io.read64(crate::dev::hub::IO_REGISTERS_LOCAL + hub_regs::PI_CPU_ENABLE_B::OFFSET).data, 0);
        // A reset enables the processor again.
        m.reset();
        assert_eq!(io.read64(enable).data, 1);
        m.stop();
    });
}

#[test]
fn the_bridge_answers_through_the_hub_at_every_widget_but_its_own() {
    big_stack(|| {
        let dir = TempPath::dir("sn0-o200-bridge");
        let m = Ip27Machine::new(&config(&dir)).unwrap();
        let io = m.node_at(0).chip.cpu_view(crate::sn_nodes::UncachedSpace::Io);
        let widget = |w: u64| w << 24;
        // The Bridge's identification register, in the data half of its slot.
        let id = |w: u64| io.read32(widget(w) + 4).data;
        assert_ne!(id(WIDGET_HUB.into()), id(8), "widget 1 is the Hub's, not the Bridge's");
        for w in [0, 2, 8, 15] {
            assert_eq!(id(w), id(8), "widget {w}");
        }
        assert_eq!(io.write32(widget(8) + crate::dev::bridge::reg::INT_ENABLE as u64 + 4, 0), BUS_OK);
    });
}

/// A write to the Hub's soft-reset register through the processor's I/O space reaches the
/// processor's interrupt word, and the processor restarts at the reset vector on its next step.
#[test]
fn a_soft_reset_write_through_the_hubs_io_space_restarts_the_processor() {
    use std::sync::atomic::Ordering;
    big_stack(|| {
        let dir = TempPath::dir("sn0-o200-softreset");
        let m = Ip27Machine::new(&config(&dir)).unwrap();
        let cpu = m.cpu_at(0);
        cpu.step_n_inline(5).unwrap();
        assert_ne!(cpu.state_digest().unwrap().pc, 0xffff_ffff_bfc0_0000, "the processor has left the reset vector");
        let word = unsafe { &*cpu.interrupts_ptr() };
        assert_eq!(word.load(Ordering::SeqCst) & crate::cpu::mips_exec::SOFT_RESET_BIT, 0);
        let io = m.node_at(0).chip.cpu_view(crate::sn_nodes::UncachedSpace::Io);
        assert_eq!(io.write64(crate::dev::hub::IO_REGISTERS_LOCAL + crate::dev::hub::regs::PI_SOFTRESET::OFFSET, 1), BUS_OK);
        assert_ne!(word.load(Ordering::SeqCst) & crate::cpu::mips_exec::SOFT_RESET_BIT, 0, "the write raised the line");
        cpu.step_n_inline(1).unwrap();
        assert_eq!(word.load(Ordering::SeqCst) & crate::cpu::mips_exec::SOFT_RESET_BIT, 0, "the processor took it");
        assert_eq!(cpu.state_digest().unwrap().pc, 0xffff_ffff_bfc0_0000);
    });
}

#[test]
fn the_flash_carries_the_products_machine_type_and_cycle_counts() {
    big_stack(|| {
        let dir = TempPath::dir("sn0-o200-flash");
        let m = Ip27Machine::new(&config(&dir)).unwrap();
        let word = |at: u64| m.node_at(0).chip.cpu_view(crate::sn_nodes::UncachedSpace::HubSpecial).read32(0x1000_0000 + at).data;
        // Offsets 0x8c, 0x90, 0x94 and 0x9c of the configuration block.
        assert_eq!(word(0x8c), 15);
        assert_eq!(word(0x90), 1);
        assert_eq!(word(0x9c), 4);
        // The processor module's mode word and clock.
        assert_eq!(word(0x64), 0x1211_a785);
        assert_eq!(word(0x74), 180_000_000);
    });
}

#[test]
fn a_scratch_volume_is_the_machines_scratch_path() {
    big_stack(|| {
        let dir = TempPath::dir("sn0-o200-scratch");
        let mut cfg = config(&dir);
        let disk = dir.join("scratch.raw");
        cfg.scsi.clear();
        cfg.scsi.insert(2, toml::from_str(&format!("path = {:?}\nscratch = true\nsize_mb = 4", disk.to_string_lossy())).unwrap());
        let m = Ip27Machine::new(&cfg).unwrap();
        assert_eq!(m.scratch_path(), Some(disk.as_path()));
    });
}

/// A controller whose next due time and outputs the test sets.
struct Scripted {
    due: Mutex<Option<Duration>>,
    outputs: Mutex<Vec<ModuleOutput>>,
    wake: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>>,
}

impl Scripted {
    fn new() -> Arc<Self> {
        Arc::new(Self { due: Mutex::new(None), outputs: Mutex::new(Vec::new()), wake: Default::default() })
    }

    /// The controller's schedule moved, as a bus access would have moved it.
    fn schedule(&self, due: Option<Duration>, outputs: &[ModuleOutput]) {
        *self.due.lock() = due;
        self.outputs.lock().extend_from_slice(outputs);
        (self.wake.get().expect("the watch installs its wake"))();
    }
}

impl ModuleController for Scripted {
    fn reset(&self) {}
    fn take_events(&self) -> Vec<ModuleOutput> {
        *self.due.lock() = None;
        std::mem::take(&mut *self.outputs.lock())
    }
    fn next_due(&self) -> Option<Duration> {
        *self.due.lock()
    }
    fn set_wake(&self, wake: Box<dyn Fn() + Send + Sync>) {
        let _ = self.wake.set(wake);
    }
}

/// The controller is visited when it says it is due and not before, a change to its schedule
/// re-arms the one timer rather than adding another, and a stopped machine holds no timer.
#[test]
fn the_controller_is_visited_when_due_and_not_while_stopped() {
    let timers = Arc::new(TimerManager::new());
    let (tx, rx) = channel();
    let controller = Scripted::new();
    let watch = ControllerWatch::new(controller.clone(), tx, &timers);
    let wait = |what: &str| rx.recv_timeout(Duration::from_secs(10)).unwrap_or_else(|_| panic!("{what}: no output"));
    // The visit sends its outputs, then arms for what is next: wait for the
    // arming to settle on nothing.
    let settled = |watch: &ControllerWatch| {
        let end = std::time::Instant::now() + Duration::from_secs(10);
        while watch.armed().is_some() && std::time::Instant::now() < end {
            std::thread::sleep(Duration::from_millis(5));
        }
        watch.armed().is_none()
    };

    // Nothing pending: nothing armed, and a stopped watch arms nothing.
    controller.schedule(Some(Duration::from_millis(10)), &[ModuleOutput::Power(false)]);
    assert!(watch.armed().is_none(), "stopped");
    watch.start();
    assert!(watch.armed().is_none() || controller.next_due().is_some());
    assert!(matches!(wait("power"), MachineEvent::PowerOff));
    assert!(settled(&watch), "nothing is pending after the visit");

    // A schedule that moves while armed replaces the timer.
    controller.schedule(Some(Duration::from_secs(3600)), &[]);
    let far = watch.armed().expect("armed for an hour");
    controller.schedule(Some(Duration::from_millis(10)), &[ModuleOutput::Reset]);
    assert_ne!(watch.armed(), Some(far));
    assert!(matches!(wait("reset"), MachineEvent::HardReset));

    // Stopping removes the timer, and a later move arms none.
    controller.schedule(Some(Duration::from_secs(3600)), &[]);
    assert!(watch.armed().is_some());
    watch.stop();
    assert!(watch.armed().is_none());
    controller.schedule(Some(Duration::from_secs(3600)), &[]);
    assert!(watch.armed().is_none());
}

/// A real machine arms its controller when started, for whatever the
/// controller has pending, and holds nothing once stopped.
#[test]
fn a_machines_controller_is_armed_for_what_it_has_pending_and_stopped_with_the_machine() {
    big_stack(|| {
        let dir = TempPath::dir("sn0-o200-timer");
        let mut m = Ip27Machine::new(&config(&dir)).unwrap();
        assert!(m.watch.armed().is_none());
        m.restart_peripherals();
        assert_eq!(m.watch.armed().is_some(), m.io.controller.next_due().is_some());
        m.stop();
        assert!(m.watch.armed().is_none());
    });
}

/// The Hub publishes straight into its processor's interrupt word, so the
/// machine lets go of the word before the processor goes: a cabled Hub, or any
/// other holder of this one, can still raise an interrupt afterwards.
#[test]
fn a_hub_that_outlives_its_machine_publishes_into_no_freed_processor() {
    big_stack(|| {
        let dir = TempPath::dir("sn0-o200-interrupt-word");
        let m = Ip27Machine::new(&config(&dir)).unwrap();
        let hub = m.node_at(0).chip.clone();
        let cpu = Arc::downgrade(m.cpu_at(0));
        assert!(hub.pi.publishes_to(0), "the machine wired the processor's word");
        drop(m);
        assert_eq!(cpu.strong_count(), 0, "the processor is freed with the machine");
        assert!(!hub.pi.publishes_to(0), "and the Hub no longer holds its word");
        // Raising and unmasking an interrupt now publishes nowhere.
        assert_eq!(hub.write64(hub_regs::PI_INT_MASK0_A::OFFSET, !0), BUS_OK);
        hub.interrupt_target().set_line(3, true);
    });
}

/// The PROM of a board's second node keeps its stack in uncached Hub-special
/// memory above the 4 KiB cached alias it programs, and renumbers the node
/// between saving its return address there and reloading it. A store-conditional is a
/// compare-and-swap once processors share memory, so a cached reference by number reaches it.
#[test]
fn renumbering_a_lone_node_keeps_its_memory_and_lets_a_reference_by_number_compare_and_swap() {
    big_stack(|| {
        let dir = TempPath::dir("sn0-o200-renumber");
        let m = Ip27Machine::new(&config(&dir)).unwrap();
        let hub = &m.node_at(0).chip;
        const SLOT: u64 = 0x1bef_e78;
        assert_eq!(hub.write64(crate::dev::hub::regs::PI_CALIAS_SIZE::OFFSET, 1), BUS_OK);
        assert_eq!(hub.cpu_view(crate::sn_nodes::UncachedSpace::HubSpecial).write64(SLOT, 0x5a5a), BUS_OK);
        assert_eq!(hub.write64(crate::dev::hub::regs::NI_STATUS_REV_ID::OFFSET, 1 << 8), BUS_OK);
        assert_eq!(hub.cpu_view(crate::sn_nodes::UncachedSpace::HubSpecial).read64(SLOT).data, 0x5a5a, "the stack slot");
        let phys = m.node_at(0).chip.decode.physical();
        assert_eq!(phys.read64((1 << 32) + SLOT).data, 0x5a5a, "node 1 by number");
        assert_eq!(phys.read64(SLOT).data, 0x5a5a, "node zero");
        const WORD: u64 = (1 << 32) + 0x61_c000;
        assert_eq!(phys.write64(WORD, 0x99), BUS_OK);
        assert_eq!(phys.cas64(WORD, 0x99, 0x9a), Some(true));
        assert_eq!(phys.cas64(WORD, 0x99, 0x9b), Some(false));
        assert_eq!(phys.cas32(WORD + 4, 0x9a, 0x9c), Some(true));
        assert_eq!(hub.cpu_view(crate::sn_nodes::UncachedSpace::HubSpecial).read64(0x61_c000).data, 0x9c);
    });
}
