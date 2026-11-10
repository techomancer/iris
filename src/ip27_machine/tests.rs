use super::*;
use crate::config::GraphicsBoard;
use super::test_support::{self, big_stack, Scratch};
use crate::dev::hub::reg::{ni as hub_ni, pi as hub_pi};
use crate::traits::BUS_OK;

fn config(dir: &Path) -> MachineConfig {
    test_support::config(MachineProfile::Origin200Ip29, dir, 1)
}

#[test]
fn the_processor_config_is_the_mode_word_under_the_primary_cache_fields() {
    let r10000 = cpu_module(&parts::R10000_180).unwrap();
    assert_eq!(r10000.config(), 0x6c11_a785);
    assert_eq!(r10000.count_hz(), 90_000_000);
    assert_eq!(cpu_module(&parts::R12000_270).unwrap().config(), 0x6c13_ab85);
    assert!(cpu_module(&parts::R14000_600).is_none());
}

/// The module's secondary cache size is the Config word's (SS, bits 18:16, as
/// 512 KB << SS), the part table's and the cache type the machine builds.
#[test]
fn a_modules_config_word_part_and_cache_type_say_the_same_secondary_cache() {
    for (part, module_kb, model_kb) in [(&parts::R10000_180, 1024, 1024), (&parts::R12000_270, 4096, 4096)] {
        let module = cpu_module(part).unwrap();
        assert_eq!(512u32 << ((module.config() >> 16) & 7), module_kb, "{}", part.name());
        assert_eq!(part.l2_kb, module_kb, "{}", part.name());
        let model = if part.l2_kb == 4096 {
            <crate::cpu::mips_cache_shadow::R12000Module4MShadowCache as crate::cpu::mips_cache_v2::MipsCache>::L2_SIZE as u32 / 1024
        } else {
            crate::with_cpu!(part, true, false, C => <C as crate::cpu::mips_cache_v2::MipsCache>::L2_SIZE as u32 / 1024)
        };
        assert_eq!(model, model_kb, "{}", part.name());
    }
}

#[test]
fn the_r12000_module_has_no_record_on_its_one_wire_part() {
    let mut cfg = config(&std::env::temp_dir());
    cfg.set_cpu(&parts::R12000_270);
    assert!(Origin200::specs(&cfg).unwrap()[0].node_record.is_none());
    cfg.set_cpu(&parts::R10000_180);
    assert_eq!(Origin200::specs(&cfg).unwrap()[0].node_record.unwrap().name, "PIMM_1XT5_1MB");
}

#[test]
fn the_profile_takes_an_r10000_or_r12000_and_fills_banks_from_zero() {
    let dir = Scratch::new("o200-config");
    let mut cfg = config(&dir.0);
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
fn the_profile_is_chosen_by_its_snake_case_name() {
    assert_eq!(MachineProfile::from_name("origin200_ip29"), Ok(MachineProfile::Origin200Ip29));
    assert_eq!(MachineProfile::Origin200Ip29.name(), "origin200_ip29");
}

#[test]
fn the_reset_vector_reaches_the_flash_and_the_processor_is_enabled() {
    big_stack(|| {
        let dir = Scratch::new("o200-reset");
        let m = Origin200Machine::new(&config(&dir.0)).unwrap();
        // Node memory view, the Hub-special views of the flash (own and named),
        // all show the PROM's first word at the reset address and the window.
        let memory = m.node().hub.cpu_memory();
        assert_eq!(memory.read32(0x1fc0_0000).data, 0x3c08_a600);
        let special = m.node().hub.cpu_special();
        assert_eq!(special.read32(0x1000_0000).data, 0x3c08_a600);
        assert_eq!(special.read32(0x3000_0000).data, 0x3c08_a600);
        // The Hub's own register: PI_CPU_ENABLE_A reads one without the PROM
        // having stored it.
        let io = m.node().hub.claim(Space::Io);
        let enable = crate::dev::hub::IO_REGISTERS_LOCAL + hub_pi::CPU_ENABLE_A;
        assert_eq!(io.read64(enable).data, 1);
        // And its slice B, which a one-processor module does not fit, is clear.
        assert_eq!(io.read64(crate::dev::hub::IO_REGISTERS_LOCAL + hub_pi::CPU_ENABLE_B).data, 0);
    });
}

#[test]
fn the_bridge_answers_through_the_hub_at_every_widget_but_its_own() {
    big_stack(|| {
        let dir = Scratch::new("o200-bridge");
        let m = Origin200Machine::new(&config(&dir.0)).unwrap();
        let io = m.node().hub.claim(Space::Io);
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

#[test]
fn a_reset_enables_the_processor_again() {
    big_stack(|| {
        let dir = Scratch::new("o200-restart");
        let mut m = Origin200Machine::new(&config(&dir.0)).unwrap();
        m.reset();
        let io = m.node().hub.claim(Space::Io);
        assert_eq!(io.read64(crate::dev::hub::IO_REGISTERS_LOCAL + hub_pi::CPU_ENABLE_A).data, 1);
        m.stop();
    });
}

/// A write to the Hub's soft-reset register through the processor's I/O
/// space reaches the processor's interrupt word, and the processor restarts at
/// the reset vector on its next step.
#[test]
fn a_soft_reset_write_through_the_hubs_io_space_restarts_the_processor() {
    use std::sync::atomic::Ordering;
    big_stack(|| {
        let dir = Scratch::new("o200-softreset");
        let m = Origin200Machine::new(&config(&dir.0)).unwrap();
        let cpu = m.cpu_at(0);
        cpu.step_n_inline(5).unwrap();
        assert_ne!(cpu.state_digest().unwrap().pc, 0xffff_ffff_bfc0_0000, "the processor has left the reset vector");
        let word = unsafe { &*cpu.interrupts_ptr() };
        assert_eq!(word.load(Ordering::SeqCst) & crate::cpu::mips_exec::SOFT_RESET_BIT, 0);
        let io = m.node().hub.claim(Space::Io);
        assert_eq!(io.write64(crate::dev::hub::IO_REGISTERS_LOCAL + hub_pi::SOFTRESET, 1), BUS_OK);
        assert_ne!(word.load(Ordering::SeqCst) & crate::cpu::mips_exec::SOFT_RESET_BIT, 0, "the write raised the line");
        cpu.step_n_inline(1).unwrap();
        assert_eq!(word.load(Ordering::SeqCst) & crate::cpu::mips_exec::SOFT_RESET_BIT, 0, "the processor took it");
        assert_eq!(cpu.state_digest().unwrap().pc, 0xffff_ffff_bfc0_0000);
    });
}

#[test]
fn the_flash_carries_the_products_machine_type_and_cycle_counts() {
    big_stack(|| {
        let dir = Scratch::new("o200-flash");
        let m = Origin200Machine::new(&config(&dir.0)).unwrap();
        let word = |at: u64| m.node().hub.cpu_special().read32(0x1000_0000 + at).data;
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
        let dir = Scratch::new("o200-scratch");
        let mut cfg = config(&dir.0);
        let disk = dir.0.join("scratch.raw");
        cfg.scsi.clear();
        cfg.scsi.insert(2, toml::from_str(&format!("path = {:?}\nscratch = true\nsize_mb = 4", disk.to_string_lossy())).unwrap());
        let m = Origin200Machine::new(&cfg).unwrap();
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

/// The controller is visited when it says it is due and not before, a change
/// to its schedule re-arms the one timer rather than adding another, and a
/// stopped machine holds no timer.
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
        let dir = Scratch::new("o200-timer");
        let mut m = Origin200Machine::new(&config(&dir.0)).unwrap();
        assert!(m.watch.armed().is_none());
        m.restart_peripherals();
        assert_eq!(m.watch.armed().is_some(), m.io.controller.next_due().is_some());
        m.stop();
        assert!(m.watch.armed().is_none());
    });
}

/// Every dropped machine frees its Hub, its memory decode table and its timer
/// manager. The process holds a few dozen decode tables, so a leak here ends
/// a test run.
#[test]
fn a_dropped_machine_frees_its_hub_memory_and_timers() {
    big_stack(|| {
        let dir = Scratch::new("o200-free");
        let m = Origin200Machine::new(&config(&dir.0)).unwrap();
        // A single node has nothing that holds its Hub from its own memory
        // map. Route the other node numbers through the Hub, as a board with
        // a cable does, so that only the release in `Drop` can free it.
        m.node().memory.route_nodes(m.node().hub.claim(Space::Memory));
        let hub = Arc::downgrade(&m.node().hub);
        let memory = Arc::downgrade(m.node().memory.physical());
        let timers = Arc::downgrade(&m.node().timers);
        let cpu = Arc::downgrade(m.cpu_at(0));
        let bridge = Arc::downgrade(m.bridge());
        drop(m);
        assert_eq!(hub.strong_count(), 0, "the Hub");
        assert_eq!(memory.strong_count(), 0, "the memory decode table");
        assert_eq!(timers.strong_count(), 0, "the timer manager");
        assert_eq!(cpu.strong_count(), 0, "the processor");
        assert_eq!(bridge.strong_count(), 0, "the Bridge");
    });
}

/// The Hub publishes straight into its processor's interrupt word, so the
/// machine lets go of the word before the processor goes: a cabled Hub, or any
/// other holder of this one, can still raise an interrupt afterwards.
#[test]
fn a_hub_that_outlives_its_machine_publishes_into_no_freed_processor() {
    big_stack(|| {
        let dir = Scratch::new("o200-interrupt-word");
        let m = Origin200Machine::new(&config(&dir.0)).unwrap();
        let hub = m.node().hub.clone();
        let cpu = Arc::downgrade(m.cpu_at(0));
        assert!(hub.publishes_to_cpu(0), "the machine wired the processor's word");
        drop(m);
        assert_eq!(cpu.strong_count(), 0, "the processor is freed with the machine");
        assert!(!hub.publishes_to_cpu(0), "and the Hub no longer holds its word");
        // Raising and unmasking an interrupt now publishes nowhere.
        assert_eq!(hub.write64(hub_pi::INT_MASK0_A, !0), BUS_OK);
        hub.interrupt_target().set_line(3, true);
    });
}

/// The PROM of a board's second node keeps its stack in uncached Hub-special
/// memory above the 4 KiB cached alias it programs, and renumbers the node
/// between saving its return address there and reloading it. The Hub-special
/// space and the memory claimed by the node's own number stay the node's own
/// memory; only node zero's window beyond the alias leaves the node, for the
/// Hub that holds zero, of which there is none here.
#[test]
fn renumbering_moves_only_node_zero_beyond_the_alias_off_the_node() {
    big_stack(|| {
        let dir = Scratch::new("o200-renumber");
        let m = Origin200Machine::new(&config(&dir.0)).unwrap();
        let hub = &m.node().hub;
        m.node().memory.route_nodes(hub.claim(Space::Memory));
        const SLOT: u64 = 0x1bef_e78;
        assert_eq!(hub.write64(hub_pi::CALIAS_SIZE, 1), BUS_OK);
        assert_eq!(hub.cpu_special().write64(SLOT, 0x5a5a), BUS_OK);
        assert_eq!(hub.write64(hub_ni::STATUS, 1 << 8), BUS_OK);
        assert_eq!(hub.cpu_special().read64(SLOT).data, 0x5a5a, "the stack slot");
        assert_eq!(hub.claim(Space::Memory).read64((1 << 32) + SLOT).data, 0x5a5a, "node 1 by number");
        assert_ne!(m.node().memory.physical().read64(SLOT).status, BUS_OK, "cached, node zero");
        assert_ne!(hub.cpu_memory().read64(SLOT).status, BUS_OK, "uncached memory, node zero");
    });
}

/// A store-conditional is a compare-and-swap once processors share memory,
/// and a cached reference to a node by number reaches it through the Hub.
/// Without one there, the processor would fall back to a plain store and two
/// nodes could both take the same lock.
#[test]
fn a_reference_by_node_number_can_compare_and_swap() {
    big_stack(|| {
        let dir = Scratch::new("o200-cas");
        let m = Origin200Machine::new(&config(&dir.0)).unwrap();
        let hub = &m.node().hub;
        m.node().memory.route_nodes(hub.claim(Space::Memory));
        assert_eq!(hub.write64(hub_ni::STATUS, 1 << 8), BUS_OK);
        let phys = m.node().memory.physical();
        const WORD: u64 = (1 << 32) + 0x61_c000;
        assert_eq!(phys.write64(WORD, 0x99), BUS_OK);
        assert_eq!(phys.cas64(WORD, 0x99, 0x9a), Some(true));
        assert_eq!(phys.cas64(WORD, 0x99, 0x9b), Some(false));
        assert_eq!(phys.cas32(WORD + 4, 0x9a, 0x9c), Some(true));
        assert_eq!(hub.cpu_special().read64(0x61_c000).data, 0x9c);
    });
}
