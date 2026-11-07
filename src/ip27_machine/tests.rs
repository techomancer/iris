use super::*;
use crate::config::{GraphicsBoard, NetMode};
use crate::dev::hub_pi;
use crate::traits::BUS_OK;

struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("iris-o200-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A PROM stand-in: a recognisable first word and a configuration block in
/// the shape the flash initialiser patches.
fn prom(dir: &Path) -> String {
    let mut p = vec![0u8; 0x1000];
    p[..4].copy_from_slice(&0x3c08_a600u32.to_be_bytes());
    p[0x68..0x70].copy_from_slice(&0x6970_3237_636f_6e66u64.to_be_bytes());
    let path = dir.join("prom.bin");
    std::fs::write(&path, p).unwrap();
    path.to_string_lossy().into_owned()
}

fn config(dir: &Path) -> MachineConfig {
    let mut cfg = MachineConfig::default();
    cfg.machine.profile = MachineProfile::Origin200Ip29;
    cfg.set_cpu(&parts::R10000_180);
    cfg.banks = [64, 0, 0, 0, 0, 0, 0, 0];
    cfg.prom = prom(dir);
    cfg.nvram = dir.join("nvram.bin").to_string_lossy().into_owned();
    cfg.headless = true;
    cfg.network.mode = NetMode::Off;
    let held: Vec<_> = (0..3).map(|_| std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap()).collect();
    cfg.monitor_port = Some(held[0].local_addr().unwrap().port());
    cfg.serial_port_a = Some(held[1].local_addr().unwrap().port());
    cfg.serial_port_b = Some(held[2].local_addr().unwrap().port());
    let disk = dir.join("disk.raw");
    std::fs::write(&disk, vec![0u8; 4 << 20]).unwrap();
    cfg.scsi.clear();
    cfg.scsi.insert(1, toml::from_str(&format!("path = {:?}\noverlay = true", disk.to_string_lossy())).unwrap());
    cfg
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
    let dir = Scratch::new("config");
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

/// Building a machine puts more than a megabyte on the stack.
fn big_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().stack_size(64 << 20).spawn(body).unwrap().join().expect("thread panicked");
}

#[test]
fn the_reset_vector_reaches_the_flash_and_the_processor_is_enabled() {
    big_stack(|| {
        let dir = Scratch::new("reset");
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
        let enable = crate::dev::hub::IO_REGISTERS_LOCAL + hub_pi::reg::CPU_ENABLE_A;
        assert_eq!(io.read64(enable).data, 1);
        // And its slice B, which a one-processor module does not fit, is clear.
        assert_eq!(io.read64(crate::dev::hub::IO_REGISTERS_LOCAL + hub_pi::reg::CPU_ENABLE_B).data, 0);
    });
}

#[test]
fn the_bridge_answers_through_the_hub_at_every_widget_but_its_own() {
    big_stack(|| {
        let dir = Scratch::new("bridge");
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
        let dir = Scratch::new("restart");
        let mut m = Origin200Machine::new(&config(&dir.0)).unwrap();
        m.reset();
        let io = m.node().hub.claim(Space::Io);
        assert_eq!(io.read64(crate::dev::hub::IO_REGISTERS_LOCAL + hub_pi::reg::CPU_ENABLE_A).data, 1);
        m.stop();
    });
}

#[test]
fn the_flash_carries_the_products_machine_type_and_cycle_counts() {
    big_stack(|| {
        let dir = Scratch::new("flash");
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
        let dir = Scratch::new("scratch");
        let mut cfg = config(&dir.0);
        let disk = dir.0.join("scratch.raw");
        cfg.scsi.clear();
        cfg.scsi.insert(2, toml::from_str(&format!("path = {:?}\nscratch = true\nsize_mb = 4", disk.to_string_lossy())).unwrap());
        let m = Origin200Machine::new(&cfg).unwrap();
        assert_eq!(m.scratch_path(), Some(disk.as_path()));
    });
}

#[test]
fn the_controller_is_brought_up_to_date_by_a_timer_and_stopped_with_the_machine() {
    big_stack(|| {
        let dir = Scratch::new("timer");
        let mut m = Origin200Machine::new(&config(&dir.0)).unwrap();
        assert!(m.controller_timer.is_none());
        m.restart_peripherals();
        let first = m.controller_timer;
        assert!(first.is_some());
        m.restart_peripherals();
        assert_ne!(m.controller_timer, first, "a restart replaces the timer, never doubling it");
        m.stop();
        assert!(m.controller_timer.is_none());
    });
}

/// Every dropped machine frees its Hub, its memory decode table and its timer
/// manager. The process holds a few dozen decode tables, so a leak here ends
/// a test run.
#[test]
fn a_dropped_machine_frees_its_hub_memory_and_timers() {
    big_stack(|| {
        let dir = Scratch::new("free");
        let m = Origin200Machine::new(&config(&dir.0)).unwrap();
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
