use super::origin3200::{Fitting, IBRICK};
use super::*;
use crate::config::{MachineConfig, MachineProfile, NetMode};
use crate::cpu::parts;
use crate::test_temp::TempPath;
use crate::traits::{BUS_ERR, BUS_OK};
use std::sync::Weak;

/// A PROM stand-in: a recognisable first word.
pub(crate) fn prom() -> Vec<u8> {
    let mut p = vec![0u8; 0x1000];
    p[..4].copy_from_slice(&0x3c08_a600u32.to_be_bytes());
    p
}

/// A headless configuration of `profile` that builds in `dir`: the PROM stand-in, NVRAM in
/// the directory, no network, and every port the machine listens on left to any free one.
pub(crate) fn headless(dir: &Path, profile: MachineProfile) -> MachineConfig {
    let mut cfg = MachineConfig::default();
    cfg.machine.profile = profile;
    let prom_path = dir.join("prom.bin");
    std::fs::write(&prom_path, prom()).unwrap();
    cfg.prom = prom_path.to_string_lossy().into_owned();
    cfg.nvram = dir.join("nvram.bin").to_string_lossy().into_owned();
    cfg.headless = true;
    cfg.network.mode = NetMode::Off;
    cfg.monitor_port = Some(0);
    cfg.serial_port_a = Some(0);
    cfg.serial_port_b = Some(0);
    cfg
}

/// Build the machine `cfg` describes on a big stack, as `main` does, run `body` on it, and
/// drop it there.
pub(crate) fn with_machine<R: Send + 'static>(cfg: MachineConfig, body: impl FnOnce(&mut Ip35Machine) -> R + Send + 'static) -> R {
    crate::display_stop_tests::big_stack(move || body(&mut Ip35Machine::new(&cfg).unwrap()))
}

fn config(dir: &Path, c_bricks: u8) -> MachineConfig {
    let mut cfg = headless(dir, MachineProfile::Origin3200Ip35);
    cfg.origin3200.c_bricks = c_bricks;
    cfg.origin3200.l1_port = Some(0);
    cfg
}

/// A Fuel with the volatile boot flash and its listening ports left to any free one.
pub(crate) fn fuel_config(dir: &Path) -> MachineConfig {
    let mut cfg = headless(dir, MachineProfile::FuelIp34);
    cfg.banks = crate::config::banks_from4([256, 256, 0, 0]);
    cfg.banks_set = true;
    cfg.fuel.l1_port = Some(0);
    cfg.scsi.clear();
    cfg
}

/// A Tezro of `cpus` processors, its listening ports left to any free one.
pub(crate) fn tezro_config(dir: &Path, cpus: u8) -> MachineConfig {
    let mut cfg = headless(dir, MachineProfile::TezroIp53);
    cfg.machine.cpus = Some(cpus);
    cfg.tezro.l1_port = Some(0);
    (cfg.tezro.serial_port_c, cfg.tezro.serial_port_d) = (Some(0), Some(0));
    cfg.scsi.clear();
    cfg
}

/// A PCI head as a processor reaches it through PIO: configuration cycles at the head's
/// widget, and a device's window onto PCI memory.
pub(crate) struct Head {
    pio: Arc<dyn BusDevice>,
    widget: u64,
}

impl Head {
    /// Head `widget` of the node whose PIO window is `pio`.
    pub(crate) fn new(pio: Arc<dyn BusDevice>, widget: u8) -> Self {
        Self { pio, widget: u64::from(widget) }
    }

    fn at(&self, offset: u64) -> u64 {
        (self.widget << 24) | offset
    }

    pub(crate) fn cfg_read(&self, device: u32, reg: u32) -> u32 {
        self.pio.read32(self.at(u64::from(crate::dev::bridge::CFG0_BASE + device * 0x1000 + reg))).data
    }

    pub(crate) fn cfg_write(&self, device: u32, reg: u32, value: u32) {
        assert_eq!(self.pio.write32(self.at(u64::from(crate::dev::bridge::CFG0_BASE + device * 0x1000 + reg)), value), BUS_OK);
    }

    /// Write the head's widget register at `offset` (a doubleword register's low word).
    pub(crate) fn set(&self, offset: u32, value: u32) {
        assert_eq!(self.pio.write32(self.at(u64::from(offset) + 4), value), BUS_OK);
    }

    /// Put the function at `device`'s memory BAR `bar` (a configuration offset) at PCI memory
    /// `base`, enable its memory space and aim the device's window there. Returns the window.
    pub(crate) fn map_bar(&self, device: u32, bar: u32, base: u32) -> Window {
        use crate::dev::bridge::{device_window_offset, reg, DEV_MEMORY_SPACE};
        self.cfg_write(device, bar, base);
        self.cfg_write(device, 0x04, 2);
        self.set(reg::device(device), DEV_MEMORY_SPACE | (base >> 20));
        Window { pio: self.pio.clone(), at: self.at(u64::from(device_window_offset(device))) }
    }
}

/// A device's window onto PCI memory, with the bridge's byte lanes at their reset setting:
/// a register is read at the address the device decodes.
pub(crate) struct Window {
    pio: Arc<dyn BusDevice>,
    at: u64,
}

impl Window {
    pub(crate) fn read32(&self, offset: u64) -> u32 {
        let r = self.pio.read32(self.at + offset);
        assert_eq!(r.status, BUS_OK, "read of {offset:#x}");
        r.data
    }

    pub(crate) fn write32(&self, offset: u64, value: u32) {
        assert_eq!(self.pio.write32(self.at + offset, value), BUS_OK, "write of {offset:#x}");
    }

    /// A halfword register, reached on the lane its address names.
    pub(crate) fn read16(&self, offset: u64) -> u32 {
        let r = self.pio.read16(self.at + (offset ^ 2));
        assert_eq!(r.status, BUS_OK, "read of {offset:#x}");
        u32::from(r.data)
    }

    pub(crate) fn write16(&self, offset: u64, value: u16) {
        assert_eq!(self.pio.write16(self.at + (offset ^ 2), value), BUS_OK, "write of {offset:#x}");
    }
}

/// A PIO address: the widget in bits 27:24.
fn widget(widget: u64, offset: u64) -> u64 {
    (widget << 24) | offset
}

/// The I-brick's slots read back as the row says: the Fibre Channel adapter
/// at device 1, the IOC3 at 4 and the USB controller at 5 of head F, nothing
/// anywhere on head E, and nothing at the second host port.
#[test]
fn the_ibricks_slots_read_back_as_the_row_says() {
    let dir = TempPath::dir("o3k-slots");
    with_machine(config(&dir, 1), |m| {
        let pio = m.nodes[0].chip.pio();
        let [head_e, head_f] = IBRICK.heads;
        let id = |head: u8, device: u64| pio.read32(widget(u64::from(head), 0x2_0000 + (device << 12)));
        for &(device, fitting) in IBRICK.head_f {
            let read = id(head_f, u64::from(device));
            match fitting {
                Fitting::Ioc3(_) => assert_eq!(read.data, 0x0003_10a9, "device {device}"),
                Fitting::Usb => assert_eq!(read.data, 0x5802_11c1, "device {device}"),
                Fitting::FibreChannel => assert_eq!(read.data, 0x2200_1077, "device {device}"),
            }
        }
        for device in 0..8 {
            let read = id(head_e, device);
            assert!(read.status != BUS_OK || read.data == 0xffff_ffff, "head E device {device} is empty");
        }
        assert_eq!(pio.read32(widget(u64::from(IBRICK.spare_port), 0)).status, BUS_ERR, "port B is empty");
    });
}

/// A processor's thread is a processor of the node it runs on and of no
/// other: the second node's chip numbers the machine's CPU 1 its own
/// processor 0, and takes the first node's thread for no processor.
#[test]
fn a_node_numbers_only_its_own_processors() {
    let dir = TempPath::dir("o3k-cpus");
    let specs = specs(&config(&dir, 2)).unwrap();
    let second = specs[1].processors.with_first_cpu(1);
    assert_eq!((second.local(0), second.local(1), second.local(2)), (None, Some(0), None));
    assert_eq!(second.slot_of_cpu(1), Some((0, 0)));
    assert_eq!(specs[0].processors.local(1), None);
}

fn refused(cfg: &MachineConfig) -> String {
    cfg.validate().unwrap_err()
}

/// The row's limits: one or two C-bricks, banks a C-brick takes in matched
/// pairs, disks on the loop and not on `[scsi.N]`, no option board keys, no
/// graphics.
#[test]
fn the_configuration_refuses_what_the_row_does_not_fit() {
    let dir = TempPath::dir("o3k-config");
    config(&dir, 1).validate().unwrap();
    config(&dir, 2).validate().unwrap();
    assert!(refused(&config(&dir, 3)).contains("origin3200.c_bricks"));
    let mut cfg = config(&dir, 1);
    cfg.banks = [512, 512, 0, 0, 0, 0, 0, 0];
    cfg.banks_set = true;
    // A 512 MiB bank can sit among banks 0 to 3 and cover the PROM's window.
    cfg.validate().unwrap();
    cfg.banks = [256, 256, 256, 256, 512, 512, 0, 0];
    cfg.validate().unwrap();
    cfg.banks = [512, 256, 0, 0, 0, 0, 0, 0];
    assert!(refused(&cfg).contains("share one size"), "{}", refused(&cfg));
    cfg.banks = [256, 128, 0, 0, 0, 0, 0, 0];
    assert!(refused(&cfg).contains("256 or 512"));
    let fc = |extra: &str| -> MachineConfig {
        let mut cfg = config(&dir, 1);
        cfg.fibre_channel = Some(toml::from_str(extra).unwrap());
        cfg
    };
    fc("[disk.1]\npath = \"d.img\"").validate().unwrap();
    for key in ["board = 0", "slot = 1", "bridge = true"] {
        assert!(refused(&fc(key)).contains("on the I-brick"), "{key}");
    }
    assert!(refused(&fc("[disk.126]\npath = \"d.img\"")).contains("loop ID"));
    let mut cfg = config(&dir, 1);
    cfg.scsi.insert(2, toml::from_str("path = \"d.raw\"").unwrap());
    assert!(refused(&cfg).contains("Fibre Channel"));
    let mut cfg = config(&dir, 1);
    cfg.graphics.board = Some(crate::config::GraphicsBoard::V12);
    assert!(refused(&cfg).contains("Origin 3200"));
}

/// `r12000` and `r12000_400` name the C-brick's processor, the 400 MHz part
/// with 8 MiB of secondary cache; the Octane's 2 MiB part is not offered.
#[test]
fn the_c_brick_takes_the_400_mhz_r12000_with_the_8_mib_cache() {
    for name in [None, Some("r12000"), Some("r12000_400")] {
        assert_eq!(MachineProfile::Origin3200Ip35.part(name).unwrap(), &parts::R12000_400_8M, "{name:?}");
    }
    assert!(MachineProfile::Origin3200Ip35.part(Some("r12000_360")).is_err());
}

/// With a 512 MiB bank 0 the memory covers the reset PROM's window. Once the
/// controller maps it, the processor still fetches the PROM's first word at
/// the reset vector, where the cached plane answers RAM.
#[test]
fn the_prom_still_fetches_under_a_512_mib_bank() {
    use crate::gdb_stub::StopReason;
    use gdbstub_arch::mips::reg::id::MipsRegId;
    let dir = TempPath::dir("o3k-bank512");
    let mut cfg = config(&dir, 1);
    cfg.banks = [512, 512, 0, 0, 0, 0, 0, 0];
    cfg.banks_set = true;
    with_machine(cfg, |m| {
        let node = &m.nodes[0];
        // Banks 0 and 1 enabled, size code 4 (512 MiB): slot 0 holds RAM past 0x1fc00000.
        assert_eq!(node.chip.registers().write64(crate::dev::bedrock::regs::md::BASE + 8, 0x43), BUS_OK);
        assert!(node.chip.decode.physical().read32(0x1fc0_0000).status == BUS_OK, "RAM answers the PROM's window in the cached view");
        assert_eq!(node.chip.decode.physical().read32(0x1fc0_0000).data, 0, "and it is not the PROM's word");
        let cpu = m.node_cpus[0][0].clone().debug_adapter();
        assert_eq!(cpu.step_one(), StopReason::DoneStep);
        let t0 = cpu.read_reg(MipsRegId::Gpr(8)).unwrap();
        assert_eq!(t0, 0xffff_ffff_a600_0000, "the first instruction was the PROM's lui, not RAM's zero word");
    });
}

/// A page compiled on a node of two watches the counter the banks bump. The
/// processor's stores and a bus master's both reach the banks, so both must
/// advance the counter the JIT was handed, or code written after it was
/// compiled keeps running.
#[cfg(feature = "jitv2")]
#[test]
fn a_two_node_machines_code_pages_see_processor_and_dma_stores() {
    use std::sync::atomic::Ordering;
    let dir = TempPath::dir("o3k-gen");
    with_machine(config(&dir, 2), |m| {
        for (n, node) in m.nodes.iter().enumerate() {
            // Banks 0 and 1 enabled, 256 MiB each.
            assert_eq!(node.chip.registers().write64(crate::dev::bedrock::regs::md::BASE + 8, 0x33), BUS_OK);
            let at = 0x2000;
            let counter = node.chip.decode.physical().gen_ptr(at);
            assert!(!counter.is_null(), "node {n}");
            let read = || unsafe { (*counter).load(Ordering::Acquire) };
            let before = read();
            assert_eq!(node.chip.decode.physical().write32(at, 1), BUS_OK);
            let after_store = read();
            assert!(after_store > before, "node {n}: a processor store left the page's counter at {before}");
            assert_eq!(node.chip.decode.physical().write32(at, 2), BUS_OK);
            assert!(read() > after_store, "node {n}: a DMA store left the page's counter at {after_store}");
        }
    });
}

/// The `[fibre_channel.disk.N]` disks are on the I-brick's loop at the loop IDs
/// they name, and a machine with none still has the adapter.
#[test]
fn the_loop_has_the_configured_disks() {
    let dir = TempPath::dir("o3k-loop");
    let image = dir.join("d.raw");
    std::fs::write(&image, vec![0u8; 1 << 20]).unwrap();
    let mut cfg = config(&dir, 1);
    cfg.fibre_channel = Some(toml::from_str(&format!("[disk.3]\npath = {:?}\noverlay = true", image.to_string_lossy())).unwrap());
    let attached = |m: &mut Ip35Machine| {
        let mut attached = Vec::new();
        (m.io.disks)(&mut |buses| {
            for bus in buses {
                attached.push(bus.with_targets(|t| t.iter().enumerate().filter(|(_, d)| d.is_some()).map(|(i, _)| i).collect::<Vec<_>>()));
            }
        });
        attached
    };
    assert_eq!(with_machine(cfg, attached), [vec![3]]);
    assert_eq!(with_machine(config(&dir, 1), attached), [Vec::<usize>::new()], "the adapter is on the I-brick");
}

/// A snapshot is refused with the reason, in both directions, rather than
/// saved from a machine whose chips and node memory have nothing to save.
#[test]
fn a_snapshot_is_refused_with_a_reason() {
    let dir = TempPath::dir("o3k-snapshot");
    with_machine(config(&dir, 1), |m| {
        for result in [Workstation::save_snapshot(m, "s"), Workstation::load_snapshot(m, "s")] {
            let why = result.unwrap_err();
            assert!(why.contains("Origin 3200") && why.contains("not supported yet") && why.contains("no saved state"), "{why}");
        }
    });
}

/// `[jitv2] compile` reaches the machine: `async` starts each node's compile
/// pool and `sync` leaves it stopped, for the processor to compile inline. The
/// machine is then dropped, which must join the pool rather than hang.
#[cfg(all(feature = "jitv2", not(feature = "jitv2_lockstep")))]
#[test]
fn the_jitv2_compile_setting_reaches_the_machine_and_drop_joins_the_pool() {
    use crate::config::JitCompile;
    for (mode, pool_runs) in [(JitCompile::Async, true), (JitCompile::Sync, false)] {
        let dir = TempPath::dir("o3k-jitmode");
        let mut cfg = config(&dir, 1);
        cfg.jitv2.compile = mode;
        with_machine(cfg, move |m| {
            for cpus in &m.node_cpus {
                assert_eq!(cpus[0].jitv2().lock().compile_queue.is_running(), pool_runs, "{mode:?}");
            }
        });
    }
}

/// The Fuel and the Tezro honour `[jitv2] compile` too. Each started its pool
/// directly and compiled on it whatever the setting said.
#[cfg(all(feature = "jitv2", not(feature = "jitv2_lockstep")))]
#[test]
fn the_fuel_and_the_tezro_honour_the_jitv2_compile_setting() {
    use crate::config::JitCompile;
    let pools = |cfg: MachineConfig| with_machine(cfg, |m| m.cpus().map(|c| c.jitv2().lock().compile_queue.is_running()).collect::<Vec<_>>());
    for (mode, pool_runs) in [(JitCompile::Async, true), (JitCompile::Sync, false)] {
        let dir = TempPath::dir("ip35-jitmode");
        let mut fuel = fuel_config(&dir);
        fuel.jitv2.compile = mode;
        assert_eq!(pools(fuel), [pool_runs], "the Fuel, {mode:?}");
        let mut tezro = tezro_config(&dir, 1);
        tezro.jitv2.compile = mode;
        assert_eq!(pools(tezro), [pool_runs], "the Tezro, {mode:?}");
    }
}

/// A dropped machine frees every processor, and through their batch hooks
/// the nodes and the boards and what they hold open: the Origin 3200 of one
/// and two C-bricks, the Fuel and the Tezro of one and four processors.
/// Under jitv2 a machine once kept a strong handle on each processor after
/// giving its compile pool a weak one, so every file a board had open stayed
/// open.
#[test]
fn a_dropped_machine_frees_its_processors() {
    fn dropped(cfg: MachineConfig, what: &str) {
        let (cpus, chips) = with_machine_dropped(cfg, what);
        assert!(cpus.iter().all(|c| c.upgrade().is_none()), "{what}: a processor outlives the machine");
        assert!(chips.iter().all(|c| c.upgrade().is_none()), "{what}: a node's chip outlives the machine");
    }
    let dir = TempPath::dir("ip35-drop");
    for c_bricks in [1u8, 2] {
        dropped(config(&dir, c_bricks), &format!("{c_bricks} C-bricks"));
    }
    dropped(fuel_config(&dir), "the Fuel");
    for cpus in [1u8, 4] {
        dropped(tezro_config(&dir, cpus), &format!("a Tezro of {cpus}"));
    }
}

/// Build the machine, drop it on its big stack, and hand back weak handles on what it held.
#[allow(clippy::type_complexity)]
fn with_machine_dropped(cfg: MachineConfig, what: &str) -> (Vec<Weak<dyn CpuDevice>>, Vec<Weak<crate::dev::bedrock::Bedrock>>) {
    let what = what.to_string();
    crate::display_stop_tests::big_stack(move || {
        let m = Ip35Machine::new(&cfg).unwrap();
        // A second node takes number one, so each node's table points into
        // the other's banks when the machine goes.
        if let [a, b] = &m.nodes[..] {
            use crate::dev::bedrock::regs::{lb, md};
            for node in [a, b] {
                assert_eq!(node.chip.registers().write64(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x33), crate::traits::BUS_OK);
            }
            assert_eq!(b.chip.registers().write64(lb::BASE + lb::REV_ID::OFFSET, 1 << 32), crate::traits::BUS_OK);
            assert_eq!(a.chip.decode.physical().read64(1 << 33).status, crate::traits::BUS_OK, "{what}: node 0 shows node 1's memory");
        }
        let cpus: Vec<Weak<dyn CpuDevice>> = m.cpus().map(Arc::downgrade).collect();
        let chips: Vec<Weak<crate::dev::bedrock::Bedrock>> = m.nodes.iter().map(|n| Arc::downgrade(&n.chip)).collect();
        drop(m);
        (cpus, chips)
    })
}
