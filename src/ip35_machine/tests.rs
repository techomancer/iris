use super::origin3200::{Fitting, IBRICK};
use super::*;
use crate::config::{MachineConfig, MachineProfile, NetMode};
use crate::test_temp::TempPath;
use crate::traits::{BUS_ERR, BUS_OK};
use std::sync::Weak;

/// A PROM stand-in: a recognisable first word.
fn prom() -> Vec<u8> {
    let mut p = vec![0u8; 0x1000];
    p[..4].copy_from_slice(&0x3c08_a600u32.to_be_bytes());
    p
}

fn config(dir: &Path, c_bricks: u8) -> MachineConfig {
    let mut cfg = MachineConfig::default();
    cfg.machine.profile = MachineProfile::Origin3200Ip35;
    cfg.origin3200.c_bricks = c_bricks;
    cfg.nvram = dir.join("nvram.bin").to_string_lossy().into_owned();
    cfg.headless = true;
    cfg.network.mode = NetMode::Off;
    cfg.serial_port_a = Some(0);
    cfg.serial_port_b = Some(0);
    cfg
}

fn chassis(dir: &Path, c_bricks: u8) -> Chassis<Origin3200> {
    Chassis::<Origin3200>::build(&config(dir, c_bricks), &prom(), Arc::new(|| 0)).unwrap()
}

/// A PIO address: the widget in bits 27:24.
fn widget(widget: u64, offset: u64) -> u64 {
    (widget << 24) | offset
}

/// One and two C-bricks build and are freed: nothing the chips, their
/// registered decodes, the windows or the I-brick hold keeps a chip alive.
#[test]
fn one_and_two_c_bricks_build_and_are_freed() {
    for count in [1u8, 2] {
        let dir = TempPath::dir("o3k-free");
        let chassis = chassis(&dir, count);
        assert_eq!(chassis.nodes.len(), usize::from(count));
        assert_eq!(chassis.windows.is_some(), count == 2);
        let chips: Vec<Weak<Bedrock>> = chassis.nodes.iter().map(|n| Arc::downgrade(&n.bedrock)).collect();
        let windows = chassis.windows.as_ref().map(Arc::downgrade);
        drop(chassis);
        assert!(chips.iter().all(|c| c.upgrade().is_none()), "{count} C-bricks: a chip is held by a ring");
        assert!(windows.is_none_or(|w| w.upgrade().is_none()));
    }
}

/// The I-brick's slots read back as the row says: the Fibre Channel adapter
/// at device 1, the IOC3 at 4 and the USB controller at 5 of head F, nothing
/// anywhere on head E, and nothing at the second host port.
#[test]
fn the_ibricks_slots_read_back_as_the_row_says() {
    let dir = TempPath::dir("o3k-slots");
    let chassis = chassis(&dir, 1);
    let pio = chassis.nodes[0].bedrock.pio();
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
}

/// Each node's processors fetch the reset PROM from their own node's flash
/// at the uncached-fetch window, which the node routing would otherwise read
/// as node number 1.
#[test]
fn the_reset_fetch_window_stays_on_each_node() {
    let dir = TempPath::dir("o3k-fetch");
    let chassis = chassis(&dir, 2);
    for (bus, node) in chassis.processors.iter().zip(&chassis.nodes) {
        let r = bus.read32(crate::ip35::RESET_WINDOW_FETCH);
        assert_eq!((r.status, r.data), (BUS_OK, 0x3c08_a600));
        assert_eq!(node.bus.read32(crate::ip35::RESET_WINDOW_FETCH).data, 0x3c08_a600);
    }
}

/// A processor's thread is a processor of the node it runs on and of no
/// other: the second node's chip numbers the machine's CPU 1 its own
/// processor 0, and takes the first node's thread for no processor.
#[test]
fn a_node_numbers_only_its_own_processors() {
    let dir = TempPath::dir("o3k-cpus");
    let chassis = chassis(&dir, 2);
    let second = chassis.specs[1].processors.with_first_cpu(1);
    assert_eq!((second.local(0), second.local(1), second.local(2)), (None, Some(0), None));
    assert_eq!(second.slot_of_cpu(1), Some((0, 0)));
    assert_eq!(chassis.specs[0].processors.local(1), None);
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
    crate::display_stop_tests::big_stack(|| {
        use crate::gdb_stub::StopReason;
        use gdbstub_arch::mips::reg::id::MipsRegId;
        let dir = TempPath::dir("o3k-bank512");
        let mut cfg = config(&dir, 1);
        let path = dir.join("prom.bin");
        std::fs::write(&path, prom()).unwrap();
        cfg.prom = path.to_string_lossy().into_owned();
        (cfg.origin3200.l1_port, cfg.serial_port_a, cfg.serial_port_b) = (Some(0), Some(0), Some(0));
        cfg.banks = [512, 512, 0, 0, 0, 0, 0, 0];
        cfg.banks_set = true;
        let m = Origin3200Machine::new(&cfg).unwrap();
        let node = &m.units[0].node;
        // Banks 0 and 1 enabled, size code 4 (512 MiB): slot 0 holds RAM past 0x1fc00000.
        assert_eq!(node.bedrock.registers().write64(crate::dev::bedrock::reg::MD_BASE + 8, 0x43), BUS_OK);
        assert!(node.cached.read32(0x1fc0_0000).status == BUS_OK, "RAM answers the PROM's window in the cached view");
        assert_eq!(node.cached.read32(0x1fc0_0000).data, 0, "and it is not the PROM's word");
        let cpu = m.units[0].cpu.clone().debug_adapter();
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
    let chassis = chassis(&dir, 2);
    for (n, node) in chassis.nodes.iter().enumerate() {
        // Banks 0 and 1 enabled, 256 MiB each.
        assert_eq!(node.bedrock.registers().write64(crate::dev::bedrock::reg::MD_BASE + 8, 0x33), BUS_OK);
        let at = 0x2000;
        let counter = node.cached.gen_ptr(at);
        assert!(!counter.is_null(), "node {n}");
        let read = || unsafe { (*counter).load(Ordering::Acquire) };
        let before = read();
        assert_eq!(node.cached.write32(at, 1), BUS_OK);
        let after_store = read();
        assert!(after_store > before, "node {n}: a processor store left the page's counter at {before}");
        assert_eq!(node.bedrock.dma_view().write32(at, 2), BUS_OK);
        assert!(read() > after_store, "node {n}: a DMA store left the page's counter at {after_store}");
    }
}

/// The second node's memory sits at node number 1, from 8 GiB, and the code
/// the JIT compiles from there is that node's RAM, not the reset PROM's
/// uncached-fetch copy. The second node's kernel text lies at its first
/// frame, so a PROM alias on those addresses has its processor run the PROM.
#[cfg(feature = "jitv2")]
#[test]
fn the_second_nodes_memory_is_not_hidden_by_the_proms_fetch_alias() {
    use crate::dev::bedrock::reg::{lb, LB_BASE, MD_BASE};
    let dir = TempPath::dir("o3k-alias");
    let chassis = chassis(&dir, 2);
    for node in &chassis.nodes {
        assert_eq!(node.bedrock.registers().write64(MD_BASE + 8, 0x33), BUS_OK);
    }
    assert_eq!(chassis.nodes[1].bedrock.registers().write64(LB_BASE + lb::NODE_REGION, 1 << 32), BUS_OK);
    let word = 0x2400_0001;
    assert_eq!(chassis.nodes[1].bedrock.dma_view().write32(0, word), BUS_OK);
    let second = 1u64 << 33;
    for (n, node) in chassis.nodes.iter().enumerate() {
        let read = node.code.read32(second);
        assert_eq!((read.status, read.data), (BUS_OK, word), "node {n} reads the second node's first word");
        assert!(!node.code.gen_ptr(second).is_null(), "node {n}: the word is on a page the JIT can compile");
    }
    // And the PROM is still fetchable where it is presented.
    for node in &chassis.nodes {
        assert_eq!(node.code.read32(crate::ip35::RESET_WINDOW_FETCH).data, 0x3c08_a600);
    }
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
    let chassis = Chassis::<Origin3200>::build(&cfg, &prom(), Arc::new(|| 0)).unwrap();
    let fibre = chassis.io.fibre.as_ref().expect("the adapter is on the I-brick");
    use crate::scsi_monitor::ScsiTargets;
    let attached: Vec<usize> = fibre.scsi_bus().with_targets(|t| t.iter().enumerate().filter(|(_, d)| d.is_some()).map(|(i, _)| i).collect());
    assert_eq!(attached, [3]);
    let bare = Chassis::<Origin3200>::build(&config(&dir, 1), &prom(), Arc::new(|| 0)).unwrap();
    assert!(bare.io.fibre.is_some());
}

/// A snapshot is refused with the reason, in both directions, rather than
/// saved from a machine whose chips and node memory have nothing to save.
#[test]
fn a_snapshot_is_refused_with_a_reason() {
    crate::display_stop_tests::big_stack(|| {
        let dir = TempPath::dir("o3k-snapshot");
        let mut cfg = config(&dir, 1);
        let path = dir.join("prom.bin");
        std::fs::write(&path, prom()).unwrap();
        cfg.prom = path.to_string_lossy().into_owned();
        // Port 0 takes any free host port.
        (cfg.origin3200.l1_port, cfg.serial_port_a, cfg.serial_port_b) = (Some(0), Some(0), Some(0));
        let mut m = Origin3200Machine::new(&cfg).unwrap();
        for result in [Workstation::save_snapshot(&mut m, "s"), Workstation::load_snapshot(&mut m, "s")] {
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
    crate::display_stop_tests::big_stack(|| {
        for (mode, pool_runs) in [(JitCompile::Async, true), (JitCompile::Sync, false)] {
            let dir = TempPath::dir("o3k-jitmode");
            let mut cfg = config(&dir, 1);
            let path = dir.join("prom.bin");
            std::fs::write(&path, prom()).unwrap();
            cfg.prom = path.to_string_lossy().into_owned();
            (cfg.origin3200.l1_port, cfg.serial_port_a, cfg.serial_port_b) = (Some(0), Some(0), Some(0));
            cfg.jitv2.compile = mode;
            let m = Origin3200Machine::new(&cfg).unwrap();
            for u in &m.units {
                assert_eq!(u.cpu.jitv2().lock().compile_queue.is_running(), pool_runs, "{mode:?}");
            }
            drop(m);
        }
    });
}

/// A dropped Origin 3200 frees every processor, and through their batch
/// hooks the nodes and what they hold open. Under jitv2 the machine once kept
/// a strong handle on each processor after giving its compile pool a weak one.
#[test]
fn a_dropped_origin_frees_its_processors() {
    crate::display_stop_tests::big_stack(|| {
        for c_bricks in [1u8, 2] {
            let dir = TempPath::dir("o3k-drop");
            let mut cfg = config(&dir, c_bricks);
            let path = dir.join("prom.bin");
            std::fs::write(&path, prom()).unwrap();
            cfg.prom = path.to_string_lossy().into_owned();
            (cfg.origin3200.l1_port, cfg.serial_port_a, cfg.serial_port_b) = (Some(0), Some(0), Some(0));
            let m = Origin3200Machine::new(&cfg).unwrap();
            let cpus: Vec<_> = m.units.iter().map(|u| Arc::downgrade(&u.cpu)).collect();
            let chips: Vec<Weak<Bedrock>> = m.units.iter().map(|u| Arc::downgrade(&u.node.bedrock)).collect();
            drop(m);
            assert!(cpus.iter().all(|c| c.upgrade().is_none()), "{c_bricks} C-bricks: a processor outlives the machine");
            assert!(chips.iter().all(|c| c.upgrade().is_none()), "{c_bricks} C-bricks: a node's chip outlives the machine");
        }
    });
}
