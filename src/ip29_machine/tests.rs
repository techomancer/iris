use super::*;
use crate::config::{GraphicsBoard, MachineProfile};
use crate::dev::hub::regs as hub_regs;
use crate::ip27::test_support::{self, big_stack};
use crate::test_temp::TempPath;
use crate::traits::{BusDevice, BUS_OK};
use crate::workstation::Workstation;

fn config(dir: &Path) -> MachineConfig {
    test_support::config(MachineProfile::Origin200Ip29, dir, 1)
}

#[test]
fn the_r12000_module_has_no_record_on_its_one_wire_part() {
    let dir = TempPath::dir("ip27-o200-node-record");
    let mut cfg = config(&dir);
    cfg.set_cpu(&parts::R12000_270);
    assert!(node_specs(&cfg).unwrap()[0].node_record.is_none());
    cfg.set_cpu(&parts::R10000_180);
    assert_eq!(node_specs(&cfg).unwrap()[0].node_record.unwrap().name, "PIMM_1XT5_1MB");
}

#[test]
fn the_profile_takes_an_r10000_or_r12000_and_fills_banks_from_zero() {
    let dir = TempPath::dir("ip27-o200-config");
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
    cfg.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::V12)));
    assert!(cfg.validate().unwrap_err().contains("no free slot for a graphics board on the Origin 200"));
}

#[test]
fn the_reset_vector_reaches_the_flash_and_the_processor_is_enabled() {
    big_stack(|| {
        let dir = TempPath::dir("ip27-o200-reset");
        let mut m = test_support::machine(&config(&dir)).unwrap();
        // Node memory view, the Hub-special views of the flash (own and named),
        // all show the PROM's first word at the reset address and the window.
        let memory = m.nodes[0].chip.decode.physical();
        assert_eq!(memory.read32(0x1fc0_0000).data, 0x3c08_a600);
        let special = m.nodes[0].chip.cpu_view(crate::sn::UncachedSpace::HubSpecial);
        assert_eq!(special.read32(0x1000_0000).data, 0x3c08_a600);
        assert_eq!(special.read32(0x3000_0000).data, 0x3c08_a600);
        // The Hub's own register: PI_CPU_ENABLE_A reads one without the PROM
        // having stored it.
        let io = m.nodes[0].chip.cpu_view(crate::sn::UncachedSpace::Io);
        let enable = crate::dev::hub::IO_REGISTERS_LOCAL + crate::dev::hub::regs::PI_CPU_ENABLE_A::OFFSET;
        assert_eq!(io.read64(enable).data, 1);
        // And its slice B, which a one-processor module does not populate, is clear.
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
        let dir = TempPath::dir("ip27-o200-bridge");
        let m = test_support::machine(&config(&dir)).unwrap();
        let io = m.nodes[0].chip.cpu_view(crate::sn::UncachedSpace::Io);
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
        let dir = TempPath::dir("ip27-o200-softreset");
        let m = test_support::machine(&config(&dir)).unwrap();
        let cpu = &m.node_cpus[0][0];
        cpu.step_n_inline(5).unwrap();
        assert_ne!(cpu.state_digest().unwrap().pc, 0xffff_ffff_bfc0_0000, "the processor has left the reset vector");
        let word = unsafe { &*cpu.interrupts_ptr() };
        assert_eq!(word.load(Ordering::SeqCst) & crate::cpu::mips_exec::SOFT_RESET_BIT, 0);
        let io = m.nodes[0].chip.cpu_view(crate::sn::UncachedSpace::Io);
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
        let dir = TempPath::dir("ip27-o200-flash");
        let m = test_support::machine(&config(&dir)).unwrap();
        let word = |at: u64| m.nodes[0].chip.cpu_view(crate::sn::UncachedSpace::HubSpecial).read32(0x1000_0000 + at).data;
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
        let dir = TempPath::dir("ip27-o200-scratch");
        let mut cfg = config(&dir);
        let disk = dir.join("scratch.raw");
        cfg.devices.clear();
        cfg.set_disk(2, toml::from_str(&format!("path = {:?}\nscratch = true\nsize_mb = 4", disk.to_string_lossy())).unwrap());
        let m = test_support::machine(&cfg).unwrap();
        assert_eq!(m.scratch_path(), Some(disk.as_path()));
    });
}


/// The PROM waits for an arbitration grant from the module controller before every exchange with
/// it, and prints "Timeout waiting for arbitration signal" when none comes. The controller grants
/// from the moment the machine is built, and a machine reset leaves it running, mid-command.
#[test]
fn the_origin_200_controller_grants_from_the_start_and_survives_a_reset() {
    use crate::dev::pcf8584::ctl;
    use crate::dev::serial_host::CiSerialBackend;
    let now = crate::test_board::Stepped::new();
    let hub = Hub::new(HubConfig { cpus: 1, ni_revision: 3, ..Default::default() }, &now.tm, None).unwrap();
    let controller = Origin200Controller::new(&hub, &now.tm, None).unwrap();
    controller.elsc.start();
    let grants = || {
        // The PROM's setup as an I2C observer: a nonzero own address, then the interface on.
        controller.i2c.write_register(8, ctl::PIN);
        controller.i2c.write_register(0, 0x09);
        controller.i2c.write_register(8, ctl::PIN | ctl::ESO);
        crate::test_board::until(|| {
            now.step(1_000_000);
            controller.i2c.read_register(0) == 0xe0
        })
    };
    assert!(grants(), "processor 0's arbitration grant never appeared");

    let console = Arc::new(CiSerialBackend::new());
    controller.elsc.set_serial_backend(console.clone());
    controller.elsc.serial_input(0x14);
    console.drain_guest();
    controller.reset();
    controller.elsc.serial_input(b'x');
    assert_eq!(console.drain_guest(), b"x", "the line being typed was echoed, so the command mode held");
    assert!(grants(), "processor 0's arbitration grant stopped after a machine reset");
}
