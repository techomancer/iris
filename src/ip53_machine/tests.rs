use crate::traits::BusDevice;
use super::*;
use crate::cpu::mips_cache_v2::{CpuModel as CacheCpuModel, MipsCache};
use crate::cpu::mips_exec::MipsExecutor;
use crate::cpu::mips_tlb::MipsTlb;
use crate::ip35::serial::connect_serial_ports;
use crate::ip35::test_support::{machine, tezro_config, widget, with_machine};
use crate::dev::bedrock::Bedrock;
use crate::dev::bedrock_flash::Flash;
use crate::dev::pic::{read_bytes, write_bytes};
use crate::sn::Machine;
use crate::test_temp::TempPath;
use crate::traits::{BUS_ERR, BUS_OK};

/// The Config bits 24:0 a module's mode word defines.
const CONFIG0_MODE_MASK: u32 = 0x01ff_ffff;

/// A Tezro of `cpus` processors with `chip` populated, in `dir`.
fn chip_config(dir: &TempPath, chip: &TezroChip, cpus: u8) -> MachineConfig {
    let mut cfg = tezro_config(dir, cpus);
    cfg.set_cpu(chip.module.part);
    cfg
}

/// A processor of type `C` with the machine chip's identity and reset-mode
/// inputs, on the node's bus, as the machine builds it.
fn build_cpu<C: CacheCpuModel + From<Arc<dyn BusDevice>>>(m: &Machine<Bedrock, Flash>, chip: &TezroChip) -> Box<MipsExecutor<MipsTlb, C>> {
    let memory = Arc::new(crate::ip35::memory_map(&m.nodes[0].chip.md.ram_extent()));
    let timers = Arc::new(crate::hptimer::TimerManager::new());
    let cached = m.nodes[0].chip.decode.physical().clone();
    crate::cpu::build::build_cpu(m.nodes[0].bus.clone(), cached, memory, chip.module, &timers, chip.module.part.count_hz(), 0, false)
}

/// A PIO offset in Bedrock IO's small window: widget in bits 27:24.
/// The II's PIO face as the processor reaches it, a `size`-byte access at
/// a time.
struct Pio(Arc<dyn BusDevice>);
impl Pio {
    fn pio_read(&self, a: u64, size: u8) -> crate::traits::BusRead64 {
        match size {
            8 => self.0.read64(a),
            4 => {
                let r = self.0.read32(a);
                crate::traits::BusRead64 { data: u64::from(r.data), status: r.status }
            }
            _ => unreachable!("tests read words and doublewords"),
        }
    }
    fn pio_write(&self, a: u64, size: u8, value: u64) -> u32 {
        match size {
            8 => self.0.write64(a, value),
            4 => self.0.write32(a, value as u32),
            _ => unreachable!("tests write words and doublewords"),
        }
    }
}
fn pio_read(m: &Machine<Bedrock, Flash>, a: u64, size: u8) -> crate::traits::BusRead64 {
    Pio(m.nodes[0].chip.pio()).pio_read(a, size)
}

/// Configuration register 0 of `device` on the IO9 PIC's head 0.
fn config_id(device: u64) -> u64 {
    widget(15, 0x2_0000 + (device << 12))
}

/// Aim the IO9 PIC's type 1 configuration window at `device` of bus `bus`.
fn aim_type1(io: &Pio, bus: u64, device: u64) {
    assert_eq!(io.pio_write(widget(15, 0xc8), 8, bus << 16 | device << 11), BUS_OK);
}

/// The processor finds the PXBOW at widget 0 and the IO9's PIC at widget 15, as the QEMU profile
/// maps them; the one-processor board has no widget 11 and no other widget answers.
#[test]
fn widget_0_is_the_pxbow_and_widget_15_the_io9_pic() {
    let dir = TempPath::dir("tezro-widgets");
    with_machine(tezro_config(&dir, 1), |m| {
        let io = Pio(m.nodes[0].chip.pio());
        let pxbow = io.pio_read(widget(0, 4), 4);
        assert_eq!(pxbow.status, BUS_OK);
        assert_eq!((pxbow.data >> 12) & 0xffff, u64::from(crate::dev::pxbow::PART));
        let pic = io.pio_read(widget(15, 0), 8);
        assert_eq!(pic.status, BUS_OK);
        assert_eq!(pic.data, u64::from(crate::dev::pic::identity(0)));
        for absent in [8, 9, 11, 12, 13, 14] {
            assert_eq!(io.pio_read(widget(absent, 0), 8).status, BUS_ERR, "widget {absent}");
        }
        // The one-processor board has no second PIC: link 11 reads empty, link 15 stays usable.
        assert_eq!(live_links(m), 1 << 10 | 1 << 15);
        assert_eq!(alive_and_present(m, 11), (false, false));
        assert!(!prom_uses(m, 11) && !kernel_uses(m, 11));
        assert!(prom_uses(m, 15) && kernel_uses(m, 15));
        assert_eq!(pio_read(m, widget(11, 4), 4).status, BUS_ERR);
    });
}

/// IO9's functions sit at PCI devices 1, 3 and 4 of head 0 with the
/// revisions Tezro consoles print (the TI bridge at device 2 has its own test).
#[test]
fn the_io9_bus_holds_the_ioc4_the_isp12160_and_a_bcm5701_on_a_64_bit_slot() {
    let dir = TempPath::dir("tezro-io9");
    with_machine(tezro_config(&dir, 1), |m| {
        let io = Pio(m.nodes[0].chip.pio());
        for (device, id, revision) in [(1, 0x100a_10a9, 0x53), (3, 0x1216_1077, 0x06), (4, 0x1645_14e4, 0x15)] {
            assert_eq!(io.pio_read(config_id(device), 4).data, id, "device {device}");
            assert_eq!(io.pio_read(config_id(device) + 8, 4).data & 0xff, revision, "device {device}");
        }
        // The BCM5701's slot drives REQ64#, so it reports a 64-bit bus, before and after a reset.
        const PCISTATE: u64 = 0x70;
        const BUS_32BIT: u64 = 1 << 4;
        assert_eq!(io.pio_read(config_id(ETH_DEVICE.into()) + PCISTATE, 4).data & BUS_32BIT, 0);
        (m.io.reset)();
        assert_eq!(io.pio_read(config_id(ETH_DEVICE.into()) + PCISTATE, 4).data & BUS_32BIT, 0);
    });
}

/// The board identity the PROM reads: Bedrock board word 0x2001 and the
/// L1's Tezro station address.
#[test]
fn the_board_reports_the_tezro_board_word() {
    let dir = TempPath::dir("tezro-board-word");
    let mac = tezro_config(&dir, 1).network().mac;
    with_machine(tezro_config(&dir, 1), move |m| {
        let id = m.nodes[0].bus.read64(0x1000_0000 + 0x400_0000);
        assert_eq!((id.status, id.data), (BUS_OK, BOARD_ID));
        assert_eq!(m.l1s[0].station_mac(), mac);
    });
}

/// The fields of an L1 EEPROM area after its six-byte header, up to its
/// end mark: each a descriptor byte whose low six bits give the length.
fn area_fields(area: &[u8]) -> Vec<&[u8]> {
    let mut fields = Vec::new();
    let mut at = 6;
    while area[at] != 0xc1 {
        let len = usize::from(area[at] & 0x3f);
        fields.push(&area[at + 1..at + 1 + len]);
        at += 1 + len;
    }
    fields
}

/// The PROM's start-up check compares the boot flash's configuration block
/// with the CPU configuration in the node board's L1 record (component 0's
/// board area): the mode word at +0x04 with Config0, the clocks at +0x10 and
/// +0x18 with the CPU and Hub MHz, and the word at +0x30 with the class it
/// derives from the board word's top nibble (0x4 gives 2, 0x8 gives 1, any other 0).
#[test]
fn the_node_board_record_agrees_with_the_flash_configuration() {
    let class = match BOARD_ID & 0xf000 {
        0x4000 => 2u32,
        0x8000 => 1,
        _ => 0,
    };
    let dir = TempPath::dir("tezro-node-board");
    for cpus in [1, 2, 4] {
        with_machine(tezro_config(&dir, cpus as u8), move |m| {
        let area = m.l1s[0].record(0, 2).expect("component 0's board area");
        assert_eq!(area.iter().fold(0u8, |s, b| s.wrapping_add(*b)), 0, "the area's checksum");
        assert_eq!(area[0], 2, "the format that carries CPU fields");
        let fields = area_fields(&area);
        let identity = node_board(&CHIPS[0], cpus);
        assert_eq!([fields[1], fields[3], fields[5]], [identity.name, identity.part, identity.revision].map(str::as_bytes),
            "{cpus} CPUs: the node board's name, part and revision");
        assert_eq!(m.l1s[0].record(1, 2), None, "{cpus} CPUs: no I/O board record");
        assert_eq!(fields[8], [3], "the CPU-field selector");
        let word = |f: &[u8]| u32::from_be_bytes(f.try_into().unwrap());
        let (config0, hub_mhz, cpu_mhz) = (word(fields[10]), word(fields[14]), word(fields[15]));
        for empty in [9, 11, 12, 13, 16, 17, 18] {
            assert!(fields[empty].is_empty(), "field {empty} is unknown and left empty");
        }
        let block = &m.nodes[0].flash.contents()[0x60..0xb0];
        let at = |o: usize, n: usize| block[o..o + n].iter().fold(0u64, |v, b| v << 8 | u64::from(*b));
        assert_eq!(at(0x04, 4), u64::from(config0), "{cpus} CPUs: the mode word");
        assert_eq!(at(0x10, 8), u64::from(cpu_mhz) * 1_000_000, "{cpus} CPUs: the CPU clock");
        assert_eq!(at(0x18, 8), u64::from(hub_mhz) * 1_000_000, "{cpus} CPUs: the Hub clock");
        assert_eq!(at(0x30, 4), u64::from(class), "{cpus} CPUs: the board word's class");
        });
    }
    for chip in &CHIPS {
        let context = cpu_context(&chip.firmware);
        assert_eq!(context.mode_word, chip.firmware.cpu_mode_word, "{}", chip.module.part.label());
        assert_eq!(context.cpu_mhz * 1_000_000, chip.firmware.cpu_clock_hz, "{}", chip.module.part.label());
        assert_eq!(chip.firmware.machine_type, class, "{}", chip.module.part.label());
    }
}

/// The L1 answers the system serial request with the text the emu_id gives, and nothing else moves
/// with it: the node board's record keeps its EMUN00 serial, which the PROM ties its NVRAM to, so a
/// machine given a new emu_id keeps the settings it saved.
#[test]
fn the_system_serial_moves_and_the_board_serial_stays() {
    let dir = TempPath::dir("tezro-serial");
    let machine = |emu_id: u32| {
        let mut cfg = tezro_config(&dir, 1);
        cfg.machine.emu_id = emu_id;
        with_machine(cfg, |m| {
            let l1 = &m.l1s[0];
            let records: Vec<_> = (0..2).flat_map(|component| [1, 2, 4].map(|area| l1.record(component, area))).collect();
            (l1.config(), l1.station_mac(), records)
        })
    };
    let (first, second) = (machine(42), machine(43));
    assert_eq!(first.0.system_serial.as_deref(), Some("LEMU200420"));
    assert_eq!(second.0.system_serial.as_deref(), Some("LEMU200430"));
    assert_eq!(first.1, [0x08, 0x00, 0x69, 0xe2, 0x02, 0xa0]);
    assert_eq!(first.2, second.2, "the board records keep their serial");
    let area = first.2[1].clone().expect("component 0's board area");
    assert_eq!(area_fields(&area)[2], b"EMUN00");
    assert_eq!(first.0.brick_serial, second.0.brick_serial);
}

/// The node board the L1 names follows the chip as well as the processor count: the 800 MHz chip
/// sits on 030-1956-002 and the 1 GHz chip on the IP59 boards, as their inventories print, and
/// every name is one the PROM's socket-mask table holds.
#[test]
fn the_node_board_record_names_the_board_for_the_chip_and_count() {
    let expected: [(&str, usize, [&str; 3]); 7] = [
        ("700", 1, ["IP53_1CPU", "", ""]),
        ("700", 2, ["IP53_2CPU", "030-1869-001", "-C"]),
        ("700", 4, ["IP53_4CPU", "030-1868-001", "-C"]),
        ("800", 4, ["IP53_4CPU", "030-1956-002", "-C"]),
        ("900", 4, ["IP53_4CPU", "", ""]),
        ("1000", 2, ["IP59_2CPU", "030-2059-002", "-A"]),
        ("1000", 4, ["IP59_4CPU", "030-1989-003", "-C"]),
    ];
    let dir = TempPath::dir("tezro-board-names");
    for (mhz, cpus, [name, part, revision]) in expected {
        let chip = CHIPS.iter().find(|c| c.module.part.mhz.to_string() == mhz).unwrap();
        let fields = with_machine(chip_config(&dir, chip, cpus as u8), |m| {
            let area = m.l1s[0].record(0, 2).expect("component 0's board area");
            area_fields(&area).iter().map(|f| f.to_vec()).collect::<Vec<_>>()
        });
        assert_eq!([&fields[1][..], &fields[3], &fields[5]], [name, part, revision].map(str::as_bytes), "{mhz} MHz, {cpus} CPUs");
    }
    // Every (chip, count) the board accepts has a board, and no other.
    for chip in &CHIPS {
        assert_eq!(chip.boards.iter().map(|(n, _)| *n).collect::<Vec<_>>(), chip.cpus, "{}", chip.module.part.label());
    }
}

/// The IO9's head 0 carries a TI PCI-to-PCI bridge at device 2, revision 1, with a type 1 header
/// and no interrupt pin, on every board; nothing answers on the bus behind it.
#[test]
fn the_io9_has_a_ti_bridge_with_an_empty_bus_at_device_2() {
    let dir = TempPath::dir("tezro-ti");
    for cpus in [1, 2, 4] {
        with_machine(tezro_config(&dir, cpus), move |m| {
            let io = Pio(m.nodes[0].chip.pio());
            let config = |reg: u64| io.pio_read(config_id(2) + reg, 4);
            assert_eq!(config(0).data, 0xac28_104c, "{cpus} CPUs: vendor and device");
            assert_eq!(config(8).data, 0x0604_0001, "{cpus} CPUs: a PCI-to-PCI bridge, revision 1");
            assert_eq!((config(0xc).data >> 16) & 0x7f, 1, "{cpus} CPUs: header type 1");
            assert_eq!((config(0x3c).data >> 8) & 0xff, 0, "{cpus} CPUs: no interrupt pin");
            assert_eq!(io.pio_write(config_id(u64::from(TI_BRIDGE_DEVICE)) + 0x18, 4, 0x0001_0100), BUS_OK, "number the bus behind it 1");
            // An unclaimed read behind a bridge ends in master abort: all ones.
            for device in 0..32 {
                aim_type1(&io, 1, device);
                let r = io.pio_read(widget(15, 0x2_8000), 4);
                assert_eq!((r.status, r.data), (BUS_OK, u64::from(u32::MAX)), "device {device} behind the bridge");
            }
        });
    }
}

/// With an `ice1724` entry, the ICE1724 answers at device 2 of the TI bridge's secondary bus and
/// nowhere else on it, and its reset state is the DM8 board's; without it the bus stays empty.
#[test]
fn the_audio_controller_is_the_only_function_behind_the_bridge_when_audio_is_on() {
    let dir = TempPath::dir("tezro-audio");
    let behind = |cfg: MachineConfig| {
        with_machine(cfg, |m| {
            let io = Pio(m.nodes[0].chip.pio());
            assert_eq!(io.pio_write(config_id(u64::from(TI_BRIDGE_DEVICE)) + 0x18, 4, 0x0001_0100), BUS_OK);
            let behind = |device: u64, register: u64| {
                aim_type1(&io, 1, device);
                let r = io.pio_read(widget(15, 0x2_8000 + register), 4);
                assert_eq!(r.status, BUS_OK, "device {device} register {register:#x}");
                r.data as u32
            };
            let ids: Vec<_> = (0..32).map(|device| Some(behind(device, 0))).collect();
            let at = u64::from(AUDIO_DEVICE);
            let registers = [behind(at, 8), behind(at, 0x2c), behind(at, 0x3c)].map(Some);
            (ids, registers, m.io.monitor.len(), m.io.monitor.iter().any(|d| d.audio_report().is_some()))
        })
    };
    let mut silent = tezro_config(&dir, 1);
    silent.devices.push(crate::devices::ConfiguredDevice { at: None, ..crate::devices::ConfiguredDevice::new("", crate::devices::DeviceModel::Ice1724) });
    silent.no_audio = true;
    let (ids, [class, subsystem, interrupt], monitors, reports) = behind(silent);
    assert!(reports, "the monitor has an audio report with the card");
    for (device, id) in ids.iter().enumerate() {
        assert_eq!(*id, Some(if device == usize::from(AUDIO_DEVICE) { 0x1724_1412 } else { u32::MAX }), "device {device}");
    }
    assert_eq!(class, Some(0x0401_0001), "an audio device, revision 1");
    assert_eq!(subsystem, Some(0x3630_1412), "the DM8 board's subsystem pair");
    assert_eq!(interrupt.map(|v| v >> 8 & 0xff), Some(1), "INTA");
    let (ids, _, without, reports) = behind(tezro_config(&dir, 1));
    assert!(!reports, "and only then");
    assert!(ids.iter().all(|id| *id == Some(u32::MAX)), "the bus behind the bridge is empty");
    assert_eq!(monitors, without + 1, "the audio card takes monitor commands");
}

/// PIC DMA reaches memory through Bedrock IO in naturally aligned legs.
#[test]
fn pic_dma_reaches_memory_through_bedrock_io() {
    let dir = TempPath::dir("tezro-pic-dma");
    with_machine(tezro_config(&dir, 1), |m| {
        let chip = &m.nodes[0].chip;
        // Enable bank pair 0, as the Fuel's DMA tests do.
        assert_eq!(chip.registers().write64(crate::dev::bedrock::regs::md::BASE + 8, 0x33), BUS_OK);
        let peer = chip.inbound(15);
        let data: Vec<u8> = (1..=19).collect();
        assert!(write_bytes(&*peer, 0x1003, &data));
        let mut back = vec![0; data.len()];
        assert!(read_bytes(&*peer, 0x1003, &mut back));
        assert_eq!(back, data);
        assert_eq!(chip.decode.physical().read64(0x1000).data, 0x0000_0001_0203_0405);
    });
}

/// A transfer of many pages arrives as the bytes it carried, whatever its alignment, and what it
/// leaves in memory is what the same bytes would have left through single accesses.
#[test]
fn a_long_pic_transfer_lands_byte_for_byte_in_memory() {
    let dir = TempPath::dir("tezro-long-dma");
    with_machine(tezro_config(&dir, 1), |m| {
        let chip = &m.nodes[0].chip;
        assert_eq!(chip.registers().write64(crate::dev::bedrock::regs::md::BASE + 8, 0x33), BUS_OK);
        let peer = chip.inbound(15);
        let memory = chip.decode.physical();
        for (start, len) in [(0x10_0000u64, 20_000usize), (0x20_0003, 9_000), (0x30_0004, 4_100), (0x40_0000, 4_103)] {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
            assert!(write_bytes(&*peer, start, &data));
            for (i, byte) in data.iter().enumerate() {
                assert_eq!(memory.read8(start + i as u64).data, *byte, "write at {start:#x}+{i}");
            }
            let mut back = vec![0; len];
            assert!(read_bytes(&*peer, start, &mut back));
            assert_eq!(back, data, "read at {start:#x}");
        }
        // Memory no bank holds refuses the transfer.
        let mut back = [0u8; 64];
        assert!(!read_bytes(&*peer, 6 << 30, &mut back));
        assert!(!write_bytes(&*peer, 6 << 30, &back));
    });
}

/// The two- and four-processor board adds a second PIC at widget 11 with nothing on its buses,
/// keeps the IO9 on widget 15 behind widget 0, and leaves the V12's widget 12 empty.
#[test]
fn two_or_four_processors_bring_the_second_pic_at_widget_11() {
    let dir = TempPath::dir("tezro-second-pic");
    for cpus in [2, 4] {
        with_machine(tezro_config(&dir, cpus), move |m| {
            let io = Pio(m.nodes[0].chip.pio());
            assert_eq!(io.pio_read(widget(11, 0), 8).data, u64::from(crate::dev::pic::identity(0)));
            assert_eq!(io.pio_read(widget(15, 0), 8).data, u64::from(crate::dev::pic::identity(0)));
            assert_eq!(io.pio_read(widget(12, 0), 8).status, BUS_ERR, "no V12");
            assert_eq!((io.pio_read(widget(0, 4), 4).data >> 12) & 0xffff, u64::from(crate::dev::pxbow::PART));
            for device in 1..=4 {
                assert_eq!(io.pio_read(widget(11, 0x2_0000 + (device << 12)), 4).status, BUS_ERR, "widget 11 device {device}");
            }
            assert_eq!(io.pio_read(config_id(1), 4).data, 0x100a_10a9);
        });
    }
}

// Widget 0's crossbar as PROM 6.210 and the IP35 kernel walk it, from how the PROM finds
// widgets 0xb and 0xf.
// QEMU's sn/builder/tezro-two-four-processor-board in
// tests/qtest/sn-builder-test.c still expects link 11 absent; this board
// follows the PROM's walk instead.

/// Link `n`'s register block in widget 0, as the 32-bit word address the
/// PROM issues: the low word of each doubleword.
fn link_word(n: u64, register: u64) -> u64 {
    widget(0, 0x100 + (n - 8) * 0x40 + register + 4)
}
const LINK_CONTROL: u64 = 0x08;
const LINK_STATUS: u64 = 0x10;
const LINK_ARB_UPPER: u64 = 0x18;
const LINK_ARB_LOWER: u64 = 0x20;
const LINK_STATUS_CLEAR: u64 = 0x28;
const LINK_RESET: u64 = 0x30;
const LINK_AUX: u64 = 0x38;
const ALIVE: u64 = 1 << 31;
const PRESENT: u64 = 1 << 5;
const FAILURE_MODE: u64 = 1 << 6;

fn read_word(board: &Machine<Bedrock, Flash>, address: u64) -> u64 {
    let r = pio_read(board, address, 4);
    assert_eq!(r.status, BUS_OK, "{address:#x}");
    r.data
}

/// Widget 0's part and manufacturer, from its identity word.
fn widget_zero_part(board: &Machine<Bedrock, Flash>) -> (u64, u64) {
    let id = read_word(board, widget(0, 4));
    ((id >> 12) & 0xffff, (id >> 1) & 0x7ff)
}

fn alive_and_present(board: &Machine<Bedrock, Flash>, n: u64) -> (bool, bool) {
    (read_word(board, link_word(n, LINK_STATUS)) & ALIVE != 0, read_word(board, link_word(n, LINK_AUX)) & PRESENT != 0)
}

/// The PROM's per-link test: alive and present, except that link 15 is
/// taken regardless when widget 0 is part 0xd100.
fn prom_uses(board: &Machine<Bedrock, Flash>, n: u64) -> bool {
    let (alive, present) = alive_and_present(board, n);
    (alive && present) || (n == 15 && widget_zero_part(board).0 == 0xd100)
}

/// The kernel's: alive, and present unless widget 0 is part 0xd100 with
/// manufacturer 0 and the link is 15.
fn kernel_uses(board: &Machine<Bedrock, Flash>, n: u64) -> bool {
    let (alive, present) = alive_and_present(board, n);
    alive && (present || (n == 15 && widget_zero_part(board) == (0xd100, 0)))
}

/// Links 8 to 15 that pass both tests on their own bits, without either
/// guest's link-15 exception, as a widget mask.
fn live_links(board: &Machine<Bedrock, Flash>) -> u16 {
    (8..16).filter(|&n| alive_and_present(board, n) == (true, true)).fold(0, |m, n| m | 1 << n)
}

/// On the two- and four-processor board widget 0's crossbar shows the
/// hub's link 10 and both PICs' links 11 and 15 alive and present, with no
/// failure mode and arbitration bits 0 and 2 clear, so the PROM and the
/// kernel both take 11 and 15 without leaning on the link-15 exception.
#[test]
fn widget_0_shows_both_pic_links_live_on_the_larger_board() {
    let dir = TempPath::dir("tezro-links");
    for cpus in [2, 4] {
        with_machine(tezro_config(&dir, cpus), move |board| {
            for pass in ["built", "after power-on"] {
                assert_eq!(widget_zero_part(board), (0xd100, 0), "{cpus} CPUs, {pass}");
                assert_eq!(read_word(board, widget(0, 4)) & 1, 1);
                assert_eq!(live_links(board), 1 << 10 | 1 << 11 | 1 << 15, "{cpus} CPUs, {pass}");
                for n in [11, 15] {
                    assert!(prom_uses(board, n) && kernel_uses(board, n), "link {n}, {cpus} CPUs, {pass}");
                    assert_eq!(read_word(board, link_word(n, LINK_AUX)) & FAILURE_MODE, 0, "link {n}");
                    assert_eq!(read_word(board, link_word(n, LINK_ARB_UPPER)) & 0b101, 0, "link {n}: unclaimed");
                }
                for n in [8, 9, 12, 13, 14] {
                    assert_eq!(alive_and_present(board, n), (false, false), "link {n}, {cpus} CPUs, {pass}");
                    assert!(!prom_uses(board, n) && !kernel_uses(board, n));
                }
                (board.io.reset)();
            }
        });
    }
}

/// Widget 11 answers as a bridge-class PIC with both heads, the identity
/// the PROM's "Found PIC" path needs (manufacturer 0x24, bit 0 set, part
/// 0xd102 on head 0), and takes the PROM's control-word writes on both heads.
#[test]
fn widget_11_answers_as_a_pic_with_both_heads() {
    use crate::dev::pic::HEAD_OFFSET;
    let dir = TempPath::dir("tezro-pic-heads");
    with_machine(tezro_config(&dir, 4), |board| {
        let io = Pio(board.nodes[0].chip.pio());
        for (head, part) in [(0, 0xd102), (HEAD_OFFSET, 0xd112)] {
            let id = read_word(board, widget(11, head + 4));
            assert_eq!((id & 1, (id >> 1) & 0x7ff, (id >> 12) & 0xffff), (1, 0x24, part), "head at {head:#x}");
            // Widget control, as the PROM's table-driven field writes it.
            let control = widget(11, head + 0x24);
            let before = read_word(board, control);
            assert_eq!(io.pio_write(control, 4, (before & !0xf000) | 0x4000), BUS_OK);
            assert_eq!(read_word(board, control) & 0xf000, 0x4000, "head at {head:#x}");
        }
        assert_eq!(read_word(board, widget(15, 4)) >> 12 & 0xffff, 0xd102);
        // The PROM's bring-up of link 11 (clear status, credit 2, clear both arbitration registers,
        // then credit 3) leaves it alive and present with the credit field reading back.
        read_word(board, link_word(11, LINK_STATUS_CLEAR));
        assert_eq!(io.pio_write(link_word(11, LINK_CONTROL), 4, 0x0008_0000), BUS_OK);
        assert_eq!(io.pio_write(link_word(11, LINK_ARB_UPPER), 4, 0), BUS_OK);
        assert_eq!(io.pio_write(link_word(11, LINK_ARB_LOWER), 4, 0), BUS_OK);
        assert_eq!(alive_and_present(board, 11), (true, true));
        let control = read_word(board, link_word(11, LINK_CONTROL));
        assert_eq!((control >> 18) & 0x1f, 2);
        assert_eq!(io.pio_write(link_word(11, LINK_CONTROL), 4, (control & !(0x1f << 18)) | 3 << 18), BUS_OK);
        assert_eq!((read_word(board, link_word(11, LINK_CONTROL)) >> 18) & 0x1f, 3);
        assert_eq!(alive_and_present(board, 11), (true, true));
        assert_eq!(io.pio_read(widget(11, 4), 4).status, BUS_OK);
    });
}

/// The processor reaches widget 11 through the crossbar, not around it: a
/// reset of link 11 or of the hub's link 10 leaves widget 11 reachable,
/// registers kept.
#[test]
fn widget_11_is_routed_through_widget_0s_crossbar() {
    let dir = TempPath::dir("tezro-route-11");
    with_machine(tezro_config(&dir, 4), |board| {
        let io = Pio(board.nodes[0].chip.pio());
        let route = widget(11, 0x130);
        assert_eq!(io.pio_write(route, 8, 0x1122_3344_5566_7788), BUS_OK);
        for link in [11, 10, 15] {
            assert_eq!(io.pio_write(link_word(link, LINK_RESET), 4, 0), BUS_OK);
            let r = io.pio_read(route, 8);
            assert_eq!((r.status, r.data), (BUS_OK, 0x1122_3344_5566_7788), "after link {link}'s reset");
            assert_eq!(alive_and_present(board, 11), (true, true));
        }
        assert_eq!(live_links(board), 1 << 10 | 1 << 11 | 1 << 15);
    });
}

/// Only the counts a board holds are accepted, and only those its chip is populated with.
#[test]
fn the_board_refuses_a_population_it_cannot_hold() {
    let dir = TempPath::dir("tezro-refused");
    let refusal = |cfg: MachineConfig| crate::display_stop_tests::big_stack(move || machine(&cfg).err());
    for cpus in [0, 3, 5] {
        assert!(refusal(tezro_config(&dir, cpus)).is_some(), "{cpus} processors");
    }
    let one_ghz = CHIPS.iter().find(|c| c.module.part.mhz == 1000).unwrap();
    let refused = refusal(chip_config(&dir, one_ghz, 1));
    assert!(refused.is_some_and(|e| e.contains("populated with 2, 4 processors only")));
}

/// Every chip's Config word decodes, by the PROM editor's formulas, to the
/// fields its inventory states: SS gives the cache size, SC the printed cache
/// clock (2 * CPU / (SC + 1)) and EC the 200 MHz Bedrock (code 2 * CPU /
/// Bedrock - 1), with K0 5, PM 3, SB and BE set.
#[test]
fn each_tezro_configs_fields_decode_to_the_stated_sizes_and_clocks() {
    // The cache clocks as the inventories print them.
    for (c, scache_mhz) in CHIPS.iter().zip([350, 400, 450, 333]) {
        let word = c.firmware.reset_mode();
        let name = c.module.part.name();
        let field = |lo: u32, bits: u32| (word >> lo) & ((1 << bits) - 1);
        let mhz = c.module.part.mhz;
        let bytes = crate::with_cpu!(c.module.part, true, false, C => <C as MipsCache>::L2_SIZE as u32);
        assert_eq!((field(0, 3), field(7, 2), field(13, 1), field(15, 1)), (5, 3, 1, 1), "{name}: K0 PM SB BE");
        assert_eq!(field(9, 4), 2 * mhz / 200 - 1, "{name}: EC");
        assert_eq!(2 * mhz / (field(19, 3) + 1), scache_mhz, "{name}: SC gives the cache clock");
        assert_eq!(512u32 << 10 << field(16, 3), bytes, "{name}: SS gives the cache size");
        assert_eq!((word >> 23) & 1, 1, "{name}: bit 23 is set in every row of 500 MHz and up");
        assert_eq!(word >> 24, 0, "{name}: bit 24 clear");
    }
    let word = chip(&parts::R16000_1000).firmware.reset_mode();
    assert_eq!(((word >> 16) & 7, (word >> 19) & 7, (word >> 9) & 15, (word >> 22) & 1), (5, 5, 9, 1), "1 GHz SS SC EC and bit 22");
}

/// A chip and its flash record are two statements of one chip: the clock,
/// and the Config word the PROM compares against the processor's, must agree.
#[test]
fn each_tezro_chip_agrees_with_its_flash_record() {
    for c in &CHIPS {
        let record = c.firmware;
        let name = c.module.part.name();
        assert_eq!(u64::from(record.cpu_clock_hz), u64::from(c.module.part.mhz) * 1_000_000, "{name}");
        assert_eq!(record.cpu_mode_word, c.module.mode_word, "{name}: the record and the module");
        assert_eq!(record.cpu_mode_word & CONFIG0_MODE_MASK, record.reset_mode(), "{name}");
        assert!(c.cpus.iter().all(|&n| n <= 4) && !c.cpus.is_empty(), "{name}");
    }
    assert_eq!(CHIPS.len(), CPUS.len());
    assert!(CHIPS.iter().zip(CPUS).all(|(c, &model)| c.module.part == model), "CPUS lists the chips in table order");
}

/// A bare family name builds the 700 MHz chip, and a chip name its own: PRId, Config, flash record
/// and Count rate follow end to end from a configuration, and a CPU reset keeps them, reloading
/// PRId and FIR.
#[test]
fn the_family_name_builds_the_700_mhz_chip_and_a_chip_name_its_own() {
    for (cpu, prid, mhz, mib, config0) in [
        ("r16000", 0x0f41, 700, 4u32, 0x009b_ad85),
        ("r16000_800", 0x0f42, 800, 4, 0x009b_af85),
        ("r16000_900", 0x0f50, 900, 8, 0x009c_b185),
        ("r16000_1000", 0x0f50, 1000, 16, 0x00ed_b385),
    ] {
        let dir = TempPath::dir("tezro-family");
        let mut cfg = tezro_config(&dir, if cpu == "r16000_1000" { 2 } else { 4 });
        cfg.machine.cpu = Some(cpu.into());
        let part = cfg.cpu_part().unwrap();
        let chip = chip(part);
        with_machine(cfg, move |m| {
            let (l2, config) = crate::with_cpu!(part, true, false, C => {
                let mut core = build_cpu::<C>(m, chip).core;
                assert_eq!((core.cp0_prid, core.reset_prid, core.fpu_fir, core.reset_fir), (prid, prid, 0x900, 0x900), "{cpu}");
                let config = core.cp0_config;
                core.fpu_fir = 0;
                core.cp0_prid = 0;
                core.reset(false);
                assert_eq!((core.cp0_prid, core.fpu_fir, core.cp0_config), (prid, 0x900, config), "{cpu}: after reset");
                (<C as MipsCache>::L2_SIZE as u32, config)
            });
            assert_eq!(config & CONFIG0_MODE_MASK, config0, "{cpu}");
            assert_eq!(config >> 26, 0b011_011, "{cpu}: 32 KB primary caches");
            assert_eq!(part.prid(), prid, "{cpu}: the chip's PRId");
            assert_eq!(l2 >> 20, mib, "{cpu}: secondary cache");
            assert_eq!((config >> 16) & 7, 1 + mib.trailing_zeros(), "{cpu}: Config SS");
            assert_eq!(part.count_hz(), u64::from(mhz) * 500_000, "{cpu}: Count at half the clock");
            assert_eq!(chip.firmware.cpu_clock_hz, mhz * 1_000_000, "{cpu}: the flash record");
            assert_eq!(chip.module.config(), config, "{cpu}: record and processor agree");
        });
    }
}

/// IOC4 reads its 32-bit registers in the processor's byte order through
/// the IO9 PIC's self-ordered slot: what the processor writes is the
/// register's value, as on the QEMU Tezro, where PROM 6.210 writes the
/// serial DMA control 0x100201ff and the TX producer 0x3f8.
#[test]
fn the_processor_reads_and_writes_ioc4_registers_in_its_own_byte_order() {
    let dir = TempPath::dir("tezro-ioc4");
    with_machine(tezro_config(&dir, 1), |m| {
        let io = Pio(m.nodes[0].chip.pio());
        map_ioc4(&io);
        let reg = |offset: u64| widget(15, 0x20_0000 + offset);
        assert_eq!(io.pio_read(reg(0x24), 4).data, 1, "MISC: the 66 MHz clock");
        assert_eq!(io.pio_write(reg(0x314), 4, 0x3f8), BUS_OK);
        assert_eq!(io.pio_read(reg(0x314), 4).data, 0x3f8, "port 0 TX producer");
    });
}

/// IOC4's BAR at PCI memory 0x200000, decoding memory, reached through head 0's first short
/// window, as the PROM maps it.
fn map_ioc4(io: &Pio) {
    assert_eq!(io.pio_write(config_id(1) + 0x10, 4, 0x20_0000), BUS_OK);
    assert_eq!(io.pio_write(config_id(1) + 4, 4, 2), BUS_OK);
    assert_eq!(io.pio_write(widget(15, 0x200), 8, (1 << 12) | 2), BUS_OK);
}

/// The PROM keeps its environment in the 32 KiB DS1744 window at IOC4 BAR offset 0x80000.
#[test]
fn a_store_through_the_ioc4_nvram_window_reaches_the_nvram_file() {
    let dir = TempPath::dir("tezro-nvram");
    let path = dir.join("nvram");
    let mut cfg = tezro_config(&dir, 1);
    cfg.nvram = path.to_string_lossy().into_owned();
    let window = |offset: u64| widget(15, 0x20_0000 + 0x8_0000 + offset);

    let (written, path_in) = (cfg.clone(), path.clone());
    with_machine(written, move |m| {
        let pio = m.nodes[0].chip.pio();
        map_ioc4(&Pio(pio.clone()));
        for (offset, byte) in [(0x428u64, 0xc0u8), (0x429, 0xa8), (0x7ff0, 0x5a)] {
            assert_eq!(pio.write8(window(offset), byte), BUS_OK);
        }
        let on_disk = std::fs::read(&path_in).unwrap();
        assert_eq!(on_disk.len(), crate::dev::ds1744::SIZE);
        assert_eq!((on_disk[0x428], on_disk[0x429], on_disk[0x7ff0]), (0xc0, 0xa8, 0x5a), "written through at once");
    });

    with_machine(cfg, move |m| {
        let pio = m.nodes[0].chip.pio();
        map_ioc4(&Pio(pio.clone()));
        let read = |offset: u64| pio.read8(window(offset)).data;
        assert_eq!((read(0x428), read(0x429), read(0x7ff0)), (0xc0, 0xa8, 0x5a), "a later board sees the stores");
        assert_eq!(read(0x42a), 0, "and only those");
    });
}

// VPro host-interface registers the tests below touch, as byte offsets in
// the board's widget space. The board's own constants are private to it.
const ODSY_IDENTITY: u64 = 0x00;

fn config_with_vpro(dir: &TempPath, cpus: u8) -> MachineConfig {
    let mut cfg = tezro_config(dir, cpus);
    cfg.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::V12)));
    cfg
}

/// The V12 answers as widget 12 behind the PXBOW on either board, and widget 12 stays empty without
/// one.
#[test]
fn the_vpro_answers_at_widget_12_on_both_boards() {
    let dir = TempPath::dir("tezro-v12");
    for cpus in [1, 4] {
        with_machine(config_with_vpro(&dir, cpus), move |m| {
            // The identity word's low half, as the Fuel's V12 reads there.
            let id = pio_read(m, widget(12, ODSY_IDENTITY + 4), 4);
            assert_eq!((id.status, id.data), (BUS_OK, 0xac01_3047), "{cpus} CPUs");
            assert_eq!(alive_and_present(m, 12), (true, true), "{cpus} CPUs: the crossbar's link 12");
        });
    }
    with_machine(tezro_config(&dir, 1), |m| assert_eq!(pio_read(m, widget(12, ODSY_IDENTITY), 4).status, BUS_ERR));
}

/// IRIX names the V12 from the board area the node's L1 gives for component
/// 2, and logs two alerts at attach when the L1 refuses it.
#[test]
fn the_l1_names_an_attached_vpro_in_component_2() {
    let dir = TempPath::dir("tezro-v12-l1");
    let (area, chassis) = with_machine(config_with_vpro(&dir, 2), |m| (m.l1s[0].record(2, 2), m.l1s[0].record(2, 1)));
    let area = area.expect("component 2's board area with a V12");
    assert_eq!(area.iter().fold(0u8, |s, b| s.wrapping_add(*b)), 0, "the area's checksum");
    // Manufacturer, name, serial, part, the file byte and revision, read by
    // position: a one-character revision's type byte is 0xc1, which
    // area_fields would take for the end.
    let mut fields = Vec::new();
    let mut at = 6;
    for _ in 0..6 {
        let len = usize::from(area[at] & 0x3f);
        fields.push(&area[at + 1..at + 1 + len]);
        at += 1 + len;
    }
    assert_eq!([fields[1], fields[3], fields[5]], [b"ODY128VERSIONB".as_slice(), b"030-1611-001", b"C"]);
    assert_eq!(chassis, None, "no chassis area for the graphics board");
    let plain = with_machine(tezro_config(&dir, 2), |m| m.l1s[0].record(2, 2));
    assert_eq!(plain, None, "nothing in component 2 without graphics");
}

/// The cached view the processors' caches sit on swaps RAM above 4 GiB, so SC stays atomic there
/// too.
#[test]
fn the_cached_view_swaps_ram_above_4_gib() {
    let dir = TempPath::dir("tezro-cas");
    let mut cfg = tezro_config(&dir, 4);
    cfg.banks = [1024, 1024, 512, 512, 512, 512, 0, 0];
    with_machine(cfg, |m| {
        let chip = &m.nodes[0].chip;
        // Banks 0 to 5: two 1 GiB banks and four of 512 MiB, as MD's tests set them.
        assert_eq!(chip.registers().write64(crate::dev::bedrock::regs::md::BASE + 8, 0x53 | (0x43 << 8) | (0x43 << 16)), BUS_OK);
        let cached: Arc<dyn BusDevice> = chip.decode.physical().clone();
        let at = (5u64 << 30) + 0x80;
        assert_eq!(cached.write64(at, 1), BUS_OK);
        assert_eq!(cached.cas64(at, 1, 2), Some(true));
        assert_eq!(cached.cas64(at, 1, 3), Some(false));
        assert_eq!(cached.read64(at).data, 2);
    });
}

/// A fresh flash holds no environment records: with the node board named
/// for its processor count the PROM tries only the populated positions, so no
/// board needs a Disable record to boot straight through.
#[test]
fn a_fresh_flash_starts_with_no_environment_records() {
    let dir = TempPath::dir("tezro-flash");
    for cpus in [1, 2, 4] {
        let bytes = with_machine(tezro_config(&dir, cpus), |m| m.nodes[0].flash.contents());
        assert!(bytes[0x9e0100..0x9e0200].iter().all(|&b| b == 0xff), "{cpus} CPUs");
    }
}

/// The connector adds to an existing console log, and starts it over when `serial_log_overwrite` is
/// set.
#[test]
fn the_serial_connector_appends_to_the_log_unless_told_to_overwrite() {
    let dir = crate::test_temp::TempPath::dir("ip53-serial-log");
    let log = dir.join("console.log");
    for (overwrite, want) in [(false, &b"stale"[..]), (true, &b""[..])] {
        std::fs::write(&log, "stale").unwrap();
        let mut cfg = MachineConfig::default();
        cfg.ci = true;
        cfg.serial_log = Some(log.to_string_lossy().into_owned());
        cfg.serial_log_overwrite = overwrite;
        let ci = Arc::new(CiSerialBackend::new());
        let ports = [SerialPort { label: "L1", port: 0, setting: "test", ci: ci.clone(), log: None, socket: None, ttyd: None, connect: Box::new(|_| ()) }];
        connect_serial_ports(&cfg, "Test", &ports, 0).unwrap();
        assert_eq!(std::fs::read(&log).unwrap(), want, "serial_log_overwrite = {overwrite}");
    }
}

/// A hub register as the processor reaches it through the two register windows of uncached space 1:
/// the local form at 0x0100_0000 and the node form at 0x0180_0000, both onto this node's registers
/// on a board whose processors sit on PI0 alone.
fn hub_register(board: &Machine<Bedrock, Flash>, window: u64, offset: u64) -> crate::traits::BusRead64 {
    board.nodes[0].bus.read64((1 << 57) | window | offset)
}
const HUB_WINDOWS: [u64; 2] = [0x0100_0000, 0x0180_0000];

/// The PROM writes the hub's widget control (0x400020) with destination
/// credits 4 in bits 15:13 for the integrated crossbar; the register keeps
/// them, and the hub's widget number in bits 3:0, through either window.
#[test]
fn the_hub_keeps_the_proms_destination_credits_of_four() {
    let dir = TempPath::dir("tezro-hub-credits");
    for window in HUB_WINDOWS {
        with_machine(tezro_config(&dir, 1), move |board| {
            let control = hub_register(board, window, 0x40_0020).data;
            assert_eq!(control & 0xf, 10, "the hub's own widget number");
            let value = (control & !(7 << 13)) | (4 << 13);
            assert_eq!(board.nodes[0].bus.write64((1 << 57) | window | 0x40_0020, value), BUS_OK);
            let back = hub_register(board, window, 0x40_0020).data;
            assert_eq!((back >> 13) & 7, 4, "window {window:#x}");
            assert_eq!(back & 0xf, 10);
        });
    }
}

/// A one-node boot reads the network interface as an uncabled port: status
/// 1, control 0, error status 0 and the error-clear register 0 however often
/// it is read, through both windows, and neither network error bit (39, 40)
/// is up in either processor interface's second pending register.
#[test]
fn a_one_node_boot_reads_the_network_interface_quiet() {
    use crate::dev::bedrock::regs::{ni, pi, PI1_BASE};
    let dir = TempPath::dir("tezro-ni");
    for cpus in [1, 4] {
        with_machine(tezro_config(&dir, cpus), move |board| {
            for window in HUB_WINDOWS {
                for (offset, expected) in [(ni::PORT_STATUS::OFFSET, 1), (ni::RESET_ENABLE::OFFSET, 0), (ni::PORT_ERRORS::OFFSET, 0), (ni::PORT_ERROR_CLEAR::OFFSET, 0)] {
                    for count in 0..4 {
                        let r = hub_register(board, window, ni::BASE + offset);
                        assert_eq!((r.status, r.data), (BUS_OK, expected), "{cpus} CPUs, {window:#x} + NI {offset:#x}, read {count}");
                    }
                }
            }
            let registers = board.nodes[0].chip.registers();
            for interface in 0..2 {
                let pending = registers.read64(interface * PI1_BASE + pi::INT_PEND1::OFFSET).data;
                assert_eq!(pending & (1 << 39 | 1 << 40), 0, "{cpus} CPUs, PI{interface}");
            }
        });
    }
}

/// The IP35 PROM's crossbar check on the Tezro: it writes the read-only lower
/// error address, wants the register-access error with the hub's widget in
/// the status source, then enables that condition, writes the upper error
/// address, and wants the error again and vector 0x10 at the PI.
#[test]
fn the_proms_crossbar_check_passes_on_the_tezro_board() {
    use crate::dev::bedrock::regs::pi;
    let dir = TempPath::dir("tezro-xbow-check");
    with_machine(tezro_config(&dir, 1), |board| {
        let io = Pio(board.nodes[0].chip.pio());
        let status_clear = widget(0, 0x54);
        let (err_upper, err_lower, control) = (widget(0, 0x14), widget(0, 0x1c), widget(0, 0x24));
        let registers = board.nodes[0].chip.registers();
        let pending_0 = || registers.read64(pi::INT_PEND0::OFFSET).data;
        // Vector 0x10 to the hub, at PI 0x90 of its register space (address
        // bit 47 selects the registers).
        assert_eq!(io.pio_write(widget(0, 0x34), 4, (0x10 << 24) | (10 << 16) | 0x8000), BUS_OK);
        assert_eq!(io.pio_write(widget(0, 0x3c), 4, 0x0180_0090), BUS_OK);

        assert_eq!(io.pio_write(err_lower, 4, 0xdead_beef), BUS_OK, "no bus error");
        let status = read_word(board, status_clear);
        assert_ne!(status & (1 << 5), 0, "register-access error");
        assert_eq!((status >> 6) & 0xf, 10, "the hub's widget is the source");
        assert_eq!(read_word(board, status_clear) & (1 << 5), 0, "the alias cleared it");
        assert_eq!(pending_0() & (1 << 16), 0, "nothing was enabled yet");

        assert_eq!(io.pio_write(control, 4, 0x20), BUS_OK);
        assert_eq!(io.pio_write(err_upper, 4, 0), BUS_OK, "no bus error");
        let status = read_word(board, widget(0, 0x0c));
        assert_ne!(status & (1 << 5), 0, "register-access error again");
        assert_eq!((status >> 6) & 0xf, 10);
        assert_ne!(pending_0() & (1 << 16), 0, "vector 0x10 reached the PI");
        // Acknowledging the error lowers the vector again.
        assert_ne!(read_word(board, status_clear) & (1 << 5), 0);
        assert_eq!(pending_0() & (1 << 16), 0);

        // The PROM's lock: bit 1 of link 15's arbitration-upper register.
        let lock = link_word(15, LINK_ARB_UPPER);
        assert_eq!(io.pio_write(lock, 4, 2), BUS_OK);
        assert_eq!(read_word(board, lock) & 2, 2);
    });
}

/// A guest power-off exits the process after the machine's flush, without
/// dropping the board: an overlay's writes on either ISP12160 channel must
/// reach the next run that way.
#[test]
fn flushed_overlay_writes_survive_a_power_off_that_skips_drop() {
    use crate::scsi::{scsi_cmd, ScsiDataLength, ScsiRequest};
    for channel in 0..2 {
        // The overlay and its dirty-sector record live beside the image, so
        // the directory takes them all when it goes.
        let dir = TempPath::dir(&format!("ip53-flush-{channel}"));
        let (writer_dir, reader_dir) = (TempPath::dir("ip53-flush-writer"), TempPath::dir("ip53-flush-reader"));
        let path = dir.join("disk.img");
        std::fs::write(&path, vec![0u8; 4 * 512]).unwrap();
        let attach = move |m: &Machine<Bedrock, Flash>| {
            m.with_disks(|buses| {
                buses[channel].attach_device(1, path.to_str().unwrap(), false, Vec::new(), true, None, true).unwrap();
            })
        };
        let write = ScsiRequest {
            cdb: vec![scsi_cmd::WRITE_10, 0, 0, 0, 0, 2, 0, 0, 1, 0],
            data_len: ScsiDataLength::Unlimited,
            data_in: Some(vec![0x5A; 512]),
        };
        let read = ScsiRequest {
            cdb: vec![scsi_cmd::READ_10, 0, 0, 0, 0, 2, 0, 0, 1, 0],
            data_len: ScsiDataLength::Unlimited,
            data_in: None,
        };
        let (reader_cfg, attach_again) = (tezro_config(&reader_dir, 1), attach.clone());
        with_machine(tezro_config(&writer_dir, 1), move |writer| {
            attach(writer);
            writer.with_disks(|buses| {
                buses[channel].request(1, &write).unwrap().unwrap();
                buses[channel].flush().unwrap();
            });
            // A second machine reads the disk while the first is still alive, so nothing the
            // first one's drop does can be what it sees.
            with_machine(reader_cfg, move |reader| {
                attach_again(reader);
                let got = reader.with_disks(|buses| buses[channel].request(1, &read).unwrap().unwrap());
                assert_eq!(got.data, vec![0x5A; 512], "channel {channel}");
            });
        });
    }
}

/// Saving the NVRAM names the failure when the file cannot be written.
#[test]
fn a_failed_nvram_save_is_an_error() {
    let dir = TempPath::dir("tezro-nvram-save");
    let cfg = tezro_config(&dir, 1);
    let nvram = std::path::PathBuf::from(&cfg.nvram);
    with_machine(cfg, move |m| {
        assert!((m.io.flush_nvram)().is_ok());
        // A directory where the file was: the write cannot succeed, whoever runs the test.
        std::fs::remove_file(&nvram).unwrap();
        std::fs::create_dir(&nvram).unwrap();
        assert!((m.io.flush_nvram)().is_err(), "the save reported success");
    });
}

/// virtio goes in the IO9 PIC's empty bus 1, devices 1 and 2, in entry order.
#[cfg(unix)]
#[test]
fn virtio_takes_the_second_bus_of_the_io9() {
    use crate::dev::pic::HEAD_OFFSET;
    let dir = TempPath::dir("tezro-virtio");
    let mut cfg = tezro_config(&dir, 1);
    for text in ["type = \"virtio_serial\"", "type = \"virtio_9p\"\npath = \"/tmp\""] {
        cfg.devices.push(crate::devices::ConfiguredDevice::from_table(toml::from_str(text).unwrap()).unwrap());
    }
    with_machine(cfg, |m| {
        let io = Pio(m.nodes[0].chip.pio());
        let id = |device: u64| io.pio_read(widget(15, HEAD_OFFSET + 0x2_0000 + (device << 12)), 4).data;
        assert_eq!((id(1), id(2)), (0x1043_1af4, 0x1049_1af4));
    });
}
