use super::*;
use crate::traits::{BUS_ERR, BUS_OK};
use std::sync::atomic::AtomicU64;

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
fn pio_read(bedrock: &Arc<Bedrock>, a: u64, size: u8) -> crate::traits::BusRead64 {
    Pio(bedrock.pio()).pio_read(a, size)
}

fn widget(widget: u64, offset: u64) -> u64 {
    (widget << 24) | offset
}

/// Configuration register 0 of `device` on the IO9 PIC's head 0.
fn config_id(device: u64) -> u64 {
    widget(15, 0x2_0000 + (device << 12))
}

/// The processor finds the PXBOW at widget 0 and the IO9's PIC at widget
/// 15, as the QEMU profile maps them; the one-processor board has no widget
/// 11 and no other widget answers.
#[test]
fn widget_0_is_the_pxbow_and_widget_15_the_io9_pic() {
    let board = TezroBoard::new(1).unwrap();
    let io = Pio(board.bedrock.pio());
    let pxbow = io.pio_read(widget(0, 4), 4);
    assert_eq!(pxbow.status, BUS_OK);
    assert_eq!((pxbow.data >> 12) & 0xffff, u64::from(crate::dev::pxbow::PART));
    let pic = io.pio_read(widget(15, 0), 8);
    assert_eq!(pic.status, BUS_OK);
    assert_eq!(pic.data, u64::from(crate::dev::pic::identity(0)));
    for absent in [8, 9, 11, 12, 13, 14] {
        assert_eq!(io.pio_read(widget(absent, 0), 8).status, BUS_ERR, "widget {absent}");
    }
    assert_eq!(board.pics.len(), 1);
}

/// IO9's functions sit at PCI devices 1, 3 and 4 of head 0 with the
/// revisions Tezro consoles print (the TI bridge at device 2 has its own
/// test).
#[test]
fn the_io9_bus_holds_the_ioc4_the_isp12160_and_the_bcm5701() {
    let board = TezroBoard::new(1).unwrap();
    let io = Pio(board.bedrock.pio());
    for (device, id, revision) in [(1, 0x100a_10a9, 0x53), (3, 0x1216_1077, 0x06), (4, 0x1645_14e4, 0x15)] {
        assert_eq!(io.pio_read(config_id(device), 4).data, id, "device {device}");
        assert_eq!(io.pio_read(config_id(device) + 8, 4).data & 0xff, revision, "device {device}");
    }
}

/// The BCM5701's slot drives REQ64#, so the chip reports a 64-bit bus,
/// and still does after the board is reset.
#[test]
fn the_bcm5701_sees_req64_before_and_after_a_reset() {
    use crate::pci::PciFunction;
    const PCISTATE: usize = 0x70 / 4;
    const BUS_32BIT: u32 = 1 << 4;
    let board = TezroBoard::new(1).unwrap();
    assert_eq!(board.eth.config_dword(PCISTATE) & BUS_32BIT, 0);
    board.power_on();
    assert_eq!(board.eth.config_dword(PCISTATE) & BUS_32BIT, 0);
}

/// The board identity the PROM reads: Bedrock board word 0x2001 and the
/// L1's Tezro station address.
#[test]
fn the_board_reports_the_tezro_board_word() {
    let board = TezroBoard::new(1).unwrap();
    let id = board.bus.read64(0x1000_0000 + 0x400_0000);
    assert_eq!((id.status, id.data), (BUS_OK, BOARD_ID));
    assert_eq!(board.l1.station_mac(), L1Config::default().mac);
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
/// derives from the board word's top nibble (0x4 gives 2, 0x8 gives 1, any
/// other 0). Any difference prints "CPU configuration values mismatch".
#[test]
fn the_node_board_record_agrees_with_the_flash_configuration() {
    let class = match BOARD_ID & 0xf000 {
        0x4000 => 2u32,
        0x8000 => 1,
        _ => 0,
    };
    for cpus in [1, 2, 4] {
        let board = TezroBoard::new(cpus).unwrap();
        let area = board.l1.record(0, 2).expect("component 0's board area");
        assert_eq!(area.iter().fold(0u8, |s, b| s.wrapping_add(*b)), 0, "the area's checksum");
        assert_eq!(area[0], 2, "the format that carries CPU fields");
        let fields = area_fields(&area);
        let identity = node_board(&CHIPS[0], cpus);
        assert_eq!([fields[1], fields[3], fields[5]], [identity.name, identity.part, identity.revision].map(str::as_bytes),
            "{cpus} CPUs: the node board's name, part and revision");
        assert_eq!(board.l1.record(1, 2), None, "{cpus} CPUs: no I/O board record");
        assert_eq!(fields[8], [3], "the CPU-field selector");
        let word = |f: &[u8]| u32::from_be_bytes(f.try_into().unwrap());
        let (config0, hub_mhz, cpu_mhz) = (word(fields[10]), word(fields[14]), word(fields[15]));
        for empty in [9, 11, 12, 13, 16, 17, 18] {
            assert!(fields[empty].is_empty(), "field {empty} is unknown and left empty");
        }
        let block = &board.flash.contents()[0x60..0xb0];
        let at = |o: usize, n: usize| block[o..o + n].iter().fold(0u64, |v, b| v << 8 | u64::from(*b));
        assert_eq!(at(0x04, 4), u64::from(config0), "{cpus} CPUs: the mode word");
        assert_eq!(at(0x10, 8), u64::from(cpu_mhz) * 1_000_000, "{cpus} CPUs: the CPU clock");
        assert_eq!(at(0x18, 8), u64::from(hub_mhz) * 1_000_000, "{cpus} CPUs: the Hub clock");
        assert_eq!(at(0x30, 4), u64::from(class), "{cpus} CPUs: the board word's class");
    }
    for chip in &CHIPS {
        let context = cpu_context(&chip.firmware);
        assert_eq!(context.mode_word, chip.firmware.cpu_mode_word, "{}", chip.chip.label());
        assert_eq!(context.cpu_mhz * 1_000_000, chip.firmware.cpu_clock_hz, "{}", chip.chip.label());
        assert_eq!(chip.firmware.machine_type, class, "{}", chip.chip.label());
    }
}

/// The L1 answers the system serial request with the text the emu_id gives,
/// and nothing else moves with it: the node board's record keeps its EMUN00
/// serial, which the PROM ties its NVRAM to, so a machine given a new emu_id
/// keeps the settings it saved.
#[test]
fn the_system_serial_moves_and_the_board_serial_stays() {
    let board = |emu_id: u32| {
        let mut cfg = crate::config::MachineConfig::default();
        cfg.machine.profile = crate::config::MachineProfile::TezroIp53;
        cfg.machine.emu_id = emu_id;
        TezroBoard::with_config(TezroBoardConfig {
            payload: &[],
            clock: Arc::new(|| 0),
            flash_path: None,
            station_mac: cfg.network().mac,
            system_serial: Some(cfg.tezro_system_serial()),
            timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
            memory: &MemoryPopulation::default(),
            cpus: 1,
            chip: &CHIPS[0],
            v12: false,
            graphics_pool: (2, 32),
            graphics_raster: (crate::config::RasterUnit::Cpu, "auto"),
        })
        .unwrap()
    };
    let (first, second) = (board(42), board(43));
    assert_eq!(first.l1.config().system_serial.as_deref(), Some("LEMU200420"));
    assert_eq!(second.l1.config().system_serial.as_deref(), Some("LEMU200430"));
    assert_eq!(first.l1.station_mac(), [0x08, 0x00, 0x69, 0xe2, 0x02, 0xa0]);
    for component in 0..2 {
        for area in [1, 2, 4] {
            assert_eq!(first.l1.record(component, area), second.l1.record(component, area), "component {component}, area {area}");
        }
    }
    let area = first.l1.record(0, 2).expect("component 0's board area");
    assert_eq!(area_fields(&area)[2], b"EMUN00");
    assert_eq!(first.l1.config().brick_serial, second.l1.config().brick_serial);
}

/// A board of `cpus` processors of `chip`, with the defaults otherwise.
fn board_with_chip(cpus: usize, chip: &'static TezroChip) -> TezroBoard {
    TezroBoard::with_config(TezroBoardConfig {
        payload: &[],
        clock: Arc::new(|| 0),
        flash_path: None,
        station_mac: L1Config::default().mac,
        system_serial: None,
        timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
        memory: &MemoryPopulation::default(),
        cpus,
        chip,
        v12: false,
        graphics_pool: (2, 32),
        graphics_raster: (crate::config::RasterUnit::Cpu, "auto"),
    })
    .unwrap()
}

/// The node board the L1 names follows the chip as well as the processor
/// count: the 800 MHz chip sits on 030-1956-002 and the 1 GHz chip on the
/// IP59 boards, as their inventories print, and every name is one the
/// PROM's socket-mask table holds.
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
    for (mhz, cpus, [name, part, revision]) in expected {
        let chip = CHIPS.iter().find(|c| c.chip.mhz.to_string() == mhz).unwrap();
        let board = board_with_chip(cpus, chip);
        let area = board.l1.record(0, 2).expect("component 0's board area");
        let fields = area_fields(&area);
        assert_eq!([fields[1], fields[3], fields[5]], [name, part, revision].map(str::as_bytes), "{mhz} MHz, {cpus} CPUs");
    }
    // Every (chip, count) the board accepts has a board, and no other.
    for chip in &CHIPS {
        assert_eq!(chip.boards.iter().map(|(n, _)| *n).collect::<Vec<_>>(), chip.cpus, "{}", chip.chip.label());
    }
}

/// The IO9's head 0 carries a TI PCI-to-PCI bridge at device 2, revision
/// 1, with a type 1 header and no interrupt pin, on every board; nothing
/// answers on the bus behind it.
#[test]
fn the_io9_has_a_ti_bridge_with_an_empty_bus_at_device_2() {
    for cpus in [1, 2, 4] {
        let board = TezroBoard::new(cpus).unwrap();
        let io = Pio(board.bedrock.pio());
        let config = |reg: u64| io.pio_read(config_id(2) + reg, 4);
        assert_eq!(config(0).data, 0xac28_104c, "{cpus} CPUs: vendor and device");
        assert_eq!(config(8).data, 0x0604_0001, "{cpus} CPUs: a PCI-to-PCI bridge, revision 1");
        assert_eq!((config(0xc).data >> 16) & 0x7f, 1, "{cpus} CPUs: header type 1");
        assert_eq!((config(0x3c).data >> 8) & 0xff, 0, "{cpus} CPUs: no interrupt pin");
        let io9 = board.pics.last().unwrap().bus(0);
        let at = DevFn { device: TI_BRIDGE_DEVICE, function: 0 };
        assert!(io9.config_write(at, 0x18, 4, 0x0001_0100), "number the bus behind it 1");
        let behind = io9.find_bus(1).expect("the bridge's secondary bus");
        // An unclaimed read behind a bridge ends in master abort: all ones.
        for device in 0..32 {
            assert_eq!(behind.config_read(DevFn { device, function: 0 }, 0, 4), Some(u32::MAX), "device {device} behind the bridge");
        }
    }
}

/// PIC DMA reaches memory through Bedrock IO in naturally aligned legs, and
/// DMA aimed at any widget but Bedrock IO is refused.
#[test]
fn pic_dma_reaches_memory_through_bedrock_io_only() {
    let board = TezroBoard::new(1).unwrap();
    // Enable bank pair 0, as the Fuel's DMA tests do.
    assert_eq!(board.bedrock.registers().write64(crate::dev::bedrock::reg::MD_BASE + 8, 0x33), BUS_OK);
    let peer = BedrockPeer(Arc::downgrade(&board.bedrock));
    let data: Vec<u8> = (1..=19).collect();
    assert_eq!(peer.dma_write(15, HOST_WIDGET, 0x1003, &data), Ok(()));
    let mut back = vec![0; data.len()];
    assert_eq!(peer.dma_read(15, HOST_WIDGET, 0x1003, &mut back), Ok(()));
    assert_eq!(back, data);
    let memory = board.bedrock.dma_view();
    assert_eq!(memory.read64(0x1000).data, 0x0000_0001_0203_0405);
    assert_eq!(peer.dma_write(15, 12, 0x1003, &data), Err(Refused));
    assert_eq!(peer.dma_read(15, 0, 0x1003, &mut back), Err(Refused));
}

#[test]
fn transfer_legs_keep_aligned_widths() {
    let legs: Vec<_> = BedrockPeer::legs(0x1003, 19).collect();
    assert_eq!(legs, [(0, 1), (1, 4), (5, 8), (13, 4), (17, 2)]);
    assert_eq!(BedrockPeer::legs(0x2000, 16).collect::<Vec<_>>(), [(0, 8), (8, 8)]);
    assert_eq!(BedrockPeer::legs(0x2000, 0).count(), 0);
}

/// The two- and four-processor board adds a second PIC at widget 11 with
/// nothing on its buses, keeps the IO9 on widget 15 behind widget 0, and
/// leaves the V12's widget 12 empty.
#[test]
fn two_or_four_processors_bring_the_second_pic_at_widget_11() {
    for cpus in [2, 4] {
        let board = TezroBoard::new(cpus).unwrap();
        let io = Pio(board.bedrock.pio());
        assert_eq!(board.pics.iter().map(|p| p.widget_number()).collect::<Vec<_>>(), [11, 15]);
        assert_eq!(io.pio_read(widget(11, 0), 8).data, u64::from(crate::dev::pic::identity(0)));
        assert_eq!(io.pio_read(widget(15, 0), 8).data, u64::from(crate::dev::pic::identity(0)));
        assert_eq!(io.pio_read(widget(12, 0), 8).status, BUS_ERR, "no V12");
        assert_eq!((io.pio_read(widget(0, 4), 4).data >> 12) & 0xffff, u64::from(crate::dev::pxbow::PART));
        for device in 1..=4 {
            assert_eq!(io.pio_read(widget(11, 0x2_0000 + (device << 12)), 4).status, BUS_ERR, "widget 11 device {device}");
        }
        assert_eq!(io.pio_read(config_id(1), 4).data, 0x100a_10a9);
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

fn read_word(board: &TezroBoard, address: u64) -> u64 {
    let r = pio_read(&board.bedrock, address, 4);
    assert_eq!(r.status, BUS_OK, "{address:#x}");
    r.data
}

/// Widget 0's part and manufacturer, from its identity word.
fn widget_zero_part(board: &TezroBoard) -> (u64, u64) {
    let id = read_word(board, widget(0, 4));
    ((id >> 12) & 0xffff, (id >> 1) & 0x7ff)
}

fn alive_and_present(board: &TezroBoard, n: u64) -> (bool, bool) {
    (read_word(board, link_word(n, LINK_STATUS)) & ALIVE != 0, read_word(board, link_word(n, LINK_AUX)) & PRESENT != 0)
}

/// The PROM's per-link test: alive and present, except that link 15 is
/// taken regardless when widget 0 is part 0xd100.
fn prom_uses(board: &TezroBoard, n: u64) -> bool {
    let (alive, present) = alive_and_present(board, n);
    (alive && present) || (n == 15 && widget_zero_part(board).0 == 0xd100)
}

/// The kernel's: alive, and present unless widget 0 is part 0xd100 with
/// manufacturer 0 and the link is 15.
fn kernel_uses(board: &TezroBoard, n: u64) -> bool {
    let (alive, present) = alive_and_present(board, n);
    alive && (present || (n == 15 && widget_zero_part(board) == (0xd100, 0)))
}

/// Links 8 to 15 that pass both tests on their own bits, without either
/// guest's link-15 exception, as a widget mask.
fn live_links(board: &TezroBoard) -> u16 {
    (8..16).filter(|&n| alive_and_present(board, n) == (true, true)).fold(0, |m, n| m | 1 << n)
}

/// A board whose clock the test advances.
fn board_with_clock(cpus: usize) -> (TezroBoard, Arc<AtomicU64>) {
    let now = Arc::new(AtomicU64::new(0));
    let clock = now.clone();
    let board = TezroBoard::with_config(TezroBoardConfig {
        payload: &[],
        clock: Arc::new(move || clock.load(Ordering::SeqCst)),
        flash_path: None,
        station_mac: L1Config::default().mac,
        system_serial: None,
        timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
        memory: &MemoryPopulation::default(),
        cpus,
        chip: &CHIPS[0],
        v12: false,
        graphics_pool: (2, 32),
        graphics_raster: (crate::config::RasterUnit::Cpu, "auto"),
    })
    .unwrap();
    (board, now)
}

/// On the two- and four-processor board widget 0's crossbar shows the
/// hub's link 10 and both PICs' links 11 and 15 alive and present, with no
/// failure mode and arbitration bits 0 and 2 clear, so the PROM and the
/// kernel both take 11 and 15 without leaning on the link-15 exception.
/// Every other link reads empty. A power-on keeps the same picture.
#[test]
fn widget_0_shows_both_pic_links_live_on_the_larger_board() {
    for cpus in [2, 4] {
        let board = TezroBoard::new(cpus).unwrap();
        for pass in ["built", "after power-on"] {
            assert_eq!(widget_zero_part(&board), (0xd100, 0), "{cpus} CPUs, {pass}");
            assert_eq!(read_word(&board, widget(0, 4)) & 1, 1);
            assert_eq!(live_links(&board), 1 << 10 | 1 << 11 | 1 << 15, "{cpus} CPUs, {pass}");
            for n in [11, 15] {
                assert!(prom_uses(&board, n) && kernel_uses(&board, n), "link {n}, {cpus} CPUs, {pass}");
                assert_eq!(read_word(&board, link_word(n, LINK_AUX)) & FAILURE_MODE, 0, "link {n}");
                assert_eq!(read_word(&board, link_word(n, LINK_ARB_UPPER)) & 0b101, 0, "link {n}: unclaimed");
            }
            for n in [8, 9, 12, 13, 14] {
                assert_eq!(alive_and_present(&board, n), (false, false), "link {n}, {cpus} CPUs, {pass}");
                assert!(!prom_uses(&board, n) && !kernel_uses(&board, n));
            }
            board.power_on();
        }
    }
}

/// The one-processor board has no second PIC: link 11 reads empty and
/// widget 11 does not answer, while link 15 stays usable.
#[test]
fn the_one_processor_board_shows_no_link_11() {
    let board = TezroBoard::new(1).unwrap();
    assert_eq!(live_links(&board), 1 << 10 | 1 << 15);
    assert_eq!(alive_and_present(&board, 11), (false, false));
    assert!(!prom_uses(&board, 11) && !kernel_uses(&board, 11));
    assert!(prom_uses(&board, 15) && kernel_uses(&board, 15));
    assert_eq!(pio_read(&board.bedrock, widget(11, 4), 4).status, BUS_ERR);
}

/// Widget 11 answers as a bridge-class PIC with both heads, the identity
/// the PROM's "Found PIC" path needs (manufacturer 0x24, bit 0 set, part
/// 0xd102 on head 0), and takes the PROM's control-word writes on both
/// heads. Widget 15 keeps its own.
#[test]
fn widget_11_answers_as_a_pic_with_both_heads() {
    use crate::dev::pic::HEAD_OFFSET;
    let board = TezroBoard::new(4).unwrap();
    let io = Pio(board.bedrock.pio());
    for (head, part) in [(0, 0xd102), (HEAD_OFFSET, 0xd112)] {
        let id = read_word(&board, widget(11, head + 4));
        assert_eq!((id & 1, (id >> 1) & 0x7ff, (id >> 12) & 0xffff), (1, 0x24, part), "head at {head:#x}");
        // Widget control, as the PROM's table-driven field writes it.
        let control = widget(11, head + 0x24);
        let before = read_word(&board, control);
        assert_eq!(io.pio_write(control, 4, (before & !0xf000) | 0x4000), BUS_OK);
        assert_eq!(read_word(&board, control) & 0xf000, 0x4000, "head at {head:#x}");
    }
    assert_eq!(read_word(&board, widget(15, 4)) >> 12 & 0xffff, 0xd102);
}

/// The PROM's bring-up of link 11 (read the status clear, write credit 2,
/// clear both arbitration registers, then the 0xd102 credit 3) leaves the
/// link alive and present and the credit field reading back.
#[test]
fn the_proms_link_bring_up_keeps_link_11_alive() {
    let board = TezroBoard::new(4).unwrap();
    let io = Pio(board.bedrock.pio());
    read_word(&board, link_word(11, LINK_STATUS_CLEAR));
    assert_eq!(io.pio_write(link_word(11, LINK_CONTROL), 4, 0x0008_0000), BUS_OK);
    assert_eq!(io.pio_write(link_word(11, LINK_ARB_UPPER), 4, 0), BUS_OK);
    assert_eq!(io.pio_write(link_word(11, LINK_ARB_LOWER), 4, 0), BUS_OK);
    assert_eq!(alive_and_present(&board, 11), (true, true));
    let control = read_word(&board, link_word(11, LINK_CONTROL));
    assert_eq!((control >> 18) & 0x1f, 2);
    assert_eq!(io.pio_write(link_word(11, LINK_CONTROL), 4, (control & !(0x1f << 18)) | 3 << 18), BUS_OK);
    assert_eq!((read_word(&board, link_word(11, LINK_CONTROL)) >> 18) & 0x1f, 3);
    assert_eq!(alive_and_present(&board, 11), (true, true));
    assert_eq!(io.pio_read(widget(11, 4), 4).status, BUS_OK);
}

/// The processor reaches widget 11 through the crossbar, not around it: a
/// reset of link 11 or of the hub's link 10 takes widget 11 out of reach
/// for the pulse, and it answers again, registers kept, once it ends. A
/// reset of link 15 leaves widget 11 alone.
#[test]
fn widget_11_is_routed_through_widget_0s_crossbar() {
    let (board, now) = board_with_clock(4);
    let io = Pio(board.bedrock.pio());
    let route = widget(11, 0x130);
    assert_eq!(io.pio_write(route, 8, 0x1122_3344_5566_7788), BUS_OK);
    for link in [11, 10] {
        now.fetch_add(100, Ordering::SeqCst);
        assert_eq!(io.pio_write(link_word(link, LINK_RESET), 4, 0), BUS_OK);
        assert_eq!(io.pio_read(route, 8).status, BUS_ERR, "down during link {link}'s pulse");
        now.fetch_add(20, Ordering::SeqCst);
        let r = io.pio_read(route, 8);
        assert_eq!((r.status, r.data), (BUS_OK, 0x1122_3344_5566_7788), "after link {link}'s pulse");
        assert_eq!(alive_and_present(&board, 11), (true, true));
    }
    now.fetch_add(100, Ordering::SeqCst);
    assert_eq!(io.pio_write(link_word(15, LINK_RESET), 4, 0), BUS_OK);
    assert_eq!(io.pio_read(route, 8).status, BUS_OK, "link 15's pulse");
    assert_eq!(io.pio_read(widget(15, 4), 4).status, BUS_ERR);
    assert_eq!(board.pics[1].pxbow().unwrap().endpoint_widgets(), 1 << 11 | 1 << 15);
    assert!(board.pics[0].pxbow().is_none(), "the second PIC has no crossbar of its own");
}

/// Every fitted processor slot reads present, in the board's order (CPU 1
/// in PI1/A, CPUs 2 and 3 in the B slots), and every fitted processor runs
/// from reset with its enable set, as QEMU's node grants them.
#[test]
fn fitted_processors_read_present_and_start_enabled() {
    use crate::dev::bedrock::reg::{pi::{CPU_A_ENABLE, CPU_B_ENABLE, INTERFACE_SLOT_A, INTERFACE_SLOT_B}, PI1_BASE};
    for (cpus, present) in [(1, [[1, 0], [0, 0]]), (2, [[1, 0], [1, 0]]), (4, [[1, 1], [1, 1]])] {
        let board = TezroBoard::new(cpus).unwrap();
        assert_eq!(board.processors.count(), cpus);
        let regs = board.bedrock.registers();
        for interface in 0..2u64 {
            let pi = |at: u64| regs.read64(interface * PI1_BASE + at);
            let slots = [pi(INTERFACE_SLOT_A).data, pi(INTERFACE_SLOT_B).data];
            assert_eq!(slots, present[interface as usize], "{cpus} CPUs, PI{interface}");
            let enabled = [pi(CPU_A_ENABLE).data & 1, pi(CPU_B_ENABLE).data & 1];
            assert_eq!(enabled, present[interface as usize], "{cpus} CPUs, PI{interface}");
        }
    }
}

/// Only the counts a board fits are accepted, and only those its chip is
/// fitted with.
#[test]
fn the_board_refuses_a_population_it_cannot_fit() {
    assert!(TezroBoard::new(3).is_err());
    assert!(TezroBoard::new(0).is_err());
    assert!(TezroBoard::new(5).is_err());
    let one_ghz = CHIPS.iter().find(|c| c.chip.mhz == 1000).unwrap();
    let refused = TezroBoard::with_config(TezroBoardConfig {
        payload: &[],
        clock: Arc::new(|| 0),
        flash_path: None,
        station_mac: L1Config::default().mac,
        system_serial: None,
        timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
        memory: &MemoryPopulation::default(),
        cpus: 1,
        chip: one_ghz,
        v12: false,
        graphics_pool: (2, 32),
        graphics_raster: (crate::config::RasterUnit::Cpu, "auto"),
    });
    assert!(refused.err().is_some_and(|e| e.contains("fitted with 2, 4 processors only")));
}

/// A board of one GiB of memory with `chip` fitted, as the fewest
/// processors it takes.
fn board_with(chip: &Part) -> TezroBoard {
    TezroBoard::with_config(TezroBoardConfig {
        payload: &[],
        clock: Arc::new(|| 0),
        flash_path: None,
        station_mac: L1Config::default().mac,
        system_serial: None,
        timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
        memory: &MemoryPopulation::one_gib(),
        cpus: usize::from(self::chip(chip).cpus[0]),
        chip: self::chip(chip),
        v12: false,
        graphics_pool: (2, 32),
        graphics_raster: (crate::config::RasterUnit::Cpu, "auto"),
    })
    .unwrap()
}

/// Run `body` on a thread with the stack a debug build's processor needs.
fn on_big_stack(body: impl FnOnce() + Send + 'static) {
    if let Err(panic) = std::thread::Builder::new().stack_size(64 << 20).spawn(body).unwrap().join() {
        std::panic::resume_unwind(panic);
    }
}

/// Each chip resets with its own PRId, FIR and Config word, and keeps them
/// across a CPU reset, which reloads PRId and FIR from the reset values. The
/// board's flash record and Count rate follow the chip's clock. These are the
/// facts the chips grade as inferred or approximated, so each is pinned here.
#[test]
fn each_tezro_chip_resets_with_its_identity_and_clock() {
    fn check<C: CacheCpuModel + From<Arc<dyn BusDevice>>>(model: &Part, prid: u32, config0: u32, mhz: u32) {
        let board = board_with(model);
        let mut cpu = build_cpu::<C>(&board);
        let name = model.name();
        assert_eq!((cpu.core.cp0_prid, cpu.core.reset_prid), (prid, prid), "{name}");
        assert_eq!(model.prid(), prid, "{name}: the chip's PRId");
        assert_eq!((cpu.core.fpu_fir, cpu.core.reset_fir), (0x900, 0x900), "{name}");
        assert_eq!(cpu.core.cp0_config & CONFIG0_MODE_MASK, config0, "{name}");
        assert_eq!(cpu.core.cp0_config >> 26, 0b011_011, "{name}: 32 KB primary caches");
        let config = cpu.core.cp0_config;
        cpu.core.fpu_fir = 0;
        cpu.core.cp0_prid = 0;
        cpu.core.reset(false);
        assert_eq!((cpu.core.cp0_prid, cpu.core.fpu_fir, cpu.core.cp0_config), (prid, 0x900, config), "{name}: after reset");
        assert_eq!(model.mhz, mhz);
        assert_eq!(model.count_hz(), u64::from(mhz) * 500_000, "{name}: Count at half the clock");
        assert_eq!(chip(model).firmware.cpu_clock_hz, mhz * 1_000_000, "{name}: the PROM's clock record");
    }
    use crate::cpu::mips_cache_shadow::{R16000Scache16ShadowCache, R16000Scache8ShadowCache, R16000ShadowCache};
    on_big_stack(|| {
        check::<R16000ShadowCache>(&parts::R16000_700_EARLY, 0x0f41, 0x009b_ad85, 700);
        check::<R16000ShadowCache>(&parts::R16000_800, 0x0f42, 0x009b_af85, 800);
        check::<R16000Scache8ShadowCache>(&parts::R16000_900, 0x0f50, 0x009c_b185, 900);
        check::<R16000Scache16ShadowCache>(&parts::R16000_1000, 0x0f50, 0x00ed_b385, 1000);
    });
}

/// Every chip's Config word decodes, by the PROM editor's formulas, to the
/// fields its inventory states: SS gives the cache size, SC the printed cache
/// clock (2 * CPU / (SC + 1)) and EC the 200 MHz Bedrock (code 2 * CPU /
/// Bedrock - 1), with K0 5, PM 3, SB and BE set. The 1 GHz word is the only
/// approximated one, and this is what it is built from: SS 5, SC 5, EC 9.
#[test]
fn each_tezro_configs_fields_decode_to_the_stated_sizes_and_clocks() {
    for c in &CHIPS {
        let word = c.firmware.reset_mode();
        let name = c.chip.name();
        let field = |lo: u32, bits: u32| (word >> lo) & ((1 << bits) - 1);
        let mhz = c.chip.mhz;
        let bytes = crate::with_cpu!(c.chip, true, false, C => <C as MipsCache>::L2_SIZE as u32);
        assert_eq!((field(0, 3), field(7, 2), field(13, 1), field(15, 1)), (5, 3, 1, 1), "{name}: K0 PM SB BE");
        assert_eq!(field(9, 4), 2 * mhz / 200 - 1, "{name}: EC");
        assert_eq!(2 * mhz / (field(19, 3) + 1), c.scache_mhz, "{name}: SC gives the cache clock");
        assert_eq!(512u32 << 10 << field(16, 3), bytes, "{name}: SS gives the cache size");
        assert_eq!((word >> 23) & 1, 1, "{name}: bit 23 is set in every row of 500 MHz and up");
        assert_eq!(word >> 24, 0, "{name}: bit 24 clear");
    }
    let word = chip(&parts::R16000_1000).firmware.reset_mode();
    assert_eq!(((word >> 16) & 7, (word >> 19) & 7, (word >> 9) & 15, (word >> 22) & 1), (5, 5, 9, 1), "1 GHz SS SC EC and bit 22");
}

/// A part's revision, as hinv prints it, and the PRId register's raw one
/// differ for an R16000: the register's major is the displayed one plus 2 (the
/// rule Linux and OpenBSD apply). The raw bytes are written out here, not
/// derived, so this ties the Tezro's "2.1", "2.2", "3.0" and "3.0" to 0x0f41,
/// 0x0f42 and 0x0f50, the 1 GHz one also printed by the PROM banner as `f50`.
#[test]
fn every_parts_revision_is_the_one_its_prid_displays() {
    for (part, shown, raw) in
        CPUS.iter().zip(["2.1", "2.2", "3.0", "3.0"]).zip([0x41, 0x42, 0x50, 0x50]).map(|((p, s), r)| (p, s, r))
    {
        assert_eq!(format!("{}.{}", part.rev.0, part.rev.1), shown, "{}", part.label());
        assert_eq!(part.prid(), 0x0f00 | raw, "{}", part.label());
    }
}

/// A chip and its flash record are two statements of one chip: the clock,
/// and the Config word the PROM compares against the processor's, must agree.
#[test]
fn each_tezro_chip_agrees_with_its_flash_record() {
    for c in &CHIPS {
        let record = c.firmware;
        let name = c.chip.name();
        assert_eq!(u64::from(record.cpu_clock_hz), u64::from(c.chip.mhz) * 1_000_000, "{name}");
        assert_eq!(record.cpu_mode_word & CONFIG0_MODE_MASK, record.reset_mode(), "{name}");
        assert_eq!(board_with(c.chip).chip.chip, c.chip);
        assert!(c.cpus.iter().all(|&n| n <= 4) && !c.cpus.is_empty(), "{name}");
    }
    assert_eq!(CHIPS.len(), CPUS.len());
    assert!(CHIPS.iter().zip(CPUS).all(|(c, &model)| c.chip == model), "CPUS lists the chips in table order");
}

/// The cache type `with_cpu!` builds follows the chip: the secondary-cache
/// size the chip is inventoried with, and nothing else about the type is the
/// chip's.
#[test]
fn the_processors_take_the_cache_type_of_the_chips_size() {
    for (model, mib) in [
        (&parts::R16000_700_EARLY, 4u32),
        (&parts::R16000_800, 4),
        (&parts::R16000_900, 8),
        (&parts::R16000_1000, 16),
    ] {
        let (name, bytes) = crate::with_cpu!(model, true, false, C => (<C as CacheCpuModel>::NAME, <C as MipsCache>::L2_SIZE as u32));
        assert_eq!((name, bytes >> 20), ("R16000", mib), "{}", model.name());
    }
}

/// A bare family name builds the 700 MHz chip, and a chip name its own:
/// PRId, flash record and Count rate follow, end to end from a configuration.
#[test]
fn the_family_name_builds_the_700_mhz_chip_and_a_chip_name_its_own() {
    on_big_stack(|| {
        for (cpu, prid, mhz, mib) in [("r16000", 0x0f41, 700, 4u32), ("r16000_800", 0x0f42, 800, 4), ("r16000_900", 0x0f50, 900, 8), ("r16000_1000", 0x0f50, 1000, 16)] {
            let cfg: MachineConfig = toml::from_str(&format!("[machine]\nprofile = \"tezro_ip53\"\ncpu = \"{cpu}\"\n")).unwrap();
            let part = cfg.cpu_part().unwrap();
            let board = board_with(part);
            let (core_prid, core_config, l2) = crate::with_cpu!(part, true, false, C => {
                let cpu_core = build_cpu::<C>(&board);
                (cpu_core.core.cp0_prid, cpu_core.core.cp0_config, <C as MipsCache>::L2_SIZE as u32)
            });
            assert_eq!(core_prid, prid, "{cpu}");
            assert_eq!(l2 >> 20, mib, "{cpu}: secondary cache");
            assert_eq!((core_config >> 16) & 7, 1 + mib.trailing_zeros(), "{cpu}: Config SS");
            assert_eq!(board.chip.firmware.cpu_clock_hz, mhz * 1_000_000, "{cpu}: the flash record");
            assert_eq!(board.chip.firmware.reset_config0(), core_config, "{cpu}: record and processor agree");
        }
    });
}

/// IOC4 reads its 32-bit registers in the processor's byte order through
/// the IO9 PIC's self-ordered slot: what the processor writes is the
/// register's value, as on the QEMU Tezro, where PROM 6.210 writes the
/// serial DMA control 0x100201ff and the TX producer 0x3f8.
#[test]
fn the_processor_reads_and_writes_ioc4_registers_in_its_own_byte_order() {
    let board = TezroBoard::new(1).unwrap();
    let io = Pio(board.bedrock.pio());
    // IOC4's BAR at PCI memory 0x200000, decoding memory, reached through
    // head 0's first short window.
    assert_eq!(io.pio_write(config_id(1) + 0x10, 4, 0x20_0000), BUS_OK);
    assert_eq!(io.pio_write(config_id(1) + 4, 4, 2), BUS_OK);
    assert_eq!(io.pio_write(widget(15, 0x200), 8, (1 << 12) | 2), BUS_OK);
    let reg = |offset: u64| widget(15, 0x20_0000 + offset);
    assert_eq!(io.pio_read(reg(0x24), 4).data, 1, "MISC: the 66 MHz clock");
    assert_eq!(io.pio_write(reg(0x314), 4, 0x3f8), BUS_OK);
    assert_eq!(io.pio_read(reg(0x314), 4).data, 0x3f8, "port 0 TX producer");
}

// VPro host-interface registers the tests below touch, as byte offsets in
// the board's widget space. The board's own constants are private to it.
const ODSY_IDENTITY: u64 = 0x00;

fn board_with_v12(cpus: usize) -> TezroBoard {
    TezroBoard::with_config(TezroBoardConfig {
        payload: &[],
        clock: Arc::new(|| 0),
        flash_path: None,
        station_mac: L1Config::default().mac,
        system_serial: None,
        timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
        memory: &MemoryPopulation::default(),
        cpus,
        chip: &CHIPS[0],
        v12: true,
        graphics_pool: (2, 32),
        graphics_raster: (crate::config::RasterUnit::Cpu, "auto"),
    })
    .unwrap()
}

/// The V12 answers as widget 12 behind the PXBOW on either board, and
/// widget 12 stays empty without one. (On the one-processor board the
/// placement is inferred; see V12_WIDGET.)
#[test]
fn the_v12_answers_at_widget_12_on_both_boards() {
    for cpus in [1, 4] {
        let board = board_with_v12(cpus);
        let id = pio_read(&board.bedrock, widget(12, ODSY_IDENTITY), 4);
        let direct = board.v12.as_ref().unwrap().read32(ODSY_IDENTITY);
        assert_eq!((id.status, id.data), (BUS_OK, u64::from(direct.data)), "{cpus} CPUs");
        assert_eq!(board.pics[board.pics.len() - 1].pxbow().unwrap().endpoint_widgets() & (1 << 12), 1 << 12);
    }
    assert_eq!(pio_read(&TezroBoard::new(1).unwrap().bedrock, widget(12, ODSY_IDENTITY), 4).status, BUS_ERR);
}

/// The monitor's `vpro` commands resolve on a Tezro that has a V12, and are
/// unknown on one that has none.
#[test]
fn the_monitor_resolves_vpro_commands_on_a_tezro_with_a_v12() {
    let resolves = |board: &TezroBoard, command: &str| {
        board.monitor_devices().iter().any(|d| d.register_commands().iter().any(|(c, _)| c == command))
    };
    let with = board_with_v12(4);
    assert!(resolves(&with, "vpro"), "a Tezro with a V12 answers `vpro`");
    assert!(resolves(&with, "ps2"), "the other board commands stay");
    let without = TezroBoard::new(1).unwrap();
    assert!(!resolves(&without, "vpro"), "a Tezro with no V12 has no `vpro`");
    assert!(resolves(&without, "ps2"));
}

/// The cached view the processors' caches sit on swaps RAM above 4 GiB, so
/// SC stays atomic there too.
#[test]
fn the_cached_view_swaps_ram_above_4_gib() {
    let board = TezroBoard::new(4).unwrap();
    // Banks 0 to 5: two 1 GiB banks and four of 512 MiB, as MD's tests set them.
    assert_eq!(board.bedrock.registers().write64(crate::dev::bedrock::reg::MD_BASE + 8, 0x53 | (0x43 << 8) | (0x43 << 16)), BUS_OK);
    let cached = board.cached_bus();
    let at = (5u64 << 30) + 0x80;
    assert_eq!(cached.write64(at, 1), BUS_OK);
    assert_eq!(cached.cas64(at, 1, 2), Some(true));
    assert_eq!(cached.cas64(at, 1, 3), Some(false));
    assert_eq!(cached.read64(at).data, 2);
}

/// A fresh flash holds no environment records: with the node board named
/// for its processor count the PROM tries only the fitted positions, so no
/// board needs a Disable record to boot straight through.
#[test]
fn a_fresh_flash_starts_with_no_environment_records() {
    for cpus in [1, 2, 4] {
        let bytes = TezroBoard::new(cpus).unwrap().flash.contents();
        assert!(bytes[0x9e0100..0x9e0200].iter().all(|&b| b == 0xff), "{cpus} CPUs");
    }
}
