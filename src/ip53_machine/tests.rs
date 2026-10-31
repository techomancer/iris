use super::*;
use crate::traits::{BUS_ERR, BUS_OK};
use std::sync::atomic::AtomicU64;

/// A PIO offset in Bedrock IO's small window: widget in bits 27:24.
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
    let io = &board.node.io;
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
/// revisions the QEMU profile selects, and device 2 is empty.
#[test]
fn the_io9_bus_holds_the_ioc4_the_isp12160_and_the_bcm5701() {
    let board = TezroBoard::new(1).unwrap();
    let io = &board.node.io;
    for (device, id, revision) in [(1, 0x100a_10a9, 0x53), (3, 0x1216_1077, 0x06), (4, 0x1645_14e4, 0x00)] {
        assert_eq!(io.pio_read(config_id(device), 4).data, id, "device {device}");
        assert_eq!(io.pio_read(config_id(device) + 8, 4).data & 0xff, revision, "device {device}");
    }
    assert_eq!(io.pio_read(config_id(2), 4).status, BUS_ERR, "device 2 is empty");
}

/// The BCM5701's slot drives REQ64#, so the chip reports a 64-bit bus,
/// and still does after the board is reset.
#[test]
fn the_bcm5701_sees_req64_before_and_after_a_reset() {
    use crate::pci::PciBusDevice;
    const PCISTATE: usize = 0x70 / 4;
    const BUS_32BIT: u32 = 1 << 4;
    let board = TezroBoard::new(1).unwrap();
    assert_eq!(board.eth.config().read(PCISTATE) & BUS_32BIT, 0);
    board.power_on();
    assert_eq!(board.eth.config().read(PCISTATE) & BUS_32BIT, 0);
}

/// The board identity the PROM reads: Bedrock board word 0x2001 and the
/// L1's Tezro station address.
#[test]
fn the_board_reports_the_tezro_board_word() {
    let board = TezroBoard::new(1).unwrap();
    let id = board.node.bus.read64(0x1000_0000 + 0x400_0000);
    assert_eq!((id.status, id.data), (BUS_OK, BOARD_ID));
    assert_eq!(board.node.l1.station_mac(), L1Config::default().mac);
}

/// PIC DMA reaches memory through Bedrock IO in naturally aligned legs, and
/// DMA aimed at any widget but Bedrock IO is refused.
#[test]
fn pic_dma_reaches_memory_through_bedrock_io_only() {
    let board = TezroBoard::new(1).unwrap();
    // Enable bank pair 0, as the Fuel's DMA tests do.
    assert_eq!(board.node.md.md_bus().write64(8, 0x33), BUS_OK);
    let peer = BedrockPeer(Arc::downgrade(&board.node.io));
    let data: Vec<u8> = (1..=19).collect();
    assert_eq!(peer.dma_write(15, HOST_WIDGET, 0x1003, &data), Ok(()));
    let mut back = vec![0; data.len()];
    assert_eq!(peer.dma_read(15, HOST_WIDGET, 0x1003, &mut back), Ok(()));
    assert_eq!(back, data);
    let memory = board.node.md.dma_view();
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
        let io = &board.node.io;
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
    let r = board.node.io.pio_read(address, 4);
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
        timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
        memory: &MemoryPopulation::default(),
        cpus,
        part: TezroPart::Mhz700,
        v12: false,
        graphics_pool: (2, 32),
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
    assert_eq!(board.node.io.pio_read(widget(11, 4), 4).status, BUS_ERR);
}

/// Widget 11 answers as a bridge-class PIC with both heads, the identity
/// the PROM's "Found PIC" path needs (manufacturer 0x24, bit 0 set, part
/// 0xd102 on head 0), and takes the PROM's control-word writes on both
/// heads. Widget 15 keeps its own.
#[test]
fn widget_11_answers_as_a_pic_with_both_heads() {
    use crate::dev::pic::HEAD_OFFSET;
    let board = TezroBoard::new(4).unwrap();
    let io = &board.node.io;
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
    let io = &board.node.io;
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
    let io = &board.node.io;
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
/// in PI1/A, CPUs 2 and 3 in the B slots), and only CPU 0 is enabled.
#[test]
fn fitted_processors_read_present_and_only_cpu_0_is_enabled() {
    use crate::dev::bedrock_pi::reg::{CPU_A_ENABLE, CPU_B_ENABLE, INTERFACE_SLOT_A, INTERFACE_SLOT_B};
    for (cpus, present) in [(1, [[1, 0], [0, 0]]), (2, [[1, 0], [1, 0]]), (4, [[1, 1], [1, 1]])] {
        let board = TezroBoard::new(cpus).unwrap();
        for (interface, pi) in board.node.pi.iter().enumerate() {
            let slots = [pi.read64(INTERFACE_SLOT_A).data, pi.read64(INTERFACE_SLOT_B).data];
            assert_eq!(slots, present[interface], "{cpus} CPUs, PI{interface}");
            let enabled = [pi.read64(CPU_A_ENABLE).data & 1, pi.read64(CPU_B_ENABLE).data & 1];
            assert_eq!(enabled, [u64::from(interface == 0), 0], "{cpus} CPUs, PI{interface}");
        }
    }
}

/// What the PIs ask of CPU 0 that the board cannot do is counted, not
/// dropped: an NMI, and a write that clears its enable bit. Setting the bit,
/// and NMIs or enables for the other slots, which the PI records, are not.
#[test]
fn an_nmi_for_cpu_0_and_a_write_that_stops_it_are_refused() {
    use crate::dev::bedrock_pi::reg::{CPU_A_ENABLE, CPU_B_ENABLE, NMI_SLOT_A, NMI_SLOT_B};
    let board = TezroBoard::new(4).unwrap();
    let count = || (board.cpu0_refused.nmi.load(Ordering::Relaxed), board.cpu0_refused.stop.load(Ordering::Relaxed));
    let [pi0, pi1] = [&board.node.pi[0], &board.node.pi[1]];
    pi0.write64(CPU_A_ENABLE, 1);
    pi0.write64(NMI_SLOT_B, 1);
    pi0.write64(CPU_B_ENABLE, 0);
    pi1.write64(NMI_SLOT_A, 1);
    pi1.write64(CPU_A_ENABLE, 0);
    assert_eq!(count(), (0, 0), "not CPU 0's, or not a refusal");
    pi0.write64(NMI_SLOT_A, 1);
    pi0.write64(NMI_SLOT_A, 1);
    pi0.write64(CPU_A_ENABLE, 0);
    assert_eq!(count(), (2, 1));
    use crate::dev::bedrock_pi::Refused as PiRefused;
    assert_eq!(pi0.refused(), [(PiRefused::Nmi { interface: 0, slot: 1 }, 1)], "the PI leaves CPU 0's to the board");
}

/// Only the counts a board fits are accepted.
#[test]
fn the_board_refuses_a_population_it_cannot_fit() {
    assert!(TezroBoard::new(3).is_err());
    assert!(TezroBoard::new(0).is_err());
    assert!(TezroBoard::new(5).is_err());
}

/// Each R16000 part resets with the PRId and Config word the CPU models give
/// it, and the board's flash and count rate follow its clock. `r10000` takes
/// the default 700 MHz part; other parts are refused.
#[test]
fn each_r16000_part_resets_with_its_own_identity_and_clock() {
    // Building a processor takes more stack than a debug build's test thread has.
    std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
        use crate::config::CpuModel;
        use crate::cpu::mips_cache_shadow::{CONFIG0_RESET_MASK, R16000_800_CONFIG0, R16000_MHZ700_CONFIG0};
        assert_eq!(TezroPart::of(CpuModel::R16000Mhz700), Ok(TezroPart::Mhz700));
        assert_eq!(TezroPart::of(CpuModel::R10000), Ok(TezroPart::Mhz700));
        assert_eq!(TezroPart::of(CpuModel::R16000), Ok(TezroPart::Mhz800));
        for cpu in [CpuModel::R4400, CpuModel::R5000, CpuModel::R12000, CpuModel::R14000] {
            assert!(TezroPart::of(cpu).is_err(), "{cpu:?}");
        }
        for (part, prid, config0, mhz) in [
            (TezroPart::Mhz700, R16000_MHZ700_PRID, R16000_MHZ700_CONFIG0, 700),
            (TezroPart::Mhz800, 0x0f42, R16000_800_CONFIG0, 800),
        ] {
            // One GiB keeps the memory below 4 GiB, which a jitv2 build's
            // uncached code views need.
            let board = TezroBoard::with_config(TezroBoardConfig {
                payload: &[],
                clock: Arc::new(|| 0),
                flash_path: None,
                station_mac: L1Config::default().mac,
                timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
                memory: &MemoryPopulation::one_gib(),
                cpus: 1,
                part,
                v12: false,
                graphics_pool: (2, 32),
            })
            .unwrap();
            let cpu = build_cpu::<R16000ShadowCache>(&board);
            assert_eq!((cpu.core.cp0_prid, cpu.core.reset_prid), (prid, prid), "{part:?}");
            assert_eq!(cpu.core.cp0_config & CONFIG0_RESET_MASK, config0, "{part:?}");
            assert_eq!(cpu.core.cp0_config >> 26, 0b011_011, "{part:?}: 32 KB primary caches");
            assert_eq!(part.mhz(), mhz);
            assert_eq!(part.count_hz(), u64::from(mhz) * 500_000, "{part:?}: Count at half the clock");
            assert_eq!(board.part.firmware().cpu_clock_hz, mhz * 1_000_000);
        }
    }).unwrap().join().unwrap();
}

/// The poll returns lifted checked pages to the direct window once every
/// interval of board time while the board runs, not on every batch.
#[test]
fn the_poll_retires_lifted_pages_once_an_interval() {
    let now = Arc::new(AtomicU64::new(0));
    let clock = now.clone();
    let board = TezroBoard::with_config(TezroBoardConfig {
        payload: &[],
        clock: Arc::new(move || clock.load(Ordering::SeqCst)),
        flash_path: None,
        station_mac: L1Config::default().mac,
        timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
        memory: &MemoryPopulation::default(),
        cpus: 1,
        part: TezroPart::Mhz700,
        v12: false,
        graphics_pool: (2, 32),
    })
    .unwrap();
    board.running.store(true, Ordering::SeqCst);
    now.store(RETIRE_INTERVAL_NS - 1, Ordering::SeqCst);
    board.poll();
    assert_eq!(board.last_retire.load(Ordering::SeqCst), 0);
    now.store(RETIRE_INTERVAL_NS, Ordering::SeqCst);
    board.poll();
    assert_eq!(board.last_retire.load(Ordering::SeqCst), RETIRE_INTERVAL_NS);
    board.running.store(false, Ordering::SeqCst);
}

/// IOC4 reads its 32-bit registers in the processor's byte order through
/// the IO9 PIC's self-ordered slot: what the processor writes is the
/// register's value, as on the QEMU Tezro, where PROM 6.210 writes the
/// serial DMA control 0x100201ff and the TX producer 0x3f8.
#[test]
fn the_processor_reads_and_writes_ioc4_registers_in_its_own_byte_order() {
    let board = TezroBoard::new(1).unwrap();
    let io = &board.node.io;
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
const ODSY_CMD_FIFO: u64 = 0x11_0000;

fn board_with_v12(cpus: usize) -> TezroBoard {
    TezroBoard::with_config(TezroBoardConfig {
        payload: &[],
        clock: Arc::new(|| 0),
        flash_path: None,
        station_mac: L1Config::default().mac,
        timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
        memory: &MemoryPopulation::default(),
        cpus,
        part: TezroPart::Mhz700,
        v12: true,
        graphics_pool: (2, 32),
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
        let id = board.node.io.pio_read(widget(12, ODSY_IDENTITY), 4);
        let direct = board.v12.as_ref().unwrap().read32(ODSY_IDENTITY);
        assert_eq!((id.status, id.data), (BUS_OK, u64::from(direct.data)), "{cpus} CPUs");
        assert_eq!(board.pics[board.pics.len() - 1].pxbow().unwrap().endpoint_widgets() & (1 << 12), 1 << 12);
    }
    assert_eq!(TezroBoard::new(1).unwrap().node.io.pio_read(widget(12, ODSY_IDENTITY), 4).status, BUS_ERR);
}

/// A CPU 0 store into its enabled graphics page is counted against PI0's
/// slot A credits, and the credit comes back along the node's graphics map
/// as the V12 takes the store. A store outside the page costs nothing.
/// Follows sn1/pi-gfx/ip53-odyssey-credit-return and
/// uncovered-stores-not-counted in tests/qtest/sn1-pi-gfx-test.c.
#[test]
fn a_counted_store_to_the_v12_returns_its_credit() {
    use crate::dev::bedrock_io::reg::GFX_MAP_BASE;
    use crate::dev::odyssey::{PI_GFX_PAGE_ADDRESS, PI_GFX_PAGE_ENABLE};
    let board = board_with_v12(4);
    let (pi, io, bus) = (&board.node.pi[0], &board.node.io, &board.node.bus);
    let credit = || pi.read64(0x308).data;
    // The page holding the V12's command FIFO, enabled, with 1000 credits.
    let page = widget(12, ODSY_CMD_FIFO) & PI_GFX_PAGE_ADDRESS;
    assert_eq!(pi.write64(0x300, PI_GFX_PAGE_ENABLE | page), BUS_OK);
    assert_eq!(pi.write64(0x308, 1000), BUS_OK);
    // Widget 12's credits go to node 0, PI0, slot A.
    assert_eq!(io.write64(GFX_MAP_BASE, 12), BUS_OK);
    let io_view = 1u64 << 57;
    assert_eq!(bus.write64(io_view | widget(12, ODSY_CMD_FIFO), 0x0001_0046), BUS_OK);
    assert_eq!(credit(), 1000, "one store, one credit, back");
    // Outside the page: neither counted nor returned.
    let _ = bus.write64(io_view | widget(11, 0x100), 0);
    assert_eq!(credit(), 1000);
}

/// A counted store the bus answers busy is made again by the processor and
/// counted again, so the refused attempt must cost nothing: the node puts
/// the credit back. A refund also undoes a count across the wrap.
#[test]
fn a_counted_store_refused_busy_is_refunded() {
    use crate::dev::odyssey::{PI_GFX_PAGE_ADDRESS, PI_GFX_PAGE_ENABLE};
    struct Busy;
    impl BusDevice for Busy {
        fn write64(&self, _addr: u64, _val: u64) -> u32 {
            crate::traits::BUS_BUSY
        }
    }
    let board = board_with_v12(1);
    let (pi, bus) = (&board.node.pi[0], &board.node.bus);
    board.node.io.connect_widget(9, Arc::new(Busy));
    let page = widget(9, 0) & PI_GFX_PAGE_ADDRESS;
    assert_eq!(pi.write64(0x300, PI_GFX_PAGE_ENABLE | page), BUS_OK);
    assert_eq!(pi.write64(0x308, 5), BUS_OK);
    let io_view = 1u64 << 57;
    assert_eq!(bus.write64(io_view | widget(9, 0), 0), crate::traits::BUS_BUSY);
    assert_eq!(pi.read64(0x308).data, 5, "refunded");
    assert_eq!(pi.write64(0x308, 0), BUS_OK);
    assert_eq!(bus.write64(io_view | widget(9, 0), 0), crate::traits::BUS_BUSY);
    assert_eq!(pi.read64(0x308).data, 0, "back across the wrap");
}

/// Follows sn1/pi-gfx/map-rewrite-settles-owed-credits: owed credits are
/// delivered when the entry naming the widget is rewritten, a write of the
/// value already there settles nothing, a repayment of settled credits is
/// dropped, one still owed is routed, and a reset's abandonment cancels
/// what it threw away.
#[test]
fn a_map_rewrite_settles_owed_credits() {
    let mut l = OwedCredits::default();
    let entry = 0xc;
    l.owe(0xc, 5);
    assert_eq!(l.settle(entry, entry), None, "same value");
    assert_eq!(l.settle(entry, 0), Some((0xc, 5)));
    assert_eq!(l.repay(0xc, 5), 0, "already settled");
    assert_eq!(l.settle(0, entry), None, "widget 0 owes nothing");
    l.owe(0xc, 3);
    assert_eq!(l.repay(0xc, 2), 2);
    l.owe(0xc, -1);
    assert_eq!(l.settle(entry, 0), None, "nothing left owed");
    l.owe(0xb, -1);
    assert_eq!(l, OwedCredits::default(), "never below zero");
}

/// On the board, credits the V12 owes reach PI0's slot A when the map
/// entry naming widget 12 is rewritten, under the entry as it stood, and
/// the V12's later repayment of them is dropped.
#[test]
fn the_board_settles_the_v12s_owed_credits_on_a_map_write() {
    use crate::dev::bedrock_io::reg::GFX_MAP_BASE;
    let board = board_with_v12(4);
    let (pi, io) = (&board.node.pi[0], &board.node.io);
    let link = board.v12.as_ref().unwrap().pi_link().unwrap();
    assert_eq!(pi.write64(0x308, 100), BUS_OK);
    assert_eq!(io.write64(GFX_MAP_BASE, 12), BUS_OK);
    link.owe_credits(12, 8);
    assert_eq!(pi.read64(0x308).data, 100, "owed, not yet returned");
    assert_eq!(io.write64(GFX_MAP_BASE, 0), BUS_OK, "the detach");
    assert_eq!(pi.read64(0x308).data, 108);
    assert_eq!(io.write64(GFX_MAP_BASE, 12), BUS_OK);
    link.repay_credits(12, 8);
    assert_eq!(pi.read64(0x308).data, 108, "settled credits are not paid twice");
    link.owe_credits(12, 2);
    link.repay_credits(12, 2);
    assert_eq!(pi.read64(0x308).data, 110);
}

fn stalled_gfx(credits: u64, bias: u64) -> (BedrockGfx, Arc<std::sync::atomic::AtomicU64>) {
    use crate::dev::odyssey::{PI_GFX_BIAS, PI_GFX_CREDIT, PI_GFX_PAGE, PI_GFX_PAGE_ENABLE};
    let now = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let clock = now.clone();
    let mut g = BedrockGfx::new(Arc::new(move || clock.load(Ordering::SeqCst)));
    g.write(PI_GFX_PAGE, PI_GFX_PAGE_ENABLE | GFX_PAGE);
    g.write(PI_GFX_CREDIT, credits);
    g.write(PI_GFX_BIAS, bias);
    (g, now)
}

const GFX_PAGE: u64 = 0x0140_0000;

/// Follows sn1/pi-gfx/stall-below-bias: below the bias the slot stops and
/// its next counted store waits, uncounted, until credits lift the counter
/// above the bias or the processor takes the interrupt; each counted store
/// below the bias asks for the interrupt again. Stores outside the page
/// and the other slot go on.
#[test]
fn a_slot_below_its_bias_holds_its_next_store() {
    use crate::dev::odyssey::{PI_GFX_BIAS, PI_GFX_CREDIT, PI_GFX_INT_COUNT};
    let (mut g, _) = stalled_gfx(6, 5);
    assert_eq!(g.store(0, GFX_PAGE), Store::Counted { below_bias: false }, "at the bias");
    assert_eq!(g.store(0, GFX_PAGE), Store::Counted { below_bias: true });
    assert_eq!(g.store(0, GFX_PAGE), Store::Held);
    assert_eq!(g.read(PI_GFX_CREDIT), 4, "a held store is not counted");
    assert_eq!(g.store(0, 0x0142_0158), Store::Uncounted);
    assert_eq!(g.store(1, GFX_PAGE), Store::Uncounted);
    g.add_credits(0, 1);
    assert_eq!(g.store(0, GFX_PAGE), Store::Held, "5 is not above the bias");
    g.add_credits(0, 1);
    assert_eq!(g.store(0, GFX_PAGE), Store::Counted { below_bias: false });
    assert_eq!(g.store(0, GFX_PAGE), Store::Counted { below_bias: true });
    g.taken(0);
    assert_eq!(g.store(0, GFX_PAGE), Store::Counted { below_bias: true }, "taken: goes at once");
    g.write(PI_GFX_INT_COUNT, 0);
    assert_eq!(g.store(0, GFX_PAGE), Store::Counted { below_bias: true }, "the handler's counter write");
    g.write(PI_GFX_BIAS, 0);
    assert_eq!(g.store(0, GFX_PAGE), Store::Counted { below_bias: false }, "a lower bias");
}

/// The hold is bounded, as QEMU's is, so a slot nobody releases cannot
/// stop its processor for good.
#[test]
fn a_held_store_goes_after_the_limit() {
    use crate::dev::odyssey::PI_GFX_CREDIT;
    let (mut g, now) = stalled_gfx(3, 5);
    assert_eq!(g.store(0, GFX_PAGE), Store::Counted { below_bias: true });
    now.store(10, Ordering::SeqCst);
    assert_eq!(g.store(0, GFX_PAGE), Store::Held);
    now.store(10 + STALL_LIMIT_NS - 1, Ordering::SeqCst);
    assert_eq!(g.store(0, GFX_PAGE), Store::Held);
    now.store(10 + STALL_LIMIT_NS, Ordering::SeqCst);
    assert_eq!(g.store(0, GFX_PAGE), Store::Counted { below_bias: true });
    assert_eq!(g.read(PI_GFX_CREDIT), 1);
    now.store(20 + STALL_LIMIT_NS, Ordering::SeqCst);
    assert_eq!(g.store(0, GFX_PAGE), Store::Held, "and stops again");
}

/// On the board a held store is answered busy without reaching the V12,
/// and the PI's pending-modify clear of the slot's graphics bit, which the
/// processor's interrupt handler makes, lets the next one go.
#[test]
fn the_board_holds_pi0s_store_below_the_bias_until_the_interrupt_is_taken() {
    use crate::dev::odyssey::{PI_GFX_PAGE_ADDRESS, PI_GFX_PAGE_ENABLE};
    let board = board_with_v12(4);
    let (pi, bus) = (&board.node.pi[0], &board.node.bus);
    let page = widget(12, ODSY_CMD_FIFO) & PI_GFX_PAGE_ADDRESS;
    assert_eq!(pi.write64(0x300, PI_GFX_PAGE_ENABLE | page), BUS_OK);
    assert_eq!(pi.write64(0x308, 1000), BUS_OK);
    assert_eq!(pi.write64(0x310, 1000), BUS_OK, "slot A's bias");
    let io_view = 1u64 << 57;
    let store = || bus.write64(io_view | widget(12, ODSY_CMD_FIFO), 0x0001_0046);
    assert_eq!(store(), BUS_OK, "999 is below the bias: the slot stops");
    assert_eq!(store(), crate::traits::BUS_BUSY);
    assert_eq!(pi.read64(0x308).data, 999, "held, not counted; no map entry returns credits");
    assert_eq!(pi.write64(0x90, 1), BUS_OK, "pending-modify: clear bit 1");
    assert_eq!(store(), BUS_OK);
}

/// A hub register as the processor reaches it through the two register
/// windows of uncached space 1: the local form at 0x0100_0000 and the
/// remote form at 0x0180_0000, both onto this node's registers.
fn hub_registers(board: &TezroBoard, window: u64, offset: u64) -> crate::traits::BusRead64 {
    board.node.bus.read64((1 << 57) | window | offset)
}
const HUB_WINDOWS: [u64; 2] = [0x0100_0000, 0x0180_0000];

/// The PROM writes the hub's widget control (0x400020) with destination
/// credits 4 in bits 15:13 for the integrated crossbar, and keeps the
/// widget number in bits 3:0.
#[test]
fn the_hub_keeps_the_proms_destination_credits_of_four() {
    for window in HUB_WINDOWS {
        let board = TezroBoard::new(1).unwrap();
        let control = hub_registers(&board, window, 0x40_0020).data;
        assert_eq!(control & 0xf, 10, "the hub's own widget number");
        let value = (control & !(7 << 13)) | (4 << 13);
        assert_eq!(board.node.bus.write64((1 << 57) | window | 0x40_0020, value), BUS_OK);
        let back = hub_registers(&board, window, 0x40_0020).data;
        assert_eq!((back >> 13) & 7, 4, "window {window:#x}");
        assert_eq!(back & 0xf, 10);
    }
}

/// A one-node boot reads the network interface as an uncabled port: status
/// 1, reset enable 0, error status 0 and the error-clear register 0 however
/// often it is read, and neither network error bit (39, 40) is up in the
/// processor interface's second pending register. Through both windows.
#[test]
fn a_one_node_boot_reads_the_network_interface_quiet() {
    for cpus in [1, 4] {
        let board = TezroBoard::new(cpus).unwrap();
        for window in HUB_WINDOWS {
            for (offset, expected) in [(0x68_0000, 1), (0x68_0010, 0), (0x68_0100, 0), (0x68_0120, 0)] {
                for count in 0..4 {
                    let r = hub_registers(&board, window, offset);
                    assert_eq!((r.status, r.data), (BUS_OK, expected), "{cpus} CPUs, {window:#x} + {offset:#x}, read {count}");
                }
            }
        }
        for (interface, pi) in board.node.pi.iter().enumerate() {
            let pending = pi.read64(crate::dev::bedrock_pi::reg::PENDING_STATUS_B).data;
            assert_eq!(pending & (1 << 39 | 1 << 40), 0, "{cpus} CPUs, PI{interface}");
        }
    }
}

/// PROM 6.210's crossbar check on the Tezro: it writes the read-only lower
/// error address, wants the register-access error with the hub's widget in
/// the status source, then enables that condition, writes the upper error
/// address, and wants the error again and vector 0x10 at the PI. The stores
/// complete, and the arbitration lock of link 15 (bit 1) holds its value.
#[test]
fn the_proms_crossbar_check_passes_on_the_tezro_board() {
    let board = TezroBoard::new(1).unwrap();
    let io = &board.node.io;
    let status_clear = widget(0, 0x54);
    let (err_upper, err_lower, control) = (widget(0, 0x14), widget(0, 0x1c), widget(0, 0x24));
    let pending_0 = || board.node.pi[0].read64(crate::dev::bedrock_pi::reg::PENDING_STATUS_A).data;
    // Vector 0x10 to the hub, at PI 0x90 of its register space (address
    // bit 47 selects the registers).
    assert_eq!(io.pio_write(widget(0, 0x34), 4, (0x10 << 24) | (10 << 16) | 0x8000), BUS_OK);
    assert_eq!(io.pio_write(widget(0, 0x3c), 4, 0x0180_0090), BUS_OK);

    assert_eq!(io.pio_write(err_lower, 4, 0xdead_beef), BUS_OK, "no bus error");
    let status = read_word(&board, status_clear);
    assert_ne!(status & (1 << 5), 0, "register-access error");
    assert_eq!((status >> 6) & 0xf, 10, "the hub's widget is the source");
    assert_eq!(read_word(&board, status_clear) & (1 << 5), 0, "the alias cleared it");
    assert_eq!(pending_0() & (1 << 16), 0, "nothing was enabled yet");

    assert_eq!(io.pio_write(control, 4, 0x20), BUS_OK);
    assert_eq!(io.pio_write(err_upper, 4, 0), BUS_OK, "no bus error");
    let status = read_word(&board, widget(0, 0x0c));
    assert_ne!(status & (1 << 5), 0, "register-access error again");
    assert_eq!((status >> 6) & 0xf, 10);
    assert_ne!(pending_0() & (1 << 16), 0, "vector 0x10 reached the PI");
    // Acknowledging the error lowers the vector again.
    assert_ne!(read_word(&board, status_clear) & (1 << 5), 0);
    assert_eq!(pending_0() & (1 << 16), 0);

    // The PROM's lock: bit 1 of link 15's arbitration-upper register.
    let lock = link_word(15, LINK_ARB_UPPER);
    assert_eq!(io.pio_write(lock, 4, 2), BUS_OK);
    assert_eq!(read_word(&board, lock) & 2, 2);
}

/// A jitv2 build refuses memory that reaches past 4 GiB, as the default
/// layout does, with a message naming a layout it takes, and builds the
/// processor for four 1024 MiB banks, which end at 4 GiB.
#[cfg(feature = "jitv2")]
#[test]
fn a_jitv2_build_takes_memory_only_below_4_gib() {
    let refused = check_jitv2_memory(&MemoryPopulation::default()).unwrap_err();
    assert!(refused.contains("banks = [1024, 1024, 1024, 1024]"), "{refused}");
    let four = MemoryPopulation::from_mib(&[1024; 4]).unwrap();
    check_jitv2_memory(&four).unwrap();
    // Building a processor takes more stack than a debug build's test thread has.
    std::thread::Builder::new().stack_size(64 << 20).spawn(move || {
        let board = TezroBoard::with_config(TezroBoardConfig {
            payload: &[],
            clock: Arc::new(|| 0),
            flash_path: None,
            station_mac: L1Config::default().mac,
            timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
            memory: &four,
            cpus: 1,
            part: TezroPart::Mhz700,
            v12: false,
            graphics_pool: (2, 32),
        })
        .unwrap();
        drop(build_cpu::<R16000ShadowCache>(&board));
    }).unwrap().join().unwrap();
}

/// A guest power-off exits the process without dropping the board, so the
/// flush is what carries an overlay's writes to the next run, on either
/// ISP12160 channel. A second board reads the disk while the first is still
/// alive, so nothing the first board's drop does can be what it sees; both
/// are then dropped, so no board is leaked.
#[test]
fn flushed_overlay_writes_survive_a_power_off_that_skips_drop() {
    use crate::scsi::{scsi_cmd, ScsiDataLength, ScsiRequest};
    for channel in 0..2 {
        let path = std::env::temp_dir().join(format!("iris-tezro-flush-{}-{channel}", std::process::id()));
        std::fs::write(&path, vec![0u8; 4 * 512]).unwrap();
        let attach = |board: &TezroBoard| {
            let bus = board.scsi.bus_for_channel(channel).unwrap();
            bus.attach_device(1, path.to_str().unwrap(), false, Vec::new(), true, None, true).unwrap();
        };
        let board = TezroBoard::new(1).unwrap();
        attach(&board);
        let write = ScsiRequest {
            cdb: vec![scsi_cmd::WRITE_10, 0, 0, 0, 0, 2, 0, 0, 1, 0],
            data_len: ScsiDataLength::Unlimited,
            data_in: Some(vec![0x5A; 512]),
        };
        board.scsi.bus_for_channel(channel).unwrap().request(1, &write).unwrap().unwrap();
        board.flush_disks().unwrap();
        let writer = board;

        let board = TezroBoard::new(1).unwrap();
        attach(&board);
        let read = ScsiRequest {
            cdb: vec![scsi_cmd::READ_10, 0, 0, 0, 0, 2, 0, 0, 1, 0],
            data_len: ScsiDataLength::Unlimited,
            data_in: None,
        };
        let got = board.scsi.bus_for_channel(channel).unwrap().request(1, &read).unwrap().unwrap();
        assert_eq!(got.data, vec![0x5A; 512], "channel {channel}");
        drop(board);
        drop(writer);
        let _ = std::fs::remove_file(format!("{}.overlay", path.display()));
        let _ = std::fs::remove_file(format!("{}.overlay.dirty", path.display()));
        let _ = std::fs::remove_file(&path);
    }
}
