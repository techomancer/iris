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
        chip: &CHIPS[0],
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
/// in PI1/A, CPUs 2 and 3 in the B slots), and every fitted processor runs
/// from reset with its enable set, as QEMU's node grants them.
#[test]
fn fitted_processors_read_present_and_start_enabled() {
    use crate::dev::bedrock_pi::reg::{CPU_A_ENABLE, CPU_B_ENABLE, INTERFACE_SLOT_A, INTERFACE_SLOT_B};
    for (cpus, present) in [(1, [[1, 0], [0, 0]]), (2, [[1, 0], [1, 0]]), (4, [[1, 1], [1, 1]])] {
        let board = TezroBoard::new(cpus).unwrap();
        assert_eq!(board.processors.count(), cpus);
        for (interface, pi) in board.node.pi.iter().enumerate() {
            let slots = [pi.read64(INTERFACE_SLOT_A).data, pi.read64(INTERFACE_SLOT_B).data];
            assert_eq!(slots, present[interface], "{cpus} CPUs, PI{interface}");
            let enabled = [pi.read64(CPU_A_ENABLE).data & 1, pi.read64(CPU_B_ENABLE).data & 1];
            assert_eq!(enabled, present[interface], "{cpus} CPUs, PI{interface}");
        }
    }
}

/// A store a processor makes into its graphics page is counted against its
/// own slot, and its interrupt raised on its own interface: CPU 1 sits in
/// PI1 slot A, CPU 2 in PI0 slot B.
#[test]
fn each_processors_graphics_stores_count_against_its_own_slot() {
    let board = Arc::new(board_with_v12(4));
    let stores = SlotStores { pi: board.node.pi.clone(), gfx: board.gfx.clone(), running: board.processors };
    let stores = Arc::new(stores);
    for (cpu, interface, slot) in [(1u8, 1usize, 0usize), (2, 0, 1)] {
        let (stores, board) = (stores.clone(), board.clone());
        std::thread::spawn(move || {
            crate::cpu::smp::set_current_cpu(cpu);
            assert_eq!(stores.slot(), (interface, slot), "CPU {cpu}");
            // No page enabled: uncounted, and no slot's credit moves.
            assert_eq!(stores.count(0x1000), Store::Uncounted);
            let _ = &board;
        })
        .join()
        .unwrap();
    }
    assert_eq!(stores.slot(), (0, 0), "a thread that runs no processor counts as CPU 0");
}

/// Only the counts a board fits are accepted.
#[test]
fn the_board_refuses_a_population_it_cannot_fit() {
    assert!(TezroBoard::new(3).is_err());
    assert!(TezroBoard::new(0).is_err());
    assert!(TezroBoard::new(5).is_err());
}

/// A board of one processor and one GiB of memory with `chip` fitted.
fn board_with(chip: CpuModel) -> TezroBoard {
    TezroBoard::with_config(TezroBoardConfig {
        payload: &[],
        clock: Arc::new(|| 0),
        flash_path: None,
        station_mac: L1Config::default().mac,
        timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
        memory: &MemoryPopulation::one_gib(),
        cpus: 1,
        chip: self::chip(chip),
        v12: false,
        graphics_pool: (2, 32),
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
    fn check<C: CacheCpuModel + From<Arc<dyn BusDevice>>>(model: CpuModel, prid: u32, config0: u32, mhz: u32) {
        let board = board_with(model);
        let mut cpu = build_cpu::<C>(&board);
        let name = model.name();
        assert_eq!((cpu.core.cp0_prid, cpu.core.reset_prid), (prid, prid), "{name}");
        assert_eq!(model.prid(), Some(prid), "{name}: the chip's PRId");
        assert_eq!((cpu.core.fpu_fir, cpu.core.reset_fir), (0x900, 0x900), "{name}");
        assert_eq!(cpu.core.cp0_config & CONFIG0_MODE_MASK, config0, "{name}");
        assert_eq!(cpu.core.cp0_config >> 26, 0b011_011, "{name}: 32 KB primary caches");
        let config = cpu.core.cp0_config;
        cpu.core.fpu_fir = 0;
        cpu.core.cp0_prid = 0;
        cpu.core.reset(false);
        assert_eq!((cpu.core.cp0_prid, cpu.core.fpu_fir, cpu.core.cp0_config), (prid, 0x900, config), "{name}: after reset");
        assert_eq!(model.mhz(), mhz);
        assert_eq!(model.count_hz(), u64::from(mhz) * 500_000, "{name}: Count at half the clock");
        assert_eq!(chip(model).firmware.cpu_clock_hz, mhz * 1_000_000, "{name}: the PROM's clock record");
    }
    use crate::cpu::mips_cache_shadow::{R16000Scache16ShadowCache, R16000Scache8ShadowCache, R16000ShadowCache};
    on_big_stack(|| {
        check::<R16000ShadowCache>(CpuModel::R16000_700, 0x0f41, 0x009b_ad85, 700);
        check::<R16000ShadowCache>(CpuModel::R16000_800, 0x0f42, 0x009b_af85, 800);
        check::<R16000Scache8ShadowCache>(CpuModel::R16000_900, 0x0f50, 0x009c_b185, 900);
        check::<R16000Scache16ShadowCache>(CpuModel::R16000_1000, 0x0f50, 0x00ed_b385, 1000);
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
        let word = c.firmware.reset_mode;
        let name = c.chip.name();
        let field = |lo: u32, bits: u32| (word >> lo) & ((1 << bits) - 1);
        let mhz = c.chip.mhz();
        let bytes = crate::with_cpu!(c.chip, true, C => <C as MipsCache>::L2_SIZE as u32);
        assert_eq!((field(0, 3), field(7, 2), field(13, 1), field(15, 1)), (5, 3, 1, 1), "{name}: K0 PM SB BE");
        assert_eq!(field(9, 4), 2 * mhz / 200 - 1, "{name}: EC");
        assert_eq!(2 * mhz / (field(19, 3) + 1), c.scache_mhz, "{name}: SC gives the cache clock");
        assert_eq!(512u32 << 10 << field(16, 3), bytes, "{name}: SS gives the cache size");
        assert_eq!((word >> 23) & 1, 1, "{name}: bit 23 is set in every row of 500 MHz and up");
        assert_eq!(word >> 24, 0, "{name}: bit 24 clear");
    }
    let word = chip(CpuModel::R16000_1000).firmware.reset_mode;
    assert_eq!(((word >> 16) & 7, (word >> 19) & 7, (word >> 9) & 15, (word >> 22) & 1), (5, 5, 9, 1), "1 GHz SS SC EC and bit 22");
}

/// A chip's revision, as the inventory displays it, derives from its PRId:
/// the low byte holds the raw major and minor, and on implementation 0x0f a
/// raw major above 2 is an R16000, displayed as the major minus 2 (the rule
/// Linux and OpenBSD apply). This ties the Tezro's "2.1", "2.2", "3.0" and
/// "3.0" to 0x0f41, 0x0f42 and 0x0f50.
#[test]
fn every_chips_revision_is_the_one_its_prid_displays() {
    let shown = ["2.1", "2.2", "3.0", "3.0"];
    assert_eq!(CPUS.len(), shown.len());
    for (chip, shown) in CPUS.iter().zip(shown) {
        let prid = chip.prid().unwrap();
        let (implementation, raw) = ((prid >> 8) & 0xff, prid & 0xff);
        let (major, minor) = (raw >> 4, raw & 0xf);
        let major = if implementation == 0x0f && major > 2 { major - 2 } else { major };
        assert_eq!(shown, format!("{major}.{minor}"), "{} (PRId {prid:#06x})", chip.label());
    }
}

/// A chip and its flash record are two statements of one chip: the clock,
/// and the Config word the PROM compares against the processor's, must agree.
#[test]
fn each_tezro_chip_agrees_with_its_flash_record() {
    for c in &CHIPS {
        let record = c.firmware;
        let name = c.chip.name();
        assert_eq!(u64::from(record.cpu_clock_hz), u64::from(c.chip.mhz()) * 1_000_000, "{name}");
        assert_eq!(record.cpu_mode_word & CONFIG0_MODE_MASK, record.reset_mode, "{name}");
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
        (CpuModel::R16000_700, 4u32),
        (CpuModel::R16000_800, 4),
        (CpuModel::R16000_900, 8),
        (CpuModel::R16000_1000, 16),
    ] {
        let (name, bytes) = crate::with_cpu!(model, true, C => (<C as CacheCpuModel>::NAME, <C as MipsCache>::L2_SIZE as u32));
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
            let board = board_with(cfg.machine.cpu);
            let (core_prid, core_config, l2) = crate::with_cpu!(cfg.machine.cpu, true, C => {
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
        chip: &CHIPS[0],
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
        chip: &CHIPS[0],
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

/// The PROM runs from 0x1fc00000 uncached while memory covers that window,
/// so the processor presents an uncached instruction fetch of it at the
/// PROM's own address, `RESET_WINDOW_FETCH`, which the node maps to the same
/// PROM. A cached fetch, a data access and a fetch past the window are left
/// alone.
#[test]
fn an_uncached_fetch_of_the_reset_window_is_presented_at_the_proms_own_address() {
    use crate::cpu::mips_tlb::AccessType;
    use crate::ip35::{RESET_WINDOW, RESET_WINDOW_FETCH, RESET_WINDOW_SIZE};
    on_big_stack(|| {
        let board = board_with(CpuModel::R16000_700);
        let mut cpu = build_cpu::<crate::cpu::mips_cache_shadow::R16000ShadowCache>(&board);
        let mut phys = |va: u64, at: AccessType| {
            let result = (cpu.translate_fn)(&mut cpu, va, at);
            assert!(!result.is_exception(), "{va:#x}");
            result.phys
        };
        let kseg1 = 0xffff_ffff_a000_0000u64;
        assert_eq!(phys(kseg1 + RESET_WINDOW, AccessType::Fetch), RESET_WINDOW_FETCH);
        assert_eq!(phys(kseg1 + RESET_WINDOW + 0x1234, AccessType::Fetch), RESET_WINDOW_FETCH + 0x1234);
        let last = RESET_WINDOW_SIZE - 4;
        assert_eq!(phys(kseg1 + RESET_WINDOW + last, AccessType::Fetch), RESET_WINDOW_FETCH + last);
        assert_eq!(phys(kseg1 + RESET_WINDOW, AccessType::Read), RESET_WINDOW, "data reads keep the plain address");
        assert_eq!(phys(0xffff_ffff_8000_0000 + RESET_WINDOW, AccessType::Fetch), RESET_WINDOW, "a cached fetch is not aliased");
        // The alias reaches the same PROM the plain window does.
        let bus = board.node.bus.clone();
        assert_eq!(bus.read32(RESET_WINDOW_FETCH).status, bus.read32(RESET_WINDOW).status);
    });
}

/// A jitv2 build takes the default six banks, which reach past 4 GiB: the
/// PROM's uncached code is keyed by its own address above every memory slot,
/// not by a frame number that has to stay below the memory.
#[cfg(feature = "jitv2")]
#[test]
fn a_jitv2_build_takes_the_default_memory_layout() {
    on_big_stack(|| {
        let memory = MemoryPopulation::default();
        assert!(!memory.fits_direct_window());
        let board = TezroBoard::with_config(TezroBoardConfig {
            payload: &[],
            clock: Arc::new(|| 0),
            flash_path: None,
            station_mac: L1Config::default().mac,
            timekeeper: Arc::new(Ds1744::new(YearBase::From(1968)).unwrap()),
            memory: &memory,
            cpus: 1,
            chip: &CHIPS[0],
            v12: false,
            graphics_pool: (2, 32),
        })
        .unwrap();
        drop(build_cpu::<crate::cpu::mips_cache_shadow::R16000ShadowCache>(&board));
    });
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

/// The cached view the processors' caches sit on swaps RAM above 4 GiB, so
/// SC stays atomic where the direct window does not reach.
#[test]
fn the_cached_view_swaps_ram_above_4_gib() {
    let board = TezroBoard::new(4).unwrap();
    // Banks 0 to 5: two 1 GiB banks and four of 512 MiB, as MD's tests set them.
    assert_eq!(board.node.md.md_bus().write64(8, 0x53 | (0x43 << 8) | (0x43 << 16)), BUS_OK);
    let cached = board.cached_bus();
    let at = (5u64 << 30) + 0x80;
    assert_eq!(cached.write64(at, 1), BUS_OK);
    assert_eq!(cached.cas64(at, 1, 2), Some(true));
    assert_eq!(cached.cas64(at, 1, 3), Some(false));
    assert_eq!(cached.read64(at).data, 2);
}

/// Only the two-processor board's fresh flash marks its empty B and D
/// positions disabled.
#[test]
fn only_the_two_processor_flash_starts_with_b_and_d_disabled() {
    for (cpus, disabled) in [(1, false), (2, true), (4, false)] {
        let bytes = TezroBoard::new(cpus).unwrap().node.flash.contents();
        assert_eq!(&bytes[0x9e0101..0x9e0109] == b"DisableB", disabled, "{cpus} CPUs");
        assert_eq!(&bytes[0x9e0141..0x9e0149] == b"DisableD", disabled, "{cpus} CPUs");
    }
}
