use super::*;
use crate::traits::{BUS_ERR, BUS_OK};

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

/// Only the counts a board fits are accepted, and the memory must meet
/// the processors' minimum.
#[test]
fn the_board_refuses_a_population_it_cannot_fit() {
    assert!(TezroBoard::new(3).is_err());
    assert!(TezroBoard::new(0).is_err());
}

/// A board of one processor and one GiB of memory (below 4 GiB, which a
/// jitv2 build's uncached code views need) with `chip` fitted.
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
    use std::sync::atomic::AtomicU64;
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
