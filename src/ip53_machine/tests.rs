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
    assert_eq!(io.pio_read(config_id(2), 4).data, 0xffff_ffff);
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
