//! The board catalogue and the Octane attach: what each marketed VPro
//! answers as, and how the widget windows reach it.

use super::*;

#[test]
fn the_board_answers_with_its_identity_and_widget_number() {
    let (mut s, _) = board("v12");
    assert_eq!(s.mmio_read(IDENTITY, 4) as u32, IDENTITY_DEFAULT);
    assert_eq!(s.mmio_read(CONTROL, 4) as u32 & ODSY_CONTROL_WIDGET_MASK, VPRO_WIDGET);
}

#[test]
fn each_marketed_profile_keeps_its_generation_and_capacity_class() {
    // (product, revision, memory size code)
    let cases = [("v6", 2u32, 0u32), ("v8", 2, 2), ("v10", 0xa, 0), ("v12", 0xa, 2)];

    for (name, revision, size_code) in cases {
        let (mut s, rec) = board(name);
        let identity = s.mmio_read(IDENTITY, 4) as u32;
        let memory = s.mmio_read(MEMORY_CONFIG, 4) as u32;

        assert_eq!(identity >> 28, revision, "{name}");
        assert_eq!(identity & 0x0fff_ffff, IDENTITY_DEFAULT & 0x0fff_ffff, "{name}");
        assert_eq!((memory >> 30) & 3, size_code, "{name}");
        assert_eq!(s.mmio_read(CONTROL, 4) as u32 & ODSY_CONTROL_WIDGET_MASK, VPRO_WIDGET);

        s.mmio_write(ROUTE_DEST_UPPER, HEART_WIDGET << 25, 4);
        s.mmio_write(ROUTE_DEST_LOWER, HEART_PACKET, 4);
        s.mmio_write(ROUTE_VECTORS, TEST_VECTOR as u64, 4);
        s.mmio_write(INTR_ENABLE_SET, INTR_FIFO_LOW_WATER, 4);
        assert!(rec.level(TEST_VECTOR), "{name}: vector raised");
        s.mmio_write(INTR_CLEAR, INTR_FIFO_LOW_WATER, 4);
        assert!(!rec.level(TEST_VECTOR), "{name}: vector lowered");
    }
}


mod widget {
    //! The Octane VPro integration checks:
    //! the board answers through both widget-11 windows, the Crossbow
    //! carries its link only when it is fitted, and a routed interrupt
    //! sets and clears its vector in HEART's pending register.
    use super::*;
    use crate::dev::heart::{host_clock, Heart, HeartConfig, PIU_BASE};
    use crate::dev::heart_mem::HeartRam;
    use crate::dev::xbow::Xbow;

    const WIDGET: u8 = 11;
    const VECTOR: u32 = 18;

    fn board() -> (Arc<Heart>, Arc<Odyssey>) {
        let heart = Arc::new(Heart::new(HeartConfig::octane([128 << 20, 0, 0, 0]), host_clock()));
        let ram = Arc::new(HeartRam::try_new([128, 0, 0, 0]).expect("ram"));
        let v = Odyssey::new(WIDGET, "v12", heart.clone(), ram.dma_view(), Default::default(), Default::default()).expect("board");
        (heart, v)
    }

    fn pending(heart: &Heart) -> u64 {
        heart.read64(PIU_BASE + 0x10030).data
    }

    #[test]
    fn the_board_answers_through_both_widget_windows() {
        let (_heart, v) = board();
        for base in [widget_small_base(WIDGET), widget_large_base(WIDGET)] {
            assert_eq!(v.read32(base + 0x4).data, 0xac01_3047);
            assert_eq!(v.read32(base + 0x24).data & 0xf, WIDGET as u32);
        }
    }

    #[test]
    fn the_crossbow_carries_the_graphics_link_only_when_fitted() {
        let fitted = Xbow::new(4, (1 << 8) | (1 << 15) | (1 << WIDGET));
        let empty = Xbow::new(4, (1 << 8) | (1 << 15));
        assert!(fitted.carries(WIDGET));
        assert!(!empty.carries(WIDGET));
    }

    /// Starting the device starts the board's thread and the display
    /// thread; stopping it joins both and finishes the work still
    /// queued, so nothing is left behind a stopped machine.
    ///
    /// QEMU: command-work-stop-and-reset.
    #[test]
    fn stopping_the_device_finishes_the_work_the_board_thread_had_queued() {
        use crate::traits::Device;
        let (_heart, v) = board();
        assert!(!v.is_running());

        v.start();
        v.start();
        assert!(v.is_running(), "a second start changes nothing");
        v.stop();

        assert!(!v.is_running());
        assert!(v.state.fifo.thread.lock().is_none());
        assert!(!v.state.fifo.pending());
    }

    #[test]
    fn a_routed_interrupt_sets_and_clears_its_heart_vector() {
        let (heart, v) = board();
        let b = widget_small_base(WIDGET);
        v.write32(b + 0x1044, 8 << 25);
        v.write32(b + 0x104c, 0x80);
        v.write32(b + 0x1054, VECTOR);
        assert_eq!(pending(&heart) & 1 << VECTOR, 0);
        // The FIFO is empty, so its low-water flag is already pending.
        v.write32(b + 0x1034, 0x200);
        assert_ne!(pending(&heart) & 1 << VECTOR, 0);
        v.write32(b + 0x101c, 0x200);
        assert_eq!(pending(&heart) & 1 << VECTOR, 0);
    }
}


/// Each marketed board presents its own identity and memory word. The
/// kernel turns widget revision 1-3 into the first part generation and
/// 9-15 into the next; the 128 MB boards report four banks and the 32 MB
/// boards two.
///
/// QEMU: profile-selects-identity-and-memory.
#[test]
fn each_profile_presents_its_own_identity_and_memory_word() {
    for (product, identity, memory) in [
        ("v12", IDENTITY_DEFAULT, 0xa100_0000),
        ("v10", IDENTITY_DEFAULT, 0x0100_0000),
        ("v8", 0x2c01_3047, 0xa100_0000),
        ("v6", 0x2c01_3047, 0x0100_0000),
    ] {
        let (mut s, _) = board(product);
        assert_eq!(reg_read(&mut s, IDENTITY), identity, "{product}");
        assert_eq!(reg_read(&mut s, MEMORY_CONFIG), memory, "{product}");
    }
}

/// Bits 29 (four banks) and 28 (CAS latency 2) of the memory word,
/// decomposed from the V12's measured 0xa1000000: four banks, latency 3,
/// and the constant copy field 1 in bits 27:24.
///
/// QEMU: v12-profile-memory-word-bits.
#[test]
fn the_v12_memory_word_decodes_to_four_banks_and_latency_three() {
    let (mut s, _) = board("v12");
    let word = reg_read(&mut s, MEMORY_CONFIG);

    assert_eq!(word, 0xa100_0000);
    assert_eq!(word & 1 << 29, 1 << 29);
    assert_eq!(word & 1 << 28, 0);
    assert_eq!((word >> 24) & 0xf, 1);
}

/// gfxinfo's "PB&J version" is bits 2:0 of 0x23380c. Captured V6, V10 and
/// V12 boards all print 1, so every profile answers 1, and a write changes
/// nothing.
///
/// QEMU: backend-version-is-pinned.
#[test]
fn every_profile_answers_back_end_version_one_and_ignores_a_write() {
    for product in ["v12", "v10", "v8", "v6"] {
        let (mut s, _) = board(product);
        assert_eq!(s.mmio_read(ODSY_BACKEND_VERSION, 4), 1, "{product}");
        s.mmio_write(ODSY_BACKEND_VERSION, 0, 4);
        assert_eq!(s.mmio_read(ODSY_BACKEND_VERSION, 4), 1, "{product}");
    }
}

/// A profile's board memory is a reported capacity, not storage: objects
/// come from one fixed pool. A destination base at or beyond what a 32 MiB
/// profile reports is a guest bug against the board it was told it has, so
/// the selection is dropped and noted, and the board keeps answering. The
/// first value out of range is 0x200000 in 16-byte units.
///
/// QEMU: board-memory-bound-is-a-guest-error.
#[test]
fn a_destination_beyond_the_reported_memory_is_dropped() {
    let (mut s, _) = board("v10");
    cmd(&mut s, select(ODSY_REG_DEST_SELECT));
    for operand in [0x20_0000, 16, 0, 0] {
        cmd(&mut s, operand);
    }
    assert!(s.board_oob_reported);
    assert_eq!(s.obj_count, 0, "no object was made for the dropped base");
    assert_eq!(reg_read(&mut s, IDENTITY), IDENTITY_DEFAULT);

    // The same base is in range on a 128 MiB board.
    let (mut s, _) = board("v12");
    cmd(&mut s, select(ODSY_REG_DEST_SELECT));
    for operand in [0x20_0000, 16, 0, 0] {
        cmd(&mut s, operand);
    }
    assert!(!s.board_oob_reported);
}

/// Each name the configuration accepts selects a board, VPro boards are
/// offered on the Octane only, and the PROM revision check the upgrade kit
/// calls for is a property of the board.
///
/// QEMU: octane-vpro-test absent/widget-identity (the machine offers the
/// board only where its Crossbow and supply allow it).
#[test]
fn vpro_boards_are_offered_on_the_octane_only() {
    use crate::config::{GraphicsBoard, MachineProfile};

    for (name, board) in [
        ("v6", GraphicsBoard::V6),
        ("vpro:v8", GraphicsBoard::V8),
        ("vpro_v10", GraphicsBoard::V10),
        ("V12", GraphicsBoard::V12),
    ] {
        assert_eq!(name.parse::<GraphicsBoard>().unwrap(), board, "{name}");
        assert!(board.is_vpro());
        assert!(board.supports(MachineProfile::OctaneIp30));
        assert!(!board.supports(MachineProfile::Indigo2Ip22));
    }
    assert!(!GraphicsBoard::SolidImpact.is_vpro());
    assert!("v14".parse::<GraphicsBoard>().is_err());
}
