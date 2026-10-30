//! Interrupt flags, routing and the packets that reach HEART.

use super::*;

/// A raised and a lowered board interrupt each reach HEART, setting and
/// clearing the vector's pending bit.
#[test]
fn a_raised_and_a_lowered_board_interrupt_each_reach_heart() {
    let (mut s, rec) = board("v12");

    s.mmio_write(ROUTE_DEST_UPPER, HEART_WIDGET << 25, 4);
    s.mmio_write(ROUTE_DEST_LOWER, HEART_PACKET, 4);
    s.mmio_write(ROUTE_VECTORS, TEST_VECTOR as u64, 4);
    assert!(!rec.level(TEST_VECTOR));

    // The FIFO is empty, so its low-water flag is already pending.
    s.mmio_write(INTR_ENABLE_SET, INTR_FIFO_LOW_WATER, 4);
    assert!(rec.level(TEST_VECTOR));

    s.mmio_write(INTR_CLEAR, INTR_FIFO_LOW_WATER, 4);
    assert!(!rec.level(TEST_VECTOR));
}


/// The mask the kernel driver's interrupt set-up writes, from a boot trace.
/// It leaves the low-water source off.
const INIT_MASK: u32 = 0x79e4_259e;
const LOW_WATER: u32 = ODSY_INTR_FIFO_LOW_WATER;
const DMA_DONE: u32 = 0x0000_0800;

/// The flag, enable and acknowledge trio, and the rule that an enable on
/// its own raises nothing unless a source is already pending: the empty
/// FIFO leaves low water pending out of reset, so enabling asserts.
///
/// QEMU: interrupt-registers.
#[test]
fn enabling_a_pending_source_asserts_and_acknowledging_clears_it() {
    let (mut s, rec) = board("v12");
    route_to_heart(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS), LOW_WATER);
    assert_eq!(reg_read(&mut s, ODSY_INTR_ENABLE_SET), 0);
    assert!(rec.events(FLAG_VECTOR).is_empty());

    reg_write(&mut s, ODSY_INTR_ENABLE_SET, 0xffff_ffff);
    assert_eq!(reg_read(&mut s, ODSY_INTR_ENABLE_SET), 0xffff_ffff);
    assert!(rec.level(FLAG_VECTOR));

    // Write-one-to-clear over the flags, with nothing readable behind it.
    reg_write(&mut s, ODSY_INTR_CLEAR, 0xffff_ffff);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS), 0);
    assert_eq!(reg_read(&mut s, ODSY_INTR_CLEAR), 0);
    assert!(!rec.level(FLAG_VECTOR));
    assert_eq!(reg_read(&mut s, ODSY_INTR_ENABLE_SET), 0xffff_ffff, "clearing flags leaves the enables alone");

    // The software-sync pair: a token written, then read back.
    reg_write(&mut s, ODSY_SW_TOKEN, 0x5a5a_5a5a);
    assert_eq!(reg_read(&mut s, ODSY_SW_TOKEN), 0x5a5a_5a5a);
}

/// The enable mask's two halves, the way the driver drives them: the set
/// half adds the bits written and reads back the whole mask, the clear
/// half removes them. Clearing an enable leaves its flag pending.
///
/// QEMU: interrupt-enable-set-and-clear.
#[test]
fn the_enable_mask_is_set_and_cleared_by_its_two_halves() {
    let (mut s, rec) = board("v12");
    route_to_heart(&mut s);

    // The initial mask leaves low water off: no level.
    reg_write(&mut s, ODSY_INTR_ENABLE_SET, INIT_MASK);
    assert_eq!(reg_read(&mut s, ODSY_INTR_ENABLE_SET), INIT_MASK);
    assert!(!rec.level(FLAG_VECTOR));

    // The high-water path adds one bit; the rest of the mask stays.
    reg_write(&mut s, ODSY_INTR_ENABLE_SET, LOW_WATER);
    assert_eq!(reg_read(&mut s, ODSY_INTR_ENABLE_SET), INIT_MASK | LOW_WATER);
    assert!(rec.level(FLAG_VECTOR));

    // The low-water handler removes it: the level drops, the flag stays.
    reg_write(&mut s, ODSY_INTR_ENABLE_CLEAR, LOW_WATER);
    assert_eq!(reg_read(&mut s, ODSY_INTR_ENABLE_SET), INIT_MASK);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS) & LOW_WATER, LOW_WATER);
    assert!(!rec.level(FLAG_VECTOR));

    // Disabling with all ones empties the mask.
    reg_write(&mut s, ODSY_INTR_ENABLE_SET, LOW_WATER);
    reg_write(&mut s, ODSY_INTR_ENABLE_CLEAR, 0xffff_ffff);
    assert_eq!(reg_read(&mut s, ODSY_INTR_ENABLE_SET), 0);
    assert!(!rec.level(FLAG_VECTOR));
}

/// The flag handler's acknowledge writes the mask it dispatched on and
/// clears those flags and nothing else. The DMA submission's write of
/// 0x800 clears a flag the model never raises, so it changes nothing.
///
/// QEMU: interrupt-clear-acknowledges-flags.
#[test]
fn an_acknowledge_clears_only_the_flags_it_names() {
    let (mut s, rec) = board("v12");
    route_to_heart(&mut s);
    reg_write(&mut s, ODSY_INTR_ENABLE_SET, LOW_WATER);
    assert!(rec.level(FLAG_VECTOR));

    reg_write(&mut s, ODSY_INTR_CLEAR, DMA_DONE);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS), LOW_WATER);
    assert!(rec.level(FLAG_VECTOR));

    reg_write(&mut s, ODSY_INTR_CLEAR, LOW_WATER);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS), 0);
    assert_eq!(reg_read(&mut s, ODSY_INTR_ENABLE_SET), LOW_WATER);
    assert!(!rec.level(FLAG_VECTOR));
}

/// A driver writes the flags register itself: its flow control stores the
/// low-water bit before enabling that source, and a DMA diagnostic clears
/// one flag by writing back the flags it read without it. A write replaces
/// the flags, and the interrupt follows.
///
/// QEMU: interrupt-flags-take-written-value.
#[test]
fn a_write_to_the_flags_replaces_them() {
    let (mut s, rec) = board("v12");
    route_to_heart(&mut s);
    reg_write(&mut s, ODSY_INTR_ENABLE_SET, LOW_WATER);
    reg_write(&mut s, ODSY_INTR_CLEAR, LOW_WATER);
    assert!(!rec.level(FLAG_VECTOR));

    reg_write(&mut s, ODSY_INTR_FLAGS, LOW_WATER);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS), LOW_WATER);
    assert!(rec.level(FLAG_VECTOR));

    // Without the low-water bit it is dropped, as a set-only write would not.
    reg_write(&mut s, ODSY_INTR_FLAGS, DMA_DONE);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS), DMA_DONE);
    assert!(!rec.level(FLAG_VECTOR));

    reg_write(&mut s, ODSY_INTR_FLAGS, 0);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS), 0);
}

/// A submission that does not wait never lifts the FIFO to its low-water
/// mark, so it raises nothing where the driver has disabled the source.
/// The driver's high-water path then latches the flag itself through the
/// flags register: no interrupt until it enables the source, and that
/// asserts at once.
///
/// QEMU: lowater-not-raised-while-disabled.
#[test]
fn low_water_is_not_raised_while_the_source_is_disabled() {
    let (mut s, rec) = board("v12");
    route_to_heart(&mut s);
    reg_write(&mut s, ODSY_INTR_ENABLE_SET, INIT_MASK | LOW_WATER);
    reg_write(&mut s, ODSY_INTR_ENABLE_CLEAR, LOW_WATER);
    reg_write(&mut s, ODSY_INTR_CLEAR, LOW_WATER);
    assert!(!rec.level(FLAG_VECTOR));

    cmd(&mut s, 0x0001_0046);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS) & LOW_WATER, 0);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0) & 0x20000, 0x20000);

    reg_write(&mut s, ODSY_INTR_FLAGS, LOW_WATER);
    assert!(!rec.level(FLAG_VECTOR), "latched, but the source is off");
    reg_write(&mut s, ODSY_INTR_ENABLE_SET, LOW_WATER);
    assert!(rec.level(FLAG_VECTOR));
}

/// The status register the driver's periodic credit check reads: with the
/// command FIFO empty it reports enabled, empty and below the low-water
/// mark. A clear low-water or enabled bit makes the driver log a graphics
/// error and take its board-reset recovery path.
///
/// QEMU: board-status-reports-fifo-ready.
#[test]
fn the_board_status_reports_an_empty_fifo_as_ready() {
    let (mut s, _) = board("v12");
    assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1003_0000);
}

/// The low-water interrupt the driver waits on to replenish its credits:
/// pending out of reset, so enabling asserts; acknowledging clears it; and
/// a later submission that does not wait leaves it clear. Without the
/// reset-pending latch the kernel's post-reset watchdog never sees it and
/// gives up starting the X server.
///
/// QEMU: command-fifo-lowater-interrupt.
#[test]
fn the_reset_latched_low_water_interrupt_asserts_once_and_stays_clear() {
    let (mut s, rec) = board("v12");
    route_to_heart(&mut s);

    reg_write(&mut s, ODSY_INTR_ENABLE_SET, LOW_WATER);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS) & LOW_WATER, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true]);

    reg_write(&mut s, ODSY_INTR_CLEAR, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true, false], "clearing the flag sends the clearing packet");

    cmd(&mut s, 0x0001_0046);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS) & LOW_WATER, 0);
    assert_eq!(rec.events(FLAG_VECTOR), [true, false]);
}

/// A reset lowers the flag level with the flags and enables, quietly: the
/// destination is reset too, so there is nowhere to send a clearing packet
/// and none is sent. The next edge is a rising one and sends its set
/// packet; a different flag vector after the reset tells it from the one
/// before.
///
/// QEMU: reset-clears-flag-level.
#[test]
fn a_reset_lowers_the_flag_level_without_a_packet() {
    let (mut s, rec) = board("v12");
    route_to_heart(&mut s);
    reg_write(&mut s, ODSY_INTR_ENABLE_SET, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true]);

    s.reset();
    assert_eq!(rec.events(FLAG_VECTOR), [true], "the reset sends no clearing packet");

    route_to_heart(&mut s);
    reg_write(&mut s, ODSY_ROUTE_VECTORS, FLAG_VECTOR + 1);
    reg_write(&mut s, ODSY_INTR_ENABLE_SET, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR + 1), [true]);
    assert_eq!(rec.events(FLAG_VECTOR), [true]);
}

/// The packet an edge composes decodes the routed vector and destination,
/// and nothing is sent before an edge or with no destination programmed.
/// The crosstalk error destination reads back and carries nothing.
///
/// QEMU: interrupt-packet-composition.
#[test]
fn a_packet_needs_a_destination_and_an_edge() {
    let (mut s, rec) = board("v12");

    // No destination: the rising edge sends nothing but still latches, so
    // the next edge is the falling one.
    reg_write(&mut s, ODSY_INTR_ENABLE_SET, LOW_WATER);
    assert!(rec.silent());

    reg_write(&mut s, ODSY_INT_DEST_UPPER as u64, 0x1234_5678);
    reg_write(&mut s, ODSY_INT_DEST_LOWER as u64, 0x9abc_def0);
    assert_eq!(reg_read(&mut s, ODSY_INT_DEST_UPPER as u64), 0x1234_5678);
    assert_eq!(reg_read(&mut s, ODSY_INT_DEST_LOWER as u64), 0x9abc_def0);
    route_to_heart(&mut s);
    assert!(rec.silent(), "programming the destination alone is not an edge");

    // Acknowledging the still-pending flag is the falling edge.
    reg_write(&mut s, ODSY_INTR_CLEAR, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [false]);

    // The flag set again readdresses the same destination, setting it.
    reg_write(&mut s, ODSY_INTR_FLAGS, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [false, true]);
}

/// The kernel gives the routed table's low byte to its flag handler and
/// the crosstalk error destination to its error handler, so low water goes
/// through the routed destination with the flag vector.
///
/// QEMU: flag-interrupt-uses-routed-vector.
#[test]
fn the_flag_interrupt_uses_the_routed_flag_vector() {
    let (mut s, rec) = board("v12");
    reg_write(&mut s, ODSY_INT_DEST_UPPER as u64, 0x7000_0000);
    reg_write(&mut s, ODSY_INT_DEST_LOWER as u64, 0x40);
    route_to_heart(&mut s);
    reg_write(&mut s, ODSY_ROUTE_VECTORS, (SOFTSYNC_VECTOR << 24) | 4);

    reg_write(&mut s, ODSY_INTR_ENABLE_SET, LOW_WATER);
    assert_eq!(rec.events(4), [true]);
    reg_write(&mut s, ODSY_INTR_CLEAR, LOW_WATER);
    assert_eq!(rec.events(4), [true, false]);
}
