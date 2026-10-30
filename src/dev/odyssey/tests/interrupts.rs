//! Interrupt flags, routing and the packets that reach HEART.

use crate::dev::odyssey::consts::{
    intr, route, ODSY_INT_DEST_LOWER, ODSY_INT_DEST_UPPER, ODSY_STATUS0, ODSY_SW_TOKEN,
};
use crate::dev::odyssey::tests::{
    board, cmd, reg_read, reg_write, route_to_heart, DMA_VECTOR, FLAG_VECTOR, RETRACE_VECTOR,
    SOFTSYNC_VECTOR,
};
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
const LOW_WATER: u32 = intr::FIFO_LOW_WATER;
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
    assert_eq!(reg_read(&mut s, intr::FLAGS), LOW_WATER);
    assert_eq!(reg_read(&mut s, intr::ENABLE_SET), 0);
    assert!(rec.events(FLAG_VECTOR).is_empty());

    reg_write(&mut s, intr::ENABLE_SET, 0xffff_ffff);
    assert_eq!(reg_read(&mut s, intr::ENABLE_SET), 0xffff_ffff);
    assert!(rec.level(FLAG_VECTOR));

    // Write-one-to-clear over the flags, with nothing readable behind it.
    reg_write(&mut s, intr::CLEAR, 0xffff_ffff);
    assert_eq!(reg_read(&mut s, intr::FLAGS), 0);
    assert_eq!(reg_read(&mut s, intr::CLEAR), 0);
    assert!(!rec.level(FLAG_VECTOR));
    assert_eq!(reg_read(&mut s, intr::ENABLE_SET), 0xffff_ffff, "clearing flags leaves the enables alone");

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
    reg_write(&mut s, intr::ENABLE_SET, INIT_MASK);
    assert_eq!(reg_read(&mut s, intr::ENABLE_SET), INIT_MASK);
    assert!(!rec.level(FLAG_VECTOR));

    // The high-water path adds one bit; the rest of the mask stays.
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    assert_eq!(reg_read(&mut s, intr::ENABLE_SET), INIT_MASK | LOW_WATER);
    assert!(rec.level(FLAG_VECTOR));

    // The low-water handler removes it: the level drops, the flag stays.
    reg_write(&mut s, intr::ENABLE_CLEAR, LOW_WATER);
    assert_eq!(reg_read(&mut s, intr::ENABLE_SET), INIT_MASK);
    assert_eq!(reg_read(&mut s, intr::FLAGS) & LOW_WATER, LOW_WATER);
    assert!(!rec.level(FLAG_VECTOR));

    // Disabling with all ones empties the mask.
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    reg_write(&mut s, intr::ENABLE_CLEAR, 0xffff_ffff);
    assert_eq!(reg_read(&mut s, intr::ENABLE_SET), 0);
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
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    assert!(rec.level(FLAG_VECTOR));

    reg_write(&mut s, intr::CLEAR, DMA_DONE);
    assert_eq!(reg_read(&mut s, intr::FLAGS), LOW_WATER);
    assert!(rec.level(FLAG_VECTOR));

    reg_write(&mut s, intr::CLEAR, LOW_WATER);
    assert_eq!(reg_read(&mut s, intr::FLAGS), 0);
    assert_eq!(reg_read(&mut s, intr::ENABLE_SET), LOW_WATER);
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
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    reg_write(&mut s, intr::CLEAR, LOW_WATER);
    assert!(!rec.level(FLAG_VECTOR));

    reg_write(&mut s, intr::FLAGS, LOW_WATER);
    assert_eq!(reg_read(&mut s, intr::FLAGS), LOW_WATER);
    assert!(rec.level(FLAG_VECTOR));

    // Without the low-water bit it is dropped, as a set-only write would not.
    reg_write(&mut s, intr::FLAGS, DMA_DONE);
    assert_eq!(reg_read(&mut s, intr::FLAGS), DMA_DONE);
    assert!(!rec.level(FLAG_VECTOR));

    reg_write(&mut s, intr::FLAGS, 0);
    assert_eq!(reg_read(&mut s, intr::FLAGS), 0);
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
    reg_write(&mut s, intr::ENABLE_SET, INIT_MASK | LOW_WATER);
    reg_write(&mut s, intr::ENABLE_CLEAR, LOW_WATER);
    reg_write(&mut s, intr::CLEAR, LOW_WATER);
    assert!(!rec.level(FLAG_VECTOR));

    cmd(&mut s, 0x0001_0046);
    assert_eq!(reg_read(&mut s, intr::FLAGS) & LOW_WATER, 0);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0) & 0x20000, 0x20000);

    reg_write(&mut s, intr::FLAGS, LOW_WATER);
    assert!(!rec.level(FLAG_VECTOR), "latched, but the source is off");
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
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

    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    assert_eq!(reg_read(&mut s, intr::FLAGS) & LOW_WATER, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true]);

    reg_write(&mut s, intr::CLEAR, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true, false], "clearing the flag sends the clearing packet");

    cmd(&mut s, 0x0001_0046);
    assert_eq!(reg_read(&mut s, intr::FLAGS) & LOW_WATER, 0);
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
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true]);

    s.reset();
    assert_eq!(rec.events(FLAG_VECTOR), [true], "the reset sends no clearing packet");

    route_to_heart(&mut s);
    reg_write(&mut s, route::VECTORS, FLAG_VECTOR + 1);
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
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

    // No destination: the rising edge sends nothing but is held.
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    assert!(rec.silent());

    reg_write(&mut s, ODSY_INT_DEST_UPPER, 0x1234_5678);
    reg_write(&mut s, ODSY_INT_DEST_LOWER, 0x9abc_def0);
    assert_eq!(reg_read(&mut s, ODSY_INT_DEST_UPPER), 0x1234_5678);
    assert_eq!(reg_read(&mut s, ODSY_INT_DEST_LOWER), 0x9abc_def0);
    assert!(rec.silent(), "the crosstalk error destination carries nothing");
    route_to_heart(&mut s);
    assert_eq!(rec.events(FLAG_VECTOR), [true], "the held level goes out with the routing");

    // Acknowledging the still-pending flag is the falling edge.
    reg_write(&mut s, intr::CLEAR, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true, false]);

    // The flag set again readdresses the same destination, setting it.
    reg_write(&mut s, intr::FLAGS, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true, false, true]);
}

/// MODEL CHOICE: the kernel can
/// raise low water through flow control before its interrupt set-up
/// programs the routing, so a flag raised with no destination is held and
/// sent once the routing is written, in the kernel's order (vectors, lower
/// half, upper half), and only once. A level that falls while held sends
/// nothing.
#[test]
fn a_flag_raised_before_the_routing_is_sent_once_it_is_programmed() {
    let (mut s, rec) = board("v12");
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    assert!(rec.silent(), "no packet without a destination");

    reg_write(&mut s, route::VECTORS, SOFTSYNC_VECTOR << 24 | RETRACE_VECTOR << 16 | DMA_VECTOR << 8 | FLAG_VECTOR);
    reg_write(&mut s, route::DEST_LOWER, HEART_PACKET as u32);
    assert!(rec.silent(), "half a destination takes no packet");
    reg_write(&mut s, route::DEST_UPPER, (HEART_WIDGET << 25) as u32);
    assert_eq!(rec.events(FLAG_VECTOR), [true]);

    // Writing the routing again sends nothing more.
    route_to_heart(&mut s);
    assert_eq!(rec.0.lock().len(), 1, "exactly one packet");

    // A level that rises and falls before any routing sends nothing.
    s.reset();
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    reg_write(&mut s, intr::CLEAR, LOW_WATER);
    route_to_heart(&mut s);
    assert_eq!(rec.0.lock().len(), 1, "a level gone before the routing sends nothing");
}

/// A level that falls while its packet is held sends nothing, not even a
/// clear, with a routing only partly programmed. Both halves of the
/// destination without the vector table is a partial routing that composes
/// a packet address, so only the rule that all three are written keeps it
/// silent. Writing the rest of the routing afterwards finds no level and
/// sends nothing.
#[test]
fn a_flag_that_falls_while_held_sends_nothing() {
    let (mut s, rec) = board("v12");
    reg_write(&mut s, route::VECTORS, SOFTSYNC_VECTOR << 24 | RETRACE_VECTOR << 16 | DMA_VECTOR << 8 | FLAG_VECTOR);
    reg_write(&mut s, route::DEST_LOWER, HEART_PACKET as u32);
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    reg_write(&mut s, intr::CLEAR, LOW_WATER);
    assert!(rec.silent(), "vectors and a lower half are no destination");

    s.reset();
    reg_write(&mut s, route::DEST_UPPER, (HEART_WIDGET << 25) as u32);
    reg_write(&mut s, route::DEST_LOWER, HEART_PACKET as u32);
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    assert!(rec.silent(), "both halves without the vectors take no packet");
    reg_write(&mut s, intr::CLEAR, LOW_WATER);
    assert!(rec.silent(), "a held level that falls was never sent, so it sends no clear");

    reg_write(&mut s, route::VECTORS, SOFTSYNC_VECTOR << 24 | RETRACE_VECTOR << 16 | DMA_VECTOR << 8 | FLAG_VECTOR);
    assert!(rec.silent(), "the level is gone when the routing is complete");
}

/// A reset drops a held flag with the level: programming the routing
/// afterwards sends nothing, and the next rise and fall is an ordinary
/// pair of packets.
#[test]
fn a_reset_drops_a_held_flag() {
    let (mut s, rec) = board("v12");
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    assert!(rec.silent(), "no packet without a destination");

    s.reset();
    route_to_heart(&mut s);
    assert!(rec.silent(), "the reset dropped the held flag");

    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    reg_write(&mut s, intr::CLEAR, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true, false], "a rise and its fall are both sent");
}

/// The kernel gives the routed table's low byte to its flag handler and
/// the crosstalk error destination to its error handler, so low water goes
/// through the routed destination with the flag vector.
///
/// QEMU: flag-interrupt-uses-routed-vector.
#[test]
fn the_flag_interrupt_uses_the_routed_flag_vector() {
    let (mut s, rec) = board("v12");
    reg_write(&mut s, ODSY_INT_DEST_UPPER, 0x7000_0000);
    reg_write(&mut s, ODSY_INT_DEST_LOWER, 0x40);
    route_to_heart(&mut s);
    reg_write(&mut s, route::VECTORS, (SOFTSYNC_VECTOR << 24) | 4);

    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    assert_eq!(rec.events(4), [true]);
    reg_write(&mut s, intr::CLEAR, LOW_WATER);
    assert_eq!(rec.events(4), [true, false]);
}

/// Program the routing with `upper` as the destination's upper half and
/// raise the retrace, softsync and DMA-done sources once each, in that
/// order. The returned vectors are the sources that sent a packet.
fn raise_each_source(upper: u32) -> Vec<u32> {
    let (mut s, rec) = board("v12");
    route_to_heart(&mut s);
    reg_write(&mut s, route::DEST_UPPER, upper);
    s.retrace();
    s.front.lock().deliver_softsync();
    s.dma_done();
    [RETRACE_VECTOR, SOFTSYNC_VECTOR, DMA_VECTOR].into_iter().filter(|v| rec.events(*v) == [true]).collect()
}

const HEART_UPPER: u32 = (HEART_WIDGET << 25) as u32;

/// MODEL CHOICE: each of bits 16, 17
/// and 18 of the destination's upper half enables one non-flag source (16
/// DMA-done, 17 retrace, 18 softsync). A source raised while its bit is
/// clear sends nothing and is dropped; the other two are unaffected. The
/// bit-to-source assignment is the model's, not a measurement.
#[test]
fn a_clear_source_enable_bit_suppresses_only_its_source() {
    for (bit, silent) in [
        (route::ENABLE_DMA, DMA_VECTOR),
        (route::ENABLE_RETRACE, RETRACE_VECTOR),
        (route::ENABLE_SOFTSYNC, SOFTSYNC_VECTOR),
    ] {
        let sent = raise_each_source(HEART_UPPER | (ROUTE_ENABLES & !bit));
        let want: Vec<u32> = [RETRACE_VECTOR, SOFTSYNC_VECTOR, DMA_VECTOR].into_iter().filter(|v| *v != silent).collect();
        assert_eq!(sent, want, "bit {:#x} clear suppresses vector {}", bit, silent);
    }
}

/// The control for the test above: with all three bits set every source
/// sends its packet, and with all three clear none does.
#[test]
fn a_set_source_enable_bit_sends_its_source() {
    assert_eq!(raise_each_source(HEART_UPPER | ROUTE_ENABLES), [RETRACE_VECTOR, SOFTSYNC_VECTOR, DMA_VECTOR]);
    for bit in [route::ENABLE_DMA, route::ENABLE_RETRACE, route::ENABLE_SOFTSYNC] {
        let sent = raise_each_source(HEART_UPPER | bit);
        assert_eq!(sent.len(), 1, "bit {:#x} alone sends exactly one source", bit);
    }
    assert!(raise_each_source(HEART_UPPER).is_empty(), "no enable bit set sends nothing");
}

/// The value the shipped kernel stores in the upper half, 0x35cf8000, has
/// all three enable bits set. Its address (15:0) and target widget (28:25)
/// are replaced with HEART's so the packets can be recorded; every other bit
/// is the kernel's. All three sources are delivered.
#[test]
fn the_kernels_upper_half_enables_every_source() {
    let kernel = 0x35cf_8000u32;
    assert_eq!(kernel & ROUTE_ENABLES, ROUTE_ENABLES);
    let upper = (kernel & !(0xffff | (0xf << 25))) | HEART_UPPER;
    assert_eq!(raise_each_source(upper), [RETRACE_VECTOR, SOFTSYNC_VECTOR, DMA_VECTOR]);
}

/// The flag has no bit in the upper half: it is gated by the enable mask
/// alone, so a low-water flag is delivered with every source enable clear.
#[test]
fn the_flag_ignores_the_source_enable_bits() {
    let (mut s, rec) = board("v12");
    route_to_heart(&mut s);
    reg_write(&mut s, route::DEST_UPPER, HEART_UPPER);
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true]);
    reg_write(&mut s, intr::CLEAR, LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true, false]);
}

/// A flag held for want of a routing is sent when the routing completes
/// without the enable bits.
#[test]
fn a_held_flag_is_sent_with_the_source_enables_clear() {
    let (mut s, rec) = board("v12");
    reg_write(&mut s, intr::ENABLE_SET, LOW_WATER);
    reg_write(&mut s, route::VECTORS, SOFTSYNC_VECTOR << 24 | RETRACE_VECTOR << 16 | DMA_VECTOR << 8 | FLAG_VECTOR);
    reg_write(&mut s, route::DEST_LOWER, HEART_PACKET as u32);
    reg_write(&mut s, route::DEST_UPPER, HEART_UPPER);
    assert_eq!(rec.events(FLAG_VECTOR), [true]);
}
