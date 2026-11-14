// Cases follow tests/qtest/sn0-hub-test.c in jamesbraid/qemu-sgi-origami:
// status-field-access, network-link-presence,
// a-node-with-no-link-reads-down, peerless-link-reset,
// local-port-reset-resets-the-node, network-scratch-registers,
// vector-pio-direct-peer, vector-launch-no-fabric-stays-silent,
// llp-port-registers, llp-port-error-clear-contract,
// vector-writes-land-only-on-rows-that-store, the *-answer-vector-decode
// cases, io-protect-registers, ni-protection-config, ni-diag-parms,
// network-routes-reset-to-zero and network-ageing-controls.

use super::*;
use std::sync::atomic::AtomicU32;

struct Node {
    ni: Arc<Ni>,
}
fn node(revision: u8) -> Node {
    Node { ni: Arc::new(Ni::new(revision, Routing::new())) }
}
/// Two Hubs on a direct cable, the two-module Origin 200.
fn pair() -> (Node, Node) {
    let (a, b) = (node(3), node(3));
    let (wa, wb): (Weak<dyn NetPeer>, Weak<dyn NetPeer>) = (Arc::downgrade(&a.ni) as _, Arc::downgrade(&b.ni) as _);
    a.ni.connect_peer(Some(wb));
    b.ni.connect_peer(Some(wa));
    (a, b)
}
fn r(ni: &Ni, off: u64) -> u64 {
    ni.read64(off).data
}
fn w(ni: &Ni, off: u64, value: u64) -> Effect {
    ni.write(off, value)
}

/// Run a vector operation from `ni` to the Hub across the cable and return
/// the completion status.
fn vector(ni: &Ni, route: u64, off: u64, ty: u64, data: u64) -> u64 {
    w(ni, reg::NI_VECTOR::OFFSET, route);
    w(ni, reg::NI_VECTOR_DATA::OFFSET, data);
    w(ni, reg::NI_VECTOR_PARMS::OFFSET, (off - reg::NI_STATUS_REV_ID::OFFSET) | ty);
    r(ni, reg::NI_VECTOR_STATUS::OFFSET)
}
const VALID: u64 = 1 << 63;

#[test]
fn the_status_register_names_the_node_the_mode_and_the_chip_revision() {
    let n = node(3);
    let status = r(&n.ni, reg::NI_STATUS_REV_ID::OFFSET);
    // Region size set, M mode (bit 18 clear), never out of reset, revision 3.
    assert_eq!(status, STATUS_REGION_SIZE | STATUS_NEVER_OUT_OF_RESET | (3 << 4));
    // Firmware assigns node numbers through this register.
    w(&n.ni, reg::NI_STATUS_REV_ID::OFFSET, STATUS_NODE_MASK);
    assert_eq!(r(&n.ni, reg::NI_STATUS_REV_ID::OFFSET) & STATUS_NODE_MASK, STATUS_NODE_MASK);
    assert_eq!(n.ni.number(), 0x1ff);
    // The revision and the link bits are not writable.
    w(&n.ni, reg::NI_STATUS_REV_ID::OFFSET, !0);
    assert_eq!(r(&n.ni, reg::NI_STATUS_REV_ID::OFFSET) & 0xff, 3 << 4);
    assert_eq!(r(&n.ni, reg::NI_STATUS_REV_ID::OFFSET) & (STATUS_LINK_UP | STATUS_EIGHT_BIT), 0);
    // Mode: bit 18 writes (N mode), region size writes.
    w(&n.ni, reg::NI_STATUS_REV_ID::OFFSET, 0);
    assert_eq!(r(&n.ni, reg::NI_STATUS_REV_ID::OFFSET) & (STATUS_MORE_NODES | STATUS_REGION_SIZE), 0);
}

#[test]
fn a_new_node_number_is_returned_for_the_machine_and_mirrored_for_the_decode() {
    let n = node(3);
    assert!(matches!(w(&n.ni, reg::NI_STATUS_REV_ID::OFFSET, 5 << 8), Effect::NodeNumber));
    assert_eq!(n.ni.number(), 5);
    // Writing the number it already has is not a change.
    assert!(matches!(w(&n.ni, reg::NI_STATUS_REV_ID::OFFSET, 5 << 8), Effect::None));
}

#[test]
fn a_node_with_no_link_never_came_out_of_reset() {
    // The guest kernel treats down with the never-out-of-reset reason clear as a lost
    // CrayLink on every bus error, so the reason must read one.
    let n = node(3);
    let down = |n: &Node| r(&n.ni, reg::NI_STATUS_REV_ID::OFFSET) & (STATUS_LINK_UP | STATUS_NEVER_OUT_OF_RESET);
    assert_eq!(down(&n), STATUS_NEVER_OUT_OF_RESET);
    n.ni.power_on();
    assert_eq!(down(&n), STATUS_NEVER_OUT_OF_RESET);
    // Read-only: a guest write cannot claim the link failed.
    w(&n.ni, reg::NI_STATUS_REV_ID::OFFSET, 0);
    assert_eq!(down(&n), STATUS_NEVER_OUT_OF_RESET);
}

#[test]
fn a_link_reset_with_no_peer_latches_the_event_and_leaves_the_link_down() {
    let n = node(3);
    w(&n.ni, reg::NI_STATUS_REV_ID::OFFSET, 9 << 8);
    w(&n.ni, reg::NI_SCRATCH_REG0::OFFSET, 0x55);
    assert_eq!(r(&n.ni, reg::NI_PORT_ERROR::OFFSET), 0);
    w(&n.ni, reg::NI_PORT_RESET::OFFSET, PORT_RESET_LINK);
    assert_eq!(r(&n.ni, reg::NI_PORT_ERROR::OFFSET), PORT_ERROR_LINK_RESET);
    assert_eq!(r(&n.ni, reg::NI_STATUS_REV_ID::OFFSET) & (STATUS_LINK_UP | STATUS_NEVER_OUT_OF_RESET), STATUS_NEVER_OUT_OF_RESET);
    // The node number, scratch state and routing keep their values.
    assert_eq!((n.ni.number(), r(&n.ni, reg::NI_SCRATCH_REG0::OFFSET)), (9, 0x55));
    // The kernel's power-off path reads the clearing alias until bit 37 clears: a
    // single clearing read sees it set once and then clear.
    assert_eq!(r(&n.ni, reg::NI_PORT_ERROR_CLEAR::OFFSET), PORT_ERROR_LINK_RESET);
    assert_eq!(r(&n.ni, reg::NI_PORT_ERROR_CLEAR::OFFSET), 0);

}

#[test]
fn the_local_bit_of_the_port_reset_restarts_the_machine_with_either_companion() {
    for value in [0x81, 0x03, 0x01] {
        let n = node(3);
        assert!(matches!(w(&n.ni, reg::NI_PORT_RESET::OFFSET, value), Effect::Restart), "{value:#x}");
        // Nothing is stored and the link reset is superseded.
        assert_eq!(r(&n.ni, reg::NI_PORT_ERROR::OFFSET), 0);
    }
    // Bit 7 alone is logged and does nothing.
    let n = node(3);
    assert!(matches!(w(&n.ni, reg::NI_PORT_RESET::OFFSET, PORT_RESET_WARM), Effect::None));
}

#[test]
fn two_cabled_hubs_read_link_up_and_a_lost_peer_reads_failed() {
    let (a, b) = pair();
    let up = |n: &Node| r(&n.ni, reg::NI_STATUS_REV_ID::OFFSET) & (STATUS_LINK_UP | STATUS_NEVER_OUT_OF_RESET | STATUS_EIGHT_BIT);
    // Coming up clears the never-out-of-reset reason until the next reset.
    assert_eq!((up(&a), up(&b)), (STATUS_LINK_UP, STATUS_LINK_UP));
    a.ni.power_on();
    assert_eq!(up(&a), STATUS_LINK_UP);
    // The peer's module loses power: this end reads down, with the reason
    // left at "failed".
    b.ni.set_enabled(false);
    assert_eq!(r(&a.ni, reg::NI_STATUS_REV_ID::OFFSET) & (STATUS_LINK_UP | STATUS_NEVER_OUT_OF_RESET), 0);
    b.ni.set_enabled(true);
    assert_eq!(up(&a), STATUS_LINK_UP);
}

#[test]
fn a_link_reset_settles_the_width_from_both_ends_force_bits() {
    let (a, b) = pair();
    let eight = |n: &Node| r(&n.ni, reg::NI_STATUS_REV_ID::OFFSET) & STATUS_EIGHT_BIT != 0;
    // Writing the force bit alone changes nothing until the next link reset.
    w(&b.ni, reg::NI_DIAG_PARMS::OFFSET, DIAG_EIGHT_BIT);
    assert_eq!((eight(&a), eight(&b)), (false, false));
    w(&a.ni, reg::NI_PORT_RESET::OFFSET, PORT_RESET_LINK);
    // Eight bits if either end forces it, reported at both ends.
    assert_eq!((eight(&a), eight(&b)), (true, true));
    // Link up stays set, and both Hub ends latch the link-reset bit.
    assert_eq!(r(&a.ni, reg::NI_STATUS_REV_ID::OFFSET) & STATUS_LINK_UP, STATUS_LINK_UP);
    assert_eq!((r(&a.ni, reg::NI_PORT_ERROR::OFFSET), r(&b.ni, reg::NI_PORT_ERROR::OFFSET)), (PORT_ERROR_LINK_RESET, PORT_ERROR_LINK_RESET));
    // Sixteen again when nobody forces eight.
    w(&b.ni, reg::NI_DIAG_PARMS::OFFSET, 0);
    w(&a.ni, reg::NI_PORT_RESET::OFFSET, PORT_RESET_LINK);
    assert_eq!((eight(&a), eight(&b)), (false, false));
}

#[test]
fn vector_pio_reaches_the_direct_peer() {
    let (a, b) = pair();
    // Read the peer's scratch register.
    w(&b.ni, reg::NI_SCRATCH_REG0::OFFSET, 0x0123_4567_89ab_cdef);
    let status = vector(&a.ni, 0, reg::NI_SCRATCH_REG0::OFFSET, VECTOR_READ, 0);
    assert_eq!(status & (VALID | 3), VALID | VECTOR_READ);
    assert_eq!(r(&a.ni, reg::NI_VECTOR_READ_DATA::OFFSET), 0x0123_4567_89ab_cdef);
    // Reading the clearing alias drops the completion.
    assert_eq!(r(&a.ni, reg::NI_VECTOR_CLEAR::OFFSET), 0);
    assert_eq!(r(&a.ni, reg::NI_VECTOR_STATUS::OFFSET) & VALID, 0);
    // Write.
    let status = vector(&a.ni, 0, reg::NI_SCRATCH_REG1::OFFSET, VECTOR_WRITE, 0xfedc_ba98_7654_3210);
    assert_eq!(status & (VALID | 3), VALID | VECTOR_WRITE);
    assert_eq!(r(&b.ni, reg::NI_SCRATCH_REG1::OFFSET), 0xfedc_ba98_7654_3210);
    // An exchange returns the old value and writes only over a zero.
    w(&b.ni, reg::NI_SCRATCH_REG1::OFFSET, 0);
    let status = vector(&a.ni, 0, reg::NI_SCRATCH_REG1::OFFSET, VECTOR_EXCHANGE, 0x1122_3344_5566_7788);
    assert_eq!(status & (VALID | 3), VALID | VECTOR_EXCHANGE);
    assert_eq!((r(&a.ni, reg::NI_VECTOR_READ_DATA::OFFSET), r(&b.ni, reg::NI_SCRATCH_REG1::OFFSET)), (0, 0x1122_3344_5566_7788));
    vector(&a.ni, 0, reg::NI_SCRATCH_REG1::OFFSET, VECTOR_EXCHANGE, 0x8877_6655_4433_2211);
    assert_eq!((r(&a.ni, reg::NI_VECTOR_READ_DATA::OFFSET), r(&b.ni, reg::NI_SCRATCH_REG1::OFFSET)), (0x1122_3344_5566_7788, 0x1122_3344_5566_7788));
    // The routing tables answer: a local entry keeps four bits.
    assert_eq!(r(&b.ni, reg::NI_LOCAL_TABLE0::OFFSET), 0);
    vector(&a.ni, 0, reg::NI_LOCAL_TABLE0::OFFSET, VECTOR_WRITE, 0x1b);
    assert_eq!(r(&b.ni, reg::NI_LOCAL_TABLE0::OFFSET), 0xb);
    vector(&a.ni, 0, reg::NI_LOCAL_TABLE0::OFFSET, VECTOR_READ, 0);
    assert_eq!(r(&a.ni, reg::NI_VECTOR_READ_DATA::OFFSET), 0xb);
    let last = reg::NI_META_TABLE0::OFFSET + 8 * 31;
    vector(&a.ni, 0, last, VECTOR_WRITE, 0x16);
    assert_eq!(r(&b.ni, last), 6);
}

#[test]
fn a_vector_completion_carries_the_answering_node_the_ids_and_the_address() {
    let (a, b) = pair();
    w(&b.ni, reg::NI_STATUS_REV_ID::OFFSET, 0x1a5 << 8);
    let ids = (0x2a5 << 40) | (0x5c << 32);
    w(&a.ni, reg::NI_VECTOR_DATA::OFFSET, 0);
    w(&a.ni, reg::NI_VECTOR::OFFSET, 0);
    w(&a.ni, reg::NI_VECTOR_PARMS::OFFSET, ids | (reg::NI_SCRATCH_REG0::OFFSET - reg::NI_STATUS_REV_ID::OFFSET) | VECTOR_READ);
    // Valid, the target in 60:51, the identifiers, address bits 31:3, the type.
    assert_eq!(r(&a.ni, reg::NI_VECTOR_STATUS::OFFSET), VALID | (0x1a5 << 51) | ids | 0x100 | VECTOR_READ);
    // Only a zero write clears the completion.
    w(&a.ni, reg::NI_VECTOR_STATUS::OFFSET, 5);
    assert_ne!(r(&a.ni, reg::NI_VECTOR_STATUS::OFFSET), 0);
    w(&a.ni, reg::NI_VECTOR_STATUS::OFFSET, 0);
    assert_eq!(r(&a.ni, reg::NI_VECTOR_STATUS::OFFSET), 0);
}

#[test]
fn a_vector_that_the_cable_cannot_carry_or_the_target_does_not_decode_errors_out() {
    let (a, b) = pair();
    // A nonzero route needs a Router.
    let status = vector(&a.ni, 3, reg::NI_SCRATCH_REG0::OFFSET, VECTOR_READ, 0);
    assert_eq!(status & (VALID | 7), VALID | VECTOR_ADDRESS_ERROR);
    // A register outside the vector decode, a type the protocol lacks, and a
    // write to a read-to-clear alias all answer with the address error.
    for (off, ty) in [(reg::NI_VECTOR_PARMS::OFFSET, VECTOR_READ), (reg::NI_SCRATCH_REG0::OFFSET, 2), (reg::NI_PORT_ERROR_CLEAR::OFFSET, VECTOR_WRITE)] {
        let status = vector(&a.ni, 0, off, ty, !0);
        assert_eq!(status & (VALID | 7), VALID | VECTOR_ADDRESS_ERROR, "{off:#x} type {ty}");
    }
    assert_eq!(r(&b.ni, reg::NI_SCRATCH_REG0::OFFSET), 0, "a refused request wrote nothing");
    // A powered-down peer answers nothing: the poll times out.
    w(&a.ni, reg::NI_VECTOR_STATUS::OFFSET, 0);
    b.ni.set_enabled(false);
    assert_eq!(vector(&a.ni, 0, reg::NI_SCRATCH_REG0::OFFSET, VECTOR_READ, 0), 0);
}

#[test]
fn a_vector_launch_with_no_peer_latches_nothing() {
    let n = node(3);
    w(&n.ni, reg::NI_VECTOR_DATA::OFFSET, 0x0123_4567_89ab_cdef);
    assert_eq!(vector(&n.ni, 0, reg::NI_SCRATCH_REG0::OFFSET, VECTOR_READ, 0), 0);
    assert_eq!(vector(&n.ni, 0, reg::NI_SCRATCH_REG1::OFFSET, VECTOR_WRITE, 5), 0);
    assert_eq!(r(&n.ni, reg::NI_VECTOR_READ_DATA::OFFSET), 0);
}

#[test]
fn vector_writes_land_only_on_rows_that_store() {
    let (a, b) = pair();
    // The port error answers the vector decode with its write-one-to-clear rule.
    b.ni.latch_port_error(0x300);
    let status = vector(&a.ni, 0, reg::NI_PORT_ERROR::OFFSET, VECTOR_WRITE, !0);
    assert_eq!(status & (VALID | 7), VALID | VECTOR_WRITE);
    assert_eq!(r(&b.ni, reg::NI_PORT_ERROR::OFFSET), 0);
    // The clearing alias returns and clears on a vector read.
    b.ni.latch_port_error(0x5);
    vector(&a.ni, 0, reg::NI_PORT_ERROR_CLEAR::OFFSET, VECTOR_READ, 0);
    assert_eq!((r(&a.ni, reg::NI_VECTOR_READ_DATA::OFFSET), r(&b.ni, reg::NI_PORT_ERROR::OFFSET)), (5, 0));
    // The remote vector write of the node number renumbers the neighbour,
    // which the launching Hub reports for the neighbour's machine, and the
    // number each end mirrors for the decode follows.
    w(&a.ni, reg::NI_VECTOR::OFFSET, 0);
    w(&a.ni, reg::NI_VECTOR_DATA::OFFSET, 3 << 8);
    let effect = w(&a.ni, reg::NI_VECTOR_PARMS::OFFSET, (reg::NI_STATUS_REV_ID::OFFSET - reg::NI_STATUS_REV_ID::OFFSET) | VECTOR_WRITE);
    assert!(matches!(effect, Effect::PeerNodeNumber), "{effect:?}");
    assert_eq!(b.ni.number(), 3);
    assert_eq!(a.ni.neighbour_number(), Some(3));
}

#[test]
fn the_protection_and_diagnostic_registers_store_and_answer_the_vector_decode() {
    let (a, b) = pair();
    // Protection: one permissive bit, resetting set, never consulted.
    assert_eq!(r(&a.ni, reg::NI_PROTECTION::OFFSET), 1);
    w(&a.ni, reg::NI_PROTECTION::OFFSET, !0);
    assert_eq!(r(&a.ni, reg::NI_PROTECTION::OFFSET), 1);
    w(&a.ni, reg::NI_PROTECTION::OFFSET, 0);
    assert_eq!(r(&a.ni, reg::NI_PROTECTION::OFFSET), 0);
    // Link diagnostics: 19 bits stored.
    w(&a.ni, reg::NI_DIAG_PARMS::OFFSET, !0);
    assert_eq!(r(&a.ni, reg::NI_DIAG_PARMS::OFFSET), 0x7_ffff);
    // PIO protection resets all ones and stores the whole word; the override
    // is write-only and stores nothing.
    assert_eq!(r(&a.ni, reg::NI_IO_PROTECT::OFFSET), !0);
    w(&a.ni, reg::NI_IO_PROTECT::OFFSET, 0x0123_4567_89ab_cdef);
    w(&a.ni, reg::NI_IO_PROT_OVRRD::OFFSET, 0x5347_4972_756c_6573);
    assert_eq!((r(&a.ni, reg::NI_IO_PROTECT::OFFSET), r(&a.ni, reg::NI_IO_PROT_OVRRD::OFFSET)), (0x0123_4567_89ab_cdef, 0));
    // The stock PROM's link diagnostic vector-writes a peer's PIO protection
    // and vector-reads its port error; the kernel vector-accesses the diagnostics.
    for off in [reg::NI_IO_PROTECT::OFFSET, reg::NI_PROTECTION::OFFSET, reg::NI_DIAG_PARMS::OFFSET, reg::NI_PORT_PARMS::OFFSET, reg::NI_PORT_ERROR::OFFSET, reg::NI_AGE_CPU0_MEMORY::OFFSET] {
        let status = vector(&a.ni, 0, off, VECTOR_READ, 0);
        assert_eq!(status & (VALID | 7), VALID | VECTOR_READ, "{off:#x}");
    }
    vector(&a.ni, 0, reg::NI_IO_PROTECT::OFFSET, VECTOR_WRITE, 0xfedc_ba98_7654_3210);
    assert_eq!(r(&b.ni, reg::NI_IO_PROTECT::OFFSET), 0xfedc_ba98_7654_3210);
    // The override is write-only: its vector read keeps the undecoded answer.
    let status = vector(&a.ni, 0, reg::NI_IO_PROT_OVRRD::OFFSET, VECTOR_READ, 0);
    assert_eq!(status & (VALID | 7), VALID | VECTOR_ADDRESS_ERROR);
    let status = vector(&a.ni, 0, reg::NI_IO_PROT_OVRRD::OFFSET, VECTOR_WRITE, 0x5347_4972_756c_6573);
    assert_eq!(status & (VALID | 7), VALID | VECTOR_WRITE);
}

#[test]
fn the_port_registers_reset_to_the_models_values() {
    let n = node(3);
    // The null-packet timeout in bits 15:10 and the maximum burst in 9:0.
    assert_eq!(r(&n.ni, reg::NI_PORT_PARMS::OFFSET), 0x1bf0);
    assert_eq!((r(&n.ni, reg::NI_PORT_PARMS::OFFSET) >> 10) & 0x3f, 6);
    w(&n.ni, reg::NI_PORT_PARMS::OFFSET, !0);
    assert_eq!(r(&n.ni, reg::NI_PORT_PARMS::OFFSET), 0xffff);
    // The error register holds 38 bits; writing ones clears them, and the
    // clearing alias returns and clears.
    n.ni.latch_port_error(!0);
    assert_eq!(r(&n.ni, reg::NI_PORT_ERROR::OFFSET), 0x3f_ffff_ffff);
    w(&n.ni, reg::NI_PORT_ERROR::OFFSET, 0x1);
    assert_eq!(r(&n.ni, reg::NI_PORT_ERROR::OFFSET), 0x3f_ffff_fffe);
    assert_eq!(r(&n.ni, reg::NI_PORT_ERROR_CLEAR::OFFSET), 0x3f_ffff_fffe);
    assert_eq!(r(&n.ni, reg::NI_PORT_ERROR::OFFSET), 0);
    // A 32-bit half reads the alias without clearing it.
    n.ni.latch_port_error(0x10);
    assert_eq!(n.ni.read(reg::NI_PORT_ERROR_CLEAR::OFFSET, false), 0x10);
    assert_eq!(n.ni.read(reg::NI_PORT_ERROR_CLEAR::OFFSET, true), 0x10);
    n.ni.power_on();
    assert_eq!(r(&n.ni, reg::NI_PORT_PARMS::OFFSET), 0x1bf0);
}

#[test]
fn a_check_bit_error_count_saturates_at_255() {
    let n = node(3);
    assert!(!n.ni.sends_data_error());
    w(&n.ni, reg::NI_DIAG_PARMS::OFFSET, DIAG_SEND_DATA_ERROR);
    assert!(n.ni.sends_data_error());
    for _ in 0..300 {
        n.ni.record_check_error();
    }
    assert_eq!(r(&n.ni, reg::NI_PORT_ERROR::OFFSET) >> 8 & 0xff, 255);
    // Nothing else moved.
    assert_eq!(r(&n.ni, reg::NI_PORT_ERROR::OFFSET) & !0xff00, 0);
}

#[test]
fn the_routing_tables_hold_four_bits_and_a_fresh_entry_routes_nothing() {
    let n = node(3);
    w(&n.ni, reg::NI_STATUS_REV_ID::OFFSET, 0x23 << 8);
    for i in 0..32 {
        assert_eq!(r(&n.ni, reg::NI_META_TABLE0::OFFSET + i * 8), 0);
    }
    for i in 0..16 {
        assert_eq!(r(&n.ni, reg::NI_LOCAL_TABLE0::OFFSET + i * 8), 0);
    }
    // An entry not written since reset routes nothing, even though it reads zero.
    assert_eq!(n.ni.first_turn(0x25), None);
    assert_eq!(n.ni.first_turn(0x41), None);
    // A destination in this Hub's own group of sixteen uses the local entry
    // for its low four bits; any other uses the meta entry for its high bits.
    w(&n.ni, reg::NI_LOCAL_TABLE0::OFFSET + 8 * 5, 0xfe);
    w(&n.ni, reg::NI_META_TABLE0::OFFSET + 8 * 4, 0x7);
    assert_eq!((r(&n.ni, reg::NI_LOCAL_TABLE0::OFFSET + 8 * 5), r(&n.ni, reg::NI_META_TABLE0::OFFSET + 8 * 4)), (0xe, 7));
    assert_eq!(n.ni.first_turn(0x25), Some(0xe));
    assert_eq!(n.ni.first_turn(0x41), Some(7));
    assert_eq!(n.ni.first_turn(0x26), None);
    assert_eq!(n.ni.first_turn(256), None);
    // Reset forgets a route.
    n.ni.power_on();
    assert_eq!(n.ni.first_turn(0x41), None);
}

#[test]
fn the_scratch_ageing_and_vector_registers_store_what_is_written() {
    let n = node(3);
    w(&n.ni, reg::NI_SCRATCH_REG0::OFFSET, 0x1111_2222_3333_4444);
    w(&n.ni, reg::NI_SCRATCH_REG1::OFFSET, !0);
    assert_eq!((r(&n.ni, reg::NI_SCRATCH_REG0::OFFSET), r(&n.ni, reg::NI_SCRATCH_REG1::OFFSET)), (0x1111_2222_3333_4444, !0));
    for i in 0..8 {
        assert_eq!(r(&n.ni, reg::NI_AGE_CPU0_MEMORY::OFFSET + i * 8), 0);
    }
    w(&n.ni, reg::NI_AGE_CPU0_MEMORY::OFFSET, !0);
    assert_eq!(r(&n.ni, reg::NI_AGE_CPU0_MEMORY::OFFSET), 0xfff);
    for i in 1..8 {
        w(&n.ni, reg::NI_AGE_CPU0_MEMORY::OFFSET + i * 8, i);
        assert_eq!(r(&n.ni, reg::NI_AGE_CPU0_MEMORY::OFFSET + i * 8), i);
    }
    for off in [reg::NI_VECTOR::OFFSET, reg::NI_VECTOR_DATA::OFFSET, reg::NI_RETURN_VECTOR::OFFSET, reg::NI_VECTOR_READ_DATA::OFFSET] {
        w(&n.ni, off, 0xa5a5_a5a5_a5a5_a5a5);
        assert_eq!(r(&n.ni, off), 0xa5a5_a5a5_a5a5_a5a5);
    }
    n.ni.power_on();
    assert_eq!((r(&n.ni, reg::NI_SCRATCH_REG0::OFFSET), r(&n.ni, reg::NI_AGE_CPU0_MEMORY::OFFSET), r(&n.ni, reg::NI_VECTOR::OFFSET)), (0, 0, 0));
    // Unimplemented offsets read zero.
    w(&n.ni, reg::NI_STATUS_REV_ID::OFFSET + 0x1f8, 3);
    assert_eq!(r(&n.ni, reg::NI_STATUS_REV_ID::OFFSET + 0x1f8), 0);
}
