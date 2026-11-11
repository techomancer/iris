//! The loop commands the drivers issue beyond discovery: port names, retry
//! count and the five IRIX-only commands.
//!
//! Ported from tests/qtest/qlogic-isp2200-test.c in
//! jamesbraid/qemu-sgi-origami.

use super::loop_tests::BRINGUP_NS;
use super::*;

const ICB_OPT_HARD: u16 = ICB_OPT_HARD_ADDR;

impl Rig {
    fn port_name(&self, loop_id: u8, option: u8) -> u16 {
        self.call(mbox::GET_PORT_NAME, u16::from(loop_id) << 8 | u16::from(option), 0, 0)
    }

    /// The name a get port name reply carries, in wire order.
    fn port_name_bytes(&self) -> [u8; 8] {
        let mut name = [0u8; 8];
        for (i, slot) in [2, 3, 6, 7].into_iter().enumerate() {
            name[i * 2..i * 2 + 2].copy_from_slice(&self.rd(mb(slot)).to_le_bytes());
        }
        name
    }

    fn assert_retry_count(&self, count: u16, delay: u16) {
        assert_eq!(self.call(mbox::GET_RETRY_COUNT, 0, 0, 0), mbox::COMPLETE);
        assert_eq!([self.rd(mb(1)), self.rd(mb(2)), self.rd(mb(3))], [count, delay, 200]);
        self.ack();
    }

    /// Initialise with the retry bytes in the control block.
    fn rings_retry(&self, count: u8, delay: u8) {
        self.rqlen.set(8);
        self.rslen.set(8);
        self.rq_next.set(0);
        self.rs_next.set(0);
        let mut icb = icb_build(8, 8, u64::from(RQST_AT), u64::from(RESP_AT), 0);
        icb[10] = count;
        icb[11] = delay;
        self.mem.put(ICB_AT, &icb);
        assert_eq!(self.init_firmware(u64::from(ICB_AT)), mbox::COMPLETE);
        self.ack();
    }
}

/// Approximation: get port name. Option 0 is the port name, option 1 the node
/// name, as Linux and NetBSD read them. Each mailbox carries two wire-order
/// bytes with the earlier in its low half, the order NetBSD and qlogicfc
/// decode on this part; mainline Linux decodes the other way. The name agrees
/// byte for byte with the port database's. An empty loop ID, the adapter's
/// own, the fabric port 0x7e and anything above it are refused as the port
/// database refuses them, so is every loop ID before the loop is ready, and
/// an option other than 0 or 1 is a parameter error.
#[test]
fn port_name() {
    let r = Rig::with_disks("portname", &[1, 5]);
    r.start();
    r.rings_with(16, 16, ICB_OPT_HARD, 0, 3);

    // Not ready yet: every loop ID, present or not.
    for id in [1, 5] {
        assert_eq!(r.port_name(id, 0), mbox::COMMAND_ERROR);
        r.ack();
    }
    r.step(BRINGUP_NS);
    r.expect_events(&[event::LIP_OCCURRED, event::LOOP_UP]);
    assert_eq!(r.call(mbox::GET_ADAPTER_ID, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(r.rd(mb(1)), 3);
    r.ack();

    for id in [1u8, 5] {
        assert_eq!(r.call_dma(mbox::GET_PORT_DB, u16::from(id) << 8, 0x9000), mbox::COMPLETE);
        r.ack();
        let pd = r.mem.get(0x9000, 128);
        for option in 0..2 {
            let want = if option == 1 { node_wwn(id) } else { port_wwn(id) };
            assert_eq!(r.port_name(id, option), mbox::COMPLETE);
            // Byte 0 is the name's most significant byte.
            assert_eq!(r.rd(mb(2)), ((want >> 48) as u16).swap_bytes());
            assert_eq!(r.rd(mb(3)), ((want >> 32) as u16).swap_bytes());
            assert_eq!(r.rd(mb(6)), ((want >> 16) as u16).swap_bytes());
            assert_eq!(r.rd(mb(7)), (want as u16).swap_bytes());
            let name = r.port_name_bytes();
            r.ack();
            assert_eq!(name, pd[if option == 1 { 12 } else { 20 }..][..8]);
        }
    }

    for id in [0u8, 3, 6, 0x7d, 0x7e, 0x7f, 0xff] {
        for option in 0..2 {
            assert_eq!(r.port_name(id, option), mbox::COMMAND_ERROR, "loop ID {id}");
            r.ack();
        }
    }
    assert_eq!(r.port_name(5, 2), mbox::PARAM_ERROR);
    r.ack();
    assert_eq!(r.port_name(1, 0xff), mbox::PARAM_ERROR);
    r.ack();
}

/// Approximation: IRIX 6.5.30 rang get firmware state on the model while the
/// bring-up's loop-up event was posted and unread. The event stays readable
/// until its release, and the completion then follows with the loop ready, so
/// neither is lost.
#[test]
fn writes_behind_loop_up() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 8);
    r.step(BRINGUP_NS);
    assert_eq!(r.rd(mb(0)), event::LIP_OCCURRED);
    r.ack();
    assert_eq!(r.rd(mb(0)), event::LOOP_UP);

    r.wr(mb(0), mbox::GET_FW_STATE);
    r.wr(reg::HCCR, HCCR_SET_HOST);
    assert_eq!(r.rd(mb(0)), event::LOOP_UP);
    assert_eq!(r.rd(mb(1)), 0);
    r.ack();

    assert_eq!(r.rd(reg::SEMAPHORE), SEM_LOCK);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    assert_eq!(r.rd(mb(1)), FW_STATE_READY);
    r.ack();
    r.expect_no_event();
}

/// Approximation: get retry count returns the control block's retry count in
/// m1 and retry delay in m2, the reading Linux and NetBSD give the command,
/// and a constant m3 no public source settles. It answers from the moment
/// initialise-firmware completes, loop up or not, and a later initialisation
/// replaces the pair. Before any control block it is refused with 0x4005, and
/// before execute with 0x4001.
#[test]
fn retry_count() {
    let r = Rig::bare();
    assert_eq!(r.call_ack(mbox::GET_RETRY_COUNT, 0, 0, 0), mbox::INVALID);
    r.start();
    assert_eq!(r.call_ack(mbox::GET_RETRY_COUNT, 0, 0, 0), mbox::COMMAND_ERROR);

    r.rings_retry(6, 9);
    assert_eq!(r.fw_state(), FW_STATE_WAIT_AL_PA);
    r.assert_retry_count(6, 9);
    r.step(BRINGUP_NS);
    r.expect_events(&[event::LIP_OCCURRED, event::LOOP_UP]);
    assert_eq!(r.fw_state(), FW_STATE_READY);
    r.assert_retry_count(6, 9);

    r.rings_retry(2, 3);
    r.assert_retry_count(2, 3);

    // Each byte is zero-extended, not sign-extended.
    r.rings_retry(0xfe, 0x81);
    r.assert_retry_count(0xfe, 0x81);

    // A reset drops the control block with the firmware.
    r.wr(reg::HCCR, HCCR_RESET);
    r.wr(reg::HCCR, HCCR_RELEASE);
    assert_eq!(r.call_ack(mbox::GET_RETRY_COUNT, 0, 0, 0), mbox::INVALID);
    r.start();
    assert_eq!(r.call_ack(mbox::GET_RETRY_COUNT, 0, 0, 0), mbox::COMMAND_ERROR);
}

/// Approximation: get port queue parameters, set retry count, reset link
/// status, get link status and set port queue parameters are sent by IRIX
/// 6.5.30 and no public source settles the firmware's replies. The model
/// completes each: queue depth 256 and throttle 255 for a loop ID that has a
/// port, retry count accepted and not stored, link counters all zero. A loop
/// ID with no port is refused with 0x4005, never 0x4001, which IRIX prints as
/// an invalid command. The adapter's own loop ID is answered by get link
/// status, which the driver polls for it, with the same zero counters, and
/// refused by get port queue parameters, which it asks of discovered ports
/// only. While the loop is down all of these but set port queue parameters
/// are refused. That command completes with 0x4000 for any loop ID and leaves
/// the guest's registers as written, since IRIX writes only m0 and m1 and
/// ignores the reply.
#[test]
fn irix_loop_commands() {
    let r = Rig::with_disks("irixcmds", &[1, 5]);
    let buf = 0x9000u32;

    // The loop is not up yet: a port-addressed command has no port.
    r.start();
    r.mem.put(buf, &[0xa5; 64]);
    assert_eq!(r.call_ack(mbox::GET_PORT_QUEUE, 1 << 8, 0, 0), mbox::COMMAND_ERROR);
    assert_eq!(r.call_dma(mbox::GET_LINK_STATUS, 1 << 8, u64::from(buf)), mbox::COMMAND_ERROR);
    r.ack();
    assert_eq!(r.mem.get(buf, 64), vec![0xa5; 64]);
    assert_eq!(r.call_ack(mbox::RESET_LINK_STATUS, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(r.call_ack(mbox::SET_PORT_QUEUE, 1 << 8, 0, 0), mbox::COMPLETE);

    r.rings_retry(4, 15);
    r.step(BRINGUP_NS);
    r.expect_events(&[event::LIP_OCCURRED, event::LOOP_UP]);

    // Loop ID 0 is the adapter's own: link status only.
    assert_eq!(r.call_ack(mbox::GET_PORT_QUEUE, 0, 0, 0), mbox::COMMAND_ERROR);
    r.mem.put(buf, &[0xa5; 64]);
    assert_eq!(r.call_dma(mbox::GET_LINK_STATUS, 0, u64::from(buf)), mbox::COMPLETE);
    r.ack();
    let seen = r.mem.get(buf, 64);
    assert_eq!(seen[..32], [0; 32]);
    assert_eq!(seen[32..], [0xa5; 32]);

    for id in [1u16, 5] {
        assert_eq!(r.call(mbox::GET_PORT_QUEUE, id << 8, 0, 0), mbox::COMPLETE);
        assert_eq!((r.rd(mb(2)), r.rd(mb(3))), (256, 255));
        r.ack();

        // Exactly 32 bytes, the length IRIX copies, and all zero.
        r.mem.put(buf, &[0xa5; 64]);
        assert_eq!(r.call_dma(mbox::GET_LINK_STATUS, id << 8, u64::from(buf)), mbox::COMPLETE);
        r.ack();
        let seen = r.mem.get(buf, 64);
        assert_eq!((seen[..32].to_vec(), seen[32..].to_vec()), (vec![0; 32], vec![0xa5; 32]));
    }

    let absent = [2u16, 6, 0x7d, 0x7e, 0xff];
    for id in absent {
        assert_eq!(r.call_ack(mbox::GET_PORT_QUEUE, id << 8, 0, 0), mbox::COMMAND_ERROR);
        r.mem.put(buf, &[0xa5; 64]);
        assert_eq!(r.call_dma(mbox::GET_LINK_STATUS, id << 8, u64::from(buf)), mbox::COMMAND_ERROR);
        r.ack();
        assert_eq!(r.mem.get(buf, 64), vec![0xa5; 64]);
    }

    // Set port queue parameters: the driver writes m0 and m1 only, so m2 and
    // m3 hold what was there before. The reply is 0x4000, not 0x4001, for any
    // loop ID, and the registers read back as written.
    assert_eq!(r.call(mbox::GET_PORT_QUEUE, 1 << 8, 0, 0), mbox::COMPLETE);
    r.ack();
    r.wr(mb(2), 0x1234);
    r.wr(mb(3), 0x5678);
    r.wr(mb(1), 1 << 8);
    r.wr(mb(0), mbox::SET_PORT_QUEUE);
    r.wr(reg::HCCR, HCCR_SET_HOST);
    assert_eq!([r.rd(mb(0)), r.rd(mb(1)), r.rd(mb(2)), r.rd(mb(3))], [mbox::COMPLETE, 1 << 8, 0x1234, 0x5678]);
    r.ack();
    for id in absent {
        assert_eq!(r.call_ack(mbox::SET_PORT_QUEUE, id << 8, 1, 1), mbox::COMPLETE);
    }
    assert_eq!(r.call_ack(mbox::SET_PORT_QUEUE, 0, 0xffff, 0xffff), mbox::COMPLETE);
    // Another unserved command is still refused.
    assert_eq!(r.call_ack(0x003a, 0, 0, 0), mbox::INVALID);

    // Neither disturbs the loop, the events, or the retry pair.
    assert_eq!(r.call_ack(mbox::RESET_LINK_STATUS, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(r.call(mbox::SET_RETRY_COUNT, 7, 8, 0), mbox::COMPLETE);
    for i in 1..8 {
        assert_eq!(r.rd(mb(i)), 0);
    }
    r.ack();
    r.expect_no_event();
    assert_eq!(r.fw_state(), FW_STATE_READY);
    r.assert_retry_count(4, 15);
}

/// A refused DMA is 0x4002, as for the model's other DMA commands.
#[test]
fn link_status_dma_refused() {
    let r = Rig::with_disks("linkstatus", &[1]);
    r.loop_up(0, 0);
    r.isp.state().config().write(1, 0x3);
    assert_eq!(r.call_dma(mbox::GET_LINK_STATUS, 1 << 8, 0x9000), mbox::HOST_INTERFACE_ERROR);
    r.ack();
}
