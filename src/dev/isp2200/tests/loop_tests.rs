//! The private loop: discovery, bring-up, SCSI commands to the disks on the
//! loop and recovery.
//!
//! Ported from tests/qtest/qlogic-isp2200-test.c in
//! jamesbraid/qemu-sgi-origami.

use super::*;
use crate::scsi::{ScsiResponse, ScsiTarget};
use std::io;
use std::path::PathBuf;

/// The model's loop bring-up time.
pub(super) const BRINGUP_NS: i64 = BRINGUP.as_nanos() as i64;

const ICB_OPT_HARD: u16 = ICB_OPT_HARD_ADDR;
const ICB_OPT_PORT_DB: u16 = ICB_OPT_PORT_DB_EVENTS;

const BRINGUP_EVENTS: [u16; 3] = [event::LIP_OCCURRED, event::LOOP_UP, event::PORT_DB_CHANGED];

fn image(name: &str, blocks: usize) -> PathBuf {
    let path = std::env::temp_dir().join(format!("iris-isp2200-{}-{name}.img", std::process::id()));
    std::fs::write(&path, vec![0u8; blocks * isp1040::BLOCK_SIZE]).expect("test image");
    path
}

impl Rig {
    /// An adapter with a small disk at each of `ids`.
    pub(super) fn with_disks(name: &str, ids: &[u8]) -> Rig {
        let r = Rig::bare();
        for &id in ids {
            let path = image(&format!("{name}-{id}"), 64);
            r.isp.bus().attach_image(id, &path, false).expect("attach");
            std::fs::remove_file(path).ok();
        }
        r
    }

    /// Move the adapter's clock on and let the ring thread see it.
    pub(super) fn step(&self, ns: i64) {
        self.now.set(self.now.get() + ns);
        self.isp.set_test_clock(self.now.get());
        self.settle();
    }

    pub(super) fn fw_state(&self) -> u16 {
        assert_eq!(self.call(mbox::GET_FW_STATE, 0, 0, 0), mbox::COMPLETE);
        let state = self.rd(mb(1));
        assert_eq!(self.rd(mb(2)), 0);
        self.ack();
        state
    }

    /// The event now posted: its code, and m1..m3 into the array. Then
    /// released.
    pub(super) fn take_event(&self) -> (u16, [u16; 3]) {
        assert_eq!(self.rd(reg::SEMAPHORE) & SEM_LOCK, SEM_LOCK);
        assert_eq!(self.rd(reg::ISR) & ISR_RISC, ISR_RISC);
        let code = self.rd(mb(0));
        let m = [self.rd(mb(1)), self.rd(mb(2)), self.rd(mb(3))];
        self.ack();
        (code, m)
    }

    pub(super) fn expect_no_event(&self) {
        assert_eq!(self.rd(reg::SEMAPHORE), 0);
        assert_eq!(self.rd(reg::ISR), 0);
    }

    /// Exactly these events, in order, with the payloads the design gives.
    pub(super) fn expect_events(&self, codes: &[u16]) {
        for &code in codes {
            let (got, m) = self.take_event();
            assert_eq!(got, code);
            assert_eq!(m, [if code == event::PORT_DB_CHANGED { 0xffff } else { 0 }, 0, 0]);
        }
        self.expect_no_event();
    }

    /// Download, execute, initialise with `options` and `hard` address, then
    /// let the loop come up and take the events a bring-up posts.
    pub(super) fn loop_up(&self, options: u16, hard: u16) {
        self.start();
        self.rings_with(16, 16, options, 0, hard);
        assert_eq!(self.fw_state(), FW_STATE_WAIT_AL_PA);
        self.step(BRINGUP_NS);
        self.expect_events(&BRINGUP_EVENTS[..if options & ICB_OPT_PORT_DB != 0 { 3 } else { 2 }]);
        assert_eq!(self.fw_state(), FW_STATE_READY);
    }

    /// As `rings_at` with the standard places, a hard address in the block.
    pub(super) fn rings_with(&self, rqlen: u16, rslen: u16, options: u16, xopts: u8, hard: u16) {
        self.rqlen.set(rqlen);
        self.rslen.set(rslen);
        self.rq_next.set(0);
        self.rs_next.set(0);
        self.mem.put(RESP_AT, &vec![0u8; usize::from(rslen) * ENTRY_SIZE]);
        let mut icb = icb_build(rqlen, rslen, u64::from(RQST_AT), u64::from(RESP_AT), options);
        icb[64] = xopts;
        icb[20..22].copy_from_slice(&hard.to_le_bytes());
        self.mem.put(ICB_AT, &icb);
        assert_eq!(self.init_firmware(u64::from(ICB_AT)), mbox::COMPLETE);
        self.ack();
    }

    /// A mailbox command with a DMA address in mapping D.
    pub(super) fn call_dma(&self, cmd: u16, m1: u16, addr: u64) -> u16 {
        self.mbox(&[cmd, m1, (addr >> 16) as u16, addr as u16, 0, 0, (addr >> 48) as u16, (addr >> 32) as u16])
    }

    /// Get ID List's own address split: m1 31:16, m2 15:0, m3 63:48, m6 47:32.
    pub(super) fn get_id_list(&self, addr: u64) -> u16 {
        self.mbox(&[mbox::GET_ID_LIST, (addr >> 16) as u16, addr as u16, (addr >> 48) as u16, 0, 0, (addr >> 32) as u16])
    }

    /// The ID list as "alpa:loop" pairs, for a compact comparison.
    pub(super) fn assert_id_list(&self, expect: &str) {
        let buf = 0x8000u32;
        self.mem.put(buf, &[0xa5; 512]);
        assert_eq!(self.get_id_list(u64::from(buf)), mbox::COMPLETE);
        let count = usize::from(self.rd(mb(1)));
        self.ack();
        assert!(count <= 126);
        let list = self.mem.get(buf, 512);
        let mut seen = Vec::new();
        for entry in list.chunks_exact(4).take(count) {
            // AL_PA, area 0, domain 0, loop ID.
            assert_eq!((entry[1], entry[2]), (0, 0));
            seen.push(format!("{:02x}:{}", entry[0], entry[3]));
        }
        // Nothing past the last entry is written.
        assert_eq!(list[count * 4], 0xa5);
        assert_eq!(seen.join(" "), expect);
    }
}

#[test]
fn loop_identity() {
    let r = Rig::with_disks("identity", &[1, 5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);

    // The adapter takes loop ID 0, the lowest free one; loop ID 5 is AL_PA
    // 0xe0 and loop ID 1 is 0xe8 in the standard table.
    assert_eq!(r.call(mbox::GET_ADAPTER_ID, 0, 0, 0), mbox::COMPLETE);
    assert_eq!([r.rd(mb(1)), r.rd(mb(2)), r.rd(mb(3)), r.rd(mb(6)), r.rd(mb(7))], [0, 0x00ef, 0, 0, 0]);
    r.ack();

    // Ascending loop ID, without the adapter.
    r.assert_id_list("e8:1 e0:5");

    // Approximation: the port name list's name is the eight wire-order bytes
    // reversed, which OpenBSD decodes as little-endian to the same name it
    // reads big-endian from the port database.
    r.mem.put(0x9000, &[0xa5; 32]);
    assert_eq!(r.call_dma(mbox::PORT_NAME_LIST, 0x08, 0x9000), mbox::COMPLETE);
    assert_eq!(r.rd(mb(1)), 20);
    r.ack();
    let names = r.mem.get(0x9000, 32);
    assert_eq!(u64::from_le_bytes(names[0..8].try_into().unwrap()), port_wwn(1));
    assert_eq!(g16(&names, 8), 1);
    assert_eq!(u64::from_le_bytes(names[10..18].try_into().unwrap()), port_wwn(5));
    assert_eq!(g16(&names, 18), 5);
    assert_eq!(names[20], 0xa5);
    // Option bits other than 0x02 and 0x08 are a parameter error.
    assert_eq!(r.call_dma(mbox::PORT_NAME_LIST, 0x01, 0x9000), mbox::PARAM_ERROR);
    r.ack();

    // The port database: every byte the design does not name is zero.
    r.mem.put(0x9100, &[0xa5; 129]);
    assert_eq!(r.call_dma(mbox::GET_PORT_DB, 5 << 8, 0x9100), mbox::COMPLETE);
    r.ack();
    let pd = r.mem.get(0x9100, 129);
    let mut want = [0u8; 128];
    want[2] = 6;
    want[3] = 6;
    want[6] = 0xe0;
    want[10] = 0xe0;
    want[12..20].copy_from_slice(&node_wwn(5).to_be_bytes());
    want[20..28].copy_from_slice(&port_wwn(5).to_be_bytes());
    want[78] = 0x10;
    want[80] = 5;
    for i in 0..128 {
        assert_eq!(pd[i], want[i], "port database byte {i}");
    }
    assert_eq!(pd[128], 0xa5);

    // Refused: an empty loop ID, the adapter's own, past 125.
    for id in [6u16, 0, 126] {
        assert_eq!(r.call_dma(mbox::GET_PORT_DB, id << 8, 0x9100), mbox::COMMAND_ERROR);
        r.ack();
    }

    // Approximation: the count, the adapter, then the ports by loop ID.
    r.mem.put(0x9100, &[0xa5; 129]);
    assert_eq!(r.call_dma(mbox::POSITION_MAP, 0, 0x9100), mbox::COMPLETE);
    r.ack();
    let pd = r.mem.get(0x9100, 129);
    assert_eq!(pd[..4], [3, 0xef, 0xe8, 0xe0]);
    assert!(pd[4..128].iter().all(|&b| b == 0));
    assert_eq!(pd[128], 0xa5);
}

/// Approximation: the adapter's loop ID. A requested hard address it can have
/// is taken; one a port holds, or one past 125, gives the lowest free loop ID
/// instead. With nothing requested it is the lowest free one too.
#[test]
fn loop_hard_address() {
    let cases: [(u16, u16, u16, u16); 5] = [
        (ICB_OPT_HARD, 0, 1, 0xe8),
        (ICB_OPT_HARD, 3, 3, 0xe2),
        (ICB_OPT_HARD, 2, 1, 0xe8),
        (ICB_OPT_HARD, 126, 1, 0xe8),
        (0, 3, 1, 0xe8),
    ];
    let r = Rig::with_disks("hard", &[0, 2]);
    for (options, hard, loop_id, alpa) in cases {
        r.loop_up(options, hard);
        assert_eq!(r.call(mbox::GET_ADAPTER_ID, 0, 0, 0), mbox::COMPLETE);
        assert_eq!(r.rd(mb(1)), loop_id);
        assert_eq!(r.rd(mb(2)), alpa);
        r.ack();
        r.assert_id_list("ef:0 e4:2");
        // An HCCR reset drops the firmware and the loop ID with it.
        r.wr(reg::HCCR, HCCR_RESET);
        r.wr(reg::HCCR, HCCR_RELEASE);
    }
}

/// State 0 until initialise-firmware, 1 while the loop comes up, 3 after.
/// Until ready, Get Adapter ID refuses with m1 7, Linux's "loop in
/// transition", and the discovery commands refuse too.
#[test]
fn loop_state_timeline() {
    let r = Rig::bare();
    let buf = 0x9000;
    r.start();
    assert_eq!(r.fw_state(), FW_STATE_CONFIG_WAIT);
    assert_eq!(r.call(mbox::GET_ADAPTER_ID, 0, 0, 0), mbox::COMMAND_ERROR);
    assert_eq!(r.rd(mb(1)), 7);
    r.ack();
    // A LIP before the loop exists is refused and starts nothing.
    assert_eq!(r.call_ack(mbox::INITIATE_LIP, 0, 0, 0), mbox::COMMAND_ERROR);
    assert_eq!(r.call_ack(mbox::FULL_LOGIN_LIP, 0, 0, 0), mbox::COMMAND_ERROR);

    r.rings(8, 8);
    assert_eq!(r.fw_state(), FW_STATE_WAIT_AL_PA);
    assert_eq!(r.call(mbox::GET_ADAPTER_ID, 0, 0, 0), mbox::COMMAND_ERROR);
    assert_eq!(r.rd(mb(1)), 7);
    r.ack();
    assert_eq!(r.get_id_list(buf), mbox::COMMAND_ERROR);
    r.ack();
    assert_eq!(r.call_dma(mbox::PORT_NAME_LIST, 0, buf), mbox::COMMAND_ERROR);
    r.ack();
    assert_eq!(r.call_dma(mbox::POSITION_MAP, 0, buf), mbox::COMMAND_ERROR);
    r.ack();

    // The bring-up time is a model constant; one tick short is not up.
    r.step(BRINGUP_NS - 1);
    assert_eq!(r.fw_state(), FW_STATE_WAIT_AL_PA);
    r.expect_no_event();
    r.step(1);
    r.expect_events(&BRINGUP_EVENTS[..2]);
    assert_eq!(r.fw_state(), FW_STATE_READY);

    // An adapter alone on the loop is ready with an empty list.
    assert_eq!(r.get_id_list(buf), mbox::COMPLETE);
    assert_eq!(r.rd(mb(1)), 0);
    r.ack();

    // A host LIP brings the loop up again.
    assert_eq!(r.call_ack(mbox::FULL_LOGIN_LIP, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(r.fw_state(), FW_STATE_WAIT_AL_PA);
    r.step(BRINGUP_NS);
    r.expect_events(&BRINGUP_EVENTS[..2]);
    assert_eq!(r.fw_state(), FW_STATE_READY);

    // A reset takes the loop back to state 0.
    r.wr(reg::HCCR, HCCR_RESET);
    r.wr(reg::HCCR, HCCR_RELEASE);
    r.start();
    assert_eq!(r.fw_state(), FW_STATE_CONFIG_WAIT);
    r.step(BRINGUP_NS);
    r.expect_no_event();
}

/// Approximation: the connection mode in extended options bits 6:4. Loop
/// only, loop preferred and point-to-point preferred bring the loop up;
/// point-to-point only and the undefined values leave state 0 and post
/// nothing. A reduced-completion mode in bits 3:0 changes nothing.
#[test]
fn loop_connection_mode() {
    let cases: [(u8, bool); 7] =
        [(0x00, true), (0x20, true), (0x30, true), (0x24, true), (0x10, false), (0x40, false), (0x70, false)];
    let r = Rig::bare();
    r.start();
    for (xopts, up) in cases {
        r.rings_with(8, 8, 0, xopts, 0);
        r.step(BRINGUP_NS);
        if up {
            r.expect_events(&BRINGUP_EVENTS[..2]);
            assert_eq!(r.fw_state(), FW_STATE_READY);
        } else {
            r.expect_no_event();
            assert_eq!(r.fw_state(), FW_STATE_CONFIG_WAIT);
        }
    }
}

/// Approximation: the port-database event follows loop up only with
/// control-block option 0x0100. A new trigger drops every event an earlier
/// bring-up queued and has not posted; one already posted stays until the
/// host releases it, and a mailbox result waits behind it intact.
#[test]
fn loop_events() {
    let r = Rig::bare();
    r.loop_up(0, 0);

    // Let the next bring-up post its first event and leave it unread.
    assert_eq!(r.call_ack(mbox::INITIATE_LIP, 0, 0, 0), mbox::COMPLETE);
    r.step(BRINGUP_NS);
    assert_eq!(r.rd(mb(0)), event::LIP_OCCURRED);

    // The host reads the event, then runs a second trigger before it
    // releases it; the result waits in its slot until the release.
    assert_eq!(r.rd(mb(1)), 0);
    r.call(mbox::INITIATE_LIP, 0, 0, 0);
    assert_eq!(r.rd(reg::HCCR) & HCCR_HOST_INT, 0);
    r.ack();
    // Its result, then nothing: loop up from the first one is gone.
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    r.ack();
    r.expect_no_event();
    assert_eq!(r.fw_state(), FW_STATE_WAIT_AL_PA);

    // The second bring-up posts its own whole set.
    r.step(BRINGUP_NS);
    r.expect_events(&BRINGUP_EVENTS[..2]);

    // An initialisation with option 0x0100 set adds the third event.
    r.rings_with(8, 8, ICB_OPT_PORT_DB, 0, 0);
    r.step(BRINGUP_NS);
    r.expect_events(&BRINGUP_EVENTS);
}

/// With the loop left down by the connection mode, firmware state 0 as for a
/// disconnected adapter, a command to a present port completes "port
/// unavailable" without reaching the disk, and the LIP commands refuse. A
/// later initialisation in loop mode brings the port back.
#[test]
fn loop_down_commands() {
    let r = Rig::with_disks("down", &[5]);
    r.start();
    r.rings_with(8, 8, ICB_OPT_PORT_DB, 0x10, 0);
    r.step(BRINGUP_NS);
    r.expect_no_event();
    assert_eq!(r.fw_state(), FW_STATE_CONFIG_WAIT);
    for cmd in [mbox::INITIATE_LIP, mbox::FULL_LOGIN_LIP] {
        assert_eq!(r.call_ack(cmd, 0, 0, 0), mbox::COMMAND_ERROR);
    }
    assert_eq!(r.call_ack(mbox::LIP_RESET, 0xff00, 0, 0), mbox::COMMAND_ERROR);

    r.publish(&entry_command(ENTRY_CMD64, 0xdd00, 5, 0, 0));
    let e = r.response(0);
    assert_eq!((g16(&e, 8), g16(&e, 10)), (0, COMPLETION_PORT_UNAVAILABLE));
    r.consume(1);

    r.rings_with(8, 8, ICB_OPT_PORT_DB, 0, 0);
    r.step(BRINGUP_NS);
    r.expect_events(&BRINGUP_EVENTS);
    r.publish(&entry_command(ENTRY_CMD64, 0xdd01, 5, 0, 0));
    let e = r.response(0);
    assert_eq!(g16(&e, 10), IOCB_COMPLETE);
}

/// Approximation: a discovery command whose result DMA the host bridge
/// refuses answers 0x4002, as the host interface's other refused DMAs do.
#[test]
fn loop_dma_refused() {
    let r = Rig::with_disks("refused", &[5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    let beyond = 1u64 << 40;
    assert_eq!(r.get_id_list(beyond), mbox::HOST_INTERFACE_ERROR);
    r.ack();
    assert_eq!(r.call_dma(mbox::PORT_NAME_LIST, 0x08, beyond), mbox::HOST_INTERFACE_ERROR);
    r.ack();
    assert_eq!(r.call_dma(mbox::GET_PORT_DB, 5 << 8, beyond), mbox::HOST_INTERFACE_ERROR);
    r.ack();
    assert_eq!(r.call_dma(mbox::POSITION_MAP, 0, beyond), mbox::HOST_INTERFACE_ERROR);
    r.ack();
}

/// Approximation: Linux asks for reduced-interrupt mode 4. It is accepted and
/// ignored: completions are ordinary 0x03 status entries.
#[test]
fn rio_mode_ignored() {
    let r = Rig::bare();
    r.start();
    r.rings_with(8, 8, 0, 4, 0);
    r.tur(0x40);
    r.tur(0x41);
}

/// OpenBSD writes the port name list's inputs without mailbox 7, so the
/// address takes bits 47:32 from what the host last wrote there. Here that is
/// 1, from an earlier command: the list lands above 4 GiB, which the bridge
/// has no memory for, and the command is refused.
#[test]
fn port_name_list_retains_m7() {
    let r = Rig::with_disks("m7", &[1, 5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    // A No-op that writes mailbox 7 as 1.
    for i in 0..8 {
        r.wr(mb(i), if i == 7 { 1 } else { 0 });
    }
    r.wr(reg::HCCR, HCCR_SET_HOST);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    r.ack();
    let input = [mbox::PORT_NAME_LIST, 0x08, 0, 0x9000, 0, 0, 0];
    assert_eq!(r.mbox(&input), mbox::HOST_INTERFACE_ERROR);
    r.ack();
    // With mailbox 7 written as zero it lands in memory.
    r.wr(mb(7), 0);
    assert_eq!(r.mbox(&input), mbox::COMPLETE);
    assert_eq!(r.rd(mb(1)), 20);
    r.ack();
}
