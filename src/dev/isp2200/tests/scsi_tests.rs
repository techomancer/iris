//! Commands to the disks on the loop, and the recovery commands that cancel
//! them.
//!
//! Ported from tests/qtest/qlogic-isp2200-test.c in
//! jamesbraid/qemu-sgi-origami.

use super::loop_tests::BRINGUP_NS;
use super::*;
use crate::scsi::{ScsiResponse, ScsiTarget};
use parking_lot::Condvar;
use std::io;
use std::sync::atomic::AtomicU32;
use std::time::{Duration, Instant};

const COMPLETION_COMPLETE: u16 = IOCB_COMPLETE;
const COMPLETION_UNDERRUN: u16 = IOCB_DATA_UNDERRUN;
const COMPLETION_ABORTED: u16 = IOCB_ABORTED;
const COMPLETION_RESET: u16 = IOCB_RESET;
const SCSI_CHECK_CONDITION: u16 = 0x02;
const ICB_OPT_PORT_DB: u16 = ICB_OPT_PORT_DB_EVENTS;
const BUF: u32 = 0x10000;

#[derive(Debug, Default)]
struct ScsiResult {
    scsi_status: u16,
    completion: u16,
    sense_len: u16,
    residual: u32,
    sense: Vec<u8>,
}

fn decode_status(e: &Entry, handle: u32) -> ScsiResult {
    assert_eq!(g32(e, 4), handle);
    // State and status flags: unknown meaning, written as zero.
    assert_eq!((g16(e, 12), g16(e, 14)), (0, 0));
    ScsiResult {
        scsi_status: g16(e, 8),
        completion: g16(e, 10),
        sense_len: g16(e, 18),
        residual: g32(e, 20),
        sense: e[32..64].to_vec(),
    }
}

const CDB_TUR: [u8; 6] = [0; 6];

fn cdb_inquiry(evpd: bool, page: u8, alloc_len: u16) -> Vec<u8> {
    let mut cdb = vec![0u8; 6];
    cdb[0] = 0x12;
    cdb[1] = u8::from(evpd);
    cdb[2] = page;
    cdb[3..5].copy_from_slice(&alloc_len.to_be_bytes());
    cdb
}

fn cdb_rw10(op: u8, lba: u32, blocks: u16) -> Vec<u8> {
    let mut cdb = vec![0u8; 10];
    cdb[0] = op;
    cdb[2..6].copy_from_slice(&lba.to_be_bytes());
    cdb[7..9].copy_from_slice(&blocks.to_be_bytes());
    cdb
}

impl Rig {
    /// A 64-bit command with at most two segments. `control` is the direction
    /// flag; the segments split `len` at `split`.
    fn scsi_entry(&self, handle: u32, loop_id: u8, lun: u16, cdb: &[u8], control: u16, buf: u64, len: u32, split: u32) -> Entry {
        let mut e = entry_command(ENTRY_CMD64, handle, loop_id, lun, cdb[0]);
        e[20..20 + cdb.len()].copy_from_slice(cdb);
        p16(&mut e, 12, control);
        p32(&mut e, 36, len);
        if len != 0 && split != 0 && split < len {
            p16(&mut e, 18, 2);
            put_seg64(&mut e, 40, buf, split);
            put_seg64(&mut e, 52, buf + u64::from(split), len - split);
        } else if len != 0 {
            p16(&mut e, 18, 1);
            put_seg64(&mut e, 40, buf, len);
        }
        e
    }

    /// One status entry: the adapter's thread has run the command by the time
    /// a register write returns, so the entry is there to read.
    fn wait_response(&self) -> Entry {
        assert_eq!(self.rd(reg::ISR) & ISR_RISC, ISR_RISC);
        assert_eq!(self.rd(reg::SEMAPHORE), 0);
        assert_eq!(self.rd(reg::RESPONSE_QUEUE), (self.rs_next.get() + 1) % self.rslen.get());
        let e = self.response(self.rs_next.get());
        assert_eq!(e[0], ENTRY_STATUS);
        self.consume(1);
        e
    }

    #[allow(clippy::too_many_arguments)]
    fn scsi(&self, handle: u32, loop_id: u8, lun: u16, cdb: &[u8], control: u16, buf: u64, len: u32, split: u32) -> ScsiResult {
        self.publish(&self.scsi_entry(handle, loop_id, lun, cdb, control, buf, len, split));
        decode_status(&self.wait_response(), handle)
    }

    fn tur_to(&self, handle: u32, loop_id: u8, lun: u16) -> ScsiResult {
        self.scsi(handle, loop_id, lun, &CDB_TUR, 0, 0, 0, 0)
    }

    /// The sense key a TEST UNIT READY draws, or 0 for good.
    fn tur_sense(&self, loop_id: u8, lun: u16) -> u8 {
        let r = self.tur_to(0x7e00, loop_id, lun);
        assert_eq!(r.completion, COMPLETION_COMPLETE);
        if r.scsi_status == 0 {
            return 0;
        }
        assert_eq!(r.scsi_status, SCSI_CHECK_CONDITION | SCSI_STATUS_SENSE_VALID);
        r.sense[2] & 0x0f
    }
}

/// INQUIRY reaches the disk at its loop ID, and a LUN the port lacks is
/// refused as the SCSI layer below refuses it.
#[test]
fn scsi_inquiry() {
    let r = Rig::with_disks("inquiry", &[5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    let res = r.scsi(0x1201, 5, 0, &cdb_inquiry(false, 0, 36), 0x20, u64::from(BUF), 36, 0);
    assert_eq!((res.completion, res.scsi_status, res.residual, res.sense_len), (COMPLETION_COMPLETE, 0, 0, 0));
    let data = r.mem.get(BUF, 36);
    assert_eq!(data[0], 0x00);
    assert_eq!(&data[8..16], b"SGI     ");
}

/// WRITE(10) then READ(10) of the same eight blocks, with the segments split
/// differently each way: the data goes to the disk and comes back the same.
#[test]
fn scsi_read_write() {
    let r = Rig::with_disks("rw", &[5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    let len = 8 * 512;
    let (out, back) = (0x10000u32, 0x20000u32);
    let pattern: Vec<u8> = (0..len).map(|i| (i * 7 + (i >> 9) + 1) as u8).collect();
    r.mem.put(out, &pattern);

    let res = r.scsi(0x2a2a, 5, 0, &cdb_rw10(0x2a, 16, 8), 0x40, u64::from(out), len as u32, 1536);
    assert_eq!((res.completion, res.scsi_status, res.residual), (COMPLETION_COMPLETE, 0, 0));

    // Read back through OpenBSD's segment-list form, three pieces.
    let mut list = [0u8; 36];
    put_seg64(&mut list, 0, u64::from(back), 1024);
    put_seg64(&mut list, 12, u64::from(back) + 1024, 1024);
    put_seg64(&mut list, 24, u64::from(back) + 2048, 2048);
    r.mem.put(0x6000, &list);
    let mut e = r.scsi_entry(0x2828, 5, 0, &cdb_rw10(0x28, 16, 8), 0x20, 0, len as u32, 0);
    e[0] = ENTRY_CMD_LIST;
    p16(&mut e, 18, 3);
    p16(&mut e, 40, 1);
    p64(&mut e, 46, 0x6000);
    r.publish(&e);
    let res = decode_status(&r.wait_response(), 0x2828);
    assert_eq!((res.completion, res.scsi_status), (COMPLETION_COMPLETE, 0));
    assert_eq!(r.mem.get(back, len), pattern);
}

/// Approximation: the expanded-LUN layout -- target in the high byte of the
/// field at 8, LUN at 10 -- reaches the disk at that loop ID, and the LUN
/// reaches it in the CDB, where a disk with only LUN 0 refuses it.
#[test]
fn expanded_lun() {
    let r = Rig::with_disks("expanded-lun", &[7]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    let mut e = entry_command(ENTRY_CMD64, 0x46, 0, 0, 0x00);
    p16(&mut e, 8, 7 << 8);
    p16(&mut e, 10, 3);
    r.publish(&e);
    let status = r.wait_response();
    assert_eq!(g32(&status, 4), 0x46);
    assert_eq!(g16(&status, 10), IOCB_COMPLETE, "the port at loop ID 7 took it");
    assert_eq!(g16(&status, 8) & 0xff, 0x02, "and answered CHECK CONDITION for the LUN");
}

/// Segments reach the disk whole and in order from every place a command
/// keeps them: inline, in a continuation entry, and in a list in host memory.
/// A WRITE BUFFER carries the pieces to the target, and a READ BUFFER into
/// one buffer brings the target's copy back.
#[test]
fn segments_from_every_entry_reach_the_disk_in_order() {
    let r = Rig::with_disks("segments", &[3]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    let cdb = |op: u8, len: u8| vec![op, 0x02, 0, 0, 0, 0, 0, 0, len, 0];
    let (write, read) = (0x3b, 0x3c);
    let sent: Vec<u8> = (1..=15).collect();
    let spots = [0x20000u32, 0x20100, 0x20200, 0x20300];
    let pieces = [1usize, 2, 4, 8];
    let mut at = 0;
    for (spot, len) in spots.iter().zip(pieces) {
        r.mem.put(*spot, &sent[at..at + len]);
        at += len;
    }

    // 32-bit: three inline, one in a 0x02 continuation from byte 8.
    let mut e = r.scsi_entry(0x3201, 3, 0, &cdb(write, 15), 0x40, 0, 15, 0);
    e[0] = ENTRY_CMD32;
    e[1] = 2;
    p16(&mut e, 18, 4);
    for (i, (spot, len)) in spots.iter().zip(pieces).take(3).enumerate() {
        put_seg32(&mut e, 40 + i * 8, *spot, len as u32);
    }
    let mut c = [0u8; ENTRY_SIZE];
    c[0] = ENTRY_CONT32;
    c[1] = 1;
    put_seg32(&mut c, 8, spots[3], pieces[3] as u32);
    r.put_request(0, &e);
    r.put_request(1, &c);
    r.rq_next.set(2);
    r.wr(reg::REQUEST_QUEUE, 2);
    let res = decode_status(&r.wait_response(), 0x3201);
    assert_eq!((res.completion, res.scsi_status), (COMPLETION_COMPLETE, 0));
    let res = r.scsi(0x3202, 3, 0, &cdb(read, 15), 0x20, u64::from(BUF), 15, 0);
    assert_eq!((res.completion, res.scsi_status), (COMPLETION_COMPLETE, 0));
    assert_eq!(r.mem.get(BUF, 15), sent);

    // 64-bit: two inline, two in a 0x0a continuation from byte 4.
    r.mem.put(BUF, &[0; 16]);
    let mut e = r.scsi_entry(0x6401, 3, 0, &cdb(write, 15), 0x40, 0, 15, 0);
    e[1] = 2;
    p16(&mut e, 18, 4);
    put_seg64(&mut e, 40, u64::from(spots[0]), pieces[0] as u32);
    put_seg64(&mut e, 52, u64::from(spots[1]), pieces[1] as u32);
    let mut c = [0u8; ENTRY_SIZE];
    c[0] = ENTRY_CONT64;
    c[1] = 1;
    put_seg64(&mut c, 4, u64::from(spots[2]), pieces[2] as u32);
    put_seg64(&mut c, 16, u64::from(spots[3]), pieces[3] as u32);
    let slot = r.rq_next.get();
    r.put_request(slot, &e);
    r.put_request(slot + 1, &c);
    r.rq_next.set(slot + 2);
    r.wr(reg::REQUEST_QUEUE, slot + 2);
    let res = decode_status(&r.wait_response(), 0x6401);
    assert_eq!((res.completion, res.scsi_status), (COMPLETION_COMPLETE, 0));
    let res = r.scsi(0x6402, 3, 0, &cdb(read, 15), 0x20, u64::from(BUF), 15, 0);
    assert_eq!((res.completion, res.scsi_status), (COMPLETION_COMPLETE, 0));
    assert_eq!(r.mem.get(BUF, 15), sent);

    // A list in host memory, as OpenBSD sends it. The bytes are new, so a
    // list read in another order, or with the right total, leaves the
    // target holding something else than the first writes left there.
    let resent: Vec<u8> = (101..=115).collect();
    let mut at = 0;
    for (spot, len) in spots.iter().zip(pieces) {
        r.mem.put(*spot, &resent[at..at + len]);
        at += len;
    }
    let mut list = [0u8; 48];
    for (i, (spot, len)) in spots.iter().zip(pieces).enumerate() {
        put_seg64(&mut list, i * 12, u64::from(*spot), len as u32);
    }
    r.mem.put(0x6000, &list);
    let mut e = r.scsi_entry(0x1501, 3, 0, &cdb(write, 15), 0x40, 0, 15, 0);
    e[0] = ENTRY_CMD_LIST;
    p16(&mut e, 18, 4);
    p16(&mut e, 40, 1);
    p64(&mut e, 46, 0x6000);
    r.publish(&e);
    let res = decode_status(&r.wait_response(), 0x1501);
    assert_eq!((res.completion, res.scsi_status), (COMPLETION_COMPLETE, 0));
    r.mem.put(BUF, &[0; 16]);
    let res = r.scsi(0x1502, 3, 0, &cdb(read, 15), 0x20, u64::from(BUF), 15, 0);
    assert_eq!((res.completion, res.scsi_status), (COMPLETION_COMPLETE, 0));
    assert_eq!(r.mem.get(BUF, 15), resent, "the list's segments, in the list's order");
}

/// Commands go to the loop ID they name: the disks tell themselves apart by
/// what each holds.
#[test]
fn scsi_addressing() {
    let r = Rig::with_disks("addressing", &[5, 1]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    r.mem.put(BUF, &[0x11; 512]);
    let res = r.scsi(0x8000, 5, 0, &cdb_rw10(0x2a, 3, 1), 0x40, u64::from(BUF), 512, 0);
    assert_eq!(res.scsi_status, 0);
    r.mem.put(BUF + 0x1000, &[0; 512]);
    r.scsi(0x8001, 5, 0, &cdb_rw10(0x28, 3, 1), 0x20, u64::from(BUF + 0x1000), 512, 0);
    assert_eq!(r.mem.get(BUF + 0x1000, 512), vec![0x11; 512]);
    r.scsi(0x8002, 1, 0, &cdb_rw10(0x28, 3, 1), 0x20, u64::from(BUF + 0x1000), 512, 0);
    assert_eq!(r.mem.get(BUF + 0x1000, 512), vec![0; 512]);
}

/// Approximation: completion precedence. A failed SCSI status completes
/// 0x0000 with the status, sense valid and the sense bytes, and with
/// residual-under and the residual when the transfer was short. A good short
/// transfer is underrun 0x0015 with the residual; a good full one is 0x0000
/// with no flags.
#[test]
fn scsi_check_condition() {
    let r = Rig::with_disks("cc", &[1]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    let buf = u64::from(BUF);

    // A READ to a LUN the port lacks moves nothing: all 512 are residual.
    let res = r.scsi(0x0603, 1, 7, &cdb_rw10(0x28, 0, 1), 0x20, buf, 512, 0);
    assert_eq!(res.completion, COMPLETION_COMPLETE);
    assert_eq!(res.scsi_status, SCSI_CHECK_CONDITION | SCSI_STATUS_SENSE_VALID | SCSI_STATUS_RESID_UNDER);
    assert_eq!((res.sense[2] & 0x0f, res.sense[12]), (0x05, 0x25));
    assert_eq!(res.residual, 512);

    // INQUIRY to that LUN: peripheral qualifier 3, no device.
    r.mem.put(BUF, &[0; 36]);
    let res = r.scsi(0x0604, 1, 7, &cdb_inquiry(false, 0, 36), 0x20, buf, 36, 0);
    assert_eq!(res.scsi_status & 0xff, 0);
    assert_eq!(r.mem.get(BUF, 1)[0] & 0xe0, 0x60);

    // A LUN from 8 to 255 does not fit the CDB's three bits, and is a LUN the
    // disk lacks like LUN 7: it never reads LUN 0's blocks.
    for lun in [8, 200, 255] {
        let before = r.isp.total_commands();
        r.mem.put(BUF, &[0x77; 512]);
        let res = r.scsi(0x0610 + u32::from(lun), 1, lun, &cdb_rw10(0x28, 0, 1), 0x20, buf, 512, 0);
        assert_eq!(res.scsi_status, SCSI_CHECK_CONDITION | SCSI_STATUS_SENSE_VALID | SCSI_STATUS_RESID_UNDER, "LUN {lun}");
        assert_eq!((res.sense[2] & 0x0f, res.sense[12], res.residual), (0x05, 0x25, 512), "LUN {lun}");
        assert_eq!(r.mem.get(BUF, 512), vec![0x77; 512], "LUN {lun}: nothing moved");
        assert_eq!(r.isp.total_commands(), before + 1, "LUN {lun} reached the disk");

        r.mem.put(BUF, &[0; 36]);
        let res = r.scsi(0x0710 + u32::from(lun), 1, lun, &cdb_inquiry(false, 0, 36), 0x20, buf, 36, 0);
        assert_eq!(res.scsi_status & 0xff, 0, "LUN {lun}");
        assert_eq!(r.mem.get(BUF, 1)[0] & 0xe0, 0x60, "LUN {lun}: peripheral qualifier 3");
    }

    // A LUN past 255: CHECK CONDITION, the same sense, nothing moved and the
    // disk never asked.
    let before = r.isp.total_commands();
    let res = r.scsi(0x0605, 1, 300, &cdb_rw10(0x28, 0, 1), 0x20, buf, 512, 0);
    assert_eq!(res.scsi_status, SCSI_CHECK_CONDITION | SCSI_STATUS_SENSE_VALID | SCSI_STATUS_RESID_UNDER);
    assert_eq!((res.sense[2] & 0x0f, res.sense[12], res.residual), (0x05, 0x25, 512));
    assert_eq!(r.isp.total_commands(), before, "it did not reach the disk");

    // A good transfer shorter than the host's buffer is an underrun.
    let res = r.scsi(0x0606, 1, 0, &cdb_inquiry(false, 0, 36), 0x20, buf, 255, 0);
    assert_eq!(res.completion, COMPLETION_UNDERRUN);
    assert_eq!(res.scsi_status, SCSI_STATUS_RESID_UNDER);
    assert_eq!((res.residual, res.sense_len), (255 - 36, 0));

    // So is one block read into two blocks' worth of segments.
    let res = r.scsi(0x0608, 1, 0, &cdb_rw10(0x28, 0, 1), 0x20, buf, 1024, 512);
    assert_eq!((res.completion, res.scsi_status, res.residual), (COMPLETION_UNDERRUN, SCSI_STATUS_RESID_UNDER, 512));

    // And a good full READ has no flags at all.
    let res = r.scsi(0x0609, 1, 0, &cdb_rw10(0x28, 0, 1), 0x20, buf, 512, 256);
    assert_eq!((res.completion, res.scsi_status, res.residual), (COMPLETION_COMPLETE, 0, 0));
}

/// A target that records the CDB and the data each command brings, and
/// answers GOOD with `reply` as its data.
struct Recorder {
    seen: Arc<Mutex<Vec<(Vec<u8>, Option<Vec<u8>>)>>>,
    reply: Vec<u8>,
}

impl ScsiTarget for Recorder {
    fn request(&mut self, req: &ScsiRequest) -> io::Result<ScsiResponse> {
        self.seen.lock().push((req.cdb.clone(), req.data_in.clone()));
        Ok(ScsiResponse { status: 0, data: self.reply.clone() })
    }
}

/// A 12- and a 16-byte CDB reach the target with every byte, the LUN bits
/// of byte 1 aside, and an opcode the adapter has no transfer rule for moves
/// what the entry's direction flag and byte count say.
#[test]
fn long_cdbs_and_unknown_opcodes_reach_the_target_whole() {
    let r = Rig::bare();
    let seen = Arc::new(Mutex::new(Vec::new()));
    r.isp.bus().attach(6, Box::new(Recorder { seen: seen.clone(), reply: (1..=64).collect() })).unwrap();
    r.loop_up(ICB_OPT_PORT_DB, 0);
    let buf = u64::from(BUF);

    // READ(12), then READ(16): bytes 10 to 15 differ from each other, so a
    // CDB cut at ten bytes shows.
    let cdb12: Vec<u8> = [0xa8u8, 0x05, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0xaa, 0xbb].to_vec();
    let cdb16: Vec<u8> = [0x88u8, 0x05, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0xdd, 0xee].to_vec();
    for (handle, cdb) in [(0x1201u32, &cdb12), (0x1601, &cdb16)] {
        r.mem.put(BUF, &[0xee; 16]);
        let res = r.scsi(handle, 6, 0, cdb, 0x20, buf, 8, 0);
        assert_eq!((res.completion, res.scsi_status, res.residual), (COMPLETION_COMPLETE, 0, 0));
        assert_eq!(seen.lock().last().unwrap().0, *cdb, "all {} bytes", cdb.len());
        assert_eq!(r.mem.get(BUF, 16), [1, 2, 3, 4, 5, 6, 7, 8, 0xee, 0xee, 0xee, 0xee, 0xee, 0xee, 0xee, 0xee], "eight bytes, as the entry says");
    }

    // A vendor opcode, data out: the target gets the bytes the host named.
    r.mem.put(BUF, &[9, 8, 7, 6, 5, 4]);
    let res = r.scsi(0xc001, 6, 0, &[0xc0, 0, 0, 0, 0, 0], 0x40, buf, 5, 0);
    assert_eq!((res.completion, res.scsi_status, res.residual), (COMPLETION_COMPLETE, 0, 0));
    let (cdb, data) = seen.lock().last().cloned().unwrap();
    assert_eq!(cdb, [0xc0, 0, 0, 0, 0, 0]);
    assert_eq!(data.as_deref(), Some(&[9u8, 8, 7, 6, 5][..]));

    // The same opcode with no direction flag moves nothing.
    r.mem.put(BUF, &[0xee; 4]);
    let res = r.scsi(0xc002, 6, 0, &[0xc0, 0, 0, 0, 0, 0], 0, buf, 4, 0);
    assert_eq!((res.completion, res.scsi_status), (COMPLETION_UNDERRUN, SCSI_STATUS_RESID_UNDER));
    assert_eq!(r.mem.get(BUF, 4), [0xee; 4]);
    assert_eq!(r.isp.total_commands(), 4);
}

/// An entry the adapter rejects, or cannot read, never reaches a disk; one
/// whose response cannot be written still runs.
#[test]
fn rejected_entries_stay_off_the_disk_and_a_command_runs_without_its_response() {
    let r = Rig::with_disks("rejects", &[5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    let tur = |handle| entry_command(ENTRY_CMD64, handle, 5, 0, 0x00);

    let mut e = tur(0xbad1);
    e[1] = 0;
    r.publish(&e);
    r.assert_rejected(ENTRY_CMD64, 0xbad1, ENTRY_FLAG_BAD_COUNT);
    let mut e = tur(0xbad2);
    e[0] = 0x7f;
    r.publish(&e);
    r.assert_rejected(0x7f, 0xbad2, ENTRY_FLAG_BAD_TYPE);
    let mut e = tur(0xbad3);
    e[0] = ENTRY_CMD_LIST;
    p16(&mut e, 18, 2);
    p16(&mut e, 40, 1);
    p64(&mut e, 46, 1 << 40);
    r.publish(&e);
    r.assert_rejected(ENTRY_CMD_LIST, 0xbad3, ENTRY_FLAG_DMA_ERROR);
    assert_eq!(r.isp.total_commands(), 0, "none of them reached the disk");
    r.publish(&tur(0x600d));
    assert_eq!(decode_status(&r.wait_response(), 0x600d).completion, COMPLETION_COMPLETE);
    assert_eq!(r.isp.total_commands(), 1);

    // The request ring cannot be read: nothing is consumed, nothing runs.
    let r = Rig::with_disks("unreadable", &[5]);
    r.start();
    r.rings_at(16, 16, 1 << 40, u64::from(RESP_AT), ICB_OPT_PORT_DB, 0);
    r.step(BRINGUP_NS);
    r.expect_events(&[event::LIP_OCCURRED, event::LOOP_UP, event::PORT_DB_CHANGED]);
    r.wr(reg::REQUEST_QUEUE, 1);
    r.assert_event(event::REQUEST_DMA_ERROR);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 0);
    assert_eq!(r.isp.total_commands(), 0);

    // The response ring cannot be written: the command ran, the event says
    // the entry was lost.
    let r = Rig::with_disks("unwritable", &[5]);
    r.start();
    r.rings_at(16, 16, u64::from(RQST_AT), 1 << 40, ICB_OPT_PORT_DB, 0);
    r.step(BRINGUP_NS);
    r.expect_events(&[event::LIP_OCCURRED, event::LOOP_UP, event::PORT_DB_CHANGED]);
    let block = [0x6bu8; 512];
    r.mem.put(BUF, &block);
    r.publish(&r.scsi_entry(0xd0d0, 5, 0, &cdb_rw10(0x2a, 2, 1), 0x40, u64::from(BUF), 512, 0));
    r.assert_event(event::RESPONSE_DMA_ERROR);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 0);
    assert_eq!(r.isp.total_commands(), 1, "the command ran");
    let stored = r.isp.bus().lock()[5].as_mut().unwrap().request(&ScsiRequest {
        cdb: cdb_rw10(0x28, 2, 1),
        data_len: ScsiDataLength::Unlimited,
        data_in: None,
    });
    assert_eq!(stored.unwrap().data, block, "and the disk holds what it wrote");
}

/// Approximation: a command to a loop ID with no port, to the adapter's own
/// loop ID or past 125 completes "port unavailable" with no SCSI status. A
/// marker between commands answers nothing.
#[test]
fn scsi_unavailable() {
    let r = Rig::with_disks("unavailable", &[1, 5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    for (i, id) in [2u8, 0, 126, 255].into_iter().enumerate() {
        let res = r.tur_to(0x2800 + i as u32, id, 0);
        assert_eq!((res.completion, res.scsi_status, res.residual), (COMPLETION_PORT_UNAVAILABLE, 0, 0));

        let mut e = [0u8; ENTRY_SIZE];
        e[0] = ENTRY_MARKER;
        e[1] = 1;
        e[9] = id;
        e[10] = 1;
        r.publish(&e);
        assert_eq!(r.rd(reg::REQUEST_QUEUE), r.rq_next.get());
        r.expect_no_event();
    }
}

/// The first command each disk takes after a soft reset reports the reset as
/// a unit attention, once. A bus reset the guest did not ask for does not.
#[test]
fn scsi_unit_attention_after_soft_reset() {
    let r = Rig::with_disks("ua", &[1]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    assert_eq!(r.tur_sense(1, 0), 0);
    r.soft_reset();
    r.wr(reg::HCCR, HCCR_RELEASE);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    assert_eq!(r.tur_sense(1, 0), 0x06);
    assert_eq!(r.tur_sense(1, 0), 0);
}

/// A device that holds a READ until the test lets it go, counting how many
/// it has taken and finished.
struct Gate {
    open: Mutex<bool>,
    wake: Condvar,
    started: AtomicU32,
    finished: AtomicU32,
}

impl Gate {
    fn new() -> Arc<Gate> {
        Arc::new(Gate { open: Mutex::new(false), wake: Condvar::new(), started: AtomicU32::new(0), finished: AtomicU32::new(0) })
    }

    fn release(&self) {
        *self.open.lock() = true;
        self.wake.notify_all();
    }

    fn close(&self) {
        *self.open.lock() = false;
    }

    fn wait_for(&self, counter: &AtomicU32, n: u32) {
        let end = Instant::now() + Duration::from_secs(10);
        while counter.load(Ordering::Acquire) < n {
            assert!(Instant::now() < end, "the device did not get there");
            std::thread::yield_now();
        }
    }
}

struct SlowDisk {
    gate: Arc<Gate>,
    attention: bool,
}

impl ScsiTarget for SlowDisk {
    fn request(&mut self, req: &ScsiRequest) -> io::Result<ScsiResponse> {
        if req.cdb[0] == scsi_cmd::REQUEST_SENSE {
            let mut sense = vec![0u8; 18];
            sense[0] = 0x70;
            sense[2] = 0x06;
            sense[12] = 0x29;
            return Ok(ScsiResponse { status: 0, data: sense });
        }
        if std::mem::take(&mut self.attention) {
            return Ok(ScsiResponse { status: CHECK_CONDITION, data: Vec::new() });
        }
        if req.cdb[0] != scsi_cmd::READ_10 {
            return Ok(ScsiResponse { status: 0, data: Vec::new() });
        }
        self.gate.started.fetch_add(1, Ordering::AcqRel);
        let mut open = self.gate.open.lock();
        while !*open {
            self.gate.wake.wait(&mut open);
        }
        drop(open);
        self.gate.finished.fetch_add(1, Ordering::AcqRel);
        Ok(ScsiResponse { status: 0, data: vec![0x5a; 512] })
    }

    fn bus_reset(&mut self) {
        self.attention = true;
    }
}

impl Rig {
    /// An adapter with a slow disk at each of `ids`, behind one gate.
    fn with_slow_disks(ids: &[u8]) -> (Rig, Arc<Gate>) {
        let r = Rig::bare();
        let gate = Gate::new();
        for &id in ids {
            r.isp.bus().attach(id, Box::new(SlowDisk { gate: gate.clone(), attention: false })).unwrap();
        }
        (r, gate)
    }

    /// Publish a READ(10) and wait until the slow disk has it. The adapter's
    /// thread is then busy until the gate opens.
    fn submit_read(&self, gate: &Gate, handle: u32, loop_id: u8, lun: u16, buf: u32) {
        gate.close();
        let before = gate.started.load(Ordering::Acquire);
        let e = self.scsi_entry(handle, loop_id, lun, &cdb_rw10(0x28, 0, 1), 0x20, u64::from(buf), 512, 0);
        self.put_request(self.rq_next.get(), &e);
        self.rq_next.set((self.rq_next.get() + 1) % self.rqlen.get());
        self.isp.write(reg::REQUEST_QUEUE, u32::from(self.rq_next.get()), 2);
        gate.wait_for(&gate.started, before + 1);
        self.busy.set(true);
    }

    /// Open the gate and wait for the ring thread to finish what it held.
    fn release(&self, gate: &Gate, finished: u32) {
        gate.release();
        gate.wait_for(&gate.finished, finished);
        self.busy.set(false);
        self.isp.wait_idle();
    }

    fn abort(&self, loop_id: u8, lun: u16, handle: u32) -> u16 {
        self.mbox(&[mbox::ABORT_COMMAND, u16::from(loop_id) << 8, handle as u16, (handle >> 16) as u16, 0, 0, lun])
    }

    /// The posted mailbox result's code, released.
    fn take_result(&self) -> u16 {
        let code = self.rd(mb(0));
        self.ack();
        code
    }

    /// Exactly these entries, handle and completion, in order, as a driver
    /// walking the ring reads them.
    fn collect(&self, want: &[(u32, u16)]) {
        let mut seen = 0;
        while seen < want.len() {
            assert_eq!(self.rd(reg::ISR) & ISR_RISC, ISR_RISC, "entry {seen} of {want:?}");
            let (rslen, next) = (self.rslen.get(), self.rs_next.get());
            let n = (self.rd(reg::RESPONSE_QUEUE) + rslen - next) % rslen;
            assert!(n > 0 && seen + usize::from(n) <= want.len(), "{n} entries after {seen} of {want:?}");
            for i in 0..n {
                let e = self.response((next + i) % rslen);
                assert_eq!((g32(&e, 4), g16(&e, 10)), want[seen + usize::from(i)]);
            }
            self.consume(n);
            seen += usize::from(n);
        }
        assert_eq!(self.rd(reg::ISR), 0);
    }

    fn assert_untouched(&self, addr: u32, len: usize, fill: u8) {
        assert_eq!(self.mem.get(addr, len), vec![fill; len]);
    }
}

/// Retiring a command in flight: the mailbox result posts first, then the
/// command's own entry, with the completion the recovery owes.
#[test]
fn abort_command() {
    let (r, gate) = Rig::with_slow_disks(&[1, 5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    r.mem.put(BUF, &[0xa5; 1024]);
    // Entry bytes 00 00 00 05, then 05 00 00 00.
    r.submit_read(&gate, 0x00000005, 5, 0, BUF);
    r.expect_no_event();

    // The right handle at the wrong LUN or loop ID matches nothing.
    assert_eq!(r.abort(5, 1, 5), mbox::COMMAND_ERROR);
    r.ack();
    assert_eq!(r.abort(1, 0, 5), mbox::COMMAND_ERROR);
    r.ack();
    assert_eq!(r.abort(5, 0, 0x0005_0000), mbox::COMMAND_ERROR);
    r.ack();

    // The mailbox result posts before any entry.
    assert_eq!(r.abort(5, 0, 5), mbox::COMPLETE);
    assert_eq!(r.take_result(), mbox::COMPLETE);
    r.collect(&[(5, COMPLETION_ABORTED)]);

    // Gone now: a second abort finds nothing, and when the disk lets go
    // nothing reaches the ring and no data lands.
    assert_eq!(r.abort(5, 0, 5), mbox::COMMAND_ERROR);
    r.ack();
    r.release(&gate, 1);
    r.assert_untouched(BUF, 1024, 0xa5);
    r.expect_no_event();
}

/// Approximation: abort target retires the command running at the loop ID
/// with "reset" and resets the device there; LUN reset does the same for the
/// one LUN the device has. Each leaves the other ports alone, and the reset
/// device reports it as a unit attention.
#[test]
fn target_and_lun_reset() {
    let (r, gate) = Rig::with_slow_disks(&[1, 5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);

    r.submit_read(&gate, 0xa0, 5, 0, BUF);
    // Behind it in the ring, to another port.
    let e = r.scsi_entry(0xa2, 1, 0, &cdb_rw10(0x28, 0, 1), 0x20, u64::from(BUF + 512), 512, 0);
    r.publish(&e);
    assert_eq!(r.call(mbox::ABORT_TARGET, 5 << 8, 0, 0), mbox::COMPLETE);
    assert_eq!(r.take_result(), mbox::COMPLETE);
    r.collect(&[(0xa0, COMPLETION_RESET)]);
    r.release(&gate, 2);
    r.collect(&[(0xa2, COMPLETION_COMPLETE)]);
    assert_eq!(r.tur_sense(5, 0), 0x06);
    assert_eq!(r.tur_sense(1, 0), 0);

    r.submit_read(&gate, 0xb1, 5, 0, BUF);
    assert_eq!(r.call(mbox::LUN_RESET, 5 << 8, 0, 0), mbox::COMPLETE);
    assert_eq!(r.take_result(), mbox::COMPLETE);
    r.collect(&[(0xb1, COMPLETION_RESET)]);
    r.release(&gate, 3);
    assert_eq!(r.tur_sense(5, 0), 0x06);

    // An empty loop ID: nothing to retire, still complete.
    assert_eq!(r.call_ack(mbox::ABORT_TARGET, 7 << 8, 0, 0), mbox::COMPLETE);
    r.expect_no_event();
}

/// LIP reset retires the named loop ID's command, or any port's for 0xff,
/// resets those devices and brings the loop up again. Approximation: with
/// set-firmware-options bit 1 clear it posts loop up and port database
/// changed but no LIP event; with it set, LIP reset first.
#[test]
fn lip_reset() {
    let (r, gate) = Rig::with_slow_disks(&[1, 5]);
    r.start();
    assert_eq!(r.call_ack(mbox::LIP_RESET, 0xff00, 0, 0), mbox::COMMAND_ERROR);
    r.loop_up(ICB_OPT_PORT_DB, 0);

    r.submit_read(&gate, 0xc5, 5, 0, BUF);
    let e = r.scsi_entry(0xc1, 1, 0, &cdb_rw10(0x28, 0, 1), 0x20, u64::from(BUF + 512), 512, 0);
    r.publish(&e);
    assert_eq!(r.call(mbox::LIP_RESET, 5 << 8, 0, 0), mbox::COMPLETE);
    assert_eq!(r.take_result(), mbox::COMPLETE);
    r.collect(&[(0xc5, COMPLETION_RESET)]);
    r.release(&gate, 2);
    r.collect(&[(0xc1, COMPLETION_COMPLETE)]);
    assert_eq!(r.fw_state(), FW_STATE_WAIT_AL_PA);
    r.step(BRINGUP_NS);
    r.expect_events(&[event::LOOP_UP, event::PORT_DB_CHANGED]);
    assert_eq!(r.tur_sense(5, 0), 0x06);
    assert_eq!(r.tur_sense(1, 0), 0);

    assert_eq!(r.call_ack(mbox::SET_FW_OPTIONS, 0x0002, 0, 0), mbox::COMPLETE);
    assert_eq!(r.call_ack(mbox::LIP_RESET, 0xff00, 0, 0), mbox::COMPLETE);
    r.step(BRINGUP_NS);
    r.expect_events(&[event::LIP_RESET, event::LOOP_UP, event::PORT_DB_CHANGED]);
    assert_eq!(r.tur_sense(5, 0), 0x06);
    assert_eq!(r.tur_sense(1, 0), 0x06);
}

/// A soft reset cancels the command in flight without a status and resets
/// the loop's devices: once the disk's read is over, no data has reached the
/// host's buffer, no entry has reached the ring and ISR is clear.
#[test]
fn soft_reset_cancels() {
    let (r, gate) = Rig::with_slow_disks(&[1, 5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    r.mem.put(BUF, &[0xa5; 1024]);
    r.submit_read(&gate, 0xe5, 5, 0, BUF);
    let e = r.scsi_entry(0xe1, 1, 0, &cdb_rw10(0x28, 0, 1), 0x20, u64::from(BUF + 512), 512, 0);
    r.publish(&e);
    r.soft_reset();
    assert_eq!(r.rd(reg::ISR), 0);
    r.wr(reg::HCCR, HCCR_RELEASE);
    r.release(&gate, 1);
    assert_eq!(r.rd(reg::ISR), 0);
    r.assert_untouched(BUF, 1024, 0xa5);

    r.loop_up(ICB_OPT_PORT_DB, 0);
    assert_eq!(r.tur_sense(5, 0), 0x06);
    assert_eq!(r.tur_sense(1, 0), 0x06);
    assert_eq!(r.tur_sense(1, 0), 0);
}

/// An HCCR RISC reset cancels every command in flight without a status, as a
/// soft reset does, but leaves the loop's devices alone: none reports a unit
/// attention afterwards.
#[test]
fn hccr_reset_cancels() {
    let (r, gate) = Rig::with_slow_disks(&[1, 5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    r.mem.put(BUF, &[0xa5; 1024]);
    let slot = r.response(r.rs_next.get());
    r.submit_read(&gate, 0xe5, 5, 0, BUF);
    r.wr(reg::HCCR, HCCR_RESET);
    assert_eq!(r.rd(reg::ISR), 0);
    r.wr(reg::HCCR, HCCR_RELEASE);
    r.release(&gate, 1);
    assert_eq!(r.rd(reg::ISR), 0);
    r.assert_untouched(BUF, 1024, 0xa5);
    assert_eq!(r.response(r.rs_next.get()), slot);

    r.loop_up(ICB_OPT_PORT_DB, 0);
    assert_eq!(r.tur_sense(5, 0), 0);
    assert_eq!(r.tur_sense(1, 0), 0);
}

/// Initialise-firmware again with a command in flight cancels it without a
/// status. Once its disk's read is over, no data has reached the host's
/// buffer, the ring has no entry for it, ISR is clear, and the ring's first
/// entry is the next command's.
#[test]
fn init_firmware_cancels() {
    let (r, gate) = Rig::with_slow_disks(&[1, 5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    r.mem.put(BUF, &[0xa5; 512]);
    r.submit_read(&gate, 0xf5, 5, 0, BUF);
    r.rings_with(16, 16, ICB_OPT_PORT_DB, 0, 0);
    r.step(BRINGUP_NS);
    r.expect_events(&[event::LIP_OCCURRED, event::LOOP_UP, event::PORT_DB_CHANGED]);
    r.release(&gate, 1);
    assert_eq!(r.rd(reg::ISR), 0);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 0);
    r.assert_untouched(BUF, 512, 0xa5);
    assert_eq!(r.response(0), [0u8; ENTRY_SIZE]);

    // The new ring's first entry is this command's.
    r.tur_sense(1, 0);
    let e = r.response(0);
    assert_eq!((e[0], g32(&e, 4)), (ENTRY_STATUS, 0x7e00));
}

/// A data DMA the host bridge refuses completes the command 0x0002 (DMA
/// error) with no SCSI status, whether the adapter is delivering read data or
/// fetching write data, and the port keeps working.
#[test]
fn scsi_dma_error() {
    let r = Rig::with_disks("dma", &[5]);
    r.loop_up(ICB_OPT_PORT_DB, 0);
    let beyond = 1u64 << 40;
    let res = r.scsi(0xd0a0, 5, 0, &cdb_rw10(0x28, 0, 1), 0x20, beyond, 512, 0);
    assert_eq!((res.completion, res.scsi_status), (0x0002, 0));
    // The whole command's data, from an unmapped buffer.
    let res = r.scsi(0xd0a1, 5, 0, &cdb_rw10(0x2a, 0, 1), 0x40, beyond, 512, 0);
    assert_eq!((res.completion, res.scsi_status), (0x0002, 0));
    // A buffer shorter than the command's eight blocks, unmapped too.
    let res = r.scsi(0xd0a2, 5, 0, &cdb_rw10(0x2a, 0, 8), 0x40, beyond, 512, 0);
    assert_eq!((res.completion, res.scsi_status), (0x0002, 0));
    assert_eq!(r.tur_sense(5, 0), 0);
}
