//! The commands: admin (5) and NVM (6), and the PRP data pointers (4.3) they
//! move data with.

use super::spec::*;
use super::*;
use crate::scsi::{scsi_cmd, ScsiDataLength, ScsiRequest};

/// The most a Read or Write moves: MDTS 5 is 2^5 minimum pages, 128 KiB.
const MDTS: u8 = 5;
const MAX_TRANSFER: usize = 1 << (12 + MDTS);
const BLOCK: usize = 512;
/// The Identify data structures are 4 KiB (5.15).
const IDENTIFY_SIZE: usize = 4096;
/// Async Event Request commands that may wait at once (AERL + 1).
const AER_LIMIT: usize = 4;

type Done = (Status, u32);

fn put16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

impl Nvme {
    // ---- data pointers ----------------------------------------------------

    /// The (address, length) pieces of memory a PRP1/PRP2 pair names for a
    /// transfer of `len` bytes, with memory pages of `page` bytes (4.3).
    fn prp_segments(&self, page: u64, prp1: u64, prp2: u64, len: usize) -> Result<Vec<(u64, usize)>, Status> {
        let dma = self.dma().ok_or(Status::DATA_TRANSFER_ERROR)?;
        // PRP1 may start anywhere in a page, at a dword boundary.
        if prp1 & 3 != 0 {
            return Err(Status::INVALID_FIELD);
        }
        let first = len.min((page - (prp1 & (page - 1))) as usize);
        let mut segs = vec![(prp1, first)];
        let mut left = len - first;
        if left == 0 {
            return Ok(segs);
        }
        if left as u64 <= page {
            // PRP2 is the second and last page.
            if prp2 & (page - 1) != 0 {
                return Err(Status::INVALID_FIELD);
            }
            segs.push((prp2, left));
            return Ok(segs);
        }
        // PRP2 points to a list of 8-byte page addresses; the last entry of
        // a list that does not hold the rest points to the next list. A list
        // with room for one entry can only point on, so it carries no data
        // and would never end.
        let mut list = prp2;
        if list & 7 != 0 {
            return Err(Status::INVALID_FIELD);
        }
        while left > 0 {
            let room = ((page - (list & (page - 1))) / 8) as usize;
            let pages = (left as u64).div_ceil(page) as usize;
            let (data, chained) = if pages > room { (room - 1, true) } else { (pages, false) };
            if chained && room < 2 {
                return Err(Status::INVALID_FIELD);
            }
            let mut raw = vec![0u8; (data + chained as usize) * 8];
            if !dma.dma_read(list, &mut raw).is_ok() {
                return Err(Status::DATA_TRANSFER_ERROR);
            }
            let entry = |i: usize| u64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap());
            for i in 0..data {
                let addr = entry(i);
                if addr & (page - 1) != 0 {
                    return Err(Status::INVALID_FIELD);
                }
                let n = left.min(page as usize);
                segs.push((addr, n));
                left -= n;
            }
            if chained {
                list = entry(data);
                if list & 7 != 0 {
                    return Err(Status::INVALID_FIELD);
                }
            }
        }
        Ok(segs)
    }

    /// Copy `data` to the host memory a command's data pointer names.
    fn to_host(&self, page: u64, sqe: &Sqe, data: &[u8]) -> Status {
        let segs = match self.prp_segments(page, sqe.prp1, sqe.prp2, data.len()) {
            Ok(s) => s,
            Err(e) => return e,
        };
        let dma = self.dma().unwrap();
        let mut at = 0;
        for (addr, n) in segs {
            if !dma.dma_write(addr, &data[at..at + n]).is_ok() {
                return Status::DATA_TRANSFER_ERROR;
            }
            at += n;
        }
        Status::SUCCESS
    }

    /// Read `len` bytes from the host memory a command's data pointer names.
    fn from_host(&self, page: u64, sqe: &Sqe, len: usize) -> Result<Vec<u8>, Status> {
        let segs = self.prp_segments(page, sqe.prp1, sqe.prp2, len)?;
        let dma = self.dma().unwrap();
        let mut out = vec![0u8; len];
        let mut at = 0;
        for (addr, n) in segs {
            if !dma.dma_read(addr, &mut out[at..at + n]).is_ok() {
                return Err(Status::DATA_TRANSFER_ERROR);
            }
            at += n;
        }
        Ok(out)
    }

    fn page_size(&self) -> u64 {
        1 << (12 + cc_mps(self.st.lock().unwrap().cc))
    }

    // ---- admin commands ---------------------------------------------------

    /// Run an admin command. None: it completes later (an Async Event
    /// Request waits for an event).
    pub(super) fn admin(&self, sqe: &Sqe) -> Option<Done> {
        let mut st = self.st.lock().unwrap();
        let page = 1u64 << (12 + cc_mps(st.cc));
        let r = match sqe.opcode {
            admin::DELETE_IO_SQ => self.delete_sq(&mut st, sqe),
            admin::CREATE_IO_SQ => self.create_sq(&mut st, sqe, page),
            admin::GET_LOG_PAGE => self.get_log_page(&mut st, sqe, page),
            admin::DELETE_IO_CQ => self.delete_cq(&mut st, sqe),
            admin::CREATE_IO_CQ => self.create_cq(&mut st, sqe, page),
            admin::IDENTIFY => self.identify(&mut st, sqe, page),
            admin::ABORT => self.abort(&mut st, sqe),
            admin::SET_FEATURES => self.set_features(&mut st, sqe),
            admin::GET_FEATURES => self.get_features(&mut st, sqe),
            admin::ASYNC_EVENT_REQUEST => {
                if st.aers.len() >= AER_LIMIT {
                    Err(Status::AER_LIMIT_EXCEEDED)
                } else {
                    st.aers.push_back(sqe.cid);
                    return None;
                }
            }
            op => {
                *st.refused.entry((false, op)).or_default() += 1;
                Err(Status::INVALID_OPCODE)
            }
        };
        Some(match r {
            Ok(dw0) => (Status::SUCCESS, dw0),
            Err(s) => (s, 0),
        })
    }

    fn io_queue_id(&self, qid: u32) -> bool {
        qid >= 1 && qid as usize <= MAX_IO_QUEUES
    }

    /// Whether `qid` is within the queue counts Set Features last allocated.
    fn allocated(st: &State, qid: u32, completion: bool) -> bool {
        let n = if completion { st.feat.queues >> 16 } else { st.feat.queues & 0xffff };
        qid >= 1 && qid as usize <= MAX_IO_QUEUES && qid <= n + 1
    }

    fn create_cq(&self, st: &mut State, sqe: &Sqe, page: u64) -> Result<u32, Status> {
        let (qid, entries) = (sqe.cdw[0] & 0xffff, (sqe.cdw[0] >> 16) + 1);
        let (pc, ien, iv) = (sqe.cdw[1] & 1, sqe.cdw[1] >> 1 & 1, sqe.cdw[1] >> 16);
        if !Self::allocated(st, qid, true) || st.cqs[qid as usize].is_some() {
            return Err(Status::INVALID_QUEUE_ID);
        }
        if entries < 2 || entries > MAX_QUEUE_ENTRIES {
            return Err(Status::INVALID_QUEUE_SIZE);
        }
        // CAP.CQR is set: the queue must be physically contiguous.
        if pc == 0 || sqe.prp1 & (page - 1) != 0 {
            return Err(Status::INVALID_FIELD);
        }
        // Pin-based interrupts have one vector.
        if iv != 0 {
            return Err(Status::INVALID_INTERRUPT_VECTOR);
        }
        st.cqs[qid as usize] = Some(Cq { base: sqe.prp1, entries, phase: true, iv: iv as u16, ien: ien == 1, ..Default::default() });
        Ok(0)
    }

    fn create_sq(&self, st: &mut State, sqe: &Sqe, page: u64) -> Result<u32, Status> {
        let (qid, entries) = (sqe.cdw[0] & 0xffff, (sqe.cdw[0] >> 16) + 1);
        let (pc, cqid) = (sqe.cdw[1] & 1, sqe.cdw[1] >> 16);
        if !Self::allocated(st, qid, false) || st.sqs[qid as usize].is_some() {
            return Err(Status::INVALID_QUEUE_ID);
        }
        // The admin queue's completion queue is not for I/O.
        if !self.io_queue_id(cqid) || st.cqs[cqid as usize].is_none() {
            return Err(Status::INVALID_COMPLETION_QUEUE);
        }
        if entries < 2 || entries > MAX_QUEUE_ENTRIES {
            return Err(Status::INVALID_QUEUE_SIZE);
        }
        if pc == 0 || sqe.prp1 & (page - 1) != 0 {
            return Err(Status::INVALID_FIELD);
        }
        st.sqs[qid as usize] = Some(Sq { base: sqe.prp1, entries, cqid: cqid as u16, ..Default::default() });
        st.feat.queues_set = true;
        Ok(0)
    }

    fn delete_sq(&self, st: &mut State, sqe: &Sqe) -> Result<u32, Status> {
        let qid = sqe.cdw[0] & 0xffff;
        if !self.io_queue_id(qid) || st.sqs[qid as usize].is_none() {
            return Err(Status::INVALID_QUEUE_ID);
        }
        st.sqs[qid as usize] = None;
        Ok(0)
    }

    fn delete_cq(&self, st: &mut State, sqe: &Sqe) -> Result<u32, Status> {
        let qid = sqe.cdw[0] & 0xffff;
        if !self.io_queue_id(qid) || st.cqs[qid as usize].is_none() {
            return Err(Status::INVALID_QUEUE_ID);
        }
        // A completion queue with a submission queue still bound to it
        // cannot go.
        if st.sqs.iter().flatten().any(|s| u32::from(s.cqid) == qid) {
            return Err(Status::INVALID_QUEUE_DELETION);
        }
        st.cqs[qid as usize] = None;
        self.update_pin(st);
        Ok(0)
    }

    /// Abort (5.1): a command that has reached the controller has already
    /// run, since commands run one at a time to completion. The exception is
    /// an Async Event Request, which waits, and is completed as aborted.
    fn abort(&self, st: &mut State, sqe: &Sqe) -> Result<u32, Status> {
        let (sqid, cid) = (sqe.cdw[0] & 0xffff, (sqe.cdw[0] >> 16) as u16);
        if sqid as usize > MAX_IO_QUEUES || st.sqs[sqid as usize].is_none() {
            return Err(Status::INVALID_QUEUE_ID);
        }
        if sqid == 0 {
            if let Some(i) = st.aers.iter().position(|&c| c == cid) {
                st.aers.remove(i);
                self.post(st, 0, cid, 0, Status::ABORT_REQUESTED);
                return Ok(0);
            }
        }
        // Bit 0 set: the command was not aborted.
        Ok(1)
    }

    fn identify(&self, st: &mut State, sqe: &Sqe, page: u64) -> Result<u32, Status> {
        let cns = sqe.cdw[0] & 0xff;
        let have_ns = self.blocks() != 0;
        let mut d = vec![0u8; IDENTIFY_SIZE];
        match cns {
            0x00 => {
                if sqe.nsid != 1 || !have_ns {
                    return Err(Status::INVALID_NAMESPACE);
                }
                let n = self.blocks();
                put64(&mut d, 0, n); // NSZE
                put64(&mut d, 8, n); // NCAP
                put64(&mut d, 16, n); // NUSE
                // NSFEAT 0, NLBAF 0 (one format), FLBAS 0, no metadata, no
                // protection, no sharing.
                d[104..120].copy_from_slice(&*self.nguid.lock().unwrap()); // NGUID
                // LBA format 0: no metadata, 2^9 data size, best performance.
                d[128 + 2] = 9;
            }
            0x01 => {
                put16(&mut d, 0, PCI_VENDOR); // VID
                put16(&mut d, 2, 0); // SSVID
                ascii_field(&mut d[4..24], "IRIS0001"); // SN
                ascii_field(&mut d[24..64], "iris NVMe controller"); // MN
                ascii_field(&mut d[64..72], "1.0"); // FR
                d[72] = 0; // RAB
                // IEEE OUI (75:73) unassigned. CMIC 0: one controller, one port.
                d[77] = MDTS;
                put16(&mut d, 78, 0); // CNTLID
                put32(&mut d, 80, VERSION); // VER
                d[259] = 3; // AERL: four outstanding
                d[258] = 3; // ACL: four outstanding aborts
                d[260] = 0x01; // FRMW: slot 1 is read-only
                d[261] = 0; // LPA
                d[262] = 0; // ELPE: one error log entry
                d[263] = 0; // NPSS: one power state
                d[512] = 0x66; // SQES: 64 bytes
                d[513] = 0x44; // CQES: 16 bytes
                put16(&mut d, 514, 0); // MAXCMD: no limit stated
                put32(&mut d, 516, u32::from(have_ns)); // NN
                put16(&mut d, 520, 0x0008); // ONCS: Write Zeroes
                d[525] = 0; // VWC: no volatile write cache
                let nqn = b"nqn.2026-11.dev.iris:nvme";
                d[768..768 + nqn.len()].copy_from_slice(nqn); // SUBNQN
                // Power state descriptor 0: 25 W maximum, always operational.
                put16(&mut d, 2048, 2500);
            }
            0x02 => {
                // Active namespace IDs greater than the one given.
                if have_ns && sqe.nsid < 1 {
                    put32(&mut d, 0, 1);
                }
            }
            0x03 => {
                if sqe.nsid != 1 || !have_ns {
                    return Err(Status::INVALID_NAMESPACE);
                }
                d[0] = 2; // NIDT: NGUID
                d[1] = 16; // NIDL
                d[4..20].copy_from_slice(&*self.nguid.lock().unwrap());
            }
            _ => {
                *st.refused.entry((false, admin::IDENTIFY)).or_default() += 1;
                return Err(Status::INVALID_FIELD);
            }
        }
        match self.to_host(page, sqe, &d) {
            Status::SUCCESS => Ok(0),
            e => Err(e),
        }
    }

    fn get_log_page(&self, st: &mut State, sqe: &Sqe, page: u64) -> Result<u32, Status> {
        let lid = sqe.cdw[0] & 0xff;
        let numd = ((sqe.cdw[1] & 0xffff) << 16 | sqe.cdw[0] >> 16) as usize + 1;
        let offset = (u64::from(sqe.cdw[3]) << 32 | u64::from(sqe.cdw[2])) as usize;
        let mut d = vec![0u8; 512];
        match lid {
            // Error Information: no entries to report (zero error count).
            0x01 => d.truncate(64),
            0x02 => {
                // SMART / Health Information (5.14.1.2).
                d[1..3].copy_from_slice(&300u16.to_le_bytes()); // composite temperature, K
                d[3] = 100; // available spare
                d[4] = 10; // available spare threshold
                // Data units are thousands of 512-byte blocks, rounded up.
                let units = |b: u64| b.div_ceil(1000) as u128;
                let c = &st.counters;
                d[32..48].copy_from_slice(&units(c.blocks_read).to_le_bytes());
                d[48..64].copy_from_slice(&units(c.blocks_written).to_le_bytes());
                d[64..80].copy_from_slice(&u128::from(c.host_reads).to_le_bytes());
                d[80..96].copy_from_slice(&u128::from(c.host_writes).to_le_bytes());
            }
            // Firmware Slot Information: slot 1 active, revision "1.0".
            0x03 => {
                d[0] = 1;
                ascii_field(&mut d[8..16], "1.0");
            }
            _ => return Err(Status::INVALID_LOG_PAGE),
        }
        if offset & 3 != 0 || offset >= d.len() {
            return Err(Status::INVALID_FIELD);
        }
        let len = (numd * 4).min(d.len() - offset);
        match self.to_host(page, sqe, &d[offset..offset + len]) {
            Status::SUCCESS => Ok(0),
            e => Err(e),
        }
    }

    fn set_features(&self, st: &mut State, sqe: &Sqe) -> Result<u32, Status> {
        let (fid, v) = ((sqe.cdw[0] & 0xff) as u8, sqe.cdw[1]);
        // No feature is saveable.
        if sqe.cdw[0] >> 31 != 0 {
            return Err(Status::FEATURE_NOT_SAVEABLE);
        }
        match fid {
            feature::ARBITRATION => st.feat.arbitration = v,
            feature::POWER_MANAGEMENT => {
                // One power state.
                if v & 0x1f != 0 {
                    return Err(Status::INVALID_FIELD);
                }
                st.feat.power_management = v;
            }
            feature::TEMPERATURE_THRESHOLD => st.feat.temperature_threshold = v,
            feature::ERROR_RECOVERY => {
                // Deallocated or unwritten block error (bit 16) is not offered.
                if v & 1 << 16 != 0 {
                    return Err(Status::INVALID_FIELD);
                }
                st.feat.error_recovery = v;
            }
            feature::NUMBER_OF_QUEUES => {
                if st.feat.queues_set {
                    return Err(Status::COMMAND_SEQUENCE_ERROR);
                }
                if v & 0xffff == 0xffff || v >> 16 == 0xffff {
                    return Err(Status::INVALID_FIELD);
                }
                let cap = MAX_IO_QUEUES as u32 - 1;
                st.feat.queues = (v & 0xffff).min(cap) | (v >> 16).min(cap) << 16;
                return Ok(st.feat.queues);
            }
            feature::INTERRUPT_COALESCING => st.feat.interrupt_coalescing = v & 0xffff,
            feature::INTERRUPT_VECTOR_CONFIG => {
                if v & 0xffff != 0 {
                    return Err(Status::INVALID_FIELD);
                }
            }
            feature::WRITE_ATOMICITY => st.feat.write_atomicity = v & 1,
            feature::ASYNC_EVENT_CONFIG => st.feat.async_event_config = v,
            _ => {
                *st.refused_features.entry(fid).or_default() += 1;
                return Err(Status::INVALID_FIELD);
            }
        }
        Ok(0)
    }

    fn get_features(&self, st: &mut State, sqe: &Sqe) -> Result<u32, Status> {
        let (fid, sel) = ((sqe.cdw[0] & 0xff) as u8, sqe.cdw[0] >> 8 & 7);
        let f = if sel == 1 { Features::default() } else { st.feat.clone() };
        let v = match fid {
            feature::ARBITRATION => f.arbitration,
            feature::POWER_MANAGEMENT => f.power_management,
            feature::TEMPERATURE_THRESHOLD => f.temperature_threshold,
            feature::ERROR_RECOVERY => f.error_recovery,
            feature::NUMBER_OF_QUEUES => f.queues,
            feature::INTERRUPT_COALESCING => f.interrupt_coalescing,
            // Vector 0, coalescing not disabled.
            feature::INTERRUPT_VECTOR_CONFIG => sqe.cdw[1] & 0xffff,
            feature::WRITE_ATOMICITY => f.write_atomicity,
            feature::ASYNC_EVENT_CONFIG => f.async_event_config,
            _ => {
                *st.refused_features.entry(fid).or_default() += 1;
                return Err(Status::INVALID_FIELD);
            }
        };
        match sel {
            0 | 1 => Ok(v),
            // Supported capabilities (1.4): changeable, not saveable, not
            // namespace specific.
            3 => Ok(1 << 2),
            // Saved values: no feature is saveable.
            _ => Err(Status::INVALID_FIELD),
        }
    }

    // ---- NVM commands -----------------------------------------------------

    /// Run an NVM command on the namespace.
    pub(super) fn nvm(&self, sqe: &Sqe) -> Done {
        let r = match sqe.opcode {
            io::FLUSH => self.flush(sqe),
            io::WRITE => self.write(sqe),
            io::READ => self.read(sqe),
            io::WRITE_ZEROES => self.write_zeroes(sqe),
            op => {
                *self.st.lock().unwrap().refused.entry((true, op)).or_default() += 1;
                Err(Status::INVALID_OPCODE)
            }
        };
        match r {
            Ok(()) => (Status::SUCCESS, 0),
            Err(s) => (s, 0),
        }
    }

    /// The namespace named by a command, or why there is none. FFFFFFFFh
    /// means every namespace, which only Flush accepts.
    fn namespace(&self, sqe: &Sqe) -> Result<(), Status> {
        let all_ok = sqe.opcode == io::FLUSH && sqe.nsid == u32::MAX;
        if (sqe.nsid != 1 && !all_ok) || self.blocks() == 0 {
            return Err(Status::INVALID_NAMESPACE);
        }
        Ok(())
    }

    /// The range a Read, Write or Write Zeroes covers, checked against the
    /// namespace: (first block, blocks).
    fn range(&self, sqe: &Sqe) -> Result<(u64, usize), Status> {
        self.namespace(sqe)?;
        // Protection information is not offered (DPS is 0), so PRINFO is
        // reserved; the limited-retry, FUA and stream bits are read below.
        if sqe.cdw[2] >> 26 & 0xf != 0 {
            return Err(Status::INVALID_FIELD);
        }
        let (slba, nlb) = (sqe.slba(), sqe.nlb() as usize);
        if slba.checked_add(nlb as u64).is_none_or(|end| end > self.blocks()) {
            return Err(Status::LBA_OUT_OF_RANGE);
        }
        Ok((slba, nlb))
    }

    /// Run one SCSI command against the namespace's disk.
    fn disk(&self, cdb: Vec<u8>, data_in: Option<Vec<u8>>) -> Result<Vec<u8>, Status> {
        let req = ScsiRequest { cdb, data_len: ScsiDataLength::Unlimited, data_in };
        match self.bus.request(0, &req) {
            Some(Ok(r)) if r.status == 0 => Ok(r.data),
            _ => Err(Status::DATA_TRANSFER_ERROR),
        }
    }

    fn rw10(op: u8, fua: bool, lba: u64, blocks: usize) -> Vec<u8> {
        let mut c = vec![op, if fua { 0x08 } else { 0 }];
        c.extend((lba as u32).to_be_bytes());
        c.push(0);
        c.extend((blocks as u16).to_be_bytes());
        c.push(0);
        c
    }

    fn read(&self, sqe: &Sqe) -> Result<(), Status> {
        let (lba, blocks) = self.range(sqe)?;
        if blocks * BLOCK > MAX_TRANSFER {
            return Err(Status::INVALID_FIELD);
        }
        let data = self.disk(Self::rw10(scsi_cmd::READ_10, false, lba, blocks), None)?;
        if data.len() != blocks * BLOCK {
            return Err(Status::DATA_TRANSFER_ERROR);
        }
        match self.to_host(self.page_size(), sqe, &data) {
            Status::SUCCESS => {}
            e => return Err(e),
        }
        let mut st = self.st.lock().unwrap();
        st.counters.host_reads += 1;
        st.counters.blocks_read += blocks as u64;
        Ok(())
    }

    fn write(&self, sqe: &Sqe) -> Result<(), Status> {
        let (lba, blocks) = self.range(sqe)?;
        if blocks * BLOCK > MAX_TRANSFER {
            return Err(Status::INVALID_FIELD);
        }
        let data = self.from_host(self.page_size(), sqe, blocks * BLOCK)?;
        let fua = sqe.cdw[2] >> 30 & 1 == 1;
        self.disk(Self::rw10(scsi_cmd::WRITE_10, fua, lba, blocks), Some(data))?;
        let mut st = self.st.lock().unwrap();
        st.counters.host_writes += 1;
        st.counters.blocks_written += blocks as u64;
        Ok(())
    }

    fn write_zeroes(&self, sqe: &Sqe) -> Result<(), Status> {
        let (mut lba, mut left) = self.range(sqe)?;
        let fua = sqe.cdw[2] >> 30 & 1 == 1;
        let total = left as u64;
        while left > 0 {
            let n = left.min(MAX_TRANSFER / BLOCK);
            self.disk(Self::rw10(scsi_cmd::WRITE_10, fua, lba, n), Some(vec![0; n * BLOCK]))?;
            lba += n as u64;
            left -= n;
        }
        let mut st = self.st.lock().unwrap();
        st.counters.host_writes += 1;
        st.counters.blocks_written += total;
        Ok(())
    }

    fn flush(&self, sqe: &Sqe) -> Result<(), Status> {
        self.namespace(sqe)?;
        self.disk(vec![scsi_cmd::SYNCHRONIZE_CACHE_10, 0, 0, 0, 0, 0, 0, 0, 0, 0], None).map(|_| ())
    }
}
