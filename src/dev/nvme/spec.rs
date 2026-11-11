//! Constants and wire formats from the NVM Express Base Specification,
//! revision 1.4. Section numbers in the comments are that document's.

/// Controller register offsets (3.1).
pub const REG_CAP: u32 = 0x00;
pub const REG_VS: u32 = 0x08;
pub const REG_INTMS: u32 = 0x0c;
pub const REG_INTMC: u32 = 0x10;
pub const REG_CC: u32 = 0x14;
pub const REG_CSTS: u32 = 0x1c;
pub const REG_NSSR: u32 = 0x20;
pub const REG_AQA: u32 = 0x24;
pub const REG_ASQ: u32 = 0x28;
pub const REG_ACQ: u32 = 0x30;
pub const REG_CMBLOC: u32 = 0x38;
pub const REG_CMBSZ: u32 = 0x3c;
/// The doorbells start here; with CAP.DSTRD = 0 each is 4 bytes, submission
/// tail then completion head for queue 0, then queue 1, and so on (3.1.10).
pub const DOORBELLS: u32 = 0x1000;

/// Size of BAR0: the registers, the doorbells for the admin queue and every
/// I/O queue pair, rounded to a power of two.
pub const BAR_SIZE: u32 = 0x2000;

/// I/O queue pairs the controller can create.
pub const MAX_IO_QUEUES: usize = 16;
/// CAP.MQES is zero based: 1024 entries.
pub const MAX_QUEUE_ENTRIES: u32 = 1024;
/// The admin queue size comes from a 12-bit zero-based field.
pub const MAX_ADMIN_ENTRIES: u32 = 4096;

/// Version register: 1.4.0.
pub const VERSION: u32 = 0x0001_0400;

/// CAP (3.1.1): MQES 1023, CQR, AMS none, TO 10 (5 s), DSTRD 0, NSSRS 0,
/// CSS bit 0 (NVM command set), BPS 0, MPSMIN 0 (4 KiB), MPSMAX 4 (64 KiB).
pub const CAP: u64 = 1023 | 1 << 16 | 10 << 24 | 1 << 37 | 4 << 52;

/// CC fields (3.1.5).
pub const CC_EN: u32 = 1;
pub const CC_WRITABLE: u32 = 0x00ff_fff1;
pub fn cc_css(cc: u32) -> u32 {
    cc >> 4 & 7
}
pub fn cc_mps(cc: u32) -> u32 {
    cc >> 7 & 15
}
pub fn cc_ams(cc: u32) -> u32 {
    cc >> 11 & 7
}
pub fn cc_shn(cc: u32) -> u32 {
    cc >> 14 & 3
}
pub fn cc_iosqes(cc: u32) -> u32 {
    cc >> 16 & 15
}
pub fn cc_iocqes(cc: u32) -> u32 {
    cc >> 20 & 15
}

/// CSTS fields (3.1.6).
pub const CSTS_RDY: u32 = 1;
pub const CSTS_CFS: u32 = 2;
/// SHST = 10b: shutdown processing complete.
pub const CSTS_SHST_DONE: u32 = 2 << 2;

/// Submission and completion queue entry sizes (4.1, 4.6).
pub const SQE_SIZE: usize = 64;
pub const CQE_SIZE: usize = 16;

/// Admin command opcodes (Figure 139).
pub mod admin {
    pub const DELETE_IO_SQ: u8 = 0x00;
    pub const CREATE_IO_SQ: u8 = 0x01;
    pub const GET_LOG_PAGE: u8 = 0x02;
    pub const DELETE_IO_CQ: u8 = 0x04;
    pub const CREATE_IO_CQ: u8 = 0x05;
    pub const IDENTIFY: u8 = 0x06;
    pub const ABORT: u8 = 0x08;
    pub const SET_FEATURES: u8 = 0x09;
    pub const GET_FEATURES: u8 = 0x0a;
    pub const ASYNC_EVENT_REQUEST: u8 = 0x0c;
}

/// NVM command set opcodes (Figure 346).
pub mod io {
    pub const FLUSH: u8 = 0x00;
    pub const WRITE: u8 = 0x01;
    pub const READ: u8 = 0x02;
    pub const WRITE_ZEROES: u8 = 0x08;
}

/// Feature identifiers (Figure 317).
pub mod feature {
    pub const ARBITRATION: u8 = 0x01;
    pub const POWER_MANAGEMENT: u8 = 0x02;
    pub const TEMPERATURE_THRESHOLD: u8 = 0x04;
    pub const ERROR_RECOVERY: u8 = 0x05;
    pub const VOLATILE_WRITE_CACHE: u8 = 0x06;
    pub const NUMBER_OF_QUEUES: u8 = 0x07;
    pub const INTERRUPT_COALESCING: u8 = 0x08;
    pub const INTERRUPT_VECTOR_CONFIG: u8 = 0x09;
    pub const WRITE_ATOMICITY: u8 = 0x0a;
    pub const ASYNC_EVENT_CONFIG: u8 = 0x0b;
}

/// Status as a completion entry holds it: status code in bits 7:0, status
/// code type in 10:8, and the do-not-retry bit in 14 (4.6.1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Status(pub u16);

impl Status {
    pub const SUCCESS: Status = Status(0);
    const fn generic(sc: u16) -> Status {
        Status(sc | DNR)
    }
    const fn specific(sc: u16) -> Status {
        Status(sc | 1 << 8 | DNR)
    }
    pub const INVALID_OPCODE: Status = Status::generic(0x01);
    pub const INVALID_FIELD: Status = Status::generic(0x02);
    pub const DATA_TRANSFER_ERROR: Status = Status::generic(0x04);
    pub const ABORT_REQUESTED: Status = Status::generic(0x07);
    pub const INVALID_NAMESPACE: Status = Status::generic(0x0b);
    pub const COMMAND_SEQUENCE_ERROR: Status = Status::generic(0x0c);
    pub const LBA_OUT_OF_RANGE: Status = Status::generic(0x80);
    pub const INVALID_COMPLETION_QUEUE: Status = Status::specific(0x00);
    pub const INVALID_QUEUE_ID: Status = Status::specific(0x01);
    pub const INVALID_QUEUE_SIZE: Status = Status::specific(0x02);
    pub const AER_LIMIT_EXCEEDED: Status = Status::specific(0x05);
    pub const INVALID_LOG_PAGE: Status = Status::specific(0x09);
    pub const INVALID_INTERRUPT_VECTOR: Status = Status::specific(0x08);
    pub const INVALID_QUEUE_DELETION: Status = Status::specific(0x0c);
    pub const FEATURE_NOT_SAVEABLE: Status = Status::specific(0x0d);

    /// The word as it sits in the upper half of completion dword 3 (the
    /// phase tag is bit 0, so the status starts at bit 1).
    pub fn field(self) -> u16 {
        self.0 << 1
    }
}

const DNR: u16 = 1 << 14;

/// A submission queue entry, the fields every command shares (4.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sqe {
    pub opcode: u8,
    pub cid: u16,
    pub nsid: u32,
    pub prp1: u64,
    pub prp2: u64,
    /// Command dwords 10 to 15.
    pub cdw: [u32; 6],
}

impl Sqe {
    pub fn parse(b: &[u8; SQE_SIZE]) -> Sqe {
        let d = |i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        let q = |i: usize| u64::from(d(i)) | u64::from(d(i + 1)) << 32;
        Sqe {
            opcode: b[0],
            cid: (d(0) >> 16) as u16,
            nsid: d(1),
            prp1: q(6),
            prp2: q(8),
            cdw: [d(10), d(11), d(12), d(13), d(14), d(15)],
        }
    }

    pub fn encode(&self) -> [u8; SQE_SIZE] {
        let mut b = [0u8; SQE_SIZE];
        let mut put = |i: usize, v: u32| b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        put(0, u32::from(self.opcode) | u32::from(self.cid) << 16);
        put(1, self.nsid);
        put(6, self.prp1 as u32);
        put(7, (self.prp1 >> 32) as u32);
        put(8, self.prp2 as u32);
        put(9, (self.prp2 >> 32) as u32);
        for (i, v) in self.cdw.iter().enumerate() {
            put(10 + i, *v);
        }
        b
    }

    /// Command dword 10 and 11 as the 64-bit starting LBA of a Read or Write.
    pub fn slba(&self) -> u64 {
        u64::from(self.cdw[0]) | u64::from(self.cdw[1]) << 32
    }

    /// The number of logical blocks of a Read, Write or Write Zeroes (zero
    /// based in the command).
    pub fn nlb(&self) -> u32 {
        (self.cdw[2] & 0xffff) + 1
    }
}

/// Copy `s` into `dst`, padded with spaces: the form the ASCII strings of
/// Identify Controller take (Figure 247).
pub fn ascii_field(dst: &mut [u8], s: &str) {
    dst.fill(b' ');
    let n = s.len().min(dst.len());
    dst[..n].copy_from_slice(&s.as_bytes()[..n]);
}
