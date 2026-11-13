//! The fake machine of a test: a HEART-like pending-bit recorder as the
//! interrupt controller, and guest RAM on the bus. A replay's host is the same
//! machine with its memory served from a recording (`TestHost::replaying`).

use super::*;
use crate::irq::InterruptController;
use crate::traits::{BusDevice, BusRead32, BusRead8};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

/// One `set_line(vector, level)` call, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::dev::kona) struct Sent {
    pub vector: u8,
    pub level: bool,
}

pub(in crate::dev::kona) struct TestHost {
    log: Mutex<Vec<Sent>>,
    /// The packets the board sent, as `packet` took them: source widget and address.
    packets: Mutex<Vec<(u8, u64)>>,
    /// Pending bits, one per vector: a `true` call sets a bit, as HEART's
    /// `set_vector` does; nothing here clears one but `clear`.
    pending: Mutex<[bool; 256]>,
    pub ram: Mutex<Vec<u8>>,
    /// Addresses at and above this fail to read and write.
    pub limit: u64,
    /// Guest memory served from a recording instead of RAM: reads come from its
    /// records, writes are kept, and neither fails.
    recorded: Option<Recorded>,
}

/// A recording's view of guest memory, for a replay: the reads the board is
/// expected to make, and the writes it made with the record index the replay was at.
pub(in crate::dev::kona) struct Recorded {
    pub reads: Mutex<Reads>,
    pub writes: Mutex<Vec<(usize, u64, Vec<u8>)>>,
    at: Arc<AtomicUsize>,
}

pub(in crate::dev::kona) struct ReadRec {
    pub src: u8,
    pub addr: u64,
    pub bytes: Vec<u8>,
    /// Bytes at the front already served.
    pub served: usize,
}

#[derive(Default)]
pub(in crate::dev::kona) struct Reads {
    pub recs: Vec<ReadRec>,
    /// The first record not yet wholly served.
    pub cursor: usize,
    /// Reads answered from a record.
    pub matched: u64,
    /// Reads answered from a record later than the first unserved one.
    pub skipped_over: u64,
    /// Bytes of reads no record covered, answered with zeros, with the first
    /// few addresses.
    pub unmatched: u64,
    pub unmatched_at: Vec<(u64, usize)>,
    /// The pull being served: its record, the bytes served and the address the
    /// next read must have to continue it.
    pub pull: Option<(usize, usize, u64)>,
}


impl Recorded {
    /// Continue where `other` is in the recording: which reads are served, the cursor and the pull in
    /// progress, so a board started part way through the stream is answered as the first one is.
    pub fn continue_from(&self, other: &Recorded) {
        let (mut mine, theirs) = (self.reads.lock(), other.reads.lock());
        for (m, t) in mine.recs.iter_mut().zip(&theirs.recs) {
            m.served = t.served;
        }
        mine.cursor = theirs.cursor;
        mine.pull = theirs.pull;
    }

    /// Read `buf.len()` bytes at `addr` from the records.
    pub fn read_into(&self, addr: u64, buf: &mut [u8]) {
        let mut r = self.reads.lock();
        let mut done = 0;
        while done < buf.len() {
            let want = addr + done as u64;
            // The first record from the cursor with unserved bytes at `want`.
            let found = (r.cursor..r.recs.len()).find(|&i| {
                let rec = &r.recs[i];
                // A pull's record (source 4) is keyed by its offset in the pool, not by an address.
                rec.src != 4 && want >= rec.addr + rec.served as u64 && want < rec.addr + rec.bytes.len() as u64
            });
            let Some(i) = found else {
                // A list pull reads the pool by offset, which the board turns into host
                // addresses through its map RAM; the recording keeps the offset. Pulls
                // happen in the order they were recorded, so serve the next one, from
                // its first byte, as the reads that follow it continue.
                let current = r.pull.filter(|&(i, used, next)| next == want && used < r.recs[i].bytes.len());
                let started = current.or_else(|| (0..r.recs.len()).find(|&i| r.recs[i].src == 4 && r.recs[i].served == 0).map(|i| (i, 0, want)));
                if let Some((i, used, _)) = started {
                    let n = (r.recs[i].bytes.len() - used).min(buf.len() - done);
                    buf[done..done + n].copy_from_slice(&r.recs[i].bytes[used..used + n]);
                    r.recs[i].served = used + n;
                    r.pull = Some((i, used + n, want + n as u64));
                    done += n;
                    continue;
                }
                r.unmatched += (buf.len() - done) as u64;
                if r.unmatched_at.len() < 8 {
                    r.unmatched_at.push((want, buf.len() - done));
                }
                buf[done..].fill(0);
                return;
            };
            if i > r.cursor {
                r.skipped_over += 1;
            }
            let rec = &mut r.recs[i];
            let from = (want - rec.addr) as usize;
            let n = (rec.bytes.len() - from).min(buf.len() - done);
            buf[done..done + n].copy_from_slice(&rec.bytes[from..from + n]);
            rec.served = from + n;
            done += n;
            while r.cursor < r.recs.len() && r.recs[r.cursor].served == r.recs[r.cursor].bytes.len() {
                r.cursor += 1;
            }
        }
        r.matched += 1;
    }
}



/// The configuration of a test board: its interrupt packets go to widget 9, where
/// the kernel's programming in the tests sends them.
pub(in crate::dev::kona) fn test_cfg() -> KonaConfig {
    KonaConfig { host_widget: 9, ..KonaConfig::default() }
}

pub(in crate::dev::kona) const RAM: usize = 16 << 20;

impl TestHost {
    pub fn new() -> Arc<TestHost> {
        Arc::new(TestHost { log: Mutex::new(Vec::new()), packets: Mutex::default(), pending: Mutex::new([false; 256]), ram: Mutex::new(vec![0; RAM]), limit: RAM as u64, recorded: None })
    }

    /// A host that serves the board's reads from `reads` (source, address, bytes),
    /// in order, and keeps its writes with the index `at` held when it made them.
    pub fn replaying(reads: Vec<(u8, u64, Vec<u8>)>, at: Arc<AtomicUsize>) -> Arc<TestHost> {
        let recs = reads.into_iter().map(|(src, addr, bytes)| ReadRec { src, addr, bytes, served: 0 }).collect();
        let recorded = Recorded { reads: Mutex::new(Reads { recs, ..Default::default() }), writes: Mutex::new(Vec::new()), at };
        Arc::new(TestHost { log: Mutex::new(Vec::new()), packets: Mutex::default(), pending: Mutex::new([false; 256]), ram: Mutex::new(Vec::new()), limit: 0, recorded: Some(recorded) })
    }

    /// The recording's memory, for a host made by `replaying`.
    pub fn recorded(&self) -> &Recorded {
        self.recorded.as_ref().expect("a replaying host")
    }

    pub fn word(&self, addr: u64) -> u32 {
        let r = self.ram.lock();
        u32::from_be_bytes(r[addr as usize..addr as usize + 4].try_into().unwrap())
    }

    pub fn put(&self, addr: u64, bytes: &[u8]) {
        self.ram.lock()[addr as usize..addr as usize + bytes.len()].copy_from_slice(bytes);
    }

    pub fn put_words(&self, addr: u64, words: &[u32]) {
        for (i, w) in words.iter().enumerate() {
            self.put(addr + 4 * i as u64, &w.to_be_bytes());
        }
    }

    pub fn bytes(&self, addr: u64, len: usize) -> Vec<u8> {
        self.ram.lock()[addr as usize..addr as usize + len].to_vec()
    }

    /// Every interrupt call, in order.
    pub fn sent(&self) -> Vec<Sent> {
        self.log.lock().clone()
    }

    /// The source widget and address of each interrupt packet, in order.
    pub fn packets(&self) -> Vec<(u8, u64)> {
        self.packets.lock().clone()
    }

    /// Whether pending bit `vector` is set.
    pub fn pending(&self, vector: u8) -> bool {
        self.pending.lock()[vector as usize]
    }

    /// The processor clearing a pending bit.
    pub fn clear(&self, vector: u8) {
        self.pending.lock()[vector as usize] = false;
    }
}

/// A register file and a stream state with no thread, over a fake machine: for a
/// test that calls the front end or a back end's port directly.
pub(in crate::dev::kona) fn bare(t: &Arc<TestHost>, cfg: KonaConfig) -> (super::super::front::Host, super::super::pipe::Core) {
    let host = super::super::front::Host::new(super::super::front::Front::new(&cfg), t.clone(), t.clone());
    (host, super::super::pipe::Core::new(cfg, super::super::flash::Flash::fresh()))
}

impl InterruptController for TestHost {
    fn packet(&self, source: u8, address: u64, line: u32, level: bool) -> bool {
        self.packets.lock().push((source, address));
        self.set_line(line, level);
        true
    }

    fn set_line(&self, line: u32, level: bool) {
        self.log.lock().push(Sent { vector: line as u8, level });
        if level {
            self.pending.lock()[line as usize & 0xff] = true;
        }
    }
}

impl BusDevice for TestHost {
    fn read8(&self, addr: u64) -> BusRead8 {
        if let Some(r) = &self.recorded {
            let mut b = [0u8];
            r.read_into(addr, &mut b);
            return BusRead8::ok(b[0]);
        }
        if addr >= self.limit {
            return BusRead8::err();
        }
        BusRead8::ok(self.ram.lock()[addr as usize])
    }

    fn write8(&self, addr: u64, val: u8) -> u32 {
        if let Some(r) = &self.recorded {
            r.writes.lock().push((r.at.load(Relaxed), addr, vec![val]));
            return BUS_OK;
        }
        if addr >= self.limit {
            return BUS_ERR;
        }
        self.ram.lock()[addr as usize] = val;
        BUS_OK
    }

    fn read32(&self, addr: u64) -> BusRead32 {
        if let Some(r) = &self.recorded {
            let mut b = [0u8; 4];
            r.read_into(addr, &mut b);
            return BusRead32::ok(u32::from_be_bytes(b));
        }
        if addr + 4 > self.limit {
            return BusRead32::err();
        }
        BusRead32::ok(u32::from_be_bytes(self.ram.lock()[addr as usize..addr as usize + 4].try_into().unwrap()))
    }

    fn write32(&self, addr: u64, val: u32) -> u32 {
        if let Some(r) = &self.recorded {
            r.writes.lock().push((r.at.load(Relaxed), addr, val.to_be_bytes().to_vec()));
            return BUS_OK;
        }
        if addr + 4 > self.limit {
            return BUS_ERR;
        }
        self.ram.lock()[addr as usize..addr as usize + 4].copy_from_slice(&val.to_be_bytes());
        BUS_OK
    }
}
