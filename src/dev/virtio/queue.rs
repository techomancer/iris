//! A split virtqueue (virtio 1.x, "modern" layout) over guest memory.
//!
//! Everything in the rings is little-endian, whatever the guest CPU is: a
//! big-endian MIPS driver swaps, the way a Linux `virtio_*16_to_cpu` does.
//! Guest memory is read and written as bytes ([`GuestMemory`]), so this file
//! has no opinion about the bus the memory sits on.

use super::GuestMemory;

/// `VIRTQ_DESC_F_NEXT`: the chain continues at `next`.
const DESC_F_NEXT: u16 = 1;
/// `VIRTQ_DESC_F_WRITE`: the device writes this buffer.
const DESC_F_WRITE: u16 = 2;
/// `VIRTQ_AVAIL_F_NO_INTERRUPT`: the driver does not want a used-buffer interrupt.
const AVAIL_F_NO_INTERRUPT: u16 = 1;

/// Largest ring the device offers. Guests choose any power of two up to it.
pub const QUEUE_MAX: u16 = 256;

/// One buffer of a descriptor chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Desc {
    pub addr: u64,
    pub len: u32,
    /// The device writes it (`VIRTQ_DESC_F_WRITE`); otherwise the device reads it.
    pub write: bool,
}

/// A descriptor chain taken from the available ring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chain {
    /// Index of the head descriptor, the id the used ring reports back.
    pub head: u16,
    pub descs: Vec<Desc>,
}

impl Chain {
    /// Every device-readable buffer, in order, as one byte string.
    /// `None` if guest memory did not answer.
    pub fn read_all(&self, mem: &dyn GuestMemory) -> Option<Vec<u8>> {
        let total: usize = self.descs.iter().filter(|d| !d.write).map(|d| d.len as usize).sum();
        let mut out = vec![0u8; total];
        let mut at = 0;
        for d in self.descs.iter().filter(|d| !d.write) {
            let n = d.len as usize;
            if !mem.read(d.addr, &mut out[at..at + n]) {
                return None;
            }
            at += n;
        }
        Some(out)
    }

    /// Room for the device to write, in bytes.
    pub fn writable_len(&self) -> usize {
        self.descs.iter().filter(|d| d.write).map(|d| d.len as usize).sum()
    }

    /// Fill the device-writable buffers from the front with `data`. Returns
    /// the bytes written, which is less than `data.len()` when the chain is
    /// short. `None` if guest memory did not answer.
    pub fn write_all(&self, mem: &dyn GuestMemory, data: &[u8]) -> Option<usize> {
        let mut at = 0;
        for d in self.descs.iter().filter(|d| d.write) {
            if at == data.len() {
                break;
            }
            let n = (d.len as usize).min(data.len() - at);
            if !mem.write(d.addr, &data[at..at + n]) {
                return None;
            }
            at += n;
        }
        Some(at)
    }
}

/// One queue's state as the transport programs it and the device consumes it.
#[derive(Clone, Debug, Default)]
pub struct Queue {
    /// Ring size the driver chose (0 until it does).
    pub size: u16,
    pub ready: bool,
    pub desc: u64,
    pub avail: u64,
    pub used: u64,
    /// Next available-ring entry to take.
    last_avail: u16,
    /// Value of the used ring's `idx` the device has published.
    used_idx: u16,
}

impl Queue {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// The driver has set this queue up and the rings are plausible.
    pub fn usable(&self) -> bool {
        self.ready && self.size != 0 && self.size <= QUEUE_MAX && self.size.is_power_of_two()
            && self.desc != 0 && self.avail != 0 && self.used != 0
    }

    fn rd16(&self, mem: &dyn GuestMemory, addr: u64) -> Option<u16> {
        let mut b = [0u8; 2];
        mem.read(addr, &mut b).then(|| u16::from_le_bytes(b))
    }

    /// Buffers the driver has made available and the device has not taken.
    pub fn pending(&self, mem: &dyn GuestMemory) -> u16 {
        if !self.usable() {
            return 0;
        }
        match self.rd16(mem, self.avail + 2) {
            Some(idx) => idx.wrapping_sub(self.last_avail),
            None => 0,
        }
    }

    /// Take the next available chain. `None` when the ring is empty, the
    /// queue is not set up, or the chain is malformed (a loop, an index out
    /// of range, or longer than the ring): a broken chain is consumed so one
    /// bad entry cannot wedge the queue.
    pub fn pop(&mut self, mem: &dyn GuestMemory) -> Option<Chain> {
        if self.pending(mem) == 0 {
            return None;
        }
        let slot = (self.last_avail % self.size) as u64;
        let head = self.rd16(mem, self.avail + 4 + 2 * slot)?;
        self.last_avail = self.last_avail.wrapping_add(1);
        let mut descs = Vec::new();
        let mut i = head;
        for _ in 0..self.size {
            if i >= self.size {
                return self.drop_broken(mem, head);
            }
            let mut raw = [0u8; 16];
            if !mem.read(self.desc + 16 * i as u64, &mut raw) {
                return self.drop_broken(mem, head);
            }
            let addr = u64::from_le_bytes(raw[0..8].try_into().unwrap());
            let len = u32::from_le_bytes(raw[8..12].try_into().unwrap());
            let flags = u16::from_le_bytes(raw[12..14].try_into().unwrap());
            let next = u16::from_le_bytes(raw[14..16].try_into().unwrap());
            descs.push(Desc { addr, len, write: flags & DESC_F_WRITE != 0 });
            if flags & DESC_F_NEXT == 0 {
                return Some(Chain { head, descs });
            }
            i = next;
        }
        self.drop_broken(mem, head)
    }

    /// Return a malformed chain to the driver unused.
    fn drop_broken(&mut self, mem: &dyn GuestMemory, head: u16) -> Option<Chain> {
        self.push_used(mem, head, 0);
        None
    }

    /// Hand `head` back to the driver having written `written` bytes.
    /// Returns whether the driver wants an interrupt for it.
    pub fn push_used(&mut self, mem: &dyn GuestMemory, head: u16, written: u32) -> bool {
        if !self.usable() {
            return false;
        }
        let slot = (self.used_idx % self.size) as u64;
        let mut entry = [0u8; 8];
        entry[0..4].copy_from_slice(&(head as u32).to_le_bytes());
        entry[4..8].copy_from_slice(&written.to_le_bytes());
        // Entry first, then the index that publishes it.
        if !mem.write(self.used + 4 + 8 * slot, &entry) {
            return false;
        }
        self.used_idx = self.used_idx.wrapping_add(1);
        if !mem.write(self.used + 2, &self.used_idx.to_le_bytes()) {
            return false;
        }
        self.rd16(mem, self.avail).map_or(true, |f| f & AVAIL_F_NO_INTERRUPT == 0)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use super::super::VecMemory;

    /// Lay out a queue of `size` at fixed addresses in `mem`.
    pub fn setup(size: u16) -> Queue {
        Queue { size, ready: true, desc: 0x1000, avail: 0x2000, used: 0x3000, ..Default::default() }
    }

    pub fn put_desc(mem: &VecMemory, i: u16, addr: u64, len: u32, flags: u16, next: u16) {
        let mut raw = [0u8; 16];
        raw[0..8].copy_from_slice(&addr.to_le_bytes());
        raw[8..12].copy_from_slice(&len.to_le_bytes());
        raw[12..14].copy_from_slice(&flags.to_le_bytes());
        raw[14..16].copy_from_slice(&next.to_le_bytes());
        assert!(mem.write(0x1000 + 16 * i as u64, &raw));
    }

    /// Publish `head` as the next available entry.
    pub fn make_avail(mem: &VecMemory, q: &Queue, slot_count: u16, head: u16) {
        let mut b = [0u8; 2];
        mem.read(q.avail + 2, &mut b);
        let idx = u16::from_le_bytes(b);
        mem.write(q.avail + 4 + 2 * (idx % slot_count) as u64, &head.to_le_bytes());
        mem.write(q.avail + 2, &idx.wrapping_add(1).to_le_bytes());
    }

    pub fn used_idx(mem: &VecMemory, q: &Queue) -> u16 {
        let mut b = [0u8; 2];
        mem.read(q.used + 2, &mut b);
        u16::from_le_bytes(b)
    }

    pub fn used_entry(mem: &VecMemory, q: &Queue, slot: u64) -> (u32, u32) {
        let mut b = [0u8; 8];
        mem.read(q.used + 4 + 8 * slot, &mut b);
        (u32::from_le_bytes(b[0..4].try_into().unwrap()), u32::from_le_bytes(b[4..8].try_into().unwrap()))
    }

    #[test]
    fn pops_a_chain_and_publishes_used() {
        let mem = VecMemory::new(0x10000);
        let mut q = setup(8);
        // head 3 -> 5; 3 readable, 5 writable.
        put_desc(&mem, 3, 0x4000, 4, DESC_F_NEXT, 5);
        put_desc(&mem, 5, 0x5000, 16, DESC_F_WRITE, 0);
        mem.write(0x4000, b"ping");
        make_avail(&mem, &q, 8, 3);

        assert_eq!(q.pending(&mem), 1);
        let c = q.pop(&mem).unwrap();
        assert_eq!(c.head, 3);
        assert_eq!(c.read_all(&mem).unwrap(), b"ping");
        assert_eq!(c.writable_len(), 16);
        assert_eq!(c.write_all(&mem, b"pong!").unwrap(), 5);
        assert!(q.pop(&mem).is_none());

        assert!(q.push_used(&mem, c.head, 5));
        assert_eq!(used_idx(&mem, &q), 1);
        assert_eq!(used_entry(&mem, &q, 0), (3, 5));
        let mut out = [0u8; 5];
        mem.read(0x5000, &mut out);
        assert_eq!(&out, b"pong!");
    }

    #[test]
    fn a_looping_chain_is_returned_unused() {
        let mem = VecMemory::new(0x10000);
        let mut q = setup(4);
        put_desc(&mem, 0, 0x4000, 1, DESC_F_NEXT, 1);
        put_desc(&mem, 1, 0x4001, 1, DESC_F_NEXT, 0);
        make_avail(&mem, &q, 4, 0);
        assert!(q.pop(&mem).is_none());
        assert_eq!(used_idx(&mem, &q), 1);
        assert_eq!(used_entry(&mem, &q, 0), (0, 0));
    }

    #[test]
    fn no_interrupt_flag_is_honoured_and_indices_wrap() {
        let mem = VecMemory::new(0x10000);
        let mut q = setup(2);
        mem.write(q.avail, &AVAIL_F_NO_INTERRUPT.to_le_bytes());
        for n in 0..5u16 {
            put_desc(&mem, 0, 0x4000, 1, 0, 0);
            make_avail(&mem, &q, 2, 0);
            let c = q.pop(&mem).unwrap();
            assert!(!q.push_used(&mem, c.head, 0), "round {n}");
        }
        assert_eq!(used_idx(&mem, &q), 5);
    }

    #[test]
    fn an_unready_queue_yields_nothing() {
        let mem = VecMemory::new(0x10000);
        let mut q = setup(8);
        q.ready = false;
        make_avail(&mem, &q, 8, 0);
        assert!(q.pop(&mem).is_none());
        assert!(!q.push_used(&mem, 0, 0));
    }
}
