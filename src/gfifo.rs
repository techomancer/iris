//! MPSC register-write ring buffer shared by the graphics boards.
//!
//! Extracted from `rex3.rs` so GR2 (`src/dev/gr2`) can use the same queue for
//! its HQ2 command FIFO and RE3 register FIFO. The depth is a const generic;
//! the entry stays `(addr: u32, val: u64)` so every user shares one proven
//! implementation. Back-pressure rules (`try_push` returning false → the bus
//! write reports `BUS_BUSY`) are documented on the methods below.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crossbeam_utils::CachePadded;

/// Address marker for the payload entries that follow a `push_batch` token.
pub const GFIFO_PAYLOAD: u32 = 0xFFFF_0003;

/// A GFIFO entry: plain addr + val, no synchronization fields.
/// Ordering is guaranteed by the push spinlock (producers) and the
/// tail Release store / head Acquire load (consumer).
#[derive(Clone, Copy)]
pub struct GFIFOEntry {
    pub addr: u32,
    pub val:  u64,
}

/// MPSC ring buffer for the GFIFO, or with `HEADS > 1` a broadcast queue:
/// every consumer ("head") sees every entry, each at its own pace.
///
/// Multiple producers are serialized by `lock` (an atomic spinlock).
/// The lock ensures only one producer writes at a time, so tail can be
/// updated atomically after the write — no per-slot ready flag needed.
/// Each consumer thread owns one head (`h` in the `_at` methods; the plain
/// methods are head 0) and advances only that. A slot is free again once
/// the slowest head has passed it, so the producer checks room against
/// the head furthest behind; with one head that is just `head`.
/// Heads and `tail` are on separate cache lines to avoid false sharing.
/// Capacity is `DEPTH` entries; holds at most `DEPTH - 1` live entries.
pub struct GFifo<const DEPTH: usize, const HEADS: usize = 1> {
    lock: AtomicBool,
    /// Heads in use (`set_heads`); 0 means all `HEADS`.
    active: AtomicUsize,
    head: [CachePadded<AtomicUsize>; HEADS],
    tail: CachePadded<AtomicUsize>,
    /// Producer's cached slowest head.
    shadow_head: CachePadded<Cell<usize>>,
    /// Each consumer's cached tail and private (unpublished) head.
    shadow_tail: [CachePadded<Cell<usize>>; HEADS],
    local_head: [CachePadded<Cell<usize>>; HEADS],
    buf:  [GFIFOEntry; DEPTH],
}

// SAFETY: GFifo is always heap-allocated. buf access is serialized:
// producers via lock, each consumer via exclusive ownership of its head.
unsafe impl<const DEPTH: usize, const HEADS: usize> Send for GFifo<DEPTH, HEADS> {}
unsafe impl<const DEPTH: usize, const HEADS: usize> Sync for GFifo<DEPTH, HEADS> {}

impl<const DEPTH: usize, const HEADS: usize> GFifo<DEPTH, HEADS> {
    /// Index mask; `DEPTH` must be a power of two (checked at compile time).
    const MASK: usize = {
        assert!(DEPTH.is_power_of_two(), "GFifo DEPTH must be a power of two");
        assert!(HEADS >= 1, "GFifo needs a consumer head");
        DEPTH - 1
    };

    /// Consumer heads.
    pub const HEADS: usize = HEADS;

    /// Use only the first `n` heads (1..=HEADS): the others' consumers are
    /// not running, and must not hold up the producer. Call while no
    /// consumer runs, before `reset`.
    pub fn set_heads(&self, n: usize) {
        assert!((1..=HEADS).contains(&n), "GFifo has {HEADS} heads, not {n}");
        self.active.store(n, Ordering::Release);
    }

    /// Heads in use.
    #[inline]
    pub fn heads(&self) -> usize {
        if HEADS == 1 {
            return 1;
        }
        match self.active.load(Ordering::Relaxed) {
            0 => HEADS,
            n => n,
        }
    }

    /// The published head furthest behind `tail` (the producer's limit).
    #[inline]
    fn slowest_head(&self, tail: usize) -> usize {
        let mut slow = self.head[0].load(Ordering::Acquire);
        for h in &self.head[1..self.heads()] {
            let v = h.load(Ordering::Acquire);
            if tail.wrapping_sub(v) & Self::MASK > tail.wrapping_sub(slow) & Self::MASK {
                slow = v;
            }
        }
        slow
    }

    /// Capacity in entries (one slot is always kept empty).
    pub const DEPTH: usize = DEPTH;

    pub fn new() -> Self {
        // SAFETY: always heap-allocated; zeroed GFIFOEntry (u32+u64) is valid.
        unsafe { std::mem::zeroed() }
    }

    /// Entries the consumers have all published as consumed since reset
    /// (wraps). A change means the slowest consumer made progress; the
    /// consumers publish in batches, so it lags by up to one batch.
    #[inline]
    pub fn consumed(&self) -> usize {
        self.slowest_head(self.tail.load(Ordering::Acquire))
    }

    /// Returns the approximate number of entries currently in the queue
    /// (for the consumer furthest behind).
    #[inline]
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        tail.wrapping_sub(self.slowest_head(tail)) & Self::MASK
    }

    /// Entries head `h` has still to consume (approximate, as `len`).
    #[inline]
    pub fn len_at(&self, h: usize) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        tail.wrapping_sub(self.head[h].load(Ordering::Acquire)) & Self::MASK
    }

    /// Try to push an entry without blocking. Returns `false` if another
    /// producer holds the lock or the queue is full — the caller should report
    /// back-pressure and retry rather than spin here.
    ///
    /// Spinning inside the CPU's store path is what this exists to avoid. That
    /// spin runs with no interrupt servicing, so a sustained full queue starves
    /// IP7 delivery — and because the guest's own clock is driven by IP7, it
    /// also *dilates guest time*: wall-clock advances while guest-visible time
    /// does not. Any guest-side benchmark then reports inflated throughput, and
    /// inflated most for whatever configuration spins most. Returning `false`
    /// lets the bus write report `BUS_BUSY` (== `EXEC_RETRY`), so the CPU leaves
    /// the store, re-enters `step()` — sampling interrupts in `step_preamble!`
    /// — and re-dispatches the same instruction. Nothing is lost by not making
    /// progress here: if the queue is full the CPU cannot retire this store
    /// anyway.
    #[inline]
    pub fn try_push(&self, addr: u32, val: u64) -> bool {
        // Acquire the spinlock — uncontested in the common case (one active
        // producer: IRIX only drives DMA for pixmap blits, never while the CPU
        // is writing REX3 registers), so a failure here is rare.
        if self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            return false;
        }
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = tail.wrapping_add(1) & Self::MASK;
        let mut cached_head = self.shadow_head.get();
        if next_tail == cached_head {
            cached_head = self.slowest_head(tail);
            self.shadow_head.set(cached_head);
            if next_tail == cached_head {
                // Full — release the lock and let the caller retry once the
                // consumer has drained something.
                self.lock.store(false, Ordering::Release);
                return false;
            }
        }
        // SAFETY: we hold the lock; no other producer touches this slot.
        unsafe {
            let slot = self.buf.as_ptr().add(tail) as *mut GFIFOEntry;
            (*slot).addr = addr;
            (*slot).val  = val;
        }
        // Release: consumer's Acquire on tail sees the slot write above.
        self.tail.store(next_tail, Ordering::Release);
        self.lock.store(false, Ordering::Release);
        true
    }

    /// Push two consecutive register writes as one atomic unit.
    ///
    /// A 64-bit store to REX3 is two register writes, and IRIX/GL issues them
    /// constantly — coordinate pairs, colour pairs, Bresenham terms. Pushing
    /// both under one lock acquisition rather than two costs three atomics
    /// instead of six and skips a second trip through the `write32` register
    /// match.
    ///
    /// **It also fixes a real bug.** The old path called `write32` twice and
    /// returned the second one's status, so a queue that filled between them
    /// left the first word pushed and still reported `BUS_BUSY` — and the CPU
    /// re-executes the *whole* store on retry, pushing that first word a second
    /// time. Duplicated register writes into the GFIFO, exactly what
    /// `try_push`'s "commit no other state first" rule exists to prevent. Here
    /// the capacity check covers both slots before either is written, so the
    /// pair either lands completely or not at all.
    #[inline]
    pub fn try_push2(&self, addr0: u32, val0: u64, addr1: u32, val1: u64) -> bool {
        if self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            return false;
        }
        let tail = self.tail.load(Ordering::Relaxed);
        let next = tail.wrapping_add(1) & Self::MASK;
        let next2 = tail.wrapping_add(2) & Self::MASK;
        // Room for BOTH before writing either: a partial push is what the old
        // two-`write32` path got wrong.
        let mut cached_head = self.shadow_head.get();
        if next == cached_head || next2 == cached_head {
            cached_head = self.slowest_head(tail);
            self.shadow_head.set(cached_head);
            if next == cached_head || next2 == cached_head {
                self.lock.store(false, Ordering::Release);
                return false;
            }
        }
        // SAFETY: we hold the lock; no other producer touches these slots.
        unsafe {
            let slot0 = self.buf.as_ptr().add(tail) as *mut GFIFOEntry;
            (*slot0).addr = addr0;
            (*slot0).val  = val0;
            let slot1 = self.buf.as_ptr().add(next) as *mut GFIFOEntry;
            (*slot1).addr = addr1;
            (*slot1).val  = val1;
        }
        // One Release publishes both slots: the consumer's Acquire on tail
        // orders it after every write above.
        self.tail.store(next2, Ordering::Release);
        self.lock.store(false, Ordering::Release);
        true
    }

    /// Push `entries` (address, value) as one unit: all of them or, when the
    /// lock is taken or there is not room for all, none (`false`). One lock
    /// and one Release of `tail` for the lot.
    pub fn try_push_slice(&self, entries: &[(u32, u64)]) -> bool {
        let n = entries.len();
        if n == 0 {
            return true;
        }
        debug_assert!(n < DEPTH);
        if self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            return false;
        }
        let tail = self.tail.load(Ordering::Relaxed);
        // Free slots: one stays empty to tell full from empty.
        let room = |head: usize| head.wrapping_sub(tail).wrapping_sub(1) & Self::MASK;
        if room(self.shadow_head.get()) < n {
            let head = self.slowest_head(tail);
            self.shadow_head.set(head);
            if room(head) < n {
                self.lock.store(false, Ordering::Release);
                return false;
            }
        }
        for (i, &(addr, val)) in entries.iter().enumerate() {
            // SAFETY: we hold the lock; these slots are free (checked above).
            unsafe {
                let slot = self.buf.as_ptr().add((tail + i) & Self::MASK) as *mut GFIFOEntry;
                (*slot).addr = addr;
                (*slot).val = val;
            }
        }
        self.tail.store((tail + n) & Self::MASK, Ordering::Release);
        self.lock.store(false, Ordering::Release);
        true
    }

    /// Push a batch token followed by its payload words, as one atomic unit.
    ///
    /// The token carries the count; the `vals.len()` entries after it carry the
    /// data, and the consumer streams them into `ctx.hostrw[]`. Payload entries
    /// reuse `GFIFO_PAYLOAD` as their address so a stray read of one outside a
    /// batch is inert rather than being mistaken for a register write.
    ///
    /// Capacity for token + payload is checked before a single slot is written,
    /// so this is all-or-nothing in the same way `try_push2` is: the DMA worker
    /// has no EXEC_RETRY, and a partial commit followed by a retry would
    /// duplicate the prefix.
    ///
    /// Blocks rather than reporting busy. The batch may be larger than the
    /// queue, so "wait for room" is the only workable contract; callers are the
    /// DMA worker, which has nothing better to do, never the CPU store path.
    pub fn push_batch(&self, token: u32, count_val: u64, vals: &[u64]) {
        let need = vals.len() + 1;
        assert!(need < DEPTH, "batch of {} exceeds GFIFO capacity", vals.len());
        loop {
            if self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
                std::hint::spin_loop();
                continue;
            }
            let tail = self.tail.load(Ordering::Relaxed);
            let head = self.slowest_head(tail);
            self.shadow_head.set(head);
            // Free slots, keeping the one-empty-slot invariant the ring uses to
            // distinguish full from empty.
            let used = tail.wrapping_sub(head) & Self::MASK;
            let free = Self::MASK - used;
            if free < need {
                // Not enough room yet — drop the lock so the consumer can drain.
                self.lock.store(false, Ordering::Release);
                std::hint::spin_loop();
                continue;
            }
            // SAFETY: we hold the lock and have verified capacity for all
            // `need` slots; no other producer touches them.
            unsafe {
                let slot = self.buf.as_ptr().add(tail) as *mut GFIFOEntry;
                (*slot).addr = token;
                (*slot).val  = count_val;
                let mut idx = tail;
                for &v in vals {
                    idx = idx.wrapping_add(1) & Self::MASK;
                    let slot = self.buf.as_ptr().add(idx) as *mut GFIFOEntry;
                    (*slot).addr = GFIFO_PAYLOAD;
                    (*slot).val  = v;
                }
            }
            let new_tail = tail.wrapping_add(need) & Self::MASK;
            // One Release publishes the token and every payload slot.
            self.tail.store(new_tail, Ordering::Release);
            self.lock.store(false, Ordering::Release);
            return;
        }
    }

    /// Push an entry, spinning until it fits. Safe to call from multiple
    /// producers concurrently.
    ///
    /// For callers with no way to report back-pressure: shutdown sentinels, and
    /// MC's VDMA worker thread, which has no EXEC_RETRY mechanism of its own.
    /// The CPU store path uses `try_push` instead — see its doc comment.
    #[inline]
    pub fn push(&self, addr: u32, val: u64) {
        while !self.try_push(addr, val) {
            std::hint::spin_loop();
        }
    }

    /// Peek at the next entry without advancing head. Returns `None` if empty.
    /// Must be called from the consumer thread only.
    #[inline]
    pub fn peek(&self) -> Option<(u32, u64)> {
        self.peek_at(0)
    }

    /// `peek` for consumer head `h` (called only from that head's thread).
    #[inline]
    pub fn peek_at(&self, h: usize) -> Option<(u32, u64)> {
        let head = self.local_head[h].get();
        let mut cached_tail = self.shadow_tail[h].get();
        if head == cached_tail {
            cached_tail = self.tail.load(Ordering::Acquire);
            self.shadow_tail[h].set(cached_tail);
            if head == cached_tail {
                return None;
            }
        }
        // SAFETY: consumer owns head; no producer touches a slot before tail is published,
        // and our Acquire on tail pairs with the producer's Release store.
        let slot = unsafe { &*self.buf.as_ptr().add(head) };
        Some((slot.addr, slot.val))
    }

    /// The entry `n` places behind the one `peek_at(h)` returns, if the
    /// producer has published that far. Called only from head `h`'s thread;
    /// nothing is consumed.
    #[inline]
    pub fn peek_nth_at(&self, h: usize, n: usize) -> Option<(u32, u64)> {
        let head = self.local_head[h].get();
        let tail = self.tail.load(Ordering::Acquire);
        if tail.wrapping_sub(head) & Self::MASK <= n {
            return None;
        }
        // SAFETY: as in peek_at: the slot is published, and only this head's
        // thread reads behind its own head.
        let slot = unsafe { &*self.buf.as_ptr().add(head.wrapping_add(n) & Self::MASK) };
        Some((slot.addr, slot.val))
    }

    /// Advance head past the current entry after it has been fully processed.
    /// Must follow a successful `peek()`.
    #[inline]
    /// Consume up to `n` payload entries that follow a batch token, copying
    /// their values into `dst`. Returns how many were taken.
    ///
    /// The producer wrote the token and all of its payload under one lock with
    /// a single Release on tail, so once the token is visible every payload
    /// entry behind it is too — this never has to wait.
    /// Test hook: pretend only `t` entries have been published yet.
    ///
    /// `push_batch` publishes a whole batch with one Release, so a consumer
    /// that sees the token always sees the payload too — which makes it
    /// impossible to test the drain loop's mid-batch behaviour honestly
    /// without this. Not compiled into a normal build.
    #[cfg(test)]
    pub fn rewind_tail_for_test(&self, t: usize) {
        self.tail.store(t & Self::MASK, Ordering::Release);
        for st in &self.shadow_tail {
            st.set(t & Self::MASK);
        }
    }

    /// Test hook: publish up to `t` entries, undoing `rewind_tail_for_test`.
    #[cfg(test)]
    pub fn restore_tail_for_test(&self, t: usize) {
        self.tail.store(t & Self::MASK, Ordering::Release);
    }

    pub fn drain_payload<D: std::ops::IndexMut<usize, Output = u64> + ?Sized>(&self, n: usize, dst: &mut D) -> usize {
        self.drain_payload_at(0, n, dst)
    }

    /// `drain_payload` for consumer head `h`.
    pub fn drain_payload_at<D: std::ops::IndexMut<usize, Output = u64> + ?Sized>(&self, h: usize, n: usize, dst: &mut D) -> usize {
        // `local_head` still points at the token: the consumer loop peeks and
        // only calls `consume()` after process_register returns. Step over it
        // so we start at the first payload slot, and leave it stepped — the
        // caller's `consume()` then retires the last payload entry instead of
        // the token, keeping the head advance exactly `1 + taken` overall.
        let mut head = self.local_head[h].get().wrapping_add(1) & Self::MASK;
        let mut tail = self.tail.load(Ordering::Acquire);
        let mut taken = 0;
        // Drain EXACTLY the promised count. The token says N words follow, and
        // `push_batch` publishes the token and all N under one Release, so they
        // are guaranteed to be there — but `tail` is sampled once and a batch
        // is far bigger than the 64-entry head-publish interval, so a single
        // sample can sit mid-batch. Returning short there would silently drop
        // the rest of an image: the caller has no way to tell a truncated
        // drain from a complete one, and the pixels are simply gone.
        //
        // Re-sample `tail` instead of stopping. A payload entry that is not yet
        // visible is only a matter of waiting for the producer's Release store,
        // which has already happened logically — this cannot deadlock, because
        // `push_batch` publishes the whole batch before it ever returns.
        while taken < n {
            if head == tail {
                tail = self.tail.load(Ordering::Acquire);
                if head == tail {
                    std::hint::spin_loop();
                    continue;
                }
            }
            // SAFETY: consumer owns head; the slot was published with the token.
            let slot = unsafe { &*self.buf.as_ptr().add(head) };
            // There is no legitimate non-payload slot inside a batch:
            // `push_batch` writes the token and all N payload entries under
            // one lock and publishes them with a single Release. Anything else
            // here means the token/payload pairing is broken and pixel data is
            // already being lost, so say so loudly rather than absorbing it.
            // Still consumes the slot: stopping short would desync the head
            // from the count the caller was promised.
            #[cfg(feature = "developer")]
            if slot.addr != GFIFO_PAYLOAD {
                eprintln!("!!!!!!!!!! GFIFO BATCH CORRUPTION !!!!!!!!!! slot {taken} of {n} \
has addr={:#010x}, expected GFIFO_PAYLOAD ({:#010x}) — token/payload pairing is broken \
and pixel data is being lost", slot.addr, GFIFO_PAYLOAD);
            }
            dst[taken] = slot.val;
            taken += 1;
            head = head.wrapping_add(1) & Self::MASK;
        }
        self.shadow_tail[h].set(tail);
        // Rewind by one: the caller's consume() advances past the final entry.
        self.local_head[h].set(head.wrapping_sub(1) & Self::MASK);
        taken
    }

    #[inline]
    pub fn consume(&self) {
        self.consume_at(0)
    }

    /// `consume` for consumer head `h`.
    #[inline]
    pub fn consume_at(&self, h: usize) {
        let head = self.local_head[h].get();
        let next_head = head.wrapping_add(1) & Self::MASK;
        self.local_head[h].set(next_head);
        // Publish every 64 entries to massively reduce cache line invalidations
        // — but ALSO publish immediately whenever this consume just drained
        // the queue to empty (next_head == tail), regardless of the
        // batching boundary. External observers (is_empty/len — REX3's own
        // `busy_or_val!` register-read gate among them) read `head` directly
        // and have no way to know the consumer thread's private
        // `local_head` is already caught up; without this, consuming the
        // last entry of an otherwise-empty run (whenever that count isn't a
        // multiple of 64 — the overwhelmingly common case) leaves `head`
        // stale, so `is_empty()` keeps reporting "not empty" — and if the
        // consumer thread then exits (register_processor's own `else`
        // branch normally catches this on its *next* loop iteration via
        // `flush_head()`, but a `stop()`-driven `GFIFO_EXIT` can end the
        // loop before that next iteration ever runs) the stale `head` is
        // permanent: nothing ever re-derives it, and every future
        // busy_or_val!-gated register read reports busy forever even though
        // the queue is, and has been, genuinely empty. (Found live: `rex
        // status` showing `DRAW BUSY: no` with `GFIFO: 1/65536` — gfxbusy
        // correctly cleared, but `head` never got the memo.)
        let tail = self.tail.load(Ordering::Acquire);
        if next_head & 63 == 0 || next_head == tail {
            self.head[h].store(next_head, Ordering::Release);
        }
    }

    #[inline]
    pub fn flush_head(&self) {
        self.flush_head_at(0)
    }

    /// `flush_head` for consumer head `h`.
    #[inline]
    pub fn flush_head_at(&self, h: usize) {
        self.head[h].store(self.local_head[h].get(), Ordering::Release);
    }

    /// Reconcile the published `tail` from a reader thread, under the producer
    /// lock.
    ///
    /// `try_push` publishes `tail` on every push today, so this is a no-op in
    /// the current topology — but it is the hook a deferred/batched tail needs,
    /// and taking the lock is what makes it safe to call from a thread that is
    /// neither the producer nor the consumer. See
    /// `clanker/rex3/gfifo-batching-constraints.md`: a batched producer must
    /// publish before any `busy_or_val!` or STATUS read, or the reader sees an
    /// emptier queue than reality and skips the retry it owed.
    ///
    /// Returns `false` if the lock was contended — the caller should treat that
    /// as "busy, retry" rather than spin, for the same reason `try_push` does:
    /// spinning inside the CPU's load path starves IP7 and dilates guest time.
    #[inline]
    pub fn publish_tail(&self) -> bool {
        if self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            return false;
        }
        // A producer-local tail would be published here. With eager publication
        // the authoritative `tail` is already current; re-storing it under the
        // lock is harmless and keeps this the single place that changes when
        // batching lands.
        let tail = self.tail.load(Ordering::Relaxed);
        self.tail.store(tail, Ordering::Release);
        self.lock.store(false, Ordering::Release);
        true
    }

    /// True when no entries are pending for any consumer.
    ///
    /// Reads the *published* `head`, which the consumer advances in batches of
    /// 64 (plus immediately on drain-to-empty, see `consume`). So a `true` here
    /// is authoritative — the consumer publishes the moment it empties the ring
    /// — while a `false` may be up to 63 entries pessimistic mid-drain. That
    /// direction is the safe one for every caller: it over-reports busy, never
    /// under-reports.
    #[inline]
    pub fn is_empty(&self) -> bool {
        let tail = self.tail.load(Ordering::Acquire);
        self.head[..self.heads()].iter().all(|h| h.load(Ordering::Acquire) == tail)
    }

    /// True when head `h` has consumed everything published.
    #[inline]
    pub fn is_empty_at(&self, h: usize) -> bool {
        self.head[h].load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    /// Force the queue to the empty state, discarding any pending entries.
    /// Only safe to call with the consumer thread stopped (checkpoint
    /// restore's own contract — `restore_live_checkpoint` always calls
    /// `self.stop()` first) — this bypasses the normal producer lock/
    /// consumer head-ownership discipline entirely, so a live producer or
    /// consumer racing this call would corrupt the ring buffer's invariants.
    #[inline]
    pub fn reset(&self) {
        let tail = self.tail.load(Ordering::Relaxed);
        for h in 0..HEADS {
            self.head[h].store(tail, Ordering::Release);
            self.local_head[h].set(tail);
            self.shadow_tail[h].set(tail);
        }
        self.shadow_head.set(tail);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_head_sees_every_entry() {
        let f: Box<GFifo<16, 2>> = Box::new(GFifo::new());
        for i in 0..5 {
            assert!(f.try_push(i, i as u64 * 10));
        }
        for h in 0..2 {
            for i in 0..5 {
                assert_eq!(f.peek_at(h), Some((i, i as u64 * 10)));
                f.consume_at(h);
            }
            assert_eq!(f.peek_at(h), None);
        }
        assert!(f.is_empty());
    }

    #[test]
    fn peek_nth_looks_behind_the_head_across_the_wrap() {
        let f: Box<GFifo<8>> = Box::new(GFifo::new());
        for i in 0..5 {
            assert!(f.try_push(i, 0));
        }
        for _ in 0..5 {
            f.consume();
        }
        // The next entries wrap past the end of the buffer.
        for i in 10..16 {
            assert!(f.try_push(i, i as u64));
        }
        assert_eq!(f.peek_nth_at(0, 0), f.peek());
        assert_eq!(f.peek_nth_at(0, 5), Some((15, 15)));
        assert_eq!(f.peek_nth_at(0, 6), None, "not pushed");
        f.consume();
        assert_eq!(f.peek_nth_at(0, 4), Some((15, 15)));
    }

    #[test]
    fn the_slowest_head_limits_the_producer() {
        let f: Box<GFifo<8, 2>> = Box::new(GFifo::new());
        for i in 0..7 {
            assert!(f.try_push(i, 0));
        }
        assert!(!f.try_push(7, 0), "full: one slot stays empty");
        // Head 0 drains everything; head 1 has not moved, so still full.
        for _ in 0..7 {
            f.consume_at(0);
        }
        f.flush_head_at(0);
        assert!(f.is_empty_at(0) && !f.is_empty());
        assert_eq!(f.len(), 7);
        assert!(!f.try_push(7, 0));
        // Head 1 takes two: two slots free up.
        for _ in 0..2 {
            assert!(f.peek_at(1).is_some());
            f.consume_at(1);
        }
        f.flush_head_at(1);
        assert!(f.try_push(7, 0) && f.try_push(8, 0) && !f.try_push(9, 0));
        assert_eq!(f.len_at(0), 2);
        assert_eq!(f.len_at(1), 7);
    }

    #[test]
    fn unused_heads_do_not_hold_up_the_producer() {
        let f: Box<GFifo<8, 2>> = Box::new(GFifo::new());
        f.set_heads(1);
        for round in 0..3 {
            for i in 0..7 {
                assert!(f.try_push(i, round), "round {round} entry {i}");
            }
            assert!(!f.try_push(7, 0));
            for _ in 0..7 {
                f.consume_at(0);
            }
            f.flush_head_at(0);
            assert!(f.is_empty());
        }
    }

    #[test]
    fn batches_reach_both_heads() {
        let f: Box<GFifo<64, 2>> = Box::new(GFifo::new());
        f.push_batch(0x42, 3, &[1, 2, 3]);
        for h in 0..2 {
            assert_eq!(f.peek_at(h), Some((0x42, 3)));
            let mut out = [0u64; 3];
            assert_eq!(f.drain_payload_at(h, 3, &mut out[..]), 3);
            f.consume_at(h);
            assert_eq!(out, [1, 2, 3]);
            assert_eq!(f.peek_at(h), None);
        }
        assert!(f.is_empty());
    }

    #[test]
    fn two_consumer_threads_agree() {
        let f: std::sync::Arc<GFifo<256, 2>> = std::sync::Arc::from(Box::new(GFifo::new()));
        const N: u64 = 100_000;
        let consumers: Vec<_> = (0..2)
            .map(|h| {
                let f = f.clone();
                std::thread::spawn(move || {
                    let (mut sum, mut n) = (0u64, 0u64);
                    while n < N {
                        if let Some((_, v)) = f.peek_at(h) {
                            assert_eq!(v, n, "head {h} out of order");
                            sum += v;
                            n += 1;
                            f.consume_at(h);
                        } else {
                            std::hint::spin_loop();
                        }
                    }
                    f.flush_head_at(h);
                    sum
                })
            })
            .collect();
        for i in 0..N {
            f.push(1, i);
        }
        let sums: Vec<u64> = consumers.into_iter().map(|c| c.join().unwrap()).collect();
        assert_eq!(sums, vec![N * (N - 1) / 2; 2]);
        assert!(f.is_empty());
    }
}
