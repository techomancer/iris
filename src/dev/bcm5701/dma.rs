//! The descriptor engines: the send ring, the standard and jumbo receive
//! rings and the return ring, the status block, and the host coalescing that
//! decides when the driver is told.
//!
//! Descriptors and the status block are the chip's big-endian internal image,
//! reaching the bus through the MODE_CTL swap controls; frame data uses the
//! frame-data controls. What the host sees of those bytes is the bridge's
//! business ([`PciDma`]).

use std::time::{Duration, Instant};

use super::state::{reg, Refused, State};
use crate::pci::{PciDma, PciDmaStatus};

/// The largest frame the engines carry, transmit or receive. The part does
/// jumbo frames; the figure matches what the other Origin adapters carry so a
/// guest configured for jumbo behaves the same whichever card it has. It is a
/// model limit, not the hardware's.
pub const MAX_FRAME: usize = 9600;

const SRAM_SEND_RING_RCB: u32 = 0x0100;
const SRAM_RX_RETURN_RCB: u32 = 0x0200;
/// `nicNoMoreRXBDs`, the one statistics counter the model keeps: the number of
/// times the chip ran out of receive descriptors.
const SRAM_STATS_NO_MORE_RXBDS: u32 = 0x0898;
const RCB_HOSTADDR_HI: u32 = 0x00;
const RCB_HOSTADDR_LO: u32 = 0x04;
const RCB_MAXLEN_FLAGS: u32 = 0x08;
const RCB_FLAG_EXT_RX_ENABLE: u32 = 1;
const RCB_FLAG_DISABLE_RING: u32 = 2;

const TXBD_SIZE: u64 = 16;
// Offsets in the descriptors' little-endian layout; a 16-bit field's
// image offset is its offset xor 2.
const TXBD_FLAGS: usize = 0x08;
const TXBD_LEN: usize = 0x0a;
const TXBD_MSS: usize = 0x0e;
const TXBDFLAG_TCP_UDP_CKSUM: u16 = 0x0001;
const TXBDFLAG_IP_CKSUM: u16 = 0x0002;
const TXBDFLAG_END: u16 = 0x0004;
const TXBDFLAG_INSERT_SRC_ADDR: u16 = 0x1000;
const TXBDFLAG_CHOOSE_SRC_ADDR: u16 = 0x6000;
const TXBDFLAG_SRC_ADDR_SHIFT: u16 = 13;

const RXBD_SIZE: u64 = 32;
const RXBD_IDX_LEN: usize = 0x08;
const RXBD_TYPE_FLAGS: usize = 0x0c;
const RXBD_CSUM: usize = 0x10;
const RXBD_OPAQUE: usize = 0x1c;
const RXBDFLAG_END: u32 = 0x0004;
const RXBDFLAG_JUMBO_RING: u32 = 0x0020;
const RXBDFLAG_IP_CKSUM_OK: u32 = 0x1000;
const RXBDFLAG_TCP_UDP_CKSUM_OK: u32 = 0x2000;
const RXBDFLAG_TCP: u32 = 0x4000;
const MODE_RX_NO_PSEUDO: u32 = 1 << 23;
const MODE_NO_FRAME_CRACK: u32 = 1 << 9;

/// The standard ring holds a fixed number of descriptors. Its control block's
/// length is the size of the buffer each descriptor points at, an MTU rather
/// than a count, while the return ring's really is a ring length.
const RX_STD_RING_LEN: u16 = 512;
const RX_JUMBO_RING_LEN: u16 = 256;
/// The chip copies a quarter of each producer ring into its own memory
/// ahead of use [`57XX-PG105-R`, printed page 118]; Linux tg3 sizes the
/// same caches at 128 and 64 descriptors for this chip.
const RX_STD_BD_CACHE: u16 = RX_STD_RING_LEN / 4;
const RX_JUMBO_BD_CACHE: u16 = RX_JUMBO_RING_LEN / 4;
const FCS_LEN: u32 = 4;
const MIN_FRAME: usize = 60;

const STATUS_BLK_SIZE: usize = 0x50;
const STATFLAG_UPDATED: u32 = 1;
const STATUS_RX_PROD_OFF: usize = 16;
const STATUS_TX_CONS_OFF: usize = 18;
const STATUS_TAG_OFF: usize = 4;
const STATUS_RX_STD_CONS_IMAGE_OFF: usize = 8;
const STATUS_RX_JUMBO_CONS_IMAGE_OFF: usize = 10;
const HCC_MODE_ENABLE: u32 = 2;

/// Both coalescing tick budgets are programmed in microseconds.
const COAL_TICK: Duration = Duration::from_micros(1);
/// A status DMA the guest's address refused is retried no slower than this.
const COAL_RETRY_MAX: Duration = Duration::from_millis(250);
/// Completions the arrival line can hold; a full line refuses frames.
const RX_LATENCY_DEPTH: usize = 256;
/// The default minimum between a frame arriving and its completion.
pub const RX_ARRIVAL_LATENCY_DEFAULT: Duration = Duration::from_micros(200);
/// The default minimum in processor cycles: how far the processor that rang
/// the send doorbell must have run past it before a frame arriving after it is
/// reported. See [`HeldCompletion`].
pub const RX_ARRIVAL_CYCLES_DEFAULT: u64 = 20_000;
/// How soon a completion whose host-time deadline has passed looks again at
/// whether the sending processor has run far enough.
const RX_CYCLES_RECHECK: Duration = Duration::from_micros(50);

/// A completion in guest memory whose report waits for its frame's arrival.
///
/// The arrival floor stands for the time a reply takes on a real network, so
/// that it never lands before the transmit that provoked it has finished. On
/// the host the reply comes back almost at once, and host time alone does not
/// hold it back far enough: the guest's processor runs slower than the part
/// it models, and while the host has its thread descheduled it does not run
/// at all, so a deadline in host time can pass with the guest still inside
/// the send path that rang the doorbell. An echo reply reported then is
/// counted by IRIX's ICMP input yet never reaches the `ping` still sending
/// the request. So the report also waits until every processor that rang
/// the doorbell before the frame arrived has run `rx_floor_cycles` past its
/// last ring. A processor parked in its idle loop still counts cycles, so
/// nothing waits long on an idle guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HeldCompletion {
    /// The host-time deadline.
    pub(super) due: Instant,
    /// The producer index the completion publishes.
    pub(super) prod: u16,
    /// Each processor's last send doorbell before the frame arrived.
    pub(super) after: Doorbells,
}

/// Each processor's cycle count when it last rang the send doorbell, by CPU
/// number.
pub(super) type Doorbells = [Option<u64>; crate::cpu::smp::MAX_CPUS];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ring {
    Standard,
    Jumbo,
}

/// The mailboxes whose index is range-checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mailbox {
    TxProducer,
    RxStandardProducer,
    RxJumboProducer,
    RxReturnConsumer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DmaWhat {
    TxDescriptor,
    TxPayload,
    RxDescriptor,
    RxPayload,
    RxCompletion,
    StatusBlock,
}

/// The timers the engines keep, ordered by when they fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Timer {
    PhySettle,
    BufferManager,
    RxCoalescing,
    TxCoalescing,
    RxArrival,
}

/// What the receive path did with a frame the backend offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RxOutcome {
    /// Delivered, filtered or dropped: the backend may discard it.
    Consumed,
    /// Not taken; the frame stays at the head of the backend's queue until
    /// the guest makes room.
    Deferred,
}

/// A bus master with nowhere to go, until the bridge connects one.
pub(super) struct NullDma;
impl PciDma for NullDma {
    fn dma_read(&self, _addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        buf.fill(0);
        PciDmaStatus::MasterAbort
    }
    fn dma_write(&self, _addr: u64, _data: &[u8]) -> PciDmaStatus {
        PciDmaStatus::MasterAbort
    }
}

fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes(b[off..off + 4].try_into().unwrap())
}

/// A 16-bit field of the chip's image sits in the half of its 32-bit word
/// the big-endian structure puts it in.
fn ld16(b: &[u8], off: usize) -> u16 {
    let o = off ^ 2;
    u16::from_be_bytes([b[o], b[o + 1]])
}

fn st32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_be_bytes());
}

fn st16(b: &mut [u8], off: usize, v: u16) {
    let o = off ^ 2;
    b[o..o + 2].copy_from_slice(&v.to_be_bytes());
}

/// The one's-complement sum of a span, folded.
fn ones_sum(buf: &[u8], seed: u32) -> u16 {
    let mut sum = seed;
    for pair in buf.chunks(2) {
        sum += u32::from(pair[0]) << 8 | u32::from(*pair.get(1).unwrap_or(&0));
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    sum as u16
}

/// Fill in the IP header and/or TCP and UDP checksums of an untagged IPv4
/// frame. A fragment keeps its transport checksum.
fn checksum_calculate(frame: &mut [u8], ip: bool, transport: bool) {
    if frame.len() < 14 + 20 || frame[14] >> 4 != 4 {
        return;
    }
    let ihl = usize::from(frame[14] & 0xf) * 4;
    if ihl < 20 || frame.len() < 14 + ihl {
        return;
    }
    if ip {
        frame[24] = 0;
        frame[25] = 0;
        let sum = !ones_sum(&frame[14..14 + ihl], 0);
        frame[24..26].copy_from_slice(&sum.to_be_bytes());
    }
    if u16::from_be_bytes([frame[20], frame[21]]) & 0x3fff != 0 {
        return;
    }
    let total = usize::from(u16::from_be_bytes([frame[16], frame[17]]));
    if total < ihl {
        return;
    }
    let data_len = (total - ihl).min(frame.len() - 14 - ihl);
    let proto = frame[23];
    let at = match proto {
        6 if transport && data_len >= 20 => 16,
        17 if transport && data_len >= 8 => 6,
        _ => return,
    };
    let start = 14 + ihl;
    frame[start + at] = 0;
    frame[start + at + 1] = 0;
    let pseudo = ones_pseudo(&frame[26..34], proto, data_len);
    let sum = !ones_sum(&frame[start..start + data_len], pseudo);
    frame[start + at..start + at + 2].copy_from_slice(&sum.to_be_bytes());
}

fn ones_pseudo(addresses: &[u8], proto: u8, len: usize) -> u32 {
    addresses.chunks(2).map(|p| u32::from(p[0]) << 8 | u32::from(p[1])).sum::<u32>() + u32::from(proto) + len as u32
}

/// Finish the checksums the driver left to the card. The lengths in the frame
/// were composed by the guest, so a header shorter than its own minimum, or
/// lengths reaching past what arrived, decline the offload rather than walk
/// off the buffer.
fn tx_offload_checksum(frame: &mut [u8], flags: u16) {
    if frame.len() < 14 + 20 || frame[12..14] != [0x08, 0x00] || frame[14] >> 4 != 4 {
        return;
    }
    let ihl = usize::from(frame[14] & 0xf) * 4;
    let total = usize::from(u16::from_be_bytes([frame[16], frame[17]]));
    if ihl < 20 || total < ihl || 14 + total > frame.len() {
        return;
    }
    checksum_calculate(frame, flags & TXBDFLAG_IP_CKSUM != 0, flags & TXBDFLAG_TCP_UDP_CKSUM != 0);
}

struct LsoLayout {
    ihl: usize,
    thl: usize,
    hdr: usize,
    payload: usize,
}

/// Whether `frame` is an unfragmented IPv4 TCP frame that `mss` can cut up.
fn lso_layout(frame: &[u8], mss: usize) -> Option<LsoLayout> {
    if frame.len() < 14 + 20 + 20 || frame[12..14] != [0x08, 0x00] || frame[14] >> 4 != 4 || frame[23] != 6 {
        return None;
    }
    let ihl = usize::from(frame[14] & 0xf) * 4;
    if ihl < 20 || 14 + ihl + 20 > frame.len() {
        return None;
    }
    let ip_len = usize::from(u16::from_be_bytes([frame[16], frame[17]]));
    if u16::from_be_bytes([frame[20], frame[21]]) & 0x3fff != 0 || ip_len < ihl + 20 || 14 + ip_len != frame.len() {
        return None;
    }
    let thl = usize::from(frame[14 + ihl + 12] >> 4) * 4;
    let hdr = 14 + ihl + thl;
    if thl < 20 || ihl + thl > ip_len || mss == 0 || hdr + mss > MAX_FRAME {
        return None;
    }
    let payload = ip_len - ihl - thl;
    if payload <= mss {
        return None;
    }
    Some(LsoLayout { ihl, thl, hdr, payload })
}

/// The checksums of an arriving frame, as the part reports them: the sums it
/// computed rather than a verdict, a field that checks out summing to 0xffff.
/// Anything not plainly IPv4 TCP or UDP is left unclaimed, because reporting
/// a sum for something never summed would have the stack skip its own pass.
fn rx_offload_checksum(frame: &[u8], mode: u32, flags: &mut u32) -> u32 {
    if mode & MODE_NO_FRAME_CRACK != 0 {
        return 0;
    }
    if frame.len() < 34 || frame[12..14] != [0x08, 0x00] || frame[14] >> 4 != 4 {
        return 0;
    }
    let ihl = usize::from(frame[14] & 0xf) * 4;
    let total = usize::from(u16::from_be_bytes([frame[16], frame[17]]));
    if ihl < 20 || total < ihl || 14 + total > frame.len() {
        return 0;
    }
    let ip_sum = ones_sum(&frame[14..14 + ihl], 0);
    if ip_sum == 0 || ip_sum == 0xffff {
        *flags |= RXBDFLAG_IP_CKSUM_OK;
    }
    let ip_only = u32::from(ip_sum) << 16;
    let proto = frame[23];
    if proto != 6 && proto != 17 {
        return ip_only;
    }
    if proto == 6 {
        *flags |= RXBDFLAG_TCP;
    }
    if u16::from_be_bytes([frame[20], frame[21]]) & 0x3fff != 0 {
        return ip_only;
    }
    let l4len = total - ihl;
    if (proto == 6 && l4len < 20) || (proto == 17 && l4len < 8) {
        return ip_only;
    }
    let mut transport_checksum_present = true;
    if proto == 6 {
        let tcp_hlen = usize::from(frame[14 + ihl + 12] >> 4) * 4;
        if tcp_hlen < 20 || tcp_hlen > l4len {
            return ip_only;
        }
    } else {
        let udp_len = usize::from(u16::from_be_bytes([frame[14 + ihl + 4], frame[14 + ihl + 5]]));
        if udp_len != l4len {
            return ip_only;
        }
        transport_checksum_present = frame[14 + ihl + 6..14 + ihl + 8] != [0, 0];
    }
    let pseudo = if mode & MODE_RX_NO_PSEUDO == 0 { ones_pseudo(&frame[26..34], proto, l4len) } else { 0 };
    let l4_sum = ones_sum(&frame[14 + ihl..14 + ihl + l4len], pseudo);
    if l4_sum == 0xffff && transport_checksum_present && mode & MODE_RX_NO_PSEUDO == 0 {
        *flags |= RXBDFLAG_TCP_UDP_CKSUM_OK;
    }
    ip_only | u32::from(l4_sum)
}

/// Where a producer ring's fetch index stands after the chip fetches what it
/// may. `used` is the next descriptor a frame takes and `fetched` the next
/// the chip copies, so the cache holds those in between. A producer written
/// back into the cached span, which no driver should do, fetches nothing.
fn fetch_ahead(prod: u16, used: u16, fetched: u16, ring_len: u16, cache: u16, thresh: u32) -> u16 {
    let posted = (prod + ring_len - used) % ring_len;
    let cached = (fetched + ring_len - used) % ring_len;
    let available = posted.saturating_sub(cached);
    let room = cache.saturating_sub(cached);
    if available == 0 || room == 0 || u32::from(available) < thresh {
        return fetched;
    }
    (fetched + available.min(room)) % ring_len
}

impl State {
    /// The engine state a reset returns to zero.
    pub(super) fn reset_engines(&mut self) {
        self.tx_cons = 0;
        self.status_tx_cons = 0;
        self.status_rx_prod = 0;
        self.rx_prod_staged = 0;
        self.rx_std_cons = 0;
        self.rx_jumbo_cons = 0;
        self.rx_std_fetched = 0;
        self.rx_jumbo_fetched = 0;
        self.rx_shortage_drops = 0;
        self.rx_std_shortage_drops = 0;
        self.status_tag = 0;
        self.rx_coal_pending = 0;
        self.tx_coal_pending = 0;
        self.rx_coal_expired = false;
        self.tx_coal_expired = false;
        self.rx_coal_timer = None;
        self.tx_coal_timer = None;
        self.rx_latency.clear();
        self.tx_doorbell = [None; crate::cpu::smp::MAX_CPUS];
        self.wire.clear();
        self.wake_rx = true;
        self.purge_rx = true;
    }

    /// Whatever made the chip ready again, a posted buffer or a consumer
    /// index the driver moved, is the moment to offer the backend's waiting
    /// frame again.
    pub(super) fn retry_rx_flush(&mut self) {
        self.wake_rx = true;
    }

    // ---- host memory ----

    /// The 64-bit host address a pair of control-block words names: the high
    /// word at the lower offset, the low word at the next. With
    /// ENDIAN_WORDSWAP set, a 32-bit target access at +0 of an on-chip
    /// eight-byte field reaches its more significant word (Figure 93 of the
    /// family programmer's guide), which is the setting the drivers use, so
    /// the pair takes no further exchange. The swap-clear 32-bit case (the
    /// low word at +0) is not modelled; no driver leaves the swap clear.
    fn compose_hostaddr(&self, hi: u32, lo: u32) -> u64 {
        u64::from(hi) << 32 | u64::from(lo)
    }

    /// The chip's DMA byte-swap controls, MODE_CTL bits 5 and 4 for frame data
    /// and bits 2 and 1 for everything else. The chip is big-endian
    /// internally with a 64-bit memory: its internal byte at lane l of a
    /// doubleword (lane 0 the most significant) goes on PCI byte 7 - l with no
    /// swap, and the swaps exchange the halves and reverse bytes within them.
    /// The frame-data combinations are established by the programmer's guide;
    /// the non-frame single-swap cases are inferred. A transfer's chip-side
    /// lane is its PCI address modulo 8, an approximation: the chip's
    /// internal buffer alignment is not modelled.
    fn lane_xor(&self, frame: bool) -> usize {
        let mode = self.rf(reg::MODE_CTL);
        let (byte, word) = if frame { (0x10, 0x20) } else { (0x2, 0x4) };
        7 ^ if mode & byte != 0 { 3 } else { 0 } ^ if mode & word != 0 { 4 } else { 0 }
    }

    /// Read `len` bytes of the chip's internal image from `addr`.
    fn dma_read_image(&self, addr: u64, len: usize, frame: bool) -> Option<Vec<u8>> {
        if len == 0 {
            return Some(Vec::new());
        }
        let t = self.lane_xor(frame);
        let astart = addr & !7;
        let lead = (addr - astart) as usize;
        let span = (lead + len + 7) & !7;
        let mut raw = vec![0; span];
        if !self.dma.dma_read(astart, &mut raw).is_ok() {
            return None;
        }
        Some(
            (0..len)
                .map(|j| {
                    let p = lead + j;
                    raw[(p & !7) + ((p & 7) ^ t)]
                })
                .collect(),
        )
    }

    /// Write `data` of the chip's internal image to `addr`, touching only the
    /// lanes it covers as the byte enables do: whole doublewords as one
    /// transfer, the partial ends a byte at a time.
    fn dma_write_image(&self, addr: u64, data: &[u8], frame: bool) -> bool {
        let t = self.lane_xor(frame);
        let mut i = 0;
        while i < data.len() {
            let a = addr + i as u64;
            if a & 7 == 0 && data.len() - i >= 8 {
                let run = (data.len() - i) & !7;
                let mut raw = vec![0; run];
                for j in 0..run {
                    raw[(j & !7) + ((j & 7) ^ t)] = data[i + j];
                }
                if !self.dma.dma_write(a, &raw).is_ok() {
                    return false;
                }
                i += run;
            } else {
                let at = (a & !7) + ((a & 7) as usize ^ t) as u64;
                if !self.dma.dma_write(at, &data[i..i + 1]).is_ok() {
                    return false;
                }
                i += 1;
            }
        }
        true
    }

    // ---- status block and coalescing ----

    /// DMA the status block to the host address the driver programmed, only
    /// while host coalescing is enabled. The chip owns the block, so it is
    /// built whole each time. `raise` marks an update the driver is meant to
    /// be interrupted for. Returns whether it reached the host.
    fn post_status_block(&mut self, raise: bool) -> bool {
        if self.rf(reg::HCC_MODE) & HCC_MODE_ENABLE == 0 {
            return false;
        }
        let base = self.compose_hostaddr(self.rf(reg::HCC_STATUSBLK_ADDR_HI), self.rf(reg::HCC_STATUSBLK_ADDR_LO));
        let next_tag = self.status_tag.wrapping_add(1);
        let mut blk = [0u8; STATUS_BLK_SIZE];
        st32(&mut blk, 0, STATFLAG_UPDATED);
        st32(&mut blk, STATUS_TAG_OFF, u32::from(next_tag));
        // A producer ring's consumer is how far the chip has fetched, not
        // how far frames have used [`57XX-PG105-R`, printed pages 104, 123
        // and 132].
        blk[STATUS_RX_STD_CONS_IMAGE_OFF..][..2].copy_from_slice(&self.rx_std_fetched.to_be_bytes());
        blk[STATUS_RX_JUMBO_CONS_IMAGE_OFF..][..2].copy_from_slice(&self.rx_jumbo_fetched.to_be_bytes());
        st16(&mut blk, STATUS_TX_CONS_OFF, self.status_tx_cons);
        st16(&mut blk, STATUS_RX_PROD_OFF, self.status_rx_prod);
        if !self.dma_write_image(base, &blk, false) {
            self.refuse(Refused::DmaFault(DmaWhat::StatusBlock));
            return false;
        }
        self.status_tag = next_tag;
        // The block now carries an update the driver has not taken. The line
        // asserts while that latch is set and interrupts are unmasked, and
        // drops when the driver acknowledges through the interrupt mailbox.
        if raise {
            self.irq_pending = true;
            self.update_irq(false);
        }
        true
    }

    fn coal_arm_one(&mut self, rx: bool) {
        let ticks = self.rf(if rx { reg::HCC_RX_COAL_TICKS } else { reg::HCC_TX_COAL_TICKS });
        let deadline = (ticks != 0).then(|| self.now + COAL_TICK * ticks);
        if rx {
            self.rx_coal_expired = false;
            self.rx_coal_timer = deadline;
        } else {
            self.tx_coal_expired = false;
            self.tx_coal_timer = deadline;
        }
    }

    /// Both budgets restart together: one update reports every ring.
    fn coal_arm(&mut self) {
        self.coal_arm_one(true);
        self.coal_arm_one(false);
    }

    /// One host update. All the ring indices are in one status block, so
    /// whichever criterion fired reports every ring and restarts every count
    /// and timer.
    fn status_update(&mut self) -> bool {
        if !self.post_status_block(true) {
            return false;
        }
        self.rx_coal_pending = 0;
        self.tx_coal_pending = 0;
        self.coal_arm();
        true
    }

    /// Retry later instead of spinning on an address the guest has not
    /// repaired.
    fn coal_retry(&mut self, rx: bool) {
        let ticks = self.rf(if rx { reg::HCC_RX_COAL_TICKS } else { reg::HCC_TX_COAL_TICKS });
        let wait = (COAL_TICK * ticks.max(1)).min(COAL_RETRY_MAX);
        let deadline = Some(self.now + wait);
        if rx {
            self.rx_coal_timer = deadline;
        } else {
            self.tx_coal_timer = deadline;
        }
    }

    fn rx_coal_flush(&mut self) {
        if self.rx_coal_pending == 0 || self.status_update() {
            return;
        }
        self.coal_retry(true);
    }

    /// The receive budget ran out. It reports whatever arrived since the last
    /// update; with nothing to report it stays expired, and the next frame is
    /// reported as it lands.
    pub(super) fn rx_coal_timer_fired(&mut self) {
        self.rx_coal_timer = None;
        self.rx_coal_expired = true;
        self.rx_coal_flush();
    }

    fn tx_coal_flush(&mut self) {
        if self.tx_coal_pending == 0 || self.status_update() {
            return;
        }
        self.coal_retry(false);
    }

    pub(super) fn tx_coal_timer_fired(&mut self) {
        self.tx_coal_timer = None;
        self.tx_coal_expired = true;
        self.tx_coal_flush();
    }

    /// Count the descriptors this doorbell completed, and report them once
    /// the count criterion is met or the send budget has already run out.
    ///
    /// A doorbell that completes nothing is reported anyway, which no
    /// criterion covers: bring-up leans on that update arriving. This is a
    /// known deviation from the coalescing contract, not an established
    /// hardware behaviour.
    fn tx_coalesce(&mut self, completed: usize) {
        let max_bds = self.rf(reg::HCC_TX_MAX_COAL_BDS);
        if completed == 0 {
            self.status_update();
            return;
        }
        self.tx_coal_pending = self.tx_coal_pending.wrapping_add(completed as u16);
        if (max_bds != 0 && u32::from(self.tx_coal_pending) >= max_bds) || self.tx_coal_expired {
            self.tx_coal_flush();
        }
    }

    /// A received frame filled a return-ring descriptor.
    fn rx_coalesce(&mut self) {
        let max_bds = self.rf(reg::HCC_RX_MAX_COAL_BDS);
        self.rx_coal_pending = self.rx_coal_pending.wrapping_add(1);
        if (max_bds != 0 && u32::from(self.rx_coal_pending) >= max_bds) || self.rx_coal_expired {
            self.rx_coal_flush();
        }
    }

    pub(super) fn hcc_mode_written(&mut self) {
        if self.rf(reg::HCC_MODE) & HCC_MODE_ENABLE != 0 {
            // The budgets run from here until the first update reloads them.
            self.coal_arm();
            self.retry_rx_flush();
        } else {
            self.rx_coal_timer = None;
            self.tx_coal_timer = None;
        }
    }

    /// A budget was reprogrammed. A running timer keeps the value it was
    /// loaded with, since the chip reloads one only at a status-block update;
    /// one that is not running starts here, because nothing else would.
    pub(super) fn coal_ticks_written(&mut self, rx: bool) {
        let (expired, running) = if rx {
            (self.rx_coal_expired, self.rx_coal_timer.is_some())
        } else {
            (self.tx_coal_expired, self.tx_coal_timer.is_some())
        };
        if self.rf(reg::HCC_MODE) & HCC_MODE_ENABLE == 0 || expired || running {
            return;
        }
        self.coal_arm_one(rx);
    }

    // ---- transmit ----

    fn tx_rollback_to_frame(&mut self, frame_start: u16, completed: usize) {
        self.tx_cons = frame_start;
        if completed != 0 {
            self.status_tx_cons = frame_start;
            self.tx_coalesce(completed);
        }
    }

    /// Replace the frame's source address with a filter slot's, as the
    /// descriptor asks.
    fn tx_replace_source(&self, frame: &mut [u8], flags: u16) {
        if flags & TXBDFLAG_INSERT_SRC_ADDR == 0 || frame.len() < 12 {
            return;
        }
        let slot = u32::from((flags & TXBDFLAG_CHOOSE_SRC_ADDR) >> TXBDFLAG_SRC_ADDR_SHIFT);
        let hi = self.rf(reg::MAC_ADDR_0_HIGH + slot * reg::MAC_ADDR_STRIDE);
        let lo = self.rf(reg::MAC_ADDR_0_LOW + slot * reg::MAC_ADDR_STRIDE);
        frame[6] = (hi >> 8) as u8;
        frame[7] = hi as u8;
        frame[8..12].copy_from_slice(&lo.to_be_bytes());
    }

    /// Cut one oversized TCP buffer into segments the peer will accept. The
    /// guest hands down a single frame holding many segments' worth of
    /// payload; each piece goes out with its own copy of the headers, its own
    /// sequence number, and the finishing flags only on the last. Anything
    /// that does not look like the IPv4 TCP frame this is for is sent
    /// unchanged rather than guessed at.
    fn tx_segment(&mut self, mut frame: Vec<u8>, mss: usize, flags: u16) {
        let Some(LsoLayout { ihl, thl, hdr, payload }) = lso_layout(&frame, mss) else {
            if flags & (TXBDFLAG_IP_CKSUM | TXBDFLAG_TCP_UDP_CKSUM) != 0 {
                tx_offload_checksum(&mut frame, flags);
            }
            self.wire.push(frame);
            return;
        };
        let seq = be32(&frame, 14 + ihl + 4);
        let ip_id = u16::from_be_bytes([frame[18], frame[19]]);
        let mut off = 0;
        while off < payload {
            let piece = mss.min(payload - off);
            let mut seg = frame[..hdr].to_vec();
            seg.extend_from_slice(&frame[hdr + off..hdr + off + piece]);
            seg[16..18].copy_from_slice(&((ihl + thl + piece) as u16).to_be_bytes());
            seg[18..20].copy_from_slice(&ip_id.wrapping_add((off / mss) as u16).to_be_bytes());
            seg[14 + ihl + 4..14 + ihl + 8].copy_from_slice(&seq.wrapping_add(off as u32).to_be_bytes());
            if off + piece < payload {
                // Only the final piece finishes the sender's message.
                seg[14 + ihl + 13] &= !0x09;
            }
            seg[24..26].fill(0);
            seg[14 + ihl + 16..14 + ihl + 18].fill(0);
            checksum_calculate(&mut seg, true, true);
            self.wire.push(seg);
            off += piece;
        }
    }

    /// Ring the transmit doorbell: send every descriptor from the chip's
    /// consumer up to the producer index the driver just wrote, then report
    /// the new consumer through the status block.
    ///
    /// The producer is a descriptor index the driver reduces modulo the ring
    /// length itself; the failure mode is an index past the ring, which would
    /// leave the walk chasing a consumer it never reaches. An out-of-range
    /// index is refused, and the walk is bounded by the ring length
    /// regardless. A frame whose descriptors are not all there yet, or whose
    /// memory cannot be read, is left to retry from its first descriptor.
    pub(super) fn tx_kick(&mut self, prod_raw: u32) {
        if let Some(cpu) = crate::cpu::smp::on_cpu_thread() {
            if let (Some(cycles), Some(mark)) = (self.cpu_cycles.get(cpu), self.tx_doorbell.get_mut(cpu)) {
                *mark = Some(cycles.get());
            }
        }
        let prod = (prod_raw & 0xffff) as u16;
        let ring_len = (self.sram_read(SRAM_SEND_RING_RCB + RCB_MAXLEN_FLAGS) >> 16) as u16;
        if ring_len == 0 || prod >= ring_len {
            self.refuse(Refused::ProducerOutOfRange(Mailbox::TxProducer));
            return;
        }
        let ring_base = self.compose_hostaddr(
            self.sram_read(SRAM_SEND_RING_RCB + RCB_HOSTADDR_HI),
            self.sram_read(SRAM_SEND_RING_RCB + RCB_HOSTADDR_LO),
        );
        let mut frame: Vec<u8> = Vec::new();
        let mut walked = 0usize;
        let mut completed = 0usize;
        let mut frame_flags = 0u16;
        let mut frame_start = self.tx_cons;
        let mut frame_mss = 0usize;
        let mut dropped = false;
        while self.tx_cons != prod && walked < usize::from(ring_len) {
            let Some(bd) = self.dma_read_image(ring_base + u64::from(self.tx_cons) * TXBD_SIZE, TXBD_SIZE as usize, false)
            else {
                self.refuse(Refused::DmaFault(DmaWhat::TxDescriptor));
                self.tx_rollback_to_frame(frame_start, completed);
                return;
            };
            let buf_addr = u64::from(be32(&bd, 0)) << 32 | u64::from(be32(&bd, 4));
            let len = usize::from(ld16(&bd, TXBD_LEN));
            let flags = ld16(&bd, TXBD_FLAGS);
            if !dropped && frame.len() + len <= MAX_FRAME {
                let Some(payload) = self.dma_read_image(buf_addr, len, true) else {
                    self.refuse(Refused::DmaFault(DmaWhat::TxPayload));
                    self.tx_rollback_to_frame(frame_start, completed);
                    return;
                };
                frame.extend_from_slice(&payload);
            } else if !dropped {
                // The frame overran the buffer. Drop the whole frame and stay
                // dropping across the remaining fragments so a later one is
                // not mistaken for a fresh packet; resume at the next END.
                self.refuse(Refused::TxFrameTooLarge);
                dropped = true;
            }
            frame_flags |= flags;
            if frame_mss == 0 {
                frame_mss = usize::from(ld16(&bd, TXBD_MSS));
            }
            if flags & TXBDFLAG_END != 0 {
                if !dropped && !frame.is_empty() {
                    let mut whole = std::mem::take(&mut frame);
                    if frame_mss == 0 {
                        self.tx_replace_source(&mut whole, frame_flags);
                    }
                    self.tx_segment(whole, frame_mss, frame_flags);
                }
                frame.clear();
                frame_flags = 0;
                frame_mss = 0;
                dropped = false;
            }
            self.tx_cons = (self.tx_cons + 1) % ring_len;
            walked += 1;
            if flags & TXBDFLAG_END != 0 {
                frame_start = self.tx_cons;
                completed = walked;
            }
        }
        if self.tx_cons != frame_start {
            self.tx_rollback_to_frame(frame_start, completed);
            return;
        }
        self.status_tx_cons = prod;
        self.tx_coalesce(completed);
    }

    // ---- receive mailboxes ----

    pub(super) fn rx_mailbox_written(&mut self, off: u32, val: u32) {
        let index = (val & 0xffff) as u16;
        match off {
            reg::MBX_RX_JUMBO_PROD_LO => {
                // Posting jumbo-ring buffers; guard the index like the others.
                if index >= RX_JUMBO_RING_LEN {
                    self.refuse(Refused::ProducerOutOfRange(Mailbox::RxJumboProducer));
                    return;
                }
                self.set_rf(off, val);
                self.rx_fetch();
            }
            reg::MBX_RX_STD_PROD_LO => {
                let std_len = self.rf(reg::RX_STD_RCB_MAXLEN_FLAGS) >> 16;
                if std_len == 0 || index >= RX_STD_RING_LEN {
                    self.refuse(Refused::ProducerOutOfRange(Mailbox::RxStandardProducer));
                    return;
                }
                self.set_rf(off, val);
                self.rx_fetch();
            }
            _ => {
                // The driver's return-ring acknowledgement, same range guard.
                let ret_len = (self.sram_read(SRAM_RX_RETURN_RCB + RCB_MAXLEN_FLAGS) >> 16) as u16;
                if ret_len == 0 || index >= ret_len {
                    self.refuse(Refused::ProducerOutOfRange(Mailbox::RxReturnConsumer));
                    return;
                }
                self.set_rf(off, val);
            }
        }
        // Room where there was none: take anything the peer had waiting.
        self.retry_rx_flush();
    }

    /// A producer ring's control block or replenish threshold changed, which
    /// can let a fetch start that the last producer write could not.
    pub(super) fn rx_ring_setup_written(&mut self) {
        self.rx_fetch();
        self.retry_rx_flush();
    }

    // ---- descriptor fetch ----

    /// Copy posted descriptors into the chip's cache, on both producer rings.
    ///
    /// The status block's producer-ring consumers report how far this has
    /// gone, which runs ahead of the descriptors frames have used. The chip
    /// fetches only while its cache has room, never past the producer, and
    /// starts a fetch only once the host has made at least the ring's
    /// replenish threshold of descriptors available [`57XX-PG105-R`, printed
    /// pages 118, 119 and 443]. A fetch fills the cache as far as both allow,
    /// at once: the guide gives neither the chip's burst size nor its fetch
    /// latency, so those are approximated. So is when the chip looks again:
    /// here, at every producer write and every time a frame frees a cache
    /// entry. Only the index is tracked; a frame reads its descriptor from
    /// host memory when it uses it, where the chip uses the copy it fetched,
    /// which differs only for a driver that rewrites a posted descriptor.
    pub(super) fn rx_fetch(&mut self) {
        if self.rf(reg::RX_STD_RCB_MAXLEN_FLAGS) >> 16 != 0 {
            let prod = (self.rf(reg::MBX_RX_STD_PROD_LO) & 0xffff) as u16;
            let thresh = self.rf(reg::RX_STD_REPLENISH_THRESH);
            self.rx_std_fetched =
                fetch_ahead(prod, self.rx_std_cons, self.rx_std_fetched, RX_STD_RING_LEN, RX_STD_BD_CACHE, thresh);
            // A fetch that gave the chip a buffer ends the shortage.
            if self.rx_std_fetched != self.rx_std_cons {
                self.rx_std_shortage_drops = 0;
            }
        }
        if self.jumbo_ring_enabled() {
            let prod = (self.rf(reg::MBX_RX_JUMBO_PROD_LO) & 0xffff) as u16;
            let thresh = self.rf(reg::RX_JUMBO_REPLENISH_THRESH);
            self.rx_jumbo_fetched = fetch_ahead(
                prod,
                self.rx_jumbo_cons,
                self.rx_jumbo_fetched,
                RX_JUMBO_RING_LEN,
                RX_JUMBO_BD_CACHE,
                thresh,
            );
            if self.rx_jumbo_fetched != self.rx_jumbo_cons {
                self.rx_shortage_drops = 0;
            }
        }
    }

    // ---- receive ----

    /// Whether the jumbo ring can take a frame at all, independent of whether
    /// one is posted: it needs a host base and must not be disabled. Extended
    /// RX Enable asks for the 64-byte descriptor format this model does not
    /// parse, so a ring asking for it is treated as disabled. Max_Len plays no
    /// part: hardware ignores it for the jumbo ring.
    fn jumbo_ring_enabled(&self) -> bool {
        let flags = self.rf(reg::RX_JUMBO_RCB_MAXLEN_FLAGS);
        let base = self.compose_hostaddr(self.rf(reg::RX_JUMBO_RCB_HADDR_HI), self.rf(reg::RX_JUMBO_RCB_HADDR_LO));
        base != 0 && flags & (RCB_FLAG_DISABLE_RING | RCB_FLAG_EXT_RX_ENABLE) == 0
    }

    /// Whether the engine can take a frame: the rings must be set up and host
    /// coalescing enabled, and the chip must hold a fetched buffer it has not
    /// yet used.
    pub(super) fn rx_ready(&self) -> bool {
        let std_len = self.rf(reg::RX_STD_RCB_MAXLEN_FLAGS) >> 16;
        let ret_len = (self.sram_read(SRAM_RX_RETURN_RCB + RCB_MAXLEN_FLAGS) >> 16) as u16;
        // A completion the driver has not read yet must not be overwritten;
        // when the next would land on the entry the driver has consumed up to,
        // the ring is full and the frame is refused.
        let ret_cons = (self.rf(reg::MBX_RX_CONS0_LO) & 0xffff) as u16;
        // Either ring holding a buffer is enough; which one a frame takes
        // depends on its size.
        let posted = (std_len != 0 && self.rx_std_fetched != self.rx_std_cons)
            || (self.jumbo_ring_enabled() && self.rx_jumbo_fetched != self.rx_jumbo_cons);
        if self.rx_latency.len() >= RX_LATENCY_DEPTH {
            return false;
        }
        // Against the staged producer, not the published one: a return
        // descriptor whose completion is still held has already been written.
        ret_len != 0
            && posted
            && self.rf(reg::HCC_MODE) & HCC_MODE_ENABLE != 0
            && (self.rx_prod_staged + 1) % ret_len != ret_cons
    }

    /// The address the part filters on is MAC address filter slot 0, which the
    /// driver programs, not the address the board gave the device: IRIX
    /// installs the system Ethernet address over whatever the card came up
    /// with.
    fn frame_accepted(&self, frame: &[u8]) -> bool {
        if frame.len() < 6 {
            return false;
        }
        if frame[0] & 1 != 0 {
            return true;
        }
        let hi = self.rf(reg::MAC_ADDR_0_HIGH);
        let lo = self.rf(reg::MAC_ADDR_0_LOW);
        frame[..6] == [(hi >> 8) as u8, hi as u8, (lo >> 24) as u8, (lo >> 16) as u8, (lo >> 8) as u8, lo as u8]
    }

    fn count_no_more_rxbds(&mut self) {
        let n = self.sram_read(SRAM_STATS_NO_MORE_RXBDS).wrapping_add(1);
        self.sram_write(SRAM_STATS_NO_MORE_RXBDS, n);
    }

    /// A frame from the backend. Not ready, or no buffer fetched, defers it:
    /// the only answer that leaves the frame at the peer's queue head. A frame
    /// the address filter drops is consumed, never deferred, so the peer is
    /// not wedged waiting on a frame that will never be taken. Otherwise the
    /// frame is DMA'd into the next posted buffer, a completed descriptor is
    /// written into the return ring, and the producer the status block reports
    /// advances once the frame's arrival deadline passes.
    pub(super) fn receive(&mut self, frame: &[u8]) -> RxOutcome {
        // A halted receive MAC is not listening, so the frame is gone rather
        // than held: a backend that queued it would hand the driver traffic
        // from before the interface existed.
        if self.rf(reg::RX_MODE) & 2 == 0 {
            return RxOutcome::Consumed;
        }
        if !self.rx_ready() {
            return RxOutcome::Deferred;
        }
        if !self.frame_accepted(frame) {
            return RxOutcome::Consumed;
        }
        if frame.len() > MAX_FRAME {
            // Drop it rather than deliver part of it: a frame cut short looks
            // whole, and the failure surfaces wherever the truncated payload
            // is finally parsed.
            self.refuse(Refused::RxFrameTooLarge);
            return RxOutcome::Consumed;
        }
        // Pad a short frame to the 60-byte Ethernet minimum: a backend can
        // hand over a runt, and the driver rejects a descriptor below it.
        let mut padded;
        let buf: &[u8] = if frame.len() < MIN_FRAME {
            padded = frame.to_vec();
            padded.resize(MIN_FRAME, 0);
            &padded
        } else {
            frame
        };
        let framelen = buf.len() as u32;

        // Pick the ring by what the frame fits in. The standard ring's Max_Len
        // is the driver's statement of how much room its buffers hold; a
        // larger frame goes to the jumbo ring when it is enabled, and is
        // dropped rather than written past the end of someone's buffer when
        // it is not.
        let std_size = u32::from((self.rf(reg::RX_STD_RCB_MAXLEN_FLAGS) >> 16) as u16);
        // A frame takes a buffer the chip has fetched, never one only posted.
        let std_empty = self.rx_std_fetched == self.rx_std_cons;
        let jumbo_empty = self.rx_jumbo_fetched == self.rx_jumbo_cons;
        let jumbo_enabled = self.jumbo_ring_enabled();
        let (ring, ring_base, idx) = if framelen <= std_size {
            if std_empty {
                // The frame fits a standard buffer and the chip holds none; it
                // cannot take a jumbo one, because a frame's ring is chosen by
                // its length against the standard ring's Max_Len. Discard
                // rather than refuse: refusing stops the backend reading and
                // strands the frames behind it that the jumbo ring can carry.
                // Only while the chip holds a jumbo buffer; without one the caller
                // is owed the refusal.
                if !jumbo_enabled || jumbo_empty {
                    return RxOutcome::Deferred;
                }
                if self.rx_std_shortage_drops == 0 {
                    self.refuse(Refused::RxShortage(Ring::Standard));
                }
                self.rx_std_shortage_drops += 1;
                self.count_no_more_rxbds();
                return RxOutcome::Consumed;
            }
            let base = self
                .compose_hostaddr(self.rf(reg::RX_STD_RCB_HADDR_HI), self.rf(reg::RX_STD_RCB_HADDR_LO));
            (Ring::Standard, base, self.rx_std_cons)
        } else if jumbo_enabled {
            if jumbo_empty {
                // A frame longer than the standard Max_Len consumes a
                // descriptor from the jumbo ring, and the chip holds none.
                // Discard rather than refuse, for the reason above, and only
                // while it holds a standard buffer.
                if std_size == 0 || std_empty {
                    return RxOutcome::Deferred;
                }
                if self.rx_shortage_drops == 0 {
                    self.refuse(Refused::RxShortage(Ring::Jumbo));
                }
                self.rx_shortage_drops += 1;
                self.count_no_more_rxbds();
                return RxOutcome::Consumed;
            }
            let base = self
                .compose_hostaddr(self.rf(reg::RX_JUMBO_RCB_HADDR_HI), self.rf(reg::RX_JUMBO_RCB_HADDR_LO));
            (Ring::Jumbo, base, self.rx_jumbo_cons)
        } else {
            self.refuse(Refused::RxNoRoom(Ring::Standard));
            return RxOutcome::Consumed;
        };

        let ret_len = (self.sram_read(SRAM_RX_RETURN_RCB + RCB_MAXLEN_FLAGS) >> 16) as u16;
        let ret_base = self.compose_hostaddr(
            self.sram_read(SRAM_RX_RETURN_RCB + RCB_HOSTADDR_HI),
            self.sram_read(SRAM_RX_RETURN_RCB + RCB_HOSTADDR_LO),
        );

        // The posted buffer's host address, from the chosen ring's descriptor.
        let Some(std_bd) = self.dma_read_image(ring_base + u64::from(idx) * RXBD_SIZE, RXBD_SIZE as usize, false)
        else {
            self.refuse(Refused::DmaFault(DmaWhat::RxDescriptor));
            return RxOutcome::Deferred;
        };
        let buf_addr = u64::from(be32(&std_bd, 0)) << 32 | u64::from(be32(&std_bd, 4));
        let mut rx_flags = RXBDFLAG_END;
        if ring == Ring::Jumbo {
            // The jumbo ring carries no ring-wide buffer size; the driver
            // states each buffer's size in the descriptor it posts.
            let buf_len = be32(&std_bd, RXBD_IDX_LEN) & 0xffff;
            if framelen > buf_len {
                self.refuse(Refused::RxNoRoom(Ring::Jumbo));
                return RxOutcome::Consumed;
            }
            rx_flags |= RXBDFLAG_JUMBO_RING;
        }
        if !self.dma_write_image(buf_addr, buf, true) {
            self.refuse(Refused::DmaFault(DmaWhat::RxPayload));
            return RxOutcome::Deferred;
        }
        let csum = rx_offload_checksum(buf, self.rf(reg::MODE_CTL), &mut rx_flags);

        let mut ret_bd = [0u8; RXBD_SIZE as usize];
        st32(&mut ret_bd, 0, (buf_addr >> 32) as u32);
        st32(&mut ret_bd, 4, buf_addr as u32);
        // The length includes the four-byte FCS the driver strips.
        st32(&mut ret_bd, RXBD_IDX_LEN, u32::from(idx) << 16 | ((framelen + FCS_LEN) & 0xffff));
        st32(&mut ret_bd, RXBD_TYPE_FLAGS, rx_flags);
        st32(&mut ret_bd, RXBD_CSUM, csum);
        st32(&mut ret_bd, RXBD_OPAQUE, be32(&std_bd, RXBD_OPAQUE));
        if !self.dma_write_image(ret_base + u64::from(self.rx_prod_staged) * RXBD_SIZE, &ret_bd, false) {
            self.refuse(Refused::DmaFault(DmaWhat::RxCompletion));
            return RxOutcome::Deferred;
        }
        match ring {
            Ring::Jumbo => self.rx_jumbo_cons = (self.rx_jumbo_cons + 1) % RX_JUMBO_RING_LEN,
            Ring::Standard => self.rx_std_cons = (self.rx_std_cons + 1) % RX_STD_RING_LEN,
        }
        // The cache entry the frame used is free for the next fetch.
        self.rx_fetch();
        self.rx_prod_staged = (self.rx_prod_staged + 1) % ret_len;
        // The frame and its return descriptor are in guest memory; the
        // producer the status block reports, and the interrupt after it, wait
        // for the frame's arrival deadline.
        self.rx_hold(self.rx_prod_staged);
        RxOutcome::Consumed
    }

    // ---- the arrival line ----

    fn rx_publish(&mut self, status_rx_prod: u16) {
        self.status_rx_prod = status_rx_prod;
        self.rx_coalesce();
    }

    /// Hold a completion until the frame's minimum arrival latency has passed.
    fn rx_hold(&mut self, status_rx_prod: u16) {
        let held = HeldCompletion { due: self.now + self.rx_floor, prod: status_rx_prod, after: self.tx_doorbell };
        if self.rx_floor.is_zero() && self.rx_latency.is_empty() && self.senders_ran_past(&held.after) {
            self.rx_publish(status_rx_prod);
        } else {
            self.rx_latency.push_back(held);
        }
    }

    /// Whether every processor that rang the send doorbell has run the
    /// cycle floor past its ring. A counter that went backwards was reset
    /// with its processor, which has left the send path too.
    fn senders_ran_past(&self, after: &Doorbells) -> bool {
        after.iter().zip(&self.cpu_cycles).all(|(mark, cycles)| {
            let Some(at) = *mark else { return true };
            let now = cycles.get();
            now < at || now - at >= self.rx_floor_cycles
        })
    }

    /// The head completion's deadline passed. One whose sender has not yet
    /// run far enough looks again shortly after `entered`, the time the
    /// model was entered at.
    pub(super) fn rx_arrival_due(&mut self, entered: Instant) {
        let mut published = false;
        while let Some(held) = self.rx_latency.front().copied() {
            if held.due > self.now {
                break;
            }
            if !self.senders_ran_past(&held.after) {
                self.rx_latency[0].due = entered.max(self.now) + RX_CYCLES_RECHECK;
                break;
            }
            self.rx_latency.pop_front();
            self.rx_publish(held.prod);
            published = true;
        }
        if published {
            self.retry_rx_flush();
        }
    }
}
