//! The Ethernet MAC inside IOC3, with the Octane ICS1890 or Fuel DP83843
//! transceiver on its management bus.
//!
//! IOC3 forwards the register offsets 0xf0..=0x150 of its register BAR here
//! through [`Ioc3EthPort`]. The MAC is SGI's own design with no vendor
//! document, so what it does is the contract the public drivers (Linux and
//! OpenBSD) depend on:
//!
//! - receive: a ring of 512 buffer addresses; the engine writes a two-word
//!   header and the frame into the buffer and advances the consume index;
//! - transmit: a ring of 128-byte descriptors (128 or 512 of them) with an
//!   inline data area and two buffer pointers;
//! - one interrupt status/enable pair, receive sources low, transmit high;
//! - the management interface to the PHY.
//!
//! Memory is reached through the [`PciDma`] IOC3 hands over in `connect`.
//! Ring entries and descriptor words are big-endian bytes at bus addresses,
//! frame bytes are in wire order; the Bridge applies any programmed swap.
//!
//! Frames use iris's shared network link. The machine attaches it through
//! [`Ioc3Eth::attach`].
//!
//! Approximations (the QEMU model this was ported from has the same ones,
//! plus one it dropped):
//! - Receive completions are published as soon as the frame is in memory.
//!   The QEMU model holds each one back for a modelled minimum arrival
//!   latency; this port has no such delay.
//! - A non-zero receive timer reload is stored but not timed; only the
//!   "interrupt immediately" setting every public driver uses is modelled.
//! - The PHY's 100 ms negotiation delay is wall-clock time (the QEMU model
//!   used virtual time) and the synthetic peer never faults.
//! - Quick Poll status (PHY register 0x11) reports live state without the
//!   per-bit latches the datasheet describes.
//! - A PHY register the model does not implement reads 0. The QEMU model
//!   answers 0xffff, and the ICS1890 datasheet has an unanswered management
//!   read float to all ones.
//! - Frames go to the wire only while the PHY reports link. The QEMU model
//!   sends whenever the PHY's data path is open (not powered down or
//!   isolated), link or not.
//!
//! Ported from hw/net/sgi-ioc3-eth.c in jamesbraid/qemu-sgi-origami. The
//! Octane PHY model lives in `ics1890.rs`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use parking_lot::Mutex;
use std::time::{Duration, Instant};

use crate::dev::ioc3_port::Ioc3EthPort;
use crate::dev::ics1890::Ics1890;
use crate::pci::{PciDma, PciDmaStatus};
use crate::traits::{Device, Resettable};

#[cfg(test)]
mod tests;

/// Register offsets within IOC3's register BAR.
pub mod reg {
    pub const MAC_CONTROL: u32 = 0x0f0;
    pub const INT_STATUS: u32 = 0x0f4;
    pub const INT_ENABLE: u32 = 0x0f8;
    pub const RX_CONTROL: u32 = 0x0fc;
    pub const RX_RING_HI: u32 = 0x100;
    pub const RX_RING_LO: u32 = 0x104;
    pub const RX_BARRIER: u32 = 0x108;
    pub const RX_CONSUME: u32 = 0x10c;
    pub const RX_PRODUCE: u32 = 0x110;
    pub const RX_TIMER: u32 = 0x114;
    pub const TX_CONTROL: u32 = 0x118;
    pub const RANDOM_SEED: u32 = 0x11c;
    pub const COUNTERS: u32 = 0x120;
    pub const BUFFER_INDEX: u32 = 0x124;
    pub const TX_RING_HI: u32 = 0x128;
    pub const TX_RING_LO: u32 = 0x12c;
    pub const TX_CONSUME: u32 = 0x130;
    pub const TX_PRODUCE: u32 = 0x134;
    pub const MAC_ADDR_HI: u32 = 0x138;
    pub const MAC_ADDR_LO: u32 = 0x13c;
    pub const HASH_HI: u32 = 0x140;
    pub const HASH_LO: u32 = 0x144;
    pub const MII_CONTROL: u32 = 0x148;
    pub const MII_DATA_R: u32 = 0x14c;
    pub const MII_DATA_W: u32 = 0x150;
}

/// Register bit assignments.
pub mod bits {
    pub const MCR_PROMISC: u32 = 1 << 1;
    pub const MCR_PAD_EN: u32 = 1 << 2;
    pub const MCR_RXOFF_MASK: u32 = 0x1f8;
    pub const MCR_RXOFF_SHIFT: u32 = 3;
    pub const MCR_RAM_PARITY: u32 = 1 << 9;
    pub const MCR_TX_DMA_EN: u32 = 1 << 13;
    pub const MCR_TX_EN: u32 = 1 << 14;
    pub const MCR_RX_DMA_EN: u32 = 1 << 15;
    pub const MCR_RX_EN: u32 = 1 << 16;
    pub const MCR_LOOPBACK: u32 = 1 << 17;
    pub const MCR_ARB_DIAG_IDLE: u32 = 1 << 21;
    pub const MCR_RESET: u32 = 1 << 31;

    pub const INT_RX_TIMER: u32 = 1 << 0;
    pub const INT_RX_THRESHOLD: u32 = 1 << 1;
    pub const INT_TX_EMPTY: u32 = 1 << 16;
    pub const INT_TX_COMPLETE: u32 = 1 << 22;

    pub const RX_RING_ENTRIES: u32 = 512;
    pub const RX_RING_SHIFT: u32 = 3;
    pub const RX_BASE_MASK: u32 = 0xffff_f000;
    pub const RX_INDEX_MASK: u32 = (RX_RING_ENTRIES << RX_RING_SHIFT) - 1;
    pub const RCSR_THRESH_MASK: u32 = 0x1ff;
    pub const RTR_RELOAD_MASK: u32 = 0x7ff;
    pub const RPIR_ARM: u32 = 1 << 31;

    pub const RXH_VALID: u32 = 1 << 31;
    pub const RXH_BYTECNT_MASK: u32 = 0x07ff_0000;
    pub const RXH_BYTECNT_SHIFT: u32 = 16;
    pub const RXE_LEN_LO_SHIFT: u32 = 12;
    pub const RXE_LEN_HI_SHIFT: u32 = 16;
    pub const RXE_MULTICAST: u32 = 1 << 26;
    pub const RXE_BROADCAST: u32 = 1 << 27;
    pub const RXE_LONGEVENT: u32 = 1 << 28;
    pub const RXE_BADPKT: u32 = 1 << 29;
    pub const RXE_GOODPKT: u32 = 1 << 30;

    pub const TX_DESC_SIZE: u32 = 128;
    pub const TX_RING_SMALL: u32 = 128;
    pub const TX_RING_LARGE: u32 = 512;
    pub const TX_RING_SEL: u32 = 1;
    pub const TX_BASE_MASK: u32 = 0xffff_c000;
    pub const TX_CONSUME_IDLE: u32 = 1 << 31;
    pub const TX_CONSUME_MASK: u32 = 0xffff;

    pub const TXD_BYTECNT_MASK: u32 = 0x7ff;
    pub const TXD_INT_DONE: u32 = 1 << 12;
    pub const TXD_DATA_VALID: u32 = 1 << 16;
    pub const TXD_BUF1_VALID: u32 = 1 << 17;
    pub const TXD_BUF2_VALID: u32 = 1 << 18;
    pub const TXD_DOCHECKSUM: u32 = 1 << 19;
    pub const TXD_DATA_LEN_MASK: u32 = 0x7f;
    pub const TXD_BUF1_LEN_MASK: u32 = 0x0007_ff00;
    pub const TXD_BUF1_LEN_SHIFT: u32 = 8;
    pub const TXD_BUF2_LEN_MASK: u32 = 0x7ff0_0000;
    pub const TXD_BUF2_LEN_SHIFT: u32 = 20;
    /// The inline area holds 104 bytes; its length field can name 127.
    pub const TXD_DATA_OFFSET: usize = 24;
    pub const TXD_DATA_SIZE: usize = 128 - TXD_DATA_OFFSET;

    pub const MII_READ: u32 = 1 << 10;
    pub const MII_BUSY: u32 = 1 << 11;
    /// All-ones, or a driver's scan stops at the wrong address.
    pub const MII_ABSENT: u16 = 0xffff;
}

use bits::*;
use reg::*;

/// The eleven-bit byte-count field's limit.
const FRAME_MAX: usize = 2048;
const NREGS: usize = ((MII_DATA_W + 4 - MAC_CONTROL) / 4) as usize;
/// The PHY's management address on the Octane's IOC3.
pub const PHY_ADDR: u8 = 1;
/// How long the synthetic link partner takes to complete negotiation.
pub const NEGOTIATION_DELAY: Duration = Duration::from_millis(100);

/// The management and packet-path behavior that varies between IOC3 PHYs.
pub(crate) trait MiiPhy: Send {
    fn address(&self) -> u8;
    fn deadline(&self) -> Option<Instant>;
    fn wire_enabled(&self) -> bool;
    fn in_loopback(&self) -> bool;
    fn sync(&mut self, now: Instant);
    fn read(&mut self, register: u8, now: Instant) -> u16;
    fn write(&mut self, register: u8, value: u16, now: Instant);
    fn set_backend_link(&mut self, up: bool, now: Instant);
    fn power_on(&mut self);
}

fn ridx(off: u32) -> usize {
    ((off - MAC_CONTROL) / 4) as usize
}

// ---------------------------------------------------------------------------
// ICS1890 transceiver
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Checksums and the multicast hash
// ---------------------------------------------------------------------------

fn csum_add(data: &[u8]) -> u32 {
    let mut sum = 0u32;
    for (i, b) in data.iter().enumerate() {
        sum += u32::from(*b) << if i & 1 == 0 { 8 } else { 0 };
    }
    sum
}

fn csum_fold(mut sum: u32) -> u16 {
    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    sum as u16
}

/// Fill in the TCP or UDP checksum of an IPv4 frame; anything else is left
/// alone. The IP header checksum is the driver's own.
fn tcp_udp_checksum(frame: &mut [u8]) {
    if frame.len() < 14 + 20 || frame[12] != 0x08 || frame[13] != 0x00 {
        return;
    }
    let ip = &frame[14..];
    if ip[0] >> 4 != 4 {
        return;
    }
    let hlen = usize::from(ip[0] & 0xf) * 4;
    let ip_len = usize::from(u16::from_be_bytes([ip[2], ip[3]]));
    let frag = u16::from_be_bytes([ip[6], ip[7]]) & 0x3fff;
    if hlen < 20 || ip_len < hlen || 14 + ip_len > frame.len() || frag != 0 {
        return;
    }
    let proto = ip[9];
    let plen = ip_len - hlen;
    let csum_at = match proto {
        6 if plen >= 20 => 16,
        17 if plen >= 8 => 6,
        _ => return,
    };
    let start = 14 + hlen;
    frame[start + csum_at] = 0;
    frame[start + csum_at + 1] = 0;
    let mut sum = csum_add(&frame[26..34]);
    sum += u32::from(proto) + plen as u32;
    sum += csum_add(&frame[start..start + plen]);
    let c = !csum_fold(sum);
    frame[start + csum_at..start + csum_at + 2].copy_from_slice(&c.to_be_bytes());
}

/// Linux's `ioc3_hash()`: the low six bits of the little-endian CRC, themselves
/// bit-reversed.
fn mcast_hash(addr: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for b in &addr[..6] {
        crc ^= u32::from(*b);
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { 0xedb8_8320 } else { 0 };
        }
    }
    let low = crc & 0x3f;
    (0..6).fold(0, |acc, bit| (acc << 1) | ((low >> bit) & 1))
}

// ---------------------------------------------------------------------------
// The MAC
// ---------------------------------------------------------------------------

struct NullDma;
impl PciDma for NullDma {
    fn dma_read(&self, _addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        buf.fill(0);
        PciDmaStatus::MasterAbort
    }
    fn dma_write(&self, _addr: u64, _data: &[u8]) -> PciDmaStatus {
        PciDmaStatus::MasterAbort
    }
}

fn dma_read(dma: &dyn PciDma, addr: u64, buf: &mut [u8]) -> bool {
    if addr.checked_add(buf.len() as u64).is_none() {
        buf.fill(0);
        return false;
    }
    dma.dma_read(addr, buf).is_ok()
}

fn dma_write(dma: &dyn PciDma, addr: u64, data: &[u8]) -> bool {
    if addr.checked_add(data.len() as u64).is_none() {
        return false;
    }
    dma.dma_write(addr, data).is_ok()
}

struct State {
    regs: [u32; NREGS],
    /// Sources raised while the same status bit was already set and
    /// unacknowledged: the guest's next write-one-to-clear of that bit must
    /// turn into a real edge for this one too rather than clearing both.
    int_deferred: u32,
    mii_data: u16,
    mii_data_w: u16,
    /// Apart from its index: an index-only write must not disarm.
    rx_timer_armed: bool,
    phy: Box<dyn MiiPhy>,
    irq_asserted: bool,
    /// Level changes not yet delivered to the callback, in order.
    irq_events: Vec<bool>,
    frame: Vec<u8>,
    tx_frames: u64,
    rx_frames: u64,
}

impl State {
    fn new(phy: Box<dyn MiiPhy>) -> Self {
        Self {
            regs: [0; NREGS],
            int_deferred: 0,
            mii_data: MII_ABSENT,
            mii_data_w: 0,
            rx_timer_armed: false,
            phy,
            irq_asserted: false,
            irq_events: Vec::new(),
            frame: vec![0; FRAME_MAX],
            tx_frames: 0,
            rx_frames: 0,
        }
    }

    fn power_on(&mut self) {
        self.phy.power_on();
        self.regs = [0; NREGS];
        self.int_deferred = 0;
        self.mii_data = MII_ABSENT;
        self.mii_data_w = 0;
        self.rx_timer_armed = false;
        self.irq_asserted = false;
        self.irq_events.clear();
        self.frame.fill(0);
        self.tx_frames = 0;
        self.rx_frames = 0;
    }

    fn get(&self, off: u32) -> u32 {
        self.regs[ridx(off)]
    }
    fn set(&mut self, off: u32, v: u32) {
        self.regs[ridx(off)] = v;
    }

    fn update_irq(&mut self) {
        let level = self.get(INT_STATUS) & self.get(INT_ENABLE) != 0;
        if level != self.irq_asserted {
            self.irq_asserted = level;
            self.irq_events.push(level);
        }
    }

    fn raise_int(&mut self, sources: u32) {
        let status = self.get(INT_STATUS);
        self.int_deferred |= status & sources;
        self.set(INT_STATUS, status | sources);
        self.update_irq();
    }

    /// The engine reset keeps the ring bases: the power-on diagnostic
    /// programs them once and never rewrites them across its reset.
    fn reset_registers(&mut self, keep_rings: bool) {
        let keep = [self.get(RX_RING_HI), self.get(RX_RING_LO), self.get(TX_RING_HI), self.get(TX_RING_LO)];
        self.int_deferred = 0;
        self.regs = [0; NREGS];
        if keep_rings {
            self.set(RX_RING_HI, keep[0]);
            self.set(RX_RING_LO, keep[1]);
            self.set(TX_RING_HI, keep[2]);
            self.set(TX_RING_LO, keep[3]);
        }
        self.mii_data = MII_ABSENT;
        self.mii_data_w = 0;
        self.rx_timer_armed = false;
        self.update_irq();
    }

    fn mac_loopback(&self) -> bool {
        self.get(MAC_CONTROL) & MCR_LOOPBACK != 0
    }

    fn looping_back(&self) -> bool {
        self.mac_loopback() || self.phy.in_loopback()
    }

    fn tx_enabled(&self) -> bool {
        let need = MCR_TX_DMA_EN | MCR_TX_EN;
        self.get(MAC_CONTROL) & need == need
    }

    /// Whether a wire frame can be taken now.
    fn rx_ready(&self) -> bool {
        let need = MCR_RX_DMA_EN | MCR_RX_EN;
        if self.get(MAC_CONTROL) & need != need {
            return false;
        }
        if !self.phy.wire_enabled() {
            return false;
        }
        // In loopback, wire traffic would shift a diagnostic's comparisons.
        if self.looping_back() {
            return false;
        }
        self.get(RX_PRODUCE) & RX_INDEX_MASK != self.get(RX_CONSUME) & RX_INDEX_MASK
    }

    /// Filter against the guest-programmed station address.
    fn rx_filter(&self, buf: &[u8]) -> bool {
        if self.get(MAC_CONTROL) & MCR_PROMISC != 0 {
            return true;
        }
        if buf.len() < 6 {
            return false;
        }
        let hi = self.get(MAC_ADDR_HI);
        let lo = self.get(MAC_ADDR_LO);
        let station = [lo as u8, (lo >> 8) as u8, (lo >> 16) as u8, (lo >> 24) as u8, hi as u8, (hi >> 8) as u8];
        if buf[..6] == station || buf[..6] == [0xff; 6] {
            return true;
        }
        if buf[0] & 1 == 0 {
            return false;
        }
        let index = mcast_hash(buf);
        let word = if index < 32 { self.get(HASH_LO) } else { self.get(HASH_HI) };
        (word >> (index & 31)) & 1 != 0
    }

    fn rx_sources(&self, consume: u32) -> u32 {
        let reload = self.get(RX_TIMER) & RTR_RELOAD_MASK;
        let threshold = self.get(RX_CONTROL) & RCSR_THRESH_MASK;
        let produce = self.get(RX_PRODUCE) & RX_INDEX_MASK;
        let available = (produce.wrapping_sub(consume) & RX_INDEX_MASK) >> RX_RING_SHIFT;
        let mut sources = 0;
        // A non-zero reload is not modelled; only an immediate timeout is.
        if self.rx_timer_armed && reload == 0 {
            sources |= INT_RX_TIMER;
        }
        // At-or-below: strictly-below is one frame late for OpenBSD.
        if available <= threshold {
            sources |= INT_RX_THRESHOLD;
        }
        sources
    }

    /// Put one frame into the next receive buffer. `short_circuit` marks a
    /// MAC-loopback frame, turned round ahead of the FCS append and status
    /// vector; PHY loopback goes the whole way and gets both. A frame whose
    /// ring entry or buffer cannot be reached is dropped.
    fn deliver(&mut self, dma: &dyn PciDma, buf: &[u8], short_circuit: bool) {
        let mut size = buf.len();
        let mut truncated = false;
        if size > FRAME_MAX - 8 {
            size = FRAME_MAX - 8;
            truncated = true;
        }
        let buf = &buf[..size];
        let consume = self.get(RX_CONSUME) & RX_INDEX_MASK;
        let ring_base = (u64::from(self.get(RX_RING_HI)) << 32) | u64::from(self.get(RX_RING_LO) & RX_BASE_MASK);

        // A refused entry fetch leaves no destination; drop the frame.
        let mut entry = [0u8; 8];
        if !ring_base.checked_add(u64::from(consume)).is_some_and(|a| dma_read(dma, a, &mut entry)) {
            return;
        }
        let buf_addr = u64::from_be_bytes(entry);

        let mcr = self.get(MAC_CONTROL);
        let offset = u64::from((mcr & MCR_RXOFF_MASK) >> MCR_RXOFF_SHIFT) * 2;

        // Guests read the wire length FCS included; the backend delivers no
        // FCS, so four phantom bytes are added or minimum frames read short.
        let wire_len = size as u32 + if short_circuit { 0 } else { 4 };
        let mut word0 = RXH_VALID | ((wire_len << RXH_BYTECNT_SHIFT) & RXH_BYTECNT_MASK);
        // The checksum-offload partial must be exact: Linux finishes the
        // transport checksum from it and trusts the fold.
        if !short_circuit {
            word0 |= u32::from(csum_fold(csum_add(buf)));
        }
        let lenlo = wire_len & 7;
        let lenhi = (wire_len >> 3) & 0x3ff;
        let mut word1 = RXE_GOODPKT | (lenlo << RXE_LEN_LO_SHIFT) | (lenhi << RXE_LEN_HI_SHIFT);
        if truncated {
            word1 |= RXE_LONGEVENT | RXE_BADPKT;
        }
        if size >= 6 && buf[..6] == [0xff; 6] {
            word1 |= RXE_BROADCAST;
        } else if size >= 1 && buf[0] & 1 != 0 {
            word1 |= RXE_MULTICAST;
        }

        // The buffer address is the guest's: one whose parts would run past
        // the top of the bus is a refused transfer, like any other.
        let at = |off: u64| buf_addr.checked_add(off);
        let write = |off: Option<u64>, data: &[u8]| off.is_some_and(|a| dma_write(dma, a, data));
        let mut ok = write(at(4), &(if short_circuit { 0 } else { word1 }).to_be_bytes());
        ok &= write(at(offset), buf);
        if !short_circuit {
            // The phantom FCS bytes, written as the zero they were summed as.
            ok &= write(at(offset).and_then(|a| a.checked_add(size as u64)), &[0; 4]);
        }
        // An unreachable buffer is not a delivered frame; the index stays.
        // The valid bit hands the buffer over, so it goes last.
        if !ok || !dma_write(dma, buf_addr, &word0.to_be_bytes()) {
            return;
        }
        let consume = (consume + (1 << RX_RING_SHIFT)) & RX_INDEX_MASK;
        self.set(RX_CONSUME, consume);
        self.rx_frames += 1;
        let sources = self.rx_sources(consume);
        self.raise_int(sources);
    }

    /// Assemble a frame from a descriptor into `self.frame`. None if a buffer
    /// could not be read.
    fn tx_gather(&mut self, dma: &dyn PciDma, cmd: u32, bufcnt: u32, buf1: u64, buf2: u64, inline: &[u8]) -> Option<usize> {
        let mut len = 0usize;
        let mut ok = true;
        if cmd & TXD_DATA_VALID != 0 {
            let n = ((bufcnt & TXD_DATA_LEN_MASK) as usize).min(TXD_DATA_SIZE).min(FRAME_MAX - len);
            self.frame[len..len + n].copy_from_slice(&inline[..n]);
            len += n;
        }
        for (valid, mask, shift, addr) in [
            (TXD_BUF1_VALID, TXD_BUF1_LEN_MASK, TXD_BUF1_LEN_SHIFT, buf1),
            (TXD_BUF2_VALID, TXD_BUF2_LEN_MASK, TXD_BUF2_LEN_SHIFT, buf2),
        ] {
            if cmd & valid != 0 {
                let n = (((bufcnt & mask) >> shift) as usize).min(FRAME_MAX - len);
                ok &= dma_read(dma, addr, &mut self.frame[len..len + n]);
                len += n;
            }
        }
        ok.then_some(len)
    }

    /// Walk the transmit ring up to `produce`. Frames for the wire go on
    /// `wire`; loopback frames are delivered to the receive side directly.
    fn do_tx(&mut self, dma: &dyn PciDma, produce: u32, wire: &mut Vec<Vec<u8>>) {
        let tx_lo = self.get(TX_RING_LO);
        let entries = if tx_lo & TX_RING_SEL != 0 { TX_RING_LARGE } else { TX_RING_SMALL };
        let ring_bytes = entries * TX_DESC_SIZE;
        let tx_base = (u64::from(self.get(TX_RING_HI)) << 32) | u64::from(tx_lo & TX_BASE_MASK);
        let loopback = self.looping_back();
        let mut raised = 0;
        let mut completed = false;

        // The walk stops on equality: an index it can never land on would be
        // an emulator hang from one register write, so reduce and align both.
        let align = |v: u32| (v % ring_bytes) & !(TX_DESC_SIZE - 1);
        let produce = align(produce);
        let mut consume = align(self.get(TX_CONSUME) & TX_CONSUME_MASK);

        while consume != produce {
            // The whole descriptor in one transfer, header and inline area.
            let mut d = [0u8; TX_DESC_SIZE as usize];
            // A refused fetch stops the walk with the index on the entry; a
            // descriptor address past the top of the bus is one.
            if !tx_base.checked_add(u64::from(consume)).is_some_and(|desc| dma_read(dma, desc, &mut d)) {
                break;
            }
            let be32 = |o: usize| u32::from_be_bytes(d[o..o + 4].try_into().unwrap());
            let be64 = |o: usize| u64::from_be_bytes(d[o..o + 8].try_into().unwrap());
            let (cmd, bufcnt, buf1, buf2) = (be32(0), be32(4), be64(8), be64(16));
            let Some(mut len) = self.tx_gather(dma, cmd, bufcnt, buf1, buf2, &d[TXD_DATA_OFFSET..]) else {
                break;
            };
            len = len.min((cmd & TXD_BYTECNT_MASK) as usize);

            // Transport sum only; the IP header checksum is the driver's own.
            if cmd & TXD_DOCHECKSUM != 0 {
                tcp_udp_checksum(&mut self.frame[..len]);
            }
            // Pad to the 802.3 minimum: the guest relies on the engine for it.
            if self.get(MAC_CONTROL) & MCR_PAD_EN != 0 && len < 60 {
                self.frame[len..60].fill(0);
                len = 60;
            }

            if len > 0 {
                let frame = self.frame[..len].to_vec();
                if loopback {
                    let short = self.mac_loopback();
                    self.deliver(dma, &frame, short);
                } else if self.phy.wire_enabled() {
                    wire.push(frame);
                    self.tx_frames += 1;
                }
            }
            if cmd & TXD_INT_DONE != 0 {
                raised |= INT_TX_COMPLETE;
            }
            consume = (consume + TX_DESC_SIZE) % ring_bytes;
            completed = true;
        }
        if completed {
            raised |= INT_TX_EMPTY;
        }
        // Idle is derived from the index, not from the loop returning.
        self.set(TX_CONSUME, consume | if consume == produce { TX_CONSUME_IDLE } else { 0 });
        self.raise_int(raised);
    }

    fn read(&mut self, off: u32, now: Instant) -> u32 {
        self.phy.sync(now);
        match off {
            // Reading the counters clears them.
            COUNTERS => std::mem::take(&mut self.regs[ridx(COUNTERS)]),
            MII_CONTROL => self.get(MII_CONTROL) & !MII_BUSY,
            MII_DATA_R => u32::from(self.mii_data),
            _ => self.get(off),
        }
    }

    fn write(&mut self, off: u32, val: u32, now: Instant, dma: &dyn PciDma, wire: &mut Vec<Vec<u8>>, wake_rx: &mut bool) {
        self.phy.sync(now);
        match off {
            INT_STATUS => {
                let deferred = self.int_deferred & val;
                let i = ridx(INT_STATUS);
                self.regs[i] &= !val;
                self.int_deferred &= !val;
                self.update_irq();
                if deferred != 0 {
                    // A completion landed on one of these bits after the
                    // driver's read but before this acknowledge: restoring the
                    // deferred ones now raises the line again as a real edge,
                    // rather than the single write silently absorbing both.
                    // All three public drivers write back what they read.
                    self.regs[i] |= deferred;
                    self.update_irq();
                }
            }
            INT_ENABLE => {
                // Enabling a set source must raise the line; OpenBSD relies on it.
                self.set(off, val);
                self.update_irq();
            }
            MII_CONTROL => {
                let phy = ((val >> 5) & 0x1f) as u8;
                let r = (val & 0x1f) as u8;
                let responds = phy == self.phy.address();
                // Unmatched accesses must not run PHY register side effects.
                if val & MII_READ != 0 {
                    self.mii_data = if responds { self.phy.read(r, now) } else { MII_ABSENT };
                } else if responds {
                    let was_enabled = self.phy.wire_enabled();
                    let was_looping = self.looping_back();
                    self.phy.write(r, self.mii_data_w, now);
                    if (!was_enabled && self.phy.wire_enabled()) || (was_looping && !self.looping_back()) {
                        *wake_rx = true;
                    }
                }
                self.set(off, val);
            }
            MII_DATA_W => {
                self.mii_data_w = val as u16;
                self.set(off, val);
            }
            TX_PRODUCE => {
                self.set(off, val);
                if self.tx_enabled() {
                    self.do_tx(dma, val, wire);
                }
            }
            MAC_CONTROL => {
                if val & MCR_RESET != 0 {
                    self.reset_registers(true);
                    self.set(off, MCR_RESET | MCR_ARB_DIAG_IDLE);
                    return;
                }
                let was_enabled = self.tx_enabled();
                self.set(off, val);
                *wake_rx = true;
                // Enabling starts the engine (the diagnostic queues first);
                // only on the transition, or a rewrite sends them twice.
                if !was_enabled && self.tx_enabled() {
                    let produce = self.get(TX_PRODUCE);
                    self.do_tx(dma, produce, wire);
                }
            }
            RX_PRODUCE => {
                // Bit 31 is a command: a bare index write must not disarm.
                if val & RPIR_ARM != 0 {
                    self.rx_timer_armed = true;
                }
                self.set(off, val);
                // Posting buffers restarts placement.
                *wake_rx = true;
            }
            RX_CONSUME => {
                self.set(off, val);
                // Freed entries can unblock a waiting frame.
                *wake_rx = true;
            }
            COUNTERS | BUFFER_INDEX => {
                let active = MCR_TX_DMA_EN | MCR_TX_EN | MCR_RX_DMA_EN | MCR_RX_EN | MCR_RESET;
                if self.get(MAC_CONTROL) & active != 0 {
                    // Ignored with the MAC enabled or in reset.
                    return;
                }
                // Stopped-MAC storage approximation, not live buffer positions.
                self.set(off, val & if off == BUFFER_INDEX { 0xffff_df1f } else { u32::MAX });
            }
            MII_DATA_R => {} // read-only
            _ => self.set(off, val),
        }
    }
}

/// How the MAC drives its interrupt line: called with each new level.
type IrqFn = Box<dyn Fn(bool) + Send + Sync>;

/// IOC3's Ethernet.
///
/// Lock order: `link`, then `st`; `irq_order` before `st` and before `irq`. The receive thread delivers a frame into guest memory
/// with `st` held, as a register write does a transmit: the DMA goes down
/// through the Bridge and HEART, which never call back into this device.
pub struct Ioc3Eth {
    mac: [u8; 6],
    thread_prefix: &'static str,
    st: Mutex<State>,
    dma: Mutex<Option<Arc<dyn PciDma>>>,
    irq: Mutex<Option<IrqFn>>,
    /// Serialises delivery of level changes to the interrupt callback.
    irq_order: Mutex<()>,
    link: Mutex<Option<crate::net::Rings>>,
    rx_wake: Mutex<Option<Arc<crate::net::Wake>>>,
    /// Between `start` and `stop`; the NAT engine watches it too.
    running: Arc<AtomicBool>,
    /// The flag the network backend watches, set and cleared with `running`.
    /// It belongs to the link, which arrives after `running` exists and
    /// whose backend this device runs on its own thread.
    net_running: Mutex<Option<Arc<AtomicBool>>>,
    /// The network backend, between runs. A NAT engine keeps its flows across
    /// a stop.
    backend: Arc<Mutex<Option<Box<dyn crate::net::NetBackend>>>>,
    /// The NAT and receive threads, joined in `stop`.
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
    me: Weak<Self>,
    /// The status bar's activity bits, shared with the display thread.
    heartbeat: OnceLock<Arc<AtomicU64>>,
}

/// How long the receive thread sleeps with no negotiation to wait for: it is
/// woken by the guest and the NAT engine, so this is only a bound.
const RX_IDLE: Duration = Duration::from_secs(60);

impl Ioc3Eth {
    /// `mac` is the station address the board's one-wire chip carries. The
    /// guest programs the receive filter itself; this is kept for the
    /// integrator (`mac()`).
    pub fn new(mac: [u8; 6]) -> Arc<Self> {
        Self::with_negotiation_delay(mac, NEGOTIATION_DELAY)
    }

    /// As [`Ioc3Eth::new`], with the PHY's synthetic negotiation delay set.
    pub fn with_negotiation_delay(mac: [u8; 6], delay: Duration) -> Arc<Self> {
        Self::with_phy(mac, Box::new(Ics1890::new(PHY_ADDR, delay)), "ip30-eth")
    }

    /// Fuel uses the DP83843 at management address 1.
    pub fn new_fuel(mac: [u8; 6]) -> Arc<Self> {
        Self::with_phy(mac, Box::new(crate::dev::dp83843::Dp83843::new(PHY_ADDR)), "ioc3-eth")
    }

    fn with_phy(mac: [u8; 6], phy: Box<dyn MiiPhy>, thread_prefix: &'static str) -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            mac,
            thread_prefix,
            st: Mutex::new(State::new(phy)),
            dma: Mutex::new(None),
            irq: Mutex::new(None),
            irq_order: Mutex::new(()),
            link: Mutex::new(None),
            rx_wake: Mutex::new(None),
            running: Arc::new(AtomicBool::new(false)),
            net_running: Mutex::new(None),
            backend: Arc::new(Mutex::new(None)),
            threads: Mutex::new(Vec::new()),
            me: me.clone(),
            heartbeat: OnceLock::new(),
        })
    }

    pub fn mac(&self) -> [u8; 6] {
        self.mac
    }

    /// Light the status bar's Ethernet activity bits in `heartbeat`.
    pub fn set_heartbeat(&self, heartbeat: Arc<AtomicU64>) {
        let _ = self.heartbeat.set(heartbeat);
    }

    fn blink(&self, bit: u64) {
        if let Some(hb) = self.heartbeat.get() {
            hb.fetch_or(bit, Ordering::Relaxed);
        }
    }

    /// A machine reset: the MAC's registers and the PHY as at power-on,
    /// the interrupt dropped. The network link stays. The drop is delivered
    /// under `irq_order`, so a raise the receive thread was delivering
    /// lands before it rather than after, which would leave the line high
    /// with the state saying low.
    pub fn reset(&self) {
        let _order = self.irq_order.lock();
        self.st.lock().power_on();
        if let Some(cb) = self.irq.lock().as_ref() {
            cb(false);
        }
        self.wake_rx();
    }

    fn dma(&self) -> Arc<dyn PciDma> {
        self.dma.lock().clone().unwrap_or_else(|| Arc::new(NullDma))
    }

    /// Deliver queued interrupt level changes, in order, with no lock held
    /// that the callback could need.
    fn irq_flush(&self) {
        let _order = self.irq_order.lock();
        let events = std::mem::take(&mut self.st.lock().irq_events);
        if events.is_empty() {
            return;
        }
        if let Some(cb) = self.irq.lock().as_ref() {
            for level in events {
                cb(level);
            }
        }
    }

    fn send(&self, frames: Vec<Vec<u8>>) {
        if frames.is_empty() {
            return;
        }
        self.blink(crate::dev::ng1::rex3::Rex3::HB_ENET_TX);
        if let Some(link) = self.link.lock().as_mut() {
            for f in frames {
                let _ = link.to_net.push(f);
            }
            link.tx_wake.notify();
        }
    }

    fn wake_rx(&self) {
        if let Some(w) = self.rx_wake.lock().as_ref() {
            w.notify();
        }
    }

    /// Connect to the backend `[network]` selected: frames sent go on
    /// `link.to_net`, frames on `link.from_net` are received, and
    /// `link.rx_wake` is notified when the guest does something that can
    /// unblock a waiting frame. The backend and receive threads run between
    /// `start` and `stop`; a NAT engine, which only trades frames with the
    /// host and never reaches the guest, keeps its flows across a stop.
    pub fn attach(&self, mut link: crate::net::Link) {
        let backend = link.backend();
        let running = link.running();
        let rings = link.into_rings();
        *self.rx_wake.lock() = Some(rings.rx_wake.clone());
        *self.link.lock() = Some(rings);
        *self.net_running.lock() = Some(running.clone());
        *self.backend.lock() = backend;
        if self.running.load(Ordering::SeqCst) {
            running.store(true, Ordering::SeqCst);
            self.spawn_nat();
        }
    }

    fn spawn_nat(&self) {
        let slot = self.backend.clone();
        let Some(mut engine) = slot.lock().take() else { return };
        let thread_name = format!("{}-net", self.thread_prefix);
        let t = std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                engine.run();
                *slot.lock() = Some(engine);
            })
            .expect("ioc3 eth NAT thread");
        self.threads.lock().push(t);
    }

    /// Take frames into guest memory as they arrive and as the guest makes
    /// room, until the device stops. The wait ends early for a wake from the
    /// guest or the NAT engine, or for the PHY's negotiation deadline.
    fn rx_loop(me: Weak<Self>, running: Arc<AtomicBool>, wake: Arc<crate::net::Wake>) {
        while running.load(Ordering::SeqCst) {
            let timeout = {
                let Some(this) = me.upgrade() else { return };
                let now = Instant::now();
                let mut st = this.st.lock();
                st.phy.sync(now);
                st.phy.deadline().and_then(|d| d.checked_duration_since(now)).unwrap_or(RX_IDLE)
            };
            wake.wait(timeout);
            if !running.load(Ordering::SeqCst) {
                return;
            }
            let Some(this) = me.upgrade() else { return };
            this.poll();
        }
    }

    /// Name the long-lived locks in the lock registry, which holds only a
    /// weak reference to the device.
    pub fn register_locks(self: &Arc<Self>) {
        for (what, probe) in [
            ("st", (|e: &Ioc3Eth| e.st.is_locked()) as fn(&Ioc3Eth) -> bool),
            ("link", |e| e.link.is_locked()),
            ("irq_order", |e| e.irq_order.is_locked()),
        ] {
            let me = Arc::downgrade(self);
            crate::locks::register_lock(format!("ioc3-eth::{what}"), move || me.upgrade().is_some_and(|e| probe(&e)));
        }
    }

    /// The packet backend's cable state; the PHY reports it as link status.
    pub fn set_backend_link(&self, up: bool) {
        self.st.lock().phy.set_backend_link(up, Instant::now());
        self.wake_rx();
    }

    /// Frames sent and received so far.
    pub fn counts(&self) -> (u64, u64) {
        let s = self.st.lock();
        (s.tx_frames, s.rx_frames)
    }

    /// Take frames from the backend into the buffers the driver has posted.
    /// Frames the filter rejects are discarded; a frame that cannot be taken
    /// yet (no buffer, MAC off) waits in the backend queue. Returns whether
    /// any frame was consumed.
    ///
    /// The link is held only to take one frame at a time, so a transmit on
    /// the CPU thread never waits behind a burst of deliveries. A frame
    /// taken just as the guest turns receive off is dropped, as one
    /// arriving a moment later would be.
    pub fn poll(&self) -> bool {
        let dma = self.dma();
        let mut moved = false;
        loop {
            let frame = {
                let mut link = self.link.lock();
                let Some(link) = link.as_mut() else { break };
                {
                    let mut st = self.st.lock();
                    st.phy.sync(Instant::now());
                    if !st.rx_ready() {
                        break;
                    }
                }
                let Ok(frame) = link.from_net.pop() else { break };
                frame
            };
            moved = true;
            {
                let mut st = self.st.lock();
                if st.rx_ready() && st.rx_filter(&frame) {
                    st.deliver(&*dma, &frame, false);
                    self.blink(crate::dev::ng1::rex3::Rex3::HB_ENET_RX);
                }
            }
            self.irq_flush();
        }
        moved
    }
}

impl Device for Ioc3Eth {
    fn step(&self, _cycles: u64) {}

    /// Start the receive thread, and the NAT engine's if one is attached.
    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }
        let wake = self.rx_wake.lock().get_or_insert_with(crate::net::Wake::new).clone();
        let (me, running) = (self.me.clone(), self.running.clone());
        let thread_name = format!("{}-rx", self.thread_prefix);
        let rx = std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || Self::rx_loop(me, running, wake))
            .expect("ioc3 eth rx thread");
        self.threads.lock().push(rx);
        if let Some(r) = self.net_running.lock().as_ref() {
            r.store(true, Ordering::SeqCst);
        }
        self.spawn_nat();
        // Frames queued while stopped.
        self.wake_rx();
    }

    /// Join the threads: no frame reaches guest memory and no line moves
    /// afterwards. Frames the NAT engine takes meanwhile queue for `start`.
    fn stop(&self) {
        if !self.running.swap(false, Ordering::SeqCst) {
            return;
        }
        if let Some(r) = self.net_running.lock().as_ref() {
            r.store(false, Ordering::SeqCst);
        }
        self.wake_rx();
        if let Some(l) = self.link.lock().as_ref() {
            l.tx_wake.notify();
        }
        // The receive thread can hold the last reference for a moment, and
        // then this runs on it; it cannot join itself.
        let me = std::thread::current().id();
        let threads: Vec<_> = self.threads.lock().drain(..).collect();
        for t in threads {
            if t.thread().id() != me {
                let _ = t.join();
            }
        }
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    fn get_clock(&self) -> u64 {
        0
    }
}

impl Resettable for Ioc3Eth {
    fn power_on(&self) {
        self.reset();
    }
}

impl Drop for Ioc3Eth {
    fn drop(&mut self) {
        Device::stop(&*self);
    }
}

impl Ioc3EthPort for Ioc3Eth {
    fn read32(&self, off: u32) -> u32 {
        if !(MAC_CONTROL..=MII_DATA_W).contains(&off) || off & 3 != 0 {
            return 0;
        }
        let v = self.st.lock().read(off, Instant::now());
        self.irq_flush();
        v
    }

    fn write32(&self, off: u32, val: u32) {
        if !(MAC_CONTROL..=MII_DATA_W).contains(&off) || off & 3 != 0 {
            return;
        }
        let dma = self.dma();
        let mut wire = Vec::new();
        let mut wake = false;
        {
            let mut st = self.st.lock();
            let before = st.phy.deadline();
            st.write(off, val, Instant::now(), &*dma, &mut wire, &mut wake);
            // A negotiation started or ended: the receive thread's wait moves.
            wake |= st.phy.deadline() != before;
        }
        self.send(wire);
        self.irq_flush();
        if wake {
            self.wake_rx();
        }
    }

    fn connect(&self, dma: Arc<dyn PciDma>, irq: Box<dyn Fn(bool) + Send + Sync>) {
        *self.dma.lock() = Some(dma);
        *self.irq.lock() = Some(irq);
        self.irq_flush();
    }
}
