//! Register-contract tests for the IOC3 Ethernet, ported from the QEMU
//! model's qtests: a fake bus-master memory, a fake backend queue pair and an
//! interrupt log stand in for the rest of the machine.
//!
//! Ported from tests/qtest/sgi-ioc3-eth-test.c in jamesbraid/qemu-sgi-origami.

use super::*;
use std::sync::Mutex;

const MEM_SIZE: usize = 0x80_0000;
const RX_RING: u32 = 0x1_0000;
const RX_BUFS: u32 = 0x2_0000;
const TX_RING: u32 = 0x4_0000;
const TX_BUFS: u32 = 0x6_0000;
const STATION: [u8; 6] = [0x08, 0x00, 0x69, 0x12, 0x34, 0x56];

struct Mem(Mutex<Vec<u8>>);

impl PciDma for Mem {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        let m = self.0.lock().unwrap();
        let a = addr as usize;
        if a.saturating_add(buf.len()) > m.len() {
            buf.fill(0);
            return PciDmaStatus::MasterAbort;
        }
        buf.copy_from_slice(&m[a..a + buf.len()]);
        PciDmaStatus::Ok
    }
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        let mut m = self.0.lock().unwrap();
        let a = addr as usize;
        if a.saturating_add(data.len()) > m.len() {
            return PciDmaStatus::MasterAbort;
        }
        m[a..a + data.len()].copy_from_slice(data);
        PciDmaStatus::Ok
    }
}

struct Rig {
    eth: Arc<Ioc3Eth>,
    mem: Arc<Mem>,
    irq: Arc<Mutex<Vec<bool>>>,
    sent: rtrb::Consumer<Vec<u8>>,
    inject: rtrb::Producer<Vec<u8>>,
}

fn rig() -> Rig {
    let eth = Ioc3Eth::with_negotiation_delay(STATION, Duration::ZERO);
    let mem = Arc::new(Mem(Mutex::new(vec![0; MEM_SIZE])));
    let irq = Arc::new(Mutex::new(Vec::new()));
    let log = irq.clone();
    eth.connect(mem.clone(), Box::new(move |l| log.lock().unwrap().push(l)));
    let (link, sent, inject) = crate::net::Link::loopback();
    eth.attach(link);
    Rig { eth, mem, irq, sent, inject }
}

impl Rig {
    fn w(&self, off: u32, v: u32) {
        self.eth.write32(off, v);
    }
    fn r(&self, off: u32) -> u32 {
        self.eth.read32(off)
    }
    fn put(&self, addr: u32, data: &[u8]) {
        assert!(self.mem.dma_write(addr.into(), data).is_ok());
    }
    fn get(&self, addr: u32, n: usize) -> Vec<u8> {
        let mut v = vec![0; n];
        let _ = self.mem.dma_read(addr.into(), &mut v);
        v
    }
    fn get32(&self, addr: u32) -> u32 {
        u32::from_be_bytes(self.get(addr, 4).try_into().unwrap())
    }
    fn irq_level(&self) -> bool {
        self.irq.lock().unwrap().last().copied().unwrap_or(false)
    }

    /// Post `n` receive buffers, 2 KB apart, and enable receive with a
    /// 66-byte offset (33 halfwords).
    fn rx_setup(&self, n: u32) {
        for i in 0..n {
            self.put(RX_RING + i * 8, &u64::from(RX_BUFS + i * 0x800).to_be_bytes());
        }
        self.w(RX_RING_LO, RX_RING);
        self.w(RX_CONSUME, 0);
        self.w(RX_TIMER, 0);
        self.w(RX_CONTROL, 0);
        self.w(INT_ENABLE, INT_RX_TIMER | INT_RX_THRESHOLD);
        self.w(RX_PRODUCE, (n * 8) | RPIR_ARM);
        self.w(MAC_CONTROL, MCR_RX_DMA_EN | MCR_RX_EN | (33 << MCR_RXOFF_SHIFT));
        self.w(MAC_ADDR_LO, u32::from_le_bytes(STATION[..4].try_into().unwrap()));
        self.w(MAC_ADDR_HI, u32::from(STATION[4]) | u32::from(STATION[5]) << 8);
    }

    fn deliver(&mut self, frame: &[u8]) {
        self.inject.push(frame.to_vec()).unwrap();
        self.eth.poll();
    }

    fn tx_setup(&self, mcr_extra: u32) {
        self.w(TX_RING_LO, TX_RING);
        self.w(TX_CONSUME, 0);
        self.w(TX_PRODUCE, 0);
        let mcr = self.r(MAC_CONTROL);
        self.w(MAC_CONTROL, mcr | MCR_TX_DMA_EN | MCR_TX_EN | mcr_extra);
    }

    fn desc(&self, idx: u32, cmd: u32, bufcnt: u32, b1: u64, b2: u64, inline: &[u8]) {
        let mut d = [0u8; 128];
        d[0..4].copy_from_slice(&cmd.to_be_bytes());
        d[4..8].copy_from_slice(&bufcnt.to_be_bytes());
        d[8..16].copy_from_slice(&b1.to_be_bytes());
        d[16..24].copy_from_slice(&b2.to_be_bytes());
        d[24..24 + inline.len()].copy_from_slice(inline);
        self.put(TX_RING + idx * 128, &d);
    }

    fn mii(&self, phy: u32, reg: u32) -> u32 {
        self.w(MII_CONTROL, MII_READ | (phy << 5) | reg);
        self.r(MII_DATA_R)
    }
    fn mii_w(&self, phy: u32, reg: u32, v: u32) {
        self.w(MII_DATA_W, v);
        self.w(MII_CONTROL, (phy << 5) | reg);
    }
}

fn frame(dst: [u8; 6], len: usize) -> Vec<u8> {
    let mut f = vec![0u8; len];
    f[..6].copy_from_slice(&dst);
    f[6..12].copy_from_slice(&[0x02, 0, 0xde, 0xad, 0xbe, 0xef]);
    f[12] = 0x08;
    for (i, b) in f.iter_mut().enumerate().skip(14) {
        *b = i as u8;
    }
    f
}

// ---- management interface and the PHY -------------------------------------

#[test]
fn only_the_octane_address_answers_the_scan() {
    let t = rig();
    for a in 0..32 {
        let id1 = t.mii(a, 2);
        let id2 = t.mii(a, 3);
        if a == 1 {
            assert_eq!((id1, id2), (0x0015, 0xf423));
        } else {
            assert_eq!((id1, id2), (0xffff, 0xffff), "address {a}");
        }
    }
}

#[test]
fn mii_control_never_reads_busy() {
    let t = rig();
    t.w(MII_CONTROL, MII_READ | (1 << 5) | 2 | MII_BUSY);
    assert_eq!(t.r(MII_CONTROL) & MII_BUSY, 0);
    // The data-read register is read-only.
    t.w(MII_DATA_R, 0x1234);
    assert_eq!(t.r(MII_DATA_R), 0x0015);
}

#[test]
fn software_reset_restores_defaults_and_self_clears() {
    let t = rig();
    t.mii_w(1, 0, 0x1100); // isolate-free forced mode tweak
    t.mii_w(1, 4, 0x0101);
    t.mii_w(1, 0, 0x8000);
    assert_eq!(t.mii(1, 0), 0x3000);
    assert_eq!(t.mii(1, 4), 0x01e1);
    assert_eq!(t.mii(1, 0x10), 0x0058, "address 1, bit 3 and the fixed read bit 4");
    assert_eq!(t.mii(1, 0x13), 0x4001);
}

#[test]
fn wrong_address_writes_do_not_reach_the_phy() {
    let t = rig();
    t.mii_w(0, 0, 0x0000);
    t.mii_w(31, 0, 0x0000);
    assert_eq!(t.mii(1, 0), 0x3000);
}

#[test]
fn negotiation_completes_and_reports_link() {
    let t = rig();
    let bmsr = t.mii(1, 1);
    assert_eq!(bmsr & 0x0024, 0x0024, "complete and link, got {bmsr:#06x}");
    assert_eq!(t.mii(1, 5), 0x01e1);
    // Link-low is latched: after a loss and recovery, one read reports low.
    t.eth.set_backend_link(false);
    t.eth.set_backend_link(true);
    assert_eq!(t.mii(1, 1) & 0x0004, 0);
    assert_eq!(t.mii(1, 1) & 0x0004, 0x0004);
}

#[test]
fn negotiation_waits_for_its_deadline() {
    let eth = Ioc3Eth::with_negotiation_delay(STATION, Duration::from_secs(3600));
    eth.write32(MII_CONTROL, MII_READ | (1 << 5) | 1);
    let bmsr = eth.read32(MII_DATA_R);
    assert_eq!(bmsr & 0x0024, 0, "no link before the peer answers");
}

#[test]
fn command_override_arms_one_write() {
    let t = rig();
    // Without override the identity does not move.
    t.mii_w(1, 3, 0x1111);
    assert_eq!(t.mii(1, 3), 0xf423);
    // The arming write sets bit 15; the next write may touch identity.
    t.mii_w(1, 0x10, 0x8008);
    assert_eq!(t.mii(1, 0x10) & 0x8000, 0x8000, "reads do not consume it");
    // A wrong-address write does not consume it either.
    t.mii_w(7, 3, 0x2222);
    t.mii_w(1, 3, 0x1111);
    assert_eq!(t.mii(1, 3), 0x1111);
    // Consumed: a second write is refused.
    t.mii_w(1, 3, 0x3333);
    assert_eq!(t.mii(1, 3), 0x1111);
}

#[test]
fn phy_loopback_turns_a_frame_round_and_isolate_stops_it() {
    let mut t = rig();
    t.mii_w(1, 0, 0x4000 | 0x2000 | 0x0100); // forced 100 FD, loopback
    t.rx_setup(4);
    t.tx_setup(0);
    let f = frame([0xff; 6], 64);
    t.put(TX_BUFS, &f);
    t.desc(0, 64 | TXD_BUF1_VALID, 64 << TXD_BUF1_LEN_SHIFT, u64::from(TX_BUFS), 0, &[]);
    t.w(TX_PRODUCE, 128);
    assert!(t.sent.pop().is_err(), "loopback must not reach the backend");
    assert_eq!(t.get(RX_BUFS + 66, 64), f);
    // PHY loopback gets the full receive treatment: FCS and status.
    assert_eq!(t.get32(RX_BUFS) & RXH_BYTECNT_MASK, 68 << RXH_BYTECNT_SHIFT);
    assert_ne!(t.get32(RX_BUFS + 4) & RXE_GOODPKT, 0);
    assert_eq!(t.r(RX_CONSUME), 8);

    // Isolate takes precedence over loopback; management stays usable.
    t.mii_w(1, 0, 0x4000 | 0x0400 | 0x2000 | 0x0100);
    assert_eq!(t.mii(1, 0) & 0x0400, 0x0400);
    t.desc(1, 64 | TXD_BUF1_VALID, 64 << TXD_BUF1_LEN_SHIFT, u64::from(TX_BUFS), 0, &[]);
    t.w(TX_PRODUCE, 256);
    assert_eq!(t.r(RX_CONSUME), 8, "nothing delivered while isolated");
    assert!(t.sent.pop().is_err());
    t.deliver(&frame(STATION, 60));
    assert_eq!(t.r(RX_CONSUME), 8);
}

#[test]
fn power_down_blocks_the_wire_and_drops_link() {
    let mut t = rig();
    t.rx_setup(4);
    t.mii_w(1, 0, 0x3800);
    assert_eq!(t.mii(1, 1) & 0x0004, 0);
    t.deliver(&frame(STATION, 60));
    assert_eq!(t.r(RX_CONSUME), 0);
}

// ---- transmit --------------------------------------------------------------

#[test]
fn inline_frame_is_sent_and_ring_goes_idle() {
    let mut t = rig();
    t.w(INT_ENABLE, INT_TX_EMPTY | INT_TX_COMPLETE);
    t.tx_setup(0);
    let f = frame(STATION, 60);
    t.desc(0, 60 | TXD_DATA_VALID | TXD_INT_DONE, 60, 0, 0, &f);
    t.w(TX_PRODUCE, 128);
    assert_eq!(t.sent.pop().unwrap(), f);
    assert_eq!(t.r(TX_CONSUME), 128 | TX_CONSUME_IDLE);
    assert_eq!(t.r(INT_STATUS), INT_TX_EMPTY | INT_TX_COMPLETE);
    assert!(t.irq_level());
    t.w(INT_STATUS, INT_TX_EMPTY | INT_TX_COMPLETE);
    assert!(!t.irq_level());
    assert_eq!(t.eth.counts().0, 1);
}

#[test]
fn gather_from_two_buffers() {
    let mut t = rig();
    t.tx_setup(0);
    let f = frame(STATION, 100);
    t.put(TX_BUFS, &f[..40]);
    t.put(TX_BUFS + 0x100, &f[40..]);
    let cnt = (40 << TXD_BUF1_LEN_SHIFT) | (60 << TXD_BUF2_LEN_SHIFT);
    t.desc(0, 100 | TXD_BUF1_VALID | TXD_BUF2_VALID, cnt, u64::from(TX_BUFS), u64::from(TX_BUFS + 0x100), &[]);
    t.w(TX_PRODUCE, 128);
    assert_eq!(t.sent.pop().unwrap(), f);
}

#[test]
fn inline_area_boundary_clamps_to_104() {
    let mut t = rig();
    t.tx_setup(0);
    let data: Vec<u8> = (0..104).collect();
    // The length field names 127; the area holds 104.
    t.desc(0, 127 | TXD_DATA_VALID, 127, 0, 0, &data);
    t.w(TX_PRODUCE, 128);
    assert_eq!(t.sent.pop().unwrap(), data);
}

#[test]
fn burst_behind_one_doorbell_and_ring_wraps() {
    let mut t = rig();
    t.tx_setup(0);
    // Start near the end so a burst wraps the 128-descriptor ring.
    t.w(TX_CONSUME, 126 * 128);
    for (n, idx) in [126u32, 127, 0, 1].iter().enumerate() {
        let f = frame(STATION, 60 + n);
        t.desc(*idx, f.len() as u32 | TXD_DATA_VALID, f.len() as u32, 0, 0, &f);
    }
    t.w(TX_PRODUCE, 2 * 128);
    for n in 0..4 {
        assert_eq!(t.sent.pop().unwrap().len(), 60 + n);
    }
    assert_eq!(t.r(TX_CONSUME), 2 * 128 | TX_CONSUME_IDLE);
}

#[test]
fn ring_select_moves_the_wrap() {
    let mut t = rig();
    // With 512 descriptors the 200th index is a real slot, not a wrap.
    let mcr = MCR_TX_DMA_EN | MCR_TX_EN;
    t.w(TX_RING_LO, 0x2_0000 | TX_RING_SEL);
    let f = frame(STATION, 60);
    let mut d = [0u8; 128];
    d[0..4].copy_from_slice(&(60 | TXD_DATA_VALID).to_be_bytes());
    d[4..8].copy_from_slice(&60u32.to_be_bytes());
    d[24..24 + 60].copy_from_slice(&f);
    t.put(0x2_0000 + 200 * 128, &d);
    t.w(TX_CONSUME, 200 * 128);
    t.w(MAC_CONTROL, mcr);
    t.w(TX_PRODUCE, 201 * 128);
    assert_eq!(t.sent.pop().unwrap(), f);
    assert_eq!(t.r(TX_CONSUME), 201 * 128 | TX_CONSUME_IDLE);
}

#[test]
fn unreadable_descriptor_stops_the_walk_on_it() {
    let mut t = rig();
    t.w(TX_RING_LO, 0x7f_c000); // last 16 KB of the fake memory, then past it
    let mcr = MCR_TX_DMA_EN | MCR_TX_EN;
    t.w(TX_CONSUME, 0);
    t.w(MAC_CONTROL, mcr);
    // Descriptor 127 ends at 0x80_0000, the end of memory: readable. Move the
    // base one slot past the end instead.
    t.w(TX_RING_LO, 0x7f_c000 + 0x4000);
    t.w(TX_PRODUCE, 128);
    assert_eq!(t.r(TX_CONSUME) & TX_CONSUME_MASK, 0, "index stays on the entry");
    assert!(t.sent.pop().is_err());
}

#[test]
fn padding_to_the_ethernet_minimum() {
    let mut t = rig();
    t.tx_setup(MCR_PAD_EN);
    let f = frame(STATION, 30);
    t.desc(0, 30 | TXD_DATA_VALID, 30, 0, 0, &f);
    t.w(TX_PRODUCE, 128);
    let out = t.sent.pop().unwrap();
    assert_eq!(out.len(), 60);
    assert_eq!(&out[..30], &f[..]);
    assert!(out[30..].iter().all(|b| *b == 0));
}

#[test]
fn udp_checksum_offload_matches_a_known_sum() {
    let mut t = rig();
    t.tx_setup(0);
    // 192.0.2.1 -> 192.0.2.2, UDP 1000 -> 2000, payload "hi".
    let mut f = vec![0u8; 14 + 20 + 8 + 2];
    f[12] = 0x08;
    f[14] = 0x45;
    f[16..18].copy_from_slice(&30u16.to_be_bytes());
    f[22] = 64;
    f[23] = 17;
    f[26..30].copy_from_slice(&[192, 0, 2, 1]);
    f[30..34].copy_from_slice(&[192, 0, 2, 2]);
    f[34..36].copy_from_slice(&1000u16.to_be_bytes());
    f[36..38].copy_from_slice(&2000u16.to_be_bytes());
    f[38..40].copy_from_slice(&10u16.to_be_bytes());
    f[42..44].copy_from_slice(b"hi");
    t.desc(0, f.len() as u32 | TXD_DATA_VALID | TXD_DOCHECKSUM, f.len() as u32, 0, 0, &f);
    t.w(TX_PRODUCE, 128);
    let out = t.sent.pop().unwrap();
    // Verify by folding the pseudo-header, header and payload: all ones.
    let mut sum = csum_add(&out[26..34]) + 17 + 10 + csum_add(&out[34..44]);
    sum = u32::from(csum_fold(sum));
    assert_eq!(sum, 0xffff);
    assert_ne!(&out[40..42], &[0, 0]);
}

#[test]
fn enabling_the_engine_runs_queued_descriptors_once() {
    let mut t = rig();
    t.w(TX_RING_LO, TX_RING);
    let f = frame(STATION, 60);
    t.desc(0, 60 | TXD_DATA_VALID, 60, 0, 0, &f);
    t.w(TX_PRODUCE, 128); // queued while the engine is off
    assert!(t.sent.pop().is_err());
    t.w(MAC_CONTROL, MCR_TX_DMA_EN | MCR_TX_EN);
    assert_eq!(t.sent.pop().unwrap(), f);
    t.w(MAC_CONTROL, MCR_TX_DMA_EN | MCR_TX_EN | MCR_PAD_EN); // rewrite
    assert!(t.sent.pop().is_err(), "a rewrite must not send twice");
}

// ---- receive ---------------------------------------------------------------

#[test]
fn frame_lands_in_the_oldest_buffer_with_header_and_fcs() {
    let mut t = rig();
    t.rx_setup(4);
    let f = frame(STATION, 60);
    t.deliver(&f);
    assert_eq!(t.get(RX_BUFS + 66, 60), f);
    let w0 = t.get32(RX_BUFS);
    assert_ne!(w0 & RXH_VALID, 0);
    assert_eq!((w0 & RXH_BYTECNT_MASK) >> RXH_BYTECNT_SHIFT, 64, "length includes the phantom FCS");
    assert_eq!(w0 & 0xffff, u32::from(csum_fold(csum_add(&f))));
    let w1 = t.get32(RX_BUFS + 4);
    assert_ne!(w1 & RXE_GOODPKT, 0);
    assert_eq!(w1 & (RXE_BROADCAST | RXE_MULTICAST), 0);
    assert_eq!(t.get(RX_BUFS + 66 + 60, 4), vec![0; 4]);
    assert_eq!(t.r(RX_CONSUME), 8);
    assert_eq!(t.eth.counts().1, 1);
}

#[test]
fn broadcast_and_multicast_flags() {
    let mut t = rig();
    t.rx_setup(4);
    t.deliver(&frame([0xff; 6], 60));
    assert_ne!(t.get32(RX_BUFS + 4) & RXE_BROADCAST, 0);
    // A multicast needs its hash bit.
    let mc = [0x01, 0x00, 0x5e, 0x00, 0x00, 0x01];
    t.deliver(&frame(mc, 60));
    assert_eq!(t.r(RX_CONSUME), 8, "unhashed multicast is filtered");
    let h = mcast_hash(&mc);
    t.w(if h < 32 { HASH_LO } else { HASH_HI }, 1 << (h & 31));
    t.deliver(&frame(mc, 60));
    assert_eq!(t.r(RX_CONSUME), 16);
    assert_ne!(t.get32(RX_BUFS + 0x800 + 4) & RXE_MULTICAST, 0);
}

#[test]
fn frames_for_other_stations_are_filtered_unless_promiscuous() {
    let mut t = rig();
    t.rx_setup(4);
    t.deliver(&frame([0x08, 0, 0x69, 0, 0, 1], 60));
    assert_eq!(t.r(RX_CONSUME), 0);
    let mcr = t.r(MAC_CONTROL);
    t.w(MAC_CONTROL, mcr | MCR_PROMISC);
    t.deliver(&frame([0x08, 0, 0x69, 0, 0, 1], 60));
    assert_eq!(t.r(RX_CONSUME), 8);
}

#[test]
fn receive_interrupts_timer_and_threshold() {
    let mut t = rig();
    t.rx_setup(4);
    t.w(RX_CONTROL, 2); // tell me when two or fewer buffers remain
    t.deliver(&frame(STATION, 60)); // 3 left
    assert_eq!(t.r(INT_STATUS), INT_RX_TIMER);
    assert!(t.irq_level());
    t.w(INT_STATUS, INT_RX_TIMER);
    assert!(!t.irq_level());
    t.deliver(&frame(STATION, 60)); // 2 left: at-or-below the threshold
    assert_eq!(t.r(INT_STATUS), INT_RX_TIMER | INT_RX_THRESHOLD);
}

#[test]
fn a_bare_index_write_does_not_disarm_the_timer() {
    let mut t = rig();
    t.rx_setup(4);
    t.w(RX_PRODUCE, 4 * 8); // no arm bit
    t.deliver(&frame(STATION, 60));
    assert_ne!(t.r(INT_STATUS) & INT_RX_TIMER, 0);
}

#[test]
fn dry_ring_holds_the_frame_until_a_buffer_is_posted() {
    let mut t = rig();
    t.rx_setup(1);
    t.deliver(&frame(STATION, 60));
    assert_eq!(t.r(RX_CONSUME), 8);
    t.deliver(&frame(STATION, 61));
    assert_eq!(t.r(RX_CONSUME), 8, "no buffer: the frame waits");
    t.put(RX_RING + 8, &u64::from(RX_BUFS + 0x800).to_be_bytes());
    t.w(RX_PRODUCE, 16);
    t.eth.poll();
    assert_eq!(t.r(RX_CONSUME), 16);
    assert_eq!(t.get(RX_BUFS + 0x800 + 66, 61)[..], frame(STATION, 61)[..]);
}

#[test]
fn receive_ring_wraps_at_512_entries() {
    let mut t = rig();
    t.rx_setup(1);
    t.w(RX_CONSUME, 511 * 8);
    t.put(RX_RING + 511 * 8, &u64::from(RX_BUFS).to_be_bytes());
    t.w(RX_PRODUCE, 0);
    t.deliver(&frame(STATION, 60));
    assert_eq!(t.r(RX_CONSUME), 0);
}

#[test]
fn oversize_frames_are_truncated_and_flagged() {
    let mut t = rig();
    t.rx_setup(2);
    t.deliver(&vec![0xff; 2100]);
    let w1 = t.get32(RX_BUFS + 4);
    assert_ne!(w1 & RXE_LONGEVENT, 0);
    assert_ne!(w1 & RXE_BADPKT, 0);
    let w0 = t.get32(RX_BUFS);
    assert_eq!((w0 & RXH_BYTECNT_MASK) >> RXH_BYTECNT_SHIFT, 2044);
}

#[test]
fn unreadable_ring_entry_drops_the_frame() {
    let mut t = rig();
    t.rx_setup(2);
    // Entry names a buffer past the end of memory: the write fails, the
    // frame is dropped and the index stays.
    t.put(RX_RING, &0x7f_ffffu64.to_be_bytes());
    t.deliver(&frame(STATION, 60));
    assert_eq!(t.r(RX_CONSUME), 0);
    assert_eq!(t.get32(0x7f_ffff & !3) & RXH_VALID, 0);
}

#[test]
fn mac_loopback_is_turned_round_before_the_fcs() {
    let mut t = rig();
    t.rx_setup(2);
    let mcr = t.r(MAC_CONTROL);
    t.w(MAC_CONTROL, mcr | MCR_LOOPBACK);
    t.tx_setup(0);
    let f = frame(STATION, 64);
    t.desc(0, 64 | TXD_DATA_VALID, 64, 0, 0, &f);
    t.w(TX_PRODUCE, 128);
    assert!(t.sent.pop().is_err());
    assert_eq!(t.get(RX_BUFS + 66, 64), f);
    assert_eq!((t.get32(RX_BUFS) & RXH_BYTECNT_MASK) >> RXH_BYTECNT_SHIFT, 64);
    assert_eq!(t.get32(RX_BUFS + 4), 0, "no status vector in MAC loopback");
    // Wire traffic is ignored while looping back.
    t.deliver(&frame(STATION, 60));
    assert_eq!(t.r(RX_CONSUME), 8);
}

// ---- interrupts and registers ------------------------------------------------

#[test]
fn enabling_a_set_source_raises_the_line() {
    let mut t = rig();
    t.rx_setup(2);
    t.w(INT_ENABLE, 0);
    t.deliver(&frame(STATION, 60));
    assert!(!t.irq_level());
    t.w(INT_ENABLE, INT_RX_TIMER);
    assert!(t.irq_level());
}

#[test]
fn a_completion_landing_on_a_set_bit_comes_back_as_an_edge() {
    let mut t = rig();
    t.rx_setup(4);
    t.deliver(&frame(STATION, 60));
    t.deliver(&frame(STATION, 60)); // same bit while unacknowledged
    let status = t.r(INT_STATUS);
    t.irq.lock().unwrap().clear();
    t.w(INT_STATUS, status);
    assert_eq!(*t.irq.lock().unwrap(), vec![false, true], "low then high, not silently absorbed");
    assert_ne!(t.r(INT_STATUS) & INT_RX_TIMER, 0);
    t.w(INT_STATUS, INT_RX_TIMER | INT_RX_THRESHOLD);
    assert_eq!(t.r(INT_STATUS), 0);
}

#[test]
fn reset_keeps_ring_bases_and_reports_idle() {
    let t = rig();
    t.w(RX_RING_LO, RX_RING);
    t.w(TX_RING_LO, TX_RING | 1);
    t.w(RX_CONTROL, 5);
    t.w(MAC_CONTROL, MCR_RESET);
    assert_eq!(t.r(MAC_CONTROL), MCR_RESET | MCR_ARB_DIAG_IDLE);
    assert_eq!(t.r(RX_RING_LO), RX_RING);
    assert_eq!(t.r(TX_RING_LO), TX_RING | 1);
    assert_eq!(t.r(RX_CONTROL), 0);
}

#[test]
fn counters_clear_on_read_and_storage_is_stopped_mac_only() {
    let t = rig();
    t.w(COUNTERS, 0xa5a5_1234);
    assert_eq!(t.r(COUNTERS), 0xa5a5_1234);
    assert_eq!(t.r(COUNTERS), 0);
    t.w(BUFFER_INDEX, 0xffff_ffff);
    assert_eq!(t.r(BUFFER_INDEX), 0xffff_df1f);
    t.w(MAC_CONTROL, MCR_RX_EN);
    t.w(COUNTERS, 7);
    t.w(BUFFER_INDEX, 0);
    assert_eq!(t.r(COUNTERS), 0);
    assert_eq!(t.r(BUFFER_INDEX), 0xffff_df1f, "writes ignored with the MAC enabled");
}

#[test]
fn offsets_outside_the_block_are_inert() {
    let t = rig();
    t.w(0x154, 5);
    t.w(0xec, 5);
    assert_eq!(t.r(0x154), 0);
    assert_eq!(t.r(0xec), 0);
}

#[test]
fn the_station_address_is_kept_for_the_integrator() {
    let t = rig();
    assert_eq!(t.eth.mac(), STATION);
}

/// Records the bus address of each transfer.
struct AddrProbe(Mutex<Vec<u64>>);

impl PciDma for AddrProbe {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        self.0.lock().unwrap().push(addr);
        buf.fill(0);
        PciDmaStatus::Ok
    }
    fn dma_write(&self, addr: u64, _data: &[u8]) -> PciDmaStatus {
        self.0.lock().unwrap().push(addr);
        PciDmaStatus::Ok
    }
}

#[test]
fn dma_above_4_gib_reaches_the_bus_in_full() {
    let probe = AddrProbe(Mutex::new(Vec::new()));
    let mut b = [0u8; 4];
    assert!(dma_read(&probe, 0x1_2345_6000, &mut b));
    assert!(dma_write(&probe, 0xffff_ffff_ffff_f000, &b));
    // A transfer that would wrap past the top of the space is refused.
    assert!(!dma_write(&probe, 0xffff_ffff_ffff_fffe, &b));
    assert_eq!(*probe.0.lock().unwrap(), [0x1_2345_6000, 0xffff_ffff_ffff_f000]);
}

/// A receive buffer the guest put at the top of the 64-bit space: the
/// header and frame addresses would wrap, so the frame is dropped like any
/// refused transfer and the index stays.
#[test]
fn a_receive_buffer_at_the_top_of_the_bus_drops_the_frame() {
    let mut t = rig();
    t.rx_setup(2);
    t.put(RX_RING, &0xffff_ffff_ffff_fff0u64.to_be_bytes());
    t.deliver(&frame(STATION, 60));
    assert_eq!(t.r(RX_CONSUME), 0);
    // The next buffer is reachable again.
    t.put(RX_RING, &u64::from(RX_BUFS).to_be_bytes());
    t.deliver(&frame(STATION, 60));
    assert_eq!(t.r(RX_CONSUME), 8);
}

/// A transmit ring based at the top of the space: a descriptor whose
/// address would wrap stops the walk on it rather than overflowing.
#[test]
fn a_transmit_ring_at_the_top_of_the_bus_stops_the_walk() {
    let mut t = rig();
    t.w(TX_RING_HI, 0xffff_ffff);
    t.w(TX_RING_LO, 0xffff_c000 | TX_RING_SEL);
    t.w(TX_CONSUME, 0x8000);
    t.w(MAC_CONTROL, MCR_TX_DMA_EN | MCR_TX_EN);
    t.w(TX_PRODUCE, 0x8080);
    assert_eq!(t.r(TX_CONSUME) & TX_CONSUME_MASK, 0x8000, "index stays on the entry");
    assert!(t.sent.pop().is_err());
}

/// A reset that lands while the receive thread is delivering a raise
/// leaves the line low: the reset's drop goes out after the raise. The
/// test plays the receive thread's part of `irq_flush` by hand, stopping
/// between taking the queued raise and delivering it.
#[test]
fn a_reset_during_an_interrupt_delivery_leaves_the_line_low() {
    let t = rig();
    t.w(INT_ENABLE, INT_RX_TIMER);
    let order = t.eth.irq_order.lock();
    let events = {
        let mut st = t.eth.st.lock();
        st.raise_int(INT_RX_TIMER);
        std::mem::take(&mut st.irq_events)
    };
    assert_eq!(events, vec![true]);
    std::thread::scope(|s| {
        let reset = s.spawn(|| t.eth.reset());
        // Give a reset that does not wait its chance to drop the line first.
        let start = std::time::Instant::now();
        while t.irq.lock().unwrap().is_empty() && start.elapsed() < Duration::from_millis(100) {
            std::thread::yield_now();
        }
        if let Some(cb) = t.eth.irq.lock().as_ref() {
            for level in events {
                cb(level);
            }
        }
        drop(order);
        reset.join().unwrap();
    });
    let levels = t.irq.lock().unwrap().clone();
    assert_eq!(levels.last(), Some(&false), "levels: {levels:?}");
}

/// While stopped no frame reaches guest memory; on start the queued one is
/// taken, with nobody calling `poll`.
#[test]
fn a_stopped_mac_takes_no_frames_until_it_starts() {
    let mut t = rig();
    let (link, _sent, inject) = crate::net::Link::loopback();
    let wake = link.rx_wake();
    t.eth.attach(link);
    t.inject = inject;
    t.rx_setup(2);
    t.eth.start();
    t.inject.push(frame(STATION, 60)).unwrap();
    wake.notify();
    assert!(waits_for(|| t.r(RX_CONSUME) == 8), "taken while running");
    t.eth.stop();
    assert!(!t.eth.is_running());
    assert!(t.eth.threads.lock().is_empty());
    t.inject.push(frame(STATION, 60)).unwrap();
    wake.notify();
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(t.r(RX_CONSUME), 8, "nothing taken while stopped");
    t.eth.start();
    assert!(waits_for(|| t.r(RX_CONSUME) == 16), "the queued frame is taken on start");
    t.eth.stop();
}

/// A frame held back by the PHY still negotiating is taken when the
/// negotiation ends, with no wake from anyone: the receive thread's wait ends
/// at the PHY's deadline.
#[test]
fn a_frame_waits_for_the_negotiation_deadline_without_a_wake() {
    let eth = Ioc3Eth::with_negotiation_delay(STATION, Duration::from_millis(60));
    let mem = Arc::new(Mem(Mutex::new(vec![0; MEM_SIZE])));
    eth.connect(mem.clone(), Box::new(|_| {}));
    let (link, _sent, mut inject) = crate::net::Link::loopback();
    eth.attach(link);
    let t = Rig { eth, mem, irq: Arc::new(Mutex::new(Vec::new())), sent: rtrb::RingBuffer::new(1).1, inject: rtrb::RingBuffer::new(1).0 };
    t.rx_setup(2);
    t.eth.start();
    inject.push(frame(STATION, 60)).unwrap();
    assert!(waits_for(|| t.r(RX_CONSUME) == 8), "taken after the negotiation");
    t.eth.stop();
}

fn waits_for(cond: impl Fn() -> bool) -> bool {
    let end = Instant::now() + Duration::from_secs(2);
    while Instant::now() < end {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    cond()
}

/// Stopping joins the device's NAT and receive threads and a restart brings
/// both back with the NAT engine's state intact; dropping a running device
/// ends them.
#[test]
fn stopping_the_mac_joins_its_threads() {
    let eth = Ioc3Eth::with_negotiation_delay(STATION, Duration::ZERO);
    let mut net = crate::config::NetworkConfig::default();
    net.port_forward.clear();
    eth.attach(crate::net::Link::open(&net).unwrap().expect("NAT is the default"));
    assert!(eth.threads.lock().is_empty(), "no thread before start");
    eth.start();
    assert_eq!(eth.threads.lock().len(), 2);
    eth.stop();
    assert!(eth.threads.lock().is_empty());
    assert!(eth.backend.lock().is_some(), "the engine is back for the next run");
    eth.start();
    assert_eq!(eth.threads.lock().len(), 2);
    let probe = Arc::downgrade(&eth);
    drop(eth);
    assert!(probe.upgrade().is_none());
}

#[test]
fn fuel_phy_mdio_decoder_and_machine_reset() {
    let eth = Ioc3Eth::new_fuel(STATION);
    assert_eq!(eth.mac(), STATION);
    let read = |phy: u32, r: u32| {
        eth.write32(MII_CONTROL, MII_READ | (phy << 5) | r);
        eth.read32(MII_DATA_R)
    };
    assert_eq!(read(1, 2), 0x2000);
    assert_eq!(read(1, 3), 0x5c10);
    assert_eq!(read(0, 2), 0xffff);
    eth.write32(MII_DATA_W, 7);
    eth.write32(MII_CONTROL, (1 << 5) | 0x19);
    assert_eq!(read(1, 2), 0xffff);
    assert_eq!(read(7, 2), 0x2000);
    // A wrong-address reset has no effect.
    eth.write32(MII_DATA_W, 0x8000);
    eth.write32(MII_CONTROL, 1 << 5);
    assert_eq!(read(7, 2), 0x2000);
    eth.reset();
    assert_eq!(read(1, 3), 0x5c10);
    assert_eq!(read(7, 2), 0xffff);
    // The MAC soft reset leaves PHY programming intact.
    eth.write32(MII_DATA_W, 0x21);
    eth.write32(MII_CONTROL, (1 << 5) | 4);
    eth.write32(MAC_CONTROL, MCR_RESET);
    assert_eq!(read(1, 4), 0x21);
    // Forced mode exposes the configured backend cable without waiting.
    eth.write32(MII_DATA_W, 0x2100);
    eth.write32(MII_CONTROL, 1 << 5);
    for up in [false, true] {
        eth.set_backend_link(up);
        assert_eq!(read(1, 1) & 4 != 0, up);
    }
}
