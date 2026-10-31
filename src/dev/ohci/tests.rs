// The register and clock cases follow tests/qtest/usb-lucent-ohci-test.c in
// jamesbraid/qemu-sgi-origami: root-hub-descriptor-a-is-writable and
// frame-number-tracks-clock-while-timer-starved. The descriptor engine cases
// drive a test-only function and have no qtest counterpart.
use super::*;
use crate::pci::{PciBusDevice, PciDmaStatus};
use crate::traits::{Device, Resettable};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Memory {
    bytes: Mutex<Vec<u8>>,
    refuse_read: Mutex<Option<u32>>,
    refuse_write: Mutex<Option<u32>>,
    writes: Mutex<Vec<u64>>,
}
impl Memory {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            bytes: Mutex::new(vec![0xa5; 0x10000]),
            refuse_read: Mutex::new(None),
            refuse_write: Mutex::new(None),
            writes: Mutex::new(vec![]),
        })
    }
    fn words(&self, at: u32, words: &[u32]) {
        for (i, w) in words.iter().enumerate() {
            self.bytes.lock()[at as usize + i * 4..at as usize + i * 4 + 4]
                .copy_from_slice(&w.to_le_bytes());
        }
    }
    fn word(&self, at: u32) -> u32 {
        u32::from_le_bytes(
            self.bytes.lock()[at as usize..at as usize + 4]
                .try_into()
                .unwrap(),
        )
    }
}
impl PciDma for Memory {
    fn dma_read(&self, addr: u64, dst: &mut [u8]) -> PciDmaStatus {
        if self
            .refuse_read
            .lock()
            .is_some_and(|x| (addr..addr + dst.len() as u64).contains(&(x as u64)))
        {
            return PciDmaStatus::MasterAbort;
        }
        let b = self.bytes.lock();
        let Some(s) = b.get(addr as usize..addr as usize + dst.len()) else {
            return PciDmaStatus::MasterAbort;
        };
        dst.copy_from_slice(s);
        PciDmaStatus::Ok
    }
    fn dma_write(&self, addr: u64, src: &[u8]) -> PciDmaStatus {
        if self
            .refuse_write
            .lock()
            .is_some_and(|x| (addr..addr + src.len() as u64).contains(&(x as u64)))
        {
            return PciDmaStatus::MasterAbort;
        }
        let mut b = self.bytes.lock();
        let Some(s) = b.get_mut(addr as usize..addr as usize + src.len()) else {
            return PciDmaStatus::MasterAbort;
        };
        s.copy_from_slice(src);
        self.writes.lock().push(addr);
        PciDmaStatus::Ok
    }
}
struct Function {
    address: AtomicUsize,
    calls: AtomicUsize,
    reply: Mutex<UsbOutcome>,
    packets: Mutex<Vec<UsbPacket>>,
}
impl Function {
    fn new(reply: UsbOutcome) -> Arc<Self> {
        Arc::new(Self {
            address: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            reply: Mutex::new(reply),
            packets: Mutex::new(vec![]),
        })
    }
}
impl UsbFunction for Function {
    fn reset(&self) {}
    fn address(&self) -> u8 {
        self.address.load(Ordering::SeqCst) as u8
    }
    fn transact(&self, packet: UsbPacket) -> UsbOutcome {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.packets.lock().push(packet);
        self.reply.lock().clone()
    }
}
fn start_port_reset(h: &Ohci, port: u32) {
    let status = 0x54 + port * 4;
    h.bar_write(0, status, 0x100, 4);
    h.bar_write(0, status, 0x10, 4);
}

fn setup(
    pid: Pid,
    reply: UsbOutcome,
    cbp: u32,
    be: u32,
    flags: u32,
) -> (Arc<Ohci>, Arc<Memory>, Arc<Function>) {
    let h = Ohci::new();
    let m = Memory::new();
    let f = Function::new(reply);
    h.connect_dma(m.clone());
    h.attach(0, f.clone()).unwrap();
    h.start();
    start_port_reset(&h, 0);
    h.advance_frames(10);
    m.words(0x1000, &[0; 64]);
    m.words(0x2000, &[(8 << 16), 0x3020, 0x3000, 0]);
    m.words(
        0x3000,
        &[0xf000_0000 | flags | ((pid as u32) << 19), cbp, 0x3020, be],
    );
    h.config().write(1, 6);
    h.bar_write(0, 0x18, 0x1000, 4);
    h.bar_write(0, 0x20, 0x2000, 4);
    h.bar_write(0, 0x08, 2, 4);
    h.bar_write(0, 0x04, 0x90, 4);
    (h, m, f)
}
/// The guest receives fatal status if the controller cannot access a descriptor.
#[test]
fn dma_refusal_preserves_descriptors_and_reports_a_fatal_error() {
    for periodic in [false, true] {
        for pid in [Pid::Out, Pid::In] {
            let (h, m, f) = setup(pid, UsbOutcome::Ack(vec![5; 8]), 0x4000, 0x4007, 0);
            if periodic {
                m.words(0x1000, &[0x2000; 32]);
                h.bar_write(0, 4, 0x84, 4);
            }
            if pid == Pid::In {
                *m.refuse_write.lock() = Some(0x4000);
            } else {
                *m.refuse_read.lock() = Some(0x4000);
            }
            let before = m.bytes.lock()[0x3000..0x3010].to_vec();
            h.advance_frames(1);
            let frame = h.bar_read(0, 0x3c, 4);
            h.advance_frames(3);
            assert_eq!(f.calls.load(Ordering::SeqCst), usize::from(pid == Pid::In));
            assert_eq!(&m.bytes.lock()[0x3000..0x3010], &before);
            assert_eq!(m.word(0x2008), 0x3000);
            assert_eq!(m.word(0x1084), 0);
            assert_eq!(h.bar_read(0, 0x3c, 4), frame);
            assert_ne!(h.bar_read(0, 0xc, 4) & UE, 0);
            assert_eq!(h.config().read(1) >> 16, 0x8200);
            h.config().write(1, 0x8000_0006);
            assert_eq!(h.config().read(1) >> 16, 0x0200);
            assert_ne!(h.bar_read(0, 0xc, 4) & UE, 0);
        }
    }
}

/// Short and overlong USB replies leave bytes outside the packet untouched.
#[test]
fn short_or_overlong_in_packets_preserve_unwritten_bytes() {
    for (name, payload, end, flags, bytes, current, condition) in [
        ("short", vec![1, 2, 3], 0x401f, (1 << 18) | (3 << 24), [1, 2, 3, 0xa5, 0xa5], 0x4003, 0),
        ("overrun", vec![1; 8], 0x4003, 3 << 24, [1, 1, 1, 1, 0xa5], 0x4000, 8),
    ] {
        let (h, m, _) = setup(Pid::In, UsbOutcome::Ack(payload), 0x4000, end, flags);
        h.advance_frames(1);
        assert_eq!(&m.bytes.lock()[0x4000..0x4005], &bytes, "{name}");
        assert_eq!(m.word(0x3004), current, "{name}");
        assert_eq!(m.word(0x3000) >> 28, condition, "{name}");
        assert_eq!((m.word(0x3000) >> 24) & 3, 2, "{name}");
        if condition == 0 {
            assert_eq!(m.word(0x2008) & !15, 0x3020);
            assert_eq!(m.word(0x1084) & !15, 0x3000);
        }
    }
}

/// The guest retries NAKs, stops on STALL, and retains consistent short-transfer state.
#[test]
fn nak_stall_and_short_in_completions_keep_the_endpoint_consistent() {
    for (name, reply, flags, condition) in [
        ("NAK", UsbOutcome::Nak, 0, 15),
        ("STALL", UsbOutcome::Stall, 3 << 24, 4),
        ("no response", UsbOutcome::NoResponse, 0, 5),
        ("short without rounding", UsbOutcome::Ack(vec![2]), 0, 9),
    ] {
        let (h, m, f) = setup(Pid::In, reply, 0x4000, 0x4007, flags);
        h.bar_write(0, 0x34, 255, 4);
        let before = m.bytes.lock()[0x3000..0x3010].to_vec();
        h.advance_frames(1);
        if condition == 5 {
            assert_eq!(f.calls.load(Ordering::SeqCst), 1);
            assert_eq!((m.word(0x3000) >> 26) & 3, 1);
            assert_eq!(m.word(0x2008), 0x3000);
            h.advance_frames(2);
        }
        assert_eq!(m.word(0x3000) >> 28, condition, "{name}");
        if condition == 15 {
            assert_eq!(&m.bytes.lock()[0x3000..0x3010], &before);
            assert_eq!(m.word(0x2008), 0x3000);
        } else {
            assert_ne!(m.word(0x2008) & 1, 0, "{name}");
        }
        if condition == 4 {
            assert_eq!(m.word(0x3004), 0x4000);
            assert_eq!((m.word(0x3000) >> 24) & 3, 3);
            assert_eq!(m.word(0x2008) & 3, 3);
        }
    }
}

/// Port and controller reset paths retain attached USB functions.
#[test]
fn attached_functions_survive_port_and_controller_resets() {
    let h = Ohci::new();
    h.start();
    h.attach(1, Function::new(UsbOutcome::Nak)).unwrap();
    assert_eq!(h.bar_read(0, 0x58, 4) & 1, 0);
    start_port_reset(&h, 1);
    h.advance_frames(9);
    assert_ne!(h.bar_read(0, 0x58, 4) & 16, 0);
    h.advance_frames(1);
    assert_eq!(h.bar_read(0, 0x58, 4) & 16, 0);
    assert_ne!(h.bar_read(0, 0x58, 4) & (1 << 20), 0);
    h.bar_write(0, 8, 1, 4);
    assert_ne!(h.bar_read(0, 0x58, 4) & 1, 0);
    h.bar_write(0, 0x58, 0x200, 4);
    assert_eq!(h.bar_read(0, 0x58, 4) & 0x117, 0);
    h.bar_write(0, 0x58, PPS, 4);
    assert_ne!(h.bar_read(0, 0x58, 4) & 1, 0);
    h.power_on();
    assert_eq!(h.bar_read(0, 0x58, 4), 0);
    h.bar_write(0, 0x58, PPS, 4);
    assert_ne!(h.bar_read(0, 0x58, 4) & 1, 0);
}

/// The guest observes descriptor updates before the done bit and acknowledges them later.
#[test]
fn done_publication_orders_writes_and_waits_for_acknowledgement() {
    let (h, m, _) = setup(Pid::In, UsbOutcome::Ack(vec![8; 8]), 0x4000, 0x4007, 0);
    h.advance_frames(1);
    {
        let writes = m.writes.lock();
        assert!(writes.iter().position(|a| *a == 0x2008).unwrap() < writes.iter().position(|a| *a == 0x3000).unwrap());
    }
    assert_eq!(m.word(0x1084) & !15, 0x3000);
    m.words(0x3020, &[0xf0000000 | (2 << 19), 0x4100, 0x3040, 0x4107]);
    m.words(0x2004, &[0x3040]);
    h.bar_write(0, 8, 2, 4);
    h.advance_frames(1);
    assert_eq!(m.word(0x1084) & !15, 0x3000);
    assert_eq!(h.bar_read(0, 0x30, 4), 0x3020);
    h.bar_write(0, 0xc, 2, 4);
    h.advance_frames(1);
    assert_eq!(m.word(0x1084) & !15, 0x3020);
    for delay in [0, 2, 7] {
        let (h, m, _) = setup(Pid::In, UsbOutcome::Ack(vec![5; 8]), 0x4000, 0x4007, delay << 21);
        h.advance_frames(1);
        if delay != 0 { assert_eq!(m.word(0x1084), 0); }
        h.advance_frames(u64::from(delay));
        assert_eq!(m.word(0x1084) & !15, if delay == 7 { 0 } else { 0x3000 });
    }
    let (h, m, _) = setup(Pid::In, UsbOutcome::Ack(vec![1; 8]), 0x4000, 0x4007, 0);
    *m.refuse_write.lock() = Some(0x1084);
    h.advance_frames(1);
    assert_ne!(h.bar_read(0, 0xc, 4) & UE, 0);
    assert_eq!(h.bar_read(0, 0xc, 4) & WDH, 0);
    assert_eq!(m.word(0x1084), 0);
}

/// OHCI must yield to the other transfer list when a guest marks one list skipped.
#[test]
fn scheduler_shares_service_and_yields_past_skipped_lists() {
    for (cbsr, skip_control, skip_bulk) in [(0, false, false), (3, false, false), (3, true, false), (0, false, true)] {
        let (h, m, f) = setup(Pid::In, UsbOutcome::Nak, 0x4000, 0x4007, 0);
        m.words(0x2000, &[(8 << 16) | (1 << 7) | (u32::from(skip_control) << 14), 0x3020, 0x3000, if skip_control { 0x2000 } else { 0 }]);
        m.words(0x2100, &[(8 << 16) | (2 << 7) | (u32::from(skip_bulk) << 14), 0x3120, 0x3100, if skip_bulk { 0x2100 } else { 0 }]);
        m.words(0x3100, &[0xf0100000, 0x4100, 0x3120, 0x4107]);
        h.bar_write(0, 0x28, 0x2100, 4);
        h.bar_write(0, 8, 6, 4);
        h.bar_write(0, 4, 0xb0 | cbsr, 4);
        h.advance_frames(1);
        let packets = f.packets.lock();
        let control = packets.iter().filter(|p| p.endpoint == 1).count();
        let bulk = packets.iter().filter(|p| p.endpoint == 2).count();
        assert_eq!(control > 0, !skip_control);
        assert_eq!(bulk > 0, !skip_bulk);
        if skip_bulk {
            assert!(control > 1, "control resumes after its credit is exhausted");
        }
        if !skip_control && !skip_bulk {
            assert!(control >= bulk * (cbsr as usize + 1));
            assert!(control <= bulk * (cbsr as usize + 1) + cbsr as usize + 1);
        }
        assert_eq!(h.bar_read(0, 0xc, 4) & UE, 0);
    }
}

/// The controller routes address zero only while one attached function can claim it.
#[test]
fn address_zero_is_routed_only_to_a_unique_eligible_function() {
    for (name, address, peer_status, first_selected, peer_selected) in [
        ("ambiguous", 0, PPS | 1 | 2, false, false),
        ("disabled peer", 0, PPS | 1, true, false),
        ("assigned first", 5, PPS | 1 | 2, false, true),
    ] {
        let (h, m, f) = setup(Pid::In, UsbOutcome::Nak, 0x4000, 0x4007, 0);
        let peer = Function::new(UsbOutcome::Nak);
        h.attach(1, peer.clone()).unwrap();
        h.state.lock().r[22] = peer_status;
        f.address.store(address, Ordering::SeqCst);
        h.advance_frames(1);
        assert_eq!(f.calls.load(Ordering::SeqCst) > 0, first_selected, "{name}");
        assert_eq!(peer.calls.load(Ordering::SeqCst) > 0, peer_selected, "{name}");
        if address != 0 {
            let peer_calls = peer.calls.load(Ordering::SeqCst);
            m.words(0x2000, &[(8 << 16) | 5]);
            h.advance_frames(1);
            assert!(f.calls.load(Ordering::SeqCst) > 0);
            assert_eq!(peer.calls.load(Ordering::SeqCst), peer_calls);
        } else if !first_selected {
            assert_eq!(m.word(0x3000) >> 28, 5);
            assert_eq!((m.word(0x3000) >> 26) & 3, 3);
        }
    }
}


/// A guest USB packet can cross physically nonadjacent DMA pages.
#[test]
fn packet_crosses_nonadjacent_pages() {
    let (h, m, _) = setup(
        Pid::In,
        UsbOutcome::Ack(vec![1, 2, 3, 4, 5, 6, 7, 8]),
        0x4ffc,
        0x9003,
        0,
    );
    h.advance_frames(1);
    assert_eq!(&m.bytes.lock()[0x4ffc..0x5000], &[1, 2, 3, 4]);
    assert_eq!(&m.bytes.lock()[0x9000..0x9004], &[5, 6, 7, 8]);
    assert_eq!(m.bytes.lock()[0x5000], 0xa5);
    assert_eq!(m.word(0x3004), 0);
}

/// The frame counter catches up after its host timer was delayed.
#[test]
fn clock_read_catches_up_after_timer_starvation() {
    let h = Ohci::new();
    h.start();
    h.bar_write(0, 4, 0x80, 4);
    *h.clock.lock() = Some(Instant::now() - Duration::from_millis(5));
    assert!(h.bar_read(0, 0x3c, 4) >= 5);
    h.stop();
    let frame = h.bar_read(0, 0x3c, 4);
    h.advance_frames(10);
    assert_eq!(h.bar_read(0, 0x3c, 4), frame);
}

/// An interrupt handler can acknowledge a line while the controller raises it.
#[test]
fn reentrant_irq_acknowledgement_leaves_final_line_clear() {
    struct Ack {
        hc: Weak<Ohci>,
        level: AtomicBool,
        once: AtomicBool,
    }
    impl crate::irq::InterruptController for Ack {
        fn set_line(&self, _: u32, level: bool) {
            if level && !self.once.swap(true, Ordering::SeqCst) {
                self.hc.upgrade().unwrap().bar_write(0, 0xc, RHSC, 4);
            }
            self.level.store(level, Ordering::SeqCst);
        }
    }
    let h = Ohci::new();
    let ack = Arc::new(Ack {
        hc: Arc::downgrade(&h),
        level: AtomicBool::new(false),
        once: AtomicBool::new(false),
    });
    h.connect_interrupt(IrqLine::new(ack.clone(), 0));
    h.bar_write(0, 0x10, 0x80000040, 4);
    h.attach(0, Function::new(UsbOutcome::Nak)).unwrap();
    assert!(!ack.level.load(Ordering::SeqCst));
}

/// The guest sees the documented write masks, HCR exemptions, and ownership bit.
#[test]
fn register_masks_preserve_hcr_exempt_fields_and_ownership_status() {
    let h = Ohci::new();
    h.bar_write(0, 0x18, 0x123456ff, 4);
    assert_eq!(h.bar_read(0, 0x18, 4), 0x12345600);
    h.bar_write(0, 0x20, 0x123456ff, 4);
    assert_eq!(h.bar_read(0, 0x20, 4), 0x123456f0);
    h.bar_write(0, 0x10, u32::MAX, 4);
    assert_eq!(h.bar_read(0, 0x10, 4), 0xc000007f);
    h.bar_write(0, 8, 8, 4);
    assert_eq!(h.bar_read(0, 0xc, 4), 1 << 30);
    h.bar_write(0, 0xc, 1 << 31, 4);
    assert_eq!(h.bar_read(0, 0xc, 4), 1 << 30);
    h.bar_write(0, 0xc, 1 << 30, 4);
    assert_eq!(h.bar_read(0, 0xc, 4), 0);
    h.bar_write(0, 0x48, u32::MAX, 4);
    assert_eq!(h.bar_read(0, 0x48, 4), 0xff001b02);
    h.bar_write(0, 4, 0x700, 4);
    h.bar_write(0, 8, 1, 4);
    assert_eq!(h.bar_read(0, 4, 4), 0x3c0);
    assert_eq!(h.bar_read(0, 0x48, 4), 0xff001b02);
    assert_eq!(h.bar_read(0, 0x10, 4), 0);
    h.bar_write(0, 0x3c, 0xffff, 4);
    assert_eq!(h.bar_read(0, 0x3c, 4), 0);
    h.bar_write(0, 0x18, 1, 1);
    assert_eq!(h.bar_read(0, 0x18, 4), 0);
}

/// The periodic phase controls how the guest control and bulk lists share service.
#[test]
fn periodic_phase_preserves_control_and_bulk_credit() {
    let (h, m, f) = setup(Pid::In, UsbOutcome::Nak, 0x4000, 0x4007, 0);
    m.words(0x2000, &[(8 << 16) | (1 << 7), 0x3020, 0x3000, 0]);
    m.words(0x2100, &[(8 << 16) | (2 << 7), 0x3120, 0x3100, 0]);
    m.words(0x3100, &[0xf0100000, 0x4100, 0x3120, 0x4107]);
    m.words(0x2200, &[(8 << 16) | (3 << 7), 0x3220, 0x3200, 0]);
    m.words(0x3200, &[0xf0100000, 0x4200, 0x3220, 0x4207]);
    for i in 0..32 {
        m.words(0x1000 + i * 4, &[0x2200]);
    }
    let mut s = h.state.lock().clone();
    s.active_fi = 5 * 256 - 1;
    s.r[16] = 3 * 256;
    s.r[1] = 0xb5;
    s.r[10] = 0x2100;
    s.r[2] |= 4;
    h.schedule(&mut s).unwrap();
    let order: Vec<_> = f.packets.lock().iter().map(|p| p.endpoint).collect();
    assert_eq!(order, [1, 1, 3, 2, 1]);
    assert_eq!(s.r[14], 0);
}
