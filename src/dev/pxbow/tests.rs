// Cases follow tests/qtest/sn-builder-test.c in jamesbraid/qemu-sgi-origami:
// ip53-widget-zero-pxbow-identity, ip53-widget-zero-status-source and
// ip53-pxbow-route-fences-and-interrupt, and the widget-12 absence check of
// tests/qtest/sgi-graphics-pipes-test.c (ip53-v12). The qtest's machine
// properties, link-pulse timing and migration have no counterpart here.
use super::*;
use crate::dev::pic::{PicPeer, Refused, Pic};
use crate::traits::{BusRead32, BUS_OK};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

const STATUS: u64 = 0x0c;
const STATUS_CLEAR: u64 = 0x54;
const LINK_STATUS_ALIVE: u32 = 1 << 31;
const AUX_PRESENT: u32 = 1 << 5;

/// A clock the test advances, in nanoseconds.
fn clock() -> (Clock, Arc<AtomicU64>) {
    let now = Arc::new(AtomicU64::new(0));
    let c = now.clone();
    (Arc::new(move || c.load(Ordering::SeqCst)), now)
}

fn link(widget: u64, register: u64) -> u64 {
    0x100 + (widget - 8) * 0x40 + register + 4
}

/// A widget that answers one word and remembers a write.
struct Stub(Mutex<u32>);
impl BusDevice for Stub {
    fn read32(&self, addr: u64) -> BusRead32 {
        if addr == 0x40 { BusRead32::ok(*self.0.lock()) } else { BusRead32::err() }
    }
    fn write32(&self, addr: u64, value: u32) -> u32 {
        if addr != 0x40 {
            return BUS_ERR;
        }
        *self.0.lock() = value;
        BUS_OK
    }
}

#[derive(Default)]
struct Peer(Mutex<Vec<(u8, u64, u64)>>);
impl PicPeer for Peer {
    fn interrupt(&self, source: u8, address: u64, payload: u64) -> Result<(), Refused> {
        self.0.lock().push((source, address, payload));
        Ok(())
    }
    fn dma_read(&self, _: u8, _: u8, _: u64, _: &mut [u8]) -> Result<(), Refused> {
        Err(Refused)
    }
    fn dma_write(&self, _: u8, _: u8, _: u64, _: &[u8]) -> Result<(), Refused> {
        Err(Refused)
    }
}

/// Widget zero names itself PXBOW, not Crossbow and not the PIC, and shows
/// the host's and the PIC's links up.
#[test]
fn widget_zero_identifies_as_pxbow_with_the_host_and_pic_links_up() {
    let p = Pxbow::new(10, 15, clock().0).unwrap();
    let w = p.widget_zero();
    let id = w.read32(4).data;
    assert_eq!(id, 0x3d10_0001);
    assert_eq!((id >> 28) & 0xf, 3);
    assert_eq!((id >> 12) & 0xffff, 0xd100);
    assert_eq!((id >> 1) & 0x7ff, 0);
    assert_eq!(id & 1, 1);
    for widget in [10, 15] {
        assert_ne!(w.read32(link(widget, 0x10)).data & LINK_STATUS_ALIVE, 0, "widget {widget} alive");
        assert_ne!(w.read32(link(widget, 0x38)).data & AUX_PRESENT, 0);
    }
    for widget in [8, 9, 11, 12, 13, 14] {
        assert_eq!(w.read32(link(widget, 0x10)).data & LINK_STATUS_ALIVE, 0, "widget {widget} absent");
        assert_eq!(w.read32(link(widget, 0x38)).data & AUX_PRESENT, 0);
    }
    assert_eq!(p.endpoint_widgets(), 0);
}

#[test]
fn status_reads_carry_the_hosts_widget_as_their_source() {
    let p = Pxbow::new(10, 15, clock().0).unwrap();
    let w = p.widget_zero();
    // Seed the status with an access error first.
    let _ = w.read32(0x1004);
    let status = w.read32(STATUS).data;
    assert_eq!((status >> 6) & 0xf, 10);
    assert_ne!(status & (1 << 5), 0, "the seeded access error shows");
    assert_eq!((w.read32(STATUS_CLEAR).data >> 6) & 0xf, 10);
    assert_eq!(w.read32(STATUS).data & (1 << 5), 0, "the alias acknowledged it");
    let other = Pxbow::new(11, 14, clock().0).unwrap();
    assert_eq!((other.widget_zero().read32(STATUS).data >> 6) & 0xf, 11);
}

/// PROM 6.210's crossbar check on the Tezro writes 0xdeadbeef to the
/// read-only lower error address and 0 to the upper one. The store
/// completes, the registers keep their value, and widget 0's status reports
/// the access error (NEC U10278EJ4V0UM p.104).
#[test]
fn the_proms_error_address_writes_complete_and_are_reported() {
    let p = Pxbow::new(10, 15, clock().0).unwrap();
    let w = p.widget_zero();
    assert_eq!(w.write32(0x1c, 0xdead_beef), BUS_OK);
    assert_eq!(w.write32(0x14, 0), BUS_OK);
    assert_eq!(w.read32(0x1c).data, 0);
    assert_ne!(w.read32(STATUS).data & (1 << 5), 0, "the access error shows");
    assert!(p.xbow().unsupported().contains(&(0x1c, 4, true)), "the write stays observable");
    assert_ne!(w.write32(0x04, 0), BUS_OK, "the identification register still refuses");
}

/// Revision 5 decodes the fences and the lock; the lock takes a write only
/// while it reads zero and any write to its clear alias releases it.
#[test]
fn fences_and_the_lock_are_stored_and_read_back() {
    let p = Pxbow::new(10, 15, clock().0).unwrap();
    let w = p.widget_zero();
    for (off, v) in [(0x78u64, 0xff_u32), (0x80, 0xff), (0xb8, 0xff), (0xc0, 0), (0xc8, 0)] {
        assert_eq!(w.read32(off + 4).data, v, "reset value at {off:#x}");
    }
    for (off, v) in [(0x78u64, 0x5a), (0x90, 0xa5), (0xb8, 0x3c)] {
        assert_eq!(w.write32(off + 4, v), BUS_OK);
        assert_eq!(w.read32(off + 4).data, v);
    }
    assert_eq!(w.read32(0x78 + 4).data, 0x5a);
    assert_eq!(w.write32(0xc0 + 4, 0x11), BUS_OK);
    assert_eq!(w.write32(0xc0 + 4, 0x22), BUS_OK);
    assert_eq!(w.read32(0xc0 + 4).data, 0x11, "a held lock ignores a second write");
    assert_eq!(w.read32(0xc8 + 4).data, 0x11, "both offsets read the one value");
    assert_eq!(w.write32(0xc8 + 4, 0), BUS_OK);
    assert_eq!(w.read32(0xc0 + 4).data, 0);
    assert_eq!(w.write32(0xc0 + 4, 0x22), BUS_OK);
    assert_eq!(w.read32(0xc0 + 4).data, 0x22);
    crate::traits::Resettable::power_on(&*p.xbow().clone());
    assert_eq!(w.read32(0x78 + 4).data, 0xff, "reset restores the fences");
    assert_eq!(w.read32(0xc0 + 4).data, 0, "and releases the lock");
    // The upper word of a slot and an eight-byte access are still refused.
    assert!(!w.read32(0x78).is_ok());
    assert!(!w.read64(0x78).is_ok());
}

/// Widget zero's own error interrupt is a packet to the host, with the
/// vector and address its destination registers hold.
#[test]
fn a_widget_zero_error_sends_an_interrupt_packet_to_the_host() {
    let pic = Pic::new(15);
    let peer = Arc::new(Peer::default());
    pic.connect_peer(peer.clone()).unwrap();
    let p = pic.enable_pxbow(10, clock().0).unwrap();
    let w = p.widget_zero();
    let address = 0x1_0180_0090u64;
    assert_eq!(w.write32(0x34, (0x34 << 24) | (10 << 16) | ((address >> 32) as u32 & 0xffff)), BUS_OK);
    assert_eq!(w.write32(0x3c, address as u32), BUS_OK);
    assert_eq!(w.write32(0x24, 1 << 5), BUS_OK, "enable the access-error condition");
    assert!(peer.0.lock().is_empty());
    let _ = w.read32(0x1004);
    assert_ne!(w.read32(STATUS).data & (1 << 23), 0, "the request bit is up");
    assert_eq!(peer.0.lock().as_slice(), [(0, address, 0x134)]);
    let _ = w.read32(STATUS_CLEAR);
    assert_eq!(peer.0.lock().last().copied(), Some((0, address, 0x34)), "acknowledging lowers it");
    // A target that is not the host goes nowhere.
    assert_eq!(w.write32(0x34, (0x34 << 24) | (11 << 16)), BUS_OK);
    let _ = w.read32(0x1004);
    assert_eq!(peer.0.lock().len(), 2);
}

/// The V12's widget is absent until the machine fits it, then forwards to
/// the endpoint with the widget-relative address unchanged.
#[test]
fn a_widget_is_unreachable_until_an_endpoint_is_attached() {
    let p = Pxbow::new(10, 15, clock().0).unwrap();
    let w12 = p.route(12);
    assert!(!w12.read32(0x40).is_ok());
    assert_eq!(w12.write32(0x40, 1), BUS_ERR);
    let stub = Arc::new(Stub(Mutex::new(0)));
    p.attach_endpoint(12, stub.clone()).unwrap();
    assert_eq!(p.endpoint_widgets(), 1 << 12);
    let wz = p.widget_zero();
    assert_ne!(wz.read32(link(12, 0x10)).data & LINK_STATUS_ALIVE, 0, "its link comes up");
    assert_ne!(wz.read32(link(12, 0x38)).data & AUX_PRESENT, 0);
    assert_eq!(w12.write32(0x40, 0xcafe), BUS_OK);
    assert_eq!(*stub.0.lock(), 0xcafe);
    assert_eq!(w12.read32(0x40).data, 0xcafe);
    assert!(!w12.read32(0x44).is_ok(), "the endpoint's own refusal passes through");
    assert!(!p.route(11).read32(0x40).is_ok(), "other widgets stay absent");
}

#[test]
fn attachment_refuses_the_host_a_taken_widget_and_a_non_link() {
    let p = Pxbow::new(10, 15, clock().0).unwrap();
    let stub: Arc<dyn BusDevice> = Arc::new(Stub(Mutex::new(0)));
    assert_eq!(p.attach_endpoint(10, stub.clone()), Err(AttachError::Host(10)));
    assert_eq!(p.attach_endpoint(3, stub.clone()), Err(AttachError::NotALink(3)));
    assert_eq!(p.attach_endpoint(12, stub.clone()), Ok(()));
    assert_eq!(p.attach_endpoint(12, stub.clone()), Err(AttachError::Occupied(12)));
    assert!(Pxbow::new(10, 10, clock().0).is_err());
    assert!(Pxbow::new(10, 2, clock().0).is_err());
}

/// The PIC that owns IO9 supplies widget 0; the other PIC does not, and
/// the PIC's own widget is reached through the switch.
#[test]
fn only_the_pic_that_enables_it_has_a_pxbow_and_it_forwards_to_the_pic() {
    let with = Pic::new(15);
    let without = Pic::new(11);
    assert!(with.pxbow().is_none());
    let p = with.enable_pxbow(10, clock().0).unwrap();
    assert!(without.pxbow().is_none());
    assert_eq!(p.pic_widget(), 15);
    assert_eq!(p.endpoint_widgets(), 1 << 15);
    assert!(Arc::ptr_eq(with.pxbow().unwrap(), &p));
    assert!(with.enable_pxbow(10, clock().0).is_err(), "once");
    let via_switch = p.route(15);
    let id = via_switch.read64(0).data;
    assert_eq!((id >> 28) & 0xf, 3, "the PIC's head 0 answers through widget 15");
    assert_eq!(via_switch.read64(crate::dev::pic::HEAD_OFFSET).data >> 12 & 0xffff, 0xd112);
    assert!(Pic::new(15).enable_pxbow(15, clock().0).is_err(), "host and PIC differ");
}

/// A link reset on the host's link or the PIC's takes the PIC out of reach
/// for the pulse and brings it back, as the qtest's 20 ns clock step does.
#[test]
fn a_link_reset_makes_the_pic_unreachable_for_the_pulse() {
    let (c, now) = clock();
    let pic = Pic::new(15);
    let p = pic.enable_pxbow(10, c).unwrap();
    let via = p.route(15);
    let wz = p.widget_zero();
    assert!(via.read64(0).is_ok());
    for (widget, reset) in [(10u64, 0x1b4u64), (15, 0x2f4)] {
        now.fetch_add(100, Ordering::SeqCst);
        assert_eq!(wz.write32(reset, 0), BUS_OK, "widget {widget}'s link reset");
        assert!(!via.read64(0).is_ok(), "down during the pulse");
        assert_eq!(via.write64(0x20, 0), BUS_ERR);
        now.fetch_add(20, Ordering::SeqCst);
        assert!(via.read64(0).is_ok(), "back after 20 ns");
        assert_eq!(via.read64(0x20).data, 0x0f00_320f, "the PIC kept its registers");
    }
}
