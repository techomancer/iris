//! PXBOW, the crossbar-like widget 0 that sits behind the PIC on Tezro and
//! the Origin 350, and the forwarding of processor accesses through it.
//!
//! The model reuses the Crossbow ([`Xbow`]) at revision 5 behaviour with
//! PXBOW's identity word `0x3d100001` (revision 3, part `0xd100`,
//! manufacturer 0), and a source field in its
//! status reads. That is a software choice, not a claim that the part is a
//! Crossbow 2.0 or that the processor's route to widget 0 passes a real
//! switch: its physical route, fitted revision and register set beyond the
//! reused ones are unknown and stay unsupported and observable
//! ([`Xbow::unsupported`]).
//!
//! The chip knows two links when it is built: the host's widget (Bedrock
//! IO, 10) and the PIC's own. A further widget, such as the optional V12 at
//! widget 12 on the two- and four-processor Tezro, becomes reachable when
//! the machine attaches it with [`Pxbow::attach_endpoint`]; until then its
//! link is absent and an access to it is a bus error. Each processor access
//! to a widget through [`Pxbow::route`] needs both the host's link and the
//! target's to be alive, so a link reset takes the PIC out of reach for the
//! 20 ns of its pulse.
//!
//! Ported from hw/pci-host/sgi-pic.c and hw/misc/sgi-xbow.c in
//! jamesbraid/qemu-sgi-origami.

use super::heart::Clock;
use super::xbow::Xbow;
use crate::irq::InterruptController;
use crate::xtalk::XtalkHost;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_ERR};
use parking_lot::RwLock;
use std::sync::Arc;

/// The Crossbow revision the PIC's PXBOW selects for behaviour.
pub const REVISION: u32 = 5;
/// The revision its identification word reports. Established from real
/// Tezro, Origin 350 and Onyx 4 PROM inventories (revision 3).
pub const REPORTED_REVISION: u32 = 3;
// Part and manufacturer are INFERRED; a 2009 OpenBSD dmesg names the part
// Bedrock, a conflict that stays open.
pub const PART: u32 = 0xd100;
pub const MANUFACTURER: u32 = 0;

/// Why a widget could not be attached.
#[derive(Debug, PartialEq, Eq)]
pub enum AttachError {
    /// Links carry widgets 8 to 15 only.
    NotALink(u8),
    /// The host's own widget has no endpoint behind the switch.
    Host(u8),
    /// The widget already has an endpoint.
    Occupied(u8),
}

pub struct Pxbow {
    xbow: Arc<Xbow>,
    host: u8,
    pic: u8,
    endpoints: RwLock<[Option<Arc<dyn BusDevice>>; 16]>,
}

impl Pxbow {
    /// A PXBOW reached from `host` (the widget Bedrock IO answers as) with
    /// its PIC at `pic`. Both links are present and alive. `clock` is the
    /// machine's virtual time in nanoseconds, which times link resets.
    pub fn new(host: u8, pic: u8, clock: Clock) -> Result<Arc<Self>, AttachError> {
        for widget in [host, pic] {
            if !(8..16).contains(&widget) {
                return Err(AttachError::NotALink(widget));
            }
        }
        if host == pic {
            return Err(AttachError::Occupied(pic));
        }
        let xbow = Xbow::new(REVISION, (1 << host) | (1 << pic))
            .with_identity(PART, MANUFACTURER)
            .with_reported_revision(REPORTED_REVISION)
            .with_status_source(host)
            // PROM 6.210's crossbar check writes the read-only lower error
            // address on purpose.
            .with_posted_error_address_writes()
            .with_clock(clock);
        Ok(Arc::new(Self {
            xbow: Arc::new(xbow),
            host,
            pic,
            endpoints: RwLock::new(std::array::from_fn(|_| None)),
        }))
    }

    pub fn host_widget(&self) -> u8 {
        self.host
    }

    pub fn pic_widget(&self) -> u8 {
        self.pic
    }

    /// The register block the processor reaches as widget 0.
    pub fn xbow(&self) -> &Arc<Xbow> {
        &self.xbow
    }

    /// Widget 0 as a bus device, for Bedrock IO's widget 0.
    pub fn widget_zero(&self) -> Arc<dyn BusDevice> {
        self.xbow.clone()
    }

    /// Where widget 0's own interrupt goes: packets to the host widget,
    /// whose address names the destination. Once, before the machine runs.
    pub fn connect_host(&self, host: Arc<dyn InterruptController>) {
        let xtalk = XtalkHost { widget: self.host, int_packet: 0, int_packet_mask: 0, vectors: 256 };
        self.xbow.connect_host(xtalk, host);
    }

    /// Put `device` behind the switch as `widget` and bring its link up.
    /// The PIC's own endpoint is set this way too.
    pub fn attach_endpoint(&self, widget: u8, device: Arc<dyn BusDevice>) -> Result<(), AttachError> {
        if !(8..16).contains(&widget) {
            return Err(AttachError::NotALink(widget));
        }
        if widget == self.host {
            return Err(AttachError::Host(widget));
        }
        let mut endpoints = self.endpoints.write();
        if endpoints[widget as usize].is_some() {
            return Err(AttachError::Occupied(widget));
        }
        endpoints[widget as usize] = Some(device);
        drop(endpoints);
        self.xbow.attach(widget);
        Ok(())
    }

    /// Widgets with an endpoint, as a mask with bit n for widget n.
    pub fn endpoint_widgets(&self) -> u16 {
        self.endpoints.read().iter().enumerate().fold(0, |m, (w, e)| m | (u16::from(e.is_some()) << w))
    }

    /// `widget` as the processor reaches it through the switch: forwarded
    /// to the endpoint while the host's link and the widget's are alive, a
    /// bus error otherwise.
    pub fn route(self: &Arc<Self>, widget: u8) -> Arc<dyn BusDevice> {
        assert!((8..16).contains(&widget));
        Arc::new(Routed { pxbow: self.clone(), widget })
    }
}

struct Routed {
    pxbow: Arc<Pxbow>,
    widget: u8,
}

impl Routed {
    /// The endpoint, if the route is open.
    fn open(&self) -> Option<Arc<dyn BusDevice>> {
        let p = &self.pxbow;
        if !p.xbow.carries_timed(p.host) || !p.xbow.carries_timed(self.widget) {
            return None;
        }
        p.endpoints.read()[self.widget as usize].clone()
    }
}

macro_rules! forward {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, addr: u64) -> $out {
            self.open().map_or_else($out::err, |d| d.$r(addr))
        }
        fn $w(&self, addr: u64, value: $t) -> u32 {
            self.open().map_or(BUS_ERR, |d| d.$w(addr, value))
        }
    };
}

impl BusDevice for Routed {
    forward!(read8, write8, u8, BusRead8);
    forward!(read16, write16, u16, BusRead16);
    forward!(read32, write32, u32, BusRead32);
    forward!(read64, write64, u64, BusRead64);
}

#[cfg(test)]
mod tests;
