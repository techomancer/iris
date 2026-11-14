//! Bedrock as the host of an XBridge: the crossbar's port 10, where a widget's
//! DMA enters the chip as inbound memory requests and its interrupt packets as
//! inbound writes to a processor interface.
//!
//! The Fuel's XBridge host and the Origin 3200's are the same port, so both
//! machines use this one.
//!
//! Ported from hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

use crate::dev::bedrock::Bedrock;
use crate::dev::widths::access_widths;
use crate::traits::{BusDevice, BusRead64, BUS_ERR, BUS_OK};
use crate::xtalk::XtalkHost;
use std::sync::{Arc, Weak};

/// Bedrock's crossbar port: widget 10, taking interrupt packets at any
/// address (the address picks the processor interface).
pub(crate) const BEDROCK_PORT: XtalkHost =
    XtalkHost { widget: crate::dev::xbridge::HOST_WIDGET, int_packet: 0, int_packet_mask: 0, vectors: 256 };

/// Bedrock as a widget sending through the XBridge reaches it: the
/// widget's DMA as inbound reads and writes, or blocks of doublewords
/// that take the admission and routing once, and its interrupt packets as inbound doubleword
/// writes to the address they name. Bedrock checks the widget's
/// permissions on each, for every master on the widget.
pub(crate) struct Inbound {
    pub(crate) bedrock: Weak<Bedrock>,
    pub(crate) widget: u8,
}

impl Inbound {
    /// The port widget `widget` of an XBridge on `bedrock` sends through.
    pub(crate) fn port(bedrock: &Arc<Bedrock>, widget: u8) -> Arc<Self> {
        Arc::new(Self { bedrock: Arc::downgrade(bedrock), widget })
    }

    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        let Some(bedrock) = self.bedrock.upgrade() else { return BusRead64::err() };
        match value {
            Some(v) => BusRead64 { status: bedrock.inbound_write(self.widget, None, a, size as u8, v), data: 0 },
            None => bedrock.inbound_read(self.widget, None, a, size as u8),
        }
    }
}
impl BusDevice for Inbound {
    access_widths!();
    // A block takes Bedrock's admission and routing once, so a run that
    // crosses the cable window goes where each word's node number sends it.
    fn read_block(&self, a: u64, buf: &mut [u64]) -> u32 {
        let Some(bedrock) = self.bedrock.upgrade() else { return BUS_ERR };
        bedrock.inbound_read_block(self.widget, None, a, buf)
    }
    fn write_block(&self, a: u64, buf: &[u64]) -> u32 {
        let Some(bedrock) = self.bedrock.upgrade() else { return BUS_ERR };
        bedrock.inbound_write_block(self.widget, None, a, buf)
    }
}

impl crate::irq::InterruptController for Inbound {
    fn set_line(&self, line: u32, _: bool) {
        crate::dlog!(crate::devlog::LogModule::Bedrock, "widget {} interrupt {line} with no packet address dropped", self.widget);
    }
    fn packet(&self, source: u8, address: u64, line: u32, level: bool) -> bool {
        let Some(bedrock) = self.bedrock.upgrade() else { return false };
        bedrock.inbound_write(source, None, address, 8, u64::from(line) | if level { 1 << 8 } else { 0 }) == BUS_OK
    }
}
