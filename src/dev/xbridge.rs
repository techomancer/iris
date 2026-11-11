//! The Fuel's XBridge: two PCI heads at crossbar ports 14 and 15, and the
//! crossbar that joins them, and any XIO board, to the host at port 10.
//!
//! Each head is a [`Bridge`] built as [`Part::XBridge`], and the crossbar a
//! Crossbow built as [`Xbow::xbridge`]; `bridge.rs` and `xbow.rs` say where
//! they differ from the Octane's chips. This module wires them together:
//! every head sends its DMA and interrupt packets to the host the board
//! names, and a link reset of a head's port resets the head and runs RST#
//! on its bus before the board resets what it attached.
//!
//! # What is approximated here
//!
//! The crossbar's reset fences are kept but gate nothing: no warm reset
//! travels through it. Link training is not modelled; a link reset brings a
//! fitted link back alive at once, as on the Crossbow.
//!
//! Ported from hw/pci-host/sgi-xbridge.c and hw/misc/sgi-xbridge-xbow.c
//! in jamesbraid/qemu-sgi-origami, revision
//! fe128635b6dfe2511f1bd451684cb30007686df1.

use crate::dev::bridge::{Bridge, Part};
use crate::dev::xbow::Xbow;
use crate::irq::InterruptController;
use crate::pci::{IntxRouting, PciBus, PciDma, PciHost};
use crate::traits::{BusDevice, Resettable};
use crate::xtalk::{Upstream, XtalkHost};
use parking_lot::RwLock;
use std::sync::{Arc, Weak};

pub const HOST_WIDGET: u8 = 10;
pub const HEAD_E: u8 = 14;
pub const HEAD_F: u8 = 15;

/// The head's PCI interrupt routing. Device `d`'s INTA reaches interrupt pin `d`; the
/// pins are shared with the board's own IRQs (the IOC3's SuperIO interrupt
/// is `BoardIrq(0)`). The Fuel uses devices 1, 4 and 5; the others follow the
/// same rule, and INTB to INTD are not wired.
pub static PCI_ROUTING: IntxRouting = IntxRouting {
    devices: 8,
    irqs: 8,
    intx: &[
        (0, [Some(0), None, None, None]),
        (1, [Some(1), None, None, None]),
        (2, [Some(2), None, None, None]),
        (3, [Some(3), None, None, None]),
        (4, [Some(4), None, None, None]),
        (5, [Some(5), None, None, None]),
        (6, [Some(6), None, None, None]),
        (7, [Some(7), None, None, None]),
    ],
    board_irqs: &[0, 1, 2, 3, 4, 5, 6, 7],
};

/// What the board resets when a port's link reset reaches it.
type ResetOutput = Arc<dyn Fn(u8) + Send + Sync>;

/// One chip: both heads and the crossbar.
pub struct XBridge {
    heads: [Arc<Bridge>; 2],
    /// Each head's XIO space, as the host reaches it at the head's port.
    windows: [Arc<dyn BusDevice>; 2],
    xbow: Arc<Xbow>,
    reset_output: Arc<RwLock<Option<ResetOutput>>>,
}

impl XBridge {
    /// An XBridge whose crossbar has `host` at its port. `port(widget)` is
    /// the host as a widget sending through the crossbar reaches it: the
    /// memory its DMA lands in and the input its interrupt packets set.
    /// The crossbar's own packets come from widget 0.
    pub fn new(host: XtalkHost, port: &dyn Fn(u8) -> (Arc<dyn BusDevice>, Arc<dyn InterruptController>)) -> Arc<Self> {
        let xbow = Arc::new(Xbow::xbridge(1 << host.widget | 1 << HEAD_E | 1 << HEAD_F));
        xbow.connect_host(host, port(0).1);
        let heads = [HEAD_E, HEAD_F].map(|widget| {
            let (mem, irq) = port(widget);
            let head = Bridge::with_part(Part::XBridge, Upstream::single(host, mem, irq), &PCI_ROUTING);
            head.set_revision(2);
            head.set_widget(widget);
            // Head E's bus runs at 66 MHz (speed code 1), head F's at 33.
            head.set_control_strap(u32::from(widget == HEAD_E) << 4);
            let links = xbow.clone();
            head.set_link_alive(Box::new(move |w| links.carries(w)));
            head
        });
        let reset_output: Arc<RwLock<Option<ResetOutput>>> = Default::default();
        let resets: [Weak<Bridge>; 2] = heads.each_ref().map(Arc::downgrade);
        let output = reset_output.clone();
        xbow.on_link_reset(Box::new(move |port| {
            if let Some(head) = port.checked_sub(HEAD_E).and_then(|i| resets.get(usize::from(i))).and_then(Weak::upgrade) {
                head.reset();
                head.reset_pci();
            }
            // Cloned out, so the board may replace it or reach the chip.
            let out = output.read().clone();
            if let Some(out) = out {
                out(port);
            }
        }));
        let windows = heads.each_ref().map(|head| head.head_window());
        Arc::new(Self { heads, windows, xbow, reset_output })
    }

    pub fn register_locks(&self) {
        for head in &self.heads {
            head.register_locks();
        }
        self.xbow.register_locks();
    }

    fn index(widget: u8) -> usize {
        assert!((HEAD_E..=HEAD_F).contains(&widget), "widget {widget} is not an XBridge head");
        usize::from(widget - HEAD_E)
    }

    /// What the host connects at crossbar port `widget`: the crossbar's
    /// registers at 0 and at the host's own port, a head's XIO space at its
    /// port. A board's port is the one [`XBridge::attach_widget`] returns.
    pub fn widget(&self, widget: u8) -> Option<Arc<dyn BusDevice>> {
        match widget {
            0 | HOST_WIDGET => Some(self.xbow.clone()),
            HEAD_E | HEAD_F => Some(self.windows[Self::index(widget)].clone()),
            _ => None,
        }
    }

    /// The crossbar's widget 0 registers.
    pub fn crossbar(&self) -> Arc<dyn BusDevice> {
        self.xbow.clone()
    }

    /// The PCI bus behind head `widget`: functions are fitted with
    /// [`PciBus::attach`] and the SuperIO interrupt is `bus.board_irq(BoardIrq(0))`.
    pub fn bus(&self, widget: u8) -> &PciBus {
        self.heads[Self::index(widget)].bus()
    }

    /// The DMA head `widget` gives the function in `slot`.
    pub fn dma(&self, widget: u8, slot: u8) -> Arc<dyn PciDma> {
        assert!(slot < 8);
        self.heads[Self::index(widget)].dma_for(slot)
    }

    /// Fit a board at crossbar port `widget`, its link up. What the host
    /// connects at that port is returned: the board while its link is up.
    pub fn attach_widget(&self, widget: u8, board: Arc<dyn BusDevice>) -> Arc<dyn BusDevice> {
        assert!((8..HEAD_E).contains(&widget) && widget != HOST_WIDGET);
        self.xbow.attach(widget);
        self.xbow.port(widget, board)
    }

    /// Told of each port's link reset, after a head's own reset and RST# on
    /// its bus and with no chip lock held: the board resets the engines it
    /// attached behind that port.
    pub fn set_reset_output(&self, output: ResetOutput) {
        *self.reset_output.write() = Some(output);
    }

    pub fn reset(&self) {
        for head in &self.heads {
            head.reset();
            head.reset_pci();
        }
        self.xbow.power_on();
    }
}

impl Resettable for XBridge {
    fn power_on(&self) {
        self.reset();
    }
}

#[cfg(test)]
mod tests;
