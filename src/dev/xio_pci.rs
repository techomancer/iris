//! PCI option boards for an XIO slot: a Bridge on a Crossbow port of its own,
//! owning a PCI bus with a few slots. Two products share the model, which is
//! the onboard Bridge ([`super::bridge`]) unchanged on another port.
//!
//! | board                 | Octane port | slots (PCI device numbers) |
//! |-----------------------|-------------|----------------------------|
//! | shoehorn              | 9 to 12     | 1                          |
//! | Octane PCI module     | 13          | 1, 2 and 3                 |
//!
//! Grades below are the datasheets' own: established (a published SGI
//! document or a real capture), inferred, unknown.
//!
//! The PCI module is on its own Crossbow port, 13, beside the four XIO slots
//! at 9 to 12 (established), and holds up to three boards, PCI ID 1 to 3
//! (OCTANE PCI Module Installation Guide 007-3547-001, printed pages 3 and
//! 13, and its `ioconfig.conf` table, page 65; established). That slot N is
//! configuration device N is inferred: the guide and real hardware graphs
//! name the software slot, which the guide says matches the physical one,
//! and `/hw/node/xtalk/13/pci/N` for N of 1, 2 and 3, but a software slot
//! name is not a device number. Devices 0 and 4 to 7 are not slots; whether
//! the module's Bridge answers there is unknown, and here it does not, since
//! nothing is fitted. A shoehorn has one slot, device 1: established on the
//! Origin 2000, inferred on the Octane.
//!
//! Inferred or unknown, and what is picked here:
//!
//! * Interrupts, inferred for both boards: a slot's INTA and INTC reach the
//!   Bridge pin numbered as its device, INTB and INTD that number with bit
//!   two flipped. That is the default Bridge board wiring OpenBSD and Linux
//!   use for any Bridge and the onboard one has ([`crate::dev::bridge::PIN_WIRING`]); no
//!   board trace is held.
//! * The widget is a Bridge-class one, established; that it reports the Bridge
//!   (part 0xc002) and not the XBridge (0xd002) is inferred. Its revision is
//!   unknown for the module, so it carries the machine's own Bridge revision
//!   as a placeholder. A shoehorn's is revision 4 (Bridge D), established on
//!   the Origin 2000 and unknown on the Octane.
//! * One-wire records. The part, revision and name are established by `hinv`
//!   captures: the shoehorn is 030-1275-004 rev A `XTALK_PCI`, the module
//!   030-0952-005 rev E `XTALKPCI` (the most common of the revisions seen).
//!   That each board keeps its record on its Bridge's one-wire part is
//!   inferred from the adapter and the Origin 2000 carriers; no dump of
//!   either is held. The barcodes are marked synthetic.
//! * Which slots are 32 or 64 bit is unknown; every slot is 32 bit here.
//! * The Bridge's flash pins: IRIX reads the start of the flash aperture
//!   (offset 0xc00000) and its second select (0xe00000) on every Bridge it
//!   finds, and a bus error on either is a kernel fault. What a board with no
//!   flash fitted answers is not known, so both chip selects hold a blank
//!   part that reads all ones and keeps no writes, the approximation the
//!   onboard Bridge's second select already is.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::dev::bridge::{Bridge, RawFlash, PIN_WIRING};
use crate::dev::w1;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use crate::pci::{DevFn, PciBus, PciFunction};
use crate::xtalk::Upstream;

/// Which product, as `[[xio_board]] kind` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    /// The single-slot XIO-to-PCI adapter.
    #[serde(rename = "pci-shoehorn")]
    Shoehorn,
    /// The Octane's three-slot PCI module.
    #[serde(rename = "pci-cage")]
    Cage,
}

/// The Octane PCI module's XIO number.
pub const CAGE_WIDGET: u8 = 13;
/// The blank part on each of a board's flash selects: one select's worth.
const BLANK_FLASH_SIZE: u32 = 2 << 20;
/// A shoehorn's Bridge revision: Bridge D, as the Origin 2000's PROM prints it
/// for the same adapter.
const SHOEHORN_BRIDGE_REVISION: u32 = 4;

impl Kind {
    /// The name `[[xio_board]] kind` uses.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Shoehorn => "pci-shoehorn",
            Kind::Cage => "pci-cage",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Kind::Shoehorn => "PCI shoehorn",
            Kind::Cage => "PCI module",
        }
    }

    /// The PCI device numbers of the board's slots, in the order a card with
    /// no slot named fills them.
    pub fn slots(self) -> &'static [u8] {
        match self {
            Kind::Shoehorn => &[1],
            Kind::Cage => &[1, 2, 3],
        }
    }
}

/// The one-wire record of a board: the part, revision and name its hinv
/// line shows, a synthetic barcode `EMUO` and the port, and a synthetic serial.
fn record(kind: Kind, widget: u8) -> w1::Bus {
    let (part, revision, name) = match kind {
        Kind::Shoehorn => ("030-1275-004", "A", "XTALK_PCI"),
        Kind::Cage => ("030-0952-005", "E", "XTALKPCI"),
    };
    let barcode = format!("EMUO{widget:02}");
    let serial = 0x5145_4d55_4f00 | widget as u64;
    let mut bus = w1::Bus::new();
    bus.add(Box::new(board_record_part(&BoardRecord { barcode: &barcode, part, revision, name }, serial)));
    bus
}

/// One board: its Bridge, which the machine maps at the board's port.
pub struct XioPciBoard {
    kind: Kind,
    widget: u8,
    bridge: Arc<Bridge>,
}

impl XioPciBoard {
    /// A board of `kind` on Crossbow port `widget`. Its Bridge masters memory
    /// and interrupts the hosts `up` names; `revision` is the machine's
    /// Bridge revision, which a PCI module takes as a placeholder. The machine
    /// maps [`Self::bridge`] at the port, tells the Bridge which links are
    /// alive ([`Bridge::set_link_alive`]) and resets the board with the
    /// machine.
    pub fn new(kind: Kind, widget: u8, up: Upstream, revision: u32) -> Self {
        let bridge = Bridge::new(up, &PIN_WIRING);
        bridge.set_widget(widget);
        bridge.set_revision(match kind {
            Kind::Shoehorn => SHOEHORN_BRIDGE_REVISION,
            Kind::Cage => revision,
        });
        bridge.set_microlan(record(kind, widget));
        bridge.attach_flash(Arc::new(RawFlash::new(&[], BLANK_FLASH_SIZE as usize)), BLANK_FLASH_SIZE);
        bridge.attach_flash_select1(Arc::new(RawFlash::new(&[], BLANK_FLASH_SIZE as usize)), BLANK_FLASH_SIZE);
        Self { kind, widget, bridge }
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// The Crossbow port the board is on.
    pub fn widget(&self) -> u8 {
        self.widget
    }

    pub fn bridge(&self) -> &Arc<Bridge> {
        &self.bridge
    }

    pub fn bus(&self) -> &PciBus {
        self.bridge.bus()
    }

    /// `device` as the board's slot, free to take a function.
    pub fn check_slot(&self, device: u8) -> Result<(), String> {
        if !self.kind.slots().contains(&device) {
            return Err(format!("the {} has no slot {device} (slots {:?})", self.kind.label(), self.kind.slots()));
        }
        if self.bus().occupied(device) {
            return Err(format!("the {}'s slot {device} is taken", self.kind.label()));
        }
        Ok(())
    }

    /// The first slot with nothing in it.
    pub fn free_slot(&self) -> Option<u8> {
        self.kind.slots().iter().copied().find(|&d| !self.bus().occupied(d))
    }

    /// Put `f` (function 0) in slot `device`.
    pub fn attach(&self, device: u8, f: Arc<dyn PciFunction>) -> Result<(), String> {
        self.check_slot(device)?;
        self.bus().attach(DevFn { device, function: 0 }, f);
        Ok(())
    }
}

#[cfg(test)]
#[path = "xio_pci_tests.rs"]
mod tests;
