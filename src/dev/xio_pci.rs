//! The PCI shoehorn: an XIO board with a Bridge on a Crossbow port of its own,
//! owning a PCI bus with one slot. The Bridge is the onboard one
//! ([`super::bridge`]) unchanged, on another port.
//!
//! The shoehorn fits any of the Octane's four XIO slots, ports 9 to 12 (real
//! Octane2 captures put it at 10), and its card is configuration device 1.
//! Grades below are the datasheets' own: established (a published SGI
//! document or a real capture), inferred, unknown.
//!
//! * The slot is device 1: established on the Origin 2000, whose hardware
//!   graph shows the card at `.../xtalk_pci/pci/1`, and inferred on the
//!   Octane. Devices other than 1 hold no function; the Bridge decodes all
//!   eight, and a configuration cycle to a position nothing answers is a
//!   master time-out, as for an empty slot.
//! * Interrupts, inferred: INTA and INTC reach the Bridge pin numbered as the
//!   device (pin 1), INTB and INTD that number with bit two flipped (pin 5).
//!   That is the rule OpenBSD and Linux use for any Bridge and the onboard
//!   one has ([`crate::ip30::PCI0`]); no board trace is held.
//! * The Bridge is the Bridge ASIC (part 0xc002), inferred. Its revision is 4
//!   (Bridge D): established on the Origin 2000, where the PROM prints
//!   `ASIC BRIDGE Rev 4` for every such board, and unknown on the Octane, so
//!   the Origin 2000's stands.
//! * The one-wire record is part 030-1275-004, revision A, name `XTALK_PCI`,
//!   which real Octane and Origin 2000 `hinv` captures print. That the board
//!   keeps it on its Bridge's one-wire part is inferred; no dump is held. The
//!   barcode and serial are marked synthetic.
//! * The slot is 32 bit; the board's width is unknown.
//! * The Bridge's flash pins: IRIX reads the start of the flash aperture
//!   (offset 0xc00000) and its second select (0xe00000) on every Bridge it
//!   finds, and a bus error on either is a kernel fault. What a board with no
//!   flash fitted answers is not known, so both chip selects hold a blank
//!   part that reads all ones and keeps no writes, the approximation the
//!   onboard Bridge's second select already is.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::dev::bridge::{Bridge, RawFlash};
use crate::dev::w1;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use crate::pci::{DevFn, PciBus, PciFunction};
use crate::xtalk::Upstream;

/// Which board, as `[[xio_board]] kind` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    /// The single-slot XIO-to-PCI adapter.
    #[serde(rename = "pci-shoehorn")]
    Shoehorn,
}

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
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Kind::Shoehorn => "PCI shoehorn",
        }
    }

    /// The PCI device numbers of the board's slots, in the order a card with
    /// no slot named fills them.
    pub fn slots(self) -> &'static [u8] {
        match self {
            Kind::Shoehorn => &[1],
        }
    }
}

/// The one-wire record of a board: the part, revision and name its hinv
/// line shows, a synthetic barcode `EMUO` and the port, and a synthetic serial.
fn record(kind: Kind, widget: u8) -> w1::Bus {
    let (part, revision, name) = match kind {
        Kind::Shoehorn => ("030-1275-004", "A", "XTALK_PCI"),
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
    /// and interrupts the hosts `up` names. The machine maps
    /// [`Self::bridge`] at the port, tells the Bridge which links are alive
    /// ([`Bridge::set_link_alive`]) and resets the board with the machine.
    pub fn new(kind: Kind, widget: u8, up: Upstream) -> Self {
        let bridge = Bridge::new(up, &crate::ip30::PCI0);
        bridge.set_widget(widget);
        bridge.set_revision(match kind {
            Kind::Shoehorn => SHOEHORN_BRIDGE_REVISION,
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
