//! The add-in cards a PCI slot can be fitted with.
//!
//! A card is a PCI function plus whatever the machine has to do for it
//! beyond giving it a slot: the SCSI buses it brings, a thread to start
//! and stop, state to save. The list of cards is a static table; each
//! entry builds a fresh card on every use, so a machine can fit several
//! of the same kind.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::dev::isp12160::Isp12160;
use crate::pci::PciBusDevice;
use crate::scsi::ScsiBus;
use crate::traits::{Device, Resettable, Saveable};

/// The `key = value` pairs of a `[[pci]]` entry beyond `slot`, `bus` and
/// `card`.
pub type Options = BTreeMap<String, toml::Value>;

/// A card fitted in a slot.
pub trait PciCard: Send + Sync {
    fn name(&self) -> &'static str;

    /// The card's PCI function, which the bus attaches.
    fn function(&self) -> Arc<dyn PciBusDevice>;

    /// The SCSI buses the card brings, in channel order. The machine numbers
    /// them as controllers after its own.
    fn scsi_buses(&self) -> Vec<&ScsiBus> {
        Vec::new()
    }

    /// Light the status bar's activity bits in `heartbeat`.
    fn set_heartbeat(&self, _heartbeat: Arc<AtomicU64>) {}

    /// Name the card's locks in the lock registry.
    fn register_locks(&self) {}

    /// Start the card's worker, if it has one. Stop finishes the command it
    /// is running, so a snapshot finds none half done.
    fn start(&self) {}
    fn stop(&self) {}

    fn power_on(&self) {}
    fn save_state(&self) -> toml::Value;
    fn load_state(&self, v: &toml::Value) -> Result<(), String>;
}

/// Build a card from its options, given the controller number its first SCSI
/// bus will have.
pub type BuildCard = fn(&Options, u8) -> Result<Box<dyn PciCard>, String>;

/// One kind of card.
pub struct CardDef {
    /// The name a `[[pci]]` entry gives as `card`.
    pub name: &'static str,
    pub summary: &'static str,
    /// The option keys the card takes.
    pub options: &'static [&'static str],
    /// How many SCSI controllers (buses) it adds to the machine's.
    pub scsi_controllers: u8,
    pub build: BuildCard,
}

pub static CARDS: &[CardDef] = &[CardDef {
    name: "isp12160",
    summary: "QLogic ISP12160, two Ultra160 SCSI buses on one PCI function",
    options: &[],
    scsi_controllers: 2,
    build: |_, _| Ok(Box::new(IspCard(Arc::new(Isp12160::default())))),
}];

pub fn card_def(name: &str) -> Option<&'static CardDef> {
    CARDS.iter().find(|c| c.name == name)
}

/// The card names, for an error message.
pub fn card_names() -> String {
    CARDS.iter().map(|c| c.name).collect::<Vec<_>>().join(", ")
}

struct IspCard(Arc<Isp12160>);

impl PciCard for IspCard {
    fn name(&self) -> &'static str {
        "isp12160"
    }

    fn function(&self) -> Arc<dyn PciBusDevice> {
        self.0.clone()
    }

    fn scsi_buses(&self) -> Vec<&ScsiBus> {
        (0..2).filter_map(|channel| self.0.bus_for_channel(channel)).collect()
    }

    fn set_heartbeat(&self, heartbeat: Arc<AtomicU64>) {
        self.0.set_heartbeat(heartbeat);
    }

    fn register_locks(&self) {
        self.0.register_locks();
    }

    fn start(&self) {
        Device::start(&*self.0);
    }

    fn stop(&self) {
        Device::stop(&*self.0);
    }

    fn power_on(&self) {
        Resettable::power_on(&*self.0);
    }

    fn save_state(&self) -> toml::Value {
        Saveable::save_state(&*self.0)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        Saveable::load_state(&*self.0, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_isp12160_brings_two_buses_and_a_qlogic_function() {
        let def = card_def("isp12160").expect("registered");
        let card = (def.build)(&Options::new(), 2).unwrap();
        assert_eq!(card.scsi_buses().len(), def.scsi_controllers as usize);
        let id = card.function().config().read(0);
        assert_eq!((id & 0xffff, id >> 16), (0x1077, 0x1216), "QLogic ISP12160");
        assert!(card_def("rad9").is_none());
    }

    #[test]
    fn two_cards_do_not_share_buses_or_state() {
        let def = card_def("isp12160").unwrap();
        let (a, b) = ((def.build)(&Options::new(), 2).unwrap(), (def.build)(&Options::new(), 2).unwrap());
        let (ba, bb) = (a.scsi_buses()[0] as *const ScsiBus, b.scsi_buses()[0] as *const ScsiBus);
        assert_ne!(ba, bb);
        b.load_state(&a.save_state()).expect("the same card's state loads");
    }
}
