//! The cards a user can fit in an add-in slot.
//!
//! A card is chosen by name, as a CPU or a machine is: [`PciCard`] is the
//! list of them, and [`PciCard::desc`] is the one place that names each
//! card's own description. Everything else about a card is in its own
//! file: its options as a typed struct, how it is built, and the
//! abilities it offers the machine through [`PciBusDevice`]'s defaulted
//! methods (a SCSI bus to bind disks to, a heartbeat, a thread to start).

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::PciBusDevice;

/// A card that can be fitted. `card = "aic7880"` in `[[pci]]`, and the same
/// word after `=` in `--pci`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
#[clap(rename_all = "lower")]
pub enum PciCard {
    /// Adaptec AIC-7880 Ultra SCSI controller: one bus, ids 0 to 15.
    #[serde(alias = "AIC7880", alias = "Aic7880")]
    #[clap(alias = "AIC7880", alias = "Aic7880")]
    Aic7880,
}

/// The SCSI buses a card brings the machine, for numbering controllers and
/// checking `[scsi.N]` before anything is built.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ScsiSpec {
    pub controllers: u8,
    /// The highest target id a bus on this card takes.
    pub max_id: u8,
}

/// What a card's constructor is given.
pub struct CardBuild<'a> {
    /// The entry's options beside `slot` and `card`; the card reads them
    /// into its own typed struct.
    pub options: &'a toml::Table,
    /// The number of the first SCSI controller the card takes.
    pub first_controller: u8,
}

/// What a card file says about its card.
pub struct CardDesc {
    pub summary: &'static str,
    pub scsi: ScsiSpec,
    /// Refuse options the card does not have or cannot take, without
    /// building anything.
    pub check: fn(&toml::Table) -> Result<(), String>,
    pub build: fn(&CardBuild) -> Result<Arc<dyn PciBusDevice>, String>,
}

impl PciCard {
    /// The word that names the card in a config and on the command line.
    pub fn name(self) -> String {
        clap::ValueEnum::to_possible_value(&self).expect("every card has a name").get_name().to_string()
    }

    /// The card's own description, from its own file.
    pub fn desc(self) -> &'static CardDesc {
        match self {
            PciCard::Aic7880 => &crate::dev::aic7880::CARD,
        }
    }
}

/// Read a card's options into its typed struct: fields it does not have
/// are refused by name, and the ones it has fall back to its defaults.
pub fn read_options<T: serde::de::DeserializeOwned>(options: &toml::Table) -> Result<T, String> {
    toml::Value::Table(options.clone()).try_into().map_err(|e: toml::de::Error| e.message().to_string())
}

/// The options as `k=v,...` in key order, for saved state and messages.
pub fn describe_options(options: &toml::Table) -> String {
    options.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ValueEnum;

    #[test]
    fn cards_are_named_in_lower_case_by_serde_and_clap_alike() {
        #[derive(Deserialize, Serialize)]
        struct Entry {
            card: PciCard,
        }
        let e: Entry = toml::from_str("card = \"aic7880\"").unwrap();
        assert_eq!(e.card, PciCard::Aic7880);
        assert_eq!(toml::to_string(&e).unwrap().trim(), "card = \"aic7880\"");
        assert_eq!(PciCard::Aic7880.name(), "aic7880");
        assert_eq!(PciCard::from_str("aic7880", false), Ok(PciCard::Aic7880));
        for spelling in ["AIC7880", "Aic7880"] {
            assert_eq!(PciCard::from_str(spelling, false), Ok(PciCard::Aic7880), "--pci {spelling}");
            let e: Entry = toml::from_str(&format!("card = \"{spelling}\"")).unwrap();
            assert_eq!(e.card, PciCard::Aic7880, "toml {spelling}");
        }
        assert!(PciCard::from_str("aIc7880", false).is_err());
        assert!(PciCard::from_str("nosuch", true).is_err());
        assert!(toml::from_str::<Entry>("card = \"nosuch\"").is_err());
    }

    #[test]
    fn options_are_described_in_key_order() {
        let t: toml::Table = toml::from_str("b = 2\na = \"x\"").unwrap();
        assert_eq!(describe_options(&t), "a=\"x\",b=2");
        assert_eq!(describe_options(&toml::Table::new()), "");
    }
}
