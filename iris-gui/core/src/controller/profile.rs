//! The one per-profile table the GUI keeps (D8). The core says what hardware a
//! profile has (`MachineProfile::cpus()`, `memory_banks()`, `GraphicsBoard`);
//! what it does not say, and every shell needs, lives here, so adding a machine
//! is the core profile, one row here and one picture.

use iris::config::MachineProfile;

/// The groups the library shows machines in by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Family {
    Indy,
    Indigo2,
    O2,
    Octane,
    FuelTezro,
    OriginOnyx,
}

impl Family {
    pub const ALL: &'static [Family] =
        &[Family::Indy, Family::Indigo2, Family::O2, Family::Octane, Family::FuelTezro, Family::OriginOnyx];

    pub fn title(self) -> String {
        match self {
            Family::Indy => t!("library-family-indy"),
            Family::Indigo2 => t!("library-family-indigo2"),
            Family::O2 => t!("library-family-o2"),
            Family::Octane => t!("library-family-octane"),
            Family::FuelTezro => t!("library-family-fuel-tezro"),
            Family::OriginOnyx => t!("library-family-origin-onyx"),
        }
    }
}

/// A picture of the machine for the library and the summary: a front view in
/// black line art (`iris-gui/assets/icons/<svg_name>.svg`), which every shell
/// tints to its own theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MachineIcon {
    Indy,
    Indigo2,
    Indigo2Impact,
    O2,
    Octane,
    Origin200,
    Origin2000,
    Onyx2,
    Fuel,
    Tezro,
    Origin3200,
    Onyx3000,
}

impl MachineIcon {
    /// The file stem in `iris-gui/assets/icons/`. Machines in the same case
    /// share one (Indigo2 IMPACT, Onyx2, Onyx 3000).
    pub fn svg_name(self) -> &'static str {
        match self {
            MachineIcon::Indy => "machine-indy",
            MachineIcon::Indigo2 | MachineIcon::Indigo2Impact => "machine-indigo2",
            MachineIcon::O2 => "machine-o2",
            MachineIcon::Octane => "machine-octane",
            MachineIcon::Origin200 => "machine-origin200",
            MachineIcon::Origin2000 | MachineIcon::Onyx2 => "machine-origin2000",
            MachineIcon::Fuel => "machine-fuel",
            MachineIcon::Tezro => "machine-tezro",
            MachineIcon::Origin3200 | MachineIcon::Onyx3000 => "machine-origin3200",
        }
    }

    /// The SVG itself: 64×64, black on transparent, so a shell can use it
    /// as a mask or template and colour it.
    pub fn svg(self) -> &'static str {
        macro_rules! art {
            ($stem:literal) => {
                include_str!(concat!("../../../assets/icons/", $stem, ".svg"))
            };
        }
        match self {
            MachineIcon::Indy => art!("machine-indy"),
            MachineIcon::Indigo2 | MachineIcon::Indigo2Impact => art!("machine-indigo2"),
            MachineIcon::O2 => art!("machine-o2"),
            MachineIcon::Octane => art!("machine-octane"),
            MachineIcon::Origin200 => art!("machine-origin200"),
            MachineIcon::Origin2000 | MachineIcon::Onyx2 => art!("machine-origin2000"),
            MachineIcon::Fuel => art!("machine-fuel"),
            MachineIcon::Tezro => art!("machine-tezro"),
            MachineIcon::Origin3200 | MachineIcon::Onyx3000 => art!("machine-origin3200"),
        }
    }

    pub const ALL: [MachineIcon; 12] = [
        MachineIcon::Indy,
        MachineIcon::Indigo2,
        MachineIcon::Indigo2Impact,
        MachineIcon::O2,
        MachineIcon::Octane,
        MachineIcon::Origin200,
        MachineIcon::Origin2000,
        MachineIcon::Onyx2,
        MachineIcon::Fuel,
        MachineIcon::Tezro,
        MachineIcon::Origin3200,
        MachineIcon::Onyx3000,
    ];
}

/// The machine types as a menu shows them: workstations, then servers.
pub fn menu_groups() -> Vec<(String, Vec<MachineProfile>)> {
    use MachineProfile as P;
    vec![
        (
            t!("settings-group-workstations"),
            vec![P::IndyIp24, P::Indigo2Ip22, P::Indigo2Ip28, P::O2Ip32, P::OctaneIp30, P::FuelIp34, P::TezroIp53],
        ),
        (t!("settings-group-servers"), vec![P::Origin200Ip29, P::Origin2000Ip27, P::Onyx2Ip27, P::Origin3200Ip35, P::Onyx3000Ip35]),
    ]
}

/// Every profile in menu order: what a grouped choice's index counts in.
pub fn menu_order() -> Vec<MachineProfile> {
    menu_groups().into_iter().flat_map(|(_, v)| v).collect()
}

/// The machine type choice: grouped, with `p` selected.
pub fn profile_choice(p: MachineProfile) -> super::settings::FieldKind {
    let order = menu_order();
    super::settings::FieldKind::GroupedChoice {
        groups: menu_groups().into_iter().map(|(title, v)| (title, v.iter().map(|q| q.label().to_string()).collect())).collect(),
        selected: order.iter().position(|q| *q == p).unwrap_or(0),
    }
}

/// A processor as the GUI shows it: "MIPS R10000 (195 MHz)".
pub fn cpu_label(part: &iris::cpu::parts::Part) -> String {
    format!("MIPS R{} ({} MHz)", part.family as u32, part.mhz)
}

/// What the GUI needs to know about a profile beyond the core's tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileTraits {
    pub profile: MachineProfile,
    pub family: Family,
    pub icon: MachineIcon,
    /// The product as people say it: "Indy", "Indigo2 IMPACT", "Origin 3200".
    pub model: &'static str,
    /// The board number: "IP24".
    pub board: &'static str,
}

const fn row(profile: MachineProfile, family: Family, icon: MachineIcon, model: &'static str, board: &'static str) -> ProfileTraits {
    ProfileTraits { profile, family, icon, model, board }
}

/// One row per profile, in `MachineProfile::ALL` order.
pub const TRAITS: &[ProfileTraits] = &[
    row(MachineProfile::IndyIp24, Family::Indy, MachineIcon::Indy, "Indy", "IP24"),
    row(MachineProfile::Indigo2Ip22, Family::Indigo2, MachineIcon::Indigo2, "Indigo2", "IP22"),
    row(MachineProfile::Indigo2Ip28, Family::Indigo2, MachineIcon::Indigo2Impact, "Indigo2 IMPACT", "IP28"),
    row(MachineProfile::O2Ip32, Family::O2, MachineIcon::O2, "O2", "IP32"),
    row(MachineProfile::OctaneIp30, Family::Octane, MachineIcon::Octane, "Octane", "IP30"),
    row(MachineProfile::Origin200Ip29, Family::OriginOnyx, MachineIcon::Origin200, "Origin 200", "IP29"),
    row(MachineProfile::Origin2000Ip27, Family::OriginOnyx, MachineIcon::Origin2000, "Origin 2000", "IP27"),
    row(MachineProfile::Onyx2Ip27, Family::OriginOnyx, MachineIcon::Onyx2, "Onyx2", "IP27"),
    row(MachineProfile::FuelIp34, Family::FuelTezro, MachineIcon::Fuel, "Fuel", "IP34"),
    row(MachineProfile::TezroIp53, Family::FuelTezro, MachineIcon::Tezro, "Tezro", "IP53"),
    row(MachineProfile::Origin3200Ip35, Family::OriginOnyx, MachineIcon::Origin3200, "Origin 3200", "IP35"),
    row(MachineProfile::Onyx3000Ip35, Family::OriginOnyx, MachineIcon::Onyx3000, "Onyx 3000", "IP35"),
];

/// The traits row for `profile`.
pub fn traits(profile: MachineProfile) -> &'static ProfileTraits {
    TRAITS.iter().find(|t| t.profile == profile).expect("every profile has a ProfileTraits row")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_menu_lists_every_machine_once() {
        let order: Vec<_> = menu_order();
        assert_eq!(order.len(), MachineProfile::ALL.len());
        for p in MachineProfile::ALL {
            assert_eq!(order.iter().filter(|q| **q == p).count(), 1, "{p:?}");
        }
    }

    #[test]
    fn every_machine_picture_is_its_named_file() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets/icons");
        for icon in MachineIcon::ALL {
            let file = std::fs::read_to_string(dir.join(format!("{}.svg", icon.svg_name()))).unwrap();
            assert_eq!(icon.svg(), file, "{icon:?}");
            assert!(file.contains(r#"viewBox="0 0 64 64""#), "{icon:?}");
        }
        for t in TRAITS {
            assert!(MachineIcon::ALL.contains(&t.icon));
        }
    }

    #[test]
    fn every_profile_has_exactly_one_row() {
        for p in MachineProfile::ALL {
            assert_eq!(TRAITS.iter().filter(|t| t.profile == p).count(), 1, "{p:?}");
        }
        assert_eq!(TRAITS.len(), MachineProfile::ALL.len());
    }

    #[test]
    fn the_board_matches_the_core_label() {
        for t in TRAITS {
            assert!(t.profile.label().contains(t.board), "{:?}", t.profile);
        }
    }
}
