//! `[[devices]]`: what a config places, and the device types it can name. An entry has a `type`, an optional `at` (a slot path,
//! which may start with a machine alias or another entry's `name`), a `name` and the type's options; each type is one row of [`DEVICE_TYPES`].

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::config::{ScsiDeviceConfig, ScsiKind};
use crate::topology::DeviceFamily;

/// What a device is, with its options. A family step adds a variant here and
/// its rows to [`DEVICE_TYPES`].
#[derive(Clone, Debug)]
pub enum DeviceModel {
    /// A target on a SCSI bus: `hdd`, `cdrom` or `daynaport`, by its kind.
    Scsi(ScsiDeviceConfig),
}

#[derive(Clone, Debug)]
pub struct ConfiguredDevice {
    pub name: Option<String>,
    pub at: Option<String>,
    pub device: DeviceModel,
    /// One of the disks a config with none gets: placed after everything else and
    /// dropped when it finds no slot. Never written to a file.
    pub default: bool,
}

pub struct DeviceTypeSpec {
    pub name: &'static str,
    pub family: DeviceFamily,
    /// The option keys the type takes besides `type`, `name` and `at`.
    keys: &'static [&'static str],
    parse: fn(&DeviceTypeSpec, toml::Table) -> Result<DeviceModel, String>,
}

/// Every device type, one row each.
pub const DEVICE_TYPES: &[DeviceTypeSpec] = &[
    DeviceTypeSpec { name: "hdd", family: DeviceFamily::Scsi, keys: &["path", "write_cache", "overlay", "scratch", "size_mb"], parse: scsi_target },
    DeviceTypeSpec { name: "cdrom", family: DeviceFamily::Scsi, keys: &["path", "discs", "overlay"], parse: scsi_target },
    DeviceTypeSpec { name: "daynaport", family: DeviceFamily::Scsi, keys: &["mac", "subnet"], parse: scsi_target },
];

fn scsi_kind(name: &str) -> ScsiKind {
    match name {
        "cdrom" => ScsiKind::Cdrom,
        "daynaport" => ScsiKind::Daynaport,
        _ => ScsiKind::Disk,
    }
}

fn scsi_target(t: &DeviceTypeSpec, opts: toml::Table) -> Result<DeviceModel, String> {
    let mut dev: ScsiDeviceConfig = toml::Value::Table(opts).try_into().map_err(|e: toml::de::Error| e.message().to_string())?;
    dev.kind_field = scsi_kind(t.name);
    dev.cdrom = dev.kind_field == ScsiKind::Cdrom;
    Ok(DeviceModel::Scsi(dev))
}

impl DeviceModel {
    pub fn spec(&self) -> &'static DeviceTypeSpec {
        let name = match self {
            DeviceModel::Scsi(d) => match d.kind() {
                ScsiKind::Disk => "hdd",
                ScsiKind::Cdrom => "cdrom",
                ScsiKind::Daynaport => "daynaport",
            },
        };
        DEVICE_TYPES.iter().find(|t| t.name == name).expect("every variant has a type")
    }

    pub fn scsi(&self) -> Option<&ScsiDeviceConfig> {
        match self {
            DeviceModel::Scsi(d) => Some(d),
        }
    }

    pub fn scsi_mut(&mut self) -> Option<&mut ScsiDeviceConfig> {
        match self {
            DeviceModel::Scsi(d) => Some(d),
        }
    }

    fn options(&self) -> toml::Table {
        let keys = self.spec().keys;
        let (now, default) = match self {
            DeviceModel::Scsi(d) => (toml::Table::try_from(d), toml::Table::try_from(ScsiDeviceConfig::default())),
        };
        let default = default.expect("options serialize");
        now.expect("options serialize").into_iter().filter(|(k, v)| keys.contains(&k.as_str()) && default.get(k) != Some(v)).collect()
    }
}

impl ConfiguredDevice {
    pub fn new(at: &str, device: DeviceModel) -> ConfiguredDevice {
        ConfiguredDevice { name: None, at: Some(at.to_string()), device, default: false }
    }

    pub fn from_table(mut t: toml::Table) -> Result<ConfiguredDevice, String> {
        let mut text = |key: &str| match t.remove(key) {
            None => Ok(None),
            Some(toml::Value::String(s)) => Ok(Some(s)),
            Some(v) => Err(format!("{key} = {v} is not a string")),
        };
        let name = text("type")?.ok_or("a device needs a type")?;
        let (at, local) = (text("at")?, text("name")?);
        let ty = DEVICE_TYPES.iter().find(|t| t.name == name).ok_or_else(|| {
            format!("unknown device type \"{name}\" (types: {})", DEVICE_TYPES.iter().map(|t| t.name).collect::<Vec<_>>().join(", "))
        })?;
        if let Some(key) = t.keys().find(|k| !ty.keys.contains(&k.as_str())) {
            return Err(format!("{name} takes no key {key} (keys: type, name, at, {})", ty.keys.join(", ")));
        }
        let device = (ty.parse)(ty, t).map_err(|e| format!("{name}: {e}"))?;
        Ok(ConfiguredDevice { name: local, at, device, default: false })
    }

    /// The entry's keys in the order a file lists them: `type`, `name`, `at`, then the options.
    fn keys(&self) -> Vec<(String, toml::Value)> {
        let mut keys = vec![("type".to_string(), self.device.spec().name.into())];
        keys.extend(self.name.clone().map(|n| ("name".to_string(), n.into())));
        keys.extend(self.at.clone().map(|a| ("at".to_string(), a.into())));
        keys.extend(self.device.options());
        keys
    }
}

impl Serialize for ConfiguredDevice {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let keys = self.keys();
        let mut m = s.serialize_map(Some(keys.len()))?;
        for (k, v) in &keys {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

impl<'de> Deserialize<'de> for ConfiguredDevice {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        ConfiguredDevice::from_table(toml::Table::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MachineProfile;

    fn entry(text: &str) -> Result<ConfiguredDevice, String> {
        ConfiguredDevice::from_table(toml::from_str(text).unwrap())
    }

    #[test]
    fn each_scsi_type_sets_its_kind_and_round_trips() {
        for (ty, kind) in [("hdd", ScsiKind::Disk), ("cdrom", ScsiKind::Cdrom), ("daynaport", ScsiKind::Daynaport)] {
            let e = entry(&format!("type = \"{ty}\"\nat = \"scsi0/t3\"")).unwrap();
            assert_eq!(e.device.scsi().unwrap().kind(), kind);
            assert_eq!(e.device.spec().name, ty);
            let back = ConfiguredDevice::from_table(toml::Table::try_from(&e).unwrap()).unwrap();
            assert_eq!((back.device.spec().name, back.at.as_deref()), (ty, Some("scsi0/t3")));
        }
        let e = entry("type = \"hdd\"\npath = \"a.raw\"\nwrite_cache = false").unwrap();
        #[derive(Serialize)]
        struct File { devices: Vec<ConfiguredDevice> }
        let text = toml::to_string(&File { devices: vec![e] }).unwrap();
        assert_eq!(text, "[[devices]]\ntype = \"hdd\"\npath = \"a.raw\"\nwrite_cache = false\n", "type first, defaults left out");
    }

    #[test]
    fn a_key_the_type_lacks_or_an_unknown_type_is_an_error() {
        assert!(entry("type = \"cdrom\"\nscratch = true").unwrap_err().contains("cdrom takes no key scratch"));
        assert!(entry("type = \"hdd\"\ncontroller = 1").unwrap_err().contains("no key controller"));
        assert!(entry("type = \"floppy\"").unwrap_err().contains("types: hdd, cdrom, daynaport"));
        assert!(entry("at = \"scsi0/t1\"").unwrap_err().contains("needs a type"));
    }
}
