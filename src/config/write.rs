//! The one writer of an iris.toml: the config in the current spellings, whole for a front end that saves it and minimal for
//! `config dump`, which leaves out what the machine's defaults already say.

use super::*;

/// How much of the config a file holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dump {
    /// Everything, as a front end that saves and reloads the config wants it.
    Full,
    /// Only what differs from a default config of the same machine, with each device's slot spelled out.
    Minimal,
}

impl MachineConfig {
    /// The config as a file: [`Dump::Minimal`] is what `config dump` writes, [`Dump::Full`] what the GUI saves.
    /// A config settled by [`Self::resolve_slots`] gives a file that reads back to the same machine.
    pub fn to_toml(&self, detail: Dump) -> Result<String, String> {
        let mut table = self.written()?;
        let profile = table.get("machine").and_then(|m| m.get("profile")).cloned();
        let mut devices = Some(self.devices.clone());
        if detail == Dump::Minimal {
            let base = MachineConfig { machine: MachineSection { profile: self.machine.profile, ..Default::default() }, ..Default::default() };
            prune(&mut table, &base.written()?);
            if let (Some(p), Some(machine)) = (profile, table.entry("machine").or_insert_with(|| toml::Table::new().into()).as_table_mut()) {
                machine.insert("profile".into(), p);
            }
            devices = self.minimal_devices();
        }
        // Scalars, then `[machine]`, then the other tables: a scalar after a table would belong to it.
        let (mut tables, scalars): (toml::Table, toml::Table) = table.into_iter().partition(|(_, v)| v.is_table() || is_tables(v));
        let machine = tables.remove("machine").map(|m| toml::Table::from_iter([("machine".to_string(), m)]));
        #[derive(Serialize)]
        struct Devices<'a> {
            devices: &'a [ConfiguredDevice],
        }
        let list = match &devices {
            Some(d) => toml::to_string(&Devices { devices: d }).map_err(|e| e.to_string())?,
            None => String::new(),
        };
        // `devices = []` is a scalar and goes first; entries are tables and go last.
        let empty = devices.as_ref().is_some_and(Vec::is_empty);
        let mut parts = vec![if empty { list.clone() } else { String::new() }];
        for part in [&scalars, &machine.unwrap_or_default(), &tables] {
            parts.push(toml::to_string(part).map_err(|e| e.to_string())?);
        }
        parts.push(if empty { String::new() } else { list });
        let mut text: Vec<&str> = parts.iter().map(|p| p.trim()).filter(|p| !p.is_empty()).collect();
        if detail == Dump::Minimal {
            text.insert(0, "# Only what differs from this machine's defaults.");
        }
        Ok(text.join("\n\n") + "\n")
    }

    /// Write the minimal file to `path`, which must not exist: a dump never replaces a file.
    pub fn dump_to(&self, path: &str) -> Result<(), String> {
        use std::io::Write;
        let text = self.to_toml(Dump::Minimal)?;
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path).map_err(|e| format!("{path}: {e}"))?;
        file.write_all(text.as_bytes()).map_err(|e| format!("{path}: {e}"))
    }

    /// The config as the table a file holds, minus `[[devices]]`: keys left to the machine stay unwritten so they keep
    /// their meaning after a reload (an automatic `[jitv2] threads`, and `banks` while at the generic default).
    fn written(&self) -> Result<toml::Table, String> {
        let mut table: toml::Table = toml::from_str(&toml::to_string(&self.for_writing()).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        table.remove("devices");
        if !self.banks_set && self.banks == default_banks() {
            table.remove("banks");
        }
        if !self.jitv2_threads_set {
            if let Some(toml::Value::Table(jit)) = table.get_mut("jitv2") {
                jit.remove("threads");
                if jit.is_empty() {
                    table.remove("jitv2");
                }
            }
        }
        Ok(table)
    }

    /// The devices a minimal file lists: `None` when all are the machine's own defaults, else every one, each at the slot it
    /// took, so the file means the same after the machine's placement changes.
    fn minimal_devices(&self) -> Option<Vec<ConfiguredDevice>> {
        if !self.devices.is_empty() && self.devices.iter().all(|d| d.default) {
            return None;
        }
        let placed = DevicePlacement::lenient(self, &self.devices);
        let mut out = self.devices.clone();
        for (i, d) in out.iter_mut().enumerate() {
            d.at = placed.at_text(i).or_else(|| d.at.take());
        }
        Some(out)
    }
}

fn is_tables(v: &toml::Value) -> bool {
    matches!(v, toml::Value::Array(a) if !a.is_empty() && a.iter().all(toml::Value::is_table))
}

/// Drop every key of `t` that `base` has with the same value.
fn prune(t: &mut toml::Table, base: &toml::Table) {
    t.retain(|k, v| match (v, base.get(k)) {
        (v, Some(b)) if v == b => false,
        (toml::Value::Table(vt), Some(toml::Value::Table(bt))) => {
            prune(vt, bt);
            !vt.is_empty()
        }
        _ => true,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(text: &str) -> MachineConfig {
        MachineConfig::from_toml_str(text).unwrap().resolved().unwrap()
    }

    /// A machine nobody configured dumps to its profile alone, on every machine: the baseline each file is pruned
    /// against is that machine's own defaults.
    #[test]
    fn a_default_machine_dumps_to_its_profile() {
        for p in MachineProfile::ALL {
            let cfg = load(&format!("[machine]\nprofile = \"{}\"\n", p.name()));
            assert_eq!(cfg.to_toml(Dump::Minimal).unwrap(), format!("# Only what differs from this machine's defaults.\n\n[machine]\nprofile = \"{}\"\n", p.name()));
        }
    }

    /// What a run is set to survives dump, load, dump unchanged, with each device at its resolved slot and the machine's
    /// defaults (the two default disks, an automatic thread count) left to the machine.
    #[test]
    fn a_dump_reads_back_to_the_same_file() {
        let cfg = load(r#"
            banks = [64, 64, 0, 0]
            serial_port_a = 9000
            [machine]
            profile = "octane_ip30"
            [[devices]]
            type = "hdd"
            path = "irix.img"
            [[devices]]
            type = "hostserial"
            at = "serial1"
            path = "/dev/ttyUSB0"
            [[port_forward]]
            proto = "tcp"
            host_port = 2222
            guest_port = 22
        "#);
        let text = cfg.to_toml(Dump::Minimal).unwrap();
        assert_eq!(text, r#"# Only what differs from this machine's defaults.

banks = [64, 64, 0, 0]
serial_port_a = 9000

[machine]
profile = "octane_ip30"

[[port_forward]]
bind = "localhost"
guest_port = 22
host_port = 2222
proto = "tcp"

[[devices]]
type = "hdd"
at = "scsi0/t1"
path = "irix.img"

[[devices]]
type = "hostserial"
at = "serial1"
path = "/dev/ttyUSB0"
"#);
        assert_eq!(load(&text).to_toml(Dump::Minimal).unwrap(), text);
        // A machine with no disks keeps saying so.
        let none = load("devices = []\n").to_toml(Dump::Minimal).unwrap();
        assert!(none.contains("devices = []") && load(&none).devices.is_empty(), "{none}");
    }

    /// The GUI's file is whole: it holds what a minimal one leaves out, and reads back too.
    #[test]
    fn the_full_file_holds_every_setting() {
        let cfg = load("[machine]\nprofile = \"indy_ip24\"\n");
        let text = cfg.to_toml(Dump::Full).unwrap();
        let table: toml::Table = toml::from_str(&text).unwrap();
        assert!(text.contains("[audio]") && text.contains("path = \"scsi1.raw\"") && !text.contains("banks") && table["jitv2"].get("threads").is_none(), "{text}");
        assert_eq!(load(&text).to_toml(Dump::Full).unwrap(), text);
    }
}
