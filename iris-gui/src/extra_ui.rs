//! Less common machine settings and a validated editor for the complete config.
use egui::{ComboBox, DragValue, TextEdit, Ui};
use iris::config::*;
use iris::dev::xio_pci::Kind;
use iris::topology::{slots_take, DeviceFamily, DevicePlacement, MachineSlotKind};
use iris::devices::{DeviceModel, ConfiguredDevice};

pub use iris_gui_core::model::hardware::{
    board_supported, builtin_prom, has_nveeprom, mc_machine, new_nvme_path, parse_complete, scsi_name, switch_profile, takes_fibre_card,
};

/// A button filled with `fill` (the green Start, the red Stop, the blue
/// confirm buttons). Its text is white: the theme's text colour is dark grey
/// in light mode, which can hardly be read on these fills.
pub fn filled_button<'a>(text: impl Into<egui::RichText>, fill: egui::Color32) -> egui::Button<'a> {
    egui::Button::new(text.into().color(egui::Color32::WHITE)).fill(fill)
}

fn text(ui: &mut Ui, label: &str, value: &mut Option<String>) {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut s = value.clone().unwrap_or_default();
        if ui.text_edit_singleline(&mut s).changed() {
            *value = (!s.is_empty()).then_some(s);
        }
    });
}
fn port(ui: &mut Ui, label: &str, value: &mut Option<u16>) {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut n = value.unwrap_or(0);
        if ui.add(DragValue::new(&mut n).range(0..=65535)).changed() {
            *value = (n != 0).then_some(n);
        }
    });
}
/// The port that serves CI and the console window, from the ones the machine has; the first is its own.
fn console_port(ui: &mut Ui, cfg: &mut MachineConfig) {
    let names = iris::dev::serial_dev::consoles(cfg.machine.profile);
    ui.horizontal(|ui| {
        ui.label(t!("extra-console-select"));
        ComboBox::from_id_salt("console_port").selected_text(cfg.machine.console.as_deref().unwrap_or(&names[0])).show_ui(ui, |ui| {
            for (i, name) in names.iter().enumerate() {
                if ui.selectable_label(cfg.machine.console.as_deref().unwrap_or(&names[0]) == name, name).clicked() {
                    cfg.machine.console = (i != 0).then(|| name.clone());
                }
            }
        });
    });
}
/// Each serial port's Unix socket and log file; its TCP port is under Host ports.
fn serial_ends(ui: &mut Ui, cfg: &mut MachineConfig) {
    use iris::dev::serial_dev::{SerialDeviceConfig, SerialKind};
    for (n, default) in iris::dev::serial_dev::ports(cfg.machine.profile) {
        let at = format!("serial{n}");
        let entry = cfg.devices.iter().position(|d| d.at.as_deref() == Some(&at));
        let mut line = match entry.map(|i| &cfg.devices[i].device) {
            None => SerialDeviceConfig { kind: SerialKind::Line, backend: default.map(|p| format!("tcp:{p}")).into_iter().collect(), ..Default::default() },
            Some(DeviceModel::Serial(s)) if s.kind == SerialKind::Line => s.clone(),
            Some(_) => continue,
        };
        let mut changed = false;
        for (kind, label) in [("unix", t!("extra-serial-socket", line = at.clone())), ("log", t!("extra-serial-log", line = at.clone()))] {
            let old = line.end(kind).map(str::to_string);
            let mut value = old.clone();
            text(ui, &label, &mut value);
            if value != old {
                line.set_end(kind, value.as_deref());
                changed = true;
            }
        }
        match (changed, entry) {
            (false, _) => {}
            (true, Some(i)) => cfg.devices[i].device = DeviceModel::Serial(line),
            (true, None) => cfg.devices.push(ConfiguredDevice::new(&at, DeviceModel::Serial(line))),
        }
    }
}
/// A labelled picker over `choices`, each shown by its configuration spelling.
fn choice<T: serde::Serialize + serde::de::DeserializeOwned + Copy>(
    ui: &mut Ui,
    label: &str,
    id: &str,
    value: &mut T,
    choices: &[T],
) {
    let name = |v: &T| {
        serde_json::to_value(v)
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned()
    };
    ui.horizontal(|ui| {
        ui.label(label);
        ComboBox::from_id_salt(id)
            .selected_text(name(value))
            .show_ui(ui, |ui| {
                for c in choices {
                    if ui
                        .selectable_label(name(value) == name(c), name(c))
                        .clicked()
                    {
                        *value = *c;
                    }
                }
            });
    });
}

pub fn show(ui: &mut Ui, cfg: &mut MachineConfig, running: bool) {
    ui.heading(t!("extra-heading"));
    ui.label(t!("extra-restart"));
    ui.add_enabled_ui(!running, |ui| {
        ui.push_id("platform_extras", |ui| {
            platform(ui, cfg);
            expansion(ui, cfg);
        });
        ui.collapsing(t!("extra-rendering"), |ui| {
            choice(
                ui,
                &t!("extra-renderer"),
                "renderer",
                &mut cfg.renderer,
                &[UiRenderer::Gl, UiRenderer::Wgpu],
            );
            choice(
                ui,
                &t!("extra-raster"),
                "raster",
                &mut cfg.graphics.raster,
                &[RasterUnit::Cpu, RasterUnit::Wgpu],
            );
            ui.horizontal(|ui| {
                ui.label(t!("extra-gpu-backend"));
                ui.text_edit_singleline(&mut cfg.graphics.gpu_backend);
            });
            ui.horizontal(|ui| {
                ui.label(t!("extra-raster-threads"));
                ui.add(DragValue::new(&mut cfg.graphics.threads).range(1..=256));
            });
            ui.horizontal(|ui| {
                ui.label(t!("extra-tile"));
                ComboBox::from_id_salt("raster_tile")
                    .selected_text(cfg.graphics.tile.to_string())
                    .show_ui(ui, |ui| {
                        for n in [8, 16, 32, 64, 128, 256] {
                            ui.selectable_value(&mut cfg.graphics.tile, n, n.to_string());
                        }
                    });
            });
            text(ui, &t!("extra-graphics-flash"), &mut cfg.graphics.flash);
        });
        ui.collapsing(t!("extra-connections"), |ui| {
            console_port(ui, cfg);
            serial_ends(ui, cfg);
            choice(
                ui,
                &t!("extra-log-timestamps"),
                "log_timestamps",
                &mut cfg.serial_log_timestamps,
                &[
                    iris::dev::serial_log::SerialLogTimestamps::Off,
                    iris::dev::serial_log::SerialLogTimestamps::Wall,
                    iris::dev::serial_log::SerialLogTimestamps::Ecs,
                ],
            );
            let mut modes = vec![NetMode::Nat, NetMode::Stream];
            if iris::build_features::PCAP {
                modes.push(NetMode::Pcap);
            }
            if !mc_machine(cfg.machine.profile) {
                modes.push(NetMode::Off);
            }
            choice(ui, &t!("extra-network-mode"), "network_mode", &mut cfg.network.mode, &modes);
            text(ui, &t!("extra-mac"), &mut cfg.network.mac);
            text(ui, &t!("extra-stream"), &mut cfg.network.stream);
            let mut hold = cfg.network.rx_hold.unwrap_or(true);
            if ui.checkbox(&mut hold, t!("extra-rx-hold")).changed() {
                cfg.network.rx_hold = Some(hold);
            }
            ui.checkbox(&mut cfg.lock_aspect_ratio, t!("extra-aspect"));
            ui.horizontal(|ui| {
                ui.label(t!("extra-scroll"));
                ui.add(DragValue::new(&mut cfg.mouse_scroll_pixels_per_line).range(0.0..=1000.0));
            });
        });
    });
    ui.add_enabled_ui(!running, |ui| {
        ui.collapsing(t!("extra-diagnostics"), |ui| {
            choice(
                ui,
                &t!("extra-jit-compile"),
                "jit_compile",
                &mut cfg.jitv2.compile,
                &[JitCompile::Async, JitCompile::Sync],
            );
            if mc_machine(cfg.machine.profile) {
                let mut on = cfg.has_device("test_device");
                if ui.checkbox(&mut on, t!("extra-test-device")).changed() {
                    cfg.set_device("test_device", on);
                }
                if let Some(DeviceModel::TestDevice(t)) = cfg.devices.iter_mut().map(|d| &mut d.device).find(|d| matches!(d, DeviceModel::TestDevice(_))) {
                    text(ui, &t!("extra-test-config"), &mut t.run_config);
                }
            }
            ui.checkbox(&mut cfg.cheritest_dump_hook, t!("extra-cheritest"));
        });
    });
    complete_editor(ui, cfg, running);
}

fn platform(ui: &mut Ui, cfg: &mut MachineConfig) {
    ui.collapsing(t!("extra-platform"), |ui| {
        let p = cfg.machine.profile;
        if p.is_ip27() {
            let section = match p {
                MachineProfile::Origin200Ip29 => &mut cfg.origin200,
                MachineProfile::Origin2000Ip27 => &mut cfg.origin2000,
                _ => &mut cfg.onyx2,
            };
            if p != MachineProfile::Origin200Ip29 {
                ui.horizontal(|ui| {
                    ui.label(t!("extra-nodes"));
                    ui.add(DragValue::new(&mut section.nodes).range(1..=2));
                });
                text(ui, &t!("extra-io-prom"), &mut section.io_prom);
            }
            text(ui, &t!("extra-flash"), &mut section.flash);
            text(
                ui,
                &t!("extra-controller-store"),
                &mut section.controller_store,
            );
        }
        match p {
            MachineProfile::FuelIp34 => {
                let s = &mut cfg.fuel;
                text(ui, &t!("extra-flash"), &mut s.flash);
                port(ui, &t!("extra-l1-port"), &mut s.l1_port);
                text(ui, &t!("extra-l1-log"), &mut s.l1_log);
                text(ui, &t!("extra-l1-revision"), &mut s.l1_revision);
                text(ui, &t!("extra-brick-serial"), &mut s.brick_serial);
                ui.checkbox(&mut s.ps2_input, t!("extra-ps2"));
                ui.checkbox(&mut s.usb_input, t!("extra-usb-input"));
            }
            MachineProfile::TezroIp53 => {
                let s = &mut cfg.tezro;
                text(ui, &t!("extra-flash"), &mut s.flash);
                port(ui, &t!("extra-l1-port"), &mut s.l1_port);
                text(ui, &t!("extra-l1-log"), &mut s.l1_log);
            }
            MachineProfile::Origin3200Ip35 => {
                let s = &mut cfg.origin3200;
                ui.horizontal(|ui| {
                    ui.label(t!("extra-c-bricks"));
                    let mut count = s.c_bricks.unwrap_or(1);
                    if ui.add(DragValue::new(&mut count).range(1..=2)).changed() {
                        s.c_bricks = Some(count);
                    }
                });
                text(ui, &t!("extra-flash"), &mut s.flash);
                port(ui, &t!("extra-l1-port"), &mut s.l1_port);
                text(ui, &t!("extra-l1-log"), &mut s.l1_log);
                text(ui, &t!("extra-l1-revision"), &mut s.l1_revision);
                text(ui, &t!("extra-brick-serial"), &mut s.brick_serial);
            }
            MachineProfile::Onyx3000Ip35 => {
                let mut count = cfg.c_bricks() as u8;
                let s = &mut cfg.onyx3000;
                ui.horizontal(|ui| {
                    ui.label(t!("extra-c-bricks"));
                    if ui.add(DragValue::new(&mut count).range(1..=2)).changed() {
                        s.c_bricks = Some(count);
                    }
                });
                text(ui, &t!("extra-flash"), &mut s.flash);
                port(ui, &t!("extra-l1-port"), &mut s.l1_port);
                text(ui, &t!("extra-l1-log"), &mut s.l1_log);
                text(ui, &t!("extra-l1-revision"), &mut s.l1_revision);
                text(ui, &t!("extra-brick-serial"), &mut s.brick_serial);
            }
            _ => (),
        }
        if matches!(p, MachineProfile::FuelIp34 | MachineProfile::TezroIp53) || p.is_ip35_bricks() {
            let mut on = cfg.has_device("ice1724");
            if ui.checkbox(&mut on, t!("extra-tezro-audio")).changed() {
                cfg.set_device("ice1724", on);
            }
        }
        // Only the Fuel and the Tezro take an identity (`validate`).
        if matches!(p, MachineProfile::FuelIp34 | MachineProfile::TezroIp53) {
            ui.horizontal(|ui| {
                ui.label(t!("extra-emu-id"));
                ui.add(DragValue::new(&mut cfg.machine.emu_id).range(0..=iris::identity::EMU_ID_MAX));
            });
        }
    });
}

/// The virtio-serial options being edited, if it is on.
fn virtio_serial(cfg: &mut MachineConfig) -> Option<&mut VirtioSerialConfig> {
    cfg.devices.iter_mut().find_map(|d| match &mut d.device {
        DeviceModel::VirtioSerial(v) => Some(v),
        _ => None,
    })
}

fn virtio_9p(cfg: &mut MachineConfig) -> Option<&mut Virtio9pConfig> {
    cfg.devices.iter_mut().find_map(|d| match &mut d.device {
        DeviceModel::Virtio9p(v) => Some(v),
        _ => None,
    })
}

fn virtio_serial_ui(ui: &mut Ui, v: &mut VirtioSerialConfig) {
    let mut remove = None;
    for (i, p) in v.ports.iter_mut().enumerate() {
        ui.push_id(i, |ui| {
            ui.group(|ui| {
                text(ui, &t!("extra-name"), &mut p.name);
                ui.checkbox(&mut p.console, t!("extra-console"));
                port(ui, &t!("extra-tcp-port"), &mut p.tcp_port);
                text(ui, &t!("extra-socket"), &mut p.socket);
                if ui.button(t!("extra-remove")).clicked() {
                    remove = Some(i);
                }
            });
        });
    }
    if let Some(i) = remove {
        v.ports.remove(i);
    }
    if ui.button(t!("extra-add-port")).clicked() {
        v.ports.push(Default::default());
    }
}

/// virtio-9p shares a host directory; the device needs a Unix host.
fn virtio_9p_ui(ui: &mut Ui, v: &mut Virtio9pConfig) {
    ui.horizontal(|ui| {
        ui.label(t!("extra-9p-path"));
        crate::config_ui::path_row(ui, "virtio_9p_path", &mut v.path, crate::config_ui::Pick::Dir, &[]);
    });
    ui.horizontal(|ui| {
        ui.label(t!("extra-9p-tag"));
        ui.text_edit_singleline(&mut v.tag);
    });
    ui.checkbox(&mut v.readonly, t!("extra-9p-readonly"));
    choice(
        ui,
        &t!("extra-9p-owners"),
        "virtio_9p_owners",
        &mut v.owners,
        &[Virtio9pOwners::Squash, Virtio9pOwners::Passthrough],
    );
    ui.horizontal(|ui| {
        ui.label(t!("extra-9p-uid"));
        ui.add(DragValue::new(&mut v.uid));
        ui.label(t!("extra-9p-gid"));
        ui.add(DragValue::new(&mut v.gid));
    });
    if v.owners == Virtio9pOwners::Passthrough {
        ui.horizontal(|ui| {
            ui.label(t!("extra-9p-uid-map"));
            ui.add(TextEdit::singleline(&mut v.uid_map).hint_text("0:1000,1000:1001:5"));
        });
        ui.horizontal(|ui| {
            ui.label(t!("extra-9p-gid-map"));
            ui.add(TextEdit::singleline(&mut v.gid_map).hint_text("0:1000"));
        });
    }
}

pub fn expansion(ui: &mut Ui, cfg: &mut MachineConfig) {
    let p = cfg.machine.profile;
    // What the machine has room for, from its slot list and what `validate` refuses.
    let slots = p.slots();
    let takes = |f| slots_take(&slots, f);
    let (virtio, xio) = (takes(DeviceFamily::Virtio), takes(DeviceFamily::Xio));
    let nvme = takes(DeviceFamily::Pci) && p.refuse_nvme().is_ok();
    let fibre_card = takes_fibre_card(p);
    let fibre = fibre_card || slots.iter().any(|s| s.kind == MachineSlotKind::Fixed("qla2200"));
    if !(virtio || nvme || fibre) {
        return;
    }
    ui.collapsing(t!("extra-expansion"), |ui| {
        if virtio {
            let mut on = virtio_serial(cfg).is_some();
            if ui.checkbox(&mut on, t!("extra-virtio")).changed() {
                cfg.set_device("virtio_serial", on);
            }
            if let Some(v) = virtio_serial(cfg) {
                virtio_serial_ui(ui, v);
            }
            if cfg!(unix) {
                ui.separator();
                let mut on = virtio_9p(cfg).is_some();
                if ui.checkbox(&mut on, t!("extra-9p")).changed() {
                    cfg.set_device("virtio_9p", on);
                }
                if let Some(v) = virtio_9p(cfg) {
                    virtio_9p_ui(ui, v);
                }
            }
        }
        if xio {
            ui.separator();
            let mut remove = None;
            for i in 0..cfg.devices.len() {
                let DeviceModel::XioBoard(mut kind) = cfg.devices[i].device else { continue };
                ui.push_id(("xio", i), |ui| {
                    ui.horizontal(|ui| {
                        choice(ui, &t!("extra-xio-kind"), "kind", &mut kind, &[Kind::Cage, Kind::Shoehorn]);
                        cfg.devices[i].device = DeviceModel::XioBoard(kind);
                        slot_choice(ui, cfg, i);
                        if ui.button(t!("extra-remove")).clicked() {
                            remove = Some(i);
                        }
                    });
                });
            }
            if let Some(i) = remove {
                cfg.devices.remove(i);
            }
            if ui.button(t!("extra-add-xio")).clicked() {
                cfg.devices.push(ConfiguredDevice { at: None, ..ConfiguredDevice::new("", DeviceModel::XioBoard(Kind::Shoehorn)) });
            }
        }
        if nvme {
            ui.separator();
            let mut remove = None;
            for i in 0..cfg.devices.len() {
                if !matches!(cfg.devices[i].device, DeviceModel::Nvme(_)) {
                    continue;
                }
                ui.push_id(("nvme", i), |ui| {
                    ui.label(t!("extra-nvme"));
                    let n = cfg.devices[i].device.nvme_mut().expect("an nvme entry");
                    crate::config_ui::path_row_disk(ui, ("nvme_path", i), &mut n.path, crate::config_ui::Pick::OpenFile, &crate::config_ui::disk_filters());
                    ui.checkbox(&mut n.overlay, t!("extra-overlay"));
                    slot_choice(ui, cfg, i);
                    if ui.button(t!("extra-remove")).clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove {
                cfg.devices.remove(i);
            }
            if ui.button(t!("extra-nvme-add")).clicked() {
                let path = new_nvme_path(cfg);
                cfg.devices.push(ConfiguredDevice { at: None, ..ConfiguredDevice::new("", DeviceModel::Nvme(NvmeConfig { path, ..Default::default() })) });
            }
        }
        if fibre {
            ui.separator();
            let card = |cfg: &MachineConfig| cfg.devices.iter().position(|d| matches!(d.device, DeviceModel::Qla2200));
            if fibre_card {
                let mut fitted = card(cfg).is_some();
                if ui.checkbox(&mut fitted, t!("extra-fc")).changed() {
                    match card(cfg) {
                        Some(i) => {
                            let name = cfg.devices.remove(i).name;
                            cfg.devices.retain(|d| name.is_none() || d.at.as_deref().and_then(|a| a.split('/').next()) != name.as_deref());
                        }
                        None => cfg.devices.push(ConfiguredDevice { at: None, name: Some(LOOP_CARD.into()), ..ConfiguredDevice::new("", DeviceModel::Qla2200) }),
                    }
                }
                if let Some(i) = card(cfg) {
                    slot_choice(ui, cfg, i);
                }
            }
            // The loop's slot: the Octane card's name, or the I-brick's fixed adapter.
            let base = match card(cfg) {
                Some(i) => cfg.devices[i].name.clone(),
                None => p.slots().iter().find(|s| s.kind == MachineSlotKind::Fixed("qla2200")).map(|s| s.path.to_string()),
            };
            if let Some(base) = base {
                let mut remove = None;
                for (id, i) in cfg.loop_disks() {
                    let disk = cfg.devices[i].device.scsi_mut().expect("listed by loop_disks");
                    ui.push_id(("fc", id), |ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("ID {id}"));
                            ui.text_edit_singleline(&mut disk.path);
                            ui.checkbox(&mut disk.overlay, t!("extra-overlay"));
                            ui.checkbox(&mut disk.write_cache, t!("extra-write-cache"));
                            if ui.button(t!("extra-remove")).clicked() {
                                remove = Some(i);
                            }
                        });
                    });
                }
                if let Some(i) = remove {
                    cfg.devices.remove(i);
                }
                if ui.button(t!("extra-add-disk")).clicked() {
                    let used: Vec<u8> = cfg.loop_disks().into_iter().map(|d| d.0).collect();
                    if let Some(id) = (0..=iris::dev::isp2200::LOOP_ID_MAX).find(|id| !used.contains(id)) {
                        cfg.devices.push(ConfiguredDevice::new(&format!("{base}/loop{id}"), DeviceModel::Scsi(Default::default())));
                    }
                }
            }
        }
    });
}

/// The local name the GUI gives the Fibre Channel card, so its loop's disks can name it.
const LOOP_CARD: &str = "fibre_channel";

/// Where entry `i` goes: automatic, or a free slot of its family.
fn slot_choice(ui: &mut Ui, cfg: &mut MachineConfig, i: usize) {
    // What sits at or below this entry goes with it, so its own slot is not in the way.
    let name = cfg.devices[i].name.clone();
    let others: Vec<ConfiguredDevice> = cfg.devices.iter().enumerate()
        .filter(|(j, d)| *j != i && (name.is_none() || d.at.as_deref().and_then(|a| a.split('/').next()) != name.as_deref()))
        .map(|(_, d)| d.clone()).collect();
    let others = DevicePlacement::lenient(cfg, &others);
    let family = cfg.devices[i].device.spec().family;
    let free: Vec<String> = others.slots.iter()
        .filter(|s| s.kind.takes(family, false) && !others.placed().any(|(_, at)| *at == s.path))
        .map(|s| s.path.to_string())
        .collect();
    let auto = t!("common-auto").to_string();
    let at = &mut cfg.devices[i].at;
    ui.label(t!("extra-slot"));
    ComboBox::from_id_salt(("slot", i)).selected_text(at.clone().unwrap_or_else(|| auto.clone())).show_ui(ui, |ui| {
        ui.selectable_value(at, None, auto);
        for p in free {
            ui.selectable_value(at, Some(p.clone()), p);
        }
    });
}

#[derive(Clone, Default)]
struct Draft {
    text: String,
    baseline: String,
    error: String,
}
fn complete_editor(ui: &mut Ui, cfg: &mut MachineConfig, running: bool) {
    ui.collapsing(t!("extra-complete"), |ui| {
        ui.label(t!("extra-complete-help"));
        let id = ui.id().with("complete_config");
        let current = crate::machines::to_toml(cfg).unwrap_or_default();
        let mut draft = ui
            .data_mut(|d| d.get_temp::<Draft>(id))
            .unwrap_or_else(|| Draft {
                text: current.clone(),
                baseline: current.clone(),
                error: String::new(),
            });
        // Automatically refresh an untouched draft after edits on another tab.
        if draft.text == draft.baseline {
            draft.text = current.clone();
            draft.baseline = current.clone();
        }
        let stale = draft.baseline != current;
        if stale {
            ui.colored_label(egui::Color32::YELLOW, t!("extra-stale"));
        }
        ui.add_enabled(
            !running,
            TextEdit::multiline(&mut draft.text)
                .code_editor()
                .desired_rows(24)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal(|ui| {
            if ui.button(t!("extra-reload")).clicked() {
                draft = Draft {
                    text: current.clone(),
                    baseline: current.clone(),
                    error: String::new(),
                };
            }
            if ui
                .add_enabled(!running && !stale, egui::Button::new(t!("extra-apply")))
                .clicked()
            {
                match parse_complete(&draft.text) {
                    Ok(next) => {
                        *cfg = next;
                        draft.baseline = crate::machines::to_toml(cfg).unwrap_or_default();
                        draft.text = draft.baseline.clone();
                        draft.error.clear();
                    }
                    Err(e) => draft.error = e,
                }
            }
        });
        if !draft.error.is_empty() {
            ui.colored_label(egui::Color32::LIGHT_RED, &draft.error);
        }
        ui.data_mut(|d| d.insert_temp(id, draft));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_controls_render_for_every_profile() {
        let ctx = egui::Context::default();
        for p in MachineProfile::ALL {
            let mut cfg = MachineConfig::default();
            let from = cfg.machine.profile;
            cfg.machine.profile = p;
            switch_profile(&mut cfg, from);
            ctx.run_ui(Default::default(), |ui| show(ui, &mut cfg, false))
                .drop_without_applying_deltas();
        }
    }
}
