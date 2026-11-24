//! Less common machine settings and a validated editor for the complete config.
use egui::{ComboBox, DragValue, TextEdit, Ui};
use iris::config::*;
use iris::dev::xio_pci::Kind;
use iris::topology::{scsi_target_slot, scsi_target, slots_take, DeviceFamily, DevicePlacement, MachineSlotKind};
use iris::devices::{DeviceModel, ConfiguredDevice};

/// The Indy and Indigo2 family: the machines built around the MC memory
/// controller, with Newport resolution presets, a DS1386 NVRAM and (on the
/// Indigo2) a motherboard EEPROM.
pub fn mc_machine(profile: MachineProfile) -> bool {
    matches!(profile, MachineProfile::IndyIp24 | MachineProfile::Indigo2Ip22 | MachineProfile::Indigo2Ip28)
}

/// A button filled with `fill` (the green Start, the red Stop, the blue
/// confirm buttons). Its text is white: the theme's text colour is dark grey
/// in light mode, which can hardly be read on these fills.
pub fn filled_button<'a>(text: impl Into<egui::RichText>, fill: egui::Color32) -> egui::Button<'a> {
    egui::Button::new(text.into().color(egui::Color32::WHITE)).fill(fill)
}

/// Whether `profile` has the Indigo2's motherboard EEPROM (`nveeprom`). The
/// Indy keeps its PROM environment in the NVRAM, the O2 in its flash.
pub fn has_nveeprom(profile: MachineProfile) -> bool {
    matches!(profile, MachineProfile::Indigo2Ip22 | MachineProfile::Indigo2Ip28)
}

/// Whether `cfg.prom` runs the PROM built into iris: `(embedded)`, an empty
/// value the GUI writes as `(embedded)` before a start, or the default name
/// with neither it nor the machine's release file name in the machine
/// folder (`prom_image::resolve`).
pub fn builtin_prom(cfg: &MachineConfig) -> bool {
    let prom = cfg.prom.as_str();
    let exists = |p: &str| std::path::Path::new(p).exists();
    prom.is_empty()
        || iris::dev::prom_image::is_embedded_keyword(prom)
        || (prom == DEFAULT_PROM
            && !exists(prom)
            && !cfg.machine.profile.release_prom_file().is_some_and(exists))
}

/// How the target at slot `key` (`scsi_slot(controller, id)`) is named to
/// the user: its id on controller 0, `controller:id` on another.
pub fn scsi_name(key: u8) -> String {
    if key < 16 { key.to_string() } else { format!("{}:{}", key / 16, key % 16) }
}

/// Choices reflect the boards actually fitted by machine constructors. The O2
/// has a fixed GBE display; its shared board field is only a placeholder.
pub fn board_supported(board: GraphicsBoard, profile: MachineProfile) -> bool {
    board.supports(profile)
        && (profile != MachineProfile::O2Ip32 || board == GraphicsBoard::Newport)
        && (profile != MachineProfile::Onyx2Ip27 || board != GraphicsBoard::Ir3)
}

/// Refit hardware when the user explicitly changes profiles. Do not run this
/// each frame: it would erase intentional edits and imported configurations.
pub fn switch_profile(cfg: &mut MachineConfig, from: MachineProfile) {
    let p = cfg.machine.profile;
    cfg.machine.cpu = None;
    cfg.machine.cpus = None;
    cfg.clock.fixed_mhz = None;
    cfg.banks = crate::dialogs::new_machine::distribute_ram(crate::ram::ram_presets(p)[0], p);
    cfg.banks_set = true;
    cfg.graphics.board = Some(if p.ip28() {
        GraphicsBoard::SolidImpact
    } else {
        GraphicsBoard::default_for(p)
    });
    // `compile = "sync"` needs one CPU; the new machine may start with more.
    if cfg.machine.cpus() > 1 {
        cfg.jitv2.compile = JitCompile::Async;
    }
    cfg.graphics.resolution = Default::default();
    cfg.octane.graphics = None;
    // SN0 rejects another product's private table, even when it is inactive.
    cfg.origin200 = Default::default();
    cfg.origin2000 = Default::default();
    cfg.onyx2 = Default::default();
    rehome_disks(cfg, from);
    // Cards need a GIO slot; a second head also needs a Newport board.
    if !mc_machine(p) || cfg.graphics_board() != GraphicsBoard::Newport {
        cfg.set_device("newport", false);
    }
    // What `MachineConfig::validate` refuses on the new machine, and the
    // Advanced tab no longer shows for it.
    if !matches!(p, MachineProfile::FuelIp34 | MachineProfile::TezroIp53) {
        cfg.machine.emu_id = 0;
    }
    if mc_machine(p) && cfg.network.mode == NetMode::Off {
        cfg.network.mode = NetMode::Nat;
    }
    // The console's own channel is logged by `serial_log`.
    if matches!(p, MachineProfile::OctaneIp30 | MachineProfile::O2Ip32) {
        cfg.serial_log_a = None;
    }
    if mc_machine(p) {
        cfg.serial_log_b = None;
    }
}

/// Whether a Fibre Channel adapter goes in one of the machine's PCI slots: not on the O2, and
/// not on the I-brick, whose adapter is built in.
fn takes_fibre_card(p: MachineProfile) -> bool {
    iris::topology::slots_take(&p.slots(), DeviceFamily::Pci) && p != MachineProfile::O2Ip32 && !p.is_ip35_bricks()
}

/// Keep each SCSI device through a profile switch at the same controller and target, dropping it where the new machine has
/// no such slot. Other devices keep their slot where they can, else take any, else go.
fn rehome_disks(cfg: &mut MachineConfig, from: MachineProfile) {
    let p = cfg.machine.profile;
    let (old, new) = (from.slots(), p.slots());
    let brick = new.iter().find(|s| s.kind == MachineSlotKind::Fixed("qla2200")).map(|s| s.path.clone());
    let mut old_cfg = cfg.clone();
    old_cfg.machine.profile = from;
    let was = DevicePlacement::lenient(&old_cfg, &cfg.devices);
    let places: Vec<_> = (0..cfg.devices.len()).map(|i| was.at(i).cloned()).collect();
    let devices = std::mem::take(&mut cfg.devices);
    let mut others = Vec::new();
    for (mut d, at) in devices.into_iter().zip(places) {
        if matches!(d.device, DeviceModel::Qla2200) && !takes_fibre_card(p) {
            continue;
        }
        if matches!(d.device, DeviceModel::Nvme(_)) && p.refuse_nvme().is_err() {
            continue;
        }
        if d.device.scsi().is_none() {
            others.push(d);
            continue;
        }
        if let Some(id) = at.as_ref().and_then(|a| a.loop_number()) {
            if let Some(b) = &brick {
                d.at = Some(b.loop_id(id).to_string());
                cfg.devices.push(d);
            }
            continue;
        }
        let target = at.and_then(|at| scsi_target(&old, &at)).and_then(|(c, id)| scsi_target_slot(&new, c, id));
        let Some(target) = target.filter(|_| p.takes_daynaport() || !d.device.scsi().is_some_and(|s| s.is_daynaport())) else { continue };
        // An alias that names the same slot on the new machine is kept as written.
        if DevicePlacement::lenient(cfg, std::slice::from_ref(&d)).at(0) != Some(&target) {
            d.at = Some(target.to_string());
        }
        cfg.devices.push(d);
    }
    for d in others {
        cfg.devices.push(d);
        if DevicePlacement::new(cfg).is_err() {
            cfg.devices.last_mut().expect("just pushed").at = None;
            if DevicePlacement::new(cfg).is_err() {
                cfg.devices.pop();
            }
        }
    }
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
            text(ui, &t!("extra-socket-a"), &mut cfg.serial_socket_a);
            text(ui, &t!("extra-socket-b"), &mut cfg.serial_socket_b);
            text(ui, &t!("extra-log-a"), &mut cfg.serial_log_a);
            text(ui, &t!("extra-log-b"), &mut cfg.serial_log_b);
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
                choice(
                    ui,
                    &t!("extra-console-select"),
                    "fuel_console",
                    &mut s.console,
                    &[Ip35Console::L1, Ip35Console::Ioc3A, Ip35Console::Ioc3B],
                );
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
                choice(
                    ui,
                    &t!("extra-console-select"),
                    "tezro_console",
                    &mut s.console,
                    &[
                        TezroConsole::L1,
                        TezroConsole::Ioc4A,
                        TezroConsole::Ioc4B,
                        TezroConsole::Ioc4C,
                        TezroConsole::Ioc4D,
                    ],
                );
                port(ui, &t!("extra-l1-port"), &mut s.l1_port);
                text(ui, &t!("extra-l1-log"), &mut s.l1_log);
                port(ui, &t!("extra-port-c"), &mut s.serial_port_c);
                port(ui, &t!("extra-port-d"), &mut s.serial_port_d);
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
                choice(
                    ui,
                    &t!("extra-console-select"),
                    "origin_console",
                    &mut s.console,
                    &[
                        Ip35Console::L1,
                        Ip35Console::Ioc3A,
                        Ip35Console::Ioc3B,
                    ],
                );
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
                choice(
                    ui,
                    &t!("extra-console-select"),
                    "onyx_console",
                    &mut s.console,
                    &[
                        Ip35Console::L1,
                        Ip35Console::Ioc3A,
                        Ip35Console::Ioc3B,
                    ],
                );
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
            let card = |cfg: &MachineConfig| cfg.devices.iter().position(|d| matches!(d.device, DeviceModel::Nvme(_)));
            let mut fitted = card(cfg).is_some();
            if ui.checkbox(&mut fitted, t!("extra-nvme-fit")).changed() {
                match card(cfg) {
                    Some(i) => {
                        let mut gone: Vec<usize> = behind(cfg, i).into_iter().chain([i]).collect();
                        gone.sort_unstable();
                        gone.into_iter().rev().for_each(|g| drop(cfg.devices.remove(g)));
                    }
                    None => cfg.devices.push(ConfiguredDevice {
                        at: None,
                        ..ConfiguredDevice::new("", DeviceModel::Nvme(NvmeConfig { path: "disks/nvme.raw".into(), ..Default::default() }))
                    }),
                }
            }
            if let Some(i) = card(cfg) {
                ui.label(t!("extra-nvme"));
                let n = cfg.devices[i].device.nvme_mut().expect("an nvme entry");
                crate::config_ui::path_row_disk(ui, "nvme_path", &mut n.path, crate::config_ui::Pick::OpenFile, &crate::config_ui::disk_filters());
                ui.checkbox(&mut n.overlay, t!("extra-overlay"));
                let mut bridged = behind(cfg, i).is_some();
                if ui.checkbox(&mut bridged, t!("extra-bridge")).changed() {
                    set_bridge(cfg, i, bridged);
                }
                let i = card(cfg).expect("still there");
                // A card behind a bridge moves with it.
                slot_choice(ui, cfg, behind(cfg, i).unwrap_or(i));
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
                        cfg.devices.push(ConfiguredDevice::new(&format!("{base}/l{id}"), DeviceModel::Scsi(Default::default())));
                    }
                }
            }
        }
    });
}

/// The local name the GUI gives the Fibre Channel card, so its loop's disks can name it.
const LOOP_CARD: &str = "fibre_channel";

/// The bridge the GUI puts a card behind.
const CARD_BRIDGE: &str = "nvme_bridge";

/// The `pci_bridge` entry card `i` sits behind, if it is behind the GUI's one.
fn behind(cfg: &MachineConfig, i: usize) -> Option<usize> {
    let first = cfg.devices[i].at.as_deref()?.strip_suffix("/pci0/0").filter(|n| *n == CARD_BRIDGE)?;
    cfg.devices.iter().position(|d| d.name.as_deref() == Some(first))
}

/// Put card `i` behind a bridge in its slot, or back in the bridge's slot.
fn set_bridge(cfg: &mut MachineConfig, i: usize, on: bool) {
    match behind(cfg, i) {
        Some(b) if !on => {
            let bridge = cfg.devices.remove(b);
            cfg.devices[if b < i { i - 1 } else { i }].at = bridge.at;
        }
        None if on => {
            let at = cfg.devices[i].at.replace(format!("{CARD_BRIDGE}/pci0/0"));
            let bridge = ConfiguredDevice { name: Some(CARD_BRIDGE.into()), at, ..ConfiguredDevice::new("", DeviceModel::PciBridge) };
            cfg.devices.insert(i, bridge);
        }
        _ => {}
    }
}

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
fn parse_complete(text: &str) -> Result<MachineConfig, String> {
    let cfg = MachineConfig::from_toml_str(text).map_err(|e| e.to_string())?;
    cfg.validate()?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iris::topology::scsi_buses;
    use iris::devices::ConfiguredDevice;

    fn switch(cfg: &mut MachineConfig, to: MachineProfile) {
        let from = cfg.machine.profile;
        cfg.machine.profile = to;
        switch_profile(cfg, from);
    }

    #[test]
    fn every_profile_can_be_created_saved_and_switched_to() {
        assert_eq!(MachineProfile::ALL, MachineProfile::EVERY);
        for p in MachineProfile::ALL {
            let mut cfg = MachineConfig::default();
            switch(&mut cfg, p);
            cfg.validate().unwrap_or_else(|e| panic!("{p:?}: {e}"));
            let loaded = parse_complete(&toml::to_string(&cfg).unwrap()).unwrap();
            assert_eq!(loaded.machine.profile, p);
            assert_eq!(loaded.banks, cfg.banks);
            // Changing away from an SN0 table must not leave a foreign table.
            cfg.origin200.flash = Some("other-flash.bin".into());
            switch(&mut cfg, p);
            cfg.validate()
                .unwrap_or_else(|e| panic!("switch to {p:?}: {e}"));
        }
    }

    /// Switching away from a machine with expansion, extra disks and an
    /// identity leaves nothing the new machine refuses.
    #[test]
    fn switching_from_a_populated_machine_leaves_a_valid_config() {
        let nvme = || ConfiguredDevice { at: None, ..ConfiguredDevice::new("", DeviceModel::Nvme(NvmeConfig { path: "nvme.raw".into(), ..Default::default() })) };
        for from in [MachineProfile::OctaneIp30, MachineProfile::TezroIp53, MachineProfile::O2Ip32] {
            for to in MachineProfile::ALL {
                let mut cfg = MachineConfig::default();
                switch(&mut cfg, from);
                match from {
                    MachineProfile::OctaneIp30 => {
                        cfg.devices.push(ConfiguredDevice { at: None, ..ConfiguredDevice::new("", DeviceModel::XioBoard(Kind::Shoehorn)) });
                        cfg.devices.push(ConfiguredDevice { at: None, ..ConfiguredDevice::new("", DeviceModel::XioBoard(Kind::Cage)) });
                        cfg.devices.push(nvme());
                        cfg.devices.push(ConfiguredDevice { name: Some("fibre_channel".into()), at: None, ..ConfiguredDevice::new("", DeviceModel::Qla2200) });
                        cfg.serial_log_b = Some("b.log".into());
                        cfg.set_disk(iris::config::scsi_slot(1, 12), ScsiDeviceConfig { path: "w.raw".into(), ..Default::default() });
                    }
                    MachineProfile::TezroIp53 => {
                        cfg.machine.emu_id = 7;
                        cfg.machine.cpus = Some(1);
                        cfg.jitv2.compile = JitCompile::Sync;
                        cfg.network.mode = NetMode::Off;
                    }
                    _ => {
                        cfg.devices.push(nvme());
                        cfg.serial_log_b = Some("b.log".into());
                    }
                }
                cfg.validate().unwrap_or_else(|e| panic!("populated {from:?}: {e}"));
                switch(&mut cfg, to);
                cfg.validate().unwrap_or_else(|e| panic!("{from:?} -> {to:?}: {e}"));
            }
        }
    }

    /// Every disk keeps its controller and target through a switch, or is dropped where the new machine has no such slot.
    #[test]
    fn disks_follow_their_controller_and_target_through_a_switch() {
        let disk = |at: &str| ConfiguredDevice::new(at, DeviceModel::Scsi(ScsiDeviceConfig { path: format!("{at}.raw"), ..Default::default() }));
        for from in MachineProfile::ALL {
            for to in MachineProfile::ALL {
                let mut cfg = MachineConfig::default();
                switch(&mut cfg, from);
                cfg.devices = scsi_buses(&from.slots()).iter().flat_map(|(_, at, ids)| [1, *ids.last().unwrap()].map(|id| disk(&format!("{at}/t{id}")))).collect();
                let before: Vec<u8> = cfg.disks().iter().map(|d| d.0).collect();
                switch(&mut cfg, to);
                cfg.validate().unwrap_or_else(|e| panic!("{from:?} -> {to:?}: {e}"));
                let after: Vec<u8> = cfg.disks().iter().map(|d| d.0).collect();
                let new = to.slots();
                let fits: Vec<u8> = before.iter().copied().filter(|&s| scsi_target_slot(&new, s / 16, s % 16).is_some()).collect();
                assert_eq!(after, fits, "{from:?} -> {to:?}");
                assert_eq!(cfg.devices.iter().filter(|d| d.device.scsi().is_some()).count(), after.len(), "{from:?} -> {to:?}: nothing without a slot is kept");
            }
        }
        // An alias stays as written; a full slot is rewritten for the new machine.
        let mut cfg = MachineConfig::default();
        cfg.devices = vec![disk("scsi0/t2"), disk("hpc3/c0/t3")];
        switch(&mut cfg, MachineProfile::OctaneIp30);
        let at: Vec<_> = cfg.devices.iter().filter(|d| d.device.scsi().is_some()).map(|d| d.at.clone().unwrap()).collect();
        assert_eq!(at, ["scsi0/t2", "xio15/pci0/0/t3"]);
    }

    /// A loop disk moves to the I-brick's loop, and nowhere else keeps it or the card.
    #[test]
    fn a_loop_disk_moves_to_the_ibricks_loop() {
        let mut cfg = MachineConfig::default();
        switch(&mut cfg, MachineProfile::OctaneIp30);
        cfg.devices.push(ConfiguredDevice { name: Some("cage".into()), ..ConfiguredDevice::new("shoebox", DeviceModel::XioBoard(Kind::Cage)) });
        cfg.devices.push(ConfiguredDevice { name: Some("fibre_channel".into()), ..ConfiguredDevice::new("cage/pci0", DeviceModel::Qla2200) });
        cfg.devices.push(ConfiguredDevice::new("fibre_channel/l7", DeviceModel::Scsi(ScsiDeviceConfig { path: "fc.raw".into(), ..Default::default() })));
        assert_eq!(cfg.loop_disks().iter().map(|d| d.0).collect::<Vec<_>>(), [7]);
        for to in [MachineProfile::Origin3200Ip35, MachineProfile::Onyx3000Ip35] {
            let mut moved = cfg.clone();
            switch(&mut moved, to);
            moved.validate().unwrap();
            let kept = moved.devices.iter().filter(|d| d.device.scsi().is_some());
            assert_eq!(kept.map(|d| d.at.clone()).collect::<Vec<_>>(), [Some("xio15/pci0/1/l7".to_string())]);
            assert!(!moved.has_device("qla2200"));
        }
        switch(&mut cfg, MachineProfile::IndyIp24);
        assert!(cfg.loop_disks().is_empty() && !cfg.has_device("qla2200"));
    }

    #[test]
    fn all_offered_ram_presets_validate() {
        for p in MachineProfile::ALL {
            for &mb in crate::ram::ram_presets(p) {
                let mut cfg = MachineConfig::default();
                switch(&mut cfg, p);
                cfg.banks = crate::dialogs::new_machine::distribute_ram(mb, p);
                assert_eq!(cfg.banks.iter().sum::<u32>(), mb, "{p:?}: {mb}");
                cfg.validate()
                    .unwrap_or_else(|e| panic!("{p:?} {mb} MB: {e}"));
            }
        }
    }

    #[test]
    fn complete_editor_rejects_invalid_and_unknown_settings() {
        assert!(parse_complete("scale = 0").is_err());
        assert!(parse_complete("made_up = true").is_err());
        assert!(parse_complete("[graphics]\ntile = 19").is_err());
    }

    #[test]
    fn new_controls_render_for_every_profile() {
        let ctx = egui::Context::default();
        for p in MachineProfile::ALL {
            let mut cfg = MachineConfig::default();
            switch(&mut cfg, p);
            ctx.run_ui(Default::default(), |ui| show(ui, &mut cfg, false))
                .drop_without_applying_deltas();
        }
    }
}
