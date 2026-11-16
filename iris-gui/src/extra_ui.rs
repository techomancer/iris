//! Less common machine settings and a validated editor for the complete config.
use egui::{ComboBox, DragValue, TextEdit, Ui};
use iris::config::*;

/// The Indy and Indigo2 family: the machines built around the MC memory
/// controller, with Newport resolution presets, a DS1386 NVRAM and (on the
/// Indigo2) a motherboard EEPROM.
pub fn mc_machine(profile: MachineProfile) -> bool {
    matches!(profile, MachineProfile::IndyIp24 | MachineProfile::Indigo2Ip22 | MachineProfile::Indigo2Ip28)
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

/// The `[scsi.KEY]` key of the target at (`controller`, `id`), whichever
/// spelling the config used: the bare id with `controller = 1`, or `16 + id`.
pub fn scsi_key(cfg: &MachineConfig, controller: u8, id: u8) -> Option<u8> {
    cfg.scsi.iter().find(|(k, d)| d.placement(**k) == (controller, id)).map(|(k, _)| *k)
}

/// How a `[scsi.KEY]` target is named to the user: its id on controller 0,
/// `controller:id` on another (so key 19 reads "1:3", not "19").
pub fn scsi_name(cfg: &MachineConfig, key: u8) -> String {
    let (controller, id) = cfg.scsi.get(&key).map_or((0, key), |d| d.placement(key));
    if controller == 0 { id.to_string() } else { format!("{controller}:{id}") }
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
pub fn switch_profile(cfg: &mut MachineConfig) {
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
    cfg.graphics.heads = 1;
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
    if p.is_ip35_bricks() {
        // These machines use the I-brick's Fibre Channel adapter. The core
        // expects its historical default SCSI map to remain untouched.
        cfg.scsi = MachineConfig::default().scsi;
    } else {
        // Targets the new machine's buses cannot hold, which the Disks tab
        // would no longer show: a second controller, wide ids, DaynaPort.
        let (controllers, max_id) = (p.scsi_controllers(), p.max_scsi_id());
        let daynaport = matches!(p, MachineProfile::IndyIp24 | MachineProfile::Indigo2Ip22
            | MachineProfile::Indigo2Ip28 | MachineProfile::OctaneIp30);
        cfg.scsi.retain(|k, d| {
            let (c, id) = d.placement(*k);
            c < controllers && id <= max_id && (daynaport || !d.is_daynaport())
        });
    }
    // What `MachineConfig::validate` refuses on the new machine, and the
    // Advanced tab no longer shows for it.
    if !matches!(p, MachineProfile::FuelIp34 | MachineProfile::TezroIp53) {
        cfg.machine.emu_id = 0;
    }
    let octane = p == MachineProfile::OctaneIp30;
    if !octane {
        cfg.xio_board.clear();
    }
    if !(octane || p == MachineProfile::O2Ip32) {
        cfg.nvme = Default::default();
    } else if !octane {
        // The O2 has one slot: no option board to name.
        cfg.nvme.board = None;
        cfg.nvme.slot = None;
    }
    if !(octane || p.is_ip35_bricks()) {
        cfg.fibre_channel = None;
    } else if let (false, Some(fc)) = (octane, &mut cfg.fibre_channel) {
        fc.board = None;
        fc.slot = None;
        fc.bridge = false;
    }
    // On the Octane both cards sit on an `[[xio_board]]`; with none left
    // (an O2's NVMe, an IP35 loop) there is no slot for them.
    if octane && cfg.xio_board.is_empty() {
        cfg.nvme = Default::default();
        cfg.fibre_channel = None;
    }
    if !(mc_machine(p) || octane || p == MachineProfile::TezroIp53) {
        cfg.virtio_serial.enabled = false;
        cfg.virtio_9p.enabled = false;
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
            ui.checkbox(&mut cfg.test_device, t!("extra-test-device"));
            ui.checkbox(&mut cfg.cheritest_dump_hook, t!("extra-cheritest"));
            text(
                ui,
                &t!("extra-test-config"),
                &mut cfg.test_device_run_config,
            );
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
                ui.checkbox(&mut s.audio, t!("extra-tezro-audio"));
            }
            MachineProfile::Origin3200Ip35 => {
                let s = &mut cfg.origin3200;
                ui.horizontal(|ui| {
                    ui.label(t!("extra-c-bricks"));
                    ui.add(DragValue::new(&mut s.c_bricks).range(1..=2));
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
                let s = &mut cfg.onyx3000;
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
        // Only the Fuel and the Tezro take an identity (`validate`).
        if matches!(p, MachineProfile::FuelIp34 | MachineProfile::TezroIp53) {
            ui.horizontal(|ui| {
                ui.label(t!("extra-emu-id"));
                ui.add(DragValue::new(&mut cfg.machine.emu_id).range(0..=iris::identity::EMU_ID_MAX));
            });
        }
    });
}

pub fn expansion(ui: &mut Ui, cfg: &mut MachineConfig) {
    let p = cfg.machine.profile;
    // What each machine has room for, as `MachineConfig::validate` decides:
    // virtio on a GIO slot of the Indy and Indigo2 or on the Octane's and
    // Tezro's PCI; XIO option boards on the Octane; NVMe on the O2 or an
    // Octane option board; Fibre Channel on an Octane option board or the
    // IP35 I-brick.
    let gio = mc_machine(p);
    let octane = p == MachineProfile::OctaneIp30;
    let virtio = gio || octane || p == MachineProfile::TezroIp53;
    let nvme = octane || p == MachineProfile::O2Ip32;
    let fibre = octane || p.is_ip35_bricks();
    if !(virtio || nvme || fibre) {
        return;
    }
    ui.collapsing(t!("extra-expansion"), |ui| {
        if virtio {
            ui.checkbox(&mut cfg.virtio_serial.enabled, t!("extra-virtio"));
            if cfg.virtio_serial.enabled {
                if gio {
                    choice(
                        ui,
                        &t!("extra-gio-slot"),
                        "virtio_slot",
                        &mut cfg.virtio_serial.slot,
                        &[VirtioGioSlot::Exp0, VirtioGioSlot::Exp1],
                    );
                }
                let mut remove = None;
                for (i, p) in cfg.virtio_serial.ports.iter_mut().enumerate() {
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
                    cfg.virtio_serial.ports.remove(i);
                }
                if ui.button(t!("extra-add-port")).clicked() {
                    cfg.virtio_serial.ports.push(Default::default());
                }
            }
            // virtio-9p shares a host directory; the device needs a Unix host.
            if cfg!(unix) {
                ui.separator();
                let mut on = cfg.virtio_9p.enabled;
                if ui.checkbox(&mut on, t!("extra-9p")).changed() {
                    cfg.virtio_9p.enabled = on;
                }
                if cfg.virtio_9p.enabled {
                    let v = &mut cfg.virtio_9p;
                    if gio {
                        choice(
                            ui,
                            &t!("extra-gio-slot"),
                            "virtio_9p_slot",
                            &mut v.slot,
                            &[VirtioGioSlot::Exp0, VirtioGioSlot::Exp1],
                        );
                    }
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
            }
        }
        if octane {
            ui.separator();
            let mut remove = None;
            for (i, b) in cfg.xio_board.iter_mut().enumerate() {
                ui.push_id(("xio", i), |ui| {
                    ui.horizontal(|ui| {
                        ui.label(format!("XIO {i}"));
                        choice(
                            ui,
                            &t!("extra-xio-kind"),
                            "kind",
                            &mut b.kind,
                            &[
                                iris::dev::xio_pci::Kind::Cage,
                                iris::dev::xio_pci::Kind::Shoehorn,
                            ],
                        );
                        if ui.button(t!("extra-remove")).clicked() {
                            remove = Some(i);
                        }
                    });
                    let mut widget = b.widget.unwrap_or(0);
                    ui.horizontal(|ui| {
                        ui.label(t!("extra-widget"));
                        if ui.add(DragValue::new(&mut widget).range(0..=15)).changed() {
                            b.widget = (widget != 0).then_some(widget);
                        }
                    });
                });
            }
            if let Some(i) = remove {
                cfg.xio_board.remove(i);
            }
            if ui.button(t!("extra-add-xio")).clicked() {
                cfg.xio_board.push(XioBoardConfig {
                    kind: iris::dev::xio_pci::Kind::Shoehorn,
                    widget: None,
                });
            }
        }
        if nvme {
            ui.separator();
            // `[nvme]` is fitted when it differs from the default; unticking
            // removes the whole table, so no half-set card is left behind.
            let mut fitted = !cfg.nvme.is_default();
            if ui.checkbox(&mut fitted, t!("extra-nvme-fit")).changed() {
                cfg.nvme = Default::default();
                if fitted {
                    cfg.nvme.path = "disks/nvme.raw".into();
                }
            }
            if fitted {
                ui.label(t!("extra-nvme"));
                crate::config_ui::path_row_disk(ui, "nvme_path", &mut cfg.nvme.path, crate::config_ui::Pick::OpenFile, &crate::config_ui::disk_filters());
                ui.checkbox(&mut cfg.nvme.overlay, t!("extra-overlay"));
                ui.checkbox(&mut cfg.nvme.bridge, t!("extra-bridge"));
                if octane {
                    placement(ui, &mut cfg.nvme.board, &mut cfg.nvme.slot);
                }
            }
        }
        if fibre {
            ui.separator();
            let mut enabled = cfg.fibre_channel.is_some();
            if ui.checkbox(&mut enabled, t!("extra-fc")).changed() {
                cfg.fibre_channel = enabled.then(Default::default);
            }
            if let Some(fc) = &mut cfg.fibre_channel {
                if octane {
                    ui.checkbox(&mut fc.bridge, t!("extra-bridge"));
                    placement(ui, &mut fc.board, &mut fc.slot);
                }
                let mut remove = None;
                for (&id, disk) in &mut fc.disks {
                    ui.push_id(("fc", id), |ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("ID {id}"));
                            ui.text_edit_singleline(&mut disk.path);
                            ui.checkbox(&mut disk.overlay, t!("extra-overlay"));
                            ui.checkbox(&mut disk.write_cache, t!("extra-write-cache"));
                            if ui.button(t!("extra-remove")).clicked() {
                                remove = Some(id);
                            }
                        });
                    });
                }
                if let Some(id) = remove {
                    fc.disks.remove(&id);
                }
                if ui.button(t!("extra-add-disk")).clicked() {
                    if let Some(id) = (0..=iris::dev::isp2200::LOOP_ID_MAX).find(|id| !fc.disks.contains_key(id)) {
                        fc.disks.insert(id, Default::default());
                    }
                }
            }
        }
    });
}
fn placement(ui: &mut Ui, board: &mut Option<usize>, slot: &mut Option<u8>) {
    ui.horizontal(|ui| {
        ui.label(t!("extra-board"));
        let mut n = board.unwrap_or(0);
        if ui.add(DragValue::new(&mut n).range(0..=3)).changed() {
            *board = Some(n);
        }
        ui.label(t!("extra-slot"));
        let mut s = slot.unwrap_or(0);
        if ui.add(DragValue::new(&mut s).range(0..=31)).changed() {
            *slot = Some(s);
        }
        if ui.button(t!("common-auto")).clicked() {
            *board = None;
            *slot = None;
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

    #[test]
    fn every_profile_can_be_created_saved_and_switched_to() {
        assert_eq!(MachineProfile::ALL, MachineProfile::EVERY);
        for p in MachineProfile::ALL {
            let mut cfg = MachineConfig::default();
            cfg.machine.profile = p;
            switch_profile(&mut cfg);
            cfg.validate().unwrap_or_else(|e| panic!("{p:?}: {e}"));
            let loaded = parse_complete(&toml::to_string(&cfg).unwrap()).unwrap();
            assert_eq!(loaded.machine.profile, p);
            assert_eq!(loaded.banks, cfg.banks);
            // Changing away from an SN0 table must not leave a foreign table.
            cfg.origin200.flash = Some("other-flash.bin".into());
            switch_profile(&mut cfg);
            cfg.validate()
                .unwrap_or_else(|e| panic!("switch to {p:?}: {e}"));
        }
    }

    /// Switching away from a machine with expansion, extra disks and an
    /// identity leaves nothing the new machine refuses.
    #[test]
    fn switching_from_a_populated_machine_leaves_a_valid_config() {
        for from in [MachineProfile::OctaneIp30, MachineProfile::TezroIp53, MachineProfile::O2Ip32] {
            for to in MachineProfile::ALL {
                let mut cfg = MachineConfig::default();
                cfg.machine.profile = from;
                switch_profile(&mut cfg);
                match from {
                    MachineProfile::OctaneIp30 => {
                        cfg.xio_board.push(XioBoardConfig { kind: iris::dev::xio_pci::Kind::Shoehorn, widget: None });
                        cfg.nvme.path = "nvme.raw".into();
                        cfg.serial_log_b = Some("b.log".into());
                        cfg.scsi.insert(iris::config::scsi_slot(1, 12), ScsiDeviceConfig { path: "w.raw".into(), ..Default::default() });
                    }
                    MachineProfile::TezroIp53 => {
                        cfg.machine.emu_id = 7;
                        cfg.machine.cpus = Some(1);
                        cfg.jitv2.compile = JitCompile::Sync;
                        cfg.network.mode = NetMode::Off;
                    }
                    _ => {
                        cfg.nvme.path = "nvme.raw".into();
                        cfg.serial_log_b = Some("b.log".into());
                    }
                }
                cfg.validate().unwrap_or_else(|e| panic!("populated {from:?}: {e}"));
                cfg.machine.profile = to;
                switch_profile(&mut cfg);
                cfg.validate().unwrap_or_else(|e| panic!("{from:?} -> {to:?}: {e}"));
            }
        }
    }

    /// An older `[scsi.3] controller = 1` table is found at (1, 3) only, and
    /// is named 1:3; key 19 says the same.
    #[test]
    fn scsi_targets_are_found_by_placement() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.scsi.clear();
        cfg.scsi.insert(3, ScsiDeviceConfig { controller: 1, ..Default::default() });
        cfg.scsi.insert(iris::config::scsi_slot(1, 5), ScsiDeviceConfig::default());
        assert_eq!(scsi_key(&cfg, 0, 3), None);
        assert_eq!(scsi_key(&cfg, 1, 3), Some(3));
        assert_eq!(scsi_key(&cfg, 1, 5), Some(21));
        assert_eq!((scsi_name(&cfg, 3), scsi_name(&cfg, 21)), ("1:3".to_string(), "1:5".to_string()));
    }

    #[test]
    fn all_offered_ram_presets_validate() {
        for p in MachineProfile::ALL {
            for &mb in crate::ram::ram_presets(p) {
                let mut cfg = MachineConfig::default();
                cfg.machine.profile = p;
                switch_profile(&mut cfg);
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
            cfg.machine.profile = p;
            switch_profile(&mut cfg);
            ctx.run_ui(Default::default(), |ui| show(ui, &mut cfg, false))
                .drop_without_applying_deltas();
        }
    }
}
