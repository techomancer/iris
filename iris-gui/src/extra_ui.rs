//! Less common machine settings and a validated editor for the complete config.
use egui::{ComboBox, DragValue, TextEdit, Ui};
use iris::config::*;

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
fn choice<T: serde::Serialize + serde::de::DeserializeOwned + Copy>(
    ui: &mut Ui,
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
                "renderer",
                &mut cfg.renderer,
                &[UiRenderer::Gl, UiRenderer::Wgpu],
            );
            choice(
                ui,
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
                "log_timestamps",
                &mut cfg.serial_log_timestamps,
                &[
                    iris::dev::serial_log::SerialLogTimestamps::Off,
                    iris::dev::serial_log::SerialLogTimestamps::Wall,
                    iris::dev::serial_log::SerialLogTimestamps::Ecs,
                ],
            );
            choice(
                ui,
                "network_mode",
                &mut cfg.network.mode,
                &[NetMode::Nat, NetMode::Stream, NetMode::Off],
            );
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
            text(ui, &t!("extra-perfetto-path"), &mut cfg.perfetto);
            text(ui, &t!("extra-perfetto-cats"), &mut cfg.perfetto_cats);
            text(ui, &t!("extra-perfetto-clock"), &mut cfg.perfetto_clock);
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
                    "fuel_console",
                    &mut s.console,
                    &[FuelConsole::L1, FuelConsole::Ioc3A, FuelConsole::Ioc3B],
                );
                port(ui, &t!("extra-l1-port"), &mut s.l1_port);
                text(ui, &t!("extra-l1-log"), &mut s.l1_log);
                text(ui, &t!("extra-l1-revision"), &mut s.l1_revision);
                text(ui, &t!("extra-brick-serial"), &mut s.brick_serial);
                ui.checkbox(&mut s.ps2_input, t!("extra-ps2"));
            }
            MachineProfile::TezroIp53 => {
                let s = &mut cfg.tezro;
                text(ui, &t!("extra-flash"), &mut s.flash);
                choice(
                    ui,
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
                    ui.add(DragValue::new(&mut s.c_bricks).range(1..=2));
                });
                text(ui, &t!("extra-flash"), &mut s.flash);
                choice(
                    ui,
                    "origin_console",
                    &mut s.console,
                    &[
                        Origin3200Console::L1,
                        Origin3200Console::Ioc3A,
                        Origin3200Console::Ioc3B,
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
                    "onyx_console",
                    &mut s.console,
                    &[
                        Origin3200Console::L1,
                        Origin3200Console::Ioc3A,
                        Origin3200Console::Ioc3B,
                    ],
                );
                port(ui, &t!("extra-l1-port"), &mut s.l1_port);
                text(ui, &t!("extra-l1-log"), &mut s.l1_log);
                text(ui, &t!("extra-l1-revision"), &mut s.l1_revision);
                text(ui, &t!("extra-brick-serial"), &mut s.brick_serial);
            }
            _ => (),
        }
        ui.horizontal(|ui| {
            ui.label(t!("extra-emu-id"));
            ui.add(DragValue::new(&mut cfg.machine.emu_id));
        });
    });
}

pub fn expansion(ui: &mut Ui, cfg: &mut MachineConfig) {
    ui.collapsing(t!("extra-expansion"), |ui| {
        ui.checkbox(&mut cfg.virtio_serial.enabled, t!("extra-virtio"));
        if cfg.virtio_serial.enabled {
            choice(
                ui,
                "virtio_slot",
                &mut cfg.virtio_serial.slot,
                &[VirtioGioSlot::Exp0, VirtioGioSlot::Exp1],
            );
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
        ui.separator();
        let mut remove = None;
        for (i, b) in cfg.xio_board.iter_mut().enumerate() {
            ui.push_id(("xio", i), |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("XIO {i}"));
                    choice(
                        ui,
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
        ui.separator();
        ui.label(t!("extra-nvme"));
        ui.text_edit_singleline(&mut cfg.nvme.path);
        ui.checkbox(&mut cfg.nvme.overlay, t!("extra-overlay"));
        ui.checkbox(&mut cfg.nvme.bridge, t!("extra-bridge"));
        placement(ui, &mut cfg.nvme.board, &mut cfg.nvme.slot);
        let mut enabled = cfg.fibre_channel.is_some();
        if ui.checkbox(&mut enabled, t!("extra-fc")).changed() {
            cfg.fibre_channel = enabled.then(Default::default);
        }
        if let Some(fc) = &mut cfg.fibre_channel {
            ui.checkbox(&mut fc.bridge, t!("extra-bridge"));
            placement(ui, &mut fc.board, &mut fc.slot);
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
                if let Some(id) = (0..=126).find(|id| !fc.disks.contains_key(id)) {
                    fc.disks.insert(id, Default::default());
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
        let current = toml::to_string_pretty(cfg).unwrap_or_default();
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
                        draft.baseline = toml::to_string_pretty(cfg).unwrap_or_default();
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
    let mut cfg: MachineConfig = toml::from_str(text).map_err(|e| e.to_string())?;
    cfg.banks_set = toml::from_str::<toml::Table>(text)
        .map_err(|e| e.to_string())?
        .contains_key("banks");
    cfg.fold_deprecated();
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
