use egui::{Color32, ComboBox, DragValue, Grid, RichText, ScrollArea, TextEdit, Ui};
use iris::build_features;
use std::path::Path;
use iris::config::{
    ForwardBind, ForwardProto, MachineConfig, NetMode, NfsConfig, PortForwardConfig,
    ScsiDeviceConfig, VinoSource, VinoStandard, VALID_BANK_SIZES,
};
use iris::nfsudp::NfsVersion;

/// A host network interface candidate for the PCAP backend selector. This is a
/// GUI-local, feature-independent copy of `iris::net_pcap::NetInterface` so the
/// App state and `show_network` signature don't need `#[cfg(feature = "pcap")]`.
#[derive(Debug, Clone)]
pub struct PcapIface {
    pub name: String,
    pub description: Option<String>,
    pub addrs: Vec<String>,
    pub up: bool,
    pub running: bool,
    pub loopback: bool,
}

impl PcapIface {
    /// One-line summary for the dropdown row.
    fn summary(&self) -> String {
        let mut s = self.name.clone();
        let mut tags = Vec::new();
        if self.up { tags.push("up"); }
        if self.running { tags.push("running"); }
        if self.loopback { tags.push("loopback"); }
        if !tags.is_empty() {
            s.push_str(&format!("  [{}]", tags.join(",")));
        }
        if let Some(ip) = self.addrs.first() {
            s.push_str(&format!("  {ip}"));
        }
        // Windows device names are opaque GUIDs; the description (NIC model) is
        // far more useful, so append it when present.
        if let Some(desc) = self.description.as_deref().filter(|d| !d.is_empty()) {
            s.push_str(&format!("  — {desc}"));
        }
        s
    }
}

/// Enumerate host interfaces for the PCAP selector. Returns the candidate list,
/// or an error string (insufficient privileges / no driver / feature missing).
/// Only does real work when built with `--features pcap`; otherwise returns a
/// hint so the UI can explain why the dropdown is unavailable.
pub fn enumerate_pcap_ifaces() -> Result<Vec<PcapIface>, String> {
    #[cfg(feature = "pcap")]
    {
        iris::net_pcap::list_interfaces().map(|list| {
            list.into_iter()
                .map(|i| PcapIface {
                    name: i.name,
                    description: i.description,
                    addrs: i.addresses.iter().map(|a| a.to_string()).collect(),
                    up: i.up,
                    running: i.running,
                    loopback: i.loopback,
                })
                .collect()
        })
    }
    #[cfg(not(feature = "pcap"))]
    {
        Err("this build lacks --features pcap; rebuild iris-gui with `--features pcap` \
             to enumerate and bridge onto host interfaces"
            .to_string())
    }
}

/// Which config tab is focused. Toolbar quick-buttons set this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    General,
    Disks,
    Network,
    Memory,
    Display,
    VideoIn,
    Debug,
    Ci,
}

impl Tab {
    /// Tabs to show for the active build. The Debug/JIT tab is hidden in
    /// lightning builds (the JIT debug paths it drives are compiled out), and
    /// the CI/Automation tab is hidden in App Store builds (the iris-ci socket
    /// is a developer automation feature, not something a sandboxed end user
    /// can use). Both fall back to the full set for ordinary builds.
    pub fn visible() -> Vec<Tab> {
        let mut tabs = vec![
            Tab::General, Tab::Disks, Tab::Network, Tab::Memory,
            Tab::Display, Tab::VideoIn,
        ];
        if !build_features::LIGHTNING {
            tabs.push(Tab::Debug);
        }
        if !cfg!(feature = "appstore") {
            tabs.push(Tab::Ci);
        }
        tabs
    }
    pub fn label(self) -> &'static str {
        match self {
            Tab::General => "General",
            Tab::Disks   => "Disks",
            Tab::Network => "Networking",
            Tab::Memory  => "Memory",
            Tab::Display => "Display",
            Tab::VideoIn => "Video-In",
            Tab::Debug   => "Debug / JIT",
            Tab::Ci      => "CI / Automation",
        }
    }
}

/// IRIS_JIT* environment variables exposed as GUI fields. These get exported
/// into the process env before `Machine::new` is called (whether iris is
/// hosted in-process or spawned). All optional; empty means "leave default".
#[derive(Debug, Clone, Default)]
pub struct JitEnv {
    pub iris_jit: bool,
    pub max_tier: Option<u8>,
    pub verify: bool,
    pub no_stores: bool,
    pub probe: String,
    pub trace_file: String,
    pub profile_file: String,
    pub no_idle: bool,
    pub debug_log: String,
}

impl JitEnv {
    /// Apply to current process env. Called by iris-gui before Machine::new.
    pub fn export(&self) {
        if self.iris_jit { std::env::set_var("IRIS_JIT", "1"); }
        if let Some(t) = self.max_tier { std::env::set_var("IRIS_JIT_MAX_TIER", t.to_string()); }
        if self.verify    { std::env::set_var("IRIS_JIT_VERIFY", "1"); }
        if self.no_stores { std::env::set_var("IRIS_JIT_NO_STORES", "1"); }
        if !self.probe.is_empty()         { std::env::set_var("IRIS_JIT_PROBE", &self.probe); }
        if !self.trace_file.is_empty()    { std::env::set_var("IRIS_JIT_TRACE", &self.trace_file); }
        if !self.profile_file.is_empty()  { std::env::set_var("IRIS_JIT_PROFILE", &self.profile_file); }
        if self.no_idle { std::env::set_var("IRIS_NO_IDLE", "1"); }
        if !self.debug_log.is_empty() { std::env::set_var("IRIS_DEBUG_LOG", &self.debug_log); }
    }
}

/// Action a config tab asks the app to perform that needs app-level state
/// (e.g. a confirmation modal) the immediate-mode tab UI doesn't own.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ConfigAction {
    #[default]
    None,
    /// User clicked "Use embedded PROM"; the app should confirm with the user
    /// and, if accepted, clear `cfg.prom` (an empty path falls back to the
    /// built-in PROM in `iris::prom::Prom::from_file_or_embedded`).
    RequestEmbeddedProm,
    /// User clicked "Test Camera" on the Video-In tab; the app should open the
    /// host camera and show a live preview (using the current `[vino]` standard
    /// and camera index).
    TestCamera,
    /// User clicked "Refresh" on the Network tab's PCAP selector; the app should
    /// re-enumerate host interfaces and update its cache.
    RefreshPcapIfaces,
    /// User clicked "Enable packet capture…" in the Network tab's PCAP section;
    /// the app should run the platform's privilege flow (Linux setcap/pkexec,
    /// macOS ChmodBPF install, Windows driver check) via `capture_access`.
    EnablePacketCapture,
    /// User picked a disc image for a CD-ROM while the machine is running —
    /// send Cmd::LoadDisc immediately without waiting for restart.
    LoadDisc { id: u8, path: String },
}

/// Everything a config tab hands back to the app for one frame.
#[derive(Default)]
pub struct TabOutcome {
    pub action: ConfigAction,
    pub net: NetworkOutcome,
    /// A SCSI image/disc path changed this frame (typed or picked) — mark dirty.
    pub disks_changed: bool,
    /// A SCSI image/disc path was just assigned via the Browse picker — the cue
    /// to (re)check CHD folder-grant permissions (see `check_chd_folder_grants`).
    pub disk_picked: bool,
}

pub fn show_tab(
    ui: &mut Ui,
    tab: Tab,
    cfg: &mut MachineConfig,
    jit: &mut JitEnv,
    host: &[crate::netplan::HostIface],
    disk_folders: &[String],
    pcap_ifaces: &Option<Result<Vec<PcapIface>, String>>,
) -> TabOutcome {
    ScrollArea::vertical().show(ui, |ui| match tab {
        Tab::General => TabOutcome { action: show_general(ui, cfg), ..Default::default() },
        Tab::Disks   => { let (e, a) = show_disks(ui, cfg); TabOutcome { action: a, disks_changed: e.changed, disk_picked: e.picked, ..Default::default() } }
        Tab::Network => {
            let net = show_network(ui, cfg, host, disk_folders, pcap_ifaces);
            TabOutcome { action: net.action.clone(), net, ..Default::default() }
        }
        Tab::Memory  => { show_memory(ui, cfg); TabOutcome::default() }
        Tab::Display => { show_display(ui, cfg); TabOutcome::default() }
        Tab::VideoIn => TabOutcome { action: show_vino(ui, cfg), ..Default::default() },
        Tab::Debug   => { show_debug(ui, cfg, jit); TabOutcome::default() }
        Tab::Ci      => { show_ci(ui, cfg); TabOutcome::default() }
    }).inner
}

fn show_general(ui: &mut Ui, cfg: &mut MachineConfig) -> ConfigAction {
    let mut action = ConfigAction::None;
    ui.heading("General");
    Grid::new("general_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("PROM image");
        path_row(ui, "prom", &mut cfg.prom, Pick::OpenFile, PROM_FILTERS);
        ui.end_row();

        // Leaving the PROM path empty boots the built-in PROM. Expose that as
        // an explicit button so reverting from a (possibly missing) custom PROM
        // is discoverable instead of "delete the text by hand". Disabled when
        // already empty, so the confirm prompt only ever appears when a custom
        // PROM is selected.
        ui.label("");
        ui.horizontal(|ui| {
            let custom = !cfg.prom.is_empty();
            if ui.add_enabled(custom, egui::Button::new("Use embedded PROM"))
                .on_hover_text("Boot IRIS's built-in PROM instead of a file")
                .clicked()
            {
                action = ConfigAction::RequestEmbeddedProm;
            }
            if !custom {
                ui.label(RichText::new("(using built-in PROM)").weak());
            }
        });
        ui.end_row();

        ui.label("NVRAM file");
        path_row(ui, "nvram", &mut cfg.nvram, Pick::SaveFile, NVRAM_FILTERS);
        ui.end_row();

        ui.label("Serial log (ttyd1 -> file)");
        path_row_opt(ui, "serial_log", &mut cfg.serial_log, Pick::SaveFile, ANY_FILTERS);
        ui.end_row();
    });

    // N64 development board (Ultra64). A single runtime toggle — the GIO device
    // and POSIX shm bridge (/iris_n64_bridge) are only created when this is on,
    // read once at VM start. Hidden in App Store builds: the sandbox can't open
    // the named shm or run the external gopher64 process it talks to.
    #[cfg(not(feature = "appstore"))]
    {
        ui.separator();
        ui.checkbox(&mut cfg.ultra64.enabled, "N64 development board (Ultra64)")
            .on_hover_text(
                "Emulate the SGI Indy N64 development board. Requires the gopher64 \
                 'ultra64' fork running alongside IRIS; load ROMs with `gload` from \
                 IRIX. Applies on next Start. See docs/ultra64.md.",
            );
    }

    action
}

fn show_memory(ui: &mut Ui, cfg: &mut MachineConfig) {
    ui.heading("Processor");
    Grid::new("cpu_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("CPU");
        ui.label(RichText::new(build_features::CPU).strong());
        ui.end_row();
    });
    ui.label(RichText::new(
        "The CPU is fixed at build time — the R4400 and R5000 differ in their cache \
         architecture, so it can't be switched at runtime. To use a different CPU, \
         download that build.")
        .weak());
    ui.separator();

    ui.heading("Memory");
    ui.label("RAM bank sizes in MB (valid: 0, 8, 16, 32, 64, 128)");
    Grid::new("mem_grid").num_columns(2).striped(true).show(ui, |ui| {
        for i in 0..4 {
            ui.label(format!("Bank {i}"));
            let cur = cfg.banks[i];
            ComboBox::from_id_salt(("bank", i)).selected_text(format!("{cur} MB"))
                .show_ui(ui, |ui| {
                    for &sz in VALID_BANK_SIZES {
                        ui.selectable_value(&mut cfg.banks[i], sz, format!("{sz} MB"));
                    }
                });
            ui.end_row();
        }
    });
    let total: u32 = cfg.banks.iter().sum();
    ui.label(format!("Total: {total} MB"));
}

fn show_display(ui: &mut Ui, cfg: &mut MachineConfig) {
    ui.heading("Display");
    Grid::new("disp_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("Window scale");
        ComboBox::from_id_salt("scale").selected_text(format!("{}×", cfg.scale))
            .show_ui(ui, |ui| {
                for s in 1u32..=4 {
                    ui.selectable_value(&mut cfg.scale, s, format!("{s}×"));
                }
            });
        ui.end_row();

        ui.label("Headless (no REX3 graphics)");
        ui.checkbox(&mut cfg.headless, "");
        ui.end_row();

        ui.label("No audio (disable HAL2)");
        ui.checkbox(&mut cfg.no_audio, "");
        ui.end_row();
    });
}

fn show_disks(ui: &mut Ui, cfg: &mut MachineConfig) -> (PathEdit, ConfigAction) {
    let mut edit = PathEdit::default();
    let mut action = ConfigAction::None;
    ui.heading("SCSI devices");
    ui.horizontal(|ui| {
        ui.label("IDs 1–7. CD-ROMs typically use 4–6.");
        if build_features::CHD {
            ui.label(RichText::new("[CHD support: ON]").color(Color32::LIGHT_GREEN).small());
        } else {
            ui.label(RichText::new("[CHD support: OFF — rebuild with --features chd]")
                .color(Color32::from_rgb(220, 170, 90)).small());
        }
    });
    let mut to_delete: Option<u8> = None;
    for id in 1u8..=7 {
        ui.separator();
        let exists = cfg.scsi.contains_key(&id);
        ui.horizontal(|ui| {
            ui.strong(format!("scsi{id}"));
            if exists {
                if ui.button("Remove").clicked() {
                    to_delete = Some(id);
                }
            } else if ui.button("Attach…").clicked() {
                cfg.scsi.insert(id, ScsiDeviceConfig {
                    path: format!("scsi{id}.raw"),
                    discs: vec![],
                    cdrom: false,
                    overlay: false,
                    scratch: false,
                    size_mb: None,
                });
            }
        });
        if let Some(dev) = cfg.scsi.get_mut(&id) {
            Grid::new(("scsi_grid", id)).num_columns(2).striped(true).show(ui, |ui| {
                ui.label("Image path");
                let e = path_row(ui, ("scsi_path", id), &mut dev.path,
                    if dev.scratch { Pick::SaveFile } else { Pick::OpenFile },
                    DISK_FILTERS);
                edit.changed |= e.changed;
                edit.picked |= e.picked;
                // For CD-ROMs, picking a path immediately loads the disc into
                // the running machine (equivalent to Ctrl+F12). The core's
                // count-driven queue decides whether this replaces the current
                // disc or joins the changer cycle.
                if dev.cdrom && e.picked && !dev.path.is_empty() {
                    action = ConfigAction::LoadDisc { id, path: dev.path.clone() };
                }
                ui.end_row();
                if dev.path.ends_with(".chd") && !build_features::CHD {
                    ui.label("");
                    ui.label(RichText::new("⚠ .chd path but this build lacks CHD support — rebuild with --features chd")
                        .color(Color32::from_rgb(230, 140, 70)));
                    ui.end_row();
                }
                // Active copy-on-write overlay for a compressed CHD: show exactly
                // which `.diff.chd` is in use (the path honours IRIS_CHD_DIFF_DIR,
                // so on the sandbox build this is the container sidecar) and its
                // size, so it's unambiguous that changes are landing here and that
                // this is the file folded back into the disk on a clean exit.
                if build_features::CHD && dev.path.ends_with(".chd") {
                    let diff = iris::chd_disk::diff_path_for(Path::new(&dev.path));
                    if let Ok(meta) = std::fs::metadata(&diff) {
                        let mb = meta.len() as f64 / (1024.0 * 1024.0);
                        ui.label("Active overlay");
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(format!("{}  ({mb:.1} MB)", diff.display()))
                                .weak().small())
                                .on_hover_text("This CHD's session changes accumulate here until they're \
                                                folded back into the disk on a clean exit.");
                            if ui.small_button("📂").on_hover_text("Reveal in file manager").clicked() {
                                reveal_in_file_manager(&diff.to_string_lossy());
                            }
                        });
                        ui.end_row();
                    }
                }

                ui.label("Type");
                let was_cd = dev.cdrom;
                let mut is_cd = dev.cdrom;
                ComboBox::from_id_salt(("type", id))
                    .selected_text(if is_cd { "CD-ROM" } else { "HDD" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut is_cd, false, "HDD");
                        ui.selectable_value(&mut is_cd, true, "CD-ROM");
                    });
                // Switching to CD-ROM defaults to an empty drive (no media):
                // clear the auto-generated HDD placeholder path so it doesn't
                // look like a (missing) disc. Load media via "Insert disc…" in
                // the SCSI menu, or just type a path here.
                if is_cd && !was_cd && dev.path == format!("scsi{id}.raw") {
                    dev.path.clear();
                }
                dev.cdrom = is_cd;
                ui.end_row();
                if dev.cdrom && dev.path.is_empty() {
                    ui.label("");
                    ui.label(RichText::new("empty drive (no media) — insert a disc via the SCSI menu")
                        .weak().small());
                    ui.end_row();
                }

                ui.label("Copy-on-write")
                    .on_hover_text(
                        "Keep this session's changes in a separate overlay instead of writing the disk \
                         directly, so you can roll back. Off: changes go into the disk (a compressed CHD \
                         still uses an overlay, folded back on a clean exit). Raw → .overlay; CHD → \
                         .diff.chd. Apply or discard with the monitor: `cow commit` / `cow reset`.");
                ui.checkbox(&mut dev.overlay, "")
                    .on_hover_text("Keep changes separate (roll back with `cow reset`, apply with `cow commit`)");
                ui.end_row();

                ui.label("Scratch volume");
                ui.checkbox(&mut dev.scratch, "");
                ui.end_row();

                if dev.scratch {
                    ui.label("Scratch size (MB)");
                    let mut sz = dev.size_mb.unwrap_or(64);
                    if ui.add(DragValue::new(&mut sz).range(1..=8192)).changed() {
                        dev.size_mb = Some(sz);
                    }
                    ui.end_row();
                }
            });

            if dev.cdrom {
                ui.label("Extra changer discs:");
                let mut drop_idx: Option<usize> = None;
                for (i, disc) in dev.discs.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        let e = path_row(ui, ("disc", id, i), disc, Pick::OpenFile, DISK_FILTERS);
                        edit.changed |= e.changed;
                        edit.picked |= e.picked;
                        if ui.button("×").clicked() { drop_idx = Some(i); }
                    });
                }
                if let Some(i) = drop_idx { dev.discs.remove(i); }
                if ui.button("+ Add disc").clicked() {
                    dev.discs.push(String::new());
                }
            }
        }
    }
    if let Some(id) = to_delete { cfg.scsi.remove(&id); }
    (edit, action)
}

/// A soft-invalid subnet the user just entered, surfaced to the app so it can
/// pop the "Override Sanity Checks / Cancel" confirmation modal.
pub struct NetSanityPrompt {
    /// Why it's questionable (non-RFC1918 or a host-network conflict).
    pub reason: String,
    /// A known-good CIDR offered as the safe alternative.
    pub suggestion: String,
    /// The `nat_subnet` value before this edit, restored if the user cancels.
    pub revert_to: Option<String>,
}

/// What the Networking tab asks the app to do beyond mutating `cfg`. Routed up
/// through [`show_tab`] because the immediate-mode tab can't own app-level state
/// (the dirty flag, the confirmation modal).
#[derive(Default)]
pub struct NetworkOutcome {
    /// A networking field changed this frame → the app should mark cfg dirty.
    pub changed: bool,
    /// A port-forward rule was added/removed/edited → the app should rebind the
    /// running NAT's forward listeners live.
    pub forwards_changed: bool,
    /// The PCAP host interface was changed and committed (dropdown pick, or the
    /// manual field lost focus) → reopen the running PcapEngine's capture on the
    /// new NIC without a guest reboot.
    pub iface_changed: bool,
    /// A soft-invalid subnet was just committed → pop the override modal.
    pub prompt: Option<NetSanityPrompt>,
    /// An app-level action requested from the tab (e.g. the PCAP "Refresh"
    /// button asking the app to re-enumerate host interfaces).
    pub action: ConfigAction,
}

fn show_network(
    ui: &mut Ui,
    cfg: &mut MachineConfig,
    host: &[crate::netplan::HostIface],
    disk_folders: &[String],
    pcap_ifaces: &Option<Result<Vec<PcapIface>, String>>,
) -> NetworkOutcome {
    use crate::netplan;
    let mut out = NetworkOutcome::default();
    ui.heading("Networking");

    // The backend selector (and the entire PCAP UI) is only shown when this
    // build actually has PCAP support. App Store / bundled builds compile
    // without `--features pcap`, where NAT is the only backend — so PCAP must
    // not appear anywhere in the UI (a dangling, non-functional option risks an
    // App Store rejection). Such builds also force NAT at runtime regardless of
    // a stale `mode = "pcap"` carried in from an imported config.
    if build_features::PCAP {
        Grid::new("net_mode_grid").num_columns(2).striped(true).show(ui, |ui| {
            ui.label("Backend");
            ComboBox::from_id_salt("net_mode")
                .selected_text(match cfg.network.mode {
                    NetMode::Nat  => "NAT gateway",
                    NetMode::Pcap => "PCAP (bridged)",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut cfg.network.mode, NetMode::Nat, "NAT gateway");
                    ui.selectable_value(&mut cfg.network.mode, NetMode::Pcap, "PCAP (bridged)");
                });
            ui.end_row();
        });

        if cfg.network.mode == NetMode::Pcap {
            let (a, iface_committed) = pcap_interface_picker(ui, cfg, pcap_ifaces);
            if let ConfigAction::RefreshPcapIfaces = a {
                out.action = a;
            }
            if iface_committed {
                out.iface_changed = true;
                out.changed = true;
            }

            ui.colored_label(Color32::from_rgb(0xd0, 0xa0, 0x40),
                "PCAP mode bridges onto a real interface — the guest joins your real LAN \
                 directly. The NAT gateway and port forwards don't apply here and are \
                 hidden below. Requires elevated privileges (root/CAP_NET_RAW on Unix, or \
                 a WinPcap-compatible driver + Administrator on Windows).");

            // Explicit, OS-specific way to grant capture permission up front (the
            // other trigger is automatic: the app pops the same prompt if a
            // pcap-mode machine hits a permission error on Start). The hint text
            // and the action are platform-specific — see `capture_access`.
            ui.horizontal(|ui| {
                if ui.button("Enable packet capture…")
                    .on_hover_text("Grant IRIS permission to capture on a real interface. \
                                    A one-time admin/root step per the note below.")
                    .clicked()
                {
                    out.action = ConfigAction::EnablePacketCapture;
                }
                ui.label(RichText::new(crate::capture_access::permission_hint()).weak());
            });
            ui.separator();
        }
    }

    // The NAT subnet settings and port forwards only apply to the software
    // gateway. In PCAP bridged mode the guest is directly on your real LAN, so
    // they're meaningless — hide the whole block. (The NFS share further below
    // works in both modes.)
    if cfg.network.mode != NetMode::Pcap {
    ui.label(RichText::new(
        "IRIS gives the Indy its own private NAT network, the same trick your home router uses. \
         The Indy reaches the internet through IRIS, but nothing on your real network can see it. \
         Pick a subnet that does not overlap a network your computer already uses (Wi-Fi, Ethernet, \
         VPN, Docker, etc.). If it does, IRIS flags it below.")
        .weak());
    ui.add_space(6.0);

    // The UI exposes the network base address (a plain IPv4, not CIDR) and the
    // mask separately; they compose into `cfg.nat_subnet` (a CIDR string the
    // backend wants, always snapped to a clean network address). The custom-mode
    // flags and the base text buffer persist in egui memory so partial typing
    // and the "Custom" reveal survive across frames; `last_id` tracks the value
    // we last stored so an external change (Cancel revert, machine switch)
    // re-syncs the controls.
    let before = cfg.nat_subnet.clone();
    let (base0, prefix0) = netplan::parse_cidr(cfg.nat_subnet.as_deref());

    let net_custom_id  = ui.make_persistent_id("net_net_custom");
    let mask_custom_id = ui.make_persistent_id("net_mask_custom");
    let base_buf_id    = ui.make_persistent_id("net_base_buf");
    let last_id        = ui.make_persistent_id("net_last_composed");

    let preset_mask = |p: u8| netplan::MASK_PRESETS.contains(&p);
    let mut net_custom:  bool = ui.data_mut(|d| d.get_temp::<bool>(net_custom_id)).unwrap_or(false);
    let mut mask_custom: bool = ui.data_mut(|d| d.get_temp::<bool>(mask_custom_id)).unwrap_or(!preset_mask(prefix0));
    let mut base_text: String = ui.data_mut(|d| d.get_temp::<String>(base_buf_id)).unwrap_or_else(|| base0.to_string());

    let mut prefix = prefix0;
    let fa = [
        netplan::first_free(netplan::PrivateBlock::C, prefix, host),
        netplan::first_free(netplan::PrivateBlock::B, prefix, host),
        netplan::first_free(netplan::PrivateBlock::A, prefix, host),
    ];

    // Re-sync the controls if the stored subnet changed outside this code.
    let cur = cfg.nat_subnet.clone().unwrap_or_default();
    let last: String = ui.data_mut(|d| d.get_temp::<String>(last_id)).unwrap_or_default();
    if cur != last {
        net_custom = !fa.contains(&base0);
        mask_custom = !preset_mask(prefix0);
        base_text = base0.to_string();
    }
    let mut base = if net_custom { base_text.parse().unwrap_or(base0) } else { base0 };

    let mut changed = false;     // any subnet field changed → recompose + mark dirty
    let mut committed = false;   // a deliberate commit → eligible for the override modal

    Grid::new("nat_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("Network");
        ui.horizontal(|ui| {
            let sel = if net_custom { "Custom".to_string() } else { base.to_string() };
            ComboBox::from_id_salt("net_preset").selected_text(sel).show_ui(ui, |ui| {
                for addr in fa {
                    if ui.selectable_label(!net_custom && base == addr, addr.to_string()).clicked() {
                        base = addr; net_custom = false; changed = true; committed = true;
                    }
                }
                if ui.selectable_label(net_custom, "Custom").clicked() {
                    net_custom = true;
                    base_text = base.to_string(); // seed the field with the current address
                }
            });
            if net_custom {
                let resp = ui.add(TextEdit::singleline(&mut base_text)
                    .hint_text("192.168.0.0").desired_width(130.0));
                if resp.changed() {
                    if let Ok(ip) = base_text.parse::<std::net::Ipv4Addr>() { base = ip; changed = true; }
                }
                if resp.lost_focus() { committed = true; }
            }
            // A custom mask shows the prefix right beside the base address.
            if mask_custom {
                ui.label("/");
                let mut bits = prefix.clamp(8, netplan::MAX_PREFIX);
                if ui.add(DragValue::new(&mut bits).range(8..=netplan::MAX_PREFIX)).changed() {
                    prefix = bits; changed = true; committed = true;
                }
            }
        });
        ui.end_row();

        ui.label("Subnet mask");
        let sel = if mask_custom { format!("Custom  (/{prefix})") } else { netplan::mask_label(prefix) };
        ComboBox::from_id_salt("net_mask").selected_text(sel).show_ui(ui, |ui| {
            for &p in netplan::MASK_PRESETS {
                if ui.selectable_label(!mask_custom && prefix == p, netplan::mask_label(p)).clicked() {
                    prefix = p; mask_custom = false; changed = true; committed = true;
                }
            }
            if ui.selectable_label(mask_custom, "Custom").clicked() {
                mask_custom = true;
                if preset_mask(prefix) { prefix = 24; } // default a fresh custom mask to /24
                changed = true;
            }
        });
        ui.end_row();
    });

    if changed {
        cfg.nat_subnet = Some(netplan::to_cidr(base, prefix));
        out.changed = true;
    }
    ui.data_mut(|d| {
        d.insert_temp(net_custom_id, net_custom);
        d.insert_temp(mask_custom_id, mask_custom);
        d.insert_temp(base_buf_id, base_text);
        d.insert_temp(last_id, cfg.nat_subnet.clone().unwrap_or_default());
    });

    // Derived addressing + sanity, from the live (unsnapped) base + prefix so the
    // snap note stays consistent across frames.
    let assess = netplan::classify(base, prefix, host);
    if let Some(msg) = &assess.hard_error {
        ui.label(RichText::new(format!("Invalid: {msg}")).color(Color32::from_rgb(0xd9, 0x4a, 0x3d)));
    } else if let Some(d) = &assess.derived {
        ui.label(RichText::new(format!(
            "Gateway (IRIS host) {}, Indy ec0 {}, {} usable hosts, broadcast {}",
            d.gateway, d.client, netplan::commas(d.usable_hosts), d.broadcast)).weak());
        if let Some(typed) = assess.off_boundary {
            ui.label(RichText::new(format!(
                "{typed} is not a network address; using {}/{}.", d.network, d.prefix)).weak());
        }
        match &assess.soft {
            Some(w) => {
                ui.horizontal(|ui| {
                    let sug = netplan::to_cidr(w.suggestion_net, w.suggestion_prefix);
                    ui.label(RichText::new(format!("Warning: {}", w.reason))
                        .color(Color32::from_rgb(0xd9, 0x9a, 0x3d)));
                    if ui.button(format!("Use {sug}")).clicked() {
                        cfg.nat_subnet = Some(sug); // external-change re-sync repaints the controls
                        out.changed = true;
                    }
                });
            }
            None => {
                ui.label(RichText::new("OK: private range, no conflict with your host networks")
                    .color(Color32::from_rgb(0x35, 0xb8, 0x4a)));
            }
        }
        // Pop the override modal when a deliberate edit (preset/mask pick, custom
        // mask bits, or the base field losing focus) lands on a soft-invalid
        // subnet. Live keystrokes don't trigger it — only a committed value does.
        if committed && assess.soft.is_some() {
            let w = assess.soft.as_ref().unwrap();
            out.prompt = Some(NetSanityPrompt {
                reason: w.reason.clone(),
                suggestion: netplan::to_cidr(w.suggestion_net, w.suggestion_prefix),
                revert_to: before.clone(),
            });
        }
    }

    ui.separator();
    ui.strong("Port forwards");
    let mut drop: Option<usize> = None;
    for (i, pf) in cfg.port_forward.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            let mut c = false;
            ComboBox::from_id_salt(("proto", i))
                .selected_text(match pf.proto { ForwardProto::Tcp => "tcp", ForwardProto::Udp => "udp" })
                .show_ui(ui, |ui| {
                    c |= ui.selectable_value(&mut pf.proto, ForwardProto::Tcp, "tcp").changed();
                    c |= ui.selectable_value(&mut pf.proto, ForwardProto::Udp, "udp").changed();
                });
            ui.label("host");
            c |= ui.add(DragValue::new(&mut pf.host_port).range(1..=65535)).changed();
            ui.label("to guest");
            c |= ui.add(DragValue::new(&mut pf.guest_port).range(1..=65535)).changed();
            ComboBox::from_id_salt(("bind", i))
                .selected_text(match pf.bind { ForwardBind::Localhost => "localhost", ForwardBind::Any => "any" })
                .show_ui(ui, |ui| {
                    c |= ui.selectable_value(&mut pf.bind, ForwardBind::Localhost, "localhost").changed();
                    c |= ui.selectable_value(&mut pf.bind, ForwardBind::Any, "any").changed();
                });
            if ui.button("Remove").clicked() { drop = Some(i); }
            out.changed |= c;
            out.forwards_changed |= c;
        });
    }
    if let Some(i) = drop { cfg.port_forward.remove(i); out.changed = true; out.forwards_changed = true; }

    let has_port = |p: u16| cfg.port_forward.iter().any(|f| f.guest_port == p);
    let mut add: Option<PortForwardConfig> = None;
    ui.menu_button("+ Add forward", |ui| {
        if ui.add_enabled(!has_port(23), egui::Button::new("Telnet (host 2323 to guest 23)"))
            .on_hover_text("Log in with: telnet localhost 2323. Needs the guest on IRIS's NAT subnet with telnetd running.")
            .clicked()
        {
            add = Some(PortForwardConfig { proto: ForwardProto::Tcp, host_port: 2323, guest_port: 23, bind: ForwardBind::Localhost });
            ui.close_menu();
        }
        if ui.add_enabled(!has_port(21), egui::Button::new("FTP (host 2121 to guest 21)"))
            .on_hover_text("Reach the guest's FTP server. Forwards the control port; file transfer also needs the data channel (see docs).")
            .clicked()
        {
            add = Some(PortForwardConfig { proto: ForwardProto::Tcp, host_port: 2121, guest_port: 21, bind: ForwardBind::Localhost });
            ui.close_menu();
        }
        if ui.add_enabled(!has_port(177), egui::Button::new("XDMCP (host 11177 to guest 177, UDP)"))
            .on_hover_text("Remote X login: an X server XDMCP-queries the guest's xdm. Binds all interfaces (LAN X servers OK). Stock X servers use UDP 177 — redirect 177→11177 on the X-server host, or use a chooser that accepts host:port.")
            .clicked()
        {
            add = Some(PortForwardConfig { proto: ForwardProto::Udp, host_port: 11177, guest_port: 177, bind: ForwardBind::Any });
            ui.close_menu();
        }
        if ui.button("Custom (empty row)").clicked() {
            add = Some(PortForwardConfig { proto: ForwardProto::Tcp, host_port: 0, guest_port: 0, bind: ForwardBind::Localhost });
            ui.close_menu();
        }
    });
    if let Some(pf) = add { cfg.port_forward.push(pf); out.changed = true; out.forwards_changed = true; }

    ui.label(RichText::new(
        "A port forward maps a port on your computer to a port on the Indy, so host tools can reach \
         guest services (log in, copy files, and so on). Inbound only, and it works once the guest is \
         up on the NAT subnet. None exist by default.")
        .weak());
    } // end: NAT-only section (hidden in PCAP mode)

    ui.separator();
    ui.strong("NFS share");
    ui.label(RichText::new(
        "The Indy speaks NFS natively — the easiest way to move files between your computer and the \
         emulated machine. IRIS serves NFS itself, in-process, backed by the folder you pick below: \
         nothing to install on any platform, and no NFS setup on the Indy side.")
        .weak());
    let mut has_nfs = cfg.nfs.is_some();
    if ui.checkbox(&mut has_nfs, "Enable NFS").changed() {
        cfg.nfs = if has_nfs {
            Some(NfsConfig { shared_dir: String::new(), version: NfsVersion::Auto })
        } else { None };
        out.changed = true;
    }
    // PCAP mode: the in-core NFS server is presented as its own virtual L2 host
    // on the bridged LAN (see `NfsVirtualHost`), so it needs its own IP on the
    // guest's subnet. NAT mode mounts from the gateway and doesn't use this. The
    // change takes effect on the next Start (the virtual host is built then).
    if cfg.nfs.is_some() && cfg.network.mode == NetMode::Pcap {
        out.changed |= pcap_nfs_ip_row(ui, cfg, host);
    }
    if let Some(nfs) = cfg.nfs.as_mut() {
        Grid::new("nfs_grid").num_columns(2).striped(true).show(ui, |ui| {
            ui.label("Shared dir");
            out.changed |= path_row(ui, "nfs_shared", &mut nfs.shared_dir, Pick::Dir, ANY_FILTERS).changed;
            ui.end_row();
            ui.label("NFS version");
            ComboBox::from_id_salt("nfs_ver")
                .selected_text(match nfs.version {
                    NfsVersion::Auto => "Auto",
                    NfsVersion::V2 => "v2 (IRIX 5.3)",
                    NfsVersion::V3 => "v3 (IRIX 6.x)",
                })
                .show_ui(ui, |ui| {
                    out.changed |= ui.selectable_value(&mut nfs.version, NfsVersion::Auto, "Auto").changed();
                    out.changed |= ui.selectable_value(&mut nfs.version, NfsVersion::V2, "v2 (IRIX 5.3)").changed();
                    out.changed |= ui.selectable_value(&mut nfs.version, NfsVersion::V3, "v3 (IRIX 6.x)").changed();
                });
            ui.end_row();
        });

        // App Store: the in-core NFS server can only reach a folder the sandbox
        // has granted. Easiest path — put the share inside a granted disk folder,
        // whose recursive grant flows down to it (you can also pick any folder
        // above; the folder picker grants that one directly).
        if cfg!(feature = "appstore") {
            ui.add_space(4.0);
            if disk_folders.is_empty() {
                ui.label(RichText::new(
                    "On the App Store build the shared folder must live somewhere the app has been \
                     granted. Grant a disk folder first (File » \"Grant a disk folder…\"), then create \
                     a shared folder inside it here — or pick any folder above to grant it directly.")
                    .weak());
            } else {
                ui.label(RichText::new("Or create a \"shared\" folder inside a granted disk folder:").weak());
                for folder in disk_folders {
                    let shared = std::path::Path::new(folder).join("shared");
                    if ui.button(format!("Use {}", shared.display())).clicked()
                        && std::fs::create_dir_all(&shared).is_ok()
                    {
                        nfs.shared_dir = shared.to_string_lossy().into_owned();
                        out.changed = true;
                    }
                }
            }
        }

        // Live mount command — the server IP fills in to match the backend. The
        // export is the single root, so the path is just "/". NAT: the gateway IP
        // of the configured subnet. PCAP: the in-process NFS server's own virtual
        // LAN IP (the guest is bridged, so there's no NAT gateway to mount from).
        let gw = if cfg.network.mode == NetMode::Pcap {
            cfg.network.nfs_pcap_ip
                .map(|ip| ip.to_string())
                .unwrap_or_else(|| "<NFS IP>".into())
        } else {
            let (b, p) = netplan::parse_cidr(cfg.nat_subnet.as_deref());
            netplan::classify(b, p, host)
                .derived
                .map(|d| d.gateway.to_string())
                .unwrap_or_else(|| "192.168.0.1".into())
        };
        ui.label(RichText::new("Pick a folder, boot the Indy, then mount it:").weak());
        ui.code(format!("mkdir /shared\nmount {gw}:/ /shared"));
        ui.label(RichText::new("Your files then appear at /shared on the Indy.").weak());
    }

    out
}

/// PCAP mode: edit the virtual NFS host's IPv4 address (`cfg.network.nfs_pcap_ip`).
/// The server answers ARP + portmap/NFS/mountd UDP for this single address on the
/// bridged LAN, so it only needs an IP on the guest's subnet — no netmask or
/// gateway (it never routes). Returns true if the value changed this frame.
fn pcap_nfs_ip_row(ui: &mut Ui, cfg: &mut MachineConfig, host: &[crate::netplan::HostIface]) -> bool {
    #[cfg(not(feature = "pcap"))]
    let _ = host;

    let buf_id  = ui.make_persistent_id("nfs_pcap_ip_buf");
    let last_id = ui.make_persistent_id("nfs_pcap_ip_last");

    let cur = cfg.network.nfs_pcap_ip.map(|i| i.to_string()).unwrap_or_default();
    let mut text: String = ui.data_mut(|d| d.get_temp::<String>(buf_id)).unwrap_or_else(|| cur.clone());
    // Re-sync the buffer if the stored IP changed outside this code (machine switch).
    let last: String = ui.data_mut(|d| d.get_temp::<String>(last_id)).unwrap_or_default();
    if cur != last {
        text = cur.clone();
    }

    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("NFS server IP");
        let resp = ui.add(TextEdit::singleline(&mut text)
            .hint_text("e.g. 192.168.1.213").desired_width(130.0));
        if resp.changed() {
            cfg.network.nfs_pcap_ip = text.trim().parse::<std::net::Ipv4Addr>().ok();
            changed = true;
        }
        // One-click suggestion: a free address high in the host LAN's host range.
        #[cfg(feature = "pcap")]
        if let Some(ip) = suggest_nfs_ip(cfg, host) {
            if ui.button(format!("Use {ip}"))
                .on_hover_text("A free address on your LAN's subnet")
                .clicked()
            {
                text = ip.to_string();
                cfg.network.nfs_pcap_ip = Some(ip);
                changed = true;
            }
        }
    });
    ui.label(RichText::new(
        "Give the NFS server a free address on the same subnet your Indy uses (your real LAN). \
         Then on the Indy: mount it from this IP. No gateway needed — it's reachable directly.")
        .weak());

    ui.data_mut(|d| {
        d.insert_temp(buf_id, text);
        d.insert_temp(last_id, cfg.network.nfs_pcap_ip.map(|i| i.to_string()).unwrap_or_default());
    });
    changed
}

/// Suggest a free IPv4 for the PCAP NFS host: take the subnet of the selected
/// capture interface (else the first host interface), reserve the host's own
/// addresses, and pick the first [`nfs_ip_candidates`] entry.
///
/// [`nfs_ip_candidates`]: iris::net_pcap::nfs_ip_candidates
#[cfg(feature = "pcap")]
fn suggest_nfs_ip(cfg: &MachineConfig, host: &[crate::netplan::HostIface]) -> Option<std::net::Ipv4Addr> {
    let want = cfg.network.pcap_interface.as_deref().filter(|s| !s.is_empty());
    let iface = want
        .and_then(|n| host.iter().find(|h| h.name == n))
        .or_else(|| host.first())?;
    let reserved: Vec<_> = host.iter().map(|h| h.addr).collect();
    iris::net_pcap::nfs_ip_candidates(iface.network, iface.prefix, &reserved)
        .into_iter()
        .next()
}

/// PCAP interface picker: a dropdown of enumerated host interfaces (with an
/// "Auto-pick" entry and a "Manual…" escape hatch), plus a Refresh button.
/// Stores the choice by interface *name* in `cfg.network.pcap_interface`
/// (`None` = auto-pick). Returns `RefreshPcapIfaces` when the user asks to
/// re-enumerate.
/// Returns `(action, committed)` — `committed` is true when the interface was
/// deliberately changed (a dropdown pick, or the manual field losing focus), the
/// cue to reopen a running PcapEngine's capture. Per-keystroke text edits don't
/// commit, so typing "bridge100" doesn't thrash through `b`, `br`, …
fn pcap_interface_picker(
    ui: &mut Ui,
    cfg: &mut MachineConfig,
    pcap_ifaces: &Option<Result<Vec<PcapIface>, String>>,
) -> (ConfigAction, bool) {
    let mut action = ConfigAction::None;
    let mut committed = false;

    // Selected text for the combo: the current name, "Auto-pick", or the raw
    // value if it's something not in the list (e.g. an index or manual name).
    let current = cfg.network.pcap_interface.clone();
    let selected_text = match &current {
        None => "Auto-pick (first up, non-loopback)".to_string(),
        Some(v) if v.is_empty() => "Auto-pick (first up, non-loopback)".to_string(),
        Some(v) => v.clone(),
    };

    ui.horizontal(|ui| {
        ui.label("PCAP interface");

        ComboBox::from_id_salt("pcap_iface")
            .selected_text(selected_text)
            .width(320.0)
            .show_ui(ui, |ui| {
                // Auto-pick entry.
                let mut is_auto = current.as_deref().unwrap_or("").is_empty();
                if ui.selectable_label(is_auto, "Auto-pick (first up, non-loopback)").clicked() {
                    cfg.network.pcap_interface = None;
                    is_auto = true;
                    committed = true;
                }
                let _ = is_auto;

                match pcap_ifaces {
                    Some(Ok(list)) if !list.is_empty() => {
                        ui.separator();
                        for iface in list {
                            let selected = current.as_deref() == Some(iface.name.as_str());
                            if ui.selectable_label(selected, iface.summary()).clicked() {
                                cfg.network.pcap_interface = Some(iface.name.clone());
                                committed = true;
                            }
                        }
                    }
                    Some(Ok(_)) => {
                        ui.separator();
                        ui.label(RichText::new("(no interfaces enumerated)").weak());
                    }
                    Some(Err(e)) => {
                        ui.separator();
                        ui.label(RichText::new(format!("(cannot list: {e})")).weak());
                    }
                    None => {
                        ui.separator();
                        ui.label(RichText::new("(click Refresh to enumerate)").weak());
                    }
                }
            });

        if ui.button("⟳ Refresh").on_hover_text("Re-enumerate host interfaces").clicked() {
            action = ConfigAction::RefreshPcapIfaces;
        }
    });

    // Manual entry escape hatch: lets the user type an index ("1"), an exact
    // name, or a Windows \Device\NPF_{...} string the dropdown can't show well.
    ui.horizontal(|ui| {
        ui.label("   or type index/name");
        let mut manual = current.clone().unwrap_or_default();
        let resp = ui.add(TextEdit::singleline(&mut manual)
            .hint_text("e.g. 1, eth0, or blank = auto")
            .desired_width(260.0));
        if resp.changed() {
            cfg.network.pcap_interface = if manual.trim().is_empty() { None } else { Some(manual) };
        }
        // Commit (reopen the live capture) only when the field loses focus, so a
        // running reswap doesn't fire on every keystroke.
        if resp.lost_focus() {
            committed = true;
        }
    });

    // Show an inline error if enumeration failed.
    if let Some(Err(e)) = pcap_ifaces {
        ui.colored_label(Color32::from_rgb(0xe0, 0x60, 0x60), format!("Interface list unavailable: {e}"));
    }

    (action, committed)
}

fn show_vino(ui: &mut Ui, cfg: &mut MachineConfig) -> ConfigAction {
    let mut action = ConfigAction::None;
    ui.heading("Video-In (IndyCam)");
    Grid::new("vino_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("Source");
        ComboBox::from_id_salt("vino_src")
            .selected_text(match cfg.vino.source {
                VinoSource::Camera      => "camera",
                VinoSource::TestPattern => "test_pattern",
                VinoSource::Black       => "black",
                VinoSource::Off         => "off (disabled)",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut cfg.vino.source, VinoSource::Off, "off (disabled)");
                ui.selectable_value(&mut cfg.vino.source, VinoSource::TestPattern, "test_pattern");
                let camera_label = if build_features::CAMERA {
                    "camera"
                } else {
                    "camera (needs --features camera)"
                };
                ui.selectable_value(&mut cfg.vino.source, VinoSource::Camera, camera_label);
                ui.selectable_value(&mut cfg.vino.source, VinoSource::Black, "black");
            });
        ui.end_row();

        ui.label("Standard");
        ComboBox::from_id_salt("vino_std")
            .selected_text(match cfg.vino.standard { VinoStandard::Ntsc => "ntsc", VinoStandard::Pal => "pal" })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut cfg.vino.standard, VinoStandard::Ntsc, "ntsc");
                ui.selectable_value(&mut cfg.vino.standard, VinoStandard::Pal, "pal");
            });
        ui.end_row();

        ui.label("Camera index");
        ui.add(DragValue::new(&mut cfg.vino.camera_index).range(0..=15));
        ui.end_row();
    });

    // Live host-camera test: opens the selected camera directly (no IRIX boot
    // needed) so the user can confirm the capture path works — and, on macOS,
    // grant the camera permission. This exercises the same host-capture code
    // the VINO/IndyCam source uses.
    ui.add_space(8.0);
    if build_features::CAMERA {
        ui.horizontal(|ui| {
            if ui.button("📷 Test Camera").clicked() {
                action = ConfigAction::TestCamera;
            }
            ui.label(
                RichText::new(format!(
                    "Preview host camera #{} live ({}).",
                    cfg.vino.camera_index,
                    match cfg.vino.standard { VinoStandard::Ntsc => "NTSC", VinoStandard::Pal => "PAL" },
                ))
                .weak(),
            );
        });
        ui.label(
            RichText::new(
                "On first use macOS will ask for camera permission. The camera \
                 is released when you close the preview.",
            )
            .weak()
            .small(),
        );
    } else {
        ui.label(
            RichText::new("Camera test unavailable — this build was compiled without --features camera.")
                .weak(),
        );
    }

    action
}

fn show_debug(ui: &mut Ui, cfg: &mut MachineConfig, jit: &mut JitEnv) {
    ui.heading("Debug / JIT");
    if build_features::LIGHTNING {
        ui.label(RichText::new(
            "⚡ Lightning build — interactive debugging is disabled \
             (no breakpoints, no GDB stub, no traceback). Rebuild without \
             --features lightning to re-enable.").color(Color32::from_rgb(220, 170, 90)));
        ui.separator();
    } else {
        Grid::new("dbg_grid").num_columns(2).striped(true).show(ui, |ui| {
            ui.label("GDB stub port");
            let mut port = cfg.gdb_port.unwrap_or(0);
            if ui.add(DragValue::new(&mut port).range(0..=65535)).changed() {
                cfg.gdb_port = if port == 0 { None } else { Some(port) };
            }
            ui.end_row();
        });
    }
    ui.separator();
    ui.label("JIT (requires `cargo build --features jit`)");
    Grid::new("jit_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("Enable JIT (IRIS_JIT=1)");
        ui.checkbox(&mut jit.iris_jit, "");
        ui.end_row();

        ui.label("Max tier (0=ALU, 1=Loads, 2=Full)");
        let mut t = jit.max_tier.unwrap_or(2);
        if ui.add(DragValue::new(&mut t).range(0..=2)).changed() {
            jit.max_tier = Some(t);
        }
        ui.end_row();

        ui.label("Verify against interpreter");
        ui.checkbox(&mut jit.verify, "");
        ui.end_row();

        ui.label("Disable JIT stores (diagnostic)");
        ui.checkbox(&mut jit.no_stores, "");
        ui.end_row();

        ui.label("Probe interval");
        ui.add(TextEdit::singleline(&mut jit.probe).hint_text("default 200").desired_width(120.0));
        ui.end_row();

        ui.label("Trace file");
        path_row(ui, "jit_trace", &mut jit.trace_file, Pick::SaveFile, ANY_FILTERS);
        ui.end_row();

        ui.label("Profile file");
        path_row(ui, "jit_profile", &mut jit.profile_file, Pick::SaveFile, ANY_FILTERS);
        ui.end_row();
    });
    ui.separator();
    Grid::new("misc_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("Disable idle park (IRIS_NO_IDLE)");
        ui.checkbox(&mut jit.no_idle, "");
        ui.end_row();
        ui.label("Devlog spec (IRIS_DEBUG_LOG)");
        ui.add(TextEdit::singleline(&mut jit.debug_log).hint_text("all, or e.g. mc,mips").desired_width(280.0));
        ui.end_row();
    });
}

fn show_ci(ui: &mut Ui, cfg: &mut MachineConfig) {
    ui.heading("CI / Automation");
    Grid::new("ci_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("Enable CI mode");
        ui.checkbox(&mut cfg.ci, "");
        ui.end_row();
        ui.label("CI socket path");
        path_row(ui, "ci_socket", &mut cfg.ci_socket, Pick::SaveFile, SOCKET_FILTERS);
        ui.end_row();
        ui.label("Keep window visible (--ci-display)");
        ui.checkbox(&mut cfg.ci_display, "");
        ui.end_row();
    });
}

/// Serialize `cfg` back to TOML string in the same style as iris.toml.
pub fn cfg_to_toml(cfg: &MachineConfig) -> Result<String, String> {
    toml::to_string_pretty(cfg).map_err(|e| e.to_string())
}

/// How a Browse button should pick a path.
#[derive(Clone, Copy)]
enum Pick {
    OpenFile,
    SaveFile,
    Dir,
}

/// Reveal `path` in the host file manager, selecting it (Finder on macOS,
/// Explorer on Windows, the default manager on the containing dir elsewhere).
/// Best-effort — failures (e.g. a sandbox blocking the spawn) are ignored.
pub fn reveal_in_file_manager(path: &str) {
    #[cfg(target_os = "macos")]
    {
        // NSWorkspace, not `open` — sandbox-safe (see macos_sandbox::reveal_in_finder).
        crate::macos_sandbox::reveal_in_finder(path);
    }
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("explorer").arg(format!("/select,{path}")).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let dir = Path::new(path).parent().unwrap_or_else(|| Path::new("."));
        let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
    }
}

/// Outcome of a [`path_row`]: whether `value` changed this frame (typed text or
/// a Browse pick — both must persist), and whether the change specifically came
/// from the Browse *picker*. The latter is the "assignment" moment — the only
/// safe time to (re)check folder-grant permissions, since reacting to every
/// keystroke would pop a dialog mid-typing.
#[derive(Default, Clone, Copy)]
struct PathEdit {
    changed: bool,
    picked: bool,
}

/// A TextEdit + 📁 Browse button that updates `value` in place. See [`PathEdit`].
fn path_row(
    ui: &mut Ui,
    id: impl std::hash::Hash,
    value: &mut String,
    mode: Pick,
    filters: &[(&str, &[&str])],
) -> PathEdit {
    let mut out = PathEdit::default();
    ui.push_id(id, |ui| {
        ui.horizontal(|ui| {
            out.changed |= ui.add(TextEdit::singleline(value).desired_width(320.0)).changed();
            if ui.button("📁").on_hover_text("Browse…").clicked() {
                let mut d = rfd::FileDialog::new();
                // Start the dialog in the existing path's directory if any.
                if !value.is_empty() {
                    let p = Path::new(value);
                    if let Some(parent) = p.parent() {
                        if parent.as_os_str().len() > 0 && parent.exists() {
                            d = d.set_directory(parent);
                        }
                    }
                    if let Some(name) = p.file_name() {
                        d = d.set_file_name(name.to_string_lossy());
                    }
                }
                if matches!(mode, Pick::OpenFile | Pick::SaveFile) {
                    for (label, exts) in filters {
                        d = d.add_filter(*label, exts);
                    }
                }
                let picked = match mode {
                    Pick::OpenFile => d.pick_file(),
                    Pick::SaveFile => d.save_file(),
                    Pick::Dir      => d.pick_folder(),
                };
                if let Some(p) = picked {
                    *value = p.to_string_lossy().into_owned();
                    out.changed = true;
                    out.picked = true;
                }
            }
            // Reveal an existing path in the host file manager (Finder, Explorer,
            // …) so the user can find/open it without navigating by hand.
            if !value.trim().is_empty() && Path::new(value.trim()).exists()
                && ui.button("📂").on_hover_text("Reveal in file manager").clicked()
            {
                reveal_in_file_manager(value.trim());
            }
        });
    });
    out
}

/// Same as `path_row` but for `Option<String>` — Browse populates Some,
/// the user can clear by emptying the text.
fn path_row_opt(
    ui: &mut Ui,
    id: impl std::hash::Hash,
    value: &mut Option<String>,
    mode: Pick,
    filters: &[(&str, &[&str])],
) {
    let mut s = value.clone().unwrap_or_default();
    path_row(ui, id, &mut s, mode, filters);
    *value = if s.is_empty() { None } else { Some(s) };
}

/// Common file filters.
const PROM_FILTERS:   &[(&str, &[&str])] = &[("PROM image", &["bin"]), ("All", &["*"])];
const NVRAM_FILTERS:  &[(&str, &[&str])] = &[("NVRAM",      &["bin"]), ("All", &["*"])];
const DISK_FILTERS:   &[(&str, &[&str])] = &[
    ("Disk images", &["raw", "img", "chd"]),
    ("ISO images",  &["iso"]),
    ("All",         &["*"]),
];
const ANY_FILTERS:    &[(&str, &[&str])] = &[("All", &["*"])];
const SOCKET_FILTERS: &[(&str, &[&str])] = &[("Unix socket", &["sock"]), ("All", &["*"])];

