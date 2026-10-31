use egui::{Color32, ComboBox, DragValue, Grid, RichText, ScrollArea, TextEdit, Ui};
use iris::build_features;
use std::path::Path;
use iris::config::{AudioHost, CacheModel,
    format_unix_utc, CpuModel, ForwardBind, ForwardProto, GraphicsBoard, KeyboardLayout,
    MachineConfig, MachineProfile, NetMode, NfsConfig, PortForwardConfig, RtcOffset,
    ScsiDeviceConfig, ScsiKind, VinoSource, VinoStandard, VALID_BANK_SIZES,
};
use iris::net::nfsudp::NfsVersion;
use iris::dev::ng1::vc2_timings::NewportResolution;

use crate::i18n::{subtitle, ResponseExt};
use crate::ram::{ram_summary, ram_presets};

/// Memory-tab context: whether the VM is running and what banks were last started with.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryUiContext {
    pub running: bool,
    pub started_banks: Option<[u32; iris::config::MAX_BANKS]>,
    /// The CPU the running guest actually booted on, so a pending change is
    /// shown as pending rather than as fact. `None` when nothing is running.
    pub started_cpu: Option<CpuModel>,
}

/// A host network interface candidate for the PCAP backend selector. This is a
/// GUI-local, feature-independent copy of `iris::net::net_pcap::NetInterface` so the
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
        if self.up { tags.push(t!("config-pcap-tag-up")); }
        if self.running { tags.push(t!("config-pcap-tag-running")); }
        if self.loopback { tags.push(t!("config-pcap-tag-loopback")); }
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
        iris::net::net_pcap::list_interfaces().map(|list| {
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
        Err(t!("config-pcap-not-built"))
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
    Bench,
}

impl Tab {
    /// Tabs to show for the active build. The Debug tab is hidden in
    /// lightning builds (the debug paths it drives are compiled out), and
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
        // The benchmark runs entirely inside this process — the guest image is
        // linked in and the emulated machine is one this app builds itself — so
        // unlike the CI tab it works in a sandbox and ships to everyone. Its
        // developer half (the matrix runner, which builds one emulator per
        // cell) is hidden inside the tab instead.
        tabs.push(Tab::Bench);
        tabs
    }
    pub fn label(self) -> String {
        match self {
            Tab::General => t!("config-tab-general"),
            Tab::Disks   => t!("config-tab-disks"),
            Tab::Network => t!("config-tab-network"),
            Tab::Memory  => t!("config-tab-memory"),
            Tab::Display => t!("config-tab-display"),
            Tab::VideoIn => t!("config-tab-video-in"),
            Tab::Debug   => t!("config-tab-debug"),
            Tab::Ci      => t!("config-tab-ci"),
            Tab::Bench   => t!("config-tab-bench"),
        }
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
    /// profile's built-in PROM loader in `iris::dev::prom::Prom`).
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
    LoadDisc { id: u8, path: String, remount: bool },
    /// Spawn headless iris with the machine TOML and ci=true, then iris-ci ping.
    TestCi,
    /// Spawn ensure-build.bat for CLI or GUI premiere profile.
    RebuildProfile { gui: bool },
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
    host: &[crate::netplan::HostIface],
    disk_folders: &[String],
    pcap_ifaces: &Option<Result<Vec<PcapIface>, String>>,
    mem_ctx: MemoryUiContext,
    bench: &mut crate::bench_ui::BenchState,
) -> TabOutcome {
    ScrollArea::vertical().show(ui, |ui| match tab {
        Tab::General => TabOutcome { action: show_general(ui, cfg, mem_ctx), ..Default::default() },
        Tab::Disks   => { let (e, a) = show_disks(ui, cfg); TabOutcome { action: a, disks_changed: e.changed, disk_picked: e.picked, ..Default::default() } }
        Tab::Network => {
            let net = show_network(ui, cfg, host, disk_folders, pcap_ifaces, mem_ctx.running);
            TabOutcome { action: net.action.clone(), net, ..Default::default() }
        }
        Tab::Memory  => { show_memory(ui, cfg, mem_ctx); TabOutcome::default() }
        Tab::Display => { show_display(ui, cfg, mem_ctx.running); TabOutcome::default() }
        Tab::VideoIn => TabOutcome { action: show_vino(ui, cfg), ..Default::default() },
        Tab::Debug   => TabOutcome { action: show_debug(ui, cfg), ..Default::default() },
        Tab::Ci      => TabOutcome { action: show_ci(ui, cfg), ..Default::default() },
        Tab::Bench   => { crate::bench_ui::show(ui, bench, mem_ctx.running); TabOutcome::default() }
    }).inner
}

fn show_general(ui: &mut Ui, cfg: &mut MachineConfig, mem_ctx: MemoryUiContext) -> ConfigAction {
    let mut action = ConfigAction::None;
    ui.heading(t!("config-tab-general"));
    ui.label(t!("config-platform", platform = cfg.machine.profile.label()));
    let profile_before = cfg.machine.profile;
    ComboBox::from_id_salt("machine_profile")
        .selected_text(cfg.machine.profile.label())
        .show_ui(ui, |ui| {
            for p in MachineProfile::ALL {
                ui.selectable_value(&mut cfg.machine.profile, p, p.label());
            }
        });
    if cfg.machine.profile == MachineProfile::OctaneIp30 && profile_before != MachineProfile::OctaneIp30 {
        octane_defaults(cfg);
    }
    let octane = cfg.machine.profile == MachineProfile::OctaneIp30;
    // A board the new machine does not take would fail `validate()` at
    // Start: GR2 Extreme moving to Indy keeps the closest board (XZ); any
    // other falls back to the machine's default.
    let board = cfg.graphics_board();
    if !board.supports(cfg.machine.profile) {
        cfg.graphics.board = (board == GraphicsBoard::Extreme && !octane).then_some(GraphicsBoard::Xz);
    }
    // Only IP28 can express 256/512 MB banks. Clamp them when switching to
    // IP22/IP24 so the new profile passes `validate()` at Start.
    if !matches!(cfg.machine.profile, MachineProfile::Indigo2Ip28 | MachineProfile::OctaneIp30) {
        for bank in &mut cfg.banks {
            if *bank > 128 { *bank = 128; }
        }
    }
    subtitle(ui, t!("config-platform-subtitle"));
    subtitle(ui, t!("config-viewport-subtitle"));
    if cfg.machine.profile == MachineProfile::Indigo2Ip22 {
        subtitle(ui, t!("config-indigo2-subtitle"));
    }
    if octane {
        subtitle(ui, t!("config-octane-subtitle"));
    }
    if cfg.machine.profile == MachineProfile::Indigo2Ip28 {
        subtitle(ui, t!("config-ip28-subtitle"));
        ui.label(
            RichText::new(t!("config-ip28-graphics-warning"))
            .color(Color32::from_rgb(220, 170, 90))
            .small(),
        );
    }
        ui.heading(t!("config-processor"));
    Grid::new("cpu_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label(t!("config-cpu"));
        // A real setting, not a read-out of how this binary was built. Both
        // CPU models are compiled in and `Machine::new` picks between them at
        // construction, so the choice belongs to the machine's config the same
        // way its RAM does.
        // The Octane has one CPU option, the 195 MHz R10000.
        ui.add_enabled_ui(!mem_ctx.running && !octane, |ui| {
            ComboBox::from_id_salt("cpu_model")
                .selected_text(cfg.machine.cpu.label())
                .show_ui(ui, |ui| {
                    for c in CpuModel::ALL {
                        ui.selectable_value(&mut cfg.machine.cpu, c, c.label());
                    }
                });
        });
        ui.end_row();

        // `[machine] cache`. The R10000 only has the shadow model, so the
        // choice is fixed there.
        ui.label(t!("config-cache-model"));
        let r10k = cfg.machine.cpu == CpuModel::R10000;
        ui.add_enabled_ui(!mem_ctx.running && !r10k, |ui| {
            let shown = if r10k { t!("config-cache-shadow-only") } else { cache_model_label(cfg.machine.cache, cfg.machine.cpu) };
            ComboBox::from_id_salt("cache_model")
                .selected_text(shown)
                .show_ui(ui, |ui| {
                    for m in [CacheModel::Auto, CacheModel::Full, CacheModel::Shadow] {
                        ui.selectable_value(&mut cfg.machine.cache, m, cache_model_label(m, cfg.machine.cpu));
                    }
                })
                .response
                .tip(t!("config-cache-model-tip"));
        });
        ui.end_row();

        ui.label(t!("config-clock"));
        ui.horizontal(|ui| {
            // CP0 Count: half the CPU clock on the R10000 machines (the Octane's is 195 MHz).
            let auto_default = if cfg.machine.profile.ip28() || octane { 97.5 } else { 33.0 };
            let mut mhz = cfg.clock.fixed_mhz.unwrap_or(auto_default);
            let changed = ui.add(DragValue::new(&mut mhz).range(0.1..=1000.0).suffix(" MHz")).changed();
            if changed { cfg.clock.fixed_mhz = Some(mhz); }
            if ui.add_enabled(cfg.clock.fixed_mhz.is_some(), egui::Button::new(t!("common-auto")))
                .tip(t!("config-clock-auto-tip", mhz = auto_default))
                .clicked()
            {
                cfg.clock.fixed_mhz = None;
            }
        });
        ui.end_row();
    });

    if mem_ctx.running {
        ui.label(
            RichText::new(t!("config-cpu-locked"))
                .color(Color32::from_rgb(220, 170, 90)),
        );
        if let Some(started) = mem_ctx.started_cpu {
            if started != cfg.machine.cpu {
                ui.label(t!("config-pending", running = started.label(), pending = cfg.machine.cpu.label()));
            }
        }
    } else {
        subtitle(ui, t!("config-cpu-applies-subtitle"));
    }
    subtitle(ui, t!("config-cpu-subtitle"));
    if cfg.machine.cpu == CpuModel::R10000 {
        subtitle(ui, t!("config-r10000-subtitle"));
    }
    // jitv2 persistent code cache: lives here (not the Debug tab) so it's
    // reachable from lightning builds too, which hide Debug entirely
    // (build_features::LIGHTNING check in Tab::visible) even though jitv2
    // itself is an independent feature from lightning.
    if build_features::JITV2 {
        ui.add_space(4.0);
        Grid::new("jitv2_cache_grid").num_columns(2).striped(true).show(ui, |ui| {
            ui.label(t!("config-jit-threads"));
            let max = std::thread::available_parallelism().map_or(8, |n| n.get()).max(1);
            ui.add(DragValue::new(&mut cfg.jitv2.threads).range(1..=max))
                .tip(t!("config-jit-threads-tip"));
            ui.end_row();
            ui.label(t!("config-jit-cache"));
            ui.checkbox(&mut cfg.jitv2.cache, "")
                .tip(t!("config-jit-cache-tip"));
            ui.end_row();
            ui.label(t!("config-jit-cache-dir"));
            ui.add(TextEdit::singleline(&mut cfg.jitv2.cache_dir)
                .hint_text(t!("config-jit-cache-dir-hint"))
                .desired_width(280.0));
            ui.end_row();
        });
    }
    ui.separator();

    if octane {
        show_octane_graphics_picker(ui, cfg, mem_ctx.running);
    } else {
        show_board_picker(ui, cfg, mem_ctx.running);
    }
    if !octane && cfg.graphics_board() == GraphicsBoard::Newport {
        ui.horizontal(|ui| {
            ui.label(t!("config-newport-heads"));
            ui.add(egui::DragValue::new(&mut cfg.graphics.heads).range(1..=2).speed(0.1));
            if cfg.graphics.heads == 2 {
                subtitle(ui, t!("config-newport-dual-subtitle"));
            }
        });
    }
    if !octane {
        show_resolution_picker(ui, cfg, false);
    }
    ui.add_space(4.0);
    Grid::new("general_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label(t!("config-prom"));
        path_row(ui, "prom", &mut cfg.prom, Pick::OpenFile, &prom_filters());
        ui.end_row();

        // Leaving the PROM path empty boots the built-in PROM. Expose that as
        // an explicit button so reverting from a (possibly missing) custom PROM
        // is discoverable instead of "delete the text by hand". Disabled when
        // already empty, so the confirm prompt only ever appears when a custom
        // PROM is selected.
        // The Octane has no built-in PROM: it needs the IP30 PROM image.
        if !octane {
        ui.label("");
        ui.horizontal(|ui| {
            let custom = !cfg.prom.is_empty();
            if ui.add_enabled(custom, egui::Button::new(t!("config-prom-builtin")))
                .tip(t!("config-prom-builtin-tip"))
                .clicked()
            {
                action = ConfigAction::RequestEmbeddedProm;
            }
            if !custom {
                ui.label(RichText::new(t!("config-prom-builtin-active")).weak());
            }
        });
        ui.end_row();
        }

        ui.label(t!("config-nvram"));
        path_row(ui, "nvram", &mut cfg.nvram, Pick::SaveFile, &nvram_filters());
        ui.end_row();

        if octane {
            // `[octane] flash`: the boot flash the PROM writes its records to.
            ui.label(t!("config-octane-flash")).tip(t!("config-octane-flash-tip"));
            let mut flash = cfg.octane.flash.clone().unwrap_or_default();
            path_row(ui, "octane_flash", &mut flash, Pick::SaveFile, &nvram_filters());
            cfg.octane.flash = if flash.is_empty() { None } else { Some(flash) };
            ui.end_row();
        } else {
            ui.label(t!("config-nveeprom"))
                .tip(t!("config-nveeprom-tip"));
            path_row(ui, "nveeprom", &mut cfg.nveeprom, Pick::SaveFile, &nvram_filters());
            ui.end_row();
        }

        ui.label(t!("config-serial-log"));
        path_row_opt(ui, "serial_log", &mut cfg.serial_log, Pick::SaveFile, &any_filters());
        ui.end_row();
    });

    show_host_ports(ui, cfg);

    show_rtc_offset(ui, &mut cfg.rtc_offset, mem_ctx.running);

    // N64 development board (Ultra64). A single runtime toggle — the GIO device
    // and POSIX shm bridge (/iris_n64_bridge) are only created when this is on,
    // read once at VM start. The toggle is never shown in App Store builds:
    // the sandbox can't open the named shm or run the external gopher64
    // process it talks to.
    #[cfg(not(feature = "appstore"))]
    {

    ui.separator();
        ui.checkbox(&mut cfg.ultra64.enabled, t!("config-ultra64"))
            .tip(t!("config-ultra64-tip"));
        subtitle(ui, t!("config-autosave-subtitle"));
    }

    action
}

/// The monitor console's loopback port, default filled in.
pub fn monitor_port(cfg: &MachineConfig) -> u16 {
    cfg.monitor_port.unwrap_or(iris::monitor::DEFAULT_PORT)
}

/// `monitor_port`, `serial_port_a`, `serial_port_b`: the loopback TCP ports of
/// the monitor console and the two serial channels. Two machines running at
/// once (this app and a CLI `iris`, say) need different ones. Read at Start.
fn show_host_ports(ui: &mut Ui, cfg: &mut MachineConfig) {
    ui.separator();
    ui.heading(t!("config-host-ports"));
    let [def_a, def_b] = iris::dev::z85c30::DEFAULT_PORTS;
    Grid::new("host_ports_grid").num_columns(2).striped(true).show(ui, |ui| {
        port_row(ui, t!("config-port-monitor"), t!("config-port-monitor-tip"),
                 &mut cfg.monitor_port, iris::monitor::DEFAULT_PORT);
        port_row(ui, t!("config-port-serial1"), t!("config-port-serial1-tip"),
                 &mut cfg.serial_port_b, def_b);
        port_row(ui, t!("config-port-serial2"), t!("config-port-serial2-tip"), &mut cfg.serial_port_a, def_a);
    });
    let mut ports = vec![monitor_port(cfg), cfg.serial_ports()[0], cfg.serial_ports()[1]];
    if let Some(g) = cfg.gdb_port { ports.push(g); }
    let mut sorted = ports.clone();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != ports.len() {
        ui.label(RichText::new(t!("config-ports-conflict"))
            .color(Color32::from_rgb(0xd9, 0x9a, 0x3d)).small());
    }
    subtitle(ui, t!("config-host-ports-subtitle"));
}

/// One loopback port setting, where `None` means `default`.
fn port_row(ui: &mut Ui, label: String, hover: String, value: &mut Option<u16>, default: u16) {
    ui.label(label).tip(hover);
    ui.horizontal(|ui| {
        let mut port = value.unwrap_or(default);
        if ui.add(DragValue::new(&mut port).range(1..=65535)).changed() {
            *value = (port != default).then_some(port);
        }
        if ui.add_enabled(value.is_some(), egui::Button::new(t!("common-default")))
            .tip(t!("config-port-default-tip", port = default.to_string()))
            .clicked()
        {
            *value = None;
        }
    });
    ui.end_row();
}

/// A cache model's menu label; Auto says what it resolves to in this build.
fn cache_model_label(m: CacheModel, cpu: CpuModel) -> String {
    match m {
        CacheModel::Auto if m.is_shadow(cpu) => t!("config-cache-auto-shadow"),
        CacheModel::Auto => t!("config-cache-auto-full"),
        CacheModel::Full => t!("config-cache-full"),
        CacheModel::Shadow => t!("config-cache-shadow"),
    }
}

/// `[rtc_offset]`: six signed fields, shown and typed as `-n` / `+n`, with a
/// live preview of where the guest clock would start right now.
fn show_rtc_offset(ui: &mut Ui, off: &mut RtcOffset, running: bool) {
    ui.separator();
    ui.heading(t!("config-rtc"));
    subtitle(ui, t!("config-rtc-subtitle"));

    fn signed(ui: &mut Ui, v: &mut i64) {
        ui.add(
            DragValue::new(v)
                .speed(0.1)
                .custom_formatter(|n, _| format!("{:+}", n as i64))
                .custom_parser(|s| s.trim().parse::<i64>().ok().map(|n| n as f64)),
        );
    }
    Grid::new("rtc_offset_grid").num_columns(6).show(ui, |ui| {
        ui.label(t!("config-rtc-years"));   signed(ui, &mut off.years);
        ui.label(t!("config-rtc-months"));  signed(ui, &mut off.months);
        ui.label(t!("config-rtc-days"));    signed(ui, &mut off.days);
        ui.end_row();
        ui.label(t!("config-rtc-hours"));   signed(ui, &mut off.hours);
        ui.label(t!("config-rtc-minutes")); signed(ui, &mut off.minutes);
        ui.label(t!("config-rtc-seconds")); signed(ui, &mut off.seconds);
        ui.end_row();
    });

    let host = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (guest, clamped) = off.apply_clamped(host);
    ui.horizontal(|ui| {
        ui.label(t!("config-rtc-preview", time = format_unix_utc(guest)));
        if ui.add_enabled(!off.is_zero(), egui::Button::new(t!("config-rtc-reset"))).clicked() {
            *off = RtcOffset::default();
        }
    });
    if clamped {
        ui.label(
            RichText::new(t!("config-rtc-clamped"))
                .color(Color32::from_rgb(220, 170, 90)),
        );
    }
    subtitle(ui, if running {
        t!("config-rtc-running-subtitle")
    } else {
        t!("config-rtc-applies-subtitle")
    });
}

fn show_board_picker(ui: &mut Ui, cfg: &mut MachineConfig, running: bool) {
    let before = cfg.graphics_board();
    let mut choice = before;
    ui.horizontal(|ui| {
        ui.label(t!("config-graphics-board"));
        ui.add_enabled_ui(!running, |ui| {
            ComboBox::from_id_salt("graphics_board")
                .selected_text(choice.label())
                .show_ui(ui, |ui| {
                    for c in GraphicsBoard::ALL {
                        ui.add_enabled_ui(c.supports(cfg.machine.profile), |ui| {
                            ui.selectable_value(&mut choice, c, c.label())
                                .on_disabled_hover_text(t!("config-graphics-indigo2-only"));
                        });
                    }
                });
        });
    });
    // Every non-Newport choice is single-head with no VC2 presets;
    // clear those so the config still passes `validate()`.
    if choice != before {
        cfg.graphics.board = Some(choice);
        if choice != GraphicsBoard::Newport {
            cfg.graphics.heads = 1;
            cfg.graphics.resolution = NewportResolution::Guest;
        }
    }
    if running {
        ui.label(
            RichText::new(t!("config-graphics-locked"))
                .color(Color32::from_rgb(220, 170, 90))
                .small(),
        );
    }
    if matches!(choice, GraphicsBoard::Xz | GraphicsBoard::Extreme) {
        subtitle(ui, t!("config-graphics-gr2-subtitle"));
    }
    if choice.is_impact() {
        subtitle(ui, if cfg.machine.profile == MachineProfile::Indigo2Ip28 {
            t!("config-graphics-impact-ip28-subtitle")
        } else {
            t!("config-graphics-impact-subtitle")
        });
    }
}

/// The Octane's graphics port holds an IMPACT board (Solid, High as the SI
/// with texture option, Maximum as the MXI) or nothing (`[graphics] board
/// = "none"`); the other boards are not modelled there.
fn show_octane_graphics_picker(ui: &mut Ui, cfg: &mut MachineConfig, running: bool) {
    let fitted = cfg.graphics_board();
    let mut choice = fitted;
    let name = |b: GraphicsBoard| if b == GraphicsBoard::None { t!("config-octane-graphics-none") } else { b.label().to_string() };
    ui.horizontal(|ui| {
        ui.label(t!("config-graphics-board"));
        ui.add_enabled_ui(!running, |ui| {
            ComboBox::from_id_salt("octane_graphics")
                .selected_text(name(choice))
                .show_ui(ui, |ui| {
                    for b in [GraphicsBoard::SolidImpact, GraphicsBoard::HighImpact, GraphicsBoard::MaxImpact] {
                        ui.selectable_value(&mut choice, b, b.label());
                    }
                    ui.selectable_value(&mut choice, GraphicsBoard::None, t!("config-octane-graphics-none"));
                });
        });
    });
    if choice != fitted {
        cfg.graphics.board = Some(choice);
        cfg.graphics.heads = 1;
        cfg.graphics.resolution = NewportResolution::Guest;
    }
    if running {
        ui.label(
            RichText::new(t!("config-graphics-locked"))
                .color(Color32::from_rgb(220, 170, 90))
                .small(),
        );
    }
    subtitle(ui, t!("config-octane-graphics-subtitle"));
}

/// What switching a machine to the Octane changes so it passes `validate()`:
/// the R10000, a valid bank layout (bank 0 filled, 64 MB to 2 GB, no gaps)
/// and a graphics board it models.
fn octane_defaults(cfg: &mut MachineConfig) {
    cfg.machine.cpu = CpuModel::R10000;
    let sizes = MachineProfile::OctaneIp30.bank_sizes();
    let banks = &cfg.banks[..MachineProfile::OctaneIp30.memory_banks()];
    let filled = banks.iter().take_while(|&&b| b != 0).count();
    let valid = filled > 0 && banks[filled..].iter().all(|&b| b == 0) && banks.iter().all(|b| sizes.contains(b));
    if !valid {
        let total: u32 = banks.iter().sum();
        cfg.banks = crate::dialogs::new_machine::distribute_ram(total.clamp(64, 8192), MachineProfile::OctaneIp30);
    }
    if !cfg.graphics_board().supports(MachineProfile::OctaneIp30) {
        cfg.graphics.board = None;
    }
    cfg.graphics.heads = 1;
    cfg.graphics.resolution = NewportResolution::Guest;
    cfg.octane.graphics = None;
}

fn show_resolution_picker(ui: &mut Ui, cfg: &mut MachineConfig, running: bool) {
    let newport = cfg.graphics_board() == GraphicsBoard::Newport && !cfg.headless;
    ui.horizontal(|ui| {
        let label = ui.label(t!("config-resolution"));
        if !newport {
            label.tip(t!("config-resolution-newport-only-tip"));
            subtitle(ui, t!("config-resolution-newport-only-subtitle"));
            return;
        }
        ComboBox::from_id_salt("graphics_resolution")
            .selected_text(cfg.graphics.resolution.label())
            .show_ui(ui, |ui| {
                for mode in NewportResolution::ALL {
                    ui.selectable_value(&mut cfg.graphics.resolution, mode, mode.label());
                }
            });
    });
    if newport {
        subtitle(ui, t!("config-resolution-subtitle"));
        if running {
            ui.label(
                RichText::new(t!("config-resolution-locked"))
                    .color(Color32::from_rgb(220, 170, 90))
                    .small(),
            );
        }
    }
}

fn show_memory(ui: &mut Ui, cfg: &mut MachineConfig, mem_ctx: MemoryUiContext) {
    ui.heading(t!("config-tab-memory"));
    if mem_ctx.running {
        ui.label(
            RichText::new(t!("config-ram-locked"))
                .color(Color32::from_rgb(220, 170, 90)),
        );
        if let Some(started) = mem_ctx.started_banks {
            if started != cfg.banks {
                ui.label(t!("config-pending",
                    running = ram_summary(&started),
                    pending = ram_summary(&cfg.banks),
                ));
            } else {
                ui.label(t!("config-ram-running", ram = ram_summary(&started)));
            }
        }
    } else {
        subtitle(ui, t!("config-ram-applies-subtitle"));
        subtitle(ui, t!("config-ram-subtitle"));
    }
    let profile = cfg.machine.profile;
    ui.add_space(4.0);
    ui.label(t!("config-ram-presets"));
    ui.horizontal_wrapped(|ui| {
        for &p in ram_presets(profile) {
            if ui
                .add_enabled(!mem_ctx.running, egui::Button::new(t!("config-megabytes", size = p)))
                .on_disabled_hover_text(t!("config-ram-locked"))
                .clicked()
            {
                cfg.banks = crate::dialogs::new_machine::distribute_ram(p, profile);
            }
        }
    });
    ui.label(t!("config-ram-banks"));
    subtitle(ui, if profile == MachineProfile::OctaneIp30 {
        t!("config-ram-banks-octane-subtitle")
    } else if profile == MachineProfile::Indigo2Ip28 {
        t!("config-ram-banks-ip28-subtitle")
    } else {
        t!("config-ram-banks-subtitle")
    });
    Grid::new("mem_grid").num_columns(2).striped(true).show(ui, |ui| {
        for i in 0..4 {
            ui.label(t!("config-ram-bank", bank = i));
            let cur = cfg.banks[i];
            ui.add_enabled_ui(!mem_ctx.running, |ui| {
                ComboBox::from_id_salt(("bank", i)).selected_text(t!("config-megabytes", size = cur))
                    .show_ui(ui, |ui| {
                        for &sz in VALID_BANK_SIZES {
                            ui.add_enabled_ui(profile.bank_sizes().contains(&sz), |ui| {
                                ui.selectable_value(&mut cfg.banks[i], sz, t!("config-megabytes", size = sz))
                                    .on_disabled_hover_text(crate::ram::bank_size_tip(profile));
                            });
                        }
                    });
            });
            ui.end_row();
        }
    });
    ui.label(t!("config-ram-total", ram = ram_summary(&cfg.banks)));
}

fn show_display(ui: &mut Ui, cfg: &mut MachineConfig, running: bool) {
    ui.heading(t!("config-tab-display"));
    if cfg.machine.profile != MachineProfile::OctaneIp30 {
        show_resolution_picker(ui, cfg, running);
    }
    ui.add_space(6.0);
    Grid::new("disp_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label(t!("config-scale"));
        ComboBox::from_id_salt("scale").selected_text(t!("config-scale-factor", scale = cfg.scale))
            .show_ui(ui, |ui| {
                for s in 1u32..=4 {
                    ui.selectable_value(&mut cfg.scale, s, t!("config-scale-factor", scale = s));
                }
            });
        ui.end_row();

        ui.label(t!("config-headless"));
        ui.checkbox(&mut cfg.headless, "");
        ui.end_row();

        ui.label(t!("config-no-audio"));
        ui.checkbox(&mut cfg.no_audio, "");
        ui.end_row();

        ui.label(t!("config-keyboard"))
            .tip(t!("config-keyboard-tip"));
        ui.add_enabled_ui(!running, |ui| {
            ComboBox::from_id_salt("keyboard_layout").selected_text(cfg.keyboard.label())
                .show_ui(ui, |ui| {
                    for k in KeyboardLayout::ALL {
                        ui.selectable_value(&mut cfg.keyboard, k, k.label());
                    }
                });
        });
        ui.end_row();
    });
    if !cfg.no_audio {
        ui.separator();
        ui.heading(t!("config-audio"));
        subtitle(ui, t!("config-audio-subtitle"));
        Grid::new("audio_grid").num_columns(2).striped(true).show(ui, |ui| {
            if cfg!(target_os = "linux") {
                ui.label(t!("config-audio-host"));
                let label = |h: AudioHost| match h {
                    AudioHost::Default => t!("config-audio-host-default"),
                    AudioHost::Alsa => t!("config-audio-host-alsa"),
                    AudioHost::Pulseaudio => t!("config-audio-host-pulse"),
                };
                ComboBox::from_id_salt("audio_host")
                    .selected_text(label(cfg.audio.host))
                    .show_ui(ui, |ui| {
                        for h in [AudioHost::Default, AudioHost::Alsa, AudioHost::Pulseaudio] {
                            ui.selectable_value(&mut cfg.audio.host, h, label(h));
                        }
                    });
                ui.end_row();
            }
            ui.label(t!("config-audio-prebuffer"));
            ui.add(DragValue::new(&mut cfg.audio.prebuf_ms).range(5..=200));
            ui.end_row();
            ui.label(t!("config-audio-host-buffer"));
            let mut frames = cfg.audio.cpal_buffer_frames.unwrap_or(0);
            if ui.add(DragValue::new(&mut frames).range(0..=8192)).changed() {
                cfg.audio.cpal_buffer_frames = if frames == 0 { None } else { Some(frames) };
            }
            ui.end_row();
        });
        subtitle(ui, t!("config-audio-rate-subtitle"));
    }
}

fn show_disks(ui: &mut Ui, cfg: &mut MachineConfig) -> (PathEdit, ConfigAction) {
    let mut edit = PathEdit::default();
    let mut action = ConfigAction::None;
    ui.heading(t!("config-scsi"));
    ui.horizontal(|ui| {
        subtitle(ui, t!("config-scsi-subtitle"));
    });
    let mut to_delete: Option<u8> = None;
    for id in 1u8..=7 {
        ui.separator();
        let exists = cfg.scsi.contains_key(&id);
        ui.horizontal(|ui| {
            ui.strong(t!("config-scsi-id", id = id));
            if exists {
                if ui.button(t!("config-remove")).clicked() {
                    to_delete = Some(id);
                }
            } else if ui.button(t!("config-scsi-attach")).clicked() {
                cfg.scsi.insert(id, ScsiDeviceConfig {
                    path: crate::settings::GuiSettings::default_disk_path(id),
                    ..Default::default()
                });
            }
        });
        if let Some(dev) = cfg.scsi.get_mut(&id) {
            // A DaynaPort has no image, no media and no overlay — it is an
            // Ethernet adapter on the SCSI bus. Show its own short form
            // instead of the storage rows below.
            if dev.is_daynaport() {
                Grid::new(("scsi_grid", id)).num_columns(2).striped(true).show(ui, |ui| {
                    ui.label(t!("config-scsi-type"));
                    scsi_type_combo(ui, id, dev, &mut edit);
                    ui.end_row();

                    ui.label(t!("config-scsi-mac"))
                        .tip(t!("config-scsi-mac-tip"));
                    let mut mac = dev.mac.clone().unwrap_or_default();
                    if ui.add(TextEdit::singleline(&mut mac).hint_text("00:80:19:44:50:03")).changed() {
                        dev.mac = if mac.trim().is_empty() { None } else { Some(mac) };
                        edit.changed = true;
                    }
                    ui.end_row();

                    ui.label(t!("config-scsi-subnet"))
                        .tip(t!("config-scsi-subnet-tip"));
                    let mut subnet = dev.subnet.clone().unwrap_or_default();
                    if ui.add(TextEdit::singleline(&mut subnet).hint_text("192.168.10.0/24")).changed() {
                        dev.subnet = if subnet.trim().is_empty() { None } else { Some(subnet) };
                        edit.changed = true;
                    }
                    ui.end_row();
                });
                subtitle(ui, t!("config-daynaport-subtitle"));
                continue;
            }
            Grid::new(("scsi_grid", id)).num_columns(2).striped(true).show(ui, |ui| {
                ui.label(t!("config-scsi-image"));
                let e = path_row_disk(ui, ("scsi_path", id), &mut dev.path,
                    if dev.scratch { Pick::SaveFile } else { Pick::OpenFile },
                    &disk_filters());
                edit.changed |= e.changed;
                edit.picked |= e.picked;
                // For CD-ROMs, picking a path immediately loads the disc into
                // the running machine (equivalent to Ctrl+F12). The core's
                // count-driven queue decides whether this replaces the current
                // disc or joins the changer cycle.
                if dev.cdrom && e.picked && !dev.path.is_empty() {
                    action = ConfigAction::LoadDisc {
                        id, path: dev.path.clone(), remount: true,
                    };
                }
                ui.end_row();
                // Active copy-on-write overlay for a compressed CHD: show exactly
                // which `.diff.chd` is in use (the path honours IRIS_CHD_DIFF_DIR,
                // so on the sandbox build this is the container sidecar) and its
                // size, so it's unambiguous that changes are landing here and that
                // this is the file folded back into the disk on a clean exit.
                if dev.path.ends_with(".chd") {
                    let diff = iris::chd_disk::diff_path_for(Path::new(&dev.path));
                    if let Ok(meta) = std::fs::metadata(&diff) {
                        let mb = meta.len() as f64 / (1024.0 * 1024.0);
                        ui.label(t!("config-scsi-overlay"));
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(t!("config-scsi-overlay-file",
                                    path = diff.display(), size = format!("{mb:.1}")))
                                .weak().small())
                                .tip(t!("config-scsi-overlay-tip"));
                            if ui.small_button("📂").tip(t!("config-reveal-tip")).clicked() {
                                reveal_in_file_manager(&diff.to_string_lossy());
                            }
                        });
                        ui.end_row();
                    }
                }

                ui.label(t!("config-scsi-type"));
                let was_cd = dev.cdrom;
                scsi_type_combo(ui, id, dev, &mut edit);
                // Switching to CD-ROM defaults to an empty drive (no media):
                // clear the auto-generated HDD placeholder path so it doesn't
                // look like a (missing) disc. Load media via "Insert disc…" in
                // the SCSI menu, or just type a path here.
                if dev.cdrom && !was_cd && (dev.path == crate::settings::GuiSettings::default_disk_path(id) || dev.path == format!("scsi{id}.raw")) {
                    dev.path.clear();
                }
                ui.end_row();
                if dev.cdrom && dev.path.is_empty() {
                    ui.label("");
                    ui.label(RichText::new(t!("config-scsi-cdrom-empty"))
                        .weak().small());
                    ui.end_row();
                }

                ui.label(t!("config-scsi-cow"))
                    .tip(t!("config-scsi-cow-tip"));
                ui.checkbox(&mut dev.overlay, "")
                    .tip(t!("config-scsi-cow-check-tip"));
                ui.end_row();

                ui.label(t!("config-scsi-scratch"));
                ui.checkbox(&mut dev.scratch, "");
                ui.end_row();

                if dev.scratch {
                    ui.label(t!("config-scsi-scratch-size"));
                    let mut sz = dev.size_mb.unwrap_or(64);
                    if ui.add(DragValue::new(&mut sz).range(1..=8192)).changed() {
                        dev.size_mb = Some(sz);
                    }
                    ui.end_row();
                }
            });

            if dev.cdrom {
                ui.label(t!("config-scsi-changer"));
                let mut drop_idx: Option<usize> = None;
                for (i, disc) in dev.discs.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        let e = path_row_disk(ui, ("disc", id, i), disc, Pick::OpenFile, &disk_filters());
                        edit.changed |= e.changed;
                        edit.picked |= e.picked;
                        if ui.button("×").tip(t!("config-scsi-disc-remove-tip")).clicked() { drop_idx = Some(i); }
                    });
                }
                if let Some(i) = drop_idx { dev.discs.remove(i); }
                if ui.button(t!("config-scsi-add-disc")).clicked() {
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
    running: bool,
) -> NetworkOutcome {
    use crate::netplan;
    let mut out = NetworkOutcome::default();
    ui.heading(t!("config-tab-network"));

    // The Octane's address is not in its NVRAM; it comes from `[network] mac`
    // or a generated default.
    if cfg.machine.profile != MachineProfile::OctaneIp30 {
        ethernet_address_row(ui, cfg, running);
    }
    ui.add_space(4.0);

    // The backend selector (and the entire PCAP UI) is only shown when this
    // build actually has PCAP support. App Store / bundled builds compile
    // without `--features pcap`, where NAT is the only backend — so PCAP must
    // not appear anywhere in the UI (a dangling, non-functional option risks an
    // App Store rejection). Such builds also force NAT at runtime regardless of
    // a stale `mode = "pcap"` carried in from an imported config.
    if build_features::PCAP {
        Grid::new("net_mode_grid").num_columns(2).striped(true).show(ui, |ui| {
            ui.label(t!("config-net-mode"));
            ComboBox::from_id_salt("net_mode")
                .selected_text(match cfg.network.mode {
                    NetMode::Nat  => t!("config-net-mode-nat"),
                    NetMode::Pcap => t!("config-net-mode-pcap"),
                    // Only an O2 config can say this, and the GUI does not run one yet.
                    NetMode::Off  => "off".to_string(),
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut cfg.network.mode, NetMode::Nat, t!("config-net-mode-nat"));
                    ui.selectable_value(&mut cfg.network.mode, NetMode::Pcap, t!("config-net-mode-pcap"));
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

            ui.colored_label(Color32::from_rgb(0xd0, 0xa0, 0x40), t!("config-net-pcap-note"));

            // Explicit, OS-specific way to grant capture permission up front (the
            // other trigger is automatic: the app pops the same prompt if a
            // pcap-mode machine hits a permission error on Start). The hint text
            // and the action are platform-specific — see `capture_access`.
            ui.horizontal(|ui| {
                if ui.button(t!("config-net-pcap-enable"))
                    .tip(t!("config-net-pcap-enable-tip"))
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
    subtitle(ui, t!("config-net-nat-subtitle"));
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
        ui.label(t!("config-net-network"));
        ui.horizontal(|ui| {
            let sel = if net_custom { t!("config-net-custom") } else { base.to_string() };
            ComboBox::from_id_salt("net_preset").selected_text(sel).show_ui(ui, |ui| {
                for addr in fa {
                    if ui.selectable_label(!net_custom && base == addr, addr.to_string()).clicked() {
                        base = addr; net_custom = false; changed = true; committed = true;
                    }
                }
                if ui.selectable_label(net_custom, t!("config-net-custom")).clicked() {
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

        ui.label(t!("config-net-mask"));
        let sel = if mask_custom { t!("config-net-mask-custom", prefix = prefix) } else { netplan::mask_label(prefix) };
        ComboBox::from_id_salt("net_mask").selected_text(sel).show_ui(ui, |ui| {
            for &p in netplan::MASK_PRESETS {
                if ui.selectable_label(!mask_custom && prefix == p, netplan::mask_label(p)).clicked() {
                    prefix = p; mask_custom = false; changed = true; committed = true;
                }
            }
            if ui.selectable_label(mask_custom, t!("config-net-custom")).clicked() {
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
        ui.label(RichText::new(t!("config-net-invalid", reason = msg)).color(Color32::from_rgb(0xd9, 0x4a, 0x3d)));
    } else if let Some(d) = &assess.derived {
        ui.label(RichText::new(t!("config-net-derived",
            gateway = d.gateway.to_string(),
            guest = d.client.to_string(),
            hosts = netplan::commas(d.usable_hosts),
            broadcast = d.broadcast.to_string(),
        )).weak());
        if let Some(typed) = assess.off_boundary {
            ui.label(RichText::new(t!("config-net-snapped",
                typed = typed.to_string(),
                network = d.network.to_string(),
                prefix = d.prefix,
            )).weak());
        }
        match &assess.soft {
            Some(w) => {
                ui.horizontal(|ui| {
                    let sug = netplan::to_cidr(w.suggestion_net, w.suggestion_prefix);
                    ui.label(RichText::new(t!("config-net-warning", reason = &w.reason))
                        .color(Color32::from_rgb(0xd9, 0x9a, 0x3d)));
                    if ui.button(t!("config-use", value = &sug)).clicked() {
                        cfg.nat_subnet = Some(sug); // external-change re-sync repaints the controls
                        out.changed = true;
                    }
                });
            }
            None => {
                ui.label(RichText::new(t!("config-net-ok"))
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
    ui.strong(t!("config-forwards"));
    let mut drop: Option<usize> = None;
    for (i, pf) in cfg.port_forward.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            let mut c = false;
            ComboBox::from_id_salt(("proto", i))
                .selected_text(match pf.proto { ForwardProto::Tcp => t!("config-forward-tcp"), ForwardProto::Udp => t!("config-forward-udp") })
                .show_ui(ui, |ui| {
                    c |= ui.selectable_value(&mut pf.proto, ForwardProto::Tcp, t!("config-forward-tcp")).changed();
                    c |= ui.selectable_value(&mut pf.proto, ForwardProto::Udp, t!("config-forward-udp")).changed();
                });
            ui.label(t!("config-forward-host"));
            c |= ui.add(DragValue::new(&mut pf.host_port).range(1..=65535)).changed();
            ui.label(t!("config-forward-guest"));
            c |= ui.add(DragValue::new(&mut pf.guest_port).range(1..=65535)).changed();
            ComboBox::from_id_salt(("bind", i))
                .selected_text(match pf.bind { ForwardBind::Localhost => t!("config-forward-bind-local"), ForwardBind::Any => t!("config-forward-bind-any") })
                .show_ui(ui, |ui| {
                    c |= ui.selectable_value(&mut pf.bind, ForwardBind::Localhost, t!("config-forward-bind-local")).changed();
                    c |= ui.selectable_value(&mut pf.bind, ForwardBind::Any, t!("config-forward-bind-any")).changed();
                });
            if ui.button(t!("config-remove")).clicked() { drop = Some(i); }
            out.changed |= c;
            out.forwards_changed |= c;
        });
    }
    if let Some(i) = drop { cfg.port_forward.remove(i); out.changed = true; out.forwards_changed = true; }

    let has_port = |p: u16| cfg.port_forward.iter().any(|f| f.guest_port == p);
    let mut add: Option<PortForwardConfig> = None;
    ui.menu_button(t!("config-forward-add"), |ui| {
        if ui.add_enabled(!has_port(23), egui::Button::new(t!("config-forward-telnet")))
            .tip(t!("config-forward-telnet-tip"))
            .clicked()
        {
            add = Some(PortForwardConfig { proto: ForwardProto::Tcp, host_port: 2323, guest_port: 23, bind: ForwardBind::Localhost });
            ui.close();
        }
        if ui.add_enabled(!has_port(21), egui::Button::new(t!("config-forward-ftp")))
            .tip(t!("config-forward-ftp-tip"))
            .clicked()
        {
            add = Some(PortForwardConfig { proto: ForwardProto::Tcp, host_port: 2121, guest_port: 21, bind: ForwardBind::Localhost });
            ui.close();
        }
        if ui.add_enabled(!has_port(22), egui::Button::new(t!("config-forward-ssh")))
            .tip(t!("config-forward-ssh-tip"))
            .clicked()
        {
            add = Some(PortForwardConfig { proto: ForwardProto::Tcp, host_port: 2222, guest_port: 22, bind: ForwardBind::Localhost });
            ui.close();
        }
        if ui.add_enabled(!has_port(5900), egui::Button::new(t!("config-forward-vnc")))
            .tip(t!("config-forward-vnc-tip"))
            .clicked()
        {
            add = Some(PortForwardConfig { proto: ForwardProto::Tcp, host_port: 5901, guest_port: 5900, bind: ForwardBind::Localhost });
            ui.close();
        }
        if ui.add_enabled(!has_port(177), egui::Button::new(t!("config-forward-xdmcp")))
            .tip(t!("config-forward-xdmcp-tip"))
            .clicked()
        {
            add = Some(PortForwardConfig { proto: ForwardProto::Udp, host_port: 11177, guest_port: 177, bind: ForwardBind::Any });
            ui.close();
        }
        if ui.button(t!("config-forward-custom")).clicked() {
            add = Some(PortForwardConfig { proto: ForwardProto::Tcp, host_port: 0, guest_port: 0, bind: ForwardBind::Localhost });
            ui.close();
        }
    });
    if let Some(pf) = add { cfg.port_forward.push(pf); out.changed = true; out.forwards_changed = true; }

    subtitle(ui, t!("config-forwards-subtitle"));

    ui.separator();
    out.changed |= tftp_section(ui, cfg, host);
    ui.separator();
    out.changed |= rsh_section(ui, cfg, host);
    } // end: NAT-only section (hidden in PCAP mode)

    ui.separator();
    ui.strong(t!("config-nfs"));
    subtitle(ui, t!("config-nfs-subtitle"));
    let mut has_nfs = cfg.nfs.is_some();
    if ui.checkbox(&mut has_nfs, t!("config-nfs-enable")).changed() {
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
            ui.label(t!("config-nfs-dir"));
            out.changed |= path_row(ui, "nfs_shared", &mut nfs.shared_dir, Pick::Dir, &any_filters()).changed;
            ui.end_row();
            ui.label(t!("config-nfs-version"));
            ComboBox::from_id_salt("nfs_ver")
                .selected_text(match nfs.version {
                    NfsVersion::Auto => t!("common-auto"),
                    NfsVersion::V2 => t!("config-nfs-v2"),
                    NfsVersion::V3 => t!("config-nfs-v3"),
                })
                .show_ui(ui, |ui| {
                    out.changed |= ui.selectable_value(&mut nfs.version, NfsVersion::Auto, t!("common-auto")).changed();
                    out.changed |= ui.selectable_value(&mut nfs.version, NfsVersion::V2, t!("config-nfs-v2")).changed();
                    out.changed |= ui.selectable_value(&mut nfs.version, NfsVersion::V3, t!("config-nfs-v3")).changed();
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
                ui.label(RichText::new(t!("config-nfs-appstore-grant")).weak());
            } else {
                ui.label(RichText::new(t!("config-nfs-appstore-shared")).weak());
                for folder in disk_folders {
                    let shared = std::path::Path::new(folder).join("shared");
                    if ui.button(t!("config-use", value = shared.display())).clicked()
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
                .unwrap_or_else(|| t!("config-nfs-ip-placeholder"))
        } else {
            let (b, p) = netplan::parse_cidr(cfg.nat_subnet.as_deref());
            netplan::classify(b, p, host)
                .derived
                .map(|d| d.gateway.to_string())
                .unwrap_or_else(|| "192.168.0.1".into())
        };
        ui.label(RichText::new(t!("config-nfs-mount")).weak());
        ui.code(format!("mkdir /shared\nmount {gw}:/ /shared"));
        subtitle(ui, t!("config-nfs-mount-subtitle"));
    }

    out
}

/// The guest's Ethernet address (`eaddr`), which IRIX needs to attach `ec0`.
/// It lives in the NVRAM on Indy and in the motherboard EEPROM (`nveeprom`) on
/// Indigo2/IP28 (clanker/irix/networking.md), so this reads and writes the chip
/// the profile boots from, the same bytes the Start pre-flight and Reset NVRAM
/// patch. Only editable while stopped: a running machine saves its NVRAM on
/// Stop, which would undo the edit.
fn ethernet_address_row(ui: &mut Ui, cfg: &MachineConfig, running: bool) {
    use crate::settings;
    let eeprom = !cfg.machine.profile.guinness();
    let path = if eeprom { cfg.nveeprom.as_str() } else { cfg.nvram.as_str() };
    let read = |p: &str| if eeprom { settings::nveeprom_mac(p) } else { settings::nvram_mac(p) };

    // Re-read the chip at most once a second: the PROM's `setenv eaddr` and a
    // Stop both rewrite it behind this tab's back.
    let cache_id = ui.make_persistent_id("eaddr_cache");
    let buf_id   = ui.make_persistent_id("eaddr_buf");
    let last_id  = ui.make_persistent_id("eaddr_last");
    let msg_id   = ui.make_persistent_id("eaddr_msg");
    let now = ui.input(|i| i.time);
    let cached: Option<(String, Option<[u8; 6]>, f64)> = ui.data_mut(|d| d.get_temp(cache_id));
    let current = match cached {
        Some((p, m, t)) if p == path && now - t < 1.0 => m,
        _ => {
            let m = read(path);
            ui.data_mut(|d| d.insert_temp(cache_id, (path.to_string(), m, now)));
            m
        }
    };
    let cur_text = current.map(settings::mac_to_string).unwrap_or_default();
    let mut buf: String = ui.data_mut(|d| d.get_temp(buf_id)).unwrap_or_else(|| cur_text.clone());
    let last: Option<String> = ui.data_mut(|d| d.get_temp(last_id));
    let mut msg: String = ui.data_mut(|d| d.get_temp(msg_id)).unwrap_or_default();
    if last.as_deref() != Some(cur_text.as_str()) {
        buf = cur_text.clone(); // the chip changed underneath us: show what it holds now
    }

    let mut commit: Option<[u8; 6]> = None;
    Grid::new("eaddr_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label(t!("config-eaddr"))
            .tip(if eeprom {
                t!("config-eaddr-eeprom-tip", path = path)
            } else {
                t!("config-eaddr-nvram-tip", path = path)
            });
        ui.add_enabled_ui(!running, |ui| {
            ui.horizontal(|ui| {
                let resp = ui.add(TextEdit::singleline(&mut buf)
                    .hint_text("08:00:69:xx:xx:xx").desired_width(150.0));
                let entered = resp.lost_focus() && buf != cur_text;
                if entered {
                    match iris::config::parse_mac(buf.trim()) {
                        Ok(m) => commit = Some(m),
                        Err(e) => msg = e,
                    }
                }
                if ui.button(t!("config-eaddr-generate")).tip(t!("config-eaddr-generate-tip")).clicked() {
                    commit = Some(settings::generate_mac_bytes(&format!("{path}{now}")));
                }
            });
        });
        ui.end_row();
    });

    if let Some(mac) = commit {
        let r = if eeprom {
            settings::ensure_nveeprom_exists(path);
            settings::write_nveeprom_mac(path, mac)
        } else {
            settings::ensure_nvram_seeded(path);
            settings::write_nvram_mac(path, mac)
        };
        msg = match r {
            Ok(true) => String::new(),
            Ok(false) => t!("config-eaddr-no-file", path = path),
            Err(e) => t!("config-eaddr-write-failed", path = path, error = e.to_string()),
        };
        buf = settings::mac_to_string(mac);
        ui.data_mut(|d| d.remove::<(String, Option<[u8; 6]>, f64)>(cache_id));
    }

    if running {
        ui.label(RichText::new(t!("config-eaddr-locked")).weak().small());
    } else if !msg.is_empty() {
        ui.label(RichText::new(&msg).color(Color32::from_rgb(0xd9, 0x4a, 0x3d)).small());
    } else if current.is_none() && commit.is_none() {
        ui.label(RichText::new(t!("config-eaddr-unset")).weak().small());
    }
    ui.data_mut(|d| {
        d.insert_temp(buf_id, buf);
        d.insert_temp(last_id, commit.map(settings::mac_to_string).unwrap_or(cur_text));
        d.insert_temp(msg_id, msg);
    });
}

/// NAT mode: the gateway's read-only TFTP server (`[network] tftp_dir`), for
/// PROM network boot (`boot -f bootp()/unix`) and guest tftp clients. Unset
/// turns the server off. Read at Start. Returns true if the setting changed.
fn tftp_section(ui: &mut Ui, cfg: &mut MachineConfig, host: &[crate::netplan::HostIface]) -> bool {
    let mut changed = false;
    ui.strong(t!("config-tftp"));
    subtitle(ui, t!("config-tftp-subtitle"));
    let mut on = cfg.network.tftp_dir.is_some();
    if ui.checkbox(&mut on, t!("config-tftp-enable")).changed() {
        cfg.network.tftp_dir = if on {
            // A bare relative name, like `nvram`/`disks` — it lives in this
            // machine's own folder (the working directory tracks whichever
            // machine is selected; see `switch_to`/`enter_directory`), not a
            // folder shared by every machine. Create it now so an enabled
            // server never serves an empty path.
            let dir = "tftpboot";
            let _ = std::fs::create_dir_all(dir);
            Some(dir.to_string())
        } else {
            None
        };
        changed = true;
    }
    if let Some(dir) = cfg.network.tftp_dir.as_mut() {
        Grid::new("tftp_grid").num_columns(2).striped(true).show(ui, |ui| {
            ui.label(t!("config-tftp-dir"));
            changed |= path_row(ui, "tftp_dir", dir, Pick::Dir, &any_filters()).changed;
            ui.end_row();
        });
        if dir.trim().is_empty() {
            ui.label(RichText::new(t!("config-tftp-dir-empty")).weak().small());
        } else if !Path::new(dir.trim()).is_dir() {
            ui.label(RichText::new(t!("config-tftp-dir-missing"))
                .color(Color32::from_rgb(0xd9, 0x9a, 0x3d)).small());
        }
        let (b, p) = crate::netplan::parse_cidr(cfg.nat_subnet.as_deref());
        let gw = crate::netplan::classify(b, p, host).derived
            .map(|d| d.gateway.to_string())
            .unwrap_or_else(|| "192.168.0.1".into());
        ui.label(RichText::new(t!("config-tftp-boot")).weak());
        ui.code("boot -f bootp()/unix");
        ui.label(RichText::new(t!("config-tftp-irix", gateway = &gw)).weak());
        ui.code(format!("tftp {gw}"));
    }
    // A blank field means off, not "serve the working directory".
    if cfg.network.tftp_dir.as_deref().is_some_and(|d| d.trim().is_empty()) && !ui.memory(|m| m.focused().is_some()) {
        cfg.network.tftp_dir = None;
        changed = true;
    }
    changed
}

/// NAT mode: the gateway's built-in read-only rsh server (`[network] rsh_dir`),
/// a network installation source for IRIX `inst`. Unset turns the server off
/// (rsh connections then go to the host). Read at Start. Returns true if the
/// setting changed.
fn rsh_section(ui: &mut Ui, cfg: &mut MachineConfig, host: &[crate::netplan::HostIface]) -> bool {
    let mut changed = false;
    ui.strong(t!("config-rsh"));
    subtitle(ui, t!("config-rsh-subtitle"));
    let mut on = cfg.network.rsh_dir.is_some();
    if ui.checkbox(&mut on, t!("config-rsh-enable")).changed() {
        cfg.network.rsh_dir = if on {
            // A bare relative name in this machine's own folder, like
            // `tftpboot`; created now so an enabled server never serves an
            // empty path.
            let dir = "rshroot";
            let _ = std::fs::create_dir_all(dir);
            Some(dir.to_string())
        } else {
            None
        };
        changed = true;
    }
    if let Some(dir) = cfg.network.rsh_dir.as_mut() {
        Grid::new("rsh_grid").num_columns(2).striped(true).show(ui, |ui| {
            ui.label(t!("config-rsh-dir"));
            changed |= path_row(ui, "rsh_dir", dir, Pick::Dir, &any_filters()).changed;
            ui.end_row();
        });
        if dir.trim().is_empty() {
            ui.label(RichText::new(t!("config-rsh-dir-empty")).weak().small());
        } else if !Path::new(dir.trim()).is_dir() {
            ui.label(RichText::new(t!("config-rsh-dir-missing"))
                .color(Color32::from_rgb(0xd9, 0x9a, 0x3d)).small());
        }
        let (b, p) = crate::netplan::parse_cidr(cfg.nat_subnet.as_deref());
        let gw = crate::netplan::classify(b, p, host).derived
            .map(|d| d.gateway.to_string())
            .unwrap_or_else(|| "192.168.0.1".into());
        ui.label(RichText::new(t!("config-rsh-inst")).weak());
        ui.code(format!("{gw}:/"));
        ui.label(RichText::new(t!("config-rsh-readonly")).weak().small());
    }
    // A blank field means off, not "serve the working directory".
    if cfg.network.rsh_dir.as_deref().is_some_and(|d| d.trim().is_empty()) && !ui.memory(|m| m.focused().is_some()) {
        cfg.network.rsh_dir = None;
        changed = true;
    }
    changed
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
        ui.label(t!("config-nfs-pcap-ip"));
        let resp = ui.add(TextEdit::singleline(&mut text)
            .hint_text(t!("config-nfs-pcap-ip-hint")).desired_width(130.0));
        if resp.changed() {
            cfg.network.nfs_pcap_ip = text.trim().parse::<std::net::Ipv4Addr>().ok();
            changed = true;
        }
        // One-click suggestion: a free address high in the host LAN's host range.
        #[cfg(feature = "pcap")]
        if let Some(ip) = suggest_nfs_ip(cfg, host) {
            if ui.button(t!("config-use", value = ip.to_string()))
                .tip(t!("config-nfs-pcap-suggest-tip"))
                .clicked()
            {
                text = ip.to_string();
                cfg.network.nfs_pcap_ip = Some(ip);
                changed = true;
            }
        }
    });
    subtitle(ui, t!("config-nfs-pcap-subtitle"));

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
/// [`nfs_ip_candidates`]: iris::net::net_pcap::nfs_ip_candidates
#[cfg(feature = "pcap")]
fn suggest_nfs_ip(cfg: &MachineConfig, host: &[crate::netplan::HostIface]) -> Option<std::net::Ipv4Addr> {
    let want = cfg.network.pcap_interface.as_deref().filter(|s| !s.is_empty());
    let iface = want
        .and_then(|n| host.iter().find(|h| h.name == n))
        .or_else(|| host.first())?;
    let reserved: Vec<_> = host.iter().map(|h| h.addr).collect();
    iris::net::net_pcap::nfs_ip_candidates(iface.network, iface.prefix, &reserved)
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
        None => t!("config-pcap-auto"),
        Some(v) if v.is_empty() => t!("config-pcap-auto"),
        Some(v) => v.clone(),
    };

    ui.horizontal(|ui| {
        ui.label(t!("config-pcap-iface"));

        ComboBox::from_id_salt("pcap_iface")
            .selected_text(selected_text)
            .width(320.0)
            .show_ui(ui, |ui| {
                // Auto-pick entry.
                let mut is_auto = current.as_deref().unwrap_or("").is_empty();
                if ui.selectable_label(is_auto, t!("config-pcap-auto")).clicked() {
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
                        ui.label(RichText::new(t!("config-pcap-none")).weak());
                    }
                    Some(Err(e)) => {
                        ui.separator();
                        ui.label(RichText::new(t!("config-pcap-list-error", error = e)).weak());
                    }
                    None => {
                        ui.separator();
                        ui.label(RichText::new(t!("config-pcap-not-listed")).weak());
                    }
                }
            });

        if ui.button(t!("common-refresh")).tip(t!("config-pcap-refresh-tip")).clicked() {
            action = ConfigAction::RefreshPcapIfaces;
        }
    });

    // Manual entry escape hatch: lets the user type an index ("1"), an exact
    // name, or a Windows \Device\NPF_{...} string the dropdown can't show well.
    ui.horizontal(|ui| {
        ui.label(t!("config-pcap-manual"));
        let mut manual = current.clone().unwrap_or_default();
        let resp = ui.add(TextEdit::singleline(&mut manual)
            .hint_text(t!("config-pcap-manual-hint"))
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
        ui.colored_label(Color32::from_rgb(0xe0, 0x60, 0x60), t!("config-pcap-list-error", error = e));
    }

    (action, committed)
}

fn show_vino(ui: &mut Ui, cfg: &mut MachineConfig) -> ConfigAction {
    let mut action = ConfigAction::None;
    ui.heading(t!("config-vino"));
    Grid::new("vino_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label(t!("config-vino-source"));
        ComboBox::from_id_salt("vino_src")
            .selected_text(match cfg.vino.source {
                VinoSource::Camera      => t!("config-vino-source-camera"),
                VinoSource::TestPattern => t!("config-vino-source-test"),
                VinoSource::Black       => t!("config-vino-source-black"),
                VinoSource::Off         => t!("config-vino-source-off"),
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut cfg.vino.source, VinoSource::Off, t!("config-vino-source-off"));
                ui.selectable_value(&mut cfg.vino.source, VinoSource::TestPattern, t!("config-vino-source-test"));
                ui.selectable_value(&mut cfg.vino.source, VinoSource::Camera, t!("config-vino-source-camera"));
                ui.selectable_value(&mut cfg.vino.source, VinoSource::Black, t!("config-vino-source-black"));
            });
        ui.end_row();

        ui.label(t!("config-vino-standard"));
        ComboBox::from_id_salt("vino_std")
            .selected_text(match cfg.vino.standard { VinoStandard::Ntsc => t!("config-vino-ntsc"), VinoStandard::Pal => t!("config-vino-pal") })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut cfg.vino.standard, VinoStandard::Ntsc, t!("config-vino-ntsc"));
                ui.selectable_value(&mut cfg.vino.standard, VinoStandard::Pal, t!("config-vino-pal"));
            });
        ui.end_row();

        ui.label(t!("config-vino-camera"));
        ui.add(DragValue::new(&mut cfg.vino.camera_index).range(0..=15));
        ui.end_row();
    });

    // Live host-camera test: opens the selected camera directly (no IRIX boot
    // needed) so the user can confirm the capture path works — and, on macOS,
    // grant the camera permission. This exercises the same host-capture code
    // the VINO/IndyCam source uses.
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        if ui.button(t!("config-vino-test")).clicked() {
            action = ConfigAction::TestCamera;
        }
        subtitle(ui, t!("config-vino-test-subtitle"));
    });
    subtitle(ui, t!("config-vino-permission-subtitle"));

    action
}

fn show_debug(ui: &mut Ui, cfg: &mut MachineConfig) -> ConfigAction {
    let mut action = ConfigAction::None;
    ui.heading(t!("config-build"));
    Grid::new("build_features_grid").num_columns(2).striped(true).show(ui, |ui| {
        // What this machine is configured to be, not how the binary was built —
        // all CPU models are always compiled in and selected at start.
        ui.label(t!("config-build-cpu"));
        ui.label(cfg.machine.cpu.label());
        ui.end_row();
        // `gpujit` = all three graphics JITs. The App Store build turns them
        // off at run time with IRIS_NO_JIT, so report what will actually run.
        let jit = |built: bool| if !built {
            t!("config-build-not-built")
        } else if std::env::var_os("IRIS_NO_JIT").is_some() {
            t!("config-build-jit-off")
        } else {
            t!("config-build-jit-on")
        };
        ui.label(t!("config-build-rex-jit"));
        ui.label(jit(build_features::REX_JIT));
        ui.end_row();
        ui.label(t!("config-build-gr2-jit"));
        ui.label(jit(build_features::GR2_JIT));
        ui.end_row();
        ui.label(t!("config-build-impact-jit"));
        ui.label(jit(build_features::GR4_JIT));
        ui.end_row();
        ui.label(t!("config-build-cpu-jit"));
        ui.label(jit(build_features::JITV2));
        ui.end_row();
        ui.label(t!("config-build-perfetto"));
        ui.label(if cfg!(feature = "perfetto") { t!("common-yes") } else { t!("config-build-perfetto-no") });
        ui.end_row();
        ui.label(t!("config-build-lightning"));
        ui.label(if build_features::LIGHTNING { t!("config-build-lightning-yes") } else { t!("common-no") });
        ui.end_row();
        ui.label(t!("config-build-idle-pause"));
        ui.label(if build_features::IDLE_PAUSE { t!("common-yes") } else { t!("common-no") });
        ui.end_row();
        ui.label(t!("config-build-pcap"));
        ui.label(if build_features::PCAP { t!("common-yes") } else { t!("common-no") });
        ui.end_row();
        ui.label(t!("config-build-hostgl"));
        ui.label(match (build_features::HOSTCALL, build_features::HOSTGL) {
            (_, true) => t!("config-build-hostgl-on"),
            (true, false) if cfg!(any(target_os = "macos", target_os = "linux")) =>
                t!("config-build-hostgl-hostcall-only"),
            (true, false) =>
                t!("config-build-hostgl-unsupported-os"),
            (false, _) => t!("config-build-hostgl-off"),
        });
        ui.end_row();
    });
    if build_features::HOSTCALL {
        subtitle(ui, t!("config-build-hostcall-subtitle"));
    }
    subtitle(ui, t!("config-build-mips-subtitle"));
    ui.separator();
    ui.heading(t!("config-tab-debug"));
    if build_features::LIGHTNING {
        ui.label(RichText::new(t!("config-debug-lightning")).color(Color32::from_rgb(220, 170, 90)));
        ui.separator();
    } else {
        Grid::new("dbg_grid").num_columns(2).striped(true).show(ui, |ui| {
            ui.label(t!("config-debug-gdb-port"));
            let mut port = cfg.gdb_port.unwrap_or(0);
            if ui.add(DragValue::new(&mut port).range(0..=65535)).changed() {
                cfg.gdb_port = if port == 0 { None } else { Some(port) };
            }
            ui.end_row();
        });
    }
    #[cfg(feature = "perfetto")]
    show_perfetto(ui);
    ui.separator();
    ui.label(t!("config-debug-capture"));
    Grid::new("capture_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label(t!("config-debug-gl-capture"));
        ui.checkbox(&mut cfg.debug.gui_gl_capture, "")
            .tip(t!("config-debug-gl-capture-tip"));
        ui.end_row();
    });
    subtitle(ui, t!("config-debug-audio-subtitle"));
    ui.separator();
    Grid::new("misc_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label(t!("config-debug-no-idle"));
        ui.checkbox(&mut cfg.debug.no_idle, "");
        ui.end_row();
        ui.label(t!("config-debug-log"));
        ui.add(TextEdit::singleline(&mut cfg.debug.debug_log).hint_text(t!("config-debug-log-hint")).desired_width(280.0));
        ui.end_row();
    });
    ui.separator();
    ui.label(RichText::new(t!("config-debug-profiles")).strong());
    ui.horizontal(|ui| {
        if ui.button(t!("config-debug-rebuild-cli")).tip(t!("config-debug-rebuild-cli-tip")).clicked() {
            action = ConfigAction::RebuildProfile { gui: false };
        }
        if ui.button(t!("config-debug-rebuild-gui")).tip(t!("config-debug-rebuild-gui-tip")).clicked() {
            action = ConfigAction::RebuildProfile { gui: true };
        }
    });
    ui.monospace("cargo build --release --bin iris --features lightning,rex-jit,idle-pause");
    ui.monospace("cargo build -p iris-gui --release --features premiere");
    ui.separator();
    ui.heading(t!("config-perf"));
    Grid::new("perf_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label(t!("config-perf-affinity"));
        ui.checkbox(&mut cfg.perf.thread_affinity, "");
        ui.end_row();
        ui.label(t!("config-perf-cpu-core"));
        let mut cpu = cfg.perf.cpu_core.unwrap_or(0) as i32;
        if ui.add(DragValue::new(&mut cpu).range(0..=63)).changed() {
            cfg.perf.cpu_core = Some(cpu as u32);
        }
        ui.end_row();
        ui.label(t!("config-perf-rex3-core"));
        let mut rex = cfg.perf.rex3_core.unwrap_or(1) as i32;
        if ui.add(DragValue::new(&mut rex).range(0..=63)).changed() {
            cfg.perf.rex3_core = Some(rex as u32);
        }
        ui.end_row();
        ui.label(t!("config-perf-refresh-core"));
        let mut refc = cfg.perf.refresh_core.unwrap_or(2) as i32;
        if ui.add(DragValue::new(&mut refc).range(0..=63)).changed() {
            cfg.perf.refresh_core = Some(refc as u32);
        }
        ui.end_row();
    });
    action
}

/// Perfetto trace recording (`iris::perfetto`), the GUI's `--perfetto` and
/// monitor `perfetto start|stop`. The session is process-wide, so it runs
/// across machine Start/Stop until stopped here or the app quits.
#[cfg(feature = "perfetto")]
fn show_perfetto(ui: &mut Ui) {
    use iris::perfetto;
    ui.separator();
    ui.heading(t!("config-perfetto"));
    let path_id  = ui.make_persistent_id("pf_path");
    let cats_id  = ui.make_persistent_id("pf_cats");
    let clock_id = ui.make_persistent_id("pf_guest_clock");
    let msg_id   = ui.make_persistent_id("pf_msg");
    let mut path: String = ui.data_mut(|d| d.get_temp(path_id)).unwrap_or_else(|| {
        crate::settings::GuiSettings::data_dir()
            .map(|d| d.join("iris.pftrace").to_string_lossy().into_owned())
            .unwrap_or_else(|| "iris.pftrace".into())
    });
    let mut cats: String = ui.data_mut(|d| d.get_temp(cats_id)).unwrap_or_else(|| "all".into());
    let mut guest_clock: bool = ui.data_mut(|d| d.get_temp(clock_id)).unwrap_or(false);
    let mut msg: String = ui.data_mut(|d| d.get_temp(msg_id)).unwrap_or_default();
    let status = perfetto::status();
    let recording = status != "perfetto: idle";

    ui.add_enabled_ui(!recording, |ui| {
        Grid::new("perfetto_grid").num_columns(2).striped(true).show(ui, |ui| {
            ui.label(t!("config-perfetto-file"));
            path_row(ui, "pf_path_row", &mut path, Pick::SaveFile, &perfetto_filters());
            ui.end_row();
            ui.label(t!("config-perfetto-categories"));
            ui.add(TextEdit::singleline(&mut cats).hint_text(t!("config-perfetto-categories-hint")).desired_width(280.0))
                .tip(t!("config-perfetto-categories-tip",
                    names = perfetto::cat::NAMES.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")));
            ui.end_row();
            ui.label(t!("config-perfetto-clock"));
            ComboBox::from_id_salt("pf_clock")
                .selected_text(if guest_clock { t!("config-perfetto-clock-guest") } else { t!("config-perfetto-clock-host") })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut guest_clock, false, t!("config-perfetto-clock-host"));
                    ui.selectable_value(&mut guest_clock, true, t!("config-perfetto-clock-guest"));
                })
                .response
                .tip(t!("config-perfetto-clock-tip"));
            ui.end_row();
        });
    });
    ui.horizontal(|ui| {
        if !recording && ui.button(t!("config-perfetto-start")).clicked() {
            let clock = if guest_clock { perfetto::Clock::Guest } else { perfetto::Clock::Host };
            msg = match perfetto::cat::parse(&cats).and_then(|c| perfetto::start(path.trim(), c, clock)) {
                Ok(()) => String::new(),
                Err(e) => e,
            };
        }
        if recording && ui.button(t!("config-perfetto-stop")).clicked() {
            msg = perfetto::stop().unwrap_or_else(|e| e);
        }
    });
    if recording {
        ui.label(RichText::new(status).small());
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(500));
    } else if !msg.is_empty() {
        ui.label(RichText::new(&msg).small());
    }
    subtitle(ui, t!("config-perfetto-subtitle"));
    ui.data_mut(|d| {
        d.insert_temp(path_id, path);
        d.insert_temp(cats_id, cats);
        d.insert_temp(clock_id, guest_clock);
        d.insert_temp(msg_id, msg);
    });
}

fn show_ci(ui: &mut Ui, cfg: &mut MachineConfig) -> ConfigAction {
    ui.heading(t!("config-tab-ci"));
    subtitle(ui, t!("config-ci-subtitle"));
    Grid::new("ci_grid").num_columns(2).striped(true).show(ui, |ui| {
        ui.label(t!("config-ci-enable"));
        ui.checkbox(&mut cfg.ci, "");
        ui.end_row();
        ui.label(t!("config-ci-socket"));
        ui.add(TextEdit::singleline(&mut cfg.ci_socket).desired_width(280.0))
            .tip(t!("config-ci-socket-tip"));
        ui.end_row();
        ui.label(t!("config-ci-display"));
        ui.checkbox(&mut cfg.ci_display, "");
        ui.end_row();
        ui.label(t!("config-serial-log"));
        let mut log = cfg.serial_log.clone().unwrap_or_default();
        if ui.add(TextEdit::singleline(&mut log).desired_width(280.0)).changed() {
            cfg.serial_log = if log.is_empty() { None } else { Some(log) };
        }
        ui.end_row();
        ui.label(t!("config-ci-scsi-deferred"));
        ui.checkbox(&mut cfg.scsi_deferred_int, "")
            .tip(t!("config-ci-scsi-deferred-tip"));
        ui.end_row();
        ui.label(t!("config-ci-aspect"));
        ui.checkbox(&mut cfg.lock_aspect_ratio, "");
        ui.end_row();
        ui.label(t!("config-ci-scroll"));
        ui.add(DragValue::new(&mut cfg.mouse_scroll_pixels_per_line).speed(1.0).range(5.0..=200.0));
        ui.end_row();
    });
    ui.add_space(8.0);
    let mut action = ConfigAction::None;
    if ui.button(t!("config-ci-ping")).clicked() {
        let socket = if iris::config::ci_socket_is_tcp(&cfg.ci_socket) { cfg.ci_socket.clone() } else { crate::abs_path(&cfg.ci_socket) };
        let client = crate::developer_binary("iris-ci");
        let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        std::thread::spawn(move || {
            let status = std::process::Command::new(client)
                .current_dir(cwd)
                .arg("ping")
                .env("IRIS_CI_SOCKET", &socket)
                .output();
            match status {
                Ok(o) if o.status.success() => eprintln!("iris-ci ping: OK"),
                Ok(o) => eprintln!("iris-ci ping failed: {}", String::from_utf8_lossy(&o.stderr)),
                Err(e) => eprintln!("iris-ci ping: {e} (build iris-ci and start iris with ci=true)"),
            }
        });
        ui.ctx().request_repaint();
    }
    if ui.button(t!("config-ci-test")).clicked() {
        action = ConfigAction::TestCi;
    }
    subtitle(ui, t!("config-ci-test-subtitle"));
    action
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
        crate::macos_sandbox::reveal_in_finder(&crate::abs_path(path));
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

/// SCSI target-type picker. `kind`/`cdrom` are two spellings of the same
/// setting in the config, so write both from one place: a DaynaPort must not
/// keep a stale `cdrom = true`, and a disk/CD-ROM must not keep
/// `kind = "daynaport"`.
fn scsi_type_combo(ui: &mut Ui, id: u8, dev: &mut ScsiDeviceConfig, edit: &mut PathEdit) {
    let mut kind = dev.kind();
    let before = kind;
    ComboBox::from_id_salt(("type", id))
        .selected_text(match kind {
            ScsiKind::Disk => t!("config-scsi-type-disk"),
            ScsiKind::Cdrom => t!("config-scsi-type-cdrom"),
            ScsiKind::Daynaport => t!("config-scsi-type-daynaport"),
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut kind, ScsiKind::Disk, t!("config-scsi-type-disk"));
            ui.selectable_value(&mut kind, ScsiKind::Cdrom, t!("config-scsi-type-cdrom"));
            ui.selectable_value(&mut kind, ScsiKind::Daynaport, t!("config-scsi-type-daynaport"));
        });
    if kind != before {
        edit.changed = true;
        if kind == ScsiKind::Daynaport {
            // No image, no media, no overlay behind a network adapter — and
            // leaving them set makes the config fail validation at Start.
            dev.path.clear();
            dev.discs.clear();
            dev.overlay = false;
            dev.scratch = false;
        }
    }
    dev.kind_field = kind;
    dev.cdrom = kind == ScsiKind::Cdrom;
}

/// A TextEdit + 📁 Browse button that updates `value` in place. See [`PathEdit`].
///
/// Browsing opens at the folder `value` names — or, when it names nothing yet,
/// the managed folder for `anchor`. It never opens at the OS's last-used
/// location; `crate::filedialog` explains why that mattered.
fn path_row_in(
    ui: &mut Ui,
    id: impl std::hash::Hash + std::fmt::Debug, // egui 0.35 push_id needs AsIdSalt (Hash + Debug)
    value: &mut String,
    mode: Pick,
    filters: &[(String, &[&str])],
    anchor: crate::filedialog::Anchor,
) -> PathEdit {
    let mut out = PathEdit::default();
    ui.push_id(id, |ui| {
        ui.horizontal(|ui| {
            out.changed |= ui.add(TextEdit::singleline(value).desired_width(320.0)).changed();
            if ui.button("📁").tip(t!("common-browse")).clicked() {
                // Open where this file lives, or is headed — never the OS's
                // remembered folder. See `crate::filedialog`.
                let purpose = match mode {
                    Pick::SaveFile => crate::filedialog::Purpose::Save,
                    Pick::OpenFile | Pick::Dir => crate::filedialog::Purpose::Open,
                };
                let mut d = crate::filedialog::dialog(&t!("config-browse-title"), value.as_str(), anchor, purpose);
                if matches!(mode, Pick::OpenFile | Pick::SaveFile) {
                    for (label, exts) in filters {
                        d = d.add_filter(label.as_str(), exts);
                    }
                }
                let picked = match mode {
                    Pick::OpenFile => d.pick_file(),
                    Pick::SaveFile => d.save_file(),
                    Pick::Dir      => d.pick_folder(),
                };
                if let Some(p) = picked {
                    *value = crate::picked_path(&p);
                    out.changed = true;
                    out.picked = true;
                }
            }
            // Reveal an existing path in the host file manager (Finder, Explorer,
            // …) so the user can find/open it without navigating by hand.
            if !value.trim().is_empty() && Path::new(value.trim()).exists()
                && ui.button("📂").tip(t!("config-reveal-tip")).clicked()
            {
                reveal_in_file_manager(value.trim());
            }
        });
    });
    out
}

/// `path_row_in` anchored on the app's data folder: PROM, NVRAM, logs, shares.
fn path_row(
    ui: &mut Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    value: &mut String,
    mode: Pick,
    filters: &[(String, &[&str])],
) -> PathEdit {
    path_row_in(ui, id, value, mode, filters, crate::filedialog::Anchor::Data)
}

/// `path_row_in` anchored on the managed disks folder — every hard disk, CD
/// image and disc in the changer. Browsing one of these opens where that image
/// actually is, or where a not-yet-created one is destined for.
fn path_row_disk(
    ui: &mut Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    value: &mut String,
    mode: Pick,
    filters: &[(String, &[&str])],
) -> PathEdit {
    path_row_in(ui, id, value, mode, filters, crate::filedialog::Anchor::Disks)
}

/// Same as `path_row` but for `Option<String>` — Browse populates Some,
/// the user can clear by emptying the text.
fn path_row_opt(
    ui: &mut Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    value: &mut Option<String>,
    mode: Pick,
    filters: &[(String, &[&str])],
) {
    let mut s = value.clone().unwrap_or_default();
    path_row(ui, id, &mut s, mode, filters);
    *value = if s.is_empty() { None } else { Some(s) };
}

/// Common file filters: (label, extensions).
type Filters = Vec<(String, &'static [&'static str])>;
fn prom_filters() -> Filters {
    vec![(t!("config-filter-prom"), &["bin"][..]), (t!("config-filter-all"), &["*"][..])]
}
fn nvram_filters() -> Filters {
    vec![(t!("config-filter-nvram"), &["bin"][..]), (t!("config-filter-all"), &["*"][..])]
}
fn disk_filters() -> Filters {
    vec![
        (t!("config-filter-disk"), &["raw", "img", "chd"][..]),
        (t!("config-filter-iso"),  &["iso"][..]),
        (t!("config-filter-all"),  &["*"][..]),
    ]
}
fn any_filters() -> Filters {
    vec![(t!("config-filter-all"), &["*"][..])]
}
#[cfg(feature = "perfetto")]
fn perfetto_filters() -> Filters {
    vec![(t!("config-filter-perfetto"), &["pftrace"][..]), (t!("config-filter-all"), &["*"][..])]
}
