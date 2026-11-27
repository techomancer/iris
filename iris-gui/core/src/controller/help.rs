//! The Help menu's pages as documents: lines a shell styles by kind. Some
//! carry live values (the gateway the guest mounts from, the serial and
//! monitor ports, whether the Ultra64 board is on), so they are built from
//! the selected machine each time they are shown.

use super::app::AppController;
use super::debug::{build_features, monitor_port};
use crate::model::netplan;
use crate::model::serial_console::console_port;

/// One line of a help page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Line {
    Heading(String),
    /// A bold lead-in.
    Strong(String),
    Text(String),
    /// Secondary text.
    Weak(String),
    /// A caution, drawn in the warning colour.
    Warn(String),
    Error(String),
    /// Something to type in IRIX or at the monitor, monospaced and copyable.
    Code(String),
    Link { label: String, url: String },
    /// A separator.
    Rule,
}

/// A page and its title.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelpDoc {
    pub title: String,
    pub lines: Vec<Line>,
}

/// The pages of the Help menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpPage {
    About,
    CameraNetwork,
    Nfs,
    /// The N64 development board guide; not in App Store builds.
    Ultra64,
    Licenses,
    Privacy,
}

impl HelpPage {
    /// The pages this build offers, in menu order.
    pub fn all() -> Vec<HelpPage> {
        let mut v = vec![HelpPage::CameraNetwork, HelpPage::Nfs];
        if !cfg!(feature = "appstore") {
            v.push(HelpPage::Ultra64);
        }
        v.extend([HelpPage::Licenses, HelpPage::Privacy]);
        v
    }

    /// The menu item's id.
    pub fn id(self) -> &'static str {
        match self {
            HelpPage::About => "help.about",
            HelpPage::CameraNetwork => "help.camera-network",
            HelpPage::Nfs => "help.nfs",
            HelpPage::Ultra64 => "help.ultra64",
            HelpPage::Licenses => "help.licenses",
            HelpPage::Privacy => "help.privacy",
        }
    }

    pub fn from_id(id: &str) -> Option<HelpPage> {
        [HelpPage::About, HelpPage::CameraNetwork, HelpPage::Nfs, HelpPage::Ultra64, HelpPage::Licenses, HelpPage::Privacy]
            .into_iter()
            .find(|p| p.id() == id)
    }

    /// The menu item's title.
    pub fn menu_title(self) -> String {
        match self {
            HelpPage::About => t!("menu-help-title"),
            HelpPage::CameraNetwork => t!("menu-help-camera-network"),
            HelpPage::Nfs => t!("menu-help-nfs"),
            HelpPage::Ultra64 => t!("menu-help-ultra64"),
            HelpPage::Licenses => t!("menu-help-licenses"),
            HelpPage::Privacy => t!("menu-help-privacy"),
        }
    }
}

const LICENSE_BSD: &str = include_str!("../../../../LICENSE");
const LICENSE_CHD: &str = include_str!("../../../../LICENSE-libchdman-rs.txt");
const LICENSE_RB: &str = include_str!("../../../../LICENSE-rusty-backup-crates.txt");
const PRIVACY: &str = include_str!("../../../../PRIVACY.md");
const REPO: &str = "https://github.com/techomancer/iris";

impl AppController {
    /// The help page `page`. `version` and `commit` are the shell's own
    /// (each app binary stamps its build).
    pub fn help(&self, page: HelpPage, version: &str, commit: &str) -> HelpDoc {
        use Line::*;
        let cfg = self.config();
        match page {
            HelpPage::About => {
                let mut lines = vec![
                    Heading(t!("menu-help-title")),
                    Text(t!("menu-help-version", version = version, commit = commit)),
                    Rule,
                    Strong(t!("menu-help-authors")),
                    Text(t!("menu-help-authors-iris")),
                    Text(t!("menu-help-authors-gui")),
                    Link { label: format!("{} techomancer/iris", t!("menu-help-repository")), url: REPO.into() },
                    Rule,
                    Strong(t!("menu-help-build-features")),
                ];
                lines.extend(build_features().iter().map(|f| Weak(t!("menu-help-feature", name = f.name, state = f.state.as_str()))));
                HelpDoc { title: t!("app-name"), lines }
            }
            HelpPage::CameraNetwork => HelpDoc {
                title: t!("help-info-title"),
                lines: vec![
                    Heading(t!("help-camera-heading")),
                    Text(t!("help-camera-body")),
                    Strong(t!("help-how-to-use")),
                    Text(t!("help-camera-usage")),
                    Strong(t!("help-privacy")),
                    Text(t!("help-camera-privacy")),
                    Rule,
                    Heading(t!("help-network-heading")),
                    Text(t!("help-network-body")),
                    Strong(t!("help-network-subnet-heading")),
                    Text(t!("help-network-subnet")),
                    Strong(t!("help-network-serial-heading")),
                    Text(t!(
                        "help-network-serial",
                        serial = console_port(cfg).map_or("-".into(), |p| p.to_string()),
                        monitor = monitor_port(cfg).to_string()
                    )),
                    Strong(t!("help-privacy")),
                    Text(t!("help-network-privacy")),
                ],
            },
            HelpPage::Nfs => HelpDoc { title: t!("help-nfs-title"), lines: self.nfs_lines() },
            HelpPage::Ultra64 => {
                let mut lines = vec![Text(t!("help-ultra64-intro")), Strong(t!("help-ultra64-step1"))];
                lines.push(if cfg.has_device("ultra64") { Weak(t!("help-ultra64-enabled")) } else { Warn(t!("help-ultra64-disabled")) });
                lines.extend([
                    Strong(t!("help-ultra64-step2")),
                    Weak(t!("help-ultra64-step2-body")),
                    Code("git clone -b ultra64 https://github.com/techomancer/gopher64\ncd gopher64\ncargo run --features ultra64 --no-default-features".into()),
                    Link { label: format!("{} techomancer/gopher64", t!("help-ultra64-fork")), url: "https://github.com/techomancer/gopher64".into() },
                    Strong(t!("help-ultra64-step3")),
                    Weak(t!("help-ultra64-step3-body")),
                    Code("gload /path/to/game.n64".into()),
                    Weak(t!("help-ultra64-switch")),
                    Rule,
                    Strong(t!("help-ultra64-monitor-heading")),
                    Weak(t!("help-ultra64-monitor-intro")),
                ]);
                for (c, d) in [
                    ("ultra status", t!("help-ultra64-cmd-status")),
                    ("ultra reset", t!("help-ultra64-cmd-reset")),
                    ("ultra load <path> [off]", t!("help-ultra64-cmd-load")),
                    ("ultra dump <off> <size>", t!("help-ultra64-cmd-dump")),
                    ("ultra disasm <off> <n>", t!("help-ultra64-cmd-disasm")),
                ] {
                    lines.push(Code(c.into()));
                    lines.push(Weak(d));
                }
                lines.push(Weak(t!("help-ultra64-footer")));
                HelpDoc { title: t!("help-ultra64-title"), lines }
            }
            HelpPage::Licenses => HelpDoc {
                title: t!("window-license-title"),
                lines: vec![
                    Text(t!("window-license-intro")),
                    Text(t!("window-license-chd")),
                    Link { label: format!("{} techomancer/iris", t!("window-license-source")), url: REPO.into() },
                    Link { label: format!("{} libchdman-rs", t!("window-license-source")), url: "https://crates.io/crates/libchdman-rs".into() },
                    Rule,
                    Strong(t!("window-license-iris-heading")),
                    Code(LICENSE_BSD.into()),
                    Strong(t!("window-license-chd-heading")),
                    Code(LICENSE_CHD.into()),
                    Strong(t!("window-license-rb-heading")),
                    Code(LICENSE_RB.into()),
                ],
            },
            HelpPage::Privacy => HelpDoc { title: t!("window-privacy-title"), lines: vec![Text(PRIVACY.into())] },
        }
    }

    fn nfs_lines(&self) -> Vec<Line> {
        use iris::net::nfsudp::NfsVersion;
        use Line::*;
        let cfg = self.config();
        let Some(nfs) = cfg.nfs.as_ref() else {
            return vec![Strong(t!("help-nfs-disabled")), Text(t!("help-nfs-disabled-detail"))];
        };
        // Where the guest mounts from: the NFS server's own address when
        // bridged; in NAT, the gateway IRIS adopted, else the subnet's.
        let pcap = cfg.network.mode == iris::config::NetMode::Pcap;
        let gw = if pcap {
            cfg.network.nfs_pcap_ip.map(|ip| ip.to_string()).unwrap_or_else(|| t!("help-nfs-no-server-ip"))
        } else {
            let (eb, ep) = netplan::parse_cidr(cfg.nat_subnet.as_deref());
            let derived = netplan::classify(eb, ep, &[]).derived.map(|d| d.gateway.to_string());
            self.session().status().net_guest_gateway.map(|g| g.to_string()).or(derived).unwrap_or_else(|| "192.168.0.1".into())
        };
        let mut lines = vec![Text(if pcap { t!("help-nfs-intro-bridged") } else { t!("help-nfs-intro-nat") })];
        lines.push(if nfs.shared_dir.trim().is_empty() {
            Error(t!("help-nfs-no-folder"))
        } else {
            Weak(t!("help-nfs-sharing", path = nfs.shared_dir.as_str()))
        });
        lines.push(Code(format!("mkdir -p /shared\nmount {gw}:/ /shared")));
        lines.push(Weak(t!("help-nfs-mounted")));
        lines.push(Weak(match nfs.version {
            NfsVersion::Auto => t!("help-nfs-version-auto"),
            NfsVersion::V2 => t!("help-nfs-version-2"),
            NfsVersion::V3 => t!("help-nfs-version-3"),
        }));
        lines.push(match nfs.version {
            NfsVersion::V3 => Weak(t!("help-nfs-size-v3")),
            _ => Warn(t!("help-nfs-size-v2")),
        });
        let pin = match nfs.version {
            NfsVersion::V2 => "vers=2,proto=udp",
            _ => "vers=3,proto=udp",
        };
        lines.extend([
            Rule,
            Strong(t!("help-nfs-trouble-heading")),
            Text(t!("help-nfs-trouble")),
            Code(format!("mount -o {pin} {gw}:/ /shared")),
            Weak(t!("help-nfs-trouble-network")),
            Strong(t!("help-nfs-unmount")),
            Code("umount /shared".into()),
            Rule,
            Strong(t!("help-nfs-limits-heading")),
        ]);
        for (head, body) in [
            (t!("help-nfs-limit-size"), t!("help-nfs-limit-size-body")),
            (t!("help-nfs-limit-udp"), t!("help-nfs-limit-udp-body")),
            (t!("help-nfs-limit-export"), t!("help-nfs-limit-export-body")),
            (t!("help-nfs-limit-permissions"), t!("help-nfs-limit-permissions-body")),
            (t!("help-nfs-limit-auth"), t!("help-nfs-limit-auth-body")),
            (t!("help-nfs-limit-locking"), t!("help-nfs-limit-locking-body")),
            (t!("help-nfs-limit-special"), t!("help-nfs-limit-special-body")),
            (t!("help-nfs-limit-sync"), t!("help-nfs-limit-sync-body")),
        ] {
            lines.push(Strong(head));
            lines.push(Weak(body));
        }
        lines
    }
}
