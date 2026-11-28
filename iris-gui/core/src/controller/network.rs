//! Networking help: the "Check networking" diagnosis, adopting the guest's
//! subnet, and the packet-capture (PCAP) permission prompt and elevation.

use std::net::Ipv4Addr;

use super::app::AppController;
use super::{Answer, ChoiceSpec, Effect};
use crate::model::capture_access::{self, EnableOutcome};
use crate::model::handle::{Cmd, NetState};
use crate::model::job::Job;
use crate::model::{netfix, netplan};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkAction {
    /// Open the Check Networking window.
    OpenCheck,
    CloseCheck,
    /// Move NAT onto the guest's subnet (live when running).
    AdoptSubnet(String),
    /// Ask whether to allow packet capture (the Network settings button).
    AskEnableCapture,
    Answer(Answer),
}

/// One line of the diagnosis; the shell styles it by kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckLine {
    Good(String),
    Bad(String),
    Text(String),
    /// Secondary text.
    Weak(String),
    /// A command for the user to type in IRIX.
    Code(String),
    /// Bold lead-in for the code that follows.
    Strong(String),
}

/// The Check Networking window's content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetCheck {
    pub lines: Vec<CheckLine>,
    /// Offer "Use <network> for NAT" with this CIDR.
    pub adopt: Option<String>,
}

pub(crate) const ENABLE_CAPTURE: &str = "network.enable-capture";

#[derive(Default)]
pub struct NetworkController {
    check_open: bool,
    /// The automatic permission prompt fired this run.
    prompted: bool,
    job: Option<Job<EnableOutcome>>,
    host: Vec<netplan::HostIface>,
}

impl NetworkController {
    pub(crate) fn new() -> Self {
        Self { host: netplan::gather_host_ifaces(), ..Self::default() }
    }

    pub fn check_open(&self) -> bool {
        self.check_open
    }

    /// The host's own networks, for subnet presets and overlap warnings.
    pub fn host_ifaces(&self) -> &[netplan::HostIface] {
        &self.host
    }

    /// Whether the elevation is running (its OS prompt may be up).
    pub fn enabling(&self) -> bool {
        self.job.as_ref().is_some_and(|j| j.is_running())
    }

    pub(crate) fn started(&mut self) {
        self.prompted = false;
    }
}

fn capture_question() -> Effect {
    Effect::Choose(ChoiceSpec {
        purpose: ENABLE_CAPTURE,
        title: t!("modal-pcap-title"),
        message: format!("{}\n\n{}", t!("modal-pcap-body"), capture_access::permission_hint()),
        choices: vec![t!("modal-pcap-not-now"), t!("modal-pcap-enable")],
        dismiss: 0,
        default: Some(1),
        destructive: Vec::new(),
    })
}

impl AppController {
    pub fn network(&self) -> &NetworkController {
        &self.network
    }

    pub(crate) fn network_apply(&mut self, action: NetworkAction, fx: &mut Vec<Effect>) {
        match action {
            NetworkAction::OpenCheck => self.network.check_open = true,
            NetworkAction::CloseCheck => self.network.check_open = false,
            NetworkAction::AdoptSubnet(s) => {
                self.config_mut().nat_subnet = Some(s.clone());
                fx.extend(self.apply_config_edited());
                if self.session().is_running() {
                    self.session().send(Cmd::SetNatSubnet(s.clone()));
                    fx.push(Effect::Toast(t!("toast-nat-subnet-live", network = &s)));
                } else {
                    fx.push(Effect::Toast(t!("toast-nat-subnet-next-start", network = &s)));
                }
            }
            NetworkAction::AskEnableCapture => fx.push(capture_question()),
            NetworkAction::Answer(Answer::Chose { purpose: ENABLE_CAPTURE, index: 1 }) => {
                if !self.network.enabling() {
                    // The OS asks for an administrator password: a slow, blocking step.
                    self.network.job = Some(Job::spawn(t!("modal-pcap-title"), false, self.notify(), |_| {
                        Ok(capture_access::enable_packet_capture())
                    }));
                }
            }
            NetworkAction::Answer(_) => {}
        }
    }

    pub(crate) fn network_tick(&mut self, fx: &mut Vec<Effect>) {
        if let Some(outcome) = self.network.job.as_ref().and_then(|j| j.outcome()) {
            self.network.job = None;
            match outcome {
                Ok(EnableOutcome::Enabled) => fx.push(Effect::Toast(t!("toast-pcap-enabled"))),
                Ok(EnableOutcome::NeedsRelaunch(msg)) | Ok(EnableOutcome::Failed(msg)) | Err(msg) => {
                    fx.push(Effect::Toast(msg))
                }
                Ok(EnableOutcome::Cancelled) => {}
            }
        }
        // A bridged machine that could not open its capture for lack of
        // privilege: ask once per run.
        if !self.network.prompted && self.session().status().pcap_status == iris::net::PcapStatus::PermissionDenied {
            self.network.prompted = true;
            fx.push(capture_question());
        }
    }

    /// The Check Networking diagnosis for the running machine.
    pub fn net_check(&self) -> NetCheck {
        let s = self.session().status();
        let state = self.session().handle().net_state();
        let (eb, ep) = netplan::parse_cidr(self.config().nat_subnet.as_deref());
        let cfg_net = netplan::classify(eb, ep, &[]).derived;
        let mut lines = Vec::new();
        let mut adopt = None;
        match state {
            NetState::Off => lines.push(CheckLine::Text(t!("window-net-not-running"))),
            NetState::Active => {
                lines.push(CheckLine::Good(t!("window-net-up")));
                lines.push(CheckLine::Text(match s.net_guest_ip {
                    Some(i) => t!("window-net-up-detail", ip = i.to_string()),
                    None => t!("window-net-up-detail-no-ip"),
                }));
                if let Some(g) = s.net_guest_gateway {
                    lines.push(CheckLine::Weak(t!("window-net-gateway", gateway = g.to_string())));
                }
            }
            NetState::Idle => match s.net_guest_ip {
                None => {
                    lines.push(CheckLine::Strong(t!("window-net-no-traffic")));
                    lines.push(CheckLine::Text(t!("window-net-no-traffic-detail")));
                    lines.push(CheckLine::Strong(t!("window-net-enable")));
                    lines.push(CheckLine::Code("chkconfig network on".into()));
                    lines.push(CheckLine::Weak(t!("window-net-enable-detail")));
                    if let Some(d) = &cfg_net {
                        lines.push(CheckLine::Weak(t!(
                            "window-net-fresh-hint",
                            ip = d.client.to_string(),
                            gateway = d.gateway.to_string(),
                            netmask = d.netmask.to_string()
                        )));
                    }
                }
                Some(ip) => {
                    let o = ip.octets();
                    let guest_net = Ipv4Addr::new(o[0], o[1], o[2], 0);
                    let sub_gw = Ipv4Addr::new(o[0], o[1], o[2], 1);
                    lines.push(CheckLine::Bad(t!("window-net-down")));
                    lines.push(CheckLine::Text(t!("window-net-guest-ip", ip = ip.to_string(), network = format!("{guest_net}/24"))));
                    match s.net_guest_gateway {
                        Some(g) => {
                            lines.push(CheckLine::Text(t!("window-net-has-gateway", gateway = g.to_string())));
                            lines.push(CheckLine::Weak(t!("window-net-has-gateway-detail")));
                        }
                        None => {
                            lines.push(CheckLine::Text(t!("window-net-no-gateway")));
                            lines.push(CheckLine::Strong(t!("window-net-run-as-root")));
                            lines.push(CheckLine::Code(format!("/usr/etc/route add default {sub_gw} 1")));
                            lines.push(CheckLine::Weak(t!("window-net-route-temporary")));
                        }
                    }
                    if cfg_net.as_ref().map(|d| d.network) != Some(guest_net) {
                        match netplan::conflict(guest_net, 24, &self.network.host) {
                            Some(h) => {
                                lines.push(CheckLine::Bad(t!(
                                    "window-net-overlap",
                                    network = format!("{guest_net}/24"),
                                    iface = &h.name,
                                    addr = h.addr.to_string()
                                )));
                                if let Some(d) = &cfg_net {
                                    lines.push(CheckLine::Strong(t!("window-net-run-as-root")));
                                    let e = netfix::ExpectedNet { ip: d.client, gateway: d.gateway, netmask: d.netmask };
                                    lines.extend(netfix::runtime_fix_commands(&e).into_iter().map(CheckLine::Code));
                                    lines.push(CheckLine::Weak(t!(
                                        "window-net-overlap-result",
                                        ip = d.client.to_string(),
                                        gateway = d.gateway.to_string()
                                    )));
                                }
                            }
                            None => {
                                lines.push(CheckLine::Weak(match &cfg_net {
                                    Some(d) => t!("window-net-nat-subnet", network = format!("{}/{}", d.network, d.prefix)),
                                    None => t!("window-net-nat-subnet-other"),
                                }));
                                adopt = Some(format!("{guest_net}/24"));
                            }
                        }
                    }
                }
            },
        }
        NetCheck { lines, adopt }
    }
}
