// PCAP bridged-networking backend for the SEEQ 8003 emulator.
//
// An alternative to the software NAT gateway in net.rs. Instead of synthesizing
// replies, this backend bridges the guest's raw Ethernet frames straight onto a
// real host interface via libpcap. The guest then appears as a real layer-2
// host on the physical LAN: it can DHCP from the real network, be pinged from
// other machines, etc.
//
// Wiring is identical to NatEngine: it owns the same rtrb ring endpoints and
// wake condvars, runs on the "seeq-nat" thread, and implements NetBackend.
//
// Library / licensing note:
//   The `pcap` crate links the generic `wpcap` import library on Windows (NOT a
//   driver-specific one), so IRIS is not tied to any single provider. You can
//   build/link against the BSD-licensed WinPcap Developer Pack as well as Npcap.
//   We link dynamically and never bundle the driver, so the runtime driver's
//   license (e.g. Npcap's redistribution terms) does not attach to IRIS.
//
// Requirements:
//   - Build with `--features pcap` (pulls in the `pcap` crate; needs libpcap
//     headers/lib on Unix, or a WinPcap-compatible `wpcap` SDK on Windows).
//   - Run with privileges to open a raw capture (root / CAP_NET_RAW on Linux,
//     root on macOS, Administrator + a WinPcap-compatible driver on Windows).
//
// Caveats:
//   - libpcap delivers our own injected (TX) frames back on the capture handle.
//     We filter those out by dropping any captured frame whose Ethernet source
//     MAC equals the guest's MAC (learned from the first outbound frame).
//   - No NAT services are provided. The guest must obtain its IP from the real
//     network (DHCP or static) and the host interface must be on a network that
//     tolerates an extra MAC (wired bridges work best; many Wi-Fi APs reject it).

use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use crate::config::NetMode;
use crate::devlog::LogModule;
use crate::net::{eth_summary, mac_str, GatewayConfig, NatControl, NetBackend, NfsVirtualHost, PcapStatus};

/// Summary of one host interface, returned by `list_interfaces()`.
pub struct NetInterface {
    pub name: String,
    pub description: Option<String>,
    pub addresses: Vec<std::net::IpAddr>,
    pub up: bool,
    pub running: bool,
    pub loopback: bool,
}

/// Enumerate the host network interfaces libpcap can bridge onto.
///
/// Returns an error string if libpcap can't list devices (e.g. insufficient
/// privileges, or no WinPcap-compatible driver installed on Windows).
pub fn list_interfaces() -> Result<Vec<NetInterface>, String> {
    let devices = pcap::Device::list().map_err(|e| e.to_string())?;
    Ok(devices
        .into_iter()
        .map(|d| {
            let flags = &d.flags;
            NetInterface {
                name: d.name,
                description: d.desc,
                addresses: d.addresses.into_iter().map(|a| a.addr).collect(),
                up: flags.is_up(),
                running: flags.is_running(),
                loopback: flags.is_loopback(),
            }
        })
        .collect())
}

/// Format the interface list as a human-readable table for CLI/monitor output.
pub fn format_interfaces() -> String {
    match list_interfaces() {
        Ok(ifaces) => {
            if ifaces.is_empty() {
                return "No network interfaces found (libpcap returned an empty list).\n\
                        On Linux you may need root or CAP_NET_RAW; on Windows, install a\n\
                        WinPcap-compatible driver (WinPcap or Npcap).\n"
                    .to_string();
            }
            let mut out = String::new();
            out.push_str("Available network interfaces for PCAP bridging:\n");
            out.push_str("(configure with `[network] pcap_interface = \"<index-or-name>\"` in iris.toml;\n");
            out.push_str(" e.g. pcap_interface = \"1\" selects the first interface below)\n\n");
            for (idx, i) in ifaces.iter().enumerate() {
                let mut tags = Vec::new();
                if i.up { tags.push("up"); }
                if i.running { tags.push("running"); }
                if i.loopback { tags.push("loopback"); }
                let tags_str = if tags.is_empty() { String::new() } else { format!(" [{}]", tags.join(",")) };
                // 1-based index for human selection.
                out.push_str(&format!("  {:>2}. {}{}\n", idx + 1, i.name, tags_str));
                if let Some(desc) = &i.description {
                    out.push_str(&format!("        {}\n", desc));
                }
                if !i.addresses.is_empty() {
                    let addrs: Vec<String> = i.addresses.iter().map(|a| a.to_string()).collect();
                    out.push_str(&format!("        addr: {}\n", addrs.join(", ")));
                }
            }
            out
        }
        Err(e) => format!(
            "Failed to list network interfaces: {}\n\
             On Linux you may need root or CAP_NET_RAW; on Windows, install a\n\
             WinPcap-compatible driver (WinPcap or Npcap).\n",
            e
        ),
    }
}

/// Resolve `[network] pcap_interface` to a concrete device name.
///
/// The configured value may be:
///   - a 1-based index into the `--list-net-interfaces` listing (e.g. "1"),
///     which is convenient on Windows where device names are long NPF GUIDs
///     like `\Device\NPF_{....}`;
///   - an exact device name (e.g. "eth0", or the full `\Device\NPF_{...}`);
///   - empty / unset, in which case we auto-pick the first non-loopback
///     interface that is up and running with an address.
///
/// Index resolution uses the same ordering libpcap reports, so the numbers
/// match exactly what the user sees in `--list-net-interfaces`.
fn select_interface(configured: &Option<String>) -> Result<String, String> {
    let ifaces = list_interfaces()?;

    // An explicit value (index or name) always wins.
    let trimmed = configured.as_deref().map(str::trim).unwrap_or("");
    if !trimmed.is_empty() {
        return resolve_interface(Some(trimmed), &ifaces);
    }

    // No interface configured: auto-pick. Interactive selection (the numbered
    // menu shown when no interface is set) is done once at CLI startup on the
    // main thread via `prompt_for_interface`, NOT here — this runs on the
    // seeq-nat thread during machine boot, where it can't gate startup and
    // (on Windows especially) a background-thread stdin read returns EOF
    // immediately, so a prompt here would just flash the menu and auto-pick.
    auto_pick(&ifaces)
}

/// Interactively prompt on stdin for a host interface to bridge onto, for use
/// when `[network] mode = "pcap"` is set but no `pcap_interface` is configured.
///
/// Returns the chosen device name, or `None` to defer to auto-pick (blank input,
/// EOF, no interactive console, or enumeration failure). Must be called once at
/// CLI startup on the **main thread** — never from the network backend thread,
/// which runs during boot and must not block on or race the console.
pub fn prompt_for_interface() -> Option<String> {
    if !stdin_is_tty() {
        return None;
    }
    let ifaces = list_interfaces().ok()?;
    prompt_interface(&ifaces)
}

/// Pure resolution of an *explicit* (non-empty) `pcap_interface` value against
/// an enumerated list: numeric 1-based index, exact name, or verbatim
/// pass-through. Split out so it can be unit-tested without libpcap. Passing
/// `None`/empty here auto-picks (used by the unit tests).
fn resolve_interface(configured: Option<&str>, ifaces: &[NetInterface]) -> Result<String, String> {
    if let Some(raw) = configured {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            // Numeric index (1-based) selection.
            if let Ok(idx) = trimmed.parse::<usize>() {
                if idx == 0 || idx > ifaces.len() {
                    return Err(format!(
                        "pcap_interface index {} is out of range (1..={}); \
                         run `iris --list-net-interfaces`",
                        idx, ifaces.len()
                    ));
                }
                return Ok(ifaces[idx - 1].name.clone());
            }
            // Exact name match (case-sensitive; NPF GUIDs are case-sensitive).
            if let Some(found) = ifaces.iter().find(|i| i.name == trimmed) {
                return Ok(found.name.clone());
            }
            // Not an index and not a known name: pass it through verbatim so a
            // valid-but-unlisted name (or one libpcap accepts directly) still works,
            // but warn that it wasn't in the enumerated list.
            eprintln!(
                "iris: pcap_interface '{}' not found in the interface list; \
                 trying it verbatim. Run `iris --list-net-interfaces` to see valid names/indices.",
                trimmed
            );
            return Ok(trimmed.to_string());
        }
    }
    auto_pick(ifaces)
}

/// Auto-pick the first up, running, non-loopback interface that has an address.
fn auto_pick(ifaces: &[NetInterface]) -> Result<String, String> {
    ifaces
        .iter()
        .find(|i| i.up && i.running && !i.loopback && !i.addresses.is_empty())
        .or_else(|| ifaces.iter().find(|i| i.up && !i.loopback))
        .map(|i| i.name.clone())
        .ok_or_else(|| "no suitable non-loopback interface found; set [network] pcap_interface".to_string())
}

/// True when stdin is connected to an interactive terminal.
fn stdin_is_tty() -> bool {
    #[cfg(unix)]
    {
        // SAFETY: isatty just inspects the fd; STDIN_FILENO is always valid.
        unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        // GetConsoleMode succeeds only on a real console handle.
        extern "system" {
            fn GetConsoleMode(h: *mut std::ffi::c_void, mode: *mut u32) -> i32;
        }
        let handle = std::io::stdin().as_raw_handle();
        let mut mode: u32 = 0;
        unsafe { GetConsoleMode(handle as *mut std::ffi::c_void, &mut mode) != 0 }
    }
    #[cfg(not(any(unix, windows)))]
    {
        false
    }
}

/// Show a numbered menu of interfaces and read the user's choice from stdin.
/// Returns the chosen device name, or None on empty input / EOF / invalid
/// choice (caller falls back to auto-pick).
fn prompt_interface(ifaces: &[NetInterface]) -> Option<String> {
    if ifaces.is_empty() {
        return None;
    }
    eprintln!();
    eprintln!("PCAP networking: no [network] pcap_interface configured.");
    eprintln!("Select the host interface to bridge the guest onto:");
    eprintln!();
    for (idx, i) in ifaces.iter().enumerate() {
        let mut tags = Vec::new();
        if i.up { tags.push("up"); }
        if i.running { tags.push("running"); }
        if i.loopback { tags.push("loopback"); }
        let tags_str = if tags.is_empty() { String::new() } else { format!(" [{}]", tags.join(",")) };
        let addr = if i.addresses.is_empty() {
            String::new()
        } else {
            format!("  {}", i.addresses.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", "))
        };
        let desc = i.description.as_deref().unwrap_or("");
        eprintln!("  {:>2}) {}{}{}", idx + 1, i.name, tags_str, addr);
        if !desc.is_empty() {
            eprintln!("        {}", desc);
        }
    }
    eprintln!();
    eprint!("Interface number (1-{}, blank = auto-pick): ", ifaces.len());
    let _ = std::io::stderr().flush();

    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).ok()? == 0 {
        return None; // EOF
    }
    let choice = line.trim();
    if choice.is_empty() {
        return None; // blank → auto-pick
    }
    match choice.parse::<usize>() {
        Ok(idx) if idx >= 1 && idx <= ifaces.len() => {
            let name = ifaces[idx - 1].name.clone();
            eprintln!("Using interface: {}", name);
            Some(name)
        }
        _ => {
            eprintln!("Invalid selection '{}'; auto-picking instead.", choice);
            None
        }
    }
}

/// PCAP bridged backend. Mirrors the threading/ring-buffer contract of NatEngine.
pub struct PcapEngine {
    config:  GatewayConfig,
    tx_cons: rtrb::Consumer<Vec<u8>>,         // outbound frames from enet thread
    rx_prod: rtrb::Producer<Vec<u8>>,         // inbound frames to enet thread
    rx_wake: Arc<(Mutex<()>, Condvar)>,       // signal enet thread on new RX
    tx_wake: Arc<(Mutex<()>, Condvar)>,       // wait for new TX frames
    running: Arc<AtomicBool>,
    ctl:     Arc<NatControl>,
    /// Guest MAC, learned from the first outbound frame. Used to filter our own
    /// injected frames back out of the capture stream.
    guest_mac: Option<[u8; 6]>,
    /// In-process NFS server presented as its own virtual L2 host on the bridged
    /// LAN. `Some` when an NFS share is configured and `[network] nfs_pcap_ip` is
    /// set; the guest's ARP/portmap/NFS frames to that IP are answered locally
    /// instead of going to the wire.
    nfs_host: Option<NfsVirtualHost>,
}

impl PcapEngine {
    pub fn new(config: GatewayConfig,
               tx_cons: rtrb::Consumer<Vec<u8>>,
               rx_prod: rtrb::Producer<Vec<u8>>,
               rx_wake: Arc<(Mutex<()>, Condvar)>,
               tx_wake: Arc<(Mutex<()>, Condvar)>,
               running: Arc<AtomicBool>,
               ctl:     Arc<NatControl>) -> Self {
        // Stand up the virtual NFS host when an export and a virtual IP are both
        // configured — otherwise bridged frames just go to the wire.
        let nfs_host = match (config.nfs.clone(), config.nfs_pcap_ip) {
            (Some(nfs_cfg), Some(ip)) => {
                eprintln!("iris: PCAP NFS server on virtual host {} exporting {}", ip, nfs_cfg.shared_dir);
                Some(NfsVirtualHost::new(ip, nfs_cfg))
            }
            _ => None,
        };
        Self { config, tx_cons, rx_prod, rx_wake, tx_wake, running, ctl, guest_mac: None, nfs_host }
    }

    /// Open the configured (or auto-selected) capture handle in promiscuous,
    /// immediate, non-blocking mode.
    fn open_capture(&self) -> Result<pcap::Capture<pcap::Active>, String> {
        let iface = select_interface(&self.config.pcap_interface)?;
        eprintln!("iris: PCAP networking bridging onto interface '{}'", iface);
        let cap = pcap::Capture::from_device(iface.as_str())
            .map_err(|e| format!("open device '{}': {}", iface, e))?
            .promisc(true)
            .immediate_mode(true)
            .snaplen(65535)
            .timeout(1)
            .open()
            .map_err(|e| format!("activate device '{}': {} (need root/CAP_NET_RAW on Unix, or a WinPcap-compatible driver + Administrator on Windows?)", iface, e))?;
        cap.setnonblock().map_err(|e| format!("set non-blocking: {}", e))
    }
}

impl NetBackend for PcapEngine {
    fn run(&mut self) {
        let mut cap = match self.open_capture() {
            Ok(c) => {
                // Capture is live — let the GUI light "capturing" and dismiss any
                // earlier permission prompt.
                self.ctl.set_pcap_status(PcapStatus::Active);
                c
            }
            Err(e) => {
                // Classify the failure so the GUI can offer to elevate (permission)
                // vs. just report a bad/absent device. See `classify_open_error`.
                self.ctl.set_pcap_status(classify_open_error(&e));
                eprintln!("iris: PCAP backend disabled: {}", e);
                eprintln!("iris: the guest will have NO networking. Use `[network] mode = \"nat\"` for the software gateway.");
                // Drain TX so the guest's ring doesn't back up, but produce no RX.
                while self.running.load(Ordering::Relaxed) {
                    {
                        let (lock, cvar) = &*self.tx_wake;
                        let mut guard = lock.lock();
                        let _ = cvar.wait_for(&mut guard, Duration::from_millis(50));
                    }
                    while self.tx_cons.pop().is_ok() {}
                }
                return;
            }
        };

        while self.running.load(Ordering::Relaxed) {
            // Wait for outbound frames or a short timeout to poll the capture.
            {
                let (lock, cvar) = &*self.tx_wake;
                let mut guard = lock.lock();
                let _ = cvar.wait_for(&mut guard, Duration::from_millis(1));
            }

            // Machine reset: nothing stateful to flush in bridged mode, but honor
            // the flag so it doesn't stay set.
            self.ctl.reset_nat.swap(false, Ordering::AcqRel);

            // Live host-NIC reswap (GUI changed the PCAP interface on a running
            // machine). Reopen the capture in place; the guest is untouched. If
            // the new interface fails to open, keep the current one so a typo
            // doesn't drop networking.
            if self.ctl.apply_pcap_iface.swap(false, Ordering::AcqRel) {
                let new_iface = self.ctl.pending_pcap_iface.lock().clone();
                self.config.pcap_interface = new_iface;
                match self.open_capture() {
                    Ok(c) => {
                        cap = c;
                        self.ctl.set_pcap_status(PcapStatus::Active);
                        eprintln!("iris: PCAP capture re-opened on new interface");
                    }
                    Err(e) => {
                        eprintln!("iris: PCAP reopen failed: {}; keeping the previous interface", e);
                    }
                }
            }

            // ── TX: guest → host wire ────────────────────────────────────────
            while let Ok(frame) = self.tx_cons.pop() {
                if frame.len() >= 12 && self.guest_mac.is_none() {
                    let mac: [u8; 6] = frame[6..12].try_into().unwrap();
                    self.guest_mac = Some(mac);
                    dlog_dev!(LogModule::Net, "PCAP learned guest MAC {}", mac_str(&mac));
                    // Now that the guest's MAC is known, push a kernel BPF filter so
                    // promiscuous capture only delivers frames the guest can accept
                    // (its own unicast + broadcast + multicast) rather than the whole
                    // LAN's unicast chatter, which would otherwise flood the RX ring.
                    // (Our own injected broadcasts still come back and are dropped by
                    // the src==guest_mac check below.)
                    let prog = format!("ether dst {m} or ether broadcast or ether multicast", m = mac_str(&mac));
                    if let Err(e) = cap.filter(&prog, true) {
                        dlog_dev!(LogModule::Net, "PCAP set capture filter failed: {}", e);
                    } else {
                        dlog_dev!(LogModule::Net, "PCAP capture filter set: {}", prog);
                    }
                }
                // Count guest-originated IPv4 frames so the GUI's NET indicator
                // lights in PCAP mode too (the NAT engine isn't running to bump
                // this). ARP / link-layer chatter is excluded, matching NAT's
                // semantics — it's "the guest is actually carrying IP traffic".
                if frame.len() >= 14 && frame[12] == 0x08 && frame[13] == 0x00 {
                    self.ctl.guest_frames.fetch_add(1, Ordering::Relaxed);
                }
                // Intercept frames bound for the virtual NFS host (ARP / portmap /
                // NFS / mountd): answer them locally and inject the reply on the RX
                // ring, rather than leaking them onto the real LAN.
                if let Some(host) = self.nfs_host.as_mut() {
                    if let Some(replies) = host.maybe_handle(&frame) {
                        for reply in replies {
                            if self.rx_prod.slots() > 0 && self.rx_prod.push(reply).is_ok() {
                                self.rx_wake.1.notify_one();
                            }
                        }
                        continue;
                    }
                }
                if self.ctl.dbg_tcp() {
                    dlog_dev!(LogModule::Net, "PCAP TX {}", eth_summary(&frame));
                }
                if let Err(e) = cap.sendpacket(&frame[..]) {
                    dlog_dev!(LogModule::Net, "PCAP sendpacket failed: {}", e);
                }
            }

            // ── RX: host wire → guest ────────────────────────────────────────
            // Drain everything currently buffered; non-blocking next_packet
            // returns TimeoutExpired/NoMorePackets when empty.
            loop {
                if self.rx_prod.slots() == 0 { break; }
                match cap.next_packet() {
                    Ok(pkt) => {
                        let frame: &[u8] = &pkt;
                        if frame.len() < 14 { continue; }
                        // Drop our own injected frames (src MAC == guest MAC).
                        if let Some(gmac) = self.guest_mac {
                            if frame[6..12] == gmac {
                                continue;
                            }
                        }
                        // Hand the raw frame to the enet thread; address filtering
                        // (station MAC / broadcast / multicast / promiscuous) is
                        // applied there in Seeq8003::pump_rx.
                        if self.ctl.dbg_tcp() {
                            dlog_dev!(LogModule::Net, "PCAP RX {}", eth_summary(frame));
                        }
                        if self.rx_prod.push(frame.to_vec()).is_ok() {
                            self.rx_wake.1.notify_one();
                        }
                    }
                    Err(pcap::Error::TimeoutExpired) | Err(pcap::Error::NoMorePackets) => break,
                    Err(e) => {
                        dlog_dev!(LogModule::Net, "PCAP next_packet error: {}", e);
                        break;
                    }
                }
            }
        }
    }
}

/// True when the requested mode is PCAP. Convenience so callers don't need to
/// import NetMode just to branch.
pub fn is_pcap_mode(mode: NetMode) -> bool {
    mode == NetMode::Pcap
}

/// Classify a libpcap open/activate error into a [`PcapStatus`] the GUI can act
/// on. pcap 2.x has no dedicated permission variant — EPERM/EACCES surface as
/// `PcapError(String)` or `ErrnoError`, both of which Display to text containing
/// "permission denied" / "operation not permitted" on macOS (BPF) and Linux. So
/// we match the rendered message rather than a crate error variant, which is
/// robust across platforms.
fn classify_open_error(msg: &str) -> PcapStatus {
    let m = msg.to_ascii_lowercase();
    if m.contains("permission denied")
        || m.contains("operation not permitted")
        || m.contains("not permitted")
        || m.contains("eacces")
        || m.contains("eperm")
    {
        PcapStatus::PermissionDenied
    } else {
        PcapStatus::DeviceError
    }
}

/// Ordered candidate IPv4 addresses for the in-PCAP NFS server's virtual host on
/// a bridged subnet `network`/`prefix`. The GUI pings these in order and pre-fills
/// the first that doesn't answer; the [`PcapEngine`] ARP-probes similarly.
///
/// `.213` sits at offset 212 of a /24's 254 usable hosts (~83.5% up the range —
/// the "85% range"); that ratio is scaled to this subnet for the start, the scan
/// runs upward, and it never exceeds the 95% mark of the usable host range. The
/// network/broadcast addresses and any `reserved` ones (host IP, gateway, guest
/// IP, …) are excluded. At most 16 candidates are returned ("a few to ping").
pub fn nfs_ip_candidates(
    network: std::net::Ipv4Addr,
    prefix: u8,
    reserved: &[std::net::Ipv4Addr],
) -> Vec<std::net::Ipv4Addr> {
    use std::collections::HashSet;
    use std::net::Ipv4Addr;
    const MAX_CANDIDATES: usize = 16;
    // /31 and /32 (and the degenerate /0) have no usable host band to pick from.
    if prefix == 0 || prefix >= 31 {
        return Vec::new();
    }
    let host_bits = 32 - prefix as u32;
    let net = u32::from(network) & (u32::MAX << host_bits); // normalize to network addr
    let size = 1u32 << host_bits;
    let usable = size - 2;
    let first = net + 1;
    let broadcast = net + size - 1;
    let start_off = (usable as u64 * 212 / 254) as u32; // /24 → 212 → .213
    let cap_off = (usable as u64 * 95 / 100) as u32; // 95% of the usable range
    let reserved_set: HashSet<u32> = reserved
        .iter()
        .map(|a| u32::from(*a))
        .chain([net, broadcast])
        .collect();
    let mut out = Vec::new();
    let mut off = start_off;
    while off <= cap_off && out.len() < MAX_CANDIDATES {
        let Some(addr) = first.checked_add(off) else { break };
        if addr < broadcast && !reserved_set.contains(&addr) {
            out.push(Ipv4Addr::from(addr));
        }
        off += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn iface(name: &str, up: bool, running: bool, loopback: bool, has_addr: bool) -> NetInterface {
        NetInterface {
            name: name.to_string(),
            description: None,
            addresses: if has_addr { vec![IpAddr::V4(Ipv4Addr::new(192, 168, 0, 2))] } else { vec![] },
            up, running, loopback,
        }
    }

    fn sample() -> Vec<NetInterface> {
        vec![
            iface("eth0", true, true, false, true),
            iface(r"\Device\NPF_{ABC-123}", true, true, false, true),
            iface("lo", true, true, true, true),
        ]
    }

    #[test]
    fn selects_by_index() {
        let ifaces = sample();
        assert_eq!(resolve_interface(Some("1"), &ifaces).unwrap(), "eth0");
        // Index 2 resolves the awkward Windows-style name without the user typing it.
        assert_eq!(resolve_interface(Some("2"), &ifaces).unwrap(), r"\Device\NPF_{ABC-123}");
        // Whitespace tolerated.
        assert_eq!(resolve_interface(Some("  1 "), &ifaces).unwrap(), "eth0");
    }

    #[test]
    fn index_out_of_range_errors() {
        let ifaces = sample();
        assert!(resolve_interface(Some("0"), &ifaces).is_err());
        assert!(resolve_interface(Some("4"), &ifaces).is_err());
    }

    #[test]
    fn selects_by_exact_name() {
        let ifaces = sample();
        assert_eq!(resolve_interface(Some("eth0"), &ifaces).unwrap(), "eth0");
        assert_eq!(
            resolve_interface(Some(r"\Device\NPF_{ABC-123}"), &ifaces).unwrap(),
            r"\Device\NPF_{ABC-123}"
        );
    }

    #[test]
    fn unknown_name_passes_through_verbatim() {
        let ifaces = sample();
        assert_eq!(resolve_interface(Some("tap5"), &ifaces).unwrap(), "tap5");
    }

    #[test]
    fn auto_picks_first_up_running_nonloopback_with_addr() {
        let ifaces = sample();
        assert_eq!(resolve_interface(None, &ifaces).unwrap(), "eth0");
        assert_eq!(resolve_interface(Some(""), &ifaces).unwrap(), "eth0");
    }

    #[test]
    fn auto_pick_errors_when_only_loopback() {
        let ifaces = vec![iface("lo", true, true, true, true)];
        assert!(resolve_interface(None, &ifaces).is_err());
    }

    #[test]
    fn nfs_candidates_24_default_starts_at_213() {
        let c = nfs_ip_candidates(Ipv4Addr::new(192, 168, 1, 0), 24, &[]);
        assert_eq!(c[0], Ipv4Addr::new(192, 168, 1, 213), "default for /24 is .213");
        assert_eq!(c[1], Ipv4Addr::new(192, 168, 1, 214), "scans upward, contiguous");
        assert!(c.len() <= 16, "returns at most a few candidates");
        assert!(c.iter().all(|a| *a <= Ipv4Addr::new(192, 168, 1, 242)),
                "never above the 95% cap (.242 on a /24)");
    }

    #[test]
    fn nfs_candidates_skip_reserved() {
        let res = [Ipv4Addr::new(192, 168, 1, 213), Ipv4Addr::new(192, 168, 1, 214)];
        let c = nfs_ip_candidates(Ipv4Addr::new(192, 168, 1, 0), 24, &res);
        assert_eq!(c[0], Ipv4Addr::new(192, 168, 1, 215), "first free after reserved");
    }

    #[test]
    fn nfs_candidates_respect_95pct_cap() {
        // Reserve the whole band below .242 → only .242 remains, and .243+ are
        // never offered (the 95% hard cap).
        let res: Vec<_> = (213u8..=241).map(|h| Ipv4Addr::new(192, 168, 1, h)).collect();
        let c = nfs_ip_candidates(Ipv4Addr::new(192, 168, 1, 0), 24, &res);
        assert_eq!(c, vec![Ipv4Addr::new(192, 168, 1, 242)]);
    }

    #[test]
    fn nfs_candidates_normalize_and_edge_prefixes() {
        // Host bits in `network` are ignored (normalized to the network address).
        let c = nfs_ip_candidates(Ipv4Addr::new(10, 0, 0, 77), 24, &[]);
        assert_eq!(c[0], Ipv4Addr::new(10, 0, 0, 213));
        // /31, /32, /0 have no usable host band.
        assert!(nfs_ip_candidates(Ipv4Addr::new(10, 0, 0, 0), 31, &[]).is_empty());
        assert!(nfs_ip_candidates(Ipv4Addr::new(10, 0, 0, 0), 32, &[]).is_empty());
        assert!(nfs_ip_candidates(Ipv4Addr::new(10, 0, 0, 0), 0, &[]).is_empty());
    }

    #[test]
    fn classify_open_error_detects_permission() {
        // macOS BPF and Linux raw-socket permission messages map to the prompt.
        for m in [
            "activate device 'en0': (cannot open BPF device) /dev/bpf0: Permission denied",
            "libpcap error: socket: Operation not permitted",
            "You don't have permission to capture on that device (socket: Operation not permitted)",
            "open device 'eth0': EACCES",
        ] {
            assert_eq!(classify_open_error(m), PcapStatus::PermissionDenied, "{m}");
        }
        // Non-permission failures stay a generic device error (no elevation UI).
        for m in [
            "open device 'wlan9': No such device exists",
            "activate device 'en0': BIOCSETIF: Device not configured",
        ] {
            assert_eq!(classify_open_error(m), PcapStatus::DeviceError, "{m}");
        }
    }
}
