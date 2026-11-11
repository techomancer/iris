//! An adapter's connection to its network backend, made from `[network]`:
//! the two frame rings, their wake-ups and the backend at the far end. The
//! adapter takes one through `attach` and knows nothing of NAT or PCAP;
//! a new backend is one `NetBackend` implementation and a line in `Link::backend`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::config::{NetMode, NetworkConfig};

use super::stream::StreamBackend;
use super::{GatewayConfig, NatControl, NatEngine, NetBackend, Wake};

/// The adapter's ends of a link.
pub struct Rings {
    /// Frames the guest sends.
    pub to_net: rtrb::Producer<Vec<u8>>,
    /// Frames for the guest.
    pub from_net: rtrb::Consumer<Vec<u8>>,
    /// Told when a frame is on `to_net`.
    pub tx_wake: Arc<Wake>,
    /// Told by the backend when a frame is on `from_net`; the adapter notifies
    /// it too when the guest does something that can unblock a waiting frame.
    pub rx_wake: Arc<Wake>,
}

/// A backend running on its own thread, from [`Link::spawn`] until it is
/// dropped. Dropping it waits for the thread to end, so the backend's host
/// sockets (the NAT's port-forward listeners among them) are free once the
/// adapter that owned it lets go: an adapter restarting, or the next machine,
/// can bind them again.
pub struct Running {
    live: Arc<AtomicBool>,
    tx_wake: Arc<Wake>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Running {
    /// Ask the backend to end. It does on its next pass, which the wake-up
    /// starts at once.
    pub fn stop(&self) {
        self.live.store(false, Ordering::SeqCst);
        self.tx_wake.notify();
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stop();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// How a NIC runs its backend between its `start` and `stop`: the link the
/// next run starts, and the backend running now. Every NIC holds one, so a
/// machine reset behaves the same on all of them: `stop` signals the backend,
/// and the next `start` waits for it to end, which frees its host sockets,
/// before a fresh backend binds them.
#[derive(Default)]
pub struct Session {
    next: Option<Link>,
    running: Option<Running>,
}

impl Session {
    /// The link the next `start` runs. A NIC with none ignores `start`.
    pub fn attach(&mut self, link: Link) {
        self.next = Some(link);
    }

    /// End the previous run's backend and wait for it, run the attached link
    /// on a thread called `name`, and attach a fresh link for the run after.
    /// Returns the NIC's ends of the rings (`None` if it took them at attach)
    /// and the backend's status report if it has one; `None` with no link
    /// attached.
    pub fn start(&mut self, name: &str) -> Option<(Option<Rings>, Option<Arc<dyn NetBackend + Send + Sync>>)> {
        let mut link = self.next.take()?;
        match link.renew() {
            Ok(next) => self.next = Some(next),
            Err(e) => eprintln!("iris: {name}: network: {e}; down after this run"),
        }
        self.running = None;
        eprintln!("iris: {name}: backend = {}", link.describe());
        self.running = link.spawn(name);
        let reporter = link.reporter();
        Some((link.rings.take(), reporter))
    }

    /// Ask the running backend to end; it is waited for by the next `start`
    /// or when this is dropped.
    pub fn stop(&self) {
        if let Some(r) = &self.running {
            r.stop();
        }
    }
}

pub struct Link {
    /// The adapter's ends of the rings, until [`Link::take_rings`] or
    /// [`Link::into_rings`] hands them over.
    rings: Option<Rings>,
    tx_wake: Arc<Wake>,
    rx_wake: Arc<Wake>,
    /// The backend's ends of the rings, until the backend is built.
    far: Option<(rtrb::Consumer<Vec<u8>>, rtrb::Producer<Vec<u8>>)>,
    /// Guest frames processed, whichever the backend: the GUI's indicator.
    frames: Arc<AtomicU64>,
    /// The NAT's and PCAP bridge's controls and status (counters, tables,
    /// bridge state); renewed links share them. A stream segment leaves them idle.
    nat: Arc<NatControl>,
    /// The built backend, for its status report; only a backend with
    /// something to say is kept.
    reporter: Option<Arc<dyn NetBackend + Send + Sync>>,
    /// Cleared to stop the backend.
    running: Arc<AtomicBool>,
    config: GatewayConfig,
}

impl Link {
    /// `None` for `mode = "off"`: an adapter left unplugged.
    pub fn open(net: &NetworkConfig) -> Result<Option<Link>, String> {
        Self::with_gateway(GatewayConfig::from_network(net))
    }

    /// As `open`, for a gateway of the caller's own (a DaynaPort's subnet).
    pub fn with_gateway(config: GatewayConfig) -> Result<Option<Link>, String> {
        if config.mode == NetMode::Off {
            return Ok(None);
        }
        let frames = Arc::new(AtomicU64::new(0));
        Ok(Some(Self::build(config, frames.clone(), NatControl::counting(frames), Wake::new(), Wake::new())))
    }

    /// A link to the same backend with fresh rings and a backend that has not
    /// been built, sharing the wake-ups and controls: what a stopped adapter
    /// runs on its next start.
    pub fn renew(&self) -> Result<Link, String> {
        Ok(Self::build(self.config.clone(), self.frames.clone(), self.nat.clone(), self.tx_wake.clone(), self.rx_wake.clone()))
    }

    fn build(config: GatewayConfig, frames: Arc<AtomicU64>, nat: Arc<NatControl>, tx_wake: Arc<Wake>, rx_wake: Arc<Wake>) -> Link {
        let (to_net, tx_cons) = rtrb::RingBuffer::new(256);
        let (rx_prod, from_net) = rtrb::RingBuffer::new(256);
        let rings = Rings { to_net, from_net, tx_wake: tx_wake.clone(), rx_wake: rx_wake.clone() };
        Link { rings: Some(rings), tx_wake, rx_wake, far: Some((tx_cons, rx_prod)), frames, nat, reporter: None, running: Arc::new(AtomicBool::new(false)), config }
    }

    /// Which backend, for the adapter's log line.
    pub fn describe(&self) -> String {
        match self.config.mode {
            NetMode::Pcap if cfg!(feature = "pcap") => "PCAP (bridged)".into(),
            NetMode::Pcap => "NAT (software gateway): mode = \"pcap\" needs a build with --features pcap".into(),
            NetMode::Stream => match self.config.stream_endpoint() {
                Ok(ep) => format!("stream segment {ep}"),
                Err(e) => e,
            },
            _ => "NAT (software gateway)".into(),
        }
    }

    pub fn frames(&self) -> Arc<AtomicU64> {
        self.frames.clone()
    }

    pub fn nat(&self) -> Arc<NatControl> {
        self.nat.clone()
    }

    /// The wake-up the backend notifies when it has a frame for the guest.
    pub fn rx_wake(&self) -> Arc<Wake> {
        self.rx_wake.clone()
    }

    /// The built backend's status report, once it is built (`spawn` builds it):
    /// `None` for a backend that has none, as the NAT's tables are read from
    /// `nat` instead. The handle outlives the link, which `into_rings` consumes.
    pub fn reporter(&self) -> Option<Arc<dyn NetBackend + Send + Sync>> {
        self.reporter.clone()
    }

    /// The adapter's ends of the rings, once the backend has its own.
    pub fn into_rings(self) -> Rings {
        self.rings.expect("the link's rings were taken")
    }

    /// The adapter's ends of the rings, for an adapter that needs them before
    /// its backend runs. `None` once taken.
    pub fn take_rings(&mut self) -> Option<Rings> {
        self.rings.take()
    }

    /// The backend `config.mode` names: the NAT gateway, a PCAP bridge, which
    /// needs `--features pcap` (without it the NAT runs instead), or a stream
    /// segment (`validate` has parsed its peer). It is built here, not when
    /// the link is made: the NAT binds its host listeners when built. `None`
    /// if it was built already.
    fn backend(&mut self) -> Option<Box<dyn NetBackend>> {
        let (tx_cons, rx_prod) = self.far.take()?;
        let config = self.config.clone();
        let (rx_wake, tx_wake) = (self.rx_wake.clone(), self.tx_wake.clone());
        let (running, ctl) = (self.running.clone(), self.nat.clone());
        if config.mode == NetMode::Stream {
            let endpoint = config.stream_endpoint().expect("validated stream peer");
            let b = Arc::new(StreamBackend::new(endpoint, tx_cons, rx_prod, tx_wake, rx_wake, running, self.frames.clone()));
            self.reporter = Some(b.clone());
            return Some(Box::new(b));
        }
        #[cfg(feature = "pcap")]
        if config.mode == NetMode::Pcap {
            return Some(Box::new(super::net_pcap::PcapEngine::new(config, tx_cons, rx_prod, rx_wake, tx_wake, running, ctl)));
        }
        Some(Box::new(NatEngine::new(config, tx_cons, rx_prod, rx_wake, tx_wake, running, ctl)))
    }

    /// Run the backend on a thread called `name` until the returned handle
    /// stops it. `None` if the backend was built already.
    pub fn spawn(&mut self, name: &str) -> Option<Running> {
        let mut backend = self.backend()?;
        self.running.store(true, Ordering::SeqCst);
        let thread = std::thread::Builder::new().name(name.into()).spawn(move || backend.run()).expect("network backend thread");
        Some(Running { live: self.running.clone(), tx_wake: self.tx_wake.clone(), thread: Some(thread) })
    }

    /// A gateway with a TCP forward to guest port 23 from a host port held,
    /// not listening, by the returned socket, so no other test is handed the
    /// port meanwhile. Where a held port cannot be bound twice (not Unix) the
    /// socket is let go first.
    #[cfg(test)]
    pub(crate) fn forward_on_a_held_port() -> (Option<socket2::Socket>, u16, GatewayConfig) {
        use crate::config::{ForwardBind, ForwardProto, PortForwardConfig};
        let hold = socket2::Socket::new(socket2::Domain::IPV4, socket2::Type::STREAM, None).unwrap();
        #[cfg(unix)]
        hold.set_reuse_address(true).unwrap();
        hold.bind(&"127.0.0.1:0".parse::<std::net::SocketAddr>().unwrap().into()).unwrap();
        let port = hold.local_addr().unwrap().as_socket().unwrap().port();
        let hold = cfg!(unix).then_some(hold);
        let config = GatewayConfig {
            port_forwards: vec![PortForwardConfig { proto: ForwardProto::Tcp, host_port: port, guest_port: 23, bind: ForwardBind::Localhost }],
            ..GatewayConfig::default()
        };
        (hold, port, config)
    }

    /// A stop followed at once by a start, as a machine reset does, keeps the
    /// forward on `port` (from [`Link::forward_on_a_held_port`]): the old
    /// backend must let go of its listeners before the new one binds. `start`
    /// and `stop` are the adapter's own.
    #[cfg(test)]
    pub(crate) fn assert_restart_keeps_the_forward(port: u16, start: &dyn Fn(), stop: &dyn Fn()) {
        let listening = || std::net::TcpStream::connect(("127.0.0.1", port)).is_ok();
        start();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !listening() {
            assert!(std::time::Instant::now() < deadline, "the forward never listened");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        stop();
        start();
        // Longer than the old backend takes to notice it was stopped.
        std::thread::sleep(std::time::Duration::from_millis(500));
        let up = listening();
        stop();
        assert!(up, "the forward is gone after a restart");
    }

    /// Rings to nowhere, and their far ends, for adapter tests.
    #[cfg(test)]
    pub fn loopback() -> (Link, rtrb::Consumer<Vec<u8>>, rtrb::Producer<Vec<u8>>) {
        let mut link = Self::build(GatewayConfig::default(), Arc::default(), NatControl::new(), Wake::new(), Wake::new());
        let (tx_cons, rx_prod) = link.far.take().expect("a new link has its far ends");
        (link, tx_cons, rx_prod)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A renewed link has fresh rings and a backend that has not been built,
    /// and shares the wake-ups and controls.
    #[test]
    fn a_renewed_link_is_fresh() {
        let first = Link::open(&NetworkConfig::default()).unwrap().unwrap();
        let next = first.renew().unwrap();
        assert!(Arc::ptr_eq(&first.nat, &next.nat) && Arc::ptr_eq(&first.tx_wake, &next.tx_wake));
        assert!(next.far.is_some() && !Arc::ptr_eq(&first.running, &next.running));
    }

    #[test]
    fn a_backend_is_built_once() {
        let mut link = Link::open(&NetworkConfig::default()).unwrap().unwrap();
        assert!(link.backend().is_some());
        assert!(link.backend().is_none());
    }

    /// A stream link names its peer, reports once its backend is built, and
    /// is renewed without one.
    #[test]
    fn a_stream_link_reports_its_segment() {
        let net = NetworkConfig { mode: NetMode::Stream, stream: Some("tcp:127.0.0.1:1".into()), ..NetworkConfig::default() };
        let mut link = Link::open(&net).unwrap().unwrap();
        assert_eq!(link.describe(), "stream segment tcp:127.0.0.1:1");
        assert!(link.reporter().is_none());
        link.backend().unwrap();
        let mut text = Vec::new();
        link.reporter().expect("a stream reports").report(&mut text).unwrap();
        assert!(String::from_utf8(text).unwrap().contains("waiting for the peer"));
        assert!(link.renew().unwrap().reporter().is_none());
    }

    #[test]
    fn an_unplugged_network_makes_no_link() {
        let net = NetworkConfig { mode: NetMode::Off, ..NetworkConfig::default() };
        assert!(Link::open(&net).unwrap().is_none());
    }
}
