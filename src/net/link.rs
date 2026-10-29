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

pub struct Link {
    rings: Rings,
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
        let r = &self.rings;
        Ok(Self::build(self.config.clone(), self.frames.clone(), self.nat.clone(), r.tx_wake.clone(), r.rx_wake.clone()))
    }

    fn build(config: GatewayConfig, frames: Arc<AtomicU64>, nat: Arc<NatControl>, tx_wake: Arc<Wake>, rx_wake: Arc<Wake>) -> Link {
        let (to_net, tx_cons) = rtrb::RingBuffer::new(256);
        let (rx_prod, from_net) = rtrb::RingBuffer::new(256);
        let rings = Rings { to_net, from_net, tx_wake, rx_wake };
        Link { rings, far: Some((tx_cons, rx_prod)), frames, nat, reporter: None, running: Arc::new(AtomicBool::new(false)), config }
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

    /// The flag the backend watches; clear it to stop the backend.
    pub fn running(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    /// The wake-up the backend notifies when it has a frame for the guest.
    pub fn rx_wake(&self) -> Arc<Wake> {
        self.rings.rx_wake.clone()
    }

    /// The built backend's status report, once it is built (`spawn` builds it):
    /// `None` for a backend that has none, as the NAT's tables are read from
    /// `nat` instead. The handle outlives the link, which `into_rings` consumes.
    pub fn reporter(&self) -> Option<Arc<dyn NetBackend + Send + Sync>> {
        self.reporter.clone()
    }

    /// The adapter's ends of the rings, once the backend has its own.
    pub fn into_rings(self) -> Rings {
        self.rings
    }

    /// The backend `config.mode` names: the NAT gateway, a PCAP bridge, which
    /// needs `--features pcap` (without it the NAT runs instead), or a stream
    /// segment (`validate` has parsed its peer). It is built here, not when
    /// the link is made: the NAT binds its host listeners when built. `None`
    /// if it was built already.
    pub fn backend(&mut self) -> Option<Box<dyn NetBackend>> {
        let (tx_cons, rx_prod) = self.far.take()?;
        let config = self.config.clone();
        let (rx_wake, tx_wake) = (self.rings.rx_wake.clone(), self.rings.tx_wake.clone());
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

    /// Run the backend on a thread called `name` until `running` clears. The
    /// thread is detached: a stopped adapter does not wait for it, and the
    /// next start runs a renewed link on a thread of its own.
    pub fn spawn(&mut self, name: &str) {
        let Some(mut backend) = self.backend() else { return };
        self.running.store(true, Ordering::SeqCst);
        std::thread::Builder::new().name(name.into()).spawn(move || backend.run()).expect("network backend thread");
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
        assert!(Arc::ptr_eq(&first.nat, &next.nat) && Arc::ptr_eq(&first.rings.tx_wake, &next.rings.tx_wake));
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
