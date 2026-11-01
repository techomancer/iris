//! virtio-console / virtio-serial (device ID 3, virtio 1.x §5.3).
//!
//! With `VIRTIO_CONSOLE_F_MULTIPORT` the device is a bus of named ports,
//! each a pair of queues the guest opens like a character device, plus a
//! control queue pair that announces the ports and carries open/close events.
//! A port can be a login console, or a private pipe to a program on the host
//! (a guest agent, a file transfer): the host end is a [`PortBackend`].
//!
//! Queue layout (multiport): 0 receiveq(port 0), 1 transmitq(port 0),
//! 2 control receiveq, 3 control transmitq, then 2p+2 / 2p+3 for port p ≥ 1.
//! A driver that does not accept MULTIPORT uses queues 0 and 1 for port 0
//! alone, as a plain console.
//!
//! Naming: receive and transmit are the *guest's* point of view. Bytes the
//! host sends arrive on `receiveq`; bytes the guest writes leave on
//! `transmitq`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use super::backend::PortBackend;
use super::{Cx, VirtioDevice, Waker};

pub const DEVICE_ID: u32 = 3;
pub const F_SIZE: u64 = 1 << 0;
pub const F_MULTIPORT: u64 = 1 << 1;
pub const F_EMERG_WRITE: u64 = 1 << 2;

// Control events (§5.3.6.1).
const EV_DEVICE_READY: u16 = 0;
const EV_DEVICE_ADD: u16 = 1;
#[allow(dead_code)]
const EV_DEVICE_REMOVE: u16 = 2;
const EV_PORT_READY: u16 = 3;
const EV_CONSOLE_PORT: u16 = 4;
#[allow(dead_code)]
const EV_RESIZE: u16 = 5;
const EV_PORT_OPEN: u16 = 6;
const EV_PORT_NAME: u16 = 7;

/// Host input held per port before the guest has room for it.
const STAGE_CAP: usize = 64 * 1024;

/// One port as configured: its name (how the guest finds it, e.g.
/// `/dev/virtio-ports/<name>` on Linux), whether it is a console, and what
/// the host end is.
pub struct PortConfig {
    pub name: Option<String>,
    pub console: bool,
    pub backend: Arc<dyn PortBackend>,
}

/// Host-side state a port shares with the input pump thread.
struct PortIo {
    backend: Arc<dyn PortBackend>,
    staged: Mutex<VecDeque<u8>>,
}

struct Port {
    io: Arc<PortIo>,
    name: Option<String>,
    console: bool,
    /// The guest sent PORT_READY.
    ready: bool,
    /// The guest has the port open (PORT_OPEN 1), or there is no control
    /// channel to say so.
    guest_open: bool,
    /// Last host-connected state told to the guest.
    host_open_sent: bool,
}

pub struct Console {
    ports: Vec<Port>,
    multiport: bool,
    /// Control messages for the guest, waiting for a control receive buffer.
    ctrl_out: VecDeque<Vec<u8>>,
    waker: Arc<Waker>,
    stop: Arc<AtomicBool>,
    pump: Option<JoinHandle<()>>,
}

fn ctrl(id: u32, event: u16, value: u16) -> Vec<u8> {
    let mut m = Vec::with_capacity(8);
    m.extend_from_slice(&id.to_le_bytes());
    m.extend_from_slice(&event.to_le_bytes());
    m.extend_from_slice(&value.to_le_bytes());
    m
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Rx(usize),
    Tx(usize),
    CtrlRx,
    CtrlTx,
}

impl Console {
    pub fn new(ports: Vec<PortConfig>, waker: Arc<Waker>) -> Self {
        assert!(!ports.is_empty(), "a console has at least one port");
        let ports = ports
            .into_iter()
            .map(|p| Port {
                io: Arc::new(PortIo { backend: p.backend, staged: Mutex::new(VecDeque::new()) }),
                name: p.name,
                console: p.console,
                ready: false,
                guest_open: false,
                host_open_sent: false,
            })
            .collect();
        Self { ports, multiport: false, ctrl_out: VecDeque::new(), waker, stop: Arc::new(AtomicBool::new(false)), pump: None }
    }

    /// Start the thread that reads the ports' host ends and wakes the
    /// service thread when input arrives. It polls every 5 ms while there is
    /// traffic and backs off to 50 ms when idle: the backends are
    /// non-blocking sockets with nothing to wait on.
    pub fn start_pump(&mut self) {
        if self.pump.is_some() {
            return;
        }
        let ios: Vec<_> = self.ports.iter().map(|p| p.io.clone()).collect();
        let waker = self.waker.clone();
        let stop = self.stop.clone();
        self.pump = Some(
            std::thread::Builder::new()
                .name("virtio-console".into())
                .spawn(move || {
                    let mut busy_until = Instant::now();
                    let mut connected: Vec<bool> = ios.iter().map(|_| false).collect();
                    while !stop.load(Ordering::Acquire) {
                        let mut activity = false;
                        for (i, io) in ios.iter().enumerate() {
                            activity |= Self::pump_port(io);
                            let c = io.backend.connected();
                            if c != connected[i] {
                                connected[i] = c;
                                activity = true;
                            }
                        }
                        if activity {
                            busy_until = Instant::now() + Duration::from_secs(2);
                            waker.wake();
                        }
                        let nap = if Instant::now() < busy_until { 5 } else { 50 };
                        std::thread::sleep(Duration::from_millis(nap));
                    }
                })
                .expect("spawn virtio-console pump"),
        );
    }

    /// Move whatever the backend has into the staging queue. True if any.
    fn pump_port(io: &PortIo) -> bool {
        let mut got = false;
        let mut buf = [0u8; 4096];
        while io.staged.lock().len() < STAGE_CAP {
            let n = io.backend.read(&mut buf);
            if n == 0 {
                break;
            }
            io.staged.lock().extend(buf[..n].iter().copied());
            got = true;
        }
        got
    }

    fn role(&self, q: usize) -> Option<Role> {
        match q {
            0 => Some(Role::Rx(0)),
            1 => Some(Role::Tx(0)),
            2 | 3 if !self.multiport => None,
            2 => Some(Role::CtrlRx),
            3 => Some(Role::CtrlTx),
            _ if !self.multiport => None,
            _ => {
                let p = (q - 2) / 2;
                (p < self.ports.len()).then(|| if q % 2 == 0 { Role::Rx(p) } else { Role::Tx(p) })
            }
        }
    }

    fn rx_queue(p: usize) -> usize {
        if p == 0 { 0 } else { 2 * p + 2 }
    }

    /// Guest output: hand every chain on `q` to the port's backend.
    fn drain_tx(&mut self, p: usize, q: usize, cx: &mut Cx) {
        while let Some(chain) = cx.queues[q].pop(cx.mem) {
            if let Some(data) = chain.read_all(cx.mem) {
                if !data.is_empty() {
                    self.ports[p].io.backend.write(&data);
                }
            }
            cx.used(q, chain.head, 0);
        }
    }

    /// Guest control messages (driver → device).
    fn drain_ctrl_tx(&mut self, cx: &mut Cx) {
        while let Some(chain) = cx.queues[3].pop(cx.mem) {
            if let Some(m) = chain.read_all(cx.mem) {
                if m.len() >= 8 {
                    let id = u32::from_le_bytes(m[0..4].try_into().unwrap());
                    let event = u16::from_le_bytes(m[4..6].try_into().unwrap());
                    let value = u16::from_le_bytes(m[6..8].try_into().unwrap());
                    self.control(id, event, value);
                }
            }
            cx.used(3, chain.head, 0);
        }
    }

    fn control(&mut self, id: u32, event: u16, value: u16) {
        match event {
            EV_DEVICE_READY if value == 1 => {
                for p in 0..self.ports.len() {
                    self.ctrl_out.push_back(ctrl(p as u32, EV_DEVICE_ADD, 1));
                }
            }
            EV_PORT_READY if value == 1 => {
                let Some(port) = self.ports.get_mut(id as usize) else { return };
                port.ready = true;
                if port.console {
                    self.ctrl_out.push_back(ctrl(id, EV_CONSOLE_PORT, 1));
                }
                if let Some(name) = &port.name {
                    let mut m = ctrl(id, EV_PORT_NAME, 1);
                    m.extend_from_slice(name.as_bytes());
                    self.ctrl_out.push_back(m);
                }
                // `poll` tells the guest about the host end once the port is ready.
                port.host_open_sent = false;
            }
            EV_PORT_OPEN => {
                if let Some(port) = self.ports.get_mut(id as usize) {
                    port.guest_open = value != 0;
                }
            }
            _ => {}
        }
    }
}

impl Drop for Console {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(h) = self.pump.take() {
            let _ = h.join();
        }
    }
}

impl VirtioDevice for Console {
    fn device_id(&self) -> u32 {
        DEVICE_ID
    }

    fn features(&self) -> u64 {
        F_MULTIPORT | F_EMERG_WRITE
    }

    fn num_queues(&self, _features: u64) -> usize {
        2 + 2 * self.ports.len()
    }

    fn config_len(&self) -> usize {
        12
    }

    /// `cols` u16, `rows` u16, `max_nr_ports` u32, `emerg_wr` u32; little-endian.
    fn config_read(&self, off: usize, buf: &mut [u8]) {
        let mut cfg = [0u8; 12];
        cfg[4..8].copy_from_slice(&(self.ports.len() as u32).to_le_bytes());
        let end = (off + buf.len()).min(cfg.len());
        buf[..end - off].copy_from_slice(&cfg[off..end]);
    }

    /// `emerg_wr`: a character the guest writes before its queues work.
    fn config_write(&mut self, off: usize, data: &[u8]) {
        if off == 8 && !data.is_empty() {
            self.ports[0].io.backend.write(&data[..1]);
        }
    }

    fn reset(&mut self) {
        self.multiport = false;
        self.ctrl_out.clear();
        for p in self.ports.iter_mut() {
            p.ready = false;
            p.guest_open = false;
            p.host_open_sent = false;
        }
    }

    fn activate(&mut self, features: u64) {
        self.multiport = features & F_MULTIPORT != 0;
        // A plain console has no control channel to say the guest opened it.
        if !self.multiport {
            self.ports[0].guest_open = true;
        }
    }

    fn notify(&mut self, q: usize, cx: &mut Cx) {
        match self.role(q) {
            Some(Role::Tx(p)) => self.drain_tx(p, q, cx),
            Some(Role::CtrlTx) => self.drain_ctrl_tx(cx),
            // Receive buffers: `poll` fills them.
            Some(Role::Rx(_)) | Some(Role::CtrlRx) | None => {}
        }
    }

    fn poll(&mut self, cx: &mut Cx) {
        // A guest that is mid-way through a control exchange may have queued
        // replies in the same pass.
        if self.multiport && cx.queues.len() > 3 {
            self.drain_ctrl_tx(cx);
        }

        for p in 0..self.ports.len() {
            if p > 0 && !self.multiport {
                break;
            }
            let io = self.ports[p].io.clone();
            Self::pump_port(&io);

            // Tell the guest when the host end appeared or went away.
            if self.multiport && self.ports[p].ready {
                let c = io.backend.connected();
                if c != self.ports[p].host_open_sent {
                    self.ports[p].host_open_sent = c;
                    self.ctrl_out.push_back(ctrl(p as u32, EV_PORT_OPEN, c as u16));
                }
            }

            // Host input into the guest's receive buffers.
            if !self.ports[p].guest_open {
                continue;
            }
            let q = Self::rx_queue(p);
            while !io.staged.lock().is_empty() {
                let Some(chain) = cx.queues[q].pop(cx.mem) else { break };
                let take = chain.writable_len();
                let data: Vec<u8> = {
                    let mut s = io.staged.lock();
                    let n = take.min(s.len());
                    s.drain(..n).collect()
                };
                let written = chain.write_all(cx.mem, &data).unwrap_or(0);
                cx.used(q, chain.head, written as u32);
            }
        }

        // Control messages for the guest.
        if self.multiport {
            while !self.ctrl_out.is_empty() {
                let Some(chain) = cx.queues[2].pop(cx.mem) else { break };
                let m = self.ctrl_out.pop_front().unwrap();
                let written = chain.write_all(cx.mem, &m).unwrap_or(0);
                cx.used(2, chain.head, written as u32);
            }
        }
    }
}

/// The default port when `[virtio_serial]` lists none.
pub const DEFAULT_TCP_PORT: u16 = 8890;

/// Build the console device a `[virtio_serial]` section describes, with its
/// host ends bound and its input pump running. A port whose socket cannot be
/// made is left wired to nothing, with the reason on stderr.
pub fn from_config(cfg: &crate::config::VirtioSerialConfig) -> super::VirtioCore {
    use super::backend::{tcp_port, unix_port, NullPort};
    let default_port = [crate::config::VirtioSerialPort {
        console: true,
        tcp_port: Some(DEFAULT_TCP_PORT),
        ..Default::default()
    }];
    let specs: &[crate::config::VirtioSerialPort] = if cfg.ports.is_empty() { &default_port } else { &cfg.ports };
    let ports = specs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let label = p.name.clone().unwrap_or_else(|| format!("port {i}"));
            let backend: Arc<dyn PortBackend> = if let Some(path) = &p.socket {
                match unix_port(path) {
                    Ok(b) => {
                        eprintln!("iris: virtio-serial {label} on socket {path}");
                        b
                    }
                    Err(e) => {
                        eprintln!("iris: virtio-serial {label}: socket {path}: {e}");
                        Arc::new(NullPort)
                    }
                }
            } else if let Some(port) = p.tcp_port {
                match tcp_port(port) {
                    Ok((b, bound)) => {
                        eprintln!("iris: virtio-serial {label} on tcp 127.0.0.1:{bound}");
                        b
                    }
                    Err(e) => {
                        eprintln!("iris: virtio-serial {label}: tcp port {port}: {e}");
                        Arc::new(NullPort)
                    }
                }
            } else {
                Arc::new(NullPort)
            };
            PortConfig { name: p.name.clone(), console: p.console, backend }
        })
        .collect();
    let waker = Waker::new();
    let mut console = Console::new(ports, waker.clone());
    console.start_pump();
    super::VirtioCore::new(Box::new(console), waker)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::virtio::backend::MemPort;
    use crate::dev::virtio::{
        GuestMemory, VecMemory, VirtioCore, F_VERSION_1, STATUS_ACKNOWLEDGE, STATUS_DRIVER, STATUS_DRIVER_OK, STATUS_FEATURES_OK,
    };

    const RING: u16 = 8;
    /// Each queue gets 0x4000 bytes: descriptors, avail, used, then buffers.
    fn base(q: usize) -> u64 {
        0x1_0000 + 0x4000 * q as u64
    }

    struct Rig {
        core: VirtioCore,
        mem: Arc<VecMemory>,
        ports: Vec<Arc<MemPort>>,
        next_desc: Vec<u16>,
        avail_idx: Vec<u16>,
        used_seen: Vec<u16>,
    }

    impl Rig {
        fn new(specs: &[(Option<&str>, bool)], features: u64) -> Self {
            let ports: Vec<_> = specs.iter().map(|_| MemPort::new()).collect();
            let waker = Waker::new();
            let cfgs = specs
                .iter()
                .zip(&ports)
                .map(|((n, c), p)| PortConfig { name: n.map(String::from), console: *c, backend: p.clone() })
                .collect();
            let core = VirtioCore::new(Box::new(Console::new(cfgs, waker.clone())), waker);
            let mem = Arc::new(VecMemory::new(0x1_0000 + 0x4000 * 16));
            core.connect_dma(mem.clone());
            let nq = core.num_queues();
            for q in 0..nq {
                core.with_queue(q, |s| {
                    s.size = RING;
                    s.ready = true;
                    s.desc = base(q);
                    s.avail = base(q) + 0x200;
                    s.used = base(q) + 0x400;
                });
            }
            core.set_driver_features_word(0, features as u32);
            core.set_driver_features_word(1, (F_VERSION_1 >> 32) as u32);
            core.set_status(STATUS_ACKNOWLEDGE | STATUS_DRIVER);
            core.set_status(STATUS_FEATURES_OK);
            assert_ne!(core.status() & STATUS_FEATURES_OK, 0, "features refused");
            core.set_status(STATUS_DRIVER_OK);
            Self { core, mem, ports, next_desc: vec![0; nq], avail_idx: vec![0; nq], used_seen: vec![0; nq] }
        }

        /// Queue a buffer on `q`; `data` for the device to read, or `len`
        /// writable bytes. Returns its data address.
        fn give(&mut self, q: usize, data: Option<&[u8]>, len: u32) -> u64 {
            let d = self.next_desc[q];
            self.next_desc[q] += 1;
            let buf = base(q) + 0x1000 + 0x200 * d as u64;
            let (flags, n) = match data {
                Some(b) => {
                    self.mem.write(buf, b);
                    (0u16, b.len() as u32)
                }
                None => (2u16, len),
            };
            let mut raw = [0u8; 16];
            raw[0..8].copy_from_slice(&buf.to_le_bytes());
            raw[8..12].copy_from_slice(&n.to_le_bytes());
            raw[12..14].copy_from_slice(&flags.to_le_bytes());
            self.mem.write(base(q) + 16 * d as u64, &raw);
            let slot = self.avail_idx[q] % RING;
            self.mem.write(base(q) + 0x200 + 4 + 2 * slot as u64, &d.to_le_bytes());
            self.avail_idx[q] += 1;
            self.mem.write(base(q) + 0x200 + 2, &self.avail_idx[q].to_le_bytes());
            buf
        }

        fn kick(&self, q: usize) {
            self.core.notify(q);
        }

        /// New used entries on `q` since the last call: (id, len).
        fn used(&mut self, q: usize) -> Vec<(u32, u32)> {
            let mut b = [0u8; 2];
            self.mem.read(base(q) + 0x400 + 2, &mut b);
            let idx = u16::from_le_bytes(b);
            let mut out = Vec::new();
            while self.used_seen[q] != idx {
                let mut e = [0u8; 8];
                self.mem.read(base(q) + 0x400 + 4 + 8 * (self.used_seen[q] % RING) as u64, &mut e);
                out.push((u32::from_le_bytes(e[0..4].try_into().unwrap()), u32::from_le_bytes(e[4..8].try_into().unwrap())));
                self.used_seen[q] += 1;
            }
            out
        }

        fn read(&self, addr: u64, n: usize) -> Vec<u8> {
            let mut v = vec![0; n];
            self.mem.read(addr, &mut v);
            v
        }
    }

    #[test]
    fn plain_console_moves_bytes_both_ways() {
        let mut r = Rig::new(&[(None, true)], 0);
        // Guest -> host.
        r.give(1, Some(b"login: "), 0);
        r.kick(1);
        assert_eq!(r.ports[0].take_tx(), b"login: ");
        assert_eq!(r.used(1), vec![(0, 0)]);

        // Host -> guest, into a buffer that was waiting.
        let buf = r.give(0, None, 16);
        r.kick(0);
        assert!(r.used(0).is_empty());
        r.ports[0].push_rx(b"root\r");
        r.core.notify(0); // a host wake would do the same
        assert_eq!(r.used(0), vec![(0, 5)]);
        assert_eq!(r.read(buf, 5), b"root\r");
        assert_ne!(r.core.isr() & 1, 0, "used-buffer interrupt");
    }

    #[test]
    fn input_longer_than_a_buffer_spans_buffers() {
        let mut r = Rig::new(&[(None, true)], 0);
        let a = r.give(0, None, 4);
        let b = r.give(0, None, 4);
        r.ports[0].push_rx(b"abcdefg");
        r.kick(0);
        assert_eq!(r.used(0), vec![(0, 4), (1, 3)]);
        assert_eq!(r.read(a, 4), b"abcd");
        assert_eq!(r.read(b, 3), b"efg");
    }

    #[test]
    fn multiport_announces_ports_and_names() {
        let mut r = Rig::new(&[(None, true), (Some("org.iris.agent.0"), false)], F_MULTIPORT);
        // Control receive buffers for the device's messages.
        let c: Vec<u64> = (0..6).map(|_| r.give(2, None, 64)).collect();
        r.kick(2);

        // The driver is ready: the device adds both ports.
        r.give(3, Some(&ctrl(0, EV_DEVICE_READY, 1)), 0);
        r.kick(3);
        let u = r.used(2);
        assert_eq!(u.len(), 2);
        assert_eq!(r.read(c[0], 8), ctrl(0, EV_DEVICE_ADD, 1));
        assert_eq!(r.read(c[1], 8), ctrl(1, EV_DEVICE_ADD, 1));

        // Port 1 comes up: the device names it, then reports the host end
        // (MemPort is always "connected").
        r.give(3, Some(&ctrl(1, EV_PORT_READY, 1)), 0);
        r.kick(3);
        let u = r.used(2);
        assert_eq!(u.len(), 2);
        let mut name = ctrl(1, EV_PORT_NAME, 1);
        name.extend_from_slice(b"org.iris.agent.0");
        assert_eq!(r.read(c[2], name.len()), name);
        assert_eq!(u[0].1 as usize, name.len());
        assert_eq!(r.read(c[3], 8), ctrl(1, EV_PORT_OPEN, 1));

        // Port 0 is the console.
        r.give(3, Some(&ctrl(0, EV_PORT_READY, 1)), 0);
        r.kick(3);
        assert_eq!(r.read(c[4], 8), ctrl(0, EV_CONSOLE_PORT, 1));
    }

    #[test]
    fn port_data_waits_for_the_guest_to_open_it() {
        let mut r = Rig::new(&[(None, true), (Some("agent"), false)], F_MULTIPORT);
        let rx = r.give(4, None, 32); // port 1 receiveq
        r.ports[1].push_rx(b"hello");
        r.kick(4);
        assert!(r.used(4).is_empty(), "port is closed");

        r.give(3, Some(&ctrl(1, EV_PORT_OPEN, 1)), 0);
        r.kick(3);
        assert_eq!(r.used(4), vec![(0, 5)]);
        assert_eq!(r.read(rx, 5), b"hello");

        // And guest output on port 1 goes to port 1's backend only.
        r.give(5, Some(b"reply"), 0);
        r.kick(5);
        assert_eq!(r.ports[1].take_tx(), b"reply");
        assert!(r.ports[0].take_tx().is_empty());
    }

    #[test]
    fn config_space_and_emergency_write() {
        let r = Rig::new(&[(None, true), (Some("a"), false), (Some("b"), false)], F_MULTIPORT | F_EMERG_WRITE);
        let mut cfg = [0u8; 12];
        r.core.config_read(0, &mut cfg);
        assert_eq!(u32::from_le_bytes(cfg[4..8].try_into().unwrap()), 3);
        r.core.config_write(8, &[b'!', 0, 0, 0]);
        assert_eq!(r.ports[0].take_tx(), b"!");
        // Past the end reads zero.
        let mut tail = [0xffu8; 4];
        r.core.config_read(10, &mut tail);
        assert_eq!(tail, [0, 0, 0, 0]);
    }

    #[test]
    fn driver_without_version_1_is_refused() {
        let core = {
            let w = Waker::new();
            VirtioCore::new(
                Box::new(Console::new(vec![PortConfig { name: None, console: true, backend: MemPort::new() }], w.clone())),
                w,
            )
        };
        core.set_driver_features_word(0, 0);
        core.set_status(STATUS_ACKNOWLEDGE | STATUS_DRIVER);
        core.set_status(STATUS_FEATURES_OK);
        assert_eq!(core.status() & STATUS_FEATURES_OK, 0);
    }

    #[test]
    fn reset_returns_the_device_to_idle() {
        let mut r = Rig::new(&[(None, true)], F_MULTIPORT);
        r.give(1, Some(b"x"), 0);
        r.kick(1);
        assert_ne!(r.core.isr(), 0);
        r.core.set_status(0);
        assert_eq!(r.core.status(), 0);
        assert_eq!(r.core.isr(), 0);
        assert!(!r.core.with_queue(1, |q| q.ready).unwrap());
    }
}
