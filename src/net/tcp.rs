//! A guest-facing TCP endpoint. The guest's connection terminates here; what
//! is on the other side is a [`Backend`]: a host socket (outbound NAT, inbound
//! forwards) or an in-core [`TcpService`].
//!
//! Deliberately small: no congestion control (the "network" is a ring buffer
//! in the same process), go-back-N retransmission with exponential backoff,
//! in-order receive only (the guest retransmits anything past a gap), and a
//! zero-window probe. The receive buffer toward the backend bounds what we
//! ACK, so a slow host reader can never lose the guest's data.

use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, TcpStream};
use std::time::{Duration, Instant};

use socket2::{Domain, Socket, Type};

use super::packet::{Ipv4Route, MacAddr, Tcp, TcpFlags, TcpHeader, TCP_MSS};

/// Our receive buffer toward the backend, and so the largest window we
/// advertise (no window scaling: at most 65535).
pub(super) const RCV_BUF: usize = 65535;
/// Most unacknowledged data we hold for the guest (its window caps it too).
const SND_BUF: usize = 65535;
/// The guest's MSS when its SYN carries none (RFC 879).
const DEFAULT_MSS: u16 = 536;
const RTO_INITIAL: Duration = Duration::from_millis(200);
const RTO_MAX: Duration = Duration::from_secs(4);
/// Give up when the guest has not acknowledged anything for this long.
const GIVE_UP: Duration = Duration::from_secs(60);
/// A host connect that has neither succeeded nor failed by now is refused.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Absorb stray segments for a closed connection this long.
pub(super) const TIME_WAIT: Duration = Duration::from_secs(4);
/// A half-closed connection that stops moving is dropped after this long.
const HALF_CLOSED_IDLE: Duration = Duration::from_secs(120);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum State {
    /// The guest's SYN is in; the backend is still connecting.
    Connecting,
    /// We sent the guest a SYN (inbound forward); waiting for its SYN-ACK.
    SynSent,
    /// We sent the guest a SYN-ACK; waiting for its ACK.
    SynReceived,
    Established,
    /// Both sides are done; absorbing stragglers until removed.
    Closed,
}

/// What a guest connection is relayed to.
pub(super) enum Backend {
    Host(HostConn),
    Service(ServiceConn),
}

/// A non-blocking host socket, possibly still connecting.
pub(super) struct HostConn {
    sock: Option<Socket>,      // while connecting
    stream: Option<TcpStream>, // once connected
    started: Instant,
}

impl HostConn {
    /// Start a non-blocking connect to `addr`.
    pub(super) fn connect(addr: SocketAddr) -> io::Result<Self> {
        let sock = Socket::new(Domain::IPV4, Type::STREAM, None)?;
        sock.set_nonblocking(true)?;
        let _ = sock.set_tcp_nodelay(true);
        match sock.connect(&addr.into()) {
            Ok(()) => {}
            Err(e) if in_progress(&e) => {}
            Err(e) => return Err(e),
        }
        Ok(Self { sock: Some(sock), stream: None, started: Instant::now() })
    }

    /// An already-connected socket (an accepted forward connection).
    pub(super) fn connected(stream: TcpStream) -> Self {
        let _ = stream.set_nonblocking(true);
        let _ = stream.set_nodelay(true);
        Self { sock: None, stream: Some(stream), started: Instant::now() }
    }

    /// Ready(Ok) once connected, Ready(Err) if it failed, Pending meanwhile.
    fn poll_connect(&mut self) -> Option<io::Result<()>> {
        if self.stream.is_some() {
            return Some(Ok(()));
        }
        let sock = self.sock.as_ref()?;
        if let Ok(Some(e)) | Err(e) = sock.take_error().map(|o| o.map(Into::into)) {
            return Some(Err(e));
        }
        match sock.peer_addr() {
            Ok(_) => {
                let stream: TcpStream = self.sock.take().unwrap().into();
                self.stream = Some(stream);
                Some(Ok(()))
            }
            Err(_) if self.started.elapsed() > CONNECT_TIMEOUT => {
                Some(Err(io::Error::new(io::ErrorKind::TimedOut, "connect timed out")))
            }
            Err(_) => None,
        }
    }
}

fn in_progress(e: &io::Error) -> bool {
    e.kind() == io::ErrorKind::WouldBlock || e.raw_os_error() == Some(libc_einprogress())
}

#[cfg(unix)]
fn libc_einprogress() -> i32 {
    libc::EINPROGRESS
}
#[cfg(not(unix))]
fn libc_einprogress() -> i32 {
    10036 // WSAEINPROGRESS
}

/// An in-core service speaking over TCP.
pub(super) trait TcpService: Send {
    /// Bytes to send as soon as the connection is open.
    fn greeting(&mut self) -> Vec<u8> {
        Vec::new()
    }
    /// Bytes from the guest; returns bytes to send back.
    fn input(&mut self, data: &[u8]) -> Vec<u8>;
    /// Output the service produces on its own, at most `max` bytes, asked for
    /// whenever the guest's window has room and nothing is queued. This is how
    /// a service streams something larger than it wants to hold in memory.
    /// Empty means nothing right now (`done` says whether there will be more).
    fn pull(&mut self, _max: usize) -> Vec<u8> {
        Vec::new()
    }
    /// True when the service has nothing more to say: we FIN once the output drains.
    fn done(&self) -> bool;
    /// The guest closed its side.
    fn guest_closed(&mut self) {}
    /// A second connection the service wants opened toward the guest, polled
    /// after each input (rsh's stderr channel). Returned at most once.
    fn side_connect(&mut self) -> Option<SideConnect> {
        None
    }
}

/// A connection a [`TcpService`] asks the gateway to open to the guest, from
/// the same gateway address the guest connected to, and the service that runs
/// on it.
pub(super) struct SideConnect {
    pub guest_port: u16,
    pub service: Box<dyn TcpService>,
}

pub(super) struct ServiceConn {
    svc: Box<dyn TcpService>,
    out: VecDeque<u8>,
}

impl ServiceConn {
    pub(super) fn new(mut svc: Box<dyn TcpService>) -> Self {
        let out = svc.greeting().into();
        Self { svc, out }
    }
}

impl Backend {
    fn poll_connect(&mut self) -> Option<io::Result<()>> {
        match self {
            Backend::Host(h) => h.poll_connect(),
            Backend::Service(_) => Some(Ok(())),
        }
    }

    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        match self {
            Backend::Host(h) => h.stream.as_mut().ok_or(io::ErrorKind::NotConnected)?.write(data),
            Backend::Service(s) => {
                let reply = s.svc.input(data);
                s.out.extend(reply);
                Ok(data.len())
            }
        }
    }

    /// Ok(0) is end of stream; WouldBlock means nothing yet.
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Backend::Host(h) => h.stream.as_mut().ok_or(io::ErrorKind::NotConnected)?.read(buf),
            Backend::Service(s) => {
                if s.out.is_empty() {
                    let more = s.svc.pull(buf.len());
                    s.out.extend(more);
                }
                if s.out.is_empty() {
                    return if s.svc.done() { Ok(0) } else { Err(io::ErrorKind::WouldBlock.into()) };
                }
                let n = buf.len().min(s.out.len());
                for (b, v) in buf.iter_mut().zip(s.out.drain(..n)) {
                    *b = v;
                }
                Ok(n)
            }
        }
    }

    fn shutdown_write(&mut self) {
        match self {
            Backend::Host(h) => {
                if let Some(s) = &h.stream {
                    let _ = s.shutdown(Shutdown::Write);
                }
            }
            Backend::Service(s) => s.svc.guest_closed(),
        }
    }

    pub(super) fn is_host(&self) -> bool {
        matches!(self, Backend::Host(_))
    }
}

/// The two ends of a connection as the guest sees them.
#[derive(Clone, Copy, Debug)]
pub(super) struct Endpoints {
    pub guest_mac: MacAddr,
    pub guest_ip: Ipv4Addr,
    pub guest_port: u16,
    /// The far end as the guest sees it (the gateway for host loopback).
    pub remote_ip: Ipv4Addr,
    pub remote_port: u16,
}

pub(super) struct TcpConn {
    pub(super) ends: Endpoints,
    pub(super) state: State,
    backend: Backend,
    // Send side (toward the guest). `unacked` holds snd_una..snd_una+len.
    iss: u32,
    pub(super) snd_una: u32,
    pub(super) snd_nxt: u32,
    /// Highest sequence number sent: an ACK up to here is valid even after a
    /// timeout moved snd_nxt back to snd_una.
    snd_max: u32,
    unacked: VecDeque<u8>,
    pub(super) peer_win: u32,
    peer_mss: u16,
    rto: Duration,
    rto_deadline: Option<Instant>,
    last_progress: Instant,
    // Receive side (from the guest). `to_backend` is our receive buffer.
    irs: u32,
    pub(super) rcv_nxt: u32,
    to_backend: VecDeque<u8>,
    adv_win: u32,
    // Close handshake.
    guest_fin: bool,      // the guest's FIN is in (rcv_nxt covers it)
    backend_shut: bool,   // we shut the backend's write side after the guest's FIN
    backend_eof: bool,    // the backend has no more data
    pub(super) fin_sent: bool, // our FIN is out (it occupies snd_nxt - 1)
    pub(super) last_use: Instant,
    closed_at: Option<Instant>,
    /// Set when the connection must be reset (backend error, timeout).
    reset: bool,
}

/// `a` is after `b` in sequence space.
fn seq_gt(a: u32, b: u32) -> bool {
    (a.wrapping_sub(b) as i32) > 0
}

/// An initial sequence number: a clock-driven counter (RFC 793 style), so a
/// reused port pair never starts where a recent connection left off.
fn new_isn() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SALT: AtomicU32 = AtomicU32::new(0x1d4c_9e37);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let ticks = (t.as_micros() / 4) as u32; // 250 kHz, as RFC 793 suggests
    ticks.wrapping_add(SALT.fetch_add(0x0101_0101, Ordering::Relaxed))
}

impl TcpConn {
    /// The guest's SYN arrived; the backend may still be connecting.
    pub(super) fn accept(ends: Endpoints, syn: &Tcp, backend: Backend) -> Self {
        let iss = new_isn();
        let now = Instant::now();
        TcpConn {
            ends, state: State::Connecting, backend,
            iss, snd_una: iss, snd_nxt: iss, snd_max: iss, unacked: VecDeque::new(),
            peer_win: syn.window() as u32, peer_mss: syn.mss().unwrap_or(DEFAULT_MSS).min(TCP_MSS),
            rto: RTO_INITIAL, rto_deadline: None, last_progress: now,
            irs: syn.seq(), rcv_nxt: syn.seq().wrapping_add(1), to_backend: VecDeque::new(),
            adv_win: RCV_BUF as u32,
            guest_fin: false, backend_shut: false, backend_eof: false, fin_sent: false,
            last_use: now, closed_at: None, reset: false,
        }
    }

    /// We open the connection toward the guest (an inbound forward).
    pub(super) fn open(ends: Endpoints, backend: Backend) -> Self {
        let iss = new_isn();
        let now = Instant::now();
        TcpConn {
            ends, state: State::SynSent, backend,
            iss, snd_una: iss, snd_nxt: iss.wrapping_add(1), snd_max: iss.wrapping_add(1),
            unacked: VecDeque::new(), peer_win: 0, peer_mss: DEFAULT_MSS,
            rto: RTO_INITIAL, rto_deadline: Some(now + RTO_INITIAL), last_progress: now,
            irs: 0, rcv_nxt: 0, to_backend: VecDeque::new(), adv_win: RCV_BUF as u32,
            guest_fin: false, backend_shut: false, backend_eof: false, fin_sent: false,
            last_use: now, closed_at: None, reset: false,
        }
    }

    fn rcv_win(&self) -> u16 {
        RCV_BUF.saturating_sub(self.to_backend.len()) as u16
    }

    pub(super) fn in_flight(&self) -> u32 {
        self.snd_nxt.wrapping_sub(self.snd_una)
    }

    pub(super) fn unacked_len(&self) -> usize {
        self.unacked.len()
    }

    pub(super) fn is_done(&self) -> bool {
        self.state == State::Closed && self.closed_at.is_some_and(|t| t.elapsed() > TIME_WAIT)
    }

    /// A segment to the guest.
    fn seg(&self, gw_mac: MacAddr, seq: u32, flags: TcpFlags, data: &[u8]) -> Vec<u8> {
        let e = &self.ends;
        let h = TcpHeader {
            sport: e.remote_port, dport: e.guest_port, seq,
            ack: if flags.ack() { self.rcv_nxt } else { 0 }, flags,
            window: self.rcv_win(),
            mss: flags.syn().then_some(TCP_MSS),
        };
        Ipv4Route::new(e.guest_mac, gw_mac, e.remote_ip, e.guest_ip).tcp(0, &h, data)
    }

    fn arm_rto(&mut self) {
        if self.rto_deadline.is_none() {
            self.rto_deadline = Some(Instant::now() + self.rto);
        }
    }

    /// The SYN that opens a connection toward the guest.
    pub(super) fn syn(&self, gw_mac: MacAddr) -> Vec<u8> {
        self.seg(gw_mac, self.iss, TcpFlags::SYN, &[])
    }

    /// `input`, but the segment's data reaches the backend as `rewritten` (an
    /// ALG's edit). Only for a segment that is exactly in order; the guest's
    /// sequence space still advances by the original length.
    pub(super) fn input_rewritten(&mut self, gw_mac: MacAddr, tcp: &Tcp, rewritten: &[u8]) -> Vec<Vec<u8>> {
        let original = tcp.payload().len();
        if self.state != State::Established || tcp.seq() != self.rcv_nxt || self.guest_fin
            || RCV_BUF - self.to_backend.len() < rewritten.len().max(original)
        {
            return self.input(gw_mac, tcp);
        }
        self.to_backend.extend(rewritten);
        self.rcv_nxt = self.rcv_nxt.wrapping_add(original as u32);
        self.input(gw_mac, tcp)
    }

    /// A segment from the guest. Returns the frames to send back.
    pub(super) fn input(&mut self, gw_mac: MacAddr, tcp: &Tcp) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let flags = tcp.flags();
        self.last_use = Instant::now();

        if flags.rst() {
            self.state = State::Closed;
            self.closed_at = Some(Instant::now());
            return out;
        }

        match self.state {
            State::Connecting => {
                // A retransmitted SYN while the host connect is in progress:
                // nothing to say yet.
                return out;
            }
            State::SynSent => {
                if flags.syn() && flags.ack() && tcp.ack() == self.iss.wrapping_add(1) {
                    self.irs = tcp.seq();
                    self.rcv_nxt = tcp.seq().wrapping_add(1);
                    self.snd_una = tcp.ack();
                    self.peer_win = tcp.window() as u32;
                    self.peer_mss = tcp.mss().unwrap_or(DEFAULT_MSS).min(TCP_MSS);
                    self.state = State::Established;
                    self.rto = RTO_INITIAL;
                    self.rto_deadline = None;
                    self.last_progress = Instant::now();
                    out.push(self.seg(gw_mac, self.snd_nxt, TcpFlags::ACK, &[]));
                } else if flags.syn() || flags.ack() {
                    // Not the answer to our SYN.
                    self.reset = true;
                }
                return out;
            }
            State::SynReceived => {
                if flags.syn() && !flags.ack() {
                    // Our SYN-ACK was lost: send it again.
                    out.push(self.seg(gw_mac, self.iss, TcpFlags::SYN_ACK, &[]));
                    return out;
                }
                if flags.ack() && tcp.ack() == self.iss.wrapping_add(1) {
                    self.state = State::Established;
                    self.snd_una = tcp.ack();
                    self.rto = RTO_INITIAL;
                    self.rto_deadline = None;
                    self.last_progress = Instant::now();
                } else {
                    return out;
                }
            }
            State::Established | State::Closed => {
                if flags.syn() {
                    // A retransmitted SYN for a connection we already have:
                    // re-acknowledge it.
                    out.push(self.seg(gw_mac, self.snd_nxt, TcpFlags::ACK, &[]));
                    return out;
                }
            }
        }

        // ACK processing.
        if flags.ack() {
            let ack = tcp.ack();
            if seq_gt(ack, self.snd_una) && !seq_gt(ack, self.snd_max) {
                // Our FIN takes a sequence number but no buffer byte; min() covers it.
                let acked = ack.wrapping_sub(self.snd_una) as usize;
                self.unacked.drain(..acked.min(self.unacked.len()));
                self.snd_una = ack;
                if seq_gt(ack, self.snd_nxt) {
                    self.snd_nxt = ack;
                }
                self.rto = RTO_INITIAL;
                self.rto_deadline = if self.snd_una == self.snd_nxt { None } else { Some(Instant::now() + self.rto) };
                self.last_progress = Instant::now();
            }
            self.peer_win = tcp.window() as u32;
        }

        // Data: only what continues the stream at rcv_nxt, and only what fits.
        // (A segment `input_rewritten` already took is behind rcv_nxt now.)
        let data = tcp.payload();
        let mut ack_needed = false;
        if !data.is_empty() {
            ack_needed = true;
            let behind = self.rcv_nxt.wrapping_sub(tcp.seq());
            if (behind as i32) >= 0 && (behind as usize) < data.len() && !self.guest_fin {
                let fresh = &data[behind as usize..];
                let room = RCV_BUF.saturating_sub(self.to_backend.len());
                let take = fresh.len().min(room);
                self.to_backend.extend(&fresh[..take]);
                self.rcv_nxt = self.rcv_nxt.wrapping_add(take as u32);
            }
        }

        // FIN: accepted only when every byte before it is in.
        if flags.fin() {
            ack_needed = true;
            let fin_seq = tcp.seq().wrapping_add(data.len() as u32);
            if fin_seq == self.rcv_nxt && !self.guest_fin {
                self.rcv_nxt = self.rcv_nxt.wrapping_add(1);
                self.guest_fin = true;
            }
        }

        self.flush_to_backend();
        if ack_needed {
            self.adv_win = self.rcv_win() as u32;
            out.push(self.seg(gw_mac, self.snd_nxt, TcpFlags::ACK, &[]));
        }
        self.close_if_done();
        out
    }

    /// Write what the backend takes; shut its write side after the guest's FIN.
    fn flush_to_backend(&mut self) {
        while !self.to_backend.is_empty() {
            let (head, _) = self.to_backend.as_slices();
            match self.backend.write(head) {
                Ok(0) => break,
                Ok(n) => {
                    self.to_backend.drain(..n);
                    self.last_use = Instant::now();
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) if e.kind() == io::ErrorKind::NotConnected => break,
                Err(_) => {
                    self.reset = true;
                    break;
                }
            }
        }
        if self.guest_fin && self.to_backend.is_empty() && !self.backend_shut {
            self.backend.shutdown_write();
            self.backend_shut = true;
        }
    }

    fn close_if_done(&mut self) {
        if self.state == State::Established && self.guest_fin && self.fin_sent
            && self.snd_una == self.snd_max && self.to_backend.is_empty()
        {
            self.state = State::Closed;
            self.closed_at = Some(Instant::now());
        }
    }

    /// Note a segment going out that ends at `end`.
    fn sent_to(&mut self, end: u32) {
        self.snd_nxt = end;
        if seq_gt(end, self.snd_max) {
            self.snd_max = end;
        }
    }

    /// Advance timers and move data from the backend. `budget` is how many
    /// frames the caller can queue. Returns the frames to send.
    pub(super) fn poll(&mut self, gw_mac: MacAddr, budget: usize) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let now = Instant::now();

        if self.reset {
            return self.abort(gw_mac);
        }

        match self.state {
            State::Connecting => {
                match self.backend.poll_connect() {
                    None => {}
                    Some(Ok(())) => {
                        self.state = State::SynReceived;
                        self.sent_to(self.iss.wrapping_add(1));
                        self.rto_deadline = Some(now + self.rto);
                        out.push(self.seg(gw_mac, self.iss, TcpFlags::SYN_ACK, &[]));
                    }
                    Some(Err(_)) => {
                        // Refused or unreachable: reset the guest's SYN.
                        let h = TcpHeader {
                            sport: self.ends.remote_port, dport: self.ends.guest_port, seq: 0,
                            ack: self.rcv_nxt, flags: TcpFlags::RST_ACK, window: 0, mss: None,
                        };
                        let e = &self.ends;
                        out.push(Ipv4Route::new(e.guest_mac, gw_mac, e.remote_ip, e.guest_ip).tcp(0, &h, &[]));
                        self.state = State::Closed;
                        self.closed_at = Some(now);
                    }
                }
                return out;
            }
            State::SynSent | State::SynReceived => {
                if self.rto_deadline.is_some_and(|d| now >= d) {
                    if now.duration_since(self.last_progress) > GIVE_UP / 4 {
                        return self.abort(gw_mac);
                    }
                    let (seq, flags) = if self.state == State::SynSent {
                        (self.iss, TcpFlags::SYN)
                    } else {
                        (self.iss, TcpFlags::SYN_ACK)
                    };
                    out.push(self.seg(gw_mac, seq, flags, &[]));
                    self.backoff(now);
                }
                return out;
            }
            State::Closed => return out,
            State::Established => {}
        }

        // Window update when the backend caught up: once the window has grown
        // by a full segment, or opened completely.
        self.flush_to_backend();
        let win = self.rcv_win() as u32;
        if budget > out.len()
            && (win >= self.adv_win.saturating_add(TCP_MSS as u32) || (win == RCV_BUF as u32 && self.adv_win < win))
        {
            self.adv_win = win;
            out.push(self.seg(gw_mac, self.snd_nxt, TcpFlags::ACK, &[]));
        }

        // Retransmission timeout: go back to the first unacknowledged byte.
        if self.rto_deadline.is_some_and(|d| now >= d) {
            if now.duration_since(self.last_progress) > GIVE_UP {
                return self.abort(gw_mac);
            }
            if self.peer_win == 0 && !self.unacked.is_empty() && !self.fin_sent {
                // Zero window: probe with one byte so a lost window update
                // can't stall us forever.
                let byte = [self.unacked[0]];
                out.push(self.seg(gw_mac, self.snd_una, TcpFlags::ACK, &byte));
                self.sent_to(self.snd_una.wrapping_add(1));
            } else {
                self.snd_nxt = self.snd_una;
                if self.fin_sent && self.unacked.is_empty() {
                    out.push(self.seg(gw_mac, self.snd_una, TcpFlags::FIN_ACK, &[]));
                    self.sent_to(self.snd_una.wrapping_add(1));
                }
            }
            self.backoff(now);
        }

        // Fill the window: resend from snd_nxt what is already buffered, then
        // read more from the backend.
        let mss = self.peer_mss as usize;
        loop {
            if out.len() >= budget {
                break;
            }
            // Data bytes already sent (our FIN only goes after all data is acked).
            let sent = self.in_flight() as usize;
            let window_room = (self.peer_win as usize).saturating_sub(self.in_flight() as usize);
            if window_room == 0 {
                break;
            }
            if sent < self.unacked.len() {
                // Buffered but not (re)sent yet.
                let n = (self.unacked.len() - sent).min(mss).min(window_room);
                let chunk: Vec<u8> = self.unacked.range(sent..sent + n).copied().collect();
                out.push(self.seg(gw_mac, self.snd_nxt, TcpFlags::PSH_ACK, &chunk));
                self.sent_to(self.snd_nxt.wrapping_add(n as u32));
                self.arm_rto();
                continue;
            }
            if self.backend_eof || self.fin_sent || self.unacked.len() >= SND_BUF {
                break;
            }
            let n = mss.min(window_room).min(SND_BUF - self.unacked.len());
            let mut buf = vec![0u8; n];
            match self.backend.read(&mut buf) {
                Ok(0) => {
                    self.backend_eof = true;
                    break;
                }
                Ok(n) => {
                    self.unacked.extend(&buf[..n]);
                    self.last_use = now;
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    self.reset = true;
                    break;
                }
            }
        }

        // Our FIN goes once the backend is done and the guest has every byte
        // (IRIX may discard buffered data on a FIN that overtakes it).
        if self.backend_eof && !self.fin_sent && self.unacked.is_empty() && out.len() < budget {
            out.push(self.seg(gw_mac, self.snd_nxt, TcpFlags::FIN_ACK, &[]));
            self.sent_to(self.snd_nxt.wrapping_add(1));
            self.fin_sent = true;
            self.arm_rto();
        }

        // A half-closed connection that stopped moving.
        if (self.guest_fin || self.fin_sent) && now.duration_since(self.last_use) > HALF_CLOSED_IDLE {
            return self.abort(gw_mac);
        }

        self.close_if_done();
        out
    }

    fn backoff(&mut self, now: Instant) {
        self.rto = (self.rto * 2).min(RTO_MAX);
        self.rto_deadline = Some(now + self.rto);
    }

    /// Reset the guest's side and close.
    fn abort(&mut self, gw_mac: MacAddr) -> Vec<Vec<u8>> {
        self.state = State::Closed;
        self.closed_at = Some(Instant::now());
        self.reset = false;
        vec![self.seg(gw_mac, self.snd_nxt, TcpFlags::RST_ACK, &[])]
    }

    pub(super) fn backend(&self) -> &Backend {
        &self.backend
    }

    /// A connection the in-core service behind this one wants opened toward
    /// the guest. Host-socket backends never ask.
    pub(super) fn take_side_connect(&mut self) -> Option<SideConnect> {
        match &mut self.backend {
            Backend::Service(s) => s.svc.side_connect(),
            Backend::Host(_) => None,
        }
    }

    /// The next retransmission or handshake timer.
    pub(super) fn deadline(&self) -> Option<Instant> {
        self.rto_deadline
    }

    /// The host socket and the poll(2) events that would let this connection
    /// make progress: connect completion, room to write what the guest sent,
    /// data to read while the guest's window has room for it.
    #[cfg(unix)]
    pub(super) fn poll_interest(&self) -> Option<(std::os::fd::RawFd, i16)> {
        use std::os::fd::AsRawFd;
        let Backend::Host(h) = &self.backend else { return None };
        if let Some(sock) = &h.sock {
            return Some((sock.as_raw_fd(), libc::POLLOUT));
        }
        let stream = h.stream.as_ref()?;
        let mut ev = 0;
        if !self.to_backend.is_empty() {
            ev |= libc::POLLOUT;
        }
        let window_room = (self.peer_win as usize).saturating_sub(self.in_flight() as usize);
        if self.state == State::Established && !self.backend_eof && !self.fin_sent
            && window_room > 0 && self.unacked.len() < SND_BUF
        {
            ev |= libc::POLLIN;
        }
        (ev != 0).then(|| (stream.as_raw_fd(), ev))
    }
}

// ── In-core TCP services ─────────────────────────────────────────────────────

/// RFC 868 time: four bytes, then close.
pub(super) struct TimeService(pub [u8; 4]);

impl TcpService for TimeService {
    fn greeting(&mut self) -> Vec<u8> {
        self.0.to_vec()
    }
    fn input(&mut self, _: &[u8]) -> Vec<u8> {
        Vec::new()
    }
    fn done(&self) -> bool {
        true
    }
}

/// Portmap over TCP: record-marked RPC calls (RFC 5531 §11), each answered.
#[derive(Default)]
pub(super) struct PortmapService {
    inbuf: Vec<u8>,
    closed: bool,
}

impl TcpService for PortmapService {
    fn input(&mut self, data: &[u8]) -> Vec<u8> {
        self.inbuf.extend_from_slice(data);
        let mut out = Vec::new();
        // One record per call: a 4-byte mark (last-fragment bit + length).
        while self.inbuf.len() >= 4 {
            let mark = u32::from_be_bytes(self.inbuf[..4].try_into().unwrap());
            let len = (mark & 0x7fff_ffff) as usize;
            if self.inbuf.len() < 4 + len {
                break;
            }
            let call: Vec<u8> = self.inbuf.drain(..4 + len).skip(4).collect();
            let reply = super::portmap::reply(&call);
            out.extend_from_slice(&(reply.len() as u32 | 0x8000_0000).to_be_bytes());
            out.extend_from_slice(&reply);
        }
        out
    }
    fn done(&self) -> bool {
        self.closed
    }
    fn guest_closed(&mut self) {
        self.closed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isns_differ() {
        let a = new_isn();
        let b = new_isn();
        assert_ne!(a, b);
    }

    #[test]
    fn portmap_service_answers_each_record() {
        let mut s = PortmapService::default();
        let call = super::super::portmap::getport_call(5, super::super::portmap::PROG_NFS);
        let mut rec = (call.len() as u32 | 0x8000_0000).to_be_bytes().to_vec();
        rec.extend_from_slice(&call);
        // Split across two inputs: nothing until the record is whole.
        assert!(s.input(&rec[..10]).is_empty());
        let out = s.input(&rec[10..]);
        assert_eq!(u32::from_be_bytes(out[..4].try_into().unwrap()), 0x8000_001c);
        assert_eq!(u32::from_be_bytes(out[28..32].try_into().unwrap()), 2049);
        assert!(!s.done());
        s.guest_closed();
        assert!(s.done());
    }
}
