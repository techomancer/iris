//! Read-only RFC 1350 TFTP server for PROM network boot (`boot -f bootp()<file>`).
//!
//! Pure protocol: packets in, packets out, no sockets. The NAT gateway
//! (`net.rs`, next to `handle_bootp`) drives it and injects the replies, so
//! nothing here touches the host network — the same shape as `nfsudp.rs`.
//!
//! Read-only by construction: WRQ is refused and no path ever opens for writing.
//! Packet fields are network byte order, which is the Edge, so swapping is correct.
//!
//! Names are relative to the root even when they start with `/`, as tftpd
//! `-s` does: `bootp()/unix` and `tftp> get /unix` both ask for `<root>/unix`.
//! Both RFC 1350 modes are served; netascii is the default of most
//! command-line clients, IRIX's and Linux's included.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::net::Ipv4Addr;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

pub const BLOCK_SIZE: usize = 512;
/// Give up on a transfer after this many unanswered retransmits of one block.
pub const MAX_RETRIES: u8 = 5;
pub const RETRANSMIT_TIMEOUT: Duration = Duration::from_millis(1000);

const OP_RRQ: u16 = 1;
const OP_WRQ: u16 = 2;
const OP_DATA: u16 = 3;
const OP_ACK: u16 = 4;
const OP_ERROR: u16 = 5;

/// RFC 1350 error codes, only the ones a read-only server can raise.
const ERR_NOT_DEFINED: u16 = 0;
const ERR_FILE_NOT_FOUND: u16 = 1;
const ERR_ACCESS_VIOLATION: u16 = 2;
const ERR_ILLEGAL_OP: u16 = 4;
const ERR_UNKNOWN_ID: u16 = 5;

/// A client is identified by its address and source port (its TID).
pub type ClientId = (Ipv4Addr, u16);

/// With `IRIS_TFTP_TRACE` set, every request and how each transfer ended is
/// written to stderr, in any build (`dlog_dev!` output exists only in
/// `developer` builds): what was asked for, whether it was found, bytes,
/// blocks, retransmits, duplicate ACKs, the slowest ACK and the total time.
/// Lock-step TFTP is as slow as its slowest round trip, so those numbers say
/// where a slow or stalled PROM network boot spends its time.
pub fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("IRIS_TFTP_TRACE").is_some())
}

macro_rules! ttrace {
    ($s:expr, $($a:tt)*) => {
        if $s.trace {
            eprintln!("iris: tftp {}", format_args!($($a)*));
        }
    };
}

fn who(c: ClientId) -> String {
    format!("{}:{}", c.0, c.1)
}

/// A reply slower than this is reported as it happens.
const SLOW_ACK: Duration = Duration::from_millis(500);

/// Where a transfer's bytes come from. Octet streams the file; netascii
/// changes the length, so it is translated once up front.
enum Source {
    File(File),
    Netascii(Vec<u8>),
}

struct Transfer {
    source: Source,
    /// Block number of the DATA packet currently outstanding (1-based, wraps).
    block: u16,
    /// File offset of that block.
    offset: u64,
    /// The packet itself, kept so a retransmit is a resend, not a re-read.
    last_packet: Vec<u8>,
    /// True once the outstanding block was short — its ACK ends the transfer.
    final_block: bool,
    retries: u8,
    sent_at: Instant,
    // For the trace.
    name: String,
    started: Instant,
    retransmits: u32,
    dup_acks: u32,
    slowest_ack: Duration,
}

pub struct TftpServer {
    root: PathBuf,
    transfers: HashMap<ClientId, Transfer>,
    timeout: Duration,
    trace: bool,
}

impl TftpServer {
    /// Serve read-only out of `root`. Nothing outside it is ever reachable.
    pub fn new(root: PathBuf) -> Self {
        Self { root, transfers: HashMap::new(), timeout: RETRANSMIT_TIMEOUT, trace: trace_enabled() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Handle one datagram from `client`, returning the packet to send back.
    pub fn handle(&mut self, client: ClientId, packet: &[u8], now: Instant) -> Option<Vec<u8>> {
        if packet.len() < 4 {
            return None;
        }
        match u16::from_be_bytes([packet[0], packet[1]]) {
            OP_RRQ => Some(self.start_read(client, &packet[2..], now)),
            OP_WRQ => {
                ttrace!(self, "{} WRQ refused: the server is read-only", who(client));
                Some(error_packet(ERR_ACCESS_VIOLATION, "server is read-only"))
            }
            OP_ACK => self.handle_ack(client, u16::from_be_bytes([packet[2], packet[3]]), now),
            OP_ERROR => {
                let code = u16::from_be_bytes([packet[2], packet[3]]);
                let msg = String::from_utf8_lossy(packet[4..].split(|&b| b == 0).next().unwrap_or(&[])).into_owned();
                match self.transfers.remove(&client) {
                    Some(t) => ttrace!(self, "{} aborted {:?} at block {} (error {}: {:?})",
                                       who(client), t.name, t.block, code, msg),
                    None => ttrace!(self, "{} sent error {}: {:?}", who(client), code, msg),
                }
                None
            }
            op => {
                ttrace!(self, "{} unsupported opcode {}", who(client), op);
                Some(error_packet(ERR_ILLEGAL_OP, "only RRQ is supported"))
            }
        }
    }

    /// Retransmit any DATA that has gone unanswered, and drop transfers that
    /// have exhausted their retries. Returns the packets to send.
    pub fn tick(&mut self, now: Instant) -> Vec<(ClientId, Vec<u8>)> {
        let timeout = self.timeout;
        let trace = self.trace;
        let mut out = Vec::new();
        self.transfers.retain(|client, t| {
            if now.duration_since(t.sent_at) < timeout {
                return true;
            }
            if t.retries >= MAX_RETRIES {
                if trace {
                    eprintln!("iris: tftp {} gave up on {:?} at block {}: no ACK after {} retransmits ({} ms in)",
                              who(*client), t.name, t.block, MAX_RETRIES, now.duration_since(t.started).as_millis());
                }
                return false;
            }
            t.retries += 1;
            t.retransmits += 1;
            t.sent_at = now;
            if trace {
                eprintln!("iris: tftp {} retransmit {:?} block {} (try {})", who(*client), t.name, t.block, t.retries);
            }
            out.push((*client, t.last_packet.clone()));
            true
        });
        out
    }

    pub fn active_transfers(&self) -> usize {
        self.transfers.len()
    }

    /// Whether `client` still has a transfer in flight. The NAT uses this to
    /// drop the client bookkeeping it keeps alongside.
    pub fn has_transfer(&self, client: &ClientId) -> bool {
        self.transfers.contains_key(client)
    }

    fn start_read(&mut self, client: ClientId, body: &[u8], now: Instant) -> Vec<u8> {
        let Some((name, mode)) = parse_request(body) else {
            ttrace!(self, "{} malformed request ({} bytes)", who(client), body.len());
            return error_packet(ERR_NOT_DEFINED, "malformed request");
        };
        ttrace!(self, "{} RRQ {:?} ({})", who(client), name, mode);
        // The PROM always asks for octet; command-line clients default to netascii.
        let netascii = mode.eq_ignore_ascii_case("netascii");
        if !netascii && !mode.eq_ignore_ascii_case("octet") {
            ttrace!(self, "{} refused: mode {:?} is neither octet nor netascii", who(client), mode);
            return error_packet(ERR_NOT_DEFINED, "mode must be octet or netascii");
        }
        let Some(path) = self.resolve(&name) else {
            ttrace!(self, "{} refused: {:?} is not under {} (or leaves it)", who(client), name, self.root.display());
            return error_packet(ERR_FILE_NOT_FOUND, &format!("{}: not found", name));
        };
        let mut file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                ttrace!(self, "{} refused: {:?}: {}", who(client), name, e);
                return error_packet(ERR_FILE_NOT_FOUND, &format!("{}: {}", name, e));
            }
        };
        let source = if netascii {
            let mut raw = Vec::new();
            if let Err(e) = file.read_to_end(&mut raw) {
                ttrace!(self, "{} refused: {:?}: {}", who(client), name, e);
                return error_packet(ERR_NOT_DEFINED, &format!("{}: {}", name, e));
            }
            Source::Netascii(to_netascii(&raw))
        } else {
            Source::File(file)
        };

        let size = match &source {
            Source::File(f) => f.metadata().map(|m| m.len()).ok(),
            Source::Netascii(d) => Some(d.len() as u64),
        };
        // A second RRQ from the same TID replaces whatever was in flight.
        let mut t = Transfer {
            name: name.clone(),
            started: now,
            retransmits: 0,
            dup_acks: 0,
            slowest_ack: Duration::ZERO,
            source,
            block: 0,
            offset: 0,
            last_packet: Vec::new(),
            final_block: false,
            retries: 0,
            sent_at: now,
        };
        match send_block(&mut t, 1, 0, now) {
            Ok(pkt) => {
                if let Some(old) = self.transfers.insert(client, t) {
                    ttrace!(self, "{} new request replaces {:?} (was at block {})", who(client), old.name, old.block);
                }
                match size {
                    Some(n) => ttrace!(self, "{} sending {:?}: {} bytes, {} blocks", who(client), name, n, n / BLOCK_SIZE as u64 + 1),
                    None => ttrace!(self, "{} sending {:?}", who(client), name),
                }
                pkt
            }
            Err(e) => {
                ttrace!(self, "{} refused: {:?}: {}", who(client), name, e);
                error_packet(ERR_NOT_DEFINED, &format!("{}: {}", name, e))
            }
        }
    }

    fn handle_ack(&mut self, client: ClientId, block: u16, now: Instant) -> Option<Vec<u8>> {
        let trace = self.trace;
        let Some(t) = self.transfers.get_mut(&client) else {
            ttrace!(self, "{} ACK {} with no transfer in progress", who(client), block);
            return Some(error_packet(ERR_UNKNOWN_ID, "no transfer in progress"));
        };
        // A duplicate ACK (the classic Sorcerer's Apprentice case) must not
        // advance the transfer — ignore anything that isn't the block in flight.
        if block != t.block {
            t.dup_acks += 1;
            return None;
        }
        let rtt = now.duration_since(t.sent_at);
        t.slowest_ack = t.slowest_ack.max(rtt);
        if trace && rtt >= SLOW_ACK {
            eprintln!("iris: tftp {} slow ACK for {:?} block {}: {} ms", who(client), t.name, block, rtt.as_millis());
        }
        if t.final_block {
            if let Some(t) = self.transfers.remove(&client) {
                ttrace!(self, "{} complete {:?}: {} bytes, {} blocks, {} retransmits, {} duplicate ACKs, slowest ACK {} ms, {} ms total",
                        who(client), t.name, t.offset + t.last_packet.len().saturating_sub(4) as u64,
                        block, t.retransmits, t.dup_acks, t.slowest_ack.as_millis(),
                        now.duration_since(t.started).as_millis());
            }
            return None;
        }
        let next_offset = t.offset + BLOCK_SIZE as u64;
        // Block numbers wrap 65535 → 0; a file can be longer than 32 MB.
        let next_block = t.block.wrapping_add(1);
        match send_block(t, next_block, next_offset, now) {
            Ok(pkt) => Some(pkt),
            Err(e) => {
                self.transfers.remove(&client);
                ttrace!(self, "{} read error at block {}: {}", who(client), next_block, e);
                Some(error_packet(ERR_NOT_DEFINED, &e.to_string()))
            }
        }
    }

    /// Whether `name` would be served.
    pub fn exists(&self, name: &str) -> bool {
        file_exists(&self.root, name)
    }

    fn resolve(&self, name: &str) -> Option<PathBuf> {
        resolve(&self.root, name)
    }
}

/// Whether `name` is a file TFTP would serve from `root`. BOOTP asks this
/// before naming a boot file.
pub fn file_exists(root: &Path, name: &str) -> bool {
    resolve(root, name).is_some_and(|p| p.is_file())
}

/// Resolve `name` under `root`, refusing anything that escapes it: any `..`,
/// and symlinks pointing outside. Leading slashes are dropped, so an absolute
/// name is taken relative to the root.
fn resolve(root: &Path, name: &str) -> Option<PathBuf> {
    let rel = Path::new(name.trim_start_matches('/'));
    if rel.as_os_str().is_empty() {
        return None;
    }
    if !rel.components().all(|c| matches!(c, Component::Normal(_))) {
        return None;
    }
    let root = root.canonicalize().ok()?;
    let full = root.join(rel).canonicalize().ok()?;
    full.starts_with(&root).then_some(full)
}

/// Read `block` at `offset` and build its DATA packet, recording it for retransmit.
fn send_block(t: &mut Transfer, block: u16, offset: u64, now: Instant) -> std::io::Result<Vec<u8>> {
    let mut buf = [0u8; BLOCK_SIZE];
    let mut filled = 0;
    match &mut t.source {
        Source::File(file) => {
            file.seek(SeekFrom::Start(offset))?;
            while filled < BLOCK_SIZE {
                match file.read(&mut buf[filled..])? {
                    0 => break,
                    n => filled += n,
                }
            }
        }
        Source::Netascii(data) => {
            let start = (offset as usize).min(data.len());
            let chunk = &data[start..(start + BLOCK_SIZE).min(data.len())];
            buf[..chunk.len()].copy_from_slice(chunk);
            filled = chunk.len();
        }
    }

    let mut pkt = Vec::with_capacity(4 + filled);
    pkt.extend_from_slice(&OP_DATA.to_be_bytes());
    pkt.extend_from_slice(&block.to_be_bytes());
    pkt.extend_from_slice(&buf[..filled]);

    t.block = block;
    t.offset = offset;
    // A short block ends the transfer — including the zero-length one that
    // follows a file whose length is an exact multiple of 512.
    t.final_block = filled < BLOCK_SIZE;
    t.last_packet = pkt.clone();
    t.retries = 0;
    t.sent_at = now;
    Ok(pkt)
}

/// RFC 764 netascii: LF becomes CR LF, a bare CR becomes CR NUL.
fn to_netascii(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + raw.len() / 32);
    for &b in raw {
        match b {
            b'\n' => out.extend_from_slice(b"\r\n"),
            b'\r' => out.extend_from_slice(b"\r\0"),
            _ => out.push(b),
        }
    }
    out
}

fn error_packet(code: u16, msg: &str) -> Vec<u8> {
    let mut pkt = Vec::with_capacity(5 + msg.len());
    pkt.extend_from_slice(&OP_ERROR.to_be_bytes());
    pkt.extend_from_slice(&code.to_be_bytes());
    pkt.extend_from_slice(msg.as_bytes());
    pkt.push(0);
    pkt
}

/// Split an RRQ/WRQ body into its NUL-terminated filename and mode.
fn parse_request(body: &[u8]) -> Option<(String, String)> {
    let mut parts = body.split(|&b| b == 0);
    let name = String::from_utf8(parts.next()?.to_vec()).ok()?;
    let mode = String::from_utf8(parts.next()?.to_vec()).ok()?;
    Some((name, mode))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLIENT: ClientId = (Ipv4Addr::new(192, 168, 0, 2), 1069);

    fn tmp_root(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let d = std::env::temp_dir().join(format!("iris-tftp-{}-{}", tag, nanos));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn rrq(name: &str, mode: &str) -> Vec<u8> {
        let mut p = vec![0, 1];
        p.extend_from_slice(name.as_bytes());
        p.push(0);
        p.extend_from_slice(mode.as_bytes());
        p.push(0);
        p
    }

    fn ack(block: u16) -> Vec<u8> {
        let mut p = vec![0, 4];
        p.extend_from_slice(&block.to_be_bytes());
        p
    }

    fn opcode(p: &[u8]) -> u16 { u16::from_be_bytes([p[0], p[1]]) }
    fn field(p: &[u8]) -> u16 { u16::from_be_bytes([p[2], p[3]]) }

    /// Run a whole transfer, returning the reassembled file and the block count.
    fn transfer(srv: &mut TftpServer, name: &str) -> (Vec<u8>, usize) {
        let now = Instant::now();
        let mut got = Vec::new();
        let mut pkt = srv.handle(CLIENT, &rrq(name, "octet"), now).unwrap();
        let mut blocks = 0;
        loop {
            assert_eq!(opcode(&pkt), OP_DATA, "expected DATA, got {:?}", &pkt[..4.min(pkt.len())]);
            blocks += 1;
            got.extend_from_slice(&pkt[4..]);
            let last = pkt.len() - 4 < BLOCK_SIZE;
            match srv.handle(CLIENT, &ack(field(&pkt)), now) {
                Some(next) => pkt = next,
                None => {
                    assert!(last, "transfer ended on a full block");
                    break;
                }
            }
        }
        (got, blocks)
    }

    #[test]
    fn serves_a_file_that_is_not_a_multiple_of_512() {
        let root = tmp_root("short");
        let data: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
        std::fs::write(root.join("cputest"), &data).unwrap();
        let mut srv = TftpServer::new(root.clone());
        let (got, blocks) = transfer(&mut srv, "cputest");
        assert_eq!(got, data);
        assert_eq!(blocks, 2, "1000 bytes = 512 + 488");
        assert_eq!(srv.active_transfers(), 0, "transfer must be cleaned up");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn exact_multiple_of_512_ends_with_a_zero_length_block() {
        let root = tmp_root("exact");
        let data = vec![0xA5u8; 1024];
        std::fs::write(root.join("aligned"), &data).unwrap();
        let mut srv = TftpServer::new(root.clone());
        let (got, blocks) = transfer(&mut srv, "aligned");
        assert_eq!(got, data);
        assert_eq!(blocks, 3, "1024 bytes = 512 + 512 + a final empty block");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn empty_file_is_one_empty_block() {
        let root = tmp_root("empty");
        std::fs::write(root.join("nothing"), b"").unwrap();
        let mut srv = TftpServer::new(root.clone());
        let (got, blocks) = transfer(&mut srv, "nothing");
        assert!(got.is_empty());
        assert_eq!(blocks, 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn duplicate_ack_does_not_advance_the_transfer() {
        let root = tmp_root("dup");
        std::fs::write(root.join("f"), vec![7u8; 2000]).unwrap();
        let mut srv = TftpServer::new(root.clone());
        let now = Instant::now();
        srv.handle(CLIENT, &rrq("f", "octet"), now).unwrap();
        let b2 = srv.handle(CLIENT, &ack(1), now).unwrap();
        assert_eq!(field(&b2), 2);
        assert!(srv.handle(CLIENT, &ack(1), now).is_none(), "stale ACK must be ignored");
        let b3 = srv.handle(CLIENT, &ack(2), now).unwrap();
        assert_eq!(field(&b3), 3, "transfer resumes from the block actually in flight");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn retransmits_then_gives_up() {
        let root = tmp_root("rtx");
        std::fs::write(root.join("f"), vec![1u8; 2000]).unwrap();
        let mut srv = TftpServer::new(root.clone());
        let start = Instant::now();
        let first = srv.handle(CLIENT, &rrq("f", "octet"), start).unwrap();

        let mut now = start;
        for i in 1..=MAX_RETRIES {
            now += RETRANSMIT_TIMEOUT;
            let out = srv.tick(now);
            assert_eq!(out.len(), 1, "retry {} should resend", i);
            assert_eq!(out[0].1, first, "retransmit must resend the same DATA");
        }
        now += RETRANSMIT_TIMEOUT;
        assert!(srv.tick(now).is_empty(), "transfer is dropped after MAX_RETRIES");
        assert_eq!(srv.active_transfers(), 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn tick_is_quiet_before_the_timeout() {
        let root = tmp_root("quiet");
        std::fs::write(root.join("f"), vec![1u8; 2000]).unwrap();
        let mut srv = TftpServer::new(root.clone());
        let start = Instant::now();
        srv.handle(CLIENT, &rrq("f", "octet"), start).unwrap();
        assert!(srv.tick(start + Duration::from_millis(10)).is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn missing_file_is_error_1() {
        let root = tmp_root("missing");
        let mut srv = TftpServer::new(root.clone());
        let p = srv.handle(CLIENT, &rrq("nope", "octet"), Instant::now()).unwrap();
        assert_eq!(opcode(&p), OP_ERROR);
        assert_eq!(field(&p), ERR_FILE_NOT_FOUND);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_request_is_refused() {
        let root = tmp_root("wrq");
        let mut srv = TftpServer::new(root.clone());
        let mut p = vec![0, 2];
        p.extend_from_slice(b"evil\0octet\0");
        let r = srv.handle(CLIENT, &p, Instant::now()).unwrap();
        assert_eq!(opcode(&r), OP_ERROR);
        assert_eq!(field(&r), ERR_ACCESS_VIOLATION);
        assert!(!root.join("evil").exists(), "WRQ must never create a file");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn unknown_mode_is_refused() {
        let root = tmp_root("mode");
        std::fs::write(root.join("f"), b"x").unwrap();
        let mut srv = TftpServer::new(root.clone());
        let p = srv.handle(CLIENT, &rrq("f", "mail"), Instant::now()).unwrap();
        assert_eq!(opcode(&p), OP_ERROR);
        // Case-insensitive: the PROM sends lowercase "octet", others vary.
        let ok = srv.handle(CLIENT, &rrq("f", "OCTET"), Instant::now()).unwrap();
        assert_eq!(opcode(&ok), OP_DATA);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Command-line clients (IRIX, BSD, tftp-hpa) default to netascii.
    #[test]
    fn netascii_translates_line_ends_across_blocks() {
        let root = tmp_root("netascii");
        // 300 lines push the translated stream over one block.
        let mut data = Vec::new();
        for i in 0..300u32 {
            data.extend_from_slice(format!("line {}\n", i).as_bytes());
        }
        data.extend_from_slice(b"bare\rcr");
        std::fs::write(root.join("t"), &data).unwrap();
        let mut srv = TftpServer::new(root.clone());

        let now = Instant::now();
        let mut pkt = srv.handle(CLIENT, &rrq("t", "NetASCII"), now).unwrap();
        let mut got = Vec::new();
        loop {
            assert_eq!(opcode(&pkt), OP_DATA);
            got.extend_from_slice(&pkt[4..]);
            match srv.handle(CLIENT, &ack(field(&pkt)), now) {
                Some(next) => pkt = next,
                None => break,
            }
        }
        assert_eq!(got, to_netascii(&data));
        assert!(got.len() > BLOCK_SIZE);
        assert!(got.ends_with(b"bare\r\0cr"));
        assert!(got.windows(3).any(|w| w == b"0\r\n"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `bootp()/unix` and `tftp> get /unix` send absolute names; tftpd -s
    /// semantics take them relative to the root.
    #[test]
    fn absolute_names_are_relative_to_the_root() {
        let root = tmp_root("abs");
        std::fs::create_dir_all(root.join("boot")).unwrap();
        std::fs::write(root.join("boot/unix"), b"kernel").unwrap();
        let mut srv = TftpServer::new(root.clone());
        let p = srv.handle(CLIENT, &rrq("/boot/unix", "octet"), Instant::now()).unwrap();
        assert_eq!(opcode(&p), OP_DATA);
        assert_eq!(&p[4..], b"kernel");
        assert!(srv.exists("/boot/unix") && srv.exists("boot/unix"));
        assert!(!srv.exists("boot"), "a directory is not a boot file");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn refuses_paths_that_escape_the_root() {
        let root = tmp_root("escape");
        let outside = root.parent().unwrap().join(format!(
            "iris-tftp-outside-{}",
            root.file_name().unwrap().to_string_lossy()
        ));
        std::fs::write(&outside, b"secret").unwrap();
        std::fs::write(root.join("inside"), b"fine").unwrap();
        let mut srv = TftpServer::new(root.clone());

        let now = Instant::now();
        for bad in ["../etc/passwd", "/../etc/passwd", "sub/../../escape", "", "/", "/etc/passwd"] {
            let p = srv.handle(CLIENT, &rrq(bad, "octet"), now).unwrap();
            assert_eq!(opcode(&p), OP_ERROR, "{} must be refused", bad);
        }

        // A symlink pointing out of the root is refused even though it resolves.
        #[cfg(unix)]
        {
            let link = root.join("link");
            std::os::unix::fs::symlink(&outside, &link).unwrap();
            let p = srv.handle(CLIENT, &rrq("link", "octet"), now).unwrap();
            assert_eq!(opcode(&p), OP_ERROR, "symlink out of the root must be refused");
        }

        let good = srv.handle(CLIENT, &rrq("inside", "octet"), now).unwrap();
        assert_eq!(opcode(&good), OP_DATA);

        let _ = std::fs::remove_file(&outside);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn ack_without_a_transfer_is_error_5() {
        let root = tmp_root("stray");
        let mut srv = TftpServer::new(root.clone());
        let p = srv.handle(CLIENT, &ack(1), Instant::now()).unwrap();
        assert_eq!(opcode(&p), OP_ERROR);
        assert_eq!(field(&p), ERR_UNKNOWN_ID);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Tracing only writes to stderr: every path behaves as it does untraced.
    #[test]
    fn tracing_changes_no_behaviour() {
        let root = tmp_root("trace");
        std::fs::write(root.join("f"), vec![7u8; 1100]).unwrap();
        let mut s = TftpServer::new(root.clone());
        s.trace = true;
        let now = Instant::now();
        assert_eq!(opcode(&s.handle(CLIENT, &rrq("nope", "octet"), now).unwrap()), OP_ERROR);
        assert_eq!(opcode(&s.handle(CLIENT, &rrq("f", "mail"), now).unwrap()), OP_ERROR);
        assert_eq!(opcode(&s.handle(CLIENT, &[0, 2, b'x', 0, b'o', 0], now).unwrap()), OP_ERROR);
        assert_eq!(opcode(&s.handle(CLIENT, &ack(1), now).unwrap()), OP_ERROR);
        let p = s.handle(CLIENT, &rrq("f", "octet"), now).unwrap();
        assert_eq!((opcode(&p), field(&p)), (OP_DATA, 1));
        assert!(s.handle(CLIENT, &ack(9), now).is_none(), "a wrong ACK is a duplicate");
        let p = s.handle(CLIENT, &ack(1), now + Duration::from_millis(700)).unwrap();
        assert_eq!(field(&p), 2);
        let p = s.handle(CLIENT, &ack(2), now).unwrap();
        assert_eq!((field(&p), p.len()), (3, 4 + 76));
        assert!(s.handle(CLIENT, &ack(3), now).is_none());
        assert_eq!(s.active_transfers(), 0);
        // Retransmits, then giving up.
        s.handle(CLIENT, &rrq("f", "octet"), now).unwrap();
        let mut t = now;
        let mut sent = 0;
        for _ in 0..(MAX_RETRIES as usize + 2) {
            t += RETRANSMIT_TIMEOUT;
            sent += s.tick(t).len();
        }
        assert_eq!(sent, MAX_RETRIES as usize);
        assert_eq!(s.active_transfers(), 0);
        // A new request replaces one in flight; a client error ends it.
        s.handle(CLIENT, &rrq("f", "octet"), now).unwrap();
        s.handle(CLIENT, &rrq("f", "netascii"), now).unwrap();
        assert_eq!(s.active_transfers(), 1);
        let mut err = vec![0, 5, 0, 3];
        err.extend_from_slice(b"disk full\0");
        assert!(s.handle(CLIENT, &err, now).is_none());
        assert_eq!(s.active_transfers(), 0);
        let _ = std::fs::remove_dir_all(&root);
    }
}
