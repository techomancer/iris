//! Read-only RFC 1350 TFTP server for PROM network boot (`boot -f bootp()<file>`).
//!
//! Pure protocol: packets in, packets out, no sockets. The NAT gateway
//! (`net.rs`, next to `handle_bootp`) drives it and injects the replies, so
//! nothing here touches the host network — the same shape as `nfsudp.rs`.
//!
//! Read-only by construction: WRQ is refused and no path ever opens for writing.
//! Packet fields are network byte order, which is the Edge, so swapping is correct.

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

struct Transfer {
    file: File,
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
}

pub struct TftpServer {
    root: PathBuf,
    transfers: HashMap<ClientId, Transfer>,
    timeout: Duration,
}

impl TftpServer {
    /// Serve read-only out of `root`. Nothing outside it is ever reachable.
    pub fn new(root: PathBuf) -> Self {
        Self { root, transfers: HashMap::new(), timeout: RETRANSMIT_TIMEOUT }
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
            OP_WRQ => Some(error_packet(ERR_ACCESS_VIOLATION, "server is read-only")),
            OP_ACK => self.handle_ack(client, u16::from_be_bytes([packet[2], packet[3]]), now),
            OP_ERROR => {
                self.transfers.remove(&client);
                None
            }
            _ => Some(error_packet(ERR_ILLEGAL_OP, "only RRQ is supported")),
        }
    }

    /// Retransmit any DATA that has gone unanswered, and drop transfers that
    /// have exhausted their retries. Returns the packets to send.
    pub fn tick(&mut self, now: Instant) -> Vec<(ClientId, Vec<u8>)> {
        let timeout = self.timeout;
        let mut out = Vec::new();
        self.transfers.retain(|client, t| {
            if now.duration_since(t.sent_at) < timeout {
                return true;
            }
            if t.retries >= MAX_RETRIES {
                return false;
            }
            t.retries += 1;
            t.sent_at = now;
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
            return error_packet(ERR_NOT_DEFINED, "malformed request");
        };
        // The PROM only ever asks for octet (verified in the embedded PROM image);
        // netascii would need CRLF translation we deliberately don't do.
        if !mode.eq_ignore_ascii_case("octet") {
            return error_packet(ERR_NOT_DEFINED, "only octet mode is supported");
        }
        let Some(path) = self.resolve(&name) else {
            return error_packet(ERR_FILE_NOT_FOUND, &format!("{}: not found", name));
        };
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => return error_packet(ERR_FILE_NOT_FOUND, &format!("{}: {}", name, e)),
        };

        // A second RRQ from the same TID replaces whatever was in flight.
        let mut t = Transfer {
            file,
            block: 0,
            offset: 0,
            last_packet: Vec::new(),
            final_block: false,
            retries: 0,
            sent_at: now,
        };
        match send_block(&mut t, 1, 0, now) {
            Ok(pkt) => {
                self.transfers.insert(client, t);
                pkt
            }
            Err(e) => error_packet(ERR_NOT_DEFINED, &format!("{}: {}", name, e)),
        }
    }

    fn handle_ack(&mut self, client: ClientId, block: u16, now: Instant) -> Option<Vec<u8>> {
        let Some(t) = self.transfers.get_mut(&client) else {
            return Some(error_packet(ERR_UNKNOWN_ID, "no transfer in progress"));
        };
        // A duplicate ACK (the classic Sorcerer's Apprentice case) must not
        // advance the transfer — ignore anything that isn't the block in flight.
        if block != t.block {
            return None;
        }
        if t.final_block {
            self.transfers.remove(&client);
            return None;
        }
        let next_offset = t.offset + BLOCK_SIZE as u64;
        // Block numbers wrap 65535 → 0; a file can be longer than 32 MB.
        let next_block = t.block.wrapping_add(1);
        match send_block(t, next_block, next_offset, now) {
            Ok(pkt) => Some(pkt),
            Err(e) => {
                self.transfers.remove(&client);
                Some(error_packet(ERR_NOT_DEFINED, &e.to_string()))
            }
        }
    }

    /// Resolve `name` under the root, refusing anything that escapes it:
    /// absolute paths, any `..`, and symlinks pointing outside.
    fn resolve(&self, name: &str) -> Option<PathBuf> {
        let rel = Path::new(name);
        if rel.as_os_str().is_empty() || rel.is_absolute() {
            return None;
        }
        if !rel.components().all(|c| matches!(c, Component::Normal(_))) {
            return None;
        }
        let root = self.root.canonicalize().ok()?;
        let full = root.join(rel).canonicalize().ok()?;
        full.starts_with(&root).then_some(full)
    }
}

/// Read `block` at `offset` and build its DATA packet, recording it for retransmit.
fn send_block(t: &mut Transfer, block: u16, offset: u64, now: Instant) -> std::io::Result<Vec<u8>> {
    let mut buf = [0u8; BLOCK_SIZE];
    t.file.seek(SeekFrom::Start(offset))?;
    let mut filled = 0;
    while filled < BLOCK_SIZE {
        match t.file.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
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
    fn netascii_is_refused() {
        let root = tmp_root("mode");
        std::fs::write(root.join("f"), b"x").unwrap();
        let mut srv = TftpServer::new(root.clone());
        let p = srv.handle(CLIENT, &rrq("f", "netascii"), Instant::now()).unwrap();
        assert_eq!(opcode(&p), OP_ERROR);
        // Case-insensitive: the PROM sends lowercase "octet", others vary.
        let ok = srv.handle(CLIENT, &rrq("f", "OCTET"), Instant::now()).unwrap();
        assert_eq!(opcode(&ok), OP_DATA);
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
        for bad in ["../etc/passwd", "/etc/passwd", "sub/../../escape", ""] {
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
}
