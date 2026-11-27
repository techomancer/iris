//! Phase 3.4: HTTP snapshot registry.
//!
//! Pull/push iris snapshots between machines, Docker-layer-style. The CAS
//! chunk store from Phase 3.1 makes this nearly free at the wire level: only
//! chunks that the receiving side doesn't already have are transferred.
//!
//! ## URL layout
//!
//! Mirrors the on-disk layout, so any static file server pointing at
//! `saves/` (e.g. `python3 -m http.server` running in your saves directory)
//! works as a read-only pull source. Push needs a server that accepts PUT.
//!
//! ```text
//!   GET  <base>/snapshots/<name>/snapshot.toml      ← manifest, schema_version
//!   GET  <base>/snapshots/<name>/cpu.bin            ← device state
//!   GET  <base>/snapshots/<name>/chunks.bin         ← CAS hash list
//!   GET  <base>/snapshots/<name>/cow.toml           ← per-disk dirty sectors, by slot path
//!   GET  <base>/snapshots/<name>/hpc3/c0/scsi1.overlay ← a disk's overlay bytes, in its bus's directory
//!   GET  <base>/cas/<hex2>/<hex62>.chunk            ← content-addressed RAM chunk
//! ```
//!
//! ## Wire format
//!
//! Hand-rolled HTTP/1.1 over `std::net::TcpStream` — no new dependency. HTTP
//! only (no TLS); use behind a tunnel or trusted network. Single-request,
//! single-connection — no keep-alive. Plenty for snapshot transfers because
//! the per-request overhead is dwarfed by the chunk payload.
//!
//! ## Commit ordering
//!
//! Push uploads chunks first, then `snapshot.toml` LAST. An interrupted push
//! leaves orphan chunks (which `gc` on the server side will sweep) but never
//! a half-published snapshot manifest pointing at missing chunks. Pull
//! validates `chunks.bin` against fetched chunks at the end so a torn pull
//! is detectable.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::chunk_store::{ChunkHash, ChunkStore};
use crate::snapshot::{ChunksManifest, Snapshot};

const HTTP_TIMEOUT: Duration = Duration::from_secs(60);

/// Outcome of a `pull` or `push` operation. JSON-serializable for the CI socket.
#[derive(Debug, Clone, Default)]
pub struct TransferReport {
    pub chunks_fetched: u64,
    pub chunks_skipped: u64,
    pub bytes_transferred: u64,
    pub files_transferred: u64,
}

/// Pull a snapshot from `base_url` into the local `saves_dir`. Idempotent —
/// chunks already in the local store are not re-downloaded. Returns a
/// transfer report.
pub fn pull(base_url: &str, name: &str, saves_dir: &Path) -> Result<TransferReport, String> {
    if name.is_empty() || name.contains("..") {
        return Err("pull: invalid snapshot name".into());
    }
    let base = base_url.trim_end_matches('/');
    let mut report = TransferReport::default();

    let snap_dir = saves_dir.join(name);
    std::fs::create_dir_all(&snap_dir).map_err(|e| format!("create {}: {}", snap_dir.display(), e))?;

    // 1. Manifest first.
    let manifest_url = format!("{}/snapshots/{}/snapshot.toml", base, name);
    let manifest_bytes = http_get(&manifest_url).map_err(|e| format!("fetch manifest: {}", e))?;
    std::fs::write(snap_dir.join("snapshot.toml"), &manifest_bytes)
        .map_err(|e| format!("write snapshot.toml: {}", e))?;
    report.files_transferred += 1;
    report.bytes_transferred += manifest_bytes.len() as u64;

    let snap_local = Snapshot::new(&snap_dir);
    snap_local
        .read_manifest()
        .map_err(|e| format!("parse manifest: {}", e))?
        .ok_or_else(|| "manifest missing or unparseable".to_string())?;

    // 2. Per-device state (postcard).
    let device_bases = [
        "cpu", "mc", "ioc", "scc", "pit", "ps2", "rtc",
        "eeprom", "scsi", "seeq", "hpc3", "rex3",
    ];
    let suffix = "bin";
    for base_name in device_bases {
        let url = format!("{}/snapshots/{}/{}.{}", base, name, base_name, suffix);
        match http_get(&url) {
            Ok(bytes) => {
                std::fs::write(snap_dir.join(format!("{}.{}", base_name, suffix)), &bytes)
                    .map_err(|e| format!("write {}.{}: {}", base_name, suffix, e))?;
                report.files_transferred += 1;
                report.bytes_transferred += bytes.len() as u64;
            }
            Err(e) if e.contains("404") => {
                // rex3 is optional (headless configs skip it); other devices
                // could in principle be absent in a future config.
                continue;
            }
            Err(e) => return Err(format!("fetch {}.{}: {}", base_name, suffix, e)),
        }
    }

    // 3. cow.toml (overlay dirty sector lists) and the overlay file of each disk it lists.
    if let Ok(bytes) = http_get(&format!("{}/snapshots/{}/cow.toml", base, name)) {
        std::fs::write(snap_dir.join("cow.toml"), &bytes)
            .map_err(|e| format!("write cow.toml: {}", e))?;
        report.files_transferred += 1;
        report.bytes_transferred += bytes.len() as u64;

        let disks = snap_local.read_toml("cow.toml").map(|t| crate::snapshot::parse_cow_table(&t)).unwrap_or_default();
        for (at, _) in disks {
            let fname = crate::snapshot::overlay_file(&at);
            if let Ok(b) = http_get(&format!("{}/snapshots/{}/{}", base, name, fname)) {
                let path = snap_dir.join(&fname);
                std::fs::create_dir_all(path.parent().unwrap_or(&snap_dir)).map_err(|e| format!("create {}: {}", fname, e))?;
                std::fs::write(&path, &b).map_err(|e| format!("write {}: {}", fname, e))?;
                report.files_transferred += 1;
                report.bytes_transferred += b.len() as u64;
            }
        }
    }

    // 4. Fetch chunks.bin, then any chunk hashes the local store doesn't
    //    already have.
    let chunks_url = format!("{}/snapshots/{}/chunks.bin", base, name);
    let chunks_bytes = http_get(&chunks_url).map_err(|e| format!("fetch chunks.bin: {}", e))?;
    let chunks = ChunksManifest::from_bytes(&chunks_bytes)
        .map_err(|e| format!("parse chunks.bin: {}", e))?;
    std::fs::write(snap_dir.join("chunks.bin"), &chunks_bytes)
        .map_err(|e| format!("write chunks.bin: {}", e))?;
    report.files_transferred += 1;
    report.bytes_transferred += chunks_bytes.len() as u64;

    let store = ChunkStore::new(saves_dir);
    let mut seen: std::collections::HashSet<ChunkHash> = std::collections::HashSet::new();
    for hash in chunks.referenced_hashes() {
        if !seen.insert(*hash) {
            continue;
        }
        if store.has(hash) {
            report.chunks_skipped += 1;
            continue;
        }
        let url = format!("{}/cas/{}/{}.chunk", base, hex2_of(hash), hex62_of(hash));
        let bytes = http_get(&url).map_err(|e| format!("fetch chunk {}: {}", hex_of(hash), e))?;
        // Validate the server gave us the right content.
        let actual: ChunkHash = blake3::hash(&bytes).into();
        if &actual != hash {
            return Err(format!(
                "chunk hash mismatch for {}: got {}",
                hex_of(hash),
                hex_of(&actual)
            ));
        }
        store
            .put(&bytes)
            .map_err(|e| format!("store chunk {}: {}", hex_of(hash), e))?;
        report.chunks_fetched += 1;
        report.bytes_transferred += bytes.len() as u64;
    }

    Ok(report)
}

/// Push a local snapshot to `base_url`. Uploads only chunks the server
/// doesn't already have. Manifest goes LAST so an interrupted push never
/// leaves a half-committed snapshot.
pub fn push(base_url: &str, name: &str, saves_dir: &Path) -> Result<TransferReport, String> {
    if name.is_empty() || name.contains("..") {
        return Err("push: invalid snapshot name".into());
    }
    let base = base_url.trim_end_matches('/');
    let mut report = TransferReport::default();

    let snap_dir = saves_dir.join(name);
    if !snap_dir.is_dir() {
        return Err(format!("push: snapshot '{}' not found", name));
    }
    let snap_local = Snapshot::new(&snap_dir);
    snap_local
        .read_manifest()
        .map_err(|e| format!("read manifest: {}", e))?
        .ok_or_else(|| "manifest missing".to_string())?;

    // 1. Chunks first. Manifest goes last so the snapshot only becomes
    //    visible to pullers once all its chunks are in place.
    let chunks_path = snap_dir.join("chunks.bin");
    let chunks_bytes = std::fs::read(&chunks_path).map_err(|e| format!("read chunks.bin: {}", e))?;
    let chunks = ChunksManifest::from_bytes(&chunks_bytes)
        .map_err(|e| format!("parse chunks.bin: {}", e))?;
    let store = ChunkStore::new(saves_dir);
    let mut seen: std::collections::HashSet<ChunkHash> = std::collections::HashSet::new();
    for hash in chunks.referenced_hashes() {
        if !seen.insert(*hash) {
            continue;
        }
        let url = format!("{}/cas/{}/{}.chunk", base, hex2_of(hash), hex62_of(hash));
        if http_head(&url).unwrap_or(false) {
            report.chunks_skipped += 1;
            continue;
        }
        let bytes = store
            .get(hash)
            .map_err(|e| format!("read chunk {}: {}", hex_of(hash), e))?;
        http_put(&url, &bytes).map_err(|e| format!("PUT chunk {}: {}", hex_of(hash), e))?;
        report.chunks_fetched += 1;
        report.bytes_transferred += bytes.len() as u64;
    }

    // 2. Per-device state.
    let device_bases = [
        "cpu", "mc", "ioc", "scc", "pit", "ps2", "rtc",
        "eeprom", "scsi", "seeq", "hpc3", "rex3",
    ];
    let suffix = "bin";
    for base_name in device_bases {
        let p = snap_dir.join(format!("{}.{}", base_name, suffix));
        if !p.exists() {
            continue; // rex3 may legitimately be absent
        }
        let bytes = std::fs::read(&p).map_err(|e| format!("read {}: {}", p.display(), e))?;
        let url = format!("{}/snapshots/{}/{}.{}", base, name, base_name, suffix);
        http_put(&url, &bytes).map_err(|e| format!("PUT {}.{}: {}", base_name, suffix, e))?;
        report.files_transferred += 1;
        report.bytes_transferred += bytes.len() as u64;
    }

    // 3. cow.toml + each listed disk's overlay file (a sector image, can be MB).
    let cow_path = snap_dir.join("cow.toml");
    if cow_path.exists() {
        let cow_bytes = std::fs::read(&cow_path).map_err(|e| format!("read cow.toml: {}", e))?;
        // Push overlay binaries first, cow.toml last (the index that lists them).
        let disks = snap_local.read_toml("cow.toml").map(|t| crate::snapshot::parse_cow_table(&t)).unwrap_or_default();
        for (at, _) in disks {
            let fname = crate::snapshot::overlay_file(&at);
            let p = snap_dir.join(&fname);
            if p.exists() {
                let bytes = std::fs::read(&p).map_err(|e| format!("read {}: {}", fname, e))?;
                let url = format!("{}/snapshots/{}/{}", base, name, fname);
                http_put(&url, &bytes).map_err(|e| format!("PUT {}: {}", fname, e))?;
                report.files_transferred += 1;
                report.bytes_transferred += bytes.len() as u64;
            }
        }
        let url = format!("{}/snapshots/{}/cow.toml", base, name);
        http_put(&url, &cow_bytes).map_err(|e| format!("PUT cow.toml: {}", e))?;
        report.files_transferred += 1;
        report.bytes_transferred += cow_bytes.len() as u64;
    }

    // 4. chunks.bin — uploaded BEFORE manifest because pullers fetch
    //    it after manifest and would otherwise race a concurrent push.
    let chunks_bytes = std::fs::read(snap_dir.join("chunks.bin"))
        .map_err(|e| format!("read chunks.bin: {}", e))?;
    let url = format!("{}/snapshots/{}/chunks.bin", base, name);
    http_put(&url, &chunks_bytes).map_err(|e| format!("PUT chunks.bin: {}", e))?;
    report.files_transferred += 1;
    report.bytes_transferred += chunks_bytes.len() as u64;

    // 5. Manifest LAST (commit point).
    let manifest_bytes = std::fs::read(snap_dir.join("snapshot.toml"))
        .map_err(|e| format!("read snapshot.toml: {}", e))?;
    let url = format!("{}/snapshots/{}/snapshot.toml", base, name);
    http_put(&url, &manifest_bytes).map_err(|e| format!("PUT snapshot.toml: {}", e))?;
    report.files_transferred += 1;
    report.bytes_transferred += manifest_bytes.len() as u64;

    Ok(report)
}

// ---- minimal HTTP/1.1 client over std::net ----

struct ParsedUrl {
    host: String,
    port: u16,
    path: String,
}

fn parse_url(url: &str) -> Result<ParsedUrl, String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("only http:// URLs supported, got {}", url))?;
    let (host_port, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match host_port.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().map_err(|e| format!("bad port: {}", e))?),
        None => (host_port.to_string(), 80u16),
    };
    Ok(ParsedUrl {
        host,
        port,
        path: path.to_string(),
    })
}

fn http_send(method: &str, url: &str, body: Option<&[u8]>) -> Result<(u16, Vec<u8>), String> {
    let p = parse_url(url)?;
    let addr = (p.host.as_str(), p.port)
        .to_socket_addrs()
        .map_err(|e| format!("resolve {}: {}", p.host, e))?
        .next()
        .ok_or_else(|| format!("no addresses for {}", p.host))?;
    let mut s = TcpStream::connect_timeout(&addr, HTTP_TIMEOUT).map_err(|e| format!("connect: {}", e))?;
    s.set_read_timeout(Some(HTTP_TIMEOUT)).ok();
    s.set_write_timeout(Some(HTTP_TIMEOUT)).ok();

    let mut req = Vec::with_capacity(256);
    write!(req, "{} {} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n",
           method, p.path, p.host, p.port).map_err(|e| e.to_string())?;
    if let Some(b) = body {
        write!(req, "Content-Length: {}\r\nContent-Type: application/octet-stream\r\n", b.len())
            .map_err(|e| e.to_string())?;
    }
    req.extend_from_slice(b"\r\n");
    if let Some(b) = body {
        req.extend_from_slice(b);
    }
    s.write_all(&req).map_err(|e| format!("write request: {}", e))?;

    let mut buf = Vec::new();
    s.read_to_end(&mut buf).map_err(|e| format!("read response: {}", e))?;

    // Parse status + headers + body.
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| "malformed response: no header terminator".to_string())?;
    let header = std::str::from_utf8(&buf[..split])
        .map_err(|e| format!("non-utf8 header: {}", e))?;
    let body_start = split + 4;
    let mut lines = header.lines();
    let status_line = lines.next().ok_or_else(|| "empty response".to_string())?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("bad status line: {}", status_line))?;

    // Handle Content-Length and Transfer-Encoding: chunked.
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    for line in lines {
        if let Some(v) = line.strip_prefix_ignore_case("Content-Length: ") {
            content_length = v.trim().parse().ok();
        }
        if line.eq_ignore_ascii_case("Transfer-Encoding: chunked") {
            chunked = true;
        }
    }
    let body = if chunked {
        decode_chunked(&buf[body_start..])?
    } else if let Some(n) = content_length {
        buf[body_start..body_start + n.min(buf.len() - body_start)].to_vec()
    } else {
        buf[body_start..].to_vec()
    };
    Ok((status, body))
}

trait StripPrefixIgnoreCase {
    fn strip_prefix_ignore_case(&self, prefix: &str) -> Option<&str>;
}
impl StripPrefixIgnoreCase for str {
    fn strip_prefix_ignore_case(&self, prefix: &str) -> Option<&str> {
        if self.len() >= prefix.len() && self[..prefix.len()].eq_ignore_ascii_case(prefix) {
            Some(&self[prefix.len()..])
        } else {
            None
        }
    }
}

fn decode_chunked(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        // Read the size line up to \r\n.
        let crlf = data[i..]
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or_else(|| "chunked: missing size CRLF".to_string())?;
        let size_str = std::str::from_utf8(&data[i..i + crlf])
            .map_err(|_| "chunked: non-utf8 size line".to_string())?;
        let size = usize::from_str_radix(size_str.split(';').next().unwrap_or(size_str).trim(), 16)
            .map_err(|e| format!("chunked: bad size {}: {}", size_str, e))?;
        i += crlf + 2;
        if size == 0 {
            break;
        }
        if i + size > data.len() {
            return Err("chunked: short chunk data".into());
        }
        out.extend_from_slice(&data[i..i + size]);
        i += size + 2; // skip trailing \r\n
    }
    Ok(out)
}

fn http_get(url: &str) -> Result<Vec<u8>, String> {
    let (status, body) = http_send("GET", url, None)?;
    if status == 200 {
        Ok(body)
    } else {
        Err(format!("HTTP {} for {}", status, url))
    }
}

fn http_head(url: &str) -> Result<bool, String> {
    let (status, _) = http_send("HEAD", url, None)?;
    Ok(status == 200)
}

fn http_put(url: &str, body: &[u8]) -> Result<(), String> {
    let (status, resp) = http_send("PUT", url, Some(body))?;
    if (200..300).contains(&status) {
        Ok(())
    } else {
        Err(format!(
            "HTTP {} for PUT {}: {}",
            status,
            url,
            String::from_utf8_lossy(&resp)
        ))
    }
}

// ---- hex helpers (mirror chunk_store layout) ----

fn hex_of(h: &ChunkHash) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for &b in h.iter() {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}
fn hex2_of(h: &ChunkHash) -> String {
    let s = hex_of(h);
    s[..2].to_string()
}
fn hex62_of(h: &ChunkHash) -> String {
    let s = hex_of(h);
    s[2..].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_url_default_port() {
        let p = parse_url("http://example.com/foo/bar").unwrap();
        assert_eq!(p.host, "example.com");
        assert_eq!(p.port, 80);
        assert_eq!(p.path, "/foo/bar");
    }

    #[test]
    fn parse_url_explicit_port() {
        let p = parse_url("http://localhost:8080/").unwrap();
        assert_eq!(p.host, "localhost");
        assert_eq!(p.port, 8080);
        assert_eq!(p.path, "/");
    }

    #[test]
    fn parse_url_no_path() {
        let p = parse_url("http://localhost:8080").unwrap();
        assert_eq!(p.path, "/");
    }

    #[test]
    fn parse_url_rejects_https() {
        assert!(parse_url("https://example.com/").is_err());
    }

    #[test]
    fn decode_chunked_happy() {
        let data = b"4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n";
        let out = decode_chunked(data).unwrap();
        assert_eq!(out, b"Wikipedia");
    }

    /// A server on a local port: it keeps every PUT, serves what it holds and answers 404 to the rest.
    fn serve() -> String {
        use std::collections::HashMap;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            let mut held: HashMap<String, Vec<u8>> = HashMap::new();
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { break };
                let mut req = Vec::new();
                let mut chunk = [0u8; 4096];
                let end = loop {
                    let n = s.read(&mut chunk).unwrap();
                    req.extend_from_slice(&chunk[..n]);
                    if let Some(i) = req.windows(4).position(|w| w == b"\r\n\r\n") { break i + 4 }
                    if n == 0 { break req.len() }
                };
                let head = String::from_utf8_lossy(&req[..end]).into_owned();
                let mut first = head.split_whitespace();
                let (method, path) = (first.next().unwrap_or(""), first.next().unwrap_or("").to_string());
                let len = head.lines().find_map(|l| l.strip_prefix("Content-Length: ")?.trim().parse().ok()).unwrap_or(0);
                while req.len() < end + len {
                    let n = s.read(&mut chunk).unwrap();
                    req.extend_from_slice(&chunk[..n]);
                }
                let (status, body) = match (method, held.get(&path)) {
                    ("PUT", _) => { held.insert(path, req[end..end + len].to_vec()); ("201 Created", vec![]) }
                    ("GET", Some(b)) => ("200 OK", b.clone()),
                    _ => ("404 Not Found", vec![]),
                };
                let _ = write!(s, "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = s.write_all(&body);
            }
        });
        base
    }

    /// A snapshot's overlays go to the server and come back, found by the slot paths `cow.toml` names.
    #[test]
    fn overlays_travel_by_slot_path() {
        use crate::snapshot::Manifest;
        use crate::test_temp::TempPath;
        let (from, to) = (TempPath::dir("registry-from"), TempPath::dir("registry-to"));
        let overlays = [("hpc3/c0/scsi1.overlay", "disk"), ("xio13/pci0/1/scsi0.overlay", "card"), ("xio13/pci0/2/scsi5.overlay", "loop")];
        let snap = Snapshot::new(from.join("n"));
        std::fs::create_dir_all(&snap.dir).unwrap();
        for (file, bytes) in overlays {
            std::fs::create_dir_all(snap.dir.join(file).parent().unwrap()).unwrap();
            std::fs::write(snap.dir.join(file), bytes).unwrap();
        }
        let cow = r#""hpc3/c0/t1" = [1]
"xio13/pci0/1" = [2]
"xio13/pci0/2/loop5" = [3]"#;
        snap.write_toml("cow.toml", &toml::from_str::<toml::Value>(cow).unwrap()).unwrap();
        snap.write_manifest(&Manifest::for_current_save()).unwrap();
        snap.write_chunks_manifest(&ChunksManifest::default()).unwrap();
        let server = serve();
        push(&server, "n", &from).unwrap();
        pull(&server, "n", &to).unwrap();
        for (file, bytes) in overlays {
            assert_eq!(std::fs::read_to_string(to.join("n").join(file)).ok().as_deref(), Some(bytes), "{file}");
        }
    }

    #[test]
    fn hex_round_trip() {
        let h: ChunkHash = blake3::hash(b"hello").into();
        let s = hex_of(&h);
        assert_eq!(s.len(), 64);
        assert_eq!(hex2_of(&h).len(), 2);
        assert_eq!(hex62_of(&h).len(), 62);
        assert_eq!(format!("{}{}", hex2_of(&h), hex62_of(&h)), s);
    }
}
