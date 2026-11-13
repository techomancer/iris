//! Read-only CD-ROM media served over HTTP(S), like QEMU's `http://` drive.
//!
//! [`HttpNode`] is a byte-addressed, read-only [`BlockNode`] whose bytes come
//! from `Range` requests. The server must honour byte ranges (the open fails
//! otherwise: without them every 2 KiB sector read would fetch the whole
//! image). Reads are fetched in [`CHUNK`]-sized pieces and kept in a small LRU,
//! so a sequential install costs one request per chunk, not one per command.
//!
//! The client is the `http-media` feature (on by default). [`is_url`] is
//! always built, so a build without it still tells a URL from a path and
//! refuses the URL with a clear message instead of looking for a file.

use std::io;

use crate::block_node::BlockNode;
#[cfg(feature = "http-media")]
use crate::block_node::BlockStatus;
#[cfg(feature = "http-media")]
use std::collections::VecDeque;
#[cfg(feature = "http-media")]
use std::io::Read;
#[cfg(feature = "http-media")]
use std::time::Duration;

/// Bytes fetched per request.
#[cfg(feature = "http-media")]
const CHUNK: u64 = 512 * 1024;
/// Chunks kept (16 MiB).
#[cfg(feature = "http-media")]
const CACHE_CHUNKS: usize = 32;
#[cfg(feature = "http-media")]
const ATTEMPTS: u32 = 3;

/// Whether `path` names an HTTP or HTTPS image rather than a file.
pub fn is_url(path: &str) -> bool {
    let p = path.trim_start();
    let lower = |n: usize| p.get(..n).map(|s| s.to_ascii_lowercase());
    lower(7).as_deref() == Some("http://") || lower(8).as_deref() == Some("https://")
}

/// Open the CD-ROM image at `url` (see [`is_url`]).
#[cfg(feature = "http-media")]
pub fn open(url: &str) -> io::Result<Box<dyn BlockNode>> {
    Ok(Box::new(HttpNode::open(url)?))
}

/// Open the CD-ROM image at `url`: this build has no HTTP client.
#[cfg(not(feature = "http-media"))]
pub fn open(url: &str) -> io::Result<Box<dyn BlockNode>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        format!("{url}: this build has no HTTP media support (Cargo feature `http-media`)"),
    ))
}

#[cfg(feature = "http-media")]
fn other(msg: impl Into<String>) -> io::Error {
    io::Error::other(msg.into())
}

#[cfg(feature = "http-media")]
pub struct HttpNode {
    url: String,
    agent: ureq::Agent,
    size: u64,
    /// Most recently used last: `(chunk index, bytes)`.
    cache: VecDeque<(u64, Vec<u8>)>,
}

#[cfg(feature = "http-media")]
impl HttpNode {
    /// Open `url`, learning its size from a one-byte range request.
    pub fn open(url: &str) -> io::Result<Self> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(60)))
            .user_agent("iris")
            .build()
            .into();
        let mut resp = agent
            .get(url)
            .header("Range", "bytes=0-0")
            .call()
            .map_err(|e| other(format!("{url}: {e}")))?;
        let status = resp.status().as_u16();
        let size = match status {
            206 => {
                let cr = resp
                    .headers()
                    .get("content-range")
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| other(format!("{url}: 206 without Content-Range")))?;
                parse_content_range(cr)
                    .map(|(_, _, total)| total)
                    .ok_or_else(|| other(format!("{url}: unusable Content-Range '{cr}'")))?
            }
            200 => return Err(other(format!("{url}: server ignores Range requests (HTTP 200); cannot serve CD media"))),
            416 => return Err(other(format!("{url}: empty image (HTTP 416)"))),
            s => return Err(other(format!("{url}: HTTP {s}"))),
        };
        // Drain the byte so the connection can be reused.
        let _ = resp.body_mut().as_reader().read(&mut [0u8; 1]);
        if size == 0 {
            return Err(other(format!("{url}: empty image")));
        }
        Ok(Self { url: url.to_string(), agent, size, cache: VecDeque::new() })
    }

    fn fetch(&self, chunk: u64) -> io::Result<Vec<u8>> {
        let first = chunk * CHUNK;
        let last = (first + CHUNK).min(self.size) - 1;
        let want = (last - first + 1) as usize;
        let mut err = None;
        for _ in 0..ATTEMPTS {
            let mut resp = match self.agent.get(&self.url).header("Range", format!("bytes={first}-{last}")).call() {
                Ok(r) => r,
                Err(e) => {
                    err = Some(other(format!("{}: {e}", self.url)));
                    continue;
                }
            };
            let status = resp.status().as_u16();
            if status != 206 {
                err = Some(other(format!("{}: HTTP {status} for bytes {first}-{last}", self.url)));
                continue;
            }
            let total = resp
                .headers()
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .and_then(parse_content_range);
            match total {
                Some((a, _, t)) if a == first && t == self.size => {}
                _ => {
                    err = Some(other(format!("{}: image changed or bad Content-Range", self.url)));
                    continue;
                }
            }
            let mut buf = vec![0u8; want];
            match resp.body_mut().as_reader().read_exact(&mut buf) {
                Ok(()) => return Ok(buf),
                Err(e) => err = Some(e),
            }
        }
        Err(err.unwrap_or_else(|| other("HTTP read failed")))
    }

    fn chunk(&mut self, idx: u64) -> io::Result<&[u8]> {
        if let Some(pos) = self.cache.iter().position(|(i, _)| *i == idx) {
            let hit = self.cache.remove(pos).unwrap();
            self.cache.push_back(hit);
        } else {
            let data = self.fetch(idx)?;
            if self.cache.len() == CACHE_CHUNKS {
                self.cache.pop_front();
            }
            self.cache.push_back((idx, data));
        }
        Ok(&self.cache.back().unwrap().1)
    }
}

#[cfg(feature = "http-media")]
/// `bytes FIRST-LAST/TOTAL`.
fn parse_content_range(v: &str) -> Option<(u64, u64, u64)> {
    let rest = v.trim().strip_prefix("bytes")?.trim_start();
    let (range, total) = rest.split_once('/')?;
    let (a, b) = range.split_once('-')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?, total.trim().parse().ok()?))
}

#[cfg(feature = "http-media")]
fn read_only() -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, "HTTP CD media is read-only")
}

#[cfg(feature = "http-media")]
impl BlockNode for HttpNode {
    fn size(&self) -> u64 {
        self.size
    }

    fn read(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let end = offset
            .checked_add(buf.len() as u64)
            .filter(|&e| e <= self.size)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "block range past end of node"))?;
        let mut at = offset;
        while at < end {
            let idx = at / CHUNK;
            let data = self.chunk(idx)?;
            let from = (at - idx * CHUNK) as usize;
            let n = (data.len() - from).min((end - at) as usize);
            let o = (at - offset) as usize;
            buf[o..o + n].copy_from_slice(&data[from..from + n]);
            at += n as u64;
        }
        Ok(())
    }

    fn write(&mut self, _offset: u64, _buf: &[u8]) -> io::Result<()> {
        Err(read_only())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn discard(&mut self, _offset: u64, _len: u64) -> io::Result<()> {
        Err(read_only())
    }

    fn block_status(&self, offset: u64, len: u64) -> io::Result<BlockStatus> {
        if offset.checked_add(len).is_none_or(|e| e > self.size) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "block range past end of node"));
        }
        Ok(BlockStatus::data(len))
    }

    fn backing(&self) -> Option<&dyn BlockNode> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_are_told_from_paths() {
        assert!(is_url("http://h/x.iso"));
        assert!(is_url("HTTPS://h/x.iso"));
        assert!(!is_url("/tmp/http://x"));
        assert!(!is_url("ftp://h/x"));
        assert!(!is_url("disc.iso"));
    }

    #[cfg(not(feature = "http-media"))]
    #[test]
    fn a_build_without_the_client_refuses_urls() {
        let e = open("http://127.0.0.1:9/disc.iso").err().expect("must refuse");
        assert_eq!(e.kind(), io::ErrorKind::Unsupported);
        assert!(e.to_string().contains("http-media"), "{e}");
    }
}

#[cfg(all(test, feature = "http-media"))]
mod client_tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A minimal range-capable server over `data`; `ranges` = false answers
    /// every request with 200 and the whole body.
    fn serve(data: Vec<u8>, ranges: bool) -> (String, Arc<AtomicUsize>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/disc.iso", l.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let h = hits.clone();
        std::thread::spawn(move || {
            for s in l.incoming() {
                let Ok(mut s) = s else { break };
                let data = data.clone();
                let h = h.clone();
                std::thread::spawn(move || {
                    let mut r = std::io::BufReader::new(s.try_clone().unwrap());
                    loop {
                        let mut range = None;
                        let mut line = String::new();
                        if std::io::BufRead::read_line(&mut r, &mut line).unwrap_or(0) == 0 {
                            return;
                        }
                        loop {
                            line.clear();
                            std::io::BufRead::read_line(&mut r, &mut line).unwrap();
                            if line == "\r\n" {
                                break;
                            }
                            if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                                let (a, b) = v.trim().split_once('-').unwrap();
                                range = Some((a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()));
                            }
                        }
                        h.fetch_add(1, Ordering::SeqCst);
                        let head = match range.filter(|_| ranges) {
                            Some((a, b)) => {
                                let b = b.min(data.len() - 1);
                                let body = &data[a..=b];
                                let _ = write!(
                                    s,
                                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {a}-{b}/{}\r\nContent-Length: {}\r\n\r\n",
                                    data.len(),
                                    body.len()
                                );
                                s.write_all(body)
                            }
                            None => {
                                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", data.len());
                                s.write_all(&data)
                            }
                        };
                        if head.is_err() {
                            return;
                        }
                    }
                });
            }
        });
        (url, hits)
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 31 + i / 251) as u8).collect()
    }

    #[test]
    fn content_range_parses() {
        assert_eq!(parse_content_range("bytes 0-0/1234"), Some((0, 0, 1234)));
        assert_eq!(parse_content_range("bytes */1234"), None);
    }

    #[test]
    fn reads_across_chunks_and_caches() {
        let data = pattern(CHUNK as usize * 2 + 4321);
        let (url, hits) = serve(data.clone(), true);
        let mut n = HttpNode::open(&url).unwrap();
        assert_eq!(n.size(), data.len() as u64);
        let mut buf = vec![0u8; 5000];
        let at = CHUNK - 100;
        n.read(at, &mut buf).unwrap();
        assert_eq!(buf, data[at as usize..at as usize + 5000]);
        let before = hits.load(Ordering::SeqCst);
        n.read(at + 10, &mut buf[..100]).unwrap();
        assert_eq!(hits.load(Ordering::SeqCst), before, "cached read made a request");
        let mut tail = vec![0u8; 4321];
        n.read(data.len() as u64 - 4321, &mut tail).unwrap();
        assert_eq!(tail, data[data.len() - 4321..]);
        assert!(n.read(data.len() as u64 - 1, &mut [0u8; 2]).is_err());
        assert!(n.write(0, &[0]).is_err());
    }

    #[test]
    fn server_without_ranges_is_refused() {
        let (url, _) = serve(pattern(4096), false);
        let e = HttpNode::open(&url).err().expect("must refuse");
        assert!(e.to_string().contains("Range"), "{e}");
    }

    /// Against a real server: `IRIS_HTTP_TEST_URL=https://... cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn live_url() {
        let url = std::env::var("IRIS_HTTP_TEST_URL").expect("IRIS_HTTP_TEST_URL");
        let mut n = HttpNode::open(&url).unwrap();
        let mut b = vec![0u8; 2048];
        n.read(16 * 2048, &mut b).unwrap(); // ISO 9660 volume descriptor
        eprintln!("size {} pvd {:?}", n.size(), &b[..8]);
    }
}
