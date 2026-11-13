//! The PROMs iris embeds: download them from the catalog, and pack them into
//! `data/proms/<board>.deflate`. Run from the repository root:
//!
//!     cargo run -p iris-prom -- download [--into DIR]    fetch and verify the catalog's files
//!     cargo run -p iris-prom -- pack [--from DIR]        unwrap and deflate them into data/proms
//!     cargo run -p iris-prom -- pack --check             fail unless data/proms holds exactly the
//!                                                        catalog's PROMs
//!     cargo run -p iris-prom -- show FILE                what a PROM file holds
//!     cargo run -p iris-prom -- extract FILE [--segment NAME] [--out PATH]
//!                                                        write its payload, or one segment
//!
//! `data/proms/proms.toml` lists each published file with its size and sha256. What is
//! stored is the PROM a machine runs: the file unwrapped by `prom_image::decode` (the
//! installer header, the version-six container and its gzip segment); a row with
//! `unwrap = false` is stored as published, because the IP27 PROM unpacks the IO6
//! container from the BaseIO flash itself (unwrapped, it rejects the magic and falls
//! back to its own copy). A file whose PROM has not changed is left untouched, so git
//! sees only the PROMs that did. Never commit an extracted PROM: the packed files and
//! the catalog are all that belong in the repository.
//!
//! `show` and `extract` read a file with `prom_image` (`inspect`, `nested`, `decode`),
//! including the segment records some payloads carry: the IP35 PROM holds the BaseIO
//! board's PROM that way, as the segment `io7prom`.

use clap::{Parser, Subcommand};
use flate2::{read::DeflateDecoder, write::DeflateEncoder, Compression};
use iris::dev::prom_image;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(about = "Download, pack and inspect the PROMs iris embeds")]
struct Cli {
    /// The catalog of published PROM files (for download and pack).
    #[arg(long, global = true, default_value = "data/proms/proms.toml")]
    catalog: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download the catalog's files and check their size and sha256.
    Download {
        /// Where to keep them (default: a directory under the temp dir).
        #[arg(long)]
        into: Option<PathBuf>,
    },
    /// Unwrap and deflate the catalog's files into the catalog's directory.
    Pack {
        /// Read the files from this directory instead of downloading them.
        #[arg(long)]
        from: Option<PathBuf>,
        /// Change nothing; fail if a packed file does not hold the catalog's PROM.
        #[arg(long)]
        check: bool,
    },
    /// Print what a PROM file holds: its kind, version, segments (each checked) and the
    /// version banners found in its bytes.
    Show { file: PathBuf },
    /// Write a PROM file's unwrapped payload, or one named segment of it, checked
    /// against its own lengths and checksums.
    Extract {
        file: PathBuf,
        /// A segment of the file or of its payload (see `show`).
        #[arg(long)]
        segment: Option<String>,
        /// Where to write it (default: the segment's name, or FILE.payload).
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

#[derive(Deserialize)]
struct Catalog {
    format: u32,
    prom: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    board: String,
    url: String,
    size: usize,
    sha256: String,
    /// Unwrap the file with `prom_image::decode` (the default); false stores it as published.
    #[serde(default = "yes")]
    unwrap: bool,
}

fn yes() -> bool {
    true
}

impl Row {
    fn file_name(&self) -> &str {
        self.url.rsplit('/').next().unwrap()
    }
}

fn read_catalog(path: &Path) -> Catalog {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let catalog: Catalog = toml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(catalog.format, 1, "{}: format {}", path.display(), catalog.format);
    catalog
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn download(url: &str) -> Vec<u8> {
    let mut reply = ureq::get(url).call().unwrap_or_else(|e| panic!("{url}: {e}"));
    reply.body_mut().with_config().limit(64 << 20).read_to_vec().unwrap_or_else(|e| panic!("{url}: {e}"))
}

/// The row's published file from `dir`, checked against its size and sha256; downloaded
/// there first when `fetch` and it is not already there.
fn load(row: &Row, fetch: bool, dir: &Path) -> Vec<u8> {
    let name = row.file_name();
    let path = dir.join(name);
    if fetch && !path.exists() {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(&path, download(&row.url)).unwrap();
    }
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(bytes.len(), row.size, "{name}: size");
    assert_eq!(sha256(&bytes), row.sha256, "{name}: sha256");
    bytes
}

fn pack(catalog: &Catalog, out: &Path, from: Option<PathBuf>, check: bool) {
    let fetch = from.is_none();
    let dir = from.unwrap_or_else(default_dir);
    let mut stale = Vec::new();
    for row in &catalog.prom {
        let file = load(row, fetch, &dir);
        let payload = if row.unwrap { prom_image::decode(&file, 8 << 20).unwrap_or_else(|e| panic!("{}: {e}", row.url)) } else { file };
        let path = out.join(format!("{}.deflate", row.board));
        // The deflater may change between versions: what counts is what the file inflates to.
        let mut current = Vec::new();
        let same = std::fs::read(&path)
            .is_ok_and(|stored| DeflateDecoder::new(&stored[..]).read_to_end(&mut current).is_ok() && current == payload);
        if same {
            continue;
        }
        if check {
            stale.push(path);
        } else {
            let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
            encoder.write_all(&payload).unwrap();
            let packed = encoder.finish().unwrap();
            println!("{}: {} bytes -> {}", path.display(), payload.len(), packed.len());
            std::fs::write(&path, packed).unwrap();
        }
    }
    if check {
        assert!(stale.is_empty(), "these do not hold the catalog's PROMs: {stale:?}");
        println!("ok: the packed files hold the catalog's {} PROMs", catalog.prom.len());
    }
}

/// The most a payload or segment may unpack to.
const LIMIT: usize = 64 << 20;

fn read(file: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(file).map_err(|e| format!("{}: {e}", file.display()))
}

/// The file's payload as iris would run it.
fn payload(file: &Path, bytes: &[u8]) -> Result<Vec<u8>, String> {
    prom_image::decode(bytes, LIMIT).map_err(|e| format!("{}: {e}", file.display()))
}

fn segment_line(seg: &prom_image::Segment, buf: &[u8], place: &str) -> String {
    let method = if seg.gzip() { format!("gzip, {} stored", seg.stored) } else { "stored".into() };
    let checks = match seg.unpack(buf, LIMIT) {
        Ok(_) => "checksums match".to_string(),
        Err(e) => format!("BAD: {e}"),
    };
    format!("  segment {:<10} {place} record {:#x}: {method}, {} unpacked, {checks}\n", seg.name, seg.record, seg.length)
}

fn show(file: &Path) -> Result<String, String> {
    let bytes = read(file)?;
    let info = prom_image::inspect(&bytes);
    let mut text = format!("{}: {} bytes, sha256 {}\n", file.display(), bytes.len(), sha256(&bytes));
    let kind = match info.kind {
        prom_image::Kind::Raw => "a raw flash image (no header)",
        prom_image::Kind::Installer => "the installer's header, then the flash contents",
        prom_image::Kind::Container => "a version-six container",
    };
    text += &format!("  {kind}{}\n", info.version.map(|v| format!(", header version {v}")).unwrap_or_default());
    if let Some(seg) = &info.segment {
        text += &segment_line(seg, &bytes, "in the file,");
    }
    let payload = payload(file, &bytes)?;
    text += &format!("  payload {} bytes, sha256 {}\n", payload.len(), sha256(&payload));
    for seg in prom_image::nested(&payload) {
        text += &segment_line(&seg, &payload, "in the payload,");
    }
    for banner in prom_image::banners(&bytes) {
        text += &format!("  banner at {}: {}\n", banner.at, banner.text.replace('\n', " / "));
    }
    Ok(text)
}

fn extract(file: &Path, segment: Option<String>, out: Option<PathBuf>) -> Result<String, String> {
    let bytes = read(file)?;
    let payload = payload(file, &bytes)?;
    let (data, default) = match &segment {
        None => (payload, format!("{}.payload", file.file_name().unwrap().to_string_lossy())),
        Some(name) => {
            let in_file = prom_image::inspect(&bytes).segment.filter(|s| &s.name == name);
            let unpacked = match in_file {
                Some(seg) => seg.unpack(&bytes, LIMIT),
                None => match prom_image::nested(&payload).into_iter().find(|s| &s.name == name) {
                    Some(seg) => seg.unpack(&payload, LIMIT),
                    None => return Err(format!("{}: no segment {name} (see `show`)", file.display())),
                },
            };
            (unpacked.map_err(|e| format!("{}: segment {name}: {e}", file.display()))?, format!("{name}.bin"))
        }
    };
    let out = out.unwrap_or(PathBuf::from(default));
    std::fs::write(&out, &data).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(format!("wrote {} bytes to {}, sha256 {}", data.len(), out.display(), sha256(&data)))
}

fn default_dir() -> PathBuf {
    std::env::temp_dir().join("iris-prom")
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::Show { file } => show(&file),
        Cmd::Extract { file, segment, out } => extract(&file, segment, out),
        Cmd::Download { into } => {
            let catalog = read_catalog(&cli.catalog);
            let dir = into.unwrap_or_else(default_dir);
            for row in &catalog.prom {
                load(row, true, &dir);
            }
            Ok(format!("ok: the catalog's {} files are in {}", catalog.prom.len(), dir.display()))
        }
        Cmd::Pack { from, check } => {
            let catalog = read_catalog(&cli.catalog);
            pack(&catalog, cli.catalog.parent().unwrap_or(Path::new(".")), from, check);
            Ok(String::new())
        }
    };
    match result {
        Ok(text) => print!("{text}{}", if text.is_empty() || text.ends_with('\n') { "" } else { "\n" }),
        Err(e) => {
            eprintln!("iris-prom: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(v: &mut [u8], off: usize, n: u64) {
        v[off..off + 8].copy_from_slice(&n.to_be_bytes());
    }

    /// A stored version-six container around `payload`.
    fn container(payload: &[u8], version: u64) -> Vec<u8> {
        let mut v = vec![0; 4096];
        put(&mut v, 0x40, 0x4a46_4b53_5743_534d);
        put(&mut v, 0x48, version);
        put(&mut v, 0x50, (4096 + payload.len()) as u64);
        v[0x7c..0x80].copy_from_slice(&1u32.to_be_bytes());
        v[0x80..0x87].copy_from_slice(b"ip35pro");
        put(&mut v, 0x90, 0x10);
        put(&mut v, 0x98, 4096);
        put(&mut v, 0xb0, payload.len() as u64);
        put(&mut v, 0xc0, payload.iter().map(|b| *b as u64).sum());
        put(&mut v, 0xd0, payload.len() as u64);
        v.extend(payload);
        v
    }

    fn file(name: &str, bytes: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("iris-prom-{}-{name}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// `show` and `extract` describe a good container and give bad input a message.
    #[test]
    fn show_and_extract_report_good_and_bad_files() {
        let good = file("good", &container(&[1, 2, 3, 4], 6));
        let text = show(&good).unwrap();
        assert!(text.contains("version-six container") && text.contains("ip35pro") && text.contains("checksums match"), "{text}");
        let out = std::env::temp_dir().join(format!("iris-prom-{}-out", std::process::id()));
        extract(&good, Some("ip35pro".into()), Some(out.clone())).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), [1, 2, 3, 4]);
        assert!(extract(&good, Some("nosuch".into()), Some(out.clone())).unwrap_err().contains("no segment nosuch"));
        for (name, bytes) in [("v7", container(&[1, 2, 3, 4], 7)), ("cut", container(&[1, 2, 3, 4], 6)[..100].to_vec())] {
            let bad = file(name, &bytes);
            assert!(show(&bad).is_err() && extract(&bad, None, Some(out.clone())).is_err(), "{name}");
            let _ = std::fs::remove_file(bad);
        }
        let _ = std::fs::remove_file(good);
        let _ = std::fs::remove_file(out);
        assert!(show(Path::new("/no/such/prom")).is_err());
    }
}
