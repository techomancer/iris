//! The PROMs iris embeds: download them from the catalog, and pack them into
//! `data/proms/<board>.deflate`. Run from the repository root:
//!
//!     cargo run -p iris-prom -- download [--into DIR]    fetch and verify the catalog's files
//!     cargo run -p iris-prom -- pack [--from DIR]        unwrap and deflate them into data/proms
//!     cargo run -p iris-prom -- pack --check             fail unless data/proms holds exactly the
//!                                                        catalog's PROMs
//!
//! `data/proms/proms.toml` lists each published file with its size and sha256. What is
//! stored is the PROM a machine runs: the file unwrapped by `prom_image::decode` (the
//! installer header, the version-six container and its gzip segment); a row with
//! `unwrap = false` is stored as published, because the IP27 PROM unpacks the IO6
//! container from the BaseIO flash itself (unwrapped, it rejects the magic and falls
//! back to its own copy). A file whose PROM has not changed is left untouched, so git
//! sees only the PROMs that did. Never commit an extracted PROM: the packed files and
//! the catalog are all that belong in the repository.

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
    /// The catalog of published PROM files.
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
    let sha: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(sha, row.sha256, "{name}: sha256");
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

fn default_dir() -> PathBuf {
    std::env::temp_dir().join("iris-prom")
}

fn main() {
    let cli = Cli::parse();
    let catalog = read_catalog(&cli.catalog);
    let out = cli.catalog.parent().unwrap_or(Path::new("."));
    match cli.cmd {
        Cmd::Download { into } => {
            let dir = into.unwrap_or_else(default_dir);
            for row in &catalog.prom {
                load(row, true, &dir);
            }
            println!("ok: the catalog's {} files are in {}", catalog.prom.len(), dir.display());
        }
        Cmd::Pack { from, check } => pack(&catalog, out, from, check),
    }
}
