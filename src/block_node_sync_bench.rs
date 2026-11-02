//! What a synced write costs on each disk backend, and what a killed process
//! leaves behind. Both are ignored: they measure the host's disk.
//!
//! ```text
//! TMPDIR=/some/real/disk cargo test --release --lib block_node_sync_bench \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Put `TMPDIR` on the disk you care about: a tmpfs makes every sync free.
//! `BENCH_MB` sets the disk size (default 256).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use libchdman_rs::hd::{create_from_reader, HdCreateOptions};
use libchdman_rs::CHD_CODEC_ZLIB;

use crate::block_node::{BlockNode, ChdNode, CowNode, RawNode};

fn bench_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("iris-sync-bench-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn disk_bytes() -> u64 {
    std::env::var("BENCH_MB").ok().and_then(|s| s.parse().ok()).unwrap_or(256u64) << 20
}

/// A hard-disk CHD of `len` bytes of `byte`, compressed or not.
fn make_chd(path: &Path, len: u64, compressed: bool, byte: u8) {
    let codecs = if compressed { [CHD_CODEC_ZLIB, 0, 0, 0] } else { [0; 4] };
    let src = std::io::repeat(byte);
    use std::io::Read;
    create_from_reader(
        src.take(len),
        path,
        HdCreateOptions { logical_size: len, hunk_size: 4096, unit_size: 512, codecs, geometry: None, ident: None },
        &mut |_| {},
        &|| false,
    )
    .unwrap();
}

/// The byte offset of write `i` of `size` bytes: spread over the disk, so a
/// diff allocates a new hunk for every write of the first pass.
fn offset(i: u64, size: u64, disk: u64) -> u64 {
    (i * 7919 % (disk / size)) * size
}

/// Microseconds per write of `size` bytes, each followed by `flush_data`,
/// for a first pass over fresh offsets and a second over the same ones.
fn time_synced(node: &mut dyn BlockNode, size: usize, count: u64) -> (f64, f64) {
    let disk = node.size();
    let buf = vec![0x5Au8; size];
    let mut pass = || {
        let t = Instant::now();
        for i in 0..count {
            node.write(offset(i, size as u64, disk), &buf).unwrap();
            node.flush_data().unwrap();
        }
        t.elapsed().as_secs_f64() * 1e6 / count as f64
    };
    let first = pass();
    let again = pass();
    (first, again)
}

#[test]
#[ignore]
fn bench_synced_writes() {
    let disk = disk_bytes();
    let dir = bench_dir("bench");
    println!("disk {} MiB in {}", disk >> 20, dir.display());
    for (size, count) in [(4096usize, 400u64), (65536, 200)] {
        let raw = dir.join("disk.raw");
        std::fs::write(&raw, vec![0u8; disk as usize]).unwrap();
        let r = time_synced(&mut RawNode::open(&raw).unwrap(), size, count);

        let overlay = dir.join("disk.raw.overlay");
        let base = RawNode::open_readonly(&raw).unwrap();
        let mut cow = CowNode::new(Some(Box::new(base)), &overlay, disk).unwrap();
        let c = time_synced(&mut cow, size, count);
        drop(cow);

        let flat = dir.join("flat.chd");
        make_chd(&flat, disk, false, 0);
        let f = time_synced(&mut ChdNode::open(flat.to_str().unwrap(), false).unwrap(), size, count);

        let packed = dir.join("packed.chd");
        make_chd(&packed, disk, true, 0xAB);
        let d = time_synced(&mut ChdNode::open(packed.to_str().unwrap(), false).unwrap(), size, count);

        for (name, (first, again)) in
            [("raw", r), ("raw+overlay", c), ("chd in place", f), ("chd diff", d)]
        {
            println!("{:>6} B  {name:<13} first {first:>9.1} us  again {again:>9.1} us", size);
        }
        for p in std::fs::read_dir(&dir).unwrap() {
            let _ = std::fs::remove_file(p.unwrap().path());
        }
    }
    let _ = std::fs::remove_dir(&dir);
}

/// The value a write `i` fills its sectors with.
fn pattern(i: u64) -> u8 {
    (i % 251) as u8 + 1
}

/// The child half of [`kill_mid_write_leaves_a_chd_that_opens`]: write and
/// sync forever, saying which write has been acknowledged after each.
#[test]
#[ignore]
fn kill_child() {
    let Ok(path) = std::env::var("IRIS_KILL_CHD") else { return };
    let mut node = ChdNode::open(&path, false).unwrap();
    let disk = node.size();
    for i in 0.. {
        let buf = vec![pattern(i); 8192];
        node.write(offset(i, 8192, disk), &buf).unwrap();
        node.flush_data().unwrap();
        println!("ack {i}");
    }
}

/// Kill a process in the middle of synced writes to a CHD diff, then open the
/// CHD again: every acknowledged write must read back, and every other range
/// must hold the base or a write the child made.
#[test]
#[ignore]
fn kill_mid_write_leaves_a_chd_that_opens() {
    let dir = bench_dir("kill");
    let disk = 16u64 << 20;
    for round in 0..5u64 {
        let base = dir.join(format!("base{round}.chd"));
        make_chd(&base, disk, true, 0xAB);
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "block_node_sync_bench::kill_child", "--ignored", "--nocapture", "--test-threads=1"])
            .env("IRIS_KILL_CHD", &base)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut acked = None;
        let target = 50 + round * 97;
        // Read on after the kill: the child may have acknowledged more.
        for line in BufReader::new(child.stdout.take().unwrap()).lines() {
            let Ok(line) = line else { break };
            if let Some(n) = line.strip_prefix("ack ") {
                let n: u64 = n.trim().parse().unwrap();
                acked = Some(n);
                if n == target {
                    child.kill().unwrap();
                }
            }
        }
        let _ = child.wait();
        let acked = acked.expect("the child acknowledged writes");

        let mut node = ChdNode::open(base.to_str().unwrap(), false).expect("the CHD opens after a kill");
        let mut last = std::collections::HashMap::new();
        for i in 0..=acked {
            last.insert(offset(i, 8192, disk), i);
        }
        let mut buf = vec![0u8; 8192];
        for off in (0..disk).step_by(8192) {
            node.read(off, &mut buf).unwrap();
            match last.get(&off) {
                Some(&i) => {
                    let ok = buf.iter().all(|&b| b == pattern(i))
                        || buf.iter().all(|&b| b == pattern(acked + 1) || b == pattern(i));
                    assert!(ok, "round {round}: acknowledged write {i} at {off:#x} reads back");
                }
                None => {
                    let ok = buf.iter().all(|&b| b == 0xAB || b == pattern(acked + 1));
                    assert!(ok, "round {round}: {off:#x} holds the base");
                }
            }
        }
        drop(node);
        // Not `Chd::verify`: MAME never updates the header's checksum when it
        // writes, so no CHD that has been written verifies, killed or not.
        println!("round {round}: killed after write {acked}, CHD opens and reads back");
        let _ = std::fs::remove_file(crate::chd_disk::diff_path_for(&base));
        let _ = std::fs::remove_file(&base);
    }
    let _ = std::fs::remove_dir(&dir);
}
