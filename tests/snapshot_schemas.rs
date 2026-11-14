//! Snapshots saved by older builds still load.
//!
//! Each directory under `tests/snapshots/` is the `saves/` directory of an
//! older iris build (named in the directory, `<machine>-v<schema>`) that ran
//! the machine on an all-zero PROM and saved it once, as `fixture`.
//! `tests/snapshots/generate.sh` makes them. The test copies one into a fresh
//! working directory, points its provenance at this build, starts the real
//! `iris` binary on the same machine and loads the snapshot through the
//! monitor.
//!
//! A snapshot whose schema this build reads must load; the fixtures hold no
//! PROM or guest data, so what they exercise is the format of each machine's
//! tables and of `chunks.bin`. Three are kept: the Indy at schema 6 (the
//! oldest schema it still reads, which also covers `chunks.bin` before the
//! KONA lists) and the O2 and Octane at schema 7, the ones with a PCI layer
//! whose saved table can change. They are 19.5 KiB as packed git objects.

#![cfg(unix)]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dest = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &dest);
        } else {
            std::fs::copy(e.path(), dest).unwrap();
        }
    }
}

/// Replace the provenance a different host or build would not share. Those
/// say where the snapshot was taken, not what is in it.
fn retarget(manifest: &Path) {
    let text = std::fs::read_to_string(manifest).unwrap();
    let out: Vec<String> = text
        .lines()
        .map(|l| match l.split_once(" = ").map(|(k, _)| k) {
            Some("host_arch") => format!("host_arch = \"{}\"", std::env::consts::ARCH),
            Some("features") => format!("features = {:?}", iris::snapshot::enabled_features()),
            _ => l.to_string(),
        })
        .collect();
    std::fs::write(manifest, out.join("\n") + "\n").unwrap();
}

fn read_until_prompt(s: &mut TcpStream, deadline: Instant) -> String {
    let mut out = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        if out.ends_with(b"\r\n> ") {
            return String::from_utf8_lossy(&out).replace("\r\n", "\n");
        }
        match s.read(&mut buf) {
            Ok(0) => panic!("monitor closed: {:?}", String::from_utf8_lossy(&out)),
            Ok(n) => out.extend_from_slice(&buf[..n]),
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {
                assert!(Instant::now() < deadline, "monitor prompt: {:?}", String::from_utf8_lossy(&out));
            }
            Err(e) => panic!("monitor read: {e}"),
        }
    }
}

/// The monitor's port, read from the line iris prints once it has bound it.
fn monitor_port(stdout: &Path, deadline: Instant) -> u16 {
    loop {
        let out = std::fs::read_to_string(stdout).unwrap_or_default();
        if let Some(port) = out
            .lines()
            .find_map(|l| l.strip_prefix("Monitor listening on 127.0.0.1:"))
            .and_then(|p| p.trim().parse().ok())
        {
            return port;
        }
        assert!(Instant::now() < deadline, "iris never reported its monitor port: {out:?}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn connect(port: u16, deadline: Instant) -> TcpStream {
    loop {
        if let Ok(mut s) = TcpStream::connect(("127.0.0.1", port)) {
            s.set_read_timeout(Some(Duration::from_millis(100))).unwrap();
            read_until_prompt(&mut s, deadline);
            return s;
        }
        assert!(Instant::now() < deadline, "monitor never accepted a connection");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn command(s: &mut TcpStream, cmd: &str) -> String {
    s.write_all(format!("{cmd}\r\n").as_bytes()).unwrap();
    read_until_prompt(s, Instant::now() + Duration::from_secs(60))
}

/// Load `tests/snapshots/<fixture>` into `profile` and check that the load
/// reported success and the machine saved again at this build's schema.
fn loads(fixture: &str, profile: &str, prom_bytes: usize, banks: &str) {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots").join(fixture);
    assert!(src.is_dir(), "fixture {} is missing", src.display());
    let dir: PathBuf = std::env::temp_dir().join(format!("iris-snapshot-schemas-{fixture}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    copy_dir(&src, &dir.join("saves"));
    let manifest = dir.join("saves/fixture/snapshot.toml");
    retarget(&manifest);
    // The cache model is a build default (a build with the translation cache
    // runs the Indy's cache as a shadow), and a snapshot refuses a different
    // one, so ask for the one the fixture was saved with.
    let cache = std::fs::read_to_string(&manifest)
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix("cpu_cache = \"")?.strip_suffix('"').map(str::to_string))
        .expect("the fixture records its cache model");
    std::fs::write(dir.join("prom.bin"), vec![0u8; prom_bytes]).unwrap();

    // Port 0 everywhere: iris binds, the OS picks, and the test reads the
    // monitor's port back, so tests starting together cannot collide.
    std::fs::write(
        dir.join("iris.toml"),
        format!(
            "headless = true\nprom = \"prom.bin\"\nbanks = {banks}\n\
             monitor_port = 0\nserial_port_a = 0\nserial_port_b = 0\nscsi = {{}}\n\
             [machine]\nprofile = \"{profile}\"\ncache = \"{cache}\"\n{}{}",
            // The Indy goes without its HAL2, the others' cards play nowhere.
            if profile == "indy_ip24" { "audio = false\n" } else { "[audio]\noutput = \"none\"\n" },
            // Only the Octane has a graphics board to leave out.
            if profile == "octane_ip30" { "[graphics]\nboard = \"none\"\n" } else { "" }
        ),
    )
    .unwrap();
    let stdout = dir.join("stdout.log");
    let _child = ChildGuard(
        Command::new(env!("CARGO_BIN_EXE_iris"))
            .current_dir(&dir)
            .args(["--config", "iris.toml", "--ci", "--ci-socket"])
            .arg(dir.join("iris.sock"))
            .stdin(Stdio::null())
            .stdout(std::fs::File::create(&stdout).unwrap())
            .stderr(std::fs::File::create(dir.join("stderr.log")).unwrap())
            .spawn()
            .expect("spawn iris"),
    );
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut s = connect(monitor_port(&stdout, deadline), deadline);
    let reply = command(&mut s, "load fixture");
    let out = std::fs::read_to_string(&stdout).unwrap();
    let err = std::fs::read_to_string(dir.join("stderr.log")).unwrap_or_default();
    assert!(out.contains("Snapshot loaded from"), "{fixture} did not load\nreply: {reply}\nstdout: {out}\nstderr: {err}");

    command(&mut s, "save again");
    let m = std::fs::read_to_string(dir.join("saves/again/snapshot.toml")).expect("the machine saved again");
    assert!(m.contains(&format!("schema_version = {}", iris::snapshot::SCHEMA_VERSION)), "{m}");
    let _ = std::fs::remove_dir_all(&dir);
}

const INDY: (&str, usize, &str) = ("indy_ip24", 512 << 10, "[8, 0, 0, 0]");
const O2: (&str, usize, &str) = ("o2_ip32", 512 << 10, "[32, 0, 0, 0]");
const OCTANE: (&str, usize, &str) = ("octane_ip30", 1 << 20, "[64, 0, 0, 0]");

#[test]
fn indy_loads_schema_6() {
    loads("indy-v6", INDY.0, INDY.1, INDY.2);
}
#[test]
fn o2_loads_schema_7() {
    loads("o2-v7", O2.0, O2.1, O2.2);
}
#[test]
fn octane_loads_schema_7() {
    loads("octane-v7", OCTANE.0, OCTANE.1, OCTANE.2);
}
