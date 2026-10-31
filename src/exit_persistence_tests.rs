//! Guest disk writes must outlive every way the process can end.
//!
//! A copy-on-write overlay keeps which sectors it holds in a record that is
//! written only when the disk is flushed or dropped, and most ways out of
//! iris (a guest power-off, `quit`, a window close) end the process without
//! dropping anything. Each test here builds a whole machine, writes sectors
//! through the SCSI bus with WRITE(10), ends the session by one of those ways,
//! builds the machine again over the same images as the next boot would, and
//! reads every sector back.
//!
//! Every machine gets one disk of each kind at once, so a way out that skips
//! one kind fails the test that names it:
//!
//! | id | disk                                              |
//! |----|---------------------------------------------------|
//! | 1  | raw image with a copy-on-write overlay            |
//! | 2  | raw image written in place                        |
//! | 3  | compressed CHD, written through its diff          |
//! | 4  | CHD with the copy-on-write flag                   |
//! | 5  | scratch volume                                    |
//!
//! The process is not really ended: `IRIS_NO_EXIT_ON_POWEROFF` turns the final
//! `process::exit` into a return, after the flush has run, and the test leaks
//! the machine with `mem::forget` so nothing is dropped either. The exits
//! that end in a drop are tested as drops.
//!
//! The bus is driven at the SCSI command level (WRITE(10) to the target);
//! the controllers' register-level DMA is not exercised, since the disk side
//! of a transfer is the same call. Partial-sector writes are covered where
//! they can happen, in `block_node`'s tests.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use crate::config::{CpuModel, MachineConfig, MachineProfile, ScsiDeviceConfig};
use crate::scsi::{scsi_cmd, ScsiBus, ScsiDataLength, ScsiRequest};
use crate::workstation::{self, Workstation};

const BLOCK: usize = 512;
const BLOCKS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    RawOverlay,
    RawPlain,
    ChdDiff,
    ChdCow,
    Scratch,
}

const DISKS: [(u8, Kind); 5] = [
    (1, Kind::RawOverlay),
    (2, Kind::RawPlain),
    (3, Kind::ChdDiff),
    (4, Kind::ChdCow),
    (5, Kind::Scratch),
];

/// What a fresh image holds, so a read that falls through to the base shows.
fn base_byte(kind: Kind) -> u8 {
    match kind {
        Kind::RawOverlay | Kind::RawPlain => 0xAA,
        Kind::ChdDiff | Kind::ChdCow => 0xAB,
        Kind::Scratch => 0x00,
    }
}

/// A scratch directory under the container's temp dir, removed when dropped.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let p = std::env::temp_dir().join(format!("iris-persist-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
        // `--ci` keeps its overlays under /tmp, named for the process.
        if let Ok(rd) = std::fs::read_dir("/tmp") {
            let prefix = format!("iris-ci-{}-scsi", std::process::id());
            for e in rd.flatten() {
                if e.file_name().to_string_lossy().starts_with(&prefix) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
    }
}

fn make_chd(path: &Path, fill: u8) {
    use libchdman_rs::hd::{create_from_reader, HdCreateOptions};
    use libchdman_rs::CHD_CODEC_ZLIB;
    // Big enough for chdman's geometry guess to keep every sector.
    let logical = 256 * 1024u64;
    create_from_reader(
        std::io::Cursor::new(vec![fill; logical as usize]),
        path,
        HdCreateOptions {
            logical_size: logical,
            hunk_size: 4096,
            unit_size: 512,
            codecs: [CHD_CODEC_ZLIB, 0, 0, 0],
            geometry: None,
            ident: None,
        },
        &mut |_| {},
        &|| false,
    )
    .unwrap();
}

/// The images the disks sit on, made once per test.
fn image_name(kind: Kind) -> String {
    match kind {
        // iris tells a CHD by its extension.
        Kind::ChdDiff | Kind::ChdCow => format!("{kind:?}.chd"),
        _ => format!("{kind:?}"),
    }
}

fn make_images(dir: &Dir) {
    for (_, kind) in DISKS {
        let name = image_name(kind);
        match kind {
            Kind::RawOverlay | Kind::RawPlain | Kind::Scratch => {
                std::fs::write(dir.file(&name), vec![base_byte(kind); BLOCKS * BLOCK]).unwrap();
            }
            Kind::ChdDiff | Kind::ChdCow => make_chd(&dir.file(&name), base_byte(kind)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Machine {
    Indy,
    Indigo2,
    Indigo2Ip28,
    O2,
    Octane,
}

fn config(machine: Machine, dir: &Dir, ci: bool) -> MachineConfig {
    let mut cfg = crate::bench_runner::bench_config([32, 0, 0, 0], CpuModel::default());
    cfg.machine.profile = match machine {
        Machine::Indy => MachineProfile::IndyIp24,
        Machine::Indigo2 => MachineProfile::Indigo2Ip22,
        Machine::Indigo2Ip28 => MachineProfile::Indigo2Ip28,
        Machine::O2 => MachineProfile::O2Ip32,
        Machine::Octane => MachineProfile::OctaneIp30,
    };
    cfg.machine.cpu = match machine {
        Machine::Indy | Machine::Indigo2 => CpuModel::R4400,
        Machine::O2 => CpuModel::R5000,
        Machine::Indigo2Ip28 | Machine::Octane => CpuModel::R10000,
    };
    // The O2 and the Octane have no embedded PROM; an empty one builds.
    match machine {
        Machine::O2 => {
            std::fs::write(dir.file("prom"), vec![0u8; 512 * 1024]).unwrap();
            cfg.prom = dir.file("prom").display().to_string();
        }
        Machine::Octane => {
            std::fs::write(dir.file("prom"), vec![0u8; 1024 * 1024]).unwrap();
            cfg.prom = dir.file("prom").display().to_string();
            cfg.graphics.board = Some(crate::config::GraphicsBoard::None);
        }
        _ => {}
    }
    if matches!(machine, Machine::Octane) {
        cfg.banks = crate::config::banks_from4([64, 0, 0, 0]);
    }
    cfg.nvram = dir.file("nvram").display().to_string();
    cfg.nveeprom = dir.file("nveeprom").display().to_string();
    // CI mode keeps the CPU parked, gives the machine a serial backend for the
    // control socket and moves the overlays to /tmp. Without it the machine is
    // never started, so nothing runs. Port 0: the monitor takes any free port.
    cfg.ci = ci;
    cfg.monitor_port = Some(0);
    cfg.scsi.clear();
    for (id, kind) in DISKS {
        let path = dir.file(&image_name(kind)).display().to_string();
        cfg.scsi.insert(
            id,
            ScsiDeviceConfig {
                path,
                overlay: matches!(kind, Kind::RawOverlay | Kind::ChdCow),
                scratch: kind == Kind::Scratch,
                ..Default::default()
            },
        );
    }
    cfg
}

/// Build the machine on a thread with the stack `Machine::new` wants.
fn build(cfg: MachineConfig) -> Box<dyn Workstation> {
    workstation::build(cfg).expect("machine builds")
}

/// Run `body` on a big-stack thread, as `main` builds and drops machines.
fn on_big_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

/// The SCSI buses of whichever machine this is.
fn with_buses<R>(m: &mut dyn Workstation, f: impl FnOnce(&[&ScsiBus]) -> R) -> R {
    let any = m.as_any_mut();
    if let Some(m) = any.downcast_mut::<crate::machine::Machine>() {
        return f(&m.scsi_buses());
    }
    if let Some(m) = any.downcast_mut::<crate::ip32_machine::O2Machine>() {
        let ctl = m.bus.scsi.lock().unwrap().clone();
        let buses: Vec<&ScsiBus> = ctl.iter().map(|c| c.bus()).collect();
        return f(&buses);
    }
    if let Some(m) = any.downcast_mut::<crate::ip30_machine::OctaneMachine>() {
        let buses: Vec<&ScsiBus> = m.scsi.iter().map(|c| c.bus()).collect();
        return f(&buses);
    }
    panic!("unknown machine type");
}

fn cdb10(op: u8, lba: usize, blocks: usize) -> Vec<u8> {
    let l = (lba as u32).to_be_bytes();
    let n = (blocks as u16).to_be_bytes();
    vec![op, 0, l[0], l[1], l[2], l[3], 0, n[0], n[1], 0]
}

fn write_blocks(m: &mut dyn Workstation, id: u8, lba: usize, data: &[u8]) {
    assert_eq!(data.len() % BLOCK, 0);
    let req = ScsiRequest {
        cdb: cdb10(scsi_cmd::WRITE_10, lba, data.len() / BLOCK),
        data_len: ScsiDataLength::Unlimited,
        data_in: Some(data.to_vec()),
    };
    let r = with_buses(m, |b| b[0].request(id, &req)).expect("target present").expect("write ok");
    if r.status != 0 {
        let sense = ScsiRequest { cdb: vec![scsi_cmd::REQUEST_SENSE, 0, 0, 0, 18, 0], data_len: ScsiDataLength::Unlimited, data_in: None };
        let sense = with_buses(m, |b| b[0].request(id, &sense)).unwrap().unwrap();
        panic!("WRITE(10) to id {id} lba {lba} refused, status {}, sense {:02x?}", r.status, sense.data);
    }
}

fn read_disk(m: &mut dyn Workstation, id: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(BLOCKS * BLOCK);
    for lba in 0..BLOCKS {
        let req = ScsiRequest {
            cdb: cdb10(scsi_cmd::READ_10, lba, 1),
            data_len: ScsiDataLength::Unlimited,
            data_in: None,
        };
        let r = with_buses(m, |b| b[0].request(id, &req)).expect("target present").expect("read ok");
        assert_eq!(r.status, 0, "READ(10) from id {id} lba {lba} refused");
        out.extend_from_slice(&r.data);
    }
    out
}

/// What the disks should hold: starts as the images, takes each write.
struct Expect([Vec<u8>; 5]);

impl Expect {
    fn new() -> Expect {
        Expect(DISKS.map(|(_, kind)| vec![base_byte(kind); BLOCKS * BLOCK]))
    }

    /// One wave of writes to every disk: a three-block span starting at an
    /// odd block, a single block, a block of zeros over non-zero data, the
    /// last block, and (from the second wave) a rewrite of an earlier span and
    /// a final write to fresh ground.
    /// `wave` changes the bytes so a stale sector shows.
    fn write_wave(&mut self, m: &mut dyn Workstation, wave: u8) {
        for (i, (id, _)) in DISKS.iter().enumerate() {
            let mut put = |m: &mut dyn Workstation, lba: usize, data: Vec<u8>| {
                write_blocks(m, *id, lba, &data);
                self.0[i][lba * BLOCK..lba * BLOCK + data.len()].copy_from_slice(&data);
            };
            let fill = |salt: u8, blocks: usize| -> Vec<u8> {
                (0..blocks * BLOCK).map(|n| (n / BLOCK) as u8 ^ salt ^ wave.wrapping_mul(0x31)).collect()
            };
            if wave == 1 {
                put(m, 5, fill(0x11, 3));
                put(m, 0, fill(0x22, 1));
                put(m, 9, vec![0u8; BLOCK]);
                put(m, BLOCKS - 1, fill(0x33, 1));
            } else {
                // Rewrite part of the first wave's span, and add more.
                put(m, 6, fill(0x44, 2));
                put(m, 9, fill(0x66, 1));
                put(m, 20, fill(0x55, 4));
                // Last, and a part of the disk nothing has touched: what a
                // disk that holds writes back would lose.
                put(m, 40, fill(0x77, 1));
            }
        }
    }

    fn check(&self, m: &mut dyn Workstation, what: &str) {
        for (i, (id, kind)) in DISKS.iter().enumerate() {
            let got = read_disk(m, *id);
            if got != self.0[i] {
                let bad: Vec<usize> = (0..BLOCKS).filter(|b| got[b * BLOCK..(b + 1) * BLOCK] != self.0[i][b * BLOCK..(b + 1) * BLOCK]).collect();
                panic!("{what}: disk id {id} ({kind:?}) lost guest writes; blocks that differ: {bad:?}");
            }
        }
    }
}

// ---- ways out ---------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Exit {
    /// The guest switches itself off: the machine-events thread flushes.
    PowerOff,
    /// `quit` over the CI socket.
    CiQuit,
    /// `quit` typed at the terminal console: the monitor hangs up.
    ConsoleQuit,
    /// `main` returning, which is what closing the window comes to.
    MainReturn,
    /// The machine dropped.
    Drop,
}

/// Wait for a suppressed `exit_process` (see `workstation`) to have run.
fn wait_for_exit_request(before: usize) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while workstation::EXIT_REQUESTS.load(Ordering::SeqCst) == before {
        assert!(std::time::Instant::now() < deadline, "the exit never ran");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// The guest's own power-off, down the device path each machine has.
fn guest_power_off(m: &mut dyn Workstation) {
    use crate::dev::ioc3::reg;
    use crate::pci::PciBusDevice;
    use crate::traits::BusDevice;
    let any = m.as_any_mut();
    if let Some(m) = any.downcast_mut::<crate::machine::Machine>() {
        // The IOC's front-panel latch sends this; the chip needs a bus cycle
        // the tests have no cheap way to make.
        m.event_tx.send(crate::traits::MachineEvent::PowerOff).unwrap();
    } else if let Some(m) = any.downcast_mut::<crate::ip32_machine::O2Machine>() {
        // The clock's extended control register, in bank 1, as the PROM sets it.
        let rtc = &m.bus.mace.rtc;
        rtc.write8((0x0a << 8) | 7, 0x10);
        rtc.write8((0x4a << 8) | 7, 0x08);
    } else if let Some(m) = any.downcast_mut::<crate::ip30_machine::OctaneMachine>() {
        let w8 = |a: u32, v: u8| m.ioc3.bar_write(0, a ^ 3, v as u32, 1);
        w8(reg::BYTEBUS_DEV1, 0x0a);
        w8(reg::BYTEBUS_DEV2, 0x10);
        w8(reg::BYTEBUS_DEV1, 0x4a);
        w8(reg::BYTEBUS_DEV2, 0x08);
    } else {
        panic!("unknown machine type");
    }
}

fn ci_send(path: &Path, line: &str) -> String {
    use std::io::{BufRead, Write};
    let mut s = std::os::unix::net::UnixStream::connect(path).unwrap();
    s.write_all(line.as_bytes()).unwrap();
    s.write_all(b"\n").unwrap();
    let mut reply = String::new();
    std::io::BufReader::new(s).read_line(&mut reply).unwrap();
    reply
}

/// End the session `exit`'s way, with the machine left as the process would
/// leave it: not dropped, except by `Exit::Drop`.
fn end_session(mut m: Box<dyn Workstation>, exit: Exit, dir: &Dir) {
    let before = workstation::EXIT_REQUESTS.load(Ordering::SeqCst);
    match exit {
        Exit::PowerOff => {
            guest_power_off(&mut *m);
            wait_for_exit_request(before);
            std::mem::forget(m);
        }
        Exit::CiQuit => {
            let sock = dir.file("ci.sock");
            let _server = crate::ci::start_server(&mut *m, sock.to_str().unwrap()).unwrap();
            let reply = ci_send(&sock, r#"{"cmd":"quit"}"#);
            assert!(reply.contains("\"ok\":true"), "quit refused: {reply}");
            wait_for_exit_request(before);
            std::mem::forget(m);
        }
        Exit::ConsoleQuit => {
            use std::io::Write;
            let addr = m.monitor_addr().expect("monitor is listening");
            let slot = m.machine_slot();
            let mut s = std::net::TcpStream::connect(addr).unwrap();
            let relay = crate::machine::Machine::relay_monitor_output(s.try_clone().unwrap(), workstation::console_quit(slot));
            s.write_all(b"quit\n").unwrap();
            relay.join().unwrap();
            std::mem::forget(m);
        }
        Exit::MainReturn => {
            workstation::stop_and_flush(&mut *m, "exit");
            std::mem::forget(m);
        }
        Exit::Drop => drop(m),
    }
}

/// One session, one exit, and the next boot.
fn round_trip(machine: Machine, exit: Exit, reset_between: bool, ci: bool) {
    // A power-off, quit or console quit would end this process.
    std::env::set_var("IRIS_NO_EXIT_ON_POWEROFF", "1");
    let tag = format!("{machine:?}-{exit:?}-{reset_between}-{ci}");
    on_big_stack(move || {
        let dir = Dir::new(&tag);
        make_images(&dir);
        let cfg = config(machine, &dir, ci);

        let mut m = build(cfg.clone());
        m.register_system_controller();
        if ci {
            m.start();
        }
        let mut expect = Expect::new();
        expect.write_wave(&mut *m, 1);
        expect.check(&mut *m, "same session, before exit");
        if reset_between {
            // A guest reboot: the hardware goes back to power-on and the
            // disks stay.
            m.reset();
            expect.check(&mut *m, "after the reset");
        }
        expect.write_wave(&mut *m, 2);
        end_session(m, exit, &dir);

        // The next boot, over the same files.
        let mut next = build(cfg);
        expect.check(&mut *next, &format!("next boot after {exit:?}"));
        std::mem::forget(next);
    });
}

/// `--ci` tests start the machine (CPU parked) and so can use the control
/// socket, the monitor and, on the O2, the service loop. The `plain` ones
/// leave the machine unstarted and keep the overlays beside their images, as
/// an interactive run does.
macro_rules! round_trips {
    ($($name:ident: $machine:expr, $o2:expr;)*) => {$(
        mod $name {
            use super::*;
            #[test] fn power_off() { round_trip($machine, Exit::PowerOff, false, true) }
            #[test] fn ci_quit() { round_trip($machine, Exit::CiQuit, false, true) }
            #[test] fn console_quit() { round_trip($machine, Exit::ConsoleQuit, false, true) }
            #[test] fn main_return() { round_trip($machine, Exit::MainReturn, false, true) }
            #[test] fn drop() { round_trip($machine, Exit::Drop, false, true) }
            #[test] fn reset_then_power_off() { round_trip($machine, Exit::PowerOff, true, true) }
            #[test] fn reset_then_ci_quit() { round_trip($machine, Exit::CiQuit, true, true) }
            #[test] fn reset_then_drop() { round_trip($machine, Exit::Drop, true, true) }
            #[test] fn plain_main_return() { round_trip($machine, Exit::MainReturn, false, false) }
            #[test] fn plain_drop() { round_trip($machine, Exit::Drop, false, false) }
            #[test] fn plain_reset_then_drop() { round_trip($machine, Exit::Drop, true, false) }
            // The O2 switches itself off from its service loop, which only
            // runs on a started machine.
            #[test] fn plain_power_off() { if !$o2 { round_trip($machine, Exit::PowerOff, false, false) } }
        }
    )*};
}

round_trips! {
    indy: Machine::Indy, false;
    indigo2: Machine::Indigo2, false;
    indigo2_ip28: Machine::Indigo2Ip28, false;
    o2: Machine::O2, true;
    octane: Machine::Octane, false;
}

/// The O2 has a second way to switch off, for a machine nothing is
/// listening to: its service loop flushes the buses and exits itself.
#[test]
fn o2_service_loop_power_off_without_a_controller() {
    std::env::set_var("IRIS_NO_EXIT_ON_POWEROFF", "1");
    on_big_stack(|| {
        let dir = Dir::new("o2-fallback");
        make_images(&dir);
        let cfg = config(Machine::O2, &dir, true);
        let mut m = build(cfg.clone());
        // No register_system_controller.
        m.start();
        let mut expect = Expect::new();
        expect.write_wave(&mut *m, 1);
        let before = workstation::EXIT_REQUESTS.load(Ordering::SeqCst);
        guest_power_off(&mut *m);
        wait_for_exit_request(before);
        std::mem::forget(m);

        let mut next = build(cfg);
        expect.check(&mut *next, "next boot after the O2 service loop's own power-off");
        std::mem::forget(next);
    });
}

// ---- guard: no way out of the process skips the flush -------------------

/// Every `std::process::exit` in the library goes through
/// `workstation::exit_process`, which makes its caller say how the disks are
/// flushed. A new exit anywhere else in the machine and device code fails
/// here until it does the same (or, if it truly cannot touch a disk, is
/// added to the list below with the reason).
#[test]
fn no_process_exit_bypasses_the_flush() {
    // File (under src/) -> exits allowed in it, and why.
    let allowed: &[(&str, usize, &str)] = &[
        ("workstation.rs", 1, "exit_process itself"),
        // Machine::new's two startup failures, before a guest has run.
        ("machine.rs", 2, "bad SCSI configuration at start-up"),
        // The bare-metal test device ends the run with the guest's exit code.
        ("dev/testdev.rs", 1, "bare-metal test device, no disks"),
        // Command-line and configuration front ends: nothing is running yet.
        // (main.rs is skipped: its exits are start-up failures too.)
        ("config.rs", 5, "command-line and configuration errors"),
        ("iris_ci_main.rs", 1, "the iris-ci client"),
        ("crash_diag.rs", 1, "fatal-signal handler: the process is already dying"),
    ];
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found: Vec<(String, usize)> = Vec::new();
    fn walk(dir: &Path, root: &Path, found: &mut Vec<(String, usize)>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                // The binaries are programs of their own.
                if p.file_name().unwrap() != "bin" {
                    walk(&p, root, found);
                }
            } else if p.extension().is_some_and(|x| x == "rs") {
                let text = std::fs::read_to_string(&p).unwrap();
                let rel = p.strip_prefix(root).unwrap().display().to_string();
                if rel.ends_with("exit_persistence_tests.rs") {
                    continue;
                }
                let n = text
                    .lines()
                    .filter(|l| !l.trim_start().starts_with("//"))
                    .filter(|l| l.contains("process::exit("))
                    .count();
                if n > 0 {
                    found.push((rel, n));
                }
            }
        }
    }
    walk(&root, &root, &mut found);
    for (file, n) in &found {
        if file == "main.rs" {
            continue;
        }
        let ok = allowed.iter().any(|(f, max, _)| f == file && n <= max);
        assert!(
            ok,
            "{file} calls std::process::exit {n} time(s). Machine and device code must end the process through \
             workstation::exit_process so the disks are flushed first; if this one cannot touch a disk, add it to \
             this test's list with the reason."
        );
    }
    // A listed file that no longer exits should leave the list, so the list
    // does not grow stale slack another exit could hide in.
    for (f, max, why) in allowed {
        let n = found.iter().find(|(file, _)| file == f).map_or(0, |(_, n)| *n);
        assert_eq!(n, *max, "{f} ({why}) now has {n} exits, the list says {max}");
    }
}

/// Every call of `exit_process` hands it the stop-and-flush work, rather than
/// an empty closure.
#[test]
fn every_exit_process_call_flushes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut calls = 0;
    fn walk(dir: &Path, calls: &mut usize) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, calls);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "rs") || p.ends_with("exit_persistence_tests.rs") {
                continue;
            }
            let lines: Vec<String> = std::fs::read_to_string(&p).unwrap().lines().map(str::to_string).collect();
            for (i, l) in lines.iter().enumerate() {
                let t = l.trim_start();
                if t.starts_with("//") || t.starts_with("pub fn exit_process") || !l.contains("exit_process(") {
                    continue;
                }
                *calls += 1;
                let window = lines[i..lines.len().min(i + 6)].join("\n");
                assert!(
                    window.contains("flush"),
                    "{}:{}: exit_process call that does not flush the disks:\n{window}",
                    p.display(),
                    i + 1
                );
            }
        }
    }
    walk(&root, &mut calls);
    assert!(calls >= 4, "found only {calls} exit_process calls; has the helper been renamed?");
}

/// `main` ends with the stop-and-flush after the window closes. The binary
/// cannot be run here, so look for the call after the event loop returns.
#[test]
fn main_flushes_after_the_window_closes() {
    let main = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs")).unwrap();
    let run = main.find("ui.run(event_loop)").expect("main runs the window");
    let tail = &main[run..];
    assert!(
        tail.contains("stop_and_flush(&mut *machine"),
        "main must stop and flush the machine after the window closes"
    );
}
