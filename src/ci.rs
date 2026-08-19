//! CI control socket.
//!
//! Unix domain socket (Linux/macOS) or TCP localhost (Windows) that drives the
//! emulator for automated testing. The protocol is newline-delimited JSON,
//! strict request/response, single client. See `ci_mode_plan.md` in the repo root.

use std::io::{BufRead, BufReader, Read, Write};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::net::{TcpListener, TcpStream};

use crate::config::{ci_socket_is_tcp, ci_socket_tcp_addr};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::machine::Machine;
use crate::rex3::Rex3;
use crate::z85c30::CiSerialBackend;

/// Set at `start_server`; consulted by `quit` so the socket file is cleaned up
/// before `std::process::exit` (which skips Drop).
static SOCKET_PATH: Mutex<Option<String>> = Mutex::new(None);

#[derive(Deserialize)]
struct Request {
    cmd: String,
    #[serde(default)]
    args: Value,
}

#[derive(Serialize)]
struct Response {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl Response {
    fn ok() -> Self { Self { ok: true, data: None, error: None } }
    fn data(v: Value) -> Self { Self { ok: true, data: Some(v), error: None } }
    fn err(msg: impl Into<String>) -> Self {
        Self { ok: false, data: None, error: Some(msg.into()) }
    }
}

// ----------------------------------------------------------------------------
// Server
// ----------------------------------------------------------------------------

/// Holder for the raw `*mut Machine` passed in from `main`. The pointer is
/// valid for the process lifetime because `Machine` lives on main's stack.
/// Mirrors the `SystemController` pattern in `machine.rs`.
struct MachinePtr(*mut Machine);
unsafe impl Send for MachinePtr {}
unsafe impl Sync for MachinePtr {}

pub struct CiServer {
    socket_path: String,
    machine: Arc<Mutex<MachinePtr>>,
    ci_serial: Arc<CiSerialBackend>,
    /// Optional in case --headless is also passed (no REX3). Screenshot
    /// commands return an error in that case.
    rex3: Option<Arc<Rex3>>,
    rex3_head1: Option<Arc<Rex3>>,
}

impl Drop for CiServer {
    fn drop(&mut self) {
        cleanup_socket_path(&self.socket_path);
    }
}

fn cleanup_socket_path(path: &str) {
    if ci_socket_is_tcp(path) {
        return;
    }
    #[cfg(unix)]
    let _ = std::fs::remove_file(path);
}

impl CiServer {
    fn with_machine<R>(&self, f: impl FnOnce(&mut Machine) -> R) -> R {
        let mut guard = self.machine.lock();
        // SAFETY: pointer is valid for process lifetime; this mutex serializes
        // all Machine accesses from CI command handlers. CPU/peripheral threads
        // observe state changes only when the methods we call stop them first
        // (ci_restore/ci_rollback do).
        let machine = unsafe { &mut *(guard.0) };
        f(machine)
    }
}

/// Bind the control socket, spawn the accept thread, return a handle.
///
/// # Safety
/// `machine_ptr` must remain valid for the process lifetime. Pass the address
/// of a `Machine` owned by `main`'s stack (or a heap-pinned Box that `main`
/// keeps alive).
pub fn start_server(
    machine_ptr: *mut Machine,
    socket_path: &str,
) -> Result<Arc<CiServer>, String> {
    // SAFETY: caller guarantees the pointer is valid.
    let ci_serial = unsafe { (*machine_ptr).get_ci_serial() }
        .ok_or_else(|| "CI mode: CiSerialBackend not installed on Machine".to_string())?;
    let rex3 = unsafe { (*machine_ptr).get_rex3() };
    let rex3_head1 = unsafe { (*machine_ptr).get_rex3_head1() };

    let path = socket_path.to_string();
    let server = Arc::new(CiServer {
        socket_path: path.clone(),
        machine: Arc::new(Mutex::new(MachinePtr(machine_ptr))),
        ci_serial,
        rex3,
        rex3_head1,
    });

    *SOCKET_PATH.lock() = Some(path.clone());

    if ci_socket_is_tcp(&path) {
        start_tcp_server(server.clone(), &path)?;
    } else {
        #[cfg(unix)]
        start_unix_server(server.clone(), &path)?;
        #[cfg(not(unix))]
        return Err(format!(
            "CI socket path {} requires a Unix domain socket; use host:port on this platform",
            path
        ));
    }

    Ok(server)
}

#[cfg(unix)]
fn start_unix_server(server: Arc<CiServer>, path: &str) -> Result<(), String> {
    cleanup_socket_path(path);
    let listener = UnixListener::bind(path)
        .map_err(|e| format!("failed to bind {}: {}", path, e))?;
    eprintln!("iris: --ci control socket listening at {}", path);
    let server_clone = server.clone();
    thread::Builder::new()
        .name("iris-ci-accept".into())
        .spawn(move || {
            for conn in listener.incoming() {
                match conn {
                    Ok(stream) => {
                        let s = server_clone.clone();
                        thread::Builder::new()
                            .name("iris-ci-handler".into())
                            .spawn(move || handle_client(s, stream))
                            .ok();
                    }
                    Err(e) => eprintln!("iris-ci-accept: {}", e),
                }
            }
        })
        .map_err(|e| format!("failed to spawn CI accept thread: {}", e))?;
    Ok(())
}

#[cfg(windows)]
fn start_tcp_server(server: Arc<CiServer>, path: &str) -> Result<(), String> {
    start_tcp_server_impl(server, path)
}

#[cfg(not(windows))]
fn start_tcp_server(server: Arc<CiServer>, path: &str) -> Result<(), String> {
    start_tcp_server_impl(server, path)
}

fn start_tcp_server_impl(server: Arc<CiServer>, path: &str) -> Result<(), String> {
    let addr = ci_socket_tcp_addr(path);
    let listener = TcpListener::bind(&addr)
        .map_err(|e| format!("failed to bind TCP CI socket {}: {}", addr, e))?;
    eprintln!("iris: --ci control socket listening at tcp:{}", addr);
    let server_clone = server.clone();
    thread::Builder::new()
        .name("iris-ci-accept".into())
        .spawn(move || {
            for conn in listener.incoming() {
                match conn {
                    Ok(stream) => {
                        let _ = stream.set_nodelay(true);
                        let s = server_clone.clone();
                        thread::Builder::new()
                            .name("iris-ci-handler".into())
                            .spawn(move || handle_client_tcp(s, stream))
                            .ok();
                    }
                    Err(e) => eprintln!("iris-ci-accept: {}", e),
                }
            }
        })
        .map_err(|e| format!("failed to spawn CI accept thread: {}", e))?;
    Ok(())
}

fn handle_client_lines(server: Arc<CiServer>, reader: impl BufRead, mut writer: impl Write) {
    for line in reader.lines() {
        let Ok(line) = line else { break };
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        let response = match serde_json::from_str::<Request>(trimmed) {
            Ok(req) => dispatch(&server, &req),
            Err(e) => Response::err(format!("invalid json: {}", e)),
        };

        let mut out = match serde_json::to_vec(&response) {
            Ok(v) => v,
            Err(e) => serde_json::to_vec(&Response::err(format!("encode: {}", e))).unwrap_or_default(),
        };
        out.push(b'\n');
        if writer.write_all(&out).is_err() { break; }
    }
}

#[cfg(unix)]
fn handle_client(server: Arc<CiServer>, stream: UnixStream) {
    match stream.try_clone() {
        Ok(writer) => {
            let reader = BufReader::new(stream);
            handle_client_lines(server, reader, writer);
        }
        Err(e) => eprintln!("iris-ci-handler: clone failed: {}", e),
    }
}

fn handle_client_tcp(server: Arc<CiServer>, stream: TcpStream) {
    match stream.try_clone() {
        Ok(writer) => {
            let reader = BufReader::new(stream);
            handle_client_lines(server, reader, writer);
        }
        Err(e) => eprintln!("iris-ci-handler: clone failed: {}", e),
    }
}

// ----------------------------------------------------------------------------
// Dispatch
// ----------------------------------------------------------------------------

fn dispatch(server: &CiServer, req: &Request) -> Response {
    match req.cmd.as_str() {
        "ping" => Response::ok(),
        "quit" => cmd_quit(server, &req.args),
        "start" => cmd_start(server),
        "save" => cmd_save(server, &req.args),
        "restore" => cmd_restore(server, &req.args),
        "rollback" => cmd_rollback(server),
        "list" => cmd_list(&req.args),
        "info" => cmd_info(&req.args),
        "delete" => cmd_delete(&req.args),
        "serial-send" => cmd_serial_send(server, &req.args),
        "serial-read" => cmd_serial_read(server),
        "wait-serial" => cmd_wait_serial(server, &req.args),
        "screenshot" => cmd_screenshot(server, &req.args),
        "scratch-write" => cmd_scratch_write(server, &req.args),
        "scratch-read"  => cmd_scratch_read(server, &req.args),
        "scratch-clear" => cmd_scratch_clear(server),
        "scratch-info"  => cmd_scratch_info(server),
        "validate"      => cmd_validate(server, &req.args),
        "gc"            => cmd_gc(),
        "diff"          => cmd_diff(&req.args),
        "tree"          => cmd_tree(),
        "pull"          => cmd_pull(&req.args),
        "push"          => cmd_push(&req.args),
        "rtc-save"      => cmd_rtc_save(server, &req.args),
        "cdrom-eject"   => cmd_cdrom_eject(server, &req.args),
        "cdrom-load"    => cmd_cdrom_load(server, &req.args),
        "chd-sync"      => cmd_chd_sync(server, &req.args),
        other => Response::err(format!("unknown command: {}", other)),
    }
}

fn cmd_rtc_save(server: &CiServer, args: &Value) -> Response {
    let explicit_path = args.get("path").and_then(|v| v.as_str()).map(|s| s.to_string());
    let result = server.with_machine(|m| {
        let rtc = m.hpc3().rtc();
        let path = explicit_path.clone().unwrap_or_else(|| rtc.nvram_path().to_string());
        rtc.save_nvram(&path).map(|_| path)
    });
    match result {
        Ok(path) => Response::data(serde_json::json!({ "path": path })),
        Err(e) => Response::err(format!("rtc-save: {}", e)),
    }
}

fn cmd_cdrom_eject(server: &CiServer, args: &Value) -> Response {
    let id = match args.get("id").and_then(|v| v.as_u64()) {
        Some(n) => n as usize,
        None => return Response::err("cdrom-eject: missing 'id' arg"),
    };
    let result = server.with_machine(|m| {
        m.hpc3().scsi().eject_disc(id)
    });
    match result {
        Ok(path) => Response::data(serde_json::json!({ "id": id, "new_disc": path })),
        Err(e) => Response::err(format!("cdrom-eject: {}", e)),
    }
}

fn cmd_cdrom_load(server: &CiServer, args: &Value) -> Response {
    let id = match args.get("id").and_then(|v| v.as_u64()) {
        Some(n) => n as usize,
        None => return Response::err("cdrom-load: missing 'id' arg"),
    };
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => return Response::err("cdrom-load: missing 'path' arg"),
    };
    let result = server.with_machine(|m| {
        m.hpc3().scsi().load_disc(id, path.clone())
    });
    match result {
        Ok(loaded) => Response::data(serde_json::json!({ "id": id, "disc": loaded })),
        Err(e) => Response::err(format!("cdrom-load: {}", e)),
    }
}

/// Parse a SCSI target selector: absent, null or "all" mean every disk.
fn scsi_selector(v: Option<&Value>) -> Result<Option<usize>, String> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(v) if v.as_str() == Some("all") => Ok(None),
        Some(v) => match v.as_u64() {
            Some(n) if n < 8 => Ok(Some(n as usize)),
            _ => Err(format!("bad scsi id {v} (expected 0-7 or \"all\")")),
        },
    }
}

/// Fold pending CHD diffs into their bases. The guest must be stopped: folding
/// releases the disk backend, so a running IRIX would lose the drive mid-flight.
fn cmd_chd_sync(server: &CiServer, args: &Value) -> Response {
    let only = match scsi_selector(args.get("id")) {
        Ok(v) => v,
        Err(e) => return Response::err(e),
    };
    if server.with_machine(|m| m.cpu_is_running()) {
        return Response::err("CPU is running — stop the guest first \
                              (folding releases the disk backend)");
    }
    match server.with_machine(|m| m.sync_chd_disks(only, &mut |_, _, _| {}, &|| false)) {
        Ok(n) => Response::data(serde_json::json!({ "synced": n })),
        Err(e) => Response::err(e.to_string()),
    }
}

fn cmd_quit(server: &CiServer, args: &Value) -> Response {
    // `sync_chd` present => fold before exiting. Stop the machine first so the
    // fold is quiesced, mirroring the GUI's "Synchronizing disks…" exit step.
    let mut synced = serde_json::Value::Null;
    if args.get("sync_chd").is_some() {
        let only = match scsi_selector(args.get("sync_chd")) {
            Ok(v) => v,
            Err(e) => return Response::err(e),
        };
        let r = server.with_machine(|m| {
            m.stop();
            m.sync_chd_disks(only, &mut |_, _, _| {}, &|| false)
        });
        match r {
            Ok(n) => synced = serde_json::json!(n),
            Err(e) => return Response::err(format!("chd sync failed: {e}")),
        }
    }
    // Schedule process exit after a brief delay so the response flushes.
    thread::spawn(|| {
        thread::sleep(Duration::from_millis(50));
        if let Some(p) = SOCKET_PATH.lock().take() {
            cleanup_socket_path(&p);
        }
        // Same escape hatch as the PowerOff handler: library hosts set
        // IRIS_NO_EXIT_ON_POWEROFF=1 so a `quit` over the CI socket does
        // not kill the embedder.
        if std::env::var_os("IRIS_NO_EXIT_ON_POWEROFF").is_none() {
            std::process::exit(0);
        }
    });
    Response::data(serde_json::json!({ "synced": synced }))
}

fn cmd_start(server: &CiServer) -> Response {
    server.with_machine(|m| m.cpu_start());
    Response::ok()
}

fn cmd_save(server: &CiServer, args: &Value) -> Response {
    let name = match args.get("name").and_then(|v| v.as_str()) {
        Some(n) => n.to_string(),
        None => return Response::err("save: missing 'name' arg"),
    };
    match server.with_machine(|m| m.save_snapshot(&name)) {
        Ok(()) => Response::ok(),
        Err(e) => Response::err(format!("save failed: {}", e)),
    }
}

fn cmd_restore(server: &CiServer, args: &Value) -> Response {
    let name = match args.get("name").and_then(|v| v.as_str()) {
        Some(n) => n.to_string(),
        None => return Response::err("restore: missing 'name' arg"),
    };
    match server.with_machine(|m| m.ci_restore(&name)) {
        Ok(()) => Response::ok(),
        Err(e) => Response::err(format!("restore failed: {}", e)),
    }
}

fn cmd_rollback(server: &CiServer) -> Response {
    match server.with_machine(|m| m.ci_rollback()) {
        Ok(()) => Response::ok(),
        Err(e) => Response::err(format!("rollback failed: {}", e)),
    }
}

fn cmd_list(_args: &Value) -> Response {
    // Walk saves/ recursively, return every directory that contains a
    // snapshot.toml (current format) OR a cpu.toml (legacy v0). Names are
    // returned slash-joined relative to saves/.
    let root = std::path::Path::new("saves");
    if !root.is_dir() {
        return Response::data(serde_json::json!({ "snapshots": [] }));
    }
    let mut out: Vec<String> = Vec::new();
    let mut stack: Vec<std::path::PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let mut subdirs: Vec<std::path::PathBuf> = Vec::new();
        let mut is_snapshot = false;
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                subdirs.push(p);
            } else if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if name == "snapshot.toml" || name == "cpu.toml" {
                    is_snapshot = true;
                }
            }
        }
        if is_snapshot {
            if let Ok(rel) = dir.strip_prefix(root) {
                let s = rel.to_string_lossy().replace('\\', "/");
                if !s.is_empty() {
                    out.push(s);
                }
            }
        }
        for s in subdirs {
            stack.push(s);
        }
    }
    out.sort();
    Response::data(serde_json::json!({ "snapshots": out }))
}

fn cmd_info(args: &Value) -> Response {
    let name = match args.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => return Response::err("info: missing 'name' arg"),
    };
    let dir = std::path::Path::new("saves").join(name);
    if !dir.is_dir() {
        return Response::err(format!("info: snapshot '{}' not found", name));
    }
    let snap = crate::snapshot::Snapshot::new(&dir);
    let manifest = match snap.read_manifest() {
        Ok(Some(m)) => Some(m),
        Ok(None) => None,
        Err(e) => return Response::err(format!("info: manifest read failed: {}", e)),
    };

    // Disk usage rollup: sum file sizes inside the snapshot dir.
    let mut bytes_on_disk: u64 = 0;
    if let Ok(walker) = std::fs::read_dir(&dir) {
        for e in walker.flatten() {
            if let Ok(meta) = e.metadata() {
                if meta.is_file() {
                    bytes_on_disk += meta.len();
                }
            }
        }
    }

    let mut out = serde_json::Map::new();
    out.insert("name".into(), Value::String(name.to_string()));
    out.insert("bytes_on_disk".into(), Value::Number(bytes_on_disk.into()));
    if let Some(m) = manifest {
        out.insert("schema_version".into(), Value::Number(m.schema_version.into()));
        out.insert("host_arch".into(), Value::String(m.host_arch));
        out.insert("created_at_unix".into(), Value::Number(m.created_at_unix.into()));
        if let Some(rev) = m.iris_git_rev { out.insert("iris_git_rev".into(), Value::String(rev)); }
        if let Some(p) = m.parent { out.insert("parent".into(), Value::String(p)); }
        if let Some(d) = m.description { out.insert("description".into(), Value::String(d)); }
        out.insert("installed_bundles".into(),
            Value::Array(m.installed_bundles.into_iter().map(Value::String).collect()));
    } else {
        out.insert("schema_version".into(), Value::Number(0.into()));
        out.insert("legacy".into(), Value::Bool(true));
    }
    Response::data(Value::Object(out))
}

fn cmd_delete(args: &Value) -> Response {
    let name = match args.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => return Response::err("delete: missing 'name' arg"),
    };
    if name.is_empty() || name.contains("..") {
        return Response::err("delete: invalid name");
    }
    let dir = std::path::Path::new("saves").join(name);
    if !dir.is_dir() {
        return Response::err(format!("delete: snapshot '{}' not found", name));
    }
    if let Err(e) = std::fs::remove_dir_all(&dir) {
        return Response::err(format!("delete: {}: {}", dir.display(), e));
    }
    Response::ok()
}

fn cmd_serial_send(server: &CiServer, args: &Value) -> Response {
    let data = match args.get("data").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return Response::err("serial-send: missing 'data' arg"),
    };
    server.ci_serial.push_host(data.as_bytes());
    Response::ok()
}

fn cmd_serial_read(server: &CiServer) -> Response {
    let bytes = server.ci_serial.drain_guest();
    let s = String::from_utf8_lossy(&bytes).into_owned();
    Response::data(Value::String(s))
}

fn cmd_screenshot(server: &CiServer, args: &Value) -> Response {
    let head = args.get("head").and_then(|v| v.as_u64()).unwrap_or(0);
    let rex3 = match head {
        0 => server.rex3.as_ref(),
        1 => server.rex3_head1.as_ref(),
        _ => return Response::err("screenshot: head must be 0 or 1"),
    };
    let Some(rex3) = rex3 else {
        return Response::err(format!(
            "screenshot: REX3 head {} not present (headless or graphics.heads < 2?)",
            head
        ));
    };
    let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
        return Response::err("screenshot: missing 'path' arg");
    };

    // Snapshot the framebuffer under the screen lock; unlock before the PNG
    // encode so the refresh thread isn't blocked during disk I/O.
    let (width, height, rgba_copy) = {
        let screen = rex3.screen.lock();
        let w = screen.width;
        let h = screen.height;
        let mut out = Vec::with_capacity(w * h);
        // `rgba` has row stride 2048; copy the visible window.
        for y in 0..h {
            let base = y * 2048;
            out.extend_from_slice(&screen.rgba[base..base + w]);
        }
        (w, h, out)
    };

    // Encode each u32 0xFFRRGGBB as 3 RGB bytes in the order the PNG encoder
    // expects.
    let mut rgb = Vec::with_capacity(width * height * 3);
    for px in &rgba_copy {
        rgb.push(((px >> 16) & 0xff) as u8);
        rgb.push(((px >> 8) & 0xff) as u8);
        rgb.push((px & 0xff) as u8);
    }

    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => return Response::err(format!("screenshot: create {}: {}", path, e)),
    };
    let bw = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(bw, width as u32, height as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = match enc.write_header() {
        Ok(w) => w,
        Err(e) => return Response::err(format!("screenshot: png header: {}", e)),
    };
    if let Err(e) = writer.write_image_data(&rgb) {
        return Response::err(format!("screenshot: png write: {}", e));
    }

    Response::data(serde_json::json!({
        "path": path,
        "head": head,
        "width": width,
        "height": height,
        "bytes": rgb.len() + 100,  // rough
    }))
}

fn cmd_wait_serial(server: &CiServer, args: &Value) -> Response {
    let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => return Response::err("wait-serial: missing 'pattern' arg"),
    };
    let timeout_ms = args.get("timeout_ms").and_then(|v| v.as_u64()).unwrap_or(10_000);

    match server.ci_serial.wait_for(pattern.as_bytes(), Duration::from_millis(timeout_ms)) {
        Some(consumed) => {
            let s = String::from_utf8_lossy(&consumed).into_owned();
            Response::data(Value::String(s))
        }
        None => Response::err(format!("wait-serial: timeout after {}ms waiting for {:?}", timeout_ms, pattern)),
    }
}

// ----------------------------------------------------------------------------
// Scratch volume (Phase 2.4): file injection / extraction without networking.
//
// The scratch device is a raw SCSI LUN (`scratch = true` in iris.toml).
// iris pre-formats the underlying file with a minimal SGI Volume Header at
// sector 0 so IRIX recognises it (without the VH, /dev/rdsk/dks0dN*
// returns I/O error on every read). The VH defines three partitions
// (see src/sgi_vh.rs):
//   - slot  0 ("s0",  payload):     PT_RAW,    sectors 8..end
//   - slot  8 ("vh",  volhdr):      PT_VOLHDR, sectors 0..7
//   - slot 10 ("vol", whole-disk):  PT_VOLUME, sectors 0..end
//
// Wire convention:
//   - `scratch-write` and `scratch-read` operate on the *payload* area —
//     `offset = 0` means the first byte after the VH (raw byte 4096 in the
//     underlying file). The VH is never touched by these commands.
//   - The guest reads/writes the same payload at offset 0 of
//     /dev/rdsk/dks0dNs0 (NOT /dev/rdsk/dks0dNvol — that one starts at
//     sector 0 and so begins with the volume header itself; writing to it
//     also clobbers the VH so future scratch ops misalign).
//   - Typical guest read:  `dd if=/dev/rdsk/dks0d2s0 bs=64k | tar xf -`.
//   - Typical guest write: `dd if=mydata of=/dev/rdsk/dks0d2s0 bs=512 conv=sync`
//     (sector-aligned; conv=sync zero-pads the trailing partial sector).
//
// Each scratch op briefly stops the machine to quiesce in-flight SCSI I/O
// (Machine::with_paused). The CPU is restarted only if it was running before
// — a scratch-write issued before the harness `start`s the CPU does not
// auto-start it.
// ----------------------------------------------------------------------------

use crate::sgi_vh::SCRATCH_PAYLOAD_OFFSET;

/// Reject names that would escape the host or smuggle in shell metachars. The
/// host-side path is read by serde_json so quoting is already handled, but a
/// caller-supplied "../" can still escape an intended sandbox.
fn validate_host_path(p: &str) -> Result<std::path::PathBuf, String> {
    if p.is_empty() {
        return Err("path: empty".into());
    }
    let pb = std::path::PathBuf::from(p);
    if pb.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return Err(format!("path: '..' components not allowed in {:?}", p));
    }
    Ok(pb)
}

fn cmd_scratch_write(server: &CiServer, args: &Value) -> Response {
    let host_path = match args.get("host_path").and_then(|v| v.as_str()) {
        Some(p) => match validate_host_path(p) {
            Ok(pb) => pb,
            Err(e) => return Response::err(format!("scratch-write: {}", e)),
        },
        None => return Response::err("scratch-write: missing 'host_path' arg"),
    };
    let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0);

    let bytes = match std::fs::read(&host_path) {
        Ok(b) => b,
        Err(e) => return Response::err(format!("scratch-write: read {}: {}", host_path.display(), e)),
    };

    let result = server.with_machine(|m| {
        let scratch = match m.scratch_path() {
            Some(p) => p.to_path_buf(),
            None => return Err("scratch volume not configured (set `scratch = true` on a SCSI device in iris.toml)".to_string()),
        };
        m.with_paused(|| -> Result<u64, String> {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .open(&scratch)
                .map_err(|e| format!("open {}: {}", scratch.display(), e))?;
            // Skip the VH partition; offset is relative to the payload area.
            let raw_offset = SCRATCH_PAYLOAD_OFFSET.checked_add(offset)
                .ok_or_else(|| "offset overflow".to_string())?;
            f.seek(SeekFrom::Start(raw_offset)).map_err(|e| format!("seek: {}", e))?;
            f.write_all(&bytes).map_err(|e| format!("write: {}", e))?;
            f.sync_all().map_err(|e| format!("fsync: {}", e))?;
            Ok(bytes.len() as u64)
        })
    });

    match result {
        Ok(n) => Response::data(serde_json::json!({
            "bytes_written": n,
            "offset": offset,
            "host_path": host_path.display().to_string(),
        })),
        Err(e) => Response::err(format!("scratch-write: {}", e)),
    }
}

fn cmd_scratch_read(server: &CiServer, args: &Value) -> Response {
    let to_path = match args.get("to_path").and_then(|v| v.as_str()) {
        Some(p) => match validate_host_path(p) {
            Ok(pb) => pb,
            Err(e) => return Response::err(format!("scratch-read: {}", e)),
        },
        None => return Response::err("scratch-read: missing 'to_path' arg"),
    };
    let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0);
    let length = args.get("length").and_then(|v| v.as_u64());

    let result = server.with_machine(|m| {
        let scratch = match m.scratch_path() {
            Some(p) => p.to_path_buf(),
            None => return Err("scratch volume not configured (set `scratch = true` on a SCSI device in iris.toml)".to_string()),
        };
        m.with_paused(|| -> Result<u64, String> {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = std::fs::File::open(&scratch)
                .map_err(|e| format!("open {}: {}", scratch.display(), e))?;
            let total = f.metadata().map(|m| m.len()).unwrap_or(0);
            let payload_total = total.saturating_sub(SCRATCH_PAYLOAD_OFFSET);
            let raw_offset = SCRATCH_PAYLOAD_OFFSET.checked_add(offset)
                .ok_or_else(|| "offset overflow".to_string())?;
            let len = match length {
                Some(n) => n.min(payload_total.saturating_sub(offset)),
                None => payload_total.saturating_sub(offset),
            };
            f.seek(SeekFrom::Start(raw_offset)).map_err(|e| format!("seek: {}", e))?;
            let mut buf = vec![0u8; len as usize];
            f.read_exact(&mut buf).map_err(|e| format!("read: {}", e))?;
            std::fs::write(&to_path, &buf)
                .map_err(|e| format!("write {}: {}", to_path.display(), e))?;
            Ok(buf.len() as u64)
        })
    });

    match result {
        Ok(n) => Response::data(serde_json::json!({
            "bytes_read": n,
            "offset": offset,
            "to_path": to_path.display().to_string(),
        })),
        Err(e) => Response::err(format!("scratch-read: {}", e)),
    }
}

fn cmd_scratch_clear(server: &CiServer) -> Response {
    let result = server.with_machine(|m| {
        let scratch = match m.scratch_path() {
            Some(p) => p.to_path_buf(),
            None => return Err("scratch volume not configured".to_string()),
        };
        m.with_paused(|| -> Result<u64, String> {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .open(&scratch)
                .map_err(|e| format!("open {}: {}", scratch.display(), e))?;
            let size = f.metadata().map(|m| m.len()).unwrap_or(0);
            // Zero only the payload area (after the VH). Zero in 1 MiB chunks
            // rather than allocating a buffer the full size of the volume.
            let chunk = vec![0u8; 1024 * 1024];
            f.seek(SeekFrom::Start(SCRATCH_PAYLOAD_OFFSET))
                .map_err(|e| format!("seek: {}", e))?;
            let mut remaining = size.saturating_sub(SCRATCH_PAYLOAD_OFFSET);
            while remaining > 0 {
                let n = remaining.min(chunk.len() as u64) as usize;
                f.write_all(&chunk[..n]).map_err(|e| format!("write: {}", e))?;
                remaining -= n as u64;
            }
            f.sync_all().map_err(|e| format!("fsync: {}", e))?;
            Ok(size.saturating_sub(SCRATCH_PAYLOAD_OFFSET))
        })
    });

    match result {
        Ok(n) => Response::data(serde_json::json!({ "bytes_cleared": n })),
        Err(e) => Response::err(format!("scratch-clear: {}", e)),
    }
}

fn cmd_scratch_info(server: &CiServer) -> Response {
    let path = server.with_machine(|m| m.scratch_path().map(|p| p.to_path_buf()));
    let Some(path) = path else {
        return Response::err("scratch-info: scratch volume not configured");
    };
    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    Response::data(serde_json::json!({
        "path": path.display().to_string(),
        "size_bytes": size,
        "payload_offset": SCRATCH_PAYLOAD_OFFSET,
        "payload_size_bytes": size.saturating_sub(SCRATCH_PAYLOAD_OFFSET),
    }))
}

// ----------------------------------------------------------------------------
// Snapshot determinism validator (Phase 3.3)
// ----------------------------------------------------------------------------

fn cmd_validate(server: &CiServer, args: &Value) -> Response {
    let name = match args.get("name").and_then(|v| v.as_str()) {
        Some(n) => n.to_string(),
        None => return Response::err("validate: missing 'name' arg"),
    };
    let n = args
        .get("n_instructions")
        .and_then(|v| v.as_u64())
        .unwrap_or(1_000_000);

    let report_result = server.with_machine(|m| {
        crate::validate::validate_snapshot_determinism(m, &name, n)
    });

    match report_result {
        Ok(report) => Response::data(serde_json::json!({
            "deterministic": report.deterministic,
            "instructions_run": report.instructions_run,
            "summary": report.summary(),
            "diffs": report.diffs.iter().map(|(f, a, b)| {
                serde_json::json!({"field": f, "a": a, "b": b})
            }).collect::<Vec<_>>(),
            "pc": format!("0x{:016x}", report.state_a.pc),
        })),
        Err(e) => Response::err(format!("validate: {}", e)),
    }
}

// ----------------------------------------------------------------------------
// Snapshot library: gc / diff / tree (Phase 3.2)
// ----------------------------------------------------------------------------

/// Walk every snapshot directory under `saves/`, parse each `chunks.bin`, and
/// collect the set of referenced chunk hashes. Used by `gc` to figure out
/// which chunks are still live.
fn collect_live_chunks() -> std::io::Result<std::collections::HashSet<crate::chunk_store::ChunkHash>> {
    use std::collections::HashSet;
    let mut live: HashSet<crate::chunk_store::ChunkHash> = HashSet::new();
    let root = std::path::Path::new("saves");
    if !root.is_dir() {
        return Ok(live);
    }
    let mut stack: Vec<std::path::PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir)?.flatten() {
            let p = e.path();
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if name == ".cas" { continue; }
            }
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.file_name().and_then(|n| n.to_str()) == Some("chunks.bin") {
                if let Ok(bytes) = std::fs::read(&p) {
                    if let Ok(m) = postcard::from_bytes::<crate::snapshot::ChunksManifest>(&bytes) {
                        for h in m.referenced_hashes() {
                            live.insert(*h);
                        }
                    }
                }
            }
        }
    }
    Ok(live)
}

fn cmd_gc() -> Response {
    let live = match collect_live_chunks() {
        Ok(l) => l,
        Err(e) => return Response::err(format!("gc: collect live: {}", e)),
    };
    let store = crate::chunk_store::ChunkStore::new("saves");
    let total_before = store.total_size().unwrap_or(0);
    match store.gc(&live) {
        Ok((removed, bytes)) => {
            // Drop now-empty shard dirs so saves/.cas stays tidy.
            if let Ok(entries) = std::fs::read_dir(store.root()) {
                for e in entries.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        let empty = std::fs::read_dir(&p).map(|mut it| it.next().is_none()).unwrap_or(false);
                        if empty {
                            let _ = std::fs::remove_dir(&p);
                        }
                    }
                }
            }
            Response::data(serde_json::json!({
                "live_chunks": live.len(),
                "removed_chunks": removed,
                "bytes_freed": bytes,
                "bytes_before": total_before,
                "bytes_after": total_before.saturating_sub(bytes),
            }))
        }
        Err(e) => Response::err(format!("gc: {}", e)),
    }
}

/// Diff two snapshots: per-device state diffs, RAM chunk-level deltas, COW
/// overlay sector deltas. Heavy lifting reuses BinValue's `PartialEq` (toml
/// equality) for device state and ChunksManifest hashes for RAM/framebuffer
/// regions.
fn cmd_diff(args: &Value) -> Response {
    let a = match args.get("a").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return Response::err("diff: missing 'a' arg"),
    };
    let b = match args.get("b").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return Response::err("diff: missing 'b' arg"),
    };
    if a.is_empty() || a.contains("..") || b.is_empty() || b.contains("..") {
        return Response::err("diff: invalid name");
    }

    let dir_a = std::path::PathBuf::from("saves").join(&a);
    let dir_b = std::path::PathBuf::from("saves").join(&b);
    if !dir_a.is_dir() {
        return Response::err(format!("diff: snapshot '{}' not found", a));
    }
    if !dir_b.is_dir() {
        return Response::err(format!("diff: snapshot '{}' not found", b));
    }

    let snap_a = crate::snapshot::Snapshot::new(&dir_a);
    let snap_b = crate::snapshot::Snapshot::new(&dir_b);

    let sv_a = snap_a.read_manifest().ok().flatten().map(|m| m.schema_version).unwrap_or(0);
    let sv_b = snap_b.read_manifest().ok().flatten().map(|m| m.schema_version).unwrap_or(0);

    // Per-device state. The eight devices we track here are the ones every
    // configuration writes; rex3 is optional so it's handled separately.
    let device_bases = [
        "cpu", "mc", "ioc", "scc", "pit", "ps2", "rtc",
        "eeprom", "scsi", "seeq", "hpc3",
    ];
    let mut devices_changed: Vec<&'static str> = Vec::new();
    let mut devices_unchanged: Vec<&'static str> = Vec::new();
    for &base in &device_bases {
        let va = snap_a.read_state(base, sv_a).ok();
        let vb = snap_b.read_state(base, sv_b).ok();
        match (va, vb) {
            (Some(va), Some(vb)) => {
                if va == vb { devices_unchanged.push(base); }
                else        { devices_changed.push(base); }
            }
            _ => devices_changed.push(base),
        }
    }
    // REX3 separately because it's optional.
    let rex_a = snap_a.read_state("rex3", sv_a).ok();
    let rex_b = snap_b.read_state("rex3", sv_b).ok();
    let rex3_changed = match (rex_a, rex_b) {
        (Some(va), Some(vb)) => Some(va != vb),
        (None, None)         => None,
        _                    => Some(true),
    };

    // RAM bank deltas via chunks.bin (v3+ only).
    let mut bank_changed_chunks = [0u32; 4];
    let mut bank_total_chunks   = [0u32; 4];
    let mut framebuffer_changed_chunks: Option<(u32, u32)> = None;
    if sv_a >= 3 && sv_b >= 3 {
        if let (Ok(ma), Ok(mb)) = (snap_a.read_chunks_manifest(), snap_b.read_chunks_manifest()) {
            for i in 0..4 {
                let ah = &ma.bank_chunks[i];
                let bh = &mb.bank_chunks[i];
                let n = ah.len().max(bh.len());
                bank_total_chunks[i] = n as u32;
                let mut changed = 0u32;
                for k in 0..n {
                    let av = ah.get(k);
                    let bv = bh.get(k);
                    if av != bv { changed += 1; }
                }
                bank_changed_chunks[i] = changed;
            }
            if let (Some((rgb_a, aux_a)), Some((rgb_b, aux_b))) =
                (&ma.framebuffer_chunks, &mb.framebuffer_chunks)
            {
                let n = rgb_a.len().max(rgb_b.len()) + aux_a.len().max(aux_b.len());
                let mut changed = 0u32;
                for k in 0..rgb_a.len().max(rgb_b.len()) {
                    if rgb_a.get(k) != rgb_b.get(k) { changed += 1; }
                }
                for k in 0..aux_a.len().max(aux_b.len()) {
                    if aux_a.get(k) != aux_b.get(k) { changed += 1; }
                }
                framebuffer_changed_chunks = Some((changed, n as u32));
            }
        }
    }

    // COW overlay sector deltas from cow.toml.
    let cow_a = snap_a.read_toml("cow.toml").ok();
    let cow_b = snap_b.read_toml("cow.toml").ok();
    let mut cow_diff_per_id: Vec<(usize, u64, u64, u64)> = Vec::new(); // (id, only_a, only_b, both)
    if let (Some(ca), Some(cb)) = (cow_a, cow_b) {
        let mut ids: std::collections::BTreeSet<usize> = Default::default();
        if let Some(t) = ca.as_table() {
            for k in t.keys() {
                if let Some(s) = k.strip_prefix("scsi") {
                    if let Ok(n) = s.parse::<usize>() { ids.insert(n); }
                }
            }
        }
        if let Some(t) = cb.as_table() {
            for k in t.keys() {
                if let Some(s) = k.strip_prefix("scsi") {
                    if let Ok(n) = s.parse::<usize>() { ids.insert(n); }
                }
            }
        }
        for id in ids {
            let key = format!("scsi{}", id);
            let set_a: std::collections::HashSet<u64> = ca.get(&key)
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_integer().map(|i| i as u64)).collect())
                .unwrap_or_default();
            let set_b: std::collections::HashSet<u64> = cb.get(&key)
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_integer().map(|i| i as u64)).collect())
                .unwrap_or_default();
            let only_a = set_a.difference(&set_b).count() as u64;
            let only_b = set_b.difference(&set_a).count() as u64;
            let both = set_a.intersection(&set_b).count() as u64;
            cow_diff_per_id.push((id, only_a, only_b, both));
        }
    }

    Response::data(serde_json::json!({
        "a": a,
        "b": b,
        "schema_a": sv_a,
        "schema_b": sv_b,
        "devices_changed": devices_changed,
        "devices_unchanged": devices_unchanged,
        "rex3_changed": rex3_changed,
        "bank_changed_chunks": bank_changed_chunks,
        "bank_total_chunks":   bank_total_chunks,
        "framebuffer_changed_chunks": framebuffer_changed_chunks,
        "cow_diff": cow_diff_per_id.into_iter().map(|(id, only_a, only_b, both)| {
            serde_json::json!({"scsi_id": id, "only_a": only_a, "only_b": only_b, "both": both})
        }).collect::<Vec<_>>(),
    }))
}

/// Walk every snapshot under `saves/`, build a parent → children map, render
/// indented tree text. Snapshots without a parent (or with a parent that
/// doesn't exist locally) hang off a synthetic `(none)` root.
fn cmd_tree() -> Response {
    use std::collections::BTreeMap;
    let root = std::path::Path::new("saves");
    if !root.is_dir() {
        return Response::data(serde_json::json!({"tree": "(no saves directory)"}));
    }

    // (name, parent) for each snapshot.
    let mut entries: Vec<(String, Option<String>)> = Vec::new();
    let mut stack: Vec<std::path::PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(it) = std::fs::read_dir(&dir) else { continue };
        let mut subdirs: Vec<std::path::PathBuf> = Vec::new();
        let mut found_manifest = false;
        let mut found_legacy_cpu = false;
        for e in it.flatten() {
            let p = e.path();
            if let Some(n) = p.file_name().and_then(|n| n.to_str()) {
                if n == ".cas" { continue; }
            }
            if p.is_dir() {
                subdirs.push(p);
            } else if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if name == "snapshot.toml" { found_manifest = true; }
                if name == "cpu.toml" { found_legacy_cpu = true; }
            }
        }
        if found_manifest || found_legacy_cpu {
            if let Ok(rel) = dir.strip_prefix(root) {
                let display_name = rel.to_string_lossy().replace('\\', "/");
                if !display_name.is_empty() {
                    let snap = crate::snapshot::Snapshot::new(&dir);
                    let parent = snap.read_manifest().ok().flatten().and_then(|m| m.parent);
                    entries.push((display_name, parent));
                }
            }
        }
        for s in subdirs { stack.push(s); }
    }

    // Build parent → children map (None parent → top-level).
    let mut by_parent: BTreeMap<Option<String>, Vec<String>> = BTreeMap::new();
    let names: std::collections::HashSet<String> = entries.iter().map(|(n, _)| n.clone()).collect();
    for (name, parent) in &entries {
        let key = match parent {
            Some(p) if names.contains(p) => Some(p.clone()),
            _ => None,
        };
        by_parent.entry(key).or_default().push(name.clone());
    }
    for v in by_parent.values_mut() { v.sort(); }

    fn render(out: &mut String, by_parent: &BTreeMap<Option<String>, Vec<String>>, parent: Option<&str>, depth: usize) {
        let key = parent.map(String::from);
        if let Some(children) = by_parent.get(&key) {
            for child in children {
                for _ in 0..depth { out.push_str("  "); }
                out.push_str("- ");
                out.push_str(child);
                out.push('\n');
                render(out, by_parent, Some(child), depth + 1);
            }
        }
    }
    let mut text = String::new();
    render(&mut text, &by_parent, None, 0);
    if text.is_empty() { text.push_str("(no snapshots)\n"); }

    Response::data(serde_json::json!({
        "snapshots": entries.iter().map(|(n, p)| {
            serde_json::json!({"name": n, "parent": p})
        }).collect::<Vec<_>>(),
        "tree": text.trim_end_matches('\n').to_string(),
    }))
}

// ----------------------------------------------------------------------------
// HTTP snapshot registry (Phase 3.4)
// ----------------------------------------------------------------------------

fn cmd_pull(args: &Value) -> Response {
    let url = match args.get("url").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return Response::err("pull: missing 'url' arg"),
    };
    let name = match args.get("name").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return Response::err("pull: missing 'name' arg"),
    };
    let saves = std::path::PathBuf::from("saves");
    if !saves.is_dir() {
        if let Err(e) = std::fs::create_dir_all(&saves) {
            return Response::err(format!("pull: create saves/: {}", e));
        }
    }
    match crate::registry::pull(&url, &name, &saves) {
        Ok(report) => Response::data(serde_json::json!({
            "name": name,
            "url": url,
            "chunks_fetched": report.chunks_fetched,
            "chunks_skipped": report.chunks_skipped,
            "files_transferred": report.files_transferred,
            "bytes_transferred": report.bytes_transferred,
        })),
        Err(e) => Response::err(format!("pull: {}", e)),
    }
}

fn cmd_push(args: &Value) -> Response {
    let url = match args.get("url").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return Response::err("push: missing 'url' arg"),
    };
    let name = match args.get("name").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return Response::err("push: missing 'name' arg"),
    };
    match crate::registry::push(&url, &name, std::path::Path::new("saves")) {
        Ok(report) => Response::data(serde_json::json!({
            "name": name,
            "url": url,
            "chunks_uploaded": report.chunks_fetched,
            "chunks_skipped": report.chunks_skipped,
            "files_transferred": report.files_transferred,
            "bytes_transferred": report.bytes_transferred,
        })),
        Err(e) => Response::err(format!("push: {}", e)),
    }
}
