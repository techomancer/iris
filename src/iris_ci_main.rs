//! `iris-ci` — ergonomic wrapper around the iris CI control socket.
//!
//! Replaces the raw `printf '...' | nc -U /tmp/iris.sock` pattern that's
//! awkward to type, brittle to quote, and tedious to compose. Every
//! socket-level operation gets a typed clap subcommand, plus macros for
//! the recurring multi-step rituals (boot, login, run, put/get).
//!
//! The headline ergonomic wins:
//! - `iris-ci boot` does the full PROM-menu-to-login dance in one command.
//! - `iris-ci run "ls /tmp"` sends the command, waits for the prompt, and
//!   returns just the captured stdout + exit status.
//! - `iris-ci put localfile.tar` copies a host file into the guest with
//!   the right `dd bs=512 count=N` recipe baked in — no foot-gun.
//! - `iris-ci script tests/scenario.iris` runs a sequence of commands
//!   and prints per-step status + duration.
//!
//! Returns 0 on success, 1 on socket error, 2 on iris error response,
//! 3 on local error (file not found etc).

use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::net::UnixStream;

#[cfg(unix)]
const DEFAULT_SOCKET: &str = "/tmp/iris.sock";
#[cfg(windows)]
const DEFAULT_SOCKET: &str = "127.0.0.1:19851";
#[cfg(not(any(unix, windows)))]
const DEFAULT_SOCKET: &str = "/tmp/iris.sock";
const PROMPT_RE: &str = "IRIS"; // Match "IRIS N# " — N is a counter that increments
const RC_MARKER: &str = "IRIS-CI-RC=";

#[derive(Parser, Debug)]
#[command(
    name = "iris-ci",
    about = "Drive the iris CI control socket without raw nc + JSON.",
    version
)]
struct Cli {
    /// Path to the iris CI Unix socket. Override with $IRIS_SOCKET.
    #[arg(long, global = true)]
    socket: Option<PathBuf>,

    /// Print raw JSON responses instead of pretty output.
    #[arg(long, global = true)]
    json: bool,

    /// Be silent on success (for use in scripts).
    #[arg(long, short = 'q', global = true)]
    quiet: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Liveness check — returns "ok" if the socket is reachable.
    Ping,
    /// Start the CPU thread (no-op if already running).
    Start,
    /// Cleanly shut down iris. `--sync-chd [ID|all]` folds pending CHD diffs
    /// back into their bases first (default: all disks).
    Quit {
        #[arg(long, num_args = 0..=1, default_missing_value = "all", value_name = "ID|all")]
        sync_chd: Option<String>,
    },

    /// Save the current machine state to saves/<name>/.
    Save {
        name: String,
        #[arg(long)]
        description: Option<String>,
    },
    /// Disk-backed full restore (~145 ms cold, sets up rollback checkpoint).
    Restore { name: String },
    /// In-memory rewind to the last `restore` checkpoint (~40 ms).
    Rollback,
    /// List all saved snapshots.
    List,
    /// Show metadata (manifest, schema_version, size) for one snapshot.
    Info { name: String },
    /// Delete a snapshot (does NOT free CAS chunks; run `gc` after).
    Delete { name: String },
    /// Render the snapshot parent-chain tree.
    Tree,
    /// Compare two snapshots: device + RAM-chunk + COW-sector deltas.
    Diff { a: String, b: String },
    /// Sweep CAS chunks not referenced by any kept snapshot.
    Gc,
    /// Run the snapshot determinism validator.
    Validate {
        name: String,
        /// Number of instructions to step in each pass (default 1_000_000).
        #[arg(short = 'n', long, default_value_t = 1_000_000)]
        n: u64,
    },
    /// Save the REX3 framebuffer to a PNG.
    Screenshot { path: PathBuf },

    /// Send keystrokes to the IRIX serial console.
    SerialSend {
        text: String,
        /// Don't append \r to the text.
        #[arg(long)]
        no_cr: bool,
    },
    /// Drain the serial output buffer and print it.
    SerialRead,
    /// Wait until `pattern` appears in serial output (or timeout).
    SerialWait {
        pattern: String,
        #[arg(long, default_value_t = 30)]
        timeout: u64,
    },

    /// Boot from PROM menu through to the IRIS console login prompt.
    Boot {
        /// Total timeout in seconds for boot to reach the login prompt.
        #[arg(long, default_value_t = 240)]
        timeout: u64,
    },
    /// Send root login + dismiss the vt100 prompt + wait for the shell.
    Login {
        #[arg(default_value = "root")]
        user: String,
        /// Optional password (most IRIX root accounts have none).
        #[arg(long)]
        password: Option<String>,
    },
    /// Send a shell command, wait for the prompt, return stdout + exit code.
    Run {
        command: String,
        /// Guest shell. csh uses $status; sh uses $?.
        #[arg(long, default_value = "csh")]
        shell: String,
        #[arg(long, default_value_t = 60)]
        timeout: u64,
    },
    /// Drain output and wait for the next shell prompt.
    WaitPrompt {
        #[arg(long, default_value_t = 30)]
        timeout: u64,
    },

    /// Copy a host file into the guest. Handles bs=512 + count automatically.
    Put {
        host_path: PathBuf,
        /// Where to put it inside IRIX. Defaults to /tmp/<basename>.
        #[arg(long)]
        to: Option<String>,
        #[arg(long, default_value_t = 120)]
        timeout: u64,
    },
    /// Pull a guest file out to the host. Handles tar + scratch round-trip.
    Get {
        guest_path: String,
        /// Where to write on the host. Defaults to ./<basename>.
        #[arg(long)]
        to: Option<PathBuf>,
        #[arg(long, default_value_t = 120)]
        timeout: u64,
    },

    /// Raw scratch volume operations (bypass guest interaction).
    #[command(subcommand)]
    Scratch(ScratchCmd),

    /// Pull a snapshot from a remote registry (e.g. `http://localhost:8765`).
    Pull { url: String, name: String },
    /// Push a snapshot to a remote registry.
    Push { url: String, name: String },

    /// Run a sequence of iris-ci commands from a file (one per line, # comments).
    Script { path: PathBuf },

    /// Persist the emulated DS1386 NVRAM/RTC state to a file (default nvram.bin).
    RtcSave {
        #[arg(long)]
        path: Option<String>,
    },

    /// Cycle the CD changer on a SCSI ID to the next disc.
    CdromEject { id: u64 },

    /// Load an arbitrary CD image into a SCSI ID and make it the active disc.
    CdromLoad { id: u64, path: String },

    /// Fold pending CHD diffs back into their bases (guest must be stopped).
    ChdSync {
        /// SCSI id to consolidate, or "all" (default).
        #[arg(default_value = "all", value_name = "ID|all")]
        id: String,
    },
}

#[derive(Subcommand, Debug)]
enum ScratchCmd {
    /// Copy raw bytes from a host file into the scratch payload area.
    Write {
        path: PathBuf,
        #[arg(long, default_value_t = 0)]
        offset: u64,
    },
    /// Copy raw bytes from the scratch payload area into a host file.
    Read {
        path: PathBuf,
        #[arg(long, default_value_t = 0)]
        offset: u64,
        #[arg(long)]
        length: Option<u64>,
    },
    /// Zero the scratch payload area (preserves the SGI VH at sector 0).
    Clear,
    /// Show scratch volume size + payload offset.
    Info,
}

// ---- main / dispatch ---------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    let socket = cli
        .socket
        .clone()
        .or_else(|| std::env::var_os("IRIS_SOCKET").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SOCKET));
    let opts = Opts {
        socket,
        json: cli.json,
        quiet: cli.quiet,
    };

    let exit = match dispatch(&opts, cli.cmd) {
        Ok(()) => 0,
        Err(Error::Local(e)) => {
            eprintln!("iris-ci: {}", e);
            3
        }
        Err(Error::Connection(e)) => {
            // A read timeout is not a failure to connect, and saying so sends
            // the reader looking at the wrong end. The socket was fine; the
            // command outlived the read deadline.
            match e.kind() {
                std::io::ErrorKind::WouldBlock
                | std::io::ErrorKind::TimedOut => eprintln!(
                    "iris-ci: timed out waiting for {} to answer: {}",
                    opts.socket.display(), e),
                _ => eprintln!(
                    "iris-ci: connect {}: {}", opts.socket.display(), e),
            }
            1
        }
        Err(Error::Iris(e)) => {
            eprintln!("iris-ci: iris error: {}", e);
            2
        }
    };
    std::process::exit(exit);
}

fn dispatch(opts: &Opts, cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Ping        => simple(opts, "ping", json!({}), "ok"),
        Cmd::Start       => simple(opts, "start", json!({}), "started"),
        Cmd::Quit { sync_chd } => match sync_chd {
            // A fold recompresses the whole base, so it can far outrun the 300s default.
            Some(sel) => {
                let data = send_until(opts, "quit", json!({"sync_chd": scsi_sel(&sel)?}), FOLD_TIMEOUT)?;
                print_response(opts, "quit", &data);
                Ok(())
            }
            None => simple(opts, "quit", json!({}), "quit"),
        },
        Cmd::Save { name, description } => {
            let mut args = json!({"name": name});
            if let Some(d) = description { args["description"] = Value::String(d); }
            simple(opts, "save", args, &format!("saved: {}", "")) // detailed status logged below
        }
        Cmd::Restore  { name } => simple(opts, "restore",  json!({"name": name}), "restored"),
        Cmd::Rollback        => simple(opts, "rollback", json!({}),                "rolled back"),
        Cmd::List            => cmd_list(opts),
        Cmd::Info     { name }     => cmd_info(opts, &name),
        Cmd::Delete   { name }     => simple(opts, "delete",  json!({"name": name}), "deleted"),
        Cmd::Tree                  => cmd_tree(opts),
        Cmd::Diff     { a, b }     => cmd_diff(opts, &a, &b),
        Cmd::Gc                    => cmd_gc(opts),
        Cmd::Validate { name, n }  => cmd_validate(opts, &name, n),
        Cmd::Screenshot { path }   => simple(opts, "screenshot", json!({"path": path.display().to_string()}), "screenshot"),

        Cmd::SerialSend { text, no_cr } => {
            let data = if no_cr { text } else { format!("{}\r", text) };
            simple(opts, "serial-send", json!({"data": data}), "sent")
        }
        Cmd::SerialRead => cmd_serial_read(opts),
        Cmd::SerialWait { pattern, timeout } => cmd_serial_wait(opts, &pattern, timeout * 1000),

        Cmd::Boot   { timeout } => cmd_boot(opts, timeout),
        Cmd::Login  { user, password } => cmd_login(opts, &user, password.as_deref()),
        Cmd::Run    { command, shell, timeout } => cmd_run(opts, &command, &shell, timeout * 1000),
        Cmd::WaitPrompt { timeout } => cmd_wait_prompt(opts, timeout * 1000),

        Cmd::Put { host_path, to, timeout } => cmd_put(opts, &host_path, to.as_deref(), timeout * 1000),
        Cmd::Get { guest_path, to, timeout } => cmd_get(opts, &guest_path, to.as_deref(), timeout * 1000),

        Cmd::Scratch(s) => cmd_scratch(opts, s),

        Cmd::Pull { url, name } => cmd_pull(opts, &url, &name),
        Cmd::Push { url, name } => cmd_push(opts, &url, &name),

        Cmd::Script { path } => cmd_script(opts, &path),

        Cmd::RtcSave { path } => {
            let args = match path {
                Some(p) => json!({"path": p}),
                None => json!({}),
            };
            simple(opts, "rtc-save", args, "nvram saved")
        }
        Cmd::CdromEject { id } => simple(opts, "cdrom-eject", json!({"id": id}), "ejected"),
        Cmd::CdromLoad { id, path } => simple(opts, "cdrom-load", json!({"id": id, "path": path}), "loaded"),
        Cmd::ChdSync { id } => {
            let data = send_until(opts, "chd-sync", json!({"id": scsi_sel(&id)?}), FOLD_TIMEOUT)?;
            print_response(opts, "synced", &data);
            Ok(())
        }
    }
}

// ---- error type --------------------------------------------------------------

#[derive(Debug)]
enum Error {
    Local(String),
    Connection(std::io::Error),
    Iris(String),
}
type Result<T> = std::result::Result<T, Error>;
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self { Error::Connection(e) }
}

struct Opts {
    socket: PathBuf,
    json: bool,
    quiet: bool,
}

// ---- socket client -----------------------------------------------------------

/// Read timeout for commands that answer promptly.
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(300);
/// CHD folds recompress a whole base image, which can take far longer than the default.
const FOLD_TIMEOUT: Duration = Duration::from_secs(3 * 3600);

/// `"all"` stays a string; anything else must be a SCSI id 0-7.
fn scsi_sel(s: &str) -> Result<Value> {
    if s.eq_ignore_ascii_case("all") { return Ok(json!("all")); }
    match s.parse::<u64>() {
        Ok(n) if n < 8 => Ok(json!(n)),
        _ => Err(Error::Iris(format!("bad scsi id '{s}' (expected 0-7 or \"all\")"))),
    }
}

/// Grace on top of a guest deadline, so the server's own timeout fires first
/// and reports which pattern was missed rather than the client giving up.
const READ_TIMEOUT_GRACE: Duration = Duration::from_secs(30);

fn read_timeout_for(timeout_ms: u64) -> Duration {
    Duration::from_millis(timeout_ms) + READ_TIMEOUT_GRACE
}

/// Send one JSON command, return the parsed response data on `ok:true`,
/// or an Error on connection failure or `ok:false`.
///
/// Protocol detail: the server (`src/ci.rs::handle_client`) keeps the
/// connection open and reads requests in a loop, expecting the client to
/// close. We send our single request, then read exactly one newline-
/// terminated response line, then drop the stream — the server's reader
/// loop sees EOF and exits cleanly.
fn send(opts: &Opts, cmd: &str, args: Value) -> Result<Value> {
    send_until(opts, cmd, args, DEFAULT_READ_TIMEOUT)
}

/// Same, with the read timeout the caller needs.
///
/// `wait-serial` blocks on the server for as long as the caller asked, so a
/// fixed client-side read timeout silently caps every command that uses it.
/// At 300s that made any guest command over five minutes fail with EAGAIN
/// while the guest carried on working, and `--timeout` appeared to do nothing.
fn send_until(opts: &Opts, cmd: &str, args: Value, read_timeout: Duration) -> Result<Value> {
    let addr = opts.socket.to_string_lossy();
    if iris::config::ci_socket_is_tcp(&addr) {
        send_tcp(&iris::config::ci_socket_tcp_addr(&addr), cmd, args, read_timeout)
    } else {
        #[cfg(unix)]
        {
            send_unix(&opts.socket, cmd, args, read_timeout)
        }
        #[cfg(not(unix))]
        {
            Err(Error::Iris(format!(
                "CI socket {} is not TCP; on this platform use host:port (default {})",
                addr, DEFAULT_SOCKET
            )))
        }
    }
}

fn send_tcp(addr: &str, cmd: &str, args: Value, read_timeout: Duration) -> Result<Value> {
    let s = TcpStream::connect(addr)?;
    s.set_read_timeout(Some(read_timeout)).ok();
    let mut writer = s.try_clone()?;
    let mut reader = BufReader::new(s);
    exchange(&mut reader, &mut writer, cmd, args)
}

#[cfg(unix)]
fn send_unix(socket: &PathBuf, cmd: &str, args: Value, read_timeout: Duration) -> Result<Value> {
    let s = UnixStream::connect(socket)?;
    s.set_read_timeout(Some(read_timeout)).ok();
    let mut writer = s.try_clone()?;
    let mut reader = BufReader::new(s);
    exchange(&mut reader, &mut writer, cmd, args)
}

fn exchange(reader: &mut impl BufRead, writer: &mut impl Write, cmd: &str, args: Value) -> Result<Value> {
    let req = json!({"cmd": cmd, "args": args});
    let line = format!("{}\n", serde_json::to_string(&req).expect("json"));
    writer.write_all(line.as_bytes())?;
    writer.flush()?;
    let mut buf = String::new();
    reader.read_line(&mut buf)?;
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        return Err(Error::Iris("empty response".into()));
    }
    let resp: Value = serde_json::from_str(trimmed).map_err(|e| {
        Error::Iris(format!("bad response: {}: {}", e, trimmed))
    })?;
    if resp.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        let msg = resp
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        return Err(Error::Iris(format!("{}: {}", cmd, msg)));
    }
    Ok(resp.get("data").cloned().unwrap_or(Value::Null))
}

// ---- 1:1 commands with pretty output -----------------------------------------

fn simple(opts: &Opts, cmd: &str, args: Value, ok_msg: &str) -> Result<()> {
    let data = send(opts, cmd, args)?;
    print_response(opts, ok_msg, &data);
    Ok(())
}

fn print_response(opts: &Opts, ok_msg: &str, data: &Value) {
    if opts.json {
        println!("{}", serde_json::to_string_pretty(data).unwrap_or_else(|_| data.to_string()));
        return;
    }
    if opts.quiet { return; }
    if data.is_null() || (data.is_object() && data.as_object().map(|m| m.is_empty()).unwrap_or(true)) {
        println!("{}", ok_msg);
    } else {
        println!("{}: {}", ok_msg, data);
    }
}

fn cmd_list(opts: &Opts) -> Result<()> {
    let data = send(opts, "list", json!({}))?;
    if opts.json { println!("{}", serde_json::to_string_pretty(&data).unwrap_or_default()); return Ok(()); }
    if let Some(arr) = data.get("snapshots").and_then(|v| v.as_array()) {
        for s in arr {
            if let Some(name) = s.as_str() {
                println!("{}", name);
            }
        }
    }
    Ok(())
}

fn cmd_info(opts: &Opts, name: &str) -> Result<()> {
    let data = send(opts, "info", json!({"name": name}))?;
    if opts.json { println!("{}", serde_json::to_string_pretty(&data).unwrap_or_default()); return Ok(()); }
    let f = |k: &str| data.get(k).cloned().unwrap_or(Value::Null);
    println!("name             {}", f("name"));
    println!("schema_version   {}", f("schema_version"));
    println!("host_arch        {}", f("host_arch"));
    println!("created_at_unix  {}", f("created_at_unix"));
    println!("bytes_on_disk    {}", f("bytes_on_disk"));
    if let Some(p) = data.get("parent") { if !p.is_null() { println!("parent           {}", p); } }
    if let Some(d) = data.get("description") { if !d.is_null() { println!("description      {}", d); } }
    if let Some(b) = data.get("installed_bundles") { println!("installed        {}", b); }
    Ok(())
}

fn cmd_tree(opts: &Opts) -> Result<()> {
    let data = send(opts, "tree", json!({}))?;
    if opts.json { println!("{}", serde_json::to_string_pretty(&data).unwrap_or_default()); return Ok(()); }
    if let Some(t) = data.get("tree").and_then(|v| v.as_str()) {
        println!("{}", t);
    }
    Ok(())
}

fn cmd_diff(opts: &Opts, a: &str, b: &str) -> Result<()> {
    let data = send(opts, "diff", json!({"a": a, "b": b}))?;
    if opts.json { println!("{}", serde_json::to_string_pretty(&data).unwrap_or_default()); return Ok(()); }
    println!("diff {} → {}", a, b);
    if let Some(arr) = data.get("devices_changed").and_then(|v| v.as_array()) {
        let names: Vec<String> = arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
        if !names.is_empty() {
            println!("  devices changed:   {}", names.join(", "));
        }
    }
    if let Some(arr) = data.get("devices_unchanged").and_then(|v| v.as_array()) {
        let names: Vec<String> = arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
        if !names.is_empty() {
            println!("  devices unchanged: {}", names.join(", "));
        }
    }
    if let (Some(c), Some(t)) = (
        data.get("bank_changed_chunks").and_then(|v| v.as_array()),
        data.get("bank_total_chunks").and_then(|v| v.as_array()),
    ) {
        for i in 0..c.len().min(t.len()) {
            let c = c[i].as_u64().unwrap_or(0);
            let t = t[i].as_u64().unwrap_or(0);
            if t > 0 {
                println!("  bank{}: {}/{} chunks changed", i, c, t);
            }
        }
    }
    if let Some(arr) = data.get("cow_diff").and_then(|v| v.as_array()) {
        for entry in arr {
            let id = entry.get("scsi_id").and_then(|v| v.as_u64()).unwrap_or(0);
            let only_a = entry.get("only_a").and_then(|v| v.as_u64()).unwrap_or(0);
            let only_b = entry.get("only_b").and_then(|v| v.as_u64()).unwrap_or(0);
            let both = entry.get("both").and_then(|v| v.as_u64()).unwrap_or(0);
            println!("  scsi{}: only-a={} only-b={} both={}", id, only_a, only_b, both);
        }
    }
    Ok(())
}

fn cmd_gc(opts: &Opts) -> Result<()> {
    let data = send(opts, "gc", json!({}))?;
    if opts.json { println!("{}", serde_json::to_string_pretty(&data).unwrap_or_default()); return Ok(()); }
    let removed = data.get("removed_chunks").and_then(|v| v.as_u64()).unwrap_or(0);
    let bytes = data.get("bytes_freed").and_then(|v| v.as_u64()).unwrap_or(0);
    let live = data.get("live_chunks").and_then(|v| v.as_u64()).unwrap_or(0);
    println!("gc: {} chunks removed, {} bytes freed, {} live", removed, bytes, live);
    Ok(())
}

fn cmd_validate(opts: &Opts, name: &str, n: u64) -> Result<()> {
    let data = send(opts, "validate", json!({"name": name, "n_instructions": n}))?;
    if opts.json { println!("{}", serde_json::to_string_pretty(&data).unwrap_or_default()); return Ok(()); }
    if let Some(s) = data.get("summary").and_then(|v| v.as_str()) {
        println!("{}", s);
    }
    if data.get("deterministic").and_then(|v| v.as_bool()) != Some(true) {
        // Validation surfaced a real divergence — exit with iris-error code so
        // scripts can branch on it.
        return Err(Error::Iris("non-deterministic".into()));
    }
    Ok(())
}

// ---- serial helpers ----------------------------------------------------------

fn cmd_serial_read(opts: &Opts) -> Result<()> {
    let data = send(opts, "serial-read", json!({}))?;
    if let Some(s) = data.as_str() {
        if !s.is_empty() {
            // Re-render \r\n cleanly — IRIX uses CRLF on the wire.
            print!("{}", s.replace("\r\n", "\n").replace('\r', "\n"));
        }
    }
    Ok(())
}

fn cmd_serial_wait(opts: &Opts, pattern: &str, timeout_ms: u64) -> Result<()> {
    let data = send_until(
        opts,
        "wait-serial",
        json!({"pattern": pattern, "timeout_ms": timeout_ms}),
        read_timeout_for(timeout_ms),
    )?;
    if let Some(s) = data.as_str() {
        if !opts.quiet {
            print!("{}", s.replace("\r\n", "\n").replace('\r', "\n"));
        }
    }
    Ok(())
}

// ---- boot/login/run macros --------------------------------------------------

fn cmd_boot(opts: &Opts, timeout_s: u64) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(timeout_s);
    if !opts.quiet { eprintln!("boot: starting CPU"); }
    send(opts, "start", json!({}))?;
    if !opts.quiet { eprintln!("boot: waiting for PROM menu"); }
    wait_with_deadline(opts, "Option?", deadline)?;
    if !opts.quiet { eprintln!("boot: PROM reached, selecting 1) Start System"); }
    send(opts, "serial-send", json!({"data": "1\r"}))?;
    if !opts.quiet { eprintln!("boot: waiting for kernel boot to login prompt"); }
    wait_with_deadline(opts, "IRIS console login", deadline)?;
    if !opts.quiet { eprintln!("boot: ready at login"); }
    Ok(())
}

fn cmd_login(opts: &Opts, user: &str, password: Option<&str>) -> Result<()> {
    send(opts, "serial-send", json!({"data": format!("{}\r", user)}))?;
    // IRIX presents `TERM = (vt100)` after the username; pressing enter accepts.
    std::thread::sleep(Duration::from_millis(2000));
    if let Some(p) = password {
        send_until(opts, "wait-serial",
        json!({"pattern": "Password:", "timeout_ms": 5000}),
        read_timeout_for(5000))?;
        send(opts, "serial-send", json!({"data": format!("{}\r", p)}))?;
    }
    send(opts, "serial-send", json!({"data": "\r"}))?;
    let deadline = Instant::now() + Duration::from_secs(15);
    wait_with_deadline(opts, "#", deadline)?;
    if !opts.quiet { eprintln!("login: shell ready"); }
    Ok(())
}

fn cmd_wait_prompt(opts: &Opts, timeout_ms: u64) -> Result<()> {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    wait_with_deadline(opts, PROMPT_RE, deadline)?;
    Ok(())
}

/// Run a command and return captured stdout + exit code. Internal helper
/// shared by `cmd_run` (which prints stdout) and `cmd_get` (which parses it).
fn run_capture(opts: &Opts, command: &str, shell: &str, timeout_ms: u64) -> Result<(String, i32)> {
    let rc_var = match shell {
        "csh" | "tcsh" => "$status",
        "sh" | "bash" | "ksh" => "$?",
        other => return Err(Error::Local(format!("unknown shell {}", other))),
    };
    // Drain anything stale before sending.
    let _ = send(opts, "serial-read", json!({}))?;
    let line = format!("{}; echo {}{}\r", command, RC_MARKER, rc_var);
    send(opts, "serial-send", json!({"data": line}))?;
    // Single wait: pattern `\nIRIS-CI-RC=` only matches at the start of the
    // output line (the typed-input echo line has `IRIS-CI-RC=$status` inline,
    // so it has no preceding newline immediately before the marker).
    let pat = format!("\n{}", RC_MARKER);
    let captured = send_until(
        opts,
        "wait-serial",
        json!({"pattern": pat, "timeout_ms": timeout_ms}),
        read_timeout_for(timeout_ms),
    )?;
    let raw = captured.as_str().unwrap_or("").to_string();
    // Drain trailing chars (rc digits + next prompt).
    std::thread::sleep(Duration::from_millis(150));
    let trailing = send(opts, "serial-read", json!({}))?;
    let trailing_s = trailing.as_str().unwrap_or("");
    let rc = parse_rc(&format!("{}{}", RC_MARKER, trailing_s)).unwrap_or(-1);
    let stdout = extract_run_stdout(&raw);
    Ok((stdout, rc))
}

/// Detect the guest's interactive login shell so `get`/`put` use redirect +
/// rc-marker syntax that actually parses there. Historically IRIX root ran csh
/// (`>&`, `$status`); some installs (e.g. the klindert 6.5 disk) switch root to
/// `/bin/sh` (`2>&1`, `$?`), where the csh forms fail ("bad file unit number")
/// and the rc marker comes back empty. `$0` expands to the shell in both, so
/// probe it with a self-chosen sentinel (no rc marker, so it can't itself depend
/// on the shell). Defaults to csh on any ambiguity (the historical assumption).
fn detect_guest_shell(opts: &Opts) -> &'static str {
    const SENTINEL: &str = "ZZSHELLZZ=";
    let _ = send(opts, "serial-read", json!({}));
    if send(opts, "serial-send", json!({"data": format!("echo {}$0\r", SENTINEL)})).is_err() {
        return "csh";
    }
    // Wait for the value line (the typed echo also contains the sentinel; read a
    // bit more so the rsplit lands on the command's actual output).
    let _ = send_until(opts, "wait-serial",
        json!({"pattern": SENTINEL, "timeout_ms": 8000}),
        read_timeout_for(8000));
    std::thread::sleep(Duration::from_millis(250));
    let more = send(opts, "serial-read", json!({})).ok();
    let buf = more.as_ref().and_then(|v| v.as_str()).unwrap_or("");
    let val = buf.rsplit(SENTINEL).next().unwrap_or("");
    // `$0` is e.g. "-sh", "/bin/sh", "/bin/csh", "-csh", "tcsh".
    if val.contains("csh") { "csh" } else if val.contains("sh") { "sh" } else { "csh" }
}

/// Shell-specific "discard stdout+stderr" suffix.
fn devnull_redirect(shell: &str) -> &'static str {
    match shell {
        "sh" | "bash" | "ksh" => "> /dev/null 2>&1",
        _ => ">& /dev/null", // csh / tcsh
    }
}

/// Send a command, wait for a sentinel, print stdout, fail on non-zero exit.
/// csh: appends `; echo IRIS-CI-RC=$status`. sh: appends `; echo IRIS-CI-RC=$?`.
fn cmd_run(opts: &Opts, command: &str, shell: &str, timeout_ms: u64) -> Result<()> {
    let (stdout, rc) = run_capture(opts, command, shell, timeout_ms)?;
    if !stdout.is_empty() {
        println!("{}", stdout);
    }
    if rc != 0 {
        return Err(Error::Iris(format!("guest exit {}", rc)));
    }
    Ok(())
}

/// `wait-serial` for `\nIRIS-CI-RC=` returns bytes shaped like:
///
///   <typed-echo-line>\r\n<stdout>\r?\n<pattern>
///
/// Skip the first newline (end of the typed echo), strip the trailing
/// pattern + its leading newline, normalise CRLF, return.
fn extract_run_stdout(buf: &str) -> String {
    // Drop the typed-echo-line (everything up through and including its
    // first \n).
    let after_echo = match buf.find('\n') {
        Some(i) => &buf[i + 1..],
        None => buf,
    };
    // Drop the trailing `\nIRIS-CI-RC=` (we waited for `\nIRIS-CI-RC=`).
    let trimmed = match after_echo.rfind(RC_MARKER) {
        Some(p) => &after_echo[..p],
        None => after_echo,
    };
    trimmed
        .trim_end_matches(['\r', '\n'])
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}

/// Pull the digits after IRIS-CI-RC= out of a buffer.
fn parse_rc(buf: &str) -> Option<i32> {
    let pos = buf.rfind(RC_MARKER)?;
    let tail = &buf[pos + RC_MARKER.len()..];
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit() || *c == '-').collect();
    digits.parse().ok()
}


fn wait_with_deadline(opts: &Opts, pattern: &str, deadline: Instant) -> Result<()> {
    let now = Instant::now();
    if now >= deadline {
        return Err(Error::Iris(format!("wait {}: deadline already passed", pattern)));
    }
    let timeout_ms = (deadline - now).as_millis() as u64;
    send_until(
        opts,
        "wait-serial",
        json!({"pattern": pattern, "timeout_ms": timeout_ms}),
        read_timeout_for(timeout_ms),
    )
    .map(|_| ())
}

// ---- put / get (the bs=512 foot-gun killers) --------------------------------

fn cmd_put(opts: &Opts, host_path: &std::path::Path, to: Option<&str>, timeout_ms: u64) -> Result<()> {
    let bytes = std::fs::read(host_path)
        .map_err(|e| Error::Local(format!("read {}: {}", host_path.display(), e)))?;
    let basename = host_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("inject.bin");
    let guest_path = to
        .map(String::from)
        .unwrap_or_else(|| format!("/tmp/{}", basename));

    // 1. Write to scratch volume payload area at offset 0.
    let scratch_payload = send(opts, "scratch-info", json!({}))?;
    let payload_size = scratch_payload
        .get("payload_size_bytes")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| Error::Iris("scratch-info: no payload_size_bytes".into()))?;
    if (bytes.len() as u64) > payload_size {
        return Err(Error::Local(format!(
            "{} bytes too large for {} byte scratch payload",
            bytes.len(),
            payload_size
        )));
    }
    let host_path_for_socket = host_path.canonicalize()
        .unwrap_or_else(|_| host_path.to_path_buf());
    send(
        opts,
        "scratch-write",
        json!({"host_path": host_path_for_socket.display().to_string()}),
    )?;
    if !opts.quiet {
        eprintln!("put: {} bytes staged in scratch", bytes.len());
    }

    // 2. Drive the guest to read exactly the right number of 512-byte sectors.
    //    Redirect + rc-marker syntax must match the guest's actual login shell
    //    (csh `>&`/`$status` vs sh `2>&1`/`$?`), or the command errors ("bad file
    //    unit number") / the rc comes back empty. cmd_run appends the marker.
    let shell = detect_guest_shell(opts);
    let nul = devnull_redirect(shell);
    let sectors = (bytes.len() as u64).div_ceil(512);
    let dd_cmd = format!(
        "dd if=/dev/rdsk/dks0d2s0 of={} bs=512 count={} {}",
        guest_path, sectors, nul
    );
    cmd_run(opts, &dd_cmd, shell, timeout_ms)?;

    // 3. Truncate the guest file to the original byte length (dd reads in
    //    sector multiples, so a 28-byte input becomes 512 bytes on the guest).
    //    `dd of=FILE bs=1 seek=N count=0` is POSIX and IRIX-clean.
    let dd_trunc = format!(
        "dd if=/dev/null of={} bs=1 seek={} count=0 {}",
        guest_path,
        bytes.len(),
        nul
    );
    cmd_run(opts, &dd_trunc, shell, 10_000)?;

    if !opts.quiet {
        eprintln!("put: {} → {} ({} bytes)", host_path.display(), guest_path, bytes.len());
    }
    Ok(())
}

fn cmd_get(opts: &Opts, guest_path: &str, to: Option<&std::path::Path>, timeout_ms: u64) -> Result<()> {
    let host_path: PathBuf = match to {
        Some(p) => p.to_path_buf(),
        None => {
            let basename = guest_path
                .rsplit('/')
                .next()
                .filter(|s| !s.is_empty())
                .unwrap_or("captured.bin");
            PathBuf::from(basename)
        }
    };

    // 1. Zero scratch payload so trailing zeros after the file are unambiguous.
    send(opts, "scratch-clear", json!({}))?;

    // 2. Drive the guest to write the file to scratch with conv=sync padding.
    //    Redirect + rc-marker syntax must match the guest's login shell (csh
    //    `>&`/`$status` vs sh `2>&1`/`$?`). cmd_run adds the rc-marker echo.
    let shell = detect_guest_shell(opts);
    let nul = devnull_redirect(shell);
    let dd_cmd = format!(
        "dd if={} of=/dev/rdsk/dks0d2s0 bs=512 conv=sync,notrunc {}",
        guest_path, nul
    );
    cmd_run(opts, &dd_cmd, shell, timeout_ms)?;

    // 3. Look up the guest file size so we know how much to slice off the
    //    scratch payload (which is now padded to a 512-byte boundary). Use
    //    a pure-shell approach: `wc -c` outputs just the byte count.
    //    `awk` is also available but `wc -c` is simpler to parse.
    let stat_cmd = format!("wc -c < {}", guest_path);
    let (stat_stdout, stat_rc) = run_capture(opts, &stat_cmd, shell, 10_000)?;
    if stat_rc != 0 {
        return Err(Error::Iris(format!(
            "guest stat of {} failed (exit {})", guest_path, stat_rc
        )));
    }
    let size_bytes = stat_stdout
        .lines()
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .next()
        .ok_or_else(|| Error::Iris(format!(
            "couldn't parse byte count from `wc -c < {}`: {:?}",
            guest_path, stat_stdout
        )))?;

    // 5. Read the exact number of bytes back from scratch.
    let host_abs = std::path::absolute(&host_path).unwrap_or_else(|_| host_path.clone());
    send(
        opts,
        "scratch-read",
        json!({
            "to_path": host_abs.display().to_string(),
            "length": size_bytes,
            "offset": 0,
        }),
    )?;
    if !opts.quiet {
        eprintln!(
            "get: {} ({} bytes) → {}",
            guest_path,
            size_bytes,
            host_path.display()
        );
    }
    Ok(())
}

// ---- scratch raw -------------------------------------------------------------

fn cmd_scratch(opts: &Opts, s: ScratchCmd) -> Result<()> {
    match s {
        ScratchCmd::Write { path, offset } => {
            let abs = path
                .canonicalize()
                .map_err(|e| Error::Local(format!("{}: {}", path.display(), e)))?;
            simple(
                opts,
                "scratch-write",
                json!({"host_path": abs.display().to_string(), "offset": offset}),
                "wrote",
            )
        }
        ScratchCmd::Read { path, offset, length } => {
            let abs = std::path::absolute(&path).unwrap_or(path);
            let mut args = json!({"to_path": abs.display().to_string(), "offset": offset});
            if let Some(n) = length {
                args["length"] = json!(n);
            }
            simple(opts, "scratch-read", args, "read")
        }
        ScratchCmd::Clear => simple(opts, "scratch-clear", json!({}), "cleared"),
        ScratchCmd::Info => {
            let data = send(opts, "scratch-info", json!({}))?;
            if opts.json {
                println!("{}", serde_json::to_string_pretty(&data).unwrap_or_default());
            } else {
                println!("path                {}", data.get("path").cloned().unwrap_or(Value::Null));
                println!("size_bytes          {}", data.get("size_bytes").cloned().unwrap_or(Value::Null));
                println!("payload_offset      {}", data.get("payload_offset").cloned().unwrap_or(Value::Null));
                println!("payload_size_bytes  {}", data.get("payload_size_bytes").cloned().unwrap_or(Value::Null));
            }
            Ok(())
        }
    }
}

// ---- pull / push -------------------------------------------------------------

fn cmd_pull(opts: &Opts, url: &str, name: &str) -> Result<()> {
    let data = send(opts, "pull", json!({"url": url, "name": name}))?;
    if opts.json { println!("{}", serde_json::to_string_pretty(&data).unwrap_or_default()); return Ok(()); }
    let f = |k: &str| data.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
    println!(
        "pull {}: {} chunks fetched, {} skipped, {} files, {} bytes",
        name,
        f("chunks_fetched"),
        f("chunks_skipped"),
        f("files_transferred"),
        f("bytes_transferred"),
    );
    Ok(())
}

fn cmd_push(opts: &Opts, url: &str, name: &str) -> Result<()> {
    let data = send(opts, "push", json!({"url": url, "name": name}))?;
    if opts.json { println!("{}", serde_json::to_string_pretty(&data).unwrap_or_default()); return Ok(()); }
    let f = |k: &str| data.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
    println!(
        "push {}: {} chunks uploaded, {} skipped, {} files, {} bytes",
        name,
        f("chunks_uploaded"),
        f("chunks_skipped"),
        f("files_transferred"),
        f("bytes_transferred"),
    );
    Ok(())
}

// ---- script file mode --------------------------------------------------------

/// Parse a script line into argv tokens. Supports double-quoted strings with
/// `\"` and `\\` escapes — same surface as a typical shell so users can
/// write `run "echo hello"` without bash being involved.
fn tokenize(line: &str) -> std::result::Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quote = false;
    let mut escape = false;
    let mut started = false;
    for c in line.chars() {
        if escape {
            cur.push(c);
            escape = false;
            continue;
        }
        if in_quote {
            match c {
                '\\' => escape = true,
                '"' => {
                    in_quote = false;
                    out.push(std::mem::take(&mut cur));
                    started = false;
                }
                _ => cur.push(c),
            }
            continue;
        }
        match c {
            '"' => { in_quote = true; started = true; }
            ' ' | '\t' => {
                if started { out.push(std::mem::take(&mut cur)); started = false; }
            }
            _ => { cur.push(c); started = true; }
        }
    }
    if in_quote { return Err("unterminated quote".into()); }
    if started { out.push(cur); }
    Ok(out)
}

fn cmd_script(opts: &Opts, path: &std::path::Path) -> Result<()> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Local(format!("read {}: {}", path.display(), e)))?;
    let mut overall_failed = false;
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens = tokenize(line).map_err(|e| Error::Local(format!("line {}: {}", lineno + 1, e)))?;
        if tokens.is_empty() {
            continue;
        }

        // Re-parse via clap to dispatch.
        let mut argv = vec!["iris-ci".to_string()];
        argv.extend(tokens.iter().cloned());
        let cli = match Cli::try_parse_from(&argv) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[line {}] parse error: {}", lineno + 1, e);
                overall_failed = true;
                break;
            }
        };

        // Inherit our --socket / --json / --quiet from the outer invocation.
        let sub_opts = Opts {
            socket: opts.socket.clone(),
            json: opts.json || cli.json,
            quiet: opts.quiet || cli.quiet,
        };

        let pretty = format_step(line);
        let t = Instant::now();
        let res = dispatch(&sub_opts, cli.cmd);
        let elapsed = t.elapsed();
        match res {
            Ok(()) => {
                if !opts.quiet {
                    println!("[ok {:>6.0?}] {}", elapsed, pretty);
                }
            }
            Err(e) => {
                eprintln!("[FAIL {:>6.0?}] {}: {:?}", elapsed, pretty, e);
                overall_failed = true;
                break;
            }
        }
    }
    if overall_failed {
        return Err(Error::Local("script aborted on error".into()));
    }
    Ok(())
}

/// Truncate long script lines for display.
fn format_step(line: &str) -> String {
    if line.len() <= 72 { line.to_string() } else { format!("{}…", &line[..71]) }
}
