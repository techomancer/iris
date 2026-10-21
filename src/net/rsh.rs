//! Read-only rsh server (BSD `rshd`, TCP 514) for the NAT gateway.
//!
//! Pure protocol: bytes in, bytes out, no sockets, the same shape as
//! `tftp.rs`. The gateway (`services.rs`, `nat_tcp.rs`) terminates the guest's
//! TCP connection and hands the byte stream to an [`RshSession`].
//!
//! **This is not a shell and does not run host programs.** `rsh gateway cmd`
//! runs one of a few built-in commands (`cat`, `dd`, `ls`, `echo`, `pwd`) against a
//! single configured directory, read-only, the way the TFTP server serves one
//! directory. Nothing is ever executed on the host, nothing outside the root is
//! reachable (`..` and symlinks leading out are refused), and nothing is written.
//! There is no authentication: like TFTP and NFS here, the service exists only
//! on the guest-facing side of the NAT, so only the guest can reach it. As with
//! TFTP, the root is assumed not to be rewritten under the server by something
//! else on the host: a path is checked when the command starts and opened a
//! moment later, so a symlink swapped in between could slip past the check.
//!
//! The wire protocol (BSD `rcmd`/`rshd`):
//!
//! ```text
//! client -> server   <stderr port, ASCII decimal>\0 <local user>\0 <remote user>\0 <command>\0
//! server -> client   \0                       (accepted; or \1 <message>\n and close)
//!                    <command stdout ...>
//! server -> client   (second connection, from a reserved port to the stderr
//!                    port the client listens on) <command stderr ...>
//! ```
//!
//! A stderr port of `0` means the client wants no second connection; error
//! text then goes down the main connection. That connection is requested
//! through [`TcpService::side_connect`]. The protocol carries no exit status,
//! so a failing command is reported only in its error text.
//!
//! **The connection back must not wait for the rest of the request.** `rcmd`
//! sends *only* the stderr port, then blocks accepting the second connection,
//! and sends the user names and the command only after that. A server that
//! waits for the whole request before connecting back deadlocks with it until
//! the client gives up (inst's 120 s timeout). So the connection is requested
//! as soon as the first field has arrived.
//!
//! `dd if=FILE [bs=N] [ibs=N] [count=N] [iseek=N|skip=N]` is what IRIX `inst`
//! uses to read a distribution from a remote host: it restarts the remote `dd`
//! with `iseek` for a long forward seek (see inst(1), `network_seek_threshold`).
//! Output is only ever the file's bytes; dd's "records in/out" report goes to
//! the stderr connection and is dropped when there is none, so it can never
//! end up inside the data.

use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

use super::tcp::{SideConnect, TcpService};
use crate::devlog::LogModule;

pub const RSH_PORT: u16 = 514;

/// The request header (four NUL-terminated fields) may not exceed this.
const MAX_HEADER: usize = 8192;
/// Longest command line accepted.
const MAX_COMMAND: usize = 4096;
/// Longest user name field accepted (BSD uses 16; be lenient).
const MAX_USER: usize = 64;
/// A directory listing is cut off here rather than built without bound.
const MAX_LS_ENTRIES: usize = 100_000;

// ── Request parsing ──────────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Eq)]
pub struct Request {
    /// Port the client listens on for stderr; 0 for none.
    pub stderr_port: u16,
    pub local_user: String,
    pub remote_user: String,
    pub command: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Parse {
    /// Not all four fields have arrived yet.
    Need,
    /// Malformed; the text goes back after a `\1`.
    Bad(&'static str),
    /// A whole request.
    Done(Request),
}

const PROTOCOL_FAILURE: &str = "rshd: protocol failure in circuit setup";

/// Parse the request header from the bytes received so far. Bytes after the
/// fourth NUL (the client's stdin) are not looked at.
pub fn parse_request(buf: &[u8]) -> Parse {
    let mut fields: Vec<&[u8]> = Vec::with_capacity(4);
    let mut start = 0;
    for (i, &b) in buf.iter().enumerate() {
        if b == 0 {
            fields.push(&buf[start..i]);
            start = i + 1;
            if fields.len() == 4 {
                break;
            }
        }
    }
    // Reject what is already wrong, whether or not the rest has arrived.
    if let Some(port) = fields.first().copied().or((!buf.is_empty()).then_some(buf)) {
        if port.len() > 5 || !port.iter().all(u8::is_ascii_digit) {
            return Parse::Bad(PROTOCOL_FAILURE);
        }
    }
    if fields.len() < 4 {
        let pending = &buf[start..];
        let limit = match fields.len() {
            1 | 2 => MAX_USER,
            3 => MAX_COMMAND,
            _ => 5,
        };
        if pending.len() > limit || buf.len() > MAX_HEADER {
            return Parse::Bad(if fields.len() == 3 { "rshd: command too long" } else { PROTOCOL_FAILURE });
        }
        return Parse::Need;
    }
    let text = |f: &[u8]| String::from_utf8_lossy(f).into_owned();
    let port = match std::str::from_utf8(fields[0]).unwrap_or("0") {
        "" => 0,
        s => match s.parse::<u32>() {
            Ok(p) if p <= u16::MAX as u32 => p as u16,
            _ => return Parse::Bad(PROTOCOL_FAILURE),
        },
    };
    if fields[1].len() > MAX_USER || fields[2].len() > MAX_USER {
        return Parse::Bad(PROTOCOL_FAILURE);
    }
    if fields[3].len() > MAX_COMMAND {
        return Parse::Bad("rshd: command too long");
    }
    Parse::Done(Request {
        stderr_port: port,
        local_user: text(fields[1]),
        remote_user: text(fields[2]),
        command: text(fields[3]),
    })
}

/// The stderr port field: empty (rcmd's "none") or up to five decimal digits.
fn parse_port(field: &[u8]) -> Option<u16> {
    if field.len() > 5 || !field.iter().all(u8::is_ascii_digit) {
        return None;
    }
    match std::str::from_utf8(field).ok()? {
        "" => Some(0),
        s => s.parse::<u16>().ok(),
    }
}

/// Split a command line into words: whitespace separates, single and double
/// quotes group, backslash escapes. Anything that would need a real shell
/// (pipes, redirects, expansion, wildcards) is refused instead of being
/// silently taken literally.
pub fn split_words(cmd: &str) -> Result<Vec<String>, &'static str> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut have = false;
    let mut chars = cmd.chars();
    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' | '\n' | '\r' => {
                if have {
                    words.push(std::mem::take(&mut cur));
                    have = false;
                }
            }
            '\'' => {
                have = true;
                loop {
                    match chars.next() {
                        Some('\'') => break,
                        Some(q) => cur.push(q),
                        None => return Err("Unmatched '."),
                    }
                }
            }
            '"' => {
                have = true;
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some(e @ ('"' | '\\')) => cur.push(e),
                            Some(e) => {
                                cur.push('\\');
                                cur.push(e);
                            }
                            None => return Err("Unmatched \"."),
                        },
                        Some(q) => cur.push(q),
                        None => return Err("Unmatched \"."),
                    }
                }
            }
            '\\' => {
                have = true;
                match chars.next() {
                    Some(e) => cur.push(e),
                    None => cur.push('\\'),
                }
            }
            '|' | '&' | ';' | '<' | '>' | '(' | ')' | '$' | '`' => {
                return Err("shell syntax (pipes, redirects, expansion) is not supported.");
            }
            '*' | '?' | '[' => return Err("wildcards are not supported."),
            _ => {
                have = true;
                cur.push(c);
            }
        }
    }
    if have {
        words.push(cur);
    }
    Ok(words)
}

// ── The sandbox ──────────────────────────────────────────────────────────────

/// Resolve `name` under `root`: leading slashes are dropped (so `/x` is
/// `<root>/x`), `.` is skipped, `..` is refused, and the result must still be
/// inside `root` once symlinks are followed. An empty name or `/` is the root.
fn resolve(root: &Path, name: &str) -> io::Result<PathBuf> {
    let mut rel = PathBuf::new();
    for c in Path::new(name.trim_start_matches('/')).components() {
        match c {
            Component::Normal(p) => rel.push(p),
            Component::CurDir => {}
            _ => return Err(io::ErrorKind::PermissionDenied.into()),
        }
    }
    let root = root.canonicalize()?;
    let full = root.join(rel).canonicalize()?;
    if full.starts_with(&root) {
        Ok(full)
    } else {
        Err(io::ErrorKind::PermissionDenied.into())
    }
}

/// A dd number: decimal digits with an optional `b` (512), `k` (1024) or `w`
/// (2) suffix, as IRIX dd takes them.
fn parse_size(s: &str) -> Option<u64> {
    let (digits, mult) = match s.chars().last()? {
        'b' => (&s[..s.len() - 1], 512),
        'k' => (&s[..s.len() - 1], 1024),
        'w' => (&s[..s.len() - 1], 2),
        _ => (s, 1),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok()?.checked_mul(mult)
}

fn errstr(e: &io::Error) -> String {
    match e.kind() {
        io::ErrorKind::NotFound => "No such file or directory".into(),
        io::ErrorKind::PermissionDenied => "Permission denied".into(),
        _ => e.to_string(),
    }
}

// ── Output channels ──────────────────────────────────────────────────────────

/// What the stderr connection drains: filled by the session, closed when the
/// command is over.
#[derive(Default)]
struct Chan {
    data: VecDeque<u8>,
    closed: bool,
}

/// The service on the stderr connection the gateway opens to the guest.
struct StderrService(Arc<Mutex<Chan>>);

impl TcpService for StderrService {
    /// The client may write signal numbers here; there is no process to signal.
    fn input(&mut self, _: &[u8]) -> Vec<u8> {
        Vec::new()
    }
    fn pull(&mut self, max: usize) -> Vec<u8> {
        let mut c = self.0.lock();
        let n = max.min(c.data.len());
        c.data.drain(..n).collect()
    }
    fn done(&self) -> bool {
        let c = self.0.lock();
        c.closed && c.data.is_empty()
    }
}

// ── The session ──────────────────────────────────────────────────────────────

#[derive(PartialEq, Eq)]
enum Phase {
    /// Collecting the request header.
    Header,
    /// Request accepted; the command's output is being produced.
    Running,
}

/// A command whose output is streamed from files rather than built up front.
enum Job {
    None,
    Cat { files: VecDeque<PathBuf>, cur: Option<File> },
    /// `dd`: `left` input bytes still to copy (None = to the end of the file),
    /// `total` copied so far, `ibs` for the records report.
    Dd { file: File, left: Option<u64>, total: u64, ibs: u64 },
}

/// With `IRIS_RSH_TRACE` set, every request and how it ended is written to
/// stderr, in any build (`dlog_dev!` output exists only in `developer` builds).
/// It shows what a client asks for, how many bytes each request moved and how
/// long it took, and requests that never finished, which is what slow or hung
/// installs need.
pub(super) fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("IRIS_RSH_TRACE").is_some())
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub struct RshSession {
    id: u64,
    trace: bool,
    started: Instant,
    command: String,
    /// Stdout bytes handed to the connection so far.
    sent: u64,
    root: PathBuf,
    phase: Phase,
    inbuf: Vec<u8>,
    /// Stdout bytes ready to go.
    out: VecDeque<u8>,
    job: Job,
    chan: Option<Arc<Mutex<Chan>>>,
    side: Option<SideConnect>,
    finished: bool,
    /// The stderr port field has arrived and been acted on.
    port_seen: bool,
    /// A whole request has arrived and the command was started.
    requested: bool,
    /// The stderr port the client named, for the trace.
    stderr_port: u16,
}

impl RshSession {
    /// Serve the built-in commands out of `root`. Nothing outside it is reachable.
    pub fn new(root: PathBuf) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed), trace: trace_enabled(),
            started: Instant::now(), command: String::new(), sent: 0,
            root, phase: Phase::Header, inbuf: Vec::new(), out: VecDeque::new(),
            job: Job::None, chan: None, side: None, finished: false,
            port_seen: false, requested: false, stderr_port: 0,
        }
    }

    /// The command is over: no more output, and the stderr channel can close.
    fn finish(&mut self) {
        self.finished = true;
        self.job = Job::None;
        if let Some(c) = &self.chan {
            c.lock().closed = true;
        }
    }

    /// Error text for the client: on the stderr connection when it has one,
    /// else down the main connection.
    fn err(&mut self, msg: &str) {
        match &self.chan {
            Some(c) => c.lock().data.extend(msg.as_bytes()),
            None => self.out.extend(msg.as_bytes()),
        }
    }

    fn reject(&mut self, msg: &str) {
        self.out.push_back(1);
        self.out.extend(msg.as_bytes());
        self.out.push_back(b'\n');
        self.finish();
    }

    /// Ask for the connection back to the client's stderr port, once.
    fn open_stderr(&mut self, port: u16) {
        self.stderr_port = port;
        if port == 0 || self.chan.is_some() {
            return;
        }
        if self.trace {
            eprintln!("iris: rsh #{} stderr port {}: connecting back to the guest", self.id, port);
        }
        let chan = Arc::new(Mutex::new(Chan::default()));
        self.side = Some(SideConnect { guest_port: port, service: Box::new(StderrService(chan.clone())) });
        self.chan = Some(chan);
    }

    fn start(&mut self, req: Request) {
        dlog_dev!(LogModule::Net, "NAT rsh local user {:?}, remote user {:?}: {:?}",
                  req.local_user, req.remote_user, req.command);
        self.phase = Phase::Running;
        if self.trace {
            eprintln!("iris: rsh #{} {}@{}: {:?} (stderr port {})", self.id, req.remote_user,
                      req.local_user, req.command, req.stderr_port);
        }
        self.command = req.command.clone();
        self.requested = true;
        self.open_stderr(req.stderr_port); // a no-op: the first field already did
        self.out.push_back(0); // accepted
        let words = match split_words(&req.command) {
            Ok(w) => w,
            Err(e) => {
                self.err(&format!("rshd: {e}\n"));
                return self.finish();
            }
        };
        let Some((name, args)) = words.split_first() else {
            self.err("rshd: no command given (interactive login is not supported)\n");
            return self.finish();
        };
        match name.as_str() {
            "cat" => self.cmd_cat(args),
            "dd" => self.cmd_dd(args),
            "ls" => self.cmd_ls(args),
            "echo" => self.cmd_echo(args),
            "pwd" => {
                self.out.extend(b"/\n");
                self.finish();
            }
            other => {
                self.err(&format!("{other}: Command not found.\n"));
                self.finish();
            }
        }
    }

    fn cmd_echo(&mut self, args: &[String]) {
        let (newline, args) = match args.split_first() {
            Some((f, rest)) if f == "-n" => (false, rest),
            _ => (true, args),
        };
        self.out.extend(args.join(" ").bytes());
        if newline {
            self.out.push_back(b'\n');
        }
        self.finish();
    }

    fn cmd_cat(&mut self, args: &[String]) {
        if args.is_empty() {
            self.err("cat: reading standard input is not supported\n");
            return self.finish();
        }
        let mut files = VecDeque::new();
        for a in args {
            match resolve(&self.root, a) {
                Ok(p) if p.is_dir() => self.err(&format!("cat: {a}: Is a directory\n")),
                Ok(p) => files.push_back(p),
                Err(e) => self.err(&format!("cat: {a}: {}\n", errstr(&e))),
            }
        }
        if files.is_empty() {
            return self.finish();
        }
        self.job = Job::Cat { files, cur: None };
    }

    fn cmd_dd(&mut self, args: &[String]) {
        match self.parse_dd(args) {
            Ok(job) => self.job = job,
            Err(msg) => {
                self.err(&format!("dd: {msg}\n"));
                self.finish();
            }
        }
    }

    /// Open the input and position it, or say what is wrong with the operands.
    fn parse_dd(&self, args: &[String]) -> Result<Job, String> {
        let (mut input, mut ibs, mut count, mut seek) = (None, 512u64, None, 0u64);
        for a in args {
            let (key, val) = a.split_once('=').ok_or_else(|| format!("bad arg: {a}"))?;
            let num = || parse_size(val).ok_or_else(|| format!("bad numeric arg: {a}"));
            match key {
                "if" => input = Some(val),
                "bs" | "ibs" => ibs = num()?,
                "obs" | "cbs" => {
                    num()?;
                }
                "count" => count = Some(num()?),
                "iseek" | "skip" => seek = num()?,
                "of" if val == "/dev/stdout" || val == "-" => {}
                "of" => return Err("writing files is not supported".into()),
                _ => return Err(format!("bad arg: {a}")),
            }
        }
        let name = input.ok_or("reading standard input is not supported")?;
        if ibs == 0 {
            return Err("bad block size".into());
        }
        let path = resolve(&self.root, name).map_err(|e| format!("{name}: {}", errstr(&e)))?;
        if path.is_dir() {
            return Err(format!("{name}: Is a directory"));
        }
        let mut file = File::open(&path).map_err(|e| format!("{name}: {}", errstr(&e)))?;
        let start = seek.checked_mul(ibs).ok_or("seek too large")?;
        let left = count.map(|c| c.checked_mul(ibs).ok_or("count too large")).transpose()?;
        file.seek(SeekFrom::Start(start)).map_err(|e| format!("{name}: {}", errstr(&e)))?;
        Ok(Job::Dd { file, left, total: 0, ibs })
    }

    fn cmd_ls(&mut self, args: &[String]) {
        let (mut all, mut long) = (false, false);
        let mut operands = Vec::new();
        let mut opts_done = false;
        for a in args {
            if !opts_done && a == "--" {
                opts_done = true;
            } else if !opts_done && a.len() > 1 && a.starts_with('-') {
                for c in a[1..].chars() {
                    match c {
                        'a' => all = true,
                        'l' => long = true,
                        '1' => {}
                        _ => {
                            self.err(&format!("ls: illegal option -- {c}\nusage: ls [-1al] [file ...]\n"));
                            return self.finish();
                        }
                    }
                }
            } else {
                operands.push(a.as_str());
            }
        }
        if operands.is_empty() {
            operands.push(".");
        }
        let headers = operands.len() > 1;
        let mut text = String::new();
        for (i, name) in operands.iter().enumerate() {
            let path = match resolve(&self.root, name) {
                Ok(p) => p,
                Err(e) => {
                    self.err(&format!("ls: {name}: {}\n", errstr(&e)));
                    continue;
                }
            };
            if !path.is_dir() {
                match fs::symlink_metadata(&path) {
                    Ok(m) => text.push_str(&ls_line(name, &m, long)),
                    Err(e) => self.err(&format!("ls: {name}: {}\n", errstr(&e))),
                }
                continue;
            }
            if headers {
                if i > 0 {
                    text.push('\n');
                }
                text.push_str(&format!("{name}:\n"));
            }
            let entries = match fs::read_dir(&path) {
                Ok(rd) => rd,
                Err(e) => {
                    self.err(&format!("ls: {name}: {}\n", errstr(&e)));
                    continue;
                }
            };
            let mut rows: Vec<(String, fs::Metadata)> = entries
                .flatten()
                .filter_map(|e| Some((e.file_name().to_string_lossy().into_owned(), e.metadata().ok()?)))
                .filter(|(n, _)| all || !n.starts_with('.'))
                .take(MAX_LS_ENTRIES)
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            for (n, m) in &rows {
                text.push_str(&ls_line(n, m, long));
            }
        }
        self.out.extend(text.bytes());
        self.finish();
    }

    /// Move up to `want` bytes of a streaming job into `out`.
    fn advance_job(&mut self, want: usize) {
        match self.job {
            Job::None => {}
            Job::Cat { .. } => self.advance_cat(want),
            Job::Dd { .. } => self.advance_dd(want),
        }
    }

    fn advance_dd(&mut self, want: usize) {
        let Job::Dd { file, left, total, ibs } = &mut self.job else { return };
        let cap = left.map_or(want as u64, |l| l.min(want as u64)) as usize;
        let mut buf = vec![0u8; cap];
        let res = loop {
            if cap == 0 {
                break Ok(0);
            }
            match file.read(&mut buf) {
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                r => break r,
            }
        };
        match res {
            Ok(n) if n > 0 => {
                *total += n as u64;
                if let Some(l) = left {
                    *l -= n as u64;
                }
                self.out.extend(&buf[..n]);
            }
            Ok(_) => {
                // The report goes to the stderr connection only: with none, it
                // would land in the middle of the data.
                let (full, partial) = (*total / *ibs, u64::from(*total % *ibs != 0));
                if let Some(c) = &self.chan {
                    let report = format!("{full}+{partial} records in\n{full}+{partial} records out\n");
                    c.lock().data.extend(report.as_bytes());
                }
                self.finish();
            }
            Err(e) => {
                let msg = format!("dd: read error: {}\n", errstr(&e));
                self.err(&msg);
                self.finish();
            }
        }
    }

    fn advance_cat(&mut self, want: usize) {
        let mut buf = vec![0u8; want];
        loop {
            let Job::Cat { files, cur } = &mut self.job else { return };
            if cur.is_none() {
                let Some(path) = files.pop_front() else {
                    return self.finish();
                };
                match File::open(&path) {
                    Ok(f) => *cur = Some(f),
                    Err(e) => {
                        // `path` is already resolved; show the host path only
                        // by its last component, not the layout of the host.
                        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        let msg = format!("cat: {name}: {}\n", errstr(&e));
                        self.err(&msg);
                        continue;
                    }
                }
            }
            let file = cur.as_mut().unwrap();
            match file.read(&mut buf) {
                Ok(0) => *cur = None,
                Ok(n) => {
                    self.out.extend(&buf[..n]);
                    return;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    *cur = None;
                    let msg = format!("cat: read error: {}\n", errstr(&e));
                    self.err(&msg);
                }
            }
        }
    }
}

/// One `ls` output line: the bare name, or a long line with type/permissions
/// and size (there is no owner or date: the host's are meaningless to IRIX).
fn ls_line(name: &str, m: &fs::Metadata, long: bool) -> String {
    if !long {
        return format!("{name}\n");
    }
    format!("{} {:>10} {name}\n", mode_string(m), m.len())
}

fn mode_string(m: &fs::Metadata) -> String {
    let kind = if m.is_dir() { 'd' } else if m.file_type().is_symlink() { 'l' } else { '-' };
    #[cfg(unix)]
    let bits = {
        use std::os::unix::fs::PermissionsExt;
        m.permissions().mode()
    };
    #[cfg(not(unix))]
    let bits = if m.permissions().readonly() { 0o555 } else { 0o755 };
    let mut s = String::with_capacity(10);
    s.push(kind);
    for shift in [6, 3, 0] {
        let b = (bits >> shift) & 7;
        s.push(if b & 4 != 0 { 'r' } else { '-' });
        s.push(if b & 2 != 0 { 'w' } else { '-' });
        s.push(if b & 1 != 0 { 'x' } else { '-' });
    }
    s
}

impl Drop for RshSession {
    fn drop(&mut self) {
        if !self.trace {
            return;
        }
        let ms = self.started.elapsed().as_millis();
        let how = if !self.requested {
            format!("the guest never sent a whole request ({} bytes received, stderr port {})",
                    self.inbuf.len(), if self.port_seen { self.stderr_port.to_string() } else { "not yet seen".into() })
        } else if self.finished && self.out.is_empty() {
            "complete".to_string()
        } else if self.finished {
            format!("closed with {} bytes unsent", self.out.len())
        } else {
            "closed by the guest before the command finished".to_string()
        };
        eprintln!("iris: rsh #{} {:?}: {} bytes in {} ms, {}", self.id, self.command, self.sent, ms, how);
    }
}

impl TcpService for RshSession {
    fn input(&mut self, data: &[u8]) -> Vec<u8> {
        // Anything after the header is the client's stdin: there is no process
        // to feed it to.
        if self.phase != Phase::Header || self.finished {
            return Vec::new();
        }
        self.inbuf.extend_from_slice(data);
        // rcmd sends the stderr port alone and waits for us to connect back
        // before it sends anything else: act on it as soon as it is here.
        if !self.port_seen {
            match self.inbuf.iter().position(|&b| b == 0) {
                Some(end) => match parse_port(&self.inbuf[..end]) {
                    Some(port) => {
                        self.port_seen = true;
                        self.open_stderr(port);
                    }
                    None => {
                        self.phase = Phase::Running;
                        self.reject(PROTOCOL_FAILURE);
                        return Vec::new();
                    }
                },
                None if self.inbuf.len() > 5 => {
                    self.phase = Phase::Running;
                    self.reject(PROTOCOL_FAILURE);
                    return Vec::new();
                }
                None => return Vec::new(),
            }
        }
        match parse_request(&self.inbuf) {
            Parse::Need => {}
            Parse::Bad(msg) => {
                self.phase = Phase::Running;
                self.reject(msg);
            }
            Parse::Done(req) => self.start(req),
        }
        Vec::new()
    }

    fn pull(&mut self, max: usize) -> Vec<u8> {
        if self.out.len() < max {
            self.advance_job(max - self.out.len());
        }
        let n = max.min(self.out.len());
        self.sent += n as u64;
        self.out.drain(..n).collect()
    }

    fn done(&self) -> bool {
        self.finished && self.out.is_empty()
    }

    fn guest_closed(&mut self) {
        // A client that hangs up before finishing its request gets nothing.
        // After the request a half-close (`rsh -n`) is normal: keep going.
        if self.phase == Phase::Header {
            self.phase = Phase::Running;
            self.finish();
        }
    }

    fn side_connect(&mut self) -> Option<SideConnect> {
        self.side.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(port: &str, cmd: &str) -> Vec<u8> {
        format!("{port}\0guest\0root\0{cmd}\0").into_bytes()
    }

    fn tmp_root(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let d = std::env::temp_dir().join(format!("iris-rsh-{tag}-{nanos}"));
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// Run one request to completion in small pulls; returns (stdout, stderr).
    /// `stderr` is None when the client asked for no second connection.
    fn run(root: &Path, request: &[u8]) -> (Vec<u8>, Option<Vec<u8>>) {
        let mut s = RshSession::new(root.to_path_buf());
        assert!(s.input(request).is_empty());
        let mut side = s.side_connect();
        assert!(s.side_connect().is_none(), "the side connection is requested once");
        let mut out = Vec::new();
        let mut err = Vec::new();
        for _ in 0..100_000 {
            let c = s.pull(7);
            out.extend_from_slice(&c);
            if let Some(sc) = side.as_mut() {
                err.extend(sc.service.pull(5));
            }
            if c.is_empty() && s.done() && side.as_ref().is_none_or(|sc| sc.service.done()) {
                break;
            }
        }
        assert!(s.done());
        (out, side.map(|_| err))
    }

    #[test]
    fn parses_a_whole_request() {
        let r = parse_request(&req("1023", "cat /x"));
        assert_eq!(r, Parse::Done(Request {
            stderr_port: 1023, local_user: "guest".into(), remote_user: "root".into(), command: "cat /x".into(),
        }));
        // An empty stderr port is "none".
        assert!(matches!(parse_request(&req("", "pwd")), Parse::Done(Request { stderr_port: 0, .. })));
    }

    /// rcmd sends the stderr port alone and waits for the connection back
    /// before it sends the rest. Waiting for the whole request deadlocks.
    #[test]
    fn the_connection_back_does_not_wait_for_the_command() {
        let root = tmp_root("handshake");
        fs::write(root.join("f"), b"data").unwrap();
        let mut s = RshSession::new(root.clone());
        assert!(s.input(b"40").is_empty());
        assert!(s.side_connect().is_none(), "half a port number is not a port");
        s.input(b"00\0");
        let side = s.side_connect().expect("asked for as soon as the port is complete");
        assert_eq!(side.guest_port, 4000);
        assert!(s.pull(64).is_empty() && !s.done(), "nothing is accepted before the command");
        s.input(b"guest\0root\0cat f\0");
        assert!(s.side_connect().is_none(), "asked for once");
        let mut out = Vec::new();
        while !s.done() {
            out.extend(s.pull(64));
        }
        assert_eq!(out, b"\0data");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rcmd_without_a_stderr_connection_sends_an_empty_port() {
        let root = tmp_root("noerr");
        let mut s = RshSession::new(root.clone());
        s.input(b"\0");
        assert!(s.side_connect().is_none());
        s.input(b"a\0b\0pwd\0");
        let mut out = Vec::new();
        while !s.done() {
            out.extend(s.pull(64));
        }
        assert_eq!(out, b"\0/\n");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_bad_port_is_refused_at_once() {
        for bad in [&b"12x\0"[..], b"123456\0", b"99999\0", b"abcdefg"] {
            let mut s = RshSession::new(std::env::temp_dir());
            s.input(bad);
            assert!(s.side_connect().is_none());
            assert_eq!(s.pull(100)[0], 1, "{bad:?}");
            assert!(s.done());
        }
    }

    #[test]
    fn a_partial_request_waits() {
        let full = req("5000", "ls");
        for n in 0..full.len() {
            assert_eq!(parse_request(&full[..n]), Parse::Need, "{n} bytes");
        }
    }

    #[test]
    fn bytes_after_the_header_are_ignored() {
        let mut b = req("0", "pwd");
        b.extend_from_slice(b"stdin data");
        assert!(matches!(parse_request(&b), Parse::Done(_)));
    }

    #[test]
    fn malformed_headers_are_refused() {
        assert_eq!(parse_request(b"12x\0"), Parse::Bad(PROTOCOL_FAILURE));
        assert_eq!(parse_request(b"123456\0a\0b\0c\0"), Parse::Bad(PROTOCOL_FAILURE));
        assert_eq!(parse_request(b"99999\0a\0b\0c\0"), Parse::Bad(PROTOCOL_FAILURE)); // > 65535
        // Refused as soon as it is wrong, without waiting for a NUL.
        assert_eq!(parse_request(b"abcdef"), Parse::Bad(PROTOCOL_FAILURE));
        assert_eq!(parse_request(&vec![b'1'; 9000]), Parse::Bad(PROTOCOL_FAILURE));
        let long_cmd = req("0", &"x".repeat(MAX_COMMAND + 1));
        assert!(matches!(parse_request(&long_cmd), Parse::Bad(_)));
        let mut endless = b"0\0a\0b\0".to_vec();
        endless.extend(std::iter::repeat_n(b'y', MAX_COMMAND + 10));
        assert!(matches!(parse_request(&endless), Parse::Bad(_)), "an unterminated command is cut off");
    }

    #[test]
    fn splits_words_like_a_shell_would() {
        assert_eq!(split_words("cat  a   b").unwrap(), ["cat", "a", "b"]);
        assert_eq!(split_words("echo 'a  b' \"c d\" e\\ f").unwrap(), ["echo", "a  b", "c d", "e f"]);
        assert_eq!(split_words("echo ''").unwrap(), ["echo", ""]);
        assert_eq!(split_words("echo \"say \\\"hi\\\"\"").unwrap(), ["echo", "say \"hi\""]);
        assert!(split_words("   ").unwrap().is_empty());
        assert!(split_words("echo 'oops").is_err());
        for bad in ["cat a | wc", "cat a > b", "echo $HOME", "echo `id`", "a; b", "a && b", "cat *", "ls ?.c", "ls [a]"] {
            assert!(split_words(bad).is_err(), "{bad:?} must be refused");
        }
        // Quoted metacharacters are just text.
        assert_eq!(split_words("echo '|;*' \"$x\"").unwrap(), ["echo", "|;*", "$x"]);
    }

    #[test]
    fn resolve_stays_inside_the_root() {
        let root = tmp_root("resolve");
        fs::create_dir_all(root.join("d")).unwrap();
        fs::write(root.join("d/f"), b"x").unwrap();
        let canon = root.canonicalize().unwrap();
        assert_eq!(resolve(&root, "/d/f").unwrap(), canon.join("d/f"));
        assert_eq!(resolve(&root, "d/./f").unwrap(), canon.join("d/f"));
        assert_eq!(resolve(&root, "").unwrap(), canon);
        assert_eq!(resolve(&root, "/").unwrap(), canon);
        assert_eq!(resolve(&root, ".").unwrap(), canon);
        for bad in ["..", "../x", "d/../../x", "/d/../.."] {
            assert_eq!(resolve(&root, bad).unwrap_err().kind(), io::ErrorKind::PermissionDenied, "{bad}");
        }
        // An absolute name is relative to the root, not the host.
        assert_eq!(resolve(&root, "/etc/passwd").unwrap_err().kind(), io::ErrorKind::NotFound);
        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_out_of_the_root_is_refused() {
        let root = tmp_root("symlink");
        let outside = tmp_root("outside");
        fs::write(outside.join("secret"), b"nope").unwrap();
        fs::write(root.join("ok"), b"fine").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("dir")).unwrap();
        std::os::unix::fs::symlink(outside.join("secret"), root.join("file")).unwrap();
        std::os::unix::fs::symlink("ok", root.join("inside")).unwrap();
        for bad in ["dir", "dir/secret", "file"] {
            assert_eq!(resolve(&root, bad).unwrap_err().kind(), io::ErrorKind::PermissionDenied, "{bad}");
        }
        assert!(resolve(&root, "inside").is_ok(), "a link that stays inside is fine");
        let (out, err) = run(&root, &req("77", "cat file dir/secret inside"));
        assert_eq!(out, b"\0fine");
        assert_eq!(String::from_utf8(err.unwrap()).unwrap(),
                   "cat: file: Permission denied\ncat: dir/secret: Permission denied\n");
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&outside);
    }

    #[test]
    fn cat_streams_files_in_order() {
        let root = tmp_root("cat");
        let big: Vec<u8> = (0..70_000u32).map(|i| (i % 251) as u8).collect();
        fs::write(root.join("a"), b"first\n").unwrap();
        fs::write(root.join("big"), &big).unwrap();
        let (out, err) = run(&root, &req("0", "cat a /big a"));
        assert!(err.is_none());
        let mut want = vec![0u8];
        want.extend_from_slice(b"first\n");
        want.extend_from_slice(&big);
        want.extend_from_slice(b"first\n");
        assert_eq!(out, want);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn cat_errors_go_to_stderr_when_there_is_one() {
        let root = tmp_root("caterr");
        fs::write(root.join("a"), b"A").unwrap();
        fs::create_dir(root.join("d")).unwrap();
        let (out, err) = run(&root, &req("4000", "cat missing a d ../a"));
        assert_eq!(out, b"\0A");
        assert_eq!(String::from_utf8(err.unwrap()).unwrap(),
                   "cat: missing: No such file or directory\ncat: d: Is a directory\ncat: ../a: Permission denied\n");
        // With no stderr connection the same text comes down stdout.
        let (out, err) = run(&root, &req("0", "cat missing"));
        assert!(err.is_none());
        assert_eq!(out, b"\0cat: missing: No such file or directory\n");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn ls_lists_sorted_without_dotfiles() {
        let root = tmp_root("ls");
        for n in ["b", "a", ".hidden", "c"] {
            fs::write(root.join(n), b"12345").unwrap();
        }
        fs::create_dir(root.join("sub")).unwrap();
        fs::write(root.join("sub/x"), b"").unwrap();
        assert_eq!(run(&root, &req("0", "ls")).0, b"\0a\nb\nc\nsub\n");
        assert_eq!(run(&root, &req("0", "ls -a /")).0, b"\0.hidden\na\nb\nc\nsub\n");
        assert_eq!(run(&root, &req("0", "ls sub")).0, b"\0x\n");
        assert_eq!(run(&root, &req("0", "ls a")).0, b"\0a\n");
        assert_eq!(run(&root, &req("0", "ls . sub")).0, b"\0.:\na\nb\nc\nsub\n\nsub:\nx\n");
        let long = String::from_utf8(run(&root, &req("0", "ls -l")).0).unwrap();
        let lines: Vec<&str> = long.trim_start_matches('\0').lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with('-') && lines[0].ends_with("5 a"), "{:?}", lines[0]);
        assert!(lines[3].starts_with('d') && lines[3].ends_with(" sub"), "{:?}", lines[3]);
        let (out, err) = run(&root, &req("9", "ls -z"));
        assert_eq!(out, b"\0");
        assert!(String::from_utf8(err.unwrap()).unwrap().starts_with("ls: illegal option -- z\n"));
        let (_, err) = run(&root, &req("9", "ls nothing .."));
        assert_eq!(String::from_utf8(err.unwrap()).unwrap(),
                   "ls: nothing: No such file or directory\nls: ..: Permission denied\n");
        let _ = fs::remove_dir_all(&root);
    }

    fn dd_fixture(tag: &str) -> (PathBuf, Vec<u8>) {
        let root = tmp_root(tag);
        let data: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        fs::write(root.join("dist"), &data).unwrap();
        (root, data)
    }

    #[test]
    fn dd_copies_the_whole_file_and_reports_on_stderr() {
        let (root, data) = dd_fixture("dd1");
        let (out, err) = run(&root, &req("4000", "dd if=/dist bs=10240"));
        let mut want = vec![0u8];
        want.extend_from_slice(&data);
        assert_eq!(out, want);
        // 100000 = 9 full 10240-byte records and a partial one.
        assert_eq!(err.unwrap(), b"9+1 records in\n9+1 records out\n");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn dd_seeks_and_counts_in_blocks() {
        let (root, data) = dd_fixture("dd2");
        // inst restarts the remote dd like this after a long forward seek.
        let (out, err) = run(&root, &req("1", "dd if=dist bs=10240 iseek=3 count=2"));
        assert_eq!(&out[1..], &data[30720..51200]);
        assert_eq!(err.unwrap(), b"2+0 records in\n2+0 records out\n");
        // skip= is the same thing; ibs= alone sets the input block size; k suffix.
        let (out, _) = run(&root, &req("1", "dd if=dist ibs=1k skip=2 count=3"));
        assert_eq!(&out[1..], &data[2048..5120]);
        // Default block size is 512; a seek past the end copies nothing.
        let (out, _) = run(&root, &req("1", "dd if=dist count=1"));
        assert_eq!(&out[1..], &data[..512]);
        let (out, err) = run(&root, &req("1", "dd if=dist iseek=100000"));
        assert_eq!(out, b"\0");
        assert_eq!(err.unwrap(), b"0+0 records in\n0+0 records out\n");
        let (out, _) = run(&root, &req("1", "dd if=dist count=0"));
        assert_eq!(out, b"\0");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn dd_never_puts_its_report_in_the_data() {
        let (root, data) = dd_fixture("dd3");
        let (out, err) = run(&root, &req("0", "dd if=dist bs=4k"));
        assert!(err.is_none());
        assert_eq!(&out[1..], &data[..], "with no stderr connection the stream is exactly the file");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn dd_refuses_what_it_cannot_do() {
        let (root, _) = dd_fixture("dd4");
        for (cmd, msg) in [
            ("dd", "dd: reading standard input is not supported\n"),
            ("dd if=dist of=/tmp/x", "dd: writing files is not supported\n"),
            ("dd if=nothing", "dd: nothing: No such file or directory\n"),
            ("dd if=../dist", "dd: ../dist: Permission denied\n"),
            ("dd if=/", "dd: /: Is a directory\n"),
            ("dd if=dist bs=0", "dd: bad block size\n"),
            ("dd if=dist bs=x", "dd: bad numeric arg: bs=x\n"),
            ("dd if=dist conv=sync", "dd: bad arg: conv=sync\n"),
            ("dd if=dist junk", "dd: bad arg: junk\n"),
            ("dd if=dist iseek=99999999999999999999", "dd: bad numeric arg: iseek=99999999999999999999\n"),
        ] {
            let (out, err) = run(&root, &req("9", cmd));
            assert_eq!(out, b"\0", "{cmd}");
            assert_eq!(String::from_utf8(err.unwrap()).unwrap(), msg, "{cmd}");
        }
        // of=/dev/stdout is what a caller might spell out; accepted.
        assert_eq!(run(&root, &req("0", "dd if=dist count=1 of=/dev/stdout")).0.len(), 513);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn dd_sizes() {
        assert_eq!(parse_size("512"), Some(512));
        assert_eq!(parse_size("2b"), Some(1024));
        assert_eq!(parse_size("10k"), Some(10240));
        assert_eq!(parse_size("3w"), Some(6));
        for bad in ["", "k", "1m", "-1", "1.5", "x", "99999999999999999999k"] {
            assert_eq!(parse_size(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn echo_pwd_and_the_unknown() {
        let root = tmp_root("misc");
        assert_eq!(run(&root, &req("0", "echo hello   'big world'")).0, b"\0hello big world\n");
        assert_eq!(run(&root, &req("0", "echo -n hi")).0, b"\0hi");
        assert_eq!(run(&root, &req("0", "echo")).0, b"\0\n");
        assert_eq!(run(&root, &req("0", "pwd")).0, b"\0/\n");
        let (out, err) = run(&root, &req("5", "rm -rf /"));
        assert_eq!(out, b"\0");
        assert_eq!(err.unwrap(), b"rm: Command not found.\n");
        let (_, err) = run(&root, &req("5", "cat a | sh"));
        assert_eq!(String::from_utf8(err.unwrap()).unwrap(),
                   "rshd: shell syntax (pipes, redirects, expansion) is not supported.\n");
        let (_, err) = run(&root, &req("5", ""));
        assert!(String::from_utf8(err.unwrap()).unwrap().contains("interactive login is not supported"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_bad_header_is_answered_with_a_one_byte_and_a_message() {
        let root = tmp_root("bad");
        let mut s = RshSession::new(root.clone());
        s.input(b"nonsense\0");
        assert!(s.side_connect().is_none());
        let mut out = Vec::new();
        loop {
            let c = s.pull(64);
            if c.is_empty() {
                break;
            }
            out.extend(c);
        }
        assert_eq!(out, format!("\u{1}{PROTOCOL_FAILURE}\n").into_bytes());
        assert!(s.done());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_request_may_arrive_in_pieces() {
        let root = tmp_root("pieces");
        fs::write(root.join("f"), b"data").unwrap();
        let mut s = RshSession::new(root.clone());
        let full = req("0", "cat f");
        for b in &full[..full.len() - 1] {
            s.input(&[*b]);
            assert!(s.pull(64).is_empty() && !s.done(), "nothing happens before the header is whole");
        }
        s.input(&full[full.len() - 1..]);
        let mut out = Vec::new();
        while !s.done() {
            out.extend(s.pull(64));
        }
        assert_eq!(out, b"\0data");
        // Stdin after the request goes nowhere, and a half-close is fine.
        assert!(s.input(b"more").is_empty());
        s.guest_closed();
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_half_close_mid_command_keeps_the_output_coming() {
        let root = tmp_root("halfclose");
        fs::write(root.join("f"), vec![7u8; 5000]).unwrap();
        let mut s = RshSession::new(root.clone());
        s.input(&req("0", "cat f"));
        assert_eq!(s.pull(100).len(), 100);
        s.guest_closed(); // rsh -n
        let mut n = 100;
        while !s.done() {
            n += s.pull(1000).len();
        }
        assert_eq!(n, 5001);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn hanging_up_before_the_request_ends_the_session() {
        let mut s = RshSession::new(std::env::temp_dir());
        s.input(b"12");
        assert!(!s.done());
        s.guest_closed();
        assert!(s.pull(10).is_empty());
        assert!(s.done());
    }
}
