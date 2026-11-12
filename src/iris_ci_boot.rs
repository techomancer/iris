//! `iris-ci boot-test`: boot an installed IRIX disk twice, once with the
//! console on serial and once on the graphics board, and check that each boot
//! reaches the login.
//!
//! What to seed and what to check is data, in `tests/irix/boot.toml` (see
//! [`iris::guest_steps`]). Each boot is a seeding run, a boot, and a quit, and each
//! is a case of the JUnit report written to `<out>/boot-test.xml`, also after
//! a failure that stops the test early.

use super::*;
use iris::junit::{self, Duration as JDuration, TestCase};
use iris::guest_steps::{self, Boot, Nvram};
use std::collections::HashMap;
use std::path::Path;
use std::process::{Child, Command, Stdio};

#[derive(clap::Args, Debug)]
pub struct Args {
    /// The installed disk, a CHD or a raw image. iris opens it with an
    /// overlay that keeps writes in a file beside it and reuses that file on
    /// the next open, so each run wants a fresh copy.
    disk: PathBuf,
    /// Where the logs, screenshots, NVRAM and boot-test.xml go.
    #[arg(long, default_value = "boot-test")]
    out: PathBuf,
    /// The iris binary. Defaults to the one beside iris-ci.
    #[arg(long)]
    iris: Option<PathBuf>,
    /// The nvram and boot tables, in place of the built-in ones
    /// (tests/irix/boot.toml). A screenshot a boot names is looked for beside
    /// this file first.
    #[arg(long)]
    steps: Option<PathBuf>,
    /// A screenshot to match in place of the one each graphics boot names.
    #[arg(long)]
    golden: Option<PathBuf>,
    /// The machine the disk was installed on, by its `[machine] profile` name.
    /// The disk is a stock install for one machine and boots only there.
    #[arg(long, default_value = "indy_ip24")]
    machine: String,
    /// The PROM image, for a machine with no PROM built in (the O2).
    #[arg(long)]
    prom: Option<PathBuf>,
    /// Seconds each boot may take to reach its login.
    #[arg(long, default_value_t = 900)]
    timeout: u64,
}

/// tests/irix/boot.toml, so the test needs no checkout beside it.
const BOOT_TOML: &str = include_str!("../tests/irix/boot.toml");

/// The screenshots a boot may name, by file name.
const GOLDENS: &[(&str, &[u8])] = &[
    ("indy-irix53.png", include_bytes!("../scripts/boot-goldens/indy-irix53.png")),
];

/// The screenshot a boot names: the file beside the steps file when there is
/// one, else the built-in of that name.
fn golden(name: &Path, base: Option<&Path>) -> std::result::Result<(String, Rgb), String> {
    let label = name.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    if let Some(path) = base.map(|b| b.join(name)).filter(|p| p.is_file()) {
        return Ok((label, read_rgb(&path)?));
    }
    let (_, png) = GOLDENS.iter().find(|(n, _)| *n == name.to_string_lossy())
        .ok_or_else(|| format!("no screenshot {} to match: name a file with --golden", name.display()))?;
    Ok((label, decode_rgb(std::io::Cursor::new(png), &name.display().to_string())?))
}

/// Seconds to wait for the CI socket after starting iris, for the PROM menu
/// of a headless run, and for each Command Monitor prompt.
const SOCKET_WAIT: u32 = 60;
const MENU_WAIT: u64 = 180;
const PROMPT_WAIT: u64 = 30;
/// Seconds iris has to quit when asked before it is killed.
const QUIT_WAIT: u32 = 5;
/// Seconds between drains of the serial console while waiting for its login.
const SERIAL_POLL: std::time::Duration = std::time::Duration::from_secs(5);
/// Seconds the guest has to print a login prompt and to answer `nvram`.
const GUEST_WAIT: u64 = 60;

/// The check under way failed in a way that ends the test.
struct Abort(String);

type Step<T = ()> = std::result::Result<T, Abort>;

pub fn run(args: Args) -> Result<()> {
    let local = |what: &Path, e: std::io::Error| Error::Local(format!("{}: {e}", what.display()));
    let file = match &args.steps {
        Some(p) => guest_steps::load(p),
        None => guest_steps::parse(BOOT_TOML, "built-in boot.toml"),
    }.map_err(Error::Local)?;
    if file.boot.is_empty() {
        return Err(Error::Local("the steps file has no [[boot]] tables".into()));
    }
    let base = args.steps.as_deref().map(|p| p.parent().unwrap_or(Path::new("")));
    // Resolved before anything starts, so a missing screenshot fails at once.
    let mut screens = Vec::new();
    for boot in &file.boot {
        screens.push(match (&args.golden, &boot.screen) {
            (Some(g), Some(_)) => Some((g.display().to_string(), read_rgb(g).map_err(Error::Local)?)),
            (None, Some(s)) => Some(golden(s, base).map_err(Error::Local)?),
            _ => None,
        });
    }
    let abs = |p: &Path| std::path::absolute(p).map_err(|e| local(p, e));
    let machine = iris::config::MachineProfile::from_name(&args.machine).map_err(Error::Local)?;
    let prom = match (&args.prom, machine.has_embedded_prom()) {
        (Some(p), _) => Some(abs(p)?),
        (None, true) => None,
        (None, false) => return Err(Error::Local(format!("{} has no PROM built in: name one with --prom", machine.name()))),
    };
    if let Some(p) = prom.as_ref().filter(|p| !p.is_file()) {
        return Err(Error::Local(format!("{}: no such PROM image", p.display())));
    }
    let disk = abs(&args.disk)?;
    if !disk.is_file() {
        return Err(Error::Local(format!("{}: no such disk image", disk.display())));
    }
    let iris = match &args.iris {
        Some(p) => abs(p)?,
        None => std::env::current_exe().map_err(|e| local(Path::new("iris-ci"), e))?
            .with_file_name(format!("iris{}", std::env::consts::EXE_SUFFIX)),
    };
    if !iris.is_file() {
        return Err(Error::Local(format!("{}: no iris here, name one with --iris", iris.display())));
    }
    let out = abs(&args.out)?;
    std::fs::create_dir_all(&out).map_err(|e| local(&out, e))?;
    for f in ["nvram.bin", "serial.log", "phase-a.log", "boot-test.xml"] {
        let _ = std::fs::remove_file(out.join(f));
    }

    let socket = ci_socket();
    std::fs::write(out.join("boot-test.toml"), config(&disk, &socket, machine.name(), prom.as_deref())).map_err(|e| local(&out, e))?;

    let mut t = Test {
        out: out.clone(),
        iris,
        opts: Opts { socket: PathBuf::from(socket), json: false, quiet: true },
        child: None,
        prompt: monitor_prompt(machine),
        check: String::new(),
        cases: Vec::new(),
        failed: false,
    };
    let aborted = (|| -> Step {
        for (boot, screen) in file.boot.iter().zip(&screens) {
            t.boot(boot, &file.nvram, screen.as_ref(), args.timeout)?;
        }
        Ok(())
    })();
    t.stop_iris();
    let suite = format!("boot.{}", std::env::consts::OS);
    let report = out.join("boot-test.xml");
    if let Err(e) = junit::write(junit::suite(&suite, std::mem::take(&mut t.cases)), &report) {
        // CI annotates from this file, so a run that cannot leave it is not a pass.
        log(&format!("could not write {}: {e}", report.display()));
        t.failed = true;
    }
    verdict(aborted, t.failed)
}

/// How the run ends: a failure that stopped it, or one that let it go on.
/// `failed` is any case filed as a failure, whichever check filed it.
fn verdict(aborted: Step, failed: bool) -> Result<()> {
    match aborted {
        Err(Abort(why)) => Err(Error::Failed(why)),
        Ok(()) if failed => Err(Error::Failed("a boot check failed; see boot-test.xml".into())),
        Ok(()) => Ok(()),
    }
}

/// The Command Monitor's prompt. The O2's PROM prints a single `>`; the
/// others print `>>`. Waiting for `> ` on all of them would also stop at any
/// `> ` inside a listing that precedes the prompt.
fn monitor_prompt(machine: iris::config::MachineProfile) -> &'static str {
    if machine == iris::config::MachineProfile::O2Ip32 { "> " } else { ">> " }
}

/// The CI socket: TCP on Windows, and elsewhere a Unix socket in the host's
/// temporary directory, as iris's own default is. A path under the output
/// directory could pass macOS's 104-byte limit on a Unix socket path.
fn ci_socket() -> String {
    if cfg!(windows) {
        "127.0.0.1:19851".to_string()
    } else {
        std::env::temp_dir().join(format!("iris-boot-{}.sock", std::process::id())).to_string_lossy().into_owned()
    }
}

/// The iris config of every run: the machine and its PROM, the disk on SCSI 1
/// with an overlay, the NVRAM (the PROM's flash on an O2) and the serial mirror
/// in the working directory.
fn config(disk: &Path, socket: &str, machine: &str, prom: Option<&Path>) -> String {
    let mut scsi1 = toml::Table::new();
    // Forward slashes in TOML work on every host, Windows included.
    scsi1.insert("path".into(), disk.to_string_lossy().replace('\\', "/").into());
    scsi1.insert("cdrom".into(), false.into());
    scsi1.insert("overlay".into(), true.into());
    let mut scsi = toml::Table::new();
    scsi.insert("1".into(), scsi1.into());
    let mut m = toml::Table::new();
    m.insert("profile".into(), machine.into());
    let mut t = toml::Table::new();
    t.insert("machine".into(), m.into());
    if let Some(p) = prom {
        t.insert("prom".into(), p.to_string_lossy().replace('\\', "/").into());
    }
    t.insert("no_audio".into(), true.into());
    t.insert("nvram".into(), "nvram.bin".into());
    t.insert("serial_log".into(), "serial.log".into());
    t.insert("ci".into(), true.into());
    t.insert("ci_socket".into(), socket.into());
    t.insert("scsi".into(), scsi.into());
    format!("# Written by iris-ci boot-test.\n{}", toml::to_string(&t).expect("a table of plain values"))
}

fn log(msg: &str) {
    let day = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) % 86400;
    println!("[boot-test {:02}:{:02}:{:02}] {msg}", day / 3600, day / 60 % 60, day % 60);
}

/// The `name=value` lines of `text`, `printenv` or `nvram` output. The first
/// of a name wins.
fn settings(text: &str) -> HashMap<&str, &str> {
    let mut seen = HashMap::new();
    for line in text.lines() {
        if let Some((k, v)) = line.trim().split_once('=') {
            seen.entry(k.trim()).or_insert(v.trim());
        }
    }
    seen
}

/// How each of `want` differs from what `text` lists, empty when all match.
fn unmatched(text: &str, want: &[(&str, &str)]) -> Vec<String> {
    let seen = settings(text);
    want.iter().filter_map(|&(k, v)| match seen.get(k) {
        Some(&got) if got == v => None,
        Some(got) => Some(format!("{k}: seeded {v:?}, read {got:?}")),
        None => Some(format!("{k}: seeded {v:?}, not listed")),
    }).collect()
}

/// What a boot's console was seeded with, and what the variables say.
fn seeded<'a>(vars: &'a [Nvram], console: &'a str, guest_only: bool) -> Vec<(&'a str, &'a str)> {
    vars.iter().filter(|v| v.guest || !guest_only).map(|v| (v.name.as_str(), v.value.as_str()))
        .chain([("console", console)]).collect()
}

struct Test {
    out: PathBuf,
    iris: PathBuf,
    opts: Opts,
    child: Option<Child>,
    /// What the PROM's Command Monitor prints when it wants a line.
    prompt: &'static str,
    /// The case a failure that ends the test is filed under.
    check: String,
    cases: Vec<TestCase>,
    /// Some case was filed as a failure. The exit status follows this, not
    /// what each check happens to return, so a check that only files its
    /// case cannot leave the run looking green.
    failed: bool,
}

impl Test {
    fn pass(&mut self, name: &str, secs: f64, text: &str) {
        let mut c = TestCase::success(name, JDuration::seconds_f64(secs));
        if !text.is_empty() { c.set_system_out(text); }
        self.cases.push(c);
    }

    fn fail(&mut self, name: &str, secs: f64, why: &str) {
        log(why);
        self.failed = true;
        self.cases.push(TestCase::failure(name, JDuration::seconds_f64(secs), "check", why));
    }

    /// End the test, filing `why` under the check that was under way.
    fn abort(&mut self, why: String) -> Abort {
        log(&why);
        let name = self.check.clone();
        self.fail(&name, 0.0, &why);
        Abort(why)
    }

    /// Start iris with its output in `logname`, and wait for its CI socket.
    fn start_iris(&mut self, logname: &str, extra: &[&str]) -> Step {
        let logf = self.out.join(logname);
        let open = || std::fs::File::create(&logf);
        let (stdout, stderr) = match (open(), open()) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => return Err(self.abort(format!("{}: {e}", logf.display()))),
        };
        let child = Command::new(&self.iris).arg("--config").arg("boot-test.toml").args(extra)
            .current_dir(&self.out).stdin(Stdio::null()).stdout(stdout).stderr(stderr).spawn();
        match child {
            Ok(c) => self.child = Some(c),
            Err(e) => return Err(self.abort(format!("{}: {e}", self.iris.display()))),
        }
        for _ in 0..SOCKET_WAIT {
            if send(&self.opts, "ping", json!({})).is_ok() {
                return Ok(());
            }
            if self.exited().is_some() {
                self.dump(&logf);
                return Err(self.abort("iris exited before the CI socket came up".into()));
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        self.dump(&logf);
        Err(self.abort("CI socket never came up".into()))
    }

    fn dump(&self, path: &Path) {
        print!("{}", std::fs::read_to_string(path).unwrap_or_default());
    }

    /// How iris ended if it has, and forget it.
    fn exited(&mut self) -> Option<std::process::ExitStatus> {
        let status = self.child.as_mut()?.try_wait().ok().flatten()?;
        self.child = None;
        Some(status)
    }

    /// Ask iris to quit and kill it if it has not within `QUIT_WAIT` seconds.
    /// Says how it ended: "exit N" when it quit by itself, "killed" otherwise.
    fn stop_iris(&mut self) -> Option<String> {
        self.child.as_ref()?;
        let _ = send(&self.opts, "quit", json!({}));
        let mut how = None;
        for _ in 0..QUIT_WAIT {
            if let Some(s) = self.exited() {
                how = Some(s.code().map_or_else(|| "killed by a signal".to_string(), |c| format!("exit {c}")));
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        if !self.opts.socket.to_string_lossy().contains(':') {
            let _ = std::fs::remove_file(&self.opts.socket);
        }
        Some(how.unwrap_or_else(|| "killed".into()))
    }

    /// Stop iris and check that it quit by itself, status 0. A machine that
    /// will not stop when asked would hang a user's session too.
    fn check_quit(&mut self, name: &str) -> bool {
        let Some(how) = self.stop_iris() else { return true };
        if how == "exit 0" {
            self.pass(name, 0.0, "");
            true
        } else {
            self.fail(name, 0.0, &format!("iris after quit: {how}"));
            false
        }
    }

    /// Wait on the serial console for `pattern`, keeping what arrives in
    /// phase-a.log.
    fn prom_wait(&self, pattern: &str, secs: u64) -> Option<String> {
        let said = send_until(&self.opts, "wait-serial", json!({"pattern": pattern, "timeout_ms": secs * 1000}),
                              read_timeout_for(secs * 1000));
        let ok = said.is_ok();
        let text = said.ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(self.out.join("phase-a.log")) {
            let _ = f.write_all(text.as_bytes());
        }
        ok.then_some(text)
    }

    /// Type a Command Monitor line and wait for the prompt. What the monitor
    /// printed comes back.
    fn monitor(&mut self, line: &str) -> Step<String> {
        let sent = send(&self.opts, "serial-send", json!({"data": format!("{line}\r")}));
        match sent.is_ok().then(|| self.prom_wait(self.prompt, PROMPT_WAIT)).flatten() {
            Some(text) => Ok(text),
            None => {
                self.dump(&self.out.join("phase-a.log"));
                Err(self.abort(format!("phase A: '{line}' got no prompt back")))
            }
        }
    }

    /// A blank NVRAM has no boot path, so run the PROM headless on serial and
    /// set the variables, with `console` saying where IRIX puts its login.
    /// Then read them back from the PROM.
    fn seed(&mut self, vars: &[Nvram], console: &str) -> Step {
        log(&format!("phase A: seeding NVRAM (headless, console={console})"));
        let name = format!("seed NVRAM (console={console})");
        self.check = name.clone();
        let start = Instant::now();
        // A seeded NVRAM has AutoLoad on, and the PROM would boot past its menu.
        let _ = std::fs::remove_file(self.out.join("nvram.bin"));
        self.start_iris(&format!("iris-seed-{console}.log"), &["--headless"])?;
        let _ = send(&self.opts, "start", json!({}));
        if self.prom_wait("Option?", MENU_WAIT).is_none() {
            self.dump(&self.out.join("phase-a.log"));
            return Err(self.abort("phase A: PROM menu never appeared on serial".into()));
        }
        // Enter Command Monitor.
        let _ = send(&self.opts, "serial-send", json!({"data": "5\r"}));
        if self.prom_wait(self.prompt, PROMPT_WAIT).is_none() {
            self.dump(&self.out.join("phase-a.log"));
            return Err(self.abort("phase A: no Command Monitor prompt".into()));
        }
        for (k, v) in seeded(vars, console, false) {
            self.monitor(&format!("setenv -f {k} {v}"))?;
        }
        let printed = self.monitor("printenv")?;
        if send(&self.opts, "rtc-save", json!({})).is_err() {
            return Err(self.abort("phase A: rtc-save failed".into()));
        }
        self.stop_iris();
        self.pass(&name, start.elapsed().as_secs_f64(), "");
        log("phase A: done");

        self.check_printenv(&printed, vars, console);
        Ok(())
    }

    /// Compare what the PROM's printenv listed with what was seeded.
    fn check_printenv(&mut self, printed: &str, vars: &[Nvram], console: &str) {
        let name = format!("NVRAM reads back as seeded (console={console})");
        let bad = unmatched(printed, &seeded(vars, console, false));
        if bad.is_empty() {
            self.pass(&name, 0.0, "");
        } else {
            self.fail(&name, 0.0, &format!("printenv: {}", bad.join(", ")));
        }
    }

    /// One boot: seed, start, check the login, check what the guest booted
    /// with, quit. Failures are filed as cases; an `Abort` is a failure that
    /// leaves nothing to check.
    fn boot(&mut self, boot: &Boot, vars: &[Nvram], screen: Option<&(String, Rgb)>, timeout: u64) -> Step {
        let console = boot.console.as_str();
        self.seed(vars, console)?;
        self.check = boot.name.clone();
        self.start_iris(&format!("iris-{console}.log"), &[])?;
        log(&format!("booting with console={console}; deadline {timeout}s"));
        let _ = send(&self.opts, "start", json!({}));
        let mut ok = true;
        if let Some(pattern) = &boot.serial {
            ok &= self.await_serial(boot, pattern, timeout);
        }
        if let Some(golden) = screen {
            ok &= self.await_screen(boot, golden, timeout);
        }
        if self.child.is_none() {
            return Ok(());
        }
        self.guest_readback(vars, console, ok);
        self.check_quit(&boot.quit);
        Ok(())
    }

    /// Drain the serial console until `pattern` has come out of it, iris has
    /// exited or `timeout` seconds have passed. The transcript is kept in
    /// transcript-<console>.log.
    fn await_serial(&mut self, boot: &Boot, pattern: &str, timeout: u64) -> bool {
        let start = Instant::now();
        let path = self.out.join(format!("transcript-{}.log", boot.console));
        let mut transcript = String::new();
        let ok = loop {
            let elapsed = start.elapsed().as_secs();
            if let Ok(v) = send(&self.opts, "serial-read", json!({})) {
                transcript.push_str(&v.as_str().unwrap_or("").replace("\r\n", "\n").replace('\r', "\n"));
            }
            if self.exited().is_some() {
                self.fail(&boot.name, elapsed as f64, &format!("iris exited after {elapsed}s"));
                break false;
            }
            if transcript.contains(pattern) {
                log(&format!("{pattern:?} after {elapsed}s"));
                self.pass(&boot.name, elapsed as f64, "");
                break true;
            }
            if elapsed >= timeout {
                self.fail(&boot.name, elapsed as f64, &format!("no {pattern:?} within {timeout}s"));
                break false;
            }
            std::thread::sleep(SERIAL_POLL);
        };
        let _ = std::fs::write(&path, &transcript);
        log("last serial output:");
        for l in transcript.lines().rev().take(40).collect::<Vec<_>>().into_iter().rev() {
            println!("{l}");
        }
        ok
    }

    /// Wait for the screen to match `golden`. On a mismatch gfx-screen.png and
    /// gfx-diff.png are left in the output directory.
    fn await_screen(&mut self, boot: &Boot, (name, golden): &(String, Rgb), timeout: u64) -> bool {
        let start = Instant::now();
        let loud = Opts { socket: self.opts.socket.clone(), json: false, quiet: false };
        let said = screen_wait(&loud, name, golden, timeout, &self.out.join("gfx-screen.png"),
                               &self.out.join("gfx-diff.png"), SCREEN_THRESHOLD, SCREEN_MAX_PIXELS);
        let secs = start.elapsed().as_secs_f64();
        match said {
            Ok(msg) => {
                println!("{msg}");
                log(&format!("screen matches {name}"));
                self.pass(&boot.name, secs, &msg);
                true
            }
            Err(e) => {
                let why = match e {
                    Error::Timeout(m) | Error::Iris(m) | Error::Local(m) => m,
                    other => format!("{other:?}"),
                };
                println!("{why}");
                self.fail(&boot.name, secs, &why);
                false
            }
        }
    }

    /// Log in on serial and compare the guest's own `nvram` with what was
    /// seeded, which shows what the kernel booted with. IRIX runs a getty on
    /// serial whichever console the PROM chose, and a CR makes it print its
    /// login again, so this works after a graphics boot too.
    fn guest_readback(&mut self, vars: &[Nvram], console: &str, login_ok: bool) {
        let name = format!("guest booted with the seeded NVRAM (console={console})");
        if !login_ok {
            self.cases.push(TestCase::skipped_with_cause(&name, "skipped", "the login check failed"));
            return;
        }
        let start = Instant::now();
        let asked = (|| -> Result<String> {
            send(&self.opts, "serial-send", json!({"data": "\r"}))?;
            send_until(&self.opts, "wait-serial", json!({"pattern": "login:", "timeout_ms": GUEST_WAIT * 1000}),
                       read_timeout_for(GUEST_WAIT * 1000))?;
            cmd_login(&self.opts, "root", None)?;
            Ok(run_capture(&self.opts, &guest_steps::wrap("nvram"), "csh", GUEST_WAIT * 1000)?.0)
        })();
        let secs = start.elapsed().as_secs_f64();
        match asked {
            Err(e) => self.fail(&name, secs, &format!("could not read nvram in the guest: {e:?}")),
            Ok(text) => {
                let bad = unmatched(&text, &seeded(vars, console, true));
                if bad.is_empty() {
                    self.pass(&name, secs, "");
                } else {
                    self.fail(&name, secs, &format!("nvram: {}", bad.join(", ")));
                }
            }
        }
    }
}

impl Drop for Test {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str, value: &str, guest: bool) -> Nvram {
        Nvram { name: name.into(), value: value.into(), guest }
    }

    const PRINTENV: &str = "printenv\r\nTimeZone=PST8PDT\r\n\r\nconsole=g\r\n\r\nAutoLoad=Yes\r\n\r\n>> ";

    #[test]
    fn a_setting_must_be_listed_with_the_seeded_value() {
        assert!(unmatched(PRINTENV, &[("console", "g"), ("AutoLoad", "Yes")]).is_empty());
        assert_eq!(unmatched(PRINTENV, &[("console", "d")]), ["console: seeded \"d\", read \"g\""]);
        assert_eq!(unmatched(PRINTENV, &[("OSLoadFilename", "/unix")]),
                   ["OSLoadFilename: seeded \"/unix\", not listed"]);
    }

    fn idle_test() -> Test {
        Test {
            out: PathBuf::new(),
            iris: PathBuf::new(),
            opts: Opts { socket: PathBuf::new(), json: false, quiet: true },
            child: None,
            prompt: ">> ",
            check: String::new(),
            cases: Vec::new(),
            failed: false,
        }
    }

    /// The run's exit status as `run` computes it.
    fn exit_status(t: &Test) -> i32 {
        match verdict(Ok(()), t.failed) {
            Ok(()) => 0,
            Err(_) => 6,
        }
    }

    #[test]
    fn a_wrong_nvram_value_read_back_by_the_prom_fails_the_run() {
        let mut t = idle_test();
        let vars = [var("OSLoadFilename", "/unix", false)];
        t.check_printenv(PRINTENV, &vars, "g");
        assert_eq!(exit_status(&t), 6);
    }

    #[test]
    fn a_run_whose_checks_all_passed_exits_clean() {
        let mut t = idle_test();
        let vars = [var("console", "g", false)];
        t.check_printenv(PRINTENV, &vars, "g");
        assert_eq!(exit_status(&t), 0);
    }

    #[test]
    fn the_first_listing_of_a_name_wins() {
        assert_eq!(settings("a=1\r\na=2\r\n")["a"], "1");
        // A typed command can hold an equals sign; it is not a setting.
        assert!(!settings("sh -c 'PATH=/bin; nvram'\r\n").contains_key("PATH"));
    }

    #[test]
    fn the_guest_is_asked_only_about_the_variables_it_lists() {
        let vars = [var("SystemPartition", "p8", true), var("OSLoadFilename", "/unix", false)];
        let names = |g| seeded(&vars, "d", g).iter().map(|v| v.0).collect::<Vec<_>>();
        assert_eq!(names(false), ["SystemPartition", "OSLoadFilename", "console"]);
        assert_eq!(names(true), ["SystemPartition", "console"]);
    }

    #[cfg(unix)]
    #[test]
    fn the_socket_is_in_the_temporary_directory_and_short_enough_for_macos() {
        let sock = ci_socket();
        assert!(Path::new(&sock).starts_with(std::env::temp_dir()), "{sock}");
        assert!(sock.contains(&std::process::id().to_string()));
        // macOS's own temporary directory, about 50 bytes, leaves room.
        assert!(sock.len() - std::env::temp_dir().as_os_str().len() < 40);
    }

    #[test]
    fn the_config_names_the_disk_whatever_its_path_holds() {
        let disk = Path::new("C:\\a \"b\"\\irix.chd");
        let t: toml::Table = toml::from_str(&config(disk, "127.0.0.1:19851", "indy_ip24", None)).unwrap();
        let scsi1 = &t["scsi"]["1"];
        assert_eq!(scsi1["path"].as_str(), Some("C:/a \"b\"/irix.chd"));
        assert_eq!((scsi1["overlay"].as_bool(), t["ci"].as_bool()), (Some(true), Some(true)));
        assert_eq!(t["ci_socket"].as_str(), Some("127.0.0.1:19851"));
        assert_eq!(t["machine"]["profile"].as_str(), Some("indy_ip24"));
        assert!(!t.contains_key("prom"));
    }

    #[test]
    fn the_o2_prom_prompts_with_one_angle_bracket() {
        use iris::config::MachineProfile::*;
        assert_eq!((monitor_prompt(O2Ip32), monitor_prompt(IndyIp24)), ("> ", ">> "));
    }

    #[test]
    fn an_o2_config_names_the_machine_and_its_prom() {
        let t: toml::Table = toml::from_str(&config(Path::new("d.chd"), "s", "o2_ip32", Some(Path::new(r"C:\p\ip32prom.bin")))).unwrap();
        assert_eq!(t["machine"]["profile"].as_str(), Some("o2_ip32"));
        assert_eq!(t["prom"].as_str(), Some("C:/p/ip32prom.bin"));
    }

    /// The built-in file: the same seeded variables as before, a serial and a
    /// graphics boot, and a golden that decodes. It is the file in the tree.
    #[test]
    fn the_shipped_boots_are_the_serial_and_the_graphics_login() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/irix/boot.toml");
        assert_eq!(BOOT_TOML, std::fs::read_to_string(&path).unwrap());
        let file = guest_steps::parse(BOOT_TOML, "boot.toml").unwrap();
        let seeded: Vec<_> = file.nvram.iter().map(|v| (v.name.as_str(), v.value.as_str())).collect();
        assert_eq!(seeded, [
            ("SystemPartition", "scsi(0)disk(1)rdisk(0)partition(8)"),
            ("OSLoadPartition", "scsi(0)disk(1)rdisk(0)partition(0)"),
            ("OSLoadFilename", "/unix"),
            ("AutoLoad", "Yes"),
        ]);
        let guest: Vec<_> = file.nvram.iter().filter(|v| v.guest).map(|v| v.name.as_str()).collect();
        assert_eq!(guest, ["SystemPartition", "OSLoadPartition", "OSLoadFilename"]);
        let boots: Vec<_> = file.boot.iter().map(|b| (b.console.as_str(), b.serial.is_some(), b.screen.is_some())).collect();
        assert_eq!(boots, [("d", true, false), ("g", false, true)]);
        let (label, img) = golden(file.boot[1].screen.as_ref().unwrap(), None).unwrap();
        assert_eq!((label.as_str(), img.width > 0), ("indy-irix53.png", true));
    }

    /// The O2's file seeds the same variables as the Indy's and checks only
    /// the serial login: no golden screenshot of that disk exists.
    #[test]
    fn the_o2_boot_is_the_serial_login_with_the_indys_seeding() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/irix");
        let o2 = guest_steps::load(&dir.join("boot-irix6530-o2.toml")).unwrap();
        let indy = guest_steps::parse(BOOT_TOML, "boot.toml").unwrap();
        let vars = |f: &guest_steps::StepFile| f.nvram.iter().map(|v| (v.name.clone(), v.value.clone(), v.guest)).collect::<Vec<_>>();
        assert_eq!(vars(&o2), vars(&indy));
        let boots: Vec<_> = o2.boot.iter().map(|b| (b.console.as_str(), b.serial.as_deref(), b.screen.is_some())).collect();
        assert_eq!(boots, [("d", Some("login:"), false)]);
    }

    #[test]
    fn a_screenshot_is_found_beside_the_steps_file_or_built_in_and_not_otherwise() {
        let dir = std::env::temp_dir().join(format!("iris-ci-golden-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("indy-irix53.png"), b"not a png").unwrap();
        let beside = golden(Path::new("indy-irix53.png"), Some(&dir));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(beside.is_err(), "the file beside the steps wins over the built-in one");
        assert!(golden(Path::new("indy-irix53.png"), Some(&dir)).map(|g| g.0).is_ok(), "gone, the built-in one serves");
        assert!(golden(Path::new("other.png"), None).err().unwrap().contains("--golden"));
    }
}
