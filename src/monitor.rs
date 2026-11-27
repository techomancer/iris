use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use parking_lot::Mutex;
use std::thread;
use std::net::{TcpListener, TcpStream};
use std::io::{Write, BufReader, BufRead, BufWriter, Read};
use crate::traits::Device;

/// The monitor's TCP port on 127.0.0.1 when the config names none.
pub const DEFAULT_PORT: u16 = 8888;

/// The longest command line the monitor accepts, newline included. Real
/// commands are a few dozen bytes; this leaves room for a pasted address
/// list and bounds what a client with no newline can make us hold.
const MAX_LINE: usize = 4096;

pub struct Monitor {
    devices: Arc<Mutex<Vec<Arc<dyn Device>>>>,
    /// Set by `shutdown`; the accept loop exits (dropping its listener, which
    /// frees the port) the next time it wakes.
    shutdown: Arc<AtomicBool>,
    /// The address actually bound, once the server thread has bound it —
    /// `shutdown` connects here to wake the blocking `accept`.
    bound: Mutex<Option<std::net::SocketAddr>>,
}

impl Monitor {
    pub fn new() -> Self {
        Self {
            devices: Arc::new(Mutex::new(Vec::new())),
            shutdown: Arc::new(AtomicBool::new(false)),
            bound: Mutex::new(None),
        }
    }

    /// Stop serving: the listener closes (freeing the port for the next
    /// Machine's monitor) and the device list is emptied, releasing this
    /// monitor's references to the Machine's devices. A client that is already
    /// connected stays connected but finds no commands.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.devices.lock().clear();
        if let Some(addr) = self.bound.lock().take() {
            let _ = TcpStream::connect(addr); // wake the blocking accept()
        }
    }

    /// The address the server bound, once it has.
    pub fn bound_addr(&self) -> Option<std::net::SocketAddr> {
        *self.bound.lock()
    }

    /// Every command name a client can type, for Tab completion: the
    /// devices' own and the three `handle_client` answers itself.
    pub fn command_names(&self) -> Vec<String> {
        let devs = self.devices.lock();
        let own = devs.iter().flat_map(|d| d.register_commands()).map(|(name, _)| name);
        let mut names: Vec<String> = own.chain(["help", "quit", "exit", "watch", "ports", "serial"].map(String::from)).collect();
        names.sort();
        names.dedup();
        names
    }

    pub fn register_device(&self, device: Arc<dyn Device>) {
        self.devices.lock().push(device);
    }

    /// What `cmd args` prints, from every device that offers it.
    #[cfg(test)]
    pub(crate) fn run(&self, cmd: &str, args: &[&str]) -> String {
        struct Capture(Arc<Mutex<Vec<u8>>>);
        impl Write for Capture {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let out = Arc::new(Mutex::new(Vec::new()));
        let devs: Vec<_> = self.devices.lock().iter().filter(|d| d.register_commands().iter().any(|(c, _)| c == cmd)).cloned().collect();
        for d in devs {
            if let Err(e) = d.execute_command(cmd, args, Box::new(Capture(out.clone()))) {
                out.lock().extend_from_slice(format!("Error: {e}\n").as_bytes());
            }
        }
        let text = String::from_utf8_lossy(&out.lock()).into_owned();
        text
    }

    /// What `audio status` prints.
    #[cfg(test)]
    pub fn audio_status(&self) -> String {
        audio_status(&self.devices.lock())
    }

    /// Frames every sound card has moved (see [`Device::audio_frames`]), or
    /// `None` on a machine with no sound card.
    pub fn audio_frames(&self) -> Option<u64> {
        self.devices.lock().iter().filter_map(|d| d.audio_frames()).reduce(|a, b| a + b)
    }

    /// Bind `addr` and serve it on a new thread. The bind happens before
    /// this returns, so `bound_addr` answers at once whether this monitor
    /// has its port.
    pub fn start_server(self: Arc<Self>, addr: String) {
        // Fail soft (rather than panic-aborting the whole process) if the
        // port can't be bound — because another iris process is using it
        // (give each its own `monitor_port`), or a previous machine
        // instance's monitor thread from the same GUI session is still
        // holding it. Dropping a Machine shuts its monitor down and frees
        // the port, but a Machine that is merely stopped keeps it. The
        // machine still boots; the monitor console is just unavailable.
        let listener = match TcpListener::bind(&addr) {
            Ok(l) => l,
            Err(e) => {
                log::warn!("monitor disabled: failed to bind {addr}: {e}");
                eprintln!("iris: no monitor: can't bind {addr} ({e}); is another iris using it? Set monitor_port (--monitor-port) to run both.");
                return;
            }
        };
        // Publish the bound address before checking the flag: a concurrent
        // shutdown() either sees the address (and wakes us) or we see the
        // flag here.
        let bound = listener.local_addr().ok();
        *self.bound.lock() = bound;
        if self.shutdown.load(Ordering::SeqCst) {
            return;
        }
        let port = bound.map(|a| crate::ports::register("tcp", a, "monitor"));
        thread::spawn(move || {
            let _port = port;
            // The address as bound: with port 0 the OS chose the port, and
            // this line is how a caller finds it.
            println!("Monitor listening on {}", bound.map_or(addr, |a| a.to_string()));
            for stream in listener.incoming() {
                if self.shutdown.load(Ordering::SeqCst) {
                    break; // drops the listener: the port is free again
                }
                match stream {
                    Ok(stream) => {
                        let devices = self.devices.clone();
                        thread::spawn(move || {
                            handle_client(stream, devices);
                        });
                    }
                    Err(e) => eprintln!("Monitor accept error: {}", e),
                }
            }
        });
    }
}

/// Every sound card's report, or why there is none.
fn audio_status(devices: &[Arc<dyn Device>]) -> String {
    let reports: String = devices.iter().filter_map(|d| d.audio_report()).collect();
    if reports.is_empty() { "No audio device.\n".to_string() } else { reports }
}

fn handle_client(stream: TcpStream, devices: Arc<Mutex<Vec<Arc<dyn Device>>>>) {
    // Register this connection with DevLog so log output is broadcast here.
    // Wrap in CrlfWriter so bare \n from log messages renders as CRLF on the
    // telnet client side.
    let sink = crate::devlog::DEVLOG.get().map(|dl| {
        let w: crate::devlog::DevLogWriter = Arc::new(Mutex::new(
            BufWriter::new(crate::net::telnet::CrlfWriter::new(stream.try_clone().unwrap()))
        ));
        dl.add_sink(w.clone());
        (dl, w)
    });

    // Monitor is line-oriented: stay in NVT (passive telnet — strip inbound
    // IAC, decline whatever the client offers, but don't initiate). The
    // client keeps its local echo and line editor. Outbound goes through
    // CrlfWriter so bare \n becomes CRLF on the wire as NVT requires.
    let mut reader = BufReader::new(crate::net::telnet::TelnetReader::new_passive(stream.try_clone().unwrap()));
    let mut writer = BufWriter::new(crate::net::telnet::CrlfWriter::new(stream.try_clone().unwrap()));
    let mut bytes = Vec::new();
    
    {
        let logo = include_str!("ascii-art.txt");
        for line in logo.lines() {
            let _ = write!(writer, "{}\r\n", line);
        }
    }
    let _ = write!(writer, "\r\n{} Monitor\r\n> ", crate::machine::emulator_name());
    let _ = writer.flush();
    
    loop {
        bytes.clear();
        match (&mut reader).take(MAX_LINE as u64).read_until(b'\n', &mut bytes) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if bytes.len() == MAX_LINE && bytes.last() != Some(&b'\n') {
            // Drop the rest of the line, then say so and carry on.
            loop {
                let Ok(buf) = reader.fill_buf() else { return };
                let n = buf.len();
                if n == 0 {
                    return;
                }
                let end = buf.iter().position(|&b| b == b'\n');
                reader.consume(end.map_or(n, |i| i + 1));
                if end.is_some() {
                    break;
                }
            }
            let _ = write!(writer, "Error: line longer than {MAX_LINE} bytes\n> ");
            let _ = writer.flush();
            continue;
        }
        // A client that is not UTF-8 (a Latin-1 terminal) still gets an
        // answer: the bad bytes show up as U+FFFD in "Unknown command".
        let line = String::from_utf8_lossy(&bytes);
        let trimmed = line.trim();
        if trimmed.is_empty() {
            let _ = write!(writer, "> ");
            let _ = writer.flush();
            continue;
        }
        
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        let cmd = parts[0];
        let args = &parts[1..];
        
        if cmd == "quit" || cmd == "exit" {
            break;
        }

        if cmd == "watch" {
            match watch(&mut reader, &mut writer, &devices, args) {
                Ok(()) => {}
                Err(WatchEnd::HungUp) => break,
                Err(WatchEnd::Usage(e)) => { let _ = write!(writer, "Error: {e}\n"); }
            }
            let _ = write!(writer, "> ");
            let _ = writer.flush();
            continue;
        }
        
        let mut targets: Vec<Arc<dyn Device>> = Vec::new();
        let mut is_help = false;
        let audio = cmd == "audio" && (args.is_empty() || args == ["status"]);
        if cmd == "ports" {
            let _ = write!(writer, "{}> ", crate::ports::report());
            let _ = writer.flush();
            continue;
        }
        // `serial` and `serial map`: every serial port and what it is
        // connected to. Other `serial` subcommands (the Indy's `serial
        // status`) go to the devices that offer them.
        if cmd == "serial" && (args.is_empty() || args == ["map"]) {
            let _ = write!(writer, "{}> ", crate::dev::serial_map::report());
            let _ = writer.flush();
            continue;
        }
        
        if cmd == "help" {
            is_help = true;
        }

        {
            let devs = devices.lock();
            if is_help {
                let cmds: Vec<_> = devs.iter().flat_map(|d| d.register_commands()).collect();
                let _ = writeln!(writer, "{:12} - {}", "watch", WATCH_HELP);
                let _ = writeln!(writer, "{:12} - {}", "ports", "ports: the host sockets iris listens on (monitor, telnet and other port forwards, VNC, GDB, CI, serial and virtio sockets)");
                let _ = writeln!(writer, "{:12} - {}", "serial", "serial [map]: each serial port, its name (serialN), the console, and what it is connected to (sockets and clients, CI, log, a host serial port and its line); serial status: the SCC's registers (Indy, Indigo2)");
                if !cmds.iter().any(|(c, _)| c == "audio") {
                    let _ = writeln!(writer, "{:12} - {}", "audio", "audio status: each sound card's frames moved and skipped, and its host stream");
                }
                // Several devices may offer one command (`pci`, one per host): list it once.
                let mut seen = std::collections::HashSet::new();
                seen.insert("serial".to_string());
                for (c, h) in cmds {
                    if seen.insert(c.clone()) {
                        let _ = writeln!(writer, "{:12} - {}", c, h);
                    }
                }
            } else if !audio {
                for dev in devs.iter() {
                    let cmds = dev.register_commands();
                    if cmds.iter().any(|(c, _)| c == cmd) {
                        targets.push(dev.clone());
                    }
                }
            }
        }
        
        if audio {
            let text = audio_status(&devices.lock());
            let _ = write!(writer, "{}", text);
        } else if !is_help {
            // Every device that offers the command runs it, in registration order.
            if !targets.is_empty() {
                for dev in targets {
                    let cmd_writer = Box::new(BufWriter::new(crate::net::telnet::CrlfWriter::new(stream.try_clone().unwrap())));
                    match dev.execute_command(cmd, args, cmd_writer) {
                        Ok(_) => {
                        }
                        Err(e) => {
                            let _ = write!(writer, "Error: {}\n", e);
                        }
                    }
                }
            } else {
            let _ = writeln!(writer, "Unknown command. Type 'help' for list.");
            }
        }
        
        let _ = write!(writer, "> ");
        let _ = writer.flush();
    }
    if let Some((dl, w)) = sink {
        dl.remove_sink(&w);
    }
    // Hang up. Closing our handles is not enough: a device may keep the
    // writer a command was given, whose handle keeps the connection open,
    // and the console client waits for it to close before it exits.
    let _ = stream.shutdown(std::net::Shutdown::Both);
}

const WATCH_HELP: &str = "watch [-n SECONDS] <command> [args]: run a command again every second (or SECONDS), its answer redrawn on one line, until Enter";

enum WatchEnd {
    /// The client went away.
    HungUp,
    Usage(String),
}

/// A writer into a shared buffer: a command's answer, for `watch`.
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

/// `watch`: run a command over and over, redrawing its answer on one line,
/// until the client sends a line (Enter).
fn watch(
    reader: &mut BufReader<crate::net::telnet::TelnetReader<TcpStream>>,
    writer: &mut impl Write,
    devices: &Mutex<Vec<Arc<dyn Device>>>,
    args: &[&str],
) -> Result<(), WatchEnd> {
    let (every, args) = match args {
        ["-n", secs, rest @ ..] => {
            let s: f64 = secs.parse().map_err(|_| WatchEnd::Usage(format!("watch: bad interval {secs}")))?;
            (std::time::Duration::from_secs_f64(s.clamp(0.1, 3600.0)), rest)
        }
        _ => (std::time::Duration::from_secs(1), args),
    };
    let Some((&cmd, rest)) = args.split_first() else {
        return Err(WatchEnd::Usage(WATCH_HELP.into()));
    };
    if cmd == "watch" || cmd == "quit" || cmd == "exit" {
        return Err(WatchEnd::Usage(format!("watch: cannot watch {cmd}")));
    }
    let dev = devices.lock().iter()
        .find(|d| d.register_commands().iter().any(|(c, _)| c == cmd))
        .cloned()
        .ok_or_else(|| WatchEnd::Usage(format!("watch: unknown command {cmd}")))?;
    let _ = write!(writer, "(Enter stops)\n");
    let result = loop {
        let out = Arc::new(Mutex::new(Vec::new()));
        let text = match dev.execute_command(cmd, rest, Box::new(Capture(out.clone()))) {
            Ok(()) => {
                let raw = String::from_utf8_lossy(&out.lock()).into_owned();
                raw.lines().map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join("  ")
            }
            Err(e) => format!("Error: {e}"),
        };
        // Back to the start of the line, then erase what the last answer
        // left beyond this one.
        let _ = write!(writer, "\r{text}\x1b[K");
        if writer.flush().is_err() {
            break Err(WatchEnd::HungUp);
        }
        // Set it on the handle the reader uses: a timeout set on another
        // clone of the socket does not apply on Windows.
        let _ = reader.get_ref().get_ref().set_read_timeout(Some(every));
        match reader.fill_buf() {
            Ok([]) => break Err(WatchEnd::HungUp),
            Ok(buf) => {
                // Eat the line that stopped us.
                let n = buf.iter().position(|&b| b == b'\n').map_or(buf.len(), |i| i + 1);
                reader.consume(n);
                break Ok(());
            }
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
            Err(_) => break Err(WatchEnd::HungUp),
        }
    };
    let _ = reader.get_ref().get_ref().set_read_timeout(None);
    let _ = write!(writer, "\n");
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn wait_for<T>(what: &str, mut f: impl FnMut() -> Option<T>) -> T {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(v) = f() { return v; }
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(5));
        }
    }

    /// A dropped Machine's monitor must give the port back, or the next Machine
    /// in the same process runs without a monitor while the old one keeps
    /// answering for a Machine that no longer exists.
    #[test]
    fn shutdown_releases_the_port() {
        let monitor = Arc::new(Monitor::new());
        monitor.clone().start_server("127.0.0.1:0".to_string());
        let addr = wait_for("bind", || monitor.bound_addr());
        monitor.shutdown();
        wait_for("port release", || TcpListener::bind(addr).ok());
    }

    struct Ping;

    impl Device for Ping {
        fn step(&self, _cycles: u64) {}
        fn stop(&self) {}
        fn start(&self) {}
        fn is_running(&self) -> bool { false }
        fn register_commands(&self) -> Vec<(String, String)> {
            vec![("ping".to_string(), "ping status".to_string())]
        }
    }

    #[test]
    fn command_names_lists_the_devices_commands_and_the_built_ins() {
        let monitor = Monitor::new();
        monitor.register_device(Arc::new(Ping));
        let names = monitor.command_names();
        for want in ["ping", "help", "quit"] {
            assert!(names.iter().any(|n| n == want), "{want} missing from {names:?}");
        }
    }

    /// A command two devices both offer runs on each, and `help` lists it once.
    #[test]
    fn a_shared_command_runs_on_every_device_that_offers_it() {
        struct Say(&'static str);
        impl Device for Say {
            fn step(&self, _: u64) {}
            fn stop(&self) {}
            fn start(&self) {}
            fn is_running(&self) -> bool { true }
            fn register_commands(&self) -> Vec<(String, String)> { vec![("say".into(), "say".into())] }
            fn execute_command(&self, _: &str, _: &[&str], mut w: Box<dyn Write + Send>) -> Result<(), String> {
                writeln!(w, "{}", self.0).map_err(|e| e.to_string())
            }
        }
        let both = || vec![Arc::new(Say("one")) as Arc<dyn Device>, Arc::new(Say("two"))];
        let out = converse_with(both(), b"say\nhelp\n");
        assert!(out.find("one").unwrap() < out.find("two").unwrap(), "{out}");
        assert_eq!(out.lines().filter(|l| l.trim_start_matches("> ").starts_with("say ")).count(), 1, "{out}");
    }

    /// A port already in use (another iris's monitor) leaves this monitor
    /// unbound, and says so as soon as `start_server` returns — the console
    /// client must not go looking for a monitor on that port.
    #[test]
    fn taken_port_leaves_the_monitor_unbound() {
        let other = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = other.local_addr().unwrap();
        let monitor = Arc::new(Monitor::new());
        monitor.clone().start_server(addr.to_string());
        assert_eq!(monitor.bound_addr(), None);
    }

    /// A stub sound card; with `own_command` it registers `audio` itself.
    struct Card(&'static str, bool);
    impl Device for Card {
        fn step(&self, _: u64) {}
        fn stop(&self) {}
        fn start(&self) {}
        fn is_running(&self) -> bool { true }
        fn audio_report(&self) -> Option<String> { Some(format!("audio: {}\n", self.0)) }
        fn audio_frames(&self) -> Option<u64> { Some(self.0.len() as u64) }
        fn register_commands(&self) -> Vec<(String, String)> {
            if self.1 { vec![("audio".into(), "own".into())] } else { vec![] }
        }
    }

    /// `help` lists `audio` once, even where a card registers its own.
    #[test]
    fn help_lists_audio_once() {
        let lines = |cards: Vec<Card>| {
            let out = converse_with(cards.into_iter().map(|c| Arc::new(c) as Arc<dyn Device>).collect(), b"help\n");
            out.lines().filter(|l| l.trim_start_matches("> ").starts_with("audio ")).count()
        };
        assert_eq!(lines(vec![]), 1);
        assert_eq!(lines(vec![Card("a", true)]), 1, "the card's own line, not a second");
    }

    /// `audio status` lists every card registered, several at once, and on a
    /// machine with none says so rather than calling the command unknown.
    #[test]
    fn audio_status_lists_every_sound_card() {
        for line in [&b"audio status\n"[..], b"audio\n"] {
            let out = converse(line);
            assert!(out.contains("No audio device"), "{out}");
            assert!(!out.contains("Unknown command"), "{out}");
        }
        let monitor = Monitor::new();
        monitor.register_device(Arc::new(Card("one", false)));
        monitor.register_device(Arc::new(Card("two", false)));
        assert_eq!(monitor.audio_status(), "audio: one\naudio: two\n");
        // The activity count sums the cards, and is absent with none.
        assert_eq!(monitor.audio_frames(), Some(6));
        assert_eq!(Monitor::new().audio_frames(), None);
    }

    /// `quit` ends the connection even though the machine's log sink keeps a
    /// handle on the socket, so the terminal console sees the end and exits.
    #[test]
    fn quit_hangs_up_despite_the_log_sink() {
        use std::io::{Read, Write};
        crate::devlog::init_devlog();
        let monitor = Arc::new(Monitor::new());
        monitor.clone().start_server("127.0.0.1:0".to_string());
        let addr = wait_for("bind", || monitor.bound_addr());
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        s.write_all(b"quit\n").unwrap();
        let mut sink = Vec::new();
        s.read_to_end(&mut sink).expect("the monitor closed the connection");
        monitor.shutdown();
    }

    /// Send `input` to a fresh monitor, then `quit`, and return everything it
    /// said before hanging up.
    fn converse(input: &[u8]) -> String {
        converse_with(Vec::new(), input)
    }

    fn converse_with(devices: Vec<Arc<dyn Device>>, input: &[u8]) -> String {
        use std::io::{Read, Write};
        crate::devlog::init_devlog();
        let monitor = Arc::new(Monitor::new());
        for d in devices {
            monitor.register_device(d);
        }
        monitor.clone().start_server("127.0.0.1:0".to_string());
        let addr = wait_for("bind", || monitor.bound_addr());
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        s.write_all(input).unwrap();
        s.write_all(b"quit\n").unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).expect("the monitor closed the connection");
        monitor.shutdown();
        String::from_utf8_lossy(&out).into_owned()
    }

    /// A byte that is not UTF-8 is a bad command, not the end of the
    /// session: the next line is still read.
    #[test]
    fn a_non_utf8_line_does_not_hang_up() {
        let out = converse(b"caf\xe9\nhelp\nnosuch\n");
        assert_eq!(out.matches("Unknown command").count(), 2, "{out}");
    }

    /// A line past the cap is refused and the connection survives, with the
    /// following line answered.
    #[test]
    fn an_overlong_line_is_refused_and_the_session_continues() {
        let mut input = vec![b'a'; 3 * MAX_LINE];
        input.extend_from_slice(b"\nnosuch\n");
        let out = converse(&input);
        assert!(out.contains("line longer than"), "{out}");
        assert_eq!(out.matches("Unknown command").count(), 1, "{out}");
    }

    #[test]
    fn shutdown_before_bind_does_not_serve() {
        let monitor = Arc::new(Monitor::new());
        monitor.shutdown();
        monitor.clone().start_server("127.0.0.1:0".to_string());
        let addr = wait_for("bind", || monitor.bound_addr());
        wait_for("port release", || TcpListener::bind(addr).ok());
    }

    /// Counts its runs: what `watch` repeats.
    struct Counter(std::sync::atomic::AtomicU32);

    impl Device for Counter {
        fn step(&self, _cycles: u64) {}
        fn stop(&self) {}
        fn start(&self) {}
        fn is_running(&self) -> bool { false }
        fn register_commands(&self) -> Vec<(String, String)> {
            vec![("count".to_string(), "count".to_string())]
        }
        fn execute_command(&self, _: &str, _: &[&str], mut w: Box<dyn Write + Send>) -> Result<(), String> {
            let n = self.0.fetch_add(1, Ordering::Relaxed) + 1;
            writeln!(w, "run {n}").map_err(|e| e.to_string())
        }
    }

    /// `watch` redraws the answer on one line until a line arrives, then
    /// the session carries on.
    #[test]
    fn watch_repeats_until_enter() {
        let c = Arc::new(Counter(std::sync::atomic::AtomicU32::new(0)));
        let monitor = Arc::new(Monitor::new());
        monitor.register_device(c.clone());
        monitor.clone().start_server("127.0.0.1:0".to_string());
        let addr = wait_for("bind", || monitor.bound_addr());
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        s.write_all(b"watch -n 0.1 count\n").unwrap();
        wait_for("three runs", || (c.0.load(Ordering::Relaxed) >= 3).then_some(()));
        s.write_all(b"\ncount\nquit\n").unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).unwrap();
        monitor.shutdown();
        let out = String::from_utf8_lossy(&out);
        assert!(out.contains("\rrun 2\x1b[K"), "{out}");
        let runs = c.0.load(Ordering::Relaxed);
        assert!(out.contains(&format!("run {runs}")), "the plain command still answers: {out}");
    }

    #[test]
    fn watch_needs_a_known_command() {
        let out = converse(b"watch nosuch\nwatch\n");
        assert!(out.contains("watch: unknown command nosuch"), "{out}");
        assert!(out.contains("run a command again"), "{out}");
    }
}
