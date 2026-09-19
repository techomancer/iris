use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use parking_lot::Mutex;
use std::thread;
use std::net::{TcpListener, TcpStream};
use std::io::{Write, BufReader, BufRead, BufWriter};
use crate::traits::Device;

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

    pub fn register_device(&mut self, device: Arc<dyn Device>) {
        self.devices.lock().push(device);
    }

    pub fn start_server(self: Arc<Self>, addr: String) {
        thread::spawn(move || {
            // Fail soft (rather than panic-aborting the whole process) if the
            // port can't be bound — most commonly because a previous machine
            // instance's monitor thread from the same GUI session is still
            // holding it. Dropping a Machine shuts its monitor down and frees
            // the port, but a Machine that is merely stopped keeps it. The
            // machine still boots; the monitor console is just unavailable.
            let listener = match TcpListener::bind(&addr) {
                Ok(l) => l,
                Err(e) => {
                    log::warn!("monitor disabled: failed to bind {addr}: {e}");
                    return;
                }
            };
            // Publish the bound address before checking the flag: a concurrent
            // shutdown() either sees the address (and wakes us) or we see the
            // flag here.
            *self.bound.lock() = listener.local_addr().ok();
            if self.shutdown.load(Ordering::SeqCst) {
                return;
            }
            println!("Monitor listening on {}", addr);
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

fn handle_client(stream: TcpStream, devices: Arc<Mutex<Vec<Arc<dyn Device>>>>) {
    // Register this connection with DevLog so log output is broadcast here.
    // Wrap in CrlfWriter so bare \n from log messages renders as CRLF on the
    // telnet client side.
    if let Some(dl) = crate::devlog::DEVLOG.get() {
        let w: crate::devlog::DevLogWriter = Arc::new(Mutex::new(
            BufWriter::new(crate::telnet::CrlfWriter::new(stream.try_clone().unwrap()))
        ));
        dl.add_sink(w);
    }

    // Monitor is line-oriented: stay in NVT (passive telnet — strip inbound
    // IAC, decline whatever the client offers, but don't initiate). The
    // client keeps its local echo and line editor. Outbound goes through
    // CrlfWriter so bare \n becomes CRLF on the wire as NVT requires.
    let mut reader = BufReader::new(crate::telnet::TelnetReader::new_passive(stream.try_clone().unwrap()));
    let mut writer = BufWriter::new(crate::telnet::CrlfWriter::new(stream.try_clone().unwrap()));
    let mut line = String::new();
    
    {
        let logo = include_str!("ascii-art.txt");
        for line in logo.lines() {
            let _ = write!(writer, "{}\r\n", line);
        }
    }
    let _ = write!(writer, "\r\n{} Monitor\r\n> ", crate::machine::emulator_name());
    let _ = writer.flush();
    
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        
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
        
        let mut target_device: Option<Arc<dyn Device>> = None;
        let mut is_help = false;
        
        if cmd == "help" {
            is_help = true;
        }

        {
            let devs = devices.lock();
            if is_help {
                for dev in devs.iter() {
                    for (c, h) in dev.register_commands() {
                        let _ = writeln!(writer, "{:12} - {}", c, h);
                    }
                }
            } else {
                for dev in devs.iter() {
                    let cmds = dev.register_commands();
                    if cmds.iter().any(|(c, _)| c == cmd) {
                        target_device = Some(dev.clone());
                        break;
                    }
                }
            }
        }
        
        if !is_help {
            if let Some(dev) = target_device {
                let cmd_writer = Box::new(BufWriter::new(crate::telnet::CrlfWriter::new(stream.try_clone().unwrap())));
                match dev.execute_command(cmd, args, cmd_writer) {
                    Ok(_) => {
                    }
                    Err(e) => {
                        let _ = write!(writer, "Error: {}\n", e);
                    }
                }
            } else {
            let _ = writeln!(writer, "Unknown command. Type 'help' for list.");
            }
        }
        
        let _ = write!(writer, "> ");
        let _ = writer.flush();
    }
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

    #[test]
    fn shutdown_before_bind_does_not_serve() {
        let monitor = Arc::new(Monitor::new());
        monitor.shutdown();
        monitor.clone().start_server("127.0.0.1:0".to_string());
        let addr = wait_for("bind", || monitor.bound_addr());
        wait_for("port release", || TcpListener::bind(addr).ok());
    }
}
