//! End-to-end readiness and state reporting (#102).
//!
//! Boots the real `iris` binary in `--ci` mode with no disk, then drives the
//! monitor over TCP:
//!
//! * a command sent before the monitor binds is refused (the client must poll,
//!   not sleep), and the same poll then observes `starting` and `monitor_ready`;
//! * `machine-start` acknowledges `starting -> running` and is a no-op the
//!   second time;
//! * `machine-stop` halts, and `machine-start` after the halt and after a
//!   `reset` acknowledges `halted -> starting -> running`;
//! * `machine-status` text and JSON agree with each acknowledgement.
//!
//! The state-machine unit tests live in `src/machine_state.rs`; this is the
//! proof that the command surface, the TCP monitor and `--ci` startup order all
//! report the same thing.

#![cfg(unix)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(20);

/// Kills iris if a test assertion fires before the clean `quit`, so a failure
/// cannot leave an orphan holding this test's ports.
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

fn free_ports(n: usize) -> Vec<u16> {
    let listeners: Vec<_> = (0..n)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port"))
        .collect();
    listeners.iter().map(|l| l.local_addr().unwrap().port()).collect()
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "iris-monitor-ready-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test directory");
    dir
}

fn write_config(dir: &Path, monitor: u16, serial_a: u16, serial_b: u16, ci_socket: &Path) {
    let text = format!(
        "headless = true\n\
         monitor_port = {monitor}\n\
         serial_port_a = {serial_a}\n\
         serial_port_b = {serial_b}\n\
         ci_socket = \"{}\"\n\
         \n\
         # No disk: the embedded PROM is enough for this test, and the default\n\
         # scsi1.raw would otherwise abort startup when it is absent.\n\
         scsi = {{}}\n\
         \n\
         # Keep the card the PROM probes; send its sound nowhere.\n\
         [audio]\n\
         output = \"none\"\n",
        ci_socket.display()
    );
    std::fs::write(dir.join("iris.toml"), text).expect("write test config");
}

/// Retry the connect (the readiness poll a harness should use) and drain the
/// monitor's banner up to its first prompt.
fn connect_polling(port: u16, deadline: Instant) -> TcpStream {
    loop {
        match TcpStream::connect(("127.0.0.1", port)) {
            Ok(mut stream) => {
                stream.set_read_timeout(Some(Duration::from_millis(100))).unwrap();
                read_until_prompt(&mut stream, deadline);
                return stream;
            }
            Err(_) => {
                assert!(Instant::now() < deadline, "monitor on port {port} never accepted a connection");
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

/// One CI control-socket request, retried until the socket appears (it is
/// created after the monitor binds).
fn ci_command(socket: &Path, cmd: &str, deadline: Instant) -> serde_json::Value {
    loop {
        if let Ok(mut stream) = UnixStream::connect(socket) {
            stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let request = format!("{{\"cmd\":\"{cmd}\",\"args\":{{}}}}\n");
            stream.write_all(request.as_bytes()).unwrap();
            stream.flush().unwrap();
            let mut line = String::new();
            BufReader::new(stream).read_line(&mut line).unwrap();
            let response: serde_json::Value =
                serde_json::from_str(line.trim()).expect("CI response parses");
            assert_eq!(response["ok"], true, "CI {cmd}: {response}");
            return response["data"].clone();
        }
        assert!(
            Instant::now() < deadline,
            "CI socket {} never appeared",
            socket.display()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn read_until_prompt(stream: &mut TcpStream, deadline: Instant) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => panic!("monitor closed the connection: {:?}", String::from_utf8_lossy(&out)),
            Ok(n) => {
                out.extend_from_slice(&buf[..n]);
                // The monitor's prompt is a line of its own: responses end in
                // CRLF followed by "> ". Matching any "> " would stop on the
                // arrow in "halted -> starting -> running".
                if out.ends_with(b"\r\n> ") {
                    return out;
                }
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for the monitor prompt; got {:?}",
                    String::from_utf8_lossy(&out)
                );
            }
            Err(e) => panic!("monitor read failed: {e}"),
        }
    }
}

fn command(stream: &mut TcpStream, cmd: &str) -> String {
    stream.write_all(cmd.as_bytes()).unwrap();
    stream.write_all(b"\r\n").unwrap();
    stream.flush().unwrap();
    let bytes = read_until_prompt(stream, Instant::now() + Duration::from_secs(10));
    let text = String::from_utf8_lossy(&bytes).replace("\r\n", "\n");
    // Drop the trailing prompt the monitor writes after every command.
    text.trim_end_matches("> ").trim().to_string()
}

fn status_json(reply: &str) -> serde_json::Value {
    let start = reply.find('{').unwrap_or_else(|| panic!("no JSON in {reply:?}"));
    let end = reply.rfind('}').unwrap_or_else(|| panic!("no JSON in {reply:?}"));
    serde_json::from_str(&reply[start..=end]).expect("machine-status JSON parses")
}

fn wait_for_exit(child: &mut Child, deadline: Instant) -> std::process::ExitStatus {
    loop {
        match child.try_wait().expect("wait for iris") {
            Some(status) => return status,
            None if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            None => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("iris did not exit after quit");
            }
        }
    }
}

#[test]
fn monitor_start_is_acknowledged_and_idempotent() {
    let dir = temp_dir("commands");
    let ports = free_ports(3);
    let (monitor_port, serial_a, serial_b) = (ports[0], ports[1], ports[2]);
    let ci_socket = dir.join("iris.sock");
    write_config(&dir, monitor_port, serial_a, serial_b, &ci_socket);

    // Nothing is listening yet. This is the case the old harness hid with a
    // fixed sleep: the first connect fails, so the caller must retry rather
    // than assume the command was delivered.
    assert!(
        TcpStream::connect(("127.0.0.1", monitor_port)).is_err(),
        "port {monitor_port} was already in use; retry the test"
    );

    let stderr = std::fs::File::create(dir.join("stderr.log")).unwrap();
    let mut child = ChildGuard(
        Command::new(env!("CARGO_BIN_EXE_iris"))
            .current_dir(&dir)
            .arg("--config")
            .arg(dir.join("iris.toml"))
            .arg("--ci")
            .arg("--ci-socket")
            .arg(&ci_socket)
            .arg("--monitor-port")
            .arg(monitor_port.to_string())
            .arg("--serial-port-a")
            .arg(serial_a.to_string())
            .arg("--serial-port-b")
            .arg(serial_b.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr))
            .spawn()
            .expect("spawn iris"),
    );

    let mut stream = connect_polling(monitor_port, Instant::now() + STARTUP_TIMEOUT);

    let status = status_json(&command(&mut stream, "machine-status json"));
    assert_eq!(status["state"], "starting", "CI mode parks the CPU until machine-start");
    assert_eq!(status["cpu_running"], false);
    assert_eq!(status["monitor_ready"], true);
    assert_eq!(
        status["monitor_addr"],
        format!("127.0.0.1:{monitor_port}"),
        "the status names the monitor's own address"
    );

    // The CI socket is created after the monitor binds, so a successful answer
    // is itself proof of readiness; both surfaces must report the same state.
    let ci = ci_command(&ci_socket, "status", Instant::now() + STARTUP_TIMEOUT);
    assert_eq!(ci, status, "CI status and machine-status disagree");

    let text = command(&mut stream, "machine-status");
    assert!(text.contains("state:   starting"), "{text}");
    assert!(text.contains("monitor: ready (127.0.0.1:"), "{text}");

    // First start: the acknowledgement names the whole transition.
    let ack = command(&mut stream, "machine-start");
    assert_eq!(ack, "machine-start: starting -> running");
    let status = status_json(&command(&mut stream, "machine-status json"));
    assert_eq!(status["state"], "running");
    assert_eq!(status["cpu_running"], true);

    // Duplicate start: no second start, no error.
    let ack = command(&mut stream, "machine-start");
    assert_eq!(ack, "machine-start: running (no change)");

    // Start after halt.
    assert_eq!(command(&mut stream, "machine-stop"), "machine-stop: running -> halted");
    let status = status_json(&command(&mut stream, "machine-status json"));
    assert_eq!(status["state"], "halted");
    assert_eq!(status["cpu_running"], false);
    assert_eq!(status["monitor_ready"], true, "a halted machine keeps its monitor");
    assert_eq!(command(&mut stream, "machine-start"), "machine-start: halted -> starting -> running");

    // Start after reset: reset leaves the CPU stopped and the monitor alive.
    let reset = command(&mut stream, "reset");
    assert!(reset.contains("Resetting machine..."), "{reset}");
    let status = status_json(&command(&mut stream, "machine-status json"));
    assert_eq!(status["state"], "halted", "reset leaves the machine halted");
    assert_eq!(command(&mut stream, "machine-start"), "machine-start: halted -> starting -> running");

    // Shut down cleanly through the CI socket; there is no guest disk here, so
    // stopping the machine first is the whole safety story.
    assert_eq!(command(&mut stream, "machine-stop"), "machine-stop: running -> halted");
    drop(stream);
    ci_command(&ci_socket, "quit", Instant::now() + Duration::from_secs(10));
    let status = wait_for_exit(&mut child.0, Instant::now() + Duration::from_secs(10));
    assert!(status.success(), "iris exited with {status}");
    let _ = std::fs::remove_dir_all(&dir);
}
