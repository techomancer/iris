//! A host serial port (`/dev/ttyUSB0`, `/dev/ttyACM0`, `/dev/ttyS0`) as a
//! guest serial line, raw: the guest's bytes go out as they are, the port's
//! rate, framing, DTR, RTS and break follow the guest's UART, and the port's
//! CTS, DCD, DSR and RI come back as the UART's inputs. A device that takes its
//! power from DTR and RTS, as a Spaceball does, is powered when the guest's
//! driver raises them, as on the machine's own connector.
//!
//! The port joins the host-IO thread's poller for input and room to write. Modem inputs
//! have no event to wait on, so a machine-clock timer polls them while DTR or RTS is raised.

use std::collections::VecDeque;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use crate::dev::serial_host::{LineConfig, ModemIn, ModemOut, SerialBackend};
use crate::hptimer::{Timer, TimerManager};
use crate::net::wake::{host_io, HostPump};

/// The most input kept for a guest that is not reading.
const RX_CAP: usize = 64 * 1024;
/// How often the modem inputs are polled while the guest has the lines raised, in machine ns.
const MODEM_POLL_NS: u64 = 5_000_000;

/// Modem-input bits as the poll stores them.
const IN_CTS: u8 = 1;
const IN_DCD: u8 = 2;
const IN_DSR: u8 = 4;
const IN_RI: u8 = 8;

struct Shared {
    fd: i32,
    path: String,
    /// A serial port; otherwise a raw MIDI port, which has no line to set.
    tty: bool,
    rx: Mutex<VecDeque<u8>>,
    tx: Mutex<VecDeque<u8>>,
    modem: AtomicU8,
    /// The port in the host-IO poller, until it is gone.
    watch: Mutex<Option<sys::Watch>>,
    /// The poller is watching for room to write.
    want_write: AtomicBool,
    /// Polls the modem inputs; bound by the UART when the machine's timers exist.
    modem_timer: OnceLock<Timer>,
    /// What the port was last set to, for the log and to skip repeats.
    line: Mutex<Option<LineConfig>>,
    out: Mutex<Option<ModemOut>>,
}

pub struct HostTty {
    shared: Arc<Shared>,
    /// Reads the port into `rx` on the host-IO thread.
    pump: Option<HostPump>,
}

#[cfg(unix)]
mod sys {
    use super::*;
    use std::ffi::CString;

    fn check(r: i32) -> io::Result<i32> {
        if r < 0 { Err(io::Error::last_os_error()) } else { Ok(r) }
    }

    /// Open `path` non-blocking. With `plain_ok`, a device that is not a tty
    /// (a raw MIDI port) is opened as it is, and write-only when it has no
    /// input; a tty is set up as [`open`] does. `Ok((fd, is_tty))`.
    pub fn open_any(path: &str, plain_ok: bool) -> io::Result<(i32, bool)> {
        let c = CString::new(path).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path has a NUL"))?;
        let flags = libc::O_NOCTTY | libc::O_NONBLOCK | libc::O_CLOEXEC;
        let fd = match check(unsafe { libc::open(c.as_ptr(), libc::O_RDWR | flags) }) {
            Err(e) if plain_ok && matches!(e.raw_os_error(), Some(libc::ENXIO | libc::ENODEV | libc::EINVAL | libc::EACCES)) => {
                check(unsafe { libc::open(c.as_ptr(), libc::O_WRONLY | flags) }).map_err(|_| e)?
            }
            r => r?,
        };
        if unsafe { libc::isatty(fd) } == 1 {
            return setup_tty(fd).map(|()| (fd, true));
        }
        if plain_ok {
            return Ok((fd, false));
        }
        unsafe { libc::close(fd) };
        Err(io::Error::new(io::ErrorKind::InvalidInput, "not a serial port (tty)"))
    }

    /// Open the port raw, non-blocking and exclusive, at 9600 8N1 with DTR
    /// and RTS dropped: the guest's UART raises them when its driver does.
    pub fn open(path: &str) -> io::Result<i32> {
        open_any(path, false).map(|(fd, _)| fd)
    }

    fn setup_tty(fd: i32) -> io::Result<()> {
        let setup = || -> io::Result<()> {
            check(unsafe { libc::ioctl(fd, libc::TIOCEXCL as _) })
                .map_err(|e| io::Error::new(e.kind(), format!("cannot take it exclusively: {e}")))?;
            let mut t: libc::termios = unsafe { std::mem::zeroed() };
            check(unsafe { libc::tcgetattr(fd, &mut t) })?;
            unsafe { libc::cfmakeraw(&mut t) };
            t.c_cflag |= libc::CLOCAL | libc::CREAD;
            t.c_cflag &= !libc::CRTSCTS;
            t.c_iflag &= !(libc::IXON | libc::IXOFF | libc::IXANY);
            t.c_cc[libc::VMIN] = 0;
            t.c_cc[libc::VTIME] = 0;
            unsafe { libc::cfsetspeed(&mut t, libc::B9600) };
            check(unsafe { libc::tcsetattr(fd, libc::TCSANOW, &t) })?;
            set_modem(fd, ModemOut::default())?;
            unsafe { libc::tcflush(fd, libc::TCIOFLUSH) };
            Ok(())
        };
        setup().inspect_err(|_| unsafe {
            libc::close(fd);
        })
    }

    pub fn close(fd: i32) {
        unsafe { libc::close(fd) };
    }

    /// The standard rate within 2% of `baud`, so a UART whose clock gives
    /// 9562 for 9600 sets the host port to 9600.
    fn standard(baud: u32) -> Option<(u32, libc::speed_t)> {
        const RATES: &[(u32, libc::speed_t)] = &[
            (300, libc::B300), (600, libc::B600), (1200, libc::B1200), (2400, libc::B2400),
            (4800, libc::B4800), (9600, libc::B9600), (19200, libc::B19200), (38400, libc::B38400),
            (57600, libc::B57600), (115200, libc::B115200), (230400, libc::B230400),
        ];
        RATES.iter().copied().find(|&(r, _)| (baud as i64 - r as i64).unsigned_abs() * 50 <= r as u64)
    }

    /// Set rate and framing. A rate that is no standard one (31250 for MIDI)
    /// is set exactly on Linux; elsewhere the nearest standard rate is used.
    /// Returns the rate the port now runs at.
    pub fn set_line(fd: i32, line: LineConfig) -> io::Result<Option<u32>> {
        let mut t: libc::termios = unsafe { std::mem::zeroed() };
        check(unsafe { libc::tcgetattr(fd, &mut t) })?;
        t.c_cflag &= !(libc::CSIZE | libc::PARENB | libc::PARODD | libc::CSTOPB);
        t.c_cflag |= match line.data_bits {
            5 => libc::CS5,
            6 => libc::CS6,
            7 => libc::CS7,
            _ => libc::CS8,
        };
        match line.parity {
            crate::dev::serial_host::Parity::None => {}
            crate::dev::serial_host::Parity::Odd => t.c_cflag |= libc::PARENB | libc::PARODD,
            crate::dev::serial_host::Parity::Even => t.c_cflag |= libc::PARENB,
        }
        if line.stop_bits >= 2 {
            t.c_cflag |= libc::CSTOPB;
        }
        let mut rate = None;
        let mut exact = None;
        if let Some(baud) = line.baud {
            match standard(baud) {
                Some((r, s)) => {
                    unsafe { libc::cfsetspeed(&mut t, s) };
                    rate = Some(r);
                }
                None => exact = Some(baud),
            }
        }
        check(unsafe { libc::tcsetattr(fd, libc::TCSANOW, &t) })?;
        if let Some(baud) = exact {
            rate = Some(set_exact(fd, baud)?);
        }
        Ok(rate)
    }

    #[cfg(target_os = "linux")]
    fn set_exact(fd: i32, baud: u32) -> io::Result<u32> {
        let mut t2: libc::termios2 = unsafe { std::mem::zeroed() };
        check(unsafe { libc::ioctl(fd, libc::TCGETS2 as _, &mut t2) })?;
        t2.c_cflag &= !libc::CBAUD;
        t2.c_cflag |= libc::BOTHER;
        t2.c_ispeed = baud;
        t2.c_ospeed = baud;
        check(unsafe { libc::ioctl(fd, libc::TCSETS2 as _, &t2) })?;
        Ok(baud)
    }

    #[cfg(not(target_os = "linux"))]
    fn set_exact(fd: i32, baud: u32) -> io::Result<u32> {
        let near = [300u32, 600, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200, 230400]
            .into_iter()
            .min_by_key(|r| (baud as i64 - *r as i64).unsigned_abs())
            .unwrap();
        let (_, s) = standard(near).unwrap();
        let mut t: libc::termios = unsafe { std::mem::zeroed() };
        check(unsafe { libc::tcgetattr(fd, &mut t) })?;
        unsafe { libc::cfsetspeed(&mut t, s) };
        check(unsafe { libc::tcsetattr(fd, libc::TCSANOW, &t) })?;
        Ok(near)
    }

    pub fn set_modem(fd: i32, out: ModemOut) -> io::Result<()> {
        let on = (if out.dtr { libc::TIOCM_DTR } else { 0 }) | (if out.rts { libc::TIOCM_RTS } else { 0 });
        let off = (libc::TIOCM_DTR | libc::TIOCM_RTS) & !on;
        if on != 0 {
            check(unsafe { libc::ioctl(fd, libc::TIOCMBIS as _, &on) })?;
        }
        if off != 0 {
            check(unsafe { libc::ioctl(fd, libc::TIOCMBIC as _, &off) })?;
        }
        let brk = if out.brk { libc::TIOCSBRK } else { libc::TIOCCBRK };
        check(unsafe { libc::ioctl(fd, brk as _) })?;
        Ok(())
    }

    pub fn modem_in(fd: i32) -> Option<u8> {
        let mut bits: libc::c_int = 0;
        if unsafe { libc::ioctl(fd, libc::TIOCMGET as _, &mut bits) } < 0 {
            return None;
        }
        Some(
            (if bits & libc::TIOCM_CTS != 0 { IN_CTS } else { 0 })
                | (if bits & libc::TIOCM_CAR != 0 { IN_DCD } else { 0 })
                | (if bits & libc::TIOCM_DSR != 0 { IN_DSR } else { 0 })
                | (if bits & libc::TIOCM_RNG != 0 { IN_RI } else { 0 }),
        )
    }

    /// The port as the poller sees it; `HostTty` closes it, after the poller has let go.
    pub struct Fd(i32);

    impl std::os::fd::AsFd for Fd {
        fn as_fd(&self) -> std::os::fd::BorrowedFd<'_> {
            // SAFETY: the port stays open until `HostTty` drops its `Watch`.
            unsafe { std::os::fd::BorrowedFd::borrow_raw(self.0) }
        }
    }

    pub type Watch = crate::net::wake::Watched<Fd>;

    pub fn watch(fd: i32) -> io::Result<Watch> {
        crate::net::wake::Watched::new(Fd(fd), &host_io().wake, true, false)
    }

    pub fn watch_write(w: &mut Watch, on: bool) {
        w.set_interest(true, on);
    }

    /// Non-blocking read; 0 when nothing is there.
    pub fn read(fd: i32, buf: &mut [u8]) -> io::Result<usize> {
        let n = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) };
        if n < 0 {
            let e = io::Error::last_os_error();
            return if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted) { Ok(0) } else { Err(e) };
        }
        Ok(n as usize)
    }

    /// Non-blocking write; how much the port took.
    pub fn write(fd: i32, buf: &[u8]) -> io::Result<usize> {
        let n = unsafe { libc::write(fd, buf.as_ptr().cast(), buf.len()) };
        if n < 0 {
            let e = io::Error::last_os_error();
            return if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted) { Ok(0) } else { Err(e) };
        }
        Ok(n as usize)
    }
}

#[cfg(not(unix))]
mod sys {
    use super::*;
    fn unsupported<T>() -> io::Result<T> {
        Err(io::Error::new(io::ErrorKind::Unsupported, "host serial ports are not supported on this platform yet"))
    }
    pub fn open(_: &str) -> io::Result<i32> { unsupported() }
    pub fn open_any(_: &str, _: bool) -> io::Result<(i32, bool)> { unsupported() }
    pub fn close(_: i32) {}
    pub fn set_line(_: i32, _: LineConfig) -> io::Result<Option<u32>> { unsupported() }
    pub fn set_modem(_: i32, _: ModemOut) -> io::Result<()> { unsupported() }
    pub fn modem_in(_: i32) -> Option<u8> { None }
    pub type Watch = ();
    pub fn watch(_: i32) -> io::Result<Watch> { unsupported() }
    pub fn watch_write(_: &mut Watch, _: bool) {}
    pub fn read(_: i32, _: &mut [u8]) -> io::Result<usize> { unsupported() }
    pub fn write(_: i32, _: &[u8]) -> io::Result<usize> { unsupported() }
}

impl HostTty {
    /// A host serial port.
    pub fn open(path: &str) -> io::Result<Self> {
        Self::start(path, sys::open(path)?, true)
    }

    /// A host MIDI port: a raw MIDI device (Linux's `/dev/snd/midiC1D0` or
    /// `/dev/midi1`), whose bytes pass as they are, or a serial port with a
    /// MIDI interface on it, set up as [`HostTty::open`] does.
    pub fn open_midi(path: &str) -> io::Result<Self> {
        let (fd, tty) = sys::open_any(path, true)?;
        Self::start(path, fd, tty)
    }

    fn start(path: &str, fd: i32, tty: bool) -> io::Result<Self> {
        let shared = Arc::new(Shared {
            fd,
            path: path.to_string(),
            tty,
            rx: Mutex::new(VecDeque::new()),
            tx: Mutex::new(VecDeque::new()),
            modem: AtomicU8::new(sys::modem_in(fd).unwrap_or(0)),
            watch: Mutex::new(None),
            want_write: AtomicBool::new(false),
            modem_timer: OnceLock::new(),
            line: Mutex::new(None),
            out: Mutex::new(None),
        });
        match sys::watch(fd) {
            Ok(w) => *shared.watch.lock() = Some(w),
            Err(e) => log::warn!("serial: {path}: not watched for input: {e}"),
        }
        let s = shared.clone();
        let pump = Some(host_io().add(move || s.read_in()));
        Ok(Self { shared, pump })
    }
}

impl Shared {
    /// Take what the port has into `rx`; whether any came. A gone port leaves the poller.
    fn read_in(&self) -> bool {
        if self.watch.lock().is_none() {
            return false;
        }
        let mut buf = [0u8; 256];
        match sys::read(self.fd, &mut buf) {
            Ok(0) => false,
            Ok(n) => {
                let mut rx = self.rx.lock();
                rx.extend(&buf[..n]);
                while rx.len() > RX_CAP {
                    rx.pop_front();
                }
                true
            }
            Err(e) => {
                log::warn!("serial: {}: {e}; the port is gone", self.path);
                self.watch.lock().take();
                false
            }
        }
    }

    /// Re-read the modem inputs, wake the UART's pump on a change, and keep polling while the lines are raised.
    fn poll_modem(&self) {
        if let Some(m) = sys::modem_in(self.fd) {
            if self.modem.swap(m, Ordering::Relaxed) != m {
                host_io().wake.notify();
            }
        }
        if let (Some(t), Some(out)) = (self.modem_timer.get(), *self.out.lock()) {
            t.set_due((out.dtr || out.rts).then(|| t.now() + MODEM_POLL_NS));
        }
    }

    fn write_out(&self, tx: &mut VecDeque<u8>) {
        self.try_write(tx);
        let held = !tx.is_empty();
        if self.want_write.swap(held, Ordering::Relaxed) != held {
            if let Some(w) = self.watch.lock().as_mut() {
                sys::watch_write(w, held);
            }
        }
    }

    fn try_write(&self, tx: &mut VecDeque<u8>) {
        while !tx.is_empty() {
            let n = match sys::write(self.fd, tx.as_slices().0) {
                Ok(n) => n,
                Err(e) => {
                    log::warn!("serial: {}: write: {e}; output dropped", self.path);
                    tx.clear();
                    return;
                }
            };
            if n == 0 {
                return;
            }
            tx.drain(..n);
        }
    }
}

impl SerialBackend for HostTty {
    fn send_byte(&self, byte: u8) {
        let mut tx = self.shared.tx.lock();
        tx.push_back(byte);
        self.shared.write_out(&mut tx);
    }

    fn flush(&self) {
        let mut tx = self.shared.tx.lock();
        self.shared.write_out(&mut tx);
    }

    fn queued(&self) -> bool {
        !self.shared.tx.lock().is_empty()
    }

    fn recv_byte(&self) -> io::Result<u8> {
        self.shared.rx.lock().pop_front().ok_or_else(|| io::ErrorKind::WouldBlock.into())
    }

    fn set_line(&self, line: LineConfig) {
        if !self.shared.tty {
            return;
        }
        let mut last = self.shared.line.lock();
        if *last == Some(line) {
            return;
        }
        *last = Some(line);
        match sys::set_line(self.shared.fd, line) {
            Ok(Some(rate)) => log::info!("serial: {} set to {} ({rate} on the port)", self.shared.path, line),
            Ok(None) => log::info!("serial: {} framing {} (rate left as it was)", self.shared.path, line),
            Err(e) => log::warn!("serial: {}: cannot set {line}: {e}", self.shared.path),
        }
    }

    fn set_modem(&self, out: ModemOut) {
        if !self.shared.tty {
            return;
        }
        let mut last = self.shared.out.lock();
        if *last == Some(out) {
            return;
        }
        *last = Some(out);
        drop(last);
        match sys::set_modem(self.shared.fd, out) {
            Ok(()) => log::info!("serial: {} DTR {} RTS {}{}", self.shared.path, on(out.dtr), on(out.rts), if out.brk { " break" } else { "" }),
            Err(e) => log::warn!("serial: {}: cannot set the modem lines: {e}", self.shared.path),
        }
        self.shared.poll_modem();
    }

    fn modem_in(&self) -> ModemIn {
        if !self.shared.tty {
            return ModemIn::IDLE;
        }
        let m = self.shared.modem.load(Ordering::Relaxed);
        ModemIn { cts: m & IN_CTS != 0, dcd: m & IN_DCD != 0, dsr: m & IN_DSR != 0, ri: m & IN_RI != 0 }
    }

    fn raw(&self) -> bool {
        true
    }

    fn bind_timers(&self, tm: &Arc<TimerManager>) {
        self.shared.modem_timer.get_or_init(|| {
            let t = Timer::new(tm, &Arc::downgrade(&self.shared), |s: &Shared, _| s.poll_modem());
            t.start();
            t
        });
    }

    fn describe(&self) -> String {
        if !self.shared.tty {
            return format!("MIDI port {}", self.shared.path);
        }
        let line = self.shared.line.lock().map_or_else(|| "rate not set yet".into(), |l| l.to_string());
        let out = self.shared.out.lock().unwrap_or_default();
        let m = self.modem_in();
        format!(
            "{} {line}, DTR {} RTS {}{}; CTS {} DCD {} DSR {}",
            self.shared.path,
            on(out.dtr),
            on(out.rts),
            if out.brk { ", break" } else { "" },
            on(m.cts),
            on(m.dcd),
            on(m.dsr)
        )
    }
}

fn on(b: bool) -> &'static str {
    if b { "on" } else { "off" }
}

impl Drop for HostTty {
    fn drop(&mut self) {
        if let Some(t) = self.shared.modem_timer.get() {
            t.stop();
        }
        self.pump.take();
        self.shared.watch.lock().take();
        sys::close(self.shared.fd);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pseudo-terminal pair: the master end's fd and the slave's path.
    #[cfg(unix)]
    fn pty() -> (std::fs::File, String) {
        use std::os::fd::FromRawFd;
        unsafe {
            let m = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY);
            assert!(m >= 0 && libc::grantpt(m) == 0 && libc::unlockpt(m) == 0);
            let mut name = [0 as libc::c_char; 128];
            assert_eq!(libc::ptsname_r(m, name.as_mut_ptr(), name.len()), 0);
            (std::fs::File::from_raw_fd(m), std::ffi::CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned())
        }
    }

    /// The slave as a port without `setup_tty`, which a pty cannot take (it has no modem lines to set).
    #[cfg(unix)]
    fn pty_port(slave: &str) -> HostTty {
        let c = std::ffi::CString::new(slave).unwrap();
        let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDWR | libc::O_NOCTTY | libc::O_NONBLOCK) };
        assert!(fd >= 0);
        HostTty::start(slave, fd, true).unwrap()
    }

    /// A byte arriving on the port reaches `recv_byte` with nothing polling it.
    #[cfg(unix)]
    #[test]
    fn input_on_the_port_wakes_the_host_io_thread() {
        use std::io::Write;
        let (mut master, slave) = pty();
        let tty = pty_port(&slave);
        master.write_all(b"k\n").unwrap(); // canonical mode: the line ends it
        assert!(crate::test_board::until(|| tty.recv_byte().is_ok_and(|b| b == b'k')));
    }

    /// The modem inputs are polled on the machine clock only while the guest has DTR or RTS raised.
    #[cfg(unix)]
    #[test]
    fn the_modem_poll_runs_only_while_the_lines_are_raised() {
        let (_master, slave) = pty();
        let clock = crate::test_board::Stepped::new();
        let tty = pty_port(&slave);
        tty.bind_timers(&clock.tm);
        let due = || tty.shared.modem_timer.get().unwrap().due();
        assert_eq!(due(), None);
        tty.set_modem(ModemOut { dtr: true, rts: false, brk: false });
        assert!(due().is_some(), "raised: polling");
        clock.step(MODEM_POLL_NS);
        assert!(due().is_some(), "and it keeps going");
        tty.set_modem(ModemOut::default());
        assert_eq!(due(), None, "dropped: it stops");
    }

    /// Against a real device, opt-in: `IRIS_TEST_TTY=/dev/ttyUSB0 cargo test
    /// --lib host_tty -- --ignored`. A Spaceball 5000 FLX (Magellan protocol)
    /// is powered from DTR and RTS and answers `vQ` with its version.
    #[test]
    #[ignore]
    fn a_real_device_answers_once_the_guest_powers_it() {
        let Ok(path) = std::env::var("IRIS_TEST_TTY") else { return };
        let tty = HostTty::open(&path).expect("open");
        tty.set_line(LineConfig { baud: Some(9562), ..LineConfig::default() });
        tty.set_modem(ModemOut { dtr: true, rts: true, brk: false });
        std::thread::sleep(std::time::Duration::from_millis(500));
        while tty.recv_byte().is_ok() {}
        for &b in b"vQ\r" {
            tty.send_byte(b);
        }
        let mut got = Vec::new();
        let until = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while std::time::Instant::now() < until && !got.ends_with(b"\r") || got.len() < 3 {
            match tty.recv_byte() {
                Ok(b) => got.push(b),
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
            if std::time::Instant::now() >= until {
                break;
            }
        }
        let text = String::from_utf8_lossy(&got);
        eprintln!("reply: {text:?}, modem in: {:?}", tty.modem_in());
        assert!(text.starts_with('v'), "a version reply, got {text:?}");
    }
}

#[cfg(test)]
mod midi_tests {
    use super::*;

    /// A raw MIDI device, opt-in: `IRIS_TEST_MIDI=/dev/snd/midiC3D0 cargo
    /// test --lib open_midi -- --ignored`. Sends one Active Sensing byte,
    /// which sounds nothing; the card's Tx count shows it went.
    #[test]
    #[ignore]
    fn open_midi_passes_bytes_to_a_raw_midi_device() {
        let Ok(path) = std::env::var("IRIS_TEST_MIDI") else { return };
        let port = HostTty::open_midi(&path).expect("open");
        assert!(port.describe().starts_with("MIDI port"), "{}", port.describe());
        assert_eq!(port.modem_in(), ModemIn::IDLE);
        port.set_line(LineConfig::default());
        port.send_byte(0xfe);
        port.flush();
        assert!(!port.queued(), "the device took it");
    }
}
