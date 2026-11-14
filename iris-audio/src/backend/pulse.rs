//! The PulseAudio backend: a native-protocol client over the server's
//! Unix socket. PipeWire's `pipewire-pulse` speaks the same protocol, so
//! this covers both.
//!
//! One blocking thread per stream owns the socket. The server clocks the
//! stream: it sends a request for so many bytes, the thread renders
//! exactly that and sends it back.
//!
//! The server does most of what a backend has to do elsewhere. A stream
//! not tied to a sink follows the default sink when the user changes it,
//! and the server converts if the new sink runs at another rate, so the
//! format given to [`Render::configure`] holds for the life of a
//! connection.
//!
//! If the server goes away the thread tries every half second to reach it
//! again. Meanwhile it takes the guest's frames by the host's clock and
//! discards them, as the [null backend](super::null) does: the guest sees a
//! device that never stopped, and what it plays while there is no server
//! is lost, not kept for later. With
//! [`OutputRequest::null_fallback`] a stream can also be opened while no
//! server is running; it starts in that state and plays when one appears.
//!
//! Nothing here is specific to Linux. The backend is built on every Unix,
//! and is the default wherever there is no backend for the system's own
//! sound: on FreeBSD and the other BSDs it plays through a PulseAudio or
//! pipewire-pulse server if the user runs one.

use std::ffi::CString;
use std::io::{BufReader, Cursor, Read};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime};

use pulseaudio::protocol as pa;

use crate::Error;
use crate::backend::null::Pacer;
use crate::backend::{Backend, OutputRequest, Render, RenderInfo, StreamHandle};
use crate::format::DeviceFormat;

/// The device latency asked for when the request names none. The server
/// treats it as the whole of its side: its stream buffer plus the sink.
const DEFAULT_LATENCY: Duration = Duration::from_millis(20);
/// How often the stream's real latency is asked for.
const LATENCY_INTERVAL: Duration = Duration::from_millis(250);
/// How long to wait between attempts to reach a server that went away.
const RETRY: Duration = Duration::from_millis(500);
/// How often frames are taken and discarded while there is no server.
const NULL_PERIOD: Duration = Duration::from_millis(5);
/// How long a server may take over each answer while a stream is set up.
/// One that accepts a connection and then says nothing must not hold the
/// thread, which has a guest to keep consuming.
const HANDSHAKE: Duration = Duration::from_secs(1);
/// The rate a stream runs at before any server has named one.
const NULL_RATE: u32 = 48000;
/// The tag on latency queries, to tell their replies from anything else.
const LATENCY_SEQ: u32 = 0x4c41_5459;
/// No sane control message is larger; a longer one means a lost framing.
const MAX_MESSAGE: u32 = 16 << 20;

pub struct PulseBackend {
    app: String,
    server: Option<PathBuf>,
}

impl PulseBackend {
    /// A backend whose streams belong to the application `app` in the
    /// host's mixer. Whether a server is running is found out when a
    /// stream is opened; [`PulseBackend::available`] asks beforehand.
    pub fn new(app: impl Into<String>) -> Self {
        Self { app: app.into(), server: None }
    }

    /// Use the server whose socket is at `path`, not the one the
    /// environment names (`PULSE_SERVER`, `PULSE_RUNTIME_PATH`,
    /// `XDG_RUNTIME_DIR`).
    pub fn server(mut self, path: impl Into<PathBuf>) -> Self {
        self.server = Some(path.into());
        self
    }

    /// Whether the environment names a server's socket and it exists. A
    /// program choosing between backends can ask this first.
    pub fn available() -> bool {
        pulseaudio::socket_path_from_env().is_some()
    }
}

fn fail(e: impl std::fmt::Display) -> Error {
    Error::Backend { backend: "pulse", message: e.to_string() }
}

fn cstring(s: &str) -> CString {
    CString::new(s.replace('\0', " ")).unwrap_or_default()
}

/// The layout a stream of `channels` is opened with: the smallest standard
/// one that holds them, in the order [`OutputConfig::channels`] documents.
///
/// [`OutputConfig::channels`]: crate::OutputConfig::channels
fn channel_map(channels: usize) -> pa::ChannelMap {
    use pa::ChannelPosition::*;
    match channels {
        1 => pa::ChannelMap::mono(),
        2 => pa::ChannelMap::stereo(),
        3 | 4 => pa::ChannelMap::new([FrontLeft, FrontRight, RearLeft, RearRight]),
        5 | 6 => pa::ChannelMap::new([FrontLeft, FrontRight, FrontCenter, Lfe, RearLeft, RearRight]),
        _ => pa::ChannelMap::new([FrontLeft, FrontRight, FrontCenter, Lfe, RearLeft, RearRight, SideLeft, SideRight]),
    }
}

/// A connection with one playback stream on it.
struct Session {
    sock: BufReader<UnixStream>,
    version: u16,
    /// The stream's channel on the connection.
    channel: u32,
    format: DeviceFormat,
    /// Bytes the server wants before its first request.
    requested: u32,
    device: String,
}

/// Reach the server. Failing here means there is none now; one may come.
fn dial(server: Option<&Path>) -> Result<UnixStream, Error> {
    let path = match server {
        Some(path) => path.to_owned(),
        None => pulseaudio::socket_path_from_env().ok_or_else(|| fail("no PulseAudio (or pipewire-pulse) socket found"))?,
    };
    UnixStream::connect(&path).map_err(|e| fail(format!("{}: {e}", path.display())))
}

fn connect(server: Option<&Path>, app: &str, request: &OutputRequest) -> Result<Session, Error> {
    open(dial(server)?, app, request)
}

/// Open the stream on a connection just made.
fn open(sock: UnixStream, app: &str, request: &OutputRequest) -> Result<Session, Error> {
    sock.set_read_timeout(Some(HANDSHAKE))?;
    let mut sock = BufReader::new(sock);

    let cookie = pulseaudio::cookie_path_from_env().and_then(|p| std::fs::read(p).ok()).unwrap_or_default();
    let auth = pa::AuthParams { version: pa::MAX_VERSION, supports_shm: false, supports_memfd: false, cookie };
    pa::write_command_message(sock.get_mut(), 0, &pa::Command::Auth(auth), pa::MAX_VERSION).map_err(fail)?;
    let (_, reply) = pa::read_reply_message::<pa::AuthReply>(&mut sock, pa::MAX_VERSION).map_err(fail)?;
    let version = pa::MAX_VERSION.min(reply.version);

    let mut props = pa::Props::new();
    props.set(pa::Prop::ApplicationName, cstring(app));
    pa::write_command_message(sock.get_mut(), 1, &pa::Command::SetClientName(props), version).map_err(fail)?;
    pa::read_reply_message::<pa::SetClientNameReply>(&mut sock, version).map_err(fail)?;

    // Open at the server's own rate, so that the only conversion on the
    // way to the speaker is ours.
    pa::write_command_message(sock.get_mut(), 2, &pa::Command::GetServerInfo, version).map_err(fail)?;
    let (_, server) = pa::read_reply_message::<pa::ServerInfo>(&mut sock, version).map_err(fail)?;
    let rate = if server.sample_spec.sample_rate > 0 { server.sample_spec.sample_rate } else { 48000 };

    let map = channel_map(request.channels);
    let channels = map.num_channels();
    let frame_bytes = u32::from(channels) * 4;
    let latency = request.latency.unwrap_or(DEFAULT_LATENCY).as_secs_f64();
    let target = ((latency * f64::from(rate)) as u32).max(64) * frame_bytes;
    let mut props = pa::Props::new();
    props.set(pa::Prop::MediaName, cstring(&request.name));
    let params = pa::PlaybackStreamParams {
        sample_spec: pa::SampleSpec { format: pa::SampleFormat::Float32Le, channels, sample_rate: rate },
        channel_map: map,
        cvolume: Some(pa::ChannelVolume::norm(channels)),
        buffer_attr: pa::stream::BufferAttr {
            max_length: u32::MAX,
            target_length: target,
            // Play at once and keep playing: the renderer does its own
            // prebuffering and always fills what is asked for.
            pre_buffering: 0,
            minimum_request_length: (target / 4).max(frame_bytes),
            fragment_size: u32::MAX,
        },
        // The target is the whole of the server's latency, sink included.
        flags: pa::stream::StreamFlags { adjust_latency: true, ..Default::default() },
        props,
        ..Default::default()
    };
    pa::write_command_message(sock.get_mut(), 3, &pa::Command::CreatePlaybackStream(params), version).map_err(fail)?;
    let (_, stream) = pa::read_reply_message::<pa::CreatePlaybackStreamReply>(&mut sock, version).map_err(fail)?;
    sock.get_ref().set_read_timeout(None)?;

    Ok(Session {
        sock,
        version,
        channel: stream.channel,
        format: DeviceFormat { rate: stream.sample_spec.sample_rate, channels: usize::from(stream.sample_spec.channels) },
        requested: stream.requested_bytes,
        device: stream.sink_name.map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
    })
}

/// What the stream's thread and its handle share.
struct Control {
    stop: AtomicBool,
    /// The live connection's socket, for the handle to shut down: that is
    /// what wakes the thread from a blocking read.
    sock: Mutex<Option<UnixStream>>,
    /// The sink's name; empty while there is no server.
    device: Mutex<String>,
    underruns: AtomicU64,
}

impl Control {
    /// Make `session` the one a stop will interrupt. False if the stop has
    /// already happened.
    fn adopt(&self, session: &Session) -> bool {
        let mut sock = self.sock.lock().unwrap();
        if self.stop.load(Relaxed) {
            return false;
        }
        *sock = session.sock.get_ref().try_clone().ok();
        *self.device.lock().unwrap() = session.device.clone();
        true
    }

    /// The connection is gone.
    fn orphan(&self) {
        *self.sock.lock().unwrap() = None;
        self.device.lock().unwrap().clear();
    }
}

/// Render `bytes` worth of whole frames and send them.
fn feed(s: &mut Session, render: &mut dyn Render, bytes: u32, info: &RenderInfo, samples: &mut Vec<f32>, wire: &mut Vec<u8>) -> Result<(), Error> {
    let n = bytes as usize / 4 / s.format.channels * s.format.channels;
    if n == 0 {
        return Ok(());
    }
    samples.clear();
    samples.resize(n, 0.0);
    render.render(samples, info);
    wire.clear();
    wire.extend(samples.iter().flat_map(|v| v.to_le_bytes()));
    pa::write_memblock(s.sock.get_mut(), s.channel, wire, 0).map_err(fail)
}

/// Serve one connection until it fails or the stream is stopped.
fn serve(s: &mut Session, render: &mut dyn Render, control: &Control) -> Result<(), Error> {
    let frame_bytes = (s.format.channels * 4) as f64;
    let (mut message, mut samples, mut wire) = (Vec::new(), Vec::new(), Vec::new());
    let mut info = RenderInfo { device_delay: None, device_underruns: control.underruns.load(Relaxed) };
    let mut asked = Instant::now();

    feed(s, render, s.requested, &info, &mut samples, &mut wire)?;
    loop {
        // Read a whole message whatever it is, so that one this code does
        // not understand cannot put the stream out of step.
        let desc = pa::read_descriptor(&mut s.sock).map_err(fail)?;
        if control.stop.load(Relaxed) {
            return Ok(());
        }
        if desc.length > MAX_MESSAGE {
            return Err(fail("the server sent an impossible message length"));
        }
        message.resize(desc.length as usize, 0);
        s.sock.read_exact(&mut message)?;
        if desc.channel != u32::MAX {
            continue; // Stream data; a playback stream receives none.
        }
        let mut body = Cursor::new(&message[..]);
        let Ok((seq, command)) = pa::Command::read_tag_prefixed(&mut body, s.version) else { continue };
        match command {
            pa::Command::Request(r) if r.channel == s.channel => {
                info.device_underruns = control.underruns.load(Relaxed);
                feed(s, render, r.length, &info, &mut samples, &mut wire)?;
                if asked.elapsed() >= LATENCY_INTERVAL {
                    asked = Instant::now();
                    let query = pa::Command::GetPlaybackLatency(pa::LatencyParams { channel: s.channel, now: SystemTime::now() });
                    pa::write_command_message(s.sock.get_mut(), LATENCY_SEQ, &query, s.version).map_err(fail)?;
                }
            }
            pa::Command::Reply if seq == LATENCY_SEQ => {
                let mut ts = pa::TagStructReader::new(&mut body, s.version);
                if let Ok(l) = ts.read::<pa::PlaybackLatency>() {
                    // What is in the server's stream buffer, plus the sink.
                    let buffered = (l.write_offset - l.read_offset).max(0) as f64 / frame_bytes / f64::from(s.format.rate);
                    info.device_delay = Some(Duration::from_micros(l.sink_usec) + Duration::from_secs_f64(buffered));
                }
            }
            pa::Command::Underflow(_) => {
                control.underruns.fetch_add(1, Relaxed);
            }
            pa::Command::PlaybackStreamMoved(m) => {
                *control.device.lock().unwrap() = m.device_name.to_string_lossy().into_owned();
            }
            pa::Command::PlaybackStreamKilled(_) => return Err(fail("the server ended the stream")),
            _ => {}
        }
    }
}

/// What the stream's thread is given to start with.
struct Start {
    /// The first connection, or `None` to start with no server.
    session: Option<Session>,
    render: Box<dyn Render>,
    control: Arc<Control>,
    server: Option<PathBuf>,
    app: String,
    request: OutputRequest,
}

fn run(start: Start) {
    let Start { mut session, mut render, control, server, app, request } = start;
    let mut format = match &session {
        Some(s) => s.format,
        None => DeviceFormat { rate: NULL_RATE, channels: usize::from(channel_map(request.channels).num_channels()) },
    };
    render.configure(format);
    loop {
        if let Some(s) = &mut session {
            let _ = serve(s, &mut *render, &control);
            control.orphan();
        }
        // There is no server. Play to nothing, and try for a connection
        // now and then, until there is one or the stream is stopped.
        let mut pacer = Pacer::new(format);
        let next = loop {
            let waited = Instant::now();
            while waited.elapsed() < RETRY {
                if control.stop.load(Relaxed) {
                    return;
                }
                std::thread::sleep(NULL_PERIOD);
                pacer.step(&mut *render, control.underruns.load(Relaxed));
            }
            match connect(server.as_deref(), &app, &request) {
                Ok(s) if control.adopt(&s) => break s,
                Ok(_) => return,
                Err(_) => {}
            }
        };
        if next.format != format {
            format = next.format;
            render.configure(format);
        }
        session = Some(next);
    }
}

struct PulseStream {
    control: Arc<Control>,
    thread: Option<JoinHandle<()>>,
}

impl StreamHandle for PulseStream {
    fn device(&self) -> String { self.control.device.lock().unwrap().clone() }
}

impl Drop for PulseStream {
    fn drop(&mut self) {
        {
            let sock = self.control.sock.lock().unwrap();
            self.control.stop.store(true, Relaxed);
            if let Some(s) = sock.as_ref() {
                let _ = s.shutdown(Shutdown::Both);
            }
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Backend for PulseBackend {
    fn name(&self) -> &'static str { "pulse" }

    fn open_output(&self, request: &OutputRequest, render: Box<dyn Render>) -> Result<Box<dyn StreamHandle>, Error> {
        // The first connection is made here, so that a refused cookie or a
        // refused stream is an error to the caller. So is having no server,
        // unless the caller asked for a stream regardless: a server may
        // start later, but one that refuses will refuse again.
        let session = match dial(self.server.as_deref()) {
            Ok(sock) => Some(open(sock, &self.app, request)?),
            Err(_) if request.null_fallback => None,
            Err(e) => return Err(e),
        };
        let control = Arc::new(Control {
            stop: AtomicBool::new(false),
            sock: Mutex::new(None),
            device: Mutex::new(String::new()),
            underruns: AtomicU64::new(0),
        });
        if let Some(session) = &session {
            control.adopt(session);
        }
        let start = Start {
            session,
            render,
            control: control.clone(),
            server: self.server.clone(),
            app: self.app.clone(),
            request: request.clone(),
        };
        let thread = std::thread::Builder::new().name("iris-audio pulse".into()).spawn(move || run(start))?;
        Ok(Box::new(PulseStream { control, thread: Some(thread) }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Audio, OutputConfig};

    /// A socket path with no server behind it.
    fn nowhere() -> PathBuf {
        std::env::temp_dir().join(format!("iris-audio-no-server-{}", std::process::id()))
    }

    #[test]
    fn no_server_is_an_error_unless_a_stream_is_wanted_regardless() {
        let audio = Audio::with_backend(Arc::new(PulseBackend::new("test").server(nowhere())));
        assert!(matches!(audio.open_output(OutputConfig::default()), Err(Error::Backend { backend: "pulse", .. })));
    }

    #[test]
    fn a_stream_opened_with_no_server_is_consumed_at_a_devices_pace() {
        let audio = Audio::with_backend(Arc::new(PulseBackend::new("test").server(nowhere())));
        let cfg = OutputConfig { rate: 22050, null_fallback: true, ..Default::default() };
        let mut out = audio.open_output(cfg).unwrap();
        assert_eq!(out.device().unwrap(), "");

        let start = Instant::now();
        let mut written = 0u64;
        // Long enough to cross an attempt to reconnect.
        while start.elapsed() < Duration::from_millis(1200) {
            let due = (start.elapsed().as_secs_f64() * 22050.0) as u64;
            out.write(&vec![0.25f32; (due - written) as usize * 2]);
            written = due;
            std::thread::sleep(Duration::from_millis(1));
        }
        let stats = out.stats();
        assert_eq!(stats.device, Some(DeviceFormat { rate: NULL_RATE, channels: 2 }));
        assert_eq!(stats.dropped, 0);
        let taken = stats.played * 22050 / u64::from(NULL_RATE);
        assert!(taken + 22050 / 5 >= stats.written, "{taken} of {} frames", stats.written);
    }
}
