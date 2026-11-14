//! The WASAPI backend: a shared-mode, event-driven render stream on the
//! default output device of Windows.
//!
//! One thread per stream owns every COM object the stream uses. The audio
//! engine sets an event each time it has taken a period from the stream's
//! buffer; the thread wakes, asks how much is still queued, and renders
//! enough to bring the queue back to its target. The target is two periods
//! unless the request asks for more, so the engine always has one period in
//! hand while the thread is being woken.
//!
//! Where the driver allows it the stream runs at a period shorter than the
//! engine's usual 10 ms (`IAudioClient3`); otherwise it runs at the
//! default. The stream is opened in the engine's own mix format, so the
//! only rate conversion on the way to the speaker is the renderer's. Two
//! cases are handed to Windows to convert instead: a mix format that is not
//! 32-bit float or has more channels than the library carries, and a guest
//! with more channels than the device, which Windows folds down properly.
//!
//! Windows does not move a running stream when the user picks another
//! default device. The backend registers for the notification, and when
//! the default changes the stream's thread closes its stream and opens one
//! on the new device, calling [`Render::configure`] if the format differs.
//! A device that is unplugged or disabled, or whose format is changed in
//! the control panel, invalidates the stream, which is handled the same
//! way.
//!
//! While there is no output device at all the thread asks for one twice a
//! second and on every notification, and stands in for the device
//! meanwhile: it takes the frames that become due by the host's clock, in
//! the format of the device that went, and discards them, as the
//! [null backend](super::null) does. The guest sees a device that never
//! went away, and the sound continues from where the guest is by then. The
//! first open is different: with no device then, [`Backend::open_output`]
//! fails, unless the request asks for the fallback
//! ([`OutputRequest::null_fallback`]). In that case the stream starts on
//! the stand-in, at the guest's rate and channels since there is no device
//! to take a format from, and moves to a device the same way.
//!
//! All streams of a process share one entry in the Windows volume mixer,
//! named after the program, so [`OutputRequest::name`] is not used.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Foundation::{CloseHandle, E_POINTER, HANDLE, PROPERTYKEY, WAIT_FAILED, WAIT_OBJECT_0};
use windows::Win32::Media::Audio::{
    AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
    AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, DEVICE_STATE, EDataFlow, ERole, IAudioClient, IAudioClient3, IAudioRenderClient,
    IMMDevice, IMMDeviceEnumerator, IMMNotificationClient, IMMNotificationClient_Impl, MMDeviceEnumerator, WAVEFORMATEX,
    WAVEFORMATEXTENSIBLE, WAVEFORMATEXTENSIBLE_0, eConsole, eRender,
};
use windows::Win32::System::Com::StructuredStorage::{PropVariantClear, PropVariantToStringAlloc};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, STGM_READ,
};
use windows::Win32::System::Threading::{
    AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW, CreateEventW, SetEvent, WaitForMultipleObjects,
    WaitForSingleObject,
};
use windows::core::{GUID, Interface, PCWSTR, PWSTR, implement, w};

use crate::Error;
use crate::backend::null::Pacer;
use crate::backend::{Backend, OutputRequest, Render, RenderInfo, StreamHandle};
use crate::format::{DeviceFormat, MAX_CHANNELS};

/// The device-side queue aimed for when the request names none. The period
/// asked of the engine is half of it; a driver that only runs at the
/// engine's default 10 ms period gives a 20 ms queue instead.
const DEFAULT_LATENCY: Duration = Duration::from_millis(10);
/// How many periods the queue holds at the least: the one the engine takes
/// next and one more to cover the time it takes this thread to wake.
const MIN_PERIODS: u32 = 2;
/// How long to wait between attempts to open a device when there is none,
/// and after a stream that failed as soon as it was opened.
const RETRY: Duration = Duration::from_millis(500);
/// How often frames are taken from the renderer and discarded while there
/// is no device to play them.
const NULL_PERIOD: Duration = Duration::from_millis(5);
/// If the engine has not asked for frames for this long, the stream is
/// probed for an error instead of being trusted to still be alive.
const WATCHDOG: Duration = Duration::from_millis(1000);
/// Which of the three default devices Windows keeps is followed: the one
/// for games and system sounds, as opposed to music or calls. The sound
/// settings change all of them together unless the user says otherwise.
const ROLE: ERole = eConsole;

const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xfffe;
const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: GUID = GUID::from_u128(0x00000003_0000_0010_8000_00aa00389b71);

pub struct WasapiBackend;

impl WasapiBackend {
    pub fn new() -> Self { Self }
}

impl Default for WasapiBackend {
    fn default() -> Self { Self::new() }
}

fn fail(e: impl std::fmt::Display) -> Error {
    Error::Backend { backend: "wasapi", message: e.to_string() }
}

/// An auto-reset event, closed when dropped.
struct Event(HANDLE);

// An event handle is a process-wide name for a kernel object; setting it
// and waiting on it are safe from any thread.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    fn new() -> windows::core::Result<Self> {
        unsafe { CreateEventW(None, false, false, PCWSTR::null()).map(Self) }
    }

    fn set(&self) {
        let _ = unsafe { SetEvent(self.0) };
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let _ = unsafe { CloseHandle(self.0) };
    }
}

/// What is known about devices outside the stream's thread.
#[derive(Default)]
struct Device {
    /// The endpoint the stream is open on; empty while there is none.
    id: String,
    /// Its name as the sound settings show it.
    name: String,
    /// The default endpoint according to the latest notification.
    wanted: Option<String>,
}

/// What the stream's thread, its handle and the notification share.
struct Control {
    stop: AtomicBool,
    /// The default device is no longer the one the stream is open on.
    changed: AtomicBool,
    /// Wakes the thread to look at `stop` and `changed`.
    signal: Event,
    device: Mutex<Device>,
}

impl Control {
    /// The lock is taken inside a COM callback, where a panic must not
    /// happen, so a poisoned lock is used as it is.
    fn device(&self) -> MutexGuard<'_, Device> {
        self.device.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Record the device a new stream is open on. A notification that came
    /// while it was being opened still counts if it named another device.
    fn adopt(&self, session: &Session) {
        let mut device = self.device();
        device.id.clone_from(&session.id);
        device.name.clone_from(&session.name);
        let moved = device.wanted.take().is_some_and(|wanted| wanted != device.id);
        self.changed.store(moved, Relaxed);
    }

    /// Record that there is no stream.
    fn lose(&self) {
        let mut device = self.device();
        device.id.clear();
        device.name.clear();
    }
}

/// Receives Windows' device notifications, on a thread of the system's.
/// It only compares names and wakes the stream's thread, which does the
/// work: every WASAPI call for a stream is made on that stream's thread.
#[implement(IMMNotificationClient)]
struct Notifier {
    control: Arc<Control>,
}

impl IMMNotificationClient_Impl for Notifier_Impl {
    fn OnDefaultDeviceChanged(&self, flow: EDataFlow, role: ERole, id: &PCWSTR) -> windows::core::Result<()> {
        if flow == eRender && role == ROLE {
            // A null name means the last output device has gone.
            let id = if id.is_null() { String::new() } else { unsafe { id.to_string() }.unwrap_or_default() };
            let mut device = self.control.device();
            // Windows repeats the notification for a device that is already
            // the default; reopening for it would be a gap for nothing.
            let moved = id != device.id;
            device.wanted = Some(id);
            drop(device);
            self.control.changed.store(moved, Relaxed);
            if moved {
                self.control.signal.set();
            }
        }
        Ok(())
    }

    fn OnDeviceStateChanged(&self, _id: &PCWSTR, _state: DEVICE_STATE) -> windows::core::Result<()> { Ok(()) }
    fn OnDeviceAdded(&self, _id: &PCWSTR) -> windows::core::Result<()> { Ok(()) }
    fn OnDeviceRemoved(&self, _id: &PCWSTR) -> windows::core::Result<()> { Ok(()) }
    fn OnPropertyValueChanged(&self, _id: &PCWSTR, _key: &PROPERTYKEY) -> windows::core::Result<()> { Ok(()) }
}

/// The device enumerator with the notification registered on it.
struct Host {
    enumerator: IMMDeviceEnumerator,
    notifier: IMMNotificationClient,
}

impl Host {
    unsafe fn new(control: &Arc<Control>) -> windows::core::Result<Self> {
        unsafe {
            let enumerator: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
            let notifier: IMMNotificationClient = Notifier { control: control.clone() }.into();
            enumerator.RegisterEndpointNotificationCallback(&notifier)?;
            Ok(Self { enumerator, notifier })
        }
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        let _ = unsafe { self.enumerator.UnregisterEndpointNotificationCallback(&self.notifier) };
    }
}

/// The thread's place in the multimedia scheduler's "Pro Audio" class,
/// given up when dropped.
struct Priority(HANDLE);

impl Priority {
    fn raise() -> Option<Self> {
        let mut task = 0;
        unsafe { AvSetMmThreadCharacteristicsW(w!("Pro Audio"), &mut task) }.ok().map(Self)
    }
}

impl Drop for Priority {
    fn drop(&mut self) {
        let _ = unsafe { AvRevertMmThreadCharacteristics(self.0) };
    }
}

/// A string Windows allocated for the caller, copied and freed.
unsafe fn take_string(s: PWSTR) -> String {
    if s.is_null() {
        return String::new();
    }
    unsafe {
        let text = s.to_string().unwrap_or_default();
        CoTaskMemFree(Some(s.as_ptr().cast()));
        text
    }
}

/// The device's name as the sound settings show it.
unsafe fn friendly_name(device: &IMMDevice) -> windows::core::Result<String> {
    unsafe {
        let store = device.OpenPropertyStore(STGM_READ)?;
        let mut value = store.GetValue(&PKEY_Device_FriendlyName)?;
        let name = PropVariantToStringAlloc(&value).map(|s| take_string(s));
        let _ = PropVariantClear(&mut value);
        name
    }
}

/// The engine's mix format, freed when dropped.
struct MixFormat(*mut WAVEFORMATEX);

impl MixFormat {
    /// The rate, the channel count, and whether samples are 32-bit float.
    fn describe(&self) -> (u32, usize, bool) {
        // The structures are packed, so they are copied out, not borrowed.
        let format = unsafe { self.0.read_unaligned() };
        let float = match format.wFormatTag {
            WAVE_FORMAT_IEEE_FLOAT => true,
            WAVE_FORMAT_EXTENSIBLE if format.cbSize >= 22 => {
                let subformat = unsafe { self.0.cast::<WAVEFORMATEXTENSIBLE>().read_unaligned() }.SubFormat;
                subformat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
            }
            _ => false,
        };
        (format.nSamplesPerSec, usize::from(format.nChannels), float && format.wBitsPerSample == 32)
    }
}

impl Drop for MixFormat {
    fn drop(&mut self) {
        unsafe { CoTaskMemFree(Some(self.0.cast())) };
    }
}

/// The layout a stream of `channels` is opened with when Windows is to
/// convert: the smallest standard one that holds them, in the order
/// [`OutputConfig::channels`] documents, as a channel count and a speaker
/// mask.
///
/// [`OutputConfig::channels`]: crate::OutputConfig::channels
fn layout(channels: usize) -> (u16, u32) {
    match channels {
        1 => (1, 0x4),      // front centre
        2 => (2, 0x3),      // front left and right
        3 | 4 => (4, 0x33), // and rear left and right
        5 | 6 => (6, 0x3f), // front, centre, LFE, rear
        _ => (8, 0x63f),    // and side left and right
    }
}

/// 32-bit float in the given layout.
fn float_format(channels: u16, mask: u32, rate: u32) -> WAVEFORMATEXTENSIBLE {
    WAVEFORMATEXTENSIBLE {
        Format: WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_EXTENSIBLE,
            nChannels: channels,
            nSamplesPerSec: rate,
            nAvgBytesPerSec: rate * u32::from(channels) * 4,
            nBlockAlign: channels * 4,
            wBitsPerSample: 32,
            cbSize: 22,
        },
        Samples: WAVEFORMATEXTENSIBLE_0 { wValidBitsPerSample: 32 },
        dwChannelMask: mask,
        SubFormat: KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
    }
}

/// A stream on one device.
struct Session {
    client: IAudioClient,
    render: IAudioRenderClient,
    /// Set by the engine each time it has taken a period.
    wake: Event,
    format: DeviceFormat,
    /// Frames the queue is topped up to on every wake.
    target: u32,
    /// What the engine and the device add after the stream's buffer.
    latency: Duration,
    id: String,
    name: String,
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = unsafe { self.client.Stop() };
    }
}

/// Initialise `client` in the mix format at the shortest period that suits
/// `want` frames of queue, and return the period. Fails on Windows before
/// 10 and wherever the engine refuses; the caller then opens the ordinary
/// way, on a fresh client.
unsafe fn low_latency(client: &IAudioClient, mix: &MixFormat, want: u32) -> windows::core::Result<u32> {
    unsafe {
        let client: IAudioClient3 = client.cast()?;
        let (mut default, mut fundamental, mut min, mut max) = (0, 0, 0, 0);
        client.GetSharedModeEnginePeriod(mix.0, &mut default, &mut fundamental, &mut min, &mut max)?;
        // Periods are multiples of the fundamental one. Nothing longer than
        // the default is wanted: it is what the ordinary way gives.
        let fundamental = fundamental.max(1);
        let period = (want / MIN_PERIODS / fundamental * fundamental).clamp(min, default.max(min));
        client.InitializeSharedAudioStream(AUDCLNT_STREAMFLAGS_EVENTCALLBACK, period, mix.0, None)?;
        Ok(period)
    }
}

/// Open a stream on the default output device, ready to be served.
unsafe fn open(enumerator: &IMMDeviceEnumerator, request: &OutputRequest) -> windows::core::Result<Session> {
    unsafe {
        let device = enumerator.GetDefaultAudioEndpoint(eRender, ROLE)?;
        let id = take_string(device.GetId()?);
        let name = friendly_name(&device).unwrap_or_else(|_| id.clone());
        let wake = Event::new()?;

        let first: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
        let mix = MixFormat(first.GetMixFormat()?);
        let (rate, mix_channels, float) = mix.describe();
        let latency = request.latency.unwrap_or(DEFAULT_LATENCY);
        let want = (latency.as_secs_f64() * f64::from(rate)) as u32;
        // The mix format is used as it is when the renderer can fill it and
        // loses nothing by mapping the guest's channels into it.
        let direct = float && (1..=MAX_CHANNELS).contains(&mix_channels) && request.channels <= mix_channels;

        let short = if direct { low_latency(&first, &mix, want).ok() } else { None };
        let (client, channels, period) = match short {
            Some(period) => (first, mix_channels, period),
            None => {
                // A client that failed to initialise is not reused.
                let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
                let mut period = 0i64; // In units of 100 ns, as the buffer's length below.
                client.GetDevicePeriod(Some(&mut period), None)?;
                let buffer = (latency.as_nanos() / 100).max(period as u128 * u128::from(MIN_PERIODS)) as i64;
                let period = (period as f64 * f64::from(rate) / 1e7).round() as u32;
                if direct {
                    client.Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, buffer, 0, mix.0, None)?;
                    (client, mix_channels, period)
                } else {
                    let (channels, mask) = layout(request.channels);
                    let format = float_format(channels, mask, rate);
                    let flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK
                        | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
                        | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
                    client.Initialize(AUDCLNT_SHAREMODE_SHARED, flags, buffer, 0, (&raw const format).cast(), None)?;
                    (client, usize::from(channels), period)
                }
            }
        };

        client.SetEventHandle(wake.0)?;
        let buffer = client.GetBufferSize()?;
        let render: IAudioRenderClient = client.GetService()?;
        let engine = client.GetStreamLatency().unwrap_or(0).max(0) as u64;
        Ok(Session {
            client,
            render,
            wake,
            format: DeviceFormat { rate, channels },
            target: want.max(period * MIN_PERIODS).min(buffer),
            latency: Duration::from_nanos(engine * 100),
            id,
            name,
        })
    }
}

/// Render `frames` straight into the stream's buffer.
unsafe fn fill(s: &Session, render: &mut dyn Render, info: &RenderInfo, frames: u32) -> windows::core::Result<()> {
    unsafe {
        let data = s.render.GetBuffer(frames)?.cast::<f32>();
        if data.is_null() || !data.is_aligned() {
            let _ = s.render.ReleaseBuffer(0, 0);
            return Err(E_POINTER.into());
        }
        let out = std::slice::from_raw_parts_mut(data, frames as usize * s.format.channels);
        render.render(out, info);
        s.render.ReleaseBuffer(frames, 0)
    }
}

/// Serve one stream until it fails (an error), or until the thread is told
/// to stop or that the default device has changed (no error).
unsafe fn serve(s: &Session, render: &mut dyn Render, control: &Control, underruns: &mut u64) -> windows::core::Result<()> {
    unsafe {
        let rate = f64::from(s.format.rate);
        let queued = |frames: u32| s.latency + Duration::from_secs_f64(f64::from(frames) / rate);
        let mut info = RenderInfo { device_delay: Some(queued(0)), device_underruns: *underruns };

        // The engine starts taking frames at once, so the queue is filled
        // before the stream is started.
        fill(s, render, &info, s.target)?;
        s.client.Start()?;
        let handles = [s.wake.0, control.signal.0];
        loop {
            let woke = WaitForMultipleObjects(&handles, false, WATCHDOG.as_millis() as u32);
            if control.stop.load(Relaxed) || control.changed.load(Relaxed) {
                return Ok(());
            }
            if woke == WAIT_FAILED {
                return Err(windows::core::Error::from_thread());
            }
            // After a wait that timed out, this is also what finds out that
            // the stream has died quietly.
            let padding = s.client.GetCurrentPadding()?;
            if padding == 0 {
                // The engine took the last frame there was: this thread
                // was at least a period late, and the device has very
                // likely played a gap.
                *underruns += 1;
            }
            let frames = s.target.saturating_sub(padding);
            if frames > 0 {
                info = RenderInfo { device_delay: Some(queued(padding)), device_underruns: *underruns };
                fill(s, render, &info, frames)?;
            }
        }
    }
}

/// The stream's thread, between initialising COM and leaving it.
unsafe fn stream(control: &Arc<Control>, request: &OutputRequest, render: &mut dyn Render, ready: SyncSender<Result<(), Error>>) {
    unsafe {
        // The first open happens here, and the caller waits for it, so
        // that a refused stream is an error to the caller. So is no device,
        // unless the caller asked for a stream regardless.
        let host = match Host::new(control) {
            Ok(host) => host,
            Err(e) => {
                let _ = ready.send(Err(fail(e.message())));
                return;
            }
        };
        let mut session = match open(&host.enumerator, request) {
            Ok(first) => {
                control.adopt(&first);
                Some(first)
            }
            Err(_) if request.null_fallback => None,
            Err(e) => {
                let _ = ready.send(Err(fail(e.message())));
                return;
            }
        };
        if ready.send(Ok(())).is_err() {
            return;
        }
        drop(ready);

        let _priority = Priority::raise();
        let mut format = match &session {
            Some(first) => first.format,
            // With no device to take a format from, the stand-in runs at
            // the guest's own.
            None => DeviceFormat { rate: request.rate.max(1), channels: request.channels.clamp(1, MAX_CHANNELS) },
        };
        render.configure(format);
        let mut pacer = None;
        let mut underruns = 0;
        loop {
            let mut broken = false;
            if let Some(s) = &session {
                let began = Instant::now();
                broken = serve(s, render, control, &mut underruns).is_err() && began.elapsed() < RETRY;
            }
            // From here on the stream is closed, so nothing below is in the
            // way of sound: it may take its time.
            session = None;
            if control.stop.load(Relaxed) {
                return;
            }
            if broken {
                // A stream that fails as soon as it starts is not retried
                // in a tight loop.
                idle(render, format, &mut pacer, control, underruns);
                if control.stop.load(Relaxed) {
                    return;
                }
            }
            match open(&host.enumerator, request) {
                Ok(s) => {
                    control.adopt(&s);
                    if s.format != format {
                        format = s.format;
                        render.configure(format);
                    }
                    // The next time without a device starts its own clock.
                    pacer = None;
                    session = Some(s);
                }
                Err(_) => {
                    // No device. A new default wakes the thread at once;
                    // anything else is found by asking again. Meanwhile
                    // the guest is still served.
                    control.lose();
                    idle(render, format, &mut pacer, control, underruns);
                }
            }
        }
    }
}

/// Stand in for a device that is not there: for [`RETRY`], or until a
/// notification or the order to stop, take the frames that become due from
/// the renderer and discard them. `pacer` is the clock of the whole stretch
/// without a device, made here when the stretch begins; `format` is what
/// the renderer was last configured with. No stream is open meanwhile, so
/// the renderer is this thread's to call.
unsafe fn idle(render: &mut dyn Render, format: DeviceFormat, pacer: &mut Option<Pacer>, control: &Control, underruns: u64) {
    let pacer = match pacer {
        Some(pacer) if pacer.format() == format => pacer,
        _ => pacer.insert(Pacer::new(format)),
    };
    let until = Instant::now() + RETRY;
    loop {
        pacer.step(render, underruns);
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() || control.stop.load(Relaxed) {
            return;
        }
        // The signal means a stop or a new default device: either way the
        // caller has something to do now.
        let wait = left.min(NULL_PERIOD).as_millis().max(1) as u32;
        if unsafe { WaitForSingleObject(control.signal.0, wait) } == WAIT_OBJECT_0 {
            return;
        }
    }
}

fn run(control: Arc<Control>, request: OutputRequest, mut render: Box<dyn Render>, ready: SyncSender<Result<(), Error>>) {
    unsafe {
        if let Err(e) = CoInitializeEx(None, COINIT_MULTITHREADED).ok() {
            let _ = ready.send(Err(fail(e.message())));
            return;
        }
        // Every COM object is released inside, before COM is left.
        stream(&control, &request, &mut *render, ready);
        CoUninitialize();
    }
}

struct WasapiStream {
    control: Arc<Control>,
    thread: Option<JoinHandle<()>>,
}

impl StreamHandle for WasapiStream {
    fn device(&self) -> String { self.control.device().name.clone() }
}

impl Drop for WasapiStream {
    fn drop(&mut self) {
        self.control.stop.store(true, Relaxed);
        self.control.signal.set();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Backend for WasapiBackend {
    fn name(&self) -> &'static str { "wasapi" }

    fn open_output(&self, request: &OutputRequest, render: Box<dyn Render>) -> Result<Box<dyn StreamHandle>, Error> {
        let control = Arc::new(Control {
            stop: AtomicBool::new(false),
            changed: AtomicBool::new(false),
            signal: Event::new().map_err(|e| fail(e.message()))?,
            device: Mutex::new(Device::default()),
        });
        let (ready, opened) = sync_channel(1);
        let (control_t, request) = (control.clone(), request.clone());
        let thread = std::thread::Builder::new()
            .name("iris-audio wasapi".into())
            .spawn(move || run(control_t, request, render, ready))?;
        match opened.recv() {
            Ok(Ok(())) => Ok(Box::new(WasapiStream { control, thread: Some(thread) })),
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(_) => {
                let _ = thread.join();
                Err(fail("the stream's thread ended before it opened a device"))
            }
        }
    }
}
