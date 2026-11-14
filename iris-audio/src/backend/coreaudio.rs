//! The CoreAudio backend: an output audio unit on the default output device
//! of macOS.
//!
//! The stream is a HAL output unit, the audio unit that plays to one
//! device. CoreAudio calls the unit's render callback on the device's own
//! real-time thread each time the device wants a buffer, and the callback
//! is the [`Render::render`] call: there is no queue between the renderer
//! and the device beyond the one buffer being filled. The unit is given
//! 32-bit float frames at the device's own rate, so the only rate
//! conversion on the way to the speaker is the renderer's.
//!
//! The size of that buffer is the device-side latency, and the backend sets
//! it: [`OutputRequest::latency`] if given, 5 ms otherwise, within what the
//! device allows. The size belongs to the process, not the stream: all
//! streams a process plays to one device are served in the same buffers,
//! and the last one opened decides their size.
//!
//! A HAL output unit stays on the device it was given. One thread per
//! stream owns the unit and makes every call that changes it. The system's
//! notifications (a new default output device, a device that has gone, a
//! change of its rate or channels) arrive on a thread of CoreAudio's and
//! only wake the stream's thread, which checks whether the stream is still
//! on the default device in the format it was opened with. If not, it
//! closes the unit and opens one on the device that is now the default,
//! calling [`Render::configure`] if the format differs.
//!
//! While there is no output device at all (the last one unplugged, or one
//! that will not start) the thread stands in for it: it goes on taking
//! frames from the renderer at the pace and in the format of the device
//! that went, and discards them, as the [null backend](super::null) does.
//! The guest notices nothing: its frames are consumed, none are dropped,
//! and when a device appears, which the thread looks for twice a second
//! and on every notification, the sound continues from where the guest is
//! by then. The first open is different: with no device then,
//! [`Backend::open_output`] fails, unless the request asks for the fallback
//! ([`OutputRequest::null_fallback`]). In that case the stream starts on
//! the stand-in, at the guest's rate and channels since there is no device
//! to take a format from, and moves to a device the same way.
//!
//! The device's channels are used as they are, up to the eight the library
//! carries, and the renderer maps the guest's frame into them. macOS has no
//! per-stream entry in a mixer, so [`OutputRequest::name`] is not used.
//!
//! The system's interface is declared here by hand (see `sys` below): the
//! part used is small, stable since Mac OS X 10.6, and lives in frameworks
//! every Mac has, so the backend adds no dependency.

use std::ffi::{CStr, c_void};
use std::mem::{MaybeUninit, size_of};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Once};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::Error;
use crate::backend::null::Pacer;
use crate::backend::{Backend, OutputRequest, Render, RenderInfo, StreamHandle};
use crate::format::{DeviceFormat, MAX_CHANNELS};

/// The device's buffer when the request names no latency. The system's own
/// default is 512 frames, 11 ms at the usual rates; half of that is still
/// far above what any Mac's hardware needs.
const DEFAULT_LATENCY: Duration = Duration::from_millis(5);
/// How long to wait between attempts to open a device when there is none,
/// and after a stream that failed as soon as it was opened.
const RETRY: Duration = Duration::from_millis(500);
/// A stream that has been asked for frames before, and then is not for
/// this long, is taken for dead and opened again. A stream that has not
/// been asked yet is left alone: some devices (Bluetooth, AirPlay) take
/// seconds to start.
const WATCHDOG: Duration = Duration::from_millis(2000);
/// How often frames are taken from the renderer and discarded while there
/// is no device to play them.
const NULL_PERIOD: Duration = Duration::from_millis(5);

/// The part of CoreAudio, the audio unit interface and CoreFoundation that
/// the backend uses, as Apple's headers declare it.
#[allow(non_snake_case, non_upper_case_globals)]
mod sys {
    use std::ffi::{c_char, c_void};

    pub type OSStatus = i32;
    pub type AudioObjectID = u32;
    pub type AudioComponent = *mut c_void;
    pub type AudioUnit = *mut c_void;
    pub type CFStringRef = *const c_void;

    /// Apple's four-character codes.
    pub const fn code(text: &[u8; 4]) -> u32 { u32::from_be_bytes(*text) }

    pub const kAudioObjectSystemObject: AudioObjectID = 1;
    pub const kAudioObjectPropertyScopeGlobal: u32 = code(b"glob");
    pub const kAudioObjectPropertyScopeOutput: u32 = code(b"outp");
    pub const kAudioObjectPropertyElementMain: u32 = 0;
    pub const kAudioObjectPropertyName: u32 = code(b"lnam");
    pub const kAudioHardwarePropertyDefaultOutputDevice: u32 = code(b"dOut");
    pub const kAudioHardwarePropertyRunLoop: u32 = code(b"rnlp");
    pub const kAudioDevicePropertyDeviceIsAlive: u32 = code(b"livn");
    pub const kAudioDevicePropertyNominalSampleRate: u32 = code(b"nsrt");
    pub const kAudioDevicePropertyStreamConfiguration: u32 = code(b"slay");
    pub const kAudioDevicePropertyBufferFrameSize: u32 = code(b"fsiz");
    pub const kAudioDevicePropertyBufferFrameSizeRange: u32 = code(b"fsz#");
    pub const kAudioDevicePropertyLatency: u32 = code(b"ltnc");
    pub const kAudioDevicePropertySafetyOffset: u32 = code(b"saft");
    pub const kAudioDevicePropertyStreams: u32 = code(b"stm#");
    pub const kAudioStreamPropertyLatency: u32 = code(b"ltnc");
    pub const kAudioDeviceProcessorOverload: u32 = code(b"over");

    pub const kAudioUnitType_Output: u32 = code(b"auou");
    pub const kAudioUnitSubType_HALOutput: u32 = code(b"ahal");
    pub const kAudioUnitManufacturer_Apple: u32 = code(b"appl");
    pub const kAudioUnitScope_Global: u32 = 0;
    pub const kAudioUnitScope_Input: u32 = 1;
    pub const kAudioUnitProperty_StreamFormat: u32 = 8;
    pub const kAudioUnitProperty_SetRenderCallback: u32 = 23;
    pub const kAudioOutputUnitProperty_CurrentDevice: u32 = 2000;

    pub const kAudioFormatLinearPCM: u32 = code(b"lpcm");
    pub const kAudioFormatFlagIsFloat: u32 = 1 << 0;
    pub const kAudioFormatFlagIsBigEndian: u32 = 1 << 1;
    pub const kAudioFormatFlagIsPacked: u32 = 1 << 3;
    pub const kAudioTimeStampHostTimeValid: u32 = 1 << 1;
    pub const kCFStringEncodingUTF8: u32 = 0x0800_0100;

    #[repr(C)]
    pub struct AudioObjectPropertyAddress {
        pub mSelector: u32,
        pub mScope: u32,
        pub mElement: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct AudioValueRange {
        pub mMinimum: f64,
        pub mMaximum: f64,
    }

    #[repr(C)]
    pub struct AudioStreamBasicDescription {
        pub mSampleRate: f64,
        pub mFormatID: u32,
        pub mFormatFlags: u32,
        pub mBytesPerPacket: u32,
        pub mFramesPerPacket: u32,
        pub mBytesPerFrame: u32,
        pub mChannelsPerFrame: u32,
        pub mBitsPerChannel: u32,
        pub mReserved: u32,
    }

    #[repr(C)]
    pub struct AudioBuffer {
        pub mNumberChannels: u32,
        pub mDataByteSize: u32,
        pub mData: *mut c_void,
    }

    /// Followed in memory by the rest of its buffers.
    #[repr(C)]
    pub struct AudioBufferList {
        pub mNumberBuffers: u32,
        pub mBuffers: [AudioBuffer; 1],
    }

    #[repr(C)]
    pub struct SMPTETime {
        pub mSubframes: i16,
        pub mSubframeDivisor: i16,
        pub mCounter: u32,
        pub mType: u32,
        pub mFlags: u32,
        pub mHours: i16,
        pub mMinutes: i16,
        pub mSeconds: i16,
        pub mFrames: i16,
    }

    #[repr(C)]
    pub struct AudioTimeStamp {
        pub mSampleTime: f64,
        pub mHostTime: u64,
        pub mRateScalar: f64,
        pub mWordClockTime: u64,
        pub mSMPTETime: SMPTETime,
        pub mFlags: u32,
        pub mReserved: u32,
    }

    #[repr(C)]
    pub struct AudioComponentDescription {
        pub componentType: u32,
        pub componentSubType: u32,
        pub componentManufacturer: u32,
        pub componentFlags: u32,
        pub componentFlagsMask: u32,
    }

    pub type AURenderCallback = unsafe extern "C" fn(
        refcon: *mut c_void,
        flags: *mut u32,
        stamp: *const AudioTimeStamp,
        bus: u32,
        frames: u32,
        data: *mut AudioBufferList,
    ) -> OSStatus;

    #[repr(C)]
    pub struct AURenderCallbackStruct {
        pub inputProc: AURenderCallback,
        pub inputProcRefCon: *mut c_void,
    }

    pub type AudioObjectPropertyListenerProc = unsafe extern "C" fn(
        object: AudioObjectID,
        count: u32,
        addresses: *const AudioObjectPropertyAddress,
        client: *mut c_void,
    ) -> OSStatus;

    #[repr(C)]
    pub struct mach_timebase_info_data_t {
        pub numer: u32,
        pub denom: u32,
    }

    #[link(name = "CoreAudio", kind = "framework")]
    unsafe extern "C" {
        pub fn AudioObjectGetPropertyDataSize(
            object: AudioObjectID,
            address: *const AudioObjectPropertyAddress,
            qualifier_size: u32,
            qualifier: *const c_void,
            size: *mut u32,
        ) -> OSStatus;
        pub fn AudioObjectGetPropertyData(
            object: AudioObjectID,
            address: *const AudioObjectPropertyAddress,
            qualifier_size: u32,
            qualifier: *const c_void,
            size: *mut u32,
            data: *mut c_void,
        ) -> OSStatus;
        pub fn AudioObjectSetPropertyData(
            object: AudioObjectID,
            address: *const AudioObjectPropertyAddress,
            qualifier_size: u32,
            qualifier: *const c_void,
            size: u32,
            data: *const c_void,
        ) -> OSStatus;
        pub fn AudioObjectAddPropertyListener(
            object: AudioObjectID,
            address: *const AudioObjectPropertyAddress,
            listener: AudioObjectPropertyListenerProc,
            client: *mut c_void,
        ) -> OSStatus;
        pub fn AudioObjectRemovePropertyListener(
            object: AudioObjectID,
            address: *const AudioObjectPropertyAddress,
            listener: AudioObjectPropertyListenerProc,
            client: *mut c_void,
        ) -> OSStatus;
    }

    #[link(name = "AudioToolbox", kind = "framework")]
    unsafe extern "C" {
        pub fn AudioComponentFindNext(after: AudioComponent, description: *const AudioComponentDescription) -> AudioComponent;
        pub fn AudioComponentInstanceNew(component: AudioComponent, instance: *mut AudioUnit) -> OSStatus;
        pub fn AudioComponentInstanceDispose(instance: AudioUnit) -> OSStatus;
        pub fn AudioUnitInitialize(unit: AudioUnit) -> OSStatus;
        pub fn AudioUnitUninitialize(unit: AudioUnit) -> OSStatus;
        pub fn AudioUnitSetProperty(
            unit: AudioUnit,
            property: u32,
            scope: u32,
            element: u32,
            data: *const c_void,
            size: u32,
        ) -> OSStatus;
        pub fn AudioOutputUnitStart(unit: AudioUnit) -> OSStatus;
        pub fn AudioOutputUnitStop(unit: AudioUnit) -> OSStatus;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub fn CFStringGetLength(string: CFStringRef) -> isize;
        pub fn CFStringGetMaximumSizeForEncoding(length: isize, encoding: u32) -> isize;
        pub fn CFStringGetCString(string: CFStringRef, buffer: *mut c_char, size: isize, encoding: u32) -> u8;
        pub fn CFRelease(object: *const c_void);
    }

    unsafe extern "C" {
        pub fn mach_absolute_time() -> u64;
        pub fn mach_timebase_info(info: *mut mach_timebase_info_data_t) -> i32;
    }
}

use sys::{AudioObjectID, AudioUnit, OSStatus};

pub struct CoreAudioBackend;

impl CoreAudioBackend {
    pub fn new() -> Self {
        // CoreAudio delivers its notifications on the run loop of the
        // process's main thread unless told otherwise, and a program whose
        // main thread runs no such loop (a command-line emulator) would
        // never hear of a device change. Naming no run loop makes CoreAudio
        // use a thread of its own, for the whole process.
        static OWN_THREAD: Once = Once::new();
        OWN_THREAD.call_once(|| unsafe {
            let none: *const c_void = null();
            let _ = set(sys::kAudioObjectSystemObject, sys::kAudioHardwarePropertyRunLoop, &none);
        });
        Self
    }
}

impl Default for CoreAudioBackend {
    fn default() -> Self { Self::new() }
}

/// A call that CoreAudio refused.
struct Failure {
    what: &'static str,
    status: OSStatus,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Most of CoreAudio's error codes are four characters.
        let text = self.status.to_be_bytes();
        match std::str::from_utf8(&text) {
            Ok(text) if text.bytes().all(|c| c.is_ascii_graphic()) => write!(f, "{} failed ('{text}')", self.what),
            _ => write!(f, "{} failed ({})", self.what, self.status),
        }
    }
}

fn check(what: &'static str, status: OSStatus) -> Result<(), Failure> {
    if status == 0 { Ok(()) } else { Err(Failure { what, status }) }
}

fn fail(e: impl std::fmt::Display) -> Error {
    Error::Backend { backend: "coreaudio", message: e.to_string() }
}

fn address(selector: u32, scope: u32) -> sys::AudioObjectPropertyAddress {
    sys::AudioObjectPropertyAddress { mSelector: selector, mScope: scope, mElement: sys::kAudioObjectPropertyElementMain }
}

/// Read a property of fixed size. `T` must be the plain data CoreAudio
/// documents for the property.
unsafe fn get<T: Copy>(object: AudioObjectID, selector: u32, scope: u32, what: &'static str) -> Result<T, Failure> {
    unsafe {
        let mut value = MaybeUninit::<T>::zeroed();
        let mut size = size_of::<T>() as u32;
        let status =
            sys::AudioObjectGetPropertyData(object, &address(selector, scope), 0, null(), &mut size, value.as_mut_ptr().cast());
        check(what, status)?;
        if size as usize != size_of::<T>() {
            return Err(Failure { what, status: -1 });
        }
        Ok(value.assume_init())
    }
}

/// Write a property of global scope.
unsafe fn set<T: Copy>(object: AudioObjectID, selector: u32, value: &T) -> OSStatus {
    unsafe {
        let address = address(selector, sys::kAudioObjectPropertyScopeGlobal);
        sys::AudioObjectSetPropertyData(object, &address, 0, null(), size_of::<T>() as u32, (&raw const *value).cast())
    }
}

/// The device the system plays to by default.
unsafe fn default_output() -> Result<AudioObjectID, Failure> {
    unsafe {
        let what = "finding the default output device";
        let device: AudioObjectID = get(
            sys::kAudioObjectSystemObject,
            sys::kAudioHardwarePropertyDefaultOutputDevice,
            sys::kAudioObjectPropertyScopeGlobal,
            what,
        )?;
        // Zero is how CoreAudio says that there is none.
        if device == 0 { Err(Failure { what, status: sys::code(b"!dev") as OSStatus }) } else { Ok(device) }
    }
}

/// The rate the device runs at and how many channels it plays.
unsafe fn device_format(device: AudioObjectID) -> Result<(f64, usize), Failure> {
    unsafe {
        let rate: f64 =
            get(device, sys::kAudioDevicePropertyNominalSampleRate, sys::kAudioObjectPropertyScopeGlobal, "reading the device's rate")?;

        // The device's output streams, as a list of buffers whose length is
        // only known by asking; each buffer is one stream of some channels.
        let what = "reading the device's channels";
        let address = address(sys::kAudioDevicePropertyStreamConfiguration, sys::kAudioObjectPropertyScopeOutput);
        let mut size = 0u32;
        check(what, sys::AudioObjectGetPropertyDataSize(device, &address, 0, null(), &mut size))?;
        // Words, so that the list is aligned.
        let mut list = vec![0u64; (size as usize).div_ceil(8).max(1)];
        check(what, sys::AudioObjectGetPropertyData(device, &address, 0, null(), &mut size, list.as_mut_ptr().cast()))?;
        let header = size_of::<sys::AudioBufferList>() - size_of::<sys::AudioBuffer>();
        let count = (list[0] as u32 as usize).min((size as usize).saturating_sub(header) / size_of::<sys::AudioBuffer>());
        let buffers = list.as_ptr().cast::<u8>().add(header).cast::<sys::AudioBuffer>();
        let channels = (0..count).map(|i| (*buffers.add(i)).mNumberChannels as usize).sum();
        Ok((rate, channels))
    }
}

/// The device's name as the sound settings show it.
unsafe fn device_name(device: AudioObjectID) -> Option<String> {
    unsafe {
        let name: sys::CFStringRef =
            get(device, sys::kAudioObjectPropertyName, sys::kAudioObjectPropertyScopeGlobal, "reading the device's name").ok()?;
        if name.is_null() {
            return None;
        }
        let size = sys::CFStringGetMaximumSizeForEncoding(sys::CFStringGetLength(name), sys::kCFStringEncodingUTF8) + 1;
        let mut text = vec![0u8; size.max(1) as usize];
        let copied = sys::CFStringGetCString(name, text.as_mut_ptr().cast(), text.len() as isize, sys::kCFStringEncodingUTF8);
        // The caller of the property owns the string it returns.
        sys::CFRelease(name);
        if copied == 0 {
            return None;
        }
        Some(CStr::from_bytes_until_nul(&text).ok()?.to_string_lossy().into_owned())
    }
}

/// Frames of delay between the device taking a frame and the speaker
/// playing it: the device's own latency and that of its first stream.
unsafe fn device_latency(device: AudioObjectID) -> u32 {
    unsafe {
        let output = sys::kAudioObjectPropertyScopeOutput;
        let device_frames: u32 = get(device, sys::kAudioDevicePropertyLatency, output, "").unwrap_or(0);
        // Only the first of the device's streams is asked: it is the one
        // that plays the front channels.
        let stream: Option<AudioObjectID> = get(device, sys::kAudioDevicePropertyStreams, output, "").ok();
        let stream_frames: u32 = stream
            .and_then(|s| get(s, sys::kAudioStreamPropertyLatency, sys::kAudioObjectPropertyScopeGlobal, "").ok())
            .unwrap_or(0);
        device_frames + stream_frames
    }
}

/// What the stream's thread, its handle and the notifications share.
struct Control {
    stop: AtomicBool,
    /// Something about the devices has changed and the stream may no
    /// longer be where it should be.
    changed: AtomicBool,
    /// The name of the device the stream is open on; empty while there is
    /// none. Its lock is also the one the thread sleeps under.
    name: Mutex<String>,
    wake: Condvar,
    /// Times CoreAudio reported that a buffer was not filled in time.
    overloads: AtomicU64,
    /// Buffers rendered since the stream opened, for the watchdog.
    calls: AtomicU64,
}

impl Control {
    /// The lock is taken inside callbacks from the system, where a panic
    /// must not happen, so a poisoned lock is used as it is.
    fn name(&self) -> MutexGuard<'_, String> {
        self.name.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Set a flag and wake the thread. The flag is set under the lock, so
    /// the thread cannot miss it between looking and falling asleep.
    fn signal(&self, flag: &AtomicBool) {
        let _held = self.name();
        flag.store(true, Relaxed);
        self.wake.notify_all();
    }

    /// Sleep until a flag is set, or for `timeout`.
    fn wait(&self, timeout: Duration) {
        let held = self.name();
        if !self.stop.load(Relaxed) && !self.changed.load(Relaxed) {
            let _ = self.wake.wait_timeout(held, timeout);
        }
    }
}

/// The streams that notifications are delivered to, by the number each
/// listener is registered with. CoreAudio does not promise that a listener
/// is not running, or about to, when its removal returns; a listener that
/// arrives late finds no stream under its number, where a pointer handed
/// to CoreAudio instead would by then point at freed memory.
static STREAMS: Mutex<Vec<(usize, Arc<Control>)>> = Mutex::new(Vec::new());
static NEXT_TOKEN: AtomicUsize = AtomicUsize::new(1);

/// A stream's entry in [`STREAMS`], removed when dropped.
struct Token(usize);

impl Token {
    fn new(control: &Arc<Control>) -> Self {
        let token = NEXT_TOKEN.fetch_add(1, Relaxed);
        STREAMS.lock().unwrap_or_else(|e| e.into_inner()).push((token, control.clone()));
        Self(token)
    }

    fn client(&self) -> *mut c_void { self.0 as *mut c_void }

    fn find(client: *mut c_void) -> Option<Arc<Control>> {
        let streams = STREAMS.lock().unwrap_or_else(|e| e.into_inner());
        streams.iter().find(|(token, _)| *token == client as usize).map(|(_, control)| control.clone())
    }
}

impl Drop for Token {
    fn drop(&mut self) {
        STREAMS.lock().unwrap_or_else(|e| e.into_inner()).retain(|(token, _)| *token != self.0);
    }
}

/// Called by CoreAudio, on a thread of its own, when the default output
/// device or something about the stream's device has changed. It only
/// wakes the stream's thread, which looks at what happened.
unsafe extern "C" fn on_change(
    _object: AudioObjectID,
    _count: u32,
    _addresses: *const sys::AudioObjectPropertyAddress,
    client: *mut c_void,
) -> OSStatus {
    if let Some(control) = Token::find(client) {
        control.signal(&control.changed);
    }
    0
}

/// Called by CoreAudio when the device's thread took too long over a
/// buffer and the device played a gap.
unsafe extern "C" fn on_overload(
    _object: AudioObjectID,
    _count: u32,
    _addresses: *const sys::AudioObjectPropertyAddress,
    client: *mut c_void,
) -> OSStatus {
    if let Some(control) = Token::find(client) {
        control.overloads.fetch_add(1, Relaxed);
    }
    0
}

/// A notification asked of CoreAudio, taken back when dropped.
struct Listener {
    object: AudioObjectID,
    address: sys::AudioObjectPropertyAddress,
    listener: sys::AudioObjectPropertyListenerProc,
    client: *mut c_void,
}

impl Listener {
    unsafe fn new(
        object: AudioObjectID,
        selector: u32,
        scope: u32,
        listener: sys::AudioObjectPropertyListenerProc,
        token: &Token,
    ) -> Result<Self, Failure> {
        let address = address(selector, scope);
        let status = unsafe { sys::AudioObjectAddPropertyListener(object, &address, listener, token.client()) };
        check("asking for device notifications", status)?;
        Ok(Self { object, address, listener, client: token.client() })
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        let _ = unsafe { sys::AudioObjectRemovePropertyListener(self.object, &self.address, self.listener, self.client) };
    }
}

/// What the render callback works with. The stream's thread sets it up
/// while no unit is running; from the start of a unit to its stop it
/// belongs to the device's thread alone.
struct Feed {
    render: Box<dyn Render>,
    control: Arc<Control>,
    /// Samples per frame of the buffers the unit asks for.
    channels: usize,
    /// What the device adds after the time it gives for a buffer.
    after: Duration,
    /// The delay to report for a buffer that comes without a time.
    nominal: Duration,
    /// Nanoseconds per tick of the host's clock, as a fraction.
    tick: (u64, u64),
}

/// The unit's render callback, on the device's real-time thread.
unsafe extern "C" fn on_render(
    refcon: *mut c_void,
    _flags: *mut u32,
    stamp: *const sys::AudioTimeStamp,
    _bus: u32,
    frames: u32,
    data: *mut sys::AudioBufferList,
) -> OSStatus {
    unsafe {
        let feed = &mut *refcon.cast::<Feed>();
        let samples = frames as usize * feed.channels;
        let (out, size) = match (*data).mNumberBuffers {
            1 => ((*data).mBuffers[0].mData.cast::<f32>(), (*data).mBuffers[0].mDataByteSize as usize),
            _ => (null_mut(), 0),
        };
        if out.is_null() || !out.is_aligned() || size < samples * 4 {
            // Not the interleaved buffer that was asked for. This is not
            // known to happen; silence is the safe answer if it does.
            for i in 0..(*data).mNumberBuffers as usize {
                let buffer = (&raw mut (*data).mBuffers).cast::<sys::AudioBuffer>().add(i);
                if !(*buffer).mData.is_null() {
                    (*buffer).mData.cast::<u8>().write_bytes(0, (*buffer).mDataByteSize as usize);
                }
            }
            return 0;
        }

        // The buffer comes with the time on the host's clock at which the
        // device will take its first frame.
        let ahead = if (*stamp).mFlags & sys::kAudioTimeStampHostTimeValid != 0 {
            let ticks = (*stamp).mHostTime.saturating_sub(sys::mach_absolute_time());
            Duration::from_nanos((u128::from(ticks) * u128::from(feed.tick.0) / u128::from(feed.tick.1)) as u64)
        } else {
            feed.nominal
        };
        let info =
            RenderInfo { device_delay: Some(ahead + feed.after), device_underruns: feed.control.overloads.load(Relaxed) };
        feed.control.calls.fetch_add(1, Relaxed);
        feed.render.render(std::slice::from_raw_parts_mut(out, samples), &info);
        0
    }
}

/// An audio unit on one device.
struct Session {
    unit: AudioUnit,
    device: AudioObjectID,
    /// What the device ran at when the unit was opened.
    native: (f64, usize),
    /// What the unit asks the renderer for.
    format: DeviceFormat,
    after: Duration,
    nominal: Duration,
    name: String,
    /// Dropped, and so taken back, before the unit is closed.
    listeners: Vec<Listener>,
}

impl Session {
    /// Whether the stream is still where it should be: on the default
    /// device, and that device as it was when the stream was opened.
    unsafe fn current(&self) -> bool {
        unsafe {
            let alive = || get::<u32>(self.device, sys::kAudioDevicePropertyDeviceIsAlive, sys::kAudioObjectPropertyScopeGlobal, "");
            default_output().is_ok_and(|device| device == self.device)
                && alive().is_ok_and(|alive| alive != 0)
                && device_format(self.device).is_ok_and(|native| native == self.native)
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            // Stopping waits for a render callback that is running, so the
            // device's thread is out of the `Feed` when this returns.
            let _ = sys::AudioOutputUnitStop(self.unit);
            self.listeners.clear();
            let _ = sys::AudioUnitUninitialize(self.unit);
            let _ = sys::AudioComponentInstanceDispose(self.unit);
        }
    }
}

/// Open a unit on the default output device, set up but not started.
unsafe fn open(request: &OutputRequest, feed: *mut Feed, token: &Token) -> Result<Session, Failure> {
    unsafe {
        let device = default_output()?;
        let native = device_format(device)?;
        let (rate, device_channels) = native;
        if !(rate >= 1.0 && rate <= f64::from(u32::MAX)) || device_channels == 0 {
            return Err(Failure { what: "reading the device's format", status: sys::code(b"!fmt") as OSStatus });
        }
        let format = DeviceFormat { rate: rate.round() as u32, channels: device_channels.min(MAX_CHANNELS) };

        // The buffer the device asks for at a time is the latency on the
        // device's side. A device that refuses the size keeps its own.
        let global = sys::kAudioObjectPropertyScopeGlobal;
        let want = request.latency.unwrap_or(DEFAULT_LATENCY).as_secs_f64() * rate;
        let range = get::<sys::AudioValueRange>(device, sys::kAudioDevicePropertyBufferFrameSizeRange, global, "");
        if let Ok(range) = range
            && range.mMinimum <= range.mMaximum
        {
            let frames = want.clamp(range.mMinimum.max(1.0), range.mMaximum.max(1.0)).round() as u32;
            let _ = set(device, sys::kAudioDevicePropertyBufferFrameSize, &frames);
        }
        let period: u32 = get(device, sys::kAudioDevicePropertyBufferFrameSize, global, "").unwrap_or(want as u32);
        let safety: u32 = get(device, sys::kAudioDevicePropertySafetyOffset, sys::kAudioObjectPropertyScopeOutput, "").unwrap_or(0);
        let after = device_latency(device);
        let time = |frames: u32| Duration::from_secs_f64(f64::from(frames) / rate);

        let description = sys::AudioComponentDescription {
            componentType: sys::kAudioUnitType_Output,
            componentSubType: sys::kAudioUnitSubType_HALOutput,
            componentManufacturer: sys::kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0,
        };
        let component = sys::AudioComponentFindNext(null_mut(), &description);
        if component.is_null() {
            return Err(Failure { what: "finding the output audio unit", status: -1 });
        }
        let mut unit: AudioUnit = null_mut();
        check("creating the output audio unit", sys::AudioComponentInstanceNew(component, &mut unit))?;
        // From here on dropping the session closes the unit.
        let mut session = Session {
            unit,
            device,
            native,
            format,
            after: time(after),
            nominal: time(period + safety + after),
            name: device_name(device).unwrap_or_else(|| format!("device {device}")),
            listeners: Vec::new(),
        };

        let property = |property: u32, scope: u32, data: *const c_void, size: usize| {
            sys::AudioUnitSetProperty(unit, property, scope, 0, data, size as u32)
        };
        check(
            "giving the audio unit its device",
            property(
                sys::kAudioOutputUnitProperty_CurrentDevice,
                sys::kAudioUnitScope_Global,
                (&raw const device).cast(),
                size_of::<AudioObjectID>(),
            ),
        )?;
        // What the unit is fed: its input, in the sense of the property.
        let channels = format.channels as u32;
        let fed = sys::AudioStreamBasicDescription {
            mSampleRate: rate,
            mFormatID: sys::kAudioFormatLinearPCM,
            mFormatFlags: sys::kAudioFormatFlagIsFloat
                | sys::kAudioFormatFlagIsPacked
                | if cfg!(target_endian = "big") { sys::kAudioFormatFlagIsBigEndian } else { 0 },
            mBytesPerPacket: 4 * channels,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4 * channels,
            mChannelsPerFrame: channels,
            mBitsPerChannel: 32,
            mReserved: 0,
        };
        check(
            "setting the stream's format",
            property(
                sys::kAudioUnitProperty_StreamFormat,
                sys::kAudioUnitScope_Input,
                (&raw const fed).cast(),
                size_of::<sys::AudioStreamBasicDescription>(),
            ),
        )?;
        let callback = sys::AURenderCallbackStruct { inputProc: on_render, inputProcRefCon: feed.cast() };
        check(
            "setting the render callback",
            property(
                sys::kAudioUnitProperty_SetRenderCallback,
                sys::kAudioUnitScope_Input,
                (&raw const callback).cast(),
                size_of::<sys::AURenderCallbackStruct>(),
            ),
        )?;

        // A stream that cannot hear of its device going would play to
        // nothing for ever, so that notification is required; the rest are
        // taken if they can be had.
        session.listeners.push(Listener::new(device, sys::kAudioDevicePropertyDeviceIsAlive, global, on_change, token)?);
        let wanted = [
            (sys::kAudioDevicePropertyNominalSampleRate, global, on_change as sys::AudioObjectPropertyListenerProc),
            (sys::kAudioDevicePropertyStreamConfiguration, sys::kAudioObjectPropertyScopeOutput, on_change),
            (sys::kAudioDeviceProcessorOverload, global, on_overload),
        ];
        for (selector, scope, listener) in wanted {
            session.listeners.extend(Listener::new(device, selector, scope, listener, token).ok());
        }
        Ok(session)
    }
}

/// Start a unit that [`open`] set up. `format` is what the renderer was
/// last configured with.
unsafe fn start(s: &Session, feed: *mut Feed, format: &mut Option<DeviceFormat>) -> Result<(), Failure> {
    unsafe {
        // No unit is running, so the feed is this thread's to change.
        let feed = &mut *feed;
        if *format != Some(s.format) {
            feed.render.configure(s.format);
            *format = Some(s.format);
        }
        feed.channels = s.format.channels;
        feed.after = s.after;
        feed.nominal = s.nominal;
        check("initialising the audio unit", sys::AudioUnitInitialize(s.unit))?;
        check("starting the audio unit", sys::AudioOutputUnitStart(s.unit))
    }
}

/// Watch over a running stream until it has to be opened again, or the
/// thread is told to stop.
unsafe fn watch(s: &Session, control: &Control) {
    let mut seen = control.calls.load(Relaxed);
    let (mut running, mut checked) = (false, Instant::now());
    loop {
        control.wait(WATCHDOG);
        if control.stop.load(Relaxed) {
            return;
        }
        // Most notifications are of no consequence: a device that is not
        // the stream's coming or going, or a change repeated.
        if control.changed.swap(false, Relaxed) && !unsafe { s.current() } {
            return;
        }
        if checked.elapsed() >= WATCHDOG {
            let calls = control.calls.load(Relaxed);
            if running && calls == seen {
                return;
            }
            running |= calls != seen;
            (seen, checked) = (calls, Instant::now());
        }
    }
}

/// Stand in for the device while there is none: take the frames that are
/// due from the renderer and discard them, until `until`, a notification or
/// the order to stop. `pacer` is the clock of the whole stretch without a
/// device, made here when the stretch begins.
unsafe fn idle(feed: *mut Feed, format: DeviceFormat, pacer: &mut Option<Pacer>, control: &Control, until: Instant) {
    // No unit is running, so the feed is this thread's to use.
    let feed = unsafe { &mut *feed };
    let pacer = match pacer {
        Some(pacer) if pacer.format() == format => pacer,
        _ => pacer.insert(Pacer::new(format)),
    };
    loop {
        pacer.step(&mut *feed.render, control.overloads.load(Relaxed));
        let now = Instant::now();
        if control.stop.load(Relaxed) || control.changed.load(Relaxed) || now >= until {
            return;
        }
        control.wait(NULL_PERIOD.min(until - now));
    }
}

/// The stream's thread.
unsafe fn stream(control: &Arc<Control>, request: &OutputRequest, render: Box<dyn Render>, ready: SyncSender<Result<(), Error>>) {
    unsafe {
        let mut tick = sys::mach_timebase_info_data_t { numer: 1, denom: 1 };
        if sys::mach_timebase_info(&mut tick) != 0 || tick.denom == 0 {
            tick = sys::mach_timebase_info_data_t { numer: 1, denom: 1 };
        }
        // Declared before everything that hands its address to CoreAudio,
        // so it is dropped after all of it.
        let mut feed = Box::new(Feed {
            render,
            control: control.clone(),
            channels: 0,
            after: Duration::ZERO,
            nominal: Duration::ZERO,
            tick: (u64::from(tick.numer), u64::from(tick.denom)),
        });
        let feed: *mut Feed = &raw mut *feed;
        let token = Token::new(control);
        let mut format = None;

        // The first open happens here, and the caller waits for it, so
        // that no device or a refused stream is an error to the caller,
        // unless the caller asked to be given a stream regardless.
        let default = Listener::new(
            sys::kAudioObjectSystemObject,
            sys::kAudioHardwarePropertyDefaultOutputDevice,
            sys::kAudioObjectPropertyScopeGlobal,
            on_change,
            &token,
        );
        let _default = match default {
            Ok(listener) => listener,
            Err(e) => {
                let _ = ready.send(Err(fail(e)));
                return;
            }
        };
        let first = match open(request, feed, &token).and_then(|s| start(&s, feed, &mut format).map(|()| s)) {
            Ok(first) => {
                control.name().clone_from(&first.name);
                Some(first)
            }
            Err(_) if request.null_fallback => {
                // With no device to take a format from, the stand-in runs
                // at the guest's own. A device that was opened but would
                // not start has set the format already.
                if format.is_none() {
                    let guest = DeviceFormat { rate: request.rate.max(1), channels: request.channels.clamp(1, MAX_CHANNELS) };
                    (*feed).render.configure(guest);
                    format = Some(guest);
                }
                None
            }
            Err(e) => {
                let _ = ready.send(Err(fail(e)));
                return;
            }
        };
        if ready.send(Ok(())).is_err() {
            return;
        }
        drop(ready);

        let mut session = first;
        let mut pacer = None;
        loop {
            if let Some(s) = &session {
                watch(s, control);
            }
            // From here on the stream is closed, so nothing below is in the
            // way of sound: it may take its time.
            session = None;
            if control.stop.load(Relaxed) {
                return;
            }
            // A change from here on is one the new stream has to look at.
            control.changed.store(false, Relaxed);
            let began = Instant::now();
            match open(request, feed, &token).and_then(|s| start(&s, feed, &mut format).map(|()| s)) {
                Ok(s) => {
                    control.name().clone_from(&s.name);
                    session = Some(s);
                    pacer = None;
                }
                Err(_) => {
                    // No device, or one that will not start. A notification
                    // ends the wait at once; anything else is found by
                    // asking again. Meanwhile the guest is still served.
                    control.name().clear();
                    match format {
                        Some(format) => idle(feed, format, &mut pacer, control, began + RETRY),
                        None => control.wait(RETRY.saturating_sub(began.elapsed())),
                    }
                }
            }
        }
    }
}

struct CoreAudioStream {
    control: Arc<Control>,
    thread: Option<JoinHandle<()>>,
}

impl StreamHandle for CoreAudioStream {
    fn device(&self) -> String { self.control.name().clone() }
}

impl Drop for CoreAudioStream {
    fn drop(&mut self) {
        self.control.signal(&self.control.stop);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Backend for CoreAudioBackend {
    fn name(&self) -> &'static str { "coreaudio" }

    fn open_output(&self, request: &OutputRequest, render: Box<dyn Render>) -> Result<Box<dyn StreamHandle>, Error> {
        let control = Arc::new(Control {
            stop: AtomicBool::new(false),
            changed: AtomicBool::new(false),
            name: Mutex::new(String::new()),
            wake: Condvar::new(),
            overloads: AtomicU64::new(0),
            calls: AtomicU64::new(0),
        });
        let (ready, opened) = sync_channel(1);
        let (control_t, request) = (control.clone(), request.clone());
        let thread = std::thread::Builder::new()
            .name("iris-audio coreaudio".into())
            .spawn(move || unsafe { stream(&control_t, &request, render, ready) })?;
        match opened.recv() {
            Ok(Ok(())) => Ok(Box::new(CoreAudioStream { control, thread: Some(thread) })),
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
