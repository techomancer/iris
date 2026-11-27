//! MIDI on a guest serial port: `[midi]`.
//!
//! The guest's serial MIDI is a byte stream (see `docs/peripherals/midi.md`).
//! [`MidiParser`] turns it back into whole messages: running status, real-time
//! bytes that may arrive anywhere, even inside another message, and SysEx.
//! [`MidiSynth`] plays the channel messages on rustysynth from a SoundFont,
//! into an output stream of its own on the host's audio, so a guest with no
//! MIDI hardware behind it still sounds. The synth sends nothing back.
//!
//! The synth renders on a thread of its own in short blocks paced by the host
//! clock. A message plays at the start of the block after it arrived, a few
//! milliseconds late at most, which keeps the emulator's uneven pace out of
//! the rhythm.

use std::collections::VecDeque;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::dev::serial_host::SerialBackend;

/// One MIDI message, as the parser puts it back together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MidiMessage {
    /// A channel message: the status byte (command and channel) and its data
    /// (the second byte is 0 for the one-byte kinds).
    Channel { status: u8, data1: u8, data2: u8 },
    /// System common (F1 to F6) with its data.
    Common { status: u8, data: [u8; 2] },
    /// A real-time byte (F8 to FF).
    RealTime(u8),
    /// A whole SysEx, F0 and F7 included.
    SysEx(Vec<u8>),
}

/// The longest SysEx kept; a longer one is dropped whole.
const SYSEX_MAX: usize = 64 * 1024;

/// Data bytes a status takes.
fn data_len(status: u8) -> usize {
    match status {
        0x80..=0xbf | 0xe0..=0xef => 2,
        0xc0..=0xdf => 1,
        0xf1 | 0xf3 => 1,
        0xf2 => 2,
        _ => 0,
    }
}

/// Byte stream to messages.
#[derive(Default)]
pub struct MidiParser {
    /// The status the data bytes belong to: a channel status stays for
    /// running status, a system common one only for its own message.
    status: Option<u8>,
    data: [u8; 2],
    have: usize,
    sysex: Option<Vec<u8>>,
    /// The SysEx in progress grew past [`SYSEX_MAX`] and is being dropped.
    sysex_dropping: bool,
}

impl MidiParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take one byte; a message when it completes one.
    pub fn push(&mut self, b: u8) -> Option<MidiMessage> {
        if b >= 0xf8 {
            // Real-time: anywhere, and it disturbs nothing.
            return Some(MidiMessage::RealTime(b));
        }
        if b & 0x80 != 0 {
            // A status byte ends a SysEx (F7 properly, any other status as well).
            let ended = self.sysex.take().filter(|_| !std::mem::take(&mut self.sysex_dropping)).map(|mut s| {
                s.push(0xf7);
                s
            });
            if b == 0xf7 {
                return ended.map(MidiMessage::SysEx);
            }
            self.have = 0;
            self.data = [0; 2];
            match b {
                0xf0 => {
                    self.status = None;
                    self.sysex = Some(vec![0xf0]);
                    return ended.map(MidiMessage::SysEx);
                }
                0xf4 | 0xf5 => {
                    self.status = None;
                    return ended.map(MidiMessage::SysEx);
                }
                0xf6 => {
                    self.status = None;
                    return Some(ended.map_or(MidiMessage::Common { status: 0xf6, data: [0; 2] }, MidiMessage::SysEx));
                }
                _ => {
                    self.status = Some(b);
                    // A message cut short by this status is lost; a SysEx it ends is not.
                    return ended.map(MidiMessage::SysEx);
                }
            }
        }
        // A data byte.
        if let Some(s) = self.sysex.as_mut() {
            if s.len() < SYSEX_MAX {
                s.push(b);
            } else {
                self.sysex_dropping = true;
            }
            return None;
        }
        let status = self.status?;
        self.data[self.have] = b;
        self.have += 1;
        if self.have < data_len(status) {
            return None;
        }
        self.have = 0;
        if status >= 0xf0 {
            self.status = None;
            return Some(MidiMessage::Common { status, data: self.data });
        }
        let data2 = if data_len(status) == 2 { self.data[1] } else { 0 };
        Some(MidiMessage::Channel { status, data1: self.data[0], data2 })
    }
}

/// The synth's output rate.
const RATE: u32 = 44_100;
/// How often the synth renders, and so how late a message may play.
const BLOCK: Duration = Duration::from_millis(5);
/// The most the synth renders at once after it fell behind (a stall, a
/// stopped machine): the lost time is skipped, not caught up.
const MAX_CATCH_UP: u64 = (RATE / 10) as u64;

/// The GM System On and GS Reset messages, which reset the synth.
const GM_ON: &[u8] = &[0xf0, 0x7e, 0x7f, 0x09, 0x01, 0xf7];
const GS_RESET: &[u8] = &[0xf0, 0x41, 0x10, 0x42, 0x12, 0x40, 0x00, 0x7f, 0x00, 0x41, 0xf7];

struct Shared {
    /// Messages for the synth, in arrival order.
    queue: Mutex<VecDeque<MidiMessage>>,
    running: AtomicBool,
    messages: AtomicU64,
    notes: AtomicU64,
    soundfont: String,
    /// Where the sound goes, for the monitor.
    output: String,
}

/// A SoundFont synth as the device on a serial port.
pub struct MidiSynth {
    parser: Mutex<MidiParser>,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl MidiSynth {
    /// Load `soundfont` and start playing, at `gain` times rustysynth's own
    /// level, into a host stream that `audio` describes; with `mute` (the
    /// machine's `no_audio`) the synth parses and renders nothing. When the
    /// machine's audio goes to a file, the synth's goes to one beside it
    /// (`out.wav` → `out-midi.wav`): each stream writes a file of its own.
    pub fn open(soundfont: &str, gain: f32, audio: &crate::config::AudioConfig, mute: bool) -> Result<Self, String> {
        let font = {
            let mut f = std::io::BufReader::new(std::fs::File::open(soundfont).map_err(|e| format!("soundfont {soundfont}: {e}"))?);
            Arc::new(rustysynth::SoundFont::new(&mut f).map_err(|e| format!("soundfont {soundfont}: {e:?}"))?)
        };
        let mut synth = rustysynth::Synthesizer::new(&font, &rustysynth::SynthesizerSettings::new(RATE as i32)).map_err(|e| format!("synth: {e:?}"))?;
        synth.set_master_volume(synth.get_master_volume() * gain);
        let mut audio = audio.clone();
        audio.file = audio.file.as_deref().map(beside);
        let out = if mute { None } else { crate::host_audio::open_output(&audio, "MIDI synth", 2, RATE, "IRIS_MIDI_CAPTURE") };
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            running: AtomicBool::new(true),
            messages: AtomicU64::new(0),
            notes: AtomicU64::new(0),
            soundfont: soundfont.to_string(),
            output: match (&out, mute) {
                (_, true) => "muted (no_audio)".into(),
                (Some(o), _) => o.device().filter(|d| !d.is_empty()).unwrap_or_else(|| "host audio".into()),
                (None, _) => "no host audio".into(),
            },
        });
        let s = shared.clone();
        let thread = std::thread::Builder::new()
            .name("MIDI synth".into())
            .spawn(move || render_loop(&s, synth, out))
            .map_err(|e| format!("synth thread: {e}"))?;
        Ok(Self { parser: Mutex::new(MidiParser::new()), shared, thread: Some(thread) })
    }
}

/// `out.wav` → `out-midi.wav`.
fn beside(p: &std::path::Path) -> std::path::PathBuf {
    let stem = p.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    let name = match p.extension() {
        Some(e) => format!("{stem}-midi.{}", e.to_string_lossy()),
        None => format!("{stem}-midi"),
    };
    p.with_file_name(name)
}

fn render_loop(s: &Shared, mut synth: rustysynth::Synthesizer, mut out: Option<crate::host_audio::Output>) {
    let start = Instant::now();
    let mut made: u64 = 0;
    let mut left = vec![0f32; MAX_CATCH_UP as usize];
    let mut right = vec![0f32; MAX_CATCH_UP as usize];
    let mut frames = Vec::with_capacity(2 * MAX_CATCH_UP as usize);
    while s.running.load(Ordering::Relaxed) {
        std::thread::sleep(BLOCK);
        let queued: Vec<MidiMessage> = s.queue.lock().drain(..).collect();
        for m in queued {
            apply(&mut synth, &m, s);
        }
        let due = start.elapsed().as_nanos() as u64 * RATE as u64 / 1_000_000_000;
        if due.saturating_sub(made) > MAX_CATCH_UP {
            made = due - MAX_CATCH_UP;
        }
        let n = (due - made) as usize;
        if n == 0 {
            continue;
        }
        made = due;
        let Some(out) = out.as_mut() else { continue };
        synth.render(&mut left[..n], &mut right[..n]);
        frames.clear();
        for i in 0..n {
            frames.push(left[i]);
            frames.push(right[i]);
        }
        out.write(&frames);
    }
}

fn apply(synth: &mut rustysynth::Synthesizer, m: &MidiMessage, s: &Shared) {
    match m {
        MidiMessage::Channel { status, data1, data2 } => {
            if status & 0xf0 == 0x90 && *data2 != 0 {
                s.notes.fetch_add(1, Ordering::Relaxed);
            }
            synth.process_midi_message((status & 0x0f) as i32, (status & 0xf0) as i32, *data1 as i32, *data2 as i32);
        }
        // System Reset.
        MidiMessage::RealTime(0xff) => synth.reset(),
        MidiMessage::SysEx(x) if x == GM_ON || x == GS_RESET => synth.reset(),
        _ => {}
    }
}

impl SerialBackend for MidiSynth {
    fn send_byte(&self, byte: u8) {
        let Some(m) = self.parser.lock().push(byte) else { return };
        // Clock and active sensing are many and do nothing here.
        if matches!(m, MidiMessage::RealTime(0xf8 | 0xfe)) {
            return;
        }
        self.shared.messages.fetch_add(1, Ordering::Relaxed);
        self.shared.queue.lock().push_back(m);
    }

    fn recv_byte(&self) -> io::Result<u8> {
        Err(io::ErrorKind::WouldBlock.into())
    }

    fn raw(&self) -> bool {
        true
    }

    fn describe(&self) -> String {
        format!(
            "MIDI synth {} to {}; {} messages, {} notes",
            self.shared.soundfont,
            self.shared.output,
            self.shared.messages.load(Ordering::Relaxed),
            self.shared.notes.load(Ordering::Relaxed)
        )
    }
}

impl Drop for MidiSynth {
    fn drop(&mut self) {
        self.shared.running.store(false, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(bytes: &[u8]) -> Vec<MidiMessage> {
        let mut p = MidiParser::new();
        bytes.iter().filter_map(|&b| p.push(b)).collect()
    }

    fn ch(status: u8, data1: u8, data2: u8) -> MidiMessage {
        MidiMessage::Channel { status, data1, data2 }
    }

    #[test]
    fn running_status_carries_the_command_on() {
        assert_eq!(parse(&[0x90, 60, 100, 64, 100, 60, 0]), [ch(0x90, 60, 100), ch(0x90, 64, 100), ch(0x90, 60, 0)]);
        assert_eq!(parse(&[0xc3, 5, 7]), [ch(0xc3, 5, 0), ch(0xc3, 7, 0)], "a one-byte kind runs too");
    }

    #[test]
    fn real_time_bytes_land_inside_messages_without_breaking_them() {
        assert_eq!(parse(&[0x90, 0xf8, 60, 0xfe, 100]), [MidiMessage::RealTime(0xf8), MidiMessage::RealTime(0xfe), ch(0x90, 60, 100)]);
    }

    #[test]
    fn sysex_is_kept_whole_and_survives_real_time_bytes() {
        assert_eq!(parse(&[0xf0, 0x7e, 0xf8, 0x7f, 0x09, 0x01, 0xf7]), [MidiMessage::RealTime(0xf8), MidiMessage::SysEx(GM_ON.to_vec())]);
        // A status byte ends an unterminated SysEx and starts its own message.
        assert_eq!(parse(&[0xf0, 0x43, 0x90, 60, 1]), [MidiMessage::SysEx(vec![0xf0, 0x43, 0xf7]), ch(0x90, 60, 1)]);
    }

    #[test]
    fn system_common_cancels_running_status() {
        assert_eq!(parse(&[0x90, 60, 1, 0xf3, 2, 61, 1]), [ch(0x90, 60, 1), MidiMessage::Common { status: 0xf3, data: [2, 0] }]);
        assert_eq!(parse(&[0xf6, 0x80, 60, 0]), [MidiMessage::Common { status: 0xf6, data: [0; 2] }, ch(0x80, 60, 0)]);
    }

    #[test]
    fn a_file_output_gets_one_of_its_own() {
        assert_eq!(beside("out.wav".as_ref()), std::path::Path::new("out-midi.wav"));
        assert_eq!(beside("/tmp/run/audio".as_ref()), std::path::Path::new("/tmp/run/audio-midi"));
    }

    #[test]
    fn data_with_no_status_is_dropped() {
        assert_eq!(parse(&[60, 100, 0x90, 60]), []);
    }

    /// With the user's SoundFont present: a note sounds. Opt-in, it loads 30 MB.
    #[test]
    #[ignore]
    fn the_soundfont_plays_a_note() {
        let path = "ignore/midi/GeneralUser-GS/GeneralUser-GS.sf2";
        let Ok(mut f) = std::fs::File::open(path) else { return };
        let font = Arc::new(rustysynth::SoundFont::new(&mut f).unwrap());
        let mut synth = rustysynth::Synthesizer::new(&font, &rustysynth::SynthesizerSettings::new(RATE as i32)).unwrap();
        let mut p = MidiParser::new();
        for b in [0x90u8, 60, 100] {
            if let Some(MidiMessage::Channel { status, data1, data2 }) = p.push(b) {
                synth.process_midi_message((status & 0xf) as i32, (status & 0xf0) as i32, data1 as i32, data2 as i32);
            }
        }
        let (mut l, mut r) = (vec![0f32; 4410], vec![0f32; 4410]);
        synth.render(&mut l, &mut r);
        assert!(l.iter().chain(&r).any(|v| v.abs() > 0.01), "the note is heard");
    }
}
