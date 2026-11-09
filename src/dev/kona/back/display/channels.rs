//! The display generator's channels: the state the video commands set.
//!
//! The X server describes the screen to the display generator with video commands: it loads a
//! video format into each channel, says which rectangle of the managed area the channel shows,
//! and enables or blanks it. This module keeps what those setters leave behind, per physical
//! channel, so the Gets can answer from it (`reply`) and a combination can run through it
//! (`start`). The layouts are those of the server's combination load and its channel requests;
//! word 0 of every channel setter is the physical channel, the board mask's bit number.
//!
//! - DisableChannel [channel]: the channel stops and holds no format until the next LoadFormat.
//! - LoadFormat [channel, 1, ...]: words 2 to 18 are the format file's name and its datestamp,
//!   kept for GetChannelConfig; words 23 and 24 are the active lines and pixels, the output
//!   size. The channel runs.
//! - EnableChannel [channel, on, own timing]: word 1 nonzero runs a channel that holds a format.
//! - Resize [channel, x, top row from the bottom, width, height, 1, flicker, 0]: the input
//!   rectangle, and GetResize's pan rectangle when word 7 is 0.
//! - PlacementRes [channel, x code, y code, w3, 1, largest width, height, smallest width,
//!   height]: GetResize words 8 to 15.
//! - ChannelBlanking [channel, blank]; ChannelGammaMap [channel, map]; CursorPriority
//!   [channel, priority]; Vampire [channel, source, ...].
//!
//! Inferred: the combination load sends DisableChannel for every channel, then each enabled
//! channel's LoadFormat, Resize and blanking, then two EnableChannel passes, [channel, on, 0]
//! and [channel, 0, own timing]. Word 1 cannot be the channel's state, or the second pass would
//! turn every channel off. So a channel runs from its LoadFormat, and EnableChannel's word 1 is
//! only a request to run a channel that has had a LoadFormat since its last DisableChannel.
//!
//! The Gets answer from this state. A reply is the reply id, the data words, then the id plus the
//! data words, written from byte 8 of the return page ahead of the ready word; the reply ids and
//! sizes are the server's tables (`tables::VIDEO_REPLY`), and the words not set by `reply_words`
//! are zero:
//!
//! - GetBoardConfig: version 1, the board mask (bit n: physical channel n exists).
//! - GetChannelConfig: word 0 bit 0x1 present, 0x2 the encoder (channel 0), 0xc enabled and
//!   active; words 1 to 17 the format's name and datestamp, as LoadFormat gave them; word 18 the
//!   cursor priority.
//! - GetResize: the input rectangle, the pan rectangle, then PlacementRes's words.
//! - GetVampire: word 0 the channel whose timing this one takes.
//! - GetChannelBlanking: word 0 whether the channel is blanked. GetChannelEnables: word 1 whether
//!   it is enabled.
//!
//! Approximated: the layouts are the server's readers'; no reply was measured from a pipe. The
//! version, the pan rectangle at power-up and every word no reader takes are the model's choices.
//! A reply goes out as the Get's checksum arrives, where a pipe answers after queued work.
//!
//! Starting the firmware (`start`) is what the display generator's processor does each time the
//! loader releases it: rebuild the default state, blank every channel and run the video format
//! combination stored in the flash, or, when that image is rejected or a command in it raises the
//! error flag, rebuild again and run the built-in fallback combination. The stored words are
//! video commands framed as on the command page: the command, its data words and their checksum;
//! the image is accepted only whole, every command known with the table's data-word count and
//! every checksum matching. The run records the size SetupFrameBuffer gave (`frame`), but the
//! board keeps the managed size it was configured with and reports that one to the X server; a
//! combination that names another is logged and the other size is ignored.
//!
//! Approximated: the default-state rebuild is the model's reset channel state with every channel
//! disabled and blanked and its format name cleared; a command the table gives no length for
//! rejects the image; a Get in the image runs for no effect; and a system reset runs nothing,
//! where a pipe's power-up also runs the stored image before the PROM.
//!
//! Approximated: a channel that takes another's timing (Vampire), one bound to an X window, the
//! timing words of LoadFormat are kept or logged and change nothing on the screen; the board shows
//! one channel's worth of pixels at the managed size. Blanking (ChannelBlanking and
//! TempBlankAllChannels) blacks the frame when it blanks channel 0 or every channel.
//!
//! Ported from hw/misc/sgi-kona.c in jamesbraid/qemu-sgi-origami.

use crate::dev::kona::defs::*;
use crate::dev::kona::stream::Stream;
use crate::devlog::LogModule;

/// The channels a display generator drives (the board reports eight).
pub const CHANNELS: usize = crate::dev::kona::ident::CHANNELS as usize;

/// The words a channel's format is named by: sixteen of the file's name and its datestamp.
pub const NAME_WORDS: usize = 17;

/// The managed area the display generator starts with.
pub const FB: (u32, u32) = (1280, 1024);

/// The largest width or height a command may name.
const MAX_DIM: u32 = 16384;

/// The raster managers' pixel memory, and the width the budget rounds up to: a combination fits
/// when managers x 20,971,520 / (height x width to a multiple of 160) is at least 8 for Small
/// pixels and 16 for Medium.
const RM_PIXEL_BYTES: u64 = 20_971_520;
const TILE_WIDTH: u64 = 160;

/// One channel's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Channel {
    pub enabled: bool,
    /// A LoadFormat since its DisableChannel.
    pub loaded: bool,
    /// The format's active size: pixels, then lines.
    pub format: (u32, u32),
    /// The input rectangle: x, the top row counted from the bottom of the managed area, width,
    /// height.
    pub input: [u32; 4],
    pub blanked: bool,
    pub gamma_map: u32,
    pub cursor_priority: u32,
    /// The channel whose timing this one takes.
    pub source: u32,
    pub name: [u32; NAME_WORDS],
    /// GetResize words 4 to 7.
    pub pan: [u32; 4],
    /// GetResize words 8 to 15: PlacementRes's words 1 to 8.
    pub placement: [u32; 8],
}

impl Channel {
    /// A channel as the display generator starts it: off, taking its own timing, panning over
    /// the whole managed area.
    fn off(index: usize) -> Channel {
        Channel {
            enabled: false,
            loaded: false,
            format: (0, 0),
            input: [0; 4],
            blanked: false,
            gamma_map: 0,
            cursor_priority: 0,
            source: index as u32,
            name: [0; NAME_WORDS],
            pan: [0, 0, FB.0, FB.1],
            placement: [1, 1, 1, 1, FB.0, FB.1, 0, 0],
        }
    }

    fn visit<S: Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { enabled: _, loaded: _, format: _, input: _, blanked: _, gamma_map: _, cursor_priority: _, source: _, name: _, pan: _, placement: _ } = self;
        s.bool(&mut self.enabled);
        s.bool(&mut self.loaded);
        s.u32(&mut self.format.0);
        s.u32(&mut self.format.1);
        s.u32s(&mut self.input);
        s.bool(&mut self.blanked);
        s.u32(&mut self.gamma_map);
        s.u32(&mut self.cursor_priority);
        s.u32(&mut self.source);
        s.u32s(&mut self.name);
        s.u32s(&mut self.pan);
        s.u32s(&mut self.placement);
    }
}

/// The display generator's video state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Channels {
    pub channel: [Channel; CHANNELS],
    /// TempBlankAllChannels' last word.
    pub blanked: bool,
    /// The managed area SetupFrameBuffer last named.
    pub frame: (u32, u32),
    /// A setter raised the error flag, which stops a stored combination.
    pub error: bool,
    rm_count: u32,
}

impl Channels {
    /// The display generator at power-up: channel 0 shows the whole managed area at 1280x1024,
    /// the format the PROM falls back to, and every other channel is off.
    ///
    /// Approximated: a real display generator runs the combination stored in its flash at
    /// power-up; the board runs one only when the firmware is released (`start`), so a reset
    /// starts the single-channel state every captured pipe can show.
    pub fn new(rm_count: u32) -> Channels {
        let mut channel: [Channel; CHANNELS] = std::array::from_fn(Channel::off);
        let c = &mut channel[0];
        c.enabled = true;
        c.loaded = true;
        c.format = FB;
        c.input = [0, FB.1 - 1, FB.0, FB.1];
        Channels { channel, blanked: false, frame: FB, error: false, rm_count }
    }

    pub fn reset(&mut self) {
        *self = Channels::new(self.rm_count);
    }

    /// The state, as the stream module reads and writes it.
    pub fn visit<S: Stream>(&mut self, s: &mut S) {
        let Self { channel: _, blanked: _, frame: _, error: _, rm_count: _ } = self;
        self.channel.iter_mut().for_each(|c| c.visit(s));
        s.bool(&mut self.blanked);
        s.u32(&mut self.frame.0);
        s.u32(&mut self.frame.1);
        s.bool(&mut self.error);
    }

    /// Raster-manager pixel budget at a managed size: how many bytes a pixel may take.
    fn budget(&self, width: u32, height: u32) -> u64 {
        let area = (width as u64).div_ceil(TILE_WIDTH) * TILE_WIDTH * height as u64;
        if area == 0 {
            0
        } else {
            self.rm_count.max(1) as u64 * RM_PIXEL_BYTES / area
        }
    }

    /// SetupFrameBuffer [width, height, pixel depth], as the firmware runs it: a budget below
    /// Small's, or a named depth (0 to 3) the budget does not cover, raises the error flag; -1,
    /// -2 and other values pick a depth that fits. It never replies.
    fn setup_frame_buffer(&mut self, width: u32, height: u32, depth: u32) {
        const NEED: [u64; 4] = [8, 16, 32, 64];
        let budget = self.budget(width, height);
        if budget < NEED[0] {
            crate::dlog!(LogModule::Kona, "kona: not enough raster managers for Small pixels at {width}x{height}");
            self.error = true;
        } else if let Some(&need) = NEED.get(depth as usize) {
            if budget < need {
                self.error = true;
            }
        }
        self.frame = (width, height);
    }

    /// The reply to a video Get, as the bytes that go on the return page from byte 8: the reply id,
    /// the data words and their sum with the id, big-endian. None for a command nobody waits on,
    /// one past the table, or a per-channel Get naming a channel the board lacks (logged).
    pub fn reply(&self, command: u32, data: &[u32]) -> Option<Vec<u8>> {
        let (id, bytes) = crate::dev::kona::tables::VIDEO_REPLY.get(command as usize).copied().flatten()?;
        let channel = if names_channel(command) { data.first().copied().unwrap_or(0) as usize } else { 0 };
        if channel >= CHANNELS {
            crate::dlog!(LogModule::Kona, "kona: unsupported video channel {channel} for {command:#x}");
            return None;
        }
        let mut words = vec![0u32; bytes as usize / 4];
        self.reply_words(command, channel, &mut words);
        let sum = words.iter().fold(id as u32, |s, w| s.wrapping_add(*w));
        let mut out = Vec::with_capacity(8 + bytes as usize);
        out.extend_from_slice(&(id as u32).to_be_bytes());
        out.extend(words.iter().flat_map(|w| w.to_be_bytes()));
        out.extend_from_slice(&sum.to_be_bytes());
        Some(out)
    }

    /// The data words of a Get's reply, from the channel state.
    fn reply_words(&self, command: u32, channel: usize, words: &mut [u32]) {
        let ch = &self.channel[channel];
        let mut put = |at: usize, v: &[u32]| {
            for (i, w) in v.iter().enumerate() {
                if let Some(slot) = words.get_mut(at + i) {
                    *slot = *w;
                }
            }
        };
        match command {
            VIDEO_GET_BOARD_CONFIG => put(0, &[1, (1u32 << CHANNELS) - 1]),
            VIDEO_GET_CHANNEL_CONFIG => {
                put(0, &[1 | if channel == 0 { 2 } else { 0 } | if ch.enabled { 0xc } else { 0 }]);
                put(1, &ch.name);
                put(18, &[ch.cursor_priority]);
            }
            VIDEO_GET_RESIZE => {
                put(0, &ch.input);
                put(4, &ch.pan);
                put(8, &ch.placement);
            }
            VIDEO_GET_VAMPIRE => put(0, &[ch.source]),
            VIDEO_GET_CHANNEL_BLANKING => put(0, &[ch.blanked as u32]),
            VIDEO_GET_CHANNEL_ENABLES => put(1, &[ch.enabled as u32]),
            _ => {}
        }
    }

    /// The display generator's default state, as its rebuild resets each channel: the timing
    /// source, the first and last name words (the name's first word and the datestamp), the pan
    /// rectangle, the enabled bit and gamma map 0. The input rectangle, placement, the other name
    /// words and the cursor priority keep their values. Then every channel is blanked, as the
    /// display set-up that follows does.
    fn default_state(&mut self) {
        for (i, ch) in self.channel.iter_mut().enumerate() {
            ch.enabled = false;
            ch.loaded = false;
            ch.source = i as u32;
            ch.name[0] = 0;
            ch.name[NAME_WORDS - 1] = 0;
            ch.pan = [0, 0, FB.0, FB.1];
            ch.gamma_map = 0;
            ch.blanked = true;
        }
    }

    /// Start the firmware: run `stored`, the combination in the flash (empty when its marker or
    /// count is bad), else the built-in one. Returns the managed area the run left.
    pub fn start(&mut self, stored: &[u32]) -> (u32, u32) {
        self.default_state();
        let ran = !stored.is_empty() && image_valid(stored) && self.run(stored);
        if !ran {
            if !stored.is_empty() {
                crate::dlog!(LogModule::Kona, "kona: stored combination of {} words failed to load; running the built-in one", stored.len());
                self.default_state();
            }
            self.run(&fallback());
        }
        self.frame
    }

    /// Run a combination `image_valid` accepted, command by command; false when one raises the
    /// error flag.
    fn run(&mut self, words: &[u32]) -> bool {
        self.error = false;
        let mut at = 0;
        while at < words.len() {
            let command = words[at];
            let n = crate::dev::kona::tables::VIDEO[command as usize].unwrap_or(0) as usize;
            self.set(command, &words[at + 1..at + 1 + n]);
            if self.error {
                return false;
            }
            at += n + 2;
        }
        true
    }

    /// Run one video command that sets state. `data` is the words between the command number and
    /// its checksum; a word past its end reads as zero. Gets and commands the model gives no
    /// meaning change nothing.
    pub fn set(&mut self, command: u32, data: &[u32]) {
        let d = |i: usize| data.get(i).copied().unwrap_or(0);
        match command {
            VIDEO_TEMP_BLANK => {
                self.blanked = d(0) != 0;
                crate::dlog!(LogModule::Kona, "kona: TempBlankAllChannels {}", d(0));
            }
            VIDEO_SETUP_FRAME_BUFFER => self.setup_frame_buffer(d(0), d(1), d(2)),
            VIDEO_DISABLE_CHANNEL | VIDEO_LOAD_FORMAT | VIDEO_ENABLE_CHANNEL | VIDEO_RESIZE | VIDEO_CHANNEL_BLANKING | VIDEO_CHANNEL_GAMMA_MAP | VIDEO_CURSOR_PRIORITY | VIDEO_VAMPIRE | VIDEO_BIND_TO_X_WINDOW | VIDEO_PLACEMENT_RES => {
                let channel = d(0) as usize;
                if channel >= CHANNELS {
                    crate::dlog!(LogModule::Kona, "kona: unsupported video channel {channel} for {command:#x}");
                    return;
                }
                self.set_channel(command, channel, data);
            }
            _ => {}
        }
    }

    fn set_channel(&mut self, command: u32, channel: usize, data: &[u32]) {
        let d = |i: usize| data.get(i).copied().unwrap_or(0);
        let ch = &mut self.channel[channel];
        match command {
            VIDEO_DISABLE_CHANNEL => {
                ch.enabled = false;
                ch.loaded = false;
            }
            VIDEO_LOAD_FORMAT => {
                if d(23) == 0 || d(24) == 0 || d(23) > MAX_DIM || d(24) > MAX_DIM {
                    crate::dlog!(LogModule::Kona, "kona: channel {channel} format {}x{} is outside the sizes presented", d(24), d(23));
                    return;
                }
                ch.format = (d(24), d(23));
                for (i, w) in ch.name.iter_mut().enumerate() {
                    *w = d(2 + i);
                }
                ch.loaded = true;
                ch.enabled = true;
            }
            VIDEO_ENABLE_CHANNEL => {
                if d(1) != 0 && ch.loaded {
                    ch.enabled = true;
                }
            }
            VIDEO_RESIZE => {
                if (1..=4).any(|i| d(i) > MAX_DIM) {
                    crate::dlog!(LogModule::Kona, "kona: channel {channel} input {}x{} at {},{} is outside the sizes presented", d(3), d(4), d(1), d(2));
                    return;
                }
                ch.input = [d(1), d(2), d(3), d(4)];
                if d(7) == 0 {
                    ch.pan = ch.input;
                }
            }
            VIDEO_PLACEMENT_RES => {
                for (i, w) in ch.placement.iter_mut().enumerate() {
                    *w = d(1 + i);
                }
            }
            VIDEO_CHANNEL_BLANKING => {
                ch.blanked = d(1) != 0;
                crate::dlog!(LogModule::Kona, "kona: channel {channel} blanking {}", d(1));
            }
            VIDEO_CHANNEL_GAMMA_MAP => {
                if d(1) != 0 && d(1) != VIDEO_GAMMA_MAP_LINEAR {
                    crate::dlog!(LogModule::Kona, "kona: channel {channel} gamma map {}: the model holds map 0 alone", d(1) as i32);
                    return;
                }
                ch.gamma_map = d(1);
            }
            VIDEO_CURSOR_PRIORITY => ch.cursor_priority = d(1),
            VIDEO_VAMPIRE => {
                ch.source = d(1);
                if d(1) as usize != channel {
                    crate::dlog!(LogModule::Kona, "kona: channel {channel} taking channel {}'s timing is not modelled", d(1));
                }
            }
            VIDEO_BIND_TO_X_WINDOW => {
                if d(1) != 0 {
                    crate::dlog!(LogModule::Kona, "kona: channel {channel} bound to X window {:#x} is not modelled", d(1));
                }
            }
            _ => unreachable!("set_channel is called for channel setters only"),
        }
    }
}

/// Whether a stored combination is whole: every command known, with a data-word count in the
/// table, its checksum (the command plus its data) matching, and nothing left over.
fn image_valid(words: &[u32]) -> bool {
    let mut at = 0;
    while at < words.len() {
        let command = words[at] as usize;
        let Some(Some(n)) = crate::dev::kona::tables::VIDEO.get(command).copied() else { return false };
        let n = n as usize;
        if n + 1 >= words.len() - at {
            return false;
        }
        let sum = words[at..=at + n].iter().fold(0u32, |s, w| s.wrapping_add(*w));
        if sum != words[at + n + 1] {
            return false;
        }
        at += n + 2;
    }
    true
}

/// Append video command `command` with `data` and its checksum to `image`.
fn append(image: &mut Vec<u32>, command: u32, data: &[u32]) {
    image.push(command);
    image.extend_from_slice(data);
    image.push(data.iter().fold(command, |s, w| s.wrapping_add(*w)));
}

/// LoadFormat for `channel`, named `name` and datestamped `stamp`, 1280x1024 at 60 Hz in the
/// words the model reads.
fn append_format(image: &mut Vec<u32>, channel: u32, name: &str, stamp: u32, size: (u32, u32)) {
    let mut format = [0u32; 383];
    format[0] = channel;
    format[1] = 1;
    let mut padded = [0u8; 64];
    padded[..name.len()].copy_from_slice(name.as_bytes());
    for (w, c) in format[2..18].iter_mut().zip(padded.chunks_exact(4)) {
        *w = u32::from_be_bytes(c.try_into().unwrap());
    }
    format[18] = stamp;
    format[19] = 60 << 16;
    format[20] = 1;
    format[21] = 1065;
    format[22] = 1680;
    format[23] = size.1;
    format[24] = size.0;
    append(image, VIDEO_LOAD_FORMAT, &format);
}

/// The firmware's built-in fallback combination: a 1280x1024 managed area, channel 0 showing
/// 1280x1024_60.vfo, channel 1 named 640x486_30i.vfo with channel 0's timing, taking channel 0's
/// timing and blanked. Established from the firmware's built-in table.
///
/// Approximated: only the commands the model gives meaning are built, so not the clock, encoder,
/// gamma, sync, gain, pedestal and phase settings, the cursor priorities (which stay 0, as at
/// reset), or the commands for the option channel; and LoadFormat's words past the size and rate
/// are zero. Both channels' Resize and PlacementRes words are the firmware's.
fn fallback() -> Vec<u32> {
    let mut i = Vec::new();
    for c in 0..2 {
        append(&mut i, VIDEO_DISABLE_CHANNEL, &[c]);
        append(&mut i, VIDEO_CHANNEL_BLANKING, &[c, 1]);
    }
    append(&mut i, VIDEO_SETUP_FRAME_BUFFER, &[1280, 1024, u32::MAX]);
    append_format(&mut i, 0, "1280x1024_60.vfo", 0x31ea_4b22, (1280, 1024));
    append(&mut i, VIDEO_VAMPIRE, &[0, 0, 1280, 1024, 0, 5, 3]);
    append(&mut i, VIDEO_RESIZE, &[0, 0, 1023, 1280, 1024, 1, 1, 0]);
    append(&mut i, VIDEO_CHANNEL_BLANKING, &[0, 0]);
    append_format(&mut i, 1, "640x486_30i.vfo", 0x31ea_4b3c, (1280, 1024));
    append(&mut i, VIDEO_VAMPIRE, &[1, 0, 640, 486, 1, 5, 3]);
    append(&mut i, VIDEO_RESIZE, &[1, 0, 1023, 1280, 1024, 1, 1, 0]);
    append(&mut i, VIDEO_PLACEMENT_RES, &[0, 4, 1, 4, 1, 1280, 1024, 4, 1]);
    append(&mut i, VIDEO_PLACEMENT_RES, &[1, 4, 1, 4, 1, 2048, 2048, 640, 486]);
    append(&mut i, VIDEO_CHANNEL_BLANKING, &[1, 1]);
    append(&mut i, VIDEO_ENABLE_CHANNEL, &[1, 1, 0]);
    append(&mut i, VIDEO_ENABLE_CHANNEL, &[1, 0, 0]);
    append(&mut i, VIDEO_ENABLE_CHANNEL, &[0, 1, 0]);
    append(&mut i, VIDEO_ENABLE_CHANNEL, &[0, 0, 1]);
    i
}

/// Whether a Get's first data word names a physical channel.
fn names_channel(command: u32) -> bool {
    matches!(command, VIDEO_GET_CHANNEL_CONFIG | VIDEO_GET_RESIZE | VIDEO_GET_VAMPIRE | VIDEO_GET_CHANNEL_ENABLES | VIDEO_GET_FORMAT_INFO | VIDEO_GET_CHANNEL_BLANKING)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A LoadFormat request for `channel`: name words, datestamp and the active size.
    fn format(channel: u32, name: &str, stamp: u32, width: u32, height: u32) -> Vec<u32> {
        let mut w = vec![0u32; 383];
        w[0] = channel;
        w[1] = 1;
        let mut padded = [0u8; 64];
        padded[..name.len()].copy_from_slice(name.as_bytes());
        for i in 0..16 {
            w[2 + i] = u32::from_be_bytes(padded[4 * i..4 * i + 4].try_into().unwrap());
        }
        w[18] = stamp;
        w[23] = height;
        w[24] = width;
        w
    }

    #[test]
    fn power_up_shows_channel_zero_alone() {
        let c = Channels::new(2);
        assert!(c.channel[0].enabled && c.channel[0].loaded);
        assert_eq!((c.channel[0].format, c.channel[0].input), ((1280, 1024), [0, 1023, 1280, 1024]));
        assert!(c.channel[1..].iter().all(|c| !c.enabled && !c.loaded));
        assert_eq!(c.channel[5].source, 5);
        assert_eq!(c.channel[5].pan, [0, 0, 1280, 1024]);
    }

    #[test]
    fn a_format_runs_the_channel_and_a_disable_stops_it() {
        let mut c = Channels::new(2);
        c.set(VIDEO_LOAD_FORMAT, &format(3, "640x486_30i.vfo", 0x31ea_4b3c, 640, 486));
        assert!(c.channel[3].enabled && c.channel[3].loaded);
        assert_eq!(c.channel[3].format, (640, 486));
        assert_eq!(c.channel[3].name[0], u32::from_be_bytes(*b"640x"));
        assert_eq!(c.channel[3].name[16], 0x31ea_4b3c);
        c.set(VIDEO_DISABLE_CHANNEL, &[3]);
        assert!(!c.channel[3].enabled && !c.channel[3].loaded);
        // An enable asks to run a channel that holds a format; this one lost it.
        c.set(VIDEO_ENABLE_CHANNEL, &[3, 1, 0]);
        assert!(!c.channel[3].enabled);
        // Its format size stays a format of a sort, but a format with no size is refused.
        c.set(VIDEO_LOAD_FORMAT, &format(3, "bad.vfo", 0, 0, 486));
        assert!(!c.channel[3].loaded);
    }

    #[test]
    fn the_second_enable_pass_does_not_turn_a_running_channel_off() {
        let mut c = Channels::new(2);
        c.set(VIDEO_LOAD_FORMAT, &format(1, "a.vfo", 0, 640, 486));
        c.set(VIDEO_ENABLE_CHANNEL, &[1, 1, 0]);
        c.set(VIDEO_ENABLE_CHANNEL, &[1, 0, 0]);
        assert!(c.channel[1].enabled);
    }

    #[test]
    fn a_resize_sets_the_input_and_the_pan_unless_word_seven_is_set() {
        let mut c = Channels::new(2);
        c.set(VIDEO_RESIZE, &[1, 10, 500, 640, 480, 1, 1, 0]);
        assert_eq!((c.channel[1].input, c.channel[1].pan), ([10, 500, 640, 480], [10, 500, 640, 480]));
        c.set(VIDEO_RESIZE, &[1, 20, 400, 320, 240, 1, 1, 1]);
        assert_eq!((c.channel[1].input, c.channel[1].pan), ([20, 400, 320, 240], [10, 500, 640, 480]));
        c.set(VIDEO_RESIZE, &[1, 0, 0, 99_999, 1, 1, 1, 0]);
        assert_eq!(c.channel[1].input, [20, 400, 320, 240], "a size past the limit is refused");
    }

    #[test]
    fn the_other_setters_land_in_their_channel() {
        let mut c = Channels::new(2);
        c.set(VIDEO_PLACEMENT_RES, &[2, 4, 1, 4, 1, 2048, 2048, 640, 486]);
        assert_eq!(c.channel[2].placement, [4, 1, 4, 1, 2048, 2048, 640, 486]);
        c.set(VIDEO_CHANNEL_BLANKING, &[2, 7]);
        assert!(c.channel[2].blanked);
        c.set(VIDEO_CHANNEL_BLANKING, &[2, 0]);
        assert!(!c.channel[2].blanked);
        c.set(VIDEO_CURSOR_PRIORITY, &[2, 5]);
        assert_eq!(c.channel[2].cursor_priority, 5);
        c.set(VIDEO_VAMPIRE, &[2, 0, 1280, 1024, 0, 5, 3]);
        assert_eq!(c.channel[2].source, 0);
        c.set(VIDEO_CHANNEL_GAMMA_MAP, &[2, VIDEO_GAMMA_MAP_LINEAR]);
        assert_eq!(c.channel[2].gamma_map, VIDEO_GAMMA_MAP_LINEAR);
        c.set(VIDEO_CHANNEL_GAMMA_MAP, &[2, 3]);
        assert_eq!(c.channel[2].gamma_map, VIDEO_GAMMA_MAP_LINEAR, "another map is refused");
        c.set(VIDEO_TEMP_BLANK, &[1]);
        assert!(c.blanked);
        c.set(VIDEO_TEMP_BLANK, &[0]);
        assert!(!c.blanked);
    }

    #[test]
    fn a_channel_the_board_lacks_changes_nothing() {
        let mut c = Channels::new(2);
        let before = c.clone();
        for cmd in [VIDEO_DISABLE_CHANNEL, VIDEO_CHANNEL_BLANKING, VIDEO_CURSOR_PRIORITY] {
            c.set(cmd, &[CHANNELS as u32, 1, 1]);
        }
        assert_eq!(c, before);
    }

    #[test]
    fn setting_up_the_frame_buffer_checks_the_raster_managers_budget() {
        let mut c = Channels::new(2);
        c.set(VIDEO_SETUP_FRAME_BUFFER, &[1280, 1024, u32::MAX]);
        assert!(!c.error);
        assert_eq!(c.frame, (1280, 1024));
        // Two managers hold 1280x1024 at a budget of 32 bytes a pixel: depth 2 fits, depth 3 does not.
        c.set(VIDEO_SETUP_FRAME_BUFFER, &[1280, 1024, 2]);
        assert!(!c.error);
        c.set(VIDEO_SETUP_FRAME_BUFFER, &[1280, 1024, 3]);
        assert!(c.error);
        let mut one = Channels::new(1);
        one.set(VIDEO_SETUP_FRAME_BUFFER, &[4096, 4096, u32::MAX]);
        assert!(one.error, "not even Small pixels fit");
    }

    #[test]
    fn a_short_request_reads_missing_words_as_zero() {
        let mut c = Channels::new(2);
        c.set(VIDEO_CURSOR_PRIORITY, &[4]);
        assert_eq!(c.channel[4].cursor_priority, 0);
    }

    /// The words of a reply after the id, without the closing sum.
    fn words(reply: &[u8]) -> Vec<u32> {
        reply.chunks(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())).collect()
    }

    #[test]
    fn the_board_config_reply_names_the_eight_channels() {
        let r = Channels::new(2).reply(VIDEO_GET_BOARD_CONFIG, &[0]).unwrap();
        // Reply id 6, five data words (version, board mask, three zeros), id plus data.
        assert_eq!(words(&r), [6, 1, 0xff, 0, 0, 0, 0x106]);
        assert_eq!(r.len(), 28);
    }

    #[test]
    fn a_channel_config_reply_carries_the_loaded_format() {
        let mut c = Channels::new(2);
        c.set(VIDEO_LOAD_FORMAT, &format(0, "1280x1024_60.vfo", 0x31ea_4b22, 1280, 1024));
        c.set(VIDEO_CURSOR_PRIORITY, &[0, 3]);
        let w = words(&c.reply(VIDEO_GET_CHANNEL_CONFIG, &[0]).unwrap());
        assert_eq!(w.len(), 1 + 19 + 1);
        assert_eq!((w[0], w[1]), (7, 0xf), "id 7, present, the encoder, enabled and active");
        assert_eq!(&w[2..6], &[u32::from_be_bytes(*b"1280"), u32::from_be_bytes(*b"x102"), u32::from_be_bytes(*b"4_60"), u32::from_be_bytes(*b".vfo")]);
        assert_eq!((w[18], w[19]), (0x31ea_4b22, 3));
        assert_eq!(w[20], w[..20].iter().fold(0u32, |s, v| s.wrapping_add(*v)) , "the sum is the id plus every data word");
        // A channel that is off says it is present and nothing more.
        let off = words(&c.reply(VIDEO_GET_CHANNEL_CONFIG, &[4]).unwrap());
        assert_eq!(off[1], 1);
    }

    #[test]
    fn a_resize_reply_returns_the_rectangles() {
        let mut c = Channels::new(2);
        c.set(VIDEO_RESIZE, &[1, 0, 1023, 1280, 1024, 1, 1, 0]);
        c.set(VIDEO_PLACEMENT_RES, &[1, 4, 1, 4, 1, 2048, 2048, 640, 486]);
        let w = words(&c.reply(VIDEO_GET_RESIZE, &[1]).unwrap());
        assert_eq!(&w[1..17], &[0, 1023, 1280, 1024, 0, 1023, 1280, 1024, 4, 1, 4, 1, 2048, 2048, 640, 486]);
        assert_eq!(w.len(), 1 + 18 + 1);
        // The power-up channel pans over the whole area and takes steps of one pixel.
        let fresh = words(&Channels::new(2).reply(VIDEO_GET_RESIZE, &[5]).unwrap());
        assert_eq!(&fresh[5..9], &[0, 0, 1280, 1024]);
        assert_eq!(&fresh[9..17], &[1, 1, 1, 1, 1280, 1024, 0, 0]);
    }

    #[test]
    fn the_small_gets_read_one_channel() {
        let mut c = Channels::new(2);
        c.set(VIDEO_CHANNEL_BLANKING, &[2, 1]);
        c.set(VIDEO_VAMPIRE, &[2, 0, 1280, 1024, 0, 5, 3]);
        assert_eq!(words(&c.reply(VIDEO_GET_CHANNEL_BLANKING, &[2]).unwrap()), [9, 1, 10]);
        assert_eq!(words(&c.reply(VIDEO_GET_CHANNEL_BLANKING, &[3]).unwrap()), [9, 0, 9]);
        let v = words(&c.reply(VIDEO_GET_VAMPIRE, &[2]).unwrap());
        assert_eq!((v.len(), v[1], v[2]), (8, 0, 0), "the source is channel 0");
        assert_eq!(words(&c.reply(VIDEO_GET_CHANNEL_ENABLES, &[0]).unwrap()), [19, 0, 1, 20]);
    }

    #[test]
    fn a_get_for_a_missing_channel_or_a_setter_posts_nothing() {
        let c = Channels::new(2);
        assert_eq!(c.reply(VIDEO_GET_CHANNEL_CONFIG, &[CHANNELS as u32]), None);
        assert_eq!(c.reply(VIDEO_LOAD_FORMAT, &[0]), None);
        assert_eq!(c.reply(0x5d, &[0]), None, "a number past the table");
        // A Get that names no channel answers from channel 0 and is not refused for its word.
        assert!(c.reply(VIDEO_GET_BOARD_CONFIG, &[99]).is_some());
    }

    #[test]
    fn every_reply_id_is_the_servers_and_sized_by_its_table() {
        use crate::dev::kona::tables::VIDEO_REPLY;
        // The reply sizes by reply id, as the display-request table lists them (id 26 has no sender: the
        // monitor command, which has a payload, has no reply).
        const SIZES: [u8; 32] = [4, 4, 4, 4, 4, 4, 20, 76, 72, 4, 4, 24, 4, 4, 4, 4, 24, 12, 4, 8, 4, 4, 4, 20, 16, 68, 136, 4, 4, 8, 8, 164];
        let rows: Vec<_> = VIDEO_REPLY.iter().flatten().collect();
        assert_eq!(rows.len(), 31);
        let ids: Vec<usize> = rows.iter().map(|(id, _)| *id as usize).collect();
        assert_eq!(ids, (0..32).filter(|&i| i != 26).collect::<Vec<_>>(), "in command order, every id but 26");
        for (id, bytes) in rows {
            assert_eq!(*bytes, SIZES[*id as usize], "reply {id}");
        }
        assert_eq!(VIDEO_REPLY[0x51], None);
        assert_eq!(VIDEO_REPLY.len(), crate::dev::kona::tables::VIDEO.len());
    }

    /// The words of a framed combination: each command, its data and checksum.
    fn framed(commands: &[(u32, Vec<u32>)]) -> Vec<u32> {
        let mut image = Vec::new();
        for (c, d) in commands {
            append(&mut image, *c, d);
        }
        image
    }

    #[test]
    fn the_built_in_combination_is_valid_and_runs_two_channels() {
        let image = fallback();
        assert!(image_valid(&image));
        let mut c = Channels::new(2);
        assert_eq!(c.start(&[]), (1280, 1024));
        assert!(!c.error);
        // Channel 0 runs 1280x1024_60.vfo, unblanked; channel 1 holds its format blanked; the rest are off.
        assert!(c.channel[0].enabled && !c.channel[0].blanked);
        assert!(c.channel[1].enabled && c.channel[1].blanked);
        assert_eq!(c.channel[1].source, 0, "channel 1 takes channel 0's timing");
        assert!(c.channel[2..].iter().all(|c| !c.enabled && c.blanked));
        let name = words(&c.reply(VIDEO_GET_CHANNEL_CONFIG, &[0]).unwrap());
        assert_eq!(&name[2..6], &[u32::from_be_bytes(*b"1280"), u32::from_be_bytes(*b"x102"), u32::from_be_bytes(*b"4_60"), u32::from_be_bytes(*b".vfo")]);
        assert_eq!(name[18], 0x31ea_4b22);
        let two = words(&c.reply(VIDEO_GET_RESIZE, &[1]).unwrap());
        assert_eq!(&two[9..17], &[4, 1, 4, 1, 2048, 2048, 640, 486], "PlacementRes of channel 1");
    }

    #[test]
    fn a_whole_stored_combination_replaces_the_built_in_one() {
        let mut image = framed(&[(VIDEO_SETUP_FRAME_BUFFER, vec![1280, 1024, u32::MAX])]);
        append(&mut image, VIDEO_DISABLE_CHANNEL, &[0]);
        append(&mut image, VIDEO_DISABLE_CHANNEL, &[1]);
        append_format(&mut image, 2, "1600x1200_60.vfo", 5, (1600, 1200));
        let mut c = Channels::new(2);
        c.start(&image);
        assert!(!c.channel[0].enabled && !c.channel[1].enabled);
        assert!(c.channel[2].enabled);
        assert_eq!(c.channel[2].format, (1600, 1200));
    }

    #[test]
    fn a_damaged_or_failing_combination_falls_back_to_the_built_in_one() {
        let good = framed(&[(VIDEO_DISABLE_CHANNEL, vec![0])]);
        let mut bad_sum = good.clone();
        *bad_sum.last_mut().unwrap() += 1;
        let unknown = framed(&[(0x7f, vec![1])]);
        // A command's data must be the table's count: Test takes one word.
        let wrong_length = framed(&[(0x00, vec![0, 0])]);
        let cut = good[..good.len() - 1].to_vec();
        let mut trailing = good.clone();
        trailing.push(VIDEO_DISABLE_CHANNEL);
        assert!(image_valid(&good));
        // Two managers cannot hold 8192x8192 pixels: a whole image whose run raises the error flag.
        let too_big = framed(&[(VIDEO_DISABLE_CHANNEL, vec![0]), (VIDEO_SETUP_FRAME_BUFFER, vec![8192, 8192, u32::MAX])]);
        assert!(image_valid(&too_big));
        let mut want = Channels::new(2);
        want.start(&[]);
        for (what, image) in [("sum", bad_sum), ("unknown", unknown), ("wrong length", wrong_length), ("cut", cut), ("trailing", trailing), ("error", too_big)] {
            if what != "error" {
                assert!(!image_valid(&image), "{what}");
            }
            let mut c = Channels::new(2);
            c.start(&image);
            assert_eq!(c, want, "{what}: ran the built-in combination");
        }
    }

    #[test]
    fn starting_again_rebuilds_the_default_state_first() {
        let mut c = Channels::new(2);
        c.start(&[]);
        c.set(VIDEO_LOAD_FORMAT, &format(5, "x.vfo", 7, 640, 480));
        c.set(VIDEO_VAMPIRE, &[5, 0]);
        c.set(VIDEO_CHANNEL_GAMMA_MAP, &[5, VIDEO_GAMMA_MAP_LINEAR]);
        c.start(&[]);
        let f = &c.channel[5];
        assert!(!f.enabled && !f.loaded && f.blanked);
        assert_eq!((f.source, f.gamma_map, f.name[0], f.name[16]), (5, 0, 0, 0));
    }

    #[test]
    fn the_state_survives_a_stream_round_trip_and_a_reset_clears_it() {
        use crate::dev::kona::stream::{Reader, Writer};
        let mut c = Channels::new(2);
        c.set(VIDEO_LOAD_FORMAT, &format(1, "640x486_30i.vfo", 9, 640, 486));
        c.set(VIDEO_TEMP_BLANK, &[1]);
        let mut w = Writer::default();
        c.visit(&mut w);
        let bytes = w.out;
        let mut back = Channels::new(2);
        let mut r = Reader::new(&bytes);
        back.visit(&mut r);
        assert!(r.finish().is_ok());
        assert_eq!(back, c);
        back.reset();
        assert_eq!(back, Channels::new(2));
    }
}
