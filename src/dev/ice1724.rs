//! IC Ensemble ICE1724 (Envy24HT) PCI audio (1412:1724 revision 1) as the
//! SGI DMediaPro DM8 option card has it, with the converter wiring of an
//! M-Audio Revolution 7.1 board, and an AC'97 variant of the same controller.
//!
//! BAR0 (32 bytes, I/O) is the controller block, "CCS"; BAR1 (128 bytes,
//! I/O) is the multi-channel block, "MT". Both are little-endian byte
//! register files: a wider access is byte steps in address order. Seven DMA
//! engines share one sample rate. The `dm8` variant has two write-only AKM
//! converters behind bit-banged GPIO pins and a configuration EEPROM on the
//! I2C port; the `ac97` variant has no converter pins and no EEPROM, and a
//! plain AC'97 register file behind MT04 to MT06.
//!
//! Where a fact is established, the code follows it. Where this file departs
//! from one, or fills an unknown with a safe behaviour, the comment says
//! "Approximation". An access to a register or value that nothing
//! establishes is logged (`ice1724` in the monitor's `log` command), never
//! silently accepted.
//!
//! Approximations kept from the model:
//!
//! * Soft reset (CCS00 bit 7) is not described for this chip. It restores
//!   every register except CCS00 and CCS04 to CCS07, as the VT1720 sibling
//!   part does; the bit does not self-clear. The converters and a frame in
//!   flight are not touched, nor does PRST# (MT05 bit 7, which also drives
//!   it in I2S mode) reach any converter.
//! * The converter recipe is a stated model choice. Which subsystem ID,
//!   EEPROM and converters the Tezro's board has is unknown; the `dm8`
//!   variant answers 0x3630 at config 0x2e so a guest driver takes the
//!   recipe with an AK4381, an AK4355 and an AK5380 input, and the converter
//!   frames it sends then reflect that choice, not the Tezro.
//! * The EEPROM image is synthesized from constraints alone: the subsystem
//!   pair, a 49.152 MHz crystal with four stereo DACs and an I2S converter,
//!   the converter pins as outputs, CCLK, the selects and the mute line
//!   high. Everything else is inert. The crystals are unknown, and so is
//!   whether the board sets "external S/PDIF out implemented" (CCS07 bit 0;
//!   the image keeps the reset value).
//! * An I2C transfer completes at once, where hardware takes about a third
//!   of a millisecond. A transfer to an address nothing answers is logged
//!   and leaves CCS12 alone; NAK handling is not described.
//! * An input GPIO pin reads its weak pull-up (1). What an input reads is
//!   "the pin"; board straps and pull-ups are not documented.
//! * Every complete converter frame is accepted and recorded; the parts
//!   ignore writes while their PDN pin is low, and how the board wires PDN
//!   is unknown. A frame of other than 16 bits, a read command, a wrong chip
//!   address or an undefined register is refused and counted. Analog
//!   behaviour (soft mute, attenuation, the mute line on GPIO 22, the rate
//!   fields) is not modelled: samples go to the host unchanged. Which parts
//!   the DM8 carries on which selects is inferred (board wiring).
//! * CCS1C reserved bits 6 and 1 read 0 and ignore writes. Bit 0, the
//!   multi-channel I2S serial clock, stops every engine when set; that the
//!   engines stop is inferred from the clock being stopped.
//! * The AC'97 index decodes as `(MT04 & 0x7e) >> 1` into a 64-entry file;
//!   only "bits 6:0 read/write" is established. Cycles finish at once, there
//!   is one codec (ID 0), ready after the cold reset bit clears; its part,
//!   reset values and volume and rate registers are unknown. In I2S mode and
//!   on the `dm8` variant no codec answers and cycles read zero.
//! * Underrun and overrun (MT1A, MT00 bit 3) are set only by a failed bus
//!   transfer, never by pacing. A failed transfer is not described either:
//!   the engine keeps its clock, plays or records silence, the PCI status
//!   records a master abort, and MT1A flags the engine. A transfer also
//!   fails when bus mastering is off in the command register.
//! * The second record engine (RDMA1) records silence, and the first
//!   records silence too: host audio input is not connected. Their buffers
//!   are written and their position and interval run.
//! * MT3C: every field but the validity flag (bit 15) must be written only
//!   while the S/PDIF transmitter is disabled (CCS07 bit 7 clear). The model
//!   enforces that rule: bit 15 stays writable, the rest ignore writes and
//!   log. The register reads 0000 at reset, where the chip gives x000.
//! * A masked source does not latch into MT00. The chip's wording suggests
//!   this; it does not say.
//! * Terminal counts are write-only and read 0. A write to an address or
//!   count register restarts the buffer position, a write to a terminal
//!   count the interval, also while running. Counts reload and the interval
//!   repeats.
//! * MT01: the external master clock (bit 4) is ignored and the rate code
//!   sets the pace. The reserved code 1101 keeps the previous rate. The
//!   128x/256x ratio and the crystal do not change timing; an impossible
//!   192 kHz setting is logged.
//! * Output routing (MT2C) and the MPU-401 UART store what is written and
//!   log it; the peak meters read zero.
//! * Power management: D1 is refused, D2 and D3hot only close both BARs (an
//!   access to a closed BAR floats high), and D3hot to D0 resets the
//!   registers and engines but not configuration space, which the chip also
//!   resets: a function cannot reset its own header without leaving the
//!   bus's decode table stale. Whether registers answer in D2 and D3hot is
//!   not described. The subsystem mask is at 0x42; the chip also shows it at
//!   0x40. A configuration read of a register nothing describes (BARs 2 and
//!   3, 0x40, 0x41, 0x43, 0x60) reads 0 and ignores writes, and is logged;
//!   so is a whole-dword write there, but a narrower write is not seen.
//! * The sample clock runs on host time, less the time the machine was
//!   stopped, not on the emulated CPU's progress. A jump of more than a
//!   second is not replayed through memory; the engine resumes at the
//!   present.
//! * Hold-up approximation, an emulator policy with no hardware
//!   counterpart: a timer more than one flow tick behind its deadline means
//!   the host held up the timer thread, so the card's clock is set back to
//!   the deadline (at most 200 ms in all) instead of replaying the time in
//!   one burst; on-time ticks win it back (see
//!   [`CardClock::absorb_stall`]). The sample clock falls behind wall time
//!   by the hold-ups.
//! * Read-ahead policy, an emulator policy with no hardware counterpart,
//!   as HAL2 has for the same IRIX habit: while an engine's interrupt is
//!   masked and the guest has read its address or count within 50 ms, the
//!   engine moves no more than 1.5 ms past the position read, and waits
//!   there; past 100 ms the time waited is dropped (see
//!   [`Engine::steer_limit`]). A guest the host holds up then hears a gap
//!   instead of the previous lap of its ring. An engine taking its
//!   interrupt, or not polled, runs on its clock. Only IRIX, which polls
//!   about once a millisecond, was measured. Risk for other guests: one
//!   that masks the interrupt, polls further apart than 1.5 ms and has
//!   written a whole ring ahead is held to 1.5 ms of audio per poll (10 ms
//!   polls give about 15% of real time, with gaps). The hold does not cover
//!   a guest held up between its read and its write: the engine may run
//!   1.5 ms past the position read before that write lands, while IRIX's
//!   write after its previous read reaches only about 1 ms past it (90
//!   frames past a position about 44 frames back), so a hold-up of more
//!   than about 1 ms there plays up to 0.5 ms of the previous lap.
//! * `dropped_frames` counts frames played into a host ring left with no
//!   room.
//! * The interleaved engine's slots, and the S/PDIF engine, play through one
//!   host stream with as many channels as the host device has (7.1, 5.1,
//!   quad or stereo). Which speaker a slot drives is not established; the
//!   model takes the usual host orders: 2 slots front left and right; 4
//!   adds the back pair; 6 is 5.1 (front, centre, low frequency, back); 8
//!   adds the side pair. A host with fewer speakers than the slots folds
//!   them: centre, back and side channels at -3 dB into the front pair and
//!   the low-frequency channel dropped, as the ITU-R BS.775 stereo downmix
//!   does, with no scaling down: a speaker's own channel plays at unity and
//!   a fold past full scale saturates. A side pair goes into the back pair
//!   of a host that has one, at unity, which BS.775 does not cover. All of
//!   that is a choice, not the chip's: IRIX plays stereo in two of eight
//!   slots, and a fold scaled to sum to unity played it 9.9 dB quiet. Extra
//!   host speakers stay silent. The S/PDIF pair adds, with saturation, to
//!   the front pair, frame for frame with the analog stream when both run.
//! * The stereo engines PDMA1 to PDMA3 are moved and counted, and heard
//!   nowhere.
//!
//! Ported from hw/audio/sgi-dm8.c in jamesbraid/qemu-sgi-origami.

use std::fmt;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex, MutexGuard};

use crate::config::AudioConfig;
use crate::dev::host_pcm::{CardClock, PcmOut};
use crate::devlog::LogModule;
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::pci::{Bar, PciConfig, PciConfigPolicy, PciFunction, PciFunctionState, PciRegisterPolicy, PCI_COMMAND_IO, PCI_INTA};
use crate::snapshot::{field, field_bool, field_u32, field_u64, field_u8, field_u8s, load_u16_slice, StateTable};
use crate::traits::{Device, Saveable};

pub const PCI_VENDOR_ICE: u16 = 0x1412;
pub const PCI_DEVICE_ICE1724: u16 = 0x1724;
pub const ICE1724_REVISION: u8 = 1;
/// Multimedia, audio device, programming interface 0.
const PCI_CLASS_AUDIO: u32 = 0x04_01_00;
/// The subsystem ID the DM8's EEPROM supplies, inferred for that board. A
/// guest driver that reads it at config 0x2e and finds 0x3630 takes the
/// Revolution-style converter recipe (AK4381 front, AK4355 surround). Using
/// it on the Tezro is a stated model choice, not Tezro evidence: the Tezro
/// board's subsystem ID, EEPROM and converters are unknown.
const SUBSYSTEM_DM8: u16 = 0x3630;
/// The chip's default when no EEPROM supplies one.
const SUBSYSTEM_NO_EEPROM: u16 = 0x1724;

pub const CCS_SIZE: u32 = 0x20;
pub const MT_SIZE: u32 = 0x80;

const PCI_COMMAND_BUS_MASTER: u32 = 0x4;
/// PCI status: interrupt pending, which the chip drives, and medium DEVSEL.
/// The capability-list bit is the header's, from the capability pointer.
const STATUS_INTX: u32 = 0x0008_0000;
const STATUS_MEDIUM_DEVSEL: u16 = 0x0200;
/// Parity, system error, received target and master abort: write one to
/// clear.
const STATUS_W1C: u16 = 0xf000;
const STATUS_MASTER_ABORT: u16 = 0x2000;
const CAP_POINTER_DWORD: usize = 13;
const PM_CAP_DWORD: usize = 0x80 / 4;
const PMCSR_DWORD: usize = PM_CAP_DWORD + 1;
const PMCSR_STATE: u32 = 3;
/// Capability ID 1, end of list, version 1.0 with D2 and no D1 or PME.
const PM_CAP_HEADER: u32 = 0x0401_0001;
const SVID_MASK_DWORD: usize = 0x40 / 4;
const SVID_WRITE_ENABLE: u32 = 0x80 << 16;
const SUBSYSTEM_DWORD: usize = 0x2c / 4;
const LINE_PIN_DWORD: usize = 15;

// Controller block registers (BAR0).
const CCS_CONTROL: u32 = 0x00;
const CCS_INT_MASK: u32 = 0x01;
const CCS_INT_STATUS: u32 = 0x02;
const CCS_SYS_CONFIG: u32 = 0x04;
const CCS_AC_CONFIG: u32 = 0x05;
const CCS_I2S_FEATURES: u32 = 0x06;
const CCS_SPDIF_CONFIG: u32 = 0x07;
const CCS_UART_TX_COUNT: u32 = 0x0a;
const CCS_UART_RX_COUNT: u32 = 0x0b;
const CCS_UART_DATA: u32 = 0x0c;
const CCS_UART_COMMAND: u32 = 0x0d;
const CCS_UART_SETTING: u32 = 0x0e;
const CCS_I2C_DEVICE: u32 = 0x10;
const CCS_I2C_BYTE: u32 = 0x11;
const CCS_I2C_DATA: u32 = 0x12;
const CCS_I2C_STATUS: u32 = 0x13;
const CCS_GPIO_DATA_LO: u32 = 0x14;
const CCS_GPIO_MASK_LO: u32 = 0x16;
const CCS_GPIO_DIR: u32 = 0x18;
const CCS_POWER_DOWN: u32 = 0x1c;
const CCS_GPIO_DATA_HI: u32 = 0x1e;
const CCS_GPIO_MASK_HI: u32 = 0x1f;

const CCS_INT_MPU_RX: u8 = 0x80;
const CCS_INT_MPU_TX: u8 = 0x20;
const CCS_INT_MT: u8 = 0x10;
const CCS_INT_STATUS_WRITABLE: u8 = 0xe0;
const CCS_CONTROL_SOFT_RESET: u8 = 0x80;
const CCS_AC_I2S: u8 = 0x80;
const CCS_SYS_CRYSTAL_49: u8 = 0x40;
const CCS_I2C_EEPROM: u8 = 0x80;
const CCS_PD_MT_CLOCK: u8 = 0x01;

// Multi-channel block registers (BAR1).
const MT_STATUS: u32 = 0x00;
const MT_RATE: u32 = 0x01;
const MT_FORMAT: u32 = 0x02;
const MT_MASK: u32 = 0x03;
const MT_AC97_INDEX: u32 = 0x04;
const MT_AC97_CMD: u32 = 0x05;
const MT_AC97_DATA: u32 = 0x06;
const MT_START: u32 = 0x18;
const MT_CHANNELS: u32 = 0x19;
const MT_FIFO_ERROR: u32 = 0x1a;
const MT_PAUSE: u32 = 0x1b;
const MT_ROUTING: u32 = 0x2c;
const MT_SPDIF_CTRL: u32 = 0x3c;
const MT_PEAK_INDEX: u32 = 0x3e;
const MT_PEAK_DATA: u32 = 0x3f;

const MT_AC97_COLD_RESET: u8 = 0x80;
const MT_AC97_WRITE: u8 = 0x20;
const MT_AC97_READ: u8 = 0x10;
const MT_AC97_READY: u8 = 0x08;
const MT_FORMAT_128X: u8 = 0x08;

// The Revolution 7.1 pin assignment.
const GPIO_CCLK: u32 = 1 << 1;
const GPIO_CDOUT: u32 = 1 << 3;
const GPIO_CS_FRONT: u32 = 1 << 5;
const GPIO_CS_SURROUND: u32 = 1 << 6;
const GPIO_MASK_ALL: u32 = 0x7f_ffff;

/// Engines are numbered by their bit in MT00, MT18, MT1A and MT1B. Bit 3 of
/// MT00 is the FIFO error source and has no engine.
const ENGINES: usize = 8;
const ENG_PDMAI: usize = 0;
const ENG_RDMA0: usize = 1;
const ENG_RDMA1: usize = 2;
const ENG_PDMA1: usize = 4;
const ENG_PDMA2: usize = 5;
const ENG_PDMA3: usize = 6;
const ENG_PDMA4: usize = 7;
const ENGINE_BITS: u8 = 0xf7;
const FIFO_ERROR_SOURCE: usize = 3;

/// A chunk is a whole number of frames for 2, 4, 6 and 8 slots.
const CHUNK_DWORDS: u32 = 3840;
/// Engines are brought up to date at least this often while they run.
const FLOW_NS: i64 = 2_000_000;
/// Most channels the host stream is opened with: the interleaved engine's
/// eight slots.
const MAX_HOST_CHANNELS: usize = 8;
const NS_PER_SEC: u128 = 1_000_000_000;
const MAX_CATCHUP_SECONDS: u64 = 1;
/// How far an engine whose position the guest steers by may run past the
/// position last read, and how long a read keeps steering it (see
/// [`Engine::steer_limit`]). HAL2 holds IRIX the same way, with 1 ms; here
/// 1 ms held the engine below real time whenever a poll came late. IRIX's
/// polls measured 1.00 ms apart at the median, p90 1.03 to 1.07 ms and p99
/// 1.1 to 2.8 ms on a loaded Linux host (p99 up to 13 ms on a loaded Mac),
/// so 1.5 ms sits above most poll gaps and below the 2 ms IRIX writes
/// ahead of the position it read, though not of the next one (see the
/// module notes).
const READ_AHEAD_NS: u64 = 1_500_000;
const STEER_NS: i64 = 50_000_000;
/// Most of an engine's clock held back while it waits for a poll; beyond
/// it the time is dropped rather than read later in one run.
const MAX_STEER_BACKLOG_NS: i64 = 100_000_000;

const EEPROM_SIZE: usize = 256;
const EEPROM_I2C_ADDRESS: u32 = 0x50;
const AC97_REGS: usize = 64;
const AK4381_REGS: usize = 5;
const AK4355_REGS: usize = 11;
const AK4381_CHIP_ADDRESS: u32 = 1;
const AK4355_CHIP_ADDRESS: u32 = 3;
/// Sample rates by MT01 bits 3:0; code 13 is not listed.
const RATES: [u32; 16] = [
    48000, 24000, 12000, 9600, 32000, 16000, 8000, 96000, 44100, 22050, 11025, 88200, 176400, 0, 192000, 64000,
];

/// The synthesized EEPROM image (see the module notes).
const EEPROM_IMAGE: [u8; 19] = [
    0x36, 0x30, 0x14, 0x12, // 0..3: subsystem pair
    0x13, 0x02, // 4: size, 5: version
    0x43, 0x80, 0x01, 0x01, // 6..9: CCS04 to CCS07
    0x7a, 0x00, 0x40, // 10..12: direction, GPIO 1, 3, 4, 5, 6, 22
    0x85, 0xff, 0xbf, // 13..15: write mask, those pins open
    0x72, 0x00, 0x40, // 16..18: data, lines idle, not muted
];

/// Which of the controller's two personalities this card has.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Variant {
    /// The DMediaPro DM8: AKM converters behind GPIO pins, an EEPROM.
    Dm8,
    /// A board with an AC'97 codec: no converter pins, no EEPROM.
    Ac97,
}

impl Variant {
    fn eeprom_fitted(self) -> bool {
        self == Variant::Dm8
    }
}

/// A register of a DMA engine.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Field {
    Addr,
    Count,
    Tc,
}

/// Register offsets of each engine's address, count and terminal count. The
/// count and terminal count of the interleaved engine are 19 bits in three
/// bytes; the stereo engines have 16 bits in two. Engine 3 has none.
const ENGINE_REGS: [(u32, u32, u32); ENGINES] = [
    (0x10, 0x14, 0x1c),
    (0x20, 0x24, 0x26),
    (0x30, 0x34, 0x36),
    (0, 0, 0),
    (0x70, 0x74, 0x76),
    (0x60, 0x64, 0x66),
    (0x50, 0x54, 0x56),
    (0x40, 0x44, 0x46),
];

/// The engine, register and byte an MT offset falls on.
fn engine_decode(off: u32) -> Option<(usize, Field, u32)> {
    for (i, &(addr, count, tc)) in ENGINE_REGS.iter().enumerate() {
        if addr == 0 {
            continue;
        }
        let wide = if i == ENG_PDMAI { 3 } else { 2 };
        if (addr..addr + 4).contains(&off) {
            return Some((i, Field::Addr, off - addr));
        }
        if (count..count + wide).contains(&off) {
            return Some((i, Field::Count, off - count));
        }
        if (tc..tc + wide).contains(&off) {
            return Some((i, Field::Tc, off - tc));
        }
    }
    None
}

#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Engine {
    /// Buffer base, bits 31:2.
    addr: u32,
    /// Buffer size in dwords, minus one.
    count: u32,
    /// Interrupt interval in dwords, minus one.
    tc: u32,
    /// Dwords moved in this pass of the buffer.
    pos: u32,
    /// Dwords moved in this interval.
    tc_pos: u32,
    /// Frames moved since the clock anchor.
    frames: u64,
    /// The card clock when the frame count started, in nanoseconds.
    anchor: i64,
    /// The clock is running for this engine.
    active: bool,
    /// When the guest last read this engine's address or count, and the
    /// frames moved then. Not saved: a restored engine waits for a new read.
    polled: Option<(i64, u64)>,
}

impl Engine {
    /// The most frames the engine may have moved at `now` when the guest
    /// steers it by the position it polls: `None` when it does not, either
    /// because it takes the engine's interrupt (`masked` false) or has not
    /// read the position for [`STEER_NS`].
    ///
    /// Emulator policy, not hardware: the chip reads on whatever software
    /// does. IRIX masks the engine's interrupt, reads the address register
    /// about once a millisecond and writes about 2 ms (90 frames at
    /// 44.1 kHz, 100 at 48 kHz) past what it read. While the host does not
    /// run the guest the card's clock goes on, and an engine that read on
    /// passed what was written and played the previous lap: a buzz under
    /// host load.
    fn steer_limit(&self, now: i64, masked: bool, rate: u32) -> Option<u64> {
        let (_, frames) = self.polled.filter(|&(at, _)| masked && now - at < STEER_NS)?;
        Some(frames + (u128::from(READ_AHEAD_NS) * u128::from(rate) / NS_PER_SEC) as u64)
    }

    fn field(&mut self, f: Field) -> &mut u32 {
        match f {
            Field::Addr => &mut self.addr,
            Field::Count => &mut self.count,
            Field::Tc => &mut self.tc,
        }
    }
}

/// What the converters and the engines have done, for tests and for
/// comparing against a guest driver's intent.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Stats {
    /// Frames the interleaved engine moved.
    pub analog_frames: u64,
    /// The last dword of the last analog chunk moved.
    pub analog_last: u32,
    pub spdif_frames: u64,
    pub capture_frames: u64,
    /// Frames the host stream had no room for.
    pub dropped_frames: u64,
    pub ak4381_writes: u32,
    pub ak4355_writes: u32,
    /// Converter frames refused.
    pub codec_rejects: u32,
}

struct State {
    ccs: [u8; CCS_SIZE as usize],
    mt: [u8; MT_SIZE as usize],
    pm_state: u8,
    irq_level: bool,
    eeprom: [u8; EEPROM_SIZE],
    eng: [Engine; ENGINES],
    rate: u32,
    /// The level on each converter pin after the last GPIO change.
    gpio_pins: u32,
    frame_bits: [u8; 2],
    frame_word: [u16; 2],
    ak4381: [u8; AK4381_REGS],
    ak4355: [u8; AK4355_REGS],
    ac97: [u16; AC97_REGS],
    stats: Stats,
    /// The timer that next brings the engines up to date, and which of
    /// them is current.
    timer: Option<TimerId>,
    timer_gen: u64,
    /// When the pending timer is due, on the card's clock.
    timer_due: i64,
    /// Samples read from or written to guest memory, a chunk at a time.
    scratch: Vec<u8>,
    /// A transfer is under way with `state` let go; nothing else moves the
    /// engines until it ends (see [`Inner::lock_idle`]).
    moving: bool,
}

impl State {
    fn new(variant: Variant) -> Self {
        let mut eeprom = [0xff; EEPROM_SIZE];
        if variant.eeprom_fitted() {
            eeprom[..EEPROM_IMAGE.len()].copy_from_slice(&EEPROM_IMAGE);
        }
        let mut s = Self {
            ccs: [0; CCS_SIZE as usize],
            mt: [0; MT_SIZE as usize],
            pm_state: 0,
            irq_level: false,
            eeprom,
            eng: [Engine::default(); ENGINES],
            rate: 48000,
            gpio_pins: 0,
            frame_bits: [0; 2],
            frame_word: [0; 2],
            ak4381: [0; AK4381_REGS],
            ak4355: [0; AK4355_REGS],
            ac97: [0; AC97_REGS],
            stats: Stats::default(),
            timer: None,
            timer_gen: 0,
            timer_due: 0,
            scratch: Vec::new(),
            moving: false,
        };
        s.reset_registers(variant, false);
        s.reset_codecs();
        s
    }

    /// The registers as the chip powers up, or after a soft reset, which
    /// keeps CCS04 to CCS07 (see the module notes).
    fn reset_registers(&mut self, variant: Variant, soft: bool) {
        let mut keep = [0u8; 4];
        keep.copy_from_slice(&self.ccs[CCS_SYS_CONFIG as usize..CCS_SYS_CONFIG as usize + 4]);
        self.ccs = [0; CCS_SIZE as usize];
        self.mt = [0; MT_SIZE as usize];
        self.ccs[CCS_INT_MASK as usize] = 0xfe;
        self.ccs[CCS_SYS_CONFIG as usize] = 0x0f;
        self.ccs[CCS_I2S_FEATURES as usize] = 0x01;
        self.ccs[CCS_SPDIF_CONFIG as usize] = 0x01;
        self.ccs[CCS_GPIO_MASK_LO as usize] = 0xff;
        self.ccs[CCS_GPIO_MASK_LO as usize + 1] = 0xff;
        self.ccs[CCS_GPIO_MASK_HI as usize] = 0xff;
        if variant.eeprom_fitted() {
            self.ccs[CCS_I2C_STATUS as usize] = CCS_I2C_EEPROM;
        }
        self.mt[MT_MASK as usize] = 0xff;
        self.rate = 48000;
        self.eng = [Engine::default(); ENGINES];
        self.gpio_pins = self.pins();
        self.frame_bits = [0; 2];
        if soft {
            self.ccs[CCS_SYS_CONFIG as usize..CCS_SYS_CONFIG as usize + 4].copy_from_slice(&keep);
            self.ccs[CCS_CONTROL as usize] = CCS_CONTROL_SOFT_RESET;
        }
    }

    fn reset_codecs(&mut self) {
        self.ak4381 = [0; AK4381_REGS];
        self.ak4355 = [0; AK4355_REGS];
        // Both parts reset their attenuation registers to full level.
        self.ak4381[3] = 0xff;
        self.ak4381[4] = 0xff;
        self.ak4355[4..10].fill(0xff);
        self.ac97 = [0; AC97_REGS];
    }

    fn ccs(&self, reg: u32) -> u8 {
        self.ccs[reg as usize]
    }

    fn mt(&self, reg: u32) -> u8 {
        self.mt[reg as usize]
    }

    /// Dwords in one frame of an engine: the interleaved engine carries 8,
    /// 6, 4 or 2 slots by MT19, every other engine a stereo pair.
    fn frame_dwords(&self, idx: usize) -> u32 {
        if idx == ENG_PDMAI {
            8 - 2 * (self.mt(MT_CHANNELS) & 3) as u32
        } else {
            2
        }
    }

    fn engine_available(&self, idx: usize) -> bool {
        let streams = self.mt(MT_CHANNELS) & 3;
        match idx {
            ENG_PDMA1 => streams == 3,
            ENG_PDMA2 => streams >= 2,
            ENG_PDMA3 => streams >= 1,
            _ => true,
        }
    }

    fn engine_clocked(&self, idx: usize) -> bool {
        self.mt(MT_START) & (1 << idx) != 0
            && self.mt(MT_PAUSE) & (1 << idx) == 0
            && self.engine_available(idx)
            && self.ccs(CCS_POWER_DOWN) & CCS_PD_MT_CLOCK == 0
    }

    fn count_mask(idx: usize) -> u32 {
        if idx == ENG_PDMAI {
            0x7_ffff
        } else {
            0xffff
        }
    }

    /// A source raises its MT00 bit unless MT03 masks it. Approximation: a
    /// masked source does not latch (see the module notes).
    fn raise(&mut self, bit: usize) {
        if self.mt(MT_MASK) & (1 << bit) == 0 {
            self.mt[MT_STATUS as usize] |= 1 << bit;
        }
    }

    fn gpio_data(&self) -> u32 {
        u32::from(self.ccs(CCS_GPIO_DATA_LO))
            | u32::from(self.ccs(CCS_GPIO_DATA_LO + 1)) << 8
            | u32::from(self.ccs(CCS_GPIO_DATA_HI) & 0x7f) << 16
    }

    fn gpio_direction(&self) -> u32 {
        (u32::from(self.ccs(CCS_GPIO_DIR))
            | u32::from(self.ccs(CCS_GPIO_DIR + 1)) << 8
            | u32::from(self.ccs(CCS_GPIO_DIR + 2)) << 16)
            & GPIO_MASK_ALL
    }

    /// The level on each pin: an output drives its last written value, an
    /// input floats to its weak pull-up.
    fn pins(&self) -> u32 {
        let dir = self.gpio_direction();
        (self.gpio_data() & dir) | (GPIO_MASK_ALL & !dir)
    }

    fn ak_reg(&mut self, chip: u32, reg: usize) -> Option<&mut u8> {
        if chip == AK4381_CHIP_ADDRESS {
            self.ak4381.get_mut(reg)
        } else {
            self.ak4355.get_mut(reg)
        }
    }

    fn ac97_ready(&self, variant: Variant) -> bool {
        variant == Variant::Ac97 && self.ccs(CCS_AC_CONFIG) & CCS_AC_I2S == 0 && self.mt(MT_AC97_CMD) & MT_AC97_COLD_RESET == 0
    }
}

/// The ICE1724 audio function. The timer callback holds a weak reference to the
/// same state.
pub struct Ice1724 {
    inner: Arc<Inner>,
}

impl Deref for Ice1724 {
    type Target = Inner;
    fn deref(&self) -> &Inner {
        &self.inner
    }
}

impl Ice1724 {
    pub fn new(variant: Variant) -> Self {
        Self::with_clock(variant, CardClock::new())
    }

    /// The card on a sample clock of the caller's.
    pub(crate) fn with_clock(variant: Variant, clock: CardClock) -> Self {
        Self { inner: Arc::new_cyclic(|me| Inner::new(variant, me.clone(), clock)) }
    }

    /// Name the card's state lock in the lock registry, which holds only a
    /// weak reference to it.
    pub fn register_locks(&self) {
        let me = Arc::downgrade(&self.inner);
        crate::locks::register_lock("ice1724::state", move || me.upgrade().is_some_and(|i| i.state.is_locked()));
    }
}

/// The header dwords the chip defines beyond the common ones: the capability
/// pointer and the power-management capability (read-only but for the state
/// in PMCSR bits 1:0), and the subsystem mask, of which only PCI42 bit 7
/// exists. Every other dword from 0x40 up reads 0 whatever the guest wrote
/// (see `Inner::config_read`).
const CONFIG_REGISTERS: [PciRegisterPolicy; 4] = [
    PciRegisterPolicy { reg_dword: CAP_POINTER_DWORD, initial: PM_CAP_DWORD as u32 * 4, writable_mask: 0 },
    PciRegisterPolicy { reg_dword: SVID_MASK_DWORD, initial: 0, writable_mask: SVID_WRITE_ENABLE },
    PciRegisterPolicy { reg_dword: PM_CAP_DWORD, initial: PM_CAP_HEADER, writable_mask: 0 },
    PciRegisterPolicy { reg_dword: PMCSR_DWORD, initial: 0, writable_mask: PMCSR_STATE },
];

/// Lock order: `tick_pass`, then `state`, then `sink`; `stopped_at` (in
/// [`CardClock`]) is a leaf, taken by `now` with or without `state` held. Engines are moved by
/// the thread that holds `state`, but a transfer to or from guest memory runs
/// with `state` let go, so the bridge and memory locks below the card are
/// never taken inside it; `moving` keeps every other path that reads or moves
/// an engine out until the transfer ends ([`Inner::lock_idle`]). The
/// interrupt line is driven on a change, with `state`
/// held, as the other PCI functions do.
pub struct Inner {
    variant: Variant,
    pci: PciFunctionState,
    state: Mutex<State>,
    /// Signalled when a transfer that let go of `state` ends.
    idle: Condvar,
    /// The sample clock and the timer bookkeeping shared with other audio
    /// cards; reached through `Deref`.
    clock: CardClock,
    timers: OnceLock<Arc<TimerManager>>,
    /// `[audio]` settings once host output is enabled.
    audio_cfg: Mutex<Option<AudioConfig>>,
    sink: Mutex<Option<PcmOut>>,
    /// Accesses that nothing establishes, counted.
    unexpected: AtomicU64,
    /// The messages behind them, for tests.
    #[cfg(test)]
    seen: Mutex<Vec<String>>,
    me: Weak<Inner>,
}

impl Deref for Inner {
    type Target = CardClock;
    fn deref(&self) -> &CardClock {
        &self.clock
    }
}

impl Inner {
    fn new(variant: Variant, me: Weak<Inner>, clock: CardClock) -> Self {
        // The header as the chip powers up (see [`config_policy`]); the
        // board's subsystem pair survives a function reset.
        let config = PciConfig::new_with_policy(
            PCI_VENDOR_ICE,
            PCI_DEVICE_ICE1724,
            PCI_CLASS_AUDIO,
            ICE1724_REVISION,
            [Bar::Io(CCS_SIZE), Bar::Io(MT_SIZE), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused],
            PCI_INTA,
            PciConfigPolicy {
                status: STATUS_MEDIUM_DEVSEL,
                status_w1c_mask: STATUS_W1C,
                command_mask: (PCI_COMMAND_IO | PCI_COMMAND_BUS_MASTER) as u16,
                // The latency timer keeps bits 7:3; the cache line size
                // reads 0.
                latency_mask: 0xf8,
                cache_line_mask: 0,
                registers: &CONFIG_REGISTERS,
            },
        );
        let id = if variant.eeprom_fitted() { SUBSYSTEM_DM8 } else { SUBSYSTEM_NO_EEPROM };
        config.set_subsystem_identity(u32::from(id) << 16 | u32::from(PCI_VENDOR_ICE));
        let pci = PciFunctionState::new(config);
        // The chip powers up with 0xff in the interrupt line.
        pci.config().write(LINE_PIN_DWORD, 0xff);
        Self {
            variant,
            pci,
            state: Mutex::new(State::new(variant)),
            idle: Condvar::new(),
            clock,
            timers: OnceLock::new(),
            audio_cfg: Mutex::new(None),
            sink: Mutex::new(None),
            unexpected: AtomicU64::new(0),
            #[cfg(test)]
            seen: Mutex::new(Vec::new()),
            me,
        }
    }

    /// Log an access or value that nothing establishes, and count it.
    fn note(&self, args: fmt::Arguments) {
        self.unexpected.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        self.seen.lock().push(args.to_string());
        crate::dlog!(LogModule::Ice1724, "ice1724: {args}");
    }

    /// The state lock, once no transfer is under way. A transfer lets go of
    /// the lock while it is in guest memory, and the engines' positions are
    /// not settled until it is back, so every path that reads or moves an
    /// engine waits here, as it would have waited on the lock.
    fn lock_idle(&self) -> MutexGuard<'_, State> {
        let mut s = self.state.lock();
        while s.moving {
            self.idle.wait(&mut s);
        }
        s
    }

    /// Run `f`, a transfer to or from guest memory, with the state lock let
    /// go. The memory path below the card never calls back into it, but it
    /// takes locks of its own that other threads hold while they reach the
    /// card.
    fn dma_unlocked<R>(&self, s: &mut MutexGuard<'_, State>, f: impl FnOnce() -> R) -> R {
        // A transfer that reached the card's own registers would wait on
        // itself; none does. A panic in `f` still clears the flag, so no
        // later access waits on a transfer that is gone.
        s.moving = true;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| MutexGuard::unlocked(&mut *s, f)));
        s.moving = false;
        self.idle.notify_all();
        match r {
            Ok(r) => r,
            Err(e) => std::panic::resume_unwind(e),
        }
    }

    /// How many accesses or values nothing establishes the card has seen.
    pub fn unexpected(&self) -> u64 {
        self.unexpected.load(Ordering::Relaxed)
    }

    /// Play the output engines through the host with `audio`'s settings. The
    /// host stream opens when the machine starts, before the guest runs.
    pub fn enable_output(&self, audio: AudioConfig) {
        *self.audio_cfg.lock() = Some(audio);
    }

    /// Give the card the machine's timers. Called before `Device::start`.
    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.timers.set(tm);
    }

    // ---- clock and timer ----

    /// Whether the timer may be armed: started, on a real clock, with a
    /// manager.
    fn timers_live(&self) -> Option<&Arc<TimerManager>> {
        if !self.armed.load(Ordering::SeqCst) || !self.runs_by_itself() {
            return None;
        }
        self.timers.get()
    }

    /// When the next running engine's interrupt interval ends, or a flow
    /// tick, whichever is first; `None` when no engine runs.
    fn next_deadline(&self, s: &State, now: i64) -> Option<i64> {
        let mut next: Option<i64> = None;
        for (i, e) in s.eng.iter().enumerate() {
            if !e.active {
                continue;
            }
            let mut when = now + FLOW_NS;
            if s.mt(MT_MASK) & (1 << i) == 0 {
                let frames = (e.tc - e.tc_pos + 1).div_ceil(s.frame_dwords(i));
                let ns = (u128::from(e.frames + u64::from(frames)) * NS_PER_SEC).div_ceil(u128::from(s.rate)) as i64;
                when = when.min((e.anchor + ns).max(now));
            }
            next = Some(next.map_or(when, |n| n.min(when)));
        }
        next
    }

    /// Replace the pending timer with one for the engines' present state.
    fn arm_timer(&self, s: &mut State, now: i64) {
        s.timer_gen += 1;
        if let Some(id) = s.timer.take() {
            if let Some(tm) = self.timers.get() {
                tm.remove(id);
            }
        }
        let Some(tm) = self.timers_live() else { return };
        let Some(when) = self.next_deadline(s, now) else { return };
        s.timer_due = when;
        let (me, gen) = (self.me.clone(), s.timer_gen);
        s.timer = Some(tm.add_one_shot(Duration::from_nanos((when - now).max(0) as u64), (), move |_| {
            let Some(i) = me.upgrade() else { return TimerReturn::Delete };
            i.timer_fire(gen)
        }));
    }

    /// The timer fired: bring the engines up to date and wait for the next
    /// deadline.
    fn timer_fire(&self, gen: u64) -> TimerReturn {
        let _pass = self.tick_pass.lock();
        if self.stopped() {
            return TimerReturn::Delete;
        }
        let mut s = self.lock_idle();
        if s.timer_gen != gen {
            return TimerReturn::Delete;
        }
        // A timer far behind its deadline means the host held up this
        // thread; the clock is set back rather than replaying the time in a
        // burst (see the module notes).
        let now = self.absorb_stall(s.timer_due, FLOW_NS);
        self.sync(&mut s, now);
        match self.next_deadline(&s, now) {
            Some(when) => {
                s.timer_due = when;
                TimerReturn::RescheduleOneShot(Duration::from_nanos((when - now).max(0) as u64))
            }
            None => {
                s.timer = None;
                TimerReturn::Delete
            }
        }
    }

    fn pause(&self) {
        self.stop_clock();
        self.armed.store(false, Ordering::SeqCst);
        {
            let mut s = self.state.lock();
            s.timer_gen += 1;
            if let Some(id) = s.timer.take() {
                if let Some(tm) = self.timers.get() {
                    tm.remove(id);
                }
            }
        }
        // Once this returns no timer callback is under way.
        drop(self.tick_pass.lock());
    }

    fn resume(&self) {
        self.open_sink();
        self.start_clock();
        self.armed.store(true, Ordering::SeqCst);
        let now = self.now();
        let mut s = self.lock_idle();
        self.arm_timer(&mut s, now);
    }

    // ---- host audio ----

    /// Open the host stream, once, when output is enabled. It is as wide as
    /// `[audio] channels` asks; the card folds its slots down to that.
    fn open_sink(&self) {
        let Some(cfg) = self.audio_cfg.lock().clone() else { return };
        // Read before the sink lock: engines take the state lock, then the
        // sink's.
        let rate = self.state.lock().rate;
        let mut sink = self.sink.lock();
        if sink.is_some() {
            return;
        }
        let channels = match cfg.channels {
            n @ (2 | 4 | 6 | 8) => n.min(MAX_HOST_CHANNELS),
            _ => 2,
        };
        let host = crate::host_audio::open_output(&cfg, "ICE1724 audio", channels, rate, "IRIS_ICE1724_AUDIO_CAPTURE");
        crate::dlog!(LogModule::Ice1724, "ice1724: host audio {}", if host.is_some() { "open" } else { "failed to open" });
        *sink = Some(PcmOut::new(host));
    }

    // ---- interrupts ----

    /// Recompute the controller's interrupt bits and drive the line if its
    /// level changed. Inferred: the macro bit (CCS02 bit 4) is the OR of the
    /// unmasked MT00 bits, and INTA# is asserted while any unmasked CCS02
    /// bit is set.
    fn update_irq(&self, s: &mut State) {
        if s.mt(MT_STATUS) & !s.mt(MT_MASK) != 0 {
            s.ccs[CCS_INT_STATUS as usize] |= CCS_INT_MT;
        } else {
            s.ccs[CCS_INT_STATUS as usize] &= !CCS_INT_MT;
        }
        let level = s.ccs(CCS_INT_STATUS) & !s.ccs(CCS_INT_MASK) & (CCS_INT_MPU_RX | CCS_INT_MPU_TX | CCS_INT_MT) != 0;
        if level != s.irq_level {
            s.irq_level = level;
            if let Some(l) = self.pci.irq() {
                l.set(level);
            }
        }
    }

    // ---- DMA engines ----

    /// Move `dwords` through one engine, no more than one chunk, and no
    /// further than the end of its buffer per DMA transfer.
    fn move_dwords(&self, s: &mut MutexGuard<'_, State>, idx: usize, dwords: u32) {
        debug_assert!(dwords <= CHUNK_DWORDS);
        let record = idx == ENG_RDMA0 || idx == ENG_RDMA1;
        let frame = s.frame_dwords(idx);
        let bytes = dwords as usize * 4;
        let mut buf = std::mem::take(&mut s.scratch);
        // Recording writes silence: no host input is connected.
        buf.clear();
        buf.resize(bytes, 0);
        let dma = self.pci.dma().filter(|_| self.pci.config().command() & PCI_COMMAND_BUS_MASTER != 0);
        let mut done = 0u32;
        let mut failed = false;
        while done < dwords {
            let e = &mut s.eng[idx];
            let span = (dwords - done).min(e.count + 1 - e.pos);
            let at = u64::from(e.addr & !3) + 4 * u64::from(e.pos);
            let part = &mut buf[done as usize * 4..(done + span) as usize * 4];
            let ok = match dma {
                Some(d) if record => self.dma_unlocked(s, || d.dma_write(at, part).is_ok()),
                Some(d) => self.dma_unlocked(s, || d.dma_read(at, part).is_ok()),
                None => false,
            };
            let e = &mut s.eng[idx];
            if !ok {
                failed = true;
                if !record {
                    part.fill(0);
                }
            }
            done += span;
            e.pos += span;
            if e.pos > e.count {
                e.pos = 0;
            }
            let interval = e.tc + 1;
            let moved = e.tc_pos + span;
            e.tc_pos = moved % interval;
            if moved / interval != 0 {
                s.raise(idx);
            }
        }

        if failed {
            // Approximation: see the module notes on a failed transfer.
            self.note(format_args!("DMA engine {idx} transfer failed"));
            self.pci.config().set_status_bits(STATUS_MASTER_ABORT);
            s.mt[MT_FIFO_ERROR as usize] |= 1 << idx;
            s.raise(FIFO_ERROR_SOURCE);
        }

        let frames = u64::from(dwords / frame);
        match idx {
            ENG_PDMAI => {
                if let Some(sink) = self.sink.lock().as_mut() {
                    // Each slot's top 16 bits; the host side folds the slots
                    // to its speakers.
                    sink.analog_stage(frame as usize).extend(buf.as_chunks::<4>().0.iter().map(|d| i16::from_le_bytes([d[2], d[3]])));
                }
                if !failed {
                    s.stats.analog_last = u32::from_le_bytes(buf[bytes - 4..].try_into().unwrap());
                }
                s.stats.analog_frames += frames;
            }
            ENG_PDMA4 => {
                if let Some(sink) = self.sink.lock().as_mut() {
                    sink.spdif_stage().extend(buf.as_chunks::<4>().0.iter().map(|d| i16::from_le_bytes([d[2], d[3]])));
                }
                s.stats.spdif_frames += frames;
            }
            ENG_RDMA0 => s.stats.capture_frames += frames,
            _ => {}
        }
        s.scratch = buf;
    }

    /// Move the frames an engine's clock has made due since its anchor.
    fn advance(&self, s: &mut MutexGuard<'_, State>, idx: usize, now: i64) {
        let frame = s.frame_dwords(idx);
        let e = &s.eng[idx];
        let due = (u128::from((now - e.anchor).max(0) as u64) * u128::from(s.rate) / NS_PER_SEC) as u64;
        if due > e.frames + u64::from(s.rate) * MAX_CATCHUP_SECONDS {
            // Approximation: a clock jump of more than a second is not
            // replayed through memory; the engine resumes at the present.
            self.note(format_args!("DMA clock jump skipped"));
            s.eng[idx].frames = due;
            return;
        }
        let masked = s.mt(MT_MASK) & (1 << idx) != 0;
        let rate = s.rate;
        let e = &mut s.eng[idx];
        let due = match e.steer_limit(now, masked, rate) {
            Some(limit) if due > limit => {
                // Wait for the guest. Clock time held back past the bound
                // is dropped, so a late poll does not let it out in one run.
                let reached = e.anchor + (u128::from(limit) * NS_PER_SEC).div_ceil(u128::from(rate)) as i64;
                e.anchor += (now - reached - MAX_STEER_BACKLOG_NS).max(0);
                limit
            }
            _ => due,
        };
        while s.eng[idx].frames < due {
            let frames = ((due - s.eng[idx].frames) as u32).min(CHUNK_DWORDS / frame);
            self.move_dwords(s, idx, frames * frame);
            s.eng[idx].frames += u64::from(frames);
        }
    }

    /// Bring every running engine up to the present.
    fn sync(&self, s: &mut MutexGuard<'_, State>, now: i64) {
        for i in 0..ENGINES {
            if s.eng[i].active {
                self.advance(s, i, now);
            }
        }
        if let Some(sink) = self.sink.lock().as_mut() {
            if let Some(dropped) = sink.flush(s.rate, s.eng[ENG_PDMAI].active, s.eng[ENG_PDMA4].active) {
                s.stats.dropped_frames += dropped;
            }
        }
        self.update_irq(s);
    }

    /// Follow a change of start, pause, availability, rate or clock gating:
    /// engines that begin running anchor their clock at the present, and
    /// every running engine's anchor restarts when `restart` is set.
    /// Callers sync before they change the state.
    fn clock_changed(&self, s: &mut State, now: i64, restart: bool) {
        for i in 0..ENGINES {
            let clocked = s.engine_clocked(i);
            let e = &mut s.eng[i];
            if clocked && (!e.active || restart) {
                e.anchor = now;
                e.frames = 0;
                e.polled = None;
            }
            e.active = clocked;
        }
        self.arm_timer(s, now);
    }

    fn apply_rate(&self, s: &mut State) {
        let rate_reg = s.mt(MT_RATE);
        let code = rate_reg & 0xf;
        let rate = RATES[code as usize];
        if rate_reg & 0x10 != 0 {
            // Approximation: the external clock has no source here.
            self.note(format_args!("external master clock is not modelled"));
        }
        if rate == 0 {
            self.note(format_args!("reserved sample rate code {code}"));
            return;
        }
        if code == 14 && s.ccs(CCS_SYS_CONFIG) & CCS_SYS_CRYSTAL_49 == 0 && s.mt(MT_FORMAT) & MT_FORMAT_128X == 0 {
            self.note(format_args!("192 kHz needs a 49.152 MHz crystal or 128x"));
        }
        s.rate = rate;
    }

    // ---- converters behind the GPIO pins ----

    fn codec_latch(&self, s: &mut State, port: usize) {
        let chip = if port == 1 { AK4355_CHIP_ADDRESS } else { AK4381_CHIP_ADDRESS };
        let word = s.frame_word[port];
        let header = u32::from(word >> 8);
        let reg = (header & 0x1f) as usize;
        if s.frame_bits[port] == 0 {
            // A select pulse with no clock is not a frame.
            return;
        }
        if s.frame_bits[port] != 16 {
            self.note(format_args!("converter frame of {} bits ignored", s.frame_bits[port]));
            s.stats.codec_rejects += 1;
            return;
        }
        // The chip address is fixed in each part and the frame is write-only.
        if header >> 6 != chip || header & 0x20 == 0 {
            self.note(format_args!("converter frame {word:#06x} is not a write to chip {chip}"));
            s.stats.codec_rejects += 1;
            return;
        }
        let Some(slot) = s.ak_reg(chip, reg) else {
            // The parts say these addresses must not be written.
            self.note(format_args!("write to undefined register {reg} of chip {chip}"));
            s.stats.codec_rejects += 1;
            return;
        };
        *slot = word as u8;
        if chip == AK4381_CHIP_ADDRESS {
            s.stats.ak4381_writes += 1;
        } else {
            s.stats.ak4355_writes += 1;
        }
    }

    /// Follow the converter pins after a GPIO register change. Each part
    /// sees the shared CCLK and CDOUT lines and its own select: a frame
    /// starts when the select falls, takes a bit on each rising CCLK edge
    /// while the select is held across the change, and takes effect when
    /// the select rises. Edges that arrive in one register write are taken
    /// as the parts' timing orders them: a select that rises or falls with
    /// CCLK does not see that clock edge. Approximation: every complete
    /// frame is accepted (see the module notes).
    fn codec_pins(&self, s: &mut State) {
        const SELECTS: [u32; 2] = [GPIO_CS_FRONT, GPIO_CS_SURROUND];
        let pins = s.pins();
        let old = s.gpio_pins;
        let clock_rise = pins & GPIO_CCLK != 0 && old & GPIO_CCLK == 0;
        s.gpio_pins = pins;
        if self.variant == Variant::Ac97 {
            return;
        }
        for (port, sel) in SELECTS.into_iter().enumerate() {
            let was = old & sel == 0;
            let now = pins & sel == 0;
            if clock_rise && was && now {
                if s.frame_bits[port] < 16 {
                    s.frame_word[port] = s.frame_word[port] << 1 | u16::from(pins & GPIO_CDOUT != 0);
                }
                s.frame_bits[port] = s.frame_bits[port].saturating_add(1);
            }
            if now && !was {
                s.frame_bits[port] = 0;
                s.frame_word[port] = 0;
            } else if was && !now {
                self.codec_latch(s, port);
                s.frame_bits[port] = 0;
            }
        }
    }

    // ---- I2C port and EEPROM ----

    fn i2c_start(&self, s: &mut State) {
        let cmd = s.ccs(CCS_I2C_DEVICE);
        let device = u32::from(cmd >> 1);
        let write = cmd & 1 != 0;
        // Approximation: a transfer finishes at once; one nothing answers
        // is logged and leaves the data register alone.
        if device != EEPROM_I2C_ADDRESS || !self.variant.eeprom_fitted() {
            self.note(format_args!("I2C {} of device {device:#04x} has no answer modelled", if write { "write" } else { "read" }));
            return;
        }
        let at = s.ccs(CCS_I2C_BYTE) as usize;
        if write {
            s.eeprom[at] = s.ccs(CCS_I2C_DATA);
        } else {
            s.ccs[CCS_I2C_DATA as usize] = s.eeprom[at];
        }
    }

    // ---- AC'97 port ----

    fn ac97_cycle(&self, s: &mut State, cmd: u8) {
        let index = ((s.mt(MT_AC97_INDEX) & 0x7e) >> 1) as usize;
        if s.ccs(CCS_AC_CONFIG) & CCS_AC_I2S != 0 {
            self.note(format_args!("AC'97 cycle in I2S mode has no meaning"));
            return;
        }
        if !s.ac97_ready(self.variant) || cmd & 3 != 0 {
            // Approximation: only one codec, ready once the cold reset is
            // released, exists. A cycle with none to answer does nothing
            // and reads back zero.
            self.note(format_args!("AC'97 cycle with no codec to answer (command {cmd:#04x})"));
            if cmd & MT_AC97_READ != 0 {
                s.mt[MT_AC97_DATA as usize] = 0;
                s.mt[MT_AC97_DATA as usize + 1] = 0;
            }
            return;
        }
        if cmd & MT_AC97_WRITE != 0 {
            s.ac97[index] = u16::from(s.mt(MT_AC97_DATA)) | u16::from(s.mt(MT_AC97_DATA + 1)) << 8;
        } else {
            let v = s.ac97[index];
            s.mt[MT_AC97_DATA as usize] = v as u8;
            s.mt[MT_AC97_DATA as usize + 1] = (v >> 8) as u8;
        }
    }

    // ---- reset ----

    /// RST#: the chip as it powers up, configuration space included. The
    /// machine calls this after the bridge has reset the function's header
    /// (the header's interrupt line is set here, after that reset, as the
    /// chip powers up with 0xff there). The EEPROM keeps what was written to
    /// it, and the converter counters keep counting.
    pub fn reset(&self) {
        let now = self.now();
        {
            let mut s = self.lock_idle();
            self.reset_state(&mut s, now);
        }
        self.pci.config().write(LINE_PIN_DWORD, 0xff);
    }

    fn reset_state(&self, s: &mut State, now: i64) {
        s.reset_registers(self.variant, false);
        s.reset_codecs();
        s.pm_state = 0;
        self.clock_changed(s, now, true);
        self.update_irq(s);
    }

    fn soft_reset(&self, s: &mut State, now: i64) {
        s.reset_registers(self.variant, true);
        self.clock_changed(s, now, true);
        self.update_irq(s);
    }

    // ---- controller block registers (BAR0) ----

    fn ccs_read(&self, s: &State, off: u32) -> u8 {
        let value = s.ccs(off);
        match off {
            CCS_UART_TX_COUNT | CCS_UART_RX_COUNT => 0,
            CCS_UART_DATA | CCS_UART_COMMAND => {
                self.note(format_args!("MPU-401 UART read at CCS {off:#04x}"));
                value
            }
            CCS_GPIO_DATA_LO | 0x15 => (s.pins() >> ((off - CCS_GPIO_DATA_LO) * 8)) as u8,
            CCS_GPIO_DATA_HI => (s.pins() >> 16) as u8,
            0x1a => value & 0x7f,
            0x03 | 0x08 | 0x09 | 0x0f | 0x1b | 0x1d => {
                self.note(format_args!("read of unassigned CCS {off:#04x}"));
                0
            }
            _ => value,
        }
    }

    fn ccs_write(&self, s: &mut State, off: u32, v: u8, now: i64) {
        let o = off as usize;
        match off {
            CCS_CONTROL => {
                s.ccs[o] = v;
                if v & CCS_CONTROL_SOFT_RESET != 0 {
                    self.soft_reset(s, now);
                }
            }
            CCS_INT_MASK => s.ccs[o] = v,
            // Write one to clear; the macro bit follows MT00.
            CCS_INT_STATUS => s.ccs[o] &= !(v & CCS_INT_STATUS_WRITABLE),
            CCS_SYS_CONFIG | CCS_I2S_FEATURES | CCS_I2C_BYTE | CCS_I2C_DATA => s.ccs[o] = v,
            CCS_GPIO_DIR | 0x19 => {
                s.ccs[o] = v;
                self.codec_pins(s);
            }
            0x1a => {
                s.ccs[o] = v & 0x7f;
                self.codec_pins(s);
            }
            CCS_AC_CONFIG => s.ccs[o] = v & 0x82,
            // Bit 6, the transmitter-implemented flag, is read only.
            CCS_SPDIF_CONFIG => s.ccs[o] = (s.ccs[o] & 0x40) | (v & 0xbf),
            CCS_UART_DATA | CCS_UART_COMMAND | CCS_UART_SETTING => {
                s.ccs[o] = v;
                self.note(format_args!("MPU-401 UART write at CCS {off:#04x}"));
            }
            CCS_UART_TX_COUNT | CCS_UART_RX_COUNT => self.note(format_args!("write to read-only CCS {off:#04x}")),
            CCS_I2C_DEVICE => {
                s.ccs[o] = v;
                self.i2c_start(s);
            }
            // Bit 7 is the EEPROM strap and bit 0 is busy: both read only.
            CCS_I2C_STATUS => s.ccs[o] = (s.ccs[o] & 0x81) | (v & 0x02),
            CCS_GPIO_DATA_LO | 0x15 => {
                // Only the bits whose write-mask bit is clear take the value.
                let mask = s.ccs[(CCS_GPIO_MASK_LO + off - CCS_GPIO_DATA_LO) as usize];
                s.ccs[o] = (s.ccs[o] & mask) | (v & !mask);
                self.codec_pins(s);
            }
            CCS_GPIO_DATA_HI => {
                let mask = s.ccs(CCS_GPIO_MASK_HI);
                s.ccs[o] = (s.ccs[o] & mask) | (v & !mask & 0x7f);
                self.codec_pins(s);
            }
            CCS_GPIO_MASK_LO | 0x17 | CCS_GPIO_MASK_HI => s.ccs[o] = v,
            CCS_POWER_DOWN => {
                s.ccs[o] = v & 0xbd;
                self.clock_changed(s, now, false);
            }
            // A 32-bit store to CCS18 (the 24-bit direction register) spills
            // its fourth byte here. The chip side is inferred: the byte is
            // accepted and dropped, as benign. What it reads back is unknown;
            // it reads 0.
            _ => self.note(format_args!("write of {v:#04x} to unassigned CCS {off:#04x}")),
        }
    }

    // ---- multi-channel block registers (BAR1) ----

    fn mt_read(&self, s: &State, off: u32) -> u8 {
        if let Some((idx, field, byte)) = engine_decode(off) {
            let e = &s.eng[idx];
            let word = match field {
                Field::Addr => (e.addr & !3).wrapping_add(4 * e.pos),
                Field::Count => e.count - e.pos,
                Field::Tc => {
                    // Approximation: terminal counts are write only.
                    self.note(format_args!("read of write-only terminal count {off:#04x}"));
                    0
                }
            };
            return (word >> (8 * byte)) as u8;
        }
        let value = s.mt(off);
        match off {
            MT_STATUS | MT_RATE | MT_FORMAT | MT_MASK | MT_START | MT_CHANNELS | MT_FIFO_ERROR | MT_PAUSE | MT_PEAK_INDEX
            | MT_ROUTING..=0x2f | MT_SPDIF_CTRL | 0x3d => value,
            MT_AC97_INDEX | MT_AC97_DATA | 0x07 => {
                if self.variant != Variant::Ac97 || s.ccs(CCS_AC_CONFIG) & CCS_AC_I2S != 0 {
                    self.note(format_args!("read of AC'97 register {off:#04x} with no AC-link codec"));
                }
                value
            }
            MT_AC97_CMD => (value & !(MT_AC97_READY | 0x34)) | if s.ac97_ready(self.variant) { MT_AC97_READY } else { 0 },
            MT_PEAK_DATA => {
                // Approximation: the peak meters are not modelled.
                self.note(format_args!("peak meters are not modelled"));
                0
            }
            _ => {
                self.note(format_args!("read of unassigned MT {off:#04x}"));
                0
            }
        }
    }

    fn engine_write(&self, s: &mut State, idx: usize, field: Field, byte: u32, v: u8) {
        let mask = if field == Field::Addr { u32::MAX } else { State::count_mask(idx) };
        let e = &mut s.eng[idx];
        // Approximation: the effect of a write to a running engine is not
        // stated, so a write reloads the position or interval.
        let reg = e.field(field);
        *reg = (*reg & !(0xff << (8 * byte)) | u32::from(v) << (8 * byte)) & mask;
        if field == Field::Addr {
            *reg &= !3;
        }
        if field == Field::Tc {
            e.tc_pos = 0;
        } else {
            e.pos = 0;
        }
    }

    fn mt_write(&self, s: &mut State, off: u32, v: u8, now: i64) {
        if let Some((idx, field, byte)) = engine_decode(off) {
            self.engine_write(s, idx, field, byte, v);
            self.arm_timer(s, now);
            return;
        }
        let o = off as usize;
        match off {
            MT_STATUS => s.mt[o] &= !v,
            MT_RATE => {
                s.mt[o] = v & 0x1f;
                self.apply_rate(s);
                self.clock_changed(s, now, true);
            }
            MT_FORMAT => {
                s.mt[o] = v & 0x0f;
                if v & 3 != 0 {
                    self.note(format_args!("reserved data format {}", v & 3));
                }
                if v & 4 != 0 {
                    self.note(format_args!("MT02 bit 2 has no known meaning"));
                }
            }
            MT_MASK => {
                s.mt[o] = v;
                self.arm_timer(s, now);
            }
            MT_AC97_INDEX => s.mt[o] = v & 0x7f,
            MT_AC97_DATA | 0x07 => s.mt[o] = v,
            MT_AC97_CMD => {
                s.mt[o] = v & 0xc3;
                // Bit 7 drives PRST# to the converters in I2S mode too; no
                // converter reacts to it in this model.
                if v & (MT_AC97_WRITE | MT_AC97_READ) != 0 {
                    // Approximation: a cycle completes at once.
                    self.ac97_cycle(s, v);
                }
            }
            MT_START => {
                if v & !ENGINE_BITS != 0 {
                    self.note(format_args!("MT18 bit 3 is reserved"));
                }
                s.mt[o] = v & ENGINE_BITS;
                for i in 0..ENGINES {
                    if v & (1 << i) != 0 && i != FIFO_ERROR_SOURCE && !s.engine_available(i) {
                        self.note(format_args!("engine {i} started while MT19 gives its slots to the interleaved stream"));
                    }
                }
                self.clock_changed(s, now, false);
            }
            MT_CHANNELS => {
                s.mt[o] = v & 3;
                self.clock_changed(s, now, true);
            }
            MT_FIFO_ERROR => s.mt[o] &= !(v & ENGINE_BITS),
            MT_PAUSE => {
                s.mt[o] = v & ENGINE_BITS;
                self.clock_changed(s, now, false);
            }
            MT_ROUTING..=0x2f => {
                s.mt[o] = v;
                if v != 0 {
                    self.note(format_args!("output routing is not modelled"));
                }
            }
            MT_SPDIF_CTRL | 0x3d => {
                // Every field but the validity flag (bit 15) must be written
                // only while the transmitter is disabled. Approximation: the
                // model enforces that rule (see the module notes).
                let mut v = v;
                if s.ccs(CCS_SPDIF_CONFIG) & 0x80 != 0 {
                    if off == MT_SPDIF_CTRL {
                        self.note(format_args!("MT3C fields written while the S/PDIF transmitter is enabled"));
                        return;
                    }
                    if (v ^ s.mt[o]) & 0x7f != 0 {
                        self.note(format_args!("MT3C fields written while the S/PDIF transmitter is enabled"));
                    }
                    v = (s.mt[o] & 0x7f) | (v & 0x80);
                }
                s.mt[o] = v;
            }
            MT_PEAK_INDEX => s.mt[o] = v & 0x1f,
            MT_PEAK_DATA => self.note(format_args!("write to read-only peak data")),
            // A 32-bit store to MT14 (the 3-byte count) spills its fourth
            // byte into 0x17. The chip side is inferred: accepted and
            // dropped, as benign. The read value is unknown; it reads 0.
            _ => self.note(format_args!("write of {v:#04x} to unassigned MT {off:#04x}")),
        }
    }

    // ---- bus accesses ----

    fn read(&self, bar: usize, off: u32, width: usize) -> u32 {
        let size = if bar == 0 { CCS_SIZE } else { MT_SIZE };
        let now = self.now();
        let mut s = self.lock_idle();
        if s.pm_state != 0 {
            drop(s);
            self.note(format_args!("read of BAR {bar} outside D0"));
            return u32::MAX >> (32 - 8 * width);
        }
        self.sync(&mut s, now);
        let mut v = 0u32;
        for i in 0..width as u32 {
            let o = off + i;
            let b = if o >= size {
                self.note(format_args!("read past the end of BAR {bar} at {o:#04x}"));
                0
            } else if bar == 0 {
                self.ccs_read(&s, o)
            } else {
                self.mt_read(&s, o)
            };
            v |= u32::from(b) << (8 * i);
        }
        // A read of an engine's position: the guest may be steering by it
        // (see [`Engine::steer_limit`]).
        if let Some((idx, Field::Addr | Field::Count, _)) = (bar == 1).then(|| engine_decode(off)).flatten() {
            let e = &mut s.eng[idx];
            e.polled = Some((now, e.frames));
        }
        v
    }

    fn write(&self, bar: usize, off: u32, val: u32, width: usize) {
        let size = if bar == 0 { CCS_SIZE } else { MT_SIZE };
        let now = self.now();
        let mut s = self.lock_idle();
        if s.pm_state != 0 {
            drop(s);
            self.note(format_args!("write of {val:#x} to BAR {bar} outside D0"));
            return;
        }
        self.sync(&mut s, now);
        for i in 0..width as u32 {
            let o = off + i;
            let b = (val >> (8 * i)) as u8;
            if o >= size {
                self.note(format_args!("write past the end of BAR {bar} at {o:#04x}"));
            } else if bar == 0 {
                self.ccs_write(&mut s, o, b, now);
            } else {
                self.mt_write(&mut s, o, b, now);
            }
        }
        self.update_irq(&mut s);
    }

    // ---- configuration space ----

    /// Dword `index` as a guest reads it. Registers the chip leaves
    /// undescribed (BARs 2 and 3, the subsystem mask at its other offset,
    /// 0x43, the hardware configuration control and everything between
    /// 0x44 and 0x7f) read 0.
    fn config_read(&self, index: usize) -> u32 {
        if Self::config_unknown(index) {
            self.note(format_args!("PCI config read of unknown register at {:#04x}", index * 4));
            return 0;
        }
        match index {
            // The interrupt status bit is the chip's own.
            1 => self.pci.config().read(1) | if self.state.lock().irq_level { STATUS_INTX } else { 0 },
            // Only bit 7 of the subsystem mask byte exists.
            SVID_MASK_DWORD => self.pci.config().read(index) & SVID_WRITE_ENABLE,
            // The rest of the space the chip does not describe.
            _ if index >= 16 && !matches!(index, PM_CAP_DWORD | PMCSR_DWORD) => 0,
            _ => self.pci.config().read(index),
        }
    }

    /// Registers the chip leaves undescribed that the header would
    /// otherwise answer.
    fn config_unknown(index: usize) -> bool {
        matches!(index, 6 | 7 | 24)
    }

    /// Whether the subsystem pair is writable: PCI42 bit 7.
    fn subsystem_writable(&self) -> bool {
        self.pci.config().read(SVID_MASK_DWORD) & SVID_WRITE_ENABLE != 0
    }

    /// A whole-dword write the header would take: note what the chip
    /// leaves undescribed. A narrower write is not seen here.
    fn config_write_noted(&self, index: usize, val: u32) -> bool {
        if Self::config_unknown(index) || index == SVID_MASK_DWORD && val & !SVID_WRITE_ENABLE != 0 {
            self.note(format_args!("PCI config write of {val:#x} to unknown register at {:#04x}", index * 4));
        }
        if index == SUBSYSTEM_DWORD && !self.subsystem_writable() {
            self.note(format_args!("PCI config write of {val:#x} to the subsystem pair while it is read-only"));
            return true;
        }
        false
    }

    /// The header changed dword `index` from `old` to `new`.
    fn config_changed(&self, index: usize, old: u32, new: u32) {
        match index {
            // The subsystem pair is writable only while PCI42 bit 7 is set.
            SUBSYSTEM_DWORD if !self.subsystem_writable() => {
                self.note(format_args!("PCI config write of {new:#x} to the subsystem pair while it is read-only"));
                self.pci.config().set_subsystem_identity(old);
            }
            PMCSR_DWORD => self.power_state_written(old, new),
            _ => {}
        }
    }

    /// PMCSR bits 1:0 changed. D1 is not advertised and the write is
    /// refused. Established (vendor documentation): D3hot to D0
    /// resets the chip including configuration space, so BARs and command
    /// return to defaults. Approximation: this resets the registers and
    /// engines but not the header. A function cannot reset its own header
    /// without leaving the bus's decode table stale, because `PciBus` rebuilds
    /// it after a configuration write only when the command register or a BAR
    /// changed. Honouring it needs the bus to rebuild after any
    /// `config_written` (or a way for the function to ask), neither of which
    /// exists; none was invented. Not stated: what the subsystem IDs, strapped
    /// fields, EEPROM re-read and PMCSR read-back do. IRIX never writes PMCSR.
    fn power_state_written(&self, old: u32, new: u32) {
        let (old, new) = ((old & PMCSR_STATE) as u8, (new & PMCSR_STATE) as u8);
        if new == 1 {
            self.pci.config().write(PMCSR_DWORD, u32::from(old));
            return;
        }
        let now = self.now();
        let mut s = self.lock_idle();
        s.pm_state = new;
        if old == 3 && new == 0 {
            self.note(format_args!("D3hot to D0 resets the registers, not the configuration space"));
            self.reset_state(&mut s, now);
            s.pm_state = 0;
        } else if old == 0 && new != 0 {
            self.note(format_args!("D2 and D3hot power-down of the blocks in CCS1C is not modelled beyond the clock gate"));
        }
    }

    // ---- observation ----

    /// The engines brought up to date, then what they and the converters
    /// have done.
    pub fn stats(&self) -> Stats {
        let now = self.now();
        let mut s = self.lock_idle();
        self.sync(&mut s, now);
        s.stats
    }

    /// What the front DAC (AK4381) was last told, register by register.
    pub fn ak4381(&self) -> [u8; AK4381_REGS] {
        self.state.lock().ak4381
    }

    /// What the surround DAC (AK4355) was last told.
    pub fn ak4355(&self) -> [u8; AK4355_REGS] {
        self.state.lock().ak4355
    }
}

impl Device for Ice1724 {
    fn step(&self, _cycles: u64) {}
    fn start(&self) {
        self.inner.resume()
    }
    fn stop(&self) {
        self.inner.pause()
    }
    fn is_running(&self) -> bool {
        self.inner.armed.load(Ordering::SeqCst)
    }
    fn get_clock(&self) -> u64 {
        0
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("ice1724".into(), "ICE1724 audio controller: ice1724 status".into())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write as _;
        if cmd != "ice1724" || args != ["status"] {
            return Err("Usage: ice1724 status".into());
        }
        let st = self.stats();
        let _ = writeln!(
            w,
            "frames: analog {} (last dword {:#010x}), S/PDIF {}, capture {}, dropped by the host {}",
            st.analog_frames, st.analog_last, st.spdif_frames, st.capture_frames, st.dropped_frames
        );
        let _ = writeln!(
            w,
            "converter writes: front {}, surround {}, refused {}; accesses nothing establishes: {}",
            st.ak4381_writes, st.ak4355_writes, st.codec_rejects, self.unexpected()
        );
        Ok(())
    }
}

impl PciFunction for Ice1724 {
    fn state(&self) -> &PciFunctionState {
        &self.inner.pci
    }

    fn config_dword(&self, index: usize) -> u32 {
        self.inner.config_read(index)
    }

    fn config_write(&self, index: usize, val: u32) -> bool {
        self.inner.config_write_noted(index, val)
    }

    fn config_written(&self, index: usize, old: u32, new: u32) {
        self.inner.config_changed(index, old, new)
    }

    /// `off` is the byte offset the register file sees; a wider access is
    /// byte steps in address order.
    fn bar_read(&self, bar: usize, off: u32, width: usize) -> u32 {
        if bar > 1 {
            self.inner.note(format_args!("read of unimplemented BAR {bar}"));
            return 0;
        }
        self.inner.read(bar, off, width)
    }

    fn bar_write(&self, bar: usize, off: u32, val: u32, width: usize) {
        if bar > 1 {
            self.inner.note(format_args!("write to unimplemented BAR {bar}"));
            return;
        }
        self.inner.write(bar, off, val, width)
    }

    /// Drive the line to what the chip is holding now.
    fn attached(&self) {
        let i = &self.inner;
        let mut s = i.lock_idle();
        if let Some(l) = i.pci.irq() {
            l.set(s.irq_level);
        }
        i.update_irq(&mut s);
    }

    /// RST# has reset the header: the chip returns to power-on state.
    fn reset(&self) {
        self.inner.reset();
    }
}

impl State {
    /// A loaded state must not make the card divide by zero or index past
    /// a buffer; refuse it, naming the field.
    fn check(&self) -> Result<(), String> {
        if !RATES.contains(&self.rate) || self.rate == 0 {
            return Err(format!("rate: {} Hz is not one the card runs at", self.rate));
        }
        for (i, e) in self.eng.iter().enumerate() {
            if e.count > Self::count_mask(i) || e.tc > Self::count_mask(i) {
                return Err(format!("engine {i}: count or interval past the register width"));
            }
            if e.pos > e.count || e.tc_pos > e.tc {
                return Err(format!("engine {i}: position past the end of its buffer or interval"));
            }
        }
        if self.pm_state > 3 || self.pm_state == 1 {
            return Err(format!("pm_state: {} is not a state the card enters", self.pm_state));
        }
        Ok(())
    }
}

impl Saveable for Ice1724 {
    /// The registers and engines, with each running engine's age on the
    /// card's clock. The header is the bridge's to save; the host stream is
    /// this process's own and is not saved. The timer is armed again when the
    /// card starts.
    fn save_state(&self) -> toml::Value {
        let i = &self.inner;
        let now = i.now();
        let s = i.lock_idle();
        let engines = s.eng.iter().map(|e| {
            StateTable::new()
                .u32("addr", e.addr)
                .u32("count", e.count)
                .u32("tc", e.tc)
                .u32("pos", e.pos)
                .u32("tc_pos", e.tc_pos)
                .u64("frames", e.frames)
                .u64("age_ns", if e.active { (now - e.anchor).max(0) as u64 } else { 0 })
                .bool("active", e.active)
                .build()
        });
        StateTable::new()
            .bool("ac97", i.variant == Variant::Ac97)
            .u8s("ccs", &s.ccs)
            .u8s("mt", &s.mt)
            .u8("pm_state", s.pm_state)
            .u8s("eeprom", &s.eeprom)
            .value("eng", toml::Value::Array(engines.collect()))
            .u32("rate", s.rate)
            .u8s("frame_bits", &s.frame_bits)
            .u16s("frame_word", &s.frame_word)
            .u8s("ak4381", &s.ak4381)
            .u8s("ak4355", &s.ak4355)
            .u16s("ac97_regs", &s.ac97)
            .u64("analog_frames", s.stats.analog_frames)
            .u32("analog_last", s.stats.analog_last)
            .u64("spdif_frames", s.stats.spdif_frames)
            .u64("capture_frames", s.stats.capture_frames)
            .u64("dropped_frames", s.stats.dropped_frames)
            .u32("ak4381_writes", s.stats.ak4381_writes)
            .u32("ak4355_writes", s.stats.ak4355_writes)
            .u32("codec_rejects", s.stats.codec_rejects)
            .build()
    }

    /// Called with the card stopped.
    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let i = &self.inner;
        if field_bool(v, "ac97")? != (i.variant == Variant::Ac97) {
            return Err("ice1724: saved for the other variant".into());
        }
        let now = i.now();
        let saved = field(v, "eng")?.as_array().ok_or("eng: not a list")?;
        if saved.len() != ENGINES {
            return Err(format!("eng: {} engines, expected {ENGINES}", saved.len()));
        }
        let mut eng = [Engine::default(); ENGINES];
        for (e, t) in eng.iter_mut().zip(saved) {
            // The clock anchor is the card's clock now, less the engine's
            // age when it was saved.
            *e = Engine {
                addr: field_u32(t, "addr")?,
                count: field_u32(t, "count")?,
                tc: field_u32(t, "tc")?,
                pos: field_u32(t, "pos")?,
                tc_pos: field_u32(t, "tc_pos")?,
                frames: field_u64(t, "frames")?,
                anchor: now - field_u64(t, "age_ns")? as i64,
                active: field_bool(t, "active")?,
                polled: None,
            };
        }
        let mut ac97 = [0u16; AC97_REGS];
        load_u16_slice(field(v, "ac97_regs")?, &mut ac97);
        let mut frame_word = [0u16; 2];
        load_u16_slice(field(v, "frame_word")?, &mut frame_word);
        let mut s = i.lock_idle();
        let loaded = State {
            ccs: field_u8s(v, "ccs")?,
            mt: field_u8s(v, "mt")?,
            pm_state: field_u8(v, "pm_state")?,
            irq_level: s.irq_level,
            eeprom: field_u8s(v, "eeprom")?,
            eng,
            rate: field_u32(v, "rate")?,
            gpio_pins: 0,
            frame_bits: field_u8s(v, "frame_bits")?,
            frame_word,
            ak4381: field_u8s(v, "ak4381")?,
            ak4355: field_u8s(v, "ak4355")?,
            ac97,
            stats: Stats {
                analog_frames: field_u64(v, "analog_frames")?,
                analog_last: field_u32(v, "analog_last")?,
                spdif_frames: field_u64(v, "spdif_frames")?,
                capture_frames: field_u64(v, "capture_frames")?,
                dropped_frames: field_u64(v, "dropped_frames")?,
                ak4381_writes: field_u32(v, "ak4381_writes")?,
                ak4355_writes: field_u32(v, "ak4355_writes")?,
                codec_rejects: field_u32(v, "codec_rejects")?,
            },
            timer: None,
            timer_gen: s.timer_gen + 1,
            timer_due: 0,
            scratch: std::mem::take(&mut s.scratch),
            moving: false,
        };
        loaded.check()?;
        if s.timer.is_some() {
            return Err("ice1724 loaded while its timer runs".into());
        }
        *s = loaded;
        s.gpio_pins = s.pins();
        i.update_irq(&mut s);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
