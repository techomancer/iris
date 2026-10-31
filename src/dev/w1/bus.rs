//! The bus: which parts are selected, the ROM command phases (Read, Match,
//! Search and Skip ROM), then a memory function command, its two address
//! bytes and the selected parts' reply streams, wired-AND.
//!
//! Ported from hw/1wire/core.c in jamesbraid/qemu-sgi-origami.

use crate::snapshot::{field, field_bytes, field_u32, field_u64s, field_u8, toml_u8, StateTable};

/// A part on the bus.
pub trait Part: Send {
    /// The 64-bit registration number: family code, 48-bit serial (least
    /// significant byte first), CRC-8 of the first seven bytes.
    fn rom(&self) -> [u8; 8];

    /// What the part sends after memory function command `cmd` and target
    /// address `addr`, every byte to the end of the stream, CRCs included.
    /// `None` for a command it refuses (writes, anything unknown): the bus
    /// then goes quiet until the next reset.
    fn reply(&self, cmd: u8, addr: u16) -> Option<Vec<u8>>;

    /// Bit `i` (0..64) of the ROM, least significant bit of byte 0 first.
    fn rom_bit(&self, i: usize) -> bool {
        self.rom()[i / 8] >> (i % 8) & 1 != 0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Phase {
    /// Collecting a ROM command.
    Rom,
    ReadRom(usize),
    Match(usize),
    Search { bit: usize, step: u8 },
    /// Collecting a memory function command.
    Func,
    /// Collecting its two address bytes.
    Addr,
    Stream(usize),
    /// Nobody is listening until the next reset.
    Dead,
}

/// One wire and the parts on it.
pub struct Bus {
    parts: Vec<Box<dyn Part>>,
    sel: Vec<bool>,
    replies: Vec<Vec<u8>>,
    phase: Phase,
    acc: u32,
    nbits: u32,
    cmd: u8,
}

impl Default for Bus {
    fn default() -> Bus {
        Bus::new()
    }
}

impl Bus {
    /// A bus with nothing on it: a reset gets no presence.
    pub fn new() -> Bus {
        Bus { parts: vec![], sel: vec![], replies: vec![], phase: Phase::Dead, acc: 0, nbits: 0, cmd: 0 }
    }

    /// Put a part on the wire.
    pub fn add(&mut self, part: Box<dyn Part>) {
        self.parts.push(part);
        self.sel.push(false);
    }

    /// The parts' ROMs, in the order they were added.
    pub fn roms(&self) -> Vec<[u8; 8]> {
        self.parts.iter().map(|p| p.rom()).collect()
    }

    /// A reset pulse: every part selected, the ROM command phase begins.
    /// True if any part answered with presence.
    pub fn reset(&mut self) -> bool {
        self.phase = Phase::Rom;
        self.acc = 0;
        self.nbits = 0;
        self.replies.clear();
        self.sel.fill(true);
        !self.parts.is_empty()
    }

    /// Collect `want` bits, least significant first.
    fn take_bits(&mut self, bit: bool, want: u32) -> Option<u32> {
        self.acc |= (bit as u32) << self.nbits;
        self.nbits += 1;
        if self.nbits == want {
            let v = self.acc;
            self.acc = 0;
            self.nbits = 0;
            Some(v)
        } else {
            None
        }
    }

    /// The wired-AND of `f` over the selected parts.
    fn and_over(&self, f: impl Fn(&dyn Part) -> bool) -> bool {
        self.parts.iter().zip(&self.sel).filter(|(_, s)| **s).all(|(p, _)| f(p.as_ref()))
    }

    /// Deselect the parts whose ROM bit `i` is not `bit`.
    fn deselect_mismatch(&mut self, i: usize, bit: bool) {
        for (p, s) in self.parts.iter().zip(self.sel.iter_mut()) {
            if *s && p.rom_bit(i) != bit {
                *s = false;
            }
        }
    }

    /// Whether ROM bit `i` was the last: on to the function command if
    /// anything is still selected.
    fn after_selection(&self) -> Phase {
        if self.sel.iter().any(|s| *s) { Phase::Func } else { Phase::Dead }
    }

    /// One bit slot: the master writes `bit` (a one is also a read slot).
    /// Returns the level on the wire, the AND of the master's bit and
    /// every selected part's.
    pub fn slot(&mut self, bit: bool) -> bool {
        match self.phase.clone() {
            Phase::Dead => true,
            Phase::Rom => {
                if let Some(c) = self.take_bits(bit, 8) {
                    self.phase = match c as u8 {
                        0x33 => Phase::ReadRom(0),
                        0x55 => Phase::Match(0),
                        0xcc => Phase::Func,
                        0xf0 => Phase::Search { bit: 0, step: 0 },
                        // Overdrive (0x3c, 0x69) is not modelled; nor is anything else.
                        _ => Phase::Dead,
                    };
                }
                bit
            }
            Phase::ReadRom(i) => {
                let level = self.and_over(|p| p.rom_bit(i));
                self.phase = if i + 1 == 64 { Phase::Func } else { Phase::ReadRom(i + 1) };
                level
            }
            Phase::Match(i) => {
                // A part leaves at the first bit that differs.
                self.deselect_mismatch(i, bit);
                self.phase = if i + 1 == 64 { self.after_selection() } else { Phase::Match(i + 1) };
                bit
            }
            Phase::Search { bit: i, step } => match step {
                0 => {
                    let l = self.and_over(|p| p.rom_bit(i));
                    self.phase = Phase::Search { bit: i, step: 1 };
                    l
                }
                1 => {
                    let l = self.and_over(|p| !p.rom_bit(i));
                    self.phase = Phase::Search { bit: i, step: 2 };
                    l
                }
                _ => {
                    self.deselect_mismatch(i, bit);
                    self.phase =
                        if i + 1 == 64 { self.after_selection() } else { Phase::Search { bit: i + 1, step: 0 } };
                    bit
                }
            },
            Phase::Func => {
                if let Some(c) = self.take_bits(bit, 8) {
                    self.cmd = c as u8;
                    self.phase = Phase::Addr;
                }
                bit
            }
            Phase::Addr => {
                if let Some(a) = self.take_bits(bit, 16) {
                    let (cmd, ta) = (self.cmd, a as u16);
                    let mut replies = Vec::new();
                    for (p, s) in self.parts.iter().zip(&self.sel) {
                        if *s {
                            match p.reply(cmd, ta) {
                                Some(r) => replies.push(r),
                                // Write commands and anything unknown.
                                None => {
                                    self.phase = Phase::Dead;
                                    return bit;
                                }
                            }
                        }
                    }
                    self.replies = replies;
                    self.phase = Phase::Stream(0);
                }
                bit
            }
            Phase::Stream(i) => {
                let byte = i / 8;
                let sh = i % 8;
                // Past its end a part lets go and the wire floats high.
                let level = self.replies.iter().all(|r| r.get(byte).is_none_or(|b| b >> sh & 1 != 0));
                self.phase = Phase::Stream(i + 1);
                level
            }
        }
    }
}

// A 1-Wire bus's transaction state for a snapshot: the phase, the bits
// collected so far and which parts are still selected. The parts are the
// board's ROMs and stay as built; their ROM numbers are saved only to
// refuse a load onto a bus with other parts.
impl Phase {
    fn to_words(&self) -> [u64; 3] {
        match *self {
            Phase::Rom => [0, 0, 0],
            Phase::ReadRom(i) => [1, i as u64, 0],
            Phase::Match(i) => [2, i as u64, 0],
            Phase::Search { bit, step } => [3, bit as u64, u64::from(step)],
            Phase::Func => [4, 0, 0],
            Phase::Addr => [5, 0, 0],
            Phase::Stream(i) => [6, i as u64, 0],
            Phase::Dead => [7, 0, 0],
        }
    }
    fn from_words([tag, a, b]: [u64; 3]) -> Result<Self, String> {
        // The ROM phases index a part's 64-bit ROM by bit.
        if matches!(tag, 1..=3) && a >= 64 {
            return Err(format!("1-Wire phase {tag}: ROM bit {a}, past the 64"));
        }
        Ok(match tag {
            0 => Phase::Rom,
            1 => Phase::ReadRom(a as usize),
            2 => Phase::Match(a as usize),
            3 => Phase::Search { bit: a as usize, step: b as u8 },
            4 => Phase::Func,
            5 => Phase::Addr,
            6 => Phase::Stream(a as usize),
            7 => Phase::Dead,
            _ => return Err(format!("1-Wire phase {tag} is not a bus phase")),
        })
    }
}

impl Bus {
    pub fn save_state(&self) -> toml::Value {
        let roms: Vec<u8> = self.roms().concat();
        let sel: Vec<u8> = self.sel.iter().map(|&s| u8::from(s)).collect();
        let replies = self.replies.iter().map(|r| crate::snapshot::u8_slice_to_toml(r)).collect();
        StateTable::new()
            .u8s("roms", &roms)
            .u8s("selected", &sel)
            .value("replies", toml::Value::Array(replies))
            .u64s("phase", &self.phase.to_words())
            .u32("acc", self.acc)
            .u32("nbits", self.nbits)
            .u8("cmd", self.cmd)
            .build()
    }

    pub fn load_state(&mut self, v: &toml::Value) -> Result<(), String> {
        if field_bytes(v, "roms")? != self.roms().concat() {
            return Err("the saved 1-Wire bus has other parts".into());
        }
        let sel = field_bytes(v, "selected")?;
        if sel.len() != self.parts.len() {
            return Err("selected: not one entry per part".into());
        }
        let replies = field(v, "replies")?.as_array().ok_or("replies: not a list")?;
        let replies = replies.iter()
            .map(|r| r.as_array()?.iter().map(toml_u8).collect::<Option<Vec<u8>>>())
            .collect::<Option<Vec<_>>>().ok_or("replies: an entry is not a byte list")?;
        self.phase = Phase::from_words(field_u64s(v, "phase")?)?;
        self.sel = sel.into_iter().map(|s| s != 0).collect();
        self.replies = replies;
        self.acc = field_u32(v, "acc")?;
        self.nbits = field_u32(v, "nbits")?;
        self.cmd = field_u8(v, "cmd")?;
        Ok(())
    }
}
