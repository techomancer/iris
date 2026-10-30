//! The AMD flash command set: unlock, word program, sector erase and CFI
//! query state, over storage supplied by the caller.
//!
//! Array bytes and bus translation belong to the enclosing device. Fuel uses
//! this 8 MiB command engine for each boot-flash array. The CFI data and timing
//! are compatibility values; they do not identify the fitted flash chip.
//! Buffered writes, protection, bypass and erase suspend are absent.
//!
//! Ported from MAME's intelfsh.cpp AMD command path, revision
//! 2ee9b1e30299b122b131cab0bd7912abc2b095ba (BSD-3-Clause).

// Copyright (c) Aaron Giles

pub const SECTOR_SIZE: usize = 64 * 1024;
const SECTORS: usize = 128;
pub const PART_SIZE: usize = SECTOR_SIZE * SECTORS;
pub const ERASE_SETUP_NS: u64 = 50_000;
pub const ERASE_NS: u64 = 512_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Array,
    Unlock1,
    Unlock2,
    Program,
    Erase1,
    Erase2,
    Erase3,
    Query,
    Identify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    None,
    Program { offset: usize, value: u16 },
}

pub struct AmdCore {
    mode: Mode,
    selected: [bool; SECTORS],
    /// Remaining time includes the sector-selection interval.
    remaining_ns: u64,
    setup_ns: u64,
    toggle: bool,
    failed: bool,
}

impl Default for AmdCore {
    fn default() -> Self {
        Self {
            mode: Mode::Array,
            selected: [false; SECTORS],
            remaining_ns: 0,
            setup_ns: 0,
            toggle: false,
            failed: false,
        }
    }
}

impl AmdCore {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    pub fn busy(&self) -> bool {
        self.remaining_ns != 0 || self.failed
    }
    pub fn selecting(&self) -> bool {
        self.setup_ns != 0
    }
    pub fn array_reads(&self) -> bool {
        !self.busy() && self.mode != Mode::Query && self.mode != Mode::Identify
    }
    /// None selects ordinary backing bytes. Polling never advances time.
    pub fn read16(&mut self, offset: usize) -> Option<u16> {
        if self.busy() {
            let selected = self.selected[offset / SECTOR_SIZE];
            self.toggle = !self.toggle;
            return Some(
                (if self.toggle {
                    0x40 | if selected { 4 } else { 0 }
                } else {
                    0
                }) | if self.setup_ns == 0 { 8 } else { 0 }
                    | if self.failed { 0x20 } else { 0 },
            );
        }
        match self.mode {
            Mode::Query => Some(query(offset / 2) as u16),
            // IDs remain unspecified until a part/profile is selected.
            Mode::Identify => Some(0),
            _ => None,
        }
    }
    pub fn write16(&mut self, offset: usize, value: u16) -> Result<Effect, ()> {
        if offset & 1 != 0 || offset >= PART_SIZE {
            return Err(());
        }
        let cmd = value as u8;
        let word = offset / 2;
        if self.busy() {
            if (self.failed || self.setup_ns != 0) && cmd == 0xf0 {
                self.reset();
            } else if self.setup_ns != 0 && cmd == 0x30 {
                self.selected[offset / SECTOR_SIZE] = true;
                self.setup_ns = ERASE_SETUP_NS;
                self.remaining_ns = ERASE_SETUP_NS + ERASE_NS;
            } else if self.setup_ns != 0 {
                self.reset();
                crate::dlog!(
                    crate::devlog::LogModule::Bedrock,
                    "amd-flash: sector selection cancelled by command {cmd:#x} at word offset {word:#x}"
                );
                return Err(());
            } else if cmd != 0xf0 {
                crate::dlog!(
                    crate::devlog::LogModule::Bedrock,
                    "amd-flash: unsupported command {cmd:#x} while erasing at word offset {word:#x}"
                );
                return Err(());
            }
            return Ok(Effect::None);
        }
        // The pending program data cycle consumes every bit, including F0.
        if self.mode == Mode::Program {
            self.mode = Mode::Array;
            return Ok(Effect::Program { offset, value });
        }
        if cmd == 0xf0 {
            self.reset();
            return Ok(Effect::None);
        }
        let next = match (self.mode, word, cmd) {
            (Mode::Array, 0x55, 0x98) => Mode::Query,
            (Mode::Array, 0x555, 0xaa) => Mode::Unlock1,
            (Mode::Unlock1, 0x2aa, 0x55) => Mode::Unlock2,
            (Mode::Unlock2, 0x555, 0xa0) => Mode::Program,
            (Mode::Unlock2, 0x555, 0x80) => Mode::Erase1,
            (Mode::Unlock2, 0x555, 0x90) => Mode::Identify,
            (Mode::Erase1, 0x555, 0xaa) => Mode::Erase2,
            (Mode::Erase2, 0x2aa, 0x55) => Mode::Erase3,
            (Mode::Erase3, _, 0x30) => {
                self.selected[offset / SECTOR_SIZE] = true;
                self.setup_ns = ERASE_SETUP_NS;
                self.remaining_ns = ERASE_SETUP_NS + ERASE_NS;
                Mode::Array
            }
            _ => {
                crate::dlog!(
                    crate::devlog::LogModule::Bedrock,
                    "amd-flash: unsupported cycle {value:#06x} at byte offset {offset:#x} in {:?}",
                    self.mode
                );
                self.mode = Mode::Array;
                return Err(());
            }
        };
        self.mode = next;
        Ok(Effect::None)
    }
    /// The owner persists and applies selected sectors before calling finish.
    pub fn advance_ns(&mut self, ns: u64) -> bool {
        if self.remaining_ns == 0 || self.failed {
            return false;
        }
        self.setup_ns = self.setup_ns.saturating_sub(ns);
        self.remaining_ns = self.remaining_ns.saturating_sub(ns);
        self.remaining_ns == 0
    }
    pub fn selected_sectors(&self) -> impl Iterator<Item = usize> + '_ {
        self.selected
            .iter()
            .enumerate()
            .filter_map(|(i, selected)| selected.then_some(i))
    }
    pub fn finish_erase(&mut self, success: bool) {
        if success {
            self.reset();
        } else {
            self.failed = true;
        }
    }
}

// Generic model response; voltage/timing do not identify Fuel's fitted part.
// Field positions/units: AMD publication 21444 rev E+2, pp14-15.
// Region encoding also follows Renode AMDCFIFlash.cs (MIT), revision
// 5448105bc7875aa6c5305761d97a371dc640dd7f. No Renode mutation code is used.
fn query(index: usize) -> u8 {
    const PROFILE: [u8; 33] = [
        b'Q', b'R', b'Y', 2, 0, 0, 0, 0, 0, 0, 0, 0x27, 0x36, 0, 0, 3, 0, 9, 0, 4, 0, 4, 0, 23, 1,
        0, 0, 0, 1, 0x7f, 0, 0, 1,
    ];
    index
        .checked_sub(0x10)
        .and_then(|i| PROFILE.get(i))
        .copied()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    // Cases ported from upstream QEMU's tests/qtest/pflash-cfi02-test.c.
    use super::*;

    // Word addresses from the command sequences, as byte offsets on the 16-bit bus.
    const UNLOCK0: usize = 0x555 * 2;
    const UNLOCK1: usize = 0x2aa * 2;
    const CFI: usize = 0x55 * 2;

    fn cmd(f: &mut AmdCore, offset: usize, value: u16) -> Effect {
        f.write16(offset, value).unwrap()
    }

    fn unlock(f: &mut AmdCore) {
        cmd(f, UNLOCK0, 0xaa);
        cmd(f, UNLOCK1, 0x55);
    }

    fn start_sector_erase(f: &mut AmdCore, offset: usize) {
        unlock(f);
        cmd(f, UNLOCK0, 0x80);
        unlock(f);
        cmd(f, offset, 0x30);
    }

    /// The guest's erase loop polls DQ3, DQ6 and DQ7; get the phases wrong and
    /// firmware either times out or reads the sector before it is blank.
    #[test]
    fn a_sector_erase_reports_toggle_status_until_it_completes() {
        let mut f = AmdCore::default();
        let sector = 5 * SECTOR_SIZE;
        start_sector_erase(&mut f, sector);
        assert!(f.busy() && f.selecting());
        assert!(!f.array_reads());
        // DQ3 clear: still inside the window where more sectors may be added.
        let first = f.read16(sector).unwrap();
        assert_eq!(first & 0x08, 0);
        assert_eq!(first & 0x80, 0);
        f.advance_ns(ERASE_SETUP_NS);
        assert!(!f.selecting());
        let a = f.read16(sector).unwrap();
        let b = f.read16(sector).unwrap();
        assert_eq!(a & 0x08, 0x08);
        assert_eq!(a & 0x80, 0);
        assert_eq!(a & 0x40, !b & 0x40, "DQ6 toggles on successive reads");
        assert!(f.advance_ns(ERASE_NS));
        assert_eq!(f.selected_sectors().collect::<Vec<_>>(), [5]);
        f.finish_erase(true);
        assert!(f.array_reads());
        assert_eq!(f.read16(sector), None);
    }

    /// Firmware erases several sectors with one command; each further 0x30
    /// inside the selection window must join the set, not start a second erase.
    #[test]
    fn extra_sectors_join_a_pending_erase_inside_the_selection_window() {
        let mut f = AmdCore::default();
        start_sector_erase(&mut f, 0);
        cmd(&mut f, 3 * SECTOR_SIZE, 0x30);
        assert_eq!(f.selected_sectors().collect::<Vec<_>>(), [0, 3]);
        // The window restarts on each selection.
        f.advance_ns(ERASE_SETUP_NS - 1);
        assert!(f.selecting());
        // A non-erase command in the window cancels the whole operation.
        assert!(f.write16(0, 0x55).is_err());
        assert!(!f.busy());
        assert_eq!(f.selected_sectors().count(), 0);
    }

    /// Reset (F0) is how firmware backs out of a half-finished command or
    /// leaves CFI mode, so it must always return the part to array reads.
    #[test]
    fn a_reset_command_abandons_a_partial_unlock_and_leaves_cfi_mode() {
        let mut f = AmdCore::default();
        cmd(&mut f, UNLOCK0, 0xaa);
        cmd(&mut f, 0, 0xf0);
        // The unlock is forgotten: a program setup now is not a valid cycle.
        assert!(f.write16(UNLOCK1, 0x55).is_err());

        cmd(&mut f, CFI, 0x98);
        assert!(!f.array_reads());
        cmd(&mut f, 0, 0xf0);
        assert!(f.array_reads());
        assert_eq!(f.read16(0x10 * 2), None);
    }

    /// Firmware that probes the part reads the CFI "QRY" signature and the
    /// device-size exponent to learn how much flash it has.
    #[test]
    fn the_cfi_query_returns_the_qry_signature_and_the_size_exponent() {
        let mut f = AmdCore::default();
        cmd(&mut f, CFI, 0x98);
        let q = |f: &mut AmdCore, word: usize| f.read16(word * 2).unwrap();
        assert_eq!([q(&mut f, 0x10), q(&mut f, 0x11), q(&mut f, 0x12)], [b'Q' as u16, b'R' as u16, b'Y' as u16]);
        assert_eq!(1usize << q(&mut f, 0x27), PART_SIZE);
    }

    /// Word program is the unlock sequence, A0, then one data cycle; the data
    /// cycle goes to the owner as an effect and is not decoded as a command.
    #[test]
    fn a_program_sequence_hands_the_data_word_to_the_owner() {
        let mut f = AmdCore::default();
        unlock(&mut f);
        cmd(&mut f, UNLOCK0, 0xa0);
        // 0xf0 as data must not be taken for a reset.
        assert_eq!(cmd(&mut f, 0x1234 * 2, 0xa5f0), Effect::Program { offset: 0x1234 * 2, value: 0xa5f0 });
        assert!(f.array_reads());
        // Odd byte offsets are not word cycles.
        assert!(f.write16(UNLOCK0 + 1, 0xaa).is_err());
    }
}
