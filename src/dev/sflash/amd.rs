//! The AMD flash command set: unlock, word program, sector erase and CFI
//! query state, over storage supplied by the caller.
//!
//! Array bytes and bus translation belong to the enclosing device. A
//! [`Geometry`] says what the part is: its bus width, sectors, unlock
//! addresses, identification codes and which commands it answers. The default
//! is the 8 MiB x16 part Fuel uses for each boot-flash array; an SN0 node's
//! boot flash (1 MiB x8) and a BaseIO card's (1 MiB x16, read-only) are the
//! other two. The CFI data and timing are compatibility values; they do not
//! identify a fitted flash chip. Buffered writes, protection, bypass and erase
//! suspend are absent.
//!
//! Ported from MAME's intelfsh.cpp AMD command path, revision
//! 2ee9b1e30299b122b131cab0bd7912abc2b095ba (BSD-3-Clause).

// Copyright (c) Aaron Giles

use crate::devlog::LogModule;

pub const SECTOR_SIZE: usize = 64 * 1024;
/// The most sectors a part can have here, and Fuel's part has.
const SECTORS: usize = 128;
pub const PART_SIZE: usize = SECTOR_SIZE * SECTORS;
pub const ERASE_SETUP_NS: u64 = 50_000;
pub const ERASE_NS: u64 = 512_000_000;

/// What the part answers in autoselect mode: the manufacturer, device and
/// sector-protection words at the first three command addresses, and `other`
/// everywhere else.
#[derive(Clone, Copy, Debug)]
pub struct Identity {
    pub codes: [u16; 3],
    pub other: u16,
}

impl Identity {
    fn code(&self, index: usize) -> u16 {
        self.codes.get(index).copied().unwrap_or(self.other)
    }
}

/// What distinguishes one part from another to the command engine. Command
/// addresses are word addresses on a x16 part and byte addresses on a x8
/// part; a bus cycle carries a word or a byte accordingly.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub x8: bool,
    pub sector_size: usize,
    pub sectors: usize,
    /// The two unlock command addresses.
    pub unlock: [usize; 2],
    /// The bits of a command address the part compares.
    pub address_mask: usize,
    pub identity: Identity,
    /// Whether the CFI query command (98 at 0x55) is answered.
    pub cfi: bool,
    /// Whether program and erase are refused rather than carried out.
    pub read_only: bool,
    /// Whether sector erase takes time and shows toggle status meanwhile. An
    /// untimed erase is selected and left to the owner at once: see
    /// [`AmdCore::selected_sectors`] and [`AmdCore::finish_erase`].
    pub timed_erase: bool,
    /// Whether a new command sequence may start from autoselect mode without
    /// a reset between.
    pub unlock_from_identify: bool,
    /// Where an unsupported cycle is logged.
    pub log: LogModule,
}

impl Geometry {
    /// Fuel's 8 MiB x16 part.
    pub const FUEL: Self = Self {
        x8: false,
        sector_size: SECTOR_SIZE,
        sectors: SECTORS,
        unlock: [0x555, 0x2aa],
        address_mask: usize::MAX,
        identity: Identity { codes: [0; 3], other: 0 },
        cfi: true,
        read_only: false,
        timed_erase: true,
        unlock_from_identify: false,
        log: LogModule::Bedrock,
    };

    fn size(&self) -> usize {
        self.sector_size * self.sectors
    }

    /// The command address a bus offset names.
    fn command_address(&self, offset: usize) -> usize {
        (if self.x8 { offset } else { offset / 2 }) & self.address_mask
    }
}

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
    geometry: Geometry,
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
        Self::new(Geometry::FUEL)
    }
}

impl AmdCore {
    pub fn new(geometry: Geometry) -> Self {
        assert!(geometry.sectors <= SECTORS, "a part has at most {SECTORS} sectors here");
        Self {
            geometry,
            mode: Mode::Array,
            selected: [false; SECTORS],
            remaining_ns: 0,
            setup_ns: 0,
            toggle: false,
            failed: false,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.geometry);
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
            let selected = self.selected[offset / self.geometry.sector_size];
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
            // The geometry's codes; Fuel's part leaves them unspecified.
            Mode::Identify => Some(self.geometry.identity.code(self.geometry.command_address(offset) & 0xff)),
            _ => None,
        }
    }
    /// One bus cycle: a word on a x16 part, a byte on a x8 part, at byte
    /// offset `offset`.
    pub fn write16(&mut self, offset: usize, value: u16) -> Result<Effect, ()> {
        let g = self.geometry;
        if (!g.x8 && offset & 1 != 0) || offset >= g.size() {
            return Err(());
        }
        let cmd = value as u8;
        let word = g.command_address(offset);
        if self.busy() {
            if (self.failed || self.setup_ns != 0) && cmd == 0xf0 {
                self.reset();
            } else if self.setup_ns != 0 && cmd == 0x30 {
                self.selected[offset / g.sector_size] = true;
                self.setup_ns = ERASE_SETUP_NS;
                self.remaining_ns = ERASE_SETUP_NS + ERASE_NS;
            } else if self.setup_ns != 0 {
                self.reset();
                crate::dlog!(
                    g.log,
                    "amd-flash: sector selection cancelled by command {cmd:#x} at word offset {word:#x}"
                );
                return Err(());
            } else if cmd != 0xf0 {
                crate::dlog!(
                    g.log,
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
        let [unlock0, unlock1] = g.unlock;
        let next = match (self.mode, word, cmd) {
            (Mode::Array, 0x55, 0x98) if g.cfi => Mode::Query,
            (Mode::Array, w, 0xaa) if w == unlock0 => Mode::Unlock1,
            (Mode::Identify, w, 0xaa) if g.unlock_from_identify && w == unlock0 => Mode::Unlock1,
            (Mode::Unlock1, w, 0x55) if w == unlock1 => Mode::Unlock2,
            (Mode::Unlock2, w, 0xa0) if w == unlock0 && !g.read_only => Mode::Program,
            (Mode::Unlock2, w, 0x80) if w == unlock0 && !g.read_only => Mode::Erase1,
            (Mode::Unlock2, w, 0x90) if w == unlock0 => Mode::Identify,
            (Mode::Erase1, w, 0xaa) if w == unlock0 => Mode::Erase2,
            (Mode::Erase2, w, 0x55) if w == unlock1 => Mode::Erase3,
            (Mode::Erase3, _, 0x30) => {
                self.selected[offset / g.sector_size] = true;
                if g.timed_erase {
                    self.setup_ns = ERASE_SETUP_NS;
                    self.remaining_ns = ERASE_SETUP_NS + ERASE_NS;
                }
                Mode::Array
            }
            _ => {
                crate::dlog!(
                    g.log,
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

    const NODE: Geometry = Geometry {
        x8: true,
        sector_size: SECTOR_SIZE,
        sectors: 16,
        unlock: [0x5555, 0x2aaa],
        address_mask: usize::MAX,
        identity: Identity { codes: [0x01, 0xd5, 0], other: 0xff },
        cfi: false,
        read_only: false,
        timed_erase: false,
        unlock_from_identify: true,
        log: LogModule::Hub,
    };

    /// A byte-wide part takes commands at byte addresses and a data byte per
    /// cycle, its erase is selected for the owner to do at once with no busy
    /// status, and it answers neither a CFI query nor an offset past its end.
    #[test]
    fn a_byte_wide_part_takes_byte_cycles_and_erases_at_once() {
        let mut f = AmdCore::new(NODE);
        let cycle = |f: &mut AmdCore, at: usize, v: u16| f.write16(at, v).unwrap();
        cycle(&mut f, 0x5555, 0xaa);
        cycle(&mut f, 0x2aaa, 0x55);
        cycle(&mut f, 0x5555, 0xa0);
        assert_eq!(cycle(&mut f, 0x1235, 0xf0), Effect::Program { offset: 0x1235, value: 0xf0 }, "an odd address, and F0 as data");
        for (at, v) in [(0x5555, 0xaa), (0x2aaa, 0x55), (0x5555, 0x80), (0x5555, 0xaa), (0x2aaa, 0x55), (3 * SECTOR_SIZE + 9, 0x30)] {
            cycle(&mut f, at, v);
        }
        assert!(!f.busy() && f.array_reads(), "no erase status is ever shown");
        assert_eq!(f.selected_sectors().collect::<Vec<_>>(), [3]);
        f.finish_erase(true);
        assert_eq!(f.selected_sectors().count(), 0);
        assert!(f.write16(0x55, 0x98).is_err(), "no CFI query");
        assert!(f.write16(16 * SECTOR_SIZE, 0xaa).is_err(), "past the end of the part");
    }

    /// The codes come from the geometry: addresses 0, 1 and 2 of autoselect,
    /// and the part's filler elsewhere. A new sequence starts from autoselect
    /// only on a part that allows it.
    #[test]
    fn autoselect_answers_the_geometrys_codes() {
        let mut f = AmdCore::new(NODE);
        for (at, v) in [(0x5555, 0xaa), (0x2aaa, 0x55), (0x5555, 0x90)] {
            f.write16(at, v).unwrap();
        }
        assert_eq!([0, 1, 2, 3, 0x100].map(|a| f.read16(a).unwrap()), [0x01, 0xd5, 0, 0xff, 0x01]);
        f.write16(0x5555, 0xaa).expect("this part starts a sequence from autoselect");
        let mut strict = AmdCore::new(Geometry { unlock_from_identify: false, ..NODE });
        for (at, v) in [(0x5555, 0xaa), (0x2aaa, 0x55), (0x5555, 0x90)] {
            strict.write16(at, v).unwrap();
        }
        assert!(strict.write16(0x5555, 0xaa).is_err());
    }

    /// A read-only part gets as far as autoselect and refuses program and
    /// erase setup.
    #[test]
    fn a_read_only_part_refuses_program_and_erase() {
        let mut f = AmdCore::new(Geometry { read_only: true, ..NODE });
        for setup in [0xa0, 0x80] {
            f.write16(0x5555, 0xaa).unwrap();
            f.write16(0x2aaa, 0x55).unwrap();
            assert!(f.write16(0x5555, setup).is_err(), "{setup:#x}");
        }
    }

    /// The default geometry is Fuel's, unchanged by the others.
    #[test]
    fn the_default_geometry_is_fuels_part() {
        let g = AmdCore::default().geometry;
        assert_eq!((g.x8, g.sector_size * g.sectors, g.unlock, g.timed_erase, g.cfi), (false, PART_SIZE, [0x555, 0x2aa], true, true));
    }
}
