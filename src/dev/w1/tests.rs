//! The bus, the parts and the slot decode, driven the way a guest does.
//!
//! Ported from tests/unit/test-onewire.c in jamesbraid/qemu-sgi-origami.

use super::crc::{crc16, crc8};
use super::slot::{self, fields, ONE_PULSE, RESET_PULSE};
use super::*;
use crate::dev::w1::sgi::{board_record_part, ethernet_address_part, BoardRecord, ETHERNET_PART_SERIAL};
use crate::dev::w1::Ds2502;

const MAC: [u8; 6] = [0x08, 0x00, 0x69, 0x12, 0x34, 0x56];

/// A master on one bus: slots at pulse widths, as a MicroLAN register does.
struct M(Bus);

impl M {
    fn reset(&mut self) -> bool {
        !slot::slot(&mut self.0, 500, 0)
    }
    fn bit(&mut self, b: bool) -> bool {
        slot::slot(&mut self.0, if b { 6 } else { 80 }, 13)
    }
    fn write(&mut self, b: u8) {
        for i in 0..8 {
            self.bit(b >> i & 1 != 0);
        }
    }
    fn read(&mut self) -> u8 {
        (0..8).fold(0, |a, i| a | (self.bit(true) as u8) << i)
    }
    fn read_n(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.read()).collect()
    }
    /// Skip ROM, a memory command and its address.
    fn command(&mut self, cmd: u8, ta: u16) {
        assert!(self.reset());
        self.write(0xcc);
        self.write(cmd);
        self.write(ta as u8);
        self.write((ta >> 8) as u8);
    }
    fn skip(&mut self, cmd: u8, ta: u16, n: usize) -> Vec<u8> {
        self.command(cmd, ta);
        self.read_n(n)
    }
    fn match_rom(&mut self, rom: &[u8; 8]) {
        assert!(self.reset());
        self.write(0x55);
        for b in rom {
            self.write(*b);
        }
    }
    /// Search ROM, enumerating every part: depth first over the
    /// discrepancy tree, each pass replaying the choices so far.
    fn search_all(&mut self) -> Vec<[u8; 8]> {
        fn walk(m: &mut M, path: &[bool], found: &mut Vec<[u8; 8]>) {
            assert!(m.reset());
            m.write(0xf0);
            let mut rom = [0u8; 8];
            let mut branch_at = None;
            for i in 0..64 {
                let b = m.bit(true);
                let c = m.bit(true);
                let choice = match (b, c) {
                    (true, true) => return,
                    (true, false) => true,
                    (false, true) => false,
                    (false, false) if i < path.len() => path[i],
                    (false, false) => {
                        branch_at = Some(i);
                        false
                    }
                };
                m.bit(choice);
                rom[i / 8] |= (choice as u8) << (i % 8);
            }
            match branch_at {
                Some(at) => {
                    let prefix: Vec<bool> = (0..at).map(|i| rom[i / 8] >> (i % 8) & 1 != 0).collect();
                    for v in [false, true] {
                        let mut p = prefix.clone();
                        p.push(v);
                        walk(m, &p, found);
                    }
                }
                None if !found.contains(&rom) => found.push(rom),
                None => {}
            }
        }
        let mut found = vec![];
        walk(self, &[], &mut found);
        found
    }
}

fn image(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 3 + 1) as u8).collect()
}

/// The IOC3's wire: the Ethernet part and two board records.
fn octane_wire() -> M {
    let mut b = Bus::new();
    b.add(Box::new(ethernet_address_part(MAC, ETHERNET_PART_SERIAL)));
    b.add(Box::new(board_record_part(
        &BoardRecord { barcode: "EMUF00", part: "030-0891-003", revision: "A", name: "FP1" },
        0x5145_4d55_4600,
    )));
    b.add(Box::new(board_record_part(
        &BoardRecord { barcode: "EMUS00", part: "060-0028-003", revision: "A", name: "PWR.SPPLY.SR" },
        0x5145_4d55_5300,
    )));
    M(b)
}

fn one(part: Box<dyn Part>) -> M {
    let mut b = Bus::new();
    b.add(part);
    M(b)
}

// ---------------------------------------------------------------------------
// CRCs
// ---------------------------------------------------------------------------

#[test]
fn crc8_over_a_whole_rom_is_zero() {
    // The Dallas application-note example ROM.
    let rom = [0x02, 0x1c, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xa2];
    assert_eq!(crc8(&rom[..7]), 0xa2);
    assert_eq!(crc8(&rom), 0);
}

#[test]
fn crc8_accumulates_and_is_not_inverted() {
    assert_eq!(crc8(&[]), 0);
    assert_eq!(crc8(&[0x01]), 0x5e);
    assert_ne!(crc8(&[0x00, 0x00]), 0xff);
}

#[test]
fn crc16_matches_the_standard_check_value_and_accumulates() {
    assert_eq!(crc16(0, b"123456789"), 0xbb3d);
    let (a, b) = b"123456789".split_at(4);
    assert_eq!(crc16(crc16(0, a), b), 0xbb3d);
    // A message and its inverted CRC, low byte first, leave 0xb001.
    let msg = b"hello";
    let c = !crc16(0, msg);
    let mut with = msg.to_vec();
    with.extend_from_slice(&c.to_le_bytes());
    assert_eq!(crc16(0, &with), 0xb001);
}

// ---------------------------------------------------------------------------
// The bus
// ---------------------------------------------------------------------------

#[test]
fn an_empty_bus_gets_no_presence_and_reads_ones() {
    let mut m = M(Bus::new());
    assert!(!m.reset());
    m.write(0x33);
    assert_eq!(m.read_n(8), vec![0xff; 8]);
}

#[test]
fn the_bus_is_quiet_until_its_first_reset() {
    let mut m = one(Box::new(Ds2502::new(0x09, 1, &[])));
    assert_eq!(m.read(), 0xff);
    m.write(0x33);
    assert_eq!(m.read(), 0xff, "no ROM command was taken");
    assert!(m.reset());
}

#[test]
fn read_rom_returns_the_identity_and_a_valid_crc() {
    let p = Ds2502::new(0x91, 0x5e74_54d5_5000, &[]);
    let rom = p.rom();
    let mut m = one(Box::new(p));
    assert!(m.reset());
    m.write(0x33);
    assert_eq!(m.read_n(8), rom);
    assert_eq!(rom[0], 0x91);
    assert_eq!(&rom[1..7], &0x5e74_54d5_5000u64.to_le_bytes()[..6]);
    assert_eq!(crc8(&rom), 0);
}

#[test]
fn a_reset_does_not_change_rom_or_memory() {
    let mut m = one(Box::new(Ds2502::new(0x09, 7, &image(128))));
    let a = m.skip(0xf0, 0, 20);
    assert!(m.reset());
    assert_eq!(m.skip(0xf0, 0, 20), a);
}

#[test]
fn a_search_finds_every_part_with_a_valid_rom() {
    let mut m = octane_wire();
    let found = m.search_all();
    assert_eq!(found.len(), 3, "the Ethernet part and two board records");
    for rom in &found {
        assert_eq!(crc8(rom), 0, "ROM CRC-8");
    }
    let eth = found.iter().find(|r| r[0] == 0x91).expect("family 0x91");
    // UniqueWare prefix 0x5e7 in the top twelve bits of the serial.
    assert_eq!(eth[6], 0x5e);
    assert_eq!(eth[5] >> 4, 0x7);
    assert_eq!(found.iter().filter(|r| r[0] == 0x0b).count(), 2);
    assert_eq!(found.len(), m.0.roms().len());
}

#[test]
fn skip_rom_with_several_parts_wires_their_answers_together() {
    let mut m = octane_wire();
    let found = m.search_all();
    assert!(m.reset());
    m.write(0x33);
    let got = m.read_n(8);
    let mut and = [0xffu8; 8];
    for rom in &found {
        for i in 0..8 {
            and[i] &= rom[i];
        }
    }
    assert_eq!(got, and);
}

#[test]
fn a_completed_search_leaves_the_part_selected() {
    let mut m = octane_wire();
    let eth = *m.search_all().iter().find(|r| r[0] == 0x91).unwrap();
    assert!(m.reset());
    m.write(0xf0);
    for i in 0..64 {
        let want = eth[i / 8] >> (i % 8) & 1 != 0;
        m.bit(true);
        m.bit(true);
        m.bit(want);
    }
    m.write(0xc3);
    m.write(0);
    m.write(0);
    assert_eq!(m.read(), crc8(&[0xc3, 0, 0]));
    assert_eq!(m.read(), 0x0a, "marker");
}

#[test]
fn match_rom_selects_only_on_an_exact_identity() {
    let mut m = octane_wire();
    let eth = *m.search_all().iter().find(|r| r[0] == 0x91).unwrap();
    m.match_rom(&eth);
    m.write(0xf0);
    m.write(0);
    m.write(0);
    assert_eq!(m.read(), crc8(&[0xf0, 0, 0]));
    assert_eq!(m.read(), 0x0a);
}

#[test]
fn a_wrong_match_rom_leaves_every_part_unselected() {
    let mut m = octane_wire();
    let mut eth = *m.search_all().iter().find(|r| r[0] == 0x91).unwrap();
    eth[3] ^= 1;
    m.match_rom(&eth);
    m.write(0xf0);
    m.write(0);
    m.write(0);
    assert_eq!(m.read(), 0xff, "nobody answers");
    // The next reset brings them back.
    assert!(m.reset());
}

/// A DS2502, a DS2501 and a DS2505 with distinct images: families 0x09,
/// 0x91 and 0x0b, so their ROMs part ways in the second bit.
fn three_parts() -> [Box<dyn Part>; 3] {
    let img = |n: usize, mul: usize, add: usize| (0..n).map(|i| (i * mul + add) as u8).collect::<Vec<_>>();
    [
        Box::new(Ds2502::new(0x09, 0x0123_4567_89ab, &img(128, 3, 1))),
        Box::new(Ds2502::ds2501(0x91, 0x5e74_54d5_5000, &img(64, 5, 2))),
        Box::new(Ds2505::new(0x0b, 0x5145_4d55_4600, &img(64, 7, 9))),
    ]
}

fn three_part_wire() -> M {
    let mut b = Bus::new();
    for p in three_parts() {
        b.add(p);
    }
    M(b)
}

#[test]
fn search_rom_on_three_parts_answers_the_and_of_the_parts_still_selected() {
    let roms: Vec<[u8; 8]> = three_parts().iter().map(|p| p.rom()).collect();
    let mut discrepancies = 0;
    // The master follows each part's ROM in turn.
    for target in 0..3 {
        let mut m = three_part_wire();
        assert!(m.reset());
        m.write(0xf0);
        let mut selected = vec![0, 1, 2];
        for i in 0..64 {
            let bit = |k: usize| roms[k][i / 8] >> (i % 8) & 1 != 0;
            // Every selected part sends its bit, then its complement; the
            // wire is low if any of them sends a zero.
            let want_bit = selected.iter().all(|&k| bit(k));
            let want_complement = selected.iter().all(|&k| !bit(k));
            assert_eq!(m.bit(true), want_bit, "target {target} bit {i}");
            assert_eq!(m.bit(true), want_complement, "target {target} complement {i}");
            if !want_bit && !want_complement {
                discrepancies += 1;
            }
            let choice = bit(target);
            m.bit(choice);
            selected.retain(|&k| bit(k) == choice);
        }
        assert_eq!(selected, [target], "only the followed part is left");
        // The part left selected answers the next memory command.
        m.write(0xf0);
        m.write(0);
        m.write(0);
        let want = one(three_parts().into_iter().nth(target).unwrap()).skip(0xf0, 0, 5);
        assert_eq!(m.read_n(5), want, "target {target}");
    }
    assert!(discrepancies > 0, "the families differ, so the parts disagree somewhere");
}

#[test]
fn skip_rom_read_memory_on_three_parts_wires_the_streams_together() {
    let mut want = vec![0xffu8; 16];
    for p in three_parts() {
        for (w, b) in want.iter_mut().zip(one(p).skip(0xf0, 0, 16)) {
            *w &= b;
        }
    }
    assert_eq!(three_part_wire().skip(0xf0, 0, 16), want);
}

#[test]
fn read_rom_on_three_parts_is_the_and_of_their_roms_and_a_bad_match_selects_none() {
    let roms: Vec<[u8; 8]> = three_parts().iter().map(|p| p.rom()).collect();
    let mut and = [0xffu8; 8];
    for rom in &roms {
        for i in 0..8 {
            and[i] &= rom[i];
        }
    }
    let mut m = three_part_wire();
    assert!(m.reset());
    m.write(0x33);
    assert_eq!(m.read_n(8), and);
    // One wrong bit in a real ROM matches no part, so none answers.
    let mut bad = roms[0];
    bad[3] ^= 1;
    m.match_rom(&bad);
    m.write(0xf0);
    m.write(0);
    m.write(0);
    assert_eq!(m.read_n(2), [0xff; 2]);
}

#[test]
fn a_part_leaves_a_match_at_the_first_differing_bit() {
    // Two parts whose ROMs differ in the first byte: after the bit that
    // separates them only the matching one answers.
    let a = Ds2502::new(0x09, 1, &image(128));
    let b = Ds2502::new(0x0a, 1, &image(128).iter().map(|x| !x).collect::<Vec<_>>());
    let rom = b.rom();
    let mut bus = Bus::new();
    bus.add(Box::new(a));
    bus.add(Box::new(b));
    let mut m = M(bus);
    m.match_rom(&rom);
    m.write(0xf0);
    m.write(0);
    m.write(0);
    m.read();
    assert_eq!(m.read(), !1u8, "the second part's image, not the wired-AND of both");
}

#[test]
fn a_driven_zero_is_not_a_driven_one() {
    // A zero written during a ROM command is a zero bit of the command.
    let mut m = one(Box::new(Ds2502::new(0x09, 1, &[])));
    assert!(m.reset());
    m.write(0x32);
    assert_eq!(m.read(), 0xff, "0x32 is no ROM command: the bus goes quiet");
}

#[test]
fn unknown_rom_commands_silence_the_bus_until_the_next_reset() {
    let mut m = one(Box::new(Ds2502::new(0x09, 1, &image(128))));
    assert!(m.reset());
    m.write(0x3c); // overdrive skip: not modelled
    m.write(0xf0);
    m.write(0);
    m.write(0);
    assert_eq!(m.read(), 0xff);
    assert_eq!(m.skip(0xf0, 0, 1)[0], crc8(&[0xf0, 0, 0]), "a reset recovers");
}

#[test]
fn write_commands_are_refused_and_not_misread() {
    for cmd in [0x0f, 0x55] {
        let mut m = one(Box::new(Ds2502::new(0x09, 1, &image(128))));
        assert_eq!(m.skip(cmd, 0, 4), vec![0xff; 4], "ds2502 {cmd:#04x}");
    }
    for cmd in [0x0f, 0xf3, 0x55, 0xf5, 0xc3] {
        let mut m = one(Box::new(Ds2505::new(0x0b, 1, &image(64))));
        assert_eq!(m.skip(cmd, 0, 4), vec![0xff; 4], "ds2505 {cmd:#04x}");
    }
    // Neither part has the other's commands.
    let mut m = one(Box::new(Ds2502::new(0x09, 1, &[])));
    assert_eq!(m.skip(0xa5, 0, 4), vec![0xff; 4]);
}

// ---------------------------------------------------------------------------
// DS2501 / DS2502
// ---------------------------------------------------------------------------

#[test]
fn ds2502_plain_read_has_no_page_crc() {
    let img = image(128);
    let mut m = one(Box::new(Ds2502::new(0x09, 1, &img)));
    let got = m.skip(0xf0, 0, 1 + 128 + 1);
    assert_eq!(got[0], crc8(&[0xf0, 0, 0]));
    assert_eq!(&got[1..129], &img[..], "the pages run straight on");
    assert_eq!(got[129], crc8(&img), "one CRC-8 at the end of memory");
}

#[test]
fn ds2502_read_starting_mid_part_ends_with_one_crc_over_what_was_read() {
    let img = image(128);
    let mut m = one(Box::new(Ds2502::new(0x09, 1, &img)));
    let got = m.skip(0xf0, 0x45, 1 + 59 + 1);
    assert_eq!(got[0], crc8(&[0xf0, 0x45, 0]));
    assert_eq!(&got[1..60], &img[0x45..]);
    assert_eq!(got[60], crc8(&img[0x45..]));
}

#[test]
fn ds2502_read_data_crc_closes_every_32_byte_page() {
    let img = image(128);
    let mut m = one(Box::new(Ds2502::new(0x09, 1, &img)));
    let got = m.skip(0xc3, 0, 1 + 4 * 33);
    assert_eq!(got[0], crc8(&[0xc3, 0, 0]));
    for p in 0..4 {
        let page = &got[1 + p * 33..1 + p * 33 + 33];
        assert_eq!(&page[..32], &img[p * 32..p * 32 + 32]);
        assert_eq!(page[32], crc8(&img[p * 32..p * 32 + 32]), "page {p}");
    }
}

#[test]
fn ds2502_read_data_crc_first_page_can_start_mid_page() {
    let img = image(128);
    let mut m = one(Box::new(Ds2502::new(0x09, 1, &img)));
    let got = m.skip(0xc3, 0x45, 1 + 27 + 1 + 33);
    assert_eq!(&got[1..28], &img[0x45..0x60]);
    assert_eq!(got[28], crc8(&img[0x45..0x60]), "CRC over just the part read");
    assert_eq!(got[29 + 32], crc8(&img[0x60..0x80]), "later pages are whole");
}

#[test]
fn ds2502_start_address_above_top_is_masked_and_the_echoed_crc_covers_the_mask() {
    let img = image(128);
    for ta in [0x00c5u16, 0x0145, 0xff45] {
        let mut m = one(Box::new(Ds2502::new(0x09, 1, &img)));
        let got = m.skip(0xf0, ta, 1 + 59 + 1);
        assert_eq!(&got[1..60], &img[0x45..], "{ta:#06x} aliases onto 0x45");
        // The part checked the address it will read, not the one sent: a
        // master that checks its own bytes sees the mismatch.
        assert_eq!(got[0], crc8(&[0xf0, 0x45, 0]));
        assert_ne!(got[0], crc8(&[0xf0, ta as u8, (ta >> 8) as u8]));
    }
}

#[test]
fn ds2502_status_serves_the_real_field() {
    let mut m = one(Box::new(Ds2502::new(0x09, 1, &[])));
    let got = m.skip(0xaa, 0, 1 + 8 + 1);
    assert_eq!(got[0], crc8(&[0xaa, 0, 0]));
    assert_eq!(&got[1..8], &[0xff; 7], "seven programmable bytes, erased");
    assert_eq!(got[8], 0x00, "factory byte at 0007");
    assert_eq!(got[9], crc8(&got[1..9]));
    let mid = m.skip(0xaa, 3, 1 + 5 + 1);
    assert_eq!(mid[0], crc8(&[0xaa, 3, 0]));
    assert_eq!(mid[5], 0x00);
    assert_eq!(mid[6], crc8(&mid[1..6]));
}

#[test]
fn ds2502_status_start_address_is_not_masked() {
    for ta in [0x0008u16, 0x0080, 0x0107] {
        let mut m = one(Box::new(Ds2502::new(0x09, 1, &[])));
        let got = m.skip(0xaa, ta, 12);
        assert_eq!(got[0], crc8(&[0xaa, ta as u8, (ta >> 8) as u8]), "the CRC covers the address as sent");
        assert_eq!(&got[1..], &[0xff; 11], "past the field: erased bytes, no closing CRC and no wrap to 0007");
    }
}

#[test]
fn ds2502_pages_bound_the_programmable_area() {
    let img = image(128);
    let full = Ds2502::new(0x09, 1, &img);
    assert_eq!(full.programmable(), 128);
    assert_eq!(full.memory(), &img[..]);
    let half = Ds2502::ds2501(0x91, 1, &img);
    assert_eq!(half.programmable(), 64);
    assert_eq!(&half.memory()[..64], &img[..64]);
    assert_eq!(&half.memory()[64..], &[0xff; 64][..], "pages beyond the part read erased");
    // An image longer than the memory is cut, not wrapped.
    let long = Ds2502::new(0x09, 1, &[0x11; 300]);
    assert_eq!(long.memory().len(), 128);
}

#[test]
fn ds2501_start_address_is_masked_to_the_128_byte_space() {
    // The shared model follows the DS2502 datasheet's 7-bit mask for both
    // parts; the earlier DS2501 model masked to 63. The PROM reads this
    // part from 0 only (14 bytes), where the two agree.
    let mut m = one(Box::new(Ds2502::ds2501(0x91, 1, &image(64))));
    let got = m.skip(0xf0, 0x40, 1 + 64);
    assert_eq!(&got[1..], &[0xff; 64], "0x40 is an erased page, not an alias of 0");
    let aliased = m.skip(0xf0, 0x80, 3);
    assert_eq!(&aliased[1..], &image(64)[..2], "0x80 wraps to 0");
}

// ---------------------------------------------------------------------------
// DS2505
// ---------------------------------------------------------------------------

fn inv(c: u16) -> [u8; 2] {
    (!c).to_le_bytes()
}

#[test]
fn ds2505_plain_read_has_no_header_crc_or_page_crc() {
    let img = image(64);
    let mut m = one(Box::new(Ds2505::new(0x0b, 1, &img)));
    let got = m.skip(0xf0, 0, 70);
    assert_eq!(&got[..64], &img[..], "data starts straight after the address");
    assert_eq!(&got[64..], &[0xff; 6][..], "then the erased tail, no page CRC");
}

#[test]
fn ds2505_read_memory_emits_one_crc16_after_the_last_byte() {
    let img = image(64);
    let mut m = one(Box::new(Ds2505::new(0x0b, 1, &img)));
    let got = m.skip(0xf0, 0, 2048 + 2);
    let mut mem = img.clone();
    mem.resize(2048, 0xff);
    let crc = crc16(crc16(0, &[0xf0, 0, 0]), &mem);
    assert_eq!(&got[2048..], &inv(crc), "inverted, low byte first, covering the header too");
    // From a mid-part start, over the header with that start and the bytes read.
    let mut m = one(Box::new(Ds2505::new(0x0b, 1, &img)));
    let got = m.skip(0xf0, 0x7f0, 16 + 2);
    assert_eq!(&got[16..], &inv(crc16(crc16(0, &[0xf0, 0xf0, 0x07]), &mem[0x7f0..])));
}

#[test]
fn ds2505_status_pages_carry_crc16s_the_first_covering_the_header() {
    let mut m = one(Box::new(Ds2505::new(0x0b, 1, &[])));
    let got = m.skip(0xaa, 0x100, 2 * 10);
    assert_eq!(&got[..8], &[0xff; 8]);
    let first = crc16(crc16(0, &[0xaa, 0x00, 0x01]), &[0xff; 8]);
    assert_eq!(&got[8..10], &inv(first));
    // Later pages' CRCs start from zero, over their own bytes.
    assert_eq!(&got[10..18], &[0xff; 8]);
    assert_eq!(&got[18..20], &inv(crc16(0, &[0xff; 8])));
}

#[test]
fn ds2505_status_start_address_is_masked_to_the_top() {
    let mut a = one(Box::new(Ds2505::new(0x0b, 1, &[])));
    let mut b = one(Box::new(Ds2505::new(0x0b, 1, &[])));
    assert_eq!(a.skip(0xaa, 0x813, 40), b.skip(0xaa, 0x013, 40));
}

#[test]
fn ds2505_extended_read_runs_redirection_byte_crc_page_crc() {
    let img = image(64);
    let mut m = one(Box::new(Ds2505::new(0x0b, 1, &img)));
    let got = m.skip(0xa5, 0, 2 * 37);
    // Page 0: the redirection byte, a CRC-16 over header + that byte, the
    // 32 data bytes, a CRC-16 over just them.
    assert_eq!(got[0], 0xff);
    assert_eq!(&got[1..3], &inv(crc16(crc16(0, &[0xa5, 0, 0]), &[0xff])));
    assert_eq!(&got[3..35], &img[..32]);
    assert_eq!(&got[35..37], &inv(crc16(0, &img[..32])));
    // Page 1: its redirection CRC covers the byte alone.
    assert_eq!(got[37], 0xff);
    assert_eq!(&got[38..40], &inv(crc16(0, &[0xff])));
    assert_eq!(&got[40..72], &img[32..64]);
}

#[test]
fn ds2505_extended_read_first_page_can_start_mid_page() {
    let img = image(64);
    let mut m = one(Box::new(Ds2505::new(0x0b, 1, &img)));
    let got = m.skip(0xa5, 0x25, 3 + 27 + 2);
    assert_eq!(&got[1..3], &inv(crc16(crc16(0, &[0xa5, 0x25, 0]), &[0xff])));
    assert_eq!(&got[3..30], &img[0x25..0x40][..27].to_vec()[..]);
    assert_eq!(&got[30..32], &inv(crc16(0, &img[0x25..0x40])), "CRC over just what was read of the page");
}

#[test]
fn ds2505_start_address_is_masked_and_the_crc_covers_the_mask() {
    let mut mem = image(64);
    mem.resize(2048, 0xff);
    let mut m = one(Box::new(Ds2505::new(0x0b, 1, &image(64))));
    let got = m.skip(0xf0, 0x8f0, 0x10 + 2);
    assert_eq!(&got[..16], &mem[0xf0..0x100]);
    let end = m.skip(0xf0, 0x8f0, 2048 - 0xf0 + 2);
    assert_eq!(&end[2048 - 0xf0..], &inv(crc16(crc16(0, &[0xf0, 0xf0, 0x00]), &mem[0xf0..])));
}

#[test]
fn ds2505_reads_past_the_top_pad_with_ones_and_do_not_wrap() {
    let mut m = one(Box::new(Ds2505::new(0x0b, 1, &image(64))));
    let got = m.skip(0xf0, 0x7ff, 1 + 2 + 8);
    assert_eq!(got[0], 0xff);
    assert_eq!(&got[3..], &[0xff; 8], "nothing after the CRC-16, and no wrap to 0");
    // An A5 stream stops after the last page too.
    let mut m = one(Box::new(Ds2505::new(0x0b, 1, &image(64))));
    let got = m.skip(0xa5, 0x7e0, 37 + 8);
    assert_eq!(&got[37..], &[0xff; 8]);
}

#[test]
fn ds2505_image_longer_than_the_memory_is_cut() {
    let p = Ds2505::new(0x0b, 1, &vec![0x11; 4000]);
    assert_eq!(p.memory().len(), 2048);
}

// ---------------------------------------------------------------------------
// Slot decode
// ---------------------------------------------------------------------------

#[test]
fn the_register_fields_are_the_pulse_and_the_sample() {
    assert_eq!(fields(520 << 10 | 65 << 2), (520, 65));
    assert_eq!(fields(0x3ff << 10 | 0xff << 2 | 3), (0x3ff, 0xff));
    assert_eq!(fields(1 << 20 | 5 << 10), (5, 0), "bits above 19 are not the pulse");
}

#[test]
fn slot_reset_reports_presence_with_a_part_and_absence_without() {
    let mut with = Bus::new();
    with.add(Box::new(Ds2502::new(0x09, 1, &[])));
    assert!(!slot::slot(&mut with, 500, 0), "a part pulls the wire low");
    let mut empty = Bus::new();
    assert!(slot::slot(&mut empty, 500, 0), "an empty wire floats high");
}

#[test]
fn slot_reset_threshold_is_closed_at_400() {
    let mut b = Bus::new();
    b.add(Box::new(Ds2502::new(0x09, 1, &[])));
    assert_eq!(RESET_PULSE, 400);
    assert!(!slot::slot(&mut b, 400, 0), "400 is a reset: the part answers");
    // Mid Read ROM (the family byte's low bit is 1) a 399 is one more slot
    // reading the next ROM bit, where a 400 would restart the bus.
    for i in 0..8 {
        slot::slot(&mut b, if 0x33 >> i & 1 != 0 { 6 } else { 80 }, 0);
    }
    assert!(slot::slot(&mut b, 399, 0), "399 is a slot, not a reset");
    assert!(!slot::slot(&mut b, 400, 0), "400 is a reset");
}

#[test]
fn slot_zero_threshold_is_closed_at_40() {
    assert_eq!(ONE_PULSE, 40);
    let mut b = Bus::new();
    b.add(Box::new(Ds2502::new(0x09, 1, &[])));
    // Reset then the Read ROM command byte 0x01 would be a one; use the
    // ROM phase, where the level echoes the master's bit.
    let mut bit_at = |pulse| {
        slot::slot(&mut b, 500, 0);
        slot::slot(&mut b, pulse, 0)
    };
    assert!(bit_at(39), "39 is a one");
    assert!(!bit_at(40), "40 is a zero");
}

#[test]
fn slot_short_pulse_reads_the_part_and_mid_pulse_writes_zero() {
    let p = Ds2502::new(0x09, 1, &[]);
    let rom = p.rom();
    let mut b = Bus::new();
    b.add(Box::new(p));
    slot::slot(&mut b, 500, 0);
    for i in 0..8 {
        slot::slot(&mut b, if 0x33 >> i & 1 != 0 { 10 } else { 100 }, 13);
    }
    let level: Vec<bool> = (0..8).map(|_| slot::slot(&mut b, 10, 13)).collect();
    let got = level.iter().enumerate().fold(0u8, |a, (i, l)| a | (*l as u8) << i);
    assert_eq!(got, rom[0]);
}

#[test]
fn slot_wired_sees_both_nets_and_ands_the_levels() {
    let mut own = Bus::new();
    let mut shared = Bus::new();
    shared.add(Box::new(Ds2502::new(0x09, 1, &[])));
    assert!(!slot::slot_wired(&mut own, Some(&mut shared), 500, 0), "presence on the second net alone");
    assert!(slot::slot_wired(&mut own, None, 500, 0), "nothing on this one");
    let mut a = Bus::new();
    a.add(Box::new(Ds2502::new(0x0a, 2, &[])));
    assert!(!slot::slot_wired(&mut a, Some(&mut Bus::new()), 500, 0));
    // Both nets track the same command bytes.
    let mut a = Bus::new();
    let mut b = Bus::new();
    a.add(Box::new(Ds2502::new(0x09, 1, &image(128))));
    b.add(Box::new(Ds2502::new(0x0a, 2, &image(128))));
    slot::slot_wired(&mut a, Some(&mut b), 500, 0);
    for i in 0..8 {
        slot::slot_wired(&mut a, Some(&mut b), if 0xcc >> i & 1 != 0 { 6 } else { 80 }, 0);
    }
    assert_eq!(a.roms().len() + b.roms().len(), 2);
}

// ---------------------------------------------------------------------------
// What the PROM and guests read
// ---------------------------------------------------------------------------

fn read_page_via(m: &mut M, rom: &[u8; 8], cmd: u8) -> [u8; 32] {
    m.match_rom(rom);
    m.write(cmd);
    m.write(0);
    m.write(0);
    assert_eq!(m.read(), crc8(&[cmd, 0, 0]), "leading CRC-8 over the command");
    let mut page = [0u8; 32];
    for b in &mut page {
        *b = m.read();
    }
    page
}

fn assert_ethernet_page(page: &[u8; 32]) {
    assert_eq!(page[0], 0x0a);
    let mut mac = [0u8; 6];
    for i in 0..6 {
        mac[i] = page[10 - i];
    }
    assert_eq!(mac, MAC, "station address stored reversed");
    assert_eq!(crc16(0, &page[..13]), 0xb001, "embedded CRC-16 residue");
}

fn ethernet_rom(m: &mut M) -> [u8; 8] {
    *m.search_all().iter().find(|r| r[0] == 0x91).expect("the Ethernet-address part")
}

#[test]
fn reading_the_ethernet_page_via_read_memory() {
    let mut m = octane_wire();
    let eth = ethernet_rom(&mut m);
    assert_ethernet_page(&read_page_via(&mut m, &eth, 0xf0));
}

#[test]
fn reading_the_ethernet_page_via_read_memory_crc() {
    let mut m = octane_wire();
    let eth = ethernet_rom(&mut m);
    let page = read_page_via(&mut m, &eth, 0xc3);
    assert_ethernet_page(&page);
    assert_eq!(m.read(), crc8(&page), "CRC-8 closing the page");
}

#[test]
fn plain_read_memory_has_no_page_crc() {
    let mut m = octane_wire();
    let eth = ethernet_rom(&mut m);
    let _ = read_page_via(&mut m, &eth, 0xf0);
    assert_eq!(m.read(), 0xff, "the second page's first byte, erased");
}

#[test]
fn read_status_returns_no_redirection() {
    let mut m = octane_wire();
    let eth = ethernet_rom(&mut m);
    m.match_rom(&eth);
    m.write(0xaa);
    m.write(1);
    m.write(0);
    assert_eq!(m.read(), crc8(&[0xaa, 1, 0]));
    let st = m.read_n(7);
    assert_eq!(&st[..4], &[0xff; 4], "no page redirected");
    assert_eq!(st[6], 0x00, "factory-programmed byte at 0007");
    assert_eq!(m.read(), crc8(&st));
}

#[test]
fn the_board_records_read_back_as_text() {
    let mut m = octane_wire();
    let found = m.search_all();
    let mut names = vec![];
    for rom in found.iter().filter(|r| r[0] == 0x0b) {
        m.match_rom(rom);
        m.write(0xf0);
        m.write(0);
        m.write(0);
        let rec = m.read_n(64);
        assert_eq!(rec[0], 0x01, "format byte");
        assert_eq!(crc16(0, &rec[..32]), 0xb001, "page 0 residue");
        assert_eq!(crc16(0, &rec[32..64]), 0xb001, "page 1 residue");
        assert_eq!(&rec[42..48], &[0xff; 6]);
        names.push(String::from_utf8(rec[48..62].to_vec()).unwrap().trim_end().to_string());
    }
    names.sort();
    assert_eq!(names, ["FP1", "PWR.SPPLY.SR"]);
}

fn bus() -> Bus {
    let mut b = Bus::new();
    b.add(Box::new(Ds2502::new(0x09, 0x1234, b"hello")));
    b
}

/// A bus saved partway through reading a part's ROM carries on the read
/// from the next bit after a load.
#[test]
fn a_read_rom_in_progress_continues_after_a_load() {
    let mut src = bus();
    assert!(src.reset());
    for i in 0..8 {
        src.slot(0x33 >> i & 1 != 0);
    }
    let first: Vec<bool> = (0..10).map(|_| src.slot(true)).collect();
    let v = src.save_state();
    let mut dst = bus();
    dst.load_state(&v).unwrap();
    assert_eq!(v, dst.save_state());
    let rest: Vec<bool> = (0..54).map(|_| dst.slot(true)).collect();
    let rom = dst.roms()[0];
    let bits: Vec<bool> = (0..64).map(|i| rom[i / 8] >> (i % 8) & 1 != 0).collect();
    assert_eq!([first, rest].concat(), bits);

    let mut other = Bus::new();
    other.add(Box::new(Ds2502::new(0x09, 0x9999, b"hello")));
    assert!(other.load_state(&v).is_err());
}

/// A saved ROM phase past the ROM's 64 bits is refused rather than loaded
/// to index past it on the next slot.
#[test]
fn a_rom_bit_past_the_rom_is_refused() {
    let src = bus();
    for tag in 1..=3u64 {
        let mut v = src.save_state();
        let phase = crate::snapshot::u64_slice_to_toml(&[tag, 64, 0]);
        v.as_table_mut().unwrap().insert("phase".into(), phase);
        let e = bus().load_state(&v).unwrap_err();
        assert!(e.contains("ROM bit 64"), "phase {tag}: {e}");
    }
}
