//! SGI's "Number In a Can" records: what a board's Dallas add-only memory
//! holds, a part number, name and barcode, or the Ethernet address. The bus,
//! the parts and the master's slot decode live in the parent module `w1`; this
//! module only knows the SGI record formats and builds parts that carry
//! them. The barcodes and ROM serials we ship are synthetic.
//!
//! Ported from hw/misc/sgi-board-record.c in jamesbraid/qemu-sgi-origami.

use crate::dev::w1;
use crate::dev::w1::crc::crc16 as crc16_seed;

/// Text fields of an SGI board record, as a part's first two 32-byte pages
/// hold them, each page closed by an inverted CRC-16.
#[derive(Clone, Copy, Debug)]
pub struct BoardRecord<'a> {
    pub barcode: &'a str,
    /// Twelve characters: `030-0891-003`.
    pub part: &'a str,
    pub revision: &'a str,
    pub name: &'a str,
}

/// Field widths. The barcode sits at byte 1 followed by spaces; the part
/// number is split around page 0's check: nine characters at 21, three at 35.
const BARCODE_AT: usize = 1;
const BARCODE_LEN: usize = 10;
const PART_LEN: usize = 12;
const PART_BASE_AT: usize = 21;
const PART_BASE_LEN: usize = 9;
const PART_DASH_AT: usize = 35;
const REVISION_AT: usize = 38;
const REVISION_LEN: usize = 4;
const FLAGS_AT: usize = 42;
const FLAGS_LEN: usize = 6;
const NAME_AT: usize = 48;
const NAME_LEN: usize = 14;
const PAGE: usize = 32;

/// A text field: non-empty, printable ASCII without `:` or `;` (the
/// readers substitute those), at most `width` characters.
fn check_field(what: &str, value: &str, width: usize) -> Result<(), String> {
    if value.is_empty() || value.len() > width {
        return Err(format!("board record {what} is required and must fit {width} characters"));
    }
    if value.bytes().any(|c| !(0x20..=0x7e).contains(&c) || c == b':' || c == b';') {
        return Err(format!("board record {what} must be printable, without ':' or ';'"));
    }
    Ok(())
}

impl BoardRecord<'_> {
    /// Whether the fields fit the record: the part number is twelve digits
    /// and dashes (the readers reject anything else), the barcode at most
    /// ten characters, the revision four, the name fourteen.
    pub fn check(&self) -> Result<(), String> {
        if self.part.len() != PART_LEN || !self.part.bytes().all(|c| c.is_ascii_digit() || c == b'-') {
            return Err(format!("board record part must be {PART_LEN} digits and dashes"));
        }
        check_field("barcode", self.barcode, BARCODE_LEN)?;
        check_field("revision", self.revision, REVISION_LEN)?;
        check_field("name", self.name, NAME_LEN)
    }

    /// The two pages, each closed by an inverted CRC-16 so a check over the
    /// whole page leaves the residue 0xb001. Panics if `check` fails: the
    /// records are machine constants and configuration is checked earlier.
    pub fn image(&self) -> [u8; 64] {
        self.check().expect("invalid board record");
        let mut rom = [b' '; 64];
        rom[0] = 0x01;
        rom[BARCODE_AT..BARCODE_AT + self.barcode.len()].copy_from_slice(self.barcode.as_bytes());
        rom[PART_BASE_AT..PART_BASE_AT + PART_BASE_LEN].copy_from_slice(&self.part.as_bytes()[..PART_BASE_LEN]);
        rom[PART_DASH_AT..PART_DASH_AT + PART_LEN - PART_BASE_LEN]
            .copy_from_slice(&self.part.as_bytes()[PART_BASE_LEN..]);
        rom[REVISION_AT..REVISION_AT + self.revision.len()].copy_from_slice(self.revision.as_bytes());
        rom[FLAGS_AT..FLAGS_AT + FLAGS_LEN].fill(0xff);
        rom[NAME_AT..NAME_AT + self.name.len()].copy_from_slice(self.name.as_bytes());
        for page in 0..2 {
            let at = &mut rom[page * PAGE..(page + 1) * PAGE];
            let c = !crc16_seed(0, &at[..PAGE - 2]);
            at[PAGE - 2..].copy_from_slice(&c.to_le_bytes());
        }
        rom
    }
}

/// A board record on a DS2505 (family 0x0b) with the 48-bit `serial`.
pub fn board_record_part(record: &BoardRecord, serial: u64) -> w1::Ds2505 {
    w1::Ds2505::new(0x0b, serial, &record.image())
}

/// The Ethernet-address part: a DS2501 (family 0x91) whose page 0 holds the
/// marker 0x0a, the station address reversed at bytes 5..=10 and an
/// inverted CRC-16 over bytes 0..=10 at 11..=12.
pub fn ethernet_address_part(mac: [u8; 6], serial: u64) -> w1::Ds2502 {
    let mut page = [0u8; PAGE];
    page[0] = 0x0a;
    for i in 0..6 {
        page[10 - i] = mac[i];
    }
    let c = !crc16_seed(0, &page[..11]);
    page[11..13].copy_from_slice(&c.to_le_bytes());
    w1::Ds2502::ds2501(0x91, serial, &page)
}

/// The O2's Ethernet-address part: a DS2502 (family 0x09, the one its PROM
/// insists on) whose serial is the station address itself, in order, and
/// whose memory starts with the address reversed, the rest erased.
pub fn o2_ethernet_address_part(mac: [u8; 6]) -> w1::Ds2502 {
    let mut rev = mac;
    rev.reverse();
    let mut serial = [0u8; 8];
    serial[..6].copy_from_slice(&mac);
    w1::Ds2502::new(0x09, u64::from_le_bytes(serial), &rev)
}

/// The synthetic ROM serial of the Ethernet-address part: the UniqueWare
/// prefix 0x5e7 in its top twelve bits (what OpenBSD matches on), then the
/// "EMU" marker and instance 0.
pub const ETHERNET_PART_SERIAL: u64 = 0x5e74_54d5_5000;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::w1::crc::crc8;
    use crate::dev::w1::slot;
    use crate::dev::w1::{Bus, Part};

    /// The 1-wire protocol, the way Linux's sgi_w1.c drives the register:
    /// `pulse << 10 | sample << 2`.
    struct Master(Bus);
    impl Master {
        fn reset(&mut self) -> bool {
            let (p, s) = slot::fields(520 << 10 | 65 << 2);
            !slot::slot(&mut self.0, p, s)
        }
        fn bit(&mut self, b: bool) -> bool {
            let (p, s) = slot::fields(if b { 6 << 10 | 13 << 2 } else { 80 << 10 | 30 << 2 });
            slot::slot(&mut self.0, p, s)
        }
        fn byte(&mut self, b: u8) -> u8 {
            (0..8).fold(0, |acc, i| acc | (self.bit(b >> i & 1 != 0) as u8) << i)
        }
        /// The standard search: returns every ROM on the wire.
        fn search(&mut self) -> Vec<[u8; 8]> {
            let mut roms = Vec::new();
            let mut last_disc: i32 = -1;
            let mut rom = [0u8; 8];
            loop {
                assert!(self.reset());
                self.byte(0xf0);
                let mut disc: i32 = -1;
                for i in 0..64 {
                    let b = self.bit(true);
                    let nb = self.bit(true);
                    let choose = match (b, nb) {
                        (true, true) => panic!("no part answered bit {i}"),
                        (false, false) => {
                            // A discrepancy: take 0 first, then 1 on the next pass.
                            if (i as i32) == last_disc {
                                true
                            } else if (i as i32) > last_disc {
                                disc = i as i32;
                                false
                            } else {
                                rom[i / 8] >> (i % 8) & 1 != 0
                            }
                        }
                        (b, _) => b,
                    };
                    if choose { rom[i / 8] |= 1 << (i % 8) } else { rom[i / 8] &= !(1 << (i % 8)) }
                    self.bit(choose);
                }
                roms.push(rom);
                if disc < 0 {
                    return roms;
                }
                last_disc = disc;
            }
        }
    }

    fn bus_of(parts: Vec<Box<dyn Part>>) -> Master {
        let mut b = Bus::new();
        for p in parts {
            b.add(p);
        }
        Master(b)
    }

    fn rec<'a>(barcode: &'a str, part: &'a str, revision: &'a str, name: &'a str) -> BoardRecord<'a> {
        BoardRecord { barcode, part, revision, name }
    }

    #[test]
    fn the_mac_nic_answers_read_rom_and_read_memory() {
        let mac = [0x08, 0x00, 0x69, 0x12, 0x34, 0x56];
        let mut m = bus_of(vec![Box::new(ethernet_address_part(mac, ETHERNET_PART_SERIAL))]);
        assert!(m.reset(), "presence pulse");
        m.byte(0x33);
        let rom: Vec<u8> = (0..8).map(|_| m.byte(0xff)).collect();
        assert_eq!(rom[0], 0x91);
        assert_eq!(rom[7], crc8(&rom[..7]));
        // Linux's ioc3_get_eaddr_nic: match ROM, read memory from 0.
        assert!(m.reset());
        m.byte(0x55);
        for &b in &rom {
            m.byte(b);
        }
        m.byte(0xf0);
        m.byte(0);
        m.byte(0);
        let mut nic = [0u8; 14];
        for i in (0..14).rev() {
            nic[i] = m.byte(0xff);
        }
        assert_eq!(nic[13], crc8(&[0xf0, 0, 0]));
        assert_eq!(&nic[2..8], &mac);
        // The page's CRC16, inverted and low byte first, closes it.
        let page: Vec<u8> = (2..=12).rev().map(|i| nic[i]).collect();
        assert_eq!(!crc16_seed(0, &page), u16::from_le_bytes([nic[1], nic[0]]));
    }

    #[test]
    fn search_rom_finds_every_part_on_the_wire() {
        let mut m = bus_of(vec![
            Box::new(ethernet_address_part([1, 2, 3, 4, 5, 6], ETHERNET_PART_SERIAL)),
            Box::new(board_record_part(&rec("EMUF00", "030-0891-003", "A", "FP1"), 0x1234)),
            Box::new(board_record_part(&rec("EMUS00", "060-0028-003", "A", "PWR.SPPLY.SR"), 0x5678)),
        ]);
        let roms = m.search();
        let mut fams: Vec<u8> = roms.iter().map(|r| r[0]).collect();
        fams.sort();
        assert_eq!(fams, vec![0x0b, 0x0b, 0x91]);
        for r in &roms {
            assert_eq!(r[7], crc8(&r[..7]));
        }
        // No part on the wire: no presence.
        assert!(!bus_of(vec![]).reset());
    }

    #[test]
    fn a_board_nic_reads_as_openbsd_and_linux_expect() {
        let mut m = bus_of(vec![Box::new(board_record_part(&rec("EMUN00", "030-0887-003", "F", "IP30"), 0x1234))]);
        // owserial: the redirection table from the status memory at 0x100,
        // eight bytes and a CRC16 at a time.
        assert!(m.reset());
        m.byte(0xcc);
        m.byte(0xaa);
        m.byte(0x00);
        m.byte(0x01);
        let mut redir = Vec::new();
        for g in 0..8 {
            let group: Vec<u8> = (0..8).map(|_| m.byte(0xff)).collect();
            let crc = [m.byte(0xff), m.byte(0xff)];
            // The CRC16 covers the command and address with the first
            // group, the eight bytes alone after; with its inverted CRC
            // appended the check value is 0xb001.
            let mut covered = if g == 0 { vec![0xaa, 0x00, 0x01] } else { Vec::new() };
            covered.extend_from_slice(&group);
            covered.extend_from_slice(&crc);
            assert_eq!(crc16_seed(0, &covered), 0xb001, "group {g}");
            redir.extend_from_slice(&group);
        }
        assert!(redir.iter().all(|&b| b == 0xff), "no page redirected");
        // Then pages 0 and 1, with no CRC8 in front. (Read Memory on this
        // part now ends in one CRC16 after the last byte of its 2 KB; the
        // old model stopped at the end of the data. Only that trailing
        // CRC16 is new, and it comes after everything read here.)
        assert!(m.reset());
        m.byte(0xcc);
        m.byte(0xf0);
        m.byte(0);
        m.byte(0);
        let buf: Vec<u8> = (0..64).map(|_| m.byte(0xff)).collect();
        assert_eq!(buf[0], 0x01);
        assert_eq!(&buf[1..7], b"EMUN00");
        assert_eq!(&buf[21..30], b"030-0887-");
        assert_eq!(&buf[32 + 3..32 + 6], b"003");
        assert!(buf[32 + 10..32 + 16].iter().all(|&b| b == 0xff));
        assert_eq!(&buf[32 + 16..32 + 20], b"IP30");
        // Linux: crc16 over each 32-byte page comes out as 0xb001.
        assert_eq!(crc16_seed(0, &buf[..32]), 0xb001);
        assert_eq!(crc16_seed(0, &buf[32..64]), 0xb001);
        // The trailing CRC16 follows the 2048th byte, covering the header.
        let rest: Vec<u8> = (0..(2048 - 64) + 2).map(|_| m.byte(0xff)).collect();
        let mut mem = buf.clone();
        mem.extend_from_slice(&rest[..2048 - 64]);
        assert_eq!(&rest[2048 - 64..], &(!crc16_seed(crc16_seed(0, &[0xf0, 0, 0]), &mem)).to_le_bytes());
    }

    #[test]
    fn the_ethernet_address_part_is_a_ds2501_with_64_programmable_bytes() {
        let p = ethernet_address_part([8, 0, 0x69, 1, 2, 3], 0x5e7_454d55_000);
        assert_eq!(p.programmable(), 64);
        assert_eq!(p.rom()[0], 0x91);
        assert_eq!((p.memory()[0], p.memory()[5], p.memory()[10]), (0x0a, 3, 8), "marker, station address reversed");
        assert_eq!(ethernet_address_part([8, 0, 0x69, 1, 2, 3], ETHERNET_PART_SERIAL).rom()[6], 0x5e, "UniqueWare prefix");
    }

    #[test]
    fn a_board_record_is_validated_like_the_readers_expect() {
        let ok = rec("EMUF00", "030-0891-003", "A", "FP1");
        assert!(ok.check().is_ok());
        let img = ok.image();
        assert_eq!(&img[1..7], b"EMUF00");
        assert_eq!(&img[7..21], &[b' '; 14]);
        assert_eq!(&img[21..30], b"030-0891-");
        assert_eq!(&img[35..38], b"003");
        assert_eq!(&img[38..39], b"A");
        assert_eq!(&img[42..48], &[0xff; 6]);
        assert_eq!(&img[48..51], b"FP1");
        assert_eq!(crc16_seed(0, &img[..32]), 0xb001);
        assert_eq!(crc16_seed(0, &img[32..]), 0xb001);
        for bad in [
            rec("EMUF00", "030-0891-00", "A", "FP1"),
            rec("EMUF00", "030-0891-00x", "A", "FP1"),
            rec("", "030-0891-003", "A", "FP1"),
            rec("ABCDEFGHIJK", "030-0891-003", "A", "FP1"),
            rec("EMUF00", "030-0891-003", "ABCDE", "FP1"),
            rec("EMUF00", "030-0891-003", "A", "0123456789abcde"),
            rec("EMUF00", "030-0891-003", "A", "A:B"),
            rec("EMUF00", "030-0891-003", "A", "A;B"),
            rec("EMUF00", "030-0891-003", "A", "caf\u{e9}"),
            rec("EMU\n00", "030-0891-003", "A", "FP1"),
        ] {
            assert!(bad.check().is_err(), "{:?}", (bad.barcode, bad.part, bad.revision, bad.name));
        }
    }
}
