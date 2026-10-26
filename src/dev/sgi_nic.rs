//! SGI's "Number In a Can": the 1-wire master register that HEART (its
//! MicroLAN control), the Bridge (its NIC register) and the IOC3 (MCR) each
//! have, and the Dallas add-only memory parts hanging off them that carry
//! a board's part number, name and serial, or the Ethernet address.
//!
//! The master is one register. Software writes the time it wants the line
//! held low (bits 19:10, microseconds) and when to sample it (bits 9:2),
//! then reads back DONE (bit 1) and the line as sampled (bit 0); Linux's
//! `drivers/w1/masters/sgi_w1.c` packs it as `pulse << 10 | sample << 2`.
//! A low time of 480 µs or more is a reset, under 15 µs a 1 (or a read
//! slot, which is the same thing on the wire), anything between a 0. No
//! timing is modelled: the slot completes in the write.
//!
//! Several parts share a wire, as on the Octane (the IOC3's carries the
//! MAC part, the front plane's and the power supply's); the line is the
//! AND of what each drives, which is what makes Search ROM work.

pub const MCR_RD_DATA: u32 = 1 << 0;
pub const MCR_DONE: u32 = 1 << 1;

/// One master and the parts on its wire.
#[derive(Default)]
pub struct OneWire {
    last: u32,
    line: bool,
    devs: Vec<Ds2502>,
}

impl OneWire {
    pub fn with(dev: Ds2502) -> Self {
        Self::with_all(vec![dev])
    }

    pub fn with_all(devs: Vec<Ds2502>) -> Self {
        OneWire { devs, ..Default::default() }
    }

    /// Software drives one slot.
    pub fn write(&mut self, val: u32) {
        self.last = val;
        let pulse = (val >> 10) & 0x3ff;
        if pulse >= 480 {
            // A reset: the line reads low while any part answers with its
            // presence pulse.
            let mut present = false;
            for d in &mut self.devs {
                present |= d.reset();
            }
            self.line = !present;
        } else {
            let master = pulse < 15;
            let mut line = master;
            for d in &mut self.devs {
                line &= d.slot(master);
            }
            self.line = line;
        }
    }

    /// The slot is always done; the line is as the parts left it.
    pub fn read(&self) -> u32 {
        (self.last & !(MCR_RD_DATA | MCR_DONE)) | MCR_DONE | self.line as u32
    }
}

/// Dallas's 1-wire CRC-8 (x^8 + x^5 + x^4 + 1).
pub fn crc8(bytes: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in bytes {
        let mut b = b;
        for _ in 0..8 {
            let mix = (crc ^ b) & 1;
            crc >>= 1;
            if mix != 0 {
                crc ^= 0x8c;
            }
            b >>= 1;
        }
    }
    crc
}

/// Dallas's CRC-16 (x^16 + x^15 + x^2 + 1), as the parts append it,
/// inverted, to their pages: over the page and its two CRC bytes it comes
/// out as 0xb001.
pub fn crc16(bytes: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &b in bytes {
        crc ^= b as u16;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { crc >> 1 ^ 0xa001 } else { crc >> 1 };
        }
    }
    crc
}

/// What a part is doing between slots.
enum WireState {
    Idle,
    /// Taking `n` bits, least significant first (`i` so far), then acting
    /// on them.
    Recv { n: u32, i: u32, got: u64, then: Next },
    /// Sending `bits` from `i` on; after them, `then`.
    Send { bits: Vec<bool>, i: usize, then: Next },
    /// Search ROM at ROM bit `bit`: phase 0 sends it, 1 its complement,
    /// 2 takes the master's choice.
    Search { bit: u32, phase: u8 },
}

#[derive(Clone, Copy, Debug)]
enum Next {
    RomCmd,
    MatchRom,
    FuncCmd,
    ReadAddr(u8),
    Idle,
}

/// A DS2502-family add-only memory on the wire: the DS1981U that holds
/// the Ethernet address, the DS2505 that identifies a board. Answers the
/// ROM commands (read 0x33, match 0x55, skip 0xcc, search 0xf0), Read
/// Memory (0xf0) and Read Status (0xaa).
pub struct Ds2502 {
    rom: [u8; 8],
    mem: Vec<u8>,
    /// The status memory: the page redirection table and the rest, as
    /// Read Status returns it.
    status: Vec<u8>,
    /// Read Status returns a CRC16 after every eight bytes on the larger
    /// parts; the DS1981U returns none.
    status_crc16: bool,
    state: WireState,
}

fn bits_of(bytes: &[u8]) -> Vec<bool> {
    bytes.iter().flat_map(|&b| (0..8).map(move |i| b >> i & 1 != 0)).collect()
}

fn trace() -> bool {
    std::env::var_os("IRIS_IP30_TRACE").is_some()
}

/// A 32-byte page: `body` (up to 30 bytes, padded with `pad`) and its
/// inverted CRC16.
fn page(body: &[u8], pad: u8) -> [u8; 32] {
    let mut p = [pad; 32];
    p[..body.len().min(30)].copy_from_slice(&body[..body.len().min(30)]);
    let crc = !crc16(&p[..30]);
    p[30] = crc as u8;
    p[31] = (crc >> 8) as u8;
    p
}

impl Ds2502 {
    pub fn new(family: u8, serial: [u8; 6], mem: Vec<u8>, status: Vec<u8>, status_crc16: bool) -> Self {
        let mut rom = [0u8; 8];
        rom[0] = family;
        rom[1..7].copy_from_slice(&serial);
        rom[7] = crc8(&rom[..7]);
        Ds2502 { rom, mem, status, status_crc16, state: WireState::Idle }
    }

    /// The part's family code.
    pub fn family(&self) -> u8 {
        self.rom[0]
    }

    /// The MAC NIC on the IOC3: a DS1981U (family 0x91). The PROM, like
    /// Linux's ioc3-eth.c, reads 14 bytes after Read Memory at 0, the
    /// first of them the CRC8 of the command, into nic[13..0], and takes
    /// the station address from nic[2..8]. That puts it reversed at memory
    /// 5..11, the end of a TMEX page: a length byte (10), the data, and the
    /// inverted CRC16 of both (OpenBSD's owmac.c reads the same bytes).
    pub fn mac_nic(mac: [u8; 6]) -> Self {
        let mut mem = vec![0xffu8; 128];
        mem[0] = 10;
        mem[1..5].fill(0);
        for j in 0..6 {
            mem[10 - j] = mac[j];
        }
        let crc = !crc16(&mem[..11]);
        mem[11] = crc as u8;
        mem[12] = (crc >> 8) as u8;
        // A UniqueWare 0x91 part's serial starts 0x5e7 (OpenBSD's owmac.c
        // matches on it); the rest is from the MAC.
        Ds2502::new(0x91, [mac[5], mac[4], mac[3], mac[2], 0x70 | (mac[1] & 0x0f), 0x5e], mem, vec![0xff; 8], false)
    }

    /// A board's identification NIC: a DS2505 (family 0x0b, 2 KB in 64
    /// pages) laid out as OpenBSD's owserial.c reads it, which is what
    /// `hinv` shows as "IP30 Board: barcode FGT687 part 030-0887-003 rev G":
    /// page 0 is 0x01, the barcode (20 characters), the first nine of the
    /// part number; page 1 the last three, the revision, six 0xff bytes
    /// and the name (14 characters); each page ends in its inverted CRC16.
    /// The status memory's redirection table (0x100..0x13f) says no page
    /// is redirected.
    pub fn board_nic(serial_no: u32, name: &str, part: &str, rev: &str, barcode: &str) -> Self {
        let mut mem = vec![0xffu8; 2048];
        let part = format!("{part:<12}");
        let mut p0 = vec![0x01u8];
        p0.extend_from_slice(format!("{barcode:<20}").as_bytes());
        p0.extend_from_slice(&part.as_bytes()[..9]);
        mem[..32].copy_from_slice(&page(&p0, b' '));
        let mut p1 = vec![b' '; 3];
        p1.extend_from_slice(&part.as_bytes()[9..12]);
        p1.extend_from_slice(format!("{rev:<4}").as_bytes());
        p1.extend_from_slice(&[0xff; 6]);
        p1.extend_from_slice(format!("{name:<14}").as_bytes());
        mem[32..64].copy_from_slice(&page(&p1, b' '));
        let s = serial_no.to_le_bytes();
        Ds2502::new(0x0b, [s[0], s[1], s[2], s[3], 0, 0], mem, vec![0xff; 0x200], true)
    }

    fn rom_bit(&self, bit: u32) -> bool {
        self.rom[(bit / 8) as usize] >> (bit % 8) & 1 != 0
    }

    /// A reset: present, and waiting for a ROM command.
    fn reset(&mut self) -> bool {
        if trace() {
            let sent = match &self.state { WireState::Send { i, .. } => *i as i64, _ => -1 };
            eprintln!("ip30: 1-wire {:02x}: reset (after {} bits sent)", self.rom[0], sent);
        }
        self.state = WireState::Recv { n: 8, i: 0, got: 0, then: Next::RomCmd };
        true
    }

    fn recv(n: u32, then: Next) -> WireState {
        WireState::Recv { n, i: 0, got: 0, then }
    }

    /// What Read Status streams from `addr`. The 1 Kbit parts (the DS1981U)
    /// send the CRC8 of the command and address, then the status bytes.
    /// The DS2505 sends the status bytes straight away, an inverted CRC16
    /// after every eight of them: over the command, address and bytes for
    /// the first group, over the bytes alone after (what the PROM's
    /// nic_read_mfg checks, and OpenBSD's owserial skips).
    fn status_stream(&self, addr: u16) -> Vec<u8> {
        let hdr = [0xaa, addr as u8, (addr >> 8) as u8];
        let data = &self.status[(addr as usize).min(self.status.len())..];
        if !self.status_crc16 {
            let mut out = vec![crc8(&hdr)];
            out.extend_from_slice(data);
            return out;
        }
        let mut out = Vec::new();
        let mut first = true;
        for group in data.chunks(8) {
            out.extend_from_slice(group);
            let mut covered = Vec::new();
            if first {
                covered.extend_from_slice(&hdr);
                first = false;
            }
            covered.extend_from_slice(group);
            let crc = !crc16(&covered);
            out.push(crc as u8);
            out.push((crc >> 8) as u8);
        }
        out
    }

    /// What follows `n` bits arriving for `then`.
    fn act(&mut self, then: Next, got: u64) -> WireState {
        if trace() {
            eprintln!("ip30: 1-wire {:02x}: {:?} {:#x}", self.rom[0], then, got);
        }
        match then {
            Next::RomCmd => match got as u8 {
                0x33 => WireState::Send { bits: bits_of(&self.rom), i: 0, then: Next::FuncCmd },
                0x55 => Self::recv(64, Next::MatchRom),
                0xcc => Self::recv(8, Next::FuncCmd),
                0xf0 => WireState::Search { bit: 0, phase: 0 },
                _ => WireState::Idle,
            },
            Next::MatchRom if got == u64::from_le_bytes(self.rom) => Self::recv(8, Next::FuncCmd),
            Next::MatchRom => WireState::Idle,
            Next::FuncCmd => match got as u8 {
                c @ (0xf0 | 0xaa) => Self::recv(16, Next::ReadAddr(c)),
                _ => WireState::Idle,
            },
            Next::ReadAddr(cmd) => {
                let addr = got as u16;
                let out = if cmd == 0xf0 {
                    // Read Memory: the 1 Kbit parts lead with the CRC8 of
                    // the command and address; the DS2505 does not
                    // (owserial reads its pages straight after the address).
                    let mut out = if self.status_crc16 { Vec::new() } else { vec![crc8(&[cmd, addr as u8, (addr >> 8) as u8])] };
                    out.extend(self.mem.iter().skip(addr as usize));
                    out
                } else {
                    self.status_stream(addr)
                };
                WireState::Send { bits: bits_of(&out), i: 0, then: Next::Idle }
            }
            Next::Idle => WireState::Idle,
        }
    }

    /// One slot: `master` is what the master drove (1 also for a read
    /// slot); returns what this part leaves the line at.
    fn slot(&mut self, master: bool) -> bool {
        let state = std::mem::replace(&mut self.state, WireState::Idle);
        let (line, next) = match state {
            WireState::Idle => (master, WireState::Idle),
            WireState::Recv { n, i, got, then } => {
                let got = got | (master as u64) << i;
                if i + 1 == n {
                    (master, self.act(then, got))
                } else {
                    (master, WireState::Recv { n, i: i + 1, got, then })
                }
            }
            WireState::Send { bits, i, then } => {
                // Past the end the part has let go: the line floats high.
                let b = bits.get(i).copied().unwrap_or(true);
                let next = if i + 1 >= bits.len() {
                    match then {
                        Next::FuncCmd => Self::recv(8, Next::FuncCmd),
                        _ => WireState::Send { bits, i: i + 1, then },
                    }
                } else {
                    WireState::Send { bits, i: i + 1, then }
                };
                (master && b, next)
            }
            WireState::Search { bit, phase } => {
                let b = self.rom_bit(bit);
                match phase {
                    0 => (master && b, WireState::Search { bit, phase: 1 }),
                    1 => (master && !b, WireState::Search { bit, phase: 2 }),
                    // The master chose the other branch: out until the next reset.
                    _ if master != b => (master, WireState::Idle),
                    _ if bit == 63 => (master, Self::recv(8, Next::FuncCmd)),
                    _ => (master, WireState::Search { bit: bit + 1, phase: 0 }),
                }
            }
        };
        self.state = next;
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 1-wire protocol, the way sgi_w1.c drives it.
    struct Master<'a>(&'a mut OneWire);
    impl Master<'_> {
        fn reset(&mut self) -> bool {
            self.0.write(520 << 10 | 65 << 2);
            self.0.read() & MCR_RD_DATA == 0
        }
        fn bit(&mut self, b: bool) -> bool {
            self.0.write(if b { 6 << 10 | 13 << 2 } else { 80 << 10 | 30 << 2 });
            self.0.read() & MCR_RD_DATA != 0
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
                    if !choose && (b, nb) == (false, false) && disc != i as i32 {
                        // kept from a previous pass
                    }
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

    #[test]
    fn the_mac_nic_answers_read_rom_and_read_memory() {
        let mac = [0x08, 0x00, 0x69, 0x0b, 0x7e, 0xba];
        let mut w = OneWire::with(Ds2502::mac_nic(mac));
        let mut m = Master(&mut w);
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
        assert_eq!(!crc16(&page), u16::from_le_bytes([nic[1], nic[0]]));
    }

    #[test]
    fn search_rom_finds_every_part_on_the_wire() {
        let mut w = OneWire::with_all(vec![
            Ds2502::mac_nic([1, 2, 3, 4, 5, 6]),
            Ds2502::board_nic(0x1234, "FP1", "030-0891-003", "A", "EDJ162"),
            Ds2502::board_nic(0x5678, "PWR.SPPLY.ER", "060-0035-003", "A", "AAE3230158"),
        ]);
        let roms = Master(&mut w).search();
        let mut fams: Vec<u8> = roms.iter().map(|r| r[0]).collect();
        fams.sort();
        assert_eq!(fams, vec![0x0b, 0x0b, 0x91]);
        for r in &roms {
            assert_eq!(r[7], crc8(&r[..7]));
        }
        // No part on the wire: no presence.
        let mut empty = OneWire::default();
        assert!(!Master(&mut empty).reset());
    }

    #[test]
    fn a_board_nic_reads_as_openbsd_and_linux_expect() {
        let mut w = OneWire::with(Ds2502::board_nic(0x1234, "IP30", "030-0887-003", "G", "FGT687"));
        let mut m = Master(&mut w);
        // owserial: the redirection table from the status memory at 0x100,
        // eight bytes and a CRC16 at a time.
        assert!(m.reset());
        m.byte(0xcc);
        m.byte(0xaa);
        m.byte(0x00);
        m.byte(0x01);
        let mut redir = Vec::new();
        for g in 0..8 {
            let mut group: Vec<u8> = (0..8).map(|_| m.byte(0xff)).collect();
            let crc = [m.byte(0xff), m.byte(0xff)];
            // The CRC16 covers the command and address with the first
            // group, the eight bytes alone after; with its inverted CRC
            // appended the check value is 0xb001.
            let mut covered = if g == 0 { vec![0xaa, 0x00, 0x01] } else { Vec::new() };
            covered.append(&mut group.clone());
            covered.extend_from_slice(&crc);
            assert_eq!(crc16(&covered), 0xb001, "group {g}");
            redir.append(&mut group);
        }
        assert!(redir.iter().all(|&b| b == 0xff), "no page redirected");
        // Then pages 0 and 1, with no CRC8 in front.
        assert!(m.reset());
        m.byte(0xcc);
        m.byte(0xf0);
        m.byte(0);
        m.byte(0);
        let buf: Vec<u8> = (0..64).map(|_| m.byte(0xff)).collect();
        assert_eq!(buf[0], 0x01);
        assert_eq!(&buf[1..7], b"FGT687");
        assert_eq!(&buf[21..30], b"030-0887-");
        assert_eq!(&buf[32 + 3..32 + 6], b"003");
        assert!(buf[32 + 10..32 + 16].iter().all(|&b| b == 0xff));
        assert_eq!(&buf[32 + 16..32 + 20], b"IP30");
        // Linux: crc16 over each 32-byte page comes out as 0xb001.
        assert_eq!(crc16(&buf[..32]), 0xb001);
        assert_eq!(crc16(&buf[32..64]), 0xb001);
    }
}
