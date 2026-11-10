//! Identity, plain-storage and map RAM registers, and the textport's mailbox
//! protocol. Ported from tests/qtest/sn0-origin2000-kona-test.c.

use super::*;
use crate::dev::w1::crc::crc8;
use crate::dev::w1::sgi::BoardRecord;

const WIDGET_ID: u64 = 0x04;
const WIDGET_CONTROL: u64 = 0x24;
const PROBE: u64 = 0x64;
const ONEWIRE: u64 = 0x98;
const NIC: u64 = 0x9c;
const LOW_WINDOW: u64 = 0x7c;
const CREDIT: u64 = 0x800208;
const CREDIT_CONFIG: u64 = 0x800210;
const RESET_ARG: u64 = 0x800220;
const INT_ENABLE: u64 = 0x800230;
const INT_CAUSE: u64 = 0x800238;
const MAILBOX: u64 = 0x800240;
const RESET_STROBE: u64 = 0x800300;
const EEPROM_ENABLE: u64 = 0x800308;
const PORT_ADDR: u64 = 0x8003e8;
const PORT_DATA: u64 = 0x8003f0;
const DMA_FIFO_PUSH: u64 = 0x80007c;
const MAPRAM: u64 = 0x100000;
const MAPRAM_ENTRIES: u64 = 0x40000;
const CAUSE_MAILBOX: u32 = 0x10;

/// The 1-wire protocol the way Linux's sgi_w1.c drives the register:
/// `pulse << 10 | sample << 2`, one slot per register write, the sampled wire
/// in bit 0 of the register once DONE (bit 1) is set.
pub(super) struct Wire<'a> {
    f: &'a Fixture,
    reg: u64,
}

impl<'a> Wire<'a> {
    pub fn new(f: &'a Fixture, reg: u64) -> Self {
        Wire { f, reg }
    }

    fn slot(&self, pulse: u32, sample: u32) -> bool {
        self.f.w(self.reg, pulse << 10 | sample << 2);
        let r = self.f.r(self.reg);
        assert_eq!(r & 2, 2, "the slot completes in the write that started it");
        r & 1 != 0
    }

    /// A reset: true if a part answers presence (the wire reads low).
    pub fn reset(&self) -> bool {
        !self.slot(520, 65)
    }

    fn bit(&self, b: bool) -> bool {
        if b { self.slot(6, 13) } else { self.slot(80, 30) }
    }

    pub fn byte(&self, b: u8) -> u8 {
        (0..8).fold(0, |acc, i| acc | (self.bit(b >> i & 1 != 0) as u8) << i)
    }

    /// Match ROM on `rom`: true if a part answered the reset.
    pub fn match_rom(&self, rom: &[u8; 8]) -> bool {
        let present = self.reset();
        self.byte(0x55);
        for b in rom {
            self.byte(*b);
        }
        present
    }

    /// Read Memory from `addr`, `len` bytes after the address CRC.
    pub fn read_memory(&self, addr: u8, len: usize) -> (u8, Vec<u8>) {
        self.byte(0xf0);
        self.byte(addr);
        self.byte(0);
        let crc = self.byte(0xff);
        (crc, (0..len).map(|_| self.byte(0xff)).collect())
    }

    /// The standard search: every ROM on the wire.
    pub fn search(&self) -> Vec<[u8; 8]> {
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

/// The ROM of the geometry board's part: family 0x09, the synthetic serial
/// marker 'G' 0, then the CRC-8 of the first seven bytes.
pub(super) fn geometry_rom() -> [u8; 8] {
    let mut rom = [0x09, 0x00, 0x47, 0x55, 0x4d, 0x45, 0x51, 0];
    rom[7] = crc8(&rom[..7]);
    rom
}

fn second_rom() -> [u8; 8] {
    let mut rom = [0x09, 0x01, 0x47, 0x55, 0x4d, 0x45, 0x51, 0];
    rom[7] = crc8(&rom[..7]);
    rom
}

/// The widget id register reads the presence
/// bit, the manufacturer, the part and the revision; the aperture ends at
/// 16 MiB; the registers are word-sized.
#[test]
fn identity_and_aperture() {
    let f = Fixture::new();
    let id = f.r(WIDGET_ID);
    assert_eq!(id & 1, 1);
    assert_eq!((id >> 1) & 0x7ff, 0x2aa);
    assert_eq!((id >> 12) & 0xffff, 0xc102);
    assert_eq!(id >> 28, 0);
    assert_eq!(f.kona.bus_read(APERTURE, 4).0, BUS_ERR);
    assert_eq!(f.kona.bus_write(APERTURE, 4, 0), BUS_ERR);
    assert_eq!(f.kona.bus_read(WIDGET_ID, 2).0, BUS_ERR);
    assert_eq!(f.kona.bus_read(WIDGET_ID, 1).0, BUS_ERR);
    assert_eq!(f.kona.bus_write(WIDGET_CONTROL, 1, 0), BUS_ERR);
    // The last word of the aperture is inside it.
    assert_eq!(f.kona.bus_read(APERTURE - 4, 4).0, BUS_OK);
}

/// Widget control is plain storage.
#[test]
fn widget_control_is_plain_storage() {
    let f = Fixture::new();
    assert_eq!(f.r(WIDGET_CONTROL), 9 << 22, "the test board's host widget");
    f.w(WIDGET_CONTROL, 0x0003_f00f);
    assert_eq!(f.r(WIDGET_CONTROL), 0x0003_f00f);
}

/// Both the board-type check and
/// the driver's presence check require the field to read 0, and its bits
/// 5:3 give the map RAM its largest size.
#[test]
fn probe_reads_documented_reset_state() {
    let f = Fixture::new();
    assert_eq!(f.r(PROBE), 0);
    f.w(PROBE, 0x3f);
    assert_eq!(f.r(PROBE), 0, "the probe register is read-only");
}

/// An offset no register
/// owns reads zero and a write to it does not persist.
#[test]
fn unknown_offset_reads_zero_and_accepts_writes() {
    let f = Fixture::new();
    f.w(0x6c, 0x1234);
    assert_eq!(f.r(0x6c), 0);
    f.w(0x800390, 0);
    assert_eq!(f.r(0x800390), 0);
}

/// The register reads zero at rest,
/// the value both the PROM's (exactly zero) and the driver's (not above zero)
/// waits proceed on.
#[test]
fn credit_register_reads_zero_at_rest() {
    let f = Fixture::new();
    assert_eq!(f.r(CREDIT), 0);
}

/// The register is a fill level: the words
/// the command-port framers hold of a packet still arriving, rounded up to two
/// (bit 0 never sets). A kernel 0x1f1 with its token and one of three words
/// holds 2, a textport command with its selector and two of four words holds 3
/// and reads 4, a DMA packet with its selector and three of six words on the
/// push register holds 4. Each drops to zero once its last word arrives. A
/// model that read a fixed zero fails the first mid-packet read. A single
/// held word reads 2: the level rounds up to two and saturates at 0x7fe
/// (`a_full_ring_answers_busy_never_spins` checks a full ring reads 0x7fe).
#[test]
fn credit_follows_consumption() {
    let f = Fixture::new();
    f.cmd(&[0x1f1]);
    assert_eq!(f.r(CREDIT), 2, "one held word rounds up to two");
    f.cmd(&[0x1000_0000]);
    assert_eq!(f.r(CREDIT), 2);
    f.cmd(&[0, 0x4000]);
    assert_eq!(f.r(CREDIT), 0);

    f.cmd(&[1, 3, 7, 1]);
    assert_eq!(f.r(CREDIT), 4);
    f.cmd(&[2, 3]);
    assert_eq!(f.r(CREDIT), 0);

    for w in [0x1eb, 1, 0x9000, 4] {
        f.w(DMA_FIFO_PUSH, w);
    }
    assert_eq!(f.r(CREDIT), 4);
    assert_eq!(f.r(CREDIT) & 1, 0, "bit 0 is never set");
}

/// The driver's one write of 0x700 is kept and
/// read back and changes nothing the credit register says.
#[test]
fn credit_config_is_stored_and_changes_nothing() {
    let f = Fixture::new();
    assert_eq!(f.r(CREDIT_CONFIG), 0);
    f.w(CREDIT_CONFIG, 0x700);
    assert_eq!(f.r(CREDIT_CONFIG), 0x700);
    assert_eq!(f.r(CREDIT), 0);
    f.cmd(&[0x1f1, 0x1000_0000]);
    assert_eq!(f.r(CREDIT), 2);
}

/// Low window write is plain storage.
#[test]
fn low_window_write_is_plain_storage() {
    let f = Fixture::new();
    f.w(LOW_WINDOW, 0x23);
    assert_eq!(f.r(LOW_WINDOW), 0x23);
}

/// Reset and eeprom control registers are plain storage.
#[test]
fn reset_and_eeprom_control_registers_are_plain_storage() {
    let f = Fixture::new();
    for (reg, v) in [(RESET_ARG, 0x8000), (EEPROM_ENABLE, 1), (RESET_STROBE, 2)] {
        f.w(reg, v);
        assert_eq!(f.r(reg), v);
    }
    // Software never reads the enable register: it holds the bits written,
    // within the status register's twelve.
    f.w(INT_ENABLE, 0xf7c);
    assert_eq!(f.r(INT_ENABLE), 0xf7c);
    f.w(INT_ENABLE, 0xffff_ffff);
    assert_eq!(f.r(INT_ENABLE), 0xfff);
}

/// The PROM and
/// the driver read the flash header's state word through the indirect port;
/// every other address in the backed range reads zero.
#[test]
fn eeprom_data_answers_checked_address_and_zero_elsewhere() {
    let f = Fixture::new();
    f.w(PORT_ADDR, 0x87ff1c);
    assert_eq!(f.r(PORT_DATA), 2);
    f.w(PORT_ADDR, 0x87ff00);
    assert_eq!(f.r(PORT_DATA), 0);
    f.w(PORT_ADDR, 0x87ff20);
    assert_eq!(f.r(PORT_DATA), 0);
}

/// "1,9,0" on the wire: the
/// mailbox's single reply word drains in one step with nothing pending behind
/// it, then the cause bit stays clear.
#[test]
fn textport_sync_probe_mailbox_protocol() {
    let f = Fixture::new();
    f.cmd(&[1, 9, 0]);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, CAUSE_MAILBOX);
    assert_eq!(f.r(MAILBOX), 0x1234abcd);
    f.w(MAILBOX, 0);
    f.w(INT_CAUSE, CAUSE_MAILBOX);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0);
    assert_eq!(f.r(MAILBOX), 0);
    // The textport is in step for the next command.
    f.cmd(&[1, 9, 0]);
    assert_eq!(f.r(MAILBOX), 0x1234abcd);
}

/// "1,0xc,0" replies five
/// words, width and height (established) at indices 3 and 4. The cause bit
/// re-arms synchronously behind each of the first four drains and clears only
/// after the fifth.
#[test]
fn textport_graphics_info_mailbox_protocol() {
    let f = Fixture::new();
    f.cmd(&[1, 0xc, 0]);
    let mut got = Vec::new();
    for i in 0..5 {
        assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, CAUSE_MAILBOX, "word {i} is ready");
        got.push(f.r(MAILBOX));
        f.w(MAILBOX, 0);
        f.w(INT_CAUSE, CAUSE_MAILBOX);
    }
    assert_eq!(got, [0, 0, 0, 1280, 1024]);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0);
}

/// A selector outside the
/// textport's table is dropped and the port expects a sync word again, so a
/// probe completing right after shows it stayed in step.
#[test]
fn textport_unknown_selector_resynchronizes() {
    let f = Fixture::new();
    f.cmd(&[1, 0x77]);
    f.cmd(&[1, 9, 0]);
    assert_eq!(f.r(MAILBOX), 0x1234abcd);
}

/// A command argument
/// equal to the bitmap selector (0xa) mid-packet is not a fresh blit start:
/// command 3's own r/g/b arguments routinely carry it.
#[test]
fn bitmap_value_mid_packet_does_not_start_a_blit() {
    let f = Fixture::hooked();
    f.cmd(&[1, 3, 0xa, 0xa, 0xa, 0xa]);
    assert_eq!(f.jobs(), ["Textport 0x3 [a, a, a, a] packed=0x0 handled=false"]);
    f.cmd(&[1, 9, 0]);
    assert_eq!(f.r(MAILBOX), 0x1234abcd);
}

/// , up to the seam: six header
/// words and a packed 1-bpp payload of ceil(width * height / 32) words reach
/// the back end whole, and the port is in step afterwards.
#[test]
fn textport_bitmap_blit_reaches_the_back_end() {
    let f = Fixture::hooked();
    // 4x2 bits: one payload word.
    f.cmd(&[1, 0xa, 4, 2, 0, 0, 5, 0, 0xa500_0000]);
    assert_eq!(f.jobs(), ["bitmap [4, 2, 0, 0, 5, 0] [a5000000]"]);
    // 33x1 bits: two payload words.
    f.take_jobs();
    f.cmd(&[1, 0xa, 33, 1, 0, 0, 0, 0, 0xffff_ffff, 0x8000_0000]);
    assert_eq!(f.jobs(), ["bitmap [21, 1, 0, 0, 0, 0] [ffffffff, 80000000]"]);
    f.cmd(&[1, 9, 0]);
    assert_eq!(f.r(MAILBOX), 0x1234abcd);
}

/// A blit of no bits takes no payload.
#[test]
fn an_empty_bitmap_blit_takes_no_payload() {
    let f = Fixture::hooked();
    f.cmd(&[1, 0xa, 0, 7, 0, 0, 0, 0]);
    f.cmd(&[1, 9, 0]);
    assert_eq!(f.r(MAILBOX), 0x1234abcd);
    assert!(f.jobs().is_empty());
}

/// Command 8 is accepted and
/// handed to the back end; the size graphics info reports stays fixed.
#[test]
fn textport_screen_size_is_a_job_only() {
    let f = Fixture::hooked();
    f.cmd(&[1, 8, 640, 480]);
    assert_eq!(f.jobs(), ["Textport 0x8 [280, 1e0] packed=0x0 handled=false"]);
    f.cmd(&[1, 0xc, 0]);
    for _ in 0..3 {
        f.w(MAILBOX, 0);
        f.w(INT_CAUSE, CAUSE_MAILBOX);
    }
    assert_eq!(f.r(MAILBOX), 1280);
}

/// Selectors 1, 4
/// and 7 are framed and handed on; the front end answers none of them.
#[test]
fn reserved_textport_selectors_are_framed_and_handed_on() {
    let f = Fixture::hooked();
    f.cmd(&[1, 1, 0, 1, 4, 100, 200, 1, 7, 1, 2]);
    assert_eq!(f.jobs().len(), 3);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, 0);
}

/// The driver's map RAM probe: write
/// 0x10000000 | i, read back, expect an exact match, over a sample plus the
/// first and last entry; entry 1 is re-checked at the end so a stride bug
/// cannot hide behind the entry written last.
#[test]
fn mapram_stride_sample() {
    let f = Fixture::new();
    let sample = [0, 1, 2, 3, 7, 255, 256, 4095, 4096, 65535, 65536, MAPRAM_ENTRIES - 1];
    for i in sample {
        f.w(MAPRAM + 4 * i, 0x1000_0000 | i as u32);
        assert_eq!(f.r(MAPRAM + 4 * i), 0x1000_0000 | i as u32, "entry {i}");
    }
    assert_eq!(f.r(MAPRAM + 4), 0x1000_0001);
}

/// An entry's physical width is unknown;
/// the model stores all of it.
#[test]
fn mapram_stores_full_32_bits() {
    let f = Fixture::new();
    f.w(MAPRAM + 8, 0xffff_ffff);
    assert_eq!(f.r(MAPRAM + 8), 0xffff_ffff);
    f.w(MAPRAM + 8, 0x2000_0000 | (0x1234 << 2));
    assert_eq!(f.r(MAPRAM + 8), 0x2000_48d0);
}

/// The first offset past
/// the window reads zero and keeps no write; the table's last entry takes one.
#[test]
fn mapram_bounds_fall_through_to_unimplemented() {
    let f = Fixture::new();
    f.w(MAPRAM + 4 * MAPRAM_ENTRIES, 0x55);
    assert_eq!(f.r(MAPRAM + 4 * MAPRAM_ENTRIES), 0);
    f.w(MAPRAM + 4 * (MAPRAM_ENTRIES - 1), 0x77);
    assert_eq!(f.r(MAPRAM + 4 * (MAPRAM_ENTRIES - 1)), 0x77);
    assert_eq!(f.r(MAPRAM - 4), 0);
}

/// Mapram reset clears table.
#[test]
fn mapram_reset_clears_the_table() {
    let f = Fixture::new();
    f.w(MAPRAM, 0x1234);
    f.w(MAPRAM + 4 * 100, 0x5678);
    f.kona.power_on();
    assert_eq!(f.r(MAPRAM), 0);
    assert_eq!(f.r(MAPRAM + 4 * 100), 0);
}

/// A nonzero size code in the
/// probe register's bits 5:3 shrinks the active table: N = 1 selects 1 << 15
/// entries. The last entry below the count stores; the first at the count
/// reads zero and keeps no write.
#[test]
fn mapram_size_code_limits_entry_count() {
    let host = TestHost::new();
    let kona = Kona::new(KonaConfig { probe: 1 << 3, ..test_cfg() }, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
    let f = Fixture::of(host, kona);
    assert_eq!(f.r(PROBE), 1 << 3);
    f.w(MAPRAM + 4 * ((1 << 15) - 1), 0xabcd);
    assert_eq!(f.r(MAPRAM + 4 * ((1 << 15) - 1)), 0xabcd);
    f.w(MAPRAM + 4 * (1 << 15), 0xabcd);
    assert_eq!(f.r(MAPRAM + 4 * (1 << 15)), 0);
}

/// Mapram ordinary mapping write read back.
#[test]
fn mapram_ordinary_mapping_round_trips() {
    let f = Fixture::new();
    let paddr: u64 = 0x0123_4000;
    let entry = 0x2000_0000 | ((paddr >> 14) << 2) as u32;
    f.w(MAPRAM + 4 * 9, entry);
    assert_eq!(f.r(MAPRAM + 4 * 9), entry);
}

/// The driver's attach-time probe over the
/// whole table: write 0x10000000 | i to every entry, then read every entry
/// back and expect an exact match.
#[test]
fn mapram_full_window_sweep() {
    let f = Fixture::new();
    for i in 0..MAPRAM_ENTRIES {
        f.w(MAPRAM + 4 * i, 0x1000_0000 | i as u32);
    }
    for i in 0..MAPRAM_ENTRIES {
        assert_eq!(f.r(MAPRAM + 4 * i), 0x1000_0000 | i as u32, "entry {i}");
    }
}

fn geometry_image() -> [u8; 64] {
    BoardRecord { barcode: "EMUG00", part: "030-1129-002", revision: "F", name: "GE14-4" }.image()
}

/// , with the deskside's record. Match ROM on
/// the geometry board's part, then Read Memory returns its barcode, part
/// number, revision and name.
#[test]
fn onewire_geometry_identity() {
    let f = Fixture::new();
    let w = Wire::new(&f, ONEWIRE);
    assert!(w.match_rom(&geometry_rom()), "the part answers presence");
    let (crc, mem) = w.read_memory(0, 64);
    assert_eq!(crc, crc8(&[0xf0, 0, 0]));
    assert_eq!(&mem[..], &geometry_image()[..]);
    let text = |a: usize, n: usize| String::from_utf8(mem[a..a + n].to_vec()).unwrap();
    assert_eq!(text(1, 6), "EMUG00");
    assert_eq!(text(21, 9), "030-1129-");
    assert_eq!(text(35, 3), "002");
    assert_eq!(text(38, 1), "F");
    assert_eq!(text(48, 6), "GE14-4");
}

/// A deskside pipe has no second
/// host-interface record: Match ROM against its identity selects no part, and
/// a search finds the geometry board's record alone.
#[test]
fn onewire_has_no_second_record() {
    let f = Fixture::new();
    let w = Wire::new(&f, ONEWIRE);
    w.match_rom(&second_rom());
    let (_, mem) = w.read_memory(0, 8);
    assert_eq!(mem, [0xff; 8], "no part answers a Read Memory after a Match ROM nothing matched");
    assert_eq!(w.search(), vec![geometry_rom()]);
}

/// +0x9c is the kernel
/// driver's offset onto the identical chain: a slot issued through one offset
/// reads back through the other with no further pulse, and the same records
/// answer through either.
#[test]
fn nic_register_mirrors_the_onewire_register() {
    let f = Fixture::new();
    f.w(ONEWIRE, 520 << 10 | 65 << 2);
    assert_eq!(f.r(NIC), f.r(ONEWIRE));
    assert_eq!(f.r(NIC) & 3, 2, "done, wire low: presence");
    f.w(NIC, 6 << 10 | 13 << 2);
    assert_eq!(f.r(ONEWIRE), f.r(NIC));
    let w = Wire::new(&f, NIC);
    assert!(w.match_rom(&geometry_rom()));
    assert_eq!(w.read_memory(0, 64).1, geometry_image());
}

/// Mutated widget control and a one-wire
/// transaction left mid-slot return to zero on reset.
#[test]
fn reset_clears_plain_storage() {
    let f = Fixture::hooked();
    f.w(WIDGET_CONTROL, 0xffff);
    f.w(ONEWIRE, 520 << 10 | 65 << 2);
    f.w(LOW_WINDOW, 0x23);
    f.w(RESET_ARG, 0x8000);
    f.w(0x34, 0x98000);
    f.w(0x3c, 0x1800090);
    f.w(0x5c, 0x3e2210);
    f.kona.power_on();
    for reg in [WIDGET_CONTROL, ONEWIRE, LOW_WINDOW, RESET_ARG, 0x34, 0x3c, 0x5c, INT_CAUSE, MAILBOX] {
        // Widget control resets to the host widget (9 on a test board).
        assert_eq!(f.r(reg), if reg == WIDGET_CONTROL { 9 << 22 } else { 0 }, "register {reg:#x}");
    }
    // The board's identity survives.
    assert!(Wire::new(&f, ONEWIRE).match_rom(&geometry_rom()));
}

/// The kernel driver's
/// own initialisation: strobes the reset register (1, read, 0, read), then
/// writes 0x100 to the enable and 0x8000 to the reset argument. The strobe
/// reads back what was written.
#[test]
fn the_drivers_initialisation_sequence_is_accepted() {
    let f = Fixture::new();
    f.w(RESET_STROBE, 1);
    assert_eq!(f.r(RESET_STROBE), 1);
    f.w(RESET_STROBE, 0);
    assert_eq!(f.r(RESET_STROBE), 0);
    f.w(INT_ENABLE, 0x100);
    f.w(RESET_ARG, 0x8000);
    assert_eq!(f.r(RESET_ARG), 0x8000);
}

/// The widget a pipe's DMA goes to is a field of the widget control register
/// (bits 25:22), which host software fills with its interrupt target.
#[test]
fn the_dma_widget_is_a_field_of_widget_control() {
    let f = Fixture::new();
    assert_eq!(f.kona.dma_widget(), 9);
    f.w(WIDGET_CONTROL, 3 << 22 | 0xb);
    assert_eq!(f.kona.dma_widget(), 3);
}

/// The widget control register resets to the host widget in bits 25:22, so a
/// pipe's DMA has a destination before software programs it (the model's
/// choice; the power-on value is unknown).
#[test]
fn widget_control_resets_to_the_host_widget() {
    let host = TestHost::new();
    let cfg = KonaConfig::default().with_host_widget(9).unwrap();
    let kona = Kona::new(cfg, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
    let f = Fixture::of(host, kona);
    assert_eq!(f.r(WIDGET_CONTROL), 9 << 22);
    assert_eq!(f.kona.dma_widget(), 9);
    f.w(WIDGET_CONTROL, 0x3 << 22);
    assert_eq!(f.kona.dma_widget(), 3);
    f.kona.power_on();
    assert_eq!(f.r(WIDGET_CONTROL), 9 << 22);
}
