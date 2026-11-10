//! The serial EEPROM behind the NVRAM register, read the way the drivers
//! read it.
//!
//! Ported from tests/qtest/qlogic-isp2200-test.c in
//! jamesbraid/qemu-sgi-origami.

use super::*;

impl Rig {
    /// One word of the NVRAM, clocked the way the drivers clock it.
    fn nvram_read_word(&self, word: u16) -> u16 {
        let cmd = (6 << 8) | word;
        let mut val = 0;
        self.wr(reg::NVRAM, 0x0002);
        for bit in 0..11 {
            let out = if cmd & (1 << (10 - bit)) != 0 { 0x0004 } else { 0 };
            self.wr(reg::NVRAM, 0x0002 | out);
            self.wr(reg::NVRAM, 0x0003 | out);
            self.wr(reg::NVRAM, 0x0002 | out);
        }
        for _ in 0..16 {
            self.wr(reg::NVRAM, 0x0003);
            val = (val << 1) | u16::from(self.rd(reg::NVRAM) & 0x0008 != 0);
            self.wr(reg::NVRAM, 0x0002);
        }
        self.wr(reg::NVRAM, 0);
        val
    }

    fn nvram_read_image(&self) -> [u8; 256] {
        let mut img = [0u8; 256];
        for w in 0..128 {
            img[w * 2..w * 2 + 2].copy_from_slice(&self.nvram_read_word(w as u16).to_le_bytes());
        }
        img
    }

    /// The clocked pieces of an NVRAM read, for cases that stop part way.
    fn nvram_bit(&self, bit: bool) {
        let out = if bit { 0x0004 } else { 0 };
        self.wr(reg::NVRAM, 0x0002 | out);
        self.wr(reg::NVRAM, 0x0003 | out);
        self.wr(reg::NVRAM, 0x0002 | out);
    }

    fn nvram_command(&self, cmd: u16) {
        self.wr(reg::NVRAM, 0x0002);
        for bit in 0..11 {
            self.nvram_bit(cmd & (1 << (10 - bit)) != 0);
        }
    }

    fn nvram_data_bit(&self) -> u16 {
        self.wr(reg::NVRAM, 0x0003);
        let bit = u16::from(self.rd(reg::NVRAM) & 0x0008 != 0);
        self.wr(reg::NVRAM, 0x0002);
        bit
    }
}

/// Approximation: the default image is a model choice, not a capture. It has
/// to pass the checks the shipped drivers make, read the way they read: 256
/// bytes summing to zero, an eight-byte port name at byte 18 with QLogic's
/// company ID in name bytes 2 to 4, a first name byte of 0x20 or 0x21, and
/// serial bytes that are not all ones. A model that returns one word for
/// every address, or an all-zero image, fails.
#[test]
fn nvram_default_image() {
    let r = Rig::bare();
    assert_eq!(r.nvram_read_word(0), 0x5349);
    assert_eq!(r.nvram_read_word(1), 0x2050);
    let img = r.nvram_read_image();
    assert_eq!(img.iter().fold(0u8, |sum, b| sum.wrapping_add(*b)), 0);
    assert_eq!(&img[..4], b"ISP ");
    assert!(img[4] >= 1);
    assert!(img[18] == 0x20 || img[18] == 0x21);
    assert_eq!(img[20..23], [0x00, 0xe0, 0x8b]);
    assert_ne!(img[19], 0xff);
    assert!(img[23..26] != [0xff; 3]);
    assert!(img[23..26] != [0; 3]);
    // Words 9 and 127 are the port name's first pair and the checksum.
    assert_eq!(r.nvram_read_word(9), u16::from_le_bytes([img[18], img[19]]));
    assert_eq!(r.nvram_read_word(127), u16::from_le_bytes([img[254], img[255]]));
}

/// Approximation: the serial bytes of the port name come from the adapter's
/// serial, and without one from its index, so two adapters in one machine
/// differ and one adapter keeps its name from run to run. Both sides of that
/// are pinned, with the image still summing to zero.
#[test]
fn nvram_serial() {
    let r = Rig::on(Isp2200 { core: Arc::new(Core::new_at(0, Isp2200::new_bus(), 0x123456)) });
    let img = r.nvram_read_image();
    assert_eq!(img, r.nvram_read_image());
    assert_eq!(img[23..26], [0x12, 0x34, 0x56]);
    // The node name carries the same serial.
    assert_eq!(img[32..38], img[20..26]);
    assert_eq!(img[30], 0x20);
    assert_eq!(img.iter().fold(0u8, |sum, b| sum.wrapping_add(*b)), 0);
}

/// Approximation: the part is modelled for reads only, and a word address of
/// 128 or more, past the 256-byte image, reads 0xffff. A command whose opcode
/// is not read is ignored, drives nothing and leaves the image as it was.
/// Clocking past the word's sixteenth bit reads zero, not the next word. A
/// zero before the start bit is skipped. Deselecting part way through a word
/// forgets it, so the next read starts clean.
#[test]
fn nvram_edges() {
    let r = Rig::bare();
    for word in [128, 255] {
        assert_eq!(r.nvram_read_word(word), 0xffff);
    }

    // Write opcode, then more clocks than a word has: nothing comes out.
    r.nvram_command(0x500 | 1);
    for _ in 0..20 {
        assert_eq!(r.nvram_data_bit(), 0);
    }
    r.wr(reg::NVRAM, 0);
    assert_eq!(r.nvram_read_word(1), 0x2050);

    // Past the sixteenth bit of a read.
    r.nvram_command(0x600);
    for _ in 0..16 {
        r.nvram_data_bit();
    }
    for _ in 0..20 {
        assert_eq!(r.nvram_data_bit(), 0);
    }
    r.wr(reg::NVRAM, 0);

    // Deselected after the command and five data bits.
    r.nvram_command(0x600 | 1);
    for _ in 0..5 {
        r.nvram_data_bit();
    }
    r.wr(reg::NVRAM, 0);
    assert_eq!(r.nvram_read_word(1), 0x2050);

    // Deselected after seven command bits.
    r.wr(reg::NVRAM, 0x0002);
    for _ in 0..7 {
        r.nvram_bit(true);
    }
    r.wr(reg::NVRAM, 0);
    assert_eq!(r.nvram_read_word(0), 0x5349);

    // Two zero bits before the start bit change nothing.
    r.wr(reg::NVRAM, 0x0002);
    r.nvram_bit(false);
    r.nvram_bit(false);
    for i in 0..11 {
        r.nvram_bit((0x600 | 1) & (1 << (10 - i)) != 0);
    }
    let mut v = 0;
    for _ in 0..16 {
        v = (v << 1) | r.nvram_data_bit();
    }
    assert_eq!(v, 0x2050);
    r.wr(reg::NVRAM, 0);
}

/// Without a serial the port name's is FNV-1a over the adapter's name,
/// modulo 0xfffffe, plus one. A second adapter in the same machine therefore
/// has another name, the same adapter keeps its name across a reset, and the
/// value is pinned to the algorithm so a constant, or a different hash,
/// fails.
#[test]
fn nvram_default_serial() {
    fn fnv1a_serial(name: &str) -> u32 {
        name.bytes().fold(0x811c_9dc5u32, |hash, byte| (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)) % 0xff_fffe + 1
    }
    fn serial(r: &Rig) -> u32 {
        let (w11, w12) = (r.nvram_read_word(11), r.nvram_read_word(12));
        (u32::from(w11 >> 8) << 16) | (u32::from(w12 & 0xff) << 8) | u32::from(w12 >> 8)
    }
    let first = Rig::on(Isp2200::new_at(0, Isp2200::new_bus()));
    let second = Rig::on(Isp2200::new_at(1, Isp2200::new_bus()));
    assert_eq!(serial(&first), fnv1a_serial("isp2200-0"));
    assert_eq!(serial(&second), fnv1a_serial("isp2200-1"));
    assert_ne!(serial(&first), serial(&second));
    let before = serial(&first);
    first.soft_reset();
    assert_eq!(serial(&first), before);
}
