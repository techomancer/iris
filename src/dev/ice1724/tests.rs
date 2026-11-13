//! ICE1724 audio register-contract tests.
//!
//! Ported from tests/qtest/sgi-dm8-test.c in jamesbraid/qemu-sgi-origami.
//! Register values are spelled from the register contract, not taken from
//! the model's constants. The reset values of the GPIO data registers
//! expect the model's weak pull-up on input pins, which is the model's
//! reading of "an input bit reads the pin", not an established value.
//!
//! The host presents bytes in register order, so these tests read and
//! write the register file as the function sees it: byte offsets, wider
//! accesses as byte steps. Configuration writes take the path the bus gives
//! them ([`cfg_dw`], [`cfg_wb`], [`cfg_ww`]), and `on_a_bus` places the
//! card on a real bus to check that the bus agrees.

use super::*;
use crate::dev::host_pcm::test_time::{FakeNs, TestTime};
use crate::dev::host_pcm::PcmOut;
use crate::host_audio::{FrameResampler, HostOut};
use crate::irq::LevelProbe;
use crate::pci::{PciDma, PciDmaStatus, PCI_COMMAND_IO};

const CCS_BAR: u32 = 0x10_0000;
const MT_BAR: u32 = 0x10_0080;
/// Where the playback buffer sits in the test memory.
const BUFFER: u64 = 0x1000;

struct Ram(std::sync::Mutex<Vec<u8>>);

impl PciDma for Ram {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        let m = self.0.lock().unwrap();
        match m.get(addr as usize..addr as usize + buf.len()) {
            Some(src) => {
                buf.copy_from_slice(src);
                PciDmaStatus::Ok
            }
            None => PciDmaStatus::MasterAbort,
        }
    }
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        let mut m = self.0.lock().unwrap();
        match m.get_mut(addr as usize..addr as usize + data.len()) {
            Some(dst) => {
                dst.copy_from_slice(data);
                PciDmaStatus::Ok
            }
            None => PciDmaStatus::MasterAbort,
        }
    }
}

/// A card on a fake clock with both BARs placed, I/O decode and bus
/// mastering on, its interrupt line probed and memory behind it.
struct Rig {
    c: Ice1724,
    irq: Arc<LevelProbe>,
    ram: Arc<Ram>,
}

impl Deref for Rig {
    type Target = Ice1724;
    fn deref(&self) -> &Ice1724 {
        &self.c
    }
}

fn rig(variant: Variant) -> Rig {
    let r = bare(variant);
    r.c.state().connect_dma(r.ram.clone());
    r
}

/// As [`rig`], with no way to memory: every transfer fails.
fn bare(variant: Variant) -> Rig {
    let c = Ice1724::with_clock(variant, TestTime::clock());
    c.fake_ns().store(0, Ordering::Relaxed);
    let (irq, line) = LevelProbe::line();
    c.state().connect_irq(line);
    c.attached();
    let r = Rig { c, irq, ram: Arc::new(Ram(std::sync::Mutex::new(vec![0; 0x10_0000]))) };
    map(&r.c);
    r
}

fn map(c: &Ice1724) {
    cfg_dw(c, 4, CCS_BAR | 1);
    cfg_dw(c, 5, MT_BAR | 1);
    cfg_dw(c, 1, 5);
}

/// A configuration write of `width` bytes at byte `off`, as
/// [`crate::pci::PciBus::config_write`] delivers it: a whole dword is
/// offered to the function first, and any write goes to the header and, if
/// it changed the dword, is reported to the function.
fn cfg_write(c: &Ice1724, off: usize, width: usize, val: u32) {
    if width == 4 && c.config_write(off / 4, val) {
        return;
    }
    let (cfg, reg, shift) = (c.state().config(), off / 4, (off & 3) * 8);
    let mask = (u32::MAX >> (32 - 8 * width)) << shift;
    let old = cfg.read(reg);
    cfg.write_masked(reg, val << shift, mask);
    let new = cfg.read(reg);
    if new != old {
        c.config_written(reg, old, new);
    }
}

fn cfg_dw(c: &Ice1724, reg: usize, val: u32) {
    cfg_write(c, reg * 4, 4, val);
}

fn cfg_wb(c: &Ice1724, off: usize, v: u8) {
    cfg_write(c, off, 1, u32::from(v));
}

fn cfg_ww(c: &Ice1724, off: usize, v: u16) {
    cfg_write(c, off, 2, u32::from(v));
}

fn cfg_b(c: &Ice1724, off: usize) -> u8 {
    (c.config_dword(off / 4) >> (8 * (off % 4))) as u8
}

fn ccs_rb(c: &Ice1724, off: u32) -> u8 { c.bar_read(0, off, 1) as u8 }
fn ccs_rw(c: &Ice1724, off: u32) -> u16 { c.bar_read(0, off, 2) as u16 }
fn ccs_rl(c: &Ice1724, off: u32) -> u32 { c.bar_read(0, off, 4) }
fn ccs_wb(c: &Ice1724, off: u32, v: u8) { c.bar_write(0, off, u32::from(v), 1) }
fn ccs_ww(c: &Ice1724, off: u32, v: u16) { c.bar_write(0, off, u32::from(v), 2) }
fn mt_rb(c: &Ice1724, off: u32) -> u8 { c.bar_read(1, off, 1) as u8 }
fn mt_rw(c: &Ice1724, off: u32) -> u16 { c.bar_read(1, off, 2) as u16 }
fn mt_rl(c: &Ice1724, off: u32) -> u32 { c.bar_read(1, off, 4) }
fn mt_wb(c: &Ice1724, off: u32, v: u8) { c.bar_write(1, off, u32::from(v), 1) }
fn mt_ww(c: &Ice1724, off: u32, v: u16) { c.bar_write(1, off, u32::from(v), 2) }
fn mt_wl(c: &Ice1724, off: u32, v: u32) { c.bar_write(1, off, v, 4) }

/// Let the fake clock run `ns` and bring the engines up to date, as the
/// timer would.
fn step(c: &Ice1724, ns: i64) {
    c.fake_ns().fetch_add(ns, Ordering::Relaxed);
    let _ = c.stats();
}

fn seen(c: &Ice1724, what: &str) -> bool {
    c.seen.lock().iter().any(|m| m.contains(what))
}

// ---- EEPROM and converter helpers ----

fn eeprom_byte(c: &Ice1724, index: u8) -> u8 {
    ccs_wb(c, CCS_I2C_BYTE, index);
    ccs_wb(c, CCS_I2C_DEVICE, 0xa0);
    assert_eq!(ccs_rb(c, CCS_I2C_STATUS) & 1, 0, "the transfer is not left busy");
    ccs_rb(c, CCS_I2C_DATA)
}

/// The sequence a driver runs: direction, mask, then data, from the EEPROM.
fn load_gpio_from_eeprom(c: &Ice1724) {
    let b = |i| u32::from(eeprom_byte(c, i));
    let (dir, mask, data) = (b(10) | b(11) << 8 | b(12) << 16, b(13) | b(14) << 8 | b(15) << 16, b(16) | b(17) << 8 | b(18) << 16);
    ccs_ww(c, 0x18, dir as u16);
    ccs_wb(c, 0x1a, (dir >> 16) as u8);
    ccs_ww(c, 0x16, mask as u16);
    ccs_wb(c, 0x1f, (mask >> 16) as u8);
    ccs_ww(c, 0x14, data as u16);
    ccs_wb(c, 0x1e, (data >> 16) as u8);
}

const IDLE: u8 = 0x72;
const CCLK: u8 = 0x02;
const CDOUT: u8 = 0x08;
const CS_FRONT: u8 = 0x20;
const CS_SURROUND: u8 = 0x40;

fn pins(c: &Ice1724, v: u8) {
    ccs_wb(c, 0x14, v);
}

/// One frame of `nbits` as the parts take it, most significant bit first.
fn frame_bits(c: &Ice1724, select: u8, word: u16, nbits: u32) {
    let low = IDLE & !select;
    pins(c, low);
    for i in (0..nbits).rev() {
        let data = if u32::from(word) >> i & 1 != 0 { CDOUT } else { 0 };
        pins(c, low & !CCLK);
        pins(c, low & !CCLK | data);
        pins(c, low | data);
    }
    pins(c, IDLE);
}

fn codec_write(c: &Ice1724, select: u8, chip: u16, reg: u16, v: u8) {
    frame_bits(c, select, chip << 14 | 0x2000 | reg << 8 | u16::from(v), 16);
}

// ---- identity and reset ----

#[test]
fn config_identity() {
    let c = rig(Variant::Dm8);
    assert_eq!(c.config_dword(0), 0x1724_1412);
    // Revision 1, class 0401, programming interface 0.
    assert_eq!(c.config_dword(2), 0x0401_0001);
    assert_eq!(c.config_dword(11), 0x3630_1412);
    // Medium DEVSEL, capability list present; no interrupt, no errors.
    assert_eq!(c.config_dword(1) >> 16, 0x0210);
    assert_eq!(cfg_b(&c, 0x0c), 0);
    assert_eq!(cfg_b(&c, 0x0e), 0);
    assert_eq!(c.config_dword(4), CCS_BAR | 1);
    assert_eq!(c.config_dword(5), MT_BAR | 1);
    assert_eq!(c.config_dword(12), 0);
    assert_eq!(cfg_b(&c, 0x34), 0x80);
    assert_eq!(cfg_b(&c, 0x3d), 1);
    assert_eq!(cfg_b(&c, 0x3c), 0xff);
    // Power management: version 1.0, D2 only, no PME; state D0.
    assert_eq!(c.config_dword(0x80 / 4), 0x0401_0001);
    assert_eq!(c.config_dword(0x84 / 4), 0);
}

#[test]
fn config_writes_obey_the_masks() {
    let c = Ice1724::with_clock(Variant::Dm8, TestTime::clock());
    // BAR sizes: 32 and 128 bytes of I/O space.
    cfg_dw(&c, 4, 0xffff_ffff);
    cfg_dw(&c, 5, 0xffff_ffff);
    assert_eq!(c.config_dword(4), 0xffff_ffe1);
    assert_eq!(c.config_dword(5), 0xffff_ff81);
    // BARs 2 and 3 are unimplemented: all-ones reads back 0 (PCI convention;
    // inferred for this chip).
    cfg_dw(&c, 6, 0xffff_ffff);
    cfg_dw(&c, 7, 0xffff_ffff);
    assert_eq!((c.config_dword(6), c.config_dword(7)), (0, 0));
    cfg_dw(&c, 4, 0);
    cfg_dw(&c, 5, 0);

    // Only I/O decode and bus mastering are writable in the command.
    cfg_dw(&c, 1, 0xffff);
    assert_eq!(c.config_dword(1) & 0xffff, 5);
    cfg_dw(&c, 1, 0);

    // The latency timer keeps bits 7:3, the cache line size is read only,
    // and the interrupt line is storage.
    cfg_wb(&c, 0x0d, 0xff);
    assert_eq!(cfg_b(&c, 0x0d), 0xf8);
    cfg_wb(&c, 0x0c, 0xff);
    assert_eq!(cfg_b(&c, 0x0c), 0);
    cfg_wb(&c, 0x3c, 0x05);
    assert_eq!(cfg_b(&c, 0x3c), 5);
    cfg_wb(&c, 0x3d, 0x03);
    assert_eq!(cfg_b(&c, 0x3d), 1, "the pin is wiring");

    // The subsystem pair is read only until PCI42 bit 7 is set.
    cfg_dw(&c, 11, 0x1234_5678);
    assert_eq!(c.config_dword(11), 0x3630_1412);
    cfg_wb(&c, 0x42, 0x80);
    cfg_dw(&c, 11, 0x1234_5678);
    assert_eq!(c.config_dword(11), 0x1234_5678);
    cfg_wb(&c, 0x42, 0);

    // Nothing else past the header is writable.
    cfg_dw(&c, 0x88 / 4, 0xffff_ffff);
    assert_eq!(c.config_dword(0x88 / 4), 0);
    cfg_dw(&c, 0, 0);
    assert_eq!(c.config_dword(0), 0x1724_1412);
}

#[test]
fn power_state_follows_the_capability() {
    let c = rig(Variant::Dm8);
    ccs_wb(&c, 0x04, 0x4f);
    assert_eq!(ccs_rb(&c, 0x04), 0x4f);

    // D1 is not supported: the write leaves D0.
    cfg_dw(&c, 0x84 / 4, 1);
    assert_eq!(c.config_dword(0x84 / 4) & 3, 0);
    assert_eq!(ccs_rb(&c, 0x04), 0x4f);
    // Outside D0 both BARs are closed and float high.
    cfg_dw(&c, 0x84 / 4, 3);
    assert_eq!(c.config_dword(0x84 / 4) & 3, 3);
    assert_eq!(ccs_rl(&c, 0x04), u32::MAX);
    ccs_wb(&c, 0x04, 0x00);
    // D3hot to D0 resets the chip's registers. Approximation: the chip also
    // resets configuration space, which this model leaves as firmware set
    // it (a function cannot reset its own header; see the module notes).
    cfg_dw(&c, 0x84 / 4, 0);
    assert_eq!(c.config_dword(0x84 / 4) & 3, 0);
    assert_eq!(ccs_rb(&c, 0x04), 0x0f, "the registers are reset");
    assert_eq!(c.config_dword(1) & 0xffff, 5, "the header is not");
    assert!(seen(&c, "D3hot to D0 resets the registers, not the configuration space"));
}

/// The bridge merges a guest's partial write against the stored header, so
/// a write to a byte beside the power state must leave the state alone, and
/// a halfword write at 0x84, as drivers make it, must change it.
#[test]
fn a_neighbouring_byte_does_not_change_the_power_state() {
    let c = rig(Variant::Dm8);
    ccs_wb(&c, 0x04, 0x4f);
    cfg_ww(&c, 0x84, 3);
    assert_eq!(c.config_dword(0x84 / 4) & 3, 3);
    cfg_wb(&c, 0x85, 0x00);
    cfg_wb(&c, 0x86, 0xff);
    cfg_wb(&c, 0x87, 0xff);
    assert_eq!(c.config_dword(0x84 / 4) & 3, 3, "still D3hot");
    assert_eq!(c.config_dword(1) & 0xffff, 5, "and not reset");
    cfg_wb(&c, 0x84, 2);
    assert_eq!(c.config_dword(0x84 / 4) & 3, 2);
    cfg_ww(&c, 0x84, 0);
    assert_eq!(c.config_dword(0x84 / 4) & 3, 0);
    assert_eq!(c.config_dword(1) & 0xffff, 5, "D2 to D0 does not reset the header");
}

#[test]
fn reset_values_and_soft_reset() {
    let c = rig(Variant::Dm8);
    // Dword views of the byte registers, little-endian.
    assert_eq!(ccs_rl(&c, 0x00), 0x0000_fe00); // 00 01 02 03
    assert_eq!(ccs_rl(&c, 0x04), 0x0101_000f); // 04 05 06 07
    assert_eq!(ccs_rl(&c, 0x08), 0);
    assert_eq!(ccs_rl(&c, 0x0c), 0);
    // The EEPROM strap reads back in CCS13 bit 7.
    assert_eq!(ccs_rl(&c, 0x10), 0x8000_0000);
    // GPIO data is the pull-up on every pin, mask all ones, direction 0.
    assert_eq!(ccs_rw(&c, 0x16), 0xffff);
    assert_eq!(ccs_rb(&c, 0x1f), 0xff);
    assert_eq!(ccs_rw(&c, 0x18), 0);
    assert_eq!(ccs_rb(&c, 0x1c), 0);
    // Multi-channel block.
    assert_eq!(mt_rb(&c, 0x00), 0);
    assert_eq!(mt_rb(&c, 0x01), 0);
    assert_eq!(mt_rb(&c, 0x02), 0);
    assert_eq!(mt_rb(&c, 0x03), 0xff);
    assert_eq!(mt_rb(&c, 0x18), 0);
    assert_eq!(mt_rb(&c, 0x19), 0);
    assert_eq!(mt_rb(&c, 0x1a), 0);
    assert_eq!(mt_rb(&c, 0x1b), 0);
    assert_eq!(mt_rl(&c, 0x10), 0);
    assert_eq!(mt_rl(&c, 0x2c), 0);

    // Write-masked bits: MT01 bits 7:5 read 0, CCS07 bit 6 is read only.
    mt_wb(&c, 0x01, 0xff);
    assert_eq!(mt_rb(&c, 0x01), 0x1f);
    ccs_wb(&c, 0x07, 0xff);
    assert_eq!(ccs_rb(&c, 0x07), 0xbf);
    ccs_wb(&c, 0x04, 0x43);
    ccs_wb(&c, 0x01, 0x00);
    ccs_ww(&c, 0x16, 0);

    // The soft reset keeps CCS04 to CCS07 and restores the rest.
    ccs_wb(&c, 0x00, 0x80);
    assert_eq!(ccs_rb(&c, 0x04), 0x43);
    assert_eq!(ccs_rb(&c, 0x07), 0xbf);
    assert_eq!(ccs_rb(&c, 0x01), 0xfe);
    assert_eq!(ccs_rw(&c, 0x16), 0xffff);
    assert_eq!(mt_rb(&c, 0x01), 0);

    // A system reset restores everything, configuration space too: the
    // bridge resets the header, then the machine resets the function.
    c.state().config().reset();
    c.reset();
    assert_eq!(c.config_dword(4), 1);
    assert_eq!(c.config_dword(1) & 0xffff, 0);
    assert_eq!(cfg_b(&c, 0x3c), 0xff, "the line is set after the bridge's reset");
    map(&c);
    assert_eq!(ccs_rl(&c, 0x04), 0x0101_000f);
    assert_eq!(ccs_rb(&c, 0x01), 0xfe);
}

#[test]
fn a_system_reset_stops_a_running_engine() {
    let c = rig(Variant::Dm8);
    ccs_wb(&c, 0x04, 0x43);
    mt_wb(&c, 0x01, 0x08);
    mt_wb(&c, 0x19, 3);
    mt_wb(&c, 0x03, 0xfe);
    mt_wb(&c, 0x18, 0x01);
    step(&c, 1_000_000);
    assert_eq!(c.stats().analog_frames, 44);

    c.state().config().reset();
    c.reset();
    map(&c);
    assert_eq!(ccs_rb(&c, 0x04), 0x0f);
    assert_eq!(mt_rb(&c, 0x01), 0);
    assert_eq!(mt_rb(&c, 0x03), 0xff);
    assert_eq!(mt_rb(&c, 0x18), 0);
    assert_eq!(cfg_b(&c, 0x3c), 0xff);
    step(&c, 1_000_000);
    assert_eq!(c.stats().analog_frames, 44, "a stopped engine moves nothing");
}

// ---- I2C EEPROM ----

#[test]
fn eeprom_reads_writes_and_misses() {
    let c = rig(Variant::Dm8);
    assert_eq!(ccs_rb(&c, 0x13) & 0x80, 0x80);
    // The subsystem pair, stored the way the public tables read it.
    for (i, v) in [0x36, 0x30, 0x14, 0x12].into_iter().enumerate() {
        assert_eq!(eeprom_byte(&c, i as u8), v);
    }
    // The configuration bytes give an I2S board with four stereo DACs.
    assert_eq!(eeprom_byte(&c, 6) & 3, 3);
    assert_eq!(eeprom_byte(&c, 7) & 0x80, 0x80);
    // Past the image the part is erased.
    assert_eq!(eeprom_byte(&c, 100), 0xff);

    // A write cycle sticks.
    ccs_wb(&c, 0x11, 100);
    ccs_wb(&c, 0x12, 0x5a);
    ccs_wb(&c, 0x10, 0xa1);
    assert_eq!(eeprom_byte(&c, 100), 0x5a);

    // Bit 7 and bit 0 of CCS13 are read only; bit 1 is storage.
    ccs_wb(&c, 0x13, 0x7f);
    assert_eq!(ccs_rb(&c, 0x13), 0x82);

    // A transfer to a device nothing answers leaves the data alone.
    ccs_wb(&c, 0x12, 0x77);
    ccs_wb(&c, 0x10, 0xa4);
    assert_eq!(ccs_rb(&c, 0x12), 0x77);
    assert!(seen(&c, "I2C read of device 0x52"));
}

// ---- the converters' write protocol ----

#[test]
fn converter_frames_follow_the_write_protocol() {
    let c = rig(Variant::Dm8);
    load_gpio_from_eeprom(&c);
    assert_eq!(c.ak4381()[3], 0xff);
    assert_eq!(c.ak4355()[9], 0xff);
    assert_eq!(c.ak4381()[0], 0);

    // Front DAC, chip address 1, select on GPIO 5.
    codec_write(&c, CS_FRONT, 1, 0, 0x0c);
    assert_eq!(c.ak4381()[0], 0x0c);
    codec_write(&c, CS_FRONT, 1, 0, 0x0f);
    codec_write(&c, CS_FRONT, 1, 3, 0x80);
    codec_write(&c, CS_FRONT, 1, 4, 0x40);
    assert_eq!(c.ak4381(), [0x0f, 0, 0, 0x80, 0x40]);
    assert_eq!(c.stats().ak4381_writes, 4);

    // Surround DAC, chip address 3, select on GPIO 6, eleven registers.
    codec_write(&c, CS_SURROUND, 3, 1, 0x01);
    codec_write(&c, CS_SURROUND, 3, 2, 0x2e);
    codec_write(&c, CS_SURROUND, 3, 10, 0x55);
    let ak = c.ak4355();
    assert_eq!((ak[1], ak[2], ak[10]), (0x01, 0x2e, 0x55));
    assert_eq!(c.stats().ak4355_writes, 3);
    // The two ports do not see each other's frames.
    assert_eq!(c.ak4381()[1], 0);
    assert_eq!(c.stats().codec_rejects, 0);

    // Frames the parts would not take are counted and dropped.
    codec_write(&c, CS_FRONT, 3, 0, 0x99); // wrong chip
    codec_write(&c, CS_SURROUND, 1, 1, 0x99); // wrong chip
    codec_write(&c, CS_FRONT, 1, 5, 0x99); // no register 5
    codec_write(&c, CS_SURROUND, 3, 11, 0x99); // no register 11
    frame_bits(&c, CS_FRONT, 0x600f, 15); // short frame
    frame_bits(&c, CS_FRONT, 0x600f, 17); // long frame
    frame_bits(&c, CS_FRONT, 0x400f, 16); // read command
    let st = c.stats();
    assert_eq!((st.codec_rejects, st.ak4381_writes, st.ak4355_writes), (7, 4, 3));
    assert_eq!(c.ak4381()[0], 0x0f);

    // Data changes while CCLK stays high do not clock a frame.
    pins(&c, IDLE & !CS_FRONT);
    for _ in 0..16 {
        pins(&c, IDLE & !CS_FRONT | CDOUT);
        pins(&c, IDLE & !CS_FRONT);
    }
    pins(&c, IDLE);
    let st = c.stats();
    assert_eq!((st.codec_rejects, st.ak4381_writes), (7, 4));
    assert_eq!(c.ak4381()[0], 0x0f);

    // A reset powers the converters back to their defaults.
    c.state().config().reset();
    c.reset();
    map(&c);
    assert_eq!(c.ak4381()[0], 0);
    assert_eq!(c.ak4381()[3], 0xff);
}

#[test]
fn the_gpio_write_mask_protects_outputs() {
    let c = rig(Variant::Dm8);
    // Masked at reset: a data write changes nothing but the pull-ups.
    ccs_ww(&c, 0x18, 0x00ff);
    ccs_ww(&c, 0x14, 0x0000);
    assert_eq!(ccs_rw(&c, 0x14), 0xff00);
    ccs_ww(&c, 0x16, 0xfff0); // open the low nibble
    ccs_ww(&c, 0x14, 0x0005);
    assert_eq!(ccs_rw(&c, 0x14), 0xff05);
    // An unmasked output reads back what was written.
    ccs_ww(&c, 0x14, 0x000a);
    assert_eq!(ccs_rw(&c, 0x14), 0xff0a);
    // The upper block has its own mask and direction.
    ccs_wb(&c, 0x1a, 0x40);
    ccs_wb(&c, 0x1f, 0xbf);
    ccs_wb(&c, 0x1e, 0x00);
    assert_eq!(ccs_rb(&c, 0x1e), 0x3f);
    ccs_wb(&c, 0x1e, 0x40);
    assert_eq!(ccs_rb(&c, 0x1e), 0x7f);
    assert_eq!(ccs_rb(&c, 0x1a), 0x40);
    // A masked pin that is an output keeps its value: bits 16 and 17 are
    // outputs here, protected by the mask, so a write to them does not stick.
    ccs_wb(&c, 0x1a, 0x43);
    ccs_wb(&c, 0x1e, 0x43);
    assert_eq!(ccs_rb(&c, 0x1e), 0x7c);
    // Bit 23 of the direction register is reserved.
    ccs_wb(&c, 0x1a, 0xff);
    assert_eq!(ccs_rb(&c, 0x1a), 0x7f);
}

#[test]
fn spdif_control_writes_are_limited_while_enabled() {
    let c = rig(Variant::Dm8);
    // MT3C is writable while the transmitter is disabled.
    mt_ww(&c, 0x3c, 0x1234);
    assert_eq!(mt_rw(&c, 0x3c), 0x1234);
    // Enabled: only the validity flag, bit 15, takes a write.
    ccs_wb(&c, 0x07, 0x81);
    mt_ww(&c, 0x3c, 0xffff);
    assert_eq!(mt_rw(&c, 0x3c), 0x9234);
    mt_ww(&c, 0x3c, 0x0000);
    assert_eq!(mt_rw(&c, 0x3c), 0x1234);
    // Disabled again: everything writable.
    ccs_wb(&c, 0x07, 0x01);
    mt_ww(&c, 0x3c, 0x0000);
    assert_eq!(mt_rw(&c, 0x3c), 0);
}

// ---- DMA playback, interrupts, record ----

/// Numeric samples in the order the card reads them from the bus.
fn fill_playback_buffer(r: &Rig) {
    let mut m = r.ram.0.lock().unwrap();
    for i in 0..1024u32 {
        let at = BUFFER as usize + i as usize * 4;
        m[at..at + 4].copy_from_slice(&(0x0100_0000 + i).to_le_bytes());
    }
}

/// A stereo stream of 960 dwords with an interrupt every 480.
fn program_playback(c: &Ice1724) {
    mt_wb(c, 0x19, 3);
    mt_wl(c, 0x10, BUFFER as u32);
    mt_ww(c, 0x14, 959);
    mt_ww(c, 0x1c, 479);
}

fn inta(r: &Rig) -> bool {
    r.config_dword(1) >> 16 & 0x08 != 0
}

#[test]
fn playback_raises_its_interrupt_on_the_exact_frame() {
    let r = rig(Variant::Dm8);
    fill_playback_buffer(&r);
    program_playback(&r);

    // Registers read back as programmed.
    assert_eq!(mt_rl(&r, 0x10), BUFFER as u32);
    assert_eq!(mt_rw(&r, 0x14), 959);

    // Reset masks everything: a run raises nothing.
    mt_wb(&r, 0x18, 0x01);
    step(&r, 5_000_000);
    assert_eq!(mt_rb(&r, 0x00), 0);
    assert!(!inta(&r) && !r.irq.level());
    mt_wb(&r, 0x18, 0x00);

    // Unmask the interleaved stream and the multi-channel source.
    let first_run = r.stats().analog_frames;
    assert_eq!(first_run, 240);
    program_playback(&r);
    mt_wb(&r, 0x03, 0xfe);
    ccs_wb(&r, 0x01, 0xef);
    mt_wb(&r, 0x18, 0x01);

    // 235 frames have gone by at 4.9 ms: 470 dwords, no interrupt yet.
    step(&r, 4_900_000);
    assert_eq!(mt_rb(&r, 0x00), 0);
    assert!(!inta(&r));
    assert_eq!(mt_rw(&r, 0x14), 959 - 470);
    assert_eq!(mt_rl(&r, 0x10), BUFFER as u32 + 470 * 4);

    // At 5 ms the 480th dword has moved.
    step(&r, 100_000);
    assert_eq!(mt_rb(&r, 0x00), 0x01);
    assert_eq!(ccs_rb(&r, 0x02) & 0x10, 0x10);
    assert!(inta(&r) && r.irq.level());
    assert_eq!(mt_rw(&r, 0x14), 959 - 480);
    assert_eq!(mt_rl(&r, 0x10), BUFFER as u32 + 480 * 4);
    assert_eq!(r.stats().analog_frames, first_run + 240);
    assert_eq!(r.stats().analog_last, 0x0100_0000 + 479);
    assert_eq!(r.stats().dropped_frames, 0);

    // Write 1 to clear; the macro bit and the line follow.
    mt_wb(&r, 0x00, 0x01);
    assert_eq!(mt_rb(&r, 0x00), 0);
    assert_eq!(ccs_rb(&r, 0x02) & 0x10, 0);
    assert!(!inta(&r) && !r.irq.level());

    // The buffer wraps and the interval repeats.
    step(&r, 5_000_000);
    assert_eq!(mt_rb(&r, 0x00), 0x01);
    assert_eq!(mt_rw(&r, 0x14), 959);
    assert_eq!(mt_rl(&r, 0x10), BUFFER as u32);
    assert_eq!(r.stats().analog_last, 0x0100_0000 + 959);
    // A masked source stops reaching the line, and does not latch.
    mt_wb(&r, 0x00, 0x01);
    mt_wb(&r, 0x03, 0xff);
    step(&r, 5_000_000);
    assert_eq!(mt_rb(&r, 0x00), 0);
    assert!(!inta(&r));
    // The mask in CCS01 gates the line, not MT00.
    mt_wb(&r, 0x03, 0xfe);
    step(&r, 5_000_000);
    assert_eq!(mt_rb(&r, 0x00), 1);
    assert!(inta(&r));
    ccs_wb(&r, 0x01, 0xfe);
    assert!(!inta(&r) && !r.irq.level());
    ccs_wb(&r, 0x01, 0xef);
    assert!(inta(&r) && r.irq.level());
}

/// The timer is armed for the frame that completes an interval, when that
/// is sooner than a flow tick: at 48 kHz the 96th dword (48 frames) is at
/// exactly 1 ms.
#[test]
fn the_timer_deadline_is_the_frame_that_ends_the_interval() {
    let r = rig(Variant::Dm8);
    program_playback(&r);
    mt_ww(&r, 0x1c, 95);
    mt_wb(&r, 0x03, 0xfe);
    mt_wb(&r, 0x18, 0x01);
    assert_eq!(r.next_deadline(&r.state.lock(), 0), Some(1_000_000));
    // A longer interval waits for the flow tick instead.
    mt_ww(&r, 0x1c, 479);
    assert_eq!(r.next_deadline(&r.state.lock(), 0), Some(FLOW_NS));
    // A masked source is only a flow tick away.
    mt_ww(&r, 0x1c, 95);
    mt_wb(&r, 0x03, 0xff);
    assert_eq!(r.next_deadline(&r.state.lock(), 0), Some(FLOW_NS));
    // The rate does not divide a nanosecond: 48 frames of 44.1 kHz are
    // 1088435.37 ns, so the deadline rounds up.
    mt_wb(&r, 0x03, 0xfe);
    mt_wb(&r, 0x01, 0x08);
    assert_eq!(r.next_deadline(&r.state.lock(), 0), Some(1_088_436));
}

/// Collect the frames the card hands the host, with `channels` to a frame.
fn tapped(r: &Rig, channels: usize) -> Arc<Mutex<Vec<i16>>> {
    let tap = Arc::new(Mutex::new(Vec::new()));
    let mut out = PcmOut::new(None, None);
    out.tap = Some(tap.clone());
    out.tap_channels = channels;
    *r.sink.lock() = Some(out);
    tap
}

/// Fill 1024 dwords at `at` with the sample `f(index)` in each dword's top
/// 16 bits.
fn fill_samples(r: &Rig, at: usize, f: impl Fn(usize) -> i16) {
    let mut m = r.ram.0.lock().unwrap();
    for i in 0..1024 {
        let o = at + i * 4;
        m[o..o + 4].copy_from_slice(&(u32::from(f(i) as u16) << 16).to_le_bytes());
    }
}

/// Where the S/PDIF engine's buffer sits.
const SPDIF_BUFFER: usize = 0x4000;

/// Slot `k` of every `slots`-slot frame carries `(k + 1) * 0x100`. The
/// eight-slot order is front left and right, centre, low frequency, back
/// pair, side pair.
fn fill_slot_numbers(r: &Rig, slots: usize) {
    fill_samples(r, BUFFER as usize, |i| (i % slots + 1) as i16 * 0x100);
}

fn program_spdif(c: &Ice1724) {
    mt_wl(c, 0x40, SPDIF_BUFFER as u32);
    mt_ww(c, 0x44, 959);
}

#[test]
fn every_slot_reaches_a_host_with_eight_channels() {
    let r = rig(Variant::Dm8);
    let tap = tapped(&r, 8);
    fill_slot_numbers(&r, 8);
    mt_wl(&r, 0x10, BUFFER as u32);
    mt_ww(&r, 0x14, 959);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 1_000_000);
    let t = tap.lock();
    assert_eq!(t.len(), 48 * 8);
    for f in t.chunks(8) {
        assert_eq!(f, [0x100, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700, 0x800]);
    }
}

#[test]
fn fewer_slots_play_on_their_own_speakers() {
    let r = rig(Variant::Dm8);
    let tap = tapped(&r, 8);
    fill_slot_numbers(&r, 6);
    // Six slots are a 5.1 frame; two of eight host channels stay silent.
    mt_wb(&r, 0x19, 1);
    mt_wl(&r, 0x10, BUFFER as u32);
    mt_ww(&r, 0x14, 959);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 1_000_000);
    {
        let t = tap.lock();
        assert_eq!(t.len(), 48 * 8);
        for f in t.chunks(8) {
            assert_eq!(f, [0x100, 0x200, 0x300, 0x400, 0x500, 0x600, 0, 0]);
        }
    }
    // Four slots are front and back: the back pair takes the 5.1 and 7.1
    // back positions, not the centre and low-frequency ones.
    mt_wb(&r, 0x18, 0);
    mt_wb(&r, 0x19, 2);
    fill_slot_numbers(&r, 4);
    tap.lock().clear();
    mt_wl(&r, 0x10, BUFFER as u32);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 1_000_000);
    let t = tap.lock();
    assert_eq!(t.len(), 48 * 8);
    for f in t.chunks(8) {
        assert_eq!(f, [0x100, 0x200, 0, 0, 0x300, 0x400, 0, 0]);
    }
}

#[test]
fn a_stereo_host_gets_a_downmix_of_all_eight_slots() {
    let r = rig(Variant::Dm8);
    let tap = tapped(&r, 2);
    fill_slot_numbers(&r, 8);
    mt_wl(&r, 0x10, BUFFER as u32);
    mt_ww(&r, 0x14, 959);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 1_000_000);
    let t = tap.lock();
    assert_eq!(t.len(), 48 * 2);
    // The front pair at unity, centre and the back and side channels at
    // -3 dB, no low-frequency channel.
    let k = std::f64::consts::FRAC_1_SQRT_2;
    let left = 256.0 + k * (768.0 + 1280.0 + 1792.0);
    let right = 512.0 + k * (768.0 + 1536.0 + 2048.0);
    for f in t.chunks(2) {
        assert!((f[0] as f64 - left).abs() <= 1.0, "left {} want {left}", f[0]);
        assert!((f[1] as f64 - right).abs() <= 1.0, "right {} want {right}", f[1]);
    }
}

#[test]
fn spdif_is_mixed_into_the_front_pair() {
    let r = rig(Variant::Dm8);
    let tap = tapped(&r, 8);
    // Stereo analog stream at 0x100, S/PDIF at 0x200.
    mt_wb(&r, 0x19, 3);
    fill_samples(&r, BUFFER as usize, |_| 0x100);
    fill_samples(&r, SPDIF_BUFFER, |_| 0x200);
    mt_wl(&r, 0x10, BUFFER as u32);
    mt_ww(&r, 0x14, 959);
    program_spdif(&r);

    // S/PDIF alone plays on the front pair.
    mt_wb(&r, 0x18, 0x80);
    step(&r, 1_000_000);
    {
        let mut t = tap.lock();
        assert_eq!(t.len(), 48 * 8);
        for f in t.chunks(8) {
            assert_eq!(f, [0x200, 0x200, 0, 0, 0, 0, 0, 0]);
        }
        t.clear();
    }
    // The analog engine starts a millisecond later; the two sum frame for
    // frame.
    mt_wb(&r, 0x18, 0x81);
    step(&r, 1_000_000);
    let t = tap.lock();
    assert_eq!(t.len(), 48 * 8);
    for f in t.chunks(8) {
        assert_eq!(f, [0x300, 0x300, 0, 0, 0, 0, 0, 0]);
    }
}

#[test]
fn a_stopped_engine_adds_nothing() {
    let r = rig(Variant::Dm8);
    let tap = tapped(&r, 2);
    mt_wb(&r, 0x19, 3);
    fill_samples(&r, BUFFER as usize, |_| 0x100);
    fill_samples(&r, SPDIF_BUFFER, |_| 0x200);
    mt_wl(&r, 0x10, BUFFER as u32);
    mt_ww(&r, 0x14, 959);
    program_spdif(&r);
    mt_wb(&r, 0x18, 0x81);
    step(&r, 1_000_000);
    mt_wb(&r, 0x18, 0x01);
    tap.lock().clear();
    step(&r, 1_000_000);
    let t = tap.lock();
    assert_eq!(t.len(), 48 * 2);
    assert!(t.iter().all(|&s| s == 0x100), "{:?}", &t[..4]);
}

/// A host stream behind a ring the test reads back, at `rate`.
fn detached_host(r: &Rig, rate: u32, channels: usize) -> rtrb::Consumer<i16> {
    let (p, c) = rtrb::RingBuffer::<i16>::new(1 << 16);
    let out = HostOut::detached_channels(rate, channels, p);
    let rs = FrameResampler::new(48000, rate, channels);
    *r.sink.lock() = Some(PcmOut::new(Some((out, rs)), None));
    c
}

#[test]
fn the_host_ring_gets_whole_frames_at_the_stream_rate() {
    for (rate, frames) in [(48000, 48), (96000, 96)] {
        let r = rig(Variant::Dm8);
        let mut c = detached_host(&r, rate, 8);
        fill_slot_numbers(&r, 8);
        mt_wl(&r, 0x10, BUFFER as u32);
        mt_ww(&r, 0x14, 959);
        mt_wb(&r, 0x18, 0x01);
        step(&r, 1_000_000);
        let got: Vec<i16> = std::iter::from_fn(|| c.pop().ok()).collect();
        assert_eq!(got.len(), frames * 8, "{rate} Hz");
        assert_eq!(r.stats().dropped_frames, 0);
        // The interpolation settles on each channel's own level.
        let last = &got[got.len() - 8..];
        assert_eq!(last, [0x100, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700, 0x800], "{rate} Hz");
    }
}

#[test]
fn a_full_host_ring_counts_dropped_frames() {
    let r = rig(Variant::Dm8);
    let (p, _c) = rtrb::RingBuffer::<i16>::new(20);
    let out = HostOut::detached_channels(48000, 8, p);
    *r.sink.lock() = Some(PcmOut::new(Some((out, FrameResampler::new(48000, 48000, 8))), None));
    fill_slot_numbers(&r, 8);
    mt_wl(&r, 0x10, BUFFER as u32);
    mt_ww(&r, 0x14, 959);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 1_000_000);
    // Two whole frames fit in 20 samples.
    assert_eq!(r.stats().dropped_frames, 46);
}

#[test]
fn pause_rate_and_clock_gate_stop_the_engine() {
    let r = rig(Variant::Dm8);
    fill_playback_buffer(&r);
    program_playback(&r);

    // 44.1 kHz: 220 frames in 5 ms, 440 dwords of the 960.
    mt_wb(&r, 0x01, 0x08);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 5_000_000);
    assert_eq!(r.stats().analog_frames, 220);
    assert_eq!(mt_rw(&r, 0x14), 959 - 440);

    // Pause freezes the position; resume continues from it.
    mt_wb(&r, 0x1b, 0x01);
    step(&r, 5_000_000);
    assert_eq!(mt_rw(&r, 0x14), 959 - 440);
    mt_wb(&r, 0x1b, 0x00);
    step(&r, 1_000_000);
    assert_eq!(mt_rw(&r, 0x14), 959 - 440 - 88);

    // Stop: the position stays.
    mt_wb(&r, 0x18, 0x00);
    step(&r, 5_000_000);
    assert_eq!(mt_rw(&r, 0x14), 959 - 440 - 88);

    // The clock gate in CCS1C stops the engine too.
    mt_wb(&r, 0x18, 0x01);
    ccs_wb(&r, 0x1c, 0x01);
    step(&r, 5_000_000);
    assert_eq!(mt_rw(&r, 0x14), 959 - 440 - 88);
    ccs_wb(&r, 0x1c, 0x00);
    step(&r, 1_000_000);
    assert_eq!(mt_rw(&r, 0x14), 959 - 440 - 88 - 88);
}

#[test]
fn slots_per_frame_follow_mt19() {
    let r = rig(Variant::Dm8);
    fill_playback_buffer(&r);
    // MT19 = 0: eight slots per frame, a 960-dword buffer holds 120.
    mt_wl(&r, 0x10, BUFFER as u32);
    mt_ww(&r, 0x14, 959);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 1_000_000);
    assert_eq!(r.stats().analog_frames, 48);
    assert_eq!(mt_rw(&r, 0x14), 959 - 48 * 8);
    // Six slots.
    mt_wb(&r, 0x18, 0);
    mt_wb(&r, 0x19, 1);
    mt_wl(&r, 0x10, BUFFER as u32);
    mt_wb(&r, 0x18, 1);
    step(&r, 1_000_000);
    assert_eq!(mt_rw(&r, 0x14), 959 - 48 * 6);
}

#[test]
fn a_failed_transfer_flags_the_engine_and_the_bus() {
    let r = bare(Variant::Dm8);
    // No way to memory: the transfer fails.
    program_playback(&r);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 1_000_000);
    assert_eq!(mt_rb(&r, 0x1a), 0x01);
    // A master abort in the PCI status, and the engine keeps going.
    assert_eq!(r.config_dword(1) >> 16 & 0x2000, 0x2000);
    assert!(mt_rw(&r, 0x14) < 959);
    // MT03 masks the FIFO-error source at reset: nothing in MT00.
    assert_eq!(mt_rb(&r, 0x00), 0);

    // Write 1 to clear both.
    mt_wb(&r, 0x1a, 0x01);
    assert_eq!(mt_rb(&r, 0x1a), 0);
    cfg_dw(&r, 1, 0x2000_0000 | 5);
    assert_eq!(r.config_dword(1) >> 16 & 0x2000, 0);

    // Unmasked, the error reaches MT00 bit 3 and the line.
    mt_wb(&r, 0x03, 0xf7);
    ccs_wb(&r, 0x01, 0xef);
    step(&r, 1_000_000);
    assert_eq!(mt_rb(&r, 0x00), 0x08);
    assert!(inta(&r));
}

#[test]
fn a_transfer_needs_bus_mastering() {
    let r = rig(Variant::Dm8);
    fill_playback_buffer(&r);
    program_playback(&r);
    cfg_dw(&r, 1, PCI_COMMAND_IO);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 1_000_000);
    assert_eq!(mt_rb(&r, 0x1a), 0x01, "the engine reports a failed transfer");
    assert_eq!(r.stats().analog_last, 0, "and played silence");
}

#[test]
fn record_writes_silence_over_the_buffer() {
    let r = rig(Variant::Dm8);
    r.ram.0.lock().unwrap()[BUFFER as usize..BUFFER as usize + 512].fill(0xaa);

    mt_wl(&r, 0x20, BUFFER as u32);
    mt_ww(&r, 0x24, 63);
    mt_ww(&r, 0x26, 31);
    mt_wb(&r, 0x03, 0xfd);
    mt_wb(&r, 0x18, 0x02);
    step(&r, 500_000);
    // 24 frames: 48 dwords, past the 32-dword interval, short of the end.
    assert_eq!(mt_rb(&r, 0x00), 0x02);
    assert_eq!(mt_rw(&r, 0x24), 63 - 48);
    assert_eq!(r.stats().capture_frames, 24);
    let m = r.ram.0.lock().unwrap();
    let buf = &m[BUFFER as usize..BUFFER as usize + 512];
    assert!(buf[..48 * 4].iter().all(|&b| b == 0));
    assert!(buf[48 * 4..].iter().all(|&b| b == 0xaa));
    drop(m);

    // The second record engine runs independently.
    mt_wl(&r, 0x30, BUFFER as u32 + 0x400);
    mt_ww(&r, 0x34, 63);
    mt_wb(&r, 0x18, 0x06);
    step(&r, 250_000);
    assert_eq!(mt_rw(&r, 0x34), 63 - 24);
}

#[test]
fn spdif_and_stereo_engines_run_when_mt19_gives_them_slots() {
    let r = rig(Variant::Dm8);
    fill_playback_buffer(&r);
    mt_wl(&r, 0x40, BUFFER as u32);
    mt_ww(&r, 0x44, 255);
    mt_wb(&r, 0x18, 0x80);
    step(&r, 1_000_000);
    assert_eq!(r.stats().spdif_frames, 48);
    assert_eq!(mt_rw(&r, 0x44), 255 - 96);
    assert_eq!(mt_rl(&r, 0x40), BUFFER as u32 + 96 * 4);

    // PDMA3 needs MT19 above 0; with 0 it does not run.
    mt_wl(&r, 0x50, BUFFER as u32);
    mt_ww(&r, 0x54, 255);
    mt_wb(&r, 0x18, 0xc0);
    step(&r, 1_000_000);
    assert_eq!(mt_rw(&r, 0x54), 255);
    assert!(seen(&r, "engine 6 started while MT19"));
    mt_wb(&r, 0x19, 1);
    step(&r, 1_000_000);
    assert_eq!(mt_rw(&r, 0x54), 255 - 96);
}

/// Ticks that arrive when they are due hold nothing back, however many
/// there are: each is measured against its own deadline, not the first.
#[test]
fn on_time_ticks_hold_nothing_back() {
    let r = rig(Variant::Dm8);
    fill_playback_buffer(&r);
    program_playback(&r);
    mt_wb(&r, 0x18, 0x01);
    let gen = r.state.lock().timer_gen;
    let first = r.next_deadline(&r.state.lock(), 0).unwrap();
    r.state.lock().timer_due = first;
    for _ in 0..100 {
        let due = r.state.lock().timer_due;
        r.fake_ns().store(due, Ordering::Relaxed);
        assert!(matches!(r.timer_fire(gen), TimerReturn::RescheduleOneShot(_)));
    }
    assert_eq!(r.stalled_ns.load(Ordering::SeqCst), 0);
    // 100 flow ticks of 2 ms at 48 kHz.
    assert_eq!(r.stats().analog_frames, 9600);
}

/// A timer far behind its deadline means the host held up its thread: the
/// clock goes back to the deadline, by at most 200 ms, and the engines do
/// not replay the gap.
#[test]
fn a_held_up_timer_does_not_replay_the_gap() {
    let r = rig(Variant::Dm8);
    fill_playback_buffer(&r);
    program_playback(&r);
    mt_wb(&r, 0x18, 0x01);
    let gen = {
        let mut s = r.state.lock();
        s.timer_due = FLOW_NS;
        s.timer_gen
    };
    r.fake_ns().store(500_000_000, Ordering::Relaxed);
    r.timer_fire(gen);
    // 500 ms late, 200 ms taken back: 300 ms of audio, not 500.
    assert_eq!(r.stats().analog_frames, 300 * 48);
    assert_eq!(r.stalled_ns.load(Ordering::SeqCst), 200_000_000);
    // A tick on time wins a share back.
    let due = r.state.lock().timer_due;
    r.fake_ns().store(due + 1, Ordering::Relaxed);
    r.timer_fire(gen);
    assert!(r.stalled_ns.load(Ordering::SeqCst) < 200_000_000);
}

#[test]
fn a_long_look_away_is_not_replayed_through_memory() {
    let r = rig(Variant::Dm8);
    fill_playback_buffer(&r);
    program_playback(&r);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 1_000_000);
    assert_eq!(r.stats().analog_frames, 48);
    step(&r, 3_000_000_000);
    assert_eq!(r.stats().analog_frames, 48, "the frames of the gap were skipped");
    assert!(seen(&r, "DMA clock jump skipped"));
    step(&r, 1_000_000);
    assert_eq!(r.stats().analog_frames, 96, "and the clock goes on from the present");
}

const IRIX_RING_FRAMES: u64 = 1024;
/// How far past the position it reads IRIX writes, in frames.
const IRIX_LEAD_FRAMES: u64 = 90;

/// IRIX's playback as a guest run shows it: the engine's interrupt masked,
/// a ring of 1024 eight-slot frames at 44.1 kHz, the address register read
/// about once a millisecond and samples written about 90 frames (2 ms) past
/// the position read. Each frame's front-left sample is its number plus one.
struct IrixWriter {
    /// Frames written, and the card's position when last read, counted
    /// from the start.
    written: u64,
    read: u64,
}

impl IrixWriter {
    fn start(r: &Rig) -> Self {
        mt_wb(r, 0x01, 8);
        mt_wb(r, 0x19, 0);
        mt_wl(r, 0x10, BUFFER as u32);
        mt_wl(r, 0x14, IRIX_RING_FRAMES as u32 * 8 - 1);
        let mut w = Self { written: 0, read: 0 };
        w.fill(r, IRIX_LEAD_FRAMES);
        mt_wb(r, 0x18, 0x01);
        w
    }

    fn fill(&mut self, r: &Rig, upto: u64) {
        let mut m = r.ram.0.lock().unwrap();
        while self.written < upto {
            let at = BUFFER as usize + (self.written % IRIX_RING_FRAMES) as usize * 32;
            let v = (self.written + 1) as u16;
            m[at..at + 4].copy_from_slice(&(u32::from(v) << 16).to_le_bytes());
            self.written += 1;
        }
    }

    /// Read the position and write up to 90 frames past it.
    fn poll(&mut self, r: &Rig) {
        let pos = u64::from((mt_rl(r, 0x10) - BUFFER as u32) / 32);
        self.read += (pos + IRIX_RING_FRAMES - self.read % IRIX_RING_FRAMES) % IRIX_RING_FRAMES;
        self.fill(r, self.read + IRIX_LEAD_FRAMES);
    }
}

/// The front-left samples the card played are the frame numbers in order:
/// nothing replayed from an earlier lap, nothing skipped.
fn assert_played_in_order(tap: &Mutex<Vec<i16>>) {
    let left: Vec<u16> = tap.lock().chunks_exact(8).map(|f| f[0] as u16).collect();
    assert_eq!(left.first(), Some(&1));
    for (i, w) in left.windows(2).enumerate() {
        assert_eq!(w[1], w[0].wrapping_add(1), "frame {}: sample {} after {}", i + 1, w[1], w[0]);
    }
}

/// A guest that steers by the position it polls writes only a little ahead
/// of it. While the host does not run that guest, the card's clock goes on;
/// a card that read on would pass what was written and play the previous
/// lap again, the 23 ms buzz heard under load. It reads at most 1.5 ms
/// past the polled position and waits there.
#[test]
fn a_polled_engine_does_not_read_past_what_the_guest_had_time_to_write() {
    let r = rig(Variant::Dm8);
    let tap = tapped(&r, 8);
    let mut w = IrixWriter::start(&r);
    for _ in 0..50 {
        step(&r, 1_000_000);
        w.poll(&r);
    }
    // The guest is not run for 15 ms; the card's timer still is.
    for _ in 0..15 {
        step(&r, 1_000_000);
    }
    for _ in 0..50 {
        step(&r, 1_000_000);
        w.poll(&r);
    }
    assert_played_in_order(&tap);
}

/// The wait is only for a guest steering by polls: one that has not read
/// the position for 50 ms, or that takes the engine's interrupt, is not held
/// back.
#[test]
fn an_engine_not_steered_by_polls_runs_on_its_clock() {
    let r = rig(Variant::Dm8);
    let mut w = IrixWriter::start(&r);
    step(&r, 1_000_000);
    w.poll(&r);
    let at = r.stats().analog_frames;
    step(&r, 60_000_000);
    assert_eq!(r.stats().analog_frames - at, 2646, "60 ms with no poll");

    let r = rig(Variant::Dm8);
    let mut w = IrixWriter::start(&r);
    mt_wb(&r, 0x03, 0xfe);
    step(&r, 1_000_000);
    w.poll(&r);
    let at = r.stats().analog_frames;
    step(&r, 10_000_000);
    assert_eq!(r.stats().analog_frames - at, 441, "10 ms with the interrupt unmasked");
}

/// Polls further apart than the read-ahead leave the engine behind its
/// clock. At most 100 ms of that is kept: when the guest stops steering,
/// the engine does not run off the rest in one burst.
#[test]
fn a_held_engine_keeps_at_most_100_ms_to_make_up() {
    let r = rig(Variant::Dm8);
    let mut w = IrixWriter::start(&r);
    // 300 ms of polls 10 ms apart: 1.5 ms of audio each, 255 ms behind.
    for _ in 0..30 {
        step(&r, 10_000_000);
        w.poll(&r);
    }
    let at = r.stats().analog_frames;
    // No poll for 60 ms: the engine runs on its clock again.
    step(&r, 60_000_000);
    let burst = r.stats().analog_frames - at;
    // 100 ms kept plus the 60 ms, at 44.1 kHz.
    assert!(burst <= 7056 + 66, "made up {burst} frames at once");
}

// ---- unknown and unexpected accesses ----

#[test]
fn unknown_registers_read_zero_and_are_logged() {
    let r = rig(Variant::Dm8);
    // Unassigned offsets read 0, ignore writes, and are logged.
    assert_eq!(mt_rb(&r, 0x28), 0);
    mt_wb(&r, 0x28, 0xff);
    assert_eq!(mt_rb(&r, 0x28), 0);
    assert_eq!(ccs_rb(&r, 0x08), 0);
    ccs_wb(&r, 0x1d, 0xff);
    // Write-only terminal counts, the peak meters and the MPU-401 UART.
    assert_eq!(mt_rw(&r, 0x1c), 0);
    assert_eq!(mt_rb(&r, 0x3f), 0);
    ccs_wb(&r, 0x0c, 0x55);
    // Rates and formats nothing defines.
    mt_wb(&r, 0x01, 0x0d);
    mt_wb(&r, 0x02, 0x01);
    // Writes to read-only registers.
    ccs_wb(&r, 0x0a, 0x01);
    mt_wb(&r, 0x3f, 0x01);
    // Configuration registers nothing describes.
    assert_eq!(r.config_dword(6), 0);
    assert_eq!(r.config_dword(24), 0);
    cfg_dw(&r, 16, 0xffff_ffff);
    // Routing is stored but not modelled.
    mt_wl(&r, 0x2c, 2);
    assert_eq!(mt_rl(&r, 0x2c), 2);
    // Past the end of a block.
    assert_eq!(r.bar_read(0, 0x20, 1), 0);

    for what in [
        "read of unassigned MT 0x28",
        "write of 0xff to unassigned MT 0x28",
        "read of unassigned CCS 0x08",
        "write of 0xff to unassigned CCS 0x1d",
        "read of write-only terminal count 0x1c",
        "peak meters are not modelled",
        "MPU-401 UART write at CCS 0x0c",
        "reserved sample rate code 13",
        "reserved data format 1",
        "write to read-only CCS 0x0a",
        "write to read-only peak data",
        "output routing is not modelled",
        "PCI config read of unknown register at 0x18",
        "PCI config read of unknown register at 0x60",
        "PCI config write of 0xffffffff to unknown register at 0x40",
        "read past the end of BAR 0",
    ] {
        assert!(seen(&r, what), "no log line for: {what}");
    }
    assert_eq!(r.unexpected(), r.seen.lock().len() as u64);
}

// ---- the AC'97 variant ----

#[test]
fn the_ac97_variant_has_a_codec_and_no_converters() {
    let r = rig(Variant::Ac97);
    // No EEPROM: the default subsystem pair and strap.
    assert_eq!(r.config_dword(0), 0x1724_1412);
    assert_eq!(r.config_dword(11), 0x1724_1412);
    assert_eq!(ccs_rb(&r, 0x13) & 0x80, 0);
    ccs_wb(&r, 0x12, 0x77);
    ccs_wb(&r, 0x11, 0);
    ccs_wb(&r, 0x10, 0xa0);
    assert_eq!(ccs_rb(&r, 0x12), 0x77, "nothing answers the EEPROM address");
    // AC-link mode at reset, codec ready once PRST# is released.
    assert_eq!(ccs_rb(&r, 0x05) & 0x80, 0);
    assert_eq!(mt_rb(&r, 0x05) & 0x08, 0x08);

    // Write cycle then read cycle at index 0x18.
    mt_wb(&r, 0x04, 0x18);
    mt_ww(&r, 0x06, 0x1234);
    mt_wb(&r, 0x05, 0x20);
    assert_eq!(mt_rb(&r, 0x05) & 0x30, 0);
    mt_ww(&r, 0x06, 0);
    mt_wb(&r, 0x05, 0x10);
    assert_eq!(mt_rb(&r, 0x05) & 0x30, 0);
    assert_eq!(mt_rw(&r, 0x06), 0x1234);
    mt_wb(&r, 0x04, 0x1a);
    mt_wb(&r, 0x05, 0x10);
    assert_eq!(mt_rw(&r, 0x06), 0);
    mt_wb(&r, 0x04, 0x18);
    mt_wb(&r, 0x05, 0x10);
    assert_eq!(mt_rw(&r, 0x06), 0x1234);

    // The cold-reset bit holds the codec not ready; cycles then do nothing.
    mt_wb(&r, 0x05, 0x80);
    assert_eq!(mt_rb(&r, 0x05) & 0x88, 0x80);
    mt_wb(&r, 0x05, 0x90);
    assert_eq!(mt_rw(&r, 0x06), 0);
    mt_wb(&r, 0x05, 0x00);

    // There are no converter pins: GPIO frames are not decoded.
    ccs_ww(&r, 0x18, 0x7a);
    ccs_ww(&r, 0x16, 0);
    ccs_ww(&r, 0x14, 0x72);
    codec_write(&r, CS_FRONT, 1, 0, 0x0f);
    assert_eq!(r.stats().ak4381_writes, 0);

    // Stereo playback is the same engine.
    fill_playback_buffer(&r);
    program_playback(&r);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 1_000_000);
    assert_eq!(r.stats().analog_frames, 48);
}

#[test]
fn the_dm8_variant_has_no_ac97_codec() {
    let r = rig(Variant::Dm8);
    // The DM8 is an I2S board: no codec answers on the AC-link.
    assert_eq!(mt_rb(&r, 0x05) & 0x08, 0);
    mt_wb(&r, 0x04, 0x18);
    mt_ww(&r, 0x06, 0x1234);
    mt_wb(&r, 0x05, 0x20);
    mt_wb(&r, 0x05, 0x10);
    assert_eq!(mt_rw(&r, 0x06), 0);
}

// ---- interrupt wiring ----

#[test]
fn a_line_connected_late_is_driven_to_the_present_level() {
    let c = Ice1724::with_clock(Variant::Dm8, TestTime::clock());
    c.fake_ns().store(0, Ordering::Relaxed);
    map(&c);
    c.state().connect_dma(Arc::new(Ram(std::sync::Mutex::new(vec![0; 0x2000]))));
    program_playback(&c);
    mt_wb(&c, 0x03, 0xfe);
    ccs_wb(&c, 0x01, 0xef);
    mt_wb(&c, 0x18, 1);
    step(&c, 5_000_000);
    let (irq, line) = LevelProbe::line();
    assert!(!irq.level());
    c.state().connect_irq(line);
    c.attached();
    assert!(irq.level());
}

// ---- on a real bus ----

/// A host whose memory every requester reaches, which records who asked
/// and every interrupt change.
struct BusHost {
    ram: Arc<Ram>,
    requesters: std::sync::Mutex<Vec<u8>>,
    irqs: std::sync::Mutex<Vec<(u8, bool)>>,
}

impl crate::pci::PciHost for BusHost {
    fn dma_for(&self, requester: u8) -> Arc<dyn PciDma> {
        self.requesters.lock().unwrap().push(requester);
        self.ram.clone()
    }

    fn set_irq(&self, irq: u8, level: bool) {
        self.irqs.lock().unwrap().push((irq, level));
    }
}

/// Device 2 with INTA and INTC routed, as the IO9's PIC routes them.
static WIRING: crate::pci::IntxRouting = crate::pci::IntxRouting {
    devices: 8,
    irqs: 4,
    intx: &[(2, [Some(0), None, Some(1), None])],
    board_irqs: &[],
};

fn at(device: u8) -> crate::pci::DevFn {
    crate::pci::DevFn { device, function: 0 }
}

/// The card as the Tezro has it: device 2 behind a PCI-to-PCI bridge at
/// device 2, its I/O BARs inside the window firmware opens on the bridge,
/// its transfers and its interrupt on the bridge's slot.
#[test]
fn behind_a_bridge_the_io_window_reaches_the_bars_and_the_slot_carries_dma_and_interrupt() {
    use crate::pci::{PciBus, PciHost, PciToPciBridge, Space};
    let host = Arc::new(BusHost {
        ram: Arc::new(Ram(std::sync::Mutex::new(vec![0; 0x10_0000]))),
        requesters: Default::default(),
        irqs: Default::default(),
    });
    let up: std::sync::Weak<dyn PciHost> = Arc::downgrade(&(host.clone() as Arc<dyn PciHost>));
    let bus = PciBus::new(0, &WIRING, up);
    let ppb = PciToPciBridge::with_wide_windows(&bus, at(2), 0x104c, 0xac28, 1);
    bus.attach(at(2), ppb.clone());
    let card = Arc::new(Ice1724::with_clock(Variant::Dm8, TestTime::clock()));
    card.fake_ns().store(0, Ordering::Relaxed);
    ppb.secondary().attach(at(2), card.clone());
    assert_eq!(host.requesters.lock().unwrap().as_slice(), [2, 2], "the bridge's slot masters for both");

    // Nothing is reachable until firmware opens the bridge's I/O window, which
    // reaches 32-bit I/O addresses as the PIC's short windows address them.
    let io = |a| bus.decode(Space::Io, a).map(|(_, bar, off)| (bar, off));
    ppb.secondary().config_write(at(2), 0x10, 4, 0x0012_3001);
    ppb.secondary().config_write(at(2), 0x14, 4, 0x0012_3081);
    ppb.secondary().config_write(at(2), 0x04, 2, 5);
    assert_eq!(io(0x0012_3004), None);
    bus.config_write(at(2), 0x18, 4, 0x0001_0100);
    bus.config_write(at(2), 0x1c, 2, 0xf000);
    bus.config_write(at(2), 0x30, 4, 0x0012_0012);
    bus.config_write(at(2), 0x04, 2, 1);
    assert_eq!(io(0x0012_3004), Some((0, 4)), "CCS04");
    assert_eq!(io(0x0012_3090), Some((1, 0x10)), "MT10");
    assert_eq!(io(0x0013_3004), None, "outside the window");

    // A register through the window, and playback from memory through the
    // bridge's slot, interrupting on the slot's INTC.
    let (f, bar, off) = bus.decode(Space::Io, 0x0012_3004).unwrap();
    assert_eq!(f.bar_read(bar, off, 1), 0x0f);
    let ram = host.ram.clone();
    {
        let mut m = ram.0.lock().unwrap();
        for i in 0..960u32 {
            let at = BUFFER as usize + i as usize * 4;
            m[at..at + 4].copy_from_slice(&(0x0100_0000 + i).to_le_bytes());
        }
    }
    let wr = |a: u32, v: u32, w| {
        let (f, bar, off) = bus.decode(Space::Io, a).unwrap();
        f.bar_write(bar, off, v, w);
    };
    wr(0x0012_3099, 3, 1);
    wr(0x0012_3090, BUFFER as u32, 4);
    wr(0x0012_3094, 959, 2);
    wr(0x0012_309c, 479, 2);
    wr(0x0012_3083, 0xfe, 1);
    wr(0x0012_3001, 0xef, 1);
    wr(0x0012_3098, 1, 1);
    step(&card, 5_000_000);
    assert_eq!(card.stats().analog_last, 0x0100_0000 + 479);
    assert_eq!(host.irqs.lock().unwrap().last(), Some(&(1, true)), "INTA at device 2 is the bridge's INTC");
    // With the bridge's I/O enable off the card is out of reach.
    bus.config_write(at(2), 0x04, 2, 0);
    assert_eq!(io(0x0012_3004), None);
}

// ---- monitor ----

#[test]
fn the_monitor_reports_what_the_engines_did() {
    use crate::traits::Device;
    let r = rig(Variant::Dm8);
    fill_playback_buffer(&r);
    program_playback(&r);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 5_000_000);
    struct Out(Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for Out {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
    r.execute_command("ice1724", &["status"], Box::new(Out(buf.clone()))).unwrap();
    let text = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
    assert!(text.contains("analog 240 (last dword 0x010001df)"), "{text}");
    assert!(r.execute_command("ice1724", &[], Box::new(Out(buf))).is_err());
    assert_eq!(r.register_commands().len(), 1);
}

// ---- save and restore ----

#[test]
fn a_running_engine_resumes_where_it_was_saved() {
    use crate::traits::Saveable;
    let a = rig(Variant::Dm8);
    fill_playback_buffer(&a);
    program_playback(&a);
    mt_wb(&a, 0x03, 0xfe);
    ccs_wb(&a, 0x01, 0xef);
    load_gpio_from_eeprom(&a);
    codec_write(&a, CS_FRONT, 1, 4, 0x33);
    mt_wb(&a, 0x18, 0x01);
    step(&a, 2_000_000);

    let saved = a.save_state();
    // The restored card is built the machine's way: the clock is wherever
    // its own host is, and the saved age puts the anchor behind it.
    let b = rig(Variant::Dm8);
    b.fake_ns().store(a.fake_ns().load(Ordering::Relaxed) + 7_000_000, Ordering::Relaxed);
    b.ram.0.lock().unwrap().copy_from_slice(&a.ram.0.lock().unwrap());
    b.load_state(&saved).expect("load");
    assert_eq!(b.save_state(), saved, "a load and a save round-trip");

    assert_eq!(mt_rw(&b, 0x14), mt_rw(&a, 0x14));
    assert_eq!(mt_rl(&b, 0x10), mt_rl(&a, 0x10));
    assert_eq!(b.ak4381()[4], 0x33);
    assert_eq!(b.stats(), a.stats());
    assert_eq!(mt_rb(&b, 0x18), 1);

    // The restored engine keeps its pace, interval and interrupt.
    step(&a, 3_000_000);
    step(&b, 3_000_000);
    assert_eq!(mt_rw(&b, 0x14), mt_rw(&a, 0x14));
    assert_eq!(b.stats(), a.stats());
    assert_eq!(mt_rb(&b, 0x00), mt_rb(&a, 0x00));
    assert_eq!(mt_rb(&b, 0x00), 0x01);
    assert!(b.irq.level() && a.irq.level());
}

#[test]
fn a_save_that_would_break_the_card_is_refused() {
    use crate::traits::Saveable;
    let a = rig(Variant::Dm8);
    program_playback(&a);
    mt_wb(&a, 0x18, 1);
    step(&a, 1_000_000);
    let good = a.save_state();

    let ac97 = Ice1724::with_clock(Variant::Ac97, TestTime::clock());
    assert!(ac97.load_state(&good).unwrap_err().contains("other variant"));

    let mut bad = good.clone();
    bad["rate"] = crate::snapshot::hex_u32(0);
    assert!(rig(Variant::Dm8).load_state(&bad).unwrap_err().contains("rate"));

    let mut bad = good.clone();
    bad["eng"][0]["pos"] = crate::snapshot::hex_u32(5000);
    assert!(rig(Variant::Dm8).load_state(&bad).unwrap_err().contains("engine 0"));

    let mut bad = good;
    bad["ccs"] = crate::snapshot::u8_slice_to_toml(&[0; 3]);
    assert!(rig(Variant::Dm8).load_state(&bad).is_err(), "a short register file");
}

/// Memory that notes whether the card's state lock was held, and whether
/// another thread could take it, when a transfer reached it.
struct LockProbe {
    ram: Arc<Ram>,
    card: std::sync::Weak<Inner>,
    held: std::sync::atomic::AtomicU32,
    calls: std::sync::atomic::AtomicU32,
}

impl PciDma for LockProbe {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.card.upgrade().is_some_and(|c| c.state.is_locked()) {
            self.held.fetch_add(1, Ordering::Relaxed);
        }
        self.ram.dma_read(addr, buf)
    }
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.card.upgrade().is_some_and(|c| c.state.is_locked()) {
            self.held.fetch_add(1, Ordering::Relaxed);
        }
        self.ram.dma_write(addr, data)
    }
}

#[test]
fn the_state_lock_is_not_held_across_a_dma_transfer() {
    let r = bare(Variant::Dm8);
    let probe = Arc::new(LockProbe {
        ram: r.ram.clone(),
        card: Arc::downgrade(&r.c.inner),
        held: 0.into(),
        calls: 0.into(),
    });
    r.c.state().connect_dma(probe.clone());
    fill_slot_numbers(&r, 2);
    mt_wl(&r, 0x10, BUFFER as u32);
    mt_ww(&r, 0x14, 959);
    mt_wb(&r, 0x18, 0x01);
    step(&r, 1_000_000);
    assert!(probe.calls.load(Ordering::Relaxed) > 0, "no transfer reached memory");
    assert_eq!(probe.held.load(Ordering::Relaxed), 0, "a transfer ran with the state lock held");
}

/// Memory whose first transfer panics; later ones work.
struct PanicRam(Ram, std::sync::atomic::AtomicBool);

impl PciDma for PanicRam {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        if !self.1.swap(true, Ordering::Relaxed) {
            panic!("transfer panicked");
        }
        self.0.dma_read(addr, buf)
    }
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        self.0.dma_write(addr, data)
    }
}

#[test]
fn a_panicking_transfer_does_not_leave_the_card_waiting() {
    let r = bare(Variant::Dm8);
    r.c.state().connect_dma(Arc::new(PanicRam(Ram(std::sync::Mutex::new(vec![0; 0x10_0000])), false.into())));
    let c = &r.c;
    let died = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mt_wl(c, 0x10, BUFFER as u32);
        mt_ww(c, 0x14, 959);
        mt_wb(c, 0x18, 0x01);
        step(c, 1_000_000);
    }));
    assert!(died.is_err());
    assert!(!r.state.lock().moving);
    // The card answers again instead of waiting on the transfer that died.
    let _ = ccs_rb(&r, 0x00);
}

/// The link's far end for a real PIC: widget 10's memory, as the bytes the
/// guest stored in address order.
struct PicMemory(std::sync::Mutex<Vec<u8>>);

impl crate::dev::pic::PicPeer for PicMemory {
    fn interrupt(&self, _: u8, _: u64, _: u64) -> Result<(), crate::dev::pic::Refused> {
        Ok(())
    }
    fn dma_read(&self, _: u8, _: u8, address: u64, data: &mut [u8]) -> Result<(), crate::dev::pic::Refused> {
        let m = self.0.lock().unwrap();
        let src = m.get(address as usize..address as usize + data.len()).ok_or(crate::dev::pic::Refused)?;
        data.copy_from_slice(src);
        Ok(())
    }
    fn dma_write(&self, _: u8, _: u8, address: u64, data: &[u8]) -> Result<(), crate::dev::pic::Refused> {
        let mut m = self.0.lock().unwrap();
        let dst = m.get_mut(address as usize..address as usize + data.len()).ok_or(crate::dev::pic::Refused)?;
        dst.copy_from_slice(data);
        Ok(())
    }
}

/// IRIX stores each sample as a big-endian slot word with the sample in its
/// top half, so a 16-bit sample 0x0de9 is the bytes 0d e9 00 00 in memory.
/// The slot's Device register has bit 19 clear, as the guest left it
/// (0x10000002), and the card must read the sample from the top half of the
/// little-endian dword it is handed.
#[test]
fn a_big_endian_slot_word_through_a_real_pic_slot_plays_its_sample() {
    use crate::dev::pic::{Function, Pic};
    use crate::traits::{BusDevice, BUS_OK};
    let pic = Pic::new(15);
    let mem = Arc::new(PicMemory(std::sync::Mutex::new(vec![0; 0x10000])));
    pic.connect_peer(mem.clone()).unwrap();
    let bus = pic.widget();
    // The direct map reaches widget 10; device 2's attributes as observed.
    assert_eq!(bus.write64(0x80, 10 << 20), BUS_OK);
    assert_eq!(bus.write64(0x208, 0x1000_0002), BUS_OK);
    let c = Ice1724::with_clock(Variant::Dm8, TestTime::clock());
    c.fake_ns().store(0, Ordering::Relaxed);
    let c = Arc::new(c);
    pic.attach(0, 2, c.clone(), Function::default());
    map(&c);
    let tap = Arc::new(Mutex::new(Vec::new()));
    let mut out = PcmOut::new(None, None);
    out.tap = Some(tap.clone());
    // Every slot unfolded, as it was read.
    out.tap_channels = 8;
    *c.sink.lock() = Some(out);
    {
        let mut m = mem.0.lock().unwrap();
        for w in m[0x1000..0x2000].chunks_exact_mut(4) {
            w.copy_from_slice(&[0x0d, 0xe9, 0x00, 0x00]);
        }
    }
    mt_wl(&c, 0x10, 0x8000_1000);
    mt_ww(&c, 0x14, 959);
    mt_wb(&c, 0x18, 0x01);
    step(&c, 1_000_000);
    let t = tap.lock();
    assert!(!t.is_empty(), "nothing reached the host");
    assert!(t.iter().all(|&s| s == 0x0de9), "samples {:?}", &t[..t.len().min(8)]);
}

/// The ICE1724 playing a tone, the guest stopping the output for 0.3 s so
/// the host stream runs dry, then starting it again. The guest steers like
/// IRIX: the engine's interrupt masked, the address read about every half
/// millisecond and eight-slot frames written a little ahead of it. In the
/// middle of each stretch the guest also looks away for 30 ms, which the
/// read-ahead hold must carry: a card that read on would play the ring's
/// previous lap again. Both sides of the gap must be the tone with the
/// front pair on its own sides, as long as the engine moved, and the gap
/// must count as underrun. The tone is in slots 0 and 1, which the host
/// gets at unity as its front pair, and it drifts in pitch so that no
/// frame is the same as one a lap earlier; with an `[audio] file` the
/// frames heard must follow one another in the guest's order, none
/// replayed.
fn a_tone_survives_an_underrun(backend: crate::host_audio::test_signal::Backend) {
    use crate::host_audio::test_signal::*;
    use std::time::{Duration, Instant};
    let exact = matches!(backend, Backend::File);
    let Some(ear) = Listener::new(backend, "ice1724-underrun") else { return };
    const RATE: f64 = 48000.0;
    let (left, right) = (440.0, 880.0);
    let c = Ice1724::with_clock(Variant::Dm8, TestTime::clock());
    let ram = Arc::new(Ram(std::sync::Mutex::new(vec![0; 0x10_0000])));
    c.state().connect_dma(ram.clone());
    c.attached();
    map(&c);
    c.set_timer_manager(Arc::new(TimerManager::new()));
    c.enable_output(ear.cfg.clone());
    Device::start(&c);

    // Frame `n` of the tone is in slots 0 and 1 of a ring frame, each
    // sample in the top half of its dword; the other six slots are silent.
    // The pitch rises 2 Hz a second, well inside the 1% the check allows.
    let tone = |hz: f64, n: u64| {
        let t = n as f64 / RATE;
        (16000.0 * (std::f64::consts::TAU * (hz * t + t * t)).sin()).round() as i16
    };
    let fill = |written: &mut u64, upto: u64| {
        let mut m = ram.0.lock().unwrap();
        while *written < upto {
            let at = BUFFER as usize + (*written % IRIX_RING_FRAMES) as usize * 32;
            for (slot, hz) in [(0, left), (1, right)] {
                let v = u32::from(tone(hz, *written) as u16) << 16;
                m[at + slot * 4..at + slot * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            *written += 1;
        }
    };
    let position = |read: &mut u64| {
        let pos = u64::from((mt_rl(&c, 0x10) - BUFFER as u32) / 32);
        *read += (pos + IRIX_RING_FRAMES - *read % IRIX_RING_FRAMES) % IRIX_RING_FRAMES;
    };
    // The guest writes two read-aheads past the position it reads. The
    // card may run one read-ahead past that position before the write
    // lands, and the position is at most one read-ahead past the one read
    // before, whose write has landed: with two, every frame the card reads
    // is written, however long the host holds the guest up between its
    // read and its write. IRIX writes less ahead (see the module notes).
    let lead = 2 * (u128::from(READ_AHEAD_NS) * 48000 / NS_PER_SEC) as u64;
    // Eight slots to a frame, at the reset rate of 48 kHz: the host
    // stream takes the frames unresampled.
    mt_wb(&c, 0x19, 0);
    mt_wl(&c, 0x10, BUFFER as u32);
    mt_wl(&c, 0x14, IRIX_RING_FRAMES as u32 * 8 - 1);
    let (mut written, mut read) = (0u64, 0u64);
    // The hold lasts while reads come less than 50 ms apart; until the
    // first read after a start the card runs on its clock, through the lap
    // written before it. Each bound spans from before one read, or the
    // start, to after the next read, or the stop, so it covers the card's
    // own instants.
    let (mut longest_gap, mut slowest_start) = (Duration::ZERO, Duration::ZERO);
    // A stretch: write a lap from where the engine stopped, start it, and
    // for 0.6 s read the position and write ahead of it about every half
    // millisecond, looking away for 30 ms halfway; then stop it.
    let mut stretch = || {
        position(&mut read);
        fill(&mut written, read + IRIX_RING_FRAMES);
        let started = Instant::now();
        mt_wb(&c, 0x18, 0x01);
        let mut last: Option<Instant> = None;
        for half in 0..2 {
            let end = Instant::now() + Duration::from_millis(300);
            while Instant::now() < end {
                let before = Instant::now();
                position(&mut read);
                match last {
                    Some(t) => longest_gap = longest_gap.max(t.elapsed()),
                    None => slowest_start = slowest_start.max(started.elapsed()),
                }
                last = Some(before);
                fill(&mut written, read + lead);
                std::thread::sleep(Duration::from_micros(500));
            }
            if half == 0 {
                std::thread::sleep(Duration::from_millis(30));
            }
        }
        mt_wb(&c, 0x18, 0x00);
        longest_gap = longest_gap.max(last.map_or(Duration::ZERO, |t| t.elapsed()));
    };
    let moved = || c.stats().analog_frames as f64;
    let underruns = || c.sink.lock().as_ref().and_then(|s| s.host.as_ref()).map_or(0, |(h, _)| h.underruns.load(Ordering::Relaxed));
    stretch();
    let (first, before) = (moved(), underruns());
    std::thread::sleep(Duration::from_millis(300));
    let (paused, underran) = (moved(), underruns() - before);
    stretch();
    let second = moved() - paused;
    // What the host still holds plays out.
    std::thread::sleep(Duration::from_millis(300));
    Device::stop(&c);
    c.sink.lock().take();
    let heard = ear.heard();
    assert_a_tone_either_side_of_a_pause(&heard, left, right, 8000, [first / RATE, second / RATE]);
    assert!(underran > 0, "the host stream ran dry while the output was stopped");
    assert_eq!(paused, first, "the engine moved while it was stopped");
    // A host that kept the guest from reading for 40 ms lets the 50 ms
    // hold run out, or for 18 ms after a start lets the card through the
    // 21 ms lap written ahead less the 1.5 ms it may read past that first
    // read, as it would a real guest: the card then reads frames not yet
    // written, and the order check below would be a false alarm. So would
    // a host ring that dropped more frames than the check looks ahead.
    let dropped = c.stats().dropped_frames;
    let calm = longest_gap < Duration::from_millis(40) && slowest_start < Duration::from_millis(18) && dropped < 4800;
    if !calm {
        eprintln!("the guest was kept away for {longest_gap:?}, {slowest_start:?} after a start, {dropped} frames dropped: not checking the order of the frames");
    }
    if exact && calm {
        assert_eq!((heard.rate, heard.channels), (crate::host_audio::FILE_RATE, 2));
        // The frames played, silence left out, are the guest's frames in
        // order: each is later in the stream than the one before and no
        // more than 100 ms after it. A replayed lap goes back.
        let mut n = 0u64;
        for (i, f) in heard.pcm.chunks_exact(heard.channels).filter(|f| f.iter().any(|&v| v != 0)).enumerate() {
            let at = (n..n + 4800).find(|&k| f[0] == tone(left, k) && f[1] == tone(right, k));
            let Some(k) = at else { panic!("frame {i} of the sound ({}, {}) is not the guest's frame {n} or any of the next 4800", f[0], f[1]) };
            n = k + 1;
        }
        // Unity gain into the front pair: each side reaches the tone's peak.
        for c in 0..2 {
            assert!(peak(&channel(&heard.pcm, 2, c)) >= 15800, "channel {c} peak {}", peak(&channel(&heard.pcm, 2, c)));
        }
        assert!(n as f64 >= 0.5 * (first + second), "{n} of the {} frames the card moved were heard", first + second);
    }
}

#[test]
fn a_tone_survives_an_underrun_into_a_file() {
    a_tone_survives_an_underrun(crate::host_audio::test_signal::Backend::File);
}

#[test]
fn a_tone_survives_an_underrun_on_the_real_device() {
    a_tone_survives_an_underrun(crate::host_audio::test_signal::Backend::RealDevice);
}
