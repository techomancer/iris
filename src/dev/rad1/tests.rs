//! RAD1 register tests.
//!
//! Ported from tests/qtest/octane-test.c in jamesbraid/qemu-sgi-origami.

use super::*;
use crate::hptimer::TimerManager;
use crate::traits::Device;
use std::sync::{Arc, Mutex};

const BAR: u64 = 0x1300_0000;

/// A card with BAR0 assigned and memory decoding on, as firmware leaves it.
fn card() -> Rad1 {
    let r = Rad1::new();
    r.config().write(4, BAR as u32);
    r.config().write(1, PCI_COMMAND_MEMORY_ON);
    r.fake_ns.store(0, Ordering::Relaxed);
    r
}
const PCI_COMMAND_MEMORY_ON: u32 = 2;

fn wr(r: &Rad1, off: u32, v: u32) {
    r.bar_write(0, off, v, 4);
}
fn rd(r: &Rad1, off: u32) -> u32 {
    r.bar_read(0, off, 4)
}

const STARTUP: [(u32, u32); 28] = [
    (0x44, 0xa0000008), (0x48, 0x10), (0x4c, 0x03000000), (0x50, 0x60000000),
    (0x54, 0x4000000c), (0x70, 0x1440), (0x74, 0x08000010), (0x78, 0x00fac688),
    (0x7c, 0xd6d6d6d6), (0x80, 0xffe20000), (0x88, 3), (0x8c, 2), (0x90, 2),
    (0xa0, 0x02000001), (0xa4, 0x12345678), (0xa8, 1), (0xac, 2), (0xb0, 3),
    (0xb4, 4), (0xb8, 0x9fffffff), (0xbc, 0x9fffffff), (0x4c0, 0xa000c000),
    (0x4c8, 0xa000c800), (0x4d0, 0xa000d000), (0x4d8, 0xa000d800),
    (0x4e0, 0xa000e000), (0x4e8, 0xa000e800), (0x4f0, 0xa000f000),
];

#[test]
fn pci_identity() {
    let r = Rad1::new();
    let c = r.config();
    assert_eq!(c.read(0), 0x0005_10a9);
    assert_eq!(c.read(2) & 0xff, 0xc0);
    assert_eq!(c.read(2), 0xc0);
    // Identity, class and revision do not take writes.
    c.write(2, 0xffff_ffff);
    assert_eq!(c.read(2), 0xc0);
    c.write(2, 0x0401_0070);
    assert_eq!(c.read(2), 0xc0);
    assert_eq!(c.interrupt_pin(), 0);
    assert_eq!(c.read(4), 0);
}

#[test]
fn bar_sizing_and_assignment() {
    let r = Rad1::new();
    let c = r.config();
    c.write(4, 0xffff_ffff);
    assert_eq!(c.read(4), 0xffff_e000);
    c.write(4, BAR as u32);
    c.write(1, 2);
    assert_eq!(c.read(4), BAR as u32);
    assert_eq!(c.read(1) & 0xffff, 2);
    assert_eq!(c.decode(crate::pci::Space::Mem, BAR as u32 + 0x30), Some((0, 0x30)));
    assert_eq!(c.decode(crate::pci::Space::Mem, BAR as u32 + 0x2000), None);
}

#[test]
fn startup_words_store_and_reset() {
    let r = card();
    for (off, _) in STARTUP {
        assert_eq!(rd(&r, off), 0, "{off:#x} resets to zero");
    }
    // The bring-up and teardown sequence reads each write back.
    for (off, v) in [
        (0x80, 0xffffffff), (0x80, 0xffe3cffe), (0x80, 0xffe3c0fe), (0x80, 0xffe24070),
        (0x80, 0xffe20000), (0x70, 0x1500), (0x70, 0x1540), (0x70, 0x1440), (0x70, 0x1540),
        (0x88, 3), (0x88, 2), (0x88, 0), (0x7c, 0xc0c0c0c0), (0x70, 0x480), (0x80, 0xfffffffe),
        (0x80, 0xffffcbfc), (0x80, 0xffff8bfc), (0xa8, 0x13579bdf), (0xac, 0x2468ace0),
        (0xb0, 0x89abcdef), (0xb4, 0xfedcba98),
    ] {
        wr(&r, off, v);
        assert_eq!(rd(&r, off), v, "{off:#x}");
    }
    r.reset();
    wr(&r, 0x80, 0xffe20000);
    for (off, v) in STARTUP {
        // 0x4d0 and 0x4e0 are descriptor registers, covered separately.
        if ![0x4d0, 0x4e0].contains(&off) {
            wr(&r, off, v);
            assert_eq!(rd(&r, off), v, "{off:#x}");
        }
    }
    r.reset();
    assert_eq!(rd(&r, 0x80), 0);
    assert_eq!(rd(&r, 0x4c0), 0);
}

#[test]
fn startup_refuses_unsupported_enables() {
    let r = card();
    wr(&r, 0x44, 1);
    wr(&r, 0x44, 4);
    assert_eq!(rd(&r, 0x44), 0);
    wr(&r, 0x44, 0xa000_0008);
    assert_eq!(rd(&r, 0x44), 0xa000_0008);
    for off in [0x48, 0x50, 0x54] {
        wr(&r, off, 1);
        assert_eq!(rd(&r, off), 0, "{off:#x}");
    }
    // Analog input enable with no ring programmed.
    wr(&r, 0x4c, 1);
    assert_eq!(rd(&r, 0x4c), 0);
    // AES3 subcode enable likewise.
    wr(&r, 0x48, 4);
    assert_eq!(rd(&r, 0x48), 0);
}

#[test]
fn subcode_window() {
    let r = card();
    for off in (0x1000..=0x1218).step_by(4) {
        let masked = matches!(off & !7, 0x1078 | 0x10f8);
        if masked {
            assert_eq!(rd(&r, off), 0);
            wr(&r, off, 0xdead_beef);
            assert_eq!(rd(&r, off), 0, "{off:#x} is not part of the window");
        } else {
            wr(&r, off, off ^ 0x5a5a_a5a5);
            assert_eq!(rd(&r, off), off ^ 0x5a5a_a5a5, "{off:#x}");
        }
    }
    // The two halves of a doubleword are independent words.
    assert_eq!(rd(&r, 0x1004), 0x1004 ^ 0x5a5a_a5a5);
    r.reset();
    assert_eq!(rd(&r, 0x1000), 0);
    // Not a word access.
    wr(&r, 0x1002, 1);
    assert_eq!(r.bar_read(0, 0x1000, 2), 0);
}

#[test]
fn descriptor_bank_storage() {
    let r = card();
    for (off, v) in [(0x500, 0x1000), (0x528, 0), (0x504, 0xffff_fffc), (0x4b4, 0x1000)] {
        wr(&r, off, v);
        assert_eq!(rd(&r, off), v);
    }
    // Every descriptor RAM entry (0x400 + 12 i, 16 of them) is stored; past
    // them, words that belong to no supported ring are refused.
    for i in 0..16u32 {
        wr(&r, 0x408 + 12 * i, 0x100 + i);
        assert_eq!(rd(&r, 0x408 + 12 * i), 0x100 + i, "entry {i}");
    }
    wr(&r, 0x4c4, 7);
    assert_eq!(rd(&r, 0x4c4), 0);
    r.reset();
    assert_eq!(rd(&r, 0x500), 0);
    assert_eq!(rd(&r, 0x420), 0);
}

#[test]
fn ust_counter_and_control() {
    let r = card();
    let at = |ns: i64| r.fake_ns.store(ns, Ordering::Relaxed);
    assert_eq!(rd(&r, 0x30), 0);
    at(100_000);
    assert_eq!(rd(&r, 0x30), 0, "stopped at reset");
    // Control must be armed (0x0e) before it is started (0x0a).
    wr(&r, 0x40, 0x0a);
    at(200_000);
    assert_eq!(rd(&r, 0x30), 0);
    wr(&r, 0x40, 0x0e);
    wr(&r, 0x40, 0x0a);
    at(200_000 + 960 * 10 + 5);
    assert_eq!(rd(&r, 0x30), 10);
    at(200_000 + 960 * 20);
    assert_eq!(rd(&r, 0x30), 20, "phase carries");
    wr(&r, 0x40, 0x02);
    at(200_000 + 960 * 50);
    assert_eq!(rd(&r, 0x30), 20, "stopped");
    // Other control values are refused.
    wr(&r, 0x40, 0x0f);
    wr(&r, 0x40, 0x0a);
    at(200_000 + 960 * 60);
    assert_eq!(rd(&r, 0x30), 20);
    // The counter is read-only.
    wr(&r, 0x30, 0x1234);
    assert_eq!(rd(&r, 0x30), 20);
}

#[test]
fn ust_wraps_to_32_bits() {
    let r = card();
    r.state.lock().ust_count = u32::MAX - 1;
    wr(&r, 0x40, 0x0e);
    wr(&r, 0x40, 0x0a);
    r.fake_ns.store(960 * 3, Ordering::Relaxed);
    assert_eq!(rd(&r, 0x30), 1);
}

#[test]
fn position_registers_read_zero() {
    let r = card();
    for off in (0x04..=0x2c).step_by(4) {
        assert_eq!(rd(&r, off), 0);
        wr(&r, off, 0xffff_ffff);
        assert_eq!(rd(&r, off), 0, "{off:#x} is read-only");
    }
}

/// Program a ring (set A and B alike) the way the driver's channel start does.
fn program(r: &Rad1, ring: &RingRegs, base: u32, ctrl: u32) {
    for set in [0, 3] {
        wr(r, ring[set], base);
        wr(r, ring[set + 1], 0);
        wr(r, ring[set + 2], ctrl);
    }
}

#[test]
fn analog_output_enable_needs_a_matching_ring() {
    let r = card();
    // Disable is always accepted.
    wr(&r, 0x58, 0x2000_0000);
    assert_eq!(rd(&r, 0x58), 0x2000_0000);
    wr(&r, 0x58, 0x2000_0001);
    assert_eq!(rd(&r, 0x58), 0x2000_0000, "no ring programmed");
    // 4 KiB ring: 32 units of 128 bytes; length is stored negated at bit 7,
    // and the model's mask for this ring is 0xfffff008 plus the tag.
    program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    wr(&r, 0x58, 0x2000_0001);
    assert_eq!(rd(&r, 0x58), 0x2000_0001);
    assert!(r.state.lock().chan[CH_OUT].running);
    assert_eq!(r.state.lock().chan[CH_OUT].len, 0x1000);
    // The ring cannot be rewritten while it runs.
    wr(&r, 0x4f8, 0x20_0000);
    assert_eq!(rd(&r, 0x4f8), 0x10_0000);
    wr(&r, 0x58, 0x2000_0000);
    assert!(!r.state.lock().chan[CH_OUT].running);
    wr(&r, 0x4f8, 0x20_0000);
    assert_eq!(rd(&r, 0x4f8), 0x20_0000);
    // Values other than the two enables are refused.
    wr(&r, 0x58, 0x2000_0002);
    assert_eq!(rd(&r, 0x58), 0x2000_0000);
}

#[test]
fn analog_input_and_subcode_enable() {
    let r = card();
    program(&r, &RING_INPUT, 0x11_0000, 0xffff_f000);
    wr(&r, 0x4c, 0x0300_0001);
    assert_eq!(rd(&r, 0x4c), 0x0300_0001);
    assert!(r.state.lock().chan[CH_IN].running);
    wr(&r, 0x4c, 0x0300_0000);
    assert!(!r.state.lock().chan[CH_IN].running);

    program(&r, &RING_AES_SUB, 0x12_0000, 0xffff_f030);
    wr(&r, 0x48, 4);
    assert_eq!(rd(&r, 0x48), 4);
    assert!(r.state.lock().chan[CH_AES_SUB].running);
}

#[test]
fn status_dma_control() {
    let r = card();
    // Stop is always accepted.
    wr(&r, 0x5c, 0x1fffc);
    assert_eq!(rd(&r, 0x5c), 0x1fffc);
    // Start with no ring is refused.
    wr(&r, 0x5c, 0x40001);
    assert_eq!(rd(&r, 0x5c), 0x1fffc);
    wr(&r, 0x500, 0x30_0000);
    wr(&r, 0x504, 0xffff_fffc);
    wr(&r, 0x5c, 0x40001);
    assert_eq!(rd(&r, 0x5c), 0x40001, "one-shot");
    // Ring start: interval in bits 2..16, limit 0x78 minus slots * 0x80.
    wr(&r, 0x504, 0xffff_c078);
    wr(&r, 0x4b4, 0x30_0000);
    wr(&r, 0x4bc, 0xffff_c078);
    wr(&r, 0x5c, 0x40003 | (330 << 2));
    assert_eq!(rd(&r, 0x5c), 0x40003 | (330 << 2));
    assert!(r.state.lock().status_running);
    // Rewriting the limit while the ring runs is refused.
    wr(&r, 0x504, 0);
    assert_eq!(rd(&r, 0x504), 0xffff_c078);
    wr(&r, 0x5c, 0x1fffc);
    assert!(!r.state.lock().status_running);
    // A zero interval names no period.
    wr(&r, 0x5c, 0x40003);
    assert_ne!(rd(&r, 0x5c), 0x40003);
}

#[test]
fn unrecognised_accesses_read_zero_and_store_nothing() {
    let r = card();
    wr(&r, 0x34, 5);
    assert_eq!(rd(&r, 0x34), 0);
    wr(&r, 0x1ffc, 5);
    assert_eq!(rd(&r, 0x1ffc), 0);
    // Byte access to a start-up word is not recognised.
    r.bar_write(0, 0x80, 0xff, 1);
    assert_eq!(rd(&r, 0x80), 0);
}

// Engines and status DMA.

use crate::pci::PciDmaStatus;

/// Bus memory that is `data` from bus address 0, little-endian as the card
/// sees it.
struct Ram(Mutex<Vec<u8>>);

impl PciDma for Ram {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        let m = self.0.lock().unwrap();
        let a = addr as usize;
        match m.get(a..a + buf.len()) {
            Some(src) => {
                buf.copy_from_slice(src);
                PciDmaStatus::Ok
            }
            None => PciDmaStatus::MasterAbort,
        }
    }
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        let mut m = self.0.lock().unwrap();
        let a = addr as usize;
        match m.get_mut(a..a + data.len()) {
            Some(dst) => {
                dst.copy_from_slice(data);
                PciDmaStatus::Ok
            }
            None => PciDmaStatus::MasterAbort,
        }
    }
}

fn dma_card() -> (Rad1, Arc<Ram>) {
    let r = card();
    let ram = Arc::new(Ram(Mutex::new(vec![0xaa; 0x40_0000])));
    r.connect_dma(ram.clone());
    (r, ram)
}

fn word(ram: &Ram, addr: u64) -> u32 {
    let m = ram.0.lock().unwrap();
    u32::from_le_bytes(m[addr as usize..addr as usize + 4].try_into().unwrap())
}

fn at(r: &Rad1, ns: i64) {
    r.fake_ns.store(ns, Ordering::Relaxed);
}

fn start_ust(r: &Rad1) {
    wr(r, 0x40, 0x0e);
    wr(r, 0x40, 0x0a);
}

#[test]
fn one_shot_status_fills_a_slot() {
    let (r, ram) = dma_card();
    start_ust(&r);
    at(&r, 960 * 100);
    wr(&r, 0x500, 0x30_0000);
    wr(&r, 0x504, 0xffff_fffc);
    wr(&r, 0x5c, 0x40001);
    let slot = 0x30_0000;
    // UST mirror at +0x44 + (0x30 - 0x04).
    assert_eq!(word(&ram, slot + 0x40 + 0x30), 100);
    // Sync status and the unused position registers read zero.
    assert_eq!(word(&ram, slot + 0x40 + 0x34), 0);
    assert_eq!(word(&ram, slot + 0x40 + 0x04), 0);
    // No channel is running, so every pointer word is zero.
    for o in [0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38] {
        assert_eq!(word(&ram, slot + o), 0, "pointer {o:#x}");
    }
    assert_eq!(word(&ram, slot + 0x78), 0);
    assert_eq!(word(&ram, slot + 0x7c), 0x8008_0000);
}

#[test]
fn unmapped_status_ring_does_not_complete() {
    let (r, _ram) = dma_card();
    wr(&r, 0x500, 0xf000_0000);
    wr(&r, 0x504, 0xffff_fffc);
    wr(&r, 0x5c, 0x40001);
    assert_eq!(rd(&r, 0x5c), 0x40001);
}

#[test]
fn output_ring_is_consumed_at_the_sample_rate() {
    let (r, ram) = dma_card();
    start_ust(&r);
    program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    wr(&r, 0x58, 0x2000_0001);
    // 10 ms of 44.1 kHz is 441 frames.
    at(&r, 10_000_000);
    let b = rd(&r, 0x18);
    assert_eq!((b >> 4) & 0x3fff, 441, "output count in B");
    let a = rd(&r, 0x2c);
    assert_eq!(a >> 18, 441, "output count in A");
    // A's low bits are the UST at the 441st frame boundary.
    let boundary = (441u64 * 1_000_000_000).div_ceil(44100) as i64;
    assert_eq!(a & 0x3ffff, (boundary / 960) as u32);
    assert_eq!(rd(&r, 0x14), 0, "input untouched");
    // The status slot reports the pointer: 441 frames of 8 bytes in.
    wr(&r, 0x500, 0x30_0000);
    wr(&r, 0x504, 0xffff_fffc);
    wr(&r, 0x5c, 0x40001);
    assert_eq!(word(&ram, 0x30_0000 + 0x38), 0x10_0000 + 441 * 8);
    // Past the 512-frame ring the pointer wraps.
    at(&r, 20_000_000);
    assert_eq!(rd(&r, 0x2c) >> 18, 882);
    assert_eq!(r.state.lock().chan[CH_OUT].offset, (882 - 512) * 8);
    // Stopping holds the pointer where the sample clock had got to.
    at(&r, 30_000_000);
    wr(&r, 0x58, 0x2000_0000);
    at(&r, 40_000_000);
    assert_eq!(rd(&r, 0x2c) >> 18, 1323);
}

#[test]
fn output_ring_unmapped_stops_the_channel() {
    let (r, _ram) = dma_card();
    program(&r, &RING_OUTPUT, 0xf000_0000, 0xffff_f008);
    wr(&r, 0x58, 0x2000_0001);
    at(&r, 10_000_000);
    assert_eq!(rd(&r, 0x18), 0);
    assert!(!r.state.lock().chan[CH_OUT].running);
}

#[test]
fn input_ring_records_silence() {
    let (r, ram) = dma_card();
    program(&r, &RING_INPUT, 0x11_0000, 0xffff_f000);
    wr(&r, 0x4c, 1);
    at(&r, 1_000_000);
    // 44 frames: 352 bytes of silence, then the untouched fill.
    assert_eq!(rd(&r, 0x14) >> 18, 44);
    assert_eq!(rd(&r, 0x18) >> 18, 44);
    assert_eq!(word(&ram, 0x11_0000), 0);
    assert_eq!(word(&ram, 0x11_0000 + 351 - 3), 0);
    assert_eq!(word(&ram, 0x11_0000 + 352), 0xaaaa_aaaa);
}

/// Program clock generator `cg` as both drivers do: hold it in reset, pick
/// its master clock, write the remainder word, then the control word that
/// names it, and release it.
fn set_clock(r: &Rad1, cg: u32, mux: u32, rem: u32, ictl_low: u32) {
    wr(r, 0x80, 0xffe2_0000 | (0x100 << cg));
    wr(r, 0xa8 + 4 * (3 - cg), mux);
    wr(r, 0xa4, rem);
    wr(r, 0xa0, (0x1000_0000 << cg) | ictl_low);
    wr(r, 0x80, 0xffe2_0000);
}

/// Output frames counted after `ns` of a fresh output channel.
fn output_frames_after(r: &Rad1, ns: i64) -> u32 {
    program(r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    wr(r, 0x58, 0x2000_0001);
    at(r, ns);
    rd(r, 0x2c) >> 18
}

// The words IRIX 6.5.30 writes to the output's generator (1) for each rate,
// captured from the guest; all of them use the 192 kHz master (select 1).
const IRIX_DTOA_WORDS: [(u32, u32, u32); 7] = [
    (48000, 0x0000_ffff, 0x603),
    (44100, 0x0034_ff6d, 0x603),
    (32000, 0x0000_ffff, 0x605),
    (22050, 0x0068_ff6d, 0x207),
    (16000, 0x0000_ffff, 0x60b),
    (11025, 0x003d_ff6d, 0x610),
    (8000, 0x0000_ffff, 0x617),
];

#[test]
fn the_output_runs_at_its_clock_generator_rate() {
    for (hz, rem, ictl) in IRIX_DTOA_WORDS {
        let (r, _ram) = dma_card();
        start_ust(&r);
        set_clock(&r, 1, 1, rem, ictl);
        // 100 ms is a whole number of frames at every rate in the table.
        assert_eq!(output_frames_after(&r, 100_000_000), hz / 10, "{hz} Hz");
    }
}

#[test]
fn the_44k_master_divides_from_176400() {
    // The Linux driver's choice for 44.1 kHz rates: master 0, no remainder.
    let (r, _ram) = dma_card();
    set_clock(&r, 1, 0, 0x0000_ffff, 0x607);
    assert_eq!(output_frames_after(&r, 100_000_000), 2205);
}

#[test]
fn generator_0_clocks_the_input_and_not_the_output() {
    let (r, _ram) = dma_card();
    set_clock(&r, 0, 1, 0x0000_ffff, 0x617);
    program(&r, &RING_INPUT, 0x11_0000, 0xffff_f000);
    wr(&r, 0x4c, 1);
    assert_eq!(output_frames_after(&r, 100_000_000), 4410, "output keeps 44.1 kHz");
    assert_eq!(rd(&r, 0x14) >> 18, 800, "input at 8 kHz");
}

#[test]
fn a_control_word_naming_no_generator_changes_no_rate() {
    let (r, _ram) = dma_card();
    set_clock(&r, 1, 1, 0x0000_ffff, 0x617);
    // IRIX writes these between rate changes; no generator bit is set.
    wr(&r, 0xa4, 0x0068_ff6d);
    wr(&r, 0xa0, 0x0200_0004);
    assert_eq!(rd(&r, 0xa0), 0x0200_0004, "still stored");
    assert_eq!(output_frames_after(&r, 100_000_000), 800);
}

#[test]
fn a_rate_change_while_running_takes_effect_from_then() {
    let (r, _ram) = dma_card();
    start_ust(&r);
    assert_eq!(output_frames_after(&r, 100_000_000), 4410);
    set_clock(&r, 1, 1, 0x0068_ff6d, 0x207);
    at(&r, 200_000_000);
    assert_eq!(rd(&r, 0x2c) >> 18, 4410 + 2205);
    // A's UST stamp is the last frame's boundary at the new rate.
    let boundary = 100_000_000 + (2205u64 * 1_000_000_000).div_ceil(22050) as i64;
    assert_eq!(rd(&r, 0x2c) & 0x3ffff, (boundary / 960) as u32 & 0x3ffff);
}

#[test]
fn reset_returns_the_generators_to_the_default_rate() {
    let (r, _ram) = dma_card();
    set_clock(&r, 1, 1, 0x0000_ffff, 0x617);
    r.reset();
    assert_eq!(output_frames_after(&r, 100_000_000), 4410);
}

#[test]
fn the_wav_capture_stays_at_44100_whatever_the_card_rate() {
    let dir = std::env::temp_dir().join(format!("rad1-wav-rate-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("o.wav");
    let mut sink = Sink { host: None, wav: Wav::create(path.as_os_str()), tap: None };
    sink.play(&[0i16; 2 * 2205], 22050, false);
    sink.play(&[0i16; 2 * 4410], 44100, false);
    let b = std::fs::read(&path).unwrap();
    assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 44100);
    let frames = u32::from_le_bytes(b[40..44].try_into().unwrap()) / 4;
    assert!((8818..=8822).contains(&frames), "{frames} frames for 200 ms");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn aes_subcode_runs_at_250_hz_without_a_position_latch() {
    let (r, ram) = dma_card();
    program(&r, &RING_AES_SUB, 0x12_0000, 0xffff_f030);
    wr(&r, 0x48, 4);
    at(&r, 3_999_999);
    assert_eq!(rd(&r, 0x10), 0);
    at(&r, 8_000_000);
    // Two 256-byte records; the count sits at bit 4 of 0x10.
    assert_eq!((rd(&r, 0x10) >> 4) & 0x3fff, 2);
    assert_eq!(word(&ram, 0x12_0000 + 511 - 3), 0);
    assert_eq!(word(&ram, 0x12_0000 + 512), 0xaaaa_aaaa);
    // The status slot reports the pointer at +0x20.
    wr(&r, 0x500, 0x30_0000);
    wr(&r, 0x504, 0xffff_fffc);
    wr(&r, 0x5c, 0x40001);
    assert_eq!(word(&ram, 0x30_0000 + 0x20), 0x12_0000 + 512);
}

#[test]
fn continuous_status_fills_slots_in_ring_order() {
    let (r, ram) = dma_card();
    wr(&r, 0x500, 0x30_0000);
    wr(&r, 0x504, 0xffff_c078);
    wr(&r, 0x4b4, 0x30_0000);
    wr(&r, 0x4bc, 0xffff_c078);
    wr(&r, 0x5c, 0x40003 | (3300 << 2));
    let gen = r.state.lock().status_gen;
    assert_eq!(r.state.lock().status_slots, 128);
    // Interval 3300 at 33 MHz is 100 us.
    at(&r, 100_000);
    assert_eq!(r.status_tick(gen), Some(200_000));
    // The cursor starts one past the consumer index.
    assert_eq!(word(&ram, 0x30_0000 + 0x80 + 0x7c), 0x8008_0000);
    assert_eq!(word(&ram, 0x30_0000 + 0x7c), 0xaaaa_aaaa);
    at(&r, 200_000);
    assert_eq!(r.status_tick(gen), Some(300_000));
    assert_eq!(word(&ram, 0x30_0000 + 0x100 + 0x7c), 0x8008_0000);
    // Stopping ends the schedule.
    wr(&r, 0x5c, 0x1fffc);
    assert_eq!(r.status_tick(gen), None);
}

#[test]
fn status_schedule_restarts_when_far_behind() {
    let (r, _ram) = dma_card();
    wr(&r, 0x500, 0x30_0000);
    wr(&r, 0x504, 0xffff_c078);
    wr(&r, 0x4b4, 0x30_0000);
    wr(&r, 0x4bc, 0xffff_c078);
    wr(&r, 0x5c, 0x40003 | (3300 << 2));
    let gen = r.state.lock().status_gen;
    at(&r, 1_000_000_000);
    // A second late is a hold-up; 200 ms of it is held back and the rest
    // credited, so the schedule restarts from there.
    assert_eq!(r.status_tick(gen), Some(1_000_000_000 - 200_000_000 + 100_000));
}

#[test]
fn reset_ends_engines_and_status() {
    let (r, _ram) = dma_card();
    program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    wr(&r, 0x58, 0x2000_0001);
    wr(&r, 0x500, 0x30_0000);
    wr(&r, 0x504, 0xffff_c078);
    wr(&r, 0x4b4, 0x30_0000);
    wr(&r, 0x4bc, 0xffff_c078);
    wr(&r, 0x5c, 0x40003 | (3300 << 2));
    let gen = r.state.lock().status_gen;
    r.reset();
    assert_eq!(r.status_tick(gen), None);
    assert!(!r.state.lock().chan[CH_OUT].running);
    assert_eq!(rd(&r, 0x2c), 0);
}

/// Rings and status slots the guest based at the top of the 64-bit space
/// fail as unmapped ones do; nothing wraps or overflows.
#[test]
fn rings_at_the_top_of_the_bus_fail_rather_than_wrap() {
    let (r, _ram) = dma_card();
    for set in [0, 3] {
        wr(&r, RING_OUTPUT[set], 0xffff_f000);
        wr(&r, RING_OUTPUT[set + 1], 0xffff_ffff);
        wr(&r, RING_OUTPUT[set + 2], 0xffff_f008);
    }
    wr(&r, 0x58, 0x2000_0001);
    at(&r, 10_000_000);
    assert_eq!(rd(&r, 0x2c), 0);
    assert!(!r.state.lock().chan[CH_OUT].running);

    // A one-shot status slot whose words run past the top.
    wr(&r, 0x500, 0xffff_ffc0);
    wr(&r, 0x528, 0xffff_ffff);
    wr(&r, 0x504, 0xffff_fffc);
    wr(&r, 0x5c, 0x40001);

    // A continuous ring whose later slots would wrap.
    wr(&r, 0x504, 0xffff_c078);
    wr(&r, 0x4b4, 0xffff_ffc0);
    wr(&r, 0x4b8, 0xffff_ffff);
    wr(&r, 0x4bc, 0xffff_c078);
    wr(&r, 0x5c, 0x40003 | (3300 << 2));
    let gen = r.state.lock().status_gen;
    at(&r, 20_000_000);
    assert_eq!(r.status_tick(gen), None, "the schedule stops on the failed slot");
}

/// While the machine is stopped the card's clock stands still: a ring
/// does not jump on by the length of the stop.
#[test]
fn the_clock_stands_still_while_paused() {
    let (r, _ram) = dma_card();
    r.fake_ns.store(i64::MIN, Ordering::Relaxed);
    start_ust(&r);
    r.pause();
    let ust = rd(&r, 0x30);
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(rd(&r, 0x30), ust, "no UST ticks while paused");
    r.resume();
    std::thread::sleep(Duration::from_millis(2));
    let ticks = rd(&r, 0x30).wrapping_sub(ust);
    // About 2 ms of 960 ns ticks, short of the 22 ms slept in all.
    assert!(ticks > 0 && ticks < 20_000, "{ticks} ticks");
}

/// A card on the real clock with the machine's timers, started, and a
/// status ring programmed; the schedule is not started yet.
fn timed_card() -> (Rad1, Arc<Ram>, Arc<TimerManager>) {
    let (r, ram) = dma_card();
    r.fake_ns.store(i64::MIN, Ordering::Relaxed);
    let tm = Arc::new(TimerManager::new());
    r.set_timer_manager(tm.clone());
    Device::start(&r);
    wr(&r, 0x500, 0x30_0000);
    wr(&r, 0x504, 0xffff_c078);
    wr(&r, 0x4b4, 0x30_0000);
    wr(&r, 0x4bc, 0xffff_c078);
    (r, ram, tm)
}

/// Poll the schedule's own counter, which no guest register read touches.
fn wait_done(r: &Rad1, n: u64) -> bool {
    for _ in 0..500 {
        if r.state.lock().status_done >= n {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

/// A started status schedule delivers completions from the timer with no
/// guest access: the first slot the driver checks is written to memory.
#[test]
fn a_status_completion_fires_from_the_timer_with_no_guest_access() {
    let (r, ram, _tm) = timed_card();
    // 3300 units of the 33 MHz status clock: 100 us.
    wr(&r, 0x5c, 0x40003 | (3300 << 2));
    assert!(r.state.lock().status_timer.is_some());
    assert!(wait_done(&r, 2), "completions arrive unprompted");
    assert_eq!(word(&ram, 0x30_0000 + SLOT_STRIDE + SLOT_DONE_OFFSET), SLOT_DONE_WORD);
    Device::stop(&r);
}

/// Stopping the machine removes the timer and nothing completes until the
/// restart, which picks the schedule up where the card's clock stood.
#[test]
fn stopping_ends_the_status_timer_and_starting_resumes_it() {
    let (r, _ram, _tm) = timed_card();
    wr(&r, 0x5c, 0x40003 | (3300 << 2));
    assert!(wait_done(&r, 1));
    Device::stop(&r);
    assert!(!r.is_running());
    assert!(r.state.lock().status_timer.is_none());
    let done = r.state.lock().status_done;
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(r.state.lock().status_done, done, "nothing completes while stopped");
    Device::start(&r);
    assert!(wait_done(&r, done + 1), "the schedule runs again");
    Device::stop(&r);
}

/// Stopping the status engine removes its timer.
#[test]
fn stopping_the_status_engine_removes_its_timer() {
    let (r, _ram, _tm) = timed_card();
    wr(&r, 0x5c, 0x40003 | (3300 << 2));
    assert!(r.state.lock().status_timer.is_some());
    wr(&r, 0x5c, STATUS_STOP);
    assert!(r.state.lock().status_timer.is_none());
    Device::stop(&r);
}

/// The output pump runs only while the analog output does, and only with
/// host audio enabled.
#[test]
fn the_output_pump_runs_only_while_the_channel_does() {
    let (r, _ram, _tm) = timed_card();
    program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    wr(&r, 0x58, 0x2000_0001);
    assert!(r.state.lock().audio_timer.is_none(), "no host audio, no pump");
    wr(&r, 0x58, 0x2000_0000);
    r.enable_output(crate::config::AudioConfig::default());
    wr(&r, 0x58, 0x2000_0001);
    assert!(r.state.lock().audio_timer.is_some());
    // The pump moves frames the guest never asked about.
    std::thread::sleep(Duration::from_millis(40));
    assert!(r.state.lock().chan[CH_OUT].frames > 0);
    Device::stop(&r);
    assert!(r.state.lock().audio_timer.is_none(), "stop removes it");
    Device::start(&r);
    assert!(r.state.lock().audio_timer.is_some(), "start re-arms it");
    wr(&r, 0x58, 0x2000_0000);
    assert!(r.state.lock().audio_timer.is_none());
    Device::stop(&r);
}

/// Counts the transfers it passes on.
struct Counting(Arc<Ram>, std::sync::atomic::AtomicUsize);

impl PciDma for Counting {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        self.1.fetch_add(1, Ordering::Relaxed);
        self.0.dma_read(addr, buf)
    }
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        self.1.fetch_add(1, Ordering::Relaxed);
        self.0.dma_write(addr, data)
    }
}

/// A position read after a long look-away does not replay every missed
/// revolution of the ring, and still lands where the sample clock says.
#[test]
fn a_long_catch_up_skips_whole_revolutions() {
    let r = card();
    let ram = Arc::new(Ram(Mutex::new(vec![0xaa; 0x40_0000])));
    let dma = Arc::new(Counting(ram.clone(), std::sync::atomic::AtomicUsize::new(0)));
    r.connect_dma(dma.clone());
    program(&r, &RING_INPUT, 0x11_0000, 0xffff_f000);
    wr(&r, 0x4c, 1);
    // 1000 s of 44.1 kHz: 44.1 million frames of 8 bytes into a 4 KiB ring.
    at(&r, 1_000_000_000_000);
    let due: u64 = 44_100_000;
    assert_eq!(u64::from(rd(&r, 0x18) >> 18), due & 0x3fff);
    let c = r.state.lock().chan[CH_IN];
    assert_eq!(c.frames, due);
    assert_eq!(c.offset, (due * 8) % c.len);
    assert!(dma.1.load(Ordering::Relaxed) < 16, "{} transfers", dma.1.load(Ordering::Relaxed));
    assert_eq!(word(&ram, 0x11_0000 + 0x800), 0, "the whole ring is silence");
}

// PCM conversion and host output.

#[test]
fn sample_word_conversions() {
    // A 16-bit sample is bits [23:8], sign-extended through [31:24].
    assert_eq!(sample_to_word(0), 0);
    assert_eq!(sample_to_word(1), 0x0000_0100);
    assert_eq!(sample_to_word(0x7fff), 0x007f_ff00);
    assert_eq!(sample_to_word(-1), 0xffff_ff00);
    assert_eq!(sample_to_word(i16::MIN), 0xff80_0000);
    assert_eq!(word_to_sample(0x007f_ff00), 0x7fff);
    assert_eq!(word_to_sample(0xff80_0000), i16::MIN);
    // The low byte is dropped, the sign-extension bits ignored.
    assert_eq!(word_to_sample(0x0012_34ff), 0x1234);
    assert_eq!(word_to_sample(0x0112_3400), 0x1234);
    for s in [i16::MIN, -12345, -1, 0, 1, 321, i16::MAX] {
        assert_eq!(word_to_sample(sample_to_word(s)), s);
    }
}

#[test]
fn ring_frames_decode_left_first_and_drop_extra_words() {
    let words = [0x0000_1100u32, 0xffff_ff00, 0x0000_2200, 0x0000_3300];
    let ring: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut pcm = [0i16; 2];
    frames_to_pcm(&ring[..8], &mut pcm, 2);
    assert_eq!(pcm, [0x11, -1], "left word first");
    // Eight-word-style frames: words past the second are dropped.
    let mut pcm = [0i16; 2];
    frames_to_pcm(&ring, &mut pcm, 4);
    assert_eq!(pcm, [0x11, -1]);
    // Two frames of two words.
    let mut pcm = [0i16; 4];
    frames_to_pcm(&ring, &mut pcm, 2);
    assert_eq!(pcm, [0x11, -1, 0x22, 0x33]);
}

#[test]
fn host_frames_encode_into_ring_frames() {
    let pcm = [0x1234i16, -2, 7, i16::MIN];
    let mut ring = [0xaau8; 16];
    pcm_to_frames(&pcm, &mut ring, 2);
    let w: Vec<u32> = ring.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(w, [0x0012_3400, 0xffff_fe00, 0x0000_0700, 0xff80_0000]);
    // Words past the second record silence.
    let mut ring = [0xaau8; 16];
    pcm_to_frames(&pcm[..2], &mut ring, 4);
    let w: Vec<u32> = ring.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(w, [0x0012_3400, 0xffff_fe00, 0, 0]);
}

#[test]
fn output_ring_reaches_the_sink() {
    let (r, ram) = dma_card();
    let tap = Arc::new(parking_lot::Mutex::new(Vec::new()));
    *r.sink.lock() = Some(Sink { host: None, wav: None, tap: Some(tap.clone()) });
    // A 512-frame ring (4 KiB), each frame left = n, right = -n.
    {
        let mut m = ram.0.lock().unwrap();
        for n in 0..512usize {
            let at = 0x10_0000 + n * 8;
            m[at..at + 4].copy_from_slice(&sample_to_word(n as i16).to_le_bytes());
            m[at + 4..at + 8].copy_from_slice(&sample_to_word(-(n as i16)).to_le_bytes());
        }
    }
    start_ust(&r);
    program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    wr(&r, 0x58, 0x2000_0001);
    // 1 ms of 44.1 kHz is 44 frames.
    at(&r, 1_000_000);
    rd(&r, 0x18);
    {
        let t = tap.lock();
        assert_eq!(t.len(), 44 * 2);
        for n in 0..44usize {
            assert_eq!(t[n * 2], n as i16);
            assert_eq!(t[n * 2 + 1], -(n as i16));
        }
    }
    // Past the ring's end the stream wraps to its first frame.
    at(&r, 12_000_000);
    rd(&r, 0x18);
    let t = tap.lock();
    assert_eq!(t.len(), 529 * 2);
    assert_eq!(t[511 * 2], 511);
    assert_eq!(t[512 * 2], 0, "frame 512 is ring frame 0");
    assert_eq!(t[513 * 2], 1);
}

#[test]
fn input_ring_is_not_played() {
    let (r, _ram) = dma_card();
    let tap = Arc::new(parking_lot::Mutex::new(Vec::new()));
    *r.sink.lock() = Some(Sink { host: None, wav: None, tap: Some(tap.clone()) });
    program(&r, &RING_INPUT, 0x11_0000, 0xffff_f000);
    wr(&r, 0x4c, 1);
    at(&r, 1_000_000);
    rd(&r, 0x14);
    assert!(tap.lock().is_empty());
}

#[test]
fn wav_file_has_a_valid_header_and_data() {
    let dir = std::env::temp_dir().join(format!("rad1-wav-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("o.wav");
    let mut w = Wav::create(path.as_os_str()).unwrap();
    w.append(&[1, -1, 2, -2]);
    let b = std::fs::read(&path).unwrap();
    assert_eq!(&b[..4], b"RIFF");
    assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 36 + 8);
    assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 44100);
    assert_eq!(u32::from_le_bytes(b[40..44].try_into().unwrap()), 8);
    assert_eq!(i16::from_le_bytes(b[46..48].try_into().unwrap()), -1);
    assert_eq!(b.len(), 52);
    let _ = std::fs::remove_dir_all(&dir);
}

// Cases from the QEMU PCM conversion unit test: two frames with opposite-sign
// channels expose bit position, sign, channel and stride errors.

const ADAT_WORDS: usize = 8;
const AES_SUB_WORDS: usize = 64;
const ADAT_SUB_WORDS: usize = 32;

#[test]
fn word_and_sample_conversions_match_the_reference_values() {
    assert_eq!(word_to_sample(0x0012_3456), 0x1234);
    assert_eq!(word_to_sample(0xffab_cd12), 0xabcdu16 as i16);
    assert_eq!(word_to_sample(0xff80_0000), 0x8000u16 as i16);
    assert_eq!(word_to_sample(0x007f_ffff), 0x7fff);
    assert_eq!(sample_to_word(0x1234), 0x0012_3400);
    assert_eq!(sample_to_word(0xabcdu16 as i16), 0xffab_cd00);
    assert_eq!(sample_to_word(0x8000u16 as i16), 0xff80_0000);
    assert_eq!(sample_to_word(0x7fff), 0x007f_ff00);
    assert_eq!(sample_to_word(-1), 0xffff_ff00);
    assert_eq!(sample_to_word(0), 0);
}

/// Two ring frames of `words` words; word 0 is left, word 1 right, the rest distinct.
fn reference_ring(words: usize) -> Vec<u8> {
    let left = [0x0012_3456u32, 0xff87_6543];
    let right = [0xffab_cd12u32, 0x0034_5678];
    let mut ring = Vec::new();
    for f in 0..2 {
        for w in 0..words {
            let v = match w {
                0 => left[f],
                1 => right[f],
                _ => 0x0011_1100u32 * (w as u32 + 1),
            };
            ring.extend_from_slice(&v.to_le_bytes());
        }
    }
    ring
}

#[test]
fn reference_frames_decode_for_analog_and_adat() {
    let want = [0x1234, 0xabcdu16 as i16, 0x8765u16 as i16, 0x3456];
    for words in [2, ADAT_WORDS] {
        let mut pcm = [0i16; 4];
        frames_to_pcm(&reference_ring(words), &mut pcm, words);
        // Channels 1 and 2 reach the host, ADAT channels 3 to 8 are dropped.
        assert_eq!(pcm, want, "{words} words");
    }
}

#[test]
fn reference_host_frames_encode_for_every_frame_size() {
    let pcm = [0x1234i16, 0xabcdu16 as i16, 0x8000u16 as i16, 0x7fff];
    let words = [[0x0012_3400u32, 0xffab_cd00], [0xff80_0000, 0x007f_ff00]];
    for n in [2, ADAT_WORDS, ADAT_SUB_WORDS, AES_SUB_WORDS] {
        let mut ring = vec![0xa5u8; 3 * AES_SUB_WORDS * 4];
        pcm_to_frames(&pcm, &mut ring, n);
        let w = |i: usize| u32::from_le_bytes(ring[i * 4..i * 4 + 4].try_into().unwrap());
        for f in 0..2 {
            for k in 0..n {
                let want = if k < 2 { words[f][k] } else { 0 };
                assert_eq!(w(f * n + k), want, "{n} words, frame {f} word {k}");
            }
        }
        assert_eq!(w(2 * n), 0xa5a5_a5a5, "nothing past the two frames is written");
    }
}

#[test]
fn a_recorded_adat_frame_plays_back_unchanged() {
    let pcm = [0x1234i16, 0xabcdu16 as i16, 0x8000u16 as i16, 0x7fff];
    let mut ring = vec![0u8; 2 * ADAT_WORDS * 4];
    pcm_to_frames(&pcm, &mut ring, ADAT_WORDS);
    let mut back = [0i16; 4];
    frames_to_pcm(&ring, &mut back, ADAT_WORDS);
    assert_eq!(back, pcm);
}

const SEC: i64 = 1_000_000_000;

fn tapped(r: &Rad1) -> Arc<parking_lot::Mutex<Vec<i16>>> {
    let tap = Arc::new(parking_lot::Mutex::new(Vec::new()));
    *r.sink.lock() = Some(Sink { host: None, wav: None, tap: Some(tap.clone()) });
    tap
}

#[test]
fn the_output_follows_the_generator_0x58_names() {
    // Generator 2 is 32 kHz from the 192 kHz master and 29.4 kHz from the
    // default one. The Octane PROM loads it, then selects the master.
    let (r, _ram) = dma_card();
    let tap = tapped(&r);
    wr(&r, 0xa4, 0xffff);
    wr(&r, 0xa0, 0x4000_0405);
    wr(&r, 0xac, 1);
    program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    wr(&r, 0x58, 0x2000_0001);
    at(&r, SEC / 100);
    rd(&r, 0x18);
    assert_eq!(tap.lock().len(), 441 * 2, "generator 1 stays at 44.1 kHz");
    // Switching to generator 2 while running counts from then at 32 kHz.
    wr(&r, 0x58, 0x4000_0001);
    at(&r, 2 * SEC / 100);
    rd(&r, 0x18);
    assert_eq!(tap.lock().len(), (441 + 320) * 2);
}

#[test]
fn analog_output_enable_takes_only_the_two_known_selects() {
    let r = card();
    for v in [0x4000_0000, 0x2000_0000] {
        wr(&r, 0x58, v);
        assert_eq!(rd(&r, 0x58), v);
    }
    wr(&r, 0x58, 0x4000_0000);
    for v in [0x6000_0000, 0x0000_0000, 0x4000_0002] {
        wr(&r, 0x58, v);
        assert_eq!(rd(&r, 0x58), 0x4000_0000, "{v:#x} is unsupported");
    }
    wr(&r, 0x58, 0x4000_0001);
    assert_eq!(rd(&r, 0x58), 0x4000_0000, "no ring programmed");
}

// The Octane PROM's boot tune: one end-of-descriptor buffer on generator 2.

const TUNE_BUF: u64 = 0x10_0000;
/// 4504 units of 128 bytes (72,064 frames), end of descriptor set.
const TUNE_CTRL: u32 = 0xfff7_3404;
const TUNE_FRAMES: usize = 72_064;

/// A card with a tap on what it plays.
fn tune_card() -> (Rad1, Arc<Ram>, Arc<parking_lot::Mutex<Vec<i16>>>) {
    let (r, ram) = dma_card();
    let tap = tapped(&r);
    (r, ram, tap)
}

/// The PROM's writes that matter: load generator 2, select the master after
/// the load, program the buffer and start.
fn prom_tune(r: &Rad1, ctrl: u32) {
    wr(r, 0xa4, 0x0000_ffff);
    wr(r, 0x58, 0x4000_0000);
    wr(r, 0xa0, 0x4000_0405);
    wr(r, 0xac, 1);
    wr(r, 0x4c, 0x0300_0000);
    wr(r, 0x524, 0);
    wr(r, 0x4f8, TUNE_BUF as u32);
    wr(r, 0x4fc, ctrl);
    wr(r, 0x58, 0x4000_0001);
}

#[test]
fn the_prom_tune_plays_once_at_32khz_and_clears_the_enable_bit() {
    let (r, _ram, tap) = tune_card();
    prom_tune(&r, TUNE_CTRL);
    at(&r, SEC);
    assert_eq!(rd(&r, 0x58), 0x4000_0001, "still busy");
    assert_eq!(tap.lock().len(), 32_000 * 2);
    // One pass is 2.252 s.
    at(&r, 2_240 * SEC / 1000);
    assert_eq!(rd(&r, 0x58) & 1, 1);
    at(&r, 2_260 * SEC / 1000);
    assert_eq!(rd(&r, 0x58), 0x4000_0000, "the channel cleared the bit itself");
    at(&r, 10 * SEC);
    rd(&r, 0x58);
    assert_eq!(tap.lock().len(), TUNE_FRAMES * 2, "exactly one pass");
}

#[test]
fn a_link_to_an_unprogrammed_descriptor_plays_one_lap_and_stops() {
    let (r, _ram, tap) = tune_card();
    // Bit 2 clear, next index 0, and descriptor RAM entry 0 never written.
    prom_tune(&r, 0xfff7_3400);
    at(&r, 10 * SEC);
    rd(&r, 0x58);
    assert_eq!(tap.lock().len(), TUNE_FRAMES * 2, "the first lap only");
    assert_eq!(rd(&r, 0x58) & 1, 1, "nothing is reported as finished");
    assert!(r.state.lock().chan[CH_OUT].stalled);
}

#[test]
fn a_link_loads_the_descriptor_ram_entry_it_names() {
    for entry in [1u32, 3, 7] {
        let (r, _ram, tap) = tune_card();
        // The first 512 frames link to `entry`, which holds 256 frames
        // elsewhere with end of descriptor.
        for (i, v) in [0x20_0000, 0, 0xffff_f804].into_iter().enumerate() {
            wr(&r, 0x400 + 12 * entry + 4 * i as u32, v);
        }
        prom_tune(&r, 0xffff_f000 | entry << 3);
        at(&r, SEC);
        assert_eq!(rd(&r, 0x58), 0x4000_0000, "entry {entry} ended the transfer");
        assert_eq!(tap.lock().len(), (512 + 256) * 2, "entry {entry}");
        assert!(!r.state.lock().chan[CH_OUT].stalled);
    }
}

#[test]
fn the_relay_and_dac_mute_words_silence_the_output_but_not_the_transfer() {
    let path = std::env::temp_dir().join(format!("rad1-mute-{}.wav", std::process::id()));
    // The relay (0x88) and the DAC (0x8c): the word that mutes, the word that releases.
    for (reg, mute, release) in [(0x88, 2, 3), (0x8c, 3, 2)] {
        let (r, ram, tap) = tune_card();
        ram.0.lock().unwrap()[0x10_0000..0x10_1000].fill(0x55);
        r.sink.lock().as_mut().unwrap().wav = Wav::create(path.as_os_str());
        program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
        wr(&r, 0x58, 0x2000_0001);
        let wav_at = |ms: i64| {
            at(&r, ms * SEC / 1000);
            rd(&r, 0x58);
            std::fs::read(&path).unwrap()
        };
        let a = wav_at(10);
        assert!(a[44..].iter().any(|&b| b != 0), "not muted at power-on");
        wr(&r, reg, mute);
        let b = wav_at(20);
        assert!(b.len() > a.len() && b[a.len()..].iter().all(|&x| x == 0), "{reg:#x} silences");
        wr(&r, reg, release);
        let c = wav_at(30);
        assert!(c[b.len()..].iter().any(|&x| x != 0), "{reg:#x} releases");
        assert!(r.state.lock().chan[CH_OUT].running && tap.lock().len() > 2 * 441 * 2);
    }
    let _ = std::fs::remove_file(&path);
}

/// Completion `k` of IRIX's continuous status schedule (interval 8251 at
/// 33 MHz, 250 us) is due at this time.
fn irix_due(k: i64) -> i64 {
    k * 8251 * 1_000_000_000 / 33_000_000
}

/// The time held back from the card's clock for timer hold-ups.
fn held(r: &Rad1) -> i64 {
    r.stalled_ns.load(Ordering::SeqCst)
}

/// Output running on a 512-frame ring with that status schedule started at
/// time 0; returns the schedule's generation.
fn running_with_status(r: &Rad1) -> u64 {
    start_ust(r);
    program(r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    wr(r, 0x58, 0x2000_0001);
    wr(r, 0x500, 0x30_0000);
    wr(r, 0x504, 0xffff_c078);
    wr(r, 0x4b4, 0x30_0000);
    wr(r, 0x4bc, 0xffff_c078);
    wr(r, 0x5c, 0x480ef);
    r.state.lock().status_gen
}

/// IRIX writes only 36 to 90 frames ahead of the pointer, so a completion
/// the host held up must not carry the output through the hold-up.
#[test]
fn a_late_status_completion_does_not_carry_the_output_through_the_hold_up() {
    let (r, _ram) = dma_card();
    let gen = running_with_status(&r);
    at(&r, irix_due(1) + 5_000_000);
    assert_eq!(r.status_tick(gen), Some(irix_due(2)), "the schedule carries on from the card's clock");
    assert!(r.state.lock().chan[CH_OUT].frames <= 12, "moved over the hold-up");
    assert_eq!(r.now(), irix_due(1), "the clock is back at the due time");
    assert_eq!(held(&r), 5_000_000);
    // An on-time completion wins back 1/32 of an interval.
    at(&r, irix_due(2) + 5_000_000);
    assert_eq!(r.status_tick(gen), Some(irix_due(3)));
    assert_eq!(held(&r), 5_000_000 - irix_due(1) / 32);
}

#[test]
fn a_hold_up_is_more_than_one_interval_late_and_at_most_200ms_is_held_back() {
    let interval = irix_due(1);
    for (late, held_back) in [(interval, 0), (interval + 1_000, interval + 1_000), (SEC, 200_000_000)] {
        let (r, _ram) = dma_card();
        let gen = running_with_status(&r);
        at(&r, irix_due(1) + late);
        r.status_tick(gen).unwrap();
        assert_eq!(held(&r), held_back, "{late} ns late");
    }
}
