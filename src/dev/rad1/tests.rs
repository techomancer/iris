//! RAD1 register tests.
//!
//! Ported from tests/qtest/octane-test.c in jamesbraid/qemu-sgi-origami.

use super::*;
use crate::dev::host_pcm::test_time::{FakeNs, TestTime};
use crate::host_audio::test_signal::{a_tone_survives_a_pause, tones, CardUnderTest};
use crate::host_audio::{HostLink, HOST_RETRY};
use crate::hptimer::TimerManager;
use crate::pci::PciDma;
use crate::traits::Device;
use std::sync::{Arc, Mutex};

const BAR: u64 = 0x1300_0000;

/// A card with BAR0 assigned and memory decoding on, as firmware leaves it.
fn card() -> Rad1 {
    let r = Rad1::with_clock(TestTime::clock());
    r.state().config().write(4, BAR as u32);
    r.state().config().write(1, PCI_COMMAND_MEMORY_ON);
    r.fake_ns().store(0, Ordering::Relaxed);
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
    let c = r.state().config();
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
    let c = r.state().config();
    c.write(4, 0xffff_ffff);
    assert_eq!(c.read(4), 0xffff_e000);
    c.write(4, BAR as u32);
    c.write(1, 2);
    assert_eq!(c.read(4), BAR as u32);
    assert_eq!(c.read(1) & 0xffff, 2);
    assert_eq!(c.bar_regions(), vec![(crate::pci::Space::Mem, 0, BAR as u32, BAR0_SIZE)]);
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
    let at = |ns: i64| r.fake_ns().store(ns, Ordering::Relaxed);
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
    r.fake_ns().store(960 * 3, Ordering::Relaxed);
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

use crate::pci::testing::FlatMem as Ram;

fn dma_card() -> (Rad1, Arc<Ram>) {
    let r = card();
    let ram = Arc::new(Ram(Mutex::new(vec![0xaa; 0x40_0000])));
    r.state().connect_dma(ram.clone());
    (r, ram)
}

fn word(ram: &Ram, addr: u64) -> u32 {
    let m = ram.0.lock().unwrap();
    u32::from_le_bytes(m[addr as usize..addr as usize + 4].try_into().unwrap())
}

fn at(r: &Rad1, ns: i64) {
    r.fake_ns().store(ns, Ordering::Relaxed);
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
fn the_dump_records_each_burst_at_the_card_rate() {
    let path = crate::dev::audio_dump::tests::temp("rad1");
    let dump = crate::dev::audio_dump::AudioDump::create(path.as_os_str()).map(Arc::new);
    let mut sink = Sink { host: HostLink::default(), dump, tap: None };
    sink.play(&[1, -2, 300, -400], 44100, false);
    sink.play(&[i16::MIN, i16::MAX], 32000, false);
    // A muted card plays silence, and the dump records what is played.
    sink.play(&[5, 6], 32000, true);
    let recs = crate::dev::audio_dump::tests::records(&std::fs::read(&path).unwrap());
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        recs,
        [(44100, vec![(1, -2), (300, -400)]), (32000, vec![(i16::MIN, i16::MAX)]), (32000, vec![(0, 0)])]
    );
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
    // The 10 ms the clock made due went nowhere: counted, not moved.
    assert_eq!((r.meter.frames(CH_OUT), r.meter.skips(CH_OUT, Skip::Dry)), (0, 441));

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
    r.fake_ns().store(i64::MIN, Ordering::Relaxed);
    start_ust(&r);
    r.pause();
    let before = std::time::Instant::now();
    let ust = rd(&r, 0x30);
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(rd(&r, 0x30), ust, "no UST ticks while paused");
    r.resume();
    let mut ticks = 0;
    assert!(eventually(|| {
        ticks = rd(&r, 0x30).wrapping_sub(ust);
        ticks > 0
    }), "the clock runs again after the resume");
    // The card's time since the first read cannot include the 20 ms slept
    // stopped, however long the host took to get here: the host's own span
    // less that sleep bounds it. (A bound on the wall clock alone fails
    // whenever a loaded host oversleeps.)
    let host_ns = before.elapsed().as_nanos() as u64 - 20_000_000;
    assert!(u64::from(ticks) * 960 <= host_ns + 960, "{ticks} ticks in {host_ns} ns of running time");
}

/// A card on the real clock with the machine's timers, started, and a
/// status ring programmed; the schedule is not started yet.
fn timed_card() -> (Rad1, Arc<Ram>, Arc<TimerManager>) {
    let (r, ram) = dma_card();
    r.fake_ns().store(i64::MIN, Ordering::Relaxed);
    let tm = Arc::new(TimerManager::new());
    r.set_timer_manager(tm.clone());
    Device::start(&r);
    wr(&r, 0x500, 0x30_0000);
    wr(&r, 0x504, 0xffff_c078);
    wr(&r, 0x4b4, 0x30_0000);
    wr(&r, 0x4bc, 0xffff_c078);
    (r, ram, tm)
}

/// Poll `done` until it holds, giving up after a minute. Timer callbacks run
/// on another thread and the suite runs many cards at once, so a wait counted
/// in sleeps or milliseconds gives up on a busy host before the callback has
/// had a core; the deadline only has to catch a callback that never comes.
fn eventually(mut done: impl FnMut() -> bool) -> bool {
    let end = std::time::Instant::now() + Duration::from_secs(60);
    while !done() {
        if std::time::Instant::now() >= end {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

/// Poll the schedule's own counter, which no guest register read touches.
fn wait_done(r: &Rad1, n: u64) -> bool {
    eventually(|| r.state.lock().status_done >= n)
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
    assert!(eventually(|| r.state.lock().chan[CH_OUT].frames > 0), "the pump never ran");
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
    r.state().connect_dma(dma.clone());
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
    // The revolutions not walked are skipped, the rest moved: together the
    // frames the clock made due.
    let skipped = r.meter.skips(CH_IN, Skip::Stale);
    assert!(skipped > due - 2 * 512 && r.meter.frames(CH_IN) < 2 * 512, "{skipped} skipped, {} moved", r.meter.frames(CH_IN));
    assert_eq!(r.meter.frames(CH_IN) + skipped, due);
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
    *r.sink.lock() = Some(Sink { host: HostLink::default(), dump: None, tap: Some(tap.clone()) });
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
    *r.sink.lock() = Some(Sink { host: HostLink::default(), dump: None, tap: Some(tap.clone()) });
    program(&r, &RING_INPUT, 0x11_0000, 0xffff_f000);
    wr(&r, 0x4c, 1);
    at(&r, 1_000_000);
    rd(&r, 0x14);
    assert!(tap.lock().is_empty());
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

#[test]
fn a_failed_host_open_is_retried_after_a_wait() {
    let now = Instant::now();
    let mut sink = Sink::new();
    sink.host.enable(crate::config::AudioConfig::default());
    assert!(sink.host.retry_due(now), "never tried");
    assert_eq!(sink.host.open(now, false, |_| None, Resampler::following), Some(false));
    assert!(!sink.host.retry_due(now), "just failed");
    assert!(sink.host.retry_due(now + HOST_RETRY), "due again later");
}

const SEC: i64 = 1_000_000_000;

fn tapped(r: &Rad1) -> Arc<parking_lot::Mutex<Vec<i16>>> {
    let tap = Arc::new(parking_lot::Mutex::new(Vec::new()));
    *r.sink.lock() = Some(Sink { host: HostLink::default(), dump: None, tap: Some(tap.clone()) });
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
    // What the clock made due after the lap ended is dropped, once: moved
    // plus dry is the channel's own count, now and a second later.
    let counted = |r: &Rad1| r.meter.frames(CH_OUT) + r.meter.skips(CH_OUT, Skip::Dry);
    assert_eq!(r.meter.frames(CH_OUT), TUNE_FRAMES as u64);
    assert_eq!(counted(&r), r.state.lock().chan[CH_OUT].frames);
    let before = counted(&r);
    at(&r, 11 * SEC);
    rd(&r, 0x58);
    assert!(counted(&r) > before && counted(&r) < before + 2 * 44_100, "one more second, not the first ten again");
    assert_eq!(counted(&r), r.state.lock().chan[CH_OUT].frames);
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
    use crate::dev::audio_dump::{tests::{records, temp}, AudioDump};
    let path = temp("rad1-mute");
    // The relay (0x88) and the DAC (0x8c): the word that mutes, the word that releases.
    for (reg, mute, release) in [(0x88, 2, 3), (0x8c, 3, 2)] {
        let (r, ram, tap) = tune_card();
        ram.0.lock().unwrap()[0x10_0000..0x10_1000].fill(0x55);
        r.sink.lock().as_mut().unwrap().dump = AudioDump::create(path.as_os_str()).map(Arc::new);
        program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
        wr(&r, 0x58, 0x2000_0001);
        let played_at = |ms: i64| {
            at(&r, ms * SEC / 1000);
            rd(&r, 0x58);
            records(&std::fs::read(&path).unwrap()).into_iter().flat_map(|(_, f)| f).collect::<Vec<_>>()
        };
        let a = played_at(10);
        assert!(a.iter().any(|&f| f != (0, 0)), "not muted at power-on");
        wr(&r, reg, mute);
        let b = played_at(20);
        assert!(b.len() > a.len() && b[a.len()..].iter().all(|&f| f == (0, 0)), "{reg:#x} silences");
        wr(&r, reg, release);
        let c = played_at(30);
        assert!(c[b.len()..].iter().any(|&f| f != (0, 0)), "{reg:#x} releases");
        assert!(r.state.lock().chan[CH_OUT].running && tap.lock().len() > 2 * 441 * 2);
    }
    let _ = std::fs::remove_file(&path);
}

thread_local! {
    /// The consumer end of the ring `slow_host` opened: the test plays it.
    static HOST_RING: std::cell::RefCell<Option<rtrb::Consumer<i16>>> = const { std::cell::RefCell::new(None) };
}

/// A host stream that takes 300 ms to open, as a first launch on macOS can:
/// the card's clock runs on meanwhile. 32 kHz, so the tune passes through
/// unresampled, with the card's 200 ms ring.
fn slow_host(card: &Inner, _: &crate::config::AudioConfig) -> Option<HostOut> {
    card.fake_ns().fetch_add(300 * SEC / 1000, Ordering::Relaxed);
    let (p, c) = rtrb::RingBuffer::new(32_000 * 2 / 5);
    HOST_RING.with(|h| *h.borrow_mut() = Some(c));
    Some(HostOut::detached(card.meter.clone(), 32_000, p))
}

static REFUSED_OPENS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A host that refuses every open, counting the tries.
fn refusing_host(_: &Inner, _: &crate::config::AudioConfig) -> Option<HostOut> {
    REFUSED_OPENS.fetch_add(1, Ordering::SeqCst);
    None
}

/// A host that opens a stream at 44.1 kHz on every try.
fn plain_host(card: &Inner, _: &crate::config::AudioConfig) -> Option<HostOut> {
    Some(HostOut::detached(card.meter.clone(), 44_100, rtrb::RingBuffer::new(8).0))
}

/// The output timer's pass opens the host stream again when the host ended
/// it, once the sound is running.
#[test]
fn the_output_pump_opens_a_lost_host_stream_again() {
    let r = Rad1 { inner: Arc::new_cyclic(|me| Inner { open_host: plain_host, ..Inner::new(me.clone(), TestTime::clock(), RAD1_REVISION) }) };
    r.state().connect_dma(Arc::new(Ram(Mutex::new(vec![0; 0x40_0000]))));
    r.enable_output(crate::config::AudioConfig::default());
    Device::start(&r);
    wr(&r, 0xa4, 0xffff);
    wr(&r, 0xa0, 0x4000_0405);
    wr(&r, 0xac, 1);
    program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    wr(&r, 0x58, 0x2000_0001);
    at(&r, SEC / 100);
    r.inner.pump();
    assert_eq!(r.meter.reopens(), 0, "a healthy stream is left alone");
    r.sink.lock().as_ref().unwrap().host.out().unwrap().end(cpal::ErrorKind::DeviceNotAvailable);
    at(&r, SEC / 50);
    r.inner.pump();
    assert!(r.sink.lock().as_ref().unwrap().host.out().is_some_and(|o| !o.lost()));
    assert_eq!(r.meter.reopens(), 1);
    assert!(r.audio_report().unwrap().contains("host reopens: 1, last error: device not available\n"));
}

/// With `[audio] output = "none"` the card is enabled but never asks the host
/// for a stream, at start or on a later try; with any other output it does,
/// and a forced try asks again.
#[test]
fn output_none_never_opens_a_host_stream() {
    let card = || Rad1 { inner: Arc::new_cyclic(|me| Inner { open_host: refusing_host, ..Inner::new(me.clone(), TestTime::clock(), RAD1_REVISION) }) };
    let none = crate::config::AudioConfig { output: Some(crate::config::AudioOutput::None), ..Default::default() };
    let r = card();
    r.enable_output(none);
    Device::start(&r);
    r.inner.open_sink(true);
    r.inner.open_sink(false);
    assert_eq!(REFUSED_OPENS.load(Ordering::SeqCst), 0, "no try for a stream nobody wants");
    assert!(r.audio_report().unwrap().ends_with("host output: none (output = \"none\")\n"));
    let r = card();
    r.enable_output(crate::config::AudioConfig::default());
    Device::start(&r);
    assert_eq!(REFUSED_OPENS.load(Ordering::SeqCst), 1, "the start tries once");
    r.inner.open_sink(false);
    assert_eq!(REFUSED_OPENS.load(Ordering::SeqCst), 1, "not again before the wait");
    r.inner.open_sink(true);
    assert_eq!(REFUSED_OPENS.load(Ordering::SeqCst), 2, "a forced try skips the wait");
    assert!(r.audio_report().unwrap().ends_with("host output: none (not open)\n"));
}

/// A slow host open is over when the machine starts, before the guest
/// plays: the PROM's boot tune reaches the host whole. Opened at the tune's
/// start instead, the frames due by the first look overflow the 200 ms ring
/// and the tune loses 200 ms near its start.
#[test]
fn a_host_slow_to_open_does_not_cost_the_boot_tune_its_start() {
    let r = Rad1 { inner: Arc::new_cyclic(|me| Inner { open_host: slow_host, ..Inner::new(me.clone(), TestTime::clock(), RAD1_REVISION) }) };
    r.state().config().write(4, BAR as u32);
    r.state().config().write(1, PCI_COMMAND_MEMORY_ON);
    r.fake_ns().store(0, Ordering::Relaxed);
    let ram = Arc::new(Ram(Mutex::new(vec![0; 0x40_0000])));
    // A recognisable tune: every frame's samples count up from 1.
    for (f, w) in ram.0.lock().unwrap()[TUNE_BUF as usize..][..TUNE_FRAMES * 8].chunks_exact_mut(4).enumerate() {
        w.copy_from_slice(&sample_to_word((f / 2 % 30_000 + 1) as i16).to_le_bytes());
    }
    r.state().connect_dma(ram);
    r.enable_output(crate::config::AudioConfig::default());
    Device::start(&r);
    prom_tune(&r, TUNE_CTRL);
    let start = r.fake_ns().load(Ordering::Relaxed);
    let mut played = Vec::new();
    for step in 1..=25 {
        at(&r, start + step * 100 * SEC / 1000);
        rd(&r, 0x58);
        HOST_RING.with(|h| played.extend(std::iter::from_fn(|| h.borrow_mut().as_mut().unwrap().pop().ok())));
    }
    assert_eq!(rd(&r, 0x58) & 1, 0, "the tune is over");
    assert_eq!(played.len(), TUNE_FRAMES * 2, "every frame reached the host");
    let want: Vec<i16> = (0..TUNE_FRAMES * 2).map(|i| (i / 2 % 30_000 + 1) as i16).collect();
    assert!(played == want, "in order, none dropped");
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

/// Through late and on-time completions alike, the clock never reads behind
/// a time it has handed out, so the latest time handed out is the clock's
/// reading when the card is saved: a save need not keep the larger of the two.
#[test]
fn the_clock_never_reads_behind_a_time_it_handed_out() {
    use crate::traits::Saveable;
    let (r, _ram) = dma_card();
    let gen = running_with_status(&r);
    let mut host = 0;
    for (k, late) in [5_000_000, 0, 300_000, 40_000_000, 0, 0, 250_000_000, 10_000, 0, 7_000_000].iter().enumerate() {
        host = host.max(irix_due(k as i64 + 1) + late);
        at(&r, host);
        // A guest read hands out the time before the completion runs.
        r.now();
        r.status_tick(gen);
        let seen = r.seen_ns.load(Ordering::SeqCst);
        assert!(r.clock_locked(&None) >= seen, "completion {k}: clock behind {seen}");
    }
    let seen = r.seen_ns.load(Ordering::SeqCst);
    let v = r.save_state();
    assert_eq!(crate::snapshot::field_u64(&v, "clock_ns").unwrap() as i64, seen.max(r.now()));
}

/// A guest access that lands in a hold-up, before the late completion, sees
/// the clock the completion will hold: it moves nothing through the hold-up
/// and leaves the completion free to hold. A plain clock read here once
/// carried the output 5 ms past IRIX's writes and kept the completion from
/// holding anything.
#[test]
fn a_guest_access_racing_a_late_completion_shares_its_hold() {
    let accesses: [(&str, fn(&Rad1)); 3] = [
        ("position read", |r| {
            rd(r, 0x18);
        }),
        ("UST read", |r| {
            rd(r, 0x30);
        }),
        ("start-up word write", |r| wr(r, 0x80, 0xffe2_0000)),
    ];
    for (what, access) in accesses {
        let (r, _ram) = dma_card();
        let gen = running_with_status(&r);
        at(&r, irix_due(1) + 5_000_000);
        access(&r);
        assert!(r.state.lock().chan[CH_OUT].frames <= 12, "{what}: moved over the hold-up");
        assert_eq!(r.status_tick(gen), Some(irix_due(2)), "{what}");
        assert!(r.state.lock().chan[CH_OUT].frames <= 12, "{what}: the completion moved over the hold-up");
        assert!(held(&r) >= 5_000_000 - irix_due(1), "{what}: held {} ns", held(&r));
        // Only a completion wins time back: an on-time look leaves it held.
        let before = held(&r);
        at(&r, irix_due(1) + 5_000_000 + irix_due(1) / 2);
        access(&r);
        assert_eq!(held(&r), before, "{what}");
    }
}

/// The pump's look shares the hold too: run in a hold-up while status runs,
/// it moves nothing through it, and the late completion still holds.
#[test]
fn the_pump_in_a_hold_up_moves_nothing_through_it() {
    let (r, _ram) = dma_card();
    let gen = running_with_status(&r);
    at(&r, irix_due(1) + 5_000_000);
    r.inner.pump();
    assert!(r.state.lock().chan[CH_OUT].frames <= 12, "moved over the hold-up");
    r.status_tick(gen).unwrap();
    assert!(r.state.lock().chan[CH_OUT].frames <= 12, "the completion moved over the hold-up");
    assert!(held(&r) >= 5_000_000 - irix_due(1), "held {} ns", held(&r));
}

/// A save in a hold-up records the held clock and the time held back, so a
/// restore resumes where the late completion would have left the card.
#[test]
fn a_save_in_a_hold_up_records_the_held_clock() {
    use crate::traits::Saveable;
    let (r, _ram) = dma_card();
    running_with_status(&r);
    at(&r, irix_due(1) + 5_000_000);
    let v = r.save_state();
    assert_eq!(crate::snapshot::field_u64(&v, "clock_ns").unwrap() as i64, irix_due(1));
    assert_eq!(held(&r), 5_000_000);
}

/// A completion that ran long, with the next one already overdue when the
/// timer reschedules, does not hand that time out: the next completion still
/// holds. The timer once read the plain clock here, so a completion stalled
/// in the middle (a slow host write, a preempted timer thread) let the next
/// one play the whole stall in one burst.
#[test]
fn rescheduling_after_a_long_completion_leaves_the_next_one_free_to_hold() {
    let (r, _ram) = dma_card();
    let gen = running_with_status(&r);
    at(&r, irix_due(1));
    let next = r.status_tick(gen).unwrap();
    let frames = r.state.lock().chan[CH_OUT].frames;
    at(&r, irix_due(1) + 5_000_000);
    assert_eq!(r.wait_until(next), Duration::ZERO, "the next completion is overdue");
    r.status_tick(gen).unwrap();
    assert!(r.state.lock().chan[CH_OUT].frames <= frames + 12, "moved over the hold-up");
    assert!(held(&r) >= 5_000_000 - 2 * irix_due(1), "held {} ns", held(&r));
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

/// A restore into a card that has been in use equals one into a fresh card.
#[test]
fn rad1_load_into_a_dirtied_instance_equals_load_into_a_fresh_one() {
    use crate::traits::Saveable;
    let src = card();
    wr(&src, 0xa8, 0x1357_9bdf);
    wr(&src, 0x80, 0xffe3_cffe);
    let v = src.save_state();
    let used = card();
    wr(&used, 0xa8, 0x2468_ace0);
    wr(&used, 0x44, 0xa000_0008);
    wr(&used, 0x4c0, 0xa000_c000);
    {
        let mut s = used.state.lock();
        s.chan[CH_OUT].eod = true;
        s.out_cg = 2;
        s.relay_muted = Some(true);
    }
    let fresh = card();
    used.load_state(&v).unwrap();
    fresh.load_state(&v).unwrap();
    assert_eq!(used.save_state(), fresh.save_state());
    assert_eq!(used.save_state(), v);
}

/// A snapshot saved before register 0x84 was stored has 26 start-up words, not
/// 27. It still loads, the new register at zero and the one checked kept.
#[test]
fn a_save_from_before_register_0x84_still_loads() {
    use crate::traits::Saveable;
    let src = card();
    wr(&src, 0x7c, 0x1234_5678);
    wr(&src, 0x84, 7);
    let mut v = src.save_state();
    let words = v.as_table_mut().unwrap().get_mut("startup").unwrap().as_array_mut().unwrap();
    assert_eq!(words.len(), STARTUP_OFFSETS.len());
    words.pop();
    let dst = card();
    dst.load_state(&v).expect("a save with 26 start-up words loads");
    assert_eq!(rd(&dst, 0x7c), 0x1234_5678);
    assert_eq!(rd(&dst, 0x84), 0);
    // A list of any other length is still refused.
    v.as_table_mut().unwrap().get_mut("startup").unwrap().as_array_mut().unwrap().pop();
    let e = card().load_state(&v).unwrap_err();
    assert!(e.contains("startup: 25 entries, expected 27"), "{e}");
}

/// A saved card loads into a fresh one: its registers come back, and its
/// clock reads what it read at the save rather than the restoring card's.
#[test]
fn save_load_round_trip_keeps_the_cards_clock() {
    use crate::traits::Saveable;
    let src = card();
    wr(&src, 0x80, 0xffe3cffe);
    wr(&src, 0xa8, 0x13579bdf);
    {
        let mut s = src.state.lock();
        s.chan[CH_OUT].eod = true;
        s.chan[CH_OUT].next = 5;
        s.chan[CH_OUT].looping = true;
        s.chan[CH_OUT].stalled = true;
        s.out_cg = 2;
        s.cg_loaded[1] = Some((0x4000_0405, 0xffff));
        s.relay_muted = Some(true);
        s.dac_muted = Some(false);
    }
    src.stalled_ns.store(3_000_000, Ordering::SeqCst);
    src.fake_ns().store(8_000_000, Ordering::Relaxed);
    let v = src.save_state();
    let dst = card();
    dst.fake_ns().store(9_000_000_000, Ordering::Relaxed);
    dst.now();
    dst.load_state(&v).expect("load_state");
    assert_eq!(dst.seen_ns.load(Ordering::SeqCst), 5_000_000, "nothing later handed out than the saved time");
    assert_eq!(v, dst.save_state());
    assert_eq!(rd(&dst, 0x80), rd(&src, 0x80));
    assert_eq!(rd(&dst, 0xa8), 0x13579bdf);
    assert_eq!(dst.now(), 5_000_000, "the clock less the time held back");
    assert_eq!(held(&dst), 3_000_000);
    let s = dst.state.lock();
    let c = s.chan[CH_OUT];
    assert_eq!((c.eod, c.next, c.looping, c.stalled), (true, 5, true, true));
    assert_eq!((s.out_cg, s.cg_loaded[1], s.relay_muted, s.dac_muted), (2, Some((0x4000_0405, 0xffff)), Some(true), Some(false)));
}

/// A transfer saved partway through its first descriptor, with a link to
/// follow, the mutes written and time held back for a hold-up, carries on
/// in a fresh card exactly as in the card it came from.
#[test]
fn a_transfer_saved_partway_carries_on_the_same_in_a_fresh_card() {
    use crate::traits::Saveable;
    let entry = 3u32;
    let (src, _src_ram, src_tap) = tune_card();
    // 512 frames link to `entry`, which holds 256 frames with end of descriptor.
    for (i, v) in [0x20_0000, 0, 0xffff_f804].into_iter().enumerate() {
        wr(&src, 0x400 + 12 * entry + 4 * i as u32, v);
    }
    prom_tune(&src, 0xffff_f000 | entry << 3);
    wr(&src, 0x88, 3);
    wr(&src, 0x8c, 2);
    at(&src, 10 * SEC / 1000);
    rd(&src, 0x58);
    src.stalled_ns.store(1_000_000, Ordering::SeqCst);
    {
        let s = src.state.lock();
        let c = s.chan[CH_OUT];
        assert!(c.running && !c.eod && c.next == entry && c.frames > 0 && c.frames < 512, "partway");
        assert_eq!((s.relay_muted, s.dac_muted), (Some(false), Some(false)));
    }
    let played_before = src_tap.lock().len();
    let v = src.save_state();

    let (dst, _dst_ram, dst_tap) = tune_card();
    dst.load_state(&v).expect("load_state");
    assert_eq!(v, dst.save_state());
    for ms in [12, 16, 20, 30, 1000] {
        at(&src, ms * SEC / 1000);
        at(&dst, ms * SEC / 1000);
        assert_eq!(rd(&dst, 0x58), rd(&src, 0x58), "{ms} ms");
        assert_eq!(dst.now(), src.now(), "{ms} ms");
    }
    assert_eq!(rd(&dst, 0x58), 0x4000_0000, "the linked descriptor ended the transfer");
    assert_eq!(src_tap.lock()[played_before..], dst_tap.lock()[..], "the same frames after the save");
    assert_eq!(played_before + dst_tap.lock().len(), (512 + 256) * 2);
}

/// On the host clock too: a stopped card restored later reads the time it
/// stopped at, not the time the snapshot sat on disk.
#[test]
fn a_stopped_cards_host_clock_resumes_where_it_stopped() {
    use crate::traits::Saveable;
    let src = card();
    src.fake_ns().store(i64::MIN, Ordering::Relaxed);
    std::thread::sleep(std::time::Duration::from_millis(20));
    Device::stop(&src);
    src.stalled_ns.store(2_000_000, Ordering::SeqCst);
    let at_save = src.now();
    let v = src.save_state();
    std::thread::sleep(std::time::Duration::from_millis(50));
    let dst = card();
    dst.fake_ns().store(i64::MIN, Ordering::Relaxed);
    Device::stop(&dst);
    dst.load_state(&v).expect("load_state");
    assert_eq!(dst.now(), at_save);
    assert_eq!(held(&dst), 2_000_000);
}

/// A save holding a value the card cannot take is refused, naming the
/// field, rather than loaded to panic or misbehave later.
#[test]
fn a_corrupt_save_is_refused_naming_the_field() {
    use crate::snapshot::{hex_u32, hex_u64, hex_u8};
    use crate::traits::Saveable;
    type Edit = Box<dyn Fn(&mut toml::value::Table)>;
    fn list<'a>(t: &'a mut toml::value::Table, key: &str) -> &'a mut Vec<toml::Value> {
        t.get_mut(key).unwrap().as_array_mut().unwrap()
    }
    fn chan(t: &mut toml::value::Table, n: usize) -> &mut toml::value::Table {
        list(t, "chan")[n].as_table_mut().unwrap()
    }
    let cases: Vec<(&str, Edit)> = vec![
        ("mute word 3", Box::new(|t| { t.insert("relay_muted".into(), hex_u8(3)); })),
        ("out_cg", Box::new(|t| { t.insert("out_cg".into(), hex_u64(NUM_CG as u64)); })),
        ("cg_loaded: not a generator load", Box::new(|t| list(t, "cg_loaded")[0] = hex_u64(2))),
        ("cg_loaded", Box::new(|t| { list(t, "cg_loaded").pop(); })),
        ("chan:", Box::new(|t| { list(t, "chan").pop(); })),
        ("clock:", Box::new(|t| { list(t, "clock").pop(); })),
        ("subcode:", Box::new(|t| { list(t, "subcode").pop(); })),
        ("clock: generator 0", Box::new(|t| list(t, "clock")[0] = hex_u64(0))),
        ("clock: generator 1", Box::new(|t| list(t, "clock")[3] = hex_u64(0))),
        ("into a", Box::new(|t| { chan(t, CH_OUT).insert("offset".into(), hex_u64(1)); })),
        ("links to descriptor 16", Box::new(|t| { chan(t, CH_OUT).insert("next".into(), hex_u32(16)); })),
        ("status_slots", Box::new(|t| { t.insert("status_slots".into(), hex_u32(0x81)); })),
        ("status_cursor: slot 1 of 0", Box::new(|t| { t.insert("status_cursor".into(), hex_u32(1)); })),
        ("status_cursor: slot 4 of 4", Box::new(|t| {
            t.insert("status_slots".into(), hex_u32(4));
            t.insert("status_cursor".into(), hex_u32(4));
        })),
    ];
    let saved = card().save_state();
    card().load_state(&saved).expect("the unedited save loads");
    for (want, edit) in cases {
        let mut v = saved.clone();
        edit(v.as_table_mut().unwrap());
        let e = card().load_state(&v).expect_err(want);
        assert!(e.contains(want), "{want}: got {e}");
    }
}

#[test]
fn the_mediaio_card_presents_revision_7() {
    // The Octane's onboard part keeps 0xc0.
    assert_eq!(Rad1::mediaio().state().config().read(2), u32::from(RAD1_REVISION_MEDIAIO));
    assert_eq!(card().state().config().read(2), u32::from(RAD1_REVISION));
}

/// IRIX on the Onyx2 rewrites 0x8c every 1024 status completions, about
/// once a second, alternating 2 and 3, and writes 2 and 3 to 0x88 when it
/// shifts a word out. Neither is a mute on the MediaIO card: taking 0x8c = 3
/// as the Octane's DAC mute silenced every other second of a playback.
#[test]
fn the_mediaio_card_plays_through_its_0x88_and_0x8c_writes() {
    let r = Rad1::mediaio();
    r.state().config().write(4, BAR as u32);
    r.state().config().write(1, PCI_COMMAND_MEMORY_ON);
    r.fake_ns.store(0, Ordering::Relaxed);
    let ram = Arc::new(Ram(Mutex::new(vec![0x55; 0x40_0000])));
    r.state().connect_dma(ram.clone());
    // A 44.1 kHz host stream, so the ring's samples arrive unresampled.
    let (p, mut host) = rtrb::RingBuffer::new(44_100);
    let host_out = HostOut::detached(r.meter.clone(), 44_100, p);
    *r.sink.lock() = Some(Sink { host: HostLink::with_stream(host_out, Resampler::new(44_100, 44_100, 2)), dump: None, tap: None });
    program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    wr(&r, 0x58, 0x2000_0001);
    let writes = [(0x8c, 3), (0x8c, 2), (0x8c, 3), (0x88, 2), (0x88, 3), (0x88, 2)];
    for (n, (reg, v)) in writes.into_iter().enumerate() {
        wr(&r, reg, v);
        at(&r, (n as i64 + 1) * SEC / 100);
        rd(&r, 0x58);
        let played: Vec<i16> = std::iter::from_fn(|| host.pop().ok()).collect();
        assert_eq!(played.len(), 441 * 2, "after {reg:#x} = {v}");
        assert!(played.iter().all(|&x| x == 0x5555), "after {reg:#x} = {v} the host hears the ring");
    }
}

/// The whole way from guest memory to the host as a running machine takes
/// it: the card on the host's clock with the machine's timers, its output
/// pump moving the ring with no register access, the host output opened by
/// `enable_output` -- a file output, so no sound card -- and resampled
/// from the card's 44.1 kHz to the file's 48 kHz in real time. The file
/// holds the guest's two tones on their own channels at full level; the
/// DAC mute word silences what follows while the transfer runs on.
#[test]
fn a_tone_in_guest_memory_reaches_the_host_on_the_real_clock() {
    use crate::host_audio::test_signal::*;
    // Five and ten cycles fill the 512-frame ring exactly at 44.1 kHz.
    let (left, right) = (5.0 * 44100.0 / 512.0, 10.0 * 44100.0 / 512.0);
    let r = Rad1::new();
    r.state().config().write(4, BAR as u32);
    r.state().config().write(1, PCI_COMMAND_MEMORY_ON);
    let ram = Arc::new(Ram(Mutex::new(vec![0xaa; 0x40_0000])));
    r.state().connect_dma(ram.clone());
    {
        let mut m = ram.0.lock().unwrap();
        for (n, f) in tones(512, 44100, left, right, 16000.0).chunks_exact(2).enumerate() {
            let at = 0x10_0000 + n * 8;
            m[at..at + 4].copy_from_slice(&sample_to_word(f[0]).to_le_bytes());
            m[at + 4..at + 8].copy_from_slice(&sample_to_word(f[1]).to_le_bytes());
        }
    }
    let tm = Arc::new(TimerManager::new());
    r.set_timer_manager(tm.clone());
    let (cfg, path) = file_config("rad1-tone");
    r.enable_output(cfg);
    Device::start(&r);
    start_ust(&r);
    program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    let started = std::time::Instant::now();
    wr(&r, 0x58, 0x2000_0001);
    std::thread::sleep(Duration::from_millis(600));
    // The DAC mute word: the rest plays as silence. The frames played
    // before it are counted in the same step, with no output tick between
    // to play more unmuted.
    let loud = {
        let _tick = r.tick_pass.lock();
        let loud = r.state.lock().chan[CH_OUT].frames;
        wr(&r, 0x8c, 3);
        loud
    };
    std::thread::sleep(Duration::from_millis(300));
    Device::stop(&r);
    let wall = started.elapsed().as_secs_f64();
    let moved = r.state.lock().chan[CH_OUT].frames;
    r.sink.lock().take();
    let (rate, channels, pcm) = read_wav(&path.0);
    assert_eq!((rate, channels), (crate::host_audio::FILE_RATE, 2));
    // Half at least: a timer held up by a busy host holds the clock back.
    assert!(moved as f64 >= 0.5 * wall * 44100.0, "the clock ran: {moved} frames in {wall:.3} s");
    assert!(moved > loud, "the transfer runs on while muted");
    let heard = sounding(&pcm, 2);
    // Resampled to 48 kHz: as many frames as the card played before the
    // mute, less what was still in the ring.
    let expect = loud as f64 * 48000.0 / 44100.0;
    assert!((heard.len() / 2) as f64 >= expect - 9600.0 && (heard.len() / 2) as f64 <= expect + 48.0, "{} frames heard, {expect:.0} played", heard.len() / 2);
    assert_tones(&heard, 2, rate, left, right, 15000);
}

/// RAD1 playing a tone, the guest stopping the analog output for 0.3 s so
/// the host stream runs dry, then starting it again (see
/// [`a_tone_survives_a_pause`]). The 512-frame ring holds five and ten whole
/// cycles at 44.1 kHz, so the tones are about 431 and 861 Hz rather than 440
/// and 880; the host output resamples them.
const OCTANE_TONE: (f64, f64) = (5.0 * 44100.0 / 512.0, 10.0 * 44100.0 / 512.0);

struct Octane {
    r: Rad1,
}

impl CardUnderTest for Octane {
    const NAME: &'static str = "rad1-underrun";
    const PAUSE_IS_STALL: bool = false;
    fn build(cfg: &AudioConfig) -> Self {
        let r = Rad1::new();
        r.state().config().write(4, BAR as u32);
        r.state().config().write(1, PCI_COMMAND_MEMORY_ON);
        let ram = Arc::new(Ram(Mutex::new(vec![0xaa; 0x40_0000])));
        r.state().connect_dma(ram.clone());
        {
            let (left, right) = OCTANE_TONE;
            let mut m = ram.0.lock().unwrap();
            for (n, f) in tones(512, 44100, left, right, 16000.0).chunks_exact(2).enumerate() {
                let at = 0x10_0000 + n * 8;
                m[at..at + 4].copy_from_slice(&sample_to_word(f[0]).to_le_bytes());
                m[at + 4..at + 8].copy_from_slice(&sample_to_word(f[1]).to_le_bytes());
            }
        }
        r.set_timer_manager(Arc::new(TimerManager::new()));
        r.enable_output(cfg.clone());
        Device::start(&r);
        start_ust(&r);
        program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
        Self { r }
    }
    fn tone(&self) -> (f64, f64) {
        OCTANE_TONE
    }
    fn rate(&self) -> f64 {
        44100.0
    }
    fn play(&self) {
        wr(&self.r, 0x58, 0x2000_0001);
    }
    fn pause(&self) {
        wr(&self.r, 0x58, 0x2000_0000);
    }
    fn wait(&self, ms: u64) {
        std::thread::sleep(Duration::from_millis(ms));
    }
    fn frames(&self) -> f64 {
        self.r.meter.frames(1) as f64
    }
    fn meter(&self) -> &crate::host_audio::Meter {
        &self.r.meter
    }
    fn finish(&self) {
        Device::stop(&self.r);
        self.r.sink.lock().take();
    }
}

#[test]
fn a_tone_survives_a_pause_into_a_file() {
    a_tone_survives_a_pause::<Octane>(crate::host_audio::test_signal::Backend::File);
}

#[test]
fn a_tone_survives_a_pause_on_the_real_device() {
    a_tone_survives_a_pause::<Octane>(crate::host_audio::test_signal::Backend::RealDevice);
}

/// The MediaIO start-up writes 0x84 with 0 (and the model once dropped the
/// write as unsupported): it stores like the other start-up words.
#[test]
fn register_0x84_stores_what_is_written() {
    let r = card();
    assert_eq!(rd(&r, 0x84), 0);
    wr(&r, 0x84, 0x5a5a_a5a5);
    assert_eq!(rd(&r, 0x84), 0x5a5a_a5a5);
    r.reset();
    assert_eq!(rd(&r, 0x84), 0);
}

/// The MediaIO driver never writes the counter's control word, yet its
/// start-up requires register 0x30 to advance by 3 in 10 us.
#[test]
fn the_mediaio_counter_runs_from_reset_and_through_a_reset() {
    let r = Rad1::mediaio();
    r.state().config().write(4, BAR as u32);
    r.state().config().write(1, PCI_COMMAND_MEMORY_ON);
    r.fake_ns.store(0, Ordering::Relaxed);
    let a = rd(&r, 0x30);
    at(&r, 10_000);
    assert!(rd(&r, 0x30).wrapping_sub(a) >= 3, "advances with no control write");
    r.reset();
    at(&r, 20_000);
    let b = rd(&r, 0x30);
    at(&r, 30_000);
    assert!(rd(&r, 0x30).wrapping_sub(b) >= 3, "a card reset does not stop it");
}

/// The Octane's counter waits for its control words (see
/// `ust_counter_and_control`); only the MediaIO card runs from reset.
#[test]
fn the_octane_counter_waits_for_its_control_word() {
    let r = card();
    let a = rd(&r, 0x30);
    at(&r, 10_000);
    assert_eq!(rd(&r, 0x30), a);
}

// ---- the monitor's audio status ----

/// The channels' frames reach the card's meter across transfers, though a
/// channel's own count restarts with each one, and `audio status` reports
/// them; a query itself moves nothing.
#[test]
fn audio_status_counts_output_frames_across_transfers() {
    let (r, _ram) = dma_card();
    start_ust(&r);
    program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    wr(&r, 0x58, 0x2000_0001);
    at(&r, 10_000_000);
    rd(&r, 0x2c); // a register read brings the channel up to date
    let text = r.audio_report().unwrap();
    assert_eq!(
        text,
        "audio: RAD1\n\
         \x20 analog in: 0 frames moved, guest rate off\n\
         \x20 analog out: 441 frames moved, guest rate 44100 Hz\n\
         \x20 host underruns: 0 frames\n\
         \x20 dropped by the host: 0 frames\n\
         \x20 host reopens: 0, last error: none\n\
         \x20 host output: none (output = \"none\")\n"
    );
    at(&r, 20_000_000);
    assert_eq!(r.audio_report().unwrap(), text, "a query does not advance the card");
    wr(&r, 0x58, 0x2000_0000);
    at(&r, 30_000_000);
    wr(&r, 0x58, 0x2000_0001);
    at(&r, 40_000_000);
    rd(&r, 0x2c);
    let text = r.audio_report().unwrap();
    assert!(text.contains("analog out: 1323 frames moved"), "{text}");
}

/// The MediaIO card names itself and the host stream it plays to.
#[test]
fn audio_status_names_the_mediaio_card_and_its_host_stream() {
    let r = Rad1::mediaio();
    let (prod, _cons) = rtrb::RingBuffer::<i16>::new(64);
    let out = HostOut::detached(r.meter.clone(), 48000, prod);
    let mut sink = Sink::new();
    sink.host.enable(AudioConfig::default());
    sink.host.stream = Some((out, Resampler::new(44100, 48000, 2)));
    *r.inner.sink.lock() = Some(sink);
    let text = r.audio_report().unwrap();
    assert!(text.starts_with("audio: RAD1 (MediaIO)\n"), "{text}");
    assert!(text.contains("host output: test at 48000 Hz, 2 channels"), "{text}");
}

/// A card playing to a host ring: the output channel running, its frames
/// reaching the host, and a way to count what the silence after them would
/// add to the underruns.
fn playing_card() -> (Rad1, Box<dyn Fn() -> u64>) {
    let (r, ram) = dma_card();
    r.armed.store(true, Ordering::SeqCst);
    let (prod, cons) = rtrb::RingBuffer::<i16>::new(4096);
    let out = HostOut::detached(r.meter.clone(), 44100, prod);
    *r.sink.lock() = Some(Sink { host: HostLink::with_stream(out, Resampler::new(44100, 44100, 2)), dump: None, tap: None });
    let (meter, cons, _ram) = (r.meter.clone(), std::sync::Mutex::new(cons), ram);
    // Drain what the card wrote, then let the host find one frame of silence.
    let underrun = Box::new(move || {
        let mut c = cons.lock().unwrap();
        let n = c.slots();
        c.read_chunk(n).unwrap().commit_all();
        crate::host_audio::count_underrun(crate::host_audio::pop_frames(&mut c, 2, &mut [0i16; 2], |v| v), &meter);
        meter.underruns()
    });
    assert_eq!(underrun(), 0, "idle");
    start_ust(&r);
    program(&r, &RING_OUTPUT, 0x10_0000, 0xffff_f008);
    wr(&r, 0x58, 0x2000_0001);
    at(&r, 10_000_000);
    rd(&r, 0x2c); // brings the channel up to date: frames reach the host
    assert_eq!(underrun(), 1, "playing");
    (r, underrun)
}

/// The output channel's first frame to the host starts the card playing and
/// the guest stopping the channel ends it: the host's silence counts as
/// underruns only in between.
#[test]
fn underruns_count_from_the_first_frame_until_the_channel_stops() {
    let (r, underrun) = playing_card();
    wr(&r, 0x58, 0x2000_0000);
    assert_eq!(underrun(), 1, "stopped");
}

/// A machine pause ends it too.
#[test]
fn pausing_the_machine_ends_playing() {
    let (r, underrun) = playing_card();
    Device::stop(&r);
    assert_eq!(underrun(), 1, "paused");
}

/// So does a hard reset, which replaces the channel state with no guest write.
#[test]
fn a_reset_ends_playing() {
    let (r, underrun) = playing_card();
    r.inner.reset();
    assert_eq!(underrun(), 1, "reset");
}

/// And a channel that stops by itself, its ring gone, found by the pump.
#[test]
fn a_channel_that_stops_by_itself_ends_playing() {
    let (r, underrun) = playing_card();
    r.state.lock().chan[CH_OUT].running = false;
    r.inner.pump();
    assert_eq!(underrun(), 1, "stopped by the pump");
}
