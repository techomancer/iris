//! What the pipe presents: the 0x1e9 configuration answer, built from
//! the ported values for the deskside pipe, and the words host software reads its
//! class from. Ported from tests/qtest/sn0-origin2000-kona-config-test.c for
//! the InfiniteReality2 of the Onyx2 deskside (GE14-4, RM7, TM7, eight-channel
//! DG5).

use super::ident::tests::{parse_hex, synthetic_serial};
use super::*;

const INDIRECT_SELECT: u64 = 0x8003c0;
const INDIRECT_DATA: u64 = 0x8003d0;
const RETURN_PAGE: u64 = 0x32_0000;
const CONFIG_OFFSET: u64 = 8;
const CONFIG_SIZE: usize = 0x778;

/// A graphics process's 0x1f2, then 0x1e9 0; returns the posted block.
pub(super) fn read_config(f: &Fixture) -> Vec<u8> {
    f.cmd(&[0x1f2, 0x1000_0000 | (RETURN_PAGE >> 12) as u32, (RETURN_PAGE & 0xfff) as u32, 0x4000]);
    f.cmd(&[0x1e9, 0]);
    assert_eq!(f.host.word(RETURN_PAGE), 1);
    f.host.bytes(RETURN_PAGE + CONFIG_OFFSET, CONFIG_SIZE)
}

fn assert_config(f: &Fixture, want: &str) {
    let block = read_config(f);
    let want = parse_hex(want);
    if let Some(i) = (0..CONFIG_SIZE).step_by(4).find(|&i| block[i..i + 4] != want[i..i + 4]) {
        panic!("first difference at block offset {i:#x}: {:x?} against {:x?}", &block[i..i + 4], &want[i..i + 4]);
    }
    assert_eq!(block, want);
}

/// The deskside pipe, byte for byte against the fixed block:
/// two RM7 boards, each with its TM7, and the eight-channel DG.
#[test]
fn ir2_ge14_4_two_raster_managers() {
    assert_config(&Fixture::new(), include_str!("data/config-ir2-2rm.hex"));
}

/// A deskside pipe with a single raster manager: one RM's chip words and its
/// TM record, and the raster-manager counts follow.
#[test]
fn ir2_ge14_4_one_raster_manager() {
    let f = Fixture::with(KonaConfig::ir2(1).unwrap(), Recorder::default());
    assert_config(&f, include_str!("data/config-ir2-1rm.hex"));
}

/// The pipe slot gives each board's records their own serial instances, sixteen
/// to a slot, so two pipes never share a serial.
#[test]
fn a_pipe_slot_moves_the_serial_instances() {
    let f = Fixture::with(KonaConfig::ir2(2).unwrap().in_slot(1).unwrap(), Recorder::default());
    let block = read_config(&f);
    // The first RM record's one-wire number is the slot's instance 2.
    let serial = synthetic_serial(16 + 2);
    for i in 0..6 {
        assert_eq!(block[0x3e8 + 4 + i], (serial >> (40 - 8 * i)) as u8);
    }
    assert!(KonaConfig::ir2(2).unwrap().in_slot(4).is_err());
}

/// The two words the driver's attach classes the pipe by,
/// read through the indirect port: after selecting 0x8800 the low two bits are
/// the product's (0 for an InfiniteReality2), after 0xb00 bit 10 would make it
/// a Reality (clear).
#[test]
fn class_words() {
    let f = Fixture::new();
    f.w(INDIRECT_SELECT, 0x8800);
    assert_eq!(f.r(INDIRECT_DATA) & 3, 0);
    f.w(INDIRECT_SELECT, 0xb00);
    assert_eq!(f.r(INDIRECT_DATA) & 0x400, 0);
    assert_eq!(f.r(INDIRECT_SELECT), 0xb00);
}

/// configuration no source grounds is refused with its reason. A
/// deskside pipe holds one or two raster managers.
#[test]
fn configurations_without_a_source_are_refused() {
    for n in [0, 3, 4] {
        let e = KonaConfig::ir2(n).unwrap_err();
        assert!(e.contains("one or two raster managers"), "{e}");
    }
}

/// A store to the indirect data register under a selector the front end does
/// not answer is the display generator's: the back end is handed it.
#[test]
fn other_indirect_selectors_reach_the_back_end() {
    let f = Fixture::hooked();
    f.w(INDIRECT_SELECT, 0x101);
    f.w(INDIRECT_DATA, 0x1234);
    assert_eq!(f.take_jobs(), ["indirect 0x101 0x1234"]);
    assert_eq!(f.r(INDIRECT_DATA), 0, "its reads are not modelled");
}
