//! The vectors are literals worked out by hand from the layouts in the module
//! header, so a change to the layout shows up here. Most use kind 1, which no
//! iris machine has, so that they differ from iris's own where the kind shows.

use super::*;

const MAC: [u8; 6] = [0x08, 0x00, 0x69, 0xe1, 0x02, 0xa0];

#[test]
fn iris_is_kind_two() {
    assert_eq!(KIND, 2);
}

/// IRIX folds only the first ten characters; the first two never show.
#[test]
fn the_fold_keeps_the_low_nibbles_of_the_first_ten_characters() {
    assert_eq!(hostid_from_text("LEMU1042"), 0xc5d5_1042);
    assert_eq!(hostid_from_text("EMU1-0042"), 0xd51d_0042);
    assert_eq!(hostid_from_text("EMU1L0000042"), 0x51c0_0000);
    assert_eq!(hostid_from_text("M2001347"), 0xd200_1347);
    assert_eq!(hostid_from_text("L3962017"), 0xc396_2017);
    assert_eq!(hostid_from_text("L1000302"), 0xc100_0302);
    assert_eq!(hostid_from_text(""), 0);
}

#[test]
fn the_default_station_address_carries_the_kind_and_the_emu_id() {
    assert_eq!(default_mac(1, 42, 0).unwrap(), MAC);
    assert_eq!(hostid_from_mac(&MAC), 0x69e1_02a0);
    // Kind 2 changes only the nibble after E.
    assert_eq!(hostid_from_mac(&default_mac(2, 42, 0).unwrap()), 0x69e2_02a0);
    assert_eq!(hostid_from_mac(&default_mac(1, 0, 0).unwrap()), 0x69e1_0000);
    assert_eq!(default_mac(KIND, 0, 0).unwrap(), [0x08, 0x00, 0x69, 0xe2, 0x00, 0x00]);
    assert_eq!(hostid_from_mac(&default_mac(KIND, 7, 0).unwrap()), 0x69e2_0070);
}

/// The sub-index fills the low nibble of the last byte and never reaches the
/// kind or the emu-id, at both ends of the range.
#[test]
fn the_sub_index_never_reaches_the_emu_id() {
    for emu_id in [0, 1, 42, 255, 4094, 4095] {
        let base = default_mac(1, emu_id, 0).unwrap();
        assert_eq!(base[5] & 0xf, 0);
        for sub in 0..SUB_INDEX_COUNT {
            let mac = default_mac(1, emu_id, sub).unwrap();
            assert_eq!(mac[..5], base[..5]);
            assert_eq!(mac[5] >> 4, base[5] >> 4);
            assert_eq!(u32::from(mac[5] & 0xf), sub);
            assert_eq!(mac[3], 0xe1);
            assert_eq!(u32::from(mac[4]) << 4 | u32::from(mac[5] >> 4), emu_id);
        }
        assert_eq!(default_mac(1, emu_id, 16), None);
    }
}

#[test]
fn the_system_serial_ends_in_a_zero_and_spans_the_emu_id_range() {
    assert_eq!(system_serial(1, 42), "LEMU100420");
    assert_eq!(hostid_from_text("LEMU100420"), 0xd510_0420);
    assert_eq!(system_serial(1, 0), "LEMU100000");
    assert_eq!(system_serial(KIND, 42), "LEMU200420");
    assert_eq!(hostid_from_text(&system_serial(KIND, 42)), 0xd520_0420);
    assert_eq!(system_serial(KIND, EMU_ID_MAX), "LEMU240950");
    assert_eq!(hostid_from_text(&system_serial(KIND, EMU_ID_MAX)), 0xd524_0950);
    // Ten characters, so all of the kind and the emu_id reach the id.
    assert_eq!(system_serial(KIND, 1).len(), 10);
}

#[test]
#[should_panic]
fn the_system_serial_refuses_an_emu_id_past_the_range() {
    system_serial(KIND, EMU_ID_MAX + 1);
}
