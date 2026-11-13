//! What differs between the products the pipe can be: the board set and texture
//! size its configuration block reports, the boards its records name, the word the
//! driver's attach classes it by, and the chassis it fits. The InfiniteReality2's
//! block is checked byte for byte in `config`; these check the fields that move.

use super::config::read_config;
use super::regs::{geometry_rom, Wire};
use super::*;
use crate::dev::kona::ident::{Host, Product, PRODUCTS};

const INDIRECT_SELECT: u64 = 0x8003c0;
const INDIRECT_DATA: u64 = 0x8003d0;
const ONEWIRE: u64 = 0x98;

fn pipe(product: Product, rms: u32) -> Fixture {
    Fixture::with(KonaConfig::new(product, rms).unwrap(), Recorder::default())
}

fn word(block: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(block[at..at + 4].try_into().unwrap())
}

/// A right-justified text field of an identity record, trimmed.
fn field(block: &[u8], from: usize, to: usize) -> String {
    String::from_utf8_lossy(&block[from..to]).trim().to_string()
}

/// The first raster manager's identity record starts at 0x3e8, the first texture
/// board's at 0x564: part at +0x17, name at +0x3c.
fn rm_part_and_name(block: &[u8]) -> (String, String) {
    (field(block, 0x3e8 + 0x17, 0x3e8 + 0x2a), field(block, 0x3e8 + 0x3c, 0x3e8 + 0x4a))
}

fn tm_part_and_name(block: &[u8]) -> (String, String) {
    (field(block, 0x564 + 0x17, 0x564 + 0x2a), field(block, 0x564 + 0x3c, 0x564 + 0x4a))
}

#[test]
fn the_board_set_and_texture_size_follow_the_product() {
    // (product, board set, texture size code, texture MiB)
    for (product, set, code, mb) in [(Product::Ir2, 1, 2, 64), (Product::Ir2e, 4, 2, 64), (Product::Ir3, 5, 4, 256)] {
        let block = read_config(&pipe(product, 2));
        let name = product.spec().name;
        assert_eq!(word(&block, 0x00), set, "{name} board set");
        assert_eq!(word(&block, 0x04), 4, "{name} engines");
        assert_eq!(word(&block, 0x08), 2, "{name} raster managers");
        for rm in 0..2 {
            assert_eq!((word(&block, 0x24 + 8 * rm), word(&block, 0x28 + 8 * rm)), (1, code), "{name} RM{rm}");
        }
        // The texture pool counts 1 KiB granules.
        assert_eq!(word(&block, 0xe4 + 0x18) as u64, mb << 10, "{name} texture pool");
        assert_eq!(KonaConfig::new(product, 1).unwrap().texture_bytes_per_rm(), mb << 20);
    }
}

#[test]
fn a_single_raster_manager_leaves_the_second_slot_empty() {
    let block = read_config(&pipe(Product::Ir3, 1));
    assert_eq!((word(&block, 0x24), word(&block, 0x28)), (1, 4));
    assert_eq!((word(&block, 0x2c), word(&block, 0x30)), (0, 0));
    assert_eq!(word(&block, 0x58), 1);
}

#[test]
fn the_records_name_each_products_boards() {
    let ir2 = read_config(&pipe(Product::Ir2, 1));
    assert_eq!(rm_part_and_name(&ir2), ("030-1054-001".into(), "RM7".into()));
    assert_eq!(tm_part_and_name(&ir2), ("030-1053-001".into(), "TM7".into()));
    assert_eq!(word(&ir2, 0x114), 0x0302_0c06, "InfiniteReality2 raster board word");
    for product in [Product::Ir2e, Product::Ir3] {
        let block = read_config(&pipe(product, 1));
        assert_eq!(rm_part_and_name(&block), ("030-1402-001".into(), "RM9".into()), "{product:?}");
        assert_eq!(word(&block, 0x100), 0x80f6_1806, "{product:?} geometry board word");
        // Every product presents the DG5-8 (see `ident::Display`).
        assert_eq!(word(&block, 0x344), 0x51bf_1002, "{product:?} display word");
    }
    // The 2E's raster slot is type 6 with texture code 2, as the 2's: type 0xb
    // makes the pipe present as an InfiniteReality3 with 256 MB.
    let ir2e = read_config(&pipe(Product::Ir2e, 1));
    assert_eq!(word(&ir2e, 0x114), 0x0302_0c06);
    assert_eq!((word(&ir2e, 0x114) >> 9) & 0x7f, 6, "slot board type");
    assert_eq!(tm_part_and_name(&ir2e), ("030-1053-001".into(), "TM7".into()));
    // The 2E's RM9 and TM7 revisions are unknown, so the records leave them blank.
    assert_eq!(field(&ir2e, 0x3e8 + 0x32, 0x3e8 + 0x36), "");
    assert_eq!(field(&ir2e, 0x564 + 0x32, 0x564 + 0x36), "");
    // The 3's raster slot is type 0xb with code 4, and its 256 MB texture board
    // is the TM8, 030-1588-001, revision unknown.
    let ir3 = read_config(&pipe(Product::Ir3, 1));
    assert_eq!(word(&ir3, 0x114), 0x0304_1606);
    assert_eq!((word(&ir3, 0x114) >> 9) & 0x7f, 0xb, "slot board type");
    assert_eq!(tm_part_and_name(&ir3), ("030-1588-001".into(), "TM8".into()));
    assert_eq!(field(&ir3, 0x564 + 0x32, 0x564 + 0x36), "");
    // Board set 3 has no producer: no product emits it.
    for row in &PRODUCTS {
        assert_ne!(word(&read_config(&pipe(row.product, 1)), 0), 3, "{}", row.name);
    }
}

/// The attach names a pipe InfiniteReality2E when the geometry board's record
/// holds "GE16-", so the one-wire record is what makes the guest's `hinv`
/// differ between the 2 and the 2E.
#[test]
fn the_geometry_boards_one_wire_record_is_the_products() {
    for (product, part, revision, name) in
        [(Product::Ir2, "030-1129-002", "F", "GE14-4"), (Product::Ir2e, "030-1398-001", "E", "GE16-4"), (Product::Ir3, "030-1398-001", "E", "GE16-4")]
    {
        let f = pipe(product, 1);
        let w = Wire::new(&f, ONEWIRE);
        assert!(w.match_rom(&geometry_rom()));
        let (_, mem) = w.read_memory(0, 64);
        let text = |a: usize, n: usize| String::from_utf8(mem[a..a + n].to_vec()).unwrap();
        assert_eq!(format!("{}{}", text(21, 9), text(35, 3)), part, "{product:?}");
        assert_eq!(text(38, revision.len()), revision, "{product:?}");
        assert_eq!(text(48, name.len()), name, "{product:?}");
    }
}

/// After selecting 0x8800 the attach reads the product in the low two bits:
/// 1 classes the pipe InfiniteReality3. After 0xb00 bit 10 stays clear for all.
#[test]
fn the_class_words_follow_the_product() {
    for (product, low) in [(Product::Ir2, 0), (Product::Ir2e, 0), (Product::Ir3, 1)] {
        let f = pipe(product, 1);
        f.w(INDIRECT_SELECT, 0x8800);
        assert_eq!(f.r(INDIRECT_DATA) & 3, low, "{product:?}");
        f.w(INDIRECT_SELECT, 0xb00);
        assert_eq!(f.r(INDIRECT_DATA) & 0x400, 0, "{product:?}");
    }
}

#[test]
fn a_chassis_the_product_did_not_ship_in_is_refused_with_the_reason() {
    let spec = |p: Product| p.spec();
    assert!(spec(Product::Ir2).check_host(Host::Onyx2Deskside).is_ok());
    assert!(spec(Product::Ir2).check_host(Host::Onyx2Rack).is_err());
    assert!(spec(Product::Ir2e).check_host(Host::Onyx2Deskside).is_ok());
    assert!(spec(Product::Ir2e).check_host(Host::Onyx2Rack).is_ok());
    assert!(spec(Product::Ir3).check_host(Host::Onyx2Rack).is_ok());
    assert!(spec(Product::Ir3).check_host(Host::Onyx3000).is_ok());
    let e = spec(Product::Ir3).check_host(Host::Onyx2Deskside).unwrap_err();
    assert!(e.contains("InfiniteReality3") && e.contains("Onyx2 deskside") && e.contains("Onyx2 rack"), "{e}");
    for row in &PRODUCTS {
        assert!(!row.hosts.is_empty(), "{}", row.name);
    }
}

#[test]
fn the_deskside_raster_manager_counts_hold_for_every_product() {
    for row in &PRODUCTS {
        assert!(KonaConfig::new(row.product, 2).is_ok());
        let e = KonaConfig::new(row.product, 4).unwrap_err();
        assert!(e.contains("one or two raster managers"), "{}: {e}", row.name);
    }
}
