// Cases follow tests/qtest/sn0-hub-test.c in jamesbraid/qemu-sgi-origami:
// md-config, md-error-cluster, md-dimm-init, md-perf-cnt0, the slot-identity
// cases (flash-ready-at-both-bits, dual-node-master-slave-sense,
// origin2000-slot-and-processor-identity, onyx2-*), md-indicator-registers
// and the one-wire master cases.

use super::*;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

fn md(signals: SlotSignals) -> (Md, Arc<AtomicU64>) {
    let now = Arc::new(AtomicU64::new(0));
    let clock = now.clone();
    (Md::new(Arc::new(move || clock.load(Ordering::SeqCst)), signals), now)
}
fn r(md: &Md, off: u64) -> u64 {
    md.read64(off).data
}

/// MD_MEMORY_CONFIG is 54 bits of storage that resets to Linux's default and
/// resizes nothing.
#[test]
fn the_memory_config_stores_54_bits_and_resets_to_linuxs_default() {
    let (md, _) = md(SlotSignals::default());
    assert_eq!(r(&md, reg::CONFIG), 0x001e_7fbc_3fff_ffff);
    md.write64(reg::CONFIG, !0);
    assert_eq!(r(&md, reg::CONFIG), 0x003f_ffff_ffff_ffff);
    md.write64(reg::CONFIG, 0x1234);
    assert_eq!(r(&md, reg::CONFIG), 0x1234);
    md.power_on();
    assert_eq!(r(&md, reg::CONFIG), 0x001e_7fbc_3fff_ffff);
}

/// Firmware's memory test needs the error cluster to read zero; the DIMM
/// registers are write-only; the performance counter stores 20 bits and never
/// counts.
#[test]
fn the_error_cluster_reads_zero_and_the_performance_counter_never_counts() {
    let (md, now) = md(SlotSignals::default());
    for off in (reg::ERROR_FIRST..=reg::ERROR_LAST).step_by(8) {
        md.write64(off, !0);
        assert_eq!(r(&md, off), 0, "{off:#x}");
    }
    for off in [reg::MEM_DIMM_INIT, reg::DIR_DIMM_INIT] {
        md.write64(off, 0x55);
        assert_eq!(r(&md, off), 0);
    }
    assert_eq!(r(&md, reg::PERF_CNT0), 0);
    md.write64(reg::PERF_CNT0, !0);
    now.store(1_000_000, Ordering::SeqCst);
    assert_eq!(r(&md, reg::PERF_CNT0), 0xf_ffff);
    md.power_on();
    assert_eq!(r(&md, reg::PERF_CNT0), 0);
}

#[test]
fn the_origin_200_reads_flash_ready_and_the_master_sense() {
    let (master, _) = md(SlotSignals { slot: None, slave_module: false });
    let (slave, _) = md(SlotSignals { slot: None, slave_module: true });
    assert_eq!(r(&master, reg::SLOTID_USTAT), 0x95);
    assert_eq!(r(&slave, reg::SLOTID_USTAT), 0x15);
    // Read-only: a store changes nothing.
    master.write64(reg::SLOTID_USTAT, 0);
    assert_eq!(r(&master, reg::SLOTID_USTAT), 0x95);
}

#[test]
fn an_origin_2000_node_reads_the_slot_the_midplane_drives() {
    for slot in 0..8 {
        let (md, _) = md(SlotSignals { slot: Some(slot), slave_module: false });
        assert_eq!(r(&md, reg::SLOTID_USTAT), 0x90 | u64::from(slot));
    }
    // Only the low three bits are the slot.
    let (md, _) = md(SlotSignals { slot: Some(0xff), slave_module: false });
    assert_eq!(r(&md, reg::SLOTID_USTAT), 0x97);
}

#[test]
fn the_indicators_are_write_only() {
    let (md, _) = md(SlotSignals::default());
    for off in [reg::LED0, reg::LED1, reg::OLD_LED0, reg::OLD_LED1] {
        md.write64(off, 0x2b);
        assert_eq!(r(&md, off), 0, "{off:#x}");
    }
}

#[test]
fn unimplemented_offsets_read_zero() {
    let (md, _) = md(SlotSignals::default());
    md.write64(0x20_0400, 7);
    assert_eq!(r(&md, 0x20_0400), 0);
}

/// MD_MLAN_CTL is 34 bits wide and resets to 0x31 in both phases.
#[test]
fn the_one_wire_master_keeps_its_wide_field_and_resets_to_its_timing() {
    let (md, now) = md(SlotSignals::default());
    assert_eq!(r(&md, reg::MLAN_CTL), 0x1_8b10_0000);
    // A write cannot set done or the received data, and truncating to 32 bits
    // would rewrite the upper phase.
    md.write64(reg::MLAN_CTL, (0x66 << 27) | (60 << 10) | (0x41 << 2) | 3);
    assert_eq!(r(&md, reg::MLAN_CTL), (0x66 << 27) | (60 << 10) | (0x41 << 2));
    // Done and the sampled level arrive a microsecond later. With no part on
    // the wire a 60-unit pulse is a write-zero, which holds the wire low, and
    // a short one is a write-one, which samples high.
    now.store(999, Ordering::SeqCst);
    assert_eq!(r(&md, reg::MLAN_CTL) & 3, 0);
    now.store(1000, Ordering::SeqCst);
    assert_eq!(r(&md, reg::MLAN_CTL) & 3, 2);
    md.write64(reg::MLAN_CTL, 8 << 10);
    now.store(2000, Ordering::SeqCst);
    assert_eq!(r(&md, reg::MLAN_CTL) & 3, 3);
    // A reset cancels a slot in progress.
    md.write64(reg::MLAN_CTL, 8 << 10);
    md.power_on();
    now.store(10_000, Ordering::SeqCst);
    assert_eq!(r(&md, reg::MLAN_CTL), 0x1_8b10_0000);
}

/// A slot starts clear of done: a read before the deadline sees neither done
/// nor the answer, even though the bus answered inside the write.
#[test]
fn starting_a_slot_is_not_finishing_it() {
    let (md, now) = md(SlotSignals::default());
    md.write64(reg::MLAN_CTL, 8 << 10);
    assert_eq!(r(&md, reg::MLAN_CTL) & 3, 0);
    now.fetch_add(1000, Ordering::SeqCst);
    md.write64(reg::MLAN_CTL, 8 << 10);
    assert_eq!(r(&md, reg::MLAN_CTL) & 3, 0, "the restart clears done");
}

/// Firmware reads the node's board record over the master: a reset slot, a
/// ROM command, and then the part's serial and memory, one slot a bit.
#[test]
fn the_master_reads_the_nodes_board_record() {
    let (md, now) = md(SlotSignals::default());
    let record = BoardRecord { barcode: "EMUU0000", part: "013-1895-001", revision: "C", name: "PIMM_1XT5_1MB" };
    let mut bus = Bus::new();
    bus.add(Box::new(board_record_part(&record, 0x0123_4567_89ab)));
    md.set_onewire(bus);
    let slot = |pulse: u64| {
        md.write64(reg::MLAN_CTL, pulse << 10);
        now.fetch_add(MLAN_SLOT_NS, Ordering::SeqCst);
        r(&md, reg::MLAN_CTL)
    };
    // The part answers the reset slot with a presence pulse: the wire reads low.
    let reset = slot(520);
    assert_eq!((reset & MLAN_DONE != 0, reset & MLAN_RX != 0), (true, false));
    let bit = |one: bool| slot(if one { 6 } else { 80 }) & MLAN_RX != 0;
    // READ ROM, then the family code and serial, least significant bit first.
    for i in 0..8 {
        bit(0x33u8 >> i & 1 != 0);
    }
    let mut rom = [0u8; 8];
    for i in 0..64 {
        if bit(true) {
            rom[i / 8] |= 1 << (i % 8);
        }
    }
    assert_eq!(rom[0], 0x0b, "a DS2505's family code");
    assert_eq!(&rom[1..7], &[0xab, 0x89, 0x67, 0x45, 0x23, 0x01]);
}
