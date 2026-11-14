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
    (Md::new(Arc::new(move || clock.load(Ordering::SeqCst)), signals, [64, 0, 0, 0]), now)
}
fn r(md: &Md, off: u64) -> u64 {
    md.read(off)
}

/// MD_MEMORY_CONFIG is 54 bits of storage that resets to Linux's default and
/// resizes nothing.
#[test]
fn the_memory_config_stores_54_bits_and_resets_to_linuxs_default() {
    let (md, _) = md(SlotSignals::default());
    assert_eq!(r(&md, reg::MD_MEMORY_CONFIG::OFFSET), 0x001e_7fbc_3fff_ffff);
    md.write(reg::MD_MEMORY_CONFIG::OFFSET, !0);
    assert_eq!(r(&md, reg::MD_MEMORY_CONFIG::OFFSET), 0x003f_ffff_ffff_ffff);
    md.write(reg::MD_MEMORY_CONFIG::OFFSET, 0x1234);
    assert_eq!(r(&md, reg::MD_MEMORY_CONFIG::OFFSET), 0x1234);
    md.power_on();
    assert_eq!(r(&md, reg::MD_MEMORY_CONFIG::OFFSET), 0x001e_7fbc_3fff_ffff);
}

/// Firmware's memory test needs the error cluster to read zero; the DIMM
/// registers are write-only; the performance counter stores 20 bits and never
/// counts.
#[test]
fn the_error_cluster_reads_zero_and_the_performance_counter_never_counts() {
    let (md, now) = md(SlotSignals::default());
    for off in (reg::MD_DIR_ERROR::OFFSET..=reg::MD_MISC_ERROR_CLR::OFFSET).step_by(8) {
        md.write(off, !0);
        assert_eq!(r(&md, off), 0, "{off:#x}");
    }
    for off in [reg::MD_MEM_DIMM_INIT::OFFSET, reg::MD_DIR_DIMM_INIT::OFFSET] {
        md.write(off, 0x55);
        assert_eq!(r(&md, off), 0);
    }
    assert_eq!(r(&md, reg::MD_PERF_CNT0::OFFSET), 0);
    md.write(reg::MD_PERF_CNT0::OFFSET, !0);
    now.store(1_000_000, Ordering::SeqCst);
    assert_eq!(r(&md, reg::MD_PERF_CNT0::OFFSET), 0xf_ffff);
    md.power_on();
    assert_eq!(r(&md, reg::MD_PERF_CNT0::OFFSET), 0);
}

#[test]
fn the_origin_200_reads_flash_ready_and_the_master_sense() {
    let (master, _) = md(SlotSignals { slot: None, slave_module: false });
    let (slave, _) = md(SlotSignals { slot: None, slave_module: true });
    assert_eq!(r(&master, reg::MD_SLOTID_USTAT::OFFSET), 0x95);
    assert_eq!(r(&slave, reg::MD_SLOTID_USTAT::OFFSET), 0x15);
    // Read-only: a store changes nothing.
    master.write(reg::MD_SLOTID_USTAT::OFFSET, 0);
    assert_eq!(r(&master, reg::MD_SLOTID_USTAT::OFFSET), 0x95);
}

#[test]
fn an_origin_2000_node_reads_the_slot_the_midplane_drives() {
    for slot in 0..8 {
        let (md, _) = md(SlotSignals { slot: Some(slot), slave_module: false });
        assert_eq!(r(&md, reg::MD_SLOTID_USTAT::OFFSET), 0x90 | u64::from(slot));
    }
    // Only the low three bits are the slot.
    let (md, _) = md(SlotSignals { slot: Some(0xff), slave_module: false });
    assert_eq!(r(&md, reg::MD_SLOTID_USTAT::OFFSET), 0x97);
}

#[test]
fn the_indicators_are_write_only() {
    let (md, _) = md(SlotSignals::default());
    for off in [reg::MD_LED0::OFFSET, reg::MD_LED1::OFFSET, reg::MD_UREG1_0::OFFSET, reg::MD_UREG1_4::OFFSET] {
        md.write(off, 0x2b);
        assert_eq!(r(&md, off), 0, "{off:#x}");
    }
}

#[test]
fn unimplemented_offsets_read_zero() {
    let (md, _) = md(SlotSignals::default());
    md.write(0x20_0400, 7);
    assert_eq!(r(&md, 0x20_0400), 0);
}

/// MD_MLAN_CTL is 34 bits wide and resets to 0x31 in both phases.
#[test]
fn the_one_wire_master_keeps_its_wide_field_and_resets_to_its_timing() {
    let (md, now) = md(SlotSignals::default());
    assert_eq!(r(&md, reg::MD_MLAN_CTL::OFFSET), 0x1_8b10_0000);
    // A write cannot set done or the received data, and truncating to 32 bits
    // would rewrite the upper phase.
    md.write(reg::MD_MLAN_CTL::OFFSET, (0x66 << 27) | (60 << 10) | (0x41 << 2) | 3);
    assert_eq!(r(&md, reg::MD_MLAN_CTL::OFFSET), (0x66 << 27) | (60 << 10) | (0x41 << 2));
    // Done and the sampled level arrive a microsecond later. With no part on
    // the wire a 60-unit pulse is a write-zero, which holds the wire low, and
    // a short one is a write-one, which samples high.
    now.store(999, Ordering::SeqCst);
    assert_eq!(r(&md, reg::MD_MLAN_CTL::OFFSET) & 3, 0);
    now.store(1000, Ordering::SeqCst);
    assert_eq!(r(&md, reg::MD_MLAN_CTL::OFFSET) & 3, 2);
    md.write(reg::MD_MLAN_CTL::OFFSET, 8 << 10);
    now.store(2000, Ordering::SeqCst);
    assert_eq!(r(&md, reg::MD_MLAN_CTL::OFFSET) & 3, 3);
    // A reset cancels a slot in progress.
    md.write(reg::MD_MLAN_CTL::OFFSET, 8 << 10);
    md.power_on();
    now.store(10_000, Ordering::SeqCst);
    assert_eq!(r(&md, reg::MD_MLAN_CTL::OFFSET), 0x1_8b10_0000);
}

/// A slot starts clear of done: a read before the deadline sees neither done
/// nor the answer, even though the bus answered inside the write.
#[test]
fn starting_a_slot_is_not_finishing_it() {
    let (md, now) = md(SlotSignals::default());
    md.write(reg::MD_MLAN_CTL::OFFSET, 8 << 10);
    assert_eq!(r(&md, reg::MD_MLAN_CTL::OFFSET) & 3, 0);
    now.fetch_add(1000, Ordering::SeqCst);
    md.write(reg::MD_MLAN_CTL::OFFSET, 8 << 10);
    assert_eq!(r(&md, reg::MD_MLAN_CTL::OFFSET) & 3, 0, "the restart clears done");
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
        md.write(reg::MD_MLAN_CTL::OFFSET, pulse << 10);
        now.fetch_add(MLAN_SLOT_NS, Ordering::SeqCst);
        r(&md, reg::MD_MLAN_CTL::OFFSET)
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

#[test]
fn a_back_door_stores_two_bytes_of_each_entry_and_folds_a_bank() {
    let door = Backdoor::new([64, 0, 0, 0]);
    door.write64(0, 0x1122_3344_5566_7788);
    assert_eq!(door.read64(0).data, 0x0000_0000_0000_7788);
    // 64 MiB of memory keeps 16 MiB of storage, repeated across the stride.
    assert_eq!(door.read64(0x100_0000).data, 0x7788);
    // A bank that is not fitted reads the pattern and drops writes.
    door.write64(0x800_0000, 5);
    assert_eq!(door.read64(0x800_0000).data, 0xa5a5_a5a5_a5a5_a5a5);
    assert_eq!(door.read8(0x800_0000).data, 0xa5);
}
