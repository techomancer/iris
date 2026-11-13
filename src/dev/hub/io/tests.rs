// Cases follow tests/qtest/sn0-hub-test.c in jamesbraid/qemu-sgi-origami:
// identifies-itself, widget-control-reset, io-access-registers,
// crosstalk-timeouts, io-scratch-registers, xio-link-reports-up,
// link-state-is-not-writable, link-log-serves-a-clean-link,
// link-enabled-bit-is-writable, the inbound-link-reset cases,
// translation-entries-decode, translation-entry-width, the big-window-*
// cases, io-queue-registers, io-icrb-c-family and the bte-* cases.

use super::*;
use std::collections::BTreeMap;

/// Memory as the engines see it: 1 MiB of RAM at node 0, plus an interrupt
/// log. Any address outside it fails.
struct Fabric {
    ram: Mutex<Vec<u8>>,
    interrupts: Mutex<Vec<(u16, u64)>>,
}
impl Fabric {
    fn new() -> Arc<Self> {
        Arc::new(Self { ram: Mutex::new(vec![0; 1 << 20]), interrupts: Mutex::new(Vec::new()) })
    }
    fn fill(&self, at: usize, bytes: &[u8]) {
        self.ram.lock()[at..at + bytes.len()].copy_from_slice(bytes);
    }
    fn bytes(&self, at: usize, n: usize) -> Vec<u8> {
        self.ram.lock()[at..at + n].to_vec()
    }
}
impl NodeFabric for Fabric {
    fn read(&self, address: u64, buf: &mut [u8]) -> bool {
        let ram = self.ram.lock();
        let at = address as usize;
        if address >= 1 << 32 || at + buf.len() > ram.len() {
            return false;
        }
        buf.copy_from_slice(&ram[at..at + buf.len()]);
        true
    }
    fn write(&self, address: u64, data: &[u8]) -> bool {
        let mut ram = self.ram.lock();
        let at = address as usize;
        if address >= 1 << 32 || at + data.len() > ram.len() {
            return false;
        }
        ram[at..at + data.len()].copy_from_slice(data);
        true
    }
    fn interrupt(&self, number: u16, value: u64) -> bool {
        if number == 7 {
            return false;
        }
        self.interrupts.lock().push((number, value));
        true
    }
}

/// The I/O unit with a fabric behind it. A write that launches a block
/// transfer is run on the fabric at once, as the Hub does after the write.
struct Rig {
    io: Io,
    fabric: Arc<Fabric>,
}
impl std::ops::Deref for Rig {
    type Target = Io;
    fn deref(&self) -> &Io {
        &self.io
    }
}
impl Rig {
    fn write64(&self, off: u64, value: u64) -> u32 {
        if let Effect::Bte(engine, launch) = self.io.write(off, value) {
            self.io.run_bte(engine, launch, &*self.fabric);
        }
        BUS_OK
    }
}
fn io() -> (Rig, Arc<Fabric>) {
    let fabric = Fabric::new();
    (Rig { io: Io::new(1, 0x36), fabric: fabric.clone() }, fabric)
}
fn r(io: &Io, off: u64) -> u64 {
    io.read64(off).data
}

/// A widget whose registers store what is written, so a window's target
/// can be told apart.
#[derive(Default)]
struct Widget {
    id: u64,
    regs: Mutex<BTreeMap<u64, u64>>,
    reads: Mutex<Vec<u64>>,
}
impl BusDevice for Widget {
    fn read64(&self, a: u64) -> BusRead64 {
        self.reads.lock().push(a);
        BusRead64::ok(self.regs.lock().get(&a).copied().unwrap_or(self.id))
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        self.regs.lock().insert(a, v);
        BUS_OK
    }
}

#[test]
fn the_widget_identifies_itself_and_refuses_writes() {
    let (io, _) = io();
    // Revision 1, part 0xc101, manufacturer 0x036, and the always-one bit.
    assert_eq!(r(&io, reg::WIDGET_ID), 0x1c10_106d);
    io.write64(reg::WIDGET_ID, 0);
    assert_eq!(r(&io, reg::WIDGET_ID), 0x1c10_106d);
    // The revision and manufacturer are the machine's to choose and survive reset.
    let other = Io::new(5, 0x123);
    assert_eq!(other.widget_id(), (5 << 28) | (0xc101 << 12) | (0x123 << 1) | 1);
    other.power_on();
    assert_eq!(other.widget_id(), (5 << 28) | (0xc101 << 12) | (0x123 << 1) | 1);
    // The status register always reads zero; a write is logged.
    assert_eq!(r(&io, reg::WIDGET_STATUS), 0);
    io.write64(reg::WIDGET_STATUS, !0);
    assert_eq!(r(&io, reg::WIDGET_STATUS), 0);
}

#[test]
fn the_widget_control_register_resets_to_widget_one_with_a_credit_of_two() {
    let (io, _) = io();
    assert_eq!(r(&io, reg::WIDGET_CONTROL), 0x4001);
    // Bits 22:0 store, except bit 4 (tag mode), which reads zero.
    io.write64(reg::WIDGET_CONTROL, !0);
    assert_eq!(r(&io, reg::WIDGET_CONTROL), 0x7f_ffef);
    io.power_on();
    assert_eq!(r(&io, reg::WIDGET_CONTROL), 0x4001);
}

#[test]
fn the_access_registers_store_and_enforce_nothing() {
    let (io, _) = io();
    assert_eq!(r(&io, reg::LOCAL_ACCESS), !0);
    assert_eq!(r(&io, reg::DEVICE_ERR_MASK), !0);
    assert_eq!((r(&io, reg::OUTBOUND_ACCESS), r(&io, reg::INBOUND_ACCESS)), (0, 0));
    // The PROM opens widget 0 and widgets 8 to 15.
    io.write64(reg::LOCAL_ACCESS, !0);
    io.write64(reg::OUTBOUND_ACCESS, 0xff01);
    io.write64(reg::INBOUND_ACCESS, !0);
    assert_eq!((r(&io, reg::OUTBOUND_ACCESS), r(&io, reg::INBOUND_ACCESS)), (0xff01, 0xff01));
    // The guest kernel removes and re-enables a device by clearing and setting its bit.
    io.write64(reg::DEVICE_ERR_MASK, !(1 << 9));
    assert_eq!(r(&io, reg::DEVICE_ERR_MASK), !(1 << 9));
    // The interrupt destination keeps 29 bits, mask 0x1101ff7f.
    io.write64(reg::INT_DEST, !0);
    assert_eq!(r(&io, reg::INT_DEST), 0x1101_ff7f);
    // The password write is accepted and reads zero.
    io.write64(reg::ACCESS_OVERRIDE, 0x5347_4972_756c_6573);
    assert_eq!(r(&io, reg::ACCESS_OVERRIDE), 0);
    io.power_on();
    assert_eq!((r(&io, reg::LOCAL_ACCESS), r(&io, reg::OUTBOUND_ACCESS), r(&io, reg::INT_DEST)), (!0, 0, 0));
}

#[test]
fn the_crosstalk_timeouts_accept_a_write_at_reset() {
    let (io, _) = io();
    // The first registers Linux writes: 0xff and 0x800.
    io.write64(reg::XTALK_TIMEOUT, 0xff);
    io.write64(reg::XTALK_PRESCALE, 0x800);
    assert_eq!((r(&io, reg::XTALK_TIMEOUT), r(&io, reg::XTALK_PRESCALE)), (0xff, 0x800));
    io.write64(reg::XTALK_TIMEOUT, !0);
    io.write64(reg::XTALK_PRESCALE, !0);
    assert_eq!((r(&io, reg::XTALK_TIMEOUT), r(&io, reg::XTALK_PRESCALE)), (0xff, 0xff_ffff));
}

#[test]
fn the_scratch_registers_store_36_bits_and_the_graphics_map_21() {
    let (io, _) = io();
    for (off, mask) in [(reg::SCRATCH_0, 0xf_00f1_1fff), (reg::SCRATCH_1, 0xf_00f1_1fff)] {
        assert_eq!(r(&io, off), 0);
        io.write64(off, !0);
        assert_eq!(r(&io, off), mask);
    }
    // The PROM uses scratch 0 as boot state: bit 23 is its alive signal, and
    // bits outside the mask do not store.
    io.write64(reg::SCRATCH_0, 1 << 23);
    assert_eq!(io.scratch(0), 1 << 23);
    io.write64(reg::SCRATCH_0, 1 << 36);
    assert_eq!(io.scratch(0), 0);
    // Graphics node-widget map: widget 3:0, node 12:4, slot 16, valid 20.
    io.write64(reg::GFX_INIT_0, 0x12_3456);
    io.write64(reg::GFX_INIT_1, !0);
    assert_eq!((r(&io, reg::GFX_INIT_0), io.gfx_init(1)), (0x12_3456, 0x1f_ffff));
    io.power_on();
    assert_eq!((io.scratch(0), io.gfx_init(0), io.gfx_init(1)), (0, 0, 0));
}

#[test]
fn the_link_reads_up_with_its_limits_and_only_the_controls_write() {
    let (io, _) = io();
    // Maximum burst 0x3fc, maximum retry 0x3ff, null timeout 0xa, state 2.
    assert_eq!(r(&io, reg::LINK_STATUS), 0x3fc_03ff_200a);
    assert_ne!(r(&io, reg::LINK_STATUS) & (2 << 12), 0, "bit 13 alone answers is-the-link-up");
    // The state field cannot be written down.
    io.write64(reg::LINK_STATUS, 0);
    assert_eq!(r(&io, reg::LINK_STATUS) & (3 << 12), 2 << 12);
    // Link enable (bit 10) resets clear and writes; firmware sets it by
    // read-modify-write.
    assert_eq!(r(&io, reg::LINK_STATUS) & (1 << 10), 0);
    io.write64(reg::LINK_STATUS, r(&io, reg::LINK_STATUS) | (1 << 10));
    assert_ne!(r(&io, reg::LINK_STATUS) & (1 << 10), 0);
    // The link log counts errors the link does not commit: zero, and
    // closing the interval with a write changes nothing.
    assert_eq!(r(&io, reg::LLP_LOG), 0);
    io.write64(reg::LLP_LOG, !0);
    assert_eq!(r(&io, reg::LLP_LOG), 0);
}

#[test]
fn a_warm_reset_is_a_pulse_that_reads_zero() {
    let (io, _) = io();
    // With nothing at the far end the write is only logged.
    io.write64(reg::LINK_STATUS, 1 << 8);
    assert_eq!(r(&io, reg::LINK_STATUS) & (1 << 8), 0);
    // Setting bit 10 by read-modify-write would pulse again only if bit 8
    // read back set.
    io.write64(reg::LINK_STATUS, r(&io, reg::LINK_STATUS) | (1 << 10));
    assert_eq!(r(&io, reg::LINK_STATUS) & (1 << 8), 0);
    // An inbound link reset leaves everything as it was.
    let before = (r(&io, reg::LINK_STATUS), r(&io, reg::WIDGET_CONTROL));
    io.link_reset_in();
    assert_eq!((r(&io, reg::LINK_STATUS), r(&io, reg::WIDGET_CONTROL)), before);
}

#[test]
fn the_translation_entries_are_thirteen_bits_and_reset_to_widget_zero() {
    let (io, _) = io();
    for n in 0..7 {
        let off = reg::BWIN_TABLE + n * 8;
        assert_eq!(r(&io, off), 0);
        io.write64(off, 0x1abc | (n << 5));
        assert_eq!(r(&io, off), (0x1abc | (n << 5)) & 0x1fff);
    }
    // An eighth word is not an entry.
    io.write64(reg::BWIN_TABLE + 7 * 8, 5);
    assert_eq!(r(&io, reg::BWIN_TABLE + 7 * 8), 0);
    io.power_on();
    assert_eq!(r(&io, reg::BWIN_TABLE), 0);
}

#[test]
fn a_big_window_follows_its_entry_to_a_widget_and_slice() {
    let (io, _) = io();
    let wins = io.big_windows();
    let w3 = Arc::new(Widget { id: 3, ..Default::default() });
    let w8 = Arc::new(Widget { id: 8, ..Default::default() });
    io.connect_widget(3, w3.clone());
    io.connect_widget(8, w8.clone());
    // Entry zero names widget 0, which is not there: a read gets zero and a
    // write is dropped.
    assert_eq!(wins.read64(0x10).data, 0);
    assert_eq!(wins.write64(0x10, 5), BUS_OK);
    assert!(w3.regs.lock().is_empty());
    // Aim window 2 at widget 3: bits 11:8 the widget, 4:0 the slice.
    io.write64(reg::BWIN_TABLE + 2 * 8, 3 << 8);
    assert_eq!(wins.read64(2 * BIG_WINDOW_SIZE + 0x40).data, 3);
    assert_eq!(w3.reads.lock().as_slice(), [0x40]);
    // The slice supplies XIO address bits 33:29.
    io.write64(reg::BWIN_TABLE + 2 * 8, (8 << 8) | 5);
    wins.write64(2 * BIG_WINDOW_SIZE + 0x48, 0xabc);
    assert_eq!(w8.regs.lock().get(&((5 << 29) | 0x48)), Some(&0xabc));
    // Re-aiming moves the next access. Widget 8 serves the window now.
    assert_eq!(wins.read64(2 * BIG_WINDOW_SIZE + 0x48).data, 0xabc);
    // The last byte of the window, and nothing beyond the seventh.
    assert_eq!(wins.read64(3 * BIG_WINDOW_SIZE - 8).status, BUS_OK);
    assert_eq!(wins.read64(7 * BIG_WINDOW_SIZE).status, BUS_ERR);
    // The other windows are unaffected.
    assert_eq!(wins.read64(BIG_WINDOW_SIZE).data, 0);
}

#[test]
fn only_big_window_zero_at_slice_zero_reaches_io_space() {
    let (io, _) = io();
    let wins = io.big_windows();
    let packets = Arc::new(Widget { id: 0x1234, ..Default::default() });
    io.connect_packets(packets.clone());
    // A peer Hub's identity probe: window 0, I/O space, widget 1.
    io.write64(reg::BWIN_TABLE, (1 << 12) | (1 << 8));
    assert_eq!(wins.read64(0x1_0000).data, 0x1234);
    assert_eq!(packets.reads.lock().as_slice(), [(1 << 48) | (1 << 47) | 0x1_0000]);
    // Any other I/O-space entry reaches nothing: zero and a dropped write.
    io.write64(reg::BWIN_TABLE, (1 << 12) | (1 << 8) | 1);
    assert_eq!(wins.read64(0).data, 0);
    io.write64(reg::BWIN_TABLE + 8, (1 << 12) | (1 << 8));
    assert_eq!(wins.read64(BIG_WINDOW_SIZE).data, 0);
    assert_eq!(packets.reads.lock().len(), 1);
}

#[test]
fn the_small_windows_reach_a_widget_by_number() {
    let (io, _) = io();
    let w = Arc::new(Widget { id: 9, ..Default::default() });
    io.connect_widget(9, w.clone());
    let small = io.small_windows();
    assert_eq!(small.read64(9 << 24 | 0x20).data, 9);
    assert_eq!(w.reads.lock().as_slice(), [0x20]);
    assert_eq!(small.write64(9 << 24 | 0x28, 1), BUS_OK);
    // An empty widget fails; widget 16 is not a widget.
    assert_eq!(small.read64(10 << 24).status, BUS_ERR);
    assert_eq!(small.read64(16 << 24).status, BUS_ERR);
}

#[test]
fn a_peers_read_of_the_widget_id_is_the_one_access_the_widget_face_answers() {
    let (io, _) = io();
    assert_eq!(io.widget_face_read32(4), Some(0x1c10_106d));
    for off in [0, 8, 0x20] {
        assert_eq!(io.widget_face_read32(off), None);
    }
}

/// Nothing is ever in flight: the queue and request-buffer registers read
/// zero, and their writes complete at once.
#[test]
fn the_queue_and_request_buffer_registers_read_zero() {
    let (io, _) = io();
    for off in [reg::PIO_DEALLOC, reg::CRB_DEALLOC, reg::IOQ_FIFO_DEPTH, reg::CRB_MGMT, reg::CRB_CONTROL] {
        io.write64(off, !0);
        assert_eq!(r(&io, off), 0, "{off:#x}");
    }
    for n in 0..8 {
        let off = reg::ICRB_C_BASE + n * 0x20;
        io.write64(off, !0);
        assert_eq!(r(&io, off), 0, "{off:#x}");
    }
    // The error-shutdown enable reads zero.
    io.write64(reg::MISC_ENABLE, !0);
    assert_eq!(r(&io, reg::MISC_ENABLE), 0);
}

// ---- the block transfer engines ----

fn engine(n: u64) -> u64 {
    reg::BTE_BASE + n * BTE_STRIDE
}
fn start(io: &Rig, n: u64, source: u64, dest: u64, lines: u64, control: u64) {
    io.write64(engine(n) + bte::SOURCE, source);
    io.write64(engine(n) + bte::DEST, dest);
    io.write64(engine(n) + bte::LENGTH, BTE_BUSY | lines);
    io.write64(engine(n) + bte::CONTROL, control);
}

#[test]
fn the_engines_reset_to_zero_and_keep_their_fields() {
    let (io, _) = io();
    for n in 0..2 {
        for off in (0..=0x28).step_by(8) {
            assert_eq!(r(&io, engine(n) + off), 0);
        }
    }
    io.write64(engine(1) + bte::LENGTH, !0);
    assert_eq!(r(&io, engine(1) + bte::LENGTH), BTE_BUSY | 0xffff);
    assert_eq!(r(&io, engine(0) + bte::LENGTH), 0, "the engines are independent");
    // Addresses keep bits 39:7 and drop the rest silently.
    io.write64(engine(0) + bte::SOURCE, 0xffff_ffff_ffff_ffff);
    io.write64(engine(0) + bte::NOTIFY, 0x1_2345_6780 | 0x7f);
    assert_eq!(r(&io, engine(0) + bte::SOURCE), 0xff_ffff_ff80);
    assert_eq!(r(&io, engine(0) + bte::NOTIFY), 0x1_2345_6780);
    // Interrupt: level 22:16, node 8:0.
    io.write64(engine(0) + bte::INTERRUPT, !0);
    assert_eq!(r(&io, engine(0) + bte::INTERRUPT), (0x7f << 16) | 0x1ff);
    // Control keeps poison, notify and zero-fill only; busy clear launches nothing.
    io.write64(engine(0) + bte::CONTROL, !0);
    assert_eq!(r(&io, engine(0) + bte::CONTROL), 0x111);
    io.power_on();
    assert_eq!(r(&io, engine(0) + bte::SOURCE), 0);
}

#[test]
fn a_launch_copies_lines_and_interrupts_the_node_it_names() {
    let (io, fabric) = io();
    let data: Vec<u8> = (0..512u32).map(|i| (i * 7) as u8).collect();
    fabric.fill(0x1000, &data);
    io.write64(engine(0) + bte::INTERRUPT, (9 << 16) | 3);
    start(&io, 0, 0x1000, 0x8_0000, 4, 0);
    assert_eq!(fabric.bytes(0x8_0000, 512), data);
    // Busy and the count clear when the transfer completes.
    assert_eq!(r(&io, engine(0) + bte::LENGTH), 0);
    // Completion arrives as a device interrupt at level 9 of node 3.
    assert_eq!(fabric.interrupts.lock().as_slice(), [(3, MODIFY_SET | 9)]);
}

#[test]
fn notify_mode_writes_the_length_word_to_the_notification_address() {
    let (io, fabric) = io();
    fabric.fill(0x2000, &[0xab; 256]);
    io.write64(engine(1) + bte::NOTIFY, 0x4_0000);
    io.write64(engine(1) + bte::SOURCE, 0x2000);
    io.write64(engine(1) + bte::DEST, 0x3000);
    io.write64(engine(1) + bte::LENGTH, BTE_BUSY | 2);
    io.write64(engine(1) + bte::CONTROL, BTE_NOTIFY);
    assert_eq!(fabric.bytes(0x3000, 256), vec![0xab; 256]);
    // Exactly eight bytes, big-endian: the status after completion is zero.
    assert_eq!(fabric.bytes(0x4_0000, 9), [0; 9]);
    assert!(fabric.interrupts.lock().is_empty());
    // A later error shows in the word.
    fabric.fill(0x4_0000, &[0xff; 8]);
    io.write64(engine(1) + bte::LENGTH, BTE_BUSY | 1);
    io.write64(engine(1) + bte::CONTROL, BTE_NOTIFY);
    assert_eq!(fabric.bytes(0x4_0000, 8), [0; 8]);
}

#[test]
fn zero_fill_clears_the_destination_without_reading_the_source() {
    let (io, fabric) = io();
    fabric.fill(0x5000, &[0xee; 256]);
    // The source is outside memory: a copy would fail, a fill does not read it.
    start(&io, 0, 0xff_ffff_ff80, 0x5000, 2, BTE_ZFILL);
    assert_eq!(fabric.bytes(0x5000, 256), vec![0; 256]);
    assert_eq!(r(&io, engine(0) + bte::LENGTH), 0);
}

#[test]
fn overlapping_ranges_copy_forward_a_line_at_a_time() {
    let (io, fabric) = io();
    let line = |b: u8| vec![b; 128];
    for i in 0..4u8 {
        fabric.fill(0x6000 + usize::from(i) * 128, &line(i + 1));
    }
    // Destination one line above the source: forward propagation smears
    // the first line through the range.
    start(&io, 0, 0x6000, 0x6080, 3, 0);
    for i in 0..4 {
        assert_eq!(fabric.bytes(0x6000 + i * 128, 128), line(1), "line {i}");
    }
}

#[test]
fn a_failed_transfer_latches_error_and_the_unmoved_count_until_it_is_cleared() {
    let (io, fabric) = io();
    fabric.fill(0x7000, &[1; 128]);
    // Memory ends at 1 MiB: the third line of the destination fails.
    start(&io, 0, 0x7000, (1 << 20) - 256, 4, 0);
    let status = r(&io, engine(0) + bte::LENGTH);
    assert_eq!(status, BTE_BUSY | BTE_ERROR | 2, "two lines were not moved");
    assert!(fabric.interrupts.lock().is_empty(), "neither completion runs");
    // A store arms or disarms but never absolves.
    io.write64(engine(0) + bte::LENGTH, 0);
    assert_eq!(r(&io, engine(0) + bte::LENGTH), BTE_ERROR);
    // Only IIO_IECLR clears it, and per engine.
    io.write64(reg::ERROR_CLEAR, IECLR_BTE[1]);
    assert_eq!(r(&io, engine(0) + bte::LENGTH), BTE_ERROR);
    io.write64(reg::ERROR_CLEAR, IECLR_BTE[0]);
    assert_eq!(r(&io, engine(0) + bte::LENGTH), 0);
    // An unclaimed completion node fails soft: the interrupt is dropped.
    io.write64(engine(0) + bte::INTERRUPT, 7);
    start(&io, 0, 0x7000, 0x7800, 1, 0);
    assert_eq!(r(&io, engine(0) + bte::LENGTH), 0);
}

#[test]
fn a_zero_length_launch_completes_at_once_and_busy_clear_launches_nothing() {
    let (io, fabric) = io();
    fabric.fill(0x9000, &[9; 128]);
    start(&io, 0, 0x9000, 0x9800, 0, 0);
    assert_eq!(r(&io, engine(0) + bte::LENGTH), 0);
    assert_eq!(fabric.bytes(0x9800, 128), vec![0; 128]);
    // A control store with busy clear stores and does nothing.
    io.write64(engine(0) + bte::LENGTH, 1);
    io.write64(engine(0) + bte::CONTROL, 0);
    assert_eq!(fabric.bytes(0x9800, 128), vec![0; 128]);
    assert_eq!(r(&io, engine(0) + bte::LENGTH), 1);
}
