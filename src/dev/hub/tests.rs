// Cases follow the sn0-hub, inbound, pi-error and widget-decode qtests in
// jamesbraid/qemu-sgi-origami.

use super::*;
use parking_lot::Mutex;
use crate::traits::MachineEvent;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// A node's memory as the remap maps it at the node's number: big-endian
/// bytes at the offset in the node, failing past its end.
struct Ram(Mutex<Vec<u8>>, #[cfg(feature = "jitv2")] std::sync::atomic::AtomicU64);
impl Ram {
    fn new(size: usize, seed: u8) -> Arc<Self> {
        Arc::new(Self(
            Mutex::new((0..size).map(|i| seed ^ (i as u8)).collect()),
            #[cfg(feature = "jitv2")]
            Default::default(),
        ))
    }

    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        let a = a & (NODE_SPAN - 1);
        let mut m = self.0.lock();
        let Some(bytes) = m.get_mut(a as usize..a as usize + size) else { return BusRead64::err() };
        match value {
            Some(v) => bytes.copy_from_slice(&v.to_be_bytes()[8 - size..]),
            None => return BusRead64::ok(bytes.iter().fold(0, |v, &b| (v << 8) | u64::from(b))),
        }
        BusRead64::ok(0)
    }
}
impl BusDevice for Ram {
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, _a: u64) -> *const std::sync::atomic::AtomicU64 {
        &self.1
    }
    crate::dev::widths::access_widths!();
}

/// A few byte registers, standing in for the I2C controller.
struct Bytes(Mutex<[u8; 64]>);
impl Default for Bytes {
    fn default() -> Self {
        Self(Mutex::new([0; 64]))
    }
}
impl BusDevice for Bytes {
    fn read64(&self, a: u64) -> BusRead64 {
        BusRead64::ok(u64::from(self.0.lock()[a as usize]))
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        self.0.lock()[a as usize] = v as u8;
        BUS_OK
    }
    fn read8(&self, a: u64) -> BusRead8 {
        BusRead8::ok(self.0.lock()[a as usize])
    }
    fn write8(&self, a: u64, v: u8) -> u32 {
        self.0.lock()[a as usize] = v;
        BUS_OK
    }
}

const RAM_SIZE: usize = 1 << 20;

/// A Hub with one 1 MiB bank, byte `i` of which holds `seed ^ i`.
fn seeded(cfg: HubConfig, seed: u8) -> Arc<Hub> {
    let hub = Hub::new(HubConfig { banks_mb: [1, 0, 0, 0], nodes: 2, ..cfg }, Arc::new(|| 0), None).unwrap();
    let bank = &hub.decode.physical().banks()[0];
    for i in (0..RAM_SIZE as u64).step_by(8) {
        bank.write64(i, u64::from_be_bytes(std::array::from_fn(|b| seed ^ (i as u8).wrapping_add(b as u8))));
    }
    hub
}
fn hub(cpus: u8, ni_revision: u8) -> Arc<Hub> {
    seeded(HubConfig { cpus, ni_revision, ..Default::default() }, 0)
}
/// Two Hubs with their NI ports connected, each with a node memory that tells it apart.
fn pair() -> (Arc<Hub>, Arc<Hub>) {
    let a = hub(2, 3);
    let b = seeded(HubConfig { cpus: 2, first_cpu: 2, ..Default::default() }, 0x80);
    a.connect_peer(&b);
    b.connect_peer(&a);
    (a, b)
}
fn r64(d: &dyn BusDevice, a: u64) -> u64 {
    d.read64(a).data
}
const NODE: fn(u64) -> u64 = |n| n << NODE_SHIFT;
fn number(hub: &Hub, n: u64) {
    assert_eq!(hub.write64(regs::NI_STATUS_REV_ID::OFFSET, n << 8), BUS_OK);
}

#[test]
fn a_32_bit_access_selects_the_big_endian_half() {
    let hub = hub(1, 3);
    let status = r64(&*hub, regs::NI_STATUS_REV_ID::OFFSET);
    // The reset value is all in the low word, so a swap would show.
    assert_ne!(status, 0);
    assert_eq!(u64::from(hub.read32(regs::NI_STATUS_REV_ID::OFFSET).data), status >> 32);
    assert_eq!(u64::from(hub.read32(regs::NI_STATUS_REV_ID::OFFSET + 4).data), status & 0xffff_ffff);
    // A register with reset bits in both halves separates them sharply.
    assert_eq!(hub.read32(regs::MD_MEMORY_CONFIG::OFFSET).data, 0x001e_7fbc);
    assert_eq!(hub.read32(regs::MD_MEMORY_CONFIG::OFFSET + 4).data, 0x3fff_ffff);
    // A 32-bit write replaces its own half and leaves the other alone.
    hub.write64(regs::NI_SCRATCH_REG0::OFFSET, 0x1111_2222_3333_4444);
    hub.write32(regs::NI_SCRATCH_REG0::OFFSET, 0xaaaa_bbbb);
    assert_eq!(r64(&*hub, regs::NI_SCRATCH_REG0::OFFSET), 0xaaaa_bbbb_3333_4444);
    hub.write32(regs::NI_SCRATCH_REG0::OFFSET + 4, 0xcccc_dddd);
    assert_eq!(r64(&*hub, regs::NI_SCRATCH_REG0::OFFSET), 0xaaaa_bbbb_cccc_dddd);
    // A half write still passes through the field mask (bits 31:3).
    hub.write32(regs::PI_ERR_STACK_ADDR_A::OFFSET + 4, 0xdead_beef);
    assert_eq!(r64(&*hub, regs::PI_ERR_STACK_ADDR_A::OFFSET), 0xdead_bee8);
    hub.write32(regs::PI_ERR_STACK_ADDR_A::OFFSET, 0xffff_ffff);
    assert_eq!(r64(&*hub, regs::PI_ERR_STACK_ADDR_A::OFFSET), 0xdead_bee8);
    // Only whole words, and only inside the window.
    assert_eq!(hub.read32(2).status, BUS_ERR);
    assert_eq!(hub.write32(WINDOW_SIZE, 0), BUS_ERR);
    assert_eq!(hub.read64(4).status, BUS_ERR);
    assert_eq!(hub.read64(WINDOW_SIZE).status, BUS_ERR);
    assert_eq!(hub.read8(0x18).status, BUS_ERR);
}

/// The vector clear alias reads zero.
#[test]
fn a_32_bit_read_of_a_read_to_clear_register_does_not_clear_it() {
    let (a, b) = pair();
    b.write64(regs::NI_SCRATCH_REG0::OFFSET, 7);
    a.write64(regs::NI_VECTOR::OFFSET, 0);
    a.write64(regs::NI_VECTOR_PARMS::OFFSET, regs::NI_SCRATCH_REG0::OFFSET - regs::NI_STATUS_REV_ID::OFFSET);
    let status = r64(&*a, regs::NI_VECTOR_STATUS::OFFSET);
    assert_ne!(status, 0);
    assert_eq!(a.read32(regs::NI_VECTOR_CLEAR::OFFSET).data, 0);
    assert_eq!(r64(&*a, regs::NI_VECTOR_STATUS::OFFSET), status);
    assert_eq!(r64(&*a, regs::NI_VECTOR_CLEAR::OFFSET), 0);
    assert_eq!(r64(&*a, regs::NI_VECTOR_STATUS::OFFSET), 0);
    // And a half write of the write-one-to-clear port error does not merge with the old value.
    a.write64(regs::NI_PORT_RESET::OFFSET, 2);
    assert_eq!(r64(&*a, regs::NI_PORT_ERROR::OFFSET), 1 << 37, "a link reset latches bit 37");
    a.write32(regs::NI_PORT_ERROR::OFFSET + 4, 0x2);
    assert_eq!(r64(&*a, regs::NI_PORT_ERROR::OFFSET), 1 << 37);
    // The interrupt-modify register stores nothing: the other half is zero.
    a.write32(regs::PI_INT_PEND_MOD::OFFSET + 4, (pi::MODIFY_SET | 9) as u32);
    assert_eq!(r64(&*a, regs::PI_INT_PEND0::OFFSET), 1 << 9);
}

#[test]
fn every_section_answers_through_the_window_and_the_rest_reads_zero() {
    let hub = hub(2, 5);
    assert_eq!(r64(&*hub, regs::PI_CALIAS_SIZE::OFFSET), 15);
    assert_eq!(r64(&*hub, regs::MD_MEMORY_CONFIG::OFFSET), 0x001e_7fbc_3fff_ffff);
    assert_eq!(r64(&*hub, regs::IIO_WID::OFFSET), 0x1c10_106d);
    assert_eq!(r64(&*hub, regs::NI_STATUS_REV_ID::OFFSET) & 0xff, 5 << 4);
    assert!(r64(&*hub, regs::PI_RT_COUNT::OFFSET) < 1_000);
    for off in [0x10_0000, 0x30_0000, 0x50_0000, 0x70_0000, 0x7f_fff8] {
        assert_eq!(hub.write64(off, !0), BUS_OK);
        assert_eq!(r64(&*hub, off), 0, "{off:#x}");
    }
}

#[test]
fn the_i2c_controller_takes_narrow_accesses_over_the_md_block() {
    let hub = hub(1, 3);
    // With nothing attached, narrow accesses fail like any other.
    assert_eq!(hub.read8(regs::MD_UREG0_0::OFFSET).status, BUS_ERR);
    let i2c = Arc::new(Bytes::default());
    hub.attach_i2c(i2c.clone());
    // The PCF8584's S0 and S1, eight bytes apart.
    assert_eq!(hub.write8(regs::MD_UREG0_0::OFFSET, 0x5a), BUS_OK);
    assert_eq!(hub.write8(regs::MD_UREG0_0::OFFSET + 8, 0x3c), BUS_OK);
    let (s0, s1) = {
        let regs = i2c.0.lock();
        (regs[0], regs[8])
    };
    assert_eq!((s0, s1), (0x5a, 0x3c));
    assert_eq!(hub.read8(regs::MD_UREG0_0::OFFSET).data, 0x5a);
    // The registers between them are still the MD block's.
    assert_eq!(r64(&*hub, regs::MD_SLOTID_USTAT::OFFSET), 0x95);
    // A doubleword access to the overlay is one register access too.
    assert_eq!(hub.write64(regs::MD_UREG0_0::OFFSET, 1), BUS_OK);
    assert_eq!(i2c.0.lock()[0], 1);
    assert_eq!(hub.read64(regs::MD_UREG0_0::OFFSET + 8).data, 0x3c);
    i2c.0.lock()[0] = 0x81;
    assert_eq!(hub.read64(regs::MD_UREG0_0::OFFSET).data, 0x81, "an 8-byte read reaches the PCF8584 status");
}


#[test]
fn a_restart_reaches_the_machines_receiver_and_a_soft_reset_the_processor() {
    assert_eq!(hub(1, 3).write64(regs::NI_PORT_RESET::OFFSET, 0x81), BUS_OK, "no listener: only logged");
    let hub = hub(2, 3);
    let (tx, rx) = std::sync::mpsc::channel();
    hub.set_event_sender(tx);
    let words = [AtomicU64::new(0), AtomicU64::new(0)];
    hub.pi.set_cpu_interrupts(0, &words[0]);
    hub.pi.set_cpu_interrupts(1, &words[1]);
    // Enable and NMI are stored and reach nothing.
    hub.write64(regs::PI_CPU_ENABLE_A::OFFSET, 1);
    hub.write64(regs::PI_NMI_B::OFFSET, 1);
    assert!(rx.try_recv().is_err());
    assert_eq!(words[0].load(Ordering::SeqCst) | words[1].load(Ordering::SeqCst), 0);
    // The soft reset reaches the enabled processor only.
    hub.write64(regs::PI_SOFTRESET::OFFSET, 1);
    assert_eq!(words[0].load(Ordering::SeqCst), crate::cpu::mips_exec::SOFT_RESET_BIT);
    assert_eq!(words[1].load(Ordering::SeqCst), 0);
    assert!(rx.try_recv().is_err(), "a soft reset restarts only the processor");
    // A local port reset restarts the machine.
    hub.write64(regs::NI_PORT_RESET::OFFSET, 0x81);
    assert!(matches!(rx.try_recv(), Ok(MachineEvent::HardReset)));
    assert!(rx.try_recv().is_err());
    // A link reset alone does not.
    hub.write64(regs::NI_PORT_RESET::OFFSET, 0x2);
    assert!(rx.try_recv().is_err());
    hub.pi.clear_cpu_interrupts();
}

#[test]
fn reset_restores_the_registers_and_keeps_what_firmware_parks() {
    let hub = hub(2, 3);
    hub.write64(regs::PI_ERR_STACK_ADDR_B::OFFSET, 0xabcd_0000);
    hub.write64(regs::PI_CALIAS_SIZE::OFFSET, 1);
    hub.write64(regs::MD_MEMORY_CONFIG::OFFSET, 5);
    hub.write64(regs::IIO_SCRATCH_REG0::OFFSET, 6);
    number(&hub, 3);
    hub.power_on();
    assert_eq!(r64(&*hub, regs::PI_ERR_STACK_ADDR_B::OFFSET), 0xabcd_0000);
    assert_eq!(r64(&*hub, regs::PI_CALIAS_SIZE::OFFSET), 15);
    assert_eq!(r64(&*hub, regs::MD_MEMORY_CONFIG::OFFSET), 0x001e_7fbc_3fff_ffff);
    assert_eq!(r64(&*hub, regs::IIO_SCRATCH_REG0::OFFSET), 0);
    assert_eq!(hub.ni.number(), 0);
    let mem = hub.inbound();
    assert_eq!(mem.read8(0x10).data, 0x10, "node zero is this Hub's again");
    assert_eq!(mem.read8(NODE(3) + 0x10).status, BUS_ERR, "and node 3 is nobody's");
}


#[test]
fn an_unprogrammed_hub_claims_node_zero_and_a_programmed_one_moves() {
    let hub = hub(1, 3);
    let mem = hub.inbound();
    // Stock PROM Hubs start unnumbered: every Hub holds zero.
    assert_eq!(mem.read8(0x10).data, 0x10);
    assert_eq!(mem.read8(NODE(1) + 0x10).status, BUS_ERR, "no Hub holds node 1");
    // The boot master numbers the Hub: node 5 answers, and node 0 no longer does.
    number(&hub, 5);
    assert_eq!(mem.read8(NODE(5) + 0x10).data, 0x10);
    assert_eq!(mem.read8(0x10).status, BUS_ERR);
    assert_eq!(mem.read8(NODE(255) + 0x10).status, BUS_ERR);
    // The window is the node's, from its base, whatever its number.
    assert_eq!(mem.write32(NODE(5) + 0x20, 0xdead_beef), BUS_OK);
    assert_eq!(mem.read32(NODE(5) + 0x20).data, 0xdead_beef);
    assert_eq!(mem.read8(NODE(5) + RAM_SIZE as u64).status, BUS_ERR, "past the node's memory");
}

#[test]
fn two_connected_hubs_each_claim_the_others_programmed_number() {
    let (a, b) = pair();
    let (ma, mb) = (a.inbound(), b.inbound());
    // While both hold zero, each keeps zero for itself.
    assert_eq!(ma.read8(0x10).data, 0x10);
    assert_eq!(mb.read8(0x10).data, 0x90);
    // Numbering the peer makes it reachable by number from this side.
    number(&b, 1);
    assert_eq!(ma.read8(NODE(1) + 0x10).data, 0x90);
    assert_eq!(ma.read8(0x10).data, 0x10);
    // b no longer holds zero: node zero is a's, which b reaches by a's number.
    assert_eq!(mb.read8(0x10).data, 0x10);
    assert_eq!(mb.read8(NODE(1) + 0x10).data, 0x90);
    // Writes land in the Hub the number names.
    assert_eq!(ma.write8(NODE(1) + 0x40, 0x5a), BUS_OK);
    assert_eq!(mb.read8(NODE(1) + 0x40).data, 0x5a);
    // Renumbering moves the claim.
    number(&b, 2);
    assert_eq!(ma.read8(NODE(1) + 0x40).status, BUS_ERR);
    assert_eq!(ma.read8(NODE(2) + 0x40).data, 0x5a);
}

#[test]
fn the_hub_special_window_always_stays_reader_local() {
    let (a, b) = pair();
    a.decode.set_space(UncachedSpace::HubSpecial, Ram::new(64, 0x11));
    b.decode.set_space(UncachedSpace::HubSpecial, Ram::new(64, 0x22));
    number(&a, 3);
    number(&b, 4);
    let (sa, sb) = (a.cpu_view(crate::sn::UncachedSpace::HubSpecial), b.cpu_view(crate::sn::UncachedSpace::HubSpecial));
    // Each reader's node-zero address names its own node, not the number's holder.
    assert_eq!((sa.read8(1).data, sb.read8(1).data), (0x10, 0x23));
    // A node-qualified address is claimed by number.
    assert_eq!(sa.read8(NODE(4) + 1).data, 0x23);
}

// ---- the XIO port ----

#[test]
fn inbound_memory_reaches_the_nodes_memory_by_node_number() {
    let (a, b) = pair();
    number(&a, 1);
    number(&b, 2);
    let xio = a.inbound();
    // A widget's DMA of every size lands in node memory.
    let local = NODE(1) + 0x100;
    assert_eq!(xio.write64(local, 0x0123_4567_89ab_cdef), BUS_OK);
    assert_eq!(xio.read64(local).data, 0x0123_4567_89ab_cdef);
    assert_eq!(xio.read8(local).data, 0x01);
    assert_eq!(xio.read16(local + 2).data, 0x4567);
    assert_eq!(xio.read32(local + 4).data, 0x89ab_cdef);
    assert_eq!(xio.write16(local + 6, 0xfeed), BUS_OK);
    assert_eq!(xio.read64(local).data, 0x0123_4567_89ab_feed);
    assert_eq!(xio.read64(0x100).status, BUS_ERR);
    // A node-qualified address goes to the Hub holding the number.
    assert_eq!(xio.read8(NODE(2) + 8).data, 0x88);
    assert_eq!(xio.write8(NODE(2) + 9, 0x42), BUS_OK);
    assert_eq!(b.inbound().read8(NODE(2) + 9).data, 0x42);
    // Memory is below 2^40; the rest of the address is the I/O view.
    assert_eq!(xio.read8(1 << 40).status, BUS_ERR);
    assert_eq!(xio.read8(NODE(7)).status, BUS_ERR);
    assert_eq!(xio.read8(1 << 48).status, BUS_ERR);
}

/// A page of DMA reaches node memory as a block and ends up where the word accesses see it.
#[test]
fn inbound_dma_reaches_node_memory_a_block_at_a_time() {
    let hub = hub(1, 3);
    number(&hub, 1);
    let xio = hub.inbound();
    let at = NODE(1) + 0x2000;
    let page: Vec<u64> = (0..512u64).map(|i| i.wrapping_mul(0x0101_0101_0101_0101) ^ 0x1122_3344_5566_7788).collect();
    assert_eq!(xio.write_block(at, &page), BUS_OK);
    let mut back = vec![0u64; 512];
    assert_eq!(xio.read_block(at, &mut back), BUS_OK);
    assert_eq!(back, page);
    for (i, w) in page.iter().enumerate().step_by(97) {
        assert_eq!(xio.read64(at + 8 * i as u64).data, *w);
    }
    // Refused: past the memory or node, in the I/O view, or on a node nobody holds.
    let mut one = [0u64; 1];
    assert_eq!(xio.read_block(NODE(1) + RAM_SIZE as u64, &mut one), BUS_ERR);
    assert_eq!(xio.read_block(NODE(1) + NODE_SPAN - 8, &mut [0u64; 2]), BUS_ERR);
    assert_eq!(xio.read_block(1 << 47, &mut one), BUS_ERR);
    assert_eq!(xio.write_block(NODE(5), &one), BUS_ERR);
    assert_eq!(xio.read_block(1 << 40, &mut one), BUS_ERR);
}

#[test]
fn an_interrupt_modify_write_lands_on_the_hub_the_packet_names() {
    let (a, b) = pair();
    number(&a, 1);
    number(&b, 2);
    let xio = a.inbound();
    let packet = |node: u64| XIO_IO_VIEW | (node << XIO_NODE_SHIFT) | 0x0180_0090;
    // Local: set interrupt 70, which is bit 6 of the second bank.
    assert_eq!(xio.write64(packet(1), pi::MODIFY_SET | 70), BUS_OK);
    assert_eq!(a.pi.pending(1), 1 << 6);
    assert_eq!(b.pi.pending(1), 0);
    // Across the link, by the peer's programmed number.
    assert_eq!(xio.write64(packet(2), pi::MODIFY_SET | 9), BUS_OK);
    assert_eq!(b.pi.pending(0), 1 << 9);
    assert_eq!(xio.write64(packet(2), 9), BUS_OK);
    assert_eq!(b.pi.pending(0), 0);
    // A number nobody holds, a narrow write and any other offset fail.
    assert_eq!(xio.write64(packet(3), pi::MODIFY_SET | 1), BUS_ERR);
    assert_eq!(xio.write32(packet(1), 1), BUS_ERR);
    assert_eq!(xio.write64(packet(1) + 8, 1), BUS_ERR);
    assert_eq!(a.pi.pending(0), 0);
    // The Bridge's interrupt target reaches the same hubs by the same address.
    let target = a.interrupt_target();
    assert!(target.packet(9, packet(1), 3, true));
    assert!(target.packet(9, packet(2), 5, true));
    assert_eq!((a.pi.pending(0), b.pi.pending(0)), (1 << 3, 1 << 5), "node 2 heard it, not node 1");
    assert!(!target.packet(9, packet(3), 1, true));
    assert!(!target.packet(9, packet(1), 128, true), "a vector the Hub does not have");
    assert!(b.interrupt_target().packet(0, packet(1), 70, true));
    assert_eq!(a.pi.pending(1), 1 << 6);
}

#[test]
fn a_peers_identity_probe_reads_the_network_status_and_nothing_else_is_answered() {
    let (a, _) = pair();
    let xio = a.inbound();
    let probe = (1u64 << 47) | 0x0160_0000;
    assert_eq!(xio.read64(probe).data, a.ni.read(regs::NI_STATUS_REV_ID::OFFSET, false));
    assert_eq!(xio.read32(probe).status, BUS_ERR, "a doubleword read only");
    assert_eq!(xio.read64(probe + 8).status, BUS_ERR);
    assert_ne!(xio.write64(probe, 0), BUS_OK);
}


#[test]
fn a_peers_programmed_io_to_the_hubs_widget_reads_only_its_identity() {
    let hub = hub(1, 3);
    let port = hub.widget_port();
    assert_eq!(port.read32(4).data, 0x1c10_106d);
    for off in [0, 8, 0x20] {
        assert_eq!(port.read32(off).status, BUS_ERR);
    }
    assert_eq!(port.read64(4).status, BUS_ERR);
    assert_eq!(port.write32(4, 0), BUS_ERR);
}

#[test]
fn the_io_space_holds_the_registers_at_both_offsets_and_the_widgets_behind_them() {
    let hub = hub(1, 3);
    let io = hub.io_space();
    // Both the reader's own alias and the node-explicit window.
    let regs = [IO_REGISTERS_LOCAL, IO_REGISTERS_NAMED];
    for base in regs {
        assert_eq!(io.read64(base + regs::MD_MEMORY_CONFIG::OFFSET).data, 0x001e_7fbc_3fff_ffff);
    }
    io.write64(IO_REGISTERS_LOCAL + regs::IIO_SCRATCH_REG1::OFFSET, 0x1234);
    assert_eq!(io.read64(IO_REGISTERS_NAMED + regs::IIO_SCRATCH_REG1::OFFSET).data, 0x1234);
    assert_eq!(io.read32(IO_REGISTERS_NAMED + regs::IIO_SCRATCH_REG1::OFFSET + 4).data, 0x1234);
    // Past the second window the space is widget 2's.
    assert_eq!(io.read64(0x0200_0000).status, BUS_ERR);
    // A widget at its small window and through a big window.
    let widget = Arc::new(Bytes::default());
    hub.io.connect_widget(9, widget.clone());
    io.write8(9 << 24 | 5, 0x77);
    assert_eq!(widget.0.lock()[5], 0x77);
    hub.write64(regs::IIO_ITTE::OFFSET + 8, 9 << 8);
    io.write8(IO_BIG_WINDOWS + io::BIG_WINDOW_SIZE + 6, 0x66);
    assert_eq!(widget.0.lock()[6], 0x66);
    assert_eq!(io.read8(IO_BIG_WINDOWS + io::BIG_WINDOW_SIZE + 6).data, 0x66);
    assert_eq!(io.read8(IO_BIG_WINDOWS + 7 * io::BIG_WINDOW_SIZE).status, BUS_ERR);
}

#[test]
fn a_processors_own_register_alias_stays_on_its_node_whatever_the_number() {
    let (a, b) = pair();
    for hub in [&a, &b] {
        hub.decode.set_space(UncachedSpace::Io, hub.io_space());
    }
    number(&b, 1);
    let at = IO_REGISTERS_LOCAL + regs::IIO_SCRATCH_REG1::OFFSET;
    // The processor's view keeps node zero's alias on its own node.
    assert_eq!(a.write64(regs::IIO_SCRATCH_REG1::OFFSET, 0x11), BUS_OK);
    assert_eq!(b.cpu_view(crate::sn::UncachedSpace::Io).write64(at, 0x22), BUS_OK);
    assert_eq!(r64(&*b, regs::IIO_SCRATCH_REG1::OFFSET), 0x22);
    assert_eq!(r64(&*a, regs::IIO_SCRATCH_REG1::OFFSET), 0x11);
    // The node-explicit window still goes by node number.
    assert_eq!(b.cpu_view(crate::sn::UncachedSpace::Io).read64(IO_REGISTERS_NAMED + regs::IIO_SCRATCH_REG1::OFFSET).data, 0x11);
    assert_eq!(b.cpu_view(crate::sn::UncachedSpace::Io).read64(NODE(1) + IO_REGISTERS_NAMED + regs::IIO_SCRATCH_REG1::OFFSET).data, 0x22);
    // The alias is the one with node field zero.
    assert_eq!(b.cpu_view(crate::sn::UncachedSpace::Io).read64(NODE(0) + at).data, 0x22);
    assert_eq!(a.cpu_view(crate::sn::UncachedSpace::Io).write64(NODE(1) + at, 0x33), BUS_OK);
    assert_eq!(r64(&*b, regs::IIO_SCRATCH_REG1::OFFSET), 0x33);
    assert_eq!(r64(&*a, regs::IIO_SCRATCH_REG1::OFFSET), 0x11);
}

// ---- the engines through the node memory ----

#[test]
fn an_engine_addresses_memory_by_node_number_and_interrupts_the_node_it_names() {
    let (a, b) = pair();
    number(&a, 1);
    number(&b, 2);
    let (engine, bte) = (0, regs::IIO_IBLS_0::OFFSET);
    // Copy two lines from this node's memory into the peer's.
    a.write64(regs::IIO_IBSA_0::OFFSET, NODE(1) + 0x1000);
    a.write64(regs::IIO_IBDA_0::OFFSET, NODE(2) + 0x2000);
    a.write64(regs::IIO_IBIA_0::OFFSET, (17 << 16) | 2);
    a.write64(engine + bte, (1 << 20) | 2);
    a.write64(engine + regs::IIO_IBCT_0::OFFSET, 0);
    for i in 0..256 {
        assert_eq!(b.inbound().read8(NODE(2) + 0x2000 + i).data, a.inbound().read8(NODE(1) + 0x1000 + i).data);
    }
    // Busy and the count clear, and the completion interrupted node 2's Hub.
    assert_eq!(a.read64(engine + bte).data, 0);
    assert_eq!(b.pi.pending(0), 1 << 17);
    // An engine's addresses ignore the cached alias.
    a.write64(regs::PI_CALIAS_SIZE::OFFSET, 15);
    a.write64(engine + regs::IIO_IBSA_0::OFFSET, 0x1000);
    a.write64(engine + regs::IIO_IBDA_0::OFFSET, NODE(1) + 0x3000);
    a.write64(engine + bte, (1 << 20) | 1);
    a.write64(engine + regs::IIO_IBCT_0::OFFSET, 0);
    assert_eq!(a.read64(engine + bte).data, (1 << 20) | (1 << 16) | 1);
    // Completion aimed at a node nobody holds drops the interrupt softly.
    a.write64(engine + regs::IIO_IBLS_0::OFFSET, 0);
    a.write64(regs::IIO_IECLR::OFFSET, 1 << 17);
    a.write64(engine + regs::IIO_IBSA_0::OFFSET, NODE(1) + 0x1000);
    a.write64(engine + regs::IIO_IBIA_0::OFFSET, 9);
    a.write64(engine + bte, (1 << 20) | 1);
    a.write64(engine + regs::IIO_IBCT_0::OFFSET, 0);
    assert_eq!(a.read64(engine + bte).data, 0);
}

// ---- the processor's path ----

#[derive(Default)]
struct Downstream {
    fail: AtomicBool,
    busy: AtomicBool,
}
impl BusDevice for Downstream {
    fn read64(&self, a: u64) -> BusRead64 {
        if self.busy.load(Ordering::SeqCst) {
            BusRead64::busy()
        } else if self.fail.load(Ordering::SeqCst) {
            BusRead64::err()
        } else {
            BusRead64::ok(a)
        }
    }
    fn read32(&self, a: u64) -> BusRead32 {
        if self.busy.load(Ordering::SeqCst) {
            BusRead32::busy()
        } else if self.fail.load(Ordering::SeqCst) {
            BusRead32::err()
        } else {
            BusRead32::ok(a as u32)
        }
    }
    fn write64(&self, _a: u64, _v: u64) -> u32 {
        if self.fail.load(Ordering::SeqCst) { BUS_ERR } else { BUS_OK }
    }
}

#[test]
fn a_failed_processor_read_is_recorded_for_the_slice_that_issued_it() {
    let hub = hub(2, 3);
    let down = Arc::new(Downstream::default());
    let io = hub.processor_io(down.clone());
    let on_cpu = |cpu: u8, f: Box<dyn FnOnce() + Send>| {
        std::thread::spawn(move || {
            crate::cpu::smp::set_current_cpu(cpu);
            f();
        })
        .join()
        .unwrap()
    };
    // A good read records nothing.
    let i = io.clone();
    on_cpu(0, Box::new(move || assert_eq!(i.read64(0x0100_0000).data, 0x0100_0000)));
    assert_eq!(hub.read64(regs::PI_ERR_STATUS0_A::OFFSET).data, 0);
    // The PROM's probe.
    let i = io.clone();
    on_cpu(0, Box::new(move || {
        assert_eq!(i.read64(IO_BUS_ERROR_PROBE).status, BUS_ERR);
        assert_eq!(i.read64(NODE(9) + IO_BUS_ERROR_PROBE).status, BUS_ERR);
    }));
    let record = hub.read64(regs::PI_ERR_STATUS0_A::OFFSET).data;
    assert_eq!(record >> 62, 0b11, "valid, and overrun from the second");
    assert_eq!((record >> 25) & ((1 << 37) - 1), IO_BUS_ERROR_PROBE >> 3);
    // A narrower read of that address goes downstream instead.
    let i = io.clone();
    on_cpu(0, Box::new(move || assert_eq!(i.read32(IO_BUS_ERROR_PROBE).status, BUS_OK)));
    // A read that fails downstream records too, in the slice that issued it.
    down.fail.store(true, Ordering::SeqCst);
    let i = io.clone();
    on_cpu(1, Box::new(move || assert_eq!(i.read32(0x0300_0004).status, BUS_ERR)));
    assert_eq!(hub.read64(regs::PI_ERR_STATUS0_B::OFFSET).data, (1 << 63) | ((0x0300_0004u64 >> 3) << 25));
    // Writes are not reads: a failing store records nothing.
    hub.read64(regs::PI_ERR_STATUS0_B_RCLR::OFFSET);
    let i = io.clone();
    on_cpu(1, Box::new(move || assert_eq!(i.write64(0x10, 0), BUS_ERR)));
    assert_eq!(hub.read64(regs::PI_ERR_STATUS0_B::OFFSET).data, 0);
    // A full-width zero store to the status-1 alias ends the record (inferred).
    hub.write64(regs::PI_ERR_STATUS1_A_RCLR::OFFSET, 0);
    assert_eq!(hub.read64(regs::PI_ERR_STATUS0_A::OFFSET).data, 0);
}

/// A read the device answers busy is retried, not failed.
#[test]
fn a_busy_processor_read_records_no_error() {
    let hub = hub(1, 3);
    let down = Arc::new(Downstream::default());
    down.busy.store(true, Ordering::SeqCst);
    let io = hub.processor_io(down);
    let i = io.clone();
    std::thread::spawn(move || {
        crate::cpu::smp::set_current_cpu(0);
        assert_eq!(i.read32(0x0800_0098).status, BUS_BUSY);
        assert_eq!(i.read64(0x0800_0098).status, BUS_BUSY);
    })
    .join()
    .unwrap();
    assert_eq!(hub.read64(regs::PI_ERR_STATUS0_A::OFFSET).data, 0);
}

/// A debugger's access, a device's DMA and a processor of another node record nothing.
#[test]
fn a_failed_read_from_a_thread_that_is_not_one_of_the_nodes_processors_records_nothing() {
    let hub = hub(2, 3);
    let down = Arc::new(Downstream::default());
    down.fail.store(true, Ordering::SeqCst);
    let io = hub.processor_io(down);
    // This test thread never calls set_current_cpu.
    assert_eq!(io.read64(0x0300_0000).status, BUS_ERR);
    assert_eq!(io.read64(IO_BUS_ERROR_PROBE).status, BUS_ERR);
    assert_eq!(hub.pi.reader_slice(), None);
    assert_eq!(hub.read64(regs::PI_ERR_STATUS0_A::OFFSET).data, 0);
    assert_eq!(hub.read64(regs::PI_ERR_STATUS0_B::OFFSET).data, 0);
    // Nor does a processor numbered outside this node's.
    let (h, i) = (hub.clone(), io.clone());
    std::thread::spawn(move || {
        crate::cpu::smp::set_current_cpu(5);
        assert_eq!(h.pi.reader_slice(), None);
        assert_eq!(i.read64(0x0300_0000).status, BUS_ERR);
    })
    .join()
    .unwrap();
    assert_eq!(hub.read64(regs::PI_ERR_STATUS0_A::OFFSET).data, 0);
    assert_eq!(hub.read64(regs::PI_ERR_STATUS0_B::OFFSET).data, 0);
}

// ---- the Bridge's side ----

#[test]
fn the_interrupt_target_sets_and_clears_pending_bits_by_vector() {
    let hub = hub(1, 3);
    let target = hub.interrupt_target();
    target.set_line(70, true);
    target.set_line(3, true);
    assert_eq!((hub.pi.pending(0), hub.pi.pending(1)), (1 << 3, 1 << 6));
    target.set_line(70, false);
    assert_eq!(hub.pi.pending(1), 0);
    // Vectors are seven bits.
    target.set_line(128, true);
    assert_eq!((hub.pi.pending(0), hub.pi.pending(1)), (1 << 3, 0));
}

#[test]
fn the_hub_is_a_device_whose_counter_is_its_clock() {
    let now = Arc::new(AtomicU64::new(0));
    let clock = now.clone();
    let hub = Hub::new(HubConfig::default(), Arc::new(move || clock.load(Ordering::SeqCst)), None).unwrap();
    assert!(!hub.is_running());
    hub.start();
    assert!(hub.is_running());
    now.store(800 * 1234, Ordering::SeqCst);
    assert_eq!(hub.get_clock(), 1234);
    hub.step(1);
    hub.stop();
    assert!(!hub.is_running());
}

// ---- ownership ----

/// A node that is built, wired and dropped must free its Hub and the decode table of its memory.
#[test]
fn a_wired_hub_with_a_peer_and_attachments_is_freed_when_dropped() {
    let (a, b) = pair();
    number(&b, 1);
    let phys = a.decode.physical().clone();
    // What a machine does: the I/O space as a node space, handles given to widgets.
    a.decode.set_space(UncachedSpace::Io, a.io_space());
    // The manager's callbacks reach the processor and its bus, which reach the Hub.
    let timers = Arc::new(crate::hptimer::TimerManager::new());
    a.pi.set_timer_manager(timers.clone());
    assert_eq!(Arc::strong_count(&timers), 1, "the interface holds the timer manager strongly");
    a.attach_i2c(Arc::new(Bytes::default()));
    a.io.connect_widget(9, Arc::new(Bytes::default()));
    a.io.connect_widget(8, a.widget_port());
    let held = (
        a.inbound(),
        a.widget_port(),
        a.interrupt_target(),
        a.cpu_view(crate::sn::UncachedSpace::HubSpecial),
        a.processor_io(Arc::new(Downstream::default())),
    );
    let (weak_a, weak_b, weak_phys) = (Arc::downgrade(&a), Arc::downgrade(&b), Arc::downgrade(&phys));
    phys.release_devices();
    b.decode.physical().release_devices();
    drop((held, phys, a, b));
    assert!(weak_a.upgrade().is_none(), "the Hub is still referenced");
    assert!(weak_b.upgrade().is_none(), "the peer Hub is still referenced");
    assert!(weak_phys.upgrade().is_none(), "the node's decode table is still referenced");
}

/// The register window acts on what a 32- or 64-bit store changed.
#[test]
fn only_a_wide_register_store_acts_on_what_it_changed() {
    let hub = hub(2, 3);
    let (tx, rx) = std::sync::mpsc::channel();
    hub.set_event_sender(tx);
    let reset = regs::NI_PORT_RESET::OFFSET;
    assert_eq!((hub.write8(reset + 7, 0x81), hub.write16(reset + 6, 0x81)), (BUS_ERR, BUS_ERR));
    assert!(rx.try_recv().is_err());
    assert_eq!(hub.write32(reset + 4, 0x81), BUS_OK);
    assert!(matches!(rx.try_recv(), Ok(MachineEvent::HardReset)));
    assert_eq!(hub.write64(reset, 0x81), BUS_OK);
    assert!(matches!(rx.try_recv(), Ok(MachineEvent::HardReset)));
    assert_eq!(hub.io_space().write64(IO_REGISTERS_LOCAL + reset, 0x81), BUS_OK);
    assert!(matches!(rx.try_recv(), Ok(MachineEvent::HardReset)));
    assert!(rx.try_recv().is_err());
}

/// A vector request from node A's processor renumbers node B.
#[test]
fn a_peers_vector_request_renumbers_this_node_with_its_processor_parked() {
    use crate::sn::tests::ParkProbe;
    let probe = ParkProbe::new();
    let (a, b) = pair();
    a.decode.set_world(probe.world());
    b.decode.set_world(probe.world());
    let flushes = probe.flushes();
    assert!(probe.parks_the_other(|| {
        a.write64(regs::NI_VECTOR::OFFSET, 0);
        a.write64(regs::NI_VECTOR_DATA::OFFSET, 1 << 8);
        a.write64(regs::NI_VECTOR_PARMS::OFFSET, ni::VECTOR_WRITE);
    }));
    assert_eq!(b.decode.number(), 1);
    // Both tables moved: A shows B's memory at one, and B its own there.
    assert_eq!(a.decode.physical().read8(NODE(1) + 8).data, 0x88);
    assert_eq!(b.decode.physical().read8(NODE(1) + 8).data, 0x88);
    if cfg!(feature = "jitv2") {
        assert_eq!(probe.flushes() - flushes, 2, "one flush for each processor");
    }
}
