// Cases follow tests/qtest/sn0-hub-test.c, sn0-hub-inbound-test.c,
// sn0-pi-error-test.c and sn0-widget-decode-test.c in
// jamesbraid/qemu-sgi-origami: 32-bit-access-selects-the-big-endian-half,
// 32-bit-vector-clear-read-leaves-the-status, the I2C overlay
// cases, stock-prom-hubs-start-unnumbered,
// qualified-window-follows-the-programmed-number, unclaimed-node-number-fails,
// qualified-memory-follows-the-programmed-number, the calias-* cases,
// hub-special-window-stays-reader-local, the bte-routing and
// bte-physical-node-ownership cases, the inbound XIO cases, both-windows,
// prom-probe-latches-issuing-slice, narrow-access-matches-bridge-alias and
// status1-clear-alias-retires-record.

use super::*;
use parking_lot::Mutex;
use crate::traits::MachineEvent;
use std::sync::atomic::AtomicBool;

/// A node's memory: big-endian bytes, failing past its end.
struct Ram(Mutex<Vec<u8>>);
impl Ram {
    fn new(size: usize, seed: u8) -> Arc<Self> {
        Arc::new(Self(Mutex::new((0..size).map(|i| seed ^ (i as u8)).collect())))
    }
}
macro_rules! ram_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            let m = self.0.lock();
            const N: usize = std::mem::size_of::<$t>();
            match m.get(a as usize..a as usize + N) {
                Some(b) => $out::ok(<$t>::from_be_bytes(b.try_into().unwrap())),
                None => $out::err(),
            }
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            let mut m = self.0.lock();
            const N: usize = std::mem::size_of::<$t>();
            match m.get_mut(a as usize..a as usize + N) {
                Some(b) => {
                    b.copy_from_slice(&v.to_be_bytes());
                    BUS_OK
                }
                None => BUS_ERR,
            }
        }
    };
}
impl BusDevice for Ram {
    ram_access!(read8, write8, u8, BusRead8);
    ram_access!(read16, write16, u16, BusRead16);
    ram_access!(read32, write32, u32, BusRead32);
    ram_access!(read64, write64, u64, BusRead64);
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

fn hub(cpus: u8, ni_revision: u8) -> Arc<Hub> {
    let cfg = HubConfig { cpus, ni_revision, ..Default::default() };
    let hub = Hub::new(cfg, Arc::new(|| 0));
    hub.set_space(Space::Memory, Ram::new(RAM_SIZE, 0));
    hub
}
/// Two Hubs on a direct cable, each with a node memory that tells it apart.
fn pair() -> (Arc<Hub>, Arc<Hub>) {
    let a = hub(2, 3);
    let b = Hub::new(HubConfig { cpus: 2, cpu_base: 2, ..Default::default() }, Arc::new(|| 0));
    b.set_space(Space::Memory, Ram::new(RAM_SIZE, 0x80));
    Hub::cable(&a, &b);
    (a, b)
}
fn r64(d: &dyn BusDevice, a: u64) -> u64 {
    d.read64(a).data
}
const NODE: fn(u64) -> u64 = |n| n << NODE_SHIFT;
fn number(hub: &Hub, n: u64) {
    assert_eq!(hub.write64(ni::reg::STATUS, n << 8), BUS_OK);
}

#[test]
fn a_32_bit_access_selects_the_big_endian_half() {
    let hub = hub(1, 3);
    let status = r64(&*hub, ni::reg::STATUS);
    // The reset value is all in the low word, so a swap would show.
    assert_ne!(status, 0);
    assert_eq!(u64::from(hub.read32(ni::reg::STATUS).data), status >> 32);
    assert_eq!(u64::from(hub.read32(ni::reg::STATUS + 4).data), status & 0xffff_ffff);
    // A register with reset bits in both halves separates them sharply.
    assert_eq!(hub.read32(md::reg::CONFIG).data, 0x001e_7fbc);
    assert_eq!(hub.read32(md::reg::CONFIG + 4).data, 0x3fff_ffff);
    // A 32-bit write replaces its own half and leaves the other alone.
    hub.write64(ni::reg::SCRATCH_0, 0x1111_2222_3333_4444);
    hub.write32(ni::reg::SCRATCH_0, 0xaaaa_bbbb);
    assert_eq!(r64(&*hub, ni::reg::SCRATCH_0), 0xaaaa_bbbb_3333_4444);
    hub.write32(ni::reg::SCRATCH_0 + 4, 0xcccc_dddd);
    assert_eq!(r64(&*hub, ni::reg::SCRATCH_0), 0xaaaa_bbbb_cccc_dddd);
    // A half write still passes through the field mask (bits 31:3).
    hub.write32(pi::reg::ERR_STACK_ADDR_A + 4, 0xdead_beef);
    assert_eq!(r64(&*hub, pi::reg::ERR_STACK_ADDR_A), 0xdead_bee8);
    hub.write32(pi::reg::ERR_STACK_ADDR_A, 0xffff_ffff);
    assert_eq!(r64(&*hub, pi::reg::ERR_STACK_ADDR_A), 0xdead_bee8);
    // Only whole words, and only inside the window.
    assert_eq!(hub.read32(2).status, BUS_ERR);
    assert_eq!(hub.write32(WINDOW_SIZE, 0), BUS_ERR);
    assert_eq!(hub.read64(4).status, BUS_ERR);
    assert_eq!(hub.read64(WINDOW_SIZE).status, BUS_ERR);
    assert_eq!(hub.read8(0x18).status, BUS_ERR);
}

/// The vector clear alias reads zero and clears the status, but a 32-bit
/// read must leave the status for the other half.
#[test]
fn a_32_bit_read_of_a_read_to_clear_register_does_not_clear_it() {
    let (a, b) = pair();
    b.write64(ni::reg::SCRATCH_0, 7);
    a.write64(ni::reg::VECTOR, 0);
    a.write64(ni::reg::VECTOR_PARMS, ni::reg::SCRATCH_0 - ni::reg::STATUS);
    let status = r64(&*a, ni::reg::VECTOR_STATUS);
    assert_ne!(status, 0);
    assert_eq!(a.read32(ni::reg::VECTOR_CLEAR).data, 0);
    assert_eq!(r64(&*a, ni::reg::VECTOR_STATUS), status);
    assert_eq!(r64(&*a, ni::reg::VECTOR_CLEAR), 0);
    assert_eq!(r64(&*a, ni::reg::VECTOR_STATUS), 0);
    // And a half write of the write-one-to-clear port error does not merge
    // with the old value, which would clear more than was asked.
    a.ni().latch_port_error(0x6);
    a.write32(ni::reg::PORT_ERROR + 4, 0x2);
    assert_eq!(r64(&*a, ni::reg::PORT_ERROR), 0x4);
    // The interrupt-modify register stores nothing: the other half is zero.
    a.write32(pi::reg::INT_PEND_MOD + 4, (pi::MODIFY_SET | 9) as u32);
    assert_eq!(r64(&*a, pi::reg::INT_PEND0), 1 << 9);
}

#[test]
fn every_section_answers_through_the_window_and_the_rest_reads_zero() {
    let hub = hub(2, 5);
    assert_eq!(r64(&*hub, pi::reg::CALIAS_SIZE), 15);
    assert_eq!(r64(&*hub, md::reg::CONFIG), 0x001e_7fbc_3fff_ffff);
    assert_eq!(r64(&*hub, io::reg::WIDGET_ID), 0x1c10_106d);
    assert_eq!(r64(&*hub, ni::reg::STATUS) & 0xff, 5 << 4);
    assert_eq!(r64(&*hub, pi::reg::RT_COUNT) < 1_000, true);
    for off in [0x10_0000, 0x30_0000, 0x50_0000, 0x70_0000, 0x7f_fff8] {
        assert_eq!(hub.write64(off, !0), BUS_OK);
        assert_eq!(r64(&*hub, off), 0, "{off:#x}");
    }
}

#[test]
fn the_i2c_controller_takes_narrow_accesses_over_the_md_block() {
    let hub = hub(1, 3);
    // With nothing fitted, narrow accesses fail like any other.
    assert_eq!(hub.read8(md::reg::I2C_BASE).status, BUS_ERR);
    let i2c = Arc::new(Bytes::default());
    hub.attach_i2c(i2c.clone());
    // The PCF8584's S0 and S1, eight bytes apart.
    assert_eq!(hub.write8(md::reg::I2C_BASE, 0x5a), BUS_OK);
    assert_eq!(hub.write8(md::reg::I2C_BASE + 8, 0x3c), BUS_OK);
    let (s0, s1) = {
        let regs = i2c.0.lock();
        (regs[0], regs[8])
    };
    assert_eq!((s0, s1), (0x5a, 0x3c));
    assert_eq!(hub.read8(md::reg::I2C_BASE).data, 0x5a);
    // The registers between them are still the MD block's.
    assert_eq!(r64(&*hub, md::reg::SLOTID_USTAT), 0x95);
    // A doubleword access to the overlay is one register access too: the
    // PROM polls the controller's status with 8-byte loads.
    assert_eq!(hub.write64(md::reg::I2C_BASE, 1), BUS_OK);
    assert_eq!(i2c.0.lock()[0], 1);
    assert_eq!(hub.read64(md::reg::I2C_BASE + 8).data, 0x3c);
    i2c.0.lock()[0] = 0x81;
    assert_eq!(hub.read64(md::reg::I2C_BASE).data, 0x81, "an 8-byte read reaches the PCF8584 status");
}


#[test]
fn a_restart_reaches_the_machines_receiver_and_a_soft_reset_the_processor() {
    let hub = hub(2, 3);
    let (tx, rx) = std::sync::mpsc::channel();
    hub.set_event_sender(tx);
    let words = [AtomicU64::new(0), AtomicU64::new(0)];
    hub.set_cpu_interrupts(0, &words[0]);
    hub.set_cpu_interrupts(1, &words[1]);
    // Enable and NMI are stored and reach nothing.
    hub.write64(pi::reg::CPU_ENABLE_A, 1);
    hub.write64(pi::reg::NMI_B, 1);
    assert!(rx.try_recv().is_err());
    assert_eq!(words[0].load(Ordering::SeqCst) | words[1].load(Ordering::SeqCst), 0);
    // The soft reset reaches the enabled processor only.
    hub.write64(pi::reg::SOFTRESET, 1);
    assert_eq!(words[0].load(Ordering::SeqCst), crate::cpu::mips_exec::SOFT_RESET_BIT);
    assert_eq!(words[1].load(Ordering::SeqCst), 0);
    assert!(rx.try_recv().is_err(), "a soft reset restarts only the processor");
    // A local port reset restarts the machine.
    hub.write64(ni::reg::PORT_RESET, 0x81);
    assert!(matches!(rx.try_recv(), Ok(MachineEvent::HardReset)));
    assert!(rx.try_recv().is_err());
    // A link reset alone does not.
    hub.write64(ni::reg::PORT_RESET, 0x2);
    assert!(rx.try_recv().is_err());
    hub.clear_cpu_interrupts();
}

#[test]
fn a_restart_with_no_machine_listening_is_only_logged() {
    let hub = hub(1, 3);
    assert_eq!(hub.write64(ni::reg::PORT_RESET, 0x81), BUS_OK);
}

/// A node number or cached-alias write changes what the machine's node-zero
/// routing sends off the node, with no call from the machine in between, and
/// the same write through the processor's I/O space does the same.
#[test]
fn a_node_number_write_reaches_the_node_zero_routing() {
    use crate::ip27::NodeMemory;
    let mem = NodeMemory::new([128, 0, 0, 0]).unwrap();
    let phys = mem.physical();
    let hub = hub(1, 3);
    let (a, b) = (0x1000, (64 << 20) + 0x1000);
    assert_eq!(phys.write64(b, 0x77), BUS_OK);
    mem.route_nodes(hub.claim(Space::Memory));
    hub.set_node_zero(mem.node_zero(hub.routing().clone(), &Default::default()));
    // Unnumbered: node zero's memory is the node's own, wherever it is.
    assert_eq!(phys.read64(b).data, 0x77);
    // Numbered 6: beyond the 64 MiB sentinel alias the reference leaves the
    // node, for the Hub that holds zero, and there is none.
    number(&hub, 6);
    assert_eq!(hub.routing().number(), 6);
    assert_ne!(phys.read64(b).status, BUS_OK);
    assert_eq!(phys.write64(a, 0x55), BUS_OK, "within the alias it stays");
    // A smaller alias moves the same boundary down; through the I/O space too.
    let space = hub.io_space();
    assert_eq!(space.write64(IO_REGISTERS_LOCAL + pi::reg::CALIAS_SIZE, 1), BUS_OK);
    assert_eq!(hub.calias_bytes(), 4 << 10);
    assert_ne!(phys.read64(a).status, BUS_OK);
    // Back to zero, by reset: the memory is the node's again.
    hub.power_on();
    assert_eq!(hub.routing().number(), 0);
    assert_eq!(phys.read64(b).data, 0x77);
    // Renumbered through the I/O space.
    assert_eq!(space.write64(IO_REGISTERS_LOCAL + ni::reg::STATUS, 6 << 8), BUS_OK);
    assert_ne!(phys.read64(b).status, BUS_OK);
}

/// A neighbour that renumbers this node over the cable has its routing
/// re-read the number too.
#[test]
fn a_neighbours_renumbering_reaches_this_nodes_routing() {
    use crate::ip27::NodeMemory;
    let (a, b) = pair();
    let mem = NodeMemory::new([128, 0, 0, 0]).unwrap();
    let phys = mem.physical();
    let beyond = (64 << 20) + 0x1000;
    assert_eq!(phys.write64(beyond, 0x77), BUS_OK);
    mem.route_nodes(b.claim(Space::Memory));
    b.set_node_zero(mem.node_zero(b.routing().clone(), &Default::default()));
    assert_eq!(phys.read64(beyond).data, 0x77);
    // a's vector write of b's node number: b holds 3 now and no longer
    // answers for node zero, so its memory beyond the alias leaves it.
    a.write64(ni::reg::VECTOR, 0);
    a.write64(ni::reg::VECTOR_DATA, 3 << 8);
    a.write64(ni::reg::VECTOR_PARMS, ni::VECTOR_WRITE);
    assert_eq!(b.routing().number(), 3);
    assert_ne!(phys.read64(beyond).status, BUS_OK);
}

#[test]
fn reset_restores_the_registers_and_keeps_what_firmware_parks() {
    let hub = hub(2, 3);
    hub.write64(pi::reg::ERR_STACK_ADDR_B, 0xabcd_0000);
    hub.write64(pi::reg::CALIAS_SIZE, 1);
    hub.write64(md::reg::CONFIG, 5);
    hub.write64(io::reg::SCRATCH_0, 6);
    number(&hub, 3);
    hub.power_on();
    assert_eq!(r64(&*hub, pi::reg::ERR_STACK_ADDR_B), 0xabcd_0000);
    assert_eq!(r64(&*hub, pi::reg::CALIAS_SIZE), 15);
    assert_eq!(hub.routing().calias(), 64 << 20);
    assert_eq!(r64(&*hub, md::reg::CONFIG), 0x001e_7fbc_3fff_ffff);
    assert_eq!(r64(&*hub, io::reg::SCRATCH_0), 0);
    assert_eq!(hub.ni().number(), 0);
    let mem = hub.claim(Space::Memory);
    assert_eq!(mem.read8(0x10).data, 0x10, "node zero is this Hub's again");
    assert_eq!(mem.read8(NODE(3) + 0x10).status, BUS_ERR, "and node 3 is nobody's");
}

// ---- the claim map ----

#[test]
fn an_unprogrammed_hub_claims_node_zero_and_a_programmed_one_moves() {
    let hub = hub(1, 3);
    let mem = hub.claim(Space::Memory);
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
    // A space nobody wired fails without taking the others down.
    assert_eq!(hub.claim(Space::Special).read8(NODE(5)).status, BUS_ERR);
}

#[test]
fn two_cabled_hubs_each_claim_the_others_programmed_number() {
    let (a, b) = pair();
    let (ma, mb) = (a.claim(Space::Memory), b.claim(Space::Memory));
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
fn the_cached_alias_keeps_node_zero_local_until_the_numbering_moves_it() {
    let hub = hub(1, 3);
    let cpu = hub.cpu_memory();
    hub.write64(pi::reg::CALIAS_SIZE, 1);
    number(&hub, 6);
    // Within the 4 KiB window a node-zero address stays on this node even
    // though node 0 is held by nobody now.
    assert_eq!(cpu.read8(0x10).data, 0x10);
    assert_eq!(cpu.read8(0xfff).data, 0xff);
    // Beyond it the access goes to the claim map, which fails.
    assert_eq!(cpu.read8(0x1000).status, BUS_ERR);
    assert_eq!(cpu.read8(NODE(6) + 0x1000).data, 0x00);
    // The window tracks the register: 8 KiB, then none.
    hub.write64(pi::reg::CALIAS_SIZE, 2);
    assert_eq!(cpu.read8(0x1000).data, 0x00);
    assert_eq!(cpu.read8(0x2000).status, BUS_ERR);
    hub.write64(pi::reg::CALIAS_SIZE, 0);
    assert_eq!(cpu.read8(0x10).status, BUS_ERR);
    // Reset sets it back to its 64 MiB sentinel, and node zero is this
    // node's own again before anything is numbered.
    hub.power_on();
    assert_eq!(cpu.read8(RAM_SIZE as u64 - 1).data, 0xff);
    // The plain claim never honours the alias.
    hub.write64(pi::reg::CALIAS_SIZE, 0);
    assert_eq!(hub.claim(Space::Memory).read8(0x10).data, 0x10);
}

#[test]
fn the_hub_special_window_always_stays_reader_local() {
    let (a, b) = pair();
    a.set_space(Space::HubSpecial, Ram::new(64, 0x11));
    b.set_space(Space::HubSpecial, Ram::new(64, 0x22));
    number(&a, 3);
    number(&b, 4);
    let (sa, sb) = (a.cpu_special(), b.cpu_special());
    // Each reader's node-zero address names its own node, not the number's holder.
    assert_eq!((sa.read8(1).data, sb.read8(1).data), (0x10, 0x23));
    // A node-qualified address is claimed by number.
    assert_eq!(sa.read8(NODE(4) + 1).data, 0x23);
}

#[test]
fn the_node_view_of_a_peer_cannot_reach_a_number_nobody_holds() {
    let (a, b) = pair();
    number(&a, 1);
    number(&b, 2);
    for number in [0, 3, 200] {
        assert_eq!(a.claim(Space::Memory).read8(NODE(number) + 8).status, BUS_ERR);
        assert_eq!(b.claim(Space::Memory).read8(NODE(number) + 8).status, BUS_ERR);
    }
    assert_eq!(a.claim(Space::Memory).read8(NODE(2) + 8).data, 0x88);
    assert_eq!(b.claim(Space::Memory).read8(NODE(1) + 8).data, 0x08);
}

// ---- the XIO port ----

#[test]
fn inbound_memory_reaches_the_nodes_memory_by_node_number() {
    let (a, b) = pair();
    number(&a, 1);
    number(&b, 2);
    let xio = a.inbound();
    // A widget's DMA of every size lands in node memory. The address carries
    // the node number the Hub was given, as a processor's does: node zero is
    // held by nobody now.
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
    assert_eq!(b.claim(Space::Memory).read8(NODE(2) + 9).data, 0x42);
    // Memory is below 2^40; the rest of the address is the I/O view.
    assert_eq!(xio.read8(1 << 40).status, BUS_ERR);
    assert_eq!(xio.read8(NODE(7)).status, BUS_ERR);
    assert_eq!(xio.read8(1 << 48).status, BUS_ERR);
}

#[test]
fn an_interrupt_modify_write_lands_on_the_hub_the_packet_names() {
    let (a, b) = pair();
    number(&a, 1);
    number(&b, 2);
    let xio = a.inbound();
    let packet = |node: u64| (1 << 47) | (node << 38) | 0x0180_0090;
    // Local: set interrupt 70, which is bit 6 of the second bank.
    assert_eq!(xio.write64(packet(1), pi::MODIFY_SET | 70), BUS_OK);
    assert_eq!(a.pi().pending(1), 1 << 6);
    assert_eq!(b.pi().pending(1), 0);
    // Across the cable, by the peer's programmed number.
    assert_eq!(xio.write64(packet(2), pi::MODIFY_SET | 9), BUS_OK);
    assert_eq!(b.pi().pending(0), 1 << 9);
    assert_eq!(xio.write64(packet(2), 9), BUS_OK);
    assert_eq!(b.pi().pending(0), 0);
    // A number nobody holds, a narrow write and any other offset fail.
    assert_eq!(xio.write64(packet(3), pi::MODIFY_SET | 1), BUS_ERR);
    assert_eq!(xio.write32(packet(1), 1), BUS_ERR);
    assert_eq!(xio.write64(packet(1) + 8, 1), BUS_ERR);
    assert_eq!(a.pi().pending(0), 0);
}

#[test]
fn a_peers_identity_probe_reads_the_network_status_and_nothing_else_is_answered() {
    let (a, _) = pair();
    let xio = a.inbound();
    let probe = (1u64 << 47) | 0x0160_0000;
    assert_eq!(xio.read64(probe).data, a.ni().read64(ni::reg::STATUS).data);
    assert_eq!(xio.read32(probe).status, BUS_ERR, "a doubleword read only");
    assert_eq!(xio.read64(probe + 8).status, BUS_ERR);
    assert_eq!(xio.write64(probe, 0).status_ok(), false);
}

trait StatusOk {
    fn status_ok(self) -> bool;
}
impl StatusOk for u32 {
    fn status_ok(self) -> bool {
        self == BUS_OK
    }
}

#[test]
fn a_peers_programmed_io_to_the_hubs_widget_reads_only_its_identity() {
    let hub = hub(1, 3);
    let face = hub.widget_face();
    assert_eq!(face.read32(4).data, 0x1c10_106d);
    for off in [0, 8, 0x20] {
        assert_eq!(face.read32(off).status, BUS_ERR);
    }
    assert_eq!(face.read64(4).status, BUS_ERR);
    assert_eq!(face.write32(4, 0), BUS_ERR);
}

#[test]
fn the_io_space_holds_the_registers_at_both_offsets_and_the_widgets_behind_them() {
    let hub = hub(1, 3);
    let io = hub.io_space();
    // Both the reader's own alias and the node-explicit window.
    let regs = [IO_REGISTERS_LOCAL, IO_REGISTERS_NAMED];
    for base in regs {
        assert_eq!(io.read64(base + md::reg::CONFIG).data, 0x001e_7fbc_3fff_ffff);
    }
    io.write64(IO_REGISTERS_LOCAL + io::reg::SCRATCH_1, 0x1234);
    assert_eq!(io.read64(IO_REGISTERS_NAMED + io::reg::SCRATCH_1).data, 0x1234);
    assert_eq!(io.read32(IO_REGISTERS_NAMED + io::reg::SCRATCH_1 + 4).data, 0x1234);
    // Past the second window the space is widget 2's.
    assert_eq!(io.read64(0x0200_0000).status, BUS_ERR);
    // A widget at its small window and through a big window.
    let widget = Arc::new(Bytes::default());
    hub.io().connect_widget(9, widget.clone());
    io.write8(9 << 24 | 5, 0x77);
    assert_eq!(widget.0.lock()[5], 0x77);
    hub.write64(io::reg::BWIN_TABLE + 8, 9 << 8);
    io.write8(IO_BIG_WINDOWS + io::BIG_WINDOW_SIZE + 6, 0x66);
    assert_eq!(widget.0.lock()[6], 0x66);
    assert_eq!(io.read8(IO_BIG_WINDOWS + io::BIG_WINDOW_SIZE + 6).data, 0x66);
    assert_eq!(io.read8(IO_BIG_WINDOWS + 7 * io::BIG_WINDOW_SIZE).status, BUS_ERR);
}

#[test]
fn a_processors_own_register_alias_stays_on_its_node_whatever_the_number() {
    let (a, b) = pair();
    for hub in [&a, &b] {
        hub.set_space(Space::Io, hub.io_space());
    }
    number(&b, 1);
    let at = IO_REGISTERS_LOCAL + io::reg::SCRATCH_1;
    // The claim by node number sends node zero's alias to the node holding
    // zero; the processor's view keeps it on its own.
    assert_eq!(b.claim(Space::Io).write64(at, 0x11), BUS_OK);
    assert_eq!(r64(&*a, io::reg::SCRATCH_1), 0x11);
    assert_eq!(b.cpu_io().write64(at, 0x22), BUS_OK);
    assert_eq!(r64(&*b, io::reg::SCRATCH_1), 0x22);
    assert_eq!(r64(&*a, io::reg::SCRATCH_1), 0x11);
    // The node-explicit window still goes by node number.
    assert_eq!(b.cpu_io().read64(IO_REGISTERS_NAMED + io::reg::SCRATCH_1).data, 0x11);
    assert_eq!(b.cpu_io().read64(NODE(1) + IO_REGISTERS_NAMED + io::reg::SCRATCH_1).data, 0x22);
    // The alias is the one with node field zero: another node's number at
    // the same offset goes to that node.
    assert_eq!(b.cpu_io().read64(NODE(0) + at).data, 0x22);
    assert_eq!(a.cpu_io().write64(NODE(1) + at, 0x33), BUS_OK);
    assert_eq!(r64(&*b, io::reg::SCRATCH_1), 0x33);
    assert_eq!(r64(&*a, io::reg::SCRATCH_1), 0x11);
}

// ---- the engines through the fabric ----

#[test]
fn an_engine_addresses_memory_by_node_number_and_interrupts_the_node_it_names() {
    let (a, b) = pair();
    number(&a, 1);
    number(&b, 2);
    let (engine, bte) = (io::reg::BTE_BASE, io::bte::LENGTH);
    // Copy two lines from this node's memory into the peer's.
    a.write64(io::reg::BTE_BASE + io::bte::SOURCE, NODE(1) + 0x1000);
    a.write64(io::reg::BTE_BASE + io::bte::DEST, NODE(2) + 0x2000);
    a.write64(io::reg::BTE_BASE + io::bte::INTERRUPT, (17 << 16) | 2);
    a.write64(engine + bte, (1 << 20) | 2);
    a.write64(engine + io::bte::CONTROL, 0);
    for i in 0..256 {
        assert_eq!(b.claim(Space::Memory).read8(NODE(2) + 0x2000 + i).data, a.claim(Space::Memory).read8(NODE(1) + 0x1000 + i).data);
    }
    // Busy and the count clear, and the completion interrupted node 2's Hub.
    assert_eq!(a.read64(engine + bte).data, 0);
    assert_eq!(b.pi().pending(0), 1 << 17);
    // An engine's addresses ignore the cached alias: node zero is whichever
    // Hub holds zero, which nobody does now.
    a.write64(pi::reg::CALIAS_SIZE, 15);
    a.write64(engine + io::bte::SOURCE, 0x1000);
    a.write64(engine + io::bte::DEST, NODE(1) + 0x3000);
    a.write64(engine + bte, (1 << 20) | 1);
    a.write64(engine + io::bte::CONTROL, 0);
    assert_eq!(a.read64(engine + bte).data, (1 << 20) | (1 << 16) | 1);
    // Completion aimed at a node nobody holds drops the interrupt softly.
    a.write64(engine + io::bte::LENGTH, 0);
    a.write64(io::reg::ERROR_CLEAR, 1 << 17);
    a.write64(engine + io::bte::SOURCE, NODE(1) + 0x1000);
    a.write64(engine + io::bte::INTERRUPT, 9);
    a.write64(engine + bte, (1 << 20) | 1);
    a.write64(engine + io::bte::CONTROL, 0);
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
    assert_eq!(hub.read64(pi::reg::ERR_STATUS0_A).data, 0);
    // The PROM's probe: the Hub fails 0x0240_0208 itself, whatever node
    // it names, and the first failure latches the address.
    let i = io.clone();
    on_cpu(0, Box::new(move || {
        assert_eq!(i.read64(IO_BUS_ERROR_PROBE).status, BUS_ERR);
        assert_eq!(i.read64(NODE(9) + IO_BUS_ERROR_PROBE).status, BUS_ERR);
    }));
    let record = hub.read64(pi::reg::ERR_STATUS0_A).data;
    assert_eq!(record >> 62, 0b11, "valid, and overrun from the second");
    assert_eq!((record >> 25) & ((1 << 37) - 1), IO_BUS_ERROR_PROBE >> 3);
    // A narrower read of that address goes downstream instead.
    let i = io.clone();
    on_cpu(0, Box::new(move || assert_eq!(i.read32(IO_BUS_ERROR_PROBE).status, BUS_OK)));
    // A read that fails downstream records too, in the slice that issued it.
    down.fail.store(true, Ordering::SeqCst);
    let i = io.clone();
    on_cpu(1, Box::new(move || assert_eq!(i.read32(0x0300_0004).status, BUS_ERR)));
    assert_eq!(hub.read64(pi::reg::ERR_STATUS0_B).data, (1 << 63) | ((0x0300_0004u64 >> 3) << 25));
    // Writes are not reads: a failing store records nothing.
    hub.read64(pi::reg::ERR_STATUS0_B_RCLR);
    let i = io.clone();
    on_cpu(1, Box::new(move || assert_eq!(i.write64(0x10, 0), BUS_ERR)));
    assert_eq!(hub.read64(pi::reg::ERR_STATUS0_B).data, 0);
    // A full-width zero store to the status-1 alias ends the record
    // (inferred).
    hub.write64(pi::reg::ERR_STATUS1_A_RCLR, 0);
    assert_eq!(hub.read64(pi::reg::ERR_STATUS0_A).data, 0);
}

/// A read the device answers busy is retried, not failed: it records nothing,
/// or a pipe that makes the processor wait leaves an error behind for the
/// kernel to find.
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
    assert_eq!(hub.read64(pi::reg::ERR_STATUS0_A).data, 0);
}

/// A debugger's access, a device's DMA and a processor of another node record
/// nothing. A thread that never set its CPU is a device or monitor thread.
#[test]
fn a_failed_read_from_a_thread_that_is_not_one_of_the_nodes_processors_records_nothing() {
    let hub = hub(2, 3);
    let down = Arc::new(Downstream::default());
    down.fail.store(true, Ordering::SeqCst);
    let io = hub.processor_io(down);
    // This test thread never calls set_current_cpu.
    assert_eq!(io.read64(0x0300_0000).status, BUS_ERR);
    assert_eq!(io.read64(IO_BUS_ERROR_PROBE).status, BUS_ERR);
    assert_eq!(hub.pi().reader_slice(), None);
    assert_eq!(hub.read64(pi::reg::ERR_STATUS0_A).data, 0);
    assert_eq!(hub.read64(pi::reg::ERR_STATUS0_B).data, 0);
    // Nor does a processor numbered outside this node's.
    let (h, i) = (hub.clone(), io.clone());
    std::thread::spawn(move || {
        crate::cpu::smp::set_current_cpu(5);
        assert_eq!(h.pi().reader_slice(), None);
        assert_eq!(i.read64(0x0300_0000).status, BUS_ERR);
    })
    .join()
    .unwrap();
    assert_eq!(hub.read64(pi::reg::ERR_STATUS0_A).data, 0);
    assert_eq!(hub.read64(pi::reg::ERR_STATUS0_B).data, 0);
}

// ---- the Bridge's side ----

#[test]
fn the_interrupt_target_sets_and_clears_pending_bits_by_vector() {
    let hub = hub(1, 3);
    let target = hub.interrupt_target();
    target.set_line(70, true);
    target.set_line(3, true);
    assert_eq!((hub.pi().pending(0), hub.pi().pending(1)), (1 << 3, 1 << 6));
    target.set_line(70, false);
    assert_eq!(hub.pi().pending(1), 0);
    // Vectors are seven bits.
    target.set_line(128, true);
    assert_eq!((hub.pi().pending(0), hub.pi().pending(1)), (1 << 3, 0));
}

#[test]
fn the_hub_is_a_device_whose_counter_is_its_clock() {
    let now = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let clock = now.clone();
    let hub = Hub::new(HubConfig::default(), Arc::new(move || clock.load(Ordering::SeqCst)));
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

/// A node that is built, wired and dropped must free its Hub and the decode
/// table of its memory. The Hub owns its spaces, so no space may hold the Hub:
/// the I/O space once did, and a machine's every Hub and decode table leaked
/// (the process has only 64 tables). The memory's own routing back into the
/// Hub is released the way a machine's drop releases it.
#[test]
fn a_wired_hub_with_a_peer_and_attachments_is_freed_when_dropped() {
    use crate::physical::{Physical, RamBank};
    let (a, b) = (Hub::new(HubConfig::default(), Arc::new(|| 0)), Hub::new(HubConfig::default(), Arc::new(|| 0)));
    Hub::cable(&a, &b);
    let phys = Physical::new(vec![RamBank::new(1)]);
    a.set_space(Space::Memory, phys.clone());
    // What a machine does: the node's own I/O space as a space of the node, the
    // Hub's claim view routed into the node's memory, the Hub's handles given
    // to widgets, and attachments on the window.
    a.set_space(Space::Io, a.io_space());
    phys.map(1 << 32, 2 << 32, a.claim(Space::Memory), "node-qualified memory");
    // The manager's callbacks reach the processor and its bus, which reach the
    // Hub: the interface must not own the manager.
    let timers = Arc::new(crate::hptimer::TimerManager::new());
    a.pi().set_timer_manager(timers.clone());
    assert_eq!(Arc::strong_count(&timers), 1, "the interface holds the timer manager strongly");
    a.attach_i2c(Arc::new(Bytes::default()));
    a.io().connect_widget(9, Arc::new(Bytes::default()));
    a.io().connect_widget(8, a.widget_face());
    let held = (a.inbound(), a.widget_face(), a.interrupt_target(), a.cpu_memory(), a.cpu_special(), a.processor_io(Arc::new(Downstream::default())));
    let (weak_a, weak_b, weak_phys) = (Arc::downgrade(&a), Arc::downgrade(&b), Arc::downgrade(&phys));
    phys.release_devices();
    drop((held, phys, a, b));
    assert!(weak_a.upgrade().is_none(), "the Hub is still referenced");
    assert!(weak_b.upgrade().is_none(), "the peer Hub is still referenced");
    assert!(weak_phys.upgrade().is_none(), "the node's decode table is still referenced");
}
