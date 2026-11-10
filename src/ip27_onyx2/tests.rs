use super::*;
use crate::config::{MachineProfile, NetMode};
use crate::cpu::parts;
use crate::dev::hub::Space;
use crate::dev::{hub_md, hub_ni};
use crate::traits::{BusDevice, BUS_ERR, BUS_OK};
use std::path::{Path, PathBuf};

struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("iris-onyx2-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A PROM stand-in: a recognisable first word and a configuration block in
/// the shape the flash initialiser patches.
fn prom(dir: &Path) -> String {
    let mut p = vec![0u8; 0x1000];
    p[..4].copy_from_slice(&0x3c08_a600u32.to_be_bytes());
    p[0x68..0x70].copy_from_slice(&0x6970_3237_636f_6e66u64.to_be_bytes());
    let path = dir.join("prom.bin");
    std::fs::write(&path, p).unwrap();
    path.to_string_lossy().into_owned()
}

fn config(dir: &Path, nodes: u8) -> MachineConfig {
    let mut cfg = MachineConfig::default();
    cfg.machine.profile = MachineProfile::Onyx2Ip27;
    cfg.set_cpu(&parts::R10000_180);
    cfg.origin2000.nodes = nodes;
    cfg.banks = [64, 0, 0, 0, 0, 0, 0, 0];
    cfg.prom = prom(dir);
    cfg.nvram = dir.join("nvram.bin").to_string_lossy().into_owned();
    cfg.headless = true;
    cfg.network.mode = NetMode::Off;
    let held: Vec<_> = (0..3).map(|_| std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap()).collect();
    cfg.monitor_port = Some(held[0].local_addr().unwrap().port());
    cfg.serial_port_a = Some(held[1].local_addr().unwrap().port());
    cfg.serial_port_b = Some(held[2].local_addr().unwrap().port());
    let disk = dir.join("disk.raw");
    std::fs::write(&disk, vec![0u8; 4 << 20]).unwrap();
    cfg.scsi.clear();
    cfg.scsi.insert(1, toml::from_str(&format!("path = {:?}\noverlay = true", disk.to_string_lossy())).unwrap());
    cfg
}

/// Building a machine puts more than a megabyte on the stack.
fn big_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().stack_size(64 << 20).spawn(body).unwrap().join().expect("thread panicked");
}

fn built(name: &str, nodes: u8) -> (Scratch, Onyx2Machine) {
    let dir = Scratch::new(name);
    let m = Onyx2Machine::new(&config(&dir.0, nodes)).unwrap();
    (dir, m)
}

#[test]
fn the_profile_is_chosen_by_its_snake_case_name() {
    assert_eq!(MachineProfile::from_name("onyx2_ip27"), Ok(MachineProfile::Onyx2Ip27));
    assert_eq!(MachineProfile::Onyx2Ip27.name(), "onyx2_ip27");
}


#[test]
fn two_nodes_share_one_crossbow_and_both_reach_the_baseio() {
    big_stack(|| {
        let (_dir, m) = built("two", 2);
        let (n1, n2) = (&m.node_at(0).hub, &m.node_at(1).hub);
        // Slot identity 3 in the first node slot, 2 in the second.
        assert_eq!(n1.read64(hub_md::reg::SLOTID_USTAT).data & 7, 3);
        assert_eq!(n2.read64(hub_md::reg::SLOTID_USTAT).data & 7, 2);
        // The flash carries no machine type: the slot code selects the table.
        assert_eq!(n1.cpu_special().read32(0x1000_0090).data, 0);

        for hub in [n1, n2] {
            let io = hub.claim(Space::Io);
            let id = |w: u64| io.read32((w << 24) + 4);
            // The one Crossbow, revision 2.
            assert_eq!(id(0).data, 0x2000_0001);
            // Both Hubs' ports, and the test widget in the graphics slot.
            assert_eq!(id(9).status, BUS_OK);
            assert_eq!(id(10).status, BUS_OK);
            assert_eq!(id(8).data, TEST_WIDGET_ID);
            // The test widget refuses anything else.
            assert_eq!(io.write32((8 << 24) + 0x10, 1), BUS_ERR);
            assert_eq!(io.read32((8 << 24) + 0x10).status, BUS_ERR);
            // Nothing is fitted at 11 to 14.
            for w in 11..15 {
                assert_eq!(id(w).status, BUS_ERR, "widget {w}");
            }
        }
        // The BaseIO-G answers at widget 15 from either Hub.
        let id = |hub: &Arc<Hub>| hub.claim(Space::Io).read32((15 << 24) + 4);
        assert_eq!(id(n2).status, BUS_OK);
        assert_eq!(id(n1).status, BUS_OK);
        assert_eq!(id(n1).data, id(n2).data);

        // A big-window probe of the peer's port, as the PROM does it, reads
        // the peer's node number once the Hubs are numbered.
        assert_eq!(n2.write64(hub_ni::reg::STATUS, 1 << 8), BUS_OK);
        let io = n1.cpu_io();
        let bwin = crate::dev::hub::IO_BIG_WINDOWS;
        assert_eq!(n1.write64(crate::dev::hub_io::reg::BWIN_TABLE, (9 << 8) | 0x1000), BUS_OK);
        let status = io.read64(bwin + 0x0160_0000);
        assert_eq!(status.status, BUS_OK);
        assert_eq!((status.data >> 8) & 0x1ff, 1);
    });
}

#[test]
fn a_dropped_machine_frees_its_hubs_memory_and_timers() {
    big_stack(|| {
        let (_dir, m) = built("drop", 2);
        let hubs: Vec<_> = (0..2).map(|i| Arc::downgrade(&m.node_at(i).hub)).collect();
        let memory: Vec<_> = (0..2).map(|i| Arc::downgrade(m.node_at(i).memory.physical())).collect();
        let timers: Vec<_> = (0..2).map(|i| Arc::downgrade(&m.node_at(i).timers)).collect();
        let cpus: Vec<_> = (0..2).map(|i| Arc::downgrade(m.cpu_at(i))).collect();
        let bridge = Arc::downgrade(m.bridge());
        drop(m);
        for (i, w) in hubs.iter().enumerate() {
            assert_eq!(w.strong_count(), 0, "hub {i}");
        }
        for (i, w) in memory.iter().enumerate() {
            assert_eq!(w.strong_count(), 0, "memory map {i}");
        }
        for (i, w) in timers.iter().enumerate() {
            assert_eq!(w.strong_count(), 0, "timers {i}");
        }
        for (i, w) in cpus.iter().enumerate() {
            assert_eq!(w.strong_count(), 0, "cpu {i}");
        }
        assert_eq!(bridge.strong_count(), 0, "the Bridge");
    });
}

/// A PCI function with nothing behind it but the DMA and interrupt line the
/// Bridge hands it.
struct Card {
    state: crate::pci::PciFunctionState,
}

impl Card {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: crate::pci::PciFunctionState::new(crate::pci::PciConfig::new(
                0x1234,
                0x5678,
                0x01_0000,
                1,
                [crate::pci::Bar::Unused; 6],
                crate::pci::PCI_INTA,
            )),
        })
    }
}

impl crate::pci::PciFunction for Card {
    fn state(&self) -> &crate::pci::PciFunctionState {
        &self.state
    }
    fn bar_read(&self, _bar: usize, _off: u32, _width: usize) -> u32 {
        0
    }
    fn bar_write(&self, _bar: usize, _off: u32, _val: u32, _width: usize) {}
}

/// The BaseIO-G's Bridge sends DMA and interrupts to the Hub at the widget
/// the guest programs into it, not to a fixed one: the second node's Hub
/// (widget 9) gets both when the guest names it, and the first's (10) when
/// the guest names that.
#[test]
fn the_bridge_sends_dma_and_interrupts_to_the_hub_the_guest_names() {
    use crate::dev::bridge::{reg, DIRECT_WIDGET_SHIFT};
    big_stack(|| {
        let (_dir, m) = built("bridge", 2);
        let bridge = m.bridge().clone();
        let card = Card::new();
        bridge.bus().attach(crate::pci::DevFn { device: 4, function: 0 }, card.clone());
        let set = |off: u32, v: u32| assert_eq!(bridge.write32(u64::from(off + 4), v), BUS_OK);
        let dma = card.state.dma().unwrap().clone();
        let memory = |i: usize, off: u64| m.node_at(i).memory.physical().read32(off).data;
        let pending = |i: usize| m.node_at(i).hub.read64(crate::dev::hub_pi::reg::INT_PEND0).data;

        // Node slot 2 is widget 9 and node slot 1 is widget 10.
        for (widget, node, other) in [(9u32, 1usize, 0usize), (10, 0, 1)] {
            set(reg::DIRECT_MAP, widget << DIRECT_WIDGET_SHIFT);
            let at = 0x1000 * u64::from(widget);
            assert!(dma.dma_write(0x8000_0000 + at, &[1, 2, 3, 4]).is_ok(), "widget {widget}");
            assert_ne!(memory(node, at), 0, "widget {widget}: the named node did not get it");
            assert_eq!(memory(other, at), 0, "widget {widget}: the other node saw it");

            // The interrupt goes the same way, to the vector the pin names.
            set(reg::INT_DEST_UPPER, widget << 16);
            set(reg::INT_DEST_LOWER, 0);
            set(reg::int_host(4), 5);
            set(reg::INT_ENABLE, 1 << 4);
            card.state.irq().unwrap().set(true);
            assert_eq!(pending(node) & 1 << 5, 1 << 5, "widget {widget}");
            assert_eq!(pending(other) & 1 << 5, 0, "widget {widget}: the other node heard it");
            card.state.irq().unwrap().set(false);
            // The guest's handler clears the pending bit.
            assert_eq!(m.node_at(node).hub.write64(crate::dev::hub_pi::reg::INT_PEND_MOD, 5), BUS_OK);
        }
    });
}

/// The BaseIO-G's Bridge line carries the base board's record and the
/// MediaIO's, as two parts with distinct ROM numbers.
#[test]
fn the_bridge_carries_the_baseio_and_mediaio_records() {
    let bus = Onyx2Deskside::board_record();
    let roms = bus.roms();
    assert_eq!(roms.len(), 2);
    assert_ne!(roms[0], roms[1], "a search tells the boards apart");
}

/// The BaseIO-G's Bridge has two IOC3s, at devices 2 and 6, the SCSI
/// controllers at 0 and 1 and RAD1 at 7; nothing at 5.
#[test]
fn the_baseio_g_has_a_second_ioc3_and_the_audio_chip() {
    big_stack(|| {
        let (_dir, m) = built("mio", 1);
        let bus = m.bridge().bus();
        let id = |device| bus.config_read(crate::pci::DevFn { device, function: 0 }, 0, 4);
        assert_eq!(id(0), Some(0x1020_1077), "an ISP1040");
        assert_eq!(id(2), Some(0x0003_10a9), "the board IOC3");
        assert_eq!(id(6), Some(0x0003_10a9), "the MediaIO IOC3");
        assert_eq!(id(7), Some(0x0005_10a9), "RAD1");
        assert_eq!(id(5), None, "nothing at 5");
    });
}

/// A SuperIO source raised on the MediaIO's IOC3 reaches Bridge input 6, and
/// one raised on the board's IOC3 reaches input 4. The table row's pin is the
/// only thing that sets either, and no guest reads it back from the chip.
#[test]
fn each_ioc3_super_io_interrupt_reaches_its_own_bridge_input() {
    use crate::dev::bridge::reg::INT_STATUS as BRIDGE_INT_STATUS;
    use crate::dev::ioc3::{int, reg as ioc3_reg};
    big_stack(|| {
        let (_dir, m) = built("sio-irq", 1);
        let bridge = m.bridge().clone();
        let bus = bridge.bus();
        // The keyboard controller answers a command byte with an
        // acknowledge, which raises the keyboard source.
        let raise = |device: u8| {
            let f = bus.function(crate::pci::DevFn { device, function: 0 }).unwrap();
            f.bar_write(0, ioc3_reg::INT_ENABLE_SET, int::KEYBOARD, 4);
            f.bar_write(0, 0xa8, 0xf4, 4);
            assert_ne!(f.bar_read(0, ioc3_reg::INT_STATUS, 4) & int::KEYBOARD, 0, "device {device}: source raised");
        };
        let pending = || bridge.register(BRIDGE_INT_STATUS) & 0xff;
        assert_eq!(pending(), 0);
        raise(6);
        assert_eq!(pending(), 1 << 6, "the MediaIO's interrupt is Bridge input 6 and no other");
        raise(2);
        assert_eq!(pending(), 1 << 6 | 1 << 4, "the board IOC3's is input 4");
    });
}
