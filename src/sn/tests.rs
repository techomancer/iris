use super::*;
use crate::dev::{bedrock, hub};
use crate::cpu::smp::{set_current_cpu, ParkSlot, HOST_PARK_BIT, REQUEST_JIT_REMAP};
use crate::physical::RamBank;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// A PROM stand-in: a recognisable first word and a configuration block in the shape the flash
/// initialiser patches.
pub(crate) fn prom() -> Vec<u8> {
    let mut p = vec![0u8; 0x1000];
    p[..4].copy_from_slice(&0x3c08_a600u32.to_be_bytes());
    p[0x68..0x70].copy_from_slice(&0x6970_3237_636f_6e66u64.to_be_bytes());
    p
}

/// A headless configuration of `profile` that builds in `dir`: the PROM stand-in, NVRAM in
/// the directory, no network, and every port the machine listens on left to any free one.
pub(crate) fn headless(dir: &std::path::Path, profile: crate::config::MachineProfile) -> MachineConfig {
    let mut cfg = MachineConfig::default();
    cfg.machine.profile = profile;
    let prom_path = dir.join("prom.bin");
    std::fs::write(&prom_path, prom()).unwrap();
    cfg.prom = prom_path.to_string_lossy().into_owned();
    cfg.nvram = dir.join("nvram.bin").to_string_lossy().into_owned();
    cfg.headless = true;
    cfg.network.mode = crate::config::NetMode::Off;
    cfg.monitor_port = Some(0);
    crate::dev::serial_host::tests::free_ports(&mut cfg);
    cfg
}

/// Build the machine `cfg` describes with `build` on a big stack, as `main` does, run `body` on it,
/// and drop it there.
pub(crate) fn with_machine<M: 'static, R: Send + 'static>(
    cfg: MachineConfig,
    build: fn(&MachineConfig) -> Result<M, String>,
    body: impl FnOnce(&mut M) -> R + Send + 'static,
) -> R {
    crate::display_stop_tests::big_stack(move || body(&mut build(&cfg).unwrap()))
}

const MIB: u64 = 1 << 20;
const ALIAS: u64 = 8 << 10;

struct Absent;
impl BusDevice for Absent {}

/// Reads one byte everywhere.
struct Byte(u8);
impl BusDevice for Byte {
    fn read8(&self, _: u64) -> BusRead8 {
        BusRead8::ok(self.0)
    }
}

/// A node of one `mib` MiB bank holding `seed` at every page, with `flash`
/// at the boot window, decoded as an IP27 or IP35 node by `shift`.
fn node(mib: u64, seed: u64, flash: Option<u8>, shift: u32) -> Arc<NodeDecode<()>> {
    let phys = Physical::with_window(vec![RamBank::new(mib as usize)], 2 << shift);
    phys.set_default(Arc::new(Absent), "absent");
    for page in (0..mib * MIB).step_by(4096) {
        phys.banks()[0].write64(page, seed);
    }
    let banks = vec![BankMapping { base: 0, end: mib * MIB, bank: 0, src: 0 }].into();
    let boot = flash.map(|b| (RESET_WINDOW, 0x1000, Arc::new(Byte(b)) as Arc<dyn BusDevice>));
    let decode = NodeDecode::new(shift, 1 << shift, Node::new(Weak::new(), phys, banks, boot, shift == bedrock::NODE_SHIFT), LogModule::Hub);
    decode.set_alias(0, ALIAS);
    decode
}

/// Two nodes, their ports connected, both holding zero, node `n`'s bank holding `n` at
/// every page.
fn pair(mib: [u64; 2], flash: Option<u8>, shift: u32) -> [Arc<NodeDecode<()>>; 2] {
    let nodes = [node(mib[0], 0, flash, shift), node(mib[1], 1, None, shift)];
    nodes[0].connect_peer(nodes[1].node());
    nodes[1].connect_peer(nodes[0].node());
    nodes
}
fn two() -> [Arc<NodeDecode<()>>; 2] {
    pair([64, 64], None, hub::NODE_SHIFT)
}

fn renumber(node: &NodeDecode<()>, n: u16) {
    node.remap(&|| node.set_number(n));
}

fn release(nodes: &[Arc<NodeDecode<()>>]) {
    for n in nodes {
        n.physical().release_devices();
    }
}

#[test]
fn renumbering_maps_each_node_and_the_other_at_their_numbers() {
    let nodes = two();
    let read = |i: usize, a: u64| {
        let r = nodes[i].physical().read64(a);
        (r.status == BUS_OK).then_some(r.data)
    };
    // Both hold zero: each shows only its own memory.
    assert_eq!((read(0, 0x1000), read(1, 0x1000)), (Some(0), Some(1)));
    assert_eq!(read(0, 1 << 32), None);
    // Node 1 takes number one: its own memory below the alias and at one,
    // node zero's above the alias; node 0 shows node 1's at one.
    renumber(&nodes[1], 1);
    assert_eq!((read(1, 0), read(1, ALIAS), read(1, (1 << 32) + ALIAS)), (Some(1), Some(0), Some(1)));
    assert_eq!((read(0, ALIAS), read(0, (1 << 32) + 0x1000)), (Some(0), Some(1)));
    // Node 0 takes two: nobody holds zero, so past node 1's alias is nothing.
    renumber(&nodes[0], 2);
    assert_eq!((read(1, ALIAS), read(1, (2 << 32) + 0x1000)), (None, Some(0)));
    // Back to zero: each its own again.
    renumber(&nodes[0], 0);
    renumber(&nodes[1], 0);
    assert_eq!((read(1, ALIAS), read(0, (1 << 32) + 0x1000)), (Some(1), None));
    release(&nodes);
}

/// A remap writes RAM over RAM: an address that is memory before and after never reads as the
/// default between.
#[test]
fn no_bus_error_shows_mid_remap() {
    let nodes = two();
    let done = AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| {
            for i in 0..2000 {
                renumber(&nodes[1], (i & 1) as u16);
            }
            done.store(true, Ordering::SeqCst);
        });
        // Node 1's offsets past the alias are its own memory or node zero's,
        // never nothing.
        while !done.load(Ordering::SeqCst) {
            for a in [ALIAS, 32 * MIB, 64 * MIB - 8] {
                assert_eq!(nodes[1].physical().read64(a).status, BUS_OK, "{a:#x}");
            }
        }
    });
    release(&nodes);
}

/// The flash answers only the window pages no bank covers; a renumbering moves it over the whole
/// window and back.
#[test]
fn the_boot_flash_fills_only_what_no_bank_covers() {
    const PAGE: u64 = 4096;
    let phys = Physical::with_window(vec![RamBank::new(512)], 2 << hub::NODE_SHIFT);
    phys.set_default(Arc::new(Absent), "absent");
    phys.banks()[0].write64(RESET_WINDOW, 7);
    // The bank ends one page into a three-page flash window.
    let banks = vec![BankMapping { base: 0, end: RESET_WINDOW + PAGE, bank: 0, src: 0 }].into();
    let flash: Arc<dyn BusDevice> = Arc::new(Byte(0x5a));
    let a = NodeDecode::new(
        hub::NODE_SHIFT,
        1 << hub::NODE_SHIFT,
        Node::new(Weak::new(), phys, banks, Some((RESET_WINDOW, 3 * PAGE, flash)), false),
        LogModule::Hub,
    );
    a.set_alias(0, ALIAS);
    let b = node(64, 1, None, hub::NODE_SHIFT);
    a.connect_peer(b.node());
    b.connect_peer(a.node());
    let at = |page: u64| a.physical().read64(RESET_WINDOW + page * PAGE).data;
    let flash_at = |page: u64| a.physical().read8(RESET_WINDOW + page * PAGE);
    assert_eq!(at(0), 7, "the bank wins where it reaches");
    assert_eq!((flash_at(1).data, flash_at(2).data), (0x5a, 0x5a));
    assert_eq!(flash_at(3).status, BUS_ERR, "past the window is nothing");
    // Numbered one, `a` shows only the alias of its own banks at zero and the
    // zero holder's 64 MiB: nothing covers the window.
    renumber(&a, 1);
    assert_eq!((flash_at(0).data, flash_at(1).data, flash_at(2).data), (0x5a, 0x5a, 0x5a));
    renumber(&a, 0);
    assert_eq!((at(0), flash_at(1).data), (7, 0x5a));
    release(&[a, b]);
}

/// A processor's view takes only the two node-qualified spaces.
#[test]
#[should_panic(expected = "not node-qualified")]
fn a_processor_view_of_an_unregistered_space_panics() {
    two()[0].cpu_view(UncachedSpace::Memory, None);
}

/// A port connects once.
#[test]
#[should_panic(expected = "connected once")]
fn a_port_connects_once() {
    let nodes = two();
    nodes[0].connect_peer(nodes[1].node());
}

/// Memory by number is RAM only: DMA to a boot window shown as flash is a bus error until a bank
/// covers it.
#[test]
fn memory_by_number_does_not_reach_the_boot_flash() {
    let nodes = pair([64, 512], Some(0x5a), hub::NODE_SHIFT);
    // The small node holds zero and shows its flash at the boot window: the
    // processor's cached decode reaches it, memory by number does not.
    assert_eq!(nodes[0].physical().read8(RESET_WINDOW).data, 0x5a);
    let dma = nodes[0].clone();
    assert_eq!(dma.read8(RESET_WINDOW).status, BUS_ERR);
    assert_eq!(dma.write64(RESET_WINDOW + 0xff8, 1), BUS_ERR);
    assert_eq!(dma.read_block(RESET_WINDOW, &mut [0; 4]), BUS_ERR, "nor does a block");
    // Renumbered, the small node no longer shows its flash, and the big one
    // holds zero: memory by number zero is the big node's bank.
    renumber(&nodes[0], 1);
    assert_eq!(dma.read64(RESET_WINDOW).data, 1);
    release(&nodes);
}

/// On the IP35 the boot window at node field zero stays on the reader: a renumbered node keeps its
/// own flash there.
#[test]
fn a_boot_local_node_keeps_its_flash_at_node_field_zero() {
    let nodes = pair([64, 512], Some(0x5a), bedrock::NODE_SHIFT);
    renumber(&nodes[0], 1);
    assert_eq!(nodes[0].physical().read8(RESET_WINDOW).data, 0x5a, "the reader's own flash");
    assert_eq!(nodes[0].physical().read64(RESET_WINDOW + 0x1000).data, 1, "the zero holder's memory above it");
    assert_eq!(nodes[0].physical().read64(0x1fb0_0000).data, 1, "and below it");
    release(&nodes);
}

/// With memory of its own behind the boot window a renumbered IP35 node shows no flash there,
/// and the window is the zero holder's memory like the rest of node field zero: IRIX hands those
/// pages out, and every processor, DMA master and block transfer has to reach the same bytes.
#[test]
fn memory_in_the_boot_window_is_the_zero_holders_for_both_nodes() {
    let nodes = pair([1024, 1024], Some(0x5a), bedrock::NODE_SHIFT);
    renumber(&nodes[0], 1);
    let at = RESET_WINDOW + 0x2000;
    // The zero holder stores; the renumbered node reads the same memory, and stores back.
    assert_eq!(nodes[1].physical().write64(at, 0x1122_3344_5566_7788), BUS_OK);
    assert_eq!(nodes[0].physical().read64(at).data, 0x1122_3344_5566_7788, "the zero holder's memory, not its own");
    assert_eq!(nodes[0].physical().write64(at + 8, 0xa5a5), BUS_OK);
    assert_eq!(nodes[1].physical().read64(at + 8).data, 0xa5a5);
    // Memory by number, as DMA and the block transfer engine reach it, agrees.
    let dma: &dyn BusDevice = &*nodes[0];
    assert_eq!(dma.read64(at).data, 0x1122_3344_5566_7788);
    // The renumbered node's own memory behind the window is at its number.
    let own = (1u64 << bedrock::NODE_SHIFT) + at;
    assert_ne!(nodes[0].physical().read64(own).data, 0x1122_3344_5566_7788);
    assert_eq!(nodes[1].physical().read64(own).data, nodes[0].physical().read64(own).data);
    release(&nodes);
}

/// A remap publishes and maps with every other processor parked.
#[test]
fn a_remap_publishes_with_the_other_processor_parked() {
    let probe = ParkProbe::new();
    let nodes = two();
    for n in &nodes {
        n.set_world(probe.world());
    }
    let parked = Mutex::new(Vec::new());
    let other = probe.slots[1].clone();
    assert!(probe.parks_the_other(|| {
        nodes[1].remap(&|| {
            parked.lock().push(other.is_parked());
            nodes[1].set_number(1);
        })
    }));
    assert_eq!(*parked.lock(), [true]);
    release(&nodes);
}

/// Two worlds would park only one node's processors.
#[test]
#[should_panic(expected = "two connected nodes park the same processors")]
fn connected_nodes_share_one_world() {
    let nodes = two();
    nodes[0].set_world(Arc::new(World::new()));
    nodes[1].set_world(Arc::new(World::new()));
}

/// A second processor that runs until a store parks it, for tests that a
/// placing store parks every other processor while the remap maps memory.
pub(crate) struct ParkProbe {
    world: Arc<World>,
    slots: Vec<Arc<ParkSlot>>,
    words: Vec<Arc<AtomicU64>>,
    /// Flushes the processors took.
    flushes: Arc<AtomicUsize>,
}

impl ParkProbe {
    pub(crate) fn new() -> Self {
        let words: Vec<Arc<AtomicU64>> = (0..2).map(|_| Arc::new(AtomicU64::new(0))).collect();
        let slots: Vec<Arc<ParkSlot>> = words.iter().enumerate().map(|(cpu, w)| Arc::new(ParkSlot::new(cpu as u8, Arc::as_ptr(w)))).collect();
        let world = Arc::new(World::new());
        world.set_slots(slots.clone());
        Self { world, slots, words, flushes: Default::default() }
    }

    /// The world holding both processors, for the remap.
    pub(crate) fn world(&self) -> Arc<World> {
        self.world.clone()
    }

    /// The JIT flushes the two processors have taken so far.
    pub(crate) fn flushes(&self) -> usize {
        self.flushes.load(Ordering::SeqCst)
    }

    /// Run `store` as processor 0 while processor 1 runs: whether processor
    /// 1 parked while it did.
    pub(crate) fn parks_the_other(&self, store: impl FnOnce() + Send) -> bool {
        let running = Arc::new(AtomicBool::new(true));
        let parks = Arc::new(AtomicUsize::new(0));
        self.slots[1].set_active(true);
        let other = {
            let (slot, word, running, parks, flushes) =
                (self.slots[1].clone(), self.words[1].clone(), running.clone(), parks.clone(), self.flushes.clone());
            std::thread::spawn(move || {
                set_current_cpu(1);
                while running.load(Ordering::Relaxed) {
                    if word.load(Ordering::Relaxed) & HOST_PARK_BIT == 0 {
                        std::thread::yield_now();
                    } else if slot.release(REQUEST_JIT_REMAP) {
                        flushes.fetch_add(1, Ordering::SeqCst);
                    } else {
                        slot.park(&running);
                        parks.fetch_add(1, Ordering::SeqCst);
                    }
                }
                slot.set_active(false);
            })
        };
        std::thread::scope(|s| {
            s.spawn(|| {
                set_current_cpu(0);
                store();
                if self.slots[0].release(REQUEST_JIT_REMAP) {
                    self.flushes.fetch_add(1, Ordering::SeqCst);
                }
            });
        });
        // Processor 1 takes a flush posted to it before it stops.
        while self.slots[1].holds() & REQUEST_JIT_REMAP != 0 {
            std::thread::yield_now();
        }
        running.store(false, Ordering::Relaxed);
        other.join().unwrap();
        parks.load(Ordering::SeqCst) > 0
    }
}

/// A two-brick IP35 machine's processors reach the second node's memory at node number 1, so the
/// JIT's map runs that far.
#[test]
fn the_processors_memory_covers_the_second_ip35_node() {
    use crate::dev::bedrock::RamExtent;
    let one = crate::ip35::memory_map(&RamExtent::new(&[256, 256]));
    assert_eq!(machine_memory(std::slice::from_ref(&one), bedrock::NODE_SHIFT), one);
    let two = machine_memory(&[one.clone(), one.clone()], bedrock::NODE_SHIFT);
    assert_eq!(two.end(), (1 << bedrock::NODE_SHIFT) + one.end());
    assert!(two.is_ram((1 << bedrock::NODE_SHIFT) + 0x1000));
    // Eight 1 GiB banks a node, the most a Bedrock decodes: 16 GiB.
    let full = crate::ip35::memory_map(&RamExtent::new(&[1024; 8]));
    assert_eq!(machine_memory(&[full.clone(), full], bedrock::NODE_SHIFT).end(), 16 << 30);
}

/// A flash file that cannot be opened is refused with the table's setting and the path, on both
/// families.
#[test]
fn a_bad_flash_file_is_refused_with_its_setting_and_path() {
    use crate::config::MachineProfile;
    use crate::test_temp::TempPath;
    let dir = TempPath::dir("sn-bad-flash");
    let shown = dir.join("missing/flash.bin").display().to_string();
    let mut origin200 = crate::ip27::test_support::config(MachineProfile::Origin200Ip29, &dir, 1);
    origin200.origin200.flash = Some(shown.clone());
    let mut fuel = crate::ip35::test_support::fuel_config(&dir);
    fuel.fuel.flash = Some(shown.clone());
    crate::display_stop_tests::big_stack(move || {
        let refusals = [crate::ip27::test_support::machine(&origin200).err(), crate::ip35::test_support::machine(&fuel).err()];
        for (table, why) in ["origin200", "fuel"].into_iter().zip(refusals) {
            let why = why.expect("the machine is refused");
            assert!(why.starts_with(&format!("machine.flash {shown}: ")), "{why}");
        }
    });
}

/// Records the order in which it is started and stopped.
struct Marker(&'static str, Arc<Mutex<Vec<String>>>);
impl Device for Marker {
    fn step(&self, _: u64) {}
    fn start(&self) {
        self.1.lock().push(format!("start {}", self.0));
    }
    fn stop(&self) {
        self.1.lock().push(format!("stop {}", self.0));
    }
    fn is_running(&self) -> bool {
        false
    }
}

/// A machine starts its board's devices in order and stops them in reverse: markers before and
/// after the board's own start first to last and stop last to first, on an Origin 200 (whose module
/// controller's watch, the board's last device, runs once started and idles once stopped) and on a
/// Tezro with a V12.
#[test]
fn a_machine_starts_its_board_devices_in_order_and_stops_them_in_reverse() {
    use crate::config::{GraphicsBoard, MachineProfile};
    use crate::test_temp::TempPath;
    use crate::workstation::Workstation;
    fn order<C: Device + Resettable + 'static, F: Device + Resettable + 'static>(m: &mut Machine<C, F>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        m.io.devices.insert(0, Arc::new(Marker("first", log.clone())));
        m.io.devices.push(Arc::new(Marker("last", log.clone())));
        m.restart_peripherals();
        m.stop();
        let log = log.lock().clone();
        assert_eq!(log, ["start first", "start last", "stop last", "stop first"]);
    }
    let dir = TempPath::dir("sn-device-order");
    let origin200 = crate::ip27::test_support::config(MachineProfile::Origin200Ip29, &dir, 1);
    let mut tezro = crate::ip35::test_support::tezro_config(&dir, 2);
    tezro.graphics.board = Some(GraphicsBoard::V12);
    crate::display_stop_tests::big_stack(move || {
        let mut m = crate::ip27::test_support::machine(&origin200).unwrap();
        let watch = m.io.devices.last().unwrap().clone();
        assert!(!watch.is_running());
        m.restart_peripherals();
        assert!(watch.is_running());
        m.stop();
        assert!(!watch.is_running());
        order(&mut m);
        order(&mut crate::ip35::test_support::machine(&tezro).unwrap());
    });
}

/// A snapshot is refused with the reason, in both directions, rather than saved from a machine
/// whose chips and node memory have nothing to save.
#[test]
fn a_snapshot_is_refused_with_a_reason() {
    use crate::config::MachineProfile;
    use crate::test_temp::TempPath;
    let dir = TempPath::dir("sn-snapshot");
    let mut onyx2 = crate::ip27::test_support::config(MachineProfile::Onyx2Ip27, &dir, 2);
    onyx2.monitor_port = Some(0);
    let origin3200 = crate::ip35::test_support::config(&dir, 2);
    fn refused<C: Device + Resettable + 'static, F: Device + Resettable + 'static>(m: &mut Machine<C, F>, product: &str) {
        for result in [Workstation::save_snapshot(m, "s"), Workstation::load_snapshot(m, "s")] {
            let why = result.unwrap_err();
            assert!(why.contains(product) && why.contains("not supported yet") && why.contains("no saved state"), "{why}");
        }
    }
    with_machine(onyx2, crate::ip27::test_support::machine, |m| refused(m, "Onyx2"));
    with_machine(origin3200, crate::ip35::test_support::machine, |m| refused(m, "Origin 3200"));
}

/// A dropped machine frees every processor and node chip, the second node numbered one so each
/// node's table points into the other's banks: a two-node Onyx2 and a two-C-brick Origin 3200.
#[test]
fn a_dropped_machine_frees_its_processors_and_nodes() {
    use crate::config::MachineProfile;
    use crate::test_temp::TempPath;
    fn handles<C: Device + Resettable + 'static, F: Device + Resettable + 'static>(m: Machine<C, F>) -> (Vec<Weak<dyn CpuDevice>>, Vec<Weak<C>>) {
        let cpus = m.cpus().map(Arc::downgrade).collect();
        let chips = m.nodes.iter().map(|n| Arc::downgrade(&n.chip)).collect();
        drop(m);
        (cpus, chips)
    }
    let dir = TempPath::dir("sn-drop");
    let onyx2 = crate::ip27::test_support::config(MachineProfile::Onyx2Ip27, &dir, 2);
    let origin3200 = crate::ip35::test_support::config(&dir, 2);
    crate::display_stop_tests::big_stack(move || {
        let m = crate::ip27::test_support::machine(&onyx2).unwrap();
        assert_eq!(m.nodes[1].chip.write64(crate::dev::hub::regs::NI_STATUS_REV_ID::OFFSET, 1 << 8), BUS_OK);
        assert_eq!(m.nodes[0].chip.decode.physical().read64(1 << 32).status, BUS_OK, "node 0 shows node 1's memory");
        let (cpus, chips) = handles(m);
        assert!(cpus.iter().all(|c| c.upgrade().is_none()), "a processor outlives the Onyx2");
        assert!(chips.iter().all(|c| c.upgrade().is_none()), "a Hub outlives the Onyx2");

        let m = crate::ip35::test_support::machine(&origin3200).unwrap();
        {
            use crate::dev::bedrock::regs::{lb, md};
            for node in &m.nodes {
                assert_eq!(node.chip.registers().write64(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x33), BUS_OK);
            }
            assert_eq!(m.nodes[1].chip.registers().write64(lb::BASE + lb::REV_ID::OFFSET, 1 << 32), BUS_OK);
        }
        assert_eq!(m.nodes[0].chip.decode.physical().read64(1 << 33).status, BUS_OK, "node 0 shows node 1's memory");
        let (cpus, chips) = handles(m);
        assert!(cpus.iter().all(|c| c.upgrade().is_none()), "a processor outlives the Origin 3200");
        assert!(chips.iter().all(|c| c.upgrade().is_none()), "a node chip outlives the Origin 3200");
    });
}

/// IRIX counts the Timekeeper's year cell from 1970 on the IP27 machines and 1968 on the IP35 ones.
#[test]
fn each_family_counts_the_timekeeper_year_from_its_own_base() {
    use crate::test_temp::TempPath;
    let (o200, fuel) = (TempPath::dir("sn-year-origin200"), TempPath::dir("sn-year-fuel"));
    let (o200_cfg, fuel_cfg) = (crate::ip27::test_support::config(crate::config::MachineProfile::Origin200Ip29, &o200, 1), crate::ip35::test_support::fuel_config(&fuel));
    crate::display_stop_tests::big_stack(move || {
        drop(crate::ip27::test_support::machine(&o200_cfg).unwrap());
        drop(crate::ip35::test_support::machine(&fuel_cfg).unwrap());
    });
    let year = |d: &TempPath| std::fs::read(d.join("nvram.bin")).unwrap()[0x7fff];
    let bcd = |b: u8| u32::from(b >> 4) * 10 + u32::from(b & 15);
    assert_eq!(bcd(year(&o200)) + 1970, bcd(year(&fuel)) + 1968);
}
