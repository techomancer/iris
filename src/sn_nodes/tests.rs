use super::*;
use crate::cpu::smp::{set_current_cpu, ParkSlot, HOST_PARK_BIT, REQUEST_JIT_FLUSH};
use crate::physical::RamBank;
use crate::traits::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
/// at the boot window, decoded as an SN0 or SN1 node by `shift`.
fn node(mib: u64, seed: u64, flash: Option<u8>, shift: u32) -> Arc<NodeDecode<()>> {
    let phys = Physical::with_window(vec![RamBank::new(mib as usize)], 2 << shift);
    phys.set_default(Arc::new(Absent), "absent");
    for page in (0..mib * MIB).step_by(4096) {
        phys.banks()[0].write64(page, seed);
    }
    let banks = vec![BankMapping { base: 0, end: mib * MIB, bank: 0, src: 0 }].into();
    let boot = flash.map(|b| (RESET_WINDOW, 0x1000, Arc::new(Byte(b)) as Arc<dyn BusDevice>));
    let decode = NodeDecode::new(shift, 1 << shift, Node::new(Weak::new(), phys, banks, boot, shift == SN1_NODE_SHIFT), LogModule::Hub);
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
    pair([64, 64], None, SN0_NODE_SHIFT)
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

/// A remap writes RAM over RAM: an address that is memory before and after never reads as the default between.
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

/// The flash answers only the window pages no bank covers; a renumbering moves it over the whole window and back.
#[test]
fn the_boot_flash_fills_only_what_no_bank_covers() {
    const PAGE: u64 = 4096;
    let phys = Physical::with_window(vec![RamBank::new(512)], 2 << SN0_NODE_SHIFT);
    phys.set_default(Arc::new(Absent), "absent");
    phys.banks()[0].write64(RESET_WINDOW, 7);
    // The bank ends one page into a three-page flash window.
    let banks = vec![BankMapping { base: 0, end: RESET_WINDOW + PAGE, bank: 0, src: 0 }].into();
    let flash: Arc<dyn BusDevice> = Arc::new(Byte(0x5a));
    let a = NodeDecode::new(SN0_NODE_SHIFT, 1 << SN0_NODE_SHIFT, Node::new(Weak::new(), phys, banks, Some((RESET_WINDOW, 3 * PAGE, flash)), false), LogModule::Hub);
    a.set_alias(0, ALIAS);
    let b = node(64, 1, None, SN0_NODE_SHIFT);
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

/// Memory by number is RAM only: DMA to a boot window shown as flash is a bus error until a bank covers it.
#[test]
fn memory_by_number_does_not_reach_the_boot_flash() {
    let nodes = pair([64, 512], Some(0x5a), SN0_NODE_SHIFT);
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

/// On the SN1 the boot window at node field zero stays on the reader: a renumbered node keeps its own flash there.
#[test]
fn a_boot_local_node_keeps_its_flash_at_node_field_zero() {
    let nodes = pair([64, 512], Some(0x5a), SN1_NODE_SHIFT);
    renumber(&nodes[0], 1);
    assert_eq!(nodes[0].physical().read8(RESET_WINDOW).data, 0x5a, "the reader's own flash");
    assert_eq!(nodes[0].physical().read64(RESET_WINDOW + 0x1000).data, 1, "the zero holder's memory above it");
    assert_eq!(nodes[0].physical().read64(0x1fb0_0000).data, 1, "and below it");
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
                    } else if slot.release(REQUEST_JIT_FLUSH) {
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
                if self.slots[0].release(REQUEST_JIT_FLUSH) {
                    self.flushes.fetch_add(1, Ordering::SeqCst);
                }
            });
        });
        // Processor 1 takes a flush posted to it before it stops.
        while self.slots[1].holds() & REQUEST_JIT_FLUSH != 0 {
            std::thread::yield_now();
        }
        running.store(false, Ordering::Relaxed);
        other.join().unwrap();
        parks.load(Ordering::SeqCst) > 0
    }
}

/// A two-brick SN1's processors reach the second node's memory at node number 1, so the JIT's map runs that far.
#[test]
fn the_processors_memory_covers_the_second_sn1_node() {
    use crate::dev::bedrock::RamExtent;
    let one = crate::ip35::memory_map(&RamExtent::new(&[256, 256]));
    assert_eq!(machine_memory(&[one.clone()], SN1_NODE_SHIFT), one);
    let two = machine_memory(&[one.clone(), one.clone()], SN1_NODE_SHIFT);
    assert_eq!(two.end(), (1 << SN1_NODE_SHIFT) + one.end());
    assert!(two.is_ram((1 << SN1_NODE_SHIFT) + 0x1000));
    // Eight 1 GiB banks a node, the most a Bedrock decodes: 16 GiB.
    let full = crate::ip35::memory_map(&RamExtent::new(&[1024; 8]));
    assert_eq!(machine_memory(&[full.clone(), full], SN1_NODE_SHIFT).end(), 16 << 30);
}
