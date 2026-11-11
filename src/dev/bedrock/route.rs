//! Node-number routing between two cabled chips.
//!
//! Address bits 39:33 name a node (M mode: a 7-bit field and an 8 GiB node
//! window; the Origin 300 pair runs in it). A chip keeps a table from that
//! field to the chip that answers it, written when a node number or the link
//! changes and read with one atomic load per access, so an access takes no
//! lock and tests nothing about how many chips there are.
//!
//! A chip's board registers what answers each space of its own node
//! ([`super::Bedrock::set_space`]), addressed from the node's base, and maps
//! the views the chip hands out ([`super::Bedrock::cpu_view`],
//! [`super::Bedrock::claim`]) into its processors' buses. The registered
//! devices are the node's local decode and must not hold a view, or the
//! chips would hold each other: a chip's table points at the peer's
//! registered devices.
//!
//! # What is approximated here
//!
//! A remote access is performed on the peer's own decode of the same space,
//! so it reaches whatever the peer's processors reach there. The hardware's
//! narrower list of remotely reachable blocks is not enforced. No coherence
//! runs between the nodes: both see the same host memory.
//!
//! A node numbered other than zero sends its processors' references that name
//! node field zero to whichever node carries 0. The forms that are relative
//! to the reader stay on the node ([`Routes::stays`]): the cached alias, the
//! reset PROM's cached window, the Hub-special space's first 512 MiB and the
//! I/O space's local register form. The cached alias size is the larger of
//! the two processor interfaces' registers, and the Hub-special bound and
//! the cached PROM window are inferred from the Origin 300 firmware's
//! references; no source states them for this chip. Inbound DMA, the
//! block-transfer engines and interrupt packets name nodes by number, node
//! field zero included, with none of those forms.
//!
//! Failing an access to a number no reachable chip holds, rather than timing
//! out, is an approximation.
//!
//! Ported from hw/mips/sgi/sn1-node.c and sn1-fabric.c in
//! jamesbraid/qemu-sgi-origami.

use crate::traits::*;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::{Arc, OnceLock, Weak};

/// The node field of a physical address starts here.
const NODE_SHIFT: u32 = 33;
const NODE_COUNT: usize = 128;
const LOCAL_MASK: u64 = (1 << NODE_SHIFT) - 1;
/// A processor's accesses end here: 40 bits of physical address.
const ADDRESS_END: u64 = 1 << 40;

/// Nothing answers the number.
const NONE: u8 = 0;
/// This chip does.
const SELF: u8 = 1;
/// The chip across the cable does.
const PEER: u8 = 2;

/// The Hub-special space's reader-relative forms end here: its uncached view
/// of the first 256 MiB of node memory, the local registers and L1 window
/// and the boot windows.
const HUB_LOCAL_FORMS: u64 = 0x2000_0000;
/// The I/O space's local register form.
const IO_LOCAL_FORM: u64 = 0x0100_0000;
const IO_LOCAL_FORM_SIZE: u64 = 0x80_0000;

/// One of a node's address spaces, as a processor reaches it: cached memory
/// and the four uncached views (raw address bits 58:57).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    Cached,
    HubSpecial,
    Io,
    Fetchop,
    Memory,
}
const SPACES: usize = 5;

/// What a cable shows of a chip to the one at its other end. It holds
/// nothing back that would keep the chip alive.
pub(super) struct NodeView {
    pub(super) chip: Weak<super::Bedrock>,
    routes: Weak<Routes>,
    /// The node number LB's region register holds.
    pub(super) number: Arc<AtomicU8>,
    /// Node memory by node-local address: what DMA and the block-transfer
    /// engines of the other node reach.
    dma: Arc<dyn BusDevice>,
    /// The node's local decode of each space, registered by its board.
    spaces: [OnceLock<Arc<dyn BusDevice>>; SPACES],
}
impl NodeView {
    pub(super) fn new(chip: Weak<super::Bedrock>, routes: Weak<Routes>, number: Arc<AtomicU8>, dma: Arc<dyn BusDevice>) -> Self {
        Self { chip, routes, number, dma, spaces: Default::default() }
    }
}

/// The state of a cable, shared by both ends.
pub(super) struct Link {
    up: AtomicBool,
    /// Serialises the writers of both ends' tables.
    writers: Mutex<()>,
}

/// A chip's end of its cable: the other chip's view and the link.
pub(super) struct Cabled {
    peer: Arc<NodeView>,
    link: Arc<Link>,
}
impl Cabled {
    pub(super) fn is_up(&self) -> bool {
        self.link.up.load(Relaxed)
    }
    pub(super) fn peer(&self) -> &NodeView {
        &self.peer
    }
}

/// A chip's decode of node numbers.
pub(super) struct Routes {
    own: Arc<NodeView>,
    pub(super) cabled: OnceLock<Cabled>,
    /// Which chip answers each node field, by number alone.
    named: [AtomicU8; NODE_COUNT],
    /// The same for a processor, which differs at node field zero only: a
    /// node that does not hold 0 and has no peer that does keeps its own
    /// local forms there.
    reader: [AtomicU8; NODE_COUNT],
    /// Each processor interface's cached-alias size in bytes.
    alias: [AtomicU64; 2],
    /// Serialises the writers of an uncabled chip's tables.
    solo: Mutex<()>,
}

impl Routes {
    pub(super) fn new(make_view: impl FnOnce(&Weak<Routes>) -> NodeView) -> Arc<Self> {
        Arc::new_cyclic(|weak| {
            let routes = Self {
                own: Arc::new(make_view(weak)),
                cabled: OnceLock::new(),
                named: std::array::from_fn(|_| AtomicU8::new(NONE)),
                reader: std::array::from_fn(|_| AtomicU8::new(NONE)),
                alias: Default::default(),
                solo: Mutex::new(()),
            };
            // A chip numbered 0 holds node field zero until it is numbered.
            routes.named[0].store(SELF, Relaxed);
            routes.reader[0].store(SELF, Relaxed);
            routes
        })
    }
    pub(super) fn view(&self) -> &Arc<NodeView> {
        &self.own
    }
    /// Join the chips whose routes are `a` and `b`.
    pub(super) fn cable(a: &Arc<Self>, b: &Arc<Self>) {
        let link = Arc::new(Link { up: AtomicBool::new(true), writers: Mutex::new(()) });
        let end = |peer: &Arc<Self>| Cabled { peer: peer.own.clone(), link: link.clone() };
        assert!(a.cabled.set(end(b)).is_ok() && b.cabled.set(end(a)).is_ok(), "a Bedrock is cabled once");
        a.rebuild();
    }
    /// Mark the link up or down, for both ends.
    pub(super) fn set_link(&self, up: bool) {
        if let Some(cabled) = self.cabled.get() {
            let _writer = cabled.link.writers.lock();
            cabled.link.up.store(up, Relaxed);
            self.derive_both(cabled);
        }
    }
    /// Re-derive the tables after this chip's number changed.
    pub(super) fn rebuild(&self) {
        match self.cabled.get() {
            Some(cabled) => {
                let _writer = cabled.link.writers.lock();
                self.derive_both(cabled);
            }
            None => {
                let _writer = self.solo.lock();
                self.derive(None);
            }
        }
    }
    /// A processor interface's cached-alias register holds `code`.
    pub(super) fn set_alias(&self, interface: usize, code: u64) {
        let bytes = if code & 15 == 0 { 0 } else { 0x1000 << ((code & 15) - 1) };
        self.alias[interface].store(bytes, Relaxed);
    }
    pub(super) fn alias_bytes(&self) -> u64 {
        self.alias[0].load(Relaxed).max(self.alias[1].load(Relaxed))
    }
    /// Whether the chip across the cable answers `number`.
    pub(super) fn peer_holds(&self, number: u8) -> bool {
        usize::from(number) < NODE_COUNT && self.named[usize::from(number)].load(Relaxed) == PEER
    }

    fn derive_both(&self, cabled: &Cabled) {
        self.derive(Some(cabled));
        if let Some(peer) = cabled.peer.routes.upgrade() {
            if let Some(back) = peer.cabled.get() {
                peer.derive(Some(back));
            }
        }
    }
    /// Place this chip's claims: itself at its own number and the peer at its
    /// number while the link is up. Two chips carrying one number each reach
    /// only themselves there. Callers hold the writers' lock.
    ///
    /// Each table is computed whole and only the entries that change are
    /// stored, new claims first, so a reader on another thread never sees an
    /// entry go away and come back, and a claim that moves is never absent.
    fn derive(&self, cabled: Option<&Cabled>) {
        let index = |number: &AtomicU8| usize::from(number.load(Relaxed)) % NODE_COUNT;
        let own = index(&self.own.number);
        let mut named = [NONE; NODE_COUNT];
        if let Some(cabled) = cabled.filter(|cabled| cabled.is_up()) {
            let peer = index(&cabled.peer.number);
            if peer != own {
                named[peer] = PEER;
            }
        }
        named[own] = SELF;
        let mut reader = named;
        if reader[0] == NONE {
            reader[0] = SELF;
        }
        for (table, wanted) in [(&self.named, &named), (&self.reader, &reader)] {
            for removing in [false, true] {
                for (claim, &wanted) in table.iter().zip(wanted) {
                    if (wanted == NONE) == removing && claim.load(Relaxed) != wanted {
                        claim.store(wanted, Relaxed);
                    }
                }
            }
        }
    }

    /// Whether a processor's reference at node-local `local` in `space`,
    /// naming node field zero, stays on this node although the field is held
    /// across the cable.
    fn stays(&self, space: Space, local: u64) -> bool {
        match space {
            Space::Cached => {
                local.wrapping_sub(crate::ip35::RESET_WINDOW) < crate::ip35::RESET_WINDOW_SIZE || local < self.alias_bytes()
            }
            Space::HubSpecial => local < HUB_LOCAL_FORMS,
            Space::Io => local.wrapping_sub(IO_LOCAL_FORM) < IO_LOCAL_FORM_SIZE,
            // Memory-special has no local form, whatever the offset.
            Space::Fetchop => false,
            Space::Memory => local < self.alias_bytes(),
        }
    }
}

/// What a [`View`] reaches.
#[derive(Clone, Copy)]
enum Kind {
    /// A processor's space: node field zero is the reader's, in the forms
    /// that are relative to it, and whichever node holds 0 elsewhere.
    Processor(Space),
    /// A space named by node number alone, node field zero included.
    Named(Space),
    /// Node memory named by node number alone: what DMA and the
    /// block-transfer engines reach.
    Dma,
}

/// A node-qualified address space as a processor, an engine or a widget
/// sees it: the node field picks the chip and the rest the address within it.
pub(super) struct View {
    routes: Arc<Routes>,
    kind: Kind,
}

impl View {
    pub(super) fn processor(routes: &Arc<Routes>, space: Space) -> Self {
        Self { routes: routes.clone(), kind: Kind::Processor(space) }
    }
    pub(super) fn named(routes: &Arc<Routes>, space: Space) -> Self {
        Self { routes: routes.clone(), kind: Kind::Named(space) }
    }
    pub(super) fn dma(routes: &Arc<Routes>) -> Self {
        Self { routes: routes.clone(), kind: Kind::Dma }
    }
    fn target(&self, a: u64) -> Option<(&dyn BusDevice, u64)> {
        if a >= ADDRESS_END {
            return None;
        }
        let routes = &*self.routes;
        let number = (a >> NODE_SHIFT) as usize;
        let local = a & LOCAL_MASK;
        let (table, space) = match self.kind {
            Kind::Processor(space) => (&routes.reader, Some(space)),
            Kind::Named(space) => (&routes.named, Some(space)),
            Kind::Dma => (&routes.named, None),
        };
        let node = match table[number].load(Relaxed) {
            SELF => &routes.own,
            PEER => match self.kind {
                Kind::Processor(space) if number == 0 && routes.stays(space, local) => &routes.own,
                _ => &routes.cabled.get()?.peer,
            },
            _ => return None,
        };
        let device = match space {
            Some(space) => node.spaces[space as usize].get()?,
            None => &node.dma,
        };
        Some((device.as_ref(), local))
    }
    fn unclaimed(&self, a: u64) {
        crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock: access at {a:#x}: no chip holds node {} or its space is unwired",
            (a >> NODE_SHIFT) & (NODE_COUNT as u64 - 1));
    }
}
macro_rules! view_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            match self.target(a) {
                Some((device, at)) => device.$r(at),
                None => {
                    self.unclaimed(a);
                    $out::err()
                }
            }
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            match self.target(a) {
                Some((device, at)) => device.$w(at, v),
                None => {
                    self.unclaimed(a);
                    BUS_ERR
                }
            }
        }
    };
}
impl BusDevice for View {
    fn write64_masked(&self, a: u64, val: u64, mask: u64) -> u32 {
        self.target(a).map_or(BUS_ERR, |(device, at)| device.write64_masked(at, val, mask))
    }
    fn write_narrow_from(&self, a: u64, size: usize, gpr: u64) -> u32 {
        self.target(a).map_or(BUS_ERR, |(device, at)| device.write_narrow_from(at, size, gpr))
    }
    // A store-conditional to another node's memory is the claimant's swap,
    // so two nodes' processors cannot both win the same word.
    fn cas32(&self, a: u64, old: u32, new: u32) -> Option<bool> {
        self.target(a).and_then(|(device, at)| device.cas32(at, old, new))
    }
    fn cas64(&self, a: u64, old: u64, new: u64) -> Option<bool> {
        self.target(a).and_then(|(device, at)| device.cas64(at, old, new))
    }
    // A pointer or counter the node holding the memory hands out is valid
    // wherever the memory is reached from, as the memory is the host's.
    fn mem_ptr(&self, a: u64) -> Option<*const u64> {
        self.target(a).and_then(|(device, at)| device.mem_ptr(at))
    }
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, a: u64) -> *const AtomicU64 {
        self.target(a).map_or(std::ptr::null(), |(device, at)| device.gen_ptr(at))
    }
    fn read_block(&self, a: u64, out: &mut [u64]) -> u32 {
        self.target(a).map_or(BUS_ERR, |(device, at)| device.read_block(at, out))
    }
    fn write_block(&self, a: u64, data: &[u64]) -> u32 {
        self.target(a).map_or(BUS_ERR, |(device, at)| device.write_block(at, data))
    }
    fn dma_read64(&self, a: u64) -> BusRead64 {
        self.target(a).map_or_else(BusRead64::err, |(device, at)| device.dma_read64(at))
    }
    fn dma_write64(&self, a: u64, v: u64) -> u32 {
        self.target(a).map_or(BUS_ERR, |(device, at)| device.dma_write64(at, v))
    }
    view_access!(read8, write8, u8, BusRead8);
    view_access!(read16, write16, u16, BusRead16);
    view_access!(read32, write32, u32, BusRead32);
    view_access!(read64, write64, u64, BusRead64);
}

impl super::Bedrock {
    /// Register `device` as the node's local decode of `space`, addressed
    /// from the node's base, once, before the machine runs. It is what
    /// reaches the node when another node's processor names it, and what the
    /// node's own views reach for its own number. It must not hold a view of
    /// this chip.
    pub fn set_space(&self, space: Space, device: Arc<dyn BusDevice>) {
        let slot = &self.routes.view().spaces[space as usize];
        assert!(slot.set(device).is_ok(), "{space:?} is wired twice");
    }
    /// A processor's view of `space`, for a board to map in place of its own
    /// decode: node field zero stays on the node in the reader-relative
    /// forms and goes to the node that holds 0 beyond them; any other field
    /// goes to the node that holds that number.
    pub fn cpu_view(&self, space: Space) -> Arc<dyn BusDevice> {
        Arc::new(View::processor(&self.routes, space))
    }
    /// `space` named by node number alone: every field, zero included, goes
    /// to the node that holds it, and one nobody holds fails.
    pub fn claim(&self, space: Space) -> Arc<dyn BusDevice> {
        Arc::new(View::named(&self.routes, space))
    }
}

#[cfg(test)]
mod tests {
    // Cases follow tests/qtest/sn1-cross-node-test.c and sn1-bedrock-link-test.c
    // in jamesbraid/qemu-sgi-origami: the node-number decode and the
    // router-less pair's link.
    use super::super::reg::{bte, io, lb, md, pi, IO_BASE, LB_BASE, MD_BASE};
    use super::super::{cabled_pair, Bedrock, Rig};
    use super::*;

    const PROBE: u64 = 0x1000;
    const WORD0: u64 = 0x0a00;
    const WORD1: u64 = 0x0b11;
    /// A register of the local block, in the node-named register form of the
    /// I/O space.
    const SCRATCH2: u64 = LB_BASE + lb::PIO_WINDOW + 16;
    const NODE_NAMED: u64 = 0x0180_0000;
    const NODE_LOCAL: u64 = 0x0100_0000;

    fn node_at(n: u64) -> u64 {
        n << NODE_SHIFT
    }

    /// Answers every read with one word, wherever it is asked.
    struct Id(u64);
    impl BusDevice for Id {
        fn read64(&self, _: u64) -> BusRead64 {
            BusRead64::ok(self.0)
        }
    }
    /// The I/O space as a board decodes it: the register window in its local
    /// and node forms, and a word that names the node elsewhere.
    struct IoSpace {
        local: Arc<dyn BusDevice>,
        named: Arc<dyn BusDevice>,
        id: u64,
    }
    impl BusDevice for IoSpace {
        fn read64(&self, a: u64) -> BusRead64 {
            if a.wrapping_sub(NODE_LOCAL) < 0x80_0000 {
                self.local.read64(a - NODE_LOCAL)
            } else if a.wrapping_sub(NODE_NAMED) < 0x80_0000 {
                self.named.read64(a - NODE_NAMED)
            } else {
                BusRead64::ok(self.id)
            }
        }
    }
    /// A space device the drop test can watch.
    struct Probe(u64);
    impl BusDevice for Probe {
        fn read64(&self, _: u64) -> BusRead64 {
            BusRead64::ok(self.0)
        }
    }

    /// A node as the routing needs it: 64 MiB of memory with a word at
    /// [`PROBE`] and a register that says who answered, and the board's
    /// local decode of each space.
    fn furnish(rig: &Rig, id: u64, word: u64) {
        let chip = &rig.chip;
        rig.w(MD_BASE + md::CONFIG, 0x11);
        assert_eq!(chip.dma_view().write64(PROBE, word), BUS_OK);
        rig.w(SCRATCH2, 0x5c0 + id);
        chip.set_space(Space::Cached, chip.dma_view());
        chip.set_space(Space::Memory, chip.uncached_view());
        chip.set_space(Space::Fetchop, Arc::new(Id(0xf0 + id)));
        chip.set_space(Space::HubSpecial, Arc::new(Id(0x100 + id)));
        chip.set_space(Space::Io, Arc::new(IoSpace { local: chip.local_registers(), named: chip.registers(), id: 0x200 + id }));
    }
    /// Two cabled nodes, each holding a word that says who answered at the
    /// same node-local address.
    fn pair() -> (Rig, Rig) {
        let (a, b) = cabled_pair();
        furnish(&a, 0, WORD0);
        furnish(&b, 1, WORD1);
        (a, b)
    }
    /// The master's number store: the revision word with the node field replaced.
    fn number(rig: &Rig, n: u64) {
        rig.w(LB_BASE + lb::NODE_REGION, n << 32);
    }
    fn read(device: &Arc<dyn BusDevice>, address: u64) -> Option<u64> {
        let r = device.read64(address);
        (r.status == BUS_OK).then_some(r.data)
    }
    fn cached(rig: &Rig) -> Arc<dyn BusDevice> {
        rig.chip.cpu_view(Space::Cached)
    }
    fn view(rig: &Rig, space: Space) -> Arc<dyn BusDevice> {
        rig.chip.cpu_view(space)
    }

    /// From reset both chips carry 0, as the PROM's first store leaves them:
    /// each reaches only itself at node field zero, and nothing answers at
    /// node 1's placement.
    #[test]
    fn equal_numbers_stay_local() {
        let (a, b) = pair();
        assert_eq!(read(&cached(&a), PROBE), Some(WORD0));
        assert_eq!(read(&cached(&b), PROBE), Some(WORD1));
        assert_eq!(read(&cached(&a), node_at(1) + PROBE), None);
        assert_eq!(read(&cached(&b), node_at(1) + PROBE), None);
        assert_eq!(read(&view(&a, Space::Memory), node_at(1) + PROBE), None);
    }

    /// Once node 1 carries 1, node field 1 names it from either node, in the
    /// cached view, the uncached memory view and the I/O register forms.
    #[test]
    fn numbered_nodes_reach_each_other() {
        let (a, b) = pair();
        number(&b, 1);
        assert_eq!(read(&cached(&a), PROBE), Some(WORD0));
        assert_eq!(read(&cached(&a), node_at(1) + PROBE), Some(WORD1));
        assert_eq!(read(&cached(&b), node_at(1) + PROBE), Some(WORD1), "a node reaches itself by its own number");
        assert_eq!(read(&view(&a, Space::Memory), node_at(1) + PROBE), Some(WORD1));
        let io = view(&a, Space::Io);
        assert_eq!(read(&io, node_at(1) + NODE_NAMED + SCRATCH2), Some(0x5c1));
        assert_eq!(read(&io, node_at(0) + NODE_NAMED + SCRATCH2), Some(0x5c0));
    }

    /// The claim follows the number: node 1 renumbered to 5 answers only at 5.
    #[test]
    fn renumbering_moves_the_claim() {
        let (a, b) = pair();
        number(&b, 1);
        number(&b, 5);
        assert_eq!(read(&cached(&a), node_at(5) + PROBE), Some(WORD1));
        assert_eq!(read(&cached(&a), node_at(1) + PROBE), None);
        assert_eq!(read(&cached(&b), node_at(5) + PROBE), Some(WORD1));
        assert_eq!(read(&cached(&b), node_at(1) + PROBE), None);
    }

    /// Two nodes carrying one number each reach only themselves there.
    #[test]
    fn a_shared_number_reaches_only_the_reader() {
        let (a, b) = pair();
        number(&a, 3);
        number(&b, 3);
        assert_eq!(read(&cached(&a), node_at(3) + PROBE), Some(WORD0));
        assert_eq!(read(&cached(&b), node_at(3) + PROBE), Some(WORD1));
    }

    /// A cable marked down leaves each node only itself, and marking it up
    /// again restores the claims.
    #[test]
    fn cable_down_makes_the_peer_unreachable() {
        let (a, b) = pair();
        number(&b, 1);
        assert_eq!(read(&cached(&a), node_at(1) + PROBE), Some(WORD1));
        a.chip.set_link(false);
        assert_eq!(read(&cached(&a), node_at(1) + PROBE), None);
        assert_eq!(read(&cached(&b), node_at(1) + PROBE), Some(WORD1));
        assert_eq!(read(&cached(&a), PROBE), Some(WORD0), "the reader's own memory is unaffected");
        b.chip.set_link(true);
        assert_eq!(read(&cached(&a), node_at(1) + PROBE), Some(WORD1));
    }

    /// A remote store lands in the claimant's memory, in every width, and a
    /// node nobody claims faults rather than stopping the machine.
    #[test]
    fn remote_stores_land_at_the_claimant() {
        let (a, b) = pair();
        number(&b, 1);
        let cache = cached(&a);
        assert_eq!(cache.write64(node_at(1) + 0x2000, 0x1122_3344_5566_7788), BUS_OK);
        assert_eq!(cache.write32(node_at(1) + 0x2008, 0x99aa_bbcc), BUS_OK);
        assert_eq!(cache.write8(node_at(1) + 0x200c, 0xdd), BUS_OK);
        assert_eq!(b.chip.dma_view().read64(0x2000).data, 0x1122_3344_5566_7788);
        assert_eq!(b.chip.dma_view().read64(0x2008).data >> 32, 0x99aa_bbcc);
        assert_eq!(a.chip.dma_view().read64(0x2000).data, 0, "the writer's own memory is untouched");
        assert_eq!(cache.write64(node_at(9) + 0x2000, 1), BUS_ERR);
        assert_eq!(cache.write64(1 << 40, 1), BUS_ERR, "beyond 40 bits of address");
    }

    /// A node that carries another number reaches node field zero across the
    /// cable, except the cached alias: the first bytes of its own memory,
    /// sized by the processor interface's cached-alias register.
    #[test]
    fn a_renumbered_node_reaches_node_zero_across_the_cable() {
        let (a, b) = pair();
        assert_eq!(read(&cached(&b), PROBE), Some(WORD1), "both carry 0: each reaches itself");
        number(&b, 1);
        assert_eq!(read(&cached(&b), PROBE), Some(WORD0), "node field zero is node 0's memory now");
        assert_eq!(read(&cached(&b), node_at(1) + PROBE), Some(WORD1), "its own memory is at its number");
        assert_eq!(read(&cached(&a), PROBE), Some(WORD0));
        // Code 4 is 32 KiB, which holds PROBE.
        b.w(pi::CACHED_ALIAS, 4);
        assert_eq!(read(&cached(&b), PROBE), Some(WORD1), "inside the cached alias it stays on the node");
        assert_eq!(read(&cached(&b), 0x8000 + PROBE), Some(0), "past the alias it is node 0's, which holds nothing there");
        assert_eq!(cached(&b).write64(0x9000, 5), BUS_OK);
        assert_eq!(a.chip.dma_view().read64(0x9000).data, 5, "a store lands on node 0");
        assert_eq!(b.chip.dma_view().read64(0x9000).data, 0);
        // The larger of the two interfaces' registers sizes the alias.
        assert_eq!(read(&cached(&b), 0x9000), Some(5), "node 0's word");
        b.w(super::super::reg::PI1_BASE + pi::CACHED_ALIAS, 5);
        assert_eq!(read(&cached(&b), 0x9000), Some(0), "64 KiB of alias: its own, empty word");
    }

    /// A renumbered node reaches node field zero across the cable in the
    /// uncached spaces as well, outside the reader-relative forms: memory past
    /// the cached alias, the memory-special space, the I/O space past its local
    /// register form, and the Hub-special space past its first 512 MiB.
    /// What the cable offers the peer is the node's own decode, so the peer's
    /// references to it by number do not come back.
    #[test]
    fn a_renumbered_node_reaches_node_zero_uncached() {
        let (a, b) = pair();
        let io = |rig: &Rig, address| read(&view(rig, Space::Io), address);
        assert_eq!(read(&view(&b, Space::Memory), PROBE), Some(WORD1), "both carry 0: each reaches itself");
        assert_eq!(io(&b, NODE_NAMED + SCRATCH2), Some(0x5c1));
        number(&b, 1);
        assert_eq!(read(&view(&b, Space::Memory), PROBE), Some(WORD0), "uncached memory is node 0's");
        assert_eq!(read(&view(&b, Space::Fetchop), 0x10), Some(0xf0), "and so is its memory-special space");
        assert_eq!(io(&b, NODE_NAMED + SCRATCH2), Some(0x5c0), "node 0's node-named registers");
        assert_eq!(io(&b, NODE_LOCAL + SCRATCH2), Some(0x5c1), "the local register form stays");
        assert_eq!(io(&b, 0x0300_0000), Some(0x200), "the rest of the I/O space is node 0's");
        let hub = |rig: &Rig, address| read(&view(rig, Space::HubSpecial), address);
        assert_eq!(hub(&b, 0x1fff_fff8), Some(0x101), "the Hub-special space's first 512 MiB stay");
        assert_eq!(hub(&b, 0x2000_0000), Some(0x100), "beyond them it is node 0's");
        assert_eq!(read(&view(&b, Space::Memory), node_at(1) | PROBE), Some(WORD1), "its own memory is at its number");
        b.w(pi::CACHED_ALIAS, 4);
        assert_eq!(read(&view(&b, Space::Memory), PROBE), Some(WORD1), "inside the cached alias it stays on the node");
        assert_eq!(view(&b, Space::Memory).write64(0x9000, 5), BUS_OK);
        assert_eq!(a.chip.dma_view().read64(0x9000).data, 5, "a store past the alias lands on node 0");
        assert_eq!(b.chip.dma_view().read64(0x9000).data, 0);
        // Node 0 reaches node 1 by number, and its own decode at node field zero.
        assert_eq!(read(&view(&a, Space::Memory), node_at(1) | PROBE), Some(WORD1));
        assert_eq!(read(&view(&a, Space::Memory), PROBE), Some(WORD0));
        assert_eq!(io(&a, node_at(1) + NODE_NAMED + SCRATCH2), Some(0x5c1));
        // The memory-special space has no local form, whatever the alias.
        assert_eq!(read(&view(&b, Space::Fetchop), 0x10), Some(0xf0));
        // Nobody holding 0 leaves the local forms with their reader.
        let (c, d) = pair();
        number(&c, 2);
        number(&d, 1);
        assert_eq!(read(&cached(&c), PROBE), Some(WORD0));
        assert_eq!(read(&cached(&d), PROBE), Some(WORD1));
        assert_eq!(read(&claim(&c, Space::Cached), PROBE), None, "by number alone nobody holds 0");
    }
    fn claim(rig: &Rig, space: Space) -> Arc<dyn BusDevice> {
        rig.chip.claim(space)
    }

    /// The reset PROM's cached window is a form of the reader's own.
    #[test]
    fn the_cached_reset_window_stays_on_the_reader() {
        let (_a, b) = pair();
        number(&b, 1);
        let routes = &b.chip.routes;
        assert!(routes.stays(Space::Cached, crate::ip35::RESET_WINDOW));
        assert!(routes.stays(Space::Cached, crate::ip35::RESET_WINDOW + crate::ip35::RESET_WINDOW_SIZE - 8));
        assert!(!routes.stays(Space::Cached, crate::ip35::RESET_WINDOW + crate::ip35::RESET_WINDOW_SIZE));
        assert!(!routes.stays(Space::Memory, crate::ip35::RESET_WINDOW), "uncached memory has none");
    }

    /// A packet for the other node's hub, an interrupt for a processor there,
    /// is delivered to it by node number rather than refused.
    #[test]
    fn an_interrupt_packet_reaches_the_other_nodes_hub() {
        let (a, b) = pair();
        number(&b, 1);
        let hub = |n: u64, register: u64| (1u64 << 47) | (n << 39) | (0x180_0000 + register);
        assert_eq!(a.chip.inbound_write(15, Some(0), hub(1, 0x90), 8, 0x108), BUS_OK);
        assert_eq!(b.r(pi::PENDING_STATUS_A), 1 << 8, "node 1's processor interface took it");
        assert_eq!(a.r(pi::PENDING_STATUS_A), 0);
        assert_eq!(a.chip.inbound_write(15, Some(0), hub(1, 0x20_0090), 8, 0x109), BUS_OK, "its second interface too");
        assert_eq!(b.r(super::super::reg::PI1_BASE + pi::PENDING_STATUS_A), 1 << 9);
        assert_eq!(a.chip.inbound_write(15, Some(0), hub(0, 0x90), 8, 0x10a), BUS_OK, "its own number is still its own");
        assert_eq!(a.r(pi::PENDING_STATUS_A), 1 << 10);
        assert_eq!(a.chip.inbound_write(15, Some(0), hub(9, 0x90), 8, 0x108), BUS_ERR, "nobody carries 9");
        a.chip.set_link(false);
        assert_eq!(a.chip.inbound_write(15, Some(0), hub(1, 0x90), 8, 0x10b), BUS_ERR, "a downed link carries nothing");
    }

    /// Inbound DMA names memory by node number, node field zero included and
    /// with no cached alias: a widget on node 0 reaches node 1's memory by its
    /// number, and once node 1 carries 1 its own widgets reach node 0's memory
    /// at node field zero.
    #[test]
    fn inbound_dma_reaches_memory_by_node_number() {
        let (a, b) = pair();
        let dma = |rig: &Rig, address: u64| {
            let r = rig.chip.inbound_read(15, Some(0), address, 8);
            (r.status == BUS_OK).then_some(r.data)
        };
        assert_eq!(dma(&b, PROBE), Some(WORD1), "both carry 0: each reaches itself");
        number(&b, 1);
        b.w(pi::CACHED_ALIAS, 4);
        assert_eq!(dma(&a, PROBE), Some(WORD0));
        assert_eq!(dma(&a, node_at(1) + PROBE), Some(WORD1), "node 1 by its number");
        assert_eq!(dma(&b, node_at(1) + PROBE), Some(WORD1), "a node by its own number");
        assert_eq!(dma(&b, PROBE), Some(WORD0), "node field zero is node 0's, alias or not");
        assert_eq!(a.chip.inbound_write(15, Some(0), node_at(1) + 0x9000, 8, 6), BUS_OK);
        assert_eq!(b.chip.dma_view().read64(0x9000).data, 6, "a DMA store lands on node 1");
        assert_eq!(dma(&a, node_at(9) + PROBE), None, "nobody carries 9");
        a.chip.set_link(false);
        assert_eq!(dma(&a, node_at(1) + PROBE), None, "a downed link carries nothing");
    }

    /// A chip with no cable keeps its DMA as it was: node-qualified addresses
    /// are not memory, whatever number the node carries.
    #[test]
    fn an_uncabled_chip_does_not_route_dma_by_number() {
        let rig = Rig::new();
        furnish(&rig, 0, WORD0);
        number(&rig, 3);
        assert_eq!(rig.chip.inbound_read(15, Some(0), PROBE, 8).data, WORD0);
        assert_eq!(rig.chip.inbound_read(15, Some(0), node_at(3) + PROBE, 8).status, BUS_ERR);
        let hub = (1u64 << 47) | (3 << 39) | 0x180_0090;
        assert_eq!(rig.chip.inbound_write(15, Some(0), hub, 8, 0x108), BUS_OK);
        assert_eq!(rig.chip.inbound_write(15, Some(0), hub & !(3 << 39), 8, 0x108), BUS_ERR);
    }

    /// A store-conditional to another node's memory is the claimant's
    /// compare-and-swap: it wins once, and loses when the word has moved on.
    #[test]
    fn remote_compare_and_swap_acts_on_the_claimant() {
        let (a, b) = pair();
        number(&b, 1);
        let cache = cached(&a);
        let at = node_at(1) + 0x3000;
        assert_eq!(cache.write64(at, 0x1111_2222_3333_4444), BUS_OK);
        assert_eq!(cache.cas64(at, 0x1111_2222_3333_4444, 7), Some(true));
        assert_eq!(cache.cas64(at, 0x1111_2222_3333_4444, 9), Some(false));
        assert_eq!(b.chip.dma_view().read64(0x3000).data, 7, "the claimant's memory holds the swap");
        assert_eq!(cache.write32(at + 8, 5), BUS_OK);
        assert_eq!(cache.cas32(at + 8, 5, 6), Some(true));
        assert_eq!(cache.cas32(at + 8, 5, 8), Some(false));
        assert_eq!(cache.cas32(node_at(9) + 0x3000, 0, 1), None, "an unclaimed node offers no swap");
    }

    /// Run a block-transfer engine on `rig`: `lines` 128-byte lines from
    /// `source` to `destination`, completing by an interrupt to node `node`'s
    /// interface `pi` at level `level`.
    fn transfer(rig: &Rig, lines: u64, source: u64, destination: u64, node: u64, pi: u64, level: u64) {
        let base = IO_BASE + io::BTE1_BASE;
        rig.w(base + bte::INTERRUPT, (level << 16) | (node << 1) | pi);
        rig.w(base + bte::STATUS, lines | super::super::io::BUSY);
        rig.w(base + bte::SOURCE, source);
        rig.w(base + bte::DESTINATION, destination);
        rig.w(base + bte::CONTROL, 0);
    }

    /// The engines copy by node number, and their completion interrupt names
    /// a node: naming the other node reaches that node's interface and not the
    /// issuer's.
    #[test]
    fn a_block_transfer_completion_reaches_the_node_it_names() {
        let (a, b) = pair();
        number(&b, 1);
        for i in 0..16 {
            assert_eq!(b.chip.dma_view().write64(0x4000 + i * 8, 0xfeed_0000 | i), BUS_OK);
        }
        // Node 0's engine copies node 1's line to its own memory and tells node 1.
        transfer(&a, 1, node_at(1) + 0x4000, 0x6000, 1, 0, 53);
        for i in 0..16 {
            assert_eq!(a.chip.dma_view().read64(0x6000 + i * 8).data, 0xfeed_0000 | i);
        }
        assert_eq!(a.r(IO_BASE + io::BTE1_BASE + bte::STATUS), 0);
        assert_eq!(b.r(pi::PENDING_STATUS_A), 1 << 53, "the completion reached node 1");
        assert_eq!(a.r(pi::PENDING_STATUS_A), 0, "and not the issuer");
        // Node 1's engine writes node 0's memory by node field zero and tells node 0's second interface.
        transfer(&b, 1, node_at(1) + 0x4000, 0x7000, 0, 1, 40);
        assert_eq!(a.chip.dma_view().read64(0x7008).data, 0xfeed_0001, "node field zero is node 0 for an engine");
        assert_eq!(a.r(super::super::reg::PI1_BASE + pi::PENDING_STATUS_A), 1 << 40);
        assert_eq!(b.r(super::super::reg::PI1_BASE + pi::PENDING_STATUS_A), 0);
        // A completion naming nobody reaches nobody.
        transfer(&a, 1, 0x4000, 0x6000, 9, 0, 33);
        assert_eq!((a.r(pi::PENDING_STATUS_A), b.r(pi::PENDING_STATUS_A)), (0, 1 << 53));
        // The engines' notification write is by number too.
        let base = IO_BASE + io::BTE1_BASE;
        assert_eq!(b.chip.dma_view().write64(0x5000, 0xdead), BUS_OK);
        rig_notify(&a, base, node_at(1) + 0x5000);
        assert_eq!(b.chip.dma_view().read64(0x5000).data, 0, "the final status of a good transfer");
    }
    fn rig_notify(rig: &Rig, base: u64, notify: u64) {
        assert_eq!(rig.chip.dma_view().write64(0x4000, 1), BUS_OK);
        rig.w(base + bte::NOTIFICATION, notify);
        rig.w(base + bte::STATUS, 1 | super::super::io::BUSY);
        rig.w(base + bte::SOURCE, 0x4000);
        rig.w(base + bte::DESTINATION, 0x4000);
        rig.w(base + bte::CONTROL, 0x10);
    }

    /// A node's own claim never goes away while its peer is renumbered, and two
    /// threads renumbering the two nodes at once leave a table that matches
    /// the final numbers.
    #[test]
    fn renumbering_never_hides_a_claim_from_a_reader() {
        use std::sync::atomic::AtomicBool;
        let (a, b) = pair();
        number(&a, 3);
        number(&b, 1);
        let (a, b) = (Arc::new(a), Arc::new(b));
        let stop = Arc::new(AtomicBool::new(false));
        let reader = {
            let (stop, cache) = (stop.clone(), cached(&a));
            std::thread::spawn(move || {
                let mut misses = 0u32;
                while !stop.load(Relaxed) {
                    if cache.read64(node_at(3) + PROBE).status != BUS_OK {
                        misses += 1;
                    }
                }
                misses
            })
        };
        let writer = {
            let a = a.clone();
            std::thread::spawn(move || {
                for _ in 0..2000u64 {
                    number(&a, 3);
                }
            })
        };
        for round in 0..2000u64 {
            number(&b, 1 + (round & 1));
        }
        writer.join().unwrap();
        stop.store(true, Relaxed);
        assert_eq!(reader.join().unwrap(), 0, "a node lost sight of its own memory");
        number(&b, 5);
        assert_eq!(read(&cached(&a), node_at(5) + PROBE), Some(WORD1));
        assert_eq!(read(&cached(&a), node_at(3) + PROBE), Some(WORD0));
        assert_eq!(read(&cached(&a), node_at(1) + PROBE), None);
        assert_eq!(read(&cached(&b), node_at(5) + PROBE), Some(WORD1));
        assert_eq!(read(&cached(&b), node_at(3) + PROBE), Some(WORD0));
    }

    /// A node's reset returns its number to 0 and its claims with it.
    #[test]
    fn a_reset_returns_the_claims_to_zero() {
        let (a, b) = pair();
        number(&b, 1);
        assert_eq!(read(&cached(&a), node_at(1) + PROBE), Some(WORD1));
        b.w(LB_BASE + lb::SOFT_RESET, 1);
        assert_eq!(read(&cached(&a), node_at(1) + PROBE), None);
        assert_eq!(read(&cached(&b), PROBE), Some(WORD1));
        number(&b, 1);
        b.chip.power_on();
        assert_eq!(read(&cached(&b), node_at(1) + PROBE), None);
    }

    /// The chips, the devices registered for their spaces and the views a
    /// board maps into its processors' buses hold nothing in a ring: dropping
    /// the chips frees them, and dropping the views then frees the devices
    /// each chip showed the other.
    #[test]
    fn dropping_the_chips_frees_both_and_their_buses() {
        let (a, b) = cabled_pair();
        let spaces: Vec<Arc<Probe>> = [Probe(0xa), Probe(0xb)].into_iter().map(Arc::new).collect();
        for (rig, probe) in [&a, &b].into_iter().zip(&spaces) {
            rig.chip.set_space(Space::Cached, probe.clone());
        }
        number(&b, 1);
        // What a board maps into each node's processor bus.
        let buses = [cached(&a), cached(&b), a.chip.claim(Space::Cached), b.chip.cpu_view(Space::Memory)];
        assert_eq!(read(&buses[0], node_at(1)), Some(0xb));
        assert_eq!(read(&buses[1], node_at(0)), Some(0xa));
        let chips: Vec<Weak<Bedrock>> = [&a, &b].into_iter().map(|rig| Arc::downgrade(&rig.chip)).collect();
        let probes: Vec<Weak<Probe>> = spaces.iter().map(Arc::downgrade).collect();
        let bus_refs: Vec<Weak<dyn BusDevice>> = buses.iter().map(Arc::downgrade).collect();
        drop((a, b, spaces));
        assert!(chips.iter().all(|chip| chip.upgrade().is_none()), "a view or a cable kept a chip alive");
        assert!(probes.iter().all(|probe| probe.upgrade().is_some()), "the views still show the devices");
        assert_eq!(read(&buses[0], node_at(1)), Some(0xb), "a view outlives its chip");
        drop(buses);
        assert!(bus_refs.iter().all(|bus| bus.upgrade().is_none()));
        assert!(probes.iter().all(|probe| probe.upgrade().is_none()), "a device is held by a ring");
    }

    /// Two chips with 256 MiB and room for two node numbers, cabled, and the
    /// windows that show their memory.
    fn windowed_pair() -> (Vec<Arc<Bedrock>>, Arc<crate::node_windows::Windows>) {
        use super::super::BedrockConfig;
        let chips: Vec<_> = (0..2)
            .map(|_| Bedrock::new(BedrockConfig { banks_mib: &[256], nodes: 2, ..super::super::test_config(Arc::new(|| 0)) }).unwrap())
            .collect();
        Bedrock::cable(&chips[0], &chips[1]);
        let windows = Arc::new(crate::node_windows::Windows::new(crate::node_windows::SN1_NODE_SHIFT));
        for chip in &chips {
            windows.add_node(chip.physical(), chip.shape());
            chip.set_windows(&windows);
            assert_eq!(chip.registers().write64(MD_BASE + md::CONFIG, 0x31), BUS_OK);
        }
        (chips, windows)
    }
    /// The word the processor cache would read at `at` in chip `chip`'s
    /// window, if the window holds its region whole.
    fn seen(chip: &Bedrock, at: u64) -> Option<u64> {
        use crate::ppmem::MappedMemory;
        let sp = chip.physical().ppmem_space().unwrap();
        crate::ppmem::bitmap_maps_at(sp.mapped_bitmap(), sp.region_shift(), at)
            .then(|| unsafe { (*(sp.window_base().add(at as usize) as *const u64)).rotate_left(32) })
    }

    /// Each node's window follows the number LB's region write gives and the
    /// memory configuration register: the second node's memory appears at its
    /// number in both windows, node zero's in the second's above the alias,
    /// and a bank-select swap on the second node moves what both show.
    #[test]
    fn windows_follow_a_pairs_numbers_and_memory_configuration() {
        let (chips, _windows) = windowed_pair();
        for (n, chip) in chips.iter().enumerate() {
            assert_eq!(chip.dma_view().write64(0x1000, 0xa0 + n as u64), BUS_OK);
        }
        let at = |n: u64, off: u64| (n << NODE_SHIFT) + off;
        // Both hold zero: each window is its own memory and nothing else.
        assert_eq!((seen(&chips[0], 0x1000), seen(&chips[1], 0x1000)), (Some(0xa0), Some(0xa1)));
        assert_eq!((seen(&chips[0], at(1, 0x1000)), seen(&chips[1], at(1, 0x1000))), (None, None));
        // The second is numbered 1.
        assert_eq!(chips[1].registers().write64(LB_BASE + lb::NODE_REGION, 1 << 32), BUS_OK);
        assert_eq!(seen(&chips[0], at(1, 0x1000)), Some(0xa1));
        assert_eq!(seen(&chips[1], at(1, 0x1000)), Some(0xa1), "a node's memory is at its own number");
        assert_eq!(seen(&chips[1], 0x1000), Some(0xa0), "node field zero is node 0's, beyond the alias");
        assert_eq!(seen(&chips[0], 0x1000), Some(0xa0));
        // A swap on the second node moves its bank to the next slot, in both windows.
        assert_eq!(chips[1].registers().write64(MD_BASE + md::CONFIG, 0x31 | (1 << 32)), BUS_OK);
        let slot1 = 1u64 << 30;
        assert_eq!(chips[1].dma_view().write64(slot1 + 0x2000, 0xb1), BUS_OK);
        for chip in &chips {
            assert_eq!(seen(chip, at(1, slot1 + 0x2000)), Some(0xb1));
            assert_eq!(seen(chip, at(1, 0x1000)), None, "the bank left slot 0");
        }
        // Node 0's own window survives the second node's remap.
        assert_eq!(seen(&chips[0], 0x1000), Some(0xa0));
        // A reset returns the number to 0 and the windows to each node's own memory.
        assert_eq!(chips[1].registers().write64(LB_BASE + lb::SOFT_RESET, 1), BUS_OK);
        assert_eq!(seen(&chips[0], at(1, slot1 + 0x2000)), None);
    }

    /// A pointer or counter the owning node hands out is passed through a
    /// routed view, wherever the memory is reached from.
    #[test]
    fn a_routed_view_passes_the_owners_pointers() {
        struct Owner(u64);
        impl BusDevice for Owner {
            fn mem_ptr(&self, a: u64) -> Option<*const u64> {
                Some((self.0 + a) as *const u64)
            }
        }
        let (a, b) = cabled_pair();
        b.chip.set_space(Space::Cached, Arc::new(Owner(0x1_0000)));
        a.chip.set_space(Space::Cached, Arc::new(Owner(0x2_0000)));
        number(&b, 1);
        assert_eq!(cached(&a).mem_ptr(node_at(1) + 8), Some(0x1_0008 as *const u64));
        assert_eq!(cached(&a).mem_ptr(8), Some(0x2_0008 as *const u64));
        assert_eq!(cached(&a).mem_ptr(node_at(9)), None);
    }
}
