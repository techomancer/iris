//! The ppmem windows of the SN machines' nodes: what each node's processor
//! cache and the JIT read without the bus.
//!
//! Every node's window shows its own memory and the other nodes' at their node
//! numbers. A node number sits at a platform's node shift in the physical
//! address, 32 on the SN0 (Origin 200, Origin 2000, Onyx2) and 33 on the SN1
//! (Origin 3000), so [`Windows`] takes the shift and lays the windows alike
//! for both.
//!
//! A node tells its window what to show through [`NodeShape`]: the SN0 Hub
//! and the SN1 Bedrock publish their number, cached alias and memory
//! differently, and the layout code reads only that.
//!
//! Ported from hw/mips/sgi/sn0.c in jamesbraid/qemu-sgi-origami.

use crate::dev::hub::Routing;
use crate::devlog::LogModule;
use crate::physical::Physical;
use crate::traits::*;
use parking_lot::Mutex;
#[cfg(feature = "jitv2")]
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};

/// The node number's bit in an SN0 physical address.
pub const SN0_NODE_SHIFT: u32 = 32;
/// The node number's bit in an SN1 physical address.
pub const SN1_NODE_SHIFT: u32 = 33;

/// The view a node's Hub sends other node numbers to, held weakly: it reaches
/// the Hub, which holds this memory's users.
pub type Claim = Arc<Mutex<Option<Weak<dyn BusDevice>>>>;

/// One fitted run of node memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    /// The bank whose stride the run lies in.
    pub bank: usize,
    /// The run's RAM bank among the fitted ones, in the order they were built.
    pub index: usize,
    /// Where the run lies in the node's offsets.
    pub at: u64,
    pub len: u64,
    /// Where in its RAM bank the run starts.
    pub src: u64,
}


/// What a node publishes for its window to follow. A window is laid from the
/// numbers and memory its nodes publish and asks nothing else of them.
pub trait NodeShape: Send + Sync {
    /// The node number the node answers to.
    fn number(&self) -> u16;
    /// Bytes the cached alias spans: node zero's memory below it stays on
    /// this node once the node holds another number.
    fn calias(&self) -> u64;
    /// Whether the node routes other node numbers to the nodes that hold
    /// them, which a board with a cable does.
    fn routed(&self) -> bool;
    /// The runs of memory the node holds now. A Hub's never change; a
    /// Bedrock's follow its memory configuration register.
    fn runs(&self) -> Arc<[Run]>;
}

/// The SN0 Hub's shape: a fixed layout, and the number and alias its
/// [`Routing`] publishes.
struct HubShape {
    routing: Arc<Routing>,
    layout: Arc<[Run]>,
    claim: Claim,
}

impl NodeShape for HubShape {
    fn number(&self) -> u16 {
        self.routing.number()
    }
    fn calias(&self) -> u64 {
        self.routing.calias()
    }
    fn routed(&self) -> bool {
        self.claim.lock().as_ref().and_then(Weak::upgrade).is_some()
    }
    fn runs(&self) -> Arc<[Run]> {
        self.layout.clone()
    }
}

/// What every node's ppmem window shows, which is what its processor's cache
/// and the JIT read without the bus. It follows the decode table exactly, as
/// the Hubs route it: node zero's window is the node's own memory while it
/// holds zero, its own memory below the cached alias and the memory of the
/// node holding zero above it once it holds another number, and on a board
/// with a cable each node's memory sits at its number. So a processor reaches
/// both its own memory and the other node's at the cost of a load, whichever
/// node it is on.
///
/// A number or alias change lays only the difference, and one host mapping
/// call moves a page that stays mapped from one node's memory to the other's.
/// A range that goes leaves the region bitmap before its pages do. The other
/// processor can be running meanwhile: a renumbering never clears the window
/// it is executing from.
///
/// A store to any Hub's number or alias, from whichever processor made it,
/// publishes the numbers ([`Routing`]) and lays every node's window at once
/// from that thread, as the hardware takes the new number at once. The Hub
/// parks the other processors first ([`crate::dev::hub::Hub::set_world`]), so
/// no access sees a window half laid. [`crate::ppmem::PpMemSpace`] still
/// serializes the layout against a CACHE instruction with its own lock, taken
/// by a remap or a held-line change and never by an access that hits.
/// Windows' lock orders two renumberings; the space's lock is taken inside it,
/// never the other way round.
pub struct Windows {
    /// The bit the node number sits at in a physical address.
    shift: u32,
    nodes: Mutex<Vec<Shown>>,
}

struct Shown {
    phys: Weak<Physical>,
    shape: Arc<dyn NodeShape>,
    /// What the window holds now.
    laid: Vec<Piece>,
}

/// A stretch of one node's bank laid in a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Piece {
    /// The node whose memory it is, and the RAM bank among its fitted ones.
    owner: usize,
    index: usize,
    /// Where in the bank, and where in the window.
    src: u64,
    at: u64,
    len: u64,
}

impl Default for Windows {
    /// An SN0's windows.
    fn default() -> Self {
        Self::new(SN0_NODE_SHIFT)
    }
}

impl Windows {
    /// Windows for nodes whose number sits at address bit `shift`
    /// ([`SN0_NODE_SHIFT`] or [`SN1_NODE_SHIFT`]).
    pub fn new(shift: u32) -> Self {
        Self { shift, nodes: Mutex::default() }
    }

    /// Add a node whose memory is `phys`, laid out as `layout`, to the board.
    /// `claim` is the view [`crate::ip27::NodeMemory::route_nodes`] was given.
    pub(crate) fn add(&self, phys: &Arc<Physical>, layout: &Arc<[Run]>, claim: &Claim, routing: Arc<Routing>) {
        self.add_node(phys, Arc::new(HubShape { routing, layout: layout.clone(), claim: claim.clone() }));
    }

    /// Add a node whose memory is `phys`, as `shape` publishes it. The node's
    /// window must hold its own runs at their offsets, as a node's memory
    /// map lays them, and the windows [`Self::refresh`] lays follow from there.
    pub fn add_node(&self, phys: &Arc<Physical>, shape: Arc<dyn NodeShape>) {
        let mut nodes = self.nodes.lock();
        // What the node's window starts with: its own runs at their offsets.
        let laid = match phys.ppmem_space() {
            Some(sp) => shape
                .runs()
                .iter()
                .filter(|r| r.at + r.len <= sp.window_size())
                .map(|r| Piece { owner: nodes.len(), index: r.index, src: r.src, at: r.at, len: r.len })
                .collect(),
            None => Vec::new(),
        };
        nodes.push(Shown { phys: Arc::downgrade(phys), shape, laid });
    }

    /// The node whose memory is `phys` has remapped its window from scratch,
    /// which leaves nothing of what was laid: forget it, so the next
    /// [`Self::refresh`] lays the node's window whole.
    pub fn forget(&self, phys: &Arc<Physical>) {
        let mut nodes = self.nodes.lock();
        for node in nodes.iter_mut().filter(|n| std::ptr::eq(n.phys.as_ptr(), Arc::as_ptr(phys))) {
            node.laid.clear();
        }
    }

    /// The numbers or aliases a Hub publishes moved: one node's number moves
    /// its memory in the other's window too. Every window is laid now, from
    /// the calling thread, to the numbers as they stand under the lock; of two
    /// stores that race, the later refresh to take it lays the later values.
    pub(crate) fn refresh(&self) {
        let mut nodes = self.nodes.lock();
        let mut moved = Vec::new();
        for i in 0..nodes.len() {
            moved.extend(Self::lay(&mut nodes, i, self.shift));
        }
        #[cfg(feature = "jitv2")]
        Self::moved_under_the_jit(&nodes, &moved);
    }

    /// Lay node `i`'s window to the numbers and aliases the Hubs publish now,
    /// and return the pieces it mapped.
    fn lay(nodes: &mut [Shown], i: usize, shift: u32) -> Vec<Piece> {
        Self::lay_by(nodes, i, shift, &mut |sp, bank, p| sp.map_range_of(bank, p.src, p.at, p.len))
    }

    /// [`Self::lay`] with the host mapping call given, so a test can fail it.
    fn lay_by(
        nodes: &mut [Shown],
        i: usize,
        shift: u32,
        map: &mut dyn FnMut(&crate::ppmem::PpMemSpace, &crate::physical::RamBank, &Piece) -> std::io::Result<()>,
    ) -> Vec<Piece> {
        let numbers: Vec<_> = nodes.iter().map(|n| (n.shape.number(), n.shape.calias())).collect();
        let runs: Vec<_> = nodes.iter().map(|n| n.shape.runs()).collect();
        let phys: Vec<_> = nodes.iter().map(|n| n.phys.upgrade()).collect();
        let Some(sp) = phys[i].as_ref().and_then(|p| p.ppmem_space()) else { return Vec::new() };
        let want = Self::wanted(nodes, &runs, &numbers, i, sp, shift);
        // A piece the host could not map is not recorded as laid, so the
        // next call tries it again. What was mapped there before it is no
        // longer what the Hubs route, and nothing tracks it now: it comes out
        // of the window, and the processor reaches the span over the bus.
        let mut failed = Vec::new();
        let mut mapped = Vec::new();
        for p in want.iter().filter(|p| !nodes[i].laid.contains(p)) {
            let Some(owner) = &phys[p.owner] else { continue };
            match map(sp, &owner.banks()[p.index], p) {
                Ok(()) => mapped.push(*p),
                Err(e) => {
                    crate::dlog!(LogModule::Hub, "Hub: node {i} window at {:#x}: {e}", p.at);
                    sp.unmap_range(p.at, p.len);
                    failed.push(*p);
                }
            }
        }
        for old in &nodes[i].laid {
            for (at, len) in uncovered(old.at, old.len, &want) {
                sp.unmap_range(at, len);
            }
        }
        nodes[i].laid = want.into_iter().filter(|p| !failed.contains(p)).collect();
        mapped
    }

    /// The JIT keeps the generation counter a window page had when it claimed
    /// it, and runs the code it compiled there while the counter holds that
    /// value. A page now laid over other memory reads that memory's counter,
    /// so equal counters would leave code compiled from the old memory
    /// standing. Raise the counters of the pages in `moved` instead.
    ///
    /// A counter is `holds * GEN_HELD + writes`: a write adds one, each cache
    /// that takes lines in the page adds `GEN_HELD`, and letting go adds
    /// `1 - GEN_HELD`. The low 48 bits, the write count, only grow, until a
    /// page takes 2^48 of them and carries into the holds. The raise leaves
    /// the holds alone and lifts the write count to a floor one past the
    /// highest write count of any page of any bank, read once every piece is
    /// mapped. The counters are the banks' own, so the raise reads as a
    /// write to the owner node too, and its JIT compiles its code on those
    /// pages once more.
    ///
    /// The invariant: from the raise on, no value a laid counter takes is a
    /// value a JIT page claimed from a bank's counter before the layout. A
    /// page claims whatever counter is mapped at it when it reads. The host
    /// mapping call put the laid counter there before the floor was read, and
    /// no read of the old counter outlasts that call, so the write count the
    /// page claimed is at most the old counter's now, below the floor. The
    /// laid counter's write count is at least the floor and only grows,
    /// whatever holds come and go. Two values with different write counts
    /// are different values.
    ///
    /// The floor covers bank counters only. A span the window does not map
    /// has a counter of the window's own, zeroed when the span was unmapped
    /// or, for a cached view's private zeros, by `scrub_region_gaps`, and
    /// the scan reads neither. A value a page claimed from one of those is
    /// not bounded by the floor.
    ///
    /// When some page's write count is already 2^48 - 1 there is no floor
    /// below the holds. The laid counters take a hold of their own instead,
    /// which nothing takes off: they read as held from then on, so the JIT
    /// never compiles them again and never runs stale code there either. A
    /// page whose own writes carry into its holds stays held for good the
    /// same way, but its write count starts again from zero, under values it
    /// claimed, and a later floor read from it is too low. The invariant
    /// holds while no bank's page has taken 2^48 writes.
    ///
    /// The raise is a compare and exchange, so writes and holds that race it
    /// are kept, and they only add to the write count. Layouts run one at a
    /// time under the lock `refresh` holds. The Hub parks the other
    /// processors around a layout (`Hub::apply`), and that is what keeps
    /// them from running code between the mapping call and the raise, while
    /// the old value still stands; the Bedrock lays its windows without
    /// parking, and leaves that interval open.
    #[cfg(feature = "jitv2")]
    fn moved_under_the_jit(nodes: &[Shown], moved: &[Piece]) {
        use crate::cpu::jitv2::jitv2::GEN_HELD;
        if moved.is_empty() {
            return;
        }
        let floor = highest_write_count(nodes) + 1;
        let raise = |c: u64| {
            if floor >= GEN_HELD {
                c.wrapping_add(GEN_HELD)
            } else {
                c & !(GEN_HELD - 1) | (c & (GEN_HELD - 1)).max(floor)
            }
        };
        let page = crate::cpu::jitv2::PAGE_SIZE as usize;
        let phys: Vec<_> = nodes.iter().map(|n| n.phys.upgrade()).collect();
        // A bank shows in more than one window, so two pieces can hold the
        // same page, and past the top of the count each raise adds a hold.
        let mut pages: Vec<(usize, usize, u64)> = Vec::new();
        for p in moved {
            let Some(owner) = &phys[p.owner] else { continue };
            debug_assert!(p.src + p.len <= owner.banks()[p.index].size() as u64, "a piece lies inside its bank");
            pages.extend((p.src..p.src + p.len).step_by(page).map(|at| (p.owner, p.index, at)));
        }
        pages.sort_unstable();
        pages.dedup();
        for (owner, index, at) in pages {
            let Some(owner) = &phys[owner] else { continue };
            // SAFETY: `at` lies inside a piece, which `can_map` accepted as
            // inside the bank, and a bank's counters live as long as the bank,
            // which `owner` holds. The counter is only accessed atomically.
            // `gen_ptr` applies the bank's address mask while the window maps
            // the counters at the raw `src`, so the two must agree: they do
            // while the mask covers the whole bank, which is how SN0 banks are
            // built, and the assert above bounds the piece by the bank.
            let gen = unsafe { &*owner.banks()[index].gen_ptr(at) };
            // A write that lands meanwhile fails the exchange, and the raise
            // is worked out again from the counter it left.
            let mut c = gen.load(Ordering::Relaxed);
            while raise(c) != c {
                match gen.compare_exchange_weak(c, raise(c), Ordering::Release, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(now) => c = now,
                }
            }
        }
    }

    /// What node `i`'s window should hold, as its Hub routes a cached
    /// reference (`Hub::claim`): its own number first, then the neighbour's.
    fn wanted(nodes: &[Shown], runs: &[Arc<[Run]>], numbers: &[(u16, u64)], i: usize, sp: &crate::ppmem::PpMemSpace, shift: u32) -> Vec<Piece> {
        let (number, calias) = numbers[i];
        let routed = nodes[i].shape.routed();
        let mut want = Vec::new();
        let mut show = |owner: usize, base: u64, from: u64, to: u64| {
            for r in runs[owner].iter() {
                let (lo, hi) = (r.at.max(from), (r.at + r.len).min(to));
                let piece = Piece { owner, index: r.index, src: r.src + (lo.wrapping_sub(r.at)), at: base + lo, len: hi.wrapping_sub(lo) };
                if lo < hi && sp.can_map(piece.src, piece.at, piece.len) {
                    want.push(piece);
                }
            }
        };
        if !routed || number == 0 {
            show(i, 0, 0, u64::MAX);
        } else {
            show(i, 0, 0, calias);
            if let Some(zero) = (0..nodes.len()).find(|&k| k != i && numbers[k].0 == 0) {
                show(zero, 0, calias, u64::MAX);
            }
        }
        if routed {
            for (k, &(n, _)) in numbers.iter().enumerate() {
                if n != 0 && (k == i || n != number) {
                    show(k, u64::from(n) << shift, 0, u64::MAX);
                }
            }
        }
        want
    }
}

/// The highest write count, the low 48 bits of a generation counter, of any
/// page of any node's banks.
#[cfg(feature = "jitv2")]
fn highest_write_count(nodes: &[Shown]) -> u64 {
    let writes = crate::cpu::jitv2::jitv2::GEN_HELD - 1;
    let page = crate::cpu::jitv2::PAGE_SIZE as u64;
    let mut high = 0;
    for phys in nodes.iter().filter_map(|n| n.phys.upgrade()) {
        for bank in phys.banks() {
            for at in (0..bank.size() as u64).step_by(page as usize) {
                // SAFETY: `at` is inside the bank, whose counters live as long
                // as the bank; the counter is only read atomically.
                high = high.max(unsafe { (*bank.gen_ptr(at)).load(Ordering::Acquire) } & writes);
            }
        }
    }
    high
}

/// The parts of `[at, at + len)` that no piece of `want` covers.
fn uncovered(at: u64, len: u64, want: &[Piece]) -> Vec<(u64, u64)> {
    let mut gaps = vec![(at, at + len)];
    for p in want {
        let (pa, pe) = (p.at, p.at + p.len);
        gaps = gaps
            .into_iter()
            .flat_map(|(a, e)| {
                if pe <= a || pa >= e {
                    vec![(a, e)]
                } else {
                    [(a, pa.max(a)), (pe.min(e), e)].into_iter().filter(|(x, y)| x < y).collect()
                }
            })
            .collect();
    }
    gaps.into_iter().map(|(a, e)| (a, e - a)).collect()
}


#[cfg(test)]
mod tests;
