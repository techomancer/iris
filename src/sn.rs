//! The nodes of an IP27 or IP35 machine: one or two, each its own [`Physical`] over its own banks,
//! mapping the other's banks where the node chips route them. The chip remaps its node inside
//! [`World::stop_others`]; the peer's table holds raw pointers, so the machine calls
//! [`Physical::release_devices`] on every node before it lets any go.
//!
//! Ported from hw/mips/sgi/sn0.c and sn1-node.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::config::MachineConfig;
use crate::cpu::mips_exec::CpuDevice;
use crate::cpu::smp::World;
use crate::dev::l1::L1;
use crate::dev::ps2::Ps2Controller;
use crate::dev::serial_host::CiSerialBackend;
use crate::dev::widths::{self, access_widths};
use crate::devlog::{init_devlog, LogModule};
use crate::gdb_stub::CpuDebug;
use crate::gfx_display::GfxDisplay;
use crate::hptimer::TimerManager;
use crate::machine_state::GuestState;
use crate::monitor::{Monitor, DEFAULT_PORT as DEFAULT_MONITOR_PORT};
use crate::net::{GatewayConfig, Link};
use crate::perf_monitor::PerfMonitor;
use crate::scsi::ScsiBus;
use crate::workstation::{register_system_controller, DiscChanger, MachineRef, MachineSlot, Workstation};
use crate::physical::{BankMapping, Physical};
use crate::ppmem::devmap::{Entry, SLOT_SHIFT};
use crate::traits::*;
use crate::window_map::{Window, PA_MASK};
use parking_lot::Mutex;
use crate::net::NatControl;
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::ops::Range;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

/// The processor's reset vector, where both families' boot flash answers.
pub const RESET_WINDOW: u64 = 0x1fc0_0000;

/// The most nodes a machine holds on either family: two, joined by their NI ports.
pub const MAX_NODES: usize = 2;

/// A node number: the node field of a 40-bit physical address whose number
/// starts at a family's node shift.
pub type NodeId = u16;
pub const fn node_mask(shift: u32) -> NodeId {
    (1 << (40 - shift)) - 1
}
/// The node a field names, masked to the address's node field at `shift`.
#[inline(always)]
pub fn node_id(field: u64, shift: u32) -> NodeId {
    field as NodeId & node_mask(shift)
}

/// An interrupt-modify write sets the interrupt its low bits name when this
/// bit is set, and clears it otherwise, on both chips.
pub const MODIFY_SET: u64 = 1 << 8;
/// The node chip's registers in I/O space: the local form answers on the
/// reader's node whatever number it holds; the other names its node.
pub const IO_REGISTERS_LOCAL: u64 = 0x0100_0000;
pub const IO_REGISTERS_NAMED: u64 = 0x0180_0000;
/// Bit 47 of an XIO address selects the node chip's I/O view over memory.
pub const XIO_IO_VIEW: u64 = 1 << 47;
/// A node's I/O space: 16 MiB small windows below 256 MiB (widget n at `n << 24`),
/// then seven 512 MiB big windows, each aimed by its translation entry.
pub const SMALL_WINDOW_SHIFT: u32 = 24;
pub const IO_BIG_WINDOWS: u64 = 0x2000_0000;
pub const BIG_WINDOW_SHIFT: u32 = 29;
/// Where big-window `entry` sends `offset`: the widget and its address, with
/// [`XIO_IO_VIEW`] set when the entry names the I/O view.
pub fn big_window(entry: u64, offset: u64) -> (u8, u64) {
    let at = ((entry & 31) << BIG_WINDOW_SHIFT) | (offset & ((1 << BIG_WINDOW_SHIFT) - 1));
    (((entry >> 8) & 15) as u8, if entry & (1 << 12) != 0 { at | XIO_IO_VIEW } else { at })
}

/// Bytes a cached-alias size code spans: 0 is off, each step doubles from 4 KiB.
pub fn calias_bytes(code: u64) -> u64 {
    if code == 0 { 0 } else { (4 << 10) << (code - 1) }
}

/// The R10000 uncached attribute (TLB EntryLo bits 63:62, physical address
/// bits 58:57), which selects one of four uncached spaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum UncachedSpace {
    /// The reader's own Hub-special resources.
    HubSpecial = 0,
    /// Widgets and the node chip's registers.
    Io = 1,
    /// Fetch-and-op memory.
    MemSpecial = 2,
    /// Node memory.
    Memory = 3,
}

impl UncachedSpace {
    /// The space a processor's uncached address names.
    #[inline(always)]
    pub fn of(a: u64) -> Self {
        match (a >> 57) & 3 {
            0 => Self::HubSpecial,
            1 => Self::Io,
            2 => Self::MemSpecial,
            _ => Self::Memory,
        }
    }
}

/// The processor's uncached bus: one device per attribute (address bits 58:57);
/// compatibility-segment addresses go to Hub special.
pub struct UncachedBus {
    views: [Arc<dyn BusDevice>; 4],
}

impl UncachedBus {
    /// `views` are what attributes 0 to 3 select.
    pub fn new(views: [Arc<dyn BusDevice>; 4]) -> Self {
        Self { views }
    }

    #[inline(always)]
    fn route(&self, a: u64) -> Option<(&dyn BusDevice, u64)> {
        Some((&*self.views[UncachedSpace::of(a) as usize], a & PA_MASK))
    }
}

impl BusDevice for UncachedBus {
    access_widths!(route, forward);
}

/// One node as its own chip and the far node's NI see it. The chip publishes its
/// number, cached alias and banks inside the remap, which maps the table and window.
pub struct Node<C> {
    /// The chip that owns this node, for the far node's interrupts and vector requests.
    chip: Weak<C>,
    /// Shared with the Hub-special space's view of the node's own memory.
    number: Arc<AtomicU16>,
    /// Bytes each processor interface's cached alias spans; the remap keeps the larger.
    alias: [AtomicU64; 2],
    /// The banks the node holds, as mappings at node offsets.
    mappings: Mutex<Arc<[BankMapping]>>,
    memory: Arc<Physical>,
    /// The boot flash, mapped at node field zero wherever no bank covers its window.
    boot: Option<Arc<Window>>,
    /// Whether the boot flash stays in its window at node field zero for the node's own
    /// processors once the node is renumbered, where the node has no memory behind the
    /// window: the Bedrock keeps it, the Hub does not. Memory in the window is never the
    /// reader's own: it is the zero holder's, as everywhere above the alias.
    boot_local: bool,
    spaces: [OnceLock<Arc<dyn BusDevice>>; 2],
    /// Where the remap shows the boot flash in the node's table, `at` and `len`;
    /// length zero for nowhere.
    flash: [AtomicU64; 2],
    /// What the table holds of the node's own banks and of the far node's: the `old`
    /// of the next remap.
    placed: Mutex<[Vec<BankMapping>; 2]>,
    /// Every processor of the machine, which the remap parks.
    world: OnceLock<Arc<World>>,
}

impl<C> Node<C> {
    /// A node of `chip` holding zero over `memory`, which holds `banks`, with `boot`, a
    /// flash device addressed from zero, answering at `at..at + len`.
    pub fn new(
        chip: Weak<C>,
        memory: Arc<Physical>,
        banks: Arc<[BankMapping]>,
        boot: Option<(u64, u64, Arc<dyn BusDevice>)>,
        boot_local: bool,
    ) -> Self {
        let boot = boot.map(|(at, len, device)| {
            let window = Arc::new(Window::new(at, len, device));
            // The table holds the flash by pointer: the bus keeps it alive.
            memory.keep(window.clone(), "boot flash");
            window
        });
        Self {
            chip,
            number: Default::default(),
            alias: Default::default(),
            mappings: Mutex::new(banks),
            memory,
            boot,
            boot_local,
            spaces: Default::default(),
            flash: Default::default(),
            placed: Default::default(),
            world: OnceLock::new(),
        }
    }

    #[inline]
    fn number(&self) -> NodeId {
        self.number.load(Ordering::Acquire)
    }

    fn alias(&self) -> u64 {
        self.alias.iter().map(|a| a.load(Ordering::Acquire)).max().unwrap_or(0)
    }

    /// Where the table shows the flash now.
    fn flash_span(&self) -> Option<Range<u64>> {
        let len = self.flash[1].load(Ordering::Acquire);
        let at = self.flash[0].load(Ordering::Relaxed);
        (len != 0).then_some(at..at + len)
    }

    /// Show the flash at `span`, or nowhere. Only its length changes once set, so a
    /// racing reader sees the old answer or the new one.
    fn show_flash(&self, span: Option<Range<u64>>) {
        let span = span.unwrap_or(0..0);
        if span.end > span.start {
            self.flash[0].store(span.start, Ordering::Relaxed);
        }
        self.flash[1].store(span.end - span.start, Ordering::Release);
    }

    /// Whether `a` reaches the boot flash rather than memory.
    #[inline]
    fn shows_flash(&self, a: u64) -> bool {
        let len = self.flash[1].load(Ordering::Acquire);
        a.wrapping_sub(self.flash[0].load(Ordering::Relaxed)) < len
    }
}

/// A node chip's decode by node number: its own [`Node`], and the one at the far end
/// of its NI's port. Numbers are published in the remap, so a decode takes no lock.
pub struct NodeDecode<C> {
    shift: u32,
    /// Where an access no node holds is logged.
    log: LogModule,
    /// Where the Hub-special space's reader-relative forms at node field zero
    /// end.
    special_local: u64,
    own: Arc<Node<C>>,
    /// The node at the far end of the NI's port.
    peer: OnceLock<Arc<Node<C>>>,
}

impl<C> NodeDecode<C> {
    /// A node whose number sits at address bit `shift`, holding zero, with
    /// its memory mapped as `own` places it; the chip logs under `log`.
    pub fn new(shift: u32, special_local: u64, own: Node<C>, log: LogModule) -> Arc<Self> {
        let decode = Arc::new(Self { shift, log, special_local, own: Arc::new(own), peer: OnceLock::new() });
        decode.map_banks(&decode.own, None);
        decode
    }

    /// This node as the far node's NI sees it.
    pub fn node(&self) -> &Arc<Node<C>> {
        &self.own
    }

    /// Connect this node's NI port to the node `far` at the other end, once, before
    /// the machine runs, and map each node's memory into the other.
    pub fn connect_peer(&self, far: &Arc<Node<C>>) {
        assert!(!Arc::ptr_eq(&self.own, far), "a node's port cannot connect to itself");
        assert!(self.peer.set(far.clone()).is_ok(), "a node's port is connected once");
        if let (Some(mine), Some(theirs)) = (self.own.world.get(), far.world.get()) {
            assert!(Arc::ptr_eq(mine, theirs), "two connected nodes park the same processors");
        }
        self.remap(&|| {});
    }

    /// The machine's processors, which a remap parks: set once before they start,
    /// the same world on both ends of the port.
    pub fn set_world(&self, world: Arc<World>) {
        if let Some(theirs) = self.peer.get().and_then(|far| far.world.get()) {
            assert!(Arc::ptr_eq(theirs, &world), "two connected nodes park the same processors");
        }
        assert!(self.own.world.set(world).is_ok(), "a node's world is set once");
    }

    /// A store changed what places memory: with others parked, `publish`, remap both nodes and
    /// flush compiled code if anything moved. Without a world (chip tests) it runs unparked.
    pub fn remap(&self, publish: &dyn Fn()) {
        let run = || {
            publish();
            let far = self.peer.get().map(|far| &**far);
            let mut moved = self.map_banks(&self.own, far);
            if let Some(far) = far {
                moved |= self.map_banks(far, Some(&self.own));
            }
            if moved {
                if let Some(world) = self.own.world.get() {
                    world.layout_changed();
                }
            }
        };
        match self.own.world.get() {
            Some(world) => world.stop_others(run),
            None => run(),
        }
    }

    /// The number this node answers to.
    #[inline]
    pub fn number(&self) -> NodeId {
        self.own.number()
    }

    /// This node's memory, which the remap maps.
    pub fn physical(&self) -> &Arc<Physical> {
        &self.own.memory
    }

    /// Publish this node's number, decoded by [`node_id`]. Only the chip's publish
    /// inside the remap calls it.
    pub fn set_number(&self, n: NodeId) {
        debug_assert!(n <= node_mask(self.shift), "node {n:#x} is past the node field");
        self.own.number.store(n, Ordering::Release);
    }

    /// Publish processor interface `pi`'s cached alias, `bytes` long.
    pub fn set_alias(&self, pi: usize, bytes: u64) {
        self.own.alias[pi].store(bytes, Ordering::Release);
    }

    /// Publish the banks the node holds, as mappings at node offsets.
    pub fn set_mappings(&self, mappings: Arc<[BankMapping]>) {
        *self.own.mappings.lock() = mappings;
    }

    /// Register `device` as this node's `space` (Hub special or I/O) once,
    /// before the machine runs.
    pub fn set_space(&self, space: UncachedSpace, device: Arc<dyn BusDevice>) {
        assert!(matches!(space, UncachedSpace::HubSpecial | UncachedSpace::Io), "{space:?} is not node-qualified");
        assert!(self.own.spaces[space as usize].set(device).is_ok(), "{space:?} is wired twice");
    }

    /// The node holding `n`; while both hold the same number this one answers, as
    /// both do at reset.
    #[inline]
    fn holder(&self, n: NodeId) -> Option<&Node<C>> {
        if n == self.own.number() {
            return Some(&self.own);
        }
        self.peer_holding(n)
    }

    /// The far node, if it holds `n`.
    #[inline]
    fn peer_holding(&self, n: NodeId) -> Option<&Node<C>> {
        self.peer.get().map(|far| &**far).filter(|far| far.number() == n)
    }

    /// Whether the NI's port has a node at its far end.
    pub fn connected(&self) -> bool {
        self.peer.get().is_some()
    }

    /// The chip at the far end of the port, while it is alive.
    pub fn peer(&self) -> Option<Arc<C>> {
        self.peer.get()?.chip.upgrade()
    }

    /// Run `f` on the chip holding `n`: `this`, the chip that owns the
    /// decode, or the peer. `None` when no reachable chip holds it.
    pub fn on_holder<R>(&self, this: &C, n: NodeId, f: impl FnOnce(&C) -> R) -> Option<R> {
        if n == self.own.number() {
            return Some(f(this));
        }
        self.peer_holding(n)?.chip.upgrade().map(|chip| f(&chip))
    }

    /// The memory the address's node field names, which takes the address
    /// unchanged: RAM only, so not where the holder's table shows its flash.
    #[inline]
    fn memory_at(&self, a: u64) -> Option<&Physical> {
        let holder = self.holder(node_id(a >> self.shift, self.shift))?;
        (!holder.shows_flash(a)).then_some(&*holder.memory)
    }

    /// The node's own memory at node-relative offsets whatever number it holds, for
    /// the Hub-special space's memory window.
    pub fn own_memory(&self) -> Arc<dyn BusDevice> {
        Arc::new(OwnMemory { memory: self.own.memory.clone(), number: self.own.number.clone(), shift: self.shift })
    }

    /// Map `me`'s table and window to what its chip and `far`'s publish: own banks, the zero
    /// holder's around them, the far node's at its number, and boot flash where no bank covers it.
    /// Whether anything moved.
    fn map_banks(&self, me: &Node<C>, far: Option<&Node<C>>) -> bool {
        let (n, alias) = (me.number(), me.alias());
        // One banks lock at a time: two remaps on either end of the port take
        // them in opposite orders.
        let banks = me.mappings.lock().clone();
        let far_banks = far.map(|f| f.mappings.lock().clone());
        let (mut own, mut across) = (Vec::new(), Vec::new());
        let boot = me.boot.as_ref().map(|w| w.span());
        match far {
            Some(f) if n != 0 => {
                // What stays on the reader at node field zero: the alias, and
                // on the Bedrock the boot flash in its window. The rest is the
                // zero holder's. That includes the window itself wherever the
                // reader has memory of its own behind it and so shows no flash:
                // a cached reference there is ordinary memory by node number,
                // the same bytes for every processor, DMA master and block
                // transfer. Kept on the reader, a node with 512 MiB or more
                // read and wrote its own memory for four megabytes the kernel
                // hands out as the zero holder's pages.
                let local = boot.clone().filter(|_| me.boot_local).unwrap_or(u64::MAX..u64::MAX);
                own.extend(shifted(&banks, 0, 0..alias));
                if f.number() == 0 {
                    let far_banks = far_banks.as_deref().unwrap_or_default();
                    across.extend(shifted(far_banks, 0, alias..local.start.max(alias)));
                    for mine in shifted(&banks, 0, local.clone()) {
                        across.extend(shifted(far_banks, 0, mine.base.max(alias)..mine.end));
                    }
                    across.extend(shifted(far_banks, 0, local.end.max(alias)..u64::MAX));
                } else {
                    own.extend(shifted(&banks, 0, local.clone()));
                }
            }
            _ => own.extend(shifted(&banks, 0, 0..u64::MAX)),
        }
        if n != 0 {
            own.extend(shifted(&banks, u64::from(n) << self.shift, 0..u64::MAX));
        }
        if let (Some(f), Some(far_banks)) = (far, far_banks.as_deref()) {
            if f.number() != 0 && f.number() != n {
                across.extend(shifted(far_banks, u64::from(f.number()) << self.shift, 0..u64::MAX));
            }
        }
        let gaps = boot.map_or_else(Vec::new, |span| uncovered(span, own.iter().chain(&across)));
        let mut placed = me.placed.lock();
        let [own_old, across_old] = &mut *placed;
        // DMA is not parked: while the table changes it refuses both the old flash
        // span and the new one.
        let span = hull(gaps.iter().cloned());
        me.show_flash(hull(me.flash_span().into_iter().chain(span.clone())));
        let mut moved = false;
        if let Some(window) = &me.boot {
            let flash: *const Window = Arc::as_ptr(window);
            let entry: Entry = flash;
            for gap in &gaps {
                moved |= me.memory.fill_slots(gap.start >> SLOT_SHIFT, (gap.end - gap.start) >> SLOT_SHIFT, entry);
            }
        }
        let beyond = far.map_or(&me.memory, |f| &f.memory);
        moved |= me.memory.map_banks_of(beyond, own_old, &own, across_old, &across);
        me.show_flash(span);
        *own_old = own;
        *across_old = across;
        moved
    }
}

impl<C: Send + Sync + 'static> NodeDecode<C> {
    /// A processor's view of `space`: node field zero stays on this node in reader-relative forms,
    /// any other address goes to its holder.
    pub fn cpu_view(self: &Arc<Self>, space: UncachedSpace, own: Option<Arc<dyn BusDevice>>) -> Arc<dyn BusDevice> {
        let decode = self.clone();
        match space {
            UncachedSpace::HubSpecial => Arc::new(NodeQualifiedView::<C, false> { decode, own }),
            UncachedSpace::Io => Arc::new(NodeQualifiedView::<C, true> { decode, own }),
            _ => panic!("{space:?} is not node-qualified"),
        }
    }
}

/// A processor's view of the Hub-special space, or with `IO` the I/O space.
struct NodeQualifiedView<C, const IO: bool> {
    decode: Arc<NodeDecode<C>>,
    own: Option<Arc<dyn BusDevice>>,
}

impl<C, const IO: bool> NodeQualifiedView<C, IO> {
    const SPACE: UncachedSpace = if IO { UncachedSpace::Io } else { UncachedSpace::HubSpecial };

    #[inline(always)]
    fn target(&self, a: u64) -> Option<(&dyn BusDevice, u64)> {
        let r = &*self.decode;
        // The node field ends at bit 39, so this is node_id's number.
        let (number, local) = (((a & PA_MASK) >> r.shift) as NodeId, a & ((1 << r.shift) - 1));
        let stays = number == 0
            && if IO { local.wrapping_sub(IO_REGISTERS_LOCAL) < IO_REGISTERS_NAMED - IO_REGISTERS_LOCAL } else { local < r.special_local };
        let node = if stays || number == r.own.number() {
            if let Some(own) = &self.own {
                return Some((&**own, local));
            }
            &*r.own
        } else {
            r.peer_holding(number)?
        };
        Some((&**node.spaces[Self::SPACE as usize].get()?, local))
    }

    // Inlined into each width, so its size and direction are constants there.
    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        match self.target(a) {
            Some((d, at)) => widths::access(d, at, size, value),
            None => self.unclaimed(a),
        }
    }

    #[cold]
    #[inline(never)]
    fn unclaimed(&self, a: u64) -> BusRead64 {
        let r = &*self.decode;
        crate::dlog!(r.log, "{:?} access at {a:#x}: no node holds number {}", Self::SPACE, node_id(a >> r.shift, r.shift));
        BusRead64::err()
    }
}

impl<C: Send + Sync, const IO: bool> BusDevice for NodeQualifiedView<C, IO> {
    fn write64_masked(&self, a: u64, val: u64, mask: u64) -> u32 {
        self.target(a).map_or(BUS_ERR, |(d, at)| d.write64_masked(at, val, mask))
    }
    fn write_narrow_from(&self, a: u64, size: usize, gpr: u64) -> u32 {
        self.target(a).map_or(BUS_ERR, |(d, at)| d.write_narrow_from(at, size, gpr))
    }
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, a: u64) -> *const std::sync::atomic::AtomicU64 {
        self.target(a).map_or(std::ptr::null(), |(d, at)| d.gen_ptr(at))
    }
    #[cfg(feature = "jitv2")]
    fn code_page(&self, a: u64) -> Option<crate::traits::CodePage> {
        self.target(a).and_then(|(d, at)| d.code_page(at))
    }
    access_widths!();
}

// A decode as a bus device is node memory by number, as inbound DMA and block
// transfers reach it: RAM only, so a flash window is a bus error.
impl<C> NodeDecode<C> {
    #[inline]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        match self.memory_at(a) {
            Some(m) => widths::access(m, a, size, value),
            None => BusRead64::err(),
        }
    }

    /// Walk `words` doublewords from `a` a page at a time, each page from the node
    /// its number names; the first status that is not OK ends the walk.
    fn pages(&self, a: u64, words: usize, mut f: impl FnMut(&Physical, u64, std::ops::Range<usize>) -> u32) -> u32 {
        let mut done = 0;
        while done < words {
            let at = a.wrapping_add(8 * done as u64);
            let n = ((4096 - (at & 4095)) as usize / 8).min(words - done);
            let Some(memory) = self.memory_at(at) else { return BUS_ERR };
            let status = f(memory, at, done..done + n);
            if status != BUS_OK {
                return status;
            }
            done += n;
        }
        BUS_OK
    }
}
impl<C: Send + Sync> BusDevice for NodeDecode<C> {
    fn read_block(&self, a: u64, buf: &mut [u64]) -> u32 {
        self.pages(a, buf.len(), |m, at, words| m.read_block(at, &mut buf[words]))
    }
    fn write_block(&self, a: u64, buf: &[u64]) -> u32 {
        self.pages(a, buf.len(), |m, at, words| m.write_block(at, &buf[words]))
    }
    access_widths!();
}

struct OwnMemory {
    memory: Arc<Physical>,
    number: Arc<AtomicU16>,
    shift: u32,
}

impl OwnMemory {
    #[inline]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        widths::access(&*self.memory, (u64::from(self.number.load(Ordering::Acquire)) << self.shift) + a, size, value)
    }
}
impl BusDevice for OwnMemory {
    access_widths!();
}

/// The memory-special space: fetch-and-op on node memory, each operation one compare-and-swap;
/// other widths or operations are a bus error.
pub struct FetchOp(pub Arc<Physical>);

impl FetchOp {
    fn op(&self, a: u64, bytes: u64, store: Option<u64>) -> Option<u64> {
        let (op, at) = (a & 0x38, (a & !63) | (a & 4));
        loop {
            let old = match bytes {
                4 => Some(self.0.read32(at)).filter(|r| r.is_ok()).map(|r| u64::from(r.data)),
                _ => Some(self.0.read64(at)).filter(|r| r.is_ok()).map(|r| r.data),
            }?;
            let new = match (store, op) {
                (None, 0) => return Some(old),
                (Some(v), 0) => v,
                (_, 8) => old.wrapping_add(1),
                (_, 16) => old.wrapping_sub(1),
                (None, 24) => 0,
                (Some(v), 24) => old & v,
                (Some(v), 32) => old | v,
                _ => return None,
            };
            let swapped = match bytes {
                4 => self.0.cas32(at, old as u32, new as u32),
                _ => self.0.cas64(at, old, new),
            }?;
            if swapped {
                return Some(old);
            }
        }
    }
}

impl BusDevice for FetchOp {
    fn read32(&self, a: u64) -> BusRead32 {
        self.op(a, 4, None).map_or_else(BusRead32::err, |v| BusRead32::ok(v as u32))
    }
    fn read64(&self, a: u64) -> BusRead64 {
        self.op(a, 8, None).map_or_else(BusRead64::err, BusRead64::ok)
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        self.op(a, 4, Some(v.into())).map_or(BUS_ERR, |_| BUS_OK)
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        self.op(a, 8, Some(v)).map_or(BUS_ERR, |_| BUS_OK)
    }
}

/// Memory that is not installed: every access is a bus error, logged under the
/// node chip's category.
pub struct NoMemory(pub LogModule);
impl NoMemory {
    fn access(&self, a: u64, _: usize, written: Option<u64>) -> BusRead64 {
        match written {
            Some(v) => crate::dlog!(self.0, "write of {v:#x} to no memory {a:#x}"),
            None => crate::dlog!(self.0, "read of no memory {a:#x}"),
        }
        BusRead64::err()
    }
}
impl BusDevice for NoMemory {
    access_widths!();
}

/// The memory a processor can reach: the widest node's ranges once at every node number; the JIT
/// sizes its page table from it.
pub fn machine_memory(maps: &[crate::memory_map::MemoryMap], shift: u32) -> crate::memory_map::MemoryMap {
    use crate::memory_map::{MemRange, MemoryMap};
    assert!(maps.len() <= MAX_NODES, "{} nodes: no machine iris builds has more than {MAX_NODES}", maps.len());
    let Some(widest) = maps.iter().max_by_key(|m| m.end()) else { return MemoryMap::new(Vec::new()) };
    let ranges = (0..maps.len() as u64)
        .flat_map(|n| widest.ranges().iter().map(move |r| MemRange::new(r.name, r.kind, r.base + (n << shift), r.end() + (n << shift))))
        .collect();
    let map = MemoryMap::new(ranges);
    let bound = (MAX_NODES as u64) << shift;
    assert!(map.end() <= bound, "the processors' memory ends at {:#x}, past {bound:#x}", map.end());
    map
}

/// One node board of either family: chip, boot flash, uncached bus and first processor number
/// (`ip27.rs`, `ip35/node.rs`).
pub struct NodeBoard<C, F> {
    pub chip: Arc<C>,
    pub flash: Arc<F>,
    pub bus: Arc<UncachedBus>,
    pub first_cpu: usize,
}

/// `maps`' banks from node offset `within`, placed so offset zero lands at `base`.
fn shifted(maps: &[BankMapping], base: u64, within: Range<u64>) -> impl Iterator<Item = BankMapping> + '_ {
    maps.iter().filter_map(move |b| {
        let (lo, hi) = (b.base.max(within.start), b.end.min(within.end));
        (lo < hi).then(|| BankMapping { base: base + lo, end: base + hi, bank: b.bank, src: b.src + lo - b.base })
    })
}

/// The span `spans` cover together.
fn hull(spans: impl Iterator<Item = Range<u64>>) -> Option<Range<u64>> {
    spans.reduce(|a, b| a.start.min(b.start)..a.end.max(b.end))
}

/// The parts of `span` that no mapping of `maps` covers.
fn uncovered<'a>(span: Range<u64>, maps: impl Iterator<Item = &'a BankMapping>) -> Vec<Range<u64>> {
    let mut gaps = vec![span];
    for m in maps {
        gaps = gaps
            .into_iter()
            .flat_map(|g| {
                if m.end <= g.start || m.base >= g.end {
                    vec![g]
                } else {
                    [g.start..m.base.max(g.start), m.end.min(g.end)..g.end].into_iter().filter(|r| r.start < r.end).collect()
                }
            })
            .collect();
    }
    gaps
}

/// A file for node `node` beside the one the configuration names for the
/// first.
pub(crate) fn per_node(path: &Option<String>, node: usize) -> Option<std::path::PathBuf> {
    path.as_ref().map(|p| if node == 0 { p.into() } else { format!("{p}.{node}").into() })
}

/// The Ethernet's NAT gateway, as the front ends read it.
pub struct Nat {
    pub control: Arc<NatControl>,
    pub expected: (Ipv4Addr, Ipv4Addr, Ipv4Addr),
    pub frames: Arc<AtomicU64>,
}

/// What a board's I/O hands the machine once it is built on the first node.
pub struct BoardIo {
    /// The devices with threads or timers: started in order, stopped in reverse.
    pub devices: Vec<Arc<dyn Device>>,
    /// Devices the monitor gets commands from.
    pub monitor: Vec<Arc<dyn Device>>,
    /// Put the board's devices back to power-on.
    pub reset: Box<dyn Fn() + Send + Sync>,
    pub ps2: Arc<Ps2Controller>,
    /// Write the NVRAM back to its file.
    pub flush_nvram: Box<dyn Fn() -> Result<(), String> + Send + Sync>,
    /// The console, when it serves the CI socket.
    pub console: Option<Arc<CiSerialBackend>>,
    pub attach_net: Box<dyn Fn(Link) + Send + Sync>,
    /// Run the closure over the SCSI buses that hold the machine's disks.
    #[allow(clippy::type_complexity)]
    pub disks: Box<dyn Fn(&mut dyn FnMut(&[&ScsiBus])) + Send + Sync>,
    /// The controller whose removable media the front ends change, if any.
    pub disc_changer: Option<Arc<dyn DiscChanger>>,
    /// The `scratch = true` volume.
    pub scratch: Option<PathBuf>,
    /// The graphics board's window, when the product has one.
    pub display: Option<Arc<dyn GfxDisplay>>,
    /// What the graphics board's status bar shows.
    pub status: Option<Arc<crate::disp::StatusSource>>,
    /// IP35 only: brings the board's clocked devices up to date, every batch.
    pub poll: Option<Arc<dyn Fn() + Send + Sync>>,
    /// IP35 only: per-processor batch hook on a node of several, beside the poll.
    pub at_batch: Option<Arc<dyn Fn() + Send + Sync>>,
    /// IP35 only: the counter the status bar blinks on the first processor's Compare matches.
    pub fasttick: Option<Arc<AtomicU64>>,
    /// IP35 only: told of the machine's processors once they exist.
    #[allow(clippy::type_complexity)]
    pub cpus: Option<Box<dyn Fn(&[Arc<dyn CpuDevice>]) + Send + Sync>>,
}

/// One IP27 or IP35 machine: its node boards, L1s, processors and board I/O. Each family builds it
/// with its own `machine` function (`ip27.rs`, `ip35.rs`).
pub struct Machine<C: Device + Resettable + 'static, F: Device + Resettable + 'static> {
    /// The product's name as a sentence has it.
    pub(crate) name: &'static str,
    pub(crate) nodes: Vec<NodeBoard<C, F>>,
    /// Each node's L1, in node order: one per brick, none on the IP27 boards.
    pub(crate) l1s: Vec<Arc<L1>>,
    /// Each node's processors, in node order, the node's first first.
    pub(crate) node_cpus: Vec<Vec<Arc<dyn CpuDevice>>>,
    pub(crate) io: BoardIo,
    /// What a chip lets go of before its processors and memory are freed.
    release: fn(&C),
    timers: Arc<TimerManager>,
    monitor: Arc<Monitor>,
    monitor_port: u16,
    ci: bool,
    nvram: String,
    nat: Option<Nat>,
    events: Option<Receiver<MachineEvent>>,
    slot: Option<MachineSlot>,
    lifecycle: GuestState,
}

impl<C: Device + Resettable + 'static, F: Device + Resettable + 'static> Machine<C, F> {
    /// The machine of `nodes`, their processors and `io`, whose NAT link, status bar and monitor are
    /// wired here. `events` carries the nodes' reset and power requests to the system controller.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        cfg: &MachineConfig,
        nodes: Vec<NodeBoard<C, F>>,
        l1s: Vec<Arc<L1>>,
        node_cpus: Vec<Vec<Arc<dyn CpuDevice>>>,
        io: BoardIo,
        memory: Vec<Arc<dyn Device>>,
        timers: Arc<TimerManager>,
        events: Receiver<MachineEvent>,
        release: fn(&C),
    ) -> Result<Self, String> {
        let cpus: Vec<Arc<dyn CpuDevice>> = node_cpus.iter().flatten().cloned().collect();
        let first = &cpus[0];
        if let Some(status) = &io.status {
            status.set_cpu_cycles(first.cycles_ptr());
            status.set_count_hz(first.count_hz_atomic());
            for c in &cpus[1..] {
                status.add_cpu_stats(c.cycles_ptr(), c.fasttick_count());
            }
        }
        let monitor = Monitor::new();
        monitor.register_device(init_devlog());
        monitor.register_device(Arc::new(crate::cpu::cpu_monitor::CpuMonitor::new(cpus.clone())));
        monitor.register_device(PerfMonitor::with_cpus(first.running_flag(), cpus.iter().map(|c| c.cycles_ptr()).collect(), first.fasttick_count()));
        for node_memory in memory {
            monitor.register_device(node_memory);
        }
        for device in &io.monitor {
            monitor.register_device(device.clone());
        }
        // The guest timeline follows the first CPU's retired instructions.
        #[cfg(feature = "perfetto")]
        {
            crate::perfetto::set_guest_cycles(Some(first.cycles_ptr()));
            monitor.register_device(Arc::new(crate::perfetto::PerfettoMonitor));
        }
        let mut nat = None;
        let net = GatewayConfig::from_network(&cfg.network());
        if let Some(link) = Link::with_gateway(net.clone())? {
            nat = Some(Nat { control: link.nat(), expected: (net.client_ip, net.gateway_ip, net.netmask), frames: link.frames() });
            (io.attach_net)(link);
        }
        Ok(Self {
            name: cfg.machine.profile.product_name().unwrap_or_default(),
            nodes,
            l1s,
            node_cpus,
            io,
            release,
            timers,
            monitor: Arc::new(monitor),
            monitor_port: cfg.monitor_port.unwrap_or(DEFAULT_MONITOR_PORT),
            ci: cfg.ci,
            nvram: cfg.nvram.clone(),
            nat,
            events: Some(events),
            slot: None,
            lifecycle: GuestState::Idle,
        })
    }

    /// CPU 0 of the first node, the one the front ends drive.
    fn cpu(&self) -> &Arc<dyn CpuDevice> {
        &self.node_cpus[0][0]
    }

    pub(crate) fn cpus(&self) -> impl Iterator<Item = &Arc<dyn CpuDevice>> {
        self.node_cpus.iter().flatten()
    }

    pub(crate) fn with_disks<R>(&self, f: impl FnOnce(&[&ScsiBus]) -> R) -> R {
        let mut f = Some(f);
        let mut out = None;
        (self.io.disks)(&mut |buses| out = f.take().map(|f| f(buses)));
        out.expect("the board visits its disks once")
    }
}

impl<C: Device + Resettable + 'static, F: Device + Resettable + 'static> Workstation for Machine<C, F> {
    fn start(&mut self) {
        self.lifecycle = GuestState::Starting;
        self.monitor.clone().start_server(format!("127.0.0.1:{}", self.monitor_port));
        self.restart_peripherals();
        if !self.ci {
            self.cpu_start();
        }
    }
    fn stop(&mut self) {
        for c in self.cpus() {
            c.clone().as_device().stop();
        }
        // The guest clock points into this machine's first CPU.
        #[cfg(feature = "perfetto")]
        crate::perfetto::set_guest_cycles(None);
        for d in self.io.devices.iter().rev() {
            d.stop();
        }
        for node in &self.nodes {
            node.flash.stop();
        }
        for (n, node) in self.nodes.iter().enumerate() {
            if let Some(l1) = self.l1s.get(n) {
                l1.stop();
            }
            node.chip.stop();
        }
        if let Err(e) = (self.io.flush_nvram)() {
            crate::dlog!(LogModule::Rtc, "{} NVRAM: {e}", self.name);
        }
        if self.lifecycle != GuestState::Idle {
            self.lifecycle = GuestState::Halted;
        }
    }
    fn reset(&mut self) {
        self.stop();
        for c in self.cpus() {
            c.power_on();
        }
        for (n, node) in self.nodes.iter().enumerate() {
            node.chip.power_on();
            if let Some(l1) = self.l1s.get(n) {
                l1.power_on();
            }
            node.flash.power_on();
        }
        (self.io.reset)();
        self.restart_peripherals();
    }
    fn restart_peripherals(&mut self) {
        for (n, node) in self.nodes.iter().enumerate() {
            node.flash.start();
            node.chip.start();
            if let Some(l1) = self.l1s.get(n) {
                l1.start();
            }
        }
        for d in &self.io.devices {
            d.start();
        }
    }
    /// Every processor of every node starts at power-on; each node's PROM waits for the
    /// others' link and discovery.
    fn cpu_start(&self) {
        for c in self.cpus() {
            c.clone().as_device().start();
        }
    }
    fn cpu_is_running(&self) -> bool {
        self.cpus().any(|c| c.clone().as_device().is_running())
    }
    fn machine_slot(&mut self) -> MachineSlot {
        let ptr = self as *mut Self as *mut dyn Workstation;
        self.slot.get_or_insert_with(|| Arc::new(Mutex::new(MachineRef(ptr)))).clone()
    }
    fn register_system_controller(&mut self) {
        let slot = self.machine_slot();
        let events = self.events.take().expect("the system controller is registered once");
        register_system_controller(&self.monitor, slot, Some(events));
    }
    fn monitor(&self) -> &Arc<Monitor> {
        &self.monitor
    }
    fn lifecycle(&self) -> GuestState {
        self.lifecycle
    }
    fn set_lifecycle(&mut self, state: GuestState) {
        self.lifecycle = state;
    }
    fn get_display(&self) -> Option<Arc<dyn GfxDisplay>> {
        self.io.display.clone()
    }
    // The chips and node memory keep no saved state, so a snapshot would not restore the machine.
    fn save_snapshot(&mut self, _name: &str) -> Result<(), String> {
        Err(format!("snapshots of the {} are not supported yet: the node chips and node memory have no saved state", self.name))
    }
    fn load_snapshot(&mut self, _name: &str) -> Result<(), String> {
        Err(format!("snapshots of the {} are not supported yet: the node chips and node memory have no saved state", self.name))
    }
    fn register_monitor_device(&self, device: Arc<dyn Device>) {
        self.monitor.register_device(device);
    }
    fn get_ps2(&self) -> Arc<Ps2Controller> {
        self.io.ps2.clone()
    }
    fn get_timer_manager(&self) -> Arc<TimerManager> {
        self.timers.clone()
    }
    fn get_cpu_debug(&self) -> Arc<dyn CpuDebug> {
        self.cpu().clone().debug_adapter()
    }
    fn fasttick_count(&self) -> Arc<AtomicU64> {
        self.cpu().fasttick_count()
    }
    fn cpu_counters(&self) -> Vec<(crate::cpu::mips_core::CyclesPtr, Arc<AtomicU64>)> {
        self.cpus().map(|c| (c.cycles_ptr(), c.fasttick_count())).collect()
    }
    fn get_ci_serial(&self) -> Option<Arc<CiSerialBackend>> {
        self.io.console.clone()
    }
    fn inject_serial_console(&self, bytes: &[u8]) {
        if let Some(console) = &self.io.console {
            console.push_host(bytes);
        }
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn save_nvram(&self, path: Option<&str>) -> Result<String, String> {
        if path.is_some_and(|p| p != self.nvram) {
            return Err(format!("{} NVRAM save uses the configured path", self.name));
        }
        (self.io.flush_nvram)()?;
        Ok(self.nvram.clone())
    }
    fn scratch_path(&self) -> Option<&Path> {
        self.io.scratch.as_deref()
    }
    fn disc_changer(&self) -> Option<Arc<dyn DiscChanger>> {
        self.io.disc_changer.clone()
    }
    fn pending_chd_sync_count(&self) -> usize {
        self.with_disks(|buses| buses.iter().map(|b| b.pending_chd_sync_count()).sum())
    }
    fn disk_activity(&self) -> Vec<Vec<crate::scsi::DiskCounts>> {
        self.with_disks(|buses| buses.iter().map(|b| b.activity()).collect())
    }
    fn flush_disks(&self) -> std::io::Result<()> {
        self.with_disks(|buses| buses.iter().try_for_each(|b| b.flush()))
    }
    fn sync_chd_disks(
        &self,
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        self.with_disks(|buses| ScsiBus::sync_chd_disks(buses, only, progress, cancel))
    }
    fn net_frames(&self) -> Option<Arc<AtomicU64>> {
        self.nat.as_ref().map(|n| n.frames.clone())
    }
    fn nat_control(&self) -> Option<Arc<NatControl>> {
        self.nat.as_ref().map(|n| n.control.clone())
    }
    fn nat_expected(&self) -> Option<(Ipv4Addr, Ipv4Addr, Ipv4Addr)> {
        self.nat.as_ref().map(|n| n.expected)
    }
}

impl<C: Device + Resettable + 'static, F: Device + Resettable + 'static> Drop for Machine<C, F> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        self.stop();
        // The compile pool's workers read the node memory the machine frees.
        #[cfg(feature = "jitv2")]
        for cpus in &self.node_cpus {
            crate::cpu::jitv2::Jitv2::stop_for_drop(&cpus[0].jitv2(), || {
                for c in cpus {
                    c.clone().as_device().stop();
                }
            });
        }
        crate::workstation::flush_on_drop(self);
        self.monitor.shutdown();
        // Nothing publishes into a processor's interrupt word once the processors go, and each
        // node's table points into the other's banks, so every node lets go before any is freed.
        for node in &self.nodes {
            (self.release)(&node.chip);
        }
    }
}


#[cfg(test)]
mod bench;
#[cfg(test)]
pub(crate) mod tests;
