//! Bedrock, the node controller of the IP35 boards, as one device: two
//! processor interfaces (PI), the memory and directory controller (MD) with
//! its memory, the I/O block (II) that faces the crossbar widgets, the local
//! block (LB), the network interface (NI) and the crossbar block (XB).
//!
//! A board gives the chip its processors, banks and clock in a
//! [`BedrockConfig`], maps the faces it hands out into its processor bus,
//! and gives each processor's interrupt word to [`Bedrock::set_cpu_interrupts`].
//! The units are plain fields; each locks only its own state.
//!
//! A unit holds no reference to another and no closure. What a unit's
//! access changes beyond the unit comes back as an `Effect`, which the chip
//! acts on once the unit's lock is released. The error levels LB, NI and XB
//! drive are PI level bits that each writer sets under its own lock. The
//! chip calls nothing in the board: a node reset leaves as a
//! [`MachineEvent::HardReset`] on the board's event channel.
//!
//! # Locks
//!
//! - PI: the interrupt inputs (pending banks, masks, requests, timer and
//!   error status) are atomics, so a mask store, a pending-modify store and
//!   a counter read take no lock. `state` holds the stored words and the
//!   graphics registers; `timer` the comparators and the host timer.
//! - MD: the cell register is an atomic, and memory, cached, uncached or
//!   DMA, takes no lock; `state` and `remapping` as the controller's notes
//!   say.
//! - II: the permission words and translation entries are atomics, so PIO,
//!   inbound memory and an inbound interrupt packet take no lock; `regs`
//!   holds the rest.
//! - LB, NI and XB: `regs` each, under which a block drives its error level.
//!
//! Ported from hw/misc/sgi-bedrock*.c in jamesbraid/qemu-sgi-origami.

mod io;
mod lb;
mod md;
mod mem;
mod ni;
mod pi;
mod xb;
#[cfg(test)]
mod tests;
#[cfg(test)]
use tests::{test_config, Rig};

pub use mem::{RamExtent, BANK_SPAN, SLOTS};
pub use pi::Population;

use crate::dev::heart::Clock;
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::irq::{InterruptController, IrqLine};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Device, MachineEvent, Resettable, BUS_ERR, BUS_OK};
use io::Io;
use md::{Md, View};
use pi::{Levels, Pi};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

/// Unit bases inside the chip's 8 MiB register window, the same in both XIO
/// register forms, and the registers a board or test names.
pub mod reg {
    pub const PI0_BASE: u64 = 0x00_0000;
    pub const PI1_BASE: u64 = 0x20_0000;
    pub const IO_BASE: u64 = 0x40_0000;
    pub const LB_BASE: u64 = 0x60_0000;
    pub const NI_BASE: u64 = 0x68_0000;
    pub const XB_BASE: u64 = 0x70_0000;
    pub const MD_BASE: u64 = 0x78_0000;
    pub const WINDOW: u64 = 0x80_0000;
    /// The public Linux Hub I/O protection-override key literal.
    ///
    /// Linux `arch/mips/include/asm/sn/sn0/hubio.h` at
    /// `db2ddb87143519e20a95aa36c60b36107b736a58` defines
    /// `IIO_PROTECT_OVRRD_KEY` as this value (lines 31, 47). This establishes
    /// the literal's provenance only; it does not establish a register mapping.
    pub const IIO_PROTECT_OVRRD_KEY: u64 = 0x53474972756c6573;
    pub use super::io::{bte_reg as bte, reg as io};
    pub use super::lb::reg as lb;
    pub use super::md::reg as md;
    pub use super::ni::reg as ni;
    pub use super::pi::reg as pi;
    pub use super::xb::reg as xb;
}

/// What a board gives its Bedrock. Fixed for the chip's life.
pub struct BedrockConfig<'a> {
    /// The board's name in log lines.
    pub name: &'static str,
    /// Host nanoseconds: the PIs' counter and timers run on it, and the
    /// other units' clocked work.
    pub clock: Clock,
    /// The processor slots the board fits. CPU n sits in
    /// `processors.slot_of(n)`; a fitted slot starts with its enable bit set.
    pub processors: Population,
    /// The memory banks in MiB, locational bank 0 upward.
    pub banks_mib: &'a [u32],
    /// Where a node reset goes: LB's soft reset and NI's local reset each
    /// send [`MachineEvent::HardReset`].
    pub events: Option<Sender<MachineEvent>>,
}

/// What asked for a node reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeReset {
    /// LB's soft-reset bit.
    LocalBlock,
    /// NI's local-reset bit.
    Network,
}

/// What a unit's access changed outside the unit, returned with the unit's
/// lock released for [`Bedrock::apply`].
#[must_use]
enum Effect {
    None,
    /// A bit of the chip's levels changed: both PIs publish.
    Reflect,
    /// Run block-transfer engine `n` now.
    Bte(usize),
    /// The unit has reset itself; the node resets.
    Reset(NodeReset),
}

/// A PI's host timer: the manager it runs on and the chip it fires into.
struct HostTimer<'a> {
    tm: &'a TimerManager,
    chip: &'a Weak<Bedrock>,
}
impl HostTimer<'_> {
    /// A one-shot `delay` nanoseconds out that latches PI `interface`'s due
    /// comparators and arms the next.
    fn arm(&self, interface: u8, delay: u64) -> TimerId {
        self.tm.add_one_shot(Duration::from_nanos(delay), (self.chip.clone(), interface), |(chip, interface)| {
            if let Some(chip) = chip.upgrade() {
                chip.expire(usize::from(*interface));
            }
            TimerReturn::Delete
        })
    }
}

pub struct Bedrock {
    name: &'static str,
    clock: Clock,
    processors: Population,
    pi: [Pi; 2],
    md: Md,
    io: Io,
    lb: lb::Lb,
    ni: ni::Ni,
    xb: xb::Xb,
    levels: Levels,
    /// The node number LB's region register names; inbound packets and the
    /// BTEs' interrupts check it. Set under LB's lock.
    node: AtomicU8,
    /// MD's DMA view: the BTEs and inbound memory requests use it.
    dma: Arc<dyn BusDevice>,
    timers: OnceLock<Weak<TimerManager>>,
    events: Option<Sender<MachineEvent>>,
    weak: Weak<Self>,
}

impl Bedrock {
    pub fn new(config: BedrockConfig<'_>) -> Result<Arc<Self>, String> {
        let BedrockConfig { name, clock, processors, banks_mib, events } = config;
        let memory = mem::Memory::new(banks_mib)?;
        let now = clock();
        let chip = Arc::new_cyclic(|weak: &Weak<Self>| {
            let md = Md::new(memory, clock.clone());
            Self {
                name,
                pi: [Pi::new(0, processors, now), Pi::new(1, processors, now)],
                dma: md.dma_view(),
                md,
                io: Io::new(),
                lb: lb::Lb::new(),
                ni: ni::Ni::new(),
                xb: xb::Xb::new(),
                levels: Levels::default(),
                node: AtomicU8::new(0),
                timers: OnceLock::new(),
                events,
                clock,
                processors,
                weak: weak.clone(),
            }
        });
        chip.md.rebuild_memory();
        Ok(chip)
    }

    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_mutex_fn;
        register_mutex_fn("bedrock-pi0::state", self, |c| c.pi[0].state_locked());
        register_mutex_fn("bedrock-pi1::state", self, |c| c.pi[1].state_locked());
        register_mutex_fn("bedrock-pi0::timer", self, |c| c.pi[0].timer_locked());
        register_mutex_fn("bedrock-pi1::timer", self, |c| c.pi[1].timer_locked());
        register_mutex_fn("bedrock-pi0::out", self, |c| c.pi[0].out_locked());
        register_mutex_fn("bedrock-pi1::out", self, |c| c.pi[1].out_locked());
        register_mutex_fn("bedrock-md::state", self, |c| c.md.state_locked());
        register_mutex_fn("bedrock-md::remapping", self, |c| c.md.remapping_locked());
        register_mutex_fn("bedrock-io", self, |c| c.io.locked());
        register_mutex_fn("bedrock-lb", self, |c| c.lb.locked());
        register_mutex_fn("bedrock-ni", self, |c| c.ni.locked());
        register_mutex_fn("bedrock-xb", self, |c| c.xb.locked());
    }

    // The faces a board maps into its processor bus. Each is its own type,
    // so no access tests which face it came through.

    /// The 8 MiB register window in its node XIO form.
    pub fn registers(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(Registers::<false>(self.clone()))
    }
    /// The register window in its local XIO form: PI0's range answers the
    /// requester's own PI when the board fits processors on both PIs, as
    /// QEMU's IP35 node overlays PI1 there for a processor on it (OpenBSD
    /// expects the local form to select the requester's interface). That the
    /// chip answers so is an approximation pending a hardware read. With
    /// processors on PI0 alone it is the node form.
    pub fn local_registers(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        if (0..2).any(|slot| self.processors.fitted(1, slot)) {
            Arc::new(Registers::<true>(self.clone()))
        } else {
            self.registers()
        }
    }
    /// The II's PIO face: the processor's uncached I/O view below 4 GiB,
    /// decoded to a widget.
    pub fn pio(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(Pio(self.clone()))
    }
    /// Memory for the processor's cached view and DMA: Physical's entries.
    pub fn dma_view(&self) -> Arc<dyn BusDevice> {
        self.dma.clone()
    }
    /// Uncached memory: the banks, as MD's configuration places them.
    pub fn uncached_view(&self) -> Arc<dyn BusDevice> {
        self.md.uncached_view()
    }
    pub fn directory_bus(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        self.md.view(self.weak.clone(), View::Directory)
    }
    pub fn ecc_bus(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        self.md.view(self.weak.clone(), View::Ecc)
    }
    pub fn ecc_scrub_bus(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        self.md.view(self.weak.clone(), View::Scrub)
    }
    pub fn fetchop_bus(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        self.md.view(self.weak.clone(), View::Fetchop)
    }
    pub fn physical(&self) -> &Arc<crate::physical::Physical> {
        self.md.physical()
    }
    /// Where fitted RAM can answer under any bank-select setting.
    pub fn ram_extent(&self) -> RamExtent {
        self.md.ram_extent()
    }
    /// Reinstall RAM after the board sets Physical's default; on the CPU
    /// thread or with the CPU stopped.
    pub fn rebuild_memory(&self) {
        self.md.rebuild_memory();
    }
    /// Whether processor `cpu`'s join after the PROM's reset barrier has
    /// landed on LB scratch 1 since the LB's last reset: the first store it
    /// made there after the boot master's release. One atomic load.
    pub fn scratch1_joined(&self, cpu: usize) -> bool {
        self.lb.scratch1_joined(cpu)
    }
    /// II scratch 0 (0x150), where the IP35 PROM publishes each interface's
    /// processors, bit 4 + n for slot n (A, B on PI0, C, D on PI1). One
    /// atomic load.
    pub fn io_scratch0(&self) -> u64 {
        self.io.scratch0()
    }
    /// Whether LB scratch 3's bit 0, which the IP35 PROM takes as its hub
    /// lock, is clear. One atomic load.
    pub fn hub_lock_free(&self) -> bool {
        self.lb.hub_lock_free()
    }
    /// The processors, one bit per CPU number, that have stored to their own
    /// PI real-time compare register since their presence join on LB
    /// scratch 1: the IP35 PROM's pair liveness posts. One atomic load.
    pub fn rt_posted(&self) -> u64 {
        self.lb.rt_posted()
    }
    #[cfg(feature = "jitv2")]
    pub(crate) fn generation_base(&self) -> *mut AtomicU64 {
        self.md.generation_base()
    }
    /// MD's cell register bit 0, which gates the register window and the
    /// board's Hub-special windows. One atomic load.
    #[inline]
    pub fn cell_permits(&self) -> bool {
        self.md.cell_permits()
    }

    // Widgets.

    /// Connect crossbar widget `widget`, once, before the machine runs.
    pub fn connect_widget(&self, widget: u8, dev: Arc<dyn BusDevice>) {
        self.io.connect_widget(widget, dev);
    }
    /// A read arriving from widget `source`'s device `device` (`None`: every
    /// master on the widget must be admitted): memory below bit 47.
    pub fn inbound_read(&self, source: u8, device: Option<u8>, address: u64, size: u8) -> BusRead64 {
        self.inbound(source, device, address, size, None)
    }
    /// A write arriving from widget `source`: memory below bit 47, or an
    /// interrupt packet to a PI's pending-modify register in this node's
    /// window. A packet the PI refuses answers a bus error.
    pub fn inbound_write(&self, source: u8, device: Option<u8>, address: u64, size: u8, value: u64) -> u32 {
        self.inbound(source, device, address, size, Some(value)).status
    }
    fn inbound(&self, widget: u8, device: Option<u8>, address: u64, size: u8, value: Option<u64>) -> BusRead64 {
        if address >= (1 << 48) || u64::from(size) > (1 << 48) - address || !self.io.admits(widget, device) {
            return BusRead64::err();
        }
        if address & (1 << 47) == 0 {
            return io::access(self.dma.as_ref(), address, size, value);
        }
        let node = ((address >> 39) & 255) as u8;
        let window = address & ((1 << 39) - 1);
        if node != self.node.load(Ordering::Acquire) || window & !0x7f_ffff != 0x180_0000 {
            return BusRead64::err();
        }
        // XIO register packets reach only the two PIs' pending-modify registers.
        match (window & 0x7f_ffff, size, value) {
            (offset @ (0x90 | 0x20_0090), 8, Some(value)) if self.interrupt(usize::from(offset >= reg::PI1_BASE), value) => {
                BusRead64 { status: BUS_OK, data: 0 }
            }
            _ => BusRead64::err(),
        }
    }
    /// An interrupt packet to PI `pi`'s pending-modify register; whether the
    /// PI took it.
    fn interrupt(&self, pi: usize, value: u64) -> bool {
        self.pi[pi].io_modify(value, 0, &self.levels)
    }
    /// Widget `widget`'s DMA as a bus device: crosstalk addresses, with the
    /// widget's inbound permissions applied.
    pub fn widget_dma(self: &Arc<Self>, widget: u8) -> Arc<dyn BusDevice> {
        Arc::new(WidgetDma { chip: self.weak.clone(), widget })
    }

    // Processors.

    /// Where CPU `cpu`'s IP2..IP6 go: Cause bits 10-14 of its interrupt word
    /// (`MipsCpu::interrupts_ptr`), from the slot the board's population
    /// gives it. The chip writes no other bit, and no other device may write
    /// these. The word must outlive the chip.
    pub fn set_cpu_interrupts(&self, cpu: usize, word: *const AtomicU64) {
        if let Some((interface, slot)) = self.processors.slot_of(cpu) {
            let pi = &self.pi[usize::from(interface)];
            pi.set_word(slot, word);
            pi.publish(&self.levels);
        }
    }
    /// The node's processors, which a bank configuration write parks
    /// while it moves memory under their cached views.
    pub fn set_world(&self, world: Arc<crate::cpu::smp::World>) {
        self.md.set_world(world);
    }
    /// The PIs' comparators and profile timers fire on `tm`. Held weakly.
    pub fn set_timer_manager(&self, tm: &Arc<TimerManager>) {
        let _ = self.timers.set(Arc::downgrade(tm));
        let now = (self.clock)();
        let host = self.host_timer(tm);
        for pi in &self.pi {
            pi.reschedule(Some(&host), now);
        }
    }
    /// The L1 UART's line: bank 0 bit 4 of both PIs.
    pub fn uart_irq(self: &Arc<Self>) -> IrqLine {
        IrqLine::new(Arc::new(Uart(self.weak.clone())), 0)
    }
    fn host_timer<'a>(&'a self, tm: &'a TimerManager) -> HostTimer<'a> {
        HostTimer { tm, chip: &self.weak }
    }
    fn timer_manager(&self) -> Option<Arc<TimerManager>> {
        self.timers.get().and_then(Weak::upgrade)
    }
    /// PI `interface`'s host timer fired.
    fn expire(&self, interface: usize) {
        let Some(tm) = self.timer_manager() else { return };
        self.pi[interface].expire(&self.host_timer(&tm), (self.clock)(), &self.levels);
    }
    fn publish_all(&self) {
        for pi in &self.pi {
            pi.publish(&self.levels);
        }
    }
    fn apply(&self, effect: Effect) {
        match effect {
            Effect::None => {}
            Effect::Reflect => self.publish_all(),
            Effect::Bte(n) => {
                let node = self.node.load(Ordering::Acquire);
                if let Some(done) = self.io.run_bte(n, self.dma.as_ref(), node) {
                    if !self.interrupt(done.pi, done.value) {
                        crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO BTE completion interrupt failed");
                    }
                }
            }
            Effect::Reset(cause) => {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "{}: {cause:?} reset from CPU {:?}; restarting the node",
                    self.name, crate::cpu::smp::on_cpu_thread());
                if let Some(events) = &self.events {
                    let _ = events.send(MachineEvent::HardReset);
                }
            }
        }
    }

    /// The PI a local-form access at offset 0 reaches: the requester's own.
    /// A reader that is no processor reaches PI0.
    fn own_pi(&self) -> usize {
        crate::cpu::smp::on_cpu_thread()
            .and_then(|cpu| self.processors.slot_of(cpu))
            .map_or(0, |(interface, _)| usize::from(interface))
    }

    /// An access of `size` bytes at `offset` in the register window, a write
    /// when `value` is given; PI0's range reaches PI `pi0`. What no unit
    /// answers, the padding after a unit's registers included, is a bus error.
    fn access(&self, pi0: usize, offset: u64, size: u8, value: Option<u64>) -> BusRead64 {
        if !self.md.cell_permits() {
            return BusRead64::err();
        }
        match offset {
            reg::PI0_BASE..reg::PI1_BASE => self.pi_access(pi0, offset, size, value),
            reg::PI1_BASE..reg::IO_BASE => self.pi_access(1, offset - reg::PI1_BASE, size, value),
            reg::IO_BASE..reg::MD_BASE => self.block_access(offset, size, value),
            reg::MD_BASE..reg::WINDOW => {
                let result = self.md.access(View::Registers, offset - reg::MD_BASE, u64::from(size), value);
                result.map_or(BusRead64::err(), BusRead64::ok)
            }
            _ => BusRead64::err(),
        }
    }
    fn pi_access(&self, index: usize, a: u64, size: u8, value: Option<u64>) -> BusRead64 {
        let pi = &self.pi[index];
        let now = (self.clock)();
        match (size, value) {
            (8, None) if a < pi::SIZE && a & 7 == 0 => {
                pi.read(a, now, &self.levels).map_or(BusRead64::err(), BusRead64::ok)
            }
            (8, Some(v)) if a < pi::SIZE && a & 7 == 0 => {
                if a == pi::reg::RT_COMPARE_A || a == pi::reg::RT_COMPARE_B {
                    let slot = (index as u8, usize::from(a == pi::reg::RT_COMPARE_B));
                    if let Some(cpu) = crate::cpu::smp::on_cpu_thread().filter(|&cpu| self.processors.slot_of(cpu) == Some(slot)) {
                        self.lb.note_rt_compare(cpu);
                    }
                }
                if pi.write(a, v, now, &self.levels) {
                    let tm = self.timer_manager();
                    pi.reschedule(tm.as_deref().map(|tm| self.host_timer(tm)).as_ref(), now);
                }
                BusRead64 { status: BUS_OK, data: 0 }
            }
            // The counter's two halves, high word first.
            (4, None) if a & !4 == pi::reg::COUNTER => {
                BusRead64::ok(pi.count(now) >> (32 - (a & 4) * 8) & 0xffff_ffff)
            }
            _ => {
                pi.note(a, value.is_some());
                BusRead64::err()
            }
        }
    }
    /// An access to the II, LB, NI or XB at `offset` in the window: aligned
    /// doublewords only.
    fn block_access(&self, offset: u64, size: u8, value: Option<u64>) -> BusRead64 {
        let (base, end) = match offset {
            reg::IO_BASE..reg::LB_BASE => (reg::IO_BASE, io::SIZE),
            reg::LB_BASE..reg::NI_BASE => (reg::LB_BASE, lb::SIZE),
            reg::NI_BASE..reg::XB_BASE => (reg::NI_BASE, ni::SIZE),
            _ => (reg::XB_BASE, xb::SIZE),
        };
        let a = offset - base;
        if size != 8 || a & 7 != 0 || a >= end {
            return BusRead64::err();
        }
        let now = (self.clock)();
        let (data, effect) = match (base, value) {
            (reg::IO_BASE, None) => (self.io.read(a), Effect::None),
            (reg::IO_BASE, Some(v)) => (0, self.io.write(a, v)),
            (reg::LB_BASE, None) => (self.lb.read(a, now, crate::cpu::smp::on_cpu_thread()), Effect::None),
            (reg::LB_BASE, Some(v)) => (0, self.lb.write(a, v, now, crate::cpu::smp::on_cpu_thread(), &self.levels, &self.node)),
            (reg::NI_BASE, None) => self.ni.read(a, &self.levels),
            (reg::NI_BASE, Some(v)) => (0, self.ni.write(a, v, &self.levels)),
            (_, None) => (self.xb.read(a), Effect::None),
            (_, Some(v)) => (0, self.xb.write(a, v, &self.levels)),
        };
        self.apply(effect);
        BusRead64::ok(data)
    }
}

impl Device for Bedrock {
    fn step(&self, _cycles: u64) {}
    /// The counters run and the PIs' timers arm.
    fn start(&self) {
        let now = (self.clock)();
        self.md.start_clock();
        let tm = self.timer_manager();
        let host = tm.as_deref().map(|tm| self.host_timer(tm));
        for pi in &self.pi {
            pi.start(host.as_ref(), now);
        }
        self.publish_all();
    }
    /// The counters pause where they are and the PIs' timers stop.
    fn stop(&self) {
        let now = (self.clock)();
        let tm = self.timer_manager();
        let host = tm.as_deref().map(|tm| self.host_timer(tm));
        for pi in &self.pi {
            pi.stop(host.as_ref(), now, &self.levels);
        }
        self.md.stop_clock();
    }
    fn is_running(&self) -> bool {
        self.pi[0].is_running()
    }
    fn get_clock(&self) -> u64 {
        self.pi[0].count((self.clock)())
    }
}

impl Resettable for Bedrock {
    /// Every unit's reset. Memory keeps its contents.
    fn power_on(&self) {
        let now = (self.clock)();
        let tm = self.timer_manager();
        let host = tm.as_deref().map(|tm| self.host_timer(tm));
        for pi in &self.pi {
            pi.power_on(now, &self.levels);
            pi.reschedule(host.as_ref(), now);
        }
        self.md.power_on();
        self.lb.power_on(&self.node);
        self.io.power_on();
        let _ = self.ni.power_on(&self.levels);
        let _ = self.xb.power_on(&self.levels);
        self.publish_all();
    }
}

/// The register window. With `LOCAL`, PI0's range answers the requester's
/// own PI.
struct Registers<const LOCAL: bool>(Arc<Bedrock>);
impl<const LOCAL: bool> Registers<LOCAL> {
    #[inline]
    fn access(&self, offset: u64, size: u8, value: Option<u64>) -> BusRead64 {
        let pi0 = if LOCAL { self.0.own_pi() } else { 0 };
        self.0.access(pi0, offset, size, value)
    }
}
macro_rules! face_access {
    ($r:ident, $w:ident, $ty:ty, $result:ident, $size:expr) => {
        fn $r(&self, a: u64) -> $result {
            let r = self.access(a, $size, None);
            $result { data: r.data as $ty, status: r.status }
        }
        fn $w(&self, a: u64, v: $ty) -> u32 {
            self.access(a, $size, Some(v as u64)).status
        }
    };
}
impl<const LOCAL: bool> BusDevice for Registers<LOCAL> {
    face_access!(read8, write8, u8, BusRead8, 1);
    face_access!(read16, write16, u16, BusRead16, 2);
    face_access!(read32, write32, u32, BusRead32, 4);
    face_access!(read64, write64, u64, BusRead64, 8);
}

/// The II's PIO face.
struct Pio(Arc<Bedrock>);
impl Pio {
    #[inline]
    fn access(&self, a: u64, size: u8, value: Option<u64>) -> BusRead64 {
        self.0.io.pio(a, size, value)
    }
}
impl BusDevice for Pio {
    face_access!(read8, write8, u8, BusRead8, 1);
    face_access!(read16, write16, u16, BusRead16, 2);
    face_access!(read32, write32, u32, BusRead32, 4);
    face_access!(read64, write64, u64, BusRead64, 8);
}

/// A widget's DMA as inbound traffic. Weak: the widget's device may hold
/// it, and the chip holds the widget.
struct WidgetDma {
    chip: Weak<Bedrock>,
    widget: u8,
}
impl WidgetDma {
    #[inline]
    fn access(&self, a: u64, size: u8, value: Option<u64>) -> BusRead64 {
        match self.chip.upgrade() {
            Some(chip) => chip.inbound(self.widget, None, a, size, value),
            None => BusRead64::err(),
        }
    }
}
impl BusDevice for WidgetDma {
    face_access!(read8, write8, u8, BusRead8, 1);
    face_access!(read16, write16, u16, BusRead16, 2);
    face_access!(read32, write32, u32, BusRead32, 4);
    face_access!(read64, write64, u64, BusRead64, 8);
}

/// The L1 UART's interrupt line.
struct Uart(Weak<Bedrock>);
impl InterruptController for Uart {
    fn set_line(&self, _: u32, level: bool) {
        if let Some(chip) = self.0.upgrade() {
            if chip.levels.set(Levels::UART, level) {
                chip.publish_all();
            }
        }
    }
}
