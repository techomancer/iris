//! The Bedrock node every IP35 board shares: the processor bus with its four
//! views, the two PIs, MD and its memory, the L1, the boot flash and the
//! node's I/O, local-bus, network and crossbar blocks, wired together and to
//! the running CPU, and the board's serial ports. A board names its inputs in
//! [`NodeConfig`] and adds its own XIO widgets.
//!
//! # What is approximated here
//!
//! The two Hub-special head windows return zero on reads and acknowledge
//! writes without side effects. PROM 6.210 writes the 64-bit value 0x2000
//! at relative offsets 0 and 0x40 before DRAM discovery. Their hardware
//! meaning is unknown; Bedrock logging records accesses to these windows.
//!
//! Node composition ported from hw/mips/sgi/sn1-node.c in
//! jamesbraid/qemu-sgi-origami.

use super::{map_device, Ip35Bus, SparseWindowMap, Window};
use crate::config::MachineConfig;
use crate::cpu::mips_exec::CpuDevice;
use crate::dev::bedrock_flash::{FirmwareRecord, Flash, PART_SIZE};
use crate::dev::bedrock_io::{BedrockIo, SIZE as IO_SIZE};
use crate::dev::bedrock_lb::{BedrockLb, SIZE as LB_SIZE};
use crate::dev::bedrock_md::{BedrockMd, MemoryError};
use crate::dev::bedrock_ni::{BedrockNi, SIZE as NI_SIZE};
use crate::dev::bedrock_pi::{BedrockPi, BedrockPiConfig, Population, SIZE as PI_SIZE};
use crate::dev::bedrock_xb::{BedrockXb, SIZE as XB_SIZE};
use crate::dev::heart::Clock;
use crate::dev::l1::{L1, L1Config, L1Event};
use crate::dev::odyssey::{pi_gfx_mark_store, pi_gfx_unmark_store};
use crate::dev::serial_host::{null_serial_backend, open_log, tcp_serial_backend, CiSerialBackend, LogMode, SerialBackend, TeeBackend};
use crate::devlog::LogModule;
use crate::irq::{InterruptController, IrqLine};
use crate::ppmem::devmap::SLOT_SHIFT;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, MachineEvent, Signal, BUS_BUSY, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::path::Path;
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, OnceLock};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

/// Give a node's memory and devices to its running CPU: the direct-access
/// window, interrupts from PI0 slot A, and the resets and power events the
/// local bus, network block and L1 raise. The events arrive on the receiver
/// the system controller takes.
pub(crate) fn connect_cpu(
    name: &'static str,
    md: &Arc<BedrockMd>,
    pi0: &BedrockPi,
    lb: &BedrockLb,
    ni: &BedrockNi,
    l1: &L1,
    cpu: &Arc<dyn CpuDevice>,
) -> Receiver<MachineEvent> {
    connect(name, md, &[pi0], lb, ni, l1, Population::SINGLE, std::slice::from_ref(cpu))
}

/// Give a node's memory and devices to the processors the board runs, CPU 0
/// first, each in the slot `running` gives it: the direct-access window and
/// its page bitmaps for every one, each PI's interrupt outputs to the
/// processors in its slots, and the node's resets and power events, an L1
/// reset reaching every processor.
pub(crate) fn connect_cpus(name: &'static str, node: &Node, running: Population, cpus: &[Arc<dyn CpuDevice>]) -> Receiver<MachineEvent> {
    connect(name, &node.md, &[node.pi[0].as_ref(), node.pi[1].as_ref()], &node.lb, &node.ni, &node.l1, running, cpus)
}

/// [`connect_cpus`] over the PIs given, PI0 first.
#[allow(clippy::too_many_arguments)]
fn connect(
    name: &'static str,
    md: &Arc<BedrockMd>,
    pis: &[&BedrockPi],
    lb: &BedrockLb,
    ni: &BedrockNi,
    l1: &L1,
    running: Population,
    cpus: &[Arc<dyn CpuDevice>],
) -> Receiver<MachineEvent> {
    if let Some(space) = md.physical().ppmem_space() {
        use crate::ppmem::MappedMemory;
        unsafe {
            for (id, cpu) in cpus.iter().enumerate() {
                if id == 0 {
                    space.set_bitmap_sink(cpu.ppmem_bitmap_ptr());
                } else {
                    space.add_bitmap_sink(cpu.ppmem_bitmap_ptr());
                }
                cpu.set_tcache_window(space.window_base());
                let bitmap = cpu.tcache_bitmap_ptr();
                if !bitmap.is_null() {
                    if id == 0 { space.set_bitmap_sink2(bitmap) } else { space.add_bitmap_sink(bitmap) }
                }
                #[cfg(feature = "jitv2")]
                cpu.set_tcache_gen_window(md.generation_base());
            }
        }
        md.physical().resync_ppmem_bitmap();
        md.rebuild_memory();
    }
    for (interface, pi) in pis.iter().enumerate() {
        let slot = |slot| running.cpu_in(interface as u8, slot).and_then(|cpu| cpus.get(cpu)).map(Arc::downgrade);
        pi.set_interrupt_controller(Arc::new(SlotIrq([slot(0), slot(1)])));
    }
    node_events(name, lb, ni, l1, cpus.iter().map(Arc::downgrade).collect())
}

/// The resets and power events a node's local bus, network block and L1
/// raise, on the receiver the system controller takes. An L1 soft reset
/// reaches each of `cpus`.
fn node_events(name: &'static str, lb: &BedrockLb, ni: &BedrockNi, l1: &L1, cpus: Vec<std::sync::Weak<dyn CpuDevice>>) -> Receiver<MachineEvent> {
    let (tx, rx) = channel();
    let reset_tx = tx.clone();
    lb.connect_soft_reset(Arc::new(move || {
        let _ = reset_tx.send(MachineEvent::HardReset);
    }));
    let reset_tx = tx.clone();
    ni.connect_local_reset(Arc::new(move || {
        let _ = reset_tx.send(MachineEvent::HardReset);
    }));
    l1.set_event_handler(Arc::new(move |event| {
        match event {
            L1Event::SoftReset => {
                for cpu in cpus.iter().filter_map(|cpu| cpu.upgrade()) {
                    cpu.as_device().signal(Signal::Reset(true));
                }
            }
            L1Event::Nmi => dlog!(LogModule::L1, "{name}: NMI entry is not implemented by the CPU"),
            _ => {
                if let Some(action) = machine_action(event) {
                    let _ = tx.send(action);
                }
            }
        }
    }));
    rx
}

/// What the machine does for an L1 power or reset event. A power change
/// alone does nothing: the shutdown or restart that the L1 reports with it
/// carries the action. So a power cycle, which the PROM asks for to retry
/// a processor, restarts the node rather than powering the machine off, as
/// in the QEMU L1, where only a shutdown stops the machine.
fn machine_action(event: L1Event) -> Option<MachineEvent> {
    match event {
        L1Event::Shutdown => Some(MachineEvent::PowerOff),
        L1Event::Reset => Some(MachineEvent::HardReset),
        L1Event::Power(_) | L1Event::SoftReset | L1Event::Nmi => None,
    }
}

/// One of a board's serial connections: what it is called, the TCP port and
/// the setting that names the port, the CI backend it uses when it is the
/// console, the log it keeps when it is not, and where the host end goes.
pub(crate) struct SerialPort<'a> {
    pub label: &'static str,
    pub port: u16,
    pub setting: &'static str,
    pub ci: &'a Arc<CiSerialBackend>,
    pub log: Option<String>,
    pub connect: Box<dyn Fn(Arc<dyn SerialBackend>) + 'a>,
}

/// Connect a board's serial ports to the host. The console takes
/// `serial_log` and, under CI, the in-process backend; the others listen on
/// their TCP ports, and a port that is taken leaves its stream unconnected
/// rather than stopping the machine.
pub(crate) fn connect_serial_ports(
    cfg: &MachineConfig,
    name: &str,
    ports: &[SerialPort<'_>],
    console: usize,
) -> Result<(), String> {
    for (i, a) in ports.iter().enumerate() {
        if let Some(b) = ports[i + 1..].iter().find(|b| b.port == a.port) {
            return Err(format!("{name} {} and {} serial ports must be distinct", a.label, b.label));
        }
    }
    let mut log_config = cfg.clone();
    for (index, port) in ports.iter().enumerate() {
        log_config.serial_log = if index == console { cfg.serial_log.clone() } else { port.log.clone() };
        let tag = format!("{name} {}", port.label);
        let log = open_log(&log_config, &tag, LogMode::Fresh);
        let backend: Arc<dyn SerialBackend> = if cfg.ci && index == console {
            if let Some(log) = log {
                port.ci.set_log(log);
            }
            port.ci.clone()
        } else {
            let mut backend = null_serial_backend();
            match tcp_serial_backend(port.port) {
                Some((socket, actual_port)) => {
                    eprintln!("iris: {tag} on 127.0.0.1:{actual_port}");
                    backend = socket;
                }
                None => eprintln!(
                    "iris: {tag}: port {} unavailable; is another iris using it? ({})",
                    port.port, port.setting
                ),
            }
            match log {
                Some(log) => Arc::new(TeeBackend::new(backend, log)),
                None => backend,
            }
        };
        (port.connect)(backend);
    }
    Ok(())
}

/// The local node register blocks projected into the processor's I/O view.
pub mod reg {
    pub const PI1_BASE: u64 = 0x20_0000;
    pub const IO_BASE: u64 = 0x40_0000;
    pub const LB_BASE: u64 = 0x60_0000;
    pub const NI_BASE: u64 = 0x68_0000;
    pub const XB_BASE: u64 = 0x70_0000;
    pub const MD_BASE: u64 = 0x78_0000;
    pub const MD_SIZE: u64 = 0x8_0000;
}

/// What a board gives its Bedrock node.
pub(crate) struct NodeConfig<'a> {
    /// The board's name in logs and map names.
    pub name: &'static str,
    pub payload: &'a [u8],
    pub clock: Clock,
    pub flash_path: Option<&'a Path>,
    /// The memory banks in MiB, locational bank 0 upward.
    pub banks_mib: &'a [u32],
    /// The processor slots the board fits.
    pub population: Population,
    /// The fitted slots whose processor runs.
    pub running: Population,
    pub firmware: FirmwareRecord,
    /// The word the board reads back at the Hub-special board-id window.
    pub board_id: u64,
    pub l1: L1Config,
}

/// One Bedrock node as the IP35 boards share it: the processor bus with its
/// four views, the two PIs, MD with its memory, the L1, the boot flash and
/// the node's I/O, local-bus, network and crossbar blocks, wired together.
/// The board adds its XIO widgets to `io` and its serial backends to `l1`.
pub(crate) struct Node {
    pub bus: Arc<Ip35Bus>,
    pub pi: [Arc<BedrockPi>; 2],
    pub md: Arc<BedrockMd>,
    pub l1: Arc<L1>,
    pub flash: Arc<Flash>,
    pub lb: Arc<BedrockLb>,
    pub io: Arc<BedrockIo>,
    pub ni: Arc<BedrockNi>,
    pub xb: Arc<BedrockXb>,
    /// Set once by a board that counts graphics stores.
    io_store_counter: Arc<OnceLock<Box<dyn StoreCounter>>>,
    #[cfg(test)]
    pub irq: Arc<PendingIrq>,
}

impl Node {
    /// Count the processor's uncached I/O stores with `counter`, as a board
    /// with graphics credits does. Once, before the processor runs.
    pub fn count_io_stores(&self, counter: Box<dyn StoreCounter>) {
        assert!(self.io_store_counter.set(counter).is_ok(), "node I/O store counter already set");
    }

    pub fn build(config: NodeConfig<'_>) -> Result<Self, String> {
        let NodeConfig { name, payload, clock, flash_path, banks_mib, population, running, firmware, board_id, l1 } = config;
        let md = BedrockMd::with_clock(banks_mib, clock.clone())?;
        let phys = md.physical().clone();
        let pi: [Arc<BedrockPi>; 2] = std::array::from_fn(|interface| {
            BedrockPi::new(
                BedrockPiConfig {
                    interface: interface as u8,
                    request_region: 0,
                    population,
                    running,
                },
                clock.clone(),
            )
        });
        let l1 = Arc::new(L1::new(l1)?);
        let flash = Arc::new(match flash_path {
            Some(path) => Flash::with_file(payload, path, firmware),
            None => Flash::new(payload, firmware),
        }.map_err(|e| e.to_string())?);
        let lb = Arc::new(BedrockLb::new(clock.clone()));
        let io = Arc::new(BedrockIo::new());
        let ni = Arc::new(BedrockNi::new());
        let xb = Arc::new(BedrockXb::new());
        // A processor on PI1 finds its own interface at the local form's
        // offset 0, as QEMU's node overlays it there (OpenBSD expects the
        // local form to select the requester's interface). That the chip
        // answers PI1 there is an approximation pending a hardware read.
        let local_pi: Arc<dyn BusDevice> = if (0..2).any(|slot| running.fitted(1, slot)) {
            Arc::new(LocalPi { pi: pi.clone(), running })
        } else {
            pi[0].clone()
        };
        let mut registers = SparseWindowMap::default();
        registers.map(
            0,
            PI_SIZE,
            pi[0].clone(),
        );
        registers.map(
            reg::PI1_BASE,
            PI_SIZE,
            pi[1].clone(),
        );
        registers.map(
            reg::IO_BASE,
            IO_SIZE,
            io.clone(),
        );
        registers.map(
            reg::LB_BASE,
            LB_SIZE,
            lb.clone(),
        );
        registers.map(
            reg::NI_BASE,
            NI_SIZE,
            ni.clone(),
        );
        registers.map(
            reg::XB_BASE,
            XB_SIZE,
            xb.clone(),
        );
        registers.map(
            reg::MD_BASE,
            reg::MD_SIZE,
            md.md_bus(),
        );
        let mut local = SparseWindowMap::default();
        local.map(0, PI_SIZE, local_pi);
        local.map(reg::PI1_BASE, PI_SIZE, pi[1].clone());
        local.map(reg::IO_BASE, IO_SIZE, io.clone());
        local.map(reg::LB_BASE, LB_SIZE, lb.clone());
        local.map(reg::NI_BASE, NI_SIZE, ni.clone());
        local.map(reg::XB_BASE, XB_SIZE, xb.clone());
        local.map(reg::MD_BASE, reg::MD_SIZE, md.md_bus());
        let registers = Arc::new(registers);
        let local = Arc::new(local);
        io.connect_hub(Arc::new(IoHub(pi.clone())));
        io.connect_memory(md.dma_view());
        let p = pi.clone();
        lb.connect_error(Arc::new(move |level| {
            for pi in &p {
                pi.set_lb_error(level);
            }
        }));
        let p = pi.clone();
        ni.connect_error(Arc::new(move |level| {
            for pi in &p {
                pi.set_ni_error(level);
            }
        }));
        let p = pi.clone();
        xb.connect_error(Arc::new(move |level| {
            for pi in &p {
                pi.set_xb_error(level);
            }
        }));
        let io_node = Arc::downgrade(&io);
        lb.connect_node_changed(Arc::new(move |node| {
            if let Some(io) = io_node.upgrade() {
                io.set_node_number(node);
            }
        }));
        let p = pi.clone();
        md.on_error(Arc::new(move |kind| {
            for pi in &p {
                match kind {
                    MemoryError::Correctable => pi.set_pending(1, 61),
                    MemoryError::Uncorrectable => {
                        pi.set_memory_uncorrectable();
                    }
                }
            }
        }));
        #[cfg(test)]
        let irq = Arc::new(PendingIrq::default());
        #[cfg(test)]
        pi[0].set_interrupt_controller(irq.clone());
        l1.set_irq(IrqLine::new(Arc::new(UartIrq(pi.clone())), 0));
        let mut bus = Ip35Bus::new(phys.clone());
        #[cfg(feature = "jitv2")]
        bus.set_ram_generations(md.dma_view());
        md.rebuild_memory();
        let io_store_counter = Arc::new(OnceLock::new());
        bus.set_fallback(1, 1 << 32, Arc::new(IoPio { io: io.clone(), counter: io_store_counter.clone() }));
        // The two XIO register forms select the requester's PI0 interface.
        let permission_md = Arc::downgrade(&md);
        let permission: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || {
            permission_md.upgrade().is_some_and(|md| md.current_cell() & 1 != 0)
        });
        let protected_regs = Arc::new(Window::new(0, 0x80_0000, registers.clone()).permitted(permission.clone()));
        let protected_local = Arc::new(Window::new(0, 0x80_0000, local.clone()).permitted(permission.clone()));
        bus.map(1, 0x0100_0000, 0x80_0000, protected_local);
        bus.map(1, 0x0180_0000, 0x80_0000, protected_regs.clone());
        // Cached RAM uses MD's direct bank mappings; diagnostic views stay separate.
        bus.map(3, 0, 1 << 33, md.uncached_view());
        // Hub-special low memory: its first 256 MiB is the node memory view.
        bus.map(0, 0, 0x1000_0000, md.uncached_view());
        let prom = Arc::new(
            flash
                .array_view(0, PART_SIZE)
                .map_err(|e| e.to_string())?,
        );
        let boot = Arc::new(flash.boot_view());
        for base in [0x1800_0000, 0x3800_0000] {
            bus.map(0, base, 0x800_0000, boot.clone());
        }
        // Reset execution always sees stored bytes, independently of commands.
        bus.map(0, 0x1fc0_0000, 0x40_0000, prom.clone());
        if !payload.is_empty() {
            map_device(
                &phys,
                0x1fc0_0000,
                payload.len() as u64,
                prom.clone(),
                &format!("{name} cached reset PROM"),
            );
        }
        let leds = Arc::new(Leds {
            md: md.clone(),
            values: Mutex::new([0; 4]),
        });
        // The three windows occupy the same processor register page.
        let special = Arc::new(HubSpecialPage {
            windows: [
                Window::new(0, 0x80, Arc::new(HubHead(name))).permitted(permission.clone()),
                Window::new(0x80, 0x40, l1.clone()).permitted(permission.clone()),
                Window::new(0xc0, 0x20, leds).permitted(permission.clone()),
            ],
        });
        for base in [0x1000_0000, 0x3000_0000] {
            bus.map(0, base, 1 << SLOT_SHIFT, special.clone());
            bus.map(0, base + 0x400_0000, 8, Arc::new(BoardId { md: md.clone(), word: board_id }));
        }
        bus.map(0, 0x1_c000_0000, 1 << 30, md.directory_bus());
        bus.map(0, 0x1_4000_0000, 1 << 30, md.ecc_bus());
        bus.map(0, 0x1_0000_0000, 1 << 30, md.ecc_scrub_bus());
        bus.map(2, 0, 1 << 33, md.fetchop_bus());
        Ok(Self {
            bus: Arc::new(bus),
            io_store_counter,
            pi,
            md,
            l1,
            flash,
            lb,
            io,
            ni,
            xb,
            #[cfg(test)]
            irq,
        })
    }
}

/// The local register form's PI: the requester's own interface. A reader
/// that is no processor reaches PI0.
struct LocalPi {
    pi: [Arc<BedrockPi>; 2],
    running: Population,
}
impl LocalPi {
    fn own(&self) -> &BedrockPi {
        let interface = crate::cpu::smp::on_cpu_thread().and_then(|cpu| self.running.slot_of(cpu)).map_or(0, |(i, _)| i);
        &self.pi[usize::from(interface)]
    }
}
impl BusDevice for LocalPi {
    fn read64(&self, a: u64) -> BusRead64 {
        self.own().read64(a)
    }
    fn read32(&self, a: u64) -> BusRead32 {
        self.own().read32(a)
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        self.own().write64(a, v)
    }
}

/// A PI's interrupt outputs to the processors in its two slots: lines 0..4
/// are slot A's IP2..IP6, lines 5..9 slot B's, each to the processor the
/// board put there.
pub(crate) struct SlotIrq(pub [Option<std::sync::Weak<dyn CpuDevice>>; 2]);
impl InterruptController for SlotIrq {
    fn set_line(&self, line: u32, level: bool) {
        if let Some(cpu) = self.0.get(line as usize / 5).and_then(|c| c.as_ref()).and_then(|c| c.upgrade()) {
            cpu.as_device().signal(Signal::Interrupt(line % 5 + 2, level));
        }
    }
}

// XIO register packets reach only the two PI interrupt-modify registers.
// The processor's register projection has a separate, wider interface.
struct IoHub([Arc<BedrockPi>; 2]);
impl BusDevice for IoHub {
    fn write64(&self, a: u64, v: u64) -> u32 {
        if matches!(a, 0x90 | 0x20_0090) && self.0[(a >> 21) as usize].io_modify(v, 0) {
            BUS_OK
        } else {
            BUS_ERR
        }
    }
}

struct HubSpecialPage {
    windows: [Window; 3],
}
impl HubSpecialPage {
    fn target(&self, a: u64, n: u64) -> Option<&Window> {
        self.windows.iter().find(|window| window.contains(a, n))
    }
}
macro_rules! hub_page_access {
    ($r:ident,$w:ident,$t:ty,$out:ident,$n:expr) => {
        fn $r(&self, a: u64) -> $out {
            self.target(a, $n).map(|window| window.$r(a)).unwrap_or_else($out::err)
        }
        fn $w(&self, a: u64, value: $t) -> u32 {
            self.target(a, $n).map(|window| window.$w(a, value)).unwrap_or(BUS_ERR)
        }
    };
}
impl BusDevice for HubSpecialPage {
    hub_page_access!(read8, write8, u8, BusRead8, 1);
    hub_page_access!(read16, write16, u16, BusRead16, 2);
    hub_page_access!(read32, write32, u32, BusRead32, 4);
    hub_page_access!(read64, write64, u64, BusRead64, 8);
}

struct BoardId {
    md: Arc<BedrockMd>,
    word: u64,
}
impl BusDevice for BoardId {
    fn read64(&self, a: u64) -> BusRead64 {
        if a == 0 && self.md.current_cell() & 1 != 0 {
            BusRead64::ok(self.word)
        } else {
            BusRead64::err()
        }
    }
}
// Compatibility head windows: their register contract is not established.
// PROM 6.210 writes 0x2000 at offsets 0 and 0x40 before DRAM discovery.
// Retain the existing zero-read/acknowledged-write model, with accesses
// observable through the Bedrock log, until the register behavior is known.
/// The board name for the log.
struct HubHead(&'static str);
macro_rules! head_access {
    ($r:ident,$w:ident,$ty:ty,$result:ident) => {
        fn $r(&self, a: u64) -> $result {
            dlog!(LogModule::Bedrock,
                "{}: unidentified Hub-special read {} bytes at {a:#x}",
                self.0,
                std::mem::size_of::<$ty>()
            );
            $result::ok(0)
        }
        fn $w(&self, a: u64, v: $ty) -> u32 {
            dlog!(LogModule::Bedrock,
                "{}: unidentified Hub-special write {} bytes at {a:#x} value {v:#x}",
                self.0,
                std::mem::size_of::<$ty>()
            );
            BUS_OK
        }
    };
}
impl BusDevice for HubHead {
    head_access!(read8, write8, u8, BusRead8);
    head_access!(read16, write16, u16, BusRead16);
    head_access!(read32, write32, u32, BusRead32);
    head_access!(read64, write64, u64, BusRead64);
}
struct Leds {
    md: Arc<BedrockMd>,
    values: Mutex<[u8; 4]>,
}
impl BusDevice for Leds {
    fn read64(&self, a: u64) -> BusRead64 {
        if a < 32 && a & 7 == 0 && self.md.current_cell() & 1 != 0 {
            BusRead64::ok(self.values.lock()[a as usize / 8] as u64)
        } else {
            BusRead64::err()
        }
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        if a < 32 && a & 7 == 0 && self.md.current_cell() & 1 != 0 {
            self.values.lock()[a as usize / 8] = v as u8;
            BUS_OK
        } else {
            BUS_ERR
        }
    }
}

/// A widget's DMA as Bedrock's inbound path takes it: crosstalk addresses,
/// with the widget's inbound permissions applied.
pub(crate) struct WidgetDma {
    pub io: std::sync::Weak<BedrockIo>,
    pub widget: u8,
}
macro_rules! widget_dma_access {
    ($r:ident,$w:ident,$ty:ty,$result:ident,$size:expr) => {
        fn $r(&self, a: u64) -> $result {
            let Some(io) = self.io.upgrade() else { return $result::err() };
            let r = io.inbound_read(self.widget, None, a, $size);
            $result { data: r.data as $ty, status: r.status }
        }
        fn $w(&self, a: u64, v: $ty) -> u32 {
            self.io.upgrade().map_or(BUS_ERR, |io| io.inbound_write(self.widget, None, a, $size, v as u64))
        }
    };
}
impl BusDevice for WidgetDma {
    widget_dma_access!(read8, write8, u8, BusRead8, 1);
    widget_dma_access!(read16, write16, u16, BusRead16, 2);
    widget_dma_access!(read32, write32, u32, BusRead32, 4);
    widget_dma_access!(read64, write64, u64, BusRead64, 8);
}

/// What the processor interface does with a processor's uncached store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Store {
    /// Outside the slot's enabled graphics page.
    Uncounted,
    /// Counted; `below_bias` asks for the slot's graphics interrupt.
    Counted { below_bias: bool },
    /// The slot is stopped: the store waits, uncounted.
    Held,
}

/// The processor interface's count of uncached I/O stores against a
/// graphics page (see [`crate::dev::odyssey::PiGfx`]).
pub(crate) trait StoreCounter: Send + Sync {
    /// The store at node I/O offset `address`: uncounted, counted, or held
    /// and answered busy without being forwarded.
    fn count(&self, address: u64) -> Store;
    /// A counted store the bus answered busy did not happen: the processor
    /// will make it again, and pays for it then.
    fn refund(&self, address: u64);
}

/// The processor's uncached I/O view of the node. A board with graphics
/// counts its stores; the store is forwarded marked as counted or not.
struct IoPio {
    io: Arc<BedrockIo>,
    counter: Arc<OnceLock<Box<dyn StoreCounter>>>,
}
macro_rules! pio_access {
    ($r:ident,$w:ident,$ty:ty,$result:ident,$size:expr) => {
        fn $r(&self, a: u64) -> $result {
            let r = self.io.pio_read(a, $size);
            $result {
                data: r.data as $ty,
                status: r.status,
            }
        }
        fn $w(&self, a: u64, v: $ty) -> u32 {
            let counter = self.counter.get();
            let counted = match counter.map_or(Store::Uncounted, |c| c.count(a)) {
                Store::Held => return BUS_BUSY,
                Store::Counted { .. } => true,
                Store::Uncounted => false,
            };
            let previous = pi_gfx_mark_store(counted);
            let status = self.io.pio_write(a, $size, v as u64);
            pi_gfx_unmark_store(previous);
            if let (true, BUS_BUSY, Some(c)) = (counted, status, counter) {
                c.refund(a);
            }
            status
        }
    };
}
impl BusDevice for IoPio {
    pio_access!(read8, write8, u8, BusRead8, 1);
    pio_access!(read16, write16, u16, BusRead16, 2);
    pio_access!(read32, write32, u32, BusRead32, 4);
    pio_access!(read64, write64, u64, BusRead64, 8);
}
#[cfg(test)]
#[derive(Default)]
pub(crate) struct PendingIrq(AtomicU64);
#[cfg(test)]
impl PendingIrq {
    pub fn bits(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }
}
#[cfg(test)]
impl InterruptController for PendingIrq {
    fn set_line(&self, line: u32, level: bool) {
        if line >= 5 {
            return;
        }
        let mask = 1 << (10 + line);
        if level {
            self.0.fetch_or(mask, Ordering::Release);
        } else {
            self.0
                .fetch_and(!mask, Ordering::Release);
        }
    }
}
struct UartIrq([Arc<BedrockPi>; 2]);
impl InterruptController for UartIrq {
    fn set_line(&self, _: u32, level: bool) {
        for pi in &self.0 {
            pi.set_uart_irq(level);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test node with four processors fitted and `running` of them run.
    fn node(running: usize) -> Node {
        let population = Population::fitted_cpus(4).unwrap();
        Node::build(NodeConfig {
            name: "test",
            payload: &[],
            clock: Arc::new(|| 0),
            flash_path: None,
            banks_mib: &[512],
            population,
            running: Population::fitted_cpus(running).unwrap(),
            firmware: FirmwareRecord::TEZRO,
            board_id: 0,
            l1: L1Config::default(),
        })
        .unwrap()
    }

    /// The CPU-number register as processor `cpu` reads it through the XIO
    /// register form at `window`.
    fn cpu_number(node: &Arc<Node>, cpu: Option<u8>, window: u64) -> u64 {
        let node = node.clone();
        std::thread::spawn(move || {
            if let Some(cpu) = cpu {
                crate::cpu::smp::set_current_cpu(cpu);
            }
            node.bus.read64((1 << 57) | window | 0x20).data
        })
        .join()
        .unwrap()
    }

    /// Through the local form a processor reaches its own interface at
    /// offset 0, so every processor of four reads its own slice there; the
    /// node form keeps PI0 at 0 for everyone, and a reader that is no
    /// processor reaches PI0. Follows sn1-cpu-identity-test
    /// four-processors-know-their-slots, which reads the local form.
    #[test]
    fn the_local_form_answers_the_requesters_own_interface() {
        let node = Arc::new(node(4));
        let local: Vec<u64> = (0..4).map(|cpu| cpu_number(&node, Some(cpu), 0x0100_0000)).collect();
        assert_eq!(local, [0, 2, 1, 3]);
        let node_form: Vec<u64> = (0..4).map(|cpu| cpu_number(&node, Some(cpu), 0x0180_0000)).collect();
        assert_eq!(node_form, [0, 0, 1, 0], "PI0's register: only PI0's processors read their slot");
        assert_eq!(cpu_number(&node, None, 0x0100_0000), 0);
        // PI1 stays at its own offset in both forms.
        let pi1 = node.bus.read64((1 << 57) | 0x0100_0000 | reg::PI1_BASE | 0x20).data;
        assert_eq!(pi1, 2);
        // With PI0/A alone running, as on the Fuel, the local form is PI0.
        assert_eq!(cpu_number(&Arc::new(self::node(1)), Some(0), 0x0100_0000), 0);
    }

    /// The L1 reports a power cycle as power down, power up and a restart,
    /// and a shutdown as power down and a shutdown: one restart for the
    /// first and one power-off for the second, never a power-off on the way
    /// through a cycle.
    #[test]
    fn a_power_cycle_restarts_and_only_a_shutdown_powers_off() {
        let actions = |events: &[L1Event]| {
            format!("{:?}", events.iter().filter_map(|&e| machine_action(e)).collect::<Vec<_>>())
        };
        assert_eq!(actions(&[L1Event::Power(false), L1Event::Power(true), L1Event::Reset]), "[HardReset]");
        assert_eq!(actions(&[L1Event::Power(false), L1Event::Shutdown]), "[PowerOff]");
        assert_eq!(actions(&[L1Event::Power(true), L1Event::Reset]), "[HardReset]");
        assert_eq!(actions(&[L1Event::Reset]), "[HardReset]");
    }
}
