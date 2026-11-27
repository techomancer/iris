//! The IP27 processor bus: a node's memory, its uncached views and the Hub special windows.
//!
//! Two attribute bits beside an uncached address (XKPHYS bits 58:57) pick one of four spaces,
//! which [`crate::sn::UncachedBus`] indexes. Cached references, and uncached ones to memory
//! (attribute 3), use the node's `Physical` and its ppmem window; the Hub owns and remaps it.
//!
//! # What is approximated here
//!
//! - A node holding zero shows its memory at node-relative offsets. Once it holds another number it
//!   shows its memory there, keeps node zero's below the cached alias, and shows the memory of the
//!   node holding zero above it; on a two-node board each also shows the other's at its number.
//!   Each node reserves 4 GiB (`next_power_of_two(nodes) x 4 GiB` with two) of inaccessible host
//!   address space (`PROT_NONE` and `MAP_NORESERVE`, or a `VirtualAlloc2` placeholder), by reading
//!   the host code: only Linux was run.
//! - The low 128 KiB of the Hub-special memory view is not private to each processor of the pair.
//! - A 512 MiB bank 0 covers the boot flash's reset address: cached references reach the bank, the
//!   uncached reset fetch reaches the flash through the Hub-special view. That the hardware does
//!   the same is inferred (the PROM's memory test walks that address as RAM).
//! - A bank up to 128 MiB is one run at its 512 MiB stride; a larger one is four runs at 128 MiB.
//! - Empty memory fails every access with a logged bus error (a software expectation: the IRIX
//!   handler and SGI's event catalogue allow it, no capture shows it). The PROM's sizing reads the
//!   directory back door, which answers a fixed pattern for an absent bank, so boots log no access.
//!
//! # The machines
//!
//! One [`Machine<Hub, NodeFlash>`] serves every product: a Hub node with its memory, boot flash, module
//! controller and processors, and a board that supplies the I/O around it. A product is a
//! [`NodeSpec`] per node plus a `build_io` function handing back a [`BoardIo`]; the profile picks
//! both ([`crate::ip29_machine`] for the Origin 200, [`crate::ip27_machine`] for the desksides).
//!
//! - A node holds one or two processors, slices A and B; each node's memory is mapped by the Hubs' remap.
//! - The Hub's per-slice enable and NMI stores only log; their effect on a processor is not established.
//! - The node flash and uncached Hub-special views carry no JIT page generation (`gen_ptr`), so a
//!   jitv2 build compiles no code from them and the PROM runs on the interpreter's path.
//!
//! Ported from hw/mips/sgi/sn-machine.c and hw/mips/sgi/sn0.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::config::{MachineConfig, MachineProfile};
use crate::cpu::parts::CpuModule;
use crate::cpu::mips_cache_v2::MipsCache;
use crate::cpu::mips_exec::{CpuDevice, MipsCpu};
use crate::dev::bridge::Bridge;
use crate::dev::elsc::ModuleOutput;
use crate::clock::Clock;
use crate::dev::hub::{Hub, HubConfig};
use crate::sn::{BoardIo, Machine, NodeBoard, UncachedBus, UncachedSpace, RESET_WINDOW};
use crate::dev::ioc3::{ByteBus, Ioc3, Ioc3Board};
use crate::dev::isp1040::Isp1040;
use crate::dev::node_flash::{self, FlashValues, NodeFlash};
use crate::dev::prom_image;
use crate::dev::serial_host::CiSerialBackend;
use crate::dev::battery_clock::YearBase;
use crate::dev::timekeeper::Timekeeper;
use crate::dev::w1;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use crate::devlog::LogModule;
use crate::gdb_stub::CpuDebug;
use crate::hptimer::TimerManager;
use crate::cpu::smp::World;
use crate::ip35::presence::Cpu;
use crate::window_map::SparseWindowMap;
pub use crate::dev::hub::{bank_mappings, BANKS, BANK_STRIDE, SLOT_STRIDE};
use crate::traits::*;
use crate::workstation::{DiscChanger, Workstation};
use std::path::{Path, PathBuf};
use crate::traits::MachineEvents;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// The ranges the CPU model reports as memory for a node with `banks_mb`: the
/// installed runs, and the boot flash's reset address unless a 512 MiB bank 0 holds it.
pub fn memory_map(banks_mb: [u32; BANKS]) -> crate::memory_map::MemoryMap {
    use crate::memory_map::{MemKind, MemRange, MemoryMap};
    const NAMES: [&str; BANKS] = ["bank 0", "bank 1", "bank 2", "bank 3"];
    let end = RESET_WINDOW + crate::dev::node_flash::SIZE as u64;
    let runs = crate::dev::hub::installed_mappings(banks_mb);
    let mut ranges: Vec<_> = runs.iter().map(|r| MemRange::new(NAMES[(r.base / BANK_STRIDE) as usize], MemKind::Ram, r.base, r.end)).collect();
    if !runs.iter().any(|r| r.base < end && r.end > RESET_WINDOW) {
        ranges.push(MemRange::new("boot flash", MemKind::Rom, RESET_WINDOW, end));
    }
    MemoryMap::new(ranges)
}

/// The Hub-special space of a node: its own memory and boot flash and the
/// named node's, and the two back doors.
pub fn hub_special(memory: Arc<dyn BusDevice>, flash: &Arc<NodeFlash>, ecc: Arc<dyn BusDevice>, directory: Arc<dyn BusDevice>) -> Arc<dyn BusDevice> {
    const WINDOW: u64 = 256 << 20;
    const BACK_DOOR: u64 = 1 << 30;
    // Window bases: memory and boot flash at two each, then the ECC and directory back doors.
    const MEMORY_BASES: [u64; 2] = [0, 0x2000_0000];
    const FLASH_BASES: [u64; 2] = [0x1000_0000, 0x3000_0000];
    const ECC_BASE: u64 = 0x8000_0000;
    const DIRECTORY_BASE: u64 = 0xc000_0000;
    let mut map = SparseWindowMap::default();
    for base in MEMORY_BASES {
        map.map(base, WINDOW, memory.clone());
    }
    for base in FLASH_BASES {
        map.map(base, WINDOW, flash.window());
    }
    map.map(ECC_BASE, BACK_DOOR, ecc);
    map.map(DIRECTORY_BASE, BACK_DOOR, directory);
    Arc::new(map)
}

/// The PROM image tried when `prom` is left at the shared default, for all
/// three IP27 products.
pub const IP27_PROM_FILE: &str = "ip27prom.img";

/// A node board record's synthetic ROM serial, before the node index is ORed in.
const NODE_SERIAL_BASE: u64 = 0x5145_4d55_4e00;

/// The facts about a product the machine reads.
pub struct NodeSpec {
    pub hub: HubConfig,
    /// The processor module; its mode word is the flash block's.
    pub module: &'static CpuModule,
    /// What the flash configuration block carries for this product and module.
    pub firmware: FlashValues,
    /// The file that keeps the node's boot flash between runs; absent means volatile.
    pub flash: Option<PathBuf>,
    /// The record on the node's one-wire part, with the node's own barcode. None
    /// leaves the part unprogrammed.
    pub node_record: Option<BoardRecord<'static>>,
}

/// What the machine asks of the module controller a board wires.
pub trait ModuleController: Send + Sync {
    /// The module resets: the controller's bus side returns to power-on.
    fn reset(&self);
    /// The controller's outputs since the last call, with its time brought
    /// up to date.
    fn take_events(&self) -> Vec<ModuleOutput>;
    /// `wake` runs, with no controller lock held, on a timer thread once the controller has
    /// outputs for the machine. Once, when the machine is built.
    fn set_wake(&self, wake: Box<dyn Fn() + Send + Sync>);
    /// The device that starts and stops the controller's timers with the machine.
    fn device(&self) -> Arc<dyn Device>;
}

/// Run the module controller as the board's last device, handing its outputs to the machine as
/// its timers produce them.
pub(crate) fn attach_controller(io: &mut BoardIo, controller: Arc<dyn ModuleController>, events: &MachineEvents) {
    let (me, events) = (Arc::downgrade(&controller), events.clone());
    controller.set_wake(Box::new(move || {
        for event in me.upgrade().map(|c| c.take_events()).unwrap_or_default() {
            controller_output(event, &events);
        }
    }));
    io.devices.push(controller.device());
}

/// What the controller's outputs ask of the machine: power falling is power loss, rising or a
/// reset pulse a restart. The NMI output has no path to the processor yet and is only logged.
fn controller_output(event: ModuleOutput, machine: &MachineEvents) {
    let _ = match event {
        ModuleOutput::Power(false) => machine.send(MachineEvent::PowerOff),
        ModuleOutput::Power(true) | ModuleOutput::Reset => machine.send(MachineEvent::HardReset),
        ModuleOutput::Nmi(level) => {
            crate::dlog!(LogModule::Hub, "IP27: module controller NMI {level} (not delivered)");
            Ok(())
        }
    };
}

impl NodeBoard<Hub, NodeFlash> {
    /// Node `index`: its flash from `prom` and `spec`, the Hub and the
    /// node board's one-wire record, and the uncached bus its processor uses.
    pub fn build(table: &str, spec: &NodeSpec, index: usize, prom: &[u8], tm: &Arc<TimerManager>) -> Result<Self, String> {
        let image = node_flash::initial_image(prom, &spec.firmware)?;
        let flash = match &spec.flash {
            Some(path) => NodeFlash::with_file(image, path).map_err(|e| format!("machine.flash {}: {e}", path.display()))?,
            None => NodeFlash::new(image)?,
        };
        let hub = Hub::new(spec.hub, tm, Some(flash.chip()))?;
        // The Hub's one-wire master reads the node board's record, whose serial is the node's
        // identity: the other node's PROM reads it as the remote NIC.
        let mut onewire = w1::Bus::new();
        let barcode = format!("EMUN{index:02}");
        let serial = NODE_SERIAL_BASE | index as u64;
        match &spec.node_record {
            Some(r) => onewire.add(Box::new(board_record_part(&BoardRecord { barcode: &barcode, ..*r }, serial))),
            None => onewire.add(Box::new(w1::Ds2505::new(0x0b, serial, &[]))),
        }
        hub.md.set_onewire(onewire);
        hub.decode.set_space(
            UncachedSpace::HubSpecial,
            hub_special(hub.decode.own_memory(), &flash, hub.md.ecc_bus(), hub.md.directory_bus()),
        );
        hub.decode.set_space(UncachedSpace::Io, hub.io_space());
        // Uncached memory, attribute 3, is the node's decode, as cached is.
        let physical: Arc<dyn BusDevice> = hub.decode.physical().clone();
        let bus = Arc::new(UncachedBus::new([
            hub.cpu_view(UncachedSpace::HubSpecial),
            hub.processor_io(hub.cpu_view(UncachedSpace::Io)),
            Arc::new(crate::sn::FetchOp(hub.decode.physical().clone())),
            physical,
        ]));
        Ok(Self { chip: hub, flash, bus, first_cpu: spec.hub.first_cpu })
    }
}

/// A machine of `cfg`'s product: `node_specs` gives each node's facts and `build_io` builds the
/// board's I/O around them.
#[allow(clippy::type_complexity)]
pub fn machine(
    cfg: &MachineConfig,
    node_specs: fn(&MachineConfig) -> Result<Vec<NodeSpec>, String>,
    build_io: fn(&MachineConfig, &[NodeBoard<Hub, NodeFlash>], &Clock, &Arc<TimerManager>, &MachineEvents) -> Result<BoardIo, String>,
) -> Result<Machine<Hub, NodeFlash>, String> {
    cfg.validate()?;
    let prom = prom_image::resolve(cfg.machine.profile, &cfg.prom, node_flash::SIZE)
        .map_err(|e| format!("{} {e}", cfg.machine.profile.product_name().unwrap_or_default()))?
        .bytes;
    let specs = node_specs(cfg)?;
    let timers = Arc::new(TimerManager::new());
    let clock = timers.clock().clone();
    let mut nodes = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let node = NodeBoard::<Hub, NodeFlash>::build(cfg.machine.profile.table_name().unwrap_or_default(), spec, i, &prom, &timers)?;
        nodes.push(node);
    }
    let (tx, rx) = MachineEvents::channel(16);
    let io = build_io(cfg, &nodes, &clock, &timers, &tx)?;
    let maps: Vec<_> = specs.iter().map(|s| memory_map(s.hub.banks_mb)).collect();
    let memory = Arc::new(crate::sn::machine_memory(&maps, crate::dev::hub::NODE_SHIFT));

    // The Hubs are the memory controllers: a node number or alias store
    // remaps both nodes' memory with every other processor parked.
    if let [a, b] = &nodes[..] {
        a.chip.connect_peer(&b.chip);
        b.chip.connect_peer(&a.chip);
    }
    let world = Arc::new(World::new());
    tx.set_cpus(world.clone());
    for n in &nodes {
        n.chip.decode.set_world(world.clone());
    }
    let smp = specs.iter().map(|s| usize::from(s.hub.cpus)).sum::<usize>() > 1;
    // A node's processors, slice A first, numbered from the node's first.
    let mut node_cpus = Vec::new();
    for (node, spec) in nodes.iter().zip(&specs) {
        let mut cpus = Vec::new();
        for slice in 0..usize::from(spec.hub.cpus) {
            let cpu = build_cpu(cfg, spec, node, slice, &memory, &timers, smp)?;
            // The Hub publishes its lines straight into the processor's
            // interrupt word, which `Drop` lets go of before the processor goes.
            node.chip.pi.set_cpu_interrupts(slice, cpu.interrupts_ptr());
            cpu.register_locks();
            cpus.push(cpu);
        }
        if let [a, b] = &cpus[..] {
            let hold = |c: &Arc<dyn CpuDevice>| {
                let cycles = c.cycles_ptr();
                Cpu { slot: c.park_slot(), cycles: Box::new(move || cycles.retired()) }
            };
            node.chip.pi.set_pair([hold(a), hold(b)]);
        }
        node.chip.set_event_sender(tx.clone());
        node.chip.register_locks();
        node_cpus.push(cpus);
    }
    let cpus: Vec<Arc<dyn CpuDevice>> = node_cpus.iter().flatten().cloned().collect();
    world.set_slots(cpus.iter().map(|c| c.park_slot()).collect());
    // One JIT for the machine: both nodes' processors compile through one
    // queue into one page pool, each node an address space of it.
    #[cfg(feature = "jitv2")]
    {
        let pool: Vec<_> = nodes
            .iter()
            .zip(&specs)
            .zip(&node_cpus)
            .map(|((node, spec), cpus)| crate::cpu::jitv2::jitv2::PoolNode { cpus, module: spec.module, bus: node.chip.decode.physical().clone() })
            .collect();
        crate::cpu::jitv2::jitv2::share_pool(&pool, cfg.jitv2_pool_size(), cfg.jitv2.compile, &world)?;
    }
    eprintln!(
        "iris: {}, {} node{} of {} MiB, {} x {}",
        cfg.machine.profile.product_name().unwrap_or_default(),
        nodes.len(),
        if nodes.len() == 1 { "" } else { "s" },
        specs[0].hub.banks_mb.iter().sum::<u32>(),
        cpus.len(),
        specs[0].module.part.label(),
    );
    let memory = nodes.iter().map(|n| n.chip.decode.physical().clone() as Arc<dyn Device>).collect();
    Machine::new(cfg, nodes, Vec::new(), node_cpus, io, memory, timers, (tx, rx), |hub| {
        // Another Hub can still reach this one's processor.
        hub.pi.clear_cpu_interrupts();
        hub.decode.physical().release_devices();
    })
}

/// `smp` when the board carries more than one node: the processors then
/// share memory, and a store-conditional has to be a compare-and-swap.
fn build_cpu(
    cfg: &MachineConfig,
    spec: &NodeSpec,
    node: &NodeBoard<Hub, NodeFlash>,
    slice: usize,
    memory: &Arc<crate::memory_map::MemoryMap>,
    timers: &Arc<TimerManager>,
    smp: bool,
) -> Result<Arc<dyn CpuDevice>, String> {
    let physical = node.chip.decode.physical();
    let cached: Arc<dyn BusDevice> = physical.clone();
    let (bus, memory) = (node.bus.clone(), memory.clone());
    let module = spec.module;
    // `[clock] fixed_mhz` overrides the module's Count rate.
    let count_hz = cfg.clock.count_hz(module.part).max(1);
    let cpu = crate::with_cpu!(module.part, true, false, C => {
        let exec = crate::cpu::build::build_cpu::<C>(bus, cached, memory, module, timers, count_hz, node.first_cpu + slice, smp);
        Arc::new(MipsCpu::new(*exec)) as Arc<dyn CpuDevice>
    });
    physical.connect_cpu(&*cpu, true)?;
    Ok(cpu)
}

/// Load the BaseIO card's boot flash with the I/O PROM the machine's table names, else the
/// one built into iris (a build without `embedded-proms` leaves the flash empty).
pub(crate) fn load_io_prom(bridge: &Bridge, cfg: &MachineConfig) -> Result<(), String> {
    let table = cfg.machine.profile.table_name().unwrap_or_default();
    let configured = cfg.ip27_section().and_then(|s| s.io_prom.as_deref());
    // "(embedded)" and an unset `io_prom` both mean the built-in PROM; a named file must exist.
    let named = configured.filter(|p| !crate::dev::prom_image::is_embedded_keyword(p));
    let file = match named {
        Some(path) => std::fs::read(path).map_err(|e| format!("machine.io_prom {path}: {e}"))?,
        // Unset, and not built in: leave the flash empty, as before.
        None => match crate::dev::embedded_prom::get(crate::dev::embedded_prom::EmbeddedProm::Io6) {
            Ok(built_in) => built_in,
            Err(_) if configured.is_none() => return Ok(()),
            Err(e) => return Err(format!("machine.io_prom: {e}")),
        },
    };
    let flash = crate::dev::baseio_flash::BaseioFlash::new(&file)
        .map_err(|e| format!("machine.io_prom {}: {e}", named.unwrap_or("(embedded)")))?;
    bridge.attach_flash(0, Arc::new(flash), crate::dev::baseio_flash::SIZE as u32);
    Ok(())
}

/// The two ISP1040s behind the I/O board's Bridge.
const SCSI_DEVICES: [u8; 2] = [0, 1];

/// The Origin 2000's XIO slots IO1 to IO12: ports of the first node's Crossbow (IO1 is the BaseIO),
/// then the second node's (Origin 2000 quick-reference HMQ-380-C, p. 250).
const ORIGIN2000_SLOTS: [&str; 12] = [
    "xio8", "xio12", "xio15", "xio11", "xio14", "xio13",
    "node1/xio14", "node1/xio11", "node1/xio15", "node1/xio12", "node1/xio8", "node1/xio13",
];

/// The Onyx2 deskside's slots IO1 to IO6 on its one Crossbow: the BaseIO-G, three option slots and
/// the graphics slot.
const ONYX2_SLOTS: [&str; 6] = ["xio15", "xio11", "xio14", "xio8", "xio13", "xio12"];

/// The slots of the Origin 200, Origin 2000 and Onyx2.
pub(crate) fn slots(s: &mut Vec<crate::topology::MachineSlot>, profile: MachineProfile) {
    use crate::topology::DeviceFamily;
    let slots: &[&str] = match profile {
        MachineProfile::Origin2000Ip27 => &ORIGIN2000_SLOTS,
        MachineProfile::Onyx2Ip27 => &ONYX2_SLOTS,
        _ => &[],
    };
    for (n, at) in slots.iter().enumerate() {
        let alias = format!("io{}", n + 1);
        match n {
            0 if profile == MachineProfile::Origin2000Ip27 => crate::topology::fixed_slot(s, at, "BaseIO", &[&alias]),
            0 => crate::topology::fixed_slot(s, at, "BaseIO-G", &[&alias]),
            // The Onyx2's graphics slot, the only one of its board classes modelled.
            _ => crate::topology::open_slot(s, at, if *at == "xio8" { &[DeviceFamily::Gfx] } else { &[] }, &[&alias]),
        }
    }
    // The BaseIO Bridge's widget: 8 on the Origin 200 and 2000, 15 on the Onyx2's BaseIO-G.
    let widget = if profile == MachineProfile::Onyx2Ip27 { 15 } else { 8 };
    for d in SCSI_DEVICES {
        crate::topology::scsi_bus_slots(s, &format!("xio{widget}/pci0/{d}"), "isp1040", 15);
    }
}

/// The PCI devices both IP27 I/O boards carry behind their Bridge: two ISP1040s and an IOC3.
/// `ioc3_board` gives its identification, ports and SuperIO interrupt input; the devices set the
/// status bar's activity bits in `heartbeat`.
pub(crate) fn pci_devices(
    timers: &Arc<TimerManager>,
    cfg: &MachineConfig,
    bridge: Arc<Bridge>,
    ioc3_board: &'static Ioc3Board,
    controller: Arc<dyn ModuleController>,
    heartbeat: Arc<AtomicU64>,
) -> Result<BoardIo, String> {
    let (scsi_name, name) = (cfg.machine.profile.table_name().unwrap_or_default(), cfg.machine.profile.product_name().unwrap_or_default());
    let (bus_a, bus_b) = (Isp1040::new_bus(), Isp1040::new_bus());
    let resolved = crate::topology::DevicePlacement::new(cfg)?;
    let scratch = crate::scsi::attach_without_daynaport(cfg, scsi_name, name, &resolved, &[&bus_a, &bus_b])?;
    let scsi = vec![Arc::new(Isp1040::new_at(0, bus_a, timers)), Arc::new(Isp1040::new_at(1, bus_b, timers))];
    for s in &scsi {
        s.set_heartbeat(heartbeat.clone());
    }
    let pci = bridge.bus();
    let slot = |device| crate::pci::DevFn { device, function: 0 };
    pci.attach(slot(SCSI_DEVICES[0]), scsi[0].clone());
    pci.attach(slot(SCSI_DEVICES[1]), scsi[1].clone());
    // The Fuel's M48T35 Timekeeper is the same part; IRIX on the IP27 adds 1970 to its year cell.
    let timekeeper = Arc::new(Timekeeper::with_nvram_file(Path::new(&cfg.nvram), YearBase::From(1970), timers.clock().clone())?);
    let ioc3 = Arc::new(Ioc3::new(timers, ioc3_board, cfg.network().mac, ByteBus::M48t35(timekeeper.clone())));
    ioc3.set_heartbeat(heartbeat);
    let eth = ioc3.eth().expect("an IP27 I/O board's IOC3 has Ethernet").clone();
    pci.attach(slot(2), ioc3.clone());
    ioc3.connect_superio(pci);

    let console = cfg.ci.then(|| Arc::new(CiSerialBackend::new()));
    let serial = |board, n| crate::dev::serial_dev::connect(cfg, name, board, n, console.as_ref(), None);
    ioc3.set_serial_backend(0, serial("IOC3 A", 1)?);
    ioc3.set_serial_backend(1, serial("IOC3 B", 2)?);

    bridge.register_locks();
    ioc3.register_locks();
    for s in &scsi {
        s.register_locks();
    }

    let mut monitor: Vec<Arc<dyn Device>> = scsi.iter().map(|s| s.clone() as Arc<dyn Device>).collect();
    monitor.push(ioc3.ps2());
    monitor.push(crate::pci::pci_host_commands("Bridge", &[bridge.bus()], None));
    let devices: Vec<Arc<dyn Device>> = vec![ioc3.clone(), scsi[0].clone(), scsi[1].clone()];
    let ps2 = ioc3.ps2();
    let reset_scsi = scsi.clone();
    let disk_scsi = scsi.clone();
    let disc_changer = Some(Arc::new(crate::workstation::DiscChangers(scsi.iter().map(|c| c.clone() as _).collect())) as Arc<dyn DiscChanger>);
    let reset_clock = timekeeper.clone();
    let flush = timekeeper.clone();
    let reset: Box<dyn Fn() + Send + Sync> = Box::new(move || {
        controller.reset();
        bridge.reset();
        bridge.reset_pci();
        ioc3.power_on();
        for s in &reset_scsi {
            s.reset();
        }
        reset_clock.power_on();
    });
    Ok(BoardIo {
        devices,
        monitor,
        ps2,
        flush_nvram: Box::new(move || flush.flush_nvram()),
        console,
        attach_net: Box::new(move |link| eth.attach(link)),
        disks: Box::new(move |f| f(&disk_scsi.iter().map(|c| c.bus()).collect::<Vec<_>>())),
        disc_changer,
        scratch,
        reset,
        display: None,
        status: None,
        at_batch: None,
        fasttick: None,
        cpus: None,
    })
}

#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
