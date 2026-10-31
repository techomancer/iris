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
use crate::dev::serial_host::{null_serial_backend, open_log, tcp_serial_backend, CiSerialBackend, LogMode, SerialBackend, TeeBackend};
use crate::devlog::LogModule;
use crate::irq::{InterruptController, IrqLine};
use crate::ppmem::devmap::SLOT_SHIFT;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, MachineEvent, Signal, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::path::Path;
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
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
    if let Some(space) = md.physical().ppmem_space() {
        use crate::ppmem::MappedMemory;
        unsafe {
            space.set_bitmap_sink(cpu.ppmem_bitmap_ptr());
            cpu.set_tcache_window(space.window_base());
            let bitmap = cpu.tcache_bitmap_ptr();
            if !bitmap.is_null() { space.set_bitmap_sink2(bitmap); }
            #[cfg(feature = "jitv2")]
            cpu.set_tcache_gen_window(md.generation_base());
        }
        md.physical().resync_ppmem_bitmap();
        md.rebuild_memory();
    }
    pi0.set_interrupt_controller(Arc::new(CpuIrq(Arc::downgrade(cpu))));
    let (tx, rx) = channel();
    let reset_tx = tx.clone();
    lb.connect_soft_reset(Arc::new(move || {
        let _ = reset_tx.send(MachineEvent::HardReset);
    }));
    let reset_tx = tx.clone();
    ni.connect_local_reset(Arc::new(move || {
        let _ = reset_tx.send(MachineEvent::HardReset);
    }));
    let weak_cpu = Arc::downgrade(cpu);
    l1.set_event_handler(Arc::new(move |event| {
        match event {
            L1Event::Power(false) | L1Event::Shutdown => {
                let _ = tx.send(MachineEvent::PowerOff);
            }
            L1Event::Power(true) | L1Event::Reset => {
                let _ = tx.send(MachineEvent::HardReset);
            }
            L1Event::SoftReset => {
                if let Some(cpu) = weak_cpu.upgrade() {
                    cpu.as_device().signal(Signal::Reset(true));
                }
            }
            L1Event::Nmi => dlog!(LogModule::L1, "{name}: NMI entry is not implemented by the CPU"),
        }
    }));
    rx
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

struct CpuIrq(std::sync::Weak<dyn CpuDevice>);
impl InterruptController for CpuIrq {
    fn set_line(&self, line: u32, level: bool) {
        if line < 5 {
            if let Some(cpu) = self.0.upgrade() {
                cpu.as_device().signal(Signal::Interrupt(line + 2, level));
            }
        }
    }
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
    #[cfg(test)]
    pub irq: Arc<PendingIrq>,
}

impl Node {
    pub fn build(config: NodeConfig<'_>) -> Result<Self, String> {
        let NodeConfig { name, payload, clock, flash_path, banks_mib, population, firmware, board_id, l1 } = config;
        let md = BedrockMd::with_clock(banks_mib, clock.clone())?;
        let phys = md.physical().clone();
        let pi: [Arc<BedrockPi>; 2] = std::array::from_fn(|interface| {
            BedrockPi::new(
                BedrockPiConfig {
                    interface: interface as u8,
                    request_region: 0,
                    population,
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
        let registers = Arc::new(registers);
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
        bus.set_fallback(1, 1 << 32, Arc::new(IoPio(io.clone())));
        // The two XIO register forms select the requester's PI0 interface.
        let permission_md = Arc::downgrade(&md);
        let permission: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || {
            permission_md.upgrade().is_some_and(|md| md.current_cell() & 1 != 0)
        });
        let protected_regs = Arc::new(Window::new(0, 0x80_0000, registers.clone()).permitted(permission.clone()));
        for base in [0x0100_0000, 0x0180_0000] {
            bus.map(1, base, 0x80_0000, protected_regs.clone());
        }
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

struct IoPio(Arc<BedrockIo>);
macro_rules! pio_access {
    ($r:ident,$w:ident,$ty:ty,$result:ident,$size:expr) => {
        fn $r(&self, a: u64) -> $result {
            let r = self.0.pio_read(a, $size);
            $result {
                data: r.data as $ty,
                status: r.status,
            }
        }
        fn $w(&self, a: u64, v: $ty) -> u32 {
            self.0.pio_write(a, $size, v as u64)
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
