//! The Bedrock node every IP35 board shares: the processor bus with its four
//! views, the Bedrock chip with its memory, the L1, the boot flash and the
//! Hub-special page, wired together and to the board's processors. A board
//! names its inputs in [`NodeConfig`] and adds its own crossbar widgets.
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

use super::{map_device, Ip35Bus, Window};
use crate::cpu::mips_exec::CpuDevice;
use crate::dev::bedrock::{Bedrock, BedrockConfig, Population, BANK_SPAN, SLOTS};
use crate::dev::bedrock_flash::{FirmwareRecord, Flash, PART_SIZE};
use crate::dev::heart::Clock;
use crate::dev::l1::{L1Config, L1Event, L1};
use crate::devlog::LogModule;
use crate::ppmem::devmap::SLOT_SHIFT;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, MachineEvent, Signal, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

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
    pub processors: Population,
    pub firmware: FirmwareRecord,
    /// The word the board reads back at the Hub-special board-id window.
    pub board_id: u64,
    pub l1: L1Config,
}

/// One Bedrock node as the IP35 boards share it.
pub(crate) struct Node {
    pub bus: Arc<Ip35Bus>,
    pub bedrock: Arc<Bedrock>,
    pub l1: Arc<L1>,
    pub flash: Arc<Flash>,
    /// Cacheable memory at plain physical addresses: what a cache fills
    /// from, and the one bus the JIT and its compile pool read code through.
    pub cached: Arc<dyn BusDevice>,
    /// What the JIT compiles code through: `cached`, plus the reset PROM's
    /// uncached-fetch window ([`Ip35Bus::code_view`]).
    #[cfg(feature = "jitv2")]
    pub code: Arc<dyn BusDevice>,
    /// The node's resets and power events, for the system controller.
    pub events: Sender<MachineEvent>,
    pub receiver: Receiver<MachineEvent>,
}

impl Node {
    pub fn build(config: NodeConfig<'_>) -> Result<Self, String> {
        let NodeConfig { name, payload, clock, flash_path, banks_mib, processors, firmware, board_id, l1 } = config;
        let (events, receiver) = channel();
        let bedrock = Bedrock::new(BedrockConfig {
            name,
            clock: clock.clone(),
            processors,
            banks_mib,
            events: Some(events.clone()),
        })?;
        let phys = bedrock.physical().clone();
        let l1 = Arc::new(L1::new(l1)?);
        l1.set_irq(bedrock.uart_irq());
        let flash = Arc::new(match flash_path {
            Some(path) => Flash::with_file(payload, path, firmware),
            None => Flash::new(payload, firmware),
        }.map_err(|e| e.to_string())?);
        let mut bus = Ip35Bus::new(phys.clone());
        bedrock.rebuild_memory();
        bus.set_fallback(1, 1 << 32, bedrock.pio());
        // The local and node XIO register forms: with processors on PI0
        // alone, the same window.
        bus.map(1, 0x0100_0000, 0x80_0000, bedrock.local_registers());
        bus.map(1, 0x0180_0000, 0x80_0000, bedrock.registers());
        // Uncached RAM reaches the banks directly, as cached RAM does; the
        // diagnostic views stay separate.
        let uncached = bedrock.uncached_view();
        bus.map(3, 0, 1 << 33, uncached.clone());
        // Hub-special low memory: its first 256 MiB is the node memory view.
        bus.map(0, 0, 0x1000_0000, uncached);
        let prom = Arc::new(flash.array_view(0, PART_SIZE).map_err(|e| e.to_string())?);
        let boot = Arc::new(flash.boot_view());
        for base in [0x1800_0000, 0x3800_0000] {
            bus.map(0, base, 0x800_0000, boot.clone());
        }
        // Reset execution always sees stored bytes, independently of commands.
        bus.map(0, 0x1fc0_0000, 0x40_0000, prom.clone());
        // The same PROM where an uncached instruction fetch of that window is
        // presented (`MipsExecutor::uncached_fetch_alias`).
        bus.map(0, super::RESET_WINDOW_FETCH, super::RESET_WINDOW_SIZE, prom.clone());
        if !payload.is_empty() {
            map_device(&phys, 0x1fc0_0000, payload.len() as u64, prom.clone(), &format!("{name} cached reset PROM"));
        }
        let cell = bedrock.clone();
        let permission: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || cell.cell_permits());
        let leds = Arc::new(Leds { bedrock: bedrock.clone(), values: Mutex::new([0; 4]) });
        // The three windows occupy the same processor register page.
        let special = Arc::new(HubSpecialPage {
            windows: [
                Window::new(0, 0x80, Arc::new(HubHead(name))).permitted(permission.clone()),
                Window::new(0x80, 0x40, l1.clone()).permitted(permission.clone()),
                Window::new(0xc0, 0x20, leds).permitted(permission),
            ],
        });
        for base in [0x1000_0000, 0x3000_0000] {
            bus.map(0, base, 1 << SLOT_SHIFT, special.clone());
            bus.map(0, base + 0x400_0000, 8, Arc::new(BoardId { bedrock: bedrock.clone(), word: board_id }));
        }
        bus.map(0, 0x1_c000_0000, 1 << 30, bedrock.directory_bus());
        bus.map(0, 0x1_4000_0000, 1 << 30, bedrock.ecc_bus());
        bus.map(0, 0x1_0000_0000, 1 << 30, bedrock.ecc_scrub_bus());
        bus.map(2, 0, 1 << 33, bedrock.fetchop_bus());
        let bus = Arc::new(bus);
        let cached = bus.cached_view(bedrock.dma_view(), bedrock.ram_extent());
        #[cfg(feature = "jitv2")]
        let code = bus.code_view(cached.clone());
        Ok(Self {
            bus,
            bedrock,
            l1,
            flash,
            cached,
            #[cfg(feature = "jitv2")]
            code,
            events,
            receiver,
        })
    }

}

/// Give a node's memory and interrupts to its processors, CPU 0 first, each
/// in the slot the board's population gives it: a cached view of memory for
/// every one and its IP2..IP6 from its PI slot. The L1's power events reach
/// the system controller on `events`, and its soft reset every processor.
pub(crate) fn connect_cpus(
    name: &'static str,
    bedrock: &Bedrock,
    l1: &L1,
    events: &Sender<MachineEvent>,
    cpus: &[Arc<dyn CpuDevice>],
) -> Result<(), String> {
    if let Some(space) = bedrock.physical().ppmem_space() {
        // Each CPU reads memory through a cached view of its own. The PROM
        // keeps its stack in primary data lines before it has set up memory,
        // so lines may stand in every slot that can hold a bank without it.
        let extent = bedrock.ram_extent();
        for slot in 0..SLOTS {
            if extent.prefix(slot) > 0 {
                let at = slot as u64 * BANK_SPAN;
                space.allow_unbacked_lines(at, at + extent.prefix(slot));
            }
        }
        for cpu in cpus {
            let v = space
                .add_cached_view(&[cpu.ppmem_bitmap_ptr(), cpu.tcache_bitmap_ptr()])
                .map_err(|e| format!("ppmem: a CPU's cached view: {e}"))?;
            unsafe {
                cpu.set_tcache_window(space.view_base(v));
                cpu.set_cached_view(space, v);
                cpu.set_tcache_shift(crate::ppmem::window_shift(space.window_size()));
                #[cfg(feature = "jitv2")]
                cpu.set_tcache_gen_window(bedrock.generation_base());
            }
        }
        bedrock.rebuild_memory();
    }
    for (n, cpu) in cpus.iter().enumerate() {
        bedrock.set_cpu_interrupts(n, cpu.interrupts_ptr());
    }
    let tx = events.clone();
    l1.set_event_handler(Arc::new(move |event| {
        dlog!(LogModule::L1, "{name}: L1 event {event:?}");
        if let Some(action) = machine_action(event) {
            let _ = tx.send(action);
        }
    }));
    Ok(())
}

/// What the machine does for an L1 power or reset event. A power change
/// alone does nothing: the shutdown or restart the L1 reports with it
/// carries the action. The L1 serves a power cycle (request 0x2006, which
/// the PROM sends from its failure paths) as power down then power up, and
/// a node that powers up starts from reset, so the cycle restarts the
/// machine; only a power down (request 0x2005 or the console's `pwr d`)
/// turns it off.
fn machine_action(event: L1Event) -> Option<MachineEvent> {
    match event {
        L1Event::Shutdown => Some(MachineEvent::PowerOff),
        L1Event::Reset => Some(MachineEvent::HardReset),
        L1Event::Power(_) => None,
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
    bedrock: Arc<Bedrock>,
    word: u64,
}
impl BusDevice for BoardId {
    fn read64(&self, a: u64) -> BusRead64 {
        if a == 0 && self.bedrock.cell_permits() {
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
    bedrock: Arc<Bedrock>,
    values: Mutex<[u8; 4]>,
}
impl BusDevice for Leds {
    fn read64(&self, a: u64) -> BusRead64 {
        if a < 32 && a & 7 == 0 && self.bedrock.cell_permits() {
            BusRead64::ok(self.values.lock()[a as usize / 8] as u64)
        } else {
            BusRead64::err()
        }
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        if a < 32 && a & 7 == 0 && self.bedrock.cell_permits() {
            self.values.lock()[a as usize / 8] = v as u8;
            BUS_OK
        } else {
            BUS_ERR
        }
    }
}
