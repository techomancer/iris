//! The Bedrock node every IP35 board shares: the chip with its memory, L1, boot flash and
//! Hub-special page, and the processor bus over them. A board names its inputs in [`NodeSpec`].
//!
//! # What is approximated here
//!
//! The two Hub-special head windows read zero and acknowledge writes; PROM 6.210 writes 0x2000
//! at offsets 0 and 0x40 before DRAM discovery. Their meaning is unknown; accesses are logged.
//!
//! Node composition ported from hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

use super::{SparseWindowMap, RESET_WINDOW, RESET_WINDOW_SIZE};
use crate::cpu::mips_exec::CpuDevice;
use crate::dev::bedrock::{Bedrock, BedrockConfig, Population, BANK_SPAN, SLOTS};
use crate::cpu::parts::CpuModule;
use crate::dev::bedrock_flash::{disabled_positions_environment, FirmwareRecord, Flash, PART_SIZE};
use crate::hptimer::TimerManager;
use crate::dev::l1::{L1Config, L1Event, L1};
use crate::dev::widths::{self, access_widths};
use crate::devlog::LogModule;
use crate::sn::{FetchOp, NodeBoard, UncachedBus, UncachedSpace};
use crate::traits::{BusDevice, BusRead64, MachineEvent, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::Arc;

/// What a product says about one of its nodes: a C-brick, or a Fuel's or a
/// Tezro's one node.
pub(crate) struct NodeSpec {
    /// The node's name in logs and map names.
    pub name: &'static str,
    /// The processor module populated: its part, and the mode word its Config resets with.
    pub module: &'static CpuModule,
    /// The boot flash's configuration record, whose mode word is the module's.
    pub firmware: FirmwareRecord,
    /// The word the board reads back at the Hub-special board-id window.
    pub board_id: u64,
    /// The memory banks in MiB, locational bank 0 upward.
    pub banks_mib: Vec<u32>,
    /// The processor slots populated.
    pub processors: Population,
    /// The processor positions (letters A to D) with nothing attached; a fresh flash lists
    /// them disabled.
    pub absent: &'static str,
    pub l1: L1Config,
    /// Persistent boot flash; absent means volatile.
    pub flash: Option<PathBuf>,
}

impl NodeBoard<Bedrock, Flash> {
    /// The node `spec` names, of a machine of `nodes` nodes numbering its processors from
    /// `first_cpu`, with the PROM image `payload`, on `clock`, sending resets and power events to
    /// `events`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        table: &str,
        spec: &NodeSpec,
        payload: &[u8],
        timers: &Arc<TimerManager>,
        first_cpu: usize,
        nodes: usize,
        events: &Sender<MachineEvent>,
        l1: &Arc<L1>,
    ) -> Result<Self, String> {
        let NodeSpec { name, firmware, board_id, processors, .. } = *spec;
        let records = disabled_positions_environment(spec.absent);
        let flash = Arc::new(match &spec.flash {
            Some(path) => Flash::with_file_seeded(payload, path, firmware, &records).map_err(|e| format!("machine.flash {}: {e}", path.display())),
            None => Flash::seeded(payload, firmware, &records).map_err(|e| e.to_string()),
        }?);
        flash.use_clock(timers.clock().reader());
        let prom = Arc::new(flash.array_view(0, PART_SIZE).map_err(|e| e.to_string())?);
        // A bank 0 over the reset window keeps its RAM there for cached references;
        // otherwise the chip maps the PROM there.
        let chip = Bedrock::new(BedrockConfig {
            name,
            timers,
            processors: processors.with_first_cpu(first_cpu),
            banks_mib: &spec.banks_mib,
            nodes,
            events: Some(events.clone()),
            prom: (!payload.is_empty()).then(|| prom.clone() as Arc<dyn BusDevice>),
        })?;
        let phys = chip.decode.physical().clone();
        l1.set_irq(chip.uart_irq());
        let tx = events.clone();
        l1.set_event_handler(Arc::new(move |event| {
            dlog!(LogModule::L1, "{name}: L1 event {event:?}");
            if let Some(action) = machine_action(event) {
                let _ = tx.send(action);
            }
        }));
        let mut special = SparseWindowMap::default();
        // Hub-special low memory: its first 256 MiB is the node's own memory.
        special.map(0, 0x1000_0000, chip.decode.own_memory());
        let boot = Arc::new(flash.boot_view());
        for base in [0x1800_0000, 0x3800_0000] {
            special.map(base, 0x800_0000, boot.clone());
        }
        // Reset execution always sees stored bytes, independently of commands.
        special.map(RESET_WINDOW, RESET_WINDOW_SIZE, prom.clone());
        let cell = Arc::downgrade(&chip);
        let permission: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || cell.upgrade().is_some_and(|chip| chip.md.cell_permits()));
        let page = Arc::new(HubSpecialPage { name, permits: permission, l1: l1.clone(), leds: Mutex::new([0; 4]), board_id });
        for base in [0x1000_0000, 0x3000_0000] {
            special.map(base, BOARD_ID_AT + 8, page.clone());
        }
        special.map(0x1_c000_0000, 1 << 30, chip.directory_bus());
        special.map(0x1_4000_0000, 1 << 30, chip.ecc_bus());
        special.map(0x1_0000_0000, 1 << 30, chip.ecc_scrub_bus());
        chip.decode.set_space(UncachedSpace::HubSpecial, Arc::new(special));
        chip.decode.set_space(UncachedSpace::Io, chip.io_space());
        let bus = Arc::new(UncachedBus::new([
            chip.cpu_view(UncachedSpace::HubSpecial),
            chip.cpu_view(UncachedSpace::Io),
            Arc::new(FetchOp(phys.clone())),
            phys,
        ]));
        Ok(Self { chip, flash, bus, first_cpu })
    }
}

/// Give a node's memory and interrupts to its processors, CPU 0 first, each in the slot the board's
/// population gives it.
pub(crate) fn connect_cpus(chip: &Bedrock, cpus: &[Arc<dyn CpuDevice>]) -> Result<(), String> {
    let phys = chip.decode.physical();
    if let Some(space) = phys.ppmem_space() {
        // The PROM keeps its stack in primary data lines before memory is set up, so lines
        // may stand in any slot that can hold a bank.
        let extent = chip.md.ram_extent();
        for slot in 0..SLOTS {
            if extent.prefix(slot) > 0 {
                let at = slot as u64 * BANK_SPAN;
                space.allow_unbacked_lines(at, at + extent.prefix(slot));
            }
        }
    }
    for (n, cpu) in cpus.iter().enumerate() {
        phys.connect_cpu(&**cpu, true)?;
        chip.set_cpu_interrupts(n, cpu.interrupts_ptr());
    }
    Ok(())
}

/// What the machine does for an L1 power or reset event. A power cycle (request 0x2006) is power
/// down then up and restarts the machine; only a power down (0x2005 or `pwr d`) turns it off.
fn machine_action(event: L1Event) -> Option<MachineEvent> {
    match event {
        L1Event::Shutdown => Some(MachineEvent::PowerOff),
        L1Event::Reset => Some(MachineEvent::HardReset),
        L1Event::Power(_) => None,
    }
}

/// Where the board-id doubleword sits in the Hub-special page's span.
const BOARD_ID_AT: u64 = 0x400_0000;

/// The processor register page of the Hub-special space: head windows at 0 and 0x40, the L1 at
/// 0x80, LEDs at 0xc0 and the board id at 0x400_0000. The head windows read zero and acknowledge
/// writes (contract unknown).
struct HubSpecialPage {
    /// The board name for the log.
    name: &'static str,
    permits: Arc<dyn Fn() -> bool + Send + Sync>,
    l1: Arc<dyn BusDevice>,
    leds: Mutex<[u8; 4]>,
    /// The word read back at `BOARD_ID_AT`.
    board_id: u64,
}
impl HubSpecialPage {
    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        let end = a + size as u64;
        if !(self.permits)() {
            return BusRead64::err();
        }
        if a == BOARD_ID_AT && size == 8 && value.is_none() {
            return BusRead64::ok(self.board_id);
        }
        if a >= 0x1000 {
            return BusRead64::err();
        }
        if end <= 0x80 {
            match value {
                None => dlog!(LogModule::Bedrock, "{}: unidentified Hub-special read {size} bytes at {a:#x}", self.name),
                Some(v) => dlog!(LogModule::Bedrock, "{}: unidentified Hub-special write {size} bytes at {a:#x} value {v:#x}", self.name),
            }
            return BusRead64::ok(0);
        }
        if a >= 0x80 && end <= 0xc0 {
            return widths::access(&*self.l1, a - 0x80, size, value);
        }
        if a >= 0xc0 && end <= 0xe0 && size == 8 && a & 7 == 0 {
            let mut leds = self.leds.lock();
            let led = &mut leds[(a - 0xc0) as usize / 8];
            return match value {
                None => BusRead64::ok(u64::from(*led)),
                Some(v) => {
                    *led = v as u8;
                    BusRead64::ok(0)
                }
            };
        }
        BusRead64::err()
    }
}
impl BusDevice for HubSpecialPage {
    access_widths!();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A power cycle gives one restart and a shutdown one power-off, never a power-off during a
    /// cycle.
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

    /// A page of the Hub-special space and the switch that stands for the cell permission.
    fn page() -> (HubSpecialPage, Arc<std::sync::atomic::AtomicBool>) {
        let permitted = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let allowed = permitted.clone();
        let page = HubSpecialPage {
            name: "test",
            permits: Arc::new(move || allowed.load(std::sync::atomic::Ordering::Relaxed)),
            l1: Arc::new(crate::sn::NoMemory(LogModule::Bedrock)),
            leds: Mutex::new([0; 4]),
            board_id: 0x4004,
        };
        (page, permitted)
    }

    /// A head window reads zero and takes writes at every width; the LEDs keep a byte each, at
    /// doubleword offsets alone.
    #[test]
    fn a_head_window_reads_zero_and_takes_writes() {
        let (head, permitted) = page();
        assert_eq!((head.read8(0).data, head.read16(2).data, head.read32(4).data, head.read64(0x40).data), (0, 0, 0, 0));
        assert_eq!([head.read8(0).status, head.read16(2).status, head.read32(4).status, head.read64(0x40).status], [BUS_OK; 4]);
        assert_eq!([head.write8(0, 1), head.write16(2, 1), head.write32(4, 1), head.write64(0x40, 0x2000)], [BUS_OK; 4]);
        assert_eq!(head.write64(0xd8, 0x1a5), BUS_OK);
        assert_eq!(head.read64(0xd8).data, 0xa5, "an LED keeps its low byte");
        assert_eq!([head.read32(0xd8).status, head.write64(0xdc, 1), head.read64(0xe0).status], [BUS_ERR; 3]);
        permitted.store(false, std::sync::atomic::Ordering::Relaxed);
        assert_eq!([head.read64(0x40).status, head.write64(0x40, 0x2000), head.read64(0xd8).status], [BUS_ERR; 3]);
    }

    /// The board id answers one doubleword read at 0x400_0000 while the cell permits; all else is a
    /// bus error.
    #[test]
    fn the_board_id_answers_one_doubleword_and_the_rest_of_the_span_is_a_bus_error() {
        let (page, permitted) = page();
        let id = page.read64(BOARD_ID_AT);
        assert_eq!((id.status, id.data), (BUS_OK, 0x4004));
        assert_eq!(page.write64(BOARD_ID_AT, 1), BUS_ERR);
        assert_eq!([page.read64(BOARD_ID_AT + 8).status, page.read64(0x1000).status, page.read64(BOARD_ID_AT - 8).status], [BUS_ERR; 3]);
        permitted.store(false, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(page.read64(BOARD_ID_AT).status, BUS_ERR);
    }
}
