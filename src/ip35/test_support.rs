//! What the IP35 machine tests share: a PROM stand-in, headless configurations, the machine builder
//! and a PCI head to drive.

use super::*;
use std::path::Path;
use crate::config::{MachineConfig, MachineProfile};
use crate::dev::bedrock::regs as bedrock_reg;
use crate::sn::UncachedBus;
use crate::traits::*;
use crate::window_map::SparseWindowMap;

pub(crate) use crate::sn::tests::headless;

pub(crate) fn config(dir: &Path, c_bricks: u8) -> MachineConfig {
    let mut cfg = headless(dir, MachineProfile::Origin3200Ip35);
    cfg.origin3200.c_bricks = Some(c_bricks);
    cfg.origin3200.l1_port = Some(0);
    cfg
}
/// A Fuel with the volatile boot flash and its listening ports left to any free one.
pub(crate) fn fuel_config(dir: &Path) -> MachineConfig {
    let mut cfg = headless(dir, MachineProfile::FuelIp34);
    cfg.banks = crate::config::banks_from4([256, 256, 0, 0]);
    cfg.banks_set = true;
    cfg.fuel.l1_port = Some(0);
    cfg.devices.clear();
    crate::dev::serial_host::tests::free_ports(&mut cfg);
    cfg
}
/// A Tezro of `cpus` processors, its listening ports left to any free one. It has the 1 GiB four
/// processors need, not the 4 GiB default: Windows commits a machine's RAM when it is built, and
/// tests running side by side with 4 GiB each fill a CI runner's page file.
pub(crate) fn tezro_config(dir: &Path, cpus: u8) -> MachineConfig {
    let mut cfg = headless(dir, MachineProfile::TezroIp53);
    cfg.banks = crate::config::banks_from4([512, 512, 0, 0]);
    cfg.banks_set = true;
    cfg.machine.cpus = Some(cpus);
    cfg.tezro.l1_port = Some(0);
    cfg.devices.clear();
    crate::dev::serial_host::tests::free_ports(&mut cfg);
    cfg
}
/// A PCI head as a processor reaches it through PIO: configuration cycles at the head's
/// widget, and a device's window onto PCI memory.
pub(crate) struct Head {
    pio: Arc<dyn BusDevice>,
    widget: u64,
}
impl Head {
    /// Head `widget` of the node whose PIO window is `pio`.
    pub(crate) fn new(pio: Arc<dyn BusDevice>, widget: u8) -> Self {
        Self { pio, widget: u64::from(widget) }
    }

    fn at(&self, offset: u64) -> u64 {
        (self.widget << 24) | offset
    }

    pub(crate) fn cfg_read(&self, device: u32, reg: u32) -> u32 {
        self.pio.read32(self.at(u64::from(crate::dev::bridge::CFG0_BASE + device * 0x1000 + reg))).data
    }

    pub(crate) fn cfg_write(&self, device: u32, reg: u32, value: u32) {
        assert_eq!(self.pio.write32(self.at(u64::from(crate::dev::bridge::CFG0_BASE + device * 0x1000 + reg)), value), BUS_OK);
    }

    /// Write the head's widget register at `offset` (a doubleword register's low word).
    pub(crate) fn set(&self, offset: u32, value: u32) {
        assert_eq!(self.pio.write32(self.at(u64::from(offset) + 4), value), BUS_OK);
    }

    /// Put the function at `device`'s memory BAR `bar` (a configuration offset) at PCI memory
    /// `base`, enable its memory space and aim the device's window there. Returns the window.
    pub(crate) fn map_bar(&self, device: u32, bar: u32, base: u32) -> Window {
        use crate::dev::bridge::{device_window_offset, reg, DEV_MEMORY_SPACE};
        self.cfg_write(device, bar, base);
        self.cfg_write(device, 0x04, 2);
        self.set(reg::device(device), DEV_MEMORY_SPACE | (base >> 20));
        Window { pio: self.pio.clone(), at: self.at(u64::from(device_window_offset(device))) }
    }
}
/// A device's window onto PCI memory, with the bridge's byte lanes at their reset setting:
/// a register is read at the address the device decodes.
pub(crate) struct Window {
    pio: Arc<dyn BusDevice>,
    at: u64,
}
impl Window {
    pub(crate) fn read32(&self, offset: u64) -> u32 {
        let r = self.pio.read32(self.at + offset);
        assert_eq!(r.status, BUS_OK, "read of {offset:#x}");
        r.data
    }

    pub(crate) fn write32(&self, offset: u64, value: u32) {
        assert_eq!(self.pio.write32(self.at + offset, value), BUS_OK, "write of {offset:#x}");
    }

    /// A halfword register, reached on the lane its address names.
    pub(crate) fn read16(&self, offset: u64) -> u32 {
        let r = self.pio.read16(self.at + (offset ^ 2));
        assert_eq!(r.status, BUS_OK, "read of {offset:#x}");
        u32::from(r.data)
    }

    pub(crate) fn write16(&self, offset: u64, value: u16) {
        assert_eq!(self.pio.write16(self.at + (offset ^ 2), value), BUS_OK, "write of {offset:#x}");
    }
}
/// A PIO address: the widget in bits 27:24.
pub(crate) fn widget(widget: u64, offset: u64) -> u64 {
    (widget << 24) | offset
}

/// The machine `cfg`'s profile names, built and dropped on a big stack around `body`.
pub(crate) fn with_machine<R: Send + 'static>(cfg: MachineConfig, body: impl FnOnce(&mut Machine<Bedrock, Flash>) -> R + Send + 'static) -> R {
    crate::sn::tests::with_machine(cfg, machine, body)
}

/// The machine `cfg`'s profile names, built as `workstation::build` builds it.
pub(crate) fn machine(cfg: &MachineConfig) -> Result<Machine<Bedrock, Flash>, String> {
    use crate::{ip34_machine, ip35_machine, ip53_machine};
    match cfg.machine.profile {
        MachineProfile::FuelIp34 => crate::ip35::machine(cfg, ip34_machine::node_specs, ip34_machine::build_io),
        MachineProfile::TezroIp53 => crate::ip35::machine(cfg, ip53_machine::node_specs, ip53_machine::build_io),
        _ => crate::ip35::machine(cfg, ip35_machine::node_specs, ip35_machine::build_io),
    }
}

/// A bus whose attribute `n` space is `views[n]`, with nothing elsewhere.
pub(crate) fn bus_of(views: [SparseWindowMap; 4]) -> UncachedBus {
    UncachedBus::new(views.map(|map| Arc::new(map) as Arc<dyn BusDevice>))
}
/// The Fuel's processor on `bus`, its cache filling from `cached`, as the machine builds it.
pub(crate) fn fuel_cpu(
    bus: Arc<UncachedBus>,
    cached: Arc<dyn BusDevice>,
) -> Box<
    crate::cpu::mips_exec::MipsExecutor<
        crate::cpu::mips_tlb::MipsTlb,
        crate::cpu::mips_cache_shadow::R12000ShadowCache,
    >,
> {
    let extent = crate::dev::bedrock::RamExtent::new(&[256, 256]);
    let timers = Arc::new(crate::hptimer::TimerManager::new());
    crate::cpu::build::build_cpu(
        bus,
        cached,
        Arc::new(memory_map(&extent)),
        &crate::cpu::parts::R14000_500_SYSCLK_200,
        &timers,
        crate::cpu::parts::R14000_500_SYSCLK_200.part.count_hz(),
        0,
        false,
    )
}
/// A Bedrock unit's registers, at its base in the node's register window.
pub(crate) struct Unit(Arc<dyn BusDevice>, u64);
impl Unit {
    pub(crate) fn read64(&self, a: u64) -> BusRead64 {
        self.0.read64(self.1 + a)
    }
    pub(crate) fn write64(&self, a: u64, v: u64) -> u32 {
        self.0.write64(self.1 + a, v)
    }
}
pub(crate) fn md_regs(m: &Machine<Bedrock, Flash>) -> Unit {
    Unit(m.nodes[0].chip.registers(), bedrock_reg::md::BASE)
}
pub(crate) fn pi_regs(m: &Machine<Bedrock, Flash>, interface: u64) -> Unit {
    Unit(m.nodes[0].chip.registers(), interface * bedrock_reg::PI1_BASE)
}
/// A processor access to widget `widget` through Bedrock's small window.
pub(crate) fn pio_widget(widget: u64, offset: u64) -> u64 {
    (widget << 24) | offset
}
