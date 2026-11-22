//! The Origin 200 (IP29 System Board): one node whose Hub's XIO port goes straight to a Bridge with
//! two ISP1040s and an IOC3, no Crossbow. The shared IP27 machine is in [`crate::ip27`].
//!
//! # What is approximated here
//!
//! - The Bridge reaches the Hub as an XIO host ([`Hub::host_port`]) at widget 0xa, with no Crossbow
//!   between them, and answers at fifteen widget numbers (every one but the Hub's); real hardware's are unknown.
//! - The controller's configuration store is `ElscNvram::origin200`; the machine never looks inside it.
//! - The Origin 200 node keeps Hub revision 3 and IP29 board record 030-1244-001 H whatever the
//!   module; real 225 and 270 MHz machines report Hub revision 5 and board 030-1389-001.
//!
//! Ported from hw/mips/sgi/sn-machine.c and hw/mips/sgi/sn0.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::config::MachineConfig;
use crate::cpu::parts;
use crate::dev::bridge::{Bridge, PIN_WIRING};
use crate::dev::elsc::{Elsc, ElscConfig, ModuleOutput};
use crate::dev::elsc_nvram::ElscNvram;
use crate::dev::heart::Clock;
use crate::dev::hub::{Hub, HubConfig};
use crate::dev::i2c::{BusBusy, I2cBus};
use crate::dev::node_flash::{FlashValues, NodeFlash};
use crate::dev::pcf8584::Pcf8584;
use crate::dev::w1;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use crate::hptimer::TimerManager;
use crate::ip27::{attach_controller, pci_devices, ModuleController, NodeSpec};
use crate::sn::BoardIo;
use crate::sn::NodeBoard;
use crate::traits::{Device, MachineEvent, Resettable};
use std::sync::mpsc::Sender;
use crate::xtalk::Upstream;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// The Hub's own widget number: a small window there is not the Bridge's.
const WIDGET_HUB: u8 = 1;


/// The 180 MHz one-processor module of an Origin 200, per a published sale
/// listing; the 270 MHz R12000 module's record is unknown, so it has none.
pub const PIMM_1XT5: BoardRecord<'static> = BoardRecord { barcode: "", part: "013-1895-001", revision: "C", name: "PIMM_1XT5_1MB" };

/// The widget number the Bridge addresses the Hub as: the master widget the
/// PROM programs for DMA and interrupts.
const HUB_HOST_WIDGET: u8 = 0xa;
/// The IP29 System Board's Bridge revision.
const BRIDGE_REVISION: u32 = 4;

/// The Bridge's one-wire line carries the Origin 200's System Board record. The
/// barcode and ROM serial are synthetic.
fn board_record() -> w1::Bus {
    let mut bus = w1::Bus::new();
    let record = BoardRecord { barcode: "EMUN00", part: "030-1244-001", revision: "H", name: "IP29" };
    bus.add(Box::new(board_record_part(&record, 0x5145_4d55_4200)));
    bus
}

/// The Origin 200's module controller, on its own node's I2C controller.
struct Origin200Controller {
    elsc: Arc<Elsc>,
    i2c: Arc<Pcf8584>,
}

impl Origin200Controller {
    fn new(hub: &Hub, timers: &Arc<TimerManager>, store: Option<&Path>) -> Result<Arc<Self>, String> {
        let store = ElscNvram::origin200(store).map(Arc::new).map_err(|e| format!("machine.controller_store: {e}"))?;
        let busy = BusBusy::default();
        let elsc = Elsc::new(ElscConfig::default(), store.clone(), timers, busy.clone());
        let mut bus = I2cBus::new();
        bus.add(elsc.clone());
        bus.add(store);
        let i2c = Arc::new(Pcf8584::new(bus, busy));
        hub.attach_i2c(i2c.clone());
        // Power it on once, here: reset() leaves it running, as on the Origin 2000.
        elsc.power_on();
        Ok(Arc::new(Self { elsc, i2c }))
    }
}

impl ModuleController for Origin200Controller {
    /// The controller is independent of the machine: only its I2C adapter resets, as the deskside's do.
    fn reset(&self) {
        self.i2c.power_on();
    }
    fn take_events(&self) -> Vec<ModuleOutput> {
        self.elsc.take_events()
    }
    fn set_wake(&self, wake: Box<dyn Fn() + Send + Sync>) {
        self.elsc.set_wake(wake);
    }
    fn device(&self) -> Arc<dyn Device> {
        self.elsc.clone()
    }
}

pub(crate) fn node_specs(cfg: &MachineConfig) -> Result<Vec<NodeSpec>, String> {
    let part = cfg.cpu_part()?;
    let module = parts::cpu_module(part).ok_or("an Origin 200 has an R10000 or an R12000")?;
    Ok(vec![NodeSpec {
        // Hub 2.1 on the Origin 200; the node is not in a numbered midplane slot.
        hub: HubConfig { cpus: cfg.machine.cpus(), ni_revision: 3, banks_mb: std::array::from_fn(|i| cfg.banks[i]), ..Default::default() },
        module,
        firmware: FlashValues {
            cpu_mode_bits: module.mode_word,
            cpu_clock_hz: module.clock_hz(),
            ..FlashValues::STOCK_R10000_180
        },
        flash: cfg.ip27_section().expect("an IP27 profile").flash.as_ref().map(PathBuf::from),
        node_record: (part == &parts::R10000_180).then_some(PIMM_1XT5),
    }])
}

/// The Origin 200 (IP29 System Board): the Hub's XIO port straight to a
/// Bridge with two ISP1040s and an IOC3, no Crossbow.
pub(crate) fn build_io(
    cfg: &MachineConfig,
    nodes: &[NodeBoard<Hub, NodeFlash>],
    clock: &Clock,
    timers: &Arc<TimerManager>,
    events: &Sender<MachineEvent>,
) -> Result<BoardIo, String> {
    let hub = &nodes[0].chip;
    let controller = Origin200Controller::new(hub, timers, cfg.ip27_section().expect("an IP27 profile").controller_store.as_deref().map(Path::new))?;
    let bridge = Bridge::new(Upstream::new().with(hub.host_port(HUB_HOST_WIDGET)), &PIN_WIRING);
    bridge.set_revision(BRIDGE_REVISION);
    bridge.set_microlan(board_record());
    // The Bridge answers every small window but the Hub's own.
    for widget in (0..16).filter(|&w| w != WIDGET_HUB) {
        hub.io.connect_widget(widget, bridge.clone());
    }
    let mut io = pci_devices(timers, cfg, bridge, &crate::dev::ioc3::IP29, controller.clone(), Arc::new(AtomicU64::new(0)))?;
    attach_controller(&mut io, controller, events);
    Ok(io)
}

#[cfg(test)]
mod tests;
