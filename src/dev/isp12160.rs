//! The QLogic ISP12160: two SCSI buses behind one PCI function.
//!
//! The ISP12160 shares the ISP1040 ring, DMA, and mailbox mechanism, while
//! selecting targets from either channel. Each channel remains an explicit
//! [`ScsiBus`], including for monitor and run-time disc operations.
//!
//! Ported from hw/scsi/qlogic-isp.c and include/hw/scsi/qlogic-isp.h
//! in jamesbraid/qemu-sgi-origami.

use std::collections::VecDeque;
use std::ops::Deref;

use parking_lot::MutexGuard;

use crate::dev::isp1040::{
    mbox, reg, Bank, Core, Isp1040, IspPart, State, CFG1_BANK_SELECT, FW_VERSION, IOCB_ABORTED, IOCB_RESET, MAILBOX_COUNT,
    MAX_REQUESTS, MAX_TARGET, PCI_CLASS_SCSI, PCI_VENDOR_QLOGIC, SEMAPHORE_LOCK, STATE_ALL_PHASES,
};
use crate::irq::IrqLine;
use crate::pci::{Bar, PciBusDevice, PciConfig, PciConfigPolicy, PciDma, PciRegisterPolicy, PCI_INTA};
use crate::scsi::{ScsiBus, ScsiIdentities, ScsiIdentity};
use crate::snapshot::{get_field, hex_u16, hex_u32, load_u16_slice, toml_u16, toml_u32, toml_u8, u16_slice_to_toml};
use crate::traits::{Device, Resettable, Saveable};
use crate::workstation::DiscChanger;

const ISP_DISK: ScsiIdentity = ScsiIdentity::new(b"SGI     ", b"IRIS EMULATED   ", b"1.0 ");
const ISP_CDROM: ScsiIdentity = ISP_DISK;

pub(crate) struct Isp12160Part;

impl IspPart for Isp12160Part {
    fn name(&self) -> &'static str { "ISP12160" }
    fn channels(&self) -> usize { 2 }
    fn pci_config(&self, bars: [Bar; 6]) -> PciConfig {
        let registers = [
            PciRegisterPolicy { reg_dword: 13, initial: 0x44, writable_mask: 0 },
            PciRegisterPolicy { reg_dword: 17, initial: 0x0001_0001, writable_mask: 0 },
            PciRegisterPolicy { reg_dword: 18, initial: 0, writable_mask: 0x100 },
        ];
        PciConfig::new_with_policy(
            PCI_VENDOR_QLOGIC,
            0x1216,
            PCI_CLASS_SCSI,
            6,
            bars,
            PCI_INTA,
            PciConfigPolicy {
                status: 0x02b0,
                status_w1c_mask: 0,
                command_mask: 0x0156,
                latency_mask: 0xff,
                cache_line_mask: 0xff,
                registers: &registers,
            },
        )
    }
    fn firmware_version(&self) -> [u16; 3] { [10, 4, FW_VERSION[2]] }
    fn active_bank(&self, cfg1: u16) -> Bank {
        match (cfg1 >> 8) & 3 {
            1 => Bank::Sxp,
            2 => Bank::Sxp1,
            3 => Bank::Dma,
            _ => Bank::Risc,
        }
    }
    fn checks_ram_bounds(&self) -> bool { true }
    fn has_abort_commands(&self) -> bool { true }
    fn target_on(&self, channel: u8, target: u8) -> u8 { target | (channel << 4) }
    fn decode_target(&self, target: u8) -> (usize, u8) { ((target >> 4) as usize, target & 0x0f) }
    fn id_high(&self) -> u16 { 0x1216 }
    fn bus_cfg0(&self) -> u16 { 0 }
    fn cfg1_write_mask(&self) -> u16 { 0x0300 }
    fn unmodelled_dma_register(&self, off: u32) -> bool {
        matches!(off, reg::DMA_CMD_CFG | reg::DMA_CMD_CTRL | reg::DMA_DATA_CFG | reg::DMA_DATA_CTRL)
    }
    fn completion_phases(&self, moved: u64, total: u64) -> u16 {
        let mut flags = 0x1700;
        if moved != 0 {
            flags |= 0x0800;
            if moved == total { flags |= 0x4000; }
        }
        flags
    }
    fn mailbox_ready(&self, semaphore: u16) -> bool { semaphore & SEMAPHORE_LOCK == 0 }
    fn clear_target_parameters(&self, mailbox: &mut [u16; MAILBOX_COUNT]) { mailbox[6] = 0; }
    fn set_reset_delay(&self, st: &mut State) {
        st.reset_delay = st.mailbox[1];
        st.reset_delay1 = st.mailbox[2];
    }
    fn reset_targets(&self, st: &mut State) -> bool {
        st.reset_targets = true;
        true
    }
    fn execute_mailbox(&self, core: &Core, st: &mut MutexGuard<'_, State>) -> bool {
        let cmd = st.mailbox[0];
        match cmd {
            mbox::LOAD_RAM_A64 => st.mailbox[0] = core.ram_transfer(st, true),
            mbox::ABORT_COMMAND | mbox::ABORT_DEVICE | mbox::ABORT_TARGET => {
                let device = st.mailbox[1];
                let valid_target = (device >> 8) & 0x7f <= MAX_TARGET as u16;
                let target = ((device >> 8) & 0x7f) as u8 | (((device >> 15) as u8) << 4);
                let lun = device as u8;
                let handle = ((st.mailbox[2] as u32) << 16) | st.mailbox[3] as u32;
                let hit = valid_target && st.active.is_some_and(|(t, l, h)| t == target
                    && (cmd == mbox::ABORT_TARGET || l == lun) && (cmd != mbox::ABORT_COMMAND || h == handle));
                let completion = if cmd == mbox::ABORT_TARGET { IOCB_RESET } else { IOCB_ABORTED };
                if hit { core.cancel_active(st, completion); }
                let retired = core.retire_outstanding(st, |(t, l, h)| valid_target && t == target
                    && (cmd == mbox::ABORT_TARGET || l == lun) && (cmd != mbox::ABORT_COMMAND || h == handle), completion);
                if cmd == mbox::ABORT_TARGET && valid_target {
                    st.pending_target_resets.push_back((target >> 4, Some(target & 15)));
                }
                st.mailbox[0] = if cmd == mbox::ABORT_COMMAND && !hit && retired == 0 { mbox::COMMAND_ERROR } else { mbox::COMPLETE };
            }
            // Mailbox 1 is the reset delay; mailbox 2 selects the bus.
            mbox::BUS_RESET => {
                let channel = st.mailbox[2];
                if channel > 1 { st.mailbox[0] = mbox::PARAM_ERROR; }
                else {
                    if st.active.is_some_and(|(target, _, _)| target >> 4 == channel as u8) {
                        core.cancel_active(st, IOCB_RESET);
                    }
                    core.retire_outstanding(st, |(target, _, _)| target >> 4 == channel as u8, IOCB_RESET);
                    st.pending_target_resets.push_back((channel as u8, None));
                    core.kick(st);
                    st.mailbox[0] = mbox::COMPLETE;
                }
            }
            _ => return false,
        }
        true
    }
    fn save_extra(&self, st: &State, t: &mut toml::map::Map<String, toml::Value>) {
        t.insert("reset_targets".into(), toml::Value::Boolean(st.reset_targets));
        t.insert("target_resets".into(), toml::Value::Array(st.pending_target_resets.iter().map(|(channel, target)| {
            toml::Value::Array(vec![hex_u16(*channel as u16), hex_u16(target.unwrap_or(0xff) as u16)])
        }).collect()));
        t.insert("sxp1_bank".into(), u16_slice_to_toml(&st.sxp1_bank));
        t.insert("dma_bank".into(), u16_slice_to_toml(&st.dma_bank));
        t.insert("reset_delay1".into(), hex_u16(st.reset_delay1));
        t.insert("outstanding".into(), toml::Value::Array(st.outstanding.iter().map(|&(target, lun, handle)| {
            toml::Value::Array(vec![hex_u16(target as u16), hex_u16(lun as u16), hex_u32(handle)])
        }).collect()));
    }
    fn load_extra(&self, st: &mut State, v: &toml::Value) {
        st.reset_targets = get_field(v, "reset_targets").and_then(|x| x.as_bool()).unwrap_or(false);
        if let Some(toml::Value::Array(items)) = get_field(v, "target_resets") {
            for item in items.iter().take(64) {
                if let toml::Value::Array(fields) = item {
                    if fields.len() == 2 {
                        if let (Some(channel), Some(target)) = (toml_u8(&fields[0]), toml_u8(&fields[1])) {
                            if channel < 2 && (target < 16 || target == 0xff) {
                                st.pending_target_resets.push_back((channel, (target < 16).then_some(target)));
                            }
                        }
                    }
                }
            }
        }
        let u16f = |k: &str| get_field(v, k).and_then(toml_u16).unwrap_or(0);
        st.reset_delay1 = u16f("reset_delay1");
        if let Some(m) = get_field(v, "sxp1_bank") { load_u16_slice(m, &mut st.sxp1_bank); }
        if let Some(m) = get_field(v, "dma_bank") { load_u16_slice(m, &mut st.dma_bank); }
        if let Some(toml::Value::Array(items)) = get_field(v, "outstanding") {
            for item in items.iter().take(MAX_REQUESTS as usize) {
                if let toml::Value::Array(fields) = item {
                    if fields.len() == 3 {
                        if let (Some(target), Some(lun), Some(handle)) =
                            (toml_u8(&fields[0]), toml_u8(&fields[1]), toml_u32(&fields[2])) {
                            if target < 32 && lun < 8 { st.outstanding.push((target, lun, handle)); }
                        }
                    }
                }
            }
        }
    }
}

/// The two-channel controller. The shared ISP engine owns both buses.
pub struct Isp12160(Isp1040);

impl Default for Isp12160 {
    fn default() -> Self { Self::new(Self::new_bus(), Self::new_bus()) }
}

impl Isp12160 {
    /// Light the status bar's SCSI activity bits in `heartbeat`.
    pub fn set_heartbeat(&self, heartbeat: std::sync::Arc<std::sync::atomic::AtomicU64>) {
        self.0.set_heartbeat(heartbeat);
    }

    /// A 16-id bus with the ISP controller's disk and CD-ROM identities.
    pub fn new_bus() -> ScsiBus {
        ScsiBus::new(16, ScsiIdentities { disk: ISP_DISK, cdrom: ISP_CDROM })
    }

    /// The controller with its two channel buses already populated.
    pub fn new(bus0: ScsiBus, bus1: ScsiBus) -> Self {
        Self(Isp1040::new_12160(bus0, bus1))
    }

    /// Borrow one channel for the shared monitor commands.
    pub fn channel(&self, channel: u8) -> Option<Channel<'_>> {
        (channel < 2).then_some(Channel { adapter: self, channel })
    }

    pub fn register_locks(&self) { self.0.register_locks(); }
}

impl Deref for Isp12160 {
    type Target = Core;
    fn deref(&self) -> &Core { &self.0 }
}

/// One channel, with SCSI ids 0–15 as the monitor presents them.
pub struct Channel<'a> {
    adapter: &'a Isp12160,
    channel: u8,
}

impl crate::scsi_monitor::ScsiTargets for Channel<'_> {
    fn scsi_bus(&self) -> &ScsiBus {
        self.adapter.bus_for_channel(self.channel).expect("valid ISP12160 channel")
    }
}

/// The machine API flattens the ISP12160's channels to ids 0–31.
/// Monitor commands use a `Channel` instead, so their ids remain 0–15.
impl DiscChanger for Isp12160 {
    fn disc_status(&self) -> Vec<(usize, String, Vec<String>, u64, u64)> {
        (0..2).flat_map(|channel| {
            crate::scsi_monitor::disc_status(&self.channel(channel).expect("valid ISP12160 channel"))
                .into_iter()
                .map(move |(id, current, discs, physical, logical)| (id + channel as usize * 16, current, discs, physical, logical))
        }).collect()
    }

    fn load_disc(&self, id: usize, path: String) -> Result<String, String> {
        let (channel, target) = Self::split_id(id)?;
        crate::scsi_monitor::load_disc(&self.channel(channel).unwrap(), target, path)
    }

    fn eject_disc(&self, id: usize) -> Result<String, String> {
        let (channel, target) = Self::split_id(id)?;
        crate::scsi_monitor::eject_disc(&self.channel(channel).unwrap(), target)
    }

    fn eject_to_empty(&self, id: usize) -> Result<(), String> {
        let (channel, target) = Self::split_id(id)?;
        crate::scsi_monitor::eject_to_empty(&self.channel(channel).unwrap(), target)
    }
}

impl Isp12160 {
    fn split_id(id: usize) -> Result<(u8, usize), String> {
        if id >= 32 { return Err(format!("SCSI ID {id} is out of range")); }
        Ok(((id / 16) as u8, id % 16))
    }
}

impl Device for Isp12160 {
    fn step(&self, cycles: u64) { Device::step(&self.0, cycles) }
    fn start(&self) { Device::start(&self.0) }
    fn stop(&self) { Device::stop(&self.0) }
    fn is_running(&self) -> bool { Device::is_running(&self.0) }
    fn get_clock(&self) -> u64 { Device::get_clock(&self.0) }

    fn register_commands(&self) -> Vec<(String, String)> {
        let usage = crate::scsi_monitor::SCSI_USAGE;
        vec![
            ("scsi".into(), format!("SCSI (QLogic ISP12160, channel 0): {usage} | scsi notes")),
            ("scsi0".into(), "Alias for `scsi` (ISP12160 channel 0)".into()),
            ("cow".into(), "COW overlays on ISP12160 channel 0: cow status | cow commit [id] | cow reset [id]".into()),
            ("scsi1".into(), format!("SCSI (QLogic ISP12160, channel 1): {usage}")),
            ("cow1".into(), "COW overlays on ISP12160 channel 1: same subcommands as `cow`".into()),
        ]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write as _;
        let (channel, cow) = match cmd {
            "scsi" | "scsi0" | "cow" => (0, cmd.starts_with("cow")),
            "scsi1" | "cow1" => (1, cmd.starts_with("cow")),
            _ => return Err("Command not found".into()),
        };
        let ctl = self.channel(channel).expect("valid ISP12160 channel");
        if cow {
            return crate::scsi_monitor::cow_command(&ctl, args, &mut *w);
        }
        if let Some(done) = crate::scsi_monitor::scsi_subcommand(&ctl, args, &mut *w) {
            return done;
        }
        if channel == 0 && args.first().copied() == Some("notes") {
            let _ = writeln!(w, "QLogic ISP12160: {} commands", self.total_commands());
            for note in self.notes() { let _ = writeln!(w, "  note: {note}"); }
            return Ok(());
        }
        Err(format!("Usage: {}", crate::scsi_monitor::SCSI_USAGE))
    }
}

impl Resettable for Isp12160 {
    fn power_on(&self) {
        self.0.reset();
        PciBusDevice::config(&self.0).reset();
    }
}

impl Saveable for Isp12160 {
    fn save_state(&self) -> toml::Value { Saveable::save_state(&self.0) }
    fn load_state(&self, v: &toml::Value) -> Result<(), String> { Saveable::load_state(&self.0, v) }
}

impl PciBusDevice for Isp12160 {
    fn config(&self) -> &PciConfig { PciBusDevice::config(&self.0) }
    fn bar_read(&self, bar: usize, off: u32, width: usize) -> u32 { PciBusDevice::bar_read(&self.0, bar, off, width) }
    fn bar_write(&self, bar: usize, off: u32, val: u32, width: usize) { PciBusDevice::bar_write(&self.0, bar, off, val, width) }
    fn connect_interrupt(&self, line: IrqLine) { PciBusDevice::connect_interrupt(&self.0, line) }
    fn connect_dma(&self, dma: std::sync::Arc<dyn PciDma>) { PciBusDevice::connect_dma(&self.0, dma) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The machine disc API uses target 17 for channel 1, target 1.
    #[test]
    fn disc_status_flattens_the_second_channels_target_ids() {
        let bus0 = Isp12160::new_bus();
        let bus1 = Isp12160::new_bus();
        bus1.attach_device(1, "", true, Vec::new(), false, None).unwrap();
        let isp = Isp12160::new(bus0, bus1);

        let status = DiscChanger::disc_status(&isp);
        assert_eq!(status.iter().map(|drive| drive.0).collect::<Vec<_>>(), vec![17]);
    }
}
