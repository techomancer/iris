//! MACE, the O2's I/O ASIC, as the bus sees it: one device claiming MACE's
//! 8 MB at 0x1f000000 and the PCI bridge's windows, which decodes the offset
//! and hands the access to the block it names, as HPC3 does for its
//! children on the Indigo2. The blocks themselves are in their own files
//! (`mace_pci`, `mec`, `mace_isa`, `mace_ps2`, `mace_ust`, `mace_audio`,
//! `mace_sdma`, `ns16550`, `ds1687`, `onewire`); what has no model yet is a
//! storing register file (`regs`).

use std::sync::{Arc, Mutex};

use crate::dev::crime::Crime;
use crate::dev::crime_mem::O2Memory;
use crate::dev::ds1687::MaceRtc;
use crate::dev::mace_audio::{AudioIo, MaceAudio};
use crate::dev::mace_isa::*;
use crate::dev::mace_pci::MacePci;
use crate::dev::mace_ps2::MacePs2;
use crate::dev::mace_sdma::*;
use crate::dev::mace_ust::MaceUst;
use crate::dev::mec::Mec;
use crate::dev::ns16550::Com16550;
use crate::dev::onewire::{NicTrace, OneWireId};
use crate::ip32::{
    nic_bit, Stub, IP32_MEC_CRIME_INT, MACE_BASE, MACE_COM0, MACE_COM1, MACE_COM_SIZE, MACE_ISA_FLASH_NIC_REG,
    MACE_MEC, MACE_MEC_SIZE, MACE_PS2, MACE_PS2_SIZE, MACE_RTC, MACE_RTC_SIZE, MACE_SIZE, MACE_UST_MSC,
    MACE_UST_MSC_SIZE, MACEPCI_BASE, MACEPCI_SIZE,
};
use crate::snapshot::get_field;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Saveable, BUS_OK};

pub struct Mace {
    ram: Arc<O2Memory>,
    /// The ISA block's interrupt status and mask, cascading into CRIME.
    pub isa: Arc<MaceIsa>,
    /// Registers with no model of their own yet: they keep what is written
    /// and count accesses per offset.
    pub regs: Stub,
    /// The PCI host bridge: its registers here, its windows outside MACE.
    pub pci: MacePci,
    pub ust: Arc<MaceUst>,
    pub nic_trace: NicTrace,
    pub rtc: MaceRtc,
    pub ps2: MacePs2,
    pub onewire: OneWireId,
    pub com0: Com16550,
    pub com1: Com16550,
    /// The serial ports' DMA rings, which IRIX's console runs on.
    pub sdma: MaceSerialDma,
    /// The audio interface and its AD1843 (dev/mace_audio.rs).
    pub audio: MaceAudio,
    /// The Ethernet. Its frames go nowhere until `start_nat` (`[o2] net`).
    pub mec: Arc<Mec>,
}

impl Mace {
    /// MACE with its blocks reset, its interrupts wired into `crime`: the
    /// Ethernet straight in, the ISA peripherals (the PS/2 ports, audio, the
    /// console UART while driven by PIO, serial DMA) through the ISA block.
    pub fn new(ram: Arc<O2Memory>, crime: Arc<Crime>, ust: Arc<MaceUst>) -> Self {
        let isa = Arc::new(MaceIsa::new(crime.clone()));
        let mace = Self {
            isa,
            regs: Stub::new("mace"),
            pci: MacePci::new(crime.clone(), ram.clone()),
            ust,
            nic_trace: NicTrace::new(),
            rtc: MaceRtc::new(),
            ps2: MacePs2::new(),
            // A locally-administered address; nothing depends on the value yet.
            onewire: OneWireId::from_mac([0x08, 0x00, 0x69, 0x12, 0x34, 0x56]),
            com0: Com16550::named("com0"),
            com1: Com16550::named("com1"),
            sdma: MaceSerialDma::new(),
            audio: MaceAudio::new(),
            mec: Arc::new(Mec::new(ram.clone())),
            ram,
        };
        mace.mec.connect_irq(crate::irq::IrqLine::new(crime, IP32_MEC_CRIME_INT));
        mace.ps2.connect_irq(mace.isa.clone());
        mace.audio.connect_irq(mace.isa.clone());
        mace.com0.connect_irq(crate::irq::IrqLine::new(mace.isa.clone(), ISA_SERIAL0_PIO));
        mace.sdma.connect_irq(mace.isa.clone());
        // Revision 2, as in every O2: see `regs_wrote`.
        mace.regs.set_cell_pub(MACE_ISA_RING_BASE, MACE_REV2_BIT);
        mace
    }

    /// Whether MACE answers `addr`: its own 8 MB or a PCI window.
    pub fn contains(addr: u32) -> bool {
        (MACE_BASE..MACE_BASE + MACE_SIZE).contains(&addr) || MacePci::in_window(addr)
    }

    /// Plug in PS/2 devices: a keyboard on port 0 and, with `mouse`, a mouse
    /// on port 1.
    pub fn enable_ps2(&mut self, mouse: bool) {
        self.ps2 = MacePs2::with_devices(mouse);
        self.ps2.connect_irq(self.isa.clone());
    }

    /// What time and the outside world have done since the last call: the
    /// Ethernet's frames, the consoles' typing, audio and serial DMA.
    pub fn service(&self) {
        self.mec.poll();
        self.com0.pump();
        self.com1.pump();
        self.audio.poll(&AudioIo {
            ram: &self.ram,
            ring_base: (self.regs.cell(MACE_ISA_RING_BASE) & sdma::RING_BASE_MASK) as u32,
            ust: self.ust.counter() as u32,
        });
        if self.sdma.any_owned() {
            self.sdma.poll(&self.sdma_io());
        }
    }

    /// The devices answering the monitor, as HPC3 passes its children's
    /// commands through on the Indigo2.
    pub fn devices(&self) -> Vec<&dyn crate::traits::Device> {
        vec![
            &self.pci, &*self.ust, &self.rtc, &self.ps2, &self.onewire, &self.com0, &self.com1, &self.sdma,
            &self.audio, &*self.mec, &*self.isa,
        ]
    }

    /// The engines' view of the machine.
    fn sdma_io(&self) -> SdmaIo<'_> {
        SdmaIo {
            ram: &self.ram,
            com: [&self.com0, &self.com1],
            ring_base: (self.regs.cell(MACE_ISA_RING_BASE) & sdma::RING_BASE_MASK) as u32,
        }
    }

    fn in_sdma(addr: u32) -> bool {
        (MACE_SDMA..MACE_SDMA + MACE_SDMA_SIZE).contains(&addr)
    }

    /// A serial DMA register, read. Counted with MACE's other offsets, so a
    /// guest polling one shows up where every other hot register does.
    fn sdma_read(&self, addr: u32) -> u64 {
        self.regs.r();
        self.regs.hit(addr, false);
        self.sdma.read(addr)
    }

    fn sdma_write(&self, addr: u32, val: u64) -> u32 {
        self.regs.w();
        self.regs.hit(addr, true);
        self.sdma.write(addr, val, &self.sdma_io());
        // A pointer or control write moves the levels, and may hand the port
        // back to (or take it from) the 16550's own interrupt.
        self.com0.update_irq();
        self.sdma.update_irq();
        BUS_OK
    }

    /// An ISA block register, read: the DMA ports' bits as they stand now,
    /// not as of the last service tick.
    fn isa_reading(&self, addr: u32) {
        self.regs.r();
        self.regs.hit(addr, false);
        if self.sdma.any_owned() {
            self.sdma.update_irq();
        }
    }

    /// After a write to the ISA block's status or mask: a status write
    /// clears the serial DMA engine's pair-request latches it wrote as zero,
    /// and a DMA port's bits are read through the new mask at once.
    fn isa_wrote(&self, addr: u32) {
        if addr & !7 == MACE_ISA_INT_STATUS {
            self.sdma.status_written(self.isa.status_written());
        }
        if self.sdma.any_owned() {
            self.sdma.update_irq();
        }
    }

    /// After a write to a register held in `regs`.
    fn regs_wrote(&self, addr: u32) {
        // MACE revision 2 or later: SGI software reads bit 4 of the ISA ring
        // base register as that, and with it clear shifts every PS/2 reply
        // right one bit to work around revision 1. The revision is the chip's,
        // not the keyboard's: a headless O2 reads 2 as well.
        if addr & !7 == MACE_ISA_RING_BASE {
            let v = self.regs.cell_pub(MACE_ISA_RING_BASE);
            self.regs.set_cell_pub(MACE_ISA_RING_BASE, v | MACE_REV2_BIT);
        }
    }

    fn in_mec(addr: u32) -> bool {
        (MACE_MEC..MACE_MEC + MACE_MEC_SIZE).contains(&addr)
    }

    fn in_pci_regs(addr: u32) -> bool {
        (MACEPCI_BASE..MACEPCI_BASE + MACEPCI_SIZE).contains(&addr)
    }

    fn in_ps2(addr: u32) -> bool {
        (MACE_PS2..MACE_PS2 + MACE_PS2_SIZE).contains(&addr)
    }

    fn in_rtc(addr: u32) -> bool {
        (MACE_RTC..MACE_RTC + MACE_RTC_SIZE).contains(&addr)
    }

    fn in_ust(addr: u32) -> bool {
        (MACE_UST_MSC..MACE_UST_MSC + MACE_UST_MSC_SIZE).contains(&addr)
    }

    /// Which audio channel's MSC/UST pair a UST-block address names: the
    /// three at 0x20, 0x28 and 0x30 (input, output #1, output #2).
    fn audio_msc_slot(addr: u32) -> Option<usize> {
        let off = addr.wrapping_sub(MACE_UST_MSC);
        (0x20..0x38).contains(&off).then(|| ((off - 0x20) / 8) as usize)
    }

    fn com_for(&self, addr: u32) -> Option<&Com16550> {
        if (MACE_COM0..MACE_COM0 + MACE_COM_SIZE).contains(&addr) {
            Some(&self.com0)
        } else if (MACE_COM1..MACE_COM1 + MACE_COM_SIZE).contains(&addr) {
            Some(&self.com1)
        } else {
            None
        }
    }

    fn is_nic_reg(addr: u32) -> bool {
        (addr & !3) == MACE_ISA_FLASH_NIC_REG
    }

    /// Feed a write of the NIC register through to the 1-Wire device.
    fn nic_write(&self, val: u8) {
        self.nic_trace.record(self.ust.now(), self.ust.guest_us(), true, val);
        // DEASSERT alone decides it: clear means the master is pulling the
        // line down, set means it has let go. The firmware writes DATA=1 in
        // both cases, so DATA is an input here, not a level to drive — which
        // is exactly what the bus trace shows (only 0x08 and 0x0c are ever
        // written).
        let low = val & nic_bit::DEASSERT == 0;
        self.ust.onewire_busy();
        self.onewire.master_drive(self.ust.guest_us(), low);
    }

    /// The NIC register, with the 1-Wire line as it stands now. A read does
    /// not hold the guest clock (only the master driving the line does):
    /// IRIX reads this register about a hundred times a second for its
    /// other bits, and each read holding the clock for 5 ms ran UST at half
    /// speed.
    fn nic_read(&self) -> u8 {
        let mut v = self.regs.cell(MACE_ISA_FLASH_NIC_REG) as u8;
        if self.onewire.line_level(self.ust.guest_us()) {
            v |= nic_bit::DATA;
        } else {
            v &= !nic_bit::DATA;
        }
        self.nic_trace.record(self.ust.now(), self.ust.guest_us(), false, v);
        v
    }
}

/// Every block behind MACE, each under its own name. The UST loads first,
/// since others keep time by it, and the ISA controller last: the devices
/// drive their lines again as they load, and the controller's own record
/// of which bits devices drive is then put back as it was saved.
impl Saveable for Mace {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        for (k, d) in self.blocks() {
            t.insert(k.into(), d.save_state());
        }
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        for (k, d) in self.blocks() {
            if let Some(s) = get_field(v, k) {
                d.load_state(s).map_err(|e| format!("mace {k}: {e}"))?;
            }
        }
        Ok(())
    }
}

impl Mace {
    /// The blocks with state, in load order.
    fn blocks(&self) -> [(&'static str, &dyn Saveable); 12] {
        [
            ("ust", &*self.ust),
            ("regs", &self.regs),
            ("pci", &self.pci),
            ("rtc", &self.rtc),
            ("ps2", &self.ps2),
            ("onewire", &self.onewire),
            ("com0", &self.com0),
            ("com1", &self.com1),
            ("sdma", &self.sdma),
            ("audio", &self.audio),
            ("mec", &*self.mec),
            ("isa", &*self.isa),
        ]
    }
}

// MACE decodes 32 bits: `a` is the address its decode helpers take, and the
// full bus address goes on to the block that answers.
impl BusDevice for Mace {
    fn read8(&self, addr: u64) -> BusRead8 {
        let a = addr as u32;
        if Self::in_mec(a) {
            return BusDevice::read8(&*self.mec, addr);
        }
        if Self::in_pci_regs(a) {
            return BusRead8::ok(self.pci.read32(addr & !3).data as u8);
        }
        if Self::is_nic_reg(a) {
            return BusRead8::ok(self.nic_read());
        }
        if let Some(c) = self.com_for(a) {
            return c.read8(addr);
        }
        if MacePci::in_window(a) {
            return BusRead8::ok(self.pci.window_read(a, 1) as u8);
        }
        if Self::in_ps2(a) {
            return BusDevice::read8(&self.ps2, addr);
        }
        if Self::in_rtc(a) {
            return self.rtc.read8(addr);
        }
        if MaceAudio::contains(a) {
            return BusDevice::read8(&self.audio, addr);
        }
        if Self::in_sdma(a) {
            return BusRead8::ok(self.sdma_read(a) as u8);
        }
        if MaceIsa::contains(a) {
            self.isa_reading(a);
            return BusDevice::read8(&*self.isa, addr);
        }
        self.regs.read8(addr)
    }

    fn read16(&self, addr: u64) -> BusRead16 {
        let a = addr as u32;
        if Self::in_mec(a) {
            return BusDevice::read16(&*self.mec, addr);
        }
        if Self::in_pci_regs(a) {
            return BusRead16::ok(self.pci.read32(addr & !3).data as u16);
        }
        if let Some(c) = self.com_for(a) {
            return c.read16(addr);
        }
        if MacePci::in_window(a) {
            return BusRead16::ok(self.pci.window_read(a, 2) as u16);
        }
        if Self::in_ps2(a) {
            return BusDevice::read16(&self.ps2, addr);
        }
        if Self::in_rtc(a) {
            return self.rtc.read16(addr);
        }
        if MaceAudio::contains(a) {
            return BusDevice::read16(&self.audio, addr);
        }
        if Self::in_sdma(a) {
            return BusRead16::ok(self.sdma_read(a) as u16);
        }
        if MaceIsa::contains(a) {
            self.isa_reading(a);
            return BusDevice::read16(&*self.isa, addr);
        }
        self.regs.read16(addr)
    }

    fn read32(&self, addr: u64) -> BusRead32 {
        let a = addr as u32;
        if Self::in_mec(a) {
            return BusDevice::read32(&*self.mec, addr);
        }
        if let Some(n) = Self::audio_msc_slot(a) {
            let v = self.audio.msc_ust(n);
            return BusRead32::ok(if addr & 4 == 0 { (v >> 32) as u32 } else { v as u32 });
        }
        if Self::in_ust(a) {
            return self.ust.read32(addr);
        }
        if Self::in_pci_regs(a) {
            return self.pci.read32(addr);
        }
        if Self::is_nic_reg(a) {
            return BusRead32::ok(self.nic_read() as u32);
        }
        if let Some(c) = self.com_for(a) {
            return c.read32(addr);
        }
        if MacePci::in_window(a) {
            return BusRead32::ok(self.pci.window_read(a, 4) as u32);
        }
        if Self::in_ps2(a) {
            return BusDevice::read32(&self.ps2, addr);
        }
        if Self::in_rtc(a) {
            return self.rtc.read32(addr);
        }
        if MaceAudio::contains(a) {
            return BusDevice::read32(&self.audio, addr);
        }
        if Self::in_sdma(a) {
            // The registers are 64-bit; a 32-bit access is the half it names.
            let v = self.sdma_read(a);
            return BusRead32::ok(if addr & 4 == 0 { (v >> 32) as u32 } else { v as u32 });
        }
        if MaceIsa::contains(a) {
            self.isa_reading(a);
            return BusDevice::read32(&*self.isa, addr);
        }
        self.regs.read32(addr)
    }

    fn read64(&self, addr: u64) -> BusRead64 {
        let a = addr as u32;
        if Self::in_mec(a) {
            return BusDevice::read64(&*self.mec, addr);
        }
        if let Some(n) = Self::audio_msc_slot(a) {
            return BusRead64::ok(self.audio.msc_ust(n));
        }
        if Self::in_ust(a) {
            return self.ust.read64(addr);
        }
        if Self::in_pci_regs(a) {
            return self.pci.read64(addr);
        }
        if Self::is_nic_reg(a) {
            return BusRead64::ok(self.nic_read() as u64);
        }
        if let Some(c) = self.com_for(a) {
            return c.read64(addr);
        }
        if MacePci::in_window(a) {
            return BusRead64::ok(self.pci.window_read(a, 8));
        }
        if Self::in_ps2(a) {
            return BusDevice::read64(&self.ps2, addr);
        }
        if Self::in_rtc(a) {
            return self.rtc.read64(addr);
        }
        if MaceAudio::contains(a) {
            return BusDevice::read64(&self.audio, addr);
        }
        if Self::in_sdma(a) {
            return BusRead64::ok(self.sdma_read(a));
        }
        if MaceIsa::contains(a) {
            self.isa_reading(a);
            return BusDevice::read64(&*self.isa, addr);
        }
        self.regs.read64(addr)
    }

    fn write8(&self, addr: u64, val: u8) -> u32 {
        let a = addr as u32;
        if Self::in_mec(a) {
            return BusDevice::write8(&*self.mec, addr, val);
        }
        if Self::in_pci_regs(a) {
            return self.pci.write32(addr & !3, val as u32);
        }
        if Self::is_nic_reg(a) {
            self.nic_write(val);
            let r = self.regs.write8(addr, val);
            self.regs_wrote(a);
            return r;
        }
        if let Some(c) = self.com_for(a) {
            return c.write8(addr, val);
        }
        if MacePci::in_window(a) {
            self.pci.window_write(a, val as u64, 1);
            return BUS_OK;
        }
        if Self::in_ps2(a) {
            return BusDevice::write8(&self.ps2, addr, val);
        }
        if Self::in_rtc(a) {
            return self.rtc.write8(addr, val);
        }
        if MaceAudio::contains(a) {
            return BusDevice::write8(&self.audio, addr, val);
        }
        if Self::in_sdma(a) {
            // A narrow store is the low end of the register.
            return self.sdma_write(a, val as u64);
        }
        if MaceIsa::contains(a) {
            self.regs.w();
            self.regs.hit(a, true);
            let r = BusDevice::write8(&*self.isa, addr, val);
            self.isa_wrote(a);
            return r;
        }
        let r = self.regs.write8(addr, val);
        self.regs_wrote(a);
        r
    }

    fn write16(&self, addr: u64, val: u16) -> u32 {
        let a = addr as u32;
        if Self::in_mec(a) {
            return BusDevice::write16(&*self.mec, addr, val);
        }
        if Self::in_pci_regs(a) {
            return self.pci.write32(addr & !3, val as u32);
        }
        if let Some(c) = self.com_for(a) {
            return c.write16(addr, val);
        }
        if MacePci::in_window(a) {
            self.pci.window_write(a, val as u64, 2);
            return BUS_OK;
        }
        if Self::in_ps2(a) {
            return BusDevice::write16(&self.ps2, addr, val);
        }
        if Self::in_rtc(a) {
            return self.rtc.write16(addr, val);
        }
        if MaceAudio::contains(a) {
            return BusDevice::write16(&self.audio, addr, val);
        }
        if Self::in_sdma(a) {
            // A narrow store is the low end of the register.
            return self.sdma_write(a, val as u64);
        }
        if MaceIsa::contains(a) {
            self.regs.w();
            self.regs.hit(a, true);
            let r = BusDevice::write16(&*self.isa, addr, val);
            self.isa_wrote(a);
            return r;
        }
        let r = self.regs.write16(addr, val);
        self.regs_wrote(a);
        r
    }

    fn write32(&self, addr: u64, val: u32) -> u32 {
        let a = addr as u32;
        if Self::in_mec(a) {
            return BusDevice::write32(&*self.mec, addr, val);
        }
        if let Some(n) = Self::audio_msc_slot(a) {
            let v = self.audio.msc_ust(n);
            let v = if addr & 4 == 0 { (v & 0xffff_ffff) | (val as u64) << 32 } else { (v & !0xffff_ffff) | val as u64 };
            self.audio.set_msc_ust(n, v);
            return BUS_OK;
        }
        if Self::in_ust(a) {
            return self.ust.write32(addr, val);
        }
        if Self::in_pci_regs(a) {
            return self.pci.write32(addr, val);
        }
        if Self::is_nic_reg(a) {
            self.nic_write(val as u8);
            let r = self.regs.write32(addr, val);
            self.regs_wrote(a);
            return r;
        }
        if let Some(c) = self.com_for(a) {
            return c.write32(addr, val);
        }
        if MacePci::in_window(a) {
            self.pci.window_write(a, val as u64, 4);
            return BUS_OK;
        }
        if Self::in_ps2(a) {
            return BusDevice::write32(&self.ps2, addr, val);
        }
        if Self::in_rtc(a) {
            return self.rtc.write32(addr, val);
        }
        if MaceAudio::contains(a) {
            return BusDevice::write32(&self.audio, addr, val);
        }
        if Self::in_sdma(a) {
            // A narrow store is the low end of the register.
            return self.sdma_write(a, val as u64);
        }
        if MaceIsa::contains(a) {
            self.regs.w();
            self.regs.hit(a, true);
            let r = BusDevice::write32(&*self.isa, addr, val);
            self.isa_wrote(a);
            return r;
        }
        let r = self.regs.write32(addr, val);
        self.regs_wrote(a);
        r
    }

    fn write64(&self, addr: u64, val: u64) -> u32 {
        let a = addr as u32;
        if Self::in_mec(a) {
            return BusDevice::write64(&*self.mec, addr, val);
        }
        if let Some(n) = Self::audio_msc_slot(a) {
            self.audio.set_msc_ust(n, val);
            return BUS_OK;
        }
        if Self::in_ust(a) {
            return self.ust.write64(addr, val);
        }
        if Self::in_pci_regs(a) {
            return self.pci.write64(addr, val);
        }
        if Self::is_nic_reg(a) {
            self.nic_write(val as u8);
            let r = self.regs.write64(addr, val);
            self.regs_wrote(a);
            return r;
        }
        if let Some(c) = self.com_for(a) {
            return c.write64(addr, val);
        }
        if MacePci::in_window(a) {
            self.pci.window_write(a, val, 8);
            return BUS_OK;
        }
        if Self::in_ps2(a) {
            return BusDevice::write64(&self.ps2, addr, val);
        }
        if Self::in_rtc(a) {
            return self.rtc.write64(addr, val);
        }
        if MaceAudio::contains(a) {
            return BusDevice::write64(&self.audio, addr, val);
        }
        if Self::in_sdma(a) {
            return self.sdma_write(a, val);
        }
        if MaceIsa::contains(a) {
            self.regs.w();
            self.regs.hit(a, true);
            let r = BusDevice::write64(&*self.isa, addr, val);
            self.isa_wrote(a);
            return r;
        }
        let r = self.regs.write64(addr, val);
        self.regs_wrote(a);
        r
    }
}
