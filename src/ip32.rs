//! The SGI O2 (IP32): its physical bus. RAM, the PROM and its flash, and the
//! decode that hands everything else to CRIME (`dev::crime`), GBE
//! (`dev::o2gfx`), VICE and MACE (`dev::mace`), which subdivide their own
//! space as HPC3 does on the Indy.
//!
//! The O2 is not wired into `Physical`: its memory map has little in common
//! with the Indy's (RAM at 0x40000000 with a low alias, CRIME and MACE where
//! the Indy has GIO and HPC3), and `MipsCpu::new` takes any
//! `Arc<dyn BusDevice>`, so the CPU runs against this bus without touching
//! the IP22/IP24 path. `ip32_machine` drives it.
//!
//! Everything was derived from observing the PROM's own behaviour, the O2
//! ASIC specifications and NetBSD's published register names.
//!
//! See `docs/o2-ip32.md`; the bring-up log is `clanker/devices/ip32-o2-bringup-log.md`.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

use crate::dev::mace::Mace;
use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR};

// ── IP32 physical map ───────────────────────
/// RAM base.
///
/// **Not zero.** The PROM's SizeMEM writes its probe patterns at `base + 0`,
/// `+0x01fffff8`, `+0x02000000` and `+0x07fffff8`, and the physical address
/// those land on is 0x40000000 — visible as the target of the `sd` at
/// 0xbfc05c9c. This is the "RAM at a non-zero base address" the O2 is known
/// for, and it collides with what NetBSD's `macereg.h` calls `MACE_PCI_NATIVE_VIEW`;
/// main memory wins that address, so the PCI window is not mapped here.
pub const RAM_BASE: u32 = 0x4000_0000;
/// CRIME: memory controller, interrupt controller and video DMA.
pub const CRIME_BASE: u32 = 0x1400_0000;
pub const CRIME_SIZE: u32 = 0x0000_1000;
/// GBE, the graphics back end, which scans the frame buffer out of RAM
/// (`crate::dev::o2gfx::Gbe`).
pub const GBE_BASE: u32 = 0x1600_0000;
pub const GBE_SIZE: u32 = 0x0010_0000;
/// VICE, the video and image compression engine.
pub const VICE_BASE: u32 = 0x1700_0000;
pub const VICE_SIZE: u32 = 0x0100_0000;

/// MACE, the I/O ASIC (`crate::dev::mace`): PCI, Ethernet, the ISA block.
pub const MACE_BASE: u32 = 0x1f00_0000;
pub const MACE_SIZE: u32 = 0x0080_0000;
/// MACE's ISA-block register holding the flash write-enable and the Dallas
/// 1-Wire ("NIC") data line. The PROM bit-bangs the serial-ID chip here.
pub const MACE_ISA_FLASH_NIC_REG: u32 = MACE_BASE + 0x0031_0008;

/// Bits in [`MACE_ISA_FLASH_NIC_REG`], as named by NetBSD's `macereg.h`.
pub mod nic_bit {
    /// 1 => flash writes enabled.
    pub const FLASH_WE: u8 = 0x01;
    /// Release the 1-Wire line (let it float high).
    pub const DEASSERT: u8 = 0x04;
    /// The 1-Wire data line itself.
    pub const DATA: u8 = 0x08;
}

/// `com0`, the PROM's console UART. NetBSD attaches it at `mace0 offset
/// 0x390000`; `com1` follows at 0x398000.
/// The Ethernet, MAC-110 (`crate::dev::mec`), and its CRIME interrupt.
pub const MACE_MEC: u32 = MACE_BASE + 0x0028_0000;
pub const MACE_MEC_SIZE: u32 = 0x0000_0200;
pub const IP32_MEC_CRIME_INT: u32 = 3;
pub const MACE_COM0: u32 = MACE_BASE + 0x0039_0000;
pub const MACE_COM1: u32 = MACE_BASE + 0x0039_8000;
pub const MACE_COM_SIZE: u32 = 0x0000_0800;

/// MACE's free-running timer (`MACE_UST_MSC`, i.e. `MACE_PERIF + 0x40000`).
/// The PROM busy-waits on this; a constant hangs it forever.
/// The two PS/2 ports, keyboard at +0x00 and mouse at +0x20, each
/// `tx / rx / control / status` at eight-byte spacing.
pub const MACE_PS2: u32 = MACE_BASE + 0x0032_0000;
pub const MACE_PS2_SIZE: u32 = 0x40;

pub mod ps2_bit {
    pub const CLOCK_SIGNAL: u64 = 1 << 0;
    pub const CLOCK_INHIBIT: u64 = 1 << 1;
    pub const TX_INPROGRESS: u64 = 1 << 2;
    /// The transmit holding register has drained. The PROM spins on this for
    /// up to 20834 tries after every byte it sends, so a port that never sets
    /// it is not merely idle — it is expensive.
    pub const TX_EMPTY: u64 = 1 << 3;
    /// A byte has arrived. Never set here: nothing is plugged in.
    pub const RX_FULL: u64 = 1 << 4;
    pub const RX_INPROGRESS: u64 = 1 << 5;
}

pub const MACE_UST_MSC: u32 = MACE_BASE + 0x0034_0000;
pub const MACE_UST_MSC_SIZE: u32 = 0x0001_0000;

/// MACE's PCI host bridge register block, at MACE + 0x080000.
pub const MACEPCI_BASE: u32 = MACE_BASE + 0x0008_0000;
pub const MACEPCI_SIZE: u32 = 0x0001_0000;

/// CRIME's rendering-engine block (`crate::dev::o2gfx::CrimeRe`). NetBSD's
/// `crmfb` maps exactly `0x15000000, 0x6000` for this.
pub const CRIME_RE_BASE: u32 = 0x1500_0000;
pub const CRIME_RE_SIZE: u32 = 0x0000_6000;

/// Low memory aliases the base of RAM.
///
/// POST writes a walking pattern to physical 0x00..0x20 from 0xbfc051b4, long
/// after it has sized memory at [`RAM_BASE`]. Without an alias those are
/// unmapped, the store takes a data bus error, and the PROM spins in its
/// exception vector. The Indy models the same idea (`ALIAS_BASE` in
/// `physical.rs`), so the shape is familiar even though the base differs.
///
/// The alias is the first 256 MB of the window ([`LOW_ALIAS_SIZE`]); memory
/// above that is only at its own address. The lowest device sits at
/// 0x14000000, so the alias cannot shadow one.

/// The memory window CRIME decodes: 1 GB from [`RAM_BASE`], which its eight
/// bank controls place banks of 32 or 128 MB in. How, and what an absent bank
/// answers, is in [`crate::dev::crime_mem`].
use crate::dev::crime_mem::{Decode, O2Memory};

// The O2's devices live one to a file under `crate::dev`; their names stay
// reachable here for the bus and the machine.
pub use crate::dev::crime::*;
pub use crate::dev::ds1687::*;
pub use crate::dev::mace_isa::*;
pub use crate::dev::mace_pci::*;
pub use crate::dev::mace_ps2::*;
pub use crate::dev::mace_sdma::*;
pub use crate::dev::mace_ust::*;
pub use crate::dev::ns16550::*;
pub use crate::dev::onewire::*;


pub const RAM_WINDOW_SIZE: u32 = 8 * 128 * 1024 * 1024;
/// CRIME's no-ECC alias of the memory window: the same memory as
/// [`RAM_BASE`], read without ECC checking, at CPU physical 0x00_8000_0000.
/// IRIX maps pages it reads uncached here (`/dev/mem`, `/dev/kmem`). The bus
/// sees physical addresses as 32 bits and has MACE's 64-bit PCI window at
/// 0x8000_0000 ([`PCI_HI_MEM_BASE`]), so the CPU folds this one onto
/// `RAM_BASE` before the bus sees it (`MipsExecutor::phys_alias`).
pub const NO_ECC_BASE: u64 = 0x8000_0000;
/// How much of memory also answers from address 0: the first 256 MB.
pub const LOW_ALIAS_SIZE: u32 = 0x1000_0000;

/// Boot PROM. Same address as the Indy's, which is the one thing that carries
/// over: the MIPS reset vector is 0xbfc00000 on both.
pub const PROM_BASE: u32 = 0x1fc0_0000;
pub const PROM_SIZE: u32 = 0x0008_0000; // 512 KiB

/// Where the `firmware` section's SHDR says it loads: KSEG0 0x81000000.
pub const FIRMWARE_LOAD_VA: u32 = 0x8100_0000;
pub const FIRMWARE_SPAN: u32 = 0x0006_0000;

/// The RTC and its battery-backed NVRAM, at MACE + 0x3a0000. Registers use
/// MACE's ISA spacing, `(reg << 8) + 7`, exactly like the UART.
pub const MACE_RTC: u32 = MACE_BASE + 0x003a_0000;
pub const MACE_RTC_SIZE: u32 = 0x0001_0000;


/// What is plugged into an O2 beyond its memory: the SCSI controllers (the
/// machine attaches their disks from `[scsi.N]`, see `O2Machine::new`), and
/// where the PROM's flash (its environment) is kept.
#[derive(Clone, Debug, Default)]
pub struct Ip32Options {
    /// The two on-board AIC-7880s. Without them the PROM finds nothing to
    /// boot from.
    pub scsi: bool,
    /// A file holding the flash, so `setenv` survives from run to run.
    pub flash_file: Option<std::path::PathBuf>,
}

/// Whether the per-register access counters (MACE's, GBE's, the rendering
/// engine's) run: `IRIS_IP32_HOT`. They are for finding a driver polling a
/// bit nothing sets, and each access took a lock and a map update, every
/// status poll of a GL scene included.
pub fn hot_counters() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("IRIS_IP32_HOT").is_some())
}

/// A register file that remembers what it is told, and counts accesses per
/// offset. Stands in for MACE until there is a reason to build it properly.
///
/// Storing matters more than it looks: the PROM does read-modify-write on
/// several of these (the LED register among them), and a register that always
/// reads back zero silently discards every bit the firmware thought it had
/// set.
pub struct Stub {
    pub name: &'static str,
    reads: std::sync::atomic::AtomicU64,
    writes: std::sync::atomic::AtomicU64,
    /// Per-offset access counts. A stub that is being polled hard is the guest
    /// waiting on a bit we are not setting, and the offset says which.
    hot: Mutex<std::collections::BTreeMap<u32, (u64, u64)>>,
    /// Per-offset stored value.
    cells: Mutex<std::collections::BTreeMap<u32, u64>>,
}

impl Stub {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            reads: std::sync::atomic::AtomicU64::new(0),
            writes: std::sync::atomic::AtomicU64::new(0),
            hot: Mutex::new(std::collections::BTreeMap::new()),
            cells: Mutex::new(std::collections::BTreeMap::new()),
        }
    }
    pub fn counts(&self) -> (u64, u64) {
        (self.reads.load(Ordering::Relaxed), self.writes.load(Ordering::Relaxed))
    }
    /// Offsets by access count, busiest first: `(offset, reads, writes)`.
    pub fn hottest(&self) -> Vec<(u32, u64, u64)> {
        let mut v: Vec<(u32, u64, u64)> =
            self.hot.lock().unwrap().iter().map(|(&o, &(r, w))| (o, r, w)).collect();
        v.sort_by_key(|&(_, r, w)| std::cmp::Reverse(r + w));
        v
    }
    pub(crate) fn r(&self) {
        self.reads.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn w(&self) {
        self.writes.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn cell(&self, addr: u32) -> u64 {
        *self.cells.lock().unwrap().get(&(addr & 0x007f_ffff & !7)).unwrap_or(&0)
    }
    pub fn cell_pub(&self, addr: u32) -> u64 {
        self.cell(addr)
    }
    pub fn set_cell_pub(&self, addr: u32, val: u64) {
        self.set_cell(addr, val)
    }
    fn set_cell(&self, addr: u32, val: u64) {
        self.cells.lock().unwrap().insert(addr & 0x007f_ffff & !7, val);
    }
    pub(crate) fn hit(&self, addr: u32, write: bool) {
        if !hot_counters() {
            return;
        }
        let off = addr & 0x007f_ffff & !3;
        let mut h = self.hot.lock().unwrap();
        let e = h.entry(off).or_insert((0, 0));
        if write { e.1 += 1 } else { e.0 += 1 }
    }
}

/// The values written, by offset; the access counts are for finding a
/// driver waiting on a missing bit, not state.
impl crate::traits::Saveable for Stub {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        for (off, v) in self.cells.lock().unwrap().iter() {
            t.insert(format!("{off:#x}"), crate::snapshot::hex_u64(*v));
        }
        let mut s = toml::map::Map::new();
        s.insert("cells".into(), toml::Value::Table(t));
        toml::Value::Table(s)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut cells = self.cells.lock().unwrap();
        cells.clear();
        if let Some(toml::Value::Table(t)) = crate::snapshot::get_field(v, "cells") {
            for (k, v) in t {
                let off = u32::from_str_radix(k.trim_start_matches("0x"), 16)
                    .map_err(|_| format!("{}: bad offset {k:?}", self.name))?;
                cells.insert(off, crate::snapshot::toml_u64(v).unwrap_or(0));
            }
        }
        Ok(())
    }
}

impl BusDevice for Stub {
    fn read8(&self, a: u64) -> BusRead8 { self.r(); self.hit(a as u32, false); BusRead8::ok(self.cell(a as u32) as u8) }
    fn read16(&self, a: u64) -> BusRead16 { self.r(); self.hit(a as u32, false); BusRead16::ok(self.cell(a as u32) as u16) }
    fn read32(&self, a: u64) -> BusRead32 { self.r(); self.hit(a as u32, false); BusRead32::ok(self.cell(a as u32) as u32) }
    fn read64(&self, a: u64) -> BusRead64 { self.r(); self.hit(a as u32, false); BusRead64::ok(self.cell(a as u32)) }
    fn write8(&self, a: u64, v: u8) -> u32 { self.w(); self.hit(a as u32, true); self.set_cell(a as u32, v as u64); BUS_OK }
    fn write16(&self, a: u64, v: u16) -> u32 { self.w(); self.hit(a as u32, true); self.set_cell(a as u32, v as u64); BUS_OK }
    fn write32(&self, a: u64, v: u32) -> u32 { self.w(); self.hit(a as u32, true); self.set_cell(a as u32, v as u64); BUS_OK }
    fn write64(&self, a: u64, v: u64) -> u32 { self.w(); self.hit(a as u32, true); self.set_cell(a as u32, v); BUS_OK }
}

/// The IP32 physical bus: RAM and the PROM, and the chips that claim the
/// rest, CRIME (with its rendering engine), GBE, VICE and MACE, each of
/// which decodes its own space as HPC3 does on the Indigo2. A short chain
/// of range checks rather than the 64 KB dispatch table `Physical` uses:
/// there are five regions, and an unmapped access is easy to log.
pub struct Ip32Bus {
    pub(crate) ram: std::sync::Arc<O2Memory>,
    /// The flash. Writable, because an O2 keeps its environment variables in
    /// it -- there is no separate NVRAM chip -- and the installer saves
    /// settings there. The user's PROM file is never touched: this is a copy.
    prom: Mutex<Vec<u8>>,
    /// Writes into the flash, so a command protocol would be visible rather
    /// than silently corrupting the image.
    prom_writes: Mutex<Vec<(u32, u8)>>,
    /// Where to keep the flash between runs, if anywhere. Never the image the
    /// caller supplied: settings written by a guest belong in the user's
    /// working directory, not in their copy of the PROM.
    flash_file: Option<std::path::PathBuf>,
    /// CRIME: its registers (interrupts, memory banks) and its rendering
    /// engine, which draws into RAM through its tile tables.
    pub crime: CrimeSpace,
    /// The display back end: scans the frame buffer out of RAM.
    pub gbe: crate::dev::o2gfx::Gbe,
    /// VICE, the video/image compression engine: a register file for now.
    /// IRIX probes it at boot (OpenBSD and NetBSD never look), and a bus
    /// error there panics the kernel.
    pub vice: Stub,
    /// MACE, the I/O ASIC, and every block behind it.
    pub mace: Mace,
    /// Accesses that hit nothing, first 64 distinct kept, as `(addr,
    /// is_write, pc)`. Once full, a miss costs one relaxed load; a guest
    /// polling an absent device would otherwise lock and scan the list on
    /// every poll. Not kept in a `lightning` build.
    unmapped: Mutex<Vec<(u32, bool, u32)>>,
    unmapped_full: std::sync::atomic::AtomicBool,
    /// The ppmem window's base and the RAM bitmap the memory space keeps
    /// (one bit per 64 MB region it has fully mapped), once the machine has
    /// wired them: aligned RAM reads then skip the decode. See `read_bytes`.
    fast_ram: std::sync::OnceLock<(usize, usize)>,
    /// The SCSI controller, when one is attached, so the trace can report
    /// what the driver asked it for.
    pub scsi: Mutex<Vec<std::sync::Arc<crate::dev::aic7880::Aic7880>>>,
}

impl Ip32Bus {
    /// `ram_bytes` of memory in the usual fitting for that size, or, for a
    /// size no O2 can have, one bank of it (power-of-two sizes only). `prom`
    /// must be the raw 512 KiB IP32 PROM image.
    pub fn new(ram_bytes: usize, prom: Vec<u8>) -> Self {
        let mem = O2Memory::with_total((ram_bytes >> 20) as u32)
            .ok()
            .filter(|m| m.total() as usize == ram_bytes)
            .unwrap_or_else(|| O2Memory::single(ram_bytes));
        Self::with_options(mem, prom, Ip32Options::default())
    }

    /// A machine with `mem` fitted, configured by `opts`.
    pub fn with_options(mem: O2Memory, prom: Vec<u8>, opts: Ip32Options) -> Self {
        let ram = std::sync::Arc::new(mem);
        let ust = std::sync::Arc::new(MaceUst::new());
        let crime = std::sync::Arc::new(Crime::new(ust.clone()));
        let bus = Self {
            ram: ram.clone(),
            prom: Mutex::new(prom),
            prom_writes: Mutex::new(Vec::new()),
            flash_file: opts.flash_file.clone(),
            crime: CrimeSpace::new(crime.clone(), ram.clone()),
            vice: Stub::new("vice"),
            gbe: {
                let ust = ust.clone();
                crate::dev::o2gfx::Gbe::new(std::sync::Arc::new(move || ust.now()))
            },
            mace: Mace::new(ram.clone(), crime, ust),
            unmapped: Mutex::new(Vec::new()),
            unmapped_full: std::sync::atomic::AtomicBool::new(false),
            fast_ram: std::sync::OnceLock::new(),
            scsi: Mutex::new(Vec::new()),
        };
        // An experiment for now: the PROM probes slots 1..3 for a vendor ID
        // and finds nothing, so it has no disk and cannot boot. Putting the
        // controller a real O2 has there is how we learn how much of it the
        // PROM insists on.
        bus.gbe.connect_irq(bus.crime.regs.clone());
        if opts.scsi {
            // The controller fetches its work from host memory, which it
            // reaches through the PCI bridge it is attached to.
            let scsi = crate::dev::aic7880::Aic7880::new();
            // The second controller. Real hardware has it whether or not
            // anything is plugged into it, and a kernel that expects two
            // waits on the one that is missing.
            let scsi_b = crate::dev::aic7880::Aic7880::new_at(1);
            bus.mace.pci.attach(IP32_SCSI_SLOT_B, "aic7880-b", scsi_b.clone());
            bus.mace.pci.attach(IP32_SCSI_SLOT, "aic7880", scsi.clone());
            let mut list = bus.scsi.lock().unwrap();
            list.push(scsi);
            list.push(scsi_b);
        }
        // A saved flash carries the guest's environment -- boot paths, console
        // choice, everything the PROM's `setenv` writes -- from one run to
        // the next, which is what makes an install worth doing once.
        if let Some(path) = bus.flash_file.clone() {
            match std::fs::read(&path) {
                Ok(saved) if saved.len() == PROM_SIZE as usize => {
                    *bus.prom.lock().unwrap() = saved;
                    eprintln!("ip32: flash restored from {}", path.display());
                }
                Ok(_) => eprintln!("ip32: flash {} is the wrong size; ignoring", path.display()),
                Err(_) => {}
            }
        }
        bus
    }

    pub fn unmapped(&self) -> Vec<(u32, bool, u32)> {
        self.unmapped.lock().unwrap().clone()
    }

    #[inline]
    fn miss(&self, addr: u32, write: bool) {
        #[cfg(not(feature = "lightning"))]
        {
            if self.unmapped_full.load(Ordering::Relaxed) {
                return;
            }
            let pc = crate::ip32_machine::LAST_PC.load(Ordering::Relaxed);
            let mut u = self.unmapped.lock().unwrap();
            if !u.iter().any(|&(a, w, _)| a == addr && w == write) {
                u.push((addr, write, pc));
                if u.len() == 64 {
                    self.unmapped_full.store(true, Ordering::Relaxed);
                }
            }
        }
        #[cfg(feature = "lightning")]
        let _ = (addr, write);
    }

    /// Memory fitted, in bytes.
    pub fn ram_len(&self) -> u32 {
        self.ram.total()
    }

    /// A CRIME write that may have moved a memory bank.
    /// GBE's picture out of RAM into `out` (R, G, B, A bytes), or None while
    /// the display is off. See [`crate::dev::o2gfx::Gbe::render`].
    pub fn render_screen(&self, out: &mut [u8]) -> Option<(u32, u32)> {
        self.gbe.render_from(&self.ram, out)
    }

    /// Where CRIME's output goes: the CPU's IP2, for a machine whose CPU
    /// runs on its own thread. Without it (the harness) nothing is pushed.
    pub fn connect_cpu_irq(&self, line: crate::irq::IrqLine) {
        self.crime.regs.connect_output(line);
    }

    /// Let aligned RAM reads go straight to the ppmem window: its base and
    /// the bitmap the memory space publishes (the CPU's, as `ppmem_bitmap_ptr`).
    pub fn set_fast_ram(&self, window: *mut u8, bitmap: *mut u64) {
        if !window.is_null() && !bitmap.is_null() {
            let _ = self.fast_ram.set((window as usize, bitmap as usize));
        }
    }

    fn is_ram_addr(addr: u32) -> bool {
        addr < LOW_ALIAS_SIZE || addr >= RAM_BASE
    }

    /// What time and the outside world have done since the last call:
    /// frames from the network, console input, audio and serial DMA moving,
    /// and the beam's timing interrupts. As on the Indy, there is no
    /// refresh of interrupt lines: each device drives its own when its state
    /// changes, from the register access, the event or this tick that
    /// changed it. Returns whether CRIME is interrupting the CPU (IP2).
    pub fn service(&self) -> bool {
        self.mace.service();
        self.gbe.service();
        self.crime.regs.interrupting()
    }

    /// Plug in a PS/2 keyboard on port 0, and a mouse on port 1 if `mouse`.
    /// With a keyboard the PROM and IRIX take the graphics console; without
    /// one, the serial console. The graphics hardware is there either way.
    pub fn attach_keyboard(&mut self, mouse: bool) {
        self.mace.enable_ps2(mouse);
    }

    /// Write the flash out, if a file was named for it.
    pub fn save_flash(&self) {
        let Some(path) = &self.flash_file else { return };
        let p = self.prom.lock().unwrap();
        match std::fs::write(path, &p[..]) {
            Ok(()) => eprintln!("ip32: flash saved to {}", path.display()),
            Err(e) => eprintln!("ip32: flash {}: {e}", path.display()),
        }
    }

    /// The whole flash, PROM and environment, for a snapshot.
    pub fn flash_bytes(&self) -> Vec<u8> {
        self.prom.lock().unwrap().clone()
    }

    /// Put back a flash [`Self::flash_bytes`] took from a machine with the
    /// same PROM size.
    pub fn set_flash(&self, bytes: &[u8]) -> Result<(), String> {
        let mut p = self.prom.lock().unwrap();
        if bytes.len() != p.len() {
            return Err(format!("saved flash is {} bytes, this machine's {}", bytes.len(), p.len()));
        }
        p.copy_from_slice(bytes);
        Ok(())
    }

    /// How many bytes have been written into the flash so far.
    pub fn flash_writes(&self) -> usize {
        self.prom_writes.lock().unwrap().len()
    }

    /// Everything written into the flash, in order.
    pub fn prom_writes(&self) -> Vec<(u32, u8)> {
        self.prom_writes.lock().unwrap().clone()
    }


    fn in_prom(&self, addr: u32) -> bool {
        addr >= PROM_BASE && addr < PROM_BASE + PROM_SIZE
    }

    fn in_vice(&self, addr: u32) -> bool {
        addr >= VICE_BASE && addr < VICE_BASE + VICE_SIZE
    }

    fn in_gbe(&self, addr: u32) -> bool {
        addr >= GBE_BASE && addr < GBE_BASE + GBE_SIZE
    }

    /// Map an address through the low-memory alias, if it falls in it. The
    /// alias covers the first 256 MB only, as on an O2: above that, CRIME and
    /// MACE live at 0x14000000 and 0x1f000000, and memory beyond 256 MB is
    /// reached at its own address above `RAM_BASE`.
    fn resolve(&self, addr: u32) -> u32 {
        if addr < LOW_ALIAS_SIZE {
            RAM_BASE.wrapping_add(addr)
        } else {
            addr
        }
    }

    fn read_bytes(&self, addr: u32, n: usize) -> Option<u64> {
        let addr = self.resolve(addr);
        // Mapped RAM, read straight from the window: IRIX's idle loop and
        // uncached kernel loads came here thousands of times a second, each
        // decoding the address and assembling the value a byte at a time.
        // An aligned word is a host u32 in the window's layout (see
        // O2Memory::get_be).
        if let Some(&(win, bm)) = self.fast_ram.get() {
            let bits = unsafe { std::ptr::read_volatile(bm as *const u64) };
            if bits & (1u64 << (addr >> crate::ppmem::BITMAP_SHIFT)) != 0 && addr as usize % n == 0 {
                let p = win + addr as usize;
                let v = unsafe {
                    match n {
                        1 => *((p ^ 3) as *const u8) as u64,
                        2 => ((p ^ 2) as *const u16).read_unaligned() as u64,
                        4 => (p as *const u32).read_unaligned() as u64,
                        _ => ((p as *const u32).read_unaligned() as u64) << 32 | ((p + 4) as *const u32).read_unaligned() as u64,
                    }
                };
                return Some(v);
            }
        }
        {
            let r = &*self.ram;
            match r.decode(addr) {
                // A bank control matches but nothing is fitted: what is left
                // on the memory bus, which POST has just cleared to zero.
                Decode::Empty => return Some(0),
                Decode::Ram(i) => {
                    let mut v = 0u64;
                    for k in 0..n {
                        v = (v << 8) | r[i + k] as u64;
                    }
                    return Some(v);
                }
                Decode::None => {}
            }
        }
        if self.in_prom(addr) {
            let p = self.prom.lock().unwrap();
            let i = (addr - PROM_BASE) as usize;
            if i + n > p.len() {
                return None;
            }
            let mut v = 0u64;
            for k in 0..n {
                v = (v << 8) | p[i + k] as u64;
            }
            return Some(v);
        }
        None
    }

    fn write_bytes(&self, addr: u32, n: usize, val: u64) -> bool {
        let addr = self.resolve(addr);
        {
            let r = &*self.ram;
            match r.decode(addr) {
                // Writes into an empty bank are swallowed by the memory controller.
                Decode::Empty => return true,
                Decode::Ram(i) => {
                    r.write(i, &val.to_be_bytes()[8 - n..]);
                    return true;
                }
                Decode::None => {}
            }
        }
        if self.in_prom(addr) {
            let mut p = self.prom.lock().unwrap();
            let i = (addr - PROM_BASE) as usize;
            if i + n > p.len() {
                return false;
            }
            {
                let mut log = self.prom_writes.lock().unwrap();
                for k in 0..n {
                    if log.len() < 512 {
                        log.push((addr + k as u32, (val >> (8 * (n - 1 - k))) as u8));
                    }
                }
            }
            for k in 0..n {
                p[i + k] = (val >> (8 * (n - 1 - k))) as u8;
            }
            return true;
        }
        false
    }
}

macro_rules! bus_read {
    ($self:ident, $addr:expr, $n:expr, $ok:path, $conv:expr) => {{
        let addr = $addr;
        if CrimeSpace::contains(addr) || crate::dev::mace::Mace::contains(addr) {
            unreachable!("device reads are dispatched before this point")
        }
        match $self.read_bytes(addr, $n) {
            Some(v) => $ok($conv(v)),
            None => {
                $self.miss(addr, false);
                $ok($conv(0u64))
            }
        }
    }};
}

// The address map: CRIME (its registers and the rendering engine), GBE,
// VICE and MACE (with the PCI bridge's windows) each decode their own
// space; what is left is memory and the PROM.
// The decode is 32 bits wide (see [`NO_ECC_BASE`] and `PCI_HI_MEM_BASE` for
// the two places above 4 GB this folds): `a` is the address the decode
// helpers, memory and the PROM take; devices get the bus address.
impl Ip32Bus {
    fn read8_raw(&self, addr: u64) -> BusRead8 {
        let a = addr as u32;
        if CrimeSpace::contains(a) {
            return self.crime.read8(addr);
        }
        if self.in_gbe(a) {
            return BusRead8::ok(self.gbe.read32(addr & !3).data as u8);
        }
        if self.in_vice(a) {
            return self.vice.read8(addr);
        }
        if Mace::contains(a) {
            return self.mace.read8(addr);
        }
        bus_read!(self, a, 1, BusRead8::ok, |v: u64| v as u8)
    }

    fn read16_raw(&self, addr: u64) -> BusRead16 {
        let a = addr as u32;
        if CrimeSpace::contains(a) {
            return self.crime.read16(addr);
        }
        if self.in_gbe(a) {
            return BusRead16::ok(self.gbe.read32(addr & !3).data as u16);
        }
        if self.in_vice(a) {
            return self.vice.read16(addr);
        }
        if Mace::contains(a) {
            return self.mace.read16(addr);
        }
        bus_read!(self, a, 2, BusRead16::ok, |v: u64| v as u16)
    }

    fn read32_raw(&self, addr: u64) -> BusRead32 {
        let a = addr as u32;
        if CrimeSpace::contains(a) {
            return self.crime.read32(addr);
        }
        if self.in_gbe(a) {
            return self.gbe.read32(addr);
        }
        if self.in_vice(a) {
            return self.vice.read32(addr);
        }
        if Mace::contains(a) {
            return self.mace.read32(addr);
        }
        bus_read!(self, a, 4, BusRead32::ok, |v: u64| v as u32)
    }

    fn read64_raw(&self, addr: u64) -> BusRead64 {
        let a = addr as u32;
        if CrimeSpace::contains(a) {
            return self.crime.read64(addr);
        }
        if self.in_gbe(a) {
            return self.gbe.read64(addr);
        }
        if self.in_vice(a) {
            return self.vice.read64(addr);
        }
        if Mace::contains(a) {
            return self.mace.read64(addr);
        }
        bus_read!(self, a, 8, BusRead64::ok, |v: u64| v)
    }

    fn write8_raw(&self, addr: u64, val: u8) -> u32 {
        let a = addr as u32;
        if CrimeSpace::contains(a) {
            return self.crime.write8(addr, val);
        }
        if self.in_gbe(a) {
            return self.gbe.write32(addr & !3, val as u32);
        }
        if self.in_vice(a) {
            return self.vice.write8(addr, val);
        }
        if Mace::contains(a) {
            return self.mace.write8(addr, val);
        }
        if self.write_bytes(a, 1, val as u64) { BUS_OK } else { self.miss(a, true); BUS_ERR }
    }

    fn write16_raw(&self, addr: u64, val: u16) -> u32 {
        let a = addr as u32;
        if CrimeSpace::contains(a) {
            return self.crime.write16(addr, val);
        }
        if self.in_gbe(a) {
            return self.gbe.write32(addr & !3, val as u32);
        }
        if self.in_vice(a) {
            return self.vice.write16(addr, val);
        }
        if Mace::contains(a) {
            return self.mace.write16(addr, val);
        }
        if self.write_bytes(a, 2, val as u64) { BUS_OK } else { self.miss(a, true); BUS_ERR }
    }

    fn write32_raw(&self, addr: u64, val: u32) -> u32 {
        let a = addr as u32;
        if CrimeSpace::contains(a) {
            return self.crime.write32(addr, val);
        }
        if self.in_gbe(a) {
            return self.gbe.write32(addr, val);
        }
        if self.in_vice(a) {
            return self.vice.write32(addr, val);
        }
        if Mace::contains(a) {
            return self.mace.write32(addr, val);
        }
        if self.write_bytes(a, 4, val as u64) { BUS_OK } else { self.miss(a, true); BUS_ERR }
    }

    fn write64_raw(&self, addr: u64, val: u64) -> u32 {
        let a = addr as u32;
        if CrimeSpace::contains(a) {
            return self.crime.write64(addr, val);
        }
        if self.in_gbe(a) {
            return self.gbe.write64(addr, val);
        }
        if self.in_vice(a) {
            return self.vice.write64(addr, val);
        }
        if Mace::contains(a) {
            return self.mace.write64(addr, val);
        }
        if self.write_bytes(a, 8, val) { BUS_OK } else { self.miss(a, true); BUS_ERR }
    }
}

/// Every access goes to the `_raw` decode above. Around it, a machine with
/// an interrupt sink (`set_irq_sink`) gets its lines recomputed: before a
/// read of CRIME, so INTSTAT is live, and after any write that is not RAM,
/// so a handler's acknowledgement drops IP2 at once instead of leaving it up
/// until the next service tick (OpenBSD: "spurious crime interrupt").
/// The O2's devices answer the monitor through the bus that holds them,
/// as the IOC and HPC3 pass their children's commands through on the
/// other machines.
impl Ip32Bus {
    fn devices(&self) -> Vec<&dyn crate::traits::Device> {
        let mut d: Vec<&dyn crate::traits::Device> = vec![&*self.crime.regs, &self.crime.re, &self.gbe];
        d.extend(self.mace.devices());
        d
    }
}

/// CRIME (which maps RAM again from its bank controls), GBE, VICE, MACE
/// and the SCSI controllers. RAM and the flash are too big for this and go
/// beside it as files ([`Ip32Bus::save_raw`]), as do the disks' overlays
/// ([`crate::dev::aic7880::Aic7880::export_overlays`]).
impl crate::traits::Saveable for Ip32Bus {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("crime".into(), self.crime.save_state());
        t.insert("gbe".into(), self.gbe.save_state());
        t.insert("vice".into(), self.vice.save_state());
        t.insert("mace".into(), self.mace.save_state());
        let scsi = self.scsi.lock().unwrap().iter().map(|c| c.save_state()).collect();
        t.insert("scsi".into(), toml::Value::Array(scsi));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        use crate::snapshot::get_field;
        if let Some(m) = get_field(v, "mace") {
            self.mace.load_state(m)?;
        }
        if let Some(c) = get_field(v, "crime") {
            self.crime.load_state(c)?;
        }
        if let Some(g) = get_field(v, "gbe") {
            self.gbe.load_state(g)?;
        }
        if let Some(x) = get_field(v, "vice") {
            self.vice.load_state(x)?;
        }
        if let Some(toml::Value::Array(a)) = get_field(v, "scsi") {
            let scsi = self.scsi.lock().unwrap();
            if a.len() != scsi.len() {
                return Err(format!("saved state has {} SCSI controllers, this machine {}", a.len(), scsi.len()));
            }
            for (c, s) in scsi.iter().zip(a) {
                c.load_state(s)?;
            }
        }
        Ok(())
    }
}

impl Ip32Bus {
    /// A cold reset (CRIME's CONTROL register): the devices with a power-on
    /// state go back to it, and memory and the flash keep their contents, as
    /// on the hardware. The PROM sets up the rest again as it starts.
    pub fn power_on(&self) {
        use crate::traits::Resettable;
        self.crime.regs.power_on();
        let m = &self.mace;
        let devices: [&dyn Resettable; 8] =
            [&*m.ust, &m.rtc, &m.ps2, &m.onewire, &m.com0, &m.com1, &m.sdma, &*m.isa];
        for d in devices {
            d.power_on();
        }
    }

    /// RAM and the flash, as `ram.bin` and `flash.bin` in `dir`.
    pub fn save_raw(&self, dir: &std::path::Path) -> std::io::Result<()> {
        self.ram.save_raw(&dir.join("ram.bin"))?;
        std::fs::write(dir.join("flash.bin"), &*self.prom.lock().unwrap())
    }

    /// What [`Self::save_raw`] wrote, back.
    pub fn load_raw(&self, dir: &std::path::Path) -> std::io::Result<()> {
        self.ram.load_raw(&dir.join("ram.bin"))?;
        let flash = std::fs::read(dir.join("flash.bin"))?;
        let mut prom = self.prom.lock().unwrap();
        if flash.len() != prom.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "saved flash is the wrong size"));
        }
        *prom = flash;
        Ok(())
    }
}

impl crate::traits::Device for Ip32Bus {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        self.devices().iter().flat_map(|d| d.register_commands()).collect()
    }

    fn execute_command(&self, cmd: &str, args: &[&str], writer: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        for d in self.devices() {
            if d.register_commands().iter().any(|(c, _)| c == cmd) {
                return d.execute_command(cmd, args, writer);
            }
        }
        Err("Command not found".to_string())
    }
}

impl BusDevice for Ip32Bus {
    /// Memory pages have generations jitv2 can compile against; the PROM and
    /// devices have none, so nothing there is compiled.
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, addr: u64) -> *const std::sync::atomic::AtomicU64 {
        let addr = self.resolve(addr as u32);
        let r = &*self.ram;
        match r.decode(addr) {
            Decode::Ram(i) => r.gen_ptr(i),
            _ => std::ptr::null(),
        }
    }

    fn read8(&self, addr: u64) -> BusRead8 {
        self.read8_raw(addr)
    }
    fn read16(&self, addr: u64) -> BusRead16 {
        self.read16_raw(addr)
    }
    fn read32(&self, addr: u64) -> BusRead32 {
        self.read32_raw(addr)
    }
    fn read64(&self, addr: u64) -> BusRead64 {
        self.read64_raw(addr)
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        self.write8_raw(addr, val)
    }
    fn write16(&self, addr: u64, val: u16) -> u32 {
        self.write16_raw(addr, val)
    }
    fn write32(&self, addr: u64, val: u32) -> u32 {
        self.write32_raw(addr, val)
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        self.write64_raw(addr, val)
    }
}

#[cfg(test)]
mod tests;

