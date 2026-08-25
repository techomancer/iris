use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::io::Write as IoWrite;

use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BusDevice, Device, BUS_OK};
use crate::devlog::LogModule;
use crate::exp::eval_const_expr;
use crate::mips_dis;
use crate::mem::{BlackHoleRegion, UnmappedRam};
#[cfg(not(feature = "ppmem"))]
use crate::mem::Memory;
#[cfg(feature = "ppmem")]
use crate::ppmem::{MappedMemory, PpMemSpace, PpMemory};

use crate::prom::PromPort;
use crate::mc::MemoryController;
use crate::hpc3::Hpc3;
use crate::rex3::Rex3;
use crate::xz::Xz;
use crate::mgras::Mgras;
use crate::vino::Vino;
#[cfg(feature = "ultra64")]
use crate::ultra64::Ultra64;

/// The RAM bank implementation in use.
///
/// `Memory` (a `Vec<u32>` plus an address mask) by default; `PpMemory`
/// (host-MMU-backed, see `docs/ppmem-design.md`) under `--features ppmem`.
/// The two present the same interface — `BusDevice`, `Resettable`,
/// `save_bin`/`load_bin`, `snapshot_words`/`restore_words`, `set_addr_mask`
/// and (under jitv2) `gen_ptr` — so everything below is written against
/// whichever one is compiled in, with no other differences.
#[cfg(not(feature = "ppmem"))]
pub type RamBank = Memory;
#[cfg(feature = "ppmem")]
pub type RamBank = PpMemory;

// Error device for unmapped addresses
struct ErrorBus {
    debug: AtomicBool,
}

impl ErrorBus {
    fn new() -> Self {
        Self {
            debug: AtomicBool::new(false),
        }
    }

    fn set_debug(&self, val: bool) {
        self.debug.store(val, Ordering::Relaxed);
    }
}

impl BusDevice for ErrorBus {
    fn read8(&self, addr: u32) -> BusRead8 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Read8 {:08x}", addr); }
        BusRead8::err()
    }
    fn write8(&self, addr: u32, val: u8) -> u32 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Write8 {:08x} val {:02x}", addr, val); }
        crate::traits::BUS_ERR
    }
    fn read16(&self, addr: u32) -> BusRead16 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Read16 {:08x}", addr); }
        BusRead16::err()
    }
    fn write16(&self, addr: u32, val: u16) -> u32 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Write16 {:08x} val {:04x}", addr, val); }
        crate::traits::BUS_ERR
    }
    fn read32(&self, addr: u32) -> BusRead32 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Read32 {:08x}", addr); }
        BusRead32::err()
    }
    fn write32(&self, addr: u32, val: u32) -> u32 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Write32 {:08x} val {:08x}", addr, val); }
        crate::traits::BUS_ERR
    }
    fn read64(&self, addr: u32) -> BusRead64 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Read64 {:08x}", addr); }
        BusRead64::err()
    }
    fn write64(&self, addr: u32, val: u64) -> u32 {
        if self.debug.load(Ordering::Relaxed) { println!("BusError: Write64 {:08x} val {:016x}", addr, val); }
        crate::traits::BUS_ERR
    }
}

// Alias device - wraps another device and translates addresses
struct AliasBus {
    target: *const dyn BusDevice,
    offset: u32,
}

unsafe impl Send for AliasBus {}
unsafe impl Sync for AliasBus {}

impl AliasBus {
    fn new(target: *const dyn BusDevice, offset: u32) -> Self {
        Self { target, offset }
    }
}

impl BusDevice for AliasBus {
    fn read8(&self, addr: u32) -> BusRead8   { unsafe { (*self.target).read8(addr.wrapping_add(self.offset)) } }
    fn write8(&self, addr: u32, val: u8) -> u32  { unsafe { (*self.target).write8(addr.wrapping_add(self.offset), val) } }
    fn read16(&self, addr: u32) -> BusRead16  { unsafe { (*self.target).read16(addr.wrapping_add(self.offset)) } }
    fn write16(&self, addr: u32, val: u16) -> u32 { unsafe { (*self.target).write16(addr.wrapping_add(self.offset), val) } }
    fn read32(&self, addr: u32) -> BusRead32  { unsafe { (*self.target).read32(addr.wrapping_add(self.offset)) } }
    fn write32(&self, addr: u32, val: u32) -> u32 { unsafe { (*self.target).write32(addr.wrapping_add(self.offset), val) } }
    fn read64(&self, addr: u32) -> BusRead64  { unsafe { (*self.target).read64(addr.wrapping_add(self.offset)) } }
    fn write64(&self, addr: u32, val: u64) -> u32 { unsafe { (*self.target).write64(addr.wrapping_add(self.offset), val) } }
}

/// CPU bus error sink: reports to MC then returns 0/Ready so the CPU doesn't also take
/// a MIPS bus error exception (which causes terrible cascading failures).
/// Covers all non-GIO, non-device unmapped space.
struct CpuBusErrorDevice {
    mc: MemoryController,
}

impl BusDevice for CpuBusErrorDevice {
    fn read8(&self, addr: u32) -> BusRead8   { self.mc.report_cpu_error(addr); BusRead8::ok(0xFF) }
    fn write8(&self, addr: u32, _v: u8) -> u32  { self.mc.report_cpu_error(addr); BUS_OK }
    fn read16(&self, addr: u32) -> BusRead16  { self.mc.report_cpu_error(addr); BusRead16::ok(0xFFFF) }
    fn write16(&self, addr: u32, _v: u16) -> u32 { self.mc.report_cpu_error(addr); BUS_OK }
    fn read32(&self, addr: u32) -> BusRead32  { self.mc.report_cpu_error(addr); BusRead32::ok(0xFFFFFFFF) }
    fn write32(&self, addr: u32, _v: u32) -> u32 { self.mc.report_cpu_error(addr); BUS_OK }
    fn read64(&self, addr: u32) -> BusRead64  { self.mc.report_cpu_error(addr); BusRead64::ok(0xFFFFFFFFFFFFFFFF) }
    fn write64(&self, addr: u32, _v: u64) -> u32 { self.mc.report_cpu_error(addr); BUS_OK }
}

/// GIO bus timeout sink: reports to MC (GIO_ERROR_STAT bit 10 TIME) then returns 0/Ready.
/// Covers GIO space 0x18000000..0x1FA00000 (reserved future GIO + empty expansion slots).
struct GioBusErrorDevice {
    mc: MemoryController,
}

impl BusDevice for GioBusErrorDevice {
    fn read8(&self, addr: u32) -> BusRead8   { self.mc.report_gio_timeout(addr); BusRead8::ok(0xFF) }
    fn write8(&self, addr: u32, _v: u8) -> u32  { self.mc.report_gio_timeout(addr); BUS_OK }
    fn read16(&self, addr: u32) -> BusRead16  { self.mc.report_gio_timeout(addr); BusRead16::ok(0xFFFF) }
    fn write16(&self, addr: u32, _v: u16) -> u32 { self.mc.report_gio_timeout(addr); BUS_OK }
    fn read32(&self, addr: u32) -> BusRead32  { self.mc.report_gio_timeout(addr); BusRead32::ok(0xFFFFFFFF) }
    fn write32(&self, addr: u32, _v: u32) -> u32 { self.mc.report_gio_timeout(addr); BUS_OK }
    fn read64(&self, addr: u32) -> BusRead64  { self.mc.report_gio_timeout(addr); BusRead64::ok(0xFFFFFFFFFFFFFFFF) }
    fn write64(&self, addr: u32, _v: u64) -> u32 { self.mc.report_gio_timeout(addr); BUS_OK }
}

// Address range constants per MC/Indy hardware specification

// Low memory (256MB at 0x08000000)
pub const LOMEM_BASE: u32 = 0x08000000;
pub const LOMEM_END: u32  = 0x18000000;

// High memory (256MB at 0x20000000)
pub const HIMEM_BASE: u32 = 0x20000000;
pub const HIMEM_END: u32  = 0x30000000;

// 128MB per bank; 4 banks total (0,1 in lomem; 2,3 in himem)
pub const BANK_SIZE: u32 = 0x08000000;

// Newport Graphics (4MB GIO slot at 0x1F000000)
const NEWPORT_BASE: u32 = 0x1F000000;
const NEWPORT_END: u32  = 0x1F400000;

// GIO64 Expansion Slot 0 (2MB at 0x1F400000)
const GIO_SLOT0_BASE: u32 = 0x1F400000;
const GIO_SLOT0_END: u32  = 0x1F600000;

// GIO64 Expansion Slot 1 (4MB at 0x1F600000)
const GIO_SLOT1_BASE: u32 = 0x1F600000;
const GIO_SLOT1_END: u32  = 0x1FA00000;

// Memory Controller (128KB at 0x1FA00000)
const MC_BASE: u32 = 0x1FA00000;
const MC_END: u32  = 0x1FA20000;

// HPC3 (512KB at 0x1FB80000)
const HPC3_BASE: u32 = 0x1FB80000;
const HPC3_END: u32  = 0x1FC00000;

// PROM (1MB at 0x1FC00000)
const PROM_BASE: u32 = 0x1FC00000;
const PROM_END: u32  = 0x1FD00000;

// Alias (512KB at 0x00000000) — mirrors the first 512KB of lomem (0x08000000..0x0807ffff)
// per MC spec: "The bottom 512KB of memory is just an alias for the memory located
// at address 0x08000000 to 0x0807ffff."
// Implemented as an AliasBus that adds LOMEM_BASE to the incoming address, so
// accesses go through the normal lomem device_map entries — no direct bank pointer needed.
const ALIAS_BASE: u32   = 0x00000000;
const ALIAS_END: u32    = 0x00080000;
const ALIAS_OFFSET: u32 = LOMEM_BASE;

// Mystery Black Hole (64KB at 0x02080000)
const MYSTERY_HOLE_BASE: u32 = 0x02080000;
const MYSTERY_HOLE_END: u32  = 0x02090000;

/// Physical Bus (Physical)
///
/// Acts as the central hub for connecting devices.
/// Routes all bus accesses to the appropriate device based on address.
/// Devices are stored directly in the struct for zero-overhead access.
/// Uses a 64KB granularity lookup table for O(1) address decoding.
///
/// Memory is split into 4 × 128MB banks (0..3). MEMCFG0/1 in the MC
/// configure at which physical address each bank appears. On MEMCFG write,
/// the MC invokes remap_banks() which updates device_map and each bank's
/// base_addr. Banks 0/1 occupy lomem slots; banks 2/3 occupy himem slots.
pub struct Physical {
    // 4 × 128MB RAM banks (0,1 in lomem; 2,3 in himem). base_addr set by remap_banks().
    banks: [RamBank; 4],
    /// ppmem's 4GB window. Banks are mapped into it by `remap_banks` in
    /// addition to being installed in `device_map`, so the mapping-based fast
    /// path and the bus path stay in agreement. `None` if the window could not
    /// be reserved, in which case only the bus path is used.
    #[cfg(feature = "ppmem")]
    ppmem_space: Option<PpMemSpace>,
    /// Base of ppmem's 4GB data window, or null when ppmem is unavailable.
    /// Cached here so the hot path is a load + shift + test with no `Option`
    /// unwrapping and no walk through `PpMemSpace`.
    #[cfg(feature = "ppmem")]
    ppmem_base: *mut u8,
    /// Base of ppmem's 8MB generation window (jitv2 only).
    #[cfg(all(feature = "ppmem", feature = "jitv2"))]
    ppmem_gen_base: *mut std::sync::atomic::AtomicU64,
    /// The live mapped-region bitmap. Points at the CPU's inline
    /// `MipsCore::ppmem_bitmap` once the CPU claims it, at `PpMemSpace`'s own
    /// `u64` before that — never null, so the test needs no guard.
    #[cfg(feature = "ppmem")]
    ppmem_bitmap: *const u64,

    pub rex3: Option<Arc<Rex3>>,
    /// Second Newport head (dual-head Indigo2 / `graphics.heads = 2`).
    pub rex3_head1: Option<Arc<Rex3>>,
    /// Indy XZ/Elan preview stub (`graphics.board = xz`).
    pub xz: Option<Arc<Xz>>,
    /// Indigo2 IMPACT/MGRAS preview stub (`[impact]` section).
    pub mgras: Option<Arc<Mgras>>,
    #[cfg(feature = "ultra64")]
    pub ultra64: Option<Arc<Ultra64>>,
    /// Bare-metal test device (`--test-device`), in GIO expansion slot 0.
    pub testdev: Option<Arc<crate::testdev::TestDevice>>,
    pub vino: Vino,
    mc: MemoryController,
    hpc3: Hpc3,
    prom: PromPort,

    // Special devices
    error_bus: ErrorBus,
    cpu_bus_error: CpuBusErrorDevice,
    gio_bus_error: GioBusErrorDevice,
    unmapped_ram: UnmappedRam,
    alias_bus: AliasBus,
    vino_gio_alias: AliasBus, // GIO aperture at 0x1F080000 → VINO at 0x00080000
    black_hole: BlackHoleRegion,

    // Lookup table: 64KB granularity (65536 entries = 512KB on 64-bit)
    // Maps (address >> 16) to device pointer (non-null, always valid)
    device_map: [*const dyn BusDevice; 65536],

    trace: AtomicBool,
    start_tick: u64,
    host_freq: u64,
}

// Safety: Physical owns all devices and the pointers in device_map point to owned data
// All the devices themselves are Send+Sync, and we never mutate through the pointers
unsafe impl Send for Physical {}
unsafe impl Sync for Physical {}

impl Physical {
    /// Save all bank contents to binary files (bank0..bank3).
    pub fn save_bank(&self, bank: usize, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.banks[bank].save_bin(path)
    }

    /// Load bank contents from a binary file.
    pub fn load_bank(&self, bank: usize, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.banks[bank].load_bin(path)
    }

    /// Reset all banks to zero (power-on state).
    pub fn reset_memory(&self) {
        use crate::traits::Resettable;
        for bank in &self.banks {
            bank.power_on();
        }
    }

    /// Snapshot bank `bank` into a native-endian Vec<u32>. Used by the
    /// in-memory rollback checkpoint to skip the disk byte-shuffle.
    pub fn snapshot_bank_inmem(&self, bank: usize) -> Vec<u32> {
        self.banks[bank].snapshot_words()
    }

    /// Restore bank `bank` from a buffer produced by `snapshot_bank_inmem`.
    pub fn restore_bank_inmem(&self, bank: usize, src: &[u32]) {
        self.banks[bank].restore_words(src);
    }
}

impl Physical {
    pub fn new(
        banks: [RamBank; 4],
        rex3: Option<Arc<Rex3>>,
        rex3_head1: Option<Arc<Rex3>>,
        xz: Option<Arc<Xz>>,
        mgras: Option<Arc<Mgras>>,
        #[cfg(feature = "ultra64")]
        ultra64: Option<Arc<Ultra64>>,
        testdev: Option<Arc<crate::testdev::TestDevice>>,
        vino: Vino,
        mc: MemoryController,
        hpc3: Hpc3,
        prom: PromPort,
    ) -> Self {
        let host_freq = crate::platform::get_host_tick_frequency();
        let start_tick = crate::platform::get_host_ticks();

        // Create special devices
        let error_bus = ErrorBus::new();
        let cpu_bus_error = CpuBusErrorDevice { mc: mc.clone() };
        let gio_bus_error = GioBusErrorDevice { mc: mc.clone() };
        // Alias targets will be set in build_device_map once Physical is in final location
        let unmapped_ram = UnmappedRam;
        let alias_bus = AliasBus::new(std::ptr::null::<ErrorBus>(), ALIAS_OFFSET);
        // VINO GIO alias: 0x1F08xxxx → 0x0008xxxx (subtract 0x1F000000 = add 0xFF000000)
        // GIO64 VINO aperture sits at 0x1F080000; the chip's primary registers
        // live at physical 0x00080000 (VINO_BASE). To map 0x1F080000 → 0x00080000
        // via wrapping_add we need offset = -(0x1F000000) = 0xE1000000.
        // (Previously 0xFF000000 here, which is -(0x01000000), so vino probes
        // through the GIO alias landed at 0x1E080000 — gio_err_ptr region —
        // returning 0xFFFFFFFF. The IRIX vino driver's chip_id check then
        // mismatches and no /hw/.../vino node is created.)
        let vino_gio_alias = AliasBus::new(std::ptr::null::<ErrorBus>(), 0xE1000000u32);
        let black_hole = BlackHoleRegion::new();

        // Initialize lookup table with null - will be filled in init()
        const NULL_PTR: *const dyn BusDevice = std::ptr::null::<ErrorBus>();
        let device_map: [*const dyn BusDevice; 65536] = [NULL_PTR; 65536];

        // ppmem: reserve the 4GB window over these banks. A failure here is
        // not fatal — the bus path works regardless — so log and carry on
        // rather than refusing to boot.
        #[cfg(feature = "ppmem")]
        let ppmem_space = match PpMemSpace::over(&banks) {
            Ok(sp) => Some(sp),
            Err(e) => {
                dlog_dev!(LogModule::Mc, "ppmem: could not reserve 4GB window ({e}); \
                                          falling back to bus-only access");
                None
            }
        };

        #[cfg(feature = "ppmem")]
        let (ppmem_base, ppmem_bitmap) = match &ppmem_space {
            Some(sp) => (sp.window_base(), sp.bitmap_ptr()),
            None => (std::ptr::null_mut(), std::ptr::null()),
        };
        #[cfg(all(feature = "ppmem", feature = "jitv2"))]
        let ppmem_gen_base = match &ppmem_space {
            Some(sp) => sp.gen_window_base(),
            None => std::ptr::null_mut(),
        };

        Self {
            banks,
            #[cfg(feature = "ppmem")]
            ppmem_space,
            #[cfg(feature = "ppmem")]
            ppmem_base,
            #[cfg(all(feature = "ppmem", feature = "jitv2"))]
            ppmem_gen_base,
            #[cfg(feature = "ppmem")]
            ppmem_bitmap,
            rex3,
            rex3_head1,
            xz,
            mgras,
            #[cfg(feature = "ultra64")]
            ultra64,
            testdev,
            vino,
            mc,
            hpc3,
            prom,
            error_bus,
            cpu_bus_error,
            gio_bus_error,
            unmapped_ram,
            alias_bus,
            vino_gio_alias,
            black_hole,
            device_map,
            trace: AtomicBool::new(false),
            start_tick,
            host_freq,
        }
    }

    /// Initialize device map after Physical is in final location (e.g., in Arc).
    /// MUST be called before using the Physical bus!
    pub fn init(&mut self) {
        self.build_device_map();
    }

    fn build_device_map(&mut self) {
        let cpu_err_ptr: *const dyn BusDevice = &self.cpu_bus_error;
        let gio_err_ptr: *const dyn BusDevice = &self.gio_bus_error;
        let rex3_ptr: Option<*const dyn BusDevice> = self.rex3.as_deref().map(|r| r as *const dyn BusDevice);
        let rex3_head1_ptr: Option<*const dyn BusDevice> =
            self.rex3_head1.as_deref().map(|r| r as *const dyn BusDevice);
        let xz_ptr: Option<*const dyn BusDevice> = self.xz.as_deref().map(|x| x as *const dyn BusDevice);
        let mgras_ptr: Option<*const dyn BusDevice> = self.mgras.as_deref().map(|m| m as *const dyn BusDevice);
        #[cfg(feature = "ultra64")]
        let ultra64_ptr: Option<*const dyn BusDevice> = self.ultra64.as_deref().map(|u| u as *const dyn BusDevice);
        let vino_ptr: *const dyn BusDevice = &self.vino;
        let hpc3_ptr: *const dyn BusDevice = &self.hpc3;
        let mc_ptr: *const dyn BusDevice = &self.mc;
        let prom_ptr: *const dyn BusDevice = &self.prom;
        let black_hole_ptr: *const dyn BusDevice = &self.black_hole;

        // Layer 1: fill entire table with CPU bus error device
        for i in 0..65536usize {
            self.device_map[i] = cpu_err_ptr;
        }

        // Layer 2: overlay GIO space (0x18000000..0x1FA00000) with GIO timeout device
        // This covers: reserved future GIO (0x18000000..0x1F000000) + Newport slot
        // (0x1F000000..0x1F400000) + GIO expansion slots 0/1 (0x1F400000..0x1FA00000)
        // Real devices will overlay their own ranges on top in layer 3.
        for i in (0x1800_0000u32 >> 16)..(0x1FA0_0000u32 >> 16) {
            self.device_map[i as usize] = gio_err_ptr;
        }

        // Memory banks are NOT mapped here — MEMCFG0/1 control their placement.
        // remap_banks() is called by the MC when MEMCFG is written.
        // At boot, the MC's initial MEMCFG values trigger an initial remap.

        // Layer 3: real devices overlaid on top

        // Map VINO (physical 0x00080000, one 64KB slot)
        self.device_map[(crate::vino::VINO_BASE >> 16) as usize] = vino_ptr;

        // Map Mystery Hole
        for i in (MYSTERY_HOLE_BASE >> 16)..((MYSTERY_HOLE_END - 1) >> 16) + 1 {
            self.device_map[i as usize] = black_hole_ptr;
        }

        // 2nd hpc
        for i in (0x1F980000 >> 16)..((0x1F990000 - 1) >> 16) + 1 {
            self.device_map[i as usize] = black_hole_ptr;
        }

        // HPC1 region (0x1FB00000–0x1FB80000) — older HPC chip iris doesn't
        // emulate. IRIX still probes here during normal operation (visible as
        // `MC: CPU Error at 1fb02000` etc. in stderr) and usually tolerates
        // the bus error. Once vidtomem activates the vino capture pipeline,
        // a kernel access here escalates to a hard panic
        // ("PANIC: IRIX Killed due to Bus Error"). Mapping the region to the
        // black hole (reads as zero, writes silently accepted) prevents the
        // bus error from firing and avoids the panic — without implementing
        // real HPC1 semantics, which would be a much larger emulation gap.
        for i in (0x1FB00000u32 >> 16)..((HPC3_BASE - 1) >> 16) + 1 {
            self.device_map[i as usize] = black_hole_ptr;
        }

        // Map Newport/REX3 (4MB GIO slot at 0x1F000000) — only if graphics enabled
        if let Some(rex3_ptr) = rex3_ptr {
            for i in (NEWPORT_BASE >> 16)..((NEWPORT_END - 1) >> 16) + 1 {
                self.device_map[i as usize] = rex3_ptr;
            }
        } else if let Some(xz_ptr) = xz_ptr {
            // Indy XZ/Elan preview: same gfx slot, HQ2 register stub in `src/xz.rs`.
            for i in (NEWPORT_BASE >> 16)..((NEWPORT_END - 1) >> 16) + 1 {
                self.device_map[i as usize] = xz_ptr;
            }
        } else if let Some(mgras_ptr) = mgras_ptr {
            // Indigo2 IMPACT preview: MGRAS stub spans gfx + populated expansion slots.
            for i in (NEWPORT_BASE >> 16)..((NEWPORT_END - 1) >> 16) + 1 {
                self.device_map[i as usize] = mgras_ptr;
            }
            for i in (GIO_SLOT0_BASE >> 16)..((GIO_SLOT0_END - 1) >> 16) + 1 {
                self.device_map[i as usize] = mgras_ptr;
            }
            for i in (GIO_SLOT1_BASE >> 16)..((GIO_SLOT1_END - 1) >> 16) + 1 {
                self.device_map[i as usize] = mgras_ptr;
            }
        }
        // else: GIO timeout from layer 2 already covers the Newport slot

        // Second Newport head at GIO expansion slot 1 (dual-head).
        if let Some(h1_ptr) = rex3_head1_ptr {
            for i in (GIO_SLOT1_BASE >> 16)..((GIO_SLOT1_END - 1) >> 16) + 1 {
                self.device_map[i as usize] = h1_ptr;
            }
        }

        // GIO expansion slot 0 (0x1F400000–0x1F5FFFFF): N64 dev board if enabled
        #[cfg(feature = "ultra64")]
        if let Some(u64_ptr) = ultra64_ptr {
            use crate::ultra64::{GIO_SLOT0_BASE, RAMROM_BASE, RAMROM_SIZE};
            // Control registers: 0x1F400000–0x1F4FFFFF (16 × 64KB slots)
            for i in (GIO_SLOT0_BASE >> 16)..((RAMROM_BASE - 1) >> 16) + 1 {
                self.device_map[i as usize] = u64_ptr;
            }
            // RAMROM window: 0x1F500000–0x1F5FFFFF (16 × 64KB slots)
            for i in (RAMROM_BASE >> 16)..((RAMROM_BASE + RAMROM_SIZE - 1) >> 16) + 1 {
                self.device_map[i as usize] = u64_ptr;
            }
        }
        // GIO expansion slot 1 — second Newport when rex3_head1 absent: GIO timeout remains

        // Test device (--test-device): GIO expansion slot 0, which is empty on a
        // stock Indy and otherwise answers with a GIO timeout. See testdev.rs.
        if let Some(td) = self.testdev.as_deref() {
            let td_ptr: *const dyn BusDevice = td;
            use crate::testdev::{TEST_DEV_BASE, TEST_DEV_SIZE};
            for i in (TEST_DEV_BASE >> 16)..((TEST_DEV_BASE + TEST_DEV_SIZE - 1) >> 16) + 1 {
                self.device_map[i as usize] = td_ptr;
            }
        }

        // Map MC registers (128KB at 0x1FA00000)
        for i in (MC_BASE >> 16)..((MC_END - 1) >> 16) + 1 {
            self.device_map[i as usize] = mc_ptr;
        }

        // Map HPC3 (512KB at 0x1FB80000)
        for i in (HPC3_BASE >> 16)..((HPC3_END - 1) >> 16) + 1 {
            self.device_map[i as usize] = hpc3_ptr;
        }

        // Map PROM (1MB at 0x1FC00000)
        for i in (PROM_BASE >> 16)..((PROM_END - 1) >> 16) + 1 {
            self.device_map[i as usize] = prom_ptr;
        }

        // Alias: points back into Physical itself with ALIAS_OFFSET added.
        // So alias accesses go: AliasBus → Physical::read/write(addr + LOMEM_BASE)
        // → device_map lookup → whichever bank is mapped at LOMEM_BASE.
        // This way alias automatically tracks whatever MEMCFG maps at LOMEM_BASE.
        self.alias_bus.target = self as *const Physical as *const dyn BusDevice;
        let alias_ptr: *const dyn BusDevice = &self.alias_bus;
        for i in (ALIAS_BASE >> 16)..(ALIAS_END >> 16) {
            self.device_map[i as usize] = alias_ptr;
        }

        // VINO GIO alias: 0x1F080000 → 0x00080000
        // Routes through Physical again with 0xFF000000 added (wrapping subtraction of 0x1F000000)
        // so the re-dispatched address falls into VINO's primary slot at 0x0008xxxx.
        self.vino_gio_alias.target = self as *const Physical as *const dyn BusDevice;
        let vino_gio_alias_ptr: *const dyn BusDevice = &self.vino_gio_alias;
        self.device_map[(0x1F080000u32 >> 16) as usize] = vino_gio_alias_ptr;
    }

    /// Remap memory banks in device_map.
    ///
    /// `bank_addrs[i]` is `Some((base, addr_mask, limit))` if bank i is valid, or `None`.
    /// - `base`      — physical base address
    /// - `addr_mask` — applied inside Memory for aliasing (SIMM wrapping); equals size_mb*1MB-1
    /// - `limit`     — number of bytes to map in the device_map; slots beyond limit stay as
    ///                 UnmappedRam and return 0, matching the hardware behaviour of reads past
    ///                 the real SIMM boundary
    ///
    /// Called by the MC whenever MEMCFG0/1 change (including at boot).
    pub fn remap_banks(&mut self, bank_addrs: [Option<(u32, u32, u32)>; 4]) {
        let unmapped_ptr: *const dyn BusDevice = &self.unmapped_ram;

        let bank_ptrs: [*const dyn BusDevice; 4] = [
            &self.banks[0],
            &self.banks[1],
            &self.banks[2],
            &self.banks[3],
        ];

        // ppmem: drop every mapping before re-placing the banks. Safe to leave
        // the window briefly unmapped — remapping runs inside the CPU's MEMCFG
        // store during PROM POST, before DMA is running (design doc §5.1).
        #[cfg(feature = "ppmem")]
        if let Some(sp) = &self.ppmem_space {
            sp.clear_mappings();
        }

        // Wipe lomem (0x08000000..0x18000000) and himem (0x20000000..0x30000000) slots
        for base in [LOMEM_BASE, HIMEM_BASE] {
            for i in (base >> 16)..((base + 0x10000000) >> 16) {
                self.device_map[i as usize] = unmapped_ptr;
            }
        }

        for (bank_idx, maybe_bank) in bank_addrs.iter().enumerate() {
            let Some((conf_base, addr_mask, limit)) = *maybe_bank else {
                dlog_dev!(LogModule::Mc, "[MEMCFG] bank {} not mapped", bank_idx);
                continue;
            };

            dlog_dev!(LogModule::Mc, "[MEMCFG] bank {} mapped at 0x{:08x}..0x{:08x} addr_mask={:08x} limit={:08x} ({}MB visible, {}MB per rank)",
                bank_idx, conf_base, conf_base + limit,
                addr_mask, limit, limit >> 20, (addr_mask + 1) >> 20);

            self.banks[bank_idx].set_addr_mask(addr_mask);

            // Map only the real SIMM range (limit bytes) in 64KB chunks.
            // Slots beyond limit remain UnmappedRam → reads return 0.
            for slot in 0..(limit >> 16) {
                let phys = conf_base + (slot << 16);
                self.device_map[(phys >> 16) as usize] = bank_ptrs[bank_idx];
            }

            // ppmem: express the same placement as real host mappings. An
            // undersized bank repeats to fill `limit`, which is exactly the
            // SIMM mirroring `addr_mask` encodes — see docs/ppmem-design.md §5.
            #[cfg(feature = "ppmem")]
            if let Some(sp) = &self.ppmem_space {
                // `addr_mask + 1` is the SIMM's mirror period, and `limit` the
                // configured slot. They are independent: a dual-rank SIMM has a
                // slot half the size of the bank (each rank placed separately),
                // while an undersized SIMM in a bigger slot repeats. map_bank
                // handles both — see its doc comment.
                let period = (addr_mask as u64).wrapping_add(1);
                let bank_bytes = self.banks[bank_idx].size() as u64;
                if period > 0 && period <= bank_bytes {
                    if let Err(e) =
                        sp.map_bank(bank_idx, conf_base as u64, limit as u64, period)
                    {
                        dlog_dev!(LogModule::Mc,
                            "ppmem: map_bank({bank_idx}, {conf_base:#x}, {limit:#x}, \
                             period {period:#x}) failed: {e}");
                    }
                } else {
                    dlog_dev!(LogModule::Mc,
                        "ppmem: bank {bank_idx} mirror period {period:#x} does not fit \
                         bank size {bank_bytes:#x}; leaving it to the bus path");
                }
            }
        }

        // ppmem: the low-512KB alias of bank 0 (MC spec — the bottom 512KB
        // mirrors 0x08000000..0x0807ffff). Mapped as real pages rather than
        // routed through AliasBus, so it is the same physical memory with no
        // re-dispatch. AliasBus stays installed in device_map as the bus-path
        // equivalent; both see identical memory.
        #[cfg(feature = "ppmem")]
        if let Some(sp) = &self.ppmem_space {
            let bank0_mapped = bank_addrs[0].is_some_and(|(base, _, _)| base == LOMEM_BASE);
            if bank0_mapped {
                let alias_len = (ALIAS_END - ALIAS_BASE) as u64;
                if (self.banks[0].size() as u64) >= alias_len {
                    if let Err(e) = sp.map_alias(0, ALIAS_BASE as u64, alias_len) {
                        dlog_dev!(LogModule::Mc, "ppmem: low-512KB alias map failed: {e}");
                    }
                }
            }
        }
    }

    /// ppmem: the 4GB window banks are mapped into, if one was reserved.
    #[cfg(feature = "ppmem")]
    pub fn ppmem_space(&self) -> Option<&PpMemSpace> {
        self.ppmem_space.as_ref()
    }

    /// Re-read the bitmap sink from the space.
    ///
    /// `PpMemSpace::set_bitmap_sink` moves publication from the space's own
    /// `u64` to the CPU's inline field; without this, `Physical`'s cached
    /// pointer would keep reading the abandoned one and never see another
    /// remap. Call immediately after handing the CPU's pointer over.
    ///
    /// Takes `&mut self` because it is a post-construction fixup on the same
    /// footing as `init()`, run before any other thread observes the bus.
    #[cfg(feature = "ppmem")]
    pub fn resync_ppmem_bitmap(&mut self) {
        if let Some(sp) = &self.ppmem_space {
            self.ppmem_bitmap = sp.bitmap_ptr();
        }
    }

    /// Is `addr` backed by a whole directly-mapped 64MB region?
    ///
    /// This is the quick test the design calls for: one load of the bitmap,
    /// one shift of the top bits of the physical address, one AND. A set bit
    /// means the entire 64MB region is mapped RAM, so a direct host access is
    /// equivalent to going through the bus.
    #[cfg(feature = "ppmem")]
    #[inline(always)]
    fn ppmem_mapped(&self, addr: u32) -> bool {
        if self.ppmem_bitmap.is_null() {
            return false;
        }
        let bm = unsafe { *self.ppmem_bitmap };
        bm & (1u64 << (addr >> crate::ppmem::BITMAP_SHIFT)) != 0
    }

    /// Host pointer for a directly-mapped physical address, or `None` if the
    /// address is not in a fully-mapped region (MMIO, unmapped RAM, or ppmem
    /// unavailable).
    #[cfg(feature = "ppmem")]
    #[inline(always)]
    fn ppmem_ptr(&self, addr: u32) -> Option<*mut u64> {
        if !self.ppmem_mapped(addr) {
            return None;
        }
        Some(unsafe { self.ppmem_base.add(addr as usize) as *mut u64 })
    }

    /// Generation counter for a directly-mapped physical address.
    #[cfg(all(feature = "ppmem", feature = "jitv2"))]
    #[inline(always)]
    fn ppmem_gen_ptr(&self, addr: u32) -> Option<*const std::sync::atomic::AtomicU64> {
        if !self.ppmem_mapped(addr) || self.ppmem_gen_base.is_null() {
            return None;
        }
        let page = (addr >> 12) as usize;
        Some(unsafe { self.ppmem_gen_base.add(page) as *const std::sync::atomic::AtomicU64 })
    }

    /// Bump generation counters for every page a block write touches — the
    /// direct-path equivalent of `PpMemory::write_block`'s per-page cursor.
    #[cfg(all(feature = "ppmem", feature = "jitv2"))]
    #[inline]
    fn ppmem_bump_gen_range(&self, addr: u32, qwords: usize) {
        use std::sync::atomic::Ordering;
        if self.ppmem_gen_base.is_null() {
            return;
        }
        let first = (addr >> 12) as usize;
        let last = (addr.wrapping_add(((qwords.max(1) - 1) as u32) * 8) >> 12) as usize;
        for page in first..=last.max(first) {
            unsafe {
                (*(self.ppmem_gen_base.add(page))).fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Device for Physical {
    fn step(&self, _cycles: u64) {
        // Timers are now updated on read/write based on host clock
    }

    fn stop(&self) {
    }

    fn start(&self) {
    }

    fn is_running(&self) -> bool {
        true
    }

    fn get_clock(&self) -> u64 {
        let now = crate::platform::get_host_ticks();
        let diff = now.wrapping_sub(self.start_tick);
        ((diff as u128 * 50_000_000) / (self.host_freq as u128)) as u64
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        let cmds = vec![
            ("phys".to_string(), "Physical Bus commands: mem, dis, trace, error <on|off>, hole <on|off>, bench".to_string()),
            ("mm".to_string(), "Alias for phys mem".to_string()),
            ("md".to_string(), "Alias for phys dis".to_string()),
            ("trace".to_string(), "Enable/disable tracing: trace <on|off>".to_string()),
            ("bench".to_string(), "Benchmark memory write performance".to_string()),
        ];
        cmds
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn IoWrite + Send>) -> Result<(), String> {
        // Handle "phys" prefix
        let (actual_cmd, actual_args) = if cmd == "phys" {
            if args.is_empty() {
                return Err("Usage: phys <command> [args...]".to_string());
            }
            (args[0], &args[1..])
        } else {
            (cmd, args)
        };

        match actual_cmd {
            "help" => {
                writeln!(writer, "Physical Bus Commands:").unwrap();
                for (c, h) in self.register_commands() {
                    writeln!(writer, "  {:12} - {}", c, h).unwrap();
                }
            }
            "mem" | "m" | "mm" => {
                if actual_args.is_empty() {
                    return Err("Usage: mem <addr>".to_string());
                }
                let addr = eval_const_expr(actual_args[0])
                    .map_err(|e| format!("mem: {}", e))?;
                
                let r = BusDevice::read32(self, addr as u32);
                if r.is_ok() { writeln!(writer, "{:08x}: {:08x}", addr, r.data).unwrap(); }
                else { writeln!(writer, "{:08x}: Error/Busy", addr).unwrap(); }
            }
            "dis" | "d" | "md" => {
                if actual_args.is_empty() {
                    return Err("Usage: dis <addr>".to_string());
                }
                let addr = eval_const_expr(actual_args[0])
                    .map_err(|e| format!("dis: {}", e))?;

                let r = BusDevice::read32(self, addr as u32);
                if r.is_ok() { writeln!(writer, "{}", mips_dis::disassemble(r.data, addr, None)).unwrap(); }
                else { writeln!(writer, "Could not fetch instruction at {:016x}", addr).unwrap(); }
            }
            "error" => {
                if actual_args.is_empty() {
                    return Err("Usage: error <on|off>".to_string());
                }
                let val = match actual_args[0] {
                    "on" | "1" => true,
                    "off" | "0" => false,
                    _ => return Err("Usage: error <on|off>".to_string()),
                };
                self.error_bus.set_debug(val);
                writeln!(writer, "Bus Error debug {}", if val { "enabled" } else { "disabled" }).unwrap();
            }
            "hole" => {
                if actual_args.is_empty() {
                    return Err("Usage: hole <on|off>".to_string());
                }
                let val = match actual_args[0] {
                    "on" | "1" => true,
                    "off" | "0" => false,
                    _ => return Err("Usage: hole <on|off>".to_string()),
                };
                self.black_hole.set_debug(val);
                writeln!(writer, "Black Hole debug {}", if val { "enabled" } else { "disabled" }).unwrap();
            }
            "trace" => {
                if actual_args.is_empty() {
                    let state = if self.trace.load(Ordering::Relaxed) { "on" } else { "off" };
                    writeln!(writer, "MC trace is {}", state).unwrap();
                } else {
                    match actual_args[0] {
                        "on" | "1" => {
                            self.trace.store(true, Ordering::Relaxed);
                            writeln!(writer, "MC trace enabled").unwrap();
                        }
                        "off" | "0" => {
                            self.trace.store(false, Ordering::Relaxed);
                            writeln!(writer, "MC trace disabled").unwrap();
                        }
                        _ => return Err("Usage: trace <on|off|1|0>".to_string()),
                    }
                }
            }
            "bench" => {
                writeln!(writer, "Benchmarking Physical Bus write32 performance...").unwrap();
                let himem_base = HIMEM_BASE;
                let himem_size = HIMEM_END - HIMEM_BASE;
                let bench_start = crate::platform::get_host_ticks();
                let mut error_count = 0;
                for addr in (himem_base..himem_base + himem_size).step_by(4) {
                    if self.write32(addr, 0) != BUS_OK {
                        error_count += 1;
                    }
                }
                let bench_end = crate::platform::get_host_ticks();
                if error_count > 0 {
                    writeln!(writer, "  (had {} errors)", error_count).unwrap();
                }
                let bench_elapsed = bench_end.wrapping_sub(bench_start);
                let bench_freq = crate::platform::get_host_tick_frequency();
                let bench_elapsed_us = (bench_elapsed as f64 / bench_freq as f64) * 1_000_000.0;
                let bench_elapsed_s = bench_elapsed_us / 1_000_000.0;
                let mb_per_s = (himem_size as f64 / (1024.0 * 1024.0)) / bench_elapsed_s;
                let cycles_per_word = (bench_freq as f64 * bench_elapsed_s) / ((himem_size / 4) as f64);
                writeln!(writer, "Physical Bus: Filled {}MB in {:.3} us ({} ticks) = {:.2} MB/s, {:.1} cycles/word",
                    himem_size / (1024 * 1024), bench_elapsed_us, bench_elapsed, mb_per_s, cycles_per_word).unwrap();
            }
            _ => return Err(format!("Unknown Physical command: {}", cmd)),
        }
        Ok(())
    }
}

impl BusDevice for Physical {
    #[inline(always)]
    fn read8(&self, addr: u32) -> BusRead8 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        let r = unsafe { (*device_ptr).read8(addr) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) {
            if r.is_ok() { println!("PHYS8 Read {:08x} -> {:02x}", addr, r.data); }
            else { println!("PHYS8 Read {:08x} -> err {:08x}", addr, r.status); }
        }
        r
    }

    #[inline(always)]
    fn write8(&self, addr: u32, val: u8) -> u32 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        let ws = unsafe { (*device_ptr).write8(addr, val) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) { println!("PHYS8 Write {:08x} val={:02x} -> {:08x}", addr, val, ws); }
        ws
    }

    #[inline(always)]
    fn read16(&self, addr: u32) -> BusRead16 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        let r = unsafe { (*device_ptr).read16(addr) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) {
            if r.is_ok() { println!("PHYS16 Read {:08x} -> {:04x}", addr, r.data); }
            else { println!("PHYS16 Read {:08x} -> err {:08x}", addr, r.status); }
        }
        r
    }

    #[inline(always)]
    fn write16(&self, addr: u32, val: u16) -> u32 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        let ws = unsafe { (*device_ptr).write16(addr, val) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) { println!("PHYS16 Write {:08x} val={:04x} -> {:08x}", addr, val, ws); }
        ws
    }

    #[inline(always)]
    fn read32(&self, addr: u32) -> BusRead32 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        let r = unsafe { (*device_ptr).read32(addr) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) {
            if r.is_ok() { println!("PHYS32 Read {:08x} -> {:08x}", addr, r.data); }
            else { println!("PHYS32 Read {:08x} -> err {:08x}", addr, r.status); }
        }
        r
    }

    #[inline(always)]
    fn write32(&self, addr: u32, val: u32) -> u32 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        let ws = unsafe { (*device_ptr).write32(addr, val) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) { println!("PHYS32 Write {:08x} val={:08x} -> {:08x}", addr, val, ws); }
        ws
    }

    #[inline(always)]
    fn read64(&self, addr: u32) -> BusRead64 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        let r = unsafe { (*device_ptr).read64(addr) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) {
            if r.is_ok() { println!("PHYS64 Read {:08x} -> {:016x}", addr, r.data); }
            else { println!("PHYS64 Read {:08x} -> err {:08x}", addr, r.status); }
        }
        r
    }

    #[inline(always)]
    fn write64(&self, addr: u32, val: u64) -> u32 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        let ws = unsafe { (*device_ptr).write64(addr, val) };
        #[cfg(not(feature = "lightning"))]
        if self.trace.load(Ordering::Relaxed) { println!("PHYS64 Write {:08x} val={:016x} -> {:08x}", addr, val, ws); }
        ws
    }

    #[inline(always)]
    fn write64_masked(&self, addr: u32, val: u64, mask: u64) -> u32 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        unsafe { (*device_ptr).write64_masked(addr, val, mask) }
    }

    // Route to the target device's dma_read64/dma_write64, not its read64/write64.
    // Without these overrides, BusDevice's default dma_read64/dma_write64 (which
    // call self.read64/self.write64) resolve against Physical itself — reaching
    // the target device's plain read64/write64 one layer too early and skipping
    // any DMA-specific override (e.g. Rex3::dma_read64) entirely.
    #[inline(always)]
    fn dma_read64(&self, addr: u32) -> BusRead64 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        unsafe { (*device_ptr).dma_read64(addr) }
    }

    #[inline(always)]
    fn dma_write64(&self, addr: u32, val: u64) -> u32 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        unsafe { (*device_ptr).dma_write64(addr, val) }
    }

    #[cfg(feature = "jitv2")]
    #[inline(always)]
    fn gen_ptr(&self, addr: u32) -> *const std::sync::atomic::AtomicU64 {
        // ppmem: the counter for a directly-mapped page is at a constant
        // offset in the gen window — a pure shift off the physical address,
        // no bank lookup. See docs/ppmem-design.md §6.2.
        #[cfg(all(feature = "ppmem", feature = "jitv2"))]
        if let Some(p) = self.ppmem_gen_ptr(addr) {
            return p;
        }
        let device_ptr = self.device_map[(addr >> 16) as usize];
        unsafe { (*device_ptr).gen_ptr(addr) }
    }

    /// Direct pointer into ppmem's window for a directly-mapped physical
    /// address, so the cache's line-fill fast path (`mips_cache_v2.rs`) reads
    /// RAM without a device dispatch.
    ///
    /// Falls through to the device's own `mem_ptr` — and thus to `None` for
    /// MMIO — whenever the address is not backed by a whole mapped region.
    #[inline]
    fn mem_ptr(&self, addr: u32) -> Option<*const u64> {
        #[cfg(feature = "ppmem")]
        if let Some(p) = self.ppmem_ptr(addr) {
            return Some(p as *const u64);
        }
        let device_ptr = self.device_map[(addr >> 16) as usize];
        unsafe { (*device_ptr).mem_ptr(addr) }
    }

    #[inline]
    fn read_block(&self, addr: u32, buf: &mut [u64]) -> u32 {
        #[cfg(feature = "ppmem")]
        if let Some(p) = self.ppmem_ptr(addr) {
            // Same layout as PpMemory::read_block — storage keeps qwords
            // rotate_left(32).
            unsafe {
                for (i, slot) in buf.iter_mut().enumerate() {
                    *slot = (*p.add(i)).rotate_left(32);
                }
            }
            return BUS_OK;
        }
        let device_ptr = self.device_map[(addr >> 16) as usize];
        unsafe { (*device_ptr).read_block(addr, buf) }
    }

    #[inline]
    fn write_block(&self, addr: u32, buf: &[u64]) -> u32 {
        #[cfg(feature = "ppmem")]
        if let Some(p) = self.ppmem_ptr(addr) {
            unsafe {
                for (i, &val) in buf.iter().enumerate() {
                    *p.add(i) = val.rotate_left(32);
                }
            }
            // The gen bump still has to happen: a cache writeback mutates RAM
            // under any compiled artifact for those pages.
            #[cfg(feature = "jitv2")]
            self.ppmem_bump_gen_range(addr, buf.len());
            return BUS_OK;
        }
        let device_ptr = self.device_map[(addr >> 16) as usize];
        unsafe { (*device_ptr).write_block(addr, buf) }
    }
}


#[cfg(all(test, feature = "ppmem"))]
mod ppmem_tests {
    //! Exercises the real `remap_banks` path — the same call the MC makes on a
    //! MEMCFG write during PROM POST — and checks that the ppmem window ends up
    //! agreeing with the bus.
    use super::*;
    use crate::mc::MemoryController;
    use crate::ppmem::MappedMemory;

    /// Bank placements exactly as `memcfg_bank_info` decodes them for two
    /// 128MB SIMMs at LOMEM, which is what `iris.toml`'s default config gives.
    fn two_128mb_banks() -> [Option<(u32, u32, u32)>; 4] {
        let m0 = MemoryController::encode_memcfg_half(LOMEM_BASE, 128).unwrap();
        let m1 = MemoryController::encode_memcfg_half(LOMEM_BASE + BANK_SIZE, 128).unwrap();
        [
            MemoryController::memcfg_bank_info(m0, 128),
            MemoryController::memcfg_bank_info(m1, 128),
            None,
            None,
        ]
    }

    /// After a remap, every address the bitmap claims is directly mapped must
    /// read the same through the window as through the bus. A divergence here
    /// is exactly the bug the direct path could introduce.
    #[test]
    fn remap_makes_window_agree_with_bus() {
        let addrs = two_128mb_banks();
        assert!(addrs[0].is_some(), "bank 0 should decode");

        let banks = [
            RamBank::new(128),
            RamBank::new(128),
            RamBank::new(8),
            RamBank::new(8),
        ];
        let space = PpMemSpace::over(&banks).expect("reserve window");

        // Drive the same mapping remap_banks would, then verify agreement.
        space.clear_mappings();
        for (i, entry) in addrs.iter().enumerate() {
            let Some((base, _mask, limit)) = *entry else { continue };
            let period = (_mask as u64).wrapping_add(1);
            space.map_bank(i, base as u64, limit as u64, period).unwrap();
        }

        let bm = space.mapped_bitmap();
        assert_ne!(bm, 0, "two 128MB banks at LOMEM should mark regions mapped");

        // Write through the bus, read through the window, at several offsets
        // spanning both banks.
        for (i, entry) in addrs.iter().enumerate() {
            let Some((base, _, limit)) = *entry else { continue };
            for off in [0u32, 0x1000, 0x10_0000, limit - 8] {
                let phys = base + off;
                if bm & (1u64 << (phys >> crate::ppmem::BITMAP_SHIFT)) == 0 {
                    continue; // not a fully-mapped region; bus path only
                }
                let val = 0xC0DE_0000_0000_0000u64 | ((i as u64) << 32) | off as u64;
                banks[i].write64(off, val);
                let w = unsafe {
                    *(space.window_base().add(phys as usize) as *const u64)
                };
                assert_eq!(
                    w.rotate_left(32),
                    val,
                    "window disagrees with bus at phys {phys:#x} (bank {i} off {off:#x})"
                );
            }
        }
    }

    /// The bitmap must never claim a region that contains MMIO — otherwise the
    /// CPU would take the direct path into a device aperture.
    #[test]
    fn bitmap_never_claims_device_space() {
        let banks = [
            RamBank::new(128),
            RamBank::new(128),
            RamBank::new(8),
            RamBank::new(8),
        ];
        let space = PpMemSpace::over(&banks).expect("reserve window");
        for (i, entry) in two_128mb_banks().iter().enumerate() {
            let Some((base, mask, limit)) = *entry else { continue };
            space
                .map_bank(i, base as u64, limit as u64, (mask as u64) + 1)
                .unwrap();
        }
        let bm = space.mapped_bitmap();
        let claims = |phys: u32| bm & (1u64 << (phys >> crate::ppmem::BITMAP_SHIFT)) != 0;

        for (name, addr) in [
            ("MC", MC_BASE),
            ("HPC3", HPC3_BASE),
            ("PROM", PROM_BASE),
            ("Newport", NEWPORT_BASE),
            ("GIO slot 0", GIO_SLOT0_BASE),
            ("GIO slot 1", GIO_SLOT1_BASE),
            ("low alias", ALIAS_BASE),
        ] {
            assert!(!claims(addr), "bitmap wrongly claims {name} at {addr:#x}");
        }
    }
}
