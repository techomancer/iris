use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::io::Write as IoWrite;

use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BusDevice, Device, BUS_OK};
use crate::devlog::LogModule;
use crate::exp::eval_const_expr;
use crate::cpu::mips_dis;
use crate::dev::mem::{BlackHoleRegion, UnmappedRam};
use crate::ppmem::{MappedMemory, PpMemSpace, PpMemory};

use crate::dev::prom::PromPort;
use crate::dev::mc::MemoryController;
use crate::dev::hpc3::Hpc3;
use crate::dev::ng1::rex3::Rex3;
use crate::dev::gr2::Gr2;
use crate::dev::mgras::Mgras;
use crate::dev::vino::Vino;
use crate::dev::ultra64::Ultra64;

/// The RAM bank implementation: `PpMemory`, host-MMU-backed (see
/// `docs/ppmem-design.md`). It is still a `BusDevice`, so DMA and every other
/// bus-path access keep working whether or not the 4GB window was reserved.
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

    /// Forward the generation counter lookup too — an alias is the same
    /// physical memory, so it must resolve to the same counter.
    ///
    /// Without this the trait default returns null, and `PhysicalCodePage::
    /// claim` maps a null `gen_ptr` onto the shared `NEVER_COMPILABLE_GEN`
    /// (initialised to 0, never bumped). A page reached through the alias then
    /// reads `gen=0` forever: the JIT compiles it, the guest rewrites it via the
    /// real address, no invalidation is ever observed, and stale compiled code
    /// keeps running. Seen live as `j2 pcp` reporting `pfn=0 gen=0 entry_gen=0`
    /// on the TLB refill vector page after the kernel had patched it many times.
    ///
    /// This is the bus-path twin of the ppmem-window bug fixed in
    /// `PpMemSpace::map_alias` — both had to be wrong for the symptom to appear
    /// in every build, and fixing only one leaves the other configuration broken.
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, addr: u32) -> *const std::sync::atomic::AtomicU64 {
        unsafe { (*self.target).gen_ptr(addr.wrapping_add(self.offset)) }
    }
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

/// Where the 512 KB alias at physical 0 points.
///
/// The MC mirrors the bottom 512 KB of *memory*, and which physical address
/// that is depends on the machine: LOMEM_BASE on IP22/IP24, 0x20000000 on
/// IP28, whose RAM starts there and has nothing at lomem at all. With the
/// offset fixed at LOMEM_BASE the whole window read back as zero on IP28.
///
/// That window is not spare space. ARCS builds its system parameter block at
/// physical 0x1000 and its firmware vector table at 0x1800, and a 64-bit sash
/// loads its firmware pointer straight out of 0x1018 — so the PROM's writes
/// were being discarded and sash then dereferenced the null it read back.
///
/// Taken from the machine profile, never from the environment.
fn alias_offset_for(ip28: bool) -> u32 {
    if ip28 { HIMEM_BASE } else { LOMEM_BASE }
}


/// What one 64 KB `device_map` slot should point at after a MEMCFG write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BankSlot {
    Unmapped,
    Bank(usize),
}

/// Decide every `device_map` slot a MEMCFG write can affect: the lomem and
/// himem windows, each bank's placement, and wherever a bank sat outside
/// those windows last time. Returns one entry per slot, in slot order, plus
/// the slots this placement puts outside the windows, for the next call.
///
/// A plan rather than a sequence of stores, for two reasons:
///
/// - **Only the differences get written.** Wiping both 256 MB windows and
///   mapping the banks back over them is about 8200 stores per MEMCFG write,
///   each a non-atomic 16-byte fat pointer, while the MC's DMA worker may be
///   dispatching through the table from its own thread. Collapsing to one
///   entry per slot means a slot that ends up where it started is never
///   touched, not wiped and rewritten.
/// - **A bank placed outside the windows is unmapped again when it moves.**
///   MEMCFG's base field reaches far beyond lomem and himem, and a wipe of
///   only those two left such a slot pointing at a bank the MC had since
///   moved or invalidated.
///
/// Slots past a bank's `limit` stay `Unmapped`, so reads there return 0.
fn plan_bank_slots(
    bank_addrs: &[Option<(u32, u32, u32)>; 4],
    previously_outside: &[u32],
) -> (Vec<(u32, BankSlot)>, Vec<u32>) {
    const WINDOW: u32 = 0x1000_0000;
    let in_window = |phys: u32| {
        (LOMEM_BASE..LOMEM_BASE + WINDOW).contains(&phys)
            || (HIMEM_BASE..HIMEM_BASE + WINDOW).contains(&phys)
    };
    let mut plan = std::collections::BTreeMap::new();
    for base in [LOMEM_BASE, HIMEM_BASE] {
        for idx in (base >> 16)..((base + WINDOW) >> 16) {
            plan.insert(idx, BankSlot::Unmapped);
        }
    }
    for &idx in previously_outside {
        plan.insert(idx, BankSlot::Unmapped);
    }
    let mut outside = Vec::new();
    for (bank, entry) in bank_addrs.iter().enumerate() {
        let Some((conf_base, _addr_mask, limit)) = *entry else { continue };
        for slot in 0..(limit >> 16) {
            let phys = conf_base.wrapping_add(slot << 16);
            plan.insert(phys >> 16, BankSlot::Bank(bank));
            if !in_window(phys) {
                outside.push(phys >> 16);
            }
        }
    }
    (plan.into_iter().collect(), outside)
}

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
    ppmem_space: Option<PpMemSpace>,
    /// Base of ppmem's 4GB data window, or null when ppmem is unavailable.
    /// Cached here so the hot path is a load + shift + test with no `Option`
    /// unwrapping and no walk through `PpMemSpace`.
    ppmem_base: *mut u8,
    /// Base of ppmem's 8MB generation window (jitv2 only).
    #[cfg(feature = "jitv2")]
    ppmem_gen_base: *mut std::sync::atomic::AtomicU64,
    /// The live mapped-region bitmap. Points at the CPU's inline
    /// `MipsCore::ppmem_bitmap` once the CPU claims it, at `PpMemSpace`'s own
    /// `u64` before that — never null, so the test needs no guard.
    ppmem_bitmap: *const u64,
    /// IP28: the bank placement ppmem's window was last built for, so a
    /// MEMCFG write that moves nothing leaves the window alone.
    ppmem_last_placement: Option<([Option<(u32, u32, u32)>; 4], [usize; 4])>,

    pub rex3: Option<Arc<Rex3>>,
    /// Second Newport head (dual-head Indigo2 / `graphics.heads = 2`).
    pub rex3_head1: Option<Arc<Rex3>>,
    /// GR2 graphics (`graphics.board = xz | extreme`), see src/dev/gr2.
    pub gr2: Option<Arc<Gr2>>,
    /// IMPACT/MGRAS graphics (`graphics.board = solidimpact | highimpact | maximpact`).
    pub mgras: Option<Arc<Mgras>>,
    pub ultra64: Option<Arc<Ultra64>>,
    /// Bare-metal test device (`--test-device`), in GIO expansion slot 0.
    pub testdev: Option<Arc<crate::dev::testdev::TestDevice>>,
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

    /// 64 KB slots the last `remap_banks` placed a RAM bank in outside the
    /// lomem and himem windows. MEMCFG's base field reaches well beyond those
    /// two, and a bank parked out there has to be unmapped again when it
    /// moves — see `plan_bank_slots`.
    banks_outside_windows: Vec<u32>,
    /// Where the 512 KB alias at physical 0 points — see `alias_offset_for`.
    alias_offset: u32,
    /// This machine is an IP28. Only used to label the bank-map trace.
    is_ip28: bool,

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
        gr2: Option<Arc<Gr2>>,
        mgras: Option<Arc<Mgras>>,
        ultra64: Option<Arc<Ultra64>>,
        testdev: Option<Arc<crate::dev::testdev::TestDevice>>,
        vino: Vino,
        mc: MemoryController,
        hpc3: Hpc3,
        prom: PromPort,
        // IP28: RAM starts at 0x20000000, so the low-memory alias follows it.
        ip28: bool,
    ) -> Self {
        let host_freq = crate::platform::get_host_tick_frequency();
        let start_tick = crate::platform::get_host_ticks();

        // Create special devices
        let error_bus = ErrorBus::new();
        let cpu_bus_error = CpuBusErrorDevice { mc: mc.clone() };
        let gio_bus_error = GioBusErrorDevice { mc: mc.clone() };
        // Alias targets will be set in build_device_map once Physical is in final location
        let unmapped_ram = UnmappedRam;
        let alias_bus = AliasBus::new(std::ptr::null::<ErrorBus>(), alias_offset_for(ip28));
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

        // The lookup table is filled in init(). Until then it points at a
        // bus-error device rather than at null, although every slot is
        // written before the guest runs: the MC's DMA worker dispatches
        // through this table from its own thread while `remap_banks` rewrites
        // its 16-byte fat pointers non-atomically from the CPU's, and a torn
        // read that pairs a null data pointer with a live vtable is a
        // segfault, not a bus error. A stateless static costs nothing and
        // takes null out of the table for good.
        static BOOT_ERR: ErrorBus = ErrorBus { debug: AtomicBool::new(false) };
        const BOOT_PTR: *const dyn BusDevice = &BOOT_ERR;
        let device_map: [*const dyn BusDevice; 65536] = [BOOT_PTR; 65536];

        // ppmem: reserve the 4GB window over these banks. A failure here is
        // not fatal — the bus path works regardless — so log and carry on
        // rather than refusing to boot.
        let ppmem_space = match PpMemSpace::over(&banks) {
            Ok(sp) => Some(sp),
            Err(e) => {
                dlog_dev!(LogModule::Mc, "ppmem: could not reserve 4GB window ({e}); \
                                          falling back to bus-only access");
                None
            }
        };
        let (ppmem_base, ppmem_bitmap) = match &ppmem_space {
            Some(sp) => (sp.window_base(), sp.bitmap_ptr()),
            None => (std::ptr::null_mut(), std::ptr::null()),
        };
        #[cfg(feature = "jitv2")]
        let ppmem_gen_base = match &ppmem_space {
            Some(sp) => sp.gen_window_base(),
            None => std::ptr::null_mut(),
        };

        Self {
            banks,
            ppmem_space,
            ppmem_base,
            #[cfg(feature = "jitv2")]
            ppmem_gen_base,
            ppmem_bitmap,
            ppmem_last_placement: None,
            rex3,
            rex3_head1,
            gr2,
            mgras,
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
            banks_outside_windows: Vec::new(),
            alias_offset: alias_offset_for(ip28),
            is_ip28: ip28,
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
        let gr2_ptr: Option<*const dyn BusDevice> = self.gr2.as_deref().map(|g| g as *const dyn BusDevice);
        let mgras_ptr: Option<*const dyn BusDevice> = self.mgras.as_deref().map(|m| m as *const dyn BusDevice);
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
        self.device_map[(crate::dev::vino::VINO_BASE >> 16) as usize] = vino_ptr;

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
        } else if let Some(gr2_ptr) = gr2_ptr {
            // GR2 (XZ / Extreme): the whole 4 MB gfx slot; see src/dev/gr2.
            for i in (NEWPORT_BASE >> 16)..((NEWPORT_END - 1) >> 16) + 1 {
                self.device_map[i as usize] = gr2_ptr;
            }
        } else if let Some(mgras_ptr) = mgras_ptr {
            // IMPACT in the graphics slot. The expansion slots stay unmapped so
            // their probes bus-error, as empty slots do.
            for i in (NEWPORT_BASE >> 16)..((NEWPORT_END - 1) >> 16) + 1 {
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
        if let Some(u64_ptr) = ultra64_ptr {
            use crate::dev::ultra64::{GIO_SLOT0_BASE, RAMROM_BASE, RAMROM_SIZE};
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
            use crate::dev::testdev::{TEST_DEV_BASE, TEST_DEV_SIZE};
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

        // Alias: points back into Physical itself with `alias_offset()` added.
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

        // IP28: a MEMCFG write that leaves every bank where it was (IRIX's
        // kernel rewrites MEMCFG1 during boot to fix the refresh bits) must
        // not tear down and rebuild ppmem's window. JIT compile workers and
        // the DMA thread are running by then, and the rebuild is exactly the
        // window in which they used to fault.
        let ppmem_placement_changed = {
            let sizes = [0, 1, 2, 3].map(|i| self.banks[i].size());
            let key = (bank_addrs, sizes);
            let changed = self.ppmem_last_placement != Some(key);
            self.ppmem_last_placement = Some(key);
            changed
        };

        // ppmem: drop every mapping before re-placing the banks. The comment
        // this replaced said the window is safe to leave unmapped because
        // remapping only runs during PROM POST, before DMA; that is not true
        // on IP28 (see above), which is why ppmem scrubs rather than unmaps (see `AddrSpace::scrub`).
        if ppmem_placement_changed {
            if let Some(sp) = &self.ppmem_space {
                sp.clear_mappings();
            }
            // A bank now answering at an address another bank answered at
            // must look changed to the JIT even if the two counters happen to
            // be equal, so move every counter.
            #[cfg(feature = "jitv2")]
            for b in &self.banks {
                b.bump_gen_all();
            }
        }

        for (bank_idx, maybe_bank) in bank_addrs.iter().enumerate() {
            let Some((conf_base, addr_mask, limit)) = *maybe_bank else {
                dlog_dev!(LogModule::Mc, "[MEMCFG] bank {} not mapped", bank_idx);
                continue;
            };

            if self.is_ip28 {
                eprintln!("iris: IP28 experiment: bank {bank_idx} -> base {conf_base:#010x} mask {addr_mask:#010x} limit {limit:#010x}");
            }
            dlog_dev!(LogModule::Mc, "[MEMCFG] bank {} mapped at 0x{:08x}..0x{:08x} addr_mask={:08x} limit={:08x} ({}MB visible, {}MB per rank)",
                bank_idx, conf_base, conf_base + limit,
                addr_mask, limit, limit >> 20, (addr_mask + 1) >> 20);

            self.banks[bank_idx].set_addr_mask(addr_mask);

            // ppmem: express the same placement as real host mappings. An
            // undersized bank repeats to fill `limit`, which is exactly the
            // SIMM mirroring `addr_mask` encodes — see docs/ppmem-design.md §5.
            if let (true, Some(sp)) = (ppmem_placement_changed, &self.ppmem_space) {
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

        // Now point the table at the banks, whose masks are set, storing only
        // the slots whose contents actually change. The DMA worker can be
        // dispatching through this table right now, and each store is a
        // non-atomic 16-byte fat pointer; see `plan_bank_slots`.
        let (plan, outside) = plan_bank_slots(&bank_addrs, &self.banks_outside_windows);
        self.banks_outside_windows = outside;
        for (idx, target) in plan {
            let ptr = match target {
                BankSlot::Unmapped => unmapped_ptr,
                BankSlot::Bank(b) => bank_ptrs[b],
            };
            let slot = &mut self.device_map[idx as usize];
            if !std::ptr::addr_eq(*slot, ptr) {
                *slot = ptr;
            }
        }

        // ppmem: the low-512KB alias of bank 0 (MC spec — the bottom 512KB
        // mirrors 0x08000000..0x0807ffff). Mapped as real pages rather than
        // routed through AliasBus, so it is the same physical memory with no
        // re-dispatch. AliasBus stays installed in device_map as the bus-path
        // equivalent; both see identical memory.
        if let (true, Some(sp)) = (ppmem_placement_changed, &self.ppmem_space) {
            let bank0_mapped = bank_addrs[0].is_some_and(|(base, _, _)| base == self.alias_offset);
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
    #[inline(always)]
    fn ppmem_ptr(&self, addr: u32) -> Option<*mut u64> {
        if !self.ppmem_mapped(addr) {
            return None;
        }
        Some(unsafe { self.ppmem_base.add(addr as usize) as *mut u64 })
    }

    /// Generation counter for a directly-mapped physical address.
    #[cfg(feature = "jitv2")]
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
    #[cfg(feature = "jitv2")]
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

    // Same reason as the scalar pair above: without these, the trait's default
    // bulk loop would run against Physical and call *its* dma_write64 per word,
    // which forwards correctly but one word at a time — losing the entire point
    // of batching. Forward the whole slice so the device's own bulk override
    // (Rex3's single-token push) actually gets a chance to run.
    #[inline(always)]
    fn dma_write64_bulk(&self, addr: u32, vals: &[u64]) -> u32 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        unsafe { (*device_ptr).dma_write64_bulk(addr, vals) }
    }

    #[inline(always)]
    fn dma_read64_bulk(&self, addr: u32, out: &mut [u64]) -> u32 {
        let device_ptr = self.device_map[(addr >> 16) as usize];
        unsafe { (*device_ptr).dma_read64_bulk(addr, out) }
    }

    #[cfg(feature = "jitv2")]
    #[inline(always)]
    fn gen_ptr(&self, addr: u32) -> *const std::sync::atomic::AtomicU64 {
        // ppmem: the counter for a directly-mapped page is at a constant
        // offset in the gen window — a pure shift off the physical address,
        // no bank lookup. See docs/ppmem-design.md §6.2.
        #[cfg(feature = "jitv2")]
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
        if let Some(p) = self.ppmem_ptr(addr) {
            return Some(p as *const u64);
        }
        let device_ptr = self.device_map[(addr >> 16) as usize];
        unsafe { (*device_ptr).mem_ptr(addr) }
    }

    #[inline]
    fn read_block(&self, addr: u32, buf: &mut [u64]) -> u32 {
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


#[cfg(test)]
mod ppmem_tests {
    //! Exercises the real `remap_banks` path — the same call the MC makes on a
    //! MEMCFG write during PROM POST — and checks that the ppmem window ends up
    //! agreeing with the bus.
    use super::*;
    use crate::dev::mc::MemoryController;
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

    #[test]
    fn ip28_512_mb_banks_map_full_gigabyte() {
        use crate::dev::eeprom_93c56::Eeprom93c56;
        use parking_lot::Mutex;
        let mc = MemoryController::new_for_profile(
            Arc::new(Mutex::new(Eeprom93c56::new())), false, [512, 512, 0, 0], true,
        );
        let addrs = mc.parse_memcfg(0x7f20_7f40, 0);
        let banks = [RamBank::new(512), RamBank::new(512), RamBank::new(1), RamBank::new(1)];
        let space = PpMemSpace::over(&banks).expect("reserve window");
        let (plan, _) = plan_bank_slots(&addrs, &[]);
        for (i, entry) in addrs.iter().enumerate() {
            let Some((base, mask, limit)) = *entry else { continue };
            banks[i].set_addr_mask(mask);
            space.map_bank(i, base as u64, limit as u64, mask as u64 + 1).unwrap();
            // Distinct words on both sides of the old 256 MB limit and at
            // the end of each bank must survive on the bus and direct paths.
            let offsets = [0, (256 << 20) - 8, 256 << 20, limit - 8];
            for off in offsets {
                let phys = base + off;
                assert!(plan.contains(&(phys >> 16, BankSlot::Bank(i))));
                let val = 0xC0DE_0000_0000_0000u64 | ((i as u64) << 32) | off as u64;
                banks[i].write64(phys, val);
            }
            for off in offsets {
                let phys = base + off;
                let val = 0xC0DE_0000_0000_0000u64 | ((i as u64) << 32) | off as u64;
                assert_eq!(banks[i].read64(phys).data, val);
                assert_ne!(space.mapped_bitmap() & (1u64 << (phys >> crate::ppmem::BITMAP_SHIFT)), 0);
                let w = unsafe { *(space.window_base().add(phys as usize) as *const u64) };
                assert_eq!(w.rotate_left(32), val, "physical {phys:#x}");
            }
        }
        assert_eq!(plan.iter().filter(|(_, slot)| matches!(slot, BankSlot::Bank(_))).count(), 1024 << 4);
    }

    /// The low-512KB alias must resolve to the SAME generation counter as
    /// LOMEM — through the ppmem window AND through the bus.
    ///
    /// Two independent bugs had to be fixed for this to hold, one per path:
    ///
    /// * **window**: `PpMemSpace::map_alias` skipped mapping the alias's gen
    ///   range whenever it was below host granularity, and 512KB of data needs
    ///   only 1KB of counters. Its justification ("an alias is the same physical
    ///   pages, hence the same counters") is true of DATA — one mmap, two views
    ///   — but false of the gen window, a separate parallel mapping addressed by
    ///   a pure shift.
    /// * **bus**: `AliasBus` forwarded all eight read/write methods but not
    ///   `gen_ptr`, so it fell through to the trait default (null). A null
    ///   `gen_ptr` makes `PhysicalCodePage::claim` use the shared
    ///   `NEVER_COMPILABLE_GEN`, which is initialised to 0 and never bumped.
    ///
    /// Either one alone produces the same live symptom: `j2 pcp` reporting
    /// `pfn=0 gen=0 entry_gen=0` for the TLB refill vector page after the kernel
    /// has patched it repeatedly, because the JIT tracks the page by one address
    /// while the writes bump the counter for the other. The compiled code is
    /// then never invalidated — caught by the `fetchverify` detector as
    /// "STALE COMPILED CODE" at physical 0x48.
    #[cfg(feature = "jitv2")]
    #[test]
    fn low_alias_gen_counter_agrees_with_lomem_on_both_paths() {
        // Exercise `AliasBus` directly — constructing a whole `Physical` needs
        // ten collaborators, and the unit under test is the forwarding itself.
        struct GenBank { ctr: std::sync::atomic::AtomicU64, base: u32 }
        impl BusDevice for GenBank {
            fn read8(&self, _a: u32) -> BusRead8 { BusRead8::ok(0) }
            fn write8(&self, _a: u32, _v: u8) -> u32 { BUS_OK }
            fn read16(&self, _a: u32) -> BusRead16 { BusRead16::ok(0) }
            fn write16(&self, _a: u32, _v: u16) -> u32 { BUS_OK }
            fn read32(&self, _a: u32) -> BusRead32 { BusRead32::ok(0) }
            fn write32(&self, _a: u32, _v: u32) -> u32 { BUS_OK }
            fn read64(&self, _a: u32) -> BusRead64 { BusRead64::ok(0) }
            fn write64(&self, _a: u32, _v: u64) -> u32 { BUS_OK }
            #[cfg(feature = "jitv2")]
            fn gen_ptr(&self, addr: u32) -> *const std::sync::atomic::AtomicU64 {
                // Only the page at `base` has a counter — anything else is a
                // different page and must NOT resolve here.
                if addr & !0xFFF == self.base { &self.ctr as *const _ } else { std::ptr::null() }
            }
        }

        let target = GenBank { ctr: std::sync::atomic::AtomicU64::new(0), base: LOMEM_BASE };
        let target_ptr: *const dyn BusDevice = &target;
        // The real wiring: alias at physical 0 forwards by +LOMEM_BASE.
        let alias = AliasBus::new(target_ptr, LOMEM_BASE);

        let via_alias = alias.gen_ptr(ALIAS_BASE);
        let via_direct = target.gen_ptr(LOMEM_BASE);

        assert!(!via_alias.is_null(),
            "AliasBus must forward gen_ptr — the trait default returns null, and a null \
             gen_ptr sends the page to NEVER_COMPILABLE_GEN (always 0), so stale JIT code \
             is never invalidated");
        assert_eq!(via_alias, via_direct,
            "the alias and the real address must resolve to ONE counter");

        // And it must still translate, not just return something non-null: a
        // page one past the alias base maps to LOMEM_BASE + 0x1000, which this
        // target deliberately has no counter for.
        assert!(alias.gen_ptr(ALIAS_BASE + 0x1000).is_null(),
            "forwarding must apply the offset, not blanket-return the first counter");
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

#[cfg(test)]
mod bank_plan_tests {
    use super::*;

    const MB: u32 = 1 << 20;
    const SLOTS_PER_128MB: usize = (128 * MB >> 16) as usize;

    fn bank(base: u32, size: u32) -> Option<(u32, u32, u32)> {
        Some((base, size - 1, size))
    }

    /// `device_map` in miniature: apply a plan the way `remap_banks` does and
    /// count the stores that actually happen.
    fn apply(map: &mut [BankSlot], plan: &[(u32, BankSlot)]) -> usize {
        let mut stores = 0;
        for &(idx, target) in plan {
            if map[idx as usize] != target {
                map[idx as usize] = target;
                stores += 1;
            }
        }
        stores
    }

    #[test]
    fn every_slot_is_planned_once() {
        // One entry per slot means a slot a bank lands in is never stored as
        // Unmapped first and the bank second, which is the transient a
        // concurrent DMA dispatch could otherwise catch.
        let addrs = [bank(LOMEM_BASE, 128 * MB), None, bank(HIMEM_BASE, 128 * MB), None];
        let (plan, _) = plan_bank_slots(&addrs, &[]);
        assert!(plan.windows(2).all(|w| w[0].0 < w[1].0), "slots sorted and unique");
        let lo = (LOMEM_BASE >> 16) as usize;
        let at = |idx: usize| plan.iter().find(|p| p.0 as usize == idx).unwrap().1;
        assert_eq!(at(lo), BankSlot::Bank(0));
        assert_eq!(at(lo + SLOTS_PER_128MB), BankSlot::Unmapped, "past bank 0's limit");
    }

    #[test]
    fn a_repeated_memcfg_write_stores_nothing() {
        let addrs = [bank(LOMEM_BASE, 128 * MB), bank(LOMEM_BASE + 128 * MB, 128 * MB), None, None];
        let mut map = vec![BankSlot::Unmapped; 65536];
        let (plan, outside) = plan_bank_slots(&addrs, &[]);
        assert_eq!(apply(&mut map, &plan), 2 * SLOTS_PER_128MB);
        let (plan, _) = plan_bank_slots(&addrs, &outside);
        assert_eq!(apply(&mut map, &plan), 0, "an unchanged MEMCFG write must not touch the table");
    }

    #[test]
    fn moving_one_bank_stores_only_that_banks_slots() {
        let before = [bank(LOMEM_BASE, 128 * MB), bank(LOMEM_BASE + 128 * MB, 128 * MB), None, None];
        let after = [bank(LOMEM_BASE, 128 * MB), bank(HIMEM_BASE, 128 * MB), None, None];
        let mut map = vec![BankSlot::Unmapped; 65536];
        let (plan, outside) = plan_bank_slots(&before, &[]);
        apply(&mut map, &plan);
        let (plan, _) = plan_bank_slots(&after, &outside);
        // Unmapped where it was, mapped where it is: nothing else.
        assert_eq!(apply(&mut map, &plan), 2 * SLOTS_PER_128MB);
    }

    #[test]
    fn a_bank_parked_outside_the_windows_is_unmapped_when_it_moves() {
        // MEMCFG's base field reaches well past lomem and himem. A bank parked
        // at 0x60000000 and then moved must not leave those slots pointing at
        // it: the old wipe covered only the two windows.
        const PARKED: u32 = 0x6000_0000;
        let parked = [bank(LOMEM_BASE, 128 * MB), None, None, bank(PARKED, 16 * MB)];
        let moved = [bank(LOMEM_BASE, 128 * MB), None, None, bank(HIMEM_BASE, 16 * MB)];
        let mut map = vec![BankSlot::Unmapped; 65536];

        let (plan, outside) = plan_bank_slots(&parked, &[]);
        apply(&mut map, &plan);
        assert_eq!(map[(PARKED >> 16) as usize], BankSlot::Bank(3));
        assert_eq!(outside.len(), (16 * MB >> 16) as usize);

        let (plan, outside) = plan_bank_slots(&moved, &outside);
        apply(&mut map, &plan);
        assert_eq!(map[(PARKED >> 16) as usize], BankSlot::Unmapped,
                   "a slot outside the windows kept pointing at a bank that moved");
        assert!(outside.is_empty());
    }
}
