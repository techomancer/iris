//! The Indy and Indigo2 (IP22/IP24/IP28) on the physical bus: where their
//! devices and memory answer, and how a MEMCFG write moves the RAM banks.
//!
//! The bus itself is the machine-independent [`Physical`]. This file
//! configures it the way the MC decodes the address space: RAM in the lomem
//! and himem windows as MEMCFG places it, the 512 KB alias at physical 0,
//! the GIO slots, the MC, HPC3 and the PROM, and the MC's error reporting for
//! everything else.

use std::sync::{Arc, Mutex};

use crate::memory_map::{MemKind, MemRange, MemoryMap};
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BusDevice, BUS_OK};
use crate::devlog::LogModule;
use crate::ppmem::MappedMemory;
use crate::ppmem::devmap::{Entry, SLOT_SHIFT};
use crate::physical::Physical;

use crate::dev::prom::PromPort;
use crate::dev::mc::MemoryController;
use crate::dev::hpc3::Hpc3;
use crate::dev::ng1::rex3::Rex3;
use crate::dev::gr2::Gr2;
use crate::dev::mgras::Mgras;
use crate::dev::vino::Vino;
use crate::dev::ultra64::Ultra64;

/// CPU bus error sink: reports to MC then returns 0/Ready so the CPU doesn't also take
/// a MIPS bus error exception (which causes terrible cascading failures).
/// Covers all non-GIO, non-device unmapped space.
struct CpuBusErrorDevice {
    mc: MemoryController,
}

impl BusDevice for CpuBusErrorDevice {
    fn read8(&self, addr: u64) -> BusRead8   { self.mc.report_cpu_error(addr); BusRead8::ok(0xFF) }
    fn write8(&self, addr: u64, _v: u8) -> u32  { self.mc.report_cpu_error(addr); BUS_OK }
    fn read16(&self, addr: u64) -> BusRead16  { self.mc.report_cpu_error(addr); BusRead16::ok(0xFFFF) }
    fn write16(&self, addr: u64, _v: u16) -> u32 { self.mc.report_cpu_error(addr); BUS_OK }
    fn read32(&self, addr: u64) -> BusRead32  { self.mc.report_cpu_error(addr); BusRead32::ok(0xFFFFFFFF) }
    fn write32(&self, addr: u64, _v: u32) -> u32 { self.mc.report_cpu_error(addr); BUS_OK }
    fn read64(&self, addr: u64) -> BusRead64  { self.mc.report_cpu_error(addr); BusRead64::ok(0xFFFFFFFFFFFFFFFF) }
    fn write64(&self, addr: u64, _v: u64) -> u32 { self.mc.report_cpu_error(addr); BUS_OK }
}

/// GIO bus timeout sink: an address in GIO space no device answers (reserved
/// GIO space and empty expansion or graphics slots, 0x18000000..0x1FA00000).
/// The MC records the timeout in GIO_ERROR_ADDR/STAT and, with time-outs
/// enabled (CPUCTRL1.ABORT_EN, set by the PROM at power-on), aborts a read
/// with a bus error: that is how the PROM's and the kernel's `badaddr`
/// probes find a slot empty. Writes are dropped.
struct GioBusErrorDevice {
    mc: MemoryController,
}

impl BusDevice for GioBusErrorDevice {
    fn read8(&self, addr: u64) -> BusRead8 {
        if self.mc.report_gio_timeout(addr) { BusRead8::err() } else { BusRead8::ok(0xFF) }
    }
    fn write8(&self, addr: u64, _v: u8) -> u32 { self.mc.report_gio_timeout(addr); BUS_OK }
    fn read16(&self, addr: u64) -> BusRead16 {
        if self.mc.report_gio_timeout(addr) { BusRead16::err() } else { BusRead16::ok(0xFFFF) }
    }
    fn write16(&self, addr: u64, _v: u16) -> u32 { self.mc.report_gio_timeout(addr); BUS_OK }
    fn read32(&self, addr: u64) -> BusRead32 {
        if self.mc.report_gio_timeout(addr) { BusRead32::err() } else { BusRead32::ok(0xFFFF_FFFF) }
    }
    fn write32(&self, addr: u64, _v: u32) -> u32 { self.mc.report_gio_timeout(addr); BUS_OK }
    fn read64(&self, addr: u64) -> BusRead64 {
        if self.mc.report_gio_timeout(addr) { BusRead64::err() } else { BusRead64::ok(u64::MAX) }
    }
    fn write64(&self, addr: u64, _v: u64) -> u32 { self.mc.report_gio_timeout(addr); BUS_OK }
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
#[cfg(test)]
const GIO_SLOT0_BASE: u32 = 0x1F400000;

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
// An alias on the bus that adds the base of memory to the incoming address, so
// accesses go through the normal lomem decode-table entries — no direct bank pointer needed.
const ALIAS_BASE: u32   = 0x00000000;
const ALIAS_END: u32    = 0x00080000;

/// Where an IP28 can have RAM: from himem, up to four 512 MB banks.
const IP28_RAM_END: u32 = 0xA000_0000;

/// Where the MC can answer with memory on an Indy, an Indigo2 or an IP28.
pub fn memory_map(ip28: bool) -> MemoryMap {
    let range = |name, kind, base: u32, end: u32| MemRange::new(name, kind, base as u64, end as u64);
    let mut ranges = vec![
        range("low alias", MemKind::Alias, ALIAS_BASE, ALIAS_END),
        range("prom", MemKind::Rom, PROM_BASE, PROM_END),
    ];
    if ip28 {
        ranges.push(range("ram", MemKind::Ram, HIMEM_BASE, IP28_RAM_END));
    } else {
        ranges.push(range("lomem", MemKind::Ram, LOMEM_BASE, LOMEM_END));
        ranges.push(range("himem", MemKind::Ram, HIMEM_BASE, HIMEM_END));
    }
    MemoryMap::new(ranges)
}

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


/// What one 64 KB decode-table slot should point at after a MEMCFG write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BankSlot {
    Unmapped,
    Bank(usize),
}

/// Decide every decode-table slot a MEMCFG write can affect: the lomem and
/// himem windows, each bank's placement, and wherever a bank sat outside
/// those windows last time. Returns one entry per slot, in slot order, plus
/// the slots this placement puts outside the windows, for the next call.
///
/// A plan rather than a sequence of stores, for two reasons:
///
/// - **Only the differences get written.** Wiping both 256 MB windows and
///   mapping the banks back over them is about 8200 slots per MEMCFG write,
///   while the MC's DMA worker may be dispatching through the table from its
///   own thread. Collapsing to one entry per slot means a slot that ends up
///   where it started is never touched, not wiped and rewritten, and the
///   table is never seen half-way through a change. (The table now swaps
///   whole pages atomically, so a transient would no longer be a torn
///   pointer; it would still be a slot briefly answering as unmapped.)
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
        for idx in (base >> SLOT_SHIFT)..((base + WINDOW) >> SLOT_SHIFT) {
            plan.insert(idx, BankSlot::Unmapped);
        }
    }
    for &idx in previously_outside {
        plan.insert(idx, BankSlot::Unmapped);
    }
    let mut outside = Vec::new();
    for (bank, entry) in bank_addrs.iter().enumerate() {
        let Some((conf_base, _addr_mask, limit)) = *entry else { continue };
        for slot in 0..(limit >> SLOT_SHIFT) {
            let phys = conf_base.wrapping_add(slot << SLOT_SHIFT);
            plan.insert(phys >> SLOT_SHIFT, BankSlot::Bank(bank));
            if !in_window(phys) {
                outside.push(phys >> SLOT_SHIFT);
            }
        }
    }
    (plan.into_iter().collect(), outside)
}

// Mystery Black Hole (64KB at 0x02080000)
const MYSTERY_HOLE_BASE: u32 = 0x02080000;
const MYSTERY_HOLE_END: u32  = 0x02090000;

/// What the last MEMCFG remap left behind, for the next one.
#[derive(Default)]
struct RemapState {
    /// 64 KB slots the last remap placed a RAM bank in outside the lomem and
    /// himem windows. MEMCFG's base field reaches well beyond those two, and
    /// a bank parked out there has to be unmapped again when it moves — see
    /// `plan_bank_slots`.
    banks_outside_windows: Vec<u32>,
    /// IP28: the bank placement ppmem's window was last built for, so a
    /// MEMCFG write that moves nothing leaves the window alone.
    ppmem_last_placement: Option<([Option<(u32, u32, u32)>; 4], [usize; 4])>,
}

/// The Indy's or Indigo2's devices on its physical bus.
///
/// Memory is four banks (0..3). MEMCFG0/1 in the MC configure at which
/// physical address each bank appears. On a MEMCFG write the MC invokes
/// [`Ip22Bus::remap_banks`], which updates the decode table, each bank's
/// address mask and ppmem's window. Banks 0/1 occupy lomem slots; banks 2/3
/// occupy himem slots.
pub struct Ip22Bus {
    /// The bus everything is mapped on: what the CPU and DMA masters use.
    pub phys: Arc<Physical>,
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
    /// Where the 512 KB alias at physical 0 points — see `alias_offset_for`.
    alias_offset: u32,
    /// This machine is an IP28. Only used to label the bank-map trace.
    is_ip28: bool,
    /// Remaps run on the CPU thread (a MEMCFG store); the lock only makes
    /// that contract checkable.
    remap: Mutex<RemapState>,
}

/// The devices an Indy or Indigo2 maps, for [`Ip22Bus::new`].
pub struct Ip22Devices {
    pub rex3: Option<Arc<Rex3>>,
    pub rex3_head1: Option<Arc<Rex3>>,
    pub gr2: Option<Arc<Gr2>>,
    pub mgras: Option<Arc<Mgras>>,
    pub ultra64: Option<Arc<Ultra64>>,
    pub testdev: Option<Arc<crate::dev::testdev::TestDevice>>,
    pub vino: Vino,
    pub mc: MemoryController,
    pub hpc3: Hpc3,
    pub prom: PromPort,
}

impl Ip22Bus {
    /// Map `d` on `phys` where the MC decodes it. RAM is not placed here:
    /// MEMCFG0/1 control that, through [`Self::remap_banks`].
    pub fn new(phys: Arc<Physical>, d: Ip22Devices, ip28: bool) -> Self {
        let set = |base: u32, end: u32, dev: Arc<dyn BusDevice>, name: &str| {
            phys.map(base as u64, end as u64, dev, name);
        };

        // Whatever is never mapped below answers as a CPU bus error: the
        // whole 32-bit space now, and any slot above it the first time the
        // guest reaches it.
        phys.set_default(Arc::new(CpuBusErrorDevice { mc: d.mc.clone() }), "cpu bus error");

        // GIO space (0x18000000..0x1FA00000) answers with a GIO timeout. This
        // covers: reserved future GIO (0x18000000..0x1F000000) + Newport slot
        // (0x1F000000..0x1F400000) + GIO expansion slots 0/1 (0x1F400000..0x1FA00000)
        // Real devices overlay their own ranges on top below.
        set(0x1800_0000, 0x1FA0_0000, Arc::new(GioBusErrorDevice { mc: d.mc.clone() }), "gio timeout");

        // Memory banks are NOT mapped here — MEMCFG0/1 control their placement.
        // remap_banks() is called by the MC when MEMCFG is written.
        // At boot, the MC's initial MEMCFG values trigger an initial remap.

        // Map VINO (64KB at 0x00080000)
        set(crate::dev::vino::VINO_BASE, crate::dev::vino::VINO_BASE + 0x1_0000,
            Arc::new(d.vino.clone()), "vino");

        // Map Mystery Hole
        phys.map_black_hole(MYSTERY_HOLE_BASE as u64, MYSTERY_HOLE_END as u64);

        // 2nd hpc
        phys.map_black_hole(0x1F980000, 0x1F990000);

        // HPC1 region (0x1FB00000–0x1FB80000) — older HPC chip iris doesn't
        // emulate. IRIX still probes here during normal operation (visible as
        // `MC: CPU Error at 1fb02000` etc. in stderr) and usually tolerates
        // the bus error. Once vidtomem activates the vino capture pipeline,
        // a kernel access here escalates to a hard panic
        // ("PANIC: IRIX Killed due to Bus Error"). Mapping the region to the
        // black hole (reads as zero, writes silently accepted) prevents the
        // bus error from firing and avoids the panic — without implementing
        // real HPC1 semantics, which would be a much larger emulation gap.
        phys.map_black_hole(0x1FB00000, HPC3_BASE as u64);

        // Map Newport/REX3 (4MB GIO slot at 0x1F000000) — only if graphics enabled
        if let Some(rex3) = &d.rex3 {
            set(NEWPORT_BASE, NEWPORT_END, rex3.clone(), "newport (rex3)");
        } else if let Some(gr2) = &d.gr2 {
            // GR2 (XZ / Extreme): the whole 4 MB gfx slot; see src/dev/gr2.
            set(NEWPORT_BASE, NEWPORT_END, gr2.clone(), "gr2");
        } else if let Some(mgras) = &d.mgras {
            // IMPACT in the graphics slot. The expansion slots stay unmapped so
            // their probes bus-error, as empty slots do.
            set(NEWPORT_BASE, NEWPORT_END, mgras.clone(), "impact (mgras)");
        }
        // else: the GIO timeout above already covers the Newport slot

        // Second Newport head at GIO expansion slot 1 (dual-head).
        if let Some(h1) = &d.rex3_head1 {
            set(GIO_SLOT1_BASE, GIO_SLOT1_END, h1.clone(), "newport head 1");
        }

        // GIO expansion slot 0 (0x1F400000–0x1F5FFFFF): N64 dev board if enabled
        if let Some(u64) = &d.ultra64 {
            use crate::dev::ultra64::{GIO_SLOT0_BASE, RAMROM_BASE, RAMROM_SIZE};
            // Control registers: 0x1F400000–0x1F4FFFFF (16 × 64KB slots)
            set(GIO_SLOT0_BASE, RAMROM_BASE, u64.clone(), "ultra64");
            // RAMROM window: 0x1F500000–0x1F5FFFFF (16 × 64KB slots)
            set(RAMROM_BASE, RAMROM_BASE + RAMROM_SIZE, u64.clone(), "ultra64");
        }
        // GIO expansion slot 1 — second Newport when rex3_head1 absent: GIO timeout remains

        // Test device (--test-device): GIO expansion slot 0, which is empty on a
        // stock Indy and otherwise answers with a GIO timeout. See testdev.rs.
        if let Some(td) = &d.testdev {
            use crate::dev::testdev::{TEST_DEV_BASE, TEST_DEV_SIZE};
            set(TEST_DEV_BASE, TEST_DEV_BASE + TEST_DEV_SIZE, td.clone(), "test device");
        }

        // Map MC registers (128KB at 0x1FA00000)
        set(MC_BASE, MC_END, Arc::new(d.mc.clone()), "mc");

        // Map HPC3 (512KB at 0x1FB80000)
        set(HPC3_BASE, HPC3_END, Arc::new(d.hpc3.clone()), "hpc3");

        // Map PROM (1MB at 0x1FC00000)
        set(PROM_BASE, PROM_END, Arc::new(d.prom), "prom");

        // Alias: re-dispatches into the bus with `alias_offset` added, so it
        // reaches whichever bank is mapped at the base of memory and tracks
        // whatever MEMCFG maps there.
        let alias_offset = alias_offset_for(ip28);
        phys.map_alias(ALIAS_BASE as u64, ALIAS_END as u64, alias_offset, "low-memory alias");

        // VINO GIO alias: 0x1F080000 → 0x00080000
        // GIO64 VINO aperture sits at 0x1F080000; the chip's primary registers
        // live at physical 0x00080000 (VINO_BASE). To map 0x1F080000 → 0x00080000
        // via wrapping_add we need offset = -(0x1F000000) = 0xE1000000.
        // (Previously 0xFF000000 here, which is -(0x01000000), so vino probes
        // through the GIO alias landed at 0x1E080000 — gio_err_ptr region —
        // returning 0xFFFFFFFF. The IRIX vino driver's chip_id check then
        // mismatches and no /hw/.../vino node is created.)
        phys.map_alias(0x1F080000, 0x1F090000, 0xE1000000, "vino gio alias");

        Self {
            phys,
            rex3: d.rex3,
            rex3_head1: d.rex3_head1,
            gr2: d.gr2,
            mgras: d.mgras,
            ultra64: d.ultra64,
            testdev: d.testdev,
            vino: d.vino,
            alias_offset,
            is_ip28: ip28,
            remap: Mutex::new(RemapState::default()),
        }
    }

    /// Remap memory banks in the decode table.
    ///
    /// `bank_addrs[i]` is `Some((base, addr_mask, limit))` if bank i is valid, or `None`.
    /// - `base`      — physical base address
    /// - `addr_mask` — applied inside Memory for aliasing (SIMM wrapping); equals size_mb*1MB-1
    /// - `limit`     — number of bytes to map in the decode table; slots beyond limit stay as
    ///                 memory that reads 0, matching the hardware behaviour of reads past
    ///                 the real SIMM boundary
    ///
    /// Called by the MC whenever MEMCFG0/1 change (including at boot), on the
    /// CPU thread.
    pub fn remap_banks(&self, bank_addrs: [Option<(u32, u32, u32)>; 4]) {
        let phys = &*self.phys;
        let banks = phys.banks();
        let mut st = self.remap.lock().unwrap();

        // IP28: a MEMCFG write that leaves every bank where it was (IRIX's
        // kernel rewrites MEMCFG1 during boot to fix the refresh bits) must
        // not tear down and rebuild ppmem's window. JIT compile workers and
        // the DMA thread are running by then, and the rebuild is exactly the
        // window in which they used to fault.
        let ppmem_placement_changed = {
            let sizes = [0, 1, 2, 3].map(|i| banks[i].size());
            let key = (bank_addrs, sizes);
            let changed = st.ppmem_last_placement != Some(key);
            st.ppmem_last_placement = Some(key);
            changed
        };

        // ppmem: drop every mapping before re-placing the banks. The comment
        // this replaced said the window is safe to leave unmapped because
        // remapping only runs during PROM POST, before DMA; that is not true
        // on IP28 (see above), which is why ppmem scrubs rather than unmaps (see `AddrSpace::scrub`).
        if ppmem_placement_changed {
            if let Some(sp) = phys.ppmem_space() {
                sp.clear_mappings();
            }
            // A bank now answering at an address another bank answered at
            // must look changed to the JIT even if the two counters happen to
            // be equal, so move every counter.
            #[cfg(feature = "jitv2")]
            for b in banks {
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

            banks[bank_idx].set_addr_mask(addr_mask);

            // ppmem: express the same placement as real host mappings. An
            // undersized bank repeats to fill `limit`, which is exactly the
            // SIMM mirroring `addr_mask` encodes — see docs/ppmem-design.md §5.
            if let (true, Some(sp)) = (ppmem_placement_changed, phys.ppmem_space()) {
                // `addr_mask + 1` is the SIMM's mirror period, and `limit` the
                // configured slot. They are independent: a dual-rank SIMM has a
                // slot half the size of the bank (each rank placed separately),
                // while an undersized SIMM in a bigger slot repeats. map_bank
                // handles both — see its doc comment.
                let period = (addr_mask as u64).wrapping_add(1);
                let bank_bytes = banks[bank_idx].size() as u64;
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

        // Now point the table at the banks, whose masks are set. The plan
        // is one entry per slot; apply it as runs of one target, and leave
        // a run alone that already answers as planned. See `plan_bank_slots`.
        let (plan, outside) = plan_bank_slots(&bank_addrs, &st.banks_outside_windows);
        st.banks_outside_windows = outside;
        let mut i = 0;
        while i < plan.len() {
            let (start, target) = plan[i];
            let mut j = i + 1;
            while j < plan.len() && plan[j].1 == target && plan[j].0 == plan[j - 1].0 + 1 {
                j += 1;
            }
            let ptr: Entry = match target {
                BankSlot::Unmapped => phys.unmapped_entry(),
                BankSlot::Bank(b) => phys.bank_entry(b),
            };
            phys.fill_slots(start as u64, (j - i) as u64, ptr);
            i = j;
        }

        // ppmem: the low-512KB alias of bank 0 (MC spec — the bottom 512KB
        // mirrors 0x08000000..0x0807ffff). Mapped as real pages rather than
        // routed through the bus alias, so it is the same physical memory with
        // no re-dispatch. The bus alias stays installed in the decode table as
        // the bus-path equivalent; both see identical memory.
        if let (true, Some(sp)) = (ppmem_placement_changed, phys.ppmem_space()) {
            let bank0_mapped = bank_addrs[0].is_some_and(|(base, _, _)| base == self.alias_offset);
            if bank0_mapped {
                let alias_len = (ALIAS_END - ALIAS_BASE) as u64;
                if (banks[0].size() as u64) >= alias_len {
                    if let Err(e) = sp.map_alias(0, ALIAS_BASE as u64, alias_len) {
                        dlog_dev!(LogModule::Mc, "ppmem: low-512KB alias map failed: {e}");
                    }
                }
            }
        }
    }
}


#[cfg(test)]
mod ppmem_tests {
    //! Exercises the real `remap_banks` path — the same call the MC makes on a
    //! MEMCFG write during PROM POST — and checks that the ppmem window ends up
    //! agreeing with the bus.
    use super::*;
    use crate::dev::mc::MemoryController;
    use crate::ppmem::{MappedMemory, PpMemSpace};
    use crate::physical::RamBank;

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
                banks[i].write64(off as u64, val);
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
                assert!(plan.contains(&(phys >> SLOT_SHIFT, BankSlot::Bank(i))));
                let val = 0xC0DE_0000_0000_0000u64 | ((i as u64) << 32) | off as u64;
                banks[i].write64(phys as u64, val);
            }
            for off in offsets {
                let phys = base + off;
                let val = 0xC0DE_0000_0000_0000u64 | ((i as u64) << 32) | off as u64;
                assert_eq!(banks[i].read64(phys as u64).data, val);
                assert_ne!(space.mapped_bitmap() & (1u64 << (phys >> crate::ppmem::BITMAP_SHIFT)), 0);
                let w = unsafe { *(space.window_base().add(phys as usize) as *const u64) };
                assert_eq!(w.rotate_left(32), val, "physical {phys:#x}");
            }
        }
        assert_eq!(plan.iter().filter(|(_, slot)| matches!(slot, BankSlot::Bank(_))).count(), (1024 << 20) >> SLOT_SHIFT);
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
    const SLOTS_PER_128MB: usize = (128 * MB >> SLOT_SHIFT) as usize;

    fn bank(base: u32, size: u32) -> Option<(u32, u32, u32)> {
        Some((base, size - 1, size))
    }

    /// The decode table in miniature: apply a plan the way `remap_banks` does and
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
        let lo = (LOMEM_BASE >> SLOT_SHIFT) as usize;
        let at = |idx: usize| plan.iter().find(|p| p.0 as usize == idx).unwrap().1;
        assert_eq!(at(lo), BankSlot::Bank(0));
        assert_eq!(at(lo + SLOTS_PER_128MB), BankSlot::Unmapped, "past bank 0's limit");
    }

    #[test]
    fn a_repeated_memcfg_write_stores_nothing() {
        let addrs = [bank(LOMEM_BASE, 128 * MB), bank(LOMEM_BASE + 128 * MB, 128 * MB), None, None];
        let mut map = vec![BankSlot::Unmapped; 1 << (32 - SLOT_SHIFT)];
        let (plan, outside) = plan_bank_slots(&addrs, &[]);
        assert_eq!(apply(&mut map, &plan), 2 * SLOTS_PER_128MB);
        let (plan, _) = plan_bank_slots(&addrs, &outside);
        assert_eq!(apply(&mut map, &plan), 0, "an unchanged MEMCFG write must not touch the table");
    }

    #[test]
    fn moving_one_bank_stores_only_that_banks_slots() {
        let before = [bank(LOMEM_BASE, 128 * MB), bank(LOMEM_BASE + 128 * MB, 128 * MB), None, None];
        let after = [bank(LOMEM_BASE, 128 * MB), bank(HIMEM_BASE, 128 * MB), None, None];
        let mut map = vec![BankSlot::Unmapped; 1 << (32 - SLOT_SHIFT)];
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
        let mut map = vec![BankSlot::Unmapped; 1 << (32 - SLOT_SHIFT)];

        let (plan, outside) = plan_bank_slots(&parked, &[]);
        apply(&mut map, &plan);
        assert_eq!(map[(PARKED >> SLOT_SHIFT) as usize], BankSlot::Bank(3));
        assert_eq!(outside.len(), (16 * MB >> SLOT_SHIFT) as usize);

        let (plan, outside) = plan_bank_slots(&moved, &outside);
        apply(&mut map, &plan);
        assert_eq!(map[(PARKED >> SLOT_SHIFT) as usize], BankSlot::Unmapped,
                   "a slot outside the windows kept pointing at a bank that moved");
        assert!(outside.is_empty());
    }
}
