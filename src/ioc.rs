use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::devlog::LogModule;
use std::sync::mpsc;
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device, Resettable, Saveable, MachineEvent};
use crate::snapshot::{get_field, toml_u8, hex_u8};
use crate::z85c30::{Z85c30, IrqCallback};
use crate::pit8254::{Pit8254, TimerCallback};
use crate::mips_core::{CAUSE_IP2, CAUSE_IP3, CAUSE_IP4, CAUSE_IP5, CAUSE_IP6};
use crate::ps2::{Ps2Controller, Ps2Callback};
use crate::hptimer::TimerManager;
use std::io::Write;

pub const IOC_BASE: u32 = 0x1FBD9800;
pub const IOC_SIZE: u32 = 0x100;

/// Fullhouse-only: real physical base of the INT2 register block (HPC3 PBUS
/// PIO channel 4 — same address `crate::hpc3::HPC3_BASE + HPC3_INT2_BASE`
/// resolves to). Lower than `IOC_BASE` (PIO channel 6), so `read8`/`write8`
/// check this window *before* computing `addr - IOC_BASE`, letting callers
/// (hpc3.rs) forward whichever real address they received to the same
/// `BusDevice` methods instead of a separate accessor.
pub const IOC_INT2_BASE: u32 = 0x1FBD9000;
pub const IOC_INT2_SIZE: u32 = INT2_REG_COUNT * 4;

// Register Offsets
pub const IOC_PL_DATA: u32 = 0x00;
pub const IOC_PL_CNTL: u32 = 0x04;
pub const IOC_PL_STAT: u32 = 0x08;
pub const IOC_PL_DMA_CNTL: u32 = 0x0C;
pub const IOC_PL_INT_STAT: u32 = 0x10;
pub const IOC_PL_INT_MASK: u32 = 0x14;
pub const IOC_PL_TIMER1: u32 = 0x18;
pub const IOC_PL_TIMER2: u32 = 0x1C;
pub const IOC_PL_TIMER3: u32 = 0x20;
pub const IOC_PL_TIMER4: u32 = 0x24;

pub const IOC_SERIAL1_CMD: u32 = 0x30;
pub const IOC_SERIAL1_DATA: u32 = 0x34;
pub const IOC_SERIAL2_CMD: u32 = 0x38;
pub const IOC_SERIAL2_DATA: u32 = 0x3C;

pub const IOC_KBD_MOUSE_DATA: u32 = 0x40;
pub const IOC_KBD_MOUSE_CMD: u32 = 0x44;
pub const IOC_GC_SELECT: u32 = 0x48;
pub const IOC_GEN_CNTL: u32 = 0x4C;
pub const IOC_PANEL: u32 = 0x50;
pub const IOC_SYS_ID: u32 = 0x58;
pub const IOC_READ: u32 = 0x60;
pub const IOC_DMA_SEL: u32 = 0x68;
pub const IOC_RESET: u32 = 0x70;
pub const IOC_WRITE: u32 = 0x78;

pub const IOC_INT3_L0_STAT: u32 = 0x80;
pub const IOC_INT3_L0_MASK: u32 = 0x84;
pub const IOC_INT3_L1_STAT: u32 = 0x88;
pub const IOC_INT3_L1_MASK: u32 = 0x8C;
pub const IOC_INT3_MAP_STAT: u32 = 0x90;
pub const IOC_INT3_MAP_MASK0: u32 = 0x94;
pub const IOC_INT3_MAP_MASK1: u32 = 0x98;
pub const IOC_INT3_MAP_POL: u32 = 0x9C;
pub const IOC_INT3_TMR_CLR: u32 = 0xA0;
pub const IOC_INT3_ERR_STAT: u32 = 0xA4;

pub const IOC_TIMER_CNT0: u32 = 0xB0;
pub const IOC_TIMER_CNT1: u32 = 0xB4;
pub const IOC_TIMER_CNT2: u32 = 0xB8;
pub const IOC_TIMER_CTL: u32 = 0xBC;

/// Fullhouse-only `HPC3_EXT_IO_ADDR` (`0x1FBD9900` = `IOC_BASE + 0x100`),
/// one byte past `IOC_SIZE`'s original window — see `IocState.ext_io`'s doc
/// comment. 16-bit hardware register, but IRIX reads it as a 32-bit `uint`.
pub const IOC_EXT_IO: u32 = 0x100;

pub mod l0_regs {
    pub const MAP_INT0: u8 = 1 << 7;
    pub const GRAPHICS: u8 = 1 << 6;
    pub const PARALLEL: u8 = 1 << 5;
    pub const MC_DMA: u8 = 1 << 4;
    pub const ETHERNET: u8 = 1 << 3;
    pub const SCSI1: u8 = 1 << 2;
    pub const SCSI0: u8 = 1 << 1;
    pub const FIFO_FULL: u8 = 1 << 0;
}

pub mod l1_regs {
    pub const VERTICAL_RETRACE: u8 = 1 << 7; // LIO_GIO_2 / vert retrace
    pub const VIDEO_VSYNC: u8      = 1 << 6; // LIO_VIDEO
    pub const AC_FAIL: u8          = 1 << 5; // LIO_AC
    pub const HPC_DMA: u8          = 1 << 4; // LIO_HPC3
    pub const MAP_INT1: u8         = 1 << 3; // Mappable Interrupt 1 summary (map_stat & map_mask1)
    pub const GP2: u8              = 1 << 2; // General Purpose LOCAL1_N<2>, active low
    pub const PANEL: u8            = 1 << 1; // Panel: PWR_INT_N / UP_INT_N / DOWN_INT_N
    pub const GP0: u8              = 1 << 0; // General Purpose LOCAL1_N<0>, active low
}

pub mod map_regs {
    pub const SERIAL: u8    = 1 << 5;
    pub const KBD_MOUSE: u8 = 1 << 4;
    // On Indy (IP24/Guinness): GIO EXP slot 0 interrupt = LIO_2 bit 6,
    // dispatched via lcl2_intr → fires as L0/IP2 (VECTOR_GIOEXP0 = 22 = lcl_id 2, level 6).
    pub const GIO_EXP0: u8  = 1 << 6;
    pub const GIO_EXP1: u8  = 1 << 7;
    // IP22 fullhouse: same MAP bits 6–7 are GFX FIFO drain feedback (not expansion).
    pub const GFX_DRAIN0: u8 = 1 << 6;
    pub const GFX_DRAIN1: u8 = 1 << 7;
}

/// Fullhouse-only PORT_CONFIG bits (`IP22.h`'s `PCON_*`). All reset/clear
/// lines are active-low (`_N` suffix): 0 = asserted, 1 = deasserted/normal.
pub mod pcon_regs {
    pub const DMA_SYNC_SEL: u8       = 1 << 0; // 1=slot1 0=slot0
    pub const SG_RESET_N: u8         = 1 << 1; // reset GFX slot
    pub const S0_RESET_N: u8         = 1 << 2; // reset EXP0 slot
    pub const CLR_SG_RETRACE_N: u8   = 1 << 3; // clear GFX slot retrace latch
    pub const CLR_S0_RETRACE_N: u8   = 1 << 4; // clear EXP0 slot retrace latch
}

/// Fullhouse-only `HPC3_EXT_IO_ADDR` bits (`kern/sys/hpc3.h`'s `EXTIO_*`).
/// All active-low: 0 = interrupt/condition pending, 1 = idle. `IRQ_1` =
/// graphics (`GIO_INTERRUPT_1`), `IRQ_2` = fifo (`GIO_INTERRUPT_0`), `IRQ_3`
/// = video vsync (unrelated to the 3 GIO vectors — `VECTOR_VIDEO`).
pub mod ext_io_regs {
    pub const SG_STAT_0: u16    = 1 << 0;
    pub const SG_STAT_1: u16    = 1 << 1;
    pub const SG_RETRACE: u16   = 1 << 8;  // GioSgRetrace
    pub const SG_IRQ_1: u16     = 1 << 9;  // GioSgGraphics
    pub const SG_IRQ_2: u16     = 1 << 10; // GioSgFifo
    pub const SG_IRQ_3: u16     = 1 << 11; // vid.vsync
    pub const S0_STAT_0: u16    = 1 << 2;
    pub const S0_STAT_1: u16    = 1 << 3;
    pub const S0_RETRACE: u16   = 1 << 12; // GioS0Retrace
    pub const S0_IRQ_1: u16     = 1 << 13; // GioS0Graphics
    pub const S0_IRQ_2: u16     = 1 << 14; // GioS0Fifo
    pub const S0_IRQ_3: u16     = 1 << 15; // vid.vsync
    pub const GIO_33MHZ: u16    = 1 << 7;
    pub const EISA_BUSERR: u16  = 1 << 6;
    pub const MC_BUSERR: u16    = 1 << 5;
    pub const HPC3_BUSERR: u16  = 1 << 4;
    /// All-idle reset value: every active-low bit deasserted (1).
    pub const IDLE: u16 = 0xFFFF;
}

/// IP22 fullhouse exposes the *same* INT3-shaped interrupt registers
/// (`l0_stat`/`l0_mask`/`l1_stat`/`l1_mask`/`map_stat`/`map_mask0`/
/// `map_mask1`, plus the embedded PIT) as Indy, just at a second, more
/// compact address — HPC3 PBUS PIO channel 4 (`crate::hpc3::HPC3_INT2_BASE`)
/// — instead of PIO channel 6 (`IOC_BASE`). Confirmed against MAME's
/// `src/mame/sgi/ioc2.cpp`: `ioc2_guinness_device` and
/// `ioc2_full_house_device` both derive from the same `ioc2_device` base
/// and share identical `INT3_LOCAL0_*`/`INT3_LOCAL1_*` bit assignments and
/// IRQ source wiring (gio_int0/1/2_w, scsi0/1_int_w, enet_int_w,
/// mc_dma_done_w, hpc_dma_done_w, video_int_w — none of these differ by
/// profile). `int2_map` (fullhouse's PIO4 layout) is
/// `local_status/mask<0,1>`, `map_status`, `map_mask<0,1>`,
/// `timer_int_clear`, and the PIT.
/// MAME's `ioc2_full_house_device::int2_map` (`ioc2.cpp:591-602`) has no
/// entry at compact offset 0x1C (dword index 7), but real IRIX's
/// `ip22_newportRetrace` (disassembly, confirmed live at
/// 0xffffffff882aa5f0) does a read-modify-write on phys 0x1FBD901F
/// (= HPC3_INT2_BASE index 7) on every vertical retrace. This is
/// `PORT_CONFIG` (`kern/sys/IP22.h`: `PORT_CONFIG = HPC3_INT2_ADDR +
/// IP22BOFF(0x1c)`; `IP22BOFF(x) = x|0x3` is the `_MIPSEB` byte-lane
/// adjustment for byte-wide registers, so the dword index is still
/// `0x1c>>2 = 7` — MAME's map for this one register is simply incomplete).
/// PORT_CONFIG is fullhouse-only — no equivalent on guinness, which has no
/// GIO-slot reset/retrace-clear register at all (see `pcon_regs`).
/// `Ioc::int2_read8`/`int2_write8` take a *register index* (0-based, one
/// per dword-aligned PBUS PIO slot — matching the stride hpc3.rs already
/// uses for every other PBUS PIO channel) so hpc3.rs doesn't need to
/// replicate IOC_BASE's address math for a second base address.
/// Register index = compact byte offset from MAME's `int2_map`, matching
/// the dword-per-register packing hpc3.rs uses for every PBUS PIO channel
/// (idx = byte_offset >> 2): 0=l0_stat 1=l0_mask 2=l1_stat 3=l1_mask
/// 4=map_stat 5=map_mask0 6=map_mask1 7=port_config 8=tmr_clr (9-11 unused)
/// 12-15=PIT channel
/// 0/1/2/control (same `Pit8254` instance guinness's INT3 timers use at
/// PIO6 — one chip, two address windows, per `ioc2_device`'s single
/// `m_pit` in MAME).
pub const INT2_REG_COUNT: u32 = 16;
const INT2_TMR_CLR_IDX: u32 = 8;
const INT2_PIT_BASE_IDX: u32 = 12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IocInterrupt {
    // Local 0 Sources
    Graphics,
    Parallel,
    McDma,
    Ethernet,
    Scsi1,
    Scsi0,
    FifoFull,

    // Local 1 Sources
    VerticalRetrace,
    VideoVsync,
    AcFail,
    HpcDma,
    Gp2,   // LOCAL1_N<2>, active low general purpose
    Panel,
    Gp0,   // LOCAL1_N<0>, active low general purpose

    // Mappable Sources (LIO_2 on IP24 / MAP on IP22)
    Serial,
    KbMouse,
    GioExp0,    // LIO_GIO_EXP0 = bit 6 — GIO expansion slot 0 (Indy IP24)
    GioExp1,    // LIO_GIO_EXP1 = bit 7 — GIO expansion slot 1
    Mappable0,  // Timer 0
    Mappable1,  // Timer 1
    Mappable2,
    Mappable3,

    /// IP22 fullhouse only: the 9 per-slot GIO interrupt sources (3 slots ×
    /// fifo/graphics/retrace). All 3 slots on real Indigo2 hardware share the
    /// same 3 physical IOC2 pins/bits (`LIO_FIFO`/`LIO_GIO_1`/`LIO_GIO_2` —
    /// same `L0_STAT`/`L1_STAT` bits `FifoFull`/`Graphics`/`VerticalRetrace`
    /// already use), so each of these also sets that shared bit. What makes
    /// them distinct is the 16-bit `HPC3_EXT_IO_ADDR` register (`ext_io_*` in
    /// `IocState`): each variant additionally clears its own active-low
    /// `EXTIO_*` bit there, which is what `ip22_gio0/1/2_intr`
    /// (`kern/ml/IP22.c`) reads to decide which slot's ISR to actually call.
    /// Naming matches IRIX's own SG (GIO_SLOT_GFX) / S0 (GIO_SLOT_0) / S1
    /// (GIO_SLOT_1) convention from `kern/sys/hpc3.h`'s `EXTIO_*` defines.
    GioSgFifo,      // EXTIO_SG_IRQ_2 = 0x0400, sets L0_STAT FIFO_FULL
    GioSgGraphics,  // EXTIO_SG_IRQ_1 = 0x0200, sets L0_STAT GRAPHICS
    GioSgRetrace,   // EXTIO_SG_RETRACE = 0x0100, sets L1_STAT VERTICAL_RETRACE
    GioS0Fifo,      // EXTIO_S0_IRQ_2 = 0x4000, sets L0_STAT FIFO_FULL
    GioS0Graphics,  // EXTIO_S0_IRQ_1 = 0x2000, sets L0_STAT GRAPHICS
    GioS0Retrace,   // EXTIO_S0_RETRACE = 0x1000, sets L1_STAT VERTICAL_RETRACE
    GioS1Fifo,      // EXTIO_S1_IRQ_2 = 0x0004, sets L0_STAT FIFO_FULL
    GioS1Graphics,  // EXTIO_S1_IRQ_1 = 0x0002, sets L0_STAT GRAPHICS
    GioS1Retrace,   // EXTIO_S1_RETRACE = 0x0001, sets L1_STAT VERTICAL_RETRACE
}

/// The 3 GIO64 bus slots, indexed as MAME's `gio64_slot_device::slot_type_t`
/// (`GIO64_SLOT_GFX`/`EXP0`/`EXP1` = 0/1/2 — see `$HOME/gits/mame/src/devices/bus/gio64/gio64.h`).
/// Physically present on both Indy and Indigo2 (`indy_indigo2.cpp` instantiates
/// all 3 slots for either profile) — Indy's PROM/kernel just never populates a
/// card in EXP1, and Newport is wired to GFX by convention on Indy but can sit
/// in any slot on Indigo2's physically larger case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GioSlot {
    Gfx = 0,
    Exp0 = 1,
    Exp1 = 2,
}

/// Physical base address of each GIO64 slot (`0x1F000000` GFX / `0x1F400000`
/// EXP0 / `0x1F600000` EXP1 — see physical.rs's `NEWPORT_BASE`/`GIO_SLOT0_BASE`/
/// `GIO_SLOT1_BASE`, and MAME's `indy_indigo2.cpp:21-23`), indexed by `GioSlot as usize`.
pub const GIO_SLOT_BASES: [u32; 3] = [0x1F00_0000, 0x1F40_0000, 0x1F60_0000];

/// Which `IocInterrupt` sources a card in a given GIO slot drives, for one
/// IOC2 profile. Newport only has 2 real interrupt pins — `FIFO_INT_N`
/// (`fifo_full`) and `VV_INT_N` (vertical retrace/Kaleidoscope) — there is no
/// separate "graphics" signal on Newport (that's XZ/Elan's GE11-done pin,
/// which never fires from a real Newport board).
#[derive(Debug, Clone, Copy)]
pub struct GioSlotWiring {
    pub retrace: IocInterrupt,
    pub fifo_full: IocInterrupt,
}

/// Per-profile slot → interrupt-source wiring, indexed `[profile_idx(guinness)][GioSlot as usize]`.
/// Row 0 = guinness (Indy), row 1 = fullhouse (Indigo2).
///
/// GFX slot: guinness dedicates 2 direct pins to it — `FifoFull` (L0_STAT)
/// and `VerticalRetrace` (L1_STAT bit 7) — matching MAME's `gio_int0/2_w`.
///
/// Fullhouse's `VECTOR_GIO2`/`ip22_gio2_intr` fan-out (`L1_STAT` bit 7 +
/// `HPC3_EXT_IO_ADDR` disambiguation, `kern/ml/IP22.c`) is real and
/// implemented (`GioSg/S0/S1Retrace`, `Ioc::read16`/`write16`'s `IOC_EXT_IO`
/// branch) but is **not what IRIX's fullhouse Newport driver actually uses**:
/// `ng1_init` never calls `setgiovector(GIO_INTERRUPT_2, ...)` for Newport at
/// all (confirmed via disassembly — only levels 0 and 1 are registered, both
/// pointing at `ip22_newportInterrupt`). That handler polls REX3's `STATUS`
/// register directly for `STATUS_VRINT` (bit 5) whenever it's entered via
/// GIO_INTERRUPT_0 (fifo) or GIO_INTERRUPT_1, and calls `ip22_newportRetrace`
/// from there — so vblank has to be delivered through the *fifo/graphics*
/// EXT_IO fan-out (`GioSg/S0/S1Graphics`, GIO_INTERRUPT_1), not the retrace
/// one, or `ip22_newportInterrupt` is never entered and VRINT is never
/// checked. Confirmed working live: routing the GFX slot's `retrace` field
/// through `GioSgGraphics` instead of `GioSgRetrace` unblocks the screensaver.
///
/// EXP0 slot: guinness's Exp0 is the Ultra64 (N64 dev board) → `GioExp0`
/// (see ultra64.rs) — a real MAP-bank interrupt, unrelated to GIO_INT_2.
/// Fullhouse's Exp0 (second Newport head) uses the `GioS0*` variants.
///
/// EXP1 slot: guinness's Exp1 has no physical card by convention on Indy;
/// `GioExp1` is reserved here for symmetry with fullhouse's second-head path.
/// Fullhouse's Exp1 has no IRIS device wired yet, but uses `GioS1*` for
/// consistency (matches `EXTIO_S1_*`, "original IP22 does not set
/// EXTIO_S1_IRQ_2"/etc. per IP22.c's comments, so real hardware treats it as
/// permanently absent — IRIS doesn't drive these bits from any callback).
pub const GIO_SLOT_MAP: [[GioSlotWiring; 3]; 2] = [
    // guinness (Indy)
    [
        GioSlotWiring { retrace: IocInterrupt::VerticalRetrace, fifo_full: IocInterrupt::FifoFull },
        GioSlotWiring { retrace: IocInterrupt::GioExp1, fifo_full: IocInterrupt::FifoFull },
        GioSlotWiring { retrace: IocInterrupt::GioExp1, fifo_full: IocInterrupt::FifoFull },
    ],
    // fullhouse (Indigo2) — retrace routed through the *graphics* EXT_IO fan-out
    // (GIO_INTERRUPT_1), not the retrace one — see this const's doc comment.
    [
        GioSlotWiring { retrace: IocInterrupt::GioSgGraphics, fifo_full: IocInterrupt::GioSgFifo },
        GioSlotWiring { retrace: IocInterrupt::GioS0Graphics, fifo_full: IocInterrupt::GioS0Fifo },
        GioSlotWiring { retrace: IocInterrupt::GioS1Graphics, fifo_full: IocInterrupt::GioS1Fifo },
    ],
];

/// Row index into `GIO_SLOT_MAP` for a given profile.
#[inline]
pub fn profile_idx(guinness: bool) -> usize {
    if guinness { 0 } else { 1 }
}

struct IocState {
    sys_id: u8,
    
    // INT3 Registers
    l0_stat: u8,
    l0_mask: u8,
    l1_stat: u8,
    l1_mask: u8,
    map_stat: u8,
    map_mask0: u8,
    map_mask1: u8,
    map_pol: u8,
    err_stat: u8,
    /// Fullhouse-only PORT_CONFIG register (`IP22.h`'s `PORT_CONFIG`, INT2
    /// compact index 7 / byte offset `0x1C`-aligned/`0x1F` BE-lane). Controls
    /// GIO slot reset (`PCON_SG_RESET_N`/`PCON_S0_RESET_N`) and per-slot
    /// retrace-clear strobes (`PCON_CLR_SG_RETRACE_N`/`PCON_CLR_S0_RETRACE_N`)
    /// on IP22. Not present on guinness (IP24), which has no equivalent
    /// register — retrace there is acked via IOC1 LOCAL1 status/mask only.
    port_config: u8,
    /// Fullhouse-only `HPC3_EXT_IO_ADDR` (`0x1FBD9900`, 16-bit register read
    /// as a 32-bit `uint` by IRIX — see `ext_io_regs`). All bits active-low;
    /// idle/no-interrupt state is all-1s. `EXTIO_S1_*` (3rd GIO slot) bits
    /// numerically collide with `EXTIO_SG_STAT_*` in this same word — real
    /// IRIX (`kern/sys/hpc3.h`: "IP22-006 splits EXTIO into two registers to
    /// support 3rd gio slot") implies a second physical register for that
    /// case, but no second address is defined anywhere in IP22.c/IP26.c/
    /// IP28.c, and it's gated behind a special medical-equipment board
    /// revision (`kern/ml/IP22.c`'s `SPECIAL_GIO_RESET` comment) that no
    /// config IRIS emulates has. IRIS therefore doesn't drive S1 bits from
    /// any callback; they stay permanently deasserted (matching stock
    /// hardware, where `ip22_gio0_intr` et al. explicitly comment "original
    /// IP22 does not set EXTIO_S1_IRQ_2").
    ext_io: u16,

    // Misc Registers
    gc_select: u8,
    gen_cntl: u8,
    panel: u8,
    read_reg: u8,
    dma_sel: u8,
    reset_reg: u8,
    write_reg: u8,
    /// Raw pointer to the CPU executor's `MipsCore.interrupts` word (an
    /// inline field, not `Arc<AtomicU64>` — see that field's doc comment in
    /// mips_core.rs). Set once via `Ioc::set_interrupts` after the executor
    /// is constructed; valid for the process lifetime from then on (the
    /// executor lives in a top-level `Arc<Mutex<...>>` that outlives every
    /// device, including this one).
    interrupts: Option<*const AtomicU64>,
}

// Safety: `interrupts` points into the CPU executor's MipsCore, which
// outlives this IocState (see the field's doc comment) — IocState is always
// accessed through Arc<Mutex<IocState>>, which is what actually needs Send.
unsafe impl Send for IocState {}

struct IocIrqLine {
    state: Arc<Mutex<IocState>>,
    source: IocInterrupt,
}

struct IocTimerCallback {
    state: Arc<Mutex<IocState>>,
    source: IocInterrupt,
}

impl TimerCallback for IocTimerCallback {
    fn callback(&self) {
        let mut state = self.state.lock();
        match self.source {
            IocInterrupt::Mappable0 => state.map_stat |= 1 << 0,
            IocInterrupt::Mappable1 => state.map_stat |= 1 << 1,
            _ => {}
        }
        state.update_interrupts();
    }
}

impl IrqCallback for IocIrqLine {
    fn set_level(&self, level: bool) {
        let mut state = self.state.lock();
        match self.source {
            IocInterrupt::Serial => if level { state.map_stat |= map_regs::SERIAL } else { state.map_stat &= !map_regs::SERIAL },
            IocInterrupt::KbMouse => if level { state.map_stat |= map_regs::KBD_MOUSE } else { state.map_stat &= !map_regs::KBD_MOUSE },
            _ => {} // Only Serial supported via this callback for now
        }
        state.update_interrupts();
    }
}

impl Ps2Callback for IocIrqLine {
    fn set_interrupt(&self, active: bool) {
        self.set_level(active);
    }
}

#[derive(Clone)]
pub struct Ioc {
    state: Arc<Mutex<IocState>>,
    scc: Z85c30,
    pit: Pit8254,
    ps2: Arc<Ps2Controller>,
    guinness: bool,
    /// Sender for async machine events (power-off).
    event_tx: Arc<std::sync::OnceLock<mpsc::SyncSender<MachineEvent>>>,
    /// Shared heartbeat — IOC sets/clears HB_LED_RED/GREEN bits directly.
    heartbeat: Arc<std::sync::OnceLock<Arc<AtomicU64>>>,
    /// Shared timer manager for PIT channels.
    timer_manager: Arc<std::sync::OnceLock<Arc<TimerManager>>>,
}

impl Ioc {
    pub fn new(guinness: bool) -> Self {
        Self::new_inner(guinness, false)
    }

    /// CI-mode constructor: skips TCP serial backend binding on SCC channels
    /// so multiple instances can run in parallel without port conflicts.
    /// Caller must install backends via `scc().set_backend_{a,b}` before the
    /// first `start()`.
    pub fn new_ci(guinness: bool) -> Self {
        Self::new_inner(guinness, true)
    }

    fn new_inner(guinness: bool, ci_mode: bool) -> Self {
        let sys_id = if guinness { 0x26 } else { 0x11 }; // primarily prom looks at bit 1 to detect full house.
        let state = Arc::new(Mutex::new(IocState {
            sys_id,
            l0_stat: 0,
            l0_mask: 0,
            l1_stat: 0,
            l1_mask: 0,
            map_stat: 0,
            map_mask0: 0,
            map_mask1: 0,
            map_pol: 0,
            err_stat: 0,
            port_config: 0,
            ext_io: ext_io_regs::IDLE,
            gc_select: 0,
            gen_cntl: 0,
            panel: 1, // Power State (Bit 0) = 1 (On)
            read_reg: 0x70, // Ethernet/SCSI Power Good (Bits 6,5,4 = 1)
            dma_sel: 0,
            reset_reg: 0,
            write_reg: 0,
            interrupts: None,
        }));

        let serial_irq = Arc::new(IocIrqLine {
            state: state.clone(),
            source: IocInterrupt::Serial,
        });

        let timer0_cb = Arc::new(IocTimerCallback {
            state: state.clone(),
            source: IocInterrupt::Mappable0,
        });

        let timer1_cb = Arc::new(IocTimerCallback {
            state: state.clone(),
            source: IocInterrupt::Mappable1,
        });

        let ps2_cb = Arc::new(IocIrqLine {
            state: state.clone(),
            source: IocInterrupt::KbMouse,
        });

        let scc = if ci_mode {
            Z85c30::new_null(Some(serial_irq))
        } else {
            Z85c30::new(Some(serial_irq))
        };

        Self {
            state,
            scc,
            pit: Pit8254::new(1_000_000, Some(timer0_cb), Some(timer1_cb), None),
            ps2: Arc::new(Ps2Controller::new(Some(ps2_cb))),
            guinness,
            event_tx: Arc::new(std::sync::OnceLock::new()),
            heartbeat: Arc::new(std::sync::OnceLock::new()),
            timer_manager: Arc::new(std::sync::OnceLock::new()),
        }
    }

    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.timer_manager.set(tm.clone());
        self.pit.set_timer_manager(tm);
    }

    pub fn set_event_sender(&self, tx: mpsc::SyncSender<MachineEvent>) {
        let _ = self.event_tx.set(tx);
    }

    pub fn set_heartbeat(&self, heartbeat: Arc<AtomicU64>) {
        let _ = self.heartbeat.set(heartbeat);
    }

    /// `interrupts` must point into a `MipsCore` that outlives this `Ioc`
    /// (see `IocState.interrupts`'s doc comment) — `MipsCpu::interrupts_ptr()`
    /// is the intended source.
    pub fn set_interrupts(&self, interrupts: *const AtomicU64) {
        self.state.lock().interrupts = Some(interrupts);
    }

    /// Bit assignments and IRQ source wiring are identical for guinness
    /// (Indy) and fullhouse (Indigo2) — confirmed against MAME's
    /// `ioc2_guinness_device`/`ioc2_full_house_device`, which both derive
    /// from the same `ioc2_device` base with the same `INT3_LOCAL0_*`/
    /// `INT3_LOCAL1_*` bits and the same `gio_int0/1/2_w`, `scsi0/1_int_w`,
    /// `enet_int_w`, `mc_dma_done_w`, `hpc_dma_done_w`, `video_int_w`
    /// handlers (see INT2_BASE's doc comment). Only the register
    /// *addresses* differ by profile (PIO4 int2_map vs PIO6 base_map's
    /// upper half), handled in hpc3.rs's PIO dispatch — not here.
    pub fn set_interrupt(&self, source: IocInterrupt, active: bool) {
        // Note: map_pol (polarity) register is currently ignored.
        // We assume active-high logic internally for now.
        match source {
            IocInterrupt::VerticalRetrace | IocInterrupt::VideoVsync => {},
            _ => dlog_dev!(LogModule::Ioc, "IOC: Set Interrupt {:?} = {}", source, active),
        }
        let mut state = self.state.lock();
        match source {
            // Local 0
            IocInterrupt::Graphics => if active { state.l0_stat |= l0_regs::GRAPHICS } else { state.l0_stat &= !l0_regs::GRAPHICS },
            IocInterrupt::Parallel => if active { state.l0_stat |= l0_regs::PARALLEL } else { state.l0_stat &= !l0_regs::PARALLEL },
            IocInterrupt::McDma => if active { state.l0_stat |= l0_regs::MC_DMA } else { state.l0_stat &= !l0_regs::MC_DMA },
            IocInterrupt::Ethernet => if active { state.l0_stat |= l0_regs::ETHERNET } else { state.l0_stat &= !l0_regs::ETHERNET },
            IocInterrupt::Scsi1 => if active { state.l0_stat |= l0_regs::SCSI1 } else { state.l0_stat &= !l0_regs::SCSI1 },
            IocInterrupt::Scsi0 => if active { state.l0_stat |= l0_regs::SCSI0 } else { state.l0_stat &= !l0_regs::SCSI0 },
            IocInterrupt::FifoFull => if active { state.l0_stat |= l0_regs::FIFO_FULL } else { state.l0_stat &= !l0_regs::FIFO_FULL },

            // Local 1
            IocInterrupt::VerticalRetrace => if active { state.l1_stat |= l1_regs::VERTICAL_RETRACE } else { state.l1_stat &= !l1_regs::VERTICAL_RETRACE },
            IocInterrupt::VideoVsync => if active { state.l1_stat |= l1_regs::VIDEO_VSYNC } else { state.l1_stat &= !l1_regs::VIDEO_VSYNC },
            IocInterrupt::AcFail => if active { state.l1_stat |= l1_regs::AC_FAIL } else { state.l1_stat &= !l1_regs::AC_FAIL },
            IocInterrupt::HpcDma => if active { state.l1_stat |= l1_regs::HPC_DMA } else { state.l1_stat &= !l1_regs::HPC_DMA },
            IocInterrupt::Gp2   => if active { state.l1_stat |= l1_regs::GP2   } else { state.l1_stat &= !l1_regs::GP2   },
            IocInterrupt::Panel => if active { state.l1_stat |= l1_regs::PANEL } else { state.l1_stat &= !l1_regs::PANEL },
            IocInterrupt::Gp0   => if active { state.l1_stat |= l1_regs::GP0   } else { state.l1_stat &= !l1_regs::GP0   },

            // Mappable (LIO_2 on IP24)
            IocInterrupt::Serial  => if active { state.map_stat |= map_regs::SERIAL    } else { state.map_stat &= !map_regs::SERIAL    },
            IocInterrupt::KbMouse => if active { state.map_stat |= map_regs::KBD_MOUSE } else { state.map_stat &= !map_regs::KBD_MOUSE },
            IocInterrupt::GioExp0 => if active { state.map_stat |= map_regs::GIO_EXP0 } else { state.map_stat &= !map_regs::GIO_EXP0 },
            IocInterrupt::GioExp1 => if active { state.map_stat |= map_regs::GIO_EXP1 } else { state.map_stat &= !map_regs::GIO_EXP1 },
            IocInterrupt::Mappable0 => if active { state.map_stat |= 1 << 0 } else { state.map_stat &= !(1 << 0) },
            IocInterrupt::Mappable1 => if active { state.map_stat |= 1 << 1 } else { state.map_stat &= !(1 << 1) },
            IocInterrupt::Mappable2 => if active { state.map_stat |= 1 << 2 } else { state.map_stat &= !(1 << 2) },
            IocInterrupt::Mappable3 => if active { state.map_stat |= 1 << 3 } else { state.map_stat &= !(1 << 3) },

            // Fullhouse GIO slot fan-out (see GIO_SLOT_MAP's doc comment):
            // each sets the one shared L0/L1 bit every slot's signal uses,
            // and clears (active) / sets (idle) its own active-low EXT_IO
            // bit so ip22_gio0/1/2_intr can tell which slot fired.
            IocInterrupt::GioSgFifo => {
                if active { state.l0_stat |= l0_regs::FIFO_FULL; state.ext_io &= !ext_io_regs::SG_IRQ_2; }
                else { state.l0_stat &= !l0_regs::FIFO_FULL; state.ext_io |= ext_io_regs::SG_IRQ_2; }
            }
            IocInterrupt::GioSgGraphics => {
                if active { state.l0_stat |= l0_regs::GRAPHICS; state.ext_io &= !ext_io_regs::SG_IRQ_1; }
                else { state.l0_stat &= !l0_regs::GRAPHICS; state.ext_io |= ext_io_regs::SG_IRQ_1; }
            }
            IocInterrupt::GioSgRetrace => {
                if active { state.l1_stat |= l1_regs::VERTICAL_RETRACE; state.ext_io &= !ext_io_regs::SG_RETRACE; }
                else { state.l1_stat &= !l1_regs::VERTICAL_RETRACE; state.ext_io |= ext_io_regs::SG_RETRACE; }
            }
            IocInterrupt::GioS0Fifo => {
                if active { state.l0_stat |= l0_regs::FIFO_FULL; state.ext_io &= !ext_io_regs::S0_IRQ_2; }
                else { state.l0_stat &= !l0_regs::FIFO_FULL; state.ext_io |= ext_io_regs::S0_IRQ_2; }
            }
            IocInterrupt::GioS0Graphics => {
                if active { state.l0_stat |= l0_regs::GRAPHICS; state.ext_io &= !ext_io_regs::S0_IRQ_1; }
                else { state.l0_stat &= !l0_regs::GRAPHICS; state.ext_io |= ext_io_regs::S0_IRQ_1; }
            }
            IocInterrupt::GioS0Retrace => {
                if active { state.l1_stat |= l1_regs::VERTICAL_RETRACE; state.ext_io &= !ext_io_regs::S0_RETRACE; }
                else { state.l1_stat &= !l1_regs::VERTICAL_RETRACE; state.ext_io |= ext_io_regs::S0_RETRACE; }
            }
            // GioS1* bits have no second EXT_IO register modeled (see
            // IocState.ext_io's doc comment) — still set the shared L0/L1
            // bit so callers routing through this slot don't silently drop
            // the interrupt, but the EXT_IO side is a no-op.
            IocInterrupt::GioS1Fifo => if active { state.l0_stat |= l0_regs::FIFO_FULL } else { state.l0_stat &= !l0_regs::FIFO_FULL },
            IocInterrupt::GioS1Graphics => if active { state.l0_stat |= l0_regs::GRAPHICS } else { state.l0_stat &= !l0_regs::GRAPHICS },
            IocInterrupt::GioS1Retrace => if active { state.l1_stat |= l1_regs::VERTICAL_RETRACE } else { state.l1_stat &= !l1_regs::VERTICAL_RETRACE },
        }
        state.update_interrupts();
    }

    pub fn ps2(&self) -> Arc<Ps2Controller> {
        self.ps2.clone()
    }

    pub fn scc(&self) -> &Z85c30 {
        &self.scc
    }

    pub fn pit(&self) -> &Pit8254 {
        &self.pit
    }

    pub fn register_locks(&self) {
        use crate::locks::register_lock_fn;
        let s = self.state.clone();
        register_lock_fn("ioc::state", move || s.is_locked());
        // SCC (Z85c30) channels
        self.scc.register_locks();
        // PS/2
        let ps2 = self.ps2.clone();
        register_lock_fn("ps2::state", move || ps2.is_state_locked());
    }
}

impl Device for Ioc {
    fn step(&self, _cycles: u64) {
        // TODO: Implement timer stepping
    }

    fn stop(&self) { self.scc.stop(); self.pit.stop(); self.ps2.stop(); }
    fn start(&self) {
        dlog_dev!(LogModule::Ioc, "IOC: start() called");
        self.scc.start();
        self.pit.start();
        self.ps2.start();
    }
    fn is_running(&self) -> bool { self.scc.is_running() }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        let mut cmds = vec![("ioc".to_string(), "IOC commands: ioc status".to_string())];
        cmds.extend(self.scc.register_commands());
        cmds.extend(self.pit.register_commands());
        cmds.extend(self.ps2.register_commands());
        cmds
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        if cmd == "ioc" {
            if args.first().copied() != Some("status") {
                return Err("Usage: ioc status".to_string());
            }
            let s = self.state.lock();
            fn bits8(v: u8, names: &[(u8, &str)]) -> String {
                let mut out = Vec::new();
                for (b, n) in names { if v & b != 0 { out.push(*n); } }
                if out.is_empty() { "-".into() } else { out.join("|") }
            }
            fn bits16(v: u16, names: &[(u16, &str)]) -> String {
                let mut out = Vec::new();
                for (b, n) in names { if v & b != 0 { out.push(*n); } }
                if out.is_empty() { "-".into() } else { out.join("|") }
            }
            let l0_names: &[(u8, &str)] = &[
                (l0_regs::MAP_INT0, "MAP_INT0"), (l0_regs::GRAPHICS, "GRAPHICS"),
                (l0_regs::PARALLEL, "PARALLEL"), (l0_regs::MC_DMA, "MC_DMA"),
                (l0_regs::ETHERNET, "ETHERNET"), (l0_regs::SCSI1, "SCSI1"),
                (l0_regs::SCSI0, "SCSI0"), (l0_regs::FIFO_FULL, "FIFO_FULL"),
            ];
            let l1_names: &[(u8, &str)] = &[
                (l1_regs::VERTICAL_RETRACE, "VRETR"), (l1_regs::VIDEO_VSYNC, "VSYNC"),
                (l1_regs::AC_FAIL, "ACFAIL"), (l1_regs::HPC_DMA, "HPC_DMA"),
                (l1_regs::MAP_INT1, "MAP_INT1"), (l1_regs::GP2, "GP2"),
                (l1_regs::PANEL, "PANEL"), (l1_regs::GP0, "GP0"),
            ];
            let map_names: &[(u8, &str)] = &[
                (map_regs::GIO_EXP1,  "GIO_EXP1"),
                (map_regs::GIO_EXP0,  "GIO_EXP0"),
                (map_regs::SERIAL,    "SERIAL"),
                (map_regs::KBD_MOUSE, "KBD_MOUSE"),
                (1 << 1, "TIMER1"), (1 << 0, "TIMER0"),
            ];
            // Named by which slot/signal is PENDING, i.e. the bit is CLEAR
            // (all EXT_IO bits are active-low — see ext_io_regs).
            let ext_io_pending_names: &[(u16, &str)] = &[
                (ext_io_regs::SG_RETRACE, "SG_RETRACE"), (ext_io_regs::SG_IRQ_1, "SG_GRAPHICS"), (ext_io_regs::SG_IRQ_2, "SG_FIFO"), (ext_io_regs::SG_IRQ_3, "SG_VSYNC"),
                (ext_io_regs::S0_RETRACE, "S0_RETRACE"), (ext_io_regs::S0_IRQ_1, "S0_GRAPHICS"), (ext_io_regs::S0_IRQ_2, "S0_FIFO"), (ext_io_regs::S0_IRQ_3, "S0_VSYNC"),
                (ext_io_regs::GIO_33MHZ, "GIO_33MHZ"), (ext_io_regs::EISA_BUSERR, "EISA_BUSERR"),
                (ext_io_regs::MC_BUSERR, "MC_BUSERR"), (ext_io_regs::HPC3_BUSERR, "HPC3_BUSERR"),
                (ext_io_regs::SG_STAT_0, "SG_STAT_0"), (ext_io_regs::SG_STAT_1, "SG_STAT_1"),
                (ext_io_regs::S0_STAT_0, "S0_STAT_0"), (ext_io_regs::S0_STAT_1, "S0_STAT_1"),
            ];
            let l0_eff = s.l0_stat & s.l0_mask;
            let l1_eff = s.l1_stat & s.l1_mask;
            let map_eff0 = s.map_stat & s.map_mask0;
            let map_eff1 = s.map_stat & s.map_mask1;
            let ip2 = l0_eff != 0;
            let ip3 = l1_eff != 0;
            let ip4 = (s.map_stat & 0x01) != 0;
            let ip5 = (s.map_stat & 0x02) != 0;
            let ip6 = s.err_stat != 0;
            let _ = writeln!(writer, "IOC INT3 state:");
            let _ = writeln!(writer, "  L0  stat={:02x} [{}]  mask={:02x}  eff={:02x} [{}]",
                s.l0_stat, bits8(s.l0_stat, l0_names), s.l0_mask, l0_eff, bits8(l0_eff, l0_names));
            let _ = writeln!(writer, "  L1  stat={:02x} [{}]  mask={:02x}  eff={:02x} [{}]",
                s.l1_stat, bits8(s.l1_stat, l1_names), s.l1_mask, l1_eff, bits8(l1_eff, l1_names));
            let _ = writeln!(writer, "  MAP stat={:02x} [{}]  mask0={:02x}  eff0={:02x} [{}]  mask1={:02x}  eff1={:02x} [{}]",
                s.map_stat, bits8(s.map_stat, map_names),
                s.map_mask0, map_eff0, bits8(map_eff0, map_names),
                s.map_mask1, map_eff1, bits8(map_eff1, map_names));
            let _ = writeln!(writer, "  MAP_POL={:02x}  ERR_STAT={:02x}  PORT_CONFIG={:02x}", s.map_pol, s.err_stat, s.port_config);
            let _ = writeln!(writer, "  EXT_IO={:04x}  pending(bit=0)=[{}]", s.ext_io, bits16(!s.ext_io, ext_io_pending_names));
            let _ = writeln!(writer, "  CPU IP lines: IP2={} IP3={} IP4=TMR0:{} IP5=TMR1:{} IP6=ERR:{}",
                ip2, ip3, ip4, ip5, ip6);
            let _ = writeln!(writer, "  Misc: sys_id={:02x} gc_select={:02x} gen_cntl={:02x} panel={:02x} read_reg={:02x} dma_sel={:02x} reset_reg={:02x} write_reg={:02x}",
                s.sys_id, s.gc_select, s.gen_cntl, s.panel, s.read_reg, s.dma_sel, s.reset_reg, s.write_reg);
            // Safety: see IocState.interrupts's doc comment.
            if let Some(ints) = s.interrupts.map(|p| unsafe { &*p }) {
                let raw = ints.load(Ordering::SeqCst);
                let _ = writeln!(writer, "  Atomic interrupts word: {:016x}  (IP2={} IP3={} IP4={} IP5={} IP6={} IP7=TMR:{})",
                    raw,
                    (raw & CAUSE_IP2 as u64) != 0,
                    (raw & CAUSE_IP3 as u64) != 0,
                    (raw & CAUSE_IP4 as u64) != 0,
                    (raw & CAUSE_IP5 as u64) != 0,
                    (raw & CAUSE_IP6 as u64) != 0,
                    (raw & (1u64 << 15)) != 0);
            } else {
                let _ = writeln!(writer, "  Atomic interrupts word: NOT WIRED (set_interrupts never called)");
            }
            return Ok(());
        }
        if cmd == "serial" {
            return self.scc.execute_command(cmd, args, writer);
        }
        if cmd == "pit" {
            return self.pit.execute_command(cmd, args, writer);
        }
        if cmd == "ps2" {
            return self.ps2.execute_command(cmd, args, writer);
        }
        Err("Command not found".to_string())
    }
}

impl BusDevice for Ioc {
    fn read8(&self, addr: u32) -> BusRead8 {
        // Fullhouse-only INT2 window (PBUS PIO channel 4) — lower address
        // than IOC_BASE (PIO channel 6), checked first so it isn't caught by
        // the addr-IOC_BASE subtraction below. See IOC_INT2_BASE's doc
        // comment.
        if (IOC_INT2_BASE..IOC_INT2_BASE + IOC_INT2_SIZE).contains(&addr) {
            return self.int2_read8(addr - IOC_INT2_BASE);
        }

        let offset = (addr - IOC_BASE) & !3;

        // Lock state only for IOC registers, not for SCC/PIT passthrough
        // This prevents deadlock when SCC callback tries to lock state

        // Serial ports (SCC) - direct 8-bit access
        if offset >= IOC_SERIAL1_CMD && offset <= IOC_SERIAL2_DATA {
            let idx = (offset - IOC_SERIAL1_CMD) >> 2;
            return self.scc.read(idx);
        }

        // Timers (PIT) - direct 8-bit access
        if offset >= IOC_TIMER_CNT0 && offset <= IOC_TIMER_CTL + 3 {
            let idx = (offset - IOC_TIMER_CNT0) >> 2;
            dlog_dev!(LogModule::Ioc, "IOC: Read PIT channel {} (offset {:02x})", idx, offset);
            return self.pit.read(idx);
        }

        // PS/2 Keyboard/Mouse - direct access to avoid lock inversion
        if offset == IOC_KBD_MOUSE_DATA {
            return BusRead8::ok(self.ps2.read_data());
        }
        if offset == IOC_KBD_MOUSE_CMD {
            return BusRead8::ok(self.ps2.read_status());
        }

        // IOC registers - all 8-bit
        let state = self.state.lock();

        let val = match offset {
            IOC_SYS_ID => state.sys_id,

            IOC_PL_DATA => 0,
            IOC_PL_CNTL => 0,
            IOC_PL_STAT => 0,

            IOC_INT3_L0_STAT  => { dlog_dev!(LogModule::Ioc, "IOC: rd L0_STAT  → {:#04x}", state.l0_stat);  state.l0_stat }
            IOC_INT3_L0_MASK  => { dlog_dev!(LogModule::Ioc, "IOC: rd L0_MASK  → {:#04x}", state.l0_mask);  state.l0_mask }
            IOC_INT3_L1_STAT  => { dlog_dev!(LogModule::Ioc, "IOC: rd L1_STAT  → {:#04x}", state.l1_stat);  state.l1_stat }
            IOC_INT3_L1_MASK  => { dlog_dev!(LogModule::Ioc, "IOC: rd L1_MASK  → {:#04x}", state.l1_mask);  state.l1_mask }
            IOC_INT3_MAP_STAT => { dlog_dev!(LogModule::Ioc, "IOC: rd MAP_STAT → {:#04x}", state.map_stat); state.map_stat }
            IOC_INT3_MAP_MASK0 => { dlog_dev!(LogModule::Ioc, "IOC: rd MAP_MASK0 → {:#04x}", state.map_mask0); state.map_mask0 }
            IOC_INT3_MAP_MASK1 => { dlog_dev!(LogModule::Ioc, "IOC: rd MAP_MASK1 → {:#04x}", state.map_mask1); state.map_mask1 }
            IOC_INT3_MAP_POL  => { dlog_dev!(LogModule::Ioc, "IOC: rd MAP_POL  → {:#04x}", state.map_pol);  state.map_pol }
            IOC_INT3_ERR_STAT => { dlog_dev!(LogModule::Ioc, "IOC: rd ERR_STAT → {:#04x}", state.err_stat); state.err_stat }

            IOC_GC_SELECT => state.gc_select,
            IOC_GEN_CNTL => state.gen_cntl,
            IOC_PANEL => state.panel,
            IOC_READ => state.read_reg,
            IOC_DMA_SEL => state.dma_sel,
            IOC_RESET => state.reset_reg,
            IOC_WRITE => state.write_reg,

            _ => {
                dlog_dev!(LogModule::Ioc, "IOC: Read8 offset {:02x}", offset);
                0
            }
        };
        BusRead8::ok(val)
    }

    fn write8(&self, addr: u32, val: u8) -> u32 {
        if (IOC_INT2_BASE..IOC_INT2_BASE + IOC_INT2_SIZE).contains(&addr) {
            return self.int2_write8(addr - IOC_INT2_BASE, val);
        }

        let offset = (addr - IOC_BASE) & !3;

        // Serial ports (SCC) - direct 8-bit access
        if offset >= IOC_SERIAL1_CMD && offset <= IOC_SERIAL2_DATA {
            let idx = (offset - IOC_SERIAL1_CMD) >> 2;
            return self.scc.write(idx, val);
        }

        // Timers (PIT) - direct 8-bit access
        if offset >= IOC_TIMER_CNT0 && offset <= IOC_TIMER_CTL + 3 {
            let idx = (offset - IOC_TIMER_CNT0) >> 2;
            dlog_dev!(LogModule::Ioc, "IOC: Write PIT channel {} (offset {:02x}) val {:02x}", idx, offset, val);
            return self.pit.write(idx, val);
        }

        // PS/2 Keyboard/Mouse - direct access to avoid lock inversion
        if offset == IOC_KBD_MOUSE_DATA {
            self.ps2.write_data(val);
            return BUS_OK;
        }
        if offset == IOC_KBD_MOUSE_CMD {
            self.ps2.write_command(val);
            return BUS_OK;
        }

        let mut state = self.state.lock();

        match offset {
            IOC_PL_DATA => { dlog_dev!(LogModule::Ioc, "IOC: Write PL_DATA val {:02x}", val); },
            IOC_PL_CNTL => { dlog_dev!(LogModule::Ioc, "IOC: Write PL_CNTL val {:02x}", val); },

            IOC_INT3_L0_MASK  => { dlog_dev!(LogModule::Ioc, "IOC: L0_MASK  = {:#04x}", val); state.l0_mask  = val; }
            IOC_INT3_L1_MASK  => { dlog_dev!(LogModule::Ioc, "IOC: L1_MASK  = {:#04x}", val); state.l1_mask  = val; }
            IOC_INT3_MAP_MASK0 => { dlog_dev!(LogModule::Ioc, "IOC: MAP_MASK0 = {:#04x}", val); state.map_mask0 = val; }
            IOC_INT3_MAP_MASK1 => { dlog_dev!(LogModule::Ioc, "IOC: MAP_MASK1 = {:#04x}", val); state.map_mask1 = val; }
            IOC_INT3_MAP_POL  => { dlog_dev!(LogModule::Ioc, "IOC: MAP_POL   = {:#04x}", val); state.map_pol  = val; }
            IOC_INT3_TMR_CLR => {
                dlog_dev!(LogModule::Ioc, "IOC: Timer Clear val {:02x}", val);
                state.map_stat &= !(val & 0x3);
            }

            IOC_GC_SELECT => state.gc_select = val,
            IOC_GEN_CNTL => state.gen_cntl = val,
            IOC_PANEL => {
                // Bits 6, 4, 1 are W1C (Write 1 to Clear)
                let mut current = state.panel;
                if (val & (1 << 6)) != 0 { current &= !(1 << 6); }
                if (val & (1 << 4)) != 0 { current &= !(1 << 4); }
                if (val & (1 << 1)) != 0 { current &= !(1 << 1); }
                // Bit 0 is RW (Power State, active low: 0 = off)
                let was_on = (current & 1) != 0;
                current = (current & !1) | (val & 1);
                state.panel = current;
                let now_off = (current & 1) == 0;
                if was_on && now_off {
                    if let Some(tx) = self.event_tx.get() {
                        dlog_dev!(LogModule::Ioc, "IOC: front panel power-off requested");
                        let _ = tx.try_send(MachineEvent::PowerOff);
                    }
                }
            }
            IOC_DMA_SEL => state.dma_sel = val,
            IOC_RESET => {
                use crate::rex3::Rex3;
                state.reset_reg = val;

                // LED bits are active-low: 0x10=LED_RED_OFF, 0x20=LED_GREEN_OFF
                // bit SET = LED off, bit CLEAR = LED on — update heartbeat unconditionally
                if let Some(hb) = self.heartbeat.get() {
                    if (val & 0x10) == 0 {
                        hb.fetch_or(Rex3::HB_LED_RED, Ordering::Relaxed);
                    } else {
                        hb.fetch_and(!Rex3::HB_LED_RED, Ordering::Relaxed);
                    }
                    if (val & 0x20) == 0 {
                        hb.fetch_or(Rex3::HB_LED_GREEN, Ordering::Relaxed);
                    } else {
                        hb.fetch_and(!Rex3::HB_LED_GREEN, Ordering::Relaxed);
                    }
                }
            },
            IOC_WRITE => state.write_reg = val,

            _ => {
                dlog_dev!(LogModule::Ioc, "IOC: Write8 offset {:02x} val {:02x}", offset, val);
            }
        }
        // Update interrupts after any write that might affect masks or status
        state.update_interrupts();
        BUS_OK
    }

    fn read16(&self, addr: u32) -> BusRead16 {
        let offset = (addr - IOC_BASE) & !1;
        if offset == IOC_EXT_IO {
            let state = self.state.lock();
            dlog_dev!(LogModule::Ioc, "IOC: rd EXT_IO → {:#06x}", state.ext_io);
            return BusRead16::ok(state.ext_io);
        }
        BusRead16::err()
    }

    fn write16(&self, addr: u32, val: u16) -> u32 {
        let offset = (addr - IOC_BASE) & !1;
        if offset == IOC_EXT_IO {
            let mut state = self.state.lock();
            dlog_dev!(LogModule::Ioc, "IOC: EXT_IO = {:#06x}", val);
            state.ext_io = val;
            return BUS_OK;
        }
        BUS_ERR
    }

    fn read32(&self, addr: u32) -> BusRead32 {
        //println!("IOC: Read32 addr {:08x}", addr);
        let aligned_addr = addr & !3;
        let offset = aligned_addr - IOC_BASE;
        // HPC3_EXT_IO_ADDR: IRIX reads this specific register as a 32-bit
        // `uint` despite the hardware register being 16 bits wide (comment
        // in kern/ml/IP22.c: "HPC3_EXT_IO_ADDR is 16 bits wide") — zero-
        // extend rather than truncate through the generic read8 byte path.
        if offset == IOC_EXT_IO {
            let state = self.state.lock();
            return BusRead32::ok(state.ext_io as u32);
        }
        // IOC registers are accessed as 32-bit words with data in low 8 bits
        // Address should be word-aligned
        let r = self.read8(aligned_addr);
        if r.is_ok() { BusRead32::ok(r.data as u32) } else { BusRead32 { status: r.status, data: 0 } }
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        //println!("IOC: Write32 addr {:08x} val {:08x}", addr, val);
        let aligned_addr = addr & !3;
        let offset = aligned_addr - IOC_BASE;
        if offset == IOC_EXT_IO {
            let mut state = self.state.lock();
            state.ext_io = (val & 0xFFFF) as u16;
            return BUS_OK;
        }
        // IOC registers are accessed as 32-bit words with data in low 8 bits
        // Address should be word-aligned
        // Extract low 8 bits (bits 7:0)
        let val8 = (val & 0xFF) as u8;
        self.write8(aligned_addr, val8)
    }
}

impl Ioc {
    /// Fullhouse-only: read one of the compact INT2 registers. `rel` is the
    /// address relative to `IOC_INT2_BASE` — called only from `read8`'s
    /// INT2-window branch, which is what hpc3.rs actually reaches through
    /// the normal `BusDevice` `read8`/`write8`/`read32`/`write32` methods
    /// (see `INT2_REG_COUNT`'s doc comment for the index layout).
    fn int2_read8(&self, rel: u32) -> BusRead8 {
        let idx = (rel >> 2) & 0xF;
        if idx >= INT2_PIT_BASE_IDX {
            return self.pit.read(idx - INT2_PIT_BASE_IDX);
        }
        let state = self.state.lock();
        let val = match idx {
            0 => { dlog_dev!(LogModule::Ioc, "INT2: rd L0_STAT  → {:#04x}", state.l0_stat);  state.l0_stat }
            1 => { dlog_dev!(LogModule::Ioc, "INT2: rd L0_MASK  → {:#04x}", state.l0_mask);  state.l0_mask }
            2 => { dlog_dev!(LogModule::Ioc, "INT2: rd L1_STAT  → {:#04x}", state.l1_stat);  state.l1_stat }
            3 => { dlog_dev!(LogModule::Ioc, "INT2: rd L1_MASK  → {:#04x}", state.l1_mask);  state.l1_mask }
            4 => { dlog_dev!(LogModule::Ioc, "INT2: rd MAP_STAT → {:#04x}", state.map_stat); state.map_stat }
            5 => { dlog_dev!(LogModule::Ioc, "INT2: rd MAP_MASK0 → {:#04x}", state.map_mask0); state.map_mask0 }
            6 => { dlog_dev!(LogModule::Ioc, "INT2: rd MAP_MASK1 → {:#04x}", state.map_mask1); state.map_mask1 }
            7 => { dlog_dev!(LogModule::Ioc, "INT2: rd PORT_CONFIG → {:#04x}", state.port_config); state.port_config }
            _ => 0, // 9-11: unused; 8 (tmr_clr) is write-only
        };
        BusRead8::ok(val)
    }

    /// Fullhouse-only: write one of the compact INT2 registers. `rel` is
    /// relative to `IOC_INT2_BASE` — see `int2_read8`.
    fn int2_write8(&self, rel: u32, val: u8) -> u32 {
        let idx = (rel >> 2) & 0xF;
        if idx >= INT2_PIT_BASE_IDX {
            return self.pit.write(idx - INT2_PIT_BASE_IDX, val);
        }
        let mut state = self.state.lock();
        match idx {
            1 => { dlog_dev!(LogModule::Ioc, "INT2: L0_MASK  = {:#04x}", val); state.l0_mask = val; }
            3 => { dlog_dev!(LogModule::Ioc, "INT2: L1_MASK  = {:#04x}", val); state.l1_mask = val; }
            5 => { dlog_dev!(LogModule::Ioc, "INT2: MAP_MASK0 = {:#04x}", val); state.map_mask0 = val; }
            6 => { dlog_dev!(LogModule::Ioc, "INT2: MAP_MASK1 = {:#04x}", val); state.map_mask1 = val; }
            7 => {
                dlog_dev!(LogModule::Ioc, "INT2: PORT_CONFIG = {:#04x}", val);
                // CLR_SG_RETRACE_N / CLR_S0_RETRACE_N are active-low strobes: a
                // 0 bit clears the shared LIO_GIO_2/VERTICAL_RETRACE latch (one
                // physical retrace line fanned out to both GFX and EXP0 slots
                // via ip22_gio2_intr's EXTIO dispatch — see IP22.h's PCON_*).
                if val & (pcon_regs::CLR_SG_RETRACE_N | pcon_regs::CLR_S0_RETRACE_N)
                    != (pcon_regs::CLR_SG_RETRACE_N | pcon_regs::CLR_S0_RETRACE_N) {
                    state.l1_stat &= !l1_regs::VERTICAL_RETRACE;
                }
                state.port_config = val;
            }
            idx if idx == INT2_TMR_CLR_IDX => {
                dlog_dev!(LogModule::Ioc, "INT2: Timer Clear val {:02x}", val);
                state.map_stat &= !(val & 0x3);
            }
            _ => dlog_dev!(LogModule::Ioc, "INT2: Write8 idx {} val {:02x} (RO or unmapped)", idx, val),
        }
        state.update_interrupts();
        BUS_OK
    }

}

impl IocState {
    fn update_interrupts(&mut self) {
        // 1. Update Mappable Interrupts (MAP_INT0, MAP_INT1)
        let map_int0 = (self.map_stat & self.map_mask0) != 0;
        let map_int1 = (self.map_stat & self.map_mask1) != 0;

        // Update Local 0 Status Bit 7
        if map_int0 {
            self.l0_stat |= l0_regs::MAP_INT0;
        } else {
            self.l0_stat &= !l0_regs::MAP_INT0;
        }

        // Update Local 1 Status Bit 3
        if map_int1 {
            self.l1_stat |= l1_regs::MAP_INT1;
        } else {
            self.l1_stat &= !l1_regs::MAP_INT1;
        }

        // 2. Calculate CPU Interrupts
        // Local 0 -> IP2 (MIPS Int 0)
        let ip2 = (self.l0_stat & self.l0_mask) != 0;
        
        // Local 1 -> IP3 (MIPS Int 1)
        let ip3 = (self.l1_stat & self.l1_mask) != 0;
        // Level 2: Timer 0 -> IP4
        // Timer 0 is bit 0 of map_stat (latched)
        let ip4 = (self.map_stat & 0x01) != 0;

        // Level 3: Timer 1 -> IP5
        // Timer 1 is bit 1 of map_stat (latched)
        let ip5 = (self.map_stat & 0x02) != 0;

        // Level 4: Bus Error -> IP6
        let ip6 = self.err_stat != 0;

        // 3. Signal CPU
        // Safety: interrupts points into the CPU executor's MipsCore, which
        // outlives this Ioc (see IocState.interrupts's doc comment).
        if let Some(interrupts) = self.interrupts.map(|p| unsafe { &*p }) {
            let mut set_mask = 0;
            let mut clear_mask = 0;

            if ip2 {
                set_mask |= CAUSE_IP2 as u64;
            } else {
                clear_mask |= CAUSE_IP2 as u64;
            }
            if ip3 {
                set_mask |= CAUSE_IP3 as u64;
            } else {
                clear_mask |= CAUSE_IP3 as u64;
            }
            if ip4 { set_mask |= CAUSE_IP4 as u64; } else { clear_mask |= CAUSE_IP4 as u64; }
            if ip5 { set_mask |= CAUSE_IP5 as u64; } else { clear_mask |= CAUSE_IP5 as u64; }
            if ip6 { set_mask |= CAUSE_IP6 as u64; } else { clear_mask |= CAUSE_IP6 as u64; }

            if set_mask != 0 {
                interrupts.fetch_or(set_mask, Ordering::SeqCst);
            }
            if clear_mask != 0 {
                interrupts.fetch_and(!clear_mask, Ordering::SeqCst);
            }
        }
    }
}
// ============================================================================
// Resettable + Saveable for Ioc
// ============================================================================

impl Resettable for Ioc {
    fn power_on(&self) {
        let mut state = self.state.lock();
        state.l0_stat = 0;
        state.l0_mask = 0;
        state.l1_stat = 0;
        state.l1_mask = 0;
        state.map_stat = 0;
        state.map_mask0 = 0;
        state.map_mask1 = 0;
        state.map_pol = 0;
        state.err_stat = 0;
        state.port_config = 0;
        state.ext_io = ext_io_regs::IDLE; // active-low: all-1s = nothing pending
        state.gc_select = 0;
        state.gen_cntl = 0;
        state.panel = 1;        // power-on: power state bit = 1
        state.read_reg = 0x70;  // ethernet/SCSI power good
        state.dma_sel = 0;
        state.reset_reg = 0;
        state.write_reg = 0;
        // Clear CPU interrupt lines
        // Safety: see IocState.interrupts's doc comment.
        if let Some(irqs) = state.interrupts.map(|p| unsafe { &*p }) {
            use std::sync::atomic::Ordering;
            irqs.store(0, Ordering::SeqCst);
        }
    }
}

impl Saveable for Ioc {
    fn save_state(&self) -> toml::Value {
        let state = self.state.lock();
        let mut tbl = toml::map::Map::new();
        macro_rules! u8f { ($f:ident) => { tbl.insert(stringify!($f).into(), hex_u8(state.$f)); } }
        u8f!(l0_stat); u8f!(l0_mask); u8f!(l1_stat); u8f!(l1_mask);
        u8f!(map_stat); u8f!(map_mask0); u8f!(map_mask1); u8f!(map_pol); u8f!(err_stat);
        u8f!(port_config);
        u8f!(gc_select);
        u8f!(gen_cntl); u8f!(panel); u8f!(read_reg);
        u8f!(dma_sel); u8f!(reset_reg); u8f!(write_reg);
        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut state = self.state.lock();
        macro_rules! ldu8 { ($f:ident) => {
            if let Some(x) = get_field(v, stringify!($f)) { state.$f = toml_u8(x).unwrap_or(state.$f); }
        }}
        ldu8!(l0_stat); ldu8!(l0_mask); ldu8!(l1_stat); ldu8!(l1_mask);
        ldu8!(map_stat); ldu8!(map_mask0); ldu8!(map_mask1); ldu8!(map_pol); ldu8!(err_stat);
        ldu8!(port_config);
        ldu8!(gc_select);
        ldu8!(gen_cntl); ldu8!(panel); ldu8!(read_reg);
        ldu8!(dma_sel); ldu8!(reset_reg); ldu8!(write_reg);
        state.update_interrupts();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 1.7 round-trip: a fresh IOC loaded from a captured save_state must
    /// re-serialize byte-identically. Catches load_state forgetting any of the
    /// 16 register fields that save_state writes.
    #[test]
    fn save_load_round_trip() {
        // new_ci uses null serial backends — avoids TCP port binding under
        // concurrent test runs.
        let src = Ioc::new_ci(true);
        {
            let mut s = src.state.lock();
            s.l0_stat   = 0x12; s.l0_mask  = 0x34;
            s.l1_stat   = 0x56; s.l1_mask  = 0x78;
            s.map_stat  = 0x9a; s.map_mask0 = 0xbc; s.map_mask1 = 0xde;
            s.map_pol   = 0xf0; s.err_stat = 0x01;
            s.gc_select = 0x0f; s.gen_cntl = 0xa5; s.panel = 0x5a;
            s.read_reg  = 0xff; s.dma_sel  = 0x33;
            s.reset_reg = 0x77; s.write_reg = 0xee;
            // load_state re-runs update_interrupts, so the saved snapshot must
            // already reflect the cascade-derived bits (MAP_INT0/MAP_INT1) for
            // v1 to round-trip cleanly. In a real save these are always
            // up-to-date because the bus driver runs update_interrupts on
            // every register write.
            s.update_interrupts();
        }
        let v1 = src.save_state();

        let dst = Ioc::new_ci(true);
        dst.load_state(&v1).expect("load_state");
        let v2 = dst.save_state();

        assert_eq!(v1, v2, "Ioc save_state mismatch after load_state round-trip");
    }
}
